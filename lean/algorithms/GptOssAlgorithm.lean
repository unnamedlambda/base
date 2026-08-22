import Lean
import Std
import AlgorithmLib.Gen
import AlgorithmLib.ML
import GptOssKernels
import GptOssAttention
import LayoutScan
import ShipScan

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML AlgorithmLib.Host
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-!
  # gpt-oss-20b: one layer's mixture, dispatched by rebinding

  The first slice of the engine that runs on a device: one layer's routed
  experts, driven from the host, against the real converted checkpoint.

  ## What this establishes

  The expert pool does not fit in this card's memory — 9.48 GiB of experts
  against 12 GiB total, before attention, the KV cache or `lm_head`. The
  architecture that makes the model servable anyway is a *cache*: a fixed set
  of slots holds whichever experts the router has been asking for, and a slot
  is pointed at an expert by rewriting a buffer id, not by moving weights.

  This module ships the rebinding half of that, with every expert resident so
  that the rebinding can be checked on its own. `gBindExperts` is six integer
  moves per slot: the chosen expert's ids are copied into the entries the
  slot's kernels read. No kernel is re-emitted and no weight is copied, which
  is the property `gptoss_slots_are_bound` states — the expert half of the
  launch sequence names `GSLOT0…` and never an address in the store.

  What is *not* here yet is the cache: which expert occupies which slot is
  still decided by the caller. Turning that into an LRU over a host-pinned pool
  is the next step, and it changes the host program, not the kernels.

  ## Why the experts are resident here and will not be

  Thirty-two experts of one layer is 404 MiB, which fits. Seven hundred and
  sixty-eight of them do not, and the whole point of the design is that they
  need not. Keeping them resident *for this slice* separates two failures that
  would otherwise arrive together: a wrong dispatch and a wrong transfer.
-/

namespace GptOssAlgorithm

open GptOssKernels

/-! ## Geometry -/

/-- Experts in a layer, and how many a token routes to. -/
def NE : Nat := 32
def TOPK : Nat := 4

/-- Pieces a slot binds: packed codes, scales and bias, for each projection. -/
def PIECES : Nat := 6

/-- Bytes of each piece, in the order the converter writes them. -/
def pieceBytes : List Nat :=
  [ 2 * I * rowBytesH, 2 * I * rowScalesH, 2 * I * 4
  , H * rowBytesI, H * rowScalesI, H * 4 ]

/-- One expert's row: the unit a cache miss will transfer. -/
def ROW_BYTES : Nat := pieceBytes.foldl (· + ·) 0

/-! ## Buffers -/

def B_X : Nat := 0        -- the layer's normalised hidden state
def B_GATES : Nat := 1    -- the four gates, in a warp-wide buffer
def B_OUT : Nat := 2      -- the mixture
def B_Y : Nat := 3        -- four slot outputs
def B_HID : Nat := 7      -- four slot intermediates
def GSLOT0 : Nat := 11    -- four slots of six pieces
def GSTORE : Nat := GSLOT0 + PIECES * TOPK
def GNBUF : Nat := GSTORE + PIECES * NE

/-- Slot `j`'s piece `t`, and expert `e`'s piece `t`. -/
def slotBuf (j t : Nat) : Nat := GSLOT0 + PIECES * j + t
def storeBuf (e t : Nat) : Nat := GSTORE + PIECES * e + t

/-- Bytes per buffer.  The slot buffers are placeholders: the bind array points
    them at the store before the experts run, so what is allocated for them is
    never read. -/
def gBufBytes : List Nat :=
  [ H * 4, 32 * 4, H * 4 ]
    ++ List.replicate TOPK (H * 4)
    ++ List.replicate TOPK (I * 4)
    ++ List.replicate (PIECES * TOPK) 128
    ++ (List.range NE).flatMap (fun _ => pieceBytes)

theorem gptoss_alloc_covers :
    gBufBytes.length = GNBUF ∧ gBufBytes.all (fun n => decide (0 < n)) = true := by
  native_decide

/-! ## Memory map -/

def GPTX_OFF : Nat := 0x0100
/-- Bytes a PTX slot gets, matching `GptOssDecode.DSLOT`.

    The expert kernels are emitted straight-line over a generation-time trip
    count, so their text grows with the rows a warp takes: `down` at four rows
    is some eighty kilobytes of it. The slot is sized for the kernel rather
    than the kernel trimmed to the slot -- four slots at this stride cost under
    half a megabyte of artifact, which is nothing beside what the shape is
    worth on the one path a token is spent in. -/
def GSLOT : Nat := 0x20000
def gSlotOff (i : Nat) : Nat := GPTX_OFF + i * GSLOT
def GBIND_OFF : Nat := gSlotOff gptossPtx.length
def gBindOff (i : Nat) : Nat := GBIND_OFF + 4 * i
def GLOCAL_OFF : Nat := GBIND_OFF + 4 * GNBUF
def GMEM_SIZE : Nat := GLOCAL_OFF + 4 * 8 + 0x100
def GHOST_LEN_OFF : Nat := 0x0080

theorem gptoss_ptx_fits :
    gptossPtx.all (fun t => decide (t.utf8ByteSize + 1 ≤ GSLOT)) = true := by
  native_decide

/-- The host input region: the activation, the gates, then the whole store. -/
def gHostIn : AlgorithmLib.Layout.RegionMap :=
  let sizes := [H * 4, 32 * 4] ++ (List.range NE).flatMap (fun _ => pieceBytes)
  (List.range sizes.length).map (fun i =>
    ⟨s!"in{i}", (sizes.take i).foldl (· + ·) 0, sizes.getD i 0⟩)

theorem gptossHostIn_packed :
    AlgorithmLib.Layout.RegionMap.packedB 0 gHostIn = true := by native_decide

def GHOST_BYTES : Nat := AlgorithmLib.Layout.RegionMap.total gHostIn

def gMemMap : AlgorithmLib.Layout.RegionMap :=
  (List.range gptossPtx.length).map (fun i => ⟨s!"ptx{i}", gSlotOff i, GSLOT⟩)
    ++ [⟨"hostLen", GHOST_LEN_OFF, 4⟩,
        ⟨"bind", GBIND_OFF, 4 * GNBUF⟩, ⟨"local", GLOCAL_OFF, 4 * 8⟩]

theorem gptossMap_ok :
    gMemMap.okB = true ∧ gMemMap.withinB GMEM_SIZE = true := by native_decide

/-! ## The host program -/

def env : FnEnv := env% [.cuda, .cublas]

/-- Allocate every buffer and upload the shared inputs and the whole store. -/
def gLoadFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit ptr
  let ctxPtr ← cudaCtxPtr ptr
  for (i, nb) in (List.range GNBUF).zip gBufBytes do
    let sz ← iconst64 nb
    let id ← cudaCreateBuffer ptr sz
    storeI32 id (← absAddr ptr (gBindOff i))
  for i in List.range 2 do
    let src ← iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt gHostIn i)
    let id ← load32 (← absAddr ptr (gBindOff i))
    let bytes ← iconst64 (gBufBytes.getD i 0)
    let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, src, bytes]
  for k in List.range (PIECES * NE) do
    let src ← iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt gHostIn (2 + k))
    let id ← load32 (← absAddr ptr (gBindOff (GSTORE + k)))
    let bytes ← iconst64 (pieceBytes.getD (k % PIECES) 0)
    let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, src, bytes]

/-- **Choosing experts, as the host does it.**

    One `u32` per slot arrives in the input region.  For slot `j` holding
    expert `e`, the six ids at `GSTORE + 6e` are copied into the six entries
    slot `j`'s kernels read.  The expert index is *loaded*, so the address is
    computed at run time and nothing about which expert is chosen is baked into
    the image — which is what makes this a cache lookup rather than a
    recompilation. -/
def gBindExperts : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let four ← iconst64 4
  let six ← iconst64 PIECES
  for j in List.range TOPK do
    let e ← load32 (← iaddImm dataPtr (4 * j))
    let e64 ← uextend64 e
    let e6 ← imul e64 six
    for t in List.range PIECES do
      let srcIx ← iaddImm e6 (GSTORE + t)
      let srcOff ← imul srcIx four
      let srcAddr ← iadd (← absAddr ptr GBIND_OFF) srcOff
      let id ← load32 srcAddr
      storeI32 id (← absAddr ptr (gBindOff (slotBuf j t)))

/-- Copy the ids one launch names into its own table, so the entries
    `gBindExperts` rewrote are what the kernel reads. -/
def gBindLocal (ptr : R) (bs : List Nat) : M Unit := do
  for (j, gb) in (List.range bs.length).zip bs do
    let id ← load32 (← absAddr ptr (gBindOff gb))
    storeI32 id (← absAddr ptr (GLOCAL_OFF + 4 * j))

/-- Enqueue the kernel in slot `i` over `g` blocks of `blk` threads. -/
def gEnqueue (ptr : R) (i g blk : Nat) (bs : List Nat) : M Unit := do
  gBindLocal ptr bs
  let ptxOff ← iconst64 (gSlotOff i)
  let nBufs ← iconst32 bs.length
  let bindBase ← iconst64 GLOCAL_OFF
  let one ← iconst32 1
  let block ← iconst32 blk
  let grid ← iconst32 g
  let _ ← cudaLaunch ptr ptxOff nBufs bindBase grid one one block one one

def BLK : Nat := 32 * warpsPerCta
def GRID_I : Nat := I / warpsPerCta
def GRID_H : Nat := H / warpsPerCta
def GRID_COMB : Nat := (H + BLK - 1) / BLK

/-- **The mixture: nine launches** — slot, grid, and the buffers each names.

    Two per slot — gate/up with its activation fused, then the down projection
    — and one to sum the four against their gates. The count does not grow with
    the thirty-two experts, only with the four that were chosen, which is the
    whole reason a mixture is cheap to serve.

    Written out as data rather than woven into the emission so that *what the
    sequence names* is something a theorem can quantify over instead of
    something a reader has to trace through a loop. `gRunExperts` is built from
    this list and nothing else, which is what makes the guard below
    load-bearing. -/
def gExpertBinds : List (Nat × Nat × List Nat) :=
  (List.range TOPK).flatMap (fun j =>
    [ (0, GRID_I, [slotBuf j 0, slotBuf j 1, slotBuf j 2, B_X, B_HID + j])
    , (1, GRID_H, [slotBuf j 3, slotBuf j 4, slotBuf j 5, B_HID + j, B_Y + j]) ])
  ++ [ (2, GRID_COMB, [B_Y, B_Y + 1, B_Y + 2, B_Y + 3, B_GATES, B_OUT]) ]

def gRunExperts : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  for (i, g, bs) in gExpertBinds do
    gEnqueue ptr i g BLK bs
  let _ ← cudaSync ptr

/-- **Seam guard: the mixture never names the store.**

    Every buffer the nine launches bind is below `GSTORE` — a shared buffer or
    one of the four slots — so the only way an expert's weights reach a kernel
    is through an id `gBindExperts` copied into a slot entry. That is what makes
    the dispatch a *lookup*: were a store buffer named directly anywhere in the
    sequence, some expert would be reachable without being chosen, and the
    launch count would grow with the pool rather than with the top-k.

    It is also what will make the cache possible. A slot whose id can be
    rewritten can be pointed at a freshly streamed expert; a store buffer named
    in the image cannot. -/
theorem gptoss_slots_are_bound :
    (gExpertBinds.flatMap (fun t => t.2.2)).all (fun b => decide (b < GSTORE))
      = true := by native_decide

/-- …and the slots really do stop where the store starts, so "below `GSTORE`"
    is the statement it looks like. -/
theorem gptoss_store_is_unnamed : GSLOT0 + PIECES * TOPK = GSTORE := by decide

/-- **Seam guard: every buffer a launch names was allocated.** -/
theorem gptoss_binds_allocated :
    (gExpertBinds.flatMap (fun t => t.2.2)).all (fun b => decide (b < GNBUF))
      = true := by native_decide

def gUploadFn (b n : Nat) : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let id ← load32 (← absAddr ptr (gBindOff b))
  let bytes ← iconst64 n
  let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, dataPtr, bytes]

def gFetchFn (b n : Nat) : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let id ← load32 (← absAddr ptr (gBindOff b))
  let bytes ← iconst64 n
  let _ ← call IR.Ffi.cudaDownload.id [ctxPtr, id, outPtr, bytes]

def gShippedBodies : List HProg.Code :=
  [ gLoadFn, gBindExperts, gUploadFn B_X (H * 4), gUploadFn B_GATES (32 * 4)
  , gRunExperts, gFetchFn B_OUT (H * 4) ]

theorem gptossShipped_wf :
    gShippedBodies.all (HProg.wf env HProg.ptrParams) = true := by
  native_decide

def gClifIR : Program :=
  program <|
    noopFunction :: gShippedBodies.attach.zipIdx.map
      (fun p =>
        HProg.compileFn (p.2 + 1) p.1.1 env
          (hwf := List.all_eq_true.mp gptossShipped_wf p.1.1 p.1.2))

/-- A `Nat` as four little-endian bytes. -/
def u32le (v : Nat) : List UInt8 :=
  [ UInt8.ofNat (v % 256), UInt8.ofNat (v / 256 % 256)
  , UInt8.ofNat (v / 65536 % 256), UInt8.ofNat (v / 16777216 % 256) ]

def gSlotBytes (t : String) : List UInt8 :=
  let b := t.toUTF8.toList ++ [0]
  b ++ zeros (GSLOT - b.length)

def gInitialMemory : List UInt8 :=
  zeros GHOST_LEN_OFF ++ u32le GHOST_BYTES
    ++ zeros (GPTX_OFF - GHOST_LEN_OFF - 4)
    ++ gptossPtx.flatMap gSlotBytes
    ++ zeros (GMEM_SIZE - GBIND_OFF)

def gSetup : Setup := {
  clif := gClifIR
  memory_size := GMEM_SIZE
  initial_memory := gInitialMemory
}


/-! # One layer's attention

  The other half of the slice, and the half that reuses rather than invents:
  every kernel here is an `EWStmt` carrying the same soundness theorem the
  Qwen2 decode kernels carry (`GptOssAttention`), and the two contractions are
  cuBLAS calls the stack already makes. What is new is arrangement.

  ## What the host program has to know

  A decode step's shapes depend on the position: the contraction runs over
  `min(pos+1, cap)` keys. cuBLAS dimensions are host-side arguments, so the
  host program reads them out of the *input region* — the same integers it
  uploads to the device meta buffer for the kernels. One publication, two
  readers, and no way for the two to disagree about how long the row is.

  ## The ring

  Layer 0 slides: it attends to the last 128 positions. The cache is therefore
  128 entries and position `p` occupies `p mod 128`, so the window is always
  the entire cache and no launch needs an offset or a mask. Softmax and the
  value mix are symmetric in the keys, so the order the ring leaves them in is
  not observable — which is the argument, and it is on the ledger as
  `SwaRingIsTheWindow` until it is written down.
-/

namespace Attn

-- `H` is the hidden size in both, and the same 2880; opening both would make the
-- name ambiguous rather than agree, so the expert half's is the one in scope.
open GptOssAttention hiding H

/-- Layer 0 slides, so this slice runs at the window depth. A full-attention
    layer is the same program with `CAP_FULL` and a different emitted
    `kvStore`; M4 ships both. -/
def CAP : Nat := CAP_SWA

/-! ## PTX slots -/

def aPtx : List String :=
  [ ptxRmsNorm, ptxAdd, emitProvenKernelN "main" 3 0 ropeQEW
  , emitProvenKernelN "main" 3 0 ropeKEW
  , ptxKVStore QO CAP, ptxKVStore KO CAP
  , ptxSinkSoftmax, GptOssKernels.moduleFor GptOssKernels.narrowBf16 ]

def S_RMS : Nat := 0
def S_ADD : Nat := 1
def S_ROPEQ : Nat := 2
def S_ROPEK : Nat := 3
def S_KVK : Nat := 4
def S_KVV : Nat := 5
def S_SOFTMAX : Nat := 6
def S_NARROW : Nat := 7

/-! ## Buffers -/

def A_X : Nat := 0        -- the residual stream
def A_XN : Nat := 1       -- normalised, f32
def A_XNB : Nat := 2      -- normalised, bf16, for the projection
def A_QKVW : Nat := 3
def A_QKVB : Nat := 4
def A_QKV : Nat := 5      -- the packed row: Q, then K, then V
def A_KC : Nat := 6       -- the key ring
def A_VC : Nat := 7       -- the value ring
def A_SC : Nat := 8       -- scores
def A_PR : Nat := 9       -- probabilities
def A_ATT : Nat := 10     -- the mixed values
def A_ATTB : Nat := 11
def A_OW : Nat := 12
def A_OB : Nat := 13
def A_ANORM : Nat := 14
def A_SINKS : Nat := 15
def A_ROPE : Nat := 16    -- sines, then cosines
def A_META : Nat := 17
def A_TMP : Nat := 18     -- the output projection, before it joins the stream
def A_NMH : Nat := 19     -- "narrow H/2 pairs"
def A_NMQ : Nat := 20     -- "narrow QO/2 pairs"
def ANBUF : Nat := 21

def aBufBytes : List Nat :=
  [ H * 4, H * 4, H * 2, QKV * H * 2, QKV * 4, QKV * 4
  , NKV * CAP * HD * 4, NKV * CAP * HD * 4
  , NQ * CAP * 4, NQ * CAP * 4
  , QO * 4, QO * 2, H * QO * 2, H * 4, H * 4, NQ * 4
  , 2 * ROPE_N * HALF * 4, 64 * 4, H * 4, 4, 4 ]

theorem gptoss_attn_alloc_covers :
    aBufBytes.length = ANBUF ∧ aBufBytes.all (fun n => decide (0 < n)) = true := by
  native_decide

/-! ## Memory map -/

/-- Wider slots than the mixture's: RMSNorm over 2880 elements unrolls ninety
    strided steps twice over, and the emitted text is what that costs. -/
def ASLOT : Nat := 0x20000
def APTX_OFF : Nat := 0x0100
def aSlotOff (i : Nat) : Nat := APTX_OFF + i * ASLOT
def ABIND_OFF : Nat := aSlotOff aPtx.length
def aBindOff (i : Nat) : Nat := ABIND_OFF + 4 * i
def ALOCAL_OFF : Nat := ABIND_OFF + 4 * ANBUF
def AMEM_SIZE : Nat := ALOCAL_OFF + 4 * 8 + 0x100
def AHOST_LEN_OFF : Nat := 0x0080

theorem gptoss_attn_ptx_fits :
    aPtx.all (fun t => decide (t.utf8ByteSize + 1 ≤ ASLOT)) = true := by
  native_decide

/-- The input region: the two per-step uploads first, then the weights, then
    the two narrow-kernel counts. -/
def aHostSizes : List Nat :=
  [ H * 4, 64 * 4, H * 4, QKV * H * 2, QKV * 4, NQ * 4
  , H * QO * 2, H * 4, 2 * ROPE_N * HALF * 4, 4, 4 ]

def aHostIn : AlgorithmLib.Layout.RegionMap :=
  (List.range aHostSizes.length).map (fun i =>
    ⟨s!"in{i}", (aHostSizes.take i).foldl (· + ·) 0, aHostSizes.getD i 0⟩)

theorem gptossAttnHostIn_packed :
    AlgorithmLib.Layout.RegionMap.packedB 0 aHostIn = true := by native_decide

def AHOST_BYTES : Nat := AlgorithmLib.Layout.RegionMap.total aHostIn

def aMemMap : AlgorithmLib.Layout.RegionMap :=
  (List.range aPtx.length).map (fun i => ⟨s!"ptx{i}", aSlotOff i, ASLOT⟩)
    ++ [⟨"hostLen", AHOST_LEN_OFF, 4⟩,
        ⟨"bind", ABIND_OFF, 4 * ANBUF⟩, ⟨"local", ALOCAL_OFF, 4 * 8⟩]

theorem gptossAttnMap_ok :
    aMemMap.okB = true ∧ aMemMap.withinB AMEM_SIZE = true := by native_decide

/-! ## The host program -/

/-- Buffers that are uploaded once: input region entry `2 + i` fills buffer
    `aResident[i]`. -/
def aResident : List Nat :=
  [A_ANORM, A_QKVW, A_QKVB, A_SINKS, A_OW, A_OB, A_ROPE, A_NMH, A_NMQ]

def aLoadFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit ptr
  let ctxPtr ← cudaCtxPtr ptr
  for (i, nb) in (List.range ANBUF).zip aBufBytes do
    let sz ← iconst64 nb
    let id ← cudaCreateBuffer ptr sz
    storeI32 id (← absAddr ptr (aBindOff i))
  for (k, b) in (List.range aResident.length).zip aResident do
    let src ← iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt aHostIn (2 + k))
    let id ← load32 (← absAddr ptr (aBindOff b))
    let bytes ← iconst64 (aHostSizes.getD (2 + k) 0)
    let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, src, bytes]

/-- The two per-step uploads: this token's row of the residual stream, and the
    integers that say where it sits. -/
def aUploadStep : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  for (i, b) in [(0, A_X), (1, A_META)] do
    let src ← iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt aHostIn i)
    let id ← load32 (← absAddr ptr (aBindOff b))
    let bytes ← iconst64 (aHostSizes.getD i 0)
    let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, src, bytes]

def aBindLocal (ptr : R) (bs : List Nat) : M Unit := do
  for (j, gb) in (List.range bs.length).zip bs do
    let id ← load32 (← absAddr ptr (aBindOff gb))
    storeI32 id (← absAddr ptr (ALOCAL_OFF + 4 * j))

def aEnqueue (ptr : R) (i g blk : Nat) (bs : List Nat) : M Unit := do
  aBindLocal ptr bs
  let ptxOff ← iconst64 (aSlotOff i)
  let nBufs ← iconst32 bs.length
  let bindBase ← iconst64 ALOCAL_OFF
  let one ← iconst32 1
  let block ← iconst32 blk
  let grid ← iconst32 g
  let _ ← cudaLaunch ptr ptxOff nBufs bindBase grid one one block one one

def NBLK : Nat := 32 * GptOssKernels.warpsPerCta

/-- **One layer's attention, fifteen launches.**

    Thirteen kernels and two contractions. The shapes that move with the
    position are the two contractions' — `seqLen` is read from the input region
    — and nothing else in the sequence changes from token to token. -/
def aStepFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  -- The per-step integers, read here for the contraction shapes and uploaded
  -- unchanged for the kernels: one publication, two readers, no way to
  -- disagree about how long the row is.  Named through the region map rather
  -- than as a constant, because it is an offset into the *caller's* buffer and
  -- not an address in this program's memory.
  let seqLen ← load32 (← iaddImm dataPtr
    (AlgorithmLib.Layout.RegionMap.offAt aHostIn 1 + 4 * M_SEQ))
  let seqLen64 ← uextend64 seqLen
  let zero32 ← iconst32 0
  let one32 ← iconst32 1
  let oneF ← iconst32 0x3F800000
  -- 1/sqrt(64), the attention scale, folded into the score contraction
  let scaleF ← iconst32 0x3E000000
  let bKC ← load32 (← absAddr ptr (aBindOff A_KC))
  let bVC ← load32 (← absAddr ptr (aBindOff A_VC))
  let bQKV ← load32 (← absAddr ptr (aBindOff A_QKV))
  let bSC ← load32 (← absAddr ptr (aBindOff A_SC))
  let bPR ← load32 (← absAddr ptr (aBindOff A_PR))
  let bATT ← load32 (← absAddr ptr (aBindOff A_ATT))
  let bXNB ← load32 (← absAddr ptr (aBindOff A_XNB))
  let bATTB ← load32 (← absAddr ptr (aBindOff A_ATTB))
  let bQKVW ← load32 (← absAddr ptr (aBindOff A_QKVW))
  let bOW ← load32 (← absAddr ptr (aBindOff A_OW))
  let bTMP ← load32 (← absAddr ptr (aBindOff A_TMP))
  -- normalise, narrow, project
  aEnqueue ptr S_RMS 1 32 [A_X, A_ANORM, A_XN]
  aEnqueue ptr S_NARROW ((H / 2 + NBLK - 1) / NBLK) NBLK [A_XN, A_XNB, A_NMH]
  let mQKV ← iconst32 QKV
  let kH ← iconst32 H
  let _ ← cublasGemmExBf16 ptr one32 zero32 mQKV one32 kH oneF bQKVW bXNB zero32 bQKV
  aEnqueue ptr S_ADD (QKV / 32) 32 [A_QKV, A_QKVB]
  -- rotate, then keep
  aEnqueue ptr S_ROPEQ NQ 32 [A_QKV, A_META, A_ROPE]
  aEnqueue ptr S_ROPEK NKV 32 [A_QKV, A_META, A_ROPE]
  aEnqueue ptr S_KVK NKV 32 [A_QKV, A_KC, A_META]
  aEnqueue ptr S_KVV NKV 32 [A_QKV, A_VC, A_META]
  -- scores, softmax, mix
  let hd32 ← iconst32 HD
  let gqa32 ← iconst32 GQA
  let nkv32 ← iconst32 NKV
  let strideKV ← iconst64 (CAP * HD)
  let strideQ ← iconst64 (GQA * HD)
  let gqa64 ← iconst64 GQA
  let strideS ← imul gqa64 seqLen64
  let _ ← cublasSgemmStridedBatched ptr one32 zero32 seqLen gqa32 hd32 scaleF
    bKC strideKV bQKV strideQ zero32 bSC strideS nkv32
  aEnqueue ptr S_SOFTMAX NQ 32 [A_SC, A_META, A_PR, A_SINKS]
  let strideA ← iconst64 (GQA * HD)
  let _ ← cublasSgemmStridedBatched ptr zero32 zero32 hd32 gqa32 seqLen oneF
    bVC strideKV bPR strideS zero32 bATT strideA nkv32
  -- project out and rejoin the stream
  aEnqueue ptr S_NARROW ((QO / 2 + NBLK - 1) / NBLK) NBLK [A_ATT, A_ATTB, A_NMQ]
  let mH ← iconst32 H
  let kQO ← iconst32 QO
  let _ ← cublasGemmExBf16 ptr one32 zero32 mH one32 kQO oneF bOW bATTB zero32 bTMP
  aEnqueue ptr S_ADD (H / 32) 32 [A_TMP, A_OB]
  aEnqueue ptr S_ADD (H / 32) 32 [A_X, A_TMP]
  let _ ← cudaSync ptr

def aFetchFn (b n : Nat) : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let id ← load32 (← absAddr ptr (aBindOff b))
  let bytes ← iconst64 n
  let _ ← call IR.Ffi.cudaDownload.id [ctxPtr, id, outPtr, bytes]

def aShippedBodies : List HProg.Code :=
  [ aLoadFn, aUploadStep, aStepFn, aFetchFn A_X (H * 4)
  , aFetchFn A_QKV (QKV * 4), aFetchFn A_ATT (QO * 4) ]

theorem gptossAttnShipped_wf :
    aShippedBodies.all (HProg.wf env HProg.ptrParams) = true := by
  native_decide

def aClifIR : Program :=
  program <|
    noopFunction :: aShippedBodies.attach.zipIdx.map
      (fun p =>
        HProg.compileFn (p.2 + 1) p.1.1 env
          (hwf := List.all_eq_true.mp gptossAttnShipped_wf p.1.1 p.1.2))

def aSlotBytes (t : String) : List UInt8 :=
  let b := t.toUTF8.toList ++ [0]
  b ++ zeros (ASLOT - b.length)

def aInitialMemory : List UInt8 :=
  zeros AHOST_LEN_OFF ++ u32le AHOST_BYTES
    ++ zeros (APTX_OFF - AHOST_LEN_OFF - 4)
    ++ aPtx.flatMap aSlotBytes
    ++ zeros (AMEM_SIZE - ABIND_OFF)

def aSetup : Setup := {
  clif := aClifIR
  memory_size := AMEM_SIZE
  initial_memory := aInitialMemory
}

end Attn

/-! # One whole layer

  Attention and the mixture were checked apart, which is the right order and
  not the end of it. This repo has twice reached "every layer proven" with none
  of them applied, both times because the composition was assumed rather than
  written; so the two halves are joined here and checked joined.

  What joining costs is one round trip. The router's output decides which
  experts run, and that decision is the host's — so a layer is two entry points
  with a fetch between them, not one. `stepAttn` ends at the router logits;
  the host picks four and their gates; `stepMoe` runs them and rejoins the
  residual stream. That shape is what a cache needs anyway: the gap between
  knowing which experts are wanted and running them is exactly where a miss
  would be served.
-/

namespace Layer

open GptOssAttention hiding H
open GptOssKernels (warpsPerCta rowsPerWarpDown)

def CAP : Nat := CAP_SWA

def lPtx : List String :=
  Attn.aPtx ++ gptossPtx

def S_RMS : Nat := 0
def S_ADD : Nat := 1
def S_ROPEQ : Nat := 2
def S_ROPEK : Nat := 3
def S_KVK : Nat := 4
def S_KVV : Nat := 5
def S_SOFTMAX : Nat := 6
def S_NARROW : Nat := 7
def S_GATEUP : Nat := 8
def S_DOWN : Nat := 9
def S_COMBINE : Nat := 10
def S_TOP4 : Nat := 11

/-! ## Buffers -/

def L_X : Nat := 0
def L_XN : Nat := 1
def L_XNB : Nat := 2
def L_QKVW : Nat := 3
def L_QKVB : Nat := 4
def L_QKV : Nat := 5
def L_KC : Nat := 6
def L_VC : Nat := 7
def L_SC : Nat := 8
def L_PR : Nat := 9
def L_ATT : Nat := 10
def L_ATTB : Nat := 11
def L_OW : Nat := 12
def L_OB : Nat := 13
def L_ANORM : Nat := 14
def L_SINKS : Nat := 15
def L_ROPE : Nat := 16
def L_META : Nat := 17
def L_TMP : Nat := 18
def L_NMH : Nat := 19
def L_NMQ : Nat := 20
def L_MNORM : Nat := 21
def L_RW : Nat := 22
def L_RB : Nat := 23
def L_RLOG : Nat := 24     -- the router's row, and what the host reads
def L_XM : Nat := 25       -- the hidden state the experts see
def L_GATES : Nat := 26
def L_MOUT : Nat := 27
def L_Y : Nat := 28        -- four slot outputs
def L_HID : Nat := 32      -- four slot intermediates
def LSLOT0 : Nat := 36
def LSTORE : Nat := LSLOT0 + PIECES * TOPK
/-- The four expert ids the router picked, written by a kernel and read back by
    the host program — four integers, which is the whole of what still crosses
    between them in a layer. -/
def L_CHOSEN : Nat := LSTORE + PIECES * NE
def LNBUF : Nat := L_CHOSEN + 1

def lSlotBuf (j t : Nat) : Nat := LSLOT0 + PIECES * j + t

def lBufBytes : List Nat :=
  [ H * 4, H * 4, H * 2, QKV * H * 2, QKV * 4, QKV * 4
  , NKV * CAP * HD * 4, NKV * CAP * HD * 4
  , NQ * CAP * 4, NQ * CAP * 4
  , QO * 4, QO * 2, H * QO * 2, H * 4, H * 4, NQ * 4
  , 2 * ROPE_N * HALF * 4, 64 * 4, H * 4, 4, 4
  , H * 4, NE * H * 4, NE * 4, NE * 4, H * 4, 32 * 4, H * 4 ]
    ++ List.replicate TOPK (H * 4)
    ++ List.replicate TOPK (I * 4)
    ++ List.replicate (PIECES * TOPK) 128
    ++ (List.range NE).flatMap (fun _ => pieceBytes)
    ++ [TOPK * 4]

theorem gptoss_layer_alloc_covers :
    lBufBytes.length = LNBUF ∧ lBufBytes.all (fun n => decide (0 < n)) = true := by
  native_decide

/-! ## Memory map -/

def LSLOT : Nat := 0x20000
def LPTX_OFF : Nat := 0x0100
def lSlotOff (i : Nat) : Nat := LPTX_OFF + i * LSLOT
def LBIND_OFF : Nat := lSlotOff lPtx.length
def lBindOff (i : Nat) : Nat := LBIND_OFF + 4 * i
def LLOCAL_OFF : Nat := LBIND_OFF + 4 * LNBUF
/-- Where the four chosen ids land when they come back. -/
def LCHOSEN_OFF : Nat := LLOCAL_OFF + 4 * 8
/-- Nonzero once the weights are up.  A server loads once and answers many
    times, so the load has to be something the one entry point can skip. -/
def LINIT_OFF : Nat := LCHOSEN_OFF + 4 * TOPK
def LMEM_SIZE : Nat := LINIT_OFF + 4 + 0x100
def LHOST_LEN_OFF : Nat := 0x0080

theorem gptoss_layer_ptx_fits :
    lPtx.all (fun t => decide (t.utf8ByteSize + 1 ≤ LSLOT)) = true := by
  native_decide

/-- The input region: the two per-step uploads, the dense weights, then the
    thirty-two experts. -/
def lHostSizes : List Nat :=
  [ H * 4, 64 * 4, H * 4, QKV * H * 2, QKV * 4, NQ * 4
  , H * QO * 2, H * 4, 2 * ROPE_N * HALF * 4, 4, 4
  , H * 4, NE * H * 4, NE * 4 ]
    ++ (List.range NE).flatMap (fun _ => pieceBytes)

def lHostIn : AlgorithmLib.Layout.RegionMap :=
  (List.range lHostSizes.length).map (fun i =>
    ⟨s!"in{i}", (lHostSizes.take i).foldl (· + ·) 0, lHostSizes.getD i 0⟩)

theorem gptossLayerHostIn_packed :
    AlgorithmLib.Layout.RegionMap.packedB 0 lHostIn = true := by native_decide

def LHOST_BYTES : Nat := AlgorithmLib.Layout.RegionMap.total lHostIn

def lMemMap : AlgorithmLib.Layout.RegionMap :=
  (List.range lPtx.length).map (fun i => ⟨s!"ptx{i}", lSlotOff i, LSLOT⟩)
    ++ [⟨"hostLen", LHOST_LEN_OFF, 4⟩,
        ⟨"bind", LBIND_OFF, 4 * LNBUF⟩, ⟨"local", LLOCAL_OFF, 4 * 8⟩,
        ⟨"chosen", LCHOSEN_OFF, 4 * TOPK⟩, ⟨"init", LINIT_OFF, 4⟩]

theorem gptossLayerMap_ok :
    lMemMap.okB = true ∧ lMemMap.withinB LMEM_SIZE = true := by native_decide

/-! ## The host program -/

/-- Input region entry `2 + i` fills buffer `lResident[i]`. -/
def lResident : List Nat :=
  [L_ANORM, L_QKVW, L_QKVB, L_SINKS, L_OW, L_OB, L_ROPE, L_NMH, L_NMQ,
   L_MNORM, L_RW, L_RB]

def lLoadM : M Unit := do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit ptr
  let ctxPtr ← cudaCtxPtr ptr
  for (i, nb) in (List.range LNBUF).zip lBufBytes do
    let sz ← iconst64 nb
    let id ← cudaCreateBuffer ptr sz
    storeI32 id (← absAddr ptr (lBindOff i))
  for (k, b) in (List.range lResident.length).zip lResident do
    let src ← iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt lHostIn (2 + k))
    let id ← load32 (← absAddr ptr (lBindOff b))
    let bytes ← iconst64 (lHostSizes.getD (2 + k) 0)
    let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, src, bytes]
  for k in List.range (PIECES * NE) do
    let src ← iaddImm dataPtr
      (AlgorithmLib.Layout.RegionMap.offAt lHostIn (2 + lResident.length + k))
    let id ← load32 (← absAddr ptr (lBindOff (LSTORE + k)))
    let bytes ← iconst64 (pieceBytes.getD (k % PIECES) 0)
    let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, src, bytes]

def lUploadStepM : M Unit := do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  for (i, b) in [(0, L_X), (1, L_META)] do
    let src ← iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt lHostIn i)
    let id ← load32 (← absAddr ptr (lBindOff b))
    let bytes ← iconst64 (lHostSizes.getD i 0)
    let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, src, bytes]

def lBindLocal (ptr : R) (bs : List Nat) : M Unit := do
  for (j, gb) in (List.range bs.length).zip bs do
    let id ← load32 (← absAddr ptr (lBindOff gb))
    storeI32 id (← absAddr ptr (LLOCAL_OFF + 4 * j))

def lEnqueue (ptr : R) (i g blk : Nat) (bs : List Nat) : M Unit := do
  lBindLocal ptr bs
  let ptxOff ← iconst64 (lSlotOff i)
  let nBufs ← iconst32 bs.length
  let bindBase ← iconst64 LLOCAL_OFF
  let one ← iconst32 1
  let block ← iconst32 blk
  let grid ← iconst32 g
  let _ ← cudaLaunch ptr ptxOff nBufs bindBase grid one one block one one

def NBLK : Nat := 32 * warpsPerCta

/-- **Attention, then the router.** Ends where the host has to decide. -/
def lAttnM : M Unit := do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let seqLen ← load32 (← iaddImm dataPtr
    (AlgorithmLib.Layout.RegionMap.offAt lHostIn 1 + 4 * M_SEQ))
  let seqLen64 ← uextend64 seqLen
  let zero32 ← iconst32 0
  let one32 ← iconst32 1
  let oneF ← iconst32 0x3F800000
  let scaleF ← iconst32 0x3E000000
  let bKC ← load32 (← absAddr ptr (lBindOff L_KC))
  let bVC ← load32 (← absAddr ptr (lBindOff L_VC))
  let bQKV ← load32 (← absAddr ptr (lBindOff L_QKV))
  let bSC ← load32 (← absAddr ptr (lBindOff L_SC))
  let bPR ← load32 (← absAddr ptr (lBindOff L_PR))
  let bATT ← load32 (← absAddr ptr (lBindOff L_ATT))
  let bXNB ← load32 (← absAddr ptr (lBindOff L_XNB))
  let bATTB ← load32 (← absAddr ptr (lBindOff L_ATTB))
  let bQKVW ← load32 (← absAddr ptr (lBindOff L_QKVW))
  let bOW ← load32 (← absAddr ptr (lBindOff L_OW))
  let bTMP ← load32 (← absAddr ptr (lBindOff L_TMP))
  let bRW ← load32 (← absAddr ptr (lBindOff L_RW))
  let bXM ← load32 (← absAddr ptr (lBindOff L_XM))
  let bRLOG ← load32 (← absAddr ptr (lBindOff L_RLOG))
  lEnqueue ptr S_RMS 1 32 [L_X, L_ANORM, L_XN]
  lEnqueue ptr S_NARROW ((H / 2 + NBLK - 1) / NBLK) NBLK [L_XN, L_XNB, L_NMH]
  let mQKV ← iconst32 QKV
  let kH ← iconst32 H
  let _ ← cublasGemmExBf16 ptr one32 zero32 mQKV one32 kH oneF bQKVW bXNB zero32 bQKV
  lEnqueue ptr S_ADD (QKV / 32) 32 [L_QKV, L_QKVB]
  lEnqueue ptr S_ROPEQ NQ 32 [L_QKV, L_META, L_ROPE]
  lEnqueue ptr S_ROPEK NKV 32 [L_QKV, L_META, L_ROPE]
  lEnqueue ptr S_KVK NKV 32 [L_QKV, L_KC, L_META]
  lEnqueue ptr S_KVV NKV 32 [L_QKV, L_VC, L_META]
  let hd32 ← iconst32 HD
  let gqa32 ← iconst32 GQA
  let nkv32 ← iconst32 NKV
  let strideKV ← iconst64 (CAP * HD)
  let strideQ ← iconst64 (GQA * HD)
  let gqa64 ← iconst64 GQA
  let strideS ← imul gqa64 seqLen64
  let _ ← cublasSgemmStridedBatched ptr one32 zero32 seqLen gqa32 hd32 scaleF
    bKC strideKV bQKV strideQ zero32 bSC strideS nkv32
  lEnqueue ptr S_SOFTMAX NQ 32 [L_SC, L_META, L_PR, L_SINKS]
  let strideA ← iconst64 (GQA * HD)
  let _ ← cublasSgemmStridedBatched ptr zero32 zero32 hd32 gqa32 seqLen oneF
    bVC strideKV bPR strideS zero32 bATT strideA nkv32
  lEnqueue ptr S_NARROW ((QO / 2 + NBLK - 1) / NBLK) NBLK [L_ATT, L_ATTB, L_NMQ]
  let mH ← iconst32 H
  let kQO ← iconst32 QO
  let _ ← cublasGemmExBf16 ptr one32 zero32 mH one32 kQO oneF bOW bATTB zero32 bTMP
  lEnqueue ptr S_ADD (H / 32) 32 [L_TMP, L_OB]
  lEnqueue ptr S_ADD (H / 32) 32 [L_X, L_TMP]
  -- the router: a 32-row projection of the normalised stream, kept in f32
  -- because thirty-two rows is not what a decode step is bound by
  lEnqueue ptr S_RMS 1 32 [L_X, L_MNORM, L_XM]
  let ne32 ← iconst32 NE
  let _ ← cublasSgemv ptr one32 kH ne32 oneF bRW bXM zero32 bRLOG
  lEnqueue ptr S_ADD 1 32 [L_RLOG, L_RB]
  -- …and the choice, taken on the device.  One warp, a hundred and twenty-eight
  -- comparisons, and the four ids come back as sixteen bytes instead of the
  -- router's whole row going out and four buffer bindings coming in.
  lEnqueue ptr S_TOP4 1 32 [L_RLOG, L_GATES, L_CHOSEN]
  let _ ← cudaSync ptr
  let ctxPtr ← cudaCtxPtr ptr
  let bCh ← load32 (← absAddr ptr (lBindOff L_CHOSEN))
  let dst ← absAddr ptr LCHOSEN_OFF
  let n16 ← iconst64 (TOPK * 4)
  let _ ← call IR.Ffi.cudaDownload.id [ctxPtr, bCh, dst, n16]

/-- **Binding the four slots, from ids the device chose.**

    Reads `LCHOSEN_OFF` — memory this program wrote — rather than the caller's
    buffer, which is the difference between a layer that is one call and a
    layer that is two.  The six moves per slot are unchanged: what changed is
    where the expert index came from. -/
def lBindM : M Unit := do
  let ptr := basePtr
  let four ← iconst64 4
  let six ← iconst64 PIECES
  for j in List.range TOPK do
    let e ← load32 (← absAddr ptr (LCHOSEN_OFF + 4 * j))
    let e64 ← uextend64 e
    let e6 ← imul e64 six
    for t in List.range PIECES do
      let srcIx ← iaddImm e6 (LSTORE + t)
      let srcOff ← imul srcIx four
      let srcAddr ← iadd (← absAddr ptr LBIND_OFF) srcOff
      let id ← load32 srcAddr
      storeI32 id (← absAddr ptr (lBindOff (lSlotBuf j t)))

/-- **The mixture, and the second residual.** -/
def lExpertBinds : List (Nat × Nat × List Nat) :=
  (List.range TOPK).flatMap (fun j =>
    [ (S_GATEUP, I / (warpsPerCta * rowsPerWarpGateUp),
       [lSlotBuf j 0, lSlotBuf j 1, lSlotBuf j 2, L_XM, L_HID + j])
    , (S_DOWN, H / (warpsPerCta * rowsPerWarpDown),
       [lSlotBuf j 3, lSlotBuf j 4, lSlotBuf j 5, L_HID + j, L_Y + j]) ])
  ++ [ (S_COMBINE, (H + NBLK - 1) / NBLK,
        [L_Y, L_Y + 1, L_Y + 2, L_Y + 3, L_GATES, L_MOUT]) ]

def lMoeM : M Unit := do
  let ptr := basePtr
  for (i, g, bs) in lExpertBinds do
    lEnqueue ptr i g NBLK bs
  lEnqueue ptr S_ADD (H / 32) 32 [L_X, L_MOUT]
  let _ ← cudaSync ptr

/-- **Seam guard: the mixture half of a whole layer still never names the
    store.** The same property `gptoss_slots_are_bound` states for the isolated
    dispatch, restated here because the buffer numbering is different and a
    guard about the other numbering would say nothing about this one. -/
theorem gptoss_layer_slots_are_bound :
    (lExpertBinds.flatMap (fun t => t.2.2)).all (fun b => decide (b < LSTORE))
      = true := by native_decide

theorem gptoss_layer_binds_allocated :
    (lExpertBinds.flatMap (fun t => t.2.2)).all (fun b => decide (b < LNBUF))
      = true := by native_decide

def lUploadFn (b n : Nat) : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let id ← load32 (← absAddr ptr (lBindOff b))
  let bytes ← iconst64 n
  let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, dataPtr, bytes]

/-- Copy buffer `b` to the caller's output, `at` bytes in. -/
def lFetchM (b n at_ : Nat) : M Unit := do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let dst ← iaddImm outPtr at_
  let id ← load32 (← absAddr ptr (lBindOff b))
  let bytes ← iconst64 n
  let _ ← call IR.Ffi.cudaDownload.id [ctxPtr, id, dst, bytes]

/-- **One token through one layer, and nothing outside it.**

    The artifact carries a single entry and no extras, which is the shape
    `qwen2.json` has and the shape this was not: seven entries, once, each of
    them a place the caller held state that belonged in here.

    They went in three steps and the order matters. The router's choice became
    a kernel, so the layer stopped having to ask. The load became conditional
    on a flag in this program's own memory, so one entry can load once and
    answer many times — which is what a server does. And the diagnostic that
    was its own entry became a second write into the caller's output buffer,
    because a thing worth reading is worth returning, not worth another door.

    What the caller passes is a token's row and its position; what it gets back
    is the layer's output and, after it, the router's row. -/
def lMainFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let flag ← load32 (← absAddr ptr LINIT_OFF)
  let zero32 ← iconst32 0
  when .eq flag zero32 (do
    lLoadM
    let one32 ← iconst32 1
    storeI32 one32 (← absAddr ptr LINIT_OFF))
  lUploadStepM
  lAttnM
  lBindM
  lMoeM
  lFetchM L_X (H * 4) 0
  lFetchM L_RLOG (NE * 4) (H * 4)

def lShippedBodies : List HProg.Code := [ lMainFn ]

theorem gptossLayerShipped_wf :
    lShippedBodies.all (HProg.wf env HProg.ptrParams) = true := by
  native_decide

def lClifIR : Program :=
  program <|
    noopFunction :: lShippedBodies.attach.zipIdx.map
      (fun p =>
        HProg.compileFn (p.2 + 1) p.1.1 env
          (hwf := List.all_eq_true.mp gptossLayerShipped_wf p.1.1 p.1.2))

def lSlotBytes (t : String) : List UInt8 :=
  let b := t.toUTF8.toList ++ [0]
  b ++ zeros (LSLOT - b.length)

def lInitialMemory : List UInt8 :=
  zeros LHOST_LEN_OFF ++ u32le LHOST_BYTES
    ++ zeros (LPTX_OFF - LHOST_LEN_OFF - 4)
    ++ lPtx.flatMap lSlotBytes
    ++ zeros (LMEM_SIZE - LBIND_OFF)

def lSetup : Setup := {
  clif := lClifIR
  memory_size := LMEM_SIZE
  initial_memory := lInitialMemory
}

end Layer

#eval LayoutScan.check "GptOssAlgorithm" [``gMemMap, ``Attn.aMemMap, ``Layer.lMemMap]

def artifacts : Array Json :=
  #[ toJsonArtifact "gptoss_moe" gSetup { fn_idx := u32 1 }
       [("bindExperts", { fn_idx := u32 2 }),
        ("uploadX", { fn_idx := u32 3 }),
        ("uploadGates", { fn_idx := u32 4 }),
        ("runExperts", { fn_idx := u32 5 }),
        ("fetchOut", { fn_idx := u32 6 })]
   , toJsonArtifact "gptoss_attn" Attn.aSetup { fn_idx := u32 1 }
       [("uploadStep", { fn_idx := u32 2 }),
        ("step", { fn_idx := u32 3 }),
        ("fetchX", { fn_idx := u32 4 }),
        ("fetchQkv", { fn_idx := u32 5 }),
        ("fetchAtt", { fn_idx := u32 6 })]
   , toJsonArtifact "gptoss_layer" Layer.lSetup { fn_idx := u32 1 } [] ]

end GptOssAlgorithm

def main (args : List String) : IO Unit := do
  emitArtifacts (← requireOutputDir args) GptOssAlgorithm.artifacts

#eval ShipScan.check "GptOssAlgorithm"
