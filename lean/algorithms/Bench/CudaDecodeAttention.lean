module
public import Lean
public import Std
public import AlgorithmLib.Vocab.PTX
meta import AlgorithmLib.Vocab.PTX
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.PTX

namespace CudaDecodeAttention

/-!
  Persistent single-token decode attention benchmark.

  Fixed dimensions:
    n_heads  = 14
    head_dim = 64
    d_model  = 896

  Runtime dimension:
    seq_len <= 2048

  Layouts:
    q      [n_heads, head_dim]
    k/v    [n_heads, seq_len, head_dim]
    out    [n_heads, head_dim]

  Load payload:
    [seq_len: u64]
    [k_cache: n_heads * seq_len * head_dim f32]
    [v_cache: n_heads * seq_len * head_dim f32]

  Prep payload:
    [q: n_heads * head_dim f32]

  Timed infer:
    1. scores = scale * (K @ q)       batched over heads via cuBLAS
    2. probs  = softmax(scores)       one CUDA block per head
    3. out    = V^T @ probs           batched over heads via cuBLAS

  Entry points:
    main      — for 0 < seq_len ≤ 2048 and data holding exactly K and V:
                make the allocations, upload K and V once
    prep      — upload q, where the data holds it
    infer     — core, then finalize; stack: core 64 times, then finalize
    core      — scores + softmax + V mix, no wait or download
    finalize  — wait, then download as many bytes of out as the output
                asks for, where they fit

  The program calls the CUDA driver and cuBLAS directly, in `Drv.App`'s
  layout. Allocations, made by `main` in this order: q, K and V back to back
  as the data has them, scores, probs, out, meta.
-/

def N_HEADS : Nat := 14
def HEAD_DIM : Nat := 64
def D_MODEL : Nat := N_HEADS * HEAD_DIM
def MAX_SEQ : Nat := 2048

def D_MODEL_BYTES : Nat := D_MODEL * 4

/-- Where `main` keeps `seq_len`; where the `0.125` the scores are scaled by
    lies; where the driver's state starts; where the PTX lies and how many
    bytes are kept for it. -/
def SEQ_LEN_OFF : Nat := 0x58
def EIGHTH_OFF : Nat := 0x60
def ORG : Nat := 0x100
def PTX_OFF : Nat := 0x400
def PTX_LEN : Nat := 0x4C00

def ptxSource : String := buildModule 64 [{ name := "main", params := ["scores_buf", "meta_buf", "probs_buf"], body := do
  let scoresBuf ← ldParam "scores_buf"
  let metaPtr   ← ldParam "meta_buf"
  let probsBuf  ← ldParam "probs_buf"
  let seqLen ← freshR; ldGlobalU seqLen metaPtr
  let (tid, wid, lid) ← getWarpIds
  let headIdx  ← freshR; movR headIdx ctaX
  let headId64 ← freshRd; cvtU64 headId64 headIdx
  let seqLen64 ← freshRd; cvtU64 seqLen64 seqLen
  let headOff  ← freshRd; mulLoRd headOff headId64 seqLen64
  let byteOff  ← freshRd; shlRd byteOff headOff 2
  let scoresBase ← freshRd; addRd scoresBase scoresBuf byteOff
  let probsBase  ← freshRd; addRd probsBase  probsBuf  byteOff
  let log2e ← freshF; movFC log2e f32_log2e
  -- Phase 1: max reduction over this head's scores
  let lMax ← freshF; movFC lMax f32_0
  let mTmp ← freshF
  strideLoop tid seqLen 256 "sm_loop_max" "sm_done_max" fun i => do
    let addr ← elemAddr scoresBase i; ldGlobalF mTmp addr; maxF lMax lMax mTmp
  warpReduceMax lMax mTmp
  lane0WriteSmem lid wid "sm_skip_max_store" fun wAddr => stSharedFD wAddr lMax
  thread0Op tid "sm_skip_max_reduce" do
    let sBase ← smemBase; let gMax ← freshF
    crossWarp8 gMax mTmp sBase 0 maxF; stSharedF sBase 32 gMax
  let sBase1 ← smemBase
  let gMax ← freshF; ldSharedF gMax sBase1 32
  -- Phase 2: sum of exp(score - max)
  let lSum ← freshF; movFC lSum f32_0
  let sTmp ← freshF
  strideLoop tid seqLen 256 "sm_loop_sum" "sm_done_sum" fun i => do
    let addr ← elemAddr scoresBase i; let xi ← freshF; ldGlobalF xi addr
    subF xi xi gMax; mulF xi xi log2e; ex2 xi xi; addF lSum lSum xi
  warpReduceSum lSum sTmp
  lane0WriteSmem lid wid "sm_skip_sum_store" fun wAddr => stSharedFD wAddr lSum
  thread0Op tid "sm_skip_sum_reduce" do
    let sBase ← smemBase
    crossWarp8 lSum sTmp sBase 0 addF; rcp lSum lSum; stSharedF sBase 36 lSum
  let sBase2 ← smemBase
  let invSum ← freshF; ldSharedF invSum sBase2 36
  -- Phase 3: write softmax probabilities
  strideLoop tid seqLen 256 "sm_loop_out" "sm_done_out" fun i => do
    let sAddr ← elemAddr scoresBase i; let pAddr ← elemAddr probsBase i
    let xi ← freshF; ldGlobalF xi sAddr
    subF xi xi gMax; mulF xi xi log2e; ex2 xi xi; mulF xi xi invSum
    stGlobalF pAddr xi
  ptxRet }]

-- Load: init CUDA, alloc 7 bufs, upload K/V/meta, store buf IDs and seq_len
open AlgorithmLib.Prog

/-- **The program in the driver's standard layout**: six allocations, made by
    `main`, cuBLAS, and the softmax kernel handed scores, meta and probs. -/
def app : Drv.App :=
  { bufs := [0, 0, 0, 0, 0, 0], kernels := [⟨PTX_OFF, PTX_LEN, [2, 5, 3], (N_HEADS, 1, 1), (256, 1, 1)⟩],
    memSize := PTX_OFF + PTX_LEN, blas := true, org := ORG }

/-- Bring the driver and cuBLAS up and load the kernel; then, where the data
    is `seq_len`, `0 < seq_len ≤ 2048`, then exactly K and V: keep
    `seq_len`, make the allocations, hand them to the kernel and upload
    `seq_len` and K and V. -/
def loadCode : Prog V L Unit := do
  let a := app
  let base ← basePtr
  let dp ← dataPtr
  let dl ← dataLen
  a.setup base
  Prog.when .ule (← iconst64 8) dl do
  Prog.when .ult dl (← iconst64 (2 ^ 35)) do
    let seq ← load64 dp
    Prog.when .ne seq (← iconst64 0) do
    Prog.when .ult seq (← iconst64 (MAX_SEQ + 1)) do
      let kvBytes ← isub dl (← iconst64 8)
      Prog.when .eq (← imul seq (← iconst64 (2 * D_MODEL_BYTES))) kvBytes do
        storeI64 seq (← iaddImm base SEQ_LEN_OFF)
        let scoreBytes ← imul seq (← iconst64 (N_HEADS * 4))
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 0), ← iconst64 D_MODEL_BYTES]
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 1), kvBytes]
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 2), scoreBytes]
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 3), scoreBytes]
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 4), ← iconst64 D_MODEL_BYTES]
        let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 5), ← iconst64 8]
        a.rebind base 0
        let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (a.bufSlot 5)), ← iaddImm base SEQ_LEN_OFF,
          ← iconst64 8]
        let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (a.bufSlot 1)), ← iaddImm dp 8, kvBytes]

/-- Upload `q`, where the data holds it. -/
def prepCode : Prog V L Unit := do
  let base ← basePtr
  Prog.when .ule (← iconst64 D_MODEL_BYTES) (← dataLen) do
    let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (app.bufSlot 0)), ← dataPtr,
      ← iconst64 D_MODEL_BYTES]

/-- `scores = 0.125 · K q` per head, the softmax kernel, `out = Vᵀ probs`
    per head. K and V are `seq_len × head_dim` per head, row-major, so cuBLAS
    reads each as the column-major `head_dim × seq_len` matrix. -/
def coreCode : Prog V L Unit := do
  let a := app
  let base ← basePtr
  let seq ← load64 (← iaddImm base SEQ_LEN_OFF)
  Prog.when .ne seq (← iconst64 0) do
  Prog.when .ult seq (← iconst64 (MAX_SEQ + 1)) do
    let seq32 ← ireduce32 seq
    let seqHead ← imul seq (← iconst64 HEAD_DIM)
    let one ← iconst32 1
    let zero ← iconst32 0
    let hd ← iconst32 HEAD_DIM
    let heads ← iconst32 N_HEADS
    let h ← load64 (← iaddImm base a.BLAS)
    let kv ← load64 (← iaddImm base (a.bufSlot 1))
    let _ ← ext (.cublas .sgemmStridedBatched) %[h, one, zero, seq32, one, hd, ← iaddImm base EIGHTH_OFF,
      kv, hd, seqHead, ← load64 (← iaddImm base (a.bufSlot 0)), hd, ← iconst64 HEAD_DIM,
      ← iaddImm base a.ZERO, ← load64 (← iaddImm base (a.bufSlot 2)), seq32, seq, heads]
    a.run base 0
    let vv ← iadd kv (← imul seq (← iconst64 D_MODEL_BYTES))
    let _ ← ext (.cublas .sgemmStridedBatched) %[h, zero, zero, hd, one, seq32, ← iaddImm base a.ONE,
      vv, hd, seqHead, ← load64 (← iaddImm base (a.bufSlot 3)), seq32, seq,
      ← iaddImm base a.ZERO, ← load64 (← iaddImm base (a.bufSlot 4)), hd, ← iconst64 HEAD_DIM, heads]

/-- Wait; then download as many bytes of `out` as the caller's output asks
    for, where it asks for some and they fit. -/
def finalizeCode : Prog V L Unit := do
  let base ← basePtr
  Drv.sync
  let ol ← outLen
  Prog.when .ne ol (← iconst64 0) do
  Prog.when .ule ol (← iconst64 D_MODEL_BYTES) do
    let _ ← ext (.cuda .memcpyDtoH) %[← outPtr, ← load64 (← iaddImm base (app.bufSlot 4)), ol]

/-- Stack depth baked into the `stack` entry's wrapper. -/
def STACK_DEPTH : Nat := 64

def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 loadCode),
     Prog.entry "prep" (Prog.compileProg 2 prepCode),
     Prog.compileProg 3 coreCode,
     Prog.compileProg 4 finalizeCode,
     Prog.entry "infer" (Prog.compileProg 5 (Prog.sequenceWrapper [3, 4])),
     Prog.entry "stack" (Prog.compileProg 6 (Prog.sequenceWrapper (List.replicate STACK_DEPTH 3 ++ [4])))]

def buildSetup (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := app.memSize,
  initial_memory := app.initialMemory [ptxSource] [(EIGHTH_OFF, [0, 0, 0, 0x3e])]
}

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[
    artifactEntry "cuda_decode_attn" (buildSetup clif)
  ]

end CudaDecodeAttention
