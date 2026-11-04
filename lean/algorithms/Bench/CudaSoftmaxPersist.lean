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

namespace CudaSoftmaxPersist

/-!
Multi-block persistent softmax over `n` floats.

Allocations, made by `main` in this order:
  0  x        — input floats (n f32)
  1  y        — output floats (n f32)
  2  meta     — [n: u32, num_blocks: u32]
  3  partials — [max0, sum0, max1, sum1, ...] (num_blocks · 8 bytes)
  4  params   — [global_max: f32, global_inv_sum: f32], written by global_reduce

Entry points:
- `main`, data `[n: u64]`: brings the driver up and, for `0 < n < 2^30`,
  keeps `n` and `num_blocks = ⌈n / 256⌉`, makes the allocations, hands them
  to the kernels and uploads `meta`.
- `prep`, data `[x: n f32]`: uploads `x`, where it fits.
- `infer`: `core` then `finalize`; `stack`: `core` 64 times, then `finalize`.
  `core` runs one kernel for `n ≤ 2048` and three otherwise; `finalize`
  waits, then downloads as many bytes of `y` as the output asks for, where
  they fit.

Four PTX entry points in one module:
  small_softmax — one block, for a short row
  block_reduce  — num_blocks blocks: each block's (local_max, local_sum)
  global_reduce — one block: (global_max, global_inv_sum) into params
  normalize     — num_blocks blocks: y[i] = ex2((x[i]-global_max)·log2e) · global_inv_sum

The program calls the CUDA driver directly, in `Drv.App`'s layout; every
allocation but `meta` and `params` is sized at run time.
-/

/-- Where `main` keeps `n`, `num_blocks` and the packed `meta` it uploads;
    where the driver's state starts; where the PTX lies and how many bytes
    are kept for it. -/
def N_OFF : Nat := 0x38
def NB_OFF : Nat := 0x40
def META_OFF : Nat := 0x48
def ORG : Nat := 0x100
def PTX_OFF : Nat := 0x800
def PTX_LEN : Nat := 0x4800

/-
  PTX: three kernels share one module. smem[0..31] = 8 warp maxes,
  smem[32..63] = 8 warp sums. Online combination: given (m1,s1)+(m2,s2),
  new_max = max(m1,m2), new_sum = s1*ex2((m1-new_max)*log2e) + s2*ex2((m2-new_max)*log2e).
-/
def ptxSource : String := buildModule 64
  [ -- ── small_softmax: single-block fused path for n ≤ 2048 ─────────────────
    { name := "small_softmax", params := ["x_buf", "meta_buf", "y_buf"], body := do
      let xPtr    ← ldParam "x_buf"
      let metaPtr ← ldParam "meta_buf"
      let yPtr    ← ldParam "y_buf"
      let nReg ← freshR; ldGlobalU nReg metaPtr
      let (tid, wid, lid) ← getWarpIds
      let log2e ← freshF; movFC log2e f32_log2e
      -- Phase 1: max reduction
      let lMax ← freshF; movFC lMax f32_0
      let mTmp ← freshF
      strideLoop tid nReg 256 "sm_loop_max" "sm_done_max" fun i => do
        let addr ← elemAddr xPtr i; ldGlobalF mTmp addr; maxF lMax lMax mTmp
      warpReduceMax lMax mTmp
      lane0WriteSmem lid wid "sm_skip_max_store" fun wAddr => stSharedFD wAddr lMax
      thread0Op tid "sm_skip_max_reduce" do
        let sBase ← smemBase; let gMax ← freshF
        crossWarp8 gMax mTmp sBase 0 maxF; stSharedF sBase 32 gMax
      let sBase1 ← smemBase
      let gMax ← freshF; ldSharedF gMax sBase1 32
      -- Phase 2: sum of exp
      let lSum ← freshF; movFC lSum f32_0
      let sTmp ← freshF
      strideLoop tid nReg 256 "sm_loop_sum" "sm_done_sum" fun i => do
        let addr ← elemAddr xPtr i; let xi ← freshF; ldGlobalF xi addr
        subF xi xi gMax; mulF xi xi log2e; ex2 xi xi; addF lSum lSum xi
      warpReduceSum lSum sTmp
      lane0WriteSmem lid wid "sm_skip_sum_store" fun wAddr => stSharedFD wAddr lSum
      thread0Op tid "sm_skip_sum_reduce" do
        let sBase ← smemBase
        crossWarp8 lSum sTmp sBase 0 addF; rcp lSum lSum; stSharedF sBase 36 lSum
      let sBase2 ← smemBase
      let invSum ← freshF; ldSharedF invSum sBase2 36
      -- Phase 3: output
      strideLoop tid nReg 256 "sm_loop_out" "sm_done_out" fun i => do
        let xAddr ← elemAddr xPtr i; let yAddr ← elemAddr yPtr i
        let xi ← freshF; ldGlobalF xi xAddr
        subF xi xi gMax; mulF xi xi log2e; ex2 xi xi; mulF xi xi invSum
        stGlobalF yAddr xi
      ptxRet },
    -- ── block_reduce: each block → (local_max, local_sum) via online softmax ──
    { name := "block_reduce", params := ["x_buf", "meta_buf", "partials_buf"], body := do
      let xPtr        ← ldParam "x_buf"
      let metaPtr     ← ldParam "meta_buf"
      let partialsPtr ← ldParam "partials_buf"
      let nReg ← freshR; ldGlobalU nReg metaPtr
      let bid ← freshR; movR bid ctaX
      let bsz ← freshR; movR bsz ntidX
      let tid ← freshR; movR tid tidX
      let wid ← freshR; shrR wid tid 5
      let lid ← freshR; andR lid tid 31
      let gid ← freshR; madLoS gid bid bsz tid
      let log2e ← freshF; movFC log2e f32_log2e
      let lMax ← freshF; let lSum ← freshF
      let p ← freshP; setpGe p gid nReg; braIf p "br_oob"
      let xAddr ← elemAddr xPtr gid; ldGlobalF lMax xAddr; movFC lSum f32_1
      bra "br_reduce"
      label "br_oob"; movFC lMax f32_0; movFC lSum f32_0
      label "br_reduce"
      let t0 ← freshF; let t1 ← freshF; let nm ← freshF; let adj ← freshF
      warpReduceOnline lMax lSum t0 t1 nm adj log2e
      lane0WriteSmem lid wid "br_skip_wst" fun wAddr => do
        stSharedFD wAddr lMax
        let sumAddr ← freshR; addRI sumAddr wAddr 32; stSharedFD sumAddr lSum
      thread0Op tid "br_skip_bred" do
        let sBase ← smemBase; let bMax ← freshF; let bSum ← freshF
        crossWarp8Online bMax bSum t0 t1 nm adj log2e sBase 0 32
        let bid64 ← freshRd; cvtU64 bid64 bid
        let pOff ← freshRd; shlRd pOff bid64 3
        let pAddr ← freshRd; addRd pAddr partialsPtr pOff
        stGlobalF pAddr bMax; stGlobalFO pAddr 4 bSum
      ptxRet },
    -- ── global_reduce: combine partials → (global_max, global_inv_sum) ────────
    { name := "global_reduce", params := ["meta_buf", "partials_buf", "params_buf"], body := do
      let metaPtr     ← ldParam "meta_buf"
      let partialsPtr ← ldParam "partials_buf"
      let paramsPtr   ← ldParam "params_buf"
      let nbPtr ← freshRd; addRdI nbPtr metaPtr 4
      let numBlocks ← freshR; ldGlobalU numBlocks nbPtr
      let tid ← freshR; movR tid tidX
      let wid ← freshR; shrR wid tid 5
      let lid ← freshR; andR lid tid 31
      let log2e ← freshF; movFC log2e f32_log2e
      let lMax ← freshF; movFC lMax f32_0
      let lSum ← freshF; movFC lSum f32_0
      let nm ← freshF; let adj ← freshF; let t0 ← freshF; let t1 ← freshF
      strideLoop tid numBlocks 256 "gr_loop" "gr_done" fun i => do
        let i64 ← freshRd; cvtU64 i64 i
        let pOff ← freshRd; shlRd pOff i64 3
        let pAddr ← freshRd; addRd pAddr partialsPtr pOff
        let pMax ← freshF; ldGlobalF pMax pAddr
        let pSum ← freshF; ldGlobalFO pSum pAddr 4
        maxF nm lMax pMax
        subF adj lMax nm; mulF adj adj log2e; ex2 adj adj; mulF lSum lSum adj
        subF adj pMax nm; mulF adj adj log2e; ex2 adj adj; mulF pSum pSum adj
        addF lSum lSum pSum; movF lMax nm
      warpReduceOnline lMax lSum t0 t1 nm adj log2e
      lane0WriteSmem lid wid "gr_skip_wst" fun wAddr => do
        stSharedFD wAddr lMax
        let sumAddr ← freshR; addRI sumAddr wAddr 32; stSharedFD sumAddr lSum
      thread0Op tid "gr_skip_bred" do
        let sBase ← smemBase; let bMax ← freshF; let bSum ← freshF
        crossWarp8Online bMax bSum t0 t1 nm adj log2e sBase 0 32
        rcp bSum bSum; stGlobalF paramsPtr bMax; stGlobalFO paramsPtr 4 bSum
      ptxRet },
    -- ── normalize: y[i] = exp((x[i]-global_max)*log2e) * global_inv_sum ───────
    { name := "normalize", params := ["x_buf", "meta_buf", "params_buf", "y_buf"], body := do
      let xPtr      ← ldParam "x_buf"
      let metaPtr   ← ldParam "meta_buf"
      let paramsPtr ← ldParam "params_buf"
      let yPtr      ← ldParam "y_buf"
      let nReg   ← freshR; ldGlobalU nReg metaPtr
      let gMax   ← freshF; ldGlobalF gMax paramsPtr
      let invSum ← freshF; ldGlobalFO invSum paramsPtr 4
      let log2e  ← freshF; movFC log2e f32_log2e
      let (gid, _) ← gridStrideSetup nReg "nrm_end"
      let xAddr ← elemAddr xPtr gid; let yAddr ← elemAddr yPtr gid
      let xi ← freshF; ldGlobalF xi xAddr
      subF xi xi gMax; mulF xi xi log2e; ex2 xi xi; mulF xi xi invSum
      stGlobalF yAddr xi
      label "nrm_end"; ptxRet } ]

-- CLIF: u0:0 noop, u0:1 load, u0:2 prep, u0:3 core, u0:4 finalize
-- Load: init CUDA, alloc 5 bufs, pack/upload meta, store n/num_blocks
open AlgorithmLib.Prog

/-- **The program in the driver's standard layout**: five allocations, made
    by `main`, and the four kernels of one module. -/
def app : Drv.App :=
  { bufs := [0, 0, 0, 0, 0],
    kernels := [⟨PTX_OFF, PTX_LEN, [0, 2, 1], (1, 1, 1), (256, 1, 1)⟩,
                ⟨PTX_OFF, PTX_LEN, [0, 2, 3], (1, 1, 1), (256, 1, 1)⟩,
                ⟨PTX_OFF, PTX_LEN, [2, 3, 4], (1, 1, 1), (256, 1, 1)⟩,
                ⟨PTX_OFF, PTX_LEN, [0, 2, 4, 1], (1, 1, 1), (256, 1, 1)⟩],
    memSize := PTX_OFF + PTX_LEN, org := ORG,
    entries := ["small_softmax", "block_reduce", "global_reduce", "normalize"] }

def loadCode : Prog V L Unit := do
  let a := app
  let base ← basePtr
  let dp ← dataPtr
  a.setup base
  Prog.when .ule (← iconst64 8) (← dataLen) do
    let n ← load64 dp
    Prog.when .ne n (← iconst64 0) do
    Prog.when .ult n (← iconst64 (2 ^ 30)) do
      let nb ← ushrImm (← iaddImm n 255) 8
      storeI64 n (← iaddImm base N_OFF)
      storeI64 nb (← iaddImm base NB_OFF)
      let xBytes ← imul n (← iconst64 4)
      let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 0), xBytes]
      let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 1), xBytes]
      let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 2), ← iconst64 8]
      let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 3), ← imul nb (← iconst64 8)]
      let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot 4), ← iconst64 8]
      for j in List.range a.kernels.length do
        a.rebind base j
      storeI64 (← bor n (← ishlImm nb 32)) (← iaddImm base META_OFF)
      let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (a.bufSlot 2)), ← iaddImm base META_OFF,
        ← iconst64 8]

/-- Upload `x`, where it fits. -/
def prepCode : Prog V L Unit := do
  let base ← basePtr
  let dl ← dataLen
  let n ← load64 (← iaddImm base N_OFF)
  Prog.when .ule dl (← imul n (← iconst64 4)) do
    let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base (app.bufSlot 0)), ← dataPtr, dl]

/-- One kernel for a short row, three for a long one. -/
def coreCode : Prog V L Unit := do
  let a := app
  let base ← basePtr
  let n ← load64 (← iaddImm base N_OFF)
  let nb ← load64 (← iaddImm base NB_OFF)
  Prog.when .ne nb (← iconst64 0) do
  Prog.when .ult nb (← iconst64 (2 ^ 31)) do
    let nb32 ← ireduce32 nb
    let _ ← ifte .ule n (← iconst64 2048)
      (thn := do a.run base 0; pure %[])
      (els := do
        a.runX base 1 nb32
        a.run base 2
        a.runX base 3 nb32
        pure %[])
    pure ()

/-- Wait; then download as many bytes of `y` as the caller's output asks
    for, where it asks for some and they fit. -/
def finalizeCode : Prog V L Unit := do
  let base ← basePtr
  let n ← load64 (← iaddImm base N_OFF)
  Drv.sync
  let ol ← outLen
  Prog.when .ne ol (← iconst64 0) do
  Prog.when .ult n (← iconst64 (2 ^ 30)) do
    Prog.when .ule ol (← imul n (← iconst64 4)) do
      let _ ← ext (.cuda .memcpyDtoH) %[← outPtr, ← load64 (← iaddImm base (app.bufSlot 1)), ol]

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
  initial_memory := app.initialMemory (List.replicate 4 ptxSource)
}

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[
    artifactEntry "cuda_softmax" (buildSetup clif)
  ]

end CudaSoftmaxPersist
