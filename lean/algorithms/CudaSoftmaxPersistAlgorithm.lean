import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.PTX

namespace CudaSoftmaxPersist

/-
  Multi-block persistent softmax.

  Buffers (created in load, by index):
    0  x        — input floats (n f32)
    1  y        — output floats (n f32)
    2  meta     — [n: u32, num_blocks: u32] (8 bytes, written once in load)
    3  partials — [max0, sum0, max1, sum1, ...] (num_blocks * 8 bytes)
    4  params   — [global_max: f32, global_inv_sum: f32] (8 bytes, written by global_reduce)

  Algorithms:
    u0:1  load      — init CUDA, alloc 5 bufs, write meta, store n/num_blocks in sm
    u0:2  prep      — upload x to buf0
    u0:3  core      — launch 3 kernels, no sync/download
    u0:4  finalize  — cl_cuda_sync, optional download y

  Shared memory layout:
    0x00-0x37  reserved (56-byte IoOffsets)
    0x38-0x3F  n (i64)
    0x40-0x47  num_blocks (i64)
    0x48-0x4F  packed meta [n:u32][num_blocks:u32] staging area

  Three PTX entry points in one module (called via cl_cuda_launch_named):
    block_reduce  — gridDim=num_blocks: each block computes (local_max, local_sum) via online softmax
    global_reduce — gridDim=1: combines partials into (global_max, global_inv_sum) in params buf
    normalize     — gridDim=num_blocks: y[i] = ex2((x[i]-global_max)*log2e) * global_inv_sum
-/

-- initial_memory offsets
def NAME_BLOCK_REDUCE  : Nat := 0x0100  -- "block_reduce\0"
def NAME_GLOBAL_REDUCE : Nat := 0x0110  -- "global_reduce\0"
def NAME_NORMALIZE     : Nat := 0x0120  -- "normalize\0"
def NAME_SMALL_SOFTMAX : Nat := 0x0130  -- "small_softmax\0"
def PTX_SOURCE_OFF     : Nat := 0x0200
def BIND_K1_OFF        : Nat := 0x4700  -- [0,2,3]  x, meta, partials
def BIND_K2_OFF        : Nat := 0x4710  -- [2,3,4]  meta, partials, params
def BIND_K3_OFF        : Nat := 0x4720  -- [0,2,4,1] x, meta, params, y
def BIND_SMALL_OFF     : Nat := 0x4730  -- [0,2,1] x, meta, y
def MEM_SIZE           : Nat := 0x4740

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
def loadFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)

  cudaInit cuda ptr 0x10
  let ctxPtr ← load64 (← absAddr ptr 0x10)

  let n         ← load64 dataPtr
  let numBlocks ← ushrImm (← iaddImm n 255) 8   -- (n + 255) >> 8
  storeI64 n         (← absAddr ptr 0x38)
  storeI64 numBlocks (← absAddr ptr 0x40)

  let xBytes       ← ishlImm n 2          -- n*4
  let partialsBytes← ishlImm numBlocks 3  -- num_blocks*8
  let eight        ← iconst64 8

  -- buf order: 0=x, 1=y, 2=meta, 3=partials, 4=params
  let _ ← call cuda.fnCreateBuffer [ctxPtr, xBytes]
  let _ ← call cuda.fnCreateBuffer [ctxPtr, xBytes]
  let metaBuf ← call cuda.fnCreateBuffer [ctxPtr, eight]
  let _ ← call cuda.fnCreateBuffer [ctxPtr, partialsBytes]
  let _ ← call cuda.fnCreateBuffer [ctxPtr, eight]

  -- Pack [n:u32, num_blocks:u32] as i64 LE into staging slot at 0x48
  let n32   ← ireduce32 n
  let nb32  ← ireduce32 numBlocks
  let n64   ← uextend64 n32
  let nb64  ← uextend64 nb32
  let packed← bor n64 (← ishlImm nb64 32)
  let metaSlot ← absAddr ptr 0x48
  storeI64 packed metaSlot

  -- Upload packed meta to buf2
  let _ ← call cuda.fnUpload [ctxPtr, metaBuf, metaSlot, eight]
  ret

-- Prep: upload x from data_ptr to buf0
def prepFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let ctxPtr  ← load64 (← absAddr ptr 0x10)
  let xBuf    ← iconst32 0
  let _ ← call cuda.fnUpload [ctxPtr, xBuf, dataPtr, dataLen]
  ret

-- Core: launch kernels (small path or 3-kernel path)
def coreFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let n      ← load64 (← absAddr ptr 0x38)
  let numBlocks ← load64 (← absAddr ptr 0x40)
  let nb32   ← ireduce32 numBlocks
  let one32  ← iconst32 1
  let three32← iconst32 3
  let four32 ← iconst32 4
  let blk256 ← iconst32 256

  let smallPath ← declareBlock []
  let largePath ← declareBlock []

  brif (← icmp .ule n (← iconst64 2048)) smallPath.ref [] largePath.ref []

  -- Small path: single kernel
  startBlock smallPath
  let _ ← cudaLaunchNamed cuda ptr (← iconst64 PTX_SOURCE_OFF) (← iconst64 NAME_SMALL_SOFTMAX)
             three32 (← iconst64 BIND_SMALL_OFF) one32 one32 one32 blk256 one32 one32
  ret

  -- Large path: block_reduce → global_reduce → normalize
  startBlock largePath
  let _ ← cudaLaunchNamed cuda ptr (← iconst64 PTX_SOURCE_OFF) (← iconst64 NAME_BLOCK_REDUCE)
             three32 (← iconst64 BIND_K1_OFF) nb32 one32 one32 blk256 one32 one32
  let _ ← cudaLaunchNamed cuda ptr (← iconst64 PTX_SOURCE_OFF) (← iconst64 NAME_GLOBAL_REDUCE)
             three32 (← iconst64 BIND_K2_OFF) one32 one32 one32 blk256 one32 one32
  let _ ← cudaLaunchNamed cuda ptr (← iconst64 PTX_SOURCE_OFF) (← iconst64 NAME_NORMALIZE)
             four32 (← iconst64 BIND_K3_OFF) nb32 one32 one32 blk256 one32 one32
  ret

-- Finalize: sync, optional download y from buf1
def finalizeFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outLen ← load64 (← absAddr ptr 0x30)
  let ctxPtr ← load64 (← absAddr ptr 0x10)

  let skipDl     ← declareBlock []
  let doDownload ← declareBlock []

  let _ ← cudaSync cuda ptr 0x10
  brif (← icmpImm .eq outLen 0) skipDl.ref [] doDownload.ref []

  startBlock doDownload
  let yBuf ← iconst32 1
  let _ ← call cuda.fnDownload [ctxPtr, yBuf, outPtr, outLen]
  ret

  startBlock skipDl
  ret

/-- Stack depth baked into the `stackAlgorithm` wrapper. -/
def STACK_DEPTH : Nat := 64

def clifIR : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 loadFn ++ "\n" ++
  buildFunction 2 prepFn ++ "\n" ++
  buildFunction 3 coreFn ++ "\n" ++
  buildFunction 4 finalizeFn ++
  clifSequenceWrapper 5 [3, 4] ++                                       -- infer
  clifSequenceWrapper 6 (List.replicate STACK_DEPTH 3 ++ [4])           -- stack

-- initial_memory: names, PTX source, bind descriptors
def nameBlockReduce  : List UInt8 := "block_reduce".toUTF8.toList ++ [0]
def nameGlobalReduce : List UInt8 := "global_reduce".toUTF8.toList ++ [0]
def nameNormalize    : List UInt8 := "normalize".toUTF8.toList ++ [0]
def nameSmallSoftmax : List UInt8 := "small_softmax".toUTF8.toList ++ [0]
def ptxBytes         : List UInt8 := ptxSource.toUTF8.toList ++ [0]

-- bind descriptors: buf IDs as i32 (4 bytes each, little-endian)
def i32LE (n : UInt32) : List UInt8 :=
  [n &&& 0xff, (n >>> 8) &&& 0xff, (n >>> 16) &&& 0xff, (n >>> 24) &&& 0xff].map (·.toUInt8)

def bindK1 : List UInt8 := i32LE 0 ++ i32LE 2 ++ i32LE 3  -- x, meta, partials
def bindK2 : List UInt8 := i32LE 2 ++ i32LE 3 ++ i32LE 4  -- meta, partials, params
def bindK3 : List UInt8 := i32LE 0 ++ i32LE 2 ++ i32LE 4 ++ i32LE 1  -- x, meta, params, y
def bindSmall : List UInt8 := i32LE 0 ++ i32LE 2 ++ i32LE 1  -- x, meta, y

def buildInitialMemory : List UInt8 :=
  let names :=
    zeros (NAME_BLOCK_REDUCE) ++
    nameBlockReduce  ++ zeros (NAME_GLOBAL_REDUCE - NAME_BLOCK_REDUCE  - nameBlockReduce.length) ++
    nameGlobalReduce ++ zeros (NAME_NORMALIZE     - NAME_GLOBAL_REDUCE - nameGlobalReduce.length) ++
    nameNormalize    ++ zeros (NAME_SMALL_SOFTMAX - NAME_NORMALIZE     - nameNormalize.length) ++
    nameSmallSoftmax ++ zeros (PTX_SOURCE_OFF      - NAME_SMALL_SOFTMAX - nameSmallSoftmax.length)
  let ptx  := ptxBytes ++ zeros (BIND_K1_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind :=
    bindK1 ++ zeros (BIND_K2_OFF - BIND_K1_OFF - bindK1.length) ++
    bindK2 ++ zeros (BIND_K3_OFF - BIND_K2_OFF - bindK2.length) ++
    bindK3 ++ zeros (BIND_SMALL_OFF - BIND_K3_OFF - bindK3.length) ++
    bindSmall ++ zeros (MEM_SIZE - BIND_SMALL_OFF - bindSmall.length)
  names ++ ptx ++ bind

def buildSetup : Setup := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  initial_memory := buildInitialMemory
}

def loadAlgorithm  : Algorithm := { fn_idx := u32 1 }
def prepAlgorithm  : Algorithm := { fn_idx := u32 2 }
def inferAlgorithm : Algorithm := { fn_idx := u32 5 }
def stackAlgorithm : Algorithm := { fn_idx := u32 6 }

def artifacts : Array Json :=
  #[
    toJsonArtifact "cuda_softmax" buildSetup loadAlgorithm [
      ("prep",  prepAlgorithm),
      ("infer", inferAlgorithm),
      ("stack", stackAlgorithm)
    ]
  ]

end CudaSoftmaxPersist
