import Lean
import Std
import AlgorithmLib

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

  Algorithms:
    u0:1  load      — alloc resident buffers, upload K/V once
    u0:2  prep      — upload q
    u0:3  core      — scores + softmax + V mix, no sync/download
    u0:4  finalize  — sync and optional download
-/

def N_HEADS : Nat := 14
def HEAD_DIM : Nat := 64
def D_MODEL : Nat := N_HEADS * HEAD_DIM
def MAX_SEQ : Nat := 2048

def D_MODEL_BYTES : Nat := D_MODEL * 4

def PTX_SOURCE_OFF : Nat := 0x0200
def BIND_DESC_OFF  : Nat := 0x5000
def MEM_SIZE       : Nat := 0x5100
def TIMEOUT_MS     : Nat := 30000

-- App fields: stored starting at 0x38 (beyond the 56-byte RuntimeHeader)
def BUF_Q_OFF      : Nat := 0x38
def BUF_K_OFF      : Nat := 0x3C
def BUF_V_OFF      : Nat := 0x40
def BUF_SCORES_OFF : Nat := 0x44
def BUF_PROBS_OFF  : Nat := 0x48
def BUF_OUT_OFF    : Nat := 0x4C
def BUF_META_OFF   : Nat := 0x50
-- seq_len (i64) staging + seq_len value stored at 0x58
def SEQ_LEN_OFF    : Nat := 0x58  -- i64 seq_len value (also used as staging for GPU upload)

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
def loadFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)

  cudaInit cuda ptr 0x10
  let ctxPtr ← load64 (← absAddr ptr 0x10)

  let seqLen    ← load64 dataPtr
  let dMBytes   ← iconst64 D_MODEL_BYTES
  let nHeads64  ← iconst64 N_HEADS
  let kvBytes   ← imul seqLen dMBytes              -- seq_len * D_MODEL * 4
  let scoreBytes← ishlImm (← imul seqLen nHeads64) 2  -- seq_len * N_HEADS * 4
  let eight     ← iconst64 8

  -- buf order: 0=q, 1=K, 2=V, 3=scores, 4=probs, 5=out, 6=meta
  let bufQ      ← call cuda.fnCreateBuffer [ctxPtr, dMBytes]
  let bufK      ← call cuda.fnCreateBuffer [ctxPtr, kvBytes]
  let bufV      ← call cuda.fnCreateBuffer [ctxPtr, kvBytes]
  let bufScores ← call cuda.fnCreateBuffer [ctxPtr, scoreBytes]
  let bufProbs  ← call cuda.fnCreateBuffer [ctxPtr, scoreBytes]
  let bufOut    ← call cuda.fnCreateBuffer [ctxPtr, dMBytes]
  let bufMeta   ← call cuda.fnCreateBuffer [ctxPtr, eight]

  storeI32 bufQ      (← absAddr ptr BUF_Q_OFF)
  storeI32 bufK      (← absAddr ptr BUF_K_OFF)
  storeI32 bufV      (← absAddr ptr BUF_V_OFF)
  storeI32 bufScores (← absAddr ptr BUF_SCORES_OFF)
  storeI32 bufProbs  (← absAddr ptr BUF_PROBS_OFF)
  storeI32 bufOut    (← absAddr ptr BUF_OUT_OFF)
  storeI32 bufMeta   (← absAddr ptr BUF_META_OFF)

  -- Pack [seq_len:u32][0:u32] at SEQ_LEN_OFF, upload to meta buf
  let seqLen32 ← ireduce32 seqLen
  let seqLen64 ← uextend64 seqLen32
  let metaSlot ← absAddr ptr SEQ_LEN_OFF
  storeI64 seqLen64 metaSlot
  let _ ← call cuda.fnUpload [ctxPtr, bufMeta, metaSlot, eight]

  -- Upload K and V from data (K at data+8, V at data+8+kvBytes)
  let kSrc ← iaddImm dataPtr 8
  let _ ← call cuda.fnUpload [ctxPtr, bufK, kSrc, kvBytes]
  let vSrc ← iadd kSrc kvBytes
  let _ ← call cuda.fnUpload [ctxPtr, bufV, vSrc, kvBytes]
  ret

-- Prep: upload q from data_ptr to buf0
def prepFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let ctxPtr  ← load64 (← absAddr ptr 0x10)
  let bufQ    ← load32 (← absAddr ptr BUF_Q_OFF)
  let dMBytes ← iconst64 D_MODEL_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufQ, dataPtr, dMBytes]
  ret

-- Core: K@q scores, softmax, V^T@probs output
def coreFn : IRBuilder Unit := do
  let ptr       ← entryBlock
  let cuda      ← declareCudaFFI
  let blas      ← declareCuBlasFFI
  let ctxPtr    ← load64 (← absAddr ptr 0x10)
  let seqLen    ← load64 (← absAddr ptr SEQ_LEN_OFF)
  let seqLen32  ← ireduce32 seqLen
  let headDim64 ← iconst64 HEAD_DIM
  let headDim32 ← iconst32 HEAD_DIM
  let nHeads32  ← iconst32 N_HEADS
  let seqHead   ← imul seqLen headDim64  -- stride: seq_len * HEAD_DIM
  let one32     ← iconst32 1
  let zero32    ← iconst32 0
  let alpha0125 ← iconst32 0x3e000000   -- 0.125f
  let alpha1f   ← iconst32 0x3f800000   -- 1.0f
  let blk256    ← iconst32 256
  let three32   ← iconst32 3

  let bufQ      ← load32 (← absAddr ptr BUF_Q_OFF)
  let bufK      ← load32 (← absAddr ptr BUF_K_OFF)
  let bufV      ← load32 (← absAddr ptr BUF_V_OFF)
  let bufScores ← load32 (← absAddr ptr BUF_SCORES_OFF)
  let bufProbs  ← load32 (← absAddr ptr BUF_PROBS_OFF)
  let bufOut    ← load32 (← absAddr ptr BUF_OUT_OFF)
  let bufMeta   ← load32 (← absAddr ptr BUF_META_OFF)

  -- scores = 0.125 * K @ q, batched over N_HEADS heads
  -- sgemm(ctx, transa=1, transb=0, m=seq_len, n=1, k=head_dim, alpha=0.125,
  --       a=K, stride_a=seq*head, b=q, stride_b=head_dim, beta=0,
  --       c=scores, stride_c=seq_len, batch=N_HEADS)
  let _ ← call blas.fnSgemm
    [ctxPtr, one32, zero32, seqLen32, one32, headDim32, alpha0125,
     bufK, seqHead, bufQ, headDim64, zero32, bufScores, seqLen, nHeads32]

  -- Write bind descriptor for softmax: [bufScores, bufMeta, bufProbs]
  storeI32 bufScores (← absAddr ptr BIND_DESC_OFF)
  storeI32 bufMeta   (← absAddr ptr (BIND_DESC_OFF + 4))
  storeI32 bufProbs  (← absAddr ptr (BIND_DESC_OFF + 8))

  -- Launch softmax kernel: gridDim=(N_HEADS,1,1), blockDim=(256,1,1), 3 bufs
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_SOURCE_OFF) three32
             (← iconst64 BIND_DESC_OFF) nHeads32 one32 one32 blk256 one32 one32

  -- out = V^T @ probs, batched over N_HEADS heads
  -- sgemm(ctx, transa=0, transb=0, m=head_dim, n=1, k=seq_len, alpha=1.0,
  --       a=V, stride_a=seq*head, b=probs, stride_b=seq_len, beta=0,
  --       c=out, stride_c=head_dim, batch=N_HEADS)
  let _ ← call blas.fnSgemm
    [ctxPtr, zero32, zero32, headDim32, one32, seqLen32, alpha1f,
     bufV, seqHead, bufProbs, seqLen, zero32, bufOut, headDim64, nHeads32]
  ret

-- Finalize: sync, optional download out
def finalizeFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outLen ← load64 (← absAddr ptr 0x30)
  let ctxPtr ← load64 (← absAddr ptr 0x10)
  let bufOut ← load32 (← absAddr ptr BUF_OUT_OFF)

  let skipDl     ← declareBlock []
  let doDownload ← declareBlock []

  let _ ← cudaSync cuda ptr 0x10
  brif (← icmpImm .eq outLen 0) skipDl.ref [] doDownload.ref []

  startBlock doDownload
  let _ ← call cuda.fnDownload [ctxPtr, bufOut, outPtr, outLen]
  ret

  startBlock skipDl
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 loadFn ++ "\n" ++
  buildFunction 2 prepFn ++ "\n" ++
  buildFunction 3 coreFn ++ "\n" ++
  buildFunction 4 finalizeFn

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [3, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros PTX_SOURCE_OFF
  let ptx := ptxBytes ++ zeros (BIND_DESC_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ ptx ++ bind

def actions (src : UInt32) : List Action :=
  [{ kind := .ClifCall, dst := 0, src := src, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0,
  initial_memory := buildInitialMemory
}

def loadAlgorithm : Algorithm := { actions := actions 1, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }
def prepAlgorithm : Algorithm := { actions := actions 2, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }
def inferAlgorithm : Algorithm := {
  actions := [{ kind := .ClifCall, dst := 0, src := 3, offset := 0, size := 0 },
              { kind := .ClifCall, dst := 0, src := 4, offset := 0, size := 0 }],
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

def stackAlgorithm (depth : Nat) : Algorithm := {
  actions := List.replicate depth { kind := .ClifCall, dst := 0, src := 3, offset := 0, size := 0 } ++
             [{ kind := .ClifCall, dst := 0, src := 4, offset := 0, size := 0 }],
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

def artifacts : Array Json :=
  #[
    toJsonEntry "cuda_decode_attn_load" buildConfig loadAlgorithm,
    toJsonEntry "cuda_decode_attn_prep" buildConfig prepAlgorithm,
    toJsonEntry "cuda_decode_attn_infer" buildConfig inferAlgorithm,
    toJsonEntry "cuda_decode_attn_stack" buildConfig (stackAlgorithm 64),
  ]

end CudaDecodeAttention
