import Lean
import Std
import AlgorithmLib.Gen
import AlgorithmLib.HProgCuda

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

-- App fields: stored starting at 0x38 (beyond the 56-byte IoOffsets)
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
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- Two callee tables: `load`/`prep`/`finalize` reach CUDA alone, `core` also
    reaches cuBLAS, and a function declares only what it calls. -/


def env : FnEnv := env% [.cuda, .cublas]
def envCuda : FnEnv := env% [.cuda, .cublas]

/-- The CUDA context pointer lives at a fixed slot in shared memory. -/
def CTX_OFF : Nat := 0x10

def loadCode : HProg.Code := clif% do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)

  cudaInit ptr CTX_OFF
  let ctxPtr ← load64 (← absAddr ptr CTX_OFF)

  let seqLen    ← load64 dataPtr
  let dMBytes   ← iconst64 D_MODEL_BYTES
  let nHeads64  ← iconst64 N_HEADS
  let kvBytes   ← imul seqLen dMBytes              -- seq_len * D_MODEL * 4
  let scoreBytes← ishlImm (← imul seqLen nHeads64) 2  -- seq_len * N_HEADS * 4
  let eight     ← iconst64 8

  -- buf order: 0=q, 1=K, 2=V, 3=scores, 4=probs, 5=out, 6=meta
  let bufQ      ← call IR.Ffi.cudaCreateBuffer.id [ctxPtr, dMBytes]
  let bufK      ← call IR.Ffi.cudaCreateBuffer.id [ctxPtr, kvBytes]
  let bufV      ← call IR.Ffi.cudaCreateBuffer.id [ctxPtr, kvBytes]
  let bufScores ← call IR.Ffi.cudaCreateBuffer.id [ctxPtr, scoreBytes]
  let bufProbs  ← call IR.Ffi.cudaCreateBuffer.id [ctxPtr, scoreBytes]
  let bufOut    ← call IR.Ffi.cudaCreateBuffer.id [ctxPtr, dMBytes]
  let bufMeta   ← call IR.Ffi.cudaCreateBuffer.id [ctxPtr, eight]

  store bufQ      (← absAddr ptr BUF_Q_OFF)
  store bufK      (← absAddr ptr BUF_K_OFF)
  store bufV      (← absAddr ptr BUF_V_OFF)
  store bufScores (← absAddr ptr BUF_SCORES_OFF)
  store bufProbs  (← absAddr ptr BUF_PROBS_OFF)
  store bufOut    (← absAddr ptr BUF_OUT_OFF)
  store bufMeta   (← absAddr ptr BUF_META_OFF)

  -- Pack [seq_len:u32][0:u32] at SEQ_LEN_OFF, upload to meta buf
  let seqLen32 ← ireduce32 seqLen
  let seqLen64 ← uextend64 seqLen32
  let metaSlot ← absAddr ptr SEQ_LEN_OFF
  store seqLen64 metaSlot
  let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, bufMeta, metaSlot, eight]

  -- Upload K and V from data (K at data+8, V at data+8+kvBytes)
  let kSrc ← iaddImm dataPtr 8
  let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, bufK, kSrc, kvBytes]
  let vSrc ← iadd kSrc kvBytes
  let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, bufV, vSrc, kvBytes]

/-- Prep: upload q from data_ptr to buf0. -/
def prepCode : HProg.Code := clif% do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let ctxPtr  ← load64 (← absAddr ptr CTX_OFF)
  let bufQ    ← load32 (← absAddr ptr BUF_Q_OFF)
  let dMBytes ← iconst64 D_MODEL_BYTES
  let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, bufQ, dataPtr, dMBytes]

/-- **The softmax kernel, as a record its launch site reads.**

    Scores and probs are one row per head, as long as the sequence, which is
    a runtime value; `meta` carries the two `u32`s that say how long.  The
    bind table and the launch both take their length from `params`. -/
def softmaxK : AlgorithmLib.Kernel := {
  name   := "softmax"
  params := [{ shape := [.sta N_HEADS, .dyn], ro := true,  name := "scores" },
             { shape := [.sta 2],             ro := true,  name := "meta" },
             { shape := [.sta N_HEADS, .dyn], ro := false, name := "probs" }]
  geom   := AlgorithmLib.Kernel.Geom.static N_HEADS 1 1 256 1 1
  ptxOff := PTX_SOURCE_OFF
}

/-- Core: K@q scores, softmax, V^T@probs output. -/
def coreCode : HProg.Code := clif% do
  let ptr       := basePtr
  let ctxPtr    ← load64 (← absAddr ptr CTX_OFF)
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
  -- No operand offsets, and `0` asks the wrapper for the default leading
  -- dimension. Six arguments the signature has carried since it gained
  -- `off_a/b/c` and `ld_a/b/c`; these call sites never grew them.
  let zero64    ← iconst64 0

  let bufQ      ← load32 (← absAddr ptr BUF_Q_OFF)
  let bufK      ← load32 (← absAddr ptr BUF_K_OFF)
  let bufV      ← load32 (← absAddr ptr BUF_V_OFF)
  let bufScores ← load32 (← absAddr ptr BUF_SCORES_OFF)
  let bufProbs  ← load32 (← absAddr ptr BUF_PROBS_OFF)
  let bufOut    ← load32 (← absAddr ptr BUF_OUT_OFF)
  let bufMeta   ← load32 (← absAddr ptr BUF_META_OFF)

  -- scores = 0.125 * K @ q, batched over N_HEADS heads
  let _ ← call IR.Ffi.cublasSgemm.id
    [ctxPtr, one32, zero32, seqLen32, one32, headDim32, alpha0125,
     bufK, seqHead, bufQ, headDim64, zero32, bufScores, seqLen, nHeads32,
     zero64, zero64, zero64, zero32, zero32, zero32]

  kernelLaunchAt softmaxK ptr BIND_DESC_OFF [bufScores, bufMeta, bufProbs]

  -- out = V^T @ probs, batched over N_HEADS heads
  let _ ← call IR.Ffi.cublasSgemm.id
    [ctxPtr, zero32, zero32, headDim32, one32, seqLen32, alpha1f,
     bufV, seqHead, bufProbs, seqLen, zero32, bufOut, headDim64, nHeads32,
     zero64, zero64, zero64, zero32, zero32, zero32]

/-- Finalize: sync, then download only if the caller asked for output. -/
def finalizeCode : HProg.Code := clif% do
  let ptr    := basePtr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outLen ← load64 (← absAddr ptr 0x30)
  let ctxPtr ← load64 (← absAddr ptr CTX_OFF)
  let bufOut ← load32 (← absAddr ptr BUF_OUT_OFF)

  let _ ← cudaSync ptr CTX_OFF
  let _ ← ifte .eq outLen (← iconst64 0)
    (thn := pure [])
    (els := do
      let _ ← call IR.Ffi.cudaDownload.id [ctxPtr, bufOut, outPtr, outLen]
      pure [])
  return ()

theorem bodies_wf :
    HProg.wf envCuda HProg.ptrParams loadCode = true &&
    HProg.wf envCuda HProg.ptrParams prepCode = true &&
    HProg.wf env HProg.ptrParams coreCode = true &&
    HProg.wf envCuda HProg.ptrParams finalizeCode = true := by decide

def STACK_DEPTH : Nat := 64

def clifIR : Program :=
  program
    [noopFunction,
     HProg.compileFn 1 loadCode,
     HProg.compileFn 2 prepCode,
     HProg.compileFn 3 coreCode,
     HProg.compileFn 4 finalizeCode,
     clifSequenceWrapper 5 [3, 4],
     clifSequenceWrapper 6 (List.replicate STACK_DEPTH 3 ++ [4])]

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [3, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0]

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the runtime fills and `0x18`-`0x38` the
    input and output descriptors, so naming those is what stops an offset being
    placed where the runtime will overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"ctx_ht",     ContextSlots.ht, 8⟩,
   ⟨"ctx_wgpu",   ContextSlots.wgpu, 8⟩,
   ⟨"ctx_cuda",   CTX_OFF, 8⟩,
   ⟨"io_offsets", 0x18, 0x20⟩,
   ⟨"buf_q",      BUF_Q_OFF, 4⟩,
   ⟨"buf_k",      BUF_K_OFF, 4⟩,
   ⟨"buf_v",      BUF_V_OFF, 4⟩,
   ⟨"buf_scores", BUF_SCORES_OFF, 4⟩,
   ⟨"buf_probs",  BUF_PROBS_OFF, 4⟩,
   ⟨"buf_out",    BUF_OUT_OFF, 4⟩,
   ⟨"buf_meta",   BUF_META_OFF, 4⟩,
   ⟨"seq_len",    SEQ_LEN_OFF, 8⟩,
   ⟨"ptx",        PTX_SOURCE_OFF, BIND_DESC_OFF - PTX_SOURCE_OFF⟩,
   -- Three buffers, which is the arity the launch declares.
   ⟨"bind_desc",  BIND_DESC_OFF, 12⟩]

theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

/-- **The PTX and the bind descriptor fit the slots they are placed in.**

    `buildInitialMemory` pads with `zeros (next - this - length)`, on `Nat`: a
    text longer than its slot pads by zero rather than by a negative amount, so
    the overrun is written over whatever follows and every offset after it
    shifts. The failure shows up as a device launching garbage. -/
theorem regions_fit :
    ptxBytes.length ≤ BIND_DESC_OFF - PTX_SOURCE_OFF
      ∧ bindDesc.length ≤ MEM_SIZE - BIND_DESC_OFF := by native_decide

def buildInitialMemory : List UInt8 :=
  let reserved := zeros PTX_SOURCE_OFF
  let ptx := ptxBytes ++ zeros (BIND_DESC_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ ptx ++ bind

def buildSetup : Setup := {
  clif := clifIR,
  memory_size := MEM_SIZE,
  initial_memory := buildInitialMemory
}

def loadAlgorithm : Algorithm := { fn_idx := u32 1 }
def prepAlgorithm : Algorithm := { fn_idx := u32 2 }
def inferAlgorithm : Algorithm := { fn_idx := u32 5 }
def stackAlgorithm : Algorithm := { fn_idx := u32 6 }

def artifacts : Array Json :=
  #[
    toJsonArtifact "cuda_decode_attn" buildSetup loadAlgorithm [
      ("prep",  prepAlgorithm),
      ("infer", inferAlgorithm),
      ("stack", stackAlgorithm)
    ]
  ]

end CudaDecodeAttention
