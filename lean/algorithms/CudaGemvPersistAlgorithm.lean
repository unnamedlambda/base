import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace CudaGemvPersist

/-
  Persistent GEMV: A stays on GPU, only x is uploaded per call.

  fn1  load   — init CUDA, alloc 3 bufs (A, x, y), upload A, store m/n
  fn2  prep   — upload x to buf1
  fn3  infer  — cuBLAS SGEMV, sync, optional download y from buf2

  Shared memory app fields (after 56-byte runtime header):
    0x38  m (i64)
    0x40  n (i64)

  Buffer IDs are sequential: buf0=A, buf1=x, buf2=y (hardcoded in infer/prep).

  Data formats:
    load  data: [m: u64][n: u64][A: m*n f32]
    prep  data: [x: n f32]  (data_len = n*4)
    infer out:  [y: m f32]  (optional; compute-only if out_len=0)
-/

def MEM_SIZE   : Nat := 0x50
def M_OFF      : Nat := 0x38
def N_OFF      : Nat := 0x40

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- The CUDA and cuBLAS entry points, declared through the same helpers the
    runtime's signatures come from.

    Two callee tables, because a function declares only what it calls: `load`
    and `prep` reach CUDA alone, and carrying cuBLAS in their tables would put
    six signatures in the emitted function that nothing there uses. -/
def ffiEnv : (IR.CudaSetup × IR.CuBlasSetup) × FnEnv := (Id.run (do
  let cuda := IR.FFI.std.cuda
  let blas := IR.FFI.std.cublas
  pure (cuda, blas)), env% [.cuda, .cublas])

def cudaEnvOnly : IR.CudaSetup × FnEnv := (IR.FFI.std.cuda, env% [.cuda, .cublas])

def cuda : IR.CudaSetup := ffiEnv.1.1
def blas : IR.CuBlasSetup := ffiEnv.1.2
def env : FnEnv := ffiEnv.2
def envCuda : FnEnv := cudaEnvOnly.2

/-- The CUDA context pointer lives at a fixed slot in shared memory. -/
def CTX_OFF : Nat := 0x10

def loadCode : HProg.Code := clif% do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)

  cudaInit cuda ptr CTX_OFF
  let ctxPtr ← load64 (← absAddr ptr CTX_OFF)

  let m  ← load64 dataPtr
  let n  ← load64 (← iaddImm dataPtr 8)
  store m (← absAddr ptr M_OFF)
  store n (← absAddr ptr N_OFF)

  let mNBytes ← ishlImm (← imul m n) 2
  let nBytes  ← ishlImm n 2
  let mBytes  ← ishlImm m 2

  let buf0 ← call cuda.fnCreateBuffer.id [ctxPtr, mNBytes]  -- A (buf id 0)
  let _    ← call cuda.fnCreateBuffer.id [ctxPtr, nBytes]   -- x (buf id 1)
  let _    ← call cuda.fnCreateBuffer.id [ctxPtr, mBytes]   -- y (buf id 2)

  -- Upload A from data[16..] (after m, n header)
  let aPtr ← iaddImm dataPtr 16
  let _ ← call cuda.fnUpload.id [ctxPtr, buf0, aPtr, mNBytes]

def prepCode : HProg.Code := clif% do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let ctxPtr  ← load64 (← absAddr ptr CTX_OFF)
  let xBuf    ← iconst32 1
  let _ ← call cuda.fnUpload.id [ctxPtr, xBuf, dataPtr, dataLen]

/-- The download branch joins rather than returning from each arm: `Code` has no
    early return, so both arms reach one `ret`. -/
def inferCode : HProg.Code := clif% do
  let ptr := basePtr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outLen ← load64 (← absAddr ptr 0x30)
  let ctxPtr ← load64 (← absAddr ptr CTX_OFF)
  let m      ← load64 (← absAddr ptr M_OFF)
  let n      ← load64 (← absAddr ptr N_OFF)
  let m32    ← ireduce32 m
  let n32    ← ireduce32 n

  let alpha  ← iconst32 0x3f800000  -- 1.0f
  let zero32 ← iconst32 0
  let one32  ← iconst32 1
  let two32  ← iconst32 2
  -- sgemv(ctx, trans=1, m=n, n=m, alpha=1.0, a_buf=0, x_buf=1, beta=0, y_buf=2)
  let _ ← call blas.fnSgemv.id [ctxPtr, one32, n32, m32, alpha, zero32, one32, zero32, two32]
  let _ ← cudaSync cuda ptr CTX_OFF
  let _ ← ifte .eq outLen (← iconst64 0)
    (thn := pure [])
    (els := do
      let _ ← call cuda.fnDownload.id [ctxPtr, two32, outPtr, outLen]
      pure [])
  return ()

theorem bodies_wf :
    HProg.wf envCuda HProg.ptrParams loadCode = true &&
    HProg.wf envCuda HProg.ptrParams prepCode = true &&
    HProg.wf env HProg.ptrParams inferCode = true := by decide

def clifIR : Program :=
  IR.program
    [noopFunction,
     HProg.compileFn 1 loadCode,
     HProg.compileFn 2 prepCode,
     HProg.compileFn 3 inferCode]


def buildSetup : Setup := {
  clif := clifIR,
  memory_size := MEM_SIZE
}

def loadAlgorithm : Algorithm := { fn_idx := u32 1 }
def prepAlgorithm : Algorithm := { fn_idx := u32 2 }
def inferAlgorithm : Algorithm := { fn_idx := u32 3 }

def artifacts : Array Json :=
  #[
    toJsonArtifact "cuda_gemv" buildSetup loadAlgorithm [
      ("prep",  prepAlgorithm),
      ("infer", inferAlgorithm)
    ]
  ]

end CudaGemvPersist
