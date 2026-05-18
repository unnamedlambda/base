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

def loadFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)

  cudaInit cuda ptr 0x10
  let ctxPtr ← load64 (← absAddr ptr 0x10)

  let m  ← load64 dataPtr
  let n  ← load64 (← iaddImm dataPtr 8)
  storeI64 m (← absAddr ptr M_OFF)
  storeI64 n (← absAddr ptr N_OFF)

  let mNBytes ← ishlImm (← imul m n) 2
  let nBytes  ← ishlImm n 2
  let mBytes  ← ishlImm m 2

  let buf0 ← call cuda.fnCreateBuffer [ctxPtr, mNBytes]  -- A (buf id 0)
  let buf1 ← call cuda.fnCreateBuffer [ctxPtr, nBytes]   -- x (buf id 1)
  let _    ← call cuda.fnCreateBuffer [ctxPtr, mBytes]   -- y (buf id 2)

  -- Upload A from data[16..] (after m, n header)
  let aPtr ← iaddImm dataPtr 16
  let _ ← call cuda.fnUpload [ctxPtr, buf0, aPtr, mNBytes]
  let _ := buf1  -- suppress unused warning; id is hardcoded in prepFn
  ret

def prepFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let ctxPtr  ← load64 (← absAddr ptr 0x10)
  let xBuf    ← iconst32 1
  let _ ← call cuda.fnUpload [ctxPtr, xBuf, dataPtr, dataLen]
  ret

def inferFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let blas   ← declareCuBlasFFI
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outLen ← load64 (← absAddr ptr 0x30)
  let ctxPtr ← load64 (← absAddr ptr 0x10)
  let m      ← load64 (← absAddr ptr M_OFF)
  let n      ← load64 (← absAddr ptr N_OFF)
  let m32    ← ireduce32 m
  let n32    ← ireduce32 n

  let skipDl     ← declareBlock []
  let doDownload ← declareBlock []

  let alpha  ← iconst32 0x3f800000  -- 1.0f
  let zero32 ← iconst32 0
  let one32  ← iconst32 1
  let two32  ← iconst32 2
  -- sgemv(ctx, trans=1, m=n, n=m, alpha=1.0, a_buf=0, x_buf=1, beta=0, y_buf=2)
  let _ ← call blas.fnSgemv [ctxPtr, one32, n32, m32, alpha, zero32, one32, zero32, two32]
  let _ ← cudaSync cuda ptr 0x10
  brif (← icmpImm .eq outLen 0) skipDl.ref [] doDownload.ref []

  startBlock doDownload
  let _ ← call cuda.fnDownload [ctxPtr, two32, outPtr, outLen]
  ret

  startBlock skipDl
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 loadFn ++ "\n" ++
  buildFunction 2 prepFn ++ "\n" ++
  buildFunction 3 inferFn

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0
}

def loadAlgorithm : Algorithm := { fn_idx := u32 1 }
def prepAlgorithm : Algorithm := { fn_idx := u32 2 }
def inferAlgorithm : Algorithm := { fn_idx := u32 3 }

def artifacts : Array Json :=
  #[
    toJsonArtifact "cuda_gemv" buildConfig loadAlgorithm [
      ("prep",  prepAlgorithm),
      ("infer", inferAlgorithm)
    ]
  ]

end CudaGemvPersist
