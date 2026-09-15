import Lean
import Std
import AlgorithmLib.Gen
import LayoutScan

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

open AlgorithmLib.Prog


/-- The CUDA context pointer lives at a fixed slot in shared memory. -/
def CTX_OFF : Nat := 0x10

def loadCode : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr

  cudaInit ptr CTX_OFF
  let ctxPtr ← load64 (← absAddr ptr CTX_OFF)

  let m  ← load64 dataPtr
  let n  ← load64 (← iaddImm dataPtr 8)
  store m (← absAddr ptr M_OFF)
  store n (← absAddr ptr N_OFF)

  let mNBytes ← ishlImm (← imul m n) 2
  let nBytes  ← ishlImm n 2
  let mBytes  ← ishlImm m 2

  let buf0 ← ffi .cudaCreateBuffer %[ctxPtr, mNBytes]  -- A (buf id 0)
  let _    ← ffi .cudaCreateBuffer %[ctxPtr, nBytes]   -- x (buf id 1)
  let _    ← ffi .cudaCreateBuffer %[ctxPtr, mBytes]   -- y (buf id 2)

  -- Upload A from data[16..] (after m, n header)
  let aPtr ← iaddImm dataPtr 16
  let _ ← ffi .cudaUpload %[ctxPtr, buf0, aPtr, mNBytes]

def prepCode : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let ctxPtr  ← load64 (← absAddr ptr CTX_OFF)
  let xBuf    ← iconst32 1
  let _ ← ffi .cudaUpload %[ctxPtr, xBuf, dataPtr, dataLen]

/-- The download branch joins rather than returning from each arm: `Code` has no
    early return, so both arms reach one `ret`. -/
def inferCode : Prog V L Unit := do
  let ptr ← basePtr
  let outPtr ← outPtr
  let outLen ← outLen
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
  let _ ← ffi .cublasSgemv %[ctxPtr, one32, n32, m32, alpha, zero32, one32, zero32, two32]
  let _ ← cudaSync ptr CTX_OFF
  let _ ← ifte .eq outLen (← iconst64 0)
    (thn := pure %[])
    (els := do
      let _ ← ffi .cudaDownload %[ctxPtr, two32, outPtr, outLen]
      pure %[])
  return ()


def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.compileProg 1 loadCode,
     Prog.compileProg 2 prepCode,
     Prog.compileProg 3 inferCode]


/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the runtime fills and `0x18`-`0x38` the
    input and output descriptors, so naming those is what stops an offset being
    placed where the runtime will overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"ctx_cuda",   CTX_OFF, 8⟩,
   ⟨"m",          M_OFF, 8⟩,
   ⟨"n",          N_OFF, 8⟩]


#eval LayoutScan.check "CudaGemvPersistAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def buildSetup (clif : List FuncData) : Artifact := {
  functions := clif,
  memory_size := MEM_SIZE
}

def loadAlgorithm : UInt32 := 1
def prepAlgorithm : UInt32 := 2
def inferAlgorithm : UInt32 := 3

def artifacts (clif : List FuncData) : Array Json :=
  #[
    toJsonArtifact "cuda_gemv" (buildSetup clif)
  ]

end CudaGemvPersist
