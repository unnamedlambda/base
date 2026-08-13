import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.PTX

namespace CudaSaxpyBench

/-
  CUDA SAXPY: y[i] = 2.0 * x[i] + y[i]
  Payload: [x floats: N][y floats: N], Output: [y result floats: N]
-/

def PTX_SOURCE_OFF : Nat := 0x0100
def BIND_DESC_OFF  : Nat := 0x1100
def MEM_SIZE       : Nat := 0x1200

def ptxSource : String := buildModuleWith { version := "7.0", target := "sm_50" } [{
  name := "main", params := ["x_ptr", "y_ptr"], body := do
  let xPtr ← ldParam "x_ptr"
  let yPtr ← ldParam "y_ptr"
  let bid ← freshR; movR bid ctaX
  let tid ← freshR; movR tid tidX
  let gid ← freshR; madLoRC gid bid 256 tid
  let off ← freshRd; cvtU64 off gid; shlRd off off 2
  let xa ← freshRd; addRd xa xPtr off
  let ya ← freshRd; addRd ya yPtr off
  let fx ← freshF; ldGlobalF fx xa
  let fy ← freshF; ldGlobalF fy ya
  let fa ← freshF; movFC fa 0x40000000  -- 2.0f
  fmaRn fy fa fx fy
  stGlobalF ya fy
  ptxRet }]

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- The CUDA entry points, in the order the callee table numbers them. -/
def env : FnEnv := env% [.cuda]

def fnInit : Nat := IR.FFI.std.cuda.fnInit.id
def fnCreateBuffer : Nat := IR.FFI.std.cuda.fnCreateBuffer.id
def fnUploadPtr : Nat := IR.FFI.std.cuda.fnUpload.id
def fnDownloadPtr : Nat := IR.FFI.std.cuda.fnDownload.id
def fnLaunch : Nat := IR.FFI.std.cuda.fnLaunch.id
def fnCleanup : Nat := IR.FFI.std.cuda.fnCleanup.id

def code : HProg.Code := clif% do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)


  let ctxSlotPtr ← absAddr ptr 0x10   -- ContextSlots.cuda
  callVoid fnInit [ctxSlotPtr]
  let ctxPtr ← load64 ctxSlotPtr

  -- buf_size = data_len / 2 (each of x and y is half)
  let bufSize ← ushrImm dataLen 1
  let xBufId  ← call fnCreateBuffer [ctxPtr, bufSize]
  let yBufId  ← call fnCreateBuffer [ctxPtr, bufSize]

  -- Upload x from data_ptr, y from data_ptr + buf_size
  let _ ← call fnUploadPtr [ctxPtr, xBufId, dataPtr, bufSize]
  let yDataPtr ← iadd dataPtr bufSize
  let _ ← call fnUploadPtr [ctxPtr, yBufId, yDataPtr, bufSize]

  -- Grid: ceil(N / 256) where N = buf_size / 4
  let bigN  ← ireduce32 (← ushrImm bufSize 2)
  let c255  ← iconst32 255
  let nPlus ← iadd bigN c255
  let c256  ← iconst32 256
  let gridX ← udiv nPlus c256
  let one   ← iconst32 1

  let ptxAddr  ← absAddr ptr PTX_SOURCE_OFF
  let two      ← iconst32 2
  let bindAddr ← absAddr ptr BIND_DESC_OFF
  let _ ← call fnLaunch [ctxPtr, ptxAddr, two, bindAddr,
                          gridX, one, one, c256, one, one]

  let _ ← call fnDownloadPtr [ctxPtr, yBufId, outPtr, bufSize]

  callVoid fnCleanup [ctxSlotPtr]


theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 code]

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [0, 0, 0, 0, 1, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let ptx := ptxBytes ++ zeros (BIND_DESC_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ ptx ++ bind

def artifacts : Array Json :=
  #[toJsonEntry "cuda_saxpy_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE,
    initial_memory := buildInitialMemory
  } {
    fn_idx := u32 1
  }]

end CudaSaxpyBench
