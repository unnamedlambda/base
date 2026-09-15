import Lean
import AlgorithmLib.Gen
import LayoutScan

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

open AlgorithmLib.Prog


abbrev fnInit : Ffi := .cudaInit
abbrev fnCreateBuffer : Ffi := .cudaCreateBuffer
abbrev fnUploadPtr : Ffi := .cudaUpload
abbrev fnDownloadPtr : Ffi := .cudaDownload
abbrev fnLaunch : Ffi := .cudaLaunch
abbrev fnCleanup : Ffi := .cudaCleanup

def code : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let outPtr  ← outPtr


  let ctxSlotPtr ← absAddr ptr 0x10   -- ContextSlots.cuda
  ffiVoid fnInit %[ctxSlotPtr]
  let ctxPtr ← load64 ctxSlotPtr

  -- buf_size = data_len / 2 (each of x and y is half)
  let bufSize ← ushrImm dataLen 1
  let xBufId  ← ffi fnCreateBuffer %[ctxPtr, bufSize]
  let yBufId  ← ffi fnCreateBuffer %[ctxPtr, bufSize]

  -- Upload x from data_ptr, y from data_ptr + buf_size
  let _ ← ffi fnUploadPtr %[ctxPtr, xBufId, dataPtr, bufSize]
  let yDataPtr ← iadd dataPtr bufSize
  let _ ← ffi fnUploadPtr %[ctxPtr, yBufId, yDataPtr, bufSize]

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
  let _ ← ffi fnLaunch %[ctxPtr, ptxAddr, two, bindAddr,
                          gridX, one, one, c256, one, one]

  let _ ← ffi fnDownloadPtr %[ctxPtr, yBufId, outPtr, bufSize]

  ffiVoid fnCleanup %[ctxSlotPtr]

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [0, 0, 0, 0, 1, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let ptx := ptxBytes ++ zeros (BIND_DESC_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ ptx ++ bind

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the runtime fills and `0x18`-`0x38` the
    input and output descriptors, so naming those is what stops an offset being
    placed where the runtime will overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"ctx_cuda",   ContextSlots.cuda, 8⟩,
   ⟨"ptx",        PTX_SOURCE_OFF, BIND_DESC_OFF - PTX_SOURCE_OFF⟩,
   ⟨"bind",       BIND_DESC_OFF, MEM_SIZE - BIND_DESC_OFF⟩]


#eval LayoutScan.check "CudaSaxpyBenchAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "cuda_saxpy_algorithm" {
    functions := clif,
    memory_size := MEM_SIZE,
    initial_memory := buildInitialMemory
  }]

end CudaSaxpyBench
