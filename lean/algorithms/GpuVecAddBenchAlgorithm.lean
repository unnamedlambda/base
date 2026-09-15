import Lean
import AlgorithmLib.Gen
import LayoutScan

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.WGSL

namespace GpuVecAddBench

/-
  GPU VecAdd: C[i] = A[i] + B[i], data = [A floats][B floats], output = [C floats]
-/

def WGSL_SHADER_OFF : Nat := 0x0100
def BIND_DESC_OFF   : Nat := 0x1100
def MEM_SIZE        : Nat := 0x1200

def wgslShader : String :=
  let data : AlgorithmLib.WGSL.Expr (.arr .f32) := ⟨"data"⟩
  buildShader
    [{ binding := 0, name := "data", ty := .arr .f32 }]
    [] [] {}
    do
      let n ← letV (wArrayLen data / litU 2)
      let i ← letV gidX
      ifB (i .>= n) retV
      assign (arrIdx data i) (arrIdx data i + arrIdx data (n + i))

open AlgorithmLib.Prog


abbrev fnInit : Ffi := .gpuInit
abbrev fnCreateBuffer : Ffi := .gpuCreateBuffer
abbrev fnCreatePipeline : Ffi := .gpuCreatePipeline
abbrev fnUploadPtr : Ffi := .gpuUploadPtr
abbrev fnDispatch : Ffi := .gpuDispatch
abbrev fnDownloadPtr : Ffi := .gpuDownloadPtr
abbrev fnCleanup : Ffi := .gpuCleanup

def code : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let outPtr  ← outPtr


  let ctxSlotPtr ← absAddr ptr 8   -- ContextSlots.wgpu
  ffiVoid fnInit %[ctxSlotPtr]
  let ctxPtr ← load64 ctxSlotPtr

  -- n = data_len / 8, workgroups = (n+63)/64
  let n   ← ushrImm dataLen 3
  let wg  ← ireduce32 (← ushrImm (← iaddImm n 63) 6)
  let one ← iconst32 1

  let bufId ← ffi fnCreateBuffer %[ctxPtr, dataLen]
  let _ ← ffi fnUploadPtr %[ctxPtr, bufId, dataPtr, dataLen]

  let shaderAddr ← absAddr ptr WGSL_SHADER_OFF
  let bindAddr   ← absAddr ptr BIND_DESC_OFF
  let pipeId ← ffi fnCreatePipeline %[ctxPtr, shaderAddr, bindAddr, one]
  let _ ← ffi fnDispatch %[ctxPtr, pipeId, wg, one, one]

  let nBytes ← ishlImm n 2
  let bufOff ← iconst64 0
  let _ ← ffi fnDownloadPtr %[ctxPtr, bufId, bufOff, outPtr, nBytes]

  ffiVoid fnCleanup %[ctxSlotPtr]

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

def wgslBytes : List UInt8 :=
  wgslShader.toUTF8.toList ++ [0]

def bindDesc : List UInt8 :=
  [0, 0, 0, 0, 0, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let shader := wgslBytes ++ zeros (BIND_DESC_OFF - WGSL_SHADER_OFF - wgslBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ shader ++ bind

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the runtime fills and `0x18`-`0x38` the
    input and output descriptors, so naming those is what stops an offset being
    placed where the runtime will overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"ctx_wgpu",   ContextSlots.wgpu, 8⟩,
   ⟨"shader",     WGSL_SHADER_OFF, BIND_DESC_OFF - WGSL_SHADER_OFF⟩,
   ⟨"bind",       BIND_DESC_OFF, MEM_SIZE - BIND_DESC_OFF⟩]


#eval LayoutScan.check "GpuVecAddBenchAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "gpu_vecadd_algorithm" {
    functions := clif,
    memory_size := MEM_SIZE,
    initial_memory := buildInitialMemory
  }]

end GpuVecAddBench
