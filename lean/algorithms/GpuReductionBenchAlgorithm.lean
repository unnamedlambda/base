import Lean
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.WGSL

namespace GpuReductionBench

/-
  GPU Reduction: partial sum of N f32 values into N/64 f32 partial sums.
  data=[f32 values: N], output=[f32 partial sums: N/64]
-/

def WGSL_SHADER_OFF : Nat := 0x0100
def BIND_DESC_OFF   : Nat := 0x1100
def MEM_SIZE        : Nat := 0x1200

def wgslShader : String :=
  let data  : AlgorithmLib.WGSL.Expr (.arr .f32)     := ⟨"data"⟩
  let sdata : AlgorithmLib.WGSL.Expr (.arrN .f32 64) := ⟨"sdata"⟩
  buildShader
    [{ binding := 0, name := "data", ty := .arr .f32 }]
    [("sdata", .f32, 64)] []
    { lid := true, wid := true }
    do
      let total   ← letV    (wArrayLen data)
      let numGrps ← letV (total / litU 65)
      let inputN  ← letV  (numGrps * litU 64)
      ifElse (gidX .< inputN)
        (assign (arrIdxN sdata lidX) (arrIdx data gidX))
        (assign (arrIdxN sdata lidX) (litF "0.0"))
      wBarrier
      forU "s" (litU 32) (fun s => gtE s (litU 0)) (fun s => shrU s (litU 1)) fun s => do
        ifB (lidX .< s) do
          assign (arrIdxN sdata lidX) (arrIdxN sdata lidX + arrIdxN sdata (lidX + s))
        wBarrier
      ifB (lidX .== litU 0) do
        assign (arrIdx data (inputN + widX)) (arrIdxN sdata (litU 0))

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- The GPU entry points, in the order the callee table numbers them. -/
def env : FnEnv := env% [.gpu]

def fnInit : Nat := IR.Ffi.gpuInit.id
def fnCreateBuffer : Nat := IR.Ffi.gpuCreateBuffer.id
def fnCreatePipeline : Nat := IR.Ffi.gpuCreatePipeline.id
def fnUploadPtr : Nat := IR.Ffi.gpuUploadPtr.id
def fnDispatch : Nat := IR.Ffi.gpuDispatch.id
def fnDownloadPtr : Nat := IR.Ffi.gpuDownloadPtr.id
def fnCleanup : Nat := IR.Ffi.gpuCleanup.id

def code : HProg.Code := clif% do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)


  let ctxSlotPtr ← absAddr ptr 8
  callVoid fnInit [ctxSlotPtr]
  let ctxPtr ← load64 ctxSlotPtr

  -- N = data_len/4, num_groups = N/64, buf_size = (N + num_groups) * 4
  let bigN       ← ushrImm dataLen 2
  let numGroups  ← ushrImm bigN 6
  let bufSize    ← ishlImm (← iadd bigN numGroups) 2
  let wg         ← ireduce32 numGroups
  let one        ← iconst32 1

  let bufId ← call fnCreateBuffer [ctxPtr, bufSize]
  let _ ← call fnUploadPtr [ctxPtr, bufId, dataPtr, dataLen]

  let shaderAddr ← absAddr ptr WGSL_SHADER_OFF
  let bindAddr   ← absAddr ptr BIND_DESC_OFF
  let pipeId ← call fnCreatePipeline [ctxPtr, shaderAddr, bindAddr, one]
  let _ ← call fnDispatch [ctxPtr, pipeId, wg, one, one]

  -- Download partial sums: num_groups*4 bytes from GPU buf offset N*4 = data_len
  let dlSize ← ishlImm numGroups 2
  let _ ← call fnDownloadPtr [ctxPtr, bufId, dataLen, outPtr, dlSize]

  callVoid fnCleanup [ctxSlotPtr]


theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 code]

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
   ⟨"io_offsets", 0x18, 0x20⟩,
   ⟨"shader",     WGSL_SHADER_OFF, BIND_DESC_OFF - WGSL_SHADER_OFF⟩,
   ⟨"bind",       BIND_DESC_OFF, MEM_SIZE - BIND_DESC_OFF⟩]

theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts : Array Json :=
  #[toJsonEntry "gpu_reduction_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE,
    initial_memory := buildInitialMemory
  } {
    fn_idx := u32 1
  }]

end GpuReductionBench
