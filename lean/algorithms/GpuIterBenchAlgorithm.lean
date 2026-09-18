import Lean
import AlgorithmLib.Gen
import LayoutScan

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Clif
open AlgorithmLib.WGSL

namespace GpuIterBench


/-
  GPU Iterative: apply scale kernel (*1.001) N times, then reduce.
  Payload: [passes: i64][f32 values: M], Output: [f32 partial sums: M/64]
-/

def SCALE_SHADER_OFF : Nat := 0x0100
def REDUCE_SHADER_OFF: Nat := 0x0800
def SCALE_BIND_OFF   : Nat := 0x1100
def REDUCE_BIND_OFF  : Nat := 0x1108
def MEM_SIZE         : Nat := 0x1200

def scaleShader : String :=
  let data : AlgorithmLib.WGSL.Expr (.arr .f32) := ⟨"data"⟩
  buildShader
    [{ binding := 0, name := "data", ty := .arr .f32 }]
    [] [] {}
    do
      let n ← letV (wArrayLen data)
      let i ← letV gidX
      ifB (i .>= n) retV
      assign (arrIdx data i) (arrIdx data i * litF "1.001")

def reduceShader : String :=
  let data    : AlgorithmLib.WGSL.Expr (.arr .f32)     := ⟨"data"⟩
  let sums    : AlgorithmLib.WGSL.Expr (.arr .f32)     := ⟨"sums"⟩
  let partialArr : AlgorithmLib.WGSL.Expr (.arrN .f32 64) := ⟨"partial"⟩
  buildShader
    [{ binding := 0, name := "data",    ty := .arr .f32, ro := true },
     { binding := 1, name := "sums",    ty := .arr .f32 }]
    [("partial", .f32, 64)] []
    { lid := true, wid := true }
    do
      let n ← letV (wArrayLen data)
      assign (arrIdxN partialArr lidX) (wSelect (litF "0.0") (arrIdx data gidX) (gidX .< n))
      wBarrier
      let s ← varV (litU 32)
      whileB (gtE s (litU 0)) do
        ifB (lidX .< s) do
          assign (arrIdxN partialArr lidX) (arrIdxN partialArr lidX + arrIdxN partialArr (lidX + s))
        wBarrier
        assign s (shrU s (litU 1))
      ifB (lidX .== litU 0) do
        assign (arrIdx sums widX) (arrIdxN partialArr (litU 0))

abbrev fnInit : Ffi := .gpuInit
abbrev fnCreateBuffer : Ffi := .gpuCreateBuffer
abbrev fnCreatePipeline : Ffi := .gpuCreatePipeline
abbrev fnUploadPtr : Ffi := .gpuUploadPtr
abbrev fnDispatch : Ffi := .gpuDispatch
abbrev fnDownloadPtr : Ffi := .gpuDownloadPtr
abbrev fnCleanup : Ffi := .gpuCleanup

open AlgorithmLib.Prog in
def code : Prog V L Unit := do
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let outPtr  ← outPtr

  -- Read passes from payload start
  let passes    ← load64 dataPtr
  let floatBytes← iaddImm dataLen (-8)
  let floatPtr  ← iaddImm dataPtr 8

  let ctxSlotPtr ← absAddr (← basePtr) 8
  ffiVoid fnInit %[ctxSlotPtr]
  let ctxPtr ← load64 ctxSlotPtr

  let dataBufId ← ffi fnCreateBuffer %[ctxPtr, floatBytes]
  let sumsBufSize← ushrImm floatBytes 6   -- floatBytes / 64
  let sumsBufId  ← ffi fnCreateBuffer %[ctxPtr, sumsBufSize]
  let _ ← ffi fnUploadPtr %[ctxPtr, dataBufId, floatPtr, floatBytes]

  -- Scale pipeline (1 binding)
  let scaleShaderAddr ← absAddr (← basePtr) SCALE_SHADER_OFF
  let scaleBindAddr   ← absAddr (← basePtr) SCALE_BIND_OFF
  let one ← iconst32 1
  let scalePipeId ← ffi fnCreatePipeline %[ctxPtr, scaleShaderAddr, scaleBindAddr, one]

  -- Reduce pipeline (2 bindings)
  let reduceShaderAddr ← absAddr (← basePtr) REDUCE_SHADER_OFF
  let reduceBindAddr   ← absAddr (← basePtr) REDUCE_BIND_OFF
  let two ← iconst32 2
  let reducePipeId ← ffi fnCreatePipeline %[ctxPtr, reduceShaderAddr, reduceBindAddr, two]

  -- workgroups = (floatBytes/4 + 63) / 64 = (floatBytes + 252) / 256
  let wg ← ireduce32 (← ushrImm (← iaddImm floatBytes 252) 8)

  -- Dispatch scale `passes` times, then a final reduce
  let _ ← wloop1 (← iconst64 0)
    (head := fun i => return (contIfULt i passes, %[], ()))
    (body := fun i _ => do
      let _ ← ffi fnDispatch %[ctxPtr, scalePipeId, wg, one, one]
      return %[← iaddImm i 1])
  let _ ← ffi fnDispatch %[ctxPtr, reducePipeId, wg, one, one]
  let bufOff ← iconst64 0
  let _ ← ffi fnDownloadPtr %[ctxPtr, sumsBufId, bufOff, outPtr, sumsBufSize]

  ffiVoid fnCleanup %[ctxSlotPtr]


def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

/-- The FFI calls the *emitted* function performs, in order.

    Stated through `Clif.callsOf` — the same extractor the LZ4 and ML host
    proofs use — over the compiled body rather than the term it was written
    as. This
    is the shape those proofs take after their generators move to terms: the
    claim is unchanged, only what produced the program is. -/
theorem emitted_calls :
    Clif.callsOf (Prog.stateOf 1 code)
      = ["cl_gpu_init", "cl_gpu_create_buffer", "cl_gpu_create_buffer",
         "cl_gpu_upload_ptr", "cl_gpu_create_pipeline", "cl_gpu_create_pipeline",
         "cl_gpu_dispatch", "cl_gpu_dispatch", "cl_gpu_download_ptr",
         "cl_gpu_cleanup"] := by native_decide

/-- No loop is *recovered* from the blocks, because `loopsOf` only recognises a
    trip count that folds to a constant and this one is read from the payload.
    Recorded because it is the limit that makes recovery-from-a-CFG the weaker
    route: the term says `Piece.loop` whatever the bound is. -/
theorem emitted_loops_not_static :
    Clif.loopsOf (Prog.stateOf 1 code) = [] := by
  native_decide

def scaleShaderBytes  : List UInt8 := scaleShader.toUTF8.toList ++ [0]
def reduceShaderBytes : List UInt8 := reduceShader.toUTF8.toList ++ [0]
def scaleBindDesc  : List UInt8 := [0, 0, 0, 0, 0, 0, 0, 0]
def reduceBindDesc : List UInt8 := [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved    := zeros SCALE_SHADER_OFF
  let scale       := scaleShaderBytes ++ zeros (REDUCE_SHADER_OFF - SCALE_SHADER_OFF - scaleShaderBytes.length)
  let reduce      := reduceShaderBytes ++ zeros (SCALE_BIND_OFF - REDUCE_SHADER_OFF - reduceShaderBytes.length)
  let scaleBind   := scaleBindDesc
  let reduceBind  := reduceBindDesc ++ zeros (MEM_SIZE - REDUCE_BIND_OFF - reduceBindDesc.length)
  reserved ++ scale ++ reduce ++ scaleBind ++ reduceBind

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the FFI's init calls fill, so naming
    those is what stops an offset being placed where one of those calls will
    overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"ctx_wgpu",      ContextSlots.wgpu, 8⟩,
   ⟨"scale_shader",  SCALE_SHADER_OFF, REDUCE_SHADER_OFF - SCALE_SHADER_OFF⟩,
   ⟨"reduce_shader", REDUCE_SHADER_OFF, SCALE_BIND_OFF - REDUCE_SHADER_OFF⟩,
   ⟨"scale_bind",    SCALE_BIND_OFF, REDUCE_BIND_OFF - SCALE_BIND_OFF⟩,
   ⟨"reduce_bind",   REDUCE_BIND_OFF, MEM_SIZE - REDUCE_BIND_OFF⟩]


#eval LayoutScan.check "GpuIterBenchAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "gpu_iter_algorithm" {
    functions := clif,
    required_memory := MEM_SIZE,
    initial_memory := buildInitialMemory
  }]

end GpuIterBench
