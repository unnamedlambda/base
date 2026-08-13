import Lean
import AlgorithmLib

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

/-- The GPU entry points, declared through the same helpers every generator
    uses, so the term's callee table cannot drift from their signatures. -/
def env : FnEnv := (envOf (do
  let _ ← declareFFI "cl_gpu_init"            [.i64]                         none
  let _ ← declareFFI "cl_gpu_create_buffer"   [.i64, .i64]                   (some .i32)
  let _ ← declareFFI "cl_gpu_create_pipeline" [.i64, .i64, .i64, .i32]       (some .i32)
  let _ ← declareFFI "cl_gpu_upload_ptr"      [.i64, .i32, .i64, .i64]       (some .i32)
  let _ ← declareFFI "cl_gpu_dispatch"        [.i64, .i32, .i32, .i32, .i32] (some .i32)
  let _ ← declareFFI "cl_gpu_download_ptr"    [.i64, .i32, .i64, .i64, .i64] (some .i32)
  let _ ← declareFFI "cl_gpu_cleanup"         [.i64]                         none)).2

def fnInit : Nat := 0
def fnCreateBuffer : Nat := 1
def fnCreatePipeline : Nat := 2
def fnUploadPtr : Nat := 3
def fnDispatch : Nat := 4
def fnDownloadPtr : Nat := 5
def fnCleanup : Nat := 6

open HProg.Sur in
def code : HProg.Code := clif% env HProg.ptrParams do
  let dataPtr ← load64 (← absAddr basePtr 0x18)
  let dataLen ← load64 (← absAddr basePtr 0x20)
  let outPtr  ← load64 (← absAddr basePtr 0x28)

  -- Read passes from payload start
  let passes    ← load64 dataPtr
  let floatBytes← iaddImm dataLen (-8)
  let floatPtr  ← iaddImm dataPtr 8

  let ctxSlotPtr ← absAddr basePtr 8
  callVoid fnInit [ctxSlotPtr]
  let ctxPtr ← load64 ctxSlotPtr

  let dataBufId ← call fnCreateBuffer [ctxPtr, floatBytes]
  let sumsBufSize← ushrImm floatBytes 6   -- floatBytes / 64
  let sumsBufId  ← call fnCreateBuffer [ctxPtr, sumsBufSize]
  let _ ← call fnUploadPtr [ctxPtr, dataBufId, floatPtr, floatBytes]

  -- Scale pipeline (1 binding)
  let scaleShaderAddr ← absAddr basePtr SCALE_SHADER_OFF
  let scaleBindAddr   ← absAddr basePtr SCALE_BIND_OFF
  let one ← iconst32 1
  let scalePipeId ← call fnCreatePipeline [ctxPtr, scaleShaderAddr, scaleBindAddr, one]

  -- Reduce pipeline (2 bindings)
  let reduceShaderAddr ← absAddr basePtr REDUCE_SHADER_OFF
  let reduceBindAddr   ← absAddr basePtr REDUCE_BIND_OFF
  let two ← iconst32 2
  let reducePipeId ← call fnCreatePipeline [ctxPtr, reduceShaderAddr, reduceBindAddr, two]

  -- workgroups = (floatBytes/4 + 63) / 64 = (floatBytes + 252) / 256
  let wg ← ireduce32 (← ushrImm (← iaddImm floatBytes 252) 8)

  -- Dispatch scale `passes` times, then a final reduce
  let _ ← wloop1 (← iconst64 0)
    (head := fun i => return (contIfULt i passes, ([] : List R), ()))
    (body := fun i _ => do
      let _ ← call fnDispatch [ctxPtr, scalePipeId, wg, one, one]
      return [← iaddImm i 1])
  let _ ← call fnDispatch [ctxPtr, reducePipeId, wg, one, one]
  let bufOff ← iconst64 0
  let _ ← call fnDownloadPtr [ctxPtr, sumsBufId, bufOff, outPtr, sumsBufSize]

  callVoid fnCleanup [ctxSlotPtr]

theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 env HProg.ptrParams code]

/-- The FFI calls the *emitted* function performs, in order.

    Stated through `Clif.callsOf` — the same extractor the LZ4 and ML host
    proofs use — over `compileFn`'s output rather than a builder's state. This
    is the shape those proofs take after their generators move to terms: the
    claim is unchanged, only what produced the program is. -/
theorem emitted_calls :
    Clif.callsOf (HProg.compileFn 1 env HProg.ptrParams code).asState
      = ["cl_gpu_init", "cl_gpu_create_buffer", "cl_gpu_create_buffer",
         "cl_gpu_upload_ptr", "cl_gpu_create_pipeline", "cl_gpu_create_pipeline",
         "cl_gpu_dispatch", "cl_gpu_dispatch", "cl_gpu_download_ptr",
         "cl_gpu_cleanup"] := by native_decide

/-- No loop is *recovered* from the blocks, because `loopsOf` only recognises a
    trip count that folds to a constant and this one is read from the payload.
    Recorded because it is the limit that makes recovery-from-a-CFG the weaker
    route: the term says `Piece.loop` whatever the bound is. -/
theorem emitted_loops_not_static :
    Clif.loopsOf (HProg.compileFn 1 env HProg.ptrParams code).asState = [] := by
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

def artifacts : Array Json :=
  #[toJsonEntry "gpu_iter_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE,
    initial_memory := buildInitialMemory
  } {
    fn_idx := u32 1
  }]

end GpuIterBench
