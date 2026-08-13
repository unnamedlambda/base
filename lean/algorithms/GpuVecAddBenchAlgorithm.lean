import Lean
import AlgorithmLib

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

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- The GPU entry points, in the order the callee table numbers them. -/
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

def code : HProg.Code := clif% env HProg.ptrParams do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)


  let ctxSlotPtr ← absAddr ptr 8   -- ContextSlots.wgpu
  callVoid fnInit [ctxSlotPtr]
  let ctxPtr ← load64 ctxSlotPtr

  -- n = data_len / 8, workgroups = (n+63)/64
  let n   ← ushrImm dataLen 3
  let wg  ← ireduce32 (← ushrImm (← iaddImm n 63) 6)
  let one ← iconst32 1

  let bufId ← call fnCreateBuffer [ctxPtr, dataLen]
  let _ ← call fnUploadPtr [ctxPtr, bufId, dataPtr, dataLen]

  let shaderAddr ← absAddr ptr WGSL_SHADER_OFF
  let bindAddr   ← absAddr ptr BIND_DESC_OFF
  let pipeId ← call fnCreatePipeline [ctxPtr, shaderAddr, bindAddr, one]
  let _ ← call fnDispatch [ctxPtr, pipeId, wg, one, one]

  let nBytes ← ishlImm n 2
  let bufOff ← iconst64 0
  let _ ← call fnDownloadPtr [ctxPtr, bufId, bufOff, outPtr, nBytes]

  callVoid fnCleanup [ctxSlotPtr]


theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 env HProg.ptrParams code]

def wgslBytes : List UInt8 :=
  wgslShader.toUTF8.toList ++ [0]

def bindDesc : List UInt8 :=
  [0, 0, 0, 0, 0, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let shader := wgslBytes ++ zeros (BIND_DESC_OFF - WGSL_SHADER_OFF - wgslBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ shader ++ bind

def artifacts : Array Json :=
  #[toJsonEntry "gpu_vecadd_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE,
    initial_memory := buildInitialMemory
  } {
    fn_idx := u32 1
  }]

end GpuVecAddBench
