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
def env : FnEnv := env% [.gpu]

def fnInit : Nat := IR.FFI.std.gpu.fnInit.id
def fnCreateBuffer : Nat := IR.FFI.std.gpu.fnCreateBuffer.id
def fnCreatePipeline : Nat := IR.FFI.std.gpu.fnCreatePipeline.id
def fnUploadPtr : Nat := IR.FFI.std.gpu.fnUploadPtr.id
def fnDispatch : Nat := IR.FFI.std.gpu.fnDispatch.id
def fnDownloadPtr : Nat := IR.FFI.std.gpu.fnDownloadPtr.id
def fnCleanup : Nat := IR.FFI.std.gpu.fnCleanup.id

def code : HProg.Code := clif% do
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

def artifacts : Array Json :=
  #[toJsonEntry "gpu_vecadd_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE,
    initial_memory := buildInitialMemory
  } {
    fn_idx := u32 1
  }]

end GpuVecAddBench
