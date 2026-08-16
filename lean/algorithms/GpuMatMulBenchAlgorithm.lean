import Lean
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.WGSL

namespace GpuMatMulBench

/-
  GPU MatMul: C = A*B (square N×N), data=[A floats: N*N][B floats: N*N], output=[C floats: N*N]
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
      let total ← letV (wArrayLen data)
      let nn    ← letV    (total / litU 3)
      let bigN  ← letV     (u32OfF (wSqrt (f32OfU nn)))
      let idx   ← letV   gidX
      ifB (idx .>= nn) retV
      let ci    ← letV     (idx / bigN)
      let cj    ← letV     (idx % bigN)
      let sum   ← varV   (litF "0.0")
      forU "k" (litU 0) (fun k => ltE k bigN) (fun k => k + litU 1) fun k => do
        assign sum (sum + arrIdx data (ci * bigN + k) * arrIdx data (nn + k * bigN + cj))
      assign (arrIdx data (litU 2 * nn + idx)) sum

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

  -- nn = data_len / 8, buffer_size = nn * 12 (holds A+B+C), workgroups
  let nn  ← ushrImm dataLen 3
  let c12 ← iconst64 12
  let bufSize ← imul nn c12
  let wg  ← ireduce32 (← ushrImm (← iaddImm nn 63) 6)
  let one ← iconst32 1

  let bufId ← call fnCreateBuffer [ctxPtr, bufSize]
  let _ ← call fnUploadPtr [ctxPtr, bufId, dataPtr, dataLen]

  let shaderAddr ← absAddr ptr WGSL_SHADER_OFF
  let bindAddr   ← absAddr ptr BIND_DESC_OFF
  let pipeId ← call fnCreatePipeline [ctxPtr, shaderAddr, bindAddr, one]
  let _ ← call fnDispatch [ctxPtr, pipeId, wg, one, one]

  -- Download C: nn*4 bytes from GPU buf offset 2*nn*4
  let nnBytes ← ishlImm nn 2      -- nn * 4 (download size)
  let bufOff  ← ishlImm nn 3      -- 2*nn*4 (buf offset)
  let _ ← call fnDownloadPtr [ctxPtr, bufId, bufOff, outPtr, nnBytes]

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
  #[toJsonEntry "gpu_matmul_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE,
    initial_memory := buildInitialMemory
  } {
    fn_idx := u32 1
  }]

end GpuMatMulBench
