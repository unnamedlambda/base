import Lean
import AlgorithmLib.Gen
import LayoutScan

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


  let ctxSlotPtr ← absAddr ptr 8
  ffiVoid fnInit %[ctxSlotPtr]
  let ctxPtr ← load64 ctxSlotPtr

  -- nn = data_len / 8, buffer_size = nn * 12 (holds A+B+C), workgroups
  let nn  ← ushrImm dataLen 3
  let c12 ← iconst64 12
  let bufSize ← imul nn c12
  let wg  ← ireduce32 (← ushrImm (← iaddImm nn 63) 6)
  let one ← iconst32 1

  let bufId ← ffi fnCreateBuffer %[ctxPtr, bufSize]
  let _ ← ffi fnUploadPtr %[ctxPtr, bufId, dataPtr, dataLen]

  let shaderAddr ← absAddr ptr WGSL_SHADER_OFF
  let bindAddr   ← absAddr ptr BIND_DESC_OFF
  let pipeId ← ffi fnCreatePipeline %[ctxPtr, shaderAddr, bindAddr, one]
  let _ ← ffi fnDispatch %[ctxPtr, pipeId, wg, one, one]

  -- Download C: nn*4 bytes from GPU buf offset 2*nn*4
  let nnBytes ← ishlImm nn 2      -- nn * 4 (download size)
  let bufOff  ← ishlImm nn 3      -- 2*nn*4 (buf offset)
  let _ ← ffi fnDownloadPtr %[ctxPtr, bufId, bufOff, outPtr, nnBytes]

  ffiVoid fnCleanup %[ctxSlotPtr]

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

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
    `0x00`-`0x18` are the context slots the FFI's init calls fill, so naming
    those is what stops an offset being placed where one of those calls will
    overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"ctx_wgpu",   ContextSlots.wgpu, 8⟩,
   ⟨"shader",     WGSL_SHADER_OFF, BIND_DESC_OFF - WGSL_SHADER_OFF⟩,
   ⟨"bind",       BIND_DESC_OFF, MEM_SIZE - BIND_DESC_OFF⟩]


#eval LayoutScan.check "GpuMatMulBenchAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "gpu_matmul_algorithm" {
    functions := clif,
    required_memory := MEM_SIZE,
    initial_memory := buildInitialMemory
  }]

end GpuMatMulBench
