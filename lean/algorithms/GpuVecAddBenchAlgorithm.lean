import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace GpuVecAddBench

/-
  GPU VecAdd: C[i] = A[i] + B[i], data = [A floats][B floats], output = [C floats]
-/

def WGSL_SHADER_OFF : Nat := 0x0100
def BIND_DESC_OFF   : Nat := 0x1100
def MEM_SIZE        : Nat := 0x1200
def TIMEOUT_MS      : Nat := 120000

def wgslShader : String :=
  "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n" ++
  "\n" ++
  "@compute @workgroup_size(64)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  "    let n = arrayLength(&data) / 2u;\n" ++
  "    let i = gid.x;\n" ++
  "    if (i >= n) { return; }\n" ++
  "    data[i] = data[i] + data[n + i];\n" ++
  "}\n"

def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)

  let fnInit          ← declareFFI "cl_gpu_init"            [.i64]                    none
  let fnCreateBuffer  ← declareFFI "cl_gpu_create_buffer"   [.i64, .i64]             (some .i32)
  let fnCreatePipeline← declareFFI "cl_gpu_create_pipeline" [.i64, .i64, .i64, .i32] (some .i32)
  let fnUploadPtr     ← declareFFI "cl_gpu_upload_ptr"      [.i64, .i32, .i64, .i64] (some .i32)
  let fnDispatch      ← declareFFI "cl_gpu_dispatch"        [.i64, .i32, .i32, .i32, .i32] (some .i32)
  let fnDownloadPtr   ← declareFFI "cl_gpu_download_ptr"    [.i64, .i32, .i64, .i64, .i64] (some .i32)
  let fnCleanup       ← declareFFI "cl_gpu_cleanup"         [.i64]                    none

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
  ret

def clifIR : String := buildProgram mainFn

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
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0,
    initial_memory := buildInitialMemory
  } {
    actions := mkCallActions 1,
    cranelift_units := 0,
    timeout_ms := some TIMEOUT_MS
  }]

end GpuVecAddBench
