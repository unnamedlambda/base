import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace GpuReductionBench

/-
  GPU Reduction: partial sum of N f32 values into N/64 f32 partial sums.
  data=[f32 values: N], output=[f32 partial sums: N/64]
-/

def WGSL_SHADER_OFF : Nat := 0x0100
def BIND_DESC_OFF   : Nat := 0x1100
def MEM_SIZE        : Nat := 0x1200
def TIMEOUT_MS      : Nat := 120000

def wgslShader : String :=
  "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n" ++
  "\n" ++
  "var<workgroup> sdata: array<f32, 64>;\n" ++
  "\n" ++
  "@compute @workgroup_size(64)\n" ++
  "fn main(\n" ++
  "    @builtin(global_invocation_id) gid: vec3<u32>,\n" ++
  "    @builtin(local_invocation_id) lid: vec3<u32>,\n" ++
  "    @builtin(workgroup_id) wid: vec3<u32>\n" ++
  ") {\n" ++
  "    let total = arrayLength(&data);\n" ++
  "    let num_groups = total / 65u;\n" ++
  "    let input_n = num_groups * 64u;\n" ++
  "    if (gid.x < input_n) {\n" ++
  "        sdata[lid.x] = data[gid.x];\n" ++
  "    } else {\n" ++
  "        sdata[lid.x] = 0.0;\n" ++
  "    }\n" ++
  "    workgroupBarrier();\n" ++
  "\n" ++
  "    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {\n" ++
  "        if (lid.x < s) {\n" ++
  "            sdata[lid.x] = sdata[lid.x] + sdata[lid.x + s];\n" ++
  "        }\n" ++
  "        workgroupBarrier();\n" ++
  "    }\n" ++
  "\n" ++
  "    if (lid.x == 0u) {\n" ++
  "        data[input_n + wid.x] = sdata[0];\n" ++
  "    }\n" ++
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
  #[toJsonEntry "gpu_reduction_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0,
    initial_memory := buildInitialMemory
  } {
    actions := mkCallActions 1,
    cranelift_units := 0,
    timeout_ms := some TIMEOUT_MS
  }]

end GpuReductionBench
