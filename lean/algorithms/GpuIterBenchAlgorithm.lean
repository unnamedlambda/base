import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

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
def TIMEOUT_MS       : Nat := 300000

def scaleShader : String :=
  "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n" ++
  "\n" ++
  "@compute @workgroup_size(64)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  "    let n = arrayLength(&data);\n" ++
  "    let i = gid.x;\n" ++
  "    if (i >= n) { return; }\n" ++
  "    data[i] = data[i] * 1.001;\n" ++
  "}\n"

def reduceShader : String :=
  "@group(0) @binding(0) var<storage, read>       data: array<f32>;\n" ++
  "@group(0) @binding(1) var<storage, read_write> sums: array<f32>;\n" ++
  "\n" ++
  "var<workgroup> partial: array<f32, 64>;\n" ++
  "\n" ++
  "@compute @workgroup_size(64)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n" ++
  "        @builtin(local_invocation_id)   lid: vec3<u32>,\n" ++
  "        @builtin(workgroup_id)          wgid: vec3<u32>) {\n" ++
  "    let n = arrayLength(&data);\n" ++
  "    partial[lid.x] = select(0.0, data[gid.x], gid.x < n);\n" ++
  "    workgroupBarrier();\n" ++
  "    var s = 32u;\n" ++
  "    while s > 0u {\n" ++
  "        if lid.x < s { partial[lid.x] += partial[lid.x + s]; }\n" ++
  "        workgroupBarrier();\n" ++
  "        s >>= 1u;\n" ++
  "    }\n" ++
  "    if lid.x == 0u { sums[wgid.x] = partial[0]; }\n" ++
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

  -- Read passes from payload start
  let passes    ← load64 dataPtr
  let floatBytes← iaddImm dataLen (-8)
  let floatPtr  ← iaddImm dataPtr 8

  let ctxSlotPtr ← absAddr ptr 8
  callVoid fnInit [ctxSlotPtr]
  let ctxPtr ← load64 ctxSlotPtr

  let dataBufId ← call fnCreateBuffer [ctxPtr, floatBytes]
  let sumsBufSize← ushrImm floatBytes 6   -- floatBytes / 64
  let sumsBufId  ← call fnCreateBuffer [ctxPtr, sumsBufSize]
  let _ ← call fnUploadPtr [ctxPtr, dataBufId, floatPtr, floatBytes]

  -- Scale pipeline (1 binding)
  let scaleShaderAddr ← absAddr ptr SCALE_SHADER_OFF
  let scaleBindAddr   ← absAddr ptr SCALE_BIND_OFF
  let one ← iconst32 1
  let scalePipeId ← call fnCreatePipeline [ctxPtr, scaleShaderAddr, scaleBindAddr, one]

  -- Reduce pipeline (2 bindings)
  let reduceShaderAddr ← absAddr ptr REDUCE_SHADER_OFF
  let reduceBindAddr   ← absAddr ptr REDUCE_BIND_OFF
  let two ← iconst32 2
  let reducePipeId ← call fnCreatePipeline [ctxPtr, reduceShaderAddr, reduceBindAddr, two]

  -- workgroups = (floatBytes/4 + 63) / 64 = (floatBytes + 252) / 256
  let wg ← ireduce32 (← ushrImm (← iaddImm floatBytes 252) 8)

  -- Loop: dispatch scale `passes` times
  let loop ← declareBlock [.i64]
  let body ← declareBlock [.i64]
  let after← declareBlock []

  let i0 ← iconst64 0
  jump loop.ref [i0]

  startBlock loop
  let i := loop.param 0
  brif (← icmp .uge i passes) after.ref [] body.ref [i]

  startBlock body
  let i2 := body.param 0
  let _ ← call fnDispatch [ctxPtr, scalePipeId, wg, one, one]
  jump loop.ref [← iaddImm i2 1]

  startBlock after
  let _ ← call fnDispatch [ctxPtr, reducePipeId, wg, one, one]
  let bufOff ← iconst64 0
  let _ ← call fnDownloadPtr [ctxPtr, sumsBufId, bufOff, outPtr, sumsBufSize]

  callVoid fnCleanup [ctxSlotPtr]
  ret

def clifIR : String := buildProgram mainFn

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
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0,
    initial_memory := buildInitialMemory
  } {
    actions := mkCallActions 1,
    cranelift_units := 0,
    timeout_ms := some TIMEOUT_MS
  }]

end GpuIterBench
