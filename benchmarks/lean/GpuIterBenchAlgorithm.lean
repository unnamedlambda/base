import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace GpuIterBench

/-
  GPU Iterative benchmark algorithm — Cranelift JIT + wgpu version.

  Apply a scale kernel (*1.001) N times to GPU-resident data, then reduce.

  Payload (via execute data arg): [passes: i64][f32 values: M]
  Output (via execute_into out arg): [f32 partial sums: M/64]

  data_len includes the 8-byte passes prefix, so float_bytes = data_len - 8.

  Memory layout (shared memory):
    0x0000  RESERVED         (40 bytes, runtime-managed)
    0x0100  SCALE_SHADER     (null-terminated WGSL)
    0x0800  REDUCE_SHADER    (null-terminated WGSL)
    0x1100  SCALE_BIND_DESC  (8 bytes: [buf_id=0, read_only=0])
    0x1108  REDUCE_BIND_DESC (16 bytes: [buf_id=0, read_only=1], [buf_id=1, read_only=0])

  Single CLIF function:
    1. Read data_ptr, data_len, out_ptr from reserved region
    2. Read passes from data_ptr[0..8]
    3. cl_gpu_init
    4. float_bytes = data_len - 8
    5. Create data buffer (float_bytes) and sums buffer (float_bytes/64)
    6. Upload payload+8 to data buffer
    7. Create scale pipeline (1 binding) and reduce pipeline (2 bindings)
    8. Loop: dispatch scale kernel `passes` times
    9. Dispatch reduce kernel once
   10. Download sums buffer to out_ptr
   11. cl_gpu_cleanup
-/

def SCALE_SHADER_OFF : Nat := 0x0100
def REDUCE_SHADER_OFF: Nat := 0x0800
def SCALE_BIND_OFF   : Nat := 0x1100
def REDUCE_BIND_OFF  : Nat := 0x1108
def MEM_SIZE         : Nat := 0x1200
def TIMEOUT_MS       : Nat := 300000

-- Scale shader: multiply each f32 by 1.001, size-independent
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

-- Reduction shader: partial sum of groups of 64
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

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: GPU iterative scale + reduce orchestrator
-- Payload layout: [passes: i64 (8 bytes)][f32 data: float_bytes]
-- float_bytes = data_len - 8
def clifGpuFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++                              -- gpu_init / gpu_cleanup
  "    sig1 = (i64, i64) -> i32 system_v\n" ++                  -- gpu_create_buffer
  "    sig2 = (i64, i64, i64, i32) -> i32 system_v\n" ++        -- gpu_create_pipeline
  "    sig3 = (i64, i32, i64, i64) -> i32 system_v\n" ++        -- gpu_upload_ptr
  "    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v\n" ++   -- gpu_dispatch
  "    sig5 = (i64, i32, i64, i64, i64) -> i32 system_v\n" ++   -- gpu_download_ptr
  "\n" ++
  "    fn0 = %cl_gpu_init sig0\n" ++
  "    fn1 = %cl_gpu_create_buffer sig1\n" ++
  "    fn2 = %cl_gpu_create_pipeline sig2\n" ++
  "    fn3 = %cl_gpu_upload_ptr sig3\n" ++
  "    fn4 = %cl_gpu_dispatch sig4\n" ++
  "    fn5 = %cl_gpu_download_ptr sig5\n" ++
  "    fn6 = %cl_gpu_cleanup sig0\n" ++
  "\n" ++
  "block0(v0: i64):\n" ++
  "    v1 = load.i64 notrap aligned v0+0x08\n" ++               -- data_ptr
  "    v2 = load.i64 notrap aligned v0+0x10\n" ++               -- data_len (bytes, includes 8-byte prefix)
  "    v3 = load.i64 notrap aligned v0+0x18\n" ++               -- out_ptr
  -- Read passes from start of payload
  "    v4 = load.i64 notrap aligned v1\n" ++                    -- passes = *(i64*)data_ptr
  -- float_bytes = data_len - 8
  "    v5 = iadd_imm v2, -8\n" ++                               -- float_bytes
  -- float_ptr = data_ptr + 8
  "    v6 = iadd_imm v1, 8\n" ++                                -- float_ptr
  -- GPU init
  "    call fn0(v0)\n" ++
  -- Create data buffer (float_bytes) and sums buffer (float_bytes/64)
  "    v7 = call fn1(v0, v5)\n" ++                              -- data_buf
  "    v8 = ushr_imm v5, 6\n" ++                                -- sums_size = float_bytes / 64
  "    v9 = call fn1(v0, v8)\n" ++                              -- sums_buf
  -- Upload float data to data buffer
  "    v10 = call fn3(v0, v7, v6, v5)\n" ++
  -- Create scale pipeline (1 binding: buf0 rw)
  "    v11 = iconst.i64 256\n" ++                               -- SCALE_SHADER_OFF
  "    v12 = iconst.i64 4352\n" ++                              -- SCALE_BIND_OFF (0x1100)
  "    v13 = iconst.i32 1\n" ++
  "    v14 = call fn2(v0, v11, v12, v13)\n" ++                  -- scale_pipe
  -- Create reduce pipeline (2 bindings: buf0 ro, buf1 rw)
  "    v15 = iconst.i64 2048\n" ++                              -- REDUCE_SHADER_OFF (0x0800)
  "    v16 = iconst.i64 4360\n" ++                              -- REDUCE_BIND_OFF (0x1108)
  "    v17 = iconst.i32 2\n" ++
  "    v18 = call fn2(v0, v15, v16, v17)\n" ++                  -- reduce_pipe
  -- Compute workgroups: (float_bytes/4 + 63) / 64 = (float_bytes + 252) / 256
  "    v19 = iadd_imm v5, 252\n" ++
  "    v20 = ushr_imm v19, 8\n" ++
  "    v21 = ireduce.i32 v20\n" ++                              -- workgroups
  -- Loop: dispatch scale kernel `passes` times
  "    v22 = iconst.i64 0\n" ++
  "    jump block1(v22)\n" ++
  "\n" ++
  "block1(v23: i64):\n" ++
  "    v24 = icmp uge v23, v4\n" ++
  "    brif v24, block2, block3(v23)\n" ++
  "\n" ++
  "block3(v25: i64):\n" ++
  "    v26 = call fn4(v0, v14, v21, v13, v13)\n" ++
  "    v27 = iadd_imm v25, 1\n" ++
  "    jump block1(v27)\n" ++
  "\n" ++
  "block2:\n" ++
  -- Dispatch reduce kernel once
  "    v28 = call fn4(v0, v18, v21, v13, v13)\n" ++
  -- Download sums buffer to out_ptr (buf_offset=0)
  "    v29 = iconst.i64 0\n" ++
  "    v30 = call fn5(v0, v9, v29, v3, v8)\n" ++
  -- GPU cleanup
  "    call fn6(v0)\n" ++
  "    return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifGpuFn

def scaleShaderBytes : List UInt8 :=
  scaleShader.toUTF8.toList ++ [0]

def reduceShaderBytes : List UInt8 :=
  reduceShader.toUTF8.toList ++ [0]

-- Scale bind: [buf_id=0, read_only=0]
def scaleBindDesc : List UInt8 :=
  [0, 0, 0, 0, 0, 0, 0, 0]

-- Reduce bind: [buf_id=0, read_only=1], [buf_id=1, read_only=0]
def reduceBindDesc : List UInt8 :=
  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros SCALE_SHADER_OFF      -- 0x0000..0x00FF (includes passes at 0x28)
  let scale := scaleShaderBytes ++ zeros (REDUCE_SHADER_OFF - SCALE_SHADER_OFF - scaleShaderBytes.length)
  let reduce := reduceShaderBytes ++ zeros (SCALE_BIND_OFF - REDUCE_SHADER_OFF - reduceShaderBytes.length)
  let scaleBind := scaleBindDesc
  let reduceBind := reduceBindDesc ++ zeros (MEM_SIZE - REDUCE_BIND_OFF - reduceBindDesc.length)
  reserved ++ scale ++ reduce ++ scaleBind ++ reduceBind

def controlActions : List Action :=
  [{ kind := .ClifCall, dst := 0, src := 1, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0,
  initial_memory := buildInitialMemory
}

def buildAlgorithm : Algorithm := {
  actions := controlActions,
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

end GpuIterBench

def main : IO Unit := do
  let json := toJsonPair GpuIterBench.buildConfig GpuIterBench.buildAlgorithm
  IO.println (Json.compress json)
