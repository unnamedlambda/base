import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace GpuReductionBench

/-
  GPU Reduction benchmark algorithm — Cranelift JIT + wgpu version.

  Payload (via execute data arg): [f32 values: N floats]
  Output (via execute_into out arg): [f32 partial sums: N/64 floats]

  Buffer layout on GPU: [input: N][output: N/64] = N + N/64 floats
  N is derived from payload size: N = data_len / 4
  num_groups = N / 64

  Each workgroup of 64 threads sums 64 elements via shared memory.
  Partial sums written to output region data[N + wid.x].

  Memory layout (shared memory):
    0x0000  RESERVED        (40 bytes, runtime-managed)
    0x0100  WGSL_SHADER     (null-terminated WGSL source)
    0x1100  BIND_DESC       (8 bytes: [buf_id=0, read_only=0])

  Single CLIF function:
    1. Read data_ptr, data_len, out_ptr from reserved region
    2. cl_gpu_init
    3. Compute N = data_len/4, num_groups = N/64, buf_size = (N+num_groups)*4
    4. cl_gpu_create_buffer(buf_size)
    5. cl_gpu_upload_ptr(buf_id, data_ptr, data_len) — upload input from payload
    6. cl_gpu_create_pipeline
    7. cl_gpu_dispatch(pipe_id, num_groups, 1, 1)
    8. cl_gpu_download_ptr(buf_id, N*4, out_ptr, num_groups*4) — download partial sums
    9. cl_gpu_cleanup
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
  -- total = N + N/64 = N*65/64, so num_groups = total/65, input_n = num_groups*64
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

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: GPU Reduction orchestrator
def clifGpuFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++                              -- gpu_init / gpu_cleanup
  "    sig1 = (i64, i64) -> i32 system_v\n" ++                  -- gpu_create_buffer
  "    sig2 = (i64, i64, i64, i32) -> i32 system_v\n" ++        -- gpu_create_pipeline
  "    sig3 = (i64, i32, i64, i64) -> i32 system_v\n" ++        -- gpu_upload_ptr
  "    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v\n" ++   -- gpu_dispatch
  "    sig5 = (i64, i32, i64, i64, i64) -> i32 system_v\n" ++   -- gpu_download_ptr(ptr, buf_id, buf_offset, dst_ptr, size)
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
  "    v2 = load.i64 notrap aligned v0+0x10\n" ++               -- data_len (N * 4)
  "    v3 = load.i64 notrap aligned v0+0x18\n" ++               -- out_ptr
  -- GPU init
  "    call fn0(v0)\n" ++
  -- Compute: N = data_len/4, num_groups = N/64, buf_size = (N + num_groups) * 4
  "    v4 = ushr_imm v2, 2\n" ++                                -- N = data_len / 4
  "    v5 = ushr_imm v4, 6\n" ++                                -- num_groups = N / 64
  "    v6 = iadd v4, v5\n" ++                                   -- N + num_groups
  "    v7 = ishl_imm v6, 2\n" ++                                -- buf_size = (N + num_groups) * 4
  "    v8 = ireduce.i32 v5\n" ++                                -- workgroups (= num_groups)
  -- Create buffer (holds input + output)
  "    v10 = call fn1(v0, v7)\n" ++                              -- buf_id
  -- Upload input from payload (data_len bytes)
  "    v11 = call fn3(v0, v10, v1, v2)\n" ++
  -- Create pipeline
  "    v12 = iconst.i64 256\n" ++                                -- WGSL_SHADER_OFF
  "    v13 = iconst.i64 4352\n" ++                               -- BIND_DESC_OFF
  "    v14 = iconst.i32 1\n" ++
  "    v15 = call fn2(v0, v12, v13, v14)\n" ++
  -- Dispatch
  "    v16 = call fn4(v0, v15, v8, v14, v14)\n" ++
  -- Download partial sums: num_groups*4 bytes from GPU buf offset N*4
  "    v17 = ishl_imm v5, 2\n" ++                               -- num_groups * 4 (download size)
  "    v19 = call fn5(v0, v10, v2, v3, v17)\n" ++               -- buf_offset = data_len = N*4
  -- GPU cleanup
  "    call fn6(v0)\n" ++
  "    return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifGpuFn

def wgslBytes : List UInt8 :=
  wgslShader.toUTF8.toList ++ [0]

def bindDesc : List UInt8 :=
  [0, 0, 0, 0, 0, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let shader := wgslBytes ++ zeros (BIND_DESC_OFF - WGSL_SHADER_OFF - wgslBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ shader ++ bind

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

end GpuReductionBench

def main : IO Unit := do
  let json := toJsonPair GpuReductionBench.buildConfig GpuReductionBench.buildAlgorithm
  IO.println (Json.compress json)
