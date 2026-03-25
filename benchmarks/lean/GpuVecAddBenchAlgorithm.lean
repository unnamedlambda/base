import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace GpuVecAddBench

/-
  GPU VecAdd benchmark algorithm — Cranelift JIT + wgpu version.

  Payload (via execute data arg): [A floats][B floats]
  Output (via execute_into out arg): [C floats] (A[i] + B[i])

  Memory layout (shared memory):
    0x0000  RESERVED        (40 bytes, runtime-managed)
    0x0100  WGSL_SHADER     (null-terminated WGSL source)
    0x1100  BIND_DESC       (8 bytes: [buf_id=0, read_only=0])

  Single CLIF function:
    1. Read data_ptr, data_len, out_ptr from reserved region
    2. cl_gpu_init
    3. Compute n = data_len / 8, buffer_size = data_len, workgroups = (n+63)/64
    4. cl_gpu_create_buffer(buffer_size)
    5. cl_gpu_upload_ptr(buf_id, data_ptr, buffer_size) — zero copy from payload
    6. cl_gpu_create_pipeline(shader_off, bind_off, 1)
    7. cl_gpu_dispatch(pipe_id, workgroups, 1, 1)
    8. cl_gpu_download_ptr(buf_id, 0, out_ptr, n*4) — zero copy to output
    9. cl_gpu_cleanup
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

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: GPU VecAdd orchestrator
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
  "    v2 = load.i64 notrap aligned v0+0x10\n" ++               -- data_len
  "    v3 = load.i64 notrap aligned v0+0x18\n" ++               -- out_ptr
  -- GPU init
  "    call fn0(v0)\n" ++
  -- Compute: n = data_len / 8, workgroups = (n+63)/64
  "    v4 = ushr_imm v2, 3\n" ++                                -- n = data_len / 8
  "    v5 = iadd_imm v4, 63\n" ++
  "    v6 = ushr_imm v5, 6\n" ++                                -- workgroups = (n+63)/64
  "    v7 = ireduce.i32 v6\n" ++
  -- Create buffer (size = data_len)
  "    v10 = call fn1(v0, v2)\n" ++                              -- buf_id
  -- Upload from payload
  "    v11 = call fn3(v0, v10, v1, v2)\n" ++
  -- Create pipeline
  "    v12 = iconst.i64 256\n" ++                                -- WGSL_SHADER_OFF
  "    v13 = iconst.i64 4352\n" ++                               -- BIND_DESC_OFF (0x1100)
  "    v14 = iconst.i32 1\n" ++
  "    v15 = call fn2(v0, v12, v13, v14)\n" ++
  -- Dispatch
  "    v16 = call fn4(v0, v15, v7, v14, v14)\n" ++
  -- Download first n floats to out_ptr (buf_offset=0)
  "    v17 = ishl_imm v4, 2\n" ++                               -- n * 4 bytes
  "    v18 = iconst.i64 0\n" ++                                  -- buf_offset = 0
  "    v19 = call fn5(v0, v10, v18, v3, v17)\n" ++
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

end GpuVecAddBench

def main : IO Unit := do
  let json := toJsonPair GpuVecAddBench.buildConfig GpuVecAddBench.buildAlgorithm
  IO.println (Json.compress json)
