import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace GpuMatMulBench

/-
  GPU MatMul benchmark algorithm — Cranelift JIT + wgpu version.

  Payload (via execute data arg): [A floats: N*N][B floats: N*N]
  Output (via execute_into out arg): [C floats: N*N]

  Buffer layout on GPU: [A: N*N][B: N*N][C: N*N] = 3*N*N floats
  N is derived in WGSL from arrayLength: N = u32(sqrt(f32(arrayLength(&data) / 3u)))

  Memory layout (shared memory):
    0x0000  RESERVED        (40 bytes, runtime-managed)
    0x0100  WGSL_SHADER     (null-terminated WGSL source)
    0x1100  BIND_DESC       (8 bytes: [buf_id=0, read_only=0])

  Single CLIF function:
    1. Read data_ptr, data_len, out_ptr from reserved region
    2. cl_gpu_init
    3. Compute nn = data_len / 8, buffer_size = nn*4*3, workgroups = (nn+63)/64
    4. cl_gpu_create_buffer(buffer_size)
    5. cl_gpu_upload_ptr(buf_id, data_ptr, data_len) — upload A+B from payload
    6. cl_gpu_create_pipeline
    7. cl_gpu_dispatch
    8. cl_gpu_download_ptr — download C (last nn floats) to out_ptr
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
  "    let total = arrayLength(&data);\n" ++
  "    let nn = total / 3u;\n" ++
  "    let N = u32(sqrt(f32(nn)));\n" ++
  "    let idx = gid.x;\n" ++
  "    if (idx >= nn) { return; }\n" ++
  "    let i = idx / N;\n" ++
  "    let j = idx % N;\n" ++
  "    var sum: f32 = 0.0;\n" ++
  "    for (var k: u32 = 0u; k < N; k = k + 1u) {\n" ++
  "        sum = sum + data[i * N + k] * data[nn + k * N + j];\n" ++
  "    }\n" ++
  "    data[2u * nn + idx] = sum;\n" ++
  "}\n"

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: GPU MatMul orchestrator
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
  "    v2 = load.i64 notrap aligned v0+0x10\n" ++               -- data_len (= 2*nn*4)
  "    v3 = load.i64 notrap aligned v0+0x18\n" ++               -- out_ptr
  -- GPU init
  "    call fn0(v0)\n" ++
  -- Compute: nn = data_len / 8, buffer_size = nn*12 (3 matrices), workgroups
  "    v4 = ushr_imm v2, 3\n" ++                                -- nn = data_len / 8
  "    v5 = iconst.i64 12\n" ++
  "    v6 = imul v4, v5\n" ++                                   -- buffer_size = nn * 12 (3*nn*4)
  "    v7 = iadd_imm v4, 63\n" ++
  "    v8 = ushr_imm v7, 6\n" ++                                -- workgroups = (nn+63)/64
  "    v9 = ireduce.i32 v8\n" ++
  -- Create buffer (holds A + B + C)
  "    v10 = call fn1(v0, v6)\n" ++                              -- buf_id
  -- Upload A+B from payload (data_len bytes)
  "    v11 = call fn3(v0, v10, v1, v2)\n" ++
  -- Create pipeline
  "    v12 = iconst.i64 256\n" ++                                -- WGSL_SHADER_OFF
  "    v13 = iconst.i64 4352\n" ++                               -- BIND_DESC_OFF
  "    v14 = iconst.i32 1\n" ++
  "    v15 = call fn2(v0, v12, v13, v14)\n" ++
  -- Dispatch
  "    v16 = call fn4(v0, v15, v9, v14, v14)\n" ++
  -- Download C matrix: nn*4 bytes from GPU buffer offset 2*nn*4
  "    v17 = ishl_imm v4, 2\n" ++                               -- nn * 4 (download size)
  "    v18 = ishl_imm v4, 3\n" ++                               -- nn * 8 = 2*nn*4 (buf_offset)
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

end GpuMatMulBench

def main : IO Unit := do
  let json := toJsonPair GpuMatMulBench.buildConfig GpuMatMulBench.buildAlgorithm
  IO.println (Json.compress json)
