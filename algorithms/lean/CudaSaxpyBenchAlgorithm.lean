import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CudaSaxpyBench

/-
  CUDA SAXPY benchmark algorithm — Cranelift JIT + CUDA PTX version.

  SAXPY: y[i] = 2.0 * x[i] + y[i]

  Payload (via execute data arg): [x floats: N][y floats: N]
  Output (via execute_into out arg): [y result floats: N]

  N is derived from data_len: N = data_len / 8 (two f32 arrays)

  Memory layout (shared memory):
    0x0000  RESERVED        (40 bytes, runtime-managed)
    0x0100  PTX_SOURCE      (null-terminated PTX source, 4096 bytes)
    0x1100  BIND_DESC       (8 bytes: [buf_id=0, buf_id=1] as 2 x i32)

  Single CLIF function:
    1. Read data_ptr, data_len, out_ptr from reserved region
    2. cl_cuda_init
    3. Compute N = data_len / 8, buf_size = N * 4
    4. cl_cuda_create_buffer(buf_size) x 2 (x and y)
    5. cl_cuda_upload_ptr(x_buf, data_ptr, buf_size)
    6. cl_cuda_upload_ptr(y_buf, data_ptr + buf_size, buf_size)
    7. cl_cuda_launch(ptx_off, 2, bind_off, grid, 1, 1, 256, 1, 1)
    8. cl_cuda_download_ptr(y_buf, out_ptr, buf_size)
    9. cl_cuda_cleanup
-/

def PTX_SOURCE_OFF : Nat := 0x0100
def BIND_DESC_OFF  : Nat := 0x1100
def MEM_SIZE       : Nat := 0x1200
def TIMEOUT_MS     : Nat := 120000

-- PTX kernel: y[i] = 2.0 * x[i] + y[i]
-- Block size 256, grid = ceil(N / 256).
def ptxSource : String :=
  ".version 7.0\n" ++
  ".target sm_50\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 x_ptr,\n" ++
  "    .param .u64 y_ptr\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .u32 %r0, %r1;\n" ++
  "    .reg .u64 %rx, %ry, %off;\n" ++
  "    .reg .f32 %fx, %fy, %fa;\n" ++
  "\n" ++
  "    mov.u32 %r0, %ctaid.x;\n" ++
  "    mov.u32 %r1, %tid.x;\n" ++
  "    mad.lo.u32 %r0, %r0, 256, %r1;\n" ++
  "\n" ++
  "    cvt.u64.u32 %off, %r0;\n" ++
  "    shl.b64 %off, %off, 2;\n" ++
  "\n" ++
  "    ld.param.u64 %rx, [x_ptr];\n" ++
  "    ld.param.u64 %ry, [y_ptr];\n" ++
  "\n" ++
  "    add.u64 %rx, %rx, %off;\n" ++
  "    add.u64 %ry, %ry, %off;\n" ++
  "\n" ++
  "    ld.global.f32 %fx, [%rx];\n" ++
  "    ld.global.f32 %fy, [%ry];\n" ++
  "    mov.f32 %fa, 0f40000000;\n" ++     -- 2.0f IEEE 754
  "    fma.rn.f32 %fy, %fa, %fx, %fy;\n" ++
  "    st.global.f32 [%ry], %fy;\n" ++
  "\n" ++
  "    ret;\n" ++
  "}\n"

-- fn0: noop
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

-- fn1: CUDA SAXPY orchestrator
def clifCudaFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++                              -- cuda_init / cuda_cleanup
  "    sig1 = (i64, i64) -> i32 system_v\n" ++                  -- cuda_create_buffer
  "    sig2 = (i64, i32, i64, i64) -> i32 system_v\n" ++        -- cuda_upload_ptr / cuda_download_ptr
  "    sig3 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v\n" ++ -- cuda_launch
  "\n" ++
  "    fn0 = %cl_cuda_init sig0\n" ++
  "    fn1 = %cl_cuda_create_buffer sig1\n" ++
  "    fn2 = %cl_cuda_upload_ptr sig2\n" ++
  "    fn3 = %cl_cuda_download_ptr sig2\n" ++
  "    fn4 = %cl_cuda_launch sig3\n" ++
  "    fn5 = %cl_cuda_cleanup sig0\n" ++
  "\n" ++
  "block0(v0: i64):\n" ++
  "    v1 = load.i64 notrap aligned v0+0x08\n" ++               -- data_ptr
  "    v2 = load.i64 notrap aligned v0+0x10\n" ++               -- data_len
  "    v3 = load.i64 notrap aligned v0+0x18\n" ++               -- out_ptr
  -- CUDA init
  "    call fn0(v0)\n" ++
  -- Compute: N = data_len / 8, buf_size = N * 4 = data_len / 2
  "    v4 = ushr_imm v2, 1\n" ++                                -- buf_size = data_len / 2
  -- Create 2 device buffers (x and y)
  "    v5 = call fn1(v0, v4)\n" ++                              -- x_buf
  "    v6 = call fn1(v0, v4)\n" ++                              -- y_buf
  -- Upload x from data_ptr
  "    v7 = call fn2(v0, v5, v1, v4)\n" ++
  -- Upload y from data_ptr + buf_size
  "    v8 = iadd v1, v4\n" ++                                   -- data_ptr + buf_size
  "    v9 = call fn2(v0, v6, v8, v4)\n" ++
  -- Compute grid: ceil(N / 256) = (N + 255) / 256
  -- N = buf_size / 4
  "    v10 = ushr_imm v4, 2\n" ++                               -- N
  "    v11 = ireduce.i32 v10\n" ++                               -- N as i32
  "    v12 = iconst.i32 255\n" ++
  "    v13 = iadd v11, v12\n" ++                                 -- N + 255
  "    v14 = iconst.i32 256\n" ++
  "    v15 = udiv v13, v14\n" ++                                 -- gridX = ceil(N/256)
  "    v16 = iconst.i32 1\n" ++
  -- Launch PTX
  "    v17 = iconst.i64 256\n" ++                                -- PTX_SOURCE_OFF
  "    v18 = iconst.i32 2\n" ++                                  -- 2 buffer bindings
  "    v19 = iconst.i64 4352\n" ++                               -- BIND_DESC_OFF (0x1100)
  "    v20 = call fn4(v0, v17, v18, v19, v15, v16, v16, v14, v16, v16)\n" ++
  -- Download y result to out_ptr
  "    v21 = call fn3(v0, v6, v3, v4)\n" ++
  -- CUDA cleanup
  "    call fn5(v0)\n" ++
  "    return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifCudaFn

def ptxBytes : List UInt8 :=
  ptxSource.toUTF8.toList ++ [0]

def bindDesc : List UInt8 :=
  -- buf_id=0 (x), buf_id=1 (y) — two i32 values
  [0, 0, 0, 0, 1, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let ptx := ptxBytes ++ zeros (BIND_DESC_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ ptx ++ bind

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

end CudaSaxpyBench

def main : IO Unit := do
  let json := toJsonPair CudaSaxpyBench.buildConfig CudaSaxpyBench.buildAlgorithm
  IO.println (Json.compress json)
