import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CudaGemvPersist

/-
  Persistent GEMV: A stays on GPU, only x is uploaded per call.

  One Base instance, three Algorithm objects (same CLIF module):
    u0:1  load   — init CUDA, alloc 3 bufs, upload A, store m/n in sm, no cleanup
    u0:2  prep   — upload x to persistent buf1
    u0:3  infer  — cuBLAS SGEMV, optional download from buf2

  Shared memory layout:
    0x00-0x27  reserved (runtime-written)
    0x28-0x2F  m (i64, written by load)
    0x30-0x37  n (i64, written by load)

  Data formats:
    load  data: [m: u64][n: u64][A: m*n f32]
    prep  data: [x: n f32]
    infer out:  [y: m f32]  (optional; compute-only if out_len=0)
-/

def MEM_SIZE   : Nat := 0x40
def TIMEOUT_MS : Nat := 30000

def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n"

def clifLoadFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++
  "    sig1 = (i64, i64) -> i32 system_v\n" ++
  "    sig2 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_init sig0\n" ++
  "    fn1 = %cl_cuda_create_buffer sig1\n" ++
  "    fn2 = %cl_cuda_upload_ptr sig2\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  call fn0(v0)\n" ++
  "  v10 = load.i64 notrap aligned v1\n" ++
  "  v11 = iadd_imm v1, 8\n" ++
  "  v12 = load.i64 notrap aligned v11\n" ++
  "  v13 = iadd_imm v0, 0x28\n" ++
  "  store notrap aligned v10, v13\n" ++
  "  v14 = iadd_imm v0, 0x30\n" ++
  "  store notrap aligned v12, v14\n" ++
  "  v20 = imul v10, v12\n" ++
  "  v21 = ishl_imm v20, 2\n" ++
  "  v22 = ishl_imm v12, 2\n" ++
  "  v23 = ishl_imm v10, 2\n" ++
  "  v30 = call fn1(v0, v21)\n" ++
  "  v31 = call fn1(v0, v22)\n" ++
  "  v32 = call fn1(v0, v23)\n" ++
  "  v40 = iconst.i64 16\n" ++
  "  v41 = iadd v1, v40\n" ++
  "  v42 = call fn2(v0, v30, v41, v21)\n" ++
  "  return\n" ++
  "}\n"

def clifPrepFn : String :=
  "function u0:2(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_upload_ptr sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  v2 = load.i64 notrap aligned v0+0x10\n" ++
  "  v20 = iconst.i32 1\n" ++
  "  v21 = call fn0(v0, v20, v1, v2)\n" ++
  "  return\n" ++
  "}\n"

def clifInferFn : String :=
  "function u0:3(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    sig1 = (i64, i32, i32, i32, i32, i32, i32, i32, i32) -> i32 system_v\n" ++
  "    sig2 = (i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_download_ptr sig0\n" ++
  "    fn1 = %cl_cublas_sgemv sig1\n" ++
  "    fn2 = %cl_cuda_sync sig2\n" ++
  "block0(v0: i64):\n" ++
  "  v3 = load.i64 notrap aligned v0+0x18\n" ++
  "  v4 = load.i64 notrap aligned v0+0x20\n" ++
  "  v10 = load.i64 notrap aligned v0+0x28\n" ++
  "  v11 = load.i64 notrap aligned v0+0x30\n" ++
  "  v30 = ireduce.i32 v10\n" ++
  "  v31 = ireduce.i32 v11\n" ++
  "  v32 = iconst.i32 1\n" ++
  "  v33 = iconst.i32 0x3f800000\n" ++
  "  v34 = iconst.i32 0\n" ++
  "  v35 = iconst.i32 0\n" ++
  "  v36 = iconst.i32 2\n" ++
  "  v37 = call fn1(v0, v32, v31, v30, v33, v34, v32, v35, v36)\n" ++
  "  v38 = call fn2(v0)\n" ++
  "  v39 = iconst.i64 0\n" ++
  "  v40 = icmp eq v4, v39\n" ++
  "  brif v40, block1, block2\n" ++
  "block1:\n" ++
  "  return\n" ++
  "block2:\n" ++
  "  v41 = call fn0(v0, v36, v3, v4)\n" ++
  "  return\n" ++
  "}\n"

def clifIR : String := clifNoopFn ++ "\n" ++ clifLoadFn ++ "\n" ++ clifPrepFn ++ "\n" ++ clifInferFn

def actions (src : UInt32) : List Action :=
  [{ kind := .ClifCall, dst := 0, src := src, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0
}

def loadAlgorithm : Algorithm := {
  actions := actions 1,
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

def prepAlgorithm : Algorithm := {
  actions := actions 2,
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

def inferAlgorithm : Algorithm := {
  actions := actions 3,
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

end CudaGemvPersist

def main : IO Unit := do
  let json := Json.arr #[
    toJson CudaGemvPersist.buildConfig,
    toJson CudaGemvPersist.loadAlgorithm,
    toJson CudaGemvPersist.prepAlgorithm,
    toJson CudaGemvPersist.inferAlgorithm
  ]
  IO.println (Json.compress json)
