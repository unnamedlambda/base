import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CudaSaxpyPersist

/-
  Persistent CUDA SAXPY with alpha fixed at 2.0.

  Data formats:
    load  data: [n: u64]
    prep  data: [x: n f32][y: n f32]
    infer out:  [2*x + y: n f32]  (optional; compute-only if out_len=0)
-/

def PTX_SOURCE_OFF : Nat := 0x0100
def BIND_DESC_OFF  : Nat := 0x1400
def MEM_SIZE       : Nat := 0x1500
def TIMEOUT_MS     : Nat := 30000

def ptxSource : String :=
  ".version 8.0\n" ++
  ".target sm_86\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 meta_ptr,\n" ++
  "    .param .u64 x_ptr,\n" ++
  "    .param .u64 y_ptr\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred %p;\n" ++
  "    .reg .u32 %r<4>;\n" ++
  "    .reg .u64 %rd<8>;\n" ++
  "    .reg .f32 %f<5>;\n" ++
  "\n" ++
  "    ld.param.u64 %rd0, [meta_ptr];\n" ++
  "    ld.param.u64 %rd1, [x_ptr];\n" ++
  "    ld.param.u64 %rd2, [y_ptr];\n" ++
  "    ld.global.u32 %r0, [%rd0];\n" ++
  "    mov.u32 %r1, %ctaid.x;\n" ++
  "    mov.u32 %r2, %tid.x;\n" ++
  "    mad.lo.u32 %r1, %r1, 256, %r2;\n" ++
  "    setp.ge.u32 %p, %r1, %r0;\n" ++
  "    @%p bra DONE;\n" ++
  "    cvt.u64.u32 %rd3, %r1;\n" ++
  "    shl.b64 %rd3, %rd3, 2;\n" ++
  "    add.u64 %rd4, %rd1, %rd3;\n" ++
  "    add.u64 %rd5, %rd2, %rd3;\n" ++
  "    ld.global.f32 %f0, [%rd4];\n" ++
  "    ld.global.f32 %f1, [%rd5];\n" ++
  "    mov.f32 %f2, 0f40000000;\n" ++
  "    fma.rn.f32 %f3, %f2, %f0, %f1;\n" ++
  "    st.global.f32 [%rd5], %f3;\n" ++
  "DONE:\n" ++
  "    ret;\n" ++
  "}\n"

def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "  return\n" ++
  "}\n"

def clifLoadFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++
  "    sig1 = (i64, i64) -> i32 system_v\n" ++
  "    sig2 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_init sig0\n" ++
  "    fn1 = %cl_cuda_create_buffer sig1\n" ++
  "    fn2 = %cl_cuda_upload sig2\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  call fn0(v0)\n" ++
  "  v10 = load.i64 notrap aligned v1\n" ++
  "  store notrap aligned v10, v0+0x28\n" ++
  "  v11 = ishl_imm v10, 2\n" ++
  "  v12 = iconst.i64 8\n" ++
  "  v20 = call fn1(v0, v12)\n" ++
  "  v21 = call fn1(v0, v11)\n" ++
  "  v22 = call fn1(v0, v11)\n" ++
  "  store notrap aligned v20, v0+0x30\n" ++
  "  store notrap aligned v21, v0+0x34\n" ++
  "  store notrap aligned v22, v0+0x38\n" ++
  "  v23 = iconst.i64 0x28\n" ++
  "  v24 = call fn2(v0, v20, v23, v12)\n" ++
  "  return\n" ++
  "}\n"

def clifPrepFn : String :=
  "function u0:2(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_upload_ptr sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  v2 = load.i64 notrap aligned v0+0x28\n" ++
  "  v3 = load.i32 notrap aligned v0+0x34\n" ++
  "  v4 = load.i32 notrap aligned v0+0x38\n" ++
  "  v5 = ishl_imm v2, 2\n" ++
  "  v6 = call fn0(v0, v3, v1, v5)\n" ++
  "  v7 = iadd v1, v5\n" ++
  "  v8 = call fn0(v0, v4, v7, v5)\n" ++
  "  return\n" ++
  "}\n"

def clifInferFn : String :=
  "function u0:3(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    sig1 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v\n" ++
  "    sig2 = (i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_download_ptr sig0\n" ++
  "    fn1 = %cl_cuda_launch sig1\n" ++
  "    fn2 = %cl_cuda_sync sig2\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x18\n" ++
  "  v2 = load.i64 notrap aligned v0+0x20\n" ++
  "  v3 = load.i64 notrap aligned v0+0x28\n" ++
  "  v4 = iadd_imm v3, 255\n" ++
  "  v5 = ushr_imm v4, 8\n" ++
  "  v6 = ireduce.i32 v5\n" ++
  "  v7 = iconst.i64 " ++ toString PTX_SOURCE_OFF ++ "\n" ++
  "  v8 = iconst.i32 3\n" ++
  "  v9 = iconst.i64 " ++ toString BIND_DESC_OFF ++ "\n" ++
  "  v10 = iconst.i32 1\n" ++
  "  v11 = iconst.i32 256\n" ++
  "  v12 = call fn1(v0, v7, v8, v9, v6, v10, v10, v11, v10, v10)\n" ++
  "  v13 = call fn2(v0)\n" ++
  "  v14 = iconst.i64 0\n" ++
  "  v15 = icmp eq v2, v14\n" ++
  "  brif v15, block1, block2\n" ++
  "block1:\n" ++
  "  return\n" ++
  "block2:\n" ++
  "  v16 = load.i32 notrap aligned v0+0x38\n" ++
  "  v17 = call fn0(v0, v16, v1, v2)\n" ++
  "  return\n" ++
  "}\n"

def clifIR : String := clifNoopFn ++ "\n" ++ clifLoadFn ++ "\n" ++ clifPrepFn ++ "\n" ++ clifInferFn

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let ptx := ptxBytes ++ zeros (BIND_DESC_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ ptx ++ bind

def actions (src : UInt32) : List Action :=
  [{ kind := .ClifCall, dst := 0, src := src, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0,
  initial_memory := buildInitialMemory
}

def loadAlgorithm : Algorithm := { actions := actions 1, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }
def prepAlgorithm : Algorithm := { actions := actions 2, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }
def inferAlgorithm : Algorithm := { actions := actions 3, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }

end CudaSaxpyPersist

def main : IO Unit := do
  let json := Json.arr #[
    toJson CudaSaxpyPersist.buildConfig,
    toJson CudaSaxpyPersist.loadAlgorithm,
    toJson CudaSaxpyPersist.prepAlgorithm,
    toJson CudaSaxpyPersist.inferAlgorithm
  ]
  IO.println (Json.compress json)
