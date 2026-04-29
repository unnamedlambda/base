import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CudaRmsNormPersist

def PTX_SOURCE_OFF : Nat := 0x0100
def BIND_DESC_OFF  : Nat := 0x1100
def MEM_SIZE       : Nat := 0x1200
def TIMEOUT_MS     : Nat := 30000

def ptxSource : String :=
  ".version 8.0\n" ++
  ".target sm_86\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".shared .align 4 .b8 _smem[36];\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 buf0,\n" ++
  "    .param .u64 buf1\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred   %p;\n" ++
  "    .reg .u32    %r<12>;\n" ++
  "    .reg .u64    %rd<10>;\n" ++
  "    .reg .f32    %f<10>;\n" ++
  "\n" ++
  "    ld.param.u64  %rd0, [buf0];\n" ++
  "    ld.param.u64  %rd1, [buf1];\n" ++
  "    ld.global.u32 %r0, [%rd0];\n" ++
  "    mov.u32       %r1, %tid.x;\n" ++
  "    shr.u32       %r2, %r1, 5;\n" ++
  "    and.b32       %r3, %r1, 31;\n" ++
  "    add.u64       %rd2, %rd0, 8;\n" ++
  "    cvt.u64.u32   %rd3, %r0;\n" ++
  "    shl.b64       %rd3, %rd3, 2;\n" ++
  "    add.u64       %rd4, %rd2, %rd3;\n" ++
  "    mov.f32       %f0, 0f00000000;\n" ++
  "    mov.u32       %r4, %r1;\n" ++
  "loop1:\n" ++
  "    setp.ge.u32   %p, %r4, %r0;\n" ++
  "    @%p bra       done1;\n" ++
  "    cvt.u64.u32   %rd5, %r4;\n" ++
  "    shl.b64       %rd5, %rd5, 2;\n" ++
  "    add.u64       %rd6, %rd2, %rd5;\n" ++
  "    ld.global.f32 %f1, [%rd6];\n" ++
  "    fma.rn.f32    %f0, %f1, %f1, %f0;\n" ++
  "    add.u32       %r4, %r4, 256;\n" ++
  "    bra           loop1;\n" ++
  "done1:\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 16, 31, 0xffffffff;\n" ++
  "    add.f32       %f0, %f0, %f2;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 8, 31, 0xffffffff;\n" ++
  "    add.f32       %f0, %f0, %f2;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 4, 31, 0xffffffff;\n" ++
  "    add.f32       %f0, %f0, %f2;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 2, 31, 0xffffffff;\n" ++
  "    add.f32       %f0, %f0, %f2;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 1, 31, 0xffffffff;\n" ++
  "    add.f32       %f0, %f0, %f2;\n" ++
  "    setp.ne.u32   %p, %r3, 0;\n" ++
  "    @%p bra       skip1;\n" ++
  "    mov.u32       %r5, _smem;\n" ++
  "    shl.b32       %r6, %r2, 2;\n" ++
  "    add.u32       %r5, %r5, %r6;\n" ++
  "    st.shared.f32 [%r5], %f0;\n" ++
  "skip1:\n" ++
  "    bar.sync      0;\n" ++
  "    setp.ne.u32   %p, %r1, 0;\n" ++
  "    @%p bra       skip2;\n" ++
  "    mov.u32       %r5, _smem;\n" ++
  "    ld.shared.f32 %f0, [%r5+0];\n" ++
  "    ld.shared.f32 %f1, [%r5+4];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+8];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+12];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+16];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+20];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+24];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+28];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    cvt.rn.f32.u32 %f1, %r0;\n" ++
  "    div.rn.f32    %f0, %f0, %f1;\n" ++
  "    mov.f32       %f1, 0f3727c5ac;\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    rsqrt.approx.f32 %f0, %f0;\n" ++
  "    mov.u32       %r5, _smem;\n" ++
  "    st.shared.f32 [%r5+32], %f0;\n" ++
  "skip2:\n" ++
  "    bar.sync      0;\n" ++
  "    mov.u32       %r5, _smem;\n" ++
  "    ld.shared.f32 %f3, [%r5+32];\n" ++
  "    mov.u32       %r4, %r1;\n" ++
  "loop2:\n" ++
  "    setp.ge.u32   %p, %r4, %r0;\n" ++
  "    @%p bra       done2;\n" ++
  "    cvt.u64.u32   %rd5, %r4;\n" ++
  "    shl.b64       %rd5, %rd5, 2;\n" ++
  "    add.u64       %rd6, %rd2, %rd5;\n" ++
  "    ld.global.f32 %f4, [%rd6];\n" ++
  "    add.u64       %rd7, %rd4, %rd5;\n" ++
  "    ld.global.f32 %f5, [%rd7];\n" ++
  "    mul.f32       %f4, %f4, %f3;\n" ++
  "    mul.f32       %f4, %f4, %f5;\n" ++
  "    add.u64       %rd8, %rd1, %rd5;\n" ++
  "    st.global.f32 [%rd8], %f4;\n" ++
  "    add.u32       %r4, %r4, 256;\n" ++
  "    bra           loop2;\n" ++
  "done2:\n" ++
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
  "    sig3 = (i64, i32, i64, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_init sig0\n" ++
  "    fn1 = %cl_cuda_create_buffer sig1\n" ++
  "    fn2 = %cl_cuda_upload_ptr sig2\n" ++
  "    fn3 = %cl_cuda_upload_ptr_offset sig3\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  call fn0(v0)\n" ++
  "  v10 = load.i64 notrap aligned v1\n" ++
  "  v11 = iadd_imm v0, 0x28\n" ++
  "  store notrap aligned v10, v11\n" ++
  "  v12 = ishl_imm v10, 2\n" ++
  "  v13 = iadd_imm v12, 8\n" ++
  "  v14 = iadd v13, v12\n" ++
  "  v15 = call fn1(v0, v14)\n" ++
  "  v16 = call fn1(v0, v12)\n" ++
  "  store notrap aligned v15, v0+0x30\n" ++
  "  store notrap aligned v16, v0+0x34\n" ++
  "  v17 = iconst.i64 0\n" ++
  "  v18 = iconst.i64 8\n" ++
  "  v19 = call fn3(v0, v15, v17, v11, v18)\n" ++
  "  v20 = iadd_imm v1, 8\n" ++
  "  v21 = call fn3(v0, v15, v13, v20, v12)\n" ++
  "  return\n" ++
  "}\n"

def clifPrepFn : String :=
  "function u0:2(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_upload_ptr_offset sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  v2 = load.i64 notrap aligned v0+0x28\n" ++
  "  v3 = load.i32 notrap aligned v0+0x30\n" ++
  "  v4 = iconst.i64 8\n" ++
  "  v5 = ishl_imm v2, 2\n" ++
  "  v6 = call fn0(v0, v3, v4, v1, v5)\n" ++
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
  "  v3 = iconst.i64 " ++ toString PTX_SOURCE_OFF ++ "\n" ++
  "  v4 = iconst.i32 2\n" ++
  "  v5 = iconst.i64 " ++ toString BIND_DESC_OFF ++ "\n" ++
  "  v6 = iconst.i32 1\n" ++
  "  v7 = iconst.i32 256\n" ++
  "  v8 = call fn1(v0, v3, v4, v5, v6, v6, v6, v7, v6, v6)\n" ++
  "  v9 = call fn2(v0)\n" ++
  "  v10 = iconst.i64 0\n" ++
  "  v11 = icmp eq v2, v10\n" ++
  "  brif v11, block1, block2\n" ++
  "block1:\n" ++
  "  return\n" ++
  "block2:\n" ++
  "  v12 = load.i32 notrap aligned v0+0x34\n" ++
  "  v13 = call fn0(v0, v12, v1, v2)\n" ++
  "  return\n" ++
  "}\n"

def clifIR : String := clifNoopFn ++ "\n" ++ clifLoadFn ++ "\n" ++ clifPrepFn ++ "\n" ++ clifInferFn

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [0, 0, 0, 0, 1, 0, 0, 0]

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

end CudaRmsNormPersist

def main : IO Unit := do
  let json := Json.arr #[
    toJson CudaRmsNormPersist.buildConfig,
    toJson CudaRmsNormPersist.loadAlgorithm,
    toJson CudaRmsNormPersist.prepAlgorithm,
    toJson CudaRmsNormPersist.inferAlgorithm
  ]
  IO.println (Json.compress json)
