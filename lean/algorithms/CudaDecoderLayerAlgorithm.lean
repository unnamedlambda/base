import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CudaDecoderLayer

/-!
  Persistent single-token decoder-layer benchmark.

  Fixed dimensions:
    d_model = 896
    d_ff    = 4864

  Load payload layout:
    rms1[d_model]
    wq[d_model, d_model]
    wk[d_model, d_model]
    wv[d_model, d_model]
    wo[d_model, d_model]
    rms2[d_model]
    wg[d_ff, d_model]
    wu[d_ff, d_model]
    wd[d_model, d_ff]

  Infer payload layout:
    x[d_model]

  Output:
    y[d_model]

  Attention is simplified to the seq_len=1 case, so the attention output is v.
  We still compute q and k so the projection cost matches a real decoder layer.
-/

def D_MODEL : Nat := 896
def D_FF    : Nat := 4864

def D_MODEL_BYTES : Nat := D_MODEL * 4
def D_FF_BYTES    : Nat := D_FF * 4
def W_DM_DM_BYTES : Nat := D_MODEL * D_MODEL * 4
def W_FF_DM_BYTES : Nat := D_FF * D_MODEL * 4
def W_DM_FF_BYTES : Nat := D_MODEL * D_FF * 4

def PTX_RMS_OFF : Nat := 0x0800
def PTX_SILU_OFF : Nat := 0x1C00
def PTX_ADDRMS_OFF : Nat := 0x2800
def PTX_ADD_OFF : Nat := 0x3800
def MEM_SIZE : Nat := 0x4800
def TIMEOUT_MS : Nat := 30000

def BUF_X_OFF      : Nat := 0x28
def BUF_XN1_OFF    : Nat := 0x2C
def BUF_Q_OFF      : Nat := 0x30
def BUF_K_OFF      : Nat := 0x34
def BUF_V_OFF      : Nat := 0x38
def BUF_O_OFF      : Nat := 0x3C
def BUF_XN2_OFF    : Nat := 0x40
def BUF_G_OFF      : Nat := 0x44
def BUF_U_OFF      : Nat := 0x48
def BUF_A_OFF      : Nat := 0x4C
def BUF_D_OFF      : Nat := 0x50
def BUF_RMS1_OFF   : Nat := 0x54
def BUF_WQ_OFF     : Nat := 0x58
def BUF_WK_OFF     : Nat := 0x5C
def BUF_WV_OFF     : Nat := 0x60
def BUF_WO_OFF     : Nat := 0x64
def BUF_RMS2_OFF   : Nat := 0x68
def BUF_WG_OFF     : Nat := 0x6C
def BUF_WU_OFF     : Nat := 0x70
def BUF_WD_OFF     : Nat := 0x74

def BIND_RMS1_OFF  : Nat := 0x100
def BIND_ADDRMS_OFF : Nat := 0x110
def BIND_SILU_OFF  : Nat := 0x130
def BIND_ADD2_OFF  : Nat := 0x140

def ptxRmsNorm : String :=
  ".version 8.0\n" ++
  ".target sm_86\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".shared .align 4 .b8 _smem[36];\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 x_ptr,\n" ++
  "    .param .u64 w_ptr,\n" ++
  "    .param .u64 y_ptr\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred   %p;\n" ++
  "    .reg .u32    %r<12>;\n" ++
  "    .reg .u64    %rd<10>;\n" ++
  "    .reg .f32    %f<10>;\n" ++
  "\n" ++
  "    ld.param.u64  %rd0, [x_ptr];\n" ++
  "    ld.param.u64  %rd1, [w_ptr];\n" ++
  "    ld.param.u64  %rd2, [y_ptr];\n" ++
  "    mov.u32       %r0, " ++ toString D_MODEL ++ ";\n" ++
  "    mov.u32       %r1, %tid.x;\n" ++
  "    shr.u32       %r2, %r1, 5;\n" ++
  "    and.b32       %r3, %r1, 31;\n" ++
  "    mov.f32       %f0, 0f00000000;\n" ++
  "    mov.u32       %r4, %r1;\n" ++
  "loop1:\n" ++
  "    setp.ge.u32   %p, %r4, %r0;\n" ++
  "    @%p bra       done1;\n" ++
  "    cvt.u64.u32   %rd3, %r4;\n" ++
  "    shl.b64       %rd3, %rd3, 2;\n" ++
  "    add.u64       %rd4, %rd0, %rd3;\n" ++
  "    ld.global.f32 %f1, [%rd4];\n" ++
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
  "    mov.f32       %f1, 0f44600000;\n" ++
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
  "    cvt.u64.u32   %rd3, %r4;\n" ++
  "    shl.b64       %rd3, %rd3, 2;\n" ++
  "    add.u64       %rd4, %rd0, %rd3;\n" ++
  "    ld.global.f32 %f4, [%rd4];\n" ++
  "    add.u64       %rd5, %rd1, %rd3;\n" ++
  "    ld.global.f32 %f5, [%rd5];\n" ++
  "    mul.f32       %f4, %f4, %f3;\n" ++
  "    mul.f32       %f4, %f4, %f5;\n" ++
  "    add.u64       %rd6, %rd2, %rd3;\n" ++
  "    st.global.f32 [%rd6], %f4;\n" ++
  "    add.u32       %r4, %r4, 256;\n" ++
  "    bra           loop2;\n" ++
  "done2:\n" ++
  "    ret;\n" ++
  "}\n"

def ptxSiluGate : String :=
  ".version 8.0\n" ++
  ".target sm_86\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 gate_ptr,\n" ++
  "    .param .u64 up_ptr,\n" ++
  "    .param .u64 out_ptr\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred   %p;\n" ++
  "    .reg .u32    %r<6>;\n" ++
  "    .reg .u64    %rd<7>;\n" ++
  "    .reg .f32    %f<8>;\n" ++
  "\n" ++
  "    ld.param.u64  %rd0, [gate_ptr];\n" ++
  "    ld.param.u64  %rd1, [up_ptr];\n" ++
  "    ld.param.u64  %rd2, [out_ptr];\n" ++
  "    mov.u32       %r0, %ctaid.x;\n" ++
  "    mov.u32       %r1, %ntid.x;\n" ++
  "    mov.u32       %r2, %tid.x;\n" ++
  "    mad.lo.s32    %r3, %r0, %r1, %r2;\n" ++
  "    setp.ge.u32   %p, %r3, " ++ toString D_FF ++ ";\n" ++
  "    @%p bra       done;\n" ++
  "    cvt.u64.u32   %rd3, %r3;\n" ++
  "    shl.b64       %rd3, %rd3, 2;\n" ++
  "    add.u64       %rd4, %rd0, %rd3;\n" ++
  "    add.u64       %rd5, %rd1, %rd3;\n" ++
  "    add.u64       %rd6, %rd2, %rd3;\n" ++
  "    ld.global.f32 %f0, [%rd4];\n" ++
  "    ld.global.f32 %f1, [%rd5];\n" ++
  "    neg.f32       %f2, %f0;\n" ++
  "    mov.f32       %f3, 0f3fb8aa3b;\n" ++
  "    mul.f32       %f2, %f2, %f3;\n" ++
  "    ex2.approx.f32 %f2, %f2;\n" ++
  "    mov.f32       %f4, 0f3f800000;\n" ++
  "    add.f32       %f2, %f2, %f4;\n" ++
  "    rcp.approx.f32 %f2, %f2;\n" ++
  "    mul.f32       %f0, %f0, %f2;\n" ++
  "    mul.f32       %f0, %f0, %f1;\n" ++
  "    st.global.f32 [%rd6], %f0;\n" ++
  "done:\n" ++
  "    ret;\n" ++
  "}\n"

def ptxResidualAdd : String :=
  ".version 8.0\n" ++
  ".target sm_86\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 x_ptr,\n" ++
  "    .param .u64 add_ptr\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred   %p;\n" ++
  "    .reg .u32    %r<6>;\n" ++
  "    .reg .u64    %rd<6>;\n" ++
  "    .reg .f32    %f<4>;\n" ++
  "\n" ++
  "    ld.param.u64  %rd0, [x_ptr];\n" ++
  "    ld.param.u64  %rd1, [add_ptr];\n" ++
  "    mov.u32       %r0, %ctaid.x;\n" ++
  "    mov.u32       %r1, %ntid.x;\n" ++
  "    mov.u32       %r2, %tid.x;\n" ++
  "    mad.lo.s32    %r3, %r0, %r1, %r2;\n" ++
  "    setp.ge.u32   %p, %r3, " ++ toString D_MODEL ++ ";\n" ++
  "    @%p bra       done;\n" ++
  "    cvt.u64.u32   %rd2, %r3;\n" ++
  "    shl.b64       %rd2, %rd2, 2;\n" ++
  "    add.u64       %rd3, %rd0, %rd2;\n" ++
  "    add.u64       %rd4, %rd1, %rd2;\n" ++
  "    ld.global.f32 %f0, [%rd3];\n" ++
  "    ld.global.f32 %f1, [%rd4];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    st.global.f32 [%rd3], %f0;\n" ++
  "done:\n" ++
  "    ret;\n" ++
  "}\n"

def ptxAddRmsNorm : String :=
  ".version 8.0\n" ++
  ".target sm_86\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".shared .align 4 .b8 _smem[36];\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 x_ptr,\n" ++
  "    .param .u64 add_ptr,\n" ++
  "    .param .u64 w_ptr,\n" ++
  "    .param .u64 y_ptr\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred   %p;\n" ++
  "    .reg .u32    %r<12>;\n" ++
  "    .reg .u64    %rd<11>;\n" ++
  "    .reg .f32    %f<10>;\n" ++
  "\n" ++
  "    ld.param.u64  %rd0, [x_ptr];\n" ++
  "    ld.param.u64  %rd1, [add_ptr];\n" ++
  "    ld.param.u64  %rd2, [w_ptr];\n" ++
  "    ld.param.u64  %rd3, [y_ptr];\n" ++
  "    mov.u32       %r0, " ++ toString D_MODEL ++ ";\n" ++
  "    mov.u32       %r1, %tid.x;\n" ++
  "    shr.u32       %r2, %r1, 5;\n" ++
  "    and.b32       %r3, %r1, 31;\n" ++
  "    mov.f32       %f0, 0f00000000;\n" ++
  "    mov.u32       %r4, %r1;\n" ++
  "loop1:\n" ++
  "    setp.ge.u32   %p, %r4, %r0;\n" ++
  "    @%p bra       done1;\n" ++
  "    cvt.u64.u32   %rd4, %r4;\n" ++
  "    shl.b64       %rd4, %rd4, 2;\n" ++
  "    add.u64       %rd5, %rd0, %rd4;\n" ++
  "    add.u64       %rd6, %rd1, %rd4;\n" ++
  "    ld.global.f32 %f1, [%rd5];\n" ++
  "    ld.global.f32 %f2, [%rd6];\n" ++
  "    add.f32       %f1, %f1, %f2;\n" ++
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
  "    mov.f32       %f1, 0f44600000;\n" ++
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
  "    cvt.u64.u32   %rd4, %r4;\n" ++
  "    shl.b64       %rd4, %rd4, 2;\n" ++
  "    add.u64       %rd5, %rd0, %rd4;\n" ++
  "    add.u64       %rd6, %rd1, %rd4;\n" ++
  "    add.u64       %rd7, %rd2, %rd4;\n" ++
  "    add.u64       %rd8, %rd3, %rd4;\n" ++
  "    ld.global.f32 %f4, [%rd5];\n" ++
  "    ld.global.f32 %f5, [%rd6];\n" ++
  "    add.f32       %f4, %f4, %f5;\n" ++
  "    st.global.f32 [%rd5], %f4;\n" ++
  "    ld.global.f32 %f6, [%rd7];\n" ++
  "    mul.f32       %f4, %f4, %f3;\n" ++
  "    mul.f32       %f4, %f4, %f6;\n" ++
  "    st.global.f32 [%rd8], %f4;\n" ++
  "    add.u32       %r4, %r4, 256;\n" ++
  "    bra           loop2;\n" ++
  "done2:\n" ++
  "    ret;\n" ++
  "}\n"

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
  -- activations
  "  v10 = iconst.i64 " ++ toString D_MODEL_BYTES ++ "\n" ++
  "  v11 = iconst.i64 " ++ toString D_FF_BYTES ++ "\n" ++
  "  v20 = call fn1(v0, v10)\n" ++
  "  v21 = call fn1(v0, v10)\n" ++
  "  v22 = call fn1(v0, v10)\n" ++
  "  v23 = call fn1(v0, v10)\n" ++
  "  v24 = call fn1(v0, v10)\n" ++
  "  v25 = call fn1(v0, v10)\n" ++
  "  v26 = call fn1(v0, v10)\n" ++
  "  v27 = call fn1(v0, v11)\n" ++
  "  v28 = call fn1(v0, v11)\n" ++
  "  v29 = call fn1(v0, v11)\n" ++
  "  v30 = call fn1(v0, v10)\n" ++
  -- weights
  "  v40 = call fn1(v0, v10)\n" ++
  "  v41 = iconst.i64 " ++ toString W_DM_DM_BYTES ++ "\n" ++
  "  v42 = call fn1(v0, v41)\n" ++
  "  v43 = call fn1(v0, v41)\n" ++
  "  v44 = call fn1(v0, v41)\n" ++
  "  v45 = call fn1(v0, v41)\n" ++
  "  v46 = call fn1(v0, v10)\n" ++
  "  v47 = iconst.i64 " ++ toString W_FF_DM_BYTES ++ "\n" ++
  "  v48 = call fn1(v0, v47)\n" ++
  "  v49 = call fn1(v0, v47)\n" ++
  "  v50 = iconst.i64 " ++ toString W_DM_FF_BYTES ++ "\n" ++
  "  v51 = call fn1(v0, v50)\n" ++
  -- store ids
  "  store notrap aligned v20, v0+" ++ toString BUF_X_OFF ++ "\n" ++
  "  store notrap aligned v21, v0+" ++ toString BUF_XN1_OFF ++ "\n" ++
  "  store notrap aligned v22, v0+" ++ toString BUF_Q_OFF ++ "\n" ++
  "  store notrap aligned v23, v0+" ++ toString BUF_K_OFF ++ "\n" ++
  "  store notrap aligned v24, v0+" ++ toString BUF_V_OFF ++ "\n" ++
  "  store notrap aligned v25, v0+" ++ toString BUF_O_OFF ++ "\n" ++
  "  store notrap aligned v26, v0+" ++ toString BUF_XN2_OFF ++ "\n" ++
  "  store notrap aligned v27, v0+" ++ toString BUF_G_OFF ++ "\n" ++
  "  store notrap aligned v28, v0+" ++ toString BUF_U_OFF ++ "\n" ++
  "  store notrap aligned v29, v0+" ++ toString BUF_A_OFF ++ "\n" ++
  "  store notrap aligned v30, v0+" ++ toString BUF_D_OFF ++ "\n" ++
  "  store notrap aligned v40, v0+" ++ toString BUF_RMS1_OFF ++ "\n" ++
  "  store notrap aligned v42, v0+" ++ toString BUF_WQ_OFF ++ "\n" ++
  "  store notrap aligned v43, v0+" ++ toString BUF_WK_OFF ++ "\n" ++
  "  store notrap aligned v44, v0+" ++ toString BUF_WV_OFF ++ "\n" ++
  "  store notrap aligned v45, v0+" ++ toString BUF_WO_OFF ++ "\n" ++
  "  store notrap aligned v46, v0+" ++ toString BUF_RMS2_OFF ++ "\n" ++
  "  store notrap aligned v48, v0+" ++ toString BUF_WG_OFF ++ "\n" ++
  "  store notrap aligned v49, v0+" ++ toString BUF_WU_OFF ++ "\n" ++
  "  store notrap aligned v51, v0+" ++ toString BUF_WD_OFF ++ "\n" ++
  -- upload weights
  "  v60 = call fn2(v0, v40, v1, v10)\n" ++
  "  v61 = iadd_imm v1, " ++ toString D_MODEL_BYTES ++ "\n" ++
  "  v62 = call fn2(v0, v42, v61, v41)\n" ++
  "  v63 = iadd_imm v61, " ++ toString W_DM_DM_BYTES ++ "\n" ++
  "  v64 = call fn2(v0, v43, v63, v41)\n" ++
  "  v65 = iadd_imm v63, " ++ toString W_DM_DM_BYTES ++ "\n" ++
  "  v66 = call fn2(v0, v44, v65, v41)\n" ++
  "  v67 = iadd_imm v65, " ++ toString W_DM_DM_BYTES ++ "\n" ++
  "  v68 = call fn2(v0, v45, v67, v41)\n" ++
  "  v69 = iadd_imm v67, " ++ toString W_DM_DM_BYTES ++ "\n" ++
  "  v70 = call fn2(v0, v46, v69, v10)\n" ++
  "  v71 = iadd_imm v69, " ++ toString D_MODEL_BYTES ++ "\n" ++
  "  v72 = call fn2(v0, v48, v71, v47)\n" ++
  "  v73 = iadd_imm v71, " ++ toString W_FF_DM_BYTES ++ "\n" ++
  "  v74 = call fn2(v0, v49, v73, v47)\n" ++
  "  v75 = iadd_imm v73, " ++ toString W_FF_DM_BYTES ++ "\n" ++
  "  v76 = call fn2(v0, v51, v75, v50)\n" ++
  "  return\n" ++
  "}\n"

def clifPrepFn : String :=
  "function u0:2(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_upload_ptr sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  v2 = load.i32 notrap aligned v0+" ++ toString BUF_X_OFF ++ "\n" ++
  "  v3 = iconst.i64 " ++ toString D_MODEL_BYTES ++ "\n" ++
  "  v4 = call fn0(v0, v2, v1, v3)\n" ++
  "  return\n" ++
  "}\n"

def clifInferFn : String :=
  "function u0:3(i64) system_v {\n" ++
  "    sig0 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v\n" ++
  "    sig1 = (i64, i32, i32, i32, i32, i32, i32, i32, i32) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_launch sig0\n" ++
  "    fn1 = %cl_cublas_sgemv sig1\n" ++
"block0(v0: i64):\n" ++
  "  v10 = load.i32 notrap aligned v0+" ++ toString BUF_X_OFF ++ "\n" ++
  "  v11 = load.i32 notrap aligned v0+" ++ toString BUF_XN1_OFF ++ "\n" ++
  "  v12 = load.i32 notrap aligned v0+" ++ toString BUF_Q_OFF ++ "\n" ++
  "  v13 = load.i32 notrap aligned v0+" ++ toString BUF_K_OFF ++ "\n" ++
  "  v14 = load.i32 notrap aligned v0+" ++ toString BUF_V_OFF ++ "\n" ++
  "  v15 = load.i32 notrap aligned v0+" ++ toString BUF_O_OFF ++ "\n" ++
  "  v16 = load.i32 notrap aligned v0+" ++ toString BUF_XN2_OFF ++ "\n" ++
  "  v17 = load.i32 notrap aligned v0+" ++ toString BUF_G_OFF ++ "\n" ++
  "  v18 = load.i32 notrap aligned v0+" ++ toString BUF_U_OFF ++ "\n" ++
  "  v19 = load.i32 notrap aligned v0+" ++ toString BUF_A_OFF ++ "\n" ++
  "  v20 = load.i32 notrap aligned v0+" ++ toString BUF_D_OFF ++ "\n" ++
  "  v21 = load.i32 notrap aligned v0+" ++ toString BUF_RMS1_OFF ++ "\n" ++
  "  v22 = load.i32 notrap aligned v0+" ++ toString BUF_WQ_OFF ++ "\n" ++
  "  v23 = load.i32 notrap aligned v0+" ++ toString BUF_WK_OFF ++ "\n" ++
  "  v24 = load.i32 notrap aligned v0+" ++ toString BUF_WV_OFF ++ "\n" ++
  "  v25 = load.i32 notrap aligned v0+" ++ toString BUF_WO_OFF ++ "\n" ++
  "  v26 = load.i32 notrap aligned v0+" ++ toString BUF_RMS2_OFF ++ "\n" ++
  "  v27 = load.i32 notrap aligned v0+" ++ toString BUF_WG_OFF ++ "\n" ++
  "  v28 = load.i32 notrap aligned v0+" ++ toString BUF_WU_OFF ++ "\n" ++
  "  v29 = load.i32 notrap aligned v0+" ++ toString BUF_WD_OFF ++ "\n" ++
  "  v30 = iconst.i64 " ++ toString D_MODEL_BYTES ++ "\n" ++
  -- rms1 bind + launch
  "  store notrap aligned v10, v0+" ++ toString BIND_RMS1_OFF ++ "\n" ++
  "  store notrap aligned v21, v0+" ++ toString (BIND_RMS1_OFF + 4) ++ "\n" ++
  "  store notrap aligned v11, v0+" ++ toString (BIND_RMS1_OFF + 8) ++ "\n" ++
  "  v32 = iconst.i64 " ++ toString PTX_RMS_OFF ++ "\n" ++
  "  v33 = iconst.i32 3\n" ++
  "  v34 = iconst.i64 " ++ toString BIND_RMS1_OFF ++ "\n" ++
  "  v35 = iconst.i32 1\n" ++
  "  v36 = iconst.i32 256\n" ++
  "  v37 = call fn0(v0, v32, v33, v34, v35, v35, v35, v36, v35, v35)\n" ++
  -- q/k/v/o
  "  v40 = iconst.i32 1\n" ++
  "  v41 = iconst.i32 " ++ toString D_MODEL ++ "\n" ++
  "  v42 = iconst.i32 0x3f800000\n" ++
  "  v43 = iconst.i32 0\n" ++
  "  v44 = call fn1(v0, v40, v41, v41, v42, v22, v11, v43, v12)\n" ++
  "  v45 = call fn1(v0, v40, v41, v41, v42, v23, v11, v43, v13)\n" ++
  "  v46 = call fn1(v0, v40, v41, v41, v42, v24, v11, v43, v14)\n" ++
  "  v47 = call fn1(v0, v40, v41, v41, v42, v25, v14, v43, v15)\n" ++
  -- fused: x += o and rmsnorm(x, rms2) -> xn2
  "  store notrap aligned v10, v0+" ++ toString BIND_ADDRMS_OFF ++ "\n" ++
  "  store notrap aligned v15, v0+" ++ toString (BIND_ADDRMS_OFF + 4) ++ "\n" ++
  "  v48 = iconst.i64 " ++ toString PTX_ADD_OFF ++ "\n" ++
  "  v49 = iconst.i32 2\n" ++
  "  v50 = iconst.i64 " ++ toString BIND_ADDRMS_OFF ++ "\n" ++
  "  v51 = iconst.i32 4\n" ++
  "  v52 = call fn0(v0, v48, v49, v50, v51, v35, v35, v36, v35, v35)\n" ++
  -- rms2
  "  store notrap aligned v10, v0+" ++ toString (BIND_ADDRMS_OFF + 16) ++ "\n" ++
  "  store notrap aligned v26, v0+" ++ toString (BIND_ADDRMS_OFF + 20) ++ "\n" ++
  "  store notrap aligned v16, v0+" ++ toString (BIND_ADDRMS_OFF + 24) ++ "\n" ++
  "  v53 = iconst.i64 " ++ toString (BIND_ADDRMS_OFF + 16) ++ "\n" ++
  "  v54 = call fn0(v0, v32, v33, v53, v35, v35, v35, v36, v35, v35)\n" ++
  -- gate/up
  "  v55 = iconst.i32 " ++ toString D_FF ++ "\n" ++
  "  v56 = call fn1(v0, v40, v41, v55, v42, v27, v16, v43, v17)\n" ++
  "  v57 = call fn1(v0, v40, v41, v55, v42, v28, v16, v43, v18)\n" ++
  -- silu_gate
  "  store notrap aligned v17, v0+" ++ toString BIND_SILU_OFF ++ "\n" ++
  "  store notrap aligned v18, v0+" ++ toString (BIND_SILU_OFF + 4) ++ "\n" ++
  "  store notrap aligned v19, v0+" ++ toString (BIND_SILU_OFF + 8) ++ "\n" ++
  "  v58 = iconst.i64 " ++ toString PTX_SILU_OFF ++ "\n" ++
  "  v59 = iconst.i64 " ++ toString BIND_SILU_OFF ++ "\n" ++
  "  v60 = iconst.i32 19\n" ++
  "  v61 = call fn0(v0, v58, v33, v59, v60, v35, v35, v36, v35, v35)\n" ++
  -- down + residual
  "  v62 = call fn1(v0, v40, v55, v41, v42, v29, v19, v43, v20)\n" ++
  "  store notrap aligned v10, v0+" ++ toString BIND_ADD2_OFF ++ "\n" ++
  "  store notrap aligned v20, v0+" ++ toString (BIND_ADD2_OFF + 4) ++ "\n" ++
  "  v63 = iconst.i64 " ++ toString BIND_ADD2_OFF ++ "\n" ++
  "  v64 = call fn0(v0, v48, v49, v63, v51, v35, v35, v36, v35, v35)\n" ++
  "  return\n" ++
  "}\n"

def clifFinalizeFn : String :=
  "function u0:4(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    sig1 = (i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_download_ptr sig0\n" ++
  "    fn1 = %cl_cuda_sync sig1\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x18\n" ++
  "  v2 = load.i64 notrap aligned v0+0x20\n" ++
  "  v3 = load.i32 notrap aligned v0+" ++ toString BUF_X_OFF ++ "\n" ++
  "  v4 = call fn1(v0)\n" ++
  "  v5 = iconst.i64 0\n" ++
  "  v6 = icmp eq v2, v5\n" ++
  "  brif v6, block1, block2\n" ++
  "block1:\n" ++
  "  return\n" ++
  "block2:\n" ++
  "  v7 = call fn0(v0, v3, v1, v2)\n" ++
  "  return\n" ++
  "}\n"

def clifIR : String := clifNoopFn ++ "\n" ++ clifLoadFn ++ "\n" ++ clifPrepFn ++ "\n" ++ clifInferFn ++ "\n" ++ clifFinalizeFn

def ptxRmsBytes : List UInt8 := ptxRmsNorm.toUTF8.toList ++ [0]
def ptxSiluBytes : List UInt8 := ptxSiluGate.toUTF8.toList ++ [0]
def ptxAddRmsBytes : List UInt8 := ptxAddRmsNorm.toUTF8.toList ++ [0]
def ptxAddBytes : List UInt8 := ptxResidualAdd.toUTF8.toList ++ [0]

def buildInitialMemory : List UInt8 :=
  let pre := zeros PTX_RMS_OFF
  let rms := ptxRmsBytes ++ zeros (PTX_SILU_OFF - PTX_RMS_OFF - ptxRmsBytes.length)
  let silu := ptxSiluBytes ++ zeros (PTX_ADDRMS_OFF - PTX_SILU_OFF - ptxSiluBytes.length)
  let addrms := ptxAddRmsBytes ++ zeros (PTX_ADD_OFF - PTX_ADDRMS_OFF - ptxAddRmsBytes.length)
  let add := ptxAddBytes ++ zeros (MEM_SIZE - PTX_ADD_OFF - ptxAddBytes.length)
  pre ++ rms ++ silu ++ addrms ++ add

def actions (src : UInt32) : List Action :=
  [{ kind := .ClifCall, dst := 0, src := src, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0,
  initial_memory := buildInitialMemory
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
  actions := [{ kind := .ClifCall, dst := 0, src := 3, offset := 0, size := 0 },
              { kind := .ClifCall, dst := 0, src := 4, offset := 0, size := 0 }],
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

def stackAlgorithm (depth : Nat) : Algorithm := {
  actions := List.replicate depth { kind := .ClifCall, dst := 0, src := 3, offset := 0, size := 0 } ++
             [{ kind := .ClifCall, dst := 0, src := 4, offset := 0, size := 0 }],
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

end CudaDecoderLayer

def main : IO Unit := do
  let json := Json.arr #[
    toJson CudaDecoderLayer.buildConfig,
    toJson CudaDecoderLayer.loadAlgorithm,
    toJson CudaDecoderLayer.prepAlgorithm,
    toJson CudaDecoderLayer.inferAlgorithm,
    toJson (CudaDecoderLayer.stackAlgorithm 16),
    toJson (CudaDecoderLayer.stackAlgorithm 32)
  ]
  IO.println (Json.compress json)
