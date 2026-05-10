import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

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

-- App fields: buf IDs stored as i32 (4 bytes each), starting at 0x38
-- (beyond the 56-byte RuntimeHeader at 0x00-0x37)
def BUF_X_OFF      : Nat := 0x38
def BUF_XN1_OFF    : Nat := 0x3C
def BUF_Q_OFF      : Nat := 0x40
def BUF_K_OFF      : Nat := 0x44
def BUF_V_OFF      : Nat := 0x48
def BUF_O_OFF      : Nat := 0x4C
def BUF_XN2_OFF    : Nat := 0x50
def BUF_G_OFF      : Nat := 0x54
def BUF_U_OFF      : Nat := 0x58
def BUF_A_OFF      : Nat := 0x5C
def BUF_D_OFF      : Nat := 0x60
def BUF_RMS1_OFF   : Nat := 0x64
def BUF_WQ_OFF     : Nat := 0x68
def BUF_WK_OFF     : Nat := 0x6C
def BUF_WV_OFF     : Nat := 0x70
def BUF_WO_OFF     : Nat := 0x74
def BUF_RMS2_OFF   : Nat := 0x78
def BUF_WG_OFF     : Nat := 0x7C
def BUF_WU_OFF     : Nat := 0x80
def BUF_WD_OFF     : Nat := 0x84

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

def loadFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit cuda ptr 0x10
  let ctxPtr  ← load64 (← absAddr ptr 0x10)

  let dmBytes  ← iconst64 D_MODEL_BYTES
  let ffBytes  ← iconst64 D_FF_BYTES
  let wdmBytes ← iconst64 W_DM_DM_BYTES
  let wffBytes ← iconst64 W_FF_DM_BYTES
  let wdfBytes ← iconst64 W_DM_FF_BYTES

  -- activation buffers
  let bufX   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufXn1 ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufQ   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufK   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufV   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufO   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufXn2 ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufG   ← call cuda.fnCreateBuffer [ctxPtr, ffBytes]
  let bufU   ← call cuda.fnCreateBuffer [ctxPtr, ffBytes]
  let bufA   ← call cuda.fnCreateBuffer [ctxPtr, ffBytes]
  let bufD   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  -- weight buffers
  let bufRms1 ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufWq   ← call cuda.fnCreateBuffer [ctxPtr, wdmBytes]
  let bufWk   ← call cuda.fnCreateBuffer [ctxPtr, wdmBytes]
  let bufWv   ← call cuda.fnCreateBuffer [ctxPtr, wdmBytes]
  let bufWo   ← call cuda.fnCreateBuffer [ctxPtr, wdmBytes]
  let bufRms2 ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufWg   ← call cuda.fnCreateBuffer [ctxPtr, wffBytes]
  let bufWu   ← call cuda.fnCreateBuffer [ctxPtr, wffBytes]
  let bufWd   ← call cuda.fnCreateBuffer [ctxPtr, wdfBytes]

  storeI32 bufX   (← absAddr ptr BUF_X_OFF)
  storeI32 bufXn1 (← absAddr ptr BUF_XN1_OFF)
  storeI32 bufQ   (← absAddr ptr BUF_Q_OFF)
  storeI32 bufK   (← absAddr ptr BUF_K_OFF)
  storeI32 bufV   (← absAddr ptr BUF_V_OFF)
  storeI32 bufO   (← absAddr ptr BUF_O_OFF)
  storeI32 bufXn2 (← absAddr ptr BUF_XN2_OFF)
  storeI32 bufG   (← absAddr ptr BUF_G_OFF)
  storeI32 bufU   (← absAddr ptr BUF_U_OFF)
  storeI32 bufA   (← absAddr ptr BUF_A_OFF)
  storeI32 bufD   (← absAddr ptr BUF_D_OFF)
  storeI32 bufRms1 (← absAddr ptr BUF_RMS1_OFF)
  storeI32 bufWq  (← absAddr ptr BUF_WQ_OFF)
  storeI32 bufWk  (← absAddr ptr BUF_WK_OFF)
  storeI32 bufWv  (← absAddr ptr BUF_WV_OFF)
  storeI32 bufWo  (← absAddr ptr BUF_WO_OFF)
  storeI32 bufRms2 (← absAddr ptr BUF_RMS2_OFF)
  storeI32 bufWg  (← absAddr ptr BUF_WG_OFF)
  storeI32 bufWu  (← absAddr ptr BUF_WU_OFF)
  storeI32 bufWd  (← absAddr ptr BUF_WD_OFF)

  -- upload weights (rms1, wq, wk, wv, wo, rms2, wg, wu, wd)
  let _ ← call cuda.fnUpload [ctxPtr, bufRms1, dataPtr, dmBytes]
  let p1 ← iaddImm dataPtr D_MODEL_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWq, p1, wdmBytes]
  let p2 ← iaddImm p1 W_DM_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWk, p2, wdmBytes]
  let p3 ← iaddImm p2 W_DM_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWv, p3, wdmBytes]
  let p4 ← iaddImm p3 W_DM_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWo, p4, wdmBytes]
  let p5 ← iaddImm p4 W_DM_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufRms2, p5, dmBytes]
  let p6 ← iaddImm p5 D_MODEL_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWg, p6, wffBytes]
  let p7 ← iaddImm p6 W_FF_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWu, p7, wffBytes]
  let p8 ← iaddImm p7 W_FF_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWd, p8, wdfBytes]
  ret

def prepFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let ctxPtr  ← load64 (← absAddr ptr 0x10)
  let bufX    ← load32 (← absAddr ptr BUF_X_OFF)
  let dmBytes ← iconst64 D_MODEL_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufX, dataPtr, dmBytes]
  ret

def inferFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let blas   ← declareCuBlasFFI
  let ctxPtr ← load64 (← absAddr ptr 0x10)

  -- load all 20 buf IDs
  let bufX    ← load32 (← absAddr ptr BUF_X_OFF)
  let bufXn1  ← load32 (← absAddr ptr BUF_XN1_OFF)
  let bufQ    ← load32 (← absAddr ptr BUF_Q_OFF)
  let bufK    ← load32 (← absAddr ptr BUF_K_OFF)
  let bufV    ← load32 (← absAddr ptr BUF_V_OFF)
  let bufO    ← load32 (← absAddr ptr BUF_O_OFF)
  let bufXn2  ← load32 (← absAddr ptr BUF_XN2_OFF)
  let bufG    ← load32 (← absAddr ptr BUF_G_OFF)
  let bufU    ← load32 (← absAddr ptr BUF_U_OFF)
  let bufA    ← load32 (← absAddr ptr BUF_A_OFF)
  let bufD    ← load32 (← absAddr ptr BUF_D_OFF)
  let bufRms1 ← load32 (← absAddr ptr BUF_RMS1_OFF)
  let bufWq   ← load32 (← absAddr ptr BUF_WQ_OFF)
  let bufWk   ← load32 (← absAddr ptr BUF_WK_OFF)
  let bufWv   ← load32 (← absAddr ptr BUF_WV_OFF)
  let bufWo   ← load32 (← absAddr ptr BUF_WO_OFF)
  let bufRms2 ← load32 (← absAddr ptr BUF_RMS2_OFF)
  let bufWg   ← load32 (← absAddr ptr BUF_WG_OFF)
  let bufWu   ← load32 (← absAddr ptr BUF_WU_OFF)
  let bufWd   ← load32 (← absAddr ptr BUF_WD_OFF)

  let one32   ← iconst32 1
  let three32 ← iconst32 3
  let two32   ← iconst32 2
  let four32  ← iconst32 4
  let blk256  ← iconst32 256
  let dm32    ← iconst32 D_MODEL
  let ff32    ← iconst32 D_FF
  let alpha   ← iconst32 0x3f800000
  let zero32  ← iconst32 0

  -- rms1: normalize x with rms1 weights → xn1
  storeI32 bufX   (← absAddr ptr BIND_RMS1_OFF)
  storeI32 bufRms1 (← absAddr ptr (BIND_RMS1_OFF + 4))
  storeI32 bufXn1 (← absAddr ptr (BIND_RMS1_OFF + 8))
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_RMS_OFF) three32
             (← iconst64 BIND_RMS1_OFF) one32 one32 one32 blk256 one32 one32

  -- attention projections: q = WQ @ xn1, k = WK @ xn1, v = WV @ xn1, o = WO @ v
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, dm32, alpha, bufWq, bufXn1, zero32, bufQ]
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, dm32, alpha, bufWk, bufXn1, zero32, bufK]
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, dm32, alpha, bufWv, bufXn1, zero32, bufV]
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, dm32, alpha, bufWo, bufV,   zero32, bufO]

  -- residual add: x += o
  storeI32 bufX (← absAddr ptr BIND_ADDRMS_OFF)
  storeI32 bufO (← absAddr ptr (BIND_ADDRMS_OFF + 4))
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_ADD_OFF) two32
             (← iconst64 BIND_ADDRMS_OFF) four32 one32 one32 blk256 one32 one32

  -- rms2: normalize x with rms2 weights → xn2
  storeI32 bufX    (← absAddr ptr (BIND_ADDRMS_OFF + 16))
  storeI32 bufRms2 (← absAddr ptr (BIND_ADDRMS_OFF + 20))
  storeI32 bufXn2  (← absAddr ptr (BIND_ADDRMS_OFF + 24))
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_RMS_OFF) three32
             (← iconst64 (BIND_ADDRMS_OFF + 16)) one32 one32 one32 blk256 one32 one32

  -- FFN: gate = WG @ xn2, up = WU @ xn2
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, ff32, alpha, bufWg, bufXn2, zero32, bufG]
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, ff32, alpha, bufWu, bufXn2, zero32, bufU]

  -- SiLU-gate: a = silu(g) * u
  storeI32 bufG (← absAddr ptr BIND_SILU_OFF)
  storeI32 bufU (← absAddr ptr (BIND_SILU_OFF + 4))
  storeI32 bufA (← absAddr ptr (BIND_SILU_OFF + 8))
  let nineteen32 ← iconst32 19
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_SILU_OFF) three32
             (← iconst64 BIND_SILU_OFF) nineteen32 one32 one32 blk256 one32 one32

  -- down projection: d = WD @ a
  let _ ← call blas.fnSgemv [ctxPtr, one32, ff32, dm32, alpha, bufWd, bufA, zero32, bufD]

  -- residual add: x += d
  storeI32 bufX (← absAddr ptr BIND_ADD2_OFF)
  storeI32 bufD (← absAddr ptr (BIND_ADD2_OFF + 4))
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_ADD_OFF) two32
             (← iconst64 BIND_ADD2_OFF) four32 one32 one32 blk256 one32 one32
  ret

def finalizeFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outLen ← load64 (← absAddr ptr 0x30)
  let ctxPtr ← load64 (← absAddr ptr 0x10)
  let bufX   ← load32 (← absAddr ptr BUF_X_OFF)

  let skipDl     ← declareBlock []
  let doDownload ← declareBlock []

  let _ ← cudaSync cuda ptr 0x10
  brif (← icmpImm .eq outLen 0) skipDl.ref [] doDownload.ref []

  startBlock doDownload
  let _ ← call cuda.fnDownload [ctxPtr, bufX, outPtr, outLen]
  ret

  startBlock skipDl
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 loadFn ++ "\n" ++
  buildFunction 2 prepFn ++ "\n" ++
  buildFunction 3 inferFn ++ "\n" ++
  buildFunction 4 finalizeFn

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

def artifacts : Array Json :=
  #[
    toJsonEntry "cuda_decoder_load" buildConfig loadAlgorithm,
    toJsonEntry "cuda_decoder_prep" buildConfig prepAlgorithm,
    toJsonEntry "cuda_decoder_infer" buildConfig inferAlgorithm,
    toJsonEntry "cuda_decoder_stack16" buildConfig (stackAlgorithm 16),
    toJsonEntry "cuda_decoder_stack32" buildConfig (stackAlgorithm 32),
  ]

end CudaDecoderLayer
