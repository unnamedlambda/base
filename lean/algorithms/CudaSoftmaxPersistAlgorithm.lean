import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CudaSoftmaxPersist

/-
  Multi-block persistent softmax.

  Buffers (created in load, by index):
    0  x        — input floats (n f32)
    1  y        — output floats (n f32)
    2  meta     — [n: u32, num_blocks: u32] (8 bytes, written once in load)
    3  partials — [max0, sum0, max1, sum1, ...] (num_blocks * 8 bytes)
    4  params   — [global_max: f32, global_inv_sum: f32] (8 bytes, written by global_reduce)

  Algorithms:
    u0:1  load      — init CUDA, alloc 5 bufs, write meta, store n/num_blocks in sm
    u0:2  prep      — upload x to buf0
    u0:3  core      — launch 3 kernels, no sync/download
    u0:4  finalize  — cl_cuda_sync, optional download y

  Shared memory layout:
    0x00-0x27  reserved
    0x28-0x2F  n (i64)
    0x30-0x37  num_blocks (i64)

  Three PTX entry points in one module (called via cl_cuda_launch_named):
    block_reduce  — gridDim=num_blocks: each block computes (local_max, local_sum) via online softmax
    global_reduce — gridDim=1: combines partials into (global_max, global_inv_sum) in params buf
    normalize     — gridDim=num_blocks: y[i] = ex2((x[i]-global_max)*log2e) * global_inv_sum
-/

-- initial_memory offsets
def NAME_BLOCK_REDUCE  : Nat := 0x0100  -- "block_reduce\0"
def NAME_GLOBAL_REDUCE : Nat := 0x0110  -- "global_reduce\0"
def NAME_NORMALIZE     : Nat := 0x0120  -- "normalize\0"
def NAME_SMALL_SOFTMAX : Nat := 0x0130  -- "small_softmax\0"
def PTX_SOURCE_OFF     : Nat := 0x0200
def BIND_K1_OFF        : Nat := 0x4700  -- [0,2,3]  x, meta, partials
def BIND_K2_OFF        : Nat := 0x4710  -- [2,3,4]  meta, partials, params
def BIND_K3_OFF        : Nat := 0x4720  -- [0,2,4,1] x, meta, params, y
def BIND_SMALL_OFF     : Nat := 0x4730  -- [0,2,1] x, meta, y
def MEM_SIZE           : Nat := 0x4740
def TIMEOUT_MS         : Nat := 30000

/-
  PTX: three kernels share one module. smem[0..31] = 8 warp maxes,
  smem[32..63] = 8 warp sums. Online combination: given (m1,s1)+(m2,s2),
  new_max = max(m1,m2), new_sum = s1*ex2((m1-new_max)*log2e) + s2*ex2((m2-new_max)*log2e).
-/
def ptxSource : String :=
  ".version 8.0\n" ++
  ".target sm_86\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".shared .align 4 .b8 _smem[64];\n" ++
  "\n" ++
  -- ── kernel 0: small_softmax ──────────────────────────────────────────────
  -- buf0=x (n f32), buf1=meta [n:u32,...], buf2=y (n f32)
  -- Single-block fused path for n <= 2048.
  ".visible .entry small_softmax(\n" ++
  "    .param .u64 x_buf,\n" ++
  "    .param .u64 meta_buf,\n" ++
  "    .param .u64 y_buf\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred %p;\n" ++
  "    .reg .u32  %r<10>;\n" ++
  "    .reg .u64  %rd<6>;\n" ++
  "    .reg .f32  %f<10>;\n" ++
  "    ld.param.u64 %rd0, [x_buf];\n" ++
  "    ld.param.u64 %rd1, [meta_buf];\n" ++
  "    ld.param.u64 %rd2, [y_buf];\n" ++
  "    ld.global.u32 %r0, [%rd1];\n" ++          -- n
  "    mov.u32 %r1, %tid.x;\n" ++
  "    shr.u32 %r2, %r1, 5;\n" ++
  "    and.b32 %r3, %r1, 31;\n" ++
  "    mov.f32 %f8, 0f3fb8aa3b;\n" ++            -- log2e
  "    mov.f32 %f0, 0f00000000;\n" ++            -- local_max
  "    mov.u32 %r4, %r1;\n" ++
  "sm_loop_max:\n" ++
  "    setp.ge.u32 %p, %r4, %r0;\n" ++
  "    @%p bra sm_done_max;\n" ++
  "    cvt.u64.u32 %rd3, %r4;\n" ++
  "    shl.b64 %rd3, %rd3, 2;\n" ++
  "    add.u64 %rd4, %rd0, %rd3;\n" ++
  "    ld.global.f32 %f1, [%rd4];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    add.u32 %r4, %r4, 256;\n" ++
  "    bra sm_loop_max;\n" ++
  "sm_done_max:\n" ++
  "    shfl.sync.bfly.b32 %f1, %f0, 16, 31, 0xffffffff;\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    shfl.sync.bfly.b32 %f1, %f0, 8, 31, 0xffffffff;\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    shfl.sync.bfly.b32 %f1, %f0, 4, 31, 0xffffffff;\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    shfl.sync.bfly.b32 %f1, %f0, 2, 31, 0xffffffff;\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    shfl.sync.bfly.b32 %f1, %f0, 1, 31, 0xffffffff;\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    setp.ne.u32 %p, %r3, 0;\n" ++
  "    @%p bra sm_skip_max_store;\n" ++
  "    mov.u32 %r5, _smem;\n" ++
  "    shl.b32 %r6, %r2, 2;\n" ++
  "    add.u32 %r5, %r5, %r6;\n" ++
  "    st.shared.f32 [%r5], %f0;\n" ++
  "sm_skip_max_store:\n" ++
  "    bar.sync 0;\n" ++
  "    setp.ne.u32 %p, %r1, 0;\n" ++
  "    @%p bra sm_skip_max_reduce;\n" ++
  "    mov.u32 %r5, _smem;\n" ++
  "    ld.shared.f32 %f0, [%r5+0];\n" ++
  "    ld.shared.f32 %f1, [%r5+4];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+8];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+12];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+16];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+20];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+24];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+28];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    st.shared.f32 [%r5+32], %f0;\n" ++
  "sm_skip_max_reduce:\n" ++
  "    bar.sync 0;\n" ++
  "    mov.u32 %r5, _smem;\n" ++
  "    ld.shared.f32 %f6, [%r5+32];\n" ++        -- global max
  "    mov.f32 %f0, 0f00000000;\n" ++            -- local_sum
  "    mov.u32 %r4, %r1;\n" ++
  "sm_loop_sum:\n" ++
  "    setp.ge.u32 %p, %r4, %r0;\n" ++
  "    @%p bra sm_done_sum;\n" ++
  "    cvt.u64.u32 %rd3, %r4;\n" ++
  "    shl.b64 %rd3, %rd3, 2;\n" ++
  "    add.u64 %rd4, %rd0, %rd3;\n" ++
  "    ld.global.f32 %f1, [%rd4];\n" ++
  "    sub.f32 %f1, %f1, %f6;\n" ++
  "    mul.f32 %f1, %f1, %f8;\n" ++
  "    ex2.approx.f32 %f1, %f1;\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    add.u32 %r4, %r4, 256;\n" ++
  "    bra sm_loop_sum;\n" ++
  "sm_done_sum:\n" ++
  "    shfl.sync.bfly.b32 %f1, %f0, 16, 31, 0xffffffff;\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    shfl.sync.bfly.b32 %f1, %f0, 8, 31, 0xffffffff;\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    shfl.sync.bfly.b32 %f1, %f0, 4, 31, 0xffffffff;\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    shfl.sync.bfly.b32 %f1, %f0, 2, 31, 0xffffffff;\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    shfl.sync.bfly.b32 %f1, %f0, 1, 31, 0xffffffff;\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    setp.ne.u32 %p, %r3, 0;\n" ++
  "    @%p bra sm_skip_sum_store;\n" ++
  "    mov.u32 %r5, _smem;\n" ++
  "    shl.b32 %r6, %r2, 2;\n" ++
  "    add.u32 %r5, %r5, %r6;\n" ++
  "    st.shared.f32 [%r5], %f0;\n" ++
  "sm_skip_sum_store:\n" ++
  "    bar.sync 0;\n" ++
  "    setp.ne.u32 %p, %r1, 0;\n" ++
  "    @%p bra sm_skip_sum_reduce;\n" ++
  "    mov.u32 %r5, _smem;\n" ++
  "    ld.shared.f32 %f0, [%r5+0];\n" ++
  "    ld.shared.f32 %f1, [%r5+4];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+8];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+12];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+16];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+20];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+24];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+28];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    rcp.approx.f32 %f0, %f0;\n" ++
  "    st.shared.f32 [%r5+36], %f0;\n" ++
  "sm_skip_sum_reduce:\n" ++
  "    bar.sync 0;\n" ++
  "    mov.u32 %r5, _smem;\n" ++
  "    ld.shared.f32 %f7, [%r5+36];\n" ++        -- global inv sum
  "    mov.u32 %r4, %r1;\n" ++
  "sm_loop_out:\n" ++
  "    setp.ge.u32 %p, %r4, %r0;\n" ++
  "    @%p bra sm_done_out;\n" ++
  "    cvt.u64.u32 %rd3, %r4;\n" ++
  "    shl.b64 %rd3, %rd3, 2;\n" ++
  "    add.u64 %rd4, %rd0, %rd3;\n" ++
  "    ld.global.f32 %f1, [%rd4];\n" ++
  "    sub.f32 %f1, %f1, %f6;\n" ++
  "    mul.f32 %f1, %f1, %f8;\n" ++
  "    ex2.approx.f32 %f1, %f1;\n" ++
  "    mul.f32 %f1, %f1, %f7;\n" ++
  "    add.u64 %rd5, %rd2, %rd3;\n" ++
  "    st.global.f32 [%rd5], %f1;\n" ++
  "    add.u32 %r4, %r4, 256;\n" ++
  "    bra sm_loop_out;\n" ++
  "sm_done_out:\n" ++
  "    ret;\n" ++
  "}\n" ++
  "\n" ++
  -- ── kernel 1: block_reduce ───────────────────────────────────────────────
  -- buf0=x (n f32), buf1=meta [n:u32,num_blocks:u32], buf2=partials [max,sum per block]
  ".visible .entry block_reduce(\n" ++
  "    .param .u64 x_buf,\n" ++
  "    .param .u64 meta_buf,\n" ++
  "    .param .u64 partials_buf\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred %p;\n" ++
  "    .reg .u32  %r<10>;\n" ++
  "    .reg .u64  %rd<6>;\n" ++
  "    .reg .f32  %f<12>;\n" ++
  "    ld.param.u64 %rd0, [x_buf];\n" ++
  "    ld.param.u64 %rd1, [meta_buf];\n" ++
  "    ld.param.u64 %rd2, [partials_buf];\n" ++
  "    ld.global.u32 %r0, [%rd1];\n" ++          -- n
  "    mov.u32 %r1, %ctaid.x;\n" ++              -- block_idx
  "    mov.u32 %r2, %tid.x;\n" ++               -- tid (0..255)
  "    shr.u32 %r3, %r2, 5;\n" ++               -- warp_id
  "    and.b32 %r4, %r2, 31;\n" ++              -- lane_id
  "    mul.lo.u32 %r5, %r1, 256;\n" ++
  "    add.u32 %r5, %r5, %r2;\n" ++             -- global_idx
  "    mov.f32 %f10, 0f3fb8aa3b;\n" ++           -- log2e
  -- load x[global_idx] or use (-inf, 0) for out-of-bounds
  "    setp.ge.u32 %p, %r5, %r0;\n" ++
  "    @%p bra br_oob;\n" ++
  "    cvt.u64.u32 %rd3, %r5;\n" ++
  "    shl.b64 %rd3, %rd3, 2;\n" ++
  "    add.u64 %rd4, %rd0, %rd3;\n" ++
  "    ld.global.f32 %f0, [%rd4];\n" ++          -- f0 = local_max = x[i]
  "    mov.f32 %f1, 0f3f800000;\n" ++            -- f1 = local_sum = 1.0
  "    bra br_reduce;\n" ++
  "br_oob:\n" ++
  "    mov.f32 %f0, 0f00000000;\n" ++            -- 0.0 (safe empty: avoids -inf - -inf = NaN in butterfly)
  "    mov.f32 %f1, 0f00000000;\n" ++            -- 0.0
  "br_reduce:\n" ++
  -- warp butterfly reduction (5 steps: 16,8,4,2,1)
  "    shfl.sync.bfly.b32 %f2, %f0, 16, 31, 0xffffffff;\n" ++
  "    shfl.sync.bfly.b32 %f3, %f1, 16, 31, 0xffffffff;\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 8, 31, 0xffffffff;\n" ++
  "    shfl.sync.bfly.b32 %f3, %f1, 8, 31, 0xffffffff;\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 4, 31, 0xffffffff;\n" ++
  "    shfl.sync.bfly.b32 %f3, %f1, 4, 31, 0xffffffff;\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 2, 31, 0xffffffff;\n" ++
  "    shfl.sync.bfly.b32 %f3, %f1, 2, 31, 0xffffffff;\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 1, 31, 0xffffffff;\n" ++
  "    shfl.sync.bfly.b32 %f3, %f1, 1, 31, 0xffffffff;\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  -- lane 0 writes warp result to smem
  "    setp.ne.u32 %p, %r4, 0;\n" ++
  "    @%p bra br_skip_wst;\n" ++
  "    mov.u32 %r6, _smem;\n" ++
  "    shl.b32 %r7, %r3, 2;\n" ++
  "    add.u32 %r8, %r6, %r7;\n" ++
  "    st.shared.f32 [%r8], %f0;\n" ++           -- smem[warp_id] = warp_max
  "    add.u32 %r8, %r8, 32;\n" ++
  "    st.shared.f32 [%r8], %f1;\n" ++           -- smem[8+warp_id] = warp_sum
  "br_skip_wst:\n" ++
  "    bar.sync 0;\n" ++
  -- thread 0 combines all 8 warp results
  "    setp.ne.u32 %p, %r2, 0;\n" ++
  "    @%p bra br_skip_bred;\n" ++
  "    mov.u32 %r6, _smem;\n" ++
  "    ld.shared.f32 %f0, [%r6+0];\n" ++
  "    ld.shared.f32 %f1, [%r6+32];\n" ++
  -- combine with warps 1..7
  "    ld.shared.f32 %f2, [%r6+4];\n" ++ "    ld.shared.f32 %f3, [%r6+36];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+8];\n" ++ "    ld.shared.f32 %f3, [%r6+40];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+12];\n" ++ "    ld.shared.f32 %f3, [%r6+44];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+16];\n" ++ "    ld.shared.f32 %f3, [%r6+48];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+20];\n" ++ "    ld.shared.f32 %f3, [%r6+52];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+24];\n" ++ "    ld.shared.f32 %f3, [%r6+56];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+28];\n" ++ "    ld.shared.f32 %f3, [%r6+60];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  -- write (block_max, block_sum) to partials[block_idx*2], partials[block_idx*2+1]
  "    cvt.u64.u32 %rd3, %r1;\n" ++
  "    shl.b64 %rd3, %rd3, 3;\n" ++              -- block_idx * 8
  "    add.u64 %rd4, %rd2, %rd3;\n" ++
  "    st.global.f32 [%rd4], %f0;\n" ++
  "    add.u64 %rd4, %rd4, 4;\n" ++
  "    st.global.f32 [%rd4], %f1;\n" ++
  "br_skip_bred:\n" ++
  "    ret;\n" ++
  "}\n" ++
  "\n" ++
  -- ── kernel 2: global_reduce ──────────────────────────────────────────────
  -- buf0=meta [n:u32,num_blocks:u32], buf1=partials [max,sum per block], buf2=params [out]
  ".visible .entry global_reduce(\n" ++
  "    .param .u64 meta_buf,\n" ++
  "    .param .u64 partials_buf,\n" ++
  "    .param .u64 params_buf\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred %p;\n" ++
  "    .reg .u32  %r<10>;\n" ++
  "    .reg .u64  %rd<6>;\n" ++
  "    .reg .f32  %f<12>;\n" ++
  "    ld.param.u64 %rd0, [meta_buf];\n" ++
  "    ld.param.u64 %rd1, [partials_buf];\n" ++
  "    ld.param.u64 %rd2, [params_buf];\n" ++
  "    ld.global.u32 %r0, [%rd0+4];\n" ++        -- num_blocks
  "    mov.u32 %r1, %tid.x;\n" ++
  "    shr.u32 %r2, %r1, 5;\n" ++               -- warp_id
  "    and.b32 %r3, %r1, 31;\n" ++              -- lane_id
  "    mov.f32 %f10, 0f3fb8aa3b;\n" ++           -- log2e
  -- each thread handles one or more blocks in a stride loop
  "    mov.f32 %f0, 0f00000000;\n" ++            -- local_max = 0.0 (safe empty: avoids -inf - -inf = NaN)
  "    mov.f32 %f1, 0f00000000;\n" ++            -- local_sum = 0
  "    mov.u32 %r4, %r1;\n" ++                  -- i = tid
  "gr_loop:\n" ++
  "    setp.ge.u32 %p, %r4, %r0;\n" ++          -- i >= num_blocks?
  "    @%p bra gr_done;\n" ++
  "    cvt.u64.u32 %rd3, %r4;\n" ++
  "    shl.b64 %rd3, %rd3, 3;\n" ++             -- i * 8
  "    add.u64 %rd4, %rd1, %rd3;\n" ++
  "    ld.global.f32 %f2, [%rd4];\n" ++          -- partials[i].max
  "    ld.global.f32 %f3, [%rd4+4];\n" ++        -- partials[i].sum
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    add.u32 %r4, %r4, 256;\n" ++
  "    bra gr_loop;\n" ++
  "gr_done:\n" ++
  -- warp butterfly reduction (5 steps)
  "    shfl.sync.bfly.b32 %f2, %f0, 16, 31, 0xffffffff;\n" ++
  "    shfl.sync.bfly.b32 %f3, %f1, 16, 31, 0xffffffff;\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 8, 31, 0xffffffff;\n" ++
  "    shfl.sync.bfly.b32 %f3, %f1, 8, 31, 0xffffffff;\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 4, 31, 0xffffffff;\n" ++
  "    shfl.sync.bfly.b32 %f3, %f1, 4, 31, 0xffffffff;\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 2, 31, 0xffffffff;\n" ++
  "    shfl.sync.bfly.b32 %f3, %f1, 2, 31, 0xffffffff;\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 1, 31, 0xffffffff;\n" ++
  "    shfl.sync.bfly.b32 %f3, %f1, 1, 31, 0xffffffff;\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    setp.ne.u32 %p, %r3, 0;\n" ++
  "    @%p bra gr_skip_wst;\n" ++
  "    mov.u32 %r6, _smem;\n" ++
  "    shl.b32 %r7, %r2, 2;\n" ++
  "    add.u32 %r8, %r6, %r7;\n" ++
  "    st.shared.f32 [%r8], %f0;\n" ++
  "    add.u32 %r8, %r8, 32;\n" ++
  "    st.shared.f32 [%r8], %f1;\n" ++
  "gr_skip_wst:\n" ++
  "    bar.sync 0;\n" ++
  "    setp.ne.u32 %p, %r1, 0;\n" ++
  "    @%p bra gr_skip_bred;\n" ++
  "    mov.u32 %r6, _smem;\n" ++
  "    ld.shared.f32 %f0, [%r6+0];\n" ++ "    ld.shared.f32 %f1, [%r6+32];\n" ++
  "    ld.shared.f32 %f2, [%r6+4];\n" ++ "    ld.shared.f32 %f3, [%r6+36];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+8];\n" ++ "    ld.shared.f32 %f3, [%r6+40];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+12];\n" ++ "    ld.shared.f32 %f3, [%r6+44];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+16];\n" ++ "    ld.shared.f32 %f3, [%r6+48];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+20];\n" ++ "    ld.shared.f32 %f3, [%r6+52];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+24];\n" ++ "    ld.shared.f32 %f3, [%r6+56];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  "    ld.shared.f32 %f2, [%r6+28];\n" ++ "    ld.shared.f32 %f3, [%r6+60];\n" ++
  "    max.f32 %f4, %f0, %f2;\n" ++
  "    sub.f32 %f5, %f0, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f1, %f1, %f5;\n" ++
  "    sub.f32 %f5, %f2, %f4;\n" ++ "    mul.f32 %f5, %f5, %f10;\n" ++ "    ex2.approx.f32 %f5, %f5;\n" ++ "    mul.f32 %f3, %f3, %f5;\n" ++
  "    add.f32 %f1, %f1, %f3;\n" ++ "    mov.f32 %f0, %f4;\n" ++
  -- write global_max and global_inv_sum to params_buf
  "    rcp.approx.f32 %f1, %f1;\n" ++            -- inv_sum
  "    st.global.f32 [%rd2], %f0;\n" ++          -- params[0] = global_max
  "    st.global.f32 [%rd2+4], %f1;\n" ++        -- params[1] = global_inv_sum
  "gr_skip_bred:\n" ++
  "    ret;\n" ++
  "}\n" ++
  "\n" ++
  -- ── kernel 3: normalize ──────────────────────────────────────────────────
  -- buf0=x, buf1=meta [n:u32,...], buf2=params [global_max,global_inv_sum], buf3=y
  ".visible .entry normalize(\n" ++
  "    .param .u64 x_buf,\n" ++
  "    .param .u64 meta_buf,\n" ++
  "    .param .u64 params_buf,\n" ++
  "    .param .u64 y_buf\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred %p;\n" ++
  "    .reg .u32  %r<6>;\n" ++
  "    .reg .u64  %rd<6>;\n" ++
  "    .reg .f32  %f<6>;\n" ++
  "    ld.param.u64 %rd0, [x_buf];\n" ++
  "    ld.param.u64 %rd1, [meta_buf];\n" ++
  "    ld.param.u64 %rd2, [params_buf];\n" ++
  "    ld.param.u64 %rd3, [y_buf];\n" ++
  "    ld.global.u32 %r0, [%rd1];\n" ++          -- n
  "    mov.u32 %r1, %ctaid.x;\n" ++
  "    mov.u32 %r2, %tid.x;\n" ++
  "    mul.lo.u32 %r3, %r1, 256;\n" ++
  "    add.u32 %r3, %r3, %r2;\n" ++              -- global_idx
  "    setp.ge.u32 %p, %r3, %r0;\n" ++
  "    @%p bra nrm_end;\n" ++
  "    ld.global.f32 %f0, [%rd2];\n" ++          -- global_max
  "    ld.global.f32 %f1, [%rd2+4];\n" ++        -- global_inv_sum
  "    cvt.u64.u32 %rd4, %r3;\n" ++
  "    shl.b64 %rd4, %rd4, 2;\n" ++
  "    add.u64 %rd5, %rd0, %rd4;\n" ++
  "    ld.global.f32 %f2, [%rd5];\n" ++          -- x[i]
  "    sub.f32 %f2, %f2, %f0;\n" ++              -- x[i] - global_max
  "    mov.f32 %f3, 0f3fb8aa3b;\n" ++            -- log2e
  "    mul.f32 %f2, %f2, %f3;\n" ++
  "    ex2.approx.f32 %f2, %f2;\n" ++            -- exp(x[i] - global_max)
  "    mul.f32 %f2, %f2, %f1;\n" ++              -- * global_inv_sum
  "    add.u64 %rd5, %rd3, %rd4;\n" ++
  "    st.global.f32 [%rd5], %f2;\n" ++          -- y[i]
  "nrm_end:\n" ++
  "    ret;\n" ++
  "}\n"

-- CLIF: u0:0 noop, u0:1 load, u0:2 prep, u0:3 core, u0:4 finalize
def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "  return\n" ++
  "}\n"

/-
  Load: init CUDA, alloc 5 bufs, compute num_blocks, upload meta, store n/num_blocks in sm.
  Data: [n: u64]
  Buf order: 0=x, 1=y, 2=meta, 3=partials, 4=params
-/
def clifLoadFn : String :=
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++
  "    sig1 = (i64, i64) -> i32 system_v\n" ++
  "    sig2 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_init sig0\n" ++
  "    fn1 = %cl_cuda_create_buffer sig1\n" ++
  "    fn2 = %cl_cuda_upload_ptr sig2\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++  -- data_ptr
  "  call fn0(v0)\n" ++
  "  v10 = load.i64 notrap aligned v1\n" ++       -- n
  -- num_blocks = (n + 255) >> 8
  "  v11 = iadd_imm v10, 255\n" ++
  "  v12 = ushr_imm v11, 8\n" ++                  -- num_blocks
  "  v13 = iadd_imm v0, 0x28\n" ++
  "  store notrap aligned v10, v13\n" ++          -- sm[0x28] = n
  "  v14 = iadd_imm v0, 0x30\n" ++
  "  store notrap aligned v12, v14\n" ++          -- sm[0x30] = num_blocks
  -- buf sizes
  "  v20 = ishl_imm v10, 2\n" ++                  -- x_bytes = n*4
  "  v21 = ishl_imm v12, 3\n" ++                  -- partials_bytes = num_blocks*8
  "  v22 = iconst.i64 8\n" ++                     -- meta/params = 8 bytes each
  -- alloc: buf0=x, buf1=y, buf2=meta, buf3=partials, buf4=params
  "  v30 = call fn1(v0, v20)\n" ++               -- buf0 = x
  "  v31 = call fn1(v0, v20)\n" ++               -- buf1 = y
  "  v32 = call fn1(v0, v22)\n" ++               -- buf2 = meta
  "  v33 = call fn1(v0, v21)\n" ++               -- buf3 = partials
  "  v34 = call fn1(v0, v22)\n" ++               -- buf4 = params
  -- pack [n:u32, num_blocks:u32] into 8 bytes at sm[0x40]
  "  v40 = ireduce.i32 v10\n" ++                  -- n as i32
  "  v41 = ireduce.i32 v12\n" ++                  -- num_blocks as i32
  "  v42 = uextend.i64 v40\n" ++
  "  v43 = uextend.i64 v41\n" ++
  "  v44 = ishl_imm v43, 32\n" ++                -- num_blocks << 32
  "  v45 = bor v42, v44\n" ++                    -- [n:u32][num_blocks:u32] as i64
  "  v46 = iadd_imm v0, 0x40\n" ++
  "  store notrap aligned v45, v46\n" ++          -- sm[0x40] = packed meta
  "  v47 = call fn2(v0, v32, v46, v22)\n" ++     -- upload 8 bytes to meta buf
  "  return\n" ++
  "}\n"

-- Prep: upload x from data_ptr to buf0
def clifPrepFn : String :=
  "function u0:2(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_upload_ptr sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++   -- data_ptr (x)
  "  v2 = load.i64 notrap aligned v0+0x10\n" ++   -- data_len
  "  v3 = iconst.i32 0\n" ++                      -- buf0 (x)
  "  v4 = call fn0(v0, v3, v1, v2)\n" ++
  "  return\n" ++
  "}\n"

/-
  Infer: launch 3 kernels (block_reduce, global_reduce, normalize),
  sync GPU, then optionally download y.
  Bind layouts:
    K1 (block_reduce):  bufs [0, 2, 3]  — x, meta, partials
    K2 (global_reduce): bufs [2, 3, 4]  — meta, partials, params
    K3 (normalize):     bufs [0, 2, 4, 1] — x, meta, params, y
-/
def clifCoreFn : String :=
  "function u0:3(i64) system_v {\n" ++
  "    sig0 = (i64, i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_launch_named sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v10 = load.i64 notrap aligned v0+0x28\n" ++  -- n
  "  v11 = load.i64 notrap aligned v0+0x30\n" ++  -- num_blocks
  "  v12 = ireduce.i32 v11\n" ++                  -- num_blocks as i32
  "  v20 = iconst.i64 " ++ toString PTX_SOURCE_OFF ++ "\n" ++
  "  v21 = iconst.i64 " ++ toString NAME_BLOCK_REDUCE ++ "\n" ++
  "  v22 = iconst.i64 " ++ toString NAME_GLOBAL_REDUCE ++ "\n" ++
  "  v23 = iconst.i64 " ++ toString NAME_NORMALIZE ++ "\n" ++
  "  v24 = iconst.i64 " ++ toString NAME_SMALL_SOFTMAX ++ "\n" ++
  "  v25 = iconst.i64 " ++ toString BIND_K1_OFF ++ "\n" ++
  "  v26 = iconst.i64 " ++ toString BIND_K2_OFF ++ "\n" ++
  "  v27 = iconst.i64 " ++ toString BIND_K3_OFF ++ "\n" ++
  "  v28 = iconst.i64 " ++ toString BIND_SMALL_OFF ++ "\n" ++
  "  v30 = iconst.i32 1\n" ++
  "  v31 = iconst.i32 3\n" ++
  "  v32 = iconst.i32 4\n" ++
  "  v33 = iconst.i32 256\n" ++
  "  v34 = iconst.i64 2048\n" ++
  "  v35 = icmp ule v10, v34\n" ++
  "  brif v35, block1, block2\n" ++
  "block1:\n" ++
  "  v36 = call fn0(v0, v20, v24, v31, v28, v30, v30, v30, v33, v30, v30)\n" ++
  "  return\n" ++
  "block2:\n" ++
  -- K1: block_reduce — gridDim=(num_blocks,1,1), blockDim=(256,1,1), 3 bufs
  "  v40 = call fn0(v0, v20, v21, v31, v25, v12, v30, v30, v33, v30, v30)\n" ++
  -- K2: global_reduce — gridDim=(1,1,1), blockDim=(256,1,1), 3 bufs
  "  v41 = call fn0(v0, v20, v22, v31, v26, v30, v30, v30, v33, v30, v30)\n" ++
  -- K3: normalize — gridDim=(num_blocks,1,1), blockDim=(256,1,1), 4 bufs
  "  v42 = call fn0(v0, v20, v23, v32, v27, v12, v30, v30, v33, v30, v30)\n" ++
  "  return\n" ++
  "}\n"

def clifFinalizeFn : String :=
  "function u0:4(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    sig1 = (i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_download_ptr sig0\n" ++
  "    fn1 = %cl_cuda_sync sig1\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x18\n" ++   -- out_ptr
  "  v2 = load.i64 notrap aligned v0+0x20\n" ++   -- out_len
  "  v3 = call fn1(v0)\n" ++
  "  v4 = iconst.i64 0\n" ++
  "  v5 = icmp eq v2, v4\n" ++
  "  brif v5, block1, block2\n" ++
  "block1:\n" ++
  "  return\n" ++
  "block2:\n" ++
  "  v6 = iconst.i32 1\n" ++                     -- y is buf1
  "  v7 = call fn0(v0, v6, v1, v2)\n" ++
  "  return\n" ++
  "}\n"

def clifIR : String :=
  clifNoopFn ++ "\n" ++ clifLoadFn ++ "\n" ++ clifPrepFn ++ "\n" ++ clifCoreFn ++ "\n" ++ clifFinalizeFn

-- initial_memory: names, PTX source, bind descriptors
def nameBlockReduce  : List UInt8 := "block_reduce".toUTF8.toList ++ [0]
def nameGlobalReduce : List UInt8 := "global_reduce".toUTF8.toList ++ [0]
def nameNormalize    : List UInt8 := "normalize".toUTF8.toList ++ [0]
def nameSmallSoftmax : List UInt8 := "small_softmax".toUTF8.toList ++ [0]
def ptxBytes         : List UInt8 := ptxSource.toUTF8.toList ++ [0]

-- bind descriptors: buf IDs as i32 (4 bytes each, little-endian)
def i32LE (n : UInt32) : List UInt8 :=
  [n &&& 0xff, (n >>> 8) &&& 0xff, (n >>> 16) &&& 0xff, (n >>> 24) &&& 0xff].map (·.toUInt8)

def bindK1 : List UInt8 := i32LE 0 ++ i32LE 2 ++ i32LE 3  -- x, meta, partials
def bindK2 : List UInt8 := i32LE 2 ++ i32LE 3 ++ i32LE 4  -- meta, partials, params
def bindK3 : List UInt8 := i32LE 0 ++ i32LE 2 ++ i32LE 4 ++ i32LE 1  -- x, meta, params, y
def bindSmall : List UInt8 := i32LE 0 ++ i32LE 2 ++ i32LE 1  -- x, meta, y

def buildInitialMemory : List UInt8 :=
  let names :=
    zeros (NAME_BLOCK_REDUCE) ++
    nameBlockReduce  ++ zeros (NAME_GLOBAL_REDUCE - NAME_BLOCK_REDUCE  - nameBlockReduce.length) ++
    nameGlobalReduce ++ zeros (NAME_NORMALIZE     - NAME_GLOBAL_REDUCE - nameGlobalReduce.length) ++
    nameNormalize    ++ zeros (NAME_SMALL_SOFTMAX - NAME_NORMALIZE     - nameNormalize.length) ++
    nameSmallSoftmax ++ zeros (PTX_SOURCE_OFF      - NAME_SMALL_SOFTMAX - nameSmallSoftmax.length)
  let ptx  := ptxBytes ++ zeros (BIND_K1_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind :=
    bindK1 ++ zeros (BIND_K2_OFF - BIND_K1_OFF - bindK1.length) ++
    bindK2 ++ zeros (BIND_K3_OFF - BIND_K2_OFF - bindK2.length) ++
    bindK3 ++ zeros (BIND_SMALL_OFF - BIND_K3_OFF - bindK3.length) ++
    bindSmall ++ zeros (MEM_SIZE - BIND_SMALL_OFF - bindSmall.length)
  names ++ ptx ++ bind

def actions (src : UInt32) : List Action :=
  [{ kind := .ClifCall, dst := 0, src := src, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0,
  initial_memory := buildInitialMemory
}

def loadAlgorithm  : Algorithm := { actions := actions 1, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }
def prepAlgorithm  : Algorithm := { actions := actions 2, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }
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

end CudaSoftmaxPersist

def main : IO Unit := do
  let json := Json.arr #[
    toJson CudaSoftmaxPersist.buildConfig,
    toJson CudaSoftmaxPersist.loadAlgorithm,
    toJson CudaSoftmaxPersist.prepAlgorithm,
    toJson CudaSoftmaxPersist.inferAlgorithm,
    toJson (CudaSoftmaxPersist.stackAlgorithm 64)
  ]
  IO.println (Json.compress json)
