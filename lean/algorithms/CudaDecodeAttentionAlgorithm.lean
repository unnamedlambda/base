import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CudaDecodeAttention

/-!
  Persistent single-token decode attention benchmark.

  Fixed dimensions:
    n_heads  = 14
    head_dim = 64
    d_model  = 896

  Runtime dimension:
    seq_len <= 2048

  Layouts:
    q      [n_heads, head_dim]
    k/v    [n_heads, seq_len, head_dim]
    out    [n_heads, head_dim]

  Load payload:
    [seq_len: u64]
    [k_cache: n_heads * seq_len * head_dim f32]
    [v_cache: n_heads * seq_len * head_dim f32]

  Prep payload:
    [q: n_heads * head_dim f32]

  Timed infer:
    1. scores = scale * (K @ q)       batched over heads via cuBLAS
    2. probs  = softmax(scores)       one CUDA block per head
    3. out    = V^T @ probs           batched over heads via cuBLAS

  Algorithms:
    u0:1  load      — alloc resident buffers, upload K/V once
    u0:2  prep      — upload q
    u0:3  core      — scores + softmax + V mix, no sync/download
    u0:4  finalize  — sync and optional download
-/

def N_HEADS : Nat := 14
def HEAD_DIM : Nat := 64
def D_MODEL : Nat := N_HEADS * HEAD_DIM
def MAX_SEQ : Nat := 2048

def D_MODEL_BYTES : Nat := D_MODEL * 4

def PTX_SOURCE_OFF : Nat := 0x0200
def BIND_DESC_OFF  : Nat := 0x5000
def MEM_SIZE       : Nat := 0x5100
def TIMEOUT_MS     : Nat := 30000

def BUF_Q_OFF      : Nat := 0x30
def BUF_K_OFF      : Nat := 0x34
def BUF_V_OFF      : Nat := 0x38
def BUF_SCORES_OFF : Nat := 0x3C
def BUF_PROBS_OFF  : Nat := 0x40
def BUF_OUT_OFF    : Nat := 0x44
def BUF_META_OFF   : Nat := 0x48

def ptxSource : String :=
  ".version 8.0\n" ++
  ".target sm_86\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".shared .align 4 .b8 _smem[64];\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 scores_buf,\n" ++
  "    .param .u64 meta_buf,\n" ++
  "    .param .u64 probs_buf\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred %p;\n" ++
  "    .reg .u32  %r<12>;\n" ++
  "    .reg .u64  %rd<8>;\n" ++
  "    .reg .f32  %f<10>;\n" ++
  "    ld.param.u64 %rd0, [scores_buf];\n" ++
  "    ld.param.u64 %rd1, [meta_buf];\n" ++
  "    ld.param.u64 %rd2, [probs_buf];\n" ++
  "    ld.global.u32 %r0, [%rd1];\n" ++          -- seq_len
  "    mov.u32 %r1, %ctaid.x;\n" ++              -- head_idx
  "    mov.u32 %r2, %tid.x;\n" ++
  "    shr.u32 %r3, %r2, 5;\n" ++
  "    and.b32 %r4, %r2, 31;\n" ++
  "    cvt.u64.u32 %rd3, %r1;\n" ++
  "    cvt.u64.u32 %rd4, %r0;\n" ++
  "    mul.lo.u64 %rd5, %rd3, %rd4;\n" ++       -- head * seq_len
  "    shl.b64 %rd5, %rd5, 2;\n" ++              -- * sizeof(f32)
  "    add.u64 %rd6, %rd0, %rd5;\n" ++           -- scores head base
  "    add.u64 %rd7, %rd2, %rd5;\n" ++           -- probs head base
  "    mov.f32 %f8, 0f3fb8aa3b;\n" ++            -- log2e
  "    mov.f32 %f0, 0f00000000;\n" ++            -- local_max
  "    mov.u32 %r5, %r2;\n" ++
  "sm_loop_max:\n" ++
  "    setp.ge.u32 %p, %r5, %r0;\n" ++
  "    @%p bra sm_done_max;\n" ++
  "    cvt.u64.u32 %rd3, %r5;\n" ++
  "    shl.b64 %rd3, %rd3, 2;\n" ++
  "    add.u64 %rd4, %rd6, %rd3;\n" ++
  "    ld.global.f32 %f1, [%rd4];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    add.u32 %r5, %r5, 256;\n" ++
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
  "    setp.ne.u32 %p, %r4, 0;\n" ++
  "    @%p bra sm_skip_max_store;\n" ++
  "    mov.u32 %r6, _smem;\n" ++
  "    shl.b32 %r7, %r3, 2;\n" ++
  "    add.u32 %r6, %r6, %r7;\n" ++
  "    st.shared.f32 [%r6], %f0;\n" ++
  "sm_skip_max_store:\n" ++
  "    bar.sync 0;\n" ++
  "    setp.ne.u32 %p, %r2, 0;\n" ++
  "    @%p bra sm_skip_max_reduce;\n" ++
  "    mov.u32 %r6, _smem;\n" ++
  "    ld.shared.f32 %f0, [%r6+0];\n" ++
  "    ld.shared.f32 %f1, [%r6+4];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+8];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+12];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+16];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+20];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+24];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+28];\n" ++
  "    max.f32 %f0, %f0, %f1;\n" ++
  "    st.shared.f32 [%r6+32], %f0;\n" ++
  "sm_skip_max_reduce:\n" ++
  "    bar.sync 0;\n" ++
  "    mov.u32 %r6, _smem;\n" ++
  "    ld.shared.f32 %f6, [%r6+32];\n" ++
  "    mov.f32 %f0, 0f00000000;\n" ++            -- local_sum
  "    mov.u32 %r5, %r2;\n" ++
  "sm_loop_sum:\n" ++
  "    setp.ge.u32 %p, %r5, %r0;\n" ++
  "    @%p bra sm_done_sum;\n" ++
  "    cvt.u64.u32 %rd3, %r5;\n" ++
  "    shl.b64 %rd3, %rd3, 2;\n" ++
  "    add.u64 %rd4, %rd6, %rd3;\n" ++
  "    ld.global.f32 %f1, [%rd4];\n" ++
  "    sub.f32 %f1, %f1, %f6;\n" ++
  "    mul.f32 %f1, %f1, %f8;\n" ++
  "    ex2.approx.f32 %f1, %f1;\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    add.u32 %r5, %r5, 256;\n" ++
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
  "    setp.ne.u32 %p, %r4, 0;\n" ++
  "    @%p bra sm_skip_sum_store;\n" ++
  "    mov.u32 %r6, _smem;\n" ++
  "    shl.b32 %r7, %r3, 2;\n" ++
  "    add.u32 %r6, %r6, %r7;\n" ++
  "    st.shared.f32 [%r6], %f0;\n" ++
  "sm_skip_sum_store:\n" ++
  "    bar.sync 0;\n" ++
  "    setp.ne.u32 %p, %r2, 0;\n" ++
  "    @%p bra sm_skip_sum_reduce;\n" ++
  "    mov.u32 %r6, _smem;\n" ++
  "    ld.shared.f32 %f0, [%r6+0];\n" ++
  "    ld.shared.f32 %f1, [%r6+4];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+8];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+12];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+16];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+20];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+24];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r6+28];\n" ++
  "    add.f32 %f0, %f0, %f1;\n" ++
  "    rcp.approx.f32 %f0, %f0;\n" ++
  "    st.shared.f32 [%r6+36], %f0;\n" ++
  "sm_skip_sum_reduce:\n" ++
  "    bar.sync 0;\n" ++
  "    mov.u32 %r6, _smem;\n" ++
  "    ld.shared.f32 %f7, [%r6+36];\n" ++
  "    mov.u32 %r5, %r2;\n" ++
  "sm_loop_out:\n" ++
  "    setp.ge.u32 %p, %r5, %r0;\n" ++
  "    @%p bra sm_done_out;\n" ++
  "    cvt.u64.u32 %rd3, %r5;\n" ++
  "    shl.b64 %rd3, %rd3, 2;\n" ++
  "    add.u64 %rd4, %rd6, %rd3;\n" ++
  "    ld.global.f32 %f1, [%rd4];\n" ++
  "    sub.f32 %f1, %f1, %f6;\n" ++
  "    mul.f32 %f1, %f1, %f8;\n" ++
  "    ex2.approx.f32 %f1, %f1;\n" ++
  "    mul.f32 %f1, %f1, %f7;\n" ++
  "    add.u64 %rd5, %rd7, %rd3;\n" ++
  "    st.global.f32 [%rd5], %f1;\n" ++
  "    add.u32 %r5, %r5, 256;\n" ++
  "    bra sm_loop_out;\n" ++
  "sm_done_out:\n" ++
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
  "    fn2 = %cl_cuda_upload_ptr sig2\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  call fn0(v0)\n" ++
  "  v10 = load.i64 notrap aligned v1\n" ++               -- seq_len
  "  store notrap aligned v10, v0+0x28\n" ++
  "  v11 = iconst.i64 " ++ toString D_MODEL_BYTES ++ "\n" ++
  "  v12 = iconst.i64 " ++ toString D_MODEL ++ "\n" ++
  "  v13 = imul v10, v11\n" ++                            -- kv_bytes = seq_len * d_model * 4
  "  v14 = iconst.i64 " ++ toString N_HEADS ++ "\n" ++
  "  v15 = imul v10, v14\n" ++
  "  v16 = ishl_imm v15, 2\n" ++                          -- score_bytes = seq_len * n_heads * 4
  "  v17 = iconst.i64 8\n" ++
  "  v20 = call fn1(v0, v11)\n" ++                        -- q
  "  v21 = call fn1(v0, v13)\n" ++                        -- k
  "  v22 = call fn1(v0, v13)\n" ++                        -- v
  "  v23 = call fn1(v0, v16)\n" ++                        -- scores
  "  v24 = call fn1(v0, v16)\n" ++                        -- probs
  "  v25 = call fn1(v0, v11)\n" ++                        -- out
  "  v26 = call fn1(v0, v17)\n" ++                        -- meta
  "  store notrap aligned v20, v0+" ++ toString BUF_Q_OFF ++ "\n" ++
  "  store notrap aligned v21, v0+" ++ toString BUF_K_OFF ++ "\n" ++
  "  store notrap aligned v22, v0+" ++ toString BUF_V_OFF ++ "\n" ++
  "  store notrap aligned v23, v0+" ++ toString BUF_SCORES_OFF ++ "\n" ++
  "  store notrap aligned v24, v0+" ++ toString BUF_PROBS_OFF ++ "\n" ++
  "  store notrap aligned v25, v0+" ++ toString BUF_OUT_OFF ++ "\n" ++
  "  store notrap aligned v26, v0+" ++ toString BUF_META_OFF ++ "\n" ++
  "  v27 = ireduce.i32 v10\n" ++
  "  v28 = uextend.i64 v27\n" ++
  "  v29 = iadd_imm v0, 0x50\n" ++
  "  store notrap aligned v28, v29\n" ++                 -- [seq_len:u32][0:u32]
  "  v30 = call fn2(v0, v26, v29, v17)\n" ++
  "  v31 = iadd_imm v1, 8\n" ++
  "  v32 = call fn2(v0, v21, v31, v13)\n" ++
  "  v33 = iadd v31, v13\n" ++
  "  v34 = call fn2(v0, v22, v33, v13)\n" ++
  "  return\n" ++
  "}\n"

def clifPrepFn : String :=
  "function u0:2(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_upload_ptr sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  v2 = load.i32 notrap aligned v0+" ++ toString BUF_Q_OFF ++ "\n" ++
  "  v3 = iconst.i64 " ++ toString D_MODEL_BYTES ++ "\n" ++
  "  v4 = call fn0(v0, v2, v1, v3)\n" ++
  "  return\n" ++
  "}\n"

def clifCoreFn : String :=
  "function u0:3(i64) system_v {\n" ++
  "    sig0 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v\n" ++
  "    sig1 = (i64, i32, i32, i32, i32, i32, i32, i32, i64, i32, i64, i32, i32, i64, i32) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_launch sig0\n" ++
  "    fn1 = %cl_cublas_sgemm_strided_batched sig1\n" ++
  "block0(v0: i64):\n" ++
  "  v10 = load.i64 notrap aligned v0+0x28\n" ++  -- seq_len
  "  v11 = ireduce.i32 v10\n" ++
  "  v12 = load.i32 notrap aligned v0+" ++ toString BUF_Q_OFF ++ "\n" ++
  "  v13 = load.i32 notrap aligned v0+" ++ toString BUF_K_OFF ++ "\n" ++
  "  v14 = load.i32 notrap aligned v0+" ++ toString BUF_V_OFF ++ "\n" ++
  "  v15 = load.i32 notrap aligned v0+" ++ toString BUF_SCORES_OFF ++ "\n" ++
  "  v16 = load.i32 notrap aligned v0+" ++ toString BUF_PROBS_OFF ++ "\n" ++
  "  v17 = load.i32 notrap aligned v0+" ++ toString BUF_OUT_OFF ++ "\n" ++
  "  v18 = load.i32 notrap aligned v0+" ++ toString BUF_META_OFF ++ "\n" ++
  "  v19 = iconst.i32 1\n" ++
  "  v20 = iconst.i32 0\n" ++
  "  v21 = iconst.i32 " ++ toString HEAD_DIM ++ "\n" ++
  "  v22 = iconst.i32 " ++ toString N_HEADS ++ "\n" ++
  "  v23 = iconst.i64 " ++ toString HEAD_DIM ++ "\n" ++
  "  v24 = imul v10, v23\n" ++                    -- seq_len * head_dim (elements)
  "  v25 = iconst.i32 0x3e000000\n" ++            -- 0.125
  "  v26 = iconst.i32 0x3f800000\n" ++            -- 1.0
  -- scores = 0.125 * K @ q, batched over heads
  "  v30 = call fn1(v0, v19, v20, v11, v19, v21, v25, v13, v24, v12, v23, v20, v15, v10, v22)\n" ++
  -- probs = softmax(scores)
  "  v31 = iconst.i64 " ++ toString PTX_SOURCE_OFF ++ "\n" ++
  "  v32 = iconst.i32 3\n" ++
  "  v33 = iconst.i64 " ++ toString BIND_DESC_OFF ++ "\n" ++
  "  v34 = iconst.i32 256\n" ++
  "  store notrap aligned v15, v0+" ++ toString BIND_DESC_OFF ++ "\n" ++
  "  store notrap aligned v18, v0+" ++ toString (BIND_DESC_OFF + 4) ++ "\n" ++
  "  store notrap aligned v16, v0+" ++ toString (BIND_DESC_OFF + 8) ++ "\n" ++
  "  v35 = call fn0(v0, v31, v32, v33, v22, v19, v19, v34, v19, v19)\n" ++
  -- out = V^T @ probs, batched over heads
  "  v36 = call fn1(v0, v20, v20, v21, v19, v11, v26, v14, v24, v16, v10, v20, v17, v23, v22)\n" ++
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
  "  v3 = load.i32 notrap aligned v0+" ++ toString BUF_OUT_OFF ++ "\n" ++
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

def clifIR : String := clifNoopFn ++ "\n" ++ clifLoadFn ++ "\n" ++ clifPrepFn ++ "\n" ++ clifCoreFn ++ "\n" ++ clifFinalizeFn

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [3, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros PTX_SOURCE_OFF
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

end CudaDecodeAttention

def main : IO Unit := do
  let json := Json.arr #[
    toJson CudaDecodeAttention.buildConfig,
    toJson CudaDecodeAttention.loadAlgorithm,
    toJson CudaDecodeAttention.prepAlgorithm,
    toJson CudaDecodeAttention.inferAlgorithm,
    toJson (CudaDecodeAttention.stackAlgorithm 64)
  ]
  IO.println (Json.compress json)
