import Lean
import Std
import AlgorithmLib
import AlgorithmLib.Cuda
import Qwen2Proven

set_option maxRecDepth 4096

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.PTX
open AlgorithmLib.Tensor
open AlgorithmLib.Layout (Region RegionMap)

namespace Qwen2Common

/-!
  Architecture: hidden=896, ffn=4864, 24 layers, 14 Q heads, 2 KV heads,
                head_dim=64, vocab=151936, rope_theta=1000000

  Weight file format (all f32, produced by tools/qwen2_convert.py):
    [embed_tokens: VOCAB × D]
    For each layer 0..23: rms_attn[D], Wq[D×D], bq[D], Wk[KV_DIM×D], bk[KV_DIM],
                          Wv[KV_DIM×D], bv[KV_DIM], Wo[D×D],
                          rms_ffn[D], Wg[D_FF×D], Wu[D_FF×D], Wd[D×D_FF]
    [rms_final: D]
    [lm_head: VOCAB × D]

  Protocol per execute call:  data=[token_id:u32][pos:u32], out=[next_token:u32]
-/

-- ── Constants ────────────────────────────────────────────────────────────────

def D        : Nat := 896
def D_FF     : Nat := 4864
def N_Q      : Nat := 14
def N_KV     : Nat := 2
def HEAD_DIM : Nat := 64
def KV_DIM   : Nat := N_KV * HEAD_DIM   -- 128
def VOCAB    : Nat := 151936
def N_LAYERS : Nat := 24
def MAX_SEQ  : Nat := 2048
def GQA_RATIO : Nat := N_Q / N_KV       -- 7

def D_BYTES     : Nat := D * 4            -- 3584
def KV_BYTES    : Nat := KV_DIM * 4      -- 512
def WQ_BYTES    : Nat := D * D * 4       -- 3211264
def WK_BYTES    : Nat := KV_DIM * D * 4  -- 458752
def WO_BYTES    : Nat := WQ_BYTES
def WG_BYTES    : Nat := D_FF * D * 4   -- 17432576
def EMBED_BYTES : Nat := VOCAB * D * 4   -- 544620544
-- KV cache: [N_KV, MAX_SEQ, HEAD_DIM] (GQA-proper, broadcast at attention time)
def KV_CACHE_BYTES : Nat := N_KV * MAX_SEQ * HEAD_DIM * 4  -- 1048576 (was 7340032 GQA-expanded)

-- Per-layer file offsets from layer base:
def LF_RMS_ATTN : Nat := 0
def LF_WQ  : Nat := LF_RMS_ATTN + D_BYTES
def LF_BQ  : Nat := LF_WQ  + WQ_BYTES
def LF_WK  : Nat := LF_BQ  + D_BYTES
def LF_BK  : Nat := LF_WK  + WK_BYTES
def LF_WV  : Nat := LF_BK  + KV_BYTES
def LF_BV  : Nat := LF_WV  + WK_BYTES
def LF_WO  : Nat := LF_BV  + KV_BYTES
def LF_RMS_FFN : Nat := LF_WO + WO_BYTES
def LF_WG  : Nat := LF_RMS_FFN + D_BYTES
def LF_WU  : Nat := LF_WG + WG_BYTES
def LF_WD  : Nat := LF_WU + WG_BYTES
def LAYER_BYTES : Nat := LF_WD + WG_BYTES  -- 59649536

def FILE_EMBED_OFF     : Nat := 0
def FILE_LAYER_OFF (l : Nat) : Nat := EMBED_BYTES + l * LAYER_BYTES
def FILE_RMS_FINAL_OFF : Nat := EMBED_BYTES + N_LAYERS * LAYER_BYTES
def FILE_LM_HEAD_OFF   : Nat := FILE_RMS_FINAL_OFF + D_BYTES

-- ── Shared memory layout ─────────────────────────────────────────────────────

-- 0x0000-0x0037: IoOffsets (56 bytes, written by runtime)
def PINNED_HOST_PTR_OFF : Nat := 0x0038  -- i64: host ptr of pinned scratch buffer (cl_cuda_pinned_ptr)
def PINNED_ID_OFF       : Nat := 0x0040  -- i32: pinned buffer id (for free at finalize)

-- Chunk size for streaming weight uploads through pinned host memory.
-- 64 MiB: large enough to amortize per-call overhead, small enough to keep
-- peak host RAM bounded; uploads pipeline through the driver DMA engine.
def PINNED_CHUNK_BYTES : Nat := 64 * 1024 * 1024

-- Maximum tokenizer file size we'll hold in pinned host memory. The Qwen2
-- tokenizer is ~10 MiB; 32 MiB is plenty of headroom and bounds host RAM.
def TOK_FILE_MAX_BYTES : Nat := 32 * 1024 * 1024
-- Per-call buffer slots in shared memory at 0x0048–0x0087 (16 × i32 ids).
-- Each `BufferSlot s` makes the *shape* of the underlying tensor part of the
-- type: `slotHidden.load ptr : VecD`, no per-call cast.  The fixed hex
-- offsets are the single source of truth — no `BUF_*_OFF` constants.
def slotHidden   : BufferSlot [.sta D]              := slotOfAt 0x0048
def slotHdNorm   : BufferSlot [.sta D]              := slotOfAt 0x004C
def slotQ        : BufferSlot [.sta D]              := slotOfAt 0x0050
def slotKCur     : BufferSlot [.sta KV_DIM]         := slotOfAt 0x0054
def slotVCur     : BufferSlot [.sta KV_DIM]         := slotOfAt 0x0058
def slotAttnOut  : BufferSlot [.sta D]              := slotOfAt 0x005C
def slotFfGate   : BufferSlot [.sta D_FF]           := slotOfAt 0x0060
def slotFfUp     : BufferSlot [.sta D_FF]           := slotOfAt 0x0064
def slotFfAct    : BufferSlot [.sta D_FF]           := slotOfAt 0x0068
def slotEmbed    : BufferSlot [.sta VOCAB, .sta D]  := slotOfAt 0x006C
def slotLmHead   : BufferSlot [.sta VOCAB, .sta D]  := slotOfAt 0x0070
def slotLogits   : BufferSlot [.sta VOCAB]          := slotOfAt 0x0074
def slotRmsFinal : BufferSlot [.sta D]              := slotOfAt 0x0078
def slotScores   : BufferSlot [.sta N_Q, .dyn]      := slotOfAt 0x007C
def slotProbs    : BufferSlot [.sta N_Q, .dyn]      := slotOfAt 0x0080
def slotMeta     : BufferSlot [.sta 6]              := slotOfAt 0x0084

-- RoPE sin/cos table slot — separate region at 0x0638 since the table is
-- created once and reused across all layers.
def slotRopeTable : BufferSlot [.sta 2, .sta MAX_SEQ, .sta (HEAD_DIM/2)] :=
  slotOfAt 0x0638
-- (Rope table slot defined below alongside the typed BufferSlots; uses 0x0638.)
def LAYER_IDX_OFF     : Nat := 0x0088  -- i64: current layer for inferLayerFn
def POS_SLOT_OFF      : Nat := 0x0090  -- i64: current pos
def SEQ_LEN_SLOT_OFF  : Nat := 0x0098  -- i64: seq_len = pos+1
-- Layer buffer ID array: 14 i32s × 24 layers × 4 bytes = 1344 bytes
def LAYER_BUFS_BASE   : Nat := 0x00A0  -- 0x00A0 .. 0x05DF
def LAYER_BUF_STRIDE  : Nat := 56      -- 14 × 4

-- Per-layer buffer slots (offsets within stride):
-- Per-layer slots within a 56-byte layer cell.  Each `BufferSlot s` knows
-- its tensor shape and its offset relative to the start of the cell.  To
-- load a layer's slot: `LayerSlot.rmsAttn.load cellBaseAddr`.
namespace LayerSlot
def rmsAttn : BufferSlot [.sta D]                                 := slotOfAt 0
def wq      : BufferSlot [.sta D, .sta D]                         := slotOfAt 4
def bq      : BufferSlot [.sta D]                                 := slotOfAt 8
def wk      : BufferSlot [.sta KV_DIM, .sta D]                    := slotOfAt 12
def bk      : BufferSlot [.sta KV_DIM]                            := slotOfAt 16
def wv      : BufferSlot [.sta KV_DIM, .sta D]                    := slotOfAt 20
def bv      : BufferSlot [.sta KV_DIM]                            := slotOfAt 24
def wo      : BufferSlot [.sta D, .sta D]                         := slotOfAt 28
def rmsFfn  : BufferSlot [.sta D]                                 := slotOfAt 32
def wg      : BufferSlot [.sta D_FF, .sta D]                      := slotOfAt 36
def wu      : BufferSlot [.sta D_FF, .sta D]                      := slotOfAt 40
def wd      : BufferSlot [.sta D, .sta D_FF]                      := slotOfAt 44
def kCache  : BufferSlot [.sta N_KV, .sta MAX_SEQ, .sta HEAD_DIM] := slotOfAt 48
def vCache  : BufferSlot [.sta N_KV, .sta MAX_SEQ, .sta HEAD_DIM] := slotOfAt 52
end LayerSlot

-- Bind descriptor areas (kernel arg tables in shared memory):
def BIND_BASE    : Nat := 0x0800
def BIND_EMBED   : Nat := BIND_BASE + 0x00
def BIND_RMS1    : Nat := BIND_BASE + 0x10
def BIND_BIAS_Q  : Nat := BIND_BASE + 0x20
def BIND_BIAS_K  : Nat := BIND_BASE + 0x28
def BIND_BIAS_V  : Nat := BIND_BASE + 0x30
def BIND_ROPE_Q  : Nat := BIND_BASE + 0x38  -- 3 bufs: q_buf, meta_buf, rope_table
def BIND_ROPE_K  : Nat := BIND_BASE + 0x44  -- 3 bufs: k_buf, meta_buf, rope_table
def BIND_KV_K    : Nat := BIND_BASE + 0x50
def BIND_KV_V    : Nat := BIND_BASE + 0x5C
def BIND_SOFTMAX : Nat := BIND_BASE + 0x70
def BIND_RMS2    : Nat := BIND_BASE + 0x80
def BIND_SILU    : Nat := BIND_BASE + 0x90
def BIND_ADD1    : Nat := BIND_BASE + 0xA0
def BIND_ADD2    : Nat := BIND_BASE + 0xA8
def BIND_ARGMAX  : Nat := BIND_BASE + 0xB0

-- Staging for the GPU meta buffer: six u32s assembled here, then uploaded in
-- one shot.  Words 2..5 are the softmax loop bounds — see `Qwen2Proven`'s
-- `SEQ_SLOT`/`CHUNKS_SLOT`/`TAIL_SLOT`/`REM_SLOT`.  They are derived on the
-- host because the warp machine has no shift or mask instruction.  Sits in the
-- free span between the bind tables (which end at `BIND_BASE + 0xB8`) and
-- `PTX_EMBED_OFF`.
def META_STAGE_OFF : Nat := 0x0900  -- 24 bytes

-- Tokenizer + CLI slots (free space 0x05E0–0x07FF):
def TOK_BUF_PTR_OFF : Nat := 0x05E0  -- i64: host ptr to tokenizer file contents
def INFER_IN_OFF    : Nat := 0x0600  -- 8 bytes: [token_id:u32][pos:u32] for inferFn
def INFER_OUT_OFF   : Nat := 0x0608  -- 4 bytes: next_token:u32 from inferFn
def TOKEN_COUNT_OFF : Nat := 0x0610  -- i64: token count (prompt tokens or output tokens)
def N_PROMPT_OFF    : Nat := 0x0618  -- i64: prompt token count (saved before decode loop)
def TEXT_LEN_OFF    : Nat := 0x0620  -- i64: input/output text byte count
def HT_KEY_OFF      : Nat := 0x0628  -- 8 bytes: scratch key for ht_lookup (tok_a, tok_b)
def HT_VAL_OFF      : Nat := 0x0630  -- 8 bytes: scratch result for ht_lookup (rank, result_tok)

-- Parsed argument pointers (populated by parseArgsFn from data_ptr)
def WEIGHTS_PATH_PTR_OFF   : Nat := 0x0640  -- i64: ptr to null-terminated weights path
def TOKENIZER_PATH_PTR_OFF : Nat := 0x0648  -- i64: ptr to null-terminated tokenizer path

-- Multi-turn conversation state (persists across cliLoop iterations).
def RUNNING_POS_OFF        : Nat := 0x0650  -- i64: cumulative KV cache position

-- Pre-tokenized system prompt placed in initial_memory; fed into the KV cache
-- once at program start so every conversation is rooted in a proper Qwen-style
-- <|im_start|>system ... <|im_end|>\n preamble.
def SYSTEM_TOKENS_OFF      : Nat := 0x0700  -- 256 bytes of u32 token IDs

/-- Pre-tokenized system prompt:
    `<|im_start|>system\nYou are a friendly conversational assistant. Reply directly
    and naturally. Match the user's tone — if they greet you casually, greet back
    casually.<|im_end|>\n` -/
def systemTokenIds : List Nat :=
  [151644, 8948, 198] ++                                  -- <|im_start|>system\n
  [2610, 525, 264, 11657, 7517, 1663, 17847, 13,
   17841, 5961, 323, 17712, 13, 14152, 279, 1196, 594,
   16232, 1959, 421, 807, 40786, 498, 64614, 11,
   40786, 1182, 64614, 13] ++                             -- prompt body
  [151645, 198]                                           -- <|im_end|>\n

def SYSTEM_TOKEN_COUNT : Nat := systemTokenIds.length

-- PTX kernel source offsets in shared memory:
-- RMS PTX is ~2668 bytes so needs a 4096-byte slot (not 2048).
-- Every offset from BIAS_D onward is shifted +0x800 relative to the original layout.
def PTX_EMBED_OFF   : Nat := 0x1000
def PTX_RMS_OFF     : Nat := 0x2000
def PTX_BIAS_D_OFF  : Nat := 0x3000  -- was 0x2800; RMS needs 4096-byte slot
def PTX_BIAS_KV_OFF : Nat := 0x3800
-- The proven kernels are larger than the hand-written ones they replace (they
-- print one instruction per line with a label on each), so the slots grew.
def PTX_ROPE_Q_OFF  : Nat := 0x4000  -- proven RoPE needs a 4096-byte slot
def PTX_ROPE_K_OFF  : Nat := 0x5000
-- The proven softmax is six loops (a chunk sweep and a remainder sweep per
-- pass), so it needs an 8 KB slot; everything after it moved up.
def PTX_SOFTMAX_OFF : Nat := 0x6000
def PTX_SILU_OFF    : Nat := 0x8000
def PTX_ADD_OFF     : Nat := 0x8C00
def PTX_KVSTORE_OFF : Nat := 0x9800
def PTX_ARGMAX_OFF  : Nat := 0xA800

-- Tokenize / server buffers.
--
-- `TOKEN_BUF_CAP` and `MAX_RECV` are two constants that must agree:
-- `tokenizeInitFn` writes **one u32 per input byte**, so the token buffer must
-- hold as many entries as the reader can deliver bytes.  `tokenBuf_holds_input`
-- below is that obligation; before it existed the two numbers disagreed and a
-- single input line over the buffer's capacity wrote through into `text_in`,
-- the buffer being read.
-- The chat wrapper written around every prompt: `<|im_start|>user\n` (6 tokens)
-- and `<|im_end|>\n<|im_start|>assistant\n` (13), so 19 in total.
def PREFIX_LEN      : Nat := 6
def WRAP_LEN        : Nat := 19
/-- Tokens the decode loop may generate before stopping. -/
def MAX_DECODE      : Nat := 128

/-- Bytes `cl_stdin_readline` may deliver.  **Derived, not chosen**: one initial
    token per input byte, plus the wrapper, plus whatever decode generates, must
    all fit the KV cache alongside the system prompt.  Picking this number by
    hand is what let it sit at 8192 against a 2048-position cache. -/
def MAX_RECV        : Nat := MAX_SEQ - SYSTEM_TOKEN_COUNT - WRAP_LEN - MAX_DECODE
/-- Token slots: the initial one-per-byte pass, then the wrapper shifted in. -/
def TOKEN_BUF_CAP   : Nat := MAX_RECV + WRAP_LEN
def TOKEN_BUF_BYTES : Nat := TOKEN_BUF_CAP * 4
def TEXT_IN_BYTES   : Nat := 8 * 1024
def TEXT_OUT_BYTES  : Nat := 8 * 1024

def TOKEN_BUF_OFF   : Nat := 0xB800
def TEXT_IN_OFF     : Nat := TOKEN_BUF_OFF + TOKEN_BUF_BYTES
def TEXT_OUT_OFF    : Nat := TEXT_IN_OFF   + TEXT_IN_BYTES
def MEM_SIZE        : Nat := TEXT_OUT_OFF  + TEXT_OUT_BYTES

-- ── PTX Kernels ──────────────────────────────────────────────────────────────

def D_AS_BITS : UInt32 := 0x44600000  -- 896.0f

-- ── Tensor shape abbreviations for Qwen2 ────────────────────────────────────
abbrev VecD     := Tensor [.sta D]              -- hidden state, rmsnorm weights, residual
abbrev VecKV    := Tensor [.sta KV_DIM]         -- current K/V vector for one position
abbrev VecDff   := Tensor [.sta D_FF]           -- FFN intermediate
abbrev VecVocab := Tensor [.sta VOCAB]          -- logits, embed table row
abbrev VecMeta  := Tensor [.sta 6]              -- [token_id, pos, seqLen, chunks, tail, rem]
abbrev EmbedTbl := Tensor [.sta VOCAB, .sta D]  -- full embed/lm_head table
abbrev KVCache  := Tensor [.sta N_KV, .sta MAX_SEQ, .sta HEAD_DIM]  -- GQA-proper (one copy per KV head)
abbrev RopeTbl  := Tensor [.sta 2, .sta MAX_SEQ, .sta (HEAD_DIM/2)]
abbrev VecScores := Tensor [.sta N_Q, .dyn]     -- attention scores [head, seq_len]
abbrev MatDD     := Tensor [.sta D, .sta D]              -- Wq, Wo
abbrev MatKVD    := Tensor [.sta KV_DIM, .sta D]         -- Wk, Wv
abbrev MatDffD   := Tensor [.sta D_FF, .sta D]           -- Wg, Wu
abbrev MatDDff   := Tensor [.sta D, .sta D_FF]           -- Wd

-- Embedding row copy: out = embed[token_id]
-- Bind: [embed, meta, hidden]; Grid=(1,1,1), Block=(256,1,1)
def embedKernel : Kernel := {
  name := "main"
  params := [
    { name := "embed_buf", shape := [.sta VOCAB, .sta D], ro := true },
    { name := "meta_buf",  shape := [.sta 6],            ro := true },
    { name := "out_buf",   shape := [.sta D] }
  ]
  geom := Kernel.Geom.covering D (D / 32) 32
  ptxOff := PTX_EMBED_OFF
  ptxText := some Qwen2Proven.ptxEmbed
}
def ptxEmbedLookup : String := embedKernel.ptxSource

private def launchEmbed (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (table : EmbedTbl) (metaT : VecMeta) (outT : VecD) : IRBuilder Unit :=
  launch3 embedKernel cuda ptr bindOff table metaT outT

-- RMSNorm: y = rms_norm(x, w)
-- Bind: [x_buf, w_buf, y_buf]; Grid=(1,1,1), Block=(256,1,1), smem=36
--
-- Declarative typed kernel: shape-indexed bindings, declarative geometry,
-- and the PTX body in one value. `ptxRmsNorm` (below) is now derived.
def rmsNormKernel : Kernel := {
  name := "main"
  params := [
    { name := "x_buf", shape := [.sta D] },
    { name := "w_buf", shape := [.sta D] },
    { name := "y_buf", shape := [.sta D] }
  ]
  smemBytes := 36
  geom := Kernel.Geom.sweeping D 32 (D / 32)
  ptxOff := PTX_RMS_OFF
  ptxText := some Qwen2Proven.ptxRmsNorm
}

def ptxRmsNorm : String := rmsNormKernel.ptxSource

/-- Typed RMSNorm launcher.  `x` is the input, `w` the weights, `y` the output —
    all `[D]` f32. The bind region is the only call-site-specific value. -/
private def launchRms (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (x w y : Tensor [.sta D]) : IRBuilder Unit :=
  launch3 rmsNormKernel cuda ptr bindOff x w y

def biasAddDKernel : Kernel := {
  name := "main"
  params := [{ name := "x_buf", shape := [.sta D] },
             { name := "b_buf", shape := [.sta D], ro := true }]
  -- **Migrated.**  `x[i] + b[i]` is literally `Qwen2Proven.addSpec`, so this is
  -- the *same proven kernel* at a different width — one spec, two instances.
  geom := Kernel.Geom.covering D (D / 32) 32
  ptxOff := PTX_BIAS_D_OFF
  ptxText := some Qwen2Proven.ptxAdd
}

def biasAddKVKernel : Kernel := {
  name := "main"
  params := [{ name := "x_buf", shape := [.sta KV_DIM] },
             { name := "b_buf", shape := [.sta KV_DIM], ro := true }]
  -- **Migrated**, same spec again at the KV width.
  geom := Kernel.Geom.covering KV_DIM (KV_DIM / 32) 32
  ptxOff := PTX_BIAS_KV_OFF
  ptxText := some Qwen2Proven.ptxAdd
}

def ptxBiasAddD  : String := biasAddDKernel.ptxSource
def ptxBiasAddKV : String := biasAddKVKernel.ptxSource

private def launchBiasD (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (x b : VecD) : IRBuilder Unit :=
  launch2 biasAddDKernel cuda ptr bindOff x b

private def launchBiasKV (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (x b : VecKV) : IRBuilder Unit :=
  launch2 biasAddKVKernel cuda ptr bindOff x b

-- RoPE rotation body — identical for Q and K, parameterized by buffer-param name.
-- Thread (headIdx=ctaX, freqIdx=tidX) handles vec[head, freq] and vec[head, freq+HEAD_DIM/2].
-- meta_buf[4]=pos; rope_table = [sin; cos] each MAX_SEQ × HEAD_DIM/2 f32.
-- RoPE Q: grid=N_Q, block=HEAD_DIM/2. Rotates Q in place.
def ropeQKernel : Kernel := {
  name := "main"
  params := [{ name := "q_buf",      shape := [.sta D] },
             { name := "meta_buf",   shape := [.sta 6], ro := true },
             { name := "rope_table", shape := [.sta 2, .sta MAX_SEQ, .sta (HEAD_DIM/2)], ro := true }]
  -- **Migrated.**  `HEAD_DIM/2 = 32` was already exactly one warp, so the
  -- geometry is unchanged.  See `Qwen2Proven.rope_ptx_exact`.
  geom := Kernel.Geom.covering (N_Q * (HEAD_DIM/2)) N_Q (HEAD_DIM/2)
  ptxOff := PTX_ROPE_Q_OFF
  ptxText := some Qwen2Proven.ptxRope
}

-- RoPE K: grid=N_KV, block=HEAD_DIM/2.
def ropeKKernel : Kernel := {
  name := "main"
  params := [{ name := "k_buf",      shape := [.sta KV_DIM] },
             { name := "meta_buf",   shape := [.sta 6], ro := true },
             { name := "rope_table", shape := [.sta 2, .sta MAX_SEQ, .sta (HEAD_DIM/2)], ro := true }]
  -- **Migrated** — the same proven kernel, fewer heads.
  geom := Kernel.Geom.covering (N_KV * (HEAD_DIM/2)) N_KV (HEAD_DIM/2)
  ptxOff := PTX_ROPE_K_OFF
  ptxText := some Qwen2Proven.ptxRope
}

def ptxRoPEQ : String := ropeQKernel.ptxSource
def ptxRoPEK : String := ropeKKernel.ptxSource

private def launchRopeQ (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (q : VecD) (mb : VecMeta) (rope : RopeTbl) : IRBuilder Unit :=
  launch3 ropeQKernel cuda ptr bindOff q mb rope

private def launchRopeK (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (k : VecKV) (mb : VecMeta) (rope : RopeTbl) : IRBuilder Unit :=
  launch3 ropeKKernel cuda ptr bindOff k mb rope

-- Softmax over per-head scores.  **Migrated** to the proven stack: the trip
-- counts come from the meta buffer (`forM`), so one kernel and one theorem
-- cover every sequence length.  Grid=N_Q, Block=32, no shared memory.
def softmaxKernel : Kernel := {
  name := "main"
  params := [{ name := "scores_buf", shape := [.sta N_Q, .dyn] },
             { name := "meta_buf",   shape := [.sta 6], ro := true },
             { name := "probs_buf",  shape := [.sta N_Q, .dyn] }]
  geom := Kernel.Geom.covering (N_Q * 32) N_Q 32
  ptxOff := PTX_SOFTMAX_OFF
  ptxText := some Qwen2Proven.ptxSoftmax
}

def ptxSoftmax : String := softmaxKernel.ptxSource

private def launchSoftmax (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (scores : VecScores) (mb : VecMeta) (probs : VecScores) : IRBuilder Unit :=
  launch3 softmaxKernel cuda ptr bindOff scores mb probs

-- SiLU-gate: out = silu(gate) * up.  Grid=ceil(D_FF/256), Block=256.
def siluGateKernel : Kernel := {
  name := "main"
  params := [{ name := "gate_buf", shape := [.sta D_FF] },
             { name := "up_buf",   shape := [.sta D_FF], ro := true },
             { name := "out_buf",  shape := [.sta D_FF] }]
  geom := Kernel.Geom.covering D_FF (D_FF / 32) 32
  ptxOff := PTX_SILU_OFF
  ptxText := some Qwen2Proven.ptxSilu
}

def ptxSiluGate : String := siluGateKernel.ptxSource

private def launchSiluGate (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (gate up out_ : VecDff) : IRBuilder Unit :=
  launch3 siluGateKernel cuda ptr bindOff gate up out_

-- Residual add: x[i] += a[i], n=D. Grid=ceil(D/256), Block=256.
def residualAddKernel : Kernel := {
  name := "main"
  params := [{ name := "x_buf",   shape := [.sta D] },
             { name := "add_buf", shape := [.sta D], ro := true }]
  geom := Kernel.Geom.covering D (D / 32) 32
  ptxOff := PTX_ADD_OFF
  ptxText := some Qwen2Proven.ptxAdd
}

def ptxResidualAdd : String := residualAddKernel.ptxSource

private def launchResidualAdd (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (x add_ : VecD) : IRBuilder Unit :=
  launch2 residualAddKernel cuda ptr bindOff x add_

-- KV store (GQA-proper). Writes k_cur[kvHead, elemIdx] → kCache[kvHead, pos, elemIdx].
-- One thread per (kvHead, elemIdx). Grid=N_KV, Block=HEAD_DIM.
def kvStoreKernel : Kernel := {
  name := "main"
  params := [{ name := "k_cur_buf",   shape := [.sta KV_DIM], ro := true },
             { name := "k_cache_buf", shape := [.sta N_KV, .sta MAX_SEQ, .sta HEAD_DIM] },
             { name := "meta_buf",    shape := [.sta 6], ro := true }]
  geom := Kernel.Geom.covering (N_KV * 32) N_KV 32
  ptxOff := PTX_KVSTORE_OFF
  ptxText := some Qwen2Proven.ptxKVStore
}

def ptxKVStore : String := kvStoreKernel.ptxSource

private def launchKVStore (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (kCur : VecKV) (kCache : KVCache) (mb : VecMeta) : IRBuilder Unit :=
  launch3 kvStoreKernel cuda ptr bindOff kCur kCache mb

-- Argmax over VOCAB logits. Single-thread; writes result to meta_buf[0].
def argmaxKernel : Kernel := {
  name := "main"
  params := [{ name := "logits_buf", shape := [.sta VOCAB], ro := true },
             { name := "meta_buf",   shape := [.sta 6] }]
  geom := Kernel.Geom.sweeping VOCAB 32 (VOCAB / 32)
  ptxOff := PTX_ARGMAX_OFF
  ptxText := some Qwen2Proven.ptxArgmax
}

def ptxArgmax : String := argmaxKernel.ptxSource

private def launchArgmax (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (logits : VecVocab) (mb : VecMeta) : IRBuilder Unit :=
  launch2 argmaxKernel cuda ptr bindOff logits mb

-- ── CLIF Load Functions ───────────────────────────────────────────────────────

/-- Advance past the next null byte in a host buffer and return the pointer
    immediately after it. Used to split a concatenated `a\0b\0c\0` arg payload. -/
private def walkPastNull (start : Val) : IRBuilder Val := do
  let atNull ← whileLoop1 .i64 start
    (fun p => do icmp .ne (← uload8_64 p) (← iconst64 0))
    (fun p => iaddImm p 1)
  iaddImm atNull 1

/-- parseArgsFn (fn_37): split the caller's data buffer (two null-terminated
    strings: `weights_path\0tokenizer_path\0`) into two pointers and store them
    in shared memory so later actions can locate their argument. -/
def parseArgsFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  storeI64 dataPtr (← absAddr ptr WEIGHTS_PATH_PTR_OFF)
  let tokenizerPtr ← walkPastNull dataPtr
  storeI64 tokenizerPtr (← absAddr ptr TOKENIZER_PATH_PTR_OFF)
  ret

/-- Upload a tensor from disk into a GPU buffer via a pinned host scratch buffer.
    Statically unrolls into one (read, upload) pair per PINNED_CHUNK_BYTES chunk.
    The scratch buffer is reused for every chunk (host→GPU is synchronous), and
    each chunk goes to a distinct offset within the destination GPU buffer. -/
def uploadFromFile {s : Shape} (cuda : CudaSetup) (fnFileRead : FnRef)
    (ctxPtr pathPtr scratchPtr : Val) (t : Tensor s)
    (fileOff totalSize : Nat) : IRBuilder Unit := do
  let bufId := t.buf
  let numChunks := (totalSize + PINNED_CHUNK_BYTES - 1) / PINNED_CHUNK_BYTES
  (List.range numChunks).forM fun i => do
    let off      := i * PINNED_CHUNK_BYTES
    let thisSize := min PINNED_CHUNK_BYTES (totalSize - off)
    let fileOff64 ← iconst64 (fileOff + off)
    let size64    ← iconst64 thisSize
    let _ ← call fnFileRead [pathPtr, scratchPtr, fileOff64, size64]
    let bufOff64  ← iconst64 off
    let _ ← call cuda.fnUploadOffset [ctxPtr, bufId, bufOff64, scratchPtr, size64]

/-- Generate the RoPE sin/cos table into the pinned scratch buffer, then upload it.
    Loop: for each freq in 0..HEAD_DIM/2, inv_freq = rope_theta^(-2*freq/HEAD_DIM);
          for each pos in 0..MAX_SEQ, write sin/cos of (pos*inv_freq) to scratch.
    Table layout: [sin: MAX_SEQ × HEAD_DIM/2 f32][cos: MAX_SEQ × HEAD_DIM/2 f32].
    All trig goes through libm via FFI (cl_sinf/cl_cosf/cl_powf). -/
private def buildRopeTable (cuda : CudaSetup) (fnSinf fnCosf fnPowf : FnRef)
    (ctxPtr scratchPtr : Val) (ropeTable : RopeTbl) : IRBuilder Unit := do
  let bufRopeTable := ropeTable.buf
  let hdh   : Nat := HEAD_DIM / 2
  let tableBytes := 2 * MAX_SEQ * hdh * 4
  let cosOff     := MAX_SEQ * hdh * 4

  let hdh64     ← iconst64 hdh
  let posLim64  ← iconst64 MAX_SEQ
  let freqLim64 ← iconst64 hdh
  let four64    ← iconst64 4
  let cosOff64  ← iconst64 cosOff
  let ropeTheta ← fconst32 "0x1.e84800p19"   -- 1000000.0
  -- exponent factor = -2.0 / HEAD_DIM = -1/32 for HEAD_DIM=64
  let expFactor ← fconst32 "-0x1.000000p-5"

  forLoop .i64 freqLim64 fun freq => do
    let freqF ← fcvtFromSint .f32 freq
    let exponent ← fmul freqF expFactor
    let invFreq ← call fnPowf [ropeTheta, exponent]
    -- Inner loop carries invFreq through (loop-invariant; could also rely on
    -- dominance, but threading is more conservative across the back-edge).
    let _ ← forLoopAcc .i64 .f32 posLim64 invFreq fun pos invF => do
      let posF ← fcvtFromSint .f32 pos
      let theta ← fmul posF invF
      let sinV ← call fnSinf [theta]
      let cosV ← call fnCosf [theta]
      let row     ← imul pos hdh64
      let idx     ← iadd row freq
      let byteOff ← imul idx four64
      let sinAddr ← iadd scratchPtr byteOff
      let cosAddrOff ← iadd byteOff cosOff64
      let cosAddr ← iadd scratchPtr cosAddrOff
      storeF32 sinV sinAddr
      storeF32 cosV cosAddr
      return invF
    pure ()

  let _ ← call cuda.fnUpload [ctxPtr, bufRopeTable, scratchPtr, (← iconst64 tableBytes)]

/-- Shared body of `loadInitFn`: allocates pinned scratch, activation/embed/
    lm_head/rope buffers, streams the embed table, lm_head, and rms_final
    weights from disk.  Does NOT emit `entryBlock` or `ret` — caller wraps it.
    The caller passes in already-declared FFI handles to avoid duplicate decls. -/
def loadInitCommon (cuda : CudaSetup)
    (fnFileRead fnSinf fnCosf fnPowf : FnRef) (ptr : Val) : IRBuilder Unit := do
  -- Weights file path was parsed into a shared-memory slot by fn_37.
  let pathPtr ← load64 (← absAddr ptr WEIGHTS_PATH_PTR_OFF)

  -- Init CUDA context
  cudaInit cuda ptr 0x10
  let ctxPtr ← load64 (← absAddr ptr 0x10)

  -- Allocate pinned host scratch buffer for streaming weight uploads
  let chunkBytes64 ← iconst64 PINNED_CHUNK_BYTES
  let pinnedId  ← call cuda.fnPinnedAlloc [ctxPtr, chunkBytes64]
  let pinnedPtr ← call cuda.fnPinnedPtr   [ctxPtr, pinnedId]
  storeI32 pinnedId  (← absAddr ptr PINNED_ID_OFF)
  storeI64 pinnedPtr (← absAddr ptr PINNED_HOST_PTR_OFF)

  -- Create activation buffers
  let dBytes   ← iconst64 D_BYTES
  let kvBytes  ← iconst64 KV_BYTES
  let dffBytes ← iconst64 (D_FF * 4)

  let bufHidden  : VecD   ← Tensor.create cuda ptr dBytes
  let bufHdNorm  : VecD   ← Tensor.create cuda ptr dBytes
  let bufQ       : VecD   ← Tensor.create cuda ptr dBytes
  let bufKCur    : VecKV  ← Tensor.create cuda ptr kvBytes
  let bufVCur    : VecKV  ← Tensor.create cuda ptr kvBytes
  let bufAttnOut : VecD   ← Tensor.create cuda ptr dBytes
  let bufFfGate  : VecDff ← Tensor.create cuda ptr dffBytes
  let bufFfUp    : VecDff ← Tensor.create cuda ptr dffBytes
  let bufFfAct   : VecDff ← Tensor.create cuda ptr dffBytes

  slotHidden.store  ptr bufHidden
  slotHdNorm.store  ptr bufHdNorm
  slotQ.store       ptr bufQ
  slotKCur.store    ptr bufKCur
  slotVCur.store    ptr bufVCur
  slotAttnOut.store ptr bufAttnOut
  slotFfGate.store  ptr bufFfGate
  slotFfUp.store    ptr bufFfUp
  slotFfAct.store   ptr bufFfAct

  -- Create embed, lm_head, logits, rms_final, scores, probs, meta buffers
  let embedBytes  ← iconst64 EMBED_BYTES
  let vocabBytes  ← iconst64 (VOCAB * 4)
  let scoreBytes  ← iconst64 (N_Q * MAX_SEQ * 4)
  let metaBytes   ← iconst64 24

  let bufEmbed    : EmbedTbl  ← Tensor.create cuda ptr embedBytes
  let bufLmHead   : EmbedTbl  ← Tensor.create cuda ptr embedBytes
  let bufLogits   : VecVocab  ← Tensor.create cuda ptr vocabBytes
  let bufRmsFinal : VecD      ← Tensor.create cuda ptr dBytes
  let bufScores   : VecScores ← Tensor.create cuda ptr scoreBytes
  let bufProbs    : VecScores ← Tensor.create cuda ptr scoreBytes
  let bufMeta     : VecMeta   ← Tensor.create cuda ptr metaBytes

  slotEmbed.store    ptr bufEmbed
  slotLmHead.store   ptr bufLmHead
  slotLogits.store   ptr bufLogits
  slotRmsFinal.store ptr bufRmsFinal
  slotScores.store   ptr bufScores
  slotProbs.store    ptr bufProbs
  slotMeta.store     ptr bufMeta

  -- Create RoPE sin/cos table buffer (typed) and populate via libm-driven loop.
  let ropeTableBytes ← iconst64 (2 * MAX_SEQ * (HEAD_DIM / 2) * 4)  -- 524288
  let bufRopeTable : RopeTbl ← Tensor.create cuda ptr ropeTableBytes
  slotRopeTable.store ptr bufRopeTable
  buildRopeTable cuda fnSinf fnCosf fnPowf ctxPtr pinnedPtr bufRopeTable

  -- Stream-upload embedding, rms_final, lm_head through pinned scratch
  uploadFromFile cuda fnFileRead ctxPtr pathPtr pinnedPtr bufEmbed    FILE_EMBED_OFF     EMBED_BYTES
  uploadFromFile cuda fnFileRead ctxPtr pathPtr pinnedPtr bufRmsFinal FILE_RMS_FINAL_OFF D_BYTES
  uploadFromFile cuda fnFileRead ctxPtr pathPtr pinnedPtr bufLmHead   FILE_LM_HEAD_OFF   EMBED_BYTES



-- ── CLIF Infer Functions ──────────────────────────────────────────────────────

/-- **The meta buffer's six words, as an extracted fragment.**

    `[token_id, pos, seqLen, chunks, tail, rem]`.  `chunks`/`tail`/`rem` are the
    proven softmax's loop bounds; deriving them here keeps the kernel free of
    the shift and mask it cannot express, and keeps them lane-uniform by
    construction (they live in memory, not a register).  See `Qwen2Proven.smMax`
    for why the split is needed at all.

    Split out of `inferFn` **only** so it can be scanned: `metaStageFrag_emits`
    below is a statement about a nineteen-instruction list rather than about
    `inferFn`'s whole body, and it is what turns "the host publishes `seq`,
    `seq/32`, `(seq/32)*32`, `seq%32`" from something a reader checks by eye
    into a theorem.  The emitted instructions are unchanged — `do`-notation is
    associative and the artifact md5 is the same. -/
def metaStageFrag (ptr dataPtr pos32 seqLen64 : Val) : IRBuilder Val := do
  let tokId32   ← load32 dataPtr
  let seqLen32b ← ireduce32 seqLen64
  let chunks32  ← ushrImm seqLen32b 5
  let tail32    ← ishlImm chunks32 5
  let rem32     ← isub seqLen32b tail32
  let stage     ← absAddr ptr META_STAGE_OFF
  storeI32 tokId32   stage
  storeI32 pos32     (← iaddImm stage 4)
  storeI32 seqLen32b (← iaddImm stage 8)
  storeI32 chunks32  (← iaddImm stage 12)
  storeI32 tail32    (← iaddImm stage 16)
  storeI32 rem32     (← iaddImm stage 20)
  pure stage

/-- **Exactly what `metaStageFrag` emits**, in program order, as a function of
    the SSA counter it starts from.

    Read the four lines that matter off it: `%(n+3) = ushr %(n+1), 5`,
    `%(n+5) = ishl %(n+3), 5`, `%(n+6) = isub %(n+1), %(n+5)`, and the four
    `store.i32` at `stage + 8/12/16/20` — which are `SEQ_SLOT`, `CHUNKS_SLOT`,
    `TAIL_SLOT`, `REM_SLOT` of the meta buffer, the four values
    `Qwen2NonVacuity.smMeta_of_seqLen` needs. -/
def metaFragInsts (ptr dataPtr pos32 seqLen64 : Val) (n : Nat) : List Inst :=
  [ .load ⟨n⟩ "load.i32" dataPtr                     -- token id
  , .ireduce32 ⟨n+1⟩ seqLen64                        -- seq   := (i32) seqLen
  , .iconst ⟨n+2⟩ .i64 5
  , .ushr ⟨n+3⟩ ⟨n+1⟩ ⟨n+2⟩                          -- chunks := seq >>> 5
  , .iconst ⟨n+4⟩ .i64 5
  , .ishl ⟨n+5⟩ ⟨n+3⟩ ⟨n+4⟩                          -- tail   := chunks <<< 5
  , .isub ⟨n+6⟩ ⟨n+1⟩ ⟨n+5⟩                          -- rem    := seq - tail
  , .iconst ⟨n+7⟩ .i64 META_STAGE_OFF
  , .iadd ⟨n+8⟩ ptr ⟨n+7⟩                            -- stage
  , .storeTyped .i32 ⟨n⟩ ⟨n+8⟩                       -- stage[0]  = token id
  , .iconst ⟨n+9⟩ .i64 4
  , .iadd ⟨n+10⟩ ⟨n+8⟩ ⟨n+9⟩
  , .storeTyped .i32 pos32 ⟨n+10⟩                    -- stage[4]  = pos
  , .iconst ⟨n+11⟩ .i64 8
  , .iadd ⟨n+12⟩ ⟨n+8⟩ ⟨n+11⟩
  , .storeTyped .i32 ⟨n+1⟩ ⟨n+12⟩                    -- stage[8]  = seq     (SEQ_SLOT)
  , .iconst ⟨n+13⟩ .i64 12
  , .iadd ⟨n+14⟩ ⟨n+8⟩ ⟨n+13⟩
  , .storeTyped .i32 ⟨n+3⟩ ⟨n+14⟩                    -- stage[12] = chunks  (CHUNKS_SLOT)
  , .iconst ⟨n+15⟩ .i64 16
  , .iadd ⟨n+16⟩ ⟨n+8⟩ ⟨n+15⟩
  , .storeTyped .i32 ⟨n+5⟩ ⟨n+16⟩                    -- stage[16] = tail    (TAIL_SLOT)
  , .iconst ⟨n+17⟩ .i64 20
  , .iadd ⟨n+18⟩ ⟨n+8⟩ ⟨n+17⟩
  , .storeTyped .i32 ⟨n+6⟩ ⟨n+18⟩ ]                  -- stage[20] = rem     (REM_SLOT)

/-- **The fragment emits exactly that, from any state.**

    `currentInsts` is kept reversed for O(1) prepend, hence the `.reverse`.
    Quantified over the incoming `IRState`, so this says what the fragment does
    wherever it is called from — it is not a claim about one call site. -/
theorem metaStageFrag_emits (ptr dataPtr pos32 seqLen64 : Val) (s : IRState) :
    ((metaStageFrag ptr dataPtr pos32 seqLen64).run s).2.currentInsts
      = (metaFragInsts ptr dataPtr pos32 seqLen64 s.nextVal).reverse
          ++ s.currentInsts := rfl

open AlgorithmLib.Clif in
/-- **What the fragment leaves in memory, as expressions over the runtime
    sequence length.**

    This is the step `SmMeta` was missing.  `metaStageFrag_emits` says which
    instructions are emitted; this says what the store map holds afterwards —
    `seq`, `seq >>> 5`, `(seq >>> 5) <<< 5` and `seq − ((seq >>> 5) <<< 5)` at
    the four slots the softmax kernel reads.  `seq` itself is a runtime load, so
    none of this was expressible before `Clif.DExp`: every one of the four was
    `unknown`, and "the host publishes the loop bounds the kernel assumes" was
    not a sentence this model could write down.

    Nothing is assumed about the incoming store map — the fragment writes all
    six words itself, and `StoreMap.get?` returns the most recent.  The token id
    is left as the model's own reading of a load through a pointer it does not
    resolve; the softmax kernel does not read it.

    The `< n` hypotheses are the compiler's own convention: the values the
    fragment is handed were allocated before it, so they cannot collide with its
    temporaries.  `he` is the other half of the same convention — the descriptor
    pointer is a runtime input.

    `hnc` is the one hypothesis that is about this program rather than about the
    compiler's conventions: `stepPure` constant-folds, so a *compile-time*
    sequence length would put four constants in these slots instead of four
    expressions.  That case is not a gap — it is `Qwen2NonVacuity.smMetaW`, where
    `SmMeta` is decided outright.  This theorem is about the other one, which is
    what the shipped program does: `seq` is `pos + 1` and `pos` is a load. -/
theorem metaFrag_mem (ptr dataPtr pos32 seqLen64 : Val)
    (e : Env) (m : StoreMap) (n : Nat) (S : DExp)
    (hptr : ptr.id < n) (hdp : dataPtr.id < n) (hpo : pos32.id < n)
    (hseq : seqLen64.id < n) (he : e ptr = SymVal.unknown)
    (hS : (e seqLen64).toD? = some S) (hnc : ∀ k : Int, e seqLen64 ≠ SymVal.const k) :
    (bevalPure ⟨e, m⟩ (metaFragInsts ptr dataPtr pos32 seqLen64 n)).mem
        = [ ((ptr.id, (META_STAGE_OFF : Int) + 20),
             SymVal.derived (.sub S (.shl (.shr S 5) 5)))
          , ((ptr.id, (META_STAGE_OFF : Int) + 16),
             SymVal.derived (.shl (.shr S 5) 5))
          , ((ptr.id, (META_STAGE_OFF : Int) + 12), SymVal.derived (.shr S 5))
          , ((ptr.id, (META_STAGE_OFF : Int) + 8),  e seqLen64)
          , ((ptr.id, (META_STAGE_OFF : Int) + 4),  e pos32)
          , ((ptr.id, (META_STAGE_OFF : Int)),
             (match e dataPtr with
              | .offset p k => SymVal.slot p k
              | _           => SymVal.unknown)) ] ++ m := by
  -- the fragment's temporaries are all `n + j`, and everything it was handed
  -- sits below `n`; stating that once keeps the reduction linear
  have hpj : ∀ j : Nat, (ptr.id = n + j) = False := by
    intro j; simp only [eq_iff_iff, iff_false]; omega
  have hdj : ∀ j : Nat, (dataPtr.id = n + j) = False := by
    intro j; simp only [eq_iff_iff, iff_false]; omega
  have hoj : ∀ j : Nat, (pos32.id = n + j) = False := by
    intro j; simp only [eq_iff_iff, iff_false]; omega
  have hsj : ∀ j : Nat, (seqLen64.id = n + j) = False := by
    intro j; simp only [eq_iff_iff, iff_false]; omega
  have hp0 : (ptr.id = n) = False := by simp only [eq_iff_iff, iff_false]; omega
  have hd0 : (dataPtr.id = n) = False := by simp only [eq_iff_iff, iff_false]; omega
  have ho0 : (pos32.id = n) = False := by simp only [eq_iff_iff, iff_false]; omega
  have hs0 : (seqLen64.id = n) = False := by simp only [eq_iff_iff, iff_false]; omega
  have hnn : ∀ a b : Nat, (n + a = n + b) = (a = b) := by
    intro a b; simp only [eq_iff_iff]; omega
  have hn0 : ∀ a : Nat, (n = n + (a + 1)) = False := by
    intro a; simp only [eq_iff_iff, iff_false]; omega
  -- `toD?` allows exactly three shapes for `seq`, and `dOf` ignores the SSA
  -- name in all three — which is why the expression the fragment builds at
  -- `%(n+3)` is the one this theorem states about the caller's `seqLen64`.
  cases hv : e seqLen64
  case const k => exact absurd hv (hnc k)
  case slot q k => rw [hv] at hS; exact absurd hS (by simp [SymVal.toD?])
  case unknown => rw [hv] at hS; exact absurd hS (by simp [SymVal.toD?])
  all_goals
    (rw [hv] at hS
     simp only [SymVal.toD?, Option.some.injEq] at hS
     subst hS
     simp only [metaFragInsts, bevalPure, bstep, stepPure, stepMem, Inst.storeOf?,
       Env.set_apply, addSym, dOf, he, hv, hpj, hdj, hoj, hsj, hp0, hd0, ho0, hs0,
       hnn, hn0, if_false, reduceIte]
     simp
     all_goals rfl)

open AlgorithmLib.Clif in
/-- **The four slots the softmax kernel reads**, straight off the store map. -/
theorem metaFrag_stores (ptr dataPtr pos32 seqLen64 : Val)
    (e : Env) (m : StoreMap) (n : Nat) (S : DExp)
    (hptr : ptr.id < n) (hdp : dataPtr.id < n) (hpo : pos32.id < n)
    (hseq : seqLen64.id < n) (he : e ptr = SymVal.unknown)
    (hS : (e seqLen64).toD? = some S) (hnc : ∀ k : Int, e seqLen64 ≠ SymVal.const k) :
    let mm := (bevalPure ⟨e, m⟩ (metaFragInsts ptr dataPtr pos32 seqLen64 n)).mem
    mm.get? ptr.id ((META_STAGE_OFF : Int) + 8)  = some (e seqLen64)
      ∧ mm.get? ptr.id ((META_STAGE_OFF : Int) + 12) = some (.derived (.shr S 5))
      ∧ mm.get? ptr.id ((META_STAGE_OFF : Int) + 16)
          = some (.derived (.shl (.shr S 5) 5))
      ∧ mm.get? ptr.id ((META_STAGE_OFF : Int) + 20)
          = some (.derived (.sub S (.shl (.shr S 5) 5))) := by
  intro mm
  have hm : mm = _ :=
    metaFrag_mem ptr dataPtr pos32 seqLen64 e m n S hptr hdp hpo hseq he hS hnc
  rw [hm]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp [StoreMap.get?, META_STAGE_OFF]

/-- …and the byte offsets it stores to are the four slots the softmax kernel
    reads, by `SEQ_SLOT`/`CHUNKS_SLOT`/`TAIL_SLOT`/`REM_SLOT`'s own definitions
    — four `Nat` words at four-byte stride. -/
theorem metaFrag_slots :
    (8, 12, 16, 20)
      = (Qwen2Proven.SEQ_SLOT * 4, Qwen2Proven.CHUNKS_SLOT * 4,
         Qwen2Proven.TAIL_SLOT * 4, Qwen2Proven.REM_SLOT * 4) := by decide

/-- inferFn (fn_27): one decode step.
    Reads [token_id:u32][pos:u32] from data_ptr.
    Uploads meta to GPU, launches embed lookup, runs 24-layer loop (calls fn_28),
    then calls fn_31 for final rms+lm_head+argmax. -/
def inferFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  -- Declare colocated callees
  let fnLayerStep ← declareColocatedFFI "fn_28" [.i64] none
  let fnFinalStep ← declareColocatedFFI "fn_31" [.i64] none

  let dataPtr ← load64 (← absAddr ptr 0x18)

  -- Read token_id and pos from host data
  let pos32   ← load32 (← iaddImm dataPtr 4)
  let pos64   ← uextend64 pos32
  let seqLen64 ← iaddImm pos64 1

  -- Store pos and seqLen in shared memory for use by layer/final functions
  storeI64 pos64    (← absAddr ptr POS_SLOT_OFF)
  storeI64 seqLen64 (← absAddr ptr SEQ_LEN_SLOT_OFF)

  let stage ← metaStageFrag ptr dataPtr pos32 seqLen64
  let metaT ← slotMeta.load ptr
  let metaBytes24 ← iconst64 24
  Tensor.upload cuda ptr metaT stage metaBytes24

  -- Embedding lookup: bind = [embed_table, meta_buf, hidden_out]
  let embedT  ← slotEmbed.load ptr
  let hiddenT ← slotHidden.load ptr
  launchEmbed cuda ptr BIND_EMBED embedT metaT hiddenT

  -- 24-layer loop: calls inferLayerFn(fn_28) with layer index in LAYER_IDX_OFF
  let nLayers ← iconst64 N_LAYERS
  forLoop .i64 nLayers fun layerIdx => do
    storeI64 layerIdx (← absAddr ptr LAYER_IDX_OFF)
    callVoid fnLayerStep [ptr]
  callVoid fnFinalStep [ptr]
  ret


-- ── Attention helper types and sub-builders ───────────────────────────────────

private structure AttnBufs where
  bufRmsAttn : VecD
  bufWq      : MatDD
  bufBq      : VecD
  bufWk      : MatKVD
  bufBk      : VecKV
  bufWv      : MatKVD
  bufBv      : VecKV
  bufWo      : MatDD
  bufKCache  : KVCache
  bufVCache  : KVCache
  bufHidden  : VecD
  bufHdNorm  : VecD
  bufQ       : VecD
  bufKCur    : VecKV
  bufVCur    : VecKV
  bufAttnOut : VecD
  bufScores  : VecScores
  bufProbs   : VecScores
  bufMeta    : VecMeta

private structure AttnConsts where
  one32     : Val
  two32     : Val
  three32   : Val
  blk256    : Val
  nq32      : Val
  nkv32     : Val
  dm32      : Val
  kv32      : Val
  hdim32    : Val
  blk32_2   : Val
  hdim64    : Val
  maxSeq64  : Val
  alpha     : Val
  attnAlpha : Val
  zero32    : Val

private def load32At (base : Val) (off : Nat) : IRBuilder Val :=
  load32 =<< iaddImm base off

private def load64At (base : Val) (off : Nat) : IRBuilder Val :=
  load64 =<< iaddImm base off

private def attnLoadBufs (ptr slotBaseA : Val) : IRBuilder AttnBufs := do
  let bufRmsAttn ← LayerSlot.rmsAttn.load slotBaseA
  let bufWq      ← LayerSlot.wq.load      slotBaseA
  let bufBq      ← LayerSlot.bq.load      slotBaseA
  let bufWk      ← LayerSlot.wk.load      slotBaseA
  let bufBk      ← LayerSlot.bk.load      slotBaseA
  let bufWv      ← LayerSlot.wv.load      slotBaseA
  let bufBv      ← LayerSlot.bv.load      slotBaseA
  let bufWo      ← LayerSlot.wo.load      slotBaseA
  let bufKCache  ← LayerSlot.kCache.load  slotBaseA
  let bufVCache  ← LayerSlot.vCache.load  slotBaseA
  let bufHidden  ← slotHidden.load  ptr
  let bufHdNorm  ← slotHdNorm.load  ptr
  let bufQ       ← slotQ.load       ptr
  let bufKCur    ← slotKCur.load    ptr
  let bufVCur    ← slotVCur.load    ptr
  let bufAttnOut ← slotAttnOut.load ptr
  let bufScores  ← slotScores.load  ptr
  let bufProbs   ← slotProbs.load   ptr
  let bufMeta    ← slotMeta.load    ptr
  return { bufRmsAttn, bufWq, bufBq, bufWk, bufBk, bufWv, bufBv, bufWo,
           bufKCache, bufVCache, bufHidden, bufHdNorm, bufQ, bufKCur, bufVCur,
           bufAttnOut, bufScores, bufProbs, bufMeta }

private def mkAttnConsts : IRBuilder AttnConsts := do
  let one32 ← iconst32 1;    let two32 ← iconst32 2;    let three32 ← iconst32 3
  let blk256 ← iconst32 256; let nq32 ← iconst32 N_Q;   let nkv32 ← iconst32 N_KV
  let dm32 ← iconst32 D;     let kv32 ← iconst32 KV_DIM; let hdim32 ← iconst32 HEAD_DIM
  let blk32_2 ← iconst32 32; let hdim64 ← iconst64 HEAD_DIM
  let maxSeq64 ← iconst64 (MAX_SEQ * HEAD_DIM)
  let alpha ← iconst32 0x3F800000; let attnAlpha ← iconst32 0x3E000000
  let zero32 ← iconst32 0
  return { one32, two32, three32, blk256, nq32, nkv32, dm32, kv32, hdim32, blk32_2,
           hdim64, maxSeq64, alpha, attnAlpha, zero32 }

-- RMSNorm → QKV projections → bias adds
private def attnProjPhase (ptr : Val) (cuda : CudaSetup) (blas : CuBlasSetup)
    (b : AttnBufs) : IRBuilder Unit := do
  launchRms cuda ptr BIND_RMS1 b.bufHidden b.bufRmsAttn b.bufHdNorm
  -- Q/K/V projections: shape-typed.  Wq:[D,D]·hidden:[D] → q:[D];
  -- Wk:[KV_DIM,D]·hidden:[D] → kCur:[KV_DIM]; same for Wv.
  CuBlas.linear blas ptr b.bufWq b.bufHdNorm b.bufQ
  CuBlas.linear blas ptr b.bufWk b.bufHdNorm b.bufKCur
  CuBlas.linear blas ptr b.bufWv b.bufHdNorm b.bufVCur
  launchBiasD  cuda ptr BIND_BIAS_Q b.bufQ    b.bufBq
  launchBiasKV cuda ptr BIND_BIAS_K b.bufKCur b.bufBk
  launchBiasKV cuda ptr BIND_BIAS_V b.bufVCur b.bufBv

-- RoPE → KV store
private def attnRopePhase (ptr : Val) (cuda : CudaSetup) (b : AttnBufs)
    (_c : AttnConsts) : IRBuilder Unit := do
  let bufRopeTable ← slotRopeTable.load ptr
  launchRopeQ  cuda ptr BIND_ROPE_Q b.bufQ    b.bufMeta bufRopeTable
  launchRopeK  cuda ptr BIND_ROPE_K b.bufKCur b.bufMeta bufRopeTable
  launchKVStore cuda ptr BIND_KV_K  b.bufKCur b.bufKCache b.bufMeta
  launchKVStore cuda ptr BIND_KV_V  b.bufVCur b.bufVCache b.bufMeta

-- Attention scores → softmax → V-mix → Wo → residual
private def attnMixPhase (ptr : Val) (cuda : CudaSetup) (blas : CuBlasSetup)
    (b : AttnBufs) (c : AttnConsts) : IRBuilder Unit := do
  let seqLen64 ← load64At ptr SEQ_LEN_SLOT_OFF
  let seqLen32 ← ireduce32 seqLen64
  -- Q, AttnOut, scores, probs are flat memory; view them as GQA-grouped
  -- [N_KV, GQA_RATIO, ...] for the batched-by-KV-head GEMMs.  K/V cache is
  -- now stored once per KV head and broadcast across the gqaRatio Q heads.
  let qGqa      : Tensor [.sta N_KV, .sta GQA_RATIO, .sta HEAD_DIM] := b.bufQ.reshape
  let outGqa    : Tensor [.sta N_KV, .sta GQA_RATIO, .sta HEAD_DIM] := b.bufAttnOut.reshape
  let scoresGqa : Tensor [.sta N_KV, .sta GQA_RATIO, .dyn]          := b.bufScores.reshape
  let probsGqa  : Tensor [.sta N_KV, .sta GQA_RATIO, .dyn]          := b.bufProbs.reshape
  -- scores[kv, i, :seqLen] = attnAlpha * K[kv, :seqLen, :] @ Q[kv, i]
  CuBlas.attnScoresQK blas ptr c.attnAlpha seqLen32 seqLen64 b.bufKCache qGqa scoresGqa
  launchSoftmax cuda ptr BIND_SOFTMAX b.bufScores b.bufMeta b.bufProbs
  -- attnOut[kv, i] = V[kv, :seqLen, :]^T @ probs[kv, i, :seqLen]
  CuBlas.attnMixV blas ptr c.alpha seqLen32 seqLen64 b.bufVCache probsGqa outGqa
  -- O projection: Wo:[D,D]·attnOut:[D] → hdNorm:[D]
  CuBlas.linear blas ptr b.bufWo b.bufAttnOut b.bufHdNorm
  launchResidualAdd cuda ptr BIND_ADD1 b.bufHidden b.bufHdNorm

/-- Attention sub-layer body (RMSNorm → Q/K/V proj → biases → RoPE → KV store
    → GQA attention → Wo → residual).  Caller computes `slotBaseA` (the per-
    layer slot containing weight + K/V cache buffer IDs).  Emits `ret`. -/
def attnBody (cuda : CudaSetup) (blas : CuBlasSetup)
    (ptr slotBaseA : Val) : IRBuilder Unit := do
  let b ← attnLoadBufs ptr slotBaseA
  let c ← mkAttnConsts
  attnProjPhase ptr cuda blas b
  attnRopePhase ptr cuda b c
  attnMixPhase  ptr cuda blas b c
  ret

/-- FFN sub-layer body (RMSNorm → Wg/Wu → SiLU-gate → Wd → residual).  Caller
    computes `slotBaseA`.  Emits `ret`. -/
def ffnBody (cuda : CudaSetup) (blas : CuBlasSetup)
    (ptr slotBaseA : Val) : IRBuilder Unit := do
  let bufRmsFfn  ← LayerSlot.rmsFfn.load slotBaseA
  let bufWg      ← LayerSlot.wg.load     slotBaseA
  let bufWu      ← LayerSlot.wu.load     slotBaseA
  let bufWd      ← LayerSlot.wd.load     slotBaseA
  let bufHidden  ← slotHidden.load  ptr
  let bufHdNorm  ← slotHdNorm.load  ptr
  let bufFfGate  ← slotFfGate.load  ptr
  let bufFfUp    ← slotFfUp.load    ptr
  let bufFfAct   ← slotFfAct.load   ptr
  let bufAttnOut ← slotAttnOut.load ptr
  launchRms cuda ptr BIND_RMS2 bufHidden bufRmsFfn bufHdNorm
  -- Wg/Wu projections: Wg:[D_FF,D]·hdNorm:[D] → ffGate:[D_FF]; same for Wu.
  CuBlas.linear blas ptr bufWg bufHdNorm bufFfGate
  CuBlas.linear blas ptr bufWu bufHdNorm bufFfUp
  launchSiluGate cuda ptr BIND_SILU bufFfGate bufFfUp bufFfAct
  -- Wd down projection: Wd:[D,D_FF]·ffAct:[D_FF] → attnOut:[D] (reused as temp)
  CuBlas.linear blas ptr bufWd bufFfAct bufAttnOut
  launchResidualAdd cuda ptr BIND_ADD2 bufHidden bufAttnOut
  ret

/-- inferFinalFn (fn_31): final RMSNorm → lm_head → argmax → sync → download next_token. -/
def inferFinalFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let blas    ← declareCuBlasFFI
  let outPtr  ← load64At ptr 0x28
  let bufHidden   ← slotHidden.load   ptr
  let bufHdNorm   ← slotHdNorm.load   ptr
  let bufRmsFinal ← slotRmsFinal.load ptr
  let bufLmHead   ← slotLmHead.load   ptr
  let bufLogits   ← slotLogits.load   ptr
  let bufMeta     ← slotMeta.load     ptr
  let meta64  ← iconst64 24
  launchRms cuda ptr BIND_RMS2 bufHidden bufRmsFinal bufHdNorm
  -- LM head projection: lmHead:[VOCAB,D]·hdNorm:[D] → logits:[VOCAB]
  CuBlas.linear blas ptr bufLmHead bufHdNorm bufLogits
  launchArgmax cuda ptr BIND_ARGMAX bufLogits bufMeta
  let _ ← cudaSync cuda ptr 0x10
  -- The meta buffer is six words now (the softmax loop bounds ride along), so
  -- it lands in the staging area and only [token_id, pos] goes to the caller.
  let stage ← absAddr ptr META_STAGE_OFF
  Tensor.download cuda ptr bufMeta stage meta64
  -- The proven argmax writes the token id as an exactly-representable float;
  -- convert it in place before handing [token_id, pos] back to the caller.
  let tokF   ← loadF32 stage
  let tok32  ← fcvtToUint .i32 tokF
  storeI32 tok32 stage
  storeI64 (← load64 stage) outPtr
  ret

-- ── Tokenizer functions ───────────────────────────────────────────────────────

/-- loadTokenizerFn (fn_32): slurp tokenizer binary into a pinned host buffer, init HT,
    populate merge table.
    Tokenizer binary layout:
      [0]  n_merges: u32
      [4]  vocab_size: u32
      [8]  byte_pool_size: u32
      [12] padding: u32
      [16] byte_init[256]: u32  (byte_value → initial token id)
      [1040] merges[n_merges]: (tok_a:u32, tok_b:u32, result:u32) × n_merges
      [1040+n_merges*12] decode_offsets[vocab_size]: u32
      [1040+n_merges*12+vocab_size*4] decode_lens[vocab_size]: u32
      [1040+n_merges*12+vocab_size*8] byte_pool -/
def loadTokenizerFn : IRBuilder Unit := do
  let ptr      ← entryBlock
  let cuda     ← declareCudaFFI
  let fnFileRead ← declareFFI "cl_file_read_to_ptr" [.i64, .i64, .i64, .i64] (some .i64)
  let ht       ← declareHtFFI
  let ctxPtr   ← load64 (← absAddr ptr 0x10)
  let dataPtr  ← load64 (← absAddr ptr TOKENIZER_PATH_PTR_OFF)
  -- Allocate a pinned host buffer and slurp the tokenizer file into it.
  let tokBytes64 ← iconst64 TOK_FILE_MAX_BYTES
  let tokPinId   ← call cuda.fnPinnedAlloc [ctxPtr, tokBytes64]
  let tokBufPtr  ← call cuda.fnPinnedPtr   [ctxPtr, tokPinId]
  let zero64     ← iconst64 0
  let _ ← call fnFileRead [dataPtr, tokBufPtr, zero64, tokBytes64]
  storeI64 tokBufPtr (← absAddr ptr TOK_BUF_PTR_OFF)
  -- Init HT context (writes context ptr to ptr[0x00])
  htInit ptr
  let htCtx    ← load64At ptr 0x00
  let _        ← call ht.fnCreate [htCtx]
  -- Read n_merges from header
  let nMerges  ← uload32_64 (← iaddImm tokBufPtr 0)
  -- Merge base: offset 1040 in the binary (16 byte header + 256×4 byte_init)
  let mergeBase ← iaddImm tokBufPtr 1040
  let keyAddr  ← iaddImm ptr HT_KEY_OFF
  let valAddr  ← iaddImm ptr HT_VAL_OFF
  let keyLen8  ← iconst32 8
  let valLen8  ← iconst32 8
  let twelve64 ← iconst64 12
  -- For i in 0..n_merges, insert (tok_a, tok_b) → (rank, result) into HT
  forLoop .i64 nMerges fun i => do
    let mergeOff ← imul i twelve64
    let mergePtr ← iadd mergeBase mergeOff
    let tok_a    ← load32 (← iaddImm mergePtr 0)
    let tok_b    ← load32 (← iaddImm mergePtr 4)
    let result   ← load32 (← iaddImm mergePtr 8)
    let rank32   ← ireduce32 i
    storeI32 tok_a   keyAddr
    storeI32 tok_b   (← iaddImm keyAddr 4)
    storeI32 rank32  valAddr
    storeI32 result  (← iaddImm valAddr 4)
    callVoid ht.fnInsert [htCtx, keyAddr, keyLen8, valAddr, valLen8]
  ret

/-- tokenizeInitFn (fn_33): convert each byte of text (TEXT_IN_OFF, TEXT_LEN_OFF) to its
    initial token id using the byte_init table; store results in TOKEN_BUF_OFF.
    Sets TOKEN_COUNT_OFF = text length (before BPE). -/
def tokenizeInitFn : IRBuilder Unit := do
  let ptr       ← entryBlock
  let tokMmap   ← load64At ptr TOK_BUF_PTR_OFF
  let textLen   ← load64At ptr TEXT_LEN_OFF
  let byteInit  ← iaddImm tokMmap 16
  let textBase  ← iaddImm ptr TEXT_IN_OFF
  let tokBuf    ← iaddImm ptr TOKEN_BUF_OFF
  forLoop .i64 textLen fun i => do
    let byt     ← uload8_64 (← iadd textBase i)
    let byteOff ← ishlImm byt 2
    let initTok ← load32 (← iadd byteInit byteOff)
    let tokOff  ← ishlImm i 2
    storeI32 initTok (← iadd tokBuf tokOff)
  storeI64 textLen (← absAddr ptr TOKEN_COUNT_OFF)
  ret

/-- tokenizeBpeFn (fn_34): run BPE merge passes over TOKEN_BUF_OFF until no more merges apply.
    Uses the HT (populated by loadTokenizerFn) for O(1) pair lookups.
    Updates TOKEN_COUNT_OFF to the final token count. -/
def tokenizeBpeFn : IRBuilder Unit := do
  let ptr        ← entryBlock
  let ht         ← declareHtFFI
  let htCtx      ← load64At ptr 0x00
  let tokBuf     ← iaddImm ptr TOKEN_BUF_OFF
  let keyAddr    ← iaddImm ptr HT_KEY_OFF
  let valAddr    ← iaddImm ptr HT_VAL_OFF
  let keyLen8    ← iconst32 8
  let tokCount   ← load64At ptr TOKEN_COUNT_OFF
  let zero64     ← iconst64 0
  let one64      ← iconst64 1
  let maxRank    ← iconst32 (-1)  -- 0xFFFFFFFF: "no best found yet"
  let negOne64   ← iconst64 (-1)  -- sentinel "no best pos"
  let zero32     ← iconst32 0
  -- blocks
  let bpeCheck    ← declareBlock [.i64]
  let bpeScanHdr  ← declareBlock [.i64, .i64, .i32, .i64]
  let bpeScanBody ← declareBlock [.i64, .i64, .i32, .i64]
  let bpeScanFnd  ← declareBlock [.i64, .i64, .i32, .i64]
  let bpeScanNext ← declareBlock [.i64, .i64, .i32, .i64]
  let bpeApply    ← declareBlock [.i64, .i64]
  let bpeDoApply  ← declareBlock [.i64, .i64]
  let shiftHdr    ← declareBlock [.i64, .i64, .i64]
  let shiftBody   ← declareBlock [.i64, .i64, .i64]
  let bpeDone     ← declareBlock [.i64]
  jump bpeCheck.ref [tokCount]
  -- bpeCheck: if n_toks <= 1, done
  startBlock bpeCheck
  let n_toks := bpeCheck.param 0
  let small ← icmp .ule n_toks one64
  brif small bpeDone.ref [n_toks] bpeScanHdr.ref [n_toks, zero64, maxRank, negOne64]
  -- bpeScanHdr: scan adjacent pairs for the lowest-rank merge
  startBlock bpeScanHdr
  let sn   := bpeScanHdr.param 0
  let si   := bpeScanHdr.param 1
  let sr   := bpeScanHdr.param 2
  let sp   := bpeScanHdr.param 3
  let n1   ← iaddImm sn (-1)
  let done ← icmp .uge si n1
  brif done bpeApply.ref [sn, sp] bpeScanBody.ref [sn, si, sr, sp]
  -- bpeScanBody: look up pair (tokens[i], tokens[i+1]) in HT
  startBlock bpeScanBody
  let bn   := bpeScanBody.param 0
  let bi   := bpeScanBody.param 1
  let br   := bpeScanBody.param 2
  let bp   := bpeScanBody.param 3
  let iOff ← ishlImm bi 2
  let tokA ← load32 (← iadd tokBuf iOff)
  let tokB ← load32 (← iadd tokBuf (← iaddImm iOff 4))
  storeI32 tokA keyAddr
  storeI32 tokB (← iaddImm keyAddr 4)
  let found ← call ht.fnLookup [htCtx, keyAddr, keyLen8, valAddr]
  let notFound ← icmp .slt found zero32
  brif notFound bpeScanNext.ref [bn, bi, br, bp] bpeScanFnd.ref [bn, bi, br, bp]
  -- bpeScanFnd: pair found — check if its rank is better than current best
  startBlock bpeScanFnd
  let fn_  := bpeScanFnd.param 0
  let fi   := bpeScanFnd.param 1
  let fr   := bpeScanFnd.param 2
  let fp   := bpeScanFnd.param 3
  let rank ← load32 valAddr
  let better ← icmp .ult rank fr
  brif better bpeScanNext.ref [fn_, fi, rank, fi] bpeScanNext.ref [fn_, fi, fr, fp]
  -- bpeScanNext: advance i
  startBlock bpeScanNext
  let nn   := bpeScanNext.param 0
  let ni   := bpeScanNext.param 1
  let nr   := bpeScanNext.param 2
  let np   := bpeScanNext.param 3
  jump bpeScanHdr.ref [nn, ← iaddImm ni 1, nr, np]
  -- bpeApply: check if any merge was found
  startBlock bpeApply
  let an   := bpeApply.param 0
  let ap   := bpeApply.param 1
  let noMerge ← icmp .eq ap negOne64
  brif noMerge bpeDone.ref [an] bpeDoApply.ref [an, ap]
  -- bpeDoApply: apply merge at best_pos — re-lookup to get result token, then shift
  startBlock bpeDoApply
  let dn   := bpeDoApply.param 0
  let dp   := bpeDoApply.param 1
  let dOff ← ishlImm dp 2
  let dA   ← load32 (← iadd tokBuf dOff)
  let dB   ← load32 (← iadd tokBuf (← iaddImm dOff 4))
  storeI32 dA keyAddr
  storeI32 dB (← iaddImm keyAddr 4)
  let _ ← call ht.fnLookup [htCtx, keyAddr, keyLen8, valAddr]
  let resT ← load32 (← iaddImm valAddr 4)
  storeI32 resT (← iadd tokBuf dOff)
  jump shiftHdr.ref [dn, dp, ← iaddImm dp 1]
  -- shiftHdr: shift tokens left by one starting from j = best_pos+1
  startBlock shiftHdr
  let shn  := shiftHdr.param 0
  let shp  := shiftHdr.param 1
  let shj  := shiftHdr.param 2
  let shn1 ← iaddImm shn (-1)
  let shDone ← icmp .uge shj shn1
  brif shDone bpeCheck.ref [← iaddImm shn (-1)] shiftBody.ref [shn, shp, shj]
  -- shiftBody: tokens[j] = tokens[j+1]
  startBlock shiftBody
  let sbn  := shiftBody.param 0
  let sbp  := shiftBody.param 1
  let sbj  := shiftBody.param 2
  let sbOff  ← ishlImm sbj 2
  let nextT  ← load32 (← iadd tokBuf (← iaddImm sbOff 4))
  storeI32 nextT (← iadd tokBuf sbOff)
  jump shiftHdr.ref [sbn, sbp, ← iaddImm sbj 1]
  -- bpeDone
  startBlock bpeDone
  let finalN := bpeDone.param 0
  storeI64 finalN (← absAddr ptr TOKEN_COUNT_OFF)
  ret

/-- detokenizeFn (fn_35): convert token IDs in TOKEN_BUF_OFF (count = TOKEN_COUNT_OFF) to bytes
    in TEXT_OUT_OFF; stores output byte count in TEXT_LEN_OFF. -/
def detokenizeFn : IRBuilder Unit := do
  let ptr       ← entryBlock
  let tokMmap   ← load64At ptr TOK_BUF_PTR_OFF
  -- Compute table pointers from binary header
  let nMerges   ← uload32_64 (← iaddImm tokMmap 0)
  let vocabSize ← uload32_64 (← iaddImm tokMmap 4)
  let twelve64  ← iconst64 12
  let four64    ← iconst64 4
  let mergeBytes ← imul nMerges twelve64
  let decOffBase ← iaddImm tokMmap 1040
  let decOffPtr  ← iadd decOffBase mergeBytes
  let vocBytes   ← imul vocabSize four64
  let decLenPtr  ← iadd decOffPtr vocBytes
  let bytePool   ← iadd decLenPtr vocBytes
  let tokBuf    ← iaddImm ptr TOKEN_BUF_OFF
  let n_toks    ← load64At ptr TOKEN_COUNT_OFF
  let textOut   ← iaddImm ptr TEXT_OUT_OFF
  let zero64    ← iconst64 0
  -- Outer: counter `ti` over tokens; accumulator `tp` = output byte offset.
  -- Inner: copy `decLen` bytes from srcPtr[..] to textOut[tp..].
  let finalTp ← forLoopAcc .i64 .i64 n_toks zero64 fun ti tp => do
    let tok_id ← uload32_64 (← iadd tokBuf (← ishlImm ti 2))
    let decOff ← uload32_64 (← iadd decOffPtr (← ishlImm tok_id 2))
    let decLen ← uload32_64 (← iadd decLenPtr (← ishlImm tok_id 2))
    let srcPtr ← iadd bytePool decOff
    forLoop .i64 decLen fun i => do
      let byt ← uload8_64 (← iadd srcPtr i)
      istore8 byt (← iadd textOut (← iadd tp i))
    iadd tp decLen
  storeI64 finalTp (← absAddr ptr TEXT_LEN_OFF)
  ret

/-- cliFn (fn_36): stdin/stdout chat loop.
    Per line: read stdin → tokenize → prefill+decode via fn_27 → detokenize → write stdout.
    Exits when stdin closes (EOF). -/
def cliFn : IRBuilder Unit := do
  let ptr         ← entryBlock
  -- Colocated callees
  let fnInfer     ← declareColocatedFFI "fn_27" [.i64] none
  let fnTokInit   ← declareColocatedFFI "fn_33" [.i64] none
  let fnTokBpe    ← declareColocatedFFI "fn_34" [.i64] none
  let fnDetok     ← declareColocatedFFI "fn_35" [.i64] none
  -- Stdin/stdout FFI
  let fnStdinRead   ← declareFFI "cl_stdin_readline" [.i64, .i64, .i64] (some .i64)
  let fnStdoutWrite ← declareFFI "cl_stdout_write"   [.i64, .i64, .i64] (some .i64)
  -- Redirect inferFn's data_ptr and out_ptr to our step buffers
  let inferInAddr ← absAddr ptr INFER_IN_OFF
  let inferOutAddr ← absAddr ptr INFER_OUT_OFF
  storeI64 inferInAddr  (← absAddr ptr 0x18)
  storeI64 inferOutAddr (← absAddr ptr 0x28)
  -- Constants
  let zero64    ← iconst64 0
  let maxRecv   ← iconst64 MAX_RECV
  let maxDecode ← iconst64 MAX_DECODE
  let textInOff64  ← iconst64 TEXT_IN_OFF
  let textOutOff64 ← iconst64 TEXT_OUT_OFF
  let eosTok    ← iconst32 151643   -- <|endoftext|>
  let imEndTok  ← iconst32 151645   -- <|im_end|>
  let nlTok     ← iconst32 198      -- newline
  let imStartTok ← iconst32 151644  -- <|im_start|>
  let tokBuf    ← iaddImm ptr TOKEN_BUF_OFF
  let textInPtr ← iaddImm ptr TEXT_IN_OFF
  let textOutPtr ← iaddImm ptr TEXT_OUT_OFF
  -- Qwen chat wrapper token IDs:
  -- <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
  let t_user_u  ← iconst32 84
  let t_user_s  ← iconst32 82
  let t_user_e  ← iconst32 68
  let t_user_r  ← iconst32 81
  let t_as_a1   ← iconst32 64
  let t_as_s1   ← iconst32 82
  let t_as_s2   ← iconst32 82
  let t_as_i    ← iconst32 72
  let t_as_s3   ← iconst32 82
  let t_as_t    ← iconst32 83
  let t_as_a2   ← iconst32 64
  let t_as_n    ← iconst32 77
  let t_as_t2   ← iconst32 83
  let prefixLen ← iconst64 PREFIX_LEN
  let wrapLen   ← iconst64 WRAP_LEN
  -- Blocks
  let bootHdr      ← declareBlock [.i64]               -- (i)  system-prompt prefill loop
  let bootBody     ← declareBlock [.i64]
  let cliLoop      ← declareBlock []
  let exitBlock    ← declareBlock []
  let trimLfChk    ← declareBlock [.i64]
  let trimLfYes    ← declareBlock [.i64]
  let trimCrChk    ← declareBlock [.i64]
  let trimCrYes    ← declareBlock [.i64]
  let trimCrDone   ← declareBlock [.i64]
  let trimCrKeep   ← declareBlock [.i64]
  let trimDone     ← declareBlock [.i64]
  let wrapShiftHdr ← declareBlock [.i64]
  let wrapShiftBody ← declareBlock [.i64]
  let wrapWrite    ← declareBlock []
  let prefHdr      ← declareBlock [.i64]
  let prefBody     ← declareBlock [.i64]
  let decInit      ← declareBlock []
  let decHdr       ← declareBlock [.i64, .i32, .i64]   -- (pos, tok, n_out)
  let decBody      ← declareBlock [.i64, .i32, .i64]
  let writeResp    ← declareBlock [.i64]               -- (n_out)
  -- First, prefill the static system prompt into the KV cache once.
  jump bootHdr.ref [zero64]
  startBlock exitBlock
  ret
  startBlock bootHdr
  let bI := bootHdr.param 0
  let bDone ← icmp .uge bI (← iconst64 SYSTEM_TOKEN_COUNT)
  brif bDone cliLoop.ref [] bootBody.ref [bI]
  startBlock bootBody
  let bbI := bootBody.param 0
  let bbOff ← ishlImm bbI 2
  let bbAddr ← iadd (← iaddImm ptr SYSTEM_TOKENS_OFF) bbOff
  let bbTok  ← load32 bbAddr
  storeI32 bbTok               (← absAddr ptr INFER_IN_OFF)
  storeI32 (← ireduce32 bbI)   (← absAddr ptr (INFER_IN_OFF + 4))
  callVoid fnInfer [ptr]
  jump bootHdr.ref [(← iaddImm bbI 1)]
  startBlock cliLoop
  -- Seed the running position past the system prompt on first entry, then re-read
  -- it from shared memory on subsequent iterations (writeResp updates it per turn).
  storeI64 (← iconst64 SYSTEM_TOKEN_COUNT) (← absAddr ptr RUNNING_POS_OFF)
  let runningPos ← load64 (← absAddr ptr RUNNING_POS_OFF)
  let nRecv  ← call fnStdinRead [ptr, textInOff64, maxRecv]
  let hasInput ← icmp .ugt nRecv zero64
  brif hasInput trimLfChk.ref [nRecv] exitBlock.ref []
  startBlock trimLfChk
  let tlLen0 := trimLfChk.param 0
  let tlLast ← uload8_64 (← iadd textInPtr (← iaddImm tlLen0 (-1)))
  let isLf   ← icmp .eq tlLast (← iconst64 10)
  brif isLf trimLfYes.ref [tlLen0] trimCrChk.ref [tlLen0]
  startBlock trimLfYes
  let tlyLen0 := trimLfYes.param 0
  jump trimCrChk.ref [(← iaddImm tlyLen0 (-1))]
  startBlock trimCrChk
  let tcLen1 := trimCrChk.param 0
  let hasRemain ← icmp .ugt tcLen1 zero64
  brif hasRemain trimCrYes.ref [tcLen1] trimDone.ref [tcLen1]
  startBlock trimCrYes
  let tcyLen1 := trimCrYes.param 0
  let tcyLast ← uload8_64 (← iadd textInPtr (← iaddImm tcyLen1 (-1)))
  let isCr   ← icmp .eq tcyLast (← iconst64 13)
  brif isCr trimCrDone.ref [tcyLen1] trimCrKeep.ref [tcyLen1]
  startBlock trimCrDone
  jump trimDone.ref [(← iaddImm (trimCrDone.param 0) (-1))]
  startBlock trimCrKeep
  jump trimDone.ref [trimCrKeep.param 0]
  startBlock trimDone
  let tsLen := trimDone.param 0
  storeI64 tsLen (← absAddr ptr TEXT_LEN_OFF)
  callVoid fnTokInit [ptr]
  callVoid fnTokBpe  [ptr]
  let rawPromptN ← load64At ptr TOKEN_COUNT_OFF
  jump wrapShiftHdr.ref [rawPromptN]
  startBlock wrapShiftHdr
  let wsI := wrapShiftHdr.param 0
  let wsDone ← icmp .eq wsI zero64
  brif wsDone wrapWrite.ref [] wrapShiftBody.ref [wsI]
  startBlock wrapShiftBody
  let wbI := wrapShiftBody.param 0
  let srcIdx ← iaddImm wbI (-1)
  let srcOff ← ishlImm srcIdx 2
  let tok    ← load32 (← iadd tokBuf srcOff)
  let dstIdx ← iadd srcIdx prefixLen
  let dstOff ← ishlImm dstIdx 2
  storeI32 tok (← iadd tokBuf dstOff)
  jump wrapShiftHdr.ref [srcIdx]
  startBlock wrapWrite
  -- Prefix: <|im_start|>user\n
  storeI32 imStartTok (← iadd tokBuf (← iconst64 0))
  storeI32 t_user_u   (← iadd tokBuf (← iconst64 4))
  storeI32 t_user_s   (← iadd tokBuf (← iconst64 8))
  storeI32 t_user_e   (← iadd tokBuf (← iconst64 12))
  storeI32 t_user_r   (← iadd tokBuf (← iconst64 16))
  storeI32 nlTok      (← iadd tokBuf (← iconst64 20))
  let suffixBaseIdx ← iadd rawPromptN prefixLen
  let suffixBaseOff ← ishlImm suffixBaseIdx 2
  -- Suffix: <|im_end|>\n<|im_start|>assistant\n
  storeI32 imEndTok   (← iadd tokBuf suffixBaseOff)
  storeI32 nlTok      (← iadd tokBuf (← iaddImm suffixBaseOff 4))
  storeI32 imStartTok (← iadd tokBuf (← iaddImm suffixBaseOff 8))
  storeI32 t_as_a1    (← iadd tokBuf (← iaddImm suffixBaseOff 12))
  storeI32 t_as_s1    (← iadd tokBuf (← iaddImm suffixBaseOff 16))
  storeI32 t_as_s2    (← iadd tokBuf (← iaddImm suffixBaseOff 20))
  storeI32 t_as_i     (← iadd tokBuf (← iaddImm suffixBaseOff 24))
  storeI32 t_as_s3    (← iadd tokBuf (← iaddImm suffixBaseOff 28))
  storeI32 t_as_t     (← iadd tokBuf (← iaddImm suffixBaseOff 32))
  storeI32 t_as_a2    (← iadd tokBuf (← iaddImm suffixBaseOff 36))
  storeI32 t_as_n     (← iadd tokBuf (← iaddImm suffixBaseOff 40))
  storeI32 t_as_t2    (← iadd tokBuf (← iaddImm suffixBaseOff 44))
  storeI32 nlTok      (← iadd tokBuf (← iaddImm suffixBaseOff 48))
  let nPrompt ← iadd rawPromptN wrapLen
  storeI64 nPrompt (← absAddr ptr N_PROMPT_OFF)
  -- Reset output token count for decode phase
  storeI64 zero64 (← absAddr ptr TOKEN_COUNT_OFF)
  jump prefHdr.ref [zero64]
  -- Prefill: feed each prompt token through inferFn
  startBlock prefHdr
  let phI := prefHdr.param 0
  let phDone ← icmp .uge phI nPrompt
  brif phDone decInit.ref [] prefBody.ref [phI]
  startBlock prefBody
  let pbI := prefBody.param 0
  let pbTok ← load32 (← iadd tokBuf (← ishlImm pbI 2))
  let pbAbsPos ← iadd runningPos pbI
  storeI32 pbTok                  (← absAddr ptr INFER_IN_OFF)
  storeI32 (← ireduce32 pbAbsPos) (← absAddr ptr (INFER_IN_OFF + 4))
  callVoid fnInfer [ptr]
  jump prefHdr.ref [(← iaddImm pbI 1)]
  startBlock decInit
  let diTok ← load32 (← absAddr ptr INFER_OUT_OFF)
  let diStartPos ← iadd runningPos nPrompt
  jump decHdr.ref [diStartPos, diTok, zero64]
  -- Decode loop: generate new tokens until EOS or budget exhausted
  startBlock decHdr
  let dhPos  := decHdr.param 0
  let dhTok  := decHdr.param 1
  let dhNOut := decHdr.param 2
  -- Stop only on the real end-of-turn tokens or hitting the decode budget.
  -- Plain newlines occur naturally inside multi-line responses (lists, code).
  let isEos   ← icmp .eq dhTok eosTok
  let isImEnd ← icmp .eq dhTok imEndTok
  let stopTok ← bor isEos isImEnd
  let isFull  ← icmp .uge dhNOut maxDecode
  let dhStop  ← bor stopTok isFull
  brif dhStop writeResp.ref [dhNOut] decBody.ref [dhPos, dhTok, dhNOut]
  startBlock decBody
  let dbPos  := decBody.param 0
  let dbTok  := decBody.param 1
  let dbNOut := decBody.param 2
  storeI32 dbTok (← iadd tokBuf (← ishlImm dbNOut 2))
  storeI32 dbTok               (← absAddr ptr INFER_IN_OFF)
  storeI32 (← ireduce32 dbPos) (← absAddr ptr (INFER_IN_OFF + 4))
  callVoid fnInfer [ptr]
  let nextTok ← load32 (← absAddr ptr INFER_OUT_OFF)
  jump decHdr.ref [(← iaddImm dbPos 1), nextTok, (← iaddImm dbNOut 1)]
  -- Write response: detokenize output tokens to text_out, append newline, write to stdout.
  -- Then feed <|im_end|>\n through inferFn so the KV cache reflects a properly
  -- closed assistant turn; advance running_pos so the next turn picks up cleanly.
  startBlock writeResp
  let wrNOut := writeResp.param 0
  storeI64 wrNOut (← absAddr ptr TOKEN_COUNT_OFF)
  callVoid fnDetok [ptr]
  let outLen ← load64At ptr TEXT_LEN_OFF
  istore8 (← iconst32 10) (← iadd textOutPtr outLen)
  let _ ← call fnStdoutWrite [ptr, textOutOff64, (← iaddImm outLen 1)]
  -- Append assistant closing tokens to the cache: <|im_end|> then \n
  let endImPos ← iadd runningPos (← iadd nPrompt wrNOut)
  storeI32 imEndTok                (← absAddr ptr INFER_IN_OFF)
  storeI32 (← ireduce32 endImPos)  (← absAddr ptr (INFER_IN_OFF + 4))
  callVoid fnInfer [ptr]
  let endNlPos ← iaddImm endImPos 1
  storeI32 nlTok                   (← absAddr ptr INFER_IN_OFF)
  storeI32 (← ireduce32 endNlPos)  (← absAddr ptr (INFER_IN_OFF + 4))
  callVoid fnInfer [ptr]
  storeI64 (← iaddImm endNlPos 1) (← absAddr ptr RUNNING_POS_OFF)
  jump cliLoop.ref []


-- ── Initial memory: PTX kernel byte tail (shared by both algorithms) ─────────

theorem qwen2_all_proven_text :
    embedKernel.ptxText.isSome ∧ rmsNormKernel.ptxText.isSome
      ∧ biasAddDKernel.ptxText.isSome ∧ biasAddKVKernel.ptxText.isSome
      ∧ ropeQKernel.ptxText.isSome ∧ ropeKKernel.ptxText.isSome
      ∧ softmaxKernel.ptxText.isSome ∧ siluGateKernel.ptxText.isSome
      ∧ residualAddKernel.ptxText.isSome ∧ kvStoreKernel.ptxText.isSome
      ∧ argmaxKernel.ptxText.isSome := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- …and the text it carries is exactly the blob `Qwen2Proven` proves about —
    not a copy that could drift. -/
theorem qwen2_ships_proven_ptx :
    ptxEmbedLookup = Qwen2Proven.ptxEmbed
      ∧ ptxRmsNorm = Qwen2Proven.ptxRmsNorm
      ∧ ptxBiasAddD = Qwen2Proven.ptxAdd
      ∧ ptxBiasAddKV = Qwen2Proven.ptxAdd
      ∧ ptxRoPEQ = Qwen2Proven.ptxRope
      ∧ ptxRoPEK = Qwen2Proven.ptxRope
      ∧ ptxSoftmax = Qwen2Proven.ptxSoftmax
      ∧ ptxSiluGate = Qwen2Proven.ptxSilu
      ∧ ptxResidualAdd = Qwen2Proven.ptxAdd
      ∧ ptxKVStore = Qwen2Proven.ptxKVStore
      ∧ ptxArgmax = Qwen2Proven.ptxArgmax := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

def ptxEmbedBytes   : List UInt8 := ptxEmbedLookup.toUTF8.toList ++ [0]
def ptxRmsBytes     : List UInt8 := ptxRmsNorm.toUTF8.toList ++ [0]
def ptxBiasDBytes   : List UInt8 := ptxBiasAddD.toUTF8.toList ++ [0]
def ptxBiasKvBytes  : List UInt8 := ptxBiasAddKV.toUTF8.toList ++ [0]
def ptxRopeQBytes   : List UInt8 := ptxRoPEQ.toUTF8.toList ++ [0]
def ptxRopeKBytes   : List UInt8 := ptxRoPEK.toUTF8.toList ++ [0]
def ptxSoftmaxBytes : List UInt8 := ptxSoftmax.toUTF8.toList ++ [0]
def ptxSiluBytes    : List UInt8 := ptxSiluGate.toUTF8.toList ++ [0]
def ptxAddBytes     : List UInt8 := ptxResidualAdd.toUTF8.toList ++ [0]
def ptxKvStoreBytes : List UInt8 := ptxKVStore.toUTF8.toList ++ [0]
def ptxArgmaxBytes  : List UInt8 := ptxArgmax.toUTF8.toList ++ [0]

def u32le (n : Nat) : List UInt8 :=
  [ UInt8.ofNat (n        &&& 0xFF),
    UInt8.ofNat ((n >>> 8)  &&& 0xFF),
    UInt8.ofNat ((n >>> 16) &&& 0xFF),
    UInt8.ofNat ((n >>> 24) &&& 0xFF) ]

def systemTokenBytes : List UInt8 :=
  systemTokenIds.foldl (fun acc t => acc ++ u32le t) []

/-- Shared tail of `buildInitialMemory`: PTX kernel byte regions running from
    `PTX_EMBED_OFF` through `MEM_SIZE`.  Each algorithm prepends its own
    leading region (system tokens, plus optional KV-cache path bytes). -/
def buildInitialMemoryTail : List UInt8 :=
  let embed   := ptxEmbedBytes   ++ zeros (PTX_RMS_OFF     - PTX_EMBED_OFF   - ptxEmbedBytes.length)
  let rms     := ptxRmsBytes     ++ zeros (PTX_BIAS_D_OFF  - PTX_RMS_OFF     - ptxRmsBytes.length)
  let biasD   := ptxBiasDBytes   ++ zeros (PTX_BIAS_KV_OFF - PTX_BIAS_D_OFF  - ptxBiasDBytes.length)
  let biasKv  := ptxBiasKvBytes  ++ zeros (PTX_ROPE_Q_OFF  - PTX_BIAS_KV_OFF - ptxBiasKvBytes.length)
  let ropeQ   := ptxRopeQBytes   ++ zeros (PTX_ROPE_K_OFF  - PTX_ROPE_Q_OFF  - ptxRopeQBytes.length)
  let ropeK   := ptxRopeKBytes   ++ zeros (PTX_SOFTMAX_OFF - PTX_ROPE_K_OFF  - ptxRopeKBytes.length)
  let softmax := ptxSoftmaxBytes ++ zeros (PTX_SILU_OFF    - PTX_SOFTMAX_OFF - ptxSoftmaxBytes.length)
  let silu    := ptxSiluBytes    ++ zeros (PTX_ADD_OFF     - PTX_SILU_OFF    - ptxSiluBytes.length)
  let add     := ptxAddBytes     ++ zeros (PTX_KVSTORE_OFF - PTX_ADD_OFF     - ptxAddBytes.length)
  let kvstore := ptxKvStoreBytes ++ zeros (PTX_ARGMAX_OFF  - PTX_KVSTORE_OFF - ptxKvStoreBytes.length)
  let argmax  := ptxArgmaxBytes  ++ zeros (MEM_SIZE        - PTX_ARGMAX_OFF  - ptxArgmaxBytes.length)
  embed ++ rms ++ biasD ++ biasKv ++ ropeQ ++ ropeK ++ softmax ++ silu ++ add ++ kvstore ++ argmax

-- ── The memory map, as data ──────────────────────────────────────────────────

def memMap : RegionMap :=
  [ ⟨"io_offsets",         0x0000, 0x38⟩,
    ⟨"pinned_host_ptr",    PINNED_HOST_PTR_OFF, 8⟩,
    ⟨"pinned_id",          PINNED_ID_OFF, 4⟩,
    -- 16 typed buffer-id slots, 0x0048..0x0088
    ⟨"buffer_slots",       0x0048, 16 * 4⟩,
    ⟨"layer_idx",          LAYER_IDX_OFF, 8⟩,
    ⟨"pos",                POS_SLOT_OFF, 8⟩,
    ⟨"seq_len",            SEQ_LEN_SLOT_OFF, 8⟩,
    ⟨"layer_bufs",         LAYER_BUFS_BASE, N_LAYERS * LAYER_BUF_STRIDE⟩,
    ⟨"tok_buf_ptr",        TOK_BUF_PTR_OFF, 8⟩,
    ⟨"infer_in",           INFER_IN_OFF, 8⟩,
    ⟨"infer_out",          INFER_OUT_OFF, 4⟩,
    ⟨"token_count",        TOKEN_COUNT_OFF, 8⟩,
    ⟨"n_prompt",           N_PROMPT_OFF, 8⟩,
    ⟨"text_len",           TEXT_LEN_OFF, 8⟩,
    ⟨"ht_key",             HT_KEY_OFF, 8⟩,
    ⟨"ht_val",             HT_VAL_OFF, 8⟩,
    ⟨"rope_table_slot",    0x0638, 4⟩,
    ⟨"weights_path_ptr",   WEIGHTS_PATH_PTR_OFF, 8⟩,
    ⟨"tokenizer_path_ptr", TOKENIZER_PATH_PTR_OFF, 8⟩,
    ⟨"running_pos",        RUNNING_POS_OFF, 8⟩,
    ⟨"system_tokens",      SYSTEM_TOKENS_OFF, 256⟩,
    ⟨"bind_tables",        BIND_BASE, 0xC0⟩,
    ⟨"meta_stage",         META_STAGE_OFF, 24⟩,
    -- PTX slots.  Sizes are the gap to the next slot; `ptxFitsB` checks the
    -- emitted bytes actually fit, which `buildInitialMemoryTail` cannot — its
    -- `zeros (NEXT - THIS - len)` truncates to zero on `Nat` underflow and
    -- silently shifts every later slot.
    ⟨"ptx_embed",   PTX_EMBED_OFF,   PTX_RMS_OFF     - PTX_EMBED_OFF⟩,
    ⟨"ptx_rms",     PTX_RMS_OFF,     PTX_BIAS_D_OFF  - PTX_RMS_OFF⟩,
    ⟨"ptx_bias_d",  PTX_BIAS_D_OFF,  PTX_BIAS_KV_OFF - PTX_BIAS_D_OFF⟩,
    ⟨"ptx_bias_kv", PTX_BIAS_KV_OFF, PTX_ROPE_Q_OFF  - PTX_BIAS_KV_OFF⟩,
    ⟨"ptx_rope_q",  PTX_ROPE_Q_OFF,  PTX_ROPE_K_OFF  - PTX_ROPE_Q_OFF⟩,
    ⟨"ptx_rope_k",  PTX_ROPE_K_OFF,  PTX_SOFTMAX_OFF - PTX_ROPE_K_OFF⟩,
    ⟨"ptx_softmax", PTX_SOFTMAX_OFF, PTX_SILU_OFF    - PTX_SOFTMAX_OFF⟩,
    ⟨"ptx_silu",    PTX_SILU_OFF,    PTX_ADD_OFF     - PTX_SILU_OFF⟩,
    ⟨"ptx_add",     PTX_ADD_OFF,     PTX_KVSTORE_OFF - PTX_ADD_OFF⟩,
    ⟨"ptx_kvstore", PTX_KVSTORE_OFF, PTX_ARGMAX_OFF  - PTX_KVSTORE_OFF⟩,
    ⟨"ptx_argmax",  PTX_ARGMAX_OFF,  TOKEN_BUF_OFF   - PTX_ARGMAX_OFF⟩,
    ⟨"token_buf",   TOKEN_BUF_OFF,   TOKEN_BUF_BYTES⟩,
    ⟨"text_in",     TEXT_IN_OFF,     TEXT_IN_BYTES⟩,
    ⟨"text_out",    TEXT_OUT_OFF,    TEXT_OUT_BYTES⟩ ]

-- ---------------------------------------------------------------------------
-- The host program, against the kernels it launches
-- ---------------------------------------------------------------------------

/-- The layer-forward function, as a value. -/
def inferState : AlgorithmLib.IR.IRState := (inferFn.run {}).2

/-- **Device writes the launch model does not see, in `inferFn` itself.**

    `inferFn` directly performs the embed launch and the projection matvecs it
    does not delegate; the per-layer cuBLAS calls sit inside `layerStepFn`.
    This number is not a target to drive to zero by weakening the check — it is
    the size of the gap between "the launches are proven" and "the program is
    proven", and it goes to zero only when the calls themselves are replaced by
    kernels with `StageSpec`s. -/
def EXPECTED_UNMODELLED_WRITES : Nat := 1

/-- A device write the model **records but does not interpret** — a vendor call
    or a host→device copy.  Every field is `none`: its position in the sequence
    is what a composition needs, and claiming to have recovered a slot or a grid
    for a vendor call would be an invention. -/
def externRec (nm : String) (args : List AlgorithmLib.Clif.ArgDesc) :
    AlgorithmLib.Clif.LaunchRec :=
  { fnName := nm, kernelOff := none, nBufs := none, bindOff := none,
    gridX := none, blockX := none, args := args }

/-- **The LM-head projection's arguments, as the scan recovers them.**

    `cl_cublas_sgemv(ctx, trans, m, n, alpha, A, x, beta, y)`.  The dimensions
    are `D × VOCAB`, and the three buffer handles are identified by the slots
    they were loaded from — which is what distinguishes this call from the seven
    per-layer matvecs, three of which are the same `D × D` shape as each other.
    `1065353216` is `0x3F800000`, the bit pattern of `1.0f`. -/
def CUBLAS_LMHEAD_ARGS : List AlgorithmLib.Clif.ArgDesc :=
  [ .slot 16                       -- cuda context pointer
  , .const 1                       -- transpose
  , .const (Int.ofNat D)           -- m = 896
  , .const (Int.ofNat VOCAB)       -- n = 151936
  , .const 1065353216              -- alpha = 1.0f
  , .slot 112                      -- A = lmHead weights
  , .slot 76                       -- x = hdNorm
  , .const 0                       -- beta = 0.0f
  , .slot 116 ]                    -- y = logits

/-- **The final stage's launches**: the second RMS norm, then the argmax.  The
    LM-head projection between them is `cl_cublas_sgemv`, which is why this list
    has two entries and not three — the gap is visible in the shape of the
    declaration itself. -/
def expectedFinalLaunches : List AlgorithmLib.Clif.LaunchRec :=
  [ { fnName    := "cl_cuda_launch"
      kernelOff := some (Int.ofNat PTX_RMS_OFF)
      nBufs     := some (Int.ofNat rmsNormKernel.params.length)
      bindOff   := some (Int.ofNat BIND_RMS2)
      gridX     := some 1
      blockX    := some 32 }
    -- the LM-head projection: declared, not interpreted
  , externRec "cl_cublas_sgemv" CUBLAS_LMHEAD_ARGS
  , { fnName    := "cl_cuda_launch"
      kernelOff := some (Int.ofNat PTX_ARGMAX_OFF)
      nBufs     := some (Int.ofNat argmaxKernel.params.length)
      bindOff   := some (Int.ofNat BIND_ARGMAX)
      gridX     := some 1
      blockX    := some 32 } ]


/-- **The launch sequence this program must perform.**

    **These are now elaboration-time theorems, not a generator-time check.**
    The fallback to an `IO` check existed because `native_decide` on a statement
    about an `IRBuilder` run measured 43.5 s, and kernel `decide` did not finish
    in 180 s. Both numbers were artefacts of a bug: `Clif.Env` was a chain of
    closures storing recipes rather than values, so every lookup re-derived the
    binding it landed on and the cost doubled with depth. With `Env` a strict
    association list the same four statements elaborate in **1.26 s total**, so
    they are stated below as theorems and the check is gone.

    Kernel `decide` is still out of reach — reducing a monadic builder in the
    kernel exhausts the stack — so these carry `Lean.ofReduceBool` and
    `Lean.trustCompiler`. That is the price of a closed fact about a program;
    the *general* facts (`scanBlock_length`, `scanBlock_fnName`) are ordinary
    inductions and carry nothing. -/
def expectedLaunches : List AlgorithmLib.Clif.LaunchRec :=
  [ -- the weight upload: source and destination handles recovered from the
    -- slots they were loaded from, which is what tells one vendor-ish call
    -- from another of the same shape
    externRec "cl_cuda_upload_ptr" [.slot 16, .slot 132, .opaque, .const 24]
  , { fnName    := "cl_cuda_launch"
      kernelOff := some (Int.ofNat PTX_EMBED_OFF)
      nBufs     := some (Int.ofNat embedKernel.params.length)
      bindOff   := some 2048
      gridX     := some (Int.ofNat (D / 32))
      blockX    := some 32 } ]

/-- **A device write, reduced to what a drift check needs**: which primitive,
    which PTX slot, how many blocks.  Enough to catch a wrong kernel, a wrong
    grid, a reordering, or an op appearing or disappearing — without pinning
    every bind offset by hand. -/
def opSig (r : AlgorithmLib.Clif.LaunchRec) : String × Option Int × Option Int :=
  (r.fnName, r.kernelOff, r.gridX)

/-- Shorthands for the two kinds of entry. -/
def kl (ptx grid : Nat) : String × Option Int × Option Int :=
  ("cl_cuda_launch", some (Int.ofNat ptx), some (Int.ofNat grid))
def bl (nm : String) : String × Option Int × Option Int := (nm, none, none)

/-- **One transformer layer's attention half.**

    Ten kernel launches, four `sgemv`s and two batched `sgemm`s, in this order.
    The `sgemm`s are the score and output contractions; like the `sgemv`s they
    write device memory and are not modelled kernels, which is why they appear
    here rather than being invisible. -/
def expectedAttnOps : List (String × Option Int × Option Int) :=
  [ kl PTX_RMS_OFF 1
  , bl "cl_cublas_sgemv", bl "cl_cublas_sgemv", bl "cl_cublas_sgemv"
  , kl PTX_BIAS_D_OFF 28
  , kl PTX_BIAS_KV_OFF 4, kl PTX_BIAS_KV_OFF 4
  , kl PTX_ROPE_Q_OFF 14, kl PTX_ROPE_K_OFF 2
  , kl PTX_KVSTORE_OFF 2, kl PTX_KVSTORE_OFF 2
  , bl "cl_cublas_sgemm_strided_batched"
  , kl PTX_SOFTMAX_OFF 14
  , bl "cl_cublas_sgemm_strided_batched"
  , bl "cl_cublas_sgemv"
  , kl PTX_ADD_OFF 28 ]

/-- …and its feed-forward half: three launches and three `sgemv`s. -/
def expectedFfnOps : List (String × Option Int × Option Int) :=
  [ kl PTX_RMS_OFF 1
  , bl "cl_cublas_sgemv", bl "cl_cublas_sgemv"
  , kl PTX_SILU_OFF 152
  , bl "cl_cublas_sgemv"
  , kl PTX_ADD_OFF 28 ]

/-- **The per-token totals, derived from the parts.**

    `LAYERS · (attn + ffn) + embed + upload + final`.  Stated as a definition so
    the numbers cannot drift from the sequences above. -/
def opsPerLayer : Nat := expectedAttnOps.length + expectedFfnOps.length
def opsPerToken : Nat := N_LAYERS * opsPerLayer + 2 + 3

-- ---------------------------------------------------------------------------
-- Which launched kernels are stages — the ledger, as a theorem
-- ---------------------------------------------------------------------------

/-! A kernel proven in isolation and a kernel usable as a *step of a pipeline*
    are different claims.  The second needs a `StageSpec`: a frame condition, a
    value, and a guarantee the value ignores what the block does not own.

    Every one of the model's kernels now has one (`Qwen2Proven.Stage`).  The
    risk this section guards is not that some are missing — it is that the list
    stops being maintained while the pipeline grows, and a later reader takes
    "the kernels are proven" to mean "the pipeline is proven".

    So the list is derived from the launch sequence rather than written down:
    `stagedSlots` names what has a stage, and `unstagedSlots` is *computed* by
    filtering the actual op lists.  Add a kernel to a layer without giving it a
    stage and `unstaged_kernels` below changes — the ledger cannot silently
    drift from the pipeline. -/

/-- PTX slots whose kernel has a `StageSpec`. -/
def stagedSlots : List Nat :=
  [ PTX_EMBED_OFF, PTX_RMS_OFF, PTX_BIAS_D_OFF, PTX_BIAS_KV_OFF
  , PTX_ROPE_Q_OFF, PTX_ROPE_K_OFF, PTX_KVSTORE_OFF, PTX_SILU_OFF, PTX_ADD_OFF
  , PTX_SOFTMAX_OFF, PTX_ARGMAX_OFF ]

/-- Every kernel slot the per-token op lists actually launch. -/
def launchedSlots : List Nat :=
  (expectedAttnOps ++ expectedFfnOps).filterMap
    (fun op => match op.2.1 with | some k => some k.toNat | none => none)

/-- …minus the ones with a stage.  This is the gap, computed. -/
def unstagedSlots : List Nat :=
  launchedSlots.filter (fun k => !stagedSlots.contains k) |>.eraseDups

/-- **Every kernel the per-token loop launches is a stage.**

    Stated by `decide`, from the op lists the launch-model theorems also use, so
    it is the same sequence in both places.  Add a kernel to a layer without
    giving it a `StageSpec` and this stops being `[]`. -/
theorem unstaged_kernels : unstagedSlots = [] := by decide

def finalSlots : List Nat := [PTX_RMS_OFF, PTX_ARGMAX_OFF]

/-- …and the sampling tail adds none. -/
theorem unstaged_final :
    finalSlots.filter (fun k => !stagedSlots.contains k) = [] := by decide

/-- **All eleven shipped kernels are stages.**  The count is derived,
    not asserted: `stagedSlots` is checked against the kernel table by
    `staged_are_real` below, so a slot that is not a real kernel cannot pad it. -/
theorem staged_count : stagedSlots.length = 11 := by decide

/-- Every slot claimed to have a stage is a slot the model actually writes a
    kernel into.  Without this the ledger could be inflated with invented
    offsets and every theorem above would still hold. -/
theorem staged_are_real :
    stagedSlots.all (fun k => (memMap.map (fun r => r.off)).contains k) = true := by
  decide

-- ---------------------------------------------------------------------------
-- The device-write sequence, as theorems
-- ---------------------------------------------------------------------------

/-- Every instruction in the entry function has a meaning in this model. -/
theorem inferFn_modellable :
    AlgorithmLib.Clif.blocksModellableB (inferFn.run {}).2.allBlocks = true := by
  native_decide

/-- **The entry function's device writes are exactly the declared ones.** -/
theorem inferFn_writes :
    AlgorithmLib.Clif.launchesOf (inferFn.run {}).2 = expectedLaunches := by
  native_decide

theorem inferFinalFn_modellable :
    AlgorithmLib.Clif.blocksModellableB (inferFinalFn.run {}).2.allBlocks = true := by
  native_decide

/-- **The sampling tail's, likewise** — norm, vendor projection, argmax. -/
theorem inferFinalFn_writes :
    AlgorithmLib.Clif.launchesOf (inferFinalFn.run {}).2 = expectedFinalLaunches := by
  native_decide

/-- **No two regions overlap.** -/
theorem memMap_ok : memMap.okB = true := by decide

/-- **Every region fits inside the declared memory.** -/
theorem memMap_within : RegionMap.withinB MEM_SIZE memMap = true := by decide

/-- **The token buffer holds every token the reader can produce.**

    `tokenizeInitFn` loops `textLen` times writing one `u32` per input byte, and
    `textLen` comes from `cl_stdin_readline` bounded by `MAX_RECV`.  So the loop
    bound and the region capacity are two separate constants that must agree,
    and nothing in `memMap_ok` relates them — the regions were disjoint the
    whole time the writer was running past one of them.

    Depends on the FFI contract that `cl_stdin_readline(_, _, n)` returns at
    most `n`; the subsequent CR/LF trims only shrink it. -/
theorem tokenBuf_holds_input : memMap.holdsB "token_buf" MAX_RECV 4 = true := by decide

/-- The same obligation for the text buffers, so a wider reader cannot outrun
    the buffer it reads into either. -/
theorem textIn_holds_input : memMap.holdsB "text_in" MAX_RECV 1 = true := by decide

/-- **…and after the chat wrapper is shifted in.**  `wrapWrite` moves every
    prompt token up by `PREFIX_LEN` and appends the suffix, so the highest slot
    touched is `rawPromptN + WRAP_LEN − 1`.  Tokenization bounds `rawPromptN` by
    `MAX_RECV` (BPE only merges), which is what makes this decidable. -/
theorem tokenBuf_holds_wrapped :
    memMap.holdsB "token_buf" (MAX_RECV + WRAP_LEN) 4 = true := by decide

/-- **The whole turn fits the KV cache.**  System preamble, then the wrapped
    prompt, then everything decode may generate — all indexed into a cache with
    `MAX_SEQ` positions.  This is the constraint `MAX_RECV` is *derived* from;
    stating it separately is what catches a change to any other term. -/
theorem turn_fits_kv_cache :
    SYSTEM_TOKEN_COUNT + MAX_RECV + WRAP_LEN + MAX_DECODE ≤ MAX_SEQ := by decide

/-! **Negative guards.**  A check that cannot fail proves nothing, so these
    pin the two ways `holdsB` could be vacuously true: an over-large bound must
    be rejected, and a region that is not there must be `false` rather than
    silently unchecked. -/
example : memMap.holdsB "token_buf" (TOKEN_BUF_CAP + 1) 4 = false := by decide
example : memMap.holdsB "no_such_region" 1 1 = false := by decide

/-- Each emitted PTX module, against the slot it is written into. -/
def ptxSlotFits : List (String × Nat × Nat) :=
  [ ("embed",   ptxEmbedBytes.length,   PTX_RMS_OFF     - PTX_EMBED_OFF),
    ("rms",     ptxRmsBytes.length,     PTX_BIAS_D_OFF  - PTX_RMS_OFF),
    ("bias_d",  ptxBiasDBytes.length,   PTX_BIAS_KV_OFF - PTX_BIAS_D_OFF),
    ("bias_kv", ptxBiasKvBytes.length,  PTX_ROPE_Q_OFF  - PTX_BIAS_KV_OFF),
    ("rope_q",  ptxRopeQBytes.length,   PTX_ROPE_K_OFF  - PTX_ROPE_Q_OFF),
    ("rope_k",  ptxRopeKBytes.length,   PTX_SOFTMAX_OFF - PTX_ROPE_K_OFF),
    ("softmax", ptxSoftmaxBytes.length, PTX_SILU_OFF    - PTX_SOFTMAX_OFF),
    ("silu",    ptxSiluBytes.length,    PTX_ADD_OFF     - PTX_SILU_OFF),
    ("add",     ptxAddBytes.length,     PTX_KVSTORE_OFF - PTX_ADD_OFF),
    ("kvstore", ptxKvStoreBytes.length, PTX_ARGMAX_OFF  - PTX_KVSTORE_OFF),
    ("argmax",  ptxArgmaxBytes.length,  TOKEN_BUF_OFF   - PTX_ARGMAX_OFF) ]

def ptxFitsB : Bool := ptxSlotFits.all (fun e => e.2.1 ≤ e.2.2)

/-- **Every emitted kernel fits its slot.**  Without this, an oversized kernel
    truncates its own zero padding and shifts every later kernel's base — the
    symptom being `cl_cuda_launch: load kernel failed`, or worse, a *different*
    kernel loading successfully at the wrong offset. -/
theorem ptx_fits : ptxFitsB = true := by native_decide

-- ---------------------------------------------------------------------------
-- One layer, as a plan the shipped program realises
-- ---------------------------------------------------------------------------

section Realisation
open AlgorithmLib.Clif AlgorithmLib.ML Qwen2Proven.Stage
set_option maxHeartbeats 2000000

/-!
  **The last hand-chosen thing, removed.**

  `Qwen2Proven.Stage` proves what a layer's twenty-two device writes do to
  memory.  Which *buffers* they do it to was, until now, a numbering written
  next to the plan — so the theorem was about a layer-shaped transformation,
  not about this program's.  The two disagreed in three places, none of them
  visible to `0 sorry`: the attention half and the feed-forward half used
  different numbers for the same residual stream, five kernels read five
  different "meta" buffers, and the output projection wrote a buffer the
  residual add did not read.

  Now the numbering is `Qwen2Proven.Stage.bufOf` applied to handles
  `Clif.bindsOf` recovered from the emitted stores, the binds are computed from
  the same arrays, and the table below supplies those arrays to both.  What is
  left assumed is exactly `bufOf`: a renaming, injective, from recovered
  handles to buffer numbers.
-/

/-- The descriptor pointer.  `entryBlock` returns `v0` in every generated
    function, so a fixed layout slot is `near k` and a per-layer weight, whose
    base is a runtime-computed address, is `far`. -/
def ROOT : Nat := 0

/-- A launch, in the form `Clif.deviceOpsOf` produces. -/
def klOp (ptx nb bo grid : Nat) (bs : List BufDesc) : DeviceOp :=
  ( { fnName    := "cl_cuda_launch"
      kernelOff := some (Int.ofNat ptx)
      nBufs     := some (Int.ofNat nb)
      bindOff   := some (Int.ofNat bo)
      gridX     := some (Int.ofNat grid)
      blockX    := some 32 }
  , { bufs := some bs } )

/-- …and a vendor call.  `LaunchRec.args` is *derived* from the base-aware
    descriptors by `BufDesc.toArg`, which is what `bufDescOf_toArg` says the
    scan does — so the two views cannot be written inconsistently. -/
def vlOp (nm : String) (as : List BufDesc) : DeviceOp :=
  (externRec nm (as.map BufDesc.toArg), { args := as })

/-- `1.0f` and `0.5f` as bit patterns, and the attention scale. -/
private def F1 : BufDesc := .const 1065353216
private def FR : BufDesc := .const 1040187392   -- 1/8, the 1/√head_dim scale

/-- **The attention half's device writes, with what each one bound.**  Pinned
    to `inferLayerAttnFn` by `Qwen2.attn_ops_are`. -/
def attnOps : List DeviceOp :=
  [ klOp PTX_RMS_OFF     3 BIND_RMS1   1   BS_A_NORM
  , vlOp "cl_cublas_sgemv"
      [.near 16, .const 1, .const 896, .const 896, F1, .far 8 4,  .near 76, .const 0, .near 80]
  , vlOp "cl_cublas_sgemv"
      [.near 16, .const 1, .const 896, .const 128, F1, .far 8 12, .near 76, .const 0, .near 84]
  , vlOp "cl_cublas_sgemv"
      [.near 16, .const 1, .const 896, .const 128, F1, .far 8 20, .near 76, .const 0, .near 88]
  , klOp PTX_BIAS_D_OFF  2 BIND_BIAS_Q 28  BS_A_BIASQ
  , klOp PTX_BIAS_KV_OFF 2 BIND_BIAS_K 4   BS_A_BIASK
  , klOp PTX_BIAS_KV_OFF 2 BIND_BIAS_V 4   BS_A_BIASV
  , klOp PTX_ROPE_Q_OFF  3 BIND_ROPE_Q 14  BS_A_ROPEQ
  , klOp PTX_ROPE_K_OFF  3 BIND_ROPE_K 2   BS_A_ROPEK
  , klOp PTX_KVSTORE_OFF 3 BIND_KV_K   2   BS_A_KVK
  , klOp PTX_KVSTORE_OFF 3 BIND_KV_V   2   BS_A_KVV
  , vlOp "cl_cublas_sgemm_strided_batched"
      [.near 16, .const 1, .const 0, .near 152, .const 7, .const 64, FR,
       .far 8 48, .const 131072, .near 80, .const 448, .const 0, .near 124,
       .opaque, .const 2]
  , klOp PTX_SOFTMAX_OFF 3 BIND_SOFTMAX 14 BS_A_SOFT
  , vlOp "cl_cublas_sgemm_strided_batched"
      [.near 16, .const 0, .const 0, .const 64, .const 7, .near 152, F1,
       .far 8 52, .const 131072, .near 128, .opaque, .const 0, .near 92,
       .const 448, .const 2]
  , vlOp "cl_cublas_sgemv"
      [.near 16, .const 1, .const 896, .const 896, F1, .far 8 28, .near 92, .const 0, .near 76]
  , klOp PTX_ADD_OFF     2 BIND_ADD1   28  BS_A_ADD ]

/-- **…and the feed-forward half's.** -/
def ffnOps : List DeviceOp :=
  [ klOp PTX_RMS_OFF  3 BIND_RMS2 1   BS_FFN_NORM
  , vlOp "cl_cublas_sgemv"
      [.near 16, .const 1, .const 896, .const 4864, F1, .far 8 36, .near 76, .const 0, .near 96]
  , vlOp "cl_cublas_sgemv"
      [.near 16, .const 1, .const 896, .const 4864, F1, .far 8 40, .near 76, .const 0, .near 100]
  , klOp PTX_SILU_OFF 3 BIND_SILU 152 BS_FFN_SILU
  , vlOp "cl_cublas_sgemv"
      [.near 16, .const 1, .const 4864, .const 896, F1, .far 8 44, .near 104, .const 0, .near 92]
  , klOp PTX_ADD_OFF  2 BIND_ADD2 28  BS_FFN_ADD ]


-- ---------------------------------------------------------------------------
-- The same sequences, as a declared host program
-- ---------------------------------------------------------------------------

/-! `attnOps`, `ffnOps`, `entryOps` and `finalOps` are *lists*.  A list of
    twenty-two device writes says nothing about a loop running twenty-four
    times — `tokenOps` had `List.replicate N_LAYERS layerOps` in its definition,
    and a definition is not a theorem: it could have said 23 and nothing would
    have noticed.

    `HStmt` is the program those lists come from.  `forN 24` is a node, and
    `HStmt.launches`/`binds` recurse through it, so the repetition is *derived*.
    Everything below is written from the same literals as the ops lists via
    `argOf`, and the two are then proven equal. -/

open AlgorithmLib.Host in
/-- The attention half, declared. -/
def attnDriver (fnOf : String → FnRef) : HStmt :=
  HStmt.seqs
    [ .launch (kStep PTX_RMS_OFF     3 BIND_RMS1   1   BS_A_NORM)
    , .extern (vStep fnOf "cl_cublas_sgemv"
        [.near 16, .const 1, .const 896, .const 896, F1, .far 8 4,  .near 76, .const 0, .near 80])
    , .extern (vStep fnOf "cl_cublas_sgemv"
        [.near 16, .const 1, .const 896, .const 128, F1, .far 8 12, .near 76, .const 0, .near 84])
    , .extern (vStep fnOf "cl_cublas_sgemv"
        [.near 16, .const 1, .const 896, .const 128, F1, .far 8 20, .near 76, .const 0, .near 88])
    , .launch (kStep PTX_BIAS_D_OFF  2 BIND_BIAS_Q 28  BS_A_BIASQ)
    , .launch (kStep PTX_BIAS_KV_OFF 2 BIND_BIAS_K 4   BS_A_BIASK)
    , .launch (kStep PTX_BIAS_KV_OFF 2 BIND_BIAS_V 4   BS_A_BIASV)
    , .launch (kStep PTX_ROPE_Q_OFF  3 BIND_ROPE_Q 14  BS_A_ROPEQ)
    , .launch (kStep PTX_ROPE_K_OFF  3 BIND_ROPE_K 2   BS_A_ROPEK)
    , .launch (kStep PTX_KVSTORE_OFF 3 BIND_KV_K   2   BS_A_KVK)
    , .launch (kStep PTX_KVSTORE_OFF 3 BIND_KV_V   2   BS_A_KVV)
    , .extern (vStep fnOf "cl_cublas_sgemm_strided_batched"
        [.near 16, .const 1, .const 0, .near 152, .const 7, .const 64, FR,
         .far 8 48, .const 131072, .near 80, .const 448, .const 0, .near 124,
         .opaque, .const 2])
    , .launch (kStep PTX_SOFTMAX_OFF 3 BIND_SOFTMAX 14 BS_A_SOFT)
    , .extern (vStep fnOf "cl_cublas_sgemm_strided_batched"
        [.near 16, .const 0, .const 0, .const 64, .const 7, .near 152, F1,
         .far 8 52, .const 131072, .near 128, .opaque, .const 0, .near 92,
         .const 448, .const 2])
    , .extern (vStep fnOf "cl_cublas_sgemv"
        [.near 16, .const 1, .const 896, .const 896, F1, .far 8 28, .near 92, .const 0, .near 76])
    , .launch (kStep PTX_ADD_OFF     2 BIND_ADD1   28  BS_A_ADD) ]

open AlgorithmLib.Host in
/-- …and the feed-forward half. -/
def ffnDriver (fnOf : String → FnRef) : HStmt :=
  HStmt.seqs
    [ .launch (kStep PTX_RMS_OFF  3 BIND_RMS2 1   BS_FFN_NORM)
    , .extern (vStep fnOf "cl_cublas_sgemv"
        [.near 16, .const 1, .const 896, .const 4864, F1, .far 8 36, .near 76, .const 0, .near 96])
    , .extern (vStep fnOf "cl_cublas_sgemv"
        [.near 16, .const 1, .const 896, .const 4864, F1, .far 8 40, .near 76, .const 0, .near 100])
    , .launch (kStep PTX_SILU_OFF 3 BIND_SILU 152 BS_FFN_SILU)
    , .extern (vStep fnOf "cl_cublas_sgemv"
        [.near 16, .const 1, .const 4864, .const 896, F1, .far 8 44, .near 104, .const 0, .near 92])
    , .launch (kStep PTX_ADD_OFF  2 BIND_ADD2 28  BS_FFN_ADD) ]

open AlgorithmLib.Host in
/-- **The declared attention half performs `attnOps`.** -/
theorem attnDriver_deviceOps (fnOf : String → FnRef) :
    (attnDriver fnOf).deviceOps = attnOps := rfl

open AlgorithmLib.Host in
/-- **…and the declared feed-forward half performs `ffnOps`.** -/
theorem ffnDriver_deviceOps (fnOf : String → FnRef) :
    (ffnDriver fnOf).deviceOps = ffnOps := rfl

variable (gim : Buf → Nat → Nat)

/-- **Which PTX slot, bound to which buffers, is which proven stage.**

    Thirteen entries for a layer's thirteen launches.  Each supplies the *same*
    recovered array (`BS_*`) that the stage's own bind was derived from, so a
    table entry cannot name a stage proven about different buffers: the two
    read the same value. -/
noncomputable def layerKernels (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : List KernelBinding :=
  [ ⟨PTX_RMS_OFF,     BIND_RMS1,    BS_A_NORM,   aNormStage gim⟩
  , ⟨PTX_BIAS_D_OFF,  BIND_BIAS_Q,  BS_A_BIASQ,  aBiasQStage gim⟩
  , ⟨PTX_BIAS_KV_OFF, BIND_BIAS_K,  BS_A_BIASK,  aBiasKStage gim⟩
  , ⟨PTX_BIAS_KV_OFF, BIND_BIAS_V,  BS_A_BIASV,  aBiasVStage gim⟩
  , ⟨PTX_ROPE_Q_OFF,  BIND_ROPE_Q,  BS_A_ROPEQ,  aRopeQStage gim⟩
  , ⟨PTX_ROPE_K_OFF,  BIND_ROPE_K,  BS_A_ROPEK,  aRopeKStage gim⟩
  , ⟨PTX_KVSTORE_OFF, BIND_KV_K,    BS_A_KVK,    aKvKStage gim⟩
  , ⟨PTX_KVSTORE_OFF, BIND_KV_V,    BS_A_KVV,    aKvVStage gim⟩
  , ⟨PTX_SOFTMAX_OFF, BIND_SOFTMAX, BS_A_SOFT,   aSoftStage gim h hm⟩
  , ⟨PTX_ADD_OFF,     BIND_ADD1,    BS_A_ADD,    aAddStage gim⟩
  , ⟨PTX_RMS_OFF,     BIND_RMS2,    BS_FFN_NORM, ffnNormStage⟩
  , ⟨PTX_SILU_OFF,    BIND_SILU,    BS_FFN_SILU, ffnSiluStage⟩
  , ⟨PTX_ADD_OFF,     BIND_ADD2,    BS_FFN_ADD,  ffnAddStage⟩ ]

/-- **…and which vendor call is which declared step.**

    Keyed on the recovered argument descriptors, because a vendor call has no
    PTX slot: three of a layer's seven `sgemv`s are the same 896×896 shape and
    only the matrix handle tells them apart.  Base-aware, so `far 8 4` (Wq, at
    offset 4 of the per-layer base) is not the same descriptor as a fixed slot
    4 would be. -/
noncomputable def layerDeclared : List DeclaredBinding :=
  [ ⟨"cl_cublas_sgemv",
     [.near 16, .const 1, .const 896, .const 896, F1, .far 8 4, .near 76, .const 0, .near 80],
     cublasStep B_WQ B_XN B_Q D D⟩
  , ⟨"cl_cublas_sgemv",
     [.near 16, .const 1, .const 896, .const 128, F1, .far 8 12, .near 76, .const 0, .near 84],
     cublasStep B_WK B_XN B_K KV_DIM D⟩
  , ⟨"cl_cublas_sgemv",
     [.near 16, .const 1, .const 896, .const 128, F1, .far 8 20, .near 76, .const 0, .near 88],
     cublasStep B_WV B_XN B_V KV_DIM D⟩
  , ⟨"cl_cublas_sgemm_strided_batched",
     [.near 16, .const 1, .const 0, .near 152, .const 7, .const 64, FR,
      .far 8 48, .const 131072, .near 80, .const 448, .const 0, .near 124,
      .opaque, .const 2],
     sgemmBatchedStep B_KC B_Q B_SC MAX_SEQ HEAD_DIM⟩
  , ⟨"cl_cublas_sgemm_strided_batched",
     [.near 16, .const 0, .const 0, .const 64, .const 7, .near 152, F1,
      .far 8 52, .const 131072, .near 128, .opaque, .const 0, .near 92,
      .const 448, .const 2],
     sgemmBatchedStep B_VC B_PR B_AO D MAX_SEQ⟩
  , ⟨"cl_cublas_sgemv",
     [.near 16, .const 1, .const 896, .const 896, F1, .far 8 28, .near 92, .const 0, .near 76],
     cublasStep B_WO B_AO B_XN D D⟩
  , ⟨"cl_cublas_sgemv",
     [.near 16, .const 1, .const 896, .const 4864, F1, .far 8 36, .near 76, .const 0, .near 96],
     cublasStep B_WG B_XN B_GATE D_FF D⟩
  , ⟨"cl_cublas_sgemv",
     [.near 16, .const 1, .const 896, .const 4864, F1, .far 8 40, .near 76, .const 0, .near 100],
     cublasStep B_WU B_XN B_UP D_FF D⟩
  , ⟨"cl_cublas_sgemv",
     [.near 16, .const 1, .const 4864, .const 896, F1, .far 8 44, .near 104, .const 0, .near 92],
     cublasStep B_WD B_ACT B_AO D D_FF⟩ ]

/-- **The attention half's sixteen device writes resolve to `attnPlan`.**

    Not "a plan with sixteen steps" — *that* plan, the one `attn_computes` is
    about.  Every launch had to match on slot, bind offset, bind contents and
    grid; every vendor call on name and recovered arguments. -/
theorem attn_ops_steps (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    stepsOf? (layerKernels gim h hm) layerDeclared attnOps
      = some (attnPlan gim h hm).steps := rfl

theorem attn_ops_realise_plan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (layerKernels gim h hm) layerDeclared attnOps = some (attnPlan gim h hm) := by
  show (stepsOf? (layerKernels gim h hm) layerDeclared attnOps).map Plan.mk = _
  rw [attn_ops_steps]
  rfl

/-- **…and the feed-forward half's six resolve to `ffnPlan`.** -/
theorem ffn_ops_steps (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    stepsOf? (layerKernels gim h hm) layerDeclared ffnOps = some ffnPlan.steps := rfl

theorem ffn_ops_realise_plan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (layerKernels gim h hm) layerDeclared ffnOps = some ffnPlan := by
  show (stepsOf? (layerKernels gim h hm) layerDeclared ffnOps).map Plan.mk = _
  rw [ffn_ops_steps]
  rfl

/-- **A whole layer: twenty-two device writes, one plan.**

    Assembled from the two halves rather than re-scanned.  The scan is what
    costs — each op is a linear walk of the kernel and declared tables — so
    resolving the layer a third time after resolving both halves is the same
    search done twice.  `stepsOf?_append` is exactly the lemma that avoids it,
    and it is the same move `replicate_ops_steps` makes for the loop. -/
theorem layer_ops_realise_plan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (layerKernels gim h hm) layerDeclared (attnOps ++ ffnOps)
      = some (layerPlan gim h hm) := by
  show (stepsOf? (layerKernels gim h hm) layerDeclared (attnOps ++ ffnOps)).map Plan.mk = _
  rw [stepsOf?_append _ _ _ _ _ _ (attn_ops_steps gim h hm) (ffn_ops_steps gim h hm)]
  rfl

/-- The gap, still counted, now against the program's own write sequence. -/
theorem layer_ops_declaredCount (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    declaredCountOf (layerKernels gim h hm) layerDeclared (attnOps ++ ffnOps)
      = some 9 := by
  show ((planOf? (layerKernels gim h hm) layerDeclared (attnOps ++ ffnOps)).map
          Plan.declaredCount) = _
  rw [layer_ops_realise_plan]
  rfl

/-- **…and how much of that gap no law even describes.**

    Nine of a layer's twenty-two steps are vendor calls.  Seven of the nine are
    `cl_cublas_sgemv`, and `cublasStep_isMatvec` says what those compute given
    `Law.cublasIsMatvec`.  The remaining two are the batched contractions, for
    which no equation is stated at all — so this is the number of steps that
    are assumed *without a statement of what is assumed*. -/
theorem layer_declaredLawGap (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    (layerPlan gim h hm).declaredLawGap = 2 := rfl

-- ── The prologue and the tail, likewise ────────────────────────────────────

/-- **`inferFn`'s two device writes**: the meta upload, then the embedding
    gather.  Pinned by `entry_ops_are`. -/
def entryOps : List DeviceOp :=
  [ vlOp "cl_cuda_upload_ptr" [.near 16, .near 132, .opaque, .const 24]
  , klOp PTX_EMBED_OFF 3 BIND_EMBED 28 BS_EMBED ]

/-- **`inferFinalFn`'s three**: the final norm, the LM-head projection, the
    argmax.  The projection is a vendor GEMV, which is why this list has three
    entries and only two of them are launches. -/
def finalOps : List DeviceOp :=
  [ klOp PTX_RMS_OFF 3 BIND_RMS2 1 BS_F_NORM
  , vlOp "cl_cublas_sgemv"
      [.near 16, .const 1, .const 896, .const 151936, F1, .near 112, .near 76,
       .const 0, .near 116]
  , klOp PTX_ARGMAX_OFF 2 BIND_ARGMAX 1 BS_ARGMAX ]

/-- The layer table, extended to the whole token.

    The final RMSNorm reuses `BIND_RMS2` — the same slot *and* the same bind
    offset as the feed-forward norm — and is a different step, because it binds
    a different weight buffer.  Nothing but the recovered array distinguishes
    them, which is the case the bind check exists for. -/
noncomputable def tokenKernels (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : List KernelBinding :=
  layerKernels gim h hm ++
    [ ⟨PTX_EMBED_OFF,  BIND_EMBED,  BS_EMBED,  eEmbedStage gim⟩
    , ⟨PTX_RMS_OFF,    BIND_RMS2,   BS_F_NORM, fNormStage gim⟩
    , ⟨PTX_ARGMAX_OFF, BIND_ARGMAX, BS_ARGMAX, fArgmaxStage gim⟩ ]

noncomputable def tokenDeclared : List DeclaredBinding :=
  layerDeclared ++
    [ ⟨"cl_cuda_upload_ptr", [.near 16, .near 132, .opaque, .const 24],
       uploadStep B_META⟩
    , ⟨"cl_cublas_sgemv",
       [.near 16, .const 1, .const 896, .const 151936, F1, .near 112, .near 76,
        .const 0, .near 116],
       cublasStep B_LMH B_XN B_LOG VOCAB D⟩ ]

/-- **`inferFn`'s device writes, and what each bound.** -/
theorem entry_ops_are :
    deviceOpsOf ROOT (inferFn.run {}).2 = entryOps := by native_decide

/-- **…and `inferFinalFn`'s.** -/
theorem final_ops_are :
    deviceOpsOf ROOT (inferFinalFn.run {}).2 = finalOps := by native_decide

theorem entry_ops_realise_plan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (tokenKernels gim h hm) tokenDeclared entryOps = some (entryPlan gim) := rfl

theorem final_ops_realise_plan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (tokenKernels gim h hm) tokenDeclared finalOps = some (finalPlan gim) := rfl

/-- **The shipped prologue realises `entryPlan`.** -/
theorem entry_program_realises_plan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (tokenKernels gim h hm) tokenDeclared
        (deviceOpsOf ROOT (inferFn.run {}).2) = some (entryPlan gim) := by
  rw [entry_ops_are]; exact entry_ops_realise_plan gim h hm

/-- **…and the shipped sampling tail realises `finalPlan`.** -/
theorem final_program_realises_plan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (tokenKernels gim h hm) tokenDeclared
        (deviceOpsOf ROOT (inferFinalFn.run {}).2) = some (finalPlan gim) := by
  rw [final_ops_are]; exact final_ops_realise_plan gim h hm

/-- **Every one of the eleven staged kernels is now in a plan tied to the
    program.**  Nine sit in the layer, two outside it; before this the ledger
    said all eleven had a `StageSpec` while only nine appeared in a plan. -/
theorem every_staged_kernel_planned :
    stagedSlots.all (fun k =>
      (attnOps ++ ffnOps ++ entryOps ++ finalOps).any
        (fun op => op.1.kernelOff = some (Int.ofNat k))) = true := by
  decide

-- ── A whole token ──────────────────────────────────────────────────────────

/-- One layer's twenty-two device writes. -/
def layerOps : List DeviceOp := attnOps ++ ffnOps

/-- **A token's five hundred and thirty-three**: the prologue, twenty-four
    layers, the sampling tail.  `inferLayerFn` writes nothing itself, so a
    layer's sequence is the two halves concatenated, and the loop contributes
    that sequence `N_LAYERS` times. -/
def tokenOps : List DeviceOp :=
  entryOps ++ ((List.replicate N_LAYERS layerOps).flatten ++ finalOps)

noncomputable def tokenPlan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : Plan where
  steps := (entryPlan gim).steps
    ++ ((List.replicate N_LAYERS (layerPlan gim h hm).steps).flatten
        ++ (finalPlan gim).steps)

/-- **The same twenty-two writes, against the token's tables.**

    Derived, not re-resolved.  `tokenKernels` is `layerKernels` plus three
    entries and `tokenDeclared` is `layerDeclared` plus two, so
    `stepsOf?_appendTable` carries the layer's resolution across — and the
    layer's own resolution is itself assembled from the two halves.  Restating
    this as `rfl` costs 12.1 s: every one of the twenty-two ops rescans a table
    that is now three entries longer, and the scan is where the time goes. -/
theorem layer_ops_steps (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    stepsOf? (tokenKernels gim h hm) tokenDeclared layerOps
      = some (layerPlan gim h hm).steps :=
  stepsOf?_append _ _ _ _ _ _
    (stepsOf?_appendTable _ _ _ _ attnOps _ (attn_ops_steps gim h hm))
    (stepsOf?_appendTable _ _ _ _ ffnOps _ (ffn_ops_steps gim h hm))

/-- **The loop, as an induction.**  The layer is resolved once; repeating it
    `n` times is a theorem rather than `n` more reductions — which is what
    makes a 533-step claim cost the same as a 22-step one. -/
theorem replicate_ops_steps (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : ∀ n : Nat,
    stepsOf? (tokenKernels gim h hm) tokenDeclared (List.replicate n layerOps).flatten
      = some (List.replicate n (layerPlan gim h hm).steps).flatten := by
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
      show stepsOf? _ _ (layerOps ++ (List.replicate n layerOps).flatten)
          = some ((layerPlan gim h hm).steps
                  ++ (List.replicate n (layerPlan gim h hm).steps).flatten)
      exact stepsOf?_append _ _ _ _ _ _ (layer_ops_steps gim h hm) ih


-- ---------------------------------------------------------------------------
-- …and the whole decode step, declared
-- ---------------------------------------------------------------------------

open AlgorithmLib.Host in
/-- `inferFn`'s two device writes. -/
def entryDriver (fnOf : String → FnRef) : HStmt :=
  HStmt.seqs
    [ .extern (vStep fnOf "cl_cuda_upload_ptr" [.near 16, .near 132, .opaque, .const 24])
    , .launch (kStep PTX_EMBED_OFF 3 BIND_EMBED 28 BS_EMBED) ]

open AlgorithmLib.Host in
/-- …and `inferFinalFn`'s three. -/
def finalDriver (fnOf : String → FnRef) : HStmt :=
  HStmt.seqs
    [ .launch (kStep PTX_RMS_OFF 3 BIND_RMS2 1 BS_F_NORM)
    , .extern (vStep fnOf "cl_cublas_sgemv"
        [.near 16, .const 1, .const 896, .const 151936, F1, .near 112, .near 76,
         .const 0, .near 116])
    , .launch (kStep PTX_ARGMAX_OFF 2 BIND_ARGMAX 1 BS_ARGMAX) ]

open AlgorithmLib.Host in
/-- **One layer**: the two halves, each reached by a call — which is what the
    generator does (`inferLayerFn` dispatches to `fn_29` and `fn_30`). -/
def layerDriver (fnOf : String → FnRef) : HStmt :=
  .call (.seq (.call (attnDriver fnOf)) (.call (ffnDriver fnOf)))

open AlgorithmLib.Host in
/-- **One decode step**: the prologue, the twenty-four-layer loop, the sampling
    tail.  The loop is a `forN` node, not twenty-four copies. -/
def tokenDriver (fnOf : String → FnRef) : HStmt :=
  .seq (entryDriver fnOf)
    (.seq (.forN N_LAYERS (layerDriver fnOf)) (finalDriver fnOf))

open AlgorithmLib.Host in
theorem entryDriver_deviceOps (fnOf : String → FnRef) :
    (entryDriver fnOf).deviceOps = entryOps := rfl

open AlgorithmLib.Host in
theorem finalDriver_deviceOps (fnOf : String → FnRef) :
    (finalDriver fnOf).deviceOps = finalOps := rfl

open AlgorithmLib.Host in
/-- **A layer performs `layerOps`** — the two halves concatenated, through two
    calls, which `HStmt.deviceOps_call`/`_seq` see through. -/
theorem layerDriver_deviceOps (fnOf : String → FnRef) :
    (layerDriver fnOf).deviceOps = layerOps := by
  show (HStmt.call (.seq (.call (attnDriver fnOf)) (.call (ffnDriver fnOf)))).deviceOps = _
  rw [HStmt.deviceOps_call, HStmt.deviceOps_seq, HStmt.deviceOps_call,
      HStmt.deviceOps_call, attnDriver_deviceOps, ffnDriver_deviceOps]
  rfl

open AlgorithmLib.Host in
/-- **The loop performs the layer's twenty-two writes, twenty-four times.**

    This is the theorem `tokenOps`'s definition was standing in for.  It comes
    from `HStmt.deviceOps_forN` — a structural recursion over the `forN` node —
    so the twenty-four is the program's, not a numeral someone typed into a
    list. -/
theorem loopDriver_deviceOps (fnOf : String → FnRef) :
    (HStmt.forN N_LAYERS (layerDriver fnOf)).deviceOps
      = (List.replicate N_LAYERS layerOps).flatten := by
  rw [HStmt.deviceOps_forN, layerDriver_deviceOps]

open AlgorithmLib.Host in
/-- **A whole decode step's five hundred and thirty-three device writes,
    derived from a declared program.**

    `tokenOps` is still the list the plan machinery consumes; what changed is
    that it is now *equal to* the device-write sequence of a host program whose
    loop is a node, rather than a list whose length nobody checked. -/
theorem tokenDriver_deviceOps (fnOf : String → FnRef) :
    (tokenDriver fnOf).deviceOps = tokenOps := by
  show (HStmt.seq (entryDriver fnOf)
          (.seq (.forN N_LAYERS (layerDriver fnOf)) (finalDriver fnOf))).deviceOps = _
  rw [HStmt.deviceOps_seq, HStmt.deviceOps_seq, entryDriver_deviceOps,
      finalDriver_deviceOps, loopDriver_deviceOps]
  rfl

open AlgorithmLib.Host in
/-- …and it is five hundred and thirty-three of them. -/
theorem tokenDriver_count (fnOf : String → FnRef) :
    (tokenDriver fnOf).deviceOps.length = 533 := by
  rw [tokenDriver_deviceOps]
  rfl

-- ---------------------------------------------------------------------------
-- The control flow, recovered
-- ---------------------------------------------------------------------------

/-! What `deviceOpsOf` could never see: a static scan of `inferFn` reports the
    layer's kernels **once**, because they are behind a loop and a call.  So
    `tokenOps`'s `List.replicate N_LAYERS layerOps` was the one place in the
    chain where a number came from a definition rather than from the program.

    `Clif.loopsOf` recovers the counted loops from the emitted blocks, bound
    included.  The four facts below pin the whole control-flow shape of a decode
    step, and each is decided against the built CLIF. -/

open AlgorithmLib.Clif in
/-- **The decode step contains exactly one counted loop, and it runs
    `N_LAYERS` times.**  The bound is read out of the emitted `icmp`, resolved
    in the same environment a launch's PTX slot is resolved in — not off the
    generator's source. -/
theorem infer_loop_is_layers :
    loopsOf (inferFn.run {}).2 = [⟨1, 2, 3, N_LAYERS⟩] := by native_decide

open AlgorithmLib.Clif in
/-- **…and its body dispatches to the layer function and nothing else.** -/
theorem infer_loop_body_calls :
    (blockInsts? (inferFn.run {}).2 2).map (callsIn (inferFn.run {}).2.fns)
      = some ["fn_28"] := by native_decide

open AlgorithmLib.Clif in
/-- **The sampling tail has no loop**, so its static scan is its whole
    sequence. -/
theorem final_no_loops : loopsOf (inferFinalFn.run {}).2 = [] := by native_decide

open AlgorithmLib.Host in
/-- **The declared prologue and the built one perform the same device writes.**

    Not a restatement: the left is a `HStmt` whose structure the composition
    theorems recurse through, the right is a scan of the CLIF the generator
    actually produced.  Chaining them is what turns `tokenDriver` from a
    description into a claim about this program. -/
theorem entryDriver_is_built (fnOf : String → FnRef) :
    (entryDriver fnOf).deviceOps = deviceOpsOf ROOT (inferFn.run {}).2 := by
  rw [entry_ops_are, entryDriver_deviceOps]

open AlgorithmLib.Host in
/-- **…and the sampling tail's.** -/
theorem finalDriver_is_built (fnOf : String → FnRef) :
    (finalDriver fnOf).deviceOps = deviceOpsOf ROOT (inferFinalFn.run {}).2 := by
  rw [final_ops_are, finalDriver_deviceOps]

/-- **A whole token's device-write sequence realises `tokenPlan`.** -/
theorem token_ops_realise_plan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (tokenKernels gim h hm) tokenDeclared tokenOps = some (tokenPlan gim h hm) := by
  have he : stepsOf? (tokenKernels gim h hm) tokenDeclared entryOps
      = some (entryPlan gim).steps := rfl
  have hf : stepsOf? (tokenKernels gim h hm) tokenDeclared finalOps
      = some (finalPlan gim).steps := rfl
  show (stepsOf? _ _ (entryOps ++ ((List.replicate N_LAYERS layerOps).flatten
          ++ finalOps))).map Plan.mk = _
  rw [stepsOf?_append _ _ _ _ _ _ he
        (stepsOf?_append _ _ _ _ _ _ (replicate_ops_steps gim h hm N_LAYERS) hf)]
  rfl

theorem tokenPlan_exclusive (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : (tokenPlan gim h hm).Exclusive := by
  intro S hS
  have hrep : ∀ n : Nat, PStep.proven S ∈
      (List.replicate n (layerPlan gim h hm).steps).flatten →
      PStep.proven S ∈ (layerPlan gim h hm).steps := by
    intro n; induction n with
    | zero => intro hm'; exact absurd hm' (by simp)
    | succ n ih =>
        intro hm'
        rcases List.mem_append.mp hm' with hx | hx
        · exact hx
        · exact ih hx
  rcases List.mem_append.mp hS with hx | hx
  · exact entryPlan_exclusive gim S hx
  · rcases List.mem_append.mp hx with hy | hy
    · exact layerPlan_exclusive gim h hm S (hrep N_LAYERS hy)
    · exact finalPlan_exclusive gim S hy

/-- **What one token of the shipped program does to memory.**

    Five hundred and thirty-three device writes — the embedding gather, twenty-
    four transformer layers, the LM head and the argmax — as a single equation
    between `run` and `denote`.  `Honours R` covers every declared step and
    `Law.combinerComm` softmax's remainder pass. -/
theorem token_computes (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b)))
    (R : Realisation) (hR : Honours R) (st : WSt) :
    ((tokenPlan gim h hm).run R st).mem = (tokenPlan gim h hm).denote st.mem :=
  Plan.run_denote R hR (tokenPlan gim h hm) (tokenPlan_exclusive gim h hm) st

/-! **Negative guards.**  The theorems above are equations ending in `some`, so
    they are not vacuous — but a check that cannot *fail* still proves nothing
    about what it would catch.  These pin the three ways a launch could be
    matched to a stage it is not: a right slot with the wrong bind array, a
    right array at the wrong bind offset, and a right everything on the wrong
    number of blocks.  Each must be `none`, which is what makes the positive
    results say something. -/

example (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (layerKernels gim h hm) layerDeclared
      [klOp PTX_RMS_OFF 3 BIND_RMS1 1 BS_FFN_NORM] = none := rfl

example (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (layerKernels gim h hm) layerDeclared
      [klOp PTX_RMS_OFF 3 BIND_RMS2 1 BS_A_NORM] = none := rfl

example (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (layerKernels gim h hm) layerDeclared
      [klOp PTX_ADD_OFF 2 BIND_ADD1 27 BS_A_ADD] = none := rfl

/-- …and a launch whose bind array the scan could not recover is refused
    outright, rather than matched on its slot and grid alone. -/
example (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))) :
    planOf? (layerKernels gim h hm) layerDeclared
      [({ fnName := "cl_cuda_launch", kernelOff := some (Int.ofNat PTX_RMS_OFF),
          nBufs := some 3, bindOff := some (Int.ofNat BIND_RMS1),
          gridX := some 1, blockX := some 32 }, {})] = none := rfl

-- ── Reading a value back out of the plan ───────────────────────────────────

/-!
  **The second gap, and the first step across it.**

  `layer_computes` says the plan's `run` equals its `denote`.  It does not say
  that `denote` is a transformer layer — `denote` is a fold of opaque steps,
  and asking "what is at the gate buffer at the end" was not a question the
  development could answer.

  It is now, and the route is: split the fold (`Plan.denote_append`), show the
  later steps write other buffers (`denote_frame_list`), and rewrite the one
  step that wrote it.  For a vendor GEMV that rewrite is `cublasStep_isMatvec`,
  which is where `Law.cublasIsMatvec` enters — the law that until now
  constrained a symbol no plan mentioned.

  Below is that route run to completion for one of a layer's twenty-two steps.
  The remaining twenty-one are the same shape; the batched contractions are
  where it stops, because no equation for them exists to rewrite with.
-/

/-- **The gate projection in the shipped feed-forward plan is a matrix–vector
    product of the recovered weight buffer with the normalised activation.**

    Every buffer here is `bufOf` of a handle `Clif.bindsOf` read out of the
    emitted stores: `B_WG` is the matrix the GEMV was handed, `B_XN` the vector
    RMSNorm wrote.  So this is a statement about the program, not about a
    plan-shaped object — modulo `Law.cublasIsMatvec`, which is exactly the fold
    order NVIDIA declines to specify. -/
theorem ffn_gate_is_matvec (hl : CuBlasIsMatvec) (m : Buf → Nat → Float32)
    (i : Nat) (hi : i < D_FF) :
    ffnPlan.denote m B_GATE i
      = (List.finRange D).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WG (i * D + j.val)) (ffnNormStage.step m B_XN j.val)))
          (NumOps.ofNat 0) := by
  -- the four steps after the gate GEMV all write elsewhere
  have hlater : ffnPlan.denote m B_GATE
      = (Plan.mk [PStep.proven ffnNormStage,
                  PStep.declared (cublasStep B_WG B_XN B_GATE D_FF D)]).denote m B_GATE := by
    show (Plan.mk ([PStep.proven ffnNormStage,
                    PStep.declared (cublasStep B_WG B_XN B_GATE D_FF D)]
                   ++ [PStep.declared (cublasStep B_WU B_XN B_UP D_FF D),
                       PStep.proven ffnSiluStage,
                       PStep.declared (cublasStep B_WD B_ACT B_AO D D_FF),
                       PStep.proven ffnAddStage])).denote m B_GATE = _
    rw [Plan.denote_append]
    refine denote_frame_list _ _ B_GATE ?_
    intro s hs
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl <;> exact by decide
  rw [hlater]
  show (cublasStep B_WG B_XN B_GATE D_FF D).step (ffnNormStage.step m) B_GATE i = _
  rw [cublasStep_isMatvec hl _ _ _ _ _ _ i hi]
  -- the norm stage wrote `B_XN`, so the weight buffer still reads as it did
  have hwg : ∀ a, ffnNormStage.step m B_WG a = m B_WG a := fun a =>
    StageSpec.step_otherBuf ffnNormStage m B_WG a (by decide)
  simp only [hwg]

/-- **The up projection, likewise** — and this one runs the route through *two*
    preceding steps rather than one, which is what shows it is mechanical: the
    weight and the normalised activation are carried past the gate GEMV by its
    own `frame` field, exactly as they were carried past RMSNorm. -/
theorem ffn_up_is_matvec (hl : CuBlasIsMatvec) (m : Buf → Nat → Float32)
    (i : Nat) (hi : i < D_FF) :
    ffnPlan.denote m B_UP i
      = (List.finRange D).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WU (i * D + j.val)) (ffnNormStage.step m B_XN j.val)))
          (NumOps.ofNat 0) := by
  have hlater : ffnPlan.denote m B_UP
      = (Plan.mk [PStep.proven ffnNormStage,
                  PStep.declared (cublasStep B_WG B_XN B_GATE D_FF D),
                  PStep.declared (cublasStep B_WU B_XN B_UP D_FF D)]).denote m B_UP := by
    show (Plan.mk ([PStep.proven ffnNormStage,
                    PStep.declared (cublasStep B_WG B_XN B_GATE D_FF D),
                    PStep.declared (cublasStep B_WU B_XN B_UP D_FF D)]
                   ++ [PStep.proven ffnSiluStage,
                       PStep.declared (cublasStep B_WD B_ACT B_AO D D_FF),
                       PStep.proven ffnAddStage])).denote m B_UP = _
    rw [Plan.denote_append]
    refine denote_frame_list _ _ B_UP ?_
    intro s hs
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl <;> exact by decide
  rw [hlater]
  show (cublasStep B_WU B_XN B_UP D_FF D).step
        ((cublasStep B_WG B_XN B_GATE D_FF D).step (ffnNormStage.step m)) B_UP i = _
  rw [cublasStep_isMatvec hl _ _ _ _ _ _ i hi]
  have hwu : ∀ a, ((cublasStep B_WG B_XN B_GATE D_FF D).step (ffnNormStage.step m))
      B_WU a = m B_WU a := by
    intro a
    rw [congrFun ((cublasStep B_WG B_XN B_GATE D_FF D).frame
          (ffnNormStage.step m) B_WU (by decide)) a]
    exact StageSpec.step_otherBuf ffnNormStage m B_WU a (by decide)
  have hxn : ∀ a, ((cublasStep B_WG B_XN B_GATE D_FF D).step (ffnNormStage.step m))
      B_XN a = ffnNormStage.step m B_XN a := fun a =>
    congrFun ((cublasStep B_WG B_XN B_GATE D_FF D).frame
      (ffnNormStage.step m) B_XN (by decide)) a
  simp only [hwu, hxn]

/-- The memory the down projection reads: four steps into the plan. -/
noncomputable def ffnMem3 (m : Buf → Nat → Float32) : Buf → Nat → Float32 :=
  ffnSiluStage.step ((cublasStep B_WU B_XN B_UP D_FF D).step
    ((cublasStep B_WG B_XN B_GATE D_FF D).step (ffnNormStage.step m)))

/-- **The down projection** — the third and last GEMV of the feed-forward half.

    Its weight is carried past *four* preceding steps, two proven and two
    declared, by nothing but their frame fields.  With this the whole
    feed-forward half's vendor content is read out: every `sgemv` in `ffnPlan`
    is a matrix–vector product over buffers the program itself bound. -/
theorem ffn_down_is_matvec (hl : CuBlasIsMatvec) (m : Buf → Nat → Float32)
    (i : Nat) (hi : i < D) :
    ffnPlan.denote m B_AO i
      = (List.finRange D_FF).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WD (i * D_FF + j.val)) (ffnMem3 m B_ACT j.val)))
          (NumOps.ofNat 0) := by
  have hlater : ffnPlan.denote m B_AO
      = (Plan.mk [PStep.proven ffnNormStage,
                  PStep.declared (cublasStep B_WG B_XN B_GATE D_FF D),
                  PStep.declared (cublasStep B_WU B_XN B_UP D_FF D),
                  PStep.proven ffnSiluStage,
                  PStep.declared (cublasStep B_WD B_ACT B_AO D D_FF)]).denote m B_AO := by
    show (Plan.mk ([PStep.proven ffnNormStage,
                    PStep.declared (cublasStep B_WG B_XN B_GATE D_FF D),
                    PStep.declared (cublasStep B_WU B_XN B_UP D_FF D),
                    PStep.proven ffnSiluStage,
                    PStep.declared (cublasStep B_WD B_ACT B_AO D D_FF)]
                   ++ [PStep.proven ffnAddStage])).denote m B_AO = _
    rw [Plan.denote_append]
    refine denote_frame_list _ _ B_AO ?_
    intro s hs
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl <;> exact by decide
  rw [hlater]
  show (cublasStep B_WD B_ACT B_AO D D_FF).step (ffnMem3 m) B_AO i = _
  rw [cublasStep_isMatvec hl _ _ _ _ _ _ i hi]
  have hwd : ∀ a, ffnMem3 m B_WD a = m B_WD a := by
    intro a
    show ffnSiluStage.step _ B_WD a = _
    rw [StageSpec.step_otherBuf ffnSiluStage _ B_WD a (by decide),
        congrFun ((cublasStep B_WU B_XN B_UP D_FF D).frame _ B_WD (by decide)) a,
        congrFun ((cublasStep B_WG B_XN B_GATE D_FF D).frame _ B_WD (by decide)) a]
    exact StageSpec.step_otherBuf ffnNormStage m B_WD a (by decide)
  simp only [hwd]

/-!
  ### Attention's vendor content

  The feed-forward half read out against `ffnPlan.denote` because nothing
  later overwrites a GEMV's output.  Attention's four projections do not have
  that luxury: `attn_outputs_distinct` says `B_Q` and `B_K` each get written
  three times, since the bias add and RoPE update them in place.  So the
  *final* contents of `B_Q` are not the Q projection, and a theorem phrased
  against `attnPlan.denote m B_Q` would be about RoPE, not about cuBLAS.

  The honest statement is about memory at a point in the sequence, and
  `attnMem` names those points by `take`ing a prefix of `attnPlan`'s *own*
  step list.  Nothing here can drift from the plan the program realises: the
  prefix is cut from the same object `attn_ops_realise_plan` matches against
  the recovered launches.
-/

/-- Memory `n` device writes into the shipped attention plan. -/
noncomputable def attnMem (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (n : Nat)
    (m : Buf → Nat → Float32) : Buf → Nat → Float32 :=
  (Plan.mk ((attnPlan gim h hm).steps.take n)).denote m

/-- **Where attention's sixteen device writes land, in order.**

    `attn_outputs_distinct` covered the ten proven ones; this covers the
    vendor steps too, and does it against the plan rather than against a list
    of stages, so it is the sequence the launches realise. -/
theorem attn_outs :
    ∀ (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))),
    outsOf (attnPlan gim h hm).steps
      = [B_XN, B_Q, B_K, B_V, B_Q, B_K, B_V, B_Q, B_K, B_KC, B_VC,
         B_SC, B_PR, B_AO, B_XN, B_X] := fun _ _ => rfl

theorem attn_outs_1 (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    outsOf ((attnPlan gim h hm).steps.take 1) = [B_XN] := rfl

theorem attn_outs_2 (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    outsOf ((attnPlan gim h hm).steps.take 2) = [B_XN, B_Q] := rfl

theorem attn_outs_3 (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    outsOf ((attnPlan gim h hm).steps.take 3) = [B_XN, B_Q, B_K] := rfl

theorem attn_outs_14 (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    outsOf ((attnPlan gim h hm).steps.take 14)
      = [B_XN, B_Q, B_K, B_V, B_Q, B_K, B_V, B_Q, B_K, B_KC, B_VC,
         B_SC, B_PR, B_AO] := rfl

/-- A buffer none of the first `n` steps writes still holds its input value.
    Stated over `outsOf` so the side condition is a closed list of numerals
    even though the plan is parameterised by an index map and two laws. -/
theorem attnMem_frame (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (n : Nat)
    (m : Buf → Nat → Float32) (b : Buf)
    (hb : ∀ o ∈ outsOf ((attnPlan gim h hm).steps.take n), b ≠ o) :
    ∀ a, attnMem gim h hm n m b a = m b a := fun a =>
  congrFun (denote_frame_outs _ m b hb) a

/-- **The Q projection is a matrix–vector product** of the recovered weight
    handle with what RMSNorm wrote — read out two steps into the shipped
    attention plan. -/
theorem attn_q_is_matvec (hl : CuBlasIsMatvec) (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32)
    (i : Nat) (hi : i < D) :
    attnMem gim h hm 2 m B_Q i
      = (List.finRange D).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WQ (i * D + j.val)) (attnMem gim h hm 1 m B_XN j.val)))
          (NumOps.ofNat 0) := by
  have hwq := attnMem_frame gim h hm 1 m B_WQ (by rw [attn_outs_1]; decide)
  show (cublasStep B_WQ B_XN B_Q D D).step (attnMem gim h hm 1 m) B_Q i = _
  rw [cublasStep_isMatvec hl _ _ _ _ _ _ i hi]
  simp only [hwq]

/-- **The K projection**, one step further in — the weight is carried past the
    Q GEMV by that step's own `frame`, and so is the normalised activation. -/
theorem attn_k_is_matvec (hl : CuBlasIsMatvec) (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32)
    (i : Nat) (hi : i < KV_DIM) :
    attnMem gim h hm 3 m B_K i
      = (List.finRange D).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WK (i * D + j.val)) (attnMem gim h hm 1 m B_XN j.val)))
          (NumOps.ofNat 0) := by
  have hwk := attnMem_frame gim h hm 2 m B_WK (by rw [attn_outs_2]; decide)
  have hxn : ∀ a, attnMem gim h hm 2 m B_XN a = attnMem gim h hm 1 m B_XN a := fun a =>
    congrFun ((cublasStep B_WQ B_XN B_Q D D).frame
      (attnMem gim h hm 1 m) B_XN (by decide)) a
  show (cublasStep B_WK B_XN B_K KV_DIM D).step (attnMem gim h hm 2 m) B_K i = _
  rw [cublasStep_isMatvec hl _ _ _ _ _ _ i hi]
  simp only [hwk, hxn]

/-- **The V projection**, carried past both preceding GEMVs. -/
theorem attn_v_is_matvec (hl : CuBlasIsMatvec) (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32)
    (i : Nat) (hi : i < KV_DIM) :
    attnMem gim h hm 4 m B_V i
      = (List.finRange D).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WV (i * D + j.val)) (attnMem gim h hm 1 m B_XN j.val)))
          (NumOps.ofNat 0) := by
  have hwv := attnMem_frame gim h hm 3 m B_WV (by rw [attn_outs_3]; decide)
  have hxn2 : ∀ a, attnMem gim h hm 2 m B_XN a = attnMem gim h hm 1 m B_XN a := fun a =>
    congrFun ((cublasStep B_WQ B_XN B_Q D D).frame
      (attnMem gim h hm 1 m) B_XN (by decide)) a
  have hxn : ∀ a, attnMem gim h hm 3 m B_XN a = attnMem gim h hm 1 m B_XN a := by
    intro a
    rw [show attnMem gim h hm 3 m B_XN a
          = (cublasStep B_WK B_XN B_K KV_DIM D).step (attnMem gim h hm 2 m) B_XN a from rfl,
        congrFun ((cublasStep B_WK B_XN B_K KV_DIM D).frame
          (attnMem gim h hm 2 m) B_XN (by decide)) a, hxn2]
  show (cublasStep B_WV B_XN B_V KV_DIM D).step (attnMem gim h hm 3 m) B_V i = _
  rw [cublasStep_isMatvec hl _ _ _ _ _ _ i hi]
  simp only [hwv, hxn]

/-- **The output projection** — the fourth and last `sgemv` of the attention
    half, and the only one whose result survives to the end of the plan (the
    residual add reads `B_X`, not `B_XN`).  Its weight is carried past
    *fourteen* preceding steps — ten proven kernels and two batched
    contractions among them — on frame fields alone.

    With this, every `sgemv` in a transformer layer is read out as a matrix–
    vector product over buffers the program itself bound.  What remains
    unexplained in the layer is exactly `Plan.declaredLawGap = 2`: the two
    batched contractions, for which no equation exists to state. -/
theorem attn_o_is_matvec (hl : CuBlasIsMatvec) (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32)
    (i : Nat) (hi : i < D) :
    (attnPlan gim h hm).denote m B_XN i
      = (List.finRange D).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (m B_WO (i * D + j.val)) (attnMem gim h hm 14 m B_AO j.val)))
          (NumOps.ofNat 0) := by
  have hwo := attnMem_frame gim h hm 14 m B_WO (by rw [attn_outs_14]; decide)
  -- the one step after the output projection writes the residual stream
  have hlater : (attnPlan gim h hm).denote m B_XN
      = (Plan.mk ((attnPlan gim h hm).steps.take 15)).denote m B_XN := by
    show (Plan.mk ((attnPlan gim h hm).steps.take 15
                   ++ [PStep.proven (aAddStage gim)])).denote m B_XN = _
    rw [Plan.denote_append]
    refine denote_frame_outs _ _ B_XN ?_
    rw [show outsOf [PStep.proven (aAddStage gim)] = [B_X] from rfl]
    decide
  rw [hlater]
  show (cublasStep B_WO B_AO B_XN D D).step (attnMem gim h hm 14 m) B_XN i = _
  rw [cublasStep_isMatvec hl _ _ _ _ _ _ i hi]
  simp only [hwo]

/-!
  ### One tensor, all the way through

  `attn_q_is_matvec` says what lands in `B_Q` two steps in.  Three steps write
  that buffer — the projection, the bias add and RoPE — and eleven more run
  after the last of them, so "what attention leaves in `B_Q`" is a different
  question, and the one a model spec asks.  The three theorems below answer it
  by exhibiting the whole path: what carries between the writes, and what each
  write is.
-/

/-- The whole plan is its own sixteen-step prefix. -/
theorem attnMem_full (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32) :
    attnMem gim h hm 16 m = (attnPlan gim h hm).denote m := rfl

/-- **A tensor no step in a segment writes carries its value across it.**

    The segment is *cut from* the plan with `take`/`drop` rather than written
    out again, so the proof never has to compare two `StageSpec`s — only the
    buffer each step writes is ever computed.  Spelling the steps out instead
    is the same theorem and costs minutes of elaboration. -/
theorem attnMem_carry (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (j k : Nat) (hjk : j ≤ k)
    (m : Buf → Nat → Float32) (b : Buf)
    (hb : ∀ o ∈ outsOf (((attnPlan gim h hm).steps.take k).drop j), b ≠ o) :
    attnMem gim h hm k m b = attnMem gim h hm j m b := by
  have hmin : (attnPlan gim h hm).steps.take j
      = ((attnPlan gim h hm).steps.take k).take j := by
    rw [List.take_take, Nat.min_eq_left hjk]
  have hsplit : (attnPlan gim h hm).steps.take k
      = (attnPlan gim h hm).steps.take j ++ ((attnPlan gim h hm).steps.take k).drop j := by
    rw [hmin]; exact (List.take_append_drop j _).symm
  show (Plan.mk ((attnPlan gim h hm).steps.take k)).denote m b = _
  rw [hsplit, Plan.denote_append]
  exact denote_frame_outs _ _ b hb

theorem attn_seg_2_4 :
    ∀ (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))),
    outsOf (((attnPlan gim h hm).steps.take 4).drop 2) = [B_K, B_V] := fun _ _ => rfl

theorem attn_seg_5_7 :
    ∀ (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))),
    outsOf (((attnPlan gim h hm).steps.take 7).drop 5) = [B_K, B_V] := fun _ _ => rfl

theorem attn_seg_8_16 :
    ∀ (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))),
    outsOf (((attnPlan gim h hm).steps.take 16).drop 8)
      = [B_K, B_KC, B_VC, B_SC, B_PR, B_AO, B_XN, B_X] := fun _ _ => rfl

theorem attn_q_carried_2_4 (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32) :
    attnMem gim h hm 4 m B_Q = attnMem gim h hm 2 m B_Q :=
  attnMem_carry gim h hm 2 4 (by omega) m B_Q (by rw [attn_seg_2_4]; decide)

theorem attn_q_carried_5_7 (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32) :
    attnMem gim h hm 7 m B_Q = attnMem gim h hm 5 m B_Q :=
  attnMem_carry gim h hm 5 7 (by omega) m B_Q (by rw [attn_seg_5_7]; decide)

theorem attn_q_settled (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32) :
    (attnPlan gim h hm).denote m B_Q = attnMem gim h hm 8 m B_Q := by
  rw [← attnMem_full gim h hm m]
  exact attnMem_carry gim h hm 8 16 (by omega) m B_Q (by rw [attn_seg_8_16]; decide)

/-- **The Q path of attention, end to end.**

    Read bottom-up: the projection GEMV writes `B_Q` (`attn_q_is_matvec`);
    the K and V projections carry it (first conjunct); the bias add is the
    second write (second conjunct); the K and V bias adds carry it; RoPE is the
    third and last (third conjunct); and the eleven steps after RoPE leave it
    alone, so what the plan *ends* with is what RoPE wrote.

    Every one of those eleven is checked, not waved at — including both batched
    contractions, which read `B_Q` but do not write it.  That is the difference
    between "the GEMV computes a matrix-vector product" and "this is the query
    tensor attention produced". -/
theorem attn_q_path (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32) :
    (attnPlan gim h hm).denote m B_Q
        = (aRopeQStage gim).step (attnMem gim h hm 7 m) B_Q
      ∧ attnMem gim h hm 7 m B_Q = (aBiasQStage gim).step (attnMem gim h hm 4 m) B_Q
      ∧ attnMem gim h hm 4 m B_Q = attnMem gim h hm 2 m B_Q := by
  refine ⟨?_, ?_, attn_q_carried_2_4 gim h hm m⟩
  · exact attn_q_settled gim h hm m
  · exact attn_q_carried_5_7 gim h hm m

/-!
  The K and V paths, by the same route.  `B_K` is written three times (the
  projection, its bias, RoPE) and `B_V` twice; nothing after the last write of
  each touches it, and every intervening step is checked rather than assumed.
-/

theorem attn_seg_3_5 :
    ∀ (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))),
    outsOf (((attnPlan gim h hm).steps.take 5).drop 3) = [B_V, B_Q] := fun _ _ => rfl

theorem attn_seg_4_6 :
    ∀ (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))),
    outsOf (((attnPlan gim h hm).steps.take 6).drop 4) = [B_Q, B_K] := fun _ _ => rfl

theorem attn_seg_6_8 :
    ∀ (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))),
    outsOf (((attnPlan gim h hm).steps.take 8).drop 6) = [B_V, B_Q] := fun _ _ => rfl

theorem attn_seg_7_16 :
    ∀ (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))),
    outsOf (((attnPlan gim h hm).steps.take 16).drop 7)
      = [B_Q, B_K, B_KC, B_VC, B_SC, B_PR, B_AO, B_XN, B_X] := fun _ _ => rfl

theorem attn_seg_9_16 :
    ∀ (h : AllHold [Law.combinerComm]) (hm : SmMeta (fun b => gim (bSoft b))),
    outsOf (((attnPlan gim h hm).steps.take 16).drop 9)
      = [B_KC, B_VC, B_SC, B_PR, B_AO, B_XN, B_X] := fun _ _ => rfl

/-- **The K path of attention, end to end** — projection, bias, RoPE, and
    nothing after. -/
theorem attn_k_path (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32) :
    (attnPlan gim h hm).denote m B_K
        = (aRopeKStage gim).step (attnMem gim h hm 8 m) B_K
      ∧ attnMem gim h hm 8 m B_K = (aBiasKStage gim).step (attnMem gim h hm 5 m) B_K
      ∧ attnMem gim h hm 5 m B_K = attnMem gim h hm 3 m B_K := by
  refine ⟨?_, ?_, attnMem_carry gim h hm 3 5 (by omega) m B_K (by rw [attn_seg_3_5]; decide)⟩
  · rw [← attnMem_full gim h hm m]
    exact attnMem_carry gim h hm 9 16 (by omega) m B_K (by rw [attn_seg_9_16]; decide)
  · exact attnMem_carry gim h hm 6 8 (by omega) m B_K (by rw [attn_seg_6_8]; decide)

/-- **The V path** — projection then bias, and nothing after; `B_V` is the one
    of the three projections RoPE does not touch. -/
theorem attn_v_path (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (m : Buf → Nat → Float32) :
    (attnPlan gim h hm).denote m B_V
        = (aBiasVStage gim).step (attnMem gim h hm 6 m) B_V
      ∧ attnMem gim h hm 6 m B_V = attnMem gim h hm 4 m B_V := by
  refine ⟨?_, attnMem_carry gim h hm 4 6 (by omega) m B_V (by rw [attn_seg_4_6]; decide)⟩
  rw [← attnMem_full gim h hm m]
  exact attnMem_carry gim h hm 7 16 (by omega) m B_V (by rw [attn_seg_7_16]; decide)

end Realisation

end Qwen2Common
