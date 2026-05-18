import Lean
import Std
import AlgorithmLib
import AlgorithmLib.Cuda

set_option maxRecDepth 4096

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.PTX
open AlgorithmLib.Tensor

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
def slotMeta     : BufferSlot [.sta 2]              := slotOfAt 0x0084

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
def PTX_ROPE_Q_OFF  : Nat := 0x4000
def PTX_ROPE_K_OFF  : Nat := 0x4800
def PTX_SOFTMAX_OFF : Nat := 0x5000
def PTX_SILU_OFF    : Nat := 0x6800
def PTX_ADD_OFF     : Nat := 0x7400
def PTX_KVSTORE_OFF : Nat := 0x8000
def PTX_ARGMAX_OFF  : Nat := 0x9000

-- Tokenize / server buffers (extend MEM_SIZE to 64 KB):
def TOKEN_BUF_OFF   : Nat := 0xA000  -- 2048 × u32 = 8192 bytes
def TEXT_IN_OFF     : Nat := 0xC000  -- 8 KB text input buffer
def TEXT_OUT_OFF    : Nat := 0xE000  -- 8 KB text output buffer
def MEM_SIZE        : Nat := 0x10000 -- 64 KB total

-- ── PTX Kernels ──────────────────────────────────────────────────────────────

def D_AS_BITS : UInt32 := 0x44600000  -- 896.0f

-- ── Tensor shape abbreviations for Qwen2 ────────────────────────────────────
abbrev VecD     := Tensor [.sta D]              -- hidden state, rmsnorm weights, residual
abbrev VecKV    := Tensor [.sta KV_DIM]         -- current K/V vector for one position
abbrev VecDff   := Tensor [.sta D_FF]           -- FFN intermediate
abbrev VecVocab := Tensor [.sta VOCAB]          -- logits, embed table row
abbrev VecMeta  := Tensor [.sta 2]              -- meta buffer ([token_id, pos])
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
    { name := "meta_buf",  shape := [.sta 2],            ro := true },
    { name := "out_buf",   shape := [.sta D] }
  ]
  body := do
    let embedBuf ← ldParam "embed_buf"
    let metaBuf  ← ldParam "meta_buf"
    let outBuf   ← ldParam "out_buf"
    let tokId ← freshR; ldGlobalU tokId metaBuf
    let tokId64 ← freshRd; cvtU64 tokId64 tokId
    let rowBytes ← freshRd; mulWideRI rowBytes tokId D_BYTES
    let rowBase ← freshRd; addRd rowBase embedBuf rowBytes
    let nReg ← freshR; movRC nReg D
    let (tid, _, _) ← getWarpIds
    strideLoop tid nReg 256 "el_loop" "el_done" fun i => do
      let srcA ← elemAddr rowBase i; let dstA ← elemAddr outBuf i
      let xi ← freshF; ldGlobalF xi srcA; stGlobalF dstA xi
    ptxRet
  geom := Kernel.Geom.static 1 1 1 256 1 1
  ptxOff := PTX_EMBED_OFF
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
  body := do
    let xPtr ← ldParam "x_buf"; let wPtr ← ldParam "w_buf"; let yPtr ← ldParam "y_buf"
    let nReg ← freshR; movRC nReg D
    let (tid, wid, lid) ← getWarpIds
    rmsNormBody xPtr wPtr yPtr tid wid lid nReg D_AS_BITS ""
  geom := Kernel.Geom.static 1 1 1 256 1 1
  ptxOff := PTX_RMS_OFF
}

def ptxRmsNorm : String := rmsNormKernel.ptxSource

/-- Typed RMSNorm launcher.  `x` is the input, `w` the weights, `y` the output —
    all `[D]` f32. The bind region is the only call-site-specific value. -/
private def launchRms (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (x w y : Tensor [.sta D]) : IRBuilder Unit :=
  launch3 rmsNormKernel cuda ptr bindOff x w y

-- Bias add: x[i] += b[i]; same kernel body parameterized by N at PTX level.
-- We emit two specializations (D and KV_DIM) since N is folded as a constant.
private def biasAddBody (n : Nat) (loopL doneL : String) : PTX Unit := do
  let xBuf ← ldParam "x_buf"; let bBuf ← ldParam "b_buf"
  let nReg ← freshR; movRC nReg n
  let (tid, _, _) ← getWarpIds
  strideLoop tid nReg 256 loopL doneL fun i => do
    let xA ← elemAddr xBuf i; let bA ← elemAddr bBuf i
    let xi ← freshF; ldGlobalF xi xA; let bi ← freshF; ldGlobalF bi bA
    addF xi xi bi; stGlobalF xA xi
  ptxRet

def biasAddDKernel : Kernel := {
  name := "main"
  params := [{ name := "x_buf", shape := [.sta D] },
             { name := "b_buf", shape := [.sta D], ro := true }]
  body := biasAddBody D "bd_loop" "bd_done"
  geom := Kernel.Geom.static 1 1 1 256 1 1
  ptxOff := PTX_BIAS_D_OFF
}

def biasAddKVKernel : Kernel := {
  name := "main"
  params := [{ name := "x_buf", shape := [.sta KV_DIM] },
             { name := "b_buf", shape := [.sta KV_DIM], ro := true }]
  body := biasAddBody KV_DIM "bk_loop" "bk_done"
  geom := Kernel.Geom.static 1 1 1 256 1 1
  ptxOff := PTX_BIAS_KV_OFF
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
private def ropeBody (vecParam : String) : PTX Unit := do
  let vBuf      ← ldParam vecParam
  let metaBuf   ← ldParam "meta_buf"
  let ropeTable ← ldParam "rope_table"
  let headIdx ← freshR; movR headIdx ctaX
  let freqIdx ← freshR; movR freqIdx tidX
  let pos ← freshR; ldGlobalUO pos metaBuf 4
  let tblIdx ← freshR; madLoRC tblIdx pos (HEAD_DIM / 2) freqIdx
  let tblOff64 ← freshRd; mulWideRI tblOff64 tblIdx 4
  let sinAddr ← freshRd; addRd sinAddr ropeTable tblOff64
  let sinT ← freshF; ldGlobalF sinT sinAddr
  let cosBaseOff ← freshR; movRC cosBaseOff (MAX_SEQ * (HEAD_DIM / 2) * 4)
  let cosBaseOff64 ← freshRd; cvtU64 cosBaseOff64 cosBaseOff
  let cosAddr ← freshRd; addRd cosAddr ropeTable cosBaseOff64
  addRd cosAddr cosAddr tblOff64
  let cosT ← freshF; ldGlobalF cosT cosAddr
  let loOff ← freshR; madLoRC loOff headIdx HEAD_DIM freqIdx
  let hiOff ← freshR; addRI hiOff loOff (HEAD_DIM / 2)
  let loOff64 ← freshRd; mulWideRI loOff64 loOff 4; let loAddr ← freshRd; addRd loAddr vBuf loOff64
  let hiOff64 ← freshRd; mulWideRI hiOff64 hiOff 4; let hiAddr ← freshRd; addRd hiAddr vBuf hiOff64
  let vLo ← freshF; ldGlobalF vLo loAddr
  let vHi ← freshF; ldGlobalF vHi hiAddr
  let hiSin ← freshF; mulF hiSin vHi sinT
  let negHiSin ← freshF; negF negHiSin hiSin
  let newLo ← freshF; fmaRn newLo vLo cosT negHiSin
  let hiCos ← freshF; mulF hiCos vHi cosT
  let newHi ← freshF; fmaRn newHi vLo sinT hiCos
  stGlobalF loAddr newLo
  stGlobalF hiAddr newHi
  ptxRet

-- RoPE Q: grid=N_Q, block=HEAD_DIM/2. Rotates Q in place.
def ropeQKernel : Kernel := {
  name := "main"
  params := [{ name := "q_buf",      shape := [.sta D] },
             { name := "meta_buf",   shape := [.sta 2], ro := true },
             { name := "rope_table", shape := [.sta 2, .sta MAX_SEQ, .sta (HEAD_DIM/2)], ro := true }]
  body := ropeBody "q_buf"
  geom := Kernel.Geom.static N_Q 1 1 (HEAD_DIM/2) 1 1
  ptxOff := PTX_ROPE_Q_OFF
}

-- RoPE K: grid=N_KV, block=HEAD_DIM/2.
def ropeKKernel : Kernel := {
  name := "main"
  params := [{ name := "k_buf",      shape := [.sta KV_DIM] },
             { name := "meta_buf",   shape := [.sta 2], ro := true },
             { name := "rope_table", shape := [.sta 2, .sta MAX_SEQ, .sta (HEAD_DIM/2)], ro := true }]
  body := ropeBody "k_buf"
  geom := Kernel.Geom.static N_KV 1 1 (HEAD_DIM/2) 1 1
  ptxOff := PTX_ROPE_K_OFF
}

def ptxRoPEQ : String := ropeQKernel.ptxSource
def ptxRoPEK : String := ropeKKernel.ptxSource

private def launchRopeQ (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (q : VecD) (mb : VecMeta) (rope : RopeTbl) : IRBuilder Unit :=
  launch3 ropeQKernel cuda ptr bindOff q mb rope

private def launchRopeK (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (k : VecKV) (mb : VecMeta) (rope : RopeTbl) : IRBuilder Unit :=
  launch3 ropeKKernel cuda ptr bindOff k mb rope

-- Softmax over per-head scores. seq_len = meta_buf[4]+1 (dynamic).
-- Grid=N_Q, Block=256, smem=40.
def softmaxKernel : Kernel := {
  name := "main"
  params := [{ name := "scores_buf", shape := [.sta N_Q, .dyn] },
             { name := "meta_buf",   shape := [.sta 2], ro := true },
             { name := "probs_buf",  shape := [.sta N_Q, .dyn] }]
  smemBytes := 40
  body := do
    let scoresBuf ← ldParam "scores_buf"
    let metaPtr   ← ldParam "meta_buf"
    let probsBuf  ← ldParam "probs_buf"
    let seqLen ← freshR; ldGlobalUO seqLen metaPtr 4; addRI seqLen seqLen 1
    let (tid, wid, lid) ← getWarpIds
    let headIdx  ← freshR; movR headIdx ctaX
    let headId64 ← freshRd; cvtU64 headId64 headIdx
    let seqLen64 ← freshRd; cvtU64 seqLen64 seqLen
    let headOff  ← freshRd; mulLoRd headOff headId64 seqLen64
    let byteOff  ← freshRd; shlRd byteOff headOff 2
    let scoresBase ← freshRd; addRd scoresBase scoresBuf byteOff
    let probsBase  ← freshRd; addRd probsBase probsBuf byteOff
    let log2e ← freshF; movFC log2e f32_log2e
    let lMax ← freshF; movFC lMax 0xFF800000; let mTmp ← freshF
    strideLoop tid seqLen 256 "sm_lmax" "sm_dmax" fun i => do
      let addr ← elemAddr scoresBase i; ldGlobalF mTmp addr; maxF lMax lMax mTmp
    warpReduceMax lMax mTmp
    lane0WriteSmem lid wid "sm_skip1" fun wAddr => stSharedFD wAddr lMax
    thread0Op tid "sm_skip2" do
      let sBase ← smemBase; let gMax ← freshF
      crossWarp8 gMax mTmp sBase 0 maxF; stSharedF sBase 32 gMax
    let sBase1 ← smemBase; let gMax ← freshF; ldSharedF gMax sBase1 32
    let lSum ← freshF; movFC lSum f32_0; let sTmp ← freshF
    strideLoop tid seqLen 256 "sm_lsum" "sm_dsum" fun i => do
      let addr ← elemAddr scoresBase i; let xi ← freshF; ldGlobalF xi addr
      subF xi xi gMax; mulF xi xi log2e; ex2 xi xi; addF lSum lSum xi
    warpReduceSum lSum sTmp
    lane0WriteSmem lid wid "sm_skip3" fun wAddr => stSharedFD wAddr lSum
    thread0Op tid "sm_skip4" do
      let sBase ← smemBase
      crossWarp8 lSum sTmp sBase 0 addF; rcp lSum lSum; stSharedF sBase 36 lSum
    let sBase2 ← smemBase; let invSum ← freshF; ldSharedF invSum sBase2 36
    strideLoop tid seqLen 256 "sm_lout" "sm_dout" fun i => do
      let sAddr ← elemAddr scoresBase i; let pAddr ← elemAddr probsBase i
      let xi ← freshF; ldGlobalF xi sAddr
      subF xi xi gMax; mulF xi xi log2e; ex2 xi xi; mulF xi xi invSum
      stGlobalF pAddr xi
    ptxRet
  geom := Kernel.Geom.static N_Q 1 1 256 1 1
  ptxOff := PTX_SOFTMAX_OFF
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
  body := do
    let gatePtr ← ldParam "gate_buf"; let upPtr ← ldParam "up_buf"; let outPtr ← ldParam "out_buf"
    let nReg ← freshR; movRC nReg D_FF
    let (gid, _) ← gridStrideSetup nReg "sg_done"
    let gA ← elemAddr gatePtr gid; let uA ← elemAddr upPtr gid; let oA ← elemAddr outPtr gid
    let g ← freshF; ldGlobalF g gA; let u ← freshF; ldGlobalF u uA
    let ng ← freshF; negF ng g
    let l ← freshF; movFC l f32_log2e; mulF ng ng l; ex2 ng ng
    let one ← freshF; movFC one f32_1; addF ng ng one; rcp ng ng
    mulF g g ng; mulF g g u; stGlobalF oA g
    label "sg_done"; ptxRet
  geom := Kernel.Geom.static ((D_FF + 255) / 256) 1 1 256 1 1
  ptxOff := PTX_SILU_OFF
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
  body := do
    let xPtr ← ldParam "x_buf"; let addPtr ← ldParam "add_buf"
    let nReg ← freshR; movRC nReg D
    let (gid, _) ← gridStrideSetup nReg "ra_done"
    let xA ← elemAddr xPtr gid; let aA ← elemAddr addPtr gid
    let xi ← freshF; ldGlobalF xi xA; let ai ← freshF; ldGlobalF ai aA
    addF xi xi ai; stGlobalF xA xi
    label "ra_done"; ptxRet
  geom := Kernel.Geom.static ((D + 255) / 256) 1 1 256 1 1
  ptxOff := PTX_ADD_OFF
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
             { name := "meta_buf",    shape := [.sta 2], ro := true }]
  body := do
    let kCurBuf   ← ldParam "k_cur_buf"
    let kCacheBuf ← ldParam "k_cache_buf"
    let metaBuf   ← ldParam "meta_buf"
    let kvHead ← freshR; movR kvHead ctaX
    let elemIdx ← freshR; movR elemIdx tidX
    let pos ← freshR; ldGlobalUO pos metaBuf 4
    -- src offset: kvHead * HEAD_DIM + elemIdx (within [N_KV, HEAD_DIM] k_cur)
    let srcOff ← freshR; madLoRC srcOff kvHead HEAD_DIM elemIdx
    let srcOff64 ← freshRd; mulWideRI srcOff64 srcOff 4
    let srcAddr ← freshRd; addRd srcAddr kCurBuf srcOff64
    let kVal ← freshF; ldGlobalF kVal srcAddr
    -- dst index: kvHead * MAX_SEQ * HEAD_DIM + pos * HEAD_DIM + elemIdx
    let kvH64  ← freshRd; mulWideRI kvH64  kvHead  (MAX_SEQ * HEAD_DIM)
    let pos64  ← freshRd; mulWideRI pos64  pos     HEAD_DIM
    let elem64 ← freshRd; cvtU64    elem64 elemIdx
    addRd kvH64 kvH64 pos64
    addRd kvH64 kvH64 elem64
    let byteOff ← freshRd; shlRd byteOff kvH64 2
    let dstAddr ← freshRd; addRd dstAddr kCacheBuf byteOff
    stGlobalF dstAddr kVal
    ptxRet
  geom := Kernel.Geom.static N_KV 1 1 HEAD_DIM 1 1
  ptxOff := PTX_KVSTORE_OFF
}

def ptxKVStore : String := kvStoreKernel.ptxSource

private def launchKVStore (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (kCur : VecKV) (kCache : KVCache) (mb : VecMeta) : IRBuilder Unit :=
  launch3 kvStoreKernel cuda ptr bindOff kCur kCache mb

-- Argmax over VOCAB logits. Single-thread; writes result to meta_buf[0].
def argmaxKernel : Kernel := {
  name := "main"
  params := [{ name := "logits_buf", shape := [.sta VOCAB], ro := true },
             { name := "meta_buf",   shape := [.sta 2] }]
  body := do
    let logitsBuf ← ldParam "logits_buf"
    let metaBuf   ← ldParam "meta_buf"
    let maxVal ← freshF; movFC maxVal 0xFF800000
    let maxIdx ← freshR; movRC maxIdx 0
    let i ← freshR; movRC i 0
    label "ax_loop"
    let p1 ← freshP; setpGeI p1 i VOCAB; braIf p1 "ax_done"
    let i64 ← freshRd; cvtU64 i64 i
    let byteOff ← freshRd; shlRd byteOff i64 2
    let addr ← freshRd; addRd addr logitsBuf byteOff
    let xi ← freshF; ldGlobalF xi addr
    let p2 ← freshP; setpGtF p2 xi maxVal
    braIfNot p2 "ax_no_upd"
    movF maxVal xi; movR maxIdx i
    label "ax_no_upd"
    addRI i i 1; bra "ax_loop"
    label "ax_done"
    stGlobalU32 metaBuf maxIdx
    ptxRet
  geom := Kernel.Geom.static 1 1 1 1 1 1
  ptxOff := PTX_ARGMAX_OFF
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
  let metaBytes   ← iconst64 8

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

  -- Upload [token_id, pos] (8 bytes) to GPU meta buffer
  let metaT ← slotMeta.load ptr
  let eight ← iconst64 8
  Tensor.upload cuda ptr metaT dataPtr eight

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
  let eight64 ← iconst64 8
  launchRms cuda ptr BIND_RMS2 bufHidden bufRmsFinal bufHdNorm
  -- LM head projection: lmHead:[VOCAB,D]·hdNorm:[D] → logits:[VOCAB]
  CuBlas.linear blas ptr bufLmHead bufHdNorm bufLogits
  launchArgmax cuda ptr BIND_ARGMAX bufLogits bufMeta
  let _ ← cudaSync cuda ptr 0x10
  Tensor.download cuda ptr bufMeta outPtr eight64
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
  let maxRecv   ← iconst64 8192
  let maxDecode ← iconst64 128
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
  let prefixLen ← iconst64 6
  let wrapLen   ← iconst64 19
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

end Qwen2Common
