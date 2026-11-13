module
public import Lean
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
public import Std
public import Scan.Layout
meta import Scan.Layout
public import Scan.Ship
meta import Scan.Ship
public import GptOss.Kernels
meta import GptOss.Kernels
public import GptOss.Prefill
meta import GptOss.Prefill
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import Tokenizer.Pretok
meta import Tokenizer.Pretok
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML
open AlgorithmLib.Prog
open GptOssAttention hiding H

/-!
  # gpt-oss-20b: the whole model, and the cache that makes it fit

  Twenty-four layers, and the thing the single-layer slice deliberately did
  without: the experts are no longer resident. There are seven hundred and
  sixty-eight of them and they weigh 9.48 GiB against a card with twelve, so
  what the device holds is a fixed number of **slots**, and which expert lives
  in which slot changes as the router asks for them.

  ## Where the weights are

  The whole expert file is pinned in host memory once, at start-up. Pinned
  because the miss path is a DMA and a pageable source halves it; once, because
  9.48 GiB is not something to move twice. A miss is then six copies of a few
  megabytes each, from a host pointer this program computes to a device buffer
  it already owns — `cl_cuda_pinned_ptr_at` bounds-checks the first and
  `cudaUploadRaw` performs the second.

  Everything else lives on the device: the dense weights of all twenty-four
  layers, `lm_head`, and the key/value caches. The embedding table does not —
  a row is 5.7 KiB and reading it from the file per token costs less than the
  gigabyte of device memory holding all of them would.

  ## Why the loop is here and not above

  A decode step is twenty-four layers and each is about twenty launches, so
  unrolling would be five hundred launches in one body. It is a `forLoop`
  around one layer instead, and the layer reads which weights to use from a
  slot table the loop rewrites — the same six-integer-move trick the experts
  use, which is the whole reason that trick was worth building.

  Two kernels differ between layer types, because half the layers slide and
  their key cache is a 128-entry ring where the others hold 8192. The launch
  picks between them arithmetically: `ptxOff` is a register, so the slot a
  layer launches from is a function of its index and needs no branch.

  ## What the caller sees

  One entry, no extras, and two modes. A single step, for a caller that wants
  to drive positions itself and check logits against the model; or a whole
  chat turn — text in, the reply's text out, with the tokenizer, the split,
  the prefill, the generation loop and the detokenizer all on this side of the
  boundary.

  The second is the point. A caller that has to tokenize for itself is a caller
  that has to agree with the model about what a token is, and that agreement is
  exactly what a separate tokenizer cannot be held to.
-/

namespace GptOssDecode

open GptOssKernels (warpsPerCta rowsPerWarpDown rowsPerWarpGateUp smemBytes)

/-! ## Geometry, all of it derived -/

def NL : Nat := 24
def NE : Nat := 32
def TOPK : Nat := 4
def VOCAB : Nat := 201088
def HH : Nat := 2880
def II : Nat := 2880

/-- Slots the cache holds. A compile-time *bound*, one for every expert the
    model has: the bind table is sized for this many and the loader creates
    however many the card turns out to have room for, which it asks at run
    time. -/
def NSLOT_MAX : Nat := NL * NE

/-- What a caller who names no context gets: the depth this engine shipped with
    before the cache became a choice, so a driver that says nothing behaves as
    it always did. -/
def CAP_DEFAULT : Nat := 8192

/-- One expert, all its pieces: a row of `experts.bin` and a slot of the
    device pool alike (`GptOssKernels.expertPieces`). -/
def ROW_BYTES : Nat := GptOssKernels.EXPERT_BYTES

/-- The dense file's layout, which is regular: a fixed stride per layer and a
    fixed offset per kind inside it. Derived rather than tabulated, so a
    converter change that moved a tensor would move this too. -/
def dQkvOut : Nat := 64 * 64 + 2 * 8 * 64
def dKindBytes : List Nat :=
  [ HH * 4, dQkvOut * HH * 2, dQkvOut * 4, 64 * 4
  , HH * (64 * 64) * 2, HH * 4, HH * 4, NE * HH * 4, NE * 4 ]
def dKindOff (k : Nat) : Nat := ((dKindBytes.take k).foldl (· + ·) 0)
def LAYER_STRIDE : Nat := dKindBytes.foldl (· + ·) 0
def denseOff (layer k : Nat) : Nat := LAYER_STRIDE * layer + dKindOff k
/-- Offsets into `dense.bin`, not addresses in this program's memory — named
    so, because the layout scan reads every `…_OFF` as the latter and would be
    right to. -/
def fileFinalNorm : Nat := LAYER_STRIDE * NL
def fileLmHead : Nat := fileFinalNorm + HH * 4
def fileRope : Nat := fileLmHead + VOCAB * HH * 2
/-- **Rows in the file's rotation tables, which is not `ROPE_N`.**

    The converter writes the whole published table — one row per position the
    checkpoint declares, all 131072 of them — cosines first and then sines.
    This program serves `ROPE_N` positions, so the two slices it wants are the
    first `ROPE_N` rows of each, and they sit `ROPE_FILE_ROWS * HALF * 4` apart
    rather than adjacent.

    Both halves of that matter, and neither is visible from a size check: take
    the tables as one contiguous run and what lands is cosines twice over, at
    which point the rotation at position 0 is not the identity and every query
    and key in the model is turned. -/
def ROPE_FILE_ROWS : Nat := 131072
def fileRopeSin : Nat := fileRope + ROPE_FILE_ROWS * HALF * 4

/-- **The Q8_0 section, after the rotation tables**, which is what the decode
    multiplies by: llama.cpp's own values for the fused QKV, `o_proj` and
    `lm_head` (`prepare_q8.py` appends them, checked against the bf16 copies
    above, which the reference reads). A tensor is its `rows × cols` int8
    values and then its f32 scales, one a block of 32. -/
def q8Bytes (rows cols : Nat) : Nat := rows * cols + rows * (cols / 32) * 4
def fileQ8 : Nat := fileRopeSin + ROPE_FILE_ROWS * HALF * 4
def Q8_LAYER : Nat := q8Bytes dQkvOut HH + q8Bytes HH QO
def fileQ8Lm : Nat := fileQ8 + Q8_LAYER * NL
/-- Where a dense kind of a layer comes from: the Q8_0 section for the two
    projections, the bf16 layout for the rest. -/
def fileDense (layer k : Nat) : Nat :=
  if k == 1 then fileQ8 + Q8_LAYER * layer
  else if k == 4 then fileQ8 + Q8_LAYER * layer + q8Bytes dQkvOut HH
  else denseOff layer k
/-- A dense kind's size on the device. -/
def dDevBytes : List Nat :=
  (dKindBytes.set 1 (q8Bytes dQkvOut HH)).set 4 (q8Bytes HH QO)

/-! ## PTX slots -/

/-- Where the loop parks the layer it is on: inside the meta region, past the
    slots the kernels read, so it costs no separate mapping. -/
def DLAYER_SLOT : Nat := 32


def dPtx : List String :=
  [ GptOssKernels.Body.moduleOf rmsKernel, GptOssKernels.Body.moduleOf addKernel, GptOssKernels.Body.moduleOf (ropeKernel 0)
  , GptOssKernels.Body.moduleOf (ropeKernel QO)
  , GptOssKernels.Body.moduleOf (kvStoreKernel (QO / 2) HDW), GptOssKernels.Body.moduleOf (kvStoreKernel (KO / 2) HDW)
  , GptOssKernels.Body.moduleOf sinkKernel, GptOssKernels.Body.moduleOf GptOssKernels.Body.narrow
  , GptOssKernels.Body.moduleOf (GptOssKernels.Body.gateUpAllK DLAYER_SLOT)
  , GptOssKernels.Body.moduleOf (GptOssKernels.Body.downAllK DLAYER_SLOT)
  , GptOssKernels.Body.moduleOf GptOssKernels.Body.moeCombineAll
  , GptOssKernels.Body.moduleOf GptOssKernels.Body.routerTop4
  , GptOssKernels.Body.moduleOf GptOssKernels.Body.argmax
  , GptOssKernels.Body.moduleOf GptOssKernels.Body.widen
  , GptOssKernels.Body.moduleOf GptOssKernels.Body.sample
  , GptOssKernels.Body.moduleOf GptOssKernels.Body.attnTile
  , GptOssKernels.Body.moduleOf GptOssKernels.Body.attnCombine
  , GptOssKernels.Body.moduleOf (GptOssKernels.Body.q8Gemv dQkvOut HH)
  , GptOssKernels.Body.moduleOf (GptOssKernels.Body.q8Gemv HH QO)
  , GptOssKernels.Body.moduleOf (GptOssKernels.Body.q8Gemv VOCAB HH)
  -- a prompt's kernels, many tokens at once (`GptOssPrefill`)
  , GptOssKernels.Body.moduleOf GptOssPrefill.rmsT, GptOssKernels.Body.moduleOf GptOssPrefill.quantRows
  , GptOssKernels.Body.moduleOf (GptOssPrefill.q8Gemm dQkvOut HH), GptOssKernels.Body.moduleOf GptOssPrefill.ropeT
  , GptOssKernels.Body.moduleOf GptOssPrefill.kvNarrow, GptOssKernels.Body.moduleOf GptOssPrefill.attnT
  , GptOssKernels.Body.moduleOf GptOssPrefill.kvStoreT, GptOssKernels.Body.moduleOf (GptOssPrefill.q8Gemm HH QO)
  , GptOssKernels.Body.moduleOf GptOssPrefill.addRows, GptOssKernels.Body.moduleOf GptOssPrefill.routerLogits
  , GptOssKernels.Body.moduleOf GptOssPrefill.routerTopT, GptOssKernels.Body.moduleOf GptOssPrefill.group
  , GptOssKernels.Body.moduleOf GptOssPrefill.moeGateUp, GptOssKernels.Body.moduleOf GptOssPrefill.moeDown
  , GptOssKernels.Body.moduleOf GptOssPrefill.combine ]

def S_RMS := 0
def S_ADD := 1
def S_ROPEQ := 2
def S_ROPEK := 3
/-- Keys then values.  One pair, not two: the cache depth left the kernel and
    became `M_KVSTRIDE`, so a sliding layer and a full one launch the same
    program. -/
def S_KV := 4
def S_NARROW := 7
def S_GATEUP := 8
def S_DOWN := 9
def S_COMBINE := 10
def S_TOP4 := 11
def S_ARGMAX := 12
def S_WIDEN := 13
/-- A step picks between greedy and sampling by arithmetic on the slot, because
    `ptxOff` is a register and a branch here would be a branch around a launch.
    The distance is `S_SAMPLE - S_ARGMAX` and is computed, not assumed to be
    one: `S_WIDEN` sits between them. -/
def S_SAMPLE := 14
/-- Fused attention: one tile of the sequence a warp, then the merge.

    Appended rather than placed beside the softmax slot, whose four-kernel
    path they replace, because a slot index is a position in `dPtx` and
    inserting one renumbers every launch. -/
def S_ATTTILE := 15
def S_ATTCOMB := 16
/-- The Q8_0 projections: QKV, `o_proj`, `lm_head`. -/
def S_Q8QKV := 17
def S_Q8O := 18
def S_Q8LM := 19
/-- A prompt's kernels (`dPrefillChunkM`), in the order a layer launches them. -/
def S_PRMS := 20
def S_PQUANT := 21
def S_PQKV := 22
def S_PROPE := 23
def S_PKVN := 24
def S_PATTN := 25
def S_PKVS := 26
def S_PO := 27
def S_PADD := 28
def S_PRLOG := 29
def S_PRTOP := 30
def S_PGROUP := 31
def S_PGU := 32
def S_PDOWN := 33
def S_PCOMB := 34

/-! ## Buffers

    The first block is *this layer's*: nine dense weights, a key cache and a
    value cache. None of them is ever allocated to hold anything — the loop
    points them at the store. The experts are reached through the pool and
    `B_SLOTOF`. -/

def W_ANORM := 0
def W_QKVW := 1
def W_QKVB := 2
def W_SINKS := 3
def W_OW := 4
def W_OB := 5
def W_MNORM := 6
def W_RW := 7
def W_RB := 8
def C_KC := 9
def C_VC := 10
def B_X := 11
def B_XN := B_X + 1
def B_QKV := B_X + 2
def B_ATT := B_X + 3
def B_TMP := B_X + 4
def B_ROPE := B_X + 5
def B_META := B_X + 6
def B_NMH := B_X + 7
def B_RLOG := B_X + 8
def B_XM := B_X + 9
def B_GATES := B_X + 10
def B_MOUT := B_X + 11
def B_CHOSEN := B_X + 12
def B_LOGITS := B_X + 13
def B_LMHEAD := B_X + 14
def B_FNORM := B_X + 15
/-- The greedy token, alone in its own buffer.

    It could share `B_CHOSEN`, which has room; it must not. The runtime's
    copies assert that a transfer is the whole of a buffer, so reading four
    bytes out of sixteen is not a short read but a failure — and one that
    arrives as a panic from inside the driver rather than as a wrong answer. -/
def B_TOKEN := B_X + 16
/-- The embedding row as the table stores it: bf16, 5.7 KiB, widened on the
    device. -/
def B_EMB := B_X + 17
/-- `[vocab, seed, 1/T as bits, ln(min_p) as bits]` for the sampling kernel. -/
def B_SAMP := B_X + 18
/-- The rotated q, k and v, narrowed to bf16 and packed two to a word.

    The cache holds bf16, so k and v arrive already narrowed; and the
    attention tile reads the query packed the same way, a word a lane, as it
    reads a key row. One pass over all three, immediately after the rotation. -/
def B_QKVB := B_X + 19
/-- How many words the first narrow covers, which is a constant: q, k and v
    together are `dQkvOut` floats however long the conversation is. -/
def B_NMQKV := B_X + 20
/-- Where a tile leaves its `(m, l, acc)` for the merge, and how many tiles
    there are -- which follows the position, so the kernel is told rather than
    generated for it. -/
def B_PART := B_X + 21
/-- `(layer, expert)` to the pool slot holding it, the device's copy of
    `DSLOTOF_OFF`, which the expert launches read (`GptOssKernels.Body.expertAt`). -/
def B_SLOTOF := B_X + 22
/-- The chosen experts' outputs side by side, `HH` floats each, and their
    hidden activations likewise. -/
def B_Y := B_X + 23
def B_HID := B_X + 24
/-! A prompt's working set, `PT` tokens a chunk (`dPrefillChunkM`): the
    residual stream and its norm, int8 rows and their scales, the packed QKV,
    the chunk's narrowed keys and values, attention's output, the router's
    logits, choices and gates, the expert grouping (offsets, pairs, where each
    choice landed), the experts' hidden and output rows, the embedding rows;
    then the counts and positions the kernels read, one buffer each. -/
def B_PX := B_X + 25
def B_PXN := B_X + 26
def B_PXQ := B_X + 27
def B_PXS := B_X + 28
def B_PQKV := B_X + 29
def B_PKC := B_X + 30
def B_PVC := B_X + 31
def B_PATT := B_X + 32
def B_PTMP := B_X + 33
def B_PLOG := B_X + 34
def B_PCH := B_X + 35
def B_PGW := B_X + 36
def B_POFFS := B_X + 37
def B_PPAIR := B_X + 38
def B_PPOS := B_X + 39
def B_PHID := B_X + 40
def B_PHQ := B_X + 41
def B_PHS := B_X + 42
def B_PY := B_X + 43
def B_PEMB := B_X + 44
/-- `T`, `start`, `window`, `cap`, `stride` — the chunk's tokens, where it
    starts, and this layer's key window and cache. -/
def B_PM_T := B_X + 45
/-- `T` rounded up to the GEMMs' token tile. -/
def B_PM_TP := B_X + 46
/-- Blocks of 32 each quantisation takes: the hidden rows, attention's, the
    experts' hidden rows. -/
def B_PM_QX := B_X + 47
def B_PM_QA := B_X + 48
def B_PM_QH := B_X + 49
/-- Floats in the chunk's residual stream, and bf16 pairs in its embeddings. -/
def B_PM_TH := B_X + 50
def B_PM_WID := B_X + 51
/-- The layer, for the expert launches' slot lookup. -/
def B_PM_LAYER := B_X + 52
def DSTORE := B_X + 53                  -- 9 x NL dense weights
def KVSTORE := DSTORE + 9 * NL          -- 2 x NL caches
/-- The expert pool: one allocation, `ROW_BYTES` a slot, as many slots as the
    card has room for once everything else is allocated. -/
def B_POOL := KVSTORE + 2 * NL
def DNBUF := B_POOL + 1

def dStoreBuf (layer k : Nat) : Nat := DSTORE + 9 * layer + k
/-- The depth a layer's cache is *allowed*, which is what `dBufBytes` is about.

    What it is actually given is chosen at start-up and allocated then; this
    list fixes the number of buffers and says each is non-empty, and neither of
    those depends on the depth. Sizing it at the ceiling keeps the two claims
    the theorem below makes true without pretending the artifact knows how big
    the cache will be. -/
def capOf (layer : Nat) : Nat := if layer % 2 == 0 then CAP_SWA else CAP_MAX

/-- Tokens a prompt chunk holds, and routed pairs it can make. -/
def PT : Nat := 512
def PP : Nat := PT * TOPK

/-- The prompt working set's bytes, in `B_PX …` order. -/
def dPrefillBytes : List Nat :=
  [ PT * HH * 4, PT * HH * 4, PT * QO, PT * (QO / 32) * 4, PT * dQkvOut * 4
  , PT * NKV * HD * 2, PT * NKV * HD * 2, PT * QO * 4, PT * HH * 4
  , PT * NE * 4, PT * TOPK * 4, PT * TOPK * 4, 8 * (NE + 1), PP * 4, PP * 4
  , PP * II * 4, PP * II, PP * (II / 32) * 4, PP * HH * 4, PT * HH * 2
  , 32, 32, 32, 32, 32, 32, 32, 32 ]

def dBufBytes : List Nat :=
  (List.range 9).map (fun k => dDevBytes.getD k 0)         -- current layer: placeholders
    ++ [128, 128]
    ++ [ HH * 4, HH * 4, dQkvOut * 4
       , (64 * 64) * 4, HH * 4
       , 2 * ROPE_N * HALF * 4, 64 * 4, 4
       , NE * 4, HH * 4, 32 * 4, HH * 4, TOPK * 4
       , VOCAB * 4, q8Bytes VOCAB HH, HH * 4, 4, HH * 2, 16
       , dQkvOut * 2, 4
       , GptOssKernels.attPartialFloats * 4 * 64 * GptOssKernels.ATT_TILES_MAX
       , NL * NE * 4, TOPK * HH * 4, TOPK * II * 4 ]
    ++ dPrefillBytes
    ++ (List.range NL).flatMap (fun _ => (List.range 9).map (fun k => dDevBytes.getD k 0))
    ++ (List.range NL).flatMap (fun l => [8 * capOf l * 64 * 2, 8 * capOf l * 64 * 2])
    ++ [NSLOT_MAX * ROW_BYTES]

theorem gptoss_decode_alloc_covers :
    dBufBytes.length = DNBUF ∧ dBufBytes.all (fun n => decide (0 < n)) = true := by
  native_decide

/-! ## Memory map -/

def DSLOT : Nat := 0x30000
def DPTX_OFF : Nat := 0x0100
def dSlotOff (i : Nat) : Nat := DPTX_OFF + i * DSLOT
def DBIND_OFF : Nat := dSlotOff dPtx.length
def dBindOff (i : Nat) : Nat := DBIND_OFF + 8 * i
/-- Each kernel's module and function handles. -/
def DFN_OFF : Nat := DBIND_OFF + 8 * DNBUF
/-- A launch's eight parameter cells, then the array of pointers to them. -/
def DLOCAL_OFF : Nat := DFN_OFF + 16 * dPtx.length
def DMETA_OFF : Nat := DLOCAL_OFF + 128
def DCHOSEN_OFF : Nat := DMETA_OFF + 64 * 4
/-- `(layer, expert)` to the slot holding it, or `-1`. -/
def DSLOTOF_OFF : Nat := DCHOSEN_OFF + 4 * TOPK
/-- Slot to the `(layer, expert)` it holds, or `-1`. -/
def DRESIDENT_OFF : Nat := DSLOTOF_OFF + 4 * NL * NE
/-- The victim pointer: this cache evicts round-robin. -/
def DCLOCK_OFF : Nat := DRESIDENT_OFF + 4 * NSLOT_MAX
def DNSLOT_OFF : Nat := DCLOCK_OFF + 4
def DPOOL_OFF : Nat := DNSLOT_OFF + 4
def DSTAGE_OFF : Nat := DPOOL_OFF + 8
def DMISS_OFF : Nat := DSTAGE_OFF + 8
def DINIT_OFF : Nat := DMISS_OFF + 4
/-- Positions the full-attention layers were given room for, and `HD` times it.

    Read from the caller's `D_CTX` on the first call, clamped, and kept: the
    buffers were allocated to it, so it cannot change afterwards and every
    later step reads it from here rather than being told again. -/
def DCTX_OFF : Nat := DINIT_OFF + 4
def DCTXSTRIDE_OFF : Nat := DCTX_OFF + 4
/-! ### The tokenizer's corner

    A chat turn is text in and text out, so this program owns a tokenizer as
    well as a model. The bodies come from `TokenizerCommon` and `PretokCommon`;
    what is local is where their working arrays sit. -/

/-- The longest prompt or reply this program handles, in bytes. -/
def TEXT_MAX : Nat := 32768

def DTOK_BASE : Nat := DCTXSTRIDE_OFF + 8
def DT_PATHPTR : Nat := DTOK_BASE
def DT_BUFPTR : Nat := DTOK_BASE + 8
def DT_TOKCOUNT : Nat := DTOK_BASE + 16
def DT_TEXTLEN : Nat := DTOK_BASE + 24
def DT_HTKEY : Nat := DTOK_BASE + 32
def DT_HTVAL : Nat := DTOK_BASE + 40
def DT_CPCOUNT : Nat := DTOK_BASE + 48
def DT_OUTCOUNT : Nat := DTOK_BASE + 56
/-- Where a step leaves the token it produced. -/
def DT_NEXT : Nat := DTOK_BASE + 64
def DT_GENCOUNT : Nat := DTOK_BASE + 72
def DT_PATHCAP : Nat := DTOK_BASE + 80
def DT_TEXTIN : Nat := DTOK_BASE + 128
def DT_TEXTOUT : Nat := DT_TEXTIN + TEXT_MAX
def DT_TOKBUF : Nat := DT_TEXTOUT + TEXT_MAX
def DT_CPBUF : Nat := DT_TOKBUF + 4 * TEXT_MAX
def DT_CPBYTE : Nat := DT_CPBUF + 4 * TEXT_MAX
def DT_OUTTOK : Nat := DT_CPBYTE + 4 * (TEXT_MAX + 1)
def DT_GENTOK : Nat := DT_OUTTOK + 4 * TEXT_MAX
/-- A prompt layer's expert offsets read back (`NE + 1` words). -/
def DP_EOFF_OFF : Nat := DT_GENTOK + 4 * TEXT_MAX
/-- A prompt chunk's slot-map changes, each word the source of its own copy
    to the device (two a miss: the expert's slot, its victim's none), with
    their count; then the counts staged for a prompt's buffers. -/
def DP_SLOTQ_OFF : Nat := DP_EOFF_OFF + 4 * (NE + 1)
def DP_SLOTQN_OFF : Nat := DP_SLOTQ_OFF + 4 * (2 * NL * NE)
def DP_META_OFF : Nat := DP_SLOTQN_OFF + 8
/-- Where the embedding table sits in pinned host memory, read whole at
    start-up. -/
def DEMB_OFF : Nat := DP_META_OFF + 4 * 16
def DMEM_SIZE : Nat := DEMB_OFF + 8 + 0x100

def dTokMem : TokenizerCommon.TokMem :=
  { htCtx := ContextSlots.ht
    pathPtr := DT_PATHPTR, pathCap := DT_PATHCAP, bufPtr := DT_BUFPTR
    tokenBuf := DT_TOKBUF, tokenCount := DT_TOKCOUNT
    textIn := DT_TEXTIN, textOut := DT_TEXTOUT, textLen := DT_TEXTLEN
    htKey := DT_HTKEY, htVal := DT_HTVAL
    fileMaxBytes := 32 * 1024 * 1024, textMaxBytes := TEXT_MAX }

def dPretokMem : PretokCommon.PretokMem :=
  { cpBuf := DT_CPBUF, cpByte := DT_CPBYTE, cpCount := DT_CPCOUNT
    outTok := DT_OUTTOK, outCount := DT_OUTCOUNT }
def DHOST_LEN_OFF : Nat := 0x0080
/-- `main`, the entry point every module declares; the scalars `1` and `0`
    cuBLAS reads; the cuBLAS handle. -/
def DNAME_OFF : Nat := 0x0088
def DONE_OFF : Nat := 0x0090
def DZERO_OFF : Nat := 0x0094
def DBLAS_OFF : Nat := 0x0098

theorem gptoss_decode_ptx_fits :
    dPtx.all (fun t => decide (t.utf8ByteSize + 1 ≤ DSLOT)) = true := by
  native_decide

def dMemMap : AlgorithmLib.Layout.RegionMap :=
  (List.range dPtx.length).map (fun i => ⟨s!"ptx{i}", dSlotOff i, DSLOT⟩)
    ++ [⟨"hostLen", DHOST_LEN_OFF, 4⟩, ⟨"name", DNAME_OFF, 5⟩, ⟨"one", DONE_OFF, 4⟩,
        ⟨"zero", DZERO_OFF, 4⟩, ⟨"blas", DBLAS_OFF, 8⟩, ⟨"bind", DBIND_OFF, 8 * DNBUF⟩,
        ⟨"fns", DFN_OFF, 16 * dPtx.length⟩, ⟨"local", DLOCAL_OFF, 128⟩, ⟨"meta", DMETA_OFF, 64 * 4⟩,
        ⟨"chosen", DCHOSEN_OFF, 4 * TOPK⟩,
        ⟨"slotOf", DSLOTOF_OFF, 4 * NL * NE⟩,
        ⟨"resident", DRESIDENT_OFF, 4 * NSLOT_MAX⟩,
        ⟨"clock", DCLOCK_OFF, 4⟩, ⟨"nslot", DNSLOT_OFF, 4⟩,
        ⟨"pool", DPOOL_OFF, 8⟩, ⟨"stage", DSTAGE_OFF, 8⟩,
        ⟨"miss", DMISS_OFF, 4⟩, ⟨"init", DINIT_OFF, 4⟩,
        ⟨"ctx", DCTX_OFF, 4⟩, ⟨"ctxStride", DCTXSTRIDE_OFF, 8⟩,
        ⟨"tokPathPtr", DT_PATHPTR, 8⟩, ⟨"tokBufPtr", DT_BUFPTR, 8⟩,
        ⟨"tokCount", DT_TOKCOUNT, 8⟩, ⟨"tokTextLen", DT_TEXTLEN, 8⟩,
        ⟨"htKey", DT_HTKEY, 8⟩, ⟨"htVal", DT_HTVAL, 8⟩,
        ⟨"cpCount", DT_CPCOUNT, 8⟩, ⟨"outCount", DT_OUTCOUNT, 8⟩,
        ⟨"next", DT_NEXT, 4⟩, ⟨"genCount", DT_GENCOUNT, 8⟩, ⟨"tokPathCap", DT_PATHCAP, 8⟩,
        ⟨"textIn", DT_TEXTIN, TEXT_MAX⟩, ⟨"textOut", DT_TEXTOUT, TEXT_MAX⟩,
        ⟨"tokBuf", DT_TOKBUF, 4 * TEXT_MAX⟩,
        ⟨"cpBuf", DT_CPBUF, 4 * TEXT_MAX⟩,
        ⟨"cpByte", DT_CPBYTE, 4 * (TEXT_MAX + 1)⟩,
        ⟨"outTok", DT_OUTTOK, 4 * TEXT_MAX⟩,
        ⟨"genTok", DT_GENTOK, 4 * TEXT_MAX⟩,
        ⟨"pEoff", DP_EOFF_OFF, 4 * (NE + 1)⟩, ⟨"pSlotQ", DP_SLOTQ_OFF, 4 * (2 * NL * NE)⟩, ⟨"pSlotQN", DP_SLOTQN_OFF, 8⟩, ⟨"pMeta", DP_META_OFF, 4 * 16⟩, ⟨"emb", DEMB_OFF, 8⟩]

theorem gptossDecodeMap_ok :
    dMemMap.okB = true ∧ dMemMap.withinB DMEM_SIZE = true := by native_decide


/-! ## The host program -/


def D_POS : Nat := 4
/-- Zero for a single step, one for a whole turn. -/
def D_MODE : Nat := 8
def D_TLEN : Nat := 12
/-- Each path in the input is a field of this many bytes (`Sem.readField`). -/
def D_PATH_FIELD : Nat := 256
def D_PEXP : Nat := 16
def D_PDEN : Nat := 272
def D_PEMB : Nat := 528
def D_PTOK : Nat := 784
/-- The id a turn stops on, and how many tokens it may generate. Both are the
    caller's, because which token ends a turn is a fact about the chat template
    and not about the model. -/
def D_STOP : Nat := 1040
def D_MAXNEW : Nat := 1044

/-- **The chat template, as tokens rather than as code.**

    A turn is `pre ++ tokenize(text) ++ post`, and the caller supplies the two
    id lists. Harmony's control tokens are not text — BPE over the literal
    `<|start|>` yields its pieces, not the special id — so they cannot come
    through the tokenizer, and hard-coding them here would put one checkpoint's
    chat format inside a program that is otherwise about the model. -/
def D_NPRE : Nat := 1048
def D_NPOST : Nat := 1052
/-- `1/T` as `Float32` bits, and the seed. Zero `1/T` is greedy.

    After the two counts rather than before them, because everything below 1040
    is a path: `D_PTOK` runs to 1039, and anything tucked in there is read out
    of the middle of a filename. -/
def D_INVT : Nat := 1056
def D_SEED : Nat := 1060
/-- `ln(min_p)` as `Float32` bits: a token is out of the draw when its
    probability is below `min_p` times the most likely token's. Negative
    infinity keeps everything, which is what a caller that does not want the
    filter sends. -/
def D_LNMINP : Nat := 1064
/-- **Where in the conversation this turn begins.**

    Zero starts one. Anything else continues the cache that is already there,
    so a turn costs the tokens it actually adds rather than the whole
    transcript again -- which is the difference between a reply that takes a
    second and one that takes longer every time you speak. -/
def D_STARTPOS : Nat := 1068
/-- Positions to give the full-attention layers, read once on the first call.

    The key cache is the largest thing on this card after the experts, and how
    much of it to buy is a decision only the caller can make: every position of
    it is an expert slot not held resident, so a long context is paid for in
    misses on every token whether or not the conversation ever gets long. Zero
    asks for the default. -/
def D_CTX : Nat := 1072
/-- Room for a chat template on each side of the text.

    A client that continues the cache sends only what the turn adds, so this is
    far more than a turn needs; it is sized for one that rebuilds from position
    zero and has to replay the transcript.

    `CAP_FULL` is the real ceiling — a conversation cannot outrun the key cache
    — so this matches it rather than sitting below it and failing first with a
    message about buffers, which tells a user nothing about what went wrong. -/
def TMPL_MAX : Nat := 8192
def D_PRE : Nat := 1076
def D_POST : Nat := D_PRE + 4 * TMPL_MAX
def D_TEXT : Nat := D_POST + 4 * TMPL_MAX
def D_IN_BYTES : Nat := D_TEXT + TEXT_MAX

/-! ## What comes back

    `[n : u32][misses : i32]` then the logits, then the layer trace, then the
    turn's text. Every region has a fixed home so that one entry can serve both
    modes without the caller having to know which one it asked for. -/
def D_OUT_LOGITS : Nat := 8
def D_OUT_TRACE : Nat := D_OUT_LOGITS + VOCAB * 4
def D_OUT_TEXT : Nat := D_OUT_TRACE + 2 * NL * HH * 4
/-- The ids behind that text. A turn that returns only bytes is a turn whose
    detokenizer nobody can check: text and ids together let a caller confirm
    the two agree without running the model twice. -/
def D_OUT_NGEN : Nat := D_OUT_TEXT + TEXT_MAX
def D_OUT_GEN : Nat := D_OUT_NGEN + 8
/-- The ids the tokenizer made of the caller's text.

    A client holding a conversation needs them: the next turn's `pre` is
    everything said so far, and a client that had to tokenize its own history
    would be a client that has to agree with the model about what a token is —
    which is the thing this application exists to avoid. -/
def D_OUT_NTEXT : Nat := D_OUT_GEN + 4 * TEXT_MAX
def D_OUT_TEXTTOK : Nat := D_OUT_NTEXT + 8
def NBLK : Nat := 32 * warpsPerCta
/-- Rows a block of the Q8_0 GEMV takes. -/
def q8Rows : Nat := warpsPerCta * GptOssKernels.Body.rowsPerWarpQ8

/-- The staging buffer: wide enough for the widest dense tensor, and every
    load that exceeds it is chunked rather than trusted to fit. -/
def STAGE_BYTES : Nat := dQkvOut * HH * 2

/-- Kernel `i`, its module and function handles kept in the table. -/
def dKern (i : Nat) : Drv.Kern :=
  { ptxOff := dSlotOff i, nameOff := DNAME_OFF, modSlot := DFN_OFF + 16 * i, fnSlot := DFN_OFF + 16 * i + 8,
    paramsOff := DLOCAL_OFF + 64, bufs := [], grid := (1, 1, 1), block := (1, 1, 1) }

/-- Launch the function whose handle is at `fnAddr`, over `grid` blocks of
    `blk` threads, handed the pointers the entries `bs` hold.

    The handle's address is a *register*, which is what lets a step choose its
    own kernel without a branch around the launch. -/
def dEnqueueFn (ptr fnAddr : V .i64) (grid : V .i32) (blk : Nat) (bs : List Nat) : Prog V L Unit := do
  Drv.bind ptr DLOCAL_OFF (DLOCAL_OFF + 64) (bs.map dBindOff)
  let one ← iconst32 1
  let _ ← ext (.cuda .launchKernel) %[← load64 fnAddr, grid, one, one, ← iconst32 blk, one, one,
    ← iconst32 0, ← iconst64 0, ← iaddImm ptr (DLOCAL_OFF + 64), ← iconst64 0]

/-- Kernel `i` over `grid` blocks, the grid in a register.

    One launch needs it: the attention tiles' count is a function of the
    position, and launching the largest grid that could ever be needed with the
    threads past the count returning would be thousands of blocks doing nothing
    at the context this supports. -/
def dEnqueueG (ptr : V .i64) (i : Nat) (grid : V .i32) (blk : Nat) (bs : List Nat) : Prog V L Unit := do
  dEnqueueFn ptr (← iaddImm ptr (DFN_OFF + 16 * i + 8)) grid blk bs

def dEnqueue (ptr : V .i64) (i g blk : Nat) (bs : List Nat) : Prog V L Unit := do
  dEnqueueG ptr i (← iconst32 g) blk bs

/-- **Expert `key` (`layer · NE + e`) into slot `slot`**: its row, one copy
    from the pinned pool to the device pool, in the launches' stream, so it
    lands after every launch before it and before every one after, and the
    host does not wait. The caller keeps the two maps. -/
def dFillM (slot key : V .i64) : Prog V L Unit := do
  let ptr ← basePtr
  let rowB ← iconst64 ROW_BYTES
  let src ← iadd (← load64 (← iaddImm ptr DPOOL_OFF)) (← imul key rowB)
  let dst ← iadd (← load64 (← iaddImm ptr (dBindOff B_POOL))) (← imul slot rowB)
  let _ ← ext (.cuda .memcpyHtoDAsync) %[dst, src, rowB, ← iconst64 0]

/-- The slot map to the device, whole: `NL · NE` words. -/
def dUploadSlotsM : Prog V L Unit := do
  let ptr ← basePtr
  let b ← load64 (← iaddImm ptr (dBindOff B_SLOTOF))
  let _ ← ext (.cuda .memcpyHtoD) %[b, (← iaddImm ptr DSLOTOF_OFF), (← iconst64 (4 * NL * NE))]

/-- **Start-up.** Pin the expert file, take whatever device memory is going,
    and put the dense weights where they will stay. -/
def dInitM : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  Drv.up ptr ContextSlots.cuda
  let _ ← ext (.cublas .create) %[← iaddImm ptr DBLAS_OFF]
  for i in List.range dPtx.length do
    Drv.loadModule ptr (dKern i)
    Drv.getFunction ptr (dKern i)
  -- the expert pool: pinned once, and never moved again
  let poolBytes ← iconst64 (NL * NE * ROW_BYTES)
  let _ ← ext (.cuda .memAllocHost) %[← iaddImm ptr DPOOL_OFF, poolBytes]
  let poolPtr ← load64 (← iaddImm ptr DPOOL_OFF)
  let zero64 ← iconst64 0
  let pExp ← iaddImm dataPtr D_PEXP
  let _ ← readFileFrom pExp (← iconst64 D_PATH_FIELD) poolPtr zero64 poolBytes
  -- the embedding table, all of it, pinned beside the pool: a token's row is
  -- then a copy, not a read of the file, and no page of it is swapped out
  -- while the expert file's reading fills the page cache
  let embBytes ← iconst64 (VOCAB * HH * 2)
  let _ ← ext (.cuda .memAllocHost) %[← iaddImm ptr DEMB_OFF, embBytes]
  let emb ← load64 (← iaddImm ptr DEMB_OFF)
  let _ ← readFileFrom (← iaddImm dataPtr D_PEMB) (← iconst64 D_PATH_FIELD) emb zero64 embBytes
  -- a staging buffer, big enough for the widest dense tensor
  let stageBytes ← iconst64 STAGE_BYTES
  let _ ← ext (.cuda .memAllocHost) %[← iaddImm ptr DSTAGE_OFF, stageBytes]
  let stagePtr ← load64 (← iaddImm ptr DSTAGE_OFF)
  -- **The context, chosen here and fixed from here on.**
  --
  -- Zero means the default.  Anything else is clamped to the published
  -- rotation tables, which is the hard ceiling: past `CAP_MAX` there is no row
  -- to encode a position with.  The buffers below are allocated to whatever
  -- comes out, so this is the last moment it can be decided.
  let ctxAsk ← uload32_64 (← iaddImm dataPtr D_CTX)
  let ctxD ← ifte .eq ctxAsk zero64 (pure %[← iconst64 CAP_DEFAULT]) (pure %[ctxAsk])
  let ctxMax ← iconst64 CAP_MAX
  let ctxC ← ifte .ugt (ctxD.head) ctxMax (pure %[ctxMax]) (pure %[ctxD.head])
  let ctx := ctxC.head
  storeI32 (← ireduce32 ctx) (← iaddImm ptr DCTX_OFF)
  let ctxStride ← imul ctx (← iconst64 HD)
  storeI64 ctxStride (← iaddImm ptr DCTXSTRIDE_OFF)
  -- every buffer that is not an expert slot.  The caches are the exception:
  -- their size is the context, and a full layer's is what was just chosen.
  -- The caches hold bf16: two bytes an element, which is the whole point of
  -- narrowing before the store.
  let swaBytes ← iconst64 (8 * CAP_SWA * HD * 2)
  let fullBytes ← imul ctxStride (← iconst64 (8 * 2))
  -- one `(m, l, acc)` a query head a tile.  A fixed size, because the tile
  -- size is chosen at run time so that the tile *count* never exceeds
  -- `ATT_TILES_MAX` however long the context is.
  let partBytes ← iconst64
    (GptOssKernels.attPartialFloats * 4 * NQ * GptOssKernels.ATT_TILES_MAX)
  for (i, nb) in (List.range B_POOL).zip dBufBytes do
    let sz ←
      if KVSTORE ≤ i && i < B_POOL then
        pure (if (i - KVSTORE) / 2 % 2 == 0 then swaBytes else fullBytes)
      else if i == B_PART then pure partBytes
      else iconst64 nb
    let _ ← ext (.cuda .memAlloc) %[← iaddImm ptr (dBindOff i), sz]
  -- …and as many slots as the card turns out to have room for.  Asked, not
  -- assumed: a hardcoded count is a count that is wrong on the next card.
  let _ ← ext (.cuda .memGetInfo) %[← iaddImm ptr DLOCAL_OFF, ← iaddImm ptr (DLOCAL_OFF + 8)]
  let freeB ← load64 (← iaddImm ptr DLOCAL_OFF)
  let margin ← iconst64 (256 * 1024 * 1024)
  let usable ← isub freeB margin
  let rowB ← iconst64 ROW_BYTES
  let want ← udiv usable rowB
  let cap ← iconst64 NSLOT_MAX
  let nslotL ← ifte .ugt want cap (pure %[cap]) (pure %[want])
  let nslot := nslotL.head
  storeI32 (← ireduce32 nslot) (← iaddImm ptr DNSLOT_OFF)
  let _ ← ext (.cuda .memAlloc) %[← iaddImm ptr (dBindOff B_POOL), ← imul nslot rowB]
  -- the dense weights: read from the file, opened once, staged, uploaded, and
  -- resident
  let pDen ← iaddImm dataPtr D_PDEN
  let _ ← withFile pDen (← iconst64 D_PATH_FIELD) fsModeRead fun fDen => do
    for l in List.range NL do
      for k in List.range 9 do
        let n := dDevBytes.getD k 0
        let fo ← iconst64 (fileDense l k)
        let nb ← iconst64 n
        let _ ← ext (.fs .read) %[fDen, stagePtr, nb, fo]
        let id ← load64 (← iaddImm ptr (dBindOff (dStoreBuf l k)))
        let _ ← ext (.cuda .memcpyHtoD) %[id, stagePtr, nb]
    -- `lm_head` is 0.61 GiB and the staging buffer is 29.5 MiB, so it goes over
    -- in pieces.  Reading it in one call would write a gigabyte past the end of
    -- a pinned allocation -- not a wrong answer but a corrupted heap, and the
    -- expert pool is what sits next to it.
    for (off, n, b) in [(fileFinalNorm, HH * 4, B_FNORM),
                        (fileQ8Lm, q8Bytes VOCAB HH, B_LMHEAD)] do
      let id ← load64 (← iaddImm ptr (dBindOff b))
      let chunks := (n + STAGE_BYTES - 1) / STAGE_BYTES
      for c in List.range chunks do
        let take := min STAGE_BYTES (n - c * STAGE_BYTES)
        let fo ← iconst64 (off + c * STAGE_BYTES)
        let nb ← iconst64 take
        let _ ← ext (.fs .read) %[fDen, stagePtr, nb, fo]
        let bo ← iconst64 (c * STAGE_BYTES)
        let _ ← ext (.cuda .memcpyHtoD) %[← iadd id bo, stagePtr, nb]
    -- **The rotation tables: two slices, and sines first.**
    --
    -- The kernel indexes sine at zero and cosine at `ROPE_N * HALF`
    -- (`GptOssAttention.ropeCosIx`); the file has the opposite order and the
    -- full published height. So this is two reads, not one, and the order is
    -- the kernel's rather than the file's. See `ROPE_FILE_ROWS`.
    let ropeSlice := ROPE_N * HALF * 4
    let bRope ← load64 (← iaddImm ptr (dBindOff B_ROPE))
    for (fo, bo) in [(fileRopeSin, 0), (fileRope, ropeSlice)] do
      let f ← iconst64 fo
      let nb ← iconst64 ropeSlice
      let _ ← ext (.fs .read) %[fDen, stagePtr, nb, f]
      let b ← iconst64 bo
      let _ ← ext (.cuda .memcpyHtoD) %[← iadd bRope b, stagePtr, nb]
    iconst64 0
  -- the narrow kernel's two counts
  for (b, v) in [(B_NMH, HH / 2), (B_NMQKV, dQkvOut / 2)] do
    storeI32 (← iconst32 v) (← iaddImm ptr DMETA_OFF)
    let id ← load64 (← iaddImm ptr (dBindOff b))
    let n4 ← iconst64 4
    let _ ← ext (.cuda .memcpyHtoD) %[id, (← iaddImm ptr DMETA_OFF), n4]
  -- a full cache: expert `i` in slot `i` for every slot there is, so a card
  -- with room for them all starts with the whole model resident, and the
  -- rest of the experts are nowhere
  let minus1 ← iconst32 (-1)
  let four ← iconst64 4
  let nCache ← iconst64 (NL * NE)
  forLoop nCache fun i => do
    let off ← imul i four
    let here ← ifte .ult i nslot (pure %[← ireduce32 i]) (pure %[minus1])
    storeI32 here.head (← iadd (← iaddImm ptr DSLOTOF_OFF) off)
    storeI32 here.head (← iadd (← iaddImm ptr DRESIDENT_OFF) off)
  forLoop nslot fun i => do dFillM i i
  dUploadSlotsM
  -- every slot is taken, so the hand starts at the first
  storeI32 (← iconst32 0) (← iaddImm ptr DCLOCK_OFF)
  storeI32 (← iconst32 0) (← iaddImm ptr DMISS_OFF)


/-- **Point this layer's eleven buffers at this layer's weights.**

    The same six-integer-move idea the experts use, at a different table: nine
    dense tensors and the two halves of the cache. Nothing is copied and no
    kernel is re-emitted; a layer is a rebinding. -/
def dBindLayerM (layer : V .i32) : Prog V L Unit := do
  let ptr ← basePtr
  let four ← iconst64 4
  let nine ← iconst64 9
  let two ← iconst64 2
  let l64 ← uextend64 layer
  let base9 ← imul l64 nine
  for k in List.range 9 do
    let ix ← iaddImm base9 (DSTORE + k)
    let id ← load64 (← iadd (← iaddImm ptr DBIND_OFF) (← imul ix (← iconst64 8)))
    storeI64 id (← iaddImm ptr (dBindOff k))
  let base2 ← imul l64 two
  for j in List.range 2 do
    let ix ← iaddImm base2 (KVSTORE + j)
    let id ← load64 (← iadd (← iaddImm ptr DBIND_OFF) (← imul ix (← iconst64 8)))
    storeI64 id (← iaddImm ptr (dBindOff (C_KC + j)))

/-- **A cache lookup, and a miss if it is one.**

    `slotOf` says where an expert is, or that it is nowhere. When it is
    nowhere, the victim is whoever the clock hand is pointing at — round-robin,
    not least-recently-used, because at nine-tenths residency the two differ
    little and a scan of six hundred slots per miss would cost more than it
    saves, skipping the slots of this layer's other choices, which the
    launches will read. The row comes over from the pinned pool, both tables are
    corrected, and the miss is counted so the caller can see the rate.

    The transfer is the only thing in a decode step that touches the bus. -/
def dEnsureM (j : Nat) : Prog V L Unit := do
  let ptr ← basePtr
  let four ← iconst64 4
  let minus1 ← iconst32 (-1)
  let layer ← load32 (← iaddImm ptr (DMETA_OFF + 4 * DLAYER_SLOT))
  let e ← load32 (← iaddImm ptr (DCHOSEN_OFF + 4 * j))
  let ne64 ← iconst64 NE
  let key ← iadd (← imul (← uextend64 layer) ne64) (← uextend64 e)
  let slotAddr ← iadd (← iaddImm ptr DSLOTOF_OFF) (← imul key four)
  let slot0 ← load32 slotAddr
  when .eq slot0 minus1 do
      -- the victim, and the hand moved on: the first slot from the hand that
      -- holds none of this layer's choices, which the launches will read
      -- (one lap at most: a card with no such slot takes the hand's)
      let nslot ← load32 (← iaddImm ptr DNSLOT_OFF)
      let next (c : V .i32) : Prog V L (V .i32) := do
        let n ← iadd c (← iconst32 1)
        pure (← ifte .uge n nslot (pure %[← iconst32 0]) (pure %[n])).head
      let chosenHere (c : V .i32) : Prog V L (V .i64) := do
        let k ← load32 (← iadd (← iaddImm ptr DRESIDENT_OFF) (← imul (← uextend64 c) four))
        let mut hit ← iconst64 0
        for i in List.range TOPK do
          let ki ← iadd (← imul layer (← iconst32 NE)) (← load32 (← iaddImm ptr (DCHOSEN_OFF + 4 * i)))
          hit ← bor hit (← select (← icmp .eq k ki) (← iconst64 1) (← iconst64 0))
        pure hit
      let hand ← load32 (← iaddImm ptr DCLOCK_OFF)
      let found ← wloop %[hand, ← iconst32 0]
        (fun cs => do
          let busy ← chosenHere cs.head
          let more ← band busy (← select (← icmp .ult cs.snd nslot) (← iconst64 1) (← iconst64 0))
          pure (contIf .ne more (← iconst64 0), %[cs.head, cs.snd], ()))
        (fun cs _ => do pure %[← next cs.head, ← iadd cs.snd (← iconst32 1)])
      let clock ← select (← icmp .ult found.snd nslot) found.head hand
      storeI32 (← next clock) (← iaddImm ptr DCLOCK_OFF)
      let v64 ← uextend64 clock
      -- whoever was there is no longer anywhere
      let resAddr ← iadd (← iaddImm ptr DRESIDENT_OFF) (← imul v64 four)
      let old ← load32 resAddr
      when .ne old minus1 (do
        let oldAddr ← iadd (← iaddImm ptr DSLOTOF_OFF) (← imul (← uextend64 old) four)
        storeI32 minus1 oldAddr)
      dFillM v64 key
      storeI32 clock slotAddr
      storeI32 (← ireduce32 key) resAddr
      let m ← load32 (← iaddImm ptr DMISS_OFF)
      storeI32 (← iadd m (← iconst32 1)) (← iaddImm ptr DMISS_OFF)

/-- **The residual stream, mid-layer and end-of-layer, to the caller.**

    Forty-eight rows of eleven kilobytes, which is a fifteenth of the logits
    this program already returns. What it buys is the difference between
    knowing the answer is wrong and knowing *where*: a caller diffs every half
    of every layer against the model in a single run.

    It is here because the per-layer artifacts are separate programs. They can
    agree with the model exactly while this one does not, so passing them is
    not evidence about this, and a composition fault has nowhere else to show
    itself — the token and the logits are downstream of all twenty-four layers
    and report only that something, somewhere, was wrong.

    `half` is 0 after attention and 1 after the mixture. A single step (mode
    0) returns it; a turn does not, because each row is a wait for the device
    in the middle of a layer. -/
def dTraceM (layer : V .i32) (half : Nat) : Prog V L Unit := do
  let ptr ← basePtr
  let outPtr ← outPtr
  let mode ← load32 (← iaddImm (← dataPtr) D_MODE)
  when .eq mode (← iconst32 0) do
    let bX ← load64 (← iaddImm ptr (dBindOff B_X))
    let row ← iadd (← imul (← uextend64 layer) (← iconst64 2)) (← iconst64 half)
    let dst ← iadd outPtr
      (← iadd (← iconst64 (8 + VOCAB * 4)) (← imul row (← iconst64 (HH * 4))))
    Drv.sync
    let _ ← ext (.cuda .memcpyDtoH) %[dst, bX, (← iconst64 (HH * 4))]

set_option maxRecDepth 8000 in
/-- **One layer, whichever layer it is.** -/
def dLayerM (layer : V .i32) : Prog V L Unit := do
  let ptr ← basePtr
  dBindLayerM layer
  storeI32 layer (← iaddImm ptr (DMETA_OFF + 4 * DLAYER_SLOT))
  -- the pristine copies, not `M_SEQ`, which this body is about to overwrite
  let seqLen ← load32 (← iaddImm ptr (DMETA_OFF + 4 * (M_SEQ + 16)))
  let seqLenF ← load32 (← iaddImm ptr (DMETA_OFF + 4 * (M_SEQ + 8)))
  let zero32 ← iconst32 0
  let one32 ← iconst32 1
  let h ← load64 (← iaddImm ptr DBLAS_OFF)
  let oneF ← iaddImm ptr DONE_OFF
  let zeroF ← iaddImm ptr DZERO_OFF
  -- a sliding layer is an even one; what differs is its depth, not its kernels
  let par ← isub layer (← imul (← udiv layer (← iconst32 2)) (← iconst32 2))
  -- the effective length and the cache depth both follow from the layer type
  let lenSel ← ifte .eq par zero32 (pure %[seqLen]) (pure %[seqLenF])
  let sLen := lenSel.head
  let sLen64 ← uextend64 sLen
  let ctxStrideL ← load { ty := .i64 }
                     (← iaddImm ptr DCTXSTRIDE_OFF)
  let capSel ← ifte .eq par zero32
    (pure %[← iconst64 (CAP_SWA * HD)]) (pure %[ctxStrideL])
  let strideKV := capSel.head
  -- **The meta is per layer, not per token.**
  --
  -- The softmax reads its trip counts and its row stride out of the meta
  -- buffer, and those follow the cache depth: a sliding layer runs over at
  -- most 128 keys and a full one over everything so far.  Publishing the
  -- sliding numbers once per token and letting both kinds read them is right
  -- only while the position is below the window, which is exactly why it
  -- survived every test up to 128 and would have failed past it.  Two hundred
  -- and fifty-six bytes per layer is what correctness costs here.
  let c32 ← iconst32 32
  let posNow ← load32 (← iaddImm ptr (DMETA_OFF + 4 * M_POS))
  let ch ← udiv sLen c32
  let tl ← imul ch c32
  let ctxNow ← load32 (← iaddImm ptr DCTX_OFF)
  let capSlot ← ifte .eq par zero32 (pure %[← iconst32 CAP_SWA]) (pure %[ctxNow])
  let capV := capSlot.head
  let slotNow ← isub posNow (← imul (← udiv posNow capV) capV)
  storeI32 sLen (← iaddImm ptr (DMETA_OFF + 4 * M_SEQ))
  storeI32 ch (← iaddImm ptr (DMETA_OFF + 4 * M_CHUNKS))
  storeI32 tl (← iaddImm ptr (DMETA_OFF + 4 * M_TAIL))
  storeI32 (← isub sLen tl) (← iaddImm ptr (DMETA_OFF + 4 * M_REM))
  storeI32 slotNow (← iaddImm ptr (DMETA_OFF + 4 * M_SLOT))
  -- **How this layer's keys are split, chosen for the length it actually has.**
  --
  -- Aim for `ATT_TILES_MAX` tiles, which is what measured best at depth, but
  -- never cut below `ATT_TILE_MIN` keys: at a thousand keys the first rule
  -- alone would make tiles of one, and a tile is walked serially by a warp
  -- whatever its size. So short layers get the floor and get more tiles than
  -- the target -- which is what they were short of -- and long ones get the
  -- target, which bounds the merge and the partials buffer at once.
  let tMax ← iconst32 GptOssKernels.ATT_TILES_MAX
  let even ← udiv (← iadd sLen (← isub tMax (← iconst32 1))) tMax
  let tooSmall ← ifte .ult even (← iconst32 GptOssKernels.ATT_TILE_MIN)
    (pure %[← iconst32 GptOssKernels.ATT_TILE_MIN]) (pure %[even])
  let tileSz := tooSmall.head
  let nTiles ← udiv (← iadd sLen (← isub tileSz (← iconst32 1))) tileSz
  storeI32 tileSz (← iaddImm ptr (DMETA_OFF + 4 * M_TILESZ))
  storeI32 nTiles (← iaddImm ptr (DMETA_OFF + 4 * M_NTILES))
  -- The depth this layer's cache was allocated at, in the *words* the store
  -- kernel moves: `HDW` of them a head, because the cache is bf16 and two ride
  -- in one word.  The contractions below stride by `HD` elements over the same
  -- bytes, which is the same distance counted the other way.
  storeI32 (← imul capV (← iconst32 HDW)) (← iaddImm ptr (DMETA_OFF + 4 * M_KVSTRIDE))
  let bMetaL ← load64 (← iaddImm ptr (dBindOff B_META))
  let _ ← ext (.cuda .memcpyHtoD) %[bMetaL, (← iaddImm ptr DMETA_OFF), (← iconst64 (64 * 4))]
  let bRW ← load64 (← iaddImm ptr (dBindOff W_RW))
  let bXM ← load64 (← iaddImm ptr (dBindOff B_XM))
  let bRLOG ← load64 (← iaddImm ptr (dBindOff B_RLOG))
  dEnqueue ptr S_RMS 1 32 [B_X, W_ANORM, B_XN]
  dEnqueue ptr S_Q8QKV (dQkvOut / q8Rows) NBLK [W_QKVW, B_XN, B_QKV]
  let kH ← iconst32 HH
  dEnqueue ptr S_ADD (dQkvOut / 32) 32 [B_QKV, W_QKVB]
  dEnqueue ptr S_ROPEQ NQ 32 [B_QKV, B_META, B_ROPE]
  dEnqueue ptr S_ROPEK NKV 32 [B_QKV, B_META, B_ROPE]
  -- **Narrow once, and everything downstream is bf16.**
  --
  -- The cache holds bf16, so k and v have to be narrowed before they are
  -- stored; and q is narrowed with them, because the attention tile reads it
  -- as it reads the cache. One pass over all three.
  dEnqueue ptr S_NARROW ((dQkvOut / 2 + NBLK - 1) / NBLK) NBLK
    [B_QKV, B_QKVB, B_NMQKV]
  -- One store kernel per operand, whatever kind of layer this is: the depth it
  -- writes at is `M_KVSTRIDE`, published just above, and what it moves is a
  -- word holding two bf16 rather than one `Float32`.
  dEnqueue ptr S_KV NKV 32 [B_QKVB, C_KC, B_META]
  dEnqueue ptr (S_KV + 1) NKV 32 [B_QKVB, C_VC, B_META]
  let hd32 ← iconst32 HD
  let gqa32 ← iconst32 GQA
  let nkv32 ← iconst32 NKV
  let strideQ ← iconst64 (GQA * HD)
  let gqa64 ← iconst64 GQA
  let strideS ← imul gqa64 sLen64
  -- **Attention, in two launches rather than four.**
  --
  -- The scores are formed, softmaxed and mixed inside one kernel and never
  -- reach memory: a warp takes one key head and one tile of the sequence, and
  -- carries a running maximum and sum through it. What the four-kernel path
  -- spent most of its time on at depth was a scores buffer of `NQ * seqLen`
  -- floats that existed only because the stages were separate kernels.
  dEnqueueG ptr S_ATTTILE nTiles GptOssKernels.Body.ATT_THREADS
    [C_KC, C_VC, B_QKVB, B_PART, B_META]
  -- the merge is one warp a query head, and the sink joins here: it belongs to
  -- the head, and a tile that applied it would have it counted once per tile
  dEnqueue ptr S_ATTCOMB NQ 32 [B_PART, W_SINKS, B_ATT, B_META]
  dEnqueue ptr S_Q8O (HH / q8Rows) NBLK [W_OW, B_ATT, B_TMP]
  dEnqueue ptr S_ADD (HH / 32) 32 [B_TMP, W_OB]
  dEnqueue ptr S_ADD (HH / 32) 32 [B_X, B_TMP]
  dTraceM layer 0
  -- route
  dEnqueue ptr S_RMS 1 32 [B_X, W_MNORM, B_XM]
  let ne32 ← iconst32 NE
  let _ ← ext (.cublas .sgemv) %[h, one32, kH, ne32, oneF, bRW, kH, bXM, one32, zeroF, bRLOG, one32]
  dEnqueue ptr S_ADD 1 32 [B_RLOG, W_RB]
  dEnqueue ptr S_TOP4 1 32 [B_RLOG, B_GATES, B_CHOSEN]
  -- **The cache, when it can miss.** With every expert resident the launches
  -- find their slots on the device and the host does not wait for the router;
  -- otherwise it reads the choice, brings in what is missing, and hands the
  -- device the corrected map.
  let nslot ← load32 (← iaddImm ptr DNSLOT_OFF)
  when .ult nslot (← iconst32 (NL * NE)) do
    Drv.sync
    let bCh ← load64 (← iaddImm ptr (dBindOff B_CHOSEN))
    let _ ← ext (.cuda .memcpyDtoH) %[(← iaddImm ptr DCHOSEN_OFF), bCh, (← iconst64 (TOPK * 4))]
    let miss0 ← load32 (← iaddImm ptr DMISS_OFF)
    for j in List.range TOPK do
      dEnsureM j
    when .ne miss0 (← load32 (← iaddImm ptr DMISS_OFF)) do dUploadSlotsM
  -- one launch for all four gate/ups and one for all four downs
  dEnqueue ptr S_GATEUP (TOPK * (II / (warpsPerCta * rowsPerWarpGateUp))) NBLK
    [B_POOL, B_SLOTOF, B_CHOSEN, B_META, B_XM, B_HID]
  dEnqueue ptr S_DOWN (TOPK * (HH / (warpsPerCta * rowsPerWarpDown))) NBLK
    [B_POOL, B_SLOTOF, B_CHOSEN, B_META, B_HID, B_Y]
  dEnqueue ptr S_COMBINE ((HH + NBLK - 1) / NBLK) NBLK
    [B_Y, B_GATES, B_MOUT]
  dEnqueue ptr S_ADD (HH / 32) 32 [B_X, B_MOUT]
  dTraceM layer 1

/-- **The head**, from the residual stream in `B_X` at position `pos`: the
    final norm, `lm_head`, and the greedy token or a draw from the tempered
    softmax, which it returns and leaves at `DT_NEXT`. -/
def dHeadM (pos : V .i32) : Prog V L (V .i32) := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  dEnqueue ptr S_RMS 1 32 [B_X, B_FNORM, B_XN]
  dEnqueue ptr S_Q8LM (VOCAB / q8Rows) NBLK [B_LMHEAD, B_XN, B_LOGITS]
  let zz ← iconst32 0
  let oo ← iconst32 1
  -- **Greedy, or a draw from the tempered softmax.**
  --
  -- `1/T` of zero means the caller wants the largest logit, and that goes to
  -- the argmax kernel rather than to the sampler at an infinite temperature.
  -- Everything else goes to `sample_logits`, which adds Gumbel noise and takes
  -- the argmax of that — the same draw, in the same one pass.
  --
  -- The choice is arithmetic on the slot, not a branch around the launch:
  -- the function handle's address is a register, so `S_ARGMAX + (sampling ?
  -- 1 : 0)` picks the kernel and the launch itself is the same instruction
  -- either way.
  let invT ← load32 (← iaddImm dataPtr D_INVT)
  storeI32 (← iconst32 VOCAB) (← iaddImm ptr (DMETA_OFF + 4 * 34))
  -- the seed advances with the position, so a turn is reproducible from one
  -- number and no two steps draw the same noise
  let seed0 ← load32 (← iaddImm dataPtr D_SEED)
  storeI32 (← iadd seed0 (← imul pos (← iconst32 0x9E3779B1)))
    (← iaddImm ptr (DMETA_OFF + 4 * 35))
  storeI32 invT (← iaddImm ptr (DMETA_OFF + 4 * 36))
  storeI32 (← load32 (← iaddImm dataPtr D_LNMINP))
    (← iaddImm ptr (DMETA_OFF + 4 * 37))
  let bSamp ← load64 (← iaddImm ptr (dBindOff B_SAMP))
  let n12 ← iconst64 16
  let _ ← ext (.cuda .memcpyHtoD) %[bSamp, (← iaddImm ptr (DMETA_OFF + 4 * 34)), n12]
  let sampling ← ifte .ne invT zz (pure %[oo]) (pure %[zz])
  let headFn ← iadd (← iaddImm ptr (DFN_OFF + 16 * S_ARGMAX + 8))
                    (← imul (← uextend64 (sampling.head))
                            (← iconst64 (16 * (S_SAMPLE - S_ARGMAX))))
  dEnqueueFn ptr headFn (← iconst32 1) GptOssKernels.Body.HEAD_T [B_LOGITS, B_TOKEN, B_SAMP]
  Drv.sync
  -- the token this step produced, into memory this program owns
  let bTok ← load64 (← iaddImm ptr (dBindOff B_TOKEN))
  let n4 ← iconst64 4
  let _ ← ext (.cuda .memcpyDtoH) %[(← iaddImm ptr DT_NEXT), bTok, n4]
  -- …and the whole logit row, to a fixed place in the caller's buffer.  Eight
  -- hundred kilobytes is real traffic and it buys the only check that can
  -- localise a fault in the head: a caller can compare against the model's own
  -- logits instead of guessing from a token id.  A partial copy is not an
  -- option -- the runtime asserts a transfer is the whole buffer -- so it is
  -- all of them or none, and a single step (mode 0) or a prompt (mode 2) is
  -- what takes them.
  let outPtr ← outPtr
  let mode ← load32 (← iaddImm dataPtr D_MODE)
  when .ne mode (← iconst32 1) do
    let _ ← ext (.cuda .memcpyDtoH) %[(← iaddImm outPtr 8), (← load64 (← iaddImm ptr (dBindOff B_LOGITS))), (← iconst64 (VOCAB * 4))]
  load32 (← iaddImm ptr DT_NEXT)

set_option maxRecDepth 8000 in
/-- **One token, from a token and a position.**

    Everything a decode step is: the embedding row, the meta this position
    implies, twenty-four layers, the final norm, `lm_head`, and the largest
    logit's index. Returns that index, and leaves the logits and the layer
    trace in the caller's buffer where a checker can read them.

    A builder rather than an entry point, because a chat turn is this in a
    loop and the loop belongs on the same side of the boundary as the model. -/
def dStepM (tok posIn : V .i32) : Prog V L (V .i32) := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  -- **No step ever runs past the cache, whatever the caller asked for.**
  --
  -- The cache is a ring and a modulus cannot leave its buffer, so what bounds
  -- a position is the *allocation*: the full layers were given room for
  -- `DCTX_OFF` of them, and a step past that would write a slot the layer does
  -- not own. Clamping is not the answer to a conversation that has run out of
  -- room — the caller is expected to refuse the turn, and the generation loop
  -- below stops on its own — but it is what makes the failure a stalled reply
  -- rather than a write into whatever was allocated next.
  let capF ← isub (← load32 (← iaddImm ptr DCTX_OFF)) (← iconst32 1)
  let posL ← ifte .ugt posIn capF (pure %[capF]) (pure %[posIn])
  let pos := posL.head
  -- **The embedding row, gathered here.**
  --
  -- The table is bf16 and the first RMSNorm reads f32, so the row is copied
  -- out of the host's table at `token * HH * 2`, staged, uploaded as the
  -- 5.7 KiB it is, and widened on the device. The alternative that needs no
  -- kernel is to hold all 201088 rows in device memory, which costs 1.08 GiB —
  -- a hundred and eighty expert slots — to save a copy of under six kilobytes.
  let bEmb ← load64 (← iaddImm ptr (dBindOff B_EMB))
  let stagePtrT ← load64 (← iaddImm ptr DSTAGE_OFF)
  let rowB ← iconst64 (HH * 2)
  let rowOff ← imul (← uextend64 tok) rowB
  let _ ← ext (.c .memcpy) %[stagePtrT, ← iadd (← load64 (← iaddImm ptr DEMB_OFF)) rowOff, rowB]
  let _ ← ext (.cuda .memcpyHtoD) %[bEmb, stagePtrT, rowB]
  dEnqueue ptr S_WIDEN ((HH / 2 + NBLK - 1) / NBLK) NBLK [B_EMB, B_X, B_NMH]
  -- the meta both cache depths need
  let one32 ← iconst32 1
  let len ← iadd pos one32
  let c128 ← iconst32 CAP_SWA
  let cFull ← load32 (← iaddImm ptr DCTX_OFF)
  let swaL ← ifte .ugt len c128 (pure %[c128]) (pure %[len])
  let fullL ← ifte .ugt len cFull (pure %[cFull]) (pure %[len])
  let sL := swaL.head
  let fL := fullL.head
  storeI32 pos (← iaddImm ptr (DMETA_OFF + 4 * M_POS))
  storeI32 sL (← iaddImm ptr (DMETA_OFF + 4 * M_SEQ))
  let c32 ← iconst32 32
  let ch ← udiv sL c32
  storeI32 ch (← iaddImm ptr (DMETA_OFF + 4 * M_CHUNKS))
  let tl ← imul ch c32
  storeI32 tl (← iaddImm ptr (DMETA_OFF + 4 * M_TAIL))
  storeI32 (← isub sL tl) (← iaddImm ptr (DMETA_OFF + 4 * M_REM))
  let slotSwa ← isub pos (← imul (← udiv pos c128) c128)
  storeI32 slotSwa (← iaddImm ptr (DMETA_OFF + 4 * M_SLOT))
  storeI32 fL (← iaddImm ptr (DMETA_OFF + 4 * (M_SEQ + 8)))
  -- Both lengths again, in slots no layer writes.  `M_SEQ` is what the softmax
  -- reads, so each layer publishes its own length there; that makes the slot
  -- unusable as the *source* of either length, because a full layer would
  -- otherwise hand its length to the next sliding one.  Below the window the
  -- two are equal and the fault is invisible, which is exactly why it needs
  -- somewhere pristine to be read from.
  storeI32 sL (← iaddImm ptr (DMETA_OFF + 4 * (M_SEQ + 16)))
  let bMeta ← load64 (← iaddImm ptr (dBindOff B_META))
  let mb ← iconst64 (64 * 4)
  let _ ← ext (.cuda .memcpyHtoD) %[bMeta, (← iaddImm ptr DMETA_OFF), mb]
  -- twenty-four layers, one body
  let nl ← iconst64 NL
  forLoop nl fun l => do
    dLayerM (← ireduce32 l)
  dHeadM pos


/-- Words `vals` into the small device buffer `b`, through `DP_META_OFF`. -/
def dPrefillMetaM (ptr : V .i64) (b : Nat) (vals : List (V .i32)) : Prog V L Unit := do
  for (v, i) in vals.zipIdx do storeI32 v (← iaddImm ptr (DP_META_OFF + 4 * i))
  let id ← load64 (← iaddImm ptr (dBindOff b))
  let _ ← ext (.cuda .memcpyHtoD) %[id, ← iaddImm ptr DP_META_OFF, ← iconst64 (4 * vals.length)]

/-- **Expert `e` of `layer` resident, for a prompt layer.** `dEnsureM`'s
    lookup, miss and fill. A prompt takes the layers in turn, so the victim is
    an expert of a layer this chunk is done with, the misses taking those
    layers in rotation so that what is left out is spread over them (a step
    after the prompt meets it a little in every layer, not wholly in one);
    failing that, of the layer whose turn is furthest off. The device's slot
    map is changed word by word, each from a word of `DP_SLOTQ_OFF` that
    nothing writes again before the chunk's end. -/
def dEnsurePM (layer e : V .i32) : Prog V L Unit := do
  let ptr ← basePtr
  let four ← iconst64 4
  let minus1 ← iconst32 (-1)
  let key ← iadd (← imul (← uextend64 layer) (← iconst64 NE)) (← uextend64 e)
  let slotAddr ← iadd (← iaddImm ptr DSLOTOF_OFF) (← imul key four)
  when .eq (← load32 slotAddr) minus1 do
      let nslot ← load32 (← iaddImm ptr DNSLOT_OFF)
      -- the first slot holding an expert of layer `t`, or `nslot`
      let firstOf (t : V .i32) : Prog V L (V .i32) := do
        let r ← wloop %[← iconst32 0]
          (fun cs => do
            let c := cs.head
            let k ← load32 (← iadd (← iaddImm ptr DRESIDENT_OFF) (← imul (← uextend64 (← select (← icmp .ult c nslot) c (← iconst32 0))) four))
            let hit ← band (← select (← icmp .ne k minus1) (← iconst64 1) (← iconst64 0))
              (← select (← icmp .eq (← udiv k (← iconst32 NE)) t) (← iconst64 1) (← iconst64 0))
            let more ← band (← select (← icmp .ult c nslot) (← iconst64 1) (← iconst64 0))
              (← select (← icmp .eq hit (← iconst64 0)) (← iconst64 1) (← iconst64 0))
            pure (contIf .ne more (← iconst64 0), %[c], ()))
          (fun cs _ => do pure %[← iadd cs.head (← iconst32 1)])
        pure r.head
      -- layers back from this one, the nearest that holds a slot
      let found ← wloop %[← iconst32 1, nslot]
        (fun cs => do
          let more ← band (← select (← icmp .uge cs.snd nslot) (← iconst64 1) (← iconst64 0))
            (← select (← icmp .ult cs.head (← iconst32 NL)) (← iconst64 1) (← iconst64 0))
          pure (contIf .ne more (← iconst64 0), %[cs.head, cs.snd], ()))
        (fun cs _ => do
          let t ← urem (← iadd layer (← isub (← iconst32 NL) cs.head)) (← iconst32 NL)
          pure %[← iadd cs.head (← iconst32 1), ← firstOf t])
      let fallback ← select (← icmp .ult found.snd nslot) found.snd (← iconst32 0)
      let m ← load32 (← iaddImm ptr DMISS_OFF)
      let done ← select (← icmp .eq layer (← iconst32 0)) (← iconst32 1) layer
      let first ← firstOf (← urem m done)
      let clock ← select (← band (← select (← icmp .ugt layer (← iconst32 0)) (← iconst64 1) (← iconst64 0))
        (← select (← icmp .ult first nslot) (← iconst64 1) (← iconst64 0))) first fallback
      let v64 ← uextend64 clock
      let resAddr ← iadd (← iaddImm ptr DRESIDENT_OFF) (← imul v64 four)
      let old ← load32 resAddr
      -- one word of the device's slot map, from the next word of the queue
      let devMap ← load64 (← iaddImm ptr (dBindOff B_SLOTOF))
      let tell (k : V .i64) (v : V .i32) : Prog V L Unit := do
        let qn ← load32 (← iaddImm ptr DP_SLOTQN_OFF)
        let src ← iadd (← iaddImm ptr DP_SLOTQ_OFF) (← imul (← uextend64 qn) four)
        storeI32 v src
        storeI32 (← iadd qn (← iconst32 1)) (← iaddImm ptr DP_SLOTQN_OFF)
        let _ ← ext (.cuda .memcpyHtoDAsync) %[← iadd devMap (← imul k four), src, four, ← iconst64 0]
      when .ne old minus1 (do
        let oldAddr ← iadd (← iaddImm ptr DSLOTOF_OFF) (← imul (← uextend64 old) four)
        storeI32 minus1 oldAddr
        tell (← uextend64 old) minus1)
      dFillM v64 key
      storeI32 clock slotAddr
      storeI32 (← ireduce32 key) resAddr
      tell key clock
      let m ← load32 (← iaddImm ptr DMISS_OFF)
      storeI32 (← iadd m (← iconst32 1)) (← iaddImm ptr DMISS_OFF)

set_option maxRecDepth 8000 in
/-- **A prompt chunk through all twenty-four layers**: the `T ≤ PT` tokens at
    `toks` (32-bit ids), at positions `start …`, their embedding rows read and
    widened into `B_PX`, then each layer as the prefill kernels compute it for
    all of them at once: the norm, int8 rows, QKV, the rotation, the chunk's
    keys and values narrowed, attention over the cache and the chunk, the keys
    and values into the cache, `o_proj` and the residual, the norm, the
    router, the experts grouped, made resident, and run, and their gated sum
    into the residual. The GEMMs take `T` rounded up to their token tile; the
    rows past `T` are computed and read by nothing. -/
def dPrefillChunkM (toks T start : V .i64) : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let T32 ← ireduce32 T
  let tt ← iconst32 GptOssPrefill.TT
  let Tp ← imul (← udiv (← iadd T32 (← iconst32 (GptOssPrefill.TT - 1))) tt) tt
  let cdiv (a : V .i32) (b : Nat) : Prog V L (V .i32) := do
    udiv (← iadd a (← iconst32 (b - 1))) (← iconst32 b)
  -- the embedding rows, copied from the table one a token, and widened
  let stage ← load64 (← iaddImm ptr DSTAGE_OFF)
  let rowB ← iconst64 (HH * 2)
  let emb ← load64 (← iaddImm ptr DEMB_OFF)
  forLoop T fun i => do
    let tok ← load32 (← iadd toks (← ishlImm i 2))
    let _ ← ext (.c .memcpy) %[← iadd stage (← imul i rowB), ← iadd emb (← imul (← uextend64 tok) rowB), rowB]
    pure ()
  let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm ptr (dBindOff B_PEMB)), stage, ← imul T rowB]
  let pairs ← imul T32 (← iconst32 (HH / 2))
  dPrefillMetaM ptr B_PM_WID [pairs]
  dEnqueueG ptr S_WIDEN (← cdiv pairs NBLK) NBLK [B_PEMB, B_PX, B_PM_WID]
  -- the counts every layer shares
  let nP ← imul T32 (← iconst32 TOPK)
  dPrefillMetaM ptr B_PM_TP [Tp]
  dPrefillMetaM ptr B_PM_QX [← imul Tp (← iconst32 (HH / 32))]
  dPrefillMetaM ptr B_PM_QA [← imul Tp (← iconst32 (QO / 32))]
  dPrefillMetaM ptr B_PM_QH [← imul nP (← iconst32 (II / 32))]
  dPrefillMetaM ptr B_PM_TH [← imul T32 (← iconst32 HH)]
  let start32 ← ireduce32 start
  let ctx ← load32 (← iaddImm ptr DCTX_OFF)
  -- the slot-map queue's words free again once the last chunk's copies are done
  when .ult (← load32 (← iaddImm ptr DNSLOT_OFF)) (← iconst32 (NL * NE)) do
    Drv.sync
    storeI32 (← iconst32 0) (← iaddImm ptr DP_SLOTQN_OFF)
  let tiles ← iadd (← cdiv nP GptOssPrefill.TT) (← iconst32 NE)
  let nl ← iconst64 NL
  forLoop nl fun l => do
    let layer ← ireduce32 l
    dBindLayerM layer
    -- a sliding layer is an even one: a 128-key window over a 128-slot ring
    let par ← isub layer (← imul (← udiv layer (← iconst32 2)) (← iconst32 2))
    let swa ← icmp .eq par (← iconst32 0)
    let window ← select swa (← iconst32 CAP_SWA) (← iconst32 0)
    let cap ← select swa (← iconst32 CAP_SWA) ctx
    dPrefillMetaM ptr B_PM_T [T32, start32, window, cap, ← imul cap (← iconst32 HDW)]
    dPrefillMetaM ptr B_PM_LAYER [layer]
    let tp32 ← udiv Tp tt
    dEnqueueG ptr S_PRMS (← cdiv T32 8) NBLK [B_PX, W_ANORM, B_PXN, B_PM_T]
    dEnqueueG ptr S_PQUANT (← cdiv (← imul Tp (← iconst32 (HH / 32))) 256) NBLK [B_PXN, B_PXQ, B_PXS, B_PM_QX]
    dEnqueueG ptr S_PQKV (← imul (← iconst32 (dQkvOut / GptOssPrefill.TR)) tp32) NBLK
      [W_QKVW, B_PXQ, B_PXS, B_PQKV, W_QKVB, B_PM_TP]
    dEnqueueG ptr S_PROPE (← cdiv (← imul T32 (← iconst32 ((NQ + NKV) * (HD / 2)))) 256) NBLK [B_PQKV, B_ROPE, B_PM_T]
    let kvGrid ← cdiv (← imul T32 (← iconst32 (NKV * HD / 2))) 256
    dEnqueueG ptr S_PKVN kvGrid NBLK [B_PQKV, B_PKC, B_PVC, B_PM_T]
    dEnqueueG ptr S_PATTN (← imul (← iconst32 NKV) (← cdiv T32 GptOssPrefill.AT_TOK)) 128
      [B_PQKV, B_PKC, B_PVC, C_KC, C_VC, W_SINKS, B_PATT, B_PM_T]
    dEnqueueG ptr S_PKVS kvGrid NBLK [B_PKC, B_PVC, C_KC, C_VC, B_PM_T]
    dEnqueueG ptr S_PQUANT (← cdiv (← imul Tp (← iconst32 (QO / 32))) 256) NBLK [B_PATT, B_PXQ, B_PXS, B_PM_QA]
    dEnqueueG ptr S_PO (← imul (← iconst32 ((HH + GptOssPrefill.TR - 1) / GptOssPrefill.TR)) tp32) NBLK
      [W_OW, B_PXQ, B_PXS, B_PTMP, W_OB, B_PM_TP]
    let hGrid ← cdiv (← imul T32 (← iconst32 HH)) 256
    dEnqueueG ptr S_PADD hGrid NBLK [B_PX, B_PTMP, B_PM_TH]
    dEnqueueG ptr S_PRMS (← cdiv T32 8) NBLK [B_PX, W_MNORM, B_PXN, B_PM_T]
    dEnqueueG ptr S_PRLOG (← cdiv (← imul T32 (← iconst32 NE)) 8) NBLK [B_PXN, W_RW, W_RB, B_PLOG, B_PM_T]
    dEnqueueG ptr S_PQUANT (← cdiv (← imul Tp (← iconst32 (HH / 32))) 256) NBLK [B_PXN, B_PXQ, B_PXS, B_PM_QX]
    dEnqueueG ptr S_PRTOP (← cdiv T32 256) NBLK [B_PLOG, B_PCH, B_PGW, B_PM_T]
    dEnqueueG ptr S_PGROUP (← iconst32 1) NBLK [B_PCH, B_POFFS, B_PPAIR, B_PPOS, B_PM_T]
    -- **Every expert the chunk routed to, resident.** With the whole model on
    -- the card nothing is missing and the host does not wait; otherwise it
    -- reads the layer's expert offsets and brings in, in the launches' stream,
    -- what is missing.
    let nslot ← load32 (← iaddImm ptr DNSLOT_OFF)
    when .ult nslot (← iconst32 (NL * NE)) do
      Drv.sync
      let bOff ← load64 (← iaddImm ptr (dBindOff B_POFFS))
      let _ ← ext (.cuda .memcpyDtoH) %[← iaddImm ptr DP_EOFF_OFF, bOff, ← iconst64 (4 * (NE + 1))]
      let ne ← iconst64 NE
      forLoop ne fun e => do
        let at_ (i : V .i64) : Prog V L (V .i32) := do
          load32 (← iadd (← iaddImm ptr DP_EOFF_OFF) (← ishlImm i 2))
        when .ugt (← at_ (← iaddImm e 1)) (← at_ e) do dEnsurePM layer (← ireduce32 e)
    dEnqueueG ptr S_PGU (← imul tiles (← iconst32 (II / 64))) NBLK
      [B_POOL, B_SLOTOF, B_PM_LAYER, B_PXQ, B_PXS, B_PPAIR, B_POFFS, B_PHID]
    dEnqueueG ptr S_PQUANT (← cdiv (← imul nP (← iconst32 (II / 32))) 256) NBLK [B_PHID, B_PHQ, B_PHS, B_PM_QH]
    dEnqueueG ptr S_PDOWN (← imul tiles (← iconst32 ((HH + GptOssPrefill.TR - 1) / GptOssPrefill.TR))) NBLK
      [B_POOL, B_SLOTOF, B_PM_LAYER, B_PHQ, B_PHS, B_POFFS, B_PY]
    dEnqueueG ptr S_PCOMB hGrid NBLK [B_PX, B_PY, B_PGW, B_PPOS, B_PM_T]

/-- **A prompt**: the `n` tokens at `toks` (32-bit ids), at positions
    `start …`, through the model `PT` at a time (`dPrefillChunkM`), every
    position's keys and values into the cache; then the last token's row to
    `B_X`, as a step leaves it, and the head on it (`dHeadM`). Returns the token
    that follows. -/
def dPrefillM (toks n start : V .i64) : Prog V L (V .i32) := do
  let ptr ← basePtr
  let pt ← iconst64 PT
  let nCh ← udiv (← iaddImm n (PT - 1)) pt
  forLoop nCh fun c => do
    let c0 ← imul c pt
    let rem ← isub n c0
    let T ← select (← icmp .ugt rem pt) pt rem
    dPrefillChunkM (← iadd toks (← ishlImm c0 2)) T (← iadd start c0)
  let lastT ← isub n (← imul (← isub nCh (← iconst64 1)) pt)
  let rowB ← iconst64 (HH * 4)
  let src ← iadd (← load64 (← iaddImm ptr (dBindOff B_PX))) (← imul (← isub lastT (← iconst64 1)) rowB)
  let _ ← ext (.cuda .memcpyDtoD) %[← load64 (← iaddImm ptr (dBindOff B_X)), src, rowB]
  dHeadM (← ireduce32 (← isub (← iadd start n) (← iconst64 1)))

/-- **A whole chat turn, or one step of one.**

    `D_MODE` chooses. Zero is a single step at a caller-supplied token and
    position, which is what the reference comparison drives; one is a turn —
    prompt text in, generated text out, with the tokenizer, the prefill, the
    generation loop and the detokenizer all on this side of the boundary.

    There is one entry and no extras either way. That is the property the whole
    application was arranged around: a caller that has to tokenize for itself
    is a caller that has to agree with the model about what a token is. -/
def dMainFn : Prog V L Unit :=
  do
  let ptr ← basePtr
  let flag ← load32 (← iaddImm ptr DINIT_OFF)
  let zero32 ← iconst32 0
  when .eq flag zero32 (do
    dInitM
    let dp ← dataPtr
    storeI64 (← iaddImm dp D_PTOK) (← iaddImm ptr DT_PATHPTR)
    storeI64 (← iconst64 D_PATH_FIELD) (← iaddImm ptr DT_PATHCAP)
    TokenizerCommon.loadTokenizerM dTokMem
    storeI32 (← iconst32 1) (← iaddImm ptr DINIT_OFF))
  let dataPtr ← dataPtr
  let outPtr ← outPtr
  let mode ← load32 (← iaddImm dataPtr D_MODE)
  let _ ← ifte .eq mode zero32
    (do -- one step, at what the caller asked for
        let tok ← load32 dataPtr
        let pos ← load32 (← iaddImm dataPtr D_POS)
        let nxt ← dStepM tok pos
        storeI32 nxt outPtr
        let miss ← load32 (← iaddImm ptr DMISS_OFF)
        storeI32 miss (← iaddImm outPtr 4)
        pure %[])
    (els := ifte .eq mode (← iconst32 2)
    (do -- a prompt: the `D_NPRE` ids at `D_PRE`, from `D_STARTPOS`, all at
        -- once (`dPrefillM`), as far as the cache has room
        let start ← uload32_64 (← iaddImm dataPtr D_STARTPOS)
        let nAsk ← uload32_64 (← iaddImm dataPtr D_NPRE)
        let nCap ← iconst64 TMPL_MAX
        let n1 ← select (← icmp .ugt nAsk nCap) nCap nAsk
        let ctx ← uload32_64 (← iaddImm ptr DCTX_OFF)
        let room ← select (← icmp .ult start ctx) (← isub ctx start) (← iconst64 0)
        let n ← select (← icmp .ugt n1 room) room n1
        when .ne n (← iconst64 0) do
          let nxt ← dPrefillM (← iaddImm dataPtr D_PRE) n start
          storeI32 nxt outPtr
        let miss ← load32 (← iaddImm ptr DMISS_OFF)
        storeI32 miss (← iaddImm outPtr 4)
        pure %[])
    (do -- a turn: text in, text out
        let tlen ← uload32_64 (← iaddImm dataPtr D_TLEN)
        let stop ← load32 (← iaddImm dataPtr D_STOP)
        let maxNew ← uload32_64 (← iaddImm dataPtr D_MAXNEW)
        let textIn ← iaddImm ptr DT_TEXTIN
        let src ← iaddImm dataPtr D_TEXT
        forLoop tlen fun i => do
          istore8 (← uload8_64 (← iadd src i)) (← iadd textIn i)
        storeI64 tlen (← iaddImm ptr DT_TEXTLEN)
        PretokCommon.tokenizeTextM dTokMem dPretokMem
        -- the prompt, all at once a piece: the template's opening, the text,
        -- the template's close, each through the prefill (`dPrefillM`), which
        -- leaves the token after the last at `DT_NEXT`
        let nText ← load { ty := .i64 }
                      (← iaddImm ptr DT_OUTCOUNT)
        let nPre ← uload32_64 (← iaddImm dataPtr D_NPRE)
        let nPost ← uload32_64 (← iaddImm dataPtr D_NPOST)
        let preP ← iaddImm dataPtr D_PRE
        let postP ← iaddImm dataPtr D_POST
        let outTok ← iaddImm ptr DT_OUTTOK
        -- The turn begins where the last one stopped, so what it prefills is
        -- what it adds. The cache still holds everything before `start`.
        let start ← uload32_64 (← iaddImm dataPtr D_STARTPOS)
        let afterPre ← iadd start nPre
        let afterText ← iadd afterPre nText
        -- one prefill in a loop over the three, so its body is emitted once
        forLoop (← iconst64 3) fun k => do
          let pick (a b c : V .i64) : Prog V L (V .i64) := do
            select (← icmp .eq k (← iconst64 0)) a (← select (← icmp .eq k (← iconst64 1)) b c)
          let n ← pick nPre nText nPost
          when .ne n (← iconst64 0) do
            let _ ← dPrefillM (← pick preP outTok postP) n (← pick start afterPre afterText)
            pure ()
        let nPrompt ← iadd afterText nPost
        -- …and then the model's own output, until it stops or runs out of room
        let genTok ← iaddImm ptr DT_GENTOK
        let first ← load32 (← iaddImm ptr DT_NEXT)
        let gEx ← wloopL %[(← iconst64 0), nPrompt, (← uextend64 first)]
          (head := fun _ st => return (contIf .ult (st.head) maxNew, %[st.head], ()))
          (body := fun lbl st _ => do
            let g := st.head
            let pos := st.snd
            let cur := st.thd
            let cur32 ← ireduce32 cur
            storeI32 cur32 (← iadd genTok (← ishlImm g 2))
            let g1 ← iaddImm g 1
            when .eq cur32 stop (brk lbl %[g1])
            -- and stop at the cache, which is the other end of the budget: a
            -- turn is bounded by `maxNew` tokens *and* by the room left in the
            -- key cache, and only one of those is the caller's to set
            when .uge pos (← uload32_64 (← iaddImm ptr DCTX_OFF)) (brk lbl %[g1])
            let nxt ← dStepM cur32 (← ireduce32 pos)
            return %[g1, ← iaddImm pos 1, ← uextend64 nxt])
        let _ ← iconst64 0
        let nGen := gEx.head
        storeI64 nGen (← iaddImm ptr DT_GENCOUNT)
        -- back to bytes
        let tokBuf ← iaddImm ptr DT_TOKBUF
        forLoop nGen fun i => do
          storeI32 (← load32 (← iadd genTok (← ishlImm i 2)))
                   (← iadd tokBuf (← ishlImm i 2))
        storeI64 nGen (← iaddImm ptr DT_TOKCOUNT)
        TokenizerCommon.detokenizeM dTokMem
        let nBytes ← load { ty := .i64 }
                       (← iaddImm ptr DT_TEXTLEN)
        storeI32 (← ireduce32 nBytes) outPtr
        let miss ← load32 (← iaddImm ptr DMISS_OFF)
        storeI32 miss (← iaddImm outPtr 4)
        let textOut ← iaddImm ptr DT_TEXTOUT
        let dst ← iaddImm outPtr D_OUT_TEXT
        forLoop nBytes fun i => do
          istore8 (← uload8_64 (← iadd textOut i)) (← iadd dst i)
        storeI32 (← ireduce32 nText) (← iaddImm outPtr D_OUT_NTEXT)
        let tdst ← iaddImm outPtr D_OUT_TEXTTOK
        forLoop nText fun i => do
          storeI32 (← load32 (← iadd outTok (← ishlImm i 2)))
                   (← iadd tdst (← ishlImm i 2))
        storeI32 (← ireduce32 nGen) (← iaddImm outPtr D_OUT_NGEN)
        let gdst ← iaddImm outPtr D_OUT_GEN
        forLoop nGen fun i => do
          storeI32 (← load32 (← iadd genTok (← ishlImm i 2)))
                   (← iadd gdst (← ishlImm i 2))
        pure %[]))
  pure ()

def dShippedBodies : List Prog.Body := [ dMainFn ]


def dClifIR : Except String (List FuncData) :=
  Prog.program <|
    .ok noopFunction :: dShippedBodies.zipIdx.map
      (fun p => Prog.entry "main" (Prog.compileProg (p.2 + 1) p.1))

def u32le (v : Nat) : List UInt8 :=
  [ UInt8.ofNat (v % 256), UInt8.ofNat (v / 256 % 256)
  , UInt8.ofNat (v / 65536 % 256), UInt8.ofNat (v / 16777216 % 256) ]

def dSlotBytes (t : String) : List UInt8 :=
  let b := t.toUTF8.toList ++ [0]
  b ++ zeros (DSLOT - b.length)

def dInitialMemory : List UInt8 :=
  zeros DHOST_LEN_OFF ++ u32le D_IN_BYTES
    ++ zeros (DNAME_OFF - DHOST_LEN_OFF - 4) ++ ("main".toUTF8.toList ++ [0, 0, 0, 0])
    ++ [0, 0, 0x80, 0x3f] ++ [0, 0, 0, 0]
    ++ zeros (DPTX_OFF - DZERO_OFF - 4)
    ++ dPtx.flatMap dSlotBytes
    ++ zeros (DMEM_SIZE - DBIND_OFF)

def dSetup (clif : List FuncData) : Artifact := {
  functions := clif
  required_memory := DMEM_SIZE
  initial_memory := dInitialMemory
}

#eval LayoutScan.check "GptOss.Decode" [``dMemMap]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[ artifactEntry "gptoss_decode" (dSetup clif) ]

end GptOssDecode

def GptOss.Decode.main (args : List String) : IO Unit := do
  let clif ← Prog.orDie GptOssDecode.dClifIR
  emitArtifacts (← requireOutputDir args) (GptOssDecode.artifacts clif)

#eval ShipScan.check "GptOss.Decode" `GptOss.Decode.main

