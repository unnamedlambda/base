import Lean
import Std
import AlgorithmLib.Gen
import AlgorithmLib.ML
import GptOssKernels
import GptOssAttention
import TokenizerCommon
import PretokCommon
import LayoutScan
import ShipScan

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML AlgorithmLib.Host
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur
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
def PIECES : Nat := 6
def VOCAB : Nat := 201088
def HH : Nat := 2880
def II : Nat := 2880

/-- Slots the cache holds. A compile-time *bound*: the bind table is sized for
    this many and the loader creates however many the card turns out to have
    room for, which it asks at run time. -/
def NSLOT_MAX : Nat := 640

/-- What a caller who names no context gets: the depth this engine shipped with
    before the cache became a choice, so a driver that says nothing behaves as
    it always did. -/
def CAP_DEFAULT : Nat := 8192

/-- Bytes of each piece of an expert, in the order the converter writes them. -/
def pieceBytes : List Nat :=
  [ 2 * II * (HH / 2), 2 * II * (HH / 32), 2 * II * 4
  , HH * (II / 2), HH * (II / 32), HH * 4 ]
def ROW_BYTES : Nat := pieceBytes.foldl (· + ·) 0
def pieceOff (t : Nat) : Nat := ((pieceBytes.take t).foldl (· + ·) 0)

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

/-! ## PTX slots -/

def dPtx : List String :=
  [ ptxRmsNorm, ptxAdd, emitProvenKernelN "main" 3 0 ropeQEW
  , emitProvenKernelN "main" 3 0 ropeKEW
  , ptxKVStore (QO / 2) HDW, ptxKVStore (KO / 2) HDW
  , ptxSinkSoftmax, GptOssKernels.moduleFor GptOssKernels.narrowBf16
  , GptOssKernels.moduleFor GptOssKernels.gateUpSwigluGemv
  , GptOssKernels.moduleFor GptOssKernels.downGemvBias
  , GptOssKernels.moduleFor GptOssKernels.moeCombine4
  , GptOssKernels.moduleFor GptOssKernels.routerTop4
  , GptOssKernels.moduleFor GptOssKernels.argmaxLogits
  , GptOssKernels.moduleFor GptOssKernels.widenBf16
  , GptOssKernels.moduleFor GptOssKernels.sampleLogits
  , GptOssKernels.moduleFor GptOssKernels.fusedAttnTile
  , GptOssKernels.moduleFor GptOssKernels.fusedAttnCombine ]

def S_RMS := 0
def S_ADD := 1
def S_ROPEQ := 2
def S_ROPEK := 3
/-- Keys then values.  One pair, not two: the cache depth left the kernel and
    became `M_KVSTRIDE`, so a sliding layer and a full one launch the same
    program. -/
def S_KV := 4
def S_SOFTMAX := 6
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

    Appended rather than placed beside `S_SOFTMAX`, whose four-kernel path they
    replace, because a slot index is a position in `dPtx` and inserting one
    renumbers every launch. -/
def S_ATTTILE := 15
def S_ATTCOMB := 16

/-! ## Buffers

    The first block is *this layer's*: nine dense weights, a key cache and a
    value cache, and four experts of six pieces each. None of them is ever
    allocated to hold anything — the loop points them at the store. -/

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
def C_SLOT0 := 11                       -- 4 experts x 6 pieces
def B_X := C_SLOT0 + PIECES * TOPK
def B_XN := B_X + 1
def B_XNB := B_X + 2
def B_QKV := B_X + 3
def B_SC := B_X + 4
def B_PR := B_X + 5
def B_ATT := B_X + 6
def B_ATTB := B_X + 7
def B_TMP := B_X + 8
def B_ROPE := B_X + 9
def B_META := B_X + 10
def B_NMH := B_X + 11
def B_NMQ := B_X + 12
def B_RLOG := B_X + 13
def B_XM := B_X + 14
def B_GATES := B_X + 15
def B_MOUT := B_X + 16
def B_CHOSEN := B_X + 17
def B_LOGITS := B_X + 18
def B_LMHEAD := B_X + 19
def B_FNORM := B_X + 20
/-- The greedy token, alone in its own buffer.

    It could share `B_CHOSEN`, which has room; it must not. The runtime's
    copies assert that a transfer is the whole of a buffer, so reading four
    bytes out of sixteen is not a short read but a failure — and one that
    arrives as a panic from inside the driver rather than as a wrong answer. -/
def B_TOKEN := B_X + 21
/-- The embedding row as the table stores it: bf16, 5.7 KiB, widened on the
    device.  Its own buffer rather than a borrow of `B_XNB`, which is the same
    size and free at that moment: the two hold different things and a reader
    should not have to know the order of the step to see that. -/
def B_EMB := B_X + 22
/-- `[vocab, seed, 1/T as bits, ln(min_p) as bits]` for the sampling kernel. -/
def B_SAMP := B_X + 23
/-- The rotated q, k and v, narrowed to bf16 and packed two to a word.

    The cache holds bf16, so k and v arrive already narrowed; and q has to be
    narrowed with them because cuBLAS refuses a mixed operand pair, so the
    contraction against a bf16 cache takes a bf16 query. One pass over all
    three, immediately after the rotation. -/
def B_QKVB := B_X + 24
/-- The attention probabilities, likewise, for the contraction against a bf16
    value cache. -/
def B_PRB := B_X + 25
/-- How many words that second narrow covers: `NQ * seqLen / 2`, and so a
    number that follows the position rather than the geometry. -/
def B_NMP := B_X + 26
/-- How many words the first narrow covers, which is a constant: q, k and v
    together are `dQkvOut` floats however long the conversation is. -/
def B_NMQKV := B_X + 27
/-- Where a tile leaves its `(m, l, acc)` for the merge, and how many tiles
    there are -- which follows the position, so the kernel is told rather than
    generated for it. -/
def B_PART := B_X + 28
def B_NT := B_X + 29
def B_Y := B_X + 30                     -- four
def B_HID := B_Y + TOPK                 -- four
def DSTORE := B_HID + TOPK              -- 9 x NL dense weights
def KVSTORE := DSTORE + 9 * NL          -- 2 x NL caches
def SLOTSTORE := KVSTORE + 2 * NL       -- NSLOT_MAX x 6 expert pieces
def DNBUF := SLOTSTORE + PIECES * NSLOT_MAX

def dStoreBuf (layer k : Nat) : Nat := DSTORE + 9 * layer + k
def kvStoreBuf (layer j : Nat) : Nat := KVSTORE + 2 * layer + j
/-- The depth a layer's cache is *allowed*, which is what `dBufBytes` is about.

    What it is actually given is chosen at start-up and allocated then; this
    list fixes the number of buffers and says each is non-empty, and neither of
    those depends on the depth. Sizing it at the ceiling keeps the two claims
    the theorem below makes true without pretending the artifact knows how big
    the cache will be. -/
def capOf (layer : Nat) : Nat := if layer % 2 == 0 then CAP_SWA else CAP_MAX

def dBufBytes : List Nat :=
  (List.range 9).map (fun k => dKindBytes.getD k 0)        -- current layer: placeholders
    ++ [128, 128]
    ++ List.replicate (PIECES * TOPK) 128
    ++ [ HH * 4, HH * 4, HH * 2, dQkvOut * 4
       , 64 * CAP_MAX * 4, 64 * CAP_MAX * 4
       , (64 * 64) * 4, (64 * 64) * 2, HH * 4
       , 2 * ROPE_N * HALF * 4, 64 * 4, 4, 4
       , NE * 4, HH * 4, 32 * 4, HH * 4, TOPK * 4
       , VOCAB * 4, VOCAB * HH * 2, HH * 4, 4, HH * 2, 16
       , dQkvOut * 2, 64 * CAP_MAX * 2, 4, 4
       , GptOssKernels.attPartialFloats * 4 * 64 * GptOssKernels.ATT_TILES_MAX
       , 4 ]
    ++ List.replicate TOPK (HH * 4)
    ++ List.replicate TOPK (II * 4)
    ++ (List.range NL).flatMap (fun _ => (List.range 9).map (fun k => dKindBytes.getD k 0))
    ++ (List.range NL).flatMap (fun l => [8 * capOf l * 64 * 2, 8 * capOf l * 64 * 2])
    ++ (List.range NSLOT_MAX).flatMap (fun _ => pieceBytes)

theorem gptoss_decode_alloc_covers :
    dBufBytes.length = DNBUF ∧ dBufBytes.all (fun n => decide (0 < n)) = true := by
  native_decide

/-! ## Memory map -/

def DSLOT : Nat := 0x20000
def DPTX_OFF : Nat := 0x0100
def dSlotOff (i : Nat) : Nat := DPTX_OFF + i * DSLOT
def DBIND_OFF : Nat := dSlotOff dPtx.length
def dBindOff (i : Nat) : Nat := DBIND_OFF + 4 * i
def DLOCAL_OFF : Nat := DBIND_OFF + 4 * DNBUF
def DMETA_OFF : Nat := DLOCAL_OFF + 4 * 8
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
/-- A word to stage the narrow count in before it is uploaded.

    Its own slot rather than `DLOCAL_OFF`, which the very next call overwrites
    with buffer ids: the upload happens first and the copy is synchronous, so
    sharing it would work and would be a trap for whoever reorders these two
    lines. -/
def DNARROWN_OFF : Nat := DCTXSTRIDE_OFF + 8
/-! ### The tokenizer's corner

    A chat turn is text in and text out, so this program owns a tokenizer as
    well as a model. The bodies come from `TokenizerCommon` and `PretokCommon`;
    what is local is where their working arrays sit. -/

/-- The longest prompt or reply this program handles, in bytes. -/
def TEXT_MAX : Nat := 32768

def DTOK_BASE : Nat := DNARROWN_OFF + 8
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
def DT_TEXTIN : Nat := DTOK_BASE + 128
def DT_TEXTOUT : Nat := DT_TEXTIN + TEXT_MAX
def DT_TOKBUF : Nat := DT_TEXTOUT + TEXT_MAX
def DT_CPBUF : Nat := DT_TOKBUF + 4 * TEXT_MAX
def DT_CPBYTE : Nat := DT_CPBUF + 4 * TEXT_MAX
def DT_OUTTOK : Nat := DT_CPBYTE + 4 * (TEXT_MAX + 1)
def DT_GENTOK : Nat := DT_OUTTOK + 4 * TEXT_MAX
def DMEM_SIZE : Nat := DT_GENTOK + 4 * TEXT_MAX + 0x100

def dTokMem : TokenizerCommon.TokMem :=
  { htCtx := ContextSlots.ht, cudaCtx := ContextSlots.cuda
    pathPtr := DT_PATHPTR, bufPtr := DT_BUFPTR
    tokenBuf := DT_TOKBUF, tokenCount := DT_TOKCOUNT
    textIn := DT_TEXTIN, textOut := DT_TEXTOUT, textLen := DT_TEXTLEN
    htKey := DT_HTKEY, htVal := DT_HTVAL
    fileMaxBytes := 32 * 1024 * 1024, textMaxBytes := TEXT_MAX }

def dPretokMem : PretokCommon.PretokMem :=
  { cpBuf := DT_CPBUF, cpByte := DT_CPBYTE, cpCount := DT_CPCOUNT
    outTok := DT_OUTTOK, outCount := DT_OUTCOUNT }
def DHOST_LEN_OFF : Nat := 0x0080

theorem gptoss_decode_ptx_fits :
    dPtx.all (fun t => decide (t.utf8ByteSize + 1 ≤ DSLOT)) = true := by
  native_decide

def dMemMap : AlgorithmLib.Layout.RegionMap :=
  (List.range dPtx.length).map (fun i => ⟨s!"ptx{i}", dSlotOff i, DSLOT⟩)
    ++ [⟨"hostLen", DHOST_LEN_OFF, 4⟩, ⟨"bind", DBIND_OFF, 4 * DNBUF⟩,
        ⟨"local", DLOCAL_OFF, 4 * 8⟩, ⟨"meta", DMETA_OFF, 64 * 4⟩,
        ⟨"chosen", DCHOSEN_OFF, 4 * TOPK⟩,
        ⟨"slotOf", DSLOTOF_OFF, 4 * NL * NE⟩,
        ⟨"resident", DRESIDENT_OFF, 4 * NSLOT_MAX⟩,
        ⟨"clock", DCLOCK_OFF, 4⟩, ⟨"nslot", DNSLOT_OFF, 4⟩,
        ⟨"pool", DPOOL_OFF, 8⟩, ⟨"stage", DSTAGE_OFF, 8⟩,
        ⟨"miss", DMISS_OFF, 4⟩, ⟨"init", DINIT_OFF, 4⟩,
        ⟨"ctx", DCTX_OFF, 4⟩, ⟨"ctxStride", DCTXSTRIDE_OFF, 8⟩,
        ⟨"narrowN", DNARROWN_OFF, 4⟩,
        ⟨"tokPathPtr", DT_PATHPTR, 8⟩, ⟨"tokBufPtr", DT_BUFPTR, 8⟩,
        ⟨"tokCount", DT_TOKCOUNT, 8⟩, ⟨"tokTextLen", DT_TEXTLEN, 8⟩,
        ⟨"htKey", DT_HTKEY, 8⟩, ⟨"htVal", DT_HTVAL, 8⟩,
        ⟨"cpCount", DT_CPCOUNT, 8⟩, ⟨"outCount", DT_OUTCOUNT, 8⟩,
        ⟨"next", DT_NEXT, 4⟩, ⟨"genCount", DT_GENCOUNT, 8⟩,
        ⟨"textIn", DT_TEXTIN, TEXT_MAX⟩, ⟨"textOut", DT_TEXTOUT, TEXT_MAX⟩,
        ⟨"tokBuf", DT_TOKBUF, 4 * TEXT_MAX⟩,
        ⟨"cpBuf", DT_CPBUF, 4 * TEXT_MAX⟩,
        ⟨"cpByte", DT_CPBYTE, 4 * (TEXT_MAX + 1)⟩,
        ⟨"outTok", DT_OUTTOK, 4 * TEXT_MAX⟩,
        ⟨"genTok", DT_GENTOK, 4 * TEXT_MAX⟩]

theorem gptossDecodeMap_ok :
    dMemMap.okB = true ∧ dMemMap.withinB DMEM_SIZE = true := by native_decide


/-! ## The host program -/

def env : FnEnv := env% [.ht, .cuda, .cublas, .fileIO]

/-- What the caller passes: a token, its position, and where the bank is.

    Three paths and two integers, and nothing else — no weights, no embedding
    row, no numbers the model is made of. The paths are read once; the two
    integers are read every call. -/
def D_TOK : Nat := 0
def D_POS : Nat := 4
/-- Zero for a single step, one for a whole turn. -/
def D_MODE : Nat := 8
def D_TLEN : Nat := 12
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
def D_OUT_BYTES : Nat := D_OUT_TEXTTOK + 4 * TEXT_MAX

def NBLK : Nat := 32 * warpsPerCta

/-- The staging buffer: wide enough for the widest dense tensor, and every
    load that exceeds it is chunked rather than trusted to fit. -/
def STAGE_BYTES : Nat := dQkvOut * HH * 2

def dBindLocal (ptr : R) (bs : List Nat) : M Unit := do
  for (j, gb) in (List.range bs.length).zip bs do
    let id ← load32 (← absAddr ptr (dBindOff gb))
    storeI32 id (← absAddr ptr (DLOCAL_OFF + 4 * j))

/-- Enqueue at a PTX slot named by a *register*, which is what lets a layer
    choose its own kernel from its own index. -/
def dEnqueueAt (ptr : R) (ptxOff : R) (g blk : Nat) (bs : List Nat) : M Unit := do
  dBindLocal ptr bs
  let nBufs ← iconst32 bs.length
  let bindBase ← iconst64 DLOCAL_OFF
  let one ← iconst32 1
  let block ← iconst32 blk
  let grid ← iconst32 g
  let _ ← cudaLaunch ptr ptxOff nBufs bindBase grid one one block one one

/-- The same, with the **grid** in a register too.

    One launch needs it: narrowing the attention probabilities covers
    `NQ * seqLen` of them, which is a function of the position. Launching the
    largest grid that could ever be needed and having the threads past the count
    return would be sixteen thousand blocks doing nothing at the context this
    now supports. -/
def dEnqueueG (ptr : R) (ptxOff : R) (grid : R) (blk : Nat) (bs : List Nat) : M Unit := do
  dBindLocal ptr bs
  let nBufs ← iconst32 bs.length
  let bindBase ← iconst64 DLOCAL_OFF
  let one ← iconst32 1
  let block ← iconst32 blk
  let _ ← cudaLaunch ptr ptxOff nBufs bindBase grid one one block one one

def dEnqueue (ptr : R) (i g blk : Nat) (bs : List Nat) : M Unit := do
  dEnqueueAt ptr (← iconst64 (dSlotOff i)) g blk bs

/-- **Start-up.** Pin the expert file, take whatever device memory is going,
    and put the dense weights where they will stay. -/
def dInitM : M Unit := do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit ptr
  let ctxPtr ← cudaCtxPtr ptr
  -- the expert pool: pinned once, and never moved again
  let poolBytes ← iconst64 (NL * NE * ROW_BYTES)
  let poolId ← call IR.Ffi.cudaPinnedAlloc.id [ctxPtr, poolBytes]
  let poolPtr ← call IR.Ffi.cudaPinnedPtr.id [ctxPtr, poolId]
  -- The id, and only the id.  An eight-byte pointer written at `DPOOL_OFF + 4`
  -- runs four bytes into `stage`, and nothing reads it back: every use goes
  -- through `cudaPinnedPtrAt`, which takes the id and bounds-checks the offset.
  -- It survived only because `stage` is written immediately afterwards.
  storeI32 poolId (← absAddr ptr DPOOL_OFF)
  let zero64 ← iconst64 0
  let pExp ← iaddImm dataPtr D_PEXP
  let _ ← call IR.Ffi.fileReadToPtr.id [pExp, poolPtr, zero64, poolBytes]
  -- a staging buffer, big enough for the widest dense tensor
  let stageBytes ← iconst64 STAGE_BYTES
  let stageId ← call IR.Ffi.cudaPinnedAlloc.id [ctxPtr, stageBytes]
  let stagePtr ← call IR.Ffi.cudaPinnedPtr.id [ctxPtr, stageId]
  storeI64 stagePtr (← absAddr ptr DSTAGE_OFF)
  -- **The context, chosen here and fixed from here on.**
  --
  -- Zero means the default.  Anything else is clamped to the published
  -- rotation tables, which is the hard ceiling: past `CAP_MAX` there is no row
  -- to encode a position with.  The buffers below are allocated to whatever
  -- comes out, so this is the last moment it can be decided.
  let ctxAsk ← uload32_64 (← iaddImm dataPtr D_CTX)
  let ctxD ← ifte .eq ctxAsk zero64 (pure [← iconst64 CAP_DEFAULT]) (pure [ctxAsk])
  let ctxMax ← iconst64 CAP_MAX
  let ctxC ← ifte .ugt (ctxD.headD ctxMax) ctxMax (pure [ctxMax]) (pure [ctxD.headD ctxMax])
  let ctx := ctxC.headD ctxMax
  storeI32 (← ireduce32 ctx) (← absAddr ptr DCTX_OFF)
  let ctxStride ← imul ctx (← iconst64 HD)
  storeI64 ctxStride (← absAddr ptr DCTXSTRIDE_OFF)
  -- every buffer that is not an expert slot.  The caches are the exception:
  -- their size is the context, and a full layer's is what was just chosen.
  -- The caches hold bf16: two bytes an element, which is the whole point of
  -- narrowing before the store.
  -- one key row of slack on each cache: the fused attention loads the next key
  -- while it works on the current one, so the last iteration reads one row past
  -- the keys that exist. Cheaper than a guard inside the loop.
  let slack ← iconst64 (8 * HD * 2)
  let swaBytes ← iadd (← iconst64 (8 * CAP_SWA * HD * 2)) slack
  let fullBytes ← iadd (← imul ctxStride (← iconst64 (8 * 2))) slack
  -- one row of scores per query head per key, and the same again for the
  -- probabilities the softmax writes -- those stay f32, because the softmax is
  -- where the precision is worth keeping
  let scoreBytes ← imul ctx (← iconst64 (64 * 4))
  let probBf16Bytes ← imul ctx (← iconst64 (64 * 2))
  -- one `(m, l, acc)` a query head a tile.  A fixed size, because the tile
  -- size is chosen at run time so that the tile *count* never exceeds
  -- `ATT_TILES_MAX` however long the context is.
  let partBytes ← iconst64
    (GptOssKernels.attPartialFloats * 4 * NQ * GptOssKernels.ATT_TILES_MAX)
  for (i, nb) in (List.range SLOTSTORE).zip dBufBytes do
    let sz ←
      if KVSTORE ≤ i && i < SLOTSTORE then
        pure (if (i - KVSTORE) / 2 % 2 == 0 then swaBytes else fullBytes)
      else if i == B_SC || i == B_PR then pure scoreBytes
      else if i == B_PRB then pure probBf16Bytes
      else if i == B_PART then pure partBytes
      else iconst64 nb
    let id ← cudaCreateBuffer ptr sz
    storeI32 id (← absAddr ptr (dBindOff i))
  -- …and as many slots as the card turns out to have room for.  Asked, not
  -- assumed: a hardcoded count is a count that is wrong on the next card.
  let freeB ← call IR.Ffi.cudaMemInfoFree.id [ctxPtr]
  let margin ← iconst64 (256 * 1024 * 1024)
  let usable ← isub freeB margin
  let rowB ← iconst64 ROW_BYTES
  let want ← udiv usable rowB
  let cap ← iconst64 NSLOT_MAX
  let nslotL ← ifte .ugt want cap (pure [cap]) (pure [want])
  let nslot := nslotL.headD want
  storeI32 (← ireduce32 nslot) (← absAddr ptr DNSLOT_OFF)
  let six ← iconst64 PIECES
  let four ← iconst64 4
  forLoop nslot fun i => do
    let base6 ← imul i six
    for t in List.range PIECES do
      let sz ← iconst64 (pieceBytes.getD t 0)
      let id ← cudaCreateBuffer ptr sz
      let ix ← iaddImm base6 (SLOTSTORE + t)
      let off ← imul ix four
      storeI32 id (← iadd (← absAddr ptr DBIND_OFF) off)
  -- the dense weights: read from the file, staged, uploaded, and resident
  let pDen ← iaddImm dataPtr D_PDEN
  for l in List.range NL do
    for k in List.range 9 do
      let n := dKindBytes.getD k 0
      let fo ← iconst64 (denseOff l k)
      let nb ← iconst64 n
      let _ ← call IR.Ffi.fileReadToPtr.id [pDen, stagePtr, fo, nb]
      let id ← load32 (← absAddr ptr (dBindOff (dStoreBuf l k)))
      let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, stagePtr, nb]
  -- `lm_head` is 1.08 GiB and the staging buffer is 29.5 MiB, so it goes over
  -- in pieces.  Reading it in one call would write a gigabyte past the end of
  -- a pinned allocation -- not a wrong answer but a corrupted heap, and the
  -- expert pool is what sits next to it.
  for (off, n, b) in [(fileFinalNorm, HH * 4, B_FNORM),
                      (fileLmHead, VOCAB * HH * 2, B_LMHEAD)] do
    let id ← load32 (← absAddr ptr (dBindOff b))
    let chunks := (n + STAGE_BYTES - 1) / STAGE_BYTES
    for c in List.range chunks do
      let take := min STAGE_BYTES (n - c * STAGE_BYTES)
      let fo ← iconst64 (off + c * STAGE_BYTES)
      let nb ← iconst64 take
      let _ ← call IR.Ffi.fileReadToPtr.id [pDen, stagePtr, fo, nb]
      let bo ← iconst64 (c * STAGE_BYTES)
      let _ ← call IR.Ffi.cudaUploadOffset.id [ctxPtr, id, bo, stagePtr, nb]
  -- **The rotation tables: two slices, and sines first.**
  --
  -- The kernel indexes sine at zero and cosine at `ROPE_N * HALF`
  -- (`GptOssAttention.ropeCosIx`); the file has the opposite order and the
  -- full published height. So this is two reads, not one, and the order is
  -- the kernel's rather than the file's. See `ROPE_FILE_ROWS`.
  let ropeSlice := ROPE_N * HALF * 4
  let bRope ← load32 (← absAddr ptr (dBindOff B_ROPE))
  for (fo, bo) in [(fileRopeSin, 0), (fileRope, ropeSlice)] do
    let f ← iconst64 fo
    let nb ← iconst64 ropeSlice
    let _ ← call IR.Ffi.fileReadToPtr.id [pDen, stagePtr, f, nb]
    let b ← iconst64 bo
    let _ ← call IR.Ffi.cudaUploadOffset.id [ctxPtr, bRope, b, stagePtr, nb]
  -- the narrow kernel's two counts
  for (b, v) in [(B_NMH, HH / 2), (B_NMQ, (64 * 64) / 2),
                 (B_NMQKV, dQkvOut / 2)] do
    storeI32 (← iconst32 v) (← absAddr ptr DMETA_OFF)
    let id ← load32 (← absAddr ptr (dBindOff b))
    let n4 ← iconst64 4
    let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, (← absAddr ptr DMETA_OFF), n4]
  -- an empty cache: nothing is anywhere
  let minus1 ← iconst32 (-1)
  let nCache ← iconst64 (NL * NE)
  forLoop nCache fun i => do
    let off ← imul i four
    storeI32 minus1 (← iadd (← absAddr ptr DSLOTOF_OFF) off)
  let nMax ← iconst64 NSLOT_MAX
  forLoop nMax fun i => do
    let off ← imul i four
    storeI32 minus1 (← iadd (← absAddr ptr DRESIDENT_OFF) off)
  storeI32 (← iconst32 0) (← absAddr ptr DCLOCK_OFF)
  storeI32 (← iconst32 0) (← absAddr ptr DMISS_OFF)

/-- Where the loop parks the layer it is on: inside the meta region, past the
    slots the kernels read, so it costs no separate mapping. -/
def DLAYER_SLOT : Nat := 32

/-- **Point this layer's eleven buffers at this layer's weights.**

    The same six-integer-move idea the experts use, at a different table: nine
    dense tensors and the two halves of the cache. Nothing is copied and no
    kernel is re-emitted; a layer is a rebinding. -/
def dBindLayerM (layer : R) : M Unit := do
  let ptr := basePtr
  let four ← iconst64 4
  let nine ← iconst64 9
  let two ← iconst64 2
  let l64 ← uextend64 layer
  let base9 ← imul l64 nine
  for k in List.range 9 do
    let ix ← iaddImm base9 (DSTORE + k)
    let id ← load32 (← iadd (← absAddr ptr DBIND_OFF) (← imul ix four))
    storeI32 id (← absAddr ptr (dBindOff k))
  let base2 ← imul l64 two
  for j in List.range 2 do
    let ix ← iaddImm base2 (KVSTORE + j)
    let id ← load32 (← iadd (← absAddr ptr DBIND_OFF) (← imul ix four))
    storeI32 id (← absAddr ptr (dBindOff (C_KC + j)))

/-- **A cache lookup, and a miss if it is one.**

    `slotOf` says where an expert is, or that it is nowhere. When it is
    nowhere, the victim is whoever the clock hand is pointing at — round-robin,
    not least-recently-used, because at nine-tenths residency the two differ
    little and a scan of six hundred slots per miss would cost more than it
    saves. The six pieces come over from the pinned pool, both tables are
    corrected, and the miss is counted so the caller can see the rate.

    The transfer is the only thing in a decode step that touches the bus. -/
def dEnsureM (j : Nat) : M Unit := do
  let ptr := basePtr
  let four ← iconst64 4
  let six ← iconst64 PIECES
  let minus1 ← iconst32 (-1)
  let layer ← load32 (← absAddr ptr (DMETA_OFF + 4 * DLAYER_SLOT))
  let e ← load32 (← absAddr ptr (DCHOSEN_OFF + 4 * j))
  let ne64 ← iconst64 NE
  let key ← iadd (← imul (← uextend64 layer) ne64) (← uextend64 e)
  let slotAddr ← iadd (← absAddr ptr DSLOTOF_OFF) (← imul key four)
  let slot0 ← load32 slotAddr
  let res ← ifte .eq slot0 minus1
    (do
      -- the victim, and the hand moved on
      let clock ← load32 (← absAddr ptr DCLOCK_OFF)
      let nslot ← load32 (← absAddr ptr DNSLOT_OFF)
      let nxt ← iadd clock (← iconst32 1)
      let wrapped ← ifte .uge nxt nslot (pure [← iconst32 0]) (pure [nxt])
      storeI32 (wrapped.headD nxt) (← absAddr ptr DCLOCK_OFF)
      let v64 ← uextend64 clock
      -- whoever was there is no longer anywhere
      let resAddr ← iadd (← absAddr ptr DRESIDENT_OFF) (← imul v64 four)
      let old ← load32 resAddr
      when .ne old minus1 (do
        let oldAddr ← iadd (← absAddr ptr DSLOTOF_OFF) (← imul (← uextend64 old) four)
        storeI32 minus1 oldAddr)
      -- the transfer: six pieces, pinned host to device
      let ctxPtr ← cudaCtxPtr ptr
      let poolId ← load32 (← absAddr ptr DPOOL_OFF)
      let rowBase ← imul key (← iconst64 ROW_BYTES)
      let base6 ← imul v64 six
      for t in List.range PIECES do
        let len ← iconst64 (pieceBytes.getD t 0)
        let srcOff ← iaddImm rowBase (pieceOff t)
        let src ← call IR.Ffi.cudaPinnedPtrAt.id [ctxPtr, poolId, srcOff, len]
        let ix ← iaddImm base6 (SLOTSTORE + t)
        let id ← load32 (← iadd (← absAddr ptr DBIND_OFF) (← imul ix four))
        let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, id, src, len]
      storeI32 clock slotAddr
      storeI32 (← ireduce32 key) resAddr
      let m ← load32 (← absAddr ptr DMISS_OFF)
      storeI32 (← iadd m (← iconst32 1)) (← absAddr ptr DMISS_OFF)
      pure [clock])
    (pure [slot0])
  -- whichever it is, the four slots the kernels read now name its pieces
  let slot ← uextend64 (res.headD slot0)
  let sBase ← imul slot six
  for t in List.range PIECES do
    let ix ← iaddImm sBase (SLOTSTORE + t)
    let id ← load32 (← iadd (← absAddr ptr DBIND_OFF) (← imul ix four))
    storeI32 id (← absAddr ptr (dBindOff (C_SLOT0 + PIECES * j + t)))

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

    `half` is 0 after attention and 1 after the mixture. -/
def dTraceM (layer : R) (half : Nat) : M Unit := do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let bX ← load32 (← absAddr ptr (dBindOff B_X))
  let row ← iadd (← imul (← uextend64 layer) (← iconst64 2)) (← iconst64 half)
  let dst ← iadd outPtr
    (← iadd (← iconst64 (8 + VOCAB * 4)) (← imul row (← iconst64 (HH * 4))))
  let _ ← cudaSync ptr
  let _ ← call IR.Ffi.cudaDownload.id [ctxPtr, bX, dst, (← iconst64 (HH * 4))]

set_option maxRecDepth 8000 in
/-- **One layer, whichever layer it is.** -/
def dLayerM (layer : R) : M Unit := do
  let ptr := basePtr
  dBindLayerM layer
  storeI32 layer (← absAddr ptr (DMETA_OFF + 4 * DLAYER_SLOT))
  -- the pristine copies, not `M_SEQ`, which this body is about to overwrite
  let seqLen ← load32 (← absAddr ptr (DMETA_OFF + 4 * (M_SEQ + 16)))
  let seqLenF ← load32 (← absAddr ptr (DMETA_OFF + 4 * (M_SEQ + 8)))
  let zero32 ← iconst32 0
  let one32 ← iconst32 1
  let oneF ← iconst32 0x3F800000
  let scaleF ← iconst32 0x3E000000
  -- a sliding layer is an even one; what differs is its depth, not its kernels
  let par ← isub layer (← imul (← udiv layer (← iconst32 2)) (← iconst32 2))
  -- the effective length and the cache depth both follow from the layer type
  let lenSel ← ifte .eq par zero32 (pure [seqLen]) (pure [seqLenF])
  let sLen := lenSel.headD seqLen
  let sLen64 ← uextend64 sLen
  let ctxStrideL ← load { ty := .i64, notrapAligned := true }
                     (← absAddr ptr DCTXSTRIDE_OFF)
  let capSel ← ifte .eq par zero32
    (pure [← iconst64 (CAP_SWA * HD)]) (pure [ctxStrideL])
  let strideKV := capSel.headD ctxStrideL
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
  let posNow ← load32 (← absAddr ptr (DMETA_OFF + 4 * M_POS))
  let ch ← udiv sLen c32
  let tl ← imul ch c32
  let ctxNow ← load32 (← absAddr ptr DCTX_OFF)
  let capSlot ← ifte .eq par zero32 (pure [← iconst32 CAP_SWA]) (pure [ctxNow])
  let capV := capSlot.headD c32
  let slotNow ← isub posNow (← imul (← udiv posNow capV) capV)
  storeI32 sLen (← absAddr ptr (DMETA_OFF + 4 * M_SEQ))
  storeI32 ch (← absAddr ptr (DMETA_OFF + 4 * M_CHUNKS))
  storeI32 tl (← absAddr ptr (DMETA_OFF + 4 * M_TAIL))
  storeI32 (← isub sLen tl) (← absAddr ptr (DMETA_OFF + 4 * M_REM))
  storeI32 slotNow (← absAddr ptr (DMETA_OFF + 4 * M_SLOT))
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
    (pure [← iconst32 GptOssKernels.ATT_TILE_MIN]) (pure [even])
  let tileSz := tooSmall.headD even
  let nTiles ← udiv (← iadd sLen (← isub tileSz (← iconst32 1))) tileSz
  storeI32 tileSz (← absAddr ptr (DMETA_OFF + 4 * M_TILESZ))
  storeI32 nTiles (← absAddr ptr (DMETA_OFF + 4 * M_NTILES))
  -- The depth this layer's cache was allocated at, in the *words* the store
  -- kernel moves: `HDW` of them a head, because the cache is bf16 and two ride
  -- in one word.  The contractions below stride by `HD` elements over the same
  -- bytes, which is the same distance counted the other way.
  storeI32 (← imul capV (← iconst32 HDW)) (← absAddr ptr (DMETA_OFF + 4 * M_KVSTRIDE))
  let ctxM ← cudaCtxPtr ptr
  let bMetaL ← load32 (← absAddr ptr (dBindOff B_META))
  let _ ← call IR.Ffi.cudaUpload.id
    [ctxM, bMetaL, (← absAddr ptr DMETA_OFF), (← iconst64 (64 * 4))]
  let bKC ← load32 (← absAddr ptr (dBindOff C_KC))
  let bVC ← load32 (← absAddr ptr (dBindOff C_VC))
  let bQKV ← load32 (← absAddr ptr (dBindOff B_QKV))
  let bSC ← load32 (← absAddr ptr (dBindOff B_SC))
  let bPR ← load32 (← absAddr ptr (dBindOff B_PR))
  let bATT ← load32 (← absAddr ptr (dBindOff B_ATT))
  let bXNB ← load32 (← absAddr ptr (dBindOff B_XNB))
  let bATTB ← load32 (← absAddr ptr (dBindOff B_ATTB))
  let bQKVW ← load32 (← absAddr ptr (dBindOff W_QKVW))
  let bOW ← load32 (← absAddr ptr (dBindOff W_OW))
  let bTMP ← load32 (← absAddr ptr (dBindOff B_TMP))
  let bRW ← load32 (← absAddr ptr (dBindOff W_RW))
  let bXM ← load32 (← absAddr ptr (dBindOff B_XM))
  let bRLOG ← load32 (← absAddr ptr (dBindOff B_RLOG))
  dEnqueue ptr S_RMS 1 32 [B_X, W_ANORM, B_XN]
  dEnqueue ptr S_NARROW ((HH / 2 + NBLK - 1) / NBLK) NBLK [B_XN, B_XNB, B_NMH]
  let mQKV ← iconst32 dQkvOut
  let kH ← iconst32 HH
  let _ ← cublasGemmExBf16 ptr one32 zero32 mQKV one32 kH oneF bQKVW bXNB zero32 bQKV
  dEnqueue ptr S_ADD (dQkvOut / 32) 32 [B_QKV, W_QKVB]
  dEnqueue ptr S_ROPEQ NQ 32 [B_QKV, B_META, B_ROPE]
  dEnqueue ptr S_ROPEK NKV 32 [B_QKV, B_META, B_ROPE]
  -- **Narrow once, and everything downstream is bf16.**
  --
  -- The cache holds bf16, so k and v have to be narrowed before they are
  -- stored; and q has to be narrowed with them, because cuBLAS refuses a mixed
  -- operand pair and the contraction is against that cache. One pass over all
  -- three rather than two passes and an exception.
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
  dEnqueueG ptr (← iconst64 (dSlotOff S_ATTTILE)) nTiles NBLK
    [C_KC, C_VC, B_QKVB, B_PART, B_META]
  -- the merge is one warp a query head, and the sink joins here: it belongs to
  -- the head, and a tile that applied it would have it counted once per tile
  dEnqueue ptr S_ATTCOMB NQ 32 [B_PART, W_SINKS, B_ATT, B_META]
  dEnqueue ptr S_NARROW (((64 * 64) / 2 + NBLK - 1) / NBLK) NBLK [B_ATT, B_ATTB, B_NMQ]
  let mH ← iconst32 HH
  let kQO ← iconst32 (64 * 64)
  let _ ← cublasGemmExBf16 ptr one32 zero32 mH one32 kQO oneF bOW bATTB zero32 bTMP
  dEnqueue ptr S_ADD (HH / 32) 32 [B_TMP, W_OB]
  dEnqueue ptr S_ADD (HH / 32) 32 [B_X, B_TMP]
  dTraceM layer 0
  -- route
  dEnqueue ptr S_RMS 1 32 [B_X, W_MNORM, B_XM]
  let ne32 ← iconst32 NE
  let _ ← cublasSgemv ptr one32 kH ne32 oneF bRW bXM zero32 bRLOG
  dEnqueue ptr S_ADD 1 32 [B_RLOG, W_RB]
  dEnqueue ptr S_TOP4 1 32 [B_RLOG, B_GATES, B_CHOSEN]
  let _ ← cudaSync ptr
  let ctxPtr ← cudaCtxPtr ptr
  let bCh ← load32 (← absAddr ptr (dBindOff B_CHOSEN))
  let n16 ← iconst64 (TOPK * 4)
  let _ ← call IR.Ffi.cudaDownload.id
    [ctxPtr, bCh, (← absAddr ptr DCHOSEN_OFF), n16]
  -- the cache, four times, and then the mixture
  for j in List.range TOPK do
    dEnsureM j
  for j in List.range TOPK do
    dEnqueue ptr S_GATEUP (II / (warpsPerCta * rowsPerWarpGateUp)) NBLK
      [C_SLOT0 + PIECES * j, C_SLOT0 + PIECES * j + 1, C_SLOT0 + PIECES * j + 2,
       B_XM, B_HID + j]
    dEnqueue ptr S_DOWN (HH / (warpsPerCta * rowsPerWarpDown)) NBLK
      [C_SLOT0 + PIECES * j + 3, C_SLOT0 + PIECES * j + 4, C_SLOT0 + PIECES * j + 5,
       B_HID + j, B_Y + j]
  dEnqueue ptr S_COMBINE ((HH + NBLK - 1) / NBLK) NBLK
    [B_Y, B_Y + 1, B_Y + 2, B_Y + 3, B_GATES, B_MOUT]
  dEnqueue ptr S_ADD (HH / 32) 32 [B_X, B_MOUT]
  dTraceM layer 1

set_option maxRecDepth 8000 in
/-- **One token, from a token and a position.**

    Everything a decode step is: the embedding row, the meta this position
    implies, twenty-four layers, the final norm, `lm_head`, and the largest
    logit's index. Returns that index, and leaves the logits and the layer
    trace in the caller's buffer where a checker can read them.

    A builder rather than an entry point, because a chat turn is this in a
    loop and the loop belongs on the same side of the boundary as the model. -/
def dStepM (tok posIn : R) : M R := do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let ctxPtr ← cudaCtxPtr ptr
  -- **No step ever runs past the cache, whatever the caller asked for.**
  --
  -- The cache is a ring and a modulus cannot leave its buffer, so what bounds
  -- a position is the *allocation*: the full layers were given room for
  -- `DCTX_OFF` of them, and a step past that would write a slot the layer does
  -- not own. Clamping is not the answer to a conversation that has run out of
  -- room — the caller is expected to refuse the turn, and the generation loop
  -- below stops on its own — but it is what makes the failure a stalled reply
  -- rather than a write into whatever was allocated next.
  let capF ← isub (← load32 (← absAddr ptr DCTX_OFF)) (← iconst32 1)
  let posL ← ifte .ugt posIn capF (pure [capF]) (pure [posIn])
  let pos := posL.headD posIn
  -- **The embedding row, gathered here.**
  --
  -- The table is bf16 and the first RMSNorm reads f32, so the row is read out
  -- of the file at `token * HH * 2`, staged, uploaded as the 5.7 KiB it is,
  -- and widened on the device. The alternative that needs no kernel is to hold
  -- all 201088 rows in device memory, which costs 1.08 GiB — a hundred and
  -- eighty expert slots — to save a read of under six kilobytes.
  let bEmb ← load32 (← absAddr ptr (dBindOff B_EMB))
  let stagePtrT ← load64 (← absAddr ptr DSTAGE_OFF)
  let rowB ← iconst64 (HH * 2)
  let rowOff ← imul (← uextend64 tok) rowB
  let pEmb ← iaddImm dataPtr D_PEMB
  let _ ← call IR.Ffi.fileReadToPtr.id [pEmb, stagePtrT, rowOff, rowB]
  let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, bEmb, stagePtrT, rowB]
  dEnqueue ptr S_WIDEN ((HH / 2 + NBLK - 1) / NBLK) NBLK [B_EMB, B_X, B_NMH]
  -- the meta both cache depths need
  let one32 ← iconst32 1
  let len ← iadd pos one32
  let c128 ← iconst32 CAP_SWA
  let cFull ← load32 (← absAddr ptr DCTX_OFF)
  let swaL ← ifte .ugt len c128 (pure [c128]) (pure [len])
  let fullL ← ifte .ugt len cFull (pure [cFull]) (pure [len])
  let sL := swaL.headD len
  let fL := fullL.headD len
  storeI32 pos (← absAddr ptr (DMETA_OFF + 4 * M_POS))
  storeI32 sL (← absAddr ptr (DMETA_OFF + 4 * M_SEQ))
  let c32 ← iconst32 32
  let ch ← udiv sL c32
  storeI32 ch (← absAddr ptr (DMETA_OFF + 4 * M_CHUNKS))
  let tl ← imul ch c32
  storeI32 tl (← absAddr ptr (DMETA_OFF + 4 * M_TAIL))
  storeI32 (← isub sL tl) (← absAddr ptr (DMETA_OFF + 4 * M_REM))
  let slotSwa ← isub pos (← imul (← udiv pos c128) c128)
  storeI32 slotSwa (← absAddr ptr (DMETA_OFF + 4 * M_SLOT))
  storeI32 fL (← absAddr ptr (DMETA_OFF + 4 * (M_SEQ + 8)))
  -- Both lengths again, in slots no layer writes.  `M_SEQ` is what the softmax
  -- reads, so each layer publishes its own length there; that makes the slot
  -- unusable as the *source* of either length, because a full layer would
  -- otherwise hand its length to the next sliding one.  Below the window the
  -- two are equal and the fault is invisible, which is exactly why it needs
  -- somewhere pristine to be read from.
  storeI32 sL (← absAddr ptr (DMETA_OFF + 4 * (M_SEQ + 16)))
  let bMeta ← load32 (← absAddr ptr (dBindOff B_META))
  let mb ← iconst64 (64 * 4)
  let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, bMeta, (← absAddr ptr DMETA_OFF), mb]
  -- twenty-four layers, one body
  let nl ← iconst64 NL
  forLoop nl fun l => do
    dLayerM (← ireduce32 l)
  -- the head
  dEnqueue ptr S_RMS 1 32 [B_X, B_FNORM, B_XN]
  dEnqueue ptr S_NARROW ((HH / 2 + NBLK - 1) / NBLK) NBLK [B_XN, B_XNB, B_NMH]
  let bLM ← load32 (← absAddr ptr (dBindOff B_LMHEAD))
  let bXNB ← load32 (← absAddr ptr (dBindOff B_XNB))
  let bLog ← load32 (← absAddr ptr (dBindOff B_LOGITS))
  let oneF ← iconst32 0x3F800000
  let zz ← iconst32 0
  let oo ← iconst32 1
  let mV ← iconst32 VOCAB
  let kH ← iconst32 HH
  let _ ← cublasGemmExBf16 ptr oo zz mV oo kH oneF bLM bXNB zz bLog
  storeI32 (← iconst32 VOCAB) (← absAddr ptr (DMETA_OFF + 4 * 33))
  let bNM ← load32 (← absAddr ptr (dBindOff B_NMQ))
  let n4 ← iconst64 4
  let _ ← call IR.Ffi.cudaUpload.id
    [ctxPtr, bNM, (← absAddr ptr (DMETA_OFF + 4 * 33)), n4]
  -- **Greedy, or a draw from the tempered softmax.**
  --
  -- `1/T` of zero means the caller wants the largest logit, and that goes to
  -- the argmax kernel rather than to the sampler at an infinite temperature.
  -- Everything else goes to `sample_logits`, which adds Gumbel noise and takes
  -- the argmax of that — the same draw, in the same one pass.
  --
  -- The choice is arithmetic on the slot, not a branch around the launch:
  -- `ptxOff` is a register, so `S_ARGMAX + (sampling ? 1 : 0)` picks the
  -- kernel and the launch itself is the same instruction either way.
  let invT ← load32 (← iaddImm dataPtr D_INVT)
  storeI32 (← iconst32 VOCAB) (← absAddr ptr (DMETA_OFF + 4 * 34))
  -- the seed advances with the position, so a turn is reproducible from one
  -- number and no two steps draw the same noise
  let seed0 ← load32 (← iaddImm dataPtr D_SEED)
  storeI32 (← iadd seed0 (← imul pos (← iconst32 0x9E3779B1)))
    (← absAddr ptr (DMETA_OFF + 4 * 35))
  storeI32 invT (← absAddr ptr (DMETA_OFF + 4 * 36))
  storeI32 (← load32 (← iaddImm dataPtr D_LNMINP))
    (← absAddr ptr (DMETA_OFF + 4 * 37))
  let bSamp ← load32 (← absAddr ptr (dBindOff B_SAMP))
  let n12 ← iconst64 16
  let _ ← call IR.Ffi.cudaUpload.id
    [ctxPtr, bSamp, (← absAddr ptr (DMETA_OFF + 4 * 34)), n12]
  let sampling ← ifte .ne invT zz (pure [oo]) (pure [zz])
  let headSlot ← iadd (← iconst64 (dSlotOff S_ARGMAX))
                      (← imul (← uextend64 (sampling.headD zz))
                              (← iconst64 ((S_SAMPLE - S_ARGMAX) * DSLOT)))
  dEnqueueAt ptr headSlot 1 32 [B_LOGITS, B_TOKEN, B_SAMP]
  let _ ← cudaSync ptr
  -- back to the narrow count, so the next token's projection is right again
  storeI32 (← iconst32 ((64 * 64) / 2)) (← absAddr ptr (DMETA_OFF + 4 * 33))
  let _ ← call IR.Ffi.cudaUpload.id
    [ctxPtr, bNM, (← absAddr ptr (DMETA_OFF + 4 * 33)), n4]
  -- the token this step produced, into memory this program owns
  let bTok ← load32 (← absAddr ptr (dBindOff B_TOKEN))
  let _ ← call IR.Ffi.cudaDownload.id [ctxPtr, bTok, (← absAddr ptr DT_NEXT), n4]
  -- …and the whole logit row, to a fixed place in the caller's buffer.  Eight
  -- hundred kilobytes is real traffic and it buys the only check that can
  -- localise a fault in the head: a caller can compare against the model's own
  -- logits instead of guessing from a token id.  A partial copy is not an
  -- option -- the runtime asserts a transfer is the whole buffer -- so it is
  -- all of them or none.
  let outPtr ← load64 (← absAddr ptr 0x28)
  let _ ← call IR.Ffi.cudaDownload.id [ctxPtr, bLog, (← iaddImm outPtr 8), (← iconst64 (VOCAB * 4))]
  load32 (← absAddr ptr DT_NEXT)


/-- **A whole chat turn, or one step of one.**

    `D_MODE` chooses. Zero is a single step at a caller-supplied token and
    position, which is what the reference comparison drives; one is a turn —
    prompt text in, generated text out, with the tokenizer, the prefill, the
    generation loop and the detokenizer all on this side of the boundary.

    There is one entry and no extras either way. That is the property the whole
    application was arranged around: a caller that has to tokenize for itself
    is a caller that has to agree with the model about what a token is. -/
def dMainFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let flag ← load32 (← absAddr ptr DINIT_OFF)
  let zero32 ← iconst32 0
  when .eq flag zero32 (do
    dInitM
    let dp ← load64 (← absAddr ptr 0x18)
    storeI64 (← iaddImm dp D_PTOK) (← absAddr ptr DT_PATHPTR)
    TokenizerCommon.loadTokenizerM dTokMem
    storeI32 (← iconst32 1) (← absAddr ptr DINIT_OFF))
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let outPtr ← load64 (← absAddr ptr 0x28)
  let mode ← load32 (← iaddImm dataPtr D_MODE)
  let _ ← ifte .eq mode zero32
    (do -- one step, at what the caller asked for
        let tok ← load32 dataPtr
        let pos ← load32 (← iaddImm dataPtr D_POS)
        let nxt ← dStepM tok pos
        storeI32 nxt outPtr
        let miss ← load32 (← absAddr ptr DMISS_OFF)
        storeI32 miss (← iaddImm outPtr 4)
        pure ([] : List R))
    (do -- a turn: text in, text out
        let tlen ← uload32_64 (← iaddImm dataPtr D_TLEN)
        let stop ← load32 (← iaddImm dataPtr D_STOP)
        let maxNew ← uload32_64 (← iaddImm dataPtr D_MAXNEW)
        let textIn ← iaddImm ptr DT_TEXTIN
        let src ← iaddImm dataPtr D_TEXT
        forLoop tlen fun i => do
          istore8 (← uload8_64 (← iadd src i)) (← iadd textIn i)
        storeI64 tlen (← absAddr ptr DT_TEXTLEN)
        PretokCommon.tokenizeTextM dTokMem dPretokMem
        -- the prompt, one position at a time.  There is no prefill kernel yet,
        -- so this is the decode path run over the prompt: correct, and bound by
        -- launches rather than by arithmetic.
        let nText ← load { ty := .i64, notrapAligned := true }
                      (← absAddr ptr DT_OUTCOUNT)
        let nPre ← uload32_64 (← iaddImm dataPtr D_NPRE)
        let nPost ← uload32_64 (← iaddImm dataPtr D_NPOST)
        let preP ← iaddImm dataPtr D_PRE
        let postP ← iaddImm dataPtr D_POST
        let outTok ← iaddImm ptr DT_OUTTOK
        -- The turn begins where the last one stopped, so what it prefills is
        -- what it adds. The cache still holds everything before `start`.
        let start ← uload32_64 (← iaddImm dataPtr D_STARTPOS)
        forLoop nPre fun i => do
          let t ← load32 (← iadd preP (← ishlImm i 2))
          let _ ← dStepM t (← ireduce32 (← iadd start i))
          pure ()
        let afterPre ← iadd start nPre
        forLoop nText fun i => do
          let t ← load32 (← iadd outTok (← ishlImm i 2))
          let _ ← dStepM t (← ireduce32 (← iadd afterPre i))
          pure ()
        let afterText ← iadd afterPre nText
        forLoop nPost fun i => do
          let t ← load32 (← iadd postP (← ishlImm i 2))
          let _ ← dStepM t (← ireduce32 (← iadd afterText i))
          pure ()
        let nPrompt ← iadd afterText nPost
        -- …and then the model's own output, until it stops or runs out of room
        let genTok ← iaddImm ptr DT_GENTOK
        let first ← load32 (← absAddr ptr DT_NEXT)
        let gEx ← wloop [(← iconst64 0), nPrompt, (← uextend64 first)]
          (head := fun st => return (contIf .ult (st.headD 0) maxNew, [st.headD 0], ()))
          (body := fun st _ => do
            let g := st.headD 0
            let pos := st.getD 1 0
            let cur := st.getD 2 0
            let cur32 ← ireduce32 cur
            storeI32 cur32 (← iadd genTok (← ishlImm g 2))
            let g1 ← iaddImm g 1
            when .eq cur32 stop (brk [g1])
            -- and stop at the cache, which is the other end of the budget: a
            -- turn is bounded by `maxNew` tokens *and* by the room left in the
            -- key cache, and only one of those is the caller's to set
            when .uge pos (← uload32_64 (← absAddr ptr DCTX_OFF)) (brk [g1])
            let nxt ← dStepM cur32 (← ireduce32 pos)
            return [g1, ← iaddImm pos 1, ← uextend64 nxt])
        let nGen := gEx.headD (← iconst64 0)
        storeI64 nGen (← absAddr ptr DT_GENCOUNT)
        -- back to bytes
        let tokBuf ← iaddImm ptr DT_TOKBUF
        forLoop nGen fun i => do
          storeI32 (← load32 (← iadd genTok (← ishlImm i 2)))
                   (← iadd tokBuf (← ishlImm i 2))
        storeI64 nGen (← absAddr ptr DT_TOKCOUNT)
        TokenizerCommon.detokenizeM dTokMem
        let nBytes ← load { ty := .i64, notrapAligned := true }
                       (← absAddr ptr DT_TEXTLEN)
        storeI32 (← ireduce32 nBytes) outPtr
        let miss ← load32 (← absAddr ptr DMISS_OFF)
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
        pure ([] : List R))
  pure ()

def dShippedBodies : List HProg.Code := [ dMainFn ]

theorem gptossDecodeShipped_wf :
    dShippedBodies.all (HProg.wf env HProg.ptrParams) = true := by
  native_decide

def dClifIR : Program :=
  program <|
    noopFunction :: dShippedBodies.attach.zipIdx.map
      (fun p =>
        HProg.compileFn (p.2 + 1) p.1.1 env
          (hwf := List.all_eq_true.mp gptossDecodeShipped_wf p.1.1 p.1.2))

def u32le (v : Nat) : List UInt8 :=
  [ UInt8.ofNat (v % 256), UInt8.ofNat (v / 256 % 256)
  , UInt8.ofNat (v / 65536 % 256), UInt8.ofNat (v / 16777216 % 256) ]

def dSlotBytes (t : String) : List UInt8 :=
  let b := t.toUTF8.toList ++ [0]
  b ++ zeros (DSLOT - b.length)

def dInitialMemory : List UInt8 :=
  zeros DHOST_LEN_OFF ++ u32le D_IN_BYTES
    ++ zeros (DPTX_OFF - DHOST_LEN_OFF - 4)
    ++ dPtx.flatMap dSlotBytes
    ++ zeros (DMEM_SIZE - DBIND_OFF)

def dSetup : Setup := {
  clif := dClifIR
  memory_size := DMEM_SIZE
  initial_memory := dInitialMemory
}

#eval LayoutScan.check "GptOssDecode" [``dMemMap]

def artifacts : Array Json :=
  #[ toJsonArtifact "gptoss_decode" dSetup { fn_idx := u32 1 } [] ]

end GptOssDecode

def main (args : List String) : IO Unit := do
  emitArtifacts (← requireOutputDir args) GptOssDecode.artifacts

#eval ShipScan.check "GptOssDecode"

