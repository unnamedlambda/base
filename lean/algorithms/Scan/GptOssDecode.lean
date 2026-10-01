import Scan.GptOssSurface
import GptOss.Decode

/-!
  # What the whole-model artifact claims, and what it only arranges

  `GptOssScan` scans the pieces: the attention kernels, the mixture, one layer.
  This scans the thing those pieces were for — `gptoss_decode`, a single CLIF
  function that takes a chat turn's text and returns the model's reply. The
  tokenizer, the split, the prefill, twenty-four layers a token, the expert
  cache, the head and the detokenizer are all inside it; there is no second
  entry point and no host in the loop.

  The distinction that matters here is between a claim and an arrangement.
  Everything below is a claim about *shape*: that the buffers the program names
  are allocated, that each PTX text fits the slot the program will read it out
  of, that the regions of the program's own memory do not overlap and stay
  inside it, that every shipped body is well-formed in its environment. Those
  are worth stating because each has been wrong here before, and each is
  decidable over data the generator actually emitted.

  None of them is a claim about *values*. That `gptoss_decode` computes
  gpt-oss-20b is not proven by anything in this file, and the ledger in
  `GptOssScan.openObligations` is where that debt is written down. What this
  file adds to that debt is listed in `decodeOpenObligations`, and it is the
  part the per-piece artifacts could not have: the cache, the depth, the head.
-/

open Lean

namespace GptOssDecodeScan

/-- **The claims the whole-model artifact makes.**

    All three are guards on emitted data. They are the strongest statements
    that can be made about this artifact today, and they are all structural. -/
def roots : List Name :=
  [ ``GptOssDecode.gptoss_decode_alloc_covers
  , ``GptOssDecode.gptoss_decode_ptx_fits
  , ``GptOssDecode.gptossDecodeMap_ok ]

/-- **What running the whole model rests on that running one layer did not.**

    `GptOssScan.openObligations` carries the per-kernel debt — the MXFP4 GEMVs,
    the sink softmax, the vendor contraction. These are the obligations that
    only exist because the model is now assembled: a program that picks its own
    weights, evicts its own experts and stacks its own layers can be wrong in
    ways no single layer could be.

    Each entry says what is relied on, not what is suspected. Entries leave by
    being proven. -/
def decodeOpenObligations : List String :=
  [ "LayerWeightsSelected — the layer body computes its weight buffer ids from \
     the loop index, so twenty-four layers share one body. That the id it \
     computes for layer l is the buffer holding layer l's weights is checked by \
     construction and tested end to end; it is not stated. An off-by-one here \
     is a model that runs at full speed and means nothing."
  , "CacheDepthIsWhatWasAllocated — a sliding layer and a full layer launch the \
     same store kernel and differ only in the depth published in M_KVSTRIDE, \
     which the layer computes from its own parity and the context the engine \
     was started with. That the depth it publishes is the depth that layer's \
     cache was actually allocated at is arithmetic in two places — the \
     allocation loop in dInitM and the meta write in dLayerM — and is not \
     stated. Getting it wrong writes one head's keys over another's rather \
     than out of bounds, so it would read as a quality problem."
  , "FusedAttentionValue — scores, softmax and the value mix are one kernel \
     (GptOssKernels.fusedAttnTile) plus a merge, and the scores never reach \
     memory. What it replaced was four launches of which the softmax was a \
     *proven* EW kernel with a soundness theorem from raw launch; that theorem \
     still holds and nothing launches the kernel it is about. So this is a \
     proof surrendered, not merely one not yet written, and it is the price of \
     a token at 98304 keys going from 80.4 ms to 40.4. What is claimed instead \
     is checked: the online softmax is an exact refactoring of the same sum, \
     compared against a reference that materialises the scores at eight \
     lengths on and off tile boundaries, including one whose scores overflow \
     exp outright (applications/gpt-oss/attn_fused_test.py, worst case \
     4.0e-06), and end to end against the bf16 reference at 2.5e-03. Closes \
     by: the machine growing an online-softmax form, or a proof of the merge \
     identity (two (m,l,acc) triples combine by rescaling) applied to the \
     emitted body."
  , "AttnCacheSlack — the tile kernel loads the next key while working on the \
     current one, so its last iteration reads one key row past the keys that \
     exist. dInitM allocates a row of slack on every cache for exactly this. \
     That the slack is always there, for both cache depths and at every \
     context, is arithmetic in the allocation loop rather than a theorem — and \
     a guard inside the loop was rejected on cost, so nothing checks it at run \
     time. The standalone harness models the same contract, which is how it \
     was found."
  , "RingIsTheWindow — instantiated for the decode's geometry in \
     GptOssScan.SwaRingIsTheWindow, but the decode's own use of it (position p \
     lands in slot p % 128, and every resident slot is in the window) is not."
  , "CacheServesTheRightExpert — the miss path writes six pieces into a slot \
     and rewrites slotOf/resident. That a hit therefore reads the expert the \
     router asked for, and that eviction never leaves slotOf pointing at a slot \
     holding something else, is the cache's whole correctness and is untested \
     against an adversarial schedule. The round-robin clock makes it simple; \
     simple is not proven."
  , "SlotCountFitsDevice — the number of slots comes from cudaMemInfoFree at \
     start-up, so it is not a constant anyone can check. That the slots it \
     decides on plus the dense weights plus the caches fit the device is \
     enforced by arithmetic on a value read at run time."
  , "HeadNarrowsThenProjects — the hidden state is rounded to bf16 before \
     lm_head, and the logits are compared against a reference that models that \
     rounding (applications/gpt-oss/check.py). The rounding is GptOssKernels.\
     rneBf16; that it is nearest-even, and that the projection over the narrowed \
     vector is the model's, are both tested and neither is stated."
  , "FileSliceArithmetic — every tensor this program uploads is located in \
     dense.bin by hardcoded arithmetic (LAYER_STRIDE, dKindOff, fileLmHead, \
     fileRope, ROPE_FILE_ROWS) and in experts.bin by (layer * NE + e) * \
     ROW_BYTES plus pieceOff. The converter is the only other place that \
     layout is written down, and nothing checks the two agree. This is the \
     entry that has already been wrong: the rotation tables are stored at the \
     checkpoint's full height and cosine-first, while this program wanted \
     ROPE_N rows sine-first, so one read got cosines into both halves and every \
     query and key in the model came out rotated. Evidence today: the offsets \
     are diffed against the bank's own metadata, and the per-half-layer trace \
     is compared to the reference (applications/gpt-oss/check.py). Closes by: \
     the converter emitting the layout and the generator reading it, rather \
     than both asserting it."
  , "StagedUploadCoversTheTensor — lm_head is 1.08 GiB and the staging buffer \
     is 29.5 MiB, so the load is a loop of offset uploads. That the chunks tile \
     the tensor exactly, with no gap and no overlap, is arithmetic in the \
     generator and is not stated. It was wrong once, and the symptom was every \
     logit zero rather than a wrong answer."
  , "TurnIsTheStepRepeated — a chat turn is `dStepM` over the template, the \
     tokenised text, and then its own output, and the claim is that driving it \
     from inside the artifact computes what driving it from outside does. \
     Evidence: the same harmony prompt through both paths produces the same 40 \
     token ids, and the prompt-end logits match the bf16 reference to 7.6e-3 \
     with the argmax agreeing. Not stated."
  , "StopAndBoundTerminate — a turn ends on the caller's stop id or after \
     `D_MAXNEW` tokens, and the generation loop carries both. That it cannot \
     run past the buffers it writes into rests on `TEXT_MAX` bounding both, \
     which is arithmetic in the generator rather than a theorem."
  , "TemplateIsData — harmony's control tokens arrive as ids the caller \
     supplies, spliced around the tokenised text, because BPE over the literal \
     `<|start|>` yields its pieces rather than the special id. Which ids make a \
     turn is a fact about the chat format and is not checked here at all."
  , "DetokBoundsItself — an id outside the decode table contributes nothing \
     rather than reading past it. This is *not* a nicety: harmony's control \
     tokens are `added_tokens` above the BPE vocabulary, and a table that \
     stopped at the vocabulary made the length driving the copy garbage, which \
     is a buffer overrun and arrives as a core dump. The converter now covers \
     every id the model can emit; the guard is there for the converter that \
     does not."
  , "GumbelIsASample — with a non-zero 1/T the head launches `sample_logits`, \
     which adds independent standard Gumbel noise to each tempered logit and \
     takes the argmax. That this draws from `softmax(logit/T)` is the \
     Gumbel-max identity, and it is the reason sampling costs one pass instead \
     of a normalising constant and a prefix scan over 201088 logits. The \
     identity is not stated here, the noise is `-ln(-ln u)` for `u` a hash of \
     the index and the seed, and `lg2.approx` is the hardware's approximate \
     logarithm rather than an exact one. Evidence: a seed reproduces its \
     output exactly and different seeds differ. Greedy is a separate kernel, \
     not this one at T=0, because the reciprocal would not exist."
  , "WidenIsLossless — nothing numeric reaches this program from its caller: \
     the embedding row is read from the file at token * HH * 2, uploaded as the \
     bf16 it is, and widened by GptOssKernels.widenBf16. bf16 is f32 with the \
     low sixteen mantissa bits cleared, so the widening is exact and no \
     rounding decision arises. Evidence: bit-identical logits and layer trace \
     against the host-side gather it replaced. Not stated." ]

/-- **Facts about the emitted program that are measured, not proven.**

    Reported so that a change which quietly doubles a body shows up as a number
    moving rather than as a build that still passes. -/
def measured : List String :=
  [ s!"shipped bodies: {GptOssDecode.dShippedBodies.length}"
  , s!"buffers: {GptOssDecode.DNBUF}"
  , s!"ptx slots: {GptOssDecode.dPtx.length}"
  , s!"host memory: {GptOssDecode.DMEM_SIZE} bytes"
  , s!"trace rows returned per token: {2 * GptOssDecode.NL} \
(residual stream after each half of each layer)"
  , s!"text buffers: {GptOssDecode.TEXT_MAX} bytes in and out"
  , "a turn: tokenise, prefill through the decode path, generate, detokenise"
  , "measured on an RTX 3060, optimised runtime: cold start 31.4 s, prompt \
39.3 tok/s, generate 49.0 tok/s"
  , "measured: 0.2% of expert lookups missed over 128 generated tokens \
(3.5% over a chat turn), 2.1 MiB of expert traffic a token"
  , "the prompt runs through the decode path; there is no prefill kernel" ]

end GptOssDecodeScan

/-- **Claims that rest on the compiler**, via `native_decide`.

    Every root above is one. They are large finite checks over lists the
    generator built, which is what `native_decide` is for; none of them is a
    statement about what a kernel computes. -/
def gptOssDecodeNativeRoster : List Name := GptOssDecodeScan.roots

open TrustScan GptOssDecodeScan in
#eval runScanWith gptOssSurface "gpt-oss decode" roots gptOssDecodeNativeRoster

#eval do
  IO.println s!"[gpt-oss decode] roots scanned: {GptOssDecodeScan.roots.length}"
  IO.println s!"[gpt-oss decode] open obligations: \
{GptOssDecodeScan.decodeOpenObligations.length}"
  for s in GptOssDecodeScan.decodeOpenObligations do
    IO.println s!"[gpt-oss decode] NOT YET STATED: {s}"
  for m in GptOssDecodeScan.measured do
    IO.println s!"[gpt-oss decode] measured: {m}"
