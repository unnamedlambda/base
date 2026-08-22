import GptOssSurface
import GptOssDecode

/-!
  # What the whole-model artifact claims, and what it only arranges

  `GptOssScan` scans the pieces: the attention kernels, the mixture, one layer.
  This scans the thing those pieces were for — `gptoss_decode`, a single CLIF
  function that turns a token into the next one, twenty-four layers and an
  expert cache included, with no second entry point and no host in the loop.

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

    All four are guards on emitted data. They are the strongest statements that
    can be made about this artifact today, and they are all structural. -/
def roots : List Name :=
  [ ``GptOssDecode.gptoss_decode_alloc_covers
  , ``GptOssDecode.gptoss_decode_ptx_fits
  , ``GptOssDecode.gptossDecodeMap_ok
  , ``GptOssDecode.gptossDecodeShipped_wf ]

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
  , "KernelVariantByParity — a sliding layer and a full layer differ only in \
     which PTX slot the launch names, chosen arithmetically from l % 2 because \
     ptxOff is a register. That the parity picks the variant whose capacity \
     matches the cache the layer was allocated is not stated."
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
(residual stream after each half of each layer)" ]

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
