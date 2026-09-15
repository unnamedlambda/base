import GptOssSurface
import AlgorithmLib.ML.Compose
import GptOssAttention
import GptOssAlgorithm

/-!
  # What the gpt-oss-20b artifact's claims rest on — and, mostly, do not yet

  This scanner is written **before** the artifact it scans, which is the only
  unusual thing about it.

  The decision it records is that the model ships first and the proofs follow.
  That is a defensible order — a kernel nobody can run is not evidence of
  anything either — but it has failed here twice in a recognisable way: every
  layer proven, none of them applied, discovered only when someone went looking
  (`lz4_composition_gap`). What made that possible both times was that the
  distance between "tested" and "proven" lived in people's heads.

  So `openObligations` below is the ledger, and it exists from the first commit
  of this application rather than from the first proof. Each entry names a claim
  the running program relies on, says what evidence there is for it today, and
  says what would close it. Entries leave by being proven, not by being
  forgotten; `roots` is empty until something is.

  **The distinction this file is for.** gpt-oss-20b's expert weights are MXFP4:
  two four-bit codes to a byte, one shared exponent per thirty-two of them. The
  machine this development proves kernels in addresses buffers by element and
  holds `Float32`, with no byte, no shift and no mask (`ML/QuantMX.lean` says so
  in its own header, and is otherwise a complete specification of the format:
  `fp4Val`, `mxDeq`, and `mxDot_spec`, which proves a dequantised dot is the
  same proven fold at different numbers). A kernel that unpacks nibbles is
  therefore outside the proven machine as it stands — not hard to write, and
  written here in the string builder that has no semantics attached, but outside.

  Extending the machine is what closes most of this ledger, and it is a
  five-file change to the emitter with a decidable predicate per new
  constructor. It is deliberately not on the path to the first token.
-/

open Lean

namespace GptOssScan

/-- **The claims this artifact makes.**

    It was empty when this file was written, which was before the artifact
    existed. What has arrived since is the whole attention half: every kernel
    it launches is an `EWStmt`, so each carries the same execution theorem the
    Qwen2 decode kernels carry — the emitted PTX runs the statement, from raw
    launch, including the ones whose addresses depend on memory (the rotation's
    table index, the cache's slot, the softmax's trip counts).

    Two of these are stronger than execution. `kvStore_writes` says the right
    element reaches the right address for every `(loop, lane)` the kernel
    visits, not merely that some lane wrote it; `sink_not_stored` says the only
    buffer the softmax writes is the probabilities, which is the asymmetry that
    makes a sink a sink.

    The expert half contributes its seam guards rather than kernel theorems,
    for the reason the header gives: those kernels are outside the machine. -/
def roots : List Name :=
  [ ``GptOssAttention.rms_ptx_exact
  , ``GptOssAttention.add_ptx_exact
  , ``GptOssAttention.ropeQ_ptx_exact
  , ``GptOssAttention.ropeK_ptx_exact
  , ``GptOssAttention.kvStore_ptx_exact
  , ``GptOssAttention.kvStore_writes
  , ``GptOssAttention.sinkSoftmax_ptx_exact
  , ``GptOssAttention.sink_not_stored
  , ``GptOssAlgorithm.gptoss_slots_are_bound
  , ``GptOssAlgorithm.gptoss_store_is_unnamed
  , ``GptOssAlgorithm.gptoss_binds_allocated
  , ``GptOssAlgorithm.gptoss_alloc_covers
  , ``GptOssAlgorithm.gptoss_ptx_fits
  , ``GptOssAlgorithm.gptossHostIn_packed
  , ``GptOssAlgorithm.gptossMap_ok
  , ``GptOssAlgorithm.Attn.gptoss_attn_alloc_covers
  , ``GptOssAlgorithm.Attn.gptoss_attn_ptx_fits
  , ``GptOssAlgorithm.Attn.gptossAttnHostIn_packed
  , ``GptOssAlgorithm.Attn.gptossAttnMap_ok
  , ``GptOssAlgorithm.Layer.gptoss_layer_slots_are_bound
  , ``GptOssAlgorithm.Layer.gptoss_layer_binds_allocated
  , ``GptOssAlgorithm.Layer.gptoss_layer_alloc_covers
  , ``GptOssAlgorithm.Layer.gptoss_layer_ptx_fits
  , ``GptOssAlgorithm.Layer.gptossLayerHostIn_packed
  , ``GptOssAlgorithm.Layer.gptossLayerMap_ok ]

/-- **Every claim the running program rests on that is not yet a theorem.**

    Thirteen entries, in four groups: the expert kernels, the dense
    contraction, attention, and the seams outward to the host.

    The two counts that matter are in the text, not in a summary line: what a
    decode step *computes* is assumed at every expert launch, and what it
    computes it *from* is assumed at the converter. Neither is a small
    assumption, and both are testable — which is the argument for shipping in
    this order and the reason each entry names its test. -/
def openObligations : List String :=
  [ -- ── the expert kernels: MXFP4 unpacking, outside the proven machine ──
    "Mxfp4GateUpGemvValue: the fused gate/up GEMV with the clamped SwiGLU in \
     its epilogue lands the expert's hidden row, dequantising as `QuantMX.mxDeq` \
     defines it. Evidence: the decode is checked bit-exact against `fp4Val` over \
     all 256 byte values crossed with the scale extremes, and the contraction \
     against an f32 fold. Closes by: extending the warp machine with a byte load \
     and the shifts (Warp, WarpEmit and its five decidable predicates, PtxM, \
     PtxFlat, PtxPrint's element stride), after which `mxDot_spec` hands the \
     existing `warpDotV4`/`dotStrided` theorems a dequantised buffer for free.",
    "Mxfp4DownGemvValue: the down-projection GEMV plus bias, same standing and \
     the same closure.",
    "Mxfp4DequantF32Value: the prefill path's dequantise-to-f32 kernel writes \
     exactly `mxDeq` of its input. Separate from the two above because prefill \
     is compute-bound on this hardware and may spend the bandwidth a fused \
     kernel saves; decode may not, and does not.",
    -- ── the dense contraction: a vendor call at a narrower dtype ──
    "GemmExBf16Value: what `cl_cublas_gemm_ex_bf16` lands. Weaker than the \
     other vendor calls here, and stated as such (`VendorKernel.cublasGemmExBf16` \
     is `lawless`): the operands are bf16, so the products are not the products \
     of the `Float32` values this model holds, and even the weak `some \
     association` reading is false before fold order is reached. Closes by: a \
     law over bf16-rounded operands, plus the `DeclaredStep` lemma that applies \
     it — not by promoting the existing f32 law, which does not hold here.",
    "ActivationNarrowingValue: the f32→bf16 conversion in front of every such \
     contraction rounds to nearest even. It exists because cuBLAS refuses a \
     mixed operand pair — measured, not assumed — so a bf16 weight forces a \
     bf16 activation. Evidence: checked against the converter's own rounding, \
     which is the same function.",
    -- ── attention: shapes the proven kernels do not yet cover ──
    "SinkSoftmaxValue: the learned per-head sink seeds the maximum and adds one \
     term to the denominator. The kernel is the proven dynamic-length softmax \
     with two edits in the existing expression vocabulary, so its shape, \
     register and printability guards hold as they stand; what is missing is \
     the value theorem for the edited passes.",
    "SwaRingIsTheWindow: for a sliding-window layer, the 128-entry ring — \
     position p at slot p mod 128, contracted over the whole cache — holds \
     exactly the window the model masks to, and softmax and the value mix are \
     symmetric in the keys so the order the ring leaves them in is not \
     observable. This replaces the offset-and-length obligation the first draft \
     of this ledger carried: the ring needs no offset, no mask and no runtime \
     shape, which is why it is what ships. Evidence: checked at positions 0, 5, \
     130 and 1000, the first of which is one key long and the third of which is \
     the first to overwrite an entry that has already left its own window.",
    "AttnStrideLayout: that the batched contraction's strides address the \
     grouped-query layout the model defines — query head h served by key head \
     h/8, scores for head h at h·seqLen. Numeric agreement only, at the four \
     positions above. The kernels either side of it are proven; this is the \
     arithmetic in the launch arguments between them.",
    "AttnKernelValues: what the attention kernels compute, as opposed to that \
     the emitted PTX runs their statements. RMSNorm, the elementwise add and \
     the rotation have value and store-address theorems in Qwen2Proven at that \
     model's geometry; they are not re-instantiated here, and re-instantiating \
     them is a proof-script edit rather than a proof. The cache store is the \
     exception and is already a root.",
    "PrefillGatherMaskValue: the routed-token gather and the additive causal \
     and window masks the chunked prefill uses.",
    -- ── the seams outward: host memory, host policy, the checkpoint ──
    "HostRoutingPolicy: that the host's top-4 over the router row, the softmax \
     over those four, and the cache's victim choice produce the bindings and \
     gates the tape assumes. This is a policy, not arithmetic: any choice of \
     resident experts computes *a* mixture, and what is at stake is whether it \
     is the mixture the model specifies. Evidence: a wrong binding is caught by \
     the distinctness check the existing MoE demo already runs. Closes by: \
     moving the policy into CLIF, where `ClifCheck` can decide it.",
    "ExpertFragmentPlan: no plan-realisation claim is made across a fragment \
     containing an MXFP4 launch. Launch records resolve to proven stages, and \
     these are not stages yet, so the composition theorem is unavailable rather \
     than false — the same standing the shipped sparse-dispatch demo has. \
     Closes with the first two entries.",
    "EmbedHostGather: in *these* artifacts the embedding row is read on the \
     host, widened, and uploaded, so what lands is `uploadedValue` on the \
     declared trust surface rather than a gather this development proves. It \
     is a property of the per-piece programs and not of the model: the \
     whole-model artifact reads the row itself and widens it on the device \
     (`GptOssDecodeScan.WidenIsLossless`), bit-identically to this.",
    "ConverterFidelity: the weight bank is the published checkpoint under the \
     transforms the converter declares — de-interleaving, transposition, the \
     packed byte order, and the assertion that no block scale is the reserved \
     NaN exponent. Evidence: checksums against the source tensors and a small \
     synthetic model checked end to end. Not proven, and this is the one that \
     bounds every other claim: a kernel that is exactly right about the wrong \
     bytes computes the wrong model.",
    "YarnTables: the host-precomputed rotation tables are the published scaled \
     formula. Numeric agreement only." ]

/-- **What has been measured on this machine**, kept beside the ledger because
    the case for shipping in this order rests on it and would otherwise be an
    assertion.

    None of these are claims about correctness. They are the reason the design
    is a host-resident expert pool with a device cache rather than something
    else: the pool does not fit in device memory, and the link that serves its
    misses is half as fast as the class of machine this design was published
    for. -/
def measured : List String :=
  [ "pinned host→device: 13.4 GB/s (PCIe 3.0 x16 — the socket's limit, not the card's)",
    "device memory: 333 GB/s, 12 GiB total",
    "host DRAM read, six cores: 35.2 GB/s",
    "host memory: 16 GiB, of which ~13 free — the binding constraint on which \
     model can be served at all" ]

end GptOssScan

/-- **Claims that rest on the compiler**, via `native_decide`.

    Every seam guard here is one, and deliberately: each is a decidable fact
    about a list the generator built — that the buffers are allocated, that a
    PTX text fits its slot, that the region map covers what the program names,
    that no launch names the store. These are large finite checks over emitted
    data, which is what `native_decide` is for and what `decide` cannot do at
    this size. The kernel theorems above are not on this list. -/
def gptOssNativeRoster : List Name :=
  [ ``GptOssAttention.sink_not_stored
  , ``GptOssAlgorithm.gptoss_slots_are_bound
  , ``GptOssAlgorithm.gptoss_binds_allocated
  , ``GptOssAlgorithm.gptoss_alloc_covers
  , ``GptOssAlgorithm.gptoss_ptx_fits
  , ``GptOssAlgorithm.gptossHostIn_packed
  , ``GptOssAlgorithm.gptossMap_ok
  , ``GptOssAlgorithm.Attn.gptoss_attn_alloc_covers
  , ``GptOssAlgorithm.Attn.gptoss_attn_ptx_fits
  , ``GptOssAlgorithm.Attn.gptossAttnHostIn_packed
  , ``GptOssAlgorithm.Attn.gptossAttnMap_ok
  , ``GptOssAlgorithm.Layer.gptoss_layer_slots_are_bound
  , ``GptOssAlgorithm.Layer.gptoss_layer_binds_allocated
  , ``GptOssAlgorithm.Layer.gptoss_layer_alloc_covers
  , ``GptOssAlgorithm.Layer.gptoss_layer_ptx_fits
  , ``GptOssAlgorithm.Layer.gptossLayerHostIn_packed
  , ``GptOssAlgorithm.Layer.gptossLayerMap_ok ]

open TrustScan GptOssScan in
#eval runScanWith gptOssSurface "gpt-oss" roots gptOssNativeRoster

#eval do
  IO.println s!"[gpt-oss] roots scanned: {GptOssScan.roots.length}"
  IO.println s!"[gpt-oss] open obligations: {GptOssScan.openObligations.length}"
  for s in GptOssScan.openObligations do
    IO.println s!"[gpt-oss] NOT YET STATED: {s}"
  for m in GptOssScan.measured do
    IO.println s!"[gpt-oss] measured: {m}"
  let k := AlgorithmLib.ML.VendorKernel.cublasGemmExBf16
  IO.println s!"[gpt-oss] dense contractions go through {k.symbol}, which states: \
{(AlgorithmLib.ML.VendorKernel.assumes k).map AlgorithmLib.ML.Law.title}"
  IO.println s!"[gpt-oss] and withholds: {AlgorithmLib.ML.VendorKernel.withholds k}"
