import Scan.MlSurface
import Warp.WeaveCifar
import Vit.Weave
import Scan.Core
import AlgorithmLib.ML.Model.WeaveTOp
/-!
  # What the broadcasting algebra's claims rest on — computed, not documented

  The algebra sits above the kernels, so it ships no artifact and its scan has
  nothing to say about emission.  What it must say is the opposite thing: that
  the laws are applied.  A law about broadcasting that holds of no shipped
  operation is exactly the failure this development has hit twice, so the roots
  below deliberately mix the abstract statements with the places they are
  instantiated — `ew1_comp_is_bop` and `cifarW_lowers` are here for the same
  reason `lift_comp` is.

  The surface is `MlSurface`'s, widened by exactly two hypotheses and by
  nothing else.  Both are the algebra's own side conditions, and both are
  conditions the paper leaves implicit:

  * `ReadsBelow` — an operation tiled over a block of `n` addresses must not
    read past it.  Tiling an operation that did would read its neighbour's
    data, so this is what makes `lift` well defined at all.  The paper hides
    it inside the join/split isomorphisms of its batch lifting; naming it is
    the honest form.  It is discharged, not merely carried:
    `ew1BOp_readsBelow` proves it for a shipped elementwise pass, and
    `ew1_comp_is_bop` applies the composition law using that proof.
  * `InBounds` — a coordinate lies inside its extent.  The reindexing
    composition law needs the intermediate point to be addressable in the
    shape it passes through; off that condition the round trip through
    row-major addressing is not an identity and the law genuinely fails.

  The algebra introduces no axiom and no opaque constant.
-/

open Lean

namespace WeaveScan

/-- The claims this development makes.

    The first group is the algebra: the four laws the paper states, plus the
    closure property that makes a composite of broadcasted operations one
    again.  The second is the bridge to the kernels that ship.  The third is
    the two models, which is what stops the first group being about nothing. -/
def roots : List Name :=
  [ -- the indexing category
    `AlgorithmLib.ML.Broadcast.Reindex.apply_comp
  , `AlgorithmLib.ML.Broadcast.Reindex.apply_comp_assoc
  , `AlgorithmLib.ML.Broadcast.Reindex.apply_id
  , `AlgorithmLib.ML.Broadcast.Reindex.id_comp
  , `AlgorithmLib.ML.Broadcast.Reindex.comp_id
  , `AlgorithmLib.ML.Broadcast.unflatten_flatten
  , `AlgorithmLib.ML.Broadcast.flatten_lt
    -- the four laws
  , `AlgorithmLib.ML.Broadcast.lift_id
  , `AlgorithmLib.ML.Broadcast.lift_comp
  , `AlgorithmLib.ML.Broadcast.pull_comp
  , `AlgorithmLib.ML.Broadcast.slide
  , `AlgorithmLib.ML.Broadcast.interchange
  , `AlgorithmLib.ML.Broadcast.pullFront_comp
    -- broadcasted operations
  , `AlgorithmLib.ML.Broadcast.BOp.comp_den
  , `AlgorithmLib.ML.Broadcast.mergePt_splitPt
  , `AlgorithmLib.ML.Broadcast.splitPt_inBounds
    -- the bridge to what ships
  , `AlgorithmLib.ML.Broadcast.bcast_is_reindex
  , `AlgorithmLib.ML.Broadcast.bcastReindex_wf
  , `AlgorithmLib.ML.Broadcast.rowdot_operand_is_reindex
  , `AlgorithmLib.ML.Broadcast.ew1_is_bop
  , `AlgorithmLib.ML.Broadcast.ew1BOp_readsBelow
  , `AlgorithmLib.ML.Broadcast.ew1_comp_is_bop
  , `AlgorithmLib.ML.Broadcast.ziprow_operand_is_reindex
  , `AlgorithmLib.ML.Broadcast.ziprow_reads_reindexed
    -- lowering
  , `AlgorithmLib.ML.Broadcast.WOp.lower_den
  , `AlgorithmLib.ML.Broadcast.WOp.lower_ofTOp
  , `AlgorithmLib.ML.Broadcast.WOp.ofTOp_den
  , `AlgorithmLib.ML.Broadcast.WOp.lowerAll_den
  , `AlgorithmLib.ML.Broadcast.WOp.lowerAll_ofTape
  , `AlgorithmLib.ML.Broadcast.WOp.ofTape_den
    -- programs compose
  , `AlgorithmLib.ML.Broadcast.WOp.lowerAll_append
  , `AlgorithmLib.ML.Broadcast.WOp.denAll_append
  , `AlgorithmLib.ML.Broadcast.WOp.denAll_nil
    -- the models, which is what stops the above being about nothing
  , `WeaveCifar.cifarBuild_is_cifarW
  , `WeaveCifar.cifarBuild_lowers
  , `WeaveCifar.mismatch_rejected
  , `WeaveCifar.cifarW_lowers
  , `WeaveCifar.cifarW_length
  , `WeaveCifar.cifarW_computes
  , `WeaveCifar.cifar_stages_run_the_algebra
  , `WeaveCifar.cifar_shipped_stages_run_the_algebra
  , `WeaveVit.vit_is_algebra
  , `WeaveVit.vitW_ofTape
  , `WeaveVit.vitW_length
  , `WeaveVit.vitW_lowers
  , `WeaveVit.vit_tape_computes ]

/-- What this development does **not** state.

    Kept here rather than left implicit, because a layer that describes every
    operation is the easiest place to mistake coverage for depth. -/
def notYetStated : List String :=
  [ "that a stochastic operation is a broadcasted one — the base category is \
     deterministic here, so dropout and sampling have no interpretation; the \
     terms are syntax and a second interpretation would carry them",
    "that a general permutation or an in-memory transpose lowers — the \
     fragment refuses them, because IdxE is affine with no division and no \
     kernel below this layer transposes memory",
    "that a whole tape is one BOp — programs compose by concatenation \
     (lowerAll_append, denAll_append) and two operations compose by \
     BOp.comp_den, but a tape is not exhibited as a single broadcasted \
     operation with one target and one reindexing",
    "that a weave's regrouping of axes is an address permutation — splitPt \
     and mergePt are proven inverse on points, not yet on flat addresses",
    "that Qwen2 and gpt-oss tapes are algebra programs — only CIFAR and ViT \
     are checked" ]

end WeaveScan

/-- The algebra's surface: `mlSurface`, plus the two side conditions the laws
    carry.  Written out rather than widening `MlSurface` itself, so no other
    pipeline's claims can come to rest on them by accident. -/
def weaveSurface : TrustScan.Surface :=
  { TrustScan.mlSurface with
    allowedHyp := TrustScan.allowedHyp ++
      [ `AlgorithmLib.ML.Broadcast.ReadsBelow
      , `AlgorithmLib.ML.Broadcast.InBounds ] }

/-- Claims that rest on the compiler, via `native_decide`.

    `vit_is_algebra` decides the abstraction on the shipped tape; the other
    three are stated in terms of it, so they inherit it.  Listed rather than
    hidden: what rests on the compiler here is the *check* that ViT's 2294
    operations are in the fragment, not any law about them. -/
def weaveNativeRoster : List Name :=
  [ `WeaveVit.vit_is_algebra
  , `WeaveVit.vitW_ofTape
  , `WeaveVit.vitW_length
  , `WeaveVit.vitW_lowers
  , `WeaveVit.vit_tape_computes ]

open TrustScan WeaveScan in
#eval runScanWith weaveSurface "weave" roots weaveNativeRoster

#eval do
  IO.println s!"[weave] roots scanned: {WeaveScan.roots.length}"
  IO.println s!"[weave] not yet stated: {WeaveScan.notYetStated.length}"
