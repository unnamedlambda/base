import ScanCore
import BackwardWideAlgorithm

/-!
  # What the *training* pipeline's claims rest on — computed, not documented

  The same scan `TrustScan.lean` runs over inference, run over the backward
  pipeline.  Two files rather than one because each generator defines its own
  `main`, so `Qwen2Algorithm` and `BackwardWideAlgorithm` cannot be imported
  into a single module; the machinery they share lives in `ScanCore`.

  This existing at all is the point.  The backward pipeline had no scanner, so
  "the training claims rest on the same surface as the inference claims" was a
  sentence rather than a computation — exactly the shape of thing that went
  stale before.
-/

open Lean

namespace BackwardScan

/-- **The public claims of the backward pipeline.**

    Grouped as they are argued about: the kernels compute the gradient the
    calculus says they should; the emitted PTX computes what the kernels do;
    the pipeline's stages do not collide; the host program performs the
    launches; and the whole thing realises a plan whose declared gap is two.

    The composition theorem `bwd_chain` is the headline — the *sequence* of
    launches computes `sderiv` of the layer, with the chain rule coming from
    `sderiv`'s `letE` rule rather than from an assembly by hand. -/
def roots : List Name :=
  [ -- the kernels against the calculus
    `BackwardWide.bwd_computes_spec
  , `BackwardWide.bwd_ptx_computes_spec
  , `BackwardWide.dW_stores
  , `BackwardWide.dW_ptx_exact
  , `BackwardWide.siluBwd_stores
  , `BackwardWide.siluBwd_ptx_exact
  , `BackwardWide.rmsBwd_stores
  , `BackwardWide.rmsBwd_ptx_exact
  , `BackwardWide.rmsBwd_S_spec
    -- the composition: a launch *sequence* computes the layer's derivative
  , `BackwardWide.bwd_chain
    -- non-collision, so the sequence means anything at all
  , `BackwardWide.bwdPipeline_exclusive
  , `BackwardWide.bwdPipelineFull_exclusive
  , `BackwardWide.bwdPlan_exclusive
  , `BackwardWide.bwdPipeline_runs
  , `BackwardWide.bwdPipelineFull_runs
    -- the host program performs them
  , `BackwardWide.bwdDriver_realises
  , `BackwardWide.bwdDriverBlas_realises
  , `BackwardWide.bwd_host_computes
  , `BackwardWide.bwd_host_computes_plan
    -- the gap, as a number
  , `BackwardWide.bwdPlan_declaredCount
    -- the emitted PTX is well-formed and fits where the loader puts it
  , `BackwardWide.bwd_targets_ok
  , `BackwardWide.bwd_printable
  , `BackwardWide.bwd_stage_eligible
  , `BackwardWide.bwdPtx_fits
  , `BackwardWide.bwd_bufs_bound
  , `BackwardWide.bwd_geometry
  , `BackwardWide.bwdMap_ok
  , `BackwardWide.bind_count
  , `BackwardWide.hostIn_packed
  ]

end BackwardScan

open TrustScan BackwardScan in
#eval runScan "backward" roots
