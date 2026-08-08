import MlSurface
import MlpCifarAlgorithm

/-!
  # What the CIFAR classifier's claims rest on — computed, not documented

  The third scanner, for the same reason there are two already: each generator
  defines its own `main`, so `Qwen2Algorithm`, `BackwardWideAlgorithm` and
  `MlpCifarAlgorithm` cannot share a module.  The machinery is `ScanCore`, the
  surface `MlSurface`.

  What makes this one worth having separately is that it covers a *whole
  model* — forward, loss gradient, backward and optimiser — rather than one
  layer.  If the surface it reports is the same as the backward pipeline's,
  that is the claim "a model costs geometry, not proof" being checked rather
  than asserted.
-/

open Lean

namespace MlpScan

/-- **The public claims of the CIFAR classifier.**

    Grouped as they are argued about: each kernel computes its spec; the
    emitted PTX computes what the kernel does; the backward stages do not
    collide; and the layout, geometry and slot arithmetic hold.

    The softmax and the cross-entropy gradient are deliberately absent, because
    they are not kernels — they run on the host, which the module docstring
    names.  A root here for something the GPU does not do would be the exact
    shape of a vacuous claim.

    `fwd1_agrees_unbatched` is the one to read first: it says sample `s` of the
    batched kernel and an unbatched `dotStrided` for that sample leave the same
    `Float32`, so choosing a batch size is a scheduling decision and is checked
    to be one. -/
def roots : List Name :=
  [ -- forward
    `MlpCifar.fwd1_computes_spec
  , `MlpCifar.fwd2_computes_spec
  , `MlpCifar.h_stores
  , `MlpCifar.fwd1_ptx_exact
  , `MlpCifar.fwd2_ptx_exact
    -- backward
  , `MlpCifar.dh_computes_spec
  , `MlpCifar.adj_stores
    -- batching changed the schedule, not the arithmetic
  , `MlpCifar.fwd1_agrees_unbatched
  , `AlgorithmLib.ML.batch_addr_inj
  , `MlpCifar.dw1_stores
  , `MlpCifar.dw2_stores
  , `MlpCifar.dh_ptx_exact
  , `MlpCifar.dw1_ptx_exact
  , `MlpCifar.dw2_ptx_exact
  , `MlpCifar.map_ptx_exact
    -- the optimiser
  , `MlpCifar.sgd1_stores
  , `MlpCifar.sgd2_stores
    -- non-collision, so a launch sequence means anything at all
  , `MlpCifar.fwdPipeline_exclusive
  , `MlpCifar.bwdPipeline_exclusive
  , `MlpCifar.fwdPipeline_runs
  , `MlpCifar.bwdPipeline_runs
  , `MlpCifar.stages_are_launches
    -- the emitted CLIF performs those launches, and they are that pipeline
  , `MlpCifar.fwd_ops_are
  , `MlpCifar.bwd_ops_are
  , `MlpCifar.fwd_host_computes
  , `MlpCifar.bwd_host_computes
    -- a training run, not just a training step
  , `MlpCifar.mlp_training_run
  , `MlpCifar.mlp_host_swap
    -- the cuBLAS configuration: same denotation, five declared steps
  , `MlpCifar.blas_declared
  , `MlpCifar.blas_denotes_the_same
  , `MlpCifar.blas_law_bill
  , `MlpCifar.proven_law_bill
  , `MlpCifar.blas_plans_run
  , `MlpCifar.fwd_blas_ops_are
  , `MlpCifar.bwd_blas_ops_are
  , `MlpCifar.blas_host_computes
    -- the backend is a lowering choice, and it cannot change the answer
  , `MlpCifar.builds_agree
  , `MlpCifar.builds_run
  , `MlpCifar.builds_declared
    -- the layer's addressing is derived from its shapes, not written
  , `MlpCifar.dense_derives_the_model
  , `MlpCifar.dense_derives_the_grids
  , `MlpCifar.layers_compute_the_dot
    -- the model as a graph, lowering to the shipped stages
  , `MlpCifar.program_allocates
  , `MlpCifar.cifarTen_lowers
  , `MlpCifar.ten_schedule_is_the_model
  , `MlpCifar.cifar_default_schedule
  , `MlpCifar.ten_schedule_differs
  , `MlpCifar.activation_derived
  , `MlpCifar.deep93_graph
  , `MlpCifar.deep93_lowers
  , `MlpCifar.rms_ops
  , `MlpCifar.rms_lowers
  , `MlpCifar.rms_addressing
  , `MlpCifar.k3_block_ops
  , `MlpCifar.k3_block_lowers
  , `MlpCifar.k3_eight_ops
  , `MlpCifar.k3_24_lowers
  , `MlpCifar.k3_24_ops
  , `MlpCifar.qwen_kernels_are_the_stages
  , `MlpCifar.qwen_slot_count
  , `MlpCifar.qwen_within_module_limit
  , `MlpCifar.qwen_bufs_within_regs
  , `MlpCifar.qwen_kernel_width_is_bounded
  , `MlpCifar.qwen_bufs_bound
  , `MlpCifar.qwen_targets
  , `MlpCifar.qwen_printable
  , `MlpCifar.qwen_regs_ok
  , `MlpCifar.qwen_ro_policies_differ
  , `MlpCifar.regs_ok_rejects_the_scratch
  , `MlpCifar.moe_slot_count
  , `MlpCifar.moe_kernels_are_the_stages
  , `MlpCifar.moe_slots_are_bound
  , `MlpCifar.moe_store_is_unnamed
  , `MlpCifar.moe_alloc_covers
  , `MlpCifar.moe_bufs_within_regs
  , `MlpCifar.moe_kernels_ok
  , `MlpCifar.moe_ptx_fits
  , `MlpCifar.moe_grids_are_the_stages
  , `MlpCifar.mHostIn_packed
  , `MlpCifar.moeMap_ok
  , `MlpCifar.qwen_ptx_fits
  , `AlgorithmLib.ML.laneW_unchanged
  , `MlpCifar.block_backward_derived
  , `MlpCifar.block_backward_lowers
  , `MlpCifar.k3_fused_ops
  , `MlpCifar.k3_fused_lowers
  , `MlpCifar.rms_fusion_agrees
  , `MlpCifar.qwen_fwd_fusable
  , `MlpCifar.qwen_fused_shorter
  , `MlpCifar.qwen_fused_steps_in_order
  , `MlpCifar.qwen_training_not_fusable
  , `MlpCifar.qwen_training_temp_live
  , `MlpCifar.qwen_fwd_fusion_agrees
  , `MlpCifar.rope_fusion_agrees
  , `MlpCifar.lane_deriv_is_scalar
  , `MlpCifar.norm_backward_derived
  , `MlpCifar.norm_backward_lowers
  , `MlpCifar.rope_backward_derived
  , `MlpCifar.rope_backward_lowers
  , `MlpCifar.qwen_stages_eq
  , `MlpCifar.qwenPipeline_exclusive
  , `MlpCifar.qwen_run_den
  , `MlpCifar.qwen_extents_are_typed
  , `MlpCifar.qwen_grad_slots
  , `MlpCifar.qwen_grids_are_the_stages
  , `MlpCifar.qwen_steps_in_order
  , `MlpCifar.qwen_bufs_fresh
  , `MlpCifar.qwen_working_set
  , `MlpCifar.qwen_alloc_covers
  , `MlpCifar.qHostIn_packed
  , `MlpCifar.qwenMap_ok
  , `MlpCifar.k3_moe_ops
  , `MlpCifar.k3_moe_lowers
  , `MlpCifar.cifar_backward_derived
  , `MlpCifar.cifar_backward_nonempty
  , `MlpCifar.accum_binds_then_sums
  , `MlpCifar.badNet_rejected
    -- the emitted PTX is well-formed, fits, and every kernel is launched
  , `MlpCifar.mlp_targets_fwd1
  , `MlpCifar.mlp_targets_h
  , `MlpCifar.mlp_targets_fwd2
  , `MlpCifar.mlp_targets_dw2
  , `MlpCifar.mlp_targets_dh
  , `MlpCifar.mlp_targets_adj
  , `MlpCifar.mlp_targets_dw1
  , `MlpCifar.mlp_targets_sgd1
  , `MlpCifar.mlp_targets_sgd2
  , `MlpCifar.mlp_targets_sm
  , `MlpCifar.mlp_printable_fwd1
  , `MlpCifar.mlp_printable_h
  , `MlpCifar.mlp_printable_fwd2
  , `MlpCifar.mlp_printable_dw2
  , `MlpCifar.mlp_printable_dh
  , `MlpCifar.mlp_printable_adj
  , `MlpCifar.mlp_printable_dw1
  , `MlpCifar.mlp_printable_sgd1
  , `MlpCifar.mlp_printable_sgd2
  , `MlpCifar.mlp_printable_sm
  , `MlpCifar.mlp_stage_eligible
  , `MlpCifar.mlpPtx_fits
  , `MlpCifar.mlpMap_ok
  , `MlpCifar.mlp_bufs_bound
  , `MlpCifar.mlp_bufs_within_regs
  , `MlpCifar.mlp_kernels_are_the_stages
  , `MlpCifar.scheduleBills_are_the_plans
  , `MlpCifar.cublas_law_refines
  , `MlpCifar.qwen_fuse_site
  , `MlpCifar.cifar_labels
  , `MlpCifar.k3_labels
  , `MlpCifar.qwen_capture_records_the_run
  , `MlpCifar.qwen_replay_denote
  , `MlpCifar.qwen_replay_realises
  , `MlpCifar.qwen_replay_ops_are
  , `MlpCifar.qwen_run_realises
  , `MlpCifar.qwen_run_ops_are
  , `MlpCifar.asserted_schedule_is_the_model
  , `MlpCifar.firstToBlas_sound
  , `MlpCifar.mlp_geometry
  , `MlpCifar.slot_count
  , `MlpCifar.steps_cover
  , `MlpCifar.buf_count
  , `MlpCifar.hostIn_packed
  ]

end MlpScan

open TrustScan MlpScan in
#eval runScan "mlp-cifar" roots

/-! The laws each shipped schedule rests on, printed beside the scan: a
    schedule that reached for a primitive with a different contract changes
    this line. -/
#eval do
  for (name, laws, lawless) in MlpCifar.scheduleBills do
    let bill :=
      if laws.isEmpty && lawless == 0 then "nothing"
      else if laws.isEmpty then s!"{lawless} step(s) assumed with no stated law"
      else if lawless == 0 then toString laws
      else s!"{laws}, plus {lawless} step(s) assumed with no stated law"
    IO.println s!"[mlp-cifar] schedule {name} bills {bill}"
  IO.println s!"[mlp-cifar] fusable sites the qwen forward tape offers, by name: \
{AlgorithmLib.ML.fuseNames ((MlpCifar.k3Stack 1).labels MlpCifar.QBASE) MlpCifar.qwenFwd}"
  IO.println s!"[mlp-cifar] names the cifar model bound: \
{(MlpCifar.cifarTen.labels MlpCifar.NIN).map Prod.fst}"
  IO.println s!"[mlp-cifar] names one qwen block bound: \
{((MlpCifar.k3Stack 1).labels MlpCifar.QBASE).map Prod.fst}"
  for (nm, sc) in [("cuBLAS GEMMs", MlpCifar.cifarBlasSched),
                   ("asserted", MlpCifar.assertedSched)] do
    if !sc.assertedNames.isEmpty then
      IO.println s!"[mlp-cifar] schedule {nm} asserts {sc.assertedNames} \
(discharged where a proof is given)"
  for l in [AlgorithmLib.ML.Law.cublasIsMatvec] do
    match l.weakens with
    | some w =>
        IO.println s!"[mlp-cifar] law {l.title} is a strictification of {w.title}"
    | none => pure ()
