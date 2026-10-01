import Scan.MlSurface
import Vit.Guards
import Vit.Dag
import Vit.DagStep
import Vit.Regs
import Vit.Slot
/-!
  # What the ViT artifact's claims rest on — computed, not documented

  The fourth scanner, for the same reason there are three already: each
  generator defines its own `main`, so they cannot share a module.  The
  machinery is `ScanCore`, the surface `MlSurface`.

  **This one is mostly empty, and that is the report.**  `MlpScan` names about a
  hundred and fifty roots; this names one.  A scanner over a thin root list does
  not say the artifact is trustworthy — it says exactly how much of it has been
  argued, which is what was missing while the generator was being tuned.

  What the artifact does have is numerical evidence: the forward within 8.0e-07
  of timm, all 148 parameter gradients against autograd, and twenty SGD steps
  tracking PyTorch's loss to 6.2e-07.  That is real and it is not a proof, and
  keeping the two in separate columns is the whole point of a scan.
-/

open Lean

namespace VitScan

/-- **The public claims of the twelve-block ViT.**

    In four groups: the schedule, the tape, the kernels, and the seams outward.

    **The schedule.**  That the dependence graph the twenty-stream schedule
    synchronises along orders *every* conflicting pair of launches
    (`vit_deps_ordered`) — everything the graph schedule buys, which is most of
    the speed this artifact has, is equivalent to the tape's own order only if
    that holds.  `vDepsCheckWith` carries a flag that drops one predecessor edge
    per unit; run that way the same check reports unordered pairs, which is what
    distinguishes a guard from a decoration.

    **The tape.**  That what is emitted computes what the model's own lowering
    computes (`vit_fusion_sound`, `vit_lowering_sound`, `vit_tape_is_the_term`)
    — through two rounds of fusion, orientation, producer sinking and
    re-chunking, onto a tape nothing authors: the forward is the term's
    compilation, the backward is `Ten.backwardFrom` of it, the update one
    `upd2` per parameter.  Plus the two facts that stop it being vacuous —
    that sites were really taken, and that the buffer the answer is read from is
    not one of the temporaries fusion removed.

    **The kernels.**  That every emitted kernel names no reserved register
    (`vit_regs_ok`, with `vit_regs_ok_rejects_the_scratch` to show the guard can
    fail), that its buffers are inside its own parameter list, that its branch
    targets exist and that it is printable at all — and that the units these are
    stated over are exactly the units that print (`vit_proven_units_are_printed`).
    The register hazard is not hypothetical here: a collision in this printer
    once made two of ninety kernels fail to load.  Beyond well-formedness, that
    the emitted program *executes to* the statement it was built from
    (`vit_ptx_exact`, from the census `vit_stmts_flat_idxfree` — which is also
    the statement that nothing here addresses memory with a value it read at run
    time), and that the slot each launch names holds that group's own kernel
    (`vit_slot_holds_the_kernel`), which is what the dedup's hash key would
    otherwise have to be trusted for.

    **The seams outward.**  That the emitted CLIF makes exactly the launches the
    tape says, recovered from the shipped instruction stream and compared
    against a list *derived* from `vUnits` (`vit_capture_records_the_forward`,
    `..._the_step`, with `vit_capture_nonempty` against an empty expectation);
    that a partly-filled buffer has exactly one writer, so a padded key's rows
    stay the zeros they were allocated with (`vit_pad_tail_unwritten`,
    `vit_pad_exists`); and what the contractions cost in assumptions
    (`vit_law_bill`) — 435 launches through a primitive that states one weak
    equation, `Law.cublasGemmIsSomeReassoc`, against 806 whose kernels this
    development emits.  The law is stated of a batch of one, and that the
    emitted calls are in that configuration is decided from their recovered
    arguments (`vit_fwd_contractions_are_plain_gemms`,
    `vit_step_contractions_are_plain_gemms`). -/
def roots : List Name :=
  [ -- the model the two launch streams are recovered through, sound on the
    -- bodies that ship
    `Vit.vFwd_entry_sound
  , `Vit.vFwd_entry_arg
  , `Vit.vFwd_entry_size
  , `Vit.vFwd_entry_ok
  , `Vit.vStep_entry_ok
  , `Vit.vFwd_blocks_ty_ok
  , `Vit.vStep_blocks_ty_ok
  , `Vit.vit_deps_ordered
  , `Vit.vit_fusion_sound
  , `Vit.vit_fusion_fired
  , `Vit.vit_out_survives
  , `Vit.vit_pad_tail_unwritten
  , `Vit.vit_pad_exists
  , `Vit.vit_regs_ok
  , `Vit.vit_regs_ok_rejects_the_scratch
  , `Vit.vit_targets
  , `Vit.vit_printable
  , `Vit.vit_bufs_bound
  , `Vit.vit_proven_units_are_printed
  , `Vit.vit_law_bill
  , `Vit.vit_law_bill_nonempty
  , `Vit.vit_lowering_sound
  , `Vit.vit_tape_is_the_term
  , `Vit.vit_capture_records_the_forward
  , `Vit.vit_capture_records_the_step
  , `Vit.vit_capture_nonempty
  , `Vit.vit_groups_sound
  , `Vit.vit_mask_is_padding
  , `Vit.vit_mask_bits
  , `Vit.vit_mask_below_floor
  , `Vit.vit_mask_not_from_host
  , `Vit.vit_padded_key_is_weightless
  , `Vit.vit_mask_law_applies
  , `Vit.vit_stmts_flat_idxfree
  , `Vit.vit_ptx_exact
  , `Vit.vit_slot_holds_the_kernel
  , `Vit.vit_fwd_contractions_are_plain_gemms
  , `Vit.vit_step_contractions_are_plain_gemms ]

/-- **What this artifact does not yet state.**

    One thing, and it is the seam between the two halves that *are* stated.

    `vit_lowering_sound` says the shipped tape's `TOp.den` fold is the model's.
    `vit_capture_records_the_forward` / `_the_step` say the emitted CLIF makes
    exactly the launches the tape describes.  `vit_groups_sound` says the
    conditions under which grouping operations into one kernel is sound all
    hold.  What is missing is the bridge: that **launching a group performs its
    members' denotations**.

    The library half of that bridge now exists.  `group_runs_as_pipeline`
    (`ML/Interchange.lean`) proves that one launch of `s₁ ; … ; sₙ` over `g`
    blocks lands what running each member over all `g` blocks lands — the loop
    interchange — from one hypothesis, that updates from different blocks
    commute, and `commute_of_footprints` reduces that to disjoint writes plus
    chunk-locality.  It is not vacuous: `mapStages_group_runs_as_pipeline`
    discharges every hypothesis for two fused elementwise passes, including the
    case that makes fusion worth doing, where the second reads what the first
    wrote.

    What ViT still lacks is narrower than it looks, and narrower than it was
    thought to be.  A `StageSpec` per `TOp` — `dom`, `val` and exclusivity —
    already exists: `Node.stage?` builds one per constructor and
    `TOp.step_den` proves each stage's memory action is that op's `den`.  Two
    things stand between that and a shipped group:

    * **the renaming.**  `vUnitStmt` renames each member onto the group's own
      table (`compactMap`), and the stages are in global buffer numbering, so
      the group's elaborated block is `groupBlk` of the members' blocks only up
      to that renaming.
    * **the reads.**  `commute_of_reads` wants each block's reads to miss every
      other block's writes.  The read footprints themselves now exist — one
      `ReadsIn` lemma per stage builder, for all nine shapes this model uses —
      so what is left is the arithmetic: that *these* index expressions, at
      this model's extents, land inside the block's own chunk.  That is the
      content `vChunkLocal` decides syntactically and nothing yet states.

    Kept here rather than left implicit because both ends being proven is what
    this development has repeatedly mistaken for the middle being proven. -/
def notYetStated : List String :=
  [ "that a group's renamed statement elaborates to groupBlk of its members' \
     stage blocks (the compactMap seam)",
    "that this model's index expressions land in the block's own chunk, which \
     is what turns the per-stage ReadsIn lemmas into commute_of_reads",
    "and therefore: that launching a group performs its members' denotations, \
     and that n training steps are n applications of one step" ]

end VitScan

/-- Claims that rest on the compiler, via `native_decide`. -/
def nativeRoster : List Name :=
  [ `Vit.vFwd_entry_sound
   , `Vit.vFwd_entry_arg
   , `Vit.vFwd_entry_size
   , `Vit.vFwd_entry_ok
   , `Vit.vStep_entry_ok
   , `Vit.vFwd_blocks_ty_ok
   , `Vit.vStep_blocks_ty_ok
   , `Vit.vit_deps_ordered
   , `Vit.vit_fusion_fired
   , `Vit.vit_out_survives
   , `Vit.vit_pad_tail_unwritten
   , `Vit.vit_pad_exists
   , `Vit.vit_regs_ok
   , `Vit.vit_law_bill
   , `Vit.vit_law_bill_nonempty
   , `Vit.vit_capture_records_the_forward
   , `Vit.vit_capture_records_the_step
   , `Vit.vit_capture_nonempty
   , `Vit.vit_groups_sound
   , `Vit.vit_mask_bits
   , `Vit.vit_mask_below_floor
   , `Vit.vit_padded_key_is_weightless
   , `Vit.vit_mask_law_applies
   , `Vit.vit_stmts_flat_idxfree
   , `Vit.vit_ptx_exact
   , `Vit.vit_slot_holds_the_kernel
   , `Vit.vit_fwd_contractions_are_plain_gemms
   , `Vit.vit_step_contractions_are_plain_gemms ]

open TrustScan VitScan in
#eval runScan "vit" roots nativeRoster

#eval do
  IO.println s!"[vit] roots scanned: {VitScan.roots.length}"
  for s in VitScan.notYetStated do
    IO.println s!"[vit] NOT YET STATED: {s}"
  let k := AlgorithmLib.ML.VendorKernel.cublasSgemmAt1OnStream
  let bill := (AlgorithmLib.ML.VendorKernel.assumes k).map AlgorithmLib.ML.Law.title
  IO.println s!"[vit] all contractions go through {k.symbol} at a batch of one, \
which states: {bill}"
  IO.println s!"[vit] and withholds: {AlgorithmLib.ML.VendorKernel.withholds k}"
