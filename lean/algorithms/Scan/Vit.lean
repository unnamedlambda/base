import Scan.MlSurface
import Vit.Guards
import Vit.Dag
import Vit.Body
import Vit.Host
import Vit.Slot
/-!
  # What the ViT artifact's claims rest on — computed, not documented

  The fourth scanner, for the same reason there are three already: each
  generator defines its own `main`, so they cannot share a module.  The
  machinery is `ScanCore`, the surface `MlSurface`.

  `MlpScan` names about a hundred and fifty roots; this names thirty, and
  `notYetStated` what they do not reach.  A scanner over a root list does not
  say the artifact is trustworthy — it says exactly how much of it has been
  argued.

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
    distinguishes a guard from a decoration.  And that the streams and events each capture
    records order every conflicting pair of the operations it records
    (`vit_captures_ordered`): a capture of a range drops its waits on what ran
    before it, so an ordering the whole tape had only through an operation
    outside the range is one that capture lacks.

    **The tape.**  That what is emitted computes what the model's own lowering
    computes (`vit_fusion_sound`, `vit_lowering_sound`, `vit_tape_is_the_term`)
    — through two rounds of fusion, orientation, producer sinking and
    re-chunking, onto a tape nothing authors: the forward is the term's
    compilation, the backward is `Ten.backwardFrom` of it, the update one
    `upd2` per parameter.  Plus the two facts that stop it being vacuous —
    that sites were really taken, and that the buffer the answer is read from is
    not one of the temporaries fusion removed.

    **The kernels.**  That launching every group's kernel runs the group's
    statement (`vit_body`): each word a lane of the launch touches is left as
    the warp machine's run of the statement for the lane's block leaves it,
    from the launch's memory — from the decided launch obligations
    (`vit_static`, per distinct key) and every kernel emitted whole
    (`vit_fine`).  And that the units these are stated over are exactly the
    units that print (`vit_proven_units_are_printed`), and that the slot each
    launch names holds that group's own kernel (`vit_slot_holds_the_kernel`),
    which is what the dedup's hash key would otherwise have to be trusted for.

    **The seams outward.**  That a partly-filled buffer has exactly one writer, so a padded key's rows
    stay the zeros they were allocated with (`vit_pad_tail_unwritten`,
    `vit_pad_exists`); and what the contractions cost in assumptions
    (`vit_law_bill`) — 435 launches through a primitive that states one weak
    equation, `Law.cublasGemmIsSomeReassoc`, against 806 whose kernels this
    development emits.  The law is stated of a product at a batch of one, and
    every contraction here is a strided-batched product issued at a batch of
    one. -/
def roots : List Name :=
  [ `Vit.vit_deps_ordered
  , `Vit.vit_captures_ordered
  , `Vit.vit_fusion_sound
  , `Vit.vit_fusion_fired
  , `Vit.vit_out_survives
  , `Vit.vit_pad_tail_unwritten
  , `Vit.vit_pad_exists
  , `Vit.vit_proven_units_are_printed
  , `Vit.vit_law_bill
  , `Vit.vit_law_bill_nonempty
  , `Vit.vit_lowering_sound
  , `Vit.vit_tape_is_the_term
  , `Vit.vit_groups_sound
  , `Vit.vit_mask_is_padding
  , `Vit.vit_mask_bits
  , `Vit.vit_mask_below_floor
  , `Vit.vit_mask_not_from_host
  , `Vit.vit_padded_key_is_weightless
  , `Vit.vit_mask_law_applies
  , `Vit.vit_body
  , `Vit.vit_group
  , `Vit.vit_launch
  , `Vit.vit_launch_keeps
  , `Vit.vit_drv_launch
  , `Vit.vit_schedule_computes_tape
  , `Vit.vit_schedule_computes_tape_with
  , `Vit.vit_schedule_computes_vendor_tape
  , `Vit.vit_fwd_computes_tape_with
  , `Vit.vit_gemm_words
  , `Vit.vit_slot_holds_the_kernel ]

/-- **What this artifact does not yet state.**

    The calls every run of a capture makes, which the checker reads off the
    driver calls once it reads bodies this long; and the composition of the
    launches into the tape.

    One launch is stated end to end: `vit_group` says launching a group's
    kernel leaves every word a lane touches as the group's operations'
    denotations, folded in order, leave it, at the words the hardware computes
    on.  It rests on `vit_body` (the launch runs the group's statement),
    `GroupLink.group_den` (a block's run of the renamed members is their
    pipeline, `TOp.step_den` making each step the op's `den`), and
    `vit_static`'s decided fact that no block stores a word another touches —
    so no chunk-locality arithmetic is needed: a stage's value reads only the
    words its statement touches (`GroupRun.val_reads`).

    `vit_launch` extends it to every word of the slots: a word no block
    stores keeps its value on the hardware and under the denotations, and
    `vit_launch_keeps` that the slots no member outputs keep their words: the
    kernel writes only the slots `vKSpec` flags as written. `vit_drv_launch`
    takes it through the driver: a `cuLaunchKernel` of the group leaves every
    word of device memory as the group's operations, folded over the whole
    memory, leave it.
    `vit_schedule_computes_tape` says the units' operations, folded in the
    order the schedule issues them, compute what the tape `vFused` does, and
    `VitCalls.run_calls` that the forward run makes its units' calls in
    that order, and `VitStep.step_agree` that a training step's calls,
    composed, leave the words the tape does; each group's statement kernel
    runs over buffers of the layout's sizes (`vit_runs`). At the caller's
    end, `VitIo.seed_in` puts the caller's data, as handed over, in the
    backward's start buffer, and `VitIo.fetch_out` leaves the logits buffer
    in the caller's output byte for byte.

    Kept here rather than left implicit because both ends being proven is what
    this development has repeatedly mistaken for the middle being proven. -/
def notYetStated : List String :=
  [ "that fetchAny downloads only from a live buffer within its size: the \
     buffer is chosen at run time, so its pointer and size are loads at \
     an address a run-time value chooses, which the checker does not yet \
     read",
    "that every run of a capture makes exactly the calls the tape and its \
     schedule say: the capture bodies' call traces, which check_calls decides \
     on the driver calls they make once it reads a body this long",
    "and therefore: that n training steps are n applications of one step" ]

end VitScan

/-- Claims that rest on the compiler, via `native_decide`. -/
def nativeRoster : List Name :=
  [ `Vit.vit_deps_ordered
   , `Vit.vit_captures_ordered
   , `Vit.vit_fusion_fired
   , `Vit.vit_out_survives
   , `Vit.vit_pad_tail_unwritten
   , `Vit.vit_pad_exists
   , `Vit.vit_law_bill
   , `Vit.vit_law_bill_nonempty
   , `Vit.vit_groups_sound
   , `Vit.vit_mask_bits
   , `Vit.vit_mask_below_floor
   , `Vit.vit_padded_key_is_weightless
   , `Vit.vit_mask_law_applies
   , `Vit.vit_body
   , `Vit.vit_group
   , `Vit.vit_launch
   , `Vit.vit_launch_keeps
   , `Vit.vit_drv_launch
   , `Vit.vit_schedule_computes_tape
   , `Vit.vit_schedule_computes_tape_with
   , `Vit.vit_schedule_computes_vendor_tape
   , `Vit.vit_fwd_computes_tape_with
   , `Vit.vit_slot_holds_the_kernel ]

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
