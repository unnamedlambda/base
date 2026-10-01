module
public import Vit.Launches
meta import Vit.Launches
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open AlgorithmLib AlgorithmLib.ML

/-!
  # What the emitted launch sequence is checked to be

  The second of the two evaluations this artifact performs.  Its own module so
  that it runs beside `VitGuards` rather than after it.
-/

namespace Vit
open AlgorithmLib.IR

set_option synthInstance.maxSize 400 in
/-- **The whole checked bill of the launch sequence**, decided once.

    The second and last evaluation this artifact performs; `vit_checked` is the
    other.  The projections below are what to read. -/
theorem vit_checked_dag :
    vGroupGrids = true
      ∧ vGroupChunkLocal = true
      ∧ vGroupWritesDisjoint = true
      ∧ vGroupSharedReadsOk = true
      ∧ vFwdLaunches.map vKeyOf = vCaptureKeys 0 VFWD_N
      ∧ 511 = (vCaptureKeys 0 VFWD_N).length
      ∧ 1973 = (vCaptureKeys VFWD_N VSTEP_N).length
      ∧ 1241 = vUnits.length
      ∧ vFwdLaunches.all vPlainGemm = true
      ∧ 291 = vGemmCount vFwdLaunches := by native_decide

theorem vit_groups_sound :
    vGroupGrids = true ∧ vGroupChunkLocal = true
      ∧ vGroupWritesDisjoint = true ∧ vGroupSharedReadsOk = true :=
  ⟨vit_checked_dag.1, vit_checked_dag.2.1, vit_checked_dag.2.2.1,
   vit_checked_dag.2.2.2.1⟩

/-- **Seam guard: the emitted CLIF performs exactly these launches.**

    Everything above is about kernels and their composition.  Nothing so far
    said the *emitted code* makes those launches — a driver that dropped one,
    reordered two, or launched one over the wrong grid would leave every theorem
    about the tape true and the model wrong.

    So the sequence is recovered from the shipped instruction stream by
    `Clif.launchesOf` — which walks the emitted calls, not a model of them — and
    compared against a list derived from the tape.  The expectation is a
    *function of* `vUnits`, so a unit added, removed or re-grided moves both
    sides together and the guard cannot be satisfied by editing it.

    Both halves of a capture are covered: the eager pass that makes every
    module resident, and the pooled-stream pass between `beginCapture` and
    `endCapture` that is what a replay performs.  The other half of a step is
    `vit_capture_records_the_step`, in `VitDagStep`. -/
theorem vit_capture_checked :
    vFwdLaunches.map vKeyOf = vCaptureKeys 0 VFWD_N
      ∧ 511 = (vCaptureKeys 0 VFWD_N).length
      ∧ 1973 = (vCaptureKeys VFWD_N VSTEP_N).length
      ∧ 1241 = vUnits.length :=
  ⟨vit_checked_dag.2.2.2.2.1, vit_checked_dag.2.2.2.2.2.1,
   vit_checked_dag.2.2.2.2.2.2.1, vit_checked_dag.2.2.2.2.2.2.2.1⟩

theorem vit_capture_records_the_forward :
    vFwdLaunches.map vKeyOf = vCaptureKeys 0 VFWD_N := vit_capture_checked.1

/-- **Seam guard: every contraction the forward issues is a batch of one.**

    Which is the configuration `Law.cublasGemmIsSomeReassoc` is stated of, and
    the one `cublas_gemm_batch_one_vs_plain` measures bit-identical to
    `cl_cublas_sgemm`.  Decided on the *arguments recovered from the emitted
    program*, so a schedule that started batching would fail this rather than
    silently fall outside the law it is billed. -/
theorem vit_fwd_contractions_are_plain_gemms :
    vFwdLaunches.all vPlainGemm = true ∧ 291 = vGemmCount vFwdLaunches :=
  ⟨vit_checked_dag.2.2.2.2.2.2.2.2.1, vit_checked_dag.2.2.2.2.2.2.2.2.2⟩

/-- **…and there is something to record.**  An empty expectation would satisfy
    the comparison above, and `vit_capture_records_the_step`'s, and describe a
    program that launches nothing. -/
theorem vit_capture_nonempty :
    511 = (vCaptureKeys 0 VFWD_N).length
      ∧ 1973 = (vCaptureKeys VFWD_N VSTEP_N).length
      ∧ 1241 = vUnits.length := vit_capture_checked.2


end Vit
