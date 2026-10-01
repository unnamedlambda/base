module
public import Vit.Launches
meta import Vit.Launches
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open AlgorithmLib AlgorithmLib.ML

/-!
  # The backward-and-update capture makes exactly the launches the tape says

  Its own module for one reason: it is the single longest evaluation this
  artifact performs, and beside `VitDag`'s rather than after it, it is off the
  critical path of the build.
-/

namespace Vit
open AlgorithmLib.IR

/-- The larger half of a step, recovered from the shipped instruction stream by
    `Clif.launchesOf` and compared against a list derived from the tape.  See
    `vit_capture_records_the_forward` for what the comparison is worth. -/
theorem vit_step_checked :
    vStepLaunches.map vKeyOf = vCaptureKeys VFWD_N VSTEP_N
      ∧ vStepLaunches.all vPlainGemm = true
      ∧ 581 = vGemmCount vStepLaunches := by native_decide

theorem vit_capture_records_the_step :
    vStepLaunches.map vKeyOf = vCaptureKeys VFWD_N VSTEP_N := vit_step_checked.1

/-- **…and every contraction in it is a batch of one**, the configuration
    `Law.cublasGemmIsSomeReassoc` covers.  See
    `vit_fwd_contractions_are_plain_gemms`. -/
theorem vit_step_contractions_are_plain_gemms :
    vStepLaunches.all vPlainGemm = true ∧ 581 = vGemmCount vStepLaunches :=
  ⟨vit_step_checked.2.1, vit_step_checked.2.2⟩

end Vit
