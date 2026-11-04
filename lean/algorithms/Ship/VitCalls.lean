import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import AlgorithmLib.MLHost.AppTrace
import AlgorithmLib.MLHost.AppRun
import Vit.Algorithm
import Ship.VitState
import Ship.VitDag

/-!
# The calls the ViT's ranges make

Every run of a range `0..hi` that ends makes exactly its units' calls, in the
schedule's order (`vSteps`): each group's kernel launch, each contraction's
strided-batched product at a batch of one on the default handle, then a
context wait (`RangeCalls`, for the forward `run_calls`, by `check_calls`).
What device memory does in such a run is `ML.calls_tape`'s (`VitWords`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.Check (DevPath ExtRun extRun_of CallRec)

namespace VitCalls

/-- What `main` leaves (`VitSafe.main_reaches`). -/
abbrev Run : World → Prop := Contracts.TState.holds (VitState.after 0 0)

/-- **Every run of the range `0..hi` that ends makes its units' calls**, in
    the schedule's order, then waits. -/
def RangeCalls (hi : Nat) : Prop :=
  ∀ (cfg : Cfg) (dataLen outLen : UInt64) (w : World), Run w → ∀ {obs : List Obs} {w' : World},
    Sem.run cfg (runArgs dataLen outLen) w (emit (Vit.vRangeFn 0 hi : Prog Slot Lvl Unit)) = .ok obs w' →
    (Vit.vApp.callsWith (Vit.vApp.stridedRec (drvHandle kBlas 0) [0x3f800000, 0]) (vSteps 0 hi)).Covers w obs ∧
      DevPath w w'

set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
/-- **Every run of the forward that ends makes its units' calls**, in the
    schedule's order, then waits. -/
theorem run_calls : RangeCalls Vit.VFWD_N := by
  intro cfg dataLen outLen w h obs w' hr
  check_calls

theorem after_has : ((VitState.after 0 0).contains (.part (.has .cuda) true) &&
    (VitState.after 0 0).contains (.part (.has .cublas) true) &&
    (VitState.after 0 0).contains (.part .capturing false) &&
    (VitState.after 0 0).contains .oracles) = true := by native_decide

end VitCalls

#print axioms VitCalls.run_calls
