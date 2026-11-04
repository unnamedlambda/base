import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Vit.Algorithm
import Ship.VitState
import Ship.VitDag

/-!
# What the ViT's forward capture leaves

Every run of `capture` that ends, from what `main` leaves, leaves what a
replay of the forward's graph starts from (`VitState.Replay`), by the
checker read by compiled evaluation (`check_reach`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace VitCaptureReach

set_option Elab.async true in
set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
/-- **What `capture` leaves is what a replay starts from.** -/
theorem capture_reaches (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : VitState.Run w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w)
      (emit (Vit.vCaptureAt 0 Vit.VFWD_N Vit.VGRAPH_FWD Vit.VEXEC_FWD : Prog Slot Lvl Unit))
      { ok := fun _ w' => VitState.Replay w', faultOk := false } := by
  check_reach

end VitCaptureReach

#print axioms VitCaptureReach.capture_reaches
