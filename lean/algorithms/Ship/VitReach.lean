import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Vit.Algorithm
import Ship.VitState

/-!
# What the ViT's main leaves

`main_reaches`: every run of `main` from what it starts from
(`VitState.Load`) that ends leaves what the other entries start from
(`VitState.after`), by the checker read by compiled evaluation. Apart from
`VitSafe` so the two checks of `main` build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace VitSafe

set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
/-- **What `main` leaves** is what the other entries start from
    (`VitState.after`), by the checker read by compiled evaluation. -/
theorem main_reaches (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : VitState.Load w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) (emit (Vit.vLoadFn : Prog Slot Lvl Unit))
      { ok := fun _ w' => Contracts.TState.holds (VitState.after 0 0) w', faultOk := false } := by
  check_reach

end VitSafe

#print axioms VitSafe.main_reaches
