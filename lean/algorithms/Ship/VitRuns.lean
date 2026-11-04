import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Vit.Algorithm
import Ship.VitState

/-!
# The ViT's forward and training step neither misuse a call nor fault

Each starts from what `main` leaves (`VitState.Run`, which
`VitSafe.main_reaches` proves of it). `run` issues the forward's launches on
the default stream and waits, `step` the whole training step's; the
backward, the updates and `reload` are `VitRunsBwd`'s. A range is up to the step's
1241 launches over a typestate of the 2580 allocations, 806 functions and
their parameter blocks; each is checked by compiled evaluation
(`ship_entry`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace VitRuns

ship_entry run : Vit.vRangeFn 0 Vit.VFWD_N under VitState.Run

ship_entry step : Vit.vRangeFn 0 Vit.VSTEP_N under VitState.Run

end VitRuns

#print axioms VitRuns.step_no_misuse
