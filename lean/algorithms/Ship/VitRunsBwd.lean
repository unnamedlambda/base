import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Vit.Algorithm
import Ship.VitState

/-!
# The ViT's backward, updates and reload neither misuse a call nor fault

Each starts from what `main` leaves (`VitState.Run`). `bwd` issues the
backward's launches on the default stream and waits, `sgd` the updates';
`reload` uploads the inputs again from the caller's data and waits. Each is
checked by compiled evaluation (`ship_entry`); they sit apart from `VitRuns`
so the checks build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace VitRuns

ship_entry bwd : Vit.vRangeFn Vit.VFWD_N Vit.VBWD_N under VitState.Run

ship_entry sgd : Vit.vRangeFn Vit.VBWD_N Vit.VSTEP_N under VitState.Run

ship_entry reload : Vit.vReloadFn under VitState.Run

end VitRuns

#print axioms VitRuns.bwd_no_misuse
