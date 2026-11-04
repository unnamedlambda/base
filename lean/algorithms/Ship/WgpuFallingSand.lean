import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import AlgorithmLib.Check.WgState
import Demo.FallingSand

/-!
# FallingSand's tests neither misuse a call nor fault

Each entry point, proven by the condition generator from the typestate it
starts in, for any lengths of data and output the caller hands over. The
tests call wgpu directly and start from what `Wg.App.loadTS` states of
wgpu and of its memory.

Not faulting means every load and store finds the bytes it reaches, and every
operation answers, where the caller's data and output hold the lengths it
passes. The game itself is proven in `FallingSandMain`, which builds beside
this file.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

-- FallingSand: the headless tests

/-- What the tests start from: wgpu untouched, in its terms. -/
abbrev FallingSandTestIO (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  ([.roomArg .data dl, .roomArg .out ol] ++ FallingSand.app.loadTS)

ship_entry FallingSand_grainFalls : FallingSand.testGrainFalls under FallingSandTestIO dataLen outLen

ship_entry FallingSand_conservation : FallingSand.testConservation under FallingSandTestIO dataLen outLen

end WgpuSafe

#print axioms WgpuSafe.FallingSand_grainFalls_no_misuse
#print axioms WgpuSafe.FallingSand_conservation_no_misuse
