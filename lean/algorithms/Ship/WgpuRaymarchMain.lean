import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import AlgorithmLib.Check.WgState
import Demo.RaymarchDemo

/-!
# Raymarch's game neither misuses a call nor faults

The game's `main`, proven by the condition generator from the typestate it
starts in, whether or not there is a display. Not faulting means every load and
store finds the bytes it reaches, and every operation answers; the game reads
the events a poll answered only when their count fits the slots it gave the
poll. Its tests are proven in `WgpuRaymarch`, which builds beside this file.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

/-- What the game starts in: wgpu untouched, in its terms; with a display or
    without, which the window library's answer tells the game. -/
abbrev RaymarchMain : World → Prop := Contracts.TState.holds Raymarch.app.loadTS

ship_entry Raymarch_main : Raymarch.mainBody under RaymarchMain

end WgpuSafe

#print axioms WgpuSafe.Raymarch_main_no_misuse
