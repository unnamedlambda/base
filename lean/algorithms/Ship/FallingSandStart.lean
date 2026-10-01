import AlgorithmLib.Proof.Typestate
import Demo.FallingSand

/-!
# FallingSand's game: its start

The game's `main`, proven by the condition generator from the typestate it
starts in; the two cases of the display oracle are proven in files of their
own so they build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

-- FallingSand: the game, on a display or on none

/-- What the game starts in, but for whether there is a display. -/
abbrev FallingSandMain : World → Prop := Contracts.TState.holds
  [.part .gpuAdapter true, .part .frozen false, .part .gpu false, .part .win false, .room .arena 2783568,
   .cstrIn (regionBase .arena + 12352) 2048,
   .cstrIn (regionBase .arena + 8256) 2048,
   .cstrIn (regionBase .arena + 64) 8192,
   .cstrIn (regionBase .arena + 10304) 2048]

abbrev FallingSandMainOn (d : Bool) : World → Prop := Contracts.TState.holds
  [.part .display d, .part .gpuAdapter true, .part .frozen false, .part .gpu false, .part .win false, .room .arena 2783568,
   .cstrIn (regionBase .arena + 12352) 2048,
   .cstrIn (regionBase .arena + 8256) 2048,
   .cstrIn (regionBase .arena + 64) 8192,
   .cstrIn (regionBase .arena + 10304) 2048]

theorem FallingSand_main_fine :
    Fine (emitGo (FallingSand.mainBody : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩


end WgpuSafe
