import Ship.FallingSandStart

/-!
# FallingSand's game with no display

The game's `main`, proven by the condition generator from the typestate it
starts in; the two cases of the display oracle are proven in files of their
own so they build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem FallingSand_main_headless_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : FallingSandMainOn false w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (FallingSand.mainBody : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (FallingSand.mainBody : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (FallingSandMainOn false) (runArgs dataLen outLen) (by prog_vc (FallingSandMainOn false))
    FallingSand_main_fine rfl h m

end WgpuSafe
