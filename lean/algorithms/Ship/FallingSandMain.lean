import Ship.FallingSandDisplay
import Ship.FallingSandHeadless

/-!
# FallingSand's game neither misuses a call nor faults

The game's `main`, proven by the condition generator from the typestate it
starts in; the two cases of the display oracle are proven in files of their
own so they build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

/-- **The game neither misuses a call nor faults**, whether or not there is a display. -/
theorem FallingSand_main_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : FallingSandMain w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (FallingSand.mainBody : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (FallingSand.mainBody : Prog Slot Lvl Unit)) ≠ .fault m := by
  have h' : FallingSandMainOn w.display w := by
    intro f hf
    rcases List.mem_cons.mp hf with rfl | hf
    · rfl
    · exact h f hf
  cases hd : w.display
  · rw [hd] at h'; exact FallingSand_main_headless_no_misuse cfg dataLen outLen w h' m
  · rw [hd] at h'; exact FallingSand_main_display_no_misuse cfg dataLen outLen w h' m

end WgpuSafe

#print axioms WgpuSafe.FallingSand_main_no_misuse
