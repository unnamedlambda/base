import AlgorithmLib.Proof.Typestate
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

/-- What the game starts in, but for whether there is a display. -/
abbrev RaymarchMain : World → Prop := Contracts.TState.holds
  [.part .gpuAdapter true, .part .frozen false, .part .gpu false, .part .win false, .room .arena 934112,
   .cstrIn (regionBase .arena + 80) 8192]

abbrev RaymarchMainOn (d : Bool) : World → Prop := Contracts.TState.holds
  [.part .display d, .part .gpuAdapter true, .part .frozen false, .part .gpu false, .part .win false, .room .arena 934112,
   .cstrIn (regionBase .arena + 80) 8192]

theorem Raymarch_main_fine :
    Fine (emitGo (Raymarch.mainBody : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Raymarch_main_display_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : RaymarchMainOn true w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.mainBody : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.mainBody : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (RaymarchMainOn true) (runArgs dataLen outLen) (by prog_vc (RaymarchMainOn true))
    Raymarch_main_fine rfl h m

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Raymarch_main_headless_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : RaymarchMainOn false w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.mainBody : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.mainBody : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (RaymarchMainOn false) (runArgs dataLen outLen) (by prog_vc (RaymarchMainOn false))
    Raymarch_main_fine rfl h m

/-- **The game neither misuses a call nor faults**, whether or not there is a display. -/
theorem Raymarch_main_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : RaymarchMain w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.mainBody : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.mainBody : Prog Slot Lvl Unit)) ≠ .fault m := by
  have h' : RaymarchMainOn w.display w := by
    intro f hf
    rcases List.mem_cons.mp hf with rfl | hf
    · rfl
    · exact h f hf
  cases hd : w.display
  · rw [hd] at h'; exact Raymarch_main_headless_no_misuse cfg dataLen outLen w h' m
  · rw [hd] at h'; exact Raymarch_main_display_no_misuse cfg dataLen outLen w h' m

end WgpuSafe

#print axioms WgpuSafe.Raymarch_main_no_misuse
