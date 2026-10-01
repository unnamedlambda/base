import AlgorithmLib.Proof.Typestate
import Demo.WindowDemo

/-!
# WindowDemo's game neither misuses a call nor faults

The game's `main`, proven by the condition generator from the typestate it
starts in, whether or not there is a display. Not faulting means every load and
store finds the bytes it reaches, and every operation answers; the game reads
the events a poll answered only when their count fits the slots it gave the
poll. Its tests are proven in `WgpuWindowDemo`, which builds beside this file.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

/-- What the game starts in, but for whether there is a display. -/
abbrev WindowDemoMain : World → Prop := Contracts.TState.holds
  [.part .gpuAdapter true, .part .frozen false, .part .gpu false, .part .win false, .room .arena 930008,
   .cstrIn (regionBase .arena + 80) 4096]

abbrev WindowDemoMainOn (d : Bool) : World → Prop := Contracts.TState.holds
  [.part .display d, .part .gpuAdapter true, .part .frozen false, .part .gpu false, .part .win false, .room .arena 930008,
   .cstrIn (regionBase .arena + 80) 4096]

theorem WindowDemo_main_fine :
    Fine (emitGo (WindowDemo.mainBody : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem WindowDemo_main_display_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : WindowDemoMainOn true w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.mainBody : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.mainBody : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (WindowDemoMainOn true) (runArgs dataLen outLen) (by prog_vc (WindowDemoMainOn true))
    WindowDemo_main_fine rfl h m

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem WindowDemo_main_headless_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : WindowDemoMainOn false w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.mainBody : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.mainBody : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (WindowDemoMainOn false) (runArgs dataLen outLen) (by prog_vc (WindowDemoMainOn false))
    WindowDemo_main_fine rfl h m

/-- **The game neither misuses a call nor faults**, whether or not there is a display. -/
theorem WindowDemo_main_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : WindowDemoMain w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.mainBody : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.mainBody : Prog Slot Lvl Unit)) ≠ .fault m := by
  have h' : WindowDemoMainOn w.display w := by
    intro f hf
    rcases List.mem_cons.mp hf with rfl | hf
    · rfl
    · exact h f hf
  cases hd : w.display
  · rw [hd] at h'; exact WindowDemo_main_headless_no_misuse cfg dataLen outLen w h' m
  · rw [hd] at h'; exact WindowDemo_main_display_no_misuse cfg dataLen outLen w h' m

end WgpuSafe

#print axioms WgpuSafe.WindowDemo_main_no_misuse
