import AlgorithmLib.Proof.Typestate
import Demo.WindowDemo

/-!
# WindowDemo's tests neither misuse a call nor fault

Each entry point, proven by the condition generator from the typestate it
starts in, for any lengths of data and output the caller hands over. A program
starts from its initial memory: the shader text it hands `gpuCreatePipeline`
is a NUL-terminated string inside the field the layout gives it (`cstrIn`),
which the calls before the pipeline leave alone. Buffer sizes need no facts:
an upload or download of the wrong size answers `-1`.

Not faulting means every load and store finds the bytes it reaches, and every
operation answers, where the caller's data and output hold the lengths it
passes. The game itself is proven in `WgpuWindowDemoMain`, which builds beside
this file.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

-- WindowDemo: the headless tests

abbrev WindowDemoTest : World → Prop := Contracts.TState.holds
  [.part .gpuAdapter true, .part .frozen false, .part .gpu false, .room .arena 930008, .cstrIn (regionBase .arena + 80) 4096]

/-- `WindowDemoTest`, with the caller's data and output holding the lengths it passes. -/
abbrev WindowDemoTestIO (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.roomArg .data dl, .roomArg .out ol, .part .gpuAdapter true, .part .frozen false, .part .gpu false, .room .arena 930008, .cstrIn (regionBase .arena + 80) 4096]

theorem WindowDemo_layout : WindowDemo.layoutMeta.totalSize = 930008 ∧ WindowDemo.f.shader.offset = 80 :=
  ⟨by decide +kernel, by decide +kernel⟩

theorem WindowDemo_renderPixel_fine :
    Fine (emitGo (WindowDemo.testRenderPixel : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem WindowDemo_renderPixel_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : WindowDemoTestIO dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.testRenderPixel : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.testRenderPixel : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (WindowDemoTestIO dataLen outLen) (runArgs dataLen outLen) (by prog_vc (WindowDemoTestIO dataLen outLen))
    WindowDemo_renderPixel_fine rfl h m

theorem WindowDemo_quitOnClose_fine :
    Fine (emitGo (WindowDemo.testQuitOnClose : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem WindowDemo_quitOnClose_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : WindowDemoTestIO dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.testQuitOnClose : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.testQuitOnClose : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (WindowDemoTestIO dataLen outLen) (runArgs dataLen outLen) (by prog_vc (WindowDemoTestIO dataLen outLen))
    WindowDemo_quitOnClose_fine rfl h m

theorem WindowDemo_moveRight_fine :
    Fine (emitGo (WindowDemo.testMoveRight : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem WindowDemo_moveRight_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : WindowDemoTestIO dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.testMoveRight : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.testMoveRight : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (WindowDemoTestIO dataLen outLen) (runArgs dataLen outLen) (by prog_vc (WindowDemoTestIO dataLen outLen))
    WindowDemo_moveRight_fine rfl h m

theorem WindowDemo_moveLeft_fine :
    Fine (emitGo (WindowDemo.testMoveLeft : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem WindowDemo_moveLeft_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : WindowDemoTestIO dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.testMoveLeft : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.testMoveLeft : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (WindowDemoTestIO dataLen outLen) (runArgs dataLen outLen) (by prog_vc (WindowDemoTestIO dataLen outLen))
    WindowDemo_moveLeft_fine rfl h m

theorem WindowDemo_moveUpClamp_fine :
    Fine (emitGo (WindowDemo.testMoveUpClamp : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem WindowDemo_moveUpClamp_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : WindowDemoTestIO dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.testMoveUpClamp : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (WindowDemo.testMoveUpClamp : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (WindowDemoTestIO dataLen outLen) (runArgs dataLen outLen) (by prog_vc (WindowDemoTestIO dataLen outLen))
    WindowDemo_moveUpClamp_fine rfl h m

-- WindowDemo: the game, on a display or on none

end WgpuSafe

#print axioms WgpuSafe.WindowDemo_renderPixel_no_misuse
#print axioms WgpuSafe.WindowDemo_quitOnClose_no_misuse
#print axioms WgpuSafe.WindowDemo_moveRight_no_misuse
#print axioms WgpuSafe.WindowDemo_moveLeft_no_misuse
#print axioms WgpuSafe.WindowDemo_moveUpClamp_no_misuse
