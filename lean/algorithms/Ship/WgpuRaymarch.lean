import AlgorithmLib.Proof.Typestate
import Demo.RaymarchDemo

/-!
# Raymarch's tests neither misuse a call nor fault

Each entry point, proven by the condition generator from the typestate it
starts in, for any lengths of data and output the caller hands over. A program
starts from its initial memory: the shader text it hands `gpuCreatePipeline`
is a NUL-terminated string inside the field the layout gives it (`cstrIn`),
which the calls before the pipeline leave alone. Buffer sizes need no facts:
an upload or download of the wrong size answers `-1`.

Not faulting means every load and store finds the bytes it reaches, and every
operation answers, where the caller's data and output hold the lengths it
passes. The game itself is proven in `WgpuRaymarchMain`, which builds beside
this file.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

-- Raymarch: the headless tests

abbrev RaymarchTest : World → Prop := Contracts.TState.holds
  [.part .gpuAdapter true, .part .frozen false, .part .gpu false, .room .arena 934112, .cstrIn (regionBase .arena + 80) 8192]

/-- `RaymarchTest`, with the caller's data and output holding the lengths it passes. -/
abbrev RaymarchTestIO (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.roomArg .data dl, .roomArg .out ol, .part .gpuAdapter true, .part .frozen false, .part .gpu false, .room .arena 934112, .cstrIn (regionBase .arena + 80) 8192]

theorem Raymarch_layout : Raymarch.layoutMeta.totalSize = 934112 ∧ Raymarch.f.shader.offset = 80 :=
  ⟨by decide +kernel, by decide +kernel⟩

theorem Raymarch_moveForward_fine :
    Fine (emitGo (Raymarch.testMoveForward : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Raymarch_moveForward_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : RaymarchTestIO dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.testMoveForward : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.testMoveForward : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (RaymarchTestIO dataLen outLen) (runArgs dataLen outLen) (by prog_vc (RaymarchTestIO dataLen outLen))
    Raymarch_moveForward_fine rfl h m

theorem Raymarch_strafeRight_fine :
    Fine (emitGo (Raymarch.testStrafeRight : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Raymarch_strafeRight_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : RaymarchTestIO dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.testStrafeRight : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.testStrafeRight : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (RaymarchTestIO dataLen outLen) (runArgs dataLen outLen) (by prog_vc (RaymarchTestIO dataLen outLen))
    Raymarch_strafeRight_fine rfl h m

theorem Raymarch_riseClamp_fine :
    Fine (emitGo (Raymarch.testRiseClamp : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Raymarch_riseClamp_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : RaymarchTestIO dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.testRiseClamp : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.testRiseClamp : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (RaymarchTestIO dataLen outLen) (runArgs dataLen outLen) (by prog_vc (RaymarchTestIO dataLen outLen))
    Raymarch_riseClamp_fine rfl h m

theorem Raymarch_quitOnClose_fine :
    Fine (emitGo (Raymarch.testQuitOnClose : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Raymarch_quitOnClose_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : RaymarchTestIO dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.testQuitOnClose : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.testQuitOnClose : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (RaymarchTestIO dataLen outLen) (runArgs dataLen outLen) (by prog_vc (RaymarchTestIO dataLen outLen))
    Raymarch_quitOnClose_fine rfl h m

theorem Raymarch_renderScene_fine :
    Fine (emitGo (Raymarch.testRenderScene : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Raymarch_renderScene_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : RaymarchTestIO dataLen outLen w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.testRenderScene : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raymarch.testRenderScene : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (RaymarchTestIO dataLen outLen) (runArgs dataLen outLen) (by prog_vc (RaymarchTestIO dataLen outLen))
    Raymarch_renderScene_fine rfl h m

-- Raymarch: the game, on a display or on none

end WgpuSafe

#print axioms WgpuSafe.Raymarch_moveForward_no_misuse
#print axioms WgpuSafe.Raymarch_strafeRight_no_misuse
#print axioms WgpuSafe.Raymarch_riseClamp_no_misuse
#print axioms WgpuSafe.Raymarch_quitOnClose_no_misuse
#print axioms WgpuSafe.Raymarch_renderScene_no_misuse
