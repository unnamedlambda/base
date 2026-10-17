import AlgorithmLib.Check.Entry
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

ship_entry Raymarch_moveForward : Raymarch.testMoveForward under RaymarchTestIO dataLen outLen

ship_entry Raymarch_strafeRight : Raymarch.testStrafeRight under RaymarchTestIO dataLen outLen

ship_entry Raymarch_riseClamp : Raymarch.testRiseClamp under RaymarchTestIO dataLen outLen

ship_entry Raymarch_quitOnClose : Raymarch.testQuitOnClose under RaymarchTestIO dataLen outLen

ship_entry Raymarch_renderScene : Raymarch.testRenderScene under RaymarchTestIO dataLen outLen

-- Raymarch: the game, on a display or on none

end WgpuSafe

#print axioms WgpuSafe.Raymarch_moveForward_no_misuse
#print axioms WgpuSafe.Raymarch_strafeRight_no_misuse
#print axioms WgpuSafe.Raymarch_riseClamp_no_misuse
#print axioms WgpuSafe.Raymarch_quitOnClose_no_misuse
#print axioms WgpuSafe.Raymarch_renderScene_no_misuse
