import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import AlgorithmLib.Check.WgState
import Demo.RaymarchDemo

/-!
# Raymarch's tests neither misuse a call nor fault

Each entry point, proven by the condition generator from the typestate it
starts in, for any lengths of data and output the caller hands over. The
render test calls wgpu directly and starts from what `Wg.App.loadTS` states
of wgpu and of its memory; the logic tests call nothing.

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

/-- What the logic tests start from: memory, with the caller's data and
    output holding the lengths it passes. -/
abbrev RaymarchTestIO (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.roomArg .data dl, .roomArg .out ol, .room .arena Raymarch.layoutMeta.totalSize]

/-- What the render test starts from: wgpu untouched, in its terms. -/
abbrev RaymarchRenderIO (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  ([.roomArg .data dl, .roomArg .out ol] ++ Raymarch.app.loadTS)

ship_entry Raymarch_renderScene : Raymarch.testRenderScene under RaymarchRenderIO dataLen outLen

ship_entry Raymarch_moveForward : Raymarch.testMoveForward under RaymarchTestIO dataLen outLen

ship_entry Raymarch_strafeRight : Raymarch.testStrafeRight under RaymarchTestIO dataLen outLen

ship_entry Raymarch_riseClamp : Raymarch.testRiseClamp under RaymarchTestIO dataLen outLen

ship_entry Raymarch_quitOnClose : Raymarch.testQuitOnClose under RaymarchTestIO dataLen outLen

end WgpuSafe

#print axioms WgpuSafe.Raymarch_renderScene_no_misuse
#print axioms WgpuSafe.Raymarch_moveForward_no_misuse
#print axioms WgpuSafe.Raymarch_strafeRight_no_misuse
#print axioms WgpuSafe.Raymarch_riseClamp_no_misuse
#print axioms WgpuSafe.Raymarch_quitOnClose_no_misuse
