import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import AlgorithmLib.Check.WgState
import Demo.WindowDemo

/-!
# WindowDemo's tests neither misuse a call nor fault

Each entry point, proven by the condition generator from the typestate it
starts in, for any lengths of data and output the caller hands over. The
render test calls wgpu directly and starts from what `Wg.App.loadTS` states
of wgpu and of its memory; the logic tests call nothing.

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

/-- What the logic tests start from: memory, with the caller's data and
    output holding the lengths it passes. -/
abbrev WindowDemoTestIO (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.roomArg .data dl, .roomArg .out ol, .room .arena WindowDemo.layoutMeta.totalSize]

/-- What the render test starts from: wgpu untouched, in its terms. -/
abbrev WindowDemoRenderIO (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  ([.roomArg .data dl, .roomArg .out ol] ++ WindowDemo.app.loadTS)

ship_entry WindowDemo_renderPixel : WindowDemo.testRenderPixel under WindowDemoRenderIO dataLen outLen

ship_entry WindowDemo_quitOnClose : WindowDemo.testQuitOnClose under WindowDemoTestIO dataLen outLen

ship_entry WindowDemo_moveRight : WindowDemo.testMoveRight under WindowDemoTestIO dataLen outLen

ship_entry WindowDemo_moveLeft : WindowDemo.testMoveLeft under WindowDemoTestIO dataLen outLen

ship_entry WindowDemo_moveUpClamp : WindowDemo.testMoveUpClamp under WindowDemoTestIO dataLen outLen

end WgpuSafe

#print axioms WgpuSafe.WindowDemo_renderPixel_no_misuse
#print axioms WgpuSafe.WindowDemo_quitOnClose_no_misuse
#print axioms WgpuSafe.WindowDemo_moveRight_no_misuse
#print axioms WgpuSafe.WindowDemo_moveLeft_no_misuse
#print axioms WgpuSafe.WindowDemo_moveUpClamp_no_misuse
