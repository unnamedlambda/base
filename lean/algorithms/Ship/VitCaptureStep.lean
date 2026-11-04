import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Vit.Algorithm
import Ship.VitState

/-!
# The ViT's training-step capture and replay neither misuse a call nor fault

`captureStep` runs the backward and the updates once on the default stream,
then captures them as the graph their dependences allow, as `capture` does
the forward (`VitCapture`). `replay` launches the forward's graph and waits.
`captureStep` starts from what `main` leaves (`VitState.Run`), `replay` once
`capture` has made the forward's graph. Each is checked by compiled
evaluation (`ship_entry`); apart from `VitCapture` so the checks build side
by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace VitCapture
open Contracts (Fact TState)

ship_entry captureStep : Vit.vCaptureAt Vit.VFWD_N Vit.VSTEP_N Vit.VGRAPH_STEP Vit.VEXEC_STEP under VitState.Run


ship_entry replay : Vit.vReplayAt Vit.VEXEC_FWD 1 under VitState.Replay

end VitCapture

#print axioms VitCapture.captureStep_no_misuse
