import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Vit.Algorithm
import Ship.VitState

/-!
# The ViT's forward capture neither misuses a call nor faults

`capture` runs the forward once on the default stream, then captures it as
the graph its dependences allow: forked onto the pooled streams, each launch
waiting on the events its predecessors on other streams recorded, joined back,
instantiated and uploaded; `VitCaptureStep` does the same for the backward
and the updates, and replays the forward's graph. Launches on different streams that read one buffer are not a
race: a kernel writes only its members' outputs (`Vit.vKSpec`), and the
checker reads every other buffer it binds as read.

It starts from what `main` leaves (`VitState.Run`, which
`VitSafe.main_reaches` proves of it) and is checked by compiled evaluation
(`ship_entry`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace VitCapture
open Contracts (Fact TState)

ship_entry capture : Vit.vCaptureAt 0 Vit.VFWD_N Vit.VGRAPH_FWD Vit.VEXEC_FWD under VitState.Run

end VitCapture

#print axioms VitCapture.capture_no_misuse
