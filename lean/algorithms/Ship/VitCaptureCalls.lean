import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Vit.Algorithm
import Ship.VitState
import Ship.VitDag

/-!
# The calls the ViT's forward capture makes

Every run of `capture` that ends takes the forward's steps on the default
stream and waits for that stream, then begins a capture on the capture stream and makes the
capture's calls over the pooled streams (`VitDag.vCapCalls`), ends it,
instantiates and uploads the graph, and waits (`capture_calls`, by
`check_calls`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace VitCaptureCalls

/-- The calls `capture` makes. -/
def captureRecs : List Check.CallRec :=
  ((VitCalls.vSteps 0 Vit.VFWD_N).map ML.DevCall.ofStep).map
      (ML.DevCall.record Vit.vApp (drvHandle kBlas 0) 0x3f800000 0) ++
    [ML.wordRec .streamSynchronize [0]] ++ ML.dagRecs Vit.vApp [0x3f800000, 0] (VitDag.vCapCalls 0 Vit.VFWD_N) VitDag.S0
      (Drv.App.at_ Vit.VGRAPH_FWD) (Drv.App.at_ Vit.VEXEC_FWD)

set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
/-- **Every run of `capture` that ends makes its calls**, the device changed
    only by those calls. -/
theorem capture_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : VitState.Run w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w
      (emit (Vit.vCaptureAt 0 Vit.VFWD_N Vit.VGRAPH_FWD Vit.VEXEC_FWD : Prog Slot Lvl Unit)) = .ok obs w') :
    (Check.Tr.lit captureRecs).Covers w obs ∧ Check.DevPath w w' := by
  check_calls

end VitCaptureCalls

#print axioms VitCaptureCalls.capture_calls
