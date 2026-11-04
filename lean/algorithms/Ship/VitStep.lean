import Ship.VitWords

/-!
# The words a ViT training step leaves are the tape's

A training step (`Vit.vRangeFn 0 VSTEP_N`, the forward, the backward and the
update) takes its units' steps in the schedule's order (`step_calls`, by
`check_calls`), and its units compute the whole tape
(`Vit.vit_step_pull`). So every run of it, on a device honest about each
launch and each product, leaves every word inside each buffer as the tape's
operations, all of them, leave it (`step_agree`).

A plain file, not a module, as the Ship files it reads are.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
  AlgorithmLib.HProg.Contracts AlgorithmLib.Device AlgorithmLib.ML AlgorithmLib.Kernel

namespace VitStep

set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
/-- **Every run of a training step that ends makes its units' calls**, in
    the schedule's order, then waits. -/
theorem step_calls : VitCalls.RangeCalls Vit.VSTEP_N := by
  intro cfg dataLen outLen w h obs w' hr
  check_calls

theorem step_units : (VitWords.stepUnits 0 Vit.VSTEP_N).map Prod.snd = Vit.vRangeUnits 0 Vit.VSTEP_N := by
  native_decide
theorem step_proven : (VitWords.stepUnits 0 Vit.VSTEP_N).all VitWords.provenOk = true := by native_decide
theorem step_ok : ML.stepsOk Vit.vLayout Vit.vApp (VitWords.stepOps 0 Vit.VSTEP_N) = true := by native_decide

/-- A training step's facts. -/
theorem stepFacts : VitWords.RangeFacts Vit.VSTEP_N :=
  ⟨step_calls, step_units, step_proven, step_ok, Vit.vit_step_pull⟩

/-- **The words a ViT training step leaves are the tape's**: every run of a
    training step that ends, from what `main` leaves, on a device honest about
    each group's launch (`TapeLayout.KernelHonest`) and about its products,
    leaves every word inside each buffer, at the size the layout makes it, as
    the tape's operations — the forward, the backward and the update — leave
    it from the memory the run started with, each contraction read as the
    vendor computes it. -/
theorem step_agree (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : VitCalls.Run w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (Vit.vRangeFn 0 Vit.VSTEP_N : Prog Slot Lvl Unit)) = .ok obs w')
    (hv : VendorGemmHonest w.vendor) (hvk : VendorKeeps w.vendor)
    (hHon : ∀ p ∈ VitWords.stepUnits 0 Vit.VSTEP_N, ∀ j, p.1 = .kernel j →
      Vit.vLayout.KernelHonest Vit.vApp w j (Vit.vUnitOps p.2) hw) :
    Vit.vLayout.Agree (devMem w'.dev)
      ((Vit.vFused.take Vit.VSTEP_N).foldl (fun M op => TOp.denV hw op M) (wordView (devMem w.dev))) :=
  VitWords.range_agree stepFacts hw hw32 cfg dataLen outLen w h hr hv hvk hHon

end VitStep

#print axioms VitStep.step_agree
