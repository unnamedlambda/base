import Ship.VitWords
import Ship.VitDag
import Ship.VitCaptureCalls
import Ship.VitCaptureReach

/-!
# The words the ViT's forward capture and its replay leave are the tape's

`capture` runs the forward on the default stream and waits for it, then captures it
over the pooled streams (`VitCaptureCalls.capture_calls`); read with the
capture stream alone joined, its calls record the forward's steps in the
schedule's order (`dag_sim`), so the graph it leaves holds their operations
(`ML.extRun_dagCapture`). It leaves what `replay` starts from
(`VitCaptureReach.capture_reaches`), and `replay` launches that graph once
and waits (`replay_calls`). So every run of `capture` then `replay` leaves
every word inside each buffer as the forward's tape, twice over, leaves it
(`capture_replay_agree`).

A plain file, not a module, as the Ship files it reads are.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
  AlgorithmLib.HProg.Contracts AlgorithmLib.Device AlgorithmLib.ML AlgorithmLib.Kernel

namespace VitReplayWords

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- Every run of `replay` that ends launches the forward's graph once on the
    capture stream, then waits for the stream, the device changed only by
    those calls. -/
theorem replay_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : VitState.Replay w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (Vit.vReplayAt Vit.VEXEC_FWD 1 : Prog Slot Lvl Unit)) = .ok obs w') :
    (Drv.App.replayCalls (drvHandle kExec 0) VitDag.S0 1).Covers w obs ∧ Check.DevPath w w' := by
  check_calls

/-- **A capture's calls record the forward's steps**, in the schedule's
    order, read with the capture stream alone joined. -/
theorem dag_sim : capSim Vit.vApp VitDag.vBs [VitDag.S0] [] (VitDag.vCapCalls 0 Vit.VFWD_N) =
    some ((VitWords.stepOps 0 Vit.VFWD_N).map Prod.fst) := by native_decide

theorem no_graphs : ((VitState.after 0 0).contains (.ngraphs 0) && (VitState.after 0 0).contains (.nexecs 0)) = true := by
  native_decide

/-- **The words a capture of the forward and a replay of it leave are the
    tape's, twice over**: every run of `capture` that ends, from what `main`
    leaves, then every run of `replay` that ends, on a device honest about
    each group's launch (`TapeLayout.KernelHonest`) where each runs and
    about its products, leaves every word inside each buffer, at the size
    the layout makes it, as the tape's first `VFWD_N` operations, twice over,
    leave it from the memory the capture started with, each contraction read
    as the vendor computes it. -/
theorem capture_replay_agree (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w0 : World) (h : VitState.Run w0)
    {obs1 : List Obs} {w1 : World}
    (hr1 : Sem.run cfg (runArgs dataLen outLen) w0
      (emit (Vit.vCaptureAt 0 Vit.VFWD_N Vit.VGRAPH_FWD Vit.VEXEC_FWD : Prog Slot Lvl Unit)) = .ok obs1 w1)
    (cfg' : Cfg) (dataLen' outLen' : UInt64) {obs2 : List Obs} {w2 : World}
    (hr2 : Sem.run cfg' (runArgs dataLen' outLen') w1
      (emit (Vit.vReplayAt Vit.VEXEC_FWD 1 : Prog Slot Lvl Unit)) = .ok obs2 w2)
    (hv : VendorGemmHonest w0.vendor) (hvk : VendorKeeps w0.vendor)
    (hHon : ∀ p ∈ VitWords.stepUnits 0 Vit.VFWD_N, ∀ j, p.1 = .kernel j →
      Vit.vLayout.KernelHonest Vit.vApp w0 j (Vit.vUnitOps p.2) hw)
    (hHon1 : ∀ p ∈ VitWords.stepUnits 0 Vit.VFWD_N, ∀ j, p.1 = .kernel j →
      Vit.vLayout.KernelHonest Vit.vApp w1 j (Vit.vUnitOps p.2) hw) :
    Vit.vLayout.Agree (devMem w2.dev)
      ((Vit.vFused.take Vit.VFWD_N).foldl (fun M op => TOp.denV hw op M)
        ((Vit.vFused.take Vit.VFWD_N).foldl (fun M op => TOp.denV hw op M) (wordView (devMem w0.dev)))) := by
  let ps := VitWords.stepOps 0 Vit.VFWD_N
  have hmem : ∀ p ∈ ps, ∃ q ∈ VitWords.stepUnits 0 Vit.VFWD_N, p = (q.1, Vit.vUnitOps q.2) := fun p hp => by
    obtain ⟨q, hq, e⟩ := List.mem_map.mp hp
    exact ⟨q, hq, e.symm⟩
  have hU : ∀ p ∈ ps, ∀ j, p.1 = .kernel j → Vit.vLayout.UnitOk p.2 := fun p hp j hj => by
    obtain ⟨q, hq, rfl⟩ := hmem p hp
    have := List.all_eq_true.mp VitWords.fwd_proven q hq
    dsimp only at hj
    simp only [VitWords.provenOk, hj] at this
    exact Vit.vit_unit_ok q.2 (List.contains_iff_mem.mp this)
  have hH : ∀ (w : World), (∀ p ∈ VitWords.stepUnits 0 Vit.VFWD_N, ∀ j, p.1 = .kernel j →
      Vit.vLayout.KernelHonest Vit.vApp w j (Vit.vUnitOps p.2) hw) →
      ∀ p ∈ ps, ∀ j, p.1 = .kernel j → Vit.vLayout.KernelHonest Vit.vApp w j p.2 hw :=
    fun w hh p hp j hj => by
      obtain ⟨q, hq, rfl⟩ := hmem p hp
      exact hh q hq j hj
  -- what the capture leaves, and the calls each run makes
  obtain ⟨_, hrep⟩ := ML.triple_run (VitCaptureReach.capture_reaches cfg dataLen outLen w0 h) hr1
  have hc1 := VitCaptureCalls.capture_calls cfg dataLen outLen w0 h hr1
  have hc2 := replay_calls cfg' dataLen' outLen' w1 hrep hr2
  have hng := no_graphs
  simp only [Bool.and_eq_true, List.contains_iff_mem] at hng
  have hg : w0.dev.graphs.size = 0 := h _ hng.1
  have he : w0.dev.execs.size = 0 := h _ hng.2
  have hc1' : (Check.Tr.lit ((ps.map Prod.fst).map
      (DevCall.record Vit.vApp (drvHandle kBlas 0) 0x3f800000 0) ++
      [wordRec .streamSynchronize [0]] ++ dagRecs Vit.vApp [0x3f800000, 0] (VitDag.vCapCalls 0 Vit.VFWD_N) VitDag.S0
        (arenaAt Vit.VGRAPH_FWD) (arenaAt Vit.VEXEC_FWD))).Covers w0 obs1 ∧ Check.DevPath w0 w1 := by
    rw [show ps.map Prod.fst = _ from VitWords.stepOps_calls Vit.VFWD_N]
    exact hc1
  have e := ML.dag_replay_tape Vit.vLayout Vit.vApp rfl Vit.vBufBytes_le hw hw32 ps ps VitWords.fwd_ok hU
    VitWords.fwd_ok hU (VitDag.vCapCalls 0 Vit.VFWD_N) VitDag.vBs dag_sim 1 (VitWords.ready h)
    (VitDag.vBs_ok h) hg he hr1 hc1' hr2 hc2 hv hvk (hH w0 hHon) (hH w1 hHon1)
  -- the units' operations are the tape's
  letI := F32.ops hw
  have hfl : (Vit.vRangeUnits 0 Vit.VFWD_N).flatMap Vit.vUnitOps = ps.flatMap Prod.snd := by
    rw [← VitWords.fwd_units, List.flatMap_map]
    simp only [ps, VitWords.stepOps, List.flatMap_map]
  have htape : ∀ M, (ps.flatMap Prod.snd).foldl (fun M op => TOp.denV hw op M) M =
      (Vit.vFused.take Vit.VFWD_N).foldl (fun M op => TOp.denV hw op M) M := fun M => by
    rw [← hfl]
    exact Vit.vit_range_computes_tape_with (denV_frame hw Vit.SQ) Vit.VFWD_N Vit.vit_fwd_pull M
  simp only [Nat.repeat] at e
  rw [htape, htape] at e
  exact e

end VitReplayWords

#print axioms VitReplayWords.capture_replay_agree
