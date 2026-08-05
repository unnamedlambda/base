import Lz4Splice

set_option maxRecDepth 8192

/-!
  # The loop checkpoint on the real trace

  One theorem per module past this point: the olean serializer's stack is the
  binding constraint, not the proofs.
-/

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt

set_option maxHeartbeats 1000000 in
/-- **The checkpoint, spliced onto the real trace.**  From the launch state, after
    the prologue and the `setp loopC`, the machine stands at the loop head, and at
    EVERY later step at which it stands at the match-sequence entry there is a
    coupled eval state carrying the tight token bound.

    The `∀ j` is unconditional: within the loop's step count it is the descent's
    checkpoint, and past the loop exit (pc 209) `stays_from_208` says the machine
    can never be at 124 again. -/
theorem shipped_loop_ckpt (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hw : w < (WP.mk 15).numBlk) (hw64 : w * 32 + 32 < 2 ^ 64)
    (hib40 : inPtr + w * (WP.mk 15).inStride < 2 ^ 40)
    (htop : outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride < 2 ^ 32)
    (hbuf : outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride ≤ gm.size)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hdisj : inPtr + w * (WP.mk 15).inStride + (WP.mk 15).inStride
      ≤ outPtr + w * (WP.mk 15).outStride) :
    ∃ (n : Nat),
      (siter K preSteps (initSt w inPtr outPtr gm smemB)).pc = 39 ∧
      (siter K (preSteps + 1) (initSt w inPtr outPtr gm smemB)).pc = 40 ∧
      (∀ j, (siter K (preSteps + 1 + j) (initSt w inPtr outPtr gm smemB)).pc = 124 →
        AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 15).inStride (WP.mk 15).lenOff
          (siter K (preSteps + 1 + j) (initSt w inPtr outPtr gm smemB))) ∧
      (siter K (preSteps + 1 + n) (initSt w inPtr outPtr gm smemB)).pc = 209 ∧
      ∃ wE, AlgorithmLib.LZ4WarpDSL.Couple AlgorithmLib.LZ4WarpDSL.loopR
          (siter K (preSteps + 1 + n) (initSt w inPtr outPtr gm smemB)) wE
        ∧ AlgorithmLib.LZ4WarpDSL.LoopCQ (WP.mk 15).inStride wE
        ∧ AlgorithmLib.LZ4WarpDSL.TightQ (WP.mk 15).inStride (WP.mk 15).lenOff wE
        ∧ (wE.regs "litAnchor").toNat ≤ (WP.mk 15).inStride - 5 := by
  obtain ⟨hpc39, ws1, hpc40, hc1', hmi1, hQ1, hT1, hsp1, hla1⟩ :=
    shipped_loop_head w inPtr outPtr gm smemB hw hw64 hib40 htop hbuf hderive hdisj
  have hker : K = AlgorithmLib.LZ4WarpDSL.warpKernelDSL (WP.mk 15).numBlk (WP.mk 15).inStride
      (WP.mk 15).outStride (WP.mk 15).lenOff wHashLog := rfl
  obtain ⟨F, hF⟩ : ∃ x, (WP.mk 15).inStride + 34 * (WP.mk 15).inStride = x := ⟨_, rfl⟩
  have hseg39 : AlgorithmLib.LZ4WarpDSL.SegAt K 39
      (AlgorithmLib.LZ4WarpDSL.bodyPrefixSeg (WP.mk 15).inStride wHashLog) := by
    rw [hker]; exact AlgorithmLib.LZ4WarpDSL.body_segAt _ _ _ _ _
  have hlr39 : AlgorithmLib.LZ4WarpDSL.LabelsResolve K 39
      (AlgorithmLib.LZ4WarpDSL.bodyPrefixSeg (WP.mk 15).inStride wHashLog) := by
    rw [hker]; exact AlgorithmLib.LZ4WarpDSL.body_labelsResolve _ _ _ _ _
  simp only [AlgorithmLib.LZ4WarpDSL.bodyPrefixSeg] at hseg39 hlr39
  have hseg40 := hseg39.append_right.append_left
  have hlr40 := hlr39.append_right.append_left
  obtain ⟨n, ssE, hrE, hpcE, hcE, hmiE, hAll⟩ :=
    AlgorithmLib.LZ4WarpDSL.loopC_loop_sim_ckpt (inPtr + w * (WP.mk 15).inStride)
      (WP.mk 15).inStride wHashLog F
      "Lh0" "Lx1" "Le2" "Ln3" "Lh4" "Lx5" "Le6" "Ln7" "Lh8" "Lx9" "Ch10" "Cx11" "Le12" "Ln13"
      "Lh14" "Lx15" (AlgorithmLib.LZ4WarpDSL.myLsic "litExtra")
      (AlgorithmLib.LZ4WarpDSL.myLsic "matExtra") rfl rfl (by decide) (by decide) (by decide)
      hib40 (by decide) (by rw [← hF]; decide) K 40 hseg40 hlr40
      (WP.mk 15).lenOff 124 (by decide) mb_top
      (fun q' h1 h2 => mb_noentry _ (by decide) q' h1 h2)
      F (siter K (preSteps + 1) (initSt w inPtr outPtr gm smemB)) ws1 hpc40 hc1' hmi1
      ⟨hQ1, hT1⟩ (by rw [hsp1]; omega)
      (AlgorithmLib.LZ4WarpDSL.loopC_halts (WP.mk 15).inStride wHashLog (by decide) (by decide)
        (by decide) F ws1 hQ1 (by rw [hsp1, ← hF]; decide))
  have hEq : ∀ j, siter K (preSteps + 1 + j) (initSt w inPtr outPtr gm smemB)
      = siter K j (siter K (preSteps + 1) (initSt w inPtr outPtr gm smemB)) := by
    intro j; rw [siter_add]
  have hssE : siter K (preSteps + 1 + n) (initSt w inPtr outPtr gm smemB) = ssE := by
    rw [hEq n]; exact sreaches_siter K n _ _ hrE
  have hpc209 : (siter K (preSteps + 1 + n) (initSt w inPtr outPtr gm smemB)).pc = 209 := by
    rw [hssE, hpcE]; decide
  refine ⟨n, hpc39, hpc40, fun j hj => ?_, hpc209,
    (AlgorithmLib.LZ4WarpDSL.WStmt.uwhile "loopC"
      (AlgorithmLib.LZ4WarpDSL.loopCBodyStmt (WP.mk 15).inStride wHashLog)).eval F ws1,
    ?_, ?_, ?_, ?_⟩
  · rcases Nat.lt_or_ge n j with hgt | hle
    · exfalso
      have h209 : 208 ≤ (siter K (preSteps + 1 + n) (initSt w inPtr outPtr gm smemB)).pc := by
        rw [hpc209]; omega
      have := stays_from_208 (initSt w inPtr outPtr gm smemB) (preSteps + 1 + n) h209
        (preSteps + 1 + j) (by omega)
      rw [hj] at this; omega
    · have hQj := hAll j hle
      rw [← hEq j] at hQj
      exact hQj.2 hj
  · rw [hssE]; exact hcE
  · exact AlgorithmLib.LZ4WarpDSL.loopC_loop_preservesInv (WP.mk 15).inStride wHashLog
      (by decide) (by decide) (by decide) F ws1 hQ1 (by rw [hsp1, ← hF]; decide)
  · exact AlgorithmLib.LZ4WarpDSL.loopC_loop_preservesTight (WP.mk 15).inStride wHashLog
      (WP.mk 15).lenOff (by decide) (by decide) (by decide) F ws1 hQ1 hT1
      (by rw [hsp1, ← hF]; decide)
  · exact AlgorithmLib.LZ4WarpDSL.loopC_litAnchor_bound (WP.mk 15).inStride wHashLog
      (by decide) (by decide) (by decide) F ws1 hQ1 (by rw [hla1]; omega)
      (by rw [hsp1, ← hF]; decide)

-- ── The tail: from the loop exit to the LSIC loop head ───────────────────────

end Lz4Sites

