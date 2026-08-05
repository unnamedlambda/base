import Lz4Splice64

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

theorem shipped_loop_ckpt64 (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hw : w < (WP.mk 16).numBlk) (hw64 : w * 32 + 32 < 2 ^ 64)
    (hib40 : inPtr + w * (WP.mk 16).inStride < 2 ^ 40)
    (htop : outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride < 2 ^ 32)
    (hbuf : outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride ≤ gm.size)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hdisj : inPtr + w * (WP.mk 16).inStride + (WP.mk 16).inStride
      ≤ outPtr + w * (WP.mk 16).outStride) :
    ∃ (n : Nat),
      (siter K16 preSteps (initSt w inPtr outPtr gm smemB)).pc = 39 ∧
      (siter K16 (preSteps + 1) (initSt w inPtr outPtr gm smemB)).pc = 40 ∧
      (∀ j, (siter K16 (preSteps + 1 + j) (initSt w inPtr outPtr gm smemB)).pc = 124 →
        AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 16).inStride (WP.mk 16).lenOff
          (siter K16 (preSteps + 1 + j) (initSt w inPtr outPtr gm smemB))) ∧
      (siter K16 (preSteps + 1 + n) (initSt w inPtr outPtr gm smemB)).pc = 209 ∧
      ∃ wE, AlgorithmLib.LZ4WarpDSL.Couple AlgorithmLib.LZ4WarpDSL.loopR
          (siter K16 (preSteps + 1 + n) (initSt w inPtr outPtr gm smemB)) wE
        ∧ AlgorithmLib.LZ4WarpDSL.LoopCQ (WP.mk 16).inStride wE
        ∧ AlgorithmLib.LZ4WarpDSL.TightQ (WP.mk 16).inStride (WP.mk 16).lenOff wE
        ∧ (wE.regs "litAnchor").toNat ≤ (WP.mk 16).inStride - 5 := by
  obtain ⟨hpc39, ws1, hpc40, hc1', hmi1, hQ1, hT1, hsp1, hla1⟩ :=
    shipped_loop_head64 w inPtr outPtr gm smemB hw hw64 hib40 htop hbuf hderive hdisj
  have hker : K16 = AlgorithmLib.LZ4WarpDSL.warpKernelDSL (WP.mk 16).numBlk (WP.mk 16).inStride
      (WP.mk 16).outStride (WP.mk 16).lenOff wHashLog := rfl
  obtain ⟨F, hF⟩ : ∃ x, (WP.mk 16).inStride + 34 * (WP.mk 16).inStride = x := ⟨_, rfl⟩
  have hseg39 : AlgorithmLib.LZ4WarpDSL.SegAt K16 39
      (AlgorithmLib.LZ4WarpDSL.bodyPrefixSeg (WP.mk 16).inStride wHashLog) := by
    rw [hker]; exact AlgorithmLib.LZ4WarpDSL.body_segAt _ _ _ _ _
  have hlr39 : AlgorithmLib.LZ4WarpDSL.LabelsResolve K16 39
      (AlgorithmLib.LZ4WarpDSL.bodyPrefixSeg (WP.mk 16).inStride wHashLog) := by
    rw [hker]; exact AlgorithmLib.LZ4WarpDSL.body_labelsResolve _ _ _ _ _
  simp only [AlgorithmLib.LZ4WarpDSL.bodyPrefixSeg] at hseg39 hlr39
  have hseg40 := hseg39.append_right.append_left
  have hlr40 := hlr39.append_right.append_left
  obtain ⟨n, ssE, hrE, hpcE, hcE, hmiE, hAll⟩ :=
    AlgorithmLib.LZ4WarpDSL.loopC_loop_sim_ckpt (inPtr + w * (WP.mk 16).inStride)
      (WP.mk 16).inStride wHashLog F
      "Lh0" "Lx1" "Le2" "Ln3" "Lh4" "Lx5" "Le6" "Ln7" "Lh8" "Lx9" "Ch10" "Cx11" "Le12" "Ln13"
      "Lh14" "Lx15" (AlgorithmLib.LZ4WarpDSL.myLsic "litExtra")
      (AlgorithmLib.LZ4WarpDSL.myLsic "matExtra") rfl rfl (by decide) (by decide) (by decide)
      hib40 (by decide) (by rw [← hF]; decide) K16 40 hseg40 hlr40
      (WP.mk 16).lenOff 124 (by decide) mb_top64
      (fun q' h1 h2 => mb_noentry64 _ (by decide) q' h1 h2)
      F (siter K16 (preSteps + 1) (initSt w inPtr outPtr gm smemB)) ws1 hpc40 hc1' hmi1
      ⟨hQ1, hT1⟩ (by rw [hsp1]; omega)
      (AlgorithmLib.LZ4WarpDSL.loopC_halts (WP.mk 16).inStride wHashLog (by decide) (by decide)
        (by decide) F ws1 hQ1 (by rw [hsp1, ← hF]; decide))
  have hEq : ∀ j, siter K16 (preSteps + 1 + j) (initSt w inPtr outPtr gm smemB)
      = siter K16 j (siter K16 (preSteps + 1) (initSt w inPtr outPtr gm smemB)) := by
    intro j; rw [siter_add]
  have hssE : siter K16 (preSteps + 1 + n) (initSt w inPtr outPtr gm smemB) = ssE := by
    rw [hEq n]; exact sreaches_siter K16 n _ _ hrE
  have hpc209 : (siter K16 (preSteps + 1 + n) (initSt w inPtr outPtr gm smemB)).pc = 209 := by
    rw [hssE, hpcE]; decide
  refine ⟨n, hpc39, hpc40, fun j hj => ?_, hpc209,
    (AlgorithmLib.LZ4WarpDSL.WStmt.uwhile "loopC"
      (AlgorithmLib.LZ4WarpDSL.loopCBodyStmt (WP.mk 16).inStride wHashLog)).eval F ws1,
    ?_, ?_, ?_, ?_⟩
  · rcases Nat.lt_or_ge n j with hgt | hle
    · exfalso
      have h209 : 208 ≤ (siter K16 (preSteps + 1 + n) (initSt w inPtr outPtr gm smemB)).pc := by
        rw [hpc209]; omega
      have := stays_from_20864 (initSt w inPtr outPtr gm smemB) (preSteps + 1 + n) h209
        (preSteps + 1 + j) (by omega)
      rw [hj] at this; omega
    · have hQj := hAll j hle
      rw [← hEq j] at hQj
      exact hQj.2 hj
  · rw [hssE]; exact hcE
  · exact AlgorithmLib.LZ4WarpDSL.loopC_loop_preservesInv (WP.mk 16).inStride wHashLog
      (by decide) (by decide) (by decide) F ws1 hQ1 (by rw [hsp1, ← hF]; decide)
  · exact AlgorithmLib.LZ4WarpDSL.loopC_loop_preservesTight (WP.mk 16).inStride wHashLog
      (WP.mk 16).lenOff (by decide) (by decide) (by decide) F ws1 hQ1 hT1
      (by rw [hsp1, ← hF]; decide)
  · exact AlgorithmLib.LZ4WarpDSL.loopC_litAnchor_bound (WP.mk 16).inStride wHashLog
      (by decide) (by decide) (by decide) F ws1 hQ1 (by rw [hla1]; omega)
      (by rw [hsp1, ← hF]; decide)

-- ── The tail: from the loop exit to the LSIC loop head ───────────────────────




end Lz4Sites
