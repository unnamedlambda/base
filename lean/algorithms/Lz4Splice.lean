import Lz4Cursor

set_option maxRecDepth 8192

/-!
  # The prologue-to-loop splice, and `CursorAtSites.opLe`

  A separate module because a single olean past roughly 8 MB overflows the
  serializer stack (`exit 134`, no line number).
-/

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt

theorem prologue_at_shipped (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hw : w < (WP.mk 15).numBlk) (hw64 : w * 32 + 32 < 2 ^ 64)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack)) :
    (siter K preSteps (initSt w inPtr outPtr gm smemB)).pc = 39
    ∧ AlgorithmLib.LZ4WarpDSL.MachInv (inPtr + w * (WP.mk 15).inStride)
        (siter K preSteps (initSt w inPtr outPtr gm smemB))
    ∧ AlgorithmLib.LZ4WarpDSL.Couple AlgorithmLib.LZ4WarpDSL.loopR
        (siter K preSteps (initSt w inPtr outPtr gm smemB))
        (AlgorithmLib.LZ4WarpDSL.WState.mk
          (fun r => (siter K preSteps (initSt w inPtr outPtr gm smemB)).regs r 0)
          (siter K preSteps (initSt w inPtr outPtr gm smemB)).gmem
          (siter K preSteps (initSt w inPtr outPtr gm smemB)).smem)
    ∧ (siter K preSteps (initSt w inPtr outPtr gm smemB)).regs "op" 0 = 0
    ∧ (siter K preSteps (initSt w inPtr outPtr gm smemB)).regs "litAnchor" 0 = 0
    ∧ (siter K preSteps (initSt w inPtr outPtr gm smemB)).regs "searchPos" 0 = 0
    ∧ (siter K preSteps (initSt w inPtr outPtr gm smemB)).regs "inBase" 0
        = UInt64.ofNat (inPtr + w * (WP.mk 15).inStride)
    ∧ (siter K preSteps (initSt w inPtr outPtr gm smemB)).regs "outBase" 0
        = UInt64.ofNat (outPtr + w * (WP.mk 15).outStride)
    ∧ (siter K preSteps (initSt w inPtr outPtr gm smemB)).gmem = gm := by
  have hker : K = AlgorithmLib.LZ4WarpDSL.warpKernelDSL (WP.mk 15).numBlk (WP.mk 15).inStride
      (WP.mk 15).outStride (WP.mk 15).lenOff wHashLog := rfl
  have hall := AlgorithmLib.LZ4WarpDSL.prologue_couple (WP.mk 15).numBlk (WP.mk 15).inStride
    (WP.mk 15).outStride (WP.mk 15).lenOff wHashLog w inPtr outPtr gm smemB nbpos nb64 hw hw64
    hderive (by decide)
  rw [← hker, ← siter_eq_snsteps,
    preSteps_eq.symm] at hall
  exact hall

/-- The head lemma's result packaged through a lemma whose states are VARIABLES.
    Writing the anonymous constructor directly at `siter K (preSteps + 1) (initSt …)`
    makes the term the olean writer chokes on; with variables it is a plain
    application.  Same failure family as `lsicInv_mk`. -/
theorem head_mk (S S' : SState) (ws1 : AlgorithmLib.LZ4WarpDSL.WState) (ib iS LO : Nat)
    (h1 : S.pc = 39) (h2 : S'.pc = 40)
    (h3 : AlgorithmLib.LZ4WarpDSL.Couple AlgorithmLib.LZ4WarpDSL.loopR S' ws1)
    (h4 : AlgorithmLib.LZ4WarpDSL.MachInv ib S')
    (h5 : AlgorithmLib.LZ4WarpDSL.LoopCQ iS ws1)
    (h6 : AlgorithmLib.LZ4WarpDSL.TightQ iS LO ws1)
    (h7 : (ws1.regs "searchPos").toNat = 0)
    (h8 : (ws1.regs "litAnchor").toNat = 0) :
    S.pc = 39 ∧ ∃ w : AlgorithmLib.LZ4WarpDSL.WState,
      S'.pc = 40 ∧ AlgorithmLib.LZ4WarpDSL.Couple AlgorithmLib.LZ4WarpDSL.loopR S' w
      ∧ AlgorithmLib.LZ4WarpDSL.MachInv ib S'
      ∧ AlgorithmLib.LZ4WarpDSL.LoopCQ iS w
      ∧ AlgorithmLib.LZ4WarpDSL.TightQ iS LO w
      ∧ (w.regs "searchPos").toNat = 0 ∧ (w.regs "litAnchor").toNat = 0 :=
  ⟨h1, ws1, h2, h3, h4, h5, h6, h7, h8⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
/-- **Prologue, then the `setp loopC`.** -/
theorem shipped_loop_head (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hw : w < (WP.mk 15).numBlk) (hw64 : w * 32 + 32 < 2 ^ 64)
    (hib40 : inPtr + w * (WP.mk 15).inStride < 2 ^ 40)
    (htop : outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride < 2 ^ 32)
    (hbuf : outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride ≤ gm.size)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hdisj : inPtr + w * (WP.mk 15).inStride + (WP.mk 15).inStride
      ≤ outPtr + w * (WP.mk 15).outStride) :
    (siter K preSteps (initSt w inPtr outPtr gm smemB)).pc = 39
    ∧ ∃ ws1 : AlgorithmLib.LZ4WarpDSL.WState,
        (siter K (preSteps + 1) (initSt w inPtr outPtr gm smemB)).pc = 40
        ∧ AlgorithmLib.LZ4WarpDSL.Couple AlgorithmLib.LZ4WarpDSL.loopR
            (siter K (preSteps + 1) (initSt w inPtr outPtr gm smemB)) ws1
        ∧ AlgorithmLib.LZ4WarpDSL.MachInv (inPtr + w * (WP.mk 15).inStride)
            (siter K (preSteps + 1) (initSt w inPtr outPtr gm smemB))
        ∧ AlgorithmLib.LZ4WarpDSL.LoopCQ (WP.mk 15).inStride ws1
        ∧ AlgorithmLib.LZ4WarpDSL.TightQ (WP.mk 15).inStride (WP.mk 15).lenOff ws1
        ∧ (ws1.regs "searchPos").toNat = 0
        ∧ (ws1.regs "litAnchor").toNat = 0 := by
  obtain ⟨S0, hS0⟩ : ∃ x, siter K preSteps (initSt w inPtr outPtr gm smemB) = x := ⟨_, rfl⟩
  have hall := prologue_at_shipped w inPtr outPtr gm smemB hw hw64 hderive
  rw [hS0] at hall
  obtain ⟨hpc39, hMI, hCouple, hop0, hla0, hsp0, hinB0, houtB0, hgmem⟩ := hall
  obtain ⟨ws0, hws0⟩ : ∃ x, AlgorithmLib.LZ4WarpDSL.WState.mk (fun r => S0.regs r 0) S0.gmem
      S0.smem = x := ⟨_, rfl⟩
  rw [hws0] at hCouple
  have hoT : (ws0.regs "outBase").toNat = outPtr + w * (WP.mk 15).outStride := by
    rw [← hws0]; show (S0.regs "outBase" 0).toNat = _
    rw [houtB0, UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega)]
  have hiB : (ws0.regs "inBase").toNat = inPtr + w * (WP.mk 15).inStride := by
    rw [← hws0]; show (S0.regs "inBase" 0).toNat = _
    rw [hinB0, UInt64.toNat_ofNat_of_lt' (by have hs : UInt64.size = 2 ^ 64 := rfl; omega)]
  have hgS : ws0.gmem.size = gm.size := by rw [← hws0]; show S0.gmem.size = _; rw [hgmem]
  -- Step 1: the `setp loopC`.
  have hi39 : K[S0.pc]?
      = some (.setp (.lt) "loopC" "searchPos" (SArg.imm ((WP.mk 15).inStride - 12))) := by
    rw [hpc39]; decide
  obtain ⟨hc1, hpc1s⟩ :=
    AlgorithmLib.LZ4WarpDSL.setp_sound AlgorithmLib.LZ4WarpDSL.loopR K SCmp.lt "loopC" "searchPos"
      (AlgorithmLib.LZ4WarpDSL.WArg.imm ((WP.mk 15).inStride - 12)) S0 ws0 hi39 hCouple
      (by decide) (fun n h => by cases h)
  have hmi1 : AlgorithmLib.LZ4WarpDSL.MachInv (inPtr + w * (WP.mk 15).inStride) (sstep K S0) :=
    AlgorithmLib.LZ4WarpDSL.machInv_sstep _ K S0 hMI
      (fun i hi => by rw [hi39] at hi; cases hi; exact ⟨by decide, by decide, by decide⟩)
  have hstep1 : siter K (preSteps + 1) (initSt w inPtr outPtr gm smemB) = sstep K S0 := by
    rw [siter_succ, hS0]
  have hpc40 : (siter K (preSteps + 1) (initSt w inPtr outPtr gm smemB)).pc = 40 := by
    rw [hstep1, hpc1s, hpc39]
  -- the eval state at the loop head
  obtain ⟨F, hF⟩ : ∃ x, (WP.mk 15).inStride + 34 * (WP.mk 15).inStride = x := ⟨_, rfl⟩
  obtain ⟨ws1, hws1⟩ : ∃ x, (AlgorithmLib.LZ4WarpDSL.WStmt.setp SCmp.lt "loopC" "searchPos"
      (AlgorithmLib.LZ4WarpDSL.WArg.imm ((WP.mk 15).inStride - 12))).eval F ws0 = x := ⟨_, rfl⟩
  have hc1' : AlgorithmLib.LZ4WarpDSL.Couple AlgorithmLib.LZ4WarpDSL.loopR (sstep K S0) ws1 := by
    rw [← hws1]
    have he : (AlgorithmLib.LZ4WarpDSL.WStmt.setp SCmp.lt "loopC" "searchPos"
        (AlgorithmLib.LZ4WarpDSL.WArg.imm ((WP.mk 15).inStride - 12))).eval F ws0
        = ws0.setReg "loopC" (if SCmp.run SCmp.lt (ws0.regs "searchPos")
            ((AlgorithmLib.LZ4WarpDSL.WArg.imm ((WP.mk 15).inStride - 12)).eval ws0)
          then 1 else 0) := by simp only [AlgorithmLib.LZ4WarpDSL.WStmt.eval]
    rw [he]
    exact hc1
  have hQ1 : AlgorithmLib.LZ4WarpDSL.LoopCQ (WP.mk 15).inStride ws1 := by
    rw [← hws1]
    exact AlgorithmLib.LZ4WarpDSL.setpLoopC_LoopCQ (WP.mk 15).inStride (by decide) (by decide) ws0
      (by rw [← hws0]; show S0.regs "op" 0 = 0; exact hop0)
      (by rw [← hws0]; show S0.regs "litAnchor" 0 = 0; exact hla0)
      (by rw [← hws0]; show S0.regs "searchPos" 0 = 0; exact hsp0) (by rw [hiB]; exact hib40)
      (by rw [hoT]; exact htop) (by rw [hoT, hgS]; exact hbuf)
      (by rw [hiB, hoT]; exact hdisj) F
  have hT1 : AlgorithmLib.LZ4WarpDSL.TightQ (WP.mk 15).inStride (WP.mk 15).lenOff ws1 :=
    AlgorithmLib.LZ4WarpDSL.tightQ_init _ _ ws1
      (by rw [← hws1]; simp only [AlgorithmLib.LZ4WarpDSL.WStmt.eval,
        AlgorithmLib.LZ4WarpDSL.WState.setReg, String.reduceEq, if_false]
          rw [← hws0]; show S0.regs "op" 0 = 0; exact hop0)
      (by rw [← hws1]; simp only [AlgorithmLib.LZ4WarpDSL.WStmt.eval,
        AlgorithmLib.LZ4WarpDSL.WState.setReg, String.reduceEq, if_false]
          rw [← hws0]; show S0.regs "litAnchor" 0 = 0; exact hla0)
      (by decide)
  have hsp1 : (ws1.regs "searchPos").toNat = 0 := by
    rw [← hws1]
    simp only [AlgorithmLib.LZ4WarpDSL.WStmt.eval, AlgorithmLib.LZ4WarpDSL.WState.setReg,
      String.reduceEq, if_false]
    rw [← hws0]; show (S0.regs "searchPos" 0).toNat = 0
    rw [hsp0]; rfl
  have hcS : AlgorithmLib.LZ4WarpDSL.Couple AlgorithmLib.LZ4WarpDSL.loopR
      (siter K (preSteps + 1) (initSt w inPtr outPtr gm smemB)) ws1 := by
    rw [hstep1]; exact hc1'
  have hmS : AlgorithmLib.LZ4WarpDSL.MachInv (inPtr + w * (WP.mk 15).inStride)
      (siter K (preSteps + 1) (initSt w inPtr outPtr gm smemB)) := by
    rw [hstep1]; exact hmi1
  have hlaZ : (ws1.regs "litAnchor").toNat = 0 := by
    rw [← hws1]
    simp only [AlgorithmLib.LZ4WarpDSL.WStmt.eval, AlgorithmLib.LZ4WarpDSL.WState.setReg,
      String.reduceEq, if_false]
    rw [← hws0]; show (S0.regs "litAnchor" 0).toNat = 0
    rw [hla0]; rfl
  rw [← hS0] at hpc39
  exact head_mk _ _ ws1 _ _ _ hpc39 hpc40 hcS hmS hQ1 hT1 hsp1 hlaZ
end Lz4Sites
