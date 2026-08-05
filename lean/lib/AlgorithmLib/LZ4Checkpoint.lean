import AlgorithmLib.EvalValid
import AlgorithmLib.LZ4StepDescent
import AlgorithmLib.LZ4Tight

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4Plan AlgorithmLib.LZ4WarpFind

/-- **Transcribed from `loopCBody_none_reaches`, carrying pc-confinement.** -/
theorem loopCBody_none_reaches_pcIn (ib : Nat) (inStride hashLog : Nat)
    (lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM : String)
    (lsicL lsicM : List SInstr)
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride) (hlen : inStride < 2 ^ 40)
    (ws : WState) (hQ : LoopCQ inStride ws) (hguard : (ws.regs "loopC" == 1) = true)
    (hhl : hashLog ≤ 32)
    (hw : window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat)
        (inStride - 12) (ws.regs "searchPos").toNat = none)
    (prog : Array SInstr) (base : Nat) (ss : SState) (fuel : Nat)
    (hpc : ss.pc = base)
    (hseg : SegAt prog base (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL
      cpH cpX lElseM lEndM lHM lXM lsicL lsicM))
    (hlr : LabelsResolve prog base (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL
      lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM))
    (hc : Couple loopR ss ws) (hmi : MachInv ib ss) (hib40 : ib < 2 ^ 40)
    (LO : Nat) (MB : Nat)
    (hMBdef : MB = base + (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride
      (inStride - 12) hashLog).length + 1 + 4
      + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2) :
    ∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
      ss'.pc = base + (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL
        cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length ∧
      Couple loopR ss' ((loopCBodyStmt inStride hashLog).eval fuel ws) ∧ MachInv ib ss' ∧
      AllSteps prog (PcIn base (base + (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL
        lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length)) m ss ∧
      AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) m ss := by
  obtain ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ := hQ
  have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
    loopCQ_guard inStride ws hstride hipos ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ hguard
  rw [loopCBodyEmit] at hseg hlr
  -- Step 1: coopWindow (window miss ⇒ found = 0).
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1, hst1⟩ :=
    simSLQ_coopWindow_pcIn loopR inStride (inStride - 12) hashLog (ws.regs "searchPos").toNat
      (by decide) (by decide) (by decide) hib40 (by omega) (by omega) hhl hlen (by omega) (by decide) (by decide)
      (by decide) ws rfl (fun l ss'' hc'' => by rw [hc''.reg (by decide) l, UInt64.ofNat_toNat])
      prog base ss fuel hpc hseg.append_left hlr.append_left hc hmi
  have hfound0 : ((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
      hashLog 0).eval fuel ws).regs "found" = 0 := by
    have : (WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
        hashLog 0).eval fuel ws = evalCoopWindow "found" "p0" "cand0" "searchPos" inStride
          (inStride - 12) hashLog 0 ws := by simp only [WStmt.eval]
    rw [this, coopWindow_found_val, hw]
  -- Machine `found` at ss1 is `0` (coupled, uniform).
  have hfound0m : ss1.regs "found" 0 = 0 := by rw [hc1.reg (by decide) 0, hfound0]
  -- Abbreviate the found-branch emit and the post-coopWindow base.
  obtain ⟨et, hetdef⟩ : ∃ x, foundBranchEmit (inStride - 5) lHE lEE lElseL lEndL lHL lXL cpH cpX
      lElseM lEndM lHM lXM lsicL lsicM = x := ⟨_, rfl⟩
  obtain ⟨base2, hbase2⟩ : ∃ x, base + (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride
      (inStride - 12) hashLog).length = x := ⟨_, rfl⟩
  rw [hetdef] at hseg hlr
  have hpc1' : ss1.pc = base2 := by rw [hpc1, hbase2]
  have hsegR := hseg.append_right
  have hlrR := hlr.append_right
  rw [hbase2] at hsegR hlrR
  have hsegU : SegAt prog base2 (uifEmit "found" lElse lEnd et
      [.bin .add "searchPos" "searchPos" (.imm 32)]) := hsegR.append_left
  have hlrU : LabelsResolve prog base2 (uifEmit "found" lElse lEnd et
      [.bin .add "searchPos" "searchPos" (.imm 32)]) := hlrR.append_left
  obtain ⟨hbr, hseg1⟩ := hsegU.cons
  obtain ⟨_hbra, hseg3⟩ := hseg1.append_right.cons
  obtain ⟨hlblE, hseg4⟩ := hseg3.cons
  have hsegE : SegAt prog (base2 + 1 + et.length + 1 + 1)
      [.bin .add "searchPos" "searchPos" (.imm 32)] := hseg4.append_left
  obtain ⟨hlblN, _⟩ := hseg4.append_right.cons
  have hlrE : LabelsResolve prog (base2 + 1 + et.length + 1 + 1)
      [.bin .add "searchPos" "searchPos" (.imm 32)] :=
    hlrU.cons.append_right.cons.cons.append_left
  have hLelse : sfindLabel prog lElse = base2 + 1 + et.length + 1 :=
    hlrU.cons.append_right.cons 0 lElse (by simp)
  have hcv := hc1.reg (show "found" ∈ loopR by decide) 0
  have hbr' : prog[ss1.pc]? = some (.braifnot "found" lElse) := by rw [hpc1']; exact hbr
  have hb : ¬(((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
      hashLog 0).eval fuel ws).regs "found" == 1) = true := by rw [hfound0]; decide
  -- Step 2: the `uif` takes the else branch (`found = 0`): braifnot → lElse ; lbl.
  have s0 : sstep prog ss1 = ss1.setPc (base2 + 1 + et.length + 1) := by
    rw [braifnot_step prog ss1 "found" lElse hbr', hcv, if_neg hb, hLelse]
  have hlblE' : prog[(ss1.setPc (base2 + 1 + et.length + 1)).pc]? = some (.lbl lElse) := hlblE
  have s1 : sstep prog (ss1.setPc (base2 + 1 + et.length + 1))
      = ss1.setPc (base2 + 1 + et.length + 1 + 1) := by
    rw [lbl_step prog _ lElse hlblE']; simp [SState.setPc]
  -- Step 3: the not-found branch (`searchPos += 32`).
  obtain ⟨ne, ss2, hrE, hpcE, hcE, hmiE, hstE⟩ :=
    simSLQ_notFoundBranch_pcIn ib loopR (by decide)
      prog (base2 + 1 + et.length + 1 + 1) (ss1.setPc (base2 + 1 + et.length + 1 + 1))
      ((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0).eval
        fuel ws)
      fuel rfl hsegE hlrE (couple_setPc hc1 _) (machInv_setPc ss1 _ hmi1)
  have hpcE' : ss2.pc = base2 + 1 + et.length + 1 + 1 + 1 := by
    rw [hpcE]; simp only [List.length_cons, List.length_nil]
  have hlblN' : prog[ss2.pc]? = some (.lbl lEnd) := by rw [hpcE]; exact hlblN
  have huplen : (uifEmit "found" lElse lEnd et
      [.bin .add "searchPos" "searchPos" (.imm 32)]).length = et.length + 1 + 4 := by
    rw [uifEmit_length]; simp only [List.length_cons, List.length_nil]
  have s2 : sstep prog ss2 = ss2.setPc (base2 + (uifEmit "found" lElse lEnd et
      [.bin .add "searchPos" "searchPos" (.imm 32)]).length) := by
    rw [lbl_step prog ss2 lEnd hlblN']; congr 1; rw [hpcE', huplen]; omega
  -- Step 4: the trailing `setp loopC`.
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5, hst5⟩ :=
    simSLQ_setp_pcIn ib loopR .lt "loopC" "searchPos" (.imm (inStride - 12)) (by decide)
      (fun n h => by cases h) (by decide)
      prog (base2 + (uifEmit "found" lElse lEnd et [.bin .add "searchPos" "searchPos" (.imm 32)]).length)
      (ss2.setPc (base2 + (uifEmit "found" lElse lEnd et
        [.bin .add "searchPos" "searchPos" (.imm 32)]).length))
      ((wseq [.bin .add "searchPos" "searchPos" (.imm 32)]).eval fuel
        ((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0).eval
          fuel ws))
      fuel (by simp only [SState.setPc]) hsegR.append_right hlrR.append_right
      (couple_setPc hcE _) (machInv_setPc ss2 _ hmiE)
  -- Assemble the reaches chain and the final coupling.
  have hRA : SReaches prog (n1 + (1 + (1 + (ne + 1)))) ss
      (ss2.setPc (base2 + (uifEmit "found" lElse lEnd et
        [.bin .add "searchPos" "searchPos" (.imm 32)]).length)) :=
    sreaches_trans prog n1 _ _ _ _ hr1
      (sreaches_trans prog 1 _ _ _ _ (sreaches_one_eq s0)
        (sreaches_trans prog 1 _ _ _ _ (sreaches_one_eq s1)
          (sreaches_trans prog ne 1 _ _ _ hrE (sreaches_one_eq s2))))
  refine ⟨(n1 + (1 + (1 + (ne + 1)))) + n5, ss5,
    sreaches_trans prog _ n5 _ _ _ hRA hr5, ?_, ?_, hmi5, ?_, ?_⟩
  · rw [hpc5]
    simp only [SState.setPc, loopCBodyEmit, hetdef, List.length_append, List.length_cons,
      List.length_nil, huplen]
    rw [← hbase2]; omega
  · -- `loopCBodyStmt.eval = setp.eval (notFound.eval cwWs)` since `found = 0`.
    have hevalChain : (loopCBodyStmt inStride hashLog).eval fuel ws
        = (WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval fuel
            ((wseq [.bin .add "searchPos" "searchPos" (.imm 32)]).eval fuel
              ((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
                hashLog 0).eval fuel ws)) := by
      simp only [loopCBodyStmt, wseq, WStmt.eval.eq_2, WStmt.eval.eq_10]
      rw [if_neg hb]
    rw [hevalChain]; exact hc5

  · have hHi : base + (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL
        cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length
        = base2 + (et.length + 1 + 4) + 1 := by
      simp only [loopCBodyEmit, hetdef, List.length_append, List.length_cons, List.length_nil,
        huplen]
      rw [← hbase2]; omega
    have hble : base ≤ base2 := by rw [← hbase2]; omega
    rw [hHi]
    have qA : PcIn base (base2 + (et.length + 1 + 4) + 1) ss1 := by
      show base ≤ ss1.pc ∧ ss1.pc ≤ base2 + (et.length + 1 + 4) + 1
      rw [hpc1']; omega
    have qB : PcIn base (base2 + (et.length + 1 + 4) + 1)
        (ss1.setPc (base2 + 1 + et.length + 1)) := by
      show base ≤ base2 + 1 + et.length + 1
        ∧ base2 + 1 + et.length + 1 ≤ base2 + (et.length + 1 + 4) + 1
      omega
    have qC : PcIn base (base2 + (et.length + 1 + 4) + 1)
        (ss1.setPc (base2 + 1 + et.length + 1 + 1)) := by
      show base ≤ base2 + 1 + et.length + 1 + 1
        ∧ base2 + 1 + et.length + 1 + 1 ≤ base2 + (et.length + 1 + 4) + 1
      omega
    have qD : PcIn base (base2 + (et.length + 1 + 4) + 1) ss2 := by
      show base ≤ ss2.pc ∧ ss2.pc ≤ base2 + (et.length + 1 + 4) + 1
      rw [hpcE']; omega
    have qE : PcIn base (base2 + (et.length + 1 + 4) + 1)
        (ss2.setPc (base2 + (uifEmit "found" lElse lEnd et
          [.bin .add "searchPos" "searchPos" (.imm 32)]).length)) := by
      show base ≤ base2 + (uifEmit "found" lElse lEnd et
          [.bin .add "searchPos" "searchPos" (.imm 32)]).length
        ∧ base2 + (uifEmit "found" lElse lEnd et
          [.bin .add "searchPos" "searchPos" (.imm 32)]).length
          ≤ base2 + (et.length + 1 + 4) + 1
      rw [huplen]; omega
    have W1 : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1)) n1 ss := by
      refine allSteps_weaken ?_ hst1
      intro st h
      obtain ⟨hA, hB⟩ := h
      exact ⟨by omega, by rw [← hbase2]; omega⟩
    have WE : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1)) ne
        (ss1.setPc (base2 + 1 + et.length + 1 + 1)) := by
      refine allSteps_weaken ?_ hstE
      intro st h
      obtain ⟨hA, hB⟩ := h
      simp only [List.length_cons, List.length_nil] at hB
      exact ⟨by omega, by omega⟩
    have W5 : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1)) n5
        (ss2.setPc (base2 + (uifEmit "found" lElse lEnd et
          [.bin .add "searchPos" "searchPos" (.imm 32)]).length)) := by
      refine allSteps_weaken ?_ hst5
      intro st h
      obtain ⟨hA, hB⟩ := h
      simp only [List.length_cons, List.length_nil, huplen] at hA hB
      exact ⟨by omega, by omega⟩
    have AS_D : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1)) (ne + 1)
        (ss1.setPc (base2 + 1 + et.length + 1 + 1)) :=
      allSteps_seq (n₁ := ne) (n₂ := 1) WE hrE (allSteps_one qD (by rw [s2]; exact qE))
    have AS_C : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1)) (1 + (ne + 1))
        (ss1.setPc (base2 + 1 + et.length + 1)) :=
      allSteps_seq (n₁ := 1) (n₂ := ne + 1)
        (allSteps_one qB (by rw [s1]; exact qC)) (sreaches_one_eq s1) AS_D
    have AS_B : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1))
        (1 + (1 + (ne + 1))) ss1 :=
      allSteps_seq (n₁ := 1) (n₂ := 1 + (ne + 1))
        (allSteps_one qA (by rw [s0]; exact qB)) (sreaches_one_eq s0) AS_C
    have AS_A : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1))
        (n1 + (1 + (1 + (ne + 1)))) ss :=
      allSteps_seq (n₁ := n1) (n₂ := 1 + (1 + (ne + 1))) W1 hr1 AS_B
    exact allSteps_seq (n₁ := n1 + (1 + (1 + (ne + 1)))) (n₂ := n5) AS_A hRA W5
  · rw [hbase2] at hMBdef
    have het : et.length = 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2
        + (wEmitMatchSeqEmit "litAnchor" "litLen" "off0" "ml" lElseL lEndL lHL lXL cpH cpX
            lElseM lEndM lHM lXM lsicL lsicM).length + 2 := by
      rw [← hetdef]
      simp only [foundBranchEmit, List.length_append, List.length_cons, List.length_nil]
      omega
    have f1 : (ss1.setPc (base2 + 1 + et.length + 1)).pc = base2 + 1 + et.length + 1 := rfl
    have f2 : (ss1.setPc (base2 + 1 + et.length + 1 + 1)).pc
        = base2 + 1 + et.length + 1 + 1 := rfl
    have f3 : (ss2.setPc (base2 + (uifEmit "found" lElse lEnd et
        [.bin .add "searchPos" "searchPos" (.imm 32)]).length)).pc
        = base2 + (uifEmit "found" lElse lEnd et
          [.bin .add "searchPos" "searchPos" (.imm 32)]).length := rfl
    have Q1 := allSteps_off_site (p := prog) (S := fun q => q = MB)
      (P := fun st => MatchEntryQ inStride LO st)
      (n := n1) (ss := ss) (pcIn_ne_of_gt MB hst1 (by rw [hMBdef, ← hbase2]; omega))
    have QE := allSteps_off_site (p := prog) (S := fun q => q = MB)
      (P := fun st => MatchEntryQ inStride LO st)
      (n := ne) (ss := ss1.setPc (base2 + 1 + et.length + 1 + 1))
      (pcIn_ne_of_lt MB hstE (by rw [hMBdef, het]; omega))
    have Q5 := allSteps_off_site (p := prog) (S := fun q => q = MB)
      (P := fun st => MatchEntryQ inStride LO st)
      (n := n5) (ss := ss2.setPc (base2 + (uifEmit "found" lElse lEnd et
        [.bin .add "searchPos" "searchPos" (.imm 32)]).length))
      (pcIn_ne_of_lt MB hst5 (by rw [hMBdef, huplen, het]; omega))
    have BD : AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) (ne + 1) (ss1.setPc (base2 + 1 + et.length + 1 + 1)) :=
      allSteps_seq (n₁ := ne) (n₂ := 1) QE hrE
        (allSteps_one (by intro h; exfalso; rw [hpcE', hMBdef, het] at h; omega)
          (by rw [s2]; intro h; exfalso; rw [f3, hMBdef, huplen, het] at h; omega))
    have BC : AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) (1 + (ne + 1))
        (ss1.setPc (base2 + 1 + et.length + 1)) :=
      allSteps_seq (n₁ := 1) (n₂ := ne + 1)
        (allSteps_one (by intro h; exfalso; rw [f1, hMBdef, het] at h; omega)
          (by rw [s1]; intro h; exfalso; rw [f2, hMBdef, het] at h; omega))
        (sreaches_one_eq s1) BD
    have BB : AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) (1 + (1 + (ne + 1))) ss1 :=
      allSteps_seq (n₁ := 1) (n₂ := 1 + (ne + 1))
        (allSteps_one (by intro h; exfalso; rw [hpc1', hMBdef] at h; omega)
          (by rw [s0]; intro h; exfalso; rw [f1, hMBdef, het] at h; omega))
        (sreaches_one_eq s0) BC
    have BA : AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) (n1 + (1 + (1 + (ne + 1)))) ss :=
      allSteps_seq (n₁ := n1) (n₂ := 1 + (1 + (ne + 1))) Q1 hr1 BB
    exact allSteps_seq (n₁ := n1 + (1 + (1 + (ne + 1)))) (n₂ := n5) BA hRA Q5



/-- **Transcribed from `loopCBody_found_reaches`, carrying pc-confinement.** -/
theorem loopCBody_found_reaches_pcIn (ib : Nat) (inStride hashLog : Nat)
    (lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM : String)
    (lsicL lsicM : List SInstr)
    (hLdef : lsicL = ([.mov "c255" (SArg.imm 255)]
      ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
        ++ ([.bin .sub "litExtra" "litExtra" (.imm 255)]
          ++ [.setp .ge "lsicC" "litExtra" (.imm 255)]))))
    (hMdef : lsicM = ([.mov "c255" (SArg.imm 255)]
      ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
        ++ ([.bin .sub "matExtra" "matExtra" (.imm 255)]
          ++ [.setp .ge "lsicC" "matExtra" (.imm 255)]))))
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride) (hlen : inStride < 2 ^ 40)
    (ws : WState) (hQ : LoopCQ inStride ws) (hguard : (ws.regs "loopC" == 1) = true)
    (hhl : hashLog ≤ 32) (fuel : Nat) (hfuelb : inStride ≤ fuel) (p c : Nat)
    (hw : window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat)
        (inStride - 12) (ws.regs "searchPos").toNat = some (p, c))
    (prog : Array SInstr) (base : Nat) (ss : SState)
    (hpc : ss.pc = base)
    (hseg : SegAt prog base (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL
      cpH cpX lElseM lEndM lHM lXM lsicL lsicM))
    (hlr : LabelsResolve prog base (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL
      lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM))
    (hc : Couple loopR ss ws) (hmi : MachInv ib ss) (hib40 : ib < 2 ^ 40)
    (LO : Nat) (hT : TightQ inStride LO ws) (MB : Nat)
    (hMBdef : MB = base + (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride
      (inStride - 12) hashLog).length + 1 + 4
      + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2)
    (hMBtop : ∀ q', q' ∈ succsOf prog MB → MB < q')
    (hMBno : ∀ q', MB ≤ q' → q' ≤ MB + (wEmitMatchSeqEmit "litAnchor" "litLen" "off0" "ml"
      lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length →
      MB ∉ succsOf prog q') :
    ∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
      ss'.pc = base + (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL
        cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length ∧
      Couple loopR ss' ((loopCBodyStmt inStride hashLog).eval fuel ws) ∧ MachInv ib ss' ∧
      AllSteps prog (PcIn base (base + (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL
        lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length)) m ss ∧
      AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) m ss := by
  obtain ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ := hQ
  have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
    loopCQ_guard inStride ws hstride hipos ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ hguard
  obtain ⟨cwib0, cwla_p0, cwp0lt, cwbud32, cwbudsz, cwobLB, cwcand, cwp4⟩ :=
    loopCBody_found_cwFacts inStride hashLog ws hstride hipos
      ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ hguard p c hw
  -- the found register is `1`.
  have hfound1 : (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
      hashLog 0 ws).regs "found" = 1 := by
    rw [evalCoopWindow_eq_go, hw, evalCoopWindowGo_regs]; simp [WState.setReg]
  -- the 9 wEmitMatchSeq side-conditions, on the coopWindow state.
  obtain ⟨sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9⟩ :=
    foundBranch_sideconds inStride (inStride - 5) fuel
      (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws)
      rfl hlen cwib0 cwla_p0 cwp0lt cwbud32 cwbudsz cwobLB cwcand cwp4 hfuelb
  obtain ⟨et, hetdef⟩ : ∃ x, foundBranchEmit (inStride - 5) lHE lEE lElseL lEndL lHL lXL cpH cpX
      lElseM lEndM lHM lXM lsicL lsicM = x := ⟨_, rfl⟩
  obtain ⟨base2, hbase2⟩ : ∃ x, base + (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride
      (inStride - 12) hashLog).length = x := ⟨_, rfl⟩
  rw [loopCBodyEmit] at hseg hlr
  -- Step 1: coopWindow, coupling p0/cand0.
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1, hst1⟩ :=
    coopWindow_couple_found_pcIn loopR inStride (inStride - 12) hashLog (ws.regs "searchPos").toNat p c
      (by decide) (by decide) (by decide) hib40 (by omega) (by omega) hhl hlen (by omega) (by decide) (by decide)
      (by decide) ws rfl (fun l ss'' hc'' => by rw [hc''.reg (by decide) l, UInt64.ofNat_toNat]) hw
      prog base ss fuel hpc hseg.append_left hlr.append_left hc hmi
  rw [hetdef] at hseg hlr
  have hpc1' : ss1.pc = base2 := by rw [hpc1, hbase2]
  have hsegR := hseg.append_right
  have hlrR := hlr.append_right
  rw [hbase2] at hsegR hlrR
  have hsegU : SegAt prog base2 (uifEmit "found" lElse lEnd et
      [.bin .add "searchPos" "searchPos" (.imm 32)]) := hsegR.append_left
  have hlrU : LabelsResolve prog base2 (uifEmit "found" lElse lEnd et
      [.bin .add "searchPos" "searchPos" (.imm 32)]) := hlrR.append_left
  obtain ⟨hbr, hseg1⟩ := hsegU.cons
  have hsegT : SegAt prog (base2 + 1) (foundBranchEmit (inStride - 5) lHE lEE lElseL lEndL lHL lXL
    cpH cpX lElseM lEndM lHM lXM lsicL lsicM) := by rw [hetdef]; exact hseg1.append_left
  obtain ⟨hbra, hseg3⟩ := hseg1.append_right.cons
  obtain ⟨hlblE, hseg4⟩ := hseg3.cons
  obtain ⟨hlblN, _⟩ := hseg4.append_right.cons
  have hlrT : LabelsResolve prog (base2 + 1) (foundBranchEmit (inStride - 5) lHE lEE lElseL lEndL
    lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM) := by rw [hetdef]; exact hlrU.cons.append_left
  have hLend : sfindLabel prog lEnd = base2 + 1 + et.length + 1 + 1 + 1 := by
    have := hlrU.cons.append_right.cons.cons.append_right 0 lEnd (by simp)
    simpa only [List.length_cons, List.length_nil] using this
  have hcv := hc1.reg (show "found" ∈ "p0" :: "cand0" :: loopR by decide) 0
  have hbr' : prog[ss1.pc]? = some (.braifnot "found" lElse) := by rw [hpc1']; exact hbr
  have hb1 : (((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
      hashLog 0 ws).regs "found" == 1) = true) := by rw [hfound1]; decide
  -- Step 2: the `uif` takes the then branch (`found = 1`).
  have s0 : sstep prog ss1 = ss1.setPc (base2 + 1) := by
    rw [braifnot_step prog ss1 "found" lElse hbr', hcv, if_pos hb1, hpc1']
  -- Step 3: the found branch (`simSL'_foundBranch`) at `R' = p0 :: cand0 :: loopR`.
  rw [hbase2] at hMBdef
  obtain ⟨n4, ss4, hr4, hpc4, hc4, hmi4, hst4, hck4⟩ :=
    simSLQ_foundBranch_ckpt ib ("p0" :: "cand0" :: loopR) inStride (inStride - 5)
      lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM
      ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide⟩ rfl hlen
      hib40 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) lsicL lsicM hLdef hMdef
      prog (base2 + 1) (ss1.setPc (base2 + 1))
      (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws)
      fuel rfl hsegT hlrT
      (couple_setPc hc1 _) (machInv_setPc ss1 _ hmi1)
      (by rw [← hMBdef]; exact hMBtop) (by rw [← hMBdef]; exact hMBno)
      cwcand cwp4 (by have := cwp4; have := hfuelb; have := hipos; omega)
      sc1 sc2 sc3 sc4 sc5 sc6 sc7 sc8 sc9
  -- the tight bound at the match entry: `op` at the END of this iteration is `≤ LO`
  -- (the invariant `LZ4Tight` carries), and the token's own bytes are the difference.
  have hbndBr : (((foundBranchStmt inStride (inStride - 5)).eval fuel
      (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws)).regs
        "op").toNat ≤ LO := by
    have hTb := loopCBody_tight inStride hashLog fuel LO ws hstride hipos (by omega)
      hfuelb ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ hguard hT
    have hb1' : (((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
        hashLog 0).eval fuel ws).regs "found" == 1) = true := by
      simp only [WStmt.eval]; rw [hfound1]; decide
    have hchain : (loopCBodyStmt inStride hashLog).eval fuel ws
        = (WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval fuel
            ((foundBranchStmt inStride (inStride - 5)).eval fuel
              (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
                hashLog 0 ws)) := by
      simp only [loopCBodyStmt, wseq, WStmt.eval.eq_2, WStmt.eval.eq_10]
      rw [if_pos hb1']
      simp only [WStmt.eval]
    simp only [TightQ, tightRem, hchain] at hTb
    simp only [WStmt.eval, WState.setReg, String.reduceEq, if_false] at hTb
    omega
  have htok := foundMatchEntry_tokBound inStride (inStride - 5) fuel LO
    (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws)
    rfl hlen cwla_p0 cwp0lt cwbud32 cwcand cwp4 hfuelb hbndBr
  have hck4' : AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st)
      n4 (ss1.setPc (base2 + 1)) := by
    rw [hMBdef]
    refine allSteps_weaken (fun st h hq => ⟨_, h hq, ?_, ?_, htok.1, htok.2⟩) hck4
    · exact (foundMatchEntry_budget inStride (inStride - 5) fuel
        (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws)
        cwbud32 cwbudsz).1
    · exact (foundMatchEntry_budget inStride (inStride - 5) fuel
        (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws)
        cwbud32 cwbudsz).2
  have hcwe : (WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
      hashLog 0).eval fuel ws
      = evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws := by
    simp only [WStmt.eval]
  rw [← hcwe] at hc4
  have hpc4' : ss4.pc = base2 + 1 + et.length := by
    rw [hpc4, hetdef]
  have hbra' : prog[ss4.pc]? = some (.bra lEnd) := by rw [hpc4']; exact hbra
  have s1 : sstep prog ss4 = ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1) := by
    rw [bra_step prog ss4 lEnd hbra', hLend]
  have hlblN' : prog[(ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1)).pc]? = some (.lbl lEnd) :=
    hlblN
  have s2 : sstep prog (ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1))
      = ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1 + 1) := by
    rw [lbl_step prog _ lEnd hlblN']; simp [SState.setPc]
  have huplen : (uifEmit "found" lElse lEnd et
      [.bin .add "searchPos" "searchPos" (.imm 32)]).length = et.length + 1 + 4 := by
    rw [uifEmit_length]; simp only [List.length_cons, List.length_nil]
  -- Step 4: the trailing `setp loopC`.
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5, hst5⟩ :=
    simSLQ_setp_pcIn ib ("p0" :: "cand0" :: loopR) .lt "loopC" "searchPos" (.imm (inStride - 12))
      (by decide) (fun n h => by cases h) (by decide)
      prog (base2 + (uifEmit "found" lElse lEnd et [.bin .add "searchPos" "searchPos" (.imm 32)]).length)
      (ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1 + 1))
      ((foundBranchStmt inStride (inStride - 5)).eval fuel
        ((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0).eval
          fuel ws))
      fuel (by simp only [SState.setPc, huplen]; omega) hsegR.append_right hlrR.append_right
      (couple_setPc hc4 _) (machInv_setPc ss4 _ hmi4)
  have hRA : SReaches prog (n1 + (1 + (n4 + (1 + 1)))) ss
      (ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1 + 1)) :=
    sreaches_trans prog n1 _ _ _ _ hr1
      (sreaches_trans prog 1 _ _ _ _ (sreaches_one_eq s0)
        (sreaches_trans prog n4 _ _ _ _ hr4
          (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s1) (sreaches_one_eq s2))))
  refine ⟨(n1 + (1 + (n4 + (1 + 1)))) + n5, ss5,
    sreaches_trans prog _ n5 _ _ _ hRA hr5,
    ?_, ?_, hmi5, ?_, ?_⟩
  · rw [hpc5]
    simp only [SState.setPc, loopCBodyEmit, hetdef, List.length_append, List.length_cons,
      List.length_nil, huplen]
    rw [← hbase2]; omega
  · -- `loopCBodyStmt.eval = setp.eval (foundBranch.eval cwWs)` since `found = 1`, then `drop2`.
    have hevalChain : (loopCBodyStmt inStride hashLog).eval fuel ws
        = (WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval fuel
            ((foundBranchStmt inStride (inStride - 5)).eval fuel
              ((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
                hashLog 0).eval fuel ws)) := by
      have hb1' : (((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
          hashLog 0).eval fuel ws).regs "found" == 1) = true := by rw [hcwe, hfound1]; decide
      simp only [loopCBodyStmt, wseq, WStmt.eval.eq_2, WStmt.eval.eq_10]
      rw [if_pos hb1']
    rw [hevalChain]
    exact Couple.drop2 loopR ss5 _ "p0" "cand0" hc5

  · have hHi : base + (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL
        cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length
        = base2 + (et.length + 1 + 4) + 1 := by
      simp only [loopCBodyEmit, hetdef, List.length_append, List.length_cons, List.length_nil,
        huplen]
      rw [← hbase2]; omega
    have hble : base ≤ base2 := by rw [← hbase2]; omega
    rw [hHi]
    have qA : PcIn base (base2 + (et.length + 1 + 4) + 1) ss1 := by
      show base ≤ ss1.pc ∧ ss1.pc ≤ base2 + (et.length + 1 + 4) + 1
      rw [hpc1']; omega
    have qB : PcIn base (base2 + (et.length + 1 + 4) + 1) (ss1.setPc (base2 + 1)) := by
      show base ≤ base2 + 1 ∧ base2 + 1 ≤ base2 + (et.length + 1 + 4) + 1
      omega
    have qC : PcIn base (base2 + (et.length + 1 + 4) + 1) ss4 := by
      show base ≤ ss4.pc ∧ ss4.pc ≤ base2 + (et.length + 1 + 4) + 1
      rw [hpc4']; omega
    have qD : PcIn base (base2 + (et.length + 1 + 4) + 1)
        (ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1)) := by
      show base ≤ base2 + 1 + et.length + 1 + 1 + 1
        ∧ base2 + 1 + et.length + 1 + 1 + 1 ≤ base2 + (et.length + 1 + 4) + 1
      omega
    have qE : PcIn base (base2 + (et.length + 1 + 4) + 1)
        (ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1 + 1)) := by
      show base ≤ base2 + 1 + et.length + 1 + 1 + 1 + 1
        ∧ base2 + 1 + et.length + 1 + 1 + 1 + 1 ≤ base2 + (et.length + 1 + 4) + 1
      omega
    have W1 : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1)) n1 ss := by
      refine allSteps_weaken ?_ hst1
      intro st h
      obtain ⟨hA, hB⟩ := h
      exact ⟨by omega, by rw [← hbase2]; omega⟩
    have W4 : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1)) n4
        (ss1.setPc (base2 + 1)) := by
      refine allSteps_weaken ?_ hst4
      intro st h
      obtain ⟨hA, hB⟩ := h
      simp only [hetdef] at hB
      exact ⟨by omega, by omega⟩
    have W5 : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1)) n5
        (ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1 + 1)) := by
      refine allSteps_weaken ?_ hst5
      intro st h
      obtain ⟨hA, hB⟩ := h
      simp only [List.length_cons, List.length_nil, huplen] at hA hB
      exact ⟨by omega, by omega⟩
    have AS_2 : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1)) (1 + 1) ss4 :=
      allSteps_seq (n₁ := 1) (n₂ := 1)
        (allSteps_one qC (by rw [s1]; exact qD)) (sreaches_one_eq s1)
        (allSteps_one qD (by rw [s2]; exact qE))
    have AS_3 : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1)) (n4 + (1 + 1))
        (ss1.setPc (base2 + 1)) :=
      allSteps_seq (n₁ := n4) (n₂ := 1 + 1) W4 hr4 AS_2
    have AS_4 : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1))
        (1 + (n4 + (1 + 1))) ss1 :=
      allSteps_seq (n₁ := 1) (n₂ := n4 + (1 + 1))
        (allSteps_one qA (by rw [s0]; exact qB)) (sreaches_one_eq s0) AS_3
    have AS_5 : AllSteps prog (PcIn base (base2 + (et.length + 1 + 4) + 1))
        (n1 + (1 + (n4 + (1 + 1)))) ss :=
      allSteps_seq (n₁ := n1) (n₂ := 1 + (n4 + (1 + 1))) W1 hr1 AS_4
    exact allSteps_seq (n₁ := n1 + (1 + (n4 + (1 + 1)))) (n₂ := n5) AS_5 hRA W5
  · have het : et.length = 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2
        + (wEmitMatchSeqEmit "litAnchor" "litLen" "off0" "ml" lElseL lEndL lHL lXL cpH cpX
            lElseM lEndM lHM lXM lsicL lsicM).length + 2 := by
      rw [← hetdef]
      simp only [foundBranchEmit, List.length_append, List.length_cons, List.length_nil]
      omega
    have e1 : (ss1.setPc (base2 + 1)).pc = base2 + 1 := rfl
    have e2 : (ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1)).pc
        = base2 + 1 + et.length + 1 + 1 + 1 := rfl
    have e3 : (ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1 + 1)).pc
        = base2 + 1 + et.length + 1 + 1 + 1 + 1 := rfl
    have P1 := allSteps_off_site (p := prog) (S := fun q => q = MB)
      (P := fun st => MatchEntryQ inStride LO st)
      (n := n1) (ss := ss) (pcIn_ne_of_gt MB hst1 (by rw [hMBdef, ← hbase2]; omega))
    have P5 := allSteps_off_site (p := prog) (S := fun q => q = MB)
      (P := fun st => MatchEntryQ inStride LO st)
      (n := n5) (ss := ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1 + 1))
      (pcIn_ne_of_lt MB hst5 (by rw [hMBdef, huplen, het]; omega))
    have AS2 : AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) (1 + 1) ss4 :=
      allSteps_seq (n₁ := 1) (n₂ := 1)
        (allSteps_one (by intro h; exfalso; rw [hpc4', hMBdef, het] at h; omega)
          (by rw [s1]; intro h; exfalso; rw [e2, hMBdef, het] at h; omega))
        (sreaches_one_eq s1)
        (allSteps_one (by intro h; exfalso; rw [e2, hMBdef, het] at h; omega)
          (by rw [s2]; intro h; exfalso; rw [e3, hMBdef, het] at h; omega))
    have AS3 : AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) (n4 + (1 + 1)) (ss1.setPc (base2 + 1)) :=
      allSteps_seq (n₁ := n4) (n₂ := 1 + 1) hck4' hr4 AS2
    have AS4 : AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) (1 + (n4 + (1 + 1))) ss1 :=
      allSteps_seq (n₁ := 1) (n₂ := n4 + (1 + 1))
        (allSteps_one (by intro h; exfalso; rw [hpc1', hMBdef] at h; omega)
          (by rw [s0]; intro h; exfalso; rw [e1, hMBdef] at h; omega))
        (sreaches_one_eq s0) AS3
    have AS5 : AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) (n1 + (1 + (n4 + (1 + 1)))) ss :=
      allSteps_seq (n₁ := n1) (n₂ := 1 + (n4 + (1 + 1))) P1 hr1 AS4
    exact allSteps_seq (n₁ := n1 + (1 + (n4 + (1 + 1)))) (n₂ := n5) AS5 hRA P5



/-- **Transcribed from `loopCBody_bodySim`, carrying pc-confinement.** -/
theorem loopCBody_bodySim_pcIn (ib : Nat) (inStride hashLog F : Nat)
    (lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM : String)
    (lsicL lsicM : List SInstr)
    (hLdef : lsicL = ([.mov "c255" (SArg.imm 255)]
      ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
        ++ ([.bin .sub "litExtra" "litExtra" (.imm 255)]
          ++ [.setp .ge "lsicC" "litExtra" (.imm 255)]))))
    (hMdef : lsicM = ([.mov "c255" (SArg.imm 255)]
      ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
        ++ ([.bin .sub "matExtra" "matExtra" (.imm 255)]
          ++ [.setp .ge "lsicC" "matExtra" (.imm 255)]))))
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride) (hlen : inStride < 2 ^ 40)
    (hib40 : ib < 2 ^ 40)
    (hhl : hashLog ≤ 32) (hFb : 2 * inStride + 1 ≤ F)
    (prog : Array SInstr) (base : Nat) (ss : SState) (ws : WState) (fuel : Nat)
    (hpc : ss.pc = base + 1 + 1)
    (hseg : SegAt prog (base + 1 + 1) (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL
      lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM))
    (hlr : LabelsResolve prog (base + 1 + 1) (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE
      lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM))
    (hc : Couple loopR ss ws) (hmi : MachInv ib ss) (hQ : LoopCQ inStride ws)
    (hguard : (ws.regs "loopC" == 1) = true)
    (hfloor : F ≤ (ws.regs "searchPos").toNat + fuel + 1)
    (LO : Nat) (hT : TightQ inStride LO ws) (MB : Nat)
    (hMBdef : MB = base + 1 + 1 + (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride
      (inStride - 12) hashLog).length + 1 + 4
      + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2)
    (hMBtop : ∀ q', q' ∈ succsOf prog MB → MB < q')
    (hMBno : ∀ q', MB ≤ q' → q' ≤ MB + (wEmitMatchSeqEmit "litAnchor" "litLen" "off0" "ml"
      lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length →
      MB ∉ succsOf prog q') :
    (∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
      ss'.pc = base + 1 + 1 + (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL
        lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length ∧
      Couple loopR ss' ((loopCBodyStmt inStride hashLog).eval fuel ws) ∧ MachInv ib ss' ∧
      AllSteps prog (PcIn (base + 1 + 1) (base + 1 + 1 + (loopCBodyEmit inStride hashLog lElse lEnd
        lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length)) m ss ∧
      AllSteps prog (fun st => st.pc = MB → MatchEntryQ inStride LO st) m ss)
    ∧ (LoopCQ inStride ((loopCBodyStmt inStride hashLog).eval fuel ws)
        ∧ TightQ inStride LO ((loopCBodyStmt inStride hashLog).eval fuel ws))
    ∧ (ws.regs "searchPos").toNat + 1
        ≤ (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "searchPos").toNat := by
  have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
    loopCQ_guard inStride ws hstride hipos hQ hguard
  have hfuelb : inStride ≤ fuel := by omega
  refine ⟨?_, ⟨(loopCBody_Qadvance inStride hashLog fuel ws hstride hipos (by omega)
    hfuelb hQ hguard).1,
    loopCBody_tight inStride hashLog fuel LO ws hstride hipos (by omega) hfuelb hQ hguard hT⟩,
    (loopCBody_Qadvance inStride hashLog fuel ws hstride hipos (by omega)
    hfuelb hQ hguard).2.1⟩
  cases hw : window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat)
      (inStride - 12) (ws.regs "searchPos").toNat with
  | none =>
      exact loopCBody_none_reaches_pcIn ib inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX
        lElseM lEndM lHM lXM lsicL lsicM hstride hipos hlen ws hQ hguard hhl hw prog (base + 1 + 1)
        ss fuel hpc hseg hlr hc hmi hib40 LO MB hMBdef
  | some pc =>
      obtain ⟨p, c⟩ := pc
      exact loopCBody_found_reaches_pcIn ib inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX
        lElseM lEndM lHM lXM lsicL lsicM hLdef hMdef hstride hipos hlen ws hQ hguard hhl fuel hfuelb
        p c hw prog (base + 1 + 1) ss hpc hseg hlr hc hmi hib40 LO hT MB hMBdef hMBtop hMBno


/-- **The `loopC` loop, exporting a coupled eval state at every visit to its head.**

    Identical to `loopC_loop_sim` except for the extra conclusion: at each of the
    `n` steps, if the machine is at the loop head then there is a `ws` it couples
    to satisfying the loop invariant.  That is what the output-cursor bound needs
    at the token entry, and nothing in the simulation exported it before.

    "The body never returns to the head" is not a hypothesis: the body's own
    pc-confinement (`loopCBody_bodySim_pcIn`) says every step of one iteration sits
    at or above `base + 2`, and `pcIn_ne_of_lt` turns that into the off-site
    condition `allSteps_off_site` needs. -/
theorem loopC_loop_sim_ckpt (ib : Nat) (inStride hashLog F : Nat)
    (lHeadC lEndC lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM : String)
    (lsicL lsicM : List SInstr)
    (hLdef : lsicL = ([.mov "c255" (SArg.imm 255)]
      ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
        ++ ([.bin .sub "litExtra" "litExtra" (.imm 255)]
          ++ [.setp .ge "lsicC" "litExtra" (.imm 255)]))))
    (hMdef : lsicM = ([.mov "c255" (SArg.imm 255)]
      ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
        ++ ([.bin .sub "matExtra" "matExtra" (.imm 255)]
          ++ [.setp .ge "lsicC" "matExtra" (.imm 255)]))))
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride) (hlen : inStride < 2 ^ 40)
    (hib40 : ib < 2 ^ 40)
    (hhl : hashLog ≤ 32) (hFb : 2 * inStride + 1 ≤ F)
    (prog : Array SInstr) (base : Nat)
    (hseg : SegAt prog base (uwhileEmit "loopC" lHeadC lEndC (loopCBodyEmit inStride hashLog lElse
      lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM)))
    (hlr : LabelsResolve prog base (uwhileEmit "loopC" lHeadC lEndC (loopCBodyEmit inStride hashLog
      lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM)))
    (LO : Nat) (MB : Nat)
    (hMBdef : MB = base + 1 + 1 + (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride
      (inStride - 12) hashLog).length + 1 + 4
      + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2)
    (hMBtop : ∀ q', q' ∈ succsOf prog MB → MB < q')
    (hMBno : ∀ q', MB ≤ q' → q' ≤ MB + (wEmitMatchSeqEmit "litAnchor" "litLen" "off0" "ml"
      lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length →
      MB ∉ succsOf prog q')
    :
    ∀ (fuel : Nat) (ss : SState) (ws : WState),
      ss.pc = base → Couple loopR ss ws → MachInv ib ss →
      (LoopCQ inStride ws ∧ TightQ inStride LO ws) →
      F ≤ (ws.regs "searchPos").toNat + fuel →
      WhileHalts "loopC" (loopCBodyStmt inStride hashLog) fuel ws →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base + (uwhileEmit "loopC" lHeadC lEndC (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM)).length ∧
        Couple loopR ss' ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval fuel ws)
          ∧ MachInv ib ss' ∧ AllSteps prog
          (fun st => (st.pc = base → ∃ w, Couple loopR st w ∧ LoopCQ inStride w
              ∧ TightQ inStride LO w)
      ∧ (st.pc = MB → MatchEntryQ inStride LO st)) n ss :=
  simSL'_measureLoopSteps ib loopR "loopC" lHeadC lEndC
    (fun st => (st.pc = base → ∃ w, Couple loopR st w ∧ LoopCQ inStride w
        ∧ TightQ inStride LO w)
      ∧ (st.pc = MB → MatchEntryQ inStride LO st))
    (fun w => LoopCQ inStride w ∧ TightQ inStride LO w) (fun ws => (ws.regs "searchPos").toNat) F (loopCBodyStmt inStride hashLog)
    (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM) prog base (by decide)
    (fun st w hsc hc hmi hq =>
      ⟨fun _ => ⟨w, hc, hq.1, hq.2⟩,
       fun hmb => absurd hmb (by
         have hlen2 : (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL
             cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length
             = (coopWindowEmit "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
                 hashLog).length
               + ((foundBranchEmit (inStride - 5) lHE lEE lElseL lEndL lHL lXL cpH cpX
                     lElseM lEndM lHM lXM lsicL lsicM).length + 1 + 4) + 1 := by
           simp only [loopCBodyEmit, uifEmit_length, List.length_append, List.length_cons,
             List.length_nil]
           omega
         have hfb : 4 + (uwhileEmit "extC" lHE lEE foundExtBodyEmit).length + 2
             ≤ (foundBranchEmit (inStride - 5) lHE lEE lElseL lEndL lHL lXL cpH cpX
                 lElseM lEndM lHM lXM lsicL lsicM).length := by
           simp only [foundBranchEmit, List.length_append, List.length_cons, List.length_nil]
           omega
         intro he
         rcases hsc with e | e | e | e | e <;> rw [hMBdef] at he <;> omega)⟩)
    (fun ss ws fuel h1 h2 h3 h4 h5 h6 h7 h8 => by
      obtain ⟨⟨m, ss', hr, hpc, hc, hmi, hst, hckB⟩, hQ2, hmu⟩ :=
        loopCBody_bodySim_pcIn ib inStride hashLog F lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX
          lElseM lEndM lHM lXM lsicL lsicM hLdef hMdef hstride hipos hlen hib40 hhl hFb prog base
          ss ws fuel h1 h2 h3 h4 h5 h6.1 h7 h8 LO h6.2 MB hMBdef hMBtop hMBno
      exact ⟨⟨m, ss', hr, hpc, hc, hmi, fun j hj =>
        ⟨allSteps_off_site (p := prog) (S := fun q => q = base)
          (P := fun st => ∃ w, Couple loopR st w ∧ LoopCQ inStride w ∧ TightQ inStride LO w)
          (n := m) (ss := ss)
          (pcIn_ne_of_lt base hst (by omega)) j hj, hckB j hj⟩⟩, hQ2, hmu⟩)
    hseg hlr

end AlgorithmLib.LZ4WarpDSL
