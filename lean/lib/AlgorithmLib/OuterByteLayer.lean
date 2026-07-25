import AlgorithmLib.ByteLayer
import AlgorithmLib.EmitContent
import AlgorithmLib.EncodeLen
import AlgorithmLib.LZ4WarpEvalBytes_proofs

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib.LZ4Plan AlgorithmLib.LZ4WarpFind AlgorithmLib.LZ4
open AlgorithmLib.LZ4Simt (SCmp)

-- Sanity: the model-side plan generator and encoder are in scope.
example (inp : List UInt8) (p : Plan) : LZ4.Block := planToBlock inp p

-- ── `smem` is untouched by any `WStmt` containing no `.stshU`/`.coopWindow`. ──
-- (The only two `WStmt.eval` cases that write `smem`.)  Structural, fuel-generic.
def NoSmem : WStmt → Prop
  | .skip => True
  | .seq a b => NoSmem a ∧ NoSmem b
  | .bin _ _ _ _ => True
  | .setp _ _ _ _ => True
  | .mov _ _ => True
  | .ldgB _ _ _ => True
  | .ldshU _ _ => True
  | .stgB _ _ => True
  | .stshU _ _ => False
  | .uif _ t e => NoSmem t ∧ NoSmem e
  | .uwhile _ body => NoSmem body
  | .coopCopy _ _ _ => True
  | .coopWindow .. => False
  | .coopExtendStep .. => True

theorem noSmem_eval_smem : ∀ (s : WStmt), NoSmem s → ∀ (fuel : Nat) (st : WState),
    (s.eval fuel st).smem = st.smem
  | .skip, _, _, _ => by simp [WStmt.eval]
  | .seq a b, ⟨hna, hnb⟩, fuel, st => by
      simp only [WStmt.eval]
      rw [noSmem_eval_smem b hnb fuel (a.eval fuel st), noSmem_eval_smem a hna fuel st]
  | .bin _ _ _ _, _, _, _ => by simp [WStmt.eval, WState.setReg]
  | .setp _ _ _ _, _, _, _ => by simp [WStmt.eval, WState.setReg]
  | .mov _ _, _, _, _ => by simp [WStmt.eval, WState.setReg]
  | .ldgB _ _ _, _, _, _ => by simp [WStmt.eval, WState.setReg]
  | .ldshU _ _, _, _, _ => by simp [WStmt.eval, WState.setReg]
  | .stgB _ _, _, _, _ => by simp [WStmt.eval, WState.stgByte]
  | .uif cond t e, ⟨hnt, hne⟩, fuel, st => by
      simp only [WStmt.eval]
      split
      · exact noSmem_eval_smem t hnt fuel st
      · exact noSmem_eval_smem e hne fuel st
  | .uwhile cond body, hn, fuel, st => by
      induction fuel generalizing st with
      | zero => simp [WStmt.eval]
      | succ n ih =>
          simp only [WStmt.eval]
          split
          · rw [ih (body.eval n st), noSmem_eval_smem body hn n st]
          · rfl
  | .coopCopy _ _ _, _, _, _ => by simp [WStmt.eval]
  | .coopExtendStep .., _, _, _ => by simp [WStmt.eval, evalCoopExtendStep, WState.setReg]

-- ── Generic register-frame: `s` preserves register `r` if `r` is never a ────
-- write-target anywhere in `s` (`.bin/.setp/.mov/.ldgB` write `d`; `.stgB`/
-- `.stshU` write memory, not a register; `.coopCopy` writes only `gmem`;
-- `.coopWindow`/`.coopExtendStep` write their own fixed named outputs).
def NoWrite (r : String) : WStmt → Prop
  | .skip => True
  | .seq a b => NoWrite r a ∧ NoWrite r b
  | .bin _ d _ _ => r ≠ d
  | .setp _ d _ _ => r ≠ d
  | .mov d _ => r ≠ d
  | .ldgB d _ _ => r ≠ d
  | .ldshU d _ => r ≠ d
  | .stgB _ _ => True
  | .stshU _ _ => True
  | .uif _ t e => NoWrite r t ∧ NoWrite r e
  | .uwhile _ body => NoWrite r body
  | .coopCopy _ _ _ => True
  | .coopWindow found p0 cand0 _ _ _ _ _ => r ≠ found ∧ r ≠ p0 ∧ r ≠ cand0
  | .coopExtendStep adv _ _ ml _ _ => r ≠ adv ∧ r ≠ ml

theorem noWrite_eval_reg : ∀ (s : WStmt) (r : String), NoWrite r s →
    ∀ (fuel : Nat) (st : WState), (s.eval fuel st).regs r = st.regs r := by
  intro s
  induction s with
  | skip => intro _ _ _ _; simp [WStmt.eval]
  | seq a b iha ihb =>
      intro r ⟨hna, hnb⟩ fuel st
      simp only [WStmt.eval]
      rw [ihb r hnb fuel (a.eval fuel st), iha r hna fuel st]
  | bin o d a b =>
      intro r hr _ _
      simp only [WStmt.eval, WState.setReg]; exact if_neg hr
  | setp c d a b =>
      intro r hr _ _
      simp only [WStmt.eval, WState.setReg]; exact if_neg hr
  | mov d a =>
      intro r hr _ _
      simp only [WStmt.eval, WState.setReg]; exact if_neg hr
  | ldgB d addr off =>
      intro r hr _ _
      simp only [WStmt.eval, WState.setReg]; exact if_neg hr
  | ldshU d addr =>
      intro r hr _ _
      simp only [WStmt.eval, WState.setReg]; exact if_neg hr
  | stgB addr val => intro _ _ _ _; simp [WStmt.eval, WState.stgByte]
  | stshU addr val => intro _ _ _ _; simp [WStmt.eval, WState.stshU16]
  | uif cond t e iht ihe =>
      intro r ⟨hnt, hne⟩ fuel st
      simp only [WStmt.eval]
      split
      · exact iht r hnt fuel st
      · exact ihe r hne fuel st
  | uwhile cond body ihb =>
      intro r hn fuel st
      induction fuel generalizing st with
      | zero => simp [WStmt.eval]
      | succ n ih =>
          simp only [WStmt.eval]
          split
          · rw [ih (body.eval n st), ihb r hn n st]
          · rfl
  | coopCopy dst src len => intro _ _ _ _; simp [WStmt.eval]
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      intro r ⟨hadv, hml⟩ fuel st
      simp only [WStmt.eval, evalCoopExtendStep, WState.setReg]
      exact if_neg hadv
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      intro r ⟨hf, hp, hc⟩ fuel st
      simp only [WStmt.eval]
      rw [evalCoopWindow_eq_go, evalCoopWindowGo_regs]
      split
      · simp only [WState.setReg]
        rw [if_neg hc, if_neg hp, if_neg hf]
      · simp only [WState.setReg]
        exact if_neg hf

-- ── `wEmitMatchSeq`/`wEmitFinalSeq` preserve any register outside their fixed ──
-- scratch set, via the generic `noWrite_eval_reg` (no `.stshU`/`.coopWindow`
-- inside either body, and `r` disjoint from every scratch name they write).
theorem wEmitMatchSeq_frame (litStart litLen off ml r : String) (ws : WState) (fuel : Nat)
    (hr1 : r ≠ "mlm") (hr2 : r ≠ "tokLo") (hr3 : r ≠ "tokHi") (hr4 : r ≠ "tok")
    (hr5 : r ≠ "sbAddr") (hr6 : r ≠ "op") (hr7 : r ≠ "pLitBig") (hr8 : r ≠ "litExtra")
    (hr9 : r ≠ "c255") (hr10 : r ≠ "lsicC") (hr11 : r ≠ "cpDst") (hr12 : r ≠ "cpSrc")
    (hr13 : r ≠ "offLo") (hr14 : r ≠ "offHi") (hr15 : r ≠ "pMatBig") (hr16 : r ≠ "matExtra") :
    ((wEmitMatchSeq litStart litLen off ml).eval fuel ws).regs r = ws.regs r := by
  apply noWrite_eval_reg
  simp only [wEmitMatchSeq, wEmitToken, wEmitLSIC, wStoreByte, wseq, NoWrite]
  repeat' constructor
  all_goals first
    | exact hr1 | exact hr2 | exact hr3 | exact hr4 | exact hr5 | exact hr6 | exact hr7
    | exact hr8 | exact hr9 | exact hr10 | exact hr11 | exact hr12 | exact hr13 | exact hr14
    | exact hr15 | exact hr16 | trivial

-- ── The `found` branch of one `uwhile "loopC"` iteration: window hit at `(p,c)`. ──
-- After `coopWindow` (found=1,p0=p,cand0=c), the extend sub-loop, and
-- `wEmitMatchSeq`, output = `encodeSeq ⟨lits, p-c, ml⟩` where `ml = extendFrom`;
-- `searchPos`/`litAnchor` both advance to `p+ml`.
theorem emitStepFound_eq (inStride searchLim hashLog s p c : Nat) (F fuelF : Nat)
    (ws : WState) (anchor : Nat)
    (hstride : inStride ≤ 65536) (hsl : searchLim = inStride - 12) (hsl' : searchLim ≤ inStride)
    (hwin : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) searchLim s = some (p, c))
    (has : anchor ≤ s) (hsval : (ws.regs "searchPos").toNat = s)
    (hlaval : (ws.regs "litAnchor").toNat = anchor)
    (hCfuel : (inStride - 5) - (p + 4) ≤ 32 * F) (hFfuel : (inStride - 5) - (p + 4) ≤ F)
    (hfuelL : (p - anchor) / 255 < F) (hfuelM : (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
        (inStride - 5) F 4 - 4) / 255 < F)
    (hp64 : inStride < 2 ^ 64)
    (hnwFull : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
        + (encNib (p - anchor)).length + (p - anchor) + 2
        + (encNib (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) F 4 - 4)).length
        < 2 ^ 64)
    (hnw2 : (ws.regs "inBase").toNat + anchor + (p - anchor) < 2 ^ 64)
    (hsize : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
        + (encNib (p - anchor)).length + (p - anchor) ≤ ws.gmem.size)
    (hdisj : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
        + (encNib (p - anchor)).length + (p - anchor) ≤ (ws.regs "inBase").toNat + anchor
      ∨ (ws.regs "inBase").toNat + anchor + (p - anchor) ≤ (ws.regs "outBase").toNat + (ws.regs "op").toNat) :
    let endCap := inStride - 5
    let ml := extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c endCap F 4
    let st' := (wseq [ .coopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0,
        .uif "found" (wseq
          [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)), .mov "ml" (.imm 4),
            .setp .ge "extC" "ml" (.imm 0),
            .uwhile "extC" (wseq
              [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
                .bin .add "ml" "ml" (.reg "adv"),
                .setp .eq "extC" "adv" (.imm 32) ]),
            .bin .sub "off0" "p0" (.reg "cand0"),
            .bin .sub "litLen" "p0" (.reg "litAnchor"),
            wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
            .bin .add "litAnchor" "p0" (.reg "ml"),
            .mov "searchPos" (.reg "litAnchor") ])
          (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]) ]).eval F ws
    st'.gmem = EmitContent.putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
          (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
              (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0), p - c, ml⟩)
        ∧ (st'.regs "searchPos").toNat = p + ml ∧ (st'.regs "litAnchor").toNat = p + ml
        ∧ st'.regs "outBase" = ws.regs "outBase" ∧ st'.regs "inBase" = ws.regs "inBase"
        ∧ st'.regs "op" = ws.regs "op" + UInt64.ofNat (1 + (encNib (p - anchor)).length
            + (p - anchor) + 2 + (encNib (ml - 4)).length)
        ∧ st'.smem = tableInsert ws.smem ws.gmem hashLog 0 searchLim (ws.regs "inBase").toNat s (p - s) := by
  intro endCap ml st'
  show st'.gmem = _ ∧ (st'.regs "searchPos").toNat = _ ∧ (st'.regs "litAnchor").toNat = _
    ∧ st'.regs "outBase" = _ ∧ st'.regs "inBase" = _ ∧ st'.regs "op" = _ ∧ st'.smem = _
  subst st'
  -- Step 1: the `.coopWindow` step, routed through `evalCoopWindow_eq_go` (never
  -- `simp`-unfolding `evalCoopWindow`/`window` directly — kernel-size lesson).
  obtain ⟨st1, hst1def⟩ :
      ∃ st1, evalCoopWindowGo "found" "p0" "cand0" "searchPos" hashLog 0 searchLim (ws.regs "inBase").toNat
        (some (p, c)) ws = st1 := ⟨_, rfl⟩
  have hst1_found : st1.regs "found" = 1 := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_p0 : st1.regs "p0" = UInt64.ofNat p := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_cand0 : st1.regs "cand0" = UInt64.ofNat c := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_gmem : st1.gmem = ws.gmem := by rw [← hst1def]; rfl
  have hst1_la : st1.regs "litAnchor" = ws.regs "litAnchor" := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_ib : st1.regs "inBase" = ws.regs "inBase" := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_ob : st1.regs "outBase" = ws.regs "outBase" := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_op : st1.regs "op" = ws.regs "op" := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_smem : st1.smem = tableInsert ws.smem ws.gmem hashLog 0 searchLim (ws.regs "inBase").toNat s (p - s) := by
    rw [← hst1def, ← hsval]; simp [evalCoopWindowGo]
  have step1 : (WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride searchLim
      hashLog 0).eval F ws = st1 := by
    rw [WStmt.eval.eq_14, evalCoopWindow_eq_go, hsval, hwin, hst1def]
  rw [show wseq [.coopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0,
      .uif "found" (wseq
        [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)), .mov "ml" (.imm 4),
          .setp .ge "extC" "ml" (.imm 0),
          .uwhile "extC" (wseq
            [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
              .bin .add "ml" "ml" (.reg "adv"),
              .setp .eq "extC" "adv" (.imm 32) ]),
          .bin .sub "off0" "p0" (.reg "cand0"),
          .bin .sub "litLen" "p0" (.reg "litAnchor"),
          wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
          .bin .add "litAnchor" "p0" (.reg "ml"),
          .mov "searchPos" (.reg "litAnchor") ])
        (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ])]
      = WStmt.seq (.coopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0)
          (.uif "found" (wseq
            [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)), .mov "ml" (.imm 4),
              .setp .ge "extC" "ml" (.imm 0),
              .uwhile "extC" (wseq
                [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
                  .bin .add "ml" "ml" (.reg "adv"),
                  .setp .eq "extC" "adv" (.imm 32) ]),
              .bin .sub "off0" "p0" (.reg "cand0"),
              .bin .sub "litLen" "p0" (.reg "litAnchor"),
              wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
              .bin .add "litAnchor" "p0" (.reg "ml"),
              .mov "searchPos" (.reg "litAnchor") ])
            (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ])) from rfl]
  rw [WStmt.eval.eq_2, step1]
  have hif : (st1.regs "found" == 1) = true := by rw [hst1_found]; rfl
  rw [WStmt.eval.eq_10, if_pos hif]
  simp only [wseq]
  rw [WStmt.eval.eq_2, WStmt.eval.eq_2, WStmt.eval.eq_2, WStmt.eval.eq_2, WStmt.eval.eq_2,
      WStmt.eval.eq_2, WStmt.eval.eq_2, WStmt.eval.eq_2]
  -- `mov ecR`, `mov ec1`, `mov ml 4`, `setp extC ml≥0`: four scratch steps, none of
  -- which touch `p0`/`cand0`/`litAnchor`/`inBase`/`outBase`/`op`/`gmem`/`smem`.
  -- Safe to `simp` fully here — no recursive/heavy subterms in this segment.
  obtain ⟨st2, hst2def⟩ : ∃ st2, ((WStmt.mov "ecR" (WArg.imm endCap)).eval F st1) = st2 := ⟨_, rfl⟩
  obtain ⟨st3, hst3def⟩ : ∃ st3, ((WStmt.mov "ec1" (WArg.imm (endCap - 1))).eval F st2) = st3 := ⟨_, rfl⟩
  obtain ⟨st4, hst4def⟩ : ∃ st4, ((WStmt.mov "ml" (WArg.imm 4)).eval F st3) = st4 := ⟨_, rfl⟩
  obtain ⟨st5, hst5def⟩ : ∃ st5,
      ((WStmt.setp SCmp.ge "extC" "ml" (WArg.imm 0)).eval F st4) = st5 := ⟨_, rfl⟩
  rw [hst2def, hst3def, hst4def, hst5def]
  have hst5_p0 : st5.regs "p0" = UInt64.ofNat p := by
    rw [← hst5def, ← hst4def, ← hst3def, ← hst2def, ← hst1_p0]
    simp [WStmt.eval, WState.setReg, WArg.eval]
  have hst5_cand0 : st5.regs "cand0" = UInt64.ofNat c := by
    rw [← hst5def, ← hst4def, ← hst3def, ← hst2def, ← hst1_cand0]
    simp [WStmt.eval, WState.setReg, WArg.eval]
  have hst5_ml : st5.regs "ml" = UInt64.ofNat 4 := by
    rw [← hst5def, ← hst4def, ← hst3def, ← hst2def]
    simp [WStmt.eval, WState.setReg, WArg.eval]
  have hst5_extC : st5.regs "extC" = 1 := by
    rw [← hst5def, ← hst4def, ← hst3def, ← hst2def]
    simp only [WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
    have h04 : (0:UInt64) ≤ UInt64.ofNat 4 := by decide
    simp [h04]
  have hst5_gmem : st5.gmem = ws.gmem := by
    rw [← hst5def, EmitContent.setp_gmem, ← hst4def, EmitContent.mov_gmem, ← hst3def,
      EmitContent.mov_gmem, ← hst2def, EmitContent.mov_gmem, hst1_gmem]
  have hst5_la : st5.regs "litAnchor" = ws.regs "litAnchor" := by
    rw [← hst5def, ← hst4def, ← hst3def, ← hst2def, ← hst1_la]
    simp [WStmt.eval, WState.setReg]
  have hst5_ib : st5.regs "inBase" = ws.regs "inBase" := by
    rw [← hst5def, ← hst4def, ← hst3def, ← hst2def, ← hst1_ib]
    simp [WStmt.eval, WState.setReg]
  have hst5_ob : st5.regs "outBase" = ws.regs "outBase" := by
    rw [← hst5def, ← hst4def, ← hst3def, ← hst2def, ← hst1_ob]
    simp [WStmt.eval, WState.setReg]
  have hst5_op : st5.regs "op" = ws.regs "op" := by
    rw [← hst5def, ← hst4def, ← hst3def, ← hst2def, ← hst1_op]
    simp [WStmt.eval, WState.setReg]
  have hst5_smem : st5.smem = st1.smem := by
    rw [← hst5def, noSmem_eval_smem (.setp SCmp.ge "extC" "ml" (WArg.imm 0)) trivial,
      ← hst4def, noSmem_eval_smem (.mov "ml" (WArg.imm 4)) trivial,
      ← hst3def, noSmem_eval_smem (.mov "ec1" (WArg.imm (endCap - 1))) trivial,
      ← hst2def, noSmem_eval_smem (.mov "ecR" (WArg.imm endCap)) trivial]
  -- Continue into the `.seq` after the `uwhile "extC"` sub-loop.
  rw [WStmt.eval.eq_2]
  have hwc := AlgorithmLib.LZ4WarpFind.window_sound (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
    (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) searchLim s p c hwin
  have hpN64 : p < 2 ^ 64 := by
    have : p < searchLim := hwc.2.1
    omega
  have hp0nat : st5.regs "p0" = UInt64.ofNat p := hst5_p0
  have hpToNat : (st5.regs "p0").toNat = p := by
    rw [hp0nat]; exact AlgorithmLib.LZ4Ptx.toNat_ofNat_lt p hpN64
  have hcN64 : c < 2 ^ 64 := by
    have : c < p := hwc.2.2.1
    omega
  have hcToNat : (st5.regs "cand0").toNat = c := by
    rw [hst5_cand0]; exact AlgorithmLib.LZ4Ptx.toNat_ofNat_lt c hcN64
  obtain ⟨st6, hst6def⟩ :
      ∃ st6, (WStmt.uwhile "extC" (WStmt.seq (.coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap)
        (WStmt.seq (.bin .add "ml" "ml" (.reg "adv")) (.setp .eq "extC" "adv" (.imm 32))))).eval F st5
        = st6 := ⟨_, rfl⟩
  rw [hst6def]
  have hendCap : endCap ≤ inStride := by omega
  have hm4 : (4:Nat) ≤ inStride := by omega
  have hpmlN64 : p + ml < 2 ^ 64 := by
    rcases AlgorithmLib.LZ4WarpFind.extendFrom_cap (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c endCap F 4
        with heq | hle
    · rw [show ml = extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c endCap F 4 from rfl, heq]; omega
    · have := hle; omega
  have hmlN64 : ml < 2 ^ 64 := by omega
  have hCfuel' : endCap - ((st5.regs "p0").toNat + 4) ≤ 32 * F := by rw [hpToNat]; exact hCfuel
  have hFfuel' : endCap - ((st5.regs "p0").toNat + 4) ≤ F := by rw [hpToNat]; exact hFfuel
  have hextRun : st6.regs "ml" = UInt64.ofNat ml := by
    rw [← hst6def, show ml = extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c endCap F 4 from rfl,
      ← hpToNat, ← hcToNat]
    exact inner_extend_extendFrom inStride endCap (ws.regs "inBase").toNat ws.gmem
      (st5.regs "p0") (st5.regs "cand0")
      F F 4 st5 hstride hendCap hm4 hCfuel' hFfuel' hst5_gmem (by rw [hst5_ib]) rfl rfl
      hst5_extC hst5_ml
  have hinner := inner_extend inStride endCap (ws.regs "inBase").toNat ws.gmem
    (st5.regs "p0") (st5.regs "cand0")
    hstride hendCap F 4 st5 hm4 hst5_gmem (by rw [hst5_ib]) rfl rfl hst5_extC hst5_ml
  have hst6_frame : ∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" → st6.regs r = st5.regs r := by
    intro r h1 h2 h3; rw [← hst6def]; exact hinner.2.1 r h1 h2 h3
  have hst6_gmem : st6.gmem = st5.gmem := by rw [← hst6def]; exact hinner.2.2.1
  have hst6_smem : st6.smem = st5.smem := by rw [← hst6def]; exact hinner.2.2.2
  have hst6_p0 : st6.regs "p0" = UInt64.ofNat p := by
    rw [hst6_frame "p0" (by decide) (by decide) (by decide), hst5_p0]
  have hst6_cand0 : st6.regs "cand0" = UInt64.ofNat c := by
    rw [hst6_frame "cand0" (by decide) (by decide) (by decide), hst5_cand0]
  have hst6_la : st6.regs "litAnchor" = ws.regs "litAnchor" := by
    rw [hst6_frame "litAnchor" (by decide) (by decide) (by decide), hst5_la]
  have hst6_ib : st6.regs "inBase" = ws.regs "inBase" := by
    rw [hst6_frame "inBase" (by decide) (by decide) (by decide), hst5_ib]
  have hst6_ob : st6.regs "outBase" = ws.regs "outBase" := by
    rw [hst6_frame "outBase" (by decide) (by decide) (by decide), hst5_ob]
  have hst6_op : st6.regs "op" = ws.regs "op" := by
    rw [hst6_frame "op" (by decide) (by decide) (by decide), hst5_op]
  have hst6_gmem' : st6.gmem = ws.gmem := by rw [hst6_gmem, hst5_gmem]
  obtain ⟨st7, hst7def⟩ : ∃ st7,
      (WStmt.bin WOp.sub "off0" "p0" (WArg.reg "cand0")).eval F st6 = st7 := ⟨_, rfl⟩
  obtain ⟨st8, hst8def⟩ : ∃ st8,
      (WStmt.bin WOp.sub "litLen" "p0" (WArg.reg "litAnchor")).eval F st7 = st8 := ⟨_, rfl⟩
  rw [hst7def, hst8def]
  have hst7_off0 : st7.regs "off0" = UInt64.ofNat p - UInt64.ofNat c := by
    rw [← hst7def]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run, hst6_p0, hst6_cand0]
  have hst7_gmem : st7.gmem = st6.gmem := by rw [← hst7def]; exact EmitContent.bin_gmem _ _ _ _ _ _
  have hst7_p0 : st7.regs "p0" = UInt64.ofNat p := by
    rw [← hst7def, EmitContent.bin_reg WOp.sub "off0" "p0" "p0" (WArg.reg "cand0") st6 F (by decide),
      hst6_p0]
  have hst7_la : st7.regs "litAnchor" = ws.regs "litAnchor" := by
    rw [← hst7def,
      EmitContent.bin_reg WOp.sub "off0" "p0" "litAnchor" (WArg.reg "cand0") st6 F (by decide),
      hst6_la]
  have hst7_ib : st7.regs "inBase" = ws.regs "inBase" := by
    rw [← hst7def,
      EmitContent.bin_reg WOp.sub "off0" "p0" "inBase" (WArg.reg "cand0") st6 F (by decide),
      hst6_ib]
  have hst7_ob : st7.regs "outBase" = ws.regs "outBase" := by
    rw [← hst7def,
      EmitContent.bin_reg WOp.sub "off0" "p0" "outBase" (WArg.reg "cand0") st6 F (by decide),
      hst6_ob]
  have hst7_op : st7.regs "op" = ws.regs "op" := by
    rw [← hst7def, EmitContent.bin_reg WOp.sub "off0" "p0" "op" (WArg.reg "cand0") st6 F (by decide),
      hst6_op]
  have hst7_off0N : st7.regs "off0" = UInt64.ofNat (p - c) := by
    rw [hst7_off0, AlgorithmLib.LZ4Ptx.u64_sub_ofNat' p c (by omega) hpN64]
  have haN64 : anchor < 2 ^ 64 := by have := hwc.1; omega
  have hla_eq : ws.regs "litAnchor" = UInt64.ofNat anchor := by
    apply UInt64.toNat_inj.mp
    rw [hlaval, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt anchor haN64]
  have hst8_litLen : st8.regs "litLen" = UInt64.ofNat p - UInt64.ofNat anchor := by
    rw [← hst8def]
    simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run, hst7_p0, hst7_la, hla_eq]
  have hst8_litLenN : st8.regs "litLen" = UInt64.ofNat (p - anchor) := by
    rw [hst8_litLen, AlgorithmLib.LZ4Ptx.u64_sub_ofNat' p anchor (by have := hwc.1; omega) hpN64]
  have hst8_off0 : st8.regs "off0" = UInt64.ofNat (p - c) := by
    rw [← hst8def, EmitContent.bin_reg WOp.sub "litLen" "p0" "off0" (WArg.reg "litAnchor")
      st7 F (by decide), hst7_off0N]
  have hst8_ml : st8.regs "ml" = UInt64.ofNat ml := by
    rw [← hst8def, EmitContent.bin_reg WOp.sub "litLen" "p0" "ml" (WArg.reg "litAnchor")
      st7 F (by decide)]
    rw [← hst7def, EmitContent.bin_reg WOp.sub "off0" "p0" "ml" (WArg.reg "cand0")
      st6 F (by decide)]
    exact hextRun
  have hst8_gmem : st8.gmem = ws.gmem := by
    rw [← hst8def, EmitContent.bin_gmem, ← hst7def, EmitContent.bin_gmem, hst6_gmem']
  have hst8_ob : st8.regs "outBase" = ws.regs "outBase" := by
    rw [← hst8def, EmitContent.bin_reg WOp.sub "litLen" "p0" "outBase" (WArg.reg "litAnchor")
      st7 F (by decide), hst7_ob]
  have hst8_op : st8.regs "op" = ws.regs "op" := by
    rw [← hst8def, EmitContent.bin_reg WOp.sub "litLen" "p0" "op" (WArg.reg "litAnchor")
      st7 F (by decide), hst7_op]
  have hst8_ib : st8.regs "inBase" = ws.regs "inBase" := by
    rw [← hst8def, EmitContent.bin_reg WOp.sub "litLen" "p0" "inBase" (WArg.reg "litAnchor")
      st7 F (by decide), hst7_ib]
  have hst8_la : st8.regs "litAnchor" = ws.regs "litAnchor" := by
    rw [← hst8def, EmitContent.bin_reg WOp.sub "litLen" "p0" "litAnchor" (WArg.reg "litAnchor")
      st7 F (by decide), hst7_la]
  obtain ⟨st9, hst9def⟩ : ∃ st9, (wEmitMatchSeq "litAnchor" "litLen" "off0" "ml").eval F st8 = st9 :=
    ⟨_, rfl⟩
  rw [hst9def]
  have hml4' : 4 ≤ (st8.regs "ml").toNat := by
    rw [hst8_ml, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt ml hmlN64]
    have : extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c endCap F 4
        = extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c endCap F 4 := rfl
    exact extendFrom_le (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c endCap F 4
  have hfuelL' : (st8.regs "litLen").toNat / 255 < F := by
    rw [hst8_litLenN, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (p - anchor) (by omega)]; exact hfuelL
  have hfuelM' : ((st8.regs "ml").toNat - 4) / 255 < F := by
    rw [hst8_ml, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt ml hmlN64]; exact hfuelM
  have hnwFull' : (st8.regs "outBase").toNat + (st8.regs "op").toNat + 1
      + (encNib (st8.regs "litLen").toNat).length + (st8.regs "litLen").toNat + 2
      + (encNib ((st8.regs "ml").toNat - 4)).length < 2 ^ 64 := by
    rw [hst8_ob, hst8_op, hst8_litLenN, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (p - anchor) (by omega),
      hst8_ml, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt ml hmlN64]
    exact hnwFull
  have hnw2' : (st8.regs "inBase").toNat + (st8.regs "litAnchor").toNat
      + (st8.regs "litLen").toNat < 2 ^ 64 := by
    rw [hst8_ib, hst8_la, hla_eq, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt anchor haN64,
      hst8_litLenN, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (p - anchor) (by omega)]
    exact hnw2
  have hsize' : (st8.regs "outBase").toNat + (st8.regs "op").toNat + 1
      + (encNib (st8.regs "litLen").toNat).length + (st8.regs "litLen").toNat ≤ st8.gmem.size := by
    rw [hst8_ob, hst8_op, hst8_litLenN, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (p - anchor) (by omega),
      hst8_gmem]
    exact hsize
  have hdisj' : (st8.regs "outBase").toNat + (st8.regs "op").toNat + 1
      + (encNib (st8.regs "litLen").toNat).length + (st8.regs "litLen").toNat
      ≤ (st8.regs "inBase").toNat + (st8.regs "litAnchor").toNat
    ∨ (st8.regs "inBase").toNat + (st8.regs "litAnchor").toNat + (st8.regs "litLen").toNat
      ≤ (st8.regs "outBase").toNat + (st8.regs "op").toNat := by
    rw [hst8_ob, hst8_op, hst8_litLenN, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (p - anchor) (by omega),
      hst8_ib, hst8_la, hla_eq, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt anchor haN64]
    exact hdisj
  have hst9_gmem0 : st9.gmem = EmitContent.putBytesU st8.gmem (st8.regs "outBase" + st8.regs "op")
      (LZ4.encodeSeq ⟨(List.range (st8.regs "litLen").toNat).map
        (fun i => st8.gmem.getD ((st8.regs "inBase").toNat + (st8.regs "litAnchor").toNat + i) 0),
        (st8.regs "off0").toNat, (st8.regs "ml").toNat⟩) := by
    rw [← hst9def]
    apply EmitContent.eval_wEmitMatchSeq_content "litAnchor" "litLen" "off0" "ml" st8 F
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
      hml4' hfuelL' hfuelM' hnwFull' hnw2' hsize' hdisj'
  have hst9_gmem : st9.gmem = EmitContent.putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
      (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
        (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0), p - c, ml⟩) := by
    rw [hst9_gmem0, hst8_gmem, hst8_ob, hst8_op, hst8_ib, hst8_la, hla_eq,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt anchor haN64, hst8_litLenN,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (p - anchor) (by omega), hst8_off0,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (p - c) (by omega), hst8_ml,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt ml hmlN64]
  have hst8_p0 : st8.regs "p0" = UInt64.ofNat p := by
    rw [← hst8def,
      EmitContent.bin_reg WOp.sub "litLen" "p0" "p0" (WArg.reg "litAnchor") st7 F (by decide),
      hst7_p0]
  have hst9_op : st9.regs "op" = ws.regs "op" + UInt64.ofNat (1 + (encNib (p - anchor)).length
      + (p - anchor) + 2 + (encNib (ml - 4)).length) := by
    rw [← hst9def]
    have := LZ4WarpEvalBytes.eval_wEmitMatchSeq "litAnchor" "litLen" "off0" "ml" st8 F
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) hml4' hfuelL' hfuelM'
    rw [this.1, hst8_op, hst8_litLenN, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (p - anchor) (by omega),
      hst8_ml, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt ml hmlN64]
  have hst9_p0 : st9.regs "p0" = UInt64.ofNat p := by
    rw [← hst9def, wEmitMatchSeq_frame "litAnchor" "litLen" "off0" "ml" "p0" st8 F
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide), hst8_p0]
  have hst9_ob : st9.regs "outBase" = ws.regs "outBase" := by
    rw [← hst9def, wEmitMatchSeq_frame "litAnchor" "litLen" "off0" "ml" "outBase" st8 F
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide), hst8_ob]
  have hst9_ib : st9.regs "inBase" = ws.regs "inBase" := by
    rw [← hst9def, wEmitMatchSeq_frame "litAnchor" "litLen" "off0" "ml" "inBase" st8 F
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide), hst8_ib]
  have hst9_ml : st9.regs "ml" = UInt64.ofNat ml := by
    rw [← hst9def, wEmitMatchSeq_frame "litAnchor" "litLen" "off0" "ml" "ml" st8 F
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide), hst8_ml]
  obtain ⟨st10, hst10def⟩ : ∃ st10,
      (WStmt.bin WOp.add "litAnchor" "p0" (WArg.reg "ml")).eval F st9 = st10 := ⟨_, rfl⟩
  obtain ⟨st11, hst11def⟩ : ∃ st11,
      (WStmt.mov "searchPos" (WArg.reg "litAnchor")).eval F st10 = st11 := ⟨_, rfl⟩
  have hst10_la : st10.regs "litAnchor" = UInt64.ofNat p + UInt64.ofNat ml := by
    rw [← hst10def]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run, hst9_p0, hst9_ml]
  have hst10_laN : (st10.regs "litAnchor").toNat = p + ml := by
    rw [hst10_la, AlgorithmLib.LZ4Ptx.u64_add_ofNat p ml hpmlN64]
  have hst10_gmem : st10.gmem = st9.gmem := by rw [← hst10def]; exact EmitContent.bin_gmem _ _ _ _ _ _
  have hst10_op : st10.regs "op" = st9.regs "op" := by
    rw [← hst10def, EmitContent.bin_reg WOp.add "litAnchor" "p0" "op" (WArg.reg "ml") st9 F (by decide)]
  have hst10_ob : st10.regs "outBase" = st9.regs "outBase" := by
    rw [← hst10def,
      EmitContent.bin_reg WOp.add "litAnchor" "p0" "outBase" (WArg.reg "ml") st9 F (by decide)]
  have hst10_ib : st10.regs "inBase" = st9.regs "inBase" := by
    rw [← hst10def,
      EmitContent.bin_reg WOp.add "litAnchor" "p0" "inBase" (WArg.reg "ml") st9 F (by decide)]
  have hst11_sp : st11.regs "searchPos" = st10.regs "litAnchor" := by
    rw [← hst11def]; simp [WStmt.eval, WState.setReg, WArg.eval]
  have hst11_la : st11.regs "litAnchor" = st10.regs "litAnchor" := by
    rw [← hst11def]; simp [WStmt.eval, WState.setReg, WArg.eval]
  have hst11_gmem : st11.gmem = st10.gmem := by rw [← hst11def]; exact EmitContent.mov_gmem _ _ _ _
  have hst11_op : st11.regs "op" = st10.regs "op" := by
    rw [← hst11def]; exact EmitContent.mov_reg "searchPos" "op" (WArg.reg "litAnchor") st10 F (by decide)
  have hst11_ob : st11.regs "outBase" = st10.regs "outBase" := by
    rw [← hst11def]
    exact EmitContent.mov_reg "searchPos" "outBase" (WArg.reg "litAnchor") st10 F (by decide)
  have hst11_ib : st11.regs "inBase" = st10.regs "inBase" := by
    rw [← hst11def]
    exact EmitContent.mov_reg "searchPos" "inBase" (WArg.reg "litAnchor") st10 F (by decide)
  -- smem is preserved from `st5` (= `st1`, the table-inserted state) onward: none of
  -- `st6..st11`'s statements write `smem` (`.coopCopy`/`.bin`/`wEmitMatchSeq`/`.mov`).
  have hst7_smem : st7.smem = st6.smem := by
    rw [← hst7def]; exact noSmem_eval_smem (.bin WOp.sub "off0" "p0" (WArg.reg "cand0")) trivial F st6
  have hst8_smem : st8.smem = st7.smem := by
    rw [← hst8def]; exact noSmem_eval_smem (.bin WOp.sub "litLen" "p0" (WArg.reg "litAnchor")) trivial F st7
  have hEmitNoSmem : NoSmem (wEmitMatchSeq "litAnchor" "litLen" "off0" "ml") := by
    simp only [wEmitMatchSeq, wStoreByte, wEmitToken, wEmitLSIC, wseq, NoSmem]
    repeat' apply And.intro
    all_goals trivial
  have hst9_smem : st9.smem = st8.smem := by
    rw [← hst9def]; exact noSmem_eval_smem _ hEmitNoSmem F st8
  have hst10_smem : st10.smem = st9.smem := by
    rw [← hst10def]; exact noSmem_eval_smem (.bin WOp.add "litAnchor" "p0" (WArg.reg "ml")) trivial F st9
  have hst11_smem : st11.smem = st10.smem := by
    rw [← hst11def]; exact noSmem_eval_smem (.mov "searchPos" (WArg.reg "litAnchor")) trivial F st10
  rw [hst10def, hst11def]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hst11_gmem, hst10_gmem, hst9_gmem]
  · rw [hst11_sp, hst10_laN]
  · rw [hst11_la, hst10_laN]
  · rw [hst11_ob, hst10_ob, hst9_ob]
  · rw [hst11_ib, hst10_ib, hst9_ib]
  · rw [hst11_op, hst10_op, hst9_op]
  · rw [hst11_smem, hst10_smem, hst9_smem, hst8_smem, hst7_smem, hst6_smem, hst5_smem, hst1_smem]

-- ── The `not found` branch of one `uwhile "loopC"` iteration: window miss. ───
-- After `coopWindow` (found=0), no emit call; `searchPos += 32`.  gmem untouched.
theorem emitStepNotFound_eq (inStride searchLim hashLog s : Nat) (F : Nat) (ws : WState)
    (hwin : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) searchLim s = none)
    (hsval : (ws.regs "searchPos").toNat = s) (hs64 : s + 32 < 2 ^ 64) :
    let endCap := inStride - 5
    let st' := (wseq [ .coopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0,
        .uif "found" (wseq
          [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)), .mov "ml" (.imm 4),
            .setp .ge "extC" "ml" (.imm 0),
            .uwhile "extC" (wseq
              [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
                .bin .add "ml" "ml" (.reg "adv"),
                .setp .eq "extC" "adv" (.imm 32) ]),
            .bin .sub "off0" "p0" (.reg "cand0"),
            .bin .sub "litLen" "p0" (.reg "litAnchor"),
            wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
            .bin .add "litAnchor" "p0" (.reg "ml"),
            .mov "searchPos" (.reg "litAnchor") ])
          (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]) ]).eval F ws
    st'.gmem = ws.gmem ∧ (st'.regs "searchPos").toNat = s + 32
      ∧ st'.regs "litAnchor" = ws.regs "litAnchor" ∧ st'.regs "outBase" = ws.regs "outBase"
      ∧ st'.regs "inBase" = ws.regs "inBase" ∧ st'.regs "op" = ws.regs "op"
      ∧ st'.smem = tableInsert ws.smem ws.gmem hashLog 0 searchLim (ws.regs "inBase").toNat s 32 := by
  intro endCap st'
  show st'.gmem = _ ∧ (st'.regs "searchPos").toNat = _ ∧ st'.regs "litAnchor" = _
    ∧ st'.regs "outBase" = _ ∧ st'.regs "inBase" = _ ∧ st'.regs "op" = _ ∧ st'.smem = _
  subst st'
  obtain ⟨st1, hst1def⟩ :
      ∃ st1, evalCoopWindowGo "found" "p0" "cand0" "searchPos" hashLog 0 searchLim (ws.regs "inBase").toNat
        none ws = st1 := ⟨_, rfl⟩
  have hst1_found : st1.regs "found" = 0 := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_gmem : st1.gmem = ws.gmem := by rw [← hst1def]; rfl
  have hst1_la : st1.regs "litAnchor" = ws.regs "litAnchor" := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_ob : st1.regs "outBase" = ws.regs "outBase" := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_ib : st1.regs "inBase" = ws.regs "inBase" := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_op : st1.regs "op" = ws.regs "op" := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_sp : st1.regs "searchPos" = ws.regs "searchPos" := by
    rw [← hst1def]; simp [evalCoopWindowGo, WState.setReg]
  have hst1_smem : st1.smem = tableInsert ws.smem ws.gmem hashLog 0 searchLim (ws.regs "inBase").toNat s 32 := by
    rw [← hst1def, ← hsval]; simp [evalCoopWindowGo]
  have step1 : (WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride searchLim
      hashLog 0).eval F ws = st1 := by
    rw [WStmt.eval.eq_14, evalCoopWindow_eq_go, hsval, hwin, hst1def]
  simp only [wseq]
  rw [WStmt.eval.eq_2, step1]
  have hif : (st1.regs "found" == 1) = false := by rw [hst1_found]; rfl
  rw [WStmt.eval.eq_10, if_neg (by rw [hif]; decide)]
  obtain ⟨st2, hst2def⟩ :
      ∃ st2, (WStmt.bin WOp.add "searchPos" "searchPos" (WArg.imm 32)).eval F st1 = st2 := ⟨_, rfl⟩
  rw [hst2def]
  have hst2_gmem : st2.gmem = st1.gmem := by rw [← hst2def]; exact EmitContent.bin_gmem _ _ _ _ _ _
  have hst2_sp : st2.regs "searchPos" = st1.regs "searchPos" + UInt64.ofNat 32 := by
    rw [← hst2def]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
  have hst2_spN : (st2.regs "searchPos").toNat = s + 32 := by
    rw [hst2_sp, hst1_sp, UInt64.toNat_add, hsval,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt 32 (by omega), Nat.mod_eq_of_lt hs64]
  have hst2_la : st2.regs "litAnchor" = st1.regs "litAnchor" := by
    rw [← hst2def, EmitContent.bin_reg WOp.add "searchPos" "searchPos" "litAnchor" (WArg.imm 32)
      st1 F (by decide)]
  have hst2_ob : st2.regs "outBase" = st1.regs "outBase" := by
    rw [← hst2def, EmitContent.bin_reg WOp.add "searchPos" "searchPos" "outBase" (WArg.imm 32)
      st1 F (by decide)]
  have hst2_ib : st2.regs "inBase" = st1.regs "inBase" := by
    rw [← hst2def, EmitContent.bin_reg WOp.add "searchPos" "searchPos" "inBase" (WArg.imm 32)
      st1 F (by decide)]
  have hst2_op : st2.regs "op" = st1.regs "op" := by
    rw [← hst2def, EmitContent.bin_reg WOp.add "searchPos" "searchPos" "op" (WArg.imm 32)
      st1 F (by decide)]
  have hst2_smem : st2.smem = st1.smem := by
    rw [← hst2def]; exact noSmem_eval_smem (.bin WOp.add "searchPos" "searchPos" (WArg.imm 32))
      trivial F st1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hst2_gmem, hst1_gmem]
  · exact hst2_spN
  · rw [hst2_la, hst1_la]
  · rw [hst2_ob, hst1_ob]
  · rw [hst2_ib, hst1_ib]
  · rw [hst2_op, hst1_op]
  · rw [hst2_smem, hst1_smem]

-- ── `genLoop` is fuel-irrelevant once fuel ≥ the remaining span `searchLim - s`. ──
-- (Each step strictly increases `s`; found: `s→p+ml` with `p≥s, ml≥4`; not-found:
-- `s→s+32`.  So excess fuel is never consumed.)  Strong induction on the span.
theorem genLoop_fuel_irrel (inp : List UInt8) (mkOracle : Array UInt8 → (Nat → Option Nat))
    (upd : Array UInt8 → Nat → Nat → Array UInt8) (searchLim endCap : Nat) :
    ∀ (span : Nat), ∀ (f1 f2 : Nat) (smem : Array UInt8) (anchor s : Nat),
      searchLim - s ≤ span → searchLim - s ≤ f1 → searchLim - s ≤ f2 →
      genLoop inp mkOracle upd searchLim endCap f1 smem anchor s
        = genLoop inp mkOracle upd searchLim endCap f2 smem anchor s := by
  intro span
  induction span with
  | zero =>
      intro f1 f2 smem anchor s hspan _ _
      have hs : ¬ s < searchLim := by omega
      cases f1 <;> cases f2 <;> simp only [genLoop, if_neg hs]
  | succ n ih =>
      intro f1 f2 smem anchor s hspan hf1 hf2
      by_cases hs : s < searchLim
      · obtain ⟨g1, hg1⟩ : ∃ g1, f1 = g1 + 1 := by
          cases f1 with | zero => omega | succ k => exact ⟨k, rfl⟩
        obtain ⟨g2, hg2⟩ : ∃ g2, f2 = g2 + 1 := by
          cases f2 with | zero => omega | succ k => exact ⟨k, rfl⟩
        subst hg1 hg2
        unfold genLoop
        rw [if_pos hs, if_pos hs]
        cases hw : AlgorithmLib.LZ4WarpFind.window inp (mkOracle smem) searchLim s with
        | none =>
            dsimp only
            exact ih g1 g2 (upd smem s 32) anchor (s + 32) (by omega) (by omega) (by omega)
        | some pc =>
            obtain ⟨p, c⟩ := pc
            have hsp := AlgorithmLib.LZ4WarpFind.window_sound inp (mkOracle smem) searchLim s p c hw
            have hml4 := AlgorithmLib.LZ4WarpFind.extendFrom_le inp p c endCap (endCap - (p + 4)) 4
            dsimp only
            rw [ih g1 g2 (upd smem s (p - s))
              (p + AlgorithmLib.LZ4WarpFind.extendFrom inp p c endCap (endCap - (p + 4)) 4)
              (p + AlgorithmLib.LZ4WarpFind.extendFrom inp p c endCap (endCap - (p + 4)) 4)
              (by have := hsp.1; omega) (by have := hsp.1; omega) (by have := hsp.1; omega)]
      · cases f1 <;> cases f2 <;> simp only [genLoop, if_neg hs]

-- ── The trailing `.setp .lt "loopC" "searchPos" (.imm searchLim)` guard ──────
-- recompute at the end of each `uwhile "loopC"` iteration: sets `loopC` from
-- the new `searchPos`, preserves everything else (frame via `noWrite_eval_reg`
-- + `noSmem_eval_smem`, since `.setp` only ever writes its own `d`).
theorem loopC_recompute (searchLim : Nat) (F : Nat) (st : WState) :
    ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm searchLim)).eval F st).regs "loopC"
        = (if st.regs "searchPos" < UInt64.ofNat searchLim then 1 else 0)
      ∧ (∀ r, r ≠ "loopC" →
          ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm searchLim)).eval F st).regs r
            = st.regs r)
      ∧ ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm searchLim)).eval F st).gmem = st.gmem
      ∧ ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm searchLim)).eval F st).smem = st.smem := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  · intro r hr; exact EmitContent.setp_reg _ _ _ _ _ st F hr
  · exact EmitContent.setp_gmem _ _ _ _ st F
  · exact noSmem_eval_smem (.setp SCmp.lt "loopC" "searchPos" (WArg.imm searchLim)) trivial F st

-- ── Abstract (kernel-cheap) facts about `genLoop`/`planBlockFrom` at the ─────
-- loop's two terminal shapes — split out from `emitLoop_eq` to keep every
-- proof term the kernel checks small (same lesson as `CoopWindowLeaf.lean`'s
-- kernel-deep-recursion fix: never build these inside a huge-context theorem).
theorem genLoop_notFound_empty (inp : List UInt8) (mkOracle : Array UInt8 → (Nat → Option Nat))
    (upd : Array UInt8 → Nat → Nat → Array UInt8) (searchLim endCap fuel : Nat)
    (smem : Array UInt8) (anchor s : Nat) (hs : ¬ s < searchLim) :
    (genLoop inp mkOracle upd searchLim endCap (fuel + 1) smem anchor s).steps = [] := by
  unfold genLoop
  rw [if_neg hs]

-- ── window-miss WHILE still inside `searchLim`: `genLoop` advances `s+=32`, ──
-- inserts into the table (`upd smem s 32`), and recurses with the SAME anchor.
-- No `PlanStep` is prepended, so `.steps`/`.finalLen` are exactly the recursion's.
theorem genLoop_notFound_step (inp : List UInt8) (mkOracle : Array UInt8 → (Nat → Option Nat))
    (upd : Array UInt8 → Nat → Nat → Array UInt8) (searchLim endCap fuel : Nat)
    (smem : Array UInt8) (anchor s : Nat) (rest : Plan) (hs : s < searchLim)
    (hw : AlgorithmLib.LZ4WarpFind.window inp (mkOracle smem) searchLim s = none)
    (hrest : rest = genLoop inp mkOracle upd searchLim endCap fuel (upd smem s 32) anchor (s + 32)) :
    genLoop inp mkOracle upd searchLim endCap (fuel + 1) smem anchor s = rest := by
  unfold genLoop
  rw [if_pos hs, hw, ← hrest]

-- ── `encNib`'s extension bytes never exceed `n+1` in length (`ext n` has ────
-- length `n/255+1`, and `encNib n ≤ ext n` in the saturating case) — needed
-- to bound the emitted byte length against the layout invariant's budget.
theorem encNib_length_le (n : Nat) : (encNib n).length ≤ n + 1 := by
  unfold encNib
  split
  · simp
  · unfold ext
    simp only [List.length_append, List.length_replicate, List.length_cons, List.length_nil]
    omega

-- ── `extendFrom` saturates: past the room `endCap - (p+ml)`, extra fuel is inert. ──
-- Bridges the fuel mismatch between `genLoop`'s found branch (fuel `endCap-(p+4)`)
-- and `emitStepFound_eq`/`bodyFound_eq` (loop fuel `n`, with `endCap-(p+4) ≤ n`).
theorem extendFrom_fuel_sat (inp : List UInt8) (p c endCap : Nat) :
    ∀ (fuel ml : Nat), endCap - (p + ml) ≤ fuel →
      AlgorithmLib.LZ4WarpFind.extendFrom inp p c endCap fuel ml
        = AlgorithmLib.LZ4WarpFind.extendFrom inp p c endCap (endCap - (p + ml)) ml := by
  intro fuel
  induction fuel with
  | zero => intro ml h; simp only [Nat.le_zero] at h; rw [h]
  | succ n ih =>
      intro ml h
      rw [AlgorithmLib.LZ4WarpFind.extendFrom.eq_2]
      by_cases hcond : p + ml < endCap ∧ AlgorithmLib.LZ4WarpFind.byte inp (p + ml)
          = AlgorithmLib.LZ4WarpFind.byte inp (c + ml)
      · rw [if_pos hcond]
        have hroom : endCap - (p + (ml + 1)) ≤ n := by omega
        rw [ih (ml + 1) hroom]
        have hpos : endCap - (p + ml) = (endCap - (p + (ml + 1))).succ := by omega
        rw [hpos, AlgorithmLib.LZ4WarpFind.extendFrom.eq_2, if_pos hcond]
      · rw [if_neg hcond]
        by_cases hlt : p + ml < endCap
        · have hpos : endCap - (p + ml) = (endCap - (p + (ml + 1))).succ := by omega
          rw [hpos, AlgorithmLib.LZ4WarpFind.extendFrom.eq_2, if_neg hcond]
        · have : endCap - (p + ml) = 0 := by omega
          rw [this, AlgorithmLib.LZ4WarpFind.extendFrom.eq_1]

-- ── Bridge: the input-slice `(gmemInp g len).drop a |>.take k` (as `planBlockFrom` ──
-- reads literals) equals the explicit byte-map `emitStepFound_eq` produces, given
-- the slice stays in-bounds (`a + k ≤ len`).  Both index `g` from offset 0.
theorem gmemInpAt_slice_eq (g : Array UInt8) (ib len a k : Nat) (h : a + k ≤ len) :
    ((gmemInpAt g ib len).drop a).take k = (List.range k).map (fun i => g.getD (ib + a + i) 0) := by
  apply List.ext_getElem
  · simp [gmemInpAt]; omega
  · intro i h1 h2
    simp only [List.getElem_take, List.getElem_drop, gmemInpAt, List.getElem_map,
      List.getElem_range, Nat.add_assoc]

theorem gmemInp_slice_eq (g : Array UInt8) (len a k : Nat) (h : a + k ≤ len) :
    ((gmemInp g len).drop a).take k = (List.range k).map (fun i => g.getD (a + i) 0) := by
  apply List.ext_getElem
  · simp [gmemInp]; omega
  · intro i h1 h2
    simp only [List.getElem_take, List.getElem_drop, gmemInp, List.getElem_map,
      List.getElem_range]

-- ── The hash table reads `g` only through `wLoad4` at `[pos, pos+3]`; a ──────
-- `putBytesU` write to a disjoint region above `pos+3` leaves it invariant.
theorem wLoad4_putBytesU_lt (g : Array UInt8) (base : UInt64) (xs : List UInt8) (pos : Nat)
    (hnw : base.toNat + xs.length < 2 ^ 64) (hpos : pos + 3 < base.toNat) :
    wLoad4 (EmitContent.putBytesU g base xs) pos = wLoad4 g pos := by
  unfold wLoad4
  rw [EmitContent.putBytesU_getD_lt pos xs g base hnw (by omega),
    EmitContent.putBytesU_getD_lt (pos + 1) xs g base hnw (by omega),
    EmitContent.putBytesU_getD_lt (pos + 2) xs g base hnw (by omega),
    EmitContent.putBytesU_getD_lt (pos + 3) xs g base hnw (by omega)]

-- `tableInsert` reads `g₁`/`g₂` only via `wHash` at guarded positions `sp+k <
-- searchLim`; if the two arrays' hashes agree there, the whole insert coincides.
-- (Generic over the two arrays — `putBytesU` invariance is the special case below.)
-- Generic: two step functions agreeing on every list element give equal `foldl`.
-- (Kept opaque in the big step lambda — never unfolds `wHash`, so no deep recursion.)
theorem foldl_step_congr {α β : Type} (f₁ f₂ : β → α → β) :
    ∀ (l : List α) (acc : β), (∀ b a, a ∈ l → f₁ b a = f₂ b a) →
      l.foldl f₁ acc = l.foldl f₂ acc := by
  intro l
  induction l with
  | nil => intro acc _; rfl
  | cons hd tl ih =>
      intro acc h
      simp only [List.foldl_cons]
      rw [h acc hd (by simp)]
      exact ih _ (fun b a ha => h b a (by simp [ha]))

theorem tableInsert_hash_congr (g₁ g₂ sm : Array UInt8) (hashLog searchLim ib sp upto : Nat)
    (hagree : ∀ pos, pos < ib + searchLim → wHash g₁ hashLog pos = wHash g₂ hashLog pos) :
    tableInsert sm g₁ hashLog 0 searchLim ib sp upto
      = tableInsert sm g₂ hashLog 0 searchLim ib sp upto := by
  unfold tableInsert
  apply foldl_step_congr
  intro acc k _
  by_cases hc : k ≤ upto ∧ sp + k < searchLim
  · rw [if_pos hc, if_pos hc, hagree (ib + (sp + k)) (by omega)]
  · rw [if_neg hc, if_neg hc]

theorem tableInsert_putBytesU_eq (g : Array UInt8) (base : UInt64) (xs : List UInt8)
    (hashLog searchLim ib : Nat) (hnw : base.toNat + xs.length < 2 ^ 64)
    (hlim : ib + searchLim + 3 ≤ base.toNat) (sm : Array UInt8) (sp upto : Nat) :
    tableInsert sm (EmitContent.putBytesU g base xs) hashLog 0 searchLim ib sp upto
      = tableInsert sm g hashLog 0 searchLim ib sp upto :=
  tableInsert_hash_congr _ _ sm hashLog searchLim ib sp upto (fun pos hpos => by
    unfold wHash; rw [wLoad4_putBytesU_lt g base xs pos hnw (by omega)])

-- `probe` calls `oracle p` then discards it unless `p < searchLim`; so oracles
-- agreeing below `searchLim` give identical probes at every `p`.
theorem probe_oracle_congr (inp : List UInt8) (o1 o2 : Nat → Option Nat) (searchLim p : Nat)
    (hagree : p < searchLim → o1 p = o2 p) :
    AlgorithmLib.LZ4WarpFind.probe inp o1 searchLim p
      = AlgorithmLib.LZ4WarpFind.probe inp o2 searchLim p := by
  unfold AlgorithmLib.LZ4WarpFind.probe
  by_cases hp : p < searchLim
  · rw [hagree hp]
  · cases h1 : o1 p with
    | none =>
        cases h2 : o2 p with
        | none => rfl
        | some c2 => simp only [hp, false_and, if_false]
    | some c1 =>
        cases h2 : o2 p with
        | none => simp only [hp, false_and, if_false]
        | some c2 => simp only [hp, false_and, if_false]

theorem windowGo_oracle_congr (inp : List UInt8) (o1 o2 : Nat → Option Nat) (searchLim s : Nat)
    (hagree : ∀ p, p < searchLim → o1 p = o2 p) :
    ∀ k lane, AlgorithmLib.LZ4WarpFind.windowGo inp o1 searchLim s k lane
      = AlgorithmLib.LZ4WarpFind.windowGo inp o2 searchLim s k lane := by
  intro k
  induction k with
  | zero => intro lane; rfl
  | succ n ih =>
      intro lane
      unfold AlgorithmLib.LZ4WarpFind.windowGo
      rw [probe_oracle_congr inp o1 o2 searchLim (s + lane) (fun h => hagree (s + lane) h)]
      cases AlgorithmLib.LZ4WarpFind.probe inp o2 searchLim (s + lane) with
      | none => rw [ih (lane + 1)]
      | some c => rfl

theorem window_oracle_congr (inp : List UInt8) (o1 o2 : Nat → Option Nat) (searchLim s : Nat)
    (hagree : ∀ p, p < searchLim → o1 p = o2 p) :
    AlgorithmLib.LZ4WarpFind.window inp o1 searchLim s
      = AlgorithmLib.LZ4WarpFind.window inp o2 searchLim s :=
  windowGo_oracle_congr inp o1 o2 searchLim s hagree 32 0

-- `genLoop`'s result depends on `mkOracle` only through `window … s` at `s < searchLim`
-- and on `upd`; if both oracles agree below `searchLim` (for every threaded table)
-- and the updates are equal, the whole recursion coincides.
theorem genLoop_oracle_congr (inp : List UInt8)
    (mkO1 mkO2 : Array UInt8 → (Nat → Option Nat))
    (upd : Array UInt8 → Nat → Nat → Array UInt8)
    (searchLim endCap : Nat)
    (hO : ∀ sm pos, pos < searchLim → mkO1 sm pos = mkO2 sm pos) :
    ∀ (fuel : Nat) (smem : Array UInt8) (anchor s : Nat),
      genLoop inp mkO1 upd searchLim endCap fuel smem anchor s
        = genLoop inp mkO2 upd searchLim endCap fuel smem anchor s := by
  intro fuel
  induction fuel with
  | zero => intro smem anchor s; rfl
  | succ n ih =>
      intro smem anchor s
      unfold genLoop
      by_cases hs : s < searchLim
      · rw [if_pos hs, if_pos hs,
          window_oracle_congr inp (mkO1 smem) (mkO2 smem) searchLim s
            (fun pos hpos => hO smem pos hpos)]
        cases hw : AlgorithmLib.LZ4WarpFind.window inp (mkO2 smem) searchLim s with
        | none => dsimp only; rw [ih (upd smem s 32) anchor (s + 32)]
        | some pc =>
            obtain ⟨p, c⟩ := pc
            dsimp only
            rw [ih (upd smem s (p - s)) (p + AlgorithmLib.LZ4WarpFind.extendFrom inp p c endCap
              (endCap - (p + 4)) 4) (p + AlgorithmLib.LZ4WarpFind.extendFrom inp p c endCap
              (endCap - (p + 4)) 4)]
      · rw [if_neg hs, if_neg hs]

-- ── Abstract `genLoop`/`planBlockFrom` unfold at a window hit — the found ────
-- case's telescoping fact, kept as its own kernel-cheap top-level lemma.
theorem genLoop_found_step (inp : List UInt8) (mkOracle : Array UInt8 → (Nat → Option Nat))
    (upd : Array UInt8 → Nat → Nat → Array UInt8) (searchLim endCap fuel : Nat)
    (smem : Array UInt8) (anchor s p c ml : Nat) (rest : Plan) (hs : s < searchLim)
    (hw : AlgorithmLib.LZ4WarpFind.window inp (mkOracle smem) searchLim s = some (p, c))
    (hml : ml = AlgorithmLib.LZ4WarpFind.extendFrom inp p c endCap (endCap - (p + 4)) 4)
    (hrest : rest = genLoop inp mkOracle upd searchLim endCap fuel (upd smem s (p - s))
        (p + ml) (p + ml)) :
    (genLoop inp mkOracle upd searchLim endCap (fuel + 1) smem anchor s).steps
      = ⟨p - anchor, p - c, ml⟩ :: rest.steps := by
  unfold genLoop
  rw [if_pos hs, hw]
  dsimp only
  rw [← hml, ← hrest]

-- Companion to `genLoop_found_step` for `.finalLen`: the found step keeps the
-- recursion's `finalLen` verbatim (`⟨step :: rest.steps, rest.finalLen⟩`).
theorem genLoop_found_finalLen (inp : List UInt8) (mkOracle : Array UInt8 → (Nat → Option Nat))
    (upd : Array UInt8 → Nat → Nat → Array UInt8) (searchLim endCap fuel : Nat)
    (smem : Array UInt8) (anchor s p c ml : Nat) (rest : Plan) (hs : s < searchLim)
    (hw : AlgorithmLib.LZ4WarpFind.window inp (mkOracle smem) searchLim s = some (p, c))
    (hml : ml = AlgorithmLib.LZ4WarpFind.extendFrom inp p c endCap (endCap - (p + 4)) 4)
    (hrest : rest = genLoop inp mkOracle upd searchLim endCap fuel (upd smem s (p - s))
        (p + ml) (p + ml)) :
    (genLoop inp mkOracle upd searchLim endCap (fuel + 1) smem anchor s).finalLen
      = rest.finalLen := by
  unfold genLoop
  rw [if_pos hs, hw]
  dsimp only
  rw [← hml, ← hrest]

theorem planBlockFrom_found_step (inp : List UInt8) (anchor p c ml : Nat) (rest : List PlanStep)
    (fl : Nat) :
    (planBlockFrom inp anchor (⟨p - anchor, p - c, ml⟩ :: rest) fl).seqs.flatMap LZ4.encodeSeq
      = LZ4.encodeSeq ⟨(inp.drop anchor).take (p - anchor), p - c, ml⟩
        ++ (planBlockFrom inp (anchor + (p - anchor) + ml) rest fl).seqs.flatMap LZ4.encodeSeq := by
  show (⟨(inp.drop anchor).take (p - anchor), p - c, ml⟩ ::
      (planBlockFrom inp (anchor + (p - anchor) + ml) rest fl).seqs).flatMap LZ4.encodeSeq = _
  rfl

-- ── The full loop `body` (`coopWindow ; uif ; setp loopC`) in the window-miss ──
-- case: composes `emitStepNotFound_eq` (the `coopWindow ; uif` prefix) with the
-- trailing `.setp .lt "loopC" ...` recompute (`loopC_recompute`).  Produces the
-- full post-`body` state's key facts, ready to feed the induction hypothesis.
theorem bodyNotFound_eq (inStride hashLog s : Nat) (F : Nat) (ws : WState)
    (hwin : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) s = none)
    (hsval : (ws.regs "searchPos").toNat = s) (hs64 : s + 32 < 2 ^ 64) :
    let body := wseq
      [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
        .uif "found" (wseq
          [ .mov "ecR" (.imm (inStride - 5)), .mov "ec1" (.imm (inStride - 5 - 1)),
            .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
            .uwhile "extC" (wseq
              [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride (inStride - 5),
                .bin .add "ml" "ml" (.reg "adv"),
                .setp .eq "extC" "adv" (.imm 32) ]),
            .bin .sub "off0" "p0" (.reg "cand0"),
            .bin .sub "litLen" "p0" (.reg "litAnchor"),
            wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
            .bin .add "litAnchor" "p0" (.reg "ml"),
            .mov "searchPos" (.reg "litAnchor") ])
          (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]),
        .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ]
    let st' := body.eval F ws
    st'.gmem = ws.gmem ∧ (st'.regs "searchPos").toNat = s + 32
      ∧ st'.regs "litAnchor" = ws.regs "litAnchor" ∧ st'.regs "outBase" = ws.regs "outBase"
      ∧ st'.regs "inBase" = ws.regs "inBase" ∧ st'.regs "op" = ws.regs "op"
      ∧ st'.smem = tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s 32
      ∧ st'.regs "loopC" = (if UInt64.ofNat (s + 32) < UInt64.ofNat (inStride - 12) then 1 else 0) := by
  intro body st'
  -- `body = .seq coopWindow (.seq uif setp)`; peel the leading `coopWindow ; uif`
  -- prefix via `emitStepNotFound_eq`, then the trailing `.setp` via `loopC_recompute`.
  obtain ⟨stP, hstPdef⟩ : ∃ stP,
      (wseq [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
        .uif "found" (wseq
          [ .mov "ecR" (.imm (inStride - 5)), .mov "ec1" (.imm (inStride - 5 - 1)),
            .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
            .uwhile "extC" (wseq
              [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride (inStride - 5),
                .bin .add "ml" "ml" (.reg "adv"),
                .setp .eq "extC" "adv" (.imm 32) ]),
            .bin .sub "off0" "p0" (.reg "cand0"),
            .bin .sub "litLen" "p0" (.reg "litAnchor"),
            wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
            .bin .add "litAnchor" "p0" (.reg "ml"),
            .mov "searchPos" (.reg "litAnchor") ])
          (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]) ]).eval F ws = stP := ⟨_, rfl⟩
  obtain ⟨hP_gmem, hP_sp, hP_la, hP_ob, hP_ib, hP_op, hP_smem⟩ :=
    emitStepNotFound_eq inStride (inStride - 12) hashLog s F ws hwin hsval hs64
  rw [hstPdef] at hP_gmem hP_sp hP_la hP_ob hP_ib hP_op hP_smem
  -- `st' = (setp loopC).eval F stP` after unfolding the two nested `.seq`s.
  have hbody : st' = (WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F stP := by
    show (WStmt.eval F body ws) = _
    rw [← hstPdef]
    simp only [body, wseq, WStmt.eval.eq_2]
  obtain ⟨hR_loopC, hR_reg, hR_gmem, hR_smem⟩ := loopC_recompute (inStride - 12) F stP
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hbody, hR_gmem, hP_gmem]
  · rw [hbody, hR_reg "searchPos" (by decide), hP_sp]
  · rw [hbody, hR_reg "litAnchor" (by decide), hP_la]
  · rw [hbody, hR_reg "outBase" (by decide), hP_ob]
  · rw [hbody, hR_reg "inBase" (by decide), hP_ib]
  · rw [hbody, hR_reg "op" (by decide), hP_op]
  · rw [hbody, hR_smem, hP_smem]
  · rw [hbody, hR_loopC]
    have : stP.regs "searchPos" = UInt64.ofNat (s + 32) := by
      have h1 : (stP.regs "searchPos").toNat = s + 32 := hP_sp
      rw [← h1, UInt64.ofNat_toNat]
    rw [this]

-- ── The full loop `body` in the window-HIT case: composes `emitStepFound_eq` ──
-- (the `coopWindow ; uif` emit prefix) with the trailing `.setp .lt "loopC"` guard
-- recompute.  Gives the post-`body` state's key facts incl. the byte content and
-- the new `loopC` guard — ready to feed the induction hypothesis at `(p+ml,p+ml)`.
theorem bodyFound_eq (inStride hashLog s p c : Nat) (F : Nat) (ws : WState) (anchor : Nat)
    (hstride : inStride ≤ 65536)
    (hwin : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) s = some (p, c))
    (has : anchor ≤ s) (hsval : (ws.regs "searchPos").toNat = s)
    (hlaval : (ws.regs "litAnchor").toNat = anchor)
    (hCfuel : (inStride - 5) - (p + 4) ≤ 32 * F) (hFfuel : (inStride - 5) - (p + 4) ≤ F)
    (hfuelL : (p - anchor) / 255 < F) (hfuelM : (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
        (inStride - 5) F 4 - 4) / 255 < F)
    (hp64 : inStride < 2 ^ 64)
    (hnwFull : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
        + (encNib (p - anchor)).length + (p - anchor) + 2
        + (encNib (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) F 4 - 4)).length
        < 2 ^ 64)
    (hnw2 : (ws.regs "inBase").toNat + anchor + (p - anchor) < 2 ^ 64)
    (hsize : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
        + (encNib (p - anchor)).length + (p - anchor) ≤ ws.gmem.size)
    (hdisj : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
        + (encNib (p - anchor)).length + (p - anchor) ≤ (ws.regs "inBase").toNat + anchor
      ∨ (ws.regs "inBase").toNat + anchor + (p - anchor) ≤ (ws.regs "outBase").toNat + (ws.regs "op").toNat)
    (hpml64 : p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) F 4 < 2 ^ 64) :
    let ml := extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) F 4
    let body := wseq
      [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
        .uif "found" (wseq
          [ .mov "ecR" (.imm (inStride - 5)), .mov "ec1" (.imm (inStride - 5 - 1)),
            .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
            .uwhile "extC" (wseq
              [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride (inStride - 5),
                .bin .add "ml" "ml" (.reg "adv"),
                .setp .eq "extC" "adv" (.imm 32) ]),
            .bin .sub "off0" "p0" (.reg "cand0"),
            .bin .sub "litLen" "p0" (.reg "litAnchor"),
            wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
            .bin .add "litAnchor" "p0" (.reg "ml"),
            .mov "searchPos" (.reg "litAnchor") ])
          (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]),
        .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ]
    let st' := body.eval F ws
    st'.gmem = EmitContent.putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
          (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
              (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0), p - c, ml⟩)
      ∧ (st'.regs "searchPos").toNat = p + ml ∧ (st'.regs "litAnchor").toNat = p + ml
      ∧ st'.regs "outBase" = ws.regs "outBase" ∧ st'.regs "inBase" = ws.regs "inBase"
      ∧ st'.regs "op" = ws.regs "op" + UInt64.ofNat (1 + (encNib (p - anchor)).length
          + (p - anchor) + 2 + (encNib (ml - 4)).length)
      ∧ st'.smem = tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s (p - s)
      ∧ st'.regs "loopC" = (if UInt64.ofNat (p + ml) < UInt64.ofNat (inStride - 12) then 1 else 0) := by
  intro ml body st'
  -- Peel the `coopWindow ; uif` emit prefix via `emitStepFound_eq`.
  obtain ⟨stP, hstPdef⟩ : ∃ stP,
      (wseq [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
        .uif "found" (wseq
          [ .mov "ecR" (.imm (inStride - 5)), .mov "ec1" (.imm (inStride - 5 - 1)),
            .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
            .uwhile "extC" (wseq
              [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride (inStride - 5),
                .bin .add "ml" "ml" (.reg "adv"),
                .setp .eq "extC" "adv" (.imm 32) ]),
            .bin .sub "off0" "p0" (.reg "cand0"),
            .bin .sub "litLen" "p0" (.reg "litAnchor"),
            wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
            .bin .add "litAnchor" "p0" (.reg "ml"),
            .mov "searchPos" (.reg "litAnchor") ])
          (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]) ]).eval F ws = stP := ⟨_, rfl⟩
  obtain ⟨hP_gmem, hP_sp, hP_la, hP_ob, hP_ib, hP_op, hP_smem⟩ :=
    emitStepFound_eq inStride (inStride - 12) hashLog s p c F F ws anchor hstride rfl (by omega)
      hwin has hsval hlaval hCfuel hFfuel hfuelL hfuelM hp64 hnwFull hnw2 hsize hdisj
  rw [hstPdef] at hP_gmem hP_sp hP_la hP_ob hP_ib hP_op hP_smem
  -- `st' = (setp loopC).eval F stP` after unfolding the two nested `.seq`s.
  have hbody : st' = (WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F stP := by
    show (WStmt.eval F body ws) = _
    rw [← hstPdef]
    simp only [body, wseq, WStmt.eval.eq_2]
  obtain ⟨hR_loopC, hR_reg, hR_gmem, hR_smem⟩ := loopC_recompute (inStride - 12) F stP
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hbody, hR_gmem, hP_gmem]
  · rw [hbody, hR_reg "searchPos" (by decide), hP_sp]
  · rw [hbody, hR_reg "litAnchor" (by decide), hP_la]
  · rw [hbody, hR_reg "outBase" (by decide), hP_ob]
  · rw [hbody, hR_reg "inBase" (by decide), hP_ib]
  · rw [hbody, hR_reg "op" (by decide), hP_op]
  · rw [hbody, hR_smem, hP_smem]
  · rw [hbody, hR_loopC]
    have hsp : stP.regs "searchPos" = UInt64.ofNat (p + ml) := by
      have h1 : (stP.regs "searchPos").toNat = p + ml := hP_sp
      rw [← h1, UInt64.ofNat_toNat]
    rw [hsp]

-- ── The full `uwhile "loopC"` fuel induction: mirrors `genLoop_valid`'s ──────
-- skeleton exactly (`by_cases hs : s < searchLim`, `cases window`), composing
-- `emitStepFound_eq`/`emitStepNotFound_eq` + `loopC_recompute` per iteration.
-- The layout-invariant (`hnwFull`/`hnw2`/`hsize`/`hdisj`-shaped) is threaded
-- as `hlayout`, restated at the NEW `(s, anchor, op)` each step — valid because
-- `op` only grows and the output region only shrinks the remaining `gmem.size`
-- budget monotonically, so if it fit before a smaller remaining plan still fits.
theorem emitLoop_eq (inStride hashLog : Nat) (hstride : inStride ≤ 65536)
    (hp64 : inStride < 2 ^ 64) :
    ∀ (F : Nat) (anchor s : Nat) (ws : WState),
      anchor ≤ s → (ws.regs "searchPos").toNat = s → (ws.regs "litAnchor").toNat = anchor →
      (ws.regs "loopC") = (if UInt64.ofNat s < UInt64.ofNat (inStride - 12) then 1 else 0) →
      -- fuel sufficiency: enough for every remaining outer iteration (`F` decrements
      -- by 1 per `WStmt.eval`'s own `uwhile` recursion, `s` strictly increases each
      -- time so `searchLim - s` bounds the remaining iteration count) PLUS headroom
      -- for the inner extend-loop launched with the CURRENT remaining fuel at every
      -- step (nested loops don't share a consumed budget in `WStmt.eval`'s model).
      ((inStride - 12) - s) + 33 * inStride ≤ F →
      -- layout: the output region from `outBase+op` for the REMAINING run only
      -- (bounded generously by 9 bytes of encoded output per remaining input
      -- byte — indexed by `anchor` rather than `s` since the literal run length
      -- `p - anchor` is what the budget must cover, and `anchor ≤ s` always —
      -- the budget SHRINKS as `anchor→searchLim`, which is what makes this
      -- invariant actually inductive: each step consumes ≤ 9*(p-anchor+4) of
      -- the budget while `anchor` advances to `p+ml`, i.e. by the same amount)
      -- stays in-bounds and disjoint from the (fixed, size-`inStride`) input region.
      (ws.regs "outBase").toNat + (ws.regs "op").toNat + 9 * (inStride - anchor) < 2 ^ 64 →
      (ws.regs "outBase").toNat + (ws.regs "op").toNat + 9 * (inStride - anchor)
          ≤ ws.gmem.size →
      ((ws.regs "inBase").toNat + inStride
          ≤ (ws.regs "outBase").toNat + (ws.regs "op").toNat) →
      -- the (fixed, size-`inStride`) input region itself never overflows `UInt64`
      (ws.regs "inBase").toNat + inStride < 2 ^ 64 →
      let st' := (WStmt.uwhile "loopC" (wseq
        [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
          .uif "found" (wseq
            [ .mov "ecR" (.imm (inStride - 5)), .mov "ec1" (.imm (inStride - 5 - 1)),
              .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
              .uwhile "extC" (wseq
                [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride (inStride - 5),
                  .bin .add "ml" "ml" (.reg "adv"),
                  .setp .eq "extC" "adv" (.imm 32) ]),
              .bin .sub "off0" "p0" (.reg "cand0"),
              .bin .sub "litLen" "p0" (.reg "litAnchor"),
              wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
              .bin .add "litAnchor" "p0" (.reg "ml"),
              .mov "searchPos" (.reg "litAnchor") ])
            (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]),
          .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ])).eval F ws
      let plan := genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
          (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
          (inStride - 12) (inStride - 5) F ws.smem anchor s
      (st'.gmem = EmitContent.putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
          ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) anchor plan.steps plan.finalLen).seqs.flatMap
            LZ4.encodeSeq))
      ∧ (st'.regs "litAnchor").toNat = anchor + AlgorithmLib.LZ4Plan.stepsLen plan.steps
      ∧ st'.regs "outBase" = ws.regs "outBase" ∧ st'.regs "inBase" = ws.regs "inBase"
      ∧ st'.regs "op" = ws.regs "op" + UInt64.ofNat
          ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) anchor plan.steps plan.finalLen).seqs.flatMap
            LZ4.encodeSeq).length := by
  intro F
  induction F with
  | zero =>
      intro anchor s ws has hsval hlaval hloopC hFsuff hnwFull hsize hdisj hinBound
      intro st' plan
      have hi0 : inStride = 0 := by omega
      have hst'0 : st' = ws := WStmt.eval.eq_11 ws "loopC" _
      have hplan0 : plan.steps = [] := by
        show (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
          (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
          (inStride - 12) (inStride - 5) 0 ws.smem anchor s).steps = []
        rfl
      -- All facts collapse because the plan is empty (`anchor = inStride = 0`).
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · rw [hst'0, hplan0]; simp [planBlockFrom, EmitContent.putBytesU]
      · rw [hst'0, hplan0]; simp only [AlgorithmLib.LZ4Plan.stepsLen, Nat.add_zero, hlaval]
      · rw [hst'0]
      · rw [hst'0]
      · rw [hst'0, hplan0]; simp [planBlockFrom]
  | succ n ih =>
      intro anchor s ws has hsval hlaval hloopC hFsuff hnwFull hsize hdisj hinBound
      intro st' plan
      rw [show st' = WStmt.eval (n+1) (WStmt.uwhile "loopC" _) ws from rfl, WStmt.eval.eq_12]
      by_cases hs : s < inStride - 12
      · have hguard : (ws.regs "loopC" == 1) = true := by
          rw [hloopC]
          have : UInt64.ofNat s < UInt64.ofNat (inStride - 12) := by
            rw [UInt64.lt_iff_toNat_lt, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt s (by omega),
              AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (inStride - 12) (by omega)]
            exact hs
          rw [if_pos this]; rfl
        rw [if_pos hguard]
        have hs64 : s < 2 ^ 64 := by rw [← hsval]; exact (ws.regs "searchPos").toNat_lt
        cases hw : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) s with
        | none =>
            -- Abbreviate the post-`body` state `stN = body.eval n ws`.
            obtain ⟨stN, hstNdef⟩ : ∃ stN, (wseq
              [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
                .uif "found" (wseq
                  [ .mov "ecR" (.imm (inStride - 5)), .mov "ec1" (.imm (inStride - 5 - 1)),
                    .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
                    .uwhile "extC" (wseq
                      [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride (inStride - 5),
                        .bin .add "ml" "ml" (.reg "adv"),
                        .setp .eq "extC" "adv" (.imm 32) ]),
                    .bin .sub "off0" "p0" (.reg "cand0"),
                    .bin .sub "litLen" "p0" (.reg "litAnchor"),
                    wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
                    .bin .add "litAnchor" "p0" (.reg "ml"),
                    .mov "searchPos" (.reg "litAnchor") ])
                  (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]),
                .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ]).eval n ws = stN := ⟨_, rfl⟩
            obtain ⟨hN_gmem, hN_sp, hN_la, hN_ob, hN_ib, hN_op, hN_smem, hN_loopC⟩ :=
              bodyNotFound_eq inStride hashLog s n ws hw hsval (by omega)
            rw [hstNdef] at hN_gmem hN_sp hN_la hN_ob hN_ib hN_op hN_smem hN_loopC
            -- The loop continues on `stN`; rewrite the LHS to `(uwhile ...).eval n stN`.
            rw [hstNdef]
            -- Apply the induction hypothesis at `(anchor, s+32, stN)`.
            obtain ⟨hIH_g, hIH_la, hIH_ob, hIH_ib, hIH_op⟩ := ih anchor (s + 32) stN (by omega) hN_sp
              (by rw [hN_la]; exact hlaval) hN_loopC
              (by omega)
              (by rw [hN_ob, hN_op]; exact hnwFull)
              (by rw [hN_ob, hN_op, hN_gmem]; exact hsize)
              (by rw [hN_ob, hN_op, hN_ib]; exact hdisj)
              (by rw [hN_ib]; exact hinBound)
            -- Fold the `genLoop` recursion: the `(n+1)` plan (via `plan`) equals the
            -- `n` plan on `stN`'s inserted table (`anchor` unchanged, `s → s+32`).
            have hgenEq : genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) (n + 1) ws.smem anchor s
              = genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) n
                (tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s 32) anchor (s + 32) :=
              genLoop_notFound_step (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) n ws.smem anchor s _ hs hw rfl
            -- `plan` (the `(n+1)` genLoop) rewrites to the `stN`-table `n` genLoop.
            have hplan_eq : plan = genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) n
                (tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s 32) anchor (s + 32) := hgenEq
            rw [hplan_eq]
            refine ⟨?_, ?_, ?_, ?_, ?_⟩
            · rw [hIH_g, hN_ib, hN_gmem, hN_ob, hN_op, hN_smem]
            · rw [hIH_la, hN_ib, hN_smem, hN_gmem]
            · rw [hIH_ob, hN_ob]
            · rw [hIH_ib, hN_ib]
            · rw [hIH_op, hN_ib, hN_op, hN_smem, hN_gmem]
        | some pc =>
            obtain ⟨p, c⟩ := pc
            have hwc := AlgorithmLib.LZ4WarpFind.window_sound (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
              (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) s p c hw
            have hp64' : p < 2 ^ 64 := by have := hwc.2.1; omega
            have hc64' : c < 2 ^ 64 := by have := hwc.2.2.1; omega
            have hp_le : p ≤ inStride := by have := hwc.2.1; omega
            have hml_le : p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) n 4
                ≤ inStride - 5 := by
              rcases AlgorithmLib.LZ4WarpFind.extendFrom_cap (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
                  (inStride - 5) n 4 with heq | hle
              · rw [heq]; omega
              · exact hle
            have hencnib1 : (encNib (p - anchor)).length ≤ (p - anchor) + 1 := encNib_length_le _
            have hencnib2 : (encNib (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) n 4
                - 4)).length ≤ extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) n 4 - 4 + 1 :=
              encNib_length_le _
            have hib64 : (ws.regs "inBase").toNat < 2 ^ 64 := (ws.regs "inBase").toNat_lt
            -- `ml` abbreviates the extend length (loop fuel `n`); `hpml*` are the bounds.
            obtain ⟨ml, hmldef⟩ : ∃ ml,
                extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) n 4 = ml := ⟨_, rfl⟩
            rw [hmldef] at hml_le hencnib2
            have hpml_le : p + ml ≤ inStride - 5 := hml_le
            have hpml64 : p + ml < 2 ^ 64 := by omega
            have has_p : anchor ≤ p := Nat.le_trans has hwc.1
            -- Abbreviate the post-`body` state `stF = body.eval n ws`.
            obtain ⟨stF, hstFdef⟩ : ∃ stF, (wseq
              [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
                .uif "found" (wseq
                  [ .mov "ecR" (.imm (inStride - 5)), .mov "ec1" (.imm (inStride - 5 - 1)),
                    .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
                    .uwhile "extC" (wseq
                      [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride (inStride - 5),
                        .bin .add "ml" "ml" (.reg "adv"),
                        .setp .eq "extC" "adv" (.imm 32) ]),
                    .bin .sub "off0" "p0" (.reg "cand0"),
                    .bin .sub "litLen" "p0" (.reg "litAnchor"),
                    wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
                    .bin .add "litAnchor" "p0" (.reg "ml"),
                    .mov "searchPos" (.reg "litAnchor") ])
                  (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]),
                .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ]).eval n ws = stF := ⟨_, rfl⟩
            obtain ⟨hF_gmem, hF_sp, hF_la, hF_ob, hF_ib, hF_op, hF_smem, hF_loopC⟩ :=
              bodyFound_eq inStride hashLog s p c n ws anchor hstride hw has hsval hlaval
                (by omega) (by omega) (by omega) (by rw [hmldef]; omega) hp64
                (by rw [hmldef]; omega) (by omega) (by omega) (by omega) (by rw [hmldef]; omega)
            rw [hstFdef] at hF_gmem hF_sp hF_la hF_ob hF_ib hF_op hF_smem hF_loopC
            rw [hmldef] at hF_gmem hF_sp hF_la hF_op hF_loopC
            -- The loop continues on `stF`; rewrite LHS to `(uwhile ...).eval n stF`.
            rw [hstFdef]
            -- The input region `[0, inStride)` is disjoint from the write region
            -- `[outBase+op, …)` (via `hdisj` + `hib0`), so `gmemInp` is unchanged.
            -- From `hdisj` (2nd disjunct, since `hib0` rules out the 1st unless empty)
            -- + `hnwFull`: the write region begins at `outBase+op ≥ inStride`.
            have hInLtBaseSum : (ws.regs "inBase").toNat + inStride
                ≤ (ws.regs "outBase").toNat + (ws.regs "op").toNat := hdisj
            have hbaseN : (ws.regs "outBase" + ws.regs "op").toNat
                = (ws.regs "outBase").toNat + (ws.regs "op").toNat := by
              rw [UInt64.toNat_add]; apply Nat.mod_eq_of_lt
              have h1 := encNib_length_le (p - anchor)
              have h2 := encNib_length_le (ml - 4); omega
            have hwritelen : (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
                (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0), p - c, ml⟩).length
                = 1 + (encNib (p - anchor)).length + (p - anchor) + 2 + (encNib (ml - 4)).length := by
              rw [LZ4WarpDSL.encodeSeq_length]; simp
            have hwrite_nw : (ws.regs "outBase" + ws.regs "op").toNat
                + (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
                  (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0), p - c, ml⟩).length
                < 2 ^ 64 := by
              rw [hbaseN, hwritelen]
              have h1 := encNib_length_le (p - anchor)
              have h2 := encNib_length_le (ml - 4); omega
            have hgmemInp_eq : gmemInpAt (EmitContent.putBytesU ws.gmem
                (ws.regs "outBase" + ws.regs "op") (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
                  (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0), p - c, ml⟩))
                (ws.regs "inBase").toNat inStride = gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride := by
              unfold gmemInpAt
              apply List.map_congr_left
              intro i hi
              rw [List.mem_range] at hi
              rw [EmitContent.putBytesU_getD_lt ((ws.regs "inBase").toNat + i) _ _ _ hwrite_nw
                (by rw [hbaseN]; omega)]
            -- `stF.op = op + encLen`; compute its `.toNat` (no overflow — the budget
            -- inequality `hbudget` keeps it under `2^64`) once, reuse for all layouts.
            have h1 := encNib_length_le (p - anchor)
            have h2 := encNib_length_le (ml - 4)
            have hml4 : 4 ≤ ml := by
              rw [← hmldef]; exact AlgorithmLib.LZ4WarpFind.extendFrom_le _ p c (inStride - 5) n 4
            have hsplit : inStride - anchor = (p + ml - anchor) + (inStride - (p + ml)) := by omega
            have hbudget : (1 + (encNib (p - anchor)).length + (p - anchor) + 2
                + (encNib (ml - 4)).length) + 9 * (inStride - (p + ml))
                ≤ 9 * (inStride - anchor) := by
              rw [hsplit, Nat.mul_add]; omega
            have hF_opN : (stF.regs "op").toNat = (ws.regs "op").toNat
                + (1 + (encNib (p - anchor)).length + (p - anchor) + 2 + (encNib (ml - 4)).length) := by
              rw [hF_op, UInt64.toNat_add, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega)]
              apply Nat.mod_eq_of_lt; omega
            -- Apply the induction hypothesis at `(p+ml, p+ml, stF)`.
            obtain ⟨hIH_g, hIH_la, hIH_ob, hIH_ib, hIH_op⟩ :=
              ih (p + ml) (p + ml) stF (Nat.le_refl _) hF_sp hF_la hF_loopC
              (by omega)
              (by rw [hF_ob, hF_opN]; omega)
              (by rw [hF_ob, hF_opN, hF_gmem, EmitContent.putBytesU_size]; omega)
              (by rw [hF_ob, hF_opN, hF_ib]; omega)
              (by rw [hF_ib]; exact hinBound)
            -- Fold `genLoop` at the hit + telescope `planBlockFrom` + `putBytesU_append`.
            have hgenEq : (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) (n + 1) ws.smem anchor s).steps
              = ⟨p - anchor, p - c, ml⟩ :: (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) n
                (tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s (p - s))
                (p + ml) (p + ml)).steps :=
              genLoop_found_step (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) n ws.smem anchor s p c ml _ hs hw
                (by rw [← hmldef,
                      extendFrom_fuel_sat (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) n 4 (by omega)]) rfl
            -- Companion telescoping for `finalLen` (found step keeps recursion's).
            have hgenFl : (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) (n + 1) ws.smem anchor s).finalLen
              = (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) n
                (tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s (p - s))
                (p + ml) (p + ml)).finalLen :=
              genLoop_found_finalLen (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) n ws.smem anchor s p c ml _ hs hw
                (by rw [← hmldef,
                      extendFrom_fuel_sat (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) n 4 (by omega)]) rfl
            -- input region (hash reads live at `pos < inStride-12`) lies below base.
            have hbaseN2 : (ws.regs "inBase").toNat + (inStride - 12) + 3
                ≤ (ws.regs "outBase" + ws.regs "op").toNat := by
              rw [hbaseN]; omega
            have hoAgree : ∀ (sm : Array UInt8) (pos : Nat), pos < inStride - 12 →
                tableOracle (EmitContent.putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
                    (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
                      (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0),
                      p - c, ml⟩)) sm hashLog 0 (ws.regs "inBase").toNat pos
                  = tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat pos := by
              intro sm pos hpos
              unfold tableOracle wHash
              rw [wLoad4_putBytesU_lt ws.gmem _ _ ((ws.regs "inBase").toNat + pos) hwrite_nw
                (by omega)]
            have hupdEq : (fun sm sp upto => tableInsert sm (EmitContent.putBytesU ws.gmem
                  (ws.regs "outBase" + ws.regs "op") (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
                    (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0), p - c, ml⟩))
                  hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                = (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto) := by
              funext sm sp upto
              exact tableInsert_putBytesU_eq ws.gmem _ _ hashLog (inStride - 12)
                (ws.regs "inBase").toNat hwrite_nw hbaseN2 sm sp upto
            have htail : (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle (EmitContent.putBytesU ws.gmem
                  (ws.regs "outBase" + ws.regs "op") (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
                    (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0), p - c, ml⟩))
                  sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm (EmitContent.putBytesU ws.gmem
                  (ws.regs "outBase" + ws.regs "op") (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
                    (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0), p - c, ml⟩))
                  hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) n
                (tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s (p - s)) (p + ml) (p + ml))
              = genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                (inStride - 12) (inStride - 5) n
                (tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s (p - s)) (p + ml) (p + ml) := by
              rw [hupdEq]
              exact genLoop_oracle_congr (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) _ _ _ (inStride - 12)
                (inStride - 5) hoAgree n _ (p + ml) (p + ml)
            -- The encoded first-step byte length (`op` advance / base offset).
            have hlen : (LZ4.encodeSeq ⟨(List.range (p - anchor)).map
                (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + anchor + i) 0), p - c, ml⟩).length
                = 1 + (encNib (p - anchor)).length + (p - anchor) + 2 + (encNib (ml - 4)).length := by
              rw [LZ4WarpDSL.encodeSeq_length]; simp
            -- `stepsLen` of the telescoped tail — the terminal litAnchor's model side.
            refine ⟨?_, ?_, ?_, ?_, ?_⟩
            · -- gmem: the byte-content telescoping (original found-branch proof).
              rw [hIH_g]
              simp only [hF_ib, hF_gmem, hF_ob, hF_op, hF_smem, hgmemInp_eq]
              show _ = EmitContent.putBytesU ws.gmem _
                ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) anchor
                  (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                    (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                    (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                    (inStride - 12) (inStride - 5) (n + 1) ws.smem anchor s).steps
                  (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                    (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                    (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                    (inStride - 12) (inStride - 5) (n + 1) ws.smem anchor s).finalLen).seqs.flatMap
                  LZ4.encodeSeq)
              rw [hgenEq, hgenFl, planBlockFrom_found_step, EmitContent.putBytesU_append, htail,
                gmemInpAt_slice_eq ws.gmem (ws.regs "inBase").toNat inStride anchor (p - anchor) (by omega),
                show anchor + (p - anchor) + ml = p + ml from by omega]
              congr 1
              · rw [hlen, show ws.regs "outBase" + ws.regs "op"
                    + UInt64.ofNat (1 + (encNib (p - anchor)).length + (p - anchor) + 2
                      + (encNib (ml - 4)).length)
                  = ws.regs "outBase" + (ws.regs "op" + UInt64.ofNat (1 + (encNib (p - anchor)).length
                      + (p - anchor) + 2 + (encNib (ml - 4)).length)) from by ac_rfl]
            · -- litAnchor: `(p+ml) + stepsLen tail = anchor + stepsLen (step :: tail)`.
              rw [hIH_la]
              simp only [hF_ib, hF_gmem, hF_smem, hgmemInp_eq, htail]
              rw [hgenEq]
              simp only [AlgorithmLib.LZ4Plan.stepsLen]
              omega
            · -- outBase preserved.
              rw [hIH_ob, hF_ob]
            · -- inBase preserved.
              rw [hIH_ib, hF_ib]
            · -- op: `ws.op + firstStep + tail = ws.op + full-plan bytes`.
              rw [hIH_op, hF_op]
              simp only [hF_ib, hF_gmem, hF_smem, hgmemInp_eq, htail]
              rw [hgenEq, hgenFl, planBlockFrom_found_step,
                gmemInpAt_slice_eq ws.gmem (ws.regs "inBase").toNat inStride anchor (p - anchor) (by omega),
                show anchor + (p - anchor) + ml = p + ml from by omega,
                List.length_append, hlen]
              rw [show ws.regs "op"
                    + UInt64.ofNat (1 + (encNib (p - anchor)).length + (p - anchor) + 2
                      + (encNib (ml - 4)).length)
                    + UInt64.ofNat ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (p + ml)
                        (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                          (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                          (inStride - 12) (inStride - 5) n
                          (tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s (p - s))
                          (p + ml) (p + ml)).steps
                        (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                          (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                          (inStride - 12) (inStride - 5) n
                          (tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s (p - s))
                          (p + ml) (p + ml)).finalLen).seqs.flatMap LZ4.encodeSeq).length
                  = ws.regs "op" + (UInt64.ofNat (1 + (encNib (p - anchor)).length + (p - anchor) + 2
                      + (encNib (ml - 4)).length)
                    + UInt64.ofNat ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (p + ml)
                        (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                          (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                          (inStride - 12) (inStride - 5) n
                          (tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s (p - s))
                          (p + ml) (p + ml)).steps
                        (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
                          (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
                          (inStride - 12) (inStride - 5) n
                          (tableInsert ws.smem ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat s (p - s))
                          (p + ml) (p + ml)).finalLen).seqs.flatMap LZ4.encodeSeq).length)
                  from by ac_rfl, ← UInt64.ofNat_add]
      · have hs64 : s < 2 ^ 64 := by rw [← hsval]; exact (ws.regs "searchPos").toNat_lt
        have hguard : (ws.regs "loopC" == 1) = false := by
          rw [hloopC]
          have hns : ¬ UInt64.ofNat s < UInt64.ofNat (inStride - 12) := by
            rw [UInt64.lt_iff_toNat_lt, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt s hs64,
              AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (inStride - 12) (by omega)]
            exact hs
          rw [if_neg hns]; rfl
        rw [if_neg (by rw [hguard]; decide)]
        -- Loop exits (`st' = ws`); the plan is empty, so all facts are immediate.
        have hplanE : (genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
            (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
            (inStride - 12) (inStride - 5) (n + 1) ws.smem anchor s).steps = [] :=
          genLoop_notFound_empty (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
            (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
            (inStride - 12) (inStride - 5) n ws.smem anchor s hs
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · rw [hplanE]; simp [planBlockFrom, EmitContent.putBytesU]
        · rw [hplanE]; simp only [AlgorithmLib.LZ4Plan.stepsLen, Nat.add_zero, hlaval]
        · rfl
        · rfl
        · rw [hplanE]; simp [planBlockFrom]

end AlgorithmLib.LZ4WarpDSL
