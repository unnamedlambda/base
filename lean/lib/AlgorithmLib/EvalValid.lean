import AlgorithmLib.LZ4WarpDSL
import AlgorithmLib.LZ4WarpFind
import AlgorithmLib.OuterByteLayer
import AlgorithmLib.SimSLAssembly

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib.LZ4Plan AlgorithmLib.LZ4WarpFind
open AlgorithmLib.LZ4Simt (SCmp)

/-- `genLoop` produces a valid plan-suffix for ANY oracle-maker `mkOracle` and
    state update `upd` — mirror of `loop_valid`, with the threaded state `smem`
    generalized through the induction. Uses only `window_sound` + `extendFrom_*`,
    all oracle-independent. -/
theorem genLoop_valid (inp : List UInt8)
    (mkOracle : Array UInt8 → (Nat → Option Nat))
    (upd : Array UInt8 → Nat → Nat → Array UInt8)
    (hlen : inp.length ≤ 65536) :
    ∀ fuel anchor s smem, anchor ≤ s → anchor ≤ inp.length →
      ValidStepsFrom inp anchor
        (genLoop inp mkOracle upd (inp.length - 12) (inp.length - 5)
          fuel smem anchor s).steps
        (genLoop inp mkOracle upd (inp.length - 12) (inp.length - 5)
          fuel smem anchor s).finalLen := by
  intro fuel
  induction fuel with
  | zero =>
      intro anchor s smem _ hal
      simp only [genLoop, ValidStepsFrom]
      refine ⟨by omega, ?_⟩
      simp [List.length_drop]
  | succ n ih =>
      intro anchor s smem has hal
      unfold genLoop
      by_cases hs : s < inp.length - 12
      · rw [if_pos hs]
        cases hw : window inp (mkOracle smem) (inp.length - 12) s with
        | none =>
            exact ih anchor (s + 32) _ (by omega) hal
        | some pc =>
            obtain ⟨p, c⟩ := pc
            obtain ⟨hsp, hpsl, hcp, hv⟩ := window_sound inp _ _ s p c hw
            dsimp only
            generalize hmldef :
              extendFrom inp p c (inp.length - 5) (inp.length - 5 - (p + 4)) 4 = ml
            have hml4 : 4 ≤ ml :=
              hmldef ▸ extendFrom_le inp p c (inp.length - 5) (inp.length - 5 - (p + 4)) 4
            have hcap : p + ml ≤ inp.length - 5 := by
              rcases extendFrom_cap inp p c (inp.length - 5) (inp.length - 5 - (p + 4)) 4
                  with heq | hle
              · rw [hmldef] at heq; omega
              · rw [hmldef] at hle; exact hle
            have hagree : ∀ i, i < ml → byte inp (p + i) = byte inp (c + i) := by
              rw [← hmldef]
              exact extendFrom_agree inp p c _ _ 4 (verify4_agree inp p c hv)
            simp only [ValidStepsFrom, stepAgree]
            constructor
            · refine ⟨by omega, by omega, hml4, by omega, ?_⟩
              intro i hi
              have h1 : anchor + (p - anchor) + i = p + i := by omega
              have h2 : p + i - (p - c) = c + i := by omega
              rw [h1, h2]
              exact hagree i hi
            · have h3 : anchor + (p - anchor) + ml = p + ml := by omega
              rw [h3]
              exact ih (p + ml) (p + ml) _ (Nat.le_refl _) (by omega)
      · rw [if_neg hs]
        simp only [ValidStepsFrom]
        refine ⟨by omega, ?_⟩
        simp [List.length_drop]

/-- The eval's plan is valid for the block input. -/
theorem evalPlan_valid (gmem smem0 : Array UInt8) (ib inStride hashLog : Nat)
    (hlen : inStride ≤ 65536) :
    ValidPlan (gmemInpAt gmem ib inStride) (evalPlan gmem smem0 ib inStride hashLog) := by
  have hlen' : (gmemInpAt gmem ib inStride).length = inStride := by simp [gmemInpAt]
  unfold ValidPlan evalPlan evalPlan_loop
  have := genLoop_valid (gmemInpAt gmem ib inStride)
    (fun sm => tableOracle gmem sm hashLog 0 ib)
    (fun sm sp upto => tableInsert sm gmem hashLog 0 (inStride - 12) ib sp upto)
    (by rw [hlen']; exact hlen) inStride 0 0 smem0 (Nat.le_refl 0) (by rw [hlen']; exact Nat.zero_le _)
  rw [hlen'] at this
  exact this

/-- The encode-producing prefix of the compressor body: `uwhile loopC` (the main
    loop) then `mov fLen inStride; sub fLen litAnchor; wEmitFinalSeq litAnchor fLen`.
    (The trailing 4-byte length header — which writes to a separate region — is
    handled at the SimSL/prologue layer.) -/
def bodyEncodePrefix (inStride hashLog : Nat) : WStmt :=
  wseq
  [ .setp .lt "loopC" "searchPos" (.imm (inStride - 12)),
    .uwhile "loopC" (wseq
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
        .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ]),
    .mov "fLen" (.imm inStride),
    .bin .sub "fLen" "fLen" (.reg "litAnchor"),
    wEmitFinalSeq "litAnchor" "fLen" ]

/-- **The `houtput` bridge**: the sequential `eval` of the encode-producing prefix
    writes, at `outBase`, exactly `(planToBlock inp evalPlan).encode`.  Preconditions
    are the post-prologue state: `op=litAnchor=searchPos=0`, `inBase=0`, `loopC` set,
    and the layout invariants (in-bounds, disjoint output region). -/
theorem compressorBody_output_eq (inStride hashLog : Nat) (ws : WState)
    (hstride : inStride ≤ 65536) (hp64 : inStride < 2 ^ 64) (hipos : 12 ≤ inStride)
    (hop0 : ws.regs "op" = 0) (hla0 : ws.regs "litAnchor" = 0)
    (hsp0 : ws.regs "searchPos" = 0) (hib0 : (ws.regs "inBase").toNat < 2 ^ 40)
    (hobN : (ws.regs "outBase").toNat + 9 * inStride < 2 ^ 64)
    (hsize : (ws.regs "outBase").toNat + 9 * inStride ≤ ws.gmem.size)
    (hdisj : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat)
    (hinB : (ws.regs "inBase").toNat + inStride < 2 ^ 64)
    -- the encoded block fits the generous `9*inStride` output-region budget.
    (hencLen : (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length ≤ 9 * inStride) :
    ((bodyEncodePrefix inStride hashLog).eval (inStride + 34 * inStride) ws).gmem
      = EmitContent.putBytesU ws.gmem (ws.regs "outBase")
          (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode
    ∧ ((bodyEncodePrefix inStride hashLog).eval (inStride + 34 * inStride) ws).regs "op"
      = UInt64.ofNat
          (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length
    ∧ ((bodyEncodePrefix inStride hashLog).eval (inStride + 34 * inStride) ws).regs "outBase"
      = ws.regs "outBase" := by
  have hlen' : (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride).length = inStride := by simp [gmemInpAt]
  -- Peel the outer 5-statement `wseq` into `.seq`s but keep the loop body folded
  -- as `wseq` (so `emitLoop_eq`'s statement matches syntactically).
  rw [show bodyEncodePrefix inStride hashLog
      = (WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).seq
        ((WStmt.uwhile "loopC" (wseq
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
            .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ])).seq
          ((WStmt.mov "fLen" (WArg.imm inStride)).seq
            ((WStmt.bin WOp.sub "fLen" "fLen" (WArg.reg "litAnchor")).seq
              (wEmitFinalSeq "litAnchor" "fLen")))) from rfl]
  simp only [WStmt.eval.eq_2]
  -- Step 0: the leading `setp loopC` recomputes `loopC` from `searchPos=0`.
  obtain ⟨ws0, hws0def⟩ : ∃ ws0,
      (WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval
        (inStride + 34 * inStride) ws = ws0 := ⟨_, rfl⟩
  rw [hws0def]
  have hws0_gmem : ws0.gmem = ws.gmem := by rw [← hws0def]; exact EmitContent.setp_gmem _ _ _ _ _ _
  have hws0_smem : ws0.smem = ws.smem := by
    rw [← hws0def]; exact noSmem_eval_smem (.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12)))
      trivial _ ws
  have hws0_sp : ws0.regs "searchPos" = ws.regs "searchPos" := by
    rw [← hws0def]; exact EmitContent.setp_reg _ _ _ _ _ _ _ (by decide)
  have hws0_la : ws0.regs "litAnchor" = ws.regs "litAnchor" := by
    rw [← hws0def]; exact EmitContent.setp_reg _ _ _ _ _ _ _ (by decide)
  have hws0_ob : ws0.regs "outBase" = ws.regs "outBase" := by
    rw [← hws0def]; exact EmitContent.setp_reg _ _ _ _ _ _ _ (by decide)
  have hws0_ib : ws0.regs "inBase" = ws.regs "inBase" := by
    rw [← hws0def]; exact EmitContent.setp_reg _ _ _ _ _ _ _ (by decide)
  have hws0_op : ws0.regs "op" = ws.regs "op" := by
    rw [← hws0def]; exact EmitContent.setp_reg _ _ _ _ _ _ _ (by decide)
  have hws0_loopC : ws0.regs "loopC"
      = (if UInt64.ofNat 0 < UInt64.ofNat (inStride - 12) then 1 else 0) := by
    rw [← hws0def]
    have hsp' : ws.regs "searchPos" = UInt64.ofNat 0 := by rw [hsp0]; rfl
    simp only [WStmt.eval, WState.setReg, WArg.eval, SCmp.run, hsp', if_true]
    by_cases h : UInt64.ofNat 0 < UInt64.ofNat (inStride - 12)
    · rw [if_pos h, decide_eq_true h, if_pos rfl]
    · rw [if_neg h, decide_eq_false h, if_neg (by simp)]
  -- Step 1: the main loop, via `emitLoop_eq` at `anchor = s = 0` on `ws0`.
  -- Generalize the loop-eval subterm (matches the `.seq`-unfolded body by defeq).
  obtain ⟨wsL, hwsLdef⟩ : ∃ wsL, WStmt.eval (inStride + 34 * inStride)
      (WStmt.uwhile "loopC" (wseq
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
        .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ])) ws0 = wsL := ⟨_, rfl⟩
  rw [hwsLdef]
  have hsp0N : (ws0.regs "searchPos").toNat = 0 := by rw [hws0_sp, hsp0]; rfl
  have hla0N : (ws0.regs "litAnchor").toNat = 0 := by rw [hws0_la, hla0]; rfl
  have hLoop := emitLoop_eq inStride hashLog hstride hp64 (inStride + 34 * inStride) 0 0 ws0
      (Nat.le_refl 0) hsp0N hla0N hws0_loopC
      (by omega)
      (by rw [hws0_ob, hws0_op, hop0]; simpa using hobN)
      (by rw [hws0_ob, hws0_op, hop0, hws0_gmem]; simpa using hsize)
      (by rw [hws0_ob, hws0_op, hop0, hws0_ib]; simpa using hdisj)
      (by rw [hws0_ib]; exact hinB)
  simp only at hLoop
  obtain ⟨hL_g, hL_la, hL_ob, hL_ib, hL_op⟩ := hLoop
  rw [hwsLdef] at hL_g hL_la hL_ob hL_ib hL_op
  -- Terminal-loop facts in terms of `ws` and the model plan.
  have hplan_def : evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog
      = genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
        (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
        (inStride - 12) (inStride - 5) (inStride + 34 * inStride) ws.smem 0 0 := by
    -- `evalPlan` uses fuel `inStride`; the loop used `inStride + 34*inStride`.  Both
    -- are ≥ the span `searchLim - 0 = inStride - 12`, so fuel-irrelevance applies.
    show genLoop (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
        (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
        (inStride - 12) (inStride - 5) inStride ws.smem 0 0 = _
    exact genLoop_fuel_irrel (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
      (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12) (ws.regs "inBase").toNat sp upto)
      (inStride - 12) (inStride - 5) inStride inStride (inStride + 34 * inStride) ws.smem 0 0
      (by omega) (by omega) (by omega)
  -- The loop wrote the seqs bytes; rewrite `hL_*` from `ws0`/fuel form to `ws`/plan.
  rw [hws0_ob, hws0_op, hop0, hws0_gmem, hws0_smem, hws0_ib, ← hplan_def] at hL_g
  rw [hws0_gmem, hws0_smem, hws0_ib, ← hplan_def] at hL_la
  rw [hws0_op, hop0, hws0_gmem, hws0_smem, hws0_ib, ← hplan_def] at hL_op
  rw [hws0_ob] at hL_ob
  rw [hws0_ib] at hL_ib
  -- Terminal litAnchor = stepsLen steps (= inStride − finalLen).
  have hstepsLen : AlgorithmLib.LZ4Plan.stepsLen (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
      + (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen = inStride := by
    have hv := evalPlan_valid ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog hstride
    have := AlgorithmLib.LZ4Plan.ValidStepsFrom_total (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
      (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
      (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen hv
    have hln : (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride).length = inStride := by simp [gmemInpAt]
    rw [hln] at this; omega
  -- Step 2: `mov fLen inStride`.
  obtain ⟨wsA, hwsAdef⟩ : ∃ wsA,
      (WStmt.mov "fLen" (WArg.imm inStride)).eval (inStride + 34 * inStride) wsL = wsA := ⟨_, rfl⟩
  rw [hwsAdef]
  have hA_gmem : wsA.gmem = wsL.gmem := by rw [← hwsAdef]; exact EmitContent.mov_gmem _ _ _ _
  have hA_fLen : wsA.regs "fLen" = UInt64.ofNat inStride := by
    rw [← hwsAdef]; simp [WStmt.eval, WState.setReg, WArg.eval]
  have hA_la : wsA.regs "litAnchor" = wsL.regs "litAnchor" := by
    rw [← hwsAdef]; exact EmitContent.mov_reg _ _ _ _ _ (by decide)
  have hA_ob : wsA.regs "outBase" = wsL.regs "outBase" := by
    rw [← hwsAdef]; exact EmitContent.mov_reg _ _ _ _ _ (by decide)
  have hA_ib : wsA.regs "inBase" = wsL.regs "inBase" := by
    rw [← hwsAdef]; exact EmitContent.mov_reg _ _ _ _ _ (by decide)
  have hA_op : wsA.regs "op" = wsL.regs "op" := by
    rw [← hwsAdef]; exact EmitContent.mov_reg _ _ _ _ _ (by decide)
  -- Step 3: `sub fLen fLen litAnchor` ⇒ `fLen = ofNat (inStride − stepsLen steps) = ofNat finalLen`.
  obtain ⟨wsB, hwsBdef⟩ : ∃ wsB,
      (WStmt.bin WOp.sub "fLen" "fLen" (WArg.reg "litAnchor")).eval
        (inStride + 34 * inStride) wsA = wsB := ⟨_, rfl⟩
  rw [hwsBdef]
  have hB_gmem : wsB.gmem = wsA.gmem := by rw [← hwsBdef]; exact EmitContent.bin_gmem _ _ _ _ _ _
  have hla_le : AlgorithmLib.LZ4Plan.stepsLen (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
      ≤ inStride := by
    have := hstepsLen; omega
  have hlaN64 : AlgorithmLib.LZ4Plan.stepsLen (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
      < 2 ^ 64 := Nat.lt_of_le_of_lt hla_le hp64
  have hA_laN : wsA.regs "litAnchor"
      = UInt64.ofNat (AlgorithmLib.LZ4Plan.stepsLen (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps) := by
    apply UInt64.toNat_inj.mp
    rw [hA_la, hL_la, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ hlaN64, Nat.zero_add]
  have hB_fLen : wsB.regs "fLen" = UInt64.ofNat
      ((evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen) := by
    rw [← hwsBdef]
    simp only [WStmt.eval, WState.setReg, WArg.eval, WOp.run, hA_fLen, hA_laN]
    rw [AlgorithmLib.LZ4Ptx.u64_sub_ofNat' inStride _ hla_le hp64]
    have hfe : inStride - AlgorithmLib.LZ4Plan.stepsLen
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
        = (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen := by
      have := hstepsLen; omega
    rw [hfe, if_true]
  have hfl64 : (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen < 2 ^ 64 := by
    have := hstepsLen; omega
  have hB_fLenN : (wsB.regs "fLen").toNat = (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen := by
    rw [hB_fLen, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ hfl64]
  -- Frame the other registers through `mov`/`sub` onto `wsB`.
  have hB_la : wsB.regs "litAnchor" = wsA.regs "litAnchor" := by
    rw [← hwsBdef]; exact EmitContent.bin_reg _ _ _ _ _ _ _ (by decide)
  have hB_ob : wsB.regs "outBase" = wsA.regs "outBase" := by
    rw [← hwsBdef]; exact EmitContent.bin_reg _ _ _ _ _ _ _ (by decide)
  have hB_ib : wsB.regs "inBase" = wsA.regs "inBase" := by
    rw [← hwsBdef]; exact EmitContent.bin_reg _ _ _ _ _ _ _ (by decide)
  have hB_op : wsB.regs "op" = wsA.regs "op" := by
    rw [← hwsBdef]; exact EmitContent.bin_reg _ _ _ _ _ _ _ (by decide)
  -- Consolidated `wsB` facts (all in `ws`/plan terms).
  have hB_laN : (wsB.regs "litAnchor").toNat
      = AlgorithmLib.LZ4Plan.stepsLen (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps := by
    rw [hB_la, hA_la, hL_la, Nat.zero_add]
  have hB_obE : wsB.regs "outBase" = ws.regs "outBase" := by rw [hB_ob, hA_ob, hL_ob]
  have hB_ibE : wsB.regs "inBase" = ws.regs "inBase" := by rw [hB_ib, hA_ib, hL_ib]
  have hB_opE : wsB.regs "op" = UInt64.ofNat
      ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).seqs.flatMap LZ4.encodeSeq).length := by
    rw [hB_op, hA_op, hL_op, UInt64.zero_add]
  have hB_gmemE : wsB.gmem = EmitContent.putBytesU ws.gmem (ws.regs "outBase" + 0)
      ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).seqs.flatMap LZ4.encodeSeq) := by
    rw [hB_gmem, hA_gmem, hL_g]
  -- Step 4: `wEmitFinalSeq litAnchor fLen` writes the final literal run.
  -- `encode = seqsBytes ++ encodeFinal final`, so `encode.length = seqsBytes.length +
  -- finalBytes.length`; combined with `hencLen` this bounds each part.
  have hencSplit : ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).seqs.flatMap LZ4.encodeSeq).length
      + (LZ4.encodeFinal (planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).final).length
      ≤ 9 * inStride := by
    have : (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length
        = ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).seqs.flatMap LZ4.encodeSeq).length
          + (LZ4.encodeFinal (planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).final).length := by
      simp only [planToBlock, LZ4.Block.encode, List.length_append]
    omega
  have hB_ib0N : (wsB.regs "inBase").toNat = (ws.regs "inBase").toNat := by rw [hB_ibE]
  -- `wsB.op.toNat = seqsBytes.length` and the seqs-part length bound.
  have hB_opN : (wsB.regs "op").toNat
      = ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).seqs.flatMap LZ4.encodeSeq).length := by
    rw [hB_opE, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt]
    have := hencSplit; omega
  have hobN' : (ws.regs "outBase").toNat + 9 * inStride < 2 ^ 64 := hobN
  -- The final literal run's length = `finalLen`, and its encoded length.
  have hfinalN : (planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
      (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
      (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).final.length
      = (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen := by
    rw [AlgorithmLib.LZ4Plan.planBlockFrom_final]
    simp only [List.length_take, List.length_drop, hlen']
    have := hstepsLen; omega
  have hfinEnc : (LZ4.encodeFinal (planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
      (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
      (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).final).length
      = 1 + (LZ4.encNib (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).length
        + (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen := by
    rw [encodeFinal_length, hfinalN]
  refine ⟨?_, ?_, ?_⟩
  · -- Apply the final-seq content lemma on `wsB`.
    rw [EmitContent.eval_wEmitFinalSeq_content "litAnchor" "fLen" wsB (inStride + 34 * inStride)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by rw [hB_fLenN]; omega)
      (by rw [hB_obE, hB_opN, hB_fLenN]
          rw [hfinEnc] at hencSplit; omega)
      (by rw [hB_ibE, hB_laN, hB_fLenN]; have := hstepsLen; omega)
      (by rw [hB_obE, hB_opN, hB_fLenN, hB_gmemE, EmitContent.putBytesU_size]
          rw [hfinEnc] at hencSplit; omega)
      (by rw [hB_obE, hB_opN, hB_fLenN, hB_ibE, hB_laN]
          right
          have := hstepsLen; omega)]
    -- Compose: `putBytesU (putBytesU g outBase seqs) (outBase+seqsLen) final = Block.encode`.
    rw [hB_gmemE, hB_obE, hB_opE, hB_ibE, hB_laN, hB_fLenN]
    -- The final literals are read from the WRITTEN gmem but at input positions
    -- `[stepsLen, stepsLen+finalLen) ⊂ [0, inStride)`, disjoint from the write region.
    have hreadThrough : (List.range (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).map
        (fun i => (EmitContent.putBytesU ws.gmem (ws.regs "outBase" + 0)
          ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).seqs.flatMap LZ4.encodeSeq)).getD
          ((ws.regs "inBase").toNat + AlgorithmLib.LZ4Plan.stepsLen (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps + i) 0)
        = (List.range (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).map
          (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + AlgorithmLib.LZ4Plan.stepsLen
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps + i) 0) := by
      apply List.map_congr_left
      intro i hi
      rw [List.mem_range] at hi
      have hbaseN : (ws.regs "outBase" + 0).toNat = (ws.regs "outBase").toNat := by
        rw [UInt64.add_zero]
      have hj : (ws.regs "inBase").toNat
            + AlgorithmLib.LZ4Plan.stepsLen (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps + i
          < (ws.regs "outBase" + 0).toNat := by
        rw [hbaseN]; have := hstepsLen; omega
      have hnw : (ws.regs "outBase" + 0).toNat
          + ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).seqs.flatMap LZ4.encodeSeq).length
          < 2 ^ 64 := by
        rw [hbaseN]; rw [hfinEnc] at hencSplit; omega
      rw [EmitContent.putBytesU_getD_lt _ _ _ _ hnw hj]
    rw [hreadThrough]
    -- The final literal bytes: `getD (stepsLen + i)` over `range finalLen` = `.final`.
    have hfinalBytes : (List.range (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).map
        (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + AlgorithmLib.LZ4Plan.stepsLen
          (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps + i) 0)
        = (planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
          (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
          (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).final := by
      rw [AlgorithmLib.LZ4Plan.planBlockFrom_final, Nat.zero_add,
        ← gmemInpAt_slice_eq ws.gmem (ws.regs "inBase").toNat inStride
          (AlgorithmLib.LZ4Plan.stepsLen (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps)
          (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen (by have := hstepsLen; omega)]
    rw [hfinalBytes, UInt64.add_zero, ← EmitContent.putBytesU_append]
    -- `seqs ++ encodeFinal final = Block.encode`.
    congr 1

  · -- `op` advance: seqs bytes + final-seq bytes = |encode|
    rw [(LZ4WarpEvalBytes.eval_wEmitFinalSeq "litAnchor" "fLen" wsB (inStride + 34 * inStride)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
      (by rw [hB_fLenN]; omega)).1]
    rw [hB_opE, hB_fLenN]
    have hsplit : (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length
        = ((planBlockFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) 0
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).steps
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).seqs.flatMap LZ4.encodeSeq).length
          + (1 + (LZ4.encNib (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen).length
              + (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog).finalLen) := by
      have := hfinEnc
      simp only [planToBlock, LZ4.Block.encode, List.length_append]
      omega
    rw [hsplit]
    simp only [UInt64.ofNat_add]
  · -- `outBase` is preserved by the whole prefix
    rw [(LZ4WarpEvalBytes.eval_wEmitFinalSeq "litAnchor" "fLen" wsB (inStride + 34 * inStride)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
      (by rw [hB_fLenN]; omega)).2]
    exact hB_obE

/-- **Top theorem** (shape locked; `houtput` is the outer-loop byte-layer
    induction's conclusion, to be discharged once the byte-content assembly
    lands). Given that the compressor's emitted output bytes equal
    `encode(evalPlan)` for a block, the standard LZ4 decompressor recovers the
    exact input — i.e. the DSL kernel's compressor output roundtrips. -/
theorem warpKernelDSL_roundtrips (gmem smem0 : Array UInt8) (inStride hashLog : Nat)
    (outBytes : List UInt8) (hlen : inStride ≤ 65536)
    (houtput : outBytes
        = (planToBlock (gmemInpAt gmem ib inStride) (evalPlan gmem smem0 ib inStride hashLog)).encode) :
    AlgorithmLib.LZ4Imp.decompress outBytes (gmemInpAt gmem ib inStride).length
      = some (gmemInpAt gmem ib inStride) := by
  rw [houtput]
  exact plan_roundtrip (gmemInpAt gmem ib inStride) (evalPlan gmem smem0 ib inStride hashLog)
    (evalPlan_valid gmem smem0 ib inStride hashLog hlen)

/-- **Fully-closed top theorem** (no `houtput` hypothesis): running the DSL
    compressor body's encode-producing prefix on a valid post-prologue state
    writes an output block into `gmem` such that reading the written window back
    and decompressing it recovers the exact input.  The window's bytes are
    *derived* to equal `encode(evalPlan)` (via `compressorBody_output_eq` +
    `putBytesU_getD_win`), then `warpKernelDSL_roundtrips` closes the roundtrip.
    This is the end-to-end `eval`-level correctness of the measured kernel. -/
theorem warpKernelDSL_body_roundtrips (inStride hashLog : Nat) (ws : WState)
    (hstride : inStride ≤ 65536) (hp64 : inStride < 2 ^ 64) (hipos : 12 ≤ inStride)
    (hop0 : ws.regs "op" = 0) (hla0 : ws.regs "litAnchor" = 0)
    (hsp0 : ws.regs "searchPos" = 0) (hib0 : (ws.regs "inBase").toNat < 2 ^ 40)
    (hobN : (ws.regs "outBase").toNat + 9 * inStride < 2 ^ 64)
    (hsize : (ws.regs "outBase").toNat + 9 * inStride ≤ ws.gmem.size)
    (hdisj : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat)
    (hinB : (ws.regs "inBase").toNat + inStride < 2 ^ 64)
    (hencLen : (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length ≤ 9 * inStride) :
    -- read the written output window `[outBase, outBase + encodeLen)` back...
    AlgorithmLib.LZ4Imp.decompress
        ((List.range (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length).map
          (fun i => ((bodyEncodePrefix inStride hashLog).eval (inStride + 34 * inStride) ws).gmem.getD
            ((ws.regs "outBase").toNat + i) 0))
        (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride).length
      = some (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) := by
  -- The read-back window equals `encode(evalPlan)` (the write's own bytes).
  have hbase : (ws.regs "outBase").toNat = (ws.regs "outBase").toNat := rfl
  have hread : (List.range (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length).map
      (fun i => ((bodyEncodePrefix inStride hashLog).eval (inStride + 34 * inStride) ws).gmem.getD
        ((ws.regs "outBase").toNat + i) 0)
      = (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode := by
    rw [(compressorBody_output_eq inStride hashLog ws hstride hp64 hipos hop0 hla0 hsp0 hib0
      hobN hsize hdisj hinB hencLen).1]
    apply List.ext_getElem
    · simp
    · intro i h1 h2
      simp only [List.getElem_map, List.getElem_range]
      rw [show (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode[i]
          = (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.getD i 0 from by
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2]; rfl,
        EmitContent.putBytesU_getD_win (ws.regs "outBase")
        (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode
        ws.gmem i h2 (by omega) (by omega)]
  rw [hread]
  exact warpKernelDSL_roundtrips ws.gmem ws.smem inStride hashLog _ hstride rfl

/-- **The loopC body preserves `LoopCQ` and strictly advances `searchPos`.**  The
    pure-eval core of the loop induction: mirrors `emitLoop_eq`'s inductive step at the
    single-body level (case on the window result, apply `bodyFound_eq`/`bodyNotFound_eq`,
    discharge the layout budget via `encNib_length_le`/`extendFrom_cap`).  `simSL'_measureLoop`
    consumes this (the guard gives `searchPos < searchLim`; `μ = searchPos` advances). -/
theorem loopCBody_Qadvance (inStride hashLog fuel : Nat) (ws : WState)
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride) (hp64 : inStride < 2 ^ 64)
    (hfuel : inStride ≤ fuel)
    (hQ : LoopCQ inStride ws) (hguard : (ws.regs "loopC" == 1) = true) :
    LoopCQ inStride ((loopCBodyStmt inStride hashLog).eval fuel ws)
    ∧ (ws.regs "searchPos").toNat + 1
        ≤ (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "searchPos").toNat := by
  obtain ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ := hQ
  have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
    loopCQ_guard inStride ws hstride hipos ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ hguard
  cases hw : window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat)
      (inStride - 12) (ws.regs "searchPos").toNat with
  | none =>
      obtain ⟨hN_gmem, hN_sp, hN_la, hN_ob, hN_ib, hN_op, _hN_smem, hN_loopC⟩ :=
        bodyNotFound_eq inStride hashLog (ws.regs "searchPos").toNat fuel ws hw rfl (by omega)
      have e_gmem : ((loopCBodyStmt inStride hashLog).eval fuel ws).gmem = ws.gmem := hN_gmem
      have e_sp : (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "searchPos").toNat
          = (ws.regs "searchPos").toNat + 32 := hN_sp
      have e_la : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "litAnchor"
          = ws.regs "litAnchor" := hN_la
      have e_ob : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "outBase"
          = ws.regs "outBase" := hN_ob
      have e_ib : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "inBase"
          = ws.regs "inBase" := hN_ib
      have e_op : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "op"
          = ws.regs "op" := hN_op
      have e_loopC : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "loopC"
          = (if UInt64.ofNat ((ws.regs "searchPos").toNat + 32) < UInt64.ofNat (inStride - 12)
             then 1 else 0) := hN_loopC
      refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
      · rw [e_ib]; exact hib0
      · rw [e_la, e_sp]; omega
      · rw [e_loopC, e_sp]
      · rw [e_ob, e_ib]; exact hobLB
      · rw [e_ob, e_op, e_la]; exact hbud32
      · rw [e_ob, e_op, e_la, e_gmem]; exact hbudsz
      · rw [e_sp]; omega
  | some pc =>
      obtain ⟨p, c⟩ := pc
      obtain ⟨hsps, hpsl, _hcp, _hv⟩ := window_sound (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) (ws.regs "searchPos").toNat p c hw
      have hml4 : 4 ≤ extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 :=
        extendFrom_le _ p c (inStride - 5) fuel 4
      have hmlcap : p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4
          ≤ inStride - 5 := by
        rcases extendFrom_cap (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 with heq | hle
        · omega
        · exact hle
      have h1 := encNib_length_le (p - (ws.regs "litAnchor").toNat)
      have h2 := encNib_length_le
        (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 - 4)
      have hbudget : (1 + (LZ4.encNib (p - (ws.regs "litAnchor").toNat)).length
          + (p - (ws.regs "litAnchor").toNat) + 2
          + (LZ4.encNib (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 - 4)).length)
          + 9 * (inStride - (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4))
          ≤ 9 * (inStride - (ws.regs "litAnchor").toNat) := by
        have hsplit : inStride - (ws.regs "litAnchor").toNat
            = (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4
                - (ws.regs "litAnchor").toNat)
              + (inStride - (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4)) := by
          omega
        rw [hsplit, Nat.mul_add]; omega
      obtain ⟨hF_gmem, hF_sp, hF_la, hF_ob, hF_ib, hF_op, _hF_smem, hF_loopC⟩ :=
        bodyFound_eq inStride hashLog (ws.regs "searchPos").toNat p c fuel ws
          (ws.regs "litAnchor").toNat hstride hw hlaSp rfl rfl
          (by omega) (by omega)
          (by have := Nat.div_le_self (p - (ws.regs "litAnchor").toNat) 255; omega)
          (by have := Nat.div_le_self
                (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 - 4) 255; omega)
          hp64
          (by omega) (by omega) (by omega) (Or.inr (by omega)) (by omega)
      -- `op` grows by the encoded length; compute its `toNat` (no overflow).
      have hF_opN : (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "op").toNat
          = (ws.regs "op").toNat + (1 + (LZ4.encNib (p - (ws.regs "litAnchor").toNat)).length
              + (p - (ws.regs "litAnchor").toNat) + 2
              + (LZ4.encNib (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 - 4)).length) := by
        have : (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "op") = _ := hF_op
        rw [this, UInt64.toNat_add, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega)]
        apply Nat.mod_eq_of_lt; omega
      have e_gmem : ((loopCBodyStmt inStride hashLog).eval fuel ws).gmem
          = EmitContent.putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op") _ := hF_gmem
      have e_sp : (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "searchPos").toNat
          = p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 := hF_sp
      have e_la : (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "litAnchor").toNat
          = p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 := hF_la
      have e_ob : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "outBase"
          = ws.regs "outBase" := hF_ob
      have e_ib : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "inBase"
          = ws.regs "inBase" := hF_ib
      have e_loopC : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "loopC"
          = (if UInt64.ofNat (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4)
                < UInt64.ofNat (inStride - 12) then 1 else 0) := hF_loopC
      refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
      · rw [e_ib]; exact hib0
      · rw [e_la, e_sp]; omega
      · rw [e_loopC, e_sp]
      · rw [e_ob, e_ib]; exact hobLB
      · rw [e_ob, hF_opN, e_la]; omega
      · rw [e_ob, hF_opN, e_la, e_gmem, EmitContent.putBytesU_size]; omega
      · rw [e_sp]; omega

/-- One active `loopC` iteration keeps `litAnchor ≤ inStride − 5` (a not-found step
    leaves `litAnchor`; a found step sets it to the match end `p + ml ≤ endCap`). -/
theorem loopCBody_litAnchor_le (inStride hashLog fuel : Nat) (ws : WState)
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride) (hp64 : inStride < 2 ^ 64)
    (hfuel : inStride ≤ fuel) (hla5 : (ws.regs "litAnchor").toNat ≤ inStride - 5)
    (hQ : LoopCQ inStride ws) (hguard : (ws.regs "loopC" == 1) = true) :
    (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "litAnchor").toNat ≤ inStride - 5 := by
  obtain ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ := hQ
  have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
    loopCQ_guard inStride ws hstride hipos ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ hguard
  cases hw : window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat)
      (inStride - 12) (ws.regs "searchPos").toNat with
  | none =>
      obtain ⟨_, _, hN_la, _, _, _, _, _⟩ :=
        bodyNotFound_eq inStride hashLog (ws.regs "searchPos").toNat fuel ws hw rfl (by omega)
      have e_la : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "litAnchor"
          = ws.regs "litAnchor" := hN_la
      rw [e_la]; exact hla5
  | some pc =>
      obtain ⟨p, c⟩ := pc
      obtain ⟨hsps, hpsl, _hcp, _hv⟩ := window_sound (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) (ws.regs "searchPos").toNat p c hw
      have hml4 : 4 ≤ extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 :=
        extendFrom_le _ p c (inStride - 5) fuel 4
      have hmlcap : p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4
          ≤ inStride - 5 := by
        rcases extendFrom_cap (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 with heq | hle
        · omega
        · exact hle
      have h1 := encNib_length_le (p - (ws.regs "litAnchor").toNat)
      have h2 := encNib_length_le
        (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 - 4)
      have hbudget : (1 + (LZ4.encNib (p - (ws.regs "litAnchor").toNat)).length
          + (p - (ws.regs "litAnchor").toNat) + 2
          + (LZ4.encNib (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 - 4)).length)
          + 9 * (inStride - (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4))
          ≤ 9 * (inStride - (ws.regs "litAnchor").toNat) := by
        have hsplit : inStride - (ws.regs "litAnchor").toNat
            = (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4
                - (ws.regs "litAnchor").toNat)
              + (inStride - (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4)) := by
          omega
        rw [hsplit, Nat.mul_add]; omega
      obtain ⟨_, _, hF_la, _, _, _, _, _⟩ :=
        bodyFound_eq inStride hashLog (ws.regs "searchPos").toNat p c fuel ws
          (ws.regs "litAnchor").toNat hstride hw hlaSp rfl rfl
          (by omega) (by omega)
          (by have := Nat.div_le_self (p - (ws.regs "litAnchor").toNat) 255; omega)
          (by have := Nat.div_le_self
                (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 - 4) 255; omega)
          hp64
          (by omega) (by omega) (by omega) (Or.inr (by omega)) (by omega)
      have e_la : (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "litAnchor").toNat
          = p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4 := hF_la
      rw [e_la]; exact hmlcap

section LoopAsm
open AlgorithmLib.LZ4Simt

/-- The live coupling-register set for the `loopC` loop (excludes `p0`/`cand0`, which
    `coopWindow` writes each iteration). -/
def loopR : List String :=
  ["searchPos", "found", "loopC", "inBase", "outBase", "op", "litAnchor", "ecR", "ec1",
   "ml", "extC", "adv", "off0", "litLen", "mlm", "tokLo", "tokHi", "tok", "sbAddr",
   "pLitBig", "pMatBig", "litExtra", "matExtra", "c255", "lsicC", "offLo", "offHi",
   "cpDst", "cpSrc", "fLen", "zero", "pLitBigF", "litExtraF", "cpDstF", "cpSrcF",
   -- length-store tail registers (untouched by the body, so uniformly `0` there)
   "la0", "la1", "la2", "la3", "lb", "ls"]

/-- **The `loopC` body simulates in the window-MISS case.**  `coopWindow` reports
    `found = 0` (via `simSL_coopWindow` + `coopWindow_found_val`), the `uif` takes the
    else branch (`searchPos += 32`, `simSL'_notFoundBranch`), then the `loopC` guard
    recomputes.  Mirrors `simSL'_uifBodyInv`'s else branch. -/
theorem loopCBody_none_reaches (inStride hashLog : Nat)
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
    (hc : Couple loopR ss ws) (hmi : MachInv ib ss) (hib40 : ib < 2 ^ 40) :
    ∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
      ss'.pc = base + (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL
        cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length ∧
      Couple loopR ss' ((loopCBodyStmt inStride hashLog).eval fuel ws) ∧ MachInv ib ss' := by
  obtain ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ := hQ
  have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
    loopCQ_guard inStride ws hstride hipos ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ hguard
  rw [loopCBodyEmit] at hseg hlr
  -- Step 1: coopWindow (window miss ⇒ found = 0).
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1⟩ :=
    simSL_coopWindow loopR inStride (inStride - 12) hashLog (ws.regs "searchPos").toNat
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
  obtain ⟨ne, ss2, hrE, hpcE, hcE, hmiE⟩ :=
    simSL'_notFoundBranch loopR (by decide)
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
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5⟩ :=
    simSL'_setp loopR .lt "loopC" "searchPos" (.imm (inStride - 12)) (by decide)
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
  refine ⟨(n1 + (1 + (1 + (ne + 1)))) + n5, ss5,
    sreaches_trans prog _ n5 _ _ _
      (sreaches_trans prog n1 _ _ _ _ hr1
        (sreaches_trans prog 1 _ _ _ _ (sreaches_one_eq s0)
          (sreaches_trans prog 1 _ _ _ _ (sreaches_one_eq s1)
            (sreaches_trans prog ne 1 _ _ _ hrE (sreaches_one_eq s2))))) hr5, ?_, ?_, hmi5⟩
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

/-- Facts about the post-`coopWindow` state in the window-HIT case, transferred from
    `LoopCQ` + the window semantics — the hypotheses `foundBranch_sideconds` /
    `simSL'_foundBranch` consume. -/
theorem loopCBody_found_cwFacts (inStride hashLog : Nat) (ws : WState)
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride)
    (hQ : LoopCQ inStride ws) (hguard : (ws.regs "loopC" == 1) = true)
    (p c : Nat)
    (hw : window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat)
        (inStride - 12) (ws.regs "searchPos").toNat = some (p, c)) :
    ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "inBase").toNat < 2 ^ 40
    ∧ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "litAnchor").toNat
      ≤ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "p0").toNat
    ∧ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "p0").toNat < inStride
    ∧ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "outBase").toNat
        + ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
          ws).regs "op").toNat
        + 9 * (inStride - ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
          hashLog 0 ws).regs "litAnchor").toNat) < 2 ^ 32
    ∧ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "outBase").toNat
        + ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
          ws).regs "op").toNat
        + 9 * (inStride - ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
          hashLog 0 ws).regs "litAnchor").toNat)
        ≤ (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
          ws).gmem.size
    ∧ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "inBase").toNat + inStride ≤ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "outBase").toNat
    ∧ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "cand0").toNat
      < ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "p0").toNat
    ∧ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
        ws).regs "p0").toNat + 4 ≤ inStride - 5 := by
  obtain ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ := hQ
  obtain ⟨hsps, hpsl, hcp, _⟩ := window_sound (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
    (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) (ws.regs "searchPos").toNat p c hw
  have hfound1 : (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
      hashLog 0 ws).regs "found" = 1 := by
    rw [evalCoopWindow_eq_go, hw, evalCoopWindowGo_regs]; simp [WState.setReg]
  obtain ⟨hcand, hp4⟩ := coopWindow_found_bounds inStride hashLog ws hstride hipos hfound1
  have hp0val : (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
      hashLog 0 ws).regs "p0" = UInt64.ofNat p := by
    rw [evalCoopWindow_eq_go, hw, evalCoopWindowGo_regs]; simp [WState.setReg]
  have hp0N : ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
      hashLog 0 ws).regs "p0").toNat = p := by
    rw [hp0val, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt]; omega
  -- register frames: `coopWindow` preserves everything but found/p0/cand0.
  have hib : (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
      ws).regs "inBase" = ws.regs "inBase" :=
    evalCoopWindow_regs_frame _ _ _ _ _ _ _ _ ws _ (by decide) (by decide) (by decide)
  have hla : (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
      ws).regs "litAnchor" = ws.regs "litAnchor" :=
    evalCoopWindow_regs_frame _ _ _ _ _ _ _ _ ws _ (by decide) (by decide) (by decide)
  have hob : (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
      ws).regs "outBase" = ws.regs "outBase" :=
    evalCoopWindow_regs_frame _ _ _ _ _ _ _ _ ws _ (by decide) (by decide) (by decide)
  have hop : (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
      ws).regs "op" = ws.regs "op" :=
    evalCoopWindow_regs_frame _ _ _ _ _ _ _ _ ws _ (by decide) (by decide) (by decide)
  have hgm : (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0
      ws).gmem = ws.gmem := by rw [evalCoopWindow_eq_go, evalCoopWindowGo_gmem]
  refine ⟨by rw [hib]; exact hib0, ?_, by rw [hp0N]; omega, ?_, ?_,
    by rw [hob, hib]; exact hobLB, hcand, hp4⟩
  · rw [hla, hp0N]; omega
  · rw [hob, hop, hla]; exact hbud32
  · rw [hob, hop, hla, hgm]; exact hbudsz

/-- **The `loopC` body simulates in the window-HIT case.**  `coopWindow` couples
    `p0`/`cand0` (`coopWindow_couple_found`), `found = 1`, the `uif` takes the then
    branch (`simSL'_foundBranch`, its side-conditions from `loopCBody_found_cwFacts`
    + `foundBranch_sideconds`), then the `loopC` guard recomputes; `Couple.drop2`
    restores the loop coupling set. -/
theorem loopCBody_found_reaches (inStride hashLog : Nat)
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
    (hc : Couple loopR ss ws) (hmi : MachInv ib ss) (hib40 : ib < 2 ^ 40) :
    ∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
      ss'.pc = base + (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL
        cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length ∧
      Couple loopR ss' ((loopCBodyStmt inStride hashLog).eval fuel ws) ∧ MachInv ib ss' := by
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
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1⟩ :=
    coopWindow_couple_found loopR inStride (inStride - 12) hashLog (ws.regs "searchPos").toNat p c
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
  obtain ⟨n4, ss4, hr4, hpc4, hc4, hmi4⟩ :=
    simSL'_foundBranch ("p0" :: "cand0" :: loopR) inStride (inStride - 5)
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
      cwcand cwp4 (by have := cwp4; have := hfuelb; have := hipos; omega)
      sc1 sc2 sc3 sc4 sc5 sc6 sc7 sc8 sc9
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
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5⟩ :=
    simSL'_setp ("p0" :: "cand0" :: loopR) .lt "loopC" "searchPos" (.imm (inStride - 12))
      (by decide) (fun n h => by cases h) (by decide)
      prog (base2 + (uifEmit "found" lElse lEnd et [.bin .add "searchPos" "searchPos" (.imm 32)]).length)
      (ss4.setPc (base2 + 1 + et.length + 1 + 1 + 1 + 1))
      ((foundBranchStmt inStride (inStride - 5)).eval fuel
        ((WStmt.coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0).eval
          fuel ws))
      fuel (by simp only [SState.setPc, huplen]; omega) hsegR.append_right hlrR.append_right
      (couple_setPc hc4 _) (machInv_setPc ss4 _ hmi4)
  refine ⟨(n1 + (1 + (n4 + (1 + 1)))) + n5, ss5,
    sreaches_trans prog _ n5 _ _ _
      (sreaches_trans prog n1 _ _ _ _ hr1
        (sreaches_trans prog 1 _ _ _ _ (sreaches_one_eq s0)
          (sreaches_trans prog n4 _ _ _ _ hr4
            (sreaches_trans prog 1 1 _ _ _ (sreaches_one_eq s1) (sreaches_one_eq s2))))) hr5,
    ?_, ?_, hmi5⟩
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

/-- **The `loopC` body simulates and preserves the loop invariant / measure** — the
    `hbody` obligation of `simSL'_measureLoop`.  Cases on the window result
    (`loopCBody_none_reaches` / `loopCBody_found_reaches` for the simulation half;
    `loopCBody_Qadvance` for `LoopCQ`-preservation + `searchPos`-advance). -/
theorem loopCBody_bodySim (inStride hashLog F : Nat)
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
    (hfloor : F ≤ (ws.regs "searchPos").toNat + fuel + 1) :
    (∃ (m : Nat) (ss' : SState), SReaches prog m ss ss' ∧
      ss'.pc = base + 1 + 1 + (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL
        lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM).length ∧
      Couple loopR ss' ((loopCBodyStmt inStride hashLog).eval fuel ws) ∧ MachInv ib ss')
    ∧ LoopCQ inStride ((loopCBodyStmt inStride hashLog).eval fuel ws)
    ∧ (ws.regs "searchPos").toNat + 1
        ≤ (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "searchPos").toNat := by
  have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
    loopCQ_guard inStride ws hstride hipos hQ hguard
  have hfuelb : inStride ≤ fuel := by omega
  refine ⟨?_, loopCBody_Qadvance inStride hashLog fuel ws hstride hipos (by omega) hfuelb hQ hguard⟩
  cases hw : window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat)
      (inStride - 12) (ws.regs "searchPos").toNat with
  | none =>
      exact loopCBody_none_reaches inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX
        lElseM lEndM lHM lXM lsicL lsicM hstride hipos hlen ws hQ hguard hhl hw prog (base + 1 + 1)
        ss fuel hpc hseg hlr hc hmi hib40
  | some pc =>
      obtain ⟨p, c⟩ := pc
      exact loopCBody_found_reaches inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX
        lElseM lEndM lHM lXM lsicL lsicM hLdef hMdef hstride hipos hlen ws hQ hguard hhl fuel hfuelb
        p c hw prog (base + 1 + 1) ss hpc hseg hlr hc hmi hib40

/-- The `loopC` loop terminates: each active iteration advances `searchPos` by ≥ 1
    (`loopCBody_Qadvance`) toward the search limit, and the combined measure
    `(searchLim − searchPos) + inStride ≤ fuel` both bounds the remaining iterations
    and keeps the per-iteration fuel `≥ inStride` (so the body simulation applies). -/
theorem loopC_halts (inStride hashLog : Nat) (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride)
    (hp64 : inStride < 2 ^ 64) :
    ∀ (fuel : Nat) (ws : WState), LoopCQ inStride ws →
      (inStride - 12) - (ws.regs "searchPos").toNat + inStride ≤ fuel →
      WhileHalts "loopC" (loopCBodyStmt inStride hashLog) fuel ws := by
  intro fuel
  induction fuel with
  | zero => intro ws _ hf; exact absurd hf (by omega)
  | succ n ih =>
      intro ws hQ hf
      rw [WhileHalts]
      by_cases hg : (ws.regs "loopC" == 1) = true
      · rw [if_pos hg]
        have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
          loopCQ_guard inStride ws hstride hipos hQ hg
        have hqa := loopCBody_Qadvance inStride hashLog n ws hstride hipos hp64 (by omega) hQ hg
        exact ih ((loopCBodyStmt inStride hashLog).eval n ws) hqa.1 (by omega)
      · rw [if_neg hg]; trivial

/-- `LoopCQ` is preserved across the whole `loopC` loop (each active iteration
    preserves it via `loopCBody_Qadvance`; the exit state inherits it). -/
theorem loopC_loop_preservesInv (inStride hashLog : Nat) (hstride : inStride ≤ 65536)
    (hipos : 12 ≤ inStride) (hp64 : inStride < 2 ^ 64) :
    ∀ (fuel : Nat) (ws : WState), LoopCQ inStride ws →
      (inStride - 12) - (ws.regs "searchPos").toNat + inStride ≤ fuel →
      LoopCQ inStride ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval fuel ws) := by
  intro fuel
  induction fuel with
  | zero => intro ws _ hf; exact absurd hf (by omega)
  | succ n ih =>
      intro ws hQ hf
      rw [WStmt.eval]
      by_cases hg : (ws.regs "loopC" == 1) = true
      · rw [if_pos hg]
        have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
          loopCQ_guard inStride ws hstride hipos hQ hg
        have hqa := loopCBody_Qadvance inStride hashLog n ws hstride hipos hp64 (by omega) hQ hg
        exact ih ((loopCBodyStmt inStride hashLog).eval n ws) hqa.1 (by omega)
      · rw [if_neg hg]; exact hQ

/-- `litAnchor ≤ inStride − 5` holds at the `loopC` loop exit (preserved each
    iteration via `loopCBody_litAnchor_le`); gives `fLen = inStride − litAnchor ≥ 5`. -/
theorem loopC_litAnchor_bound (inStride hashLog : Nat) (hstride : inStride ≤ 65536)
    (hipos : 12 ≤ inStride) (hp64 : inStride < 2 ^ 64) :
    ∀ (fuel : Nat) (ws : WState), LoopCQ inStride ws →
      (ws.regs "litAnchor").toNat ≤ inStride - 5 →
      (inStride - 12) - (ws.regs "searchPos").toNat + inStride ≤ fuel →
      (((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval fuel ws).regs
        "litAnchor").toNat ≤ inStride - 5 := by
  intro fuel
  induction fuel with
  | zero => intro ws _ _ hf; exact absurd hf (by omega)
  | succ n ih =>
      intro ws hQ hla5 hf
      rw [WStmt.eval]
      by_cases hg : (ws.regs "loopC" == 1) = true
      · rw [if_pos hg]
        have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
          loopCQ_guard inStride ws hstride hipos hQ hg
        have hqa := loopCBody_Qadvance inStride hashLog n ws hstride hipos hp64 (by omega) hQ hg
        have hla' := loopCBody_litAnchor_le inStride hashLog n ws hstride hipos hp64 (by omega)
          hla5 hQ hg
        exact ih ((loopCBodyStmt inStride hashLog).eval n ws) hqa.1 hla' (by omega)
      · rw [if_neg hg]; exact hla5

-- ── final-literal-run side-condition value lemmas (mirror the match-seq family) ──

/-- `finalAfterSetp` (mov `zero`; `wEmitToken`; setp `pLitBigF`) writes only
    `{zero,tokHi,tok,sbAddr,op,pLitBigF}` (+`gmem`). -/
theorem finalAfterSetp_frame (litLen r : String) (fuel : Nat) (ws : WState)
    (h1 : r ≠ "zero") (h2 : r ≠ "tokHi") (h3 : r ≠ "tok") (h4 : r ≠ "sbAddr")
    (h5 : r ≠ "op") (h6 : r ≠ "pLitBigF") :
    (finalAfterSetp litLen fuel ws).regs r = ws.regs r := by
  simp [finalAfterSetp, wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte,
    WOp.run, SCmp.run, WArg.eval, h1, h2, h3, h4, h5, h6]

/-- The token store advances `op` by one. -/
theorem finalAfterSetp_op (litLen : String) (fuel : Nat) (ws : WState) :
    (finalAfterSetp litLen fuel ws).regs "op" = ws.regs "op" + 1 := by
  simp [finalAfterSetp, wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte,
    WOp.run, SCmp.run, WArg.eval]

/-- The `pLitBigF` guard reflects `15 ≤ litLen`. -/
theorem finalAfterSetp_pLitBig (litLen : String) (fuel : Nat) (ws : WState)
    (h1 : litLen ≠ "zero") (h2 : litLen ≠ "tokHi") (h3 : litLen ≠ "tok") (h4 : litLen ≠ "sbAddr")
    (h5 : litLen ≠ "op") :
    (finalAfterSetp litLen fuel ws).regs "pLitBigF"
      = (if UInt64.ofNat 15 ≤ ws.regs litLen then 1 else 0) := by
  rw [finalAfterSetp]
  have hY : (WStmt.eval fuel (wEmitToken litLen "zero")
      (WStmt.eval fuel (.mov "zero" (.imm 0)) ws)).regs litLen = ws.regs litLen := by
    simp [wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WOp.run,
      WArg.eval, h1, h2, h3, h4, h5]
  generalize hYeq : (WStmt.eval fuel (wEmitToken litLen "zero")
      (WStmt.eval fuel (.mov "zero" (.imm 0)) ws)) = Y at hY ⊢
  simp [WStmt.eval, WState.setReg, SCmp.run, WArg.eval, hY]

/-- The `pLitBigF` literal-LSIC `uif` on `finalAfterSetp` preserves every register
    outside the final-emit scratch set. -/
theorem finalUifOut_frame (litLen r : String) (fuel : Nat) (ws : WState)
    (h1 : r ≠ "zero") (h2 : r ≠ "tokHi") (h3 : r ≠ "tok") (h4 : r ≠ "sbAddr") (h5 : r ≠ "op")
    (h6 : r ≠ "pLitBigF") (h7 : r ≠ "litExtraF") (h8 : r ≠ "lsicC") (h9 : r ≠ "c255") :
    (WStmt.eval fuel (.uif "pLitBigF"
        (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
      (finalAfterSetp litLen fuel ws)).regs r = ws.regs r := by
  simp only [WStmt.eval]
  by_cases hg : ((finalAfterSetp litLen fuel ws).regs "pLitBigF" == 1) = true
  · rw [if_pos hg, show (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ])
        = (WStmt.bin WOp.sub "litExtraF" litLen (WArg.imm 15)).seq (wEmitLSIC "litExtraF") from rfl]
    simp only [WStmt.eval.eq_2]
    rw [wEmitLSIC_frame "litExtraF" r h8 h9 h4 h5 h7]
    simp only [WStmt.eval, WState.setReg, WOp.run, WArg.eval, h7]
    exact finalAfterSetp_frame litLen r fuel ws h1 h2 h3 h4 h5 h6
  · rw [if_neg hg]
    exact finalAfterSetp_frame litLen r fuel ws h1 h2 h3 h4 h5 h6

/-- `finalCpEntry` preserves every register outside the final-emit scratch/output set. -/
theorem finalCpEntry_frame (litStart litLen r : String) (fuel : Nat) (ws : WState)
    (hcs : r ≠ "cpSrcF") (hcd : r ≠ "cpDstF") (h1 : r ≠ "zero") (h2 : r ≠ "tokHi") (h3 : r ≠ "tok")
    (h4 : r ≠ "sbAddr") (h5 : r ≠ "op") (h6 : r ≠ "pLitBigF") (h7 : r ≠ "litExtraF")
    (h8 : r ≠ "lsicC") (h9 : r ≠ "c255") :
    (finalCpEntry litStart litLen fuel ws).regs r = ws.regs r := by
  rw [finalCpEntry]
  have e1 : ∀ (st : WState),
      (WStmt.eval fuel (.bin .add "cpSrcF" "inBase" (.reg litStart)) st).regs r = st.regs r :=
    fun st => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcs]
  have e2 : ∀ (st : WState),
      (WStmt.eval fuel (.bin .add "cpDstF" "outBase" (.reg "op")) st).regs r = st.regs r :=
    fun st => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcd]
  rw [e1, e2]
  exact finalUifOut_frame litLen r fuel ws h1 h2 h3 h4 h5 h6 h7 h8 h9

/-- The literal-copy source `cpSrcF = inBase + litStart`. -/
theorem finalCpEntry_cpSrc (litStart litLen : String) (fuel : Nat) (ws : WState)
    (hcd : litStart ≠ "cpDstF") (h1 : litStart ≠ "zero") (h2 : litStart ≠ "tokHi")
    (h3 : litStart ≠ "tok") (h4 : litStart ≠ "sbAddr") (h5 : litStart ≠ "op")
    (h6 : litStart ≠ "pLitBigF") (h7 : litStart ≠ "litExtraF") (h8 : litStart ≠ "lsicC")
    (h9 : litStart ≠ "c255") :
    (finalCpEntry litStart litLen fuel ws).regs "cpSrcF" = ws.regs "inBase" + ws.regs litStart := by
  have hX : ∀ (r' : String), r' ≠ "cpDstF" → r' ≠ "zero" → r' ≠ "tokHi" → r' ≠ "tok" →
      r' ≠ "sbAddr" → r' ≠ "op" → r' ≠ "pLitBigF" → r' ≠ "litExtraF" → r' ≠ "lsicC" → r' ≠ "c255" →
      (WStmt.eval fuel (.bin .add "cpDstF" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBigF"
            (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
          (finalAfterSetp litLen fuel ws))).regs r' = ws.regs r' := by
    intro r' hcd' a1 a2 a3 a4 a5 a6 a7 a8 a9
    rw [show (WStmt.eval fuel (.bin .add "cpDstF" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBigF"
            (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
          (finalAfterSetp litLen fuel ws))).regs r'
      = (WStmt.eval fuel (.uif "pLitBigF"
            (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
          (finalAfterSetp litLen fuel ws)).regs r' from by
        simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcd']]
    exact finalUifOut_frame litLen r' fuel ws a1 a2 a3 a4 a5 a6 a7 a8 a9
  have hib := hX "inBase" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hls := hX litStart hcd h1 h2 h3 h4 h5 h6 h7 h8 h9
  rw [finalCpEntry]
  generalize (WStmt.eval fuel (.bin .add "cpDstF" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBigF"
            (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
          (finalAfterSetp litLen fuel ws))) = X at hib hls ⊢
  simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hib, hls]

/-- `finalCpEntry` preserves `gmem.size` (all writers are size-preserving). -/
theorem finalCpEntry_gmem_size (litStart litLen : String) (fuel : Nat) (ws : WState) :
    (finalCpEntry litStart litLen fuel ws).gmem.size = ws.gmem.size := by
  rw [finalCpEntry, eval_gmem_size, eval_gmem_size, eval_gmem_size, finalAfterSetp,
    eval_gmem_size, eval_gmem_size, eval_gmem_size]

/-- The `op` value after the literal-LSIC `uif`: `op + 1 + encNib(litLen).length`. -/
theorem finalUifOut_op (litLen : String) (fuel : Nat) (ws : WState)
    (h1 : litLen ≠ "zero") (h2 : litLen ≠ "tokHi") (h3 : litLen ≠ "tok") (h4 : litLen ≠ "sbAddr")
    (h5 : litLen ≠ "op") (h6 : litLen ≠ "pLitBigF")
    (hfuel : ((ws.regs litLen).toNat - 15) / 255 < fuel) :
    (WStmt.eval fuel (.uif "pLitBigF"
        (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
      (finalAfterSetp litLen fuel ws)).regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (LZ4.encNib (ws.regs litLen).toNat).length := by
  have hM_op := finalAfterSetp_op litLen fuel ws
  have hM_ll := finalAfterSetp_frame litLen litLen fuel ws h1 h2 h3 h4 h5 h6
  have hM_pb := finalAfterSetp_pLitBig litLen fuel ws h1 h2 h3 h4 h5
  generalize hMeq : finalAfterSetp litLen fuel ws = M at hM_op hM_ll hM_pb ⊢
  simp only [WStmt.eval]
  by_cases hg : UInt64.ofNat 15 ≤ ws.regs litLen
  · have hgN : 15 ≤ (ws.regs litLen).toNat := by
      rw [UInt64.le_iff_toNat_le] at hg; simpa using hg
    have hpb1 : (M.regs "pLitBigF" == 1) = true := by rw [hM_pb, if_pos hg]; rfl
    rw [if_pos hpb1, show (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ])
        = (WStmt.bin WOp.sub "litExtraF" litLen (WArg.imm 15)).seq (wEmitLSIC "litExtraF") from rfl]
    simp only [WStmt.eval.eq_2]
    generalize hM1 : (WStmt.eval fuel (.bin .sub "litExtraF" litLen (.imm 15)) M) = M1
    have hM1_le : M1.regs "litExtraF" = ws.regs litLen - UInt64.ofNat 15 := by
      rw [← hM1]
      have hstep : (WStmt.eval fuel (.bin .sub "litExtraF" litLen (.imm 15)) M).regs "litExtraF"
          = M.regs litLen - UInt64.ofNat 15 := by
        simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval]
      rw [hstep, hM_ll]
    have hM1_leN : (M1.regs "litExtraF").toNat = (ws.regs litLen).toNat - 15 := by
      rw [hM1_le, UInt64.toNat_sub, show ((UInt64.ofNat 15).toNat) = 15 from by decide,
        show 2 ^ 64 - 15 + (ws.regs litLen).toNat = 2 ^ 64 + ((ws.regs litLen).toNat - 15) from by
          omega, Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (ws.regs litLen).toNat_lt; omega)]
    have hM1_op : M1.regs "op" = ws.regs "op" + 1 := by
      rw [← hM1]
      have hstep : (WStmt.eval fuel (.bin .sub "litExtraF" litLen (.imm 15)) M).regs "op"
          = M.regs "op" := by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval]
      rw [hstep, hM_op]
    obtain ⟨hlsic_op, _, _⟩ := EmitContent.eval_wEmitLSIC "litExtraF" M1 fuel
      (by decide) (by decide) (by decide) (by decide) (by decide) (by rw [hM1_leN]; exact hfuel)
    rw [hlsic_op, hM1_op, hM1_leN, AlgorithmLib.LZ4Imp.ext_length,
      AlgorithmLib.LZ4Imp.encNib_length, if_neg (by omega)]
  · have hgN : (ws.regs litLen).toNat < 15 := by
      rw [UInt64.le_iff_toNat_le] at hg
      simp only [show ((UInt64.ofNat 15).toNat) = 15 from by decide] at hg; omega
    have hpb0 : (M.regs "pLitBigF" == 1) = false := by rw [hM_pb, if_neg hg]; rfl
    rw [if_neg (by rw [hpb0]; exact Bool.false_ne_true), hM_op,
      AlgorithmLib.LZ4Imp.encNib_length, if_pos hgN]
    simp

/-- The literal-copy destination `cpDstF = outBase + op + 1 + encNib(litLen).length`. -/
theorem finalCpEntry_cpDst (litStart litLen : String) (fuel : Nat) (ws : WState)
    (h1 : litLen ≠ "zero") (h2 : litLen ≠ "tokHi") (h3 : litLen ≠ "tok") (h4 : litLen ≠ "sbAddr")
    (h5 : litLen ≠ "op") (h6 : litLen ≠ "pLitBigF")
    (hfuel : ((ws.regs litLen).toNat - 15) / 255 < fuel) :
    (finalCpEntry litStart litLen fuel ws).regs "cpDstF"
      = ws.regs "outBase" + ws.regs "op" + 1
        + UInt64.ofNat (LZ4.encNib (ws.regs litLen).toNat).length := by
  have hob : (WStmt.eval fuel (.uif "pLitBigF"
      (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
    (finalAfterSetp litLen fuel ws)).regs "outBase" = ws.regs "outBase" :=
    finalUifOut_frame litLen "outBase" fuel ws (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
  have hop' := finalUifOut_op litLen fuel ws h1 h2 h3 h4 h5 h6 hfuel
  rw [finalCpEntry]
  generalize hUeq : (WStmt.eval fuel (.uif "pLitBigF"
      (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
    (finalAfterSetp litLen fuel ws)) = U at hob hop'
  simp only [WStmt.eval, WState.setReg, WOp.run, WArg.eval]
  rw [hob, hop']
  ac_rfl

/-- **The `wEmitFinalSeq` side-conditions** discharged from the post-loop layout facts,
    with `fLen = inStride − litAnchor`.  The caller obligation `simSL'_wEmitFinalSeq`
    consumes (mirrors `foundBranch_sideconds`). -/
theorem finalSeq_sideconds (inStride fuel : Nat) (ws : WState) (hlen : inStride < 2 ^ 40)
    (hib0 : (ws.regs "inBase").toNat < 2 ^ 40)
    (hfLen : (ws.regs "fLen").toNat = inStride - (ws.regs "litAnchor").toNat)
    (hla_le : (ws.regs "litAnchor").toNat < inStride)
    (hbud32 : (ws.regs "outBase").toNat + (ws.regs "op").toNat
        + 9 * (inStride - (ws.regs "litAnchor").toNat) < 2 ^ 32)
    (hbudsz : (ws.regs "outBase").toNat + (ws.regs "op").toNat
        + 9 * (inStride - (ws.regs "litAnchor").toNat) ≤ ws.gmem.size)
    (hobLB : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat) (hfuel : inStride ≤ fuel) :
    ((finalAfterSetp "fLen" fuel ws).regs "pLitBigF" = 1 →
        15 ≤ ((finalAfterSetp "fLen" fuel ws).regs "fLen").toNat)
    ∧ ((finalAfterSetp "fLen" fuel ws).regs "pLitBigF" = 1 →
        ((finalAfterSetp "fLen" fuel ws).regs "fLen").toNat < 255 * fuel + 15)
    ∧ ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpDstF").toNat < 2 ^ 32
    ∧ ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpSrcF").toNat < 2 ^ 32
    ∧ ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "fLen").toNat < 2 ^ 32
    ∧ (((finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpDstF").toNat
          + ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "fLen").toNat
          ≤ ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpSrcF").toNat
        ∨ ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpSrcF").toNat
          + ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "fLen").toNat
          ≤ ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpDstF").toNat)
    ∧ (((finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpDstF").toNat
          + ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "fLen").toNat
          ≤ (finalCpEntry "litAnchor" "fLen" fuel ws).gmem.size) := by
  have hAfLen : (finalAfterSetp "fLen" fuel ws).regs "fLen" = ws.regs "fLen" :=
    finalAfterSetp_frame "fLen" "fLen" fuel ws (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)
  have hApb := finalAfterSetp_pLitBig "fLen" fuel ws (by decide) (by decide) (by decide)
    (by decide) (by decide)
  have hlfuel : ((ws.regs "fLen").toNat - 15) / 255 < fuel := by
    have h := Nat.div_le_self ((ws.regs "fLen").toNat - 15) 255; omega
  have hCfLen : (finalCpEntry "litAnchor" "fLen" fuel ws).regs "fLen" = ws.regs "fLen" :=
    finalCpEntry_frame "litAnchor" "fLen" "fLen" fuel ws (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hCcs : (finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpSrcF"
      = ws.regs "inBase" + ws.regs "litAnchor" :=
    finalCpEntry_cpSrc "litAnchor" "fLen" fuel ws (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hCcd : (finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpDstF"
      = ws.regs "outBase" + ws.regs "op" + 1
        + UInt64.ofNat (LZ4.encNib (ws.regs "fLen").toNat).length :=
    finalCpEntry_cpDst "litAnchor" "fLen" fuel ws (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) hlfuel
  have hencle : (LZ4.encNib (ws.regs "fLen").toNat).length ≤ (ws.regs "fLen").toNat :=
    encNib_len_le _
  have hcsN : ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpSrcF").toNat
      = (ws.regs "inBase").toNat + (ws.regs "litAnchor").toNat := by
    rw [hCcs, UInt64.toNat_add, Nat.mod_eq_of_lt (by omega)]
  have hobN : (ws.regs "outBase").toNat < 2 ^ 32 := by omega
  have hopN : (ws.regs "op").toNat < 2 ^ 32 := by omega
  have hencN : (LZ4.encNib (ws.regs "fLen").toNat).length < 2 ^ 64 := by omega
  have hnoof : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
      + (LZ4.encNib (ws.regs "fLen").toNat).length < 2 ^ 64 := by omega
  have s_op1 : (ws.regs "op" + 1).toNat = (ws.regs "op").toNat + 1 := by
    rw [UInt64.toNat_add, show (1 : UInt64).toNat = 1 from by decide, Nat.mod_eq_of_lt (by omega)]
  have s_op1e : (ws.regs "op" + 1 + UInt64.ofNat (LZ4.encNib (ws.regs "fLen").toNat).length).toNat
      = (ws.regs "op").toNat + 1 + (LZ4.encNib (ws.regs "fLen").toNat).length := by
    rw [UInt64.toNat_add, s_op1, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ hencN, Nat.mod_eq_of_lt (by omega)]
  have hcdN : ((finalCpEntry "litAnchor" "fLen" fuel ws).regs "cpDstF").toNat
      = (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
        + (LZ4.encNib (ws.regs "fLen").toNat).length := by
    rw [hCcd, show ws.regs "outBase" + ws.regs "op" + 1
          + UInt64.ofNat (LZ4.encNib (ws.regs "fLen").toNat).length
        = ws.regs "outBase" + (ws.regs "op" + 1
          + UInt64.ofNat (LZ4.encNib (ws.regs "fLen").toNat).length) from by ac_rfl,
      UInt64.toNat_add, s_op1e, Nat.mod_eq_of_lt (by omega)]
    omega
  have hbudkey : 1 + (LZ4.encNib (ws.regs "fLen").toNat).length + (ws.regs "fLen").toNat
      ≤ 9 * (inStride - (ws.regs "litAnchor").toNat) := by omega
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro hpb; rw [hAfLen]; rw [hApb] at hpb
    by_cases hg : UInt64.ofNat 15 ≤ ws.regs "fLen"
    · rw [UInt64.le_iff_toNat_le, show ((UInt64.ofNat 15).toNat) = 15 from by decide] at hg; exact hg
    · rw [if_neg hg] at hpb; exact absurd hpb (by decide)
  · intro _; rw [hAfLen]; omega
  · rw [hcdN]; omega
  · rw [hcsN]; omega
  · rw [hCfLen]; omega
  · right; rw [hcdN, hcsN, hCfLen]; omega
  · rw [show (finalCpEntry "litAnchor" "fLen" fuel ws).gmem.size = ws.gmem.size from
        finalCpEntry_gmem_size _ _ _ _, hcdN, hCfLen]; omega

/-- **The whole `loopC` loop simulates** — `simSL'_measureLoop` instantiated with the
    body simulation `loopCBody_bodySim`, invariant `LoopCQ`, measure `searchPos`. -/
theorem loopC_loop_sim (inStride hashLog F : Nat)
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
      lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM))) :
    ∀ (fuel : Nat) (ss : SState) (ws : WState),
      ss.pc = base → Couple loopR ss ws → MachInv ib ss → LoopCQ inStride ws →
      F ≤ (ws.regs "searchPos").toNat + fuel →
      WhileHalts "loopC" (loopCBodyStmt inStride hashLog) fuel ws →
      ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
        ss'.pc = base + (uwhileEmit "loopC" lHeadC lEndC (loopCBodyEmit inStride hashLog lElse
          lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM)).length ∧
        Couple loopR ss' ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval fuel ws)
          ∧ MachInv ib ss' :=
  simSL'_measureLoop loopR "loopC" lHeadC lEndC (LoopCQ inStride)
    (fun ws => (ws.regs "searchPos").toNat) F (loopCBodyStmt inStride hashLog)
    (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM
      lHM lXM lsicL lsicM) prog base (by decide)
    (loopCBody_bodySim inStride hashLog F lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM
      lEndM lHM lXM lsicL lsicM hLdef hMdef hstride hipos hlen hib40 hhl hFb prog base)
    hseg hlr

/-- The state after `setp loopC` on a post-prologue state satisfies `LoopCQ`. -/
theorem setpLoopC_LoopCQ (inStride : Nat) (hipos : 12 ≤ inStride) (hp64 : inStride < 2 ^ 64)
    (ws : WState) (hop0 : ws.regs "op" = 0) (hla0 : ws.regs "litAnchor" = 0)
    (hsp0 : ws.regs "searchPos" = 0) (hib0 : (ws.regs "inBase").toNat < 2 ^ 40)
    (hobN : (ws.regs "outBase").toNat + 9 * inStride < 2 ^ 32)
    (hsize : (ws.regs "outBase").toNat + 9 * inStride ≤ ws.gmem.size)
    (hobLB : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat) (fuel : Nat) :
    LoopCQ inStride
      ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval fuel ws) := by
  have hsetp : (WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval fuel ws
      = ws.setReg "loopC" (if ws.regs "searchPos" < UInt64.ofNat (inStride - 12) then 1 else 0) := by
    simp [WStmt.eval, WArg.eval, SCmp.run]
  rw [hsetp]
  have hspN : (ws.regs "searchPos").toNat = 0 := by rw [hsp0]; rfl
  have h0N : (0 : UInt64).toNat = 0 := rfl
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [WState.setReg, String.reduceEq, if_false]; exact hib0
  · simp [WState.setReg, hla0, hsp0]
  · simp only [WState.setReg, if_true, hsp0, hspN, h0N]; simp
  · simp only [WState.setReg, String.reduceEq, if_false]; exact hobLB
  · simp only [WState.setReg, String.reduceEq, if_false, hop0, hla0, h0N, Nat.add_zero,
      Nat.sub_zero]; exact hobN
  · simp only [WState.setReg, String.reduceEq, if_false, hop0, hla0, h0N, Nat.add_zero,
      Nat.sub_zero]; exact hsize

/-- **The full `bodyEncodePrefix` simulates from a post-prologue state.**  `SReaches`
    glue of `setp loopC` → the `loopC` loop → `mov fLen` / `sub fLen` → `wEmitFinalSeq`;
    this is the machine-side `hbodySim` (for the concrete coupling set `loopR`). -/
theorem bodyEncodePrefix_sim (inStride hashLog : Nat)
    (lHeadC lEndC lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM
     lElseF lEndF lHF lXF cpHF cpXF : String)
    (lsicL lsicM lsicBodyF : List SInstr)
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
    (hFdef : lsicBodyF = ([.mov "c255" (SArg.imm 255)]
      ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
        ++ ([.bin .sub "litExtraF" "litExtraF" (.imm 255)]
          ++ [.setp .ge "lsicC" "litExtraF" (.imm 255)]))))
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride) (hlen : inStride < 2 ^ 40)
    (hp64 : inStride < 2 ^ 64) (hhl : hashLog ≤ 32)
    (ws : WState) (hop0 : ws.regs "op" = 0) (hla0 : ws.regs "litAnchor" = 0)
    (hsp0 : ws.regs "searchPos" = 0) (hib0 : (ws.regs "inBase").toNat < 2 ^ 40)
    (hobN : (ws.regs "outBase").toNat + 9 * inStride < 2 ^ 32)
    (hsize : (ws.regs "outBase").toNat + 9 * inStride ≤ ws.gmem.size)
    (hobLB : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat)
    (prog : Array SInstr) (base : Nat) (ss : SState)
    (hpc : ss.pc = base)
    (hseg : SegAt prog base
      ([.setp .lt "loopC" "searchPos" (.imm (inStride - 12))]
        ++ (uwhileEmit "loopC" lHeadC lEndC (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE
              lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM)
          ++ (([.mov "fLen" (.imm inStride)] : List SInstr)
            ++ (([.bin .sub "fLen" "fLen" (.reg "litAnchor")] : List SInstr)
              ++ wEmitFinalSeqEmit "litAnchor" "fLen" lElseF lEndF lHF lXF cpHF cpXF lsicBodyF)))))
    (hlr : LabelsResolve prog base
      ([.setp .lt "loopC" "searchPos" (.imm (inStride - 12))]
        ++ (uwhileEmit "loopC" lHeadC lEndC (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE
              lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM)
          ++ (([.mov "fLen" (.imm inStride)] : List SInstr)
            ++ (([.bin .sub "fLen" "fLen" (.reg "litAnchor")] : List SInstr)
              ++ wEmitFinalSeqEmit "litAnchor" "fLen" lElseF lEndF lHF lXF cpHF cpXF lsicBodyF)))))
    (hc : Couple loopR ss ws) (hmi : MachInv ib ss) (hib40 : ib < 2 ^ 40) :
    ∃ (n : Nat) (ss' : SState), SReaches prog n ss ss' ∧
      Couple loopR ss' ((bodyEncodePrefix inStride hashLog).eval (inStride + 34 * inStride) ws)
        ∧ MachInv ib ss' ∧
      ss'.pc = base +
        (([.setp .lt "loopC" "searchPos" (.imm (inStride - 12))] : List SInstr)
          ++ (uwhileEmit "loopC" lHeadC lEndC (loopCBodyEmit inStride hashLog lElse lEnd lHE lEE
                lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM)
            ++ (([.mov "fLen" (.imm inStride)] : List SInstr)
              ++ (([.bin .sub "fLen" "fLen" (.reg "litAnchor")] : List SInstr)
                ++ wEmitFinalSeqEmit "litAnchor" "fLen" lElseF lEndF lHF lXF cpHF cpXF
                    lsicBodyF)))).length := by
  obtain ⟨F, hFval⟩ : ∃ x, inStride + 34 * inStride = x := ⟨_, rfl⟩
  rw [hFval]
  -- Step 1: `setp loopC`.
  have hsegR := hseg.append_right
  have hlrR := hlr.append_right
  obtain ⟨n1, ss1, hr1, hpc1, hc1, hmi1⟩ :=
    simSL'_setp loopR .lt "loopC" "searchPos" (.imm (inStride - 12)) (by decide)
      (fun n h => by cases h) (by decide) prog base ss ws F hpc hseg.append_left hlr.append_left
      hc hmi
  have hLoopCQ1 : LoopCQ inStride
      ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws) :=
    setpLoopC_LoopCQ inStride hipos hp64 ws hop0 hla0 hsp0 hib0 hobN hsize hobLB F
  -- the post-`setp` `searchPos` is still `0`.
  have hsp1 : (((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws).regs
      "searchPos").toNat = 0 := by
    simp only [WStmt.eval, WState.setReg, WArg.eval, SCmp.run, String.reduceEq, if_false]
    rw [hsp0]; rfl
  have hpc1' : ss1.pc = base + 1 := by simpa using hpc1
  -- Step 2: the `loopC` loop.
  have hla1 : (((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws).regs
      "litAnchor").toNat ≤ inStride - 5 := by
    simp only [WStmt.eval, WState.setReg, WArg.eval, SCmp.run, String.reduceEq, if_false]
    rw [hla0]; simp
  have hFge : inStride ≤ F := by rw [← hFval]; omega
  obtain ⟨n2, ss2, hr2, hpc2, hc2, hmi2⟩ :=
    loopC_loop_sim inStride hashLog F lHeadC lEndC lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX
      lElseM lEndM lHM lXM lsicL lsicM hLdef hMdef hstride hipos hlen hib40 hhl (by rw [← hFval]; omega)
      prog (base + 1) (by simpa using hsegR.append_left) (by simpa using hlrR.append_left)
      F ss1 _ hpc1' hc1 hmi1 hLoopCQ1 (by rw [hsp1]; omega)
      (loopC_halts inStride hashLog hstride hipos hp64 F _ hLoopCQ1 (by rw [hsp1]; omega))
  have hLoopCQ2 := loopC_loop_preservesInv inStride hashLog hstride hipos hp64 F _ hLoopCQ1
    (by rw [hsp1]; omega)
  have hla2 := loopC_litAnchor_bound inStride hashLog hstride hipos hp64 F _ hLoopCQ1 hla1
    (by rw [hsp1]; omega)
  obtain ⟨hib2, hlaSp2, hlc2, hobLB2, hbud2, hbudsz2⟩ := hLoopCQ2
  -- Step 3 & 4: `mov fLen inStride` ; `sub fLen fLen litAnchor`.
  have hsegR2 := hsegR.append_right
  have hlrR2 := hlrR.append_right
  obtain ⟨n3, ss3, hr3, hpc3, hc3, hmi3⟩ :=
    simSL'_mov loopR "fLen" (.imm inStride) (fun n h => by cases h) (by decide)
      prog _ ss2 _ F hpc2 hsegR2.append_left hlrR2.append_left hc2 hmi2
  have hsegR3 := hsegR2.append_right
  have hlrR3 := hlrR2.append_right
  obtain ⟨n4, ss4, hr4, hpc4, hc4, hmi4⟩ :=
    simSL'_bin loopR .sub "fLen" "fLen" (.reg "litAnchor") (by decide) (fun n h => by cases h; decide)
      (by decide) prog _ ss3 _ F hpc3 hsegR3.append_left hlrR3.append_left hc3 hmi3
  -- The post-arith eval state `ws4`; frame everything but `fLen` back to the loop-exit `ws2`.
  have hframe4 : ∀ r, r ≠ "fLen" →
      ((WStmt.bin WOp.sub "fLen" "fLen" (WArg.reg "litAnchor")).eval F
        ((WStmt.mov "fLen" (WArg.imm inStride)).eval F
          ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
            ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)))).regs r
      = ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
          ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)).regs r :=
    fun r hr => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hr]
  have hgm4 : ((WStmt.bin WOp.sub "fLen" "fLen" (WArg.reg "litAnchor")).eval F
        ((WStmt.mov "fLen" (WArg.imm inStride)).eval F
          ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
            ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)))).gmem
      = ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
          ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)).gmem := by
    simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, WState.stgByte]
  have hfLen4 : (((WStmt.bin WOp.sub "fLen" "fLen" (WArg.reg "litAnchor")).eval F
        ((WStmt.mov "fLen" (WArg.imm inStride)).eval F
          ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
            ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)))).regs
        "fLen").toNat
      = inStride - (((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
          ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)).regs
          "litAnchor").toNat := by
    have hval : ((WStmt.bin WOp.sub "fLen" "fLen" (WArg.reg "litAnchor")).eval F
        ((WStmt.mov "fLen" (WArg.imm inStride)).eval F
          ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
            ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)))).regs
        "fLen"
        = UInt64.ofNat inStride
          - ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
              ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)).regs
              "litAnchor" := by
      simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval]
    rw [hval, UInt64.toNat_sub, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega),
      show 2 ^ 64 - (((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
            ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)).regs
            "litAnchor").toNat + inStride
        = 2 ^ 64 + (inStride - (((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
            ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)).regs
            "litAnchor").toNat) from by omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by omega)]
  have hfLen4' : (((WStmt.bin WOp.sub "fLen" "fLen" (WArg.reg "litAnchor")).eval F
        ((WStmt.mov "fLen" (WArg.imm inStride)).eval F
          ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
            ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)))).regs
        "fLen").toNat
      = inStride - (((WStmt.bin WOp.sub "fLen" "fLen" (WArg.reg "litAnchor")).eval F
          ((WStmt.mov "fLen" (WArg.imm inStride)).eval F
            ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
              ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)))).regs
          "litAnchor").toNat := by
    rw [hframe4 "litAnchor" (by decide)]; exact hfLen4
  -- discharge the 7 `wEmitFinalSeq` side-conditions from the loop-exit `LoopCQ` + `fLen`.
  obtain ⟨fc1, fc2, fc3, fc4, fc5, fc6, fc7⟩ :=
    finalSeq_sideconds inStride F _ hlen (by rw [hframe4 "inBase" (by decide)]; exact hib2)
      hfLen4' (by rw [hframe4 "litAnchor" (by decide)]; omega)
      (by rw [hframe4 "outBase" (by decide), hframe4 "op" (by decide),
          hframe4 "litAnchor" (by decide)]; exact hbud2)
      (by rw [hframe4 "outBase" (by decide), hframe4 "op" (by decide),
          hframe4 "litAnchor" (by decide), hgm4]; exact hbudsz2)
      (by rw [hframe4 "outBase" (by decide), hframe4 "inBase" (by decide)]; exact hobLB2) hFge
  -- Step 5: `wEmitFinalSeq`.
  obtain ⟨n5, ss5, hr5, hpc5, hc5, hmi5⟩ :=
    simSL'_wEmitFinalSeq loopR "litAnchor" "fLen" lElseF lEndF lHF lXF cpHF cpXF
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) lsicBodyF hFdef
      prog _ ss4 _ F hpc4 hsegR3.append_right hlrR3.append_right hc4 hmi4
      fc1 fc2 fc3 fc4 fc5 fc6 fc7
  -- Assemble: `bodyEncodePrefix.eval = wEmitFinalSeq.eval (sub.eval (mov.eval (uwhile.eval (setp.eval ws))))`.
  refine ⟨n1 + (n2 + (n3 + (n4 + n5))), ss5,
    sreaches_trans prog n1 _ _ _ _ hr1
      (sreaches_trans prog n2 _ _ _ _ hr2
        (sreaches_trans prog n3 _ _ _ _ hr3
          (sreaches_trans prog n4 n5 _ _ _ hr4 hr5))), ?_, hmi5, ?_⟩
  · have hbeq : (bodyEncodePrefix inStride hashLog).eval F ws
        = (wEmitFinalSeq "litAnchor" "fLen").eval F
            ((WStmt.bin WOp.sub "fLen" "fLen" (WArg.reg "litAnchor")).eval F
              ((WStmt.mov "fLen" (WArg.imm inStride)).eval F
                ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval F
                  ((WStmt.setp SCmp.lt "loopC" "searchPos" (WArg.imm (inStride - 12))).eval F ws)))) := by
      simp only [bodyEncodePrefix, loopCBodyStmt, foundBranchStmt, wseq, WStmt.eval.eq_2]
    rw [hbeq]; exact hc5
  · rw [hpc5]
    simp only [List.length_append, List.length_cons, List.length_nil]
    omega

end LoopAsm

/-- **The sstep-floor (machine-level) roundtrip theorem.**  Given that the machine
    `sstep`-simulates the compressor body's encode-producing prefix — i.e. from a
    machine state `ss` coupled to the post-prologue `ws`, execution reaches some `ss'`
    coupled to `bodyEncodePrefix.eval` — reading the output window from the MACHINE's
    `gmem` and decompressing recovers the exact input.  Combines the machine
    simulation (`hbodySim` — the `SimSL'` obligation the coop/loop assembly discharges)
    with the eval-level `warpKernelDSL_body_roundtrips`, via `Couple.gmem`
    (`ss'.gmem = (bodyEncodePrefix.eval …).gmem`).  This lands the measured kernel's
    compressor on the `sstep`/PTX floor. -/
theorem warpKernelDSL_sstep_roundtrips (R : List String) (inStride hashLog : Nat)
    (ws : WState) (ss ss' : AlgorithmLib.LZ4Simt.SState) (F : Nat)
    (hstride : inStride ≤ 65536) (hp64 : inStride < 2 ^ 64) (hipos : 12 ≤ inStride)
    (hop0 : ws.regs "op" = 0) (hla0 : ws.regs "litAnchor" = 0)
    (hsp0 : ws.regs "searchPos" = 0) (hib0 : (ws.regs "inBase").toNat < 2 ^ 40)
    (hobN : (ws.regs "outBase").toNat + 9 * inStride < 2 ^ 64)
    (hsize : (ws.regs "outBase").toNat + 9 * inStride ≤ ws.gmem.size)
    (hdisj : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat)
    (hinB : (ws.regs "inBase").toNat + inStride < 2 ^ 64)
    (hencLen : (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length ≤ 9 * inStride)
    (hF : F = inStride + 34 * inStride)
    -- the machine sstep-simulates the body (the `SimSL'` obligation for the assembly):
    (hbodySim : Couple R ss' ((bodyEncodePrefix inStride hashLog).eval F ws)) :
    -- reading the output window from the MACHINE's gmem and decompressing = input.
    AlgorithmLib.LZ4Imp.decompress
        ((List.range (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length).map
          (fun i => ss'.gmem.getD ((ws.regs "outBase").toNat + i) 0))
        (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride).length
      = some (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) := by
  -- The machine's final gmem = the eval's final gmem (via the coupling).
  have hgmemEq : ss'.gmem = ((bodyEncodePrefix inStride hashLog).eval F ws).gmem := hbodySim.gmem
  rw [hgmemEq, hF]
  -- Now identical to the eval-level theorem's window read.
  exact warpKernelDSL_body_roundtrips inStride hashLog ws hstride hp64 hipos hop0 hla0 hsp0 hib0
    hobN hsize hdisj hinB hencLen

section
open AlgorithmLib.LZ4Simt

/-- **The sstep-floor roundtrip, with `hbodySim` DISCHARGED.**  From a post-prologue
    machine state `ss` coupled to `ws`, the machine executes the encode-producing body's
    emit (`bodyEncodePrefix_sim`) to some `ss'`, and reading `ss'`'s output window and
    decompressing recovers the exact input.  No `SimSL'` obligation is assumed — the
    coupling is built by the loop + final-seq assembly. -/
theorem warpKernelDSL_sstep_roundtrips_discharged (inStride hashLog : Nat)
    (lHeadC lEndC lElse lEnd lHE lEE lElseL lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM
     lElseF lEndF lHF lXF cpHF cpXF : String) (lsicL lsicM lsicBodyF : List SInstr)
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
    (hFdef : lsicBodyF = ([.mov "c255" (SArg.imm 255)]
      ++ (([.bin .add "sbAddr" "outBase" (.reg "op")]
          ++ ([.stg "sbAddr" "c255"] ++ [.bin .add "op" "op" (.imm 1)]))
        ++ ([.bin .sub "litExtraF" "litExtraF" (.imm 255)]
          ++ [.setp .ge "lsicC" "litExtraF" (.imm 255)]))))
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride) (hlen : inStride < 2 ^ 40)
    (hib40 : ib < 2 ^ 40)
    (hp64 : inStride < 2 ^ 64) (hhl : hashLog ≤ 32)
    (ws : WState) (hop0 : ws.regs "op" = 0) (hla0 : ws.regs "litAnchor" = 0)
    (hsp0 : ws.regs "searchPos" = 0) (hib0 : (ws.regs "inBase").toNat < 2 ^ 40)
    (hobN : (ws.regs "outBase").toNat + 9 * inStride < 2 ^ 32)
    (hsize : (ws.regs "outBase").toNat + 9 * inStride ≤ ws.gmem.size)
    (hobLB : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat)
    (hdisj : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat)
    (hinB : (ws.regs "inBase").toNat + inStride < 2 ^ 64)
    (hencLen : (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
        (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length ≤ 9 * inStride)
    (prog : Array AlgorithmLib.LZ4Simt.SInstr) (base : Nat) (ss : AlgorithmLib.LZ4Simt.SState)
    (hpc : ss.pc = base)
    (hseg : AlgorithmLib.LZ4WarpDSL.SegAt prog base
      ([.setp .lt "loopC" "searchPos" (.imm (inStride - 12))]
        ++ (AlgorithmLib.LZ4WarpDSL.uwhileEmit "loopC" lHeadC lEndC
              (AlgorithmLib.LZ4WarpDSL.loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL
                lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM)
          ++ (([.mov "fLen" (.imm inStride)] : List AlgorithmLib.LZ4Simt.SInstr)
            ++ (([.bin .sub "fLen" "fLen" (.reg "litAnchor")] : List AlgorithmLib.LZ4Simt.SInstr)
              ++ AlgorithmLib.LZ4WarpDSL.wEmitFinalSeqEmit "litAnchor" "fLen" lElseF lEndF lHF lXF
                  cpHF cpXF lsicBodyF)))))
    (hlr : AlgorithmLib.LZ4WarpDSL.LabelsResolve prog base
      ([.setp .lt "loopC" "searchPos" (.imm (inStride - 12))]
        ++ (AlgorithmLib.LZ4WarpDSL.uwhileEmit "loopC" lHeadC lEndC
              (AlgorithmLib.LZ4WarpDSL.loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL lEndL
                lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM)
          ++ (([.mov "fLen" (.imm inStride)] : List AlgorithmLib.LZ4Simt.SInstr)
            ++ (([.bin .sub "fLen" "fLen" (.reg "litAnchor")] : List AlgorithmLib.LZ4Simt.SInstr)
              ++ AlgorithmLib.LZ4WarpDSL.wEmitFinalSeqEmit "litAnchor" "fLen" lElseF lEndF lHF lXF
                  cpHF cpXF lsicBodyF)))))
    (hc : AlgorithmLib.LZ4WarpDSL.Couple loopR ss ws) (hmi : AlgorithmLib.LZ4WarpDSL.MachInv ib ss) :
    ∃ (n : Nat) (ss' : AlgorithmLib.LZ4Simt.SState),
      AlgorithmLib.LZ4Simt.SReaches prog n ss ss' ∧
      AlgorithmLib.LZ4Imp.decompress
          ((List.range (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
              (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length).map
            (fun i => ss'.gmem.getD ((ws.regs "outBase").toNat + i) 0))
          (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride).length
        = some (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) ∧
      AlgorithmLib.LZ4WarpDSL.Couple loopR ss'
        ((bodyEncodePrefix inStride hashLog).eval (inStride + 34 * inStride) ws) ∧
      AlgorithmLib.LZ4WarpDSL.MachInv ib ss' ∧
      ((bodyEncodePrefix inStride hashLog).eval (inStride + 34 * inStride) ws).regs "outBase"
        = ws.regs "outBase" ∧
      ((bodyEncodePrefix inStride hashLog).eval (inStride + 34 * inStride) ws).regs "op"
        = UInt64.ofNat (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length ∧
      ((bodyEncodePrefix inStride hashLog).eval (inStride + 34 * inStride) ws).gmem.size
        = ws.gmem.size ∧
      ss'.pc = base +
        (([.setp .lt "loopC" "searchPos" (.imm (inStride - 12))] : List AlgorithmLib.LZ4Simt.SInstr)
          ++ (AlgorithmLib.LZ4WarpDSL.uwhileEmit "loopC" lHeadC lEndC
                (AlgorithmLib.LZ4WarpDSL.loopCBodyEmit inStride hashLog lElse lEnd lHE lEE lElseL
                  lEndL lHL lXL cpH cpX lElseM lEndM lHM lXM lsicL lsicM)
            ++ (([.mov "fLen" (.imm inStride)] : List AlgorithmLib.LZ4Simt.SInstr)
              ++ (([.bin .sub "fLen" "fLen" (.reg "litAnchor")] : List AlgorithmLib.LZ4Simt.SInstr)
                ++ AlgorithmLib.LZ4WarpDSL.wEmitFinalSeqEmit "litAnchor" "fLen" lElseF lEndF lHF
                    lXF cpHF cpXF lsicBodyF)))).length := by
  obtain ⟨n, ss', hr, hc', hmi', hpcF⟩ :=
    bodyEncodePrefix_sim inStride hashLog lHeadC lEndC lElse lEnd lHE lEE lElseL lEndL lHL lXL
      cpH cpX lElseM lEndM lHM lXM lElseF lEndF lHF lXF cpHF cpXF lsicL lsicM lsicBodyF
      hLdef hMdef hFdef hstride hipos hlen hp64 hhl ws hop0 hla0 hsp0 hib0 hobN hsize hobLB
      prog base ss hpc hseg hlr hc hmi hib40
  refine ⟨n, ss', hr, ?_, hc', hmi',
    (compressorBody_output_eq inStride hashLog ws hstride hp64 hipos hop0 hla0 hsp0 hib0
      (by omega) hsize hdisj hinB hencLen).2.2,
    (compressorBody_output_eq inStride hashLog ws hstride hp64 hipos hop0 hla0 hsp0 hib0
      (by omega) hsize hdisj hinB hencLen).2.1,
    (by rw [(compressorBody_output_eq inStride hashLog ws hstride hp64 hipos hop0 hla0 hsp0 hib0
      (by omega) hsize hdisj hinB hencLen).1, EmitContent.putBytesU_size]), hpcF⟩
  exact warpKernelDSL_sstep_roundtrips loopR inStride hashLog ws ss ss' (inStride + 34 * inStride)
    hstride hp64 hipos hop0 hla0 hsp0 hib0 (by omega) hsize hdisj hinB hencLen rfl hc'

end

end AlgorithmLib.LZ4WarpDSL
