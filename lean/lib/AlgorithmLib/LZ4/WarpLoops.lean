module
public import AlgorithmLib.LZ4.EmitContent
public import AlgorithmLib.LZ4.ByteLayer
meta import AlgorithmLib.LZ4.EmitContent
meta import AlgorithmLib.LZ4.ByteLayer
public import AlgorithmLib.LZ4.EncodeLen
meta import AlgorithmLib.LZ4.EncodeLen
public import AlgorithmLib.LZ4.Ptx
meta import AlgorithmLib.LZ4.Ptx
import all AlgorithmLib.LZ4.WarpDSL
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The compressor's loops and branches, as the warp runs them

The statements of the compressor's main loop body (`loopCBodyStmt`), its found
match's branch (`foundBranchStmt`) and the extend loop (`extBodyStmt`), with
what their runs keep: the main loop's invariant (`LoopCQ`), the extend loop's
(`extInv`) and that it halts (`extLoop_halts`), and the frames and bounds of
the states between a match sequence's statements.
-/

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib.LZ4WarpFind
open AlgorithmLib.LZ4Ptx (toNat_ofNat_lt u64_add_ofNat u64_sub_ofNat)

/-- Every `WStmt.eval` step preserves `gmem.size`: the only gmem writers are
    `stgByte` (`set!`), `coopCopy` (`copyGmem`), and the coop primitives — all
    size-preserving; every other constructor leaves `gmem` untouched. -/
theorem eval_gmem_size : ∀ (fuel : Nat) (s : WStmt) (st : WState),
    (WStmt.eval fuel s st).gmem.size = st.gmem.size := by
  intro fuel s st
  induction fuel, s, st using WStmt.eval.induct <;>
    simp_all [WStmt.eval, WState.setReg, WState.stgByte,
      evalCoopWindow, evalCoopWindowGo_gmem, evalCoopExtendStep,
      copyGmem_size, Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]

/-- `coopWindow` only writes `found`/`p0`/`cand0` (and `smem`); every other register —
    `op`, `outBase`, `inBase`, `litAnchor`, `searchPos` — is preserved. -/
theorem evalCoopWindow_regs_frame (found p0 cand0 sp : String)
    (inStride searchLim hashLog tbl : Nat) (st : WState) (r : String)
    (hf : r ≠ found) (hp : r ≠ p0) (hc : r ≠ cand0) :
    (evalCoopWindow found p0 cand0 sp inStride searchLim hashLog tbl st).regs r = st.regs r := by
  rw [evalCoopWindow_eq_go, evalCoopWindowGo_regs]
  split <;> simp [WState.setReg, hf, hp, hc]

/-- `coopWindow`'s `found` register: `1` on a window hit, `0` otherwise. -/
theorem coopWindow_found_val (inStride hashLog : Nat) (ws : WState) :
    (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws).regs
        "found"
      = (match AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) (ws.regs "searchPos").toNat with
         | some _ => 1 | none => 0) := by
  rw [evalCoopWindow_eq_go, evalCoopWindowGo_regs]
  cases AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) (ws.regs "searchPos").toNat with
  | none => simp [WState.setReg]
  | some pc => obtain ⟨p, c⟩ := pc; simp [WState.setReg]

/-- **The found-branch preconditions, from the window semantics.**  When `coopWindow`
    reports a match (`found = 1`), the eval-level `window` returned `some (p, c)`, whose
    soundness (`window_sound`: `c < p` and `p < searchLim = inStride - 12`) pins
    `cand0 = c < p = p0` and `p0 + 4 ≤ inStride - 5 = endCap`.  These discharge
    `extInv_of_foundMovs`'s hypotheses inside the loopC body. -/
theorem coopWindow_found_bounds (inStride hashLog : Nat) (ws : WState)
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride)
    (hfound1 : ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12)
        hashLog 0 ws).regs "found") = 1) :
    ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws).regs
        "cand0").toNat
      < ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws).regs
        "p0").toNat
    ∧ ((evalCoopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0 ws).regs
        "p0").toNat + 4 ≤ inStride - 5 := by
  rw [evalCoopWindow_eq_go] at hfound1 ⊢
  cases hw : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12) (ws.regs "searchPos").toNat with
  | none =>
      rw [hw, evalCoopWindowGo_regs] at hfound1
      simp [WState.setReg] at hfound1
  | some pc =>
      obtain ⟨p, c⟩ := pc
      obtain ⟨hsp, hpsl, hcp, _⟩ := AlgorithmLib.LZ4WarpFind.window_sound
        (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12)
        (ws.regs "searchPos").toNat p c hw
      have hpN : p < 2 ^ 64 := by omega
      have hcN : c < 2 ^ 64 := by omega
      have hp0 : (evalCoopWindowGo "found" "p0" "cand0" "searchPos" hashLog 0 (inStride - 12)
          (ws.regs "inBase").toNat ((tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat
            (min (ws.regs "searchPos").toNat (inStride - 4))).elim 0xffffffff UInt64.ofNat) (some (p, c)) ws).regs "p0" = UInt64.ofNat p := by
        rw [evalCoopWindowGo_regs]; simp [WState.setReg]
      have hcand0 : (evalCoopWindowGo "found" "p0" "cand0" "searchPos" hashLog 0 (inStride - 12)
          (ws.regs "inBase").toNat ((tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat
            (min (ws.regs "searchPos").toNat (inStride - 4))).elim 0xffffffff UInt64.ofNat) (some (p, c)) ws).regs "cand0" = UInt64.ofNat c := by
        rw [evalCoopWindowGo_regs]; simp [WState.setReg]
      rw [hp0, hcand0, toNat_ofNat_lt p hpN, toNat_ofNat_lt c hcN]
      exact ⟨hcp, by omega⟩

/-- Widening a byte through `UInt64.ofNat _.toNat` is injective, so a `BEq` on the
    widened values matches the `BEq` on the original bytes. -/
theorem byteEqBridge (a b : UInt8) :
    (UInt64.ofNat a.toNat == UInt64.ofNat b.toNat) = (a == b) := by
  have ha : a.toNat < 2 ^ 64 := by have := a.toNat_lt; omega
  have hb : b.toNat < 2 ^ 64 := by have := b.toNat_lt; omega
  rw [Bool.eq_iff_iff, beq_iff_eq, beq_iff_eq, ← UInt64.toNat_inj,
    toNat_ofNat_lt a.toNat ha, toNat_ofNat_lt b.toNat hb]
  constructor
  · intro h; exact UInt8.toNat_inj.mp h
  · intro h; exact congrArg UInt8.toNat h

/-- The extend loop's invariant: `ecR`/`ec1` pin the endCap constants, the
    candidate lies strictly before the match position, and the running match
    length keeps the extend position within `endCap`.  It discharges the coop
    leaf's `hboundW`/`hbytesW` and is preserved by `ml += leadingGood`
    (`leadingGood_pos_le`). -/
def extInv (inStride endCap : Nat) : WState → Prop :=
  fun ws => ws.regs "ecR" = UInt64.ofNat endCap
    ∧ ws.regs "ec1" = UInt64.ofNat (endCap - 1)
    ∧ (ws.regs "cand0").toNat < (ws.regs "p0").toNat
    ∧ (ws.regs "p0").toNat + (ws.regs "ml").toNat ≤ endCap

/-- The four found-branch setup statements (`ecR := endCap; ec1 := endCap-1;
    ml := 4; extC := ml ≥ 0`) establish `extInv`, given the match-position facts
    `cand0 < p0` (candidate before match) and `p0 + 4 ≤ endCap` (from `window_sound`:
    `p < searchLim = inStride-12 ⟹ p+4 ≤ inStride-8 < endCap`). -/
theorem extInv_of_foundMovs (endCap : Nat) (inStride : Nat) (fuel : Nat) (ws : WState)
    (hcand : (ws.regs "cand0").toNat < (ws.regs "p0").toNat)
    (hp4 : (ws.regs "p0").toNat + 4 ≤ endCap) :
    extInv inStride endCap
      ((wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
               .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws) := by
  have hml4 : ((wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
      .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws).regs "ml"
      = UInt64.ofNat 4 := by simp [wseq, WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [wseq, WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  · simp [wseq, WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  · have hcand' : ((wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
        .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws).regs "cand0"
        = ws.regs "cand0" ∧ ((wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
        .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws).regs "p0"
        = ws.regs "p0" := by
      constructor <;> simp [wseq, WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
    rw [hcand'.1, hcand'.2]; exact hcand
  · have hp0' : ((wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
        .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws).regs "p0"
        = ws.regs "p0" := by simp [wseq, WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
    rw [hp0', hml4, show ((UInt64.ofNat 4)).toNat = 4 from by decide]; exact hp4

/-- The extend body preserves `extInv` (only `adv`/`ml`/`extC` change; `ml += leadingGood`
    keeps `p0 + ml ≤ endCap` by `leadingGood_pos_le`).  Extracted from
    `bodySimInv_extCBody`'s preservation half, at the pure eval level. -/
theorem extBody_preserves_extInv (inStride endCap fuel : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hInv : extInv inStride endCap ws) :
    extInv inStride endCap ((extBodyStmt inStride endCap).eval fuel ws) := by
  obtain ⟨hecR, hec1, hcandlt, hpml⟩ := hInv
  simp only [extBodyStmt]
  have hbodyReg : ∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" →
      ((wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws).regs r
        = ws.regs r := by
    intro r hadvr hmlr hextr
    simp only [wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run, SCmp.run, WArg.eval,
      if_neg hextr, if_neg hmlr, if_neg hadvr]
  have hmlReg : ((wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws).regs "ml"
        = ws.regs "ml" + UInt64.ofNat
          (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32) := by
    simp [wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run, SCmp.run, WArg.eval]
  have hpres : (ws.regs "p0").toNat + ((ws.regs "ml").toNat
        + AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32) ≤ endCap := by
    have := AlgorithmLib.LZ4WarpSched.leadingGood_pos_le (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap 32 (ws.regs "ml").toNat hpml
    omega
  have hmlN : (ws.regs "ml" + UInt64.ofNat
        (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32)).toNat
      = (ws.regs "ml").toNat + AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32 := by
    rw [UInt64.toNat_add, toNat_ofNat_lt _ (by omega), Nat.mod_eq_of_lt (by omega)]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hbodyReg "ecR" (by decide) (by decide) (by decide)]; exact hecR
  · rw [hbodyReg "ec1" (by decide) (by decide) (by decide)]; exact hec1
  · rw [hbodyReg "cand0" (by decide) (by decide) (by decide),
      hbodyReg "p0" (by decide) (by decide) (by decide)]; exact hcandlt
  · rw [hbodyReg "p0" (by decide) (by decide) (by decide), hmlReg, hmlN]; exact hpres

/-- The extend `uwhile` preserves `extInv` (each iteration does, via
    `extBody_preserves_extInv`).  Gives `p0 + ml ≤ endCap` on the loop result. -/
theorem extInv_uwhile (inStride endCap : Nat) (hendCap : endCap = inStride - 5)
    (hlen : inStride < 2 ^ 40) :
    ∀ (fuel : Nat) (ws : WState), extInv inStride endCap ws →
      extInv inStride endCap
        ((WStmt.uwhile "extC" (extBodyStmt inStride endCap)).eval fuel ws) := by
  intro fuel
  induction fuel with
  | zero => intro ws hInv; simpa [WStmt.eval] using hInv
  | succ fuel ih =>
      intro ws hInv
      simp only [WStmt.eval]
      by_cases hb : (ws.regs "extC" == 1) = true
      · rw [if_pos hb]
        exact ih _ (extBody_preserves_extInv inStride endCap fuel ws hendCap hlen hInv)
      · rw [if_neg hb]; exact hInv

/-- The extend body only grows `ml` (`ml += leadingGood ≥ 0`), so `4 ≤ ml` is
    preserved.  Needs `extInv` for the no-overflow bound on the new `ml`. -/
theorem extBody_ml_ge4 (inStride endCap fuel : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hInv : extInv inStride endCap ws) (h4 : 4 ≤ (ws.regs "ml").toNat) :
    4 ≤ (((extBodyStmt inStride endCap).eval fuel ws).regs "ml").toNat := by
  obtain ⟨_, _, _, hpml⟩ := hInv
  simp only [extBodyStmt]
  have hlg := AlgorithmLib.LZ4WarpSched.leadingGood_pos_le (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
    (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap 32 (ws.regs "ml").toNat hpml
  have hmlReg : ((wseq [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"), .setp .eq "extC" "adv" (.imm 32) ]).eval fuel ws).regs "ml"
        = ws.regs "ml" + UInt64.ofNat
          (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32) := by
    simp [wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run, SCmp.run, WArg.eval]
  have hmlN : (ws.regs "ml" + UInt64.ofNat
        (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32)).toNat
      = (ws.regs "ml").toNat + AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32 := by
    rw [UInt64.toNat_add, toNat_ofNat_lt _ (by omega), Nat.mod_eq_of_lt (by omega)]
  rw [hmlReg, hmlN]; omega

/-- The extend `uwhile` preserves `4 ≤ ml` (each iteration grows `ml`). -/
theorem uwhile_ml_ge4 (inStride endCap : Nat) (hendCap : endCap = inStride - 5)
    (hlen : inStride < 2 ^ 40) :
    ∀ (fuel : Nat) (ws : WState), extInv inStride endCap ws → 4 ≤ (ws.regs "ml").toNat →
      4 ≤ (((WStmt.uwhile "extC" (extBodyStmt inStride endCap)).eval fuel ws).regs "ml").toNat := by
  intro fuel
  induction fuel with
  | zero => intro ws _ h4; simpa [WStmt.eval] using h4
  | succ fuel ih =>
      intro ws hInv h4
      simp only [WStmt.eval]
      by_cases hb : (ws.regs "extC" == 1) = true
      · rw [if_pos hb]
        exact ih _ (extBody_preserves_extInv inStride endCap fuel ws hendCap hlen hInv)
          (extBody_ml_ge4 inStride endCap fuel ws hendCap hlen hInv h4)
      · rw [if_neg hb]; exact h4

/-- The extend loop terminates: while it continues (`adv == 32`), `ml` grows by
    32 and `p0 + ml` stays `≤ endCap` (`extInv`), so the gap `endCap - (p0+ml)`
    strictly decreases by 32 each iteration.  Any `fuel` with `endCap ≤ 32*fuel +
    (p0+ml)` suffices. -/
theorem extLoop_halts (inStride endCap : Nat) (hendCap : endCap = inStride - 5)
    (hlen : inStride < 2 ^ 40) :
    ∀ (fuel : Nat) (ws : WState), extInv inStride endCap ws →
      endCap + 32 < 32 * fuel + ((ws.regs "p0").toNat + (ws.regs "ml").toNat) →
      WhileHalts "extC" (extBodyStmt inStride endCap) fuel ws := by
  intro fuel
  induction fuel with
  | zero =>
      intro ws hInv hmeas
      -- fuel 0 ⟹ measure `endCap+32 < p0+ml`, contradicting `extInv`'s `p0+ml ≤ endCap`.
      obtain ⟨_, _, _, hpml⟩ := hInv
      omega
  | succ fuel ih =>
      intro ws hInv hmeas
      simp only [WhileHalts]
      by_cases hb : (ws.regs "extC" == 1) = true
      · rw [if_pos hb]
        obtain ⟨hecR, hec1, hcandlt, hpml⟩ := hInv
        have hLGle := AlgorithmLib.LZ4WarpSched.leadingGood_pos_le (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap 32 (ws.regs "ml").toNat hpml
        -- eval readout: p0/cand0/ecR/ec1 unchanged, ml += LG.
        have hbReg : ∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" →
            ((extBodyStmt inStride endCap).eval fuel ws).regs r = ws.regs r := by
          intro r ha hm he
          simp only [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run,
            SCmp.run, WArg.eval, if_neg he, if_neg hm, if_neg ha]
        have hmlReg : ((extBodyStmt inStride endCap).eval fuel ws).regs "ml"
            = ws.regs "ml" + UInt64.ofNat
              (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32) := by
          simp [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run, SCmp.run,
            WArg.eval]
        have hmlN : (((extBodyStmt inStride endCap).eval fuel ws).regs "ml").toNat
            = (ws.regs "ml").toNat + AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32 := by
          rw [hmlReg, UInt64.toNat_add, toNat_ofNat_lt _ (by omega), Nat.mod_eq_of_lt (by omega)]
        have hInv' : extInv inStride endCap ((extBodyStmt inStride endCap).eval fuel ws) := by
          refine ⟨?_, ?_, ?_, ?_⟩
          · rw [hbReg "ecR" (by decide) (by decide) (by decide)]; exact hecR
          · rw [hbReg "ec1" (by decide) (by decide) (by decide)]; exact hec1
          · rw [hbReg "cand0" (by decide) (by decide) (by decide),
              hbReg "p0" (by decide) (by decide) (by decide)]; exact hcandlt
          · rw [hbReg "p0" (by decide) (by decide) (by decide), hmlN]; omega
        by_cases hLG32 : AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32 = 32
        · -- continuing step: measure drops by 32.
          apply ih ((extBodyStmt inStride endCap).eval fuel ws) hInv'
          rw [hbReg "p0" (by decide) (by decide) (by decide), hmlN, hLG32]
          omega
        · -- `adv ≠ 32` ⟹ next guard `extC = 0` ⟹ halts immediately.
          have hne : (UInt64.ofNat
              (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32)
              == (32 : UInt64)) = false := by
            rw [beq_eq_false_iff_ne]
            intro hEq
            apply hLG32
            have h2 := congrArg UInt64.toNat hEq
            rw [toNat_ofNat_lt _ (by omega), show ((32 : UInt64).toNat) = 32 from by decide] at h2
            exact h2
          have hLGne : ¬ UInt64.ofNat
              (AlgorithmLib.LZ4WarpSched.leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                (ws.regs "p0").toNat (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat 32)
              = (32 : UInt64) := beq_eq_false_iff_ne.mp hne
          have hextC0 : ((extBodyStmt inStride endCap).eval fuel ws).regs "extC" = 0 := by
            simp [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run,
              SCmp.run, WArg.eval, hLGne]
          -- The strict measure `endCap < 32*(fuel+1)+(p0+ml)` with `p0+ml ≤ endCap`
          -- forces `fuel ≥ 1`, so one more `WhileHalts` unfold hits `else True`.
          obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := by
            cases fuel with
            | zero => exact absurd hmeas (by omega)
            | succ f => exact ⟨f, rfl⟩
          simp only [WhileHalts, hextC0]
          rw [if_neg (by decide)]
          trivial
      · rw [if_neg hb]; trivial

/-- The LSIC loop terminates: while it continues (`lsicC == 1`, and the guard
    invariant `lsicC=1 → n ≥ 255` holds), `n` drops by 255 each iteration, so any
    `fuel` with `n.toNat < 255 * fuel` suffices.  The guard invariant is preserved
    by the body's trailing `setp`. -/
theorem lsicLoop_halts (n : String) (hnc : n ≠ "c255") (hnop : n ≠ "op")
    (hnsb : n ≠ "sbAddr") (hnl : n ≠ "lsicC") :
    ∀ (fuel : Nat) (ws : WState),
      (ws.regs "lsicC" = 1 → 255 ≤ (ws.regs n).toNat) →
      (ws.regs n).toNat < 255 * fuel →
      WhileHalts "lsicC" (EmitContent.lsicBody n) fuel ws := by
  intro fuel
  induction fuel with
  | zero => intro ws _ hmeas; omega
  | succ fuel ih =>
      intro ws hguard hmeas
      simp only [WhileHalts]
      by_cases hb : (ws.regs "lsicC" == 1) = true
      · rw [if_pos hb]
        have hlsic1 : ws.regs "lsicC" = 1 := by
          have := hb; rw [beq_iff_eq] at this; exact this
        have hbig : 255 ≤ (ws.regs n).toNat := hguard hlsic1
        -- eval readout: `n' = n - 255` (UInt64).
        have hnReg : ((EmitContent.lsicBody n).eval fuel ws).regs n
            = ws.regs n - UInt64.ofNat 255 := by
          simp [EmitContent.lsicBody, wseq, WStmt.eval, wStoreByte, WState.setReg, WState.stgByte,
            WOp.run, SCmp.run, WArg.eval, hnl, hnc, hnop, hnsb, Ne.symm hnl, Ne.symm hnsb]
        have hnN : (((EmitContent.lsicBody n).eval fuel ws).regs n).toNat
            = (ws.regs n).toNat - 255 := by
          rw [hnReg, UInt64.toNat_sub, show ((UInt64.ofNat 255).toNat) = 255 from by decide]
          rw [show 2 ^ 64 - 255 + (ws.regs n).toNat = 2 ^ 64 + ((ws.regs n).toNat - 255) from by
            omega, Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (ws.regs n).toNat_lt; omega)]
        -- new lsicC = (n' ≥ 255) as 1/0.
        have hlsicVal : ((EmitContent.lsicBody n).eval fuel ws).regs "lsicC"
            = (if (UInt64.ofNat 255) ≤ (ws.regs n - UInt64.ofNat 255) then 1 else 0) := by
          simp [EmitContent.lsicBody, wseq, WStmt.eval, wStoreByte, WState.setReg, WState.stgByte,
            WOp.run, SCmp.run, WArg.eval, hnl, hnc, hnop, hnsb, Ne.symm hnl, Ne.symm hnsb]
        -- guard invariant preserved: `lsicC=1 → n' ≥ 255`.
        have hguard' : ((EmitContent.lsicBody n).eval fuel ws).regs "lsicC" = 1 →
            255 ≤ (((EmitContent.lsicBody n).eval fuel ws).regs n).toNat := by
          intro hg1
          rw [hnN]
          rw [hlsicVal] at hg1
          by_cases hge : (UInt64.ofNat 255) ≤ (ws.regs n - UInt64.ofNat 255)
          · rw [UInt64.le_iff_toNat_le, show ((UInt64.ofNat 255).toNat) = 255 from by decide,
              UInt64.toNat_sub, show ((UInt64.ofNat 255).toNat) = 255 from by decide,
              show 2 ^ 64 - 255 + (ws.regs n).toNat = 2 ^ 64 + ((ws.regs n).toNat - 255) from by
                omega, Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (ws.regs n).toNat_lt; omega)]
              at hge
            omega
          · rw [if_neg hge] at hg1; exact absurd hg1 (by decide)
        apply ih ((EmitContent.lsicBody n).eval fuel ws) hguard'
        rw [hnN]; omega
      · rw [if_neg hb]; trivial

/-- The LSIC loop body writes only `c255`/`sbAddr`/`op`/`n`/`lsicC` (+`gmem`); any
    other register is preserved. -/
theorem lsicBody_frame (n r : String)
    (h1 : r ≠ "c255") (h2 : r ≠ "sbAddr") (h3 : r ≠ "op") (h4 : r ≠ n) (h5 : r ≠ "lsicC")
    (fuel : Nat) (ws : WState) : ((EmitContent.lsicBody n).eval fuel ws).regs r = ws.regs r := by
  simp [EmitContent.lsicBody, wseq, WStmt.eval, wStoreByte, WState.setReg, WState.stgByte, WOp.run,
    SCmp.run, WArg.eval, h1, h2, h3, h4, h5]

/-- The LSIC `uwhile` preserves every register outside `{c255,sbAddr,op,n,lsicC}`. -/
theorem lsicUwhile_frame (n r : String)
    (h1 : r ≠ "c255") (h2 : r ≠ "sbAddr") (h3 : r ≠ "op") (h4 : r ≠ n) (h5 : r ≠ "lsicC") :
    ∀ (fuel : Nat) (ws : WState),
      ((WStmt.uwhile "lsicC" (EmitContent.lsicBody n)).eval fuel ws).regs r = ws.regs r := by
  intro fuel
  induction fuel with
  | zero => intro ws; simp [WStmt.eval]
  | succ fuel ih =>
    intro ws; simp only [WStmt.eval]
    by_cases hb : (ws.regs "lsicC" == 1) = true
    · rw [if_pos hb, ih, lsicBody_frame n r h1 h2 h3 h4 h5]
    · rw [if_neg hb]

/-- `wEmitLSIC n` preserves every register outside `{lsicC,c255,sbAddr,op,n}`. -/
theorem wEmitLSIC_frame (n r : String)
    (h1 : r ≠ "lsicC") (h2 : r ≠ "c255") (h3 : r ≠ "sbAddr") (h4 : r ≠ "op") (h5 : r ≠ n)
    (fuel : Nat) (ws : WState) : ((wEmitLSIC n).eval fuel ws).regs r = ws.regs r := by
  have hw : wEmitLSIC n = (WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)).seq
      ((WStmt.uwhile "lsicC" (EmitContent.lsicBody n)).seq (wStoreByte n)) := by
    rw [wEmitLSIC]; rfl
  have e1 : ∀ st : WState, ((wStoreByte n).eval fuel st).regs r = st.regs r := fun st => by
    simp [wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WOp.run, WArg.eval, h3, h4]
  have e2 : ∀ st : WState,
      ((WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)).eval fuel st).regs r = st.regs r := fun st => by
    simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run, h1]
  rw [hw]
  simp only [WStmt.eval.eq_2]
  rw [e1, lsicUwhile_frame n r h2 h3 h4 h5 h1, e2]

/-- Eval of `wEmitFinalSeq` decomposed as the right-nested `.seq` chain, so a
    simulation can peel it one statement at a time (keeps `wEmitToken`/`wEmitLSIC`
    internals folded — no kernel blow-up). -/
theorem wEmitFinalSeq_eval_seq (litStart litLen : String) (fuel : Nat) (ws : WState) :
    (wEmitFinalSeq litStart litLen).eval fuel ws
      = (WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
        (WStmt.eval fuel (.coopCopy "cpDstF" "cpSrcF" litLen)
        (WStmt.eval fuel (.bin .add "cpSrcF" "inBase" (.reg litStart))
        (WStmt.eval fuel (.bin .add "cpDstF" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBigF"
            (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
        (WStmt.eval fuel (.setp .ge "pLitBigF" litLen (.imm 15))
        (WStmt.eval fuel (wEmitToken litLen "zero")
        (WStmt.eval fuel (.mov "zero" (.imm 0)) ws)))))))) := by
  simp only [wEmitFinalSeq, wseq, WStmt.eval]

/-- The `wEmitFinalSeq` eval state right after the `pLitBigF` guard `setp` (before
    the LSIC uif) — where the LSIC side-conditions are checked. -/
def finalAfterSetp (litLen : String) (fuel : Nat) (ws : WState) : WState :=
  WStmt.eval fuel (.setp .ge "pLitBigF" litLen (.imm 15))
    (WStmt.eval fuel (wEmitToken litLen "zero")
      (WStmt.eval fuel (.mov "zero" (.imm 0)) ws))

/-- The `wEmitFinalSeq` eval state at the literal `coopCopy` entry (after prefix,
    LSIC uif, and the two address computes) — where the copy layout is checked. -/
def finalCpEntry (litStart litLen : String) (fuel : Nat) (ws : WState) : WState :=
  WStmt.eval fuel (.bin .add "cpSrcF" "inBase" (.reg litStart))
    (WStmt.eval fuel (.bin .add "cpDstF" "outBase" (.reg "op"))
      (WStmt.eval fuel (.uif "pLitBigF"
          (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
        (finalAfterSetp litLen fuel ws)))

/-- Eval of `wEmitMatchSeq` as the right-nested `.seq` chain. -/
theorem wEmitMatchSeq_eval_seq (litStart litLen off ml : String) (fuel : Nat) (ws : WState) :
    (wEmitMatchSeq litStart litLen off ml).eval fuel ws
      = (WStmt.eval fuel (.uif "pMatBig"
            (wseq [ .bin .sub "matExtra" "mlm" (.imm 15), wEmitLSIC "matExtra" ]) .skip)
        (WStmt.eval fuel (.setp .ge "pMatBig" "mlm" (.imm 15))
        (WStmt.eval fuel (wStoreByte "offHi")
        (WStmt.eval fuel (.bin .band "offHi" "offHi" (.imm 255))
        (WStmt.eval fuel (.bin .shr "offHi" off (.imm 8))
        (WStmt.eval fuel (wStoreByte "offLo")
        (WStmt.eval fuel (.bin .band "offLo" off (.imm 255))
        (WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
        (WStmt.eval fuel (.coopCopy "cpDst" "cpSrc" litLen)
        (WStmt.eval fuel (.bin .add "cpSrc" "inBase" (.reg litStart))
        (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
        (WStmt.eval fuel (.setp .ge "pLitBig" litLen (.imm 15))
        (WStmt.eval fuel (wEmitToken litLen "tokLo")
        (WStmt.eval fuel (.bin .min "tokLo" "mlm" (.imm 15))
        (WStmt.eval fuel (.bin .sub "mlm" ml (.imm 4)) ws)))))))))))))))) := by
  simp only [wEmitMatchSeq, wseq, WStmt.eval]

/-- State after the `pLitBig` guard `setp` (LSIC side-conditions checked here). -/
def matchAfterSetp (litLen ml : String) (fuel : Nat) (ws : WState) : WState :=
  WStmt.eval fuel (.setp .ge "pLitBig" litLen (.imm 15))
    (WStmt.eval fuel (wEmitToken litLen "tokLo")
    (WStmt.eval fuel (.bin .min "tokLo" "mlm" (.imm 15))
    (WStmt.eval fuel (.bin .sub "mlm" ml (.imm 4)) ws)))

/-- State at the match literal `coopCopy` entry (copy layout checked here). -/
def matchCpEntry (litStart litLen ml : String) (fuel : Nat) (ws : WState) : WState :=
  WStmt.eval fuel (.bin .add "cpSrc" "inBase" (.reg litStart))
    (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op"))
    (WStmt.eval fuel (.uif "pLitBig"
        (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
      (matchAfterSetp litLen ml fuel ws)))

/-- `matchAfterSetp` (sub `mlm`; min `tokLo`; `wEmitToken`; setp `pLitBig`) is
    straight-line and writes only `{mlm,tokLo,tokHi,tok,sbAddr,op,pLitBig}` (+`gmem`). -/
theorem matchAfterSetp_frame (litLen ml r : String) (fuel : Nat) (ws : WState)
    (h1 : r ≠ "mlm") (h2 : r ≠ "tokLo") (h3 : r ≠ "tokHi") (h4 : r ≠ "tok")
    (h5 : r ≠ "sbAddr") (h6 : r ≠ "op") (h7 : r ≠ "pLitBig") :
    (matchAfterSetp litLen ml fuel ws).regs r = ws.regs r := by
  simp [matchAfterSetp, wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte,
    WOp.run, SCmp.run, WArg.eval, h1, h2, h3, h4, h5, h6, h7]

/-- The `pLitBig` literal-LSIC `uif` on `matchAfterSetp` preserves every register
    outside the match-emit scratch set. -/
theorem matchUifOut_frame (litLen ml r : String) (fuel : Nat) (ws : WState)
    (h1 : r ≠ "mlm") (h2 : r ≠ "tokLo") (h3 : r ≠ "tokHi") (h4 : r ≠ "tok")
    (h5 : r ≠ "sbAddr") (h6 : r ≠ "op") (h7 : r ≠ "pLitBig")
    (h8 : r ≠ "litExtra") (h9 : r ≠ "lsicC") (h10 : r ≠ "c255") :
    (WStmt.eval fuel (.uif "pLitBig"
        (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
      (matchAfterSetp litLen ml fuel ws)).regs r = ws.regs r := by
  simp only [WStmt.eval]
  by_cases hg : ((matchAfterSetp litLen ml fuel ws).regs "pLitBig" == 1) = true
  · rw [if_pos hg, show (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ])
        = (WStmt.bin WOp.sub "litExtra" litLen (WArg.imm 15)).seq (wEmitLSIC "litExtra") from rfl]
    simp only [WStmt.eval.eq_2]
    rw [wEmitLSIC_frame "litExtra" r h9 h10 h5 h6 h8]
    simp only [WStmt.eval, WState.setReg, WOp.run, WArg.eval, h8]
    exact matchAfterSetp_frame litLen ml r fuel ws h1 h2 h3 h4 h5 h6 h7
  · rw [if_neg hg]
    exact matchAfterSetp_frame litLen ml r fuel ws h1 h2 h3 h4 h5 h6 h7

/-- `matchCpEntry` preserves every register outside the match-emit scratch/output set;
    in particular `inBase`, `litAnchor` (= `litStart`), `outBase`, `litLen` survive. -/
theorem matchCpEntry_frame (litStart litLen ml r : String) (fuel : Nat) (ws : WState)
    (hcs : r ≠ "cpSrc") (hcd : r ≠ "cpDst") (h1 : r ≠ "mlm") (h2 : r ≠ "tokLo")
    (h3 : r ≠ "tokHi") (h4 : r ≠ "tok") (h5 : r ≠ "sbAddr") (h6 : r ≠ "op") (h7 : r ≠ "pLitBig")
    (h8 : r ≠ "litExtra") (h9 : r ≠ "lsicC") (h10 : r ≠ "c255") :
    (matchCpEntry litStart litLen ml fuel ws).regs r = ws.regs r := by
  rw [matchCpEntry]
  have e1 : ∀ (st : WState),
      (WStmt.eval fuel (.bin .add "cpSrc" "inBase" (.reg litStart)) st).regs r = st.regs r :=
    fun st => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcs]
  have e2 : ∀ (st : WState),
      (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op")) st).regs r = st.regs r :=
    fun st => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcd]
  rw [e1, e2]
  exact matchUifOut_frame litLen ml r fuel ws h1 h2 h3 h4 h5 h6 h7 h8 h9 h10

/-- The literal-copy source address `cpSrc = inBase + litStart` at the copy entry. -/
theorem matchCpEntry_cpSrc (litStart litLen ml : String) (fuel : Nat) (ws : WState)
    (hcd : litStart ≠ "cpDst") (h1 : litStart ≠ "mlm") (h2 : litStart ≠ "tokLo")
    (h3 : litStart ≠ "tokHi") (h4 : litStart ≠ "tok") (h5 : litStart ≠ "sbAddr")
    (h6 : litStart ≠ "op") (h7 : litStart ≠ "pLitBig") (h8 : litStart ≠ "litExtra")
    (h9 : litStart ≠ "lsicC") (h10 : litStart ≠ "c255") :
    (matchCpEntry litStart litLen ml fuel ws).regs "cpSrc"
      = ws.regs "inBase" + ws.regs litStart := by
  have hX : ∀ (r' : String), r' ≠ "cpDst" → r' ≠ "mlm" → r' ≠ "tokLo" → r' ≠ "tokHi" →
      r' ≠ "tok" → r' ≠ "sbAddr" → r' ≠ "op" → r' ≠ "pLitBig" → r' ≠ "litExtra" →
      r' ≠ "lsicC" → r' ≠ "c255" →
      (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
          (matchAfterSetp litLen ml fuel ws))).regs r' = ws.regs r' := by
    intro r' hcd' a1 a2 a3 a4 a5 a6 a7 a8 a9 a10
    rw [show (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
          (matchAfterSetp litLen ml fuel ws))).regs r'
      = (WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
          (matchAfterSetp litLen ml fuel ws)).regs r' from by
        simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcd']]
    exact matchUifOut_frame litLen ml r' fuel ws a1 a2 a3 a4 a5 a6 a7 a8 a9 a10
  have hib := hX "inBase" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
  have hls := hX litStart hcd h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  rw [matchCpEntry]
  generalize (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op"))
        (WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
          (matchAfterSetp litLen ml fuel ws))) = X at hib hls ⊢
  simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hib, hls]

/-- The `pLitBig` guard equals `litLen ≥ 15` (as `1`/`0`). -/
theorem matchAfterSetp_pLitBig (litLen ml : String) (fuel : Nat) (ws : WState)
    (h1 : litLen ≠ "mlm") (h2 : litLen ≠ "tokLo") (h3 : litLen ≠ "tokHi") (h4 : litLen ≠ "tok")
    (h5 : litLen ≠ "sbAddr") (h6 : litLen ≠ "op") :
    (matchAfterSetp litLen ml fuel ws).regs "pLitBig"
      = (if UInt64.ofNat 15 ≤ ws.regs litLen then 1 else 0) := by
  rw [matchAfterSetp]
  have hY : (WStmt.eval fuel (wEmitToken litLen "tokLo")
      (WStmt.eval fuel (.bin .min "tokLo" "mlm" (.imm 15))
      (WStmt.eval fuel (.bin .sub "mlm" ml (.imm 4)) ws))).regs litLen = ws.regs litLen := by
    simp [wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WOp.run,
      WArg.eval, h1, h2, h3, h4, h5, h6]
  generalize hYeq : (WStmt.eval fuel (wEmitToken litLen "tokLo")
      (WStmt.eval fuel (.bin .min "tokLo" "mlm" (.imm 15))
      (WStmt.eval fuel (.bin .sub "mlm" ml (.imm 4)) ws))) = Y at hY ⊢
  simp [WStmt.eval, WState.setReg, SCmp.run, WArg.eval, hY]

/-- State after the `pMatBig` guard `setp` (2nd LSIC side-conditions checked here). -/
def matchAfterMatSetp (litStart litLen off ml : String) (fuel : Nat) (ws : WState) : WState :=
  WStmt.eval fuel (.setp .ge "pMatBig" "mlm" (.imm 15))
    (WStmt.eval fuel (wStoreByte "offHi")
    (WStmt.eval fuel (.bin .band "offHi" "offHi" (.imm 255))
    (WStmt.eval fuel (.bin .shr "offHi" off (.imm 8))
    (WStmt.eval fuel (wStoreByte "offLo")
    (WStmt.eval fuel (.bin .band "offLo" off (.imm 255))
    (WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
    (WStmt.eval fuel (.coopCopy "cpDst" "cpSrc" litLen)
      (matchCpEntry litStart litLen ml fuel ws))))))))

/-- `mlm = ml - 4` after the match-sequence prefix. -/
theorem matchAfterSetp_mlm (litLen ml : String) (fuel : Nat) (ws : WState) (hm : ml ≠ "mlm") :
    (matchAfterSetp litLen ml fuel ws).regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
  rw [matchAfterSetp]
  generalize hZeq : (WStmt.eval fuel (.bin .sub "mlm" ml (.imm 4)) ws) = Z at *
  have hZmlm : Z.regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
    rw [← hZeq]; simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hm]
  simp [WStmt.eval, wEmitToken, wStoreByte, wseq, WState.setReg, WState.stgByte, WOp.run,
    SCmp.run, WArg.eval, hZmlm]

/-- `matchCpEntry` preserves every register the two adds / literal-LSIC `uif` don't
    write, relative to `matchAfterSetp` (so e.g. `mlm` survives to the copy entry). -/
theorem matchCpEntry_over (litStart litLen ml r : String) (fuel : Nat) (ws : WState)
    (hcs : r ≠ "cpSrc") (hcd : r ≠ "cpDst") (h5 : r ≠ "sbAddr") (h6 : r ≠ "op")
    (h8 : r ≠ "litExtra") (h9 : r ≠ "lsicC") (h10 : r ≠ "c255") :
    (matchCpEntry litStart litLen ml fuel ws).regs r = (matchAfterSetp litLen ml fuel ws).regs r := by
  rw [matchCpEntry]
  have e1 : ∀ (st : WState),
      (WStmt.eval fuel (.bin .add "cpSrc" "inBase" (.reg litStart)) st).regs r = st.regs r :=
    fun st => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcs]
  have e2 : ∀ (st : WState),
      (WStmt.eval fuel (.bin .add "cpDst" "outBase" (.reg "op")) st).regs r = st.regs r :=
    fun st => by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, hcd]
  rw [e1, e2]
  simp only [WStmt.eval]
  by_cases hg : ((matchAfterSetp litLen ml fuel ws).regs "pLitBig" == 1) = true
  · rw [if_pos hg, show (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ])
        = (WStmt.bin WOp.sub "litExtra" litLen (WArg.imm 15)).seq (wEmitLSIC "litExtra") from rfl]
    simp only [WStmt.eval.eq_2]
    rw [wEmitLSIC_frame "litExtra" r h9 h10 h5 h6 h8]
    simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval, h8]
  · rw [if_neg hg]

/-- The token store advances `op` by one; `outBase`/`inBase` unchanged. -/
theorem matchAfterSetp_op (litLen ml : String) (fuel : Nat) (ws : WState) :
    (matchAfterSetp litLen ml fuel ws).regs "op" = ws.regs "op" + 1 := by
  simp [matchAfterSetp, wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte,
    WOp.run, SCmp.run, WArg.eval]

/-- The `op` value after the literal-LSIC `uif`: `op + 1 + encNib(litLen).length`
    (token byte + the `pLitBig` continuation bytes).  Uses `EmitContent.eval_wEmitLSIC`
    for the LSIC advance; the two branches of `pLitBig` unify because
    `encNib(litLen).length = 0` when `litLen < 15`. -/
theorem matchUifOut_op (litLen ml : String) (fuel : Nat) (ws : WState)
    (hml : litLen ≠ "mlm") (htl : litLen ≠ "tokLo") (hth : litLen ≠ "tokHi") (htk : litLen ≠ "tok")
    (hsb : litLen ≠ "sbAddr") (hop : litLen ≠ "op") (hpb : litLen ≠ "pLitBig")
    (hfuel : ((ws.regs litLen).toNat - 15) / 255 < fuel) :
    (WStmt.eval fuel (.uif "pLitBig"
        (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
      (matchAfterSetp litLen ml fuel ws)).regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (LZ4.encNib (ws.regs litLen).toNat).length := by
  have hM_op := matchAfterSetp_op litLen ml fuel ws
  have hM_ll := matchAfterSetp_frame litLen ml litLen fuel ws hml htl hth htk hsb hop hpb
  have hM_pb := matchAfterSetp_pLitBig litLen ml fuel ws hml htl hth htk hsb hop
  generalize hMeq : matchAfterSetp litLen ml fuel ws = M at hM_op hM_ll hM_pb ⊢
  simp only [WStmt.eval]
  by_cases hg : UInt64.ofNat 15 ≤ ws.regs litLen
  · have hgN : 15 ≤ (ws.regs litLen).toNat := by
      rw [UInt64.le_iff_toNat_le] at hg; simpa using hg
    have hpb1 : (M.regs "pLitBig" == 1) = true := by rw [hM_pb, if_pos hg]; rfl
    rw [if_pos hpb1, show (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ])
        = (WStmt.bin WOp.sub "litExtra" litLen (WArg.imm 15)).seq (wEmitLSIC "litExtra") from rfl]
    simp only [WStmt.eval.eq_2]
    generalize hM1 : (WStmt.eval fuel (.bin .sub "litExtra" litLen (.imm 15)) M) = M1
    have hM1_le : M1.regs "litExtra" = ws.regs litLen - UInt64.ofNat 15 := by
      rw [← hM1]
      have hstep : (WStmt.eval fuel (.bin .sub "litExtra" litLen (.imm 15)) M).regs "litExtra"
          = M.regs litLen - UInt64.ofNat 15 := by
        simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval]
      rw [hstep, hM_ll]
    have hM1_leN : (M1.regs "litExtra").toNat = (ws.regs litLen).toNat - 15 := by
      rw [hM1_le, UInt64.toNat_sub, show ((UInt64.ofNat 15).toNat) = 15 from by decide,
        show 2 ^ 64 - 15 + (ws.regs litLen).toNat = 2 ^ 64 + ((ws.regs litLen).toNat - 15) from by
          omega, Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (ws.regs litLen).toNat_lt; omega)]
    have hM1_op : M1.regs "op" = ws.regs "op" + 1 := by
      rw [← hM1]
      have hstep : (WStmt.eval fuel (.bin .sub "litExtra" litLen (.imm 15)) M).regs "op"
          = M.regs "op" := by simp [WStmt.eval, WState.setReg, WOp.run, WArg.eval]
      rw [hstep, hM_op]
    obtain ⟨hlsic_op, _, _⟩ := EmitContent.eval_wEmitLSIC "litExtra" M1 fuel
      (by decide) (by decide) (by decide) (by decide) (by decide) (by rw [hM1_leN]; exact hfuel)
    rw [hlsic_op, hM1_op, hM1_leN, AlgorithmLib.LZ4Imp.ext_length,
      AlgorithmLib.LZ4Imp.encNib_length, if_neg (by omega)]
  · have hgN : (ws.regs litLen).toNat < 15 := by
      rw [UInt64.le_iff_toNat_le] at hg
      simp only [show ((UInt64.ofNat 15).toNat) = 15 from by decide] at hg; omega
    have hpb0 : (M.regs "pLitBig" == 1) = false := by rw [hM_pb, if_neg hg]; rfl
    rw [if_neg (by rw [hpb0]; exact Bool.false_ne_true), hM_op,
      AlgorithmLib.LZ4Imp.encNib_length, if_pos hgN]
    simp

/-- The literal-copy destination `cpDst = outBase + op + 1 + encNib(litLen).length`
    at the copy entry (token + literal-LSIC bytes already emitted). -/
theorem matchCpEntry_cpDst (litStart litLen ml : String) (fuel : Nat) (ws : WState)
    (hml : litLen ≠ "mlm") (htl : litLen ≠ "tokLo") (hth : litLen ≠ "tokHi") (htk : litLen ≠ "tok")
    (hsb : litLen ≠ "sbAddr") (hop : litLen ≠ "op") (hpb : litLen ≠ "pLitBig")
    (hfuel : ((ws.regs litLen).toNat - 15) / 255 < fuel) :
    (matchCpEntry litStart litLen ml fuel ws).regs "cpDst"
      = ws.regs "outBase" + ws.regs "op" + 1 + UInt64.ofNat (LZ4.encNib (ws.regs litLen).toNat).length := by
  have hob : (WStmt.eval fuel (.uif "pLitBig"
      (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
    (matchAfterSetp litLen ml fuel ws)).regs "outBase" = ws.regs "outBase" :=
    matchUifOut_frame litLen ml "outBase" fuel ws (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hop' := matchUifOut_op litLen ml fuel ws hml htl hth htk hsb hop hpb hfuel
  rw [matchCpEntry]
  generalize hUeq : (WStmt.eval fuel (.uif "pLitBig"
      (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
    (matchAfterSetp litLen ml fuel ws)) = U at hob hop'
  simp only [WStmt.eval, WState.setReg, WOp.run, WArg.eval]
  rw [hob, hop']
  ac_rfl

/-- `mlm = ml - 4` still holds after the offset-store block (nothing there writes `mlm`). -/
theorem matchAfterMatSetp_mlm (litStart litLen off ml : String) (fuel : Nat) (ws : WState)
    (hm : ml ≠ "mlm") :
    (matchAfterMatSetp litStart litLen off ml fuel ws).regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
  rw [matchAfterMatSetp]
  generalize hCeq : matchCpEntry litStart litLen ml fuel ws = C at *
  have hCmlm : C.regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
    rw [← hCeq, matchCpEntry_over litStart litLen ml "mlm" fuel ws (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide),
      matchAfterSetp_mlm litLen ml fuel ws hm]
  simp [WStmt.eval, wStoreByte, wseq, WState.setReg, WState.stgByte, WOp.run, SCmp.run,
    WArg.eval, hCmlm]

/-- The `pMatBig` guard equals `mlm ≥ 15` (`mlm = ml - 4`). -/
theorem matchAfterMatSetp_pMatBig (litStart litLen off ml : String) (fuel : Nat) (ws : WState)
    (hm : ml ≠ "mlm") :
    (matchAfterMatSetp litStart litLen off ml fuel ws).regs "pMatBig"
      = (if UInt64.ofNat 15 ≤ ws.regs ml - UInt64.ofNat 4 then 1 else 0) := by
  rw [matchAfterMatSetp]
  generalize hCeq : matchCpEntry litStart litLen ml fuel ws = C at *
  have hCmlm : C.regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
    rw [← hCeq, matchCpEntry_over litStart litLen ml "mlm" fuel ws (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide),
      matchAfterSetp_mlm litLen ml fuel ws hm]
  have hmlmChain : (WStmt.eval fuel (wStoreByte "offHi")
      (WStmt.eval fuel (.bin .band "offHi" "offHi" (.imm 255))
      (WStmt.eval fuel (.bin .shr "offHi" off (.imm 8))
      (WStmt.eval fuel (wStoreByte "offLo")
      (WStmt.eval fuel (.bin .band "offLo" off (.imm 255))
      (WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
      (WStmt.eval fuel (.coopCopy "cpDst" "cpSrc" litLen) C))))))).regs "mlm"
      = ws.regs ml - UInt64.ofNat 4 := by
    simp [WStmt.eval, wStoreByte, wseq, WState.setReg, WState.stgByte, WOp.run, WArg.eval, hCmlm]
  generalize hDeq : (WStmt.eval fuel (wStoreByte "offHi")
      (WStmt.eval fuel (.bin .band "offHi" "offHi" (.imm 255))
      (WStmt.eval fuel (.bin .shr "offHi" off (.imm 8))
      (WStmt.eval fuel (wStoreByte "offLo")
      (WStmt.eval fuel (.bin .band "offLo" off (.imm 255))
      (WStmt.eval fuel (.bin .add "op" "op" (.reg litLen))
      (WStmt.eval fuel (.coopCopy "cpDst" "cpSrc" litLen) C))))))) = D at hmlmChain ⊢
  simp [WStmt.eval, WState.setReg, SCmp.run, WArg.eval, hmlmChain]

-- ── extC loop (by its own induction: its body is a cooperative statement) ──

/-- State after the four found-setup movs (`ecR,ec1,ml,extC`). -/
def foundExtEntry (endCap : Nat) (fuel : Nat) (ws : WState) : WState :=
  (wseq [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
          .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0) ]).eval fuel ws

/-- State after the extend `uwhile` (from `foundExtEntry`). -/
def foundExtDone (inStride endCap : Nat) (fuel : Nat) (ws : WState) : WState :=
  (WStmt.uwhile "extC" (extBodyStmt inStride endCap)).eval fuel (foundExtEntry endCap fuel ws)

/-- State at the `wEmitMatchSeq` entry (after movs, extend loop, and the two subs). -/
def foundMatchEntry (inStride endCap : Nat) (fuel : Nat) (ws : WState) : WState :=
  (wseq [ .bin .sub "off0" "p0" (.reg "cand0"),
          .bin .sub "litLen" "p0" (.reg "litAnchor") ]).eval fuel (foundExtDone inStride endCap fuel ws)

/-- The `uif found` true branch as a WStmt (matches `bodyEncodePrefix`'s inline form
    when `endCap = inStride - 5`). -/
def foundBranchStmt (inStride endCap : Nat) : WStmt :=
  wseq
  [ .mov "ecR" (.imm endCap), .mov "ec1" (.imm (endCap - 1)),
    .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
    .uwhile "extC" (wseq
      [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
        .bin .add "ml" "ml" (.reg "adv"),
        .setp .eq "extC" "adv" (.imm 32) ]),
    .bin .sub "off0" "p0" (.reg "cand0"),
    .bin .sub "litLen" "p0" (.reg "litAnchor"),
    wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
    .bin .add "litAnchor" "p0" (.reg "ml"),
    .mov "searchPos" (.reg "litAnchor") ]

/-- `p0`/`cand0` survive the found-setup movs; `ml` becomes `4`. -/
theorem foundExtEntry_p0 (endCap fuel : Nat) (ws : WState) :
    (foundExtEntry endCap fuel ws).regs "p0" = ws.regs "p0" := by
  simp [foundExtEntry, wseq, WStmt.eval, WState.setReg, WOp.run, SCmp.run, WArg.eval]

theorem foundExtEntry_mlN (endCap fuel : Nat) (ws : WState) :
    ((foundExtEntry endCap fuel ws).regs "ml").toNat = 4 := by
  simp [foundExtEntry, wseq, WStmt.eval, WState.setReg, WOp.run, SCmp.run, WArg.eval]

/-- The extend `uwhile` only writes `adv`/`ml`/`extC`; every other register is
    preserved (the eval `evalCoopExtendStep` writes only `adv`, `bin add ml` writes
    `ml`, the trailing `setp` writes `extC`). -/
theorem uwhile_extBody_frame (inStride endCap : Nat) (r : String)
    (hadv : r ≠ "adv") (hml : r ≠ "ml") (hextC : r ≠ "extC") :
    ∀ (fuel : Nat) (ws : WState),
      ((WStmt.uwhile "extC" (extBodyStmt inStride endCap)).eval fuel ws).regs r = ws.regs r := by
  intro fuel
  induction fuel with
  | zero => intro ws; simp [WStmt.eval]
  | succ fuel ih =>
    intro ws
    simp only [WStmt.eval]
    by_cases hb : (ws.regs "extC" == 1) = true
    · rw [if_pos hb, ih]
      simp [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg, WOp.run,
        SCmp.run, WArg.eval, hadv, hml, hextC]
    · rw [if_neg hb]

/-- `op`/`outBase`/`inBase`/`litAnchor`/`p0`/`cand0` survive the whole found-branch
    prefix (movs, extend loop, subs), so they equal their values at the branch entry
    at the `wEmitMatchSeq` entry state. -/
theorem foundMatchEntry_frame (inStride endCap fuel : Nat) (ws : WState) (r : String)
    (h1 : r ≠ "ecR") (h2 : r ≠ "ec1") (h3 : r ≠ "ml") (h4 : r ≠ "extC")
    (h5 : r ≠ "adv") (h6 : r ≠ "off0") (h7 : r ≠ "litLen") :
    (foundMatchEntry inStride endCap fuel ws).regs r = ws.regs r := by
  rw [foundMatchEntry, foundExtDone, foundExtEntry]
  simp only [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval, SCmp.run]
  rw [uwhile_extBody_frame inStride endCap r h5 h3 h4]
  simp [WState.setReg, WOp.run, WArg.eval, SCmp.run, h1, h2, h3, h4, h6, h7]

/-- The literal length `litLen = p0 - litAnchor` at the match-sequence entry (the two
    subs compute it from the `coopWindow`/frame-preserved `p0`/`litAnchor`). -/
theorem foundMatchEntry_litLen (inStride endCap fuel : Nat) (ws : WState) :
    (foundMatchEntry inStride endCap fuel ws).regs "litLen"
      = ws.regs "p0" - ws.regs "litAnchor" := by
  rw [foundMatchEntry, foundExtDone, foundExtEntry]
  simp only [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval, SCmp.run]
  rw [uwhile_extBody_frame inStride endCap "p0" (by decide) (by decide) (by decide),
      uwhile_extBody_frame inStride endCap "litAnchor" (by decide) (by decide) (by decide)]
  simp [WState.setReg, WOp.run, WArg.eval, SCmp.run]

/-- The match length `ml` at the match-sequence entry is bounded by `endCap` (the
    extend loop preserves `extInv`'s `p0 + ml ≤ endCap`), given the found-branch
    preconditions.  Discharges the `pMatBig` LSIC-fuel side-condition. -/
theorem foundMatchEntry_ml_bound (inStride endCap fuel : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hcand : (ws.regs "cand0").toNat < (ws.regs "p0").toNat)
    (hp4 : (ws.regs "p0").toNat + 4 ≤ endCap) :
    ((foundMatchEntry inStride endCap fuel ws).regs "ml").toNat ≤ endCap := by
  have hExtInv : extInv inStride endCap (foundExtDone inStride endCap fuel ws) := by
    rw [foundExtDone]
    exact extInv_uwhile inStride endCap hendCap hlen fuel (foundExtEntry endCap fuel ws)
      (extInv_of_foundMovs endCap inStride fuel ws hcand hp4)
  have hml : (foundMatchEntry inStride endCap fuel ws).regs "ml"
      = (foundExtDone inStride endCap fuel ws).regs "ml" := by
    rw [foundMatchEntry]
    simp [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval]
  rw [hml]
  obtain ⟨_, _, _, hpml⟩ := hExtInv
  omega

/-- The match length `ml` at the match-sequence entry is at least `MINMATCH = 4`:
    `foundExtEntry` sets `ml := 4` and the extend loop only grows it.  Discharges the
    `mlm = ml - 4` well-definedness for the `pMatBig` side-conditions. -/
theorem foundMatchEntry_ml_lb (inStride endCap fuel : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hcand : (ws.regs "cand0").toNat < (ws.regs "p0").toNat)
    (hp4 : (ws.regs "p0").toNat + 4 ≤ endCap) :
    4 ≤ ((foundMatchEntry inStride endCap fuel ws).regs "ml").toNat := by
  have hEntry4 : ((foundExtEntry endCap fuel ws).regs "ml").toNat = 4 := by
    rw [foundExtEntry]
    simp [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval, SCmp.run]
  have hml : (foundMatchEntry inStride endCap fuel ws).regs "ml"
      = (foundExtDone inStride endCap fuel ws).regs "ml" := by
    rw [foundMatchEntry]
    simp [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval]
  rw [hml, foundExtDone]
  exact uwhile_ml_ge4 inStride endCap hendCap hlen fuel (foundExtEntry endCap fuel ws)
    (extInv_of_foundMovs endCap inStride fuel ws hcand hp4) (by omega)

/-- The extend `uwhile` never touches `gmem` (the coop extend step writes only `adv`). -/
theorem uwhile_gmem_extBody (inStride endCap : Nat) :
    ∀ (fuel : Nat) (ws : WState),
      ((WStmt.uwhile "extC" (extBodyStmt inStride endCap)).eval fuel ws).gmem = ws.gmem := by
  intro fuel
  induction fuel with
  | zero => intro ws; simp [WStmt.eval]
  | succ fuel ih =>
      intro ws; simp only [WStmt.eval]
      by_cases hb : (ws.regs "extC" == 1) = true
      · rw [if_pos hb, ih]
        simp [extBodyStmt, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg]
      · rw [if_neg hb]

/-- The whole found-branch prefix (movs, extend loop, subs) leaves `gmem` unchanged. -/
theorem foundMatchEntry_gmem (inStride endCap fuel : Nat) (ws : WState) :
    (foundMatchEntry inStride endCap fuel ws).gmem = ws.gmem := by
  rw [foundMatchEntry, foundExtDone, foundExtEntry]
  simp only [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval, SCmp.run]
  rw [uwhile_gmem_extBody inStride endCap]

/-- **The 9 `wEmitMatchSeq` copy-layout / LSIC-fuel side-conditions**, discharged from
    a `LoopCInv`-style hypothesis set on the (post-`coopWindow`) state `ws`. -/
theorem foundBranch_sideconds (inStride endCap fuel : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hib0 : (ws.regs "inBase").toNat < 2 ^ 40)
    (hla_p0 : (ws.regs "litAnchor").toNat ≤ (ws.regs "p0").toNat)
    (hp0_lt : (ws.regs "p0").toNat < inStride)
    (hbud32 : (ws.regs "outBase").toNat + (ws.regs "op").toNat
        + 9 * (inStride - (ws.regs "litAnchor").toNat) < 2 ^ 32)
    (hbudsz : (ws.regs "outBase").toNat + (ws.regs "op").toNat
        + 9 * (inStride - (ws.regs "litAnchor").toNat) ≤ ws.gmem.size)
    (hobLB : (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat)
    (hcand : (ws.regs "cand0").toNat < (ws.regs "p0").toNat)
    (hp4 : (ws.regs "p0").toNat + 4 ≤ endCap)
    (hfuel : inStride ≤ fuel) :
    let M := foundMatchEntry inStride endCap fuel ws
    ((matchAfterSetp "litLen" "ml" fuel M).regs "pLitBig" = 1 →
        15 ≤ ((matchAfterSetp "litLen" "ml" fuel M).regs "litLen").toNat)
    ∧ ((matchAfterSetp "litLen" "ml" fuel M).regs "pLitBig" = 1 →
        ((matchAfterSetp "litLen" "ml" fuel M).regs "litLen").toNat < 255 * fuel + 15)
    ∧ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst").toNat < 2 ^ 32
    ∧ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpSrc").toNat < 2 ^ 32
    ∧ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "litLen").toNat < 2 ^ 32
    ∧ (((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst").toNat
          + ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "litLen").toNat
          ≤ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpSrc").toNat
        ∨ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpSrc").toNat
          + ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "litLen").toNat
          ≤ ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst").toNat)
    ∧ (((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst").toNat
          + ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "litLen").toNat
          ≤ (matchCpEntry "litAnchor" "litLen" "ml" fuel M).gmem.size)
    ∧ ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel M).regs "pMatBig" = 1 →
        15 ≤ ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel M).regs "mlm").toNat)
    ∧ ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel M).regs "pMatBig" = 1 →
        ((matchAfterMatSetp "litAnchor" "litLen" "off0" "ml" fuel M).regs "mlm").toNat
          < 255 * fuel + 15) := by
  intro M
  -- litLen = p0 - litAnchor at M; frame op/outBase/inBase/litAnchor to ws; ml ≤ endCap.
  have hLL : (M.regs "litLen").toNat = (ws.regs "p0").toNat - (ws.regs "litAnchor").toNat := by
    show ((foundMatchEntry inStride endCap fuel ws).regs "litLen").toNat = _
    rw [foundMatchEntry_litLen, UInt64.toNat_sub,
      show 2 ^ 64 - (ws.regs "litAnchor").toNat + (ws.regs "p0").toNat
        = 2 ^ 64 + ((ws.regs "p0").toNat - (ws.regs "litAnchor").toNat) from by
        have := (ws.regs "p0").toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (ws.regs "p0").toNat_lt; omega)]
  have hLLlt : (M.regs "litLen").toNat < inStride := by rw [hLL]; omega
  have hop : M.regs "op" = ws.regs "op" :=
    foundMatchEntry_frame inStride endCap fuel ws "op" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hob : M.regs "outBase" = ws.regs "outBase" :=
    foundMatchEntry_frame inStride endCap fuel ws "outBase" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hib : M.regs "inBase" = ws.regs "inBase" :=
    foundMatchEntry_frame inStride endCap fuel ws "inBase" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hla : M.regs "litAnchor" = ws.regs "litAnchor" :=
    foundMatchEntry_frame inStride endCap fuel ws "litAnchor" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hgm : M.gmem = ws.gmem := foundMatchEntry_gmem inStride endCap fuel ws
  have hmlle : (M.regs "ml").toNat ≤ endCap :=
    foundMatchEntry_ml_bound inStride endCap fuel ws hendCap hlen hcand hp4
  have hml4 : 4 ≤ (M.regs "ml").toNat :=
    foundMatchEntry_ml_lb inStride endCap fuel ws hendCap hlen hcand hp4
  -- the LSIC fuel bound: (litLen - 15)/255 < fuel.
  have hlfuel : ((M.regs "litLen").toNat - 15) / 255 < fuel := by
    have h := Nat.div_le_self ((M.regs "litLen").toNat - 15) 255
    omega
  -- litLen value at matchAfterSetp / matchCpEntry (framed) and the cpSrc/cpDst values.
  have hM_ll : (matchAfterSetp "litLen" "ml" fuel M).regs "litLen" = M.regs "litLen" :=
    matchAfterSetp_frame "litLen" "ml" "litLen" fuel M (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hM_pb := matchAfterSetp_pLitBig "litLen" "ml" fuel M (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide)
  have hC_ll : (matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "litLen" = M.regs "litLen" :=
    matchCpEntry_frame "litAnchor" "litLen" "ml" "litLen" fuel M (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide)
  have hC_cs : (matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpSrc"
      = M.regs "inBase" + M.regs "litAnchor" :=
    matchCpEntry_cpSrc "litAnchor" "litLen" "ml" fuel M (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have hC_cd : (matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst"
      = M.regs "outBase" + M.regs "op" + 1 + UInt64.ofNat (LZ4.encNib (M.regs "litLen").toNat).length :=
    matchCpEntry_cpDst "litAnchor" "litLen" "ml" fuel M (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) hlfuel
  have hC_gm_size : (matchCpEntry "litAnchor" "litLen" "ml" fuel M).gmem.size = M.gmem.size := by
    rw [matchCpEntry, eval_gmem_size, eval_gmem_size, eval_gmem_size, matchAfterSetp,
      eval_gmem_size, eval_gmem_size, eval_gmem_size, eval_gmem_size]
  -- toNat of cpSrc and cpDst.
  have hcsN : ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpSrc").toNat
      = (ws.regs "inBase").toNat + (ws.regs "litAnchor").toNat := by
    rw [hC_cs, hib, hla, UInt64.toNat_add, Nat.mod_eq_of_lt (by omega)]
  have hencle : (LZ4.encNib (M.regs "litLen").toNat).length ≤ (M.regs "litLen").toNat :=
    encNib_len_le _
  have hencle' : (LZ4.encNib (M.regs "litLen").toNat).length
      ≤ (ws.regs "p0").toNat - (ws.regs "litAnchor").toNat := by omega
  have hencN : (LZ4.encNib (M.regs "litLen").toNat).length < 2 ^ 64 := by omega
  have hnoof : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
      + (LZ4.encNib (M.regs "litLen").toNat).length < 2 ^ 64 := by omega
  have s_op1 : (ws.regs "op" + 1).toNat = (ws.regs "op").toNat + 1 := by
    rw [UInt64.toNat_add, show (1 : UInt64).toNat = 1 from by decide,
      Nat.mod_eq_of_lt (by omega)]
  have s_op1e : (ws.regs "op" + 1
        + UInt64.ofNat (LZ4.encNib (M.regs "litLen").toNat).length).toNat
      = (ws.regs "op").toNat + 1 + (LZ4.encNib (M.regs "litLen").toNat).length := by
    rw [UInt64.toNat_add, s_op1, toNat_ofNat_lt _ hencN, Nat.mod_eq_of_lt (by omega)]
  have hcdN : ((matchCpEntry "litAnchor" "litLen" "ml" fuel M).regs "cpDst").toNat
      = (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
        + (LZ4.encNib (M.regs "litLen").toNat).length := by
    rw [hC_cd, hob, hop,
      show ws.regs "outBase" + ws.regs "op" + 1
          + UInt64.ofNat (LZ4.encNib (M.regs "litLen").toNat).length
        = ws.regs "outBase" + (ws.regs "op" + 1
          + UInt64.ofNat (LZ4.encNib (M.regs "litLen").toNat).length) from by ac_rfl,
      UInt64.toNat_add, s_op1e, Nat.mod_eq_of_lt (by omega)]
    omega
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro hpb; rw [hM_ll, hLL]
    rw [hM_pb] at hpb
    by_cases hg : UInt64.ofNat 15 ≤ M.regs "litLen"
    · rw [UInt64.le_iff_toNat_le, show ((UInt64.ofNat 15).toNat) = 15 from by decide] at hg
      rw [hLL] at hg; omega
    · rw [if_neg hg] at hpb; exact absurd hpb (by decide)
  · intro _; rw [hM_ll]; omega
  · rw [hcdN]; omega
  · rw [hcsN]; omega
  · rw [hC_ll]; omega
  · right; rw [hcdN, hcsN, hC_ll, hLL]; omega
  · rw [show (matchCpEntry "litAnchor" "litLen" "ml" fuel M).gmem.size = ws.gmem.size from by
        rw [hC_gm_size, hgm], hcdN, hC_ll]
    have := hencle; have := hLL; omega
  · intro hpm
    have hmlm := matchAfterMatSetp_mlm "litAnchor" "litLen" "off0" "ml" fuel M (by decide)
    have hpmb := matchAfterMatSetp_pMatBig "litAnchor" "litLen" "off0" "ml" fuel M (by decide)
    rw [hmlm]
    rw [hpmb] at hpm
    by_cases hg : UInt64.ofNat 15 ≤ M.regs "ml" - UInt64.ofNat 4
    · rw [UInt64.le_iff_toNat_le, show ((UInt64.ofNat 15).toNat) = 15 from by decide] at hg; exact hg
    · rw [if_neg hg] at hpm; exact absurd hpm (by decide)
  · intro _
    have hmlm := matchAfterMatSetp_mlm "litAnchor" "litLen" "off0" "ml" fuel M (by decide)
    rw [hmlm, UInt64.toNat_sub, show ((UInt64.ofNat 4).toNat) = 4 from by decide,
      show 2 ^ 64 - 4 + (M.regs "ml").toNat = 2 ^ 64 + ((M.regs "ml").toNat - 4) from by omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (M.regs "ml").toNat_lt; omega)]
    omega

-- ── loopC body: `coopWindow ; uif found ; setp loopC` ───────────────────────────

/-- The `loopC` loop body (matches `bodyEncodePrefix`'s inline body when
    `endCap = inStride - 5`). -/
def loopCBodyStmt (inStride hashLog : Nat) : WStmt :=
  wseq
  [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
    .uif "found" (foundBranchStmt inStride (inStride - 5))
      (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]),
    .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ]

/-- The `loopC` loop's invariant.
    Same layout facts as `LoopCInv` but WITHOUT `searchPos ≤ inStride` — that bound
    is not preserved by a not-found step (`searchPos += 32` can pass `inStride`); the
    loop body instead recovers `searchPos < searchLim` from the loop guard.  `B` bounds the
    output budget's end: `2 ^ 32` where 32-bit copy pointers need it, `2 ^ 64` for a
    kernel addressing the whole of global memory. -/
def LoopCQ (B inStride : Nat) (ws : WState) : Prop :=
  (ws.regs "inBase").toNat < 2 ^ 40
  ∧ (ws.regs "litAnchor").toNat ≤ (ws.regs "searchPos").toNat
  ∧ (ws.regs "loopC")
      = (if UInt64.ofNat (ws.regs "searchPos").toNat < UInt64.ofNat (inStride - 12) then 1 else 0)
  ∧ (ws.regs "inBase").toNat + inStride ≤ (ws.regs "outBase").toNat
  ∧ (ws.regs "outBase").toNat + (ws.regs "op").toNat
      + 9 * (inStride - (ws.regs "litAnchor").toNat) < B
  ∧ (ws.regs "outBase").toNat + (ws.regs "op").toNat
      + 9 * (inStride - (ws.regs "litAnchor").toNat) ≤ ws.gmem.size

/-- From `LoopCQ` and the loop guard (`loopC = 1`), the cursor is below the limit. -/
theorem loopCQ_guard {B : Nat} (inStride : Nat) (ws : WState) (hstride : inStride ≤ 65536)
    (hipos : 12 ≤ inStride) (hinv : LoopCQ B inStride ws) (hloopC : (ws.regs "loopC" == 1) = true) :
    (ws.regs "searchPos").toNat < inStride - 12 := by
  obtain ⟨_, _, hlc, _⟩ := hinv
  rw [beq_iff_eq, hlc] at hloopC
  by_cases h : UInt64.ofNat (ws.regs "searchPos").toNat < UInt64.ofNat (inStride - 12)
  · rw [UInt64.lt_iff_toNat_lt, toNat_ofNat_lt _ (by have := (ws.regs "searchPos").toNat_lt; omega),
      toNat_ofNat_lt _ (by omega)] at h
    exact h
  · rw [if_neg h] at hloopC; exact absurd hloopC (by decide)

end AlgorithmLib.LZ4WarpDSL
