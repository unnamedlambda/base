import AlgorithmLib.EvalValid

/-!
  # The tight output budget, as a forward invariant

  `LoopCQ` carries the *generous* budget `op + 9*(inStride - litAnchor)`, which is
  what memory safety needs and is far too weak for `Lz4Sites.CursorAtSites.opLe`:
  the cursor must stay below `lenOff = inStride + inStride/16 + 256`, and
  `9*inStride` blows straight past that.

  The obvious route — "`op` is monotone and ends at `encode.length ≤ lenOff`" —
  needs the loop's FINAL state at every interior point, which the simulation's
  induction (forward, with a fuel-free `WState → Prop` invariant) cannot carry.

  So bound the future instead of consulting it.  `tightRem` is the standard LZ4
  worst case for the input that is still unconsumed,

      (inStride - litAnchor) + (inStride - litAnchor)/255 + 2,

  and `op + tightRem ≤ LO` is *forward-preserved*: one iteration emits
  `1 + |encNib ll| + ll + 2 + |encNib (ml-4)|` bytes and consumes `ll + ml ≥ ll + 4`
  input bytes, and the four-byte minimum match pays for the token and the offset
  field while the `/255` term pays for the LSIC bytes.  No fuel, no final state,
  no monotonicity argument.

  This also covers the tail: at the loop exit `encFinalLen fl = 1 + |encNib fl| + fl`
  is `≤ tightRem` at the exit anchor, so the final literal run's stores are bounded
  by the same invariant that bounds the body's.
-/

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib AlgorithmLib.LZ4 AlgorithmLib.LZ4Simt AlgorithmLib.LZ4Plan
open AlgorithmLib.LZ4WarpFind

/-- Floor division is superadditive.  Used three times below to pay for LSIC
    bytes out of the consumed input's `/255` share. -/
theorem div_add_div_le (a b k : Nat) : a / k + b / k ≤ (a + b) / k := by
  cases k with
  | zero => simp
  | succ k =>
      rw [Nat.le_div_iff_mul_le (Nat.succ_pos k), Nat.add_mul]
      exact Nat.add_le_add (Nat.div_mul_le_self a (k + 1)) (Nat.div_mul_le_self b (k + 1))

/-- Worst-case bytes still owed for the input from `litAnchor` on. -/
def tightRem (inStride : Nat) (w : WState) : Nat :=
  (inStride - (w.regs "litAnchor").toNat) + (inStride - (w.regs "litAnchor").toNat) / 255 + 2

/-- **The tight output-cursor budget.**  `LO` is the shipped `lenOff`. -/
def TightQ (inStride LO : Nat) (w : WState) : Prop :=
  (w.regs "op").toNat + tightRem inStride w ≤ LO

/-- The bound holds at the loop entry: nothing emitted, nothing consumed, and
    `inStride/255 + 2 ≤ inStride/16 + 256`. -/
theorem tightQ_init (inStride LO : Nat) (w : WState) (hop : w.regs "op" = 0)
    (hla : w.regs "litAnchor" = 0) (hLO : inStride + inStride / 16 + 256 ≤ LO) :
    TightQ inStride LO w := by
  have h255 : inStride / 255 ≤ inStride / 16 := Nat.div_le_div_left (by omega) (by omega)
  have h0 : (0 : UInt64).toNat = 0 := rfl
  simp only [TightQ, tightRem, hop, hla, h0, Nat.sub_zero]
  omega

/-- The arithmetic core of preservation, with the program erased: emitting one
    sequence costs at most what its input consumption buys back. -/
theorem tight_step_arith (R' ll ml e1 e2 : Nat) (hml : 4 ≤ ml)
    (he1 : e1 = if ll < 15 then 0 else (ll - 15) / 255 + 1)
    (he2 : e2 = if ml - 4 < 15 then 0 else (ml - 4 - 15) / 255 + 1) :
    (1 + e1 + ll + 2 + e2) + R' + R' / 255 + 2 ≤ (R' + (ll + ml)) + (R' + (ll + ml)) / 255 + 2 := by
  have hdiv : R' / 255 + (ll + ml) / 255 ≤ (R' + (ll + ml)) / 255 := div_add_div_le _ _ _
  have hsum : ll / 255 + (ml - 4) / 255 ≤ (ll + ml) / 255 := by
    have h := div_add_div_le ll (ml - 4) 255
    have hm : (ll + (ml - 4)) / 255 ≤ (ll + ml) / 255 := Nat.div_le_div_right (by omega)
    omega
  have hd1 : (ll - 15) / 255 ≤ ll / 255 := Nat.div_le_div_right (by omega)
  have hd2 : (ml - 4 - 15) / 255 ≤ (ml - 4) / 255 := Nat.div_le_div_right (by omega)
  have hd3 : ll / 255 ≤ (ll + ml) / 255 := Nat.div_le_div_right (by omega)
  have hd4 : (ml - 4) / 255 ≤ (ll + ml) / 255 := Nat.div_le_div_right (by omega)
  by_cases h1 : ll < 15
  · by_cases h2 : ml - 4 < 15
    · rw [if_pos h1] at he1; rw [if_pos h2] at he2; omega
    · rw [if_pos h1] at he1; rw [if_neg h2] at he2; omega
  · by_cases h2 : ml - 4 < 15
    · rw [if_neg h1] at he1; rw [if_pos h2] at he2; omega
    · rw [if_neg h1] at he1; rw [if_neg h2] at he2; omega

/-- **One `loopC` iteration preserves the tight budget.**  Mirrors
    `loopCBody_Qadvance`'s case split (`bodyNotFound_eq` / `bodyFound_eq`); a
    not-found step moves neither `op` nor `litAnchor`, and a found step is
    `tight_step_arith`. -/
theorem loopCBody_tight (inStride hashLog fuel LO : Nat) (ws : WState)
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride) (hp64 : inStride < 2 ^ 64)
    (hfuel : inStride ≤ fuel)
    (hQ : LoopCQ inStride ws) (hguard : (ws.regs "loopC" == 1) = true)
    (hT : TightQ inStride LO ws) :
    TightQ inStride LO ((loopCBodyStmt inStride hashLog).eval fuel ws) := by
  obtain ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ := hQ
  have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
    loopCQ_guard inStride ws hstride hipos ⟨hib0, hlaSp, hlc, hobLB, hbud32, hbudsz⟩ hguard
  simp only [TightQ, tightRem] at hT ⊢
  cases hw : window (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat)
      (inStride - 12) (ws.regs "searchPos").toNat with
  | none =>
      obtain ⟨_, _, hN_la, _, _, hN_op, _, _⟩ :=
        bodyNotFound_eq inStride hashLog (ws.regs "searchPos").toNat fuel ws hw rfl (by omega)
      have e_la : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "litAnchor"
          = ws.regs "litAnchor" := hN_la
      have e_op : ((loopCBodyStmt inStride hashLog).eval fuel ws).regs "op"
          = ws.regs "op" := hN_op
      rw [e_la, e_op]; exact hT
  | some pc =>
      obtain ⟨p, c⟩ := pc
      obtain ⟨hsps, hpsl, _hcp, _hv⟩ :=
        window_sound (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat) (inStride - 12)
          (ws.regs "searchPos").toNat p c hw
      have hml4 : 4 ≤ extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
          (inStride - 5) fuel 4 :=
        extendFrom_le _ p c (inStride - 5) fuel 4
      have hmlcap : p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
          (inStride - 5) fuel 4 ≤ inStride - 5 := by
        rcases extendFrom_cap (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
          (inStride - 5) fuel 4 with heq | hle
        · omega
        · exact hle
      have h1 := encNib_length_le (p - (ws.regs "litAnchor").toNat)
      have h2 := encNib_length_le
        (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4
          - 4)
      have hbudget : (1 + (LZ4.encNib (p - (ws.regs "litAnchor").toNat)).length
          + (p - (ws.regs "litAnchor").toNat) + 2
          + (LZ4.encNib (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
              (inStride - 5) fuel 4 - 4)).length)
          + 9 * (inStride - (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
              p c (inStride - 5) fuel 4))
          ≤ 9 * (inStride - (ws.regs "litAnchor").toNat) := by
        have hsplit : inStride - (ws.regs "litAnchor").toNat
            = (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
                (inStride - 5) fuel 4 - (ws.regs "litAnchor").toNat)
              + (inStride - (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
                  p c (inStride - 5) fuel 4)) := by
          omega
        rw [hsplit, Nat.mul_add]; omega
      obtain ⟨_, _, hF_la, _, _, hF_op, _, _⟩ :=
        bodyFound_eq inStride hashLog (ws.regs "searchPos").toNat p c fuel ws
          (ws.regs "litAnchor").toNat hstride hw hlaSp rfl rfl
          (by omega) (by omega)
          (by have := Nat.div_le_self (p - (ws.regs "litAnchor").toNat) 255; omega)
          (by have := Nat.div_le_self
                (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
                  (inStride - 5) fuel 4 - 4) 255; omega)
          hp64
          (by omega) (by omega) (by omega) (Or.inr (by omega)) (by omega)
      have hF_opN : (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "op").toNat
          = (ws.regs "op").toNat + (1 + (LZ4.encNib (p - (ws.regs "litAnchor").toNat)).length
              + (p - (ws.regs "litAnchor").toNat) + 2
              + (LZ4.encNib (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
                  (inStride - 5) fuel 4 - 4)).length) := by
        have : (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "op") = _ := hF_op
        rw [this, UInt64.toNat_add, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega)]
        apply Nat.mod_eq_of_lt; omega
      have e_la : (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "litAnchor").toNat
          = p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
              (inStride - 5) fuel 4 := hF_la
      rw [hF_opN, e_la]
      have harith := tight_step_arith
        (inStride - (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
          (inStride - 5) fuel 4))
        (p - (ws.regs "litAnchor").toNat)
        (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c (inStride - 5) fuel 4)
        (LZ4.encNib (p - (ws.regs "litAnchor").toNat)).length
        (LZ4.encNib (extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
          (inStride - 5) fuel 4 - 4)).length
        hml4 (AlgorithmLib.LZ4Imp.encNib_length _) (AlgorithmLib.LZ4Imp.encNib_length _)
      have hR : (inStride - (p + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
            p c (inStride - 5) fuel 4))
          + ((p - (ws.regs "litAnchor").toNat)
            + extendFrom (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) p c
                (inStride - 5) fuel 4)
          = inStride - (ws.regs "litAnchor").toNat := by omega
      rw [hR] at harith
      omega

/-- **The tail is bounded by the same invariant.**  At the loop exit the final
    literal run emits `1 + |encNib fl| + fl` bytes for `fl = inStride - litAnchor`,
    which is exactly what `tightRem` reserved. -/
theorem tightQ_final (inStride LO : Nat) (w : WState) (hT : TightQ inStride LO w)
    (fl : Nat) (hfl : fl = inStride - (w.regs "litAnchor").toNat) :
    (w.regs "op").toNat + (1 + (LZ4.encNib fl).length + fl) ≤ LO := by
  have h : (LZ4.encNib fl).length = if fl < 15 then 0 else (fl - 15) / 255 + 1 :=
    AlgorithmLib.LZ4Imp.encNib_length _
  have hd : (fl - 15) / 255 ≤ fl / 255 := Nat.div_le_div_right (by omega)
  simp only [TightQ, tightRem] at hT
  rw [← hfl] at hT
  by_cases hs : fl < 15
  · rw [if_pos hs] at h; omega
  · rw [if_neg hs] at h; omega

/-- **The token's own bytes fit under the tight bound.**  At the match-sequence
    entry state — the machine's pc 124, one instruction before the token region —
    the cursor plus everything this token will emit is still `≤ LO`.

    Stated in `M`'s OWN registers (`litLen`, `ml`), not in the window's `p` and
    `extendFrom`, because the consumer is the machine side: `Couple` transports
    registers, not plan values.  The route is entirely structural — `eval_wEmitMatchSeq`
    says the cursor lands on `op + |encodeSeq|`, the two trailing statements of the
    found branch write `litAnchor`/`searchPos` and not `op`, so that landing value
    IS the branch's exit cursor, which `hbnd` bounds. -/
theorem foundMatchEntry_tokBound (inStride endCap fuel LO : Nat) (ws : WState)
    (hendCap : endCap = inStride - 5) (hlen : inStride < 2 ^ 40)
    (hla_p0 : (ws.regs "litAnchor").toNat ≤ (ws.regs "p0").toNat)
    (hp0_lt : (ws.regs "p0").toNat < inStride)
    (hbud32 : (ws.regs "outBase").toNat + (ws.regs "op").toNat
        + 9 * (inStride - (ws.regs "litAnchor").toNat) < 2 ^ 32)
    (hcand : (ws.regs "cand0").toNat < (ws.regs "p0").toNat)
    (hp4 : (ws.regs "p0").toNat + 4 ≤ endCap)
    (hfuel : inStride ≤ fuel)
    (hbnd : (((foundBranchStmt inStride endCap).eval fuel ws).regs "op").toNat ≤ LO) :
    4 ≤ ((foundMatchEntry inStride endCap fuel ws).regs "ml").toNat
    ∧ ((foundMatchEntry inStride endCap fuel ws).regs "op").toNat
        + (1 + (LZ4.encNib ((foundMatchEntry inStride endCap fuel ws).regs "litLen").toNat).length
            + ((foundMatchEntry inStride endCap fuel ws).regs "litLen").toNat + 2
            + (LZ4.encNib (((foundMatchEntry inStride endCap fuel ws).regs "ml").toNat - 4)).length)
          ≤ LO := by
  have hLL : ((foundMatchEntry inStride endCap fuel ws).regs "litLen").toNat
      = (ws.regs "p0").toNat - (ws.regs "litAnchor").toNat := by
    rw [foundMatchEntry_litLen, UInt64.toNat_sub,
      show 2 ^ 64 - (ws.regs "litAnchor").toNat + (ws.regs "p0").toNat
        = 2 ^ 64 + ((ws.regs "p0").toNat - (ws.regs "litAnchor").toNat) from by
        have := (ws.regs "p0").toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by have := (ws.regs "p0").toNat_lt; omega)]
  have hop : (foundMatchEntry inStride endCap fuel ws).regs "op" = ws.regs "op" :=
    foundMatchEntry_frame inStride endCap fuel ws "op" (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
  have hml4 : 4 ≤ ((foundMatchEntry inStride endCap fuel ws).regs "ml").toNat :=
    foundMatchEntry_ml_lb inStride endCap fuel ws hendCap hlen hcand hp4
  have hmlle : ((foundMatchEntry inStride endCap fuel ws).regs "ml").toNat ≤ endCap :=
    foundMatchEntry_ml_bound inStride endCap fuel ws hendCap hlen hcand hp4
  refine ⟨hml4, ?_⟩
  -- the cursor after `wEmitMatchSeq`, in `M`'s registers.
  obtain ⟨hEmitOp, -⟩ := LZ4WarpEvalBytes.eval_wEmitMatchSeq "litAnchor" "litLen" "off0" "ml"
    (foundMatchEntry inStride endCap fuel ws) fuel
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) hml4
    (by have := Nat.div_le_self
          ((foundMatchEntry inStride endCap fuel ws).regs "litLen").toNat 255; omega)
    (by have := Nat.div_le_self
          (((foundMatchEntry inStride endCap fuel ws).regs "ml").toNat - 4) 255; omega)
  -- the two trailing statements of the found branch do not write `op`.
  have hTailOp : ((foundBranchStmt inStride endCap).eval fuel ws).regs "op"
      = ((wEmitMatchSeq "litAnchor" "litLen" "off0" "ml").eval fuel
          (foundMatchEntry inStride endCap fuel ws)).regs "op" := by
    have hgoalEq : (foundBranchStmt inStride endCap).eval fuel ws
        = (wseq [ .bin .add "litAnchor" "p0" (.reg "ml"),
                  .mov "searchPos" (.reg "litAnchor") ]).eval fuel
            ((wEmitMatchSeq "litAnchor" "litLen" "off0" "ml").eval fuel
              (foundMatchEntry inStride endCap fuel ws)) := by
      simp only [foundBranchStmt, foundMatchEntry, foundExtDone, foundExtEntry,
        extBodyStmt, wseq, WStmt.eval]
    rw [hgoalEq]
    simp [wseq, WStmt.eval, WState.setReg, WOp.run, WArg.eval]
  -- no wraparound: the cursor is below `2 ^ 32` and the token is bounded by the stride.
  have he1 := encNib_length_le ((foundMatchEntry inStride endCap fuel ws).regs "litLen").toNat
  have he2 := encNib_length_le
    (((foundMatchEntry inStride endCap fuel ws).regs "ml").toNat - 4)
  have hopN : ((foundMatchEntry inStride endCap fuel ws).regs "op").toNat
      = (ws.regs "op").toNat := by rw [hop]
  have hllt : ((foundMatchEntry inStride endCap fuel ws).regs "litLen").toNat < inStride := by
    rw [hLL]; omega
  have hoplt : ((foundMatchEntry inStride endCap fuel ws).regs "op").toNat < 2 ^ 32 := by
    rw [hopN]; omega
  rw [hTailOp, hEmitOp, UInt64.toNat_add,
    AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by omega), Nat.mod_eq_of_lt (by omega)] at hbnd
  exact hbnd

/-- `TightQ` at the loop exit — the tail's entry condition.  Same induction as
    `loopC_loop_preservesInv`, with `loopCBody_tight` for the step. -/
theorem loopC_loop_preservesTight (inStride hashLog LO : Nat) (hstride : inStride ≤ 65536)
    (hipos : 12 ≤ inStride) (hp64 : inStride < 2 ^ 64) :
    ∀ (fuel : Nat) (ws : WState), LoopCQ inStride ws → TightQ inStride LO ws →
      (inStride - 12) - (ws.regs "searchPos").toNat + inStride ≤ fuel →
      TightQ inStride LO ((WStmt.uwhile "loopC" (loopCBodyStmt inStride hashLog)).eval fuel ws) := by
  intro fuel
  induction fuel with
  | zero => intro ws _ _ hf; exact absurd hf (by omega)
  | succ n ih =>
      intro ws hQ hT hf
      rw [WStmt.eval]
      by_cases hg : (ws.regs "loopC" == 1) = true
      · rw [if_pos hg]
        have hsp_lt : (ws.regs "searchPos").toNat < inStride - 12 :=
          loopCQ_guard inStride ws hstride hipos hQ hg
        have hqa := loopCBody_Qadvance inStride hashLog n ws hstride hipos hp64 (by omega) hQ hg
        exact ih ((loopCBodyStmt inStride hashLog).eval n ws) hqa.1
          (loopCBody_tight inStride hashLog n LO ws hstride hipos hp64 (by omega) hQ hg hT)
          (by omega)
      · rw [if_neg hg]; exact hT

/-- **What the checkpoint at the match-sequence entry exports.**  One coupled eval
    state carrying (i) the loose memory budget, unchanged, and (ii) the tight token
    bound this file proves.  Named because it appears nineteen times in the
    descent's `AllSteps` obligations. -/
def MatchEntryQ (inStride LO : Nat) (st : SState) : Prop :=
  ∃ w, Couple ("p0" :: "cand0" :: loopR) st w ∧
    (w.regs "outBase").toNat + (w.regs "op").toNat
        + 9 * (inStride - (w.regs "litAnchor").toNat) < 2 ^ 32
    ∧ (w.regs "outBase").toNat + (w.regs "op").toNat
        + 9 * (inStride - (w.regs "litAnchor").toNat) ≤ w.gmem.size
    ∧ 4 ≤ (w.regs "ml").toNat
    ∧ (w.regs "op").toNat + (1 + (LZ4.encNib (w.regs "litLen").toNat).length
        + (w.regs "litLen").toNat + 2 + (LZ4.encNib ((w.regs "ml").toNat - 4)).length) ≤ LO

end AlgorithmLib.LZ4WarpDSL
