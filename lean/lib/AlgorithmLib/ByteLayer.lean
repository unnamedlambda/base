import AlgorithmLib.LZ4WarpKernel
import AlgorithmLib.LZ4WarpFind
import AlgorithmLib.LZ4Plan
import AlgorithmLib.LZ4WarpSched

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib.LZ4Plan AlgorithmLib.LZ4 AlgorithmLib.LZ4WarpFind AlgorithmLib.LZ4WarpSched
  AlgorithmLib.LZ4Simt

/-- The inner extend `uwhile` body of `compressorBodyWStmt`. -/
def innerExtBody (inStride endCap : Nat) : WStmt := wseq
  [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride endCap,
    .bin .add "ml" "ml" (.reg "adv"),
    .setp .eq "extC" "adv" (.imm 32) ]

/-- `leadingGood` never exceeds its window cap `b`. -/
theorem leadingGood_le (inp : List UInt8) (p c endCap ml : Nat) :
    ∀ b, AlgorithmLib.LZ4WarpSched.leadingGood inp p c endCap ml b ≤ b := by
  intro b
  induction b generalizing ml with
  | zero => simp [AlgorithmLib.LZ4WarpSched.leadingGood]
  | succ n ih =>
      unfold AlgorithmLib.LZ4WarpSched.leadingGood
      by_cases h : AlgorithmLib.LZ4WarpSched.good inp p c endCap ml = true
      · rw [if_pos h]; have := ih (ml + 1); omega
      · rw [if_neg h]; omega

/-- A `uwhile` whose guard register is not `1` is a no-op, at any fuel. -/
theorem uwhile_stop (cond : String) (body : WStmt) (st : WState)
    (h : (st.regs cond == 1) = false) :
    ∀ n, (WStmt.uwhile cond body).eval n st = st := by
  intro n; cases n with
  | zero => simp [WStmt.eval]
  | succ n => simp [WStmt.eval, h]

/-- The eval of the inner extend loop computes `ml = extendChunk` (the model's
    cooperative 32-wide extend), for the register-recorded `p0`/`cand0`. Only the
    `adv`/`ml`/`extC` registers change; everything else (incl. gmem/smem) is fixed.
    Constants `g`/`pv`/`cv` are pinned so the induction hypothesis re-applies.
    The bounds keep `ml`/window values below `2^64` so the machine `UInt64` matches
    the model `Nat`, and let the recursion re-establish `m ≤ inStride`. -/
theorem inner_extend (inStride endCap ib : Nat) (g : Array UInt8) (pv cv : UInt64)
    (hstride : inStride ≤ 65536) (hendCap : endCap ≤ inStride) :
    ∀ (F m : Nat) (st : WState), m ≤ inStride → st.gmem = g →
      (st.regs "inBase").toNat = ib →
      st.regs "p0" = pv → st.regs "cand0" = cv →
      st.regs "extC" = 1 → st.regs "ml" = UInt64.ofNat m →
      ((WStmt.uwhile "extC" (innerExtBody inStride endCap)).eval F st).regs "ml"
        = UInt64.ofNat (extendChunk (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap F m)
      ∧ (∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" →
          ((WStmt.uwhile "extC" (innerExtBody inStride endCap)).eval F st).regs r = st.regs r)
      ∧ ((WStmt.uwhile "extC" (innerExtBody inStride endCap)).eval F st).gmem = st.gmem
      ∧ ((WStmt.uwhile "extC" (innerExtBody inStride endCap)).eval F st).smem = st.smem := by
  intro F
  induction F with
  | zero =>
      intro m st _ hg _ hp hc _ hml
      refine ⟨?_, ?_, ?_, ?_⟩
      · simp [WStmt.eval, extendChunk, hml]
      · intro r _ _ _; simp [WStmt.eval]
      · simp [WStmt.eval]
      · simp [WStmt.eval]
  | succ n ih =>
      intro m st hm hg hib hp hc hextC hml
      -- `(UInt64.ofNat m).toNat = m` since `m ≤ inStride ≤ 2^16 < 2^64`
      have hmb : (UInt64.ofNat m).toNat = m :=
        UInt64.toNat_ofNat_of_lt' (show m < 18446744073709551616 by omega)
      -- one iteration: extC = 1, so the body runs
      have hrun : (WStmt.uwhile "extC" (innerExtBody inStride endCap)).eval (n+1) st
          = (WStmt.uwhile "extC" (innerExtBody inStride endCap)).eval n
              ((innerExtBody inStride endCap).eval n st) := by
        simp [WStmt.eval, hextC]
      -- abbreviate the leading-good count `lg` (no Mathlib `set`)
      obtain ⟨lg, hlgdef⟩ :
          ∃ lg, leadingGood (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap m 32 = lg := ⟨_, rfl⟩
      have hlgle : lg ≤ 32 := by rw [← hlgdef]; exact leadingGood_le _ _ _ _ _ 32
      -- the body: adv := leadingGood …, ml := m + adv, extC := (adv == 32)
      have hbody : (innerExtBody inStride endCap).eval n st
          = (((st.setReg "adv" (UInt64.ofNat lg)).setReg "ml"
              (UInt64.ofNat m + UInt64.ofNat lg)).setReg "extC"
              (if UInt64.ofNat lg == (32 : UInt64) then 1 else 0)) := by
        simp only [innerExtBody, wseq, WStmt.eval, evalCoopExtendStep, WState.setReg,
          WOp.run, WArg.eval, SCmp.run, hg, hib, hp, hc, hml, hmb, hlgdef]
        rfl
      rw [hrun, hbody]
      -- abbreviate the post-body state
      obtain ⟨st', hst'⟩ :
          ∃ st', (((st.setReg "adv" (UInt64.ofNat lg)).setReg "ml"
              (UInt64.ofNat m + UInt64.ofNat lg)).setReg "extC"
              (if UInt64.ofNat lg == (32 : UInt64) then 1 else 0)) = st' := ⟨_, rfl⟩
      rw [hst']
      have e_gmem : st'.gmem = st.gmem := by rw [← hst']; simp [WState.setReg]
      have e_smem : st'.smem = st.smem := by rw [← hst']; simp [WState.setReg]
      have e_p0 : st'.regs "p0" = pv := by rw [← hst']; simp [WState.setReg, hp]
      have e_c : st'.regs "cand0" = cv := by rw [← hst']; simp [WState.setReg, hc]
      have e_ml : st'.regs "ml" = UInt64.ofNat m + UInt64.ofNat lg := by
        rw [← hst']; simp [WState.setReg]
      have e_extC : st'.regs "extC"
          = (if UInt64.ofNat lg == (32 : UInt64) then (1 : UInt64) else 0) := by
        rw [← hst']; simp [WState.setReg]
      have e_frame : ∀ r, r ≠ "adv" → r ≠ "ml" → r ≠ "extC" → st'.regs r = st.regs r := by
        intro r h1 h2 h3
        rw [← hst']; simp only [WState.setReg]
        rw [if_neg h3, if_neg h2, if_neg h1]
      have e_inBase : (st'.regs "inBase").toNat = ib := by
        rw [e_frame "inBase" (by decide) (by decide) (by decide)]; exact hib
      by_cases hlg : lg = 32
      · -- full 32-window: extC = 1, recurse with m + 32
        have hextC1 : st'.regs "extC" = 1 := by rw [e_extC, hlg]; decide
        have e_ml' : st'.regs "ml" = UInt64.ofNat (m + 32) := by
          rw [e_ml, hlg]; exact (UInt64.ofNat_add m 32).symm
        -- re-establish the recursion bound `m + 32 ≤ inStride`
        have hmbound : m + 32 ≤ inStride := by
          obtain ⟨L, hRle, hRg, hRbad⟩ :=
            Res_exists (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap m
          have hlgL : lg = min 32 (L - m) := by
            rw [← hlgdef]
            exact leadingGood_Res (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap 32 m L
              ⟨hRle, hRg, hRbad⟩
          have hEbad : good (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap
              (endCap - pv.toNat) = false := good_false_past _ _ _ _ _ (by omega)
          have hLml : L - m ≤ endCap - (pv.toNat + m) := by
            rcases Nat.lt_or_ge (endCap - pv.toNat) m with hlt | hge
            · have hmlbad : good (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap m = false :=
                good_false_past _ _ _ _ _ (by omega)
              have hmlL : L ≤ m := by
                rcases Nat.lt_or_ge m L with h | h
                · rw [hRg m (Nat.le_refl m) h] at hmlbad; exact absurd hmlbad (by simp)
                · exact h
              omega
            · have hLE : L ≤ endCap - pv.toNat := by
                rcases Nat.lt_or_ge L (endCap - pv.toNat + 1) with h | h
                · omega
                · have hEL : endCap - pv.toNat < L := by omega
                  rw [hRg (endCap - pv.toNat) hge hEL] at hEbad
                  exact absurd hEbad (by simp)
              omega
          rw [hlg] at hlgL
          omega
        obtain ⟨ih1, ih2, ih3, ih4⟩ :=
          ih (m + 32) st' hmbound (e_gmem.trans hg) e_inBase e_p0 e_c hextC1 e_ml'
        have hchunk : extendChunk (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap (n+1) m
            = extendChunk (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap n (m+32) := by
          show (if leadingGood (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap m 32 = 32
                then extendChunk (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap n (m+32)
                else m + leadingGood (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap m 32)
              = extendChunk (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap n (m+32)
          rw [hlgdef, if_pos hlg]
        refine ⟨?_, ?_, ?_, ?_⟩
        · rw [ih1, hchunk]
        · intro r a b c; rw [ih2 r a b c, e_frame r a b c]
        · rw [ih3, e_gmem]
        · rw [ih4, e_smem]
      · -- partial window: adv = lg < 32, extC = 0, loop stops
        have hlt : lg < 32 := by omega
        have hbfalse : (UInt64.ofNat lg == (32 : UInt64)) = false := by
          rw [beq_eq_false_iff_ne, ne_eq]
          intro he
          have hcontra : (UInt64.ofNat lg).toNat = (32 : UInt64).toNat := by rw [he]
          rw [UInt64.toNat_ofNat_of_lt' (show lg < 18446744073709551616 by omega),
            show (32 : UInt64).toNat = 32 from by decide] at hcontra
          omega
        have hextC0 : st'.regs "extC" = 0 := by rw [e_extC, hbfalse]; rfl
        have hstop : (WStmt.uwhile "extC" (innerExtBody inStride endCap)).eval n st' = st' :=
          uwhile_stop "extC" (innerExtBody inStride endCap) st' (by rw [hextC0]; rfl) n
        have hchunk : extendChunk (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap (n+1) m
            = m + lg := by
          show (if leadingGood (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap m 32 = 32
                then extendChunk (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap n (m+32)
                else m + leadingGood (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap m 32)
              = m + lg
          rw [hlgdef, if_neg hlg]
        rw [hstop]
        refine ⟨?_, ?_, e_gmem, e_smem⟩
        · rw [e_ml, hchunk, UInt64.ofNat_add]
        · exact e_frame

/-- **Corollary**: the inner extend loop computes the model's byte-by-byte
    `extendFrom`, via the cooperative-extend keystone `extendChunk_eq_extendFrom`. -/
theorem inner_extend_extendFrom (inStride endCap ib : Nat) (g : Array UInt8) (pv cv : UInt64)
    (F fuelF m : Nat) (st : WState)
    (hstride : inStride ≤ 65536) (hendCap : endCap ≤ inStride) (hm : m ≤ inStride)
    (hCfuel : endCap - (pv.toNat + m) ≤ 32 * F) (hFfuel : endCap - (pv.toNat + m) ≤ fuelF)
    (hg : st.gmem = g) (hib : (st.regs "inBase").toNat = ib)
    (hp : st.regs "p0" = pv) (hc : st.regs "cand0" = cv)
    (hextC : st.regs "extC" = 1) (hml : st.regs "ml" = UInt64.ofNat m) :
    ((WStmt.uwhile "extC" (innerExtBody inStride endCap)).eval F st).regs "ml"
      = UInt64.ofNat (extendFrom (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap fuelF m) := by
  rw [(inner_extend inStride endCap ib g pv cv hstride hendCap F m st hm hg hib hp hc hextC hml).1,
      extendChunk_eq_extendFrom (gmemInpAt g ib inStride) pv.toNat cv.toNat endCap F fuelF m
        hCfuel hFfuel]

/-- Telescoping: encoding a plan-block splits off the first sequence's encoding. -/
theorem encode_planBlockFrom_cons (inp : List UInt8) (anchor : Nat)
    (step : PlanStep) (rest : List PlanStep) (fl : Nat) :
    (planBlockFrom inp anchor (step :: rest) fl).encode
      = encodeSeq ⟨(inp.drop anchor).take step.litLen, step.offset, step.mlen⟩
        ++ (planBlockFrom inp (anchor + step.litLen + step.mlen) rest fl).encode := by
  simp only [planBlockFrom, Block.encode, List.flatMap_cons, List.append_assoc]

/-- Base case: an empty plan-block encodes to just the final literal run. -/
theorem encode_planBlockFrom_nil (inp : List UInt8) (anchor fl : Nat) :
    (planBlockFrom inp anchor [] fl).encode
      = encodeFinal ((inp.drop anchor).take fl) := by
  simp only [planBlockFrom, Block.encode, List.flatMap_nil, List.nil_append]

end AlgorithmLib.LZ4WarpDSL
