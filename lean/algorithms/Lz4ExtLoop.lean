import Lz4ExtGuard

set_option maxRecDepth 8192

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

variable {p : Array SInstr} [Shape p] {S : Nat} [geo : Geo p S]
-- ── The extend loop, as a region ────────────────────────────────────────────


theorem extS_entry_lt : ∀ q, q < 274 → q ∉ extS →
    ∀ q' ∈ succsOf p q, q' ∈ extS → q' = 94 :=
  ivEntry_at p 94 27 94 (Shape.size (p := p)) (by
    have hs := Shape.entryShape (p := p)
    simp only [entryShapeB, Bool.and_eq_true] at hs
    exact hs.1.1.2)

theorem extS_entry : ∀ q, q ∉ extS →
    ∀ q', q' ∈ succsOf p q → q' ∈ extS → q' = 94 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact extS_entry_lt q h hq q' hq' hin
  · rw [show succsOf p q = [q] from by
      simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

/-- The constants and carried facts of the extend region.  `ec1`/`ecR` are set on
    entry and never rewritten; `p0`/`cand0` are what the select handed over. -/
def ExtInv (S : Nat) (st : SState) : Prop :=
  (96 ≤ st.pc → ∀ l : Lane, st.regs "ec1" l = UInt64.ofNat (S - 6))
  ∧ (95 ≤ st.pc → ∀ l : Lane, st.regs "ecR" l = UInt64.ofNat (S - 5))
  ∧ (∀ l : Lane, (st.regs "p0" l).toNat < (S - 12))
  ∧ (∀ l : Lane, (st.regs "cand0" l).toNat < (st.regs "p0" l).toNat)



/-- **The extend region's carried facts, at every state inside it.** -/
theorem ext_inv (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ extS →
      ExtInv S (siter p k (initSt w inPtr outPtr gm smemB)) := by
  have h := Geo.extShape (p := p) (S := S)
  simp only [extShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨e94, e95⟩, np0⟩, ncand0⟩, nec1⟩, necR⟩, _⟩ := h
  refine inv_in_region extS 94 (ExtInv S) extS_entry (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ extS; decide) ?_
  · -- preservation
    intro jj hst _ hI
    rw [siter_succ]
    obtain ⟨st, hsteq⟩ : ∃ x, siter p jj (initSt w inPtr outPtr gm smemB) = x := ⟨_, rfl⟩
    rw [hsteq] at hst hI ⊢
    have hb : 94 ≤ st.pc ∧ st.pc ≤ 120 := by
      simp only [extS, List.mem_map, List.mem_range] at hst
      obtain ⟨j, hj, hje⟩ := hst; omega
    have frame : ∀ r : String, (p[st.pc]?.map (fun i => destOf i != some r)) = some true →
        (sstep p st).regs r = st.regs r := by
      intro r hr
      exact sstep_regs_frame p st r (fun i hi => by rw [hi] at hr; simpa using hr)
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro hlo l
      rcases Nat.lt_or_ge st.pc 96 with hs | hs
      · -- `95 mov ec1, (S - 6)` establishes it; from 94 the goal is vacuous
        rcases Nat.lt_or_ge st.pc 95 with hs2 | hs2
        · exfalso
          have : st.pc = 94 := by omega
          rw [sstep, show p[st.pc]? = some (SInstr.mov "ecR" (.imm (S - 5))) from by
            rw [this]; exact e94] at hlo
          simp only [sstepInstr, SState.setReg, SState.setPc] at hlo
          omega
        · have h95 : st.pc = 95 := by omega
          rw [sstep, show p[st.pc]? = some (SInstr.mov "ec1" (.imm (S - 6))) from by
            rw [h95]; exact e95]
          simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
      · rw [congrFun (frame "ec1" (nec1 st.pc (by
          simp only [List.mem_map, List.mem_range]; exact ⟨st.pc - 96, by omega, by omega⟩))) l]
        exact hI.1 hs l
    · intro hlo l
      rcases Nat.lt_or_ge st.pc 95 with hs | hs
      · have h94' : st.pc = 94 := by omega
        rw [sstep, show p[st.pc]? = some (SInstr.mov "ecR" (.imm (S - 5))) from by
          rw [h94']; exact e94]
        simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
      · rw [congrFun (frame "ecR" (necR st.pc (by
          simp only [List.mem_map, List.mem_range]; exact ⟨st.pc - 95, by omega, by omega⟩))) l]
        exact hI.2.1 hs l
    · intro l
      rw [congrFun (frame "p0" (np0 st.pc hst)) l]; exact hI.2.2.1 l
    · intro l
      rw [congrFun (frame "p0" (np0 st.pc hst)) l,
        congrFun (frame "cand0" (ncand0 st.pc hst)) l]
      exact hI.2.2.2 l
  · -- entry
    intro j hj
    refine ⟨fun hlo => absurd hlo (by rw [hj]; omega), fun hlo => absurd hlo (by rw [hj]; omega),
      fun l => (extend_entry (S := S) w inPtr outPtr gm smemB j hj l).1,
      fun l => (extend_entry (S := S) w inPtr outPtr gm smemB j hj l).2.1⟩

-- ── `peD`: clamped by the kernel, like the two search pointers ──────────────



/-- **`peD ≤ inStride - 6` at the load site.**  `103 peC := min pe ec1` with
    `ec1 = (S - 6)`, and `106` just copies it — so like `rp` and `rc`, the bound is
    by construction; all the region invariant has to supply is the constant. -/
theorem peD_le (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (k : Nat) (h110 : (siter p k (initSt w inPtr outPtr gm smemB)).pc = 110) (l : Lane) :
    ((siter p k (initSt w inPtr outPtr gm smemB)).regs "peD" l).toNat ≤ (S - 6) := by
  have h := Shape.extLoadShape (p := p)
  simp only [extLoadShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨hft, npeD⟩, npeC⟩, i103⟩, i106⟩ := h
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  have hf : ∀ t, t < 8 → (p[110 - t]?.map fallthroughOnlyB) = some true := by
    intro t ht
    exact hft (110 - t) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨10 - t, by omega, by omega⟩)
  have hnD : ∀ t, t < 3 → (p[110 - t - 1]?.map (fun i => destOf i != some "peD")) = some true := by
    intro t ht
    exact npeD (110 - t - 1) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨2 - t, by omega, by omega⟩)
  have hnC : ∀ t, t < 6 → (p[110 - t - 1]?.map (fun i => destOf i != some "peC")) = some true := by
    intro t ht
    exact npeC (110 - t - 1) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨5 - t, by omega, by omega⟩)
  -- `peD` at 110 is `peC` at the `mov`, and `peC` there is `peC` at 110
  have hD := regs_back p _ hinit "peD" 110 3 (by have := Geo.sBound (p := p) (S := S); omega) (fun t ht => hf t (by have := Geo.sBound (p := p) (S := S); omega)) hnD k h110
  have hstD := pre_state p _ hinit 110 3 (SInstr.mov "peD" (.reg "peC"))
    (fun t ht => hf t (by have := Geo.sBound (p := p) (S := S); omega)) (by rw [show 110 - 3 - 1 = 106 from by omega]; exact i106)
    (by have := Geo.sBound (p := p) (S := S); omega) k h110
  have hC6 := regs_back p _ hinit "peC" 110 6 (by have := Geo.sBound (p := p) (S := S); omega) (fun t ht => hf t (by have := Geo.sBound (p := p) (S := S); omega)) hnC k h110
  have hC4 := regs_back p _ hinit "peC" 110 4 (by have := Geo.sBound (p := p) (S := S); omega) (fun t ht => hf t (by have := Geo.sBound (p := p) (S := S); omega))
    (fun t ht => hnC t (by have := Geo.sBound (p := p) (S := S); omega)) k h110
  have hstC := pre_state p _ hinit 110 6 (SInstr.binr .min "peC" "pe" "ec1")
    (fun t ht => hf t (by have := Geo.sBound (p := p) (S := S); omega)) (by rw [show 110 - 6 - 1 = 103 from by omega]; exact i103)
    (by have := Geo.sBound (p := p) (S := S); omega) k h110
  rw [show k - 3 - 1 = k - 4 from by omega] at hstD
  rw [show k - 6 - 1 = k - 7 from by omega] at hstC
  -- the state on the `min` stands at 103, inside the extend region
  have hec1 : (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).regs "ec1" l
      = UInt64.ofNat (S - 6) := by
    refine (ext_inv (S := S) w inPtr outPtr gm smemB (k - 7) ?_).1 (by rw [hstC.2.1]; omega) l
    rw [hstC.2.1]
    simp only [extS, List.mem_map, List.mem_range]
    exact ⟨9, by omega, by omega⟩
  rw [congrFun hD l, hstD.1]
  simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
  rw [← congrFun hC4 l, congrFun hC6 l, hstC.1]
  simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
  refine Nat.le_trans (min_run_le_right _ _) ?_
  rw [hec1, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (S - 6)
    (by have := Geo.sBound (p := p) (S := S); omega)]
  exact Nat.le_refl _

-- ── Why the extend loop cannot run away ─────────────────────────────────────



/-- **A full round of the extend loop means every lane was still inside the
    window** — so the loop's own continue-condition bounds `ml`.

    `adv = 32` is `clz(brev(~ballot(pOk))) = 32`, i.e. no lane failed; each lane's
    `pOk` implies its `pIn`, which is `pe < ecR`; and `pe` is `p0 + ml + lane`.
    Taking the last lane gives `p0 + ml + 31 < (S - 5)`, so the next `ml`, larger by
    exactly 32, still satisfies the budget.  Without this the loop could in
    principle drive `pe` past `2^64` and wrap, which is the only way `peC - p0`
    could underflow and put the candidate read out of the stride. -/
theorem ml_step (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (k : Nat) (h118 : (siter p k (initSt w inPtr outPtr gm smemB)).pc = 118)
    (hbnd : ∀ l : Lane, ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l).toNat
      + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" l).toNat ≤ (S - 5))
    (l : Lane) :
    ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l).toNat
      + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" l).toNat
      + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "adv" l).toNat ≤ (S - 5) := by
  have h := Shape.advShape (p := p)
  simp only [advShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨hftl, hfrm⟩, i100⟩, i101⟩, i102⟩, i113⟩, i114⟩, i115⟩, i116⟩, i117⟩ := h
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  have hf : ∀ t, t < 19 → (p[118 - t]?.map fallthroughOnlyB) = some true := by
    intro t ht
    exact hftl (118 - t) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨18 - t, by omega, by omega⟩)
  have back : ∀ (r : String) (lo n : Nat),
      (r, lo, n) ∈ [("revM", 117, 1), ("mis", 116, 2), ("balOk", 115, 3), ("pOk", 114, 4),
        ("pIn", 103, 15), ("pe", 102, 16), ("ecR", 102, 16), ("p0", 101, 17),
        ("idx", 101, 17), ("ml", 100, 18), ("lane", 100, 18)] → lo + n = 118 → n ≤ 18 →
      ∀ m, m ≤ n →
      (siter p k (initSt w inPtr outPtr gm smemB)).regs r
        = (siter p (k - m) (initSt w inPtr outPtr gm smemB)).regs r := by
    intro r lo n hm hn hn18 m hmn
    refine regs_back p _ hinit r 118 m (by have := Geo.sBound (p := p) (S := S); omega) (fun t ht => hf t (by have := Geo.sBound (p := p) (S := S); omega)) ?_ k h118
    intro t ht
    exact hfrm (r, lo, n) hm (118 - t - 1) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨n - 1 - t, by omega, by omega⟩)
  have pre : ∀ (n : Nat) (i : SInstr), n ≤ 18 → p[118 - n - 1]? = some i →
      siter p (k - n) (initSt w inPtr outPtr gm smemB)
        = sstepInstr p i (siter p (k - n - 1) (initSt w inPtr outPtr gm smemB))
      ∧ (siter p (k - n - 1) (initSt w inPtr outPtr gm smemB)).pc = 118 - n - 1 := by
    intro n i hn hi
    have hx := pre_state p _ hinit 118 n i (fun t ht => hf t (by have := Geo.sBound (p := p) (S := S); omega)) hi (by have := Geo.sBound (p := p) (S := S); omega) k h118
    exact ⟨hx.1, hx.2.1⟩
  -- unwind `adv` back to the ballot of `pOk`
  have hadv118 : ∀ x : Lane, (siter p k (initSt w inPtr outPtr gm smemB)).regs "adv" x
      = clz32 (brev32 (~~~ (ballotOf (siter p k (initSt w inPtr outPtr gm smemB)).regs "pOk"))) := by
    intro x
    have e117 := pre 0 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 118 - 0 - 1 = 117 from by omega]; exact i117)
    have e116 := pre 1 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 118 - 1 - 1 = 116 from by omega]; exact i116)
    have e115 := pre 2 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 118 - 2 - 1 = 115 from by omega]; exact i115)
    have e114 := pre 3 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 118 - 3 - 1 = 114 from by omega]; exact i114)
    have hrevM := back "revM" 117 1 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 1 (by have := Geo.sBound (p := p) (S := S); omega)
    have hmis := back "mis" 116 2 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 2 (by have := Geo.sBound (p := p) (S := S); omega)
    have hbalOk := back "balOk" 115 3 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 3 (by have := Geo.sBound (p := p) (S := S); omega)
    have hpOk := back "pOk" 114 4 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 4 (by have := Geo.sBound (p := p) (S := S); omega)
    rw [show k - 0 = k from by omega] at e117
    rw [show k - 1 - 1 = k - 2 from by omega] at e116
    rw [show k - 2 - 1 = k - 3 from by omega] at e115
    rw [show k - 3 - 1 = k - 4 from by omega] at e114
    have a1 : (siter p k (initSt w inPtr outPtr gm smemB)).regs "adv" x
        = clz32 ((siter p (k - 1) (initSt w inPtr outPtr gm smemB)).regs "revM" x) := by
      rw [e117.1]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
    have a2 : (siter p (k - 1) (initSt w inPtr outPtr gm smemB)).regs "revM" x
        = brev32 ((siter p (k - 2) (initSt w inPtr outPtr gm smemB)).regs "mis" x) := by
      rw [e116.1]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
    have a3 : (siter p (k - 2) (initSt w inPtr outPtr gm smemB)).regs "mis" x
        = ~~~ ((siter p (k - 3) (initSt w inPtr outPtr gm smemB)).regs "balOk" x) := by
      rw [e115.1]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
    have a4 : (siter p (k - 3) (initSt w inPtr outPtr gm smemB)).regs "balOk" x
        = ballotOf (siter p (k - 4) (initSt w inPtr outPtr gm smemB)).regs "pOk" := by
      rw [e114.1]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
    rw [a1, a2, a3, a4, ballotOf_congr _ _ "pOk" hpOk.symm]

  -- every lane matched
  have hpOk1 : ∀ x : Lane,
      x.val < ((siter p k (initSt w inPtr outPtr gm smemB)).regs "adv" l).toNat →
      (siter p k (initSt w inPtr outPtr gm smemB)).regs "pOk" x = 1 := by
    intro x hx
    refine ballot_below_clz_not _ "pOk" x ?_
    rw [← hadv118 l]
    exact hx
  -- every lane was still inside the window
  have e113 := pre 4 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 118 - 4 - 1 = 113 from by omega]; exact i113)
  have e102 := pre 15 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 118 - 15 - 1 = 102 from by omega]; exact i102)
  have e101 := pre 16 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 118 - 16 - 1 = 101 from by omega]; exact i101)
  have e100 := pre 17 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 118 - 17 - 1 = 100 from by omega]; exact i100)
  rw [show k - 4 - 1 = k - 5 from by omega] at e113
  rw [show k - 15 - 1 = k - 16 from by omega] at e102
  rw [show k - 16 - 1 = k - 17 from by omega] at e101
  rw [show k - 17 - 1 = k - 18 from by omega] at e100
  have hpIn := back "pIn" 103 15 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega)
  have hpe := back "pe" 102 16 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega)
  have hecR := back "ecR" 102 16 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega)
  have hp0 := back "p0" 101 17 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega)
  have hidx := back "idx" 101 17 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega)
  have hml := back "ml" 100 18 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega)
  have hlane := back "lane" 100 18 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega)
  -- the argument at the last lane, then spread by uniformity
  have key : ∀ x : Lane,
      x.val < ((siter p k (initSt w inPtr outPtr gm smemB)).regs "adv" l).toNat →
      ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" x).toNat
      + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" x).toNat + x.val < (S - 5) := by
    intro x hxlt
    -- `pOk` at the `andp` gives `pIn`
    have hp : (siter p (k - 5) (initSt w inPtr outPtr gm smemB)).regs "pIn" x = 1 := by
      have hx := hpOk1 x hxlt
      rw [congrFun (back "pOk" 114 4 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 4 (by have := Geo.sBound (p := p) (S := S); omega)) x,
        e113.1] at hx
      simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true] at hx
      by_cases hc : (siter p (k - 5) (initSt w inPtr outPtr gm smemB)).regs "pIn" x == 1
          ∧ (siter p (k - 5) (initSt w inPtr outPtr gm smemB)).regs "pEqB" x == 1
      · exact beq_iff_eq.mp hc.1
      · rw [if_neg hc] at hx; exact absurd hx (by decide)
    -- `pIn` at the `setp` gives `pe < ecR`
    have hlt : (siter p (k - 16) (initSt w inPtr outPtr gm smemB)).regs "pe" x
        < (siter p (k - 16) (initSt w inPtr outPtr gm smemB)).regs "ecR" x := by
      have hx : (siter p (k - 15) (initSt w inPtr outPtr gm smemB)).regs "pIn" x = 1 := by
        rw [← congrFun (back "pIn" 103 15 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 15 (by have := Geo.sBound (p := p) (S := S); omega)) x,
          congrFun (back "pIn" 103 15 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 5 (by have := Geo.sBound (p := p) (S := S); omega)) x]
        exact hp
      rw [e102.1] at hx
      simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SCmp.run,
        if_pos rfl, if_true] at hx
      by_cases hc : (siter p (k - 16) (initSt w inPtr outPtr gm smemB)).regs "pe" x
          < (siter p (k - 16) (initSt w inPtr outPtr gm smemB)).regs "ecR" x
      · exact hc
      · rw [if_neg (by simpa using hc)] at hx; exact absurd hx (by decide)
    -- `pe = p0 + idx`, `idx = ml + lane`, and neither addition wraps
    have hidxv : (siter p (k - 17) (initSt w inPtr outPtr gm smemB)).regs "idx" x
        = (siter p (k - 18) (initSt w inPtr outPtr gm smemB)).regs "ml" x
          + (siter p (k - 18) (initSt w inPtr outPtr gm smemB)).regs "lane" x := by
      rw [e100.1]
      simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true]
    have hpev : (siter p (k - 16) (initSt w inPtr outPtr gm smemB)).regs "pe" x
        = (siter p (k - 17) (initSt w inPtr outPtr gm smemB)).regs "p0" x
          + (siter p (k - 17) (initSt w inPtr outPtr gm smemB)).regs "idx" x := by
      rw [e101.1]
      simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true]
    have hlanev : (siter p (k - 18) (initSt w inPtr outPtr gm smemB)).regs "lane" x
        = UInt64.ofNat x.val := by
      rw [← congrFun (hlane 18 (by have := Geo.sBound (p := p) (S := S); omega)) x]
      exact lane_val w inPtr outPtr gm smemB k (by rw [h118]; omega) x
    have hecRv : (siter p (k - 16) (initSt w inPtr outPtr gm smemB)).regs "ecR" x
        = UInt64.ofNat (S - 5) := by
      rw [← congrFun (hecR 16 (by have := Geo.sBound (p := p) (S := S); omega)) x]
      refine (ext_inv (S := S) w inPtr outPtr gm smemB k ?_).2.1 (by rw [h118]; omega) x
      rw [h118]; simp only [extS, List.mem_map, List.mem_range]; exact ⟨24, by omega, by omega⟩
    have hb := hbnd x
    have hxl : x.val < 32 := x.isLt
    have hmlv : (siter p (k - 18) (initSt w inPtr outPtr gm smemB)).regs "ml" x
        = (siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" x := (congrFun (hml 18 (by have := Geo.sBound (p := p) (S := S); omega)) x).symm
    have hp0v : (siter p (k - 17) (initSt w inPtr outPtr gm smemB)).regs "p0" x
        = (siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" x := (congrFun (hp0 17 (by have := Geo.sBound (p := p) (S := S); omega)) x).symm
    have hidxn : ((siter p (k - 17) (initSt w inPtr outPtr gm smemB)).regs "idx" x).toNat
        = ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" x).toNat + x.val := by
      rw [hidxv, hlanev, hmlv, UInt64.toNat_add,
        AlgorithmLib.LZ4Ptx.toNat_ofNat_lt x.val (by have := Geo.sBound (p := p) (S := S); omega), Nat.mod_eq_of_lt (by have := Geo.sBound (p := p) (S := S); omega)]
    have hpen : ((siter p (k - 16) (initSt w inPtr outPtr gm smemB)).regs "pe" x).toNat
        = ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" x).toNat
          + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" x).toNat + x.val := by
      have hlt64 : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" x).toNat
          + (((siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" x).toNat + x.val)
          < 2 ^ 64 := by have := Geo.sBound (p := p) (S := S); omega
      rw [hpev, UInt64.toNat_add, hidxn, hp0v, Nat.mod_eq_of_lt hlt64]
      omega
    have := UInt64.lt_iff_toNat_lt.mp hlt
    rw [hpen, hecRv, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (S - 5) (by have := Geo.sBound (p := p) (S := S); omega)] at this
    exact this
  -- `adv` is the first lane that failed; every lane below it was still in range
  have hadvle : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "adv" l).toNat ≤ 32 := by
    rw [hadv118 l, clz32]
    split
    · rename_i i _
      rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by have := Geo.sBound (p := p) (S := S); omega)]; omega
    · rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by have := Geo.sBound (p := p) (S := S); omega)]; omega
  rcases Nat.eq_zero_or_pos ((siter p k (initSt w inPtr outPtr gm smemB)).regs "adv" l).toNat
    with h0 | hpos
  · rw [h0]; have := hbnd l; omega
  · obtain ⟨xa, hxa⟩ : ∃ x : Lane,
        x.val = ((siter p k (initSt w inPtr outPtr gm smemB)).regs "adv" l).toNat - 1 :=
      ⟨⟨((siter p k (initSt w inPtr outPtr gm smemB)).regs "adv" l).toNat - 1,
        by show _ < 32; omega⟩, rfl⟩
    have hk := key xa (by have := Geo.sBound (p := p) (S := S); omega)
    rw [hxa, uni_at w inPtr outPtr gm smemB k "p0" (by simp [uniR]) xa l,
      uni_at w inPtr outPtr gm smemB k "ml" (by simp [uniR]) xa l] at hk
    omega


theorem sstep_pc_succ (st : SState) (i : SInstr) (hi : p[st.pc]? = some i)
    (hs : isStraightB i = true) : (sstep p st).pc = st.pc + 1 := by
  rw [sstep, hi]
  cases i <;> first | rfl | simp [isStraightB] at hs

/-- The `ml` budget, in the four forms the loop's control flow needs.  It is
    unconditional only between 100 and 118; at the loop head it is contingent on
    the continue flag, because the increment at 118 happens before the test. -/
def MlInv (S : Nat) (st : SState) : Prop :=
  97 ≤ st.pc → st.pc ≤ 120 → ∀ l : Lane,
    (st.regs "p0" l).toNat + (st.regs "ml" l).toNat ≤ (S - 5)



/-- **The `ml` budget holds throughout the extend region.** -/
theorem ml_inv (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ extS →
      MlInv S (siter p k (initSt w inPtr outPtr gm smemB)) := by
  have h := Shape.mlShape (p := p)
  simp only [mlShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨m96, m99⟩, m118⟩, m119⟩, m120⟩, m98⟩, mlx5⟩, np0⟩, nml⟩, nextC⟩, nadv⟩,
    nstr⟩ := h
  refine inv_in_region extS 94 (MlInv S) extS_entry (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ extS; decide) ?_
  · intro jj hst hst' hI
    have hb : 94 ≤ (siter p jj (initSt w inPtr outPtr gm smemB)).pc
        ∧ (siter p jj (initSt w inPtr outPtr gm smemB)).pc ≤ 120 := by
      simp only [extS, List.mem_map, List.mem_range] at hst
      obtain ⟨j, hj, hje⟩ := hst; omega
    have frame : ∀ r : String,
        (p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?.map
          (fun i => destOf i != some r)) = some true →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs r
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs r := by
      intro r hr
      rw [siter_succ]
      exact sstep_regs_frame p _ r (fun i hi => by rw [hi] at hr; simpa using hr)
    have hp0f := frame "p0" (np0 _ hst)
    have hmlf : ∀ q, ((97 ≤ q ∧ q ≤ 117) ∨ q = 119 ∨ q = 120) →
        (siter p jj (initSt w inPtr outPtr gm smemB)).pc = q →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs "ml"
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs "ml" := by
      intro q hq hpc
      refine frame "ml" (nml _ ?_)
      rw [hpc]
      simp only [List.mem_append, List.mem_map, List.mem_range, List.mem_cons,
        List.not_mem_nil, or_false]
      rcases hq with ⟨h1, h2⟩ | h | h
      · exact Or.inl ⟨q - 97, by omega, by omega⟩
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr h)
    have hstr : ∀ q, ((94 ≤ q ∧ q ≤ 98) ∨ (100 ≤ q ∧ q ≤ 119)) →
        (siter p jj (initSt w inPtr outPtr gm smemB)).pc = q →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).pc = q + 1 := by
      intro q hq hpc
      have hx : p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?.map isStraightB
          = some true := by
        rw [hpc]
        refine nstr q ?_
        simp only [List.mem_append, List.mem_map, List.mem_range]
        rcases hq with ⟨a, b⟩ | ⟨a, b⟩
        · exact Or.inl ⟨q - 94, by omega, by omega⟩
        · exact Or.inr ⟨q - 100, by omega, by omega⟩
      cases hi : p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]? with
      | none => rw [hi] at hx; exact absurd hx (by simp)
      | some i =>
          rw [hi] at hx
          rw [siter_succ, sstep_pc_succ _ i hi (by simpa using hx), hpc]
    -- the loop's back edge
    have h120 : (siter p jj (initSt w inPtr outPtr gm smemB)).pc = 120 →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).pc = 98 := by
      intro hpc
      rw [siter_succ, sstep, show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
        = some (SInstr.bra "Lh4") from by rw [hpc]; exact m120]
      simp only [sstepInstr, SState.setPc]
      exact Shape.lh4 (p := p)
    obtain ⟨P, hP⟩ : ∃ q, (siter p jj (initSt w inPtr outPtr gm smemB)).pc = q := ⟨_, rfl⟩
    rw [hP] at hb
    have hml96 : P = 96 → ∀ x : Lane,
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs "ml" x = UInt64.ofNat 4 := by
      intro h x
      rw [siter_succ, sstep, show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
        = some (SInstr.mov "ml" (.imm 4)) from by rw [hP, h]; exact m96]
      simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
    -- where the next step can land, once and for all
    have h99b : P = 99 →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).pc = 100 ∨
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).pc = 121 := by
      intro hpc
      rw [siter_succ, sstep, show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
        = some (SInstr.braifnot "extC" "Lx5") from by rw [hP, hpc]; exact m99]
      simp only [sstepInstr, SState.setPc]
      by_cases hc : (siter p jj (initSt w inPtr outPtr gm smemB)).regs "extC" 0 == 1
      · rw [if_pos hc, hP, hpc]; exact Or.inl rfl
      · rw [if_neg hc, mlx5]; exact Or.inr rfl
    have h99c : P = 99 → (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).pc = 100 →
        (siter p jj (initSt w inPtr outPtr gm smemB)).regs "extC" 0 = 1 := by
      intro hpc h100
      by_cases hc : (siter p jj (initSt w inPtr outPtr gm smemB)).regs "extC" 0 == 1
      · exact beq_iff_eq.mp hc
      · exfalso
        rw [siter_succ, sstep, show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
          = some (SInstr.braifnot "extC" "Lx5") from by rw [hP, hpc]; exact m99] at h100
        simp only [sstepInstr, SState.setPc] at h100
        rw [if_neg hc, mlx5] at h100
        omega
    obtain ⟨N, hN⟩ : ∃ q, (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).pc = q := ⟨_, rfl⟩
    have hsucc : (P ≤ 98 ∧ N = P + 1) ∨ (P = 99 ∧ N = 100) ∨ (P = 99 ∧ N = 121)
        ∨ (100 ≤ P ∧ P ≤ 119 ∧ N = P + 1) ∨ (P = 120 ∧ N = 98) := by
      rcases Nat.lt_or_ge P 99 with h1 | h1
      · exact Or.inl ⟨by omega, by rw [← hN]; exact hstr P (Or.inl ⟨by omega, by omega⟩) hP⟩
      · rcases Nat.lt_or_ge P 100 with h2 | h2
        · rcases h99b (by have := Geo.sBound (p := p) (S := S); omega) with h | h
          · exact Or.inr (Or.inl ⟨by omega, by rw [← hN]; exact h⟩)
          · exact Or.inr (Or.inr (Or.inl ⟨by omega, by rw [← hN]; exact h⟩))
        · rcases Nat.lt_or_ge P 120 with h3 | h3
          · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨by omega, by omega,
              by rw [← hN]; exact hstr P (Or.inr ⟨by omega, by omega⟩) hP⟩)))
          · exact Or.inr (Or.inr (Or.inr (Or.inr ⟨by omega,
              by rw [← hN]; exact h120 (by have := Geo.sBound (p := p) (S := S); omega)⟩)))
    -- the goal is one clause now: the budget between 97 and 120
    intro hlo hhi x
    rw [hN] at hlo hhi
    have hframe : 97 ≤ P → P ≠ 118 →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs "ml" x
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs "ml" x := by
      intro h97p hne
      have hrange : (97 ≤ P ∧ P ≤ 117) ∨ P = 119 ∨ P = 120 := by
        rcases hsucc with ⟨a, b⟩ | ⟨a, b⟩ | ⟨a, b⟩ | ⟨a, a', b⟩ | ⟨a, b⟩ <;> omega
      exact congrFun (hmlf P hrange hP) x
    rcases Nat.lt_or_ge P 97 with hlt97 | hge97
    · -- only 96 → 97 can enter the stretch from below
      have hPv : P = 96 := by
        rcases hsucc with ⟨a, b⟩ | ⟨a, b⟩ | ⟨a, b⟩ | ⟨a, a', b⟩ | ⟨a, b⟩ <;> omega
      rw [congrFun hp0f x, hml96 hPv x]
      have := (ext_inv (S := S) w inPtr outPtr gm smemB jj hst).2.2.1 x
      rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt 4 (by have := Geo.sBound (p := p) (S := S); omega)]
      omega
    · by_cases h118 : P = 118
      · -- the increment, bounded by the loop's own exit test
        have hb118 := hI (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega)
        have hstep := ml_step w inPtr outPtr gm smemB jj (by rw [hP, h118]) hb118 x
        have hmlv : (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs "ml" x
            = (siter p jj (initSt w inPtr outPtr gm smemB)).regs "ml" x
              + (siter p jj (initSt w inPtr outPtr gm smemB)).regs "adv" x := by
          rw [siter_succ, sstep, show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
            = some (SInstr.bin .add "ml" "ml" (.reg "adv")) from by rw [hP, h118]; exact m118]
          simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run,
            if_pos rfl, if_true]
        rw [congrFun hp0f x, hmlv, UInt64.toNat_add, Nat.mod_eq_of_lt (by have := Geo.sBound (p := p) (S := S); omega)]
        omega
      · rw [congrFun hp0f x, hframe hge97 h118]
        exact hI (by have := Geo.sBound (p := p) (S := S); omega) (by
          rcases hsucc with ⟨a, b⟩ | ⟨a, b⟩ | ⟨a, b⟩ | ⟨a, a', b⟩ | ⟨a, b⟩ <;> omega) x
  · intro j hj h97
    rw [hj] at h97
    omega

-- ── `caD`: bounded by the select's guarantee, not by a clamp ────────────────



/-- **`caD ≤ inStride - 6` at the candidate load.**

    This is the one address in the kernel with no clamp of its own.  What bounds
    it is the *select*: `cand0 < p0` (the chosen lane's candidate is strictly
    behind its position) and `p0 ≤ peC` (the extend never walks backwards,
    because `p0 + idx` cannot wrap — that is what the `ml` budget is for).  Then
    `caD = cand0 + (peC - p0) < peC ≤ ec1`. -/
theorem caD_le (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (k : Nat) (h111 : (siter p k (initSt w inPtr outPtr gm smemB)).pc = 111) (l : Lane) :
    ((siter p k (initSt w inPtr outPtr gm smemB)).regs "caD" l).toNat ≤ (S - 6) := by
  have h := Shape.caShape (p := p)
  simp only [caShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨hftl, hfrm⟩, c100⟩, c101⟩, c103⟩, c104⟩, c105⟩, c108⟩ := h
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  have hf : ∀ t, t < 12 → (p[111 - t]?.map fallthroughOnlyB) = some true := by
    intro t ht
    exact hftl (111 - t) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨11 - t, by omega, by omega⟩)
  have back : ∀ (r : String) (lo n : Nat),
      (r, lo, n) ∈ [("caD", 109, 2), ("caC", 106, 5), ("dfe", 105, 6), ("peC", 104, 7),
        ("pe", 102, 9), ("idx", 101, 10), ("p0", 100, 11), ("cand0", 100, 11),
        ("ec1", 100, 11), ("ml", 100, 11), ("lane", 100, 11)] → lo + n = 111 → n ≤ 11 →
      ∀ m, m ≤ n →
      (siter p k (initSt w inPtr outPtr gm smemB)).regs r
        = (siter p (k - m) (initSt w inPtr outPtr gm smemB)).regs r := by
    intro r lo n hm hn hn11 m hmn
    refine regs_back p _ hinit r 111 m (by have := Geo.sBound (p := p) (S := S); omega) (fun t ht => hf t (by have := Geo.sBound (p := p) (S := S); omega)) ?_ k h111
    intro t ht
    exact hfrm (r, lo, n) hm (111 - t - 1) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨n - 1 - t, by omega, by omega⟩)
  have pre : ∀ (n : Nat) (i : SInstr), n ≤ 11 → p[111 - n - 1]? = some i →
      siter p (k - n) (initSt w inPtr outPtr gm smemB)
        = sstepInstr p i (siter p (k - n - 1) (initSt w inPtr outPtr gm smemB)) := by
    intro n i hn hi
    exact (pre_state p _ hinit 111 n i (fun t ht => hf t (by have := Geo.sBound (p := p) (S := S); omega)) hi (by have := Geo.sBound (p := p) (S := S); omega) k h111).1
  -- region facts at this state
  have hreg := ext_inv (S := S) w inPtr outPtr gm smemB k (by
    rw [h111]; simp only [extS, List.mem_map, List.mem_range]; exact ⟨17, by omega, by omega⟩)
  have hml := ml_inv (S := S) w inPtr outPtr gm smemB k (by
    rw [h111]; simp only [extS, List.mem_map, List.mem_range]; exact ⟨17, by omega, by omega⟩)
    (by rw [h111]; omega) (by rw [h111]; omega)
  have hec1 := hreg.1 (by rw [h111]; omega)
  have hp0 := hreg.2.2.1
  have hcand0 := hreg.2.2.2
  have hlane := lane_val w inPtr outPtr gm smemB k (by rw [h111]; omega)
  -- the value chain, each register expressed at this state
  have vidx : (siter p k (initSt w inPtr outPtr gm smemB)).regs "idx" l
      = (siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" l
        + (siter p k (initSt w inPtr outPtr gm smemB)).regs "lane" l := by
    rw [congrFun (back "idx" 101 10 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 10 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      pre 10 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 111 - 10 - 1 = 100 from by omega]; exact c100)]
    simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true]
    rw [show k - 10 - 1 = k - 11 from by omega,
      congrFun (back "ml" 100 11 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 11 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      congrFun (back "lane" 100 11 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 11 (by have := Geo.sBound (p := p) (S := S); omega)) l]
  have vpe : (siter p k (initSt w inPtr outPtr gm smemB)).regs "pe" l
      = (siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l
        + (siter p k (initSt w inPtr outPtr gm smemB)).regs "idx" l := by
    rw [congrFun (back "pe" 102 9 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 9 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      pre 9 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 111 - 9 - 1 = 101 from by omega]; exact c101)]
    simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true]
    rw [show k - 9 - 1 = k - 10 from by omega,
      congrFun (back "p0" 100 11 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 10 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      congrFun (back "idx" 101 10 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 10 (by have := Geo.sBound (p := p) (S := S); omega)) l]
  have vpeC : (siter p k (initSt w inPtr outPtr gm smemB)).regs "peC" l
      = SOp.run .min ((siter p k (initSt w inPtr outPtr gm smemB)).regs "pe" l)
          ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ec1" l) := by
    rw [congrFun (back "peC" 104 7 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 7 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      pre 7 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 111 - 7 - 1 = 103 from by omega]; exact c103)]
    simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
    rw [show k - 7 - 1 = k - 8 from by omega,
      congrFun (back "pe" 102 9 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 8 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      congrFun (back "ec1" 100 11 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 8 (by have := Geo.sBound (p := p) (S := S); omega)) l]
  have vdfe : (siter p k (initSt w inPtr outPtr gm smemB)).regs "dfe" l
      = (siter p k (initSt w inPtr outPtr gm smemB)).regs "peC" l
        - (siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l := by
    rw [congrFun (back "dfe" 105 6 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 6 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      pre 6 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 111 - 6 - 1 = 104 from by omega]; exact c104)]
    simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true]
    rw [show k - 6 - 1 = k - 7 from by omega,
      congrFun (back "peC" 104 7 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 7 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      congrFun (back "p0" 100 11 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 7 (by have := Geo.sBound (p := p) (S := S); omega)) l]
  have vcaD : (siter p k (initSt w inPtr outPtr gm smemB)).regs "caD" l
      = (siter p k (initSt w inPtr outPtr gm smemB)).regs "cand0" l
        + (siter p k (initSt w inPtr outPtr gm smemB)).regs "dfe" l := by
    rw [congrFun (back "caD" 109 2 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 2 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      pre 2 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 111 - 2 - 1 = 108 from by omega]; exact c108)]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
    rw [show k - 2 - 1 = k - 3 from by omega,
      ← congrFun (back "caC" 106 5 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 3 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      congrFun (back "caC" 106 5 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 5 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      pre 5 _ (by have := Geo.sBound (p := p) (S := S); omega) (by rw [show 111 - 5 - 1 = 105 from by omega]; exact c105)]
    simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true]
    rw [show k - 5 - 1 = k - 6 from by omega,
      congrFun (back "cand0" 100 11 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 6 (by have := Geo.sBound (p := p) (S := S); omega)) l,
      congrFun (back "dfe" 105 6 (by decide) (by have := Geo.sBound (p := p) (S := S); omega) (by have := Geo.sBound (p := p) (S := S); omega) 6 (by have := Geo.sBound (p := p) (S := S); omega)) l]
  -- and now the arithmetic
  have hl31 : l.val < 32 := l.isLt
  have hmlb := hml l
  have hp0b := hp0 l
  have hidxn : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "idx" l).toNat
      = ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" l).toNat + l.val := by
    rw [vidx, hlane l, UInt64.toNat_add,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt l.val (by have := Geo.sBound (p := p) (S := S); omega), Nat.mod_eq_of_lt (by have := Geo.sBound (p := p) (S := S); omega)]
  have hpen : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "pe" l).toNat
      = ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l).toNat
        + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" l).toNat + l.val := by
    have hlt64 : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l).toNat
        + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "idx" l).toNat < 2 ^ 64 := by
      rw [hidxn]; have := Geo.sBound (p := p) (S := S); omega
    rw [vpe, UInt64.toNat_add, Nat.mod_eq_of_lt hlt64, hidxn]
    omega
  have hec1n : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ec1" l).toNat = (S - 6) := by
    rw [hec1 l, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (S - 6) (by have := Geo.sBound (p := p) (S := S); omega)]
  have hpeCn : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "peC" l).toNat
      = min (((siter p k (initSt w inPtr outPtr gm smemB)).regs "pe" l).toNat) (S - 6) := by
    rw [vpeC]
    simp only [SOp.run]
    by_cases hle : (siter p k (initSt w inPtr outPtr gm smemB)).regs "pe" l
        ≤ (siter p k (initSt w inPtr outPtr gm smemB)).regs "ec1" l
    · rw [if_pos hle, Nat.min_eq_left (by rw [← hec1n]; exact UInt64.le_iff_toNat_le.mp hle)]
    · have hgt : (S - 6) ≤ ((siter p k (initSt w inPtr outPtr gm smemB)).regs "pe" l).toNat := by
        have hx : ¬ (((siter p k (initSt w inPtr outPtr gm smemB)).regs "pe" l).toNat
            ≤ ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ec1" l).toNat) :=
          fun hc => hle (UInt64.le_iff_toNat_le.mpr hc)
        rw [hec1n] at hx; omega
      rw [if_neg hle, hec1n, Nat.min_eq_right hgt]
  -- `p0 ≤ peC`, so the subtraction does not underflow
  have hp0peC : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l).toNat
      ≤ ((siter p k (initSt w inPtr outPtr gm smemB)).regs "peC" l).toNat := by
    rw [hpeCn, hpen]; omega
  have hdfen : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "dfe" l).toNat
      = ((siter p k (initSt w inPtr outPtr gm smemB)).regs "peC" l).toNat
        - ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l).toNat := by
    rw [vdfe, UInt64.toNat_sub_of_le _ _ hp0peC]
  have hcand0b := hcand0 l
  have hcaDn : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "caD" l).toNat
      = ((siter p k (initSt w inPtr outPtr gm smemB)).regs "cand0" l).toNat
        + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "dfe" l).toNat := by
    rw [vcaD, UInt64.toNat_add, Nat.mod_eq_of_lt (by rw [hdfen, hpeCn]; have := Geo.sBound (p := p) (S := S); omega)]
  rw [hcaDn, hdfen, hpeCn]
  omega


-- ── The literal run fits inside the block ───────────────────────────────────


theorem matchS_entry_lt : ∀ q, q < 274 → q ∉ matchS →
    ∀ q' ∈ succsOf p q, q' ∈ matchS → q' = 94 :=
  ivEntry_at p 94 107 94 (Shape.size (p := p)) (by
    have hs := Shape.entryShape (p := p)
    simp only [entryShapeB, Bool.and_eq_true] at hs
    exact hs.1.2)

theorem matchS_entry : ∀ q, q ∉ matchS →
    ∀ q', q' ∈ succsOf p q → q' ∈ matchS → q' = 94 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact matchS_entry_lt q h hq q' hq' hin
  · rw [show succsOf p q = [q] from by
      simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

def MatchInv (S : Nat) (st : SState) : Prop :=
  ∀ l : Lane, (st.regs "p0" l).toNat ≤ (S - 13)
    ∧ (st.regs "searchPos" l).toNat ≤ (st.regs "p0" l).toNat



/-- **The match position, carried across the whole token emit.**  `extend_entry`
    establishes it where the select hands over; neither register is written again
    until `200 litAnchor := p0 + ml`. -/
theorem match_inv (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ matchS →
      MatchInv S (siter p k (initSt w inPtr outPtr gm smemB)) := by
  have h := Shape.matchShape (p := p)
  simp only [matchShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨np0, nsp⟩ := h
  refine inv_in_region matchS 94 (MatchInv S) matchS_entry (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ matchS; decide) ?_
  · intro jj hst _ hI x
    have frame : ∀ r : String,
        (p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?.map
          (fun i => destOf i != some r)) = some true →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs r
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs r := by
      intro r hr
      rw [siter_succ]
      exact sstep_regs_frame p _ r (fun i hi => by rw [hi] at hr; simpa using hr)
    rw [congrFun (frame "p0" (np0 _ hst)) x, congrFun (frame "searchPos" (nsp _ hst)) x]
    exact hI x
  · intro j hj x
    obtain ⟨e1, -, e3⟩ := extend_entry (S := S) w inPtr outPtr gm smemB j hj x
    exact ⟨by have := Geo.sBound (p := p) (S := S); omega, e3⟩


theorem litS_entry_lt : ∀ q, q < 274 → q ∉ litS →
    ∀ q' ∈ succsOf p q, q' ∈ litS → q' = 124 :=
  ivEntry_at p 124 76 124 (Shape.size (p := p)) (by
    have hs := Shape.entryShape (p := p)
    simp only [entryShapeB, Bool.and_eq_true] at hs
    exact hs.2)

theorem litS_entry : ∀ q, q ∉ litS →
    ∀ q', q' ∈ succsOf p q → q' ∈ litS → q' = 124 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact litS_entry_lt q h hq q' hq' hin
  · rw [show succsOf p q = [q] from by
      simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq



include geo in
/-- **The literal run ends inside the block.**

    `litLen = p0 - litAnchor` and the anchor never passes the match position, so
    the run `[litAnchor, litAnchor + litLen)` ends exactly at `p0`, which the
    search guard keeps below the limit.  This is the input-side budget, the
    mirror of the output-side one `TokInv` carries. -/
theorem lit_inv (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ litS →
      ∀ l : Lane, ((siter p k (initSt w inPtr outPtr gm smemB)).regs "litAnchor" l).toNat
        + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "litLen" l).toNat ≤ (S - 13) := by
  have h := Shape.litShape (p := p)
  simp only [litShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨nla, nll⟩, i123⟩, ft124⟩ := h
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  refine inv_in_region litS 124
    (fun st => ∀ l : Lane, (st.regs "litAnchor" l).toNat + (st.regs "litLen" l).toNat ≤ (S - 13))
    litS_entry (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ litS; decide) ?_
  · intro jj hst _ hI x
    have frame : ∀ r : String,
        (p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?.map
          (fun i => destOf i != some r)) = some true →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs r
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs r := by
      intro r hr
      rw [siter_succ]
      exact sstep_regs_frame p _ r (fun i hi => by rw [hi] at hr; simpa using hr)
    rw [congrFun (frame "litAnchor" (nla _ hst)) x, congrFun (frame "litLen" (nll _ hst)) x]
    exact hI x
  · -- entry: the `sub` at 123, with `litAnchor ≤ searchPos ≤ p0 ≤ (S - 13)`
    intro j hj x
    have hpre := pre_state p _ hinit 124 0 (SInstr.bin .sub "litLen" "p0" (.reg "litAnchor"))
      (fun t ht => by rw [show t = 0 from by omega]; exact ft124)
      (by rw [show 124 - 0 - 1 = 123 from by omega]; exact i123) (by have := Geo.sBound (p := p) (S := S); omega) j hj
    have h123 : (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc = 123 := by
      simpa using hpre.2.1
    have hstep : siter p j (initSt w inPtr outPtr gm smemB)
        = sstepInstr p (SInstr.bin .sub "litLen" "p0" (.reg "litAnchor"))
          (siter p (j - 1) (initSt w inPtr outPtr gm smemB)) := by simpa using hpre.1
    have hmi := match_inv (S := S) w inPtr outPtr gm smemB (j - 1) (by
      rw [h123]; simp only [matchS, List.mem_map, List.mem_range]; exact ⟨29, by omega, by omega⟩) x
    have hli := (loop_inv (S := S) w inPtr outPtr gm smemB (j - 1) (by
      rw [h123]; simp only [loopS, List.mem_map, List.mem_range]; exact ⟨85, by omega, by omega⟩)).1
      (by rw [h123]; omega) x
    have hle : ((siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs "litAnchor" x).toNat
        ≤ ((siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs "p0" x).toNat := by omega
    rw [hstep]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, if_pos rfl, if_true,
      if_neg (by decide : ¬ ("litAnchor" = "litLen"))]
    rw [UInt64.toNat_sub_of_le _ _ hle]
    omega

end Lz4Sites
