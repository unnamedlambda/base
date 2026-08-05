import Lz4ExtLoop

set_option maxRecDepth 8192

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

section Generic
variable {p : Array SInstr} [Shape p] [Loads p] {S : Nat} [geo : Geo p S]

-- ── The cooperative copy's source address ───────────────────────────────────



/-- **`cpCont` says exactly what the copy loop's guard means.**  Same shape as
    `loopC_iff`: the head at 156 is a label with two predecessors — the setup's
    `setp` at 155 and the back edge at 168 — so this is a second place the proof
    pays for a `succsOf` scan. -/
theorem cpCont_iff (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc = 156
      ∨ (siter p k (initSt w inPtr outPtr gm smemB)).pc = 157
      ∨ (siter p k (initSt w inPtr outPtr gm smemB)).pc = 168 →
      ∀ l : Lane, ((siter p k (initSt w inPtr outPtr gm smemB)).regs "cpCont" l = 1
        ↔ (siter p k (initSt w inPtr outPtr gm smemB)).regs "cpI" l
            < (siter p k (initSt w inPtr outPtr gm smemB)).regs "litLen" l) := by
  have h := Shape.cpShape (p := p)
  simp only [cpShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨p156, s155⟩, s167⟩, s156⟩, s157⟩, s168⟩, cx11⟩, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩,
    -⟩ := h
  have hsetp : ∀ (q : Nat), p[q]? = some (SInstr.setp .lt "cpCont" "cpI" (.reg "litLen")) →
      ∀ st : SState, st.pc = q → ∀ l : Lane,
        ((sstep p st).regs "cpCont" l = 1
          ↔ (sstep p st).regs "cpI" l < (sstep p st).regs "litLen" l) := by
    intro q hq st hpc l
    subst hpc
    rw [sstep, hq]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SCmp.run, reduceIte,
      decide_eq_true_eq]
    by_cases hc : st.regs "cpI" l < st.regs "litLen" l
    · rw [if_pos hc]; exact ⟨fun _ => hc, fun _ => rfl⟩
    · rw [if_neg hc]; exact ⟨fun h => absurd h (by decide), fun h => absurd h hc⟩
  have hnone : ∀ (q : Nat) (i : SInstr), p[q]? = some i → destOf i = none →
      ∀ (st : SState), st.pc = q → ∀ r : String, (sstep p st).regs r = st.regs r := by
    intro q i hq hd st hpc r
    exact sstep_regs_frame p st r (fun j hj => by rw [hpc, hq] at hj; cases hj; rw [hd]; simp)
  have hpred156 : ∀ st : SState, (sstep p st).pc = 156 → st.pc = 155 ∨ st.pc = 168 := by
    intro st hh
    rcases Nat.lt_or_ge st.pc 274 with hq | hq
    · have := p156 st.pc (by simp [List.mem_range, hq]) _ (sstep_pc_mem_succs p st)
      simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
        decide_eq_true_eq] at this
      rcases this with e | e
      · exact absurd hh e
      · exact e
    · exfalso
      have he : (sstep p st).pc = st.pc := by
        rw [sstep, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]
      omega
  intro k
  induction k with
  | zero =>
      intro hh
      rw [show (siter p 0 (initSt w inPtr outPtr gm smemB)).pc = 0 from rfl] at hh
      omega
  | succ m ih =>
      intro hh l
      rw [siter_succ] at hh ⊢
      rcases hh with hh | hh | hh
      · rcases hpred156 _ hh with e | e
        · exact hsetp 155 s155 _ e l
        · have f1 := hnone 168 (SInstr.bra "Ch10") s168 rfl _ e "cpCont"
          have f2 := hnone 168 (SInstr.bra "Ch10") s168 rfl _ e "cpI"
          have f3 := hnone 168 (SInstr.bra "Ch10") s168 rfl _ e "litLen"
          rw [congrFun f1 l, congrFun f2 l, congrFun f3 l]
          exact ih (Or.inr (Or.inr e)) l
      · have e : (siter p m (initSt w inPtr outPtr gm smemB)).pc = 156 := by
          have := pc_pred p _ 157 _ s157 (by intro n; simp) (by simp) hh
          omega
        have f1 := hnone 156 (SInstr.lbl "Ch10") s156 rfl _ e "cpCont"
        have f2 := hnone 156 (SInstr.lbl "Ch10") s156 rfl _ e "cpI"
        have f3 := hnone 156 (SInstr.lbl "Ch10") s156 rfl _ e "litLen"
        rw [congrFun f1 l, congrFun f2 l, congrFun f3 l]
        exact ih (Or.inl e) l
      · have e : (siter p m (initSt w inPtr outPtr gm smemB)).pc = 167 := by
          have := pc_pred p _ 168 _ s168 (by intro n; simp) (by simp) hh
          omega
        exact hsetp 167 s167 _ e l

/-- The copy loop with its setup: pcs 154–168, entered only from `153`. -/
def copyR : List Nat := (List.range 15).map (· + 154)

theorem copyR_entry : ∀ q, q ∉ copyR →
    ∀ q', q' ∈ succsOf p q → q' ∈ copyR → q' = 154 := by
  have h := Shape.cpShape (p := p)
  simp only [cpShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with hlt | hge
  · have := h.2 q (by simp [List.mem_range, hlt]) q' hq'
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at this
    rcases this with (e | e) | e
    · exact absurd hin e
    · exact e
    · exact absurd e hq
  · rw [show succsOf p q = [q] from by
      simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

include geo in
/-- **The copy reads from `inBase + litAnchor`.**  `153` computes it and nothing
    in the loop touches it again. -/
theorem cpSrc_eq (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ copyR →
      ∀ l : Lane, (siter p k (initSt w inPtr outPtr gm smemB)).regs "cpSrc" l
        = (siter p k (initSt w inPtr outPtr gm smemB)).regs "inBase" l
          + (siter p k (initSt w inPtr outPtr gm smemB)).regs "litAnchor" l := by
  have h := Shape.cpShape (p := p)
  simp only [cpShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩, nsrc⟩, nib⟩, -⟩, -⟩, i153⟩, -⟩, ft154⟩, -⟩ := h
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  refine inv_in_region copyR 154
    (fun st => ∀ l : Lane, st.regs "cpSrc" l = st.regs "inBase" l + st.regs "litAnchor" l)
    copyR_entry (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ copyR; decide) ?_
  · intro jj hst _ hI x
    have frame : ∀ r : String,
        (p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?.map
          (fun i => destOf i != some r)) = some true →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs r
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs r := by
      intro r hr
      rw [siter_succ]
      exact sstep_regs_frame p _ r (fun i hi => by rw [hi] at hr; simpa using hr)
    have hb : 154 ≤ (siter p jj (initSt w inPtr outPtr gm smemB)).pc
        ∧ (siter p jj (initSt w inPtr outPtr gm smemB)).pc ≤ 168 := by
      simp only [copyR, List.mem_map, List.mem_range] at hst
      obtain ⟨j, hj, hje⟩ := hst; omega
    have hla : (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs "litAnchor"
        = (siter p jj (initSt w inPtr outPtr gm smemB)).regs "litAnchor" := by
      refine frame "litAnchor" ?_
      have := (lit_inv (p := p) (S := S) w inPtr outPtr gm smemB)   -- litAnchor is untouched in [124,199]
      have hx : ∀ q ∈ litS, (p[q]?.map (fun i => destOf i != some "litAnchor")) = some true := by
        have hl := Shape.litShape (p := p)
        simp only [litShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at hl
        exact hl.1.1.1
      exact hx _ (by
        simp only [litS, List.mem_map, List.mem_range]
        exact ⟨(siter p jj (initSt w inPtr outPtr gm smemB)).pc - 124, by omega, by omega⟩)
    rw [congrFun (frame "cpSrc" (nsrc _ hst)) x, congrFun (frame "inBase" (nib _ hst)) x,
      congrFun hla x]
    exact hI x
  · intro j hj x
    have hpre := pre_state p _ hinit 154 0
      (SInstr.bin .add "cpSrc" "inBase" (.reg "litAnchor"))
      (fun t ht => by rw [show t = 0 from by omega]; exact ft154)
      (by rw [show 154 - 0 - 1 = 153 from by omega]; exact i153) (by omega) j hj
    have hstep : siter p j (initSt w inPtr outPtr gm smemB)
        = sstepInstr p (SInstr.bin .add "cpSrc" "inBase" (.reg "litAnchor"))
          (siter p (j - 1) (initSt w inPtr outPtr gm smemB)) := by simpa using hpre.1
    rw [hstep]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, if_pos rfl, if_true,
      if_neg (by decide : ¬ ("inBase" = "cpSrc")),
      if_neg (by decide : ¬ ("litAnchor" = "cpSrc"))]

include geo in
/-- **The copy source is inside the block, plus the warp's own overhang.**

    The copy loads unpredicated and stores predicated, so lanes past the end of
    the literal run still compute an address — up to 31 bytes beyond it.  That
    overhang is why the host places the output `copySlack` bytes above the input,
    and this is the bound that makes the placement enough: the offset never
    exceeds `(S + 18) < inStride + copySlack`. -/
theorem cpSo_off (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (k : Nat) (h164 : (siter p k (initSt w inPtr outPtr gm smemB)).pc = 164) (l : Lane) :
    ∃ off : Nat, off ≤ (S + 18) ∧
      (siter p k (initSt w inPtr outPtr gm smemB)).regs "cpSo" l
        = (siter p k (initSt w inPtr outPtr gm smemB)).regs "inBase" l + UInt64.ofNat off := by
  have h := Shape.cpShape (p := p)
  simp only [cpShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, -⟩, s157⟩, -⟩, cx11⟩, -⟩, -⟩, ncpI⟩, nll⟩, -⟩, ft⟩, -⟩, -⟩ := h
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  have hf : ∀ t, t < 8 → (p[164 - t]?.map fallthroughOnlyB) = some true := by
    intro t ht
    exact ft (164 - t) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨7 - t, by omega, by omega⟩)
  -- the guard at 157 fell through, so the copy index is still inside the run
  have hpre := pre_state p _ hinit 164 6 (SInstr.braifnot "cpCont" "Cx11")
    (fun t ht => hf t (by omega))
    (by rw [show 164 - 6 - 1 = 157 from by omega]; exact s157) (by omega) k h164
  rw [show k - 6 - 1 = k - 7 from by omega] at hpre
  have h157 : (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).pc = 157 := by
    have := hpre.2.1; omega
  have h158 : (siter p (k - 6) (initSt w inPtr outPtr gm smemB)).pc = 158 :=
    (pc_back p _ hinit 164 6 (by omega) (fun t ht => hf t (by omega)) k h164).2
  have hcc0 : (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).regs "cpCont" 0 = 1 := by
    by_cases hc : (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).regs "cpCont" 0 == 1
    · exact beq_iff_eq.mp hc
    · exfalso
      rw [hpre.1] at h158
      simp only [sstepInstr, SState.setPc] at h158
      rw [if_neg hc, cx11] at h158
      omega
  have hlt7 : (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).regs "cpI" l
      < (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).regs "litLen" l := by
    refine (cpCont_iff w inPtr outPtr gm smemB (k - 7) (Or.inr (Or.inl h157)) l).mp ?_
    rw [uni_at w inPtr outPtr gm smemB (k - 7) "cpCont" (by simp [uniR]) l 0]
    exact hcc0
  -- carry it to the load
  have bcpI := regs_back p _ hinit "cpI" 164 7 (by omega) (fun t ht => hf t (by omega))
    (fun t ht => ncpI (164 - t - 1) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨8 - t, by omega, by omega⟩)) k h164
  have bll := regs_back p _ hinit "litLen" 164 7 (by omega) (fun t ht => hf t (by omega))
    (fun t ht => nll (164 - t - 1) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨9 - t, by omega, by omega⟩)) k h164
  have hlt : (siter p k (initSt w inPtr outPtr gm smemB)).regs "cpI" l
      < (siter p k (initSt w inPtr outPtr gm smemB)).regs "litLen" l := by
    rw [congrFun bcpI l, congrFun bll l]; exact hlt7
  -- the input budget and the address shape
  have hfit := lit_inv (S := S) w inPtr outPtr gm smemB k (by
    rw [h164]; simp only [litS, List.mem_map, List.mem_range]; exact ⟨40, by omega, by omega⟩) l
  have hsrc := cpSrc_eq (S := S) w inPtr outPtr gm smemB k (by
    rw [h164]; simp only [copyR, List.mem_map, List.mem_range]; exact ⟨10, by omega, by omega⟩) l
  have hso := (Loads.loadAt (p := p) w inPtr outPtr gm smemB k l).2.2.2.2.1 h164
  have hlane := lane_val w inPtr outPtr gm smemB k (by rw [h164]; omega) l
  have hltn := UInt64.lt_iff_toNat_lt.mp hlt
  have hl31 : l.val < 32 := l.isLt
  have hoff : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "litAnchor" l).toNat
      + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "cpI" l).toNat + l.val ≤ (S + 18) := by
    have h1 : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "cpI" l).toNat
        < ((siter p k (initSt w inPtr outPtr gm smemB)).regs "litLen" l).toNat :=
      UInt64.lt_iff_toNat_lt.mp hlt
    omega
  refine ⟨((siter p k (initSt w inPtr outPtr gm smemB)).regs "litAnchor" l).toNat
    + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "cpI" l).toNat + l.val, hoff, ?_⟩
  rw [hso, hsrc, hlane, UInt64.ofNat_add, UInt64.ofNat_add,
    UInt64.ofNat_toNat, UInt64.ofNat_toNat]
  rw [UInt64.add_assoc, UInt64.add_assoc, UInt64.add_assoc]

-- ── The budget survives to the tail ─────────────────────────────────────────




theorem tailS_entry : ∀ q, q ∉ tailS →
    ∀ q', q' ∈ succsOf p q → q' ∈ tailS → q' = 121 := by
  have h := Shape.tailShape (p := p)
  simp only [tailShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with hlt | hge
  · have := h.1.1.1.1.1 q (by simp [List.mem_range, hlt]) q' hq'
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at this
    rcases this with (e | e) | e
    · exact absurd hin e
    · exact e
    · exact absurd e hq
  · rw [show succsOf p q = [q] from by
      simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

/-- **The match budget survives the extend loop's exit**, all the way to the
    instruction that moves the anchor. -/
theorem ml_tail (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ tailS →
      ∀ l : Lane, ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l).toNat
        + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "ml" l).toNat ≤ (S - 5) := by
  have h := Shape.tailShape (p := p)
  simp only [tailShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨-, nml⟩, np0⟩, n99ml⟩, n99p0⟩, p121⟩ := h
  refine inv_in_region tailS 121
    (fun st => ∀ l : Lane, (st.regs "p0" l).toNat + (st.regs "ml" l).toNat ≤ (S - 5))
    tailS_entry (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ tailS; decide) ?_
  · intro jj hst _ hI x
    have frame : ∀ r : String,
        (p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?.map
          (fun i => destOf i != some r)) = some true →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs r
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs r := by
      intro r hr
      rw [siter_succ]
      exact sstep_regs_frame p _ r (fun i hi => by rw [hi] at hr; simpa using hr)
    rw [congrFun (frame "p0" (np0 _ hst)) x, congrFun (frame "ml" (nml _ hst)) x]
    exact hI x
  · -- entry: the extend loop's guard at 99, where `ml_inv` still applies
    intro j hj x
    have hj1 : 1 ≤ j := by
      rcases Nat.eq_zero_or_pos j with h0 | h0
      · rw [h0, show siter p 0 (initSt w inPtr outPtr gm smemB)
          = initSt w inPtr outPtr gm smemB from rfl,
          show (initSt w inPtr outPtr gm smemB).pc = 0 from rfl] at hj
        omega
      · exact h0
    have h99 : (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc = 99 := by
      have hmem : (121 : Nat) ∈ succsOf p (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc := by
        rw [← hj, show j = (j - 1) + 1 from by omega, siter_succ]
        exact sstep_pc_mem_succs p _
      rcases Nat.lt_or_ge (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc 274 with hq | hq
      · have := p121 _ (by simp [List.mem_range, hq]) 121 hmem
        simpa using this
      · exfalso
        rw [show succsOf p (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc
          = [(siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc] from by
            simp only [succsOf,
              Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hmem
        rw [List.mem_singleton] at hmem
        omega
    have hstep : ∀ r : String, (p[99]?.map (fun i => destOf i != some r)) = some true →
        (siter p j (initSt w inPtr outPtr gm smemB)).regs r
          = (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs r := by
      intro r hr
      rw [show j = (j - 1) + 1 from by omega, siter_succ]
      refine sstep_regs_frame p _ r (fun i hi => ?_)
      rw [h99] at hi
      rw [hi] at hr
      simpa using hr
    rw [congrFun (hstep "p0" n99p0) x, congrFun (hstep "ml" n99ml) x]
    exact ml_inv (S := S) w inPtr outPtr gm smemB (j - 1) (by
      rw [h99]; simp only [extS, List.mem_map, List.mem_range]; exact ⟨5, by omega, by omega⟩)
      (by rw [h99]; omega) (by rw [h99]; omega) x




theorem laS_entry : ∀ q, q ∉ laS →
    ∀ q', q' ∈ succsOf p q → q' ∈ laS → q' = 38 := by
  have h := Shape.laShape (p := p)
  simp only [laShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with hlt | hge
  · have := h.1.1 q (by simp [List.mem_range, hlt]) q' hq'
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at this
    rcases this with (e | e) | e
    · exact absurd hin e
    · exact e
    · exact absurd e hq
  · rw [show succsOf p q = [q] from by
      simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

include geo in
/-- **The literal anchor stays inside the block.**  It starts at zero and is only
    ever moved to `p0 + ml`, which `ml_tail` bounds.  This is what makes the tail
    copy's `fLen := inStride - litAnchor` a real length rather than an underflow. -/
theorem litAnchor_le (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ laS →
      ∀ l : Lane, ((siter p k (initSt w inPtr outPtr gm smemB)).regs "litAnchor" l).toNat
        ≤ (S - 5) := by
  have h := Shape.laShape (p := p)
  simp only [laShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨-, nla⟩, i200⟩ := h
  have hg := Shape.loopShape (p := p)
  simp only [loopShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at hg
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩, p38⟩, g35⟩, -⟩, ft35⟩, n36la⟩, n37la⟩, -⟩ := hg
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  refine inv_in_region laS 38
    (fun st => ∀ l : Lane, (st.regs "litAnchor" l).toNat ≤ (S - 5))
    laS_entry (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ laS; decide) ?_
  · intro jj hst _ hI x
    by_cases h200 : (siter p jj (initSt w inPtr outPtr gm smemB)).pc = 200
    · -- the anchor moves to `p0 + ml`
      have hb := ml_tail (S := S) w inPtr outPtr gm smemB jj (by
        rw [h200]; simp only [tailS, List.mem_map, List.mem_range]; exact ⟨79, by omega, by omega⟩) x
      rw [siter_succ, sstep, show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
        = some (SInstr.bin .add "litAnchor" "p0" (.reg "ml")) from by rw [h200]; exact i200]
      simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run,
        if_pos rfl, if_true]
      rw [UInt64.toNat_add, Nat.mod_eq_of_lt (by
        have := Geo.sBound (p := p) (S := S); omega)]
      omega
    · have hfr : (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs "litAnchor"
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs "litAnchor" := by
        rw [siter_succ]
        refine sstep_regs_frame p _ "litAnchor" (fun i hi => ?_)
        have hx := nla _ hst
        simp only [h200, ne_eq, not_false_eq_true, decide_true, Bool.not_true,
          Bool.false_or] at hx
        rw [hi] at hx
        simpa using hx
      rw [congrFun hfr x]
      exact hI x
  · -- entry: `35 litAnchor := 0`, carried across 36 and 37
    intro j hj x
    have hj1 : 1 ≤ j := by
      rcases Nat.eq_zero_or_pos j with h0 | h0
      · rw [h0, show siter p 0 (initSt w inPtr outPtr gm smemB)
          = initSt w inPtr outPtr gm smemB from rfl, hinit] at hj
        omega
      · exact h0
    have h37 : (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc = 37 := by
      have hmem : (38 : Nat) ∈ succsOf p (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc := by
        rw [← hj, show j = (j - 1) + 1 from by omega, siter_succ]
        exact sstep_pc_mem_succs p _
      rcases Nat.lt_or_ge (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc 274 with hq | hq
      · have := p38 _ (by simp [List.mem_range, hq]) 38 hmem
        simpa using this
      · exfalso
        rw [show succsOf p (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc
          = [(siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc] from by
            simp only [succsOf,
              Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hmem
        rw [List.mem_singleton] at hmem
        omega
    have hf : ∀ t, t < 3 → (p[37 - t]?.map fallthroughOnlyB) = some true := by
      intro t ht
      exact ft35 (37 - t) (by
        simp only [List.mem_map, List.mem_range]; exact ⟨2 - t, by omega, by omega⟩)
    have hla0 : (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs "litAnchor" x
        = UInt64.ofNat 0 := by
      refine mov_imm_carried p _ hinit "litAnchor" 0 37 1 (fun t ht => hf t (by omega)) ?_
        (by rw [show 37 - 1 - 1 = 35 from by omega]; exact g35) (by omega) (j - 1) h37 x
      intro t ht
      rw [show t = 0 from by omega, show 37 - 0 - 1 = 36 from by omega]
      exact n36la
    have hfr : (siter p j (initSt w inPtr outPtr gm smemB)).regs "litAnchor"
        = (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs "litAnchor" := by
      rw [show j = (j - 1) + 1 from by omega, siter_succ]
      refine sstep_regs_frame p _ "litAnchor" (fun i hi => ?_)
      rw [h37] at hi
      rw [hi] at n37la
      simpa using n37la
    rw [congrFun hfr x, hla0]
    exact Nat.zero_le _

-- ── The tail copy ───────────────────────────────────────────────────────────




include geo in
theorem ftS_entry : ∀ q, q ∉ ftS →
    ∀ q', q' ∈ succsOf p q → q' ∈ ftS → q' = 211 := by
  have h := Geo.ftShape (p := p) (S := S)
  simp only [ftShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with hlt | hge
  · have := h.1.1.1.1.1.1 q (by simp [List.mem_range, hlt]) q' hq'
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at this
    rcases this with (e | e) | e
    · exact absurd hin e
    · exact e
    · exact absurd e hq
  · rw [show succsOf p q = [q] from by
      simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

include geo in
/-- **The final literal run reaches exactly the end of the block.**  `fLen` is
    `inStride - litAnchor`, and `litAnchor_le` is what stops that subtraction
    from underflowing. -/
theorem fLen_fit (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ ftS →
      ∀ l : Lane, ((siter p k (initSt w inPtr outPtr gm smemB)).regs "litAnchor" l).toNat
        + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "fLen" l).toNat = S := by
  have h := Geo.ftShape (p := p) (S := S)
  simp only [ftShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨-, nla⟩, nfl⟩, i209⟩, i210⟩, n210la⟩, ft210⟩ := h
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  refine inv_in_region ftS 211
    (fun st => ∀ l : Lane, (st.regs "litAnchor" l).toNat + (st.regs "fLen" l).toNat = S)
    (ftS_entry (S := S)) (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ ftS; decide) ?_
  · intro jj hst _ hI x
    have frame : ∀ r : String,
        (p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?.map
          (fun i => destOf i != some r)) = some true →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs r
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs r := by
      intro r hr
      rw [siter_succ]
      exact sstep_regs_frame p _ r (fun i hi => by rw [hi] at hr; simpa using hr)
    rw [congrFun (frame "litAnchor" (nla _ hst)) x, congrFun (frame "fLen" (nfl _ hst)) x]
    exact hI x
  · intro j hj x
    have hf : ∀ t, t < 2 → (p[211 - t]?.map fallthroughOnlyB) = some true := by
      intro t ht
      exact ft210 (211 - t) (by
        simp only [List.mem_map, List.mem_range]; exact ⟨1 - t, by omega, by omega⟩)
    have hpre := pre_state p _ hinit 211 0
      (SInstr.bin .sub "fLen" "fLen" (.reg "litAnchor"))
      (fun t ht => by rw [show t = 0 from by omega]; exact hf 0 (by omega))
      (by rw [show 211 - 0 - 1 = 210 from by omega]; exact i210) (by omega) j hj
    have h210 : (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).pc = 210 := by
      simpa using hpre.2.1
    have hstep : siter p j (initSt w inPtr outPtr gm smemB)
        = sstepInstr p (SInstr.bin .sub "fLen" "fLen" (.reg "litAnchor"))
          (siter p (j - 1) (initSt w inPtr outPtr gm smemB)) := by simpa using hpre.1
    -- `fLen = S` from the `mov` at 209
    have hfl : (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs "fLen" x
        = UInt64.ofNat S :=
      mov_imm_carried p _ hinit "fLen" S 210 0
        (fun t ht => by rw [show t = 0 from by omega]; exact hf 1 (by omega))
        (fun t ht => absurd ht (by omega))
        (by rw [show 210 - 0 - 1 = 209 from by omega]; exact i209) (by omega) (j - 1) h210 x
    have hla := litAnchor_le (S := S) w inPtr outPtr gm smemB (j - 1) (by
      rw [h210]; simp only [laS, List.mem_map, List.mem_range]; exact ⟨172, by omega, by omega⟩) x
    have hle : ((siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs "litAnchor" x).toNat
        ≤ ((siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs "fLen" x).toNat := by
      rw [hfl, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt S (by have := Geo.sBound (p := p) (S := S); omega)]; omega
    rw [hstep]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, if_pos rfl, if_true,
      if_neg (by decide : ¬ ("litAnchor" = "fLen"))]
    rw [UInt64.toNat_sub_of_le _ _ hle, hfl,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt S (by have := Geo.sBound (p := p) (S := S); omega)]
    omega



/-- The tail copy loop with its setup: pcs 240–254. -/
def copyR2 : List Nat := (List.range 15).map (· + 240)

theorem copyR2_entry : ∀ q, q ∉ copyR2 →
    ∀ q', q' ∈ succsOf p q → q' ∈ copyR2 → q' = 240 := by
  have h := Shape.cp2Shape (p := p)
  simp only [cp2ShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with hlt | hge
  · have := h.2 q (by simp [List.mem_range, hlt]) q' hq'
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at this
    rcases this with (e | e) | e
    · exact absurd hin e
    · exact e
    · exact absurd e hq
  · rw [show succsOf p q = [q] from by
      simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

/-- `cpCont_iff` for the tail loop. -/
theorem cpCont2_iff (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc = 242
      ∨ (siter p k (initSt w inPtr outPtr gm smemB)).pc = 243
      ∨ (siter p k (initSt w inPtr outPtr gm smemB)).pc = 254 →
      ∀ l : Lane, ((siter p k (initSt w inPtr outPtr gm smemB)).regs "cpCont" l = 1
        ↔ (siter p k (initSt w inPtr outPtr gm smemB)).regs "cpI" l
            < (siter p k (initSt w inPtr outPtr gm smemB)).regs "fLen" l) := by
  have h := Shape.cp2Shape (p := p)
  simp only [cp2ShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨p242, s241⟩, s253⟩, s242⟩, s243⟩, s254⟩, cx21⟩, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩,
    -⟩, -⟩, -⟩ := h
  have hsetp : ∀ (q : Nat), p[q]? = some (SInstr.setp .lt "cpCont" "cpI" (.reg "fLen")) →
      ∀ st : SState, st.pc = q → ∀ l : Lane,
        ((sstep p st).regs "cpCont" l = 1
          ↔ (sstep p st).regs "cpI" l < (sstep p st).regs "fLen" l) := by
    intro q hq st hpc l
    subst hpc
    rw [sstep, hq]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SCmp.run, reduceIte,
      decide_eq_true_eq]
    by_cases hc : st.regs "cpI" l < st.regs "fLen" l
    · rw [if_pos hc]; exact ⟨fun _ => hc, fun _ => rfl⟩
    · rw [if_neg hc]; exact ⟨fun h => absurd h (by decide), fun h => absurd h hc⟩
  have hnone : ∀ (q : Nat) (i : SInstr), p[q]? = some i → destOf i = none →
      ∀ (st : SState), st.pc = q → ∀ r : String, (sstep p st).regs r = st.regs r := by
    intro q i hq hd st hpc r
    exact sstep_regs_frame p st r (fun j hj => by rw [hpc, hq] at hj; cases hj; rw [hd]; simp)
  have hpred242 : ∀ st : SState, (sstep p st).pc = 242 → st.pc = 241 ∨ st.pc = 254 := by
    intro st hh
    rcases Nat.lt_or_ge st.pc 274 with hq | hq
    · have := p242 st.pc (by simp [List.mem_range, hq]) _ (sstep_pc_mem_succs p st)
      simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
        decide_eq_true_eq] at this
      rcases this with e | e
      · exact absurd hh e
      · exact e
    · exfalso
      have he : (sstep p st).pc = st.pc := by
        rw [sstep, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]
      omega
  intro k
  induction k with
  | zero =>
      intro hh
      rw [show (siter p 0 (initSt w inPtr outPtr gm smemB)).pc = 0 from rfl] at hh
      omega
  | succ m ih =>
      intro hh l
      rw [siter_succ] at hh ⊢
      rcases hh with hh | hh | hh
      · rcases hpred242 _ hh with e | e
        · exact hsetp 241 s241 _ e l
        · have f1 := hnone 254 (SInstr.bra "Ch20") s254 rfl _ e "cpCont"
          have f2 := hnone 254 (SInstr.bra "Ch20") s254 rfl _ e "cpI"
          have f3 := hnone 254 (SInstr.bra "Ch20") s254 rfl _ e "fLen"
          rw [congrFun f1 l, congrFun f2 l, congrFun f3 l]
          exact ih (Or.inr (Or.inr e)) l
      · have e : (siter p m (initSt w inPtr outPtr gm smemB)).pc = 242 := by
          have := pc_pred p _ 243 _ s243 (by intro n; simp) (by simp) hh
          omega
        have f1 := hnone 242 (SInstr.lbl "Ch20") s242 rfl _ e "cpCont"
        have f2 := hnone 242 (SInstr.lbl "Ch20") s242 rfl _ e "cpI"
        have f3 := hnone 242 (SInstr.lbl "Ch20") s242 rfl _ e "fLen"
        rw [congrFun f1 l, congrFun f2 l, congrFun f3 l]
        exact ih (Or.inl e) l
      · have e : (siter p m (initSt w inPtr outPtr gm smemB)).pc = 253 := by
          have := pc_pred p _ 254 _ s254 (by intro n; simp) (by simp) hh
          omega
        exact hsetp 253 s253 _ e l

/-- `cpSrc_eq` for the tail loop. -/
theorem cpSrcF_eq (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ copyR2 →
      ∀ l : Lane, (siter p k (initSt w inPtr outPtr gm smemB)).regs "cpSrcF" l
        = (siter p k (initSt w inPtr outPtr gm smemB)).regs "inBase" l
          + (siter p k (initSt w inPtr outPtr gm smemB)).regs "litAnchor" l := by
  have h := Shape.cp2Shape (p := p)
  simp only [cp2ShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩, nsrc⟩, nib⟩, nla⟩, -⟩, -⟩, i239⟩, -⟩,
    ft240⟩, -⟩ := h
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  refine inv_in_region copyR2 240
    (fun st => ∀ l : Lane, st.regs "cpSrcF" l = st.regs "inBase" l + st.regs "litAnchor" l)
    copyR2_entry (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ copyR2; decide) ?_
  · intro jj hst _ hI x
    have frame : ∀ r : String,
        (p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?.map
          (fun i => destOf i != some r)) = some true →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs r
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs r := by
      intro r hr
      rw [siter_succ]
      exact sstep_regs_frame p _ r (fun i hi => by rw [hi] at hr; simpa using hr)
    rw [congrFun (frame "cpSrcF" (nsrc _ hst)) x, congrFun (frame "inBase" (nib _ hst)) x,
      congrFun (frame "litAnchor" (nla _ hst)) x]
    exact hI x
  · intro j hj x
    have hpre := pre_state p _ hinit 240 0
      (SInstr.bin .add "cpSrcF" "inBase" (.reg "litAnchor"))
      (fun t ht => by rw [show t = 0 from by omega]; exact ft240)
      (by rw [show 240 - 0 - 1 = 239 from by omega]; exact i239) (by omega) j hj
    have hstep : siter p j (initSt w inPtr outPtr gm smemB)
        = sstepInstr p (SInstr.bin .add "cpSrcF" "inBase" (.reg "litAnchor"))
          (siter p (j - 1) (initSt w inPtr outPtr gm smemB)) := by simpa using hpre.1
    rw [hstep]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, if_pos rfl, if_true,
      if_neg (by decide : ¬ ("inBase" = "cpSrcF")),
      if_neg (by decide : ¬ ("litAnchor" = "cpSrcF"))]

include geo in
/-- **The tail copy's source, bounded.**  Same overhang as the first copy loop,
    but against the *final* literal run, which reaches exactly the end of the
    block — so the bound is `S + 31 = (S + 31)`, precisely
    `inStride + copySlack - 1`.  This is the tightest of the twelve load sites,
    and the one that fixes `copySlack` at 32. -/
theorem cpSo2_off (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (k : Nat) (h250 : (siter p k (initSt w inPtr outPtr gm smemB)).pc = 250) (l : Lane) :
    ∃ off : Nat, off ≤ (S + 31) ∧
      (siter p k (initSt w inPtr outPtr gm smemB)).regs "cpSo" l
        = (siter p k (initSt w inPtr outPtr gm smemB)).regs "inBase" l + UInt64.ofNat off := by
  have h := Shape.cp2Shape (p := p)
  simp only [cp2ShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, -⟩, s243⟩, -⟩, cx21⟩, -⟩, -⟩, -⟩, ncpI⟩, nfl⟩, -⟩, ft⟩,
    -⟩, -⟩ := h
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  have hf : ∀ t, t < 8 → (p[250 - t]?.map fallthroughOnlyB) = some true := by
    intro t ht
    exact ft (250 - t) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨7 - t, by omega, by omega⟩)
  have hpre := pre_state p _ hinit 250 6 (SInstr.braifnot "cpCont" "Cx21")
    (fun t ht => hf t (by omega))
    (by rw [show 250 - 6 - 1 = 243 from by omega]; exact s243) (by omega) k h250
  rw [show k - 6 - 1 = k - 7 from by omega] at hpre
  have h243 : (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).pc = 243 := by
    have := hpre.2.1; omega
  have h244 : (siter p (k - 6) (initSt w inPtr outPtr gm smemB)).pc = 244 :=
    (pc_back p _ hinit 250 6 (by omega) (fun t ht => hf t (by omega)) k h250).2
  have hcc0 : (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).regs "cpCont" 0 = 1 := by
    by_cases hc : (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).regs "cpCont" 0 == 1
    · exact beq_iff_eq.mp hc
    · exfalso
      rw [hpre.1] at h244
      simp only [sstepInstr, SState.setPc] at h244
      rw [if_neg hc, cx21] at h244
      omega
  have hlt7 : (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).regs "cpI" l
      < (siter p (k - 7) (initSt w inPtr outPtr gm smemB)).regs "fLen" l := by
    refine (cpCont2_iff w inPtr outPtr gm smemB (k - 7) (Or.inr (Or.inl h243)) l).mp ?_
    rw [uni_at w inPtr outPtr gm smemB (k - 7) "cpCont" (by simp [uniR]) l 0]
    exact hcc0
  have bcpI := regs_back p _ hinit "cpI" 250 7 (by omega) (fun t ht => hf t (by omega))
    (fun t ht => ncpI (250 - t - 1) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨8 - t, by omega, by omega⟩)) k h250
  have bfl := regs_back p _ hinit "fLen" 250 7 (by omega) (fun t ht => hf t (by omega))
    (fun t ht => nfl (250 - t - 1) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨9 - t, by omega, by omega⟩)) k h250
  have hlt : (siter p k (initSt w inPtr outPtr gm smemB)).regs "cpI" l
      < (siter p k (initSt w inPtr outPtr gm smemB)).regs "fLen" l := by
    rw [congrFun bcpI l, congrFun bfl l]; exact hlt7
  have hfit := fLen_fit (S := S) w inPtr outPtr gm smemB k (by
    rw [h250]; simp only [ftS, List.mem_map, List.mem_range]; exact ⟨39, by omega, by omega⟩) l
  have hsrc := cpSrcF_eq w inPtr outPtr gm smemB k (by
    rw [h250]; simp only [copyR2, List.mem_map, List.mem_range]; exact ⟨10, by omega, by omega⟩) l
  have hso := (Loads.loadAt (p := p) w inPtr outPtr gm smemB k l).2.2.2.2.2 h250
  have hlane := lane_val w inPtr outPtr gm smemB k (by rw [h250]; omega) l
  have hl31 : l.val < 32 := l.isLt
  have hoff : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "litAnchor" l).toNat
      + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "cpI" l).toNat + l.val ≤ (S + 31) := by
    have h1 : ((siter p k (initSt w inPtr outPtr gm smemB)).regs "cpI" l).toNat
        < ((siter p k (initSt w inPtr outPtr gm smemB)).regs "fLen" l).toNat :=
      UInt64.lt_iff_toNat_lt.mp hlt
    omega
  refine ⟨((siter p k (initSt w inPtr outPtr gm smemB)).regs "litAnchor" l).toNat
    + ((siter p k (initSt w inPtr outPtr gm smemB)).regs "cpI" l).toNat + l.val, hoff, ?_⟩
  rw [hso, hsrc, hlane, UInt64.ofNat_add, UInt64.ofNat_add,
    UInt64.ofNat_toNat, UInt64.ofNat_toNat]
  rw [UInt64.add_assoc, UInt64.add_assoc, UInt64.add_assoc]

-- ── The warp's input base ───────────────────────────────────────────────────



/-- **The warp's input base, at the instruction that computes it.**

    The whole prologue chain in one place: `in_ptr` and `%ctaid.x` are never
    written, so `inP` and `gwarp` are what the launch put there, and `u64_gwarp`
    turns `(ctaid*ntid + tid) >> 5` back into the CTA index. -/
theorem inBase_at_19 (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hw : w * 32 + 32 < 2 ^ 64) (k : Nat)
    (h19 : (siter p k (initSt w inPtr outPtr gm smemB)).pc = 19) (l : Lane) :
    (siter p k (initSt w inPtr outPtr gm smemB)).regs "inBase" l
      = UInt64.ofNat inPtr + UInt64.ofNat w * UInt64.ofNat S := by
  have h := Geo.ibShape (p := p) (S := S)
  simp only [ibShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨ft, i0⟩, i2⟩, i7⟩, i5⟩, i6⟩, i3⟩, i4⟩, i13⟩, i14⟩, i15⟩, i18⟩, nin⟩, nct⟩, ntd⟩, nnt⟩, ninP⟩, ngwarp⟩, ngwD⟩, ninOff⟩, ngtid⟩, nctab⟩, nntid⟩, ntid2⟩ := h
  have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
  have hf : ∀ t, t < 19 → (p[19 - t]?.map fallthroughOnlyB) = some true := by
    intro t ht
    exact ft (19 - t) (by
      simp only [List.mem_map, List.mem_range]; exact ⟨19 - t, by omega, by omega⟩)
  have back : ∀ (r : String) (n : Nat), n ≤ 19 →
      (∀ t, t < n → (p[19 - t - 1]?.map (fun i => destOf i != some r)) = some true) →
      (siter p k (initSt w inPtr outPtr gm smemB)).regs r
        = (siter p (k - n) (initSt w inPtr outPtr gm smemB)).regs r := by
    intro r n hn hnd
    exact regs_back p _ hinit r 19 n (by omega) (fun t ht => hf t (by omega)) hnd k h19
  have pre : ∀ (n : Nat) (i : SInstr), n ≤ 18 → p[19 - n - 1]? = some i →
      siter p (k - n) (initSt w inPtr outPtr gm smemB)
        = sstepInstr p i (siter p (k - n - 1) (initSt w inPtr outPtr gm smemB)) := by
    intro n i hn hi
    exact (pre_state p _ hinit 19 n i (fun t ht => hf t (by omega)) hi (by omega) k h19).1
  -- the launch registers are never written
  have hconst : ∀ r : String, noDest p r = true → ∀ j : Nat,
      (siter p j (initSt w inPtr outPtr gm smemB)).regs r
        = initRegs w inPtr outPtr r := fun r hr j => siter_regs_const p r hr _ j
  have mk : ∀ (r : String) (n : Nat) (lo len : Nat), n ≤ 19 → lo + len = 19 → lo ≤ 19 - n →
      (∀ q ∈ (List.range len).map (· + lo),
        (p[q]?.map (fun i => destOf i != some r)) = some true) →
      (siter p k (initSt w inPtr outPtr gm smemB)).regs r
        = (siter p (k - n) (initSt w inPtr outPtr gm smemB)).regs r := by
    intro r n lo len hn hsum hle hall
    refine back r n hn (fun t ht => hall (19 - t - 1) ?_)
    simp only [List.mem_map, List.mem_range]
    exact ⟨19 - t - 1 - lo, by omega, by omega⟩
  -- unwind: inBase ← inP, inOff ← gwD, gwarp ← gtid ← ctab, ntid, tid
  have vinB := pre 0 _ (by omega) (by rw [show 19 - 0 - 1 = 18 from by omega]; exact i18)
  have vinOff := pre 3 _ (by omega) (by rw [show 19 - 3 - 1 = 15 from by omega]; exact i15)
  have vinOff0 := pre 4 _ (by omega) (by rw [show 19 - 4 - 1 = 14 from by omega]; exact i14)
  have vgwD := pre 5 _ (by omega) (by rw [show 19 - 5 - 1 = 13 from by omega]; exact i13)
  have vgwarp := pre 11 _ (by omega) (by rw [show 19 - 11 - 1 = 7 from by omega]; exact i7)
  have vgtid := pre 12 _ (by omega) (by rw [show 19 - 12 - 1 = 6 from by omega]; exact i6)
  have vgtid0 := pre 13 _ (by omega) (by rw [show 19 - 13 - 1 = 5 from by omega]; exact i5)
  have vctab := pre 15 _ (by omega) (by rw [show 19 - 15 - 1 = 3 from by omega]; exact i3)
  have vntid := pre 14 _ (by omega) (by rw [show 19 - 14 - 1 = 4 from by omega]; exact i4)
  have vinP := pre 18 _ (by omega) (by rw [show 19 - 18 - 1 = 0 from by omega]; exact i0)
  rw [show k - 0 = k from by omega] at vinB
  rw [show k - 3 - 1 = k - 4 from by omega] at vinOff
  rw [show k - 4 - 1 = k - 5 from by omega] at vinOff0
  rw [show k - 5 - 1 = k - 6 from by omega] at vgwD
  rw [show k - 11 - 1 = k - 12 from by omega] at vgwarp
  rw [show k - 12 - 1 = k - 13 from by omega] at vgtid
  rw [show k - 13 - 1 = k - 14 from by omega] at vgtid0
  rw [show k - 15 - 1 = k - 16 from by omega] at vctab
  rw [show k - 14 - 1 = k - 15 from by omega] at vntid
  rw [show k - 18 - 1 = k - 19 from by omega] at vinP
  have vtid := pre 16 _ (by omega) (by rw [show 19 - 16 - 1 = 2 from by omega]; exact i2)
  rw [show k - 16 - 1 = k - 17 from by omega] at vtid
  -- put every register at the state its instruction ran in
  have e_inP : (siter p (k - 1) (initSt w inPtr outPtr gm smemB)).regs "inP" l
      = UInt64.ofNat inPtr := by
    rw [← congrFun (mk "inP" 1 1 18 (by omega) (by omega) (by omega) ninP) l,
      congrFun (mk "inP" 18 1 18 (by omega) (by omega) (by omega) ninP) l, vinP]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
    rw [congrFun (hconst "in_ptr" nin (k - 19)) l]
    rfl
  have e_ctab : (siter p (k - 14) (initSt w inPtr outPtr gm smemB)).regs "ctab" l
      = UInt64.ofNat w := by
    rw [← congrFun (mk "ctab" 14 4 15 (by omega) (by omega) (by omega) nctab) l,
      congrFun (mk "ctab" 15 4 15 (by omega) (by omega) (by omega) nctab) l, vctab]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
    rw [congrFun (hconst "%ctaid.x" nct (k - 16)) l]
    rfl
  have e_ntid : (siter p (k - 14) (initSt w inPtr outPtr gm smemB)).regs "ntid" l
      = (32 : UInt64) := by
    rw [vntid]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
    rw [congrFun (hconst "%ntid.x" nnt (k - 15)) l]
    rfl
  have e_tid : (siter p (k - 14) (initSt w inPtr outPtr gm smemB)).regs "tid" l
      = UInt64.ofNat l.val := by
    rw [← congrFun (mk "tid" 14 3 16 (by omega) (by omega) (by omega) ntid2) l,
      congrFun (mk "tid" 16 3 16 (by omega) (by omega) (by omega) ntid2) l, vtid]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
    rw [congrFun (hconst "%tid.x" ntd (k - 17)) l]
    rfl
  have e_gwarp : (siter p (k - 6) (initSt w inPtr outPtr gm smemB)).regs "gwarp" l
      = UInt64.ofNat w := by
    rw [← congrFun (mk "gwarp" 6 8 11 (by omega) (by omega) (by omega) ngwarp) l,
      congrFun (mk "gwarp" 11 8 11 (by omega) (by omega) (by omega) ngwarp) l, vgwarp]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, if_pos rfl, if_true]
    rw [vgtid]
    simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true]
    rw [vgtid0]
    simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true,
      if_neg (by decide : ¬ ("ctab" = "gtid")), if_neg (by decide : ¬ ("ntid" = "gtid")),
      if_neg (by decide : ¬ ("tid" = "gtid"))]
    rw [e_ctab, e_ntid, e_tid]
    exact u64_gwarp w hw l
  have e_gwD : (siter p (k - 4) (initSt w inPtr outPtr gm smemB)).regs "gwD" l
      = UInt64.ofNat w := by
    rw [← congrFun (mk "gwD" 4 14 5 (by omega) (by omega) (by omega) ngwD) l,
      congrFun (mk "gwD" 5 14 5 (by omega) (by omega) (by omega) ngwD) l, vgwD]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
    exact e_gwarp
  have e_inOff : (siter p (k - 1) (initSt w inPtr outPtr gm smemB)).regs "inOff" l
      = UInt64.ofNat w * UInt64.ofNat S := by
    rw [← congrFun (mk "inOff" 1 16 3 (by omega) (by omega) (by omega) ninOff) l,
      congrFun (mk "inOff" 3 16 3 (by omega) (by omega) (by omega) ninOff) l, vinOff]
    simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true]
    rw [e_gwD, vinOff0]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
  rw [vinB]
  simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true,
    if_neg (by decide : ¬ ("inP" = "inBase")), if_neg (by decide : ¬ ("inOff" = "inBase"))]
  rw [e_inP, e_inOff]

/-- pcs 19–271: everything below the base computation, stopping short of `OOB`
    at 272 so that the guard at pc 11 is not a second entry. -/

theorem ibS_entry : ∀ q, q ∉ ibS →
    ∀ q', q' ∈ succsOf p q → q' ∈ ibS → q' = 19 := by
  have h := Shape.ibReg (p := p)
  simp only [ibRegOK, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with hlt | hge
  · have := h.1 q (by simp [List.mem_range, hlt]) q' hq'
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
      decide_eq_true_eq] at this
    rcases this with (e | e) | e
    · exact absurd hin e
    · exact e
    · exact absurd e hq
  · rw [show succsOf p q = [q] from by
      simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

/-- **The warp's input base, at every step that can touch memory.** -/
theorem inBase_eq (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hw : w * 32 + 32 < 2 ^ 64) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ ibS →
      ∀ l : Lane, (siter p k (initSt w inPtr outPtr gm smemB)).regs "inBase" l
        = UInt64.ofNat inPtr + UInt64.ofNat w * UInt64.ofNat S := by
  have h := Shape.ibReg (p := p)
  simp only [ibRegOK, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  refine inv_in_region ibS 19
    (fun st => ∀ l : Lane, st.regs "inBase" l
      = UInt64.ofNat inPtr + UInt64.ofNat w * UInt64.ofNat S)
    ibS_entry (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ ibS; decide) ?_
  · intro jj hst _ hI x
    have hfr : (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs "inBase"
        = (siter p jj (initSt w inPtr outPtr gm smemB)).regs "inBase" := by
      rw [siter_succ]
      refine sstep_regs_frame p _ "inBase" (fun i hi => ?_)
      have hx := h.2 _ hst
      rw [hi] at hx
      simpa using hx
    rw [congrFun hfr x]
    exact hI x
  · intro j hj x
    exact inBase_at_19 w inPtr outPtr gm smemB hw j hj x

end Generic

-- ── `RegConfined`'s load half ───────────────────────────────────────────────

/-- **Every global read is below the output allocation.**

    Twelve sites, five address registers, one shape: each is `inBase + X` with
    `X` bounded by the block geometry.  The worst case is the tail copy's
    overhang at `w = numBlk - 1`:
    `6399 * 32768 + (32768 + 31) = 209715231 < 209715232 = outPtr - inPtr`.
    One byte of slack, which is what `copySlack` buys. -/
theorem loads_confined (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hplace : outPtr = inPtr + 209715232) (htop : inPtr + 209715232 < 2 ^ 64)
    (w : Fin (WP.mk 15).numBlk) (k : Nat) (r : String) (off : Nat)
    (hmem : ((siter K k (initSt w.val inPtr outPtr gm smemB)).pc, r, off) ∈ loadSites K)
    (l : Lane) :
    ((siter K k (initSt w.val inPtr outPtr gm smemB)).regs r l).toNat + off < outPtr := by
  have hwlt : w.val < 6400 := Nat.lt_of_lt_of_le w.isLt (by decide)
  have hw32 : w.val * 32 + 32 < 2 ^ 64 := by omega
  -- the pc, register and offset are one of twelve concrete triples
  rw [Loads.loadSitesEq (p := K)] at hmem
  -- the input base, as a number
  have hibn : ∀ j : Nat, (siter K j (initSt w.val inPtr outPtr gm smemB)).pc ∈ ibS →
      ((siter K j (initSt w.val inPtr outPtr gm smemB)).regs "inBase" l).toNat
        = inPtr + w.val * 32768 := by
    intro j hj
    rw [inBase_eq (S := 32768) w.val inPtr outPtr gm smemB hw32 j hj l, UInt64.toNat_add, UInt64.toNat_mul,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt inPtr (by omega),
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt w.val (by omega),
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt 32768 (by omega),
      Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  -- the uniform step: `regs r l = inBase + X` with `X` bounded
  have main : ∀ (q : Nat), (siter K k (initSt w.val inPtr outPtr gm smemB)).pc = q →
      q ∈ ibS → ∀ X : UInt64,
      (siter K k (initSt w.val inPtr outPtr gm smemB)).regs r l
        = (siter K k (initSt w.val inPtr outPtr gm smemB)).regs "inBase" l + X →
      X.toNat + off ≤ (32768 + 31) →
      ((siter K k (initSt w.val inPtr outPtr gm smemB)).regs r l).toNat + off < outPtr := by
    intro q hq hqm X hX hb
    have hb2 := hibn k (by rw [hq]; exact hqm)
    rw [hX, UInt64.toNat_add, hb2, Nat.mod_eq_of_lt (by omega), hplace]
    omega
  have mem19 : ∀ q : Nat, 19 ≤ q → q ≤ 271 → q ∈ ibS := by
    intro q h1 h2
    simp only [ibS, List.mem_map, List.mem_range]
    exact ⟨q - 19, by omega, by omega⟩
  obtain ⟨lat1, lat2, lat3, lat4, lat5, lat6⟩ := Loads.loadAt (p := K) w.val inPtr outPtr gm smemB k l
  simp only [List.mem_cons, List.not_mem_nil, or_false, Prod.mk.injEq] at hmem
  rcases hmem with ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩
    | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩
    | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ <;> subst hr <;> subst ho
  -- `rpA`, offsets 0–3
  · refine main 47 hp (mem19 47 (by omega) (by omega)) _ (lat1 47 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rp_clamped w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 48 hp (mem19 48 (by omega) (by omega)) _ (lat1 48 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rp_clamped w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 49 hp (mem19 49 (by omega) (by omega)) _ (lat1 49 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rp_clamped w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 50 hp (mem19 50 (by omega) (by omega)) _ (lat1 50 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rp_clamped w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  -- `rcA`, offsets 0–3
  · refine main 66 hp (mem19 66 (by omega) (by omega)) _ (lat2 66 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rc_clamped w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 67 hp (mem19 67 (by omega) (by omega)) _ (lat2 67 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rc_clamped w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 68 hp (mem19 68 (by omega) (by omega)) _ (lat2 68 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rc_clamped w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 69 hp (mem19 69 (by omega) (by omega)) _ (lat2 69 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rc_clamped w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  -- the extend's two byte reads
  · refine main 110 hp (mem19 110 (by omega) (by omega)) _ (lat3 hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right (peD_le (S := 32768) w.val inPtr outPtr gm smemB k hp l) _)
      (by decide)
  · refine main 111 hp (mem19 111 (by omega) (by omega)) _ (lat4 hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right (caD_le (S := 32768) w.val inPtr outPtr gm smemB k hp l) _)
      (by decide)
  -- the two cooperative copies
  · obtain ⟨o1, ho1, he1⟩ := cpSo_off (S := 32768) w.val inPtr outPtr gm smemB k hp l
    refine main 164 hp (mem19 164 (by omega) (by omega)) (UInt64.ofNat o1) ?_ ?_
    · exact he1
    · rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt o1 (by omega)]; omega
  · obtain ⟨o2, ho2, he2⟩ := cpSo2_off (S := 32768) w.val inPtr outPtr gm smemB k hp l
    refine main 250 hp (mem19 250 (by omega) (by omega)) (UInt64.ofNat o2) ?_ ?_
    · exact he2
    · rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt o2 (by omega)]; omega

-- ── `RegConfined`, assembled ─────────────────────────────────────────────────

/-- **Every address the kernel uses is where it should be.**

    The load half is `loads_confined`: all twelve `ldgo` sites, under the
    placement the host actually produces.  The store half is
    `stores_except_copy` for the fourteen unpredicated sites and
    `stores_cpDo165`/`stores_cpDo251` for the two `stgp`s of the cooperative
    copies, whose predicate `ActiveAt` hands over.

    This was the audit's last kernel-level obligation. -/
theorem regConfined_shipped (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 15).numBlk → inPtr + w * (WP.mk 15).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 15).numBlk →
      inPtr + w * (WP.mk 15).inStride + (WP.mk 15).inStride
        ≤ outPtr + w * (WP.mk 15).outStride) :
    RegConfined 15 inPtr outPtr gm smemB where
  loads := by
    intro w k r off hmem l
    have hplace : outPtr = inPtr + 209715232 := by
      rw [hderive, show (WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack
        = 209715232 from by decide]
    have h32 := htop32 0 (by decide)
    exact loads_confined inPtr outPtr gm smemB hplace (by omega) w k r off hmem l
  stores := by
    intro w k r hsite l hact
    by_cases hc : r = "cpDo"
    · subst hc
      have hsites : ∀ x ∈ storeSites K, x.2 = "cpDo" → x.1 = 165 ∨ x.1 = 251 :=
        Loads.storeCpDo (p := K)
      have hcp : ∀ q, (siter K k (initSt w.val inPtr outPtr gm smemB)).pc = q →
          K[q]? = some (SInstr.stgp "cpP" "cpDo" "cpB") →
          (siter K k (initSt w.val inPtr outPtr gm smemB)).regs "cpP" l = 1 := by
        intro q hq hK
        rw [ActiveAt, show K[(siter K k (initSt w.val inPtr outPtr gm smemB)).pc]?
          = some (SInstr.stgp "cpP" "cpDo" "cpB") from by rw [hq]; exact hK] at hact
        exact hact
      rcases hsites _ hsite rfl with e | e
      · exact stores_cpDo165 inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj w k l e
          (hcp 165 e (by decide))
      · exact stores_cpDo251 inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj w k l e
          (hcp 251 e (by decide))
    · exact stores_except_copy inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj w k l r
        hsite hc

/-- …and hence `KernelConfined`, with no assumption left about what any address
    register holds. -/
theorem kernelConfined_shipped (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 15).numBlk * (WP.mk 15).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 15).numBlk → inPtr + w * (WP.mk 15).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 15).numBlk →
      outPtr + w * (WP.mk 15).outStride + 9 * (WP.mk 15).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 15).numBlk →
      inPtr + w * (WP.mk 15).inStride + (WP.mk 15).inStride
        ≤ outPtr + w * (WP.mk 15).outStride) :
    Lz4Interleave.KernelConfined 15 inPtr outPtr gm smemB :=
  kernelConfined_of_regConfined32 inPtr outPtr gm smemB
    (regConfined_shipped inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj)



end Lz4Sites
