import Lz4ExtShape

set_option maxRecDepth 8192

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

variable {p : Array SInstr} [Shape p] {S : Nat} [Geo p S]
-- ── Two facts the prologue establishes for the whole run ─────────────────────



/-- **`lane` holds the lane index from pc 9 on, at every step of every trace.**

    `8 and lane, tid, 31` computes it and nothing writes it again; `tid` comes
    from `%tid.x`, which the program never writes at all.  Carried as one
    invariant because the middle link — `tid = %tid.x` — is only true from pc 3
    on, so neither half is a whole-trace fact by itself. -/
theorem lane_val (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, 9 ≤ (siter p k (initSt w inPtr outPtr gm smemB)).pc →
      ∀ l : Lane, (siter p k (initSt w inPtr outPtr gm smemB)).regs "lane" l
        = UInt64.ofNat l.val := by
  have hs := Shape.preShape (p := p)
  simp only [preShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at hs
  obtain ⟨⟨⟨⟨⟨⟨h9, h3⟩, htid⟩, hlane⟩, hnd⟩, hftid⟩, hflane⟩ := hs
  -- successors: nothing below 8 (resp. 2) can jump into the region
  have hin : ∀ (b e : Nat), (∀ q ∈ List.range 274, ∀ q' ∈ succsOf p q,
        (!decide (b ≤ q') || decide (b ≤ q ∨ q = e)) = true) →
      ∀ (st : SState), b ≤ (sstep p st).pc → b ≤ st.pc ∨ st.pc = e := by
    intro b e hb st hlt
    rcases Nat.lt_or_ge st.pc 274 with hq | hq
    · have := hb st.pc (by simp [List.mem_range, hq]) _ (sstep_pc_mem_succs p st)
      simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
        decide_eq_true_eq] at this
      rcases this with h | h
      · exact absurd hlt h
      · exact h
    · left
      have : (sstep p st).pc = st.pc := by
        rw [sstep, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]
      omega
  have hframe : ∀ (r : String) (b : Nat) (n : Nat), b + n = 274 →
      (∀ q ∈ (List.range n).map (· + b),
        p[q]?.map (fun i => destOf i != some r) = some true) →
      ∀ st : SState, b ≤ st.pc → (sstep p st).regs r = st.regs r := by
    intro r b n hbn hn st h1
    refine sstep_regs_frame p st r (fun i hi => ?_)
    rcases Nat.lt_or_ge st.pc (b + n) with h2 | h2
    · have hx := hn st.pc (by
        simp only [List.mem_map, List.mem_range]; exact ⟨st.pc - b, by omega, by omega⟩)
      rw [hi] at hx
      simpa using hx
    · rw [Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)] at hi
      exact absurd hi (by simp)
  -- the bundled invariant
  have key : ∀ k : Nat,
      (∀ l : Lane, (siter p k (initSt w inPtr outPtr gm smemB)).regs "%tid.x" l
        = UInt64.ofNat l.val)
      ∧ (3 ≤ (siter p k (initSt w inPtr outPtr gm smemB)).pc →
          ∀ l : Lane, (siter p k (initSt w inPtr outPtr gm smemB)).regs "tid" l
            = UInt64.ofNat l.val)
      ∧ (9 ≤ (siter p k (initSt w inPtr outPtr gm smemB)).pc →
          ∀ l : Lane, (siter p k (initSt w inPtr outPtr gm smemB)).regs "lane" l
            = UInt64.ofNat l.val) := by
    intro k
    induction k with
    | zero =>
        refine ⟨fun l => rfl, fun h => ?_, fun h => ?_⟩ <;>
          (rw [show (siter p 0 (initSt w inPtr outPtr gm smemB)).pc = 0 from rfl] at h; omega)
    | succ m ih =>
        obtain ⟨i0, i1, i2⟩ := ih
        refine ⟨?_, ?_, ?_⟩
        · intro l
          rw [siter_succ, congrFun (sstep_regs_frame p _ "%tid.x"
            (fun i hi => noDest_spec hnd _ i hi)) l]
          exact i0 l
        · intro hlo l
          rw [siter_succ] at hlo ⊢
          rcases hin 3 2 h3 _ hlo with hge | he
          · rw [congrFun (hframe "tid" 3 271 (by omega) hftid _ hge) l]
            exact i1 hge l
          · rw [sstep, show p[(siter p m (initSt w inPtr outPtr gm smemB)).pc]?
              = some (SInstr.mov "tid" (.reg "%tid.x")) from by rw [he]; exact htid]
            simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_true]
            exact i0 l
        · intro hlo l
          rw [siter_succ] at hlo ⊢
          rcases hin 9 8 h9 _ hlo with hge | he
          · rw [congrFun (hframe "lane" 9 265 (by omega) hflane _ hge) l]
            exact i2 hge l
          · rw [sstep, show p[(siter p m (initSt w inPtr outPtr gm smemB)).pc]?
              = some (SInstr.bin .band "lane" "tid" (.imm 31)) from by rw [he]; exact hlane]
            simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run,
              if_pos rfl, if_true]
            rw [i1 (by omega) l]
            revert l; decide
  intro k hk l
  exact (key k).2.2 hk l

-- ── The loop guard, recovered for every lane ────────────────────────────────



/-- `x < ofNat c` is `x.toNat < c`, for a `c` that fits. -/
theorem lt_ofNat_iff (x : UInt64) (c : Nat) (hc : c < 2 ^ 64) :
    (x < UInt64.ofNat c) ↔ (x.toNat < c) := by
  rw [UInt64.lt_iff_toNat_lt, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt c hc]

/-- **`loopC` says exactly what the guard means, at the three points it is
    live.**

    `39` and `206` are the same `setp`; `40` is the loop label, which the
    fallthrough from `39` and the back-edge `207 bra Lh0` both reach.  Because
    `40` is a label, `pc_pred` does not apply there, so this is the one place the
    proof pays for a `succsOf` scan of the whole array. -/
theorem loopC_iff (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc = 40
      ∨ (siter p k (initSt w inPtr outPtr gm smemB)).pc = 41
      ∨ (siter p k (initSt w inPtr outPtr gm smemB)).pc = 207 →
      ∀ l : Lane, ((siter p k (initSt w inPtr outPtr gm smemB)).regs "loopC" l = 1
        ↔ ((siter p k (initSt w inPtr outPtr gm smemB)).regs "searchPos" l).toNat < (S - 12)) := by
  have hs := Geo.guardShape (p := p) (S := S)
  simp only [guardShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at hs
  obtain ⟨⟨⟨⟨⟨⟨h40in, h39⟩, h206⟩, h40⟩, h41⟩, h207⟩, _⟩ := hs
  -- the `setp` at 39/206 establishes the equivalence one step later
  have hsetp : ∀ (q : Nat), p[q]? = some (SInstr.setp .lt "loopC" "searchPos" (.imm (S - 12))) →
      ∀ st : SState, st.pc = q → ∀ l : Lane,
        ((sstep p st).regs "loopC" l = 1
          ↔ ((sstep p st).regs "searchPos" l).toNat < (S - 12)) := by
    intro q hq st hpc l
    subst hpc
    rw [sstep, hq]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SCmp.run, reduceIte,
      decide_eq_true_eq]
    by_cases hc : st.regs "searchPos" l < UInt64.ofNat (S - 12)
    · rw [if_pos hc]
      exact ⟨fun _ => (lt_ofNat_iff _ (S - 12) (by have := Geo.sBound (p := p) (S := S); omega)).mp hc, fun _ => rfl⟩
    · rw [if_neg hc]
      exact ⟨fun h => absurd h (by decide),
        fun h => absurd ((lt_ofNat_iff _ (S - 12) (by have := Geo.sBound (p := p) (S := S); omega)).mpr h) hc⟩
  -- a state stepping to the loop label came from the fallthrough or the back-edge
  have hpred40 : ∀ st : SState, (sstep p st).pc = 40 → st.pc = 39 ∨ st.pc = 207 := by
    intro st h
    rcases Nat.lt_or_ge st.pc 274 with hq | hq
    · have := h40in st.pc (by simp [List.mem_range, hq]) _ (sstep_pc_mem_succs p st)
      simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
        decide_eq_true_eq] at this
      rcases this with e | e
      · exact absurd h e
      · exact e
    · exfalso
      have he : (sstep p st).pc = st.pc := by
        rw [sstep, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]
      omega
  -- a `lbl` or `bra` writes nothing
  have hnone : ∀ (q : Nat) (i : SInstr), p[q]? = some i → destOf i = none →
      ∀ (st : SState), st.pc = q → ∀ r : String, (sstep p st).regs r = st.regs r := by
    intro q i hq hd st hpc r
    exact sstep_regs_frame p st r (fun j hj => by
      rw [hpc, hq] at hj; cases hj; rw [hd]; simp)
  intro k
  induction k with
  | zero =>
      intro h
      rw [show (siter p 0 (initSt w inPtr outPtr gm smemB)).pc = 0 from rfl] at h
      omega
  | succ m ih =>
      intro h l
      rw [siter_succ] at h ⊢
      rcases h with h | h | h
      · rcases hpred40 _ h with e | e
        · exact hsetp 39 h39 _ e l
        · have hfl := hnone 207 (SInstr.bra "Lh0") h207 rfl _ e "loopC"
          have hfs := hnone 207 (SInstr.bra "Lh0") h207 rfl _ e "searchPos"
          rw [congrFun hfl l, congrFun hfs l]
          exact ih (Or.inr (Or.inr e)) l
      · have e : (siter p m (initSt w inPtr outPtr gm smemB)).pc = 40 := by
          have := pc_pred p _ 41 _ h41 (by intro n; simp) (by simp) h
          omega
        have hfl := hnone 40 (SInstr.lbl "Lh0") h40 rfl _ e "loopC"
        have hfs := hnone 40 (SInstr.lbl "Lh0") h40 rfl _ e "searchPos"
        rw [congrFun hfl l, congrFun hfs l]
        exact ih (Or.inl e) l
      · have e : (siter p m (initSt w inPtr outPtr gm smemB)).pc = 206 := by
          have := pc_pred p _ 207 _ h207 (by intro n; simp) (by simp) h
          omega
        exact hsetp 206 h206 _ e l


/-- `Lx1` is the loop exit, at 208 — so falling through the guard at 41 really
    does mean `loopC` was set. -/
theorem lx1_is_208 : sfindLabel p "Lx1" = 208 := Shape.lx1 (p := p)

/-- **The loop guard holds in every lane, everywhere in the window.**

    The branch at 41 reads lane 0 only, so on its own it says nothing about lane
    `l ≠ 0`.  `uni_at` closes that gap: `searchPos` is warp-uniform, so lane 0's
    guard is everybody's. -/
theorem searchPos_lt (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, 42 ≤ (siter p k (initSt w inPtr outPtr gm smemB)).pc →
      (siter p k (initSt w inPtr outPtr gm smemB)).pc ≤ 92 →
      ∀ l : Lane, ((siter p k (initSt w inPtr outPtr gm smemB)).regs "searchPos" l).toNat
        < (S - 12) := by
  have hs := Geo.guardShape (p := p) (S := S)
  simp only [guardShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at hs
  obtain ⟨⟨⟨⟨⟨⟨_, _⟩, _⟩, _⟩, h41⟩, _⟩, hfr⟩ := hs
  intro k
  induction k with
  | zero =>
      intro h _
      rw [show (siter p 0 (initSt w inPtr outPtr gm smemB)).pc = 0 from rfl] at h
      omega
  | succ m ih =>
      intro hlo hhi l
      rw [siter_succ] at hlo hhi ⊢
      -- the window is fallthrough-only, so step `m` stood one below
      have hst : (siter p m (initSt w inPtr outPtr gm smemB)).pc + 1
          = (sstep p (siter p m (initSt w inPtr outPtr gm smemB))).pc := by
        have hq := win_ftG p S (Geo.winShape (p := p) (S := S)) _ hlo hhi
        cases hi : p[(sstep p (siter p m (initSt w inPtr outPtr gm smemB))).pc]? with
        | none => rw [hi] at hq; exact absurd hq (by simp)
        | some i =>
            rw [hi] at hq
            have hf : fallthroughOnlyB i = true := by simpa using hq
            exact pc_pred p _ _ i hi
              (fun n e => by rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
              (by intro e; rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB])) rfl
      rcases Nat.lt_or_ge (siter p m (initSt w inPtr outPtr gm smemB)).pc 42 with hb | ha
      · -- came off the guard at 41: lane 0 passed it, and `searchPos` is uniform
        have e41 : (siter p m (initSt w inPtr outPtr gm smemB)).pc = 41 := by omega
        have hbr : (siter p m (initSt w inPtr outPtr gm smemB)).regs "loopC" 0 = 1 := by
          rw [sstep, show p[(siter p m (initSt w inPtr outPtr gm smemB)).pc]?
            = some (SInstr.braifnot "loopC" "Lx1") from by rw [e41]; exact h41] at hst
          simp only [sstepInstr, SState.setPc] at hst
          by_cases hc : (siter p m (initSt w inPtr outPtr gm smemB)).regs "loopC" 0 == 1
          · exact beq_iff_eq.mp hc
          · rw [if_neg hc, lx1_is_208] at hst; omega
        have h0 := ((loopC_iff (S := S) w inPtr outPtr gm smemB m (Or.inr (Or.inl e41)) 0).mp hbr)
        have hfl : (sstep p (siter p m (initSt w inPtr outPtr gm smemB))).regs "searchPos"
            = (siter p m (initSt w inPtr outPtr gm smemB)).regs "searchPos" :=
          sstep_regs_frame p _ "searchPos" (fun i hi => by
            rw [e41, h41] at hi; cases hi; simp [destOf])
        rw [congrFun hfl l,
          uni_at w inPtr outPtr gm smemB m "searchPos" (by simp [uniR]) l 0]
        exact h0
      · -- still inside the window: nothing here writes `searchPos`
        have hfl : (sstep p (siter p m (initSt w inPtr outPtr gm smemB))).regs "searchPos"
            = (siter p m (initSt w inPtr outPtr gm smemB)).regs "searchPos" := by
          refine sstep_regs_frame p _ "searchPos" (fun i hi => ?_)
          have hx := hfr (siter p m (initSt w inPtr outPtr gm smemB)).pc (by
            simp only [List.mem_map, List.mem_range]
            exact ⟨(siter p m (initSt w inPtr outPtr gm smemB)).pc - 42, by omega, by omega⟩)
          rw [hi] at hx
          simpa using hx
        rw [congrFun hfl l]
        exact ih ha (by omega) l

/-- **An invariant that holds from a region's single entry point onwards.**

    `region_entry` says a state inside the region descends from one standing at
    the entry; this adds the step-by-step induction on top, which is what an
    invariant argument actually wants. -/
theorem inv_in_region (S : List Nat) (e : Nat) (I : SState → Prop)
    (hen : ∀ q, q ∉ S → ∀ q', q' ∈ succsOf p q → q' ∈ S → q' = e)
    (ss : SState)
    (hpres : ∀ j : Nat, (siter p j ss).pc ∈ S → (siter p (j + 1) ss).pc ∈ S →
      I (siter p j ss) → I (siter p (j + 1) ss))
    (h0 : ss.pc ∉ S)
    (hentry : ∀ j, (siter p j ss).pc = e → I (siter p j ss)) :
    ∀ k, (siter p k ss).pc ∈ S → I (siter p k ss) := by
  intro k hk
  obtain ⟨j, hjk, hje, hall⟩ := region_entry p S e hen ss h0 k hk
  have step : ∀ d, j + d ≤ k → I (siter p (j + d) ss) := by
    intro d
    induction d with
    | zero => intro _; exact hentry j hje
    | succ m ih =>
        intro hle
        rw [show j + (m + 1) = (j + m) + 1 from by omega]
        exact hpres (j + m) (hall (j + m) (by omega) (by omega))
          (hall (j + m + 1) (by omega) (by omega)) (ih (by omega))
  have := step (k - j) (by omega)
  rwa [show j + (k - j) = k from by omega] at this


-- ── The literal anchor never passes the cursor ──────────────────────────────


theorem loopS_entry_lt : ∀ q, q < 274 → q ∉ loopS →
    ∀ q' ∈ succsOf p q, q' ∈ loopS → q' = 38 :=
  ivEntry_at p 38 170 38 (Shape.size (p := p)) (by
    have hs := Shape.entryShape (p := p)
    simp only [entryShapeB, Bool.and_eq_true] at hs
    exact hs.1.1.1)

theorem loopS_entry : ∀ q, q ∉ loopS →
    ∀ q', q' ∈ succsOf p q → q' ∈ loopS → q' = 38 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact loopS_entry_lt q h hq q' hq' hin
  · rw [show succsOf p q = [q] from by
      simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [Shape.size (p := p)]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq



/-- **The literal anchor never passes the cursor, and the cursor stays inside the
    block.**  `litAnchor` is only ever set to `p0 + ml` at 200, and 201 moves the
    cursor to exactly that — so the two travel together, and the only point where
    the order is broken is between those two instructions. -/
def LoopInv (S : Nat) (st : SState) : Prop :=
  (st.pc ≠ 201 → ∀ l : Lane,
      (st.regs "litAnchor" l).toNat ≤ (st.regs "searchPos" l).toNat)
  ∧ (GuardLive st.pc → ∀ l : Lane, (st.regs "searchPos" l).toNat < (S - 12))



/-- **`LoopInv` holds at every state of the body.** -/
theorem loop_inv (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    ∀ k : Nat, (siter p k (initSt w inPtr outPtr gm smemB)).pc ∈ loopS →
      LoopInv S (siter p k (initSt w inPtr outPtr gm smemB)) := by
  have h := Shape.loopShape (p := p)
  simp only [loopShapeB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hgl, nla⟩, nsp⟩, g200⟩, g201⟩, g204⟩, g41⟩, _⟩, p38⟩, g35⟩, g36⟩,
    ft35⟩, n36la⟩, n37la⟩, n37sp⟩ := h
  refine inv_in_region loopS 38 (LoopInv S) loopS_entry (initSt w inPtr outPtr gm smemB) ?_
    (by show (0 : Nat) ∉ loopS; decide) ?_
  · intro jj hst hst' hI
    obtain ⟨P, hP⟩ : ∃ q, (siter p jj (initSt w inPtr outPtr gm smemB)).pc = q := ⟨_, rfl⟩
    obtain ⟨N, hN⟩ : ∃ q, (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).pc = q := ⟨_, rfl⟩
    have hb : 38 ≤ P ∧ P ≤ 207 := by
      rw [hP] at hst
      simp only [loopS, List.mem_map, List.mem_range] at hst
      obtain ⟨j, hj, hje⟩ := hst; omega
    have frame : ∀ r : String,
        (p[P]?.map (fun i => destOf i != some r)) = some true →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs r
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs r := by
      intro r hr
      rw [siter_succ]
      refine sstep_regs_frame p _ r (fun i hi => ?_)
      rw [hP] at hi
      rw [hi] at hr
      simpa using hr
    have hlaf : P ≠ 200 → (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs "litAnchor"
        = (siter p jj (initSt w inPtr outPtr gm smemB)).regs "litAnchor" := by
      intro hne
      refine frame "litAnchor" ?_
      have := nla P (by rw [← hP]; exact hst)
      simpa [hne] using this
    have hspf : P ≠ 201 → P ≠ 204 →
        (siter p (jj + 1) (initSt w inPtr outPtr gm smemB)).regs "searchPos"
          = (siter p jj (initSt w inPtr outPtr gm smemB)).regs "searchPos" := by
      intro h1 h2
      refine frame "searchPos" ?_
      have := nsp P (by rw [← hP]; exact hst)
      simpa [h1, h2] using this
    -- the guard is only entered at 42, out of the `braifnot` at 41
    have hguard : GuardLive N → GuardLive P ∨ P = 41 := by
      intro hg
      rcases Nat.lt_or_ge P 274 with hq | hq
      · have hmem : N ∈ succsOf p P := by
          rw [← hP, ← hN, siter_succ]; exact sstep_pc_mem_succs p _
        have := hgl P (by simp [List.mem_range, hq]) N hmem
        simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
          decide_eq_true_eq] at this
        rcases this with h | h
        · exact absurd hg h
        · exact h
      · omega
    refine ⟨?_, ?_⟩
    · -- the anchor half
      intro hN201 x
      rw [hN] at hN201
      rcases Nat.lt_or_ge P 200 with hlo | hhi
      · rw [congrFun (hlaf (by omega)) x, congrFun (hspf (by omega) (by omega)) x]
        exact hI.1 (by rw [hP]; omega) x
      · rcases Nat.eq_or_lt_of_le hhi with he | hgt
        · -- P = 200: the next pc is 201, so there is nothing to prove
          exfalso
          refine hN201 ?_
          rw [← hN, siter_succ, sstep,
            show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
              = some (SInstr.bin .add "litAnchor" "p0" (.reg "ml")) from by
                rw [hP, ← he]; exact g200]
          simp only [sstepInstr, SState.setReg, SState.setPc]
          rw [hP, ← he]
        · rcases Nat.lt_or_ge P 202 with h201 | h202
          · -- P = 201: `searchPos := litAnchor`
            have hPv : P = 201 := by omega
            rw [congrFun (hlaf (by omega)) x, siter_succ, sstep,
              show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
                = some (SInstr.mov "searchPos" (.reg "litAnchor")) from by
                  rw [hP, hPv]; exact g201]
            simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_pos rfl, if_true]
            exact Nat.le_refl _
          · rcases Nat.lt_or_ge P 204 with h203 | h204
            · rw [congrFun (hlaf (by omega)) x, congrFun (hspf (by omega) (by omega)) x]
              exact hI.1 (by rw [hP]; omega) x
            · rcases Nat.eq_or_lt_of_le h204 with he4 | hgt4
              · -- P = 204: the cursor advances by 32, which cannot wrap
                have hPv : P = 204 := he4.symm
                have hsp := hI.2 (by rw [hP, hPv]; exact Or.inr ⟨by omega, by omega⟩) x
                have hla := hI.1 (by rw [hP, hPv]; omega) x
                rw [congrFun (hlaf (by omega)) x, siter_succ, sstep,
                  show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
                    = some (SInstr.bin .add "searchPos" "searchPos" (.imm 32)) from by
                      rw [hP, hPv]; exact g204]
                simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run,
                  if_pos rfl, if_true]
                have hSb := Geo.sBound (p := p) (S := S)
                rw [UInt64.toNat_add, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt 32 (by omega),
                  Nat.mod_eq_of_lt (by omega)]
                omega
              · rw [congrFun (hlaf (by omega)) x, congrFun (hspf (by omega) (by omega)) x]
                exact hI.1 (by rw [hP]; omega) x
    · -- the guard half
      intro hg x
      rw [hN] at hg
      rcases hguard hg with hgp | h41
      · by_cases h204 : P = 204
        · -- 204 advances the cursor and leaves the guarded stretch
          exfalso
          have hN205 : N = 205 := by
            rw [← hN, siter_succ, sstep,
              show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
                = some (SInstr.bin .add "searchPos" "searchPos" (.imm 32)) from by
                  rw [hP, h204]; exact g204]
            simp only [sstepInstr, SState.setReg, SState.setPc]
            rw [hP, h204]
          rw [hN205] at hg
          unfold GuardLive at hg; omega
        · have hPne : P ≠ 201 := by unfold GuardLive at hgp; omega
          rw [congrFun (hspf hPne h204) x]
          exact hI.2 (by rw [hP]; exact hgp) x
      · -- out of the guard: lane 0 passed it, and `searchPos` is warp-uniform
        have hbr : (siter p jj (initSt w inPtr outPtr gm smemB)).regs "loopC" 0 = 1 := by
          by_cases hc : (siter p jj (initSt w inPtr outPtr gm smemB)).regs "loopC" 0 == 1
          · exact beq_iff_eq.mp hc
          · exfalso
            have hN208 : N = 208 := by
              rw [← hN, siter_succ, sstep,
                show p[(siter p jj (initSt w inPtr outPtr gm smemB)).pc]?
                  = some (SInstr.braifnot "loopC" "Lx1") from by rw [hP, h41]; exact g41]
              simp only [sstepInstr, SState.setPc]
              rw [if_neg hc, lx1_is_208]
            rw [hN208] at hg
            unfold GuardLive at hg; omega
        have h0 := (loopC_iff (S := S) w inPtr outPtr gm smemB jj (Or.inr (Or.inl (by rw [hP, h41]))) 0).mp hbr
        rw [congrFun (hspf (by omega) (by omega)) x,
          uni_at w inPtr outPtr gm smemB jj "searchPos" (by simp [uniR]) x 0]
        exact h0
  · -- entry at 38: both are still the launch zeros
    intro j hj
    refine ⟨fun _ x => ?_, fun hg => absurd hg (by rw [hj]; unfold GuardLive; omega)⟩
    have hinit : (initSt w inPtr outPtr gm smemB).pc = 0 := rfl
    -- 38 is a label, and 37 is its only predecessor
    have hj1 : 1 ≤ j := by
      rcases Nat.eq_zero_or_pos j with h0 | h0
      · rw [h0, show siter p 0 (initSt w inPtr outPtr gm smemB)
          = initSt w inPtr outPtr gm smemB from rfl, hinit] at hj; omega
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
    -- `searchPos := 0` at 36 and `litAnchor := 0` at 35
    have hsp0 : (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs "searchPos" x
        = UInt64.ofNat 0 :=
      mov_imm_carried p _ hinit "searchPos" 0 37 0 (fun t ht => hf t (by omega))
        (fun t ht => absurd ht (by omega)) (by rw [show 37 - 0 - 1 = 36 from by omega]; exact g36)
        (by omega) (j - 1) h37 x
    have hla0 : (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs "litAnchor" x
        = UInt64.ofNat 0 := by
      refine mov_imm_carried p _ hinit "litAnchor" 0 37 1 (fun t ht => hf t (by omega)) ?_
        (by rw [show 37 - 1 - 1 = 35 from by omega]; exact g35) (by omega) (j - 1) h37 x
      intro t ht
      rw [show t = 0 from by omega, show 37 - 0 - 1 = 36 from by omega]
      exact n36la
    -- and 37 (`op := 0`) touches neither
    have hfr : ∀ r : String, (p[37]?.map (fun i => destOf i != some r)) = some true →
        (siter p j (initSt w inPtr outPtr gm smemB)).regs r
          = (siter p (j - 1) (initSt w inPtr outPtr gm smemB)).regs r := by
      intro r hr
      rw [show j = (j - 1) + 1 from by omega, siter_succ]
      refine sstep_regs_frame p _ r (fun i hi => ?_)
      rw [h37] at hi
      rw [hi] at hr
      simpa using hr
    rw [congrFun (hfr "litAnchor" n37la) x, congrFun (hfr "searchPos" n37sp) x, hsp0, hla0]
    exact Nat.le_refl _


-- ── What the select hands the extend loop ───────────────────────────────────




/-- `ballotOf` reads only the one register it names. -/
theorem ballotOf_congr (g g' : String → Lane → UInt64) (r : String) (h : g r = g' r) :
    ballotOf g r = ballotOf g' r := by
  unfold ballotOf; rw [h]

theorem sel_ft : ∀ t, t < 53 → (p[94 - t]?.map fallthroughOnlyB) = some true := by
  have h := Shape.selFrame (p := p)
  simp only [selFrameB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  intro t ht
  exact h.1.1 (94 - t) (by simp only [List.mem_map, List.mem_range]; exact ⟨52 - t, by omega, by omega⟩)

theorem sel_frame (r : String) (lo n : Nat) (hm : (r, lo, n) ∈ selFrames)
    (hn : lo + n = 94) :
    ∀ m, m + lo ≤ 94 → ∀ t, t < m →
      (p[94 - t - 1]?.map (fun i => destOf i != some r)) = some true := by
  have h := Shape.selFrame (p := p)
  simp only [selFrameB, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  intro m hlo t ht
  exact h.1.2 (r, lo, n) hm (94 - t - 1)
    (by simp only [List.mem_map, List.mem_range]; exact ⟨93 - t - lo, by omega, by omega⟩)

/-- **The two facts the extend loop inherits from the select**, at the state
    standing on `94 mov ecR` — the instruction the guard at 93 falls through to.

    * `p0 < (S - 12)`: the selected lane passed `pValid`, i.e. its own `posP` was
      inside the search limit, and `p0` *is* that lane's `posP`.
    * `cand0 < p0`: the selected lane passed `pCO`, i.e. its candidate was
      strictly behind its position — which is what makes the match offset
      positive and, downstream, what keeps `caD` inside the input stride.

    Both come from `ballot_select_holds`: the lane `clz∘brev∘ballot` names really
    does have `pHit` set, so the `shfl`ed values carry that lane's guards.  The
    step from "that lane's `posP`" to "*this* lane's `p0`" is `uni_at` plus
    `lane_val` — `posP = searchPos + lane`, `searchPos` is uniform, and the
    selected lane's `lane` register is its index. -/
theorem extend_entry_gen (init : SState) (hinit : init.pc = 0)
    (huni : ∀ j, Unif uniR (siter p j init))
    (hlaneval : ∀ j, 9 ≤ (siter p j init).pc →
      ∀ l : Lane, (siter p j init).regs "lane" l = UInt64.ofNat l.val)
    (k : Nat) (h94 : (siter p k init).pc = 94)
    (hspb : ∀ y : Lane, ((siter p k init).regs "searchPos" y).toNat < (S - 12)) (l : Lane) :
    ((siter p k init).regs "p0" l).toNat < (S - 12)
    ∧ ((siter p k init).regs "cand0" l).toNat < ((siter p k init).regs "p0" l).toNat
    ∧ ((siter p k init).regs "searchPos" l).toNat ≤ ((siter p k init).regs "p0" l).toNat := by
  have hsel := Geo.selShape (p := p) (S := S)
  simp only [selShapeB, Bool.and_eq_true, beq_iff_eq] at hsel
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨s42, s43⟩, s77⟩, s79⟩, s80⟩, s81⟩, s82⟩, s83⟩, s84⟩, s90⟩, s91⟩, s92⟩, s93⟩ := hsel
  have hle2 : sfindLabel p "Le2" = 203 := by
    have h := Shape.selFrame (p := p)
    simp only [selFrameB, Bool.and_eq_true, beq_iff_eq] at h
    exact h.2
  -- the instruction a state at 94 arrived by, `n` steps back
  have pre : ∀ (w : Nat) (i : SInstr), p[w]? = some i → 42 ≤ w → w ≤ 93 →
      siter p (k - (93 - w)) init = sstepInstr p i (siter p (k - (93 - w) - 1) init)
      ∧ (siter p (k - (93 - w) - 1) init).pc = w := by
    intro w i hw h1 h2
    have hx := pre_state p init hinit 94 (93 - w) i
      (fun t ht => sel_ft t (by omega))
      (by rw [show 94 - (93 - w) - 1 = w from by omega]; exact hw) (by omega) k h94
    exact ⟨hx.1, by rw [hx.2.1]; omega⟩
  -- a register unwritten since its own assignment still holds at 94
  have back : ∀ (r : String) (lo n : Nat), (r, lo, n) ∈ selFrames → lo + n = 94 → 42 ≤ lo →
      ∀ m, m + lo ≤ 94 →
      (siter p k init).regs r = (siter p (k - m) init).regs r := by
    intro r lo n hm hn hlo42 m hlo
    exact regs_back p init hinit r 94 m (by omega) (fun t ht => sel_ft t (by omega))
      (sel_frame r lo n hm hn m hlo) k h94
  -- ── the ballot, and that it is non-empty ──
  have hbal : (siter p k init).regs "bal"
      = fun _ => ballotOf (siter p k init).regs "pHit" := by
    rw [back "bal" 83 11 (by decide) (by omega) (by omega) 11 (by omega),
      (pre 82 _ s82 (by omega) (by omega)).1, show k - (93 - 82) - 1 = k - 12 from by omega]
    simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
    funext _
    exact ballotOf_congr _ _ "pHit" (back "pHit" 82 12 (by decide) (by omega) (by omega) 12 (by omega)).symm
  have hfound : (siter p k init).regs "found"
      = fun x => if (siter p k init).regs "bal" x != 0 then 1 else 0 := by
    rw [back "found" 93 1 (by decide) (by omega) (by omega) 1 (by omega),
      (pre 92 _ s92 (by omega) (by omega)).1, show k - (93 - 92) - 1 = k - 2 from by omega]
    simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SCmp.run,
      if_pos rfl, if_true]
    funext x
    rw [congrFun (back "bal" 83 11 (by decide) (by omega) (by omega) 2 (by omega)) x,
]
    rfl
  -- the guard at 93 fell through, so lane 0's `found` was set
  have hne : (siter p k init).regs "bal" l ≠ 0 := by
    have hp93 := (pre 93 _ s93 (by omega) (by omega)).2
    have hstep := (pre 93 _ s93 (by omega) (by omega)).1
    rw [show 93 - 93 = 0 from rfl] at hstep hp93
    have hpc94 : (sstepInstr p (SInstr.braifnot "found" "Le2") (siter p (k - 0 - 1) init)).pc = 94 := by
      rw [← hstep]; simpa using h94
    simp only [sstepInstr, SState.setPc] at hpc94
    have hf0 : (siter p (k - 0 - 1) init).regs "found" 0 = 1 := by
      by_cases hc : (siter p (k - 0 - 1) init).regs "found" 0 == 1
      · exact beq_iff_eq.mp hc
      · rw [if_neg hc, hle2] at hpc94; omega
    -- `found` at 93 is `found` at 94, and it is `bal ≠ 0`
    have hb : (siter p k init).regs "found" 0 = 1 := by
      rw [back "found" 93 1 (by decide) (by omega) (by omega) 1 (by omega)]
      simpa using hf0
    rw [congrFun hfound 0] at hb
    intro hz
    have : (siter p k init).regs "bal" 0 = 0 := by
      rw [huni k "bal" (by simp [uniR]) 0 l]; exact hz
    rw [this] at hb
    simp at hb
  -- ── the selected lane ──
  have hfl : (siter p k init).regs "fl"
      = fun x => clz32 (brev32 ((siter p k init).regs "bal" x)) := by
    have hrev : (siter p k init).regs "rev"
        = fun x => brev32 ((siter p k init).regs "bal" x) := by
      rw [back "rev" 84 10 (by decide) (by omega) (by omega) 10 (by omega),
        (pre 83 _ s83 (by omega) (by omega)).1, show k - (93 - 83) - 1 = k - 11 from by omega]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
      funext x
      rw [congrFun (back "bal" 83 11 (by decide) (by omega) (by omega) 11 (by omega)) x]
    rw [back "fl" 85 9 (by decide) (by omega) (by omega) 9 (by omega),
      (pre 84 _ s84 (by omega) (by omega)).1, show k - (93 - 84) - 1 = k - 10 from by omega]
    simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
    funext x
    rw [← congrFun (back "rev" 84 10 (by decide) (by omega) (by omega) 10 (by omega)) x,
      congrFun hrev x]
  have hL : (siter p k init).regs "fl" l
      = clz32 (brev32 (ballotOf (siter p k init).regs "pHit")) := by
    rw [congrFun hfl l, congrFun hbal l]
  have hlt : ((siter p k init).regs "fl" l).toNat < 32 := by
    rw [hL]
    exact clz_brev_ballot_lt _ "pHit" (by rw [← congrFun hbal l]; exact hne)
  have hhit : (siter p k init).regs "pHit" (toLane ((siter p k init).regs "fl" l)) = 1 := by
    rw [hL]
    exact ballot_select_holds _ "pHit" (by rw [← congrFun hbal l]; exact hne)
  -- ── the selected lane's guards ──
  have hLane : Lane := toLane ((siter p k init).regs "fl" l)
  -- `andp` at 81/80/79 peels `pHit` down to `pValid` and `pCO`
  have andp_at : ∀ (d a b : String) (w lo n loa na lob nb : Nat), p[w]? = some (SInstr.andp d a b) →
      (d, lo, n) ∈ selFrames → lo + n = 94 → 42 ≤ lo → lo = w + 1 →
      (a, loa, na) ∈ selFrames → loa + na = 94 → 42 ≤ loa → loa ≤ w →
      (b, lob, nb) ∈ selFrames → lob + nb = 94 → 42 ≤ lob → lob ≤ w →
      42 ≤ w → w ≤ 93 →
      ∀ x : Lane, (siter p k init).regs d x = 1 →
        (siter p k init).regs a x = 1 ∧ (siter p k init).regs b x = 1 := by
    intro d a b w lo n loa na lob nb hw hmd hnd h42d hlow hma hna h42a hwa hmb hnb h42b hwb h1 h2 x hx
    rw [back d lo n hmd hnd h42d (93 - w) (by omega), (pre w _ hw h1 h2).1] at hx
    simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true] at hx
    rw [show k - (93 - w) - 1 = k - (94 - w) from by omega] at hx
    by_cases hc : (siter p (k - (94 - w)) init).regs a x == 1
        ∧ (siter p (k - (94 - w)) init).regs b x == 1
    · refine ⟨?_, ?_⟩
      · rw [back a loa na hma hna h42a (94 - w) (by omega)]; exact beq_iff_eq.mp hc.1
      · rw [back b lob nb hmb hnb h42b (94 - w) (by omega)]; exact beq_iff_eq.mp hc.2
    · rw [if_neg hc] at hx; exact absurd hx (by decide)
  obtain ⟨hpH2, _⟩ := andp_at "pHit" "pH2" "pEq" 81 82 12 81 13 79 15
    s81 (by decide) (by omega) (by omega) (by omega) (by decide) (by omega) (by omega) (by omega)
    (by decide) (by omega) (by omega) (by omega) (by omega) (by omega) _ hhit
  obtain ⟨hpH1, hpCO⟩ := andp_at "pH2" "pH1" "pCO" 80 81 13 80 14 78 16
    s80 (by decide) (by omega) (by omega) (by omega) (by decide) (by omega) (by omega) (by omega)
    (by decide) (by omega) (by omega) (by omega) (by omega) (by omega) _ hpH2
  obtain ⟨hpValid, _⟩ := andp_at "pH1" "pValid" "pNE" 79 80 14 44 50 77 17
    s79 (by decide) (by omega) (by omega) (by omega) (by decide) (by omega) (by omega) (by omega)
    (by decide) (by omega) (by omega) (by omega) (by omega) (by omega) _ hpH1
  -- `setp` at 77 and 43 turn those predicates into inequalities
  have setp_at : ∀ (d a : String) (c : SArg) (w lo n loa na : Nat),
      p[w]? = some (SInstr.setp .lt d a c) →
      (d, lo, n) ∈ selFrames → lo + n = 94 → 42 ≤ lo → lo = w + 1 →
      (a, loa, na) ∈ selFrames → loa + na = 94 → 42 ≤ loa → loa ≤ w →
      42 ≤ w → w ≤ 93 →
      ∀ x : Lane, (siter p k init).regs d x = 1 →
        (siter p k init).regs a x < (siter p (k - (94 - w)) init).get x c := by
    intro d a c w lo n loa na hw hmd hnd h42d hlow hma hna h42a hwa h1 h2 x hx
    rw [back d lo n hmd hnd h42d (93 - w) (by omega), (pre w _ hw h1 h2).1] at hx
    simp only [sstepInstr, SState.setReg, SState.setPc, SCmp.run, if_pos rfl, if_true] at hx
    rw [show k - (93 - w) - 1 = k - (94 - w) from by omega] at hx
    rw [back a loa na hma hna h42a (94 - w) (by omega)]
    by_cases hc : (siter p (k - (94 - w)) init).regs a x < (siter p (k - (94 - w)) init).get x c
    · exact hc
    · rw [if_neg (by simpa using hc)] at hx; exact absurd hx (by decide)
  have hcandlt := setp_at "pCO" "cand" (.reg "posP") 77 78 16 64 30
    s77 (by decide) (by omega) (by omega) (by omega) (by decide) (by omega) (by omega) (by omega)
    (by omega) (by omega) _ hpCO
  have hposPlt := setp_at "pValid" "posP" (.imm (S - 12)) 43 44 50 43 51
    s43 (by decide) (by omega) (by omega) (by omega) (by decide) (by omega) (by omega) (by omega)
    (by omega) (by omega) _ hpValid
  simp only [SState.get] at hcandlt hposPlt
  rw [← back "posP" 43 51 (by decide) (by omega) (by omega) (94 - 77) (by omega)] at hcandlt
  -- ── the three values, at 94 ──
  have hposP : (siter p k init).regs "posP" (toLane ((siter p k init).regs "fl" l))
      = (siter p k init).regs "searchPos" (toLane ((siter p k init).regs "fl" l))
        + (siter p k init).regs "lane" (toLane ((siter p k init).regs "fl" l)) := by
    rw [back "posP" 43 51 (by decide) (by omega) (by omega) 51 (by omega),
      (pre 42 _ s42 (by omega) (by omega)).1, show k - (93 - 42) - 1 = k - 52 from by omega]
    simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true]
    rw [congrFun (back "searchPos" 42 52 (by decide) (by omega) (by omega) 52 (by omega)) _,
      congrFun (back "lane" 42 52 (by decide) (by omega) (by omega) 52 (by omega)) _]
  have hp0 : (siter p k init).regs "p0" l
      = (siter p k init).regs "searchPos" l + (siter p k init).regs "fl" l := by
    rw [back "p0" 91 3 (by decide) (by omega) (by omega) 3 (by omega),
      (pre 90 _ s90 (by omega) (by omega)).1, show k - (93 - 90) - 1 = k - 4 from by omega]
    simp only [sstepInstr, SState.setReg, SState.setPc, SOp.run, if_pos rfl, if_true]
    rw [congrFun (back "searchPos" 42 52 (by decide) (by omega) (by omega) 4 (by omega)) l,
      congrFun (back "fl" 85 9 (by decide) (by omega) (by omega) 4 (by omega)) l]
  have hcand0 : (siter p k init).regs "cand0" l
      = (siter p k init).regs "cand" (toLane ((siter p k init).regs "fl" l)) := by
    rw [back "cand0" 92 2 (by decide) (by omega) (by omega) 2 (by omega),
      (pre 91 _ s91 (by omega) (by omega)).1, show k - (93 - 91) - 1 = k - 3 from by omega]
    simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
    rw [congrFun (back "fl" 85 9 (by decide) (by omega) (by omega) 3 (by omega)) l,
      congrFun (back "cand" 64 30 (by decide) (by omega) (by omega) 3 (by omega)) _]
  -- the selected lane's `posP` IS this lane's `p0`
  have hsame : (siter p k init).regs "posP" (toLane ((siter p k init).regs "fl" l))
      = (siter p k init).regs "p0" l := by
    rw [hposP, hp0,
      huni k "searchPos" (by simp [uniR]) (toLane ((siter p k init).regs "fl" l)) l,
      hlaneval k (by rw [h94]; omega) (toLane ((siter p k init).regs "fl" l))]
    congr 1
    have : (toLane ((siter p k init).regs "fl" l)).val
        = ((siter p k init).regs "fl" l).toNat := Nat.mod_eq_of_lt hlt
    rw [this, UInt64.ofNat_toNat]
  rw [hsame] at hcandlt hposPlt
  refine ⟨?_, ?_, ?_⟩
  · exact (lt_ofNat_iff _ (S - 12) (by have := Geo.sBound (p := p) (S := S); omega)).mp hposPlt
  · rw [hcand0]; exact UInt64.lt_iff_toNat_lt.mp hcandlt
  · -- `p0 = searchPos + fl` and neither term is anywhere near wrapping
    have h1 := hspb l
    have h2 := hlt
    have hsum : ((siter p k init).regs "p0" l).toNat
        = ((siter p k init).regs "searchPos" l).toNat + ((siter p k init).regs "fl" l).toNat := by
      rw [hp0, UInt64.toNat_add, Nat.mod_eq_of_lt
        (by have := Geo.sBound (p := p) (S := S); omega)]
    omega



/-- `extend_entry_gen` at the launch state. -/
theorem extend_entry (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (k : Nat) (h94 : (siter p k (initSt w inPtr outPtr gm smemB)).pc = 94) (l : Lane) :
    ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l).toNat < (S - 12)
    ∧ ((siter p k (initSt w inPtr outPtr gm smemB)).regs "cand0" l).toNat
        < ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l).toNat
    ∧ ((siter p k (initSt w inPtr outPtr gm smemB)).regs "searchPos" l).toNat
        ≤ ((siter p k (initSt w inPtr outPtr gm smemB)).regs "p0" l).toNat :=
  extend_entry_gen (S := S) (initSt w inPtr outPtr gm smemB) rfl
    (uni_at w inPtr outPtr gm smemB) (lane_val w inPtr outPtr gm smemB)
    k h94 (fun y => (loop_inv w inPtr outPtr gm smemB k (by
      rw [h94]; simp only [loopS, List.mem_map, List.mem_range]
      exact ⟨56, by omega, by omega⟩)).2 (by rw [h94]; unfold GuardLive; omega) y) l

end Lz4Sites
