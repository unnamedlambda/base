module
public import AlgorithmLib.Host.RaceSound
meta import AlgorithmLib.Host.RaceSound
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The World's tracker implements the abstract one

`World.Race` keeps clocks as arrays and names an access by `(party, tick)`;
`Device.TSt` keeps clocks as functions and names an access by its place in the
history. `Refines` relates them, and each move of the World's tracker is a
move of the abstract one that keeps the relation (`op_refines`,
`join_refines`, `CReached.refines`) --- so `accepted_ordered` and
`accepted_any_order` hold of what the device model accepts. The one
difference in the check, `ordered`'s same-party disjunct, is redundant given
the clock invariants: a party's new clock is ahead of all it did before.
-/

namespace AlgorithmLib.Device

open AlgorithmLib.HProg.Sem

theorem Clock.get_eq (c : Clock) (i : Nat) : c.get i = (c[i]?).getD 0 := by
  simp [Clock.get, Array.getD_eq_getD_getElem?]

theorem Clock.get_join (a b : Clock) (i : Nat) : (a.join b).get i = max (a.get i) (b.get i) := by
  rw [Clock.get_eq, Clock.get_eq, Clock.get_eq]
  simp only [Clock.join, Array.getElem?_map, Array.getElem?_range]
  by_cases h : i < max a.size b.size
  · simp [h, Clock.get_eq]
  · have ha : a[i]? = none := Array.getElem?_eq_none (by omega)
    have hb : b[i]? = none := Array.getElem?_eq_none (by omega)
    simp [h, ha, hb]

theorem Clock.get_bump (c : Clock) (p i : Nat) :
    (c.bump p).get i = if i = p then c.get p + 1 else c.get i := by
  unfold Clock.bump
  simp only [Clock.get_eq]
  by_cases hp : p < c.size
  · simp only [hp, if_true, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
    by_cases hi : i = p
    · subst hi; simp [hp]
    · simp [Ne.symm hi, hi]
  · simp only [hp, if_false, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
    have hpc : c[p]? = none := Array.getElem?_eq_none (by omega)
    by_cases hi : i = p
    · subst hi
      simp [Array.getElem?_append, hpc]
      rw [if_pos (by omega), if_neg hp, Array.getElem?_replicate, if_pos (by omega)]
      rfl
    · simp only [Ne.symm hi, if_false, hi]
      rw [Array.getElem?_append]
      split
      · rfl
      · rename_i hic
        have : c[i]? = none := Array.getElem?_eq_none (by omega)
        rw [this]
        simp [Array.getElem?_replicate]
        split <;> rfl

theorem growTo_getD {α : Type} (xs : Array α) (i j : Nat) (d : α) :
    (growTo xs i d).getD j d = xs.getD j d := by
  simp only [growTo, Array.getD_eq_getD_getElem?]
  split
  · rfl
  · rw [Array.getElem?_append]
    split
    · rfl
    · rename_i h
      have : xs[j]? = none := Array.getElem?_eq_none (by omega)
      rw [this]; simp [Array.getElem?_replicate]; split <;> rfl

theorem growTo_size {α : Type} (xs : Array α) (i : Nat) (d : α) : i < (growTo xs i d).size := by
  simp only [growTo]; split
  · assumption
  · simp; omega

theorem set_growTo_getD {α : Type} (xs : Array α) (i j : Nat) (d v : α) :
    ((growTo xs i d).set! i v).getD j d = if j = i then v else xs.getD j d := by
  have hs := growTo_size xs i d
  simp only [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds]
  by_cases h : j = i
  · subst h; simp [hs]
  · simp only [Ne.symm h, if_false, h]
    have := growTo_getD xs i j d
    simp only [Array.getD_eq_getD_getElem?] at this
    exact this

theorem modify_growTo_getD {α : Type} (xs : Array α) (i j : Nat) (d : α) (f : α → α) :
    ((growTo xs i d).modify i f).getD j d = if j = i then f (xs.getD j d) else xs.getD j d := by
  have hs := growTo_size xs i d
  have hg := growTo_getD xs i j d
  simp only [Array.getD_eq_getD_getElem?] at hg ⊢
  rw [Array.getElem?_modify]
  by_cases h : j = i
  · subst h
    simp only [if_true]
    rw [← hg]
    rcases e : (growTo xs j d)[j]? with _ | x
    · exact absurd e (by simp [hs])
    · rfl
  · simp only [Ne.symm h, if_false, h]
    exact hg

theorem Race.clock_setClock (r : Race) (p q : Nat) (c : Clock) :
    (r.setClock p c).clock q = if q = p then c else r.clock q := by
  simp only [Race.clock, Race.setClock]
  exact set_growTo_getD r.clocks p q #[] c

def readsFold (e : Nat × Nat) (rs : List Nat) (r : Race) : Race :=
  rs.foldl (fun r b => { r with reads := (growTo r.reads b []).modify b (e :: ·) }) r

def writesFold (e : Nat × Nat) (ws : List Nat) (r : Race) : Race :=
  ws.foldl (fun r b => { r with lastW := (growTo r.lastW b none).set! b (some e),
                                 reads := (growTo r.reads b []).set! b [] }) r

theorem readsFold_other (e : Nat × Nat) : ∀ (rs : List Nat) (r : Race),
    (readsFold e rs r).clocks = r.clocks ∧ (readsFold e rs r).lastW = r.lastW
  | [], _ => ⟨rfl, rfl⟩
  | b :: rs, r => by
      have := readsFold_other e rs { r with reads := (growTo r.reads b []).modify b (e :: ·) }
      exact this

theorem readsFold_mem (e : Nat × Nat) : ∀ (rs : List Nat) (r : Race) (b : Nat) (x : Nat × Nat),
    x ∈ (readsFold e rs r).reads.getD b [] ↔ x ∈ r.reads.getD b [] ∨ (b ∈ rs ∧ x = e)
  | [], r, b, x => by simp [readsFold]
  | c :: rs, r, b, x => by
      have ih := readsFold_mem e rs { r with reads := (growTo r.reads c []).modify c (e :: ·) } b x
      simp only [readsFold, List.foldl_cons] at ih ⊢
      rw [ih]
      simp only [modify_growTo_getD]
      by_cases h : b = c
      · subst h; simp only [if_pos rfl, List.mem_cons, true_or, true_and]
        by_cases hx : x = e <;> simp [hx]
      · rw [if_neg h]; simp [List.mem_cons, h]

theorem writesFold_other (e : Nat × Nat) : ∀ (ws : List Nat) (r : Race),
    (writesFold e ws r).clocks = r.clocks
  | [], _ => rfl
  | b :: ws, r => writesFold_other e ws _

theorem writesFold_lastW (e : Nat × Nat) : ∀ (ws : List Nat) (r : Race) (b : Nat),
    (writesFold e ws r).lastW.getD b none = if b ∈ ws then some e else r.lastW.getD b none
  | [], r, b => by simp [writesFold]
  | c :: ws, r, b => by
      have ih := writesFold_lastW e ws { r with lastW := (growTo r.lastW c none).set! c (some e),
                                                  reads := (growTo r.reads c []).set! c [] } b
      simp only [writesFold, List.foldl_cons] at ih ⊢
      rw [ih]
      simp only [set_growTo_getD]
      by_cases h : b = c
      · subst h; simp
      · by_cases hw : b ∈ ws <;> simp [h, hw]

theorem writesFold_reads (e : Nat × Nat) : ∀ (ws : List Nat) (r : Race) (b : Nat),
    (writesFold e ws r).reads.getD b [] = if b ∈ ws then [] else r.reads.getD b []
  | [], r, b => by simp [writesFold]
  | c :: ws, r, b => by
      have ih := writesFold_reads e ws { r with lastW := (growTo r.lastW c none).set! c (some e),
                                                  reads := (growTo r.reads c []).set! c [] } b
      simp only [writesFold, List.foldl_cons] at ih ⊢
      rw [ih]
      simp only [set_growTo_getD]
      by_cases h : b = c
      · subst h; simp
      · by_cases hw : b ∈ ws <;> simp [h, hw]

-- ---------------------------------------------------------------------------
-- The relation
-- ---------------------------------------------------------------------------

/-- How the World names operation `i`: its party and its own tick. -/
def TSt.tag (σ : TSt) (i : Nat) : Nat × Nat := ((σ.op i).party, (σ.op i).clock (σ.op i).party)

structure Refines (r : Race) (σ : TSt) : Prop where
  clk : ∀ q x, (r.clock q).get x = σ.clk q x
  lastW : ∀ b, r.lastW.getD b none = (σ.lastW b).map σ.tag
  reads : ∀ b x, x ∈ r.reads.getD b [] ↔ ∃ i ∈ σ.reads b, x = σ.tag i

theorem access_eq {r : Race} {p : Nat} {c : Clock} {rs ws : List Nat} {r' : Race}
    (h : r.access p c rs ws = some r') :
    ((rs ++ ws).all (fun b => ((r.lastW.getD b none).map (ordered p c)).getD true)
      && ws.all (fun b => (r.reads.getD b []).all (ordered p c))) = true ∧
    r' = writesFold (p, c.get p) ws (readsFold (p, c.get p) rs r) := by
  unfold Race.access at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hok
    simp only [Option.some.injEq] at h
    exact ⟨by simpa using hok, h.symm⟩

/-- An access the World's check lets through, the abstract check lets through:
    `ordered`'s same-party case is covered because the new clock is ahead of
    everything its party did. -/
theorem ordered_hb {σ : TSt} (hg : Good σ) {p : Nat} {c : Clock} {vc : VC}
    (hc : ∀ x, c.get x = vc x) (hvp : vc p = σ.clk p p + 1) {w : Nat} (hw : w < σ.n)
    (h : ordered p c (σ.tag w) = true) : (σ.op w).clock (σ.op w).party ≤ vc (σ.op w).party := by
  simp only [ordered, TSt.tag, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  rcases h with h | h
  · rw [h, hvp]
    have := hg.2.1 _ (Or.inr (Or.inr ⟨w, hw, rfl⟩)) p
    omega
  · rw [← hc]; exact h

/-- **An operation the World accepts is one the abstract tracker accepts**, and
    the states after still correspond. -/
theorem op_refines {σ : TSt} (hg : Good σ) {r r' : Race} (hr : Refines r σ) {p : Nat}
    {rs ws : List Nat} (h : r.op p rs ws = some r') :
    σ.allows p ((VC.join (σ.clk p) (σ.clk host)).bump p) rs ws ∧
      Refines r' (σ.record p ((VC.join (σ.clk p) (σ.clk host)).bump p) rs ws) := by
  obtain ⟨hk, ho, ⟨hlw, hrd, _⟩, _⟩ := hg
  -- the World's new clock is the abstract one
  unfold Race.op Race.issue at h
  generalize hc : ((r.clock p).join (r.clock hostParty)).bump p = c at h
  generalize hvc : (VC.join (σ.clk p) (σ.clk host)).bump p = vc
  have hcv : ∀ x, c.get x = vc x := by
    intro x; subst hc; subst hvc
    rw [Clock.get_bump]
    simp only [VC.bump, VC.join, Clock.get_join, hr.clk, hostParty, host]
  have hhost : σ.clk host p ≤ σ.clk p p := ho _ (Or.inl ⟨host, rfl⟩) p
  have hvp : vc p = σ.clk p p + 1 := by subst hvc; simp [VC.bump, VC.join]; omega
  obtain ⟨hok, rfl⟩ := access_eq h
  have hlw1 : ∀ b, (r.setClock p c).lastW.getD b none = r.lastW.getD b none := fun _ => rfl
  have hrd1 : ∀ b, (r.setClock p c).reads.getD b [] = r.reads.getD b [] := fun _ => rfl
  simp only [Bool.and_eq_true, List.all_eq_true, hlw1, hrd1] at hok
  obtain ⟨hok1, hok2⟩ := hok
  have hg' : Good σ := ⟨hk, ho, ⟨hlw, hrd, ‹_›⟩, ‹_›⟩
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro b hb w hw
    have := hok1 b hb
    rw [hr.lastW b, hw] at this
    exact ordered_hb hg' hcv hvp (hlw b w hw) (by simpa using this)
  · intro b hb i hi
    have hmem : σ.tag i ∈ r.reads.getD b [] := (hr.reads b _).mpr ⟨i, hi, rfl⟩
    exact ordered_hb hg' hcv hvp (hrd b i hi) (hok2 b hb _ hmem)
  · -- the states after correspond
    have htag_old : ∀ i, i < σ.n → (σ.record p vc rs ws).tag i = σ.tag i := fun i hi => by
      simp only [TSt.tag, record_op_old hi]
    have htag_new : (σ.record p vc rs ws).tag σ.n = (p, c.get p) := by
      simp only [TSt.tag, record_op_new, hcv]
    refine ⟨?_, ?_, ?_⟩
    · intro q x
      have e1 : (writesFold (p, c.get p) ws (readsFold (p, c.get p) rs (r.setClock p c))).clock q
          = (r.setClock p c).clock q := by
        simp only [Race.clock, writesFold_other, (readsFold_other _ _ _).1]
      rw [e1, Race.clock_setClock,
        show (σ.record p vc rs ws).clk q = if q = p then vc else σ.clk q from rfl]
      by_cases hq : q = p
      · subst hq; simp [hcv]
      · simp [hq, hr.clk]
    · intro b
      rw [writesFold_lastW]
      simp only [(readsFold_other _ _ _).2, hlw1]
      rw [show (σ.record p vc rs ws).lastW b = if b ∈ ws then some σ.n else σ.lastW b from rfl]
      by_cases hb : b ∈ ws
      · simp [hb, htag_new]
      · simp only [hb, if_false]
        rw [hr.lastW b]
        cases hl : σ.lastW b with
        | none => rfl
        | some w => simp [htag_old w (hlw b w hl)]
    · intro b x
      rw [writesFold_reads,
        show (σ.record p vc rs ws).reads b
          = if b ∈ ws then [] else if b ∈ rs then σ.n :: σ.reads b else σ.reads b from rfl]
      by_cases hb : b ∈ ws
      · simp [hb]
      · simp only [hb, if_false]
        rw [readsFold_mem, hrd1, hr.reads b x]
        constructor
        · rintro (⟨i, hi, rfl⟩ | ⟨hbr, rfl⟩)
          · refine ⟨i, ?_, (htag_old i (hrd b i hi)).symm⟩
            split
            · exact List.mem_cons_of_mem _ hi
            · exact hi
          · refine ⟨σ.n, by simp [hbr], htag_new.symm⟩
        · rintro ⟨i, hi, rfl⟩
          split at hi
          · rename_i hbr
            rcases List.mem_cons.mp hi with rfl | hi
            · exact Or.inr ⟨hbr, htag_new⟩
            · exact Or.inl ⟨i, hi, htag_old i (hrd b i hi)⟩
          · exact Or.inl ⟨i, hi, htag_old i (hrd b i hi)⟩

/-- **A party learning a clock** --- a sync, a stream fence, an event wait --- is
    the abstract `learn`, when the clock is one the state holds. -/
theorem join_refines {σ : TSt} {r : Race} (hr : Refines r σ) (q : Nat) (C : Clock) (D : VC)
    (hCD : ∀ x, C.get x = D x) :
    Refines (r.joinInto q C) { σ with clk := fun q' => if q' = q then VC.join (σ.clk q) D else σ.clk q' } := by
  refine ⟨?_, hr.lastW, hr.reads⟩
  intro q' x
  simp only [Race.joinInto, Race.clock_setClock]
  by_cases h : q' = q
  · subst h; simp [Clock.get_join, hr.clk, hCD, VC.join]
  · simp [h, hr.clk]

-- ---------------------------------------------------------------------------
-- Every state the World's tracker reaches corresponds to one the abstract
-- tracker reaches
-- ---------------------------------------------------------------------------

/-- The World's tracker, with the clocks its events hold. -/
inductive CReached : Race → List Clock → Prop
  | init : CReached {} []
  /-- An operation issued and accepted. -/
  | op {r r' evs} (p : Nat) (rs ws : List Nat) : CReached r evs → r.op p rs ws = some r' →
      CReached r' evs
  /-- An event records a party's clock. -/
  | record {r evs} (p : Nat) : CReached r evs → CReached r (r.clock p :: evs)
  /-- A party learns another's clock or an event's. -/
  | learnParty {r evs} (q p : Nat) : CReached r evs → CReached (r.joinInto q (r.clock p)) evs
  | learnEvent {r evs} (q : Nat) (C : Clock) : CReached r evs → C ∈ evs →
      CReached (r.joinInto q C) evs
  /-- A party learns two parties' clocks at once: a new stream fenced behind the
      default stream and the host. -/
  | learnTwo {r evs} (q p₁ p₂ : Nat) : CReached r evs →
      CReached (r.joinInto q ((r.clock p₁).join (r.clock p₂))) evs
  /-- A party learns the empty clock: nothing. -/
  | learnNothing {r evs} (q : Nat) : CReached r evs → CReached (r.joinInto q #[]) evs

theorem refines_empty : Refines {} TSt.init := by
  refine ⟨?_, ?_, ?_⟩
  · intro q x; simp [Race.clock, Clock.get, TSt.init]
  · intro b; simp [TSt.init]
  · intro b x; simp [TSt.init]

/-- **What the World's tracker accepts, the abstract tracker accepts**: every
    state it reaches corresponds to a state the abstract tracker reaches, and
    the clocks its events hold are snapshots that state keeps. -/
theorem CReached.refines {r : Race} {evs : List Clock} (h : CReached r evs) :
    ∃ σ, Reached σ ∧ Refines r σ ∧ ∀ C ∈ evs, ∃ D ∈ σ.snaps, ∀ x, C.get x = D x := by
  induction h with
  | init => exact ⟨TSt.init, .init, refines_empty, by simp⟩
  | op p rs ws _ hop ih =>
      obtain ⟨σ, hσ, hr, hev⟩ := ih
      obtain ⟨hal, hr'⟩ := op_refines (reached_good hσ) hr hop
      exact ⟨_, .step hσ (.issue σ p rs ws hal), hr', hev⟩
  | record p _ ih =>
      obtain ⟨σ, hσ, hr, hev⟩ := ih
      refine ⟨{ σ with snaps := σ.clk p :: σ.snaps }, .step hσ (.snap σ _ (Or.inl ⟨p, rfl⟩)),
        ⟨hr.clk, hr.lastW, hr.reads⟩, ?_⟩
      intro C hC
      rcases List.mem_cons.mp hC with rfl | hC
      · exact ⟨σ.clk p, List.mem_cons_self .., hr.clk p⟩
      · obtain ⟨D, hD, hCD⟩ := hev C hC
        exact ⟨D, List.mem_cons_of_mem _ hD, hCD⟩
  | learnParty q p _ ih =>
      obtain ⟨σ, hσ, hr, hev⟩ := ih
      exact ⟨_, .step hσ (.learn σ q (σ.clk p) (Or.inl ⟨p, rfl⟩)),
        join_refines hr q _ _ (hr.clk p), hev⟩
  | learnEvent q C _ hC ih =>
      obtain ⟨σ, hσ, hr, hev⟩ := ih
      obtain ⟨D, hD, hCD⟩ := hev C hC
      exact ⟨_, .step hσ (.learn σ q D (Or.inr (Or.inl hD))), join_refines hr q C D hCD, hev⟩
  | learnTwo q p₁ p₂ _ ih =>
      obtain ⟨σ, hσ, hr, hev⟩ := ih
      have hσ₁ := Reached.step hσ (.learn σ q (σ.clk p₁) (Or.inl ⟨p₁, rfl⟩))
      by_cases h : p₂ = q
      · -- learning its own clock again teaches `q` nothing
        subst h
        refine ⟨_, hσ₁, ⟨?_, hr.lastW, hr.reads⟩, hev⟩
        intro q' x
        simp only [Race.joinInto, Race.clock_setClock]
        by_cases h' : q' = p₂
        · subst h'; simp [Clock.get_join, hr.clk, VC.join]; omega
        · simp [h', hr.clk]
      · refine ⟨_, .step hσ₁ (.learn _ q (σ.clk p₂) (Or.inl ⟨p₂, by simp [h]⟩)),
          ⟨?_, hr.lastW, hr.reads⟩, hev⟩
        intro q' x
        simp only [Race.joinInto, Race.clock_setClock]
        by_cases h' : q' = q
        · subst h'; simp [Clock.get_join, hr.clk, VC.join]
        · simp [h', hr.clk]
  | learnNothing q _ ih =>
      obtain ⟨σ, hσ, hr, hev⟩ := ih
      refine ⟨σ, hσ, ⟨?_, hr.lastW, hr.reads⟩, hev⟩
      intro q' x
      simp only [Race.joinInto, Race.clock_setClock]
      by_cases h : q' = q
      · subst h; rw [if_pos rfl, Clock.get_join, hr.clk]; simp [Clock.get]
      · simp [h, hr.clk]

/-- **So the World's tracker orders every conflict it accepts.** -/
theorem CReached.ordered {r : Race} {evs : List Clock} (h : CReached r evs) :
    ∃ σ, Reached σ ∧ Refines r σ ∧ Ordered σ := by
  obtain ⟨σ, hσ, hr, -⟩ := h.refines
  exact ⟨σ, hσ, hr, accepted_ordered hσ⟩

end AlgorithmLib.Device
