module
public import AlgorithmLib.ML.Ptx.LaneWarp
meta import AlgorithmLib.ML.Ptx.LaneWarp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.LaneSettle` — a lane's collectives depend on its earlier answers

A lane's run only adds observations (`laneI_grows`), so the collectives it has
made only grow (`collCalls_prefix`). Two runs of a lane from one state whose
answers agree below the `k`-th end alike, or both make more than `k`
collectives and agree on the first `k + 1` (`laneI_prefix`): a collective's
arguments depend only on the answers before it.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyLane

namespace AlgorithmLib.ML.LaneSettle

/-! ## A run only adds observations -/

/-- `w'` holds `w`'s observations and more after them. -/
def Grows (w w' : World) : Prop := ∃ L, w'.obs = L ++ w.obs

theorem Grows.refl (w : World) : Grows w w := ⟨[], rfl⟩

theorem Grows.trans {a b c : World} (h1 : Grows a b) (h2 : Grows b c) : Grows a c := by
  obtain ⟨L1, h1⟩ := h1
  obtain ⟨L2, h2⟩ := h2
  exact ⟨L2 ++ L1, by rw [h2, h1, List.append_assoc]⟩

theorem grows_call (w : World) (c : Callee) (args : List V) : Grows w (obsCall w c args) := ⟨[_], rfl⟩

theorem Grows.mem {w w' : World} (h : Grows w w') (m : Mem) : Grows w { w' with mem := m } := h

theorem grows_stw {w w' : World} {a : UInt64} {n : Nat} {v : UInt64} (h : stw w a n v = some w') : Grows w w' := by
  simp only [stw] at h
  obtain ⟨m, _, h⟩ := Option.bind_eq_some_iff.mp h
  cases h
  exact ⟨[_], rfl⟩

section Rel
variable (R : World → World → Prop) (hrefl : ∀ w, R w w) (htrans : ∀ a b c, R a b → R b c → R a c)
  (hcall : ∀ w c args, R w (obsCall w c args))
  (hbar : ∀ w (b : ByteArray), R w { callW w .barrier [] with mem := { (callW w .barrier []).mem with arena := b } })
  (hstw : ∀ w w' a n v, stw w a n v = some w' → R w w')
include hrefl htrans hcall

theorem exLog_rel (hw : Hw) (f : Nat → F32) : ∀ (e : WFExp) (w : World), R w (exLog hw f e w) := by
  intro e
  induction e with
  | reg _ | lit _ => intro w; exact hrefl w
  | add a b iha ihb | mul a b iha ihb | maxW a b iha ihb | geF a b iha ihb =>
      intro w; exact htrans _ _ _ (iha w) (ihb _)
  | neg a ih | inv a ih | rsqrt a ih => intro w; exact ih w
  | ex2 a ih | exp a ih => intro w; exact htrans _ _ _ (ih w) (hcall _ _ _)

omit hcall in
theorem foldlM_rel {f : LI → Nat → Option LI} (hf : ∀ j s s', f s j = some s' → R s.w s'.w) :
    ∀ (L : List Nat) (s s' : LI), L.foldlM f s = some s' → R s.w s'.w := by
  intro L
  induction L with
  | nil => intro s s' h; cases h; exact hrefl _
  | cons j js ih =>
      intro s s' h
      simp only [List.foldlM_cons] at h
      obtain ⟨s1, h1, h⟩ := Option.bind_eq_some_iff.mp h
      exact htrans _ _ _ (hf j s s1 h1) (ih s1 s' h)

omit hcall in
theorem whileL_rel {step : LI → Option LI} {go : LI → Bool} (hs : ∀ s s', step s = some s' → R s.w s'.w) :
    ∀ (n : Nat) (s s' : LI), whileL step go n s = some s' → R s.w s'.w := by
  intro n
  induction n with
  | zero =>
      intro s s' h
      simp only [whileL] at h
      split at h
      · cases h
      · cases h; exact hrefl _
  | succ n ih =>
      intro s s' h
      simp only [whileL] at h
      split at h
      · obtain ⟨s1, h1, h⟩ := Option.bind_eq_some_iff.mp h
        exact htrans _ _ _ (hs s s1 h1) (ih s1 s' h)
      · cases h; exact hrefl _

include hbar hstw in
/-- **What every step of a lane's run keeps, its run keeps**: a relation
    between worlds that is reflexive and transitive and holds across a
    collective, a barrier (its call and shared memory replaced) and a store. -/
theorem laneI_rel : ∀ (s : EWStmt) (x : LCtx) (li li' : LI), laneI x s li = some li' → R li.w li'.w := by
  intro s
  induction s with
  | skip => intro x li li' h; cases h; exact hrefl _
  | seq a b iha ihb =>
      intro x li li' h
      simp only [laneI] at h
      obtain ⟨s1, h1, h⟩ := Option.bind_eq_some_iff.mp h
      exact htrans _ _ _ (iha x li s1 h1) (ihb x s1 li' h)
  | setI d e => intro x li li' h; cases h; exact hrefl _
  | ldB d a | ldH d a =>
      intro x li li' h
      simp only [laneI] at h
      obtain ⟨v, _, h⟩ := Option.bind_eq_some_iff.mp h
      cases h; exact hrefl _
  | stB a v g | stH a v g =>
      intro x li li' h
      simp only [laneI] at h
      split at h
      · obtain ⟨w, hw, h⟩ := Option.bind_eq_some_iff.mp h
        cases h; exact hstw _ _ _ _ _ hw
      · cases h; exact hrefl _
  | ballot d p | shflIdx d v sr =>
      intro x li li' h
      simp only [laneI] at h
      split at h
      · cases h; exact hcall _ _ _
      · cases h
  | barrier =>
      intro x li li' h
      simp only [laneI] at h
      split at h
      · split at h
        · cases h; exact hbar _ _
        · cases h
      · cases h
  | iif c a b iha ihb =>
      intro x li li' h
      simp only [laneI] at h
      split at h
      · exact iha x li li' h
      · exact ihb x li li' h
  | iwhile n c cs body ih =>
      intro x li li' h
      simp only [laneI] at h
      refine whileL_rel R hrefl htrans (fun s s' hs => ?_) n li li' h
      obtain ⟨s1, h1, hs⟩ := Option.map_eq_some_iff.mp hs
      cases hs
      exact ih x s s1 h1
  | locals rs body ih =>
      intro x li li' h
      simp only [laneI] at h
      obtain ⟨s1, h1, h⟩ := Option.map_eq_some_iff.mp h
      cases h
      exact ih x li s1 h1
  | setR d e => intro x li li' h; cases h; exact exLog_rel R hrefl htrans hcall _ _ _ _
  | shflXor d sr m =>
      intro x li li' h
      simp only [laneI] at h
      split at h
      · cases h; exact hcall _ _ _
      · cases h
  | loadIdx d b ix =>
      intro x li li' h
      simp only [laneI] at h
      obtain ⟨j, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨v, _, h⟩ := Option.bind_eq_some_iff.mp h
      cases h; exact hrefl _
  | loadV4 d0 d1 d2 d3 b ix =>
      intro x li li' h
      simp only [laneI] at h
      obtain ⟨j, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨v0, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨v1, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨v2, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨v3, _, h⟩ := Option.bind_eq_some_iff.mp h
      cases h; exact hrefl _
  | storeLane b ix n =>
      intro x li li' h
      simp only [laneI] at h
      obtain ⟨j, _, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨w, hw, h⟩ := Option.bind_eq_some_iff.mp h
      cases h; exact hstw _ _ _ _ _ hw
  | storeLane0 b ix n =>
      intro x li li' h
      simp only [laneI] at h
      split at h
      · obtain ⟨j, _, h⟩ := Option.bind_eq_some_iff.mp h
        obtain ⟨w, hw, h⟩ := Option.bind_eq_some_iff.mp h
        cases h; exact hstw _ _ _ _ _ hw
      · cases h; exact hrefl _
  | forN n body ih =>
      intro x li li' h
      simp only [laneI] at h
      split at h
      · exact foldlM_rel R hrefl htrans (fun j s s' hs => ih _ s s' hs) _ li li' h
      · cases h
  | forM b a body ih =>
      intro x li li' h
      simp only [laneI] at h
      split at h
      · obtain ⟨v, _, h⟩ := Option.bind_eq_some_iff.mp h
        split at h
        · exact foldlM_rel R hrefl htrans (fun j s s' hs => ih _ s s' hs) _ li li' h
        · cases h
      · cases h
  | stSm | ldSm | cvtIF => intro x li li' h; simp [laneI] at h

end Rel

/-- **A lane's run only adds observations.** -/
theorem laneI_grows : ∀ (s : EWStmt) (x : LCtx) (li li' : LI), laneI x s li = some li' → Grows li.w li'.w :=
  laneI_rel Grows Grows.refl (fun _ _ _ => Grows.trans) grows_call (fun _ _ => (grows_call _ _ _).mem _)
    (fun _ _ _ _ _ h => grows_stw h)

/-- The collectives made since `st` only grow. -/
theorem collCalls_prefix {w w' : World} (h : Grows w w') (st : Nat) : collCalls w st <+: collCalls w' st := by
  obtain ⟨L, hL⟩ := h
  unfold collCalls
  rw [hL, List.reverse_append, List.drop_append, List.filterMap_append]
  by_cases hs : st ≤ w.obs.reverse.length
  · exact List.prefix_append _ _
  · rw [List.drop_eq_nil_of_le (by omega)]
    exact List.nil_prefix

theorem take_of_prefix {α : Type} {l₁ l₂ : List α} (h : l₁ <+: l₂) {n : Nat} (hn : n ≤ l₁.length) :
    l₂.take n = l₁.take n := by
  obtain ⟨t, rfl⟩ := h
  exact List.take_append_of_le_length hn

/-! ## Answers below the `k`-th -/

/-- Two lane states alike, or both past their `k`-th collective with the same
    first `k + 1`. -/
def Div (st k : Nat) (a b : LI) : Prop :=
  a = b ∨ (k < (collCalls a.w st).length ∧ k < (collCalls b.w st).length ∧
    (collCalls a.w st).take (k + 1) = (collCalls b.w st).take (k + 1))

theorem div_grows {st k : Nat} {a b a' b' : LI}
    (h : k < (collCalls a.w st).length ∧ k < (collCalls b.w st).length ∧
      (collCalls a.w st).take (k + 1) = (collCalls b.w st).take (k + 1))
    (ga : Grows a.w a'.w) (gb : Grows b.w b'.w) : Div st k a' b' := by
  obtain ⟨h1, h2, h3⟩ := h
  have pa := collCalls_prefix ga st
  have pb := collCalls_prefix gb st
  refine Or.inr ⟨Nat.lt_of_lt_of_le h1 pa.length_le, Nat.lt_of_lt_of_le h2 pb.length_le, ?_⟩
  rw [take_of_prefix pa (by omega), take_of_prefix pb (by omega), h3]

/-- Two runs on from states that are `Div` are `Div`, given what the runs
    from one state end as. -/
theorem div_step {st k : Nat} {a b a' b' : LI} (h : Div st k a b) (heq : a = b → Div st k a' b')
    (ga : Grows a.w a'.w) (gb : Grows b.w b'.w) : Div st k a' b' := by
  rcases h with h | h
  · exact heq h
  · exact div_grows h ga gb

theorem div_map {st k : Nat} {a b : LI} (g : LI → LI) (hg : ∀ t, (g t).w = t.w) (h : Div st k a b) :
    Div st k (g a) (g b) := by
  rcases h with rfl | ⟨h1, h2, h3⟩
  · exact Or.inl rfl
  · exact Or.inr ⟨by rw [hg]; exact h1, by rw [hg]; exact h2, by rw [hg, hg]; exact h3⟩

theorem evalL_ans (x : LCtx) (a' : List Answer) (reg : Nat → UInt64) (e : IExp) :
    e.evalL { x with ans := a' } reg = e.evalL x reg := by
  induction e <;> simp [IExp.evalL, *]

theorem idxV_ans (x : LCtx) (a' : List Answer) (m : Mem) (ix : IdxE) :
    idxV { x with ans := a' } m ix = idxV x m ix := by
  induction ix <;> simp [idxV, *]

theorem foldlM_div {st k : Nat} {f f' : LI → Nat → Option LI}
    (hf : ∀ j s s', f s j = some s' → Grows s.w s'.w) (hf' : ∀ j s s', f' s j = some s' → Grows s.w s'.w)
    (hd : ∀ j s r1 r2, f s j = some r1 → f' s j = some r2 → Div st k r1 r2) :
    ∀ (L : List Nat) (s r1 r2 : LI), L.foldlM f s = some r1 → L.foldlM f' s = some r2 → Div st k r1 r2 := by
  intro L
  induction L with
  | nil => intro s r1 r2 h1 h2; cases h1; cases h2; exact Or.inl rfl
  | cons j js ih =>
      intro s r1 r2 h1 h2
      simp only [List.foldlM_cons] at h1 h2
      obtain ⟨t1, e1, h1⟩ := Option.bind_eq_some_iff.mp h1
      obtain ⟨t2, e2, h2⟩ := Option.bind_eq_some_iff.mp h2
      exact div_step (hd j s t1 t2 e1 e2) (fun he => by subst he; exact ih t1 r1 r2 h1 h2)
        (foldlM_rel Grows Grows.refl (fun _ _ _ => Grows.trans) hf js t1 r1 h1)
        (foldlM_rel Grows Grows.refl (fun _ _ _ => Grows.trans) hf' js t2 r2 h2)

theorem whileL_div {st k : Nat} {step step' : LI → Option LI} {go : LI → Bool}
    (hs : ∀ s s', step s = some s' → Grows s.w s'.w) (hs' : ∀ s s', step' s = some s' → Grows s.w s'.w)
    (hd : ∀ s r1 r2, step s = some r1 → step' s = some r2 → Div st k r1 r2) :
    ∀ (n : Nat) (s r1 r2 : LI), whileL step go n s = some r1 → whileL step' go n s = some r2 → Div st k r1 r2 := by
  intro n
  induction n with
  | zero =>
      intro s r1 r2 h1 h2
      simp only [whileL] at h1 h2
      split at h1
      · cases h1
      · rw [if_neg (by assumption)] at h2; cases h1; cases h2; exact Or.inl rfl
  | succ n ih =>
      intro s r1 r2 h1 h2
      simp only [whileL] at h1 h2
      split at h1
      · rename_i hg
        rw [if_pos hg] at h2
        obtain ⟨t1, e1, h1⟩ := Option.bind_eq_some_iff.mp h1
        obtain ⟨t2, e2, h2⟩ := Option.bind_eq_some_iff.mp h2
        exact div_step (hd s t1 t2 e1 e2) (fun he => by subst he; exact ih t1 r1 r2 h1 h2)
          (whileL_rel Grows Grows.refl (fun _ _ _ => Grows.trans) hs n t1 r1 h1)
          (whileL_rel Grows Grows.refl (fun _ _ _ => Grows.trans) hs' n t2 r2 h2)
      · rename_i hg
        rw [if_neg hg] at h2; cases h1; cases h2; exact Or.inl rfl

/-- A collective made from one state with answers that agree at its index
    ends alike; with any answers, the runs have made the same collectives. -/
theorem div_coll {st k : Nat} {li r1 r2 : LI} (hK : k ≤ (collCalls li.w st).length → Div st k r1 r2)
    (hE : (collCalls li.w st).length < k → r1 = r2) : Div st k r1 r2 := by
  by_cases h : (collCalls li.w st).length < k
  · exact Or.inl (hE h)
  · exact hK (by omega)

theorem div_calls {st k : Nat} {li r1 r2 : LI} (c : Intr × List V)
    (h1 : collCalls r1.w st = collCalls li.w st ++ [c]) (h2 : collCalls r2.w st = collCalls li.w st ++ [c])
    (hk : k ≤ (collCalls li.w st).length) : Div st k r1 r2 :=
  Or.inr ⟨by rw [h1]; simp; omega, by rw [h2]; simp; omega, by rw [h1, h2]⟩

theorem wordAt_eq {a b : List Answer} {K : Nat} (h : a[K]? = b[K]?) : wordAt a K = wordAt b K := by
  simp only [wordAt, h]

/-- **A lane's collectives depend on its earlier answers**: two runs from one
    state whose answers agree below the `k`-th end alike, or have both made
    more than `k` collectives, the first `k + 1` the same. -/
theorem laneI_prefix (a' : List Answer) (k : Nat) :
    ∀ (s : EWStmt) (x : LCtx), (∀ j, j < k → x.ans[j]? = a'[j]?) →
      ∀ (li r1 r2 : LI), laneI x s li = some r1 → laneI { x with ans := a' } s li = some r2 →
        Div x.start k r1 r2 := by
  intro s
  induction s with
  | skip => intro x _ li r1 r2 h1 h2; cases h1; cases h2; exact Or.inl rfl
  | seq a b iha ihb =>
      intro x hag li r1 r2 h1 h2
      simp only [laneI] at h1 h2
      obtain ⟨s1, e1, h1⟩ := Option.bind_eq_some_iff.mp h1
      obtain ⟨s2, e2, h2⟩ := Option.bind_eq_some_iff.mp h2
      exact div_step (iha x hag li s1 s2 e1 e2) (fun he => by subst he; exact ihb x hag s1 r1 r2 h1 h2)
        (laneI_grows b x s1 r1 h1) (laneI_grows b _ s2 r2 h2)
  | setI d e =>
      intro x _ li r1 r2 h1 h2
      simp only [laneI, evalL_ans] at h1 h2
      rw [h1] at h2; cases h2; exact Or.inl rfl
  | ldB d a | ldH d a =>
      intro x _ li r1 r2 h1 h2
      simp only [laneI, evalL_ans] at h1 h2
      rw [h1] at h2; cases h2; exact Or.inl rfl
  | stB a v g | stH a v g =>
      intro x _ li r1 r2 h1 h2
      simp only [laneI, evalL_ans] at h1 h2
      rw [h1] at h2; cases h2; exact Or.inl rfl
  | setR d e =>
      intro x _ li r1 r2 h1 h2
      simp only [laneI] at h1 h2
      rw [h1] at h2; cases h2; exact Or.inl rfl
  | loadIdx d b ix | loadV4 d0 d1 d2 d3 b ix | storeLane b ix d | storeLane0 b ix d =>
      intro x _ li r1 r2 h1 h2
      simp only [laneI, idxV_ans] at h1 h2
      rw [h1] at h2; cases h2; exact Or.inl rfl
  | ballot d p =>
      intro x hag li r1 r2 h1 h2
      simp only [laneI, evalL_ans] at h1 h2
      split at h1
      · rename_i hs
        rw [if_pos hs] at h2
        cases h1; cases h2
        refine div_coll (li := li) (fun hk => div_calls (.ballot, [boolV (p.evalL x li.reg != 0)]) ?_ ?_ hk)
          (fun hk => by simp only [ansW, wordAt_eq (hag _ hk)])
        · exact (collCalls_call hs _ _).trans (by simp [Intr.isColl])
        · exact (collCalls_call hs _ _).trans (by simp [Intr.isColl])
      · cases h1
  | shflIdx d v sr =>
      intro x hag li r1 r2 h1 h2
      simp only [laneI, evalL_ans] at h1 h2
      split at h1
      · rename_i hs
        rw [if_pos hs] at h2
        cases h1; cases h2
        refine div_coll (li := li) (fun hk => div_calls (.shflIdx, [.sc .i32 (v.evalL x li.reg &&& 0xffffffff),
          .sc .i32 (sr.evalL x li.reg &&& 0xffffffff)]) ?_ ?_ hk)
          (fun hk => by simp only [ansW, wordAt_eq (hag _ hk)])
        · exact (collCalls_call hs _ _).trans (by simp [Intr.isColl])
        · exact (collCalls_call hs _ _).trans (by simp [Intr.isColl])
      · cases h1
  | shflXor d sr m =>
      intro x hag li r1 r2 h1 h2
      simp only [laneI] at h1 h2
      split at h1
      · rename_i hs
        rw [if_pos hs] at h2
        cases h1; cases h2
        refine div_coll (li := li) (fun hk => div_calls (.shflBfly, [.sc .i32 (li.f sr).bits,
          .sc .i32 (s32 (UInt32.ofNat m))]) ?_ ?_ hk)
          (fun hk => by simp only [ansW, wordAt_eq (hag _ hk)])
        · exact (collCalls_call hs.2 _ _).trans (by simp [Intr.isColl])
        · exact (collCalls_call hs.2 _ _).trans (by simp [Intr.isColl])
      · cases h1
  | barrier =>
      intro x hag li r1 r2 h1 h2
      simp only [laneI] at h1 h2
      split at h1
      · rename_i hs
        rw [if_pos hs] at h2
        refine div_coll (li := li) (fun hk => ?_) (fun hk => ?_)
        · split at h1
          · split at h2
            · cases h1; cases h2
              exact div_calls (.barrier, []) ((collCalls_call hs _ _).trans (by simp [Intr.isColl]))
                ((collCalls_call hs _ _).trans (by simp [Intr.isColl])) hk
            · cases h2
          · cases h1
        · rw [← hag _ hk] at h2
          rw [h1] at h2; cases h2; rfl
      · cases h1
  | iif c a b iha ihb =>
      intro x hag li r1 r2 h1 h2
      simp only [laneI, evalL_ans] at h1 h2
      split at h1
      · rename_i hc; rw [if_pos hc] at h2; exact iha x hag li r1 r2 h1 h2
      · rename_i hc; rw [if_neg hc] at h2; exact ihb x hag li r1 r2 h1 h2
  | iwhile n c cs body ih =>
      intro x hag li r1 r2 h1 h2
      simp only [laneI, evalL_ans] at h1 h2
      refine whileL_div (fun s s' hs => ?_) (fun s s' hs => ?_) (fun s t1 t2 e1 e2 => ?_) n li r1 r2 h1 h2
      · obtain ⟨s1, e, hs⟩ := Option.map_eq_some_iff.mp hs
        cases hs; exact laneI_grows body x s s1 e
      · obtain ⟨s1, e, hs⟩ := Option.map_eq_some_iff.mp hs
        cases hs; exact laneI_grows body _ s s1 e
      · obtain ⟨u1, f1, e1⟩ := Option.map_eq_some_iff.mp e1
        obtain ⟨u2, f2, e2⟩ := Option.map_eq_some_iff.mp e2
        subst e1 e2
        exact div_map (keepL cs li) (fun _ => rfl) (ih x hag s u1 u2 f1 f2)
  | locals rs body ih =>
      intro x hag li r1 r2 h1 h2
      simp only [laneI] at h1 h2
      obtain ⟨u1, f1, e1⟩ := Option.map_eq_some_iff.mp h1
      obtain ⟨u2, f2, e2⟩ := Option.map_eq_some_iff.mp h2
      subst e1 e2
      exact div_map (restoreL rs li) (fun _ => rfl) (ih x hag li u1 u2 f1 f2)
  | forN n body ih =>
      intro x hag li r1 r2 h1 h2
      simp only [laneI] at h1 h2
      split at h1
      · rename_i hn
        rw [if_pos hn] at h2
        exact foldlM_div (fun j s s' hs => laneI_grows body _ s s' hs) (fun j s s' hs => laneI_grows body _ s s' hs)
          (fun j s t1 t2 e1 e2 => ih { x with i := j } hag s t1 t2 e1 e2) _ li r1 r2 h1 h2
      · cases h1
  | forM b a body ih =>
      intro x hag li r1 r2 h1 h2
      simp only [laneI] at h1 h2
      split at h1
      · rename_i ha
        rw [if_pos ha] at h2
        obtain ⟨v, ev, h1⟩ := Option.bind_eq_some_iff.mp h1
        obtain ⟨v', ev', h2⟩ := Option.bind_eq_some_iff.mp h2
        rw [ev] at ev'; cases ev'
        split at h1
        · rename_i hv
          rw [if_pos hv] at h2
          exact foldlM_div (fun j s s' hs => laneI_grows body _ s s' hs) (fun j s s' hs => laneI_grows body _ s s' hs)
            (fun j s t1 t2 e1 e2 => ih { x with i := j } hag s t1 t2 e1 e2) _ li r1 r2 h1 h2
        · cases h1
      · cases h1
  | stSm | ldSm | cvtIF => intro x _ li r1 r2 h1; simp [laneI] at h1

/-! ## The warp's collectives, leaf by leaf -/

open LaneWarp

section Traces
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat)

theorem trace_foldl_map {α β : Type} (leaf₁ : Nat → EWStmt → WSt F32 → List α) (leaf₂ : Nat → EWStmt → WSt F32 → List β)
    (g : α → β) (b : EWStmt) (hb : ∀ j ws, trace hw cta im leaf₂ j b ws = (trace hw cta im leaf₁ j b ws).map g) :
    ∀ (L : List Nat) (p : List α × WSt F32),
      L.foldl (fun (q : List β × WSt F32) j => (q.1 ++ trace hw cta im leaf₂ j b q.2, run hw cta im b q.2 j))
          (p.1.map g, p.2) =
        ((L.foldl (fun (q : List α × WSt F32) j => (q.1 ++ trace hw cta im leaf₁ j b q.2, run hw cta im b q.2 j)) p).1.map g,
         (L.foldl (fun (q : List α × WSt F32) j => (q.1 ++ trace hw cta im leaf₁ j b q.2, run hw cta im b q.2 j)) p).2)
  | [], _ => rfl
  | j :: L, p => by
      simp only [List.foldl_cons]
      rw [hb j p.2, ← List.map_append]
      exact trace_foldl_map leaf₁ leaf₂ g b hb L (p.1 ++ trace hw cta im leaf₁ j b p.2, run hw cta im b p.2 j)

theorem trace_while_map {α β : Type} (leaf₁ : Nat → EWStmt → WSt F32 → List α) (leaf₂ : Nat → EWStmt → WSt F32 → List β)
    (g : α → β) (i : Nat) (c : IExp) (cs : List Nat) (b : EWStmt) (ws0 : WSt F32)
    (hb : ∀ ws, trace hw cta im leaf₂ i b ws = (trace hw cta im leaf₁ i b ws).map g) :
    ∀ (L : List Nat) (p : List α × WSt F32),
      L.foldl (fun (q : List β × WSt F32) _ =>
          if c.eval q.2.int cta lane0 ≠ 0 then
            (q.1 ++ trace hw cta im leaf₂ i b q.2,
              (run hw cta im b q.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b q.2 i).int))
          else q) (p.1.map g, p.2) =
        ((L.foldl (fun (q : List α × WSt F32) _ =>
          if c.eval q.2.int cta lane0 ≠ 0 then
            (q.1 ++ trace hw cta im leaf₁ i b q.2,
              (run hw cta im b q.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b q.2 i).int))
          else q) p).1.map g,
         (L.foldl (fun (q : List α × WSt F32) _ =>
          if c.eval q.2.int cta lane0 ≠ 0 then
            (q.1 ++ trace hw cta im leaf₁ i b q.2,
              (run hw cta im b q.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b q.2 i).int))
          else q) p).2)
  | [], _ => rfl
  | j :: L, p => by
      simp only [List.foldl_cons]
      by_cases hc : c.eval p.2.int cta lane0 ≠ 0
      · simp only [if_pos hc]
        rw [hb p.2, ← List.map_append]
        exact trace_while_map leaf₁ leaf₂ g i c cs b ws0 hb L (p.1 ++ trace hw cta im leaf₁ i b p.2,
          (run hw cta im b p.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b p.2 i).int))
      · simp only [if_neg hc]
        exact trace_while_map leaf₁ leaf₂ g i c cs b ws0 hb L p

/-- A trace of leaves that are another's mapped is that trace mapped. -/
theorem trace_map {α β : Type} (leaf₁ : Nat → EWStmt → WSt F32 → List α) (leaf₂ : Nat → EWStmt → WSt F32 → List β)
    (g : α → β) (h : ∀ i s ws, leaf₂ i s ws = (leaf₁ i s ws).map g) :
    ∀ (s : EWStmt) (i : Nat) (ws : WSt F32), trace hw cta im leaf₂ i s ws = (trace hw cta im leaf₁ i s ws).map g := by
  intro s
  induction s with
  | seq a b iha ihb => intro i ws; simp only [trace, List.map_append, iha, ihb]
  | iif c a b iha ihb =>
      intro i ws; simp only [trace]
      split
      · exact iha i ws
      · exact ihb i ws
  | forN n b ih =>
      intro i ws
      simp only [trace]
      exact congrArg Prod.fst (trace_foldl_map hw cta im leaf₁ leaf₂ g b (fun j ws => ih j ws) _ ([], ws))
  | forM bu a b ih =>
      intro i ws
      simp only [trace]
      exact congrArg Prod.fst (trace_foldl_map hw cta im leaf₁ leaf₂ g b (fun j ws => ih j ws) _ ([], ws))
  | iwhile n c cs b ih =>
      intro i ws
      simp only [trace]
      exact congrArg Prod.fst (trace_while_map hw cta im leaf₁ leaf₂ g i c cs b ws (fun ws => ih i ws) _ ([], ws))
  | locals rs b ih => intro i ws; simp only [trace]; exact ih i ws
  | _ => intro i ws; simp only [trace]; exact h _ _ _

/-- **Every leaf of a statement meets `Q`**, at the trip it runs at. -/
def AllAt (Q : Nat → EWStmt → Prop) : Nat → EWStmt → Prop
  | i, .seq a b => AllAt Q i a ∧ AllAt Q i b
  | i, .iif _ a b => AllAt Q i a ∧ AllAt Q i b
  | _, .forN n b => ∀ j, j < n → AllAt Q j b
  | _, .forM bu a b => ∀ j, j < im bu a → AllAt Q j b
  | i, .iwhile _ _ _ b => AllAt Q i b
  | i, .locals _ b => AllAt Q i b
  | i, s => Q i s

theorem foldl_mem {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (b : EWStmt) (e : α) :
    ∀ (L : List Nat) (p : List α × WSt F32),
      e ∈ (L.foldl (fun (q : List α × WSt F32) j => (q.1 ++ trace hw cta im leaf j b q.2, run hw cta im b q.2 j)) p).1 →
      e ∈ p.1 ∨ ∃ j ∈ L, ∃ ws, e ∈ trace hw cta im leaf j b ws
  | [], _, h => Or.inl h
  | j :: L, p, h => by
      simp only [List.foldl_cons] at h
      rcases foldl_mem leaf b e L _ h with h | ⟨j', hj', ws, h⟩
      · rcases List.mem_append.mp h with h | h
        · exact Or.inl h
        · exact Or.inr ⟨j, List.mem_cons_self, _, h⟩
      · exact Or.inr ⟨j', List.mem_cons_of_mem _ hj', ws, h⟩

theorem while_mem {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (i : Nat) (c : IExp) (cs : List Nat)
    (b : EWStmt) (ws0 : WSt F32) (e : α) :
    ∀ (L : List Nat) (p : List α × WSt F32),
      e ∈ (L.foldl (fun (q : List α × WSt F32) _ =>
          if c.eval q.2.int cta lane0 ≠ 0 then
            (q.1 ++ trace hw cta im leaf i b q.2,
              (run hw cta im b q.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b q.2 i).int))
          else q) p).1 →
      e ∈ p.1 ∨ ∃ ws, e ∈ trace hw cta im leaf i b ws
  | [], _, h => Or.inl h
  | j :: L, p, h => by
      simp only [List.foldl_cons] at h
      rcases while_mem leaf i c cs b ws0 e L _ h with h | h
      · split at h
        · rcases List.mem_append.mp h with h | h
          · exact Or.inl h
          · exact Or.inr ⟨_, h⟩
        · exact Or.inl h
      · exact Or.inr h

/-- What a trace holds, some leaf meeting `Q` gives. -/
theorem mem_trace {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (Q : Nat → EWStmt → Prop) (e : α) :
    ∀ (s : EWStmt) (i : Nat) (ws : WSt F32), AllAt im Q i s → e ∈ trace hw cta im leaf i s ws →
      ∃ i' s' ws', Q i' s' ∧ e ∈ leaf i' s' ws' := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro i ws hQ h
      simp only [trace] at h
      rcases List.mem_append.mp h with h | h
      · exact iha i ws hQ.1 h
      · exact ihb i _ hQ.2 h
  | iif c a b iha ihb =>
      intro i ws hQ h
      simp only [trace] at h
      split at h
      · exact iha i ws hQ.1 h
      · exact ihb i ws hQ.2 h
  | forN n b ih =>
      intro i ws hQ h
      simp only [trace] at h
      rcases foldl_mem hw cta im leaf b e _ ([], ws) h with h | ⟨j, hj, ws', h⟩
      · cases h
      · exact ih j ws' (hQ j (List.mem_range.mp hj)) h
  | forM bu a b ih =>
      intro i ws hQ h
      simp only [trace] at h
      rcases foldl_mem hw cta im leaf b e _ ([], ws) h with h | ⟨j, hj, ws', h⟩
      · cases h
      · exact ih j ws' (hQ j (List.mem_range.mp hj)) h
  | iwhile n c cs b ih =>
      intro i ws hQ h
      simp only [trace] at h
      rcases while_mem hw cta im leaf i c cs b ws e _ ([], ws) h with h | ⟨ws', h⟩
      · cases h
      · exact ih i ws' hQ h
  | locals rs b ih => intro i ws hQ h; simp only [trace] at h; exact ih i ws hQ h
  | _ => intro i ws hQ h; simp only [trace] at h; exact ⟨_, _, _, hQ, h⟩

/-- The collective leaves, each where the warp's run reaches it. -/
def jLeaf : Nat → EWStmt → WSt F32 → List (Nat × EWStmt × WSt F32)
  | i, .ballot d p, ws => [(i, .ballot d p, ws)]
  | i, .shflIdx d v sr, ws => [(i, .shflIdx d v sr, ws)]
  | i, .shflXor d sr m, ws => [(i, .shflXor d sr m, ws)]
  | i, .barrier, ws => [(i, .barrier, ws)]
  | _, _, _ => []

/-- The collective lane `l` makes at a collective leaf. -/
def callOne (l : Lane) (e : Nat × EWStmt × WSt F32) : Intr × List V :=
  (callLeaf cta l e.1 e.2.1 e.2.2).headD (.barrier, [])

/-- The answer lane `l` gets at a collective leaf. -/
def ansOne (S : Nat) (l : Lane) (e : Nat × EWStmt × WSt F32) : Answer :=
  (ansLeaf cta S l e.1 e.2.1 e.2.2).headD (.smem .empty)

theorem trace_call (l : Lane) (s : EWStmt) (i : Nat) (ws : WSt F32) :
    trace hw cta im (callLeaf cta l) i s ws = (trace hw cta im jLeaf i s ws).map (callOne cta l) :=
  trace_map hw cta im _ _ _ (fun i s ws => by cases s <;> rfl) s i ws

theorem trace_ans (S : Nat) (l : Lane) (s : EWStmt) (i : Nat) (ws : WSt F32) :
    trace hw cta im (ansLeaf cta S l) i s ws = (trace hw cta im jLeaf i s ws).map (ansOne cta S l) :=
  trace_map hw cta im _ _ _ (fun i s ws => by cases s <;> rfl) s i ws

end Traces

/-! ## Lanes of a warp settle on the warp's answers -/

section Settle
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (S : Nat) (MF : Lane → Buf → Nat → Prop)
  (MB : Lane → UInt64 → Prop) (MI : Buf → Nat → Prop) (WS : Lane → UInt64 → Nat → Prop)

theorem relW_ans {l : Lane} {x : LCtx} {a : List Answer} {ws : WSt F32} {cl : Bool} {li : LI}
    (h : RelW hw S MF MB MI WS im l { x with ans := a } ws cl li) : RelW hw S MF MB MI WS im l x ws cl li :=
  ⟨h.reg, h.f, h.fmem, h.bmem, h.imem, h.wr, h.asz, h.sm, h.nan, h.start, h.ptr, h.w32, by first | exact h.lent | exact (Mem.store_lent (by assumption)).trans h.lent | exact (Contracts.lanes_lent _ _ (fun _ _ _ h => Mem.store_lent h) (by assumption)).trans h.lent⟩

theorem prefix_get {α : Type} {l₁ l₂ : List α} (h : l₁ <+: l₂) {k : Nat} {x : α} (hk : l₁[k]? = some x) :
    l₂[k]? = some x := by
  obtain ⟨t, rfl⟩ := h
  obtain ⟨hk', _⟩ := List.getElem?_eq_some_iff.mp hk
  rw [List.getElem?_append_left hk']; exact hk

/-- **The lanes of a warp settle on the warp's answers**: where each lane's
    run of a statement ends, its collectives the start of `F`, and the laws
    give a lane the warp's answer at a collective where every lane's is the
    warp's (`hlaw`), each lane's run is the one `lane_warp` gives from the
    warp's answers: it ends with the warp. -/
theorem lane_settle (hw32 : ∀ k v, (hw.approx k v).toNat < 2 ^ 32) (s : EWStmt) (i : Nat) (ws : WSt F32)
    (cl : Bool) (x : Lane → LCtx) (li r : Lane → LI) (F : Lane → List (Intr × List V)) (c0 : Nat)
    (hrun : ∀ l, laneI (x l) s (li l) = some (r l))
    (hc0 : ∀ l, (collCalls (li l).w (x l).start).length = c0)
    (hlen : ∀ l, c0 ≤ (x l).ans.length)
    (hpre : ∀ l, LanePre hw cta im S MF MB MI WS l
      { x l with ans := (x l).ans.take c0 ++ trace hw cta im (ansLeaf cta S l) i s ws } i s ws cl (li l) [])
    (hF : ∀ l, collCalls (r l).w (x l).start <+: F l)
    (hlaw : ∀ j, j < (trace hw cta im jLeaf i s ws).length →
      (∀ l, (F l)[c0 + j]? = (trace hw cta im (callLeaf cta l) i s ws)[j]?) →
      ∀ l, (x l).ans[c0 + j]? = (trace hw cta im (ansLeaf cta S l) i s ws)[j]?) :
    ∀ l, LaneEnds hw cta im S MF MB MI WS l (x l) i s ws cl (li l)
      ((x l).ans.drop (c0 + (trace hw cta im jLeaf i s ws).length)) := by
  let n := (trace hw cta im jLeaf i s ws).length
  let T (l : Lane) : List Answer := (x l).ans.take c0 ++ trace hw cta im (ansLeaf cta S l) i s ws
  have hW := fun l => lane_warp hw cta im S MF MB MI WS l hw32 s _ i ws cl (li l) [] (hpre l)
  let tl (l : Lane) : LI := (hW l).choose
  have htl : ∀ l, laneI { x l with ans := T l } s (li l) = some (tl l) := fun l => (hW l).choose_spec.1
  have hrel := fun l => (hW l).choose_spec.2.1
  have hcalls := fun l => (hW l).choose_spec.2.2.1
  have hstores := fun l => (hW l).choose_spec.2.2.2.1
  have hlc : ∀ l, (trace hw cta im (callLeaf cta l) i s ws).length = n := fun l => by
    rw [trace_call]; exact List.length_map _
  have hla : ∀ l, (trace hw cta im (ansLeaf cta S l) i s ws).length = n := fun l => by
    rw [trace_ans]; exact List.length_map _
  have htk : ∀ l, ((x l).ans.take c0).length = c0 := fun l => by simp; exact Nat.min_eq_left (hlen l)
  have hTlo : ∀ l j, j < c0 → (T l)[j]? = (x l).ans[j]? := fun l j hj => by
    simp only [T]; rw [List.getElem?_append_left (by rw [htk]; exact hj), List.getElem?_take_of_lt hj]
  have hThi : ∀ l j, (T l)[c0 + j]? = (trace hw cta im (ansLeaf cta S l) i s ws)[j]? := fun l j => by
    simp only [T]; rw [List.getElem?_append_right (by rw [htk]; omega), htk]; simp
  have hct : ∀ l, collCalls (tl l).w (x l).start = collCalls (li l).w (x l).start ++
      trace hw cta im (callLeaf cta l) i s ws := hcalls
  -- the k-th collective of each lane is the warp's, once the answers below it are
  have hcall : ∀ k, k < n → (∀ l j, j < c0 + k → (x l).ans[j]? = (T l)[j]?) →
      ∀ l, (F l)[c0 + k]? = (trace hw cta im (callLeaf cta l) i s ws)[k]? := by
    intro k hk hag l
    have hd := laneI_prefix (T l) (c0 + k) s (x l) (hag l) (li l) (r l) (tl l) (hrun l) (htl l)
    obtain ⟨cv, hcv⟩ : ∃ cv, (trace hw cta im (callLeaf cta l) i s ws)[k]? = some cv :=
      ⟨_, List.getElem?_eq_getElem (by rw [hlc]; exact hk)⟩
    have htlk : (collCalls (tl l).w (x l).start)[c0 + k]? = some cv := by
      rw [hct, List.getElem?_append_right (by rw [hc0]; omega), hc0]; simpa using hcv
    rw [hcv]
    rcases hd with he | ⟨h1, h2, h3⟩
    · have hFl := hF l
      rw [he] at hFl; exact prefix_get hFl htlk
    · have : (collCalls (r l).w (x l).start)[c0 + k]? = some cv := by
        have e1 := congrArg (·[c0 + k]?) h3
        simp only [List.getElem?_take, if_pos (show c0 + k < c0 + k + 1 by omega)] at e1
        rw [e1]; exact htlk
      exact prefix_get (hF l) this
  have hagree : ∀ k, k ≤ n → ∀ l j, j < c0 + k → (x l).ans[j]? = (T l)[j]? := by
    intro k
    induction k with
    | zero => intro _ l j hj; exact (hTlo l j hj).symm
    | succ k ih =>
        intro hk l j hj
        by_cases hjk : j < c0 + k
        · exact ih (by omega) l j hjk
        · have hj' : j = c0 + k := by omega
          subst hj'
          rw [hThi l k]
          exact hlaw k (by omega) (hcall k (by omega) (ih (by omega))) l
  intro l
  have hd := laneI_prefix (T l) (c0 + n) s (x l) (hagree n (Nat.le_refl _) l) (li l) (r l) (tl l) (hrun l) (htl l)
  have hlen_tl : (collCalls (tl l).w (x l).start).length = c0 + n := by rw [hct, List.length_append, hc0, hlc]
  have hrt : r l = tl l := by
    rcases hd with he | ⟨_, h2, _⟩
    · exact he
    · omega
  refine ⟨tl l, by rw [← hrt]; exact hrun l, relW_ans hw im S MF MB MI WS (hrel l), hcalls l, hstores l, ?_⟩
  rw [hlen_tl]

end Settle

/-! ## The butterfly law -/

/-- A statement whose collectives are butterfly shuffles, masks below `2^31`. -/
def BflyOnly : Nat → EWStmt → Prop
  | _, .ballot .. | _, .shflIdx .. | _, .barrier => False
  | _, .shflXor _ _ m => m < 2 ^ 31
  | _, _ => True

theorem jLeaf_mem {i : Nat} {s : EWStmt} {ws : WSt F32} {e : Nat × EWStmt × WSt F32} (h : e ∈ jLeaf i s ws) :
    e = (i, s, ws) := by
  cases s <;> simp [jLeaf] at h <;> exact h

/-- **The laws answer butterfly shuffles as the warp does**: where every lane
    of warp `wi` has made the warp's collective at index `c0 + j`, a
    butterfly shuffle, each lane is answered the warp's answer. -/
theorem law_bfly (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (S : Nat)
    {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    (A : Lane → List Answer) (wi : Nat) (hA : ∀ l : Lane, answersOf calls snaps smem0 (32 * wi + l.val) = .ok (A l))
    (s : EWStmt) (i : Nat) (ws : WSt F32) (hQ : AllAt im BflyOnly i s) (c0 j : Nat)
    (hj : j < (trace hw cta im jLeaf i s ws).length)
    (hall : ∀ l : Lane, (calls (32 * wi + l.val))[c0 + j]? = (trace hw cta im (callLeaf cta l) i s ws)[j]?) :
    ∀ l : Lane, (A l)[c0 + j]? = (trace hw cta im (ansLeaf cta S l) i s ws)[j]? := by
  intro l
  obtain ⟨e, he⟩ : ∃ e, (trace hw cta im jLeaf i s ws)[j]? = some e := ⟨_, List.getElem?_eq_getElem hj⟩
  obtain ⟨i', s', ws', hq, hm⟩ := mem_trace hw cta im jLeaf BflyOnly e s i ws hQ (List.mem_of_getElem? he)
  have he' := jLeaf_mem hm
  subst he'
  have hc : ∀ l : Lane, (calls (32 * wi + l.val))[c0 + j]? = some (callOne cta l (i', s', ws')) := fun l => by
    rw [hall l, trace_call, List.getElem?_map, he]; rfl
  rw [trace_ans, List.getElem?_map, he]
  cases s' with
  | shflXor d sr m =>
      have hm31 : m < 2 ^ 31 := hq
      have hk := hc l
      simp only [callOne, callLeaf, List.headD_cons] at hk
      obtain ⟨args, hl, hans⟩ := answersOf_bfly (hA l) hk
      have hlt : l.val < 32 := l.isLt
      have h32 : (32 * wi + l.val) / 32 = wi := by omega
      have hl32 : (32 * wi + l.val) % 32 = l.val := by omega
      have hmn : (s32 (UInt32.ofNat m)).toNat = m := by
        simp only [s32, UInt32.toNat_toUInt64]; exact toNat_ofNat32 (by omega)
      rw [h32, hl32, hmn] at hl
      let P : Lane := xorLane l m
      have hP := hc P
      simp only [callOne, callLeaf, List.headD_cons] at hP
      unfold laneArg at hl
      rw [show 32 * wi + (l.val ^^^ m) % 32 = 32 * wi + P.val from rfl, hP] at hl
      simp only at hl
      cases hl
      rw [hans]
      simp [ansOne, ansLeaf, lane32, P, s32]
  | ballot | shflIdx | barrier => exact hq.elim
  | _ => simp [jLeaf] at hm

end AlgorithmLib.ML.LaneSettle
