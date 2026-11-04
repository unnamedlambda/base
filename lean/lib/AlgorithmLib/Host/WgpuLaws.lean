module
public import AlgorithmLib.Host.Wgpu
meta import AlgorithmLib.Host.Wgpu
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# What a wgpu step keeps

The laws the checker's wgpu rows are proven by. A shape read in a state
(`WgState.Fits`) depends on the object's own data and on which objects of the
kinds it reads are live (`Fits_congr`), so a step is described by what it does
to those: `WgStep s s' TK K g` — every object whose kind is not in `TK` keeps
its liveness and its shape (`WgData.Same`), every kind not in `K` keeps its
live objects (`KindSame`), and `g` objects are added. Steps compose
(`WgStep.trans`), and each of wgpu's own changes is one (`WgStep.set`,
`WgStep.free`, `WgStep.push`, and a submit's run, `WgStep.run`).
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR

/-- Object `j` is live and of kind `k`. -/
def WgState.LiveAt (s : WgState) (k : WgObj) (j : Nat) : Prop :=
  ∃ d, s.objs[j]? = some (true, d) ∧ d.kind = k

theorem WgState.isOne_iff {s : WgState} {k : WgObj} {i : Nat} :
    s.isOne k i = true ↔ s.LiveAt k i ∧ ∀ j, s.LiveAt k j → j = i := by
  rw [WgState.isOne_spec]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨h1, fun j ⟨d, hd, hk⟩ => h2 j d hd hk⟩
  · rintro ⟨h1, h2⟩; exact ⟨h1, fun j d hd hk => h2 j ⟨d, hd, hk⟩⟩

theorem WgState.noneOf_iff {s : WgState} {k : WgObj} : s.NoneOf k ↔ ∀ j, ¬ s.LiveAt k j := by
  constructor
  · rintro h j ⟨d, hd, hk⟩; exact h j d hd hk
  · intro h j d hd hk; exact h j ⟨d, hd, hk⟩

/-- `s'` has live objects of kind `k` exactly where `s` has. -/
def KindSame (s s' : WgState) (k : WgObj) : Prop := ∀ j, s.LiveAt k j ↔ s'.LiveAt k j

theorem KindSame.refl (s : WgState) (k : WgObj) : KindSame s s k := fun _ => Iff.rfl

theorem KindSame.trans {s₁ s₂ s₃ : WgState} {k : WgObj} (h₁ : KindSame s₁ s₂ k) (h₂ : KindSame s₂ s₃ k) :
    KindSame s₁ s₃ k := fun j => (h₁ j).trans (h₂ j)

theorem KindSame.isOne {s s' : WgState} {k : WgObj} (h : KindSame s s' k) (i : Nat) :
    s.isOne k i = s'.isOne k i := by
  apply Bool.eq_iff_iff.mpr
  rw [WgState.isOne_iff, WgState.isOne_iff]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨(h i).mp h1, fun j hj => h2 j ((h j).mpr hj)⟩
  · rintro ⟨h1, h2⟩; exact ⟨(h i).mpr h1, fun j hj => h2 j ((h j).mp hj)⟩

theorem KindSame.noneOf {s s' : WgState} {k : WgObj} (h : KindSame s s' k) : s.NoneOf k ↔ s'.NoneOf k := by
  rw [WgState.noneOf_iff, WgState.noneOf_iff]
  exact ⟨fun h1 j hj => h1 j ((h j).mpr hj), fun h1 j hj => h1 j ((h j).mp hj)⟩

theorem WgState.Fits.kind {s : WgState} {d : WgData} {sh : WgShape} (h : s.Fits d sh) : d.kind = sh.kind := by
  cases d <;> cases sh <;> first | rfl | exact h.elim

/-- A shape read in two states that agree on the live objects of the kinds it
    reads. -/
theorem WgState.Fits.congr {s s' : WgState} {d : WgData} {sh : WgShape}
    (h : ∀ k ∈ sh.reads, KindSame s s' k) : s.Fits d sh ↔ s'.Fits d sh := by
  cases d <;> cases sh <;> try exact Iff.rfl
  case cpass.cpass =>
    simp only [WgState.Fits]
    rw [(h .commandEncoder (by simp [WgShape.reads])).isOne]
  case surface.surface c cur c' =>
    simp only [WgState.Fits]
    have ht := h .texture (by simp [WgShape.reads])
    cases cur with
    | none => simp only [ht.noneOf]
    | some x => simp only [ht.isOne]
  case view.view t =>
    simp only [WgState.Fits]
    rw [(h .texture (by simp [WgShape.reads])).isOne]
  case rpass.rpass =>
    simp only [WgState.Fits]
    rw [(h .commandEncoder (by simp [WgShape.reads])).isOne, (h .textureView (by simp [WgShape.reads])).isOne]

/-- Data of the same kind, of every shape the other is of. -/
def WgData.Same (d d' : WgData) : Prop := d.kind = d'.kind ∧ ∀ (s : WgState) sh, s.Fits d sh ↔ s.Fits d' sh

theorem WgData.Same.refl (d : WgData) : d.Same d := ⟨rfl, fun _ _ => Iff.rfl⟩

theorem WgData.Same.trans {a b c : WgData} (h₁ : a.Same b) (h₂ : b.Same c) : a.Same c :=
  ⟨h₁.1.trans h₂.1, fun s sh => (h₁.2 s sh).trans (h₂.2 s sh)⟩

/-- **What a step does to wgpu's objects**: every object not at an index in
    `Ti` and whose kind is not in `TK` keeps its liveness and its shape, an
    object at an index in `Ti` is of a kind in `TK` or `K`, every kind not in
    `K` keeps its live objects, no object changes kind, and `g` objects are
    added. -/
structure WgStep (s s' : WgState) (Ti : List Nat) (TK K : List WgObj) (g : Nat) : Prop where
  same : ∀ (j : Nat) (l : Bool) (d : WgData), s.objs[j]? = some (l, d) → j ∉ Ti → d.kind ∉ TK →
    ∃ d', s'.objs[j]? = some (l, d') ∧ d.Same d'
  touched : ∀ j ∈ Ti, ∀ (l : Bool) (d : WgData), s.objs[j]? = some (l, d) → d.kind ∈ TK ∨ d.kind ∈ K
  kindKeep : ∀ (j : Nat) (l l' : Bool) (d d' : WgData), s.objs[j]? = some (l, d) → s'.objs[j]? = some (l', d') →
    d.kind = d'.kind
  kinds : ∀ k, k ∉ K → KindSame s s' k
  size : s'.objs.size = s.objs.size + g
  lost : s'.lost = s.lost

theorem WgStep.refl (s : WgState) : WgStep s s [] [] [] 0 where
  same _ _ d h _ _ := ⟨d, h, .refl d⟩
  touched _ h := nomatch h
  kindKeep _ _ _ _ _ h h' := by rw [h] at h'; cases h'; rfl
  kinds k _ := .refl s k
  size := rfl
  lost := rfl

theorem WgStep.trans {s₁ s₂ s₃ : WgState} {Ti₁ Ti₂ : List Nat} {TK₁ TK₂ K₁ K₂ : List WgObj} {g₁ g₂ : Nat}
    (h₁ : WgStep s₁ s₂ Ti₁ TK₁ K₁ g₁) (h₂ : WgStep s₂ s₃ Ti₂ TK₂ K₂ g₂) :
    WgStep s₁ s₃ (Ti₁ ++ Ti₂) (TK₁ ++ TK₂) (K₁ ++ K₂) (g₁ + g₂) where
  same j l d hj hi hk := by
    simp only [List.mem_append, not_or] at hk hi
    obtain ⟨d', h', hs⟩ := h₁.same j l d hj hi.1 hk.1
    obtain ⟨d'', h'', hs'⟩ := h₂.same j l d' h' hi.2 (by rw [← hs.1]; exact hk.2)
    exact ⟨d'', h'', hs.trans hs'⟩
  touched j hj l d hd := by
    rcases List.mem_append.mp hj with hj | hj
    · rcases h₁.touched j hj l d hd with h | h
      · exact .inl (List.mem_append_left _ h)
      · exact .inr (List.mem_append_left _ h)
    · have hlt : j < s₁.objs.size := (Array.getElem?_eq_some_iff.mp hd).1
      obtain ⟨p, hp⟩ : ∃ p, s₂.objs[j]? = some p :=
        ⟨s₂.objs[j]'(by rw [h₁.size]; omega), Array.getElem?_eq_getElem _⟩
      obtain ⟨l', d'⟩ := p
      have ek := h₁.kindKeep j l l' d d' hd hp
      rcases h₂.touched j hj l' d' hp with h | h
      · exact .inl (List.mem_append_right _ (ek ▸ h))
      · exact .inr (List.mem_append_right _ (ek ▸ h))
  kindKeep j l l' d d' h h' := by
    have hlt : j < s₁.objs.size := (Array.getElem?_eq_some_iff.mp h).1
    obtain ⟨p, hp⟩ : ∃ p, s₂.objs[j]? = some p :=
      ⟨s₂.objs[j]'(by rw [h₁.size]; omega), Array.getElem?_eq_getElem _⟩
    obtain ⟨l2, d2⟩ := p
    exact (h₁.kindKeep j l l2 d d2 h hp).trans (h₂.kindKeep j l2 l' d2 d' hp h')
  kinds k hk := by
    simp only [List.mem_append, not_or] at hk
    exact (h₁.kinds k hk.1).trans (h₂.kinds k hk.2)
  size := by rw [h₂.size, h₁.size, Nat.add_assoc]
  lost := h₂.lost.trans h₁.lost

theorem WgStep.weaken {s s' : WgState} {Ti : List Nat} {TK K TK' K' : List WgObj} {g : Nat}
    (h : WgStep s s' Ti TK K g) (hT : ∀ k ∈ TK, k ∈ TK') (hK : ∀ k ∈ K, k ∈ K') :
    WgStep s s' Ti TK' K' g where
  same j l d hj hi hk := h.same j l d hj hi (fun e => hk (hT _ e))
  touched j hj l d hd := (h.touched j hj l d hd).imp (hT _) (hK _)
  kindKeep := h.kindKeep
  kinds k hk := h.kinds k (fun e => hk (hK _ e))
  size := h.size
  lost := h.lost

/-- A touched index named by its object's kind instead. -/
theorem WgStep.forgetIdx {s s' : WgState} {i : Nat} {Ti : List Nat} {TK K : List WgObj} {g : Nat} {k : WgObj}
    (h : WgStep s s' (i :: Ti) TK K g) (hk : ∀ l d, s.objs[i]? = some (l, d) → d.kind = k) :
    WgStep s s' Ti (k :: TK) K g where
  same j l d hj hi hk' := by
    simp only [List.mem_cons, not_or] at hk'
    refine h.same j l d hj ?_ hk'.2
    simp only [List.mem_cons, not_or]
    refine ⟨?_, hi⟩
    rintro rfl; exact hk'.1 (hk l d hj)
  touched j hj l d hd := (h.touched j (List.mem_cons_of_mem _ hj) l d hd).imp (List.mem_cons_of_mem _) id
  kindKeep := h.kindKeep
  kinds := h.kinds
  size := h.size
  lost := h.lost

theorem WgState.set_get (s : WgState) (i : Nat) (d : WgData) (j : Nat) :
    (s.set i d).objs[j]? = if i = j then (s.objs[j]?).map (fun p => (p.1, d)) else s.objs[j]? := by
  unfold WgState.set
  rw [Array.getElem?_modify]

theorem WgState.set_size (s : WgState) (i : Nat) (d : WgData) : (s.set i d).objs.size = s.objs.size := by
  simp [WgState.set]

/-- Setting an object's data to data of its kind. -/
theorem WgStep.set {s : WgState} {i : Nat} {d : WgData} {k : WgObj}
    (hk : ∀ l d0, s.objs[i]? = some (l, d0) → d0.kind = k) (hd : d.kind = k) :
    WgStep s (s.set i d) [] [k] [] 0 where
  same j l d0 hj _ hk' := by
    refine ⟨d0, ?_, .refl d0⟩
    rw [WgState.set_get]
    split
    · rename_i e; subst e; exact absurd (hk l d0 hj) (by simpa using hk')
    · exact hj
  touched _ h := nomatch h
  kindKeep j l l' d0 d1 h h' := by
    rw [WgState.set_get] at h'
    split at h'
    · rename_i e; subst e
      rw [h] at h'; simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h'
      obtain ⟨-, rfl⟩ := h'
      rw [hk l d0 h, hd]
    · rw [h] at h'; cases h'; rfl
  kinds k' _ j := by
    unfold WgState.LiveAt
    rw [WgState.set_get]
    split
    · rename_i e; subst e
      cases h : s.objs[i]? with
      | none => simp
      | some p =>
        obtain ⟨l, d0⟩ := p
        have := hk l d0 h
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq]
        constructor
        · rintro ⟨d1, ⟨rfl, rfl⟩, rfl⟩; exact ⟨d, ⟨rfl, rfl⟩, by rw [hd, ← this]⟩
        · rintro ⟨d1, ⟨rfl, rfl⟩, rfl⟩; exact ⟨d0, ⟨rfl, rfl⟩, by rw [this, ← hd]⟩
    · exact Iff.rfl
  size := WgState.set_size s i d
  lost := rfl

/-- Setting an object's data to data of the same shapes. -/
theorem WgStep.setSame {s : WgState} {i : Nat} {d : WgData}
    (hs : ∀ l d0, s.objs[i]? = some (l, d0) → d0.Same d) : WgStep s (s.set i d) [] [] [] 0 where
  same j l d0 hj _ _ := by
    rw [WgState.set_get]
    split
    · rename_i e; subst e
      exact ⟨d, by rw [hj]; rfl, hs l d0 hj⟩
    · exact ⟨d0, hj, .refl d0⟩
  touched _ h := nomatch h
  kindKeep j l l' d0 d1 h h' := by
    rw [WgState.set_get] at h'
    split at h'
    · rename_i e; subst e
      rw [h] at h'; simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h'
      obtain ⟨-, rfl⟩ := h'
      exact (hs l d0 h).1
    · rw [h] at h'; cases h'; rfl
  kinds k' _ j := by
    unfold WgState.LiveAt
    rw [WgState.set_get]
    split
    · rename_i e; subst e
      cases h : s.objs[i]? with
      | none => simp
      | some p =>
        obtain ⟨l, d0⟩ := p
        have := (hs l d0 h).1
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq]
        constructor
        · rintro ⟨d1, ⟨rfl, rfl⟩, rfl⟩; exact ⟨d, ⟨rfl, rfl⟩, this.symm⟩
        · rintro ⟨d1, ⟨rfl, rfl⟩, rfl⟩; exact ⟨d0, ⟨rfl, rfl⟩, this⟩
    · exact Iff.rfl
  size := WgState.set_size s i d
  lost := rfl

theorem WgState.free_get (s : WgState) (i j : Nat) :
    (s.freeAll [i]).objs[j]? = if i = j then (s.objs[j]?).map (fun p => (false, p.2)) else s.objs[j]? := by
  simp only [WgState.freeAll, List.foldl_cons, List.foldl_nil]
  rw [Array.getElem?_modify]

/-- Using up object `i`, of kind `k`. -/
theorem WgStep.free {s : WgState} {i : Nat} {k : WgObj}
    (hk : ∀ l d0, s.objs[i]? = some (l, d0) → d0.kind = k) :
    WgStep s (s.freeAll [i]) [i] [] [k] 0 where
  same j l d0 hj hi _ := by
    refine ⟨d0, ?_, .refl d0⟩
    rw [WgState.free_get, if_neg (by simpa [eq_comm] using hi)]
    exact hj
  touched j hj l d hd := by
    simp only [List.mem_singleton] at hj; subst hj
    exact .inr (by simp [hk l d hd])
  kindKeep j l l' d0 d1 h h' := by
    rw [WgState.free_get] at h'
    split at h'
    · rw [h] at h'; simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h'
      obtain ⟨-, rfl⟩ := h'; rfl
    · rw [h] at h'; cases h'; rfl
  kinds k' hk' j := by
    unfold WgState.LiveAt
    rw [WgState.free_get]
    split
    · rename_i e; subst e
      cases h : s.objs[i]? with
      | none => simp
      | some p =>
        obtain ⟨l, d0⟩ := p
        have := hk l d0 h
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq]
        constructor
        · rintro ⟨d1, ⟨rfl, rfl⟩, e⟩; exact absurd (e.symm.trans this) (by simpa [eq_comm] using hk')
        · rintro ⟨d1, ⟨e, -⟩, -⟩; cases e
    · exact Iff.rfl
  size := by simp [WgState.freeAll]
  lost := rfl

/-- Adding an object. -/
theorem WgStep.push (s : WgState) (d : WgData) : WgStep s (s.push d).2 [] [] [d.kind] 1 where
  same j l d0 hj _ _ := by
    refine ⟨d0, ?_, .refl d0⟩
    simp only [WgState.push]
    rw [Array.getElem?_push]
    have : j < s.objs.size := (Array.getElem?_eq_some_iff.mp hj).1
    rw [if_neg (by omega)]; exact hj
  touched _ h := nomatch h
  kindKeep j l l' d0 d1 h h' := by
    simp only [WgState.push] at h'
    rw [Array.getElem?_push] at h'
    have : j < s.objs.size := (Array.getElem?_eq_some_iff.mp h).1
    rw [if_neg (by omega), h] at h'; cases h'; rfl
  kinds k hk j := by
    unfold WgState.LiveAt
    simp only [WgState.push]
    rw [Array.getElem?_push]
    split
    · rename_i e; subst e
      simp only [Option.some.injEq, Prod.mk.injEq]
      constructor
      · rintro ⟨d1, hd1, -⟩; rw [Array.getElem?_eq_none_iff.mpr (Nat.le_refl _)] at hd1; cases hd1
      · rintro ⟨d1, ⟨-, rfl⟩, e⟩; exact absurd e.symm (by simpa using hk)
    · exact Iff.rfl
  size := by simp [WgState.push]
  lost := rfl

/-- New contents for a buffer keep every object's shape. -/
theorem WgStep.putBytes (s : WgState) (i : Nat) (bs : ByteArray) : WgStep s (s.putBytes i bs) [] [] [] 0 := by
  unfold WgState.putBytes
  split
  · rename_i u old h
    refine WgStep.setSame fun l d0 hd => ?_
    have : d0 = .buffer u old := by
      simp only [WgState.data?, hd, Option.map_some, Option.some.injEq] at h; exact h
    subst this
    refine ⟨rfl, fun s' sh => ?_⟩
    cases sh <;> simp [WgState.Fits, exactly_size]
  · exact WgStep.refl s

/-- New contents for some of a list of buffers keep every object's shape. -/
theorem WgStep.putAll (l : List ((Nat × Bool) × Nat)) (f : (Nat × Bool) × Nat → ByteArray) (s : WgState) :
    WgStep s (l.foldl (fun s x => if x.1.2 then s else s.putBytes x.1.1 (f x)) s) [] [] [] 0 := by
  induction l generalizing s with
  | nil => exact WgStep.refl s
  | cons x l ih =>
      simp only [List.foldl_cons]
      split
      · exact ih s
      · exact (WgStep.putBytes _ _ _).trans (ih _)

/-- One command, run, keeps every object's shape. -/
theorem WgStep.cmd (sh : Dispatch → List ByteArray → List ByteArray) (s : WgState) (c : WgCmd) :
    WgStep s (runWgCmd sh s c) [] [] [] 0 := by
  cases c with
  | copy src so dst dof n => exact WgStep.putBytes _ _ _
  | dispatch q b xyz =>
      simp only [runWgCmd]
      split
      · exact WgStep.putAll _ _ _
      · exact WgStep.refl s
  | draw v g =>
      simp only [runWgCmd]
      split
      · rename_i t es _ _
        split
        · rename_i si dr bi ht _
          refine WgStep.setSame fun l d0 hd => ?_
          have : d0 = .texture si dr := by
            simp only [WgState.data?, hd, Option.map_some, Option.some.injEq] at ht; exact ht
          subst this
          refine ⟨rfl, fun s' sh => ?_⟩
          cases sh <;> simp [WgState.Fits]
        · exact WgStep.refl s
      · exact WgStep.refl s

/-- A command buffer's commands, run, keep every object's shape. -/
theorem WgStep.cmds (sh : Dispatch → List ByteArray → List ByteArray) (s : WgState) (cmds : List WgCmd) :
    WgStep s (cmds.foldl (runWgCmd sh) s) [] [] [] 0 := by
  induction cmds generalizing s with
  | nil => exact WgStep.refl s
  | cons c cs ih => exact (WgStep.cmd sh s c).trans (ih _)

end AlgorithmLib.HProg.Sem
