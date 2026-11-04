module
public import AlgorithmLib.ML.Model.Fuse
meta import AlgorithmLib.ML.Model.Fuse
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A tape run in another order

A schedule issues a tape's operations in an order of its own: grouped into
kernels, levelled for streams. `pull_den` says when that order computes what
the tape does, on every buffer: when each operation it issues next can be
brought to the front of what is left of the tape past only operations it
cannot observe and that cannot observe it (`den_swap`). `pull` decides that,
reading the order as indices into the tape.

Nothing of the denotations is used but that each operation writes only its
output and reads only what it reads (`TOp.Frame`), so each theorem has a
`_with` form for any meaning of the operations that keeps that frame: one
that reads a contraction as a vendor computes it, say.
-/

namespace AlgorithmLib.ML

section Carrier
variable {R : Type} [NumOps R] [NumLit R]

/-- Two operations neither of which can observe the other: they write
    different buffers, and neither reads what the other writes. -/
def TOp.Indep (batch : Nat) (a b : TOp) : Prop :=
  (TOp.outSize batch a).1 ≠ (TOp.outSize batch b).1
    ∧ (TOp.outSize batch a).1 ∉ b.reads ∧ (TOp.outSize batch b).1 ∉ a.reads

instance (batch : Nat) (a b : TOp) : Decidable (TOp.Indep batch a b) := by
  unfold TOp.Indep; infer_instance

/-- **What an operation's meaning must keep for the tape's order to be a
    choice**: it writes only its output, and reads only what it reads. -/
structure TOp.Frame (batch : Nat) (S : TOp → (Buf → Nat → R) → Buf → Nat → R) : Prop where
  out : ∀ op m b, b ≠ (TOp.outSize batch op).1 → S op m b = m b
  congr : ∀ op (m m' : Buf → Nat → R), (∀ b ∈ op.reads, m b = m' b) → ∀ b a, m b a = m' b a →
    S op m b a = S op m' b a

/-- The operations' denotations keep it. -/
theorem den_frame (batch : Nat) : TOp.Frame (R := R) batch (fun op m => op.den m) :=
  ⟨fun op m b hb => den_out batch op m b hb, fun op m m' h b a hb => den_congr op m m' h b a hb⟩

/-- **Two operations neither of which can observe the other commute**, under
    any meaning that keeps the frame. -/
theorem swap_of_frame {batch : Nat} {S : TOp → (Buf → Nat → R) → Buf → Nat → R} (hS : TOp.Frame batch S)
    (a b : TOp) (m : Buf → Nat → R)
    (hne : (TOp.outSize batch a).1 ≠ (TOp.outSize batch b).1)
    (hab : (TOp.outSize batch a).1 ∉ b.reads)
    (hba : (TOp.outSize batch b).1 ∉ a.reads) :
    S b (S a m) = S a (S b m) := by
  have hA : ∀ r ∈ a.reads, (S b m) r = m r :=
    fun r hr => hS.out b m r (fun e => hba (e ▸ hr))
  have hB : ∀ r ∈ b.reads, (S a m) r = m r :=
    fun r hr => hS.out a m r (fun e => hab (e ▸ hr))
  funext c x
  by_cases hca : c = (TOp.outSize batch a).1
  · subst hca
    rw [congrFun (hS.out b (S a m) _ hne) x]
    exact (hS.congr a (S b m) m hA _ x (congrFun (hS.out b m _ hne) x)).symm
  · by_cases hcb : c = (TOp.outSize batch b).1
    · subst hcb
      rw [congrFun (hS.out a (S b m) _ (Ne.symm hne)) x]
      exact hS.congr b (S a m) m hB _ x (congrFun (hS.out a m _ (Ne.symm hne)) x)
    · rw [congrFun (hS.out b (S a m) c hcb) x, congrFun (hS.out a m c hca) x,
        congrFun (hS.out a (S b m) c hca) x, congrFun (hS.out b m c hcb) x]

/-- **An operation independent of every one before it may run first**, under
    any meaning that keeps the frame. -/
theorem foldl_pull_with {batch : Nat} {S : TOp → (Buf → Nat → R) → Buf → Nat → R} (hS : TOp.Frame batch S)
    (x : TOp) : ∀ (pre : List TOp) (m : Buf → Nat → R),
    (∀ p ∈ pre, TOp.Indep batch p x) →
    S x (pre.foldl (fun mm o => S o mm) m) = pre.foldl (fun mm o => S o mm) (S x m) := by
  intro pre
  induction pre with
  | nil => intro m _; rfl
  | cons p ps ih =>
    intro m h
    obtain ⟨h1, h2, h3⟩ := h p List.mem_cons_self
    simp only [List.foldl_cons]
    rw [ih (S p m) (fun q hq => h q (List.mem_cons_of_mem _ hq)), swap_of_frame hS p x m h1 h2 h3]

/-- **An operation independent of every one before it may run first.** -/
theorem foldl_pull (batch : Nat) (x : TOp) : ∀ (pre : List TOp) (m : Buf → Nat → R),
    (∀ p ∈ pre, TOp.Indep batch p x) →
    x.den (pre.foldl (fun mm o => o.den mm) m) = pre.foldl (fun mm o => o.den mm) (x.den m) :=
  foldl_pull_with (den_frame batch) x

/-- **Whether an order runs what is left of a tape**: each index it names next
    is in what is left, independent of every operation left before it, and
    the rest of the order runs what is left without it. `rem` pairs each
    operation left with its index. -/
def pull (batch : Nat) (rem : List (Nat × TOp)) : List Nat → Bool
  | [] => rem.isEmpty
  | k :: ks =>
    match rem.findIdx? (·.1 == k) with
    | some i => match rem[i]? with
      | some (_, x) => (rem.take i).all (fun p => decide (TOp.Indep batch p.2 x)) && pull batch (rem.eraseIdx i) ks
      | none => false
    | none => false

/-- **An order `pull` accepts computes what the tape does**, under any
    meaning that keeps the frame, on every buffer: the operations it names, read from the tape `t`, folded in its
    order, leave memory as what is left of `t` folded in the tape's own. -/
theorem pull_den_with {batch : Nat} {S : TOp → (Buf → Nat → R) → Buf → Nat → R} (hS : TOp.Frame batch S)
    (t : List TOp) : ∀ (ks : List Nat) (rem : List (Nat × TOp)),
    (∀ p ∈ rem, t[p.1]? = some p.2) → pull batch rem ks = true → ∀ m : Buf → Nat → R,
    (ks.filterMap (t[·]?)).foldl (fun mm o => S o mm) m = (rem.map (·.2)).foldl (fun mm o => S o mm) m := by
  intro ks
  induction ks with
  | nil =>
    intro rem _ h m
    cases rem with
    | nil => rfl
    | cons _ _ => simp [pull] at h
  | cons k ks ih =>
    intro rem hrem h m
    cases hi : rem.findIdx? (·.1 == k) with
    | none => simp [pull, hi] at h
    | some i =>
      cases hx : rem[i]? with
      | none => simp [pull, hi, hx] at h
      | some yx =>
        obtain ⟨y, x⟩ := yx
        simp only [pull, hi, hx, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
        obtain ⟨hpre, hrest⟩ := h
        have hk : y = k := by
          have := List.findIdx?_eq_some_iff_getElem.mp hi
          obtain ⟨hlt, hgk, _⟩ := this
          have e : rem[i] = (y, x) := by
            rw [List.getElem?_eq_getElem hlt] at hx; exact Option.some.inj hx
          simpa [e] using hgk
        subst hk
        have hmem : (y, x) ∈ rem := List.mem_of_getElem? hx
        have htx : t[y]? = some x := hrem _ hmem
        have hlt : i < rem.length := (List.findIdx?_eq_some_iff_getElem.mp hi).1
        have hsplit : rem = rem.take i ++ (y, x) :: rem.drop (i + 1) := by
          have e : rem[i] = (y, x) := by
            rw [List.getElem?_eq_getElem hlt] at hx; exact Option.some.inj hx
          conv => lhs; rw [← List.take_append_drop i rem]
          rw [List.drop_eq_getElem_cons hlt, e]
        have herase : rem.eraseIdx i = rem.take i ++ rem.drop (i + 1) := List.eraseIdx_eq_take_drop_succ rem i
        have hrem' : ∀ p ∈ rem.eraseIdx i, t[p.1]? = some p.2 :=
          fun p hp => hrem p (List.mem_of_mem_eraseIdx hp)
        simp only [List.filterMap_cons, htx, List.foldl_cons]
        rw [ih (rem.eraseIdx i) hrem' hrest (S x m), herase]
        conv => rhs; rw [hsplit]
        simp only [List.map_append, List.map_cons, List.foldl_append, List.foldl_cons]
        rw [foldl_pull_with hS x ((rem.take i).map (·.2)) m (fun p hp => by
          obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hp
          exact hpre q hq)]

/-- **An order `pull` accepts computes what the tape does**, on every
    buffer: the operations it names, read from the tape `t`, folded in its
    order, leave memory as what is left of `t` folded in the tape's own. -/
theorem pull_den (batch : Nat) (t : List TOp) (ks : List Nat) (rem : List (Nat × TOp))
    (hrem : ∀ p ∈ rem, t[p.1]? = some p.2) (h : pull batch rem ks = true) (m : Buf → Nat → R) :
    (ks.filterMap (t[·]?)).foldl (fun mm o => o.den mm) m = (rem.map (·.2)).foldl (fun mm o => o.den mm) m :=
  pull_den_with (den_frame batch) t ks rem hrem h m

/-- **A tape run in an order `pull` accepts from the whole of it computes
    what the tape does**, under any meaning that keeps the frame. -/
theorem pull_tape_with {batch : Nat} {S : TOp → (Buf → Nat → R) → Buf → Nat → R} (hS : TOp.Frame batch S)
    (t : List TOp) (ks : List Nat)
    (h : pull batch (t.zipIdx.map fun p => (p.2, p.1)) ks = true) (m : Buf → Nat → R) :
    (ks.filterMap (t[·]?)).foldl (fun mm o => S o mm) m = t.foldl (fun mm o => S o mm) m := by
  have e : (t.zipIdx.map fun p => (p.2, p.1)).map (·.2) = t := by
    simp [Function.comp_def]
  rw [pull_den_with hS t ks _ (fun p hp => by
    obtain ⟨⟨o, j⟩, hq, rfl⟩ := List.mem_map.mp hp
    simpa using (List.mem_zipIdx_iff_getElem?.mp hq)) h m, e]

/-- **A tape run in an order `pull` accepts from the whole of it computes
    what the tape does.** -/
theorem pull_tape (batch : Nat) (t : List TOp) (ks : List Nat)
    (h : pull batch (t.zipIdx.map fun p => (p.2, p.1)) ks = true) (m : Buf → Nat → R) :
    (ks.filterMap (t[·]?)).foldl (fun mm o => o.den mm) m = t.foldl (fun mm o => o.den mm) m :=
  pull_tape_with (den_frame batch) t ks h m

end Carrier

end AlgorithmLib.ML
