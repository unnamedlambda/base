module
public import AlgorithmLib.Check.Rel
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.Trace` — the calls a region may make

A pattern (`Tr`) over the calls a run makes, in order: of each call its callee
and, of each argument, the bits where the checker knows them. `Tr.Has` says
which call sequences a pattern admits, `Traced` that a world's observations
extend another's by a sequence a pattern admits, and `Tr.norm` writes a pattern
shorter without admitting less.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Check

/-- What a record says of one value a call is observed with: nothing, its
    bits, or that its bits are a number from `lo` to `hi`. -/
inductive ArgPat where
  | any
  | is (b : UInt64)
  | within (lo hi : Nat)
  deriving Repr, DecidableEq

/-- A call as the checker records it: its callee and what it knows of each
    value the call is observed with. -/
structure CallRec where
  callee : Callee
  args : List ArgPat
  deriving Repr, DecidableEq

/-- A value is one a record's place admits. -/
def argOk : ArgPat → V → Prop
  | .any, _ => True
  | .is b, v => asBits v = some b
  | .within lo hi, v => ∃ b, asBits v = some b ∧ lo ≤ b.toNat ∧ b.toNat ≤ hi

/-- Values a record admits, place by place. -/
def argsOk : List ArgPat → List V → Prop
  | [], [] => True
  | o :: os, v :: vs => argOk o v ∧ argsOk os vs
  | _, _ => False

/-- A call a record describes. -/
def CallRec.Matches (r : CallRec) (x : Callee × List V) : Prop :=
  r.callee = x.1 ∧ argsOk r.args x.2

/-- Calls the records describe, one each, in order. -/
def recsMatch : List CallRec → List (Callee × List V) → Prop
  | [], [] => True
  | r :: rs, x :: xs => r.Matches x ∧ recsMatch rs xs
  | _, _ => False

theorem argsOk_append : ∀ {os os' : List ArgPat} {vs vs' : List V},
    argsOk os vs → argsOk os' vs' → argsOk (os ++ os') (vs ++ vs')
  | [], _, [], _, _, h => h
  | _ :: _, _, _ :: _, _, ⟨h1, h2⟩, h => ⟨h1, argsOk_append h2 h⟩
  | [], _, _ :: _, _, h, _ => h.elim
  | _ :: _, _, [], _, h, _ => h.elim

theorem recsMatch_append : ∀ {a b : List CallRec} {s t : List (Callee × List V)},
    recsMatch a s → recsMatch b t → recsMatch (a ++ b) (s ++ t)
  | [], _, [], _, _, h => h
  | _ :: _, _, _ :: _, _, ⟨h1, h2⟩, h => ⟨h1, recsMatch_append h2 h⟩
  | [], _, _ :: _, _, h, _ => h.elim
  | _ :: _, _, [], _, h, _ => h.elim

theorem recsMatch_nil {s : List (Callee × List V)} (h : recsMatch [] s) : s = [] := by
  cases s with
  | nil => rfl
  | cons _ _ => exact h.elim

/-- **A pattern over call sequences**: none, any, the calls some records
    describe, one pattern's then another's, either's, any number of one's, and
    exactly `n` of one's. -/
inductive Tr where
  | never
  | any
  | lit (cs : List CallRec)
  | seq (a b : Tr)
  | alt (a b : Tr)
  | star (a : Tr)
  | rep (n : Nat) (a : Tr)
  deriving Repr, DecidableEq

/-- The call sequences a pattern admits. -/
inductive Tr.Has : Tr → List (Callee × List V) → Prop
  | any (s : List (Callee × List V)) : Has .any s
  | lit {cs : List CallRec} {s : List (Callee × List V)} : recsMatch cs s → Has (.lit cs) s
  | seq {a b : Tr} {s t : List (Callee × List V)} : Has a s → Has b t → Has (.seq a b) (s ++ t)
  | altL {a b : Tr} {s : List (Callee × List V)} : Has a s → Has (.alt a b) s
  | altR {a b : Tr} {s : List (Callee × List V)} : Has b s → Has (.alt a b) s
  | starNil {a : Tr} : Has (.star a) []
  | starSnoc {a : Tr} {s t : List (Callee × List V)} : Has (.star a) s → Has a t → Has (.star a) (s ++ t)
  | repZero {a : Tr} : Has (.rep 0 a) []
  | repSucc {a : Tr} {n : Nat} {s t : List (Callee × List V)} :
      Has (.rep n a) s → Has a t → Has (.rep (n + 1) a) (s ++ t)

theorem Tr.Has.nil : Tr.Has (.lit []) [] := .lit trivial

theorem Tr.Has.lit_nil {s : List (Callee × List V)} (h : Tr.Has (.lit []) s) : s = [] := by
  cases h with
  | lit h => exact recsMatch_nil h

/-- Records after records are the records together. -/
theorem Tr.Has.seq_lits {t : Tr} {a b : List CallRec} {s : List (Callee × List V)}
    (h : (Tr.seq (.seq t (.lit a)) (.lit b)).Has s) : (Tr.seq t (.lit (a ++ b))).Has s := by
  cases h with
  | seq h1 h2 =>
      cases h1 with
      | seq h0 h1 =>
          cases h1 with
          | lit h1 =>
              cases h2 with
              | lit h2 => rw [List.append_assoc]; exact .seq h0 (.lit (recsMatch_append h1 h2))

/-- A pattern's sequences repeated are the repetition's. -/
theorem Tr.Has.star_append {a : Tr} {s t : List (Callee × List V)} (hs : Tr.Has (.star a) s)
    (ht : Tr.Has (.star a) t) : Tr.Has (.star a) (s ++ t) := by
  generalize he : Tr.star a = p at ht
  induction ht with
  | starNil => cases he; simpa using hs
  | starSnoc h1 h2 ih =>
      cases he
      rw [← List.append_assoc]
      exact .starSnoc (ih rfl) h2
  | _ => cases he

-- ---------------------------------------------------------------------------
-- Shorter patterns
-- ---------------------------------------------------------------------------

/-- One pattern then another, written shorter. -/
def Tr.seq' (a b : Tr) : Tr :=
  match a, b with
  | .never, _ => .never
  | _, .never => .never
  | .lit [], b => b
  | a, .lit [] => a
  | .lit x, .lit y => .lit (x ++ y)
  | .lit x, .seq (.lit y) c => .seq (.lit (x ++ y)) c
  | a, b => .seq a b

/-- Either pattern, written shorter. -/
def Tr.alt' (a b : Tr) : Tr :=
  match a, b with
  | .never, b => b
  | a, .never => a
  | a, b => if a = b then a else .alt a b

/-- Any number of a pattern, written shorter. -/
def Tr.star' (a : Tr) : Tr :=
  match a with
  | .never => .lit []
  | .lit [] => .lit []
  | .alt (.lit []) a => .star a
  | a => .star a

/-- Exactly `n` of a pattern, written shorter. -/
def Tr.rep' (n : Nat) (a : Tr) : Tr :=
  match n, a with
  | 0, _ => .lit []
  | _, .lit [] => .lit []
  | n, a => .rep n a

theorem Tr.Has.rep_nil {n : Nat} {s : List (Callee × List V)} (hs : (Tr.rep n (.lit [])).Has s) : s = [] := by
  generalize he : Tr.rep n (Tr.lit []) = p at hs
  induction hs generalizing n with
  | repZero => rfl
  | repSucc _ h2 ih =>
      cases he
      rw [h2.lit_nil, List.append_nil]
      first | exact ih rfl | exact ih _ rfl
  | _ => cases he

theorem Tr.Has.rep' {n : Nat} {a : Tr} {s : List (Callee × List V)} (hs : (Tr.rep n a).Has s) :
    (Tr.rep' n a).Has s := by
  unfold Tr.rep'
  split
  · cases hs with
    | repZero => exact .nil
  · rw [hs.rep_nil]; exact .nil
  · exact hs

theorem Tr.Has.seq' {a b : Tr} {s t : List (Callee × List V)} (hs : a.Has s) (ht : b.Has t) :
    (a.seq' b).Has (s ++ t) := by
  unfold Tr.seq'
  split
  · cases hs
  · cases ht
  · rw [hs.lit_nil]; exact ht
  · rw [ht.lit_nil, List.append_nil]; exact hs
  · cases hs with | lit h1 => cases ht with | lit h2 => exact .lit (recsMatch_append h1 h2)
  · cases hs with
    | lit h1 =>
      cases ht with
      | seq h2 h3 =>
        cases h2 with
        | lit h2 => rw [← List.append_assoc]; exact .seq (.lit (recsMatch_append h1 h2)) h3
  · exact .seq hs ht

theorem Tr.Has.altL' {a b : Tr} {s : List (Callee × List V)} (hs : a.Has s) : (a.alt' b).Has s := by
  unfold Tr.alt'
  split
  · cases hs
  · exact hs
  · split
    · exact hs
    · exact .altL hs

theorem Tr.Has.altR' {a b : Tr} {s : List (Callee × List V)} (hs : b.Has s) : (a.alt' b).Has s := by
  unfold Tr.alt'
  split
  · exact hs
  · cases hs
  · split
    · rename_i e; rw [e]; exact hs
    · exact .altR hs

theorem Tr.Has.star' {a : Tr} {s : List (Callee × List V)} (hs : (Tr.star a).Has s) : a.star'.Has s := by
  unfold Tr.star'
  split
  · generalize he : Tr.star Tr.never = p at hs
    induction hs with
    | starNil => exact .nil
    | starSnoc _ h2 _ => cases he; cases h2
    | _ => cases he
  · generalize he : Tr.star (Tr.lit []) = p at hs
    induction hs with
    | starNil => exact .nil
    | starSnoc _ h2 ih => cases he; rw [h2.lit_nil, List.append_nil]; exact ih rfl
    | _ => cases he
  · rename_i x
    generalize he : Tr.star (Tr.alt (Tr.lit []) x) = p at hs
    induction hs with
    | starNil => exact .starNil
    | starSnoc _ h2 ih =>
        cases he
        cases h2 with
        | altL h => rw [h.lit_nil, List.append_nil]; exact ih rfl
        | altR h => exact .starSnoc (ih rfl) h
    | _ => cases he
  · exact hs

/-- A pattern followed by `k`, written shorter. -/
def Tr.normK : Tr → Tr → Tr
  | .never, _ => .never
  | .any, k => Tr.seq' .any k
  | .lit cs, k => Tr.seq' (.lit cs) k
  | .seq a b, k => a.normK (b.normK k)
  | .alt a b, k => Tr.seq' (Tr.alt' (a.normK (.lit [])) (b.normK (.lit []))) k
  | .star a, k => Tr.seq' (Tr.star' (a.normK (.lit []))) k
  | .rep n a, k => Tr.seq' (Tr.rep' n (a.normK (.lit []))) k

/-- A pattern written shorter. -/
def Tr.norm (t : Tr) : Tr := t.normK (.lit [])

theorem Tr.Has.normK : ∀ {a k : Tr} {s t : List (Callee × List V)}, a.Has s → k.Has t →
    (a.normK k).Has (s ++ t)
  | .never, _, _, _, hs, _ => by cases hs
  | .any, _, _, _, hs, ht => Tr.Has.seq' hs ht
  | .lit _, _, _, _, hs, ht => Tr.Has.seq' hs ht
  | .seq a b, k, _, _, hs, ht => by
      cases hs with
      | seq h1 h2 =>
          rw [List.append_assoc]
          show (a.normK (b.normK k)).Has _
          exact Tr.Has.normK h1 (Tr.Has.normK h2 ht)
  | .alt a b, k, _, _, hs, ht => by
      show (Tr.seq' (Tr.alt' (a.normK (.lit [])) (b.normK (.lit []))) k).Has _
      cases hs with
      | altL h => exact Tr.Has.seq' (Tr.Has.altL' (by simpa using Tr.Has.normK h .nil)) ht
      | altR h => exact Tr.Has.seq' (Tr.Has.altR' (by simpa using Tr.Has.normK h .nil)) ht
  | .star a, k, s, _, hs, ht => by
      show (Tr.seq' (Tr.star' (a.normK (.lit []))) k).Has _
      refine Tr.Has.seq' (Tr.Has.star' ?_) ht
      generalize he : Tr.star a = p at hs
      induction hs with
      | starNil => exact .starNil
      | starSnoc _ h2 ih =>
          cases he
          exact .starSnoc (ih rfl) (by simpa using Tr.Has.normK h2 .nil)
      | _ => cases he
  | .rep n a, k, s, _, hs, ht => by
      show (Tr.seq' (Tr.rep' n (a.normK (.lit []))) k).Has _
      refine Tr.Has.seq' (Tr.Has.rep' ?_) ht
      generalize he : Tr.rep n a = p at hs
      induction hs generalizing n with
      | repZero => cases he; exact .repZero
      | repSucc _ h2 ih =>
          cases he
          exact .repSucc (ih _ rfl) (by simpa using Tr.Has.normK h2 .nil)
      | _ => cases he

/-- **The shorter pattern admits what the pattern does.** -/
theorem Tr.Has.norm {a : Tr} {s : List (Callee × List V)} (h : a.Has s) : a.norm.Has s := by
  simpa using Tr.Has.normK h .nil

-- ---------------------------------------------------------------------------
-- Observations
-- ---------------------------------------------------------------------------

/-- The calls among some observations. -/
def callsOf (obs : List Obs) : List (Callee × List V) :=
  obs.filterMap fun
    | .call c vs => some (c, vs)
    | .store .. => none

/-- **The calls made from `w0` to `w` are ones `t` admits**: `w`'s
    observations are `w0`'s and then those calls, in order. -/
def Traced (t : Tr) (w0 w : World) : Prop :=
  ∃ cs, callsOf w.obs = cs.reverse ++ callsOf w0.obs ∧ t.Has cs

/-- **The calls among `obs` are the ones `w` had observed and then ones `t`
    admits**: what a run from `w` that ends observing `obs` made, in order. -/
def Tr.Covers (t : Tr) (w : World) (obs : List Obs) : Prop :=
  ∃ cs, callsOf obs = (callsOf w.obs).reverse ++ cs ∧ t.Has cs

theorem callsOf_reverse (obs : List Obs) : callsOf obs.reverse = (callsOf obs).reverse :=
  List.filterMap_reverse

theorem Traced.refl (w : World) : Traced (.lit []) w w := ⟨[], rfl, .nil⟩

theorem Traced.mono {a b : Tr} {w0 w : World} (h : Traced a w0 w) (hab : ∀ s, a.Has s → b.Has s) :
    Traced b w0 w := by
  obtain ⟨cs, e, hc⟩ := h
  exact ⟨cs, e, hab cs hc⟩

theorem Traced.seq {a b : Tr} {w0 w1 w2 : World} (h1 : Traced a w0 w1) (h2 : Traced b w1 w2) :
    Traced (.seq a b) w0 w2 := by
  obtain ⟨c1, e1, k1⟩ := h1
  obtain ⟨c2, e2, k2⟩ := h2
  exact ⟨c1 ++ c2, by rw [e2, e1, List.reverse_append, List.append_assoc], .seq k1 k2⟩

theorem Traced.starSnoc {a : Tr} {w0 w1 w2 : World} (h1 : Traced (.star a) w0 w1) (h2 : Traced a w1 w2) :
    Traced (.star a) w0 w2 := by
  obtain ⟨c1, e1, k1⟩ := h1
  obtain ⟨c2, e2, k2⟩ := h2
  exact ⟨c1 ++ c2, by rw [e2, e1, List.reverse_append, List.append_assoc], .starSnoc k1 k2⟩

theorem Traced.repZero {a : Tr} (w : World) : Traced (.rep 0 a) w w := ⟨[], rfl, .repZero⟩

theorem Traced.repSucc {a : Tr} {n : Nat} {w0 w1 w2 : World} (h1 : Traced (.rep n a) w0 w1)
    (h2 : Traced a w1 w2) : Traced (.rep (n + 1) a) w0 w2 := by
  obtain ⟨c1, e1, k1⟩ := h1
  obtain ⟨c2, e2, k2⟩ := h2
  exact ⟨c1 ++ c2, by rw [e2, e1, List.reverse_append, List.append_assoc], .repSucc k1 k2⟩

/-- What no step of a host program changes: the kernel and vendor oracles
    and which libraries are present. -/
def Fixed (w w' : World) : Prop := w'.kernel = w.kernel ∧ w'.vendor = w.vendor ∧ w'.present = w.present

theorem Fixed.refl (w : World) : Fixed w w := ⟨rfl, rfl, rfl⟩

theorem Fixed.trans {a b c : World} (h1 : Fixed a b) (h2 : Fixed b c) : Fixed a c :=
  ⟨h2.1.trans h1.1, h2.2.1.trans h1.2.1, h2.2.2.trans h1.2.2⟩

/-- A byte no store among `ds` writes. -/
def Unstored (ds : List Obs) (a : UInt64) : Prop :=
  ∀ b n v, Obs.store b n v ∈ ds → ∀ i < n, a ≠ b + UInt64.ofNat i

/-- **Host memory changed only where stores were observed**: from `w` to
    `w'` the observations grew by `ds`, and every byte no store among them
    writes reads as it did. -/
def MemLogged (w w' : World) : Prop :=
  ∃ ds, w'.obs = ds ++ w.obs ∧ ∀ a, Unstored ds a → w'.mem.load a 1 = w.mem.load a 1

theorem MemLogged.refl (w : World) : MemLogged w w := ⟨[], rfl, fun _ _ => rfl⟩

theorem MemLogged.trans {a b c : World} (h1 : MemLogged a b) (h2 : MemLogged b c) : MemLogged a c := by
  obtain ⟨d1, e1, k1⟩ := h1
  obtain ⟨d2, e2, k2⟩ := h2
  refine ⟨d2 ++ d1, by rw [e2, e1, List.append_assoc], fun x hx => ?_⟩
  rw [k2 x (fun b n v hm => hx b n v (List.mem_append_left _ hm)),
    k1 x (fun b n v hm => hx b n v (List.mem_append_right _ hm))]

/-- A number outside every range `[lo, hi)` of `K`. -/
def Outside (K : List (Nat × Nat)) (x : Nat) : Prop := ∀ p ∈ K, x < p.1 ∨ p.2 ≤ x

/-- **No store observed from `w0` to `w` touches the ranges `K`**, where `K`
    names any: the observations grew by `ds`, and every byte a store among
    them writes lies outside `K`. -/
def StoresOut (K : List (Nat × Nat)) (w0 w : World) : Prop :=
  K = [] ∨ ∃ ds, w.obs = ds ++ w0.obs ∧
    ∀ b n v, Obs.store b n v ∈ ds → ∀ i < n, Outside K (b + UInt64.ofNat i).toNat

theorem StoresOut.refl (K : List (Nat × Nat)) (w : World) : StoresOut K w w :=
  .inr ⟨[], rfl, fun _ _ _ h => by cases h⟩

theorem StoresOut.trans {K : List (Nat × Nat)} {a b c : World} (h1 : StoresOut K a b) (h2 : StoresOut K b c) :
    StoresOut K a c := by
  rcases h1 with h | ⟨d1, e1, k1⟩
  · exact .inl h
  rcases h2 with h | ⟨d2, e2, k2⟩
  · exact .inl h
  refine .inr ⟨d2 ++ d1, by rw [e2, e1, List.append_assoc], fun x n v hm => ?_⟩
  rcases List.mem_append.mp hm with hm | hm
  · exact k2 x n v hm
  · exact k1 x n v hm

/-- A step observing one call observes no store. -/
theorem StoresOut.call {K : List (Nat × Nat)} {w w' : World} {c : Callee} {vs : List V}
    (ho : w'.obs = .call c vs :: w.obs) : StoresOut K w w' :=
  .inr ⟨[.call c vs], ho, fun _ _ _ h => by simp at h⟩

/-- A byte untouched by the stores of a longer stretch is untouched by those
    of a part of it. -/
theorem Unstored.of_append {ds es : List Obs} {a : UInt64} (h : Unstored (ds ++ es) a) :
    Unstored ds a ∧ Unstored es a :=
  ⟨fun b n v hm => h b n v (List.mem_append_left _ hm), fun b n v hm => h b n v (List.mem_append_right _ hm)⟩

/-- **One step of a run, as the device sees it**: one that leaves the device,
    the oracles and the calls observed as they were, host memory changed only
    where it observed stores (`MemLogged`); a call into the driver or cuBLAS,
    observed, and answered as the World's contract says; or one that made
    some other call (into the program's own functions, a library, the host),
    observed among those it adds, which leaves the device as that call does. -/
inductive DevStep : World → World → Prop
  | host {w w' : World} : w'.dev = w.dev → callsOf w'.obs = callsOf w.obs → Fixed w w' → MemLogged w w' →
      DevStep w w'
  | ext {w w' : World} {e : Ext} {vs : List V} {r : Option V} :
      extCall e vs (obsCall w (.ext e) vs) = some (r, w') →
      w'.obs = .call (.ext e) (vs ++ callWords w (.ext e) vs) :: w.obs → DevStep w w'
  | other {w w' : World} {ds : List (Callee × List V)} :
      callsOf w'.obs = ds ++ callsOf w.obs → (∃ x ∈ ds, ∀ e, x.1 ≠ .ext e) → DevStep w w'

/-- **The steps a run took, as the device sees them.** -/
inductive DevPath : World → World → Prop
  | refl (w : World) : DevPath w w
  | tail {w0 w1 w2 : World} : DevPath w0 w1 → DevStep w1 w2 → DevPath w0 w2

theorem DevPath.trans {w0 w1 w2 : World} (h1 : DevPath w0 w1) (h2 : DevPath w1 w2) : DevPath w0 w2 := by
  induction h2 with
  | refl => exact h1
  | tail _ hs ih => exact .tail ih hs

theorem DevPath.single {w w' : World} (h : DevStep w w') : DevPath w w' := .tail (.refl w) h

/-- Calls the records describe call what the records name. -/
theorem recsMatch_callees : ∀ {rs : List CallRec} {xs : List (Callee × List V)}, recsMatch rs xs →
    ∀ x ∈ xs, ∃ r ∈ rs, r.callee = x.1
  | [], [], _, _, hx => by cases hx
  | r :: rs, y :: ys, ⟨hm, h⟩, x, hx => by
      rcases List.mem_cons.mp hx with rfl | hx
      · exact ⟨r, List.mem_cons_self .., hm.1⟩
      · obtain ⟨r', hr', e⟩ := recsMatch_callees h x hx
        exact ⟨r', List.mem_cons_of_mem _ hr', e⟩

/-- **Calls a list of records into the driver and cuBLAS admits are all into
    them.** -/
theorem Tr.Has.lit_ext {rs : List CallRec} {xs : List (Callee × List V)} (h : (Tr.lit rs).Has xs)
    (hr : rs.all (fun r => match r.callee with | .ext _ => true | _ => false) = true) :
    ∀ x ∈ xs, ∃ e, x.1 = .ext e := by
  cases h with
  | lit hm =>
      intro x hx
      obtain ⟨r, hr', e⟩ := recsMatch_callees hm x hx
      have := List.all_eq_true.mp hr r hr'
      cases hc : r.callee with
      | ext e' => exact ⟨e', by rw [← e, hc]⟩
      | _ => rw [hc] at this; cases this

theorem eq_of_length_eleven {α : Type} : ∀ (l : List α), l.length = 11 →
    ∃ a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10, l = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]
  | [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10], _ => ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, rfl⟩

theorem eq_of_length_eighteen {α : Type} : ∀ (l : List α), l.length = 18 →
    ∃ a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17,
      l = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17]
  | [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17], _ =>
      ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, rfl⟩

theorem callsOf_call (c : Callee) (vs : List V) (obs : List Obs) :
    callsOf (.call c vs :: obs) = (c, vs) :: callsOf obs := rfl

/-- A run's calls are the ones made before it and then some. -/
theorem DevPath.calls {w0 w : World} (h : DevPath w0 w) : ∃ ds, callsOf w.obs = ds ++ callsOf w0.obs := by
  induction h with
  | refl => exact ⟨[], rfl⟩
  | tail _ hs ih =>
      obtain ⟨ds, e⟩ := ih
      cases hs with
      | host _ ho _ => exact ⟨ds, by rw [ho, e]⟩
      | ext _ ho => exact ⟨_ :: ds, by rw [ho, callsOf_call, e]; rfl⟩
      | other hd _ => exact ⟨_ ++ ds, by rw [hd, e, List.append_assoc]⟩

/-- **The device's run through calls `cs` into the driver and cuBLAS**, in the
    order they were made: each answered as the World's contract says, the
    steps between them leaving the device, the oracles and the calls
    observed, and host memory but where they observed stores. -/
inductive ExtRun : World → List (Callee × List V) → World → Prop
  | nil {w w' : World} : w'.dev = w.dev → callsOf w'.obs = callsOf w.obs → Fixed w w' → MemLogged w w' →
      ExtRun w [] w'
  | call {w0 w1 w2 : World} {cs : List (Callee × List V)} {e : Ext} {vs : List V} {r : Option V} :
      ExtRun w0 cs w1 → extCall e vs (obsCall w1 (.ext e) vs) = some (r, w2) →
      w2.obs = .call (.ext e) (vs ++ callWords w1 (.ext e) vs) :: w1.obs →
      ExtRun w0 (cs ++ [(.ext e, vs ++ callWords w1 (.ext e) vs)]) w2
  | host {w0 w1 w2 : World} {cs : List (Callee × List V)} :
      ExtRun w0 cs w1 → w2.dev = w1.dev → callsOf w2.obs = callsOf w1.obs → Fixed w1 w2 → MemLogged w1 w2 →
      ExtRun w0 cs w2

/-- **A run whose calls were all into the driver or cuBLAS is those calls**,
    in order, as the device sees them. -/
theorem DevPath.extRun {w0 w : World} (h : DevPath w0 w) :
    ∀ cs, callsOf w.obs = cs ++ callsOf w0.obs → (∀ x ∈ cs, ∃ e, x.1 = .ext e) → ExtRun w0 cs.reverse w := by
  induction h with
  | refl =>
      intro cs e _
      have : cs = [] := by
        have := congrArg List.length e; simpa using this
      subst this; exact .nil rfl rfl (Fixed.refl _) (MemLogged.refl _)
  | tail hp hs ih =>
      intro cs e hall
      obtain ⟨ds, hd⟩ := hp.calls
      cases hs with
      | host hdv ho hf hm =>
          rw [ho] at e
          exact .host (ih cs e hall) hdv ho hf hm
      | ext hx ho =>
          rw [ho, callsOf_call, hd, ← List.cons_append] at e
          have e' : cs = _ :: ds := (List.append_cancel_right e).symm
          subst e'
          simp only [List.reverse_cons]
          exact .call (ih ds hd (fun x hx' => hall x (List.mem_cons_of_mem _ hx'))) hx ho
      | other hds hwit =>
          rw [hds, hd, ← List.append_assoc] at e
          have e' := List.append_cancel_right e
          obtain ⟨x, hx, hne⟩ := hwit
          obtain ⟨e0, he0⟩ := hall x (by rw [← e']; exact List.mem_append_left _ hx)
          exact absurd he0 (hne e0)

/-- Worlds with the same observations have made the same calls. -/
theorem Traced.of_obs {t : Tr} {w0 w w' : World} (h : Traced t w0 w) (e : callsOf w'.obs = callsOf w.obs) :
    Traced t w0 w' := by
  obtain ⟨cs, e1, k⟩ := h
  exact ⟨cs, e.trans e1, k⟩

/-- Calls a repetition of one record admits are that record, as many times. -/
theorem Tr.Has.rep_lit {r : CallRec} {n : Nat} {s : List (Callee × List V)}
    (h : (Tr.rep n (.lit [r])).Has s) : recsMatch (List.replicate n r) s := by
  generalize e : Tr.rep n (.lit [r]) = t at h
  induction h generalizing n with
  | repZero => cases e; trivial
  | @repSucc a m s t h1 h2 ih _ =>
      cases e
      cases h2 with
      | lit hm =>
          rw [List.replicate_succ']
          exact recsMatch_append (ih rfl) hm
  | _ => cases e

/-- Calls one record admits, or none. -/
theorem Tr.Has.lit_or_nil {r : CallRec} {s : List (Callee × List V)} (h : (Tr.alt (.lit [r]) (.lit [])).Has s) :
    s = [] ∨ ∃ x, s = [x] ∧ r.Matches x := by
  generalize e : Tr.alt (.lit [r]) (.lit []) = t at h
  cases h with
  | altL h =>
      cases e
      cases h with
      | lit hm =>
          rcases s with _ | ⟨x, _ | ⟨y, s⟩⟩
          · exact hm.elim
          · exact .inr ⟨x, rfl, hm.1⟩
          · exact hm.2.elim
  | altR h => cases e; exact .inl (Tr.Has.lit_nil h)
  | _ => cases e

/-- The stores a stretch of a run observes are among the run's. -/
theorem stretch_sub {w wA wB w' : World} {dT : List Obs} (eT : w'.obs = dT ++ w.obs)
    (hX : ∃ X, w'.obs = X ++ wB.obs) (hY : ∃ Y, wA.obs = Y ++ w.obs) :
    ∀ ds, wB.obs = ds ++ wA.obs → ∀ o ∈ ds, o ∈ dT := by
  intro ds e o ho
  obtain ⟨X, hX⟩ := hX
  obtain ⟨Y, hY⟩ := hY
  rw [e, hY] at hX
  have : dT = X ++ ds ++ Y := List.append_cancel_right (by rw [← eT, hX]; simp)
  rw [this]
  exact List.mem_append_left _ (List.mem_append_right _ ho)

/-- **A byte in a kept range reads as it did across a stretch of the run**,
    where no store of the run touches the range. -/
theorem kept_load {K : List (Nat × Nat)} {dT : List Obs}
    (kT : ∀ b n v, Obs.store b n v ∈ dT → ∀ i < n, Outside K (b + UInt64.ofNat i).toNat)
    {p : Nat × Nat} (hp : p ∈ K) {a : UInt64} (ha : p.1 ≤ a.toNat ∧ a.toNat < p.2) {wA wB : World}
    (hm : MemLogged wA wB) (hsub : ∀ ds, wB.obs = ds ++ wA.obs → ∀ o ∈ ds, o ∈ dT) :
    wB.mem.load a 1 = wA.mem.load a 1 := by
  obtain ⟨ds, e, k⟩ := hm
  refine k a fun b n v hs i hi he => ?_
  have := kT b n v (hsub ds e _ hs) i hi p hp
  rw [← he] at this
  omega

end AlgorithmLib.Check
