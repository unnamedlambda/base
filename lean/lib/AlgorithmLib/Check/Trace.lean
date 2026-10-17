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

/-- A call as the checker records it: its callee and, of each argument, the
    bits where they are known. -/
structure CallRec where
  callee : Callee
  args : List (Option UInt64)
  deriving Repr, DecidableEq

/-- A value carries the bits a record gives its place, where it gives any. -/
def argOk : Option UInt64 → V → Prop
  | none, _ => True
  | some b, v => asBits v = some b

/-- Arguments carry the bits a record gives, place by place. -/
def argsOk : List (Option UInt64) → List V → Prop
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

/-- Worlds with the same observations have made the same calls. -/
theorem Traced.of_obs {t : Tr} {w0 w w' : World} (h : Traced t w0 w) (e : callsOf w'.obs = callsOf w.obs) :
    Traced t w0 w' := by
  obtain ⟨cs, e1, k⟩ := h
  exact ⟨cs, e.trans e1, k⟩

end AlgorithmLib.Check
