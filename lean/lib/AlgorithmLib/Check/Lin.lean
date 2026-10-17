module
public import AlgorithmLib.Proof.Typestate
@[expose] public section

/-!
# `Check.Lin` — linear forms over slots and region sizes

What the checker knows of a slot beyond its range: a linear form its bits
equal modulo `2 ^ 64` (`Sym.dfn`), and linear forms bounding a multiple of it
from above (`Sym.ubs`). A form reads slots' bits, as numbers, and regions'
sizes.

**`Ctx.prove`** shows a form is at most zero: it rewrites the form, a slot at a
time, by the slot's definition where that is exact (its value inside
`[0, 2 ^ 64)`, so no wrap), or by a bound on it where its coefficient is
positive, until the ranges the checker knows of every slot left bound it.
**`Ctx.prove_sound`**: where every slot is in its range and its facts hold,
a form it proves is at most zero is.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Check

/-- What a linear form reads: a slot's bits as a number, a region's size, or
    a one-operand operation of a slot's bits as a 64-bit word. -/
inductive Atom where
  | slot (r : R)
  | size (g : Region)
  | iun (k : IUn) (r : R)
  deriving DecidableEq, Repr, Inhabited

/-- A slot's bits as a number, `0` where it holds no scalar. -/
def slotVal (Γ : Env) (r : R) : Int :=
  match Γ[r]? with
  | some (.sc _ x) => x.toNat
  | _ => 0

/-- The leading zeros, or the bits reversed, of a 64-bit word, as a number;
    `0` for the other operations. -/
def iunVal : IUn → Int → Int
  | .clz, v => (AlgorithmLib.Prog.clz64 (UInt64.ofNat v.toNat)).toNat
  | .bitrev, v => (AlgorithmLib.Prog.bitrev64 (UInt64.ofNat v.toNat)).toNat
  | _, _ => 0

theorem iunVal_lt (k : IUn) (v : Int) : iunVal k v < 2 ^ 64 := by
  cases k <;> simp only [iunVal] <;>
    first
    | omega
    | (have := UInt64.toNat_lt (AlgorithmLib.Prog.clz64 (UInt64.ofNat v.toNat)); omega)
    | (have := UInt64.toNat_lt (AlgorithmLib.Prog.bitrev64 (UInt64.ofNat v.toNat)); omega)

def Atom.val (Γ : Env) (w : World) : Atom → Int
  | .slot r => slotVal Γ r
  | .size g => w.mem.sizes g
  | .iun k r => iunVal k (slotVal Γ r)

theorem slotVal_nonneg (Γ : Env) (r : R) : 0 ≤ slotVal Γ r := by
  unfold slotVal; split <;> omega

theorem slotVal_lt (Γ : Env) (r : R) : slotVal Γ r < 2 ^ 64 := by
  unfold slotVal; split
  · rename_i x _; have := x.toNat_lt; omega
  · omega

theorem Atom.val_nonneg (Γ : Env) (w : World) (a : Atom) : 0 ≤ a.val Γ w := by
  cases a with
  | slot r => exact slotVal_nonneg Γ r
  | size g => simp [val]
  | iun k r => cases k <;> simp [val, iunVal]

/-- `Σ c · a + k`. -/
structure Lin where
  terms : List (Atom × Int) := []
  k : Int := 0
  deriving DecidableEq, Repr, Inhabited

namespace Lin

def evalT (Γ : Env) (w : World) : List (Atom × Int) → Int
  | [] => 0
  | (a, c) :: ts => c * a.val Γ w + evalT Γ w ts

def eval (Γ : Env) (w : World) (L : Lin) : Int := evalT Γ w L.terms + L.k

def const (k : Int) : Lin := ⟨[], k⟩

def atom (a : Atom) : Lin := ⟨[(a, 1)], 0⟩

def add (L M : Lin) : Lin := ⟨L.terms ++ M.terms, L.k + M.k⟩

def scale (c : Int) (L : Lin) : Lin := ⟨L.terms.map fun (a, d) => (a, c * d), c * L.k⟩

def sub (L M : Lin) : Lin := L.add (M.scale (-1))

/-- The coefficient of `a`, its terms summed. -/
def coeff (a : Atom) : List (Atom × Int) → Int
  | [] => 0
  | (b, c) :: ts => (if b = a then c else 0) + coeff a ts

/-- The form without `a`. -/
def drop (a : Atom) (L : Lin) : Lin := ⟨L.terms.filter (·.1 ≠ a), L.k⟩

/-- Terms merged by atom, those whose coefficient is zero left out. -/
def insertT (a : Atom) (c : Int) : List (Atom × Int) → List (Atom × Int)
  | [] => if c = 0 then [] else [(a, c)]
  | (b, d) :: ts => if b = a then (if c + d = 0 then ts else (b, c + d) :: ts) else (b, d) :: insertT a c ts

def normT : List (Atom × Int) → List (Atom × Int)
  | [] => []
  | (a, c) :: ts => insertT a c (normT ts)

def norm (L : Lin) : Lin := ⟨normT L.terms, L.k⟩

/-- The slots a form reads. -/
def slots (L : Lin) : List R :=
  L.terms.filterMap fun | (.slot r, _) => some r | (.iun _ r, _) => some r | _ => none

/-- The regions whose sizes a form reads. -/
def regions (L : Lin) : List Region := L.terms.filterMap fun | (.size g, _) => some g | _ => none

theorem evalT_append (Γ : Env) (w : World) (s t : List (Atom × Int)) :
    evalT Γ w (s ++ t) = evalT Γ w s + evalT Γ w t := by
  induction s with
  | nil => simp [evalT]
  | cons p s ih => obtain ⟨a, c⟩ := p; simp only [List.cons_append, evalT, ih]; omega

theorem eval_add (Γ : Env) (w : World) (L M : Lin) : (L.add M).eval Γ w = L.eval Γ w + M.eval Γ w := by
  simp only [add, eval, evalT_append]; omega

theorem evalT_map_scale (Γ : Env) (w : World) (c : Int) (t : List (Atom × Int)) :
    evalT Γ w (t.map fun (a, d) => (a, c * d)) = c * evalT Γ w t := by
  induction t with
  | nil => simp [evalT]
  | cons p t ih =>
      obtain ⟨a, d⟩ := p
      simp only [List.map_cons, evalT, ih, Int.mul_add, Int.mul_assoc]

theorem eval_scale (Γ : Env) (w : World) (c : Int) (L : Lin) : (L.scale c).eval Γ w = c * L.eval Γ w := by
  simp only [scale, eval, evalT_map_scale, Int.mul_add]

theorem eval_sub (Γ : Env) (w : World) (L M : Lin) : (L.sub M).eval Γ w = L.eval Γ w - M.eval Γ w := by
  simp only [sub, eval_add, eval_scale]; omega

theorem eval_const (Γ : Env) (w : World) (k : Int) : (const k).eval Γ w = k := by
  simp [const, eval, evalT]

theorem eval_atom (Γ : Env) (w : World) (a : Atom) : (atom a).eval Γ w = a.val Γ w := by
  simp [atom, eval, evalT]

theorem evalT_split (Γ : Env) (w : World) (a : Atom) :
    ∀ t : List (Atom × Int), evalT Γ w t = coeff a t * a.val Γ w + evalT Γ w (t.filter (·.1 ≠ a))
  | [] => by simp [evalT, coeff]
  | (b, c) :: t => by
      have ih := evalT_split Γ w a t
      simp only [ne_eq] at ih ⊢
      by_cases e : b = a
      · subst e; simp only [evalT, coeff, List.filter_cons, not_true_eq_false,
          decide_false, Bool.false_eq_true, ↓reduceIte]
        rw [ih, Int.add_mul]; omega
      · simp only [evalT, coeff, List.filter_cons, e, not_false_eq_true, decide_true,
          ↓reduceIte, Int.zero_add]
        rw [ih]; omega

theorem eval_split (Γ : Env) (w : World) (a : Atom) (L : Lin) :
    L.eval Γ w = coeff a L.terms * a.val Γ w + (L.drop a).eval Γ w := by
  simp only [eval, drop]; rw [evalT_split Γ w a L.terms]; omega

theorem evalT_insertT (Γ : Env) (w : World) (a : Atom) (c : Int) :
    ∀ t : List (Atom × Int), evalT Γ w (insertT a c t) = c * a.val Γ w + evalT Γ w t
  | [] => by
      simp only [insertT]; split
      · rename_i h; subst h; simp [evalT]
      · simp [evalT]
  | (b, d) :: t => by
      simp only [insertT]
      split
      · rename_i e
        subst e
        split
        · rename_i h
          simp only [evalT]
          have : c * b.val Γ w + d * b.val Γ w = 0 := by rw [← Int.add_mul, h, Int.zero_mul]
          omega
        · simp only [evalT, Int.add_mul]; omega
      · simp only [evalT, evalT_insertT Γ w a c t]; omega

theorem evalT_normT (Γ : Env) (w : World) : ∀ t : List (Atom × Int), evalT Γ w (normT t) = evalT Γ w t
  | [] => rfl
  | (a, c) :: t => by simp only [normT, evalT_insertT, evalT, evalT_normT Γ w t]

theorem eval_norm (Γ : Env) (w : World) (L : Lin) : L.norm.eval Γ w = L.eval Γ w := by
  simp only [norm, eval, evalT_normT]

/-- `L` with `a` replaced by `D`. -/
def subst (L : Lin) (a : Atom) (D : Lin) : Lin := (L.drop a).add (D.scale (coeff a L.terms))

theorem eval_subst {Γ : Env} {w : World} {L D : Lin} {a : Atom} (h : a.val Γ w = D.eval Γ w) :
    (L.subst a D).eval Γ w = L.eval Γ w := by
  rw [subst, eval_add, eval_scale, eval_split Γ w a L, h]; omega

theorem eval_subst_mod {Γ : Env} {w : World} {L D : Lin} {a : Atom}
    (h : a.val Γ w % 2 ^ 64 = D.eval Γ w % 2 ^ 64) :
    (L.subst a D).eval Γ w % 2 ^ 64 = L.eval Γ w % 2 ^ 64 := by
  rw [subst, eval_add, eval_scale, eval_split Γ w a L]
  have hc : (coeff a L.terms * D.eval Γ w) % 2 ^ 64 = (coeff a L.terms * a.val Γ w) % 2 ^ 64 := by
    rw [Int.mul_emod, ← h, ← Int.mul_emod]
  omega

end Lin

/-- What the checker knows of a slot beyond its range: a form its bits equal
    modulo `2 ^ 64`, and forms bounding `s` times it from above. -/
structure Sym where
  dfn : Option Lin := none
  ubs : List (Nat × Lin) := []
  deriving DecidableEq, Repr, Inhabited

/-- What a `Sym` says of a slot whose bits, as a number, are `x`. -/
def Sym.Holds (σ : Sym) (Γ : Env) (w : World) (x : Int) : Prop :=
  (∀ D, σ.dfn = some D → x % 2 ^ 64 = D.eval Γ w % 2 ^ 64) ∧
    ∀ p ∈ σ.ubs, (p.1 : Int) * x ≤ p.2.eval Γ w

/-- The slots a `Sym` reads. -/
def Sym.slots (σ : Sym) : List R :=
  (σ.dfn.map Lin.slots).getD [] ++ (σ.ubs.map fun p => p.2.slots).flatten

/-- The regions whose sizes a `Sym` reads. -/
def Sym.regions (σ : Sym) : List Region :=
  (σ.dfn.map Lin.regions).getD [] ++ (σ.ubs.map fun p => p.2.regions).flatten

theorem Sym.holds_blank (Γ : Env) (w : World) (x : Int) : ({} : Sym).Holds Γ w x := by
  unfold Sym.Holds; exact ⟨fun _ h => by simp at h, fun _ h => by simp at h⟩

-- ---------------------------------------------------------------------------
-- The prover
-- ---------------------------------------------------------------------------

/-- What the prover reads of a state: a range for each slot's bits, each
    slot's `Sym`, and a least size for each region. -/
structure Ctx where
  rng : R → Int × Int
  sym : R → Sym
  sizeLo : Region → Int

/-- What a context says is true where slots hold `Γ` and the world is `w`. -/
def Ctx.Sound (C : Ctx) (Γ : Env) (w : World) : Prop :=
  (∀ r, (C.rng r).1 ≤ slotVal Γ r ∧ slotVal Γ r ≤ (C.rng r).2) ∧
    (∀ r, (C.sym r).Holds Γ w (slotVal Γ r)) ∧ ∀ g, C.sizeLo g ≤ w.mem.sizes g

namespace Ctx

def atomHi (C : Ctx) : Atom → Option Int
  | .slot r => some (C.rng r).2
  | .size _ => none
  | .iun _ _ => some (2 ^ 64 - 1)

def atomLo (C : Ctx) : Atom → Option Int
  | .slot r => some (C.rng r).1
  | .size g => some (C.sizeLo g)
  | .iun _ _ => some 0

/-- An upper bound of a sum of terms, from the atoms' bounds. -/
def hiT (C : Ctx) : List (Atom × Int) → Option Int
  | [] => some 0
  | (a, c) :: ts =>
      match C.hiT ts, (if 0 ≤ c then C.atomHi a else C.atomLo a) with
      | some s, some b => some (c * b + s)
      | _, _ => none

/-- A lower bound of a sum of terms, from the atoms' bounds. -/
def loT (C : Ctx) : List (Atom × Int) → Option Int
  | [] => some 0
  | (a, c) :: ts =>
      match C.loT ts, (if 0 ≤ c then C.atomLo a else C.atomHi a) with
      | some s, some b => some (c * b + s)
      | _, _ => none

def hiOf (C : Ctx) (L : Lin) : Option Int := (C.hiT L.terms).map (· + L.k)

def loOf (C : Ctx) (L : Lin) : Option Int := (C.loT L.terms).map (· + L.k)

theorem atomHi_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) {a : Atom} {h : Int}
    (e : C.atomHi a = some h) : a.val Γ w ≤ h := by
  cases a with
  | slot r => simp only [atomHi, Option.some.injEq] at e; subst e; exact (hC.1 r).2
  | size g => cases e
  | iun k r =>
      simp only [atomHi, Option.some.injEq] at e; subst e
      have := iunVal_lt k (slotVal Γ r)
      simp only [Atom.val]; omega

theorem atomLo_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) {a : Atom} {l : Int}
    (e : C.atomLo a = some l) : l ≤ a.val Γ w := by
  cases a with
  | slot r => simp only [atomLo, Option.some.injEq] at e; subst e; exact (hC.1 r).1
  | size g => simp only [atomLo, Option.some.injEq] at e; subst e; exact hC.2.2 g
  | iun k r => simp only [atomLo, Option.some.injEq] at e; subst e; exact Atom.val_nonneg Γ w _

theorem hiT_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) :
    ∀ {t : List (Atom × Int)} {h : Int}, C.hiT t = some h → Lin.evalT Γ w t ≤ h
  | [], h, e => by simp only [hiT, Option.some.injEq] at e; subst e; simp [Lin.evalT]
  | (a, c) :: t, h, e => by
      simp only [hiT] at e
      split at e
      · rename_i s b hs hb
        cases e
        have ih := hiT_sound hC hs
        simp only [Lin.evalT]
        split at hb
        · rename_i hc
          have := atomHi_sound hC hb
          have := Int.mul_le_mul_of_nonneg_left this hc
          omega
        · rename_i hc
          have := atomLo_sound hC hb
          have := Int.mul_le_mul_of_nonpos_left (Int.le_of_lt (Int.not_le.mp hc)) this
          omega
      · cases e

theorem loT_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) :
    ∀ {t : List (Atom × Int)} {l : Int}, C.loT t = some l → l ≤ Lin.evalT Γ w t
  | [], l, e => by simp only [loT, Option.some.injEq] at e; subst e; simp [Lin.evalT]
  | (a, c) :: t, l, e => by
      simp only [loT] at e
      split at e
      · rename_i s b hs hb
        cases e
        have ih := loT_sound hC hs
        simp only [Lin.evalT]
        split at hb
        · rename_i hc
          have := atomLo_sound hC hb
          have := Int.mul_le_mul_of_nonneg_left this hc
          omega
        · rename_i hc
          have := atomHi_sound hC hb
          have := Int.mul_le_mul_of_nonpos_left (Int.le_of_lt (Int.not_le.mp hc)) this
          omega
      · cases e

theorem hiOf_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) {L : Lin} {h : Int}
    (e : C.hiOf L = some h) : L.eval Γ w ≤ h := by
  simp only [hiOf, Option.map_eq_some_iff] at e
  obtain ⟨s, hs, rfl⟩ := e
  have := hiT_sound hC hs
  simp only [Lin.eval]; omega

theorem loOf_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) {L : Lin} {l : Int}
    (e : C.loOf L = some l) : l ≤ L.eval Γ w := by
  simp only [loOf, Option.map_eq_some_iff] at e
  obtain ⟨s, hs, rfl⟩ := e
  have := loT_sound hC hs
  simp only [Lin.eval]; omega

/-- A form the atoms' bounds show is at most zero. -/
def leZero (C : Ctx) (L : Lin) : Bool :=
  match C.hiOf L with
  | some h => decide (h ≤ 0)
  | none => false

theorem leZero_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) {L : Lin} (h : C.leZero L = true) :
    L.eval Γ w ≤ 0 := by
  unfold leZero at h
  split at h
  · rename_i b hb; have := hiOf_sound hC hb; simp at h; omega
  · cases h

/-- Whether a form congruent to slot `r`'s bits modulo `2 ^ 64` equals them:
    it lies below `2 ^ 64` and above `-2 ^ 64`, and not below zero unless the
    slot's range rules out the wrap a negative form would make. -/
def exactForm (C : Ctx) (r : R) (D : Lin) : Bool :=
  match C.loOf D, C.hiOf D with
  | some l, some h => decide ((0 ≤ l ∨ (C.rng r).2 < l + 2 ^ 64) ∧ -2 ^ 64 < l ∧ h < 2 ^ 64)
  | _, _ => false

theorem exactForm_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) {r : R} {D : Lin}
    (hm : slotVal Γ r % 2 ^ 64 = D.eval Γ w % 2 ^ 64) (h : C.exactForm r D = true) :
    slotVal Γ r = D.eval Γ w := by
  unfold exactForm at h
  split at h
  · rename_i l hh hl hhh
    simp only [decide_eq_true_eq] at h
    have h1 := loOf_sound hC hl
    have h2 := hiOf_sound hC hhh
    have h3 := slotVal_nonneg Γ r
    have h4 := slotVal_lt Γ r
    have h5 := (hC.1 r).2
    rw [Int.emod_eq_of_lt h3 h4] at hm
    omega
  · cases h

/-- The latest slot a form reads for which `f` answers. -/
def latestSlot (L : Lin) (f : R → Option Lin) : Option (R × Lin) :=
  L.terms.foldl (fun best (a, _) =>
    match a with
    | .slot r =>
        match f r with
        | some D =>
            match best with
            | some (r', _) => if r' < r then some (r, D) else best
            | none => some (r, D)
        | none => best
    | _ => best) none

theorem latestSlot_spec {L : Lin} {f : R → Option Lin} :
    ∀ {r : R} {D : Lin}, latestSlot L f = some (r, D) → f r = some D := by
  unfold latestSlot
  suffices h : ∀ (t : List (Atom × Int)) (best : Option (R × Lin)),
      (∀ r D, best = some (r, D) → f r = some D) →
      ∀ r D, t.foldl (fun best (a, _) =>
        match a with
        | .slot r =>
            match f r with
            | some D =>
                match best with
                | some (r', _) => if r' < r then some (r, D) else best
                | none => some (r, D)
            | none => best
        | _ => best) best = some (r, D) → f r = some D from
    fun {r D} e => h L.terms none (fun _ _ e => by cases e) r D e
  intro t
  induction t with
  | nil => intro best hb r D e; exact hb r D e
  | cons p t ih =>
      intro best hb r D e
      obtain ⟨a, c⟩ := p
      rw [List.foldl_cons] at e
      refine ih _ ?_ r D e
      intro r' D' e'
      cases a with
      | slot q =>
          dsimp only at e'
          split at e'
          · rename_i D0 hD0
            split at e'
            · split at e'
              · cases e'; exact hD0
              · exact hb r' D' e'
            · cases e'; exact hD0
          · exact hb r' D' e'
      | size g => exact hb r' D' e'
      | iun k q => exact hb r' D' e'

/-- A form with its slots' definitions put in, latest first: congruent to it
    modulo `2 ^ 64`. -/
def deepen (C : Ctx) : Nat → Lin → Lin
  | 0, L => L
  | n + 1, L =>
      let L := L.norm
      match latestSlot L fun r => (C.sym r).dfn with
      | some (r, D) => C.deepen n (L.subst (.slot r) D)
      | none => L

theorem deepen_mod {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) :
    ∀ (n : Nat) (L : Lin), (C.deepen n L).eval Γ w % 2 ^ 64 = L.eval Γ w % 2 ^ 64
  | 0, _ => rfl
  | n + 1, L => by
      simp only [deepen]
      split
      · rename_i r D hp
        rw [deepen_mod hC n, Lin.eval_subst_mod (by simpa [Atom.val] using (hC.2.1 r).1 D (latestSlot_spec hp)),
          Lin.eval_norm]
      · rw [Lin.eval_norm]

/-- A form's terms with each slot the context knows exactly put in as its
    number: the terms left and the number they come to. -/
def foldT (C : Ctx) : List (Atom × Int) → List (Atom × Int) × Int
  | [] => ([], 0)
  | (a, c) :: ts =>
      let p := C.foldT ts
      match a with
      | .slot r => if (C.rng r).1 = (C.rng r).2 then (p.1, p.2 + c * (C.rng r).1) else ((a, c) :: p.1, p.2)
      | _ => ((a, c) :: p.1, p.2)

/-- A form with each slot the context knows exactly put in as its number. -/
def foldKnown (C : Ctx) (L : Lin) : Lin := ⟨(C.foldT L.terms).1, L.k + (C.foldT L.terms).2⟩

theorem foldT_eval {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) :
    ∀ ts : List (Atom × Int), Lin.evalT Γ w (C.foldT ts).1 + (C.foldT ts).2 = Lin.evalT Γ w ts
  | [] => by simp [foldT, Lin.evalT]
  | (a, c) :: ts => by
      have ih := foldT_eval hC ts
      cases a with
      | slot r =>
          simp only [foldT]
          split
          · rename_i he
            have h := hC.1 r
            have e : slotVal Γ r = (C.rng r).1 := by omega
            simp only [Lin.evalT, Atom.val, e]; omega
          · simp only [Lin.evalT]; omega
      | size g => simp only [foldT, Lin.evalT]; omega
      | iun k q => simp only [foldT, Lin.evalT]; omega

theorem foldKnown_eval {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) (L : Lin) :
    (C.foldKnown L).eval Γ w = L.eval Γ w := by
  have := foldT_eval hC L.terms
  simp only [foldKnown, Lin.eval]; omega

/-- How deep a definition is unfolded to find an exact one. -/
def defDepth : Nat := 6

/-- A slot's definition, where it is exact: as it stands, or with the
    definitions it reads put in. -/
def exactDef (C : Ctx) (r : R) : Option Lin :=
  match (C.sym r).dfn with
  | some D =>
      if C.exactForm r D then some D
      else if C.exactForm r (C.deepen defDepth D) then some (C.deepen defDepth D) else none
  | none => none

theorem exactDef_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) {r : R} {D : Lin}
    (h : C.exactDef r = some D) : slotVal Γ r = D.eval Γ w := by
  unfold exactDef at h
  split at h
  · rename_i D0 hD0
    have hm := (hC.2.1 r).1 D0 hD0
    split at h
    · rename_i he; cases h; exact exactForm_sound hC hm he
    · split at h
      · rename_i he; cases h; exact exactForm_sound hC (by rw [deepen_mod hC]; exact hm) he
      · cases h
  · cases h

/-- The slot a step rewrites: the latest slot the form reads whose
    definition is exact, or which has a bound and a positive coefficient. -/
def pivot (C : Ctx) (L : Lin) : Option (R × Int) :=
  L.terms.foldl (fun best (a, c) =>
    match a with
    | .slot r =>
        if (C.exactDef r).isSome || (decide (0 < c) && !(C.sym r).ubs.isEmpty) then
          match best with
          | some (r', _) => if r' < r then some (r, c) else best
          | none => some (r, c)
        else best
    | _ => best) none

/-- `L`, scaled by `s / g`, with `c · r` replaced by `(c / g) · U`, where
    `s · r ≤ U` and `g` divides both: no smaller than the scaled form. -/
def bump (L : Lin) (r : R) (c : Int) (s : Nat) (U : Lin) : Lin :=
  ((L.drop (.slot r)).scale ((s / Nat.gcd c.toNat s : Nat) : Int)).add
    (U.scale ((c.toNat / Nat.gcd c.toNat s : Nat) : Int))

theorem bump_sound {Γ : Env} {w : World} {L U : Lin} {r : R} {s : Nat} {c : Int}
    (hc : Lin.coeff (.slot r) L.terms = c) (hc0 : 0 < c) (hu : (s : Int) * slotVal Γ r ≤ U.eval Γ w) :
    ((s / Nat.gcd c.toNat s : Nat) : Int) * L.eval Γ w ≤ (bump L r c s U).eval Γ w := by
  rw [bump, Lin.eval_add, Lin.eval_scale, Lin.eval_scale, Lin.eval_split Γ w (.slot r) L, hc]
  simp only [Atom.val]
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le (Int.le_of_lt hc0)
  simp only [Int.toNat_natCast]
  have hg1 : Nat.gcd n s ∣ n := Nat.gcd_dvd_left n s
  have hg2 : Nat.gcd n s ∣ s := Nat.gcd_dvd_right n s
  generalize Nat.gcd n s = g at hg1 hg2
  have e1 : (s / g) * n = (n / g) * s := by
    have h1 := Nat.div_mul_cancel hg2
    have h2 := Nat.mul_div_cancel' hg1
    calc (s / g) * n = (s / g) * (g * (n / g)) := by rw [h2]
      _ = (s / g * g) * (n / g) := by rw [Nat.mul_assoc]
      _ = s * (n / g) := by rw [h1]
      _ = (n / g) * s := Nat.mul_comm _ _
  have e1' : ((s / g : Nat) : Int) * (n : Int) = ((n / g : Nat) : Int) * (s : Int) := by exact_mod_cast e1
  generalize slotVal Γ r = v at hu ⊢
  generalize U.eval Γ w = u at hu ⊢
  generalize (L.drop (.slot r)).eval Γ w = E
  have h2 : ((n / g : Nat) : Int) * ((s : Int) * v) ≤ ((n / g : Nat) : Int) * u :=
    Int.mul_le_mul_of_nonneg_left hu (Int.natCast_nonneg _)
  have e2 : ((s / g : Nat) : Int) * ((n : Int) * v) = ((n / g : Nat) : Int) * ((s : Int) * v) := by
    rw [← Int.mul_assoc, e1', Int.mul_assoc]
  rw [Int.mul_add, e2]
  omega

/-- How many forms one proof may look at, across every bound it tries. -/
def proveBudget : Nat := 64

/-- **Whether a form is at most zero**: by the atoms' bounds, or, with `fuel`
    steps left and `budget` forms still to look at, by rewriting its pivot:
    by its exact definition, or by each of its bounds in turn that reads no
    slot already rewritten on the way here (`seen`). What is left of the
    budget comes back with the answer. -/
def proveB (C : Ctx) : Nat → Nat → List R → Lin → Bool × Nat
  | 0, b, _, L => (C.leZero L, b)
  | _ + 1, 0, _, L => (C.leZero L, 0)
  | fuel + 1, b + 1, seen, L =>
      let L := L.norm
      if C.leZero L then (true, b) else
        match C.pivot L with
        | none => (false, b)
        | some (r, c) =>
            match C.exactDef r with
            | some D => C.proveB fuel b (r :: seen) (L.subst (.slot r) D)
            | none =>
                if 0 < c ∧ Lin.coeff (.slot r) L.terms = c then
                  (C.sym r).ubs.attach.foldl (fun acc p =>
                    if acc.1 then acc
                    else if 0 < p.1.1 ∧ p.1.2.slots.all (fun y => !(r :: seen).contains y) then
                      C.proveB fuel acc.2 (r :: seen) (bump L r c p.1.1 p.1.2)
                    else acc) (false, b)
                else (false, b)

/-- Whether a form is at most zero, by `proveB` with the whole budget. -/
def prove (C : Ctx) (fuel : Nat) (L : Lin) : Bool := (C.proveB fuel proveBudget [] L).1

/-- **The prover is sound.** -/
theorem proveB_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) :
    ∀ {fuel b : Nat} {seen : List R} {L : Lin}, (C.proveB fuel b seen L).1 = true → L.eval Γ w ≤ 0
  | 0, b, _, L, h => leZero_sound hC h
  | _ + 1, 0, _, L, h => leZero_sound hC h
  | fuel + 1, b + 1, seen, L, h => by
      simp only [proveB] at h
      rw [← Lin.eval_norm]
      split at h
      · rename_i hz; exact leZero_sound hC hz
      · split at h
        · cases h
        · rename_i r c _
          split at h
          · rename_i D hD
            have := proveB_sound hC h
            rwa [Lin.eval_subst (by simpa [Atom.val] using exactDef_sound hC hD)] at this
          · split at h
            · rename_i hcc
              obtain ⟨hc0, hcoef⟩ := hcc
              -- every bound tried that succeeds shows the form
              have step : ∀ (s : Nat) (U : Lin), (s, U) ∈ (C.sym r).ubs → 0 < s →
                  ∀ b' sn, (C.proveB fuel b' sn (bump L.norm r c s U)).1 = true → L.norm.eval Γ w ≤ 0 := by
                intro s U hmem hs b' sn hp
                have h1 := proveB_sound hC hp
                have h2 := bump_sound hcoef hc0 ((hC.2.1 r).2 _ hmem)
                have hpos : (0 : Int) < ((s / Nat.gcd c.toNat s : Nat) : Int) := by
                  have := Nat.div_pos (Nat.gcd_le_right (m := c.toNat) hs) (Nat.gcd_pos_of_pos_right c.toNat hs)
                  exact_mod_cast this
                exact Int.le_of_mul_le_mul_left (by rw [Int.mul_zero]; exact Int.le_trans h2 h1) hpos
              suffices hf : ∀ (l : List {x // x ∈ (C.sym r).ubs}) (acc : Bool × Nat),
                  (acc.1 = true → L.norm.eval Γ w ≤ 0) →
                  (l.foldl (fun acc p =>
                    if acc.1 then acc
                    else if 0 < p.1.1 ∧ p.1.2.slots.all (fun y => !(r :: seen).contains y) then
                      C.proveB fuel acc.2 (r :: seen) (bump L.norm r c p.1.1 p.1.2)
                    else acc)
                    acc).1 = true → L.norm.eval Γ w ≤ 0 from
                hf _ (false, b) (fun e => by cases e) h
              intro l
              induction l with
              | nil => intro acc hacc h; exact hacc h
              | cons p l ih =>
                  intro acc hacc h
                  rw [List.foldl_cons] at h
                  refine ih _ ?_ h
                  intro e
                  split at e
                  · exact hacc e
                  · split at e
                    · rename_i hs; exact step _ _ p.2 hs.1 _ _ e
                    · exact hacc e
            · cases h

theorem prove_sound {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) {fuel : Nat} {L : Lin}
    (h : C.prove fuel L = true) : L.eval Γ w ≤ 0 :=
  proveB_sound hC h

/-- The latest slot a form reads whose definition is exact. -/
def pivotDef (C : Ctx) (L : Lin) : Option (R × Lin) := latestSlot L C.exactDef

theorem pivotDef_sound {C : Ctx} {L : Lin} {r : R} {D : Lin} (h : C.pivotDef L = some (r, D)) :
    C.exactDef r = some D := latestSlot_spec h

/-- A form with its slots' exact definitions put in for them, latest first. -/
def expand (C : Ctx) : Nat → Lin → Lin
  | 0, L => L
  | n + 1, L =>
      let L := L.norm
      match C.pivotDef L with
      | some (r, D) => C.expand n (L.subst (.slot r) D)
      | none => L

theorem expand_eval {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) :
    ∀ (n : Nat) (L : Lin), (C.expand n L).eval Γ w = L.eval Γ w
  | 0, _ => rfl
  | n + 1, L => by
      simp only [expand]
      split
      · rename_i r D hp
        rw [expand_eval hC n, Lin.eval_subst (by simpa [Atom.val] using exactDef_sound hC (pivotDef_sound hp)),
          Lin.eval_norm]
      · exact Lin.eval_norm Γ w L

end Ctx

end AlgorithmLib.Check
