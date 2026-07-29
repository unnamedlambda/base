import AlgorithmLib.ML.Num

namespace AlgorithmLib.ML

open NumOps

/-- Scalar-valued expressions over `Γ` inputs.  `Γ` is an *index*, not a
    parameter, because `letE` extends it. -/
inductive Expr : Nat → Type where
  | var   : {Γ : Nat} → Fin Γ → Expr Γ
  | lit   : {Γ : Nat} → Nat → Expr Γ
  | add   : {Γ : Nat} → Expr Γ → Expr Γ → Expr Γ
  | mul   : {Γ : Nat} → Expr Γ → Expr Γ → Expr Γ
  | neg   : {Γ : Nat} → Expr Γ → Expr Γ
  | inv   : {Γ : Nat} → Expr Γ → Expr Γ
  | exp   : {Γ : Nat} → Expr Γ → Expr Γ
  | rsqrt : {Γ : Nat} → Expr Γ → Expr Γ
  /-- Finite reduction.  The fold order is **committed to** (left fold over
      `finRange`), matching the sequential accumulation the kernel performs.
      That is what makes the machine refinement an exact equality rather than
      an up-to-reassociation statement. -/
  | sum   : {Γ : Nat} → (n : Nat) → (Fin n → Expr Γ) → Expr Γ
  /-- `let x = a in b`, where `x` is variable index `Γ` inside `b`.
      The reason terms stay linear rather than exponential. -/
  | letE  : {Γ : Nat} → Expr Γ → Expr (Γ + 1) → Expr Γ

namespace Expr

def sub {Γ : Nat} (a b : Expr Γ) : Expr Γ := .add a (.neg b)
def div {Γ : Nat} (a b : Expr Γ) : Expr Γ := .mul a (.inv b)

/-- A model with `out` scalar outputs. -/
abbrev Prog (Γ out : Nat) := Fin out → Expr Γ

end Expr

/-- Expressions with no `letE`.  `grad` is proven correct on this fragment.
    Sharing is supported on the forward/compile path (`compileE`), which is
    where the exponential blowup actually bit; correct AD *through* a binding
    requires de Bruijn weakening and is the next item. -/
def LetFree : {Γ : Nat} → Expr Γ → Prop
  | _, .var _    => True
  | _, .lit _    => True
  | _, .add a b  => LetFree a ∧ LetFree b
  | _, .mul a b  => LetFree a ∧ LetFree b
  | _, .neg a    => LetFree a
  | _, .inv a    => LetFree a
  | _, .exp a    => LetFree a
  | _, .rsqrt a  => LetFree a
  | _, .sum _ f  => ∀ j, LetFree (f j)
  | _, .letE _ _ => False

-- ---------------------------------------------------------------------------
-- Environments
-- ---------------------------------------------------------------------------

/-- Extend an environment with the value of a `letE` binding, at index `Γ`. -/
def extend {α : Type} {Γ : Nat} (env : Fin Γ → α) (v : α) : Fin (Γ + 1) → α :=
  fun k => if h : k.val < Γ then env ⟨k.val, h⟩ else v

@[simp] theorem extend_lt {α : Type} {Γ : Nat} (env : Fin Γ → α) (v : α)
    (k : Fin (Γ + 1)) (h : k.val < Γ) : extend env v k = env ⟨k.val, h⟩ := by
  simp [extend, h]

@[simp] theorem extend_last {α : Type} {Γ : Nat} (env : Fin Γ → α) (v : α) :
    extend env v ⟨Γ, Nat.lt_succ_self Γ⟩ = v := by
  simp [extend]

-- ---------------------------------------------------------------------------
-- Denotation — one function, every carrier
-- ---------------------------------------------------------------------------

/-- Meaning of an expression in any numeric carrier.

    `denote (R := Float)` is the exact semantics the machine is proven against.
    `denote (R := ℝ)` is the mathematical semantics `grad_hasDerivAt` speaks
    about.  Same function, same term, different instance — which is why the
    model is only ever written once. -/
def denote {R : Type} [NumOps R] : {Γ : Nat} → (Fin Γ → R) → Expr Γ → R
  | _, env, .var i    => env i
  | _, _,   .lit n    => ofNat n
  | _, env, .add a b  => add (denote env a) (denote env b)
  | _, env, .mul a b  => mul (denote env a) (denote env b)
  | _, env, .neg a    => neg (denote env a)
  | _, env, .inv a    => inv (denote env a)
  | _, env, .exp a    => exp (denote env a)
  | _, env, .rsqrt a  => rsqrt (denote env a)
  | _, env, .sum n f  => (List.finRange n).foldl (fun acc j => add acc (denote env (f j))) zero
  | _, env, .letE a b => denote (extend env (denote env a)) b

variable {R : Type} [NumOps R] {Γ : Nat}

@[simp] theorem denote_var (env : Fin Γ → R) (i : Fin Γ) :
    denote env (.var i) = env i := rfl
@[simp] theorem denote_add (env : Fin Γ → R) (a b : Expr Γ) :
    denote env (.add a b) = add (denote env a) (denote env b) := rfl
@[simp] theorem denote_mul (env : Fin Γ → R) (a b : Expr Γ) :
    denote env (.mul a b) = mul (denote env a) (denote env b) := rfl
@[simp] theorem denote_neg (env : Fin Γ → R) (a : Expr Γ) :
    denote env (.neg a) = neg (denote env a) := rfl
@[simp] theorem denote_inv (env : Fin Γ → R) (a : Expr Γ) :
    denote env (.inv a) = inv (denote env a) := rfl
@[simp] theorem denote_exp (env : Fin Γ → R) (a : Expr Γ) :
    denote env (.exp a) = exp (denote env a) := rfl
@[simp] theorem denote_rsqrt (env : Fin Γ → R) (a : Expr Γ) :
    denote env (.rsqrt a) = rsqrt (denote env a) := rfl
@[simp] theorem denote_lit (env : Fin Γ → R) (n : Nat) :
    denote env (.lit n : Expr Γ) = ofNat n := rfl
@[simp] theorem denote_sum (env : Fin Γ → R) (n : Nat) (f : Fin n → Expr Γ) :
    denote env (.sum n f)
      = (List.finRange n).foldl (fun acc j => add acc (denote env (f j))) zero := rfl
@[simp] theorem denote_letE (env : Fin Γ → R) (a : Expr Γ) (b : Expr (Γ + 1)) :
    denote env (.letE a b) = denote (extend env (denote env a)) b := rfl


-- ---------------------------------------------------------------------------
-- Renaming, and weakening under a binder
-- ---------------------------------------------------------------------------

/-! Differentiating through `letE` needs to move a term into a context with one
    more variable.  Substitution would want the binder at index 0; weakening
    works with the binder at the end, which is the convention here. -/

/-- Lift a renaming under one binder. -/
def liftRen {Γ Δ : Nat} (ρ : Fin Γ → Fin Δ) : Fin (Γ + 1) → Fin (Δ + 1) :=
  fun k => if h : k.val < Γ then ⟨(ρ ⟨k.val, h⟩).val, Nat.lt_succ_of_lt (ρ ⟨k.val, h⟩).isLt⟩
           else ⟨Δ, Nat.lt_succ_self Δ⟩

def rename : {Γ Δ : Nat} → (Fin Γ → Fin Δ) → Expr Γ → Expr Δ
  | _, _, ρ, .var i    => .var (ρ i)
  | _, _, _, .lit n    => .lit n
  | _, _, ρ, .add a b  => .add (rename ρ a) (rename ρ b)
  | _, _, ρ, .mul a b  => .mul (rename ρ a) (rename ρ b)
  | _, _, ρ, .neg a    => .neg (rename ρ a)
  | _, _, ρ, .inv a    => .inv (rename ρ a)
  | _, _, ρ, .exp a    => .exp (rename ρ a)
  | _, _, ρ, .rsqrt a  => .rsqrt (rename ρ a)
  | _, _, ρ, .sum n f  => .sum n (fun j => rename ρ (f j))
  | _, _, ρ, .letE a b => .letE (rename ρ a) (rename (liftRen ρ) b)

/-- Move a term into a context with one more variable at the end. -/
def wk {Γ : Nat} (e : Expr Γ) : Expr (Γ + 1) :=
  rename (fun i => ⟨i.val, Nat.lt_succ_of_lt i.isLt⟩) e

/-- Renaming is meaning-preserving: it just re-indexes the environment. -/
theorem denote_rename {R : Type} [NumOps R] :
    ∀ {Γ Δ : Nat} (ρ : Fin Γ → Fin Δ) (e : Expr Γ) (env : Fin Δ → R),
      denote env (rename ρ e) = denote (fun i => env (ρ i)) e := by
  intro Γ Δ ρ e
  induction e generalizing Δ with
  | var i => intro env; rfl
  | lit n => intro env; rfl
  | add a b iha ihb => intro env; show add _ _ = add _ _; rw [iha ρ env, ihb ρ env]
  | mul a b iha ihb => intro env; show mul _ _ = mul _ _; rw [iha ρ env, ihb ρ env]
  | neg a ih   => intro env; show neg _ = neg _; rw [ih ρ env]
  | inv a ih   => intro env; show inv _ = inv _; rw [ih ρ env]
  | exp a ih   => intro env; show exp _ = exp _; rw [ih ρ env]
  | rsqrt a ih => intro env; show rsqrt _ = rsqrt _; rw [ih ρ env]
  | sum n f ih =>
      intro env
      show (List.finRange n).foldl (fun acc j => add acc (denote env (rename ρ (f j)))) zero = _
      have h : ∀ j, denote env (rename ρ (f j)) = denote (fun i => env (ρ i)) (f j) :=
        fun j => ih j ρ env
      simp only [h]; rfl
  | letE a b iha ihb =>
      intro env
      show denote (extend env (denote env (rename ρ a))) (rename (liftRen ρ) b) = _
      rw [ihb (liftRen ρ) (extend env (denote env (rename ρ a))), iha ρ env]
      show denote (fun i => extend env _ (liftRen ρ i)) b = denote (extend _ _) b
      congr 1
      funext i
      show extend env _ (if _ : _ then _ else _) = (if _ : _ then _ else _)
      split
      · next h => simp only [extend, dif_pos (ρ ⟨i.val, h⟩).isLt]
      · simp only [extend, dif_neg (Nat.lt_irrefl Δ)]

/-- Weakening does not change meaning: the new variable is unused. -/
theorem denote_wk {R : Type} [NumOps R] {Γ : Nat} (e : Expr Γ) (env : Fin Γ → R) (v : R) :
    denote (extend env v) (wk e) = denote env e := by
  rw [wk, denote_rename]
  congr 1
  funext i
  simp [extend, i.isLt]

-- ---------------------------------------------------------------------------
-- Surface syntax
-- ---------------------------------------------------------------------------

/-!  Specs should read like the formulas they are.  These instances are pure
     sugar — each is definitionally a constructor, so no theorem anywhere needs
     restating and `simp`/`rfl` see straight through them.

     `x * w + b` now elaborates to `.add (.mul x w) b`; numerals become
     `.lit n` via `OfNat`; subtraction and division are the compound forms the
     spec language already committed to (`a + (−b)`, `a · b⁻¹`) — using them is
     *choosing* those roundings, which is exactly what writing the constructors
     did. -/

instance : Add (Expr Γ) := ⟨.add⟩
instance : Mul (Expr Γ) := ⟨.mul⟩
instance : Neg (Expr Γ) := ⟨.neg⟩
instance : Sub (Expr Γ) := ⟨fun a b => .add a (.neg b)⟩
instance : Div (Expr Γ) := ⟨fun a b => .mul a (.inv b)⟩
instance : Inv (Expr Γ) := ⟨.inv⟩
instance : OfNat (Expr Γ) n := ⟨.lit n⟩

@[simp] theorem hAdd_def (a b : Expr Γ) : a + b = .add a b := rfl
@[simp] theorem hMul_def (a b : Expr Γ) : a * b = .mul a b := rfl
@[simp] theorem neg_def  (a : Expr Γ)   : -a = .neg a := rfl
@[simp] theorem hSub_def (a b : Expr Γ) : a - b = .add a (.neg b) := rfl
@[simp] theorem hDiv_def (a b : Expr Γ) : a / b = .mul a (.inv b) := rfl
@[simp] theorem ofNat_def (n : Nat) : (OfNat.ofNat n : Expr Γ) = .lit n := rfl

/-- Variable `i`, with the bound discharged by `decide` — so specs write
    `x 0 * x 1 + x 2` instead of `.var ⟨0, by decide⟩`. -/
def x {Γ : Nat} (i : Nat) (h : i < Γ := by decide) : Expr Γ := .var ⟨i, h⟩

end AlgorithmLib.ML
