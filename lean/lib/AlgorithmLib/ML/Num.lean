/-!
  # Numeric carriers for the ML expression language

  The model is written once, as an `Expr`, and given meaning by `denote`
  at *any* carrier `R` with a `NumOps R` instance.  Two carriers matter:

  * `Float` — the exact, deterministic interpretation.  This is the one the
    machine refinement (`Computes`) is stated against, so those theorems are
    genuine equalities with no epsilons.
  * `ℝ`     — the mathematical interpretation, used to justify that `grad`
    really is differentiation.  It needs Mathlib and is *not* provided here.

  `NumLaws` collects the algebraic facts the autodiff proof needs.  It is a
  separate class on purpose:

      Float : NumOps      ✓      (floats compute)
      Float : NumLaws     ✗      (floats do not associate)
      ℝ     : NumOps      ✓
      ℝ     : NumLaws     ✓

  That missing instance *is* the float-vs-real gap, expressed as a type-level
  fact rather than a comment.  `grad_correct` is proven for every `NumLaws`
  carrier, so it applies to ℝ the moment Mathlib is available, and provably
  does not apply to `Float`.  See `AlgorithmLib.ML.Assumptions`.
-/

namespace AlgorithmLib.ML

/-- Operations every carrier must provide.  `Float` is an instance; so is ℝ. -/
class NumOps (R : Type) where
  zero  : R
  one   : R
  add   : R → R → R
  mul   : R → R → R
  neg   : R → R
  inv   : R → R
  exp   : R → R
  /-- `2^x`.  A *separate primitive from* `exp`, because the hardware has
      `ex2.approx.f32` and does not have `e^x`.  Keeping both means the machine
      language can name exactly what the silicon computes, and the step from
      `exp` to `ex2` becomes a **declared** rewrite rather than a false
      hypothesis buried inside the lowering. -/
  ex2   : R → R
  rsqrt : R → R
  /-- Decidable comparison.  Making this a *field* rather than adding `max` and
      a select as primitives means `ifGe`/`max` are **defined**, so their
      autodiff laws are derivable instead of assumed. -/
  le    : R → R → Bool
  /-- Carrier image of a numeral.  A field, not a fold: `ofNat 1000000` must
      not cost a million additions at evaluation time. -/
  ofNat : Nat → R

namespace NumOps

variable {R : Type} [NumOps R]

def sub (a b : R) : R := add a (neg b)
def div (a b : R) : R := mul a (inv b)

/-- `if a ≥ b then x else y`. -/
def ifGe (a b x y : R) : R := if le b a then x else y

/-- `max` is a select on its own arguments — not a separate primitive. -/
def max (a b : R) : R := ifGe a b a b

@[simp] theorem ifGe_pos (a b x y : R) (h : le b a = true) : ifGe a b x y = x := by
  simp [ifGe, h]

@[simp] theorem ifGe_neg (a b x y : R) (h : le b a = false) : ifGe a b x y = y := by
  simp [ifGe, h]

end NumOps

open NumOps in
/-- The algebraic facts `grad_correct` needs: a commutative ring.

    Deliberately minimal — every field here is discharged by an actual proof
    obligation at instantiation time, so adding a carrier is honest work. -/
class NumLaws (R : Type) extends NumOps R where
  add_assoc  : ∀ a b c : R, add (add a b) c = add a (add b c)
  add_comm   : ∀ a b : R,   add a b = add b a
  add_zero   : ∀ a : R,     add a zero = a
  add_neg    : ∀ a : R,     add a (neg a) = zero
  mul_assoc  : ∀ a b c : R, mul (mul a b) c = mul a (mul b c)
  mul_comm   : ∀ a b : R,   mul a b = mul b a
  mul_one    : ∀ a : R,     mul a one = a
  mul_zero   : ∀ a : R,     mul a zero = zero
  neg_mul    : ∀ a b : R,   mul (neg a) b = neg (mul a b)
  ofNat_zero : ofNat 0 = (zero : R)
  ofNat_one  : ofNat 1 = (one : R)
  left_distrib : ∀ a b c : R, mul a (add b c) = add (mul a b) (mul a c)

namespace NumLaws

variable {R : Type} [NumLaws R]
open NumOps

theorem zero_add (a : R) : add zero a = a := by
  rw [add_comm]; exact add_zero a

theorem zero_mul (a : R) : mul zero a = zero := by
  rw [mul_comm]; exact mul_zero a

theorem one_mul (a : R) : mul one a = a := by
  rw [mul_comm]; exact mul_one a

theorem mul_neg (a b : R) : mul a (neg b) = neg (mul a b) := by
  rw [mul_comm, neg_mul, mul_comm]

/-- `w * (a * b) = (w * b) * a` — the shape every chain-rule case reduces to,
    where `bp` pushes the local derivative into the weight and `sderiv` leaves
    it on the right. -/
theorem mul_swap_right (w a b : R) : mul (mul w b) a = mul w (mul a b) := by
  rw [mul_assoc, mul_comm b a]

end NumLaws

-- ---------------------------------------------------------------------------
-- The exact carrier
-- ---------------------------------------------------------------------------

/-- **The machine carrier is `Float32`.**

    The GPU register file is binary32.  Modelling it with Lean's `Float`
    (binary64) would make every "exact `Float` equality" theorem state something
    subtly different from what the silicon does, so the machine layer carries
    `Float32` and the spec is denoted at `Float32` alongside it.

    `Expr`/`denote`/`grad` are carrier-generic, so the spec language is
    unchanged — this is a choice of interpretation, not of language.

    `zero` is spelled `ofNat 0` rather than `0.0` **deliberately**.  They are
    the same binary32 value, but `Float32.ofScientific` is opaque, so the two
    spellings are not definitionally equal — and a kernel fold initialised from
    a `.lit (ofNat 0)` then cannot meet `denote`'s `.sum`, whose accumulator
    starts at `zero`.  Writing the instance this way makes the join `rfl`
    instead of an unprovable side goal.  It is also exactly what `NumLaws`
    demands (`ofNat_zero`), so this aligns the computable instance with the law
    an exact carrier would satisfy. -/
instance : NumOps Float32 where
  zero  := Float32.ofNat 0
  one   := 1.0
  add   := Float32.add
  mul   := Float32.mul
  neg   := Float32.neg
  inv   := fun x => 1.0 / x
  exp   := Float32.exp
  ex2   := fun x => Float32.pow 2.0 x
  rsqrt := fun x => 1.0 / Float32.sqrt x
  ofNat := Float32.ofNat
  le    := fun a b => a ≤ b

/-- `Float` computes, so it is a `NumOps`.  It is deliberately **not** a
    `NumLaws`: floating-point addition does not associate, so the autodiff
    theorem genuinely does not hold here.  Same for `Float32`. -/
instance : NumOps Float where
  zero  := Nat.toFloat 0
  one   := 1.0
  add   := Float.add
  mul   := Float.mul
  neg   := Float.neg
  inv   := fun x => 1.0 / x
  exp   := Float.exp
  ex2   := fun x => Float.pow 2.0 x
  rsqrt := fun x => 1.0 / Float.sqrt x
  ofNat := Nat.toFloat
  le    := fun a b => a ≤ b

end AlgorithmLib.ML
