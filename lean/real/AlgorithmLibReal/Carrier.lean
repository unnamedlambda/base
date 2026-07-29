import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import AlgorithmLib.ML.Grad

/-!
  # ℝ as a lawful carrier — closing assumption A1

  Everything in this package is proof-only: nothing here is executed, and
  nothing here can reach an emitted `Artifact`.  It lives in a separate Lake
  package so `AlgorithmLib` and all ~45 algorithm builds stay Mathlib-free.

  Providing `NumLaws ℝ` is what makes `grad_correct` bite.  Until now it was a
  theorem about every commutative ring with no ring instantiated — true, but
  with nothing to apply it to.  One instance changes that, and `grad_correct`
  transfers with no restatement because it was stated carrier-generically from
  the start.

  `Float` still has no `NumLaws` instance, and still cannot have one: its
  addition does not associate.  That asymmetry is assumption A2, and it stays.
-/

namespace AlgorithmLib.ML

open NumOps

noncomputable instance instNumOpsReal : NumOps ℝ where
  zero  := 0
  one   := 1
  add   := (· + ·)
  mul   := (· * ·)
  neg   := (-·)
  inv   := (·⁻¹)
  exp   := Real.exp
  ex2   := fun x => Real.exp (x * Real.log 2)
  rsqrt := fun x => (Real.sqrt x)⁻¹
  ofNat := fun n => (n : ℝ)
  le    := fun a b => decide (a ≤ b)

@[simp] theorem real_add (a b : ℝ) : add a b = a + b := rfl
@[simp] theorem real_mul (a b : ℝ) : mul a b = a * b := rfl
@[simp] theorem real_neg (a : ℝ) : neg a = -a := rfl
@[simp] theorem real_inv (a : ℝ) : inv a = a⁻¹ := rfl
@[simp] theorem real_exp (a : ℝ) : NumOps.exp a = Real.exp a := rfl
@[simp] theorem real_rsqrt (a : ℝ) : rsqrt a = (Real.sqrt a)⁻¹ := rfl
@[simp] theorem real_zero : (zero : ℝ) = 0 := rfl
@[simp] theorem real_one : (one : ℝ) = 1 := rfl
@[simp] theorem real_ofNat (n : Nat) : (ofNat n : ℝ) = (n : ℝ) := rfl

/-- **ℝ is a lawful carrier.**  Each obligation is discharged by the
    corresponding fact about the real field — no new mathematics, which is the
    point: `NumLaws` was designed to be exactly a commutative ring. -/
noncomputable instance instNumLawsReal : NumLaws ℝ where
  toNumOps     := instNumOpsReal
  add_assoc    := by intro a b c; simp only [real_add]; ring
  add_comm     := by intro a b;   simp only [real_add]; ring
  add_zero     := by intro a;     simp only [real_add, real_zero]; ring
  add_neg      := by intro a;     simp only [real_add, real_neg, real_zero]; ring
  mul_assoc    := by intro a b c; simp only [real_mul]; ring
  mul_comm     := by intro a b;   simp only [real_mul]; ring
  mul_one      := by intro a;     simp only [real_mul, real_one]; ring
  mul_zero     := by intro a;     simp only [real_mul, real_zero]; ring
  neg_mul      := by intro a b;   simp only [real_mul, real_neg]; ring
  ofNat_zero   := by simp only [real_ofNat, real_zero, Nat.cast_zero]
  ofNat_one    := by simp only [real_ofNat, real_one, Nat.cast_one]
  left_distrib := by intro a b c; simp only [real_mul, real_add]; ring

/-- **A1 closed.**  Reverse-mode autodiff agrees with symbolic differentiation
    over the real numbers.  This is `grad_correct` instantiated — the theorem
    itself needed no change. -/
theorem grad_correct_real {Γ : Nat} (env : Fin Γ → ℝ) (e : Expr Γ) (k : Fin Γ) :
    denote env (grad e k) = denote env (sderiv e k) :=
  grad_correct env e k

end AlgorithmLib.ML
