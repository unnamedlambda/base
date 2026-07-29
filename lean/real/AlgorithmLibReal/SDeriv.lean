import AlgorithmLibReal.Carrier
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Algebra.BigOperators.Fin

/-!
  # The rule table is correct — closing the gap `grad_correct` left open

  `grad_correct` proves reverse mode agrees with `sderiv`.  But `sderiv` is a
  *hand-written table*: get the rsqrt rule wrong and both differentiators are
  wrong together, consistently.  On its own `grad_correct` therefore buys
  consistency, not correctness — it catches the hard bug (adjoint accumulation,
  reassociation, shared-vector updates) and misses the easy one.

  This file removes the table from the trusted base by proving it against
  Mathlib's `HasDerivAt`.  Composed:

      sderiv_hasDerivAt :  denote env (sderiv e k)  IS  ∂e/∂xₖ
      grad_correct      :  denote env (grad e k) = denote env (sderiv e k)
      ─────────────────────────────────────────────────────────────────────
      grad_hasDerivAt   :  denote env (grad e k)  IS  ∂e/∂xₖ

  Side conditions are real and carried explicitly by `Regular`: `inv` needs a
  nonzero argument, `rsqrt` a positive one.  They are not assumed away.
-/

namespace AlgorithmLib.ML

open NumOps

variable {Γ : Nat}

/-- Vary coordinate `k`, hold the rest fixed. -/
noncomputable def coord (env : Fin Γ → ℝ) (k : Fin Γ) (t : ℝ) : Fin Γ → ℝ :=
  Function.update env k t

@[simp] theorem coord_self (env : Fin Γ → ℝ) (k : Fin Γ) :
    coord env k (env k) = env := Function.update_eq_self k env

/-- Where the expression is differentiable.  `1/x` and `x^(-1/2)` are not
    differentiable everywhere, so these conditions appear in the statement. -/
def Regular : {Γ : Nat} → (Fin Γ → ℝ) → Expr Γ → Prop
  | _, _, .var _    => True
  | _, _, .lit _    => True
  | _, env, .add a b  => Regular env a ∧ Regular env b
  | _, env, .mul a b  => Regular env a ∧ Regular env b
  | _, env, .neg a    => Regular env a
  | _, env, .inv a    => denote env a ≠ 0 ∧ Regular env a
  | _, env, .exp a    => Regular env a
  | _, env, .rsqrt a  => 0 < denote env a ∧ Regular env a
  | _, env, .sum _ f  => ∀ j, Regular env (f j)
  | _, _, .letE _ _ => False

-- ---------------------------------------------------------------------------
-- The committed fold is a Finset sum
-- ---------------------------------------------------------------------------

theorem foldl_add_eq_sum {α : Type} (g : α → ℝ) :
    ∀ (L : List α) (s : ℝ), L.foldl (fun acc j => acc + g j) s = s + (L.map g).sum := by
  intro L
  induction L with
  | nil => intro s; simp
  | cons a L ih => intro s; rw [List.foldl_cons, ih (s + g a)]; simp [add_assoc]

/-- `denote`'s committed left fold agrees with `∑`, over ℝ. -/
theorem denote_sum_eq (env : Fin Γ → ℝ) (n : Nat) (f : Fin n → Expr Γ) :
    denote env (.sum n f) = ∑ j, denote env (f j) := by
  show (List.finRange n).foldl (fun acc j => add acc (denote env (f j))) zero = _
  simp only [real_add, real_zero]
  rw [foldl_add_eq_sum (fun j => denote env (f j)) (List.finRange n) 0, zero_add,
      ← Fin.sum_univ_def]

-- ---------------------------------------------------------------------------
-- The table is the derivative
-- ---------------------------------------------------------------------------

/-- **The rule table is correct.**  Every case is discharged by the matching
    Mathlib rule — no new mathematics, which is the point: what is being
    checked is that *this table* names *those rules* correctly. -/
theorem sderiv_hasDerivAt :
    ∀ {Γ : Nat} (e : Expr Γ) (env : Fin Γ → ℝ) (k : Fin Γ), Regular env e →
      HasDerivAt (fun t => denote (coord env k t) e) (denote env (sderiv e k)) (env k) := by
  intro Γ e
  induction e with
  | letE a b _ _ => intro _ _ hr; exact absurd hr (by simp [Regular])
  | var i =>
      intro env k _
      by_cases h : k = i
      · subst h
        have : (fun t => denote (coord env k t) (Expr.var k)) = fun t => t := by
          funext t; simp [denote, coord, Function.update_self]
        rw [this]
        simpa [sderiv] using hasDerivAt_id (env k)
      · have hne : i ≠ k := fun hc => h hc.symm
        have : (fun t => denote (coord env k t) (Expr.var i)) = fun _ => env i := by
          funext t; simp [denote, coord, Function.update_of_ne hne]
        rw [this]
        simpa [sderiv, if_neg h] using hasDerivAt_const (env k) (env i)
  | lit n =>
      intro env k _
      simpa [denote, sderiv] using hasDerivAt_const (env k) (ofNat n : ℝ)
  | add a b iha ihb =>
      intro env k hr
      simpa [denote, sderiv] using (iha env k hr.1).add (ihb env k hr.2)
  | mul a b iha ihb =>
      intro env k hr
      have h := (iha env k hr.1).mul (ihb env k hr.2)
      simp only [coord_self] at h
      simpa [denote, sderiv] using h
  | neg a iha =>
      intro env k hr
      simpa [denote, sderiv] using (iha env k hr).neg
  | inv a iha =>
      intro env k hr
      have ha := iha env k hr.2
      have hnz : (fun t => denote (coord env k t) a) (env k) ≠ 0 := by
        simpa using hr.1
      have h := ha.inv hnz
      simp only [coord_self] at h
      have : -denote env (sderiv a k) / denote env a ^ 2
           = denote env (sderiv (.inv a) k) := by
        simp only [sderiv, denote, real_neg, real_mul, real_inv]
        field_simp
      rw [← this]
      simpa [denote] using h
  | exp a iha =>
      intro env k hr
      have h := (iha env k hr).exp
      simp only [coord_self] at h
      have : Real.exp (denote env a) * denote env (sderiv a k)
           = denote env (sderiv (.exp a) k) := by
        simp only [sderiv, denote, real_mul, real_exp]; ring
      rw [← this]
      simpa [denote] using h
  | rsqrt a iha =>
      intro env k hr
      have hpos : 0 < denote env a := hr.1
      have ha := iha env k hr.2
      -- sqrt ∘ a
      have hsq : HasDerivAt (fun t => Real.sqrt (denote (coord env k t) a))
          (denote env (sderiv a k) / (2 * Real.sqrt (denote env a))) (env k) := by
        have hsqrt : HasDerivAt Real.sqrt (1 / (2 * Real.sqrt (denote env a)))
            ((fun t => denote (coord env k t) a) (env k)) := by
          simpa [coord_self] using Real.hasDerivAt_sqrt (ne_of_gt hpos)
        have hc := hsqrt.comp (env k) ha
        simp only [coord_self, Function.comp] at hc
        convert hc using 1
        field_simp
      have hsnz : Real.sqrt (denote env a) ≠ 0 :=
        ne_of_gt (Real.sqrt_pos.mpr hpos)
      have h := hsq.inv (by simpa [coord_self] using hsnz)
      simp only [coord_self] at h
      have hrw : -(denote env (sderiv a k) / (2 * Real.sqrt (denote env a)))
                   / Real.sqrt (denote env a) ^ 2
           = denote env (sderiv (.rsqrt a) k) := by
        simp only [sderiv, denote, real_neg, real_mul, real_inv, real_rsqrt, real_ofNat]
        rw [Real.sq_sqrt (le_of_lt hpos)]
        field_simp
        ring
      rw [← hrw]
      simpa [denote] using h
  | sum n f ih =>
      intro env k hr
      have hfun : (fun t => denote (coord env k t) (Expr.sum n f))
          = fun t => ∑ j, denote (coord env k t) (f j) := by
        funext t; exact denote_sum_eq _ n f
      rw [hfun, show denote env (sderiv (.sum n f) k)
            = ∑ j, denote env (sderiv (f j) k) from denote_sum_eq _ n _,
          show (fun t => ∑ j, denote (coord env k t) (f j))
            = ∑ j : Fin n, (fun t => denote (coord env k t) (f j)) from by
              funext t; simp [Finset.sum_apply]]
      exact HasDerivAt.sum (fun j _ => ih j env k (hr j))

/-- **Reverse-mode autodiff computes the analytic derivative.**

    `grad` is what gets compiled; `HasDerivAt` is Mathlib's derivative.  The
    hand-written rule table no longer sits between them as a trusted step. -/
theorem grad_hasDerivAt {Γ : Nat} (env : Fin Γ → ℝ) (e : Expr Γ)
    (k : Fin Γ) (hr : Regular env e) :
    HasDerivAt (fun t => denote (coord env k t) e) (denote env (grad e k)) (env k) := by
  rw [grad_correct_real env e k]
  exact sderiv_hasDerivAt e env k hr

end AlgorithmLib.ML
