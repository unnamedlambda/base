import AlgorithmLib.ML.Grad
import AlgorithmLib.ML.TapeGrad

namespace AlgorithmLib.ML

open NumOps

-- ---------------------------------------------------------------------------
-- Integers are a lawful carrier
-- ---------------------------------------------------------------------------

/-- `Int` computes.  The transcendental slots are junk: they are never
    constrained by `NumLaws` and never used by a quantized kernel, which is
    exactly why integer arithmetic can be a *lawful* carrier while `Float`
    cannot. -/
instance : NumOps Int where
  zero  := 0
  one   := 1
  add   := (· + ·)
  mul   := (· * ·)
  neg   := (-·)
  inv   := fun _ => 0
  exp   := fun _ => 0
  ex2   := fun _ => 0
  rsqrt := fun _ => 0
  ofNat := fun n => (n : Int)
  le    := fun a b => decide (a ≤ b)

@[simp] theorem int_add (a b : Int) : add a b = a + b := rfl
@[simp] theorem int_mul (a b : Int) : mul a b = a * b := rfl
@[simp] theorem int_neg (a : Int) : neg a = -a := rfl
@[simp] theorem int_zero : (zero : Int) = 0 := rfl
@[simp] theorem int_one : (one : Int) = 1 := rfl
@[simp] theorem int_ofNat (n : Nat) : (ofNat n : Int) = (n : Int) := rfl

/-- **`Int` is a lawful carrier.**  Unlike `Float`, every ring obligation
    actually holds — so `grad_correct` is not vacuous over the quantized
    pipeline. -/
instance : NumLaws Int where
  add_assoc    := by intro a b c; simp only [int_add]; omega
  add_comm     := by intro a b;   simp only [int_add]; omega
  add_zero     := by intro a;     simp only [int_add, int_zero]; omega
  add_neg      := by intro a;     simp only [int_add, int_neg, int_zero]; omega
  mul_assoc    := by intro a b c; simp only [int_mul]; exact Int.mul_assoc a b c
  mul_comm     := by intro a b;   simp only [int_mul]; exact Int.mul_comm a b
  mul_one      := by intro a;     simp only [int_mul, int_one]; exact Int.mul_one a
  mul_zero     := by intro a;     simp only [int_mul, int_zero]; exact Int.mul_zero a
  neg_mul      := by intro a b;   simp only [int_mul, int_neg]; exact Int.neg_mul a b
  ofNat_zero   := by simp only [int_ofNat, int_zero]; rfl
  ofNat_one    := by simp only [int_ofNat, int_one]; rfl
  left_distrib := by intro a b c; simp only [int_mul, int_add]; exact Int.mul_add a b c

/-- **Autodiff is exactly correct over the quantized carrier.**

    Not "approximately" — `Int` satisfies the laws, so this is the same theorem
    `Float` cannot have. -/
theorem grad_correct_int {Γ : Nat} (env : Fin Γ → Int) (e : Expr Γ) (k : Fin Γ) :
    denote env (grad e k) = denote env (sderiv e k) :=
  grad_correct env e k

-- ---------------------------------------------------------------------------
-- The quantize / dequantize boundary
-- ---------------------------------------------------------------------------

/-- Symmetric int8 range. -/
def qMin : Int := -127
def qMax : Int := 127

def clampQ (v : Int) : Int := if v < qMin then qMin else if qMax < v then qMax else v

theorem clampQ_lower (v : Int) : qMin ≤ clampQ v := by
  unfold clampQ qMin qMax; split
  · omega
  · split <;> omega

theorem clampQ_upper (v : Int) : clampQ v ≤ qMax := by
  unfold clampQ qMin qMax; split
  · omega
  · split <;> omega

/-- A clamped value has magnitude at most 127 — the form the overflow bound
    actually needs, since `natAbs` turns products into `Nat` multiplication. -/
theorem clampQ_natAbs (v : Int) : (clampQ v).natAbs ≤ 127 := by
  have h1 : (-127 : Int) ≤ clampQ v := clampQ_lower v
  have h2 : clampQ v ≤ (127 : Int) := clampQ_upper v
  omega

-- ---------------------------------------------------------------------------
-- Accumulator overflow
-- ---------------------------------------------------------------------------

/-- The safe reduction length for `int32` accumulation of int8 products. -/
def maxSafeN : Nat := 133141

/-- A dot product of `n` clamped int8 values, accumulated left-to-right. -/
def qdot (n : Nat) (a b : Nat → Int) : Int :=
  (List.range n).foldl (fun acc k => acc + clampQ (a k) * clampQ (b k)) 0

/-- Each summand has magnitude at most `127² = 16129`. -/
theorem qterm_natAbs (x y : Int) : (clampQ x * clampQ y).natAbs ≤ 16129 := by
  rw [Int.natAbs_mul]
  exact Nat.le_trans (Nat.mul_le_mul (clampQ_natAbs x) (clampQ_natAbs y)) (by decide)

/-- **The accumulator cannot overflow.**

    After `n` terms the partial sum has magnitude at most `n·16129`, so any
    reduction shorter than 133,141 terms is safe in `int32`.  Silent accumulator
    overflow is a real production bug class in quantized kernels; here it is a
    discharged side condition. -/
theorem qdot_natAbs (a b : Nat → Int) :
    ∀ (L : List Nat) (acc : Int),
      (L.foldl (fun s k => s + clampQ (a k) * clampQ (b k)) acc).natAbs
        ≤ acc.natAbs + L.length * 16129 := by
  intro L
  induction L with
  | nil => intro acc; simp
  | cons k L ih =>
      intro acc
      rw [List.foldl_cons]
      refine Nat.le_trans (ih (acc + clampQ (a k) * clampQ (b k))) ?_
      have h1 : (acc + clampQ (a k) * clampQ (b k)).natAbs
          ≤ acc.natAbs + (clampQ (a k) * clampQ (b k)).natAbs := Int.natAbs_add_le _ _
      have h2 := qterm_natAbs (a k) (b k)
      simp only [List.length_cons]
      omega

/-- The full dot product over `[0, n)` stays inside `int32` when `n` is safe. -/
theorem qdot_fits (n : Nat) (a b : Nat → Int) (hn : n ≤ maxSafeN) :
    (qdot n a b).natAbs ≤ 2147483647 := by
  have h := qdot_natAbs a b (List.range n) 0
  simp only [Int.natAbs_zero, List.length_range] at h
  have : n * 16129 ≤ maxSafeN * 16129 := Nat.mul_le_mul_right _ hn
  unfold qdot maxSafeN at *
  omega

theorem maxSafeN_fits : maxSafeN * 16129 ≤ 2147483647 := by
  unfold maxSafeN; omega

/-- **`Int` satisfies `ZeroLaws`.**

    So the narrowed gradient's second declared proposition is not merely
    plausible — it is discharged for the quantized carrier, where integer
    arithmetic is exact and there is no signed zero to worry about.  That is the
    same reason `Int` is a `NumLaws` carrier and `Float32` is not. -/
theorem zeroLaws_Int : ZeroLaws Int := by
  refine ⟨rfl, ?_, ?_, rfl, ?_⟩
  · intro x; show (0 : Int) * x = 0; simp
  · intro x; show x * (0 : Int) = 0; simp
  · intro n
    show (List.finRange n).foldl (fun acc (_ : Fin n) => acc + (0 : Int)) 0 = 0
    have key : ∀ (L : List (Fin n)) (a : Int),
        L.foldl (fun acc (_ : Fin n) => acc + (0 : Int)) a = a := by
      intro L
      induction L with
      | nil => intro a; rfl
      | cons _ L ih => intro a; rw [List.foldl_cons, ih]; simp
    exact key _ 0

end AlgorithmLib.ML
