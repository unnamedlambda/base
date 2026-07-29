import AlgorithmLib.ML.Expr

/-!
  # Automatic differentiation, and its correctness

  Two differentiators live here:

  * `sderiv` — textbook symbolic partial differentiation, defined structurally.
    It is obviously correct by inspection: each case *is* the calculus rule,
    written as data.  It is also hopeless to run (it duplicates subterms
    exponentially), so it is never compiled — it exists only as the spec.
  * `bp` / `grad` — reverse mode.  One backward sweep computes every partial
    at once, threading an adjoint accumulator.  This is what gets compiled.

  `grad_correct` says the fast one agrees with the obvious one.  That is a real
  theorem: reverse mode is a nontrivial rearrangement — it reassociates sums,
  commutes products, and accumulates into a shared vector.

  **Where this holds.** The proof is stated for any `NumLaws` carrier, i.e. any
  commutative ring.  ℝ is one.  `Float` is *not* — its addition does not
  associate — so this theorem provably does not transfer to the float program.
  That is the honest statement of the gap, and it is why `sderiv`/`grad` are
  justified in ℝ while `Computes` is stated in `Float`.
-/

namespace AlgorithmLib.ML

open NumOps NumLaws

variable {Γ : Nat}

-- ---------------------------------------------------------------------------
-- The specification: symbolic differentiation
-- ---------------------------------------------------------------------------

/-- `sderiv e k` is ∂e/∂xₖ, one constructor at a time.  Read it as the rule
    table: this is the thing that has to be right, and it is short enough to
    check by eye against a calculus text. -/
def sderiv : {Γ : Nat} → Expr Γ → Fin Γ → Expr Γ
  | _, .var i,   k => if k = i then .lit 1 else .lit 0
  | _, .lit _,   _ => .lit 0
  | _, .add a b, k => .add (sderiv a k) (sderiv b k)
  | _, .mul a b, k => .add (.mul (sderiv a k) b) (.mul a (sderiv b k))
  | _, .neg a,   k => .neg (sderiv a k)
  -- d(1/a) = -a' / a²
  | _, .inv a,   k => .neg (.mul (sderiv a k) (.mul (.inv a) (.inv a)))
  | _, .exp a,   k => .mul (sderiv a k) (.exp a)
  -- d(a^(-1/2)) = -½ · a' · a^(-1/2) · a⁻¹
  | _, .rsqrt a, k =>
      .neg (.mul (sderiv a k) (.mul (.inv (.lit 2)) (.mul (.rsqrt a) (.inv a))))
  | _, .sum n f, k => .sum n (fun j => sderiv (f j) k)
  -- Chain rule through a binding:
  --   d/dxₖ (let v = a in b) = ∂b/∂xₖ + (∂b/∂v)·(da/dxₖ)
  -- evaluated under the same binding, so the shared subterm `a` is written once.
  -- Needs only weakening (`wk`), not substitution — which is why the binder
  -- sits at the end of the context.
  | Γ, .letE a b, k =>
      .letE a (.add (sderiv b ⟨k.val, Nat.lt_succ_of_lt k.isLt⟩)
                    (.mul (sderiv b ⟨Γ, Nat.lt_succ_self Γ⟩) (wk (sderiv a k))))

-- ---------------------------------------------------------------------------
-- The implementation: reverse mode
-- ---------------------------------------------------------------------------

/-- Accumulate adjoint `w` into slot `i`. -/
def upd (acc : Fin Γ → Expr Γ) (i : Fin Γ) (w : Expr Γ) : Fin Γ → Expr Γ :=
  fun k => if k = i then .add (acc k) w else acc k

/-- Backpropagate adjoint `w` through `e`, accumulating into `acc`.

    Each case pushes the local derivative *into the weight* and recurses, which
    is exactly what makes reverse mode cheap — and exactly what makes agreement
    with `sderiv` a theorem rather than a definition. -/
def bp : {Γ : Nat} → Expr Γ → Expr Γ → (Fin Γ → Expr Γ) → (Fin Γ → Expr Γ)
  | _, .var i,   w, acc => upd acc i w
  | _, .lit _,   _, acc => acc
  | _, .add a b, w, acc => bp b w (bp a w acc)
  | _, .mul a b, w, acc => bp b (.mul w a) (bp a (.mul w b) acc)
  | _, .neg a,   w, acc => bp a (.neg w) acc
  | _, .inv a,   w, acc => bp a (.neg (.mul w (.mul (.inv a) (.inv a)))) acc
  | _, .exp a,   w, acc => bp a (.mul w (.exp a)) acc
  | _, .rsqrt a, w, acc =>
      bp a (.neg (.mul w (.mul (.inv (.lit 2)) (.mul (.rsqrt a) (.inv a))))) acc
  | _, .sum n f, w, acc => (List.finRange n).foldl (fun a j => bp (f j) w a) acc
  -- Reverse mode does not yet propagate *through* a binding while keeping the
  -- sharing: the bound variable's adjoint lives in the extended context and
  -- cannot be handed to `bp a` without escaping it.  So `letE` falls back to
  -- the symbolic rule — correct, total, and covered by `grad_correct`, but it
  -- re-expands the shared subterm.  Efficient AD over shared terms needs a
  -- tape; that is the remaining gap.
  | _, e@(.letE _ _), w, acc => fun k => .add (acc k) (.mul w (sderiv e k))

/-- The gradient of `e`: seed the adjoint at 1, start from a zero vector.
    This is what the user calls, and what gets compiled. -/
def grad (e : Expr Γ) : Fin Γ → Expr Γ := bp e (.lit 1) (fun _ => .lit 0)

-- ---------------------------------------------------------------------------
-- Correctness
-- ---------------------------------------------------------------------------

section Proof

variable {R : Type} [NumLaws R]

@[simp] theorem denote_lit0 {Γ : Nat} (env : Fin Γ → R) :
    denote env (.lit 0 : Expr Γ) = zero := NumLaws.ofNat_zero

@[simp] theorem denote_lit1 {Γ : Nat} (env : Fin Γ → R) :
    denote env (.lit 1 : Expr Γ) = one := NumLaws.ofNat_one

/-- Pulling a fold's starting value out front.  Needed because `bp`'s `sum`
    case threads a live accumulator while `sderiv`'s starts from zero. -/
theorem fold_add_start (g : Fin n → R) :
    ∀ (L : List (Fin n)) (s0 : R),
      L.foldl (fun s j => add s (g j)) s0
        = add s0 (L.foldl (fun s j => add s (g j)) zero) := by
  intro L
  induction L with
  | nil => intro s0; simp [List.foldl]; exact (add_zero s0).symm
  | cons j L ih =>
      intro s0
      show L.foldl _ (add s0 (g j)) = add s0 (L.foldl _ (add zero (g j)))
      rw [ih (add s0 (g j)), ih (add zero (g j)), zero_add, add_assoc]

/-- The main lemma.  A backward sweep leaves each slot holding *what was
    already there* plus *the incoming adjoint times the true partial*.

    Stated over an arbitrary accumulator so the induction goes through — the
    `add`/`mul` cases run `bp` twice, each starting from the other's output. -/
theorem bp_denote :
    ∀ {Γ : Nat} (e : Expr Γ), ∀ (env : Fin Γ → R) (w : Expr Γ)
      (acc : Fin Γ → Expr Γ) (k : Fin Γ),
      denote env (bp e w acc k)
        = add (denote env (acc k)) (mul (denote env w) (denote env (sderiv e k))) := by
  intro Γ e
  induction e with
  | var i =>
      intro env w acc k
      by_cases h : k = i
      · simp only [bp, upd, sderiv, if_pos h, denote_add, denote_lit1, mul_one]
      · simp only [bp, upd, sderiv, if_neg h, denote_lit0, mul_zero, add_zero]
  | lit n =>
      intro env w acc k
      simp only [bp, sderiv, denote_lit0, mul_zero, add_zero]
  | add a b iha ihb =>
      intro env w acc k
      rw [show bp (.add a b) w acc = bp b w (bp a w acc) from rfl,
          ihb env w (bp a w acc) k, iha env w acc k]
      show add (add _ _) _ = add _ (mul _ (add _ _))
      rw [left_distrib, add_assoc]
  | mul a b iha ihb =>
      intro env w acc k
      rw [show bp (.mul a b) w acc = bp b (.mul w a) (bp a (.mul w b) acc) from rfl,
          ihb env (.mul w a) (bp a (.mul w b) acc) k, iha env (.mul w b) acc k]
      show add (add _ (mul (mul _ _) _)) (mul (mul _ _) _)
            = add _ (mul _ (add (mul _ _) (mul _ _)))
      rw [left_distrib, add_assoc,
          mul_swap_right (denote env w) (denote env (sderiv a k)) (denote env b),
          mul_swap_right (denote env w) (denote env (sderiv b k)) (denote env a),
          mul_comm (denote env (sderiv b k)) (denote env a)]
  | neg a iha =>
      intro env w acc k
      rw [show bp (.neg a) w acc = bp a (.neg w) acc from rfl, iha env (.neg w) acc k]
      show add _ (mul (neg _) _) = add _ (mul _ (neg _))
      rw [neg_mul, mul_neg]
  | inv a iha =>
      intro env w acc k
      rw [show bp (.inv a) w acc
            = bp a (.neg (.mul w (.mul (.inv a) (.inv a)))) acc from rfl,
          iha env (.neg (.mul w (.mul (.inv a) (.inv a)))) acc k]
      simp only [sderiv, denote_neg, denote_mul, denote_inv]
      rw [neg_mul, mul_neg, mul_swap_right]
  | exp a iha =>
      intro env w acc k
      rw [show bp (.exp a) w acc = bp a (.mul w (.exp a)) acc from rfl,
          iha env (.mul w (.exp a)) acc k]
      simp only [sderiv, denote_mul, denote_exp]
      rw [mul_swap_right]
  | rsqrt a iha =>
      intro env w acc k
      rw [show bp (.rsqrt a) w acc
            = bp a (.neg (.mul w (.mul (.inv (.lit 2)) (.mul (.rsqrt a) (.inv a))))) acc
            from rfl,
          iha env (.neg (.mul w (.mul (.inv (.lit 2)) (.mul (.rsqrt a) (.inv a))))) acc k]
      simp only [sderiv, denote_neg, denote_mul, denote_inv, denote_rsqrt, denote_lit]
      rw [neg_mul, mul_neg, mul_swap_right]
  | letE a b _ _ =>
      -- the fallback rule is definitionally what the statement asserts
      intro env w acc k; rfl
  | sum n f ih =>
      intro env w acc k
      show denote env ((List.finRange n).foldl (fun a j => bp (f j) w a) acc k)
            = add (denote env (acc k))
                (mul (denote env w)
                  ((List.finRange n).foldl
                    (fun s j => add s (denote env (sderiv (f j) k))) zero))
      -- generalise over the index list, then induct on it
      suffices h : ∀ (L : List (Fin n)) (acc' : Fin _ → Expr _),
          denote env (L.foldl (fun a j => bp (f j) w a) acc' k)
            = add (denote env (acc' k))
                (mul (denote env w)
                  (L.foldl (fun s j => add s (denote env (sderiv (f j) k))) zero)) from
        h (List.finRange n) acc
      intro L
      induction L with
      | nil =>
          intro acc'
          show denote env (acc' k) = add (denote env (acc' k)) (mul (denote env w) zero)
          rw [mul_zero, add_zero]
      | cons j L ihL =>
          intro acc'
          show denote env (L.foldl (fun a j => bp (f j) w a) (bp (f j) w acc') k)
                = add (denote env (acc' k))
                    (mul (denote env w)
                      (L.foldl (fun s j => add s (denote env (sderiv (f j) k)))
                        (add zero (denote env (sderiv (f j) k)))))
          rw [ihL (bp (f j) w acc'), ih j env w acc' k,
              fold_add_start (fun j => denote env (sderiv (f j) k)) L
                (add zero (denote env (sderiv (f j) k))),
              zero_add, left_distrib, add_assoc]

/-- **Reverse-mode autodiff is differentiation.**

    Holds in every commutative ring, hence in ℝ.  Does not hold in `Float`:
    there is no `NumLaws Float` instance, and that is not an oversight. -/
theorem grad_correct {Γ : Nat} (env : Fin Γ → R) (e : Expr Γ)
    (k : Fin Γ) : denote env (grad e k) = denote env (sderiv e k) := by
  rw [show grad e k = bp e (.lit 1) (fun _ => .lit 0) k from rfl,
      bp_denote e env (.lit 1) (fun _ => .lit 0) k,
      denote_lit0 env, denote_lit1 env, one_mul, zero_add]

end Proof

end AlgorithmLib.ML
