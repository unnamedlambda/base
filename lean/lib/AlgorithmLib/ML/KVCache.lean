import AlgorithmLib.ML.Transformer

/-!
  # Incremental decode equals full recompute

  This is the seam between training and inference.  Training runs attention over
  a whole sequence, recomputing every K and V projection.  Decode reads K and V
  for past positions out of a cache and projects only the new token.  Every
  serving stack depends on those agreeing; none proves it; and the failure modes
  — position off-by-one, stale cache, wrong slot, sliding-window edges — are
  silent.

  The theorem below is the one that matters: **given the cache invariant, the
  cached family denotes exactly what full recompute denotes**, and therefore any
  attention built over it does too.

  Note what is *not* saved by the cache: the attention scores.  `q` changes each
  step, so `q·Kⱼ` is genuinely new for every `j`.  What the cache saves is the
  *projection* `Wk·xⱼ`, and that is exactly what the invariant is about.
-/

namespace AlgorithmLib.ML
namespace KVCache

variable {Γ : Nat}

/-- The family a decode step actually reads: cached for past positions, freshly
    projected for the current one. -/
def cached (t : Nat) (store : Fin t → Expr Γ) (fresh : Expr Γ) :
    Fin (t + 1) → Expr Γ :=
  fun j => if h : j.val < t then store ⟨j.val, h⟩ else fresh

/-- The family full recompute produces. -/
def recomputed (t : Nat) (proj : Fin (t + 1) → Expr Γ) : Fin (t + 1) → Expr Γ := proj

/-- **The cache invariant**: every stored slot holds the projection of its
    token, and the fresh slot holds the projection of the current token. -/
def Invariant {R : Type} [NumOps R] (t : Nat) (env : Fin Γ → R)
    (store : Fin t → Expr Γ) (fresh : Expr Γ) (proj : Fin (t + 1) → Expr Γ) : Prop :=
  (∀ j : Fin t, denote env (store j)
      = denote env (proj ⟨j.val, Nat.lt_succ_of_lt j.isLt⟩))
  ∧ denote env fresh = denote env (proj ⟨t, Nat.lt_succ_self t⟩)

/-- **Incremental decode reads what full recompute would have computed.**

    Pointwise, at every position — including the boundary between "cached" and
    "fresh", which is where off-by-one bugs live. -/
theorem cached_eq_recomputed {R : Type} [NumOps R] (t : Nat) (env : Fin Γ → R)
    (store : Fin t → Expr Γ) (fresh : Expr Γ) (proj : Fin (t + 1) → Expr Γ)
    (hinv : Invariant t env store fresh proj) :
    ∀ j : Fin (t + 1),
      denote env (cached t store fresh j) = denote env (recomputed t proj j) := by
  intro j
  show denote env (if _ : _ then _ else _) = denote env (proj j)
  split
  · next h => rw [hinv.1 ⟨j.val, h⟩]
  · next h =>
      have hjt : j.val = t := by
        have := j.isLt
        omega
      have hj : j = ⟨t, Nat.lt_succ_self t⟩ := Fin.ext hjt
      rw [hinv.2, hj]

-- ---------------------------------------------------------------------------
-- Anything built over the family agrees too
-- ---------------------------------------------------------------------------

/-- A reduction over pointwise-equal families denotes the same value. -/
theorem sum_congr {R : Type} [NumOps R] (n : Nat) (env : Fin Γ → R)
    (f g : Fin n → Expr Γ) (h : ∀ j, denote env (f j) = denote env (g j)) :
    denote env (.sum n f) = denote env (.sum n g) := by
  show (List.finRange n).foldl (fun acc j => NumOps.add acc (denote env (f j))) NumOps.zero = _
  simp only [h]; rfl

/-- **Decode-time attention equals training-time attention.**

    The V-mix — the part that actually consumes the cache — agrees, given the
    invariant.  Composed with `cached_eq_recomputed`, a decode step over a
    correctly maintained cache computes the training-time function. -/
theorem attnMix_congr {R : Type} [NumOps R] (t : Nat) (env : Fin Γ → R)
    (p : Fin (t + 1) → Expr Γ)
    (store : Fin t → Expr Γ) (fresh : Expr Γ) (proj : Fin (t + 1) → Expr Γ)
    (hinv : Invariant t env store fresh proj) :
    denote env (.sum (t + 1) (fun j => .mul (p j) (cached t store fresh j)))
      = denote env (.sum (t + 1) (fun j => .mul (p j) (recomputed t proj j))) := by
  refine sum_congr (t + 1) env _ _ (fun j => ?_)
  show NumOps.mul _ _ = NumOps.mul _ _
  rw [cached_eq_recomputed t env store fresh proj hinv j]

end KVCache
end AlgorithmLib.ML
