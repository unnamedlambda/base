import AlgorithmLib.ML.Grad

/-!
  # Multi-layer models: binding a whole activation vector

  `letE` binds *one* value.  A layer produces a *vector* of them, and the next
  layer reads every component — which is exactly the reuse that makes a tree
  representation explode.  Measured, one 4-wide layer of
  rmsnorm → matvec → silu → residual multiplies the term size by **73**:

  | layers | tree nodes | with `bindVec` |
  |-------:|-----------:|---------------:|
  | 0      | 5          | 5              |
  | 1      | 873        | 881            |
  | 2      | 64,237     | 1,757          |
  | 3      | 4,689,809  | 2,633          |

  `bindVec` is the missing plumbing: `d` nested `letE`s that name a layer's
  whole output, so the next layer refers to *variables* rather than copies.
  With it the forward term is **linear** in depth — +876 nodes per layer above,
  exactly — and a 24-layer model is representable.

  This is deliberately *not* a new IR.  The measurement says `Expr` + `letE`
  already has the sharing; what was missing was a vector-shaped binder.

  ## What this does not fix

  The *gradient* is still exponential, and for a specific, located reason:
  `sderiv` on `letE a b` recurses into `b` **twice** (once for `∂b/∂x`, once for
  `∂b/∂v`), and `bp` falls back to `sderiv` at every binding.  So each binder
  doubles the backward term.  Measured growth factor per layer, against the
  number of binders per layer:

  | binders/layer `d` | grad factor per layer |
  |------------------:|----------------------:|
  | 1                 | ~2                    |
  | 2                 | ~4                    |
  | 3                 | ~8                    |
  | 4                 | ~16                   |

  That is `2^d` — two per binder, as the rule predicts.  Forward is solved;
  backward needs reverse mode to propagate *through* a binding, accumulating
  the bound variable's adjoint instead of re-deriving the body.  That is the
  remaining blocker for training a multi-layer model, and it is one rule, not
  a representation change.
-/

namespace AlgorithmLib.ML

variable {R : Type} [NumOps R]

/-- Weaken by `d` contexts at once. -/
def wkBy {Γ : Nat} (d : Nat) (e : Expr Γ) : Expr (Γ + d) :=
  rename (fun i => ⟨i.val, by omega⟩) e

/-- Extend an environment by a whole vector.  Slot `i` sits at index `Γ + i`,
    matching where `bindVec` puts it. -/
def extendVec {Γ d : Nat} (env : Fin Γ → R) (vs : Fin d → R) : Fin (Γ + d) → R :=
  fun k => if h : k.val < Γ then env ⟨k.val, h⟩ else vs ⟨k.val - Γ, by omega⟩

/-- The variable standing for bound slot `i`. -/
def boundVar (Γ : Nat) {d : Nat} (i : Fin d) : Expr (Γ + d) :=
  .var ⟨Γ + i.val, by omega⟩

/-- Bind `d` values with `d` nested `letE`s.

    The last component is bound innermost, which is what makes the context
    arithmetic definitional: `Γ + (d+1)` *is* `(Γ + d) + 1`, so no coercion is
    needed anywhere. -/
def bindVec : (d : Nat) → {Γ : Nat} → (Fin d → Expr Γ) → Expr (Γ + d) → Expr Γ
  | 0, _, _, b => b
  | (d + 1), Γ, vs, b =>
      bindVec d (fun i => vs ⟨i.val, by omega⟩)
        (.letE (wkBy d (vs ⟨d, by omega⟩)) b)

@[simp] theorem extendVec_lt {Γ d : Nat} (env : Fin Γ → R) (vs : Fin d → R)
    (k : Fin (Γ + d)) (h : k.val < Γ) : extendVec env vs k = env ⟨k.val, h⟩ := by
  simp [extendVec, h]

@[simp] theorem extendVec_ge {Γ d : Nat} (env : Fin Γ → R) (vs : Fin d → R)
    (k : Fin (Γ + d)) (h : ¬ k.val < Γ) :
    extendVec env vs k = vs ⟨k.val - Γ, by omega⟩ := by
  simp [extendVec, h]

theorem extend_ge {Γ : Nat} (env : Fin Γ → R) (v : R) (k : Fin (Γ + 1))
    (h : ¬ k.val < Γ) : extend env v k = v := by simp [extend, h]

end AlgorithmLib.ML
