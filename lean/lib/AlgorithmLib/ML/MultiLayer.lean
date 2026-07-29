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

theorem denote_wkBy {Γ : Nat} (d : Nat) (e : Expr Γ) (env : Fin (Γ + d) → R) :
    denote env (wkBy d e) = denote (fun i => env ⟨i.val, by omega⟩) e := by
  rw [wkBy, denote_rename]

@[simp] theorem extendVec_lt {Γ d : Nat} (env : Fin Γ → R) (vs : Fin d → R)
    (k : Fin (Γ + d)) (h : k.val < Γ) : extendVec env vs k = env ⟨k.val, h⟩ := by
  simp [extendVec, h]

@[simp] theorem extendVec_ge {Γ d : Nat} (env : Fin Γ → R) (vs : Fin d → R)
    (k : Fin (Γ + d)) (h : ¬ k.val < Γ) :
    extendVec env vs k = vs ⟨k.val - Γ, by omega⟩ := by
  simp [extendVec, h]

theorem extend_ge {Γ : Nat} (env : Fin Γ → R) (v : R) (k : Fin (Γ + 1))
    (h : ¬ k.val < Γ) : extend env v k = v := by simp [extend, h]

/-- **Binding a vector is meaning-preserving.**

    The body sees exactly the layer's outputs in slots `Γ … Γ+d-1`, and each is
    evaluated once — which is the whole point.  Exact in every carrier, so it is
    a `Rewrite`-tier fact, not a `Float`-only one. -/
theorem denote_bindVec : ∀ (d : Nat) {Γ : Nat} (vs : Fin d → Expr Γ)
    (b : Expr (Γ + d)) (env : Fin Γ → R),
    denote env (bindVec d vs b)
      = denote (extendVec env (fun i => denote env (vs i))) b := by
  intro d
  induction d with
  | zero =>
      intro Γ vs b env
      show denote env b = _
      congr 1
      funext k
      rw [extendVec_lt (d := 0) env (fun i => denote env (vs i)) k k.isLt]
  | succ d ih =>
      intro Γ vs b env
      show denote env (bindVec d (fun i => vs ⟨i.val, by omega⟩)
              (.letE (wkBy d (vs ⟨d, by omega⟩)) b)) = _
      rw [ih (fun i => vs ⟨i.val, by omega⟩) _ env]
      show denote (extend (extendVec env (fun i => denote env (vs ⟨i.val, by omega⟩)))
              (denote (extendVec env (fun i => denote env (vs ⟨i.val, by omega⟩)))
                (wkBy d (vs ⟨d, by omega⟩)))) b = _
      rw [denote_wkBy]
      have hV : (fun (i : Fin Γ) =>
            extendVec env (fun i => denote env (vs ⟨i.val, by omega⟩))
              (⟨i.val, by omega⟩ : Fin (Γ + d))) = env := by
        funext i
        exact extendVec_lt env _ ⟨i.val, by omega⟩ i.isLt
      rw [hV]
      congr 1
      funext k
      by_cases hk : k.val < Γ + d
      · rw [extend_lt _ _ k hk]
        by_cases hk2 : k.val < Γ
        · rw [extendVec_lt (Γ := Γ) (d := d) _ _ ⟨k.val, hk⟩ hk2,
              extendVec_lt (Γ := Γ) (d := d + 1) _ _ k hk2]
        · rw [extendVec_ge (Γ := Γ) (d := d) _ _ ⟨k.val, hk⟩ hk2,
              extendVec_ge (Γ := Γ) (d := d + 1) _ _ k hk2]
      · have hklt : k.val < Γ + (d + 1) := k.isLt
        rw [extend_ge _ _ k hk, extendVec_ge (Γ := Γ) (d := d + 1) _ _ k (by omega)]
        have : (⟨d, by omega⟩ : Fin (d + 1)) = ⟨k.val - Γ, by omega⟩ :=
          Fin.ext (show d = k.val - Γ by omega)
        rw [this]

/-- **Bound slots read back what was bound.**  The identity a layer stack needs
    to hand its outputs to the next layer. -/
@[simp] theorem denote_boundVar {Γ d : Nat} (env : Fin Γ → R) (vs : Fin d → R)
    (i : Fin d) : denote (extendVec env vs) (boundVar Γ i) = vs i := by
  show extendVec env vs ⟨Γ + i.val, by omega⟩ = vs i
  rw [extendVec_ge (Γ := Γ) (d := d) env vs ⟨Γ + i.val, by omega⟩
    (show ¬ (Γ + i.val < Γ) by omega)]
  congr 1
  exact Fin.ext (show Γ + i.val - Γ = i.val by omega)

end AlgorithmLib.ML
