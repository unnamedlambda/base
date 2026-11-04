module
public import AlgorithmLib.ML.Math.Expr
meta import AlgorithmLib.ML.Math.Expr
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # Transformer building blocks, as `Expr` combinators

  The nine constructors of `Expr` express a decoder layer's arithmetic with no
  new primitives:

  * **matvec / attention scores** — `sum` of `mul`
  * **softmax**                   — `exp`, `sum`, `inv`
  * **SiLU**                      — `x · (1 + e^{-x})⁻¹`
  * **RMSNorm**                   — `sum`, `rsqrt`
  * **residual**                  — `add`

  Every combinator here is a plain function returning an `Expr`, so a model built
  from them is one term.  It therefore inherits, with no new proof obligations:

  * `grad`             — the backward pass
  * `grad_hasDerivAt`  — that the backward pass is the analytic derivative
  * `compileW`         — a warp kernel
  * `compileWKernel_correct` — that the kernel computes the spec, exactly
  * `flatKernel_sound` — that the emitted PTX runs the kernel

  **Softmax subtracts a shift** (`softmaxAt`): the shifted form is the model,
  and `softmax`, the textbook form, is kept for reference.
-/

namespace AlgorithmLib.ML
namespace Transformer

variable {Γ : Nat}

-- ---------------------------------------------------------------------------
-- Linear algebra
-- ---------------------------------------------------------------------------

/-- Inner product of two expression vectors. -/
def dot (n : Nat) (a b : Fin n → Expr Γ) : Expr Γ :=
  .sum n (fun k => a k * b k)

/-- `y i = Σₖ W i k · x k`.  Rows of `W` are given as expression vectors, so
    this works equally for a weight matrix in memory or a fused subexpression. -/
def matvec (n : Nat) (W : Fin m → Fin n → Expr Γ) (x : Fin n → Expr Γ)
    (i : Fin m) : Expr Γ :=
  dot n (W i) x

/-- Elementwise `a + b`. -/
def vadd (v w : Fin n → Expr Γ) : Fin n → Expr Γ := fun i => v i + w i

/-- Elementwise `a · b` — the FFN's gate/up product. -/
def vmul (v w : Fin n → Expr Γ) : Fin n → Expr Γ := fun i => v i * w i

-- ---------------------------------------------------------------------------
-- Activations and normalisation
-- ---------------------------------------------------------------------------

/-- `σ(x) = (1 + e^{-x})⁻¹`. -/
def sigmoid (x : Expr Γ) : Expr Γ :=
  (1 + Expr.exp (-x))⁻¹

/-- `silu x = x · σ(x)`.  Smooth, so `grad` handles it with no side condition. -/
def silu (x : Expr Γ) : Expr Γ := x * sigmoid x

/-- **What is actually computed**: softmax about a supplied shift,
    `e^{zᵢ−s} / Σⱼ e^{zⱼ−s}`.

    This is the model, not a concession to the kernel.  Every serious softmax
    subtracts a shift before exponentiating — `e^{z}` overflows `Float32` at
    `z > 88` — and the reference transformer implementations are no exception, so the
    shifted form is what the published weights were trained and evaluated
    against.  Writing the unshifted form as *the* model and the shifted one as
    an approximation of it would have the relationship backwards.

    The shift is a parameter rather than `max z` because `Expr` has no `max`:
    adding one is a language extension needing an `sderiv` case, and `max` is
    not smooth.  Taking it as an input costs nothing — the kernel supplies the
    row max, and the correspondence theorem instantiates it there. -/
def softmaxAt (n : Nat) (shift : Expr Γ) (z : Fin n → Expr Γ) (i : Fin n) : Expr Γ :=
  Expr.exp (z i - shift) / .sum n (fun j => Expr.exp (z j - shift))

/-- The shifted form is `mul (exp (add zᵢ (neg s))) (inv (Σ …))` — exactly the
    shape the kernel's `smxVal` has.  Pinned so a change to `-`/`/` cannot move
    it silently. -/
example (n : Nat) (s : Expr Γ) (z : Fin n → Expr Γ) (i : Fin n) :
    softmaxAt n s z i
      = .mul (.exp (.add (z i) (.neg s)))
          (.inv (.sum n (fun j => .exp (.add (z j) (.neg s))))) := rfl

/-- `softmax z i = e^{zᵢ} / Σⱼ e^{zⱼ}` — the textbook form, kept for reference.
    Equal to `softmaxAt` at any shift in exact arithmetic, and **not** in
    `Float32`; `softmaxAt` is what runs. -/
def softmax (n : Nat) (z : Fin n → Expr Γ) (i : Fin n) : Expr Γ :=
  Expr.exp (z i) / .sum n (fun j => Expr.exp (z j))

/-- `rmsNorm w x i = xᵢ · wᵢ · rsqrt(Σⱼxⱼ²/n + ε)`. -/
def rmsNorm (n : Nat) (w x : Fin n → Expr Γ) (i : Fin n) : Expr Γ :=
  x i * w i * .rsqrt (.sum n (fun j => x j * x j) / .lit n + (1000000 : Expr Γ)⁻¹)

/-! **The operator form is the constructor form.**

    The combinators above are written with `+`/`*`/`-`/`/`/`⁻¹` rather than `Expr`
    constructors, which is only safe because every one of those instances is
    definitionally the constructor (`hAdd_def` and friends in `Expr.lean`).
    These pin that: change an instance and the math here changes silently
    otherwise.  Note `/` is *not* a constructor — `a / b` is `.mul a (.inv b)`
    — which is why `sigmoid` uses `⁻¹` and `softmax` uses `/`. -/
example (x : Expr Γ) : silu x = .mul x (sigmoid x) := rfl
example (x : Expr Γ) : sigmoid x = .inv (.add (.lit 1) (.exp (.neg x))) := rfl
example (n : Nat) (z : Fin n → Expr Γ) (i : Fin n) :
    softmax n z i = .mul (.exp (z i)) (.inv (.sum n (fun j => .exp (z j)))) := rfl
example (n : Nat) (w x : Fin n → Expr Γ) (i : Fin n) :
    rmsNorm n w x i
      = .mul (.mul (x i) (w i))
          (.rsqrt (.add (.mul (.sum n (fun j => .mul (x j) (x j))) (.inv (.lit n)))
                        (.inv (.lit 1000000)))) := rfl
example (n : Nat) (a b : Fin n → Expr Γ) : dot n a b = .sum n (fun k => .mul (a k) (b k)) := rfl

end Transformer
end AlgorithmLib.ML
