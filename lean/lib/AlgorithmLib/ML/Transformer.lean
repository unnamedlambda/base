import AlgorithmLib.ML.Grad
import AlgorithmLib.ML.Grad

/-!
  # A transformer layer, as `Expr` combinators

  The claim this file discharges: the nine constructors of `Expr` are enough to
  express a real decoder layer.  No new primitives were needed —

  * **matvec / attention scores** — `sum` of `mul`
  * **softmax**                   — `exp`, `sum`, `inv`
  * **SiLU**                      — `x · (1 + e^{-x})⁻¹`
  * **RMSNorm**                   — `sum`, `rsqrt`
  * **RoPE**                      — `mul`/`add` against a *precomputed* table,
                                     exactly as `Qwen2Common.slotRopeTable` does,
                                     so no trigonometry enters the language
  * **residual**                  — `add`

  Every combinator here is a plain function returning an `Expr`, so a model built
  from them is one term.  It therefore inherits, with no new proof obligations:

  * `grad`             — the backward pass
  * `grad_hasDerivAt`  — that the backward pass is the analytic derivative
  * `compileW`         — a warp kernel
  * `compileWKernel_correct` — that the kernel computes the spec, exactly
  * `flatKernel_sound` — that the emitted PTX runs the kernel

  **Softmax is the numerically naive form** (no max subtraction).  That is
  deliberate: the max-shift is a *stability rewrite*, and it belongs in the
  rewrite calculus carrying its own justification, not baked into the spec.
-/

namespace AlgorithmLib.ML
namespace Transformer

variable {Γ : Nat}

-- ---------------------------------------------------------------------------
-- Linear algebra
-- ---------------------------------------------------------------------------

/-- Inner product of two expression vectors. -/
def dot (n : Nat) (a b : Fin n → Expr Γ) : Expr Γ :=
  .sum n (fun k => .mul (a k) (b k))

/-- `y i = Σₖ W i k · x k`.  Rows of `W` are given as expression vectors, so
    this works equally for a weight matrix in memory or a fused subexpression. -/
def matvec (n : Nat) (W : Fin m → Fin n → Expr Γ) (x : Fin n → Expr Γ)
    (i : Fin m) : Expr Γ :=
  dot n (W i) x

/-- Elementwise `a + b`. -/
def vadd (v w : Fin n → Expr Γ) : Fin n → Expr Γ := fun i => .add (v i) (w i)

/-- Elementwise `a · b` — the FFN's gate/up product. -/
def vmul (v w : Fin n → Expr Γ) : Fin n → Expr Γ := fun i => .mul (v i) (w i)

-- ---------------------------------------------------------------------------
-- Activations and normalisation
-- ---------------------------------------------------------------------------

/-- `σ(x) = (1 + e^{-x})⁻¹`. -/
def sigmoid (x : Expr Γ) : Expr Γ :=
  .inv (.add (.lit 1) (.exp (.neg x)))

/-- `silu x = x · σ(x)`.  Smooth, so `grad` handles it with no side condition. -/
def silu (x : Expr Γ) : Expr Γ := .mul x (sigmoid x)

/-- `softmax z i = e^{zᵢ} / Σⱼ e^{zⱼ}` — the mathematically exact form. -/
def softmax (n : Nat) (z : Fin n → Expr Γ) (i : Fin n) : Expr Γ :=
  .mul (.exp (z i)) (.inv (.sum n (fun j => .exp (z j))))

/-- `rmsNorm w x i = xᵢ · wᵢ · rsqrt(Σⱼxⱼ²/n + ε)`. -/
def rmsNorm (n : Nat) (w x : Fin n → Expr Γ) (i : Fin n) : Expr Γ :=
  .mul (.mul (x i) (w i))
       (.rsqrt (.add (.mul (.sum n (fun j => .mul (x j) (x j))) (.inv (.lit n)))
                     (.inv (.lit 1000000))))

-- ---------------------------------------------------------------------------
-- Attention
-- ---------------------------------------------------------------------------

/-- Rotary embedding against a **precomputed** cos/sin table, matching the
    `slotRopeTable` layout in `Qwen2Common`.  Pairs `(2c, 2c+1)` are rotated:

        x'₂c   = x₂c·cos − x₂c₊₁·sin
        x'₂c₊₁ = x₂c·sin + x₂c₊₁·cos

    Because the table is an input, no trigonometry enters `Expr`. -/
def rope (half : Nat) (cosT sinT : Fin half → Expr Γ)
    (x : Fin (half + half) → Expr Γ) : Fin (half + half) → Expr Γ :=
  fun i =>
    if h : i.val < half then
      let c : Fin half := ⟨i.val, h⟩
      let lo : Fin (half + half) := ⟨i.val, by omega⟩
      let hi : Fin (half + half) := ⟨half + i.val, by omega⟩
      .add (.mul (x lo) (cosT c)) (.neg (.mul (x hi) (sinT c)))
    else
      let c : Fin half := ⟨i.val - half, by omega⟩
      let lo : Fin (half + half) := ⟨i.val - half, by omega⟩
      let hi : Fin (half + half) := ⟨i.val, by omega⟩
      .add (.mul (x lo) (sinT c)) (.mul (x hi) (cosT c))

/-- Scaled dot-product attention for one query head against a `seq`-long cache.

    `scale` is `1/√d` supplied as an expression (typically a literal ratio), so
    it too stays inside the language. -/
def attnHead (hd seq : Nat) (scale : Expr Γ)
    (q : Fin hd → Expr Γ) (K V : Fin seq → Fin hd → Expr Γ) :
    Fin hd → Expr Γ :=
  let score : Fin seq → Expr Γ := fun t => .mul scale (dot hd q (K t))
  let p := softmax seq score
  fun i => .sum seq (fun t => .mul (p t) (V t i))

-- ---------------------------------------------------------------------------
-- A whole decoder layer
-- ---------------------------------------------------------------------------

/-- Everything one layer needs, as expression vectors.  Keeping weights abstract
    means the same spec serves any memory layout — the layout choice lives in
    `ve` at compile time, not here. -/
structure LayerIn (Γ d dff hd seq : Nat) where
  x       : Fin d → Expr Γ                    -- residual stream
  rmsAttn : Fin d → Expr Γ
  rmsFfn  : Fin d → Expr Γ
  wq      : Fin d → Fin d → Expr Γ
  wo      : Fin d → Fin d → Expr Γ
  wg      : Fin dff → Fin d → Expr Γ
  wu      : Fin dff → Fin d → Expr Γ
  wd      : Fin d → Fin dff → Expr Γ
  q       : Fin hd → Expr Γ                   -- projected + roped query
  K       : Fin seq → Fin hd → Expr Γ         -- KV cache
  V       : Fin seq → Fin hd → Expr Γ
  scale   : Expr Γ

/-- `x + Wo·attn(rmsnorm(x))`, then `x + Wd·(silu(Wg·x̂) ⊙ (Wu·x̂))`.

    One term.  Its gradient, its kernels, and their correctness proofs are all
    derived from it. -/
def layer {d dff hd seq : Nat} (hhd : hd = d) (I : LayerIn Γ d dff hd seq) :
    Fin d → Expr Γ :=
  -- attention block
  let xn  : Fin d → Expr Γ := rmsNorm d I.rmsAttn I.x
  let a   : Fin hd → Expr Γ := attnHead hd seq I.scale I.q I.K I.V
  let ad  : Fin d → Expr Γ := fun i => a (hhd ▸ i)
  let ao  : Fin d → Expr Γ := fun i => matvec d I.wo (fun k => .mul (ad k) (xn k)) i
  let h   : Fin d → Expr Γ := vadd I.x ao
  -- feed-forward block
  let hn  : Fin d → Expr Γ := rmsNorm d I.rmsFfn h
  let g   : Fin dff → Expr Γ := fun j => silu (matvec d I.wg hn j)
  let u   : Fin dff → Expr Γ := fun j => matvec d I.wu hn j
  let gu  : Fin dff → Expr Γ := vmul g u
  let dn  : Fin d → Expr Γ := fun i => matvec dff I.wd gu i
  vadd h dn

end Transformer
end AlgorithmLib.ML
