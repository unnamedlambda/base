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
    `z > 88` — and the reference Qwen2 implementation is no exception, so the
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
      x lo * cosT c - x hi * sinT c
    else
      let c : Fin half := ⟨i.val - half, by omega⟩
      let lo : Fin (half + half) := ⟨i.val - half, by omega⟩
      let hi : Fin (half + half) := ⟨i.val, by omega⟩
      x lo * sinT c + x hi * cosT c

/-- Scaled dot-product attention for one query head against a `seq`-long cache.

    `scale` is `1/√d` supplied as an expression (typically a literal ratio), so
    it too stays inside the language.

    `shift` is what the softmax subtracts before exponentiating.  It is a
    parameter for the reason given at `softmaxAt`: `Expr` has no `max`, and the
    kernel supplies the row maximum it computed.  `Qwen2Spec.softmax_is_spec`
    is the instantiation. -/
def attnHead (hd seq : Nat) (scale shift : Expr Γ)
    (q : Fin hd → Expr Γ) (K V : Fin seq → Fin hd → Expr Γ) :
    Fin hd → Expr Γ :=
  let score : Fin seq → Expr Γ := fun t => scale * dot hd q (K t)
  let p := softmaxAt seq shift score
  fun i => .sum seq (fun t => p t * V t i)

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
  K       : Fin seq → Fin hd → Expr Γ         -- KV cache
  V       : Fin seq → Fin hd → Expr Γ
  scale   : Expr Γ
  /-- The softmax shift — the row maximum, supplied because `Expr` has no
      `max`.  See `softmaxAt`. -/
  shift   : Expr Γ

/-- `x + Wo·attn(Wq·rmsnorm(x))`, then `h + Wd·(silu(Wg·ĥ) ⊙ (Wu·ĥ))`.

    One term.  Its gradient, its kernels, and their correctness proofs are all
    derived from it.

    **Corrected Jul 31 2026.**  This read
    `ao := matvec d I.wo (fun k => ad k * xn k)` — the output projection applied
    to the attention result *elementwise-multiplied by the normalised residual*,
    which is not a transformer and not what the docstring above claimed.  It was
    apparently a stand-in to keep `xn` live for the gradient while `q` was taken
    as an input, which also left `I.wq` unreferenced.  The fix is to project the
    query from `xn` with `wq`, as a decoder layer actually does: `xn` is then
    live for the right reason and `wq` is used.

    Nothing was proven against this definition — that is why the error survived
    — and nothing is yet: a full `layerPlan.denote = layer` needs equations for
    the two batched contractions inside `attnHead`, and NVIDIA specifies no fold
    order for batched `sgemm`, so none exists to state.  `Qwen2Spec.layer_is_spec`
    is how far the shipped layer is tied to the model: the same double-residual
    shape, with those two contractions left as plan memory.  The lesson is the
    one `ml_gap_hunt` already recorded — a definition with no theorem attached
    is unchecked no matter how many `sorry`-free files surround it. -/
def layer {d dff hd seq : Nat} (hhd : hd = d) (I : LayerIn Γ d dff hd seq) :
    Fin d → Expr Γ :=
  -- attention block
  let xn  : Fin d → Expr Γ := rmsNorm d I.rmsAttn I.x
  let q   : Fin hd → Expr Γ := fun i => matvec d I.wq xn (hhd ▸ i)
  let a   : Fin hd → Expr Γ := attnHead hd seq I.scale I.shift q I.K I.V
  let ad  : Fin d → Expr Γ := fun i => a (hhd ▸ i)
  let ao  : Fin d → Expr Γ := fun i => matvec d I.wo ad i
  let h   : Fin d → Expr Γ := vadd I.x ao
  -- feed-forward block
  let hn  : Fin d → Expr Γ := rmsNorm d I.rmsFfn h
  let g   : Fin dff → Expr Γ := fun j => silu (matvec d I.wg hn j)
  let u   : Fin dff → Expr Γ := fun j => matvec d I.wu hn j
  let gu  : Fin dff → Expr Γ := vmul g u
  let dn  : Fin d → Expr Γ := fun i => matvec dff I.wd gu i
  vadd h dn

/-! **The operator form is the constructor form.**

    The body above is written with `+`/`*`/`-`/`/`/`⁻¹` rather than `Expr`
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

/-- The attention head's probabilities are `softmaxAt`, at the supplied shift —
    pinned so the max-subtraction cannot be dropped without breaking the build.
    Dropping it is what `Qwen2Spec.softmax_is_spec` would then fail to prove. -/
example (hd seq : Nat) (scale shift : Expr Γ) (q : Fin hd → Expr Γ)
    (K V : Fin seq → Fin hd → Expr Γ) (i : Fin hd) :
    attnHead hd seq scale shift q K V i
      = .sum seq (fun t => .mul
          (softmaxAt seq shift (fun t' => .mul scale (dot hd q (K t'))) t) (V t i)) := rfl

/-- The output projection is applied to the attention result and nothing else —
    pinned because the elementwise `⊙ xn` that used to be here typechecked. -/
example {d dff hd seq : Nat} (hhd : hd = d) (I : LayerIn Γ d dff hd seq) (i : Fin d) :
    layer hhd I i
      = .add
          (.add (I.x i)
            (matvec d I.wo
              (fun k => attnHead hd seq I.scale I.shift
                (fun j => matvec d I.wq (rmsNorm d I.rmsAttn I.x) (hhd ▸ j))
                I.K I.V (hhd ▸ k)) i))
          (matvec dff I.wd
            (fun j => .mul
              (silu (matvec d I.wg
                (rmsNorm d I.rmsFfn (vadd I.x (fun k => matvec d I.wo
                  (fun k' => attnHead hd seq I.scale I.shift
                    (fun j' => matvec d I.wq (rmsNorm d I.rmsAttn I.x) (hhd ▸ j'))
                    I.K I.V (hhd ▸ k')) k))) j))
              (matvec d I.wu
                (rmsNorm d I.rmsFfn (vadd I.x (fun k => matvec d I.wo
                  (fun k' => attnHead hd seq I.scale I.shift
                    (fun j' => matvec d I.wq (rmsNorm d I.rmsAttn I.x) (hhd ▸ j'))
                    I.K I.V (hhd ▸ k')) k))) j))
            i) := rfl

end Transformer
end AlgorithmLib.ML
