import AlgorithmLib.ML.Kernels
import AlgorithmLib.ML.Transformer
import AlgorithmLib.ML.MultiLayer
import AlgorithmLib.ML.Layered

/-!
  # A function-first surface

  `Expr Γ` is a de Bruijn term, and writing one by hand shows it:
  `.var ⟨0, by decide⟩` is not what a model looks like.  The operator instances
  in `Expr.lean` fixed the *arithmetic*; this file fixes the *binding*.

  A user writes an ordinary Lean function of Γ arguments:

      def silu1 : Expr 1 := ofFn 1 fun a => a * sigmoid a

  and `ofFn` applies it to the variables.  Nothing is interpreted or reflected —
  `ofFn f` is **definitionally** `f (var 0) (var 1) …`, so every theorem about
  the resulting `Expr` sees straight through, and `rfl` proves that a
  function-written spec equals the hand-written one.  That is the test that this
  is sugar and not a second language.

  ## Shapes are checked when you write them

  `Vec n Γ` is a length-indexed vector of scalar expressions.  `dot` takes two
  of the *same* length, `matVec` an `m × n` matrix and an `n`-vector.  A
  mismatch is a Lean type error at elaboration — before any kernel is emitted,
  before anything runs, with the two lengths named in the message.  This is the
  "shape errors at elaboration time" claim, and it is not a separate checker:
  it is the ordinary type checker on ordinary indexed families.

  ## And it reaches the GPU with no proof text

  `MapKernel.ofFn` composes this surface with the zero-proof front door in
  `Kernels.lean`: a Lean function in, a proven kernel plus its PTX out.
-/

namespace AlgorithmLib.ML

variable {Γ : Nat}

-- ---------------------------------------------------------------------------
-- Lean functions as specs
-- ---------------------------------------------------------------------------

/-- A `k`-argument function returning an `Expr Γ`. -/
def Curried : Nat → Nat → Type
  | 0,     Γ => Expr Γ
  | k + 1, Γ => Expr Γ → Curried k Γ

/-- Apply a curried function to variables `s, s+1, …, s+k-1`, in order. -/
def applyVars : (k : Nat) → {Γ : Nat} → (s : Nat) → s + k ≤ Γ → Curried k Γ → Expr Γ
  | 0,     _, _, _, e => e
  | k + 1, _, s, h, f => applyVars k (s + 1) (by omega) (f (.var ⟨s, by omega⟩))

/-- **Write the spec as a function; get the term.**

    `ofFn 2 (fun a b => a * b)` is `.mul (.var 0) (.var 1)` — definitionally, so
    this costs nothing downstream.

    The arity is explicit rather than inferred: `Curried Γ Γ` cannot be unified
    against a bare lambda before `Γ` is known, and writing the number is also
    the clearest possible statement of how many inputs the spec takes. -/
def ofFn (Γ : Nat) (f : Curried Γ Γ) : Expr Γ := applyVars Γ 0 (by omega) f

/-- The unfolding, at the arities specs actually use.  Each is `rfl`, which is
    the point: `ofFn` is not a translation step that could go wrong. -/
@[simp] theorem ofFn_one (f : Curried 1 1) : ofFn 1 f = f (.var ⟨0, by decide⟩) := rfl
@[simp] theorem ofFn_two (f : Curried 2 2) :
    ofFn 2 f = f (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩) := rfl
@[simp] theorem ofFn_three (f : Curried 3 3) :
    ofFn 3 f = f (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩) (.var ⟨2, by decide⟩) := rfl

/-- `let v = a in body v` — the binder, function-first.  The body receives the
    bound variable, so nothing counts de Bruijn indices by hand.

    This is the constructor that keeps terms linear rather than exponential
    (`Expr.lean`), so making it pleasant to write is not cosmetic. -/
def letIn (a : Expr Γ) (body : Expr (Γ + 1) → Expr (Γ + 1)) : Expr Γ :=
  .letE a (body (.var ⟨Γ, Nat.lt_succ_self Γ⟩))

-- ---------------------------------------------------------------------------
-- The smooth functions, by name
-- ---------------------------------------------------------------------------

/-!
  These are **not** new definitions.  `Transformer.lean` already fixes what
  `silu`, `softmax` and `rmsNorm` mean, and the shipped kernels are proven
  against those.  A frontend that re-spelled them would be a second, silently
  divergent definition of the model — which is the exact failure this stack
  exists to prevent.  So each name below is an abbreviation, and the shipped
  specs are `rfl`-equal to what a user writes here.
-/

/-- `eˣ`.  Lowered via `ex2.approx`, the stack's one declared approximation. -/
def expE (a : Expr Γ) : Expr Γ := .exp a

/-- `1/√x` — one instruction on the hardware, hence a constructor. -/
def rsqrtE (a : Expr Γ) : Expr Γ := .rsqrt a

/-- `σ(x) = 1/(1 + e⁻ˣ)` — `Transformer.sigmoid`. -/
abbrev sigmoid (a : Expr Γ) : Expr Γ := Transformer.sigmoid a

/-- `silu(x) = x·σ(x)` — `Transformer.silu`, the one the MLP demo, the Qwen2
    gate kernel and the silu demo all already use. -/
abbrev silu (a : Expr Γ) : Expr Γ := Transformer.silu a

/-- A finite reduction, in the **committed** left-fold order.  Writing this is
    choosing that order; see `Law.sumAssoc` for what claiming any other would
    cost. -/
def sumOver (n : Nat) (f : Fin n → Expr Γ) : Expr Γ := .sum n f

-- ---------------------------------------------------------------------------
-- Shapes
-- ---------------------------------------------------------------------------

/-!
  `Transformer`'s operations already take their length as an argument, so shape
  agreement is *already* enforced — `dot n a b` will not accept a `Fin m → Expr`
  for `b`.  What is missing is that the length has to be written out at every
  call.  `Vec n Γ` makes it an index instead, so it is inferred and checked
  rather than repeated.

  Every operation here is definitionally its `Transformer` counterpart; the
  theorems below say so, by `rfl`.
-/

/-- A length-`n` vector of scalar expressions. -/
def Vec (n : Nat) (Γ : Nat) : Type := Fin n → Expr Γ

/-- A matrix is `m` rows of length `n`. -/
def Mat (m n : Nat) (Γ : Nat) : Type := Fin m → Vec n Γ

/-!
  ### Building vectors without writing proofs

  A vector of *variables* needs index arithmetic, and index arithmetic in a
  dependently typed language means bounds proofs.  Every constructor below
  discharges its own bound with an **auto-param** (`:= by omega`), so the
  obligation is real, checked, and never appears in user code — the same
  arrangement `Expr.x` already uses for a single variable.  The tactic is
  `first | omega | decide`: `omega` for symbolic widths, `decide` for the
  concrete numerals a real model uses (where the width is a product and
  `omega` cannot multiply).

  The test of this is the demo: a model written end to end with no `by`, no
  `⟨_, _⟩`, and no `Fin` literal. -/

/-- The `Γ` model inputs, as a vector — the usual starting point. -/
def inputs (Γ : Nat) : Vec Γ Γ := fun i => .var i

/-- `n` inputs starting at `s`.  The bound is discharged by the auto-param, so
    `slice 0 896` and `slice 896 896` are what a user writes. -/
def slice (s n : Nat) {Γ : Nat} (h : s + n ≤ Γ := by first | omega | decide) :
    Vec n Γ :=
  fun i => .var ⟨s + i.val, by omega⟩

/-- An `m × n` matrix read row-major from inputs starting at `s`. -/
def matSlice (s m n : Nat) {Γ : Nat}
    (h : s + m * n ≤ Γ := by first | omega | decide) : Mat m n Γ :=
  fun i j => .var ⟨s + (i.val * n + j.val), by have := rowMajor_lt i j; omega⟩

/-- Every component the same expression. -/
def Vec.const {n : Nat} (c : Expr Γ) : Vec n Γ := fun _ => c

/-- Every component the same numeral. -/
def Vec.lit {n : Nat} (k : Nat) : Vec n Γ := fun _ => .lit k

/-- Every entry the same expression. -/
def Mat.const {m n : Nat} (c : Expr Γ) : Mat m n Γ := fun _ => Vec.const c

/-- Every entry the same numeral. -/
def Mat.lit {m n : Nat} (k : Nat) : Mat m n Γ := fun _ => Vec.lit k


/-- Pointwise application. -/
def Vec.map {n : Nat} (f : Expr Γ → Expr Γ) (v : Vec n Γ) : Vec n Γ := fun i => f (v i)

/-- Pointwise sum — `Transformer.vadd`.  Both operands must have the *same*
    length: a mismatch is a type error here, not a runtime shape assertion. -/
def Vec.add {n : Nat} (a b : Vec n Γ) : Vec n Γ := Transformer.vadd a b

/-- Pointwise product — `Transformer.vmul`. -/
def Vec.hadamard {n : Nat} (a b : Vec n Γ) : Vec n Γ := Transformer.vmul a b

/-- Scale by a scalar expression. -/
def Vec.scale {n : Nat} (s : Expr Γ) (v : Vec n Γ) : Vec n Γ := fun i => s * v i

/-- Inner product, in the committed order — `Transformer.dot`. -/
def Vec.dot {n : Nat} (a b : Vec n Γ) : Expr Γ := Transformer.dot n a b

/-- Sum of squares. -/
def Vec.sumSq {n : Nat} (v : Vec n Γ) : Expr Γ := v.dot v

/-- `W · v` — `Transformer.matvec`.  The `n`s must agree; that is the check. -/
def matVec {m n : Nat} (w : Mat m n Γ) (v : Vec n Γ) : Vec m Γ :=
  Transformer.matvec n w v

/-- RMSNorm — `Transformer.rmsNorm`, epsilon and all. -/
def rmsNorm {n : Nat} (gamma v : Vec n Γ) : Vec n Γ :=
  fun i => Transformer.rmsNorm n gamma v i

/-- Softmax — `Transformer.softmax`, the mathematically exact form. -/
def softmax {n : Nat} (z : Vec n Γ) : Vec n Γ := fun i => Transformer.softmax n z i

/-!
  ### Vectors get the operators too

  `Expr` has `+ * - / -x` and numerals (`Expr.lean`); without the same on
  `Vec`, a model reads `(a.add b).hadamard c` where the scalar case reads
  `a * b`.  These instances close that, and — as with the scalar ones — each is
  definitionally the underlying `Transformer` operation, so no theorem needs
  restating.

  `*` on vectors is **elementwise**, matching `Transformer.vmul`.  The inner
  product is `Vec.dot`, spelled out, because a `*` that silently contracts is
  the kind of ambiguity a shape-checked surface exists to remove. -/

instance {n : Nat} : Add (Vec n Γ) := ⟨Vec.add⟩
instance {n : Nat} : Mul (Vec n Γ) := ⟨Vec.hadamard⟩
instance {n : Nat} : Neg (Vec n Γ) := ⟨fun v => fun i => -(v i)⟩
instance {n : Nat} : Sub (Vec n Γ) := ⟨fun a b => fun i => a i - b i⟩
instance {n : Nat} : OfNat (Vec n Γ) k := ⟨Vec.lit k⟩

/-- Scalar × vector. -/
instance {n : Nat} : HMul (Expr Γ) (Vec n Γ) (Vec n Γ) := ⟨Vec.scale⟩

/-- Matrix × vector — the one place `*` does contract, and the shapes make it
    unambiguous. -/
instance {m n : Nat} : HMul (Mat m n Γ) (Vec n Γ) (Vec m Γ) := ⟨matVec⟩

@[simp] theorem vec_add_def {n : Nat} (a b : Vec n Γ) : a + b = Vec.add a b := rfl
@[simp] theorem vec_mul_def {n : Nat} (a b : Vec n Γ) : a * b = Vec.hadamard a b := rfl
@[simp] theorem vec_smul_def {n : Nat} (s : Expr Γ) (v : Vec n Γ) :
    s * v = Vec.scale s v := rfl
@[simp] theorem mat_mul_def {m n : Nat} (w : Mat m n Γ) (v : Vec n Γ) :
    w * v = matVec w v := rfl

-- ---------------------------------------------------------------------------
-- Stacking layers *with sharing*
-- ---------------------------------------------------------------------------

/-!
  A surface this convenient makes it easy to write the wrong thing.  Composing
  a layer three times by ordinary function application —
  `layer (layer (layer x))` — copies the whole previous layer into every
  component of the next, and measured at width 8 that is **369,227,633 nodes**
  for three layers.  Function composition is not sharing.

  `stackLayers` is the combinator that binds each layer's whole output vector
  (`bindVec`, whose measurement in `MultiLayer.lean` is what motivated it), so
  the term is linear in depth.  A user who writes a model with this gets
  sharing without knowing the word.
-/

/-- **Stack `n` layers, binding each layer's output.**  Linear in depth, where
    naive composition is exponential.

    `layer` is polymorphic in the context because each binding extends it —
    that is precisely what lets the next layer refer to *variables* rather than
    to a copy of the previous layer's term. -/
def stackLayers {d : Nat} (layer : {Δ : Nat} → Vec d Δ → Vec d Δ)
    (readout : {Δ : Nat} → Vec d Δ → Expr Δ) :
    Nat → {Γ : Nat} → Vec d Γ → Expr Γ
  | 0,     _, x => readout x
  | n + 1, Γ, x =>
      bindVec d (layer x) (stackLayers layer readout n (fun i => boundVar Γ i))

/-!
  ### Parameters, across a stack

  `stackLayers` needs a layer that works at *any* context, because each bound
  layer extends it.  But a model's parameters come from `slice`/`matSlice` and
  live at the **base** context — so a layer that mentions them is not
  polymorphic, and a user writing a deep model with real weights hits a
  weakening obligation.  That is a proof in user code, which is exactly what
  this surface exists to avoid.

  `stackParams` removes it: the combinator hands the layer a weakener `ρ` for
  each level, already composed.  The user writes one layer, at one context,
  applies `ρ` to the parameters, and never sees a bound or a `wkTo`.
-/

/-- Push a scalar context map through a vector. -/
def Vec.mapE {n Δ Δ' : Nat} (ρ : Expr Δ → Expr Δ') (v : Vec n Δ) : Vec n Δ' :=
  fun i => ρ (v i)

/-- …and through a matrix. -/
def Mat.mapE {m n Δ Δ' : Nat} (ρ : Expr Δ → Expr Δ') (w : Mat m n Δ) : Mat m n Δ' :=
  fun i => (w i).mapE ρ

/-- **Stack `n` layers whose parameters live in the base context.**

    `ρ` is the weakener from the base context into the current one, composed by
    the combinator as it descends — one `wkBy d` per bound layer.  The user's
    layer is written once and applies `ρ` to whatever parameters it reads.

    Sharing is `bindVec`, as in `stackLayers`, so the term stays linear in
    depth. -/
def stackParams {d Γ₀ : Nat}
    (layer : {Δ : Nat} → (Expr Γ₀ → Expr Δ) → Vec d Δ → Vec d Δ)
    (readout : {Δ : Nat} → (Expr Γ₀ → Expr Δ) → Vec d Δ → Expr Δ) :
    Nat → {Δ : Nat} → (Expr Γ₀ → Expr Δ) → Vec d Δ → Expr Δ
  | 0,     _, ρ, x => readout ρ x
  | n + 1, Δ, ρ, x =>
      bindVec d (layer ρ x)
        (stackParams layer readout n (fun e => wkBy d (ρ e)) (fun i => boundVar Δ i))

/-- The entry point: start at the base context with the identity weakener. -/
def runStack {d Γ₀ : Nat}
    (layer : {Δ : Nat} → (Expr Γ₀ → Expr Δ) → Vec d Δ → Vec d Δ)
    (readout : {Δ : Nat} → (Expr Γ₀ → Expr Δ) → Vec d Δ → Expr Δ)
    (n : Nat) (x : Vec d Γ₀) : Expr Γ₀ :=
  stackParams layer readout n (fun e => e) x

/-- A dense layer with a SiLU: `silu(W·x)`. -/
def denseSilu {m n : Nat} (w : Mat m n Γ) (v : Vec n Γ) : Vec m Γ :=
  (matVec w v).map silu

/-! **The surface is the same model.**  Each of these is `rfl`, which is what
    makes the shape layer sugar rather than a fork. -/

theorem Vec.dot_eq {n : Nat} (a b : Vec n Γ) : a.dot b = Transformer.dot n a b := rfl
theorem Vec.add_eq {n : Nat} (a b : Vec n Γ) : a.add b = Transformer.vadd a b := rfl
theorem Vec.hadamard_eq {n : Nat} (a b : Vec n Γ) :
    a.hadamard b = Transformer.vmul a b := rfl
theorem matVec_eq {m n : Nat} (w : Mat m n Γ) (v : Vec n Γ) :
    matVec w v = Transformer.matvec n w v := rfl
theorem rmsNorm_eq {n : Nat} (g v : Vec n Γ) (i : Fin n) :
    rmsNorm g v i = Transformer.rmsNorm n g v i := rfl
theorem softmax_eq {n : Nat} (z : Vec n Γ) (i : Fin n) :
    softmax z i = Transformer.softmax n z i := rfl

-- ---------------------------------------------------------------------------
-- …to a proven kernel, with no proof text
-- ---------------------------------------------------------------------------

/-- **A Lean function in, a proven kernel out.**

    Composes `ofFn` with `mapKernel`, so the user writes arithmetic and receives
    the `EWStmt`, the PTX, and the theorem that the output buffer holds the
    spec's denotation at every lane — as *fields*, not obligations. -/
def kernelOfFn (Γ : Nat) (f : Curried Γ Γ) (inB : Fin Γ → Buf) (out : Buf) :
    MapKernel Γ := mapKernel (ofFn Γ f) inB out

/-- …and the emitted text runs it, from raw launch.  Inherited, not restated. -/
theorem mapKernel_ofFn_ptx_exact (Γ : Nat) (f : Curried Γ Γ) (inB : Fin Γ → Buf)
    (out : Buf) (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW (kernelOfFn Γ f inB out).ew)) k (0, m)
          = some ((flatKernel (expandEW (kernelOfFn Γ f inB out).ew)).length, m')
      ∧ m'.toWSt = ((expandEW (kernelOfFn Γ f inB out).ew).elabIn cta).run m.toWSt :=
  mapKernel_ptx_exact (ofFn Γ f) inB out cta m

end AlgorithmLib.ML
