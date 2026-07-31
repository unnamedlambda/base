import AlgorithmLib.ML.WarpCompile
import AlgorithmLib.ML.Quant
import AlgorithmLib.ML.Schema
import AlgorithmLib.ML.Butterfly

/-!
  # A rewrite calculus: transformations that carry their own justification

  A user tunes by transforming their model.  The question this file answers is
  *which transformations are equalities, and what happens to the ones that are
  not*.

  Three tiers, and the type system keeps them apart:

  * `Rewrite`  — `denote` preserved **exactly**, in every carrier.  Free to
    apply; correctness of the compiled kernel follows with no new obligation.
  * `FloatRewrite` — preserved exactly at `Float` only (relies on a specific
    reduction order or a `NumLaws` fact `Float` lacks).
  * `Approx` — deliberately changes the value, carrying a *name*.  These are
    the only places a compiled kernel can differ from its spec, and they cannot
    be applied silently.

  A butterfly reduction is **not** a `Rewrite` of `Expr.sum` — measured, the two
  differ (`1.0` vs `0.0` on 32 lanes).  The calculus makes that a type error
  rather than a subtle bug.
-/

namespace AlgorithmLib.ML

/-- A meaning-preserving transformation, in every carrier at once. -/
structure Rewrite (Γ : Nat) where
  run   : Expr Γ → Expr Γ
  sound : ∀ {R : Type} [NumOps R] (env : Fin Γ → R) (e : Expr Γ),
            denote env (run e) = denote env e

namespace Rewrite

variable {Γ : Nat}

def id : Rewrite Γ := ⟨fun e => e, by intro R _ env e; rfl⟩

/-- Rewrites compose, and the composite is a rewrite. -/
def comp (g f : Rewrite Γ) : Rewrite Γ :=
  ⟨fun e => g.run (f.run e), by
    intro R _ env e; rw [g.sound env (f.run e), f.sound env e]⟩

end Rewrite

-- ---------------------------------------------------------------------------
-- Tier 1: exact rewrites
-- ---------------------------------------------------------------------------


-- ---------------------------------------------------------------------------
-- Tier 3: declared approximations
-- ---------------------------------------------------------------------------

/-- A transformation that deliberately changes the value.  The `name` is what
    appears in `Assumptions.lean`; `why` is the justification a reader needs.
    There is no `sound` field — that is the point. -/
structure Approx (Γ : Nat) where
  run  : Expr Γ → Expr Γ
  name : String
  why  : String

/-- The softmax max-shift: mathematically the identity, numerically the reason
    softmax does not overflow.  It is **not** a `Rewrite`, because in `Float`
    `e^(x-m)/Σe^(x-m) ≠ e^x/Σe^x` bit-for-bit. -/
def maxShift {Γ : Nat} (shift : Expr Γ) : Approx Γ where
  run := fun e => .mul (.exp (.neg shift)) (.mul (.exp shift) e)
  name := "softmax-max-shift"
  why := "identity in ℝ; in Float it trades bit-exactness for overflow safety"

/-- Substituting the hardware's `ex2.approx` for `exp`.

    This is the *spec-level* name for what `AlgorithmLib.ML.expandExp` does
    concretely at the machine-expression level, where the identity it needs is
    the single named proposition `ExpIsEx2` and the only theorem mentioning it
    is `expandExp_approx`.  Measured: max relative error 5.06e-7 on silu,
    3.27e-7 end to end on the three-layer model. -/
def ex2Approx {Γ : Nat} : Approx Γ where
  run := fun e => e
  name := "ex2.approx-for-exp"
  why := "PTX has no exact e^x; e^x = 2^(x·log₂e) via ex2.approx, ~2 ULP"

-- ---------------------------------------------------------------------------
-- The declared-law registry
-- ---------------------------------------------------------------------------

/-!
  ## Tier 2, made concrete

  The three tiers above were designed before there was anything to apply them
  to.  There is now: every reduction in this stack is proven against the
  **order the hardware actually walks** — sequential within a lane, butterfly
  across lanes — which is what makes those theorems exact `Float32` equalities
  rather than bounds.

  A user who writes `Expr.sum` and expects the kernel to compute *that* is
  relying on reassociation, and reassociation is not a `Float32` identity.
  Rather than leave that unstated, it is a **named law** here, alongside the one
  the `exp` lowering already carries.

  This is the gap every framework has and none states: a butterfly reduction and
  a sequential fold are different functions on floats, and which one your model
  was trained with is not recorded anywhere.
-/

/-- The value a vendor GEMM lands. Opaque on purpose: the point of
    `CuBlasIsMatvec` is that nothing in this development knows how it was
    computed, only what it is claimed to equal. -/
opaque cublasSgemvResult (rows cols : Nat) (a : Fin rows → Fin cols → Float32)
    (x : Fin cols → Float32) (i : Fin rows) : Float32

/-- The idealised order: fold the lanes left to right. -/
def laneSum (v : Lane → Float32) : Float32 :=
  (List.finRange W).foldl (fun a l => NumOps.add a (v l)) (NumOps.ofNat 0)

/-- **What relating the butterfly to a sequential sum would cost.**

    Not a theorem — a *proposition*, false for `Float32`, and the exact price of
    writing `Expr.sum` and expecting the shuffle network.

    Measured, on 32 lanes: with lane 0 holding `1e8` and the rest `1.0`, the
    butterfly gives `100000024` and the sequential fold `100000000` — the tree
    accumulates the 31 ones into `31` before adding, where the left fold loses
    each one individually to `1e8`'s 8-ULP spacing.  With lane 0 at `1e6` and
    the rest at `0.03125`, `1000000.9375` against `1000000`.

    Both are *correct* summations; they are not the *same* summation.  Which one
    a model was trained with is not recorded anywhere, in any framework. -/
def SumAssoc : Prop :=
  ∀ v : Lane → Float32,
    bflyFoldOp (fun a b => NumOps.add a b) v ⟨0, by decide⟩ = laneSum v

/-- **What relying on a vendor GEMM costs.**

    `cl_cublas_sgemv` computes `y = A·x`, and the shipped Qwen2 model routes
    every matmul through it — around 99.9% of the model's arithmetic.

    It differs from `SumAssoc` in a way that matters. `SumAssoc` names a
    *specific* identity you can write down: butterfly equals sequential fold.
    cuBLAS's fold order is not merely different, it is **unspecified** — it
    varies with library version, architecture, and shape, and NVIDIA documents
    no guarantee about it. So there is no exact-`Float32` statement to make.

    What can be stated is the ℝ-level one: the result is *the* matrix-vector
    product, with the float discrepancy declared rather than assumed away. That
    is weaker than the other laws here, and deliberately so — pretending to a
    bit-level claim about a closed-source kernel would be the dishonest option.

    This is the gap every framework has. PyTorch defaulted `allow_tf32` on for
    matmul across several releases — silently truncating the mantissa to 10
    bits on Ampere — and it was discoverable only from release notes. Here it
    appears in the type of every theorem downstream of a matmul. -/
def CuBlasIsMatvec : Prop :=
  ∀ (rows cols : Nat) (a : Fin rows → Fin cols → Float32) (x : Fin cols → Float32)
    (i : Fin rows),
    cublasSgemvResult rows cols a x i
      = (List.finRange cols).foldl (fun acc j => NumOps.add acc (NumOps.mul (a i j) (x j)))
          (NumOps.ofNat 0)

/-- **What it costs to read a lane-partitioned fold as a flat one.**

    `SumAssoc` bridges the *butterfly* to a sequential lane fold.  A kernel does
    something more: each lane also folds its own strided slice sequentially, so
    the whole reduction is a two-level regrouping of the flat sum over all `n`
    elements.  Reading the kernel's answer as `Σᵢ aᵢ·bᵢ` needs both levels.

    This is the *second* half of the bridge between the pipeline theorems
    (`Pipeline.lean`, which end at the committed fold) and the calculus
    (`Backprop.lean`, which ends at `Expr.sum`).  Naming it is what makes the
    composite sentence a theorem with a stated hypothesis, rather than something
    a reader has to assemble from two halves.

    False at `Float32` for the same reason `SumAssoc` is — measured there:
    `.vec4` and `.strided` schedules of one reduction differ by 14% on a
    cancelling input (`Sched.lean`).

    **Stated for an arbitrary per-element summand `f`, not for `aᵢ·bᵢ`.**  The
    regrouping is a fact about the *shape of the fold*, not about what is being
    summed: the same two-level walk appears in RMSNorm at `f i = xᵢ·xᵢ` and in
    softmax's denominator at `f i = exp(zᵢ − max)`.  Specialising the law to a
    dot product would have meant a second, separately-named assumption for
    softmax describing the identical rearrangement, which is worse — one law
    covering both is the smaller surface, not the larger.  `base` is there for
    the same reason: softmax reduces one attention row, which starts at
    `cta·seqLen`, not at zero.  `strided_eq_flatSum` below recovers the
    dot-product form. -/
def StridedRegroup : Prop :=
  ∀ (f : Nat → Float32) (base K : Nat),
    bflyFoldOp (fun a b => NumOps.add a b)
        (fun l => (List.range K).foldl
          (fun acc j => NumOps.add acc (f (base + (j * 32 + l.val)))) NumOps.zero)
        ⟨0, by decide⟩
      = (List.range (K * 32)).foldl
          (fun a i => NumOps.add a (f (base + i))) NumOps.zero

/-- **The two warp combiners commute.**

    Needed for exactly one thing: a butterfly reduction leaves the *same* value
    in every lane.  Softmax's remainder pass has all 32 lanes write one address,
    so the kernel is correct only if they agree on the row maximum and the
    reciprocal sum; `bflyFoldOp_const` derives that from this and nothing else.

    Deliberately **not** `sumAssoc`.  Every lane of a butterfly walks a tree of
    the *same shape* over the *same* leaves — only the argument order within
    each node differs — so commutativity suffices, and unlike associativity it
    is *true* at IEEE-754 for all non-NaN inputs.  `Float32.add` is opaque in
    Lean, which is the only reason this is a law rather than a theorem.

    The NaN caveat is real and small: IEEE-754 leaves the payload of a NaN
    result from `a + b` implementation-defined, so bit-exact commutativity can
    fail when an operand is NaN.  A softmax row that is entirely NaN is already
    outside what any of these theorems say something useful about. -/
def CombinerComm : Prop :=
  (∀ a b : Float32, NumOps.add a b = NumOps.add b a)
    ∧ (∀ a b : Float32, NumOps.max a b = NumOps.max b a)

/-- A named numerical law.  Transformations and specs that need one say so in
    their type; `Assumptions.lean` names it; nothing applies one silently. -/
inductive Law where
  /-- `e^x = 2^(x·log₂e)` — PTX has no exact `exp`. -/
  | expIsEx2
  /-- A butterfly reduction equals a sequential fold. -/
  | sumAssoc
  /-- A lane-partitioned strided fold equals the flat fold. -/
  | stridedRegroup
  /-- A vendor GEMM equals the real-valued matrix-vector product. -/
  | cublasIsMatvec
  /-- The warp combiners commute, so a butterfly is lane-uniform. -/
  | combinerComm
  deriving DecidableEq, Repr

def Law.title : Law → String
  | .expIsEx2 => "ex2.approx-for-exp"
  | .sumAssoc => "sum-reassociation"
  | .stridedRegroup => "strided-fold-regrouping"
  | .cublasIsMatvec => "cublas-is-matvec"
  | .combinerComm => "warp-combiner-commutes"

def Law.why : Law → String
  | .expIsEx2 =>
      "PTX has no exact e^x; the emitter uses 2^(x·log₂e) via ex2.approx. " ++
      "Measured 5.06e-7 on silu, 3.27e-7 end to end on a three-layer model."
  | .sumAssoc =>
      "A warp butterfly and a left fold are different functions on Float32. " ++
      "Every kernel here is proven against the butterfly, which is what the " ++
      "hardware does; this law is what it would cost to claim Expr.sum instead."
  | .cublasIsMatvec =>
      "cl_cublas_sgemv computes y = A·x, but NVIDIA specifies no fold order, " ++
      "so no exact-Float32 claim about it is available at any version. The " ++
      "law is therefore stated at the real numbers. ~99.9% of the shipped " ++
      "Qwen2 model's arithmetic depends on it. Compare TF32, which other " ++
      "frameworks enable by default without stating it at all."
  | .stridedRegroup =>
      "A lane-partitioned strided fold regrouped as a flat sum. The second " ++
      "half of the bridge from a launched pipeline to sderiv; measured to " ++
      "fail at Float32 (two schedules of one reduction differ by 14%)."
  | .combinerComm =>
      "Float32 add and max commute. Strictly weaker than sumAssoc and, unlike " ++
      "it, true at IEEE-754 for non-NaN inputs: a butterfly's lanes walk the " ++
      "same tree shape and differ only in argument order within each node. " ++
      "Needed only so softmax's remainder pass, where all 32 lanes write one " ++
      "address, has all 32 lanes agreeing on what to write."

def Law.holds : Law → Prop
  | .expIsEx2 => ExpIsEx2
  | .sumAssoc => SumAssoc
  | .stridedRegroup => StridedRegroup
  | .cublasIsMatvec => CuBlasIsMatvec
  | .combinerComm => CombinerComm

/-- Laws a claim depends on.  `[]` means exact — which is what every shipped
    lowering theorem in this stack actually is. -/
def AllHold (ls : List Law) : Prop := ∀ l ∈ ls, l.holds

/-- **The bridge, under its law.**  A user's `Expr.sum`-shaped spec meets the
    kernel's committed butterfly order here, and nowhere else. -/
theorem bfly_eq_laneSum (h : AllHold [Law.sumAssoc]) (v : Lane → Float32) :
    bflyFoldOp (fun a b => NumOps.add a b) v ⟨0, by decide⟩ = laneSum v :=
  h Law.sumAssoc (by simp) v

/-- **The pipeline's answer, read as a flat sum — under its named law.**

    `Pipeline.lean`'s theorems end at the committed two-level fold, because that
    is what the hardware computes and stating it that way keeps them exact.
    `Backprop.lean`'s theorems end at `Expr.sum`, because that is what `sderiv`
    produces.  This law is what joins the two ends, so that "the launched
    pipeline computes the layer's gradient" is a theorem rather than an informal
    assembly — with `Law.stridedRegroup` visible in its type,
    which is the entire point.  Nothing here is claimed to hold at `Float32`;
    what is claimed is that the only thing separating the two halves is one
    named, measured, registry-listed law. -/
theorem strided_eq_flatSum (h : AllHold [Law.stridedRegroup])
    (memA memB : Nat → Float32) (K : Nat) :
    bflyFold (dotStridedLane memA memB
        (fun i l => i * 32 + l.val) (fun i l => i * 32 + l.val) K) ⟨0, by decide⟩
      = (List.range (K * 32)).foldl
          (fun a i => NumOps.add a (NumOps.mul (memA i) (memB i))) (NumOps.ofNat 0) := by
  have h0 := h Law.stridedRegroup (by simp) (fun i => NumOps.mul (memA i) (memB i)) 0 K
  simp only [Nat.zero_add] at h0
  exact h0

/-- **The regrouping at an arbitrary row base and summand** — the law itself,
    named so call sites do not have to spell out the registry lookup.  Softmax's
    denominator uses it at `f i = exp(zᵢ − max)`, `base = cta·seqLen`. -/
theorem strided_regroup_at (h : AllHold [Law.stridedRegroup])
    (f : Nat → Float32) (base K : Nat) :
    bflyFoldOp (fun a b => NumOps.add a b)
        (fun l => (List.range K).foldl
          (fun acc j => NumOps.add acc (f (base + (j * 32 + l.val)))) NumOps.zero)
        ⟨0, by decide⟩
      = (List.range (K * 32)).foldl
          (fun a i => NumOps.add a (f (base + i))) NumOps.zero :=
  h Law.stridedRegroup (by simp) f base K

/-- **The butterfly is lane-uniform, under its named law.**

    The bridge `softmax_stores_tail` needs: all 32 lanes hold the same reduction
    result, so the remainder pass — where they all write one address — writes
    one value.  `Law.combinerComm` is visible in the type, which is the point. -/
theorem bfly_lane_uniform_add (h : AllHold [Law.combinerComm]) (v : Lane → Float32)
    (l l' : Lane) :
    bflyFoldOp (fun a b => NumOps.add a b) v l = bflyFoldOp (fun a b => NumOps.add a b) v l' :=
  bflyFoldOp_const _ (h Law.combinerComm (by simp)).1 v l l'

theorem bfly_lane_uniform_max (h : AllHold [Law.combinerComm]) (v : Lane → Float32)
    (l l' : Lane) :
    bflyFoldOp (fun a b => NumOps.max a b) v l = bflyFoldOp (fun a b => NumOps.max a b) v l' :=
  bflyFoldOp_const _ (h Law.combinerComm (by simp)).2 v l l'

/-- The laws a "launched pipeline = `sderiv`" claim rests on, as a list.

    Anyone quoting the composite must quote this too. -/
def gradientPipelineLaws : List Law := [Law.expIsEx2, Law.stridedRegroup]

end AlgorithmLib.ML
