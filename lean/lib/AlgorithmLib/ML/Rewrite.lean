import AlgorithmLib.ML.WarpCompile
import AlgorithmLib.ML.Quant
import AlgorithmLib.ML.Schema

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
    cancelling input (`Sched.lean`). -/
def StridedRegroup : Prop :=
  ∀ (memA memB : Nat → Float32) (K : Nat),
    bflyFold (dotStridedLane memA memB
        (fun i l => i * 32 + l.val) (fun i l => i * 32 + l.val) K) ⟨0, by decide⟩
      = (List.range (K * 32)).foldl
          (fun a i => NumOps.add a (NumOps.mul (memA i) (memB i))) (NumOps.ofNat 0)

/-- A named numerical law.  Transformations and specs that need one say so in
    their type; `Assumptions.lean` names it; nothing applies one silently. -/
inductive Law where
  /-- `e^x = 2^(x·log₂e)` — PTX has no exact `exp`. -/
  | expIsEx2
  /-- A butterfly reduction equals a sequential fold. -/
  | sumAssoc
  /-- A lane-partitioned strided fold equals the flat fold. -/
  | stridedRegroup
  deriving DecidableEq, Repr

def Law.title : Law → String
  | .expIsEx2 => "ex2.approx-for-exp"
  | .sumAssoc => "sum-reassociation"
  | .stridedRegroup => "strided-fold-regrouping"

def Law.why : Law → String
  | .expIsEx2 =>
      "PTX has no exact e^x; the emitter uses 2^(x·log₂e) via ex2.approx. " ++
      "Measured 5.06e-7 on silu, 3.27e-7 end to end on a three-layer model."
  | .sumAssoc =>
      "A warp butterfly and a left fold are different functions on Float32. " ++
      "Every kernel here is proven against the butterfly, which is what the " ++
      "hardware does; this law is what it would cost to claim Expr.sum instead."
  | .stridedRegroup =>
      "A lane-partitioned strided fold regrouped as a flat sum. The second " ++
      "half of the bridge from a launched pipeline to sderiv; measured to " ++
      "fail at Float32 (two schedules of one reduction differ by 14%)."

def Law.holds : Law → Prop
  | .expIsEx2 => ExpIsEx2
  | .sumAssoc => SumAssoc
  | .stridedRegroup => StridedRegroup

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
          (fun a i => NumOps.add a (NumOps.mul (memA i) (memB i))) (NumOps.ofNat 0) :=
  h Law.stridedRegroup (by simp) memA memB K

/-- The laws a "launched pipeline = `sderiv`" claim rests on, as a list.

    Anyone quoting the composite must quote this too. -/
def gradientPipelineLaws : List Law := [Law.expIsEx2, Law.stridedRegroup]

end AlgorithmLib.ML
