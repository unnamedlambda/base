module
public import AlgorithmLib.ML.Launch.Stages
meta import AlgorithmLib.ML.Launch.Stages
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! # None -/

namespace AlgorithmLib.ML

-- A model as a graph of tensor operations
-- ---------------------------------------------------------------------------

/-!
  `Expr` is a *scalar* language, and `Frontend`'s `Vec`/`Mat` are thin wrappers
  over it — so `w * x` elaborates to `fun i => .sum inW (…)` and the fact that
  it *was* a matvec is gone.  Nothing can then decide whether to lower it to a
  proven warp kernel or to cuBLAS, because there is no longer an operation to
  lower.

  A `Node` keeps the operation.  Each one names its operands by *position in
  the graph*, and a node's output buffer **is** its own index — so the graph is
  simultaneously the program and the allocation, and there is no separate
  buffer table to keep in step with it.

  Malformed and stage-free are kept apart: `Node.stage?` returns `some none`
  for an input (nothing to launch) and `none` for an operation whose operands
  alias its output (nothing sound to launch).  Collapsing the two would let a
  bad node vanish from the plan instead of failing it.
-/


/-- One tensor operation, with the buffer it writes.

    The output is a field rather than the node's position: a backward pass
    visits buffers in an order that is not ascending — `dW2` before `dh` — so
    program order and allocation are genuinely two things. -/
inductive Node where
  /-- A buffer the graph does not compute: an input, a parameter, a label. -/
  | input  : Ref → Node
  /-- `out[s][o] = Σᵢ W[o][i]·x[s][i]`, one warp per output unit. -/
  | matvec : (w x out : Ref) → (b inW outW : Nat) → Node
  /-- `out[s][i] = Σₒ dy[s][o]·W[o][i]` — the transposed walk. -/
  | matvecT : (w dy out : Ref) → (b inW outW : Nat) → Node
  /-- `out[o][i] = Σₛ dy[s][o]·x[s][i]` — the batch sum, inside one warp. -/
  | outer  : (dy x out : Ref) → (b inW outW : Nat) → Node
  /-- An elementwise pass over `grid·32` elements. -/
  | ew     : {Γ : Nat} → Expr Γ → (Fin Γ → Ref) → Ref → Nat → Node
  /-- An elementwise pass that **reads what it writes** — an optimiser step. -/
  | ewIP   : {Γ : Nat} → Expr Γ → (Fin Γ → Ref) → Ref → Nat → Node
  /-- Softmax and the cross-entropy gradient, one warp per row. -/
  | smce   : (logits bias oneHot out : Ref) → Nat → Node
  /-- A two-operand row pass: one row per block, each operand read at its own
      broadcast mode.  This is how a statistic or a shared vector reaches an
      elementwise pass — the row index comes from the block, so no address
      arithmetic the index language cannot express is needed. -/
  | ziprow : (a b out : Ref) → (f : WFExp) → (mA mB : BCast) →
             (n off w rows : Nat) → Node
  /-- The same row pass with a third operand — a cotangent, a statistic, a
      gate.  This is the arity a nonlinear adjoint needs, and the one two
      fused elementwise steps reach. -/
  | ziprow3 : (a b c out : Ref) → (f : WFExp) → (mA mB mC : BCast) →
              (n off w rows : Nat) → Node
  /-- The same row pass with a fourth operand.  A chain of three row passes
      reaches this arity: two of them collapse into `ziprow3`, and the third
      has nowhere to go without it. -/
  | ziprow4 : (a b c d out : Ref) → (f : WFExp) → (mA mB mC mD : BCast) →
              (n off w rows : Nat) → Node
  /-- `out[s] = Σᵢ a[…]·b[…]` — one fold per block, each operand addressed by
      its own broadcast mode.  A row sum is this against a shared vector of
      ones; a softmax denominator is that. -/
  | rowdot : (a b out : Ref) → (mA mB : BCast) → (n rows : Nat) → Node
  /-- The same fold with a row pass folded into its left factor:
      `out[s] = Σᵢ f(a,b,c)[…]·d[…]`.  Only a fusion builds one. -/
  | rowdot4 : (a b c d out : Ref) → (f : WFExp) → (mA mB mC mD : BCast) →
              (n rows : Nat) → Node
  /-- `out[s] = maxᵢ x[s][i]`, seeded below anything the row can hold.  The
      value a stable softmax subtracts, and the one an argmax reads. -/
  | rowmax : (x out : Ref) → (n rows : Nat) → (init : Float32) → Node
  /-- `out[s] = Σᵢ x[s][i]·x[s][i]` — one row's sum of squares per block.
      The reduction half of an RMS norm, and of any row statistic that is a
      fold of a product. -/
  | rowsq  : (x out : Ref) → (n rows : Nat) → Node

/-- The stage a node lowers to.

    `none` — malformed (an operand aliases an output that may not be read).
    `some none` — nothing to launch.
    `some (some S)` — stage `S`. -/
noncomputable def Node.stage? (batch : Nat) : Node → Option (Option XStage)
  | .input _ => some none
  | .matvec w x out b inW outW =>
      if h : 0 < outW ∧ w ≠ out ∧ x ≠ out then
        some (some (dotBatchedStageX w x (stride32 (.mul .ctaId (.lit inW)))
          (fun s => stride32 (.lit (s * inW))) out b (inW / 32) outW h.1 h.2.1 h.2.2))
      else none
  | .matvecT w dy out b inW outW =>
      if h : 0 < inW ∧ w ≠ out ∧ dy ≠ out then
        some (some (dotBatchedStageX w dy
          (.add (.mul (.add (.mul .loopI (.lit 32)) .laneId) (.lit inW)) .ctaId)
          (fun s => stride32 (.lit (s * outW))) out b (outW / 32) inW
          h.1 h.2.1 h.2.2))
      else none
  | .outer dy x out b inW outW =>
      if h : (inW / 32) * 32 = inW ∧ dy ≠ out ∧ x ≠ out then
        some (some (outerBatchedStageX dy x out
          (fun s => .add (.lit (s * outW)) .ctaId)
          (fun s => stride32 (.lit (s * inW))) inW b (inW / 32) outW
          h.1 h.2.1 h.2.2))
      else none
  | .ew spec ins out grid =>
      if h : ∀ j, ins j ≠ out then some (some (mapStageX spec ins out grid h)) else none
  | .ewIP spec ins out grid => some (some (mapStageIPX spec ins out grid))
  | .smce logits bias oneHot out grid =>
      if h : logits ≠ out ∧ bias ≠ out ∧ oneHot ≠ out then
        some (some (softmaxCEStageX logits bias oneHot out .laneId grid h.1 h.2.1 h.2.2))
      else none
  | .ziprow a b out f mA mB n off w rows =>
      if h : f.pairOnly = true ∧ (w / 32) * 32 = w ∧ off + w ≤ n
             ∧ a ≠ out ∧ b ≠ out then
        some (some (zipRowStageX a b out f h.1 mA mB n off (w / 32) rows
          (by rw [h.2.1]; exact h.2.2.1) h.2.2.2.1 h.2.2.2.2))
      else none
  | .rowdot a b out mA mB n rows =>
      if h : (n / 32) * 32 = n ∧ a ≠ out ∧ b ≠ out then
        some (some (reduceStageX a b mA.ix mB.ix out (n / 32) rows h.2.1 h.2.2))
      else none
  | .ziprow3 a b c out f mA mB mC n off w rows =>
      if h : f.tripleOnly = true ∧ (w / 32) * 32 = w ∧ off + w ≤ n
             ∧ a ≠ out ∧ b ≠ out ∧ c ≠ out then
        some (some (zipRow3StageX a b c out f h.1 mA mB mC n off (w / 32) rows
          (by rw [h.2.1]; exact h.2.2.1) h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2))
      else none
  | .rowdot4 a b c d out f mA mB mC mD n rows =>
      if h : f.tripleOnly = true ∧ (n / 32) * 32 = n
             ∧ a ≠ out ∧ b ≠ out ∧ c ≠ out ∧ d ≠ out then
        some (some (reduce4StageX a b c d mA.ix mB.ix mC.ix mD.ix f h.1 out
          (n / 32) rows h.2.2.1 h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2))
      else none
  | .ziprow4 a b c d out f mA mB mC mD n off w rows =>
      if h : f.quadOnly = true ∧ (w / 32) * 32 = w ∧ off + w ≤ n
             ∧ a ≠ out ∧ b ≠ out ∧ c ≠ out ∧ d ≠ out then
        some (some (zipRow4StageX a b c d out f h.1 mA mB mC mD n off (w / 32) rows
          (by rw [h.2.1]; exact h.2.2.1) h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2.1
          h.2.2.2.2.2.2))
      else none
  | .rowmax x out n rows init =>
      if h : (n / 32) * 32 = n ∧ x ≠ out then
        some (some (maxRowStageX x (stride32 (.mul .ctaId (.lit n))) out
          (n / 32) rows init h.2))
      else none
  | .rowsq x out n rows =>
      if h : (n / 32) * 32 = n ∧ x ≠ out then
        some (some (reduceStageX x x (stride32 (.mul .ctaId (.lit n)))
          (stride32 (.mul .ctaId (.lit n))) out (n / 32) rows h.2 h.2))
      else none

/-- A model: operations in the order the driver performs them. -/
abbrev Net := List Node

/-- Lower the graph.  A malformed node fails the whole lowering rather than
    being skipped — collapsing "nothing to launch" into "nothing sound to
    launch" would let a bad node vanish from the plan instead of failing it. -/
noncomputable def lowerNet (batch : Nat) : Net → Option (List XStage)
  | []      => some []
  | n :: ns =>
      match n.stage? batch with
      | none          => none
      | some none     => lowerNet batch ns
      | some (some S) => (lowerNet batch ns).map (S :: ·)

/-- **The stages a model's graph lowers to**, in graph order. -/
noncomputable def Net.stages (batch : Nat) (g : Net) : Option (List XStage) :=
  lowerNet batch g

-- ---------------------------------------------------------------------------

end AlgorithmLib.ML
