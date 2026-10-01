module
public import AlgorithmLib.ML.Launch.Stages
meta import AlgorithmLib.ML.Launch.Stages
public import AlgorithmLib.ML.Kernel.Sched
meta import AlgorithmLib.ML.Kernel.Sched
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! # None -/

namespace AlgorithmLib.ML

-- A dense layer, addressing derived
-- ---------------------------------------------------------------------------

/-!
  Everything above is still written in buffers, index expressions and grids.
  A model should not be.  `Dense` is the smallest useful step: give it two
  widths, a batch and six buffers, and it produces the three stages a dense
  layer needs — forward, input gradient, weight gradient — with every `IdxE`,
  every trip count and every grid *derived* from the shapes.

  The three addressing patterns are the ones a hand-written model gets wrong:
  the forward row walk, the **transposed** walk for `dx` (successive outputs
  are `inW` apart, which is why no `stride32` describes it), and the outer
  product's row base.  Deriving them once is the point.
-/

/-- A dense layer `y = W·x` and its two gradients, at one batch size. -/
structure Dense where
  inW   : Nat
  outW  : Nat
  batch : Nat
  w  : Buf   -- weights, row-major `W[o·inW + i]`
  x  : Buf   -- inputs  `x[s·inW + i]`
  y  : Buf   -- outputs `y[s·outW + o]`
  dy : Buf   -- `∂L/∂y`
  dx : Buf   -- `∂L/∂x`
  dw : Buf   -- `∂L/∂W`

/-- `y[s][o] = Σᵢ W[o][i]·x[s][i]` — one warp per output unit, `batch`
    accumulators, the weight row fetched once for the whole batch. -/
def Dense.fwdStage (d : Dense) (hg : 0 < d.outW := by decide)
    (h1 : d.w ≠ d.y := by decide) (h2 : d.x ≠ d.y := by decide) : XStage :=
  dotBatchedStageX d.w d.x (stride32 (.mul .ctaId (.lit d.inW)))
    (fun s => stride32 (.lit (s * d.inW))) d.y d.batch (d.inW / 32) d.outW hg h1 h2

/-- `dx[s][i] = Σₒ dy[s][o]·W[o][i]` — the transposed walk.  Successive outputs
    are `inW` apart, so this is the one index no `stride32` describes. -/
def Dense.dxStage (d : Dense) (hg : 0 < d.inW := by decide)
    (h1 : d.w ≠ d.dx := by decide) (h2 : d.dy ≠ d.dx := by decide) : XStage :=
  dotBatchedStageX d.w d.dy
    (.add (.mul (.add (.mul .loopI (.lit 32)) .laneId) (.lit d.inW)) .ctaId)
    (fun s => stride32 (.lit (s * d.outW))) d.dx d.batch (d.outW / 32) d.inW hg h1 h2

/-- `dW[o][i] = Σₛ dy[s][o]·x[s][i]` — summed over the batch *inside* one warp,
    because two blocks accumulating into one element would be a race. -/
def Dense.dwStage (d : Dense) (hn : (d.inW / 32) * 32 = d.inW := by decide)
    (h1 : d.dy ≠ d.dw := by decide) (h2 : d.x ≠ d.dw := by decide) : XStage :=
  outerBatchedStageX d.dy d.x d.dw
    (fun s => .add (.lit (s * d.outW)) .ctaId)
    (fun s => stride32 (.lit (s * d.inW)))
    d.inW d.batch (d.inW / 32) d.outW hn h1 h2

/-- **A dense layer's forward stage computes the flat dot product.**

    The kernel folds in the committed two-level order — sequential within a
    lane, then a five-round butterfly — and the model's `Transformer.dot` is a
    flat left fold.  They are the same number only up to reassociation, and
    that reassociation is `Law.laneRegroup`, named in the statement and the only
    thing assumed.  `Expr.denote` of `Transformer.dot n a b` is the fold on the
    right, over `List.finRange` rather than `List.range`.

    This is what makes `Dense` a lowering of a *spec* rather than a kernel with
    a convenient name. -/
theorem Dense.fwd_is_flatDot (h : AllHold [Law.laneRegroup]) (d : Dense)
    (hn : d.inW % 128 = 0) (hg : 0 < d.outW) (h1 : d.w ≠ d.y) (h2 : d.x ≠ d.y)
    (m : Buf → Nat → Float32) (cta a : Nat) :
    (d.fwdStage hg h1 h2).val.val m cta a
      = (List.range d.inW).foldl
          (fun acc i => NumOps.add acc
            (NumOps.mul (m d.w (i + cta * d.inW))
                        (m d.x (i + ((a - cta) / d.outW) * d.inW))))
          (NumOps.ofNat 0) :=
  by
  -- `stride32` writes the base first, `Sched.idx` last: commuted, not defeq.
  have hA : (fun (i : Nat) (l : Lane) =>
        IdxE.eval cta i l (fun _ _ => 0) (fun _ _ => 0)
          (stride32 (.mul .ctaId (.lit d.inW))))
      = Sched.strided.idx d.inW (cta * d.inW) := by
    funext i l
    show cta * d.inW + (i * 32 + l.val) = i * 32 + l.val + cta * d.inW
    omega
  have hB : (fun (i : Nat) (l : Lane) =>
        IdxE.eval cta i l (fun _ _ => 0) (fun _ _ => 0)
          (stride32 (.lit (((a - cta) / d.outW) * d.inW))))
      = Sched.strided.idx d.inW (((a - cta) / d.outW) * d.inW) := by
    funext i l
    show ((a - cta) / d.outW) * d.inW + (i * 32 + l.val)
        = i * 32 + l.val + ((a - cta) / d.outW) * d.inW
    omega
  show bflyFold (dotStridedLane (m d.w) (m d.x)
        (fun i l => IdxE.eval cta i l (fun _ _ => 0) (fun _ _ => 0)
          (stride32 (.mul .ctaId (.lit d.inW))))
        (fun i l => IdxE.eval cta i l (fun _ _ => 0) (fun _ _ => 0)
          (stride32 (.lit (((a - cta) / d.outW) * d.inW))))
        (d.inW / 32)) ⟨0, by decide⟩ = _
  rw [hA, hB]
  exact Sched.fold_eq_flatSum h .strided (m d.w) (m d.x)
    (cta * d.inW) (((a - cta) / d.outW) * d.inW) d.inW hn

-- ---------------------------------------------------------------------------

end AlgorithmLib.ML
