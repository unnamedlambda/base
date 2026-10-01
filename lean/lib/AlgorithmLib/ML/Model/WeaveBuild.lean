import AlgorithmLib.ML.Model.WeaveLower

/-!
  # Writing a model in the algebra

  `WOp` is what compiles, but a list of them is not how anyone wants to write a
  model: the buffers are numbers, and every operation restates the shapes its
  neighbours already fixed.

  This is the surface.  A `WVal` is a buffer together with the shape it holds,
  so an operation *derives* its result's shape from its operands' and allocates
  the buffer itself.  Writing `contract w x` is enough: the output extent, the
  contracted extent, the launch chunking and the buffer number all follow.  A
  shape disagreement is not silently emitted — the build fails, and
  `mismatch_rejected` shows the check is not vacuous.

  Nothing is added below `WOp`: `runBuild?` produces an ordinary program, so a
  model written here inherits `lower_den` and everything under it unchanged.
-/

namespace AlgorithmLib.ML.Broadcast

open AlgorithmLib.ML

/-- A value: the buffer holding it, and the shape it holds. -/
structure WVal where
  buf   : Buf
  shape : Ix
  deriving Repr, DecidableEq

/-- The builder's state: the next free buffer, the operations so far, and
    whether every shape has agreed. -/
structure BState where
  next : Nat
  ops  : List WOp
  ok   : Bool

abbrev WBuild := StateM BState

/-- The launch chunking a pointwise pass over a value needs: one lane per
    element, thirty-two lanes to a chunk. -/
def chunksOf (s : Ix) : Ix := [Ix.size s / 32, 32]

/-- A buffer the model is given rather than one it computes. -/
def input (b : Buf) (s : Ix) : WVal := { buf := b, shape := s }

/-- Allocate the next buffer, at a shape. -/
def alloc (s : Ix) : WBuild WVal := do
  let st ← get
  set { st with next := st.next + 1 }
  return { buf := st.next, shape := s }

def emit (op : WOp) : WBuild Unit :=
  modify fun st => { st with ops := st.ops ++ [op] }

/-- Record that two shapes that had to agree did not. -/
def reject : WBuild Unit :=
  modify fun st => { st with ok := false }

/-- `out[s][o] = Σₜ w[o][t] · x[s][t]` — the contracted extent must agree. -/
def contract (bk : Backend) (w x : WVal) : WBuild WVal :=
  match w.shape, x.shape with
  | [outW, inW], [b, inW'] => do
      if inW != inW' then reject
      let o ← alloc [b, outW]
      emit (.contract bk w.buf x.buf o.buf [b, outW] [inW] b)
      return o
  | _, _ => do reject; alloc []

/-- The transposed walk: `w : [outW, inW]`, `d : [b, outW]`, result `[b, inW]`. -/
def contractT (bk : Backend) (w d : WVal) : WBuild WVal :=
  match w.shape, d.shape with
  | [outW, inW], [b, outW'] => do
      if outW != outW' then reject
      let o ← alloc [b, inW]
      emit (.contractT bk w.buf d.buf o.buf [b, inW] [outW])
      return o
  | _, _ => do reject; alloc []

/-- The batch-summed outer product: `d : [b, outW]`, `x : [b, inW]`, result
    `[outW, inW]` — the summed extent must agree. -/
def outerProd (bk : Backend) (d x : WVal) : WBuild WVal :=
  match d.shape, x.shape with
  | [b, outW], [b', inW] => do
      if b != b' then reject
      let o ← alloc [outW, inW]
      emit (.outerProd bk d.buf x.buf o.buf [outW, inW] [b])
      return o
  | _, _ => do reject; alloc []

/-- A pointwise pass keeps the shape and picks up the chunking. -/
def pointwise1 (f : Expr 1) (a : WVal) : WBuild WVal := do
  let o ← alloc a.shape
  emit (.pointwise1 f a.buf o.buf (chunksOf a.shape))
  return o

/-- The two-operand pointwise pass — both operands must hold the same shape. -/
def pointwise2 (f : Expr 2) (a b : WVal) : WBuild WVal := do
  if a.shape != b.shape then reject
  let o ← alloc a.shape
  emit (.pointwise2 f a.buf b.buf o.buf (chunksOf a.shape))
  return o

/-- Softmax with the cross-entropy gradient folded in. -/
def softmaxCE (l bias oh : WVal) : WBuild WVal := do
  if l.shape != oh.shape then reject
  let o ← alloc l.shape
  emit (.softmaxCE l.buf bias.buf oh.buf o.buf (chunksOf l.shape))
  return o

/-- The optimiser step: writes its target in place, so it allocates nothing. -/
def update2 (f : Expr 2) (target grad : WVal) : WBuild Unit := do
  if target.shape != grad.shape then reject
  emit (.update2 f target.buf grad.buf (chunksOf target.shape))

/-- Run a model, allocating computed buffers from `base` upward.  `none` when a
    shape disagreement was recorded. -/
def runBuild? (base : Nat) (b : WBuild α) : Option (List WOp) :=
  let st := (b.run { next := base, ops := [], ok := true }).2
  if st.ok then some st.ops else none

end AlgorithmLib.ML.Broadcast
