module
public import AlgorithmLib.ML.Launch.Backend
meta import AlgorithmLib.ML.Launch.Backend
public import AlgorithmLib.ML.Model.Graph
meta import AlgorithmLib.ML.Model.Graph
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! # None -/

namespace AlgorithmLib.ML

-- Writing the graph as equations
-- ---------------------------------------------------------------------------

/-!
  A `Net` written as a list still names its own buffers.  The builder below
  removes that too: each operation *returns* the buffer it wrote, so a model is
  a sequence of bindings and the allocation is whatever the binder order
  produced.

      let z1 ← mv   w1 x  IN H
      let h  ← elem hSpec ![z1] GRIDH
      let y  ← mv   w2 h  H  C

  That is the `let h := silu (w1 * x)` surface, in the one form Lean gives for
  free: `do`-notation binds the name, and the name *is* the value, so an
  operand that does not exist is a scope error rather than a wrong number.
-/

/-- A tensor's extent.  A weight is `outW × inW`; an activation is
    `batch × width`; a per-class vector is `1 × C`.  Two numbers is all the
    shapes in this stack need, and it is enough to derive every width, trip
    count and launch grid. -/
structure Shape where
  rows : Nat
  cols : Nat
  deriving Repr, DecidableEq

/-- The builder records, per binding, the operation **and** which
    implementation it chose.  A model program leaves every choice `.proven`; a
    *schedule* is the same program with some choices replaced, and `forget`
    below is what relates the two. -/
structure BuildState where
  next   : Nat := 0
  nodes  : List (Node × Backend) := []
  shapes : List Shape := []
  /-- Cleared by any shape mismatch, so a malformed model yields `none` rather
      than a plausible graph. -/
  ok     : Bool := true

abbrev NetM := StateM BuildState

def shapeOf (r : Ref) : NetM Shape :=
  fun st => (st.shapes.getD r ⟨0, 0⟩, st)

def failBuild : NetM Unit := fun st => ((), { st with ok := false })

/-- Allocate the next buffer, at a known shape. -/
def alloc (s : Shape) : NetM Ref :=
  fun st => (st.next, { st with next := st.next + 1, shapes := st.shapes ++ [s] })

def emitAt (n : Node) (b : Backend) : NetM Unit :=
  fun st => ((), { st with nodes := st.nodes ++ [(n, b)] })

/-- `W · x`, at the given implementation.  Widths and grid are **derived**: the
    contraction is `W.cols = x.cols`, and a mismatch fails the build. -/
def mvWith (b : Backend) (w x : Ref) : NetM Ref := do
  let sw ← shapeOf w
  let sx ← shapeOf x
  if sw.cols ≠ sx.cols then failBuild
  let o ← alloc ⟨sx.rows, sw.rows⟩
  emitAt (.matvec w x o sx.rows sw.cols sw.rows) b
  pure o

/-- `Wᵀ · dy` — the input gradient. -/
def mvTWith (b : Backend) (w dy : Ref) : NetM Ref := do
  let sw ← shapeOf w
  let sd ← shapeOf dy
  if sd.cols ≠ sw.rows then failBuild
  let o ← alloc ⟨sd.rows, sw.cols⟩
  emitAt (.matvecT w dy o sd.rows sw.cols sw.rows) b
  pure o

/-! The model-level names: every choice `.proven`.  A schedule uses the `With`
    forms above to say otherwise. -/
def mv      : Ref → Ref → NetM Ref := mvWith .proven
def mvT     : Ref → Ref → NetM Ref := mvTWith .proven
/-- **A schedule, with its choices erased, is a model.**  The relation between
    the two programs: same operations, same operands, same allocation — only
    the implementations differ. -/
def forget (ps : List (Node × Backend)) : Net := ps.map Prod.fst

/-- Lower a scheduled graph: each node to its stage, carrying the choice the
    schedule made.  A malformed node fails the whole lowering. -/
noncomputable def lowerPairs (batch : Nat) :
    List (Node × Backend) → Option (List (Backend × XStage))
  | []           => some []
  | (n, b) :: ps =>
      match n.stage? batch with
      | none          => none
      | some none     => lowerPairs batch ps
      | some (some S) => (lowerPairs batch ps).map ((b, S) :: ·)

/-- **The stages a scheduled graph lowers to do not depend on the schedule.**

    Erasing the choices gives the model, and the model determines the stages —
    so a schedule can only change *which implementation* runs, never *what*
    runs.  This is the structural half of the guarantee; `lowerAll_denote` is
    the semantic half. -/
theorem lowerPairs_forget (batch : Nat) :
    ∀ (ps : List (Node × Backend)),
      (lowerPairs batch ps).map (List.map Prod.snd) = lowerNet batch (forget ps) := by
  intro ps
  induction ps with
  | nil => rfl
  | cons p ps ih =>
      obtain ⟨n, b⟩ := p
      show (match n.stage? batch with
            | none => none
            | some none => lowerPairs batch ps
            | some (some S) => (lowerPairs batch ps).map ((b, S) :: ·)).map _
          = match n.stage? batch with
            | none => none
            | some none => lowerNet batch (forget ps)
            | some (some S) => (lowerNet batch (forget ps)).map (S :: ·)
      cases n.stage? batch with
      | none => rfl
      | some o =>
          cases o with
          | none => exact ih
          | some S =>
              rw [← ih]
              cases lowerPairs batch ps <;> rfl

/-- **Two programs that erase to the same model lower to the same stages.**
    The schedule-checking theorem: write the schedule as a program, and if it
    is the model with implementations chosen, nothing about *what* is computed
    can have moved. -/
theorem schedule_agrees (batch : Nat) (sched model : List (Node × Backend))
    (h : forget sched = forget model) :
    (lowerPairs batch sched).map (List.map Prod.snd)
      = (lowerPairs batch model).map (List.map Prod.snd) := by
  rw [lowerPairs_forget, lowerPairs_forget, h]

/-- **A node's decidable signature**: its kind, every buffer it names, and every
    width and grid it carries.

    `Node` cannot have `DecidableEq` — `ew` holds an `Expr` whose `sum` case
    carries a function, and its arity is implicit — so an erasure check has to
    compare something decidable.  This captures **the whole wiring**: which
    operation, which operands (`List.ofFn` unfolds the elementwise inputs), the
    output, the widths and the grid.

    What it does *not* compare is the elementwise `Expr` itself.  A schedule
    that silently substituted a different activation would pass this check, so
    a schedule is checked to have the same *structure*, not the same
    arithmetic. -/
def Node.sig : Node → Nat × List Ref × List Nat
  | .input o                     => (0, [o], [])
  | .matvec w x o b inW outW     => (1, [w, x, o], [b, inW, outW])
  | .matvecT w dy o b inW outW   => (2, [w, dy, o], [b, inW, outW])
  | .outer dy x o b inW outW     => (3, [dy, x, o], [b, inW, outW])
  | .ew (Γ := Γ) _ ins o grid    => (4, List.ofFn ins ++ [o], [Γ, grid])
  | .ewIP (Γ := Γ) _ ins o grid  => (5, List.ofFn ins ++ [o], [Γ, grid])
  | .smce l b oh o grid          => (6, [l, b, oh, o], [grid])
  | .rowsq x o n r               => (7, [x, o], [n, r])
  | .rowdot a b o mA mB n r      => (9, [a, b, o], [mA.tag, mB.tag, n, r])
  | .rowmax x o n r _            => (11, [x, o], [n, r])
  | .ziprow a b o _ mA mB n f w r => (8, [a, b, o], [mA.tag, mB.tag, n, f, w, r])
  | .ziprow3 a b c o _ mA mB mC n f w r =>
      (10, [a, b, c, o], [mA.tag, mB.tag, mC.tag, n, f, w, r])
  | .ziprow4 a b c d o _ mA mB mC mD n f w r =>
      (12, [a, b, c, d, o], [mA.tag, mB.tag, mC.tag, mD.tag, n, f, w, r])
  | .rowdot4 a b c d o _ mA mB mC mD n r =>
      (13, [a, b, c, d, o], [mA.tag, mB.tag, mC.tag, mD.tag, n, r])

-- ---------------------------------------------------------------------------

end AlgorithmLib.ML
