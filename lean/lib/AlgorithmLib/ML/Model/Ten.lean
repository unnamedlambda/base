module
public import AlgorithmLib.ML.Tensor.Ten
meta import AlgorithmLib.ML.Tensor.Ten
public import AlgorithmLib.ML.Launch.Backend
meta import AlgorithmLib.ML.Launch.Backend
public import AlgorithmLib.ML.Model.Graph
meta import AlgorithmLib.ML.Model.Graph
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A tape, as graph nodes

Where the model meets the lowering: each tape operation as the graph node the
stage builders take, with its backend choice.
-/

namespace AlgorithmLib.ML

/-- The graph node an operation is, with its implementation choice. -/
def TOp.node : TOp → Node × Backend
  | .mv bk w x o b i ow _  => (.matvec w x o b i ow, bk)
  | .mvT bk w d o b i ow => (.matvecT w d o b i ow, bk)
  | .outer bk d x o b i ow => (.outer d x o b i ow, bk)
  | .ew1 f a o g         => (.ew f (fun _ => a) o g, .proven)
  | .ew2 f a b o g       => (.ew f (fun j : Fin 2 => if j.val = 0 then a else b) o g, .proven)
  | .ew3 f a b c o g     => (.ew f (fun j : Fin 3 => if j.val = 0 then a else if j.val = 1 then b else c) o g, .proven)
  | .ew4 f a b c d o g   => (.ew f (fun j : Fin 4 => if j.val = 0 then a else if j.val = 1 then b else if j.val = 2 then c else d) o g, .proven)
  | .smce l bi oh o g    => (.smce l bi oh o g, .proven)
  | .rowsq x o n r       => (.rowsq x o n r, .proven)
  | .rowdot a b o mA mB n r => (.rowdot a b o mA mB n r, .proven)
  | .rowmax x o n r i    => (.rowmax x o n r i, .proven)
  | .ziprow a b o f mA mB n k w r => (.ziprow a b o f mA mB n k w r, .proven)
  | .ziprow3 a b c o f mA mB mC n k w r =>
      (.ziprow3 a b c o f mA mB mC n k w r, .proven)
  | .ziprow4 a b c d o f mA mB mC mD n k w r =>
      (.ziprow4 a b c d o f mA mB mC mD n k w r, .proven)
  | .rowdot4 a b c d o f mA mB mC mD n r =>
      (.rowdot4 a b c d o f mA mB mC mD n r, .proven)
  | .upd2 f a b g        => (.ewIP f (fun j : Fin 2 => if j.val = 0 then a else b) a g, .proven)

/-- The graph a term flattens to, with computed buffers allocated from
    `base` upward. -/
def Ten.graphOf {r c : Nat} (t : Ten RefV r c) (base : Ref) : List (Node × Backend) :=
  ((t.flat base).2.2).map TOp.node

-- ---------------------------------------------------------------------------

end AlgorithmLib.ML
