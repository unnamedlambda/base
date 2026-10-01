import AlgorithmLib.ML.Model.WeaveBuild
import Warp.MlpCifar
/-!
  # The CIFAR model, written in the algebra

  `MlpCifarAlgorithm` writes the model as a `Ten` term and ships the tape it
  flattens to.  This module writes the same model as a list of broadcasted
  operations — a target primitive, the shape its result ranges over, and the
  shape it contracts away — and shows the two meet.

  Nothing here is a re-implementation.  `cifarW_lowers` says the algebra
  program lowers to the tape the model already ships, *definitionally*: the
  same operations, the same buffers, the same extents, and therefore the same
  PTX and the same artifact bytes.  What the algebra adds is that the grid and
  every extent are **derived** from the shapes rather than written down, and
  that `cifarW_computes` gives the shipped tape a denotation stated in the
  algebra's terms — which then inherits the whole chain below it, down to the
  launch sequence, through `TenProg.run_den`.
-/

namespace WeaveCifar

open AlgorithmLib.ML AlgorithmLib.ML.Broadcast MlpCifar

/-- The activation the model uses, as the scalar expression `Ten.silu` builds. -/
def siluSpec : Expr 1 := Transformer.silu (.var ⟨0, by decide⟩)

/-- **The model, as broadcasted operations.**

    Read against `MlpCifar.cifarTen`: each `tlet` there is one entry here.  No
    entry carries a grid or a lane count — `[64, 32]` is a shape, and the
    launch geometry is what `lower` computes from it. -/
def cifarW : List WOp :=
  [ -- forward
    .contract .proven w1B xB z1B [B, H] [IN] B
  , .pointwise1 siluSpec z1B hB [B * H / 32, 32]
  , .contract .proven w2B hB logB [B, C] [H] B
  , .softmaxCE logB biasB ohB dlogB [B * C / 32, 32]
    -- backward
  , .outerProd .proven dlogB hB dw2B [C, H] [B]
  , .contractT .proven w2B dlogB dhB [B, H] [C]
  , .pointwise2 adjSpec z1B dhB adjB [B * H / 32, 32]
  , .outerProd .proven adjB xB dw1B [H, IN] [B]
    -- the optimiser step
  , .update2 sgdSpec w1B dw1B [H * IN / 32, 32]
  , .update2 sgdSpec w2B dw2B [C * H / 32, 32]
  ]

/-- **The same model, written compositionally.**

    No buffer numbers and no extents: every operation derives its result's
    shape from its operands' and allocates its own buffer, and the launch
    chunking follows from the shape.  Read against `cifarW` above, this is what
    the algebra buys — the shapes are the program, and the geometry is a
    consequence. -/
def cifarBuild : WBuild WVal := do
  let x    := input xB    [B, IN]
  let w1   := input w1B   [H, IN]
  let w2   := input w2B   [C, H]
  let bias := input biasB [1, C]
  let oh   := input ohB   [B, C]
  -- forward
  let z1   ← contract .proven w1 x
  let h    ← pointwise1 siluSpec z1
  let y    ← contract .proven w2 h
  let dlog ← softmaxCE y bias oh
  -- backward
  let dw2  ← outerProd .proven dlog h
  let dh   ← contractT .proven w2 dlog
  let adj  ← pointwise2 adjSpec z1 dh
  let dw1  ← outerProd .proven adj x
  -- the optimiser step
  update2 sgdSpec w1 dw1
  update2 sgdSpec w2 dw2
  return dw1

/-- **The compositional surface produces the algebra program**, definitionally.

    So the two ways of writing this model are the same program, and the one
    written without a single buffer number or extent is the one that ships. -/
theorem cifarBuild_is_cifarW : runBuild? NIN cifarBuild = some cifarW := rfl

/-- **The algebra program lowers to the tape the model ships** — the same
    operations, definitionally, so the emitted PTX and the artifact bytes are
    untouched by this layer. -/
theorem cifarW_lowers : WOp.lowerAll cifarW = some cifarTape := rfl

/-- …and therefore so does the model as written. -/
theorem cifarBuild_lowers :
    (runBuild? NIN cifarBuild).bind WOp.lowerAll = some cifarTape := by
  rw [cifarBuild_is_cifarW]
  exact cifarW_lowers

/-- **The shape check is not vacuous**: a contraction whose inner extents
    disagree is refused, rather than emitted and left to mean something else. -/
theorem mismatch_rejected :
    runBuild? NIN (contract .proven (input w1B [H, IN]) (input xB [B, C])) = none := rfl

/-- Ten operations in, ten out: the lowering dropped nothing. -/
theorem cifarW_length : List.length cifarW = 10 := rfl

/-- **The shipped tape computes the algebra program's denotation.** -/
theorem cifarW_computes (m : Buf → Nat → Float32) :
    cifarTape.foldl (fun mm o => o.den mm) m = WOp.denAll cifarW m :=
  WOp.lowerAll_den cifarW cifarTape cifarW_lowers m

/-- **The launch sequence the model lowers to performs the algebra program.**

    The right-hand side mentions only broadcasted operations and shapes; the
    left-hand side is the pipeline of stages that actually runs.  This is the
    statement the paper's framework cannot make about its own compilation,
    because there the backend is trusted rather than derived. -/
theorem cifar_stages_run_the_algebra (ss : List XStage)
    (h : cifarTenProgram.stages B NIN = some ss) (st : WSt) :
    ((Pipeline.ofStages ss).run st).mem = WOp.denAll cifarW st.mem := by
  rw [TenProg.run_den B NIN cifarTenProgram ss h st]
  exact cifarW_computes st.mem

/-- The stages really are the ones the model ships, so the theorem above is
    about the emitted pipeline and not about an arbitrary one. -/
theorem cifar_shipped_stages_run_the_algebra (st : WSt) :
    ((Pipeline.ofStages (fwdStages ++ bwdStages)).run st).mem
      = WOp.denAll cifarW st.mem :=
  cifar_stages_run_the_algebra _ cifarTen_lowers st

end WeaveCifar
