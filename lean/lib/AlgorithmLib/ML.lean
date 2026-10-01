module
public import AlgorithmLib.ML.Assumptions
meta import AlgorithmLib.ML.Assumptions
public import AlgorithmLib.ML.Kernel.EmitFacts
meta import AlgorithmLib.ML.Kernel.EmitFacts
public import AlgorithmLib.ML.Model.Dense
meta import AlgorithmLib.ML.Model.Dense
public import AlgorithmLib.ML.Tensor.Surface
meta import AlgorithmLib.ML.Tensor.Surface
public import AlgorithmLib.ML.Model.Ten
meta import AlgorithmLib.ML.Model.Ten
public import AlgorithmLib.ML.Model.RegBound
meta import AlgorithmLib.ML.Model.RegBound
public import AlgorithmLib.ML.Model.Schedule
meta import AlgorithmLib.ML.Model.Schedule
public import AlgorithmLib.ML.Model.WeaveBuild
meta import AlgorithmLib.ML.Model.WeaveBuild
public import AlgorithmLib.ML.Model.WeaveTOp
meta import AlgorithmLib.ML.Model.WeaveTOp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The ML development

The tensor frontend, the autodiff, the kernel lowering and the proofs over
them. Separate from the LZ4 development because nothing needs both: importing
one is what keeps a model generator off the other's rebuild.
-/
