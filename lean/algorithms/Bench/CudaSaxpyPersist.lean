module
public import AlgorithmLib.ML.Kernel.Pipeline
meta import AlgorithmLib.ML.Kernel.Pipeline
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open AlgorithmLib
open AlgorithmLib.ML

namespace CudaSaxpyPersist

/-- `a·x + y` at `a = 2`, over inputs `x` and `y`. -/
def spec : Expr 2 := 2 * x 0 + x 1

def result : Except String Artifact := Pipeline.compile spec

def artifacts (r : Artifact) : Array ArtifactEntry := #[artifactEntry "cuda_saxpy_persist" r]

end CudaSaxpyPersist
