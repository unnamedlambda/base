module
public import Lean
public import Std
public import AlgorithmLib.Gen
meta import AlgorithmLib.Gen
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.CudaPipeline

namespace CudaVecAddPersist

def x : Expr 2 := Expr.input0
def y : Expr 2 := Expr.input1

def result : Except String Artifact := (x + y).compileTo 1

-- Uncomment to see elaboration-time rejection:
--
-- def badOutput : Except String Artifact := (x + y).compileTo 2
-- -- impossible: output index 2 ∉ Fin 2
--
-- def badArity : Except String Artifact :=
--   (x + (Expr.input (n := 3) ⟨2, by decide⟩ : Expr 3)).compileTo 1
-- -- type error: Expr 2 and Expr 3 cannot be combined

def artifacts (r : Artifact) : Array ArtifactEntry := #[artifactEntry "cuda_vecadd_persist" r]

end CudaVecAddPersist
