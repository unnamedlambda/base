import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.CudaPipeline

namespace CudaVecAddPersist

def x : Expr 2 := Expr.input0
def y : Expr 2 := Expr.input1

def result : CompileResult := (x + y).compileTo 1

-- Uncomment to see elaboration-time rejection:
--
-- def badOutput : CompileResult := (x + y).compileTo 2
-- -- impossible: output index 2 ∉ Fin 2
--
-- def badArity : CompileResult :=
--   (x + (Expr.input (n := 3) ⟨2, by decide⟩ : Expr 3)).compileTo 1
-- -- type error: Expr 2 and Expr 3 cannot be combined

def artifacts : Array Json := result.toArtifacts "cuda_vecadd_persist"

end CudaVecAddPersist
