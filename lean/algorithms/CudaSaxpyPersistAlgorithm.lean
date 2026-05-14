import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.CudaPipeline

namespace CudaSaxpyPersist

def alpha : Expr 2 := Expr.scalarBits "0f40000000"
def x : Expr 2 := Expr.input0
def y : Expr 2 := Expr.input1

def result : CompileResult := (Expr.saxpy alpha x y).compileTo 1

-- Uncomment to see elaboration-time rejection:
--
-- def badOutput : CompileResult := (Expr.saxpy alpha x y).compileTo 3
-- -- impossible: output index 3 ∉ Fin 2
--
-- def badMixedArity : CompileResult :=
--   (alpha * x + (Expr.input0 : Expr 1)).compileTo 1
-- -- type error: Expr 2 and Expr 1 cannot be combined

def artifacts : Array Json := result.toArtifacts "cuda_saxpy_persist"

end CudaSaxpyPersist
