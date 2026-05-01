import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.CudaTensor
open scoped AlgorithmLib.CudaTensor

namespace CudaSaxpyPersist

def alpha : Expr 2 := AlgorithmLib.CudaTensor.const (n := 2) "0f40000000"
def x : Expr 2 := AlgorithmLib.CudaTensor.input (n := 2) ⟨0, by decide⟩
def y : Expr 2 := AlgorithmLib.CudaTensor.input (n := 2) ⟨1, by decide⟩

def result : CompileResult := AlgorithmLib.CudaTensor.compile {
  expr := alpha * x + y
  output := ⟨1, by decide⟩
}

-- Uncomment to see elaboration-time rejection:
--
-- def badOutput : CompileResult := AlgorithmLib.CudaTensor.compile {
--   expr := alpha * x + y
--   output := ⟨3, by decide⟩   -- impossible: output index 3 ∉ Fin 2
-- }

-- def badMixedArity : CompileResult := AlgorithmLib.CudaTensor.compile {
--   expr := alpha * x + AlgorithmLib.CudaTensor.input (n := 1) ⟨0, by decide⟩
--   -- type error: Expr 2 and Expr 1 cannot be combined
--   output := ⟨1, by decide⟩
-- }

end CudaSaxpyPersist

def main : IO Unit := do
  IO.println (Json.compress (toJson CudaSaxpyPersist.result))
