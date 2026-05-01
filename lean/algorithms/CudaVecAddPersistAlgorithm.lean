import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.CudaTensor
open scoped AlgorithmLib.CudaTensor

namespace CudaVecAddPersist

def x : Expr 2 := AlgorithmLib.CudaTensor.input (n := 2) ⟨0, by decide⟩
def y : Expr 2 := AlgorithmLib.CudaTensor.input (n := 2) ⟨1, by decide⟩

def result : CompileResult := AlgorithmLib.CudaTensor.compile {
  expr := x + y
  output := ⟨1, by decide⟩
}

-- Uncomment to see elaboration-time rejection:
--
-- def badOutput : CompileResult := AlgorithmLib.CudaTensor.compile {
--   expr := x + y
--   output := ⟨2, by decide⟩   -- impossible: output index 2 ∉ Fin 2
-- }

-- def badArity : CompileResult := AlgorithmLib.CudaTensor.compile {
--   expr := x + AlgorithmLib.CudaTensor.input (n := 3) ⟨2, by decide⟩
--   -- type error: Expr 2 and Expr 3 cannot be combined
--   output := ⟨1, by decide⟩
-- }

end CudaVecAddPersist

def main : IO Unit := do
  IO.println (Json.compress (toJson CudaVecAddPersist.result))
