import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.CudaTensor

namespace CudaVecAddPersist

def x : AlgorithmLib.CudaTensor.Tensor 2 :=
  AlgorithmLib.CudaTensor.Tensor.input0

def y : AlgorithmLib.CudaTensor.Tensor 2 :=
  AlgorithmLib.CudaTensor.Tensor.input1

def result : CompileResult := (x + y).compileTo 1

-- Uncomment to see elaboration-time rejection:
--
-- def badOutput : CompileResult := (x + y).compileTo 2
-- -- impossible: output index 2 ∉ Fin 2

-- def badArity : CompileResult :=
--   (x + (AlgorithmLib.CudaTensor.Tensor.ofExpr
--     (AlgorithmLib.CudaTensor.input (n := 3) ⟨2, by decide⟩) :
--     AlgorithmLib.CudaTensor.Tensor 3)).compileTo 1
-- -- type error: Tensor 2 and Tensor 3 cannot be combined

def artifacts : Array Json :=
  let r := result
  #[
    toJsonEntry "cuda_vecadd_persist_load"  r.config r.loadAlgorithm,
    toJsonEntry "cuda_vecadd_persist_prep"  r.config r.prepAlgorithm,
    toJsonEntry "cuda_vecadd_persist_infer" r.config r.inferAlgorithm,
  ]

end CudaVecAddPersist
