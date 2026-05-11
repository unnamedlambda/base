import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.CudaTensor

namespace CudaSaxpyPersist

def alpha : AlgorithmLib.CudaTensor.Tensor 2 :=
  AlgorithmLib.CudaTensor.Tensor.scalarBits "0f40000000"

def x : AlgorithmLib.CudaTensor.Tensor 2 :=
  AlgorithmLib.CudaTensor.Tensor.input0

def y : AlgorithmLib.CudaTensor.Tensor 2 :=
  AlgorithmLib.CudaTensor.Tensor.input1

def result : CompileResult := (alpha.saxpy x y).compileTo 1

-- Uncomment to see elaboration-time rejection:
--
-- def badOutput : CompileResult := (alpha.saxpy x y).compileTo 3
-- -- impossible: output index 3 ∉ Fin 2

-- def badMixedArity : CompileResult :=
--   (alpha * x + (AlgorithmLib.CudaTensor.Tensor.input0 :
--     AlgorithmLib.CudaTensor.Tensor 1)).compileTo 1
-- -- type error: Tensor 2 and Tensor 1 cannot be combined

def artifacts : Array Json :=
  let r := result
  #[
    toJsonEntry "cuda_saxpy_persist_load"  r.config r.loadAlgorithm,
    toJsonEntry "cuda_saxpy_persist_prep"  r.config r.prepAlgorithm,
    toJsonEntry "cuda_saxpy_persist_infer" r.config r.inferAlgorithm,
  ]

end CudaSaxpyPersist
