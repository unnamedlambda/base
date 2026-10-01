import Lean
import Std
import AlgorithmLib.Gen
import Bench.CudaDecodeAttention
import Bench.CudaGemvPersist
import Bench.CudaRmsNormPersist
import Bench.CudaSaxpyPersist
import Bench.CudaSoftmaxPersist
import Bench.CudaVecAddPersist
import Scan.Ship

open Lean
open AlgorithmLib

/-- The device programs only the Python benchmarks run. -/
def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  -- Every body is checked while it is emitted, so a generator that built an
  -- ill-formed one stops here with a message rather than writing an artifact.
  let vecAdd ← Prog.orDie CudaVecAddPersist.result
  let saxpy ← Prog.orDie CudaSaxpyPersist.result
  let gemv ← Prog.orDie CudaGemvPersist.clifIR
  let rmsnorm ← Prog.orDie CudaRmsNormPersist.clifIR
  let softmax ← Prog.orDie CudaSoftmaxPersist.clifIR
  let decodeAttn ← Prog.orDie CudaDecodeAttention.clifIR
  emitArtifacts outDir <|
    CudaVecAddPersist.artifacts vecAdd ++
    CudaSaxpyPersist.artifacts saxpy ++
    CudaGemvPersist.artifacts gemv ++
    CudaRmsNormPersist.artifacts rmsnorm ++
    CudaSoftmaxPersist.artifacts softmax ++
    CudaDecodeAttention.artifacts decodeAttn

#eval ShipScan.check "Bench.Python"
