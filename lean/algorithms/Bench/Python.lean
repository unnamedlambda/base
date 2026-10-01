module
public import Lean
public import Std
public import AlgorithmLib.Gen
meta import AlgorithmLib.Gen
public import Bench.CudaDecodeAttention
meta import Bench.CudaDecodeAttention
public import Bench.CudaGemvPersist
meta import Bench.CudaGemvPersist
public import Bench.CudaRmsNormPersist
meta import Bench.CudaRmsNormPersist
public import Bench.CudaSaxpyPersist
meta import Bench.CudaSaxpyPersist
public import Bench.CudaSoftmaxPersist
meta import Bench.CudaSoftmaxPersist
public import Bench.CudaVecAddPersist
meta import Bench.CudaVecAddPersist
public import Scan.Ship
meta import Scan.Ship
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib

/-- The device programs only the Python benchmarks run. -/
def Bench.Python.main (args : List String) : IO Unit := do
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

#eval ShipScan.check "Bench.Python" `Bench.Python.main