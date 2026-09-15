import Lean
import Std
import AlgorithmLib.Gen
import ClampSumBenchAlgorithm
import CsvBenchAlgorithm
import CudaDecodeAttentionAlgorithm
import CudaDecoderLayerAlgorithm
import CudaGemvPersistAlgorithm
import CudaRmsNormPersistAlgorithm
import CudaSaxpyPersistAlgorithm
import CudaSoftmaxPersistAlgorithm
import CudaVecAddPersistAlgorithm
import JsonBenchAlgorithm
import PandasBenchAlgorithm
import PandasFilterBenchAlgorithm
import RegexBenchAlgorithm
import RowAffineReduceBenchAlgorithm
import RowDotBenchAlgorithm
import StringSearchAlgorithm
import VecOpsBenchAlgorithm
import WordCountAlgorithm
import ShipScan

open Lean
open AlgorithmLib

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  -- Every body is checked while it is emitted, so a generator that built an
  -- ill-formed one stops here with a message rather than writing an artifact.
  let vecAdd ← Prog.orDie CudaVecAddPersist.result
  let saxpy ← Prog.orDie CudaSaxpyPersist.result
  let csv ← Prog.orDie CsvBench.clifIR
  let json ← Prog.orDie JsonBench.clifIR
  let regex ← Prog.orDie RegexBench.clifIR
  let strsearch ← Prog.orDie StringSearchBench.clifIR
  let wordcount ← Prog.orDie WordCountBench.clifIR
  let vecops ← Prog.orDie VecOpsBench.clifIR
  let clampSum ← Prog.orDie ClampSumBench.clifIR
  let rowDot ← Prog.orDie RowDotBench.clifIR
  let rowAffine ← Prog.orDie RowAffineReduceBench.clifIR
  let pandas ← Prog.orDie PandasBench.clifIR
  let pandasFilter ← Prog.orDie PandasFilterBench.clifIR
  let gemv ← Prog.orDie CudaGemvPersist.clifIR
  let rmsnorm ← Prog.orDie CudaRmsNormPersist.clifIR
  let softmax ← Prog.orDie CudaSoftmaxPersist.clifIR
  let decoder ← Prog.orDie CudaDecoderLayer.clifIR
  let decodeAttn ← Prog.orDie CudaDecodeAttention.clifIR
  emitArtifacts outDir <|
    CsvBench.artifacts csv ++
    JsonBench.artifacts json ++
    RegexBench.artifacts regex ++
    StringSearchBench.artifacts strsearch ++
    WordCountBench.artifacts wordcount ++
    VecOpsBench.artifacts vecops ++
    ClampSumBench.artifacts clampSum ++
    RowDotBench.artifacts rowDot ++
    RowAffineReduceBench.artifacts rowAffine ++
    PandasBench.artifacts pandas ++
    PandasFilterBench.artifacts pandasFilter ++
    CudaVecAddPersist.artifacts vecAdd ++
    CudaSaxpyPersist.artifacts saxpy ++
    CudaGemvPersist.artifacts gemv ++
    CudaRmsNormPersist.artifacts rmsnorm ++
    CudaSoftmaxPersist.artifacts softmax ++
    CudaDecoderLayer.artifacts decoder ++
    CudaDecodeAttention.artifacts decodeAttn

#eval ShipScan.check "PythonBenchmarks"
