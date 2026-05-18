import Lean
import Std
import AlgorithmLib
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

open Lean
open AlgorithmLib

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir <|
    CsvBench.artifacts ++
    JsonBench.artifacts ++
    RegexBench.artifacts ++
    StringSearchBench.artifacts ++
    WordCountBench.artifacts ++
    VecOpsBench.artifacts ++
    ClampSumBench.artifacts ++
    RowDotBench.artifacts ++
    RowAffineReduceBench.artifacts ++
    PandasBench.artifacts ++
    PandasFilterBench.artifacts ++
    CudaVecAddPersist.artifacts ++
    CudaSaxpyPersist.artifacts ++
    CudaGemvPersist.artifacts ++
    CudaRmsNormPersist.artifacts ++
    CudaSoftmaxPersist.artifacts ++
    CudaDecoderLayer.artifacts ++
    CudaDecodeAttention.artifacts
