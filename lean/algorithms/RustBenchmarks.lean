import Lean
import Std
import AlgorithmLib
import CsvBenchAlgorithm
import CudaSaxpyBenchAlgorithm
import GpuIterBenchAlgorithm
import GpuMatMulBenchAlgorithm
import GpuReductionBenchAlgorithm
import GpuVecAddBenchAlgorithm
import HistogramBench1Algorithm
import HistogramBench4Algorithm
import JsonBenchAlgorithm
import MatmulBenchAlgorithm
import ReductionBenchAlgorithm
import RegexBenchAlgorithm
import SaxpyBenchAlgorithm
import SortBenchAlgorithm
import StringSearchAlgorithm
import VecOpsBenchAlgorithm
import WordCountAlgorithm

open Lean
open AlgorithmLib

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir <|
    CsvBench.artifacts ++
    RegexBench.artifacts ++
    JsonBench.artifacts ++
    StringSearchBench.artifacts ++
    WordCountBench.artifacts ++
    Algorithm.artifacts ++
    HistogramBench1.artifacts ++
    HistogramBench4.artifacts ++
    MatmulBench.artifacts ++
    VecOpsBench.artifacts ++
    ReductionBench.artifacts ++
    GpuVecAddBench.artifacts ++
    GpuMatMulBench.artifacts ++
    GpuReductionBench.artifacts ++
    CudaSaxpyBench.artifacts ++
    GpuIterBench.artifacts ++
    SortBench.artifacts
