import Lean
import Std
import AlgorithmLib.Gen
import ClampSumBenchAlgorithm
import PlainSumBenchAlgorithm
import BranchyBenchAlgorithm
import SelectBenchAlgorithm
import SelectLeaBenchAlgorithm
import SelectRotBenchAlgorithm
import SelectMaskBenchAlgorithm
import StoreBenchAlgorithm
import PminSumBenchAlgorithm
import RegPressureBenchAlgorithm
import IntSumBenchAlgorithm
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
import ShipScan

open Lean
open AlgorithmLib

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  -- Every body is checked while it is emitted, so a generator that built an
  -- ill-formed one stops here with a message rather than writing an artifact.
  let csv ← Prog.orDie CsvBench.clifIR
  let regex ← Prog.orDie RegexBench.clifIR
  let json ← Prog.orDie JsonBench.clifIR
  let strsearch ← Prog.orDie StringSearchBench.clifIR
  let wordcount ← Prog.orDie WordCountBench.clifIR
  let saxpy ← Prog.orDie Algorithm.clifIrSource
  let hist1 ← Prog.orDie HistogramBench1.clifIR
  let hist4 ← Prog.orDie HistogramBench4.clifIR
  let matmul ← Prog.orDie MatmulBench.clifIR
  let vecops ← Prog.orDie VecOpsBench.clifIR
  let reduction ← Prog.orDie ReductionBench.clifIR
  let gpuVecAdd ← Prog.orDie GpuVecAddBench.clifIR
  let gpuMatMul ← Prog.orDie GpuMatMulBench.clifIR
  let gpuReduction ← Prog.orDie GpuReductionBench.clifIR
  let cudaSaxpy ← Prog.orDie CudaSaxpyBench.clifIR
  let gpuIter ← Prog.orDie GpuIterBench.clifIR
  let sort ← Prog.orDie SortBench.clifIR
  let clampSum ← Prog.orDie ClampSumBench.clifIR
  let plainSum ← Prog.orDie PlainSumBench.clifIR
  let branchy ← Prog.orDie BranchyBench.clifIR
  let select ← Prog.orDie SelectBench.clifIR
  let selectLea ← Prog.orDie SelectLeaBench.clifIR
  let selectRot ← Prog.orDie SelectRotBench.clifIR
  let selectMask ← Prog.orDie SelectMaskBench.clifIR
  let store ← Prog.orDie StoreBench.clifIR
  let pminSum ← Prog.orDie PminSumBench.clifIR
  let regPressure ← Prog.orDie RegPressureBench.clifIR
  let intSum ← Prog.orDie IntSumBench.clifIR
  emitArtifacts outDir <|
    CsvBench.artifacts csv ++
    RegexBench.artifacts regex ++
    JsonBench.artifacts json ++
    StringSearchBench.artifacts strsearch ++
    WordCountBench.artifacts wordcount ++
    Algorithm.artifacts saxpy ++
    HistogramBench1.artifacts hist1 ++
    HistogramBench4.artifacts hist4 ++
    MatmulBench.artifacts matmul ++
    VecOpsBench.artifacts vecops ++
    ReductionBench.artifacts reduction ++
    GpuVecAddBench.artifacts gpuVecAdd ++
    GpuMatMulBench.artifacts gpuMatMul ++
    GpuReductionBench.artifacts gpuReduction ++
    CudaSaxpyBench.artifacts cudaSaxpy ++
    GpuIterBench.artifacts gpuIter ++
    SortBench.artifacts sort ++
    ClampSumBench.artifacts clampSum ++
    PlainSumBench.artifacts plainSum ++
    BranchyBench.artifacts branchy ++
    SelectBench.artifacts select ++
    SelectLeaBench.artifacts selectLea ++
    SelectRotBench.artifacts selectRot ++
    SelectMaskBench.artifacts selectMask ++
    StoreBench.artifacts store ++
    PminSumBench.artifacts pminSum ++
    RegPressureBench.artifacts regPressure ++
    IntSumBench.artifacts intSum

#eval ShipScan.check "RustBenchmarks"
