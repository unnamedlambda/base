import Lake
open Lake DSL

require algorithmLib from "../lib"

package algorithms where
  srcDir := "."

-- Benchmark algorithms
lean_exe generate_csv_bench where
  root := `CsvBenchAlgorithm
  supportInterpreter := true

lean_exe generate_regex_bench where
  root := `RegexBenchAlgorithm
  supportInterpreter := true

lean_exe generate_json_bench where
  root := `JsonBenchAlgorithm
  supportInterpreter := true

lean_exe generate_strsearch_bench where
  root := `StringSearchAlgorithm
  supportInterpreter := true

lean_exe generate_wordcount_bench where
  root := `WordCountAlgorithm
  supportInterpreter := true

lean_exe generate_saxpy_bench where
  root := `SaxpyBenchAlgorithm
  supportInterpreter := true

lean_exe generate_hist1_bench where
  root := `HistogramBench1Algorithm
  supportInterpreter := true

lean_exe generate_hist4_bench where
  root := `HistogramBench4Algorithm
  supportInterpreter := true

lean_exe generate_matmul_bench where
  root := `MatmulBenchAlgorithm
  supportInterpreter := true

lean_exe generate_vecops_bench where
  root := `VecOpsBenchAlgorithm
  supportInterpreter := true

lean_exe generate_reduction_bench where
  root := `ReductionBenchAlgorithm
  supportInterpreter := true

lean_exe generate_gpu_vecadd_bench where
  root := `GpuVecAddBenchAlgorithm
  supportInterpreter := true

lean_exe generate_gpu_matmul_bench where
  root := `GpuMatMulBenchAlgorithm
  supportInterpreter := true

lean_exe generate_gpu_reduction_bench where
  root := `GpuReductionBenchAlgorithm
  supportInterpreter := true

lean_exe generate_cuda_saxpy_bench where
  root := `CudaSaxpyBenchAlgorithm
  supportInterpreter := true

lean_exe generate_gpu_iter_bench where
  root := `GpuIterBenchAlgorithm
  supportInterpreter := true

lean_exe generate_sort_bench where
  root := `SortBenchAlgorithm
  supportInterpreter := true

-- Application algorithms
lean_exe generate_cli where
  root := `CliAlgorithm
  supportInterpreter := true

lean_exe generate_compress where
  root := `CompressAlgorithm
  supportInterpreter := true

lean_exe generate_csv where
  root := `CsvAlgorithm
  supportInterpreter := true

lean_exe generate_draw where
  root := `DrawAlgorithm
  supportInterpreter := true

lean_exe generate_fft where
  root := `FftAlgorithm
  supportInterpreter := true

lean_exe generate_lean_eval where
  root := `LeanEvalAlgorithm
  supportInterpreter := true

lean_exe generate_matmul where
  root := `MatmulAlgorithm
  supportInterpreter := true

lean_exe generate_raytrace where
  root := `RaytraceAlgorithm
  supportInterpreter := true

lean_exe generate_sat where
  root := `SatAlgorithm
  supportInterpreter := true

lean_exe generate_scene where
  root := `SceneAlgorithm
  supportInterpreter := true

lean_exe generate_sha256 where
  root := `Sha256Algorithm
  supportInterpreter := true
