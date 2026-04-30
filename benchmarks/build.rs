fn main() {
    build_support::build_all("../lean/algorithms", &[
        ("CsvBenchAlgorithm.lean",          "csv_algorithm"),
        ("RegexBenchAlgorithm.lean",         "regex_algorithm"),
        ("JsonBenchAlgorithm.lean",          "json_algorithm"),
        ("StringSearchAlgorithm.lean",       "strsearch_algorithm"),
        ("WordCountAlgorithm.lean",          "wc_algorithm"),
        ("SaxpyBenchAlgorithm.lean",         "saxpy_algorithm"),
        ("HistogramBench1Algorithm.lean",    "hist1_algorithm"),
        ("HistogramBench4Algorithm.lean",    "hist4_algorithm"),
        ("MatmulBenchAlgorithm.lean",        "matmul_algorithm"),
        ("VecOpsBenchAlgorithm.lean",        "vecops_algorithm"),
        ("ReductionBenchAlgorithm.lean",     "reduction_algorithm"),
        ("GpuVecAddBenchAlgorithm.lean",     "gpu_vecadd_algorithm"),
        ("GpuMatMulBenchAlgorithm.lean",     "gpu_matmul_algorithm"),
        ("GpuReductionBenchAlgorithm.lean",  "gpu_reduction_algorithm"),
        ("CudaSaxpyBenchAlgorithm.lean",     "cuda_saxpy_algorithm"),
        ("GpuIterBenchAlgorithm.lean",       "gpu_iter_algorithm"),
        ("SortBenchAlgorithm.lean",          "sort_algorithm"),
    ]);
}
