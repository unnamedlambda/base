use build_support::{generate_algorithms, rerun_if_changed, AlgorithmArtifact};

fn main() {
    let artifacts = [
        AlgorithmArtifact {
            lean_file: "CsvBenchAlgorithm.lean",
            output_name: "csv_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "RegexBenchAlgorithm.lean",
            output_name: "regex_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "JsonBenchAlgorithm.lean",
            output_name: "json_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "StringSearchAlgorithm.lean",
            output_name: "strsearch_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "WordCountAlgorithm.lean",
            output_name: "wc_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "SaxpyBenchAlgorithm.lean",
            output_name: "saxpy_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "HistogramBench1Algorithm.lean",
            output_name: "hist1_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "HistogramBench4Algorithm.lean",
            output_name: "hist4_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "MatmulBenchAlgorithm.lean",
            output_name: "matmul_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "VecOpsBenchAlgorithm.lean",
            output_name: "vecops_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "ReductionBenchAlgorithm.lean",
            output_name: "reduction_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "GpuVecAddBenchAlgorithm.lean",
            output_name: "gpu_vecadd_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "GpuMatMulBenchAlgorithm.lean",
            output_name: "gpu_matmul_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "GpuReductionBenchAlgorithm.lean",
            output_name: "gpu_reduction_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "CudaSaxpyBenchAlgorithm.lean",
            output_name: "cuda_saxpy_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "GpuIterBenchAlgorithm.lean",
            output_name: "gpu_iter_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "SortBenchAlgorithm.lean",
            output_name: "sort_algorithm",
        },
    ];

    let rerun_paths: Vec<_> = artifacts
        .iter()
        .map(|artifact| format!("../lean/algorithms/{}", artifact.lean_file))
        .chain(["../lean/lib/AlgorithmLib.lean".to_string()])
        .collect();
    let rerun_paths_refs: Vec<&str> = rerun_paths.iter().map(|s| s.as_str()).collect();

    rerun_if_changed(&rerun_paths_refs);

    generate_algorithms("../lean/algorithms", &artifacts);
}
