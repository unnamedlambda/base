use build_support::{generate_algorithms, rerun_if_changed, AlgorithmArtifact};

fn main() {
    let artifacts = [
        AlgorithmArtifact {
            lean_file: "lean/CsvBenchAlgorithm.lean",
            output_name: "csv_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/RegexBenchAlgorithm.lean",
            output_name: "regex_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/JsonBenchAlgorithm.lean",
            output_name: "json_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/StringSearchAlgorithm.lean",
            output_name: "strsearch_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/WordCountAlgorithm.lean",
            output_name: "wc_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/SaxpyBenchAlgorithm.lean",
            output_name: "saxpy_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/HistogramBench1Algorithm.lean",
            output_name: "hist1_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/HistogramBench4Algorithm.lean",
            output_name: "hist4_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/MatmulBenchAlgorithm.lean",
            output_name: "matmul_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/VecOpsBenchAlgorithm.lean",
            output_name: "vecops_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/ReductionBenchAlgorithm.lean",
            output_name: "reduction_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/GpuVecAddBenchAlgorithm.lean",
            output_name: "gpu_vecadd_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/GpuMatMulBenchAlgorithm.lean",
            output_name: "gpu_matmul_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/GpuReductionBenchAlgorithm.lean",
            output_name: "gpu_reduction_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/CudaSaxpyBenchAlgorithm.lean",
            output_name: "cuda_saxpy_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/GpuIterBenchAlgorithm.lean",
            output_name: "gpu_iter_algorithm",
        },
        AlgorithmArtifact {
            lean_file: "lean/SortBenchAlgorithm.lean",
            output_name: "sort_algorithm",
        },
    ];

    let rerun_paths: Vec<_> = artifacts
        .iter()
        .map(|artifact| artifact.lean_file)
        .chain(["lakefile.lean", "../lean/AlgorithmLib.lean"])
        .collect();

    rerun_if_changed(&rerun_paths);

    generate_algorithms(&artifacts);
}
