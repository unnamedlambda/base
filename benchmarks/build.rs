use std::path::{Path, PathBuf};

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let alg_dir = manifest.join("../lean/algorithms");
    let watch_dir = manifest.join("../lean");

    let files: Vec<PathBuf> = vec![
        alg_dir.join("CsvBenchAlgorithm.lean"),
        alg_dir.join("RegexBenchAlgorithm.lean"),
        alg_dir.join("JsonBenchAlgorithm.lean"),
        alg_dir.join("StringSearchAlgorithm.lean"),
        alg_dir.join("WordCountAlgorithm.lean"),
        alg_dir.join("SaxpyBenchAlgorithm.lean"),
        alg_dir.join("HistogramBench1Algorithm.lean"),
        alg_dir.join("HistogramBench4Algorithm.lean"),
        alg_dir.join("MatmulBenchAlgorithm.lean"),
        alg_dir.join("VecOpsBenchAlgorithm.lean"),
        alg_dir.join("ReductionBenchAlgorithm.lean"),
        alg_dir.join("GpuVecAddBenchAlgorithm.lean"),
        alg_dir.join("GpuMatMulBenchAlgorithm.lean"),
        alg_dir.join("GpuReductionBenchAlgorithm.lean"),
        alg_dir.join("CudaSaxpyBenchAlgorithm.lean"),
        alg_dir.join("GpuIterBenchAlgorithm.lean"),
        alg_dir.join("SortBenchAlgorithm.lean"),
    ];

    build_support::build_all(&files, &watch_dir);
}
