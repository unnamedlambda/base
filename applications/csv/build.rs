use build_support::{generate_algorithms, rerun_if_changed, AlgorithmArtifact};

fn main() {
    rerun_if_changed(&[
        "../../algorithms/lean/CsvAlgorithm.lean",
        "../../algorithms/lean/AlgorithmLib.lean",
    ]);

    generate_algorithms("../../algorithms", &[AlgorithmArtifact {
        lean_file: "lean/CsvAlgorithm.lean",
        output_name: "algorithm",
    }]);
}
