use build_support::{generate_algorithms, rerun_if_changed, AlgorithmArtifact};

fn main() {
    rerun_if_changed(&[
        "../../lean/algorithms/MatmulAlgorithm.lean",
        "../../lean/lib/AlgorithmLib.lean",
    ]);

    generate_algorithms("../../lean/algorithms", &[AlgorithmArtifact {
        lean_file: "MatmulAlgorithm.lean",
        output_name: "algorithm",
    }]);
}
