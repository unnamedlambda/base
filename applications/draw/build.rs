use build_support::{generate_algorithms, rerun_if_changed, AlgorithmArtifact};

fn main() {
    rerun_if_changed(&[
        "../../lean/algorithms/DrawAlgorithm.lean",
        "../../lean/lib/AlgorithmLib.lean",
    ]);

    generate_algorithms("../../lean/algorithms", &[AlgorithmArtifact {
        lean_file: "DrawAlgorithm.lean",
        output_name: "algorithm",
    }]);
}
