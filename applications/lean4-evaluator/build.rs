use build_support::{generate_algorithms, rerun_if_changed, AlgorithmArtifact};

fn main() {
    rerun_if_changed(&[
        "lean/LeanEval.lean",
        "lakefile.lean",
        "../../lean/AlgorithmLib.lean",
    ]);

    generate_algorithms(&[AlgorithmArtifact {
        lean_file: "lean/LeanEval.lean",
        output_name: "algorithm",
    }]);
}
