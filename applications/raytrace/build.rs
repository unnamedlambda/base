use build_support::{generate_algorithms, rerun_if_changed, AlgorithmArtifact};

fn main() {
    rerun_if_changed(&[
        "../../algorithms/lean/RaytraceAlgorithm.lean",
        "../../algorithms/lean/AlgorithmLib.lean",
    ]);

    generate_algorithms("../../algorithms", &[AlgorithmArtifact {
        lean_file: "lean/RaytraceAlgorithm.lean",
        output_name: "algorithm",
    }]);
}
