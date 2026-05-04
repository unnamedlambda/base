use std::path::Path;

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    build_support::build(
        &manifest.join("../../lean/algorithms/CompressAlgorithm.lean"),
        &manifest.join("../../lean"),
    );
}
