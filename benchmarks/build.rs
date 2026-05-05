use std::path::Path;

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let alg_dir = manifest.join("../lean/algorithms");
    let watch_dir = manifest.join("../lean");

    build_support::build(&alg_dir.join("RustBenchmarks.lean"), &watch_dir);
}
