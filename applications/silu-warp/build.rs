use std::path::Path;
fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lean_dir = manifest.join("../../lean");
    build_support::build_all(&[lean_dir.join("algorithms/SiluWarpAlgorithm.lean")], &lean_dir);
}
