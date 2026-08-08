use std::path::Path;
fn main() {
    let m = Path::new(env!("CARGO_MANIFEST_DIR"));
    let l = m.join("../../lean");
    build_support::build_all(&[l.join("algorithms/MlpCifarAlgorithm.lean")], &l);
}
