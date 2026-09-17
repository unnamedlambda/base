use base::{Artifact, Base};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("FallingSandAlgorithm/falling_sand");


fn run_scenario(base: &mut Base, entry: &str) -> (i64, i64, i64) {
    // The scenario answers pass, actual and expected in its out buffer.
    let mut out = [0u8; 24];
    base.execute(entry, &[], &mut out).expect("execute failed");
    let col = |i: usize| i64::from_le_bytes(out[i * 8..i * 8 + 8].try_into().unwrap());
    (col(0), col(1), col(2))
}

#[test]
fn sand_simulation() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    let mut base = Base::new(artifact).expect("Base::new");
    for name in [
        "test_grain_falls",
        "test_conservation",
    ] {
        let (pass, actual, expected) = run_scenario(&mut base, name);
        assert_eq!(pass, 1, "{name}: actual={actual}, expected={expected}");
    }
}
