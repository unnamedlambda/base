use base::Driver;

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::FALLING_SAND;


fn run_scenario(base: &Driver, entry: &str) -> (i64, i64, i64) {
    // The scenario answers pass, actual and expected in its out buffer.
    let mut out = [0u8; 24];
    unsafe { base.function(entry) }.expect("execute failed")(&[], &mut out);
    let col = |i: usize| i64::from_le_bytes(out[i * 8..i * 8 + 8].try_into().unwrap());
    (col(0), col(1), col(2))
}

#[test]
fn sand_simulation() {
    let base = Driver::load(ARTIFACT_BINARY).expect("Driver::load");
    for name in [
        "test_grain_falls",
        "test_conservation",
    ] {
        let (pass, actual, expected) = run_scenario(&base, name);
        assert_eq!(pass, 1, "{name}: actual={actual}, expected={expected}");
    }
}
