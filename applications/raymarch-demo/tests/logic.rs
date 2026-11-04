use base::Driver;

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::RAYMARCH_DEMO;


fn run_scenario(base: &Driver, entry: &str) -> (i64, i64, i64) {
    // The scenario answers pass, actual and expected in its out buffer.
    let mut out = [0u8; 24];
    unsafe { base.function(entry) }.expect("execute failed")(&[], &mut out);
    let col = |i: usize| i64::from_le_bytes(out[i * 8..i * 8 + 8].try_into().unwrap());
    (col(0), col(1), col(2))
}

#[test]
fn camera_scenarios() {
    let base = Driver::load(ARTIFACT_BINARY).expect("Driver::load");

    for name in [
        "test_move_forward",
        "test_strafe_right",
        "test_rise_clamp",
        "test_quit_on_close",
    ] {
        let (pass, actual, expected) = run_scenario(&base, name);
        assert_eq!(pass, 1, "{name}: actual={actual}, expected={expected}");
    }
}

#[test]
fn render_scene_scenario() {
    let base = Driver::load(ARTIFACT_BINARY).expect("Driver::load");

    let (pass, actual, expected) = run_scenario(&base, "test_render_scene");
    assert_eq!(
        pass, 1,
        "raymarch: ground-pixel blue {actual} vs sky-pixel blue {expected} (expected sky bluer)"
    );
}
