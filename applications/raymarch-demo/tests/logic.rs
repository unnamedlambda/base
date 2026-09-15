use base::{Artifact, Base};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("RaymarchDemoAlgorithm/raymarch_demo");

/// The test entry points, as this artifact's generator numbers them.
const TEST_MOVE_FORWARD: u32 = 2;
const TEST_QUIT_ON_CLOSE: u32 = 5;
const TEST_RENDER_SCENE: u32 = 6;
const TEST_RISE_CLAMP: u32 = 4;
const TEST_STRAFE_RIGHT: u32 = 3;

fn run_scenario(base: &mut Base, fn_idx: u32) -> (i64, i64, i64) {
    // The scenario answers pass, actual and expected in its out buffer.
    let mut out = [0u8; 24];
    base.execute_into(fn_idx, &[], &mut out).expect("execute failed");
    let col = |i: usize| i64::from_le_bytes(out[i * 8..i * 8 + 8].try_into().unwrap());
    (col(0), col(1), col(2))
}

#[test]
fn camera_scenarios() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    let mut base = Base::new(artifact).expect("Base::new");

    for (name, fn_idx) in [
        ("test_move_forward", TEST_MOVE_FORWARD),
        ("test_strafe_right", TEST_STRAFE_RIGHT),
        ("test_rise_clamp", TEST_RISE_CLAMP),
        ("test_quit_on_close", TEST_QUIT_ON_CLOSE),
    ] {
        let (pass, actual, expected) = run_scenario(&mut base, fn_idx);
        assert_eq!(pass, 1, "{name}: actual={actual}, expected={expected}");
    }
}

#[test]
fn render_scene_scenario() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    let mut base = Base::new(artifact).expect("Base::new");

    let (pass, actual, expected) = run_scenario(&mut base, TEST_RENDER_SCENE);
    assert_eq!(
        pass, 1,
        "raymarch: ground-pixel blue {actual} vs sky-pixel blue {expected} (expected sky bluer)"
    );
}
