use base::{Artifact, Base};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("WindowDemoAlgorithm/window_demo");

/// The test entry points, as this artifact's generator numbers them.
const TEST_MOVE_LEFT: u32 = 3;
const TEST_MOVE_RIGHT: u32 = 2;
const TEST_MOVE_UP_CLAMP: u32 = 4;
const TEST_QUIT_ON_CLOSE: u32 = 5;
const TEST_RENDER_PIXEL: u32 = 6;

/// Run one test extra on the given Base and return (pass, actual, expected).
fn run_scenario(base: &mut Base, fn_idx: u32) -> (i64, i64, i64) {
    // The scenario answers pass, actual and expected in its out buffer.
    let mut out = [0u8; 24];
    base.execute_into(fn_idx, &[], &mut out).expect("execute failed");
    let col = |i: usize| i64::from_le_bytes(out[i * 8..i * 8 + 8].try_into().unwrap());
    (col(0), col(1), col(2))
}

/// Pure game-logic scenarios — no GPU, no window. One shared Base (JIT compiled
/// once); each scenario resets state, so they can run back to back.
#[test]
fn logic_scenarios() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    let mut base = Base::new(artifact).expect("Base::new");

    for (name, fn_idx) in [
        ("test_move_right", TEST_MOVE_RIGHT),
        ("test_move_left", TEST_MOVE_LEFT),
        ("test_move_up_clamp", TEST_MOVE_UP_CLAMP),
        ("test_quit_on_close", TEST_QUIT_ON_CLOSE),
    ] {
        let (pass, actual, expected) = run_scenario(&mut base, fn_idx);
        assert_eq!(pass, 1, "{name}: actual={actual}, expected={expected}");
    }
}

/// Rendering correctness — runs the real WGSL kernel headlessly, downloads the
/// frame, and asserts the player pixel is player-coloured. Needs a GPU (present
/// is never called, so no window/display is required).
#[test]
fn render_pixel_scenario() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    let mut base = Base::new(artifact).expect("Base::new");

    let (pass, actual, expected) = run_scenario(&mut base, TEST_RENDER_PIXEL);
    assert_eq!(
        pass, 1,
        "render: player-pixel red channel was {actual}, expected ~{expected}"
    );
}
