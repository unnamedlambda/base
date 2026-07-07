use arrow_array::Int64Array;
use base::{Artifact, Base};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/WindowDemoAlgorithm/window_demo.bin"));

/// Run one test extra on the given Base and return (pass, actual, expected).
fn run_scenario(base: &mut Base, extras: &std::collections::HashMap<String, base::Algorithm>, name: &str) -> (i64, i64, i64) {
    let alg = extras.get(name).unwrap_or_else(|| panic!("missing extra {name}"));
    let batches = base.execute(alg, &[]).expect("execute failed");
    let batch = &batches[0];
    let col = |i: usize| {
        batch
            .column(i)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("i64 column")
            .value(0)
    };
    (col(0), col(1), col(2))
}

/// Pure game-logic scenarios — no GPU, no window. One shared Base (JIT compiled
/// once); each scenario resets state, so they can run back to back.
#[test]
fn logic_scenarios() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    let extras = artifact.extras.clone();
    let mut base = Base::new(artifact.setup).expect("Base::new");

    for name in [
        "test_move_right",
        "test_move_left",
        "test_move_up_clamp",
        "test_quit_on_close",
    ] {
        let (pass, actual, expected) = run_scenario(&mut base, &extras, name);
        assert_eq!(pass, 1, "{name}: actual={actual}, expected={expected}");
    }
}

/// Rendering correctness — runs the real WGSL kernel headlessly, downloads the
/// frame, and asserts the player pixel is player-coloured. Needs a GPU (present
/// is never called, so no window/display is required).
#[test]
fn render_pixel_scenario() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    let extras = artifact.extras.clone();
    let mut base = Base::new(artifact.setup).expect("Base::new");

    let (pass, actual, expected) = run_scenario(&mut base, &extras, "test_render_pixel");
    assert_eq!(
        pass, 1,
        "render: player-pixel red channel was {actual}, expected ~{expected}"
    );
}
