use arrow_array::Int64Array;
use base::{Artifact, Base};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/RaymarchDemoAlgorithm/raymarch_demo.bin"));

fn run_scenario(
    base: &mut Base,
    extras: &std::collections::HashMap<String, base::Algorithm>,
    name: &str,
) -> (i64, i64, i64) {
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

#[test]
fn camera_scenarios() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    let extras = artifact.extras.clone();
    let mut base = Base::new(artifact.setup).expect("Base::new");

    for name in [
        "test_move_forward",
        "test_strafe_right",
        "test_rise_clamp",
        "test_quit_on_close",
    ] {
        let (pass, actual, expected) = run_scenario(&mut base, &extras, name);
        assert_eq!(pass, 1, "{name}: actual={actual}, expected={expected}");
    }
}

#[test]
fn render_scene_scenario() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    let extras = artifact.extras.clone();
    let mut base = Base::new(artifact.setup).expect("Base::new");

    let (pass, actual, expected) = run_scenario(&mut base, &extras, "test_render_scene");
    assert_eq!(
        pass, 1,
        "raymarch: ground-pixel blue {actual} vs sky-pixel blue {expected} (expected sky bluer)"
    );
}
