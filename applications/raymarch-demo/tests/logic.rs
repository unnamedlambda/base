use base::{Artifact, Base};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("RaymarchDemoAlgorithm/raymarch_demo");

fn run_scenario(
    base: &mut Base,
    extras: &std::collections::BTreeMap<String, base::Algorithm>,
    name: &str,
) -> (i64, i64, i64) {
    let alg = extras.get(name).unwrap_or_else(|| panic!("missing extra {name}"));
    // The scenario answers pass, actual and expected in its out buffer.
    let mut out = [0u8; 24];
    base.execute_into(alg, &[], &mut out).expect("execute failed");
    let col = |i: usize| i64::from_le_bytes(out[i * 8..i * 8 + 8].try_into().unwrap());
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
