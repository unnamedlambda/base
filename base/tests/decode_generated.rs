//! Decodes a generated artifact, if one is pointed at by BASE_ARTIFACT_JSON.
use base_types::Artifact;

#[test]
fn generated_artifact_decodes() {
    let Ok(path) = std::env::var("BASE_ARTIFACT_JSON") else { return };
    let text = std::fs::read_to_string(&path).expect("read artifact");
    let a: Artifact = serde_json::from_str(&text).expect("artifact shape");
    let clif = base::clif_text(&a.setup.clif).expect("decode");
    println!("{clif}");
    assert!(clif.contains("function u0:1"));
}
