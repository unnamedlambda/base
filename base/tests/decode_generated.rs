//! Decodes a generated artifact, if one is pointed at by BASE_ARTIFACT.
use base_types::Artifact;

#[test]
fn generated_artifact_decodes() {
    let Ok(path) = std::env::var("BASE_ARTIFACT") else { return };
    let bytes = std::fs::read(&path).expect("read artifact");
    let a = Artifact::from_bytes(&bytes).expect("artifact shape");
    let clif = base::clif_text(&a.functions).expect("decode");
    println!("{clif}");
    assert!(clif.contains("function u0:1"));
}
