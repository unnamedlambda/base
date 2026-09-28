//! What the build embedded, checked against the one encoding the runtime reads.

use base_types::Artifact;

/// Every artifact decodes and re-encodes to the same bytes, so a Lean writer
/// that disagrees with serde on any detail of the profile is caught here rather
/// than read some other way.
#[test]
fn every_artifact_is_in_the_encoding() {
    assert!(!lean_artifacts::NAMES.is_empty(), "the build embedded no artifacts");
    for name in lean_artifacts::NAMES {
        let bytes = lean_artifacts::by_name(name).expect("every listed name resolves");
        let artifact = Artifact::from_bytes(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let again = artifact.to_bytes();
        if again.as_slice() != bytes {
            let at = bytes.iter().zip(&again).position(|(a, b)| a != b).unwrap_or(bytes.len().min(again.len()));
            panic!(
                "{name} decodes, but re-encodes differently from byte {at} ({} written, {} re-encoded)",
                bytes.len(),
                again.len()
            );
        }
    }
}

/// Nothing sits in the directory that the build did not name, so no stale file
/// can satisfy a path someone reads at run time.
#[test]
fn nothing_but_the_named_artifacts() {
    let mut present: Vec<String> = std::fs::read_dir(lean_artifacts::DIR)
        .expect("the artifact directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.is_file())
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    present.sort();
    let named: Vec<String> = lean_artifacts::NAMES.iter().map(|n| format!("{n}.cbor")).collect();
    assert_eq!(present, named);
}
