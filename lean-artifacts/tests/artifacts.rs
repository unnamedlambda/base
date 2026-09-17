//! Checks the generated tree rather than the code that generates it: every
//! artifact the Lean side emitted is one base-types can represent, in the one
//! encoding it writes, and nothing else sits beside them.

use std::fs;
use std::path::{Path, PathBuf};

use base_types::Artifact;

fn modules() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(lean_artifacts::DIR)
        .expect("artifact directory should exist after the build script runs")
        .map(|e| e.expect("readable entry").path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

/// The files directly in `dir`, not descending: a generator's side data goes in
/// a subdirectory.
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .expect("module directory should be readable")
        .map(|e| e.expect("readable entry").path())
        .filter(|p| p.is_file())
        .collect();
    paths.sort();
    paths
}

fn is_artifact(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()) == Some("cbor")
}

#[test]
fn every_generator_produced_something() {
    let modules = modules();
    assert!(
        modules.len() >= 25,
        "expected a directory per generator, found {}",
        modules.len()
    );
    for dir in modules {
        assert!(
            files(&dir).iter().any(|p| is_artifact(p)),
            "{} has no artifacts",
            dir.display()
        );
    }
}

/// Decoding and encoding again gives the file back, which is what makes the
/// file the artifact rather than one of several spellings of it.
#[test]
fn each_artifact_is_in_the_one_encoding() {
    let mut checked = 0;
    for dir in modules() {
        for path in files(&dir).into_iter().filter(|p| is_artifact(p)) {
            let bytes = fs::read(&path).expect("readable artifact");
            let artifact = Artifact::from_bytes(&bytes)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert!(artifact.to_bytes() == bytes, "{} re-encodes differently", path.display());
            checked += 1;
        }
    }
    assert!(checked >= 60, "expected many artifacts, checked {checked}");
}

/// The generator's list is the authority on what exists, and a file it does not
/// name would still satisfy an `include_bytes!`.
#[test]
fn nothing_but_the_listed_artifacts() {
    for dir in modules() {
        let list = fs::read_to_string(dir.join("generated.list")).expect("a generated.list");
        let mut listed: Vec<String> = list.lines().map(|l| format!("{l}.cbor")).collect();
        listed.push("generated.list".to_string());
        listed.sort();
        let mut present: Vec<String> = files(&dir)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        present.sort();
        assert_eq!(present, listed, "{}", dir.display());
    }
}
