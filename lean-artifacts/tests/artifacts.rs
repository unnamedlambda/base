//! Checks the generated tree rather than the code that generates it: every
//! artifact the Lean side emitted is one base-types can represent, and the
//! bincode beside it is that artifact and not a leftover.

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

/// The JSON directly in `dir`, not descending: a generator's side data goes in
/// a subdirectory, and only what is at the top level is an artifact.
fn jsons(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .expect("module directory should be readable")
        .map(|e| e.expect("readable entry").path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    paths.sort();
    paths
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
            !jsons(&dir).is_empty(),
            "{} has no generated JSON",
            dir.display()
        );
    }
}

#[test]
fn each_binary_is_its_json_serialized() {
    let mut checked = 0;
    for dir in modules() {
        for json in jsons(&dir) {
            let text = fs::read_to_string(&json).expect("readable json");
            // A top-level JSON that does not parse gets no binary, so an
            // application asking for it fails at compile time on a missing
            // path. The build script only warns about that, which is easy to
            // miss in a long build; this is where it fails.
            let artifact: Artifact = serde_json::from_str(&text).unwrap_or_else(|e| {
                panic!(
                    "{} is not an Artifact: {e}\n  a generator's non-artifact \
                     output belongs in a subdirectory",
                    json.display()
                )
            });
            let bin = json.with_extension("bin");
            let bytes =
                fs::read(&bin).unwrap_or_else(|e| panic!("{} should exist: {e}", bin.display()));
            assert_eq!(
                bytes,
                bincode::serialize(&artifact).expect("artifact serializes"),
                "{} does not match {}",
                bin.display(),
                json.display()
            );
            checked += 1;
        }
    }
    assert!(checked >= 60, "expected many artifacts, checked {checked}");
}

#[test]
fn no_orphan_binaries() {
    for dir in modules() {
        for entry in fs::read_dir(&dir).expect("readable module directory") {
            let path = entry.expect("readable entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            assert!(
                path.with_extension("json").exists(),
                "{} has no JSON, so no generator emits it",
                path.display()
            );
        }
    }
}
