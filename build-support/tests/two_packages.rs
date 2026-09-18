//! That splitting the Lean bounds what an edit costs, run rather than argued.
//!
//! The fixture is two Lake packages requiring a third, an owner crate each, a
//! consumer of one and a consumer of both. The shared package is what makes two
//! build scripts run at once and reach for the same `lake`.
//!
//! Needs `lake` on the path and builds its own cargo workspace, so it is slower
//! than the rest of this crate's tests.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/two-packages")
}

/// Build the fixture and return the crates cargo compiled.
///
/// The inner cargo must not inherit this one's environment: a jobserver it
/// cannot reach or an inherited target directory would make the result depend
/// on how the outer test was invoked.
fn build() -> Vec<String> {
    let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    for (k, _) in std::env::vars() {
        if k.starts_with("CARGO") || k.starts_with("RUST") {
            cmd.env_remove(k);
        }
    }
    let out = cmd
        .arg("build")
        .current_dir(fixture())
        .output()
        .expect("cargo should run");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "fixture build failed:\n{err}");
    err.lines()
        .filter_map(|l| l.trim().strip_prefix("Compiling "))
        .filter_map(|l| l.split_whitespace().next())
        .map(str::to_string)
        .collect()
}

/// Build until cargo has nothing left to do, so that a later build reports only
/// what the edit caused.
fn settle() {
    for _ in 0..5 {
        if build().is_empty() {
            return;
        }
    }
    panic!("the fixture never reached a steady state");
}

/// An edit that is put back however the test ends.
struct Edit {
    path: PathBuf,
    original: String,
}

impl Edit {
    fn new(rel: &str, from: &str, to: &str) -> Edit {
        let path = fixture().join(rel);
        let original = std::fs::read_to_string(&path).expect("fixture source");
        assert!(original.contains(from), "{rel} does not contain {from:?}");
        std::fs::write(&path, original.replace(from, to)).expect("write");
        Edit { path, original }
    }
}

impl Drop for Edit {
    fn drop(&mut self) {
        std::fs::write(&self.path, &self.original).expect("restore");
    }
}

/// One test, because there is one fixture: cargo serialises concurrent builds
/// of a workspace on a lock, but the artifacts an edit produces are shared
/// state, so two tests editing it would see each other's Lean.
#[test]
fn splitting_the_lean_bounds_what_an_edit_costs() {
    settle();

    // Each owner crate publishes its own package's artifacts.
    let alpha = fixture().join("alpha-artifacts/artifacts/Alpha.Gen/alpha.cbor");
    let beta = fixture().join("beta-artifacts/artifacts/Beta.Gen/beta.cbor");
    let size = |p: &Path| {
        let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        base_types::Artifact::from_bytes(&bytes).expect("an artifact").required_memory
    };
    assert_eq!(size(&alpha), 111, "alpha's own artifact");
    assert_eq!(size(&beta), 222, "beta's own artifact");

    // `uses-both` depends on both owner crates, so cargo hands its build script
    // two `DEP_*_DIR` values; it calls `consume_from("beta")` because `consume`
    // would have no way to pick. That it built is the check -- the `artifact!`
    // in its source resolved against beta's directory.
    assert!(
        fixture().join("target/debug/uses-both").exists(),
        "uses-both should have been built"
    );

    // `uses-alpha` depends on alpha-artifacts alone, so beta's sources are
    // nothing to it. This is the whole scaling claim.
    {
        let _e = Edit::new("lean/beta/Beta/Gen.lean", "Nat := 222", "Nat := 322");
        let built = build();
        assert!(built.contains(&"beta-artifacts".to_string()), "beta regenerates: {built:?}");
        assert!(built.contains(&"uses-both".to_string()), "beta's consumer rebuilds: {built:?}");
        assert!(!built.contains(&"alpha-artifacts".to_string()), "alpha is untouched: {built:?}");
        assert!(!built.contains(&"uses-alpha".to_string()), "alpha's consumer is untouched: {built:?}");

        // One rebuild, not two. Watching the directory the artifacts are
        // written to would leave the build script stale the moment it finished,
        // and every Lean edit would cost this twice.
        assert!(build().is_empty(), "a second build had work left to do");
    }
    settle();

    // The other direction: both crates consume alpha, so both rebuild. Alpha
    // and beta share a Lean dependency, and this still does not reach beta --
    // the unit of invalidation is the Lake package, not the library graph.
    {
        let _e = Edit::new("lean/alpha/Alpha/Gen.lean", "Nat := 111", "Nat := 311");
        let built = build();
        for c in ["alpha-artifacts", "uses-alpha", "uses-both"] {
            assert!(built.contains(&c.to_string()), "{c} consumes alpha: {built:?}");
        }
        assert!(!built.contains(&"beta-artifacts".to_string()), "beta is untouched: {built:?}");
    }
    settle();

    // Editing what they share invalidates both at once, so both build scripts
    // run together and both call `lake` on the same package. That must not
    // deadlock, and each artifact must still come out its own package's.
    let _e = Edit::new("lean/common/Common.lean", "bump : Nat := 0", "bump : Nat := 1");
    let built = build();
    for c in ["alpha-artifacts", "beta-artifacts", "uses-alpha", "uses-both"] {
        assert!(built.contains(&c.to_string()), "{c} depends on the shared package: {built:?}");
    }
    assert_eq!(size(&alpha), 112, "alpha picked up the shared edit");
    assert_eq!(size(&beta), 223, "beta picked up the shared edit");
}
