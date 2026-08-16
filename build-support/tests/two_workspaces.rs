//! Two cargo workspaces generating from one Lake package at the same time.
//!
//! Cargo's own lock serialises builds within a workspace, so two projects are
//! the only way two `generate` calls land on one package at once. Lake takes no
//! lock, so each would do the whole build rather than one building and the rest
//! replaying.
//!
//! Running both and checking they succeed proves nothing -- with the lock
//! removed it passes too, since concurrent lake is wasteful rather than wrong.
//! So this takes the lock itself and requires a build to wait for it.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::Duration;

use fs2::FileExt;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/two-workspaces")
}

fn lock_path() -> PathBuf {
    fixture().join("lean/shared/.lake/generate.lock")
}

/// The inner cargo must not inherit this one's environment: a jobserver it
/// cannot reach or an inherited target directory would make the result depend
/// on how the outer test was invoked.
fn cargo(workspace: &str) -> Command {
    let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    for (k, _) in std::env::vars() {
        if k.starts_with("CARGO") || k.starts_with("RUST") {
            cmd.env_remove(k);
        }
    }
    cmd.arg("build").current_dir(fixture().join(workspace));
    cmd
}

fn build(workspace: &str) {
    let out = cargo(workspace).output().expect("cargo should run");
    assert!(
        out.status.success(),
        "{workspace} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Make the next build run the generation rather than skip it.
fn invalidate() {
    let src = fixture().join("lean/shared/Shared/Gen.lean");
    let text = fs::read_to_string(&src).expect("fixture source");
    fs::write(&src, &text).expect("rewrite");
}

fn artifact(workspace: &str) -> String {
    let p = fixture().join(workspace).join("gen/artifacts/Shared.Gen/shared.json");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// A child killed however the test ends, so a failure cannot leave a cargo
/// running against the fixture and break the next test.
struct Running(Child);

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// One test, because both halves drive cargo against the same two workspaces
/// and the timing assertion below would be reading the wrong wait if another
/// build were in flight.
#[test]
fn generating_from_a_shared_package_serialises() {
    build("w1");
    build("w2");

    // Held by this test, so the build script must find it taken.
    let held = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path())
        .expect("the lock file should exist after a build");
    held.lock_exclusive().expect("take the lock");

    invalidate();
    let mut child = Running(cargo("w1").spawn().expect("spawn cargo"));

    // Long enough that a build which was not waiting would have finished: the
    // whole generation takes about a second once lake has nothing to do.
    std::thread::sleep(Duration::from_secs(4));
    assert!(
        child.0.try_wait().expect("poll").is_none(),
        "the build finished while the lock was held, so `generate` is not taking it"
    );

    FileExt::unlock(&held).expect("release the lock");
    let status = child.0.wait().expect("wait");
    assert!(status.success(), "the build should finish once the lock is free");
    assert!(artifact("w1").contains("\"memory_size\":777"), "and produce its artifact");

    // Both at once, end to end. Weak on its own -- see the module docs -- but
    // it is what would catch a deadlock, or two lakes leaving one of them with
    // no executable to run.
    invalidate();
    let mut running: Vec<Running> = ["w1", "w2"]
        .iter()
        .map(|w| Running(cargo(w).spawn().expect("spawn cargo")))
        .collect();
    for (w, r) in ["w1", "w2"].iter().zip(running.iter_mut()) {
        let status = r.0.wait().expect("wait");
        assert!(status.success(), "{w} failed under concurrency");
    }
    for w in ["w1", "w2"] {
        assert!(artifact(w).contains("\"memory_size\":777"), "{w} produced its artifact");
    }
}
