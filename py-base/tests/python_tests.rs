//! Runs the py-base Python test suite as a `cargo test`-visible integration test.
//!
//! Behavior: rebuilds the extension via `maturin develop` against the active
//! venv, then invokes `pytest`.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn python_tests() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let venv_python = manifest_dir.join(".venv/bin/python");

    assert!(
        venv_python.exists(),
        "py-base venv not found at {}. \
         Set it up with `python3 -m venv py-base/.venv && \
         source py-base/.venv/bin/activate && pip install maturin pytest pyarrow`.",
        venv_python.display(),
    );

    // Rebuild the extension into the venv so tests run against the current source.
    let build = Command::new(&venv_python)
        .args(["-m", "maturin", "develop", "-q"])
        .current_dir(&manifest_dir)
        .status()
        .expect("failed to spawn `maturin develop`");
    assert!(build.success(), "`maturin develop` failed");

    let status = Command::new(&venv_python)
        .args(["-m", "pytest", "tests", "-q"])
        .current_dir(&manifest_dir)
        .status()
        .expect("failed to spawn pytest");
    assert!(status.success(), "py-base pytest suite failed");
}
