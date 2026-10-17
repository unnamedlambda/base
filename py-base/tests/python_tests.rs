//! The Python suite, against the extension this build made: copied beside the
//! tests under the name Python imports, so nothing installed is touched.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The extension cargo built for this profile: `target/<profile>/`, two up from
/// this test's own executable in `deps/`.
fn built_extension() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let dir = exe.parent().and_then(Path::parent).unwrap();
    dir.join(if cfg!(windows) {
        "py_base.dll"
    } else if cfg!(target_os = "macos") {
        "libpy_base.dylib"
    } else {
        "libpy_base.so"
    })
}

#[test]
fn python_tests() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let python = manifest.join(".venv/bin/python");
    assert!(
        python.exists(),
        "no venv at {}: `python3 -m venv py-base/.venv && py-base/.venv/bin/pip install pytest`",
        python.display()
    );
    let module = tempfile::TempDir::new().unwrap();
    let name = if cfg!(windows) { "py_base.pyd" } else { "py_base.so" };
    std::fs::copy(built_extension(), module.path().join(name)).expect("the extension cargo built");
    let status = Command::new(&python)
        .args(["-m", "pytest", "tests", "-q", "-W", "ignore:py_base was built without optimisations:UserWarning"])
        .env("PYTHONPATH", module.path())
        .current_dir(&manifest)
        .status()
        .expect("pytest");
    assert!(status.success(), "the Python suite failed");
}
