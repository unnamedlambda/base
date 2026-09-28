//! Embed the artifacts a Lake package generates, from the build script of the
//! crate that sits in that package's directory:
//!
//! ```no_run
//! // build.rs, beside lakefile.lean
//! fn main() {
//!     base_build::lean();
//! }
//! ```
//!
//! ```ignore
//! // src/lib.rs
//! include!(concat!(env!("OUT_DIR"), "/artifacts.rs"));
//! ```
//!
//! The crate then has a `pub static` per artifact, named after it in capitals,
//! and whatever depends on it needs no build script of its own:
//!
//! ```ignore
//! let artifact = Artifact::from_bytes(lean_artifacts::BYTE_SCRUB)?;
//! ```
//!
//! Lake does the work. `lake query algorithmLib/artifacts` builds and runs the
//! package's generators, rerunning only those whose executable changed, and
//! names every file the artifacts depend on; those files are what cargo
//! watches. Nothing here decides staleness.
//!
//! Set `BASE_ARTIFACTS_DIR` to a directory of prebuilt artifacts to embed those
//! instead, without running Lake. A relative path is taken from the crate's
//! directory, where cargo runs its build script.

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The Lake target that builds a package's artifacts, defined by the library
/// every artifact-generating package requires.
const TARGET: &str = "algorithmLib/artifacts";

/// Prebuilt artifacts to embed in place of running Lake.
const PREBUILT: &str = "BASE_ARTIFACTS_DIR";

/// What Lake says when another process is re-reading the same package's
/// configuration.
const CONFIG_LOCK_BUSY: &str = "could not acquire an exclusive configuration lock";

/// Build the artifacts of the Lake package in this crate's directory and write
/// `$OUT_DIR/artifacts.rs`, which declares one `pub static` per artifact plus
/// `NAMES`, `by_name` and `DIR`.
pub fn lean() {
    println!("cargo:rerun-if-env-changed={PREBUILT}");
    let (dir, names) = match env::var_os(PREBUILT) {
        Some(dir) => prebuilt(Path::new(&dir)),
        None => from_lake(&PathBuf::from(
            env::var("CARGO_MANIFEST_DIR").expect("lean() is for use from a build script"),
        )),
    };
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    fs::write(out.join("artifacts.rs"), module(&dir, &names))
        .expect("Failed to write artifacts.rs");
}

/// Ask Lake to bring the artifacts up to date, and watch what it says they
/// depend on.
fn from_lake(package: &Path) -> (String, Vec<String>) {
    let query = || {
        Command::new("lake")
            .args(["query", TARGET])
            .current_dir(package)
            .output()
            .unwrap_or_else(|e| {
                panic!("Failed to run `lake`: {e}. Install elan, or set {PREBUILT} to prebuilt artifacts.")
            })
    };
    // Lake locks a package while it re-reads a changed lakefile, and a second
    // process refuses rather than waits. Two owner crates sharing a package
    // meet this whenever its lakefile changes, so wait for the other.
    let mut output = query();
    for _ in 0..600 {
        if output.status.success()
            || !String::from_utf8_lossy(&output.stderr).contains(CONFIG_LOCK_BUSY)
        {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
        output = query();
    }
    if !output.status.success() {
        panic!(
            "`lake query {TARGET}` failed in {}:\n{}{}",
            package.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let (mut dir, mut names) = (None, Vec::new());
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        match line.split_once(' ') {
            Some(("dir", d)) => dir = Some(d.to_string()),
            Some(("artifact", n)) => names.push(n.to_string()),
            Some(("input", f)) => println!("cargo:rerun-if-changed={f}"),
            _ => {}
        }
    }
    let dir = dir.unwrap_or_else(|| panic!("`lake query {TARGET}` named no directory"));
    (dir, names)
}

/// Every artifact in `dir`, which is watched so a new one is picked up.
fn prebuilt(dir: &Path) -> (String, Vec<String>) {
    let dir = dir
        .canonicalize()
        .unwrap_or_else(|e| panic!("{PREBUILT}={}: {e}", dir.display()));
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut names: Vec<String> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{PREBUILT}={}: {e}", dir.display()))
        .filter_map(|e| {
            let path = e.ok()?.path();
            (path.extension()? == "cbor").then(|| path.file_stem()?.to_str().map(str::to_string))?
        })
        .collect();
    names.sort();
    (dir.display().to_string(), names)
}

/// The source of the generated module.
///
/// A `static` rather than a `const` or a function: a `const` would carry its
/// bytes in the metadata every dependent loads, several times over, and a
/// function could not initialise a dependent's own `const`.
fn module(dir: &str, names: &[String]) -> String {
    let mut src = String::new();
    writeln!(src, "/// Where the artifacts are, and an artifact's side data: `DIR/<name>/`.").unwrap();
    writeln!(src, "pub const DIR: &str = {dir:?};").unwrap();
    for name in names {
        check_name(name);
        writeln!(src, "/// The `{name}` artifact.").unwrap();
        writeln!(
            src,
            "pub static {}: &[u8] = include_bytes!({:?});",
            name.to_uppercase(),
            format!("{dir}/{name}.cbor")
        )
        .unwrap();
    }
    writeln!(src, "/// Every artifact's name, sorted.").unwrap();
    writeln!(src, "pub const NAMES: &[&str] = &{names:?};").unwrap();
    writeln!(src, "/// An artifact by the name it was generated under.").unwrap();
    writeln!(src, "pub fn by_name(name: &str) -> Option<&'static [u8]> {{").unwrap();
    writeln!(src, "    match name {{").unwrap();
    for name in names {
        writeln!(src, "        {name:?} => Some({}),", name.to_uppercase()).unwrap();
    }
    writeln!(src, "        _ => None,\n    }}\n}}").unwrap();
    src
}

/// An artifact's name becomes a Rust identifier, so it has to be one, and not
/// one the generated module already declares.
fn check_name(name: &str) {
    let ok = name.starts_with(|c: char| c.is_ascii_lowercase())
        && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    assert!(
        ok,
        "artifact `{name}` is not a Rust name: start with a lowercase letter, then \
         lowercase letters, digits and `_`"
    );
    assert!(
        !matches!(name, "dir" | "names"),
        "artifact `{name}` would collide with the generated `{}`",
        name.to_uppercase()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_a_static_and_a_lookup_per_artifact() {
        let src = module("/a", &["byte_scrub".into(), "csv_app".into()]);
        assert!(src.contains(r#"pub static BYTE_SCRUB: &[u8] = include_bytes!("/a/byte_scrub.cbor");"#));
        assert!(src.contains(r#""csv_app" => Some(CSV_APP),"#));
        assert!(src.contains(r#"pub const NAMES: &[&str] = &["byte_scrub", "csv_app"];"#));
    }

    #[test]
    #[should_panic(expected = "not a Rust name")]
    fn refuses_a_name_that_is_not_an_identifier() {
        module("/a", &["byte-scrub".into()]);
    }

    #[test]
    #[should_panic(expected = "not a Rust name")]
    fn refuses_an_underscore_alone() {
        module("/a", &["_".into()]);
    }

    #[test]
    #[should_panic(expected = "collide")]
    fn refuses_a_name_the_module_declares() {
        module("/a", &["names".into()]);
    }
}
