//! Generating artifacts from a Lean package, for use from a `build.rs`.
//!
//! ```no_run
//! fn main() {
//!     build_support::generate("../lean/algorithms");
//! }
//! ```
//!
//! Lake decides what is stale; `rerun-if-changed` only decides when cargo asks
//! it. The package must declare two scripts, which is how this asks lake what
//! it knows rather than reading the lakefile as text:
//!
//! ```lean
//! script generators do
//!   for exe in (← getRootPackage).leanExes do
//!     IO.println s!"{exe.name} {exe.config.root}"
//!   return 0
//!
//! script srcdirs do
//!   for pkg in (← getWorkspace).packages do
//!     IO.println pkg.dir
//!   return 0
//! ```
//!
//! # More than one package
//!
//! The calling crate owns its Lake package, enforced by a `links` key: cargo
//! rejects a second crate claiming the same name, and `lake` takes no lock of
//! its own. Several packages means one such crate each. They invalidate
//! independently, so splitting the Lean is how the cost of an edit is bounded,
//! and the Lake package is the unit -- a crate that merely re-exports artifacts
//! still rebuilds when the crate that generated them does.
//!
//! # Limits
//!
//! Inputs are Lean sources, the toolchain pin and the resolved dependency
//! versions. A generator reading anything else is invisible to cargo.
//!
//! Outputs are not watched, for the reason [`generate`] gives, so a deleted
//! artifact fails the consumer's `include_bytes!` rather than being rebuilt;
//! any Lean edit repairs it. Presence is checked, not content.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use base_types::Artifact;

/// Build and run the generators of the Lake package at `lake_dir`, which is
/// taken relative to the crate whose build script is calling.
///
/// Artifacts are written to `artifacts/<module>/<name>.cbor` beside that crate,
/// and the path is published as `LEAN_ARTIFACT_DIR`, so the crate can expose it
/// as a constant. Every one of them decodes as an [`Artifact`] and re-encodes to
/// the same bytes; a generator that writes anything else fails the build.
pub fn generate(lake_dir: impl AsRef<Path>) {
    // The build script runs for the *calling* crate, so this is its directory
    // and not this library's.
    let manifest = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("generate is for use from a build script"),
    );
    let lake_dir = canonical(&manifest.join(lake_dir));

    let artifact_dir = manifest.join(ARTIFACT_DIR);
    fs::create_dir_all(&artifact_dir)
        .unwrap_or_else(|e| panic!("Failed to create {}: {e}", artifact_dir.display()));
    let artifact_dir = canonical(&artifact_dir);

    // Canonical paths throughout: cargo compares `rerun-if-changed` entries as
    // given, and one carrying `..` never matches, so the script would run on
    // every build.
    let packages: Vec<PathBuf> = script(&lake_dir, "srcdirs")
        .iter()
        .map(PathBuf::from)
        .collect();
    for dir in &packages {
        for path in sources(dir) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    // `artifact_dir` is deliberately not watched: cargo walks a watched
    // directory recursively, so watching what this writes leaves the script
    // stale the moment it finishes, doubling what a Lean edit costs.

    let generators = with_locks(&packages, || {
        let generators: Vec<(String, String)> = script(&lake_dir, "generators")
            .iter()
            .filter_map(|line| {
                let mut parts = line.split_whitespace();
                Some((parts.next()?.to_string(), parts.next()?.to_string()))
            })
            .collect();
        assert!(
            !generators.is_empty(),
            "No `lean_exe` targets in {}",
            lake_dir.display()
        );
        build(&lake_dir, &generators);
        generators
    });
    for (exe, module) in &generators {
        let exe = lake_dir.join(".lake/build/bin").join(exe);
        let out = artifact_dir.join(module);
        fs::create_dir_all(&out)
            .unwrap_or_else(|e| panic!("Failed to create {}: {e}", out.display()));
        // A skipped generator's output is checked all the same: what reads it
        // is base-types', and a change there leaves files that are no longer
        // artifacts beside a generator that has not changed.
        if !current(&exe, &out) || verify_all(&out).is_err() {
            let written = regenerate(module, &exe, &out);
            write_if_changed(&out.join(MANIFEST), written.join("\n").as_bytes());
        }
    }
    let modules: Vec<String> = generators.iter().map(|(_, m)| m.clone()).collect();
    drop_stale_modules(&artifact_dir, &modules);

    // For this crate's own code and tests, and for whatever depends on it,
    // which reaches the same path through `consume`.
    println!("cargo:rustc-env={ARTIFACT_DIR_VAR}={}", artifact_dir.display());
    println!("cargo:dir={}", artifact_dir.display());
}

/// The environment variable [`generate`] and [`consume`] publish the artifact
/// directory as, and that [`artifact`] reads.
const ARTIFACT_DIR_VAR: &str = "LEAN_ARTIFACT_DIR";

/// Where the generated tree goes, relative to the generating crate.
///
/// Not `OUT_DIR`: there is one per unit configuration, so build, check and test
/// would each hold their own copy of 104 MB. `git clean -xfd` removes this.
const ARTIFACT_DIR: &str = "artifacts";

/// Make the depended-on package's artifacts reachable by [`artifact`]:
///
/// ```no_run
/// fn main() {
///     build_support::consume();
/// }
/// ```
///
/// The path comes from cargo's `links` metadata, not from anything written
/// twice. With more than one package, name which with [`consume_from`].
pub fn consume() {
    match select(std::env::vars()) {
        Ok(dir) => publish(&dir),
        Err(message) => panic!("{message}"),
    }
}

/// The one artifact directory among a build script's environment, or what to
/// tell the author when there is not exactly one.
///
/// Separated from the environment so the messages can be read in a test: they
/// are what a first-time user of this crate actually meets.
fn select(vars: impl Iterator<Item = (String, String)>) -> Result<String, String> {
    let mut found: Vec<(String, String)> = vars
        .filter_map(|(k, v)| {
            let name = k.strip_prefix("DEP_")?.strip_suffix("_DIR")?;
            Some((name.to_lowercase(), v))
        })
        .collect();
    found.sort();
    match found.len() {
        1 => Ok(found.remove(0).1),
        0 => Err("No generated package to consume. This crate should depend on one \
                  whose build script calls `generate`."
            .to_string()),
        _ => {
            let names: Vec<String> = found.iter().map(|(k, _)| k.clone()).collect();
            Err(format!(
                "Several generated packages to consume: {}. Name one with \
                 `consume_from`.",
                names.join(", ")
            ))
        }
    }
}

/// [`consume`] for a crate depending on more than one generated package, naming
/// which by the `links` key of the crate that owns it.
pub fn consume_from(links: &str) {
    let var = format!("DEP_{}_DIR", links.to_uppercase());
    let dir = std::env::var(&var).unwrap_or_else(|_| {
        panic!("{var} is not set: this crate does not depend on a crate with `links = \"{links}\"`")
    });
    publish(&dir);
}

fn publish(dir: &str) {
    println!("cargo:rustc-env={ARTIFACT_DIR_VAR}={dir}");
}

/// The bytes of one artifact, by the path it was generated at: the module that
/// emitted it, then its name.
///
/// ```ignore
/// const ART: &[u8] = build_support::artifact!("Sha256Algorithm/sha256_app");
/// ```
///
/// Expands to an `include_bytes!` in the calling crate, whose build script must
/// have called [`consume`] -- or [`generate`], if it generates them itself.
#[macro_export]
macro_rules! artifact {
    ($path:literal) => {
        include_bytes!(concat!(
            env!(
                "LEAN_ARTIFACT_DIR",
                "no generated artifacts are in scope: the build script of the \
                 crate calling `artifact!` should call `build_support::consume()`"
            ),
            "/",
            $path,
            ".cbor"
        ))
    };
}

/// Hold an exclusive lock on each package directory for the duration of `f`.
///
/// Packages sharing a Lean dependency go stale together, so their build scripts
/// run at once; lake takes no lock, so each would rebuild the shared library
/// rather than one building and the rest replaying. Locking per directory keeps
/// unrelated packages from waiting. A lock that cannot be taken is skipped.
fn with_locks<T>(packages: &[PathBuf], f: impl FnOnce() -> T) -> T {
    // A fixed order, so two processes taking overlapping sets cannot deadlock.
    let mut dirs: Vec<&PathBuf> = packages.iter().collect();
    dirs.sort();

    let mut held = Vec::new();
    for dir in dirs {
        // `.lake` is lake's own output directory, so the lock lives with the
        // build rather than among the sources.
        let lake = dir.join(".lake");
        if fs::create_dir_all(&lake).is_err() {
            continue;
        }
        let Ok(file) = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lake.join("generate.lock"))
        else {
            continue;
        };
        if fs2::FileExt::lock_exclusive(&file).is_ok() {
            held.push(file);
        }
    }

    let out = f();
    drop(held);
    out
}

/// Run one of the package's Lake scripts and return its lines.
fn script(lake_dir: &Path, name: &str) -> Vec<String> {
    let output = Command::new("lake")
        .args(["run", name])
        .current_dir(lake_dir)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run `lake run {name}`: {e}"));
    check(&output, &format!("lake run {name}"));
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .filter(|l| !l.is_empty())
        .collect()
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize()
        .unwrap_or_else(|e| panic!("Failed to resolve {}: {e}", path.display()))
}

/// Whether a generator's output can be left alone.
///
/// What a generator emitted last time it ran, one name per line.
///
/// A generator is the only authority on which artifacts it has: reading the
/// directory answers "what is here", which cannot detect a gap.
const MANIFEST: &str = "generated.list";

/// Whether a generator's output can be left alone: every artifact it last
/// emitted is present and newer than the generator.
///
/// Lake leaves an executable untouched when it replays it, so an output newer
/// than the binary that produced it is current. This compares the two rather
/// than keeping a stamp: the binary is the only input a generator has.
fn current(exe: &Path, out: &Path) -> bool {
    let stamp = |p: &Path| p.metadata().and_then(|m| m.modified()).ok();
    let Some(built) = stamp(exe) else { return false };
    let Ok(list) = fs::read_to_string(out.join(MANIFEST)) else {
        return false;
    };
    let mut any = false;
    for name in list.lines().filter(|l| !l.is_empty()) {
        any = true;
        match stamp(&artifact_path(out, name)) {
            Some(t) if t > built => {}
            _ => return false,
        }
    }
    any
}

/// Every path that should make cargo ask lake again: Lean sources, the
/// toolchain pin, and the resolved dependency versions. Dot-directories are
/// lake's own output.
fn sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect(dir, &mut out);
    out.sort();
    out
}

/// Files only, never a directory: cargo walks a watched directory recursively,
/// so a package directory would drag in `.lake`, which lake rewrites during
/// this same build.
///
/// Files suffice -- a Lean file nothing imports is not in the build, and the
/// import that puts it there edits a file already watched.
fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|e| panic!("Failed to read {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_str().expect("Non-UTF8 file name");
        if path.is_dir() {
            if !name.starts_with('.') {
                collect(&path, out);
            }
        } else if name.ends_with(".lean")
            || name == "lean-toolchain"
            || name == "lake-manifest.json"
        {
            out.push(path);
        }
    }
}

fn build(lake_dir: &Path, generators: &[(String, String)]) {
    let output = Command::new("lake")
        .arg("build")
        .args(generators.iter().map(|(exe, _)| exe))
        .current_dir(lake_dir)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run lake: {e}"));
    check(&output, "lake build");
}

fn run(exe: &Path, out: &Path) {
    let output = Command::new(exe)
        .arg(out)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {e}", exe.display()));
    check(&output, &exe.display().to_string());
}

fn check(output: &std::process::Output, what: &str) {
    if output.status.success() {
        return;
    }
    eprintln!("=== {what} failed ===");
    eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    panic!("{what} failed");
}

/// The file an artifact is written to.
fn artifact_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.{EXTENSION}"))
}

const EXTENSION: &str = "cbor";

/// Where a generator runs, inside its own output directory so the results can
/// be moved into place rather than copied across filesystems.
const STAGING: &str = ".staging";

/// Run a generator into a fresh staging directory, check what it wrote, and
/// put it in place, returning the artifact names in order.
///
/// A `.cbor` at the top of the staging directory is an artifact; a generator's
/// other output goes in a subdirectory, which is moved over whole. Anything
/// else at the top level is an error rather than something to skip, so a
/// generator cannot emit a file no application will ever find.
///
/// Staging is what keeps a failed run from half-replacing the output: nothing
/// moves until every artifact has been checked.
fn regenerate(module: &str, exe: &Path, out: &Path) -> Vec<String> {
    let staging = out.join(STAGING);
    if staging.exists() {
        fs::remove_dir_all(&staging)
            .unwrap_or_else(|e| panic!("Failed to remove {}: {e}", staging.display()));
    }
    fs::create_dir_all(&staging)
        .unwrap_or_else(|e| panic!("Failed to create {}: {e}", staging.display()));
    run(exe, &staging);

    let mut artifacts = Vec::new();
    let mut staged = Vec::new();
    let mut subdirs = Vec::new();
    for path in entries(&staging) {
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        if path.is_dir() {
            subdirs.push(name);
        } else if path.extension().and_then(|e| e.to_str()) == Some(EXTENSION) {
            let bytes = fs::read(&path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
            if let Err(e) = verify(&bytes) {
                panic!("{module}/{name}: {e}");
            }
            artifacts.push(path.file_stem().unwrap_or_default().to_string_lossy().to_string());
            staged.push(bytes);
        } else {
            panic!("{module}/{name} is not an artifact; other output belongs in a subdirectory");
        }
    }
    assert!(!artifacts.is_empty(), "{module} wrote no artifacts");

    for (name, bytes) in artifacts.iter().zip(&staged) {
        write_if_changed(&artifact_path(out, name), bytes);
    }
    // What is at the top level and not just written belongs to an artifact the
    // generator no longer has, or to an encoding that is gone. Left in place it
    // would still satisfy an `include_bytes!`.
    for path in entries(out) {
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let keep = name == MANIFEST
            || name == STAGING
            || (path.is_file()
                && path.extension().and_then(|e| e.to_str()) == Some(EXTENSION)
                && artifacts.contains(&stem));
        if !keep {
            remove(&path);
        }
    }
    for name in subdirs {
        fs::rename(staging.join(&name), out.join(&name))
            .unwrap_or_else(|e| panic!("Failed to move {module}/{name} into place: {e}"));
    }
    remove(&staging);
    artifacts
}

/// That `bytes` are an artifact, written in the one encoding
/// [`Artifact::to_bytes`] produces. A Lean writer that disagrees with serde on
/// any detail of the profile fails here rather than being read some other way.
fn verify(bytes: &[u8]) -> Result<(), String> {
    let artifact = Artifact::from_bytes(bytes)?;
    let again = artifact.to_bytes();
    if again.as_slice() != bytes {
        let at = bytes.iter().zip(&again).position(|(a, b)| a != b).unwrap_or(bytes.len().min(again.len()));
        return Err(format!(
            "decodes, but is not in the artifact encoding: it re-encodes differently from byte {at} \
             ({} bytes written, {} re-encoded)",
            bytes.len(),
            again.len()
        ));
    }
    Ok(())
}

/// Every artifact a generator last emitted, checked as [`verify`] does.
fn verify_all(out: &Path) -> Result<(), String> {
    let list = fs::read_to_string(out.join(MANIFEST)).map_err(|e| e.to_string())?;
    for name in list.lines().filter(|l| !l.is_empty()) {
        let bytes = fs::read(artifact_path(out, name)).map_err(|e| e.to_string())?;
        verify(&bytes)?;
    }
    Ok(())
}

/// The entries directly in `dir`, sorted.
fn entries(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", dir.display()))
        .map(|e| e.expect("Failed to read directory entry").path())
        .collect();
    paths.sort();
    paths
}

fn remove(path: &Path) {
    let result = if path.is_dir() { fs::remove_dir_all(path) } else { fs::remove_file(path) };
    result.unwrap_or_else(|e| panic!("Failed to remove {}: {e}", path.display()));
}

/// Leave an unchanged artifact untouched. Applications reach these files with
/// `include_bytes!`, so rewriting one with identical content would still make
/// rustc recompile everything that reads it.
fn write_if_changed(path: &Path, bytes: &[u8]) {
    if fs::read(path).map(|old| old == bytes).unwrap_or(false) {
        return;
    }
    fs::write(path, bytes).unwrap_or_else(|e| panic!("Failed to write {}: {e}", path.display()));
}

/// A generator removed from the lakefile leaves a whole
/// directory of artifacts that still satisfy an `include_bytes!`, so an
/// application would keep compiling against a program nothing emits any more.
///
/// Only called with the full generator list, which `generate` has asserted is
/// not empty -- an empty one would read as "every module is stale".
fn drop_stale_modules(dir: &Path, modules: &[String]) {
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() && !modules.contains(&name) {
            fs::remove_dir_all(&path)
                .unwrap_or_else(|e| panic!("Failed to remove {}: {e}", path.display()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    /// Two callers holding the same package directory do not overlap.
    ///
    /// `flock` is per open file description rather than per process, so two
    /// handles contend here exactly as two build scripts would.
    #[test]
    fn locks_on_a_shared_package_serialize() {
        let dir = std::env::temp_dir().join(format!("bs-lock-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let dirs = vec![dir.clone()];

        let (tx, rx) = mpsc::channel();
        let start = Instant::now();
        let spans: Vec<_> = (0..2)
            .map(|_| {
                let dirs = dirs.clone();
                let tx = tx.clone();
                std::thread::spawn(move || {
                    with_locks(&dirs, || {
                        let enter = start.elapsed();
                        std::thread::sleep(Duration::from_millis(200));
                        tx.send((enter, start.elapsed())).expect("send");
                    })
                })
            })
            .collect();
        for s in spans {
            s.join().expect("thread");
        }
        drop(tx);

        let mut got: Vec<(Duration, Duration)> = rx.iter().collect();
        got.sort();
        assert_eq!(got.len(), 2, "both callers ran");
        assert!(
            got[1].0 >= got[0].1,
            "the second caller entered at {:?}, before the first left at {:?}",
            got[1].0,
            got[0].1
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// What a build script author is told when `consume` cannot choose.
    #[test]
    fn consuming_says_what_to_do_about_it() {
        let vars = |kvs: &[(&str, &str)]| {
            kvs.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<Vec<_>>()
                .into_iter()
        };
        // Cargo's other metadata is not an artifact directory.
        let one = vars(&[("DEP_ALPHA_DIR", "/a"), ("PATH", "/usr/bin"), ("OUT_DIR", "/o")]);
        assert_eq!(select(one), Ok("/a".to_string()));

        let none = select(vars(&[("PATH", "/usr/bin")])).expect_err("nothing to consume");
        assert!(none.contains("generate"), "names the call that would fix it: {none}");

        let many = select(vars(&[("DEP_BETA_DIR", "/b"), ("DEP_ALPHA_DIR", "/a")]))
            .expect_err("cannot choose");
        assert!(many.contains("consume_from"), "names the call to use: {many}");
        // The `links` keys as they are written in a manifest, so they can be
        // pasted straight into `consume_from`.
        assert!(many.contains("alpha, beta"), "lists them in order: {many}");
    }

    /// Deleting any one of a generator's outputs makes it run again.
    ///
    /// The case that matters is a module with several artifacts: asking whether
    /// the directory holds something newer than the generator answers yes on
    /// the strength of its *other* artifacts, so the gap goes unseen.
    #[test]
    fn a_deleted_output_is_not_current() {
        let dir = std::env::temp_dir().join(format!("bs-current-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let exe = dir.join("gen");
        fs::write(&exe, b"generator").expect("exe");
        // The generator's outputs are written after it, as a real run leaves
        // them; the check is `newer than the generator`.
        std::thread::sleep(Duration::from_millis(10));
        for name in ["one", "two"] {
            fs::write(artifact_path(&dir, name), b"\0").expect("artifact");
        }
        fs::write(dir.join(MANIFEST), "one\ntwo").expect("manifest");
        assert!(current(&exe, &dir), "a complete output is current");

        for gone in ["one.cbor", MANIFEST] {
            let path = dir.join(gone);
            let saved = fs::read(&path).expect("read back");
            fs::remove_file(&path).expect("remove");
            assert!(!current(&exe, &dir), "{gone} is missing, so this is not current");
            fs::write(&path, &saved).expect("restore");
        }

        // A generator lake rebuilt is newer than everything beside it.
        fs::write(&exe, b"rebuilt").expect("rebuild");
        assert!(!current(&exe, &dir), "output older than the generator is stale");
        fs::remove_dir_all(&dir).ok();
    }

    /// An artifact is checked for its encoding, not only for decoding: a head
    /// wider than it needs to be reads as the same artifact, and is still
    /// refused, because it is a writer disagreeing with the profile.
    #[test]
    fn an_artifact_in_another_encoding_is_refused() {
        let good = Artifact { functions: vec![], memory_size: 8, data: vec![] }.to_bytes();
        assert_eq!(verify(&good), Ok(()));

        // `memory_size: 8` as a one-byte head (0x08) spelled with two (0x18 0x08).
        let at = good.windows(2).rposition(|w| w == [0x08, 0x64]).expect("memory_size value");
        let mut wide = good.clone();
        wide.splice(at..at + 1, [0x18, 0x08]);
        assert_eq!(Artifact::from_bytes(&wide).map(|a| a.memory_size), Ok(8));
        let err = verify(&wide).expect_err("a wide head is not the encoding");
        assert!(err.contains("re-encodes differently"), "{err}");

        assert!(verify(b"{}").expect_err("JSON").starts_with("not an artifact"));
    }

    /// Directories with nothing in common are not made to wait for each other.
    #[test]
    fn locks_on_separate_packages_do_not_contend() {
        let base = std::env::temp_dir().join(format!("bs-free-{}", std::process::id()));
        let (a, b) = (base.join("a"), base.join("b"));
        fs::create_dir_all(&a).expect("a");
        fs::create_dir_all(&b).expect("b");

        let held = with_locks(&[a.clone()], || {
            // Taken while the first is still held: a different directory, so
            // this must not block.
            with_locks(&[b.clone()], || 7)
        });
        assert_eq!(held, 7);
        fs::remove_dir_all(&base).ok();
    }
}
