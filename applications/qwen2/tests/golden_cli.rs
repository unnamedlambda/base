//! End-to-end golden tests: spawn each qwen2 binary, pipe a prompt on stdin,
//! assert the response on stdout matches an inlined expected string.  The
//! in-memory (`qwen2`) and on-disk (`qwen2_on_disk`) binaries should produce
//! byte-identical output — only the storage strategy differs.
//!
//! Requires weights + tokenizer at the repo root. If missing, the tests SKIP
//! with a message — generate them once via `tools/qwen2_convert.py`. The .bin
//! files are gitignored.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const QWEN2_BIN: &str = env!("CARGO_BIN_EXE_qwen2");
const QWEN2_ON_DISK_BIN: &str = env!("CARGO_BIN_EXE_qwen2_on_disk");

const EXPECTED_HELLO: &str = "hello! how can i assist you today?\n";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Returns (weights_path, tokenizer_path) if both exist, else None.
/// Prefers the instruct-tuned variants when available.
fn locate_files() -> Option<(PathBuf, PathBuf)> {
    let root = repo_root();
    let weights_candidates = ["qwen2_instruct_weights.bin", "qwen2_weights.bin"];
    let tokenizer_candidates = ["qwen2_instruct_tokenizer.bin", "qwen2_tokenizer.bin"];
    let weights = weights_candidates
        .iter()
        .map(|n| root.join(n))
        .find(|p| p.exists())?;
    let tokenizer = tokenizer_candidates
        .iter()
        .map(|n| root.join(n))
        .find(|p| p.exists())?;
    Some((weights, tokenizer))
}

fn ask(bin: &str, weights: &PathBuf, tokenizer: &PathBuf, prompt: &str) -> String {
    let mut child = Command::new(bin)
        .args([weights.as_os_str(), tokenizer.as_os_str()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {bin}: {e}"));
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{prompt}").unwrap();
    drop(stdin);
    let out = child
        .wait_with_output()
        .unwrap_or_else(|e| panic!("wait {bin}: {e}"));
    assert!(out.status.success(), "{bin} exited {:?}", out.status.code());
    String::from_utf8(out.stdout).expect("utf-8 response")
}

fn skip_message(name: &str) {
    eprintln!(
        "SKIP {name}: weights/tokenizer .bin files not found at repo root. \
         Generate them with tools/qwen2_convert.py to enable this test."
    );
}

#[test]
fn cli_golden_hello() {
    let Some((weights, tokenizer)) = locate_files() else {
        skip_message("cli_golden_hello");
        return;
    };
    assert_eq!(
        ask(QWEN2_BIN, &weights, &tokenizer, "hello"),
        EXPECTED_HELLO
    );
}

/// The on-disk variant streams weights layer-by-layer from the .bin file and
/// backs the KV cache with a file on disk.  Must match `cli_golden_hello`
/// exactly — same model, same prompt, same output.
#[test]
fn on_disk_golden_hello() {
    let Some((weights, tokenizer)) = locate_files() else {
        skip_message("on_disk_golden_hello");
        return;
    };
    assert_eq!(
        ask(QWEN2_ON_DISK_BIN, &weights, &tokenizer, "hello"),
        EXPECTED_HELLO
    );
}
