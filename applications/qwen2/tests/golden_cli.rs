//! End-to-end golden test: spawn the qwen2 binary, pipe a prompt on stdin,
//! assert the response on stdout matches an inlined expected string.
//!
//! Requires weights + tokenizer at the repo root. If missing, the test SKIPs
//! with a message — generate them once via `tools/qwen2_convert.py`. The .bin
//! files are gitignored.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const QWEN2_BIN: &str = env!("CARGO_BIN_EXE_qwen2");

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Returns (weights_path, tokenizer_path) if both exist, else None.
/// Prefers the instruct-tuned variants when available.
fn locate_files() -> Option<(PathBuf, PathBuf)> {
    let root = repo_root();
    let weights_candidates = ["qwen2_instruct_weights.bin", "qwen2_weights.bin"];
    let tokenizer_candidates = ["qwen2_instruct_tokenizer.bin", "qwen2_tokenizer.bin"];
    let weights = weights_candidates.iter().map(|n| root.join(n)).find(|p| p.exists())?;
    let tokenizer = tokenizer_candidates.iter().map(|n| root.join(n)).find(|p| p.exists())?;
    Some((weights, tokenizer))
}

fn ask(weights: &PathBuf, tokenizer: &PathBuf, prompt: &str) -> String {
    let mut child = Command::new(QWEN2_BIN)
        .args([weights.as_os_str(), tokenizer.as_os_str()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn qwen2");
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{prompt}").unwrap();
    drop(stdin);
    let out = child.wait_with_output().expect("wait qwen2");
    assert!(out.status.success(), "qwen2 exited {:?}", out.status.code());
    String::from_utf8(out.stdout).expect("utf-8 response")
}

#[test]
fn cli_golden_hello() {
    let Some((weights, tokenizer)) = locate_files() else {
        eprintln!(
            "SKIP cli_golden_hello: weights/tokenizer .bin files not found at repo root. \
             Generate them with tools/qwen2_convert.py to enable this test."
        );
        return;
    };
    assert_eq!(ask(&weights, &tokenizer, "hello"), "hello! how can i assist you today?\n");
}
