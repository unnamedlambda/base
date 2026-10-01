//! Runs the LMDB corpus: what `Sem.callFfi` says the LMDB entry points do,
//! against what they do on real directories.
//!
//! One body opens two environments, puts inside and outside write
//! transactions, scans from the start and from a key, and reopens a directory
//! from a fresh context; every result code and every scan's bytes land in the
//! output buffer. The interpreter computed that buffer when the artifact was
//! generated (`lean/algorithms/Host/LmdbCorpus.lean`); here the same artifact
//! runs through the JIT, and the two must agree byte for byte.

use base_types::Artifact;

use std::path::{Path, PathBuf};

/// The directories the corpus body names.
const DIRS: &str = "/tmp/base-hprog-lmdb-corpus";

/// Result codes, `i32`s from byte 0.
const WHAT: [&str; 27] = [
    "open a",
    "open b",
    "open with a null context refused",
    "open an empty path refused",
    "put outside a transaction",
    "second put outside a transaction",
    "put with an empty key refused",
    "put with a negative key length refused",
    "put to a handle that names nothing refused",
    "begin on a",
    "put inside the transaction",
    "overwrite inside the transaction",
    "put a key that another begins",
    "scan inside the transaction",
    "commit",
    "commit with no transaction refused",
    "begin on b",
    "put on b",
    "begin on b again",
    "commit b",
    "scan b after the discarded put",
    "scan a from a start key",
    "scan a, at most one",
    "scan a handle that names nothing",
    "scan with a null context",
    "reopen a from a fresh context",
    "scan the reopened a, no limit",
];

#[test]
fn interpreter_and_library_agree() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_lmdb_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> = v["expected"]
        .as_array()
        .expect("expected")
        .iter()
        .map(|b| b.as_u64().expect("byte") as u8)
        .collect();
    let a = Artifact::from_bytes(lean_artifacts::HPROG_LMDB_CORPUS).expect("artifact shape");

    let _ = std::fs::remove_dir_all(Path::new(DIRS));
    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &[], &mut out).expect("execute");
    drop(b);
    let _ = std::fs::remove_dir_all(Path::new(DIRS));

    let mut bad = Vec::new();
    for (k, what) in WHAT.iter().enumerate() {
        let (got, want) = (&out[4 * k..4 * k + 4], &expected[4 * k..4 * k + 4]);
        if got != want {
            bad.push(format!(
                "  {what}: lean {} != lmdb {}",
                i32::from_le_bytes(want.try_into().unwrap()),
                i32::from_le_bytes(got.try_into().unwrap())
            ));
        }
    }
    for (lo, hi, what) in [
        (128, 192, "scan inside the transaction"),
        (192, 200, "scan of b"),
        (200, 224, "scan from a start key"),
        (224, 240, "scan of at most one"),
        (240, 248, "scan of a handle that names nothing"),
        (248, 256, "scan with a null context left alone"),
        (256, 320, "scan of the reopened directory"),
    ] {
        if out[lo..hi] != expected[lo..hi] {
            bad.push(format!("  {what}: lean {:?} != lmdb {:?}", &expected[lo..hi], &out[lo..hi]));
        }
    }
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
    assert_eq!(out, expected, "every byte of the output buffer");
    eprintln!("{} LMDB contract checks agree", WHAT.len() + 7);
}
