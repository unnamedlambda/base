//! wgpu, called directly, checked against wgpu.
//!
//! `lean/algorithms/Host/WgpuCorpus.lean` makes an instance, adapter, device
//! and queue, writes a storage buffer, runs a compute pipeline made inside an
//! error scope over it, copies it to a staging buffer and reads it back, and
//! has a shader wgpu rejects caught by a scope, storing every answer and the
//! bytes read. This runs the artifact and compares with what the interpreter
//! computed.

use base_types::Artifact;
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    }
}

#[test]
fn interpreter_and_wgpu_agree() {
    let dir = corpus_dir();
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_wgpu_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> =
        v["expected"].as_array().expect("expected").iter().map(|b| b.as_u64().expect("byte") as u8).collect();
    let names: Vec<String> =
        v["names"].as_array().expect("names").iter().map(|n| n.as_str().expect("name").to_string()).collect();
    let read = v["read"].as_u64().expect("read") as usize;
    let a = Artifact::from_bytes(&std::fs::read(dir.join("hprog_wgpu_corpus.cbor")).expect("read artifact"))
        .expect("artifact shape");

    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &[], &mut out).expect("execute");

    let mut bad: Vec<String> = names
        .iter()
        .enumerate()
        .filter_map(|(k, name)| {
            let (got, want) = (&out[4 * k..4 * k + 4], &expected[4 * k..4 * k + 4]);
            (got != want).then(|| format!("  {name}: lean {want:02x?} != wgpu {got:02x?}"))
        })
        .collect();
    if out[read..] != expected[read..] {
        bad.push(format!("  the bytes read back: lean {:02x?} != wgpu {:02x?}", &expected[read..], &out[read..]));
    }
    assert!(bad.is_empty(), "{} results disagree:\n{}", bad.len(), bad.join("\n"));
}
