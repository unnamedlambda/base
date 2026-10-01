//! Runs the stream corpus: what `Sem.callFfi` says created streams, events and
//! captured graphs do, against what they do on the device.
//!
//! The interpreter computed the output buffer when the artifact was generated
//! (`lean/algorithms/Host/StreamCorpus.lean`); here the same artifact runs
//! through the JIT and the two must agree byte for byte. Two streams ordered by
//! an event, then a graph captured across a fork into a second stream and a
//! join back, launched twice. The model runs every operation when it is issued
//! and refuses a conflict nothing orders, so agreeing here says both that the
//! contracts are right and that the order the model chose is the device's.

use base_types::Artifact;

use std::path::PathBuf;

const WHAT: [&str; 28] = [
    "stream 0 created",
    "stream 1 created",
    "launch on stream 0",
    "event created",
    "event recorded on stream 0",
    "stream 1 waits for it",
    "launch on stream 1",
    "sync stream 1",
    "download",
    "capture begins on stream 0",
    "captured launch on stream 0",
    "event recorded in the capture",
    "stream 1 joins the capture",
    "captured launch on stream 1",
    "event recorded on stream 1",
    "stream 0 joins it back",
    "capture ends: graph 0",
    "graph uploaded",
    "graph launched",
    "graph launched again",
    "sync stream 0",
    "download X",
    "download Y",
    "stream 1 destroyed",
    "stream 1 destroyed twice refused",
    "graph destroyed",
    "event destroyed",
    "launch on a stream that does not exist refused",
];

#[test]
fn interpreter_and_device_agree() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_stream_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> = v["expected"]
        .as_array()
        .expect("expected")
        .iter()
        .map(|b| b.as_u64().expect("byte") as u8)
        .collect();
    let a = Artifact::from_bytes(lean_artifacts::HPROG_STREAM_CORPUS).expect("artifact shape");

    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &[], &mut out).expect("execute");

    let mut bad = Vec::new();
    for (k, what) in WHAT.iter().enumerate() {
        let (got, want) = (&out[4 * k..4 * k + 4], &expected[4 * k..4 * k + 4]);
        if got != want {
            bad.push(format!(
                "  {what}: lean {} != device {}",
                i32::from_le_bytes(want.try_into().unwrap()),
                i32::from_le_bytes(got.try_into().unwrap())
            ));
        }
    }
    for (lo, hi, what) in [
        (128, 192, "X after launches on two streams ordered by an event"),
        (192, 256, "X after the graph ran twice"),
        (256, 320, "Y after the graph ran twice"),
    ] {
        if out[lo..hi] != expected[lo..hi] {
            bad.push(format!("  {what}: lean {:?} != device {:?}", &expected[lo..hi], &out[lo..hi]));
        }
    }
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
    assert_eq!(out, expected, "every byte of the output buffer");
    eprintln!("{} stream contract checks agree", WHAT.len() + 3);
}
