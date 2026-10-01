//! Runs the native-code corpus: what `Sem.callFfi` says the native entry
//! points do, against what they do on an x86-64 host.
//!
//! The body loads a few bytes of code, frees them twice, frees an address that
//! was never loaded, and asks for the architecture and two CPU features, leaving
//! each result code in the output buffer. The interpreter computed that buffer
//! when the artifact was generated (`lean/algorithms/Host/NativeCorpus.lean`),
//! with its oracles set to an x86-64 host; here the same artifact runs through
//! the JIT, and the two must agree byte for byte.

use base_types::Artifact;

use std::path::PathBuf;

/// Result codes, `i32`s from byte 0.
const WHAT: [&str; 9] = [
    "free what a null source loaded",
    "free what a zero length loaded",
    "free loaded code",
    "free it twice refused",
    "free an address never loaded refused",
    "architecture",
    "sse2 present",
    "a feature name the runtime does not know",
    "a null feature name",
];

#[cfg(target_arch = "x86_64")]
#[test]
fn interpreter_and_runtime_agree() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_native_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> = v["expected"]
        .as_array()
        .expect("expected")
        .iter()
        .map(|b| b.as_u64().expect("byte") as u8)
        .collect();
    let a = Artifact::from_bytes(lean_artifacts::HPROG_NATIVE_CORPUS).expect("artifact shape");

    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &[], &mut out).expect("execute");

    let mut bad = Vec::new();
    for (k, what) in WHAT.iter().enumerate() {
        let (got, want) = (&out[4 * k..4 * k + 4], &expected[4 * k..4 * k + 4]);
        if got != want {
            bad.push(format!(
                "  {what}: lean {} != runtime {}",
                i32::from_le_bytes(want.try_into().unwrap()),
                i32::from_le_bytes(got.try_into().unwrap())
            ));
        }
    }
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
    assert_eq!(out, expected, "every byte of the output buffer");
    eprintln!("{} native contract checks agree", WHAT.len());
}
