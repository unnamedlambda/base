//! Runs the local-call corpus: what `Sem` says a call to one of the program's
//! own functions does, against what the JIT does.
//!
//! `main` calls a function that answers, twice, and a function that answers
//! nothing and calls the first itself, storing every answer in the output
//! buffer. The interpreter computed that buffer when the artifact was generated
//! (`lean/algorithms/Host/LocalCorpus.lean`), each local call running its
//! callee's term; here the same artifact runs through the JIT, and the two must
//! agree byte for byte.

use base_types::Artifact;

use std::path::PathBuf;

#[test]
fn interpreter_and_jit_agree_on_local_calls() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_local_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> = v["expected"]
        .as_array()
        .expect("expected")
        .iter()
        .map(|b| b.as_u64().expect("byte") as u8)
        .collect();
    let a = Artifact::from_bytes(lean_artifacts::HPROG_LOCAL_CORPUS).expect("artifact shape");

    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &[], &mut out).expect("execute");

    let word = |bs: &[u8], k: usize| i64::from_le_bytes(bs[8 * k..8 * k + 8].try_into().unwrap());
    for (k, what) in [
        "triple 5",
        "triple -7",
        "spread's v",
        "spread's v + 1",
        "spread's call of triple",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(word(&out, k), word(&expected, k), "{what}");
    }
    assert_eq!(out, expected, "every byte of the output buffer");
    eprintln!("5 local-call checks agree");
}
