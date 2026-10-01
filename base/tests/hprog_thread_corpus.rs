//! Runs the thread corpus: what the model says `cl_thread_*` do, against what
//! they do.
//!
//! `main` spawns one of its own functions as a worker on a pointer into the
//! output buffer, joins it, then records the handle, the join's answer, a
//! second join of the same handle, and two refused spawns. The interpreter
//! computed that buffer when the artifact was generated
//! (`lean/algorithms/Host/ThreadCorpus.lean`), running the worker at the spawn;
//! here the same artifact runs through the JIT with a real thread, and the two
//! must agree byte for byte.

use base_types::Artifact;

use std::path::PathBuf;

#[test]
fn interpreter_and_runtime_agree_on_threads() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_thread_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> = v["expected"]
        .as_array()
        .expect("expected")
        .iter()
        .map(|b| b.as_u64().expect("byte") as u8)
        .collect();
    let a = Artifact::from_bytes(lean_artifacts::HPROG_THREAD_CORPUS).expect("artifact shape");

    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &[], &mut out).expect("execute");

    let word = |bs: &[u8], k: usize| i64::from_le_bytes(bs[8 * k..8 * k + 8].try_into().unwrap());
    for (k, what) in [
        (0, "the handle"),
        (1, "join"),
        (2, "joining again refused"),
        (3, "spawning a function that answers refused"),
        (4, "spawning with a null context refused"),
        (8, "the worker's first store"),
        (9, "the worker's second store, from its first"),
    ] {
        assert_eq!(word(&out, k), word(&expected, k), "{what}");
    }
    assert_eq!(out, expected, "every byte of the output buffer");
    eprintln!("7 thread checks agree");
}
