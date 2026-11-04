//! Threads a program starts on its own function, against the interpreter
//! (`Host/ThreadCorpus.lean`), byte for byte.

use std::path::PathBuf;

#[test]
fn interpreter_and_jit_agree_on_threads() {
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

    let b = base::Driver::load(lean_artifacts::HPROG_THREAD_CORPUS).expect("compile");
    let mut out = vec![0u8; expected.len()];
    unsafe { b.function("main") }.expect("execute")(&[], &mut out);

    let word = |bs: &[u8], k: usize| i64::from_le_bytes(bs[8 * k..8 * k + 8].try_into().unwrap());
    // Thread k sums 10k+1 .. 10k+4, leaves the sum in its chunk and answers twice it.
    for k in 0..4 {
        let sum = 40 * k as i64 + 10;
        assert_eq!(word(&out, 2 * k), sum, "the word thread {k} left");
        assert_eq!(word(&out, 2 * k + 1), 2 * sum, "thread {k}'s answer");
    }
    assert_eq!(out, expected, "every byte of the output buffer");
    eprintln!("4 threads agree with the interpreter");
}
