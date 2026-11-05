//! Calls to a program's own functions, against the interpreter
//! (`Host/LocalCorpus.lean`), byte for byte.


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

    let b = unsafe { base::Driver::load(lean_artifacts::HPROG_LOCAL_CORPUS).expect("compile") };
    let mut out = vec![0u8; expected.len()];
    b.function("main").expect("execute")(&[], &mut out);

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
