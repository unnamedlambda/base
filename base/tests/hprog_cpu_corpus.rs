//! The CPU library against the system (`Host/CpuCorpus.lean`).

use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    }
}

#[test]
fn interpreter_and_system_agree() {
    let dir = corpus_dir();
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_cpu_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> =
        v["expected"].as_array().expect("expected").iter().map(|b| b.as_u64().expect("byte") as u8).collect();
    let names: Vec<String> =
        v["names"].as_array().expect("names").iter().map(|n| n.as_str().expect("name").to_string()).collect();

    let b = unsafe { base::Driver::load(&std::fs::read(dir.join("hprog_cpu_corpus.cbor")).expect("read artifact")).expect("compile") };
    let mut out = vec![0u8; expected.len()];
    b.function("main").expect("execute")(&[], &mut out);

    let word = |bs: &[u8], k: usize| i32::from_le_bytes(bs[4 * k..4 * k + 4].try_into().unwrap());
    let bad: Vec<String> = names
        .iter()
        .enumerate()
        .filter(|&(k, _)| word(&out, k) != word(&expected, k))
        .map(|(k, name)| format!("  {name}: lean {} != system {}", word(&expected, k), word(&out, k)))
        .collect();
    assert!(bad.is_empty(), "{} results disagree:\n{}", bad.len(), bad.join("\n"));
    eprintln!("{} CPU answers agree", names.len());
}
