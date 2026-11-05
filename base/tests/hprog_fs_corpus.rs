//! The file library against the system (`Host/FsCorpus.lean`), in a
//! temporary directory of its own.

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
        &std::fs::read_to_string(dir.join("hprog_fs_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> =
        v["expected"].as_array().expect("expected").iter().map(|b| b.as_u64().expect("byte") as u8).collect();
    let names: Vec<String> =
        v["names"].as_array().expect("names").iter().map(|n| n.as_str().expect("name").to_string()).collect();
    let field = v["field"].as_u64().expect("field") as usize;

    let tmp = tempfile::TempDir::new().unwrap();
    let mut input = Vec::new();
    for name in ["x", "x/f", "x/g", "x/h"] {
        let p = tmp.path().join(name);
        let mut bytes = p.to_str().unwrap().as_bytes().to_vec();
        assert!(bytes.len() < field, "the temporary path fits its field");
        bytes.resize(field, 0);
        input.extend(bytes);
    }

    let b = unsafe { base::Driver::load(&std::fs::read(dir.join("hprog_fs_corpus.cbor")).expect("read artifact")).expect("compile") };
    let mut out = vec![0u8; expected.len()];
    b.function("main").expect("execute")(&input, &mut out);

    let word = |bs: &[u8], k: usize| i32::from_le_bytes(bs[4 * k..4 * k + 4].try_into().unwrap());
    let bad: Vec<String> = names
        .iter()
        .enumerate()
        .filter(|&(k, _)| word(&out, k) != word(&expected, k))
        .map(|(k, name)| format!("  {name}: lean {} != system {}", word(&expected, k), word(&out, k)))
        .collect();
    assert!(bad.is_empty(), "{} results disagree:\n{}", bad.len(), bad.join("\n"));
    assert!(!tmp.path().join("x").exists(), "the directory is gone");
    eprintln!("{} file answers agree", names.len());
}
