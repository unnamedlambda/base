//! The C library calls of the corpus, on a machine without the libraries.
//!
//! `lean/algorithms/Host/Corpus.lean` interprets `absentCases` in a world where
//! no library loaded: every function answers `-1` and touches nothing, and each
//! presence probe answers `0`. This makes that machine out of this one: each
//! primitive's library is renamed to one the engine does not know, which binds
//! as a library that is not installed does. Then it runs the artifact and
//! compares.

use base_types::Artifact;
use std::path::PathBuf;

#[test]
fn absent_libraries_answer_as_modelled() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_absent_corpus/expected.json"))
            .expect("read corpus"),
    )
    .expect("corpus shape");
    let stride = v["stride"].as_u64().expect("stride") as usize;
    let names: Vec<String> = v["names"]
        .as_array()
        .expect("names")
        .iter()
        .map(|n| n.as_str().expect("name").to_string())
        .collect();
    let expected: Vec<u8> = v["expected"]
        .as_array()
        .expect("expected")
        .iter()
        .map(|b| b.as_u64().expect("byte") as u8)
        .collect();
    let mut a = Artifact::from_bytes(
        &std::fs::read(dir.join("hprog_absent_corpus.cbor")).expect("read artifact"),
    )
    .expect("artifact shape");

    for p in &mut a.primitives {
        p.lib = format!("missing-{}", p.lib);
    }

    let b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    unsafe { b.execute("main", &[], &mut out) }.expect("execute");

    let bad: Vec<String> = names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            let (got, want) = (&out[i * stride..(i + 1) * stride], &expected[i * stride..(i + 1) * stride]);
            (got != want).then(|| format!("  {name}: lean {want:02x?} != jit {got:02x?}"))
        })
        .collect();
    assert!(bad.is_empty(), "{} of {} cases disagree:\n{}", bad.len(), names.len(), bad.join("\n"));
    eprintln!("{} absent-library cases agree", names.len());
}
