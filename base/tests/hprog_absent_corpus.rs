//! The corpus's library calls on a machine without the libraries: each
//! primitive's library is renamed to one the engine does not know.

use base_types::Artifact;
use std::path::PathBuf;

#[test]
fn absent_libraries_answer_as_modelled() {
    let dir = PathBuf::from(lean_artifacts::DIR);
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

    let b = unsafe { base::Driver::load(&a.to_bytes()).expect("compile") };
    let mut out = vec![0u8; expected.len()];
    b.function("main").expect("execute")(&[], &mut out);

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
