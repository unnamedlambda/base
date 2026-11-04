//! The native-code entry points on an x86-64 host, against the interpreter
//! (`Host/NativeCorpus.lean`), byte for byte.


use std::path::PathBuf;

/// Result codes, `i32`s from byte 0.
const WHAT: [&str; 6] = [
    "a null source places nothing",
    "a zero length places nothing",
    "code placed",
    "architecture",
    "sse2 present",
    "a feature name the runtime does not know",
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

    let b = base::Driver::load(lean_artifacts::HPROG_NATIVE_CORPUS).expect("compile");
    let mut out = vec![0u8; expected.len()];
    unsafe { b.function("main") }.expect("execute")(&[], &mut out);

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
