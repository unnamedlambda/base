//! Runs the wgpu corpus: what `Sem.callFfi` says the nine wgpu entry points do,
//! against what `Lib.Wgpu` does on the device over wgpu.
//!
//! The interpreter computed the output buffer when the artifact was generated
//! (`lean/algorithms/Host/GpuCorpus.lean`); here the same artifact runs through
//! the JIT, and the two must agree byte for byte. The shader adds one to every
//! `u32`, which is also the interpreter's shader oracle. One case checks the
//! queue's order rather than a value: a dispatch sees an upload made after it
//! and before the download that submits it.

use base_types::Artifact;

use std::path::PathBuf;

const WHAT: [&str; 20] = [
    "create 64",
    "create 0 refused",
    "upload",
    "upload to a buffer that does not exist refused",
    "pipeline",
    "pipeline binding a buffer that does not exist refused",
    "dispatch",
    "dispatch of a pipeline that does not exist refused",
    "download",
    "dispatch again",
    "upload after the dispatch",
    "download",
    "download at an offset",
    "upload past the buffer refused",
    "upload of a part word refused",
    "download of part of the buffer refused",
    "download past the buffer refused",
    "download at an offset past the buffer refused",
    "download at an unaligned offset refused",
    "download at an offset of a part word refused",
];

#[test]
fn interpreter_and_device_agree() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_gpu_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> = v["expected"]
        .as_array()
        .expect("expected")
        .iter()
        .map(|b| b.as_u64().expect("byte") as u8)
        .collect();
    let a = Artifact::from_bytes(lean_artifacts::HPROG_GPU_CORPUS).expect("artifact shape");

    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &[], &mut out).expect("execute");

    let mut bad = Vec::new();
    for (k, what) in WHAT.iter().enumerate() {
        let (got, want) = (&out[4 * k..4 * k + 4], &expected[4 * k..4 * k + 4]);
        if got != want {
            bad.push(format!(
                "  {what}: lean {} != device {}",
                i32::from_le_bytes(want.try_into().unwrap()),
                i32::from_le_bytes(got.try_into().unwrap())
            ));
        }
    }
    for (lo, hi, what) in [
        (128, 192, "download after one dispatch"),
        (192, 256, "download after dispatch, then upload (the upload lands first)"),
        (256, 272, "download at an offset"),
    ] {
        if out[lo..hi] != expected[lo..hi] {
            bad.push(format!("  {what}: lean {:?} != device {:?}", &expected[lo..hi], &out[lo..hi]));
        }
    }
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
    assert_eq!(out, expected, "every byte of the output buffer");
    eprintln!("{} wgpu contract checks agree", WHAT.len() + 3);
}
