//! Runs the device corpus: what `Sem.callFfi` says the CUDA entry points do,
//! against what they do on the GPU.
//!
//! One body calls each contracted CUDA and cuBLAS entry point on its success and
//! refusal paths and leaves every result code and every byte it copied back in
//! the output buffer. The interpreter computed that buffer when the artifact
//! was generated (`lean/algorithms/Host/CudaCorpus.lean`); here the same
//! artifact runs through the JIT on the device, and the two must agree byte for
//! byte. The one kernel it launches adds one to each byte, which is also the
//! interpreter's kernel oracle, and the cuBLAS calls see small integers only,
//! so their answer does not depend on summation order; a disagreement is the
//! contract's, not the kernel's or the vendor's.

use base_types::Artifact;

use std::path::PathBuf;

/// Result codes first (`i32`s from byte 0), downloaded bytes from byte 128.
const WHAT: [&str; 32] = [
    "create 64",
    "create 0 refused",
    "create 16",
    "upload",
    "upload to a buffer that does not exist",
    "upload at an offset",
    "upload past the end refused",
    "launch",
    "download",
    "download at an offset",
    "download past the end refused",
    "free",
    "free twice refused",
    "download from a freed buffer refused",
    "launch binding a freed buffer refused",
    "sync",
    "sgemv",
    "sgemv on a buffer that does not exist refused",
    "download sgemv's y",
    "strided-batched sgemm",
    "strided-batched sgemm with m = 0 refused",
    "download sgemm's C",
    "launch by entry-point name",
    "bf16 gemmEx",
    "download gemmEx's C",
    "download the named launch's buffer",
    "pinned alloc",
    "upload from pinned memory",
    "launch on the pinned data",
    "download into pinned memory",
    "pinned free",
    "pinned free twice refused",
];

/// More result codes, from byte 480: the stream variants, the pointer-array
/// batch and the events.
const WHAT2: [&str; 11] = [
    "record an event on a created stream",
    "sgemv on the stream",
    "named launch on the stream",
    "fill a pointer array",
    "pointer array entry past its buffer refused",
    "pointer-array sgemm batch on the stream",
    "stream sync",
    "elapsed time between two waited-for events answered",
    "elapsed time to a never-recorded event refused",
    "strided-batched bf16 gemmEx",
    "strided-batched bf16 gemmEx with no batches refused",
];

/// And from byte 800: the asynchronous copies.
const WHAT3: [&str; 9] = [
    "async upload from pinned memory",
    "async upload at an offset from pageable memory",
    "named launch on the stream",
    "async download into pinned memory",
    "async download into pageable memory",
    "stream sync",
    "async upload past the buffer refused",
    "async download on a stream that does not exist refused",
    "async download into the upload's source, on the same stream",
];

#[test]
fn interpreter_and_device_agree() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_cuda_corpus/expected.json")).expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> = v["expected"]
        .as_array()
        .expect("expected")
        .iter()
        .map(|b| b.as_u64().expect("byte") as u8)
        .collect();
    let a = Artifact::from_bytes(lean_artifacts::HPROG_CUDA_CORPUS).expect("artifact shape");

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
    for (k, what) in WHAT2.iter().enumerate() {
        let at = 480 + 4 * k;
        let (got, want) = (&out[at..at + 4], &expected[at..at + 4]);
        if got != want {
            bad.push(format!(
                "  {what}: lean {} != device {}",
                i32::from_le_bytes(want.try_into().unwrap()),
                i32::from_le_bytes(got.try_into().unwrap())
            ));
        }
    }
    for (k, what) in WHAT3.iter().enumerate() {
        let at = 800 + 4 * k;
        let (got, want) = (&out[at..at + 4], &expected[at..at + 4]);
        if got != want {
            bad.push(format!(
                "  {what}: lean {} != device {}",
                i32::from_le_bytes(want.try_into().unwrap()),
                i32::from_le_bytes(got.try_into().unwrap())
            ));
        }
    }
    for (lo, hi, what) in [
        (128, 192, "downloaded buffer after the launch"),
        (192, 208, "downloaded offset buffer"),
        (256, 272, "sgemv's y"),
        (272, 304, "sgemm's C, both batches"),
        (304, 320, "bf16 gemmEx's C"),
        (320, 384, "buffer after the named launch"),
        (384, 448, "pinned memory after upload, launch, download"),
        (448, 456, "pinned pointer past the end refused"),
        (456, 464, "pinned pointer after free refused"),
        (464, 472, "device reports some memory"),
        (528, 544, "sgemv's y, on the stream"),
        (544, 608, "buffer after the named launch on the stream"),
        (608, 640, "both members of the pointer-array batch"),
        (640, 672, "strided-batched bf16 gemmEx's C, both batches"),
        (672, 736, "pinned memory after the async round trip"),
        (736, 800, "pageable memory after the async download"),
        (840, 848, "the upload's source, read while in flight"),
        (848, 912, "the upload's source after the download into it"),
    ] {
        if out[lo..hi] != expected[lo..hi] {
            bad.push(format!("  {what}: lean {:?} != device {:?}", &expected[lo..hi], &out[lo..hi]));
        }
    }
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
    assert_eq!(out, expected, "every byte of the output buffer");
    eprintln!("{} device contract checks agree", WHAT.len() + WHAT2.len() + WHAT3.len() + 18);
}
