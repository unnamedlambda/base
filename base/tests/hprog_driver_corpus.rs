//! The CUDA driver, called directly, checked against the device.
//!
//! `lean/algorithms/Host/DriverCorpus.lean` makes the driver calls a program
//! makes — context, allocation, copies, a fill, modules, launches on the default
//! stream and a created one, events, page-locked memory, copies on a stream,
//! cuBLAS, frees — storing every result code and every byte
//! it copies back. This runs the artifact on the GPU and compares with what the
//! interpreter computed. Handles and device pointers differ between the two and
//! are never stored: codes and bytes are what a program can observe.

use base_types::Artifact;
use std::path::PathBuf;

#[test]
fn interpreter_and_driver_agree() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_driver_corpus/expected.json"))
            .expect("read corpus"),
    )
    .expect("corpus shape");
    let expected: Vec<u8> = v["expected"]
        .as_array()
        .expect("expected")
        .iter()
        .map(|b| b.as_u64().expect("byte") as u8)
        .collect();
    let names: Vec<String> = v["names"]
        .as_array()
        .expect("names")
        .iter()
        .map(|n| n.as_str().expect("name").to_string())
        .collect();
    let data = v["data"].as_u64().expect("data") as usize;
    let a = Artifact::from_bytes(
        &std::fs::read(dir.join("hprog_driver_corpus.cbor")).expect("read artifact"),
    )
    .expect("artifact shape");

    let mut b = base::Driver::load(a).expect("compile");
    let mut out = vec![0u8; expected.len()];
    b.execute("main", &[], &mut out).expect("execute");

    let mut bad = Vec::new();
    for (k, name) in names.iter().enumerate() {
        let (got, want) = (&out[4 * k..4 * k + 4], &expected[4 * k..4 * k + 4]);
        if got != want {
            bad.push(format!(
                "  {name}: lean {} != device {}",
                i32::from_le_bytes(want.try_into().unwrap()),
                i32::from_le_bytes(got.try_into().unwrap())
            ));
        }
    }
    if out[data..] != expected[data..] {
        bad.push(format!(
            "  downloaded bytes differ:\n    lean   {:?}\n    device {:?}",
            &expected[data..],
            &out[data..]
        ));
    }
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
    eprintln!("{} driver results and {} bytes agree", names.len(), out.len() - data);
}

/// The signatures the program calls the driver and cuBLAS with are the
/// headers': the generator writes one `static_assert` per function, over the
/// parameter and result widths `Ext.sig` gives, and this compiles it against
/// `cuda.h` and `cublas_v2.h`.
#[test]
fn signatures_match_cuda_h() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(lean_artifacts::DIR),
    };
    let include = std::env::var("CUDA_PATH")
        .map(|p| PathBuf::from(p).join("include"))
        .into_iter()
        .chain(["/usr/local/cuda/include", "/usr/include"].map(PathBuf::from))
        .find(|d| d.join("cuda.h").exists())
        .expect("no cuda.h: set CUDA_PATH to the toolkit");
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    let out = std::process::Command::new(&cxx)
        .args(["-std=c++17", "-fsyntax-only", "-I"])
        .arg(&include)
        .arg(dir.join("hprog_driver_corpus/abi_check.cpp"))
        .output()
        .expect("run the C++ compiler");
    assert!(
        out.status.success(),
        "signatures disagree with {}:\n{}",
        include.join("cuda.h").display(),
        String::from_utf8_lossy(&out.stderr)
    );
}
