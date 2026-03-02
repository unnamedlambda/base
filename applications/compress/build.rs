use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;
use base_types::{BaseConfig, Algorithm};

/// Extract JSON from Lean interpreter output, skipping any warning lines.
fn extract_json(output: &str) -> &str {
    // Our JSON always starts with '[' (toJsonPair produces an array)
    match output.find("\n[") {
        Some(pos) => &output[pos + 1..],
        None => output.trim_start(),
    }
}

fn main() {
    println!("cargo:rerun-if-changed=lean/MakeAlgorithm.lean");
    println!("cargo:rerun-if-changed=lakefile.lean");
    println!("cargo:rerun-if-changed=../../lean/AlgorithmLib.lean");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lean_file = Path::new(&manifest_dir).join("lean/MakeAlgorithm.lean");

    // Use Lean interpreter directly — skips C compilation entirely.
    let output = Command::new("lake")
        .args(&["env", "lean", "--run"])
        .arg(&lean_file)
        .current_dir(&manifest_dir)
        .output()
        .expect("Failed to run Lean interpreter");

    if !output.status.success() {
        eprintln!("=== Lean Interpretation Failed ===");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Lean code generation failed");
    }

    let raw_output = String::from_utf8(output.stdout)
        .expect("Lean output was not valid UTF-8");
    let json_str = extract_json(&raw_output);

    let out_dir = env::var("OUT_DIR").unwrap();

    fs::write(format!("{}/algorithm.json", out_dir), json_str)
        .expect("Failed to write debug JSON");

    let pair: (BaseConfig, Algorithm) = serde_json::from_str(json_str)
        .expect("BUILD FAILED: Lean JSON does not match Rust (BaseConfig, Algorithm) structure");

    let binary = bincode::serialize(&pair)
        .expect("Failed to serialize (BaseConfig, Algorithm) to bincode");

    fs::write(format!("{}/algorithm.bin", out_dir), binary)
        .expect("Failed to write binary algorithm");
}
