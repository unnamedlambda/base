use std::env;
use std::fs;
use std::process::Command;
use base_types::Algorithm;

fn main() {
    println!("cargo:rerun-if-changed=lean/MakeAlgorithm.lean");
    println!("cargo:rerun-if-changed=lakefile.lean");
    println!("cargo:rerun-if-changed=../../lean/AlgorithmLib.lean");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let lake_build = Command::new("lake")
        .arg("build")
        .current_dir(&manifest_dir)
        .output()
        .expect("Failed to run lake build");

    if !lake_build.status.success() {
        eprintln!("Lake build failed:");
        eprintln!("stdout: {}", String::from_utf8_lossy(&lake_build.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&lake_build.stderr));
        panic!("Lake build failed");
    }

    let output = Command::new("lake")
        .args(&["exe", "generate"])
        .current_dir(&manifest_dir)
        .output()
        .expect("Failed to run Lean generator");

    if !output.status.success() {
        eprintln!("=== Lean Generation Failed ===");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Lean code generation failed");
    }

    let json_str = String::from_utf8(output.stdout)
        .expect("Lean output was not valid UTF-8");

    let out_dir = env::var("OUT_DIR").unwrap();

    fs::write(format!("{}/algorithm.json", out_dir), &json_str)
        .expect("Failed to write debug JSON");

    let algorithm: Algorithm = serde_json::from_str(&json_str)
        .expect("BUILD FAILED: Lean JSON does not match Rust Algorithm structure");

    let binary = bincode::serialize(&algorithm)
        .expect("Failed to serialize algorithm to bincode");

    fs::write(format!("{}/algorithm.bin", out_dir), binary)
        .expect("Failed to write binary algorithm");
}
