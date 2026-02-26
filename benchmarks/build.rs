use std::env;
use std::fs;
use std::process::Command;
use base_types::{BaseConfig, Algorithm};

fn generate_algorithm(manifest_dir: &str, exe_name: &str, output_name: &str) {
    let output = Command::new("lake")
        .args(&["exe", exe_name])
        .current_dir(manifest_dir)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run lake exe {}: {}", exe_name, e));

    if !output.status.success() {
        eprintln!("=== {} Generation Failed ===", exe_name);
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("{} code generation failed", exe_name);
    }

    let json_str = String::from_utf8(output.stdout)
        .expect("Lean output was not valid UTF-8");

    let out_dir = env::var("OUT_DIR").unwrap();

    fs::write(format!("{}/{}.json", out_dir, output_name), &json_str)
        .expect("Failed to write debug JSON");

    let pair: (BaseConfig, Algorithm) = serde_json::from_str(&json_str)
        .unwrap_or_else(|e| panic!("BUILD FAILED: {} JSON does not match Rust (BaseConfig, Algorithm) structure: {}", exe_name, e));

    let binary = bincode::serialize(&pair)
        .expect("Failed to serialize (BaseConfig, Algorithm) to bincode");

    fs::write(format!("{}/{}.bin", out_dir, output_name), binary)
        .expect("Failed to write binary algorithm");
}

fn main() {
    println!("cargo:rerun-if-changed=lean/CsvBenchAlgorithm.lean");
    println!("cargo:rerun-if-changed=lean/RegexBenchAlgorithm.lean");
    println!("cargo:rerun-if-changed=lean/JsonBenchAlgorithm.lean");
    println!("cargo:rerun-if-changed=lean/StringSearchAlgorithm.lean");
    println!("cargo:rerun-if-changed=lean/WordCountAlgorithm.lean");
    println!("cargo:rerun-if-changed=lakefile.lean");
    println!("cargo:rerun-if-changed=../lean/AlgorithmLib.lean");

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

    generate_algorithm(&manifest_dir, "generate_csv", "csv_algorithm");
    generate_algorithm(&manifest_dir, "generate_regex", "regex_algorithm");
    generate_algorithm(&manifest_dir, "generate_json", "json_algorithm");
    generate_algorithm(&manifest_dir, "generate_string_search", "strsearch_algorithm");
    generate_algorithm(&manifest_dir, "generate_wordcount", "wc_algorithm");
}
