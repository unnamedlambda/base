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

fn generate_algorithm(manifest_dir: &str, lean_file: &Path, output_name: &str) {
    // Use Lean interpreter directly — skips C compilation entirely.
    let output = Command::new("lake")
        .args(&["env", "lean", "--run"])
        .arg(lean_file)
        .current_dir(manifest_dir)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run Lean interpreter for {:?}: {}", lean_file, e));

    if !output.status.success() {
        eprintln!("=== Lean Interpretation Failed ({:?}) ===", lean_file);
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Lean code generation failed for {:?}", lean_file);
    }

    let raw_output = String::from_utf8(output.stdout)
        .expect("Lean output was not valid UTF-8");
    let json_str = extract_json(&raw_output);

    let out_dir = env::var("OUT_DIR").unwrap();

    fs::write(format!("{}/{}.json", out_dir, output_name), json_str)
        .expect("Failed to write debug JSON");

    let pair: (BaseConfig, Algorithm) = serde_json::from_str(json_str)
        .unwrap_or_else(|e| panic!("BUILD FAILED: {} JSON does not match Rust (BaseConfig, Algorithm) structure: {}", output_name, e));

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
    println!("cargo:rerun-if-changed=lean/SaxpyBenchAlgorithm.lean");
    println!("cargo:rerun-if-changed=lakefile.lean");
    println!("cargo:rerun-if-changed=../lean/AlgorithmLib.lean");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lean_dir = Path::new(&manifest_dir).join("lean");

    generate_algorithm(&manifest_dir, &lean_dir.join("CsvBenchAlgorithm.lean"), "csv_algorithm");
    generate_algorithm(&manifest_dir, &lean_dir.join("RegexBenchAlgorithm.lean"), "regex_algorithm");
    generate_algorithm(&manifest_dir, &lean_dir.join("JsonBenchAlgorithm.lean"), "json_algorithm");
    generate_algorithm(&manifest_dir, &lean_dir.join("StringSearchAlgorithm.lean"), "strsearch_algorithm");
    generate_algorithm(&manifest_dir, &lean_dir.join("WordCountAlgorithm.lean"), "wc_algorithm");
    generate_algorithm(&manifest_dir, &lean_dir.join("SaxpyBenchAlgorithm.lean"), "saxpy_algorithm");
}
