use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Output;

use base_types::{Algorithm, BaseConfig};

pub fn build(algorithms_dir: &str, lean_file: &'static str) {
    build_all(algorithms_dir, &[(lean_file, "algorithm")]);
}

pub fn build_all(algorithms_dir: &str, artifacts: &[(&'static str, &'static str)]) {
    let manifest_dir = manifest_dir();
    let lake_path = manifest_dir.join(algorithms_dir);
    let out_dir = out_dir();

    let lib_lean = lake_path.parent().unwrap().join("lib/AlgorithmLib.lean");
    println!("cargo:rerun-if-changed={}", lib_lean.display());

    build_algorithm_lib(&lake_path);

    for &(lean_file, output_name) in artifacts {
        println!("cargo:rerun-if-changed={}", lake_path.join(lean_file).display());
        let output = run_lean_interpreter(&lake_path, lean_file);
        write_algorithm_outputs(&out_dir, output_name, &output.stdout);
    }
}

fn build_algorithm_lib(algorithms_dir: &Path) {
    // AlgorithmLib lives in ../lib relative to the algorithms package.
    let lib_dir = algorithms_dir.parent().unwrap().join("lib");
    let olean = lib_dir.join(".lake/build/lib/lean/AlgorithmLib.olean");
    if olean.exists() {
        return;
    }
    let output = Command::new("lake")
        .args(["build"])
        .current_dir(&lib_dir)
        .output()
        .unwrap_or_else(|e| panic!("Failed to build AlgorithmLib: {e}"));
    ensure_success(output, "AlgorithmLib build failed", "AlgorithmLib build failed");
}

fn manifest_dir() -> PathBuf {
    env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .expect("CARGO_MANIFEST_DIR is not set")
}

fn out_dir() -> PathBuf {
    env::var("OUT_DIR")
        .map(PathBuf::from)
        .expect("OUT_DIR is not set")
}

fn run_lean_interpreter(lake_dir: &Path, lean_file: &str) -> Output {
    let lean_path = lake_dir.join(lean_file);
    let output = Command::new("lake")
        .args(["env", "lean", "--run"])
        .arg(&lean_path)
        .current_dir(lake_dir)
        .output()
        .unwrap_or_else(|error| panic!("Failed to run Lean interpreter for {lean_file}: {error}"));

    ensure_success(
        output,
        &format!("Lean Interpretation Failed ({lean_file})"),
        &format!("Lean code generation failed for {lean_file}"),
    )
}

fn ensure_success(output: Output, header: &str, panic_message: &str) -> Output {
    if output.status.success() {
        return output;
    }

    eprintln!("=== {header} ===");
    eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    panic!("{panic_message}");
}

fn write_algorithm_outputs(out_dir: &Path, output_name: &str, stdout: &[u8]) {
    let raw_output = String::from_utf8(stdout.to_vec()).expect("Lean output was not valid UTF-8");
    let json_str = extract_json(&raw_output);

    fs::write(out_dir.join(format!("{output_name}.json")), json_str)
        .expect("Failed to write debug JSON");

    let pair: (BaseConfig, Algorithm) = serde_json::from_str(json_str).unwrap_or_else(|error| {
        panic!(
            "BUILD FAILED: {output_name} JSON does not match Rust (BaseConfig, Algorithm) structure: {error}"
        )
    });

    let binary =
        bincode::serialize(&pair).expect("Failed to serialize (BaseConfig, Algorithm) to bincode");

    fs::write(out_dir.join(format!("{output_name}.bin")), binary)
        .expect("Failed to write binary algorithm");
}

fn extract_json(output: &str) -> &str {
    match output.find("\n[") {
        Some(pos) => &output[pos + 1..],
        None => output.trim_start(),
    }
}
