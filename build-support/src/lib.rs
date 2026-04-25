use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Output;

use base_types::{Algorithm, BaseConfig};

#[derive(Clone, Copy, Debug)]
pub struct AlgorithmArtifact {
    pub lean_file: &'static str,
    pub output_name: &'static str,
}

pub fn rerun_if_changed(paths: &[&str]) {
    for path in paths {
        println!("cargo:rerun-if-changed={path}");
    }
}

pub fn generate_algorithms(lake_dir: &str, artifacts: &[AlgorithmArtifact]) {
    let manifest_dir = manifest_dir();
    let lake_path = manifest_dir.join(lake_dir);
    let out_dir = out_dir();

    build_algorithm_lib(&lake_path);

    for artifact in artifacts {
        let output = run_lean_interpreter(&lake_path, artifact.lean_file);
        write_algorithm_outputs(&out_dir, artifact.output_name, &output.stdout);
    }
}

fn build_algorithm_lib(lake_dir: &Path) {
    let olean = lake_dir.join(".lake/build/lib/lean/AlgorithmLib.olean");
    if olean.exists() {
        return;
    }
    let output = Command::new("lake")
        .args(["build", "AlgorithmLib"])
        .current_dir(lake_dir)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run lake build AlgorithmLib: {e}"));
    ensure_success(output, "lake build AlgorithmLib failed", "AlgorithmLib build failed");
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
