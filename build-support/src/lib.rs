use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Output;

use base_types::{Algorithm, BaseConfig};

pub fn build(algorithm_file: &Path, watch_dir: &Path) {
    build_all(&[algorithm_file.to_path_buf()], watch_dir);
}

pub fn build_all(algorithm_files: &[PathBuf], watch_dir: &Path) {
    let out_dir = out_dir();

    println!("cargo:rerun-if-changed={}", watch_dir.display());

    let modules: Vec<String> = algorithm_files
        .iter()
        .map(|f| {
            f.file_stem()
                .unwrap_or_else(|| panic!("No file stem for {}", f.display()))
                .to_str()
                .expect("Non-UTF8 file stem")
                .to_string()
        })
        .collect();

    let lake_dir = algorithm_files[0]
        .parent()
        .unwrap_or_else(|| panic!("Algorithm file has no parent directory"));

    build_modules(lake_dir, &modules.iter().map(|s| s.as_str()).collect::<Vec<_>>());

    for file in algorithm_files {
        let output = run_lean_interpreter(lake_dir, file);
        write_artifacts(&out_dir, file, &output.stdout);
    }
}

fn build_modules(algorithms_dir: &Path, modules: &[&str]) {
    let output = Command::new("lake")
        .arg("build")
        .args(modules)
        .current_dir(algorithms_dir)
        .output()
        .unwrap_or_else(|e| panic!("Failed to build modules: {e}"));
    ensure_success(output, "Module build failed", "Module build failed");
}

fn run_lean_interpreter(lake_dir: &Path, lean_file: &Path) -> Output {
    let output = Command::new("lake")
        .args(["env", "lean", "--run"])
        .arg(lean_file)
        .current_dir(lake_dir)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run Lean interpreter for {}: {e}", lean_file.display()));

    ensure_success(
        output,
        &format!("Lean Interpretation Failed ({})", lean_file.display()),
        &format!("Lean code generation failed for {}", lean_file.display()),
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

fn write_artifacts(out_dir: &Path, lean_file: &Path, stdout: &[u8]) {
    let raw = String::from_utf8(stdout.to_vec()).expect("Lean output was not valid UTF-8");
    let json_str = extract_json(&raw);

    let entries: Vec<(String, BaseConfig, Algorithm)> = serde_json::from_str(json_str)
        .unwrap_or_else(|e| {
            panic!(
                "BUILD FAILED: {} output does not match Vec<(String, BaseConfig, Algorithm)>: {e}",
                lean_file.display()
            )
        });

    for (name, config, algorithm) in entries {
        fs::write(
            out_dir.join(format!("{name}.json")),
            serde_json::to_string(&(&config, &algorithm)).expect("Failed to serialize JSON"),
        )
        .expect("Failed to write JSON artifact");

        let binary = bincode::serialize(&(config, algorithm)).expect("Failed to serialize bincode");
        fs::write(out_dir.join(format!("{name}.bin")), binary)
            .expect("Failed to write binary artifact");
    }
}

fn extract_json(output: &str) -> &str {
    match output.find("\n[") {
        Some(pos) => &output[pos + 1..],
        None => output.trim_start(),
    }
}

fn out_dir() -> PathBuf {
    env::var("OUT_DIR")
        .map(PathBuf::from)
        .expect("OUT_DIR is not set")
}
