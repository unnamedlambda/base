use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Output;

use base_types::{Algorithm, BaseConfig};

pub fn build(lean_file: &Path, watch_dir: &Path) {
    build_all(&[lean_file.to_path_buf()], watch_dir);
}

/// Build and run Lean generators that belong to the same Lake project directory.
///
/// Every file in `lean_files` must live under the same parent directory, which is
/// used as the working directory for `lake build` and `lake env lean --run`.
pub fn build_all(lean_files: &[PathBuf], watch_dir: &Path) {
    assert!(
        !lean_files.is_empty(),
        "build_all requires at least one Lean file"
    );

    let modules: Vec<String> = lean_files
        .iter()
        .map(|lean_file| {
            lean_file
                .file_stem()
                .unwrap_or_else(|| panic!("No file stem for {}", lean_file.display()))
                .to_str()
                .expect("Non-UTF8 file stem")
                .to_string()
        })
        .collect();

    let lake_dir = lean_files[0]
        .parent()
        .unwrap_or_else(|| panic!("Lean file has no parent directory"));

    for lean_file in &lean_files[1..] {
        let lean_file_parent = lean_file
            .parent()
            .unwrap_or_else(|| panic!("Lean file has no parent directory"));
        assert_eq!(
            lean_file_parent, lake_dir,
            "All Lean files passed to build_all must share the same parent directory: expected {}, got {} for {}",
            lake_dir.display(),
            lean_file_parent.display(),
            lean_file.display(),
        );
    }

    let out_dir = out_dir();

    println!("cargo:rerun-if-changed={}", watch_dir.display());

    build_modules(
        lake_dir,
        &modules.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );

    for lean_file in lean_files {
        let generator_out_dir = generator_output_dir(&out_dir, lean_file);
        recreate_dir(&generator_out_dir);
        run_lean_interpreter(lake_dir, lean_file, &generator_out_dir);
        write_binaries(lean_file, &generator_out_dir);
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

fn run_lean_interpreter(lake_dir: &Path, lean_file: &Path, output_dir: &Path) {
    let output = Command::new("lake")
        .args(["env", "lean", "--run"])
        .arg(lean_file)
        .arg(output_dir)
        .current_dir(lake_dir)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to run Lean interpreter for {}: {e}",
                lean_file.display()
            )
        });

    ensure_success(
        output,
        &format!("Lean Interpretation Failed ({})", lean_file.display()),
        &format!("Lean code generation failed for {}", lean_file.display()),
    );
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

fn write_binaries(lean_file: &Path, generator_out_dir: &Path) {
    for (artifact_name, config, algorithm) in read_generated_artifacts(lean_file, generator_out_dir)
    {
        let binary = bincode::serialize(&(config, algorithm)).expect("Failed to serialize bincode");
        fs::write(
            generator_out_dir.join(format!("{artifact_name}.bin")),
            binary,
        )
        .expect("Failed to write binary artifact");
    }
}

fn out_dir() -> PathBuf {
    env::var("OUT_DIR")
        .map(PathBuf::from)
        .expect("OUT_DIR is not set")
}

fn generator_output_dir(out_dir: &Path, lean_file: &Path) -> PathBuf {
    out_dir.join(lean_file_stem(lean_file))
}

fn lean_file_stem(lean_file: &Path) -> &str {
    lean_file
        .file_stem()
        .unwrap_or_else(|| panic!("No file stem for {}", lean_file.display()))
        .to_str()
        .expect("Non-UTF8 file stem")
}

fn recreate_dir(path: &Path) {
    match fs::remove_dir_all(path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => panic!("Failed to remove {}: {err}", path.display()),
    }
    fs::create_dir_all(path).expect("Failed to create output directory");
}

fn read_generated_artifacts(
    lean_file: &Path,
    generator_out_dir: &Path,
) -> Vec<(String, BaseConfig, Algorithm)> {
    let mut json_paths: Vec<PathBuf> = fs::read_dir(generator_out_dir)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", generator_out_dir.display()))
        .map(|entry| entry.expect("Failed to read directory entry").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect();
    json_paths.sort();

    let mut entries = Vec::with_capacity(json_paths.len());
    for path in json_paths {
        let artifact_name = path
            .file_stem()
            .unwrap_or_else(|| panic!("No file stem for {}", path.display()))
            .to_str()
            .expect("Non-UTF8 file stem")
            .to_string();
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
        let (config, algorithm): (BaseConfig, Algorithm) = serde_json::from_str(&text)
            .unwrap_or_else(|e| {
                panic!(
                    "BUILD FAILED: {} artifact {} does not match (BaseConfig, Algorithm): {e}",
                    lean_file.display(),
                    path.display()
                )
            });
        entries.push((artifact_name, config, algorithm));
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use base_types::{Action, Kind, RuntimeHeader};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_entry(name: &str) -> (String, BaseConfig, Algorithm) {
        (
            name.to_string(),
            BaseConfig {
                cranelift_ir: "function u0:0() { return }".to_string(),
                memory_size: 64,
                runtime_header: RuntimeHeader {
                    data_ptr_offset: 0x18,
                    data_len_offset: 0x20,
                    out_ptr_offset: 0x28,
                    out_len_offset: 0x30,
                },
                context_offset: 16,
                initial_memory: vec![1, 2, 3],
            },
            Algorithm {
                actions: vec![Action {
                    kind: Kind::ClifCall,
                    dst: 0,
                    src: 1,
                    offset: 2,
                    size: 3,
                }],
                cranelift_units: 1,
                timeout_ms: Some(123),
                output: vec![],
            },
        )
    }

    fn temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let dir = env::temp_dir().join(format!(
            "build-support-tests-{}-{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_artifact(path: &Path, config: &BaseConfig, algorithm: &Algorithm) {
        let json = serde_json::to_string(&(config, algorithm)).expect("serialize artifact");
        fs::write(path, json).expect("write artifact");
    }

    #[test]
    #[should_panic(expected = "build_all requires at least one Lean file")]
    fn build_all_rejects_empty_file_list() {
        build_all(&[], Path::new("/tmp"));
    }

    #[test]
    #[should_panic(expected = "All Lean files passed to build_all must share the same parent directory")]
    fn build_all_rejects_mixed_parent_directories() {
        let lean_files = vec![
            PathBuf::from("/tmp/project_a/Foo.lean"),
            PathBuf::from("/tmp/project_b/Bar.lean"),
        ];
        build_all(&lean_files, Path::new("/tmp"));
    }

    #[test]
    fn recreate_dir_removes_existing_files() {
        let out_dir = temp_dir();
        let subdir = out_dir.join("subdir");
        fs::create_dir_all(&subdir).expect("create subdir");
        fs::write(subdir.join("stale.json"), "stale").expect("write stale file");

        recreate_dir(&subdir);

        assert!(subdir.exists());
        assert!(!subdir.join("stale.json").exists());
        fs::remove_dir_all(out_dir).expect("cleanup temp dir");
    }

    #[test]
    fn read_generated_artifacts_reads_named_json_files() {
        let out_dir = temp_dir();
        let lean_file = Path::new("CsvBenchAlgorithm.lean");
        let generator_out_dir = generator_output_dir(&out_dir, lean_file);
        let entry = sample_entry("csv_algorithm");
        fs::create_dir_all(&generator_out_dir).expect("create generator out dir");
        write_artifact(
            &generator_out_dir.join("csv_algorithm.json"),
            &entry.1,
            &entry.2,
        );

        let entries = read_generated_artifacts(lean_file, &generator_out_dir);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "csv_algorithm");

        fs::remove_dir_all(out_dir).expect("cleanup temp dir");
    }

    #[test]
    fn write_binaries_creates_bin_alongside_json() {
        let out_dir = temp_dir();
        let lean_file = Path::new("CsvBenchAlgorithm.lean");
        let generator_out_dir = generator_output_dir(&out_dir, lean_file);
        let entry = sample_entry("csv_algorithm");
        fs::create_dir_all(&generator_out_dir).expect("create generator out dir");
        write_artifact(
            &generator_out_dir.join("csv_algorithm.json"),
            &entry.1,
            &entry.2,
        );

        write_binaries(lean_file, &generator_out_dir);

        assert!(generator_out_dir.join("csv_algorithm.json").exists());
        assert!(generator_out_dir.join("csv_algorithm.bin").exists());

        fs::remove_dir_all(out_dir).expect("cleanup temp dir");
    }
}
