use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

fn get_lean4_eval_binary() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    format!("{}/../../target/{}/lean4-eval", manifest_dir, profile)
}

fn get_temp_files() -> (String, String) {
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    (
        format!("/tmp/lean4_eval_test_{}.lean", id),
        format!("/tmp/out_{}.txt", id),
    )
}

fn run_lean(code: &str, test_file: &str) -> String {
    fs::write(test_file, code).expect("Failed to write test file");

    let output = Command::new("lean")
        .args(&["--run", test_file])
        .output()
        .expect("Failed to run lean");

    // lean --run prints to stdout, but may have stderr warnings
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn run_ours(code: &str, test_file: &str, output_file: &str) -> String {
    fs::write(test_file, code).expect("Failed to write test file");

    // Remove output file before running
    let _ = fs::remove_file(output_file);

    let binary = get_lean4_eval_binary();
    let _ = Command::new(&binary)
        .arg(test_file)
        .arg(output_file)
        .output()
        .expect(&format!("Failed to run {}", binary));

    // Read result from output file
    fs::read_to_string(output_file)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn compare(code: &str) {
    let (test_file, output_file) = get_temp_files();
    let lean_output = run_lean(code, &test_file);
    let our_output = run_ours(code, &test_file, &output_file);
    fs::remove_file(&test_file).ok();
    fs::remove_file(&output_file).ok();

    assert_eq!(
        lean_output, our_output,
        "\nCode: {}\nLean output: '{}'\nOur output: '{}'",
        code, lean_output, our_output
    );
}

#[test]
fn test_eval_nat_literal_42() {
    compare("#eval 42");
}

#[test]
fn test_eval_nat_literal_0() {
    compare("#eval 0");
}

#[test]
fn test_eval_nat_literal_1() {
    compare("#eval 1");
}

#[test]
fn test_eval_nat_literal_large() {
    compare("#eval 12345");
}

#[test]
fn test_eval_nat_literal_with_whitespace() {
    compare("#eval   42");
}

#[test]
fn test_eval_nat_literal_with_newline() {
    compare("#eval 42\n");
}
