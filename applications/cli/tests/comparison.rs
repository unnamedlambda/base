use std::io::Write;
use std::process::{Command, Stdio};

fn get_cli_binary() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    format!("{}/../../target/{}/cli", manifest_dir, profile)
}

fn run_cli(input: &str) -> String {
    let binary = get_cli_binary();
    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", binary, e));

    child
        .stdin
        .as_mut()
        .expect("Failed to open stdin")
        .write_all(input.as_bytes())
        .expect("Failed to write stdin");

    let output = child.wait_with_output().expect("Failed to wait on cli");

    if !output.status.success() {
        panic!(
            "cli failed for input {:?}: {}",
            input,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    String::from_utf8(output.stdout)
        .expect("stdout was not valid UTF-8")
        .trim()
        .to_string()
}

fn assert_cli(input: &str, expected: &str) {
    let got = run_cli(input);
    assert_eq!(
        got, expected,
        "CLI output mismatch for input {:?}: got {:?}, expected {:?}",
        input, got, expected
    );
}

#[test]
fn test_scalar_expression() {
    assert_cli("1+2\n", "3");
}

#[test]
fn test_precedence() {
    assert_cli("2*3+4\n", "10");
}

#[test]
fn test_assignment_then_expression() {
    assert_cli("x=5\nx+2\n", "7");
}

#[test]
fn test_assignment_only_produces_no_output() {
    assert_cli("x=5\n", "");
}

#[test]
fn test_array_literal_normalization() {
    assert_cli("[1,2,3]\n", "[1,2,3]");
}

#[test]
fn test_array_scale() {
    assert_cli("x=[1,2,3]\nx*2\n", "[2,4,6]");
}
