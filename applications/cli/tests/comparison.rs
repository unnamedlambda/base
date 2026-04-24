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
        .trim_end()
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
fn test_scalar_arithmetic_and_precedence() {
    assert_cli("1+2\n", "3");
    assert_cli("1+2+3\n", "6");
    assert_cli("2*3*4\n", "24");
    assert_cli("2+3*4\n", "14");
    assert_cli("2*3+4\n", "10");
    assert_cli("-5\n", "-5");
    assert_cli("-5+2\n", "-3");
}

#[test]
fn test_scalar_variables_are_stateful_within_one_session() {
    assert_cli("x=5\nx\nx+2\ny=3\nx+y\n", "5\n7\n8");
}

#[test]
fn test_undefined_and_multi_character_variables_are_weird_but_stable() {
    assert_cli("u\nu+1\nu*2\n", "0\n1\n0");
    assert_cli("foo\nf\n", "0\n0");
    assert_cli("abc=9\na\nabc\n", "0\n0\n0");
}

#[test]
fn test_assignment_lines_produce_no_output() {
    assert_cli("x=5\n", "");
    assert_cli("x=[1,2,3]\n", "");
}

#[test]
fn test_array_literals_and_spacing() {
    assert_cli("[1,2,3]\n", "[1,2,3]");
    assert_cli("[ 1, 2 , 3 ]\n", "[ 1, 2 , 3 ]");
    assert_cli("[-1,2,-3]\n", "[-1,2,-3]");
    assert_cli("[[1,2],[3,4]]\n", "[[1,2],[3,4]]");
}

#[test]
fn test_invalid_or_partial_array_inputs_fail_silently_except_star() {
    assert_cli("[1,[2,3]]\n", "");
    assert_cli("[1,2,3]+\n", "");
    assert_cli("[1,2,3]+[4,5]\n", "");
    assert_cli("[1,2]+[4,5,6]\n", "");
    assert_cli("[1,2,3]*\n", "[0,0,0]");
}

#[test]
fn test_array_literal_operations() {
    assert_cli("[1,2,3]+[4,5,6]\n", "[5,7,9]");
    assert_cli("[1,2,3]*2\n", "[2,4,6]");
    assert_cli("2*[1,2,3]\n", "[2,4,6]");
    assert_cli(
        "[[1,2],[3,4]]+[[5,6],[7,8]]\n[[1,2],[3,4]]*2\n2*[[1,2],[3,4]]\n",
        "[[6,8],[10,12]]\n[[2,4],[6,8]]\n[[2,4],[6,8]]",
    );
}

#[test]
fn test_array_variable_retrieval_copy_and_scaling() {
    assert_cli(
        "x=[1,2,3]\nx\nx*2\n2*x\ny=x\ny\nz=x*3\nz\n",
        "[1,2,3]\n[2,4,6]\n[2,4,6]\n[1,2,3]\n[3,6,9]",
    );
    assert_cli(
        "x=[[1,2],[3,4]]\nx\ny=2*x\ny\nz=x*2\nz\n",
        "[[1,2],[3,4]]\n[[2,4],[6,8]]\n[[2,4],[6,8]]",
    );
}

#[test]
fn test_array_variable_addition_is_asymmetric() {
    assert_cli("x=[1,2,3]\ny=[4,5,6]\nx+y\ny+x\n", "[1,2,3]\n[4,5,6]");
    assert_cli("y=[4,5,6]\n[1,2,3]+y\n", "");
    assert_cli("x=[1,2,3]\nx+[4,5,6]\n", "[1,2,3]");
}

#[test]
fn test_array_assignments_from_expressions_follow_current_parser_rules() {
    assert_cli(
        "a=[1,2,3]+[4,5,6]\na\nb=[1,2,3]*2\nb\nc=2*[1,2,3]\nc\n",
        "[5,7,9]\n[2,4,6]\n[2,4,6]",
    );
    assert_cli("a=[1,2,3]\nb=a+[4,5,6]\nb\n", "[1,2,3]");
    assert_cli("a=[1,2,3]\nc=[4,5,6]+a\nc\n", "0");
}

#[test]
fn test_blank_lines_and_whitespace_only_lines_print_nothing() {
    assert_cli(" \n\n", "");
}
