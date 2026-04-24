use std::io::Write;
use std::process::{Command, Stdio};

fn get_cli_binary() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
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

    String::from_utf8(output.stdout).expect("stdout was not valid UTF-8")
}

fn assert_cli(input: &str, expected: &str) {
    let got = run_cli(input);
    assert_eq!(
        got, expected,
        "CLI output mismatch for input {:?}: got {:?}, expected {:?}",
        input, got, expected
    );
}

fn assert_cli_transcript(input: &str, line_outputs: &[Option<&str>]) {
    let input_lines = input.lines().count();
    assert_eq!(
        input_lines,
        line_outputs.len(),
        "each input line must have a corresponding expected output slot"
    );

    let mut expected = String::new();
    for output in line_outputs {
        expected.push_str("> ");
        if let Some(text) = output {
            expected.push_str(text);
            expected.push('\n');
        }
    }
    expected.push_str("> ");
    assert_cli(input, &expected);
}

#[test]
fn test_scalar_arithmetic_and_precedence() {
    assert_cli_transcript("1+2\n", &[Some("3")]);
    assert_cli_transcript("1+2+3\n", &[Some("6")]);
    assert_cli_transcript("2*3*4\n", &[Some("24")]);
    assert_cli_transcript("2+3*4\n", &[Some("14")]);
    assert_cli_transcript("2*3+4\n", &[Some("10")]);
    assert_cli_transcript("-5\n", &[Some("-5")]);
    assert_cli_transcript("-5+2\n", &[Some("-3")]);
}

#[test]
fn test_scalar_variables_are_stateful_within_one_session() {
    assert_cli_transcript(
        "x=5\nx\nx+2\ny=3\nx+y\n",
        &[None, Some("5"), Some("7"), None, Some("8")],
    );
}

#[test]
fn test_undefined_and_multi_character_variables_are_weird_but_stable() {
    assert_cli_transcript("u\nu+1\nu*2\n", &[Some("0"), Some("1"), Some("0")]);
    assert_cli_transcript("foo\nf\n", &[Some("0"), Some("0")]);
    assert_cli_transcript("abc=9\na\nabc\n", &[Some("0"), Some("0"), Some("0")]);
}

#[test]
fn test_assignment_lines_produce_no_output() {
    assert_cli_transcript("x=5\n", &[None]);
    assert_cli_transcript("x=[1,2,3]\n", &[None]);
}

#[test]
fn test_array_literals_and_spacing() {
    assert_cli_transcript("[1,2,3]\n", &[Some("[1,2,3]")]);
    assert_cli_transcript("[ 1, 2 , 3 ]\n", &[Some("[ 1, 2 , 3 ]")]);
    assert_cli_transcript("[-1,2,-3]\n", &[Some("[-1,2,-3]")]);
    assert_cli_transcript("[[1,2],[3,4]]\n", &[Some("[[1,2],[3,4]]")]);
}

#[test]
fn test_invalid_or_partial_array_inputs_fail_silently_except_star() {
    assert_cli_transcript("[1,[2,3]]\n", &[None]);
    assert_cli_transcript("[1,2,3]+\n", &[None]);
    assert_cli_transcript("[1,2,3]+[4,5]\n", &[None]);
    assert_cli_transcript("[1,2]+[4,5,6]\n", &[None]);
    assert_cli_transcript("[1,2,3]*\n", &[Some("[0,0,0]")]);
}

#[test]
fn test_array_literal_operations() {
    assert_cli_transcript("[1,2,3]+[4,5,6]\n", &[Some("[5,7,9]")]);
    assert_cli_transcript("[1,2,3]*2\n", &[Some("[2,4,6]")]);
    assert_cli_transcript("2*[1,2,3]\n", &[Some("[2,4,6]")]);
    assert_cli_transcript(
        "[[1,2],[3,4]]+[[5,6],[7,8]]\n[[1,2],[3,4]]*2\n2*[[1,2],[3,4]]\n",
        &[
            Some("[[6,8],[10,12]]"),
            Some("[[2,4],[6,8]]"),
            Some("[[2,4],[6,8]]"),
        ],
    );
}

#[test]
fn test_array_variable_retrieval_copy_and_scaling() {
    assert_cli_transcript(
        "x=[1,2,3]\nx\nx*2\n2*x\ny=x\ny\nz=x*3\nz\n",
        &[
            None,
            Some("[1,2,3]"),
            Some("[2,4,6]"),
            Some("[2,4,6]"),
            None,
            Some("[1,2,3]"),
            None,
            Some("[3,6,9]"),
        ],
    );
    assert_cli_transcript(
        "x=[[1,2],[3,4]]\nx\ny=2*x\ny\nz=x*2\nz\n",
        &[
            None,
            Some("[[1,2],[3,4]]"),
            None,
            Some("[[2,4],[6,8]]"),
            None,
            Some("[[2,4],[6,8]]"),
        ],
    );
}

#[test]
fn test_array_variable_addition_is_asymmetric() {
    assert_cli_transcript(
        "x=[1,2,3]\ny=[4,5,6]\nx+y\ny+x\n",
        &[None, None, Some("[1,2,3]"), Some("[4,5,6]")],
    );
    assert_cli_transcript("y=[4,5,6]\n[1,2,3]+y\n", &[None, None]);
    assert_cli_transcript("x=[1,2,3]\nx+[4,5,6]\n", &[None, Some("[1,2,3]")]);
}

#[test]
fn test_array_assignments_from_expressions_follow_current_parser_rules() {
    assert_cli_transcript(
        "a=[1,2,3]+[4,5,6]\na\nb=[1,2,3]*2\nb\nc=2*[1,2,3]\nc\n",
        &[None, Some("[5,7,9]"), None, Some("[2,4,6]"), None, Some("[2,4,6]")],
    );
    assert_cli_transcript("a=[1,2,3]\nb=a+[4,5,6]\nb\n", &[None, None, Some("[1,2,3]")]);
    assert_cli_transcript("a=[1,2,3]\nc=[4,5,6]+a\nc\n", &[None, None, Some("0")]);
}

#[test]
fn test_blank_lines_and_whitespace_only_lines_print_nothing() {
    assert_cli_transcript(" \n\n", &[None, None]);
}
