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

#[test]
fn test_eval_addition_0_plus_0() {
    compare("#eval 0 + 0");
}

#[test]
fn test_eval_addition_1_plus_1() {
    compare("#eval 1 + 1");
}

#[test]
fn test_eval_addition_2_plus_3() {
    compare("#eval 2 + 3");
}

#[test]
fn test_eval_addition_5_plus_4() {
    compare("#eval 5 + 4");
}

#[test]
fn test_eval_addition_9_plus_0() {
    compare("#eval 9 + 0");
}

#[test]
fn test_eval_addition_0_plus_9() {
    compare("#eval 0 + 9");
}

#[test]
fn test_eval_addition_9_plus_9() {
    compare("#eval 9 + 9");
}

#[test]
fn test_eval_addition_7_plus_8() {
    compare("#eval 7 + 8");
}

#[test]
fn test_eval_addition_10_plus_20() {
    compare("#eval 10 + 20");
}

#[test]
fn test_eval_addition_99_plus_1() {
    compare("#eval 99 + 1");
}

#[test]
fn test_eval_addition_123_plus_456() {
    compare("#eval 123 + 456");
}

#[test]
fn test_eval_addition_large() {
    compare("#eval 999999999 + 1");
}

#[test]
fn test_eval_addition_three_terms() {
    compare("#eval 1 + 2 + 3");
}

#[test]
fn test_eval_addition_many_terms() {
    compare("#eval 10 + 20 + 30 + 40 + 50");
}

#[test]
fn test_eval_addition_with_spaces() {
    compare("#eval   7   +   8 +  9");
}

#[test]
fn test_eval_lambda_single_arg() {
    compare("#eval (fun x => x + 1) 5");
}

#[test]
fn test_eval_lambda_addend_7() {
    compare("#eval (fun x => x + 7) 5");
}

#[test]
fn test_eval_lambda_addend_20() {
    compare("#eval (fun x => x + 20) 10");
}

#[test]
fn test_eval_mul_2_times_3() {
    compare("#eval 2 * 3");
}

#[test]
fn test_eval_mul_0_times_5() {
    compare("#eval 0 * 5");
}

#[test]
fn test_eval_mul_5_times_0() {
    compare("#eval 5 * 0");
}

#[test]
fn test_eval_mul_1_times_1() {
    compare("#eval 1 * 1");
}

#[test]
fn test_eval_mul_large() {
    compare("#eval 123 * 456");
}

#[test]
fn test_eval_mul_chained() {
    compare("#eval 2 * 3 * 4");
}

#[test]
fn test_eval_mul_chained_many() {
    compare("#eval 2 * 3 * 4 * 5");
}

#[test]
fn test_eval_precedence_add_mul() {
    compare("#eval 2 + 3 * 4");
}

#[test]
fn test_eval_precedence_mul_add() {
    compare("#eval 2 * 3 + 4");
}

#[test]
fn test_eval_precedence_complex() {
    compare("#eval 1 + 2 * 3 + 4");
}

#[test]
fn test_eval_precedence_chain() {
    compare("#eval 2 + 3 * 4 * 5 + 6");
}

#[test]
fn test_eval_precedence_mul_chains() {
    compare("#eval 1 * 2 + 3 * 4");
}

#[test]
fn test_eval_lambda_mul_simple() {
    compare("#eval (fun x => x * 2) 5");
}

#[test]
fn test_eval_lambda_mul_by_zero() {
    compare("#eval (fun x => x * 0) 42");
}

#[test]
fn test_eval_lambda_mul_large() {
    compare("#eval (fun x => x * 100) 50");
}

#[test]
fn test_eval_mul_with_spaces() {
    compare("#eval  2  *  3  *  4");
}

#[test]
fn test_eval_sub_5_minus_3() {
    compare("#eval 5 - 3");
}

#[test]
fn test_eval_sub_10_minus_0() {
    compare("#eval 10 - 0");
}

#[test]
fn test_eval_sub_0_minus_0() {
    compare("#eval 0 - 0");
}

#[test]
fn test_eval_sub_large() {
    compare("#eval 1000 - 999");
}

#[test]
fn test_eval_sub_chained() {
    compare("#eval 10 - 3 - 2");
}

#[test]
fn test_eval_sub_chained_many() {
    compare("#eval 100 - 10 - 20 - 30");
}

#[test]
fn test_eval_add_sub_mixed() {
    compare("#eval 10 + 5 - 3");
}

#[test]
fn test_eval_sub_add_mixed() {
    compare("#eval 10 - 5 + 3");
}

#[test]
fn test_eval_sub_mul_precedence() {
    compare("#eval 10 - 2 * 3");
}

#[test]
fn test_eval_mul_sub_precedence() {
    compare("#eval 2 * 3 - 4");
}

#[test]
fn test_eval_complex_precedence() {
    compare("#eval 20 - 2 * 3 + 4");
}

#[test]
fn test_eval_lambda_sub_simple() {
    compare("#eval (fun x => x - 2) 10");
}

#[test]
fn test_eval_lambda_sub_to_zero() {
    compare("#eval (fun x => x - 5) 5");
}

#[test]
fn test_eval_sub_with_spaces() {
    compare("#eval  10  -  3  -  2");
}

#[test]
fn test_eval_let_add() {
    compare("#eval let x := 5; x + 3");
}

#[test]
fn test_eval_let_mul() {
    compare("#eval let x := 10; x * 2");
}

#[test]
fn test_eval_let_sub() {
    compare("#eval let x := 10; x - 3");
}

#[test]
fn test_eval_let_zero() {
    compare("#eval let x := 0; x + 5");
}

#[test]
fn test_eval_let_identity() {
    compare("#eval let x := 42; x + 0");
}

#[test]
fn test_eval_let_mul_zero() {
    compare("#eval let x := 5; x * 0");
}

#[test]
fn test_eval_let_large() {
    compare("#eval let x := 100; x * 100");
}

#[test]
fn test_eval_paren_simple() {
    compare("#eval (5)");
}

#[test]
fn test_eval_paren_add() {
    compare("#eval (2 + 3)");
}

#[test]
fn test_eval_paren_mul_right() {
    compare("#eval 2 * (3 + 4)");
}

#[test]
fn test_eval_paren_mul_left() {
    compare("#eval (2 + 3) * 4");
}

#[test]
fn test_eval_paren_both() {
    compare("#eval (1 + 2) * (3 + 4)");
}

#[test]
fn test_eval_paren_override_precedence() {
    compare("#eval (2 + 3) * 4");
}

#[test]
fn test_eval_paren_sub() {
    compare("#eval (10 - 3) * 2");
}

#[test]
fn test_eval_paren_complex() {
    compare("#eval 1 + (2 * 3) + 4");
}

#[test]
fn test_eval_paren_sub_mul() {
    compare("#eval (5 - 2) * (4 + 1)");
}
