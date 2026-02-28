use std::fs;
use std::process::Command;

fn get_sat_binary() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    format!("{}/../../target/{}/sat", manifest_dir, profile)
}

fn get_cnf_dir() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    format!("{}/tests/cnf", manifest_dir)
}

/// Run our SAT solver on a CNF file, return (status_line, assignment_line_or_empty)
fn run_base_sat(cnf_path: &str) -> (String, String) {
    let binary = get_sat_binary();

    // Use a unique temp directory so parallel tests don't share sat_output.txt
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");

    let output = Command::new(&binary)
        .arg(cnf_path)
        .current_dir(tmpdir.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", binary, e));

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() && stdout.is_empty() {
        panic!("SAT solver failed on {}: {}", cnf_path, stderr);
    }

    let mut status = String::new();
    let mut assignment = String::new();
    for line in stdout.lines() {
        if line.starts_with("s ") {
            status = line.to_string();
        } else if line.starts_with("v ") {
            assignment = line.to_string();
        }
    }
    (status, assignment)
}

/// Run minisat on a CNF file, return "SAT" or "UNSAT"
fn run_minisat(cnf_path: &str) -> Option<String> {
    let output = Command::new("minisat")
        .args(&[cnf_path, "/dev/null"])
        .output()
        .ok()?;

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    if combined.contains("UNSATISFIABLE") {
        Some("UNSAT".to_string())
    } else if combined.contains("SATISFIABLE") {
        Some("SAT".to_string())
    } else {
        None
    }
}

/// Parse a DIMACS CNF file, return (num_vars, clauses)
fn parse_cnf(path: &str) -> (usize, Vec<Vec<i32>>) {
    let content = fs::read_to_string(path).unwrap();
    let mut num_vars = 0;
    let mut clauses = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }
        if line.starts_with('p') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            num_vars = parts[2].parse().unwrap();
            continue;
        }
        let lits: Vec<i32> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<i32>().ok())
            .filter(|&x| x != 0)
            .collect();
        if !lits.is_empty() {
            clauses.push(lits);
        }
    }
    (num_vars, clauses)
}

/// Verify that an assignment satisfies all clauses
fn verify_assignment(clauses: &[Vec<i32>], assignment: &[i32]) -> Result<(), String> {
    // Build a set of true literals
    let mut true_lits = std::collections::HashSet::new();
    for &lit in assignment {
        if lit != 0 {
            true_lits.insert(lit);
        }
    }

    for (i, clause) in clauses.iter().enumerate() {
        let satisfied = clause.iter().any(|lit| true_lits.contains(lit));
        if !satisfied {
            return Err(format!(
                "Clause {} not satisfied: {:?}, assignment: {:?}",
                i, clause, assignment
            ));
        }
    }
    Ok(())
}

/// Parse the "v" line into a list of literals
fn parse_assignment(v_line: &str) -> Vec<i32> {
    v_line
        .strip_prefix("v ")
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|s| s.parse::<i32>().ok())
        .filter(|&x| x != 0)
        .collect()
}

fn test_cnf(filename: &str, expected_sat: bool) {
    let cnf_path = format!("{}/{}", get_cnf_dir(), filename);
    let (status, assignment) = run_base_sat(&cnf_path);

    if expected_sat {
        assert_eq!(
            status, "s SATISFIABLE",
            "Expected SAT for {}, got: {}",
            filename, status
        );
        // Verify the assignment
        let (_num_vars, clauses) = parse_cnf(&cnf_path);
        let lits = parse_assignment(&assignment);
        assert!(!lits.is_empty(), "SAT result for {} has no assignment", filename);
        verify_assignment(&clauses, &lits)
            .unwrap_or_else(|e| panic!("Invalid assignment for {}: {}", filename, e));
    } else {
        assert_eq!(
            status, "s UNSATISFIABLE",
            "Expected UNSAT for {}, got: {}",
            filename, status
        );
    }
}

fn test_cnf_against_minisat(filename: &str) {
    let cnf_path = format!("{}/{}", get_cnf_dir(), filename);

    let minisat_result = run_minisat(&cnf_path)
        .expect("minisat not found — install minisat to run comparison tests");

    let (status, assignment) = run_base_sat(&cnf_path);

    let our_result = if status.contains("UNSATISFIABLE") {
        "UNSAT"
    } else if status.contains("SATISFIABLE") {
        "SAT"
    } else {
        panic!("Unknown result for {}: {}", filename, status);
    };

    assert_eq!(
        our_result, minisat_result,
        "Mismatch on {}: ours={}, minisat={}",
        filename, our_result, minisat_result
    );

    // For SAT results, verify assignment independently
    if our_result == "SAT" {
        let (_num_vars, clauses) = parse_cnf(&cnf_path);
        let lits = parse_assignment(&assignment);
        verify_assignment(&clauses, &lits)
            .unwrap_or_else(|e| panic!("Invalid assignment for {}: {}", filename, e));
    }
}

// ===================================================================
// SAT tests
// ===================================================================

#[test]
fn test_sat_trivial() {
    test_cnf("sat_trivial.cnf", true);
}

#[test]
fn test_sat_simple() {
    test_cnf("sat_simple.cnf", true);
}

#[test]
fn test_sat_3vars() {
    test_cnf("sat_3vars.cnf", true);
}

#[test]
fn test_sat_unit_propagation() {
    test_cnf("sat_unit.cnf", true);
}

#[test]
fn test_sat_medium() {
    test_cnf("sat_medium.cnf", true);
}

#[test]
fn test_sat_backtrack_deep() {
    test_cnf("sat_backtrack_deep.cnf", true);
}

#[test]
fn test_sat_backtrack_10vars() {
    test_cnf("sat_backtrack_10vars.cnf", true);
}

#[test]
fn test_sat_wide_clauses() {
    test_cnf("sat_wide_clauses.cnf", true);
}

#[test]
fn test_sat_20vars() {
    test_cnf("sat_20vars.cnf", true);
}

#[test]
fn test_sat_50vars() {
    test_cnf("sat_50vars.cnf", true);
}

#[test]
fn test_sat_pure_literal() {
    test_cnf("sat_pure_literal.cnf", true);
}

#[test]
fn test_sat_unit_chain_long() {
    test_cnf("sat_unit_chain_long.cnf", true);
}

#[test]
fn test_sat_tautological() {
    test_cnf("sat_tautological.cnf", true);
}

#[test]
fn test_sat_mixed_widths() {
    test_cnf("sat_mixed_widths.cnf", true);
}

#[test]
fn test_sat_unique() {
    test_cnf("sat_unique.cnf", true);
}

// ===================================================================
// UNSAT tests
// ===================================================================

#[test]
fn test_unsat_trivial() {
    test_cnf("unsat_trivial.cnf", false);
}

#[test]
fn test_unsat_3vars() {
    test_cnf("unsat_3vars.cnf", false);
}

#[test]
fn test_unsat_pigeonhole_3_2() {
    test_cnf("unsat_pigeon_3_2.cnf", false);
}

#[test]
fn test_unsat_pigeonhole_4_3() {
    test_cnf("unsat_pigeon_4_3.cnf", false);
}

#[test]
fn test_unsat_pigeonhole_5_4() {
    test_cnf("unsat_pigeon_5_4.cnf", false);
}

#[test]
fn test_unsat_unit_chain() {
    test_cnf("unsat_unit_chain.cnf", false);
}

#[test]
fn test_unsat_empty_clause() {
    test_cnf("unsat_empty_clause.cnf", false);
}

#[test]
fn test_unsat_triangle_coloring() {
    test_cnf("unsat_triangle_coloring.cnf", false);
}

#[test]
fn test_unsat_resolution() {
    test_cnf("unsat_resolution.cnf", false);
}

#[test]
fn test_unsat_php_extended() {
    test_cnf("unsat_php_extended.cnf", false);
}

// ===================================================================
// Minisat comparison tests
// ===================================================================

#[test]
fn test_vs_minisat_sat_trivial() {
    test_cnf_against_minisat("sat_trivial.cnf");
}

#[test]
fn test_vs_minisat_sat_simple() {
    test_cnf_against_minisat("sat_simple.cnf");
}

#[test]
fn test_vs_minisat_unsat_trivial() {
    test_cnf_against_minisat("unsat_trivial.cnf");
}

#[test]
fn test_vs_minisat_unsat_3vars() {
    test_cnf_against_minisat("unsat_3vars.cnf");
}

#[test]
fn test_vs_minisat_pigeon_3_2() {
    test_cnf_against_minisat("unsat_pigeon_3_2.cnf");
}

#[test]
fn test_vs_minisat_sat_medium() {
    test_cnf_against_minisat("sat_medium.cnf");
}

#[test]
fn test_vs_minisat_sat_20vars() {
    test_cnf_against_minisat("sat_20vars.cnf");
}

#[test]
fn test_vs_minisat_sat_50vars() {
    test_cnf_against_minisat("sat_50vars.cnf");
}

#[test]
fn test_vs_minisat_unsat_pigeon_4_3() {
    test_cnf_against_minisat("unsat_pigeon_4_3.cnf");
}

#[test]
fn test_vs_minisat_unsat_pigeon_5_4() {
    test_cnf_against_minisat("unsat_pigeon_5_4.cnf");
}
