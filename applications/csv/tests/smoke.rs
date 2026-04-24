use std::fs;
use std::process::Command;

fn get_csv_binary() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    format!("{}/../../target/{}/csv", manifest_dir, profile)
}

fn copy_fixture_tree(root: &std::path::Path) {
    let data_dir = root.join("applications/csv/data");
    fs::create_dir_all(&data_dir).expect("failed to create csv data dir");

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("data");
    for name in ["employees.csv", "departments.csv"] {
        fs::copy(src_dir.join(name), data_dir.join(name))
            .unwrap_or_else(|e| panic!("failed to copy {}: {}", name, e));
    }
}

#[test]
fn test_csv_smoke_outputs_expected_files() {
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");
    copy_fixture_tree(tmpdir.path());

    let binary = get_csv_binary();
    let output = Command::new(&binary)
        .current_dir(tmpdir.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", binary, e));

    assert!(
        output.status.success(),
        "csv failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let scan = fs::read_to_string(tmpdir.path().join("scan.csv")).expect("missing scan.csv");
    let filter = fs::read_to_string(tmpdir.path().join("filter.csv")).expect("missing filter.csv");
    let join = fs::read_to_string(tmpdir.path().join("join.csv")).expect("missing join.csv");

    assert!(scan.starts_with("id,name,age,city,dept_id,salary\n"));
    assert!(scan.contains("1,Alice,30,Seattle,1,85000\n"));
    assert!(filter.starts_with("id,name,age,city,dept_id,salary\n"));
    assert!(filter.contains("10,Jack,41,Seattle,3,105000\n"));
    assert_eq!(join, "dept_id,dept_name,floor\n1,Engineering,3\n2,Marketing,5\n3,Sales,2\n");
}
