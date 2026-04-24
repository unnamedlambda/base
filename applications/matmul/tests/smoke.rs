use std::fs;
use std::process::Command;

fn get_matmul_binary() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    format!("{}/../../target/{}/matmul", manifest_dir, profile)
}

#[test]
fn test_matmul_smoke_writes_nontrivial_output() {
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");
    let binary = get_matmul_binary();
    let output = Command::new(&binary)
        .current_dir(tmpdir.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", binary, e));

    assert!(
        output.status.success(),
        "matmul failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let path = tmpdir.path().join("matmul_output.bin");
    let data = fs::read(&path).expect("missing matmul_output.bin");
    assert_eq!(data.len(), 1_048_576, "unexpected output size");

    let first: Vec<f32> = data[..32]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert!(first.iter().all(|v| v.is_finite()), "non-finite output values");
    assert!(
        first.iter().any(|v| *v != 0.0),
        "first output values were all zero"
    );
}
