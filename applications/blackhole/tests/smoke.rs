use std::fs;
use std::process::Command;

fn get_blackhole_binary() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    format!("{}/../../target/{}/blackhole", manifest_dir, profile)
}

fn assert_bmp_header(path: &std::path::Path, expected_width: i32, expected_height: i32) {
    let data = fs::read(path).unwrap_or_else(|e| panic!("Failed to read {:?}: {}", path, e));
    assert!(data.len() > 54, "BMP too small: {}", data.len());
    assert_eq!(&data[0..2], b"BM");
    assert_eq!(
        i32::from_le_bytes(data[18..22].try_into().unwrap()),
        expected_width
    );
    assert_eq!(
        i32::from_le_bytes(data[22..26].try_into().unwrap()),
        expected_height
    );
}

#[test]
fn test_blackhole_smoke_writes_bmp() {
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");
    let binary = get_blackhole_binary();
    let output = Command::new(&binary)
        .current_dir(tmpdir.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", binary, e));

    assert!(
        output.status.success(),
        "blackhole failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bmp = tmpdir.path().join("blackhole.bmp");
    assert_bmp_header(&bmp, 1280, -720);
    let len = fs::metadata(&bmp).expect("missing blackhole.bmp").len();
    assert!(len > 1_000_000, "unexpected BMP size: {}", len);
}
