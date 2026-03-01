use std::fs;
use std::process::Command;

fn get_sha256_binary() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    format!("{}/../../target/{}/sha256", manifest_dir, profile)
}

/// Run our SHA-256 binary on a file, return the hex digest (trimmed).
fn run_base_sha256(input_path: &str) -> String {
    let binary = get_sha256_binary();
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");

    let output = Command::new(&binary)
        .arg(input_path)
        .current_dir(tmpdir.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", binary, e));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("sha256 binary failed on {}: {}", input_path, stderr);
    }

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

/// Run system sha256sum on a file, return the hex digest.
fn run_system_sha256sum(input_path: &str) -> String {
    let output = Command::new("sha256sum")
        .arg(input_path)
        .output()
        .expect("sha256sum not found — install coreutils to run comparison tests");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("sha256sum failed on {}: {}", input_path, stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // sha256sum output format: "<hash>  <filename>"
    stdout.split_whitespace().next()
        .expect("sha256sum produced no output")
        .to_string()
}

fn test_sha256(data: &[u8], name: &str) {
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");
    let input_path = tmpdir.path().join("input.bin");
    fs::write(&input_path, data).unwrap();

    let input_str = input_path.to_str().unwrap();
    let base_hash = run_base_sha256(input_str);
    let system_hash = run_system_sha256sum(input_str);

    assert_eq!(
        base_hash, system_hash,
        "{}: base produced '{}', sha256sum produced '{}'",
        name, base_hash, system_hash
    );
}

fn test_sha256_known(data: &[u8], expected: &str, name: &str) {
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");
    let input_path = tmpdir.path().join("input.bin");
    fs::write(&input_path, data).unwrap();

    let base_hash = run_base_sha256(input_path.to_str().unwrap());

    assert_eq!(
        base_hash, expected,
        "{}: base produced '{}', expected '{}'",
        name, base_hash, expected
    );
}

// --- NIST test vectors ---

#[test]
fn test_empty() {
    test_sha256_known(
        b"",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "empty",
    );
}

#[test]
fn test_abc() {
    test_sha256_known(
        b"abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "abc",
    );
}

#[test]
fn test_nist_two_block() {
    test_sha256_known(
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "nist_two_block",
    );
}

// --- Boundary tests ---

#[test]
fn test_55_bytes() {
    // 55 + 1 (0x80) + 8 (length) = 64 — fits in one block
    let data = vec![b'a'; 55];
    test_sha256(&data, "55_bytes");
}

#[test]
fn test_56_bytes() {
    // 56 + 1 (0x80) = 57 > 56, needs two blocks
    let data = vec![b'a'; 56];
    test_sha256(&data, "56_bytes");
}

#[test]
fn test_64_bytes() {
    // Exact one data block
    let data = vec![b'a'; 64];
    test_sha256(&data, "64_bytes");
}

#[test]
fn test_one_byte() {
    test_sha256(&[0x00], "one_byte_null");
}

#[test]
fn test_all_byte_values() {
    let data: Vec<u8> = (0..=255).collect();
    test_sha256(&data, "all_byte_values");
}

#[test]
fn test_1kb() {
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    test_sha256(&data, "1kb");
}

#[test]
fn test_large() {
    // 100KB pseudo-random
    let mut data = vec![0u8; 102400];
    let mut state: u32 = 0xDEADBEEF;
    for byte in data.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *byte = (state >> 16) as u8;
    }
    test_sha256(&data, "large_100kb");
}

// --- Additional NIST / well-known vectors ---

#[test]
fn test_nist_long_message() {
    // "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
    // repeated 1,000 times (64,000 bytes) — this is a standard NIST vector
    // but it's very large so we compare against sha256sum instead of hardcoding
    let block = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno";
    let data: Vec<u8> = block.iter().cloned().cycle().take(64 * 1000).collect();
    test_sha256(&data, "nist_long_1000_blocks");
}

// --- More padding boundary tests ---

#[test]
fn test_54_bytes() {
    // 54 + 1 + 8 = 63 < 64 — still fits in one block with 1 byte slack
    let data = vec![b'z'; 54];
    test_sha256(&data, "54_bytes");
}

#[test]
fn test_57_bytes() {
    // Just past the boundary — needs 2 blocks
    let data = vec![b'x'; 57];
    test_sha256(&data, "57_bytes");
}

#[test]
fn test_63_bytes() {
    // 63 + 1 = 64 bytes used, but need 8 more for length → 2 blocks
    let data = vec![b'w'; 63];
    test_sha256(&data, "63_bytes");
}

#[test]
fn test_119_bytes() {
    // 119 + 1 + 8 = 128 → exactly 2 blocks
    let data = vec![b'm'; 119];
    test_sha256(&data, "119_bytes");
}

#[test]
fn test_120_bytes() {
    // 120 + 1 + 8 = 129 > 128 → needs 3 blocks
    let data = vec![b'n'; 120];
    test_sha256(&data, "120_bytes");
}

#[test]
fn test_128_bytes() {
    // Exactly 2 data blocks
    let data = vec![b'q'; 128];
    test_sha256(&data, "128_bytes");
}

// --- Data content stress tests ---

#[test]
fn test_all_ff() {
    // All 0xFF — stresses carry propagation in SHA-256 arithmetic
    let data = vec![0xFF; 256];
    test_sha256(&data, "all_ff_256");
}

#[test]
fn test_all_zeros() {
    let data = vec![0x00; 512];
    test_sha256(&data, "all_zeros_512");
}

#[test]
fn test_single_bit() {
    // Only the high bit set in one byte
    test_sha256(&[0x80], "single_0x80_byte");
}

#[test]
fn test_alternating_bits() {
    // Alternating 0xAA and 0x55 — exercises XOR/AND paths
    let data: Vec<u8> = (0..200).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect();
    test_sha256(&data, "alternating_aa_55");
}

#[test]
fn test_sequential_u32() {
    // Each 4 bytes is a sequential big-endian u32 — tests W loading
    let mut data = Vec::new();
    for i in 0u32..64 {
        data.extend_from_slice(&i.to_be_bytes());
    }
    test_sha256(&data, "sequential_u32");
}

#[test]
fn test_1mb() {
    // 1MB pseudo-random — stress many blocks (16384 blocks)
    let mut data = vec![0u8; 1048576];
    let mut state: u32 = 0xCAFEBABE;
    for byte in data.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *byte = (state >> 16) as u8;
    }
    test_sha256(&data, "large_1mb");
}
