use std::fs;
use std::process::Command;

fn get_compress_binary() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    format!("{}/../../target/{}/compress", manifest_dir, profile)
}

/// Compress a file using our binary, returns path to the .lz4 output
fn run_compress(input_path: &str, work_dir: &std::path::Path) -> String {
    let binary = get_compress_binary();
    let output = Command::new(&binary)
        .arg(input_path)
        .current_dir(work_dir)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", binary, e));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Compress failed on {}: {}", input_path, stderr);
    }

    work_dir.join("compress_output.lz4").to_string_lossy().to_string()
}

/// Decompress using system lz4 tool, returns decompressed bytes
fn run_lz4_decompress(lz4_path: &str, output_path: &str) -> Vec<u8> {
    let output = Command::new("lz4")
        .args(&["-d", "-f", lz4_path, output_path])
        .output()
        .expect("lz4 not found — install lz4 to run roundtrip tests");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("lz4 decompress failed on {}: {}", lz4_path, stderr);
    }

    fs::read(output_path).unwrap_or_else(|e| panic!("Failed to read {}: {}", output_path, e))
}

fn roundtrip_test(input_data: &[u8], name: &str) {
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");

    // Write input file
    let input_path = tmpdir.path().join("input.bin");
    fs::write(&input_path, input_data).unwrap();

    // Compress
    let lz4_path = run_compress(
        input_path.to_str().unwrap(),
        tmpdir.path(),
    );

    // Verify the output is a valid LZ4 frame (starts with magic number)
    let lz4_data = fs::read(&lz4_path).unwrap();
    assert!(
        lz4_data.len() >= 11,
        "{}: output too small ({} bytes)",
        name,
        lz4_data.len()
    );
    assert_eq!(
        &lz4_data[0..4],
        &[0x04, 0x22, 0x4D, 0x18],
        "{}: invalid LZ4 magic number",
        name
    );

    // Decompress with lz4
    let decompressed_path = tmpdir.path().join("decompressed.bin");
    let decompressed = run_lz4_decompress(
        &lz4_path,
        decompressed_path.to_str().unwrap(),
    );

    // Verify identity
    assert_eq!(
        decompressed.len(),
        input_data.len(),
        "{}: decompressed size mismatch (got {}, expected {})",
        name,
        decompressed.len(),
        input_data.len()
    );
    assert_eq!(
        decompressed, input_data,
        "{}: decompressed data does not match original",
        name
    );
}

#[test]
fn test_roundtrip_small_text() {
    roundtrip_test(b"Hello, World! This is a test of LZ4 compression.\n", "small_text");
}

#[test]
fn test_roundtrip_repeated_data() {
    // Highly compressible: 10KB of repeated pattern
    let pattern = b"ABCDEFGHIJ";
    let data: Vec<u8> = pattern.iter().cycle().take(10240).copied().collect();
    roundtrip_test(&data, "repeated_data");
}

#[test]
fn test_roundtrip_sequential_bytes() {
    // 1KB of sequential bytes (0, 1, 2, ..., 255, 0, 1, ...)
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    roundtrip_test(&data, "sequential_bytes");
}

#[test]
fn test_roundtrip_larger_file() {
    // ~100KB of pseudo-random-ish data (deterministic, low compressibility)
    let mut data = vec![0u8; 102400];
    let mut state: u32 = 0xDEADBEEF;
    for byte in data.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *byte = (state >> 16) as u8;
    }
    roundtrip_test(&data, "larger_file");
}

#[test]
fn test_roundtrip_multi_block() {
    // >16KB to force multiple blocks (block size = 16384)
    let data: Vec<u8> = (0..32768).map(|i| (i % 251) as u8).collect();
    roundtrip_test(&data, "multi_block");
}

#[test]
fn test_roundtrip_single_byte() {
    roundtrip_test(&[42], "single_byte");
}

#[test]
fn test_roundtrip_exactly_5_bytes() {
    // Exactly the LZ4 LASTLITERALS boundary
    roundtrip_test(b"ABCDE", "exactly_5_bytes");
}

#[test]
fn test_roundtrip_exactly_6_bytes() {
    // One byte beyond the LASTLITERALS boundary
    roundtrip_test(b"ABCDEF", "exactly_6_bytes");
}

#[test]
fn test_roundtrip_all_zeros() {
    // Maximally compressible: 16KB of zeros
    let data = vec![0u8; 16384];
    roundtrip_test(&data, "all_zeros");
}

#[test]
fn test_roundtrip_all_same_byte() {
    // 20KB of 0xFF — crosses block boundary with max compression
    let data = vec![0xFFu8; 20480];
    roundtrip_test(&data, "all_same_byte");
}

#[test]
fn test_roundtrip_exact_block_boundary() {
    // Exactly 16384 bytes = one full block, no second block
    let data: Vec<u8> = (0..16384).map(|i| (i % 256) as u8).collect();
    roundtrip_test(&data, "exact_block_boundary");
}

#[test]
fn test_roundtrip_block_boundary_plus_one() {
    // 16385 bytes = one full block + 1 byte in second block
    let data: Vec<u8> = (0..16385).map(|i| (i % 256) as u8).collect();
    roundtrip_test(&data, "block_boundary_plus_one");
}

#[test]
fn test_roundtrip_three_blocks() {
    // 48KB = exactly 3 blocks of repeated data
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let data: Vec<u8> = pattern.iter().cycle().take(49152).copied().collect();
    roundtrip_test(&data, "three_blocks");
}

#[test]
fn test_roundtrip_long_match_run() {
    // Pattern that produces very long matches (near MAX_MATCH_LEN=255)
    // 4-byte pattern repeated thousands of times
    let data: Vec<u8> = b"WXYZ".iter().cycle().take(8192).copied().collect();
    roundtrip_test(&data, "long_match_run");
}

#[test]
fn test_roundtrip_alternating_compressible_random() {
    // Alternating 1KB compressible + 1KB random-ish, 16KB total
    let mut data = Vec::with_capacity(16384);
    let mut state: u32 = 0x12345678;
    for chunk in 0..8 {
        if chunk % 2 == 0 {
            // Compressible: repeated pattern
            data.extend(b"COMPRESSIBLE!!!!".iter().cycle().take(2048));
        } else {
            // Random-ish
            for _ in 0..2048 {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                data.push((state >> 16) as u8);
            }
        }
    }
    roundtrip_test(&data, "alternating_compressible_random");
}

#[test]
fn test_roundtrip_binary_with_nulls() {
    // Data with lots of null bytes interspersed
    let mut data = vec![0u8; 4096];
    for i in 0..data.len() {
        if i % 7 == 0 { data[i] = 0xFF; }
        if i % 13 == 0 { data[i] = (i & 0xFF) as u8; }
    }
    roundtrip_test(&data, "binary_with_nulls");
}

#[test]
fn test_roundtrip_near_max_match_offset() {
    // 64KB of data where matches can be at maximum offset (65535)
    // First 32KB is unique-ish, second 32KB repeats the first
    let mut data = vec![0u8; 65536];
    let mut state: u32 = 0xCAFEBABE;
    for i in 0..32768 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        data[i] = (state >> 16) as u8;
    }
    let (first, second) = data.split_at_mut(32768);
    second.copy_from_slice(first);
    roundtrip_test(&data, "near_max_match_offset");
}
