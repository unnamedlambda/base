use base::execute;
use base_types::{Action, Algorithm, Kind, State, UnitSpec};
use std::fs;
use tempfile::TempDir;

fn create_test_algorithm(
    actions: Vec<Action>,
    payloads: Vec<u8>,
    file_units: usize,
) -> Algorithm {
    let num_actions = actions.len();

    Algorithm {
        actions,
        payloads,
        state: State {
            file_buffer_size: 65536,
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            file_units,
            cranelift_units: 0,
        },
        file_assignments: vec![0; num_actions],
        cranelift_assignments: vec![],
        worker_threads: None,
        blocking_threads: None,
        stack_size: None,
        timeout_ms: Some(5000),
        thread_name_prefix: None,
    }
}

fn create_cranelift_algorithm(
    actions: Vec<Action>,
    payloads: Vec<u8>,
    cranelift_units: usize,
    cranelift_ir_offsets: Vec<usize>,
    with_file: bool,
) -> Algorithm {
    let num_actions = actions.len();

    Algorithm {
        actions,
        payloads,
        state: State {
            file_buffer_size: if with_file { 65536 } else { 0 },
            cranelift_ir_offsets,
        },
        units: UnitSpec {
            file_units: if with_file { 1 } else { 0 },
            cranelift_units,
        },
        file_assignments: if with_file { vec![0; num_actions] } else { vec![] },
        cranelift_assignments: vec![0; num_actions],
        worker_threads: None,
        blocking_threads: None,
        stack_size: None,
        timeout_ms: Some(5000),
        thread_name_prefix: None,
    }
}

#[test]
fn test_integration_conditional_jump() {
    let temp_dir = TempDir::new().unwrap();
    let test_file_a = temp_dir.path().join("result_a.txt");
    let test_file_b = temp_dir.path().join("result_b.txt");
    let test_file_a_str = test_file_a.to_str().unwrap();
    let test_file_b_str = test_file_b.to_str().unwrap();

    let mut payloads = vec![0u8; 1024];

    // Setup filename A with null terminator
    let filename_a_bytes = format!("{}\0", test_file_a_str).into_bytes();
    payloads[0..filename_a_bytes.len()].copy_from_slice(&filename_a_bytes);

    // Setup filename B with null terminator
    let filename_b_bytes = format!("{}\0", test_file_b_str).into_bytes();
    payloads[256..256 + filename_b_bytes.len()].copy_from_slice(&filename_b_bytes);

    // Setup conditions: 1 (true) and 0 (false) as u64
    payloads[512..520].copy_from_slice(&1u64.to_le_bytes());
    payloads[520..528].copy_from_slice(&0u64.to_le_bytes());

    // Setup data value
    payloads[528..536].copy_from_slice(&42u64.to_le_bytes());

    // Completion flags
    let flag_a = 600u32;
    let flag_b = 608u32;

    let actions = vec![
        // Action 0: FileWrite to path A (data action, dispatched by index)
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 528,
            offset: 0,
            size: 8,
        },
        // Action 1: FileWrite to path B (data action, dispatched by index)
        Action {
            kind: Kind::FileWrite,
            dst: 256,
            src: 528,
            offset: 0,
            size: 8,
        },
        // Action 2: ConditionalJump with true condition (should jump to action 5)
        Action {
            kind: Kind::ConditionalJump,
            src: 512,
            dst: 5,   // Jump to second ConditionalJump (skip FileWrite A dispatch)
            offset: 0,
            size: 0,
        },
        // Action 3: AsyncDispatch FileWrite A (SKIPPED - jumped over)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,      // file unit
            src: 0,      // action index 0 (FileWrite A)
            offset: flag_a,
            size: 0,
        },
        // Action 4: Wait for FileWrite A (SKIPPED)
        Action {
            kind: Kind::Wait,
            dst: flag_a,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 5: ConditionalJump with false condition (should fall through to action 6)
        Action {
            kind: Kind::ConditionalJump,
            src: 520,
            dst: 99,
            offset: 0,
            size: 0,
        },
        // Action 6: AsyncDispatch FileWrite B (EXECUTED - fell through)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,      // file unit
            src: 1,      // action index 1 (FileWrite B)
            offset: flag_b,
            size: 0,
        },
        // Action 7: Wait for FileWrite B
        Action {
            kind: Kind::Wait,
            dst: flag_b,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1);

    execute(algorithm).unwrap();

    // Verify: File A should NOT exist (skipped)
    assert!(!test_file_a.exists());

    // Verify: File B should exist with value 42
    assert!(test_file_b.exists());
    let contents = fs::read(&test_file_b).unwrap();
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(value, 42);
}

#[test]
fn test_integration_conditional_jump_variable_size() {
    // Tests that ConditionalJump respects the size field
    // When size=4, only check first 4 bytes; when size=8 (or 0), check all 8
    let temp_dir = TempDir::new().unwrap();
    let test_file_4byte = temp_dir.path().join("result_4byte.txt");
    let test_file_8byte = temp_dir.path().join("result_8byte.txt");
    let test_file_4byte_str = test_file_4byte.to_str().unwrap();
    let test_file_8byte_str = test_file_8byte.to_str().unwrap();

    let mut payloads = vec![0u8; 1024];

    // Setup filenames
    let filename_4byte_bytes = format!("{}\0", test_file_4byte_str).into_bytes();
    payloads[0..filename_4byte_bytes.len()].copy_from_slice(&filename_4byte_bytes);

    let filename_8byte_bytes = format!("{}\0", test_file_8byte_str).into_bytes();
    payloads[256..256 + filename_8byte_bytes.len()].copy_from_slice(&filename_8byte_bytes);

    // Condition at 512: first 4 bytes are 0, next 4 bytes are non-zero
    // This means: size=4 check sees FALSE (no jump), size=8 check sees TRUE (jump)
    payloads[512..516].copy_from_slice(&0u32.to_le_bytes());      // bytes 0-3: zero
    payloads[516..520].copy_from_slice(&0xFFu32.to_le_bytes());   // bytes 4-7: non-zero

    // Data value
    payloads[528..536].copy_from_slice(&99u64.to_le_bytes());

    let flag_4byte = 600u32;
    let flag_8byte = 608u32;

    let actions = vec![
        // Action 0: FileWrite for 4-byte test
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 528,
            offset: 0,
            size: 8,
        },
        // Action 1: FileWrite for 8-byte test
        Action {
            kind: Kind::FileWrite,
            dst: 256,
            src: 528,
            offset: 0,
            size: 8,
        },
        // Action 2: ConditionalJump with size=4 (checks only first 4 bytes = 0, should NOT jump)
        Action {
            kind: Kind::ConditionalJump,
            src: 512,
            dst: 5,   // Would skip the FileWrite dispatch
            offset: 0,
            size: 4,  // Only check 4 bytes - they're all zero, so fall through
        },
        // Action 3: AsyncDispatch FileWrite 4-byte (EXECUTED - did not jump because size=4 saw zeros)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 0,
            offset: flag_4byte,
            size: 0,
        },
        // Action 4: Wait
        Action {
            kind: Kind::Wait,
            dst: flag_4byte,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 5: ConditionalJump with size=8 (checks all 8 bytes, sees non-zero, should jump)
        Action {
            kind: Kind::ConditionalJump,
            src: 512,
            dst: 8,   // Jump over the FileWrite dispatch
            offset: 0,
            size: 8,  // Check all 8 bytes - bytes 4-7 are non-zero, so jump
        },
        // Action 6: AsyncDispatch FileWrite 8-byte (SKIPPED - jumped over)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
            offset: flag_8byte,
            size: 0,
        },
        // Action 7: Wait (SKIPPED)
        Action {
            kind: Kind::Wait,
            dst: flag_8byte,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 8: End
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1);

    execute(algorithm).unwrap();

    // Verify: 4-byte file SHOULD exist (size=4 saw zeros, didn't jump, wrote file)
    assert!(test_file_4byte.exists(), "4-byte check should NOT jump when first 4 bytes are zero");
    let contents = fs::read(&test_file_4byte).unwrap();
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(value, 99);

    // Verify: 8-byte file should NOT exist (size=8 saw non-zero in bytes 4-7, jumped over write)
    assert!(!test_file_8byte.exists(), "8-byte check SHOULD jump when any of 8 bytes are non-zero");
}

#[test]
fn test_integration_file_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("source.txt");
    let verify_file = temp_dir.path().join("verify.txt");
    let test_file_str = test_file.to_str().unwrap();
    let verify_file_str = verify_file.to_str().unwrap();

    let mut payloads = vec![0u8; 1024];

    // Setup source filename with null terminator
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Setup verify filename with null terminator
    let verify_filename_bytes = format!("{}\0", verify_file_str).into_bytes();
    payloads[512..512 + verify_filename_bytes.len()].copy_from_slice(&verify_filename_bytes);

    // Setup source data (value 99)
    payloads[256..264].copy_from_slice(&99u64.to_le_bytes());

    // Completion flags
    let flag1 = 800u32;
    let flag2 = 808u32;
    let flag3 = 816u32;

    let actions = vec![
        // Action 0: FileWrite initial value to file
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 256,
            offset: 0,
            size: 8,
        },
        // Action 1: FileRead file into buffer
        Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 264,
            offset: 0,  // file byte offset (read from start)
            size: 8,
        },
        // Action 2: FileWrite buffer to verification file
        Action {
            kind: Kind::FileWrite,
            dst: 512,
            src: 264,
            offset: 0,
            size: 8,
        },
        // Action 3: AsyncDispatch FileWrite 1
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2, src: 0, offset: flag1, size: 0,
        },
        // Action 4: Wait
        Action {
            kind: Kind::Wait,
            dst: flag1, src: 0, offset: 0, size: 0,
        },
        // Action 5: AsyncDispatch FileRead
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2, src: 1, offset: flag2, size: 0,
        },
        // Action 6: Wait
        Action {
            kind: Kind::Wait,
            dst: flag2, src: 0, offset: 0, size: 0,
        },
        // Action 7: AsyncDispatch FileWrite 2
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2, src: 2, offset: flag3, size: 0,
        },
        // Action 8: Wait
        Action {
            kind: Kind::Wait,
            dst: flag3, src: 0, offset: 0, size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1);

    execute(algorithm).unwrap();

    // Verify: Source file exists with value 99
    assert!(test_file.exists());
    let source_contents = fs::read(&test_file).unwrap();
    let source_value = u64::from_le_bytes(source_contents[0..8].try_into().unwrap());
    assert_eq!(source_value, 99);

    // Verify: Verification file exists with value 99 (read from source)
    assert!(verify_file.exists());
    let verify_contents = fs::read(&verify_file).unwrap();
    let verify_value = u64::from_le_bytes(verify_contents[0..8].try_into().unwrap());
    assert_eq!(verify_value, 99);
}

#[test]
fn test_integration_filewrite_null_terminated() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("null_term_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 1024];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Setup null-terminated string data at offset 256
    let test_string = b"Hello, World!";
    payloads[256..256 + test_string.len()].copy_from_slice(test_string);
    payloads[256 + test_string.len()] = 0; // null terminator
    // Add garbage after null to verify it stops at null
    let garbage = b"GARBAGE";
    payloads[256 + test_string.len() + 1..256 + test_string.len() + 1 + garbage.len()]
        .copy_from_slice(garbage);

    let file_flag = 512u32;

    let actions = vec![
        // Action 0: FileWrite with size=0 triggers null-terminated mode
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 256,
            offset: 0,
            size: 0, // size=0 means write until null byte
        },
        // Action 1: AsyncDispatch FileWrite
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 0,
            offset: file_flag,
            size: 0,
        },
        // Action 2: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1);

    execute(algorithm).unwrap();

    assert!(result_file.exists(), "Null-terminated write result should exist");
    let contents = fs::read(&result_file).unwrap();

    // Should only contain "Hello, World!" without the null or garbage
    assert_eq!(
        contents,
        test_string,
        "FileWrite size=0 should write until null byte, not including garbage"
    );
}

#[test]
fn test_integration_file_read_with_offset() {
    // Test FileRead with offset parameter for chunked file reading
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("large_input.txt");
    let result1_file = temp_dir.path().join("chunk1.txt");
    let result2_file = temp_dir.path().join("chunk2.txt");
    let result3_file = temp_dir.path().join("chunk3.txt");

    // Create a large input file with known content
    let full_content = b"AAAAAABBBBBBCCCCCCDDDDDDEEEEEEFFFFFFGGGGGG"; // 42 bytes
    fs::write(&input_file, full_content).unwrap();

    let input_path_str = input_file.to_str().unwrap();
    let result1_path_str = result1_file.to_str().unwrap();
    let result2_path_str = result2_file.to_str().unwrap();
    let result3_path_str = result3_file.to_str().unwrap();

    let mut payloads = vec![0u8; 2048];

    // Store input filename at offset 0
    let input_bytes = format!("{}\0", input_path_str).into_bytes();
    payloads[0..input_bytes.len()].copy_from_slice(&input_bytes);

    // Store output filenames
    let result1_bytes = format!("{}\0", result1_path_str).into_bytes();
    payloads[256..256 + result1_bytes.len()].copy_from_slice(&result1_bytes);

    let result2_bytes = format!("{}\0", result2_path_str).into_bytes();
    payloads[512..512 + result2_bytes.len()].copy_from_slice(&result2_bytes);

    let result3_bytes = format!("{}\0", result3_path_str).into_bytes();
    payloads[768..768 + result3_bytes.len()].copy_from_slice(&result3_bytes);

    // Data buffers for each chunk at 1024, 1038, 1052
    let flag1 = 1536u32;
    let flag2 = 1544u32;
    let flag3 = 1552u32;
    let flag4 = 1560u32;
    let flag5 = 1568u32;
    let flag6 = 1576u32;

    let actions = vec![
        // Action 0: Read first 14 bytes (offset 0)
        Action {
            kind: Kind::FileRead,
            src: 0,           // input filename
            dst: 1024,        // destination buffer 1
            offset: 0,        // read from byte 0
            size: 14,
        },
        // Action 1: Read middle 14 bytes (offset 14)
        Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 1038,        // destination buffer 2
            offset: 14,       // read from byte 14
            size: 14,
        },
        // Action 2: Read last 14 bytes (offset 28)
        Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 1052,        // destination buffer 3
            offset: 28,       // read from byte 28
            size: 14,
        },
        // Action 3: Write chunk 1
        Action {
            kind: Kind::FileWrite,
            dst: 256,
            src: 1024,
            offset: 0,
            size: 14,
        },
        // Action 4: Write chunk 2
        Action {
            kind: Kind::FileWrite,
            dst: 512,
            src: 1038,
            offset: 0,
            size: 14,
        },
        // Action 5: Write chunk 3
        Action {
            kind: Kind::FileWrite,
            dst: 768,
            src: 1052,
            offset: 0,
            size: 14,
        },
        // Action 6-17: AsyncDispatch + Wait for each read and write
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 0,
            offset: flag1,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: flag1,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
            offset: flag2,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: flag2,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 2,
            offset: flag3,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: flag3,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 3,
            offset: flag4,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: flag4,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 4,
            offset: flag5,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: flag5,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 5,
            offset: flag6,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: flag6,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1);
    execute(algorithm).unwrap();

    // Verify each chunk was written correctly
    assert!(result1_file.exists(), "Chunk 1 should exist");
    assert!(result2_file.exists(), "Chunk 2 should exist");
    assert!(result3_file.exists(), "Chunk 3 should exist");

    let chunk1 = fs::read(&result1_file).unwrap();
    let chunk2 = fs::read(&result2_file).unwrap();
    let chunk3 = fs::read(&result3_file).unwrap();

    assert_eq!(chunk1, b"AAAAAABBBBBBCC", "Chunk 1: bytes 0-14");
    assert_eq!(chunk2, b"CCCCDDDDDDEEEE", "Chunk 2: bytes 14-28");
    assert_eq!(chunk3, b"EEFFFFFFGGGGGG", "Chunk 3: bytes 28-42");
}

#[test]
fn test_integration_file_write_with_offset() {
    // Test FileWrite with offset parameter for chunked file writing
    let temp_dir = TempDir::new().unwrap();
    let output_file = temp_dir.path().join("chunked_output.txt");

    let output_path_str = output_file.to_str().unwrap();

    let mut payloads = vec![0u8; 2048];

    // Store output filename at offset 0
    let output_bytes = format!("{}\0", output_path_str).into_bytes();
    payloads[0..output_bytes.len()].copy_from_slice(&output_bytes);

    // Data chunks to write at 512, 526, 540
    payloads[512..526].copy_from_slice(b"AAAAAABBBBBBCC"); // 14 bytes
    payloads[526..540].copy_from_slice(b"DDDDDDEEEEEEEE"); // 14 bytes
    payloads[540..554].copy_from_slice(b"FFFFFFGGGGGGGG"); // 14 bytes

    let flag1 = 1536u32;
    let flag2 = 1544u32;
    let flag3 = 1552u32;

    let actions = vec![
        // Action 0: Write first chunk at byte 0
        Action {
            kind: Kind::FileWrite,
            dst: 0,           // filename
            src: 512,         // data chunk 1
            offset: 0,        // write to byte 0 (creates file)
            size: 14,
        },
        // Action 1: Write second chunk at byte 14
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 526,         // data chunk 2
            offset: 14,       // write to byte 14 (append mode)
            size: 14,
        },
        // Action 2: Write third chunk at byte 28
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 540,         // data chunk 3
            offset: 28,       // write to byte 28 (append mode)
            size: 14,
        },
        // Action 3-8: AsyncDispatch + Wait for each write
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 0,
            offset: flag1,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: flag1,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
            offset: flag2,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: flag2,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 2,
            offset: flag3,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: flag3,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1);
    execute(algorithm).unwrap();

    // Verify the output file has all chunks written at correct positions
    assert!(output_file.exists(), "Output file should exist");

    let contents = fs::read(&output_file).unwrap();
    assert_eq!(contents, b"AAAAAABBBBBBCCDDDDDDEEEEEEEEFFFFFFGGGGGGGG", "All chunks concatenated");
}

#[test]
fn test_cranelift_basic_compilation() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_basic.txt");
    let test_file_str = test_file.to_str().unwrap();

    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    return
}
"#;

    let mut payloads = vec![0u8; 4096];

    // Store CLIF IR at offset 0 (null-terminated)
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    let cranelift_flag = 1024u32;
    let file_flag = 1032u32;
    let data_offset = 2000usize;

    // Store test value
    payloads[data_offset..data_offset + 8].copy_from_slice(&42u64.to_le_bytes());

    // Filename at offset 3000
    let filename_offset = 3000usize;
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[filename_offset..filename_offset + filename_bytes.len()]
        .copy_from_slice(&filename_bytes);

    let actions = vec![
        // Action 0: Cranelift executes (no-op, just tests compilation)
        Action { kind: Kind::FileRead, dst: data_offset as u32, src: 0, offset: 0, size: 0 },
        // Action 1: FileWrite to verify we got here
        Action { kind: Kind::FileWrite, dst: filename_offset as u32, src: data_offset as u32, offset: 0, size: 8 },
        // Action 2: AsyncDispatch Cranelift
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: cranelift_flag, size: 0 },
        // Action 3: Wait for Cranelift
        Action { kind: Kind::Wait, dst: cranelift_flag, src: 0, offset: 0, size: 0 },
        // Action 4: AsyncDispatch FileWrite
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: file_flag, size: 0 },
        // Action 5: Wait for FileWrite
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], true);
    execute(algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 42);
}

#[test]
fn test_cranelift_arithmetic_add() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_add.txt");
    let test_file_str = test_file.to_str().unwrap();

    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0
    v2 = load.i64 v0+8
    v3 = iadd v1, v2
    store.i64 v3, v0+16
    return
}
"#;

    let mut payloads = vec![0u8; 4096];

    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    let cranelift_flag = 1024u32;
    let file_flag = 1032u32;
    let data_offset = 2000usize;

    // Input values: 100 + 200
    payloads[data_offset..data_offset + 8].copy_from_slice(&100u64.to_le_bytes());
    payloads[data_offset + 8..data_offset + 16].copy_from_slice(&200u64.to_le_bytes());

    let filename_offset = 3000usize;
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[filename_offset..filename_offset + filename_bytes.len()]
        .copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::FileRead, dst: data_offset as u32, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: filename_offset as u32, src: (data_offset + 16) as u32, offset: 0, size: 8 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: cranelift_flag, size: 0 },
        Action { kind: Kind::Wait, dst: cranelift_flag, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: file_flag, size: 0 },
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], true);
    execute(algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 300);
}

#[test]
fn test_cranelift_arithmetic_multiply() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_mul.txt");
    let test_file_str = test_file.to_str().unwrap();

    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0
    v2 = load.i64 v0+8
    v3 = imul v1, v2
    store.i64 v3, v0+16
    return
}
"#;

    let mut payloads = vec![0u8; 4096];

    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    let cranelift_flag = 1024u32;
    let file_flag = 1032u32;
    let data_offset = 2000usize;

    // Input values: 7 * 9
    payloads[data_offset..data_offset + 8].copy_from_slice(&7u64.to_le_bytes());
    payloads[data_offset + 8..data_offset + 16].copy_from_slice(&9u64.to_le_bytes());

    let filename_offset = 3000usize;
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[filename_offset..filename_offset + filename_bytes.len()]
        .copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::FileRead, dst: data_offset as u32, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: filename_offset as u32, src: (data_offset + 16) as u32, offset: 0, size: 8 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: cranelift_flag, size: 0 },
        Action { kind: Kind::Wait, dst: cranelift_flag, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: file_flag, size: 0 },
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], true);
    execute(algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 63);
}

#[test]
fn test_cranelift_memory_operations() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_mem.txt");
    let test_file_str = test_file.to_str().unwrap();

    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i32 v0
    v2 = load.i32 v0+4
    v3 = load.i32 v0+8
    v4 = iadd v1, v2
    v5 = iadd v4, v3
    store.i32 v5, v0+12
    return
}
"#;

    let mut payloads = vec![0u8; 4096];

    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    let cranelift_flag = 1024u32;
    let file_flag = 1032u32;
    let data_offset = 2000usize;

    // Input values: 10 + 20 + 30 = 60
    payloads[data_offset..data_offset + 4].copy_from_slice(&10u32.to_le_bytes());
    payloads[data_offset + 4..data_offset + 8].copy_from_slice(&20u32.to_le_bytes());
    payloads[data_offset + 8..data_offset + 12].copy_from_slice(&30u32.to_le_bytes());

    let filename_offset = 3000usize;
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[filename_offset..filename_offset + filename_bytes.len()]
        .copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::FileRead, dst: data_offset as u32, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: filename_offset as u32, src: (data_offset + 12) as u32, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: cranelift_flag, size: 0 },
        Action { kind: Kind::Wait, dst: cranelift_flag, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: file_flag, size: 0 },
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], true);
    execute(algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();
    let result = u32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(result, 60);
}

#[test]
fn test_cranelift_multiple_units() {
    let temp_dir = TempDir::new().unwrap();
    let test_file1 = temp_dir.path().join("unit1_add.txt");
    let test_file2 = temp_dir.path().join("unit2_mul.txt");
    let test_file1_str = test_file1.to_str().unwrap();
    let test_file2_str = test_file2.to_str().unwrap();

    let clif_ir1 = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0
    v2 = load.i64 v0+8
    v3 = iadd v1, v2
    store.i64 v3, v0+16
    return
}
"#;

    let clif_ir2 = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0
    v2 = load.i64 v0+8
    v3 = imul v1, v2
    store.i64 v3, v0+16
    return
}
"#;

    let mut payloads = vec![0u8; 8192];

    // Store both CLIF IRs
    let clif1_bytes = format!("{}\0", clif_ir1).into_bytes();
    payloads[0..clif1_bytes.len()].copy_from_slice(&clif1_bytes);

    let ir2_offset = 600usize;
    let clif2_bytes = format!("{}\0", clif_ir2).into_bytes();
    payloads[ir2_offset..ir2_offset + clif2_bytes.len()].copy_from_slice(&clif2_bytes);

    // Unit 1 data: 5 + 3 = 8
    let data1_offset = 2000usize;
    payloads[data1_offset..data1_offset + 8].copy_from_slice(&5u64.to_le_bytes());
    payloads[data1_offset + 8..data1_offset + 16].copy_from_slice(&3u64.to_le_bytes());

    // Unit 2 data: 4 * 6 = 24
    let data2_offset = 2100usize;
    payloads[data2_offset..data2_offset + 8].copy_from_slice(&4u64.to_le_bytes());
    payloads[data2_offset + 8..data2_offset + 16].copy_from_slice(&6u64.to_le_bytes());

    // Filenames
    let file1_offset = 3000usize;
    let file1_bytes = format!("{}\0", test_file1_str).into_bytes();
    payloads[file1_offset..file1_offset + file1_bytes.len()].copy_from_slice(&file1_bytes);

    let file2_offset = 3200usize;
    let file2_bytes = format!("{}\0", test_file2_str).into_bytes();
    payloads[file2_offset..file2_offset + file2_bytes.len()].copy_from_slice(&file2_bytes);

    let actions = vec![
        // Action 0: Cranelift unit 0 computation
        Action { kind: Kind::FileRead, dst: data1_offset as u32, src: 0, offset: 0, size: 0 },
        // Action 1: Cranelift unit 1 computation
        Action { kind: Kind::FileRead, dst: data2_offset as u32, src: 0, offset: 0, size: 0 },
        // Action 2: FileWrite result 1
        Action { kind: Kind::FileWrite, dst: file1_offset as u32, src: (data1_offset + 16) as u32, offset: 0, size: 8 },
        // Action 3: FileWrite result 2
        Action { kind: Kind::FileWrite, dst: file2_offset as u32, src: (data2_offset + 16) as u32, offset: 0, size: 8 },
        // Action 4: AsyncDispatch unit 0
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        // Action 5: AsyncDispatch unit 1
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 1, offset: 1032, size: 0 },
        // Action 6: Wait for unit 0
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
        // Action 7: Wait for unit 1
        Action { kind: Kind::Wait, dst: 1032, src: 0, offset: 0, size: 0 },
        // Action 8: AsyncDispatch FileWrite 1
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 2, offset: 1040, size: 0 },
        // Action 9: AsyncDispatch FileWrite 2
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 3, offset: 1048, size: 0 },
        // Action 10: Wait for FileWrite 1
        Action { kind: Kind::Wait, dst: 1040, src: 0, offset: 0, size: 0 },
        // Action 11: Wait for FileWrite 2
        Action { kind: Kind::Wait, dst: 1048, src: 0, offset: 0, size: 0 },
    ];

    let mut algorithm = create_cranelift_algorithm(actions, payloads, 2, vec![0, ir2_offset], true);

    // Assign actions to specific units
    algorithm.cranelift_assignments = vec![
        0,   // Action 0 -> unit 0
        1,   // Action 1 -> unit 1
        255, // Action 2 (FileWrite)
        255, // Action 3 (FileWrite)
        0,   // Action 4 (Dispatch to unit 0)
        1,   // Action 5 (Dispatch to unit 1)
        0,   // Action 6 (Wait)
        1,   // Action 7 (Wait)
        255, // Action 8 (Dispatch FileWrite)
        255, // Action 9 (Dispatch FileWrite)
        255, // Action 10 (Wait)
        255, // Action 11 (Wait)
    ];

    execute(algorithm).unwrap();

    assert!(test_file1.exists());
    assert!(test_file2.exists());

    let contents1 = fs::read(&test_file1).unwrap();
    let result1 = u64::from_le_bytes(contents1[0..8].try_into().unwrap());
    assert_eq!(result1, 8);

    let contents2 = fs::read(&test_file2).unwrap();
    let result2 = u64::from_le_bytes(contents2[0..8].try_into().unwrap());
    assert_eq!(result2, 24);
}

#[test]
fn test_cranelift_conditional_logic() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_cond.txt");
    let test_file_str = test_file.to_str().unwrap();

    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0
    v2 = load.i64 v0+8
    v3 = load.i64 v0+16
    v4 = icmp_imm eq v1, 0
    brif v4, block2, block1

block1:
    store.i64 v2, v0+24
    return

block2:
    store.i64 v3, v0+24
    return
}
"#;

    let mut payloads = vec![0u8; 4096];

    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    let cranelift_flag = 1024u32;
    let file_flag = 1032u32;
    let data_offset = 2000usize;

    // condition = 1 (non-zero), value_a = 100, value_b = 200
    // Should store value_a (100)
    payloads[data_offset..data_offset + 8].copy_from_slice(&1u64.to_le_bytes());
    payloads[data_offset + 8..data_offset + 16].copy_from_slice(&100u64.to_le_bytes());
    payloads[data_offset + 16..data_offset + 24].copy_from_slice(&200u64.to_le_bytes());

    let filename_offset = 3000usize;
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[filename_offset..filename_offset + filename_bytes.len()]
        .copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::FileRead, dst: data_offset as u32, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: filename_offset as u32, src: (data_offset + 24) as u32, offset: 0, size: 8 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: cranelift_flag, size: 0 },
        Action { kind: Kind::Wait, dst: cranelift_flag, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: file_flag, size: 0 },
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], true);
    execute(algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 100); // value_a, since condition was non-zero
}

fn create_test_algorithm_with_timeout(
    actions: Vec<Action>,
    payloads: Vec<u8>,
    file_units: usize,
    timeout_ms: u64,
) -> Algorithm {
    let mut alg = create_test_algorithm(actions, payloads, file_units);
    alg.timeout_ms = Some(timeout_ms);
    alg
}

#[test]
fn test_integration_wake_then_park_progresses() {
    // Wake increments wake word from 0→1, then Park sees word != expected(0) and passes immediately.
    // FileWrite persists wake word + status for verification.
    let tmp_dir = TempDir::new().unwrap();
    let test_file = tmp_dir.path().join("park_wake.bin");
    let test_file_str = test_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    // Layout:
    // 0..8   = wake word (starts 0)
    // 8..16  = expected value (0)
    // 16..24 = status output
    // 24..32 = (spare)
    // 1000.. = filename
    let wake_addr: u32 = 0;
    let expected_addr: u32 = 8;
    let status_addr: u32 = 16;
    let filename_offset: u32 = 1000;

    // expected = 0
    payloads[expected_addr as usize..expected_addr as usize + 8].copy_from_slice(&0u64.to_le_bytes());

    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[filename_offset as usize..filename_offset as usize + filename_bytes.len()]
        .copy_from_slice(&filename_bytes);

    let file_flag: u32 = 2000;

    let actions = vec![
        // 0: Wake — increments wake word 0→1
        Action { kind: Kind::Wake, dst: wake_addr, src: 0, offset: 0, size: 0 },
        // 1: Park — wake word is already 1 != expected(0), passes immediately
        Action { kind: Kind::Park, dst: wake_addr, src: expected_addr, offset: status_addr, size: 0 },
        // 2: MemCopy wake word to offset 24 (just to ensure data is in payloads for FileWrite)
        Action { kind: Kind::FileRead, dst: wake_addr, src: 24, offset: 0, size: 0 },
        // 3: FileWrite — write 16 bytes (wake word + status) to file
        Action { kind: Kind::FileWrite, dst: filename_offset, src: wake_addr, offset: 0, size: 24 },
        // 4: AsyncDispatch file unit
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 3, offset: file_flag, size: 0 },
        // 5: Wait for file write
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_test_algorithm_with_timeout(actions, payloads, 1, 5000);
    execute(algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();

    // wake word should be 1
    let wake_val = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(wake_val, 1, "wake word should be 1 after Wake");

    // status should be 1 (woken, not timed out)
    let status_val = u64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(status_val, 1, "status should be 1 (woken)");
}

#[test]
fn test_integration_park_times_out_without_wake() {
    // Park with 10ms per-action timeout, no wake ever fires.
    // Status should be 0 (timed out).
    // WaitUntil verifies status == 0.
    let tmp_dir = TempDir::new().unwrap();
    let test_file = tmp_dir.path().join("park_timeout.bin");
    let test_file_str = test_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    // Layout:
    // 0..8   = wake word (stays 0, never woken)
    // 8..16  = expected value (0)
    // 16..24 = status output
    // 24..32 = expected status (0)
    // 1000.. = filename
    let wake_addr: u32 = 0;
    let _expected_addr: u32 = 8;
    let status_addr: u32 = 16;
    let expected_status_addr: u32 = 24;
    let filename_offset: u32 = 1000;

    // expected = 0
    payloads[8..16].copy_from_slice(&0u64.to_le_bytes());
    // expected status = 0 (for WaitUntil comparison)
    payloads[24..32].copy_from_slice(&0u64.to_le_bytes());

    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[filename_offset as usize..filename_offset as usize + filename_bytes.len()]
        .copy_from_slice(&filename_bytes);

    let file_flag: u32 = 2000;

    let actions = vec![
        // 0: Park with 10ms per-action timeout — wake word stays 0 == expected(0), so it times out
        Action { kind: Kind::Park, dst: wake_addr, src: 0, offset: status_addr, size: 10 },
        // 1: WaitUntil — wait for status == 0 (it should already be 0 = timed out)
        Action { kind: Kind::WaitUntil, dst: status_addr, src: expected_status_addr, offset: 0, size: 0 },
        // 2: FileWrite — write 32 bytes for verification
        Action { kind: Kind::FileWrite, dst: filename_offset, src: wake_addr, offset: 0, size: 32 },
        // 3: AsyncDispatch file unit
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 2, offset: file_flag, size: 0 },
        // 4: Wait for file write
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_test_algorithm_with_timeout(actions, payloads, 1, 5000);
    execute(algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();

    // wake word should still be 0 (no wake)
    let wake_val = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(wake_val, 0, "wake word should still be 0");

    // status should be 0 (timed out, not woken)
    let status_val = u64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(status_val, 0, "status should be 0 (timed out)");
}

#[test]
fn test_clif_ffi_file_write_and_read() {
    // CLIF IR writes "Hello, CLIF!" to a file via cl_file_write,
    // then reads it back via cl_file_read into a different memory region.
    // Verified by a subsequent FileWrite action that dumps the read-back data.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("clif_ffi_data.bin");
    let verify_file = temp_dir.path().join("clif_ffi_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   0..~600:    CLIF IR (null-terminated)
    //  1024..1032:  cranelift completion flag
    //  1032..1040:  file completion flag
    //  2000..2256:  data file path (null-terminated)
    //  2256..2512:  verify file path (null-terminated)
    //  3000..3012:  source data "Hello, CLIF!" (12 bytes)
    //  3100..3112:  read-back destination (12 bytes)
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    ; cl_file_write(ptr, path_off, src_off, file_offset, size) -> i64
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    ; cl_file_read(ptr, path_off, dst_off, file_offset, size) -> i64
    sig1 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig1

block0(v0: i64):
    ; write 12 bytes from offset 3000 to file
    v1 = iconst.i64 {path_off}
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 12
    v5 = call fn0(v0, v1, v2, v3, v4)

    ; read 13 bytes from file into offset 3100
    v6 = iconst.i64 3100
    v7 = call fn1(v0, v1, v6, v3, v4)

    return
}}"#, path_off = 2000);

    let mut payloads = vec![0u8; 4096];

    // Store CLIF IR at offset 0
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    assert!(clif_bytes.len() < 1024, "CLIF IR too large: {}", clif_bytes.len());
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    // Store data file path at offset 2000
    payloads[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());

    // Store verify file path at offset 2256
    payloads[2256..2256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    // Store source data at offset 3000: "Hello, CLIF!" = 12 bytes
    payloads[3000..3012].copy_from_slice(b"Hello, CLIF!");

    let cranelift_flag = 1024u32;
    let file_flag = 1032u32;

    let actions = vec![
        // Action 0: cranelift dispatch target (dst=0 so v0 = base of shared memory)
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        // Action 1: FileWrite to dump read-back data (offset 3100, 12 bytes) to verify file
        Action { kind: Kind::FileWrite, dst: 2256, src: 3100, offset: 0, size: 12 },
        // Action 2: AsyncDispatch cranelift (type 9)
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: cranelift_flag, size: 0 },
        // Action 3: Wait for cranelift
        Action { kind: Kind::Wait, dst: cranelift_flag, src: 0, offset: 0, size: 0 },
        // Action 4: AsyncDispatch FileWrite (type 2)
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: file_flag, size: 0 },
        // Action 5: Wait for FileWrite
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], true);
    execute(algorithm).unwrap();

    // The CLIF IR wrote to the data file, then read it back into memory at offset 3100.
    // The FileWrite action dumped offset 3100 to the verify file.
    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(&contents[..12], b"Hello, CLIF!");
}

#[test]
fn test_clif_ffi_file_write_read_with_offset() {
    // Tests cl_file_write at a file offset, then cl_file_read at that offset.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("clif_offset.bin");
    let verify_file = temp_dir.path().join("clif_offset_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    sig1 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig1

block0(v0: i64):
    v1 = iconst.i64 {path_off}
    v2 = iconst.i64 3000
    v3 = iconst.i64 8
    v4 = iconst.i64 5
    ; write 5 bytes from offset 3000 to file at file_offset=8
    v5 = call fn0(v0, v1, v2, v3, v4)
    ; read 5 bytes from file at file_offset=8 into offset 3100
    v6 = iconst.i64 3100
    v7 = call fn1(v0, v1, v6, v3, v4)
    return
}}"#, path_off = 2000);

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    payloads[2256..2256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[3000..3005].copy_from_slice(b"ABCDE");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 2256, src: 3100, offset: 0, size: 5 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: 1032, size: 0 },
        Action { kind: Kind::Wait, dst: 1032, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], true);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(&contents[..5], b"ABCDE");

    // Also verify the data file: first 8 bytes should be zeros (offset=8), then our data
    let raw = fs::read(&data_file).unwrap();
    assert_eq!(&raw[..8], &[0u8; 8]);
    assert_eq!(&raw[8..13], b"ABCDE");
}

#[test]
fn test_clif_ffi_file_binary_data() {
    // Tests that binary data with embedded null bytes round-trips correctly.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("clif_binary.bin");
    let verify_file = temp_dir.path().join("clif_binary_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    sig1 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig1

block0(v0: i64):
    v1 = iconst.i64 {path_off}
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    v6 = iconst.i64 3100
    v7 = call fn1(v0, v1, v6, v3, v4)
    return
}}"#, path_off = 2000);

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    payloads[2256..2256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    // Binary data with embedded nulls: [0xFF, 0x00, 0x01, 0x00, 0xAB, 0xCD, 0x00, 0xEF]
    payloads[3000..3008].copy_from_slice(&[0xFF, 0x00, 0x01, 0x00, 0xAB, 0xCD, 0x00, 0xEF]);

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 2256, src: 3100, offset: 0, size: 8 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: 1032, size: 0 },
        Action { kind: Kind::Wait, dst: 1032, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], true);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(&contents[..8], &[0xFF, 0x00, 0x01, 0x00, 0xAB, 0xCD, 0x00, 0xEF]);
}

#[test]
fn test_clif_ffi_gpu_vec_add() {
    // Tests the full GPU pipeline via CLIF FFI:
    // init → create buffers → upload → create pipeline → dispatch → download → cleanup.
    // Adds two 64-element f32 vectors element-wise.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("gpu_vec_add_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let n: usize = 64;
    let data_bytes = n * 4; // 256 bytes per buffer

    // WGSL shader for element-wise add: result[i] = a[i] + b[i]
    let wgsl = "@group(0) @binding(0) var<storage, read> a: array<f32>;\n\
                @group(0) @binding(1) var<storage, read> b: array<f32>;\n\
                @group(0) @binding(2) var<storage, read_write> result: array<f32>;\n\
                @compute @workgroup_size(64)\n\
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
                    let i = gid.x;\n\
                    if (i < arrayLength(&a)) {\n\
                        result[i] = a[i] + b[i];\n\
                    }\n\
                }\n";

    // Memory layout (must be 8-byte aligned for i64 loads):
    //    0..~1500:  CLIF IR
    // 2000..2256:   verify file path
    // 3000..3400:   shader source (null-terminated, ~300 bytes)
    // 3400..3424:   binding descriptors: 3 bindings × 8 bytes = 24 bytes
    //              [buf_id:i32, read_only:i32] × 3
    // 4000..4256:   buffer A data (64 f32s)
    // 4256..4512:   buffer B data (64 f32s)
    // 4512..4768:   result download area (64 f32s)
    let shader_off = 3000;
    let bind_off = 3400;
    let a_off = 4000;
    let b_off = 4256;
    let result_off = 4512;

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i32) -> i32 system_v
    sig5 = (i64, i32, i64, i64) -> i32 system_v
    sig6 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_upload sig2
    fn3 = %cl_gpu_create_pipeline sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig5
    fn6 = %cl_gpu_cleanup sig0
    fn7 = %cl_file_write sig6

block0(v0: i64):
    call fn0(v0)

    ; create 3 buffers (a, b, result) each {data_bytes} bytes
    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v0, v1)
    v3 = call fn1(v0, v1)
    v4 = call fn1(v0, v1)

    ; upload A to buf0
    v5 = iconst.i64 {a_off}
    v16 = call fn2(v0, v2, v5, v1)

    ; upload B to buf1
    v6 = iconst.i64 {b_off}
    v17 = call fn2(v0, v3, v6, v1)

    ; create pipeline with 3 bindings: [buf0 read, buf1 read, buf2 rw]
    v7 = iconst.i64 {shader_off}
    v8 = iconst.i64 {bind_off}
    v9 = iconst.i32 3
    v10 = call fn3(v0, v7, v8, v9)

    ; dispatch 1 workgroup of 64 threads
    v11 = iconst.i32 1
    v18 = call fn4(v0, v10, v11)

    ; download result buffer to offset {result_off}
    v12 = iconst.i64 {result_off}
    v19 = call fn5(v0, v4, v12, v1)

    ; write result to verify file
    v13 = iconst.i64 {file_off}
    v14 = iconst.i64 0
    v15 = call fn7(v0, v13, v12, v14, v1)

    call fn6(v0)
    return
}}"#,
        data_bytes = data_bytes,
        a_off = a_off,
        b_off = b_off,
        shader_off = shader_off,
        bind_off = bind_off,
        result_off = result_off,
        file_off = 2000,
    );

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    assert!(clif_bytes.len() < 2000, "CLIF IR too large: {} bytes", clif_bytes.len());
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    // Verify file path
    payloads[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    // Shader source (null-terminated)
    let shader_bytes = wgsl.as_bytes();
    payloads[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    payloads[shader_off + shader_bytes.len()] = 0;

    // Binding descriptors: [buf_id:i32, read_only:i32] × 3
    // buf0 read_only=1, buf1 read_only=1, buf2 read_only=0
    let bind = &mut payloads[bind_off..];
    bind[0..4].copy_from_slice(&0i32.to_le_bytes()); // buf0
    bind[4..8].copy_from_slice(&1i32.to_le_bytes()); // read_only
    bind[8..12].copy_from_slice(&1i32.to_le_bytes()); // buf1
    bind[12..16].copy_from_slice(&1i32.to_le_bytes()); // read_only
    bind[16..20].copy_from_slice(&2i32.to_le_bytes()); // buf2
    bind[20..24].copy_from_slice(&0i32.to_le_bytes()); // read_write

    // Fill buffer A: [1.0, 2.0, 3.0, ..., 64.0]
    for i in 0..n {
        let val = (i + 1) as f32;
        payloads[a_off + i * 4..a_off + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }
    // Fill buffer B: [100.0, 100.0, ..., 100.0]
    for i in 0..n {
        let val = 100.0f32;
        payloads[b_off + i * 4..b_off + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let mut algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    algorithm.timeout_ms = Some(15000); // GPU init can be slow
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), data_bytes, "Result file should be {} bytes", data_bytes);

    // Verify: result[i] = (i+1) + 100.0
    for i in 0..n {
        let actual = f32::from_le_bytes(contents[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 1) as f32 + 100.0;
        assert!(
            (actual - expected).abs() < 0.01,
            "Mismatch at index {}: got {}, expected {}", i, actual, expected
        );
    }
}

#[test]
fn test_clif_ffi_file_return_values() {
    // Verify that cl_file_write and cl_file_read return correct byte counts,
    // and cl_file_read on a nonexistent file returns -1.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("retval.bin");
    let missing_file = temp_dir.path().join("does_not_exist.bin");
    let verify_file = temp_dir.path().join("retval_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let missing_file_str = format!("{}\0", missing_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Layout: data file path @ 2000, missing file path @ 2200,
    //         verify file path @ 2400, source data @ 3000,
    //         results @ 3100 (3 x i64: write_ret, read_ret, missing_read_ret)
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 7
    v5 = call fn0(v0, v1, v2, v3, v4)
    store.i64 v5, v0+3100
    v6 = iconst.i64 3050
    v7 = call fn1(v0, v1, v6, v3, v4)
    store.i64 v7, v0+3108
    v8 = iconst.i64 2200
    v9 = iconst.i64 3060
    v10 = call fn1(v0, v8, v9, v3, v4)
    store.i64 v10, v0+3116
    v11 = iconst.i64 2400
    v12 = iconst.i64 3100
    v13 = iconst.i64 24
    v14 = call fn0(v0, v11, v12, v3, v13)
    return
}
"#;

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    payloads[2200..2200 + missing_file_str.len()].copy_from_slice(missing_file_str.as_bytes());
    payloads[2400..2400 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[3000..3007].copy_from_slice(b"RETVALS");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 24);
    let write_ret = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    let read_ret = i64::from_le_bytes(contents[8..16].try_into().unwrap());
    let missing_ret = i64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(write_ret, 7, "cl_file_write should return bytes written");
    assert_eq!(read_ret, 7, "cl_file_read should return bytes read");
    assert_eq!(missing_ret, -1, "cl_file_read on missing file should return -1");
}

#[test]
fn test_clif_ffi_file_read_dynamic_size() {
    // cl_file_read with size=0 should read the entire file.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("dynamic.bin");
    let verify_file = temp_dir.path().join("dynamic_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Pre-create the file with known content (23 bytes)
    fs::write(&data_file, b"dynamic size read test!").unwrap();

    // CLIF: read with size=0 → should read entire file, return 23
    // Then write the result + return value to verify file
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_read sig0
    fn1 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = call fn0(v0, v1, v2, v3, v3)
    store.i64 v4, v0+3100
    v5 = iconst.i64 2200
    v6 = iconst.i64 3100
    v7 = iconst.i64 8
    v8 = call fn1(v0, v5, v6, v3, v7)
    v9 = iconst.i64 8
    v10 = iconst.i64 23
    v11 = call fn1(v0, v5, v2, v9, v10)
    return
}
"#;

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    payloads[2200..2200 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 31, "Expected at least 31 bytes, got {}", contents.len());
    let bytes_read = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(bytes_read, 23, "size=0 should read entire 23-byte file");
    assert_eq!(&contents[8..31], b"dynamic size read test!");
}

#[test]
fn test_clif_ffi_file_write_cstring_mode() {
    // cl_file_write with size=0 writes a C-string (stops at null byte).
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("cstring.bin");
    let verify_file = temp_dir.path().join("cstring_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Source at offset 3000: "CSTR\0extra" — size=0 should write only "CSTR" (4 bytes)
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = call fn0(v0, v1, v2, v3, v3)
    store.i64 v4, v0+3100
    v5 = iconst.i64 2200
    v6 = iconst.i64 3100
    v7 = iconst.i64 8
    v8 = call fn0(v0, v5, v6, v3, v7)
    return
}
"#;

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    payloads[2200..2200 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[3000..3009].copy_from_slice(b"CSTR\0xtra");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let verify = fs::read(&verify_file).unwrap();
    let write_ret = i64::from_le_bytes(verify[0..8].try_into().unwrap());
    assert_eq!(write_ret, 4, "size=0 should write 4 bytes (up to null)");

    let raw = fs::read(&data_file).unwrap();
    assert_eq!(&raw, b"CSTR", "File should contain only 'CSTR'");
}

#[test]
fn test_clif_ffi_file_overwrite_shorter() {
    // Write long data, then overwrite with shorter data at offset 0 (truncates).
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("overwrite.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 10
    v5 = call fn0(v0, v1, v2, v3, v4)
    v6 = iconst.i64 3100
    v7 = iconst.i64 3
    v8 = call fn0(v0, v1, v6, v3, v7)
    return
}
"#;

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    payloads[3000..3010].copy_from_slice(b"LONGDATA!!");
    payloads[3100..3103].copy_from_slice(b"SML");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    // Second write with file_offset=0 uses File::create() which truncates
    let raw = fs::read(&data_file).unwrap();
    assert_eq!(&raw, b"SML", "Second write should truncate file to 3 bytes");
}

#[test]
fn test_clif_ffi_file_partial_reads() {
    // Write 100 bytes, then read in two 50-byte chunks at different offsets.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("partial.bin");
    let verify_file = temp_dir.path().join("partial_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 100
    v5 = call fn0(v0, v1, v2, v3, v4)
    v6 = iconst.i64 3200
    v7 = iconst.i64 50
    v8 = call fn1(v0, v1, v6, v3, v7)
    v9 = iconst.i64 3250
    v10 = call fn1(v0, v1, v9, v7, v7)
    v11 = iconst.i64 2200
    v12 = iconst.i64 3200
    v13 = call fn0(v0, v11, v12, v3, v4)
    return
}
"#;

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    payloads[2200..2200 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    // Fill 100 bytes at offset 3000 with pattern 0..99
    for i in 0..100u8 {
        payloads[3000 + i as usize] = i;
    }

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 100);
    // First 50 bytes should be 0..49, second 50 bytes should be 50..99
    for i in 0..100u8 {
        assert_eq!(contents[i as usize], i, "Byte {} mismatch in reassembled read", i);
    }
}

#[test]
fn test_clif_ffi_gpu_multiple_dispatches_before_download() {
    // Tests the pending_encoder batching: dispatch pipeline 3 times, then download once.
    // Uses a shader that multiplies by 2 each dispatch: data * 2 * 2 * 2 = data * 8.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("gpu_multi_dispatch.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let n: usize = 64;
    let data_bytes = n * 4;

    let wgsl = "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\
                @compute @workgroup_size(64)\n\
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
                    let i = gid.x;\n\
                    if (i < arrayLength(&data)) {\n\
                        data[i] = data[i] * 2.0;\n\
                    }\n\
                }\n";

    let shader_off = 3000;
    let bind_off = 3400;
    let data_off = 4000;

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i32) -> i32 system_v
    sig5 = (i64, i32, i64, i64) -> i32 system_v
    sig6 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_upload sig2
    fn3 = %cl_gpu_create_pipeline sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig5
    fn6 = %cl_gpu_cleanup sig0
    fn7 = %cl_file_write sig6
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v0, v1)
    v3 = iconst.i64 {data_off}
    v12 = call fn2(v0, v2, v3, v1)
    v4 = iconst.i64 {shader_off}
    v5 = iconst.i64 {bind_off}
    v6 = iconst.i32 1
    v7 = call fn3(v0, v4, v5, v6)
    v13 = call fn4(v0, v7, v6)
    v14 = call fn4(v0, v7, v6)
    v15 = call fn4(v0, v7, v6)
    v8 = iconst.i64 {result_off}
    v16 = call fn5(v0, v2, v8, v1)
    v9 = iconst.i64 2000
    v10 = iconst.i64 0
    v11 = call fn7(v0, v9, v8, v10, v1)
    call fn6(v0)
    return
}}"#,
        data_bytes = data_bytes,
        data_off = data_off,
        shader_off = shader_off,
        bind_off = bind_off,
        result_off = 4500,
    );

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let shader_bytes = wgsl.as_bytes();
    payloads[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    payloads[shader_off + shader_bytes.len()] = 0;

    // 1 binding: buf0 read_write
    payloads[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());
    payloads[bind_off + 4..bind_off + 8].copy_from_slice(&0i32.to_le_bytes());

    // Fill data: [1.0, 2.0, ..., 64.0]
    for i in 0..n {
        let val = (i + 1) as f32;
        payloads[data_off + i * 4..data_off + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let mut algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    algorithm.timeout_ms = Some(15000);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), data_bytes);
    for i in 0..n {
        let actual = f32::from_le_bytes(contents[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 1) as f32 * 8.0; // *2 three times
        assert!(
            (actual - expected).abs() < 0.01,
            "Index {}: got {}, expected {} (after 3x multiply by 2)", i, actual, expected
        );
    }
}

#[test]
fn test_clif_ffi_gpu_buffer_reuse() {
    // Upload data A, dispatch, download result A, then upload data B, dispatch, download result B.
    // Verifies that buffer state is correctly updated on re-upload.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("gpu_reuse.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let n: usize = 64;
    let data_bytes = n * 4;

    // Shader: data[i] = data[i] + 1.0
    let wgsl = "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\
                @compute @workgroup_size(64)\n\
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
                    let i = gid.x;\n\
                    if (i < arrayLength(&data)) {\n\
                        data[i] = data[i] + 1.0;\n\
                    }\n\
                }\n";

    let shader_off = 3000;
    let bind_off = 3400;
    let data_a_off = 4000;
    let data_b_off = 4300;
    let result_a_off = 4600;
    let result_b_off = result_a_off + data_bytes;

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i32) -> i32 system_v
    sig5 = (i64, i32, i64, i64) -> i32 system_v
    sig6 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_upload sig2
    fn3 = %cl_gpu_create_pipeline sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig5
    fn6 = %cl_gpu_cleanup sig0
    fn7 = %cl_file_write sig6
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v0, v1)
    v3 = iconst.i64 {shader_off}
    v4 = iconst.i64 {bind_off}
    v5 = iconst.i32 1
    v6 = call fn3(v0, v3, v4, v5)
    v7 = iconst.i64 {data_a_off}
    v15 = call fn2(v0, v2, v7, v1)
    v16 = call fn4(v0, v6, v5)
    v8 = iconst.i64 {result_a_off}
    v17 = call fn5(v0, v2, v8, v1)
    v9 = iconst.i64 {data_b_off}
    v18 = call fn2(v0, v2, v9, v1)
    v19 = call fn4(v0, v6, v5)
    v10 = iconst.i64 {result_b_off}
    v20 = call fn5(v0, v2, v10, v1)
    v11 = iconst.i64 2000
    v12 = iconst.i64 0
    v13 = iconst.i64 {two_data_bytes}
    v14 = call fn7(v0, v11, v8, v12, v13)
    call fn6(v0)
    return
}}"#,
        data_bytes = data_bytes,
        shader_off = shader_off,
        bind_off = bind_off,
        data_a_off = data_a_off,
        data_b_off = data_b_off,
        result_a_off = result_a_off,
        result_b_off = result_b_off,
        two_data_bytes = data_bytes * 2,
    );

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let shader_bytes = wgsl.as_bytes();
    payloads[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    payloads[shader_off + shader_bytes.len()] = 0;

    payloads[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());
    payloads[bind_off + 4..bind_off + 8].copy_from_slice(&0i32.to_le_bytes());

    // Data A: all 10.0
    for i in 0..n {
        payloads[data_a_off + i * 4..data_a_off + i * 4 + 4].copy_from_slice(&10.0f32.to_le_bytes());
    }
    // Data B: all 100.0
    for i in 0..n {
        payloads[data_b_off + i * 4..data_b_off + i * 4 + 4].copy_from_slice(&100.0f32.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let mut algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    algorithm.timeout_ms = Some(15000);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), data_bytes * 2);

    // Result A: each element should be 10.0 + 1.0 = 11.0
    for i in 0..n {
        let actual = f32::from_le_bytes(contents[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - 11.0).abs() < 0.01,
            "Result A index {}: got {}, expected 11.0", i, actual
        );
    }
    // Result B: each element should be 100.0 + 1.0 = 101.0
    for i in 0..n {
        let off = data_bytes + i * 4;
        let actual = f32::from_le_bytes(contents[off..off + 4].try_into().unwrap());
        assert!(
            (actual - 101.0).abs() < 0.01,
            "Result B index {}: got {}, expected 101.0", i, actual
        );
    }
}

///   - create_pipeline with out-of-range buf_id in binding returns -1
/// All error codes are stored and written to a verify file.
#[test]
fn test_clif_ffi_gpu_error_codes() {
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("gpu_errors.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   0..~1800:  CLIF IR
    //   2000..xx:  verify file path
    //   3000..3200: dummy shader (valid WGSL, needed for init)
    //   3200..3208: binding descriptor [buf_id=99, read_only=0] (invalid buf_id)
    //   4000..4032: return values (4 i32s stored as i64: create_buf_rc, upload_rc, dispatch_rc, download_rc, pipeline_bad_bind_rc)

    let wgsl = "@group(0) @binding(0) var<storage, read_write> d: array<f32>;\n\
                @compute @workgroup_size(1)\n\
                fn main() { d[0] = 1.0; }\n";

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i32) -> i32 system_v
    sig5 = (i64) system_v
    sig6 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_upload sig2
    fn3 = %cl_gpu_create_pipeline sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig2
    fn6 = %cl_gpu_cleanup sig5
    fn7 = %cl_file_write sig6
block0(v0: i64):
    call fn0(v0)

    ; create_buffer with size=0 → should return -1
    v1 = iconst.i64 0
    v2 = call fn1(v0, v1)
    v20 = sextend.i64 v2
    store.i64 v20, v0+4000

    ; upload with buf_id=99 (no buffers exist yet) → should return -1
    v3 = iconst.i32 99
    v4 = iconst.i64 3000
    v5 = iconst.i64 64
    v6 = call fn2(v0, v3, v4, v5)
    v21 = sextend.i64 v6
    store.i64 v21, v0+4008

    ; dispatch with pipeline_id=99 (no pipelines exist) → should return -1
    v7 = iconst.i32 99
    v8 = iconst.i32 1
    v9 = call fn4(v0, v7, v8)
    v22 = sextend.i64 v9
    store.i64 v22, v0+4016

    ; download with buf_id=99 → should return -1
    v10 = iconst.i64 3000
    v11 = iconst.i64 64
    v12 = call fn5(v0, v3, v10, v11)
    v23 = sextend.i64 v12
    store.i64 v23, v0+4024

    ; create a valid buffer so we can test pipeline with bad binding
    v13 = iconst.i64 256
    v14 = call fn1(v0, v13)

    ; create_pipeline with binding that references buf_id=99 → should return -1
    v15 = iconst.i64 3000
    v16 = iconst.i64 3200
    v17 = iconst.i32 1
    v18 = call fn3(v0, v15, v16, v17)
    v24 = sextend.i64 v18
    store.i64 v24, v0+4032

    ; write 40 bytes of return values to verify file
    v25 = iconst.i64 2000
    v26 = iconst.i64 4000
    v27 = iconst.i64 0
    v28 = iconst.i64 40
    v29 = call fn7(v0, v25, v26, v27, v28)

    call fn6(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    // Valid shader at 3000
    let shader_bytes = wgsl.as_bytes();
    payloads[3000..3000 + shader_bytes.len()].copy_from_slice(shader_bytes);
    payloads[3000 + shader_bytes.len()] = 0;

    // Binding descriptor at 3200: buf_id=99 (invalid), read_only=0
    payloads[3200..3204].copy_from_slice(&99i32.to_le_bytes());
    payloads[3204..3208].copy_from_slice(&0i32.to_le_bytes());

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let mut algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    algorithm.timeout_ms = Some(10000);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 40, "Expected 40 bytes of return values");

    let create_buf_rc = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    let upload_rc = i64::from_le_bytes(contents[8..16].try_into().unwrap());
    let dispatch_rc = i64::from_le_bytes(contents[16..24].try_into().unwrap());
    let download_rc = i64::from_le_bytes(contents[24..32].try_into().unwrap());
    let pipeline_bad_bind_rc = i64::from_le_bytes(contents[32..40].try_into().unwrap());

    assert_eq!(create_buf_rc, -1, "create_buffer(size=0) should return -1");
    assert_eq!(upload_rc, -1, "upload(buf_id=99) should return -1");
    assert_eq!(dispatch_rc, -1, "dispatch(pipeline_id=99) should return -1");
    assert_eq!(download_rc, -1, "download(buf_id=99) should return -1");
    assert_eq!(pipeline_bad_bind_rc, -1, "create_pipeline with invalid buf_id binding should return -1");
}

#[test]
fn test_clif_ffi_net_echo() {
    // CLIF IR: init network → connect to a Rust echo server → send "hello" → recv response → cleanup.
    // Verification: the echoed data is written to a file via cl_file_write from within CLIF.
    use std::net::TcpListener;
    use std::io::{Read, Write};

    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("net_echo_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Find a free port
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let addr_str = format!("127.0.0.1:{}\0", port);

    // Spawn echo server thread
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 64];
        let n = stream.read(&mut buf).unwrap();
        stream.write_all(&buf[..n]).unwrap();
    });

    // Memory layout:
    //   2000..2100:  server address (null-terminated)
    //   2100..2200:  verify file path (null-terminated)
    //   3000..3005:  send data "hello"
    //   3100..3105:  recv buffer
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i64 system_v
    sig2 = (i64, i64, i64, i64) -> i64 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_net_init sig0
    fn1 = %cl_net_connect sig1
    fn2 = %cl_net_send sig2
    fn3 = %cl_net_recv sig2
    fn4 = %cl_net_cleanup sig0
    fn5 = %cl_file_write sig3

block0(v0: i64):
    call fn0(v0)

    ; connect
    v1 = iconst.i64 {addr_off}
    v2 = call fn1(v0, v1)

    ; send 5 bytes from offset 3000
    v3 = iconst.i64 3000
    v4 = iconst.i64 5
    v5 = call fn2(v0, v2, v3, v4)

    ; recv into offset 3100
    v6 = iconst.i64 3100
    v7 = call fn3(v0, v2, v6, v4)

    ; write received data to verify file via cl_file_write
    v8 = iconst.i64 {file_off}
    v9 = iconst.i64 0
    v10 = call fn5(v0, v8, v6, v9, v4)

    call fn4(v0)
    return
}}"#, addr_off = 2000, file_off = 2100);

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + addr_str.len()].copy_from_slice(addr_str.as_bytes());
    payloads[2100..2100 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[3000..3005].copy_from_slice(b"hello");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();
    handle.join().unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(&contents[..5], b"hello");
}

#[test]
fn test_clif_ffi_net_listen_accept() {
    // CLIF IR: init → listen → accept → recv → send (echo) → cleanup.
    // A Rust thread connects and sends "ping!", expects echo back.
    use std::net::TcpStream;
    use std::io::{Read, Write};

    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("net_listen_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let addr_str = format!("127.0.0.1:{}\0", port);

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i64 system_v
    sig2 = (i64, i64) -> i64 system_v
    sig3 = (i64, i64, i64, i64) -> i64 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_net_init sig0
    fn1 = %cl_net_listen sig1
    fn2 = %cl_net_accept sig2
    fn3 = %cl_net_send sig3
    fn4 = %cl_net_recv sig3
    fn5 = %cl_net_cleanup sig0
    fn6 = %cl_file_write sig4

block0(v0: i64):
    call fn0(v0)

    ; listen
    v1 = iconst.i64 {addr_off}
    v2 = call fn1(v0, v1)

    ; accept a connection
    v3 = call fn2(v0, v2)

    ; recv 5 bytes into offset 3100
    v4 = iconst.i64 3100
    v5 = iconst.i64 5
    v6 = call fn4(v0, v3, v4, v5)

    ; echo: send those 5 bytes back
    v7 = call fn3(v0, v3, v4, v5)

    ; write received data to verify file
    v8 = iconst.i64 {file_off}
    v9 = iconst.i64 0
    v10 = call fn6(v0, v8, v4, v9, v5)

    call fn5(v0)
    return
}}"#, addr_off = 2000, file_off = 2100);

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + addr_str.len()].copy_from_slice(addr_str.as_bytes());
    payloads[2100..2100 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    // Spawn client thread that connects and sends "ping!"
    let port_copy = port;
    let client_handle = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(200));
        let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port_copy)).unwrap();
        stream.write_all(b"ping!").unwrap();
        let mut buf = [0u8; 5];
        stream.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"ping!");
    });

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let mut algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    algorithm.timeout_ms = Some(10000);
    execute(algorithm).unwrap();
    client_handle.join().unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(&contents[..5], b"ping!");
}

#[test]
fn test_clif_ffi_net_invalid_handle() {
    // Send/recv on handle 0 (never created) should return errors.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("net_invalid_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i64, i64) -> i64 system_v
    sig2 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_net_init sig0
    fn1 = %cl_net_send sig1
    fn2 = %cl_net_recv sig1
    fn3 = %cl_net_cleanup sig0
    fn4 = %cl_file_write sig2
block0(v0: i64):
    call fn0(v0)
    ; send on handle 0 (never created)
    v1 = iconst.i64 0
    v2 = iconst.i64 3000
    v3 = iconst.i64 5
    v4 = call fn1(v0, v1, v2, v3)
    store.i64 v4, v0+3100
    ; recv on handle 0
    v5 = iconst.i64 3050
    v6 = call fn2(v0, v1, v5, v3)
    store.i64 v6, v0+3108
    ; send on handle 42 (never created)
    v8 = iconst.i64 42
    v9 = call fn1(v0, v8, v2, v3)
    store.i64 v9, v0+3116
    ; write results to file
    v10 = iconst.i64 2200
    v11 = iconst.i64 3100
    v12 = iconst.i64 0
    v13 = iconst.i64 24
    v14 = call fn4(v0, v10, v11, v12, v13)
    call fn3(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2200..2200 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[3000..3005].copy_from_slice(b"hello");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 24);
    let send_ret = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    let recv_ret = i64::from_le_bytes(contents[8..16].try_into().unwrap());
    let send42_ret = i64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(send_ret, -1, "send on handle 0 should return -1");
    assert_eq!(recv_ret, -1, "recv on handle 0 should return -1");
    assert_eq!(send42_ret, -1, "send on handle 42 (never created) should return -1");
}

#[test]
fn test_clif_ffi_net_connect_bad_address() {
    // Connect to an address that can't be reached, verify handle=0.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("net_badaddr_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let bad_addr = "127.0.0.1:1\0";

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i64 system_v
    sig2 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_net_init sig0
    fn1 = %cl_net_connect sig1
    fn2 = %cl_net_cleanup sig0
    fn3 = %cl_file_write sig2
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 2000
    v2 = call fn1(v0, v1)
    store.i64 v2, v0+3000
    v3 = iconst.i64 2200
    v4 = iconst.i64 3000
    v5 = iconst.i64 0
    v6 = iconst.i64 8
    v7 = call fn3(v0, v3, v4, v5, v6)
    call fn2(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + bad_addr.len()].copy_from_slice(bad_addr.as_bytes());
    payloads[2200..2200 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let mut algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    algorithm.timeout_ms = Some(10000);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let handle = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(handle, 0, "connect to unreachable address should return handle 0");
}

#[test]
fn test_clif_ffi_net_recv_writes_at_offset() {
    // Verify cl_net_recv writes data at the correct shared memory offset.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("recv_offset.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);
    let addr_str = format!("127.0.0.1:{}\0", port);

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i64 system_v
    sig2 = (i64, i64) -> i64 system_v
    sig3 = (i64, i64, i64, i64) -> i64 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_net_init sig0
    fn1 = %cl_net_listen sig1
    fn2 = %cl_net_accept sig2
    fn3 = %cl_net_recv sig3
    fn4 = %cl_net_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)

    ; listen
    v1 = iconst.i64 2000
    v2 = call fn1(v0, v1)

    ; accept
    v3 = call fn2(v0, v2)

    ; recv 16 bytes into offset 3000
    v4 = iconst.i64 3000
    v5 = iconst.i64 16
    v6 = call fn3(v0, v3, v4, v5)

    ; store recv return value at 5000
    store.i64 v6, v0+5000

    ; write recv buffer (16 bytes from offset 3000) to verify file
    v7 = iconst.i64 2100
    v8 = iconst.i64 0
    v9 = call fn5(v0, v7, v4, v8, v5)

    ; append recv return value (8 bytes from offset 5000) at file offset 16
    v10 = iconst.i64 5000
    v11 = iconst.i64 16
    v12 = iconst.i64 8
    v13 = call fn5(v0, v7, v10, v11, v12)

    call fn4(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[2000..2000 + addr_str.len()].copy_from_slice(addr_str.as_bytes());
    payloads[2100..2100 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let pattern: [u8; 16] = [0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let client_port = port;
    let client = std::thread::spawn(move || {
        use std::io::Write;
        for _ in 0..40 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            if let Ok(mut stream) = std::net::TcpStream::connect(format!("127.0.0.1:{}", client_port)) {
                stream.write_all(&pattern).unwrap();
                return;
            }
        }
        panic!("client could not connect");
    });

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let mut algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    algorithm.timeout_ms = Some(10000);
    execute(algorithm).unwrap();
    client.join().unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 24, "expected 24 bytes, got {}", contents.len());

    let recv_data = &contents[0..16];
    assert_eq!(recv_data, &[0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

    let recv_ret = i64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(recv_ret, 16, "recv should return 16 bytes read");
}

#[test]
fn test_clif_ffi_lmdb_put_get_delete() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_basic");
    let verify_file = temp_dir.path().join("lmdb_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i32, i64, i32) -> i32 system_v
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_delete sig4
    fn5 = %cl_lmdb_cleanup sig0
    fn6 = %cl_file_write sig5
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 3
    v6 = iconst.i64 5010
    v7 = iconst.i32 6
    v8 = call fn2(v0, v3, v4, v5, v6, v7)
    v9 = iconst.i64 5100
    v10 = call fn3(v0, v3, v4, v5, v9)
    store.i32 v10, v0+5300
    v11 = call fn4(v0, v3, v4, v5)
    v12 = iconst.i64 5200
    v13 = call fn3(v0, v3, v4, v5, v12)
    store.i32 v13, v0+5304
    v14 = iconst.i64 4256
    v15 = iconst.i64 5300
    v16 = iconst.i64 0
    v17 = iconst.i64 8
    v18 = call fn6(v0, v14, v15, v16, v17)
    v19 = iconst.i64 8
    v20 = iconst.i64 6
    v22 = iconst.i64 5104
    v21 = call fn6(v0, v14, v22, v19, v20)
    call fn5(v0)
    return
}}"#);

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000..5003].copy_from_slice(b"foo");
    payloads[5010..5016].copy_from_slice(b"barbaz");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 14);
    let first_len = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(first_len, 6, "First get should return length 6");
    let second_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(second_ret, -1, "Get after delete should return -1");
    assert_eq!(&contents[8..14], b"barbaz");
}

#[test]
fn test_clif_ffi_lmdb_batch_write() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_batch");
    let verify_file = temp_dir.path().join("lmdb_batch_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i32) -> i32 system_v
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_begin_write_txn sig4
    fn5 = %cl_lmdb_commit_write_txn sig4
    fn6 = %cl_lmdb_cleanup sig0
    fn7 = %cl_file_write sig5
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = call fn4(v0, v3)
    v5 = iconst.i32 2
    v6 = iconst.i64 5000
    v7 = iconst.i64 5030
    v8 = call fn2(v0, v3, v6, v5, v7, v5)
    v9 = iconst.i64 5010
    v10 = iconst.i64 5040
    v11 = call fn2(v0, v3, v9, v5, v10, v5)
    v12 = iconst.i64 5020
    v13 = iconst.i64 5050
    v14 = call fn2(v0, v3, v12, v5, v13, v5)
    v15 = call fn5(v0, v3)
    v16 = iconst.i64 5100
    v17 = call fn3(v0, v3, v6, v5, v16)
    store.i32 v17, v0+5400
    v18 = iconst.i64 5200
    v19 = call fn3(v0, v3, v9, v5, v18)
    store.i32 v19, v0+5404
    v20 = iconst.i64 5300
    v21 = call fn3(v0, v3, v12, v5, v20)
    store.i32 v21, v0+5408
    v22 = iconst.i64 4256
    v23 = iconst.i64 5400
    v24 = iconst.i64 0
    v25 = iconst.i64 12
    v26 = call fn7(v0, v22, v23, v24, v25)
    v27 = iconst.i64 12
    v28 = iconst.i64 2
    v34 = iconst.i64 5104
    v29 = call fn7(v0, v22, v34, v27, v28)
    v30 = iconst.i64 14
    v35 = iconst.i64 5204
    v31 = call fn7(v0, v22, v35, v30, v28)
    v32 = iconst.i64 16
    v36 = iconst.i64 5304
    v33 = call fn7(v0, v22, v36, v32, v28)
    call fn6(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000..5002].copy_from_slice(b"k1");
    payloads[5010..5012].copy_from_slice(b"k2");
    payloads[5020..5022].copy_from_slice(b"k3");
    payloads[5030..5032].copy_from_slice(b"v1");
    payloads[5040..5042].copy_from_slice(b"v2");
    payloads[5050..5052].copy_from_slice(b"v3");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 18);
    let len1 = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let len2 = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    let len3 = i32::from_le_bytes(contents[8..12].try_into().unwrap());
    assert_eq!(len1, 2);
    assert_eq!(len2, 2);
    assert_eq!(len3, 2);
    assert_eq!(&contents[12..14], b"v1");
    assert_eq!(&contents[14..16], b"v2");
    assert_eq!(&contents[16..18], b"v3");
}

#[test]
fn test_clif_ffi_lmdb_cursor_scan() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_scan");
    let verify_file = temp_dir.path().join("lmdb_scan_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_cursor_scan sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i32 1
    v5 = iconst.i64 5000
    v6 = iconst.i64 5030
    v7 = call fn2(v0, v3, v5, v4, v6, v4)
    v8 = iconst.i64 5010
    v9 = call fn2(v0, v3, v8, v4, v6, v4)
    v10 = iconst.i64 5020
    v11 = call fn2(v0, v3, v10, v4, v6, v4)
    v12 = iconst.i64 5040
    v13 = iconst.i32 0
    v14 = iconst.i32 100
    v15 = iconst.i64 5100
    v16 = call fn3(v0, v3, v12, v13, v14, v15)
    store.i32 v16, v0+5500
    v17 = iconst.i64 4256
    v18 = iconst.i64 5500
    v19 = iconst.i64 0
    v20 = iconst.i64 4
    v21 = call fn5(v0, v17, v18, v19, v20)
    call fn4(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000] = b'a';
    payloads[5010] = b'b';
    payloads[5020] = b'c';
    payloads[5030] = b'x';

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 4);
    let count = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(count, 3, "Cursor scan should find 3 entries");
}

#[test]
fn test_clif_ffi_lmdb_get_nonexistent_key() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_nokey");
    let verify_file = temp_dir.path().join("lmdb_nokey_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 1
    v6 = iconst.i64 5010
    v7 = iconst.i32 3
    v8 = call fn2(v0, v3, v4, v5, v6, v7)
    store.i32 v8, v0+5100
    v9 = iconst.i64 5001
    v10 = iconst.i64 5200
    v11 = call fn3(v0, v3, v9, v5, v10)
    store.i32 v11, v0+5104
    v12 = iconst.i64 4256
    v13 = iconst.i64 5100
    v14 = iconst.i64 0
    v15 = iconst.i64 8
    v16 = call fn5(v0, v12, v13, v14, v15)
    call fn4(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000] = b'a';
    payloads[5001] = b'b';
    payloads[5010..5013].copy_from_slice(b"val");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let put_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let get_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(put_ret, 0, "put should succeed");
    assert_eq!(get_ret, -1, "get on non-existent key should return -1");
}

#[test]
fn test_clif_ffi_lmdb_put_overwrite() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_overwrite");
    let verify_file = temp_dir.path().join("lmdb_overwrite_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 1
    v6 = iconst.i64 5010
    v7 = iconst.i32 3
    v8 = call fn2(v0, v3, v4, v5, v6, v7)
    v9 = iconst.i64 5020
    v10 = call fn2(v0, v3, v4, v5, v9, v7)
    v11 = iconst.i64 5100
    v12 = call fn3(v0, v3, v4, v5, v11)
    store.i32 v12, v0+5200
    v13 = iconst.i64 4256
    v14 = iconst.i64 5200
    v15 = iconst.i64 0
    v16 = iconst.i64 4
    v17 = call fn5(v0, v13, v14, v15, v16)
    v18 = iconst.i64 4
    v19 = iconst.i64 3
    v20 = iconst.i64 5104
    v21 = call fn5(v0, v13, v20, v18, v19)
    call fn4(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000] = b'k';
    payloads[5010..5013].copy_from_slice(b"old");
    payloads[5020..5023].copy_from_slice(b"new");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 7);
    let get_len = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(get_len, 3);
    assert_eq!(&contents[4..7], b"new");
}

#[test]
fn test_clif_ffi_lmdb_commit_without_begin() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_nobegin");
    let verify_file = temp_dir.path().join("lmdb_nobegin_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32) -> i32 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_commit_write_txn sig2
    fn3 = %cl_lmdb_cleanup sig0
    fn4 = %cl_file_write sig3
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = call fn2(v0, v3)
    store.i32 v4, v0+5000
    v5 = iconst.i64 4256
    v6 = iconst.i64 5000
    v7 = iconst.i64 0
    v8 = iconst.i64 4
    v9 = call fn4(v0, v5, v6, v7, v8)
    call fn3(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let commit_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(commit_ret, -1, "commit without begin should return -1");
}

#[test]
fn test_clif_ffi_lmdb_cursor_scan_empty_db() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_empty");
    let verify_file = temp_dir.path().join("lmdb_empty_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_cursor_scan sig2
    fn3 = %cl_lmdb_cleanup sig0
    fn4 = %cl_file_write sig3
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 0
    v6 = iconst.i32 100
    v7 = iconst.i64 5100
    v8 = call fn2(v0, v3, v4, v5, v6, v7)
    store.i32 v8, v0+5200
    v9 = iconst.i64 4256
    v10 = iconst.i64 5200
    v11 = iconst.i64 0
    v12 = iconst.i64 4
    v13 = call fn4(v0, v9, v10, v11, v12)
    call fn3(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let count = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(count, 0, "cursor scan on empty db should return 0");
}

#[test]
fn test_clif_ffi_lmdb_cursor_scan_with_start_key() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_rangescan");
    let verify_file = temp_dir.path().join("lmdb_range_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_cursor_scan sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i32 1
    v20 = iconst.i64 5060
    v5 = iconst.i64 5000
    v6 = call fn2(v0, v3, v5, v4, v20, v4)
    v7 = iconst.i64 5001
    v8 = call fn2(v0, v3, v7, v4, v20, v4)
    v9 = iconst.i64 5002
    v10 = call fn2(v0, v3, v9, v4, v20, v4)
    v11 = iconst.i64 5003
    v12 = call fn2(v0, v3, v11, v4, v20, v4)
    v13 = iconst.i64 5004
    v14 = call fn2(v0, v3, v13, v4, v20, v4)
    v15 = iconst.i32 100
    v16 = iconst.i64 5200
    v17 = call fn3(v0, v3, v9, v4, v15, v16)
    store.i32 v17, v0+5300
    v18 = iconst.i64 4256
    v19 = iconst.i64 5300
    v21 = iconst.i64 0
    v22 = iconst.i64 4
    v23 = call fn5(v0, v18, v19, v21, v22)
    call fn4(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000] = b'a';
    payloads[5001] = b'b';
    payloads[5002] = b'c';
    payloads[5003] = b'd';
    payloads[5004] = b'e';
    payloads[5060] = b'x';

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let count = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(count, 3, "scan from 'c' should find c, d, e = 3 entries");
}

#[test]
fn test_clif_ffi_lmdb_sync() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_sync");
    let verify_file = temp_dir.path().join("lmdb_sync_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_sync sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 1
    v6 = iconst.i64 5010
    v7 = call fn2(v0, v3, v4, v5, v6, v5)
    v8 = call fn3(v0, v3)
    store.i32 v8, v0+5100
    v9 = iconst.i64 4256
    v10 = iconst.i64 5100
    v11 = iconst.i64 0
    v12 = iconst.i64 4
    v13 = call fn5(v0, v9, v10, v11, v12)
    call fn4(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000] = b'k';
    payloads[5010] = b'v';

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let sync_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(sync_ret, 0, "sync should return 0");
}

#[test]
fn test_clif_ffi_lmdb_delete_nonexistent_key() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_delnone");
    let verify_file = temp_dir.path().join("lmdb_delnone_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Delete a key that was never inserted (auto-commit path — exercises txn abort fix).
    // Then put a key and get it to prove the db still works after the failed delete.
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_delete sig2
    fn3 = %cl_lmdb_put sig3
    fn4 = %cl_lmdb_get sig4
    fn5 = %cl_lmdb_cleanup sig0
    fn6 = %cl_file_write sig5
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    ; delete key "x" which doesn't exist
    v4 = iconst.i64 5000
    v5 = iconst.i32 1
    v6 = call fn2(v0, v3, v4, v5)
    store.i32 v6, v0+5100
    ; now put key "a" = "ok" to prove db still works
    v7 = iconst.i64 5010
    v8 = iconst.i64 5020
    v9 = iconst.i32 2
    v10 = call fn3(v0, v3, v7, v5, v8, v9)
    store.i32 v10, v0+5104
    ; get key "a"
    v11 = iconst.i64 5200
    v12 = call fn4(v0, v3, v7, v5, v11)
    store.i32 v12, v0+5108
    ; write [del_ret, put_ret, get_ret, value]
    v13 = iconst.i64 4256
    v14 = iconst.i64 5100
    v15 = iconst.i64 0
    v16 = iconst.i64 12
    v17 = call fn6(v0, v13, v14, v15, v16)
    v18 = iconst.i64 12
    v19 = iconst.i64 2
    v20 = iconst.i64 5204
    v21 = call fn6(v0, v13, v20, v18, v19)
    call fn5(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000] = b'x';
    payloads[5010] = b'a';
    payloads[5020..5022].copy_from_slice(b"ok");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 14);
    let del_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let put_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    let get_ret = i32::from_le_bytes(contents[8..12].try_into().unwrap());
    assert_eq!(del_ret, -1, "delete nonexistent key should return -1");
    assert_eq!(put_ret, 0, "put after failed delete should succeed");
    assert_eq!(get_ret, 2, "get should return value length 2");
    assert_eq!(&contents[12..14], b"ok");
}

#[test]
fn test_clif_ffi_lmdb_invalid_handle() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_invhandle");
    let verify_file = temp_dir.path().join("lmdb_invhandle_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Open a db (handle 0), then do put/get/delete/scan/sync on handle 99.
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i32, i64, i32) -> i32 system_v
    sig5 = (i64, i32) -> i32 system_v
    sig6 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    sig7 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_delete sig4
    fn5 = %cl_lmdb_sync sig5
    fn6 = %cl_lmdb_cursor_scan sig6
    fn7 = %cl_lmdb_begin_write_txn sig5
    fn8 = %cl_lmdb_commit_write_txn sig5
    fn9 = %cl_lmdb_cleanup sig0
    fn10 = %cl_file_write sig7
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    ; use bogus handle 99 for everything
    v4 = iconst.i32 99
    v5 = iconst.i64 5000
    v6 = iconst.i32 1
    v7 = iconst.i64 5010
    v8 = iconst.i32 2
    ; put on bad handle
    v9 = call fn2(v0, v4, v5, v6, v7, v8)
    store.i32 v9, v0+5100
    ; get on bad handle
    v10 = iconst.i64 5200
    v11 = call fn3(v0, v4, v5, v6, v10)
    store.i32 v11, v0+5104
    ; delete on bad handle
    v12 = call fn4(v0, v4, v5, v6)
    store.i32 v12, v0+5108
    ; sync on bad handle
    v13 = call fn5(v0, v4)
    store.i32 v13, v0+5112
    ; cursor_scan on bad handle
    v14 = iconst.i32 100
    v15 = iconst.i64 5300
    v16 = call fn6(v0, v4, v5, v6, v14, v15)
    store.i32 v16, v0+5116
    ; begin_write_txn on bad handle
    v17 = call fn7(v0, v4)
    store.i32 v17, v0+5120
    ; commit_write_txn on bad handle
    v18 = call fn8(v0, v4)
    store.i32 v18, v0+5124
    ; write 28 bytes of results
    v19 = iconst.i64 4256
    v20 = iconst.i64 5100
    v21 = iconst.i64 0
    v22 = iconst.i64 28
    v23 = call fn10(v0, v19, v20, v21, v22)
    call fn9(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000] = b'k';
    payloads[5010..5012].copy_from_slice(b"vv");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 28);
    let put_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let get_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    let del_ret = i32::from_le_bytes(contents[8..12].try_into().unwrap());
    let sync_ret = i32::from_le_bytes(contents[12..16].try_into().unwrap());
    let scan_ret = i32::from_le_bytes(contents[16..20].try_into().unwrap());
    let begin_ret = i32::from_le_bytes(contents[20..24].try_into().unwrap());
    let commit_ret = i32::from_le_bytes(contents[24..28].try_into().unwrap());
    assert_eq!(put_ret, -1, "put on invalid handle");
    assert_eq!(get_ret, -1, "get on invalid handle");
    assert_eq!(del_ret, -1, "delete on invalid handle");
    assert_eq!(sync_ret, -1, "sync on invalid handle");
    assert_eq!(scan_ret, 0, "scan on invalid handle returns 0 entries");
    assert_eq!(begin_ret, -1, "begin_write_txn on invalid handle");
    assert_eq!(commit_ret, -1, "commit_write_txn on invalid handle");
}

#[test]
fn test_clif_ffi_lmdb_double_begin() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_dblbegin");
    let verify_file = temp_dir.path().join("lmdb_dblbegin_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // begin_write_txn, put k1=v1, begin_write_txn again (aborts first),
    // put k2=v2, commit. k1 should be gone, k2 should exist.
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_begin_write_txn sig2
    fn3 = %cl_lmdb_put sig3
    fn4 = %cl_lmdb_commit_write_txn sig2
    fn5 = %cl_lmdb_get sig4
    fn6 = %cl_lmdb_cleanup sig0
    fn7 = %cl_file_write sig5
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    ; first begin
    v4 = call fn2(v0, v3)
    ; put k1=v1 in first txn
    v5 = iconst.i64 5000
    v6 = iconst.i32 2
    v7 = iconst.i64 5010
    v8 = call fn3(v0, v3, v5, v6, v7, v6)
    ; second begin — should abort the first txn (k1=v1 lost)
    v9 = call fn2(v0, v3)
    ; put k2=v2 in second txn
    v10 = iconst.i64 5020
    v11 = iconst.i64 5030
    v12 = call fn3(v0, v3, v10, v6, v11, v6)
    ; commit second txn
    v13 = call fn4(v0, v3)
    ; get k1 — should fail (-1)
    v14 = iconst.i64 5100
    v15 = call fn5(v0, v3, v5, v6, v14)
    store.i32 v15, v0+5200
    ; get k2 — should succeed
    v16 = iconst.i64 5300
    v17 = call fn5(v0, v3, v10, v6, v16)
    store.i32 v17, v0+5204
    ; write results
    v18 = iconst.i64 4256
    v19 = iconst.i64 5200
    v20 = iconst.i64 0
    v21 = iconst.i64 8
    v22 = call fn7(v0, v18, v19, v20, v21)
    ; write the k2 value
    v23 = iconst.i64 8
    v24 = iconst.i64 2
    v25 = iconst.i64 5304
    v26 = call fn7(v0, v18, v25, v23, v24)
    call fn6(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000..5002].copy_from_slice(b"k1");
    payloads[5010..5012].copy_from_slice(b"v1");
    payloads[5020..5022].copy_from_slice(b"k2");
    payloads[5030..5032].copy_from_slice(b"v2");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 10);
    let get_k1 = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let get_k2 = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(get_k1, -1, "k1 should be lost (first txn aborted by second begin)");
    assert_eq!(get_k2, 2, "k2 should exist with length 2");
    assert_eq!(&contents[8..10], b"v2");
}

#[test]
fn test_clif_ffi_lmdb_empty_batch() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_emptybatch");
    let verify_file = temp_dir.path().join("lmdb_emptybatch_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // begin_write_txn then immediately commit with no puts
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32) -> i32 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_begin_write_txn sig2
    fn3 = %cl_lmdb_commit_write_txn sig2
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig3
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = call fn2(v0, v3)
    store.i32 v4, v0+5000
    v5 = call fn3(v0, v3)
    store.i32 v5, v0+5004
    v6 = iconst.i64 4256
    v7 = iconst.i64 5000
    v8 = iconst.i64 0
    v9 = iconst.i64 8
    v10 = call fn5(v0, v6, v7, v8, v9)
    call fn4(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let begin_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let commit_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(begin_ret, 0, "begin empty batch should succeed");
    assert_eq!(commit_ret, 0, "commit empty batch should succeed");
}

#[test]
fn test_clif_ffi_lmdb_put_empty_value() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_emptyval");
    let verify_file = temp_dir.path().join("lmdb_emptyval_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Put key="k" with val_len=0, then get it back
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 1
    v6 = iconst.i64 5010
    v7 = iconst.i32 0
    ; put key="k" with empty value
    v8 = call fn2(v0, v3, v4, v5, v6, v7)
    store.i32 v8, v0+5100
    ; get it back
    v9 = iconst.i64 5200
    v10 = call fn3(v0, v3, v4, v5, v9)
    store.i32 v10, v0+5104
    ; write [put_ret, get_ret]
    v11 = iconst.i64 4256
    v12 = iconst.i64 5100
    v13 = iconst.i64 0
    v14 = iconst.i64 8
    v15 = call fn5(v0, v11, v12, v13, v14)
    call fn4(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000] = b'k';

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let put_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let get_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(put_ret, 0, "put with empty value should succeed");
    assert_eq!(get_ret, 0, "get should return length 0 for empty value");
}

#[test]
fn test_clif_ffi_lmdb_multiple_databases() {
    let temp_dir = TempDir::new().unwrap();
    let db1_path = temp_dir.path().join("clif_lmdb_multi1");
    let db2_path = temp_dir.path().join("clif_lmdb_multi2");
    let verify_file = temp_dir.path().join("lmdb_multi_verify.bin");
    let db1_path_str = format!("{}\0", db1_path.to_str().unwrap());
    let db2_path_str = format!("{}\0", db2_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Open two databases. Put key="k" in db1 with val="d1", in db2 with val="d2".
    // Get from each to verify isolation.
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    ; open db1
    v3 = call fn1(v0, v1, v2)
    ; open db2
    v4 = iconst.i64 4200
    v5 = call fn1(v0, v4, v2)
    ; key "k" at 5000, val "d1" at 5010, val "d2" at 5020
    v6 = iconst.i64 5000
    v7 = iconst.i32 1
    v8 = iconst.i64 5010
    v9 = iconst.i32 2
    ; put "k"="d1" in db1
    v10 = call fn2(v0, v3, v6, v7, v8, v9)
    ; put "k"="d2" in db2
    v11 = iconst.i64 5020
    v12 = call fn2(v0, v5, v6, v7, v11, v9)
    ; get from db1
    v13 = iconst.i64 5100
    v14 = call fn3(v0, v3, v6, v7, v13)
    store.i32 v14, v0+5200
    ; get from db2
    v15 = iconst.i64 5300
    v16 = call fn3(v0, v5, v6, v7, v15)
    store.i32 v16, v0+5204
    ; write [len1, len2, val1, val2]
    v17 = iconst.i64 4400
    v18 = iconst.i64 5200
    v19 = iconst.i64 0
    v20 = iconst.i64 8
    v21 = call fn5(v0, v17, v18, v19, v20)
    v22 = iconst.i64 8
    v23 = iconst.i64 2
    v24 = iconst.i64 5104
    v25 = call fn5(v0, v17, v24, v22, v23)
    v26 = iconst.i64 10
    v27 = iconst.i64 5304
    v28 = call fn5(v0, v17, v27, v26, v23)
    call fn4(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db1_path_str.len()].copy_from_slice(db1_path_str.as_bytes());
    payloads[4200..4200 + db2_path_str.len()].copy_from_slice(db2_path_str.as_bytes());
    payloads[4400..4400 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000] = b'k';
    payloads[5010..5012].copy_from_slice(b"d1");
    payloads[5020..5022].copy_from_slice(b"d2");

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 12);
    let len1 = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let len2 = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(len1, 2);
    assert_eq!(len2, 2);
    assert_eq!(&contents[8..10], b"d1", "db1 should have val d1");
    assert_eq!(&contents[10..12], b"d2", "db2 should have val d2");
}

#[test]
fn test_clif_ffi_lmdb_cursor_scan_max_entries_limit() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_scanlimit");
    let verify_file = temp_dir.path().join("lmdb_scanlimit_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Insert 5 keys (a-e), scan with max_entries=2
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_cursor_scan sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i32 1
    v20 = iconst.i64 5060
    v5 = iconst.i64 5000
    v6 = call fn2(v0, v3, v5, v4, v20, v4)
    v7 = iconst.i64 5001
    v8 = call fn2(v0, v3, v7, v4, v20, v4)
    v9 = iconst.i64 5002
    v10 = call fn2(v0, v3, v9, v4, v20, v4)
    v11 = iconst.i64 5003
    v12 = call fn2(v0, v3, v11, v4, v20, v4)
    v13 = iconst.i64 5004
    v14 = call fn2(v0, v3, v13, v4, v20, v4)
    ; scan all but limit to 2
    v15 = iconst.i32 0
    v16 = iconst.i32 2
    v17 = iconst.i64 5200
    v18 = iconst.i64 5100
    v19 = call fn3(v0, v3, v18, v15, v16, v17)
    store.i32 v19, v0+5300
    v21 = iconst.i64 4256
    v22 = iconst.i64 5300
    v23 = iconst.i64 0
    v24 = iconst.i64 4
    v25 = call fn5(v0, v21, v22, v23, v24)
    call fn4(v0)
    return
}
"#;

    let mut payloads = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    payloads[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    payloads[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads[5000] = b'a';
    payloads[5001] = b'b';
    payloads[5002] = b'c';
    payloads[5003] = b'd';
    payloads[5004] = b'e';
    payloads[5060] = b'x';

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_cranelift_algorithm(actions, payloads, 1, vec![0], false);
    execute(algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let count = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(count, 2, "scan with max_entries=2 should return exactly 2");
}

#[test]
fn test_clif_ffi_lmdb_uncommitted_batch_cleanup() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_uncommitted");
    let verify_file = temp_dir.path().join("lmdb_uncommitted_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // begin_write_txn, put a key, then cleanup WITHOUT committing.
    // Drop should abort the txn. Then reopen the db and verify key is missing.
    // We do this in two separate execute() calls.
    let clif_ir_1 =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_begin_write_txn sig2
    fn3 = %cl_lmdb_put sig3
    fn4 = %cl_lmdb_cleanup sig0
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = call fn2(v0, v3)
    v5 = iconst.i64 5000
    v6 = iconst.i32 3
    v7 = iconst.i64 5010
    v8 = call fn3(v0, v3, v5, v6, v7, v6)
    ; cleanup without commit — Drop should abort the active write txn
    call fn4(v0)
    return
}
"#;

    let mut payloads1 = vec![0u8; 8192];
    let clif_bytes1 = format!("{}\0", clif_ir_1).into_bytes();
    payloads1[0..clif_bytes1.len()].copy_from_slice(&clif_bytes1);
    payloads1[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads1[5000..5003].copy_from_slice(b"key");
    payloads1[5010..5013].copy_from_slice(b"val");

    let actions1 = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm1 = create_cranelift_algorithm(actions1, payloads1, 1, vec![0], false);
    execute(algorithm1).unwrap();

    // Second run: reopen db, try to get the key — should not exist
    let clif_ir_2 =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_get sig2
    fn3 = %cl_lmdb_cleanup sig0
    fn4 = %cl_file_write sig3
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 3
    v6 = iconst.i64 5100
    v7 = call fn2(v0, v3, v4, v5, v6)
    store.i32 v7, v0+5200
    v8 = iconst.i64 4256
    v9 = iconst.i64 5200
    v10 = iconst.i64 0
    v11 = iconst.i64 4
    v12 = call fn4(v0, v8, v9, v10, v11)
    call fn3(v0)
    return
}
"#;

    let mut payloads2 = vec![0u8; 8192];
    let clif_bytes2 = format!("{}\0", clif_ir_2).into_bytes();
    payloads2[0..clif_bytes2.len()].copy_from_slice(&clif_bytes2);
    payloads2[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads2[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    payloads2[5000..5003].copy_from_slice(b"key");

    let actions2 = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: 1024, size: 0 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let algorithm2 = create_cranelift_algorithm(actions2, payloads2, 1, vec![0], false);
    execute(algorithm2).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let get_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(get_ret, -1, "uncommitted key should not persist after cleanup");
}
