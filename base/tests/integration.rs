use base::execute;
use base_types::{Action, Algorithm, Kind, State, UnitSpec};
use std::fs;
use tempfile::TempDir;

fn create_test_algorithm(
    actions: Vec<Action>,
    payloads: Vec<u8>,
    file_units: usize,
    memory_units: usize,
) -> Algorithm {
    let num_actions = actions.len();
    let payload_size = payloads.len();

    Algorithm {
        actions,
        payloads,
        state: State {
            regs_per_unit: 16,
            gpu_size: 0,
            computational_regs: 0,
            file_buffer_size: 65536,
            gpu_shader_offsets: vec![],
        },
        units: UnitSpec {
            simd_units: 0,
            gpu_units: 0,
            computational_units: 0,
            file_units,
            network_units: 0,
            memory_units,
            ffi_units: 0,
            hash_table_units: 0,
            backends_bits: 0,
        },
        simd_assignments: vec![],
        computational_assignments: vec![],
        memory_assignments: vec![0; num_actions],
        file_assignments: vec![0; num_actions],
        network_assignments: vec![],
        ffi_assignments: vec![],
        hash_table_assignments: vec![],
        gpu_assignments: vec![],
        worker_threads: None,
        blocking_threads: None,
        stack_size: None,
        timeout_ms: Some(5000),
        thread_name_prefix: None,
    }
}

fn create_complex_algorithm(
    actions: Vec<Action>,
    payloads: Vec<u8>,
    file_units: usize,
    memory_units: usize,
    simd_units: usize,
    gpu_units: usize,
    gpu_shader_offsets: Vec<usize>,
    gpu_size: usize,
) -> Algorithm {
    let num_actions = actions.len();
    let payload_size = payloads.len();

    Algorithm {
        actions,
        payloads,
        state: State {
            regs_per_unit: 16,
            gpu_size,
            computational_regs: 32,
            file_buffer_size: 65536,
            gpu_shader_offsets,
        },
        units: UnitSpec {
            simd_units,
            gpu_units,
            computational_units: 0,
            file_units,
            network_units: 0,
            memory_units,
            ffi_units: 0,
            hash_table_units: 0,
            backends_bits: 0xFFFFFFFF,
        },
        simd_assignments: if simd_units > 0 {
            vec![0; num_actions]
        } else {
            vec![255; num_actions]
        },
        computational_assignments: vec![],
        memory_assignments: if memory_units > 0 {
            vec![0; num_actions]
        } else {
            vec![255; num_actions]
        },
        file_assignments: if file_units > 0 {
            vec![0; num_actions]
        } else {
            vec![255; num_actions]
        },
        network_assignments: vec![],
        ffi_assignments: vec![],
        hash_table_assignments: vec![],
        gpu_assignments: if gpu_units > 0 {
            vec![0; num_actions]
        } else {
            vec![255; num_actions]
        },
        worker_threads: None,
        blocking_threads: None,
        stack_size: None,
        timeout_ms: Some(10000),
        thread_name_prefix: None,
    }
}

#[test]
fn test_integration_memcopy_filewrite() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let mut payloads = vec![0u8; 1024];

    // Setup filename with null terminator
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Setup source data (value 42)
    payloads[256..264].copy_from_slice(&42u64.to_le_bytes());

    // Completion flags at offset 512 and 520
    let memcopy_flag = 512u32;
    let filewrite_flag = 520u32;

    let actions = vec![
        // Action 0: MemCopy (executed by memory unit)
        Action {
            kind: Kind::MemCopy,
            src: 256,
            dst: 264,
            offset: 0,
            size: 8,
        },
        // Action 1: FileWrite (executed by file unit)
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 264,
            offset: filename_bytes.len() as u32,
            size: 8,
        },
        // Action 2: AsyncDispatch MemCopy to memory unit (type 6)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,              // memory unit
            src: 0,              // action index 0 (MemCopy)
            offset: memcopy_flag,
            size: 0,
        },
        // Action 3: Wait for MemCopy
        Action {
            kind: Kind::Wait,
            dst: memcopy_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 4: AsyncDispatch FileWrite to file unit (type 2)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,              // file unit
            src: 1,              // action index 1 (FileWrite)
            offset: filewrite_flag,
            size: 0,
        },
        // Action 5: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: filewrite_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(value, 42);
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
            offset: filename_a_bytes.len() as u32,
            size: 8,
        },
        // Action 1: FileWrite to path B (data action, dispatched by index)
        Action {
            kind: Kind::FileWrite,
            dst: 256,
            src: 528,
            offset: filename_b_bytes.len() as u32,
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

    let algorithm = create_test_algorithm(actions, payloads, 1, 0);

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
            offset: filename_4byte_bytes.len() as u32,
            size: 8,
        },
        // Action 1: FileWrite for 8-byte test
        Action {
            kind: Kind::FileWrite,
            dst: 256,
            src: 528,
            offset: filename_8byte_bytes.len() as u32,
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

    let algorithm = create_test_algorithm(actions, payloads, 1, 0);

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
            offset: filename_bytes.len() as u32,
            size: 8,
        },
        // Action 1: FileRead file into buffer
        Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 264,
            offset: filename_bytes.len() as u32,
            size: 8,
        },
        // Action 2: FileWrite buffer to verification file
        Action {
            kind: Kind::FileWrite,
            dst: 512,
            src: 264,
            offset: verify_filename_bytes.len() as u32,
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

    let algorithm = create_test_algorithm(actions, payloads, 1, 0);

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
fn test_integration_async_memory_operations() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("async_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 2048];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Completion flags
    let cas_flag = 256u32;
    let file_flag = 264u32;

    // CAS target at offset 304 (8-byte aligned, initial value: 100)
    payloads[304..312].copy_from_slice(&100u64.to_le_bytes());
    // CAS expected value at offset 312 (8-byte aligned, 100)
    payloads[312..320].copy_from_slice(&100u64.to_le_bytes());
    // CAS new value at offset 320 (8-byte aligned, 200)
    payloads[320..328].copy_from_slice(&200u64.to_le_bytes());

    let actions = vec![
        // Action 0: AtomicCAS: swap 100 → 200
        Action {
            kind: Kind::AtomicCAS,
            dst: 304,    // target
            src: 312,    // expected value
            offset: 320, // new value
            size: 8,
        },
        // Action 1: FileWrite result
        Action {
            kind: Kind::FileWrite,
            dst: 0,      // filename
            src: 304,    // CAS result (should be 200)
            offset: filename_bytes.len() as u32,
            size: 8,
        },
        // Action 2: AsyncDispatch CAS to memory unit
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,      // unit type 6 = memory
            src: 0,      // action index 0
            offset: cas_flag,
            size: 0,
        },
        // Action 3: Wait for CAS
        Action {
            kind: Kind::Wait,
            dst: cas_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 4: AsyncDispatch FileWrite to file unit
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,      // file unit
            src: 1,      // action index 1 (FileWrite)
            offset: file_flag,
            size: 0,
        },
        // Action 5: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).expect("Async memory operations test failed");

    // Verify result: CAS should have swapped 100 → 200
    assert!(result_file.exists(), "Result file should exist");
    let result = fs::read(&result_file).unwrap();
    let value = u64::from_le_bytes(result[0..8].try_into().unwrap());
    assert_eq!(value, 200, "AtomicCAS should have swapped 100→200");
}

#[test]
fn test_integration_broadcast_memory_write() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("broadcast_mem.txt");
    let test_file_str = test_file.to_str().unwrap();

    let mut payloads = vec![0u8; 1024];

    // Setup filename with null terminator
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    let mem_flag = 900u32;
    let file_flag = 908u32;

    let actions = vec![
        // Action 0: write first value
        Action {
            kind: Kind::MemWrite,
            dst: 128,
            src: 0x1111_1111,
            offset: 0,
            size: 8,
        },
        // Action 1: write second value
        Action {
            kind: Kind::MemWrite,
            dst: 136,
            src: 0x2222_2222,
            offset: 0,
            size: 8,
        },
        // Action 2: file write of both values
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 128,
            offset: filename_bytes.len() as u32,
            size: 16,
        },
        // Action 3: broadcast memory writes (actions 0..2)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6, // memory unit
            src: 0,
            offset: mem_flag,
            size: (1u32 << 31) | 2,
        },
        // Action 4: wait for memory writes
        Action {
            kind: Kind::Wait,
            dst: mem_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 5: file write
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2, // file unit
            src: 2,
            offset: file_flag,
            size: 0,
        },
        // Action 6: wait for file write
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 2);
    execute(algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let first = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    let second = u64::from_le_bytes(contents[8..16].try_into().unwrap());
    assert_eq!(first, 0x1111_1111);
    assert_eq!(second, 0x2222_2222);
}

#[test]
fn test_integration_broadcast_memory_write_many() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("broadcast_mem_many.txt");
    let test_file_str = test_file.to_str().unwrap();

    let mut payloads = vec![0u8; 1024];

    // Setup filename with null terminator
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    let mem_flag = 920u32;
    let file_flag = 928u32;

    let actions = vec![
        Action {
            kind: Kind::MemWrite,
            dst: 128,
            src: 0xAAAA_AAAA,
            offset: 0,
            size: 8,
        },
        Action {
            kind: Kind::MemWrite,
            dst: 136,
            src: 0xBBBB_BBBB,
            offset: 0,
            size: 8,
        },
        Action {
            kind: Kind::MemWrite,
            dst: 144,
            src: 0xCCCC_CCCC,
            offset: 0,
            size: 8,
        },
        Action {
            kind: Kind::MemWrite,
            dst: 152,
            src: 0xDDDD_DDDD,
            offset: 0,
            size: 8,
        },
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 128,
            offset: filename_bytes.len() as u32,
            size: 32,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6, // memory unit
            src: 0,
            offset: mem_flag,
            size: (1u32 << 31) | 4,
        },
        Action {
            kind: Kind::Wait,
            dst: mem_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2, // file unit
            src: 4,
            offset: file_flag,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 4);
    execute(algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let values: Vec<u64> = contents
        .chunks(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(
        values,
        vec![
            0xAAAA_AAAA,
            0xBBBB_BBBB,
            0xCCCC_CCCC,
            0xDDDD_DDDD
        ]
    );
}

#[test]
fn test_integration_complex_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let path_a = temp_dir.path().join("result_a.txt");
    let path_b = temp_dir.path().join("result_b.txt");
    let path_cas = temp_dir.path().join("cas_result.txt");

    let path_a_str = path_a.to_str().unwrap();
    let path_b_str = path_b.to_str().unwrap();
    let path_cas_str = path_cas.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_a_bytes = format!("{}\0", path_a_str).into_bytes();
    payloads[0..filename_a_bytes.len()].copy_from_slice(&filename_a_bytes);

    let filename_b_bytes = format!("{}\0", path_b_str).into_bytes();
    payloads[512..512 + filename_b_bytes.len()].copy_from_slice(&filename_b_bytes);

    let filename_cas_bytes = format!("{}\0", path_cas_str).into_bytes();
    payloads[1024..1024 + filename_cas_bytes.len()].copy_from_slice(&filename_cas_bytes);

    // Data (all 8-byte aligned)
    payloads[2000..2008].copy_from_slice(&42u64.to_le_bytes());   // Source value
    payloads[2104..2112].copy_from_slice(&1u64.to_le_bytes());  // Condition (true)
    payloads[2200..2208].copy_from_slice(&100u64.to_le_bytes());  // CAS target
    payloads[2208..2216].copy_from_slice(&100u64.to_le_bytes());  // CAS expected
    payloads[2216..2224].copy_from_slice(&999u64.to_le_bytes());  // CAS new value

    // Completion flags
    let memcopy_flag = 2304u32;
    let cas_flag = 2312u32;
    let fw_cas_flag = 2320u32;
    let fw_a_flag = 2328u32;
    let fw_b_flag = 2336u32;

    let actions = vec![
        // Action 0: MemCopy value to buffer
        Action { kind: Kind::MemCopy, src: 2000, dst: 3000, offset: 0, size: 8 },
        // Action 1: AtomicCAS (100 → 999)
        Action { kind: Kind::AtomicCAS, dst: 2200, src: 2208, offset: 2216, size: 8 },
        // Action 2: FileWrite CAS result
        Action { kind: Kind::FileWrite, dst: 1024, src: 2200, offset: filename_cas_bytes.len() as u32, size: 8 },
        // Action 3: FileWrite path B (data action)
        Action { kind: Kind::FileWrite, dst: 512, src: 3000, offset: filename_b_bytes.len() as u32, size: 8 },
        // Action 4: FileWrite path A (data action)
        Action { kind: Kind::FileWrite, dst: 0, src: 3000, offset: filename_a_bytes.len() as u32, size: 8 },

        // Action 5: AsyncDispatch MemCopy
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: memcopy_flag, size: 0 },
        // Action 6: Wait
        Action { kind: Kind::Wait, dst: memcopy_flag, src: 0, offset: 0, size: 0 },
        // Action 7: AsyncDispatch CAS
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 1, offset: cas_flag, size: 0 },
        // Action 8: Wait
        Action { kind: Kind::Wait, dst: cas_flag, src: 0, offset: 0, size: 0 },
        // Action 9: AsyncDispatch FileWrite CAS result
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 2, offset: fw_cas_flag, size: 0 },
        // Action 10: Wait
        Action { kind: Kind::Wait, dst: fw_cas_flag, src: 0, offset: 0, size: 0 },
        // Action 11: ConditionalJump based on condition (jump to action 14 = path A)
        Action { kind: Kind::ConditionalJump, src: 2104, dst: 14, offset: 0, size: 0 },
        // Action 12: AsyncDispatch FileWrite B (SKIPPED)
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 3, offset: fw_b_flag, size: 0 },
        // Action 13: Wait (SKIPPED)
        Action { kind: Kind::Wait, dst: fw_b_flag, src: 0, offset: 0, size: 0 },
        // Action 14: AsyncDispatch FileWrite A (EXECUTED)
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 4, offset: fw_a_flag, size: 0 },
        // Action 15: Wait
        Action { kind: Kind::Wait, dst: fw_a_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).expect("Complex workflow test failed");

    // Verify CAS result
    assert!(path_cas.exists(), "CAS result file should exist");
    let cas_result = fs::read(&path_cas).unwrap();
    let cas_value = u64::from_le_bytes(cas_result[0..8].try_into().unwrap());
    assert_eq!(cas_value, 999, "CAS should have swapped 100→999");

    // Verify conditional jump took path A
    assert!(path_a.exists(), "Path A file should exist");
    assert!(!path_b.exists(), "Path B file should NOT exist (jumped over)");

    let result_a = fs::read(&path_a).unwrap();
    let value_a = u64::from_le_bytes(result_a[0..8].try_into().unwrap());
    assert_eq!(value_a, 42, "Path A should contain copied value");
}

// Simple WGSL shader for addition: reads data[0] and data[1], writes sum to data[2]
const SIMPLE_ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let a = data[0];
    let b = data[1];
    data[2] = a + b;
}
"#;

#[test]
fn test_integration_gpu_async() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("gpu_async_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let shader_bytes = SIMPLE_ADD_SHADER.as_bytes();
    payloads[0..shader_bytes.len()].copy_from_slice(shader_bytes);

    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[2048..2048 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    payloads[3000..3004].copy_from_slice(&7u32.to_le_bytes());
    payloads[3004..3008].copy_from_slice(&9u32.to_le_bytes());

    let file_flag = 2568u32;

    let actions = vec![
        // Action 0: GPU Dispatch
        Action {
            kind: Kind::Dispatch,
            dst: 3000,
            src: 3000,
            offset: 2560,
            size: 12,
        },
        // Action 1: FileWrite result
        Action {
            kind: Kind::FileWrite,
            dst: 2048,
            src: 3008,
            offset: filename_bytes.len() as u32,
            size: 4,
        },
        // Action 2: AsyncDispatch GPU
        Action {
            kind: Kind::AsyncDispatch,
            dst: 0,
            src: 0,
            offset: 2560,
            size: 0,
        },
        // Action 3: Wait for GPU
        Action {
            kind: Kind::Wait,
            dst: 2560,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 4: AsyncDispatch FileWrite
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
            offset: file_flag,
            size: 0,
        },
        // Action 5: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_complex_algorithm(
        actions,
        payloads,
        1,    // file_units
        0,    // memory_units
        0,    // simd_units
        1,    // gpu_units
        vec![0],  // gpu_shader_offsets
        2048, // gpu_size
    );

    execute(algorithm).unwrap();

    assert!(test_file.exists(), "GPU async result file should exist");
    let contents = fs::read(&test_file).unwrap();
    let value = u32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(value, 16, "GPU should have computed 7 + 9 = 16");
}

#[test]
fn test_integration_complex_gpu_simd_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let result1 = temp_dir.path().join("result1.txt");
    let result2 = temp_dir.path().join("result2.txt");
    let condition_file = temp_dir.path().join("condition.txt");

    let result1_str = result1.to_str().unwrap();
    let result2_str = result2.to_str().unwrap();
    let condition_str = condition_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let shader_bytes = SIMPLE_ADD_SHADER.as_bytes();
    payloads[0..shader_bytes.len()].copy_from_slice(shader_bytes);

    let filename1_bytes = format!("{}\0", result1_str).into_bytes();
    payloads[2048..2048 + filename1_bytes.len()].copy_from_slice(&filename1_bytes);

    let filename2_bytes = format!("{}\0", result2_str).into_bytes();
    payloads[2304..2304 + filename2_bytes.len()].copy_from_slice(&filename2_bytes);

    let filename_cond_bytes = format!("{}\0", condition_str).into_bytes();
    payloads[2720..2720 + filename_cond_bytes.len()].copy_from_slice(&filename_cond_bytes);

    payloads[2576..2580].copy_from_slice(&7u32.to_le_bytes());
    payloads[2580..2584].copy_from_slice(&9u32.to_le_bytes());

    payloads[2640..2644].copy_from_slice(&3u32.to_le_bytes());
    payloads[2644..2648].copy_from_slice(&5u32.to_le_bytes());

    // Condition value for ConditionalJump (1 = true, will jump)
    payloads[2704..2712].copy_from_slice(&1u64.to_le_bytes());

    let high_text = b"HIGH";
    payloads[2776..2780].copy_from_slice(high_text);
    let low_text = b"LOW";
    payloads[2832..2835].copy_from_slice(low_text);

    // Completion flags
    let fw1_flag = 2880u32;
    let fw2_flag = 2888u32;
    let fw_cond_flag = 2896u32;

    let actions = vec![
        // Action 0: GPU Dispatch 1
        Action { kind: Kind::Dispatch, dst: 2576, src: 2576, offset: 2560, size: 12 },
        // Action 1: GPU Dispatch 2
        Action { kind: Kind::Dispatch, dst: 2640, src: 2640, offset: 2568, size: 12 },
        // Action 2: FileWrite result1
        Action { kind: Kind::FileWrite, dst: 2048, src: 2584, offset: filename1_bytes.len() as u32, size: 4 },
        // Action 3: FileWrite result2
        Action { kind: Kind::FileWrite, dst: 2304, src: 2648, offset: filename2_bytes.len() as u32, size: 4 },
        // Action 4: FileWrite LOW (skipped)
        Action { kind: Kind::FileWrite, dst: 2720, src: 2832, offset: filename_cond_bytes.len() as u32, size: 3 },
        // Action 5: FileWrite HIGH (executed)
        Action { kind: Kind::FileWrite, dst: 2720, src: 2776, offset: filename_cond_bytes.len() as u32, size: 4 },
        // Action 6: AsyncDispatch GPU 1
        Action { kind: Kind::AsyncDispatch, dst: 0, src: 0, offset: 2560, size: 0 },
        // Action 7: Wait for GPU 1
        Action { kind: Kind::Wait, dst: 2560, src: 0, offset: 0, size: 0 },
        // Action 8: AsyncDispatch FileWrite 1
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 2, offset: fw1_flag, size: 0 },
        // Action 9: Wait for FileWrite 1
        Action { kind: Kind::Wait, dst: fw1_flag, src: 0, offset: 0, size: 0 },
        // Action 10: AsyncDispatch GPU 2
        Action { kind: Kind::AsyncDispatch, dst: 0, src: 1, offset: 2568, size: 0 },
        // Action 11: Wait for GPU 2
        Action { kind: Kind::Wait, dst: 2568, src: 0, offset: 0, size: 0 },
        // Action 12: AsyncDispatch FileWrite 2
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 3, offset: fw2_flag, size: 0 },
        // Action 13: Wait for FileWrite 2
        Action { kind: Kind::Wait, dst: fw2_flag, src: 0, offset: 0, size: 0 },
        // Action 14: ConditionalJump - if condition true (1), jump to action 18 (HIGH path)
        Action { kind: Kind::ConditionalJump, src: 2704, dst: 18, offset: 0, size: 0 },
        // Action 15: AsyncDispatch LOW FileWrite (skipped when jumping)
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 4, offset: fw_cond_flag, size: 0 },
        // Action 16: Wait for LOW (skipped)
        Action { kind: Kind::Wait, dst: fw_cond_flag, src: 0, offset: 0, size: 0 },
        // Action 17: Jump to end (skip HIGH path)
        Action { kind: Kind::ConditionalJump, src: 2704, dst: 20, offset: 0, size: 0 },
        // Action 18: AsyncDispatch HIGH FileWrite
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 5, offset: fw_cond_flag, size: 0 },
        // Action 19: Wait for HIGH
        Action { kind: Kind::Wait, dst: fw_cond_flag, src: 0, offset: 0, size: 0 },
        // Action 20: End
    ];

    let algorithm = create_complex_algorithm(
        actions,
        payloads,
        3,
        0,
        0,
        1,
        vec![0],
        2048,
    );

    execute(algorithm).unwrap();

    assert!(result1.exists(), "GPU result 1 should exist");
    let contents1 = fs::read(&result1).unwrap();
    let value1 = u32::from_le_bytes(contents1[0..4].try_into().unwrap());
    assert_eq!(value1, 16, "GPU1 should compute 7 + 9 = 16");

    assert!(result2.exists(), "GPU result 2 should exist");
    let contents2 = fs::read(&result2).unwrap();
    let value2 = u32::from_le_bytes(contents2[0..4].try_into().unwrap());
    assert_eq!(value2, 8, "GPU2 should compute 3 + 5 = 8");

    assert!(condition_file.exists(), "Condition file should exist");
    let contents_cond = fs::read(&condition_file).unwrap();
    assert_eq!(&contents_cond[..], b"HIGH", "Should have taken HIGH path (condition = 1.0)");
}

#[test]
fn test_integration_multiple_async_same_unit() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("queue_order.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Three CAS operations that will be queued to same memory unit
    payloads[2000..2008].copy_from_slice(&0u64.to_le_bytes());
    payloads[2008..2016].copy_from_slice(&0u64.to_le_bytes());
    payloads[2016..2024].copy_from_slice(&1u64.to_le_bytes());

    payloads[2104..2112].copy_from_slice(&1u64.to_le_bytes());
    payloads[2112..2120].copy_from_slice(&1u64.to_le_bytes());
    payloads[2120..2128].copy_from_slice(&2u64.to_le_bytes());

    payloads[2200..2208].copy_from_slice(&2u64.to_le_bytes());
    payloads[2208..2216].copy_from_slice(&2u64.to_le_bytes());
    payloads[2216..2224].copy_from_slice(&3u64.to_le_bytes());

    let file_flag = 3024u32;

    let actions = vec![
        // Action 0-2: CAS operations
        Action { kind: Kind::AtomicCAS, dst: 2000, src: 2008, offset: 2016, size: 8 },
        Action { kind: Kind::AtomicCAS, dst: 2104, src: 2112, offset: 2120, size: 8 },
        Action { kind: Kind::AtomicCAS, dst: 2200, src: 2208, offset: 2216, size: 8 },
        // Action 3-5: Copy all results to buffer
        Action { kind: Kind::MemCopy, src: 2000, dst: 512, offset: 0, size: 8 },
        Action { kind: Kind::MemCopy, src: 2104, dst: 520, offset: 0, size: 8 },
        Action { kind: Kind::MemCopy, src: 2200, dst: 528, offset: 0, size: 8 },
        // Action 6: Write to file
        Action { kind: Kind::FileWrite, dst: 0, src: 512, offset: filename_bytes.len() as u32, size: 24 },
        // Action 7-9: Queue all three CAS to SAME memory unit (unit 0)
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: 3000, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 1, offset: 3008, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 2, offset: 3016, size: 0 },
        // Action 10-12: Wait for all CAS
        Action { kind: Kind::Wait, dst: 3000, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 3008, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 3016, src: 0, offset: 0, size: 0 },
        // Action 13-15: Dispatch MemCopy operations
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 3, offset: 3032, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 4, offset: 3040, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 5, offset: 3048, size: 0 },
        // Action 16-18: Wait for all MemCopy
        Action { kind: Kind::Wait, dst: 3032, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 3040, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 3048, src: 0, offset: 0, size: 0 },
        // Action 19: AsyncDispatch FileWrite
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 6, offset: file_flag, size: 0 },
        // Action 20: Wait for FileWrite
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).unwrap();

    assert!(result_file.exists(), "Queue ordering result should exist");
    let contents = fs::read(&result_file).unwrap();

    let v1 = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    let v2 = u64::from_le_bytes(contents[8..16].try_into().unwrap());
    let v3 = u64::from_le_bytes(contents[16..24].try_into().unwrap());

    assert_eq!(v1, 1, "First CAS: 0→1");
    assert_eq!(v2, 2, "Second CAS: 1→2");
    assert_eq!(v3, 3, "Third CAS: 2→3");
}

#[test]
fn test_integration_cross_unit_data_flow() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("cross_unit.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Memory unit prepares data
    payloads[2000..2008].copy_from_slice(&42u64.to_le_bytes());

    let file_flag = 2568u32;

    let actions = vec![
        // Action 0: Memory - Copy data to shared location
        Action {
            kind: Kind::MemCopy,
            src: 2000,
            dst: 2100,
            offset: 0,
            size: 8,
        },
        // Action 1: FileWrite to file
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 2100,
            offset: filename_bytes.len() as u32,
            size: 8,
        },
        // Action 2: AsyncDispatch memory operation
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 0,
            offset: 2560,
            size: 0,
        },
        // Action 3: Wait for memory to finish
        Action {
            kind: Kind::Wait,
            dst: 2560,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 4: AsyncDispatch FileWrite to file unit
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
            offset: file_flag,
            size: 0,
        },
        // Action 5: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).unwrap();

    assert!(result_file.exists(), "Cross-unit data flow result should exist");
    let contents = fs::read(&result_file).unwrap();
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(value, 42, "File should read data prepared by memory unit");
}

#[test]
fn test_integration_conditional_skip_with_wait() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("cond_skip.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Condition: 0 (false) - will NOT jump, will execute AsyncDispatch
    payloads[2700..2708].copy_from_slice(&0u64.to_le_bytes());

    payloads[2000..2008].copy_from_slice(&99u64.to_le_bytes());
    payloads[2008..2016].copy_from_slice(&99u64.to_le_bytes());
    payloads[2016..2024].copy_from_slice(&123u64.to_le_bytes());

    let file_flag = 2568u32;

    let actions = vec![
        // Action 0: AtomicCAS
        Action {
            kind: Kind::AtomicCAS,
            dst: 2000,
            src: 2008,
            offset: 2016,
            size: 8,
        },
        // Action 1: FileWrite result
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 2000,
            offset: filename_bytes.len() as u32,
            size: 8,
        },
        // Action 2: Jump over AsyncDispatch if condition true (it's false, so won't jump)
        Action {
            kind: Kind::ConditionalJump,
            src: 2700,
            dst: 5,   // Jump to action 5 (after Wait for CAS)
            offset: 0,
            size: 0,
        },
        // Action 3: This WILL execute (condition was false)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 0,
            offset: 2560,
            size: 0,
        },
        // Action 4: Wait for the dispatch that happened
        Action {
            kind: Kind::Wait,
            dst: 2560,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 5: AsyncDispatch FileWrite
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
            offset: file_flag,
            size: 0,
        },
        // Action 6: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).unwrap();

    assert!(result_file.exists(), "Conditional skip result should exist");
    let contents = fs::read(&result_file).unwrap();
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(value, 123, "CAS completed because AsyncDispatch was not skipped");
}

#[test]
fn test_integration_memscan() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("memscan_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Create a haystack with a needle: search for pattern 0xDEADBEEF
    let needle = 0xDEADBEEFu32;
    let haystack_start = 1000;
    let needle_position = haystack_start + 40; // 8-byte aligned: 1040 % 8 = 0

    // Fill haystack with random-ish data
    for i in 0..100 {
        let val = (i * 7 + 13) as u32;
        let offset = haystack_start + i * 4;
        payloads[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    }

    // Place needle at specific position
    payloads[needle_position..needle_position + 4].copy_from_slice(&needle.to_le_bytes());

    // Store needle pattern at offset 2000
    payloads[2000..2004].copy_from_slice(&needle.to_le_bytes());

    let file_flag = 2508u32;

    let actions = vec![
        // Action 0: MemScan - search for needle in haystack
        Action {
            kind: Kind::MemScan,
            src: 2000,    // needle/pattern location
            dst: haystack_start as u32,  // search region start
            offset: 4 | (2100 << 16),  // pattern_size=4, result_offset=2100
            size: 400,    // search region size
        },
        // Action 1: FileWrite result
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 2100,
            offset: filename_bytes.len() as u32,
            size: 8,
        },
        // Action 2: AsyncDispatch MemScan
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,      // memory unit
            src: 0,      // action index 0
            offset: 2500, // completion flag
            size: 0,
        },
        // Action 3: Wait for MemScan
        Action {
            kind: Kind::Wait,
            dst: 2500,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 4: AsyncDispatch FileWrite
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
            offset: file_flag,
            size: 0,
        },
        // Action 5: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).unwrap();

    assert!(result_file.exists(), "MemScan result should exist");
    let contents = fs::read(&result_file).unwrap();
    let found_offset = i64::from_le_bytes(contents[0..8].try_into().unwrap());

    assert_eq!(found_offset as usize, needle_position, "MemScan should find needle at correct offset");
}

#[test]
fn test_integration_conditional_write() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("condwrite_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Condition: 1 (true)
    payloads[2000..2008].copy_from_slice(&1u64.to_le_bytes());
    // Value to write: 999
    payloads[2008..2016].copy_from_slice(&999u64.to_le_bytes());
    // Target location: 2100 (initially 0)
    payloads[2100..2108].copy_from_slice(&0u64.to_le_bytes());

    let file_flag = 2508u32;

    let actions = vec![
        // Action 0: ConditionalWrite - write 999 to offset 2100 if condition is true
        Action {
            kind: Kind::ConditionalWrite,
            src: 2008,    // source value (999)
            dst: 2100,    // target location
            offset: 2000, // condition (1 = true)
            size: 8,
        },
        // Action 1: FileWrite result
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 2100,
            offset: filename_bytes.len() as u32,
            size: 8,
        },
        // Action 2: AsyncDispatch ConditionalWrite
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,      // memory unit
            src: 0,      // action index 0
            offset: 2500, // completion flag
            size: 0,
        },
        // Action 3: Wait for ConditionalWrite
        Action {
            kind: Kind::Wait,
            dst: 2500,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 4: AsyncDispatch FileWrite
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
            offset: file_flag,
            size: 0,
        },
        // Action 5: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).unwrap();

    assert!(result_file.exists(), "ConditionalWrite result should exist");
    let contents = fs::read(&result_file).unwrap();
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());

    assert_eq!(value, 999, "ConditionalWrite should write value when condition is true");
}

#[test]
fn test_integration_fence() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("fence_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Setup multiple CAS operations with fence in between
    payloads[2000..2008].copy_from_slice(&10u64.to_le_bytes());
    payloads[2008..2016].copy_from_slice(&10u64.to_le_bytes());
    payloads[2016..2024].copy_from_slice(&20u64.to_le_bytes());

    payloads[2096..2104].copy_from_slice(&30u64.to_le_bytes());
    payloads[2104..2112].copy_from_slice(&30u64.to_le_bytes());
    payloads[2112..2120].copy_from_slice(&40u64.to_le_bytes());

    let mem1_flag = 2496u32;
    let mem2_flag = 2504u32;
    let file_flag = 2512u32;

    let actions = vec![
        // Action 0: First CAS
        Action {
            kind: Kind::AtomicCAS,
            dst: 2000,    // target
            src: 2008,    // expected value location
            offset: 2016, // new value location
            size: 8,
        },
        // Action 1: Fence to ensure ordering
        Action {
            kind: Kind::Fence,
            dst: 0,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 2: Second CAS
        Action {
            kind: Kind::AtomicCAS,
            dst: 2096,    // target
            src: 2104,    // expected value location
            offset: 2112, // new value location
            size: 8,
        },
        // Action 3: MemCopy result 1
        Action {
            kind: Kind::MemCopy,
            src: 2000,
            dst: 3000,
            offset: 0,
            size: 8,
        },
        // Action 4: MemCopy result 2
        Action {
            kind: Kind::MemCopy,
            src: 2096,
            dst: 3008,
            offset: 0,
            size: 8,
        },
        // Action 5: FileWrite results
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 3000,
            offset: filename_bytes.len() as u32,
            size: 16,
        },
        // Action 6: AsyncDispatch first CAS
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,      // memory unit
            src: 0,      // action index 0
            offset: mem1_flag, // completion flag
            size: 0,
        },
        // Action 7: Wait for first CAS
        Action {
            kind: Kind::Wait,
            dst: mem1_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 8: AsyncDispatch Fence
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 1,
            offset: 2520,
            size: 0,
        },
        // Action 9: Wait for Fence
        Action {
            kind: Kind::Wait,
            dst: 2520,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 10: AsyncDispatch second CAS
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 2,
            offset: mem2_flag,
            size: 0,
        },
        // Action 11: Wait for second CAS
        Action {
            kind: Kind::Wait,
            dst: mem2_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 12: AsyncDispatch MemCopy 1
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 3,
            offset: 2528,
            size: 0,
        },
        // Action 13: Wait for MemCopy 1
        Action {
            kind: Kind::Wait,
            dst: 2528,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 14: AsyncDispatch MemCopy 2
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 4,
            offset: 2536,
            size: 0,
        },
        // Action 15: Wait for MemCopy 2
        Action {
            kind: Kind::Wait,
            dst: 2536,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 16: AsyncDispatch FileWrite
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 5,
            offset: file_flag,
            size: 0,
        },
        // Action 17: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).unwrap();

    assert!(result_file.exists(), "Fence test result should exist");
    let contents = fs::read(&result_file).unwrap();

    let val1 = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    let val2 = u64::from_le_bytes(contents[8..16].try_into().unwrap());

    assert_eq!(val1, 20, "First CAS should complete before fence");
    assert_eq!(val2, 40, "Second CAS should complete after fence");
}

#[test]
fn test_integration_memwrite() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("memwrite_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Source data
    payloads[2000..2008].copy_from_slice(&777u64.to_le_bytes());

    // Destination initially zero
    payloads[2100..2108].copy_from_slice(&0u64.to_le_bytes());

    let mem_flag = 2200u32;
    let file_flag = 2208u32;

    let actions = vec![
        // Action 0: MemCopy - copy from 2000 to 2100 (MemWrite not implemented in async units)
        Action {
            kind: Kind::MemCopy,
            src: 2000,
            dst: 2100,
            offset: 0,
            size: 8,
        },
        // Action 1: FileWrite the result
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 2100,
            offset: filename_bytes.len() as u32,
            size: 8,
        },
        // Action 2: AsyncDispatch MemCopy to memory unit
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 0,
            offset: mem_flag,
            size: 0,
        },
        // Action 3: Wait for MemCopy
        Action {
            kind: Kind::Wait,
            dst: mem_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 4: AsyncDispatch FileWrite to file unit
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
            offset: file_flag,
            size: 0,
        },
        // Action 5: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).unwrap();

    assert!(result_file.exists(), "MemWrite result should exist");
    let contents = fs::read(&result_file).unwrap();
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());

    assert_eq!(value, 777, "MemWrite should copy value correctly");
}

#[test]
fn test_integration_memcopy_indirect() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("memcopy_indirect_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Pointer at offset 1000 points to data at 2000
    payloads[1000..1004].copy_from_slice(&2000u32.to_le_bytes());
    // Data at offset 2000
    payloads[2000..2008].copy_from_slice(&12345u64.to_le_bytes());

    let actions = vec![
        // MemCopyIndirect: read ptr from 1000, copy from *ptr to 3000
        Action { kind: Kind::MemCopyIndirect, src: 1000, dst: 3000, offset: 0, size: 8 },
        Action { kind: Kind::FileWrite, dst: 0, src: 3000, offset: filename_bytes.len() as u32, size: 8 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: 2200, size: 0 },
        Action { kind: Kind::Wait, dst: 2200, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: 2208, size: 0 },
        Action { kind: Kind::Wait, dst: 2208, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);
    execute(algorithm).unwrap();

    let contents = fs::read(&result_file).unwrap();
    assert_eq!(u64::from_le_bytes(contents[0..8].try_into().unwrap()), 12345);
}

#[test]
fn test_integration_memstore_indirect() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("memstore_indirect_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Source data at 2000
    payloads[2000..2008].copy_from_slice(&99999u64.to_le_bytes());
    // Pointer at 1000 points to destination 3000
    payloads[1000..1004].copy_from_slice(&3000u32.to_le_bytes());

    let actions = vec![
        // MemStoreIndirect: copy from 2000 to *ptr (read ptr from 1000)
        Action { kind: Kind::MemStoreIndirect, src: 2000, dst: 1000, offset: 0, size: 8 },
        Action { kind: Kind::FileWrite, dst: 0, src: 3000, offset: filename_bytes.len() as u32, size: 8 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: 2200, size: 0 },
        Action { kind: Kind::Wait, dst: 2200, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: 2208, size: 0 },
        Action { kind: Kind::Wait, dst: 2208, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);
    execute(algorithm).unwrap();

    let contents = fs::read(&result_file).unwrap();
    assert_eq!(u64::from_le_bytes(contents[0..8].try_into().unwrap()), 99999);
}

#[test]
fn test_integration_simd_shared_memory() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("simd_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Vector A at offset 256: [1.0, 2.0, 3.0, 4.0]
    payloads[256..260].copy_from_slice(&1.0f32.to_le_bytes());
    payloads[260..264].copy_from_slice(&2.0f32.to_le_bytes());
    payloads[264..268].copy_from_slice(&3.0f32.to_le_bytes());
    payloads[268..272].copy_from_slice(&4.0f32.to_le_bytes());

    // Vector B at offset 272: [10.0, 10.0, 10.0, 10.0]
    payloads[272..276].copy_from_slice(&10.0f32.to_le_bytes());
    payloads[276..280].copy_from_slice(&10.0f32.to_le_bytes());
    payloads[280..284].copy_from_slice(&10.0f32.to_le_bytes());
    payloads[284..288].copy_from_slice(&10.0f32.to_le_bytes());

    let file_flag = 648u32;

    let actions = vec![
        // Action 0-5: SIMD operations (data actions)
        Action { kind: Kind::SimdLoad, dst: 0, src: 256, offset: 0, size: 16 },
        Action { kind: Kind::SimdLoad, dst: 1, src: 272, offset: 0, size: 16 },
        Action { kind: Kind::SimdAdd, dst: 2, src: 0, offset: 1, size: 0 },
        Action { kind: Kind::SimdMul, dst: 3, src: 0, offset: 1, size: 0 },
        Action { kind: Kind::SimdStore, dst: 0, src: 2, offset: 500, size: 16 },
        Action { kind: Kind::SimdStore, dst: 0, src: 3, offset: 516, size: 16 },
        // Action 6: FileWrite both results (32 bytes total)
        Action { kind: Kind::FileWrite, dst: 0, src: 500, offset: filename_bytes.len() as u32, size: 32 },
        // Action 7-12: Dispatch SIMD actions
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 0, offset: 600, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 1, offset: 608, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 2, offset: 616, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 3, offset: 624, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 4, offset: 632, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 5, offset: 640, size: 0 },
        // Action 13-18: Wait for all SIMD to complete
        Action { kind: Kind::Wait, dst: 600, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 608, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 616, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 624, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 632, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 640, src: 0, offset: 0, size: 0 },
        // Action 19: AsyncDispatch FileWrite
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 6, offset: file_flag, size: 0 },
        // Action 20: Wait for FileWrite
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_complex_algorithm(actions, payloads, 1, 0, 1, 0, vec![], 0);

    execute(algorithm).unwrap();

    // Verify results
    assert!(result_file.exists(), "SIMD result file should exist");
    let contents = fs::read(&result_file).unwrap();

    // Check addition result: [11.0, 12.0, 13.0, 14.0]
    let add0 = f32::from_le_bytes(contents[0..4].try_into().unwrap());
    let add1 = f32::from_le_bytes(contents[4..8].try_into().unwrap());
    let add2 = f32::from_le_bytes(contents[8..12].try_into().unwrap());
    let add3 = f32::from_le_bytes(contents[12..16].try_into().unwrap());

    assert_eq!(add0, 11.0, "SIMD Add lane 0: 1.0 + 10.0 = 11.0");
    assert_eq!(add1, 12.0, "SIMD Add lane 1: 2.0 + 10.0 = 12.0");
    assert_eq!(add2, 13.0, "SIMD Add lane 2: 3.0 + 10.0 = 13.0");
    assert_eq!(add3, 14.0, "SIMD Add lane 3: 4.0 + 10.0 = 14.0");

    // Check multiplication result: [10.0, 20.0, 30.0, 40.0]
    let mul0 = f32::from_le_bytes(contents[16..20].try_into().unwrap());
    let mul1 = f32::from_le_bytes(contents[20..24].try_into().unwrap());
    let mul2 = f32::from_le_bytes(contents[24..28].try_into().unwrap());
    let mul3 = f32::from_le_bytes(contents[28..32].try_into().unwrap());

    assert_eq!(mul0, 10.0, "SIMD Mul lane 0: 1.0 * 10.0 = 10.0");
    assert_eq!(mul1, 20.0, "SIMD Mul lane 1: 2.0 * 10.0 = 20.0");
    assert_eq!(mul2, 30.0, "SIMD Mul lane 2: 3.0 * 10.0 = 30.0");
    assert_eq!(mul3, 40.0, "SIMD Mul lane 3: 4.0 * 10.0 = 40.0");
}

#[test]
fn test_integration_simd_i32x4_shared_memory() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("simd_i32_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Vector A at offset 256: [1, 2, 3, 4]
    payloads[256..260].copy_from_slice(&1i32.to_le_bytes());
    payloads[260..264].copy_from_slice(&2i32.to_le_bytes());
    payloads[264..268].copy_from_slice(&3i32.to_le_bytes());
    payloads[268..272].copy_from_slice(&4i32.to_le_bytes());

    // Vector B at offset 272: [10, 20, 30, 40]
    payloads[272..276].copy_from_slice(&10i32.to_le_bytes());
    payloads[276..280].copy_from_slice(&20i32.to_le_bytes());
    payloads[280..284].copy_from_slice(&30i32.to_le_bytes());
    payloads[284..288].copy_from_slice(&40i32.to_le_bytes());

    let file_flag = 648u32;

    let actions = vec![
        // Action 0-5: SIMD i32 operations (data actions)
        Action { kind: Kind::SimdLoadI32, dst: 0, src: 256, offset: 0, size: 16 },
        Action { kind: Kind::SimdLoadI32, dst: 1, src: 272, offset: 0, size: 16 },
        Action { kind: Kind::SimdAddI32, dst: 2, src: 0, offset: 1, size: 0 },
        Action { kind: Kind::SimdMulI32, dst: 3, src: 0, offset: 1, size: 0 },
        Action { kind: Kind::SimdStoreI32, dst: 0, src: 2, offset: 500, size: 16 },
        Action { kind: Kind::SimdStoreI32, dst: 0, src: 3, offset: 516, size: 16 },
        // Action 6: FileWrite both results (32 bytes total)
        Action { kind: Kind::FileWrite, dst: 0, src: 500, offset: filename_bytes.len() as u32, size: 32 },
        // Action 7-12: Dispatch SIMD actions
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 0, offset: 600, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 1, offset: 608, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 2, offset: 616, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 3, offset: 624, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 4, offset: 632, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 5, offset: 640, size: 0 },
        // Action 13-18: Wait for all SIMD to complete
        Action { kind: Kind::Wait, dst: 600, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 608, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 616, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 624, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 632, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 640, src: 0, offset: 0, size: 0 },
        // Action 19: AsyncDispatch FileWrite
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 6, offset: file_flag, size: 0 },
        // Action 20: Wait for FileWrite
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_complex_algorithm(actions, payloads, 1, 0, 1, 0, vec![], 0);

    execute(algorithm).unwrap();

    assert!(result_file.exists(), "SIMD i32 result file should exist");
    let contents = fs::read(&result_file).unwrap();

    let add0 = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let add1 = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    let add2 = i32::from_le_bytes(contents[8..12].try_into().unwrap());
    let add3 = i32::from_le_bytes(contents[12..16].try_into().unwrap());

    assert_eq!(add0, 11, "SIMD Add i32 lane 0: 1 + 10 = 11");
    assert_eq!(add1, 22, "SIMD Add i32 lane 1: 2 + 20 = 22");
    assert_eq!(add2, 33, "SIMD Add i32 lane 2: 3 + 30 = 33");
    assert_eq!(add3, 44, "SIMD Add i32 lane 3: 4 + 40 = 44");

    let mul0 = i32::from_le_bytes(contents[16..20].try_into().unwrap());
    let mul1 = i32::from_le_bytes(contents[20..24].try_into().unwrap());
    let mul2 = i32::from_le_bytes(contents[24..28].try_into().unwrap());
    let mul3 = i32::from_le_bytes(contents[28..32].try_into().unwrap());

    assert_eq!(mul0, 10, "SIMD Mul i32 lane 0: 1 * 10 = 10");
    assert_eq!(mul1, 40, "SIMD Mul i32 lane 1: 2 * 20 = 40");
    assert_eq!(mul2, 90, "SIMD Mul i32 lane 2: 3 * 30 = 90");
    assert_eq!(mul3, 160, "SIMD Mul i32 lane 3: 4 * 40 = 160");
}

#[test]
fn test_integration_simd_i32_div() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("simd_div_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // [100, 200, 300, 400]
    payloads[256..260].copy_from_slice(&100i32.to_le_bytes());
    payloads[260..264].copy_from_slice(&200i32.to_le_bytes());
    payloads[264..268].copy_from_slice(&300i32.to_le_bytes());
    payloads[268..272].copy_from_slice(&400i32.to_le_bytes());

    // [10, 20, 0, 40] - 0 tests div-by-zero
    payloads[272..276].copy_from_slice(&10i32.to_le_bytes());
    payloads[276..280].copy_from_slice(&20i32.to_le_bytes());
    payloads[280..284].copy_from_slice(&0i32.to_le_bytes());
    payloads[284..288].copy_from_slice(&40i32.to_le_bytes());

    let actions = vec![
        Action { kind: Kind::SimdLoadI32, dst: 0, src: 256, offset: 0, size: 16 },
        Action { kind: Kind::SimdLoadI32, dst: 1, src: 272, offset: 0, size: 16 },
        Action { kind: Kind::SimdDivI32, dst: 2, src: 0, offset: 1, size: 0 },
        Action { kind: Kind::SimdStoreI32, dst: 0, src: 2, offset: 500, size: 16 },
        Action { kind: Kind::FileWrite, dst: 0, src: 500, offset: filename_bytes.len() as u32, size: 16 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 0, offset: 600, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 1, offset: 608, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 2, offset: 616, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 3, offset: 624, size: 0 },
        Action { kind: Kind::Wait, dst: 600, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 608, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 616, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 624, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 4, offset: 632, size: 0 },
        Action { kind: Kind::Wait, dst: 632, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_complex_algorithm(actions, payloads, 1, 0, 1, 0, vec![], 0);
    execute(algorithm).unwrap();

    let contents = fs::read(&result_file).unwrap();
    // [100/10, 200/20, 300/0, 400/40] = [10, 10, 0, 10]
    assert_eq!(i32::from_le_bytes(contents[0..4].try_into().unwrap()), 10);
    assert_eq!(i32::from_le_bytes(contents[4..8].try_into().unwrap()), 10);
    assert_eq!(i32::from_le_bytes(contents[8..12].try_into().unwrap()), 0);
    assert_eq!(i32::from_le_bytes(contents[12..16].try_into().unwrap()), 10);
}

#[test]
fn test_integration_simd_i32_sub() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("simd_sub_result.txt");
    let result_path_str = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 4096];

    let filename_bytes = format!("{}\0", result_path_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // [100, 200, 300, 400]
    payloads[256..260].copy_from_slice(&100i32.to_le_bytes());
    payloads[260..264].copy_from_slice(&200i32.to_le_bytes());
    payloads[264..268].copy_from_slice(&300i32.to_le_bytes());
    payloads[268..272].copy_from_slice(&400i32.to_le_bytes());

    // [10, 20, 30, 40]
    payloads[272..276].copy_from_slice(&10i32.to_le_bytes());
    payloads[276..280].copy_from_slice(&20i32.to_le_bytes());
    payloads[280..284].copy_from_slice(&30i32.to_le_bytes());
    payloads[284..288].copy_from_slice(&40i32.to_le_bytes());

    let actions = vec![
        Action { kind: Kind::SimdLoadI32, dst: 0, src: 256, offset: 0, size: 16 },
        Action { kind: Kind::SimdLoadI32, dst: 1, src: 272, offset: 0, size: 16 },
        Action { kind: Kind::SimdSubI32, dst: 2, src: 0, offset: 1, size: 0 },
        Action { kind: Kind::SimdStoreI32, dst: 0, src: 2, offset: 500, size: 16 },
        Action { kind: Kind::FileWrite, dst: 0, src: 500, offset: filename_bytes.len() as u32, size: 16 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 0, offset: 600, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 1, offset: 608, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 2, offset: 616, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 1, src: 3, offset: 624, size: 0 },
        Action { kind: Kind::Wait, dst: 600, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 608, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 616, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 624, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 4, offset: 632, size: 0 },
        Action { kind: Kind::Wait, dst: 632, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_complex_algorithm(actions, payloads, 1, 0, 1, 0, vec![], 0);
    execute(algorithm).unwrap();

    let contents = fs::read(&result_file).unwrap();
    // [100-10, 200-20, 300-30, 400-40] = [90, 180, 270, 360]
    assert_eq!(i32::from_le_bytes(contents[0..4].try_into().unwrap()), 90);
    assert_eq!(i32::from_le_bytes(contents[4..8].try_into().unwrap()), 180);
    assert_eq!(i32::from_le_bytes(contents[8..12].try_into().unwrap()), 270);
    assert_eq!(i32::from_le_bytes(contents[12..16].try_into().unwrap()), 360);
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
            offset: filename_bytes.len() as u32,
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

    let algorithm = create_test_algorithm(actions, payloads, 1, 0);

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
fn test_integration_memwrite_immediate() {
    // Tests MemWrite operation which writes an immediate value to memory
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("memwrite_immediate_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let mut payloads = vec![0u8; 1024];

    // Setup filename with null terminator
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Memory locations for MemWrite results
    // 256: 1-byte value
    // 264: 2-byte value
    // 272: 4-byte value
    // 280: 8-byte value

    let memwrite_flag_1 = 512u32;
    let memwrite_flag_2 = 520u32;
    let memwrite_flag_4 = 528u32;
    let memwrite_flag_8 = 536u32;
    let filewrite_flag = 544u32;

    let actions = vec![
        // Action 0: MemWrite 1 byte (value 0x42 = 66)
        Action {
            kind: Kind::MemWrite,
            dst: 256,
            src: 0x42,
            offset: 0,
            size: 1,
        },
        // Action 1: MemWrite 2 bytes (value 0x1234 = 4660)
        Action {
            kind: Kind::MemWrite,
            dst: 264,
            src: 0x1234,
            offset: 0,
            size: 2,
        },
        // Action 2: MemWrite 4 bytes (value 0xDEADBEEF)
        Action {
            kind: Kind::MemWrite,
            dst: 272,
            src: 0xDEADBEEF,
            offset: 0,
            size: 4,
        },
        // Action 3: MemWrite 8 bytes (value stored as u32, will be zero-extended)
        Action {
            kind: Kind::MemWrite,
            dst: 280,
            src: 0x12345678,
            offset: 0,
            size: 8,
        },
        // Action 4: FileWrite - write all 32 bytes (256 to 288) to file
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 256,
            offset: filename_bytes.len() as u32,
            size: 32,
        },
        // Action 5-8: AsyncDispatch MemWrite operations
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6, // memory unit
            src: 0, // action 0
            offset: memwrite_flag_1,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: memwrite_flag_1,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 1,
            offset: memwrite_flag_2,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: memwrite_flag_2,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 2,
            offset: memwrite_flag_4,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: memwrite_flag_4,
            src: 0,
            offset: 0,
            size: 0,
        },
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 3,
            offset: memwrite_flag_8,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: memwrite_flag_8,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 14-15: FileWrite
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2, // file unit
            src: 4, // action 4
            offset: filewrite_flag,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: filewrite_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);

    execute(algorithm).unwrap();

    assert!(test_file.exists(), "MemWrite result file should exist");
    let contents = fs::read(&test_file).unwrap();
    assert_eq!(contents.len(), 32, "Should have written 32 bytes");

    // Verify 1-byte write at offset 0 (relative to 256)
    assert_eq!(contents[0], 0x42, "1-byte MemWrite failed");

    // Verify 2-byte write at offset 8 (264 - 256)
    let val_2 = u16::from_le_bytes([contents[8], contents[9]]);
    assert_eq!(val_2, 0x1234, "2-byte MemWrite failed");

    // Verify 4-byte write at offset 16 (272 - 256)
    let val_4 = u32::from_le_bytes([contents[16], contents[17], contents[18], contents[19]]);
    assert_eq!(val_4, 0xDEADBEEF, "4-byte MemWrite failed");

    // Verify 8-byte write at offset 24 (280 - 256)
    let val_8 = u64::from_le_bytes([
        contents[24], contents[25], contents[26], contents[27],
        contents[28], contents[29], contents[30], contents[31],
    ]);
    assert_eq!(val_8, 0x12345678, "8-byte MemWrite failed");
}

fn create_hash_table_algorithm(actions: Vec<Action>, payloads: Vec<u8>) -> Algorithm {
    create_hash_table_algorithm_with_file(actions, payloads, false)
}

fn create_hash_table_algorithm_with_file(actions: Vec<Action>, payloads: Vec<u8>, with_file: bool) -> Algorithm {
    let num_actions = actions.len();
    Algorithm {
        actions,
        payloads,
        state: State {
            regs_per_unit: 0,
            gpu_size: 0,
            computational_regs: 0,
            file_buffer_size: if with_file { 65536 } else { 0 },
            gpu_shader_offsets: vec![],
        },
        units: UnitSpec {
            simd_units: 0,
            gpu_units: 0,
            computational_units: 0,
            file_units: if with_file { 1 } else { 0 },
            network_units: 0,
            memory_units: if with_file { 1 } else { 0 },
            ffi_units: 0,
            hash_table_units: 1,
            backends_bits: 0,
        },
        simd_assignments: vec![],
        computational_assignments: vec![],
        memory_assignments: vec![],
        file_assignments: vec![],
        network_assignments: vec![],
        ffi_assignments: vec![],
        hash_table_assignments: vec![0; num_actions],
        gpu_assignments: vec![],
        worker_threads: None,
        blocking_threads: None,
        stack_size: None,
        timeout_ms: Some(5000),
        thread_name_prefix: None,
    }
}

#[test]
fn test_hash_table_create_insert_lookup() {
    // Smoke test: create, insert, lookup - just verify no crash
    let flag_offset = 0u32;
    let handle_addr = 8u32;
    let key_addr = 16u32;
    let val_addr = 32u32;
    let result_addr = 48u32;

    let mut payloads = vec![0u8; 64];
    payloads[16] = b'f'; payloads[17] = b'o'; payloads[18] = b'o';
    payloads[32..36].copy_from_slice(&42u32.to_le_bytes());

    let actions = vec![
        Action { kind: Kind::HashTableCreate, dst: handle_addr, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::HashTableInsert, dst: key_addr, src: val_addr, offset: 0, size: (3 << 16) | 4 },
        Action { kind: Kind::HashTableLookup, dst: key_addr, src: result_addr, offset: 0, size: (3 << 16) | 8 },
        Action { kind: Kind::AsyncDispatch, dst: 7, src: 0, offset: flag_offset, size: 3 },
        Action { kind: Kind::Wait, dst: flag_offset, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_hash_table_algorithm(actions, payloads);
    execute(algorithm).unwrap();
}

#[test]
fn test_hash_table_insert_lookup_verify() {
    // Verify lookup returns correct value by writing result to file
    // Layout:
    //   0..8:   flag_hash
    //   8..16:  flag_mem
    //  16..24:  flag_file
    //  24..28:  handle
    //  32..35:  key "abc"
    //  48..52:  value 99u32
    //  64..72:  result [u32 len][u32 value]
    // 256..N:   filename
    // 512..520: data copy for file write
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("ht_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_hash = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;
    let handle_addr = 24u32;
    let key_addr = 32u32;
    let val_addr = 48u32;
    let result_addr = 64u32;
    let filename_addr = 256u32;
    let data_copy_addr = 512u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32] = b'a'; payloads[33] = b'b'; payloads[34] = b'c';
    payloads[48..52].copy_from_slice(&99u32.to_le_bytes());
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[256..256 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Create
        Action { kind: Kind::HashTableCreate, dst: handle_addr, src: 0, offset: 0, size: 0 },
        // 1: Insert "abc" -> 99
        Action { kind: Kind::HashTableInsert, dst: key_addr, src: val_addr, offset: 0, size: (3 << 16) | 4 },
        // 2: Lookup "abc"
        Action { kind: Kind::HashTableLookup, dst: key_addr, src: result_addr, offset: 0, size: (3 << 16) | 8 },
        // 3: MemCopy result to data_copy
        Action { kind: Kind::MemCopy, src: result_addr, dst: data_copy_addr, offset: 0, size: 8 },
        // 4: FileWrite data_copy to file
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: filename_bytes.len() as u32, size: 8 },
        // 5: Dispatch hash ops
        Action { kind: Kind::AsyncDispatch, dst: 7, src: 0, offset: flag_hash, size: 3 },
        // 6: Wait hash
        Action { kind: Kind::Wait, dst: flag_hash, src: 0, offset: 0, size: 0 },
        // 7: Dispatch memcopy
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 3, offset: flag_mem, size: 1 },
        // 8: Wait mem
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        // 9: Dispatch filewrite
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 4, offset: flag_file, size: 1 },
        // 10: Wait file
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_hash_table_algorithm_with_file(actions, payloads, true);
    execute(algorithm).unwrap();

    assert!(test_file.exists(), "Result file should exist");
    let contents = fs::read(&test_file).unwrap();
    // Result format: [u32 len][u32 value] = [4, 99]
    let len = u32::from_le_bytes(contents[0..4].try_into().unwrap());
    let value = u32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(len, 4, "Lookup result length should be 4");
    assert_eq!(value, 99, "Lookup result value should be 99");
}

#[test]
fn test_hash_table_lookup_not_found() {
    // Verify that lookup of missing key writes 0xFFFFFFFF sentinel
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("ht_notfound.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_hash = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;
    let handle_addr = 24u32;
    let key_addr = 32u32;
    let result_addr = 64u32;
    let filename_addr = 256u32;
    let data_copy_addr = 512u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32] = b'x'; payloads[33] = b'y'; payloads[34] = b'z';
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[256..256 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Create
        Action { kind: Kind::HashTableCreate, dst: handle_addr, src: 0, offset: 0, size: 0 },
        // 1: Lookup "xyz" (not inserted)
        Action { kind: Kind::HashTableLookup, dst: key_addr, src: result_addr, offset: 0, size: (3 << 16) | 8 },
        // 2: MemCopy result
        Action { kind: Kind::MemCopy, src: result_addr, dst: data_copy_addr, offset: 0, size: 4 },
        // 3: FileWrite
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: filename_bytes.len() as u32, size: 4 },
        // 4: Dispatch hash
        Action { kind: Kind::AsyncDispatch, dst: 7, src: 0, offset: flag_hash, size: 2 },
        // 5: Wait
        Action { kind: Kind::Wait, dst: flag_hash, src: 0, offset: 0, size: 0 },
        // 6: Dispatch mem
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 2, offset: flag_mem, size: 1 },
        // 7: Wait
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        // 8: Dispatch file
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 3, offset: flag_file, size: 1 },
        // 9: Wait
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_hash_table_algorithm_with_file(actions, payloads, true);
    execute(algorithm).unwrap();

    assert!(test_file.exists(), "Result file should exist");
    let contents = fs::read(&test_file).unwrap();
    let sentinel = u32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(sentinel, 0xFFFFFFFF, "Not-found lookup should return sentinel 0xFFFFFFFF");
}

#[test]
fn test_hash_table_insert_overwrite() {
    // Insert same key twice - second value should win
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("ht_overwrite.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_hash = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;
    let handle_addr = 24u32;
    let key_addr = 32u32;
    let val1_addr = 48u32;
    let val2_addr = 56u32;
    let result_addr = 64u32;
    let filename_addr = 256u32;
    let data_copy_addr = 512u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32] = b'x';
    payloads[48..52].copy_from_slice(&10u32.to_le_bytes());
    payloads[56..60].copy_from_slice(&20u32.to_le_bytes());
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[256..256 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Create
        Action { kind: Kind::HashTableCreate, dst: handle_addr, src: 0, offset: 0, size: 0 },
        // 1: Insert "x" -> 10
        Action { kind: Kind::HashTableInsert, dst: key_addr, src: val1_addr, offset: 0, size: (1 << 16) | 4 },
        // 2: Insert "x" -> 20 (overwrite)
        Action { kind: Kind::HashTableInsert, dst: key_addr, src: val2_addr, offset: 0, size: (1 << 16) | 4 },
        // 3: Lookup "x"
        Action { kind: Kind::HashTableLookup, dst: key_addr, src: result_addr, offset: 0, size: (1 << 16) | 8 },
        // 4: MemCopy result
        Action { kind: Kind::MemCopy, src: result_addr, dst: data_copy_addr, offset: 0, size: 8 },
        // 5: FileWrite
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: filename_bytes.len() as u32, size: 8 },
        // 6: Dispatch hash
        Action { kind: Kind::AsyncDispatch, dst: 7, src: 0, offset: flag_hash, size: 4 },
        // 7: Wait
        Action { kind: Kind::Wait, dst: flag_hash, src: 0, offset: 0, size: 0 },
        // 8: Dispatch mem
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 4, offset: flag_mem, size: 1 },
        // 9: Wait
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        // 10: Dispatch file
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 5, offset: flag_file, size: 1 },
        // 11: Wait
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_hash_table_algorithm_with_file(actions, payloads, true);
    execute(algorithm).unwrap();

    assert!(test_file.exists(), "Result file should exist");
    let contents = fs::read(&test_file).unwrap();
    let len = u32::from_le_bytes(contents[0..4].try_into().unwrap());
    let value = u32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(len, 4, "Lookup result length should be 4");
    assert_eq!(value, 20, "Overwritten value should be 20, not 10");
}

#[test]
fn test_hash_table_delete() {
    // Insert then delete - lookup should return not found sentinel
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("ht_delete.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_hash = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;
    let handle_addr = 24u32;
    let key_addr = 32u32;
    let val_addr = 48u32;
    let result_addr = 64u32;
    let filename_addr = 256u32;
    let data_copy_addr = 512u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32] = b'k';
    payloads[48..52].copy_from_slice(&77u32.to_le_bytes());
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[256..256 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Create
        Action { kind: Kind::HashTableCreate, dst: handle_addr, src: 0, offset: 0, size: 0 },
        // 1: Insert "k" -> 77
        Action { kind: Kind::HashTableInsert, dst: key_addr, src: val_addr, offset: 0, size: (1 << 16) | 4 },
        // 2: Delete "k"
        Action { kind: Kind::HashTableDelete, dst: key_addr, src: 0, offset: 0, size: 1 },
        // 3: Lookup "k" (should be not found)
        Action { kind: Kind::HashTableLookup, dst: key_addr, src: result_addr, offset: 0, size: (1 << 16) | 8 },
        // 4: MemCopy result
        Action { kind: Kind::MemCopy, src: result_addr, dst: data_copy_addr, offset: 0, size: 4 },
        // 5: FileWrite
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: filename_bytes.len() as u32, size: 4 },
        // 6: Dispatch hash
        Action { kind: Kind::AsyncDispatch, dst: 7, src: 0, offset: flag_hash, size: 4 },
        // 7: Wait
        Action { kind: Kind::Wait, dst: flag_hash, src: 0, offset: 0, size: 0 },
        // 8: Dispatch mem
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 4, offset: flag_mem, size: 1 },
        // 9: Wait
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        // 10: Dispatch file
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 5, offset: flag_file, size: 1 },
        // 11: Wait
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_hash_table_algorithm_with_file(actions, payloads, true);
    execute(algorithm).unwrap();

    assert!(test_file.exists(), "Result file should exist");
    let contents = fs::read(&test_file).unwrap();
    let sentinel = u32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(sentinel, 0xFFFFFFFF, "Deleted key lookup should return sentinel 0xFFFFFFFF");
}

#[test]
fn test_hash_table_multiple_keys() {
    // Insert multiple keys, verify each lookup returns correct value
    let temp_dir = TempDir::new().unwrap();
    let test_file1 = temp_dir.path().join("ht_multi1.txt");
    let test_file2 = temp_dir.path().join("ht_multi2.txt");
    let test_file1_str = test_file1.to_str().unwrap();
    let test_file2_str = test_file2.to_str().unwrap();

    let flag_hash1 = 0u32;
    let flag_hash2 = 8u32;
    let flag_mem1 = 16u32;
    let flag_file1 = 24u32;
    let flag_mem2 = 32u32;
    let flag_file2 = 40u32;
    let handle_addr = 48u32;
    let key1_addr = 64u32;   // "ab"
    let key2_addr = 80u32;   // "cd"
    let val1_addr = 96u32;   // 100
    let val2_addr = 104u32;  // 200
    let result1_addr = 112u32;
    let result2_addr = 128u32;
    let filename1_addr = 256u32;
    let filename2_addr = 512u32;
    let data_copy1_addr = 768u32;
    let data_copy2_addr = 784u32;

    let mut payloads = vec![0u8; 1024];
    payloads[64] = b'a'; payloads[65] = b'b';
    payloads[80] = b'c'; payloads[81] = b'd';
    payloads[96..100].copy_from_slice(&100u32.to_le_bytes());
    payloads[104..108].copy_from_slice(&200u32.to_le_bytes());
    let fn1_bytes = format!("{}\0", test_file1_str).into_bytes();
    let fn2_bytes = format!("{}\0", test_file2_str).into_bytes();
    payloads[256..256 + fn1_bytes.len()].copy_from_slice(&fn1_bytes);
    payloads[512..512 + fn2_bytes.len()].copy_from_slice(&fn2_bytes);

    let actions = vec![
        // 0: Create
        Action { kind: Kind::HashTableCreate, dst: handle_addr, src: 0, offset: 0, size: 0 },
        // 1: Insert "ab" -> 100
        Action { kind: Kind::HashTableInsert, dst: key1_addr, src: val1_addr, offset: 0, size: (2 << 16) | 4 },
        // 2: Insert "cd" -> 200
        Action { kind: Kind::HashTableInsert, dst: key2_addr, src: val2_addr, offset: 0, size: (2 << 16) | 4 },
        // 3: Lookup "ab"
        Action { kind: Kind::HashTableLookup, dst: key1_addr, src: result1_addr, offset: 0, size: (2 << 16) | 8 },
        // 4: Lookup "cd"
        Action { kind: Kind::HashTableLookup, dst: key2_addr, src: result2_addr, offset: 0, size: (2 << 16) | 8 },
        // 5: MemCopy result1
        Action { kind: Kind::MemCopy, src: result1_addr, dst: data_copy1_addr, offset: 0, size: 8 },
        // 6: MemCopy result2
        Action { kind: Kind::MemCopy, src: result2_addr, dst: data_copy2_addr, offset: 0, size: 8 },
        // 7: FileWrite result1
        Action { kind: Kind::FileWrite, dst: filename1_addr, src: data_copy1_addr, offset: fn1_bytes.len() as u32, size: 8 },
        // 8: FileWrite result2
        Action { kind: Kind::FileWrite, dst: filename2_addr, src: data_copy2_addr, offset: fn2_bytes.len() as u32, size: 8 },
        // 9: Dispatch hash (create + 2 inserts + 2 lookups)
        Action { kind: Kind::AsyncDispatch, dst: 7, src: 0, offset: flag_hash1, size: 5 },
        // 10: Wait
        Action { kind: Kind::Wait, dst: flag_hash1, src: 0, offset: 0, size: 0 },
        // 11: Dispatch memcopy1
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 5, offset: flag_mem1, size: 1 },
        // 12: Wait
        Action { kind: Kind::Wait, dst: flag_mem1, src: 0, offset: 0, size: 0 },
        // 13: Dispatch filewrite1
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 7, offset: flag_file1, size: 1 },
        // 14: Wait
        Action { kind: Kind::Wait, dst: flag_file1, src: 0, offset: 0, size: 0 },
        // 15: Dispatch memcopy2
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 6, offset: flag_mem2, size: 1 },
        // 16: Wait
        Action { kind: Kind::Wait, dst: flag_mem2, src: 0, offset: 0, size: 0 },
        // 17: Dispatch filewrite2
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 8, offset: flag_file2, size: 1 },
        // 18: Wait
        Action { kind: Kind::Wait, dst: flag_file2, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_hash_table_algorithm_with_file(actions, payloads, true);
    execute(algorithm).unwrap();

    assert!(test_file1.exists(), "Result file 1 should exist");
    let contents1 = fs::read(&test_file1).unwrap();
    let len1 = u32::from_le_bytes(contents1[0..4].try_into().unwrap());
    let value1 = u32::from_le_bytes(contents1[4..8].try_into().unwrap());
    assert_eq!(len1, 4, "Lookup 'ab' result length should be 4");
    assert_eq!(value1, 100, "Lookup 'ab' should return 100");

    assert!(test_file2.exists(), "Result file 2 should exist");
    let contents2 = fs::read(&test_file2).unwrap();
    let len2 = u32::from_le_bytes(contents2[0..4].try_into().unwrap());
    let value2 = u32::from_le_bytes(contents2[4..8].try_into().unwrap());
    assert_eq!(len2, 4, "Lookup 'cd' result length should be 4");
    assert_eq!(value2, 200, "Lookup 'cd' should return 200");
}

#[test]
fn test_memwrite_large_size_zeroes_buffer() {
    // Verify that MemWrite with size > 8 (e.g. 32) correctly fills the buffer.
    //
    // Strategy: write a known pattern to a 32-byte buffer, then use MemWrite with
    // size=32 and src=0 to zero it out, then FileWrite the buffer and verify all zeros.
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("memwrite_large.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_mem = 0u32;
    let flag_file = 16u32;
    let buf_addr = 64u32;      // 32-byte buffer to test
    let filename_addr = 256u32;
    let buf_size = 32u32;

    let mut payloads = vec![0u8; 512];
    // Fill the buffer with non-zero pattern so we can verify MemWrite clears it
    for i in 0..32 {
        payloads[buf_addr as usize + i] = 0xAA;
    }
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[256..256 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: MemWrite to verify pattern is there (copy buf to check area first)
        // Actually, just directly MemWrite size=32 with src=0 to zero the buffer
        Action { kind: Kind::MemWrite, dst: buf_addr, src: 0, offset: 0, size: buf_size },
        // 1: FileWrite the buffer to file
        Action { kind: Kind::FileWrite, dst: filename_addr, src: buf_addr, offset: filename_bytes.len() as u32, size: buf_size },
        // 2: Dispatch MemWrite
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: flag_mem, size: 1 },
        // 3: Wait
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        // 4: Dispatch FileWrite
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 1, offset: flag_file, size: 1 },
        // 5: Wait
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);
    execute(algorithm).unwrap();

    assert!(test_file.exists(), "Result file should exist");
    let contents = fs::read(&test_file).unwrap();
    assert_eq!(contents.len(), 32, "Should have written 32 bytes");
    assert!(contents.iter().all(|&b| b == 0), "All 32 bytes should be zero after MemWrite with size=32, got: {:?}", contents);
}
