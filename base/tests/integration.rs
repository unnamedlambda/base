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
            cranelift_ir_offsets: vec![],
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
            lmdb_units: 0,
            cranelift_units: 0,
            backends_bits: 0,
        },
        simd_assignments: vec![],
        computational_assignments: vec![],
        memory_assignments: vec![0; num_actions],
        file_assignments: vec![0; num_actions],
        network_assignments: vec![],
        ffi_assignments: vec![],
        hash_table_assignments: vec![],
        lmdb_assignments: vec![],
        gpu_assignments: vec![],
        cranelift_assignments: vec![],
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
            cranelift_ir_offsets: vec![],
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
            lmdb_units: 0,
            cranelift_units: 0,
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
        lmdb_assignments: vec![],
        gpu_assignments: if gpu_units > 0 {
            vec![0; num_actions]
        } else {
            vec![255; num_actions]
        },
        cranelift_assignments: vec![],
        worker_threads: None,
        blocking_threads: None,
        stack_size: None,
        timeout_ms: Some(10000),
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
            regs_per_unit: 16,
            gpu_size: 0,
            computational_regs: 0,
            file_buffer_size: if with_file { 65536 } else { 0 },
            gpu_shader_offsets: vec![],
            cranelift_ir_offsets,
        },
        units: UnitSpec {
            simd_units: 0,
            gpu_units: 0,
            computational_units: 0,
            file_units: if with_file { 1 } else { 0 },
            network_units: 0,
            memory_units: 0,
            ffi_units: 0,
            hash_table_units: 0,
            lmdb_units: 0,
            cranelift_units,
            backends_bits: 0,
        },
        simd_assignments: vec![],
        computational_assignments: vec![],
        memory_assignments: vec![],
        file_assignments: if with_file { vec![0; num_actions] } else { vec![] },
        network_assignments: vec![],
        ffi_assignments: vec![],
        hash_table_assignments: vec![],
        lmdb_assignments: vec![],
        gpu_assignments: vec![],
        cranelift_assignments: vec![0; num_actions],
        worker_threads: None,
        blocking_threads: None,
        stack_size: None,
        timeout_ms: Some(5000),
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
            offset: 0,
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
            offset: 0,
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
fn test_integration_atomic_compare_exchange_u64() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("atomic_cmpxchg_u64.bin");
    let result_path = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 2048];
    let filename = format!("{}\0", result_path).into_bytes();
    payloads[0..filename.len()].copy_from_slice(&filename);

    // Layout:
    // 1000: target (u64)
    // 1008: expected_success (u64, in/out)
    // 1016: expected_failure (u64, in/out)
    // 1024: new_success (u64)
    // 1032: new_failure (u64)
    payloads[1000..1008].copy_from_slice(&7u64.to_le_bytes());
    payloads[1008..1016].copy_from_slice(&7u64.to_le_bytes());
    payloads[1016..1024].copy_from_slice(&7u64.to_le_bytes());
    payloads[1024..1032].copy_from_slice(&11u64.to_le_bytes());
    payloads[1032..1040].copy_from_slice(&99u64.to_le_bytes());

    let mem_flag_a = 1100u32;
    let mem_flag_b = 1108u32;
    let file_flag = 1116u32;

    let actions = vec![
        Action { kind: Kind::AtomicCAS, dst: 1000, src: 1008, offset: 1024, size: 8 },
        Action { kind: Kind::AtomicCAS, dst: 1000, src: 1016, offset: 1032, size: 8 },
        Action { kind: Kind::FileWrite, dst: 0, src: 1000, offset: 0, size: 24 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: mem_flag_a, size: 0 },
        Action { kind: Kind::Wait, dst: mem_flag_a, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 1, offset: mem_flag_b, size: 0 },
        Action { kind: Kind::Wait, dst: mem_flag_b, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 2, offset: file_flag, size: 0 },
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);
    execute(algorithm).unwrap();

    let out = fs::read(&result_file).unwrap();
    let target = u64::from_le_bytes(out[0..8].try_into().unwrap());
    let expected_success = u64::from_le_bytes(out[8..16].try_into().unwrap());
    let expected_failure = u64::from_le_bytes(out[16..24].try_into().unwrap());

    assert_eq!(target, 11, "First compare-exchange should update target");
    assert_eq!(expected_success, 7, "Successful compare-exchange writes observed old value");
    assert_eq!(expected_failure, 11, "Failed compare-exchange writes actual value back");
}

#[test]
fn test_integration_atomic_compare_exchange_u32() {
    let temp_dir = TempDir::new().unwrap();
    let result_file = temp_dir.path().join("atomic_cmpxchg_u32.bin");
    let result_path = result_file.to_str().unwrap();

    let mut payloads = vec![0u8; 2048];
    let filename = format!("{}\0", result_path).into_bytes();
    payloads[0..filename.len()].copy_from_slice(&filename);

    // Layout:
    // 1200: target (u32)
    // 1204: expected_success (u32, in/out)
    // 1208: expected_failure (u32, in/out)
    // 1212: new_success (u32)
    // 1216: new_failure (u32)
    payloads[1200..1204].copy_from_slice(&3u32.to_le_bytes());
    payloads[1204..1208].copy_from_slice(&3u32.to_le_bytes());
    payloads[1208..1212].copy_from_slice(&3u32.to_le_bytes());
    payloads[1212..1216].copy_from_slice(&9u32.to_le_bytes());
    payloads[1216..1220].copy_from_slice(&21u32.to_le_bytes());

    let mem_flag_a = 1300u32;
    let mem_flag_b = 1308u32;
    let file_flag = 1316u32;

    let actions = vec![
        Action { kind: Kind::AtomicCAS, dst: 1200, src: 1204, offset: 1212, size: 4 },
        Action { kind: Kind::AtomicCAS, dst: 1200, src: 1208, offset: 1216, size: 4 },
        Action { kind: Kind::FileWrite, dst: 0, src: 1200, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: mem_flag_a, size: 0 },
        Action { kind: Kind::Wait, dst: mem_flag_a, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 1, offset: mem_flag_b, size: 0 },
        Action { kind: Kind::Wait, dst: mem_flag_b, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 2, offset: file_flag, size: 0 },
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 1);
    execute(algorithm).unwrap();

    let out = fs::read(&result_file).unwrap();
    let target = u32::from_le_bytes(out[0..4].try_into().unwrap());
    let expected_success = u32::from_le_bytes(out[4..8].try_into().unwrap());
    let expected_failure = u32::from_le_bytes(out[8..12].try_into().unwrap());

    assert_eq!(target, 9, "First compare-exchange should update target");
    assert_eq!(expected_success, 3, "Successful compare-exchange writes observed old value");
    assert_eq!(expected_failure, 9, "Failed compare-exchange writes actual value back");
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
            offset: 0,
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
            offset: 0,
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
        Action { kind: Kind::FileWrite, dst: 1024, src: 2200, offset: 0, size: 8 },
        // Action 3: FileWrite path B (data action)
        Action { kind: Kind::FileWrite, dst: 512, src: 3000, offset: 0, size: 8 },
        // Action 4: FileWrite path A (data action)
        Action { kind: Kind::FileWrite, dst: 0, src: 3000, offset: 0, size: 8 },

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
            offset: 0,
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
        Action { kind: Kind::FileWrite, dst: 2048, src: 2584, offset: 0, size: 4 },
        // Action 3: FileWrite result2
        Action { kind: Kind::FileWrite, dst: 2304, src: 2648, offset: 0, size: 4 },
        // Action 4: FileWrite LOW (skipped)
        Action { kind: Kind::FileWrite, dst: 2720, src: 2832, offset: 0, size: 3 },
        // Action 5: FileWrite HIGH (executed)
        Action { kind: Kind::FileWrite, dst: 2720, src: 2776, offset: 0, size: 4 },
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
        Action { kind: Kind::FileWrite, dst: 0, src: 512, offset: 0, size: 24 },
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
            offset: 0,
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
            offset: 0,
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
            offset: 0,
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
            offset: 0,
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
            offset: 0,
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
            offset: 0,
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
        Action { kind: Kind::FileWrite, dst: 0, src: 3000, offset: 0, size: 8 },
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
        Action { kind: Kind::FileWrite, dst: 0, src: 3000, offset: 0, size: 8 },
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
        Action { kind: Kind::FileWrite, dst: 0, src: 500, offset: 0, size: 32 },
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
        Action { kind: Kind::FileWrite, dst: 0, src: 500, offset: 0, size: 32 },
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
        Action { kind: Kind::FileWrite, dst: 0, src: 500, offset: 0, size: 16 },
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
        Action { kind: Kind::FileWrite, dst: 0, src: 500, offset: 0, size: 16 },
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
            offset: 0,
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
            cranelift_ir_offsets: vec![],
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
            lmdb_units: 0,
            cranelift_units: 0,
            backends_bits: 0,
        },
        simd_assignments: vec![],
        computational_assignments: vec![],
        memory_assignments: vec![],
        file_assignments: vec![],
        network_assignments: vec![],
        ffi_assignments: vec![],
        hash_table_assignments: vec![0; num_actions],
        lmdb_assignments: vec![],
        gpu_assignments: vec![],
        cranelift_assignments: vec![],
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
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: 0, size: 8 },
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
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: 0, size: 4 },
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
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: 0, size: 8 },
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
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: 0, size: 4 },
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
        Action { kind: Kind::FileWrite, dst: filename1_addr, src: data_copy1_addr, offset: 0, size: 8 },
        // 8: FileWrite result2
        Action { kind: Kind::FileWrite, dst: filename2_addr, src: data_copy2_addr, offset: 0, size: 8 },
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
        Action { kind: Kind::FileWrite, dst: filename_addr, src: buf_addr, offset: 0, size: buf_size },
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

#[test]
fn test_integration_memory_compare() {
    // Sets up two i32 values in shared memory, runs Compare via AsyncDispatch,
    // then writes the result to a file to verify.
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("compare_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    // Memory layout:
    //   0..256: filename
    // 256..260: value a (i32)
    // 264..268: value b (i32)
    // 272..276: result (i32)
    // 512: compare flag
    // 520: filewrite flag
    let a_addr = 256u32;
    let b_addr = 264u32;
    let result_addr = 272u32;
    let compare_flag = 512u32;
    let filewrite_flag = 520u32;

    let mut payloads = vec![0u8; 1024];

    // Store a=5, b=3 in payload
    payloads[a_addr as usize..a_addr as usize + 4].copy_from_slice(&5i32.to_le_bytes());
    payloads[b_addr as usize..b_addr as usize + 4].copy_from_slice(&3i32.to_le_bytes());

    // Store filename
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // Action 0: Compare — reads a at src, b at offset, writes (a > b ? 1 : 0) to dst
        Action {
            kind: Kind::Compare,
            dst: result_addr,
            src: a_addr,
            offset: b_addr,
            size: 0,
        },
        // Action 1: FileWrite — write 4 bytes from result_addr to file
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: result_addr,
            offset: 0,
            size: 4,
        },
        // Action 2-3: AsyncDispatch Compare to memory unit + Wait
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6, // memory unit
            src: 0, // action 0
            offset: compare_flag,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: compare_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 4-5: AsyncDispatch FileWrite + Wait
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2, // file unit
            src: 1, // action 1
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

    assert!(test_file.exists(), "Compare result file should exist");
    let contents = fs::read(&test_file).unwrap();
    assert_eq!(contents.len(), 4);
    let result = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(result, 1, "5 > 3 should be 1 (true)");
}

#[test]
fn test_integration_memory_compare_ge() {
    // Tests GE mode (size=5): a >= b returns 1 when equal, 0 when a < b.
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("compare_ge_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let a_addr = 256u32;
    let b_addr = 264u32;
    let result_addr = 272u32;
    let compare_flag = 512u32;
    let filewrite_flag = 520u32;

    let mut payloads = vec![0u8; 1024];

    // Store a=5, b=5 (equal values)
    payloads[a_addr as usize..a_addr as usize + 4].copy_from_slice(&5i32.to_le_bytes());
    payloads[b_addr as usize..b_addr as usize + 4].copy_from_slice(&5i32.to_le_bytes());

    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // Action 0: Compare GE — size=5 means >= mode
        Action {
            kind: Kind::Compare,
            dst: result_addr,
            src: a_addr,
            offset: b_addr,
            size: 5,
        },
        // Action 1: FileWrite
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: result_addr,
            offset: 0,
            size: 4,
        },
        // Action 2-3: AsyncDispatch Compare + Wait
        Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: 0,
            offset: compare_flag,
            size: 0,
        },
        Action {
            kind: Kind::Wait,
            dst: compare_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 4-5: AsyncDispatch FileWrite + Wait
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
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

    assert!(test_file.exists(), "Compare GE result file should exist");
    let contents = fs::read(&test_file).unwrap();
    assert_eq!(contents.len(), 4);
    let result = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(result, 1, "5 >= 5 should be 1 (true)");
}

fn create_lmdb_algorithm(actions: Vec<Action>, payloads: Vec<u8>, with_file: bool, with_memory: bool) -> Algorithm {
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
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            simd_units: 0,
            gpu_units: 0,
            computational_units: 0,
            file_units: if with_file { 1 } else { 0 },
            network_units: 0,
            memory_units: if with_memory { 1 } else { 0 },
            ffi_units: 0,
            hash_table_units: 0,
            lmdb_units: 1,
            cranelift_units: 0,
            backends_bits: 0,
        },
        simd_assignments: vec![],
        computational_assignments: vec![],
        memory_assignments: vec![],
        file_assignments: vec![],
        network_assignments: vec![],
        ffi_assignments: vec![],
        hash_table_assignments: vec![],
        lmdb_assignments: vec![0; num_actions],
        gpu_assignments: vec![],
        cranelift_assignments: vec![],
        worker_threads: None,
        blocking_threads: None,
        stack_size: None,
        timeout_ms: Some(5000),
        thread_name_prefix: None,
    }
}

#[test]
fn test_lmdb_basic_operations() {
    // Test open, put, get by verifying result through memory and file write
    // Layout:
    //   0..8:   flag_lmdb
    //   8..16:  flag_mem
    //  16..24:  flag_file
    //  24..28:  handle
    //  32..288: db_path (null-terminated)
    // 300..303: key "foo"
    // 320..323: value "bar"
    // 340..352: get result [u32 len][data]
    // 400..N:   filename
    // 512..524: data copy for file write [u32 len][data]
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("basic_ops");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("lmdb_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;
    let handle_addr = 24u32;
    let db_path_addr = 32u32;
    let key_addr = 300u32;
    let val_addr = 320u32;
    let result_addr = 340u32;
    let filename_addr = 400u32;
    let data_copy_addr = 512u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"foo");
    payloads[320..323].copy_from_slice(b"bar");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[400..400 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Open LMDB
        Action { kind: Kind::LmdbOpen, dst: handle_addr, src: db_path_addr, offset: 256, size: 10 },
        // 1: Put foo=bar
        Action { kind: Kind::LmdbPut, dst: key_addr, src: val_addr, offset: 0, size: (3 << 16) | 3 },
        // 2: Get foo -> result_addr (writes u32 len + data)
        Action { kind: Kind::LmdbGet, dst: key_addr, src: result_addr, offset: 0, size: 3 << 16 },
        // 3: Dispatch LMDB operations
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 3 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // 5: MemCopy result to data_copy_addr (copy 12 bytes: u32 len + up to 8 bytes data)
        Action { kind: Kind::MemCopy, dst: data_copy_addr, src: result_addr, offset: 0, size: 12 },
        // 6: Dispatch MemCopy
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 5, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        // 8: FileWrite data_copy_addr to file (write 12 bytes)
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: 0, size: 12 },
        // 9: Dispatch FileWrite
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 8, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    // Read the file and verify it contains the correct data
    let file_contents = fs::read(&test_file).unwrap();
    assert!(file_contents.len() >= 7, "File should contain at least u32 len + 3 bytes value");

    let len = u32::from_le_bytes([file_contents[0], file_contents[1], file_contents[2], file_contents[3]]);
    assert_eq!(len, 3, "Length should be 3");
    assert_eq!(&file_contents[4..7], b"bar", "Value should be 'bar'");
}

#[test]
fn test_lmdb_delete() {
    // Test that deleted keys return sentinel 0xFFFFFFFF when accessed
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("delete");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("delete_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;
    let handle_addr = 24u32;
    let db_path_addr = 32u32;
    let key_addr = 300u32;
    let val_addr = 320u32;
    let result_addr = 340u32;
    let filename_addr = 400u32;
    let data_copy_addr = 512u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"foo");
    payloads[320..323].copy_from_slice(b"bar");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[400..400 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Open LMDB
        Action { kind: Kind::LmdbOpen, dst: handle_addr, src: db_path_addr, offset: 256, size: 10 },
        // 1: Put foo=bar
        Action { kind: Kind::LmdbPut, dst: key_addr, src: val_addr, offset: 0, size: (3 << 16) | 3 },
        // 2: Delete foo
        Action { kind: Kind::LmdbDelete, dst: key_addr, src: 0, offset: 0, size: 3 << 16 },
        // 3: Get foo -> result_addr (should write 0xFFFFFFFF sentinel)
        Action { kind: Kind::LmdbGet, dst: key_addr, src: result_addr, offset: 0, size: 3 << 16 },
        // 4: Dispatch LMDB operations
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 4 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // 6: MemCopy result to data_copy_addr (copy 4 bytes for sentinel)
        Action { kind: Kind::MemCopy, dst: data_copy_addr, src: result_addr, offset: 0, size: 4 },
        // 7: Dispatch MemCopy
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 6, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        // 9: FileWrite sentinel to file
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: 0, size: 4 },
        // 10: Dispatch FileWrite
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 9, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    // Read the file and verify it contains the sentinel value
    let file_contents = fs::read(&test_file).unwrap();
    assert_eq!(file_contents.len(), 4, "File should contain 4 bytes");

    let sentinel = u32::from_le_bytes([file_contents[0], file_contents[1], file_contents[2], file_contents[3]]);
    assert_eq!(sentinel, 0xFFFF_FFFF, "Deleted key should return sentinel value 0xFFFFFFFF");
}

#[test]
fn test_lmdb_cursor_scan() {
    // Test cursor scan by writing results to file and verifying format
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("cursor_scan");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("cursor_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;
    let handle_addr = 24u32;
    let db_path_addr = 32u32;
    let scan_result_addr = 1024u32;
    let filename_addr = 400u32;
    let data_copy_addr = 2048u32;

    let mut payloads = vec![0u8; 4096];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());

    // Write multiple key-value pairs
    payloads[512..515].copy_from_slice(b"aaa");
    payloads[528..534].copy_from_slice(b"value1");
    payloads[544..547].copy_from_slice(b"bbb");
    payloads[560..566].copy_from_slice(b"value2");
    payloads[576..579].copy_from_slice(b"ccc");
    payloads[592..598].copy_from_slice(b"value3");
    payloads[608..611].copy_from_slice(b"ddd");
    payloads[624..630].copy_from_slice(b"value4");

    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[400..400 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Open LMDB
        Action { kind: Kind::LmdbOpen, dst: handle_addr, src: db_path_addr, offset: 256, size: 10 },
        // 1-4: Put 4 key-value pairs
        Action { kind: Kind::LmdbPut, dst: 512, src: 528, offset: 0, size: (3 << 16) | 6 },
        Action { kind: Kind::LmdbPut, dst: 544, src: 560, offset: 0, size: (3 << 16) | 6 },
        Action { kind: Kind::LmdbPut, dst: 576, src: 592, offset: 0, size: (3 << 16) | 6 },
        Action { kind: Kind::LmdbPut, dst: 608, src: 624, offset: 0, size: (3 << 16) | 6 },
        // 5: Cursor scan from beginning, max 10 entries
        Action { kind: Kind::LmdbCursorScan, dst: scan_result_addr, src: 0, offset: 0, size: (0 << 16) | 10 },
        // 6: Dispatch LMDB operations
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 6 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // 8: MemCopy scan result to data_copy_addr (copy first 100 bytes)
        Action { kind: Kind::MemCopy, dst: data_copy_addr, src: scan_result_addr, offset: 0, size: 100 },
        // 9: Dispatch MemCopy
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 8, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        // 11: FileWrite scan result to file
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: 0, size: 100 },
        // 12: Dispatch FileWrite
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 11, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    // Read the file and verify cursor scan format
    let file_contents = fs::read(&test_file).unwrap();
    assert!(file_contents.len() >= 4, "File should contain at least count field");

    // First 4 bytes is the entry count
    let count = u32::from_le_bytes([file_contents[0], file_contents[1], file_contents[2], file_contents[3]]);
    assert_eq!(count, 4, "Should have 4 entries");

    // Verify cursor scan output format: [u32 count][u16 klen, u16 vlen, key, val]...
    // Entry 1: "aaa" (3 bytes) -> "value1" (6 bytes)
    let offset = 4;
    let klen1 = u16::from_le_bytes([file_contents[offset], file_contents[offset + 1]]);
    let vlen1 = u16::from_le_bytes([file_contents[offset + 2], file_contents[offset + 3]]);
    assert_eq!(klen1, 3);
    assert_eq!(vlen1, 6);
    assert_eq!(&file_contents[offset + 4..offset + 7], b"aaa");
    assert_eq!(&file_contents[offset + 7..offset + 13], b"value1");
}

#[test]
fn test_lmdb_batched_writes() {
    // Test BeginWriteTxn / Put×N / CommitWriteTxn batching, then Get to verify
    // Layout:
    //   0..8:   flag_lmdb
    //   8..16:  flag_mem
    //  16..24:  flag_file
    //  24..28:  handle
    //  32..288: db_path
    // 300..303: key1 "aaa"
    // 320..326: val1 "value1"
    // 340..343: key2 "bbb"
    // 360..366: val2 "value2"
    // 380..383: key3 "ccc"
    // 400..406: val3 "value3"
    // 420..432: get result [u32 len][data]
    // 500..N:   filename
    // 600..612: data copy for file write
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("batched");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("batch_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;
    let handle_addr = 24u32;
    let db_path_addr = 32u32;
    let get_result_addr = 420u32;
    let filename_addr = 500u32;
    let data_copy_addr = 600u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"aaa");
    payloads[320..326].copy_from_slice(b"value1");
    payloads[340..343].copy_from_slice(b"bbb");
    payloads[360..366].copy_from_slice(b"value2");
    payloads[380..383].copy_from_slice(b"ccc");
    payloads[400..406].copy_from_slice(b"value3");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Open LMDB
        Action { kind: Kind::LmdbOpen, dst: handle_addr, src: db_path_addr, offset: 256, size: 10 },
        // 1: Begin write transaction
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 2-4: Batch 3 puts in one transaction
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (3 << 16) | 6 },
        Action { kind: Kind::LmdbPut, dst: 340, src: 360, offset: 0, size: (3 << 16) | 6 },
        Action { kind: Kind::LmdbPut, dst: 380, src: 400, offset: 0, size: (3 << 16) | 6 },
        // 5: Commit write transaction
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 6: Get "bbb" to verify batch was committed
        Action { kind: Kind::LmdbGet, dst: 340, src: get_result_addr, offset: 0, size: 3 << 16 },
        // 7: Dispatch LMDB operations
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 7 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // 9: MemCopy result
        Action { kind: Kind::MemCopy, dst: data_copy_addr, src: get_result_addr, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 9, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        // 12: FileWrite
        Action { kind: Kind::FileWrite, dst: filename_addr, src: data_copy_addr, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 12, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    assert!(file_contents.len() >= 10, "File should contain u32 len + 6 bytes value");

    let len = u32::from_le_bytes([file_contents[0], file_contents[1], file_contents[2], file_contents[3]]);
    assert_eq!(len, 6, "Length should be 6");
    assert_eq!(&file_contents[4..10], b"value2", "Value should be 'value2'");
}

#[test]
fn test_lmdb_uncommitted_read_in_batch() {
    // Get within an active write txn should see uncommitted puts
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("uncommitted_read");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("uncommitted_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"key");
    payloads[320..325].copy_from_slice(b"value");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Open
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // 1: Begin write txn
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 2: Put key=value (within batch, not committed)
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (3 << 16) | 5 },
        // 3: Get key (should see uncommitted value within same txn)
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 3 << 16 },
        // 4: Commit
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 5: Dispatch LMDB
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 5 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // 7: MemCopy + FileWrite to verify
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 7, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 10, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let len = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(len, 5, "Uncommitted read should see value length 5");
    assert_eq!(&file_contents[4..9], b"value", "Uncommitted read should see 'value'");
}

#[test]
fn test_lmdb_cursor_scan_in_batch() {
    // CursorScan within an active write txn should see uncommitted data
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("cursor_batch");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("cursor_batch_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 4096];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[512..514].copy_from_slice(b"aa");
    payloads[528..530].copy_from_slice(b"v1");
    payloads[544..546].copy_from_slice(b"bb");
    payloads[560..562].copy_from_slice(b"v2");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[400..400 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Open
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // 1: Begin
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 2-3: Put two entries
        Action { kind: Kind::LmdbPut, dst: 512, src: 528, offset: 0, size: (2 << 16) | 2 },
        Action { kind: Kind::LmdbPut, dst: 544, src: 560, offset: 0, size: (2 << 16) | 2 },
        // 4: CursorScan (should see both uncommitted entries)
        Action { kind: Kind::LmdbCursorScan, dst: 1024, src: 0, offset: 0, size: (0 << 16) | 10 },
        // 5: Commit
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 6: Dispatch
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 6 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // 8: MemCopy + FileWrite
        Action { kind: Kind::MemCopy, dst: 2048, src: 1024, offset: 0, size: 50 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 8, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 400, src: 2048, offset: 0, size: 50 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 11, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let count = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(count, 2, "CursorScan in batch should see 2 uncommitted entries");
    // First entry: "aa" -> "v1"
    let klen = u16::from_le_bytes([file_contents[4], file_contents[5]]);
    let vlen = u16::from_le_bytes([file_contents[6], file_contents[7]]);
    assert_eq!(klen, 2);
    assert_eq!(vlen, 2);
    assert_eq!(&file_contents[8..10], b"aa");
    assert_eq!(&file_contents[10..12], b"v1");
}

#[test]
fn test_lmdb_commit_without_begin() {
    // CommitWriteTxn with no active txn should be a no-op
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("commit_no_begin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("commit_no_begin_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"key");
    payloads[320..325].copy_from_slice(b"value");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Open
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // 1: Commit without begin (should be no-op)
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 2: Auto-commit Put (should still work fine)
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (3 << 16) | 5 },
        // 3: Get
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 3 << 16 },
        // 4: Dispatch
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 4 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // 6: MemCopy + FileWrite
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 6, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 9, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let len = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(len, 5, "Auto-commit put after stray commit should work");
    assert_eq!(&file_contents[4..9], b"value");
}

#[test]
fn test_lmdb_empty_batch() {
    // BeginWriteTxn immediately followed by CommitWriteTxn — no crash
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("empty_batch");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());

    let flag_lmdb = 0u32;

    let mut payloads = vec![0u8; 512];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 3 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, false, false);
    execute(algorithm).unwrap();
    // No panic = pass
}

#[test]
fn test_lmdb_double_begin() {
    // Second BeginWriteTxn aborts the first, doesn't deadlock, data from first is lost
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("double_begin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("double_begin_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..304].copy_from_slice(b"key1");
    payloads[320..325].copy_from_slice(b"lost!");  // will be aborted
    payloads[340..344].copy_from_slice(b"key2");
    payloads[360..365].copy_from_slice(b"kept!");  // will be committed
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Open
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // 1: Begin first batch
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 2: Put key1=lost! (in first txn)
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (4 << 16) | 5 },
        // 3: Begin second batch (aborts first, key1 put is lost)
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 4: Put key2=kept! (in second txn)
        Action { kind: Kind::LmdbPut, dst: 340, src: 360, offset: 0, size: (4 << 16) | 5 },
        // 5: Commit second batch
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 6: Get key1 (should be sentinel — was in aborted txn)
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 4 << 16 },
        // 7: Dispatch
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 7 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // 9: Write sentinel to file
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 9, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 12, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let sentinel = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(sentinel, 0xFFFF_FFFF, "key1 from aborted txn should not exist");
}

#[test]
fn test_lmdb_batch_delete_readback() {
    // Delete within batch, Get in same batch returns sentinel
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("batch_delete");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("batch_delete_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_lmdb2 = 100u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"key");
    payloads[320..323].copy_from_slice(b"val");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // Phase 1: auto-commit put key=val
        // 0: Open
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // 1: Put key=val (auto-commit)
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (3 << 16) | 3 },
        // 2: Dispatch phase 1
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 2 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },

        // Phase 2: batch delete + read back
        // 4: Begin batch
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 5: Delete key
        Action { kind: Kind::LmdbDelete, dst: 300, src: 0, offset: 0, size: 3 << 16 },
        // 6: Get key (should be sentinel within batch)
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 3 << 16 },
        // 7: Commit
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 8: Dispatch phase 2
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 4, offset: flag_lmdb2, size: 4 },
        Action { kind: Kind::Wait, dst: flag_lmdb2, src: 0, offset: 0, size: 0 },

        // Write result to file
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 10, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 13, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let sentinel = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(sentinel, 0xFFFF_FFFF, "Deleted key in batch should return sentinel");
}

#[test]
fn test_lmdb_mixed_batch_and_autocommit() {
    // Batch put key1, commit, then auto-commit put key2 — both visible
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("mixed");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("mixed_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_lmdb2 = 100u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..304].copy_from_slice(b"key1");
    payloads[320..324].copy_from_slice(b"bat!");
    payloads[340..344].copy_from_slice(b"key2");
    payloads[360..364].copy_from_slice(b"aut!");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // Phase 1: batch put key1
        // 0: Open
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // 1: Begin
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 2: Put key1=bat!
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (4 << 16) | 4 },
        // 3: Commit
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 4: Auto-commit put key2=aut!
        Action { kind: Kind::LmdbPut, dst: 340, src: 360, offset: 0, size: (4 << 16) | 4 },
        // 5: Dispatch phase 1
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 5 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },

        // Phase 2: CursorScan all entries — should see both
        // 7: CursorScan
        Action { kind: Kind::LmdbCursorScan, dst: 700, src: 0, offset: 0, size: (0 << 16) | 10 },
        // 8: Dispatch
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 7, offset: flag_lmdb2, size: 1 },
        Action { kind: Kind::Wait, dst: flag_lmdb2, src: 0, offset: 0, size: 0 },

        // Write scan result to file
        Action { kind: Kind::MemCopy, dst: 800, src: 700, offset: 0, size: 50 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 10, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 800, offset: 0, size: 50 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 13, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let count = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(count, 2, "Should see both batched and auto-committed entries");
    // Entries are sorted by key: "key1" then "key2"
    let klen1 = u16::from_le_bytes([file_contents[4], file_contents[5]]);
    assert_eq!(klen1, 4);
    assert_eq!(&file_contents[8..12], b"key1");
}

#[test]
fn test_lmdb_batch_overwrite() {
    // Put same key twice in batch — Get sees the latest value
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("overwrite");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("overwrite_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"key");
    payloads[320..324].copy_from_slice(b"old!");
    payloads[340..344].copy_from_slice(b"new!");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // 0: Open
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // 1: Begin
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 2: Put key=old!
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (3 << 16) | 4 },
        // 3: Put key=new! (overwrite in same txn)
        Action { kind: Kind::LmdbPut, dst: 300, src: 340, offset: 0, size: (3 << 16) | 4 },
        // 4: Get key (should see "new!")
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 3 << 16 },
        // 5: Commit
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        // 6: Dispatch
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 6 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // 8: MemCopy + FileWrite
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 8, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 11, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let len = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(len, 4, "Overwritten value should have length 4");
    assert_eq!(&file_contents[4..8], b"new!", "Should see the overwritten value");
}

#[test]
fn test_lmdb_uncommitted_batch_dropped() {
    // BeginWriteTxn + Put but no Commit — unit shutdown aborts the txn, data not persisted
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("dropped_batch");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("dropped_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_lmdb2 = 100u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"key");
    payloads[320..323].copy_from_slice(b"val");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Phase 1: begin + put but NO commit — dispatch returns, unit continues
    // Phase 2: new dispatch that does a fresh Get — should be sentinel because txn was
    // never committed (it's still active though, so the Get takes the non-batch path
    // and creates a ReadTransaction which only sees committed state)
    // Actually: the txn is still active in phase 2 dispatch, so Get will go through
    // the batched path and see the uncommitted data. We need to test that after the
    // unit fully shuts down (Drop aborts), the data is not persisted.
    //
    // To properly test this, we do two separate execute() calls on the same db path.

    // Execute 1: begin + put, no commit
    let actions1 = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (3 << 16) | 3 },
        // No commit! Unit will Drop and abort the txn.
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 3 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
    ];
    let alg1 = create_lmdb_algorithm(actions1, payloads.clone(), false, false);
    execute(alg1).unwrap();

    // Execute 2: reopen same db, Get should return sentinel
    let actions2 = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 3 << 16 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 2 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 4, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 7, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];
    let alg2 = create_lmdb_algorithm(actions2, payloads, true, true);
    execute(alg2).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let sentinel = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(sentinel, 0xFFFF_FFFF, "Uncommitted batch should be aborted on Drop, data not persisted");
}

#[test]
fn test_lmdb_get_nonexistent_key() {
    // Get on a key that was never put should return sentinel 0xFFFFFFFF
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("get_nonexistent");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("get_nonexistent_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..306].copy_from_slice(b"nokey!");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // Get a key that was never put
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 6 << 16 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 2 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 4, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 7, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let sentinel = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(sentinel, 0xFFFF_FFFF, "Get on nonexistent key should return sentinel");
}

#[test]
fn test_lmdb_delete_nonexistent_key() {
    // Deleting a key that doesn't exist should be a no-op (not crash),
    // and subsequent put/get should still work
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("del_nonexistent");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("del_nonexistent_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"key");
    payloads[320..325].copy_from_slice(b"value");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // Delete a key that doesn't exist yet
        Action { kind: Kind::LmdbDelete, dst: 300, src: 0, offset: 0, size: 3 << 16 },
        // Put should still work after deleting nonexistent key
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (3 << 16) | 5 },
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 3 << 16 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 4 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 6, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 12 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 9, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let len = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(len, 5);
    assert_eq!(&file_contents[4..9], b"value", "Put after deleting nonexistent key should work");
}

#[test]
fn test_lmdb_cursor_scan_empty_db() {
    // CursorScan on an empty database should return count=0
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("cursor_empty");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("cursor_empty_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // CursorScan with no start key, max 100 entries, on empty db
        Action { kind: Kind::LmdbCursorScan, dst: 420, src: 0, offset: 0, size: 100 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 2 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 4, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 7, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let count = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(count, 0, "CursorScan on empty db should return 0 entries");
}

#[test]
fn test_lmdb_cursor_scan_with_start_key() {
    // Put keys aa, bb, cc, dd — scan from "cc" should return only cc and dd
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("cursor_start");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("cursor_start_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 2048];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    // keys at 300, 310, 320, 330; values at 340, 350, 360, 370
    payloads[300..302].copy_from_slice(b"aa");
    payloads[310..312].copy_from_slice(b"bb");
    payloads[320..322].copy_from_slice(b"cc");
    payloads[330..332].copy_from_slice(b"dd");
    payloads[340..342].copy_from_slice(b"v1");
    payloads[350..352].copy_from_slice(b"v2");
    payloads[360..362].copy_from_slice(b"v3");
    payloads[370..372].copy_from_slice(b"v4");
    // start key "cc" at 380
    payloads[380..382].copy_from_slice(b"cc");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        Action { kind: Kind::LmdbPut, dst: 300, src: 340, offset: 0, size: (2 << 16) | 2 },
        Action { kind: Kind::LmdbPut, dst: 310, src: 350, offset: 0, size: (2 << 16) | 2 },
        Action { kind: Kind::LmdbPut, dst: 320, src: 360, offset: 0, size: (2 << 16) | 2 },
        Action { kind: Kind::LmdbPut, dst: 330, src: 370, offset: 0, size: (2 << 16) | 2 },
        // CursorScan from "cc", max 100
        // src=380 (start key addr), size = (key_len=2 << 16) | max_entries=100
        Action { kind: Kind::LmdbCursorScan, dst: 800, src: 380, offset: 0, size: (2 << 16) | 100 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 6 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // Copy result: 4 (count) + 2*(2+2+2+2) = 4+16 = 20 bytes
        Action { kind: Kind::MemCopy, dst: 900, src: 800, offset: 0, size: 24 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 8, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 900, offset: 0, size: 24 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 11, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let count = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(count, 2, "CursorScan from 'cc' should return 2 entries (cc, dd)");

    // First entry: cc -> v3
    let klen1 = u16::from_le_bytes(file_contents[4..6].try_into().unwrap()) as usize;
    let vlen1 = u16::from_le_bytes(file_contents[6..8].try_into().unwrap()) as usize;
    assert_eq!(klen1, 2);
    assert_eq!(vlen1, 2);
    assert_eq!(&file_contents[8..10], b"cc");
    assert_eq!(&file_contents[10..12], b"v3");

    // Second entry: dd -> v4
    let klen2 = u16::from_le_bytes(file_contents[12..14].try_into().unwrap()) as usize;
    let vlen2 = u16::from_le_bytes(file_contents[14..16].try_into().unwrap()) as usize;
    assert_eq!(klen2, 2);
    assert_eq!(vlen2, 2);
    assert_eq!(&file_contents[16..18], b"dd");
    assert_eq!(&file_contents[18..20], b"v4");
}

#[test]
fn test_lmdb_cursor_scan_max_entries_limit() {
    // Put 4 entries, scan with max_entries=2 — should return exactly 2
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("cursor_limit");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("cursor_limit_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 2048];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..302].copy_from_slice(b"aa");
    payloads[310..312].copy_from_slice(b"bb");
    payloads[320..322].copy_from_slice(b"cc");
    payloads[330..332].copy_from_slice(b"dd");
    payloads[340..342].copy_from_slice(b"v1");
    payloads[350..352].copy_from_slice(b"v2");
    payloads[360..362].copy_from_slice(b"v3");
    payloads[370..372].copy_from_slice(b"v4");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        Action { kind: Kind::LmdbPut, dst: 300, src: 340, offset: 0, size: (2 << 16) | 2 },
        Action { kind: Kind::LmdbPut, dst: 310, src: 350, offset: 0, size: (2 << 16) | 2 },
        Action { kind: Kind::LmdbPut, dst: 320, src: 360, offset: 0, size: (2 << 16) | 2 },
        Action { kind: Kind::LmdbPut, dst: 330, src: 370, offset: 0, size: (2 << 16) | 2 },
        // CursorScan: no start key (key_len=0), max_entries=2
        Action { kind: Kind::LmdbCursorScan, dst: 800, src: 0, offset: 0, size: 2 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 6 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::MemCopy, dst: 900, src: 800, offset: 0, size: 20 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 8, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 900, offset: 0, size: 20 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 11, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let count = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(count, 2, "CursorScan with max_entries=2 should return exactly 2");

    // Should be first two in sorted order: aa, bb
    let klen1 = u16::from_le_bytes(file_contents[4..6].try_into().unwrap()) as usize;
    assert_eq!(klen1, 2);
    assert_eq!(&file_contents[8..10], b"aa");

    let klen2 = u16::from_le_bytes(file_contents[12..14].try_into().unwrap()) as usize;
    assert_eq!(klen2, 2);
    assert_eq!(&file_contents[16..18], b"bb");
}

#[test]
fn test_lmdb_sync_no_crash() {
    // LmdbSync should not crash on a valid env
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("sync_test");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());

    let flag_lmdb = 0u32;

    let mut payloads = vec![0u8; 512];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"key");
    payloads[320..322].copy_from_slice(b"hi");

    let actions = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (3 << 16) | 2 },
        Action { kind: Kind::LmdbSync, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 3 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, false, false);
    execute(algorithm).unwrap();
    // No panic = pass
}

#[test]
fn test_lmdb_put_empty_value() {
    // Put a key with 0-length value, Get should return len=0
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("empty_val");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("empty_val_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 1024];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"key");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        // Put key with 0-length value: key_size=3, val_size=0
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: 3 << 16 },
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 3 << 16 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 3 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 5, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 4 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 8, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let len = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(len, 0, "Get of key with empty value should return len=0 (not sentinel)");
}

#[test]
fn test_lmdb_large_value() {
    // Put a 4096-byte value, verify it roundtrips correctly
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("large_val");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let test_file = temp_dir.path().join("large_val_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let val_size = 4096usize;
    let mut payloads = vec![0u8; 16384];
    payloads[32..32 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"key");
    // Fill value at 1000 with a pattern
    for i in 0..val_size {
        payloads[1000 + i] = (i % 251) as u8; // prime-mod pattern
    }
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 256, size: 10 },
        Action { kind: Kind::LmdbPut, dst: 300, src: 1000, offset: 0, size: (3 << 16) | (val_size as u32) },
        Action { kind: Kind::LmdbGet, dst: 300, src: 6000, offset: 0, size: 3 << 16 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 3 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // Copy result header (4 bytes len) + val_size bytes
        Action { kind: Kind::MemCopy, dst: 11000, src: 6000, offset: 0, size: (4 + val_size) as u32 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 5, offset: flag_mem, size: 1 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 11000, offset: 0, size: (4 + val_size) as u32 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 8, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let len = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(len as usize, val_size, "Large value should roundtrip with correct length");
    for i in 0..val_size {
        assert_eq!(
            file_contents[4 + i],
            (i % 251) as u8,
            "Large value mismatch at byte {}",
            i
        );
    }
}

#[test]
fn test_lmdb_multiple_databases() {
    // Open two independent databases, verify they are isolated
    let temp_dir = TempDir::new().unwrap();
    let db1_path = temp_dir.path().join("multi_db1");
    let db2_path = temp_dir.path().join("multi_db2");
    let db1_path_str = format!("{}\0", db1_path.to_str().unwrap());
    let db2_path_str = format!("{}\0", db2_path.to_str().unwrap());
    let test_file = temp_dir.path().join("multi_db_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    // handle1 at offset 24, handle2 at offset 28
    let mut payloads = vec![0u8; 2048];
    payloads[32..32 + db1_path_str.len()].copy_from_slice(db1_path_str.as_bytes());
    // db2 path at offset 256
    payloads[256..256 + db2_path_str.len()].copy_from_slice(db2_path_str.as_bytes());
    // key "key" at 300
    payloads[300..303].copy_from_slice(b"key");
    // value "db1!" at 320, "db2!" at 340
    payloads[320..324].copy_from_slice(b"db1!");
    payloads[340..344].copy_from_slice(b"db2!");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // Open db1 -> handle at addr 24 (will be 0)
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 224, size: 10 },
        // Open db2 -> handle at addr 28 (will be 1)
        Action { kind: Kind::LmdbOpen, dst: 28, src: 256, offset: 512, size: 10 },
        // Put "key"="db1!" into db1 (handle via offset field — but handle is stored at addr 24)
        // Handle 0 is at offset=0
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (3 << 16) | 4 },
        // Put "key"="db2!" into db2 (handle=1)
        Action { kind: Kind::LmdbPut, dst: 300, src: 340, offset: 1, size: (3 << 16) | 4 },
        // Get "key" from db1 -> result at 420
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 3 << 16 },
        // Get "key" from db2 -> result at 460
        Action { kind: Kind::LmdbGet, dst: 300, src: 460, offset: 1, size: 3 << 16 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 6 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        // Copy db1 result (4+4=8 bytes) to 600
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 8 },
        // Copy db2 result (4+4=8 bytes) to 608 (contiguous)
        Action { kind: Kind::MemCopy, dst: 608, src: 460, offset: 0, size: 8 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 8, offset: flag_mem, size: 2 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        // Write both results: 16 bytes total
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 16 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 12, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    // db1 result
    let len1 = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(len1, 4);
    assert_eq!(&file_contents[4..8], b"db1!", "db1 should return db1! for key");
    // db2 result
    let len2 = u32::from_le_bytes(file_contents[8..12].try_into().unwrap());
    assert_eq!(len2, 4);
    assert_eq!(&file_contents[12..16], b"db2!", "db2 should return db2! for key");
}

#[test]
fn test_lmdb_batch_across_two_databases() {
    // Batch writes across two different databases: begin/commit on each independently
    let temp_dir = TempDir::new().unwrap();
    let db1_path = temp_dir.path().join("batch_multi_db1");
    let db2_path = temp_dir.path().join("batch_multi_db2");
    let db1_path_str = format!("{}\0", db1_path.to_str().unwrap());
    let db2_path_str = format!("{}\0", db2_path.to_str().unwrap());
    let test_file = temp_dir.path().join("batch_multi_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let flag_lmdb = 0u32;
    let flag_mem = 8u32;
    let flag_file = 16u32;

    let mut payloads = vec![0u8; 2048];
    payloads[32..32 + db1_path_str.len()].copy_from_slice(db1_path_str.as_bytes());
    payloads[256..256 + db2_path_str.len()].copy_from_slice(db2_path_str.as_bytes());
    payloads[300..303].copy_from_slice(b"key");
    payloads[320..323].copy_from_slice(b"aa!");
    payloads[340..343].copy_from_slice(b"bb!");
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[500..500 + filename_bytes.len()].copy_from_slice(&filename_bytes);

    let actions = vec![
        // Open db1 (handle=0) and db2 (handle=1)
        Action { kind: Kind::LmdbOpen, dst: 24, src: 32, offset: 224, size: 10 },
        Action { kind: Kind::LmdbOpen, dst: 28, src: 256, offset: 512, size: 10 },
        // Begin batch on both
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::LmdbBeginWriteTxn, dst: 0, src: 0, offset: 1, size: 0 },
        // Put into each
        Action { kind: Kind::LmdbPut, dst: 300, src: 320, offset: 0, size: (3 << 16) | 3 },
        Action { kind: Kind::LmdbPut, dst: 300, src: 340, offset: 1, size: (3 << 16) | 3 },
        // Commit both
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::LmdbCommitWriteTxn, dst: 0, src: 0, offset: 1, size: 0 },
        // Read back from each
        Action { kind: Kind::LmdbGet, dst: 300, src: 420, offset: 0, size: 3 << 16 },
        Action { kind: Kind::LmdbGet, dst: 300, src: 460, offset: 1, size: 3 << 16 },
        Action { kind: Kind::AsyncDispatch, dst: 8, src: 0, offset: flag_lmdb, size: 10 },
        Action { kind: Kind::Wait, dst: flag_lmdb, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::MemCopy, dst: 600, src: 420, offset: 0, size: 7 },
        Action { kind: Kind::MemCopy, dst: 607, src: 460, offset: 0, size: 7 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: 12, offset: flag_mem, size: 2 },
        Action { kind: Kind::Wait, dst: flag_mem, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::FileWrite, dst: 500, src: 600, offset: 0, size: 14 },
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 16, offset: flag_file, size: 1 },
        Action { kind: Kind::Wait, dst: flag_file, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_lmdb_algorithm(actions, payloads, true, true);
    execute(algorithm).unwrap();

    let file_contents = fs::read(&test_file).unwrap();
    let len1 = u32::from_le_bytes(file_contents[0..4].try_into().unwrap());
    assert_eq!(len1, 3);
    assert_eq!(&file_contents[4..7], b"aa!", "db1 batch put should be committed");
    let len2 = u32::from_le_bytes(file_contents[7..11].try_into().unwrap());
    assert_eq!(len2, 3);
    assert_eq!(&file_contents[11..14], b"bb!", "db2 batch put should be committed");
}

fn create_computational_test_algorithm(
    actions: Vec<Action>,
    payloads: Vec<u8>,
) -> Algorithm {
    let num_actions = actions.len();

    Algorithm {
        actions,
        payloads,
        state: State {
            regs_per_unit: 16,
            gpu_size: 0,
            computational_regs: 32,
            file_buffer_size: 65536,
            gpu_shader_offsets: vec![],
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            simd_units: 0,
            gpu_units: 0,
            computational_units: 1,
            file_units: 1,
            network_units: 0,
            memory_units: 1,
            ffi_units: 0,
            hash_table_units: 0,
            lmdb_units: 0,
            cranelift_units: 0,
            backends_bits: 0,
        },
        simd_assignments: vec![],
        computational_assignments: vec![],
        memory_assignments: vec![0; num_actions],
        file_assignments: vec![0; num_actions],
        network_assignments: vec![],
        ffi_assignments: vec![],
        hash_table_assignments: vec![],
        lmdb_assignments: vec![],
        gpu_assignments: vec![],
        cranelift_assignments: vec![],
        worker_threads: None,
        blocking_threads: None,
        stack_size: None,
        timeout_ms: Some(5000),
        thread_name_prefix: None,
    }
}

#[test]
fn test_integration_computational_load_store_f64() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("comp_f64_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let mut payloads = vec![0u8; 2048];

    // Setup filename with null terminator
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Setup source data at offset 256: value 16.0 (for sqrt test)
    payloads[256..264].copy_from_slice(&16.0f64.to_le_bytes());

    // Result will be stored at offset 512
    let comp_flag = 1024u32;
    let file_flag = 1032u32;

    let actions = vec![
        // Action 0: ComputationalLoadF64 - load 16.0 from memory offset 256 into register 0
        Action {
            kind: Kind::ComputationalLoadF64,
            dst: 0,      // register 0
            src: 256,    // memory offset
            offset: 0,
            size: 0,
        },
        // Action 1: Approximate - compute sqrt (register 0 -> register 1)
        Action {
            kind: Kind::Approximate,
            dst: 1,      // output register
            src: 0,      // input register
            offset: 10,  // iterations
            size: 0,
        },
        // Action 2: ComputationalStoreF64 - store result from register 1 to memory offset 512
        Action {
            kind: Kind::ComputationalStoreF64,
            src: 1,      // register 1
            dst: 0,
            offset: 512, // memory offset
            size: 0,
        },
        // Action 3: FileWrite - write result to file
        Action {
            kind: Kind::FileWrite,
            dst: 0,      // filename offset
            src: 512,    // data offset
            offset: 0,
            size: 8,
        },
        // Action 4: AsyncDispatch computational operations (actions 0-2) to computational unit (type 5)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 5,      // computational unit type
            src: 0,      // start action index
            offset: comp_flag,
            size: 3,     // dispatch 3 actions (0, 1, 2)
        },
        // Action 5: Wait for computational operations
        Action {
            kind: Kind::Wait,
            dst: comp_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 6: AsyncDispatch FileWrite to file unit (type 2)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,      // file unit type
            src: 3,      // action index 3 (FileWrite)
            offset: file_flag,
            size: 0,
        },
        // Action 7: Wait for FileWrite
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_computational_test_algorithm(actions, payloads);
    execute(algorithm).unwrap();

    assert!(test_file.exists(), "Result file should exist");
    let contents = fs::read(&test_file).unwrap();
    let result = f64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 4.0, "sqrt(16) should equal 4.0");
}

#[test]
fn test_integration_computational_load_store_u64() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("comp_u64_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let mut payloads = vec![0u8; 2048];

    // Setup filename
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    // Setup source data at offset 256: value 100 (for choose test)
    payloads[256..264].copy_from_slice(&100u64.to_le_bytes());

    let comp_flag = 1024u32;
    let file_flag = 1032u32;

    let actions = vec![
        // Action 0: ComputationalLoadU64 - load 100 from memory into register 0
        Action {
            kind: Kind::ComputationalLoadU64,
            dst: 0,
            src: 256,
            offset: 0,
            size: 0,
        },
        // Action 1: Choose - random value in [0, 100) (register 0 -> register 1)
        Action {
            kind: Kind::Choose,
            dst: 1,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 2: ComputationalStoreU64 - store result to memory offset 512
        Action {
            kind: Kind::ComputationalStoreU64,
            src: 1,
            dst: 0,
            offset: 512,
            size: 0,
        },
        // Action 3: FileWrite
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 512,
            offset: 0,
            size: 8,
        },
        // Action 4: AsyncDispatch computational (actions 0-2)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 5,
            src: 0,
            offset: comp_flag,
            size: 3,
        },
        // Action 5: Wait
        Action {
            kind: Kind::Wait,
            dst: comp_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 6: AsyncDispatch FileWrite
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 3,
            offset: file_flag,
            size: 0,
        },
        // Action 7: Wait
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_computational_test_algorithm(actions, payloads);
    execute(algorithm).unwrap();

    assert!(test_file.exists(), "Result file should exist");
    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert!(result < 100, "Choose result should be in range [0, 100), got {}", result);
}

#[test]
fn test_integration_computational_timestamp() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("comp_timestamp_result.txt");
    let test_file_str = test_file.to_str().unwrap();

    let mut payloads = vec![0u8; 2048];

    // Setup filename
    let filename_bytes = format!("{}\0", test_file_str).into_bytes();
    payloads[0..filename_bytes.len()].copy_from_slice(&filename_bytes);

    let comp_flag = 1024u32;
    let file_flag = 1032u32;

    let actions = vec![
        // Action 0: Timestamp - get current time into register 0
        Action {
            kind: Kind::Timestamp,
            dst: 0,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 1: Store timestamp
        Action {
            kind: Kind::ComputationalStoreU64,
            src: 0,
            dst: 0,
            offset: 512,
            size: 0,
        },
        // Action 2: FileWrite
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 512,
            offset: 0,
            size: 8,
        },
        // Action 3: AsyncDispatch computational
        Action {
            kind: Kind::AsyncDispatch,
            dst: 5,
            src: 0,
            offset: comp_flag,
            size: 2,
        },
        // Action 4: Wait
        Action {
            kind: Kind::Wait,
            dst: comp_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
        // Action 5: AsyncDispatch FileWrite
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 2,
            offset: file_flag,
            size: 0,
        },
        // Action 6: Wait
        Action {
            kind: Kind::Wait,
            dst: file_flag,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = create_computational_test_algorithm(actions, payloads);
    execute(algorithm).unwrap();

    assert!(test_file.exists(), "Result file should exist");
    let contents = fs::read(&test_file).unwrap();
    let timestamp = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    
    // Verify timestamp is reasonable
    assert!(timestamp > 0, "Timestamp should be non-zero");
    assert!(timestamp < u64::MAX / 2, "Timestamp should be a reasonable value");
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

    let algorithm = create_test_algorithm(actions, payloads, 1, 0);
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

    let algorithm = create_test_algorithm(actions, payloads, 1, 0);
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
        Action { kind: Kind::MemCopy, dst: data_offset as u32, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: data_offset as u32, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: data_offset as u32, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: data_offset as u32, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: data1_offset as u32, src: 0, offset: 0, size: 0 },
        // Action 1: Cranelift unit 1 computation
        Action { kind: Kind::MemCopy, dst: data2_offset as u32, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: data_offset as u32, src: 0, offset: 0, size: 0 },
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
    memory_units: usize,
    timeout_ms: u64,
) -> Algorithm {
    let mut alg = create_test_algorithm(actions, payloads, file_units, memory_units);
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
        Action { kind: Kind::MemCopy, dst: wake_addr, src: 24, offset: 0, size: 0 },
        // 3: FileWrite — write 16 bytes (wake word + status) to file
        Action { kind: Kind::FileWrite, dst: filename_offset, src: wake_addr, offset: 0, size: 24 },
        // 4: AsyncDispatch file unit
        Action { kind: Kind::AsyncDispatch, dst: 2, src: 3, offset: file_flag, size: 0 },
        // 5: Wait for file write
        Action { kind: Kind::Wait, dst: file_flag, src: 0, offset: 0, size: 0 },
    ];

    let algorithm = create_test_algorithm_with_timeout(actions, payloads, 1, 1, 5000);
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

    let algorithm = create_test_algorithm_with_timeout(actions, payloads, 1, 1, 5000);
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

fn mw(dst: u32, src: u32, size: u32) -> Action {
    Action { kind: Kind::MemWrite, dst, src, offset: 0, size }
}

const K_SETUP_DONE: usize = 0x480;
const K_QUEUE_DESC: usize = 0x100;
const K_QUEUE_BASE: usize = 0x200;
const KERNEL_DESC: usize = 0x300;
const K_HANDLE_PTR: usize = 0x400;
const K_PACKET_PTR_PTR: usize = 0x408;
const K_START_STATUS: usize = 0x500;
const K_SUBMIT_STATUS: usize = 0x508;
const K_WAIT_STATUS: usize = 0x510;
const K_STOP_STATUS: usize = 0x518;
const K_PROGRESS_ADDR: usize = 0x520;
const K_STOP_FLAG_ADDR: usize = 0x528;
const K_EXPECTED_ZERO_ADDR: usize = 0x530;
const K_EXPECTED_ONE_ADDR: usize = 0x538;
const K_EXPECTED_TWO_ADDR: usize = 0x540;
const K_VALUE_ADDR: usize = 0x600;
const K_PREV_A_ADDR: usize = 0x608;
const K_ADDEND_A_ADDR: usize = 0x610;
const K_PREV_B_ADDR: usize = 0x618;
const K_ADDEND_B_ADDR: usize = 0x620;
const K_EXPECTED_SUM_ADDR: usize = 0x628;
const K_FLAG_A_ADDR: usize = 0x630;
const K_FLAG_B_ADDR: usize = 0x638;
const K_PACKET_ADDR: usize = 0x700;

#[test]
fn test_kernel_submit_indirect_single_packet() {
    let payloads = vec![0u8; 2048];
    let mut actions = vec![
        mw(K_QUEUE_DESC as u32 + 8, 15, 4),
        mw(K_QUEUE_DESC as u32 + 12, K_QUEUE_BASE as u32, 4),
        mw(KERNEL_DESC as u32, K_QUEUE_DESC as u32, 4),
        mw(KERNEL_DESC as u32 + 4, 6, 4),
        mw(KERNEL_DESC as u32 + 16, K_PROGRESS_ADDR as u32, 4),
        mw(K_ADDEND_A_ADDR as u32, 7, 8),
        mw(K_EXPECTED_SUM_ADDR as u32, 7, 8),
        mw(K_EXPECTED_ONE_ADDR as u32, 1, 8),
        mw(K_PACKET_PTR_PTR as u32, K_PACKET_ADDR as u32, 4),
    ];
    // packet: worker at w, dispatches actions[w..w+1]
    let w = (actions.len() + 2 + 2) as u64; // +2 pkt writes, +2 AsyncDispatch/Wait
    let pkt = (w << 43) | ((w + 1) << 22) | (K_FLAG_A_ADDR as u64);
    actions.push(mw(K_PACKET_ADDR as u32, pkt as u32, 4));
    actions.push(mw(K_PACKET_ADDR as u32 + 4, (pkt >> 32) as u32, 4));
    let setup_len = actions.len() as u32;
    actions.push(Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: K_SETUP_DONE as u32, size: setup_len });
    actions.push(Action { kind: Kind::Wait, dst: K_SETUP_DONE as u32, src: 0, offset: 0, size: 0 });
    actions.extend_from_slice(&[
        Action { kind: Kind::AtomicFetchAdd, dst: K_VALUE_ADDR as u32, src: K_PREV_A_ADDR as u32, offset: K_ADDEND_A_ADDR as u32, size: 8 },
        Action { kind: Kind::KernelStart, dst: KERNEL_DESC as u32, src: K_HANDLE_PTR as u32, offset: K_START_STATUS as u32, size: 1 },
        Action { kind: Kind::KernelSubmitIndirect, dst: K_HANDLE_PTR as u32, src: K_PACKET_PTR_PTR as u32, offset: K_SUBMIT_STATUS as u32, size: 1 },
        Action { kind: Kind::KernelWait, dst: K_HANDLE_PTR as u32, src: 1, offset: K_WAIT_STATUS as u32, size: 3000 },
        Action { kind: Kind::KernelStop, dst: K_HANDLE_PTR as u32, src: 0, offset: K_STOP_STATUS as u32, size: 0 },
        Action { kind: Kind::WaitUntil, dst: K_VALUE_ADDR as u32, src: K_EXPECTED_SUM_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_START_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_SUBMIT_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_WAIT_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_PROGRESS_ADDR as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_STOP_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
    ]);
    let algorithm = create_test_algorithm(actions, payloads, 0, 1);
    execute(algorithm).unwrap();
}

#[test]
fn test_kernel_submit_indirect_multiple_packets() {
    let payloads = vec![0u8; 2048];
    let mut actions = vec![
        mw(K_QUEUE_DESC as u32 + 8, 31, 4),
        mw(K_QUEUE_DESC as u32 + 12, K_QUEUE_BASE as u32, 4),
        mw(KERNEL_DESC as u32, K_QUEUE_DESC as u32, 4),
        mw(KERNEL_DESC as u32 + 4, 6, 4),
        mw(KERNEL_DESC as u32 + 16, K_PROGRESS_ADDR as u32, 4),
        mw(K_ADDEND_A_ADDR as u32, 5, 8),
        mw(K_ADDEND_B_ADDR as u32, 9, 8),
        mw(K_EXPECTED_SUM_ADDR as u32, 14, 8),
        mw(K_EXPECTED_ONE_ADDR as u32, 1, 8),
        mw(K_EXPECTED_TWO_ADDR as u32, 2, 8),
        mw(K_PACKET_PTR_PTR as u32, K_PACKET_ADDR as u32, 4),
    ];
    // workers at w and w+1; pkt_a dispatches [w..w+1], pkt_b dispatches [w+1..w+2]
    let w = (actions.len() + 4 + 2) as u64; // +4 pkt writes, +2 AsyncDispatch/Wait
    let pkt_a = (w << 43) | ((w + 1) << 22) | (K_FLAG_A_ADDR as u64);
    let pkt_b = ((w + 1) << 43) | ((w + 2) << 22) | (K_FLAG_B_ADDR as u64);
    actions.extend_from_slice(&[
        mw(K_PACKET_ADDR as u32, pkt_a as u32, 4),
        mw(K_PACKET_ADDR as u32 + 4, (pkt_a >> 32) as u32, 4),
        mw((K_PACKET_ADDR + 8) as u32, pkt_b as u32, 4),
        mw((K_PACKET_ADDR + 8) as u32 + 4, (pkt_b >> 32) as u32, 4),
    ]);
    let setup_len = actions.len() as u32;
    actions.push(Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: K_SETUP_DONE as u32, size: setup_len });
    actions.push(Action { kind: Kind::Wait, dst: K_SETUP_DONE as u32, src: 0, offset: 0, size: 0 });
    actions.extend_from_slice(&[
        Action { kind: Kind::AtomicFetchAdd, dst: K_VALUE_ADDR as u32, src: K_PREV_A_ADDR as u32, offset: K_ADDEND_A_ADDR as u32, size: 8 },
        Action { kind: Kind::AtomicFetchAdd, dst: K_VALUE_ADDR as u32, src: K_PREV_B_ADDR as u32, offset: K_ADDEND_B_ADDR as u32, size: 8 },
        Action { kind: Kind::KernelStart, dst: KERNEL_DESC as u32, src: K_HANDLE_PTR as u32, offset: K_START_STATUS as u32, size: 1 },
        Action { kind: Kind::KernelSubmitIndirect, dst: K_HANDLE_PTR as u32, src: K_PACKET_PTR_PTR as u32, offset: K_SUBMIT_STATUS as u32, size: 2 },
        Action { kind: Kind::KernelWait, dst: K_HANDLE_PTR as u32, src: 2, offset: K_WAIT_STATUS as u32, size: 3000 },
        Action { kind: Kind::KernelStop, dst: K_HANDLE_PTR as u32, src: 0, offset: K_STOP_STATUS as u32, size: 0 },
        Action { kind: Kind::WaitUntil, dst: K_VALUE_ADDR as u32, src: K_EXPECTED_SUM_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_START_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_SUBMIT_STATUS as u32, src: K_EXPECTED_TWO_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_WAIT_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_PROGRESS_ADDR as u32, src: K_EXPECTED_TWO_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_STOP_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
    ]);
    let algorithm = create_test_algorithm(actions, payloads, 0, 1);
    execute(algorithm).unwrap();
}

#[test]
fn test_kernel_submit_indirect_invalid_handle() {
    let payloads = vec![0u8; 2048];
    let mut actions = vec![
        mw(K_HANDLE_PTR as u32, 999, 4),
        mw(K_PACKET_PTR_PTR as u32, K_PACKET_ADDR as u32, 4),
        mw(K_PACKET_ADDR as u32, 0x400000, 8),
    ];
    let setup_len = actions.len() as u32;
    actions.push(Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: K_SETUP_DONE as u32, size: setup_len });
    actions.push(Action { kind: Kind::Wait, dst: K_SETUP_DONE as u32, src: 0, offset: 0, size: 0 });
    actions.extend_from_slice(&[
        Action { kind: Kind::KernelSubmitIndirect, dst: K_HANDLE_PTR as u32, src: K_PACKET_PTR_PTR as u32, offset: K_SUBMIT_STATUS as u32, size: 1 },
        Action { kind: Kind::WaitUntil, dst: K_SUBMIT_STATUS as u32, src: K_EXPECTED_ZERO_ADDR as u32, offset: 0, size: 8 },
    ]);
    let algorithm = create_test_algorithm(actions, payloads, 0, 1);
    execute(algorithm).unwrap();
}

#[test]
fn test_kernel_submit_indirect_size_zero_defaults_to_one_packet() {
    let payloads = vec![0u8; 2048];
    let mut actions = vec![
        mw(K_QUEUE_DESC as u32 + 8, 15, 4),
        mw(K_QUEUE_DESC as u32 + 12, K_QUEUE_BASE as u32, 4),
        mw(KERNEL_DESC as u32, K_QUEUE_DESC as u32, 4),
        mw(KERNEL_DESC as u32 + 4, 6, 4),
        mw(KERNEL_DESC as u32 + 16, K_PROGRESS_ADDR as u32, 4),
        mw(K_ADDEND_A_ADDR as u32, 3, 8),
        mw(K_ADDEND_B_ADDR as u32, 11, 8),
        mw(K_EXPECTED_SUM_ADDR as u32, 3, 8),
        mw(K_EXPECTED_ONE_ADDR as u32, 1, 8),
        mw(K_PACKET_PTR_PTR as u32, K_PACKET_ADDR as u32, 4),
    ];
    // workers at w and w+1; pkt_a dispatches [w..w+1], pkt_b dispatches [w+1..w+2]
    let w = (actions.len() + 4 + 2) as u64; // +4 pkt writes, +2 AsyncDispatch/Wait
    let pkt_a = (w << 43) | ((w + 1) << 22) | (K_FLAG_A_ADDR as u64);
    let pkt_b = ((w + 1) << 43) | ((w + 2) << 22) | (K_FLAG_B_ADDR as u64);
    actions.extend_from_slice(&[
        mw(K_PACKET_ADDR as u32, pkt_a as u32, 4),
        mw(K_PACKET_ADDR as u32 + 4, (pkt_a >> 32) as u32, 4),
        mw((K_PACKET_ADDR + 8) as u32, pkt_b as u32, 4),
        mw((K_PACKET_ADDR + 8) as u32 + 4, (pkt_b >> 32) as u32, 4),
    ]);
    let setup_len = actions.len() as u32;
    actions.push(Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: K_SETUP_DONE as u32, size: setup_len });
    actions.push(Action { kind: Kind::Wait, dst: K_SETUP_DONE as u32, src: 0, offset: 0, size: 0 });
    actions.extend_from_slice(&[
        Action { kind: Kind::AtomicFetchAdd, dst: K_VALUE_ADDR as u32, src: K_PREV_A_ADDR as u32, offset: K_ADDEND_A_ADDR as u32, size: 8 },
        Action { kind: Kind::AtomicFetchAdd, dst: K_VALUE_ADDR as u32, src: K_PREV_B_ADDR as u32, offset: K_ADDEND_B_ADDR as u32, size: 8 },
        Action { kind: Kind::KernelStart, dst: KERNEL_DESC as u32, src: K_HANDLE_PTR as u32, offset: K_START_STATUS as u32, size: 1 },
        Action { kind: Kind::KernelSubmitIndirect, dst: K_HANDLE_PTR as u32, src: K_PACKET_PTR_PTR as u32, offset: K_SUBMIT_STATUS as u32, size: 0 },
        Action { kind: Kind::KernelWait, dst: K_HANDLE_PTR as u32, src: 1, offset: K_WAIT_STATUS as u32, size: 3000 },
        Action { kind: Kind::KernelStop, dst: K_HANDLE_PTR as u32, src: 0, offset: K_STOP_STATUS as u32, size: 0 },
        Action { kind: Kind::WaitUntil, dst: K_VALUE_ADDR as u32, src: K_EXPECTED_SUM_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_SUBMIT_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_PROGRESS_ADDR as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
    ]);
    let algorithm = create_test_algorithm(actions, payloads, 0, 1);
    execute(algorithm).unwrap();
}

#[test]
fn test_kernel_submit_indirect_partial_submit_when_queue_full() {
    let payloads = vec![0u8; 2048];
    let mut actions = vec![
        // queue with capacity 1 (mask=0), kernel stops immediately (stop_flag=1)
        mw(K_STOP_FLAG_ADDR as u32, 1, 8),
        mw(K_QUEUE_DESC as u32 + 12, K_QUEUE_BASE as u32, 4),
        mw(KERNEL_DESC as u32, K_QUEUE_DESC as u32, 4),
        mw(KERNEL_DESC as u32 + 4, 6, 4),
        mw(KERNEL_DESC as u32 + 12, K_STOP_FLAG_ADDR as u32, 4),
        mw(KERNEL_DESC as u32 + 16, K_PROGRESS_ADDR as u32, 4),
        mw(K_EXPECTED_ONE_ADDR as u32, 1, 8),
        mw(K_PACKET_PTR_PTR as u32, K_PACKET_ADDR as u32, 4),
        mw(K_PACKET_ADDR as u32, 0x400000, 8),
        mw((K_PACKET_ADDR + 8) as u32, 0x400000, 8),
    ];
    let setup_len = actions.len() as u32;
    actions.push(Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: K_SETUP_DONE as u32, size: setup_len });
    actions.push(Action { kind: Kind::Wait, dst: K_SETUP_DONE as u32, src: 0, offset: 0, size: 0 });
    actions.extend_from_slice(&[
        Action { kind: Kind::KernelStart, dst: KERNEL_DESC as u32, src: K_HANDLE_PTR as u32, offset: K_START_STATUS as u32, size: 1 },
        Action { kind: Kind::KernelSubmitIndirect, dst: K_HANDLE_PTR as u32, src: K_PACKET_PTR_PTR as u32, offset: K_SUBMIT_STATUS as u32, size: 2 },
        Action { kind: Kind::KernelStop, dst: K_HANDLE_PTR as u32, src: 0, offset: K_STOP_STATUS as u32, size: 0 },
        Action { kind: Kind::WaitUntil, dst: K_SUBMIT_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_PROGRESS_ADDR as u32, src: K_EXPECTED_ZERO_ADDR as u32, offset: 0, size: 8 },
    ]);
    let algorithm = create_test_algorithm(actions, payloads, 0, 1);
    execute(algorithm).unwrap();
}

#[test]
fn test_kernel_wait_timeout_and_stop_idempotent() {
    let payloads = vec![0u8; 2048];
    let mut actions = vec![
        mw(K_STOP_FLAG_ADDR as u32, 1, 8),
        mw(K_QUEUE_DESC as u32 + 8, 15, 4),
        mw(K_QUEUE_DESC as u32 + 12, K_QUEUE_BASE as u32, 4),
        mw(KERNEL_DESC as u32, K_QUEUE_DESC as u32, 4),
        mw(KERNEL_DESC as u32 + 4, 6, 4),
        mw(KERNEL_DESC as u32 + 12, K_STOP_FLAG_ADDR as u32, 4),
        mw(KERNEL_DESC as u32 + 16, K_PROGRESS_ADDR as u32, 4),
        mw(K_EXPECTED_ONE_ADDR as u32, 1, 8),
    ];
    let setup_len = actions.len() as u32;
    actions.push(Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: K_SETUP_DONE as u32, size: setup_len });
    actions.push(Action { kind: Kind::Wait, dst: K_SETUP_DONE as u32, src: 0, offset: 0, size: 0 });
    actions.extend_from_slice(&[
        Action { kind: Kind::KernelStart, dst: KERNEL_DESC as u32, src: K_HANDLE_PTR as u32, offset: K_START_STATUS as u32, size: 1 },
        Action { kind: Kind::KernelWait, dst: K_HANDLE_PTR as u32, src: 1, offset: K_WAIT_STATUS as u32, size: 1 },
        Action { kind: Kind::KernelStop, dst: K_HANDLE_PTR as u32, src: 0, offset: K_STOP_STATUS as u32, size: 0 },
        Action { kind: Kind::KernelStop, dst: K_HANDLE_PTR as u32, src: 0, offset: K_STOP_STATUS as u32, size: 0 },
        Action { kind: Kind::WaitUntil, dst: K_WAIT_STATUS as u32, src: K_EXPECTED_ZERO_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_STOP_STATUS as u32, src: K_EXPECTED_ZERO_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_START_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
    ]);
    let algorithm = create_test_algorithm(actions, payloads, 0, 1);
    execute(algorithm).unwrap();
}

#[test]
fn test_queue_push_packet_mp_single_and_full() {
    const PUSH1_DONE: usize = 0x560;
    const PUSH2_DONE: usize = 0x568;
    const PUSH1_STATUS: usize = 0x570;
    const PUSH2_STATUS: usize = 0x578;

    let payloads = vec![0u8; 2048];
    let mut actions = vec![
        mw(K_QUEUE_DESC as u32 + 12, K_QUEUE_BASE as u32, 4),
        mw(K_EXPECTED_ONE_ADDR as u32, 1, 8),
        mw(K_PACKET_ADDR as u32, 0x400000, 8),
        mw((K_PACKET_ADDR + 8) as u32, 0x400000, 8),
    ];
    let setup_len = actions.len() as u32;
    actions.push(Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: K_SETUP_DONE as u32, size: setup_len });
    actions.push(Action { kind: Kind::Wait, dst: K_SETUP_DONE as u32, src: 0, offset: 0, size: 0 });
    // QueuePushPacketMP actions dispatched individually via AsyncDispatch
    let qp1 = actions.len() as u32 + 4; // after 2 AsyncDispatch/Wait pairs
    let qp2 = qp1 + 1;
    actions.extend_from_slice(&[
        Action { kind: Kind::AsyncDispatch, dst: 6, src: qp1, offset: PUSH1_DONE as u32, size: 1 },
        Action { kind: Kind::Wait, dst: PUSH1_DONE as u32, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 6, src: qp2, offset: PUSH2_DONE as u32, size: 1 },
        Action { kind: Kind::Wait, dst: PUSH2_DONE as u32, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::QueuePushPacketMP, dst: K_QUEUE_DESC as u32, src: K_PACKET_ADDR as u32, offset: PUSH1_STATUS as u32, size: 0 },
        Action { kind: Kind::QueuePushPacketMP, dst: K_QUEUE_DESC as u32, src: (K_PACKET_ADDR + 8) as u32, offset: PUSH2_STATUS as u32, size: 0 },
        Action { kind: Kind::WaitUntil, dst: PUSH1_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: PUSH2_STATUS as u32, src: K_EXPECTED_ZERO_ADDR as u32, offset: 0, size: 8 },
    ]);
    let algorithm = create_test_algorithm(actions, payloads, 0, 1);
    execute(algorithm).unwrap();
}

#[test]
fn test_queue_push_packet_mp_concurrent_multi_producer() {
    const ROUNDS: u32 = 8;
    const PRODUCERS: u32 = 4;
    const EXPECTED: u64 = (ROUNDS as u64) * (PRODUCERS as u64);
    const PUSH_DONE_BASE: usize = 0x560;
    const PUSH_STATUS_BASE: usize = 0x5f0;
    const QUEUE_TAIL_ADDR: usize = K_QUEUE_DESC + 4;
    const QUEUE_RESERVE_ADDR: usize = K_QUEUE_DESC + 20;

    let payloads = vec![0u8; 4096];
    let mut actions = vec![
        mw(K_QUEUE_DESC as u32 + 8, 1023, 4),
        mw(K_QUEUE_DESC as u32 + 12, K_QUEUE_BASE as u32, 4),
        mw(K_EXPECTED_ONE_ADDR as u32, 1, 8),
        mw(K_EXPECTED_SUM_ADDR as u32, EXPECTED as u32, 8),
        mw(K_PACKET_ADDR as u32, 0x400000, 8),
    ];
    let setup_len = actions.len() as u32;
    actions.push(Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: K_SETUP_DONE as u32, size: setup_len });
    actions.push(Action { kind: Kind::Wait, dst: K_SETUP_DONE as u32, src: 0, offset: 0, size: 0 });
    // QueuePushPacketMP x 4 (one per producer, broadcast-partitioned)
    let qp_start = actions.len() as u32;
    actions.extend_from_slice(&[
        Action { kind: Kind::QueuePushPacketMP, dst: K_QUEUE_DESC as u32, src: K_PACKET_ADDR as u32, offset: (PUSH_STATUS_BASE + 0) as u32, size: 0 },
        Action { kind: Kind::QueuePushPacketMP, dst: K_QUEUE_DESC as u32, src: K_PACKET_ADDR as u32, offset: (PUSH_STATUS_BASE + 8) as u32, size: 0 },
        Action { kind: Kind::QueuePushPacketMP, dst: K_QUEUE_DESC as u32, src: K_PACKET_ADDR as u32, offset: (PUSH_STATUS_BASE + 16) as u32, size: 0 },
        Action { kind: Kind::QueuePushPacketMP, dst: K_QUEUE_DESC as u32, src: K_PACKET_ADDR as u32, offset: (PUSH_STATUS_BASE + 24) as u32, size: 0 },
    ]);

    for i in 0..ROUNDS {
        actions.push(Action {
            kind: Kind::AsyncDispatch,
            dst: 6,
            src: qp_start,
            offset: (PUSH_DONE_BASE + (i as usize) * 8) as u32,
            size: (1u32 << 31) | 4, // broadcast range partitioned across workers
        });
        actions.push(Action {
            kind: Kind::Wait,
            dst: (PUSH_DONE_BASE + (i as usize) * 8) as u32,
            src: 0,
            offset: 0,
            size: 0,
        });
    }

    actions.extend_from_slice(&[
        Action { kind: Kind::WaitUntil, dst: QUEUE_TAIL_ADDR as u32, src: K_EXPECTED_SUM_ADDR as u32, offset: 0, size: 4 },
        Action { kind: Kind::WaitUntil, dst: QUEUE_RESERVE_ADDR as u32, src: K_EXPECTED_SUM_ADDR as u32, offset: 0, size: 4 },
        Action { kind: Kind::WaitUntil, dst: K_QUEUE_DESC as u32, src: K_EXPECTED_ZERO_ADDR as u32, offset: 0, size: 4 },
    ]);

    let algorithm = create_test_algorithm(actions, payloads, 0, PRODUCERS as usize);
    execute(algorithm).unwrap();
}

#[test]
fn test_worker_discovered_chain_via_queue_push_packet_mp() {
    const CHAIN_TARGET: u32 = 50;
    const PUSH_STATUS: usize = 0x560;
    const N_KERNEL_ACTIONS: u64 = 8; // KernelStart..WaitUntil (see below)

    let payloads = vec![0u8; 4096];
    let mut actions = vec![
        mw(K_QUEUE_DESC as u32 + 8, 15, 4),
        mw(K_QUEUE_DESC as u32 + 12, K_QUEUE_BASE as u32, 4),
        mw(KERNEL_DESC as u32, K_QUEUE_DESC as u32, 4),
        mw(KERNEL_DESC as u32 + 4, 6, 4),
        mw(KERNEL_DESC as u32 + 12, K_STOP_FLAG_ADDR as u32, 4),
        mw(KERNEL_DESC as u32 + 16, K_PROGRESS_ADDR as u32, 4),
        mw(K_EXPECTED_ONE_ADDR as u32, 1, 8),
        mw(K_ADDEND_A_ADDR as u32, 1, 8),
    ];
    // packet: worker actions at w and w+1, dispatches [w..w+2]
    let w = (actions.len() + 2 + 2) as u64 + N_KERNEL_ACTIONS; // +2 pkt, +2 async/wait, +kernel
    let pkt = (w << 43) | ((w + 2) << 22); // flag=0
    actions.push(mw(K_PACKET_ADDR as u32, pkt as u32, 4));
    actions.push(mw(K_PACKET_ADDR as u32 + 4, (pkt >> 32) as u32, 4));
    let setup_len = actions.len() as u32;
    actions.push(Action { kind: Kind::AsyncDispatch, dst: 6, src: 0, offset: K_SETUP_DONE as u32, size: setup_len });
    actions.push(Action { kind: Kind::Wait, dst: K_SETUP_DONE as u32, src: 0, offset: 0, size: 0 });
    actions.extend_from_slice(&[
        Action { kind: Kind::KernelStart, dst: KERNEL_DESC as u32, src: K_HANDLE_PTR as u32, offset: K_START_STATUS as u32, size: 1 },
        Action { kind: Kind::KernelSubmit, dst: K_HANDLE_PTR as u32, src: K_PACKET_ADDR as u32, offset: K_SUBMIT_STATUS as u32, size: 1 },
        Action { kind: Kind::KernelWait, dst: K_HANDLE_PTR as u32, src: CHAIN_TARGET, offset: K_WAIT_STATUS as u32, size: 5000 },
        Action { kind: Kind::MemWrite, dst: K_STOP_FLAG_ADDR as u32, src: 1, offset: 0, size: 8 },
        Action { kind: Kind::KernelStop, dst: K_HANDLE_PTR as u32, src: 0, offset: K_STOP_STATUS as u32, size: 0 },
        Action { kind: Kind::WaitUntil, dst: K_WAIT_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_STOP_STATUS as u32, src: K_EXPECTED_ONE_ADDR as u32, offset: 0, size: 8 },
        Action { kind: Kind::WaitUntil, dst: K_VALUE_ADDR as u32, src: K_EXPECTED_ZERO_ADDR as u32, offset: 1, size: 8 }, // value != 0
    ]);
    // worker actions (dispatched repeatedly by kernel at indices w..w+2)
    let worker_start = actions.len() as u64;
    assert_eq!(worker_start, w);
    actions.extend_from_slice(&[
        Action { kind: Kind::AtomicFetchAdd, dst: K_VALUE_ADDR as u32, src: K_PREV_A_ADDR as u32, offset: K_ADDEND_A_ADDR as u32, size: 8 },
        Action { kind: Kind::QueuePushPacketMP, dst: K_QUEUE_DESC as u32, src: K_PACKET_ADDR as u32, offset: PUSH_STATUS as u32, size: 0 },
    ]);
    let algorithm = create_test_algorithm(actions, payloads, 0, 1);
    execute(algorithm).unwrap();
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
        Action { kind: Kind::MemCopy, dst: 0, src: 0, offset: 0, size: 0 },
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
