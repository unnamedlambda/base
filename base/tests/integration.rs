use base::execute;
use base_types::{Action, Algorithm, Kind, QueueSpec, State, UnitSpec};
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
            unit_scratch_offsets: vec![],
            unit_scratch_size: 0,
            shared_data_offset: 0,
            shared_data_size: payload_size,
            gpu_offset: payload_size,
            gpu_size: 0,
            computational_regs: 0,
            file_buffer_size: 65536,
            gpu_shader_offsets: vec![],
        },
        queues: QueueSpec { capacity: 256 },
        units: UnitSpec {
            simd_units: 0,
            gpu_units: 0,
            computational_units: 0,
            file_units,
            network_units: 0,
            memory_units,
            ffi_units: 0,
            backends_bits: 0,
            features_bits: 0,
        },
        simd_assignments: vec![],
        computational_assignments: vec![],
        memory_assignments: vec![0; num_actions],
        file_assignments: vec![0; num_actions],
        network_assignments: vec![],
        ffi_assignments: vec![],
        gpu_assignments: vec![],
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

    let actions = vec![
        Action {
            kind: Kind::MemCopy,
            src: 256,
            dst: 264,
            offset: 0,
            size: 8,
        },
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 264,
            offset: filename_bytes.len() as u32,
            size: 8,
        },
    ];

    let algorithm = create_test_algorithm(actions, payloads, 1, 0);

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

    // Setup conditions: 1.0 (true) and 0.0 (false) as f64
    payloads[512..520].copy_from_slice(&1.0f64.to_le_bytes());
    payloads[520..528].copy_from_slice(&0.0f64.to_le_bytes());

    // Setup data value
    payloads[528..536].copy_from_slice(&42u64.to_le_bytes());

    let actions = vec![
        // Action 0: ConditionalJump with true condition (should jump to action 2)
        Action {
            kind: Kind::ConditionalJump,
            src: 512,
            dst: 2,
            offset: 0,
            size: 0,
        },
        // Action 1: FileWrite to path A (SKIPPED - jumped over)
        Action {
            kind: Kind::FileWrite,
            dst: 0,
            src: 528,
            offset: filename_a_bytes.len() as u32,
            size: 8,
        },
        // Action 2: ConditionalJump with false condition (should fall through to action 3)
        Action {
            kind: Kind::ConditionalJump,
            src: 520,
            dst: 99,
            offset: 0,
            size: 0,
        },
        // Action 3: FileWrite to path B (EXECUTED - fell through)
        Action {
            kind: Kind::FileWrite,
            dst: 256,
            src: 528,
            offset: filename_b_bytes.len() as u32,
            size: 8,
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

    // Buffer at 264 is initially empty (zeros)

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
