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
