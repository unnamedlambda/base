mod argv;

use base::execute;
use base_types::Algorithm;

const ALGORITHM_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/algorithm.bin"));
const OUTPUT_PATH_OFFSET: usize = 0x0010;
const OUTPUT_PATH_MAX_LEN: usize = 16;

fn main() {
    // Initialize argv module (stores args for FFI access)
    argv::init();

    // Deserialize algorithm
    let mut alg: Algorithm = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize algorithm binary");

    // Patch FFI function pointers into payloads
    // Offset 0: get_argv function pointer (8 bytes)
    let fn_ptr = argv::get_argv_ptr() as u64;
    alg.payloads[0..8].copy_from_slice(&fn_ptr.to_le_bytes());

    // Patch output file path if provided as second argument
    // Offset 0x0010: output file path (null-terminated, max 16 bytes)
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 2 {
        let output_path = &args[2];
        let path_bytes = output_path.as_bytes();
        let copy_len = path_bytes.len().min(OUTPUT_PATH_MAX_LEN - 1);
        // Clear the path area first
        alg.payloads[OUTPUT_PATH_OFFSET..OUTPUT_PATH_OFFSET + OUTPUT_PATH_MAX_LEN].fill(0);
        // Copy the new path
        alg.payloads[OUTPUT_PATH_OFFSET..OUTPUT_PATH_OFFSET + copy_len]
            .copy_from_slice(&path_bytes[..copy_len]);
    }

    // Execute
    match execute(alg) {
        Ok(()) => {}
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
