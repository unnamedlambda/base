mod argv;

use base::execute;
use base_types::Algorithm;

const ALGORITHM_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/algorithm.bin"));

fn main() {
    // Initialize argv module (stores args for FFI access)
    argv::init();

    // Deserialize algorithm
    let mut alg: Algorithm = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize algorithm binary");

    // Patch function pointer into payloads at offset 0
    // The Lean algorithm reserves 8 bytes at the start for this
    let fn_ptr = argv::get_argv_ptr() as u64;
    let fn_ptr_bytes = fn_ptr.to_le_bytes();
    assert!(
        alg.payloads.len() >= 8,
        "Algorithm payloads too small for function pointer (need 8 bytes, got {})",
        alg.payloads.len()
    );
    alg.payloads[0..8].copy_from_slice(&fn_ptr_bytes);

    // Execute
    match execute(alg) {
        Ok(()) => {}
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
