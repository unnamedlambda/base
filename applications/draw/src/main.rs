use base::{execute, Algorithm};

const ALGORITHM_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/algorithm.bin"));

fn main() {
    let alg: Algorithm = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize algorithm binary");

    match execute(alg) {
        Ok(()) => {}
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
