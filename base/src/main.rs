use base::{execute, Algorithm};

const ALGORITHM_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/algorithm.bin"));

fn main() {
    let alg: Algorithm = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize algorithm binary");
        
    println!("\nExecuting...");
    match execute(alg) {
        Ok(()) => println!("✓ Execution completed successfully"),
        Err(e) => eprintln!("✗ Execution failed: {:?}", e),
    }
}