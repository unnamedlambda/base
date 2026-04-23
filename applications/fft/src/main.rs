use base::{run, Algorithm, BaseConfig};

const ALGORITHM_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/algorithm.bin"));

/// Payload offset where the input filename is stored (must match MakeAlgorithm.lean).
const INPUT_FILENAME_OFF: usize = 0x2200;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: fft <input.bin>");
        std::process::exit(1);
    }
    let input_path = &args[1];

    let (mut config, alg): (BaseConfig, Algorithm) = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize (BaseConfig, Algorithm) binary");

    // Write input filename into initial_memory (null-terminated)
    let path_bytes = input_path.as_bytes();
    assert!(
        path_bytes.len() < 255,
        "Input path too long (max 254 chars)"
    );
    config.initial_memory[INPUT_FILENAME_OFF..INPUT_FILENAME_OFF + path_bytes.len()]
        .copy_from_slice(path_bytes);
    config.initial_memory[INPUT_FILENAME_OFF + path_bytes.len()] = 0;

    let start = std::time::Instant::now();
    match run(config, alg) {
        Ok(_) => {
            let elapsed = start.elapsed();
            let input_size = std::fs::metadata(input_path).map(|m| m.len()).unwrap_or(0);
            let n = input_size / 8;
            eprintln!(
                "FFT of {} complex numbers in {:.1}ms",
                n,
                elapsed.as_secs_f64() * 1000.0
            );
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
