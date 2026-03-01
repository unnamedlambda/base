use base::{run, Algorithm, BaseConfig};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

const ALGORITHM_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/algorithm.bin"));

/// Payload offset where the input filename is stored (must match MakeAlgorithm.lean).
const INPUT_FILENAME_OFF: usize = 0x100;

fn main() {
    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_writer(std::io::stderr)
                .with_target(true)
                .with_thread_ids(true)
                .with_filter(
                    EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| EnvFilter::new("off")),
                ),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: sha256 <input_file>");
        std::process::exit(1);
    }
    let input_path = &args[1];

    let (config, mut alg): (BaseConfig, Algorithm) = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize (BaseConfig, Algorithm) binary");

    // Write input filename into payload (null-terminated)
    let path_bytes = input_path.as_bytes();
    assert!(
        path_bytes.len() < 255,
        "Input path too long (max 254 chars)"
    );
    alg.payloads[INPUT_FILENAME_OFF..INPUT_FILENAME_OFF + path_bytes.len()]
        .copy_from_slice(path_bytes);
    alg.payloads[INPUT_FILENAME_OFF + path_bytes.len()] = 0;

    match run(config, alg) {
        Ok(_) => {
            match std::fs::read_to_string("sha256_output.txt") {
                Ok(result) => print!("{}", result),
                Err(e) => eprintln!("Failed to read output: {}", e),
            }
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
