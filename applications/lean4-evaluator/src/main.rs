use base::{execute, Algorithm};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

const ALGORITHM_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/algorithm.bin"));
const INPUT_PATH_OFFSET: usize = 0x0060;
const INPUT_PATH_MAX_LEN: usize = 256;
const OUTPUT_PATH_OFFSET: usize = 0x0020;
const OUTPUT_PATH_MAX_LEN: usize = 64;

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

    let mut alg: Algorithm = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize algorithm binary");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: lean4-eval <source.lean> [output_file]");
        std::process::exit(1);
    }

    // Patch input file path
    let input_path = &args[1];
    let input_len = input_path.len().min(INPUT_PATH_MAX_LEN - 1);
    alg.payloads[INPUT_PATH_OFFSET..INPUT_PATH_OFFSET + INPUT_PATH_MAX_LEN].fill(0);
    alg.payloads[INPUT_PATH_OFFSET..INPUT_PATH_OFFSET + input_len]
        .copy_from_slice(&input_path.as_bytes()[..input_len]);

    // Patch output file path if provided
    if args.len() > 2 {
        let output_path = &args[2];
        let output_len = output_path.len().min(OUTPUT_PATH_MAX_LEN - 1);
        alg.payloads[OUTPUT_PATH_OFFSET..OUTPUT_PATH_OFFSET + OUTPUT_PATH_MAX_LEN].fill(0);
        alg.payloads[OUTPUT_PATH_OFFSET..OUTPUT_PATH_OFFSET + output_len]
            .copy_from_slice(&output_path.as_bytes()[..output_len]);
    }

    match execute(alg) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
