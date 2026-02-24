use base::{execute, Algorithm};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

const ALGORITHM_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/algorithm.bin"));

/// Payload offset where the input filename is stored (must match MakeAlgorithm.lean).
const INPUT_FILENAME_OFF: usize = 0x4200;

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
        eprintln!("Usage: compress <input_file>");
        std::process::exit(1);
    }
    let input_path = &args[1];

    let input_size = std::fs::metadata(input_path)
        .unwrap_or_else(|e| {
            eprintln!("Cannot read {}: {}", input_path, e);
            std::process::exit(1);
        })
        .len();

    let mut alg: Algorithm = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize algorithm binary");

    // Write input filename into payload (null-terminated)
    let path_bytes = input_path.as_bytes();
    assert!(
        path_bytes.len() < 255,
        "Input path too long (max 254 chars)"
    );
    alg.payloads[INPUT_FILENAME_OFF..INPUT_FILENAME_OFF + path_bytes.len()]
        .copy_from_slice(path_bytes);
    alg.payloads[INPUT_FILENAME_OFF + path_bytes.len()] = 0;

    let start = std::time::Instant::now();
    match execute(alg) {
        Ok(()) => {
            let elapsed = start.elapsed();
            // Read header to compute actual compressed data size
            let mut actual_compressed = 0u64;
            let file_size = std::fs::metadata("compress_output.lz4")
                .map(|m| m.len())
                .unwrap_or(0);
            if file_size >= 8 {
                let header = std::fs::read("compress_output.lz4").unwrap_or_default();
                if header.len() >= 8 {
                    let num_blocks = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
                    let header_size = 8 + num_blocks * 4;
                    if header.len() >= header_size {
                        for i in 0..num_blocks {
                            let off = 8 + i * 4;
                            let block_sz = u32::from_le_bytes(header[off..off + 4].try_into().unwrap());
                            actual_compressed += block_sz as u64;
                        }
                    }
                }
            }
            let ratio = if input_size > 0 {
                actual_compressed as f64 / input_size as f64
            } else {
                0.0
            };
            eprintln!(
                "Compressed {} ({} bytes) → {} bytes in {:.1}ms ({:.1}% ratio)",
                input_path,
                input_size,
                actual_compressed,
                elapsed.as_secs_f64() * 1000.0,
                ratio * 100.0,
            );
            eprintln!("Output: compress_output.lz4 ({} bytes on disk)", file_size);
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
