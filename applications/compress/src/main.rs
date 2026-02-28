use base::{run, Algorithm, BaseConfig};
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

    let start = std::time::Instant::now();
    match run(config, alg) {
        Ok(_) => {
            let elapsed = start.elapsed();
            // Parse standard LZ4 frame to compute actual compressed data size
            let mut actual_compressed = 0u64;
            let file_size = std::fs::metadata("compress_output.lz4")
                .map(|m| m.len())
                .unwrap_or(0);
            if file_size >= 11 {
                let data = std::fs::read("compress_output.lz4").unwrap_or_default();
                // Skip 7-byte frame header (magic + FLG + BD + HC)
                let mut pos = 7usize;
                while pos + 4 <= data.len() {
                    let block_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                    if block_size == 0 {
                        break; // end mark
                    }
                    let size = (block_size & 0x7FFFFFFF) as u64;
                    actual_compressed += size;
                    pos += 4 + size as usize;
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
