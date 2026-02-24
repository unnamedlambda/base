use base::{execute, Algorithm};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

const ALGORITHM_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/algorithm.bin"));

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

    let alg: Algorithm = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize algorithm binary");

    let start = std::time::Instant::now();
    match execute(alg) {
        Ok(()) => {
            let elapsed = start.elapsed();
            eprintln!("CSV demo completed in {:.1}ms", elapsed.as_secs_f64() * 1000.0);
            eprintln!("Output files: scan.csv, filter.csv, join.csv");
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
