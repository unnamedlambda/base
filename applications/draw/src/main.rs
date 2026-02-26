use base::{run, Algorithm, BaseConfig};
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

    let (config, alg): (BaseConfig, Algorithm) = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize (BaseConfig, Algorithm) binary");

    let start = std::time::Instant::now();
    match run(config, alg) {
        Ok(_) => {
            let elapsed = start.elapsed();
            eprintln!("Mandelbrot 4096x4096 rendered in {:.1}ms", elapsed.as_secs_f64() * 1000.0);
            eprintln!("Output: mandelbrot.bmp");
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
