mod csv_bench;
mod harness;

use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

fn main() {
    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_writer(std::io::stderr)
                .with_target(true)
                .with_filter(
                    EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| EnvFilter::new("off")),
                ),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let iterations: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let results = csv_bench::run(iterations);
    harness::print_table(&results);
}
