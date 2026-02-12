mod csv_bench;
mod dispatch_bench;
mod harness;

use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

fn print_usage() {
    eprintln!("Usage: benchmarks [OPTIONS]");
    eprintln!();
    eprintln!("  --bench <name>     Benchmark to run: csv, dispatch, all (default: all)");
    eprintln!("  --profile <name>   Profile: quick, medium, full (default: medium)");
    eprintln!("  --rounds <n>       Rounds per measurement (default: 5)");
    eprintln!("  --chunk <n>        Coarse chunk size for dispatch (default: 16384)");
    eprintln!("  --workers <n>      Worker threads for dispatch (default: auto)");
    eprintln!("  --help             Show this help");
}

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

    let mut bench = "all".to_string();
    let mut profile = "medium".to_string();
    let mut rounds: usize = 5;
    let mut chunk: usize = 16384;
    let mut workers: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16)
        .max(1);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bench" => {
                i += 1;
                if i < args.len() { bench = args[i].clone(); }
            }
            "--profile" => {
                i += 1;
                if i < args.len() { profile = args[i].clone(); }
            }
            "--rounds" => {
                i += 1;
                if i < args.len() { rounds = args[i].parse().unwrap_or(5); }
            }
            "--chunk" => {
                i += 1;
                if i < args.len() { chunk = args[i].parse().unwrap_or(16384); }
            }
            "--workers" => {
                i += 1;
                if i < args.len() { workers = args[i].parse().unwrap_or(workers); }
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("Unknown flag: {}", other);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let run_csv = bench == "all" || bench == "csv";
    let run_dispatch = bench == "all" || bench == "dispatch";

    if run_csv {
        let results = csv_bench::run(rounds);
        harness::print_table(&results);
    }

    if run_dispatch {
        let cfg = dispatch_bench::Config {
            profile,
            rounds,
            chunk,
            workers,
        };
        let results = dispatch_bench::run(&cfg);
        harness::print_dispatch_table(&results);
    }
}
