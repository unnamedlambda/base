mod csv_bench;
mod dispatch_bench;
mod dynamic_bench;
mod harness;
mod histogram_bench;
mod json_bench;
mod gpu_bench;
mod gpu_iter_bench;
mod matmul_bench;
mod memory_bench;
mod network_bench;
mod reduction_bench;
mod regex_bench;
mod sort_bench;
mod string_search_bench;
mod vecops_bench;
mod wordcount_bench;

use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

fn print_usage() {
    eprintln!("Usage: benchmarks [OPTIONS]");
    eprintln!();
    eprintln!("  --bench <name>     Benchmark to run: csv, json, regex, burn, vecops, reduction,");
    eprintln!("                     dispatch, dynamic, gpu, gpu-iter, memory, network,");
    eprintln!("                     histogram, sort, strsearch, wc, all (default: all)");
    eprintln!("  --rounds <n>       Rounds per measurement (default: 10)");
    eprintln!("  --profile <p>      Profile: quick, medium, full (default: medium)");
    eprintln!("  --chunk <n>        Coarse chunk size (default: 100000)");
    eprintln!("  --workers <n>      Worker threads (default: auto)");
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
    let mut rounds: usize = 10;
    let mut profile = "medium".to_string();
    let mut chunk: usize = 100_000;
    let mut workers: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bench" => {
                i += 1;
                if i < args.len() { bench = args[i].clone(); }
            }
            "--rounds" => {
                i += 1;
                if i < args.len() { rounds = args[i].parse().unwrap_or(5); }
            }
            "--profile" => {
                i += 1;
                if i < args.len() { profile = args[i].clone(); }
            }
            "--chunk" => {
                i += 1;
                if i < args.len() { chunk = args[i].parse().unwrap_or(100_000); }
            }
            "--workers" => {
                i += 1;
                if i < args.len() { workers = args[i].parse().unwrap_or(4).clamp(1, 16); }
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
    let run_json = bench == "all" || bench == "json";
    let run_regex = bench == "all" || bench == "regex";
    let run_matmul = bench == "all" || bench == "burn";
    let run_vecops = bench == "all" || bench == "burn" || bench == "vecops";
    let run_reduction = bench == "all" || bench == "burn" || bench == "reduction";
    let run_dispatch = bench == "all" || bench == "dispatch";
    let run_dynamic = bench == "all" || bench == "dynamic";
    let run_gpu = bench == "all" || bench == "gpu";
    let run_gpu_iter = bench == "all" || bench == "gpu-iter";
    let run_memory = bench == "all" || bench == "memory";
    let run_network = bench == "all" || bench == "network";
    let run_histogram = bench == "all" || bench == "histogram";
    let run_sort = bench == "all" || bench == "sort";
    let run_strsearch = bench == "all" || bench == "strsearch";
    let run_wc = bench == "all" || bench == "wc";

    if run_csv {
        let results = csv_bench::run(rounds);
        harness::print_table(&results);
    }

    if run_json {
        let results = json_bench::run(rounds);
        harness::print_table(&results);
    }

    if run_regex {
        let results = regex_bench::run(rounds);
        harness::print_table(&results);
    }

    if run_matmul {
        let results = matmul_bench::run(rounds);
        harness::print_burn_table(&results);
    }

    if run_vecops {
        let results = vecops_bench::run(rounds);
        harness::print_burn_table(&results);
    }

    if run_reduction {
        let results = reduction_bench::run(rounds);
        harness::print_burn_table(&results);
    }

    if run_dispatch {
        let cfg = dispatch_bench::Config {
            profile: profile.clone(),
            rounds,
            chunk,
            workers,
        };
        let results = dispatch_bench::run(&cfg);
        dispatch_bench::print_dispatch_table(&results);
    }

    if run_dynamic {
        let cfg = dynamic_bench::Config {
            profile: profile.clone(),
            rounds,
            workers,
        };
        let results = dynamic_bench::run(&cfg);
        harness::print_table(&results);
    }

    if run_gpu {
        let results = gpu_bench::run(rounds);
        gpu_bench::print_gpu_table(&results);
    }

    if run_gpu_iter {
        let results = gpu_iter_bench::run(rounds);
        gpu_iter_bench::print_iter_table(&results);
    }

    if run_memory {
        let results = memory_bench::run(rounds);
        memory_bench::print_memory_table(&results);
    }

    if run_network {
        let results = network_bench::run(rounds);
        network_bench::print_network_table(&results);
    }

    if run_histogram {
        let cfg = histogram_bench::HistConfig {
            rounds,
            workers,
        };
        let results = histogram_bench::run(&cfg);
        harness::print_table(&results);
    }

    if run_sort {
        let results = sort_bench::run(rounds);
        harness::print_table(&results);
    }

    if run_strsearch {
        let results = string_search_bench::run(rounds);
        harness::print_table(&results);
    }

    if run_wc {
        let results = wordcount_bench::run(rounds);
        harness::print_table(&results);
    }
}
