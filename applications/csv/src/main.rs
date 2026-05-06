use base::{init_tracing, run, Algorithm, BaseConfig};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/CsvAlgorithm/csv_app.bin"));

fn main() {
    init_tracing();

    let (config, alg): (BaseConfig, Algorithm) = bincode::deserialize(ARTIFACT_BINARY)
        .expect("Failed to deserialize (BaseConfig, Algorithm) binary");

    let start = std::time::Instant::now();
    match run(config, alg) {
        Ok(_) => {
            let elapsed = start.elapsed();
            eprintln!(
                "CSV demo completed in {:.1}ms",
                elapsed.as_secs_f64() * 1000.0
            );
            eprintln!("Output files: scan.csv, filter.csv, join.csv");
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
