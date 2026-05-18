use base::{init_tracing, run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/CsvAlgorithm/csv_app.bin"));

fn main() {
    init_tracing();

    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);

    let start = std::time::Instant::now();
    match run(artifact.setup, artifact.main) {
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
