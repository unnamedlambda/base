use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("BlackHoleAlgorithm/blackhole_app");


fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);

    let start = std::time::Instant::now();
    match run(artifact, "main") {
        Ok(_) => {
            let elapsed = start.elapsed();
            eprintln!(
                "Black hole render completed in {:.1}ms",
                elapsed.as_secs_f64() * 1000.0
            );
            eprintln!("Output: blackhole.bmp");
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
