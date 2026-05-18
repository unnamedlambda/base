use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/RaytraceAlgorithm/raytrace_app.bin"
));

fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);

    let start = std::time::Instant::now();
    match run(artifact.config, artifact.main) {
        Ok(_) => {
            let elapsed = start.elapsed();
            eprintln!(
                "Cornell box 4096x4096 rendered in {:.1}ms",
                elapsed.as_secs_f64() * 1000.0
            );
            eprintln!("Output: cornell_box.bmp");
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
