use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::SCENE_APP;


fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY).expect("the build checked this artifact");

    let start = std::time::Instant::now();
    match run(artifact, "main") {
        Ok(_) => {
            let elapsed = start.elapsed();
            eprintln!(
                "Scene render completed in {:.1}ms",
                elapsed.as_secs_f64() * 1000.0
            );
            eprintln!("Output: scene.bmp");
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
