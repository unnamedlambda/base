use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] = lean_artifacts::RAYTRACE_APP;


pub fn main(_args: Vec<String>) {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY).expect("the build checked this artifact");

    let start = std::time::Instant::now();
    // SAFETY: the artifact is the one Lean built for this application.
    match unsafe { run(artifact, "main") } {
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
