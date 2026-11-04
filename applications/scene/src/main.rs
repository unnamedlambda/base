use crate::run;

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::SCENE_APP;


pub fn main(_args: Vec<String>) {

    let start = std::time::Instant::now();
    // SAFETY: the artifact is the one Lean built for this application.
    match unsafe { run(ARTIFACT_BINARY, "main", b"") } {
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
