use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::CLI_APP;


pub fn main(_args: Vec<String>) {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY).expect("the build checked this artifact");

    // SAFETY: the artifact is the one Lean built for this application.
    match unsafe { run(artifact, "main") } {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
