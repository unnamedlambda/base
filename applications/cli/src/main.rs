use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::CLI_APP;


fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY).expect("the build checked this artifact");

    match run(artifact, "main") {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
