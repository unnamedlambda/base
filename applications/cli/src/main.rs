use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("CliAlgorithm/cli_app");

fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);

    match run(artifact.setup, artifact.main) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
