use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/CliAlgorithm/cli_app.bin"));

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
