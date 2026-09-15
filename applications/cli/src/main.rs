use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("CliAlgorithm/cli_app");

/// Entry points of this artifact, as its generator numbers them.
const MAIN: u32 = 1;

fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);

    match run(artifact, MAIN) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
