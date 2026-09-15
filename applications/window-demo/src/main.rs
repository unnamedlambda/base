use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("WindowDemoAlgorithm/window_demo");

/// Entry points of this artifact, as its generator numbers them.
const MAIN: u32 = 1;

fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    println!("Opening window — arrow keys to move, Esc or close to quit.");
    match run(artifact, MAIN) {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("window-demo failed: {e:?}"),
    }
}
