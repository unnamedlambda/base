use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("FallingSandAlgorithm/falling_sand");

/// Entry points of this artifact, as its generator numbers them.
const MAIN: u32 = 1;

fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    println!("Falling sand — hold LEFT: sand, RIGHT: walls, MIDDLE: erase. Esc/close to quit.");
    match run(artifact, MAIN) {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("sand-demo failed: {e:?}"),
    }
}
