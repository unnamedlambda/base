use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/FallingSandAlgorithm/falling_sand.bin"));

fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    println!("Falling sand — hold LEFT: sand, RIGHT: walls, MIDDLE: erase. Esc/close to quit.");
    match run(artifact.setup, artifact.main) {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("sand-demo failed: {e:?}"),
    }
}
