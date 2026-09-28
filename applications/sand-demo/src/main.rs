use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::FALLING_SAND;


fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY).expect("the build checked this artifact");
    println!("Falling sand — hold LEFT: sand, RIGHT: walls, MIDDLE: erase. Esc/close to quit.");
    match run(artifact, "main") {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("sand-demo failed: {e:?}"),
    }
}
