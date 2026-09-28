use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::WINDOW_DEMO;


fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY).expect("the build checked this artifact");
    println!("Opening window — arrow keys to move, Esc or close to quit.");
    match run(artifact, "main") {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("window-demo failed: {e:?}"),
    }
}
