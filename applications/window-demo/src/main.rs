use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("WindowDemoAlgorithm/window_demo");


fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    println!("Opening window — arrow keys to move, Esc or close to quit.");
    match run(artifact, "main") {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("window-demo failed: {e:?}"),
    }
}
