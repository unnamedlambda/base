use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/WindowDemoAlgorithm/window_demo.bin"));

fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    println!("Opening window — arrow keys to move, Esc or close to quit.");
    match run(artifact.setup, artifact.main) {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("window-demo failed: {e:?}"),
    }
}
