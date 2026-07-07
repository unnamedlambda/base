use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/RaymarchDemoAlgorithm/raymarch_demo.bin"));

fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    println!("Raymarched 3D — arrows move/strafe, W/S rise/fall, Esc or close to quit.");
    match run(artifact.setup, artifact.main) {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("raymarch-demo failed: {e:?}"),
    }
}
