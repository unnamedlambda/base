use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("RaymarchDemoAlgorithm/raymarch_demo");

/// Entry points of this artifact, as its generator numbers them.
const MAIN: u32 = 1;

fn main() {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY);
    println!("Raymarched 3D — arrows move/strafe, W/S rise/fall, Esc or close to quit.");
    match run(artifact, MAIN) {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("raymarch-demo failed: {e:?}"),
    }
}
