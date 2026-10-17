use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::RAYMARCH_DEMO;


pub fn main(_args: Vec<String>) {
    let artifact = Artifact::from_bytes(ARTIFACT_BINARY).expect("the build checked this artifact");
    println!("Raymarched 3D — arrows move/strafe, W/S rise/fall, Esc or close to quit.");
    // SAFETY: the artifact is the one Lean built for this application.
    match unsafe { run(artifact, "main") } {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("raymarch-demo failed: {e:?}"),
    }
}
