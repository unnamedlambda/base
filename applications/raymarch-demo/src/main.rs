use crate::run;

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::RAYMARCH_DEMO;


pub fn main(_args: Vec<String>) {
    println!("Raymarched 3D — arrows move/strafe, W/S rise/fall, Esc or close to quit.");
    // SAFETY: the artifact is the one Lean built for this application.
    match unsafe { run(ARTIFACT_BINARY, "main", b"") } {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("raymarch-demo failed: {e:?}"),
    }
}
