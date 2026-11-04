use crate::run;

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::FALLING_SAND;


pub fn main(_args: Vec<String>) {
    println!("Falling sand — hold LEFT: sand, RIGHT: walls, MIDDLE: erase. Esc/close to quit.");
    // SAFETY: the artifact is the one Lean built for this application.
    match unsafe { run(ARTIFACT_BINARY, "main", b"") } {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("sand-demo failed: {e:?}"),
    }
}
