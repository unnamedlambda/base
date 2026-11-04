use crate::run;

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::WINDOW_DEMO;


pub fn main(_args: Vec<String>) {
    println!("Opening window — arrow keys to move, Esc or close to quit.");
    // SAFETY: the artifact is the one Lean built for this application.
    match unsafe { run(ARTIFACT_BINARY, "main", b"") } {
        Ok(_) => println!("Window closed."),
        Err(e) => eprintln!("window-demo failed: {e:?}"),
    }
}
