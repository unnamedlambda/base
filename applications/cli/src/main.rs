use crate::run;

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::CLI_APP;


pub fn main(_args: Vec<String>) {

    // SAFETY: the artifact is the one Lean built for this application.
    match unsafe { run(ARTIFACT_BINARY, "main", b"") } {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
