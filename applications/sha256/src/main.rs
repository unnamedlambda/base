use crate::run;

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::SHA256_APP;

pub fn main(args: Vec<String>) {
    if args.len() < 2 {
        eprintln!("Usage: sha256 <input_file>");
        std::process::exit(1);
    }
    let input_path = &args[1];

    // SAFETY: the artifact is the one Lean built for this application; its input
    // names the file it reads.
    match unsafe { run(ARTIFACT_BINARY, "main", input_path.as_bytes()) } {
        Ok(_) => match std::fs::read_to_string("sha256_output.txt") {
            Ok(result) => print!("{}", result),
            Err(e) => eprintln!("Failed to read output: {}", e),
        },
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
