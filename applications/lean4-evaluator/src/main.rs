use crate::run;

const ARTIFACT_BINARY: &[u8] = lean_artifacts::LEAN_EVAL_APP;

pub fn main(args: Vec<String>) {
    if args.len() < 2 {
        eprintln!("Usage: lean4-eval <source.lean> [output_file]");
        std::process::exit(1);
    }

    // The source path, then, after a zero, the output path if there is one.
    let mut input = args[1].as_bytes().to_vec();
    if let Some(output_path) = args.get(2) {
        input.push(0);
        input.extend_from_slice(output_path.as_bytes());
    }

    // SAFETY: the artifact is the one Lean built for this application; its input
    // names the files it reads and writes.
    match unsafe { run(ARTIFACT_BINARY, "main", &input) } {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
