use crate::run;

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::SAT_APP;

pub fn main(args: Vec<String>) {
    if args.len() < 2 {
        eprintln!("Usage: sat <input.cnf>");
        std::process::exit(1);
    }
    let input_path = &args[1];

    let start = std::time::Instant::now();
    // SAFETY: the artifact is the one Lean built for this application; its input
    // names the file it reads.
    match unsafe { run(ARTIFACT_BINARY, "main", input_path.as_bytes()) } {
        Ok(_) => {
            let elapsed = start.elapsed();
            // Read the output file produced by the solver
            match std::fs::read_to_string("sat_output.txt") {
                Ok(result) => {
                    print!("{}", result);
                    eprintln!("Solved in {:.1}ms", elapsed.as_secs_f64() * 1000.0);
                }
                Err(e) => eprintln!("Failed to read output: {}", e),
            }
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
