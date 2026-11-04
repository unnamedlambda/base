use crate::run;

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::FFT_APP;

pub fn main(args: Vec<String>) {
    if args.len() < 2 {
        eprintln!("Usage: fft <input.bin>");
        std::process::exit(1);
    }
    let input_path = &args[1];

    let start = std::time::Instant::now();
    // SAFETY: the artifact is the one Lean built for this application; its input
    // names the file it reads.
    match unsafe { run(ARTIFACT_BINARY, "main", input_path.as_bytes()) } {
        Ok(_) => {
            let elapsed = start.elapsed();
            let input_size = std::fs::metadata(input_path).map(|m| m.len()).unwrap_or(0);
            let n = input_size / 8;
            eprintln!(
                "FFT of {} complex numbers in {:.1}ms",
                n,
                elapsed.as_secs_f64() * 1000.0
            );
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
