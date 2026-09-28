use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    lean_artifacts::FFT_APP;


/// Payload offset where the input filename is stored (must match MakeAlgorithm.lean).
const INPUT_FILENAME_OFF: u64 = 0x2200;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: fft <input.bin>");
        std::process::exit(1);
    }
    let input_path = &args[1];

    let mut artifact = Artifact::from_bytes(ARTIFACT_BINARY).expect("the build checked this artifact");

    // Write the input filename into the memory the program starts from
    let path_bytes = input_path.as_bytes();
    assert!(
        path_bytes.len() < 255,
        "Input path too long (max 254 chars)"
    );
    artifact.write(INPUT_FILENAME_OFF, &[path_bytes, &[0]].concat());

    let start = std::time::Instant::now();
    match run(artifact, "main") {
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
