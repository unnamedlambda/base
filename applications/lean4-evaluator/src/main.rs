use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] = build_support::artifact!("LeanEvalAlgorithm/lean_eval_app");

/// Entry points of this artifact, as its generator numbers them.
const MAIN: u32 = 1;
const INPUT_PATH_OFFSET: usize = 0x0078;
const INPUT_PATH_MAX_LEN: usize = 256;
const OUTPUT_PATH_OFFSET: usize = 0x0038;
const OUTPUT_PATH_MAX_LEN: usize = 64;

fn main() {
    let mut artifact = Artifact::from_bytes(ARTIFACT_BINARY);

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: lean4-eval <source.lean> [output_file]");
        std::process::exit(1);
    }

    // Patch input file path
    let input_path = &args[1];
    let input_len = input_path.len().min(INPUT_PATH_MAX_LEN - 1);
    let mut path = vec![0u8; INPUT_PATH_MAX_LEN];
    path[..input_len].copy_from_slice(&input_path.as_bytes()[..input_len]);
    artifact.write(INPUT_PATH_OFFSET, &path);

    // Patch output file path if provided
    if args.len() > 2 {
        let output_path = &args[2];
        let output_len = output_path.len().min(OUTPUT_PATH_MAX_LEN - 1);
        let mut path = vec![0u8; OUTPUT_PATH_MAX_LEN];
        path[..output_len].copy_from_slice(&output_path.as_bytes()[..output_len]);
        artifact.write(OUTPUT_PATH_OFFSET, &path);
    }

    match run(artifact, MAIN) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
