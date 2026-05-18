use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/Sha256Algorithm/sha256_app.bin"));

/// Payload offset where the input filename is stored (must match MakeAlgorithm.lean).
const INPUT_FILENAME_OFF: usize = 0x100;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: sha256 <input_file>");
        std::process::exit(1);
    }
    let input_path = &args[1];

    let mut artifact = Artifact::from_bytes(ARTIFACT_BINARY);

    // Write input filename into initial_memory (null-terminated)
    let path_bytes = input_path.as_bytes();
    assert!(
        path_bytes.len() < 255,
        "Input path too long (max 254 chars)"
    );
    artifact.setup.initial_memory[INPUT_FILENAME_OFF..INPUT_FILENAME_OFF + path_bytes.len()]
        .copy_from_slice(path_bytes);
    artifact.setup.initial_memory[INPUT_FILENAME_OFF + path_bytes.len()] = 0;

    match run(artifact.setup, artifact.main) {
        Ok(_) => match std::fs::read_to_string("sha256_output.txt") {
            Ok(result) => print!("{}", result),
            Err(e) => eprintln!("Failed to read output: {}", e),
        },
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
