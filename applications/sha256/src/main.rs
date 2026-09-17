use base::{run, Artifact};

const ARTIFACT_BINARY: &[u8] =
    build_support::artifact!("Sha256Algorithm/sha256_app");


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

    // Write the input filename into the memory the program starts from
    let path_bytes = input_path.as_bytes();
    assert!(
        path_bytes.len() < 255,
        "Input path too long (max 254 chars)"
    );
    artifact.write(INPUT_FILENAME_OFF, &[path_bytes, &[0]].concat());

    match run(artifact, "main") {
        Ok(_) => match std::fs::read_to_string("sha256_output.txt") {
            Ok(result) => print!("{}", result),
            Err(e) => eprintln!("Failed to read output: {}", e),
        },
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
