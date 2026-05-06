use base::{run, Algorithm, BaseConfig};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/CliAlgorithm/cli_app.bin"));

fn main() {
    let (config, alg): (BaseConfig, Algorithm) = bincode::deserialize(ARTIFACT_BINARY)
        .expect("Failed to deserialize (BaseConfig, Algorithm) binary");

    match run(config, alg) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Execution failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
