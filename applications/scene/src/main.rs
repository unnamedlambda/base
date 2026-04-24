use base::{run, Algorithm, BaseConfig};

const ALGORITHM_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/algorithm.bin"));

fn main() {
    let (config, alg): (BaseConfig, Algorithm) = bincode::deserialize(ALGORITHM_BINARY)
        .expect("Failed to deserialize (BaseConfig, Algorithm) binary");

    let start = std::time::Instant::now();
    match run(config, alg) {
        Ok(_) => {
            let elapsed = start.elapsed();
            eprintln!(
                "Scene render completed in {:.1}ms",
                elapsed.as_secs_f64() * 1000.0
            );
            eprintln!("Output: scene.bmp");
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
