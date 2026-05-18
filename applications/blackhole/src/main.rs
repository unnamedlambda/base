use base::{run, Algorithm, BaseConfig};

const ARTIFACT_BINARY: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/BlackHoleAlgorithm/blackhole_app.bin"));

fn main() {
    let (config, alg): (BaseConfig, Algorithm) = bincode::deserialize(ARTIFACT_BINARY)
        .expect("Failed to deserialize (BaseConfig, Algorithm) binary");

    let start = std::time::Instant::now();
    match run(config, alg) {
        Ok(_) => {
            let elapsed = start.elapsed();
            eprintln!(
                "Black hole render completed in {:.1}ms",
                elapsed.as_secs_f64() * 1000.0
            );
            eprintln!("Output: blackhole.bmp");
        }
        Err(e) => eprintln!("Execution failed: {:?}", e),
    }
}
