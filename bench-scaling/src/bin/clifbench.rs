//! Decode and JIT time per artifact, one `file\tbytes\tdecode_s\tjit_s` line each:
//! `cargo build --release -p bench-scaling --bin clifbench --features jit`

use base::Driver;
use base_types::Artifact;


fn main() {
    for path in std::env::args().skip(1) {
        let file = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                println!("{path}\t0\tERR {e}");
                continue;
            }
        };
        let bytes = file.len();
        let t = std::time::Instant::now();
        let artifact = match Artifact::from_bytes(&file) {
            Ok(a) => a,
            Err(e) => {
                println!("{path}\t{bytes}\tERR {e}");
                continue;
            }
        };
        let decode = t.elapsed();
        // The arena and the data segments are replaced rather than carried, so
        // what the JIT is timed on is the functions alone.
        let artifact = Artifact {
            primitives: artifact.primitives,
            functions: artifact.functions,
            arena_size: 1 << 20,
            data: Vec::new(),
        };
        let name = path.rsplit('/').next().unwrap_or(&path).to_string();
        let t = std::time::Instant::now();
        match Driver::load(&artifact.to_bytes()) {
            Ok(b) => {
                let el = t.elapsed();
                println!(
                    "{}\t{}\t{:.3}\t{:.3}",
                    name,
                    bytes,
                    decode.as_secs_f64(),
                    el.as_secs_f64()
                );
                // keep the module alive until after the timing read
                std::hint::black_box(&b);
            }
            Err(e) => println!(
                "{}\t{}\t{:.3}\tERR {:?}",
                name,
                bytes,
                decode.as_secs_f64(),
                e
            ),
        }
    }
}
