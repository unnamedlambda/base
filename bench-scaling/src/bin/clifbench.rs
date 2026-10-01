//! Decode and JIT timing for generated CLIF programs.
//!
//! Separate binary because it links `base` (and therefore Cranelift), which
//! the harness itself has no reason to pull in. The `clif` suite invokes this if
//! it has been built; otherwise it records the generation numbers and skips JIT.
//!
//!   cargo build --release -p bench-scaling --bin clifbench --features jit
//!
//! Takes generated `.cbor` artifact files. Prints one
//! `file<TAB>bytes<TAB>decode_seconds<TAB>jit_seconds` line per input.
//!
//! The two halves are timed apart because they cover different things. Decode
//! reads the whole artifact, data segments included; the JIT below is handed
//! only the functions, against a fixed arena. For an artifact that is mostly
//! weights the first number is the one that grows.

use base::{Artifact, Driver};


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
            functions: artifact.functions,
            required_memory: 1 << 20,
            data: Vec::new(),
        };
        let name = path.rsplit('/').next().unwrap_or(&path).to_string();
        let t = std::time::Instant::now();
        match Driver::load(artifact) {
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
