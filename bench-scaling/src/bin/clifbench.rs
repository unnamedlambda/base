//! JIT timing for generated CLIF programs.
//!
//! Separate binary because it links `base` (and therefore wgpu/cudarc), which
//! the harness itself has no reason to pull in. The `clif` suite invokes this if
//! it has been built; otherwise it records the generation numbers and skips JIT.
//!
//!   cargo build --release -p bench-scaling --bin clifbench
//!
//! Takes generated `.cbor` artifact files. Prints one
//! `file<TAB>bytes<TAB>seconds` line per input.

use base::{Artifact, Base};


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
        let artifact = match Artifact::from_bytes(&file) {
            Ok(a) => a,
            Err(e) => {
                println!("{path}\t{bytes}\tERR {e}");
                continue;
            }
        };
        let artifact = Artifact {
            functions: artifact.functions,
            required_memory: 1 << 20,
            data: Vec::new(),
        };
        let name = path.rsplit('/').next().unwrap_or(&path).to_string();
        let t = std::time::Instant::now();
        match Base::new(artifact) {
            Ok(b) => {
                let el = t.elapsed();
                println!("{}\t{}\t{:.3}", name, bytes, el.as_secs_f64());
                // keep the module alive until after the timing read
                std::hint::black_box(&b);
            }
            Err(e) => println!("{}\t{}\tERR {:?}", name, bytes, e),
        }
    }
}
