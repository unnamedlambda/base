//! JIT timing for generated CLIF programs.
//!
//! Separate binary because it links `base` (and therefore wgpu/cudarc), which
//! the harness itself has no reason to pull in. The `clif` suite invokes this if
//! it has been built; otherwise it records the generation numbers and skips JIT.
//!
//!   cargo build --release -p bench-scaling --bin clifbench
//!
//! Takes generated artifact JSON files. Prints one
//! `file<TAB>bytes<TAB>seconds` line per input.

use base::{Base, Setup};
use base_types::Artifact;

fn main() {
    for path in std::env::args().skip(1) {
        let text = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                println!("{path}\t0\tERR {e}");
                continue;
            }
        };
        let bytes = text.len();
        let artifact: Artifact = match serde_json::from_str(&text) {
            Ok(a) => a,
            Err(e) => {
                println!("{path}\t{bytes}\tERR {e}");
                continue;
            }
        };
        let setup = Setup {
            clif: artifact.setup.clif,
            memory_size: 1 << 20,
            initial_memory: Vec::new(),
        };
        let name = path.rsplit('/').next().unwrap_or(&path).to_string();
        let t = std::time::Instant::now();
        match Base::new(setup) {
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
