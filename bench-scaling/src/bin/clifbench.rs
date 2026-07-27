//! Cranelift parse + JIT timing for generated CLIF.
//!
//! Separate binary because it links `base` (and therefore wgpu/cudarc), which
//! the harness itself has no reason to pull in. The `clif` suite invokes this if
//! it has been built; otherwise it records the generation numbers and skips JIT.
//!
//!   cargo build --release -p bench-scaling --bin clifbench
//!
//! Prints one `file<TAB>bytes<TAB>seconds` line per input.

use base::{Base, Setup};
use base_types::IoOffsets;

fn main() {
    for path in std::env::args().skip(1) {
        let ir = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                println!("{path}\t0\tERR {e}");
                continue;
            }
        };
        let bytes = ir.len();
        let setup = Setup {
            cranelift_ir: ir,
            memory_size: 1 << 20,
            io_offsets: IoOffsets { data_ptr: 8, data_len: 16, out_ptr: 24, out_len: 32 },
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
