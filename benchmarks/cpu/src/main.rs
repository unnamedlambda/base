//! Common workloads through Base against the same kernels in Rust.
//!
//! The Rust side (`kernels.rs`) is built with `-C target-cpu=native`. The Base
//! side is the `cpu_bench` artifact run through `Driver::execute`: `<w>_clif`,
//! and `<w>_asm` calling LLVM's code carried in the artifact. Every answer is
//! checked against Rust's before anything is timed.
//!
//! Output, read by `report.py`, is one line per timing:
//!
//!     <workload> <column> <ns>
//!
//! with column `rust`, `clif` or `asm`; `execute noop` is the fixed
//! cost of a call into an artifact. x86-64 only.

#[cfg(target_arch = "x86_64")]
mod bench;
#[cfg(target_arch = "x86_64")]
mod kernels;

#[cfg(target_arch = "x86_64")]
fn main() {
    bench::main();
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("cpu-bench compares Base with LLVM's x86-64 code; this host is not x86-64");
    std::process::exit(1);
}
