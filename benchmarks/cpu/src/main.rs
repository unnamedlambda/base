//! Workloads through Base (`cpu_bench`: CLIF, and LLVM's code carried as
//! `asm`) against the same kernels in Rust. Prints `<workload> <column> <ns>`
//! lines for `report.py`. x86-64 only.

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
