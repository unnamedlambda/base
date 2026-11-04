//! Every example application's tests, as one binary.

use std::process::Command;

/// The program `apps` running the application `name`.
pub fn app(name: &str) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_apps"));
    c.arg(name);
    c
}

#[path = "blackhole/tests/smoke.rs"]
mod blackhole_smoke;
#[path = "bytecount/tests/count.rs"]
mod bytecount_count;
#[path = "par-sum/tests/sum.rs"]
mod par_sum_sum;
#[path = "self-describing/tests/formats.rs"]
mod self_describing_formats;
#[path = "bytescrub/tests/scrub.rs"]
mod bytescrub_scrub;
#[path = "shared-sum/tests/threads.rs"]
mod shared_sum_threads;
#[path = "shared-sum/tests/rayon.rs"]
mod shared_sum_rayon;
#[path = "cli/tests/comparison.rs"]
mod cli_comparison;
#[path = "compress/tests/roundtrip.rs"]
mod compress_roundtrip;
#[path = "draw/tests/smoke.rs"]
mod draw_smoke;
#[path = "fft/tests/comparison.rs"]
mod fft_comparison;
#[path = "lean4-evaluator/tests/comparison.rs"]
mod lean4_evaluator_comparison;
#[path = "lz4-ptx/tests/compress_roundtrip.rs"]
mod lz4_ptx_compress_roundtrip;
#[path = "matmul/tests/smoke.rs"]
mod matmul_smoke;
#[path = "raymarch-demo/tests/logic.rs"]
mod raymarch_demo;
#[path = "raymarch-demo/tests/smoke.rs"]
mod raymarch_demo_smoke;
#[path = "raytrace/tests/smoke.rs"]
mod raytrace_smoke;
#[path = "sand-demo/tests/logic.rs"]
mod sand_demo;
#[path = "sand-demo/tests/smoke.rs"]
mod sand_demo_smoke;
#[path = "sat/tests/comparison.rs"]
mod sat_comparison;
#[path = "scene/tests/smoke.rs"]
mod scene_smoke;
#[path = "sha256/tests/comparison.rs"]
mod sha256_comparison;
#[path = "window-demo/tests/logic.rs"]
mod window_demo;
#[path = "window-demo/tests/smoke.rs"]
mod window_demo_smoke;
