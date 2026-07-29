//! A user's spec — `Transformer.silu (.var 0)` — compiled by `compileW`,
//! proven by `compileWKernel_correct`, emitted to PTX, run on the GPU.
//! No kernel was hand-written.

use base::{Artifact, Base};

const ART: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/SiluWarpAlgorithm/silu_warp.bin"));
const GRID: usize = 2097152;
const N: usize = GRID * 32;

fn main() {
    let input: Vec<f32> = (0..N).map(|i| ((i % 2048) as f32 - 1024.0) * 0.01).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

    let artifact = Artifact::from_bytes(ART);
    let mut base = Base::new(artifact.setup).expect("Base::new");
    let run = &artifact.extras["run"];
    let fetch = &artifact.extras["fetch"];

    let mut out = vec![0u8; N * 4];
    base.execute_into(&artifact.main, &bytes, &mut []).expect("load");
    base.execute_into(run, b"", &mut []).expect("run");
    let reps = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..reps { base.execute_into(run, b"", &mut []).expect("run"); }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    base.execute_into(fetch, b"", &mut out).expect("fetch");

    let gpu: Vec<f32> = out.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // silu(x) = x / (1 + e^-x); `exp` is the one declared approximation
    // (ex2.approx, ~2 ULP), so compare within tolerance rather than bit-exact.
    let mut worst = 0f32;
    for i in 0..N {
        let x = input[i];
        let want = x / (1.0 + (-x).exp());
        let d = (gpu[i] - want).abs() / want.abs().max(1e-6);
        if d > worst { worst = d; }
    }
    let moved = (N * 4 * 2) as f64;   // read + write
    println!("elements    : {N}");
    println!("kernel time : {:.3} ms", dt * 1e3);
    // The same spec, 8 elements per lane: 8x fewer blocks.
    let run_loop = &artifact.extras["runLoop"];
    base.execute_into(run_loop, b"", &mut []).expect("runLoop");
    let t1 = std::time::Instant::now();
    for _ in 0..reps { base.execute_into(run_loop, b"", &mut []).expect("runLoop"); }
    let dt_loop = t1.elapsed().as_secs_f64() / reps as f64;
    // The looped kernel wrote the same buffer — check *its* output too.
    let mut out2 = vec![0u8; N * 4];
    base.execute_into(fetch, b"", &mut out2).expect("fetch");
    let gpu2: Vec<f32> = out2.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut same = 0usize;
    for i in 0..N { if gpu2[i].to_bits() == gpu[i].to_bits() { same += 1; } }
    println!("looped (E=32): {:.3} ms   {:.1} GB/s  ({:.0}% of 360)",
        dt_loop * 1e3, moved / dt_loop / 1e9, moved / dt_loop / 1e9 / 3.6);
    println!("bandwidth   : {:.1} GB/s  ({:.0}% of 360)", moved / dt / 1e9,
             100.0 * (moved / dt / 1e9) / 360.0);
    println!("max rel err : {worst:.2e}   (ex2.approx, declared ~2 ULP)");
    println!("looped == single-element, bit-for-bit : {same}/{N}");
    assert_eq!(same, N, "the looped kernel computes the same spec");
    println!("gpu[0..4]   : {:?}", &gpu[..4]);
    assert!(worst < 1e-5, "silu mismatch: {worst}");
}
