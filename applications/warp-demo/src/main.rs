//! Runs the PROVEN warp reduction kernel on the GPU.
//!
//! The kernel body is `AlgorithmLib.ML.warpSumSqV4Store`, proven to compute
//! `denote env spec` exactly. This binary checks the GPU agrees with the
//! reference the proof predicts.

use base::{Artifact, Base};

const ART: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/WarpSumSqAlgorithm/warp_sumsq.bin"));

const K: usize = 128;
const CHUNK: usize = K * 4 * 32;      // 16384 floats per block
const GRID: usize = 4096;             // blocks, one warp each
const N: usize = GRID * CHUNK;        // 67,108,864 floats = 256 MiB

fn main() {
    let input: Vec<f32> = (0..N).map(|i| ((i % 1024) as f32) * 0.001).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

    let artifact = Artifact::from_bytes(ART);
    let mut base = Base::new(artifact.setup).expect("Base::new");

    let mut out = vec![0u8; GRID * 4];
    let run = &artifact.extras["run"];
    let fetch = &artifact.extras["fetch"];

    // upload once
    base.execute_into(&artifact.main, &bytes, &mut []).expect("load");
    // warm up, then time the LAUNCH ONLY
    base.execute_into(run, b"", &mut []).expect("run");
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        base.execute_into(run, b"", &mut []).expect("run");
    }
    let elapsed = t0.elapsed().as_secs_f64() / reps as f64;
    base.execute_into(fetch, b"", &mut out).expect("fetch");

    let bytes_read = (N * 4) as f64;
    println!("working set : {:.0} MiB", bytes_read / 1048576.0);
    println!("kernel time : {:.3} ms", elapsed * 1e3);
    println!("read BW     : {:.1} GB/s", bytes_read / elapsed / 1e9);
    println!("3060 peak   : 360.0 GB/s  ->  {:.0}% of roofline",
             100.0 * (bytes_read / elapsed / 1e9) / 360.0);
    let gpu = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);

    // reference for block 0: the two-level fold the proof commits to
    let mut lanes = [0f32; 32];
    for (l, lane) in lanes.iter_mut().enumerate() {
        for i in 0..K {
            let b = i * 128 + l * 4;
            for d in 0..4 {
                *lane += input[b + d] * input[b + d];
            }
        }
    }
    for mask in [16usize, 8, 4, 2, 1] {
        let prev = lanes;
        for l in 0..32 {
            lanes[l] = prev[l] + prev[l ^ mask];
        }
    }
    let expected = lanes[0];
    let naive: f32 = input[..CHUNK].iter().map(|x| x * x).sum();

    println!("block0 gpu  = {gpu:.6}");
    println!("block0 spec = {expected:.6}   (two-level fold from the proof)");
    println!("block0 naive= {naive:.6}");
    println!("gpu == spec : {}", gpu == expected);
    assert_eq!(gpu, expected, "GPU disagreed with the proven spec");
}
