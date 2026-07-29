//! A **multi-layer** model, written once as an `Expr`, shared with `bindVec`,
//! compiled by `compileW`, emitted by the proven lowering, run on the GPU.
//! No kernel was hand-written and no layer was unrolled by hand.

use base::{Artifact, Base};

const ART: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/MlpWarpAlgorithm/mlp_warp.bin"));
const D: usize = 4;
const L: usize = 3;
const GRID: usize = 16384;
const LANES: usize = GRID * 32;
const NIN: usize = LANES * D;

/// The same model the spec describes, evaluated in f32 — the machine carrier.
fn layer(x: &[f32; D]) -> [f32; D] {
    // rmsNorm w=1: hᵢ = xᵢ · rsqrt((Σⱼxⱼ²)/D + 1e-6)
    let ss: f32 = x.iter().map(|v| v * v).fold(0.0, |a, b| a + b);
    let scale = 1.0f32 / (ss * (1.0 / D as f32) + 1.0 / 1_000_000.0).sqrt();
    let h: [f32; D] = std::array::from_fn(|i| x[i] * 1.0 * scale);
    // matvec W=1: every row is Σⱼ hⱼ
    let m: f32 = h.iter().fold(0.0, |a, b| a + b);
    // silu, then residual
    let s = m * (1.0 / (1.0 + (-m).exp()));
    std::array::from_fn(|i| x[i] + s)
}

fn model(x0: &[f32; D]) -> f32 {
    let mut x = *x0;
    for _ in 0..L {
        x = layer(&x);
    }
    x.iter().fold(0.0, |a, b| a + b)
}

fn main() {
    let input: Vec<f32> = (0..NIN)
        .map(|i| ((i % 977) as f32 - 488.0) * 0.003)
        .collect();
    let bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

    let artifact = Artifact::from_bytes(ART);
    let mut base = Base::new(artifact.setup).expect("Base::new");
    let run = &artifact.extras["run"];
    let fetch = &artifact.extras["fetch"];

    let mut out = vec![0u8; LANES * 4];
    base.execute_into(&artifact.main, &bytes, &mut []).expect("load");
    base.execute_into(run, b"", &mut []).expect("run");
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        base.execute_into(run, b"", &mut []).expect("run");
    }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    base.execute_into(fetch, b"", &mut out).expect("fetch");

    let gpu: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut worst = 0f32;
    let mut worst_at = 0usize;
    for lane in 0..LANES {
        let x: [f32; D] = std::array::from_fn(|i| input[lane * D + i]);
        let want = model(&x);
        let d = (gpu[lane] - want).abs() / want.abs().max(1e-4);
        if d > worst {
            worst = d;
            worst_at = lane;
        }
    }

    println!("model        : {L} layers x width {D}  (rmsnorm -> matvec -> silu -> residual)");
    println!("lanes        : {LANES}   (each evaluates the whole model)");
    println!("kernel time  : {:.3} ms", dt * 1e3);
    println!("gpu[0..3]    : {:?}", &gpu[..3]);
    println!(
        "cpu[0..3]    : {:?}",
        (0..3)
            .map(|l| model(&std::array::from_fn(|i| input[l * D + i])))
            .collect::<Vec<_>>()
    );
    println!("max rel err  : {worst:.2e}  (lane {worst_at}; ex2.approx is the one declared approx)");
    assert!(worst < 1e-3, "multi-layer mismatch: {worst}");
    println!("PASS         : multi-layer model matches its spec");
}
