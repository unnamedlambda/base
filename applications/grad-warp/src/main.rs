//! A **gradient**, running on the GPU.
//!
//! The forward function is written once as an `Expr` with two bound
//! intermediates; `gradProg` turns it into a program whose Γ outputs share one
//! telescope, `compileVWKernel` emits that telescope *once*, and the proven
//! lowering prints the PTX. No backward kernel was hand-written.
//!
//!     v0  = x0 * x1
//!     v1  = v0 * exp x2
//!     out = v1 + x3 * v0
//!
//! `v0` is used twice — the case a naive `grad` duplicates and `gradProg` binds.

use base::{Artifact, Base};

const ART: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/GradWarpAlgorithm/grad_warp.bin"));
const G: usize = 4;
const GRID: usize = 16384;
const LANES: usize = GRID * 32;
const NIN: usize = LANES * G;

/// The analytic gradient of the same function, in f32.
fn grad(x: &[f32; G]) -> [f32; G] {
    let v0 = x[0] * x[1];
    let e = x[2].exp();
    [
        e * x[1] + x[3] * x[1], // d/dx0
        e * x[0] + x[3] * x[0], // d/dx1
        v0 * e,                 // d/dx2
        v0,                     // d/dx3
    ]
}

fn main() {
    let input: Vec<f32> = (0..NIN).map(|i| ((i % 617) as f32 - 308.0) * 0.004).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

    let artifact = Artifact::from_bytes(ART);
    let mut base = Base::new(artifact.setup).expect("Base::new");
    let run = &artifact.extras["run"];
    let run_d = &artifact.extras["runD"];
    let fetch = &artifact.extras["fetch"];

    let mut out = vec![0u8; NIN * 4];
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
    let mut worst_at = (0usize, 0usize);
    for lane in 0..LANES {
        let x: [f32; G] = std::array::from_fn(|i| input[lane * G + i]);
        let want = grad(&x);
        for k in 0..G {
            let got = gpu[lane * G + k];
            let d = (got - want[k]).abs() / want[k].abs().max(1e-4);
            if d > worst {
                worst = d;
                worst_at = (lane, k);
            }
        }
    }

    let x0: [f32; G] = std::array::from_fn(|i| input[i]);
    println!("function     : v0=x0*x1; v1=v0*exp x2; out=v1+x3*v0   (v0 shared)");
    println!("lanes        : {LANES}   (each differentiates its own {G} inputs)");
    println!("kernel time  : {:.3} ms", dt * 1e3);
    println!("gpu grad[0]  : {:?}", &gpu[..G]);
    println!("cpu grad[0]  : {:?}", grad(&x0));
    println!(
        "max rel err  : {worst:.2e}  (lane {}, k={}; ex2.approx is the one declared approx)",
        worst_at.0, worst_at.1
    );
    assert!(worst < 1e-3, "gradient mismatch: {worst}");
    println!("PASS         : the emitted gradient matches the analytic one");

    // ---- the narrowed (linear-in-depth) gradient, same harness ------------
    //
    // `gradProgD` drops the adjoint terms that are semantically zero.  That is
    // not a Float32 identity, so it carries two declared propositions
    // (`ZeroTermFree`, `ZeroLaws`) rather than being applied silently — see
    // `gradD_outputs_are_derivatives`.  This is the check that it *runs*.
    let mut out_d = vec![0u8; NIN * 4];
    base.execute_into(run_d, b"", &mut []).expect("runD");
    let t1 = std::time::Instant::now();
    for _ in 0..reps {
        base.execute_into(run_d, b"", &mut []).expect("runD");
    }
    let dt_d = t1.elapsed().as_secs_f64() / reps as f64;
    base.execute_into(fetch, b"", &mut out_d).expect("fetch");

    let gpu_d: Vec<f32> = out_d
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut worst_d = 0f32;
    let mut worst_vs_full = 0f32;
    for lane in 0..LANES {
        let x: [f32; G] = std::array::from_fn(|i| input[lane * G + i]);
        let want = grad(&x);
        for k in 0..G {
            let got = gpu_d[lane * G + k];
            let d = (got - want[k]).abs() / want[k].abs().max(1e-4);
            worst_d = worst_d.max(d);
            let full = gpu[lane * G + k];
            let dv = (got - full).abs() / full.abs().max(1e-4);
            worst_vs_full = worst_vs_full.max(dv);
        }
    }

    println!();
    println!("narrowed (gradProgD, linear in depth):");
    println!("kernel time  : {:.3} ms   (full: {:.3} ms)", dt_d * 1e3, dt * 1e3);
    println!("gpu grad[0]  : {:?}", &gpu_d[..G]);
    println!("max rel err  : {worst_d:.2e}  vs analytic");
    println!("vs full grad : {worst_vs_full:.2e}  (ZeroTermFree/ZeroLaws are the only difference)");
    assert!(worst_d < 1e-3, "narrowed gradient mismatch: {worst_d}");
    println!("PASS         : the narrowed gradient runs and agrees");
}
