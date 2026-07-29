//! Proven warp GEMV vs cuBLAS sgemv, on Qwen2's FFN shape (4864 x 896).
//! This is the decode-shaped case (M=1) where cuBLAS is beatable.

use base::{Artifact, Base};
const ART: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/GemvWarpAlgorithm/gemv_warp.bin"));
const N: usize = 896;
const M: usize = 4864;

fn time(base: &mut Base, alg: &base_types::Algorithm, reps: usize) -> f64 {
    base.execute_into(alg, b"", &mut []).expect("warm");
    let t = std::time::Instant::now();
    for _ in 0..reps { base.execute_into(alg, b"", &mut []).expect("run"); }
    t.elapsed().as_secs_f64() / reps as f64
}

fn main() {
    let a: Vec<f32> = (0..M * N).map(|i| ((i % 251) as f32 - 125.0) * 0.001).collect();
    let x: Vec<f32> = (0..N).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let mut bytes: Vec<u8> = a.iter().flat_map(|f| f.to_le_bytes()).collect();
    bytes.extend(x.iter().flat_map(|f| f.to_le_bytes()));

    let artifact = Artifact::from_bytes(ART);
    let mut base = Base::new(artifact.setup).expect("Base::new");
    let run = artifact.extras["run"].clone();
    let blas = artifact.extras["blas"].clone();
    let fetch = artifact.extras["fetch"].clone();

    base.execute_into(&artifact.main, &bytes, &mut []).expect("load");

    let t_ours = time(&mut base, &run, 200);
    let mut ours = vec![0u8; M * 4];
    base.execute_into(&fetch, b"", &mut ours).expect("fetch");
    let ours: Vec<f32> = ours.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();

    let t_blas = time(&mut base, &blas, 200);
    let mut bl = vec![0u8; M * 4];
    base.execute_into(&fetch, b"", &mut bl).expect("fetch");
    let bl: Vec<f32> = bl.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();

    let bytes_moved = ((M * N + N + M) * 4) as f64;
    let mut worst = 0f32;
    for i in 0..M {
        let d = (ours[i] - bl[i]).abs() / bl[i].abs().max(1e-3);
        if d > worst { worst = d; }
    }
    println!("shape        : {M} x {N}  (Qwen2 FFN gate/up)");
    println!("ours   : {:.4} ms   {:6.1} GB/s   {:>3.0}% roofline",
             t_ours*1e3, bytes_moved/t_ours/1e9, 100.0*(bytes_moved/t_ours/1e9)/360.0);
    println!("cuBLAS : {:.4} ms   {:6.1} GB/s   {:>3.0}% roofline",
             t_blas*1e3, bytes_moved/t_blas/1e9, 100.0*(bytes_moved/t_blas/1e9)/360.0);
    println!("speedup      : {:.2}x", t_blas / t_ours);
    println!("max rel diff : {worst:.2e}  (different reduction orders; both exact for their spec)");

    let reference = |row: usize| -> f32 {
        let mut lane = [0f32; 32];
        for (l, acc) in lane.iter_mut().enumerate() {
            for t in 0..N / 128 {
                let base_i = t * 128 + l * 4;
                for q in 0..4 {
                    *acc += a[row * N + base_i + q] * x[base_i + q];
                }
            }
        }
        for m in [16usize, 8, 4, 2, 1] {
            let prev = lane;
            for l in 0..32 {
                lane[l] = prev[l] + prev[l ^ m];
            }
        }
        lane[0]
    };

    let mut exact = 0usize;
    let mut spec_worst = 0f32;
    for row in 0..M {
        let want = reference(row);
        if ours[row].to_bits() == want.to_bits() {
            exact += 1;
        }
        let d = (ours[row] - want).abs() / want.abs().max(1e-6);
        if d > spec_worst {
            spec_worst = d;
        }
    }
    println!("vs proven spec : {exact}/{M} bit-exact, worst rel {spec_worst:.2e}");
    assert!(
        ours.iter().any(|v| v.abs() > 1e-3),
        "reference must be non-degenerate"
    );
    assert_eq!(exact, M, "GEMV must match the committed fold it is proven against");
}
