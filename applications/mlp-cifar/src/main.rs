//! Correctness and timing for the CIFAR MLP's ten kernels, at batch `B`.
//!
//! Every kernel is checked against a CPU reference written in the **committed**
//! fold order the Lean theorems are about — sequential within a lane, then a
//! five-round butterfly across lanes, and for the gradients a sequential sum
//! over the batch. Where that order is all there is, the comparison is
//! bit-for-bit; the two kernels that evaluate `silu` are compared by error
//! magnitude, because `ex2.approx` is the stack's one declared approximation.
//! The activation backward is checked on *absolute* error: `silu'` has a root,
//! and relative error at an element sitting on it measures the root, not the
//! kernel.
//!
//! The training loop itself lives in `train.py`, which drives the same artifact
//! against real CIFAR-10.

use base::{Artifact, Base};
use warp_check::{compare, dot, dot_by, floats, gbs, le, roofline, time, Lcg, Walk};

const ART: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/MlpCifarAlgorithm/mlp_cifar.bin"));

const IN: usize = 3072;
const H: usize = 256;
const C: usize = 32;
const CLASSES: usize = 10;
/// Must match `MlpCifar.B`.
const B: usize = 8;
/// Must match `MlpCifar.LR_RECIP` — the `B` is what makes the step the *mean*
/// gradient, since the gradient kernels accumulate a sum over the batch.
const LR: f32 = 1.0 / (256.0 * B as f32);

/// The activation and its derivative — what `Transformer.silu` and its `sderiv`
/// denote. The GPU differs only by `ex2.approx`.
fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

fn silu_prime(z: f32) -> f32 {
    let s = 1.0f32 / (1.0f32 + (-z).exp());
    s + z * s * s * (-z).exp()
}

/// The batch sum, in the order `outerBatched` commits to: sample 0 first.
fn batch_sum(f: impl Fn(usize) -> f32) -> f32 {
    let mut acc = 0f32;
    for s in 0..B {
        acc += f(s);
    }
    acc
}

fn main() {
    let artifact = Artifact::from_bytes(ART);

    let mut rng = Lcg::new(0x5eed_1234);
    let a1 = (6.0f32 / IN as f32).sqrt();
    let a2 = (6.0f32 / H as f32).sqrt();
    let w1 = rng.vec(H * IN, a1);
    // Padding classes start at zero and never move: their gradient is zeroed.
    let w2: Vec<f32> = (0..C * H)
        .map(|k| if k / H < CLASSES { rng.next() * a2 } else { 0.0 })
        .collect();
    let x = rng.vec(B * IN, 1.0);

    let mut blob = le(&w1);
    blob.extend(le(&w2));

    // Seam check: Lean publishes the byte count its `hostIn` layout expects, so
    // the packing here and the uploader there cannot drift apart silently.
    const HOST_LEN_OFF: usize = 0x0080;
    let want = u32::from_le_bytes(
        artifact.setup.initial_memory[HOST_LEN_OFF..HOST_LEN_OFF + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    assert_eq!(blob.len(), want, "host packing disagrees with Lean's `hostIn`");

    let mut base = Base::new(artifact.setup).expect("Base::new");
    let ex = &artifact.extras;
    let go = |b: &mut Base, k: &str, data: &[u8], out: &mut [u8]| {
        b.execute_into(&ex[k], data, out).unwrap_or_else(|e| panic!("{k}: {e:?}"));
    };

    base.execute_into(&artifact.main, &blob, &mut []).expect("load");
    println!("MLP {IN} → {H} → {CLASSES} (padded to {C}), batch {B}, lr = 1/{}", (1.0 / LR) as u32);
    println!("host packing: {} bytes, matches Lean's hostIn\n", blob.len());

    // ── forward ─────────────────────────────────────────────────────────────
    go(&mut base, "uploadX", &le(&x), &mut []);
    go(&mut base, "runFwd1", b"", &mut []);

    let mut buf_bh = vec![0u8; B * H * 4];
    go(&mut base, "fetchZ1", b"", &mut buf_bh);
    let z1 = floats(&buf_bh);
    // Output `k` is sample `k/H`, hidden unit `k%H`.
    let d = compare(&z1, |k| dot(Walk::Strided, &w1, &x, (k % H) * IN, (k / H) * IN, IN));
    println!("fwd1  z1 = W1·x        {d}");
    assert!(d.is_exact(), "fwd1 must match its proven fold order bit-for-bit");

    go(&mut base, "runAct", b"", &mut []);
    go(&mut base, "fetchH", b"", &mut buf_bh);
    let hv = floats(&buf_bh);
    let d = compare(&hv, |k| silu(z1[k]));
    println!("act   h  = silu(z1)    worst rel {:.2e}   (ex2.approx, the declared approximation)", d.worst);
    assert!(d.worst < 1e-5, "silu must match the spec to within the declared approximation");

    go(&mut base, "runFwd2", b"", &mut []);
    let mut buf_bc = vec![0u8; B * C * 4];
    go(&mut base, "fetchLogits", b"", &mut buf_bc);
    let logits = floats(&buf_bc);
    // Referenced against the *fetched* `h`, so this isolates fwd2 from `act`.
    let d = compare(&logits, |k| dot(Walk::Strided, &w2, &hv, (k % C) * H, (k / C) * H, H));
    println!("fwd2  logits = W2·h    {d}");
    assert!(d.is_exact(), "fwd2 must match its proven fold order bit-for-bit");

    // ── softmax and the cross-entropy gradient, on the device ───────────────
    // `C = 32` is a warp, so each row's max and sum are one butterfly. The
    // padding classes are masked by `bias`: very negative there, so `exp`
    // underflows to zero and they contribute nothing.
    let labels: Vec<usize> = (0..B).map(|s| (s * 3 + 1) % CLASSES).collect();
    let bias: Vec<f32> = (0..C).map(|c| if c < CLASSES { 0.0 } else { -1.0e30 }).collect();
    let mut onehot = vec![0f32; B * C];
    for s in 0..B {
        onehot[s * C + labels[s]] = 1.0;
    }
    go(&mut base, "uploadBias", &le(&bias), &mut []);
    go(&mut base, "uploadOneHot", &le(&onehot), &mut []);
    go(&mut base, "runSoftmax", b"", &mut []);
    let mut buf_dl = vec![0u8; B * C * 4];
    go(&mut base, "fetchDlog", b"", &mut buf_dl);
    let dlog = floats(&buf_dl);

    // Reference in the committed order: the max and the sum are the same
    // five-round butterfly the theorem is about, over all 32 lanes.
    let mut want_dl = vec![0f32; B * C];
    for s in 0..B {
        let lm: Vec<f32> = (0..C).map(|c| logits[s * C + c] + bias[c]).collect();
        let mut lane = [0f32; 32];
        lane.copy_from_slice(&lm);
        for m in [16usize, 8, 4, 2, 1] {
            let prev = lane;
            for l in 0..32 {
                lane[l] = prev[l].max(prev[l ^ m]);
            }
        }
        let mx = lane[0];
        let p: Vec<f32> = lm.iter().map(|v| (v - mx).exp()).collect();
        let mut sl = [0f32; 32];
        sl.copy_from_slice(&p);
        for m in [16usize, 8, 4, 2, 1] {
            let prev = sl;
            for l in 0..32 {
                sl[l] = prev[l] + prev[l ^ m];
            }
        }
        for c in 0..C {
            want_dl[s * C + c] = p[c] / sl[0] - onehot[s * C + c];
        }
    }
    let d = compare(&dlog, |k| want_dl[k]);
    println!("sm    softmax + CE grad  worst rel {:.2e}   (on the device; ex2.approx)", d.worst);
    assert!(d.worst_abs < 1e-6, "softmaxCE must match its proven spec");

    // ── backward ────────────────────────────────────────────────────────────
    go(&mut base, "runDw2", b"", &mut []);
    let mut buf_w2 = vec![0u8; C * H * 4];
    go(&mut base, "fetchDw2", b"", &mut buf_w2);
    let dw2 = floats(&buf_w2);
    let d = compare(&dw2, |k| batch_sum(|s| dlog[s * C + k / H] * hv[s * H + k % H]));
    println!("dw2   dW2 = Σₛ dlog⊗h  {d}");
    assert!(d.is_exact(), "the batched outer product must be bit-exact");

    go(&mut base, "runDh", b"", &mut []);
    go(&mut base, "fetchDh", b"", &mut buf_bh);
    let dh = floats(&buf_bh);
    // The transposed walk: successive classes are `H` floats apart, which is
    // why no `Walk` describes it and `dot_by` takes the index functions.
    let d = compare(&dh, |k| {
        let (s, i) = (k / H, k % H);
        dot_by(C / 32, 1, &w2, &dlog, |t, l| (t * 32 + l) * H + i, |t, l| s * C + t * 32 + l)
    });
    println!("dh    dh = W2ᵀ·dlog    {d}");
    assert!(d.is_exact(), "the transposed matvec must match its proven fold order");

    go(&mut base, "runAdj", b"", &mut []);
    go(&mut base, "fetchAdj", b"", &mut buf_bh);
    let adj = floats(&buf_bh);
    let d = compare(&adj, |k| dh[k] * silu_prime(z1[k]));
    println!("adj   adj = dh·silu'   worst rel {:.2e} (ref {:.2e} at {}), worst abs {:.2e}",
             d.worst, d.ref_at_worst, d.worst_at, d.worst_abs);
    // Relative error is not meaningful at a root of `silu'`; the absolute error
    // is what says the kernel agrees with the spec's derivative.
    assert!(d.worst_abs < 1e-5, "the activation backward must match the spec's derivative");

    go(&mut base, "runDw1", b"", &mut []);
    let mut buf_w1 = vec![0u8; H * IN * 4];
    go(&mut base, "fetchDw1", b"", &mut buf_w1);
    let dw1 = floats(&buf_w1);
    let d = compare(&dw1, |k| batch_sum(|s| adj[s * H + k / IN] * x[s * IN + k % IN]));
    println!("dw1   dW1 = Σₛ adj⊗x   {d}");
    assert!(d.is_exact(), "the batched outer product must be bit-exact");

    // ── the optimiser ───────────────────────────────────────────────────────
    go(&mut base, "runSgd1", b"", &mut []);
    go(&mut base, "runSgd2", b"", &mut []);
    go(&mut base, "fetchW1", b"", &mut buf_w1);
    let d = compare(&floats(&buf_w1), |k| w1[k] - LR * dw1[k]);
    println!("sgd1  W1 ← W1 − lr·dW1 {d}");
    assert!(d.is_exact(), "the optimiser step must be bit-exact");

    go(&mut base, "fetchW2", b"", &mut buf_w2);
    let d = compare(&floats(&buf_w2), |k| w2[k] - LR * dw2[k]);
    println!("sgd2  W2 ← W2 − lr·dW2 {d}");
    assert!(d.is_exact(), "the optimiser step must be bit-exact");

    // ── the same five GEMMs through cuBLAS ──────────────────────────────────
    // Identical arithmetic to what PyTorch runs, so this is not a race. What
    // it has to be checked for is the transpose/leading-dimension convention:
    // the buffers are row-major and cuBLAS is column-major, and getting that
    // wrong yields plausible numbers rather than an error. The proven kernels
    // are the oracle.
    // The optimiser above has already stepped W1/W2, so reset to the weights
    // the references were computed with before comparing.
    let mut blas_ok = true;
    base.execute_into(&artifact.main, &blob, &mut []).expect("reload");
    go(&mut base, "uploadBias", &le(&bias), &mut []);
    go(&mut base, "uploadOneHot", &le(&onehot), &mut []);
    go(&mut base, "uploadX", &le(&x), &mut []);
    go(&mut base, "runFwdBlas", b"", &mut []);
    go(&mut base, "fetchZ1", b"", &mut buf_bh);
    let chk = |name: &str, got: &[f32], want: &[f32], ok: &mut bool| {
        // Relative error is meaningless at an element that cancels to ~0, so
        // measure the worst absolute deviation against the tensor's own scale.
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-30);
        let worst = got.iter().zip(want).fold(0f32, |a, (g, w)| a.max((g - w).abs()));
        println!("blas  {name:<11} worst abs {worst:.2e} / scale {scale:.2e} = {:.2e}", worst / scale);
        *ok &= worst / scale < 1e-5;
    };
    chk("z1", &floats(&buf_bh), &z1, &mut blas_ok);
    go(&mut base, "fetchLogits", b"", &mut buf_bc);
    chk("logits", &floats(&buf_bc), &logits, &mut blas_ok);

    go(&mut base, "runBwdBlas", b"", &mut []);
    go(&mut base, "fetchDh", b"", &mut buf_bh);
    chk("dh", &floats(&buf_bh), &dh, &mut blas_ok);
    go(&mut base, "fetchDw2", b"", &mut buf_w2);
    chk("dW2", &floats(&buf_w2), &dw2, &mut blas_ok);
    go(&mut base, "fetchDw1", b"", &mut buf_w1);
    chk("dW1", &floats(&buf_w1), &dw1, &mut blas_ok);
    assert!(blas_ok, "cuBLAS must agree with the proven kernels");

    // ── per-kernel timing ───────────────────────────────────────────────────
    let reps = 200;
    // Bytes each kernel must move at minimum. The weight terms are what
    // batching amortises: they are the same at every `B`, over `B` samples.
    let traffic: &[(&str, usize)] = &[
        ("runFwd1", (H * IN + B * IN + B * H) * 4),
        ("runAct", 2 * B * H * 4),
        ("runFwd2", (C * H + B * H + B * C) * 4),
        ("runDw2", (C * H + B * C + B * H) * 4),
        ("runDh", (C * H + B * C + B * H) * 4),
        ("runAdj", 3 * B * H * 4),
        ("runDw1", (H * IN + B * H + B * IN) * 4),
        ("runSgd1", 3 * H * IN * 4),
        ("runSgd2", 3 * C * H * 4),
        ("runSoftmax", 3 * B * C * 4),
    ];
    println!("\n{:<9} {:>9} {:>11} {:>9}", "kernel", "time", "bandwidth", "roofline");
    let mut total = 0f64;
    for (k, bytes) in traffic {
        let dt = time(&mut base, &ex[*k], reps);
        total += dt;
        println!("{:<9} {:>7.1} us {:>8.1} GB/s {:>8.0}%", k, dt * 1e6,
                 gbs(*bytes, dt), roofline(*bytes, dt) * 100.0);
    }
    println!("{:<9} {:>7.1} us   (10 separate launch-and-sync calls)", "sum", total * 1e6);

    // The same ten kernels enqueued as two runs with one sync each. Same
    // device work; what disappears is seven round trips.
    let oh = le(&onehot);
    let xb = le(&x);
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        base.execute_into(&ex["uploadX"], &xb, &mut []).unwrap();
        base.execute_into(&ex["runFwd"], b"", &mut []).unwrap();
        base.execute_into(&ex["fetchLogits"], b"", &mut buf_bc).unwrap();
        base.execute_into(&ex["uploadOneHot"], &oh, &mut []).unwrap();
        base.execute_into(&ex["runBwd"], b"", &mut []).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    println!("{:<9} {:>7.1} us   (fused: 2 runs, 2 uploads, 1 download)", "step", dt * 1e6);
    println!("{:<9} {:>7.1} us   per sample at batch {B}", "", dt * 1e6 / B as f64);

    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        base.execute_into(&ex["uploadX"], &xb, &mut []).unwrap();
        base.execute_into(&ex["runFwdBlas"], b"", &mut []).unwrap();
        base.execute_into(&ex["fetchLogits"], b"", &mut buf_bc).unwrap();
        base.execute_into(&ex["uploadOneHot"], &oh, &mut []).unwrap();
        base.execute_into(&ex["runBwdBlas"], b"", &mut []).unwrap();
    }
    let dtb = t0.elapsed().as_secs_f64() / reps as f64;
    println!("{:<9} {:>7.1} us   (cuBLAS for the 5 GEMMs)", "step/blas", dtb * 1e6);
    println!("{:<9} {:>7.1} us   per sample at batch {B}", "", dtb * 1e6 / B as f64);
    println!("\nOK — every kernel matches its proven fold order.");
}
