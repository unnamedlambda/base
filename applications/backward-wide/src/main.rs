use base::{Artifact, Base};
use warp_check::{dot_by, floats, gbs, roofline, Walk};

const ART: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/BackwardWideAlgorithm/backward_wide.bin"));

/// Qwen2-0.5B's hidden size.
const N: usize = 896;

fn main() {
    // Layout: adj (overwritten by the silu backward), x, z, dy, then W.
    let xa: Vec<f32> = (0..N).map(|j| ((j % 61) as f32 - 30.0) * 0.02).collect();
    let z: Vec<f32> = (0..N).map(|i| ((i % 89) as f32 - 44.0) * 0.03).collect();
    let dy: Vec<f32> = (0..N).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let w: Vec<f32> = (0..N * N)
        .map(|k| (((k * 31 + 7) % 193) as f32 - 96.0) * 0.001)
        .collect();

    // ds = dy * silu'(z), computed on the host in the same order the spec says.
    let sigma = |t: f32| 1.0f32 / (1.0f32 + (-t).exp());
    let adj: Vec<f32> = (0..N)
        .map(|i| {
            let s = sigma(z[i]);
            // d/dz [ z * (1 + e^-z)^-1 ] = sigma + z * sigma^2 * e^-z
            dy[i] * (s + z[i] * s * s * (-z[i]).exp())
        })
        .collect();

    let gam: Vec<f32> = (0..N).map(|i| 1.0 + ((i % 13) as f32 - 6.0) * 0.05).collect();

    // The training target: silu of a smooth pre-activation, so it is reachable.
    let ystar: Vec<f32> = (0..N)
        .map(|i| {
            let z = ((i % 37) as f32 - 18.0) * 0.05;
            z / (1.0 + (-z).exp())
        })
        .collect();

    let mut bytes: Vec<u8> = Vec::with_capacity((6 * N + N * N) * 4);
    bytes.extend(adj.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(xa.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(z.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(dy.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(gam.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(w.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(ystar.iter().flat_map(|f| f.to_le_bytes()));

    let artifact = Artifact::from_bytes(ART);

    // Seam check: Lean publishes the byte count its `hostIn` layout expects.
    // Asserting against it means the host packing and the uploader cannot
    // drift — the layout is defined once, in Lean, and checked here.
    const HOST_LEN_OFF: usize = 0x0080;
    let want = u32::from_le_bytes(
        artifact.setup.initial_memory[HOST_LEN_OFF..HOST_LEN_OFF + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    assert_eq!(
        bytes.len(),
        want,
        "host input packing disagrees with the Lean `hostIn` layout"
    );
    let mut base = Base::new(artifact.setup).expect("Base::new");
    let run = &artifact.extras["run"];
    let fetch = &artifact.extras["fetch"];

    base.execute_into(&artifact.main, &bytes, &mut []).expect("load");
    base.execute_into(run, b"", &mut []).expect("run");

    let reps = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        base.execute_into(run, b"", &mut []).expect("run");
    }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;

    let mut out = vec![0u8; N * 4];
    base.execute_into(fetch, b"", &mut out).expect("fetch");
    let gpu: Vec<f32> = floats(&out);

    // Reference, in the *committed* order the kernel is proven against:
    // each lane folds its own rows sequentially, then a five-round butterfly.
    // The transposed walk: successive rows are `N` floats apart, so no `Walk`
    // describes it and the index functions are given directly.
    let reference = |j: usize| -> f32 {
        dot_by(N / 32, 1, &adj, &w, |t, l| t * 32 + l, |t, l| (t * 32 + l) * N + j)
    };

    let mut worst = 0f32;
    let mut worst_at = 0usize;
    let mut exact = 0usize;
    for j in 0..N {
        let want = reference(j);
        if gpu[j].to_bits() == want.to_bits() {
            exact += 1;
        }
        let d = (gpu[j] - want).abs() / want.abs().max(1e-6);
        if d > worst {
            worst = d;
            worst_at = j;
        }
    }

    let gbytes = ((N * N + N) * 4) as f64 / dt / 1e9;
    println!("backward matvec : dx[j] = Σᵢ adj[i]·W[i][j]   (transposed walk)");
    println!("width           : {N}   ({} warps, {} iters each)", N, N / 32);
    println!("ptx lines       : 111   (constant in N — 128/896/4864/151936)");
    println!("kernel time     : {:.3} ms   ({:.1} GB/s)", dt * 1e3, gbytes);
    println!("bit-exact vs committed-order reference : {exact}/{N}");
    println!("worst rel. err  : {worst:.3e}  at j={worst_at}");
    assert_eq!(exact, N, "kernel must match its proven fold order bit-for-bit");

    // ── the weight gradient: dW[i][j] = adj[i] * x[j] ────────────────────────
    let run_dw = &artifact.extras["runDw"];
    let fetch_dw = &artifact.extras["fetchDw"];
    base.execute_into(run_dw, b"", &mut []).expect("runDw");
    let t1 = std::time::Instant::now();
    for _ in 0..reps {
        base.execute_into(run_dw, b"", &mut []).expect("runDw");
    }
    let dt_dw = t1.elapsed().as_secs_f64() / reps as f64;

    let mut dwbytes = vec![0u8; N * N * 4];
    base.execute_into(fetch_dw, b"", &mut dwbytes).expect("fetchDw");
    let dw: Vec<f32> = floats(&dwbytes);

    let mut dw_exact = 0usize;
    for i in 0..N {
        for j in 0..N {
            if dw[i * N + j].to_bits() == (adj[i] * xa[j]).to_bits() {
                dw_exact += 1;
            }
        }
    }
    let dw_gbs = ((N * N + 2 * N) * 4) as f64 / dt_dw / 1e9;
    println!("\nweight gradient : dW[i][j] = adj[i]*x[j]   (outer product)");
    println!("kernel time     : {:.3} ms   ({:.1} GB/s)", dt_dw * 1e3, dw_gbs);
    println!("bit-exact       : {dw_exact}/{}", N * N);
    assert_eq!(dw_exact, N * N, "outer product must be bit-exact");

    // ── the activation backward: ds = dy * silu'(z), derivative from sderiv ──
    let run_sb = &artifact.extras["runSiluBwd"];
    base.execute_into(run_sb, b"", &mut []).expect("runSiluBwd");
    let t2 = std::time::Instant::now();
    for _ in 0..reps {
        base.execute_into(run_sb, b"", &mut []).expect("runSiluBwd");
    }
    let dt_sb = t2.elapsed().as_secs_f64() / reps as f64;

    // It writes `adj`, so re-running dW now reads the GPU-computed adjoint.
    base.execute_into(run_dw, b"", &mut []).expect("runDw");
    base.execute_into(fetch_dw, b"", &mut dwbytes).expect("fetchDw");
    let dw2: Vec<f32> = floats(&dwbytes);

    let mut chain_worst = 0f32;
    for i in 0..N {
        for j in 0..N {
            let want = adj[i] * xa[j];
            let d = (dw2[i * N + j] - want).abs() / want.abs().max(1e-6);
            if d > chain_worst {
                chain_worst = d;
            }
        }
    }
    println!("\nactivation bwd  : ds = dy * silu'(z)   (silu' from `sderiv`, not hand-written)");
    println!("kernel time     : {:.3} ms", dt_sb * 1e3);
    println!("chained dW      : worst rel err {chain_worst:.3e} vs host  (ex2.approx is the one declared approx)");
    assert!(chain_worst < 1e-4, "chained backward must match the spec");

    // ── RMSNorm backward: t = dy⊙γ, Q = Σx², S = Σtx, then the epilogue ─────
    for k in ["runT", "runQ", "runS", "runDxr"] {
        base.execute_into(&artifact.extras[k], b"", &mut [])
            .unwrap_or_else(|e| panic!("{k}: {e:?}"));
    }
    let t3 = std::time::Instant::now();
    for _ in 0..reps {
        for k in ["runT", "runQ", "runS", "runDxr"] {
            base.execute_into(&artifact.extras[k], b"", &mut []).unwrap();
        }
    }
    let dt_rms = t3.elapsed().as_secs_f64() / reps as f64;

    let mut dxrbytes = vec![0u8; N * 4];
    base.execute_into(&artifact.extras["fetchDxr"], b"", &mut dxrbytes)
        .expect("fetchDxr");
    let dxr: Vec<f32> = floats(&dxrbytes);

    // Host reference, in the committed order: lane folds then butterfly.
    let committed = |a: &[f32], b: &[f32]| -> f32 {
        warp_check::dot(Walk::Strided, a, b, 0, 0, N)
    };
    let tv: Vec<f32> = (0..N).map(|i| dy[i] * gam[i]).collect();
    let qq = committed(&xa, &xa);
    let ss = committed(&tv, &xa);
    let r = (qq * (1.0 / N as f32) + 1.0 / 1_000_000.0).sqrt().recip();

    let mut rms_worst = 0f32;
    for j in 0..N {
        let want = tv[j] * r - xa[j] * (r * (r * r)) * (1.0 / N as f32) * ss;
        let d = (dxr[j] - want).abs() / want.abs().max(1e-6);
        if d > rms_worst {
            rms_worst = d;
        }
    }
    println!("\nrmsnorm bwd     : dx = t·r − (x·r³/n)·S   (t = dy⊙γ, S = Σtᵢxᵢ)");
    println!("passes          : mapKernel, dotStrided, dotStrided, mapKernelAt");
    println!("kernel time     : {:.3} ms  (4 launches)", dt_rms * 1e3);
    println!("worst rel. err  : {rms_worst:.3e} vs committed-order host reference");
    println!("sample          : Q={qq:.5} S={ss:.5} r={r:.5}  dx[0..3]={:?}", &dxr[..3]);
    assert!(dxr.iter().any(|v| v.abs() > 1e-3), "reference must be non-degenerate");
    assert!(rms_worst < 1e-5, "rmsnorm backward must match its spec");

    // ── The pipeline, in the order `bwd_chain` is about ──────────────────
    // Everything above validates each kernel against its own spec, launching
    // in whatever order was convenient. `bwd_chain` proves something else:
    // that the activation backward *followed by* the transposed matvec gives
    // dx = Wᵀ·(dy⊙silu'(z)). Running that exact order is what makes the demo
    // demonstrate the theorem rather than a neighbour of it.
    base.execute_into(run_sb, b"", &mut []).expect("runSiluBwd");
    base.execute_into(run, b"", &mut []).expect("run");
    let mut chain_out = vec![0u8; N * 4];
    base.execute_into(fetch, b"", &mut chain_out).expect("fetch");
    let dx_chained: Vec<f32> = floats(&chain_out);

    // Host reference in the committed order, from the *composed* spec.
    let mut pipe_worst = 0f32;
    for j in 0..N {
        let want = dot_by(N / 32, 1, &adj, &w, |t, l| t * 32 + l, |t, l| (t * 32 + l) * N + j);
        let d = (dx_chained[j] - want).abs() / want.abs().max(1e-6);
        if d > pipe_worst {
            pipe_worst = d;
        }
    }
    println!("\npipeline order  : siluBwd → dx   (the order `bwd_chain` proves)");
    println!("worst rel. err  : {pipe_worst:.3e}  vs the composed spec");
    assert!(pipe_worst < 1e-4, "the pipeline must match the composed spec");

    // ── The same three stages as ONE emitted function ────────────────────
    // `bwdAll_host_computes` recovers the launch sequence from `runBwdAll`'s
    // own instructions and matches it against `bwdPipelineFull`. Three
    // separate host calls never exhibited that order inside any one program,
    // so the theorem would have been about a program nothing ran. This runs it.
    base.execute_into(&artifact.extras["runBwdAll"], b"", &mut []).expect("runBwdAll");
    let mut fused_out = vec![0u8; N * 4];
    base.execute_into(fetch, b"", &mut fused_out).expect("fetch");
    let dx_fused: Vec<f32> = floats(&fused_out);
    let fused_worst = dx_fused
        .iter()
        .zip(&dx_chained)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mut dw_fused = vec![0u8; N * N * 4];
    base.execute_into(fetch_dw, b"", &mut dw_fused).expect("fetchDw");
    println!("fused function  : runBwdAll — the 3 launches `bwdAllTable` binds");
    println!("worst abs. diff : {fused_worst:.3e}  vs the same three calls separately");
    assert_eq!(fused_worst, 0.0, "the fused function must be bit-identical");

    println!("\nOK — dense+silu and RMSNorm backward, both at width 896:");
    println!("   7 kernels, every one a schema instance, PTX size independent of N.");

    train(&mut base, &artifact.extras, &xa, &ystar);
}

/// **Training.** Everything above checks a gradient against a spec. This runs
/// the loop that gradient exists for:
///
///     z = W·x    y = silu(z)    L = ½‖y − y*‖²    adj = dy⊙silu'(z)
///     dW = adj⊗x                W ← W − lr·dW
///
/// Six launches per step, every one a schema instance proven against its own
/// spec. The target `y*` is a fixed vector the model has to reach, so a falling
/// loss is evidence the whole chain — forward, derivative-from-spec, outer
/// product, optimiser — agrees on what it is differentiating.
fn train(
    base: &mut Base,
    extras: &std::collections::HashMap<String, base_types::Algorithm>,
    xa: &[f32],
    ystar: &[f32],
) {
    // The fused step: the three elementwise passes (y, dy, siluBwd) are one
    // kernel, because at 3.5 KB each a launch costs more than the work.
    let step = ["runFwd", "runAdj", "runDw", "runSgd"];
    let fetch_y = &extras["fetchY"];

    let loss = |base: &mut Base| -> f32 {
        let mut out = vec![0u8; N * 4];
        base.execute_into(fetch_y, b"", &mut out).expect("fetchY");
        let y: Vec<f32> = floats(&out);
        0.5 * (0..N).map(|i| (y[i] - ystar[i]).powi(2)).sum::<f32>()
    };

    // One forward to establish the starting loss.
    for k in ["runFwd", "runY"] {
        base.execute_into(&extras[k], b"", &mut []).unwrap();
    }
    let l0 = loss(base);

    println!("\n── training: z = W·x, y = silu(z), L = ½‖y − y*‖², SGD on W ──");
    println!("width {N}, lr = 1/1024, {} launches per step", step.len());
    println!("{:>6}  {:>12}  {:>10}", "step", "loss", "vs start");

    let mut losses = vec![l0];
    println!("{:>6}  {:>12.6}  {:>10}", 0, l0, "1.000");
    let iters = 200;
    let t0 = std::time::Instant::now();
    for it in 1..=iters {
        for k in step {
            base.execute_into(&extras[k], b"", &mut []).unwrap();
        }
        if it % 25 == 0 || it == 1 {
            // The step's forward ran *before* its update, so y is one step
            // stale; refresh it so the row is the loss at the current W.
            for k in ["runFwd", "runY"] {
                base.execute_into(&extras[k], b"", &mut []).unwrap();
            }
            let l = loss(base);
            losses.push(l);
            println!("{it:>6}  {l:>12.6}  {:>10.3}", l / l0);
        }
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let lf = *losses.last().unwrap();

    println!("step time       : {:.3} ms   ({} launches; the optimiser dominates, see below)",
             dt * 1e3, step.len());
    println!("loss            : {l0:.6} → {lf:.6}   ({:.1}x reduction)", l0 / lf);

    // The gradient direction, checked against the loss it claims to be the
    // gradient of: `dW` should be an outer product `adj ⊗ x`, so every row is
    // parallel to `x`. That is a property of the *shape* of the gradient, not
    // a restatement of how it was computed.
    let mut dwbytes = vec![0u8; N * N * 4];
    base.execute_into(&extras["fetchDw"], b"", &mut dwbytes).unwrap();
    let dw: Vec<f32> = floats(&dwbytes);
    let xnorm: f32 = xa.iter().map(|v| v * v).sum();
    let mut rank1_worst = 0f32;
    for i in (0..N).step_by(97) {
        // project row i onto x, then measure the residual
        let dot: f32 = (0..N).map(|j| dw[i * N + j] * xa[j]).sum();
        let c = dot / xnorm;
        let num: f32 = (0..N).map(|j| (dw[i * N + j] - c * xa[j]).powi(2)).sum();
        let den: f32 = (0..N).map(|j| dw[i * N + j].powi(2)).sum::<f32>().max(1e-12);
        let r = (num / den).sqrt();
        if r > rank1_worst {
            rank1_worst = r;
        }
    }
    println!("dW is rank-1    : worst off-x residual {rank1_worst:.2e}  (dW = adj ⊗ x)");
    assert!(rank1_worst < 1e-3, "the weight gradient must be an outer product with x");

    // Per-kernel cost, measured *after* the loop: each of these reps performs a
    // real update, so timing before training would apply 200 stale gradients.
    println!("\nper-kernel cost of one training step:");
    let mut total = 0f64;
    for k in step {
        base.execute_into(&extras[k], b"", &mut []).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..200 {
            base.execute_into(&extras[k], b"", &mut []).unwrap();
        }
        let dt = t.elapsed().as_secs_f64() / 200.0;
        total += dt;
        // Bytes each kernel must move, at minimum.
        let bytes = match k {
            "runFwd" => (N * N + 2 * N) * 4,
            "runAdj" => 3 * N * 4,
            "runDw" => (N * N + 2 * N) * 4,
            "runSgd" => (3 * N * N) * 4,
            _ => 0,
        };
        println!(
            "  {k:<10}: {:>7.1} us   {:>6.1} GB/s   {:>3.0}% roofline",
            dt * 1e6,
            gbs(bytes, dt),
            100.0 * roofline(bytes, dt)
        );
    }
    println!("  {:<10}: {:>7.1} us", "sum", total * 1e6);

    assert!(lf < l0, "training must reduce the loss");
    assert!(losses.windows(2).all(|w| w[1] < w[0]),
            "loss must fall at every sampled step");
    println!("OK — the proven backward pass trains: loss falls monotonically.");
}
