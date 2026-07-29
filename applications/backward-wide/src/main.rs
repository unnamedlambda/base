use base::{Artifact, Base};

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

    let mut bytes: Vec<u8> = Vec::with_capacity((5 * N + N * N) * 4);
    bytes.extend(adj.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(xa.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(z.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(dy.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(gam.iter().flat_map(|f| f.to_le_bytes()));
    bytes.extend(w.iter().flat_map(|f| f.to_le_bytes()));

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
    let gpu: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Reference, in the *committed* order the kernel is proven against:
    // each lane folds its own rows sequentially, then a five-round butterfly.
    let reference = |j: usize| -> f32 {
        let mut lane = [0f32; 32];
        for (l, acc) in lane.iter_mut().enumerate() {
            for t in 0..N / 32 {
                let i = t * 32 + l;
                *acc += adj[i] * w[i * N + j];
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
    let dw: Vec<f32> = dwbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

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
    let dw2: Vec<f32> = dwbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

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
    let dxr: Vec<f32> = dxrbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Host reference, in the committed order: lane folds then butterfly.
    let committed = |a: &[f32], b: &[f32]| -> f32 {
        let mut lane = [0f32; 32];
        for (l, acc) in lane.iter_mut().enumerate() {
            for t in 0..N / 32 {
                let i = t * 32 + l;
                *acc += a[i] * b[i];
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
    let dx_chained: Vec<f32> = chain_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Host reference in the committed order, from the *composed* spec.
    let mut pipe_worst = 0f32;
    for j in 0..N {
        let mut lane = [0f32; 32];
        for (l, acc) in lane.iter_mut().enumerate() {
            for t in 0..N / 32 {
                let i = t * 32 + l;
                *acc += adj[i] * w[i * N + j];
            }
        }
        for m in [16usize, 8, 4, 2, 1] {
            let prev = lane;
            for l in 0..32 {
                lane[l] = prev[l] + prev[l ^ m];
            }
        }
        let d = (dx_chained[j] - lane[0]).abs() / lane[0].abs().max(1e-6);
        if d > pipe_worst {
            pipe_worst = d;
        }
    }
    println!("\npipeline order  : siluBwd → dx   (the order `bwd_chain` proves)");
    println!("worst rel. err  : {pipe_worst:.3e}  vs the composed spec");
    assert!(pipe_worst < 1e-4, "the pipeline must match the composed spec");

    println!("\nOK — dense+silu and RMSNorm backward, both at width 896:");
    println!("   7 kernels, every one a schema instance, PTX size independent of N.");
}
