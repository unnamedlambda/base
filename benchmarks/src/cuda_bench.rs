use crate::harness;

type CudaBackend = burn::backend::CudaJit;

fn gen_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 33) as i32;
        out.push(bits as f32 / i32::MAX as f32);
    }
    out
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}

fn cuda_device() -> burn::backend::cuda_jit::CudaDevice {
    burn::backend::cuda_jit::CudaDevice::new(0)
}

fn read_f32_file(path: &str) -> Option<Vec<f32>> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 4 || data.len() % 4 != 0 {
        return None;
    }
    let mut vals = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        vals.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Some(vals)
}

fn check_saxpy_result(actual: &[f32], expected: &[f32], impl_name: &str, label: &str) -> bool {
    if actual.len() != expected.len() {
        eprintln!(
            "  VERIFY FAIL [{}] {}: length mismatch: got {}, expected {}",
            label, impl_name, actual.len(), expected.len()
        );
        return false;
    }
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let rel = (a as f64 - e as f64).abs() / (e as f64).abs().max(1e-10);
        if rel > 0.001 {
            eprintln!(
                "  VERIFY FAIL [{}] {}: element {} got {:.6}, expected {:.6} (rel err {:.6})",
                label, impl_name, i, a, e, rel
            );
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// CPU reference: y[i] = 2.0 * x[i] + y[i]
// ---------------------------------------------------------------------------

fn cpu_saxpy(a: f32, x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| a * xi + yi)
        .collect()
}

// ---------------------------------------------------------------------------
// Burn(CUDA) SAXPY
// ---------------------------------------------------------------------------

fn burn_saxpy_cuda(
    a: f32,
    x: &[f32],
    y: &[f32],
    output_path: &str,
) {
    use burn::tensor::{Tensor, TensorData};
    let device = cuda_device();
    let n = x.len();
    let x_t = Tensor::<CudaBackend, 1>::from_data(TensorData::new(x.to_vec(), [n]), &device);
    let y_t = Tensor::<CudaBackend, 1>::from_data(TensorData::new(y.to_vec(), [n]), &device);
    let result = (x_t.mul_scalar(a) + y_t).into_data();
    std::fs::write(output_path, result.as_bytes()).ok();
}

// ---------------------------------------------------------------------------
// CLIF+CUDA (Lean-generated algorithm): SAXPY via PTX kernel
//
// Memory layout (from SaxpyBenchAlgorithm.lean):
//   0x000: reserved (64 bytes, CUDA context pointer at offset 0)
//   0x100: nElems (i32)
//   0x200: ptxSrc (4096 bytes, null-terminated)
//   0x1200: bindDesc (8 bytes: 2 x i32 buffer ids)
//   0x1208: filename (256 bytes, null-terminated)
//   0x1408: dataRegion (x[0..N] then y[0..N], f32 each)
// ---------------------------------------------------------------------------

const NELEMS_OFF: usize = 0x100;
const DATA_OFF: usize = 0x1408;
const FNAME_OFF: usize = 0x1208;

fn build_saxpy_algorithm(
    n: usize,
    x: &[f32],
    y: &[f32],
    output_path: &str,
) -> (base::BaseConfig, base::Algorithm) {
    let (config, mut algorithm): (base::BaseConfig, base::Algorithm) = {
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/saxpy_algorithm.bin"));
        bincode::deserialize(bytes).expect("Failed to deserialize saxpy_algorithm")
    };

    algorithm.payloads[NELEMS_OFF..NELEMS_OFF + 4]
        .copy_from_slice(&(n as i32).to_le_bytes());

    let fname = format!("{}\0", output_path);
    let fname_bytes = fname.as_bytes();
    algorithm.payloads[FNAME_OFF..FNAME_OFF + fname_bytes.len()]
        .copy_from_slice(fname_bytes);

    for (i, &v) in x.iter().enumerate() {
        let off = DATA_OFF + i * 4;
        algorithm.payloads[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, &v) in y.iter().enumerate() {
        let off = DATA_OFF + n * 4 + i * 4;
        algorithm.payloads[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    (config, algorithm)
}

// ---------------------------------------------------------------------------
// Table printer
// ---------------------------------------------------------------------------

pub struct CudaBenchResult {
    pub name: String,
    pub burn_ms: f64,
    pub base_ms: f64,
    pub verified: bool,
}

pub fn print_cuda_table(results: &[CudaBenchResult]) {
    let name_w = 22;
    let col_w = 14;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
        "CUDA Benchmark",
        "Burn(cuda)",
        "Base",
        "Check",
        name_w = name_w,
        col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 2 + 6 + 3));

    for r in results {
        let check_str = if r.verified { "\u{2713}" } else { "\u{2717}" };

        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
            r.name,
            format!("{:.1}ms", r.burn_ms),
            format!("{:.1}ms", r.base_ms),
            check_str,
            name_w = name_w,
            col_w = col_w
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

pub fn run(iterations: usize) -> Vec<CudaBenchResult> {
    let mut results = Vec::new();
    std::fs::create_dir_all("/tmp/cuda-bench-data").ok();

    eprintln!("\n=== CUDA Benchmarks: Burn(cuda-jit) vs Base CLIF+PTX ===");
    eprintln!("  SAXPY: y[i] = 2.0 * x[i] + y[i]");
    eprintln!("  Burn: cubecl global runtime (pooled allocs, kernel cache)");
    eprintln!("  Base: base::run() — JIT + CUDA pipeline each call\n");


    for &n in &[262_144usize, 524_288, 1_048_576] {
        eprintln!("  SAXPY {} ...", format_count(n));

        let x = gen_floats(n, 42);
        let y = gen_floats(n, 123);
        let expected = cpu_saxpy(2.0, &x, &y);

        let label = format!("saxpy_{}", format_count(n));
        let burn_out = format!("/tmp/cuda-bench-data/{}_burn.bin", label);
        let base_out = format!("/tmp/cuda-bench-data/{}_base.bin", label);

        // Burn
        let burn_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            burn_saxpy_cuda(2.0, &x, &y, &burn_out);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Base
        let (base_cfg, base_alg) = build_saxpy_algorithm(n, &x, &y, &base_out);
        let base_ms = harness::median_of(iterations, || {
            let cfg = base_cfg.clone();
            let alg = base_alg.clone();
            let start = std::time::Instant::now();
            let _ = base::run(cfg, alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Verify both
        let burn_ok = read_f32_file(&burn_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] Burn: could not read output", label); false },
            |v| check_saxpy_result(&v, &expected, "Burn", &label),
        );
        let base_ok = read_f32_file(&base_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] Base: could not read output", label); false },
            |v| check_saxpy_result(&v, &expected, "Base", &label),
        );

        results.push(CudaBenchResult {
            name: format!("SAXPY {}", format_count(n)),
            burn_ms,
            base_ms,
            verified: burn_ok && base_ok,
        });
    }

    results
}
