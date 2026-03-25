use base::{BaseConfig, Algorithm};
use crate::harness;

type CudaBackend = burn::backend::CudaJit;

const CUDA_SAXPY_ALGORITHM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_saxpy_algorithm.bin"));

fn load_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(CUDA_SAXPY_ALGORITHM).expect("Failed to deserialize cuda_saxpy algorithm")
}

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
// Burn(CUDA) SAXPY — full readback for fair comparison
// ---------------------------------------------------------------------------

fn burn_saxpy_cuda(a: f32, x: &[f32], y: &[f32]) -> Vec<f32> {
    use burn::tensor::{Tensor, TensorData};
    let device = cuda_device();
    let n = x.len();
    let x_t = Tensor::<CudaBackend, 1>::from_data(TensorData::new(x.to_vec(), [n]), &device);
    let y_t = Tensor::<CudaBackend, 1>::from_data(TensorData::new(y.to_vec(), [n]), &device);
    let result = (x_t.mul_scalar(a) + y_t).into_data();
    result.as_slice::<f32>().unwrap().to_vec()
}

// ---------------------------------------------------------------------------
// Payload builder
// ---------------------------------------------------------------------------

/// SAXPY payload: [x floats][y floats]
fn build_payload(x: &[f32], y: &[f32]) -> Vec<u8> {
    let mut payload = Vec::with_capacity((x.len() + y.len()) * 4);
    for &v in x {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    for &v in y {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    payload
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

fn f32_from_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
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
        "Base+CUDA",
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

    eprintln!("\n=== CUDA Benchmarks: Burn(cuda-jit) vs Base+CUDA ===");
    eprintln!("  SAXPY: y[i] = 2.0 * x[i] + y[i]");
    eprintln!("  Both: upload + compute + full readback via execute_into\n");

    let (config, alg) = load_algorithm();
    let mut base_instance = base::Base::new(config).expect("Base::new failed");

    for &n in &[262_144usize, 524_288, 1_048_576] {
        eprintln!("  SAXPY {} ...", format_count(n));

        let x = gen_floats(n, 42);
        let y = gen_floats(n, 123);
        let expected = cpu_saxpy(2.0, &x, &y);
        let payload = build_payload(&x, &y);
        let out_size = n * 4;
        let mut out_buf = vec![0u8; out_size];
        let label = format!("SAXPY {}", format_count(n));

        // Warmup both
        std::hint::black_box(burn_saxpy_cuda(2.0, &x, &y));
        let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);

        let burn_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            std::hint::black_box(burn_saxpy_cuda(2.0, &x, &y));
            start.elapsed().as_secs_f64() * 1000.0
        });

        let base_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Verify
        let burn_result = burn_saxpy_cuda(2.0, &x, &y);
        let base_result = f32_from_bytes(&out_buf);
        let burn_ok = check_saxpy_result(&burn_result, &expected, "Burn", &label);
        let base_ok = check_saxpy_result(&base_result, &expected, "Base", &label);

        results.push(CudaBenchResult {
            name: label,
            burn_ms,
            base_ms,
            verified: burn_ok && base_ok,
        });
    }

    results
}
