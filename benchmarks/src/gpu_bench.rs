use crate::harness::{self, BenchResult};
use base::{Algorithm, BaseConfig};
type Gpu = burn::backend::wgpu::Wgpu;

const GPU_VECADD_ALGORITHM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gpu_vecadd_algorithm.bin"));
const GPU_MATMUL_ALGORITHM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gpu_matmul_algorithm.bin"));
const GPU_REDUCTION_ALGORITHM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gpu_reduction_algorithm.bin"));

fn load_vecadd_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(GPU_VECADD_ALGORITHM).expect("Failed to deserialize gpu_vecadd algorithm")
}

fn load_matmul_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(GPU_MATMUL_ALGORITHM).expect("Failed to deserialize gpu_matmul algorithm")
}

fn load_reduction_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(GPU_REDUCTION_ALGORITHM)
        .expect("Failed to deserialize gpu_reduction algorithm")
}

use harness::{build_f32_payload, f32_sum, format_count, gen_floats};

/// Get (or create) the cached Burn GPU device.
fn burn_device() -> burn::backend::wgpu::WgpuDevice {
    use std::sync::{Arc, OnceLock};
    static DEVICE: OnceLock<burn::backend::wgpu::WgpuDevice> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .expect("No GPU adapter found");
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    required_features: adapter.features(),
                    required_limits: adapter.limits(),
                    ..Default::default()
                },
                None,
            ))
            .expect("Failed to create GPU device");
            let setup = burn::backend::wgpu::WgpuSetup {
                instance: Arc::new(instance),
                adapter: Arc::new(adapter),
                device: Arc::new(device),
                queue: Arc::new(queue),
            };
            burn::backend::wgpu::init_device(setup, Default::default())
        })
        .clone()
}

fn check_gpu_result(actual: f64, expected: f64, label: &str, n: usize) -> bool {
    let tol = if n > 500_000 { 0.05 } else { 0.01 };
    let rel = (actual - expected).abs() / expected.abs().max(1.0);
    let ok = rel <= tol;
    if !ok {
        eprintln!(
            "  VERIFY FAIL [{}]: got {:.6}, expected {:.6} (rel err {:.6})",
            label, actual, expected, rel
        );
    }
    ok
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

fn cpu_vec_add_sum(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x + y) as f64).sum()
}

fn cpu_matmul_sum(a: &[f32], b: &[f32], n: usize) -> f64 {
    let mut total = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += a[i * n + k] as f64 * b[k * n + j] as f64;
            }
            total += s;
        }
    }
    total
}

fn cpu_sum(data: &[f32]) -> f64 {
    data.iter().map(|&x| x as f64).sum()
}

// ---------------------------------------------------------------------------
// Burn GPU implementations
// ---------------------------------------------------------------------------

fn burn_vec_add_gpu(a: &[f32], b: &[f32], device: &burn::backend::wgpu::WgpuDevice) -> f64 {
    use burn::tensor::{Tensor, TensorData};
    let n = a.len();
    let a_t = Tensor::<Gpu, 1>::from_data(TensorData::new(a.to_vec(), [n]), device);
    let b_t = Tensor::<Gpu, 1>::from_data(TensorData::new(b.to_vec(), [n]), device);
    let result = (a_t + b_t).into_data();
    result
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .map(|&x| x as f64)
        .sum()
}

fn burn_matmul_gpu(
    a: &[f32],
    b: &[f32],
    n: usize,
    device: &burn::backend::wgpu::WgpuDevice,
) -> f64 {
    use burn::tensor::{Tensor, TensorData};
    let a_t = Tensor::<Gpu, 2>::from_data(TensorData::new(a.to_vec(), [n, n]), device);
    let b_t = Tensor::<Gpu, 2>::from_data(TensorData::new(b.to_vec(), [n, n]), device);
    let result = a_t.matmul(b_t).into_data();
    result
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .map(|&x| x as f64)
        .sum()
}

fn burn_reduction_gpu(
    data: &[f32],
    num_groups: usize,
    device: &burn::backend::wgpu::WgpuDevice,
) -> f64 {
    use burn::tensor::{Tensor, TensorData};
    let n = data.len();
    let t = Tensor::<Gpu, 1>::from_data(TensorData::new(data.to_vec(), [n]), device);
    // Partial reduction in groups of 64 — same as WGSL shader
    let result = t.reshape([num_groups, 64]).sum_dim(1).into_data();
    result
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .map(|&x| x as f64)
        .sum()
}

// ---------------------------------------------------------------------------
// Payload builders
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Table printer
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let mut results = Vec::new();

    eprintln!("\n=== GPU Benchmarks: Burn(wgpu) vs Base+GPU ===");
    eprintln!("  All: cached GPU device, upload + compute + download");
    eprintln!("  Base+GPU: Cranelift JIT calls GPU runtime via execute_into\n");

    let burn_dev = burn_device();

    // ---- VecAdd ----
    {
        let (config, alg) = load_vecadd_algorithm();
        let mut base_instance = base::Base::new(config).expect("Base::new failed");

        for &n in &[256_000usize, 500_000] {
            eprintln!("  VecAdd {} ...", format_count(n));

            let a = gen_floats(n, 42);
            let b = gen_floats(n, 123);
            let expected = cpu_vec_add_sum(&a, &b);
            let payload = build_f32_payload(&[&a, &b]);
            let out_size = n * 4;
            let mut out_buf = vec![0u8; out_size];

            // Warmup
            std::hint::black_box(burn_vec_add_gpu(&a, &b, &burn_dev));
            let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);

            let burn_ms = harness::median_of(iterations, || {
                let start = std::time::Instant::now();
                std::hint::black_box(burn_vec_add_gpu(&a, &b, &burn_dev));
                start.elapsed().as_secs_f64() * 1000.0
            });

            let base_ms = harness::median_of(iterations, || {
                let start = std::time::Instant::now();
                let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);
                start.elapsed().as_secs_f64() * 1000.0
            });

            let base_sum = f32_sum(&out_buf);
            let verified = check_gpu_result(
                base_sum,
                expected,
                &format!("VecAdd {}", format_count(n)),
                n,
            );

            results.push(BenchResult {
                name: format!("VecAdd {}", format_count(n)),
                col_a_ms: Some(burn_ms),
                col_b_ms: None,
                base_ms,
                verified: Some(verified),
            });
        }
    }

    // ---- MatMul ----
    {
        let (config, alg) = load_matmul_algorithm();
        let mut base_instance = base::Base::new(config).expect("Base::new failed");

        for &n in &[256usize, 512] {
            eprintln!("  MatMul {}x{} ...", n, n);

            let nn = n * n;
            let a: Vec<f32> = (0..nn).map(|i| (i % 100) as f32 / 100.0).collect();
            let b: Vec<f32> = (0..nn).map(|i| ((i + 1) % 100) as f32 / 100.0).collect();
            let expected = cpu_matmul_sum(&a, &b, n);
            let payload = build_f32_payload(&[&a, &b]);
            let out_size = nn * 4;
            let mut out_buf = vec![0u8; out_size];

            // Warmup
            std::hint::black_box(burn_matmul_gpu(&a, &b, n, &burn_dev));
            let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);

            let burn_ms = harness::median_of(iterations, || {
                let start = std::time::Instant::now();
                std::hint::black_box(burn_matmul_gpu(&a, &b, n, &burn_dev));
                start.elapsed().as_secs_f64() * 1000.0
            });

            let base_ms = harness::median_of(iterations, || {
                let start = std::time::Instant::now();
                let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);
                start.elapsed().as_secs_f64() * 1000.0
            });

            let base_sum = f32_sum(&out_buf);
            let verified = check_gpu_result(base_sum, expected, &format!("MatMul {}x{}", n, n), nn);

            results.push(BenchResult {
                name: format!("MatMul {}x{}", n, n),
                col_a_ms: Some(burn_ms),
                col_b_ms: None,
                base_ms,
                verified: Some(verified),
            });
        }
    }

    // ---- Reduction (partial sums, groups of 64) ----
    {
        let (config, alg) = load_reduction_algorithm();
        let mut base_instance = base::Base::new(config).expect("Base::new failed");

        for &n in &[256_000usize, 512_000, 896_000] {
            let num_groups = n / 64;
            eprintln!(
                "  Reduction {} ({} groups) ...",
                format_count(n),
                num_groups
            );

            let data = gen_floats(n, 42);
            let expected = cpu_sum(&data);
            let payload = build_f32_payload(&[&data]);
            let out_size = num_groups * 4;
            let mut out_buf = vec![0u8; out_size];

            // Warmup
            std::hint::black_box(burn_reduction_gpu(&data, num_groups, &burn_dev));
            let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);

            let burn_ms = harness::median_of(iterations, || {
                let start = std::time::Instant::now();
                std::hint::black_box(burn_reduction_gpu(&data, num_groups, &burn_dev));
                start.elapsed().as_secs_f64() * 1000.0
            });

            let base_ms = harness::median_of(iterations, || {
                let start = std::time::Instant::now();
                let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);
                start.elapsed().as_secs_f64() * 1000.0
            });

            // Sum partial sums for verification
            let base_sum = f32_sum(&out_buf);
            let verified = check_gpu_result(
                base_sum,
                expected,
                &format!("Reduction {}", format_count(n)),
                n,
            );

            results.push(BenchResult {
                name: format!("Reduction {}", format_count(n)),
                col_a_ms: Some(burn_ms),
                col_b_ms: None,
                base_ms,
                verified: Some(verified),
            });
        }
    }

    results
}
