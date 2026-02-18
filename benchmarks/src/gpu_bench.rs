use base_types::{Action, Kind, State, UnitSpec};
use crate::harness::{self, BenchResult};

type Gpu = burn::backend::wgpu::Wgpu;

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

/// Create a fresh wgpu GPU device for Burn on every call.
/// Matches Base's execute() which spawns a new GpuUnit (fresh wgpu device) each time.
fn burn_fresh_device() -> burn::backend::wgpu::WgpuDevice {
    use std::sync::Arc;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
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
}

/// Read f32 values from a binary file and sum them as f64.
fn read_f32_file_sum(path: &str) -> Option<f64> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 4 || data.len() % 4 != 0 {
        return None;
    }
    let mut total = 0.0f64;
    for chunk in data.chunks_exact(4) {
        let v = f32::from_le_bytes(chunk.try_into().unwrap());
        total += v as f64;
    }
    Some(total)
}

fn check_gpu_result(actual: f64, expected: f64, impl_name: &str, label: &str, n: usize) -> bool {
    let tol = if n > 500_000 { 0.05 } else { 0.01 };
    let rel = (actual - expected).abs() / expected.abs().max(1.0);
    let ok = rel <= tol;
    if !ok {
        eprintln!(
            "  VERIFY FAIL [{}] {}: got {:.6}, expected {:.6} (rel err {:.6})",
            label, impl_name, actual, expected, rel
        );
    }
    ok
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

fn cpu_vec_add_sum(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x + y) as f64)
        .sum()
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
// Burn GPU implementations (idiomatic)
//
// Each: fresh GPU init → upload → GPU compute → scalar readback.
// ---------------------------------------------------------------------------

fn burn_vec_add_gpu(a: &[f32], b: &[f32]) -> f64 {
    use burn::tensor::{Tensor, TensorData};
    let device = burn_fresh_device();
    let n = a.len();
    let a_t = Tensor::<Gpu, 1>::from_data(TensorData::new(a.to_vec(), [n]), &device);
    let b_t = Tensor::<Gpu, 1>::from_data(TensorData::new(b.to_vec(), [n]), &device);
    (a_t + b_t).sum().into_scalar() as f64
}

fn burn_matmul_gpu(a: &[f32], b: &[f32], n: usize) -> f64 {
    use burn::tensor::{Tensor, TensorData};
    let device = burn_fresh_device();
    let a_t = Tensor::<Gpu, 2>::from_data(TensorData::new(a.to_vec(), [n, n]), &device);
    let b_t = Tensor::<Gpu, 2>::from_data(TensorData::new(b.to_vec(), [n, n]), &device);
    a_t.matmul(b_t).sum().into_scalar() as f64
}

fn burn_reduction_gpu(data: &[f32]) -> f64 {
    use burn::tensor::{Tensor, TensorData};
    let device = burn_fresh_device();
    let n = data.len();
    let t = Tensor::<Gpu, 1>::from_data(TensorData::new(data.to_vec(), [n]), &device);
    t.sum().into_scalar() as f64
}

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------

const WGSL_VEC_ADD: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&data) / 2u;
    let i = gid.x;
    if (i >= n) { return; }
    data[i] = data[i] + data[n + i];
}
"#;

fn wgsl_matmul(n: usize) -> String {
    format!(
        r#"
const N: u32 = {n}u;
const NN: u32 = {nn}u;

@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= NN) {{ return; }}
    let i = idx / N;
    let j = idx % N;
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < N; k = k + 1u) {{
        sum = sum + data[i * N + k] * data[NN + k * N + j];
    }}
    data[2u * NN + idx] = sum;
}}
"#,
        n = n,
        nn = n * n
    )
}

/// Reduction: each workgroup of 64 threads sums 64 elements via shared memory.
/// Partial sums written to separate output region data[INPUT_N + wid.x].
fn wgsl_reduction(input_n: usize) -> String {
    let num_groups = input_n / 64;
    format!(
        r#"
const INPUT_N: u32 = {input_n}u;
const NUM_GROUPS: u32 = {num_groups}u;

@group(0) @binding(0) var<storage, read_write> data: array<f32>;

var<workgroup> sdata: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {{
    if (gid.x < INPUT_N) {{
        sdata[lid.x] = data[gid.x];
    }} else {{
        sdata[lid.x] = 0.0;
    }}
    workgroupBarrier();

    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{
            sdata[lid.x] = sdata[lid.x] + sdata[lid.x + s];
        }}
        workgroupBarrier();
    }}

    if (lid.x == 0u && wid.x < NUM_GROUPS) {{
        data[INPUT_N + wid.x] = sdata[0];
    }}
}}
"#,
        input_n = input_n,
        num_groups = num_groups
    )
}

// ---------------------------------------------------------------------------
// Memory layout for Base GPU algorithms
// ---------------------------------------------------------------------------

const SHADER_OFF: usize = 0x0000;
const FLAG1_ADDR: usize = 0x3000; // GPU completion flag
const FLAG2_ADDR: usize = 0x3008; // File completion flag
const FNAME_OFF: usize = 0x3100;  // Output filename (null-terminated)
const DATA_OFF: usize = 0x4000;   // Data buffer start

/// Build a Base algorithm: GPU dispatch + FileWrite.
///
/// Actions:
///   [0] Dispatch       — GPU compute (buffer at DATA_OFF)
///   [1] FileWrite      — write result region to file
///   [2] AsyncDispatch  → GPU (action 0), flag at FLAG1
///   [3] Wait(FLAG1)
///   [4] AsyncDispatch  → File (action 1), flag at FLAG2
///   [5] Wait(FLAG2)
fn build_gpu_algorithm(
    shader: &str,
    data: &[u8],
    buffer_size: usize,
    result_offset: usize,
    result_size: usize,
    output_path: &str,
) -> base::Algorithm {
    let payload_size = DATA_OFF + data.len();
    let mut payloads = vec![0u8; payload_size];

    let shader_bytes = shader.as_bytes();
    payloads[SHADER_OFF..SHADER_OFF + shader_bytes.len()].copy_from_slice(shader_bytes);

    let fname = format!("{}\0", output_path);
    let fname_bytes = fname.as_bytes();
    payloads[FNAME_OFF..FNAME_OFF + fname_bytes.len()].copy_from_slice(fname_bytes);

    payloads[DATA_OFF..DATA_OFF + data.len()].copy_from_slice(data);

    let _ = std::fs::remove_file(output_path);

    let actions = vec![
        // [0] GPU Dispatch
        Action {
            kind: Kind::Dispatch,
            src: DATA_OFF as u32,
            dst: DATA_OFF as u32,
            offset: 0,
            size: buffer_size as u32,
        },
        // [1] FileWrite: result region → file
        Action {
            kind: Kind::FileWrite,
            src: (DATA_OFF + result_offset) as u32,
            dst: FNAME_OFF as u32,
            offset: 0,
            size: result_size as u32,
        },
        // [2] AsyncDispatch → GPU (action 0)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 0,
            src: 0,
            offset: FLAG1_ADDR as u32,
            size: 0,
        },
        // [3] Wait GPU
        Action {
            kind: Kind::Wait,
            dst: FLAG1_ADDR as u32,
            src: 0,
            offset: 0,
            size: 0,
        },
        // [4] AsyncDispatch → File (action 1)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 1,
            offset: FLAG2_ADDR as u32,
            size: 0,
        },
        // [5] Wait File
        Action {
            kind: Kind::Wait,
            dst: FLAG2_ADDR as u32,
            src: 0,
            offset: 0,
            size: 0,
        },
    ];

    let num_actions = actions.len();

    base::Algorithm {
        actions,
        payloads,
        state: State {
            regs_per_unit: 0,
            gpu_size: buffer_size,
            computational_regs: 0,
            file_buffer_size: 65536,
            gpu_shader_offsets: vec![SHADER_OFF],
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            simd_units: 0,
            gpu_units: 1,
            computational_units: 0,
            file_units: 1,
            network_units: 0,
            memory_units: 0,
            ffi_units: 0,
            hash_table_units: 0,
            lmdb_units: 0,
            cranelift_units: 0,
            backends_bits: 0xFFFF_FFFF,
        },
        simd_assignments: vec![],
        computational_assignments: vec![],
        memory_assignments: vec![],
        file_assignments: vec![],
        network_assignments: vec![],
        ffi_assignments: vec![],
        hash_table_assignments: vec![],
        lmdb_assignments: vec![],
        gpu_assignments: vec![0; num_actions],
        cranelift_assignments: vec![],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(256 * 1024),
        timeout_ms: Some(120_000),
        thread_name_prefix: Some("gpu-bench".into()),
    }
}

// ---------------------------------------------------------------------------
// Base GPU algorithm builders
// ---------------------------------------------------------------------------

fn build_gpu_vec_add(a: &[f32], b: &[f32], output_path: &str) -> base::Algorithm {
    let n = a.len();
    let buffer_size = n * 4 * 2; // [a | b]
    let mut data = vec![0u8; buffer_size];
    for (i, &v) in a.iter().enumerate() {
        let off = i * 4;
        data[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, &v) in b.iter().enumerate() {
        let off = n * 4 + i * 4;
        data[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    // Result: first N f32s (a[i]+b[i]) at offset 0
    build_gpu_algorithm(WGSL_VEC_ADD, &data, buffer_size, 0, n * 4, output_path)
}

fn build_gpu_matmul(a: &[f32], b: &[f32], n: usize, output_path: &str) -> base::Algorithm {
    let nn = n * n;
    let buffer_size = nn * 4 * 3; // [A | B | C]
    let mut data = vec![0u8; buffer_size];
    for (i, &v) in a.iter().enumerate() {
        let off = i * 4;
        data[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, &v) in b.iter().enumerate() {
        let off = nn * 4 + i * 4;
        data[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    let shader = wgsl_matmul(n);
    // Result: C at offset 2*NN*4, size NN*4
    build_gpu_algorithm(&shader, &data, buffer_size, 2 * nn * 4, nn * 4, output_path)
}

fn build_gpu_reduction(data_f32: &[f32], output_path: &str) -> base::Algorithm {
    let n = data_f32.len();
    let num_groups = n / 64;
    let buffer_elems = n + num_groups;
    let buffer_size = buffer_elems * 4;
    let mut data = vec![0u8; buffer_size];
    for (i, &v) in data_f32.iter().enumerate() {
        let off = i * 4;
        data[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    let shader = wgsl_reduction(n);
    // Result: partial sums at offset N*4, size num_groups*4
    build_gpu_algorithm(&shader, &data, buffer_size, n * 4, num_groups * 4, output_path)
}

// ---------------------------------------------------------------------------
// Table printer
// ---------------------------------------------------------------------------

pub fn print_gpu_table(results: &[BenchResult]) {
    let name_w = 22;
    let col_w = 14;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
        "GPU Benchmark",
        "Burn(wgpu)",
        "Base(GPU)",
        "Check",
        name_w = name_w,
        col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 2 + 6 + 3));

    for r in results {
        let burn_str = match r.python_ms {
            Some(ms) => format!("{:.1}ms", ms),
            None => "N/A".to_string(),
        };
        let base_str = if r.base_ms.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.1}ms", r.base_ms)
        };
        let check_str = match r.verified {
            Some(true) => "\u{2713}",
            Some(false) => "\u{2717}",
            None => "\u{2014}",
        };

        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
            r.name,
            burn_str,
            base_str,
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

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let mut results = Vec::new();
    std::fs::create_dir_all("/tmp/gpu-bench-data").ok();

    eprintln!("\n=== GPU Benchmarks: Burn(wgpu) vs Base(GPU dispatch) ===");
    eprintln!("  Both: fresh GPU init every iteration, GPU does the compute\n");

    // ---- VecAdd ----
    for &n in &[256_000usize, 500_000] {
        eprintln!("  VecAdd {} ...", format_count(n));

        let a = gen_floats(n, 42);
        let b = gen_floats(n, 123);
        let expected = cpu_vec_add_sum(&a, &b);

        let label = format!("vecadd_{}", format_count(n));
        let base_out = format!("/tmp/gpu-bench-data/{}_base.bin", label);

        let base_alg = build_gpu_vec_add(&a, &b, &base_out);

        let mut burn_result = 0.0f64;
        let burn_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            burn_result = burn_vec_add_gpu(&a, &b);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let base_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let alg = base_alg.clone();
            let _ = base::execute(alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let burn_ok = check_gpu_result(burn_result, expected, "Burn", &label, n);
        let base_ok = read_f32_file_sum(&base_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] Base: could not read output", label); false },
            |v| check_gpu_result(v, expected, "Base", &label, n),
        );

        results.push(BenchResult {
            name: format!("VecAdd {}", format_count(n)),
            python_ms: Some(burn_ms),
            rust_ms: None,
            base_ms,
            verified: Some(burn_ok && base_ok),
            actions: None,
        });
    }

    // ---- MatMul ----
    for &n in &[256usize, 512] {
        eprintln!("  MatMul {}x{} ...", n, n);

        let a: Vec<f32> = (0..n * n).map(|i| (i % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i + 1) % 100) as f32 / 100.0).collect();
        let expected = cpu_matmul_sum(&a, &b, n);

        let label = format!("matmul_{}", n);
        let base_out = format!("/tmp/gpu-bench-data/{}_base.bin", label);

        let base_alg = build_gpu_matmul(&a, &b, n, &base_out);

        let mut burn_result = 0.0f64;
        let burn_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            burn_result = burn_matmul_gpu(&a, &b, n);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let base_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let alg = base_alg.clone();
            let _ = base::execute(alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let burn_ok = check_gpu_result(burn_result, expected, "Burn", &label, n * n);
        let base_ok = read_f32_file_sum(&base_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] Base: could not read output", label); false },
            |v| check_gpu_result(v, expected, "Base", &label, n * n),
        );

        results.push(BenchResult {
            name: format!("MatMul {}x{}", n, n),
            python_ms: Some(burn_ms),
            rust_ms: None,
            base_ms,
            verified: Some(burn_ok && base_ok),
            actions: None,
        });
    }

    // ---- Reduction (partial sums, groups of 64) ----
    // Sizes must be multiples of 64 so both sides partition identically.
    for &n in &[256_000usize, 512_000, 896_000] {
        let num_groups = n / 64;
        eprintln!("  Reduction {} ({} groups) ...", format_count(n), num_groups);

        let data = gen_floats(n, 42);
        let expected = cpu_sum(&data);

        let label = format!("reduction_{}", format_count(n));
        let base_out = format!("/tmp/gpu-bench-data/{}_base.bin", label);

        let base_alg = build_gpu_reduction(&data, &base_out);

        let mut burn_result = 0.0f64;
        let burn_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            burn_result = burn_reduction_gpu(&data);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let base_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let alg = base_alg.clone();
            let _ = base::execute(alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let burn_ok = check_gpu_result(burn_result, expected, "Burn", &label, n);
        let base_ok = read_f32_file_sum(&base_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] Base: could not read output", label); false },
            |v| check_gpu_result(v, expected, "Base", &label, n),
        );

        results.push(BenchResult {
            name: format!("Reduction {}", format_count(n)),
            python_ms: Some(burn_ms),
            rust_ms: None,
            base_ms,
            verified: Some(burn_ok && base_ok),
            actions: None,
        });
    }

    results
}
