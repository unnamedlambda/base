use base_types::{Action, Kind};
use crate::harness;
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

/// Get (or create) the cached Burn GPU device.
fn burn_device() -> burn::backend::wgpu::WgpuDevice {
    use std::sync::{Arc, OnceLock};
    static DEVICE: OnceLock<burn::backend::wgpu::WgpuDevice> = OnceLock::new();
    DEVICE.get_or_init(|| {
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
    }).clone()
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
// Burn GPU implementations
// ---------------------------------------------------------------------------

fn burn_vec_add_gpu(a: &[f32], b: &[f32], device: &burn::backend::wgpu::WgpuDevice, output_path: &str) {
    use burn::tensor::{Tensor, TensorData};
    let n = a.len();
    let a_t = Tensor::<Gpu, 1>::from_data(TensorData::new(a.to_vec(), [n]), device);
    let b_t = Tensor::<Gpu, 1>::from_data(TensorData::new(b.to_vec(), [n]), device);
    let result = (a_t + b_t).into_data();
    std::fs::write(output_path, result.as_bytes()).ok();
}

fn burn_matmul_gpu(a: &[f32], b: &[f32], n: usize, device: &burn::backend::wgpu::WgpuDevice, output_path: &str) {
    use burn::tensor::{Tensor, TensorData};
    let a_t = Tensor::<Gpu, 2>::from_data(TensorData::new(a.to_vec(), [n, n]), device);
    let b_t = Tensor::<Gpu, 2>::from_data(TensorData::new(b.to_vec(), [n, n]), device);
    let result = a_t.matmul(b_t).into_data();
    std::fs::write(output_path, result.as_bytes()).ok();
}

fn burn_reduction_gpu(data: &[f32], num_groups: usize, device: &burn::backend::wgpu::WgpuDevice, output_path: &str) {
    use burn::tensor::{Tensor, TensorData};
    let n = data.len();
    let t = Tensor::<Gpu, 1>::from_data(TensorData::new(data.to_vec(), [n]), device);
    // Partial reduction in groups of 64 — same as CLIF shader
    let result = t.reshape([num_groups, 64]).sum_dim(1).into_data();
    std::fs::write(output_path, result.as_bytes()).ok();
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
// CLIF+GPU+File: Cranelift JIT does GPU pipeline + file write in a single
// compiled function. Generic IR parameterized by memory layout.
// ---------------------------------------------------------------------------

// CLIF memory layout (GPU context pointer at offset 0)
const CLIF_DSIZE_OFF: usize = 0x08;       // i64: GPU buffer size in bytes
const CLIF_WORKGROUPS_OFF: usize = 0x10;   // i64: workgroups for dispatch
const CLIF_FSRC_OFF: usize = 0x18;         // i64: file_write source offset (absolute)
const CLIF_FSIZE_OFF: usize = 0x20;        // i64: file_write size in bytes
const CLIF_BIND_OFF: usize = 0x40;         // binding descriptor [buf_id=0, read_only=0]
const CLIF_SHADER_OFF: usize = 0x100;      // WGSL shader (null-terminated)
const CLIF_FNAME_OFF: usize = 0x3100;      // output filename (null-terminated)
const CLIF_IR_OFF: usize = 0x3800;         // CLIF IR source (null-terminated, ~800 bytes)
const CLIF_DATA_OFF: usize = 0x4000;       // data buffer

pub struct GpuBenchResult {
    pub name: String,
    pub burn_ms: f64,
    pub clif_ms: f64,
    pub verified: bool,
}

/// Generate CLIF IR for GPU dispatch + file write.
fn gen_gpu_clif_ir() -> String {
    format!(
r#"function u0:0(i64) system_v {{
    ; sig0: (ptr) — gpu init/cleanup
    sig0 = (i64) system_v
    ; sig1: (ptr, size) -> buf_id — gpu create_buffer
    sig1 = (i64, i64) -> i32 system_v
    ; sig2: (ptr, shader_off, bind_off, n_bindings) -> pipe_id — gpu create_pipeline
    sig2 = (i64, i64, i64, i32) -> i32 system_v
    ; sig3: (ptr, buf_id, off, size) -> i32 — gpu upload/download
    sig3 = (i64, i32, i64, i64) -> i32 system_v
    ; sig4: (ptr, pipe_id, wg_x, wg_y, wg_z) -> i32 — gpu dispatch
    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v
    ; sig5: (ptr, path_off, src_off, file_offset, size) -> i64 — file write
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_create_pipeline sig2
    fn3 = %cl_gpu_upload sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig3
    fn6 = %cl_gpu_cleanup sig0
    fn7 = %cl_file_write sig5

block0(v0: i64):
    ; --- gpu init ---
    call fn0(v0)

    ; --- create buffer ---
    v1 = load.i64 v0+{dsize_off}
    v2 = call fn1(v0, v1)

    ; --- upload all data ---
    v3 = iconst.i64 {data_off}
    v15 = call fn3(v0, v2, v3, v1)

    ; --- create pipeline (1 binding: buf0 rw) ---
    v4 = iconst.i64 {shader_off}
    v5 = iconst.i64 {bind_off}
    v6 = iconst.i32 1
    v7 = call fn2(v0, v4, v5, v6)

    ; --- dispatch ---
    v8 = load.i64 v0+{wg_off}
    v9 = ireduce.i32 v8
    v16 = iconst.i32 1
    v18 = call fn4(v0, v7, v9, v16, v16)

    ; --- download all data ---
    v17 = call fn5(v0, v2, v3, v1)

    ; --- gpu cleanup ---
    call fn6(v0)

    ; --- file write result region ---
    v10 = iconst.i64 {fname_off}
    v11 = load.i64 v0+{fsrc_off}
    v12 = iconst.i64 0
    v13 = load.i64 v0+{fsize_off}
    v14 = call fn7(v0, v10, v11, v12, v13)

    return
}}"#,
        dsize_off = CLIF_DSIZE_OFF,
        data_off = CLIF_DATA_OFF,
        shader_off = CLIF_SHADER_OFF,
        bind_off = CLIF_BIND_OFF,
        wg_off = CLIF_WORKGROUPS_OFF,
        fname_off = CLIF_FNAME_OFF,
        fsrc_off = CLIF_FSRC_OFF,
        fsize_off = CLIF_FSIZE_OFF,
    )
}

fn build_clif_gpu_algorithm(
    shader: &str,
    data: &[u8],
    buffer_size: usize,
    workgroups: u32,
    result_offset: usize,
    result_size: usize,
    output_path: &str,
) -> (base::BaseConfig, base::Algorithm) {
    let clif_source = gen_gpu_clif_ir();
    let clif_bytes = format!("{}\0", clif_source).into_bytes();
    assert!(clif_bytes.len() < (CLIF_DATA_OFF - CLIF_IR_OFF),
        "CLIF IR too large: {} bytes", clif_bytes.len());

    let payload_size = CLIF_DATA_OFF + data.len();
    let mut payloads = vec![0u8; payload_size];

    payloads[CLIF_IR_OFF..CLIF_IR_OFF + clif_bytes.len()].copy_from_slice(&clif_bytes);

    payloads[CLIF_DSIZE_OFF..CLIF_DSIZE_OFF + 8].copy_from_slice(&(buffer_size as i64).to_le_bytes());
    payloads[CLIF_WORKGROUPS_OFF..CLIF_WORKGROUPS_OFF + 8].copy_from_slice(&(workgroups as i64).to_le_bytes());
    payloads[CLIF_FSRC_OFF..CLIF_FSRC_OFF + 8].copy_from_slice(&((CLIF_DATA_OFF + result_offset) as i64).to_le_bytes());
    payloads[CLIF_FSIZE_OFF..CLIF_FSIZE_OFF + 8].copy_from_slice(&(result_size as i64).to_le_bytes());

    payloads[CLIF_BIND_OFF..CLIF_BIND_OFF + 4].copy_from_slice(&0i32.to_le_bytes());
    payloads[CLIF_BIND_OFF + 4..CLIF_BIND_OFF + 8].copy_from_slice(&0i32.to_le_bytes());

    let shader_bytes = shader.as_bytes();
    payloads[CLIF_SHADER_OFF..CLIF_SHADER_OFF + shader_bytes.len()].copy_from_slice(shader_bytes);
    payloads[CLIF_SHADER_OFF + shader_bytes.len()] = 0;

    let fname = format!("{}\0", output_path);
    payloads[CLIF_FNAME_OFF..CLIF_FNAME_OFF + fname.len()].copy_from_slice(fname.as_bytes());

    payloads[CLIF_DATA_OFF..CLIF_DATA_OFF + data.len()].copy_from_slice(data);

    let _ = std::fs::remove_file(output_path);

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    let config = base::BaseConfig {
        cranelift_ir: clif_source,
        memory_size: payloads.len(),
        context_offset: 0,
    };
    let algorithm = base::Algorithm {
        actions,
        payloads,
        cranelift_units: 0,
        timeout_ms: Some(120_000),
        output: vec![],
    };
    (config, algorithm)
}

// ---------------------------------------------------------------------------
// Table printer
// ---------------------------------------------------------------------------

pub fn print_gpu_table(results: &[GpuBenchResult]) {
    let name_w = 22;
    let col_w = 14;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
        "GPU Benchmark",
        "Burn(wgpu)",
        "CLIF+GPU",
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
            format!("{:.1}ms", r.clif_ms),
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

pub fn run(iterations: usize) -> Vec<GpuBenchResult> {
    let mut results = Vec::new();
    std::fs::create_dir_all("/tmp/gpu-bench-data").ok();

    eprintln!("\n=== GPU Benchmarks: Burn(wgpu) vs CLIF+GPU ===");
    eprintln!("  All: cached GPU device, fresh buffers/pipelines per call");
    eprintln!("  All do: upload + compute + full readback + file write");
    eprintln!("  CLIF+GPU: Cranelift JIT calls extern C GPU + file wrappers\n");

    let burn_dev = burn_device();

    // Warm up GPU devices
    eprintln!("  Warming up GPU devices ...");
    {
        let warmup_a = gen_floats(64, 1);
        let warmup_b = gen_floats(64, 2);
        let buffer_size = 64 * 4 * 2;
        let mut data_bytes = vec![0u8; buffer_size];
        for (i, &v) in warmup_a.iter().enumerate() {
            let off = i * 4;
            data_bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, &v) in warmup_b.iter().enumerate() {
            let off = 64 * 4 + i * 4;
            data_bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let (warmup_config, warmup_alg) = build_clif_gpu_algorithm(WGSL_VEC_ADD, &data_bytes, buffer_size, 1, 0, 64 * 4, "/tmp/gpu-bench-data/_warmup.bin");
        let _ = base::run(warmup_config, warmup_alg);
        let _ = std::fs::remove_file("/tmp/gpu-bench-data/_warmup.bin");
    }
    eprintln!("  GPU devices ready.\n");

    // ---- VecAdd ----
    for &n in &[256_000usize, 500_000] {
        eprintln!("  VecAdd {} ...", format_count(n));

        let a = gen_floats(n, 42);
        let b = gen_floats(n, 123);
        let expected = cpu_vec_add_sum(&a, &b);

        let label = format!("vecadd_{}", format_count(n));
        let burn_out = format!("/tmp/gpu-bench-data/{}_burn.bin", label);
        let clif_out = format!("/tmp/gpu-bench-data/{}_clif.bin", label);

        // Prepare data bytes for CLIF
        let buffer_size = n * 4 * 2;
        let mut data_bytes = vec![0u8; buffer_size];
        for (i, &v) in a.iter().enumerate() {
            let off = i * 4;
            data_bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, &v) in b.iter().enumerate() {
            let off = n * 4 + i * 4;
            data_bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let workgroups = ((n + 63) / 64) as u32;

        let burn_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            burn_vec_add_gpu(&a, &b, &burn_dev, &burn_out);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let (clif_config, clif_alg) = build_clif_gpu_algorithm(WGSL_VEC_ADD, &data_bytes, buffer_size, workgroups, 0, n * 4, &clif_out);
        let clif_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let cfg = clif_config.clone();
            let alg = clif_alg.clone();
            let _ = base::run(cfg, alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let burn_ok = read_f32_file_sum(&burn_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] Burn: could not read output", label); false },
            |v| check_gpu_result(v, expected, "Burn", &label, n),
        );
        let clif_ok = read_f32_file_sum(&clif_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] CLIF: could not read output", label); false },
            |v| check_gpu_result(v, expected, "CLIF", &label, n),
        );

        results.push(GpuBenchResult {
            name: format!("VecAdd {}", format_count(n)),
            burn_ms,
            clif_ms,
            verified: burn_ok && clif_ok,
        });

    }

    // ---- MatMul ----
    for &n in &[256usize, 512] {
        eprintln!("  MatMul {}x{} ...", n, n);

        let a: Vec<f32> = (0..n * n).map(|i| (i % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i + 1) % 100) as f32 / 100.0).collect();
        let expected = cpu_matmul_sum(&a, &b, n);

        let nn = n * n;
        let label = format!("matmul_{}", n);
        let burn_out = format!("/tmp/gpu-bench-data/{}_burn.bin", label);
        let clif_out = format!("/tmp/gpu-bench-data/{}_clif.bin", label);

        let shader = wgsl_matmul(n);

        // Prepare data bytes for CLIF
        let buffer_size = nn * 4 * 3;
        let mut data_bytes = vec![0u8; buffer_size];
        for (i, &v) in a.iter().enumerate() {
            let off = i * 4;
            data_bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, &v) in b.iter().enumerate() {
            let off = nn * 4 + i * 4;
            data_bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let workgroups = ((nn + 63) / 64) as u32;

        let burn_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            burn_matmul_gpu(&a, &b, n, &burn_dev, &burn_out);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let (clif_config, clif_alg) = build_clif_gpu_algorithm(&shader, &data_bytes, buffer_size, workgroups, 2 * nn * 4, nn * 4, &clif_out);
        let clif_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let cfg = clif_config.clone();
            let alg = clif_alg.clone();
            let _ = base::run(cfg, alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let burn_ok = read_f32_file_sum(&burn_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] Burn: could not read output", label); false },
            |v| check_gpu_result(v, expected, "Burn", &label, nn),
        );
        let clif_ok = read_f32_file_sum(&clif_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] CLIF: could not read output", label); false },
            |v| check_gpu_result(v, expected, "CLIF", &label, nn),
        );

        results.push(GpuBenchResult {
            name: format!("MatMul {}x{}", n, n),
            burn_ms,
            clif_ms,
            verified: burn_ok && clif_ok,
        });

    }

    // ---- Reduction (partial sums, groups of 64) ----
    for &n in &[256_000usize, 512_000, 896_000] {
        let num_groups = n / 64;
        eprintln!("  Reduction {} ({} groups) ...", format_count(n), num_groups);

        let data = gen_floats(n, 42);
        let expected = cpu_sum(&data);

        let label = format!("reduction_{}", format_count(n));
        let burn_out = format!("/tmp/gpu-bench-data/{}_burn.bin", label);
        let clif_out = format!("/tmp/gpu-bench-data/{}_clif.bin", label);

        let shader = wgsl_reduction(n);

        // Prepare data bytes for CLIF
        let buffer_elems = n + num_groups;
        let buffer_size = buffer_elems * 4;
        let mut data_bytes = vec![0u8; buffer_size];
        for (i, &v) in data.iter().enumerate() {
            let off = i * 4;
            data_bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let workgroups = num_groups as u32;

        let burn_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            burn_reduction_gpu(&data, num_groups, &burn_dev, &burn_out);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let (clif_config, clif_alg) = build_clif_gpu_algorithm(&shader, &data_bytes, buffer_size, workgroups, n * 4, num_groups * 4, &clif_out);
        let clif_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let cfg = clif_config.clone();
            let alg = clif_alg.clone();
            let _ = base::run(cfg, alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let burn_ok = read_f32_file_sum(&burn_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] Burn: could not read output", label); false },
            |v| check_gpu_result(v, expected, "Burn", &label, n),
        );
        let clif_ok = read_f32_file_sum(&clif_out).map_or_else(
            || { eprintln!("  VERIFY FAIL [{}] CLIF: could not read output", label); false },
            |v| check_gpu_result(v, expected, "CLIF", &label, n),
        );

        results.push(GpuBenchResult {
            name: format!("Reduction {}", format_count(n)),
            burn_ms,
            clif_ms,
            verified: burn_ok && clif_ok,
        });

    }

    results
}
