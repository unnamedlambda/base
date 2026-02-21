use base_types::{Action, Kind, State, UnitSpec};
use crate::harness;

// ---------------------------------------------------------------------------
// Iterative GPU Benchmark: apply a trivial kernel N times to the same data.
//
// Compares approaches for keeping data GPU-resident across iterations.
//
// Raw wgpu:  Keeps buffer GPU-resident, only reads back at end
// Burn:      Naturally keeps tensors GPU-resident
// CLIF+GPU:  1 upload + N kernels + 1 download (via extern "C" wrappers)
// ---------------------------------------------------------------------------

type Gpu = burn::backend::wgpu::Wgpu;

pub struct GpuIterResult {
    pub name: String,
    pub wgpu_ms: f64,
    pub burn_ms: f64,
    pub clif_ms: f64,
    pub verified: bool,
}

// Partial sum reduction: each workgroup of 64 reduces 64 elements → 1 partial sum.
const WGSL_REDUCE: &str = r#"
@group(0) @binding(0) var<storage, read>       data: array<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)   lid: vec3<u32>,
        @builtin(workgroup_id)          wgid: vec3<u32>) {
    let n = arrayLength(&data);
    partial[lid.x] = select(0.0, data[gid.x], gid.x < n);
    workgroupBarrier();
    var s = 32u;
    while s > 0u {
        if lid.x < s { partial[lid.x] += partial[lid.x + s]; }
        workgroupBarrier();
        s >>= 1u;
    }
    if lid.x == 0u { sums[wgid.x] = partial[0]; }
}
"#;

const WGSL_SCALE: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&data);
    let i = gid.x;
    if (i >= n) { return; }
    data[i] = data[i] * 1.001;
}
"#;

fn gen_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 33) as i32;
        out.push((bits as f32 / i32::MAX as f32).abs() + 0.1);
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

// ---------------------------------------------------------------------------
// CPU reference: apply *1.001 N times
// ---------------------------------------------------------------------------

fn cpu_expected_sum(data: &[f32], passes: usize) -> f64 {
    let mut vals: Vec<f64> = data.iter().map(|&x| x as f64).collect();
    for _ in 0..passes {
        for v in vals.iter_mut() {
            *v *= 1.001;
        }
    }
    vals.iter().sum()
}

// ---------------------------------------------------------------------------
// Cached devices for fair comparison (all approaches reuse one device)
// ---------------------------------------------------------------------------

fn cached_burn_device() -> burn::backend::wgpu::WgpuDevice {
    use std::sync::{Arc, OnceLock};
    static DEVICE: OnceLock<burn::backend::wgpu::WgpuDevice> = OnceLock::new();
    DEVICE.get_or_init(|| {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("No GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                ..Default::default()
            },
            None,
        ))
        .expect("Failed to create device");
        let setup = burn::backend::wgpu::WgpuSetup {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        };
        burn::backend::wgpu::init_device(setup, Default::default())
    }).clone()
}

fn cached_wgpu_device() -> (std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>) {
    use std::sync::{Arc, OnceLock};
    static GPU: OnceLock<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> = OnceLock::new();
    let (d, q) = GPU.get_or_init(|| {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("No GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
            None,
        ))
        .expect("Failed to create device");
        std::mem::forget(instance);
        std::mem::forget(adapter);
        (Arc::new(device), Arc::new(queue))
    });
    (d.clone(), q.clone())
}

// ---------------------------------------------------------------------------
// Burn: naturally keeps tensor GPU-resident across operations
// ---------------------------------------------------------------------------

fn burn_iterative(data: &[f32], passes: usize) -> f64 {
    use burn::tensor::{Tensor, TensorData};

    let dev = cached_burn_device();

    let n = data.len();
    let mut t = Tensor::<Gpu, 1>::from_data(TensorData::new(data.to_vec(), [n]), &dev);
    let scale = Tensor::<Gpu, 1>::from_data(TensorData::new(vec![1.001f32], [1]), &dev);

    for _ in 0..passes {
        t = t * scale.clone();
    }

    t.sum().into_scalar() as f64
}

// ---------------------------------------------------------------------------
// Raw wgpu: keeps buffer GPU-resident, only reads back at end
// ---------------------------------------------------------------------------

fn wgpu_iterative(data: &[f32], passes: usize) -> f64 {
    use wgpu::*;

    let (device, queue) = cached_wgpu_device();

    let n = data.len();
    let byte_size = (n * 4) as u64;

    let compute_buf = device.create_buffer(&BufferDescriptor {
        label: Some("Compute"),
        size: byte_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Scale"),
        source: ShaderSource::Wgsl(WGSL_SCALE.into()),
    });

    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: compute_buf.as_entire_binding(),
        }],
    });

    // Upload once
    let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    queue.write_buffer(&compute_buf, 0, &data_bytes);

    // N dispatches — data stays on GPU
    let workgroups = ((n as u32) + 63) / 64;
    for _ in 0..passes {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    // GPU reduction: 1M → ~15K partial sums
    let reduce_wgs = ((n as u32) + 63) / 64;
    let sums_bytes = (reduce_wgs as u64) * 4;

    let sums_buf = device.create_buffer(&BufferDescriptor {
        label: Some("Sums"), size: sums_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let reduce_staging = device.create_buffer(&BufferDescriptor {
        label: Some("ReduceStaging"), size: sums_bytes,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false,
    });

    let reduce_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Reduce"), source: ShaderSource::Wgsl(WGSL_REDUCE.into()),
    });
    let reduce_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let reduce_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&reduce_bgl], push_constant_ranges: &[],
        })),
        module: &reduce_shader, entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(), cache: None,
    });
    let reduce_bg = device.create_bind_group(&BindGroupDescriptor {
        label: None, layout: &reduce_bgl,
        entries: &[
            BindGroupEntry { binding: 0, resource: compute_buf.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: sums_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    { let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });
      pass.set_pipeline(&reduce_pipeline); pass.set_bind_group(0, &reduce_bg, &[]);
      pass.dispatch_workgroups(reduce_wgs, 1, 1); }
    encoder.copy_buffer_to_buffer(&sums_buf, 0, &reduce_staging, 0, sums_bytes);
    queue.submit(Some(encoder.finish()));

    let slice = reduce_staging.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    device.poll(Maintain::Wait);

    let mapped = slice.get_mapped_range();
    let sum: f64 = mapped.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
        .sum();
    drop(mapped);
    reduce_staging.unmap();

    sum
}

// ---------------------------------------------------------------------------
// CLIF+GPU: Cranelift JIT calls generic extern "C" GPU primitives.
// Data stays GPU-resident between dispatches.
// ---------------------------------------------------------------------------

const CLIF_PASSES_OFF: usize    = 0x08;
const CLIF_DSIZE_OFF: usize     = 0x10;
const CLIF_WORKGROUPS_OFF: usize = 0x18;
const CLIF_SUMSSIZE_OFF: usize  = 0x20;
const CLIF_SCALE_BIND_OFF: usize = 0x40;
const CLIF_REDUCE_BIND_OFF: usize = 0x48;
const CLIF_SCALE_SHADER_OFF: usize = 0x100;
const CLIF_REDUCE_SHADER_OFF: usize = 0x800;
const CLIF_FNAME_OFF: usize     = 0xC00;
const CLIF_FLAG_OFF: usize      = 0xD00;
const CLIF_IR_OFF: usize        = 0x1000;
const CLIF_DATA_OFF: usize      = 0x2000;

fn gen_gpu_clif_ir() -> String {
    format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i64, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i64) -> i32 system_v
    sig4 = (i64, i32, i32) -> i32 system_v
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
    call fn0(v0)

    v1 = load.i64 v0+{dsize_off}
    v2 = call fn1(v0, v1)

    v3 = load.i64 v0+{sums_off}
    v4 = call fn1(v0, v3)

    v5 = iconst.i64 {data_off}
    v22 = call fn3(v0, v2, v5, v1)

    v6 = iconst.i64 {scale_shader_off}
    v7 = iconst.i64 {scale_bind_off}
    v8 = iconst.i32 1
    v9 = call fn2(v0, v6, v7, v8)

    v10 = iconst.i64 {reduce_shader_off}
    v11 = iconst.i64 {reduce_bind_off}
    v12 = iconst.i32 2
    v13 = call fn2(v0, v10, v11, v12)

    v14 = load.i64 v0+{passes_off}
    v15 = load.i64 v0+{wg_off}
    v16 = ireduce.i32 v15
    v17 = iconst.i64 0
    jump block1(v17)

block1(v18: i64):
    v19 = icmp uge v18, v14
    brif v19, block2, block3(v18)

block3(v20: i64):
    v23 = call fn4(v0, v9, v16)
    v21 = iadd_imm v20, 1
    jump block1(v21)

block2:
    v24 = call fn4(v0, v13, v16)

    v25 = call fn5(v0, v4, v5, v3)

    call fn6(v0)

    v26 = iconst.i64 {fname_off}
    v27 = iconst.i64 0
    v28 = call fn7(v0, v26, v5, v27, v3)

    return
}}"#,
        dsize_off = CLIF_DSIZE_OFF,
        sums_off = CLIF_SUMSSIZE_OFF,
        data_off = CLIF_DATA_OFF,
        scale_shader_off = CLIF_SCALE_SHADER_OFF,
        scale_bind_off = CLIF_SCALE_BIND_OFF,
        reduce_shader_off = CLIF_REDUCE_SHADER_OFF,
        reduce_bind_off = CLIF_REDUCE_BIND_OFF,
        passes_off = CLIF_PASSES_OFF,
        wg_off = CLIF_WORKGROUPS_OFF,
        fname_off = CLIF_FNAME_OFF,
    )
}

fn build_clif_gpu_iter_algorithm(data: &[f32], passes: usize, output_path: &str) -> base::Algorithm {
    let n = data.len();
    let data_size = n * 4;
    let workgroups = ((n as u32) + 63) / 64;
    let sums_size = (workgroups as usize) * 4;

    let clif_source = gen_gpu_clif_ir();
    let clif_bytes = format!("{}\0", clif_source).into_bytes();
    assert!(clif_bytes.len() < (CLIF_DATA_OFF - CLIF_IR_OFF),
        "CLIF IR too large: {} bytes", clif_bytes.len());

    let payload_size = CLIF_DATA_OFF + data_size.max(sums_size);
    let mut payloads = vec![0u8; payload_size];

    payloads[CLIF_IR_OFF..CLIF_IR_OFF + clif_bytes.len()].copy_from_slice(&clif_bytes);

    payloads[CLIF_PASSES_OFF..CLIF_PASSES_OFF + 8].copy_from_slice(&(passes as i64).to_le_bytes());
    payloads[CLIF_DSIZE_OFF..CLIF_DSIZE_OFF + 8].copy_from_slice(&(data_size as i64).to_le_bytes());
    payloads[CLIF_WORKGROUPS_OFF..CLIF_WORKGROUPS_OFF + 8].copy_from_slice(&(workgroups as i64).to_le_bytes());
    payloads[CLIF_SUMSSIZE_OFF..CLIF_SUMSSIZE_OFF + 8].copy_from_slice(&(sums_size as i64).to_le_bytes());

    // Scale: [buf0, rw]
    payloads[CLIF_SCALE_BIND_OFF..CLIF_SCALE_BIND_OFF + 4].copy_from_slice(&0i32.to_le_bytes());
    payloads[CLIF_SCALE_BIND_OFF + 4..CLIF_SCALE_BIND_OFF + 8].copy_from_slice(&0i32.to_le_bytes());
    // Reduce: [buf0, ro], [buf1, rw]
    payloads[CLIF_REDUCE_BIND_OFF..CLIF_REDUCE_BIND_OFF + 4].copy_from_slice(&0i32.to_le_bytes());
    payloads[CLIF_REDUCE_BIND_OFF + 4..CLIF_REDUCE_BIND_OFF + 8].copy_from_slice(&1i32.to_le_bytes());
    payloads[CLIF_REDUCE_BIND_OFF + 8..CLIF_REDUCE_BIND_OFF + 12].copy_from_slice(&1i32.to_le_bytes());
    payloads[CLIF_REDUCE_BIND_OFF + 12..CLIF_REDUCE_BIND_OFF + 16].copy_from_slice(&0i32.to_le_bytes());

    let scale_bytes = WGSL_SCALE.as_bytes();
    payloads[CLIF_SCALE_SHADER_OFF..CLIF_SCALE_SHADER_OFF + scale_bytes.len()].copy_from_slice(scale_bytes);
    payloads[CLIF_SCALE_SHADER_OFF + scale_bytes.len()] = 0;

    let reduce_bytes = WGSL_REDUCE.as_bytes();
    payloads[CLIF_REDUCE_SHADER_OFF..CLIF_REDUCE_SHADER_OFF + reduce_bytes.len()].copy_from_slice(reduce_bytes);
    payloads[CLIF_REDUCE_SHADER_OFF + reduce_bytes.len()] = 0;

    let fname = format!("{}\0", output_path);
    payloads[CLIF_FNAME_OFF..CLIF_FNAME_OFF + fname.len()].copy_from_slice(fname.as_bytes());

    for (i, &v) in data.iter().enumerate() {
        let off = CLIF_DATA_OFF + i * 4;
        payloads[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    let _ = std::fs::remove_file(output_path);

    let actions = vec![
        Action { kind: Kind::FileRead, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 0, offset: CLIF_FLAG_OFF as u32, size: 0 },
        Action { kind: Kind::Wait, dst: CLIF_FLAG_OFF as u32, src: 0, offset: 0, size: 0 },
    ];
    let num_actions = actions.len();

    base::Algorithm {
        actions,
        payloads,
        state: State {
            file_buffer_size: 0,
            cranelift_ir_offsets: vec![CLIF_IR_OFF],
        },
        units: UnitSpec {
            file_units: 0,
            cranelift_units: 1,
        },
        file_assignments: vec![],
        cranelift_assignments: vec![0; num_actions],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(256 * 1024),
        timeout_ms: Some(300_000),
        thread_name_prefix: Some("clif-gpu-iter".into()),
    }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn check_result(actual: f64, expected: f64, label: &str, impl_name: &str) -> bool {
    let rel = (actual - expected).abs() / expected.abs().max(1.0);
    let ok = rel <= 0.05;
    if !ok {
        eprintln!(
            "  VERIFY FAIL [{}] {}: got {:.6}, expected {:.6} (rel err {:.6})",
            label, impl_name, actual, expected, rel
        );
    }
    ok
}

// ---------------------------------------------------------------------------
// Table printer
// ---------------------------------------------------------------------------

pub fn print_iter_table(results: &[GpuIterResult]) {
    let name_w = 26;
    let col_w = 14;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>6}",
        "GPU Iterative",
        "Raw wgpu",
        "Burn",
        "CLIF+GPU",
        "Check",
        name_w = name_w,
        col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 3 + 6 + 4));

    for r in results {
        let check_str = if r.verified { "    \u{2713}" } else { "    \u{2717}" };
        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {}",
            r.name,
            format!("{:.1}ms", r.wgpu_ms),
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

pub fn run(iterations: usize) -> Vec<GpuIterResult> {
    let mut results = Vec::new();

    eprintln!("\n=== GPU Iterative Benchmark: N kernel passes on same data ===");
    eprintln!("  Raw wgpu & Burn: data stays GPU-resident between passes");
    eprintln!("  CLIF+GPU:        CLIF JIT calls extern C GPU wrappers (GPU-resident)\n");

    let n = 1_000_000usize;

    for &passes in &[1, 10, 100, 500, 1000] {
        let label = format!("{}x scale {}", passes, format_count(n));
        eprintln!("  {} ...", label);

        let data = gen_floats(n, 42);
        let expected = cpu_expected_sum(&data, passes);

        // Raw wgpu (GPU-resident)
        let mut wgpu_result = 0.0f64;
        let wgpu_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            wgpu_result = wgpu_iterative(&data, passes);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Burn (GPU-resident)
        let mut burn_result = 0.0f64;
        let burn_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            burn_result = burn_iterative(&data, passes);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // CLIF+GPU (GPU-resident via extern "C" wrappers, through execute())
        let clif_out = format!("/tmp/gpu-iter-clif-{}.bin", passes);
        let clif_alg = build_clif_gpu_iter_algorithm(&data, passes, &clif_out);
        let clif_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let alg = clif_alg.clone();
            let _ = base::execute(alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Verify CLIF: read partial sums from file written by CLIF function
        let clif_result: f64 = match std::fs::read(&clif_out) {
            Ok(bytes) => bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                .sum(),
            Err(_) => f64::NAN,
        };

        let wgpu_ok = check_result(wgpu_result, expected, &label, "wgpu");
        let burn_ok = check_result(burn_result, expected, &label, "Burn");
        let clif_ok = check_result(clif_result, expected, &label, "CLIF+GPU");

        results.push(GpuIterResult {
            name: label,
            wgpu_ms,
            burn_ms,
            clif_ms,
            verified: wgpu_ok && burn_ok && clif_ok,
        });
    }

    results
}
