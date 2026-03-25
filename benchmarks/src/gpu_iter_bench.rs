use base::{BaseConfig, Algorithm};
use crate::harness::{self, BenchResult};

// ---------------------------------------------------------------------------
// Iterative GPU Benchmark: apply a trivial kernel N times to the same data.
//
// Compares approaches for keeping data GPU-resident across iterations.
//
// Raw wgpu:  Keeps buffer GPU-resident, only reads back at end
// Burn:      Naturally keeps tensors GPU-resident
// Base+GPU:  1 upload + N kernels + 1 download (via execute_into)
// ---------------------------------------------------------------------------

type Gpu = burn::backend::wgpu::Wgpu;

const GPU_ITER_ALGORITHM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gpu_iter_algorithm.bin"));

fn load_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(GPU_ITER_ALGORITHM).expect("Failed to deserialize gpu_iter algorithm")
}


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

use harness::{format_count, f32_sum};

fn gen_positive_floats(n: usize, seed: u64) -> Vec<f32> {
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

/// Payload layout: [passes: i64 (8 bytes)][f32 data]
fn build_payload(data: &[f32], passes: usize) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8 + data.len() * 4);
    payload.extend_from_slice(&(passes as i64).to_le_bytes());
    for &v in data {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    payload
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
    let scale = Tensor::<Gpu, 1>::from_data(TensorData::new(vec![1.001f32; n], [n]), &dev);

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

    // GPU reduction: N → N/64 partial sums
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


// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let mut results = Vec::new();

    eprintln!("\n=== GPU Iterative Benchmark: N kernel passes on same data ===");
    eprintln!("  Raw wgpu & Burn: data stays GPU-resident between passes");
    eprintln!("  Base+GPU:        Lean-built CLIF, data GPU-resident via execute_into\n");

    let n = 1_000_000usize;
    let num_groups = (n + 63) / 64;
    let out_size = num_groups * 4;

    let (config, alg) = load_algorithm();
    let mut base_instance = base::Base::new(config).expect("Base::new failed");

    for &passes in &[1, 10, 100, 500, 1000] {
        let label = format!("{}x scale {}", passes, format_count(n));
        eprintln!("  {} ...", label);

        let data = gen_positive_floats(n, 42);
        let expected = cpu_expected_sum(&data, passes);
        let payload = build_payload(&data, passes);
        let mut out_buf = vec![0u8; out_size];

        // Warmup all three
        std::hint::black_box(wgpu_iterative(&data, passes));
        std::hint::black_box(burn_iterative(&data, passes));
        let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);

        // Raw wgpu (GPU-resident)
        let wgpu_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            std::hint::black_box(wgpu_iterative(&data, passes));
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Burn (GPU-resident)
        let burn_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            std::hint::black_box(burn_iterative(&data, passes));
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Base+GPU (GPU-resident via CLIF loop)
        let clif_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Verify
        let wgpu_result = wgpu_iterative(&data, passes);
        let burn_result = burn_iterative(&data, passes);
        let clif_result = f32_sum(&out_buf);

        let wgpu_ok = check_result(wgpu_result, expected, &label, "wgpu");
        let burn_ok = check_result(burn_result, expected, &label, "Burn");
        let clif_ok = check_result(clif_result, expected, &label, "Base+GPU");

        results.push(BenchResult {
            name: label,
            col_a_ms: Some(wgpu_ms),
            col_b_ms: Some(burn_ms),
            base_ms: clif_ms,
            verified: Some(wgpu_ok && burn_ok && clif_ok),
        });
    }

    results
}
