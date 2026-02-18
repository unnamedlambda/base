use base_types::{Action, Kind, State, UnitSpec};
use crate::harness::{self, BenchResult};

// ---------------------------------------------------------------------------
// Iterative GPU Benchmark: apply a trivial kernel N times to the same data.
//
// Exposes the cost of round-tripping through shared memory on every dispatch.
//
// Current Base: N dispatches = N x (SharedMem→GPU→SharedMem)
// Ideal Base:   1 upload + N kernels + 1 download
// Burn:         Naturally keeps tensors GPU-resident
// Raw wgpu:     Keeps buffer GPU-resident, only reads back at end
// ---------------------------------------------------------------------------

type Gpu = burn::backend::wgpu::Wgpu;

// Partial sum reduction: each workgroup of 64 reduces 64 elements → 1 partial sum.
// For 1M floats: 15625 workgroups → ~61KB download vs 4MB for full readback.
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
// Burn: naturally keeps tensor GPU-resident across operations
// ---------------------------------------------------------------------------

fn burn_iterative(data: &[f32], passes: usize) -> f64 {
    use burn::tensor::{Tensor, TensorData};
    use std::sync::Arc;

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
    let dev = burn::backend::wgpu::init_device(setup, Default::default());

    let n = data.len();
    let mut t = Tensor::<Gpu, 1>::from_data(TensorData::new(data.to_vec(), [n]), &dev);
    let scale = Tensor::<Gpu, 1>::from_data(TensorData::new(vec![1.001f32], [1]), &dev);

    // N multiplications — tensor stays on GPU the whole time
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

    let instance = Instance::new(InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("No GPU adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &DeviceDescriptor::default(),
        None,
    ))
    .expect("Failed to create device");

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

    // GPU reduction: 1M → ~15K partial sums (~61KB download, matches Burn's scalar readback pattern)
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
// Base: current architecture — each Dispatch round-trips through SharedMemory
// ---------------------------------------------------------------------------

const SHADER_OFF: usize = 0x0000;
const FLAG_BASE: usize = 0x1000; // flags start here, 8 bytes each (need 1000*8 = 8KB)
// passes=1000 flags end at 0x2F40; 0x3000+ is safe for verify layout
const FNAME_VERIFY_OFF: usize = 0x3000; // output filename for verify run
const FLAG_FILE_VERIFY: usize = 0x3800; // FileWrite completion flag
const DATA_OFF: usize = 0x4000;  // data starts at 16KB (leaves room for 1000+ flags)

fn build_base_iterative(data: &[f32], passes: usize) -> base::Algorithm {
    let n = data.len();
    let buffer_size = n * 4;
    let payload_size = DATA_OFF + buffer_size;
    let mut payloads = vec![0u8; payload_size];

    // Write shader
    let shader_bytes = WGSL_SCALE.as_bytes();
    payloads[SHADER_OFF..SHADER_OFF + shader_bytes.len()].copy_from_slice(shader_bytes);

    // Write data
    for (i, &v) in data.iter().enumerate() {
        let off = DATA_OFF + i * 4;
        payloads[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    // Build actions: N passes queued to GPU, single Wait at end.
    //
    // GPU mailbox is single-threaded and processes in order, so
    // dispatch N+1 sees the result of dispatch N without intermediate Waits.
    //
    // Action 0: the GPU Dispatch action (reused by all passes)
    // Actions 1..N: AsyncDispatch → GPU (queued, no wait)
    // Action N+1: Wait on last flag
    let mut actions = Vec::new();

    // [0] GPU Dispatch (the actual kernel)
    actions.push(Action {
        kind: Kind::Dispatch,
        src: DATA_OFF as u32,
        dst: DATA_OFF as u32,
        offset: 0,
        size: buffer_size as u32,
    });

    // Queue all passes without waiting between them
    for i in 0..passes {
        let flag_off = (FLAG_BASE + i * 8) as u32;

        actions.push(Action {
            kind: Kind::AsyncDispatch,
            dst: 0,  // GPU unit
            src: 0,  // action index 0
            offset: flag_off,
            size: 0,
        });
    }

    // Single Wait on the last flag — GPU processes all in order
    let last_flag = (FLAG_BASE + (passes - 1) * 8) as u32;
    actions.push(Action {
        kind: Kind::Wait,
        dst: last_flag,
        src: 0,
        offset: 0,
        size: 0,
    });

    let num_actions = actions.len();

    base::Algorithm {
        actions,
        payloads,
        state: State {
            regs_per_unit: 0,
            gpu_size: buffer_size,
            computational_regs: 0,
            file_buffer_size: 0,
            gpu_shader_offsets: vec![SHADER_OFF],
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            simd_units: 0,
            gpu_units: 1,
            computational_units: 0,
            file_units: 0,
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
        timeout_ms: Some(300_000),
        thread_name_prefix: Some("gpu-iter".into()),
    }
}

/// Same as `build_base_iterative` but appends a FileWrite so the result can be read back.
fn build_base_iterative_verify(data: &[f32], passes: usize, output_path: &str) -> base::Algorithm {
    let n = data.len();
    let buffer_size = n * 4;
    let payload_size = DATA_OFF + buffer_size;
    let mut payloads = vec![0u8; payload_size];

    let shader_bytes = WGSL_SCALE.as_bytes();
    payloads[SHADER_OFF..SHADER_OFF + shader_bytes.len()].copy_from_slice(shader_bytes);

    let fname = format!("{}\0", output_path);
    payloads[FNAME_VERIFY_OFF..FNAME_VERIFY_OFF + fname.len()].copy_from_slice(fname.as_bytes());

    for (i, &v) in data.iter().enumerate() {
        let off = DATA_OFF + i * 4;
        payloads[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    let mut actions = Vec::new();
    // [0] GPU kernel (referenced by all AsyncDispatches)
    actions.push(Action { kind: Kind::Dispatch, src: DATA_OFF as u32, dst: DATA_OFF as u32, offset: 0, size: buffer_size as u32 });
    // [1..=passes] AsyncDispatch → GPU
    for i in 0..passes {
        actions.push(Action { kind: Kind::AsyncDispatch, dst: 0, src: 0, offset: (FLAG_BASE + i * 8) as u32, size: 0 });
    }
    // [passes+1] Wait last GPU flag
    actions.push(Action { kind: Kind::Wait, dst: (FLAG_BASE + (passes - 1) * 8) as u32, src: 0, offset: 0, size: 0 });
    // [passes+2] FileWrite action (target for AsyncDispatch to file)
    let fw_idx = (passes + 2) as u32;
    actions.push(Action { kind: Kind::FileWrite, src: DATA_OFF as u32, dst: FNAME_VERIFY_OFF as u32, offset: 0, size: buffer_size as u32 });
    // [passes+3] AsyncDispatch → File
    actions.push(Action { kind: Kind::AsyncDispatch, dst: 2, src: fw_idx, offset: FLAG_FILE_VERIFY as u32, size: 0 });
    // [passes+4] Wait FileWrite
    actions.push(Action { kind: Kind::Wait, dst: FLAG_FILE_VERIFY as u32, src: 0, offset: 0, size: 0 });

    let num_actions = actions.len();
    base::Algorithm {
        actions,
        payloads,
        state: State {
            regs_per_unit: 0,
            gpu_size: buffer_size,
            computational_regs: 0,
            file_buffer_size: 1024 * 1024,
            gpu_shader_offsets: vec![SHADER_OFF],
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            simd_units: 0, gpu_units: 1, computational_units: 0, file_units: 1,
            network_units: 0, memory_units: 0, ffi_units: 0, hash_table_units: 0,
            lmdb_units: 0, cranelift_units: 0, backends_bits: 0xFFFF_FFFF,
        },
        simd_assignments: vec![], computational_assignments: vec![], memory_assignments: vec![],
        file_assignments: vec![], network_assignments: vec![], ffi_assignments: vec![],
        hash_table_assignments: vec![], lmdb_assignments: vec![],
        gpu_assignments: vec![0; num_actions], cranelift_assignments: vec![],
        worker_threads: Some(1), blocking_threads: Some(1),
        stack_size: Some(256 * 1024), timeout_ms: Some(300_000),
        thread_name_prefix: Some("gpu-iter-verify".into()),
    }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn check_result(actual: f64, expected: f64, label: &str, impl_name: &str) -> bool {
    let rel = (actual - expected).abs() / expected.abs().max(1.0);
    let ok = rel <= 0.05; // 5% tolerance for accumulated floating point
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

pub fn print_iter_table(results: &[BenchResult]) {
    let name_w = 26;
    let col_w = 14;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>6}",
        "GPU Iterative",
        "Raw wgpu",
        "Burn",
        "Base",
        "Check",
        name_w = name_w,
        col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 3 + 6 + 4));

    for r in results {
        let wgpu_str = match r.rust_ms {
            Some(ms) => format!("{:.1}ms", ms),
            None => "N/A".to_string(),
        };
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
            Some(true) => "    \u{2713}",
            Some(false) => "    \u{2717}",
            None => "    \u{2014}",
        };
        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {}",
            r.name,
            wgpu_str,
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

    eprintln!("\n=== GPU Iterative Benchmark: N kernel passes on same data ===");
    eprintln!("  Raw wgpu & Burn: data stays GPU-resident between passes");
    eprintln!("  Base (current):  round-trips through SharedMemory each pass\n");

    // 1M floats = 4MB. Workgroups needed = 1M/64 = 15625 (well under 65535 limit).
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

        // Base (round-trips each pass)
        let base_alg = build_base_iterative(&data, passes);
        let mut base_result = 0.0f64;
        let base_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let alg = base_alg.clone();
            let _ = base::execute(alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Verify Base: run once with FileWrite, read back in Rust and sum
        let verify_path = format!("/tmp/gpu-iter-verify-{}.bin", passes);
        let verify_alg = build_base_iterative_verify(&data, passes, &verify_path);
        let _ = base::execute(verify_alg);
        let base_sum: f64 = match std::fs::read(&verify_path) {
            Ok(bytes) => bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                .sum(),
            Err(_) => f64::NAN,
        };

        let wgpu_ok = check_result(wgpu_result, expected, &label, "wgpu");
        let burn_ok = check_result(burn_result, expected, &label, "Burn");
        let base_ok = check_result(base_sum, expected, &label, "Base");

        results.push(BenchResult {
            name: label,
            python_ms: Some(burn_ms),  // reuse python_ms field for Burn
            rust_ms: Some(wgpu_ms),    // reuse rust_ms field for raw wgpu
            base_ms,
            verified: Some(wgpu_ok && burn_ok && base_ok),
            actions: Some(1 + passes + 1), // 1 dispatch + N*async + 1 wait
        });
    }

    results
}
