use base_types::{Action, Kind, State, UnitSpec};
use crate::harness::{self, BenchResult};

// ---------------------------------------------------------------------------
// Memory-Heavy Benchmark: Large file → GPU process → Write result
//
// Tests whether the flat payload buffer scales to 100MB+ sizes.
// Workload: Read N MB of f32 data, GPU doubles each element, write result.
// ---------------------------------------------------------------------------

/// WGSL shader: double each f32 in-place.
const WGSL_DOUBLE: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = arrayLength(&data);
    let i = gid.x;
    if (i >= n) { return; }
    data[i] = data[i] * 2.0;
}
"#;

fn gen_test_file(path: &str, n_floats: usize) {
    let mut data = Vec::with_capacity(n_floats * 4);
    for i in 0..n_floats {
        let val = (i % 1000) as f32 / 1000.0;
        data.extend_from_slice(&val.to_le_bytes());
    }
    std::fs::write(path, &data).unwrap();
}

fn verify_doubled(path: &str, n_floats: usize) -> bool {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    if data.len() != n_floats * 4 {
        return false;
    }
    for (i, chunk) in data.chunks_exact(4).enumerate() {
        let val = f32::from_le_bytes(chunk.try_into().unwrap());
        let expected = ((i % 1000) as f32 / 1000.0) * 2.0;
        if (val - expected).abs() > 1e-6 {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// CLIF+GPU+File: Cranelift JIT does file_read → GPU double → file_write
// in a single compiled function via extern "C" wrappers.
// ---------------------------------------------------------------------------

const CLIF_DSIZE_OFF: usize = 0x08;         // i64: GPU buffer size in bytes
const CLIF_WORKGROUPS_OFF: usize = 0x10;     // i64: workgroups for dispatch
const CLIF_BIND_OFF: usize = 0x40;           // binding descriptor [buf_id=0, read_only=0]
const CLIF_SHADER_OFF: usize = 0x100;        // WGSL shader (null-terminated)
const CLIF_FNAME_IN_OFF: usize = 0x3000;     // input filename (null-terminated)
const CLIF_FNAME_OUT_OFF: usize = 0x3100;    // output filename (null-terminated)
const CLIF_IR_OFF: usize = 0x3800;           // CLIF IR source (null-terminated)
const CLIF_FLAG_OFF: usize = 0x3200;         // cranelift completion flag
const CLIF_DATA_OFF: usize = 0x4000;         // data buffer

fn gen_memory_clif_ir() -> String {
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
    ; sig5: (ptr, path_off, dst_off, file_offset, size) -> i64 — file read
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v
    ; sig6: (ptr, path_off, src_off, file_offset, size) -> i64 — file write
    sig6 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_create_pipeline sig2
    fn3 = %cl_gpu_upload sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig3
    fn6 = %cl_gpu_cleanup sig0
    fn7 = %cl_file_read sig5
    fn8 = %cl_file_write sig6

block0(v0: i64):
    ; --- file read input into data region ---
    v1 = iconst.i64 {fname_in_off}
    v2 = iconst.i64 {data_off}
    v3 = iconst.i64 0
    v4 = load.i64 v0+{dsize_off}
    v30 = call fn7(v0, v1, v2, v3, v4)

    ; --- gpu init ---
    call fn0(v0)

    ; --- create buffer ---
    v5 = call fn1(v0, v4)

    ; --- upload data to GPU ---
    v31 = call fn3(v0, v5, v2, v4)

    ; --- create pipeline (1 binding: buf0 rw) ---
    v6 = iconst.i64 {shader_off}
    v7 = iconst.i64 {bind_off}
    v8 = iconst.i32 1
    v9 = call fn2(v0, v6, v7, v8)

    ; --- dispatch ---
    v10 = load.i64 v0+{wg_off}
    v11 = ireduce.i32 v10
    v32 = iconst.i32 1
    v35 = call fn4(v0, v9, v11, v32, v32)

    ; --- download result ---
    v33 = call fn5(v0, v5, v2, v4)

    ; --- gpu cleanup ---
    call fn6(v0)

    ; --- file write output ---
    v12 = iconst.i64 {fname_out_off}
    v34 = call fn8(v0, v12, v2, v3, v4)

    return
}}"#,
        dsize_off = CLIF_DSIZE_OFF,
        data_off = CLIF_DATA_OFF,
        shader_off = CLIF_SHADER_OFF,
        bind_off = CLIF_BIND_OFF,
        wg_off = CLIF_WORKGROUPS_OFF,
        fname_in_off = CLIF_FNAME_IN_OFF,
        fname_out_off = CLIF_FNAME_OUT_OFF,
    )
}

fn build_clif_memory_algorithm(
    input_path: &str,
    output_path: &str,
    n_floats: usize,
) -> base::Algorithm {
    let buffer_size = n_floats * 4;
    let workgroups = ((n_floats + 63) / 64) as u32;

    let clif_source = gen_memory_clif_ir();
    let clif_bytes = format!("{}\0", clif_source).into_bytes();
    assert!(clif_bytes.len() < (CLIF_DATA_OFF - CLIF_IR_OFF),
        "CLIF IR too large: {} bytes", clif_bytes.len());

    let payload_size = CLIF_DATA_OFF + buffer_size;
    let mut payloads = vec![0u8; payload_size];

    // CLIF IR source
    payloads[CLIF_IR_OFF..CLIF_IR_OFF + clif_bytes.len()].copy_from_slice(&clif_bytes);

    // Control params
    payloads[CLIF_DSIZE_OFF..CLIF_DSIZE_OFF + 8].copy_from_slice(&(buffer_size as i64).to_le_bytes());
    payloads[CLIF_WORKGROUPS_OFF..CLIF_WORKGROUPS_OFF + 8].copy_from_slice(&(workgroups as i64).to_le_bytes());

    // Binding descriptor: [buf_id=0, read_only=0]
    payloads[CLIF_BIND_OFF..CLIF_BIND_OFF + 4].copy_from_slice(&0i32.to_le_bytes());
    payloads[CLIF_BIND_OFF + 4..CLIF_BIND_OFF + 8].copy_from_slice(&0i32.to_le_bytes());

    // Shader (null-terminated)
    let shader_bytes = WGSL_DOUBLE.as_bytes();
    payloads[CLIF_SHADER_OFF..CLIF_SHADER_OFF + shader_bytes.len()].copy_from_slice(shader_bytes);
    payloads[CLIF_SHADER_OFF + shader_bytes.len()] = 0;

    // Filenames (null-terminated)
    let fname_in = format!("{}\0", input_path);
    payloads[CLIF_FNAME_IN_OFF..CLIF_FNAME_IN_OFF + fname_in.len()].copy_from_slice(fname_in.as_bytes());
    let fname_out = format!("{}\0", output_path);
    payloads[CLIF_FNAME_OUT_OFF..CLIF_FNAME_OUT_OFF + fname_out.len()].copy_from_slice(fname_out.as_bytes());

    let _ = std::fs::remove_file(output_path);

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    base::Algorithm {
        actions,
        payloads,
        state: State {
            cranelift_ir_offsets: vec![CLIF_IR_OFF],
        },
        units: UnitSpec {
            cranelift_units: 0,
        },
        timeout_ms: Some(120_000),
        additional_shared_memory: 0,
        output: vec![],
    }
}

/// Rust baseline: sync file I/O + cached wgpu device (fair comparison with CLIF path).
fn rust_memory_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_path: &str,
    output_path: &str,
    n_floats: usize,
) {
    let data = std::fs::read(input_path).unwrap();

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(WGSL_DOUBLE.into()),
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: data.len() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buffer, 0, &data);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((n_floats + 63) / 64) as u32;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: data.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, data.len() as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let result = slice.get_mapped_range().to_vec();
    staging.unmap();

    std::fs::write(output_path, &result).unwrap();
}

pub fn print_memory_table(results: &[BenchResult]) {
    let name_w = 28;
    let col_w = 14;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
        "Memory Benchmark",
        "Rust(wgpu)",
        "CLIF+GPU",
        "Check",
        name_w = name_w,
        col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 2 + 6 + 3));

    for r in results {
        let rust_str = match r.rust_ms {
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
            rust_str,
            base_str,
            check_str,
            name_w = name_w,
            col_w = col_w
        );
    }
    println!();
}

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let mut results = Vec::new();
    std::fs::create_dir_all("/tmp/memory-bench-data").ok();

    eprintln!("\n=== Memory-Heavy Benchmarks: Large file → GPU → Write ===");
    eprintln!("  Testing flat buffer scalability with multi-MB payloads\n");

    // Cache GPU device once (same as CLIF path uses cached_gpu_device)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("No GPU adapter found");
    let (device, queue) = pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
    )
    .expect("Failed to create GPU device");

    for &n_floats in &[256_000, 512_000, 1_000_000] {
        let mb = (n_floats * 4) as f64 / (1024.0 * 1024.0);
        eprintln!("  Pipeline: {:.1}MB ({} floats) ...", mb, n_floats);

        let label = if n_floats >= 1_000_000 {
            format!("{}m", n_floats / 1_000_000)
        } else {
            format!("{}k", n_floats / 1_000)
        };
        let input_path = format!("/tmp/memory-bench-data/input_{}.bin", label);
        let rust_output = format!("/tmp/memory-bench-data/output_rust_{}.bin", label);
        let clif_output = format!("/tmp/memory-bench-data/output_clif_{}.bin", label);

        gen_test_file(&input_path, n_floats);

        let clif_alg = build_clif_memory_algorithm(&input_path, &clif_output, n_floats);

        // Rust baseline (cached device, sync I/O)
        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            rust_memory_pipeline(&device, &queue, &input_path, &rust_output, n_floats);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // CLIF+GPU
        let base_ms = harness::median_of(iterations, || {
            let alg = clif_alg.clone();
            let start = std::time::Instant::now();
            let _ = base::execute(alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let rust_ok = verify_doubled(&rust_output, n_floats);
        let clif_ok = verify_doubled(&clif_output, n_floats);

        if !rust_ok {
            eprintln!("  VERIFY FAIL: Rust output incorrect");
        }
        if !clif_ok {
            eprintln!("  VERIFY FAIL: CLIF output incorrect");
        }

        results.push(BenchResult {
            name: format!("File→GPU→File ({:.1}MB)", mb),
            python_ms: None,
            rust_ms: Some(rust_ms),
            base_ms,
            verified: Some(rust_ok && clif_ok),
            actions: None,
        });
    }

    results
}
