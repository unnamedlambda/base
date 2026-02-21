use base_types::{Action, Kind, State, UnitSpec};
use crate::harness::{self, BenchResult};

// ---------------------------------------------------------------------------
// Memory-Heavy Benchmark: Large file → GPU process → Write result
//
// Tests whether the flat payload buffer scales to 100MB+ sizes.
// Workload: Read N MB of f32 data, GPU doubles each element, write result.
// ---------------------------------------------------------------------------

const SHADER_OFF: usize = 0x0000;
const FLAG1_ADDR: usize = 0x3000; // FileRead completion
const FLAG2_ADDR: usize = 0x3008; // GPU completion
const FLAG3_ADDR: usize = 0x3010; // FileWrite completion
const FNAME_IN_OFF: usize = 0x3100; // Input filename
const FNAME_OUT_OFF: usize = 0x3200; // Output filename
const DATA_OFF: usize = 0x4000;

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

/// Build Base algorithm: FileRead → GPU double → FileWrite.
///
/// Actions:
///   [0] FileRead (input file → DATA_OFF)
///   [1] Dispatch (GPU double in-place at DATA_OFF)
///   [2] FileWrite (DATA_OFF → output file)
///   [3] AsyncDispatch → File (action 0), flag FLAG1
///   [4] Wait(FLAG1)
///   [5] AsyncDispatch → GPU (action 1), flag FLAG2
///   [6] Wait(FLAG2)
///   [7] AsyncDispatch → File (action 2), flag FLAG3
///   [8] Wait(FLAG3)
fn build_memory_algorithm(
    input_path: &str,
    output_path: &str,
    n_floats: usize,
) -> base::Algorithm {
    let buffer_size = n_floats * 4;
    let payload_size = DATA_OFF + buffer_size;
    let mut payloads = vec![0u8; payload_size];

    // Shader
    let shader_bytes = WGSL_DOUBLE.as_bytes();
    payloads[SHADER_OFF..SHADER_OFF + shader_bytes.len()].copy_from_slice(shader_bytes);

    // Filenames
    let fname_in = format!("{}\0", input_path);
    let fname_out = format!("{}\0", output_path);
    payloads[FNAME_IN_OFF..FNAME_IN_OFF + fname_in.len()].copy_from_slice(fname_in.as_bytes());
    payloads[FNAME_OUT_OFF..FNAME_OUT_OFF + fname_out.len()].copy_from_slice(fname_out.as_bytes());

    let actions = vec![
        // [0] FileRead: input → DATA_OFF
        Action {
            kind: Kind::FileRead,
            src: FNAME_IN_OFF as u32,
            dst: DATA_OFF as u32,
            offset: 0,
            size: buffer_size as u32,
        },
        // [1] GPU Dispatch: double in-place
        Action {
            kind: Kind::Dispatch,
            src: DATA_OFF as u32,
            dst: DATA_OFF as u32,
            offset: 0,
            size: buffer_size as u32,
        },
        // [2] FileWrite: DATA_OFF → output
        Action {
            kind: Kind::FileWrite,
            src: DATA_OFF as u32,
            dst: FNAME_OUT_OFF as u32,
            offset: 0,
            size: buffer_size as u32,
        },
        // [3] AsyncDispatch → File (action 0)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 0,
            offset: FLAG1_ADDR as u32,
            size: 0,
        },
        // [4] Wait FileRead
        Action {
            kind: Kind::Wait,
            dst: FLAG1_ADDR as u32,
            src: 0,
            offset: 0,
            size: 0,
        },
        // [5] AsyncDispatch → GPU (action 1)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 0,
            src: 1,
            offset: FLAG2_ADDR as u32,
            size: 0,
        },
        // [6] Wait GPU
        Action {
            kind: Kind::Wait,
            dst: FLAG2_ADDR as u32,
            src: 0,
            offset: 0,
            size: 0,
        },
        // [7] AsyncDispatch → File (action 2)
        Action {
            kind: Kind::AsyncDispatch,
            dst: 2,
            src: 2,
            offset: FLAG3_ADDR as u32,
            size: 0,
        },
        // [8] Wait FileWrite
        Action {
            kind: Kind::Wait,
            dst: FLAG3_ADDR as u32,
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
            gpu_size: buffer_size,
            file_buffer_size: 1024 * 1024, // 1MB file buffer
            gpu_shader_offsets: vec![SHADER_OFF],
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            gpu_units: 1,
            file_units: 1,
            memory_units: 0,
            ffi_units: 0,
            cranelift_units: 0,
            backends_bits: 0xFFFF_FFFF,
        },
        memory_assignments: vec![],
        file_assignments: vec![],
        ffi_assignments: vec![],
        gpu_assignments: vec![0; num_actions],
        cranelift_assignments: vec![],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(256 * 1024),
        timeout_ms: Some(120_000),
        thread_name_prefix: Some("memory-bench".into()),
    }
}

/// Rust baseline: tokio::fs + wgpu manual.
async fn rust_memory_pipeline(input_path: &str, output_path: &str, n_floats: usize) {
    use std::sync::Arc;

    // Read file
    let data = tokio::fs::read(input_path).await.unwrap();

    // Setup wgpu
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Shader
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(WGSL_DOUBLE.into()),
    });

    // Buffer
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: data.len() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buffer, 0, &data);

    // Pipeline
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

    // Dispatch
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((n_floats + 63) / 64) as u32;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    // Readback
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: data.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, data.len() as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = tokio::sync::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.await.unwrap().unwrap();

    let result = slice.get_mapped_range().to_vec();
    drop(slice);
    staging.unmap();

    // Write file
    tokio::fs::write(output_path, &result).await.unwrap();
}

pub fn print_memory_table(results: &[BenchResult]) {
    let name_w = 28;
    let col_w = 14;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
        "Memory Benchmark",
        "Rust(tokio+wgpu)",
        "Base",
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
        let base_output = format!("/tmp/memory-bench-data/output_base_{}.bin", label);

        gen_test_file(&input_path, n_floats);

        let base_alg = build_memory_algorithm(&input_path, &base_output, n_floats);

        // Rust baseline
        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            rt.block_on(rust_memory_pipeline(&input_path, &rust_output, n_floats));
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Base
        let base_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let alg = base_alg.clone();
            let _ = base::execute(alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let rust_ok = verify_doubled(&rust_output, n_floats);
        let base_ok = verify_doubled(&base_output, n_floats);

        if !rust_ok {
            eprintln!("  VERIFY FAIL: Rust output incorrect");
        }
        if !base_ok {
            eprintln!("  VERIFY FAIL: Base output incorrect");
        }

        results.push(BenchResult {
            name: format!("File→GPU→File ({:.1}MB)", mb),
            python_ms: None,
            rust_ms: Some(rust_ms),
            base_ms,
            verified: Some(rust_ok && base_ok),
            actions: None,
        });
    }

    results
}
