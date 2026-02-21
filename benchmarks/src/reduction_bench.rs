use std::fs;
use std::path::Path;

use base_types::{Action, Kind, State, UnitSpec};
use crate::harness::{self, BenchResult};

type B = burn::backend::NdArray<f32>;

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

fn write_f32(path: &str, data: &[f32]) {
    let dir = Path::new(path).parent().unwrap();
    fs::create_dir_all(dir).ok();
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn rust_sum(data: &[f32]) -> f64 {
    data.iter().map(|&x| x as f64).sum()
}

fn burn_sum(data: &[f32]) -> f64 {
    use burn::tensor::{Tensor, TensorData};
    let n = data.len();
    let device = Default::default();
    let t = Tensor::<B, 1>::from_data(
        TensorData::new(data.to_vec(), [n]),
        &device,
    );
    t.sum().into_scalar() as f64
}

// CLIF IR: 4x-unrolled SIMD sum of f32 array → f64 result.
// Main loop: 4 independent f32x4 accumulators (16 floats/iter) for ILP.
// Memory layout:
//   0x0010: [4] n (u32), 0x0018: [8] result (f64), 0x4000: data (n * f32)
const CLIF_SUM: &str = "\
function %sum_reduce(i64) system_v {
block0(v0: i64):
  v1 = load.i32 v0+16
  v2 = sextend.i64 v1
  v3 = iconst.i64 16384
  v4 = iadd v0, v3
  v5 = ushr_imm v2, 4
  v6 = ishl_imm v5, 6
  v7 = ushr_imm v2, 2
  v8 = ishl_imm v7, 4
  v9 = ishl_imm v2, 2
  v10 = f32const 0.0
  v11 = splat.f32x4 v10
  v12 = iconst.i64 0
  jump block1(v12, v11, v11, v11, v11)

block1(v20: i64, v21: f32x4, v22: f32x4, v23: f32x4, v24: f32x4):
  v25 = icmp sge v20, v6
  brif v25, block3(v20, v21, v22, v23, v24), block2(v20, v21, v22, v23, v24)

block2(v30: i64, v31: f32x4, v32: f32x4, v33: f32x4, v34: f32x4):
  v35 = iadd v4, v30
  v36 = load.f32x4 v35
  v37 = iadd_imm v35, 16
  v38 = load.f32x4 v37
  v39 = iadd_imm v35, 32
  v40 = load.f32x4 v39
  v41 = iadd_imm v35, 48
  v42 = load.f32x4 v41
  v43 = fadd v31, v36
  v44 = fadd v32, v38
  v45 = fadd v33, v40
  v46 = fadd v34, v42
  v47 = iadd_imm v30, 64
  jump block1(v47, v43, v44, v45, v46)

block3(v50: i64, v51: f32x4, v52: f32x4, v53: f32x4, v54: f32x4):
  v55 = fadd v51, v52
  v56 = fadd v53, v54
  v57 = fadd v55, v56
  jump block4(v50, v57)

block4(v60: i64, v61: f32x4):
  v62 = icmp sge v60, v8
  brif v62, block6(v60, v61), block5(v60, v61)

block5(v70: i64, v71: f32x4):
  v72 = iadd v4, v70
  v73 = load.f32x4 v72
  v74 = fadd v71, v73
  v75 = iadd_imm v70, 16
  jump block4(v75, v74)

block6(v80: i64, v81: f32x4):
  v82 = extractlane v81, 0
  v83 = extractlane v81, 1
  v84 = extractlane v81, 2
  v85 = extractlane v81, 3
  v86 = fpromote.f64 v82
  v87 = fpromote.f64 v83
  v88 = fpromote.f64 v84
  v89 = fpromote.f64 v85
  v90 = fadd v86, v87
  v91 = fadd v88, v89
  v92 = fadd v90, v91
  jump block7(v80, v92)

block7(v100: i64, v101: f64):
  v102 = icmp sge v100, v9
  brif v102, block9(v101), block8(v100, v101)

block8(v110: i64, v111: f64):
  v112 = iadd v4, v110
  v113 = load.f32 v112
  v114 = fpromote.f64 v113
  v115 = fadd v111, v114
  v116 = iadd_imm v110, 4
  jump block7(v116, v115)

block9(v120: f64):
  v121 = iconst.i64 24
  v122 = iadd v0, v121
  store.f64 v120, v122
  return
}";

const DATA_OFF: usize = 0x4000;
const CLIF_OFF: usize = 0x0100;
const FLAG_CL: usize = 0x0008;

fn build_base_sum(data: &[f32]) -> base::Algorithm {
    let n = data.len();
    let payload_size = DATA_OFF + n * 4;
    let mut payloads = vec![0u8; payload_size];

    // n at 0x10
    payloads[0x10..0x14].copy_from_slice(&(n as u32).to_le_bytes());

    // CLIF IR at 0x100
    let ir = CLIF_SUM.as_bytes();
    payloads[CLIF_OFF..CLIF_OFF + ir.len()].copy_from_slice(ir);

    // Data at 0x4000
    for (i, &v) in data.iter().enumerate() {
        let off = DATA_OFF + i * 4;
        payloads[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    let actions = vec![
        // [0] Dispatch CL work (action 2) with flag at FLAG_CL
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 2, offset: FLAG_CL as u32, size: 1 },
        // [1] Wait for CL completion
        Action { kind: Kind::Wait, dst: FLAG_CL as u32, src: 0, offset: 0, size: 0 },
        // [2] CL work action: call function 0 with base ptr
        Action { kind: Kind::Fence, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    base::Algorithm {
        actions,
        payloads,
        state: State {
            gpu_size: 0,
            file_buffer_size: 0,
            gpu_shader_offsets: vec![],
            cranelift_ir_offsets: vec![CLIF_OFF],
        },
        units: UnitSpec {
            gpu_units: 0,
            file_units: 0, memory_units: 0,
            ffi_units: 0,
            cranelift_units: 1, backends_bits: 0,
        },
        memory_assignments: vec![],
        file_assignments: vec![],
        ffi_assignments: vec![],
        gpu_assignments: vec![],
        cranelift_assignments: vec![],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(256 * 1024),
        timeout_ms: Some(30_000),
        thread_name_prefix: Some("sum-bench".into()),
    }
}

fn close_enough(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    let diff = (a - b).abs();
    let mag = a.abs().max(b.abs()).max(1.0);
    diff / mag < 0.01
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

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let sizes: &[usize] = &[1_000_000, 10_000_000, 50_000_000, 100_000_000];
    let mut results = Vec::new();

    for &n in sizes {
        let data = gen_floats(n, 42);

        let data_path = format!("/tmp/bench-data/reduction_{}.bin", n);
        write_f32(&data_path, &data);

        // Raw Rust
        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            std::hint::black_box(rust_sum(&data));
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Burn (NdArray CPU)
        let burn_ms = {
            use burn::tensor::{Tensor, TensorData};
            let device = Default::default();
            let t = Tensor::<B, 1>::from_data(TensorData::new(data.to_vec(), [n]), &device);
            harness::median_of(iterations, || {
                let start = std::time::Instant::now();
                std::hint::black_box(t.clone().sum().into_scalar());
                start.elapsed().as_secs_f64() * 1000.0
            })
        };

        // Base (Cranelift JIT)
        let template = build_base_sum(&data);
        let base_ms = harness::median_of(iterations, || {
            let alg = template.clone();
            let start = std::time::Instant::now();
            let _ = base::execute(alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let rust_check = rust_sum(&data);
        let burn_check = burn_sum(&data);
        let verified = Some(close_enough(rust_check, burn_check));

        results.push(BenchResult {
            name: format!("Sum ({})", format_count(n)),
            python_ms: Some(burn_ms),
            rust_ms: Some(rust_ms),
            base_ms,
            verified,
            actions: None,
        });
    }

    results
}
