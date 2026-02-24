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

fn rust_vec_add(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        sum += (a[i] + b[i]) as f64;
    }
    sum
}

fn burn_vec_add(a: &[f32], b: &[f32]) -> f64 {
    use burn::tensor::{Tensor, TensorData};
    let n = a.len();
    let device = Default::default();
    let a_t = Tensor::<B, 1>::from_data(
        TensorData::new(a.to_vec(), [n]),
        &device,
    );
    let b_t = Tensor::<B, 1>::from_data(
        TensorData::new(b.to_vec(), [n]),
        &device,
    );
    let c_t = a_t + b_t;
    c_t.sum().into_scalar() as f64
}

// CLIF IR: 4x-unrolled SIMD vec_add of two f32 arrays → f64 sum.
// Main loop: 4 independent f32x4 accumulators (16 floats/iter) for ILP.
// Memory layout:
//   0x0010: [4] n (u32), 0x0018: [8] result (f64)
//   0x4000: array A (n * f32), then array B (n * f32)
const CLIF_VEC_ADD: &str = "\
function %vec_add(i64) system_v {
block0(v0: i64):
  v1 = load.i32 v0+16
  v2 = sextend.i64 v1
  v3 = iconst.i64 16384
  v4 = iadd v0, v3
  v5 = ishl_imm v2, 2
  v6 = iadd v4, v5
  v7 = ushr_imm v2, 4
  v8 = ishl_imm v7, 6
  v9 = ushr_imm v2, 2
  v10 = ishl_imm v9, 4
  v11 = ishl_imm v2, 2
  v12 = f32const 0.0
  v13 = splat.f32x4 v12
  v14 = iconst.i64 0
  jump block1(v14, v13, v13, v13, v13)

block1(v20: i64, v21: f32x4, v22: f32x4, v23: f32x4, v24: f32x4):
  v25 = icmp sge v20, v8
  brif v25, block3(v20, v21, v22, v23, v24), block2(v20, v21, v22, v23, v24)

block2(v30: i64, v31: f32x4, v32: f32x4, v33: f32x4, v34: f32x4):
  v35 = iadd v4, v30
  v36 = iadd v6, v30
  v37 = load.f32x4 v35
  v38 = load.f32x4 v36
  v39 = fadd v37, v38
  v40 = fadd v31, v39
  v41 = iadd_imm v35, 16
  v42 = iadd_imm v36, 16
  v43 = load.f32x4 v41
  v44 = load.f32x4 v42
  v45 = fadd v43, v44
  v46 = fadd v32, v45
  v47 = iadd_imm v35, 32
  v48 = iadd_imm v36, 32
  v49 = load.f32x4 v47
  v50 = load.f32x4 v48
  v51 = fadd v49, v50
  v52 = fadd v33, v51
  v53 = iadd_imm v35, 48
  v54 = iadd_imm v36, 48
  v55 = load.f32x4 v53
  v56 = load.f32x4 v54
  v57 = fadd v55, v56
  v58 = fadd v34, v57
  v59 = iadd_imm v30, 64
  jump block1(v59, v40, v46, v52, v58)

block3(v60: i64, v61: f32x4, v62: f32x4, v63: f32x4, v64: f32x4):
  v65 = fadd v61, v62
  v66 = fadd v63, v64
  v67 = fadd v65, v66
  jump block4(v60, v67)

block4(v70: i64, v71: f32x4):
  v72 = icmp sge v70, v10
  brif v72, block6(v70, v71), block5(v70, v71)

block5(v80: i64, v81: f32x4):
  v82 = iadd v4, v80
  v83 = iadd v6, v80
  v84 = load.f32x4 v82
  v85 = load.f32x4 v83
  v86 = fadd v84, v85
  v87 = fadd v81, v86
  v88 = iadd_imm v80, 16
  jump block4(v88, v87)

block6(v90: i64, v91: f32x4):
  v92 = extractlane v91, 0
  v93 = extractlane v91, 1
  v94 = extractlane v91, 2
  v95 = extractlane v91, 3
  v96 = fpromote.f64 v92
  v97 = fpromote.f64 v93
  v98 = fpromote.f64 v94
  v99 = fpromote.f64 v95
  v100 = fadd v96, v97
  v101 = fadd v98, v99
  v102 = fadd v100, v101
  jump block7(v90, v102)

block7(v110: i64, v111: f64):
  v112 = icmp sge v110, v11
  brif v112, block9(v111), block8(v110, v111)

block8(v120: i64, v121: f64):
  v122 = iadd v4, v120
  v123 = iadd v6, v120
  v124 = load.f32 v122
  v125 = load.f32 v123
  v126 = fadd v124, v125
  v127 = fpromote.f64 v126
  v128 = fadd v121, v127
  v129 = iadd_imm v120, 4
  jump block7(v129, v128)

block9(v130: f64):
  v131 = iconst.i64 24
  v132 = iadd v0, v131
  store.f64 v130, v132
  return
}";

const DATA_OFF: usize = 0x4000;
const CLIF_OFF: usize = 0x0100;
const FLAG_CL: usize = 0x0008;

fn build_base_vec_add(a: &[f32], b: &[f32]) -> base::Algorithm {
    let n = a.len();
    let payload_size = DATA_OFF + n * 4 * 2;
    let mut payloads = vec![0u8; payload_size];

    payloads[0x10..0x14].copy_from_slice(&(n as u32).to_le_bytes());

    let ir = CLIF_VEC_ADD.as_bytes();
    payloads[CLIF_OFF..CLIF_OFF + ir.len()].copy_from_slice(ir);

    for (i, &v) in a.iter().enumerate() {
        let off = DATA_OFF + i * 4;
        payloads[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    let b_start = DATA_OFF + n * 4;
    for (i, &v) in b.iter().enumerate() {
        let off = b_start + i * 4;
        payloads[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::AsyncDispatch, dst: 9, src: 2, offset: FLAG_CL as u32, size: 1 },
        Action { kind: Kind::Wait, dst: FLAG_CL as u32, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Noop, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    base::Algorithm {
        actions,
        payloads,
        state: State {
            cranelift_ir_offsets: vec![CLIF_OFF],
        },
        units: UnitSpec {
            cranelift_units: 1,
        },
cranelift_assignments: vec![],
        worker_threads: Some(1), blocking_threads: Some(1),
        stack_size: Some(256 * 1024), timeout_ms: Some(30_000),
        thread_name_prefix: Some("vecadd-bench".into()),
        additional_shared_memory: 0,
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
    let sizes: &[usize] = &[1_000_000, 10_000_000, 50_000_000];
    let mut results = Vec::new();

    for &n in sizes {
        let a = gen_floats(n, 42);
        let b = gen_floats(n, 123);

        let data_path = format!("/tmp/bench-data/vecadd_{}.bin", n);
        let mut combined = Vec::with_capacity(n * 2);
        combined.extend_from_slice(&a);
        combined.extend_from_slice(&b);
        write_f32(&data_path, &combined);

        // Raw Rust
        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            std::hint::black_box(rust_vec_add(&a, &b));
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Burn (NdArray CPU)
        let burn_ms = {
            use burn::tensor::{Tensor, TensorData};
            let device = Default::default();
            let a_t = Tensor::<B, 1>::from_data(TensorData::new(a.to_vec(), [n]), &device);
            let b_t = Tensor::<B, 1>::from_data(TensorData::new(b.to_vec(), [n]), &device);
            harness::median_of(iterations, || {
                let start = std::time::Instant::now();
                let c_t = a_t.clone() + b_t.clone();
                std::hint::black_box(c_t.sum().into_scalar());
                start.elapsed().as_secs_f64() * 1000.0
            })
        };

        // Base (Cranelift JIT)
        let template = build_base_vec_add(&a, &b);
        let base_ms = harness::median_of(iterations, || {
            let alg = template.clone();
            let start = std::time::Instant::now();
            let _ = base::execute(alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let rust_check = rust_vec_add(&a, &b);
        let burn_check = burn_vec_add(&a, &b);
        let verified = Some(close_enough(rust_check, burn_check));

        results.push(BenchResult {
            name: format!("VecAdd ({})", format_count(n)),
            python_ms: Some(burn_ms),
            rust_ms: Some(rust_ms),
            base_ms,
            verified,
            actions: None,
        });
    }

    results
}
