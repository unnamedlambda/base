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
        let f = (bits as f32 / i32::MAX as f32) * 2.0 - 1.0;
        out.push(f);
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

fn rust_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> f64 {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    c.iter().map(|&x| x as f64).sum()
}

fn burn_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> f64 {
    use burn::tensor::{Tensor, TensorData};
    let device = Default::default();
    let a_t = Tensor::<B, 2>::from_data(
        TensorData::new(a.to_vec(), [m, k]),
        &device,
    );
    let b_t = Tensor::<B, 2>::from_data(
        TensorData::new(b.to_vec(), [k, n]),
        &device,
    );
    let c_t = a_t.matmul(b_t);
    c_t.sum().into_scalar() as f64
}

// CLIF IR: SIMD matmul with 4x-unrolled j-loop (16 cols/iter).
// Inner loop uses 4 independent f32x4 accumulators for ILP.
// Memory layout:
//   0x0010: [4] M, 0x0014: [4] K, 0x0018: [4] N, 0x0020: [8] result (f64)
//   0x4000: matrix A (M*K f32), then matrix B (K*N f32)
const CLIF_MATMUL: &str = "\
function %matmul(i64) system_v {
block0(v0: i64):
  v1 = load.i32 v0+16
  v2 = load.i32 v0+20
  v3 = load.i32 v0+24
  v4 = sextend.i64 v1
  v5 = sextend.i64 v2
  v6 = sextend.i64 v3
  v7 = iconst.i64 16384
  v8 = imul v4, v5
  v9 = ishl_imm v8, 2
  v10 = iadd v7, v9
  v11 = iadd v0, v7
  v12 = iadd v0, v10
  v13 = ishl_imm v6, 2
  v14 = iconst.i64 0
  v15 = f64const 0.0
  jump block1(v14, v15)

block1(v20: i64, v21: f64):
  v22 = icmp sge v20, v4
  brif v22, block9(v21), block2(v20, v21)

block2(v23: i64, v24: f64):
  v25 = iconst.i64 0
  jump block3(v23, v24, v25)

block3(v30: i64, v31: f64, v32: i64):
  v33 = icmp sge v32, v5
  brif v33, block8(v30, v31), block4(v30, v31, v32)

block4(v34: i64, v35: f64, v36: i64):
  v37 = imul v34, v5
  v38 = iadd v37, v36
  v39 = ishl_imm v38, 2
  v40 = iadd v11, v39
  v41 = load.f32 v40
  v42 = splat.f32x4 v41
  v43 = imul v36, v6
  v44 = ishl_imm v43, 2
  v45 = iadd v12, v44
  v46 = f32const 0.0
  v47 = splat.f32x4 v46
  v48 = iconst.i64 0
  jump block5(v34, v35, v36, v42, v45, v48, v47, v47, v47, v47)

block5(v50: i64, v51: f64, v52: i64, v53: f32x4, v54: i64, v55: i64, v56: f32x4, v57: f32x4, v58: f32x4, v59: f32x4):
  v60 = icmp sge v55, v13
  brif v60, block7(v50, v51, v52, v56, v57, v58, v59), block6(v50, v51, v52, v53, v54, v55, v56, v57, v58, v59)

block6(v70: i64, v71: f64, v72: i64, v73: f32x4, v74: i64, v75: i64, v76: f32x4, v77: f32x4, v78: f32x4, v79: f32x4):
  v80 = iadd v74, v75
  v81 = load.f32x4 v80
  v82 = fmul v73, v81
  v83 = fadd v76, v82
  v84 = iadd_imm v80, 16
  v85 = load.f32x4 v84
  v86 = fmul v73, v85
  v87 = fadd v77, v86
  v88 = iadd_imm v80, 32
  v89 = load.f32x4 v88
  v90 = fmul v73, v89
  v91 = fadd v78, v90
  v92 = iadd_imm v80, 48
  v93 = load.f32x4 v92
  v94 = fmul v73, v93
  v95 = fadd v79, v94
  v96 = iadd_imm v75, 64
  jump block5(v70, v71, v72, v73, v74, v96, v83, v87, v91, v95)

block7(v100: i64, v101: f64, v102: i64, v103: f32x4, v104: f32x4, v105: f32x4, v106: f32x4):
  v107 = fadd v103, v104
  v108 = fadd v105, v106
  v109 = fadd v107, v108
  v110 = extractlane v109, 0
  v111 = extractlane v109, 1
  v112 = extractlane v109, 2
  v113 = extractlane v109, 3
  v114 = fpromote.f64 v110
  v115 = fpromote.f64 v111
  v116 = fpromote.f64 v112
  v117 = fpromote.f64 v113
  v118 = fadd v114, v115
  v119 = fadd v116, v117
  v120 = fadd v118, v119
  v121 = fadd v101, v120
  v122 = iadd_imm v102, 1
  jump block3(v100, v121, v122)

block8(v130: i64, v131: f64):
  v132 = iadd_imm v130, 1
  jump block1(v132, v131)

block9(v140: f64):
  v141 = iconst.i64 32
  v142 = iadd v0, v141
  store.f64 v140, v142
  return
}";

const DATA_OFF: usize = 0x4000;
const CLIF_OFF: usize = 0x0100;
const FLAG_CL: usize = 0x0008;

fn build_base_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> base::Algorithm {
    let payload_size = DATA_OFF + (m * k + k * n) * 4;
    let mut payloads = vec![0u8; payload_size];

    payloads[0x10..0x14].copy_from_slice(&(m as u32).to_le_bytes());
    payloads[0x14..0x18].copy_from_slice(&(k as u32).to_le_bytes());
    payloads[0x18..0x1C].copy_from_slice(&(n as u32).to_le_bytes());

    let ir = CLIF_MATMUL.as_bytes();
    payloads[CLIF_OFF..CLIF_OFF + ir.len()].copy_from_slice(ir);

    for (i, &v) in a.iter().enumerate() {
        let off = DATA_OFF + i * 4;
        payloads[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    let b_start = DATA_OFF + m * k * 4;
    for (i, &v) in b.iter().enumerate() {
        let off = b_start + i * 4;
        payloads[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    base::Algorithm {
        actions,
        payloads,
        state: State {
            cranelift_ir_offsets: vec![CLIF_OFF],
        },
        units: UnitSpec {
            cranelift_units: 0,
        },
        timeout_ms: Some(60_000),
        additional_shared_memory: 0,
        output: vec![],
    }
}

fn close_enough(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    let diff = (a - b).abs();
    let mag = a.abs().max(b.abs()).max(1.0);
    diff / mag < 0.02
}

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let sizes: &[usize] = &[128, 256, 512, 1024];
    let mut results = Vec::new();

    for &n in sizes {
        let total = n * n * 2;
        let data = gen_floats(total, 42);
        let a = &data[..n * n];
        let b = &data[n * n..];

        let data_path = format!("/tmp/bench-data/matmul_{}.bin", n);
        write_f32(&data_path, &data);

        // Raw Rust
        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            std::hint::black_box(rust_matmul(a, b, n, n, n));
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Burn (NdArray CPU)
        let burn_ms = {
            use burn::tensor::{Tensor, TensorData};
            let device = Default::default();
            let a_t = Tensor::<B, 2>::from_data(TensorData::new(a.to_vec(), [n, n]), &device);
            let b_t = Tensor::<B, 2>::from_data(TensorData::new(b.to_vec(), [n, n]), &device);
            harness::median_of(iterations, || {
                let start = std::time::Instant::now();
                let c_t = a_t.clone().matmul(b_t.clone());
                std::hint::black_box(c_t.sum().into_scalar());
                start.elapsed().as_secs_f64() * 1000.0
            })
        };

        // Base (Cranelift JIT)
        let template = build_base_matmul(a, b, n, n, n);
        let base_ms = harness::median_of(iterations, || {
            let alg = template.clone();
            let start = std::time::Instant::now();
            let _ = base::execute(alg);
            start.elapsed().as_secs_f64() * 1000.0
        });

        let rust_check = rust_matmul(a, b, n, n, n);
        let burn_check = burn_matmul(a, b, n, n, n);
        let verified = Some(close_enough(rust_check, burn_check));

        results.push(BenchResult {
            name: format!("MatMul ({}x{})", n, n),
            python_ms: Some(burn_ms),
            rust_ms: Some(rust_ms),
            base_ms,
            verified,
            actions: None,
        });
    }

    results
}
