use std::sync::Arc;
use std::io::{Read as _, Write as _};
use base_types::{Action, Kind, State, UnitSpec};
use crate::harness::{self, BenchResult};

// ---------------------------------------------------------------------------
// Parallel Histogram Benchmark
//
// N random u32 values in [0, BINS). W workers each scan a chunk, build a
// thread-local histogram, then the orchestrator merges them.
//
// Rust baseline: std::thread::spawn with per-thread local histograms.
// Base (CLIF):   cl_thread_spawn/join with per-worker regions in payload.
// ---------------------------------------------------------------------------

const BINS: usize = 256;

fn gen_data(n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push(((state >> 33) as u32) % (BINS as u32));
    }
    out
}

fn reference_histogram(data: &[u32]) -> Vec<u64> {
    let mut hist = vec![0u64; BINS];
    for &v in data {
        hist[v as usize] += 1;
    }
    hist
}

// ---------------------------------------------------------------------------
// Rust baseline
// ---------------------------------------------------------------------------

fn rust_parallel_histogram(data: &Arc<Vec<u32>>, workers: usize, out_path: &str) {
    let n = data.len();
    let chunk_size = (n + workers - 1) / workers;
    let mut handles = Vec::with_capacity(workers);

    for w in 0..workers {
        let data = data.clone();
        let start = (w * chunk_size).min(n);
        let end = ((w + 1) * chunk_size).min(n);
        handles.push(std::thread::spawn(move || {
            let mut local = vec![0u64; BINS];
            for &v in &data[start..end] {
                local[v as usize] += 1;
            }
            local
        }));
    }

    let mut merged = vec![0u64; BINS];
    for h in handles {
        let local = h.join().unwrap();
        for i in 0..BINS {
            merged[i] += local[i];
        }
    }

    let bytes: Vec<u8> = merged.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut f = std::fs::File::create(out_path).unwrap();
    f.write_all(&bytes).unwrap();
    f.sync_all().unwrap();
}

// ---------------------------------------------------------------------------
// CLIF IR generation
// ---------------------------------------------------------------------------
//
// Payload layout (HDR_BASE=0x40 avoids CL_HT_CTX clobbering offset 0):
//
// Header (at v0 = shared.ptr + HDR_BASE):
//   +0x00  data_ptr       i64  absolute pointer to external u32 data
//   +0x08  n              i64  total elements
//   +0x10  workers        i64  worker count
//   +0x20  bins           i64  = 256
//   +0x28  hist_base_rel  i64  offset to per-worker histograms (from v0)
//   +0x30  result_rel     i64  offset to merged result histogram
//   +0x38  thread_ctx     8B   for cl_thread_init
//   +0x40  handles_rel    i64  offset to handles array
//   +0x48  outpath_rel    i64  offset to output file path
//
// Worker descriptor (48 bytes):
//   +0x00  base_ptr    i64  = v0
//   +0x08  data_ptr    i64  absolute pointer to external data
//   +0x10  data_start  i64  first element index
//   +0x18  data_count  i64  elements for this worker
//   +0x20  hist_rel    i64  offset to this worker's histogram (from v0)
//   +0x28  bins        i64  = 256

const HDR_BASE: usize = 0x40;
const HDR_DATA_PTR: usize = HDR_BASE + 0x00;
const HDR_N: usize = HDR_BASE + 0x08;
const HDR_WORKERS: usize = HDR_BASE + 0x10;
const HDR_BINS: usize = HDR_BASE + 0x20;
const HDR_HIST_BASE_REL: usize = HDR_BASE + 0x28;
const HDR_RESULT_REL: usize = HDR_BASE + 0x30;
const HDR_HANDLES_REL: usize = HDR_BASE + 0x40;
const HDR_OUTPATH_REL: usize = HDR_BASE + 0x48;

const WORKER_DESC_SIZE: usize = 48;
const CLIF_IR_OFF: usize = 0x1000;
const CLIF_FLAG_OFF: usize = 0x0F00;

fn gen_clif_ir(chunk_size: usize, hist_stride: usize) -> String {
    let mut ir = String::new();

    // fn0, fn1: noop
    ir.push_str("function u0:0(i64) system_v {\nblock0(v0: i64):\n    return\n}\n\n");
    ir.push_str("function u0:1(i64) system_v {\nblock0(v0: i64):\n    return\n}\n\n");

    // fn2: orchestrator
    ir.push_str(&format!(
        r#"function u0:2(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64, i64) -> i64 system_v
    sig2 = (i64, i64) -> i64 system_v
    sig3 = (i64) system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_thread_init sig0
    fn1 = %cl_thread_spawn sig1
    fn2 = %cl_thread_join sig2
    fn3 = %cl_thread_cleanup sig3
    fn4 = %cl_file_write sig4

block0(v99: i64):
    v0 = iadd_imm v99, 0x40                    ; skip to HDR_BASE
    v1 = load.i64 notrap aligned v0+0x00      ; data_ptr
    v2 = load.i64 notrap aligned v0+0x08      ; n
    v3 = load.i64 notrap aligned v0+0x10      ; workers
    v5 = load.i64 notrap aligned v0+0x20      ; bins
    v6 = load.i64 notrap aligned v0+0x28      ; hist_base_rel
    v7 = load.i64 notrap aligned v0+0x30      ; result_rel
    v8 = load.i64 notrap aligned v0+0x40      ; handles_rel
    v13 = load.i64 notrap aligned v0+0x48     ; outpath_rel

    v9 = iadd v0, v8
    v10 = ishl_imm v3, 3
    v11 = iadd v9, v10                         ; descs_ptr

    v50 = iconst.i64 {chunk}
    v51 = iconst.i64 {desc_sz}
    v52 = iconst.i64 {hist_sz}
    v53 = iconst.i64 3

    v12 = iadd_imm v0, 0x38
    call fn0(v12)

    v60 = iconst.i64 0
    jump block1(v60)

block1(v70: i64):
    v71 = imul v70, v51
    v72 = iadd v11, v71
    store.i64 notrap aligned v0, v72           ; base_ptr
    store.i64 notrap aligned v1, v72+8         ; data_ptr
    v73 = imul v70, v50
    store.i64 notrap aligned v73, v72+16       ; data_start
    v74 = isub v2, v73
    v75 = icmp ult v74, v50
    v76 = select v75, v74, v50
    store.i64 notrap aligned v76, v72+24       ; data_count
    v77 = imul v70, v52
    v78 = iadd v6, v77
    store.i64 notrap aligned v78, v72+32       ; hist_rel
    store.i64 notrap aligned v5, v72+40        ; bins

    v79 = iadd_imm v0, 0x38
    v80 = call fn1(v79, v53, v72)
    v81 = ishl_imm v70, 3
    v82 = iadd v9, v81
    store.i64 notrap aligned v80, v82

    v83 = iadd_imm v70, 1
    v84 = icmp ult v83, v3
    brif v84, block1(v83), block2(v60)

block2(v90: i64):
    v91 = ishl_imm v90, 3
    v92 = iadd v9, v91
    v93 = load.i64 notrap aligned v92
    v94 = iadd_imm v0, 0x38
    v95 = call fn2(v94, v93)
    v96 = iadd_imm v90, 1
    v97 = icmp ult v96, v3
    brif v97, block2(v96), block3(v60)

block3(v100: i64):
    v101 = iconst.i64 0
    jump block4(v60, v101, v100)

block4(v110: i64, v111: i64, v112: i64):
    v113 = imul v110, v52
    v114 = iadd v6, v113
    v115 = ishl_imm v112, 3
    v116 = iadd v114, v115
    v117 = iadd v0, v116
    v118 = load.i64 notrap aligned v117
    v119 = iadd v111, v118
    v120 = iadd_imm v110, 1
    v121 = icmp ult v120, v3
    brif v121, block4(v120, v119, v112), block5(v119, v112)

block5(v130: i64, v131: i64):
    v132 = ishl_imm v131, 3
    v133 = iadd v7, v132
    v134 = iadd v0, v133
    store.i64 notrap aligned v130, v134
    v135 = iadd_imm v131, 1
    v136 = icmp ult v135, v5
    brif v136, block3(v135), block6

block6:
    v141 = iconst.i64 0
    v142 = ishl_imm v5, 3
    v143 = call fn4(v0, v13, v7, v141, v142)
    v140 = iadd_imm v0, 0x38
    call fn3(v140)
    return
}}

"#,
        chunk = chunk_size,
        desc_sz = WORKER_DESC_SIZE,
        hist_sz = hist_stride,
    ));

    // fn3: worker — 4x unrolled histogram scan
    ir.push_str(
        r#"function u0:3(i64) system_v {
block0(v0: i64):
    v1 = load.i64 notrap aligned v0            ; base_ptr
    v2 = load.i64 notrap aligned v0+8          ; data_ptr (absolute)
    v3 = load.i64 notrap aligned v0+16         ; data_start
    v4 = load.i64 notrap aligned v0+24         ; data_count
    v5 = load.i64 notrap aligned v0+32         ; hist_rel
    v6 = load.i64 notrap aligned v0+40         ; bins

    v8 = ishl_imm v3, 2
    v9 = iadd v2, v8                           ; &data[start]
    v10 = iadd v1, v5                          ; hist_ptr

    v11 = ishl_imm v6, 3
    v12 = iadd v10, v11                        ; hist_end
    v13 = ishl_imm v4, 2
    v14 = iadd v9, v13                         ; data_end

    v15 = band_imm v4, -4
    v16 = ishl_imm v15, 2
    v17 = iadd v9, v16                         ; data_end_4x

    v20 = iconst.i64 0
    jump block1(v10)

block1(v21: i64):
    store.i64 notrap aligned v20, v21
    v22 = iadd_imm v21, 8
    v23 = icmp ult v22, v12
    brif v23, block1(v22), block2(v9)

block2(v30: i64):
    v31 = icmp ult v30, v17
    brif v31, block3(v30), block5(v17)

block3(v40: i64):
    v41 = uload32.i64 notrap aligned v40
    v42 = ishl_imm v41, 3
    v43 = iadd v10, v42
    v44 = load.i64 notrap aligned v43
    v45 = iadd_imm v44, 1
    store.i64 notrap aligned v45, v43

    v50 = uload32.i64 notrap aligned v40+4
    v51 = ishl_imm v50, 3
    v52 = iadd v10, v51
    v53 = load.i64 notrap aligned v52
    v54 = iadd_imm v53, 1
    store.i64 notrap aligned v54, v52

    v60 = uload32.i64 notrap aligned v40+8
    v61 = ishl_imm v60, 3
    v62 = iadd v10, v61
    v63 = load.i64 notrap aligned v62
    v64 = iadd_imm v63, 1
    store.i64 notrap aligned v64, v62

    v70 = uload32.i64 notrap aligned v40+12
    v71 = ishl_imm v70, 3
    v72 = iadd v10, v71
    v73 = load.i64 notrap aligned v72
    v74 = iadd_imm v73, 1
    store.i64 notrap aligned v74, v72

    v80 = iadd_imm v40, 16
    v81 = icmp ult v80, v17
    brif v81, block3(v80), block5(v17)

block5(v90: i64):
    v91 = icmp ult v90, v14
    brif v91, block6(v90), block7

block6(v100: i64):
    v101 = uload32.i64 notrap aligned v100
    v102 = ishl_imm v101, 3
    v103 = iadd v10, v102
    v104 = load.i64 notrap aligned v103
    v105 = iadd_imm v104, 1
    store.i64 notrap aligned v105, v103
    v106 = iadd_imm v100, 4
    v107 = icmp ult v106, v14
    brif v107, block6(v106), block7

block7:
    return
}
"#,
    );

    ir
}

// ---------------------------------------------------------------------------
// Algorithm builder
// ---------------------------------------------------------------------------

fn build_algorithm(data: &[u32], workers: usize, out_path: &str) -> base::Algorithm {
    let n = data.len();
    let chunk_size = (n + workers - 1) / workers;

    let hist_stride = (BINS * 8 + 4095) & !4095; // 4KB per worker — avoids false sharing
    let clif_source = gen_clif_ir(chunk_size, hist_stride);
    let clif_bytes = format!("{}\0", clif_source).into_bytes();
    let path_bytes = format!("{}\0", out_path).into_bytes();

    // Data is NOT in the payload — we store an absolute pointer to the caller's slice.
    let ir_end = CLIF_IR_OFF + ((clif_bytes.len() + 63) & !63);
    let hist_off = (ir_end + 4095) & !4095;
    let result_off = hist_off + ((workers * hist_stride + 63) & !63);
    let handles_off = result_off + ((BINS * 8 + 63) & !63);
    let descs_off = handles_off + workers * 8;
    let outpath_off = descs_off + ((workers * WORKER_DESC_SIZE + 63) & !63);
    let payload_size = outpath_off + ((path_bytes.len() + 63) & !63);

    let mut payloads = vec![0u8; payload_size];

    payloads[HDR_DATA_PTR..HDR_DATA_PTR + 8]
        .copy_from_slice(&(data.as_ptr() as i64).to_le_bytes());
    payloads[HDR_N..HDR_N + 8].copy_from_slice(&(n as i64).to_le_bytes());
    payloads[HDR_WORKERS..HDR_WORKERS + 8].copy_from_slice(&(workers as i64).to_le_bytes());
    payloads[HDR_BINS..HDR_BINS + 8].copy_from_slice(&(BINS as i64).to_le_bytes());
    payloads[HDR_HIST_BASE_REL..HDR_HIST_BASE_REL + 8]
        .copy_from_slice(&((hist_off - HDR_BASE) as i64).to_le_bytes());
    payloads[HDR_RESULT_REL..HDR_RESULT_REL + 8]
        .copy_from_slice(&((result_off - HDR_BASE) as i64).to_le_bytes());
    payloads[HDR_HANDLES_REL..HDR_HANDLES_REL + 8]
        .copy_from_slice(&((handles_off - HDR_BASE) as i64).to_le_bytes());
    payloads[HDR_OUTPATH_REL..HDR_OUTPATH_REL + 8]
        .copy_from_slice(&((outpath_off - HDR_BASE) as i64).to_le_bytes());
    payloads[outpath_off..outpath_off + path_bytes.len()].copy_from_slice(&path_bytes);

    payloads[CLIF_IR_OFF..CLIF_IR_OFF + clif_bytes.len()].copy_from_slice(&clif_bytes);

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 2, offset: 0, size: 0 },
    ];

    base::Algorithm {
        actions,
        payloads,
        state: State { cranelift_ir_offsets: vec![CLIF_IR_OFF] },
        units: UnitSpec { cranelift_units: 0 },
        cranelift_assignments: vec![],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(512 * 1024),
        timeout_ms: Some(60_000),
        thread_name_prefix: Some("hist-bench".into()),
        additional_shared_memory: 0,
    }
}

fn read_result(path: &str) -> Option<Vec<u64>> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; BINS * 8];
    file.read_exact(&mut buf).ok()?;
    Some(buf.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect())
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
// Benchmark entry point
// ---------------------------------------------------------------------------

pub struct HistConfig {
    pub rounds: usize,
    pub workers: usize,
}

pub fn run(cfg: &HistConfig) -> Vec<BenchResult> {
    let mut results = Vec::new();

    eprintln!("\n=== Parallel Histogram Benchmark ===");
    eprintln!("  workers={}, bins={}", cfg.workers, BINS);

    for &n in &[1_000_000, 5_000_000, 10_000_000] {
        let data = Arc::new(gen_data(n, 42));
        let expected = reference_histogram(&data);

        let mut worker_counts = vec![1, 2, 4];
        worker_counts.retain(|&w| w <= cfg.workers);

        for &w in &worker_counts {
            let label = format!("hist n={} w={}", format_count(n), w);
            let rust_out = format!("/tmp/hist_bench_rust_{}_{}.bin", n, w);
            let clif_out = format!("/tmp/hist_bench_clif_{}_{}.bin", n, w);

            let clif_alg = build_algorithm(&data, w, &clif_out);
            let _ = base::execute(clif_alg.clone()); // warmup JIT

            let rust_ms = harness::median_of(cfg.rounds, || {
                let start = std::time::Instant::now();
                rust_parallel_histogram(&data, w, &rust_out);
                start.elapsed().as_secs_f64() * 1000.0
            });

            let mut clif_ok = true;
            let clif_ms = harness::median_of(cfg.rounds, || {
                let alg = clif_alg.clone();
                let start = std::time::Instant::now();
                if base::execute(alg).is_err() {
                    clif_ok = false;
                }
                start.elapsed().as_secs_f64() * 1000.0
            });

            let rust_ok = read_result(&rust_out)
                .map(|h| h == expected)
                .unwrap_or(false);
            let clif_verified = clif_ok
                && read_result(&clif_out)
                    .map(|h| h == expected)
                    .unwrap_or(false);

            let _ = std::fs::remove_file(&rust_out);
            let _ = std::fs::remove_file(&clif_out);

            if !rust_ok {
                eprintln!("  VERIFY FAIL: Rust {}", label);
            }
            if !clif_verified {
                eprintln!("  VERIFY FAIL: CLIF {}", label);
            }

            results.push(BenchResult {
                name: label,
                python_ms: None,
                rust_ms: Some(rust_ms),
                base_ms: clif_ms,
                verified: Some(rust_ok && clif_verified),
                actions: None,
            });
        }
    }

    results
}
