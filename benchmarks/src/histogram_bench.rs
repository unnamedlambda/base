use base_types::{Action, Kind};
use crate::harness::{self, BenchResult};

// ---------------------------------------------------------------------------
// Parallel Histogram Benchmark
//
// Both Rust and Base read the same input file of random u32 values in [0, BINS),
// compute a parallel histogram with W workers, and write the result to a file.
// The harness generates the input file at various sizes and compares outputs.
// ---------------------------------------------------------------------------

const BINS: usize = 256;
const MAX_DATA_BYTES: usize = 64 * 1024 * 1024; // 64MB max input file

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
// Rust baseline — reads from file, parallel histogram, writes result to file
// ---------------------------------------------------------------------------

fn rust_histogram(file_buf: &mut Vec<u8>, input_path: &str, output_path: &str, workers: usize) {
    use std::io::Read;
    let mut file = std::fs::File::open(input_path).unwrap();
    let file_len = file.metadata().unwrap().len() as usize;
    if file_buf.len() < file_len {
        file_buf.resize(file_len, 0);
    }
    file.read_exact(&mut file_buf[..file_len]).unwrap();

    assert!(file_len % 4 == 0);
    let n = file_len / 4;
    let data = unsafe { std::slice::from_raw_parts(file_buf.as_ptr() as *const u32, n) };

    let hist = if workers == 1 {
        let mut hist = vec![0u64; BINS];
        for &v in data {
            hist[v as usize] += 1;
        }
        hist
    } else {
        let chunk_size = (n + workers - 1) / workers;
        let mut handles = Vec::with_capacity(workers);
        let base_ptr = data.as_ptr() as usize;

        for w in 0..workers {
            let start = (w * chunk_size).min(n);
            let end = ((w + 1) * chunk_size).min(n);
            let ptr = base_ptr;
            handles.push(std::thread::spawn(move || {
                let slice = unsafe { std::slice::from_raw_parts(ptr as *const u32, end) };
                let mut local = vec![0u64; BINS];
                for &v in &slice[start..end] {
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
        merged
    };

    let mut result_buf = vec![0u8; BINS * 8];
    for (i, &count) in hist.iter().enumerate() {
        result_buf[i * 8..(i + 1) * 8].copy_from_slice(&count.to_le_bytes());
    }
    std::fs::write(output_path, &result_buf).unwrap();
}

// ---------------------------------------------------------------------------
// CLIF IR generation
// ---------------------------------------------------------------------------
//
// Payload (via execute data arg): "input_path\0output_path\0"
//
// Shared memory layout:
//   0x00-0x27:  reserved (runtime writes ctx_ptr, data_ptr, data_len, out_ptr, out_len)
//   0x100:      input_path    256 bytes (copied from payload by CLIF)
//   0x200:      output_path   256 bytes (copied from payload by CLIF)
//   0x300:      thread_ctx    8 bytes (multi-threaded only)
//   0x400:      hist region(s), then result/handles/descriptors (layout varies by w)
//   end:        data region   (MAX_DATA_BYTES)
//
// Worker descriptor (48 bytes, multi-threaded only):
//   +0x00  base_ptr, +0x08 data_off, +0x10 data_start, +0x18 data_count,
//   +0x20  hist_off, +0x28 bins

const INPUT_PATH_OFF: usize = 0x100;
const OUTPUT_PATH_OFF: usize = 0x200;
const THREAD_CTX_OFF: usize = 0x300;
const HIST_REGION_OFF: usize = 0x400;
const WORKER_DESC_SIZE: usize = 48;

fn compute_layout(workers: usize) -> (usize, usize, usize, usize, usize) {
    let hist_stride = (BINS * 8 + 4095) & !4095; // 4KB aligned per worker
    let hist_end = HIST_REGION_OFF + workers * hist_stride;
    let result_off = hist_end;
    let result_end = result_off + BINS * 8;
    let handles_off = (result_end + 63) & !63;
    let handles_end = handles_off + workers * 8;
    let descs_off = (handles_end + 63) & !63;
    let descs_end = descs_off + workers * WORKER_DESC_SIZE;
    let data_off = (descs_end + 63) & !63;
    let mem_size = data_off + MAX_DATA_BYTES;
    // Return: hist_stride, result_off, handles_off, descs_off, data_off
    // (mem_size is derived from data_off + MAX_DATA_BYTES)
    let _ = mem_size;
    (hist_stride, result_off, handles_off, descs_off, data_off)
}

fn gen_clif_ir(workers: usize) -> String {
    if workers == 1 {
        return gen_clif_ir_single();
    }
    gen_clif_ir_multi(workers)
}

/// Single-threaded CLIF: read file → histogram inline → write result.
/// No thread FFI calls at all.
fn gen_clif_ir_single() -> String {
    let data_off = HIST_REGION_OFF + BINS * 8; // hist at HIST_REGION_OFF, then data

    let mut ir = String::new();

    // fn0: noop
    ir.push_str("function u0:0(i64) system_v {\nblock0(v0: i64):\n    return\n}\n\n");
    // fn1: noop
    ir.push_str("function u0:1(i64) system_v {\nblock0(v0: i64):\n    return\n}\n\n");

    // fn2: single-threaded orchestrator — all in one function
    ir.push_str(&format!(
        r#"function u0:2(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    sig1 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_file_read sig0
    fn1 = %cl_file_write sig1

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08          ; data_ptr (payload)
    v200 = iconst.i64 0
    jump block1(v200)

    ; --- Copy input path from payload to shared memory ---
block1(v201: i64):
    v202 = iadd v1, v201
    v203 = uload8.i64 notrap v202
    v204 = iadd_imm v0, {input_path_off}
    v205 = iadd v204, v201
    istore8 v203, v205
    v206 = icmp_imm eq v203, 0
    v207 = iadd_imm v201, 1
    brif v206, block2(v207), block1(v207)

    ; --- Copy output path from payload to shared memory ---
block2(v210: i64):
    v211 = iconst.i64 0
    jump block3(v210, v211)

block3(v220: i64, v221: i64):
    v222 = iadd v1, v220
    v223 = uload8.i64 notrap v222
    v224 = iadd_imm v0, {output_path_off}
    v225 = iadd v224, v221
    istore8 v223, v225
    v226 = icmp_imm eq v223, 0
    v227 = iadd_imm v220, 1
    v228 = iadd_imm v221, 1
    brif v226, block4, block3(v227, v228)

    ; --- Read file, zero histogram ---
block4:
    v3 = iconst.i64 {input_path_off}
    v4 = iconst.i64 {data_off}
    v5 = iconst.i64 0
    v6 = iconst.i64 0
    v7 = call fn0(v0, v3, v4, v5, v6)              ; bytes_read
    v8 = ushr_imm v7, 2                            ; n = bytes_read / 4

    v10 = iadd_imm v0, {hist_off}                  ; hist_ptr
    v11 = iconst.i64 {hist_bytes}
    v12 = iadd v10, v11                            ; hist_end
    jump block5(v10)

    ; --- Zero histogram (8x unrolled) ---
block5(v21: i64):
    store.i64 notrap aligned v200, v21
    store.i64 notrap aligned v200, v21+8
    store.i64 notrap aligned v200, v21+16
    store.i64 notrap aligned v200, v21+24
    store.i64 notrap aligned v200, v21+32
    store.i64 notrap aligned v200, v21+40
    store.i64 notrap aligned v200, v21+48
    store.i64 notrap aligned v200, v21+56
    v22 = iadd_imm v21, 64
    v23 = icmp ult v22, v12
    brif v23, block5(v22), block6

    ; --- Prepare scan ---
block6:
    v30 = iadd_imm v0, {data_off}                  ; data_ptr
    v31 = ishl_imm v8, 2
    v32 = iadd v30, v31                            ; data_end
    v33 = band_imm v8, -4
    v34 = ishl_imm v33, 2
    v35 = iadd v30, v34                            ; data_end_4x
    v36 = icmp ult v30, v35
    brif v36, block7(v30), block8(v30)

    ; --- 4x unrolled histogram scan ---
block7(v40: i64):
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
    v81 = icmp ult v80, v35
    brif v81, block7(v80), block8(v35)

    ; --- Scalar tail ---
block8(v90: i64):
    v91 = icmp ult v90, v32
    brif v91, block9(v90), block10

block9(v100: i64):
    v101 = uload32.i64 notrap aligned v100
    v102 = ishl_imm v101, 3
    v103 = iadd v10, v102
    v104 = load.i64 notrap aligned v103
    v105 = iadd_imm v104, 1
    store.i64 notrap aligned v105, v103
    v106 = iadd_imm v100, 4
    v107 = icmp ult v106, v32
    brif v107, block9(v106), block10

    ; --- Write result ---
block10:
    v95 = iconst.i64 {output_path_off}
    v96 = iconst.i64 {hist_off}
    v97 = iconst.i64 0
    v98 = iconst.i64 {result_size}
    v99 = call fn1(v0, v95, v96, v97, v98)
    return
}}

"#,
        input_path_off = INPUT_PATH_OFF,
        output_path_off = OUTPUT_PATH_OFF,
        data_off = data_off,
        hist_off = HIST_REGION_OFF,
        hist_bytes = BINS * 8,
        result_size = BINS * 8,
    ));

    ir
}

fn gen_clif_ir_multi(workers: usize) -> String {
    let (hist_stride, result_off, handles_off, descs_off, data_off) =
        compute_layout(workers);

    let mut ir = String::new();

    // fn0: noop
    ir.push_str("function u0:0(i64) system_v {\nblock0(v0: i64):\n    return\n}\n\n");
    // fn1: noop
    ir.push_str("function u0:1(i64) system_v {\nblock0(v0: i64):\n    return\n}\n\n");

    // fn2: orchestrator
    //  1. Copy input/output paths from payload into shared memory
    //  2. cl_file_read input data
    //  3. Compute n, chunk_size
    //  4. Spawn workers, join, merge histograms
    //  5. cl_file_write result
    // Block layout for orchestrator (fn2):
    //   block0:  entry — load data_ptr, jump to copy loop
    //   block1:  copy input path byte-by-byte
    //   block2:  copy output path setup
    //   block3:  copy output path byte-by-byte
    //   block4:  read file, compute n/chunk, init threads, jump to spawn
    //   block5:  spawn loop
    //   block6:  join loop
    //   block7:  merge outer (per bin)
    //   block8:  merge inner (per worker)
    //   block9:  merge store
    //   block10: write result + cleanup
    ir.push_str(&format!(
        r#"function u0:2(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64, i64) -> i64 system_v
    sig2 = (i64, i64) -> i64 system_v
    sig3 = (i64) system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_thread_init sig0
    fn1 = %cl_thread_spawn sig1
    fn2 = %cl_thread_join sig2
    fn3 = %cl_thread_cleanup sig3
    fn4 = %cl_file_read sig5
    fn5 = %cl_file_write sig4

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08          ; data_ptr (payload)
    v200 = iconst.i64 0
    jump block1(v200)

    ; --- Copy input path from payload to shared memory ---
block1(v201: i64):
    v202 = iadd v1, v201
    v203 = uload8.i64 notrap v202
    v204 = iadd_imm v0, {input_path_off}
    v205 = iadd v204, v201
    istore8 v203, v205
    v206 = icmp_imm eq v203, 0
    v207 = iadd_imm v201, 1
    brif v206, block2(v207), block1(v207)

    ; --- Copy output path from payload to shared memory ---
block2(v210: i64):
    v211 = iconst.i64 0
    jump block3(v210, v211)

block3(v220: i64, v221: i64):
    v222 = iadd v1, v220
    v223 = uload8.i64 notrap v222
    v224 = iadd_imm v0, {output_path_off}
    v225 = iadd v224, v221
    istore8 v223, v225
    v226 = icmp_imm eq v223, 0
    v227 = iadd_imm v220, 1
    v228 = iadd_imm v221, 1
    brif v226, block4, block3(v227, v228)

    ; --- Read file, compute n/chunk, init threads ---
block4:
    v3 = iconst.i64 {input_path_off}
    v4 = iconst.i64 {data_off}
    v5 = iconst.i64 0
    v6 = iconst.i64 0
    v7 = call fn4(v0, v3, v4, v5, v6)              ; bytes_read

    v8 = ushr_imm v7, 2                            ; n = bytes_read / 4
    v9 = iconst.i64 {workers}
    v10 = iadd_imm v8, {workers_minus_1}
    v11 = udiv v10, v9                              ; chunk_size

    v12 = iconst.i64 {bins}
    v13 = iconst.i64 {hist_stride}

    v14 = iadd_imm v0, {thread_ctx_off}
    call fn0(v14)

    v16 = iconst.i64 3                              ; fn_index for worker
    v17 = iconst.i64 {descs_off}
    v18 = iconst.i64 {desc_sz}
    v19 = iconst.i64 {handles_off}
    v20 = iconst.i64 {data_off}
    jump block5(v200, v8, v11)

    ; --- Spawn workers ---
block5(v30: i64, v31: i64, v32: i64):
    v33 = imul v30, v18
    v34 = iadd v0, v17
    v35 = iadd v34, v33                             ; desc_ptr

    store.i64 notrap aligned v0, v35                ; base_ptr
    store.i64 notrap aligned v20, v35+0x08          ; data_off
    v36 = imul v30, v32
    store.i64 notrap aligned v36, v35+0x10          ; data_start
    v37 = isub v31, v36
    v38 = icmp ult v37, v32
    v39 = select v38, v37, v32
    store.i64 notrap aligned v39, v35+0x18          ; data_count
    v40 = imul v30, v13
    v41 = iadd_imm v40, {hist_region_off}
    store.i64 notrap aligned v41, v35+0x20          ; hist_off
    store.i64 notrap aligned v12, v35+0x28          ; bins

    v42 = iadd_imm v0, {thread_ctx_off}
    v43 = call fn1(v42, v16, v35)

    v44 = ishl_imm v30, 3
    v45 = iadd v0, v19
    v46 = iadd v45, v44
    store.i64 notrap aligned v43, v46

    v47 = iadd_imm v30, 1
    v48 = icmp ult v47, v9
    brif v48, block5(v47, v31, v32), block6(v200)

    ; --- Join workers ---
block6(v50: i64):
    v51 = ishl_imm v50, 3
    v52 = iadd v0, v19
    v53 = iadd v52, v51
    v54 = load.i64 notrap aligned v53
    v55 = iadd_imm v0, {thread_ctx_off}
    v56 = call fn2(v55, v54)
    v57 = iadd_imm v50, 1
    v58 = icmp ult v57, v9
    brif v58, block6(v57), block7(v200)

    ; --- Merge histograms: outer loop over bins ---
block7(v60: i64):
    v61 = iconst.i64 0
    jump block8(v200, v61, v60)

    ; --- Merge: inner loop over workers for one bin ---
block8(v70: i64, v71: i64, v72: i64):
    v73 = imul v70, v13
    v74 = iadd_imm v73, {hist_region_off}
    v75 = ishl_imm v72, 3
    v76 = iadd v74, v75
    v77 = iadd v0, v76
    v78 = load.i64 notrap aligned v77
    v79 = iadd v71, v78
    v80 = iadd_imm v70, 1
    v81 = icmp ult v80, v9
    brif v81, block8(v80, v79, v72), block9(v79, v72)

    ; --- Store merged bin count ---
block9(v85: i64, v86: i64):
    v87 = ishl_imm v86, 3
    v88 = iadd_imm v87, {result_off}
    v89 = iadd v0, v88
    store.i64 notrap aligned v85, v89
    v90 = iadd_imm v86, 1
    v91 = icmp ult v90, v12
    brif v91, block7(v90), block10

    ; --- Write result and cleanup ---
block10:
    v95 = iconst.i64 {output_path_off}
    v96 = iconst.i64 {result_off}
    v97 = iconst.i64 0
    v98 = iconst.i64 {result_size}
    v99 = call fn5(v0, v95, v96, v97, v98)

    v100 = iadd_imm v0, {thread_ctx_off}
    call fn3(v100)
    return
}}

"#,
        input_path_off = INPUT_PATH_OFF,
        output_path_off = OUTPUT_PATH_OFF,
        data_off = data_off,
        workers = workers,
        workers_minus_1 = workers - 1,
        bins = BINS,
        hist_stride = hist_stride,
        thread_ctx_off = THREAD_CTX_OFF,
        descs_off = descs_off,
        desc_sz = WORKER_DESC_SIZE,
        handles_off = handles_off,
        hist_region_off = HIST_REGION_OFF,
        result_off = result_off,
        result_size = BINS * 8,
    ));

    // fn3: worker — 4x unrolled histogram scan
    ir.push_str(
        r#"function u0:3(i64) system_v {
block0(v0: i64):
    v1 = load.i64 notrap aligned v0            ; base_ptr
    v2 = load.i64 notrap aligned v0+8          ; data_off
    v3 = load.i64 notrap aligned v0+16         ; data_start
    v4 = load.i64 notrap aligned v0+24         ; data_count
    v5 = load.i64 notrap aligned v0+32         ; hist_off (from base)
    v6 = load.i64 notrap aligned v0+40         ; bins

    ; data_ptr = base + data_off + start*4
    v7 = iadd v1, v2
    v8 = ishl_imm v3, 2
    v9 = iadd v7, v8                           ; &data[start]

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

    ; --- Zero histogram (8x unrolled) ---
block1(v21: i64):
    store.i64 notrap aligned v20, v21
    store.i64 notrap aligned v20, v21+8
    store.i64 notrap aligned v20, v21+16
    store.i64 notrap aligned v20, v21+24
    store.i64 notrap aligned v20, v21+32
    store.i64 notrap aligned v20, v21+40
    store.i64 notrap aligned v20, v21+48
    store.i64 notrap aligned v20, v21+56
    v22 = iadd_imm v21, 64
    v23 = icmp ult v22, v12
    brif v23, block1(v22), block2(v9)

    ; --- 4x unrolled scan ---
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

    ; --- Scalar tail ---
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

fn build_algorithm(workers: usize) -> (base::BaseConfig, base::Algorithm) {
    let clif_source = gen_clif_ir(workers);

    let mem_size = if workers == 1 {
        // Single-threaded layout: hist at HIST_REGION_OFF, data right after
        let data_off = HIST_REGION_OFF + BINS * 8;
        data_off + MAX_DATA_BYTES
    } else {
        let (_hist_stride, _result_off, _handles_off, _descs_off, data_off) =
            compute_layout(workers);
        data_off + MAX_DATA_BYTES
    };

    let config = base::BaseConfig {
        cranelift_ir: clif_source,
        memory_size: mem_size,
        context_offset: 0,
        initial_memory: vec![],
    };
    let algorithm = base::Algorithm {
        actions: vec![
            Action { kind: Kind::ClifCall, dst: 0, src: 2, offset: 0, size: 0 },
        ],
        cranelift_units: 0,
        timeout_ms: Some(60_000),
        output: vec![],
    };
    (config, algorithm)
}

fn read_result(path: &str) -> Option<Vec<u64>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != BINS * 8 { return None; }
    Some(bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect())
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
    let input_path = "/tmp/histogram_bench_input.bin";

    eprintln!("\n=== Parallel Histogram Benchmark ===");
    eprintln!("  workers={}, bins={}", cfg.workers, BINS);

    for &n in &[1_000_000, 10_000_000] {
        // Write input file
        let data = gen_data(n, 42);
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(input_path, &data_bytes).unwrap();
        let expected = reference_histogram(&data);

        let mut worker_counts = vec![1, 4];
        worker_counts.retain(|&w| w <= cfg.workers);

        for &w in &worker_counts {
            let label = format!("hist n={} w={}", format_count(n), w);
            let rust_out = format!("/tmp/hist_bench_rust_{}_{}.bin", n, w);
            let base_out = format!("/tmp/hist_bench_base_{}_{}.bin", n, w);

            // Build payload: "input_path\0output_path\0"
            let mut base_payload = Vec::new();
            base_payload.extend_from_slice(input_path.as_bytes());
            base_payload.push(0);
            base_payload.extend_from_slice(base_out.as_bytes());
            base_payload.push(0);

            // Build Base algorithm and JIT compile once
            let (config, alg) = build_algorithm(w);
            let mut base_instance = base::Base::new(config).expect("Base::new failed");

            // Pre-allocated file read buffer — same as Base's shared memory region
            let mut file_buf = vec![0u8; MAX_DATA_BYTES];

            // Warmup both
            rust_histogram(&mut file_buf, input_path, &rust_out, w);
            let _ = base_instance.execute(&alg, &base_payload);

            let rust_ms = harness::median_of(cfg.rounds, || {
                let start = std::time::Instant::now();
                rust_histogram(&mut file_buf, input_path, &rust_out, w);
                start.elapsed().as_secs_f64() * 1000.0
            });

            let mut base_ok = true;
            let base_ms = harness::median_of(cfg.rounds, || {
                let start = std::time::Instant::now();
                if base_instance.execute(&alg, &base_payload).is_err() {
                    base_ok = false;
                }
                start.elapsed().as_secs_f64() * 1000.0
            });

            let rust_verified = read_result(&rust_out)
                .map(|h| h == expected)
                .unwrap_or(false);
            let base_verified = base_ok
                && read_result(&base_out)
                    .map(|h| h == expected)
                    .unwrap_or(false);

            let _ = std::fs::remove_file(&rust_out);
            let _ = std::fs::remove_file(&base_out);

            if !rust_verified {
                eprintln!("  VERIFY FAIL: Rust {}", label);
            }
            if !base_verified {
                eprintln!("  VERIFY FAIL: Base {}", label);
            }

            results.push(BenchResult {
                name: label,
                python_ms: None,
                rust_ms: Some(rust_ms),
                base_ms,
                verified: Some(rust_verified && base_verified),
                actions: None,
            });
        }
    }

    let _ = std::fs::remove_file(input_path);
    results
}
