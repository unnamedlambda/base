use crate::harness::{self, gen_floats, BenchResult};
use base::Artifact;

// ---------------------------------------------------------------------------
// Codegen Benchmark — Cranelift vs LLVM on the SAME instruction sequence
//
// Every other benchmark here compares Base's algorithm against Rust's, so a
// win or a loss is mostly a statement about the algorithms.  This one holds
// the algorithm fixed: `simd_clamp_sum` is a hand-written mirror of the CLIF
// in `ClampSumBenchAlgorithm.lean` -- same 4x unroll, same four f32x4
// accumulators, same min/max order, same pairwise merge, same promote-and-add
// horizontal reduce, same scalar tail.  Both sides therefore compute the same
// value in the same order, and the results are checked BIT-EXACT rather than
// within a tolerance: if they differ at all, the mirror is wrong and the
// comparison is not measuring what it claims to.
//
// What is left over is instruction selection, register allocation and
// scheduling -- the part of the LLVM gap that emitting better CLIF cannot fix.
//
// Five sweeps, because one alone does not say where a gap lives:
//
//   clamp  the loop as originally emitted, with CLIF `fmin`/`fmax`
//   plain  the same loop with the two clamps deleted and nothing else changed
//   pmin   the clamp again, in the shape Cranelift folds into one `minps`
//   regs16 sixteen live accumulators, so both allocators must spill
//   int    four integer chains: no vector unit, different lowering path
//   store  writes as much as it reads: store ports, addressing modes
//   branchy a serial chain with an unpredictable data-dependent branch
//   select  the same chain written branchlessly
//   sel+lea and with `h * 3` strength-reduced to a shift-add
//   sel+rot and with the loop rotated: no header block, no unconditional jmp
//   sel+mask and with the redundant per-step masking hoisted out of the loop
//
// `scalar_clamp_sum` is kept for contrast.  It is what the other benchmarks
// put on the Rust side, and LLVM cannot vectorise it (fp addition is not
// associative), so the distance between the two Rust rows is the part of any
// "Base beats Rust" result that has nothing to do with the backend.
// ---------------------------------------------------------------------------

const CLAMP_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/clamp_sum_algorithm");

/// The same loop with the two `fmin`/`fmax` removed and nothing else changed.
/// CLIF's `fmin` is IEEE minimumNumber, which x86 `minps` does not implement,
/// so Cranelift has to emit a NaN-correct sequence rather than one
/// instruction.  The distance between these two artifacts prices that.
const PLAIN_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/plain_sum_algorithm");


/// The clamp again, emitted as `bitselect(bitcast(fcmp lt ..), ..)` -- the
/// shape Cranelift has a rule to fold into one `minps`.  Same arithmetic as
/// `clamp` for non-NaN input, so it must still agree bit for bit.
const PMIN_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/pmin_sum_algorithm");

const REGP_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/regpressure_sum_algorithm");

const INT_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/intsum_algorithm");

const STORE_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/store_algorithm");

const BRANCHY_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/branchy_algorithm");

const SELECT_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/select_algorithm");

const SELLEA_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/selectlea_algorithm");

const SELROT_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/selectrot_algorithm");

const SELMASK_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/selectmask_algorithm");


const HI: f32 = 0.5;
const LO: f32 = -0.5;

#[cfg(target_arch = "x86_64")]
unsafe fn simd_plain_sum(data: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let n = data.len();
    let p = data.as_ptr();
    let (mut a, mut b, mut c, mut d) = (
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
    );
    let main_end = (n / 16) * 16;
    let mut i = 0;
    while i < main_end {
        a = _mm_add_ps(a, _mm_loadu_ps(p.add(i)));
        b = _mm_add_ps(b, _mm_loadu_ps(p.add(i + 4)));
        c = _mm_add_ps(c, _mm_loadu_ps(p.add(i + 8)));
        d = _mm_add_ps(d, _mm_loadu_ps(p.add(i + 12)));
        i += 16;
    }
    let mut acc = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d));
    let simd_end = (n / 4) * 4;
    while i < simd_end {
        acc = _mm_add_ps(acc, _mm_loadu_ps(p.add(i)));
        i += 4;
    }
    let mut lanes = [0f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = (lanes[0] as f64 + lanes[1] as f64) + (lanes[2] as f64 + lanes[3] as f64);
    while i < n {
        s += data[i] as f64;
        i += 1;
    }
    s
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_plain_sum(data: &[f32]) -> f64 {
    data.iter().map(|&x| x as f64).sum()
}

/// The same reduction at 256 bits.  CLIF's vector types are fixed at 128 bits
/// (`f32x4`, `i8x16`), so this is not something Base can be asked to emit --
/// it measures a ceiling, not a lowering, and the values differ because the
/// accumulator geometry differs.  Note it is only reachable by a build that
/// opts into AVX: a stock `cargo build --release` targets x86-64 baseline,
/// which is SSE2, and at that setting 128 bits costs nothing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd256_plain_sum(data: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let n = data.len();
    let p = data.as_ptr();
    let (mut a, mut b, mut c, mut d) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let main_end = (n / 32) * 32;
    let mut i = 0;
    while i < main_end {
        a = _mm256_add_ps(a, _mm256_loadu_ps(p.add(i)));
        b = _mm256_add_ps(b, _mm256_loadu_ps(p.add(i + 8)));
        c = _mm256_add_ps(c, _mm256_loadu_ps(p.add(i + 16)));
        d = _mm256_add_ps(d, _mm256_loadu_ps(p.add(i + 24)));
        i += 32;
    }
    let acc = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d));
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = 0.0f64;
    for l in lanes {
        s += l as f64;
    }
    while i < n {
        s += data[i] as f64;
        i += 1;
    }
    s
}

/// Sixteen live f32x4 accumulators -- more than x86-64 has xmm registers once
/// the pointer and counter are accounted for, so both allocators must spill.
#[cfg(target_arch = "x86_64")]
unsafe fn simd_regp_sum(data: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let n = data.len();
    let p = data.as_ptr();
    let mut a = [_mm_setzero_ps(); 16];
    let main_end = (n / 64) * 64;
    let mut i = 0;
    while i < main_end {
        for k in 0..16 {
            a[k] = _mm_add_ps(a[k], _mm_loadu_ps(p.add(i + k * 4)));
        }
        i += 64;
    }
    // left fold, matching the CLIF
    let mut acc = a[0];
    for k in 1..16 {
        acc = _mm_add_ps(acc, a[k]);
    }
    let mut lanes = [0f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), acc);
    (lanes[0] as f64 + lanes[1] as f64) + (lanes[2] as f64 + lanes[3] as f64)
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_regp_sum(data: &[f32]) -> f64 {
    scalar_regp_sum(data)
}

/// Same reduction, one accumulator: what a straightforward version looks like.
fn scalar_regp_sum(data: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &x in &data[..(data.len() / 64) * 64] {
        s += x as f64;
    }
    s
}

/// Four independent `h = (h * 31 + (x & 0xFFFF)) & 0xFFFFFF` chains over the
/// same bytes read as i32 words.  No vector unit, no floating point.
fn int_sum(data: &[f32]) -> f64 {
    let words =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()) };
    let mut h = [1u64; 4];
    let main_end = (words.len() / 4) * 4;
    let mut i = 0;
    while i < main_end {
        for k in 0..4 {
            h[k] = (h[k].wrapping_mul(31) + (words[i + k] as u64 & 0xFFFF)) & 0xFF_FFFF;
        }
        i += 4;
    }
    let mut tot = h[0];
    for k in 1..4 {
        tot += h[k];
    }
    tot as f64
}

/// Strict left-to-right accumulation into f64: unvectorisable by construction.
fn scalar_clamp_sum(data: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &x in data {
        s += x.min(HI).max(LO) as f64;
    }
    s
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_clamp_sum(data: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let n = data.len();
    let p = data.as_ptr();
    let hi = _mm_set1_ps(HI);
    let lo = _mm_set1_ps(LO);
    let (mut a, mut b, mut c, mut d) = (
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
    );

    // main loop: 16 elements per trip, four independent accumulators
    let main_end = (n / 16) * 16;
    let mut i = 0;
    while i < main_end {
        let v0 = _mm_loadu_ps(p.add(i));
        let v1 = _mm_loadu_ps(p.add(i + 4));
        let v2 = _mm_loadu_ps(p.add(i + 8));
        let v3 = _mm_loadu_ps(p.add(i + 12));
        a = _mm_add_ps(a, _mm_max_ps(_mm_min_ps(v0, hi), lo));
        b = _mm_add_ps(b, _mm_max_ps(_mm_min_ps(v1, hi), lo));
        c = _mm_add_ps(c, _mm_max_ps(_mm_min_ps(v2, hi), lo));
        d = _mm_add_ps(d, _mm_max_ps(_mm_min_ps(v3, hi), lo));
        i += 16;
    }
    // merge: (a+b) + (c+d), matching the CLIF's association
    let mut acc = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d));

    // remaining whole vectors, single accumulator
    let simd_end = (n / 4) * 4;
    while i < simd_end {
        let v = _mm_loadu_ps(p.add(i));
        acc = _mm_add_ps(acc, _mm_max_ps(_mm_min_ps(v, hi), lo));
        i += 4;
    }

    // horizontal reduce: promote each lane, then (l0+l1) + (l2+l3)
    let mut lanes = [0f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = (lanes[0] as f64 + lanes[1] as f64) + (lanes[2] as f64 + lanes[3] as f64);

    // scalar tail
    while i < n {
        s += data[i].min(HI).max(LO) as f64;
        i += 1;
    }
    s
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_clamp_sum(data: &[f32]) -> f64 {
    scalar_clamp_sum(data)
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn scalar_plain_sum(data: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &x in data {
        s += x as f64;
    }
    s
}

/// One artifact against its hand-written mirror, swept down the cache
/// hierarchy: a backend difference shows only while the loop is compute-bound
/// and vanishes once both sides are waiting on the same memory.
fn sweep(
    tag: &str,
    art_bytes: &[u8],
    scalar: fn(&[f32]) -> f64,
    simd: unsafe fn(&[f32]) -> f64,
    iterations: usize,
    rows: &mut Vec<BenchResult>,
) {
    let sizes: &[(usize, &str)] = &[
        (16_384, "64KB L1"),
        (262_144, "1MB L2"),
        (4_194_304, "16MB L3"),
        (33_554_432, "128MB RAM"),
    ];
    // total work fixed, so ns/element is comparable across sizes
    const TOTAL: usize = 256 << 20;

    let artifact = Artifact::from_bytes(art_bytes);
    let mut inst = base::Base::new(artifact).expect("Base::new failed");
    let mut out = [0u8; 8];

    for &(n, label) in sizes {
        let data = gen_floats(n, 42);
        let bytes = as_bytes(&data);
        let reps = (TOTAL / n).max(1);
        let per_elem = |ms: f64| ms * 1e6 / (reps * n) as f64;

        let scalar_ms = harness::median_of(iterations, || {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                std::hint::black_box(scalar(&data));
            }
            t.elapsed().as_secs_f64() * 1000.0
        });

        let simd_ms = harness::median_of(iterations, || {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                std::hint::black_box(unsafe { simd(&data) });
            }
            t.elapsed().as_secs_f64() * 1000.0
        });

        let _ = inst.execute("main", bytes, &mut out);
        let base_ms = harness::median_of(iterations, || {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                let _ = inst.execute("main", bytes, &mut out);
            }
            t.elapsed().as_secs_f64() * 1000.0
        });

        // identical order of operations on both sides, so anything short of bit
        // equality means the mirror drifted and the row compares two algorithms
        let verified =
            Some(f64::from_le_bytes(out).to_bits() == unsafe { simd(&data) }.to_bits());

        rows.push(BenchResult {
            name: format!("{tag} {label}"),
            col_a_ms: Some(per_elem(scalar_ms)),
            col_b_ms: Some(per_elem(simd_ms)),
            base_ms: per_elem(base_ms),
            verified,
        });
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_double(src: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let two = _mm_set1_ps(2.0);
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let main_end = (n / 16) * 16;
    let mut i = 0;
    while i < main_end {
        for k in 0..4 {
            let v = _mm_loadu_ps(sp.add(i + k * 4));
            _mm_storeu_ps(dp.add(i + k * 4), _mm_mul_ps(v, two));
        }
        i += 64 / 4;
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn simd_double(src: &[f32], dst: &mut [f32]) {
    scalar_double(src, dst)
}

fn scalar_double(src: &[f32], dst: &mut [f32]) {
    let end = (src.len() / 16) * 16;
    for i in 0..end {
        dst[i] = src[i] * 2.0;
    }
}

/// Write-heavy: the output is as large as the input, so this is checked by
/// comparing the whole buffer rather than one scalar.
fn sweep_store(iterations: usize, rows: &mut Vec<BenchResult>) {
    let sizes: &[(usize, &str)] = &[
        (16_384, "64KB L1"),
        (262_144, "1MB L2"),
        (4_194_304, "16MB L3"),
        (33_554_432, "128MB RAM"),
    ];
    const TOTAL: usize = 256 << 20;
    let artifact = Artifact::from_bytes(STORE_ARTIFACT);
    let mut inst = base::Base::new(artifact).expect("Base::new failed");

    for &(n, label) in sizes {
        let data = gen_floats(n, 42);
        let bytes = as_bytes(&data);
        let reps = (TOTAL / n).max(1);
        let per = |ms: f64| ms * 1e6 / (reps * n) as f64;

        let mut d_scalar = vec![0f32; n];
        let scalar_ms = harness::median_of(iterations, || {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                scalar_double(&data, &mut d_scalar);
            }
            t.elapsed().as_secs_f64() * 1000.0
        });

        let mut d_simd = vec![0f32; n];
        let simd_ms = harness::median_of(iterations, || {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                unsafe { simd_double(&data, &mut d_simd) };
            }
            t.elapsed().as_secs_f64() * 1000.0
        });

        let mut d_base = vec![0u8; n * 4];
        let _ = inst.execute("main", bytes, &mut d_base);
        let base_ms = harness::median_of(iterations, || {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                let _ = inst.execute("main", bytes, &mut d_base);
            }
            t.elapsed().as_secs_f64() * 1000.0
        });

        let base_f: &[f32] =
            unsafe { std::slice::from_raw_parts(d_base.as_ptr() as *const f32, n) };
        let verified = Some(
            base_f.iter().zip(d_simd.iter()).all(|(a, b)| a.to_bits() == b.to_bits()),
        );

        rows.push(BenchResult {
            name: format!("store {label}"),
            col_a_ms: Some(per(scalar_ms)),
            col_b_ms: Some(per(simd_ms)),
            base_ms: per(base_ms),
            verified,
        });
    }
}

/// The branchy chain, written the way anyone would write it.  Unlike the other
/// mirrors this does NOT pin the instruction sequence: LLVM is free to
/// if-convert it to a `cmov` and Cranelift is free not to, which is the point
/// -- on an unpredictable branch that choice is worth more than any lowering.
fn branchy_chain(data: &[f32]) -> f64 {
    let words =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()) };
    let mut h = 1u64;
    for &w in words {
        h = if w & 1 == 0 {
            (h + w as u64) & 0xFF_FFFF
        } else {
            (h * 3) & 0xFF_FFFF
        };
    }
    h as f64
}

/// `branchy_chain` with the same masking hoist the `sel+mask` artifact makes.
/// Without this the comparison would be Base-with-an-optimisation against
/// Rust-without-it, which is the mistake the rest of this file exists to
/// avoid: the answers agree, but the programs would not.
fn branchy_chain_nomask(data: &[f32]) -> f64 {
    let words =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()) };
    let mut h = 1u64;
    for &w in words {
        h = if w & 1 == 0 {
            h.wrapping_add(w as u64)
        } else {
            h.wrapping_mul(3)
        };
    }
    (h & 0xFF_FFFF) as f64
}

pub fn run(iterations: usize) -> Vec<BenchResult> {
    // what entering Base costs at all, so a small-size row can be read knowing
    // how much of it is dispatch rather than arithmetic
    {
        let artifact = Artifact::from_bytes(CLAMP_ARTIFACT);
        let mut inst = base::Base::new(artifact).expect("Base::new failed");
        let mut out = [0u8; 8];
        let tiny = gen_floats(16, 7);
        let bytes = as_bytes(&tiny);
        let _ = inst.execute("main", bytes, &mut out);
        let reps = 100_000;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            let _ = inst.execute("main", bytes, &mut out);
        }
        println!(
            "  Base dispatch overhead: {:.0} ns/call (16-element input)",
            t.elapsed().as_secs_f64() * 1e9 / reps as f64
        );
    }

    width_ceiling(iterations);

    let mut rows = Vec::new();
    sweep("clamp", CLAMP_ARTIFACT, scalar_clamp_sum, simd_clamp_sum, iterations, &mut rows);
    sweep("plain", PLAIN_ARTIFACT, scalar_plain_sum, simd_plain_sum, iterations, &mut rows);
    sweep("pmin", PMIN_ARTIFACT, scalar_clamp_sum, simd_clamp_sum, iterations, &mut rows);
    sweep("regs16", REGP_ARTIFACT, scalar_regp_sum, simd_regp_sum, iterations, &mut rows);
    sweep("int", INT_ARTIFACT, int_sum, int_sum, iterations, &mut rows);
    sweep_store(iterations, &mut rows);
    sweep("branchy", BRANCHY_ARTIFACT, branchy_chain, branchy_chain, iterations, &mut rows);
    sweep("select", SELECT_ARTIFACT, branchy_chain, branchy_chain, iterations, &mut rows);
    sweep("sel+lea", SELLEA_ARTIFACT, branchy_chain, branchy_chain, iterations, &mut rows);
    sweep("sel+rot", SELROT_ARTIFACT, branchy_chain, branchy_chain, iterations, &mut rows);
    sweep("sel+mask", SELMASK_ARTIFACT, branchy_chain_nomask, branchy_chain_nomask, iterations, &mut rows);
    rows
}

/// Base's widest vector against LLVM's, on the loop where they are otherwise
/// at parity.  Reported apart from the table above because the two are NOT the
/// same computation -- eight lanes accumulate differently from four -- so bit
/// equality is not the check here and the row is a ceiling, not a gap.
fn width_ceiling(iterations: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::arch::is_x86_feature_detected!("avx") {
            println!("  (no AVX on this host; width ceiling not measured)");
            return;
        }
        let artifact = Artifact::from_bytes(PLAIN_ARTIFACT);
        let mut inst = base::Base::new(artifact).expect("Base::new failed");
        let mut out = [0u8; 8];
        const TOTAL: usize = 256 << 20;
        println!();
        println!(
            "{:<24} {:>14} {:>14} {:>14}",
            "ns/element", "Base 128-bit", "LLVM 256-bit", "ratio"
        );
        println!("{}", "-".repeat(24 + 14 * 3 + 4));
        for (n, label) in [
            (16_384usize, "64KB L1"),
            (262_144, "1MB L2"),
            (4_194_304, "16MB L3"),
            (33_554_432, "128MB RAM"),
        ] {
            let data = gen_floats(n, 42);
            let bytes = as_bytes(&data);
            let reps = (TOTAL / n).max(1);
            let per = |ms: f64| ms * 1e6 / (reps * n) as f64;

            let avx_ms = harness::median_of(iterations, || {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(unsafe { simd256_plain_sum(&data) });
                }
                t.elapsed().as_secs_f64() * 1000.0
            });
            let _ = inst.execute("main", bytes, &mut out);
            let base_ms = harness::median_of(iterations, || {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    let _ = inst.execute("main", bytes, &mut out);
                }
                t.elapsed().as_secs_f64() * 1000.0
            });
            println!(
                "{:<24} {:>14} {:>14} {:>14}",
                format!("plain {label}"),
                format!("{:.3}", per(base_ms)),
                format!("{:.3}", per(avx_ms)),
                format!("{:.2}x", per(base_ms) / per(avx_ms))
            );
        }
    }
}

pub fn print(results: &[BenchResult]) {
    println!();
    println!(
        "{:<24} {:>14} {:>14} {:>14} {:>6}",
        "ns/element", "Rust scalar", "Rust SIMD", "Base CLIF", "Exact"
    );
    println!("{}", "-".repeat(24 + 14 * 3 + 6 + 4));
    for r in results {
        println!(
            "{:<24} {:>14} {:>14} {:>14} {:>6}",
            r.name,
            format!("{:.3}", r.col_a_ms.unwrap_or(f64::NAN)),
            format!("{:.3}", r.col_b_ms.unwrap_or(f64::NAN)),
            format!("{:.3}", r.base_ms),
            if r.verified == Some(true) { "\u{2713}" } else { "\u{2717}" }
        );
    }
    println!();
    println!("  Rust SIMD mirrors the emitted CLIF: same unroll, same accumulators,");
    println!("  same order.  Bit-exact agreement is what makes the third column a");
    println!("  backend comparison rather than an algorithm one.");
    println!();
    println!("  `plain` isolates the backend itself; `clamp` vs `pmin` prices one");
    println!("  lowering: CLIF `fmin` is IEEE minimumNumber, which `minps` does not");
    println!("  implement, so Cranelift emits a NaN-correct sequence.  Asking for the");
    println!("  operation the hardware has closes it without changing the answer.");
}
