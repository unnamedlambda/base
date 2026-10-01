//! The harness itself; `main.rs` says what it measures.

use base::{Artifact, Driver};
use crate::kernels::*;
use std::hint::black_box;
use std::time::Instant;

/// A Rust kernel from `kernels.rs`.
type Kernel = fn(&In) -> u64;

/// Samples per timing.
const SAMPLES: usize = 7;

/// The bytes of plain numbers (no padding).
fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }.to_vec()
}

/// The output every entry is handed: the answer at 0, room from 64 for what a
/// kernel writes (stream's copy).
const OUT_BYTES: usize = 64 + 8 * STREAM_N;

/// Every workload: its name (the artifact's entries are `<name>_clif` and
/// `<name>_asm`), its Rust kernel, and the input the artifact is handed, laid
/// out as the workload's `input` in `Bench/Cpu.lean` reads it.
fn workloads() -> Vec<(&'static str, Kernel, fn(&In) -> Vec<u8>)> {
    vec![
        ("histogram", k_histogram_low, |i| i.text.to_vec()),
        ("mandel", k_mandel_low, |_| Vec::new()),
        ("chase", k_chase_low, |i| bytes_of(&i.perm)),
        ("poly", k_poly_idiom, |i| bytes_of(&i.f32a)),
        ("stream", k_stream_best, |_| {
            bytes_of(&(0..STREAM_N as u64).map(|k| k.wrapping_mul(0x9E3779B97F4A7C15)).collect::<Vec<_>>())
        }),
    ]
}

/// A column to time: a Rust kernel, or a Base entry.
enum Col {
    Rust(Kernel),
    Base(String),
}

/// Each column's fastest sample, the columns sampled in turn so that all see
/// the machine in the same state: at least `SAMPLES` samples each, and as
/// many as fill a fifth of a second per column. A sample is the mean over
/// enough calls of `run` to fill about two milliseconds, after one call that
/// is not timed, so it starts from what its own column leaves in the caches
/// rather than the last one's (a copy after the CLIF's pays to write back the
/// lines the CLIF's stores left dirty). Taking turns is what steadies a
/// kernel bound by memory, whose speed drifts over seconds.
fn time(cols: &[Col], mut run: impl FnMut(&Col)) -> Vec<f64> {
    let reps: Vec<usize> = cols
        .iter()
        .map(|c| {
            let t0 = Instant::now();
            run(c);
            ((2e-3 / t0.elapsed().as_secs_f64().max(1e-8)).ceil() as usize).max(1)
        })
        .collect();
    let start = Instant::now();
    let mut best = vec![f64::MAX; cols.len()];
    let mut k = 0;
    while k < SAMPLES || start.elapsed().as_secs_f64() < 0.2 * cols.len() as f64 {
        for (j, c) in cols.iter().enumerate() {
            run(c);
            let t = Instant::now();
            for _ in 0..reps[j] {
                run(c);
            }
            best[j] = best[j].min(t.elapsed().as_secs_f64() * 1e9 / reps[j] as f64);
        }
        k += 1;
    }
    best
}

pub fn main() {
    let art = Artifact::from_bytes(lean_artifacts::CPU_BENCH).expect("the build checked this artifact");
    let names: Vec<String> = art.functions.iter().filter_map(|f| f.entry_name.clone()).collect();
    let mut drv = Driver::load(art).expect("cpu_bench loads");
    // Map the machine code, once, as a program would at startup; every body
    // must have loaded, or an `_asm` row would be timing its fallback.
    let mut slots = vec![0u8; 8 * 64];
    drv.execute("asm_load", &[], &mut slots).expect("asm_load runs");
    let word = |k: usize| u64::from_le_bytes(slots[8 * k..8 * k + 8].try_into().unwrap());
    let bodies = word(0) as usize;
    assert!(bodies > 0 && (1..=bodies).all(|k| word(k) != 0),
        "a body did not load on this CPU, so its _asm entry would run CLIF");

    let inp = make_input();
    let mut out_buf = Lines::<u8>::zeroed(OUT_BYTES);
    let out = &mut out_buf[..];
    let t = time(&[Col::Base("noop".into())], |_| { black_box(drv.execute("noop", &[], out).unwrap()); });
    println!("execute\tnoop\t{:.1}", t[0]);

    for (w, kernel, input) in workloads() {
        let data = Lines::from_slice(&input(&inp));
        let data = &data[..];
        // stream's Rust kernel copies what Base's entries copy, to where they
        // copy it: the same memory, so the columns differ only in the code
        STREAM_SRC.store(data.as_ptr() as *mut u64, std::sync::atomic::Ordering::Relaxed);
        STREAM_DST.store(out[64..].as_mut_ptr() as *mut u64, std::sync::atomic::Ordering::Relaxed);
        let want = kernel(&inp);
        let mut cols = vec![("rust".to_string(), Col::Rust(kernel))];
        for col in ["clif", "asm"] {
            let entry = format!("{w}_{col}");
            if !names.contains(&entry) {
                continue;
            }
            // the answer's slot, so an entry that never wrote it is caught
            out[..8].fill(0);
            drv.execute(&entry, data, out).unwrap();
            let got = u64::from_le_bytes(out[..8].try_into().unwrap());
            assert_eq!(got, want, "{entry} answers {got:#x}, the Rust kernel {want:#x}");
            cols.push((col.to_string(), Col::Base(entry)));
        }
        let (labels, cols): (Vec<String>, Vec<Col>) = cols.into_iter().unzip();
        let ts = time(&cols, |c| match c {
            Col::Rust(k) => { black_box(k(black_box(&inp))); }
            Col::Base(e) => { black_box(drv.execute(e, black_box(data), out).unwrap()); }
        });
        for (label, t) in labels.iter().zip(ts) {
            println!("{w}\t{label}\t{t:.1}");
        }
    }
}
