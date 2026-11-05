//! Base's libraries (`system_bench`: hash table, files) against the Rust
//! crates a program would use instead. Prints `<workload> <column> <ns>` lines.

use base::Driver;
use std::ffi::CString;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

/// Samples per timing.
const SAMPLES: usize = 7;

/// Records `ht` handles.
const N: u64 = 100_000;

/// The file `wc` reads.
const WC_BYTES: usize = 32 * 1024 * 1024;

/// Room in the output: the answer.
const OUT_BYTES: usize = 64;

fn key_of(i: u64) -> u64 {
    (i + 1).wrapping_mul(0x9E3779B97F4A7C15)
}

// ---------------------------------------------------------------------------
// The Rust side
// ---------------------------------------------------------------------------

fn rust_ht(n: u64) -> u64 {
    let mut m: hashbrown::HashMap<Box<[u8]>, Box<[u8]>, foldhash::fast::RandomState> =
        hashbrown::HashMap::default();
    for i in 0..n {
        m.insert(Box::from(&key_of(i).to_le_bytes()[..]), Box::from(&i.to_le_bytes()[..]));
    }
    let mut sum = 0u64;
    for i in 0..n {
        let v = &m[&key_of(i).to_le_bytes()[..]];
        sum = sum.wrapping_add(u64::from_le_bytes(v[..8].try_into().unwrap()));
    }
    sum
}

/// Read the file into a buffer kept across calls, as Base reads into memory
/// it already has, and count its newlines.
fn rust_wc(path: &Path, buf: &mut Vec<u8>) -> u64 {
    use std::io::Read;
    buf.clear();
    std::fs::File::open(path).unwrap().read_to_end(buf).unwrap();
    memchr::memchr_iter(b'\n', buf).count() as u64
}

// ---------------------------------------------------------------------------
// The harness
// ---------------------------------------------------------------------------

/// Each column's fastest sample, the columns sampled in turn, each sample
/// after one untimed call of its own column (see `benchmarks/cpu`).
fn time(cols: usize, mut run: impl FnMut(usize)) -> Vec<f64> {
    let reps: Vec<usize> = (0..cols)
        .map(|c| {
            let t0 = Instant::now();
            run(c);
            ((20e-3 / t0.elapsed().as_secs_f64().max(1e-8)).ceil() as usize).max(1)
        })
        .collect();
    let start = Instant::now();
    let mut best = vec![f64::MAX; cols];
    let mut k = 0;
    while k < SAMPLES || start.elapsed().as_secs_f64() < 0.5 * cols as f64 {
        for (c, b) in best.iter_mut().enumerate() {
            run(c);
            let t = Instant::now();
            for _ in 0..reps[c] {
                run(c);
            }
            *b = b.min(t.elapsed().as_secs_f64() * 1e9 / reps[c] as f64);
        }
        k += 1;
    }
    best
}

fn answer(out: &[u8]) -> u64 {
    u64::from_le_bytes(out[..8].try_into().unwrap())
}

fn cstr(p: &Path) -> Vec<u8> {
    CString::new(p.to_str().unwrap()).unwrap().into_bytes_with_nul()
}

fn main() {
    let drv = unsafe { Driver::load(lean_artifacts::SYSTEM_BENCH).expect("system_bench loads") };
    let mut out = vec![0u8; OUT_BYTES];
    let work = std::env::temp_dir().join(format!("base-system-bench-{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();

    // ht
    let input = N.to_le_bytes().to_vec();
    let ht = drv.function("ht").unwrap();
    ht(&input, &mut out);
    assert_eq!(answer(&out), rust_ht(N), "ht");
    let t = time(2, |c| match c {
        0 => { black_box(rust_ht(black_box(N))); }
        _ => { ht(&input, &mut out); }
    });
    println!("ht\trust\t{:.0}\nht\tbase\t{:.0}", t[0], t[1]);

    // wc: lines of 1 to 80 bytes
    let file = work.join("wc.txt");
    let mut text = Vec::with_capacity(WC_BYTES);
    let mut s = 1u64;
    while text.len() < WC_BYTES - 81 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let len = (s >> 58) as usize + (s >> 60) as usize * 4 + 1;
        text.extend((0..len).map(|i| b'a' + ((s >> (i % 50)) % 26) as u8));
        text.push(b'\n');
    }
    std::fs::write(&file, &text).unwrap();
    let input = cstr(&file);
    let mut buf = Vec::with_capacity(WC_BYTES);
    let wc = drv.function("wc").unwrap();
    wc(&input, &mut out);
    assert_eq!(answer(&out), rust_wc(&file, &mut buf), "wc");
    let t = time(2, |c| match c {
        0 => { black_box(rust_wc(black_box(&file), &mut buf)); }
        _ => { wc(&input, &mut out); }
    });
    println!("wc\trust\t{:.0}\nwc\tbase\t{:.0}", t[0], t[1]);

    std::fs::remove_dir_all(&work).ok();
}
