//! The libraries a program links, at work, against the Rust crates a program
//! would use instead.
//!
//! The Base side is the `system_bench` artifact (`Bench/System.lean`), whose
//! entries call `Lib.Ht`, `Lib.Lmdb` and the file adapter; the Rust side is
//! hashbrown with foldhash, LMDB through `liblmdb-sys`, and `std::fs::read`
//! with memchr. Every answer is checked against Rust's before anything is
//! timed.
//!
//! Output, read by `run.sh`, is one line per timing:
//!
//!     <workload> <column> <ns>

use base::{Artifact, Driver};
use std::ffi::CString;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Samples per timing.
const SAMPLES: usize = 7;

/// Records `ht` and `kv` handle.
const N: u64 = 100_000;

/// The file `wc` reads.
const WC_BYTES: usize = 32 * 1024 * 1024;

/// Room in the output for `kv`'s scan: per record a `u16` key and value
/// length, the key and the value.
const OUT_BYTES: usize = 64 + 4 + (N as usize) * (4 + 8 + 32);

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

/// Open `dir`, write `n` records in one transaction in scrambled key order,
/// commit, and scan them all into `out` from byte 64 as `Lib.Lmdb`'s scan lays
/// them out.
fn rust_kv(dir: &Path, n: u64, out: &mut [u8]) -> u64 {
    use liblmdb_sys as sys;
    use lmdb_zero as lmdb;
    std::fs::create_dir_all(dir).unwrap();
    let mut b = lmdb::EnvBuilder::new().unwrap();
    b.set_mapsize(1 << 30).unwrap();
    b.set_maxdbs(1).unwrap();
    let env = unsafe { b.open(dir.to_str().unwrap(), lmdb::open::WRITEMAP | lmdb::open::NOSYNC, 0o600) }
        .unwrap();
    let dbi = lmdb::Database::open(&env, None, &lmdb::DatabaseOptions::defaults()).unwrap().into_raw();
    unsafe {
        let mut txn = std::ptr::null_mut();
        assert_eq!(sys::mdb_txn_begin(env.as_raw(), std::ptr::null_mut(), 0, &mut txn), 0);
        let mut val = [0u8; 32];
        for i in 0..n {
            let key = ((i * 7919) % n).to_be_bytes();
            for c in val.chunks_mut(8) {
                c.copy_from_slice(&key_of(i).to_le_bytes());
            }
            let mut k = sys::MDB_val { mv_size: 8, mv_data: key.as_ptr() as *const _ };
            let mut v = sys::MDB_val { mv_size: 32, mv_data: val.as_ptr() as *const _ };
            assert_eq!(sys::mdb_put(txn, dbi, &mut k, &mut v, 0), 0);
        }
        assert_eq!(sys::mdb_txn_commit(txn), 0);
        let mut txn = std::ptr::null_mut();
        assert_eq!(sys::mdb_txn_begin(env.as_raw(), std::ptr::null_mut(), sys::MDB_RDONLY, &mut txn), 0);
        let mut cur = std::ptr::null_mut();
        assert_eq!(sys::mdb_cursor_open(txn, dbi, &mut cur), 0);
        let mut k = sys::MDB_val { mv_size: 0, mv_data: std::ptr::null() };
        let mut v = sys::MDB_val { mv_size: 0, mv_data: std::ptr::null() };
        let mut at = 64 + 4;
        let mut count = 0u32;
        let mut rc = sys::mdb_cursor_get(cur, &mut k, &mut v, sys::MDB_cursor_op::MDB_FIRST);
        while rc == 0 && (count as u64) < n {
            out[at..at + 2].copy_from_slice(&(k.mv_size as u16).to_le_bytes());
            out[at + 2..at + 4].copy_from_slice(&(v.mv_size as u16).to_le_bytes());
            at += 4;
            out[at..at + k.mv_size].copy_from_slice(std::slice::from_raw_parts(k.mv_data as *const u8, k.mv_size));
            at += k.mv_size;
            out[at..at + v.mv_size].copy_from_slice(std::slice::from_raw_parts(v.mv_data as *const u8, v.mv_size));
            at += v.mv_size;
            count += 1;
            rc = sys::mdb_cursor_get(cur, &mut k, &mut v, sys::MDB_cursor_op::MDB_NEXT);
        }
        out[64..68].copy_from_slice(&count.to_le_bytes());
        sys::mdb_cursor_close(cur);
        sys::mdb_txn_abort(txn);
        count as u64
    }
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
    let art = Artifact::from_bytes(lean_artifacts::SYSTEM_BENCH).expect("the build checked this artifact");
    let mut drv = Driver::load(art).expect("system_bench loads");
    let mut out = vec![0u8; OUT_BYTES];
    let work = std::env::temp_dir().join(format!("base-system-bench-{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();

    // ht
    let input = N.to_le_bytes().to_vec();
    drv.execute("ht", &input, &mut out).unwrap();
    assert_eq!(answer(&out), rust_ht(N), "ht");
    let t = time(2, |c| match c {
        0 => { black_box(rust_ht(black_box(N))); }
        _ => { drv.execute("ht", &input, &mut out).unwrap(); }
    });
    println!("ht\trust\t{:.0}\nht\tbase\t{:.0}", t[0], t[1]);

    // kv: each side its own directory, reopened by every call
    let (dir_rust, dir_base): (PathBuf, PathBuf) = (work.join("kv-rust"), work.join("kv-base"));
    let mut input = N.to_le_bytes().to_vec();
    input.extend(cstr(&dir_base));
    let mut out_rust = vec![0u8; OUT_BYTES];
    let want = rust_kv(&dir_rust, N, &mut out_rust);
    drv.execute("kv", &input, &mut out).unwrap();
    assert_eq!(answer(&out), want, "kv count");
    assert!(out[64..] == out_rust[64..], "kv scans differ");
    let t = time(2, |c| match c {
        0 => { black_box(rust_kv(&dir_rust, N, &mut out_rust)); }
        _ => { drv.execute("kv", &input, &mut out).unwrap(); }
    });
    println!("kv\trust\t{:.0}\nkv\tbase\t{:.0}", t[0], t[1]);

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
    drv.execute("wc", &input, &mut out).unwrap();
    assert_eq!(answer(&out), rust_wc(&file, &mut buf), "wc");
    let t = time(2, |c| match c {
        0 => { black_box(rust_wc(black_box(&file), &mut buf)); }
        _ => { drv.execute("wc", &input, &mut out).unwrap(); }
    });
    println!("wc\trust\t{:.0}\nwc\tbase\t{:.0}", t[0], t[1]);

    std::fs::remove_dir_all(&work).ok();
}
