//! One driver called from many threads at once: every addition lands exactly
//! once, as the C++ host also checks.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::thread;

use base::Driver;

const TOTAL_AT: usize = 0x40;
const CALLS_AT: usize = 0x48;
const HIST_AT: usize = 0x50;

fn load() -> Driver {
    unsafe { Driver::load(lean_artifacts::SHARED_SUM).unwrap() }
}

/// The word at `at`, in one atomic load as the program's atomics write it: a
/// byte at a time could see half of a concurrent add.
fn read_u64(driver: &Driver, at: usize) -> u64 {
    let arena = driver.arena();
    assert!(at + 8 <= arena.len());
    let ptr = arena[at].as_ptr().cast::<u64>();
    assert!(ptr.is_aligned());
    // SAFETY: in bounds and aligned, and the arena lives as long as `driver`;
    // every access to it from Rust is atomic.
    unsafe { std::sync::atomic::AtomicU64::from_ptr(ptr) }.load(Relaxed)
}

/// The bytes thread `t` sends on call `i`: varied in length, zero included.
fn bytes(t: usize, i: usize) -> Vec<u8> {
    (0..(t * 131 + i * 17) % 700).map(|k| (t * 31 + i * 7 + k * 13) as u8).collect()
}

#[test]
fn concurrent_adds_on_one_driver_all_land() {
    const THREADS: usize = 8;
    const CALLS: usize = 300;
    let driver = Arc::new(load());
    let spans = Arc::new(Mutex::new(Vec::new()));

    let workers: Vec<_> = (0..THREADS)
        .map(|t| {
            let driver = Arc::clone(&driver);
            let spans = Arc::clone(&spans);
            thread::spawn(move || {
                for i in 0..CALLS {
                    let input = bytes(t, i);
                    let mut before = [0u8; 8];
                    let sum = driver.function("add").unwrap()(&input, &mut before) as u64;
                    assert_eq!(sum, input.iter().map(|&b| b as u64).sum::<u64>());
                    spans.lock().unwrap().push((u64::from_le_bytes(before), sum));
                }
            })
        })
        .collect();
    // Reads run beside the calls, and the total only grows.
    let mut last = 0;
    while !workers.iter().all(|w| w.is_finished()) {
        let now = read_u64(&driver, TOTAL_AT);
        assert!(now >= last, "the total went from {last} to {now}");
        last = now;
    }
    for w in workers {
        w.join().unwrap();
    }

    let mut sent = 0u64;
    let mut hist = [0u64; 256];
    for t in 0..THREADS {
        for i in 0..CALLS {
            for b in bytes(t, i) {
                sent += b as u64;
                hist[b as usize] += 1;
            }
        }
    }
    assert_eq!(driver.function("total").unwrap()(&[], &mut []) as u64, sent);
    assert_eq!(read_u64(&driver, TOTAL_AT), sent);
    assert_eq!(driver.function("calls").unwrap()(&[], &mut []) as usize, THREADS * CALLS);
    assert_eq!(read_u64(&driver, CALLS_AT) as usize, THREADS * CALLS);
    for (b, &n) in hist.iter().enumerate() {
        assert_eq!(read_u64(&driver, HIST_AT + 8 * b), n, "histogram counter {b}");
    }
    let mut spans = spans.lock().unwrap().clone();
    spans.sort();
    let mut at = 0;
    for (before, sum) in spans {
        assert_eq!(before, at, "two calls saw overlapping totals");
        at += sum;
    }
    assert_eq!(at, sent);
}
