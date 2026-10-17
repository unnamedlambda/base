//! One `Driver` shared by many threads, each calling into it at once.
//!
//! `shared_sum` updates its total, call count and histogram with atomics, so
//! every call's addition lands whatever the interleaving. The checks are the
//! ones the C++ host in `../cpp` makes: the program's totals equal the host's,
//! and the totals each call saw before its addition tile `[0, total)`, which is
//! what each addition applied exactly once, in one order, leaves.

use std::sync::{Arc, Mutex};
use std::thread;

use base::{Artifact, Driver};

const TOTAL_AT: usize = 0x40;
const CALLS_AT: usize = 0x48;
const HIST_AT: usize = 0x50;

fn load() -> Driver {
    Driver::load(Artifact::from_bytes(lean_artifacts::SHARED_SUM).unwrap()).unwrap()
}

fn read_u64(driver: &Driver, at: usize) -> u64 {
    let mut word = [0u8; 8];
    driver.read(at, &mut word).unwrap();
    u64::from_le_bytes(word)
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
                    let sum = driver.execute("add", &input, &mut before).unwrap() as u64;
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
    assert_eq!(driver.execute("total", &[], &mut []).unwrap() as u64, sent);
    assert_eq!(read_u64(&driver, TOTAL_AT), sent);
    assert_eq!(driver.execute("calls", &[], &mut []).unwrap() as usize, THREADS * CALLS);
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

/// A read reaching past the end is an error that names the range, and a read
/// that fits answers the bytes there.
#[test]
fn read_refuses_past_the_end() {
    let driver = load();
    let size = driver.arena_size();
    let mut two = [0u8; 2];
    assert!(driver.read(size - 2, &mut two).is_ok());
    let e = driver.read(size - 1, &mut two).unwrap_err();
    assert!(matches!(e, base::Error::Range(_)), "{e}");
    assert!(driver.read(usize::MAX, &mut two).is_err());
}
