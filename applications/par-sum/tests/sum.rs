//! `par_sum` sums the 4096 bytes in its arena as 512 words, on four threads it
//! starts itself; the patterns catch a quarter summed twice, missed, or read
//! out of place.

use base::Driver;
use std::sync::atomic::Ordering::Relaxed;

const ARTIFACT: &[u8] = lean_artifacts::PAR_SUM;
/// `ParSum.MEM`: the input, in the arena.
const BYTES: usize = 4096;

fn wrapping_sum(data: &[u8]) -> u64 {
    data.chunks_exact(8)
        .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
        .fold(0u64, u64::wrapping_add)
}

/// Writes `data` into the arena, runs `main`, and answers its eight bytes.
fn run(base: &Driver, data: &[u8]) -> u64 {
    for (cell, &b) in base.arena().iter().zip(data) {
        cell.store(b, Relaxed);
    }
    let mut out = [0u8; 8];
    base.function("main").expect("no such entry")(&[], &mut out);
    u64::from_le_bytes(out)
}

#[test]
fn four_threads_sum_every_word() {
    let base = unsafe { Driver::load(ARTIFACT).expect("Driver::load") };
    assert!(base.arena().len() >= BYTES);
    let mut cases: Vec<(&str, Vec<u8>)> = vec![
        ("zeros", vec![0; BYTES]),
        ("ones", vec![1; BYTES]),
        ("all 0xff, wrapping", vec![0xff; BYTES]),
        ("index bytes", (0..BYTES).map(|i| i as u8).collect()),
    ];
    // One word set in each quarter, distinct, so a quarter summed twice or
    // missed changes the answer.
    for q in 0..4 {
        let mut d = vec![0u8; BYTES];
        d[q * 1024 + 8..q * 1024 + 16].copy_from_slice(&(1u64 << (8 * q)).to_le_bytes());
        cases.push(("one word in a quarter", d));
    }
    for (name, data) in &cases {
        assert_eq!(run(&base, data), wrapping_sum(data), "{name}");
    }
    // Pseudo-random inputs, run after run on the same driver.
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..200 {
        let data: Vec<u8> = (0..BYTES)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        assert_eq!(run(&base, &data), wrapping_sum(&data));
    }
}

#[test]
fn no_room_writes_nothing() {
    let base = unsafe { Driver::load(ARTIFACT).expect("Driver::load") };
    let mut out = [7u8; 7];
    base.function("main").expect("no such entry")(&[], &mut out);
    assert_eq!(out, [7u8; 7]);
}
