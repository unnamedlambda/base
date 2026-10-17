//! Producers and consumers on rayon, meeting in one driver's arena.
//!
//! `shared_sum`'s ring buffer lives in the program: `push` and `pop` take its
//! lock and sleep, inside the program, while the ring is full or empty, and
//! `close` ends it. The host side is ordinary rayon: a parallel iterator
//! produces the items, and tasks spawned on the same pool consume them. Each
//! entry point is found once, and every task calls it directly: a `Function` is
//! the program's function, borrowing the one `Driver`.
//!
//! Each item is pushed once, and the consumers between them must pop every one
//! exactly once, whatever the interleaving.

use std::sync::Mutex;

use base::{Artifact, Driver};
use rayon::prelude::*;

const ITEMS: u64 = 100_000;
const CONSUMERS: usize = 4;

fn load() -> Driver {
    Driver::load(Artifact::from_bytes(lean_artifacts::SHARED_SUM).unwrap()).unwrap()
}

#[test]
fn rayon_producers_and_consumers_share_the_ring() {
    let driver = load();
    let push = driver.function("push").unwrap();
    let pop = driver.function("pop").unwrap();
    let close = driver.function("close").unwrap();
    let popped = Mutex::new(Vec::with_capacity(ITEMS as usize));

    // A consumer blocks in `pop` while the ring is empty, so the pool has room
    // for every consumer and the producers besides.
    let pool = rayon::ThreadPoolBuilder::new().num_threads(CONSUMERS + 4).build().unwrap();
    pool.scope(|s| {
        for _ in 0..CONSUMERS {
            s.spawn(|_| {
                let mut mine = Vec::new();
                let mut item = [0u8; 8];
                while unsafe { pop.call(&[], &mut item) } == 1 {
                    mine.push(u64::from_le_bytes(item));
                }
                popped.lock().unwrap().extend(mine);
            });
        }
        s.spawn(|_| {
            // Any work at all can make the items: here, a parallel map.
            (0..ITEMS).into_par_iter().map(|i| i * 3 + 1).for_each(|item| {
                assert_eq!(unsafe { push.call(&item.to_le_bytes(), &mut []) }, 1);
            });
            unsafe { close.call(&[], &mut []) };
        });
    });

    let mut popped = popped.into_inner().unwrap();
    popped.sort_unstable();
    let expected: Vec<u64> = (0..ITEMS).map(|i| i * 3 + 1).collect();
    assert_eq!(popped.len(), expected.len(), "every item popped once");
    assert_eq!(popped, expected);
    // Closed and drained: a push is refused and a pop finds nothing.
    assert_eq!(unsafe { push.call(&7u64.to_le_bytes(), &mut []) }, 0);
    assert_eq!(unsafe { pop.call(&[], &mut [0u8; 8]) }, 0);
}
