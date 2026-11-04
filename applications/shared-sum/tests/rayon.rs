//! `shared_sum`'s ring buffer between rayon producers and consumers: every
//! item is popped exactly once.

use std::sync::Mutex;

use base::Driver;
use rayon::prelude::*;

const ITEMS: u64 = 100_000;
const CONSUMERS: usize = 4;

fn load() -> Driver {
    Driver::load(lean_artifacts::SHARED_SUM).unwrap()
}

#[test]
fn rayon_producers_and_consumers_share_the_ring() {
    let driver = load();
    let push = unsafe { driver.function("push") }.unwrap();
    let pop = unsafe { driver.function("pop") }.unwrap();
    let close = unsafe { driver.function("close") }.unwrap();
    let popped = Mutex::new(Vec::with_capacity(ITEMS as usize));

    // A consumer blocks in `pop` while the ring is empty, so the pool has room
    // for every consumer and the producers besides.
    let pool = rayon::ThreadPoolBuilder::new().num_threads(CONSUMERS + 4).build().unwrap();
    pool.scope(|s| {
        for _ in 0..CONSUMERS {
            s.spawn(|_| {
                let mut mine = Vec::new();
                let mut item = [0u8; 8];
                while pop(&[], &mut item) == 1 {
                    mine.push(u64::from_le_bytes(item));
                }
                popped.lock().unwrap().extend(mine);
            });
        }
        s.spawn(|_| {
            // Any work at all can make the items: here, a parallel map.
            (0..ITEMS).into_par_iter().map(|i| i * 3 + 1).for_each(|item| {
                assert_eq!(push(&item.to_le_bytes(), &mut []), 1);
            });
            close(&[], &mut []);
        });
    });

    let mut popped = popped.into_inner().unwrap();
    popped.sort_unstable();
    let expected: Vec<u64> = (0..ITEMS).map(|i| i * 3 + 1).collect();
    assert_eq!(popped.len(), expected.len(), "every item popped once");
    assert_eq!(popped, expected);
    // Closed and drained: a push is refused and a pop finds nothing.
    assert_eq!(push(&7u64.to_le_bytes(), &mut []), 0);
    assert_eq!(pop(&[], &mut [0u8; 8]), 0);
}
