# shared-sum: one driver, many threads

A host loads an artifact once and calls it from as many threads as it likes,
all sharing the program's one arena. This is the example of that, in C++
and in Rust.

The artifact is `shared_sum` ([SharedSum.lean](../../lean/algorithms/Demo/SharedSum.lean)):

| entry | does |
|---|---|
| `add` | sums its input bytes, counts each byte into a 256-slot histogram, adds the sum to a running total and counts the call, all with atomic read-modify-writes; answers the sum, and writes the total from just before its addition to the first 8 bytes of its output |
| `total` | the running total, by an atomic load |
| `calls` | how many times `add` ran |
| `push` | puts the first 8 input bytes into a 64-slot ring as one item, sleeping while the ring is full; `1`, or `0` once closed |
| `pop` | takes an item into the first 8 output bytes, sleeping while the ring is empty; `1`, or `0` once closed and empty |
| `close` | closes the ring, waking every sleeping consumer |

The total is at `0x40`, the call count at `0x48`, and the histogram at `0x50`,
as little-endian `u64`s. The ring follows the histogram: its lock word at
`0x850` (`0` free, `1` held, `2` held with sleepers), the head at `0x854`
(items pushed, top bit set once closed), the tail at `0x858` (items popped)
and the 64 slots from `0x860`. The lock is a compare-and-swap plus the
runtime's wait and wake, generated as CLIF; nothing in the engine knows it is
a lock.

## C++ (`cpp/`)

[`main.cpp`](cpp/main.cpp) uses [`base.hpp`](../../base/include/base.hpp), a
C++20 wrapper over the C API in [`base.h`](../../base/include/base.h). A
`base::Driver` is a shared handle: copies go to threads freely, and the driver
is freed when the last copy goes. The main thread loads the driver once, then
drives it seven ways, each with its own host code around the calls:

`add` is found once with `Driver::function` and called directly after that, as
`push`, `pop` and `close` are.

1. **bare `std::thread`**: each thread runs a random walk and sends the path.
2. **a thread pool**: a job queue under a mutex and a condition variable; each
   job finds primes in a range and sends them as text.
3. **`std::async`**: each task sorts its data first, and the futures' answers
   are added up.
4. **producers and consumers** over a bounded queue.
5. **barrier phases**: `std::jthread` workers meet at a `std::barrier` whose
   completion step checks the total while nothing runs, and a monitor thread
   with a `std::stop_token` reads the total from the arena while calls run and
   checks it never goes backwards.
6. **a ring buffer shared with the program**: producer threads call `push`,
   threads calling `pop` consume, and alongside them a host thread consumes
   straight from the arena: it takes the program's own lock with a
   compare-and-swap through `std::atomic_ref`, copies the slot under it, and
   sleeps on the head with the system's wait, which the program's `push` wakes. Every item has
   to arrive exactly once.
7. **shared ownership**: main drops its handle before the threads start, and
   the last thread to finish checks the result. Its copy is the last one, so it
   is the one that frees the driver.

After each section the host checks the arena against its own
ledger of what it sent: the total, the call count, all 256 histogram counters,
and that the totals each call saw before its addition tile `[0, total)`. That
last check holds only if each addition was applied exactly once, in some
single order.

```sh
cd applications/shared-sum/cpp
make run                  # builds libbase (debug) and the example, then runs it
make run THREADS=64       # more threads than cores
make run PROFILE=release
```

`ARTIFACT` defaults to `lean/.lake/build/artifacts/shared_sum.cbor`, which any
cargo build of `lean-artifacts` or `lake query algorithmLib/artifacts` writes.
`tools/check.sh` runs it as the "c++ host" step.

## Rust (`tests/`)

- [`threads.rs`](tests/threads.rs): the `add` checks over an `Arc<Driver>`:
  `cargo test -p applications --test apps shared_sum_threads`.
- [`rayon.rs`](tests/rayon.rs): producers and consumers on a rayon pool. A
  parallel iterator makes 100,000 items and `push`es them, rayon tasks `pop`
  them until the ring is closed, and every item has to come out exactly once:
  `cargo test -p applications --test apps shared_sum_rayon`.

## What the C API promises

Four functions, used like a shared library:

| | |
|---|---|
| `base_driver_load` | compile an artifact, as `dlopen` loads a library; on failure, why is written to the caller's buffer |
| `base_driver_function` | an entry point by name, as a plain C function: `int64_t f(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len)` |
| `base_driver_arena` | the arena, at an address that stays put for the driver's life |
| `base_driver_free` | release the driver; no entry point may run during it or after |

- A host finds an entry point once and calls it directly, from any number of
  threads: nothing of base runs between host and program on a call. The
  function takes the arena's address itself, from a symbol the engine links, so the
  arena is not an argument and a host cannot pass the wrong one.
- While programs run, the host reaches bytes they may touch only atomically
  (`std::atomic_ref`), and the rest freely: the rule two threads of one C
  program follow.
- Waiting on a word, waking, and the clock are the system's own calls (a
  private futex on Linux, `WaitOnAddress` on Windows, `os_sync_wait_on_address`
  on macOS), which programs make too, so a host thread and a program's threads
  wait and wake each other at the same words. `main.cpp` makes them itself on
  the arena `base.hpp` hands over: `base.hpp` is the C API's four calls and
  nothing more.
