"""One `Driver` shared by Python threads, calling into `shared_sum` at once.

The counterpart of `applications/shared-sum/tests/threads.rs` and `rayon.rs`.
A `Function` releases the GIL while the program runs, so these threads run their
calls in parallel, and a thread blocked inside the program (`pop` on an empty
ring, or `Driver.wait`) leaves the others free to run: were the GIL held, the ring
test would deadlock rather than pass.
"""

import os
import struct
import threading

import pytest
from py_base import Driver, load_artifact, now

ARTIFACT = os.path.join(
    os.path.dirname(__file__), "..", "..", "lean", ".lake", "build", "artifacts",
    "shared_sum.cbor")

TOTAL_AT = 0x40
CALLS_AT = 0x48
HIST_AT = 0x50
# A word of the arena the program leaves alone, for waiting on.
FREE_WORD = 0x10

pytestmark = pytest.mark.skipif(
    not os.path.exists(ARTIFACT), reason="the Lean artifacts have not been generated")


def load():
    return Driver(load_artifact(ARTIFACT))


def u64(driver, at):
    return struct.unpack("<Q", driver.read(at, 8))[0]


def run_threads(targets, timeout=60):
    threads = [threading.Thread(target=t) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
        assert not t.is_alive(), "a thread is stuck"


def payload(t, i):
    """The bytes thread `t` sends on call `i`: varied in length, zero included."""
    return bytes((t * 31 + i * 7 + k * 13) % 256 for k in range((t * 131 + i * 17) % 700))


def test_concurrent_adds_all_land():
    THREADS, CALLS = 8, 300
    driver = load()
    add = driver.function("add")
    spans = []
    lock = threading.Lock()
    done = threading.Event()

    def worker(t):
        def body():
            mine = []
            for i in range(CALLS):
                data = payload(t, i)
                before = bytearray(8)
                s = add(data, before)
                assert s == sum(data)
                mine.append((struct.unpack("<Q", before)[0], s))
            with lock:
                spans.extend(mine)
        return body

    seen = []

    def reader():
        # Reads run beside the calls, and the total only grows.
        while not done.is_set():
            seen.append(u64(driver, TOTAL_AT))

    r = threading.Thread(target=reader)
    r.start()
    run_threads([worker(t) for t in range(THREADS)])
    done.set()
    r.join()
    assert seen == sorted(seen), "the total went backwards"

    sent = 0
    hist = [0] * 256
    for t in range(THREADS):
        for i in range(CALLS):
            for b in payload(t, i):
                sent += b
                hist[b] += 1
    assert driver.function("total")() == sent
    assert u64(driver, TOTAL_AT) == sent
    assert driver.function("calls")() == THREADS * CALLS
    assert u64(driver, CALLS_AT) == THREADS * CALLS
    got = struct.unpack("<256Q", driver.read(HIST_AT, 8 * 256))
    assert list(got) == hist
    # Each call saw the total before its addition: together they tile [0, sent).
    at = 0
    for before, s in sorted(spans):
        assert before == at, "two calls saw overlapping totals"
        at += s
    assert at == sent


def test_producers_and_consumers_share_the_ring():
    ITEMS, PRODUCERS, CONSUMERS = 20_000, 3, 4
    driver = load()
    push, pop, close = driver.function("push"), driver.function("pop"), driver.function("close")
    popped = []
    lock = threading.Lock()

    def consumer():
        mine = []
        item = bytearray(8)
        while pop(None, item) == 1:
            mine.append(struct.unpack("<Q", item)[0])
        with lock:
            popped.extend(mine)

    def producer(p):
        def body():
            for i in range(p, ITEMS, PRODUCERS):
                assert push(struct.pack("<Q", i * 3 + 1)) == 1
        return body

    consumers = [threading.Thread(target=consumer) for _ in range(CONSUMERS)]
    for c in consumers:
        c.start()
    run_threads([producer(p) for p in range(PRODUCERS)])
    close()
    for c in consumers:
        c.join(60)
        assert not c.is_alive(), "a consumer is stuck"

    assert sorted(popped) == [i * 3 + 1 for i in range(ITEMS)], "every item popped once"
    # Closed and drained: a push is refused and a pop finds nothing.
    assert push(struct.pack("<Q", 7)) == 0
    assert pop(None, bytearray(8)) == 0


def test_an_output_in_use_cannot_be_resized():
    """A program writes the output buffer with the GIL released: while it can,
    no other thread may resize the buffer out from under it."""
    driver = load()
    push, pop = driver.function("push"), driver.function("pop")
    item = bytearray(8)
    answers = []
    t = threading.Thread(target=lambda: answers.append(pop(None, item)))
    t.start()
    # `pop` sleeps in the program on the empty ring, holding `item`.
    refused = False
    for _ in range(1000):
        try:
            item.extend(b"x")
            del item[8:]
        except BufferError:
            refused = True
            break
    push(struct.pack("<Q", 42))
    t.join(30)
    assert not t.is_alive()
    assert refused, "the buffer was never seen held"
    assert answers == [1] and struct.unpack("<Q", item)[0] == 42
    item.extend(b"x")  # free again once the call is over


def args(*words):
    return struct.pack(f"<{len(words)}q", *words)


def test_wait_wake_and_now():
    driver = load()

    # The word does not hold what is expected: back at once.
    t0 = now()
    driver.wait(FREE_WORD, 1, 5_000_000_000)
    assert now() - t0 < 1_000_000_000
    # It does, and nothing wakes it: back by the timeout.
    driver.wait(FREE_WORD, 0, 2_000_000)
    # Not a word of the arena.
    with pytest.raises(ValueError):
        driver.wait(FREE_WORD + 1, 0, 0)
    with pytest.raises(ValueError):
        driver.wake(1 << 40, 1)

    # A Python thread asleep on the word, woken by another: it is back long
    # before its timeout.
    t0 = now()
    t = threading.Thread(target=lambda: driver.wait(FREE_WORD, 0, 30_000_000_000))
    t.start()
    while t.is_alive() and now() - t0 < 20_000_000_000:
        driver.wake(FREE_WORD, -1)
        t.join(0.01)
    assert not t.is_alive()
    assert now() - t0 < 20_000_000_000

    # The clock never goes back.
    stamps = [now() for _ in range(1000)]
    assert stamps == sorted(stamps)
