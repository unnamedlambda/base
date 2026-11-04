// One driver, loaded once on the main thread, called from many threads at once.
//
// The artifact is `shared_sum` (lean/algorithms/Demo/SharedSum.lean): `add`
// sums its input bytes into a total, counts each byte into a histogram and
// counts itself, all with atomic read-modify-writes into the program's one
// memory. Each section below drives it with a common way of organizing threads
// in C++ (bare threads, a pool, futures, a bounded pipeline, barrier phases
// with a stop-token monitor, shared ownership) and its own ordinary host code
// around each call. The host records what it sent in a ledger, and the
// program's memory has to agree with the ledger exactly, which it does only if
// no call's update was lost to another's.
//
//   usage: shared_sum <shared_sum.cbor> [threads]

#include "base.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <functional>
#include <future>
#include <latch>
#include <mutex>
#include <optional>
#include <random>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <os/os_sync_wait_on_address.h>
#elif defined(_WIN32)
#include <windows.h>
#pragma comment(lib, "Synchronization.lib")
#endif

namespace {

// Where `shared_sum` keeps its state, as SharedSum.lean lays it out.
constexpr std::size_t TOTAL_AT = 0x40;
constexpr std::size_t CALLS_AT = 0x48;
constexpr std::size_t HIST_AT = 0x50;
// The ring buffer: its lock, the head (items pushed, top bit closed), the
// tail (items popped), and 64 slots.
constexpr std::size_t LOCK_AT = HIST_AT + 8 * 256;
constexpr std::size_t HEAD_AT = LOCK_AT + 4;
constexpr std::size_t TAIL_AT = LOCK_AT + 8;
constexpr std::size_t SLOTS_AT = LOCK_AT + 16;
constexpr uint32_t CAP = 64;
constexpr uint32_t CLOSED = 0x80000000u;
constexpr uint32_t COUNT = 0x7fffffffu;

using Bytes = std::vector<std::byte>;

// --- the host's own tools, on the arena `base.hpp` hands over -------------

// The `T` at `offset` of the arena, as an atomic the host shares with running
// programs: the location their atomics reach.
template <class T>
std::atomic_ref<T> atomic(const base::Driver &driver, std::size_t offset) {
    return std::atomic_ref<T>(*reinterpret_cast<T *>(driver.arena().data() + offset));
}

template <class T>
T read(const base::Driver &driver, std::size_t offset) {
    return atomic<T>(driver, offset).load(std::memory_order_relaxed);
}

// Sleeps while the arena's word at `offset` holds `expected`, until woken or
// spuriously, through the system call programs wait with, so the two meet.
void wait(const base::Driver &driver, std::size_t offset, uint32_t expected) {
    auto *addr = reinterpret_cast<uint32_t *>(driver.arena().data() + offset);
#if defined(__linux__)
    syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
#elif defined(__APPLE__)
    os_sync_wait_on_address(addr, expected, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE);
#elif defined(_WIN32)
    WaitOnAddress(addr, &expected, 4, INFINITE);
#endif
}

// Wakes one waiter on the arena's word at `offset` when `count` is 1, all otherwise.
void wake(const base::Driver &driver, std::size_t offset, int32_t count = -1) {
    auto *addr = reinterpret_cast<uint32_t *>(driver.arena().data() + offset);
#if defined(__linux__)
    syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, count == 1 ? 1 : INT_MAX, nullptr, nullptr, 0);
#elif defined(__APPLE__)
    if (count == 1)
        os_sync_wake_by_address_any(addr, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
    else
        os_sync_wake_by_address_all(addr, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
#elif defined(_WIN32)
    if (count == 1)
        WakeByAddressSingle(addr);
    else
        WakeByAddressAll(addr);
#endif
}

base::Driver load_file(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw base::Error("load: cannot read " + path.string());
    std::vector<char> bytes{std::istreambuf_iterator<char>(in), {}};
    return base::Driver::load(std::as_bytes(std::span(bytes)));
}

[[noreturn]] void fail(const std::string &why) {
    std::fprintf(stderr, "FAILED: %s\n", why.c_str());
    std::exit(1);
}

void expect(bool ok, const std::string &why) {
    if (!ok) fail(why);
}

// What the host sent, kept independently of the program.
struct Ledger {
    std::atomic<uint64_t> sum{0};
    std::atomic<uint64_t> calls{0};
    std::array<std::atomic<uint64_t>, 256> hist{};
    // Each call's (total before it, its sum): together they must tile
    // [0, total) with no gap and no overlap.
    std::mutex tickets_mutex;
    std::vector<std::pair<uint64_t, uint64_t>> tickets;
};

// Call `add`, found once by each section and called directly, and check its
// answer against the host's own sum.
uint64_t add(const base::Function &add_fn, Ledger &ledger, const Bytes &data) {
    uint64_t sum = 0;
    for (std::byte b : data) {
        sum += std::to_integer<uint64_t>(b);
        ledger.hist[std::to_integer<std::size_t>(b)].fetch_add(1, std::memory_order_relaxed);
    }
    std::array<std::byte, 8> out{};
    int64_t status = add_fn(data, out);
    expect(static_cast<uint64_t>(status) == sum, "add answered a sum the host did not compute");
    uint64_t before;
    std::memcpy(&before, out.data(), 8);
    ledger.sum.fetch_add(sum, std::memory_order_relaxed);
    ledger.calls.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard lock(ledger.tickets_mutex);
        ledger.tickets.emplace_back(before, sum);
    }
    return sum;
}

Bytes bytes_of(const std::string &s) {
    Bytes out(s.size());
    std::memcpy(out.data(), s.data(), s.size());
    return out;
}

Bytes random_bytes(uint64_t seed, std::size_t n) {
    std::mt19937_64 rng(seed);
    Bytes out(n);
    for (auto &b : out) b = static_cast<std::byte>(rng());
    return out;
}

void check_against_ledger(const base::Driver &driver, Ledger &ledger);

// --- 1. Bare std::thread -----------------------------------------------------
// Each thread runs its own random walk and sends the positions it visits.
void bare_threads(const base::Driver &driver, Ledger &ledger, unsigned n) {
    const auto add_fn = driver.function("add");
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < n; ++t) {
        threads.emplace_back([&add_fn, &ledger, t] {
            std::mt19937 rng(t);
            int pos = 128;
            for (int round = 0; round < 50; ++round) {
                Bytes path;
                for (int step = 0; step < 512; ++step) {
                    pos = std::clamp(pos + static_cast<int>(rng() % 7) - 3, 0, 255);
                    path.push_back(static_cast<std::byte>(pos));
                }
                add(add_fn, ledger, path);
            }
        });
    }
    for (auto &t : threads) t.join();
}

// --- 2. A thread pool: a job queue under a mutex and a condition variable ---
class Pool {
public:
    explicit Pool(unsigned n) {
        for (unsigned i = 0; i < n; ++i) workers_.emplace_back([this] { run(); });
    }
    // Drains the queue, then joins.
    ~Pool() {
        {
            std::lock_guard lock(mutex_);
            closing_ = true;
        }
        ready_.notify_all();
        for (auto &w : workers_) w.join();
    }
    void submit(std::function<void()> job) {
        {
            std::lock_guard lock(mutex_);
            jobs_.push_back(std::move(job));
        }
        ready_.notify_one();
    }

private:
    void run() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [this] { return closing_ || !jobs_.empty(); });
                if (jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
    }
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> jobs_;
    bool closing_ = false;
    std::vector<std::thread> workers_;
};

// Each job finds the primes in its own range and sends them as text.
void thread_pool(const base::Driver &driver, Ledger &ledger, unsigned n) {
    const auto add_fn = driver.function("add");
    Pool pool(n);
    for (unsigned job = 0; job < 64; ++job) {
        pool.submit([&add_fn, &ledger, job] {
            std::string text;
            for (unsigned k = job * 1000; k < (job + 1) * 1000; ++k) {
                bool prime = k >= 2;
                for (unsigned d = 2; d * d <= k && prime; ++d) prime = k % d != 0;
                if (prime) text += std::to_string(k) + ",";
            }
            add(add_fn, ledger, bytes_of(text));
        });
    }
}

// --- 3. std::async and futures ----------------------------------------------
// Each task sorts its own data before sending it and answers the sum the
// program returned; the caller adds the futures up.
void async_tasks(const base::Driver &driver, Ledger &ledger, unsigned n) {
    const auto add_fn = driver.function("add");
    std::vector<std::future<uint64_t>> futures;
    uint64_t expected = 0;
    for (unsigned t = 0; t < 4 * n; ++t) {
        Bytes data = random_bytes(1000 + t, 4096);
        for (std::byte b : data) expected += std::to_integer<uint64_t>(b);
        futures.push_back(std::async(std::launch::async, [&add_fn, &ledger, data]() mutable {
            std::sort(data.begin(), data.end());
            return add(add_fn, ledger, data);
        }));
    }
    uint64_t got = 0;
    for (auto &f : futures) got += f.get();
    expect(got == expected, "the futures' sums do not add up");
}

// --- 4. Producers and consumers over a bounded queue ------------------------
class Bounded {
public:
    explicit Bounded(std::size_t cap) : cap_(cap) {}
    void push(Bytes item) {
        std::unique_lock lock(mutex_);
        not_full_.wait(lock, [this] { return items_.size() < cap_; });
        items_.push_back(std::move(item));
        not_empty_.notify_one();
    }
    // Empty once the queue is closed and drained.
    std::optional<Bytes> pop() {
        std::unique_lock lock(mutex_);
        not_empty_.wait(lock, [this] { return closed_ || !items_.empty(); });
        if (items_.empty()) return std::nullopt;
        Bytes item = std::move(items_.front());
        items_.pop_front();
        not_full_.notify_one();
        return item;
    }
    void close() {
        std::lock_guard lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable not_full_, not_empty_;
    std::deque<Bytes> items_;
    std::size_t cap_;
    bool closed_ = false;
};

void producers_consumers(const base::Driver &driver, Ledger &ledger, unsigned n) {
    const auto add_fn = driver.function("add");
    Bounded queue(8);
    unsigned producers = std::max(1u, n / 2), consumers = std::max(1u, n - producers);
    std::vector<std::thread> made, used;
    for (unsigned p = 0; p < producers; ++p)
        made.emplace_back([&queue, p] {
            for (unsigned i = 0; i < 40; ++i) queue.push(random_bytes(p * 1000 + i, 2048 + i));
        });
    for (unsigned c = 0; c < consumers; ++c)
        used.emplace_back([&] {
            while (auto item = queue.pop()) add(add_fn, ledger, *item);
        });
    for (auto &t : made) t.join();
    queue.close();
    for (auto &t : used) t.join();
}

// --- 5. Phases: std::jthread, std::barrier and a stop_token monitor ---------
// Between phases every worker waits at the barrier, so its completion step sees
// a quiet driver and checks the program's total against the ledger exactly.
// Meanwhile a monitor reads the total straight from memory while calls run,
// and checks it never goes backwards.
void barrier_phases(const base::Driver &driver, Ledger &ledger, unsigned n) {
    const auto add_fn = driver.function("add");
    constexpr int PHASES = 5;
    int phase = 0;
    std::barrier sync(n, [&]() noexcept {
        uint64_t total = static_cast<uint64_t>(driver.function("total")());
        if (total != ledger.sum.load()) fail("the total and the ledger disagree between phases");
        ++phase;
    });

    std::jthread monitor([&driver](std::stop_token stop) {
        uint64_t last = 0;
        while (!stop.stop_requested()) {
            uint64_t now = read<uint64_t>(driver, TOTAL_AT);
            if (now < last) fail("the total went backwards");
            last = now;
            std::this_thread::yield();
        }
    });

    {
        std::vector<std::jthread> workers;
        for (unsigned t = 0; t < n; ++t)
            workers.emplace_back([&, t] {
                for (int p = 0; p < PHASES; ++p) {
                    // Each phase sends twice as much as the one before.
                    add(add_fn, ledger, random_bytes(t * 100 + p, std::size_t{256} << p));
                    sync.arrive_and_wait();
                }
            });
    }  // the workers join here
    monitor.request_stop();
    expect(phase == PHASES, "not every phase completed");
}

// --- 6. Shared ownership ----------------------------------------------------
// The threads hold their own copies of the driver, and the main thread drops
// its own before they start. The last thread to finish checks the program
// against the ledger, and its copy, the last one, frees the driver.
void shared_ownership(std::optional<base::Driver> &mine, Ledger &ledger, unsigned n) {
    std::latch start(1);
    std::atomic<unsigned> running{n};
    std::vector<std::jthread> threads;
    for (unsigned t = 0; t < n; ++t)
        threads.emplace_back([copy = *mine, &ledger, &start, &running, t] {
            start.wait();
            const auto add_fn = copy.function("add");
            for (int i = 0; i < 20; ++i) add(add_fn, ledger, random_bytes(77 + t * 31 + i, 1024));
            if (running.fetch_sub(1) == 1) check_against_ledger(copy, ledger);
        });
    mine.reset();
    start.count_down();
}

// --- 7. A ring buffer, shared by the program and the host -------------------
// The program's `push` and `pop` guard the ring with a lock built from
// compare-and-swap and wait/wake, and sleep while it is full or empty.
// Producer threads call `push`, and consumers come in two kinds at once:
// threads calling the program's `pop`, and a host thread that pops straight
// from the program's memory, taking the same lock through `std::atomic_ref`
// and sleeping with base's wait and wake, which meet the program's.

// The program's lock, taken from the host. A `BasicLockable`, so
// `std::lock_guard` works with it.
class RingLock {
public:
    explicit RingLock(const base::Driver &driver)
        : driver_(driver), word_(atomic<uint32_t>(driver, LOCK_AT)) {}
    void lock() {
        uint32_t free = 0;
        if (word_.compare_exchange_strong(free, 1, std::memory_order_acquire)) return;
        while (word_.exchange(2, std::memory_order_acquire) != 0) wait(driver_, LOCK_AT, 2);
    }
    void unlock() {
        if (word_.exchange(0, std::memory_order_release) == 2) wake(driver_, LOCK_AT, 1);
    }

private:
    const base::Driver &driver_;
    std::atomic_ref<uint32_t> word_;
};

// The next item, popped by the host itself; empty once the ring is closed and
// drained. The slot is read with a plain copy: the program wrote it under the
// lock, and the host reads it under the lock.
std::optional<uint64_t> host_pop(const base::Driver &driver, RingLock &lock) {
    auto head = atomic<uint32_t>(driver, HEAD_AT);
    auto tail = atomic<uint32_t>(driver, TAIL_AT);
    for (;;) {
        uint32_t h;
        std::optional<uint64_t> item;
        {
            std::lock_guard guard(lock);
            h = head.load(std::memory_order_acquire);
            uint32_t t = tail.load(std::memory_order_relaxed);
            if (((h - t) & COUNT) != 0) {
                uint64_t v;
                std::memcpy(&v, driver.arena().data() + SLOTS_AT + 8 * (t % CAP), 8);
                item = v;
                tail.store(t + 1, std::memory_order_release);
            } else if (h & CLOSED) {
                return std::nullopt;
            }
        }
        if (item) {
            wake(driver, TAIL_AT);  // producers sleep on the tail
            return item;
        }
        wait(driver, HEAD_AT, h);  // until a push or the close changes the head
    }
}

void ring_buffer(const base::Driver &driver, Ledger &, unsigned n) {
    const auto push = driver.function("push"), pop = driver.function("pop"), close = driver.function("close");
    const unsigned producers = std::max(1u, n / 2), poppers = std::max(1u, n - producers);
    constexpr uint64_t PER_PRODUCER = 5000;

    std::mutex got_mutex;
    std::vector<uint64_t> got;
    std::atomic<uint64_t> by_host{0};
    {
        std::vector<std::jthread> consumers;
        for (unsigned c = 0; c < poppers; ++c)
            consumers.emplace_back([&] {
                std::vector<uint64_t> mine;
                std::array<std::byte, 8> out{};
                while (pop({}, out) == 1) {
                    uint64_t item;
                    std::memcpy(&item, out.data(), 8);
                    mine.push_back(item);
                }
                std::lock_guard g(got_mutex);
                got.insert(got.end(), mine.begin(), mine.end());
            });
        consumers.emplace_back([&] {
            RingLock lock(driver);
            std::vector<uint64_t> mine;
            while (auto item = host_pop(driver, lock)) mine.push_back(*item);
            by_host = mine.size();
            std::lock_guard g(got_mutex);
            got.insert(got.end(), mine.begin(), mine.end());
        });
        {
            std::vector<std::jthread> makers;
            for (unsigned p = 0; p < producers; ++p)
                makers.emplace_back([&, p] {
                    for (uint64_t i = 0; i < PER_PRODUCER; ++i) {
                        uint64_t item = p * PER_PRODUCER + i;
                        std::array<std::byte, 8> in;
                        std::memcpy(in.data(), &item, 8);
                        expect(push(in) == 1, "push refused an item");
                    }
                });
        }  // the producers join here
        close();
    }  // the consumers join here

    std::sort(got.begin(), got.end());
    expect(got.size() == producers * PER_PRODUCER, "the ring lost or repeated an item");
    for (uint64_t i = 0; i < got.size(); ++i) expect(got[i] == i, "the ring lost or repeated an item");
    std::printf("  %-20s %5zu items, %llu popped by the host directly\n", "ring buffer", got.size(),
                static_cast<unsigned long long>(by_host.load()));
}

void check_against_ledger(const base::Driver &driver, Ledger &ledger) {
    uint64_t total = static_cast<uint64_t>(driver.function("total")());
    uint64_t calls = static_cast<uint64_t>(driver.function("calls")());
    expect(total == ledger.sum, "total: program " + std::to_string(total) + ", host " +
                                    std::to_string(ledger.sum.load()));
    expect(calls == ledger.calls, "calls: program " + std::to_string(calls) + ", host " +
                                      std::to_string(ledger.calls.load()));
    std::array<uint64_t, 256> hist;
    for (std::size_t b = 0; b < hist.size(); ++b) hist[b] = read<uint64_t>(driver, HIST_AT + 8 * b);
    for (std::size_t b = 0; b < 256; ++b)
        expect(hist[b] == ledger.hist[b], "histogram counter " + std::to_string(b));
    expect(read<uint64_t>(driver, TOTAL_AT) == total, "memory and `total` disagree");
    expect(read<uint64_t>(driver, CALLS_AT) == calls, "memory and `calls` disagree");

    // Each call added its sum at the total it saw: sorted, the ranges must
    // meet end to start, from zero to the total.
    auto tickets = ledger.tickets;
    std::sort(tickets.begin(), tickets.end());
    uint64_t at = 0;
    for (auto [before, sum] : tickets) {
        expect(before == at, "a call's addition overlapped or skipped another's");
        at += sum;
    }
    expect(at == total, "the calls' ranges do not reach the total");
}

void report(const char *name, uint64_t calls, const Ledger &ledger) {
    std::printf("  %-20s %5llu calls, total %llu\n", name, static_cast<unsigned long long>(calls),
                static_cast<unsigned long long>(ledger.sum.load()));
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <shared_sum.cbor> [threads]\n", argv[0]);
        return 2;
    }
    unsigned n = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2]))
                          : std::max(2u, std::thread::hardware_concurrency());
    if (n == 0) fail("threads must be at least 1");
    try {
        std::optional<base::Driver> driver = load_file(argv[1]);
        std::printf("loaded %zu bytes of memory; %u threads\n", driver->arena().size(), n);
        Ledger ledger;

        std::pair<const char *, void (*)(const base::Driver &, Ledger &, unsigned)> sections[] = {
            {"std::thread", bare_threads},
            {"thread pool", thread_pool},
            {"std::async", async_tasks},
            {"producers/consumers", producers_consumers},
            {"barrier phases", barrier_phases},
            {"ring buffer", ring_buffer},
        };
        for (auto [name, run] : sections) {
            uint64_t calls = ledger.calls;
            run(*driver, ledger, n);
            check_against_ledger(*driver, ledger);
            if (ledger.calls != calls) report(name, ledger.calls - calls, ledger);
        }
        uint64_t calls = ledger.calls;
        shared_ownership(driver, ledger, n);
        report("shared ownership", ledger.calls - calls, ledger);
    } catch (const base::Error &e) {
        fail(e.what());
    }
    std::printf("ok: the program's memory agrees with every call the host made\n");
    return 0;
}
