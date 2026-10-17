// base's C API as C++20: a `base::Driver` is a shared handle, copied freely
// between threads, and the driver is freed when the last copy goes. A
// `Function` is a function of it as the C function it is, holding its own share.
//
// That is the one rule the C API leaves to its host: no function may run
// once `base_driver_free` has. A `std::shared_ptr` keeps it by construction,
// because whatever can call holds a share, and a share keeps the driver alive.
#pragma once

#include "base.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <os/os_sync_wait_on_address.h>
#include <time.h>
#elif defined(_WIN32)
#include <windows.h>
#pragma comment(lib, "Synchronization.lib")
#endif

namespace base {

// Waiting on a word of the arena, and the clock: the system's own calls, the same
// ones a program's waits, wakes and clock readings are. So a host's thread and a
// program's meet at a word, and read one clock, with nothing of base between.
// (`std::atomic::wait` is not this: libstdc++ and libc++ keep a waiter count of
// their own, and may skip the system call a waiter of another kind needs.)

// Sleep while the word at `addr` holds `expected`, until a wake on it,
// `timeout_ns` nanoseconds (none if negative), or for no reason. Loop on the
// condition. `addr` is 4-byte aligned, and every access to the word while
// waiters may exist is atomic.
inline void wait(uint32_t *addr, uint32_t expected, int64_t timeout_ns = -1) {
#if defined(__linux__)
    timespec ts{static_cast<time_t>(timeout_ns / 1000000000), static_cast<long>(timeout_ns % 1000000000)};
    syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, expected, timeout_ns < 0 ? nullptr : &ts, nullptr, 0);
#elif defined(__APPLE__)
    if (timeout_ns < 0)
        os_sync_wait_on_address(addr, expected, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE);
    else
        os_sync_wait_on_address_with_timeout(addr, expected, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                             OS_CLOCK_MACH_ABSOLUTE_TIME, static_cast<uint64_t>(timeout_ns));
#elif defined(_WIN32)
    DWORD ms = timeout_ns < 0 ? INFINITE : static_cast<DWORD>((timeout_ns + 999999) / 1000000);
    WaitOnAddress(addr, &expected, 4, ms);
#endif
}

// Wake one thread waiting on the word at `addr` when `count` is 1, every one for any other
// `count` but 0.
inline void wake(uint32_t *addr, int32_t count = -1) {
    if (count == 0) return;
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

// Nanoseconds on the system's monotonic clock, the one programs read.
inline int64_t now() {
#if defined(_WIN32)
    LARGE_INTEGER ticks, freq;
    QueryPerformanceCounter(&ticks);
    QueryPerformanceFrequency(&freq);
    return static_cast<int64_t>(static_cast<__int128>(ticks.QuadPart) * 1000000000 / freq.QuadPart);
#else
#if defined(__APPLE__)
    clockid_t clock = CLOCK_UPTIME_RAW;
#else
    clockid_t clock = CLOCK_MONOTONIC;
#endif
    timespec t;
    clock_gettime(clock, &t);
    return static_cast<int64_t>(t.tv_sec) * 1000000000 + t.tv_nsec;
#endif
}

// A failed call, and why.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Driver;

// A function of a driver, as the C function it is: calling it calls the
// program directly. It holds a share of the driver, so the driver lives as long
// as it does; copying it copies the share, and calling it does not touch it.
class Function {
public:
    // Run the function with `input`, answering in `output`, and return the
    // program's status. Any number of threads may call at once.
    int64_t operator()(std::span<const std::byte> input = {}, std::span<std::byte> output = {}) const {
        return fn_(reinterpret_cast<const uint8_t *>(input.data()), input.size(),
                   reinterpret_cast<uint8_t *>(output.data()), output.size());
    }

    // The C function itself, for code that calls it through a pointer while
    // keeping this `Function`, or its driver, alive.
    BaseFunction get() const { return fn_; }

private:
    friend class Driver;
    Function(std::shared_ptr<BaseDriver> driver, BaseFunction fn) : driver_(std::move(driver)), fn_(fn) {}
    std::shared_ptr<BaseDriver> driver_;
    BaseFunction fn_;
};

class Driver {
public:
    // Compile an artifact from its encoded bytes.
    static Driver load(std::span<const std::byte> artifact) {
        char err[1024];
        BaseDriver *raw = base_driver_load(reinterpret_cast<const uint8_t *>(artifact.data()),
                                           artifact.size(), err, sizeof err);
        if (raw == nullptr) throw Error(std::string("load: ") + err);
        return Driver(std::shared_ptr<BaseDriver>(raw, base_driver_free));
    }

    // Compile the artifact a generator wrote to `path`.
    static Driver load(const std::filesystem::path &path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw Error("load: cannot read " + path.string());
        std::vector<char> bytes{std::istreambuf_iterator<char>(in), {}};
        return load(std::as_bytes(std::span(bytes)));
    }

    // The function named `name`, found once and called as often as the host
    // likes.
    Function function(std::string_view name) const {
        BaseFunction fn = lookup(name);
        if (fn == nullptr) throw Error("function: the artifact has no function named " + std::string(name));
        return Function(handle_, fn);
    }

    // Copy `out.size()` bytes of the arena from `offset`, sound while
    // programs run: each naturally aligned 8-byte word is loaded whole, as one
    // atomic, and the bytes either side one at a time.
    void read(std::size_t offset, std::span<std::byte> out) const {
        auto mem = arena();
        if (offset > mem.size() || out.size() > mem.size() - offset)
            throw Error("read: " + std::to_string(offset) + "+" + std::to_string(out.size()) +
                        " is outside the " + std::to_string(mem.size()) + " bytes of the arena");
        std::size_t i = 0;
        while (i < out.size()) {
            std::byte *at = mem.data() + offset + i;
            if (reinterpret_cast<std::uintptr_t>(at) % 8 == 0 && out.size() - i >= 8) {
                uint64_t word = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(at))
                                    .load(std::memory_order_relaxed);
                std::memcpy(out.data() + i, &word, 8);
                i += 8;
            } else {
                out[i] = std::atomic_ref<std::byte>(*at).load(std::memory_order_relaxed);
                i += 1;
            }
        }
    }

    // A trivially copyable value read from the arena at `offset`.
    template <class T>
    T read(std::size_t offset) const {
        T value;
        read(offset, std::as_writable_bytes(std::span(&value, 1)));
        return value;
    }

    // Sleep while the 32-bit word at `offset` of the arena holds
    // `expected`, as `base::wait`: programs wait and wake at the same words, so
    // either side wakes the other. Loop on the condition.
    void wait(std::size_t offset, uint32_t expected, int64_t timeout_ns = -1) const {
        base::wait(word(offset), expected, timeout_ns);
    }

    // Wake waiters on the word at `offset`, a program's and the host's alike,
    // as `base::wake`.
    void wake(std::size_t offset, int32_t count = -1) const { base::wake(word(offset), count); }

    // The arena itself, valid for as long as any copy of this driver lives.
    // While calls run, bytes a program may touch are reached only
    // atomically, through `atomic` below; the rest are the host's.
    std::span<std::byte> arena() const {
        std::size_t len = 0;
        auto *p = reinterpret_cast<std::byte *>(base_driver_arena(handle_.get(), &len));
        return {p, len};
    }

    // The `T` at `offset` of the arena, as an atomic the host
    // shares with running programs: the same location their atomics reach,
    // with whatever memory order the host asks for.
    template <class T>
    std::atomic_ref<T> atomic(std::size_t offset) const {
        auto mem = arena();
        if (offset > mem.size() || sizeof(T) > mem.size() - offset)
            throw Error("atomic: offset " + std::to_string(offset) + " is outside the " +
                        std::to_string(mem.size()) + " bytes of the arena");
        if (offset % std::atomic_ref<T>::required_alignment != 0)
            throw Error("atomic: offset " + std::to_string(offset) + " is not aligned for it");
        return std::atomic_ref<T>(*reinterpret_cast<T *>(mem.data() + offset));
    }

private:
    // The word at `offset`, checked to be one: in the arena and 4-byte aligned.
    uint32_t *word(std::size_t offset) const {
        atomic<uint32_t>(offset);
        return reinterpret_cast<uint32_t *>(arena().data() + offset);
    }

    BaseFunction lookup(std::string_view name) const {
        return base_driver_function(handle_.get(), reinterpret_cast<const uint8_t *>(name.data()),
                                    name.size());
    }

    explicit Driver(std::shared_ptr<BaseDriver> handle) : handle_(std::move(handle)) {}
    std::shared_ptr<BaseDriver> handle_;
};

}  // namespace base
