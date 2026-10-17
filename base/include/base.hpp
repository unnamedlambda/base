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

namespace base {

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
    BaseFunction lookup(std::string_view name) const {
        return base_driver_function(handle_.get(), reinterpret_cast<const uint8_t *>(name.data()),
                                    name.size());
    }

    explicit Driver(std::shared_ptr<BaseDriver> handle) : handle_(std::move(handle)) {}
    std::shared_ptr<BaseDriver> handle_;
};

}  // namespace base
