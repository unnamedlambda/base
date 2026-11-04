// base's C API as C++20. A `Driver` is a shared handle, freed with its last
// copy; a `Function` holds a share, so no function outlives its driver.
#pragma once

#include "base.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace base {

// A failed call, and why.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Driver;

// A function of a driver, as the C function it is, holding a share of the driver.
class Function {
public:
    // Runs the function on `input` and `output` in place; any thread, any number at once.
    int64_t operator()(std::span<const std::byte> input = {}, std::span<std::byte> output = {}) const {
        return fn_(reinterpret_cast<const uint8_t *>(input.data()), input.size(),
                   reinterpret_cast<uint8_t *>(output.data()), output.size());
    }

private:
    friend class Driver;
    Function(std::shared_ptr<BaseDriver> driver, BaseFunction fn) : driver_(std::move(driver)), fn_(fn) {}
    std::shared_ptr<BaseDriver> driver_;
    BaseFunction fn_;
};

class Driver {
public:
    // Loads an encoded artifact.
    static Driver load(std::span<const std::byte> artifact) {
        char err[1024];
        BaseDriver *raw = base_driver_load(reinterpret_cast<const uint8_t *>(artifact.data()),
                                           artifact.size(), err, sizeof err);
        if (raw == nullptr) throw Error(std::string("load: ") + err);
        return Driver(std::shared_ptr<BaseDriver>(raw, base_driver_free));
    }

    // The function named `name`.
    Function function(std::string_view name) const {
        BaseFunction fn = base_driver_function(handle_.get(), reinterpret_cast<const uint8_t *>(name.data()),
                                               name.size());
        if (fn == nullptr) throw Error("function: the artifact has no function named " + std::string(name));
        return Function(handle_, fn);
    }

    // The arena, valid for as long as any copy of this driver lives. While
    // calls run, bytes a program may touch are reached only atomically.
    std::span<std::byte> arena() const {
        std::size_t len = 0;
        auto *p = reinterpret_cast<std::byte *>(base_driver_arena(handle_.get(), &len));
        return {p, len};
    }

private:
    explicit Driver(std::shared_ptr<BaseDriver> handle) : handle_(std::move(handle)) {}
    std::shared_ptr<BaseDriver> handle_;
};

}  // namespace base
