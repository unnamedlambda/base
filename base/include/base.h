/* base's C API, exported by libbase (`base/src/c_api.rs`).
 *
 * `base_driver_load` compiles an artifact as `dlopen` loads a library,
 * `base_driver_function` answers a function by name as `dlsym` does, and the
 * host calls it directly, from any number of threads at once. Only a load
 * fails with something to say; a trap in a program ends the process.
 *
 * Waiting on a word of the arena, waking, and the clock are the system's own
 * calls (futex, WaitOnAddress, os_sync_wait_on_address; CLOCK_MONOTONIC,
 * CLOCK_UPTIME_RAW, QueryPerformanceCounter), so hosts and programs meet at the
 * same words; base.hpp has them as base::wait, base::wake and base::now.
 */
#ifndef BASE_H
#define BASE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BaseDriver BaseDriver;

/* Every function a host calls: runs on `in`, writes `out` in place, and returns
 * the program's status. A buffer may be NULL with length 0. The caller's thread
 * is in the default floating-point environment; `in`, `out` and the arena must
 * not overlap for a function Lean builds, and are the program's for the call. */
typedef int64_t (*BaseFunction)(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len);

/* A driver for the encoded artifact, opening the system libraries it names;
 * release with `base_driver_free`. NULL on failure, with why written to `err`
 * (may be NULL), NUL-terminated and cut to `err_cap` bytes. */
BaseDriver *base_driver_load(const uint8_t *artifact, size_t len, char *err, size_t err_cap);

/* The function named `name` (UTF-8, not NUL-terminated), callable until the
 * driver is freed; NULL if there is none. */
BaseFunction base_driver_function(const BaseDriver *driver, const uint8_t *name, size_t name_len);

/* The arena and its size, page-aligned and fixed until the driver is freed.
 * While functions run, bytes a program may touch are reached only atomically. */
uint8_t *base_driver_arena(const BaseDriver *driver, size_t *len);

/* Releases a driver; NULL does nothing. No call on it, and no thread a program
 * of it started, may still be running. */
void base_driver_free(BaseDriver *driver);

#ifdef __cplusplus
}
#endif

#endif
