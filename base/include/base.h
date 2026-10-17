/* base's C API: load an artifact, get its functions as C functions, share
 * its arena, free it.
 *
 * These are the functions `base/src/capi.rs` exports from `libbase.so`; the
 * documentation there is the full contract and this header must match it.
 *
 * It works like a shared library: `base_driver_load` compiles an artifact as
 * `dlopen` loads a library, `base_driver_function` answers a function by name as
 * a plain C function pointer as `dlsym` does, and the host calls that function itself.
 * Nothing of base runs between host and program on a call.
 *
 * Threads: functions may be called from any number of threads at once, as
 * any C functions may. `base_driver_free` needs the driver to itself: no
 * function of it may be running, and none may be called after it.
 *
 * Errors: only a load can fail with something to say, and it writes why into
 * the caller's buffer. Base keeps no error state.
 *
 * What stays fixed, and why the API stays these four functions:
 *  - Every function a host calls has the one signature `BaseFunction`. What else a call
 *    needs (an instance, options, several results) travels in `in` and `out`,
 *    laid out by the program.
 *  - The arena never moves or grows: the program's code is linked to its
 *    address.
 *  - What an artifact runs with is decided when it is built, so a load takes
 *    nothing but the artifact.
 *  - Anything more a host reaches is a function of the program, rather
 *    than another function here.
 *  - A trap in a program ends the process: nothing of base runs on a call to
 *    catch one.
 */
#ifndef BASE_H
#define BASE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BaseDriver BaseDriver;

/* Every function a host calls: runs the program with `in`, answers in `out`,
 * which it reads and writes directly, and returns the program's status. */
typedef int64_t (*BaseFunction)(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len);

/* Compile an artifact (the `.cbor` a generator writes) and allocate its
 * arena. The system libraries its calls name that base knows (the CUDA
 * driver API and cuBLAS) are opened, as dlopen opens a library's
 * dependencies, running their initialisers; an artifact names a library,
 * never a file. Release with `base_driver_free`. NULL on failure, with why written
 * to `err` as a NUL-terminated string cut to fit `err_cap` bytes; `err` may be
 * NULL. */
BaseDriver *base_driver_load(const uint8_t *artifact, size_t len, char *err, size_t err_cap);

/* The function named `name` (UTF-8, not NUL-terminated), callable until the
 * driver is freed. NULL if there is none. */
BaseFunction base_driver_function(const BaseDriver *driver, const uint8_t *name, size_t name_len);

/* The arena itself, at an address that stays put until the driver is freed.
 * While functions run, bytes a program may touch are reached only with
 * atomic operations (`std::atomic_ref` in C++), as the program reaches them;
 * bytes no running program touches are the host's to use as it likes. It is
 * its own pages, mapped zeroed, so it starts page-aligned. */
uint8_t *base_driver_arena(const BaseDriver *driver, size_t *len);

/* Release a driver. NULL is accepted. No function of it may be running. */
void base_driver_free(BaseDriver *driver);

#ifdef __cplusplus
}
#endif

#endif
