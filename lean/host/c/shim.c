/* Lean's calling convention over `base/src/c_api.rs`.
 *
 * The runtime's C ABI answers a failure value, with a message where a load
 * fails. Lean's convention is an `IO` result carrying either the value or the
 * error, so each wrapper calls, and on failure turns that into the error Lean
 * throws. That way the Lean side has no error codes in it.
 *
 * Nothing here decides anything. A wrapper that does more than translate a
 * calling convention belongs in `c_api.rs`, where it is testable from Rust.
 */

#include <lean/lean.h>
#include <stdint.h>
#include <string.h>

/* Declared rather than included, so the shim's build needs nothing outside
 * this package. These must match `base/include/base.h`. */
typedef int64_t (*base_function)(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len);
void *base_driver_load(const uint8_t *artifact, size_t len, char *err, size_t err_cap);
base_function base_driver_function(const void *driver, const uint8_t *name, size_t name_len);
uint8_t *base_driver_arena(const void *driver, size_t *len);
void base_driver_free(void *driver);

/* `message` as the `IO` error Lean throws. */
static lean_obj_res base_io_error(const char *message) {
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(message)));
}

LEAN_EXPORT lean_obj_res lean_base_driver_load(b_lean_obj_arg artifact, lean_obj_arg w) {
    (void)w;
    char err[1024];
    void *handle = base_driver_load(lean_sarray_cptr(artifact), lean_sarray_size(artifact),
                                    err, sizeof err);
    if (handle == NULL) {
        return base_io_error(err);
    }
    return lean_io_result_mk_ok(lean_box_usize((size_t)handle));
}

/* The output buffer is allocated here rather than taken from the caller: a
 * `ByteArray` Lean already holds may be shared, and writing through it would
 * mutate a value another reference can observe. A fresh one is unshared by
 * construction, and returning it is how the caller reads what ran.
 *
 * The pair is the buffer and the status the program returned, which is the one
 * value it answers with without agreeing on a place in the arena to leave it. */
LEAN_EXPORT lean_obj_res lean_base_driver_execute(size_t handle, b_lean_obj_arg name,
                                                  b_lean_obj_arg input, size_t output_len,
                                                  lean_obj_arg w) {
    (void)w;
    base_function entry = base_driver_function((const void *)handle,
                                          (const uint8_t *)lean_string_cstr(name),
                                          lean_string_size(name) - 1);
    if (entry == NULL) {
        return base_io_error("the artifact has no function by that name");
    }
    lean_object *output = lean_alloc_sarray(1, output_len, output_len);
    int64_t status = entry(lean_sarray_cptr(input), lean_sarray_size(input),
                           lean_sarray_cptr(output), output_len);
    lean_object *pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, output);
    lean_ctor_set(pair, 1, lean_box_uint64((uint64_t)status));
    return lean_io_result_mk_ok(pair);
}

/* A copy, because a `ByteArray` outlives the call that read it. A range
 * reaching past the end is an error rather than a short answer, so a truncated
 * read cannot be mistaken for a result. */
LEAN_EXPORT lean_obj_res lean_base_driver_read(size_t handle, size_t offset, size_t len,
                                                      lean_obj_arg w) {
    (void)w;
    size_t have = 0;
    const uint8_t *memory = base_driver_arena((const void *)handle, &have);
    if (memory == NULL || offset > have || len > have - offset) {
        return base_io_error("that range is outside the arena");
    }
    /* Programs may be writing the arena on other threads, so it is read as
     * they write it: each aligned 8-byte word with one atomic load, and the
     * bytes either side one at a time. */
    lean_object *dst = lean_alloc_sarray(1, len, len);
    uint8_t *out = lean_sarray_cptr(dst);
    size_t i = 0;
    while (i < len) {
        const uint8_t *at = memory + offset + i;
        if (((uintptr_t)at & 7) == 0 && len - i >= 8) {
            uint64_t word = __atomic_load_n((const uint64_t *)at, __ATOMIC_RELAXED);
            memcpy(out + i, &word, 8);
            i += 8;
        } else {
            out[i] = __atomic_load_n(at, __ATOMIC_RELAXED);
            i += 1;
        }
    }
    return lean_io_result_mk_ok(dst);
}

LEAN_EXPORT lean_obj_res lean_base_driver_arena_size(size_t handle, lean_obj_arg w) {
    (void)w;
    size_t len = 0;
    base_driver_arena((const void *)handle, &len);
    return lean_io_result_mk_ok(lean_box_usize(len));
}

LEAN_EXPORT lean_obj_res lean_base_driver_free(size_t handle, lean_obj_arg w) {
    (void)w;
    base_driver_free((void *)handle);
    return lean_io_result_mk_ok(lean_box(0));
}
