/* Lean's calling convention over `base/src/capi.rs`.
 *
 * The runtime's C ABI answers a failure value and leaves a message on the
 * thread. Lean's convention is an `IO` result carrying either the value or the
 * error, so every wrapper here is the same three lines: call, and on failure
 * turn the thread's message into the error Lean throws. That way the Lean side
 * has no error codes in it.
 *
 * Nothing here decides anything. A wrapper that does more than translate a
 * calling convention belongs in `capi.rs`, where it is testable from Rust.
 */

#include <lean/lean.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Declared rather than included: the runtime is a Rust cdylib and has no
 * header. These must match `base/src/capi.rs`. */
void *base_driver_load(const uint8_t *artifact, size_t len);
int32_t base_driver_execute(void *driver, const uint8_t *name, size_t name_len,
                            const uint8_t *input, size_t input_len,
                            uint8_t *output, size_t output_len, int64_t *status);
const uint8_t *base_driver_memory(const void *driver, size_t *len);
size_t base_last_error(uint8_t *buf, size_t cap);
void base_driver_free(void *driver);

/* The thread's last error as the `IO` error Lean throws.
 *
 * `fallback` is what a caller sees when a call failed but left no message,
 * which would otherwise surface as an empty exception. */
static lean_obj_res base_io_error(const char *fallback) {
    size_t n = base_last_error(NULL, 0);
    if (n == 0) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string(fallback)));
    }
    char *buf = (char *)malloc(n + 1);
    if (buf == NULL) {
        return lean_io_result_mk_error(
            lean_mk_io_user_error(lean_mk_string(fallback)));
    }
    base_last_error((uint8_t *)buf, n);
    buf[n] = '\0';
    lean_object *message = lean_mk_string(buf);
    free(buf);
    return lean_io_result_mk_error(lean_mk_io_user_error(message));
}

LEAN_EXPORT lean_obj_res lean_base_driver_load(b_lean_obj_arg artifact, lean_obj_arg w) {
    (void)w;
    void *handle = base_driver_load(lean_sarray_cptr(artifact), lean_sarray_size(artifact));
    if (handle == NULL) {
        return base_io_error("base_driver_load failed");
    }
    return lean_io_result_mk_ok(lean_box_usize((size_t)handle));
}

/* The output buffer is allocated here rather than taken from the caller: a
 * `ByteArray` Lean already holds may be shared, and writing through it would
 * mutate a value another reference can observe. A fresh one is unshared by
 * construction, and returning it is how the caller reads what ran.
 *
 * The pair is the buffer and the status the program returned, which is the one
 * value it answers with without agreeing on a place in memory to leave it. */
LEAN_EXPORT lean_obj_res lean_base_driver_execute(size_t handle, b_lean_obj_arg name,
                                                  b_lean_obj_arg input, size_t output_len,
                                                  lean_obj_arg w) {
    (void)w;
    lean_object *output = lean_alloc_sarray(1, output_len, output_len);
    int64_t status = 0;
    int32_t rc = base_driver_execute((void *)handle,
                                     (const uint8_t *)lean_string_cstr(name),
                                     lean_string_size(name) - 1,
                                     lean_sarray_cptr(input), lean_sarray_size(input),
                                     lean_sarray_cptr(output), output_len, &status);
    if (rc != 0) {
        lean_dec_ref(output);
        return base_io_error("base_driver_execute failed");
    }
    lean_object *pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, output);
    lean_ctor_set(pair, 1, lean_box_uint64((uint64_t)status));
    return lean_io_result_mk_ok(pair);
}

/* A copy, because the runtime's memory is borrowed only until the next call
 * and a `ByteArray` is not. A range reaching past the end is an error rather
 * than a short answer, so a truncated read cannot be mistaken for a result. */
LEAN_EXPORT lean_obj_res lean_base_driver_read_memory(size_t handle, size_t offset, size_t len,
                                                      lean_obj_arg w) {
    (void)w;
    size_t have = 0;
    const uint8_t *memory = base_driver_memory((const void *)handle, &have);
    if (memory == NULL || offset > have || len > have - offset) {
        return base_io_error("that range is outside the runtime's memory");
    }
    lean_object *dst = lean_alloc_sarray(1, len, len);
    memcpy(lean_sarray_cptr(dst), memory + offset, len);
    return lean_io_result_mk_ok(dst);
}

LEAN_EXPORT lean_obj_res lean_base_driver_memory_size(size_t handle, lean_obj_arg w) {
    (void)w;
    size_t len = 0;
    base_driver_memory((const void *)handle, &len);
    return lean_io_result_mk_ok(lean_box_usize(len));
}

LEAN_EXPORT lean_obj_res lean_base_driver_free(size_t handle, lean_obj_arg w) {
    (void)w;
    base_driver_free((void *)handle);
    return lean_io_result_mk_ok(lean_box(0));
}
