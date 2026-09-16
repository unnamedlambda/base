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
void *base_new(const uint8_t *artifact_json, size_t len);
int32_t base_execute(void *handle, uint32_t fn_idx, const uint8_t *data,
                     size_t data_len, uint8_t *out, size_t out_len,
                     int64_t *status);
const uint8_t *base_memory(const void *handle, size_t *len);
size_t base_last_error(uint8_t *buf, size_t cap);
void base_free(void *handle);

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

LEAN_EXPORT lean_obj_res lean_base_new(b_lean_obj_arg artifact_json, lean_obj_arg w) {
    (void)w;
    void *handle = base_new(lean_sarray_cptr(artifact_json), lean_sarray_size(artifact_json));
    if (handle == NULL) {
        return base_io_error("base_new failed");
    }
    return lean_io_result_mk_ok(lean_box_usize((size_t)handle));
}

/* The out buffer is allocated here rather than taken from the caller: a
 * `ByteArray` Lean already holds may be shared, and writing through it would
 * mutate a value another reference can observe. A fresh one is unshared by
 * construction, and returning it is how the caller reads what ran.
 *
 * The pair is the buffer and the status the program returned, which is the one
 * value it answers with without agreeing on a place in memory to leave it. */
LEAN_EXPORT lean_obj_res lean_base_execute(size_t handle, uint32_t fn_idx,
                                           b_lean_obj_arg data, size_t out_len,
                                           lean_obj_arg w) {
    (void)w;
    lean_object *out = lean_alloc_sarray(1, out_len, out_len);
    int64_t status = 0;
    int32_t rc = base_execute((void *)handle, fn_idx,
                              lean_sarray_cptr(data), lean_sarray_size(data),
                              lean_sarray_cptr(out), out_len, &status);
    if (rc != 0) {
        lean_dec_ref(out);
        return base_io_error("base_execute failed");
    }
    lean_object *pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, out);
    lean_ctor_set(pair, 1, lean_box_uint64((uint64_t)status));
    return lean_io_result_mk_ok(pair);
}

/* A copy, because the runtime's memory is borrowed only until the next call
 * and a `ByteArray` is not. A range reaching past the end is an error rather
 * than a short answer, so a truncated read cannot be mistaken for a result. */
LEAN_EXPORT lean_obj_res lean_base_read_memory(size_t handle, size_t offset, size_t len,
                                               lean_obj_arg w) {
    (void)w;
    size_t have = 0;
    const uint8_t *memory = base_memory((const void *)handle, &have);
    if (memory == NULL || offset > have || len > have - offset) {
        return base_io_error("that range is outside the runtime's memory");
    }
    lean_object *dst = lean_alloc_sarray(1, len, len);
    memcpy(lean_sarray_cptr(dst), memory + offset, len);
    return lean_io_result_mk_ok(dst);
}

LEAN_EXPORT lean_obj_res lean_base_memory_size(size_t handle, lean_obj_arg w) {
    (void)w;
    size_t len = 0;
    base_memory((const void *)handle, &len);
    return lean_io_result_mk_ok(lean_box_usize(len));
}

LEAN_EXPORT lean_obj_res lean_base_free(size_t handle, lean_obj_arg w) {
    (void)w;
    base_free((void *)handle);
    return lean_io_result_mk_ok(lean_box(0));
}
