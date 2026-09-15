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

/* Declared rather than included: the runtime is a Rust cdylib and has no
 * header. These must match `base/src/capi.rs`. */
void *base_new(const uint8_t *artifact_json, size_t len);
int32_t base_execute(void *handle, uint32_t fn_idx, const uint8_t *data,
                     size_t data_len, uint8_t *out, size_t out_len);
size_t base_read_memory(const void *handle, size_t offset, uint8_t *dst, size_t len);
size_t base_memory_size(const void *handle);
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
 * construction, and returning it is how the caller reads what ran. */
LEAN_EXPORT lean_obj_res lean_base_execute(size_t handle, uint32_t fn_idx,
                                           b_lean_obj_arg data, size_t out_len,
                                           lean_obj_arg w) {
    (void)w;
    lean_object *out = lean_alloc_sarray(1, out_len, out_len);
    int32_t rc = base_execute((void *)handle, fn_idx,
                              lean_sarray_cptr(data), lean_sarray_size(data),
                              lean_sarray_cptr(out), out_len);
    if (rc != 0) {
        lean_dec_ref(out);
        return base_io_error("base_execute failed");
    }
    return lean_io_result_mk_ok(out);
}

/* A short read is a refused range, not a short answer -- `base_read_memory`
 * copies all of it or none of it -- so anything other than `len` is an error
 * rather than a truncated `ByteArray`. */
LEAN_EXPORT lean_obj_res lean_base_read_memory(size_t handle, size_t offset, size_t len,
                                               lean_obj_arg w) {
    (void)w;
    lean_object *dst = lean_alloc_sarray(1, len, len);
    if (base_read_memory((const void *)handle, offset, lean_sarray_cptr(dst), len) != len) {
        lean_dec_ref(dst);
        return base_io_error("base_read_memory refused the range");
    }
    return lean_io_result_mk_ok(dst);
}

LEAN_EXPORT lean_obj_res lean_base_memory_size(size_t handle, lean_obj_arg w) {
    (void)w;
    return lean_io_result_mk_ok(lean_box_usize(base_memory_size((const void *)handle)));
}

LEAN_EXPORT lean_obj_res lean_base_free(size_t handle, lean_obj_arg w) {
    (void)w;
    base_free((void *)handle);
    return lean_io_result_mk_ok(lean_box(0));
}
