//! The C library's memory functions, as the process links them.

use std::ffi::{c_int, c_void};

extern "C" {
    fn memcpy(d: *mut c_void, s: *const c_void, n: usize) -> *mut c_void;
    fn memmove(d: *mut c_void, s: *const c_void, n: usize) -> *mut c_void;
    fn memset(d: *mut c_void, c: c_int, n: usize) -> *mut c_void;
    fn calloc(n: usize, size: usize) -> *mut c_void;
    fn free(p: *mut c_void);
}

crate::builtin::symbols!(
    memcpy,
    memmove,
    memset,
    calloc,
    free,
);
