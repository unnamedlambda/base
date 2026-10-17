//! The C library's memory functions, as the process links them.

use std::ffi::{c_char, c_int, c_void};

extern "C" {
    fn memcpy(d: *mut c_void, s: *const c_void, n: usize) -> *mut c_void;
    fn memmove(d: *mut c_void, s: *const c_void, n: usize) -> *mut c_void;
    fn memset(d: *mut c_void, c: c_int, n: usize) -> *mut c_void;
    fn strlen(s: *const c_char) -> usize;
    fn calloc(n: usize, size: usize) -> *mut c_void;
    fn free(p: *mut c_void);
}

crate::ffi::symbols!(
    memcpy,
    memmove,
    memset,
    strlen,
    calloc,
    free,
);
