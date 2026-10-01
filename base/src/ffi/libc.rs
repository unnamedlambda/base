//! The C library's memory functions, as the process links them: Rust's
//! standard library links the platform C runtime, so these are always there.

use std::ffi::{c_char, c_int, c_void};

extern "C" {
    fn memcpy(d: *mut c_void, s: *const c_void, n: usize) -> *mut c_void;
    fn memmove(d: *mut c_void, s: *const c_void, n: usize) -> *mut c_void;
    fn memset(d: *mut c_void, c: c_int, n: usize) -> *mut c_void;
    fn strlen(s: *const c_char) -> usize;
    fn calloc(n: usize, size: usize) -> *mut c_void;
    fn free(p: *mut c_void);
}

pub(crate) fn linked(symbol: &str) -> Option<usize> {
    Some(match symbol {
        "memcpy" => memcpy as usize,
        "memmove" => memmove as usize,
        "memset" => memset as usize,
        "strlen" => strlen as usize,
        "calloc" => calloc as usize,
        "free" => free as usize,
        _ => return None,
    })
}
