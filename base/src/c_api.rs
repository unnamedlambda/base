//! The C API, `base/include/base.h`, which states the contract a host keeps.
//!
//! A load compiles an artifact as `dlopen` loads a library, `base_driver_function`
//! answers a function by name as `dlsym` does, and the host calls it directly.
//! Only a load can fail with something to say; a panic during it is reported
//! as its error rather than unwinding into C.

use crate::{Driver, BaseFunction};

/// `body`, with a panic as an error carrying its message. Nothing a panicking
/// load was building is reachable after it, so unwind safety holds.
fn guard<T>(body: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).unwrap_or_else(|payload| {
        let what = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a panic with no message".to_string());
        Err(format!("panic: {what}"))
    })
}

/// The `len` bytes at `ptr`; empty for a zero length, `None` for null with one.
unsafe fn slice_in<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    match (ptr.is_null(), len) {
        (_, 0) => Some(&[]),
        (true, _) => None,
        (false, _) => Some(std::slice::from_raw_parts(ptr, len)),
    }
}

/// A driver for the encoded artifact, released with [`base_driver_free`]; null
/// when the load fails, with why written to `err`, NUL-terminated and cut to fit.
///
/// # Safety
///
/// `artifact` must point to `len` readable bytes, or be null with `len` zero.
/// `err` must point to `err_cap` writable bytes, or be null.
#[no_mangle]
pub unsafe extern "C" fn base_driver_load(
    artifact: *const u8,
    len: usize,
    err: *mut u8,
    err_cap: usize,
) -> *mut Driver {
    let loaded = guard(|| {
        let bytes = slice_in(artifact, len).ok_or("artifact is null with a non-zero length")?;
        Driver::load(bytes).map_err(|e| e.to_string())
    });
    match loaded {
        Ok(driver) => Box::into_raw(Box::new(driver)),
        Err(message) => {
            if !err.is_null() && err_cap > 0 {
                let n = message.len().min(err_cap - 1);
                std::ptr::copy_nonoverlapping(message.as_ptr(), err, n);
                *err.add(n) = 0;
            }
            std::ptr::null_mut()
        }
    }
}

/// The function named `name` (UTF-8, not NUL-terminated), callable from any
/// thread until [`base_driver_free`]; null when there is none.
///
/// # Safety
///
/// `driver` must be a live pointer from [`base_driver_load`]. `name` must point
/// to `name_len` bytes, or be null with length zero.
#[no_mangle]
pub unsafe extern "C" fn base_driver_function(
    driver: *const Driver,
    name: *const u8,
    name_len: usize,
) -> Option<BaseFunction> {
    let name = std::str::from_utf8(slice_in(name, name_len)?).ok()?;
    driver.as_ref()?.exports.get(name).copied()
}

/// The arena and its size, at an address fixed until [`base_driver_free`]; null
/// with `*len` zero for no driver or an empty arena. While calls run, bytes a
/// program may touch are reached only atomically.
///
/// # Safety
///
/// `driver` must be a live pointer from [`base_driver_load`], or null. `len`
/// must be writable, or null when the caller does not want the length.
#[no_mangle]
pub unsafe extern "C" fn base_driver_arena(driver: *const Driver, len: *mut usize) -> *mut u8 {
    let (ptr, size) = match driver.as_ref() {
        Some(driver) => (driver.arena().as_ptr() as *mut u8, driver.arena().len()),
        None => (std::ptr::null_mut(), 0),
    };
    if let Some(len) = len.as_mut() {
        *len = size;
    }
    if size == 0 {
        std::ptr::null_mut()
    } else {
        ptr
    }
}

/// Releases a driver; null does nothing.
///
/// # Safety
///
/// `driver` came from [`base_driver_load`] and is not used again: no call on
/// it, and no thread a program of it started, is still running.
#[no_mangle]
pub unsafe extern "C" fn base_driver_free(driver: *mut Driver) {
    if !driver.is_null() {
        drop(Box::from_raw(driver));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_is_an_error_and_not_an_abort() {
        assert_eq!(guard(|| -> Result<i32, String> { panic!("boom") }), Err("panic: boom".into()));
    }
}
