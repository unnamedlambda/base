//! The C ABI, for a host that is not Rust.
//!
//! `Driver` is reachable from Rust as a library and from Python through
//! `py-base`. This is the same object behind a C calling convention, so a third
//! host — a Lean program that builds its own artifact and runs it in-process —
//! needs no Rust of its own.
//!
//! An artifact arrives as the CBOR a generator writes. Base works like a
//! shared library: [`base_driver_load`] compiles it, as `dlopen` would load
//! one; [`base_driver_function`] answers a function by name as a plain C
//! function pointer, as `dlsym` would; and the host calls that function itself, as
//! often as it likes, from any thread. Nothing of base runs between the host
//! and the program on a call.
//!
//! Every function a host calls has one C signature:
//!
//! ```c
//! int64_t function(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_len);
//! ```
//!
//! It runs the program with `in` and answers in `out`, which it reads and
//! writes directly, and returns the program's status (`0` from a program that
//! returns nothing). The arena is not an argument: the function takes its
//! address itself, from a symbol the engine links to the arena, as a C function
//! in a shared library reaches the library's data.
//!
//! That is all a host needs and nothing it could do itself: load and free a
//! driver, a function by name, and the arena. Waiting on a word of the arena,
//! waking, and the clock are the system's own calls (`ffi/sync.rs` names
//! them), which a host makes itself and meets a program's at; calling by name
//! or copying out of the arena are a line or two in each host's binding.
//!
//! # What stays fixed
//!
//! Every function a host calls has the one signature above; what else a call
//! needs travels in its buffers, laid out by the program. The arena never moves or
//! grows, since the program's code is linked to its address. What an artifact
//! runs with is decided when it is built, so a load takes nothing but the
//! artifact. Anything more a host reaches is a function of the program, or
//! the system's own call, rather than another function here. A trap in a program ends
//! the process: nothing of base runs on a call to catch one.
//!
//! # Results
//!
//! A program answers through the out buffer its caller passes, and whatever it
//! leaves in its arena stays there, in [`base_driver_arena`]. Base gives neither
//! a format: the generator that built the program is what knows what the bytes
//! mean.
//!
//! # What a caller provides
//!
//! What every program assumes of its caller, as compiled C code does:
//!
//! - Nothing about the alignment of `in` and `out`: `in` and `out` may start at any address,
//!   so a program reaches them with accesses that do not assume it.
//! - The calling thread's stack, which the program runs on. Frames are
//!   probed, so overflowing it reaches the guard page and ends the process,
//!   as any trap does, rather than writing past it.
//! - The default floating-point environment: rounding to nearest, subnormals
//!   kept, no exceptions trapped, which is what the programs' float semantics
//!   is.
//!
//! What else a call needs of `in`, `out` and the arena is the function's
//! own contract, as `restrict` on a C function's parameters is that
//! function's, not the loader's: base does nothing on a call that relies on
//! it. A function Lean builds states that the three do not overlap, since
//! its theorems treat them as three regions; a host that shares data without
//! copying puts it in the arena, or passes pointers to it in `in`.
//!
//! # Threads
//!
//! Functions may be called from any number of threads at once, as any C
//! functions may: each call runs on its caller's thread against the one
//! arena, and how calls share it is the program's to decide. The arena is
//! shared with running programs the way two threads of one C program share
//! memory: bytes a program may touch while it runs are reached with atomic
//! operations on both sides, and bytes no running program touches are the
//! host's to use as it likes.
//!
//! [`base_driver_free`] needs the driver to itself: no function of it may
//! still be running, and none may be called after it, as with a library after
//! `dlclose`. Keeping that is the host's part, and every host already has the
//! means: a `std::shared_ptr` in C++, an `Arc` or a borrow in Rust, an object's
//! reference count in Python.
//!
//! A program may start threads of its own (`cl_thread_start`), which run its
//! code against its arena. One still running when the driver is freed runs
//! on unmapped code, so a function that starts a thread finishes it before
//! it returns, or leaves to a function the host calls later to finish it.
//! That is the program's part, and nothing here checks it. The same holds of
//! `in` and `out`: they are the program's for the call, and a thread or copy
//! that still reads or writes them after the function returns is the
//! program's error.
//!
//! # Errors
//!
//! Only a load can fail with something to say: it answers null and writes why
//! into the buffer its caller passes. A lookup of a name the artifact has no
//! function by answers null, which says all there is. Base keeps no error state.
//!
//! A panic while loading is one of those failures. A panic cannot unwind out
//! of an `extern "C"` function — Rust aborts the process instead — so the load
//! runs under [`guard`], and the host sees null and the panic's message rather
//! than losing its own process. A trap in generated code is a signal, not a
//! panic, and still ends the process.

use crate::{Driver, FunctionPtr};
use base_types::Artifact;

/// Run `body`, turning a panic into an error carrying its message.
///
/// `AssertUnwindSafe` is sound here because a panic ends the load, and
/// nothing it was midway through is reachable afterwards.
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

/// The bytes at `ptr`, or `None` if that is not a region this may read.
///
/// A null pointer with a non-zero length is a caller bug, not an empty slice:
/// `from_raw_parts` on it is undefined behaviour rather than a failed read. A
/// zero length answers the empty slice without dereferencing anything, which is
/// what lets a caller pass `(null, 0)` for an absent buffer.
unsafe fn slice_in<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    match (ptr.is_null(), len) {
        (_, 0) => Some(&[]),
        (true, _) => None,
        (false, _) => Some(std::slice::from_raw_parts(ptr, len)),
    }
}

/// Compile an artifact and allocate its arena. The result is owned by the caller
/// and released with [`base_driver_free`].
///
/// `artifact` is an encoded [`Artifact`] — the `.cbor` file a generator writes.
///
/// Null means the load failed, and then why is written to `err` as a
/// NUL-terminated string, cut to fit its `err_cap` bytes. `err` may be null
/// when the caller does not want to know.
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
        let artifact = Artifact::from_bytes(bytes)?;
        Driver::load(artifact).map_err(|e| e.to_string())
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

/// The function of this `Driver` named `name`, as the C function a host calls
/// (see the module documentation for its signature); null when there is no
/// driver or no function by that name. `name` is UTF-8 and not
/// NUL-terminated.
///
/// The function stays callable until [`base_driver_free`], from any thread.
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
) -> Option<FunctionPtr> {
    let name = std::str::from_utf8(slice_in(name, name_len)?).ok()?;
    Some(driver.as_ref()?.function(name).ok()?.as_ptr())
}

/// This `Driver`'s arena, and how many bytes it holds: the arena itself, not a
/// copy, at an address that stays put until [`base_driver_free`]. Null with
/// `*len` zero means there is no driver, or an empty arena.
///
/// Programs run in this arena as the host uses it, so the host shares it with
/// them as one thread of a C program shares memory with another. While any
/// call on the driver runs, bytes a program may touch are reached only with
/// atomic operations, as the program reaches them (`std::atomic_ref` in C++),
/// and bytes no running program touches are the host's to read and write as
/// it likes.
///
/// The arena is its own pages, so an offset a program keeps aligned is an
/// aligned address, to any alignment up to the page.
///
/// # Safety
///
/// `driver` must be a live pointer from [`base_driver_load`], or null. `len`
/// must be writable, or null when the caller does not want the length.
#[no_mangle]
pub unsafe extern "C" fn base_driver_arena(driver: *const Driver, len: *mut usize) -> *mut u8 {
    let (ptr, size) = match driver.as_ref() {
        Some(driver) => (driver.arena_ptr(), driver.arena_size()),
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

/// Release a `Driver` from [`base_driver_load`]. Null is accepted and does nothing.
///
/// # Safety
///
/// `driver` must come from [`base_driver_load`] and must not be used afterwards:
/// no other call on it may still be running, none may start, and no thread a
/// program of it started may still be running. Freeing the same driver twice
/// is undefined behaviour.
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
