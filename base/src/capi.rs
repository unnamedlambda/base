//! The C ABI, for a host that is not Rust.
//!
//! `Base` is reachable from Rust as a library and from Python through
//! `py-base`. This is the same object behind a C calling convention, so a third
//! host — a Lean program that builds its own artifact and runs it in-process —
//! needs no Rust of its own.
//!
//! A setup arrives as the JSON a generator already prints. Nothing here decides
//! anything about a program: it is `Base::new` and `Base::execute_into` with
//! pointers instead of types.
//!
//! # Results
//!
//! A program answers through the out buffer its caller passes, and whatever it
//! leaves in its own memory stays readable through [`base_read_memory`]. Base
//! gives neither a format: the generator that built the program is what knows
//! what the bytes mean.
//!
//! # Errors
//!
//! A call that fails answers null or `-1` and leaves a message on the calling
//! thread, which [`base_last_error`] copies out. The message is per-thread and
//! not allocated for the caller, so there is nothing to free.
//!
//! A panic inside a call is one of those failures. A panic cannot unwind out
//! of an `extern "C"` function — Rust aborts the process instead — so every
//! entry point runs its body under [`guard`], and the host sees `-1` and the
//! panic's message rather than losing its own process. A trap in generated
//! code is a signal, not a panic, and still ends the process.

use std::cell::RefCell;

use crate::{Base, Error};
use base_types::{Algorithm, Setup};

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(message: impl Into<String>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = message.into());
}

fn clear_error() {
    LAST_ERROR.with(|e| e.borrow_mut().clear());
}

impl From<Error> for String {
    fn from(e: Error) -> String {
        match e {
            Error::Clif(m) => format!("clif: {m}"),
            Error::Execution(m) => format!("execution: {m}"),
        }
    }
}

/// Run `body`, turning a panic into `failed` and a message for
/// [`base_last_error`].
///
/// `AssertUnwindSafe` is sound here because a panic ends the call: nothing
/// observes the state `body` was midway through except through the handle,
/// and a host that keeps using a handle after a failed call gets whatever
/// that handle's program left in its memory, as it would after any failure.
fn guard<T>(failed: T, body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(v) => v,
        Err(payload) => {
            let what = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".to_string());
            set_error(format!("panic: {what}"));
            failed
        }
    }
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

unsafe fn slice_out<'a>(ptr: *mut u8, len: usize) -> Option<&'a mut [u8]> {
    match (ptr.is_null(), len) {
        (_, 0) => Some(&mut []),
        (true, _) => None,
        (false, _) => Some(std::slice::from_raw_parts_mut(ptr, len)),
    }
}

/// Compile a setup and take its memory. The result is owned by the caller and
/// released with [`base_free`]; null means the call failed.
///
/// `setup_json` is a serialized [`Setup`] — the `"setup"` field of the JSON a
/// generator writes.
///
/// # Safety
///
/// `setup_json` must point to `len` readable bytes, or be null with `len` zero.
#[no_mangle]
pub unsafe extern "C" fn base_new(setup_json: *const u8, len: usize) -> *mut Base {
    guard(std::ptr::null_mut(), || {
        clear_error();
        let Some(bytes) = slice_in(setup_json, len) else {
            set_error("setup_json is null with a non-zero length");
            return std::ptr::null_mut();
        };
        let setup: Setup = match serde_json::from_slice(bytes) {
            Ok(s) => s,
            Err(e) => {
                set_error(format!("setup is not a Setup: {e}"));
                return std::ptr::null_mut();
            }
        };
        match Base::new(setup) {
            Ok(base) => Box::into_raw(Box::new(base)),
            Err(e) => {
                set_error(String::from(e));
                std::ptr::null_mut()
            }
        }
    })
}

/// Make this `Base` callable from the calling thread. `0` on success, `-1` if
/// there is no handle.
///
/// The compiled functions are also held in a thread-local, because the FFI
/// entry points a program calls — `cl_thread_init` and what it spawns — reach
/// them without a `Base` to hand. `base_new` installs it on the thread that
/// called it, so a host whose scheduler moves work between threads must call
/// this on any other thread it executes from. Calling it more than once, or on
/// the creating thread, does nothing.
///
/// # Safety
///
/// `handle` must be a live pointer from [`base_new`], or null.
#[no_mangle]
pub unsafe extern "C" fn base_bind_thread(handle: *mut Base) -> i32 {
    guard(-1, || {
        clear_error();
        let Some(base) = handle.as_ref() else {
            set_error("handle is null");
            return -1;
        };
        base.bind_current_thread();
        0
    })
}

/// Run one algorithm against this `Base`. `0` on success, `-1` on failure.
///
/// `algorithm_json` is a serialized [`Algorithm`] — the `"main"` field, or one
/// value of `"extras"`. `data` is the input the program reads through its
/// `data_ptr`/`data_len` offsets and `out` the buffer it writes through
/// `out_ptr`/`out_len`; either may be `(null, 0)` when a program uses neither.
///
/// Both buffers are borrowed only for the duration of the call: the program
/// sees the caller's memory directly, and nothing retains the pointers
/// afterwards.
///
/// # Safety
///
/// `handle` must be a live pointer from [`base_new`]. Each buffer must point to
/// as many bytes as its length claims, or be null with length zero. `out` must
/// not alias `data`.
#[no_mangle]
pub unsafe extern "C" fn base_execute(
    handle: *mut Base,
    algorithm_json: *const u8,
    alg_len: usize,
    data: *const u8,
    data_len: usize,
    out: *mut u8,
    out_len: usize,
) -> i32 {
    guard(-1, || {
        clear_error();
        let Some(base) = handle.as_mut() else {
            set_error("handle is null");
            return -1;
        };
        let Some(alg_bytes) = slice_in(algorithm_json, alg_len) else {
            set_error("algorithm_json is null with a non-zero length");
            return -1;
        };
        let Some(data) = slice_in(data, data_len) else {
            set_error("data is null with a non-zero length");
            return -1;
        };
        let Some(out) = slice_out(out, out_len) else {
            set_error("out is null with a non-zero length");
            return -1;
        };
        let algorithm: Algorithm = match serde_json::from_slice(alg_bytes) {
            Ok(a) => a,
            Err(e) => {
                set_error(format!("algorithm is not an Algorithm: {e}"));
                return -1;
            }
        };
        match base.execute_into(&algorithm, data, out) {
            Ok(()) => 0,
            Err(e) => {
                set_error(String::from(e));
                -1
            }
        }
    })
}

/// Copy `len` bytes of this `Base`'s shared memory from `offset` into `dst`,
/// answering how many were copied.
///
/// This is how a host reads what a program left in its own memory, at an
/// address the program's generator says it wrote. A range reaching
/// past the end of memory copies nothing and answers `0` rather than a
/// truncation, so a short read cannot be mistaken for a short result.
///
/// # Safety
///
/// `handle` must be a live pointer from [`base_new`], and `dst` must point to
/// `len` writable bytes, or be null with `len` zero.
#[no_mangle]
pub unsafe extern "C" fn base_read_memory(
    handle: *const Base,
    offset: usize,
    dst: *mut u8,
    len: usize,
) -> usize {
    guard(0, || {
        clear_error();
        let Some(base) = handle.as_ref() else {
            set_error("handle is null");
            return 0;
        };
        let Some(dst) = slice_out(dst, len) else {
            set_error("dst is null with a non-zero length");
            return 0;
        };
        let Some(end) = offset.checked_add(len) else {
            set_error("offset + len overflows");
            return 0;
        };
        let memory = base.memory_bytes();
        if end > memory.len() {
            set_error(format!(
                "range {offset}..{end} is outside the {} bytes of memory",
                memory.len()
            ));
            return 0;
        }
        dst.copy_from_slice(&memory[offset..end]);
        len
    })
}

/// How many bytes of shared memory this `Base` holds, which bounds
/// [`base_read_memory`].
///
/// # Safety
///
/// `handle` must be a live pointer from [`base_new`], or null.
#[no_mangle]
pub unsafe extern "C" fn base_memory_size(handle: *const Base) -> usize {
    match handle.as_ref() {
        Some(base) => base.memory_bytes().len(),
        None => 0,
    }
}

/// Copy the calling thread's last error into `buf`, answering the message's
/// full length in bytes — which may exceed `cap`, in which case what was
/// written is a prefix.
///
/// The message belongs to the thread, so this reports the failure of the last
/// call *this* thread made and there is nothing to free.
///
/// # Safety
///
/// `buf` must point to `cap` writable bytes, or be null with `cap` zero.
#[no_mangle]
pub unsafe extern "C" fn base_last_error(buf: *mut u8, cap: usize) -> usize {
    LAST_ERROR.with(|e| {
        let message = e.borrow();
        let bytes = message.as_bytes();
        if let Some(buf) = slice_out(buf, cap) {
            let n = buf.len().min(bytes.len());
            buf[..n].copy_from_slice(&bytes[..n]);
        }
        bytes.len()
    })
}

/// Release a `Base` from [`base_new`]. Null is accepted and does nothing.
///
/// # Safety
///
/// `handle` must come from [`base_new`] and must not be used afterwards.
/// Freeing the same handle twice is undefined behaviour.
#[no_mangle]
pub unsafe extern "C" fn base_free(handle: *mut Base) {
    guard((), || {
        if !handle.is_null() {
            drop(Box::from_raw(handle));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A setup with no program: enough to exercise the ABI without a JIT.
    const EMPTY_SETUP: &str = r#"{
        "clif": {"functions": []},
        "memory_size": 64,
        "initial_memory": [1, 2, 3, 4]
    }"#;

    fn last_error() -> String {
        let len = unsafe { base_last_error(std::ptr::null_mut(), 0) };
        let mut buf = vec![0u8; len];
        unsafe { base_last_error(buf.as_mut_ptr(), buf.len()) };
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn a_setup_round_trips_from_json_and_frees() {
        let handle = unsafe { base_new(EMPTY_SETUP.as_ptr(), EMPTY_SETUP.len()) };
        assert!(!handle.is_null(), "{}", last_error());
        assert_eq!(unsafe { base_memory_size(handle) }, 64);
        assert_eq!(unsafe { base_bind_thread(handle) }, 0);
        unsafe { base_free(handle) };
    }

    /// `initial_memory` is what the program starts from, so reading it back is
    /// how a host confirms it got the setup it sent.
    #[test]
    fn read_memory_answers_the_initial_bytes() {
        let handle = unsafe { base_new(EMPTY_SETUP.as_ptr(), EMPTY_SETUP.len()) };
        assert!(!handle.is_null(), "{}", last_error());
        let mut got = [0u8; 4];
        assert_eq!(unsafe { base_read_memory(handle, 0, got.as_mut_ptr(), 4) }, 4);
        assert_eq!(got, [1, 2, 3, 4]);
        unsafe { base_free(handle) };
    }

    /// A range past the end copies nothing rather than a prefix: a host reading
    /// a result must not mistake a short read for a short answer.
    #[test]
    fn read_memory_refuses_a_range_past_the_end() {
        let handle = unsafe { base_new(EMPTY_SETUP.as_ptr(), EMPTY_SETUP.len()) };
        let mut got = [0xAAu8; 8];
        assert_eq!(unsafe { base_read_memory(handle, 60, got.as_mut_ptr(), 8) }, 0);
        assert_eq!(got, [0xAA; 8], "nothing was written");
        assert!(last_error().contains("outside"));
        // …and an overflowing offset is rejected before it is compared.
        assert_eq!(
            unsafe { base_read_memory(handle, usize::MAX, got.as_mut_ptr(), 8) },
            0
        );
        assert!(last_error().contains("overflows"));
        unsafe { base_free(handle) };
    }

    /// Every entry point answers its failure value on a null handle rather than
    /// dereferencing it, and says so.
    #[test]
    fn a_null_handle_is_refused_everywhere() {
        assert_eq!(unsafe { base_bind_thread(std::ptr::null_mut()) }, -1);
        assert!(last_error().contains("null"));
        assert_eq!(
            unsafe {
                base_execute(
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0,
                )
            },
            -1
        );
        assert!(last_error().contains("null"));
        assert_eq!(
            unsafe { base_read_memory(std::ptr::null(), 0, std::ptr::null_mut(), 0) },
            0
        );
        assert_eq!(unsafe { base_memory_size(std::ptr::null()) }, 0);
        // Freeing null is a no-op, not a double free.
        unsafe { base_free(std::ptr::null_mut()) };
    }

    /// A null pointer carrying a length is a caller bug, and is refused before
    /// it reaches `from_raw_parts`.
    #[test]
    fn a_null_pointer_with_a_length_is_refused() {
        assert!(unsafe { base_new(std::ptr::null(), 16) }.is_null());
        assert!(last_error().contains("non-zero length"));
    }

    #[test]
    fn malformed_json_is_an_error_and_not_a_panic() {
        let bad = b"{\"clif\":";
        assert!(unsafe { base_new(bad.as_ptr(), bad.len()) }.is_null());
        assert!(last_error().contains("not a Setup"), "{}", last_error());
    }

    /// The message is a prefix when it does not fit, and the answer is always
    /// the full length so a caller can size a second call.
    #[test]
    fn a_panic_is_an_error_and_not_an_abort() {
        assert_eq!(guard(-1, || -> i32 { panic!("boom") }), -1);
        assert_eq!(last_error(), "panic: boom");
    }

    #[test]
    fn last_error_reports_the_full_length() {
        set_error("abcdef");
        let mut buf = [0u8; 3];
        assert_eq!(unsafe { base_last_error(buf.as_mut_ptr(), 3) }, 6);
        assert_eq!(&buf, b"abc");
    }
}
