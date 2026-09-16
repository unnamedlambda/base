//! The C ABI, for a host that is not Rust.
//!
//! `Base` is reachable from Rust as a library and from Python through
//! `py-base`. This is the same object behind a C calling convention, so a third
//! host — a Lean program that builds its own artifact and runs it in-process —
//! needs no Rust of its own.
//!
//! An artifact arrives as the JSON a generator already prints, and an entry
//! point as its index. Nothing here decides anything about a program: it is
//! `Base::new` and `Base::execute_into` with pointers instead of types.
//!
//! # Results
//!
//! A program answers through the out buffer its caller passes, and whatever it
//! leaves in its own memory stays readable through [`base_memory`]. Base
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
use base_types::Artifact;

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

/// Compile an artifact and take its memory. The result is owned by the caller
/// and released with [`base_free`]; null means the call failed.
///
/// `artifact_json` is a serialized [`Artifact`] — the JSON a generator writes.
///
/// # Safety
///
/// `artifact_json` must point to `len` readable bytes, or be null with `len`
/// zero.
#[no_mangle]
pub unsafe extern "C" fn base_new(artifact_json: *const u8, len: usize) -> *mut Base {
    guard(std::ptr::null_mut(), || {
        clear_error();
        let Some(bytes) = slice_in(artifact_json, len) else {
            set_error("artifact_json is null with a non-zero length");
            return std::ptr::null_mut();
        };
        let artifact: Artifact = match serde_json::from_slice(bytes) {
            Ok(a) => a,
            Err(e) => {
                set_error(format!("artifact is not an Artifact: {e}"));
                return std::ptr::null_mut();
            }
        };
        match Base::new(artifact) {
            Ok(base) => Box::into_raw(Box::new(base)),
            Err(e) => {
                set_error(String::from(e));
                std::ptr::null_mut()
            }
        }
    })
}

/// Call one entry point of this `Base`. `0` on success, `-1` on failure.
///
/// `fn_idx` is the entry point's function index. `data` is the input the
/// program is handed and `out` the buffer it answers in; either may be
/// `(null, 0)` when a program uses neither.
///
/// `status` takes the value the program returned, or `0` from one that returns
/// nothing, and may be null when the caller does not want it. Base passes it
/// through without reading it: what it means is between the program and the
/// host. Its own failures are the `-1` and [`base_last_error`].
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
    fn_idx: u32,
    data: *const u8,
    data_len: usize,
    out: *mut u8,
    out_len: usize,
    status: *mut i64,
) -> i32 {
    guard(-1, || {
        clear_error();
        let Some(base) = handle.as_mut() else {
            set_error("handle is null");
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
        match base.execute_into(fn_idx, data, out) {
            Ok(answered) => {
                if let Some(status) = status.as_mut() {
                    *status = answered;
                }
                0
            }
            Err(e) => {
                set_error(String::from(e));
                -1
            }
        }
    })
}

/// This `Base`'s memory, and how many bytes of it, for as long as the caller
/// does not call [`base_execute`] or [`base_free`] on it.
///
/// A program's results are in here, at the addresses its generator says it
/// writes; base gives the bytes no format. Null with `*len` zero means there is
/// no handle.
///
/// The pointer is the memory itself, not a copy: reading it after the next
/// execute reads whatever that call left, and reading it after [`base_free`] is
/// undefined. A host that needs the bytes to outlive either copies them.
///
/// # Safety
///
/// `handle` must be a live pointer from [`base_new`], or null. `len` must be
/// writable, or null when the caller does not want the length.
#[no_mangle]
pub unsafe extern "C" fn base_memory(handle: *const Base, len: *mut usize) -> *const u8 {
    let memory = match handle.as_ref() {
        Some(base) => base.memory_bytes(),
        None => &[],
    };
    if let Some(len) = len.as_mut() {
        *len = memory.len();
    }
    if memory.is_empty() {
        std::ptr::null()
    } else {
        memory.as_ptr()
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

    /// An artifact with no functions: enough to exercise the ABI without a JIT.
    const EMPTY_ARTIFACT: &str = r#"{
        "functions": [],
        "memory_size": 64,
        "data": [{"offset": 0, "bytes": [1, 2, 3, 4]}]
    }"#;

    fn last_error() -> String {
        let len = unsafe { base_last_error(std::ptr::null_mut(), 0) };
        let mut buf = vec![0u8; len];
        unsafe { base_last_error(buf.as_mut_ptr(), buf.len()) };
        String::from_utf8(buf).unwrap()
    }

    /// The field `data` replaced. `data` has a default, so without the wire
    /// format refusing what it does not know, a writer left behind by that
    /// change would hand over an artifact that builds and starts from zeros —
    /// a program reading the wrong memory, not a failure anyone would trace
    /// back to the artifact.
    #[test]
    fn an_artifact_naming_a_field_that_is_gone_is_refused() {
        const STALE: &str = r#"{
            "functions": [],
            "memory_size": 64,
            "initial_memory": [1, 2, 3, 4]
        }"#;
        let handle = unsafe { base_new(STALE.as_ptr(), STALE.len()) };
        assert!(handle.is_null(), "a stale artifact should not build");
        assert!(
            last_error().contains("initial_memory"),
            "the message should name the field: {}",
            last_error()
        );
    }

    #[test]
    fn an_artifact_round_trips_from_json_and_frees() {
        let handle = unsafe { base_new(EMPTY_ARTIFACT.as_ptr(), EMPTY_ARTIFACT.len()) };
        assert!(!handle.is_null(), "{}", last_error());
        let mut len = 0usize;
        assert!(!unsafe { base_memory(handle, &mut len) }.is_null());
        assert_eq!(len, 64);
        unsafe { base_free(handle) };
    }

    /// The data segments are what the program starts from, so reading them back
    /// is how a host confirms it got the artifact it sent.
    #[test]
    fn memory_answers_the_bytes_the_artifact_starts_with() {
        let handle = unsafe { base_new(EMPTY_ARTIFACT.as_ptr(), EMPTY_ARTIFACT.len()) };
        assert!(!handle.is_null(), "{}", last_error());
        let mut len = 0usize;
        let memory = unsafe { base_memory(handle, &mut len) };
        assert_eq!(len, 64);
        assert_eq!(unsafe { std::slice::from_raw_parts(memory, 4) }, [1, 2, 3, 4]);
        assert_eq!(unsafe { std::slice::from_raw_parts(memory, len) }[4..], [0u8; 60]);
        unsafe { base_free(handle) };
    }

    /// Every entry point answers its failure value on a null handle rather than
    /// dereferencing it, and says so.
    #[test]
    fn a_null_handle_is_refused_everywhere() {
        assert_eq!(
            unsafe {
                base_execute(
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null_mut(),
                )
            },
            -1
        );
        assert!(last_error().contains("null"));
        let mut len = 7usize;
        assert!(unsafe { base_memory(std::ptr::null(), &mut len) }.is_null());
        assert_eq!(len, 0);
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
        let bad = b"{\"functions\":";
        assert!(unsafe { base_new(bad.as_ptr(), bad.len()) }.is_null());
        assert!(last_error().contains("not an Artifact"), "{}", last_error());
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
