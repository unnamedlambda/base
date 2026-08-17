pub(crate) mod cuda;
pub(crate) mod file;
pub(crate) mod ht;
pub(crate) mod lmdb;
pub(crate) mod net;
pub(crate) mod stdio;
pub(crate) mod thread;
pub(crate) mod wgpu;
pub(crate) mod window;

pub(super) unsafe fn read_ctx_ref<T>(ctx_ptr: *const T) -> Option<&'static T> {
    ctx_ptr.as_ref()
}

pub(super) unsafe fn read_ctx_mut<T>(ctx_ptr: *mut T) -> Option<&'static mut T> {
    ctx_ptr.as_mut()
}

pub(super) unsafe fn write_ctx_slot<T>(slot_ptr: *mut *mut T, raw: *mut T) -> bool {
    if slot_ptr.is_null() {
        return false;
    }
    std::ptr::write_unaligned(slot_ptr, raw);
    true
}

pub(super) unsafe fn clear_ctx_slot<T>(slot_ptr: *mut *mut T) -> *mut T {
    if slot_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let raw = std::ptr::read_unaligned(slot_ptr as *const *mut T);
    if !raw.is_null() {
        std::ptr::write_unaligned(slot_ptr, std::ptr::null_mut());
    }
    raw
}

/// Longest string any caller has a use for — every one of them reads a
/// filesystem path or a socket address.
const CSTR_MAX: usize = 4096;

pub(super) unsafe fn read_cstr(ptr: *mut u8, off: usize) -> String {
    if ptr.is_null() {
        return String::new();
    }
    read_cstr_ptr(ptr.add(off))
}

/// The bytes up to the first NUL, or empty if there is no pointer or no NUL
/// within `CSTR_MAX`.
///
/// Both guards matter: without them this walks arbitrary memory until it
/// happens on a zero byte. Empty is the safe answer rather than a truncation,
/// because every caller passes the result to something — `File::open`, a
/// socket address parse — that rejects it.
pub(super) unsafe fn read_cstr_ptr(start: *const u8) -> String {
    if start.is_null() {
        return String::new();
    }
    let mut len = 0;
    while len < CSTR_MAX && *start.add(len) != 0 {
        len += 1;
    }
    if len == CSTR_MAX {
        return String::new();
    }
    String::from_utf8_lossy(std::slice::from_raw_parts(start, len)).into_owned()
}

// Stateless libm wrappers — exposed as FFI for CLIF code that needs trig/pow.

pub(crate) unsafe extern "C" fn cl_sinf(x: f32) -> f32 {
    x.sin()
}

pub(crate) unsafe extern "C" fn cl_cosf(x: f32) -> f32 {
    x.cos()
}

pub(crate) unsafe extern "C" fn cl_powf(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_cstr_ptr_reads_up_to_the_nul() {
        let s = b"hello\0trailing";
        assert_eq!(unsafe { read_cstr_ptr(s.as_ptr()) }, "hello");
    }

    /// No pointer and no NUL both answer empty rather than walking memory.
    /// Empty is what every caller rejects, so a truncation cannot be mistaken
    /// for a path.
    #[test]
    fn read_cstr_ptr_refuses_null_and_unterminated() {
        assert_eq!(unsafe { read_cstr_ptr(std::ptr::null()) }, "");
        let unterminated = vec![b'a'; CSTR_MAX + 16];
        assert_eq!(unsafe { read_cstr_ptr(unterminated.as_ptr()) }, "");
        assert_eq!(unsafe { read_cstr(std::ptr::null_mut(), 8) }, "");
    }
}
