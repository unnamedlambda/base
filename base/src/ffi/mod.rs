pub(crate) mod cpu;
pub(crate) mod file;
pub(crate) mod libc;
pub(crate) mod native;
pub(crate) mod os;
pub(crate) mod serial;
pub(crate) mod stdio;
pub(crate) mod thread;
pub(crate) mod usb;
pub(crate) mod wgpu;
pub(crate) mod window;

/// The engine's own functions, the `Ffi` entries a program calls as the
/// built-in library `base`, at the signatures `Ffi.sig` states.
pub(crate) fn linked(symbol: &str) -> Option<usize> {
    Some(match symbol {
        "cl_native_load" => native::cl_native_load as usize,
        "cl_native_free" => native::cl_native_free as usize,
        "cl_native_arch" => native::cl_native_arch as usize,
        "cl_cpu_has" => native::cl_cpu_has as usize,
        "cl_file_read" => file::cl_file_read as usize,
        "cl_file_read_to_ptr" => file::cl_file_read_to_ptr as usize,
        "cl_file_write" => file::cl_file_write as usize,
        "cl_file_write_from_ptr" => file::cl_file_write_from_ptr as usize,
        "cl_file_create_dir_all" => file::cl_file_create_dir_all as usize,
        "cl_stdin_readline" => stdio::cl_stdin_readline as usize,
        "cl_stdout_write" => stdio::cl_stdout_write as usize,
        "cl_thread_start" => thread::cl_thread_start as usize,
        "cl_thread_finish" => thread::cl_thread_finish as usize,
        "cl_mem_lock" => os::cl_mem_lock as usize,
        "cl_mem_unlock" => os::cl_mem_unlock as usize,
        "cl_mem_advise_huge" => os::cl_mem_advise_huge as usize,
        "cl_thread_priority" => os::cl_thread_priority as usize,
        _ => return None,
    })
}

/// Longest filesystem path or kernel name a caller reads.
pub(super) const CSTR_NAME_MAX: usize = 4096;

pub(super) unsafe fn read_cstr(ptr: *mut u8, off: usize) -> String {
    if ptr.is_null() {
        return String::new();
    }
    read_cstr_ptr(ptr.add(off))
}

/// The bytes up to the first NUL, for a name-sized string.
pub(super) unsafe fn read_cstr_ptr(start: *const u8) -> String {
    read_cstr_bounded(start, CSTR_NAME_MAX)
}

/// The bytes up to the first NUL, or empty if there is no pointer, no NUL
/// within `max`, or the bytes are not UTF-8.
///
/// Both guards matter: without them this walks arbitrary memory until it
/// happens on a zero byte. Empty is the safe answer rather than a truncation
/// or a lossy decoding, because every caller passes the result to something —
/// `File::open`, a socket address parse — that rejects it; the model's
/// `readPath` reads a path the same way.
pub(super) unsafe fn read_cstr_bounded(start: *const u8, max: usize) -> String {
    if start.is_null() {
        return String::new();
    }
    let mut len = 0;
    while len < max && *start.add(len) != 0 {
        len += 1;
    }
    if len == max {
        return String::new();
    }
    String::from_utf8(std::slice::from_raw_parts(start, len).to_vec()).unwrap_or_default()
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
        let unterminated = vec![b'a'; CSTR_NAME_MAX + 16];
        assert_eq!(unsafe { read_cstr_ptr(unterminated.as_ptr()) }, "");
        assert_eq!(unsafe { read_cstr(std::ptr::null_mut(), 8) }, "");
    }
}
