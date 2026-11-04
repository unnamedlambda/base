/// `linked`, answering the address of each function listed, by its name.
macro_rules! symbols {
    ($($f:path),* $(,)?) => {
        pub(crate) fn linked(symbol: &str) -> Option<usize> {
            $(if stringify!($f).rsplit("::").next() == Some(symbol) {
                return Some($f as usize);
            })*
            None
        }
    };
}
pub(crate) use symbols;

pub(crate) mod cpu;
pub(crate) mod file;
pub(crate) mod libc;
pub(crate) mod native;
pub(crate) mod os;
pub(crate) mod serial;
pub(crate) mod stdio;
pub(crate) mod sync;
pub(crate) mod usb;
pub(crate) mod wgpu;
pub(crate) mod window;

// The engine's own functions, which a program calls as the library `base`.
crate::builtin::symbols!(
    native::cl_native_load,
    native::cl_native_free,
    native::cl_native_arch,
    native::cl_cpu_has,
    file::cl_file_read_to_ptr,
    file::cl_file_write_from_ptr,
    file::cl_file_create_dir_all,
    file::cl_file_size,
    file::cl_file_set_len,
    stdio::cl_stdin_read_line,
    stdio::cl_stdout_write,
    os::cl_mem_lock,
    os::cl_mem_unlock,
    os::cl_mem_advise_huge,
    os::cl_thread_priority,
    sync::cl_mem_wait,
    sync::cl_mem_wake,
    sync::cl_clock_now,
);

/// The string in the `cap`-byte field at `start`, up to its first NUL if any;
/// `None` for a negative `cap` or bytes that are not UTF-8.
///
/// # Safety
///
/// `start` points to `cap` readable bytes.
pub(super) unsafe fn read_field<'a>(start: *const u8, cap: i64) -> Option<&'a str> {
    let cap = usize::try_from(cap).ok()?;
    if cap == 0 {
        return Some("");
    }
    let field = std::slice::from_raw_parts(start, cap);
    let n = field.iter().position(|&b| b == 0).unwrap_or(cap);
    std::str::from_utf8(&field[..n]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_ends_at_its_first_nul_or_its_end() {
        let s = b"hello\0trailing";
        assert_eq!(unsafe { read_field(s.as_ptr(), s.len() as i64) }, Some("hello"));
        assert_eq!(unsafe { read_field(s.as_ptr(), 3) }, Some("hel"));
        assert_eq!(unsafe { read_field(std::ptr::null(), 0) }, Some(""));
    }

    #[test]
    fn a_field_that_is_not_utf8_or_has_a_negative_cap_is_none() {
        let s = b"\xff\0";
        assert_eq!(unsafe { read_field(s.as_ptr(), 2) }, None);
        assert_eq!(unsafe { read_field(s.as_ptr(), -1) }, None);
    }
}
