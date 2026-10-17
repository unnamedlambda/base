pub(crate) mod cpu;
pub(crate) mod file;
pub(crate) mod libc;
pub(crate) mod native;
pub(crate) mod os;
pub(crate) mod serial;
pub(crate) mod stdio;
pub(crate) mod sync;
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
        "cl_file_read_to_ptr" => file::cl_file_read_to_ptr as usize,
        "cl_file_write_from_ptr" => file::cl_file_write_from_ptr as usize,
        "cl_file_create_dir_all" => file::cl_file_create_dir_all as usize,
        "cl_file_size" => file::cl_file_size as usize,
        "cl_file_set_len" => file::cl_file_set_len as usize,
        "cl_stdin_read_line" => stdio::cl_stdin_read_line as usize,
        "cl_stdout_write" => stdio::cl_stdout_write as usize,
        "cl_thread_start" => thread::cl_thread_start as usize,
        "cl_thread_finish" => thread::cl_thread_finish as usize,
        "cl_mem_lock" => os::cl_mem_lock as usize,
        "cl_mem_unlock" => os::cl_mem_unlock as usize,
        "cl_mem_advise_huge" => os::cl_mem_advise_huge as usize,
        "cl_thread_priority" => os::cl_thread_priority as usize,
        "cl_mem_wait" => sync::cl_mem_wait as usize,
        "cl_mem_wake" => sync::cl_mem_wake as usize,
        "cl_clock_now" => sync::cl_clock_now as usize,
        _ => return None,
    })
}

/// The string in the `cap`-byte field at `start`: its bytes up to the first
/// NUL, or all `cap` of them if there is none, as a fixed-size `char` field
/// holds one (`sun_path`, `strnlen`). `None` for a negative `cap` or bytes that
/// are not UTF-8, which every caller answers as its refusal.
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
