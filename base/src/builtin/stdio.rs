//! Standard input, output and error.

use std::io::{Read, Write as IoWrite};

/// Bytes of standard input read into `dst`, at most `cap`, as one read takes
/// them: how many, `0` at the end, `-1` on failure or a negative `cap`.
///
/// # Safety
///
/// `dst` has room for `cap` bytes.
pub(crate) unsafe extern "C" fn cl_stdin_read(dst: *mut u8, cap: i64) -> i64 {
    let Ok(cap) = usize::try_from(cap) else { return -1 };
    let buf = if cap == 0 { &mut [][..] } else { std::slice::from_raw_parts_mut(dst, cap) };
    loop {
        match std::io::stdin().lock().read(buf) {
            Ok(n) => return n as i64,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return -1,
        }
    }
}

/// `size` bytes at `src` written to `out` and flushed: `size`, or `-1`.
unsafe fn write_all(mut out: impl IoWrite, src: *const u8, size: i64) -> i64 {
    let Ok(len) = usize::try_from(size) else { return -1 };
    let data = if len == 0 { &[][..] } else { std::slice::from_raw_parts(src, len) };
    match out.write_all(data).and_then(|()| out.flush()) {
        Ok(()) => size,
        Err(_) => -1,
    }
}

/// `size` bytes at `src` written to standard output and flushed: `size`, or `-1`.
///
/// # Safety
///
/// `src` holds `size` readable bytes.
pub(crate) unsafe extern "C" fn cl_stdout_write(src: *const u8, size: i64) -> i64 {
    write_all(std::io::stdout().lock(), src, size)
}

/// `size` bytes at `src` written to standard error: `size`, or `-1`.
///
/// # Safety
///
/// `src` holds `size` readable bytes.
pub(crate) unsafe extern "C" fn cl_stderr_write(src: *const u8, size: i64) -> i64 {
    write_all(std::io::stderr().lock(), src, size)
}
