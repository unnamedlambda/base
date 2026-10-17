//! Standard input and output.

use std::io::{BufRead, Write as IoWrite};

/// A line of standard input into `dst`, as much as `cap` holds, as `fgets`
/// reads: the bytes read, `0` at the end, `-1` on failure.
///
/// # Safety
///
/// `dst` has room for `cap` bytes.
pub(crate) unsafe extern "C" fn cl_stdin_read_line(dst: *mut u8, cap: i64) -> i64 {
    let Some(cap) = usize::try_from(cap).ok().filter(|&c| c > 0) else { return -1 };
    let mut stdin = std::io::stdin().lock();
    let mut n = 0;
    while n < cap {
        let buf = match stdin.fill_buf() {
            Ok(b) => b,
            Err(_) => return -1,
        };
        if buf.is_empty() {
            break;
        }
        let take = buf.len().min(cap - n);
        let line = buf[..take].iter().position(|&b| b == b'\n').map(|i| i + 1);
        let take = line.unwrap_or(take);
        std::ptr::copy_nonoverlapping(buf.as_ptr(), dst.add(n), take);
        stdin.consume(take);
        n += take;
        if line.is_some() {
            break;
        }
    }
    n as i64
}

/// `size` bytes at `src` written to standard output and flushed: `size`, or `-1`.
///
/// # Safety
///
/// `src` holds `size` readable bytes.
pub(crate) unsafe extern "C" fn cl_stdout_write(src: *const u8, size: i64) -> i64 {
    let Ok(len) = usize::try_from(size) else { return -1 };
    let data = if len == 0 { &[][..] } else { std::slice::from_raw_parts(src, len) };
    let mut stdout = std::io::stdout().lock();
    match stdout.write_all(data).and_then(|()| stdout.flush()) {
        Ok(()) => size,
        Err(_) => -1,
    }
}
