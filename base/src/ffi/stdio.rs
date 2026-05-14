use std::io::{BufRead, Write as IoWrite};

pub(crate) unsafe extern "C" fn cl_stdin_readline(
    ptr: *mut u8,
    dst_off: i64,
    max_len: i64,
) -> i64 {
    if max_len <= 0 {
        return 0;
    }

    let mut line = String::new();
    let mut stdin = std::io::stdin().lock();
    let read = match stdin.read_line(&mut line) {
        Ok(n) => n,
        Err(_) => return -1,
    };
    if read == 0 {
        return 0;
    }

    let buf = line.as_bytes();
    let copy_len = buf.len().min(max_len.saturating_sub(1) as usize);
    let dst = std::slice::from_raw_parts_mut(ptr.add(dst_off as usize), max_len as usize);
    dst[..copy_len].copy_from_slice(&buf[..copy_len]);
    dst[copy_len] = 0;
    copy_len as i64
}

pub(crate) unsafe extern "C" fn cl_stdout_write(ptr: *mut u8, src_off: i64, size: i64) -> i64 {
    if size < 0 {
        return -1;
    }

    let data = std::slice::from_raw_parts(ptr.add(src_off as usize), size as usize);
    let mut stdout = std::io::stdout().lock();
    match stdout.write_all(data) {
        Ok(_) => match stdout.flush() {
            Ok(_) => size,
            Err(_) => -1,
        },
        Err(_) => -1,
    }
}
