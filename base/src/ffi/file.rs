use std::fs;
use std::io::{Read as IoRead, Seek, Write as IoWrite};

use super::{read_cstr, read_cstr_ptr};

pub(crate) unsafe extern "C" fn cl_file_read(
    ptr: *mut u8,
    path_off: i64,
    dst_off: i64,
    file_offset: i64,
    size: i64,
) -> i64 {
    let filename = read_cstr(ptr, path_off as usize);
    let mut file = match fs::File::open(&filename) {
        Ok(f) => f,
        Err(_) => return -1,
    };
    if file_offset > 0 {
        let _ = file.seek(std::io::SeekFrom::Start(file_offset as u64));
    }
    if size == 0 {
        let file_len = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
        if file_len == 0 {
            return 0;
        }
        let dst = std::slice::from_raw_parts_mut(ptr.add(dst_off as usize), file_len);
        let mut total = 0;
        while total < file_len {
            match file.read(&mut dst[total..]) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(_) => break,
            }
        }
        total as i64
    } else {
        let dst = std::slice::from_raw_parts_mut(ptr.add(dst_off as usize), size as usize);
        match file.read(dst) {
            Ok(n) => n as i64,
            Err(_) => -1,
        }
    }
}

/// Write `size` bytes from an arbitrary host pointer into a file at the given
/// offset.  Opens (creating if needed), seeks, writes, and closes each call.
pub(crate) unsafe extern "C" fn cl_file_write_from_ptr(
    path_ptr: *const u8,
    src_ptr: *const u8,
    file_offset: i64,
    size: i64,
) -> i64 {
    if path_ptr.is_null() || src_ptr.is_null() || size <= 0 || file_offset < 0 {
        return -1;
    }
    let path = read_cstr_ptr(path_ptr);
    let mut file = match fs::OpenOptions::new().write(true).create(true).open(&path) {
        Ok(f) => f,
        Err(_) => return -1,
    };
    if file
        .seek(std::io::SeekFrom::Start(file_offset as u64))
        .is_err()
    {
        return -1;
    }
    let src = std::slice::from_raw_parts(src_ptr, size as usize);
    match file.write_all(src) {
        Ok(_) => size,
        Err(_) => -1,
    }
}

/// Read `size` bytes from a file directly into an arbitrary host pointer.
pub(crate) unsafe extern "C" fn cl_file_read_to_ptr(
    path_ptr: *const u8,
    dst_ptr: *mut u8,
    file_offset: i64,
    size: i64,
) -> i64 {
    if path_ptr.is_null() || dst_ptr.is_null() || size <= 0 {
        return -1;
    }
    let path = read_cstr_ptr(path_ptr);
    let mut file = match fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return -1,
    };
    if file_offset > 0 {
        if file
            .seek(std::io::SeekFrom::Start(file_offset as u64))
            .is_err()
        {
            return -1;
        }
    }
    let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
    let mut total = 0usize;
    while total < dst.len() {
        match file.read(&mut dst[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(_) => return -1,
        }
    }
    total as i64
}

pub(crate) unsafe extern "C" fn cl_file_write(
    ptr: *mut u8,
    path_off: i64,
    src_off: i64,
    file_offset: i64,
    size: i64,
) -> i64 {
    let filename = read_cstr(ptr, path_off as usize);
    let mut file = if file_offset == 0 {
        match fs::File::create(&filename) {
            Ok(f) => f,
            Err(_) => return -1,
        }
    } else {
        match fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(&filename)
        {
            Ok(mut f) => {
                let _ = f.seek(std::io::SeekFrom::Start(file_offset as u64));
                f
            }
            Err(_) => return -1,
        }
    };
    let written = if size == 0 {
        let base = ptr.add(src_off as usize);
        let mut len = 0;
        while *base.add(len) != 0 {
            len += 1;
        }
        if len > 0 {
            let data = std::slice::from_raw_parts(base, len);
            match file.write_all(data) {
                Ok(_) => len as i64,
                Err(_) => -1,
            }
        } else {
            0
        }
    } else {
        let data = std::slice::from_raw_parts(ptr.add(src_off as usize), size as usize);
        match file.write_all(data) {
            Ok(_) => size,
            Err(_) => -1,
        }
    };
    if written >= 0 {
        let _ = file.sync_all();
    }
    written
}
