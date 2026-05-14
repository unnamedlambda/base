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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use tempfile::TempDir;

    /// Layout helper: writes a null-terminated path at `path_off` and `src` bytes
    /// at `src_off` inside a fresh memory buffer. Returns (memory, path_off, src_off).
    fn make_memory(path: &str, src: &[u8]) -> (Vec<u8>, usize, usize) {
        let path_c = CString::new(path).unwrap();
        let path_bytes = path_c.as_bytes_with_nul();
        let path_off = 0;
        let src_off = 512;
        let dst_off = 1024;
        let buf_len = dst_off + src.len().max(4096);
        let mut mem = vec![0u8; buf_len];
        mem[path_off..path_off + path_bytes.len()].copy_from_slice(path_bytes);
        mem[src_off..src_off + src.len()].copy_from_slice(src);
        (mem, path_off, src_off)
    }

    #[test]
    fn write_then_read_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rt.bin");
        let payload = b"Hello, CLIF!";
        let (mut mem, path_off, src_off) = make_memory(path.to_str().unwrap(), payload);
        let dst_off = 1024;
        unsafe {
            let n = cl_file_write(
                mem.as_mut_ptr(),
                path_off as i64,
                src_off as i64,
                0,
                payload.len() as i64,
            );
            assert_eq!(n, payload.len() as i64);
            let n = cl_file_read(
                mem.as_mut_ptr(),
                path_off as i64,
                dst_off as i64,
                0,
                payload.len() as i64,
            );
            assert_eq!(n, payload.len() as i64);
        }
        assert_eq!(&mem[dst_off..dst_off + payload.len()], payload);
    }

    #[test]
    fn write_at_offset_preserves_prefix() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("off.bin");
        std::fs::write(&path, b"AAAAAAAA").unwrap();
        let suffix = b"BBBBB";
        let (mut mem, path_off, src_off) = make_memory(path.to_str().unwrap(), suffix);
        unsafe {
            let n = cl_file_write(
                mem.as_mut_ptr(),
                path_off as i64,
                src_off as i64,
                8,
                suffix.len() as i64,
            );
            assert_eq!(n, suffix.len() as i64);
        }
        let read = std::fs::read(&path).unwrap();
        assert_eq!(&read, b"AAAAAAAABBBBB");
    }

    #[test]
    fn read_at_offset() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("ro.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        let (mut mem, path_off, _) = make_memory(path.to_str().unwrap(), &[]);
        let dst_off = 1024;
        unsafe {
            let n = cl_file_read(mem.as_mut_ptr(), path_off as i64, dst_off as i64, 3, 4);
            assert_eq!(n, 4);
        }
        assert_eq!(&mem[dst_off..dst_off + 4], b"3456");
    }

    #[test]
    fn binary_data_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bin.bin");
        let payload: Vec<u8> = (0u8..=255).collect();
        let (mut mem, path_off, src_off) = make_memory(path.to_str().unwrap(), &payload);
        let dst_off = 1024;
        unsafe {
            cl_file_write(
                mem.as_mut_ptr(),
                path_off as i64,
                src_off as i64,
                0,
                payload.len() as i64,
            );
            cl_file_read(
                mem.as_mut_ptr(),
                path_off as i64,
                dst_off as i64,
                0,
                payload.len() as i64,
            );
        }
        assert_eq!(&mem[dst_off..dst_off + payload.len()], payload.as_slice());
    }

    #[test]
    fn read_size_zero_reads_whole_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("whole.bin");
        let payload = b"the entire file";
        std::fs::write(&path, payload).unwrap();
        let (mut mem, path_off, _) = make_memory(path.to_str().unwrap(), &[]);
        let dst_off = 1024;
        unsafe {
            let n = cl_file_read(mem.as_mut_ptr(), path_off as i64, dst_off as i64, 0, 0);
            assert_eq!(n, payload.len() as i64);
        }
        assert_eq!(&mem[dst_off..dst_off + payload.len()], payload);
    }

    #[test]
    fn write_size_zero_treats_src_as_cstring() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cstr.bin");
        let mut src = b"cstring mode".to_vec();
        src.push(0);
        src.extend_from_slice(b"ignored after null");
        let (mut mem, path_off, src_off) = make_memory(path.to_str().unwrap(), &src);
        unsafe {
            let n = cl_file_write(mem.as_mut_ptr(), path_off as i64, src_off as i64, 0, 0);
            assert_eq!(n, b"cstring mode".len() as i64);
        }
        assert_eq!(std::fs::read(&path).unwrap(), b"cstring mode");
    }

    #[test]
    fn create_truncates_existing_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("trunc.bin");
        std::fs::write(&path, b"longer existing contents").unwrap();
        let new_payload = b"short";
        let (mut mem, path_off, src_off) = make_memory(path.to_str().unwrap(), new_payload);
        unsafe {
            cl_file_write(
                mem.as_mut_ptr(),
                path_off as i64,
                src_off as i64,
                0,
                new_payload.len() as i64,
            );
        }
        assert_eq!(std::fs::read(&path).unwrap(), new_payload);
    }

    #[test]
    fn read_returns_neg1_for_missing_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("absent.bin");
        let (mut mem, path_off, _) = make_memory(path.to_str().unwrap(), &[]);
        unsafe {
            let n = cl_file_read(mem.as_mut_ptr(), path_off as i64, 1024, 0, 16);
            assert_eq!(n, -1);
        }
    }

    #[test]
    fn read_partial_when_size_exceeds_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("partial.bin");
        std::fs::write(&path, b"tiny").unwrap();
        let (mut mem, path_off, _) = make_memory(path.to_str().unwrap(), &[]);
        let dst_off = 1024;
        unsafe {
            let n = cl_file_read(mem.as_mut_ptr(), path_off as i64, dst_off as i64, 0, 100);
            assert_eq!(n, 4);
        }
        assert_eq!(&mem[dst_off..dst_off + 4], b"tiny");
    }

    #[test]
    fn write_from_ptr_writes_arbitrary_pointer() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("fp.bin");
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        let src = b"from_ptr payload".to_vec();
        unsafe {
            let n = cl_file_write_from_ptr(
                path_c.as_ptr() as *const u8,
                src.as_ptr(),
                0,
                src.len() as i64,
            );
            assert_eq!(n, src.len() as i64);
        }
        assert_eq!(std::fs::read(&path).unwrap(), src);
    }

    #[test]
    fn write_from_ptr_appends_at_offset() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("fp_off.bin");
        std::fs::write(&path, b"AAAA").unwrap();
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        let src = b"BB";
        unsafe {
            let n = cl_file_write_from_ptr(
                path_c.as_ptr() as *const u8,
                src.as_ptr(),
                4,
                src.len() as i64,
            );
            assert_eq!(n, src.len() as i64);
        }
        assert_eq!(std::fs::read(&path).unwrap(), b"AAAABB");
    }

    #[test]
    fn write_from_ptr_rejects_invalid() {
        let path_c = CString::new("/dev/null").unwrap();
        let src = b"x";
        unsafe {
            // null path
            assert_eq!(
                cl_file_write_from_ptr(std::ptr::null(), src.as_ptr(), 0, 1),
                -1
            );
            // null src
            assert_eq!(
                cl_file_write_from_ptr(path_c.as_ptr() as *const u8, std::ptr::null(), 0, 1),
                -1
            );
            // zero size
            assert_eq!(
                cl_file_write_from_ptr(path_c.as_ptr() as *const u8, src.as_ptr(), 0, 0),
                -1
            );
            // negative offset
            assert_eq!(
                cl_file_write_from_ptr(path_c.as_ptr() as *const u8, src.as_ptr(), -1, 1),
                -1
            );
        }
    }

    #[test]
    fn read_to_ptr_reads_arbitrary_pointer() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rp.bin");
        let payload = b"to_ptr payload";
        std::fs::write(&path, payload).unwrap();
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        let mut dst = vec![0u8; payload.len()];
        unsafe {
            let n = cl_file_read_to_ptr(
                path_c.as_ptr() as *const u8,
                dst.as_mut_ptr(),
                0,
                dst.len() as i64,
            );
            assert_eq!(n, payload.len() as i64);
        }
        assert_eq!(&dst, payload);
    }

    #[test]
    fn read_to_ptr_at_offset() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("rp_off.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        let mut dst = vec![0u8; 4];
        unsafe {
            let n = cl_file_read_to_ptr(
                path_c.as_ptr() as *const u8,
                dst.as_mut_ptr(),
                5,
                dst.len() as i64,
            );
            assert_eq!(n, 4);
        }
        assert_eq!(&dst, b"5678");
    }

    #[test]
    fn read_to_ptr_rejects_invalid() {
        let path_c = CString::new("/some/path").unwrap();
        let mut dst = [0u8; 1];
        unsafe {
            assert_eq!(
                cl_file_read_to_ptr(std::ptr::null(), dst.as_mut_ptr(), 0, 1),
                -1
            );
            assert_eq!(
                cl_file_read_to_ptr(path_c.as_ptr() as *const u8, std::ptr::null_mut(), 0, 1),
                -1
            );
            assert_eq!(
                cl_file_read_to_ptr(path_c.as_ptr() as *const u8, dst.as_mut_ptr(), 0, 0),
                -1
            );
        }
    }

    #[test]
    fn read_to_ptr_missing_file_returns_neg1() {
        let path_c = CString::new("/nonexistent/path/xyz123").unwrap();
        let mut dst = [0u8; 8];
        unsafe {
            let n = cl_file_read_to_ptr(
                path_c.as_ptr() as *const u8,
                dst.as_mut_ptr(),
                0,
                dst.len() as i64,
            );
            assert_eq!(n, -1);
        }
    }
}
