use std::fs;
use std::io::{Read as IoRead, Seek, Write as IoWrite};

use super::read_field;

/// `size` bytes from `src_ptr` written into the file at `file_offset`, creating
/// it. Each file call names its file by a path field ([`read_field`]); `-1` on failure.
pub(crate) unsafe extern "C" fn cl_file_write_from_ptr(
    path_ptr: *const u8,
    path_cap: i64,
    src_ptr: *const u8,
    file_offset: i64,
    size: i64,
) -> i64 {
    if size <= 0 || file_offset < 0 {
        return -1;
    }
    let Some(path) = read_field(path_ptr, path_cap) else { return -1 };
    let mut file = match fs::OpenOptions::new().write(true).create(true).open(path) {
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
    path_cap: i64,
    dst_ptr: *mut u8,
    file_offset: i64,
    size: i64,
) -> i64 {
    if size <= 0 {
        return -1;
    }
    let Some(path) = read_field(path_ptr, path_cap) else { return -1 };
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return -1,
    };
    // The offset unsigned, and no more than the file has.
    let from = file_offset as u64;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let want = (size as u64).min(file_len.saturating_sub(from)) as usize;
    if want == 0 {
        return 0;
    }
    if file.seek(std::io::SeekFrom::Start(from)).is_err() {
        return -1;
    }
    let dst = std::slice::from_raw_parts_mut(dst_ptr, want);
    let mut total = 0usize;
    while total < want {
        match file.read(&mut dst[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(_) => return -1,
        }
    }
    total as i64
}

/// The length of the file at the path in the `path_cap`-byte field at
/// `path_ptr` ([`read_field`]), or -1 if there is none.
pub(crate) unsafe extern "C" fn cl_file_size(path_ptr: *const u8, path_cap: i64) -> i64 {
    let Some(path) = read_field(path_ptr, path_cap) else { return -1 };
    match fs::metadata(path) {
        Ok(m) if m.is_file() => i64::try_from(m.len()).unwrap_or(-1),
        _ => -1,
    }
}

/// Make the file at `path_ptr` `len` bytes long, creating it if there is
/// none: cut short, or extended with zeros. 0, or -1.
pub(crate) unsafe extern "C" fn cl_file_set_len(path_ptr: *const u8, path_cap: i64, len: i64) -> i64 {
    let Some(path) = read_field(path_ptr, path_cap) else { return -1 };
    if len < 0 {
        return -1;
    }
    let file = fs::OpenOptions::new().write(true).create(true).truncate(false).open(path);
    match file.and_then(|f| f.set_len(len as u64)) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Create the directory at the path in the field at `path_ptr`, and every one
/// missing on the way: `0`, or `-1` when that cannot be done.
pub(crate) unsafe extern "C" fn cl_file_create_dir_all(path_ptr: *const u8, path_cap: i64) -> i64 {
    let Some(path) = read_field(path_ptr, path_cap) else { return -1 };
    match fs::create_dir_all(path) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use tempfile::TempDir;

    /// The field a C string fills, its NUL included.
    fn cap(c: &CString) -> i64 {
        c.as_bytes_with_nul().len() as i64
    }

    /// A path that fills its field has no NUL, and is read whole.
    #[test]
    fn a_path_filling_its_field_is_read_whole() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("whole.bin");
        std::fs::write(&path, b"abc").unwrap();
        let mut field = path.to_str().unwrap().as_bytes().to_vec();
        let n = field.len() as i64;
        field.extend_from_slice(b"junk");
        assert_eq!(unsafe { cl_file_size(field.as_ptr(), n) }, 3);
    }

    #[test]
    fn size_is_the_files_length_and_minus_one_for_none() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("sz.bin");
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        unsafe {
            assert_eq!(cl_file_size(path_c.as_ptr() as *const u8, cap(&path_c)), -1);
            std::fs::write(&path, b"twelve bytes").unwrap();
            assert_eq!(cl_file_size(path_c.as_ptr() as *const u8, cap(&path_c)), 12);
        }
    }

    #[test]
    fn set_len_cuts_extends_and_creates() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("len.bin");
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        let (p, c) = (path_c.as_ptr() as *const u8, cap(&path_c));
        unsafe {
            assert_eq!(cl_file_set_len(p, c, 3), 0);
            assert_eq!(std::fs::read(&path).unwrap(), [0, 0, 0]);
            std::fs::write(&path, b"abcdef").unwrap();
            assert_eq!(cl_file_set_len(p, c, 2), 0);
            assert_eq!(std::fs::read(&path).unwrap(), b"ab");
            assert_eq!(cl_file_set_len(p, c, 4), 0);
            assert_eq!(std::fs::read(&path).unwrap(), b"ab\0\0");
            assert_eq!(cl_file_set_len(p, c, -1), -1);
        }
    }

    #[test]
    fn write_from_ptr_writes_arbitrary_pointer() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("fp.bin");
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        let src = b"from_ptr payload".to_vec();
        unsafe {
            let n = cl_file_write_from_ptr(
                path_c.as_ptr() as *const u8, cap(&path_c),
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
                path_c.as_ptr() as *const u8, cap(&path_c),
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
            // zero size
            assert_eq!(
                cl_file_write_from_ptr(path_c.as_ptr() as *const u8, cap(&path_c), src.as_ptr(), 0, 0),
                -1
            );
            // negative offset
            assert_eq!(
                cl_file_write_from_ptr(path_c.as_ptr() as *const u8, cap(&path_c), src.as_ptr(), -1, 1),
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
                path_c.as_ptr() as *const u8, cap(&path_c),
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
                path_c.as_ptr() as *const u8, cap(&path_c),
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
                cl_file_read_to_ptr(path_c.as_ptr() as *const u8, cap(&path_c), dst.as_mut_ptr(), 0, 0),
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
                path_c.as_ptr() as *const u8, cap(&path_c),
                dst.as_mut_ptr(),
                0,
                dst.len() as i64,
            );
            assert_eq!(n, -1);
        }
    }
}
