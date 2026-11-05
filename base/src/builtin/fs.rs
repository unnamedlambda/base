//! Files and directories through `std::fs`. A regular file is opened by path to
//! a handle, read and written at offsets, sized, cut, synced and closed; a
//! directory is opened to a handle and listed an entry at a time. Creating a
//! directory, and removing or renaming by path, act on the path alone.
//!
//! A handle refuses what it was not opened for, as Windows does, so a program
//! sees the same answers on every system.

use std::ffi::c_char;
use std::fs;

use positioned_io::{ReadAt, WriteAt};

use super::read_field;

/// An open regular file, and what it was opened for.
pub(crate) struct File {
    file: fs::File,
    read: bool,
    write: bool,
}

/// An open directory: the entries not yet listed.
pub(crate) struct Dir(fs::ReadDir);

const READ: i32 = 1;
const WRITE: i32 = 2;
const CREATE: i32 = 4;
const TRUNCATE: i32 = 8;

/// The path in the `cap`-byte field at `path`, or `None` for one not UTF-8.
unsafe fn path<'a>(path: *const u8, cap: i64) -> Option<&'a str> {
    read_field(path, cap).filter(|p| !p.is_empty())
}

/// The regular file at the path in the `cap`-byte field at `path` ([`read_field`]),
/// opened for what `mode` sets: `1` read, `2` write, `4` create it if there is
/// none, `8` cut it to nothing. Null where it does not open: a mode with any
/// other bit, neither read nor write, create or cut without write, no file and
/// no create, or something there that is not a regular file.
unsafe extern "C" fn base_fs_open(path_ptr: *const u8, cap: i64, mode: i32) -> *mut File {
    let (read, write) = (mode & READ != 0, mode & WRITE != 0);
    if mode & !(READ | WRITE | CREATE | TRUNCATE) != 0 || !(read || write) {
        return std::ptr::null_mut();
    }
    let Some(p) = path(path_ptr, cap) else { return std::ptr::null_mut() };
    let opened = fs::OpenOptions::new()
        .read(read)
        .write(write)
        .create(mode & CREATE != 0)
        .truncate(mode & TRUNCATE != 0)
        .open(p);
    match opened {
        Ok(file) if file.metadata().is_ok_and(|m| m.is_file()) => Box::into_raw(Box::new(File { file, read, write })),
        _ => std::ptr::null_mut(),
    }
}

/// `f` closed.
unsafe extern "C" fn base_fs_close(f: *mut File) {
    drop(Box::from_raw(f));
}

/// Up to `len` bytes of `f` from `offset` read into `buf`, stopping only at the
/// file's end: how many, or `-1` — for a file not opened to read, a negative
/// length or offset, or a failed read.
unsafe extern "C" fn base_fs_read(f: *mut File, buf: *mut u8, len: i64, offset: i64) -> i64 {
    let f = &*f;
    let (Ok(len), Ok(at)) = (usize::try_from(len), u64::try_from(offset)) else { return -1 };
    if !f.read {
        return -1;
    }
    let dst = if len == 0 { &mut [][..] } else { std::slice::from_raw_parts_mut(buf, len) };
    let mut n = 0;
    while n < len {
        match f.file.read_at(at + n as u64, &mut dst[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return -1,
        }
    }
    n as i64
}

/// The `len` bytes at `buf` written into `f` at `offset`, all of them, a gap
/// before them reading back as zeros: `len`, or `-1` — for a file not opened
/// to write, a negative length or offset, or a failed write.
unsafe extern "C" fn base_fs_write(f: *mut File, buf: *const u8, len: i64, offset: i64) -> i64 {
    let f = &mut *f;
    let (Ok(n), Ok(at)) = (usize::try_from(len), u64::try_from(offset)) else { return -1 };
    if !f.write {
        return -1;
    }
    let src = if n == 0 { &[][..] } else { std::slice::from_raw_parts(buf, n) };
    let mut done = 0;
    while done < n {
        match f.file.write_at(at + done as u64, &src[done..]) {
            Ok(0) => return -1,
            Ok(k) => done += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return -1,
        }
    }
    len
}

/// `f`'s length in bytes, or `-1`.
unsafe extern "C" fn base_fs_size(f: *mut File) -> i64 {
    (*f).file.metadata().ok().and_then(|m| i64::try_from(m.len()).ok()).unwrap_or(-1)
}

/// `f` made `len` bytes long, cut short or extended with zeros: `0`, or `-1`
/// for a file not opened to write or a negative length.
unsafe extern "C" fn base_fs_set_len(f: *mut File, len: i64) -> i64 {
    let f = &*f;
    match u64::try_from(len) {
        Ok(n) if f.write && f.file.set_len(n).is_ok() => 0,
        _ => -1,
    }
}

/// `f`'s contents and length on the storage device: `0`, or `-1` for a file
/// not opened to write.
unsafe extern "C" fn base_fs_sync(f: *mut File) -> i64 {
    let f = &*f;
    if f.write && f.file.sync_all().is_ok() {
        0
    } else {
        -1
    }
}

/// A directory made at the path, whose parent must be one: `0`, or `-1`.
unsafe extern "C" fn base_fs_create_dir(path_ptr: *const u8, cap: i64) -> i64 {
    match path(path_ptr, cap).map(fs::create_dir) {
        Some(Ok(())) => 0,
        _ => -1,
    }
}

/// The regular file at the path removed: `0`, or `-1`. A handle open on it
/// still reads and writes what it had.
unsafe extern "C" fn base_fs_remove_file(path_ptr: *const u8, cap: i64) -> i64 {
    let Some(p) = path(path_ptr, cap) else { return -1 };
    match fs::symlink_metadata(p) {
        Ok(m) if m.is_file() && fs::remove_file(p).is_ok() => 0,
        _ => -1,
    }
}

/// The empty directory at the path removed: `0`, or `-1`.
unsafe extern "C" fn base_fs_remove_dir(path_ptr: *const u8, cap: i64) -> i64 {
    match path(path_ptr, cap).map(fs::remove_dir) {
        Some(Ok(())) => 0,
        _ => -1,
    }
}

/// The regular file at the first path moved to the second, replacing a
/// regular file there: `0`, or `-1`.
unsafe extern "C" fn base_fs_rename(from: *const u8, from_cap: i64, to: *const u8, to_cap: i64) -> i64 {
    let (Some(from), Some(to)) = (path(from, from_cap), path(to, to_cap)) else { return -1 };
    let is_file = |p: &str| fs::symlink_metadata(p).is_ok_and(|m| m.is_file());
    let clear = fs::symlink_metadata(to).map_or(true, |m| m.is_file());
    if is_file(from) && clear && fs::rename(from, to).is_ok() {
        0
    } else {
        -1
    }
}

/// The directory at the path, opened for listing; null where there is none.
unsafe extern "C" fn base_fs_dir_open(path_ptr: *const u8, cap: i64) -> *mut Dir {
    match path(path_ptr, cap).map(fs::read_dir) {
        Some(Ok(d)) => Box::into_raw(Box::new(Dir(d))),
        _ => std::ptr::null_mut(),
    }
}

/// The next entry of `d`, in the order the system lists them: its name's
/// length, with as many of its bytes as fit in `cap` written at `out`; `0`
/// when every entry has been listed, `-1` for one that fails or is not UTF-8.
unsafe extern "C" fn base_fs_dir_next(d: *mut Dir, out: *mut c_char, cap: i64) -> i64 {
    match (*d).0.next() {
        None => 0,
        Some(Ok(e)) => match e.file_name().to_str() {
            Some(name) => {
                let n = name.len().min(cap.max(0) as usize);
                std::ptr::copy_nonoverlapping(name.as_ptr(), out.cast(), n);
                name.len() as i64
            }
            None => -1,
        },
        Some(Err(_)) => -1,
    }
}

/// `d` closed.
unsafe extern "C" fn base_fs_dir_close(d: *mut Dir) {
    drop(Box::from_raw(d));
}

crate::builtin::symbols!(
    base_fs_open,
    base_fs_close,
    base_fs_read,
    base_fs_write,
    base_fs_size,
    base_fs_set_len,
    base_fs_sync,
    base_fs_create_dir,
    base_fs_remove_file,
    base_fs_remove_dir,
    base_fs_rename,
    base_fs_dir_open,
    base_fs_dir_next,
    base_fs_dir_close,
);

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// A path as the field a program hands over.
    fn field(p: &std::path::Path) -> (Vec<u8>, i64) {
        let b = p.to_str().unwrap().as_bytes().to_vec();
        let n = b.len() as i64;
        (b, n)
    }

    #[test]
    fn a_file_is_written_read_cut_and_synced_through_its_handle() {
        let tmp = TempDir::new().unwrap();
        let (p, n) = field(&tmp.path().join("f.bin"));
        unsafe {
            assert!(base_fs_open(p.as_ptr(), n, READ).is_null(), "no file and no create");
            assert!(base_fs_open(p.as_ptr(), n, READ | CREATE).is_null(), "create without write");
            let f = base_fs_open(p.as_ptr(), n, READ | WRITE | CREATE);
            assert!(!f.is_null());
            assert_eq!(base_fs_write(f, b"abc".as_ptr(), 3, 2), 3);
            assert_eq!(base_fs_size(f), 5);
            let mut got = [9u8; 8];
            assert_eq!(base_fs_read(f, got.as_mut_ptr(), 8, 0), 5);
            assert_eq!(&got[..5], b"\0\0abc");
            assert_eq!(base_fs_set_len(f, 1), 0);
            assert_eq!(base_fs_size(f), 1);
            assert_eq!(base_fs_sync(f), 0);
            assert_eq!(base_fs_read(f, got.as_mut_ptr(), -1, 0), -1);
            base_fs_close(f);
            let r = base_fs_open(p.as_ptr(), n, READ);
            assert_eq!(base_fs_write(r, b"x".as_ptr(), 1, 0), -1, "not opened to write");
            assert_eq!(base_fs_sync(r), -1);
            base_fs_close(r);
        }
    }

    #[test]
    fn directories_are_made_listed_and_removed() {
        let tmp = TempDir::new().unwrap();
        let (d, dn) = field(&tmp.path().join("d"));
        let (a, an) = field(&tmp.path().join("d").join("a"));
        let (b, bn) = field(&tmp.path().join("d").join("b"));
        unsafe {
            assert_eq!(base_fs_create_dir(d.as_ptr(), dn), 0);
            assert_eq!(base_fs_create_dir(d.as_ptr(), dn), -1);
            assert!(base_fs_open(d.as_ptr(), dn, READ).is_null(), "a directory is not a file");
            base_fs_close(base_fs_open(a.as_ptr(), an, WRITE | CREATE));
            assert_eq!(base_fs_rename(a.as_ptr(), an, b.as_ptr(), bn), 0);
            let dir = base_fs_dir_open(d.as_ptr(), dn);
            let mut name = [0u8; 4];
            assert_eq!(base_fs_dir_next(dir, name.as_mut_ptr().cast(), 4), 1);
            assert_eq!(name[0], b'b');
            assert_eq!(base_fs_dir_next(dir, name.as_mut_ptr().cast(), 4), 0);
            base_fs_dir_close(dir);
            assert_eq!(base_fs_remove_dir(d.as_ptr(), dn), -1, "not empty");
            assert_eq!(base_fs_remove_file(b.as_ptr(), bn), 0);
            assert_eq!(base_fs_remove_dir(d.as_ptr(), dn), 0);
        }
    }
}
