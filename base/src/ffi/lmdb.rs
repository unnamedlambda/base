use lmdb_zero as lmdb;
use std::collections::HashMap;

use super::{clear_ctx_slot, read_cstr_ptr, read_ctx_mut, read_ctx_ref, write_ctx_slot};

pub(crate) struct CraneliftLmdbContext {
    envs: HashMap<u32, (lmdb::Environment, liblmdb_sys::MDB_dbi)>,
    active_write_txns: HashMap<u32, *mut liblmdb_sys::MDB_txn>,
    next_handle: u32,
}

impl Drop for CraneliftLmdbContext {
    fn drop(&mut self) {
        for (_handle, txn) in self.active_write_txns.drain() {
            unsafe {
                liblmdb_sys::mdb_txn_abort(txn);
            }
        }
    }
}

fn lmdb_raw_begin_txn(env: &lmdb::Environment, readonly: bool) -> *mut liblmdb_sys::MDB_txn {
    let mut txn = std::ptr::null_mut();
    let flags = if readonly { liblmdb_sys::MDB_RDONLY } else { 0 };
    unsafe {
        if liblmdb_sys::mdb_txn_begin(env.as_raw(), std::ptr::null_mut(), flags, &mut txn) == 0 {
            txn
        } else {
            std::ptr::null_mut()
        }
    }
}

fn lmdb_raw_put(
    txn: *mut liblmdb_sys::MDB_txn,
    dbi: liblmdb_sys::MDB_dbi,
    key: &[u8],
    val: &[u8],
) -> bool {
    let mut k = liblmdb_sys::MDB_val {
        mv_size: key.len(),
        mv_data: key.as_ptr() as *const _,
    };
    let mut v = liblmdb_sys::MDB_val {
        mv_size: val.len(),
        mv_data: val.as_ptr() as *const _,
    };
    unsafe { liblmdb_sys::mdb_put(txn, dbi, &mut k, &mut v, 0) == 0 }
}

fn lmdb_raw_get(
    txn: *mut liblmdb_sys::MDB_txn,
    dbi: liblmdb_sys::MDB_dbi,
    key: &[u8],
) -> Option<Vec<u8>> {
    let mut k = liblmdb_sys::MDB_val {
        mv_size: key.len(),
        mv_data: key.as_ptr() as *const _,
    };
    let mut v = liblmdb_sys::MDB_val {
        mv_size: 0,
        mv_data: std::ptr::null(),
    };
    unsafe {
        if liblmdb_sys::mdb_get(txn, dbi, &mut k, &mut v) == 0 {
            Some(std::slice::from_raw_parts(v.mv_data as *const u8, v.mv_size).to_vec())
        } else {
            None
        }
    }
}

fn lmdb_raw_del(txn: *mut liblmdb_sys::MDB_txn, dbi: liblmdb_sys::MDB_dbi, key: &[u8]) -> bool {
    let mut k = liblmdb_sys::MDB_val {
        mv_size: key.len(),
        mv_data: key.as_ptr() as *const _,
    };
    unsafe { liblmdb_sys::mdb_del(txn, dbi, &mut k, std::ptr::null_mut()) == 0 }
}

fn lmdb_raw_cursor_scan(
    txn: *mut liblmdb_sys::MDB_txn,
    dbi: liblmdb_sys::MDB_dbi,
    start_key: Option<&[u8]>,
    max_entries: usize,
) -> Vec<u8> {
    let mut result = Vec::new();
    result.extend_from_slice(&0u32.to_le_bytes());
    let mut cursor: *mut liblmdb_sys::MDB_cursor = std::ptr::null_mut();
    unsafe {
        if liblmdb_sys::mdb_cursor_open(txn, dbi, &mut cursor) != 0 {
            return result;
        }
        let mut k = liblmdb_sys::MDB_val {
            mv_size: 0,
            mv_data: std::ptr::null(),
        };
        let mut v = liblmdb_sys::MDB_val {
            mv_size: 0,
            mv_data: std::ptr::null(),
        };
        let first_rc = if let Some(sk) = start_key {
            k.mv_size = sk.len();
            k.mv_data = sk.as_ptr() as *const _;
            liblmdb_sys::mdb_cursor_get(
                cursor,
                &mut k,
                &mut v,
                liblmdb_sys::MDB_cursor_op::MDB_SET_RANGE,
            )
        } else {
            liblmdb_sys::mdb_cursor_get(
                cursor,
                &mut k,
                &mut v,
                liblmdb_sys::MDB_cursor_op::MDB_FIRST,
            )
        };
        let mut count = 0u32;
        if first_rc == 0 {
            loop {
                if count >= max_entries as u32 {
                    break;
                }
                if k.mv_size > u16::MAX as usize || v.mv_size > u16::MAX as usize {
                    break;
                }
                result.extend_from_slice(&(k.mv_size as u16).to_le_bytes());
                result.extend_from_slice(&(v.mv_size as u16).to_le_bytes());
                result.extend_from_slice(std::slice::from_raw_parts(
                    k.mv_data as *const u8,
                    k.mv_size,
                ));
                result.extend_from_slice(std::slice::from_raw_parts(
                    v.mv_data as *const u8,
                    v.mv_size,
                ));
                count += 1;
                if liblmdb_sys::mdb_cursor_get(
                    cursor,
                    &mut k,
                    &mut v,
                    liblmdb_sys::MDB_cursor_op::MDB_NEXT,
                ) != 0
                {
                    break;
                }
            }
        }
        liblmdb_sys::mdb_cursor_close(cursor);
        result[0..4].copy_from_slice(&count.to_le_bytes());
    }
    result
}

pub(crate) unsafe extern "C" fn cl_lmdb_init(ctx_slot_ptr: *mut *mut CraneliftLmdbContext) {
    let ctx = Box::new(CraneliftLmdbContext {
        envs: HashMap::new(),
        active_write_txns: HashMap::new(),
        next_handle: 0,
    });
    let _ = write_ctx_slot(ctx_slot_ptr, Box::into_raw(ctx));
}

pub(crate) unsafe extern "C" fn cl_lmdb_open(
    ctx_ptr: *mut CraneliftLmdbContext,
    path_ptr: *const u8,
    map_size_mb: i32,
) -> i32 {
    let Some(ctx) = read_ctx_mut::<CraneliftLmdbContext>(ctx_ptr) else {
        return -1;
    };
    let path_str = read_cstr_ptr(path_ptr);
    let map_size = if map_size_mb <= 0 {
        1024 * 1024 * 1024
    } else {
        (map_size_mb as usize) * 1024 * 1024
    };

    if std::fs::create_dir_all(&path_str).is_err() {
        return -1;
    }

    let env = match lmdb::EnvBuilder::new() {
        Ok(mut builder) => {
            builder.set_mapsize(map_size).ok();
            builder.set_maxdbs(1).ok();
            let flags = lmdb::open::WRITEMAP | lmdb::open::NOSYNC;
            match builder.open(&path_str, flags, 0o600) {
                Ok(env) => env,
                Err(_) => return -1,
            }
        }
        Err(_) => return -1,
    };

    let dbi = match lmdb::Database::open(&env, None, &lmdb::DatabaseOptions::defaults()) {
        Ok(db) => db.into_raw(),
        Err(_) => return -1,
    };

    let handle = ctx.next_handle;
    ctx.next_handle += 1;
    ctx.envs.insert(handle, (env, dbi));
    handle as i32
}

pub(crate) unsafe extern "C" fn cl_lmdb_put(
    ctx_ptr: *mut CraneliftLmdbContext,
    handle: u32,
    key_ptr: *const u8,
    key_len: i32,
    val_ptr: *const u8,
    val_len: i32,
) -> i32 {
    let Some(ctx) = read_ctx_mut::<CraneliftLmdbContext>(ctx_ptr) else {
        return -1;
    };
    if let Some((env, dbi)) = ctx.envs.get(&handle) {
        let key = std::slice::from_raw_parts(key_ptr, key_len as usize);
        let val = std::slice::from_raw_parts(val_ptr, val_len as usize);
        let dbi = *dbi;

        if let Some(&txn) = ctx.active_write_txns.get(&handle) {
            return if lmdb_raw_put(txn, dbi, key, val) {
                0
            } else {
                -1
            };
        }
        let txn = lmdb_raw_begin_txn(env, false);
        if !txn.is_null() {
            let ok = lmdb_raw_put(txn, dbi, key, val);
            if !ok {
                liblmdb_sys::mdb_txn_abort(txn);
                return -1;
            }
            if liblmdb_sys::mdb_txn_commit(txn) != 0 {
                return -1;
            }
            return 0;
        }
    }
    -1
}

pub(crate) unsafe extern "C" fn cl_lmdb_get(
    ctx_ptr: *mut CraneliftLmdbContext,
    handle: u32,
    key_ptr: *const u8,
    key_len: i32,
    result_ptr: *mut u8,
) -> i32 {
    let Some(ctx) = read_ctx_mut::<CraneliftLmdbContext>(ctx_ptr) else {
        return -1;
    };
    if let Some((env, dbi)) = ctx.envs.get(&handle) {
        let key = std::slice::from_raw_parts(key_ptr, key_len as usize);
        let dbi = *dbi;

        let (txn, owned) = match ctx.active_write_txns.get(&handle) {
            Some(&txn) => (txn, false),
            None => (lmdb_raw_begin_txn(env, true), true),
        };
        if !txn.is_null() {
            if let Some(val) = lmdb_raw_get(txn, dbi, key) {
                let len = val.len() as u32;
                let dst = result_ptr;
                std::ptr::copy_nonoverlapping(len.to_le_bytes().as_ptr(), dst, 4);
                std::ptr::copy_nonoverlapping(val.as_ptr(), dst.add(4), val.len());
                if owned {
                    liblmdb_sys::mdb_txn_abort(txn);
                }
                return len as i32;
            }
            if owned {
                liblmdb_sys::mdb_txn_abort(txn);
            }
        }
    }
    std::ptr::copy_nonoverlapping(0xFFFF_FFFFu32.to_le_bytes().as_ptr(), result_ptr, 4);
    -1
}

pub(crate) unsafe extern "C" fn cl_lmdb_delete(
    ctx_ptr: *mut CraneliftLmdbContext,
    handle: u32,
    key_ptr: *const u8,
    key_len: i32,
) -> i32 {
    let Some(ctx) = read_ctx_mut::<CraneliftLmdbContext>(ctx_ptr) else {
        return -1;
    };
    if let Some((env, dbi)) = ctx.envs.get(&handle) {
        let key = std::slice::from_raw_parts(key_ptr, key_len as usize);
        let dbi = *dbi;

        if let Some(&txn) = ctx.active_write_txns.get(&handle) {
            return if lmdb_raw_del(txn, dbi, key) { 0 } else { -1 };
        }
        let txn = lmdb_raw_begin_txn(env, false);
        if !txn.is_null() {
            let ok = lmdb_raw_del(txn, dbi, key);
            if !ok {
                liblmdb_sys::mdb_txn_abort(txn);
                return -1;
            }
            if liblmdb_sys::mdb_txn_commit(txn) != 0 {
                return -1;
            }
            return 0;
        }
    }
    -1
}

pub(crate) unsafe extern "C" fn cl_lmdb_begin_write_txn(
    ctx_ptr: *mut CraneliftLmdbContext,
    handle: u32,
) -> i32 {
    let Some(ctx) = read_ctx_mut::<CraneliftLmdbContext>(ctx_ptr) else {
        return -1;
    };
    if let Some(old_txn) = ctx.active_write_txns.remove(&handle) {
        liblmdb_sys::mdb_txn_abort(old_txn);
    }
    if let Some((env, _)) = ctx.envs.get(&handle) {
        let txn = lmdb_raw_begin_txn(env, false);
        if !txn.is_null() {
            ctx.active_write_txns.insert(handle, txn);
            return 0;
        }
    }
    -1
}

pub(crate) unsafe extern "C" fn cl_lmdb_commit_write_txn(
    ctx_ptr: *mut CraneliftLmdbContext,
    handle: u32,
) -> i32 {
    let Some(ctx) = read_ctx_mut::<CraneliftLmdbContext>(ctx_ptr) else {
        return -1;
    };
    if let Some(txn) = ctx.active_write_txns.remove(&handle) {
        return if liblmdb_sys::mdb_txn_commit(txn) == 0 {
            0
        } else {
            -1
        };
    }
    -1
}

pub(crate) unsafe extern "C" fn cl_lmdb_cursor_scan(
    ctx_ptr: *mut CraneliftLmdbContext,
    handle: u32,
    key_ptr: *const u8,
    key_len: i32,
    max_entries: i32,
    result_ptr: *mut u8,
) -> i32 {
    let Some(ctx) = read_ctx_mut::<CraneliftLmdbContext>(ctx_ptr) else {
        return 0;
    };
    if let Some((env, dbi)) = ctx.envs.get(&handle) {
        let start_key = if key_len > 0 {
            Some(std::slice::from_raw_parts(key_ptr, key_len as usize))
        } else {
            None
        };
        let dbi = *dbi;

        let (txn, owned) = match ctx.active_write_txns.get(&handle) {
            Some(&txn) => (txn, false),
            None => (lmdb_raw_begin_txn(env, true), true),
        };
        if !txn.is_null() {
            let result = lmdb_raw_cursor_scan(txn, dbi, start_key, max_entries as usize);
            std::ptr::copy_nonoverlapping(result.as_ptr(), result_ptr, result.len());
            let count = u32::from_le_bytes(result[0..4].try_into().unwrap());
            if owned {
                liblmdb_sys::mdb_txn_abort(txn);
            }
            return count as i32;
        }
    }
    std::ptr::copy_nonoverlapping(0u32.to_le_bytes().as_ptr(), result_ptr, 4);
    0
}

pub(crate) unsafe extern "C" fn cl_lmdb_sync(
    ctx_ptr: *const CraneliftLmdbContext,
    handle: u32,
) -> i32 {
    let Some(ctx) = read_ctx_ref::<CraneliftLmdbContext>(ctx_ptr) else {
        return -1;
    };
    if let Some((env, _)) = ctx.envs.get(&handle) {
        match env.sync(true) {
            Ok(_) => return 0,
            Err(_) => return -1,
        }
    }
    -1
}

pub(crate) unsafe extern "C" fn cl_lmdb_cleanup(ctx_slot_ptr: *mut *mut CraneliftLmdbContext) {
    let ctx_ptr = clear_ctx_slot::<CraneliftLmdbContext>(ctx_slot_ptr);
    drop(Box::from_raw(ctx_ptr));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn init() -> *mut CraneliftLmdbContext {
        let mut slot: *mut CraneliftLmdbContext = std::ptr::null_mut();
        unsafe { cl_lmdb_init(&mut slot) };
        slot
    }

    unsafe fn cleanup(slot: &mut *mut CraneliftLmdbContext) {
        cl_lmdb_cleanup(slot as *mut _);
    }

    fn open_db(slot: *mut CraneliftLmdbContext, dir: &std::path::Path) -> u32 {
        let path = CString::new(dir.to_str().unwrap()).unwrap();
        let h = unsafe { cl_lmdb_open(slot, path.as_ptr() as *const u8, 10) };
        assert!(h >= 0, "cl_lmdb_open failed");
        h as u32
    }

    unsafe fn put(slot: *mut CraneliftLmdbContext, h: u32, key: &[u8], val: &[u8]) -> i32 {
        cl_lmdb_put(slot, h, key.as_ptr(), key.len() as i32, val.as_ptr(), val.len() as i32)
    }

    unsafe fn get(slot: *mut CraneliftLmdbContext, h: u32, key: &[u8]) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; 4 + 4096];
        let rc = cl_lmdb_get(slot, h, key.as_ptr(), key.len() as i32, buf.as_mut_ptr());
        if rc < 0 {
            return None;
        }
        Some(buf[4..4 + rc as usize].to_vec())
    }

    unsafe fn del(slot: *mut CraneliftLmdbContext, h: u32, key: &[u8]) -> i32 {
        cl_lmdb_delete(slot, h, key.as_ptr(), key.len() as i32)
    }

    // Decode cursor scan output: [u32 count][u16 klen][u16 vlen][key][val]...
    fn decode_scan(buf: &[u8], count: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut entries = Vec::new();
        let mut pos = 4;
        for _ in 0..count {
            let klen = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
            let vlen = u16::from_le_bytes(buf[pos + 2..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let key = buf[pos..pos + klen].to_vec();
            pos += klen;
            let val = buf[pos..pos + vlen].to_vec();
            pos += vlen;
            entries.push((key, val));
        }
        entries
    }

    // ── lifecycle ─────────────────────────────────────────────────────────────

    #[test]
    fn init_then_cleanup_lifecycle() {
        let mut slot = init();
        assert!(!slot.is_null());
        unsafe { cleanup(&mut slot) };
        assert!(slot.is_null());
    }

    #[test]
    fn open_returns_nonneg_handle() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert!(h < u32::MAX);
            cleanup(&mut slot);
        }
    }

    #[test]
    fn open_bad_path_returns_neg1() {
        let mut slot = init();
        // Path is a null byte — CString would fail, so use a known-unwritable path
        let path = CString::new("/proc/1/cannot_create_here/lmdb").unwrap();
        unsafe {
            assert_eq!(cl_lmdb_open(slot, path.as_ptr() as *const u8, 10), -1);
            cleanup(&mut slot);
        }
    }

    // ── put / get / delete ────────────────────────────────────────────────────

    #[test]
    fn put_get_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert_eq!(put(slot, h, b"foo", b"barbaz"), 0);
            assert_eq!(get(slot, h, b"foo").unwrap(), b"barbaz");
            cleanup(&mut slot);
        }
    }

    #[test]
    fn get_nonexistent_returns_neg1() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert!(get(slot, h, b"missing").is_none());
            cleanup(&mut slot);
        }
    }

    #[test]
    fn put_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert_eq!(put(slot, h, b"k", b"first"), 0);
            assert_eq!(put(slot, h, b"k", b"second"), 0);
            assert_eq!(get(slot, h, b"k").unwrap(), b"second");
            cleanup(&mut slot);
        }
    }

    #[test]
    fn put_empty_value() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert_eq!(put(slot, h, b"k", b""), 0);
            // get returns 0 (empty value), buf[4..4] is empty
            let mut buf = [0u8; 4];
            let rc = cl_lmdb_get(slot, h, b"k".as_ptr(), 1, buf.as_mut_ptr());
            assert_eq!(rc, 0);
            cleanup(&mut slot);
        }
    }

    #[test]
    fn delete_then_get_returns_neg1() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            put(slot, h, b"key", b"val");
            assert_eq!(del(slot, h, b"key"), 0);
            assert!(get(slot, h, b"key").is_none());
            cleanup(&mut slot);
        }
    }

    #[test]
    fn delete_nonexistent_returns_neg1() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert_eq!(del(slot, h, b"nope"), -1);
            cleanup(&mut slot);
        }
    }

    // ── batch transactions ────────────────────────────────────────────────────

    #[test]
    fn batch_write_begin_put_commit() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert_eq!(cl_lmdb_begin_write_txn(slot, h), 0);
            assert_eq!(put(slot, h, b"a", b"1"), 0);
            assert_eq!(put(slot, h, b"b", b"2"), 0);
            assert_eq!(put(slot, h, b"c", b"3"), 0);
            assert_eq!(cl_lmdb_commit_write_txn(slot, h), 0);
            assert_eq!(get(slot, h, b"a").unwrap(), b"1");
            assert_eq!(get(slot, h, b"b").unwrap(), b"2");
            assert_eq!(get(slot, h, b"c").unwrap(), b"3");
            cleanup(&mut slot);
        }
    }

    #[test]
    fn empty_batch_commit() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert_eq!(cl_lmdb_begin_write_txn(slot, h), 0);
            assert_eq!(cl_lmdb_commit_write_txn(slot, h), 0);
            cleanup(&mut slot);
        }
    }

    #[test]
    fn commit_without_begin_returns_neg1() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert_eq!(cl_lmdb_commit_write_txn(slot, h), -1);
            cleanup(&mut slot);
        }
    }

    #[test]
    fn double_begin_aborts_previous_txn() {
        // begin → put k=v → begin again (aborts first) → commit empty → k not present
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert_eq!(cl_lmdb_begin_write_txn(slot, h), 0);
            assert_eq!(put(slot, h, b"k", b"v"), 0);
            assert_eq!(cl_lmdb_begin_write_txn(slot, h), 0);
            assert_eq!(cl_lmdb_commit_write_txn(slot, h), 0);
            assert!(get(slot, h, b"k").is_none());
            cleanup(&mut slot);
        }
    }

    #[test]
    fn uncommitted_batch_cleanup_aborts() {
        let dir = tempfile::tempdir().unwrap();
        // Open, begin write txn, put k=v, then cleanup without committing.
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert_eq!(cl_lmdb_begin_write_txn(slot, h), 0);
            assert_eq!(put(slot, h, b"k", b"v"), 0);
            cleanup(&mut slot);
        }
        // Reopen and verify write was not persisted.
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            assert!(get(slot, h, b"k").is_none());
            cleanup(&mut slot);
        }
    }

    // ── cursor scan ───────────────────────────────────────────────────────────

    #[test]
    fn cursor_scan_full() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            put(slot, h, b"a", b"1");
            put(slot, h, b"b", b"2");
            put(slot, h, b"c", b"3");

            let mut buf = vec![0u8; 1024];
            let count = cl_lmdb_cursor_scan(slot, h, std::ptr::null(), 0, 100, buf.as_mut_ptr());
            assert_eq!(count, 3);

            let entries = decode_scan(&buf, count as usize);
            assert_eq!(entries[0], (b"a".to_vec(), b"1".to_vec()));
            assert_eq!(entries[1], (b"b".to_vec(), b"2".to_vec()));
            assert_eq!(entries[2], (b"c".to_vec(), b"3".to_vec()));
            cleanup(&mut slot);
        }
    }

    #[test]
    fn cursor_scan_empty_db() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            let mut buf = vec![0u8; 64];
            let count = cl_lmdb_cursor_scan(slot, h, std::ptr::null(), 0, 100, buf.as_mut_ptr());
            assert_eq!(count, 0);
            cleanup(&mut slot);
        }
    }

    #[test]
    fn cursor_scan_with_start_key() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            put(slot, h, b"a", b"1");
            put(slot, h, b"b", b"2");
            put(slot, h, b"c", b"3");

            let start = b"b";
            let mut buf = vec![0u8; 1024];
            let count = cl_lmdb_cursor_scan(
                slot, h, start.as_ptr(), start.len() as i32, 100, buf.as_mut_ptr(),
            );
            assert_eq!(count, 2);
            let entries = decode_scan(&buf, count as usize);
            assert_eq!(entries[0].0, b"b");
            assert_eq!(entries[1].0, b"c");
            cleanup(&mut slot);
        }
    }

    #[test]
    fn cursor_scan_max_entries_limit() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            put(slot, h, b"a", b"1");
            put(slot, h, b"b", b"2");
            put(slot, h, b"c", b"3");

            let mut buf = vec![0u8; 1024];
            let count = cl_lmdb_cursor_scan(slot, h, std::ptr::null(), 0, 2, buf.as_mut_ptr());
            assert_eq!(count, 2);
            let entries = decode_scan(&buf, count as usize);
            assert_eq!(entries[0].0, b"a");
            assert_eq!(entries[1].0, b"b");
            cleanup(&mut slot);
        }
    }

    // ── sync ──────────────────────────────────────────────────────────────────

    #[test]
    fn sync_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h = open_db(slot, dir.path());
            put(slot, h, b"k", b"v");
            assert_eq!(cl_lmdb_sync(slot, h), 0);
            cleanup(&mut slot);
        }
    }

    // ── multi-db & error cases ────────────────────────────────────────────────

    #[test]
    fn multiple_databases_isolated() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        let mut slot = init();
        unsafe {
            let h1 = open_db(slot, dir1.path());
            let h2 = open_db(slot, dir2.path());
            assert_ne!(h1, h2);
            put(slot, h1, b"key", b"db1val");
            put(slot, h2, b"key", b"db2val");
            assert_eq!(get(slot, h1, b"key").unwrap(), b"db1val");
            assert_eq!(get(slot, h2, b"key").unwrap(), b"db2val");
            cleanup(&mut slot);
        }
    }

    #[test]
    fn invalid_handle_operations_return_neg1() {
        let mut slot = init();
        let mut buf = [0u8; 32];
        unsafe {
            assert_eq!(put(slot, 999, b"k", b"v"), -1);
            assert_eq!(cl_lmdb_get(slot, 999, b"k".as_ptr(), 1, buf.as_mut_ptr()), -1);
            assert_eq!(del(slot, 999, b"k"), -1);
            assert_eq!(cl_lmdb_begin_write_txn(slot, 999), -1);
            assert_eq!(cl_lmdb_commit_write_txn(slot, 999), -1);
            assert_eq!(cl_lmdb_sync(slot, 999), -1);
            cleanup(&mut slot);
        }
    }

    #[test]
    fn null_ctx_returns_errors() {
        let null = std::ptr::null_mut::<CraneliftLmdbContext>();
        let mut buf = [0u8; 32];
        let path = b"/tmp/x\0";
        unsafe {
            assert_eq!(cl_lmdb_open(null, path.as_ptr(), 10), -1);
            assert_eq!(cl_lmdb_put(null, 0, b"k".as_ptr(), 1, b"v".as_ptr(), 1), -1);
            assert_eq!(cl_lmdb_get(null, 0, b"k".as_ptr(), 1, buf.as_mut_ptr()), -1);
            assert_eq!(cl_lmdb_delete(null, 0, b"k".as_ptr(), 1), -1);
            assert_eq!(cl_lmdb_begin_write_txn(null, 0), -1);
            assert_eq!(cl_lmdb_commit_write_txn(null, 0), -1);
            assert_eq!(cl_lmdb_sync(null as *const _, 0), -1);
            // cursor_scan returns 0 (not -1) for null ctx
            let mut sbuf = [0u8; 32];
            assert_eq!(
                cl_lmdb_cursor_scan(null, 0, std::ptr::null(), 0, 10, sbuf.as_mut_ptr()),
                0
            );
        }
    }
}
