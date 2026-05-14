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
