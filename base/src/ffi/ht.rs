use std::collections::HashMap;

use super::{clear_ctx_slot, read_ctx_mut, read_ctx_ref, write_ctx_slot};

pub(crate) struct CraneliftHashTableContext {
    tables: HashMap<u32, HashMap<Vec<u8>, Vec<u8>>>,
    next_handle: u32,
}

impl CraneliftHashTableContext {
    fn new() -> Self {
        Self {
            tables: HashMap::new(),
            next_handle: 0,
        }
    }
}

pub(crate) unsafe extern "C" fn cl_ht_init(ctx_slot_ptr: *mut *mut CraneliftHashTableContext) {
    let ctx = Box::new(CraneliftHashTableContext::new());
    let _ = write_ctx_slot(ctx_slot_ptr, Box::into_raw(ctx));
}

pub(crate) unsafe extern "C" fn cl_ht_cleanup(ctx_slot_ptr: *mut *mut CraneliftHashTableContext) {
    let ctx_ptr = clear_ctx_slot::<CraneliftHashTableContext>(ctx_slot_ptr);
    if !ctx_ptr.is_null() {
        drop(Box::from_raw(ctx_ptr));
    }
}

pub(crate) unsafe extern "C" fn cl_ht_create(ctx: *mut CraneliftHashTableContext) -> u32 {
    let Some(ctx) = read_ctx_mut::<CraneliftHashTableContext>(ctx) else {
        return u32::MAX;
    };
    let handle = ctx.next_handle;
    ctx.next_handle += 1;
    ctx.tables.insert(handle, HashMap::new());
    handle
}

pub(crate) unsafe extern "C" fn cl_ht_lookup(
    ctx: *const CraneliftHashTableContext,
    key: *const u8,
    key_len: u32,
    result: *mut u8,
) -> u32 {
    let Some(ctx) = read_ctx_ref::<CraneliftHashTableContext>(ctx) else {
        return 0xFFFF_FFFF;
    };
    let key = std::slice::from_raw_parts(key, key_len as usize);
    if let Some(table) = ctx.tables.get(&0) {
        if let Some(val) = table.get(key) {
            std::ptr::copy_nonoverlapping(val.as_ptr(), result, val.len());
            return val.len() as u32;
        }
    }
    0xFFFF_FFFF
}

pub(crate) unsafe extern "C" fn cl_ht_insert(
    ctx: *mut CraneliftHashTableContext,
    key: *const u8,
    key_len: u32,
    val: *const u8,
    val_len: u32,
) {
    let Some(ctx) = read_ctx_mut::<CraneliftHashTableContext>(ctx) else {
        return;
    };
    let key_slice = std::slice::from_raw_parts(key, key_len as usize);
    let val_slice = std::slice::from_raw_parts(val, val_len as usize);
    if let Some(table) = ctx.tables.get_mut(&0) {
        if let Some(existing) = table.get_mut(key_slice) {
            if existing.len() == val_len as usize {
                existing.copy_from_slice(val_slice);
            } else {
                *existing = val_slice.to_vec();
            }
        } else {
            table.insert(key_slice.to_vec(), val_slice.to_vec());
        }
    }
}

pub(crate) unsafe extern "C" fn cl_ht_count(ctx: *const CraneliftHashTableContext) -> u32 {
    let Some(ctx) = read_ctx_ref::<CraneliftHashTableContext>(ctx) else {
        return 0;
    };
    ctx.tables.get(&0).map(|t| t.len() as u32).unwrap_or(0)
}

pub(crate) unsafe extern "C" fn cl_ht_get_entry(
    ctx: *const CraneliftHashTableContext,
    index: u32,
    key_out: *mut u8,
    val_out: *mut u8,
) -> i32 {
    let Some(ctx) = read_ctx_ref::<CraneliftHashTableContext>(ctx) else {
        return -1;
    };
    if let Some(table) = ctx.tables.get(&0) {
        if let Some((key, val)) = table.iter().nth(index as usize) {
            std::ptr::copy_nonoverlapping(key.as_ptr(), key_out, key.len());
            std::ptr::copy_nonoverlapping(val.as_ptr(), val_out, val.len());
            return key.len() as i32;
        }
    }
    -1
}

pub(crate) unsafe extern "C" fn cl_ht_increment(
    ctx: *mut CraneliftHashTableContext,
    key: *const u8,
    key_len: u32,
    addend: i64,
) -> i64 {
    let Some(ctx) = read_ctx_mut::<CraneliftHashTableContext>(ctx) else {
        return addend;
    };
    let key_slice = std::slice::from_raw_parts(key, key_len as usize);
    if let Some(table) = ctx.tables.get_mut(&0) {
        if let Some(existing) = table.get_mut(key_slice) {
            let current = i64::from_le_bytes(existing[..8].try_into().unwrap_or([0; 8]));
            let new_val = current + addend;
            existing[..8].copy_from_slice(&new_val.to_le_bytes());
            return new_val;
        }
        table.insert(key_slice.to_vec(), addend.to_le_bytes().to_vec());
    }
    addend
}
