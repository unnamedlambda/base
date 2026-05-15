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

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn init() -> *mut CraneliftHashTableContext {
        let mut slot: *mut CraneliftHashTableContext = std::ptr::null_mut();
        cl_ht_init(&mut slot);
        assert!(!slot.is_null());
        slot
    }

    unsafe fn cleanup(ctx: *mut CraneliftHashTableContext) {
        let mut slot = ctx;
        cl_ht_cleanup(&mut slot);
        assert!(slot.is_null());
    }

    unsafe fn insert(ctx: *mut CraneliftHashTableContext, key: &[u8], val: &[u8]) {
        cl_ht_insert(
            ctx,
            key.as_ptr(),
            key.len() as u32,
            val.as_ptr(),
            val.len() as u32,
        );
    }

    unsafe fn lookup(ctx: *mut CraneliftHashTableContext, key: &[u8]) -> Option<Vec<u8>> {
        let mut out = vec![0u8; 256];
        let n = cl_ht_lookup(ctx, key.as_ptr(), key.len() as u32, out.as_mut_ptr());
        if n == 0xFFFF_FFFF {
            None
        } else {
            out.truncate(n as usize);
            Some(out)
        }
    }

    #[test]
    fn init_then_cleanup_lifecycle() {
        unsafe {
            let ctx = init();
            cleanup(ctx);
        }
    }

    #[test]
    fn create_returns_sequential_handles() {
        unsafe {
            let ctx = init();
            assert_eq!(cl_ht_create(ctx), 0);
            assert_eq!(cl_ht_create(ctx), 1);
            assert_eq!(cl_ht_create(ctx), 2);
            cleanup(ctx);
        }
    }

    #[test]
    fn insert_then_lookup_roundtrip() {
        unsafe {
            let ctx = init();
            // Other accessors hardcode table key 0; cl_ht_create(ctx) returns 0 first time.
            assert_eq!(cl_ht_create(ctx), 0);
            insert(ctx, b"foo", b"hello");
            insert(ctx, b"bar", b"world!!");
            assert_eq!(lookup(ctx, b"foo").as_deref(), Some(&b"hello"[..]));
            assert_eq!(lookup(ctx, b"bar").as_deref(), Some(&b"world!!"[..]));
            cleanup(ctx);
        }
    }

    #[test]
    fn lookup_missing_key_returns_sentinel() {
        unsafe {
            let ctx = init();
            cl_ht_create(ctx);
            assert!(lookup(ctx, b"absent").is_none());
            cleanup(ctx);
        }
    }

    #[test]
    fn lookup_before_create_returns_sentinel() {
        // No table created -> lookup against table 0 misses.
        unsafe {
            let ctx = init();
            assert!(lookup(ctx, b"k").is_none());
            cleanup(ctx);
        }
    }

    #[test]
    fn insert_overwrites_same_length() {
        unsafe {
            let ctx = init();
            cl_ht_create(ctx);
            insert(ctx, b"k", b"AAAA");
            insert(ctx, b"k", b"BBBB");
            assert_eq!(lookup(ctx, b"k").as_deref(), Some(&b"BBBB"[..]));
            cleanup(ctx);
        }
    }

    #[test]
    fn insert_overwrites_different_length() {
        unsafe {
            let ctx = init();
            cl_ht_create(ctx);
            insert(ctx, b"k", b"short");
            insert(ctx, b"k", b"much-longer-value");
            assert_eq!(
                lookup(ctx, b"k").as_deref(),
                Some(&b"much-longer-value"[..])
            );
            // Then shrink.
            insert(ctx, b"k", b"x");
            assert_eq!(lookup(ctx, b"k").as_deref(), Some(&b"x"[..]));
            cleanup(ctx);
        }
    }

    #[test]
    fn count_reflects_inserts() {
        unsafe {
            let ctx = init();
            cl_ht_create(ctx);
            assert_eq!(cl_ht_count(ctx), 0);
            insert(ctx, b"a", b"1");
            insert(ctx, b"b", b"2");
            insert(ctx, b"c", b"3");
            assert_eq!(cl_ht_count(ctx), 3);
            // Overwrite doesn't bump count.
            insert(ctx, b"a", b"9");
            assert_eq!(cl_ht_count(ctx), 3);
            cleanup(ctx);
        }
    }

    #[test]
    fn count_with_no_table_is_zero() {
        unsafe {
            let ctx = init();
            assert_eq!(cl_ht_count(ctx), 0);
            cleanup(ctx);
        }
    }

    #[test]
    fn get_entry_iterates_all_pairs() {
        unsafe {
            let ctx = init();
            cl_ht_create(ctx);
            insert(ctx, b"k1", b"v1aa");
            insert(ctx, b"k2", b"v2bb");
            insert(ctx, b"k3", b"v3cc");

            let n = cl_ht_count(ctx) as usize;
            assert_eq!(n, 3);
            let mut seen: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            for i in 0..n {
                let mut k = vec![0u8; 16];
                let mut v = vec![0u8; 16];
                let klen = cl_ht_get_entry(ctx, i as u32, k.as_mut_ptr(), v.as_mut_ptr());
                assert!(klen > 0);
                k.truncate(klen as usize);
                // Values are 4 bytes here, but get_entry returns key len only;
                // we know each value length from the insert call.
                v.truncate(4);
                seen.push((k, v));
            }
            seen.sort();
            assert_eq!(
                seen,
                vec![
                    (b"k1".to_vec(), b"v1aa".to_vec()),
                    (b"k2".to_vec(), b"v2bb".to_vec()),
                    (b"k3".to_vec(), b"v3cc".to_vec()),
                ]
            );
            cleanup(ctx);
        }
    }

    #[test]
    fn get_entry_out_of_range_returns_neg1() {
        unsafe {
            let ctx = init();
            cl_ht_create(ctx);
            insert(ctx, b"only", b"x");
            let mut k = [0u8; 16];
            let mut v = [0u8; 16];
            assert_eq!(cl_ht_get_entry(ctx, 5, k.as_mut_ptr(), v.as_mut_ptr()), -1);
            cleanup(ctx);
        }
    }

    #[test]
    fn get_entry_with_no_table_returns_neg1() {
        unsafe {
            let ctx = init();
            let mut k = [0u8; 16];
            let mut v = [0u8; 16];
            assert_eq!(cl_ht_get_entry(ctx, 0, k.as_mut_ptr(), v.as_mut_ptr()), -1);
            cleanup(ctx);
        }
    }

    #[test]
    fn increment_creates_then_accumulates() {
        unsafe {
            let ctx = init();
            cl_ht_create(ctx);
            // First call: key absent -> inserts addend as the new value.
            assert_eq!(cl_ht_increment(ctx, b"c".as_ptr(), 1, 10), 10);
            assert_eq!(cl_ht_increment(ctx, b"c".as_ptr(), 1, 5), 15);
            assert_eq!(cl_ht_increment(ctx, b"c".as_ptr(), 1, -7), 8);

            // Stored value should be the 8-byte LE encoding of the latest sum.
            let stored = lookup(ctx, b"c").unwrap();
            assert_eq!(stored.len(), 8);
            assert_eq!(i64::from_le_bytes(stored.try_into().unwrap()), 8);
            cleanup(ctx);
        }
    }

    #[test]
    fn increment_without_table_returns_addend() {
        // No cl_ht_create called -> no table 0 -> increment falls through to addend.
        unsafe {
            let ctx = init();
            assert_eq!(cl_ht_increment(ctx, b"c".as_ptr(), 1, 42), 42);
            // Nothing stored.
            assert!(lookup(ctx, b"c").is_none());
            cleanup(ctx);
        }
    }

    #[test]
    fn null_ctx_returns_sentinels() {
        let null_ctx = std::ptr::null_mut::<CraneliftHashTableContext>();
        unsafe {
            assert_eq!(cl_ht_create(null_ctx), u32::MAX);
            let key = [0u8; 1];
            let mut out = [0u8; 8];
            assert_eq!(
                cl_ht_lookup(null_ctx as *const _, key.as_ptr(), 1, out.as_mut_ptr()),
                0xFFFF_FFFF
            );
            // Insert on null ctx is a silent no-op (returns ()).
            cl_ht_insert(null_ctx, key.as_ptr(), 1, key.as_ptr(), 1);
            assert_eq!(cl_ht_count(null_ctx as *const _), 0);
            let mut k = [0u8; 1];
            let mut v = [0u8; 1];
            assert_eq!(
                cl_ht_get_entry(null_ctx as *const _, 0, k.as_mut_ptr(), v.as_mut_ptr()),
                -1
            );
            // increment returns addend unchanged when ctx is null.
            assert_eq!(cl_ht_increment(null_ctx, key.as_ptr(), 1, 99), 99);
        }
    }

    #[test]
    fn cleanup_on_null_slot_is_noop() {
        let mut null_slot: *mut CraneliftHashTableContext = std::ptr::null_mut();
        unsafe { cl_ht_cleanup(&mut null_slot) };
        assert!(null_slot.is_null());
    }
}
