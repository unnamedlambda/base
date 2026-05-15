use std::collections::HashMap;
use std::sync::Arc;

use super::{clear_ctx_slot, read_ctx_mut, read_ctx_ref, write_ctx_slot};
use crate::jit::THREAD_COMPILED_FNS;

pub(crate) struct CraneliftThreadContext {
    threads: HashMap<u32, std::thread::JoinHandle<()>>,
    next_handle: u32,
    compiled_fns: Arc<Vec<unsafe extern "C" fn(*mut u8)>>,
}

pub(crate) unsafe extern "C" fn cl_thread_init(ctx_slot_ptr: *mut *mut CraneliftThreadContext) {
    let compiled_fns = THREAD_COMPILED_FNS.with(|cell| {
        cell.borrow()
            .clone()
            .expect("cl_thread_init: no compiled functions available")
    });
    let ctx = Box::new(CraneliftThreadContext {
        threads: HashMap::new(),
        next_handle: 1,
        compiled_fns,
    });
    let raw = Box::into_raw(ctx);
    if !write_ctx_slot(ctx_slot_ptr, raw) {
        drop(Box::from_raw(raw));
    }
}

pub(crate) unsafe extern "C" fn cl_thread_spawn(
    ctx_ptr: *mut CraneliftThreadContext,
    fn_index: i64,
    thread_ptr: *mut u8,
) -> i64 {
    let Some(ctx) = read_ctx_mut::<CraneliftThreadContext>(ctx_ptr) else {
        return -1;
    };
    let idx = fn_index as usize;
    if idx >= ctx.compiled_fns.len() {
        return -1;
    }
    let func = ctx.compiled_fns[idx];
    let thread_arg = thread_ptr as usize;
    let handle_id = ctx.next_handle;
    ctx.next_handle += 1;

    let compiled_fns_clone = ctx.compiled_fns.clone();
    let join = std::thread::spawn(move || {
        THREAD_COMPILED_FNS.with(|cell| {
            *cell.borrow_mut() = Some(compiled_fns_clone);
        });
        func(thread_arg as *mut u8);
    });

    ctx.threads.insert(handle_id, join);
    handle_id as i64
}

pub(crate) unsafe extern "C" fn cl_thread_join(
    ctx_ptr: *mut CraneliftThreadContext,
    handle: i64,
) -> i64 {
    let Some(ctx) = read_ctx_mut::<CraneliftThreadContext>(ctx_ptr) else {
        return -1;
    };
    if let Some(join) = ctx.threads.remove(&(handle as u32)) {
        match join.join() {
            Ok(_) => 0,
            Err(_) => -1,
        }
    } else {
        -1
    }
}

pub(crate) unsafe extern "C" fn cl_thread_cleanup(ctx_slot_ptr: *mut *mut CraneliftThreadContext) {
    let ctx_ptr = clear_ctx_slot::<CraneliftThreadContext>(ctx_slot_ptr);
    let mut ctx = Box::from_raw(ctx_ptr);
    for (_, join) in ctx.threads.drain() {
        let _ = join.join();
    }
}

pub(crate) unsafe extern "C" fn cl_thread_call(
    ctx_ptr: *const CraneliftThreadContext,
    fn_index: i64,
    arg_ptr: *mut u8,
) -> i64 {
    let Some(ctx) = read_ctx_ref::<CraneliftThreadContext>(ctx_ptr) else {
        return -1;
    };
    let idx = fn_index as usize;
    if idx >= ctx.compiled_fns.len() {
        return -1;
    }
    let func = ctx.compiled_fns[idx];
    func(arg_ptr);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn write_42(p: *mut u8) {
        *(p as *mut u64) = 42;
    }
    unsafe extern "C" fn write_99(p: *mut u8) {
        *(p as *mut u64) = 99;
    }
    unsafe extern "C" fn write_88(p: *mut u8) {
        *(p as *mut u64) = 88;
    }
    unsafe extern "C" fn slow_write_77(p: *mut u8) {
        std::thread::sleep(std::time::Duration::from_millis(20));
        *(p as *mut u64) = 77;
    }

    fn install_fns(fns: Vec<unsafe extern "C" fn(*mut u8)>) {
        THREAD_COMPILED_FNS.with(|cell| {
            *cell.borrow_mut() = Some(Arc::new(fns));
        });
    }

    #[test]
    fn init_then_cleanup_lifecycle() {
        install_fns(vec![write_42]);
        let mut slot: *mut CraneliftThreadContext = std::ptr::null_mut();
        unsafe {
            cl_thread_init(&mut slot);
            assert!(!slot.is_null());
            cl_thread_cleanup(&mut slot);
            assert!(slot.is_null());
        }
    }

    #[test]
    fn spawn_then_join_executes_fn() {
        install_fns(vec![write_42]);
        let mut slot: *mut CraneliftThreadContext = std::ptr::null_mut();
        let mut val: u64 = 0;
        unsafe {
            cl_thread_init(&mut slot);
            let h = cl_thread_spawn(slot, 0, &mut val as *mut u64 as *mut u8);
            assert!(h > 0);
            assert_eq!(cl_thread_join(slot, h), 0);
            cl_thread_cleanup(&mut slot);
        }
        assert_eq!(val, 42);
    }

    #[test]
    fn multiple_workers_run_in_parallel() {
        install_fns(vec![write_42, write_99, write_88]);
        let mut slot: *mut CraneliftThreadContext = std::ptr::null_mut();
        let mut v0: u64 = 0;
        let mut v1: u64 = 0;
        let mut v2: u64 = 0;
        unsafe {
            cl_thread_init(&mut slot);
            let h0 = cl_thread_spawn(slot, 0, &mut v0 as *mut u64 as *mut u8);
            let h1 = cl_thread_spawn(slot, 1, &mut v1 as *mut u64 as *mut u8);
            let h2 = cl_thread_spawn(slot, 2, &mut v2 as *mut u64 as *mut u8);
            assert!(h0 > 0 && h1 > 0 && h2 > 0);
            assert!(h0 != h1 && h1 != h2 && h0 != h2);
            assert_eq!(cl_thread_join(slot, h0), 0);
            assert_eq!(cl_thread_join(slot, h1), 0);
            assert_eq!(cl_thread_join(slot, h2), 0);
            cl_thread_cleanup(&mut slot);
        }
        assert_eq!(v0, 42);
        assert_eq!(v1, 99);
        assert_eq!(v2, 88);
    }

    #[test]
    fn join_invalid_handle_returns_neg1() {
        install_fns(vec![write_42]);
        let mut slot: *mut CraneliftThreadContext = std::ptr::null_mut();
        unsafe {
            cl_thread_init(&mut slot);
            assert_eq!(cl_thread_join(slot, 999), -1);
            cl_thread_cleanup(&mut slot);
        }
    }

    #[test]
    fn double_join_returns_neg1_second_time() {
        install_fns(vec![write_42]);
        let mut slot: *mut CraneliftThreadContext = std::ptr::null_mut();
        let mut val: u64 = 0;
        unsafe {
            cl_thread_init(&mut slot);
            let h = cl_thread_spawn(slot, 0, &mut val as *mut u64 as *mut u8);
            assert_eq!(cl_thread_join(slot, h), 0);
            assert_eq!(cl_thread_join(slot, h), -1);
            cl_thread_cleanup(&mut slot);
        }
    }

    #[test]
    fn spawn_oob_fn_index_returns_neg1() {
        install_fns(vec![write_42]);
        let mut slot: *mut CraneliftThreadContext = std::ptr::null_mut();
        let mut val: u64 = 0;
        unsafe {
            cl_thread_init(&mut slot);
            assert_eq!(
                cl_thread_spawn(slot, 5, &mut val as *mut u64 as *mut u8),
                -1
            );
            assert_eq!(
                cl_thread_spawn(slot, -1, &mut val as *mut u64 as *mut u8),
                -1
            );
            cl_thread_cleanup(&mut slot);
        }
    }

    #[test]
    fn cleanup_joins_unjoined_threads() {
        install_fns(vec![slow_write_77]);
        let mut slot: *mut CraneliftThreadContext = std::ptr::null_mut();
        let mut val: u64 = 0;
        unsafe {
            cl_thread_init(&mut slot);
            let h = cl_thread_spawn(slot, 0, &mut val as *mut u64 as *mut u8);
            assert!(h > 0);
            // do NOT join — cleanup must wait for it.
            cl_thread_cleanup(&mut slot);
        }
        assert_eq!(val, 77, "cleanup should have waited for the worker");
    }

    #[test]
    fn call_runs_fn_inline_on_current_thread() {
        install_fns(vec![write_42]);
        let mut slot: *mut CraneliftThreadContext = std::ptr::null_mut();
        let mut val: u64 = 0;
        unsafe {
            cl_thread_init(&mut slot);
            let rc = cl_thread_call(slot, 0, &mut val as *mut u64 as *mut u8);
            assert_eq!(rc, 0);
            cl_thread_cleanup(&mut slot);
        }
        assert_eq!(val, 42);
    }

    #[test]
    fn call_oob_fn_index_returns_neg1() {
        install_fns(vec![write_42]);
        let mut slot: *mut CraneliftThreadContext = std::ptr::null_mut();
        let mut val: u64 = 0;
        unsafe {
            cl_thread_init(&mut slot);
            assert_eq!(
                cl_thread_call(slot, 5, &mut val as *mut u64 as *mut u8),
                -1
            );
            cl_thread_cleanup(&mut slot);
        }
    }

    #[test]
    fn null_ctx_pointers_return_neg1() {
        let null_ctx = std::ptr::null_mut::<CraneliftThreadContext>();
        let mut val: u64 = 0;
        unsafe {
            assert_eq!(
                cl_thread_spawn(null_ctx, 0, &mut val as *mut u64 as *mut u8),
                -1
            );
            assert_eq!(cl_thread_join(null_ctx, 1), -1);
            assert_eq!(
                cl_thread_call(
                    null_ctx as *const _,
                    0,
                    &mut val as *mut u64 as *mut u8
                ),
                -1
            );
        }
    }
}
