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
