use std::{pin::Pin, sync::Arc, time::Duration};
use std::sync::atomic::Ordering;
use tracing::{debug, info, info_span};
pub use base_types::{Action, Algorithm, Kind, State, UnitSpec};

mod units;

use crate::units::{
    compile_cranelift_ir,
    cranelift_unit_task_mailbox,
    init_ht_context,
    load_sized,
    order_from_u32,
    read_null_terminated_string_from_slice,
    Mailbox, SharedMemory, THREAD_COMPILED_FNS,
};

#[derive(Debug)]
pub enum Error {
    InvalidConfig(String),
    RuntimeCreation(std::io::Error),
    Execution(String),
}

pub fn execute(mut algorithm: Algorithm) -> Result<(), Error> {
    let _span = info_span!("execute",
        cranelift_units = algorithm.units.cranelift_units,
        actions_count = algorithm.actions.len(),
    ).entered();

    info!("starting execution engine");

    if algorithm.cranelift_assignments.is_empty() {
        algorithm.cranelift_assignments = vec![255; algorithm.actions.len()];
    }

    let result = execute_internal(algorithm);
    info!("execution complete");
    result
}

fn execute_internal(algorithm: Algorithm) -> Result<(), Error> {
    let _span = info_span!("execute_internal").entered();
    let mut payloads = algorithm.payloads;
    if algorithm.additional_shared_memory > 0 {
        payloads.resize(payloads.len() + algorithm.additional_shared_memory, 0);
    }
    let mut memory = Pin::new(payloads.into_boxed_slice());
    let mem_ptr = memory.as_mut().as_mut_ptr();
    let shared = Arc::new(SharedMemory::new(mem_ptr));
    let actions = algorithm.actions;

    let cranelift_assignments = Arc::new(algorithm.cranelift_assignments);

    let cranelift_mailboxes: Vec<_> = (0..algorithm.units.cranelift_units)
        .map(|_| Arc::new(Mailbox::new()))
        .collect();

    let mut thread_handles = Vec::new();

    let mut clif_compiled: std::collections::HashMap<usize, Arc<Vec<unsafe extern "C" fn(*mut u8)>>> = std::collections::HashMap::new();
    for (idx, &offset) in algorithm.state.cranelift_ir_offsets.iter().enumerate() {
        if !clif_compiled.contains_key(&idx) {
            let source = read_null_terminated_string_from_slice(&memory, offset, 64 * 1024);
            if !source.is_empty() {
                let compiled = compile_cranelift_ir(&source);
                clif_compiled.insert(idx, compiled);
            }
        }
    }

    // Keep a reference to unit 0's compiled fns for synchronous ClifCall
    let clif_fns_local = clif_compiled.get(&0).cloned();

    // Set thread-local compiled fns so FFI functions (cl_thread_init etc.) work on interpreter thread
    if let Some(ref fns) = clif_fns_local {
        THREAD_COMPILED_FNS.with(|cell| {
            *cell.borrow_mut() = Some(fns.clone());
        });
    }

    // Init HT context before any CLIF code runs (shared by ClifCall and worker threads)
    let ht_ctx_ptr = if !clif_compiled.is_empty() {
        Some(init_ht_context(&shared))
    } else {
        None
    };

    let actions_arc = Arc::new(actions);

    for (cl_id, mailbox) in cranelift_mailboxes.iter().enumerate() {
        if let Some(compiled_fns) = clif_compiled.get(&cl_id).cloned() {
            info!(cl_id, "spawning Cranelift unit thread");
            let mailbox = mailbox.clone();
            let actions = actions_arc.clone();
            let shared = shared.clone();

            thread_handles.push(std::thread::spawn(move || {
                cranelift_unit_task_mailbox(mailbox, actions, shared, compiled_fns);
            }));
        }
    }

    let actions = &*actions_arc;

    let mut pc: usize = 0;
    let timeout_start = std::time::Instant::now();
    let timeout_duration = algorithm.timeout_ms.map(Duration::from_millis);

    while pc < actions.len() {
        if let Some(timeout) = timeout_duration {
            if timeout_start.elapsed() > timeout {
                return Err(Error::Execution("Timeout".into()));
            }
        }

        let action = &actions[pc];

        match action.kind {
            Kind::ClifCall => {
                let fn_idx = action.src as usize;
                debug!(pc, fn_idx, "clif_call");
                if let Some(ref fns) = clif_fns_local {
                    let f = fns[fn_idx % fns.len()];
                    unsafe { f(mem_ptr) };
                }
                pc += 1;
            }

            Kind::ConditionalJump => {
                let check_size = if action.size == 0 { 8 } else { action.size as usize };
                let cond_bytes =
                    unsafe { shared.read(action.src as usize + action.offset as usize, check_size) };
                let cond_nonzero = cond_bytes.iter().take(check_size).any(|&b| b != 0);
                debug!(pc, cond_nonzero, target = action.dst, "conditional_jump");

                if cond_nonzero {
                    pc = action.dst as usize;
                } else {
                    pc += 1;
                }
            }

            Kind::ClifCallAsync => {
                let desc_start = action.src;
                let count = action.size;
                let flag = action.offset;

                unsafe {
                    shared.store_u64(flag as usize, 0, Ordering::Release);
                }

                debug!(pc, desc_start, count, flag, "clif_call_async");

                if !cranelift_mailboxes.is_empty() {
                    let assigned = cranelift_assignments
                        .get(pc)
                        .copied()
                        .unwrap_or(0);

                    let unit_id = if assigned == 255 {
                        0
                    } else {
                        (assigned as usize).min(cranelift_mailboxes.len() - 1)
                    };

                    cranelift_mailboxes[unit_id].post(desc_start, count, flag);
                }

                pc += 1;
            }

            Kind::Wait => {
                debug!(pc, flag_addr = action.dst, "wait_begin");
                loop {
                    let flag = unsafe { shared.load_u64(action.dst as usize, Ordering::Acquire) };
                    if flag != 0 {
                        break;
                    }
                    std::thread::yield_now();
                }
                debug!(pc, "wait_complete");
                pc += 1;
            }

            Kind::WaitUntil => {
                let invert = (action.offset & 1) != 0;
                let order = order_from_u32((action.offset >> 1) & 0x7);
                let size = if action.size == 0 { 8 } else { action.size };
                let expected = unsafe { load_sized(&shared, action.src as usize, size, order) };
                debug!(pc, dst = action.dst, expected, invert, ?order, size, "wait_until_begin");
                loop {
                    let current = unsafe { load_sized(&shared, action.dst as usize, size, order) };
                    let equal = current == expected;
                    if equal != invert {
                        break;
                    }
                    if let Some(timeout) = timeout_duration {
                        if timeout_start.elapsed() > timeout {
                            return Err(Error::Execution("Timeout".into()));
                        }
                    }
                    std::thread::yield_now();
                }
                debug!(pc, "wait_until_complete");
                pc += 1;
            }

            Kind::Park => {
                let wake_addr = action.dst as usize;
                let expected = if action.src == 0 {
                    0u64
                } else {
                    unsafe { shared.load_u64(action.src as usize, Ordering::Acquire) }
                };
                let status_addr = action.offset as usize;
                let per_timeout_ms = action.size as u64;
                debug!(pc, wake_addr, expected, status_addr, per_timeout_ms, "park_begin");

                let park_start = std::time::Instant::now();
                let per_timeout = if per_timeout_ms > 0 {
                    Some(Duration::from_millis(per_timeout_ms))
                } else {
                    None
                };

                let woken = loop {
                    let current = unsafe { shared.load_u64(wake_addr, Ordering::Acquire) };
                    if current != expected {
                        break true;
                    }
                    if let Some(pt) = per_timeout {
                        if park_start.elapsed() > pt {
                            break false;
                        }
                    }
                    if let Some(timeout) = timeout_duration {
                        if timeout_start.elapsed() > timeout {
                            return Err(Error::Execution("Timeout".into()));
                        }
                    }
                    std::thread::sleep(Duration::from_micros(50));
                };

                if status_addr != 0 {
                    let status_val: u64 = if woken { 1 } else { 0 };
                    unsafe { shared.store_u64(status_addr, status_val, Ordering::Release) };
                }
                debug!(pc, woken, "park_complete");
                pc += 1;
            }

            Kind::Wake => {
                let wake_addr = action.dst as usize;
                let delta = if action.src == 0 {
                    1u64
                } else {
                    unsafe { shared.load_u64(action.src as usize, Ordering::Acquire) }
                };
                debug!(pc, wake_addr, delta, "wake");

                loop {
                    let current = unsafe { shared.load_u64(wake_addr, Ordering::Acquire) };
                    let new_val = current.wrapping_add(delta);
                    let result = unsafe { shared.cas64(wake_addr, current, new_val) };
                    if result == current {
                        if action.offset != 0 {
                            unsafe { shared.store_u64(action.offset as usize, new_val, Ordering::Release) };
                        }
                        break;
                    }
                    std::hint::spin_loop();
                }
                pc += 1;
            }

            _ => {
                pc += 1;
            }
        }
    }

    info!("shutting down all units");
    for mailbox in cranelift_mailboxes.iter() {
        mailbox.shutdown();
    }

    for handle in thread_handles {
        let _ = handle.join();
    }
    info!("all unit threads joined");

    if let Some(ctx_ptr) = ht_ctx_ptr {
        unsafe { drop(Box::from_raw(ctx_ptr)) };
    }

    Ok(())
}
