use std::{pin::Pin, sync::Arc, time::Duration};
use std::sync::atomic::Ordering;
use tracing::{debug, info, info_span};
pub use base_types::{Action, Algorithm, Kind, OutputBatchSchema, OutputColumn, OutputType, State, UnitSpec};
pub use arrow_array::RecordBatch;
use arrow_array::{ArrayRef, Int64Array, Float64Array, StringArray};
use arrow_schema::{Schema, Field, DataType};

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

pub fn execute(algorithm: Algorithm) -> Result<Vec<RecordBatch>, Error> {
    let _span = info_span!("execute",
        cranelift_units = algorithm.units.cranelift_units,
        actions_count = algorithm.actions.len(),
    ).entered();

    info!("starting execution engine");

    let result = execute_internal(algorithm);
    info!("execution complete");
    result
}

fn execute_internal(algorithm: Algorithm) -> Result<Vec<RecordBatch>, Error> {
    let _span = info_span!("execute_internal").entered();
    let output_schemas = algorithm.output;
    let mut payloads = algorithm.payloads;
    if algorithm.additional_shared_memory > 0 {
        payloads.resize(payloads.len() + algorithm.additional_shared_memory, 0);
    }
    let mut memory = Pin::new(payloads.into_boxed_slice());
    let mem_ptr = memory.as_mut().as_mut_ptr();
    let shared = Arc::new(SharedMemory::new(mem_ptr));
    let actions = algorithm.actions;

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
                let unit_id = (action.dst as usize).min(
                    cranelift_mailboxes.len().saturating_sub(1),
                );

                unsafe {
                    shared.store_u64(flag as usize, 0, Ordering::Release);
                }

                debug!(pc, desc_start, count, flag, unit_id, "clif_call_async");

                if !cranelift_mailboxes.is_empty() {
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

    let batches = build_record_batches(&memory, &output_schemas);
    Ok(batches)
}

fn build_record_batches(memory: &[u8], schemas: &[OutputBatchSchema]) -> Vec<RecordBatch> {
    let mut batches = Vec::with_capacity(schemas.len());
    for schema in schemas {
        let row_count = if schema.row_count_offset + 8 <= memory.len() {
            let bytes: [u8; 8] = memory[schema.row_count_offset..schema.row_count_offset + 8]
                .try_into().unwrap();
            u64::from_le_bytes(bytes) as usize
        } else {
            0
        };
        if row_count == 0 {
            continue;
        }

        let mut fields = Vec::with_capacity(schema.columns.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.columns.len());

        for col in &schema.columns {
            match col.dtype {
                OutputType::I64 => {
                    fields.push(Field::new(&col.name, DataType::Int64, false));
                    let mut values = Vec::with_capacity(row_count);
                    for i in 0..row_count {
                        let off = col.data_offset + i * 8;
                        if off + 8 <= memory.len() {
                            let bytes: [u8; 8] = memory[off..off + 8].try_into().unwrap();
                            values.push(i64::from_le_bytes(bytes));
                        } else {
                            values.push(0);
                        }
                    }
                    arrays.push(Arc::new(Int64Array::from(values)) as ArrayRef);
                }
                OutputType::F64 => {
                    fields.push(Field::new(&col.name, DataType::Float64, false));
                    let mut values = Vec::with_capacity(row_count);
                    for i in 0..row_count {
                        let off = col.data_offset + i * 8;
                        if off + 8 <= memory.len() {
                            let bytes: [u8; 8] = memory[off..off + 8].try_into().unwrap();
                            values.push(f64::from_le_bytes(bytes));
                        } else {
                            values.push(0.0);
                        }
                    }
                    arrays.push(Arc::new(Float64Array::from(values)) as ArrayRef);
                }
                OutputType::Utf8 => {
                    fields.push(Field::new(&col.name, DataType::Utf8, false));
                    let mut strings = Vec::with_capacity(row_count);
                    // For Utf8: len_offset holds the byte length of each string.
                    // data_offset points to contiguous null-terminated strings.
                    // For single-row: read len_offset for byte count, slice data_offset..+len
                    // For multi-row: strings are packed sequentially, each null-terminated
                    let total_byte_len = if col.len_offset + 8 <= memory.len() {
                        let bytes: [u8; 8] = memory[col.len_offset..col.len_offset + 8]
                            .try_into().unwrap();
                        u64::from_le_bytes(bytes) as usize
                    } else {
                        0
                    };
                    if row_count == 1 {
                        let end = (col.data_offset + total_byte_len).min(memory.len());
                        let slice = &memory[col.data_offset..end];
                        let s = std::str::from_utf8(slice).unwrap_or("");
                        strings.push(s.to_string());
                    } else {
                        // Multi-row: null-terminated strings packed sequentially
                        let mut pos = col.data_offset;
                        for _ in 0..row_count {
                            let start = pos;
                            while pos < memory.len() && memory[pos] != 0 {
                                pos += 1;
                            }
                            let s = std::str::from_utf8(&memory[start..pos]).unwrap_or("");
                            strings.push(s.to_string());
                            pos += 1; // skip null terminator
                        }
                    }
                    arrays.push(Arc::new(StringArray::from(strings)) as ArrayRef);
                }
            }
        }

        if let Ok(batch) = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays) {
            batches.push(batch);
        }
    }
    batches
}
