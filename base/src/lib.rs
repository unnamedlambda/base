use std::{pin::Pin, sync::Arc, time::Duration};
use std::sync::atomic::Ordering;
use tracing::{debug, info, info_span};
pub use base_types::{Action, Algorithm, BaseConfig, Kind, OutputBatchSchema, OutputColumn, OutputType};
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
    CraneliftHashTableContext,
    Mailbox, SharedMemory, THREAD_COMPILED_FNS,
};

#[derive(Debug)]
pub enum Error {
    InvalidConfig(String),
    RuntimeCreation(std::io::Error),
    Execution(String),
}

pub struct Base {
    memory: Pin<Box<[u8]>>,
    mem_ptr: *mut u8,
    shared: Arc<SharedMemory>,
    clif_fns: Option<Arc<Vec<unsafe extern "C" fn(*mut u8)>>>,
    _module: Option<cranelift_jit::JITModule>,
    ht_ctx_ptr: Option<*mut CraneliftHashTableContext>,
    context_offset: usize,
}

impl Base {
    pub fn new(config: BaseConfig) -> Result<Self, Error> {
        let memory = vec![0u8; config.memory_size].into_boxed_slice();
        Self::from_memory(config, memory)
    }

    fn from_memory(config: BaseConfig, memory: Box<[u8]>) -> Result<Self, Error> {
        let _span = info_span!("base_new", memory_size = memory.len()).entered();
        info!("creating Base instance");

        let context_offset = config.context_offset;
        let mut memory = Pin::new(memory);
        let mem_ptr = memory.as_mut().as_mut_ptr();
        let shared = Arc::new(SharedMemory::new(mem_ptr));

        let (module, clif_fns) = if !config.cranelift_ir.is_empty() {
            let (module, fns) = compile_cranelift_ir(&config.cranelift_ir);
            (Some(module), Some(fns))
        } else {
            (None, None)
        };

        // Set thread-local compiled fns so FFI functions (cl_thread_init etc.) work on interpreter thread
        if let Some(ref fns) = clif_fns {
            THREAD_COMPILED_FNS.with(|cell| {
                *cell.borrow_mut() = Some(fns.clone());
            });
        }

        // Init HT context at context_offset (in the persistent region, safe from payload writes)
        let ht_ctx_ptr = if clif_fns.is_some() {
            Some(init_ht_context(&shared, context_offset))
        } else {
            None
        };

        info!("Base instance created");
        Ok(Base { memory, mem_ptr, shared, clif_fns, _module: module, ht_ctx_ptr, context_offset })
    }

    pub fn execute(&mut self, algorithm: Algorithm) -> Result<Vec<RecordBatch>, Error> {
        let _span = info_span!("execute",
            cranelift_units = algorithm.cranelift_units,
            actions_count = algorithm.actions.len(),
        ).entered();
        info!("starting execution");

        let Algorithm { actions, payloads, cranelift_units, timeout_ms, output: output_schemas } = algorithm;

        // Copy payloads into shared memory (payload region: 0..context_offset)
        if !payloads.is_empty() {
            let copy_len = payloads.len().min(self.context_offset).min(self.memory.len());
            self.memory[..copy_len].copy_from_slice(&payloads[..copy_len]);
        }

        let cranelift_mailboxes: Vec<_> = (0..cranelift_units)
            .map(|_| Arc::new(Mailbox::new()))
            .collect();

        let mut thread_handles = Vec::new();
        let actions_arc = Arc::new(actions);

        for (cl_id, mailbox) in cranelift_mailboxes.iter().enumerate() {
            if let Some(ref compiled_fns) = self.clif_fns {
                info!(cl_id, "spawning Cranelift unit thread");
                let mailbox = mailbox.clone();
                let actions = actions_arc.clone();
                let shared = self.shared.clone();
                let compiled_fns = compiled_fns.clone();

                thread_handles.push(std::thread::spawn(move || {
                    cranelift_unit_task_mailbox(mailbox, actions, shared, compiled_fns);
                }));
            }
        }

        let actions = &*actions_arc;
        let mem_ptr = self.mem_ptr;
        let shared = &self.shared;
        let clif_fns = &self.clif_fns;

        let mut pc: usize = 0;
        let timeout_start = std::time::Instant::now();
        let timeout_duration = timeout_ms.map(Duration::from_millis);

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
                    if let Some(ref fns) = clif_fns {
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
                    let expected = unsafe { load_sized(shared, action.src as usize, size, order) };
                    debug!(pc, dst = action.dst, expected, invert, ?order, size, "wait_until_begin");
                    loop {
                        let current = unsafe { load_sized(shared, action.dst as usize, size, order) };
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

        let batches = build_record_batches(&self.memory, &output_schemas);
        info!("execution complete");
        Ok(batches)
    }
}

impl Drop for Base {
    fn drop(&mut self) {
        if let Some(ctx_ptr) = self.ht_ctx_ptr {
            unsafe { drop(Box::from_raw(ctx_ptr)) };
        }
    }
}

pub fn run(config: BaseConfig, algorithm: Algorithm) -> Result<Vec<RecordBatch>, Error> {
    // Zero-copy path: move payloads directly into memory (resize if needed),
    // avoiding a separate allocation + memcpy.
    let Algorithm { actions, payloads, cranelift_units, timeout_ms, output } = algorithm;
    let mut payloads = payloads;
    let needed = config.memory_size.max(payloads.len());
    payloads.resize(needed, 0);
    let mut base = Base::from_memory(config, payloads.into_boxed_slice())?;
    base.execute(Algorithm {
        actions,
        payloads: Vec::new(),
        cranelift_units,
        timeout_ms,
        output,
    })
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
                        let mut pos = col.data_offset;
                        for _ in 0..row_count {
                            let start = pos;
                            while pos < memory.len() && memory[pos] != 0 {
                                pos += 1;
                            }
                            let s = std::str::from_utf8(&memory[start..pos]).unwrap_or("");
                            strings.push(s.to_string());
                            pos += 1;
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
