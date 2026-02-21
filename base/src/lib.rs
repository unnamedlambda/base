use std::{pin::Pin, sync::Arc, time::Duration};
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, info, info_span};
use wgpu::Backends;

pub use base_types::{Action, Algorithm, Kind, State, UnitSpec};

mod units;

use crate::units::{
    cranelift_unit_task_mailbox, ffi_unit_task_mailbox,
    file_unit_task_mailbox, gpu_unit_task_mailbox, hash_table_unit_task_mailbox,
    load_sized, memory_unit_task_mailbox,
    order_from_u32, queue_try_push_packet_mp,
    read_null_terminated_string_from_slice, simd_unit_task_mailbox,
    Broadcast, Mailbox, SharedMemory,
    QUEUE_BASE_OFF, QUEUE_HEAD_OFF, QUEUE_MASK_OFF, QUEUE_TAIL_OFF,
};

#[derive(Debug)]
pub enum Error {
    InvalidConfig(String),
    RuntimeCreation(std::io::Error),
    Execution(String),
    GpuInit(String),
}

const KERNEL_DESC_QUEUE_OFF: usize = 0;
const KERNEL_DESC_UNIT_TYPE_OFF: usize = 4;
const KERNEL_DESC_UNIT_ID_OFF: usize = 8;
const KERNEL_DESC_STOP_FLAG_OFF: usize = 12;
const KERNEL_DESC_PROGRESS_OFF: usize = 16;

const KERNEL_KIND_QUEUE_ROUTER: u32 = 1;

struct KernelHandle {
    stop: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
    progress_addr: u32,
    queue_desc: u32,
    pool: Option<Vec<Arc<Mailbox>>>,
}

enum DispatchTarget {
    Mailbox(Arc<Mailbox>),
    Broadcast(Arc<Broadcast>),
}

unsafe fn cas_u32(
    shared: &SharedMemory,
    offset: usize,
    current: u32,
    new: u32,
    success: Ordering,
    failure: Ordering,
) -> Result<u32, u32> {
    shared.cas_u32(offset, current, new, success, failure)
}

unsafe fn queue_desc_read(shared: &SharedMemory, queue_desc: usize) -> (u32, u32) {
    let mask = shared.load_u32(queue_desc + QUEUE_MASK_OFF, Ordering::Relaxed);
    let base = shared.load_u32(queue_desc + QUEUE_BASE_OFF, Ordering::Relaxed);
    (mask, base)
}

unsafe fn queue_try_pop_u64(shared: &SharedMemory, queue_desc: usize) -> Option<u64> {
    let (mask, base) = queue_desc_read(shared, queue_desc);
    loop {
        let head = shared.load_u32(queue_desc + QUEUE_HEAD_OFF, Ordering::Relaxed);
        let tail = shared.load_u32(queue_desc + QUEUE_TAIL_OFF, Ordering::Acquire);
        if head == tail {
            return None;
        }
        let new_head = head.wrapping_add(1);
        if cas_u32(
            shared,
            queue_desc + QUEUE_HEAD_OFF,
            head,
            new_head,
            Ordering::AcqRel,
            Ordering::Relaxed,
        )
        .is_err()
        {
            continue;
        }
        let slot = (head & mask) as usize;
        let slot_off = base as usize + slot * 8;
        let bytes = shared.read(slot_off, 8);
        return Some(u64::from_le_bytes(bytes[0..8].try_into().unwrap()));
    }
}


pub fn execute(mut algorithm: Algorithm) -> Result<(), Error> {
    let _span = info_span!("execute",
        simd_units = algorithm.units.simd_units,
        gpu_units = algorithm.units.gpu_units,
        memory_units = algorithm.units.memory_units,
        file_units = algorithm.units.file_units,
        ffi_units = algorithm.units.ffi_units,
        hash_table_units = algorithm.units.hash_table_units,
        cranelift_units = algorithm.units.cranelift_units,
        actions_count = algorithm.actions.len(),
    ).entered();

    info!("starting execution engine");

    if algorithm.simd_assignments.is_empty() {
        algorithm.simd_assignments = vec![255; algorithm.actions.len()];
        let mut unit = 0u8;
        for (i, action) in algorithm.actions.iter().enumerate() {
            match action.kind {
                Kind::SimdLoad
                | Kind::SimdAdd
                | Kind::SimdMul
                | Kind::SimdStore
                | Kind::SimdLoadI32
                | Kind::SimdAddI32
                | Kind::SimdMulI32
                | Kind::SimdStoreI32
                | Kind::SimdDivI32
                | Kind::SimdSubI32 => {
                    algorithm.simd_assignments[i] = unit;
                    unit = (unit + 1) % algorithm.units.simd_units as u8;
                }
                _ => {}
            }
        }
        debug!("auto-assigned SIMD actions across {} units", algorithm.units.simd_units);
    }

    if algorithm.memory_assignments.is_empty() {
        algorithm.memory_assignments = vec![255; algorithm.actions.len()];
        for (i, action) in algorithm.actions.iter().enumerate() {
            match action.kind {
                Kind::ConditionalWrite
                | Kind::MemCopy
                | Kind::MemScan
                | Kind::AtomicCAS
                | Kind::Fence
                | Kind::MemWrite
                | Kind::MemCopyIndirect
                | Kind::MemStoreIndirect
                | Kind::Compare
                | Kind::AtomicFetchAdd
                | Kind::QueuePushPacketMP
                 => {
                    algorithm.memory_assignments[i] = 0;
                }
                _ => {}
            }
        }
        debug!("auto-assigned memory actions across {} units", algorithm.units.memory_units);
    }

    if algorithm.ffi_assignments.is_empty() {
        algorithm.ffi_assignments = vec![255; algorithm.actions.len()];
        for (i, action) in algorithm.actions.iter().enumerate() {
            match action.kind {
                Kind::FFICall => {
                    algorithm.ffi_assignments[i] = 0;
                }
                _ => {}
            }
        }
        debug!("auto-assigned FFI actions across {} units", algorithm.units.ffi_units);
    }

    if algorithm.hash_table_assignments.is_empty() {
        algorithm.hash_table_assignments = vec![255; algorithm.actions.len()];
        for (i, action) in algorithm.actions.iter().enumerate() {
            match action.kind {
                Kind::HashTableCreate
                | Kind::HashTableInsert
                | Kind::HashTableLookup
                | Kind::HashTableDelete => {
                    algorithm.hash_table_assignments[i] = 0;
                }
                _ => {}
            }
        }
        debug!("auto-assigned hash table actions across {} units", algorithm.units.hash_table_units);
    }

    if algorithm.file_assignments.is_empty() {
        algorithm.file_assignments = vec![255; algorithm.actions.len()];
        let mut unit = 0u8;
        for (i, action) in algorithm.actions.iter().enumerate() {
            match action.kind {
                Kind::FileRead | Kind::FileWrite => {
                    algorithm.file_assignments[i] = unit;
                    unit = (unit + 1) % algorithm.units.file_units as u8;
                }
                _ => {}
            }
        }
        debug!("auto-assigned file actions across {} units", algorithm.units.file_units);
    }

    if algorithm.gpu_assignments.is_empty() {
        algorithm.gpu_assignments = vec![255; algorithm.actions.len()];
        let mut unit = 0u8;
        for (i, action) in algorithm.actions.iter().enumerate() {
            match action.kind {
                Kind::Dispatch => {
                    algorithm.gpu_assignments[i] = unit;
                    if algorithm.units.gpu_units > 1 {
                        unit = (unit + 1) % algorithm.units.gpu_units as u8;
                    }
                }
                _ => {}
            }
        }
        debug!("auto-assigned GPU actions across {} units", algorithm.units.gpu_units);
    }

    if algorithm.cranelift_assignments.is_empty() {
        algorithm.cranelift_assignments = vec![255; algorithm.actions.len()];
    }

    let result = execute_internal(algorithm);
    info!("execution complete");
    result
}

fn execute_internal(algorithm: Algorithm) -> Result<(), Error> {
    let _span = info_span!("execute_internal").entered();
    let mut memory = Pin::new(algorithm.payloads.into_boxed_slice());
    let mem_ptr = memory.as_mut().as_mut_ptr();
    let shared = Arc::new(SharedMemory::new(mem_ptr));
    let actions_arc = Arc::new(algorithm.actions);

    let gpu_assignments = Arc::new(algorithm.gpu_assignments);
    let simd_assignments = Arc::new(algorithm.simd_assignments);
    let file_assignments = Arc::new(algorithm.file_assignments);
    let ffi_assignments = Arc::new(algorithm.ffi_assignments);
    let memory_assignments = Arc::new(algorithm.memory_assignments);
    let hash_table_assignments = Arc::new(algorithm.hash_table_assignments);
    let cranelift_assignments = Arc::new(algorithm.cranelift_assignments);

    let gpu_mailboxes: Vec<Arc<Mailbox>> = (0..algorithm.units.gpu_units).map(|_| Arc::new(Mailbox::new())).collect();
    let simd_mailboxes: Vec<Arc<Mailbox>> = (0..algorithm.units.simd_units).map(|_| Arc::new(Mailbox::new())).collect();
    let file_mailboxes: Vec<Arc<Mailbox>> = (0..algorithm.units.file_units).map(|_| Arc::new(Mailbox::new())).collect();
    let ffi_mailboxes: Vec<Arc<Mailbox>> = (0..algorithm.units.ffi_units).map(|_| Arc::new(Mailbox::new())).collect();
    let memory_mailboxes: Vec<Arc<Mailbox>> = (0..algorithm.units.memory_units).map(|_| Arc::new(Mailbox::new())).collect();
    let hash_table_mailboxes: Vec<Arc<Mailbox>> = (0..algorithm.units.hash_table_units).map(|_| Arc::new(Mailbox::new())).collect();
    let cranelift_mailboxes: Vec<_> = (0..algorithm.units.cranelift_units)
        .map(|_| Arc::new(Mailbox::new()))
        .collect();
        
    let simd_broadcast = Arc::new(Broadcast::new(algorithm.units.simd_units as u32));
    let memory_broadcast = Arc::new(Broadcast::new(algorithm.units.memory_units as u32));

    let mut thread_handles = Vec::new();
    let mut kernel_handles: Vec<Option<KernelHandle>> = Vec::new();

    for (gpu_id, mailbox) in gpu_mailboxes.iter().enumerate() {
        if gpu_id < algorithm.state.gpu_shader_offsets.len() {
            info!(gpu_id, "spawning GPU unit thread");
            let gpu_size = algorithm.state.gpu_size;
            let backends = Backends::from_bits(algorithm.units.backends_bits).unwrap_or(Backends::all());
            let offset = algorithm.state.gpu_shader_offsets[gpu_id];
            let shader_source =
                read_null_terminated_string_from_slice(&memory, offset, 8192);
            let mailbox = mailbox.clone();
            let actions = actions_arc.clone();
            let shared = shared.clone();

            thread_handles.push(std::thread::spawn(move || {
                gpu_unit_task_mailbox(
                    mailbox,
                    actions,
                    shared,
                    shader_source,
                    gpu_size,
                    backends,
                );
            }));
        }
    }

    for (worker_id, mailbox) in simd_mailboxes.iter().cloned().enumerate() {
        info!(worker_id, "spawning SIMD unit thread");
        let actions = actions_arc.clone();
        let shared = shared.clone();
        let regs = algorithm.state.regs_per_unit;
        let broadcast = simd_broadcast.clone();
        thread_handles.push(std::thread::spawn(move || {
            simd_unit_task_mailbox(
                mailbox,
                broadcast,
                worker_id as u32,
                actions,
                shared,
                regs,
            );
        }));
    }

    for (file_id, mailbox) in file_mailboxes.iter().cloned().enumerate() {
        info!(file_id, "spawning file unit task");
        let actions = actions_arc.clone();
        let shared = shared.clone();
        let buffer_size = algorithm.state.file_buffer_size;
        thread_handles.push(std::thread::spawn(move || {
            file_unit_task_mailbox(mailbox, actions, shared, buffer_size);
        }));
    }

    for (ffi_id, mailbox) in ffi_mailboxes.iter().cloned().enumerate() {
        info!(ffi_id, "spawning FFI unit thread");
        let actions = actions_arc.clone();
        let shared = shared.clone();
        thread_handles.push(std::thread::spawn(move || {
            ffi_unit_task_mailbox(mailbox, actions, shared);
        }));
    }

    for (worker_id, mailbox) in memory_mailboxes.iter().cloned().enumerate() {
        info!(worker_id, "spawning memory unit thread");
        let actions = actions_arc.clone();
        let shared = shared.clone();
        let broadcast = memory_broadcast.clone();
        thread_handles.push(std::thread::spawn(move || {
            memory_unit_task_mailbox(
                mailbox,
                broadcast,
                worker_id as u32,
                actions,
                shared,
            );
        }));
    }

    for (hash_table_id, mailbox) in hash_table_mailboxes.iter().cloned().enumerate() {
        info!(hash_table_id, "spawning hash table unit thread");
        let actions = actions_arc.clone();
        let shared = shared.clone();
        thread_handles.push(std::thread::spawn(move || {
            hash_table_unit_task_mailbox(mailbox, actions, shared);
        }));
    }

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    static CLIF_CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u64, Arc<Vec<unsafe extern "C" fn(*mut u8)>>>>> = std::sync::OnceLock::new();
    let cache = CLIF_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));

    let mut clif_compiled: std::collections::HashMap<usize, Arc<Vec<unsafe extern "C" fn(*mut u8)>>> = std::collections::HashMap::new();
    for (idx, &offset) in algorithm.state.cranelift_ir_offsets.iter().enumerate() {
        if !clif_compiled.contains_key(&idx) {
            let source = read_null_terminated_string_from_slice(&memory, offset, 64 * 1024);
            if !source.is_empty() {
                let mut hasher = DefaultHasher::new();
                source.hash(&mut hasher);
                let source_hash = hasher.finish();
                let compiled = {
                    let mut cache_lock = cache.lock().unwrap();
                    if let Some(cached) = cache_lock.get(&source_hash) {
                        cached.clone()
                    } else {
                        let compiled = units::CraneliftUnit::compile(&source);
                        cache_lock.insert(source_hash, compiled.clone());
                        compiled
                    }
                };
                clif_compiled.insert(idx, compiled);
            }
        }
    }

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

    let mut pc: usize = 0;
    let actions = &*actions_arc;
    let actions_len = actions.len() as u32;
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

            Kind::AsyncDispatch => {
                unsafe {
                    shared.store_u64(action.offset as usize, 0, Ordering::Release);
                }

                let unit_type = action.dst;
                let is_broadcast = (action.size & (1 << 31)) != 0;
                let size_count = action.size & 0x7FFF_FFFF;
                let start = action.src;
                let count = if size_count == 0 { 1 } else { size_count };
                let end = start.saturating_add(count).min(actions_len);
                let flag = action.offset;

                debug!(
                    pc,
                    unit_type,
                    is_broadcast,
                    start,
                    end,
                    flag,
                    "async_dispatch"
                );

                match unit_type {
                    0 => {
                        if gpu_mailboxes.is_empty() {
                            pc += 1;
                            continue;
                        }

                        let assigned = gpu_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0
                        } else {
                            (assigned as usize).min(gpu_mailboxes.len() - 1)
                        };

                        gpu_mailboxes[unit_id].post(start, end, flag);
                    }
                    1 => {
                        if is_broadcast {
                            simd_broadcast.dispatch(start, end, flag);
                            pc += 1;
                            continue;
                        }

                        if simd_mailboxes.is_empty() {
                            pc += 1;
                            continue;
                        }

                        let assigned = simd_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0
                        } else {
                            (assigned as usize).min(simd_mailboxes.len() - 1)
                        };

                        simd_mailboxes[unit_id].post(start, end, flag);
                    }
                    2 => {
                        if file_mailboxes.is_empty() {
                            pc += 1;
                            continue;
                        }

                        let assigned = file_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0
                        } else {
                            (assigned as usize).min(file_mailboxes.len() - 1)
                        };

                        file_mailboxes[unit_id].post(start, end, flag);
                    }
                    4 => {
                        if ffi_mailboxes.is_empty() {
                            pc += 1;
                            continue;
                        }

                        let assigned = ffi_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0
                        } else {
                            (assigned as usize).min(ffi_mailboxes.len() - 1)
                        };

                        ffi_mailboxes[unit_id].post(start, end, flag);
                    }
                    6 => {
                        if is_broadcast {
                            memory_broadcast.dispatch(start, end, flag);
                            pc += 1;
                            continue;
                        }

                        if memory_mailboxes.is_empty() {
                            pc += 1;
                            continue;
                        }

                        let assigned = memory_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0
                        } else {
                            (assigned as usize).min(memory_mailboxes.len() - 1)
                        };

                        memory_mailboxes[unit_id].post(start, end, flag);
                    }
                    7 => {
                        if hash_table_mailboxes.is_empty() {
                            pc += 1;
                            continue;
                        }

                        let assigned = hash_table_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0
                        } else {
                            (assigned as usize).min(hash_table_mailboxes.len() - 1)
                        };

                        hash_table_mailboxes[unit_id].post(start, end, flag);
                    }
                    9 => {
                        if cranelift_mailboxes.is_empty() {
                            pc += 1;
                            continue;
                        }

                        let assigned = cranelift_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0
                        } else {
                            (assigned as usize).min(cranelift_mailboxes.len() - 1)
                        };

                        cranelift_mailboxes[unit_id].post(start, end, flag);
                    }
                    _ => {}
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

            Kind::KernelStart => {
                let kind = action.size;
                let status = if kind == KERNEL_KIND_QUEUE_ROUTER {
                    let (queue_desc, unit_type, unit_id, stop_flag_addr, progress_addr) = unsafe {
                        (
                            shared.load_u32(action.dst as usize + KERNEL_DESC_QUEUE_OFF, Ordering::Acquire),
                            shared.load_u32(action.dst as usize + KERNEL_DESC_UNIT_TYPE_OFF, Ordering::Acquire),
                            shared.load_u32(action.dst as usize + KERNEL_DESC_UNIT_ID_OFF, Ordering::Acquire),
                            shared.load_u32(action.dst as usize + KERNEL_DESC_STOP_FLAG_OFF, Ordering::Acquire),
                            shared.load_u32(action.dst as usize + KERNEL_DESC_PROGRESS_OFF, Ordering::Acquire),
                        )
                    };

                    let pools: [&[Arc<Mailbox>]; 9] = [
                        &gpu_mailboxes, &simd_mailboxes, &file_mailboxes,
                        &[], &ffi_mailboxes, &[],
                        &memory_mailboxes, &hash_table_mailboxes, &[],
                    ];
                    let t = unit_type as usize;

                    // Pool mode (u32::MAX - 1): round-robin across all workers
                    let pool_mailboxes: Option<Vec<Arc<Mailbox>>> = if unit_id == u32::MAX - 1 {
                        if t < pools.len() && !pools[t].is_empty() {
                            Some(pools[t].to_vec())
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let single_target: Option<DispatchTarget> = if pool_mailboxes.is_some() {
                        None
                    } else if unit_id == u32::MAX {
                        match unit_type {
                            1 => Some(DispatchTarget::Broadcast(simd_broadcast.clone())),
                            6 => Some(DispatchTarget::Broadcast(memory_broadcast.clone())),
                            _ => {
                                if t < pools.len() && !pools[t].is_empty() {
                                    Some(DispatchTarget::Mailbox(pools[t][0].clone()))
                                } else {
                                    None
                                }
                            }
                        }
                    } else {
                        if t < pools.len() && !pools[t].is_empty() {
                            let idx = (unit_id as usize).min(pools[t].len() - 1);
                            Some(DispatchTarget::Mailbox(pools[t][idx].clone()))
                        } else {
                            None
                        }
                    };

                    if let Some(pool) = pool_mailboxes {
                        let stop = Arc::new(AtomicBool::new(false));
                        let mut handle_id = kernel_handles.len();
                        for (i, slot) in kernel_handles.iter().enumerate() {
                            if slot.is_none() {
                                handle_id = i;
                                break;
                            }
                        }
                        let handle = KernelHandle {
                            stop,
                            join: None,
                            progress_addr,
                            queue_desc,
                            pool: Some(pool),
                        };
                        if handle_id == kernel_handles.len() {
                            kernel_handles.push(Some(handle));
                        } else {
                            kernel_handles[handle_id] = Some(handle);
                        }
                        unsafe {
                            shared.store_u32(action.src as usize, handle_id as u32, Ordering::Release);
                        }
                        1u64
                    } else if let Some(single_target) = single_target {
                        let stop = Arc::new(AtomicBool::new(false));
                        let stop_for_thread = stop.clone();
                        let shared_for_thread = shared.clone();

                        let join = std::thread::spawn(move || {
                            let mut drained = 0u64;
                            let mut spin_count = 0u32;
                            loop {
                                if stop_for_thread.load(Ordering::Acquire) {
                                    break;
                                }
                                if stop_flag_addr != 0 {
                                    let stop_word = unsafe { shared_for_thread.load_u64(stop_flag_addr as usize, Ordering::Acquire) };
                                    if stop_word != 0 {
                                        break;
                                    }
                                }
                                let packed = unsafe { queue_try_pop_u64(&shared_for_thread, queue_desc as usize) };
                                if let Some(packed) = packed {
                                    spin_count = 0;
                                    let start = (packed >> 43) as u32;
                                    let end = ((packed >> 22) & 0x1F_FFFF) as u32;
                                    let flag = (packed & 0x3F_FFFF) as u32;

                                    match &single_target {
                                        DispatchTarget::Mailbox(m) => m.post(start, end, flag),
                                        DispatchTarget::Broadcast(b) => b.dispatch(start, end, flag),
                                    }
                                    if flag != 0 {
                                        loop {
                                            let done = unsafe {
                                                shared_for_thread.load_u64(flag as usize, Ordering::Acquire)
                                            };
                                            if done != 0 || stop_for_thread.load(Ordering::Acquire) {
                                                break;
                                            }
                                            std::hint::spin_loop();
                                        }
                                    }
                                    drained = drained.wrapping_add(1);
                                    if progress_addr != 0 {
                                        unsafe { shared_for_thread.store_u64(progress_addr as usize, drained, Ordering::Release) };
                                    }
                                } else if spin_count < 32 {
                                    std::hint::spin_loop();
                                    spin_count += 1;
                                } else {
                                    std::thread::sleep(Duration::from_micros(50));
                                }
                            }
                        });

                        let mut handle_id = kernel_handles.len();
                        for (i, slot) in kernel_handles.iter().enumerate() {
                            if slot.is_none() {
                                handle_id = i;
                                break;
                            }
                        }
                        let handle = KernelHandle {
                            stop,
                            join: Some(join),
                            progress_addr,
                            queue_desc,
                            pool: None,
                        };
                        if handle_id == kernel_handles.len() {
                            kernel_handles.push(Some(handle));
                        } else {
                            kernel_handles[handle_id] = Some(handle);
                        }
                        unsafe {
                            shared.store_u32(action.src as usize, handle_id as u32, Ordering::Release);
                        }
                        1u64
                    } else {
                        0u64
                    }
                } else {
                    0u64
                };
                unsafe {
                    shared.store_u64(action.offset as usize, status, Ordering::Release);
                }
                pc += 1;
            }

            Kind::KernelSubmit => {
                let handle_id = unsafe { shared.load_u32(action.dst as usize, Ordering::Acquire) } as usize;
                let mut submitted = 0u64;
                if let Some(Some(handle)) = kernel_handles.get(handle_id) {
                    let count = if action.size == 0 { 1 } else { action.size };
                    if let Some(ref pool) = handle.pool {
                        // Inline dispatch: post directly to workers round-robin
                        let mut progress = if handle.progress_addr != 0 {
                            unsafe { shared.load_u64(handle.progress_addr as usize, Ordering::Acquire) }
                        } else { 0 };
                        let pool_len = pool.len();
                        for i in 0..count {
                            let src_ptr = action.src as usize + (i as usize) * 8;
                            let packed = unsafe { shared.load_u64(src_ptr, Ordering::Acquire) };
                            let start = (packed >> 43) as u32;
                            let end = ((packed >> 22) & 0x1F_FFFF) as u32;
                            let flag = (packed & 0x3F_FFFF) as u32;
                            pool[(progress as usize) % pool_len].post(start, end, flag);
                            progress += 1;
                            submitted += 1;
                        }
                        if handle.progress_addr != 0 {
                            unsafe { shared.store_u64(handle.progress_addr as usize, progress, Ordering::Release) };
                        }
                    } else {
                        for i in 0..count {
                            let src_ptr = action.src as usize + (i as usize) * 8;
                            let ok = unsafe {
                                queue_try_push_packet_mp(&shared, handle.queue_desc as usize, src_ptr)
                            };
                            if ok {
                                submitted += 1;
                            } else {
                                break;
                            }
                        }
                    }
                }
                unsafe {
                    shared.store_u64(action.offset as usize, submitted, Ordering::Release);
                }
                pc += 1;
            }

            Kind::KernelSubmitIndirect => {
                let handle_id =
                    unsafe { shared.load_u32(action.dst as usize, Ordering::Acquire) } as usize;
                let packet_base = unsafe { shared.load_u32(action.src as usize, Ordering::Acquire) };
                let mut submitted = 0u64;
                if let Some(Some(handle)) = kernel_handles.get(handle_id) {
                    let count = if action.size == 0 { 1 } else { action.size };
                    if let Some(ref pool) = handle.pool {
                        let mut progress = if handle.progress_addr != 0 {
                            unsafe { shared.load_u64(handle.progress_addr as usize, Ordering::Acquire) }
                        } else { 0 };
                        let pool_len = pool.len();
                        for i in 0..count {
                            let src_ptr = packet_base as usize + (i as usize) * 8;
                            let packed = unsafe { shared.load_u64(src_ptr, Ordering::Acquire) };
                            let start = (packed >> 43) as u32;
                            let end = ((packed >> 22) & 0x1F_FFFF) as u32;
                            let flag = (packed & 0x3F_FFFF) as u32;
                            pool[(progress as usize) % pool_len].post(start, end, flag);
                            progress += 1;
                            submitted += 1;
                        }
                        if handle.progress_addr != 0 {
                            unsafe { shared.store_u64(handle.progress_addr as usize, progress, Ordering::Release) };
                        }
                    } else {
                        for i in 0..count {
                            let src_ptr = packet_base as usize + (i as usize) * 8;
                            let ok = unsafe {
                                queue_try_push_packet_mp(&shared, handle.queue_desc as usize, src_ptr)
                            };
                            if ok {
                                submitted += 1;
                            } else {
                                break;
                            }
                        }
                    }
                }
                unsafe {
                    shared.store_u64(action.offset as usize, submitted, Ordering::Release);
                }
                pc += 1;
            }

            Kind::KernelWait => {
                let handle_id = unsafe { shared.load_u32(action.dst as usize, Ordering::Acquire) } as usize;
                let target = action.src as u64;
                let timeout_ms = action.size as u64;
                let start_t = std::time::Instant::now();
                let mut ok = false;
                if let Some(Some(handle)) = kernel_handles.get(handle_id) {
                    loop {
                        let progress = if handle.progress_addr == 0 {
                            0
                        } else {
                            unsafe { shared.load_u64(handle.progress_addr as usize, Ordering::Acquire) }
                        };
                        if progress >= target {
                            ok = true;
                            break;
                        }
                        if timeout_ms != 0 && start_t.elapsed() >= Duration::from_millis(timeout_ms) {
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
                unsafe {
                    shared.store_u64(action.offset as usize, if ok { 1 } else { 0 }, Ordering::Release);
                }
                pc += 1;
            }

            Kind::KernelStop => {
                let handle_id = unsafe { shared.load_u32(action.dst as usize, Ordering::Acquire) } as usize;
                let mut ok = false;
                if handle_id < kernel_handles.len() {
                    if let Some(mut handle) = kernel_handles[handle_id].take() {
                        handle.stop.store(true, Ordering::Release);
                        if let Some(join) = handle.join.take() {
                            let _ = join.join();
                        }
                        ok = true;
                    }
                }
                unsafe {
                    shared.store_u64(action.offset as usize, if ok { 1 } else { 0 }, Ordering::Release);
                }
                pc += 1;
            }

            _ => {
                pc += 1;
            }
        }
    }

    for slot in kernel_handles.iter_mut() {
        if let Some(mut handle) = slot.take() {
            handle.stop.store(true, Ordering::Release);
            if let Some(join) = handle.join.take() {
                let _ = join.join();
            }
        }
    }

    info!("shutting down all units");
    for mailbox in gpu_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in simd_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in file_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in ffi_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in memory_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in hash_table_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in cranelift_mailboxes.iter() {
        mailbox.shutdown();
    }
    simd_broadcast.shutdown();
    memory_broadcast.shutdown();

    for handle in thread_handles {
        let _ = handle.join();
    }
    info!("all unit threads joined");

    Ok(())
}
