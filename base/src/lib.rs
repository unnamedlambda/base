use std::{pin::Pin, sync::Arc, time::Duration};
use std::sync::atomic::Ordering;
use wgpu::Backends;

pub use base_types::{Action, Algorithm, Kind, QueueSpec, State, UnitSpec};

mod units;
mod validation;

use crate::units::{
    computational_unit_task_mailbox, ffi_unit_task_mailbox, file_unit_task_mailbox,
    gpu_unit_task_mailbox, memory_unit_task_mailbox, network_unit_task_mailbox,
    read_null_terminated_string_from_slice, simd_unit_task_mailbox, Broadcast, Mailbox,
    SharedMemory,
};
use crate::validation::validate;

#[derive(Debug)]
pub enum Error {
    InvalidConfig(String),
    RuntimeCreation(std::io::Error),
    Execution(String),
    GpuInit(String),
}

pub fn execute(mut algorithm: Algorithm) -> Result<(), Error> {
    validate(&algorithm)?;

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
    }

    if algorithm.computational_assignments.is_empty() {
        algorithm.computational_assignments = vec![255; algorithm.actions.len()];
        for (i, action) in algorithm.actions.iter().enumerate() {
            match action.kind {
                Kind::Approximate | Kind::Choose | Kind::Compare | Kind::Timestamp => {
                    algorithm.computational_assignments[i] = 0;
                }
                _ => {}
            }
        }
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
                | Kind::MemStoreIndirect => {
                    algorithm.memory_assignments[i] = 0;
                }
                _ => {}
            }
        }
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
    }

    if algorithm.network_assignments.is_empty() {
        algorithm.network_assignments = vec![255; algorithm.actions.len()];
        for (i, action) in algorithm.actions.iter().enumerate() {
            match action.kind {
                Kind::NetConnect | Kind::NetAccept | Kind::NetSend | Kind::NetRecv => {
                    algorithm.network_assignments[i] = 0;
                }
                _ => {}
            }
        }
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
    }

    if algorithm.gpu_assignments.is_empty() {
        algorithm.gpu_assignments = vec![255; algorithm.actions.len()];
        let mut unit = 0u8;
        for (i, action) in algorithm.actions.iter().enumerate() {
            match action.kind {
                Kind::CreateBuffer | Kind::WriteBuffer | Kind::CreateShader
                | Kind::CreatePipeline | Kind::Dispatch | Kind::ReadBuffer => {
                    algorithm.gpu_assignments[i] = unit;
                    if action.kind == Kind::Dispatch && algorithm.units.gpu_units > 1 {
                        unit = (unit + 1) % algorithm.units.gpu_units as u8;
                    }
                }
                _ => {}
            }
        }
    }

    let mut builder = tokio::runtime::Builder::new_multi_thread();

    if let Some(workers) = algorithm.worker_threads {
        builder.worker_threads(workers);
    }

    if let Some(blocking) = algorithm.blocking_threads {
        builder.max_blocking_threads(blocking);
    }

    if let Some(stack) = algorithm.stack_size {
        builder.thread_stack_size(stack);
    }

    if let Some(prefix) = &algorithm.thread_name_prefix {
        builder.thread_name(prefix);
    }

    let runtime = builder
        .enable_all()
        .build()
        .map_err(Error::RuntimeCreation)?;

    let result = runtime.block_on(execute_internal(algorithm));
    runtime.shutdown_timeout(Duration::from_millis(100));
    result
}

async fn execute_internal(algorithm: Algorithm) -> Result<(), Error> {
    let mut memory = Pin::new(algorithm.payloads.clone().into_boxed_slice());
    let mem_ptr = memory.as_mut().as_mut_ptr();
    let shared = Arc::new(SharedMemory::new(mem_ptr));
    let actions_arc = Arc::new(algorithm.actions.clone());

    let gpu_assignments = Arc::new(algorithm.gpu_assignments.clone());
    let simd_assignments = Arc::new(algorithm.simd_assignments.clone());
    let file_assignments = Arc::new(algorithm.file_assignments.clone());
    let network_assignments = Arc::new(algorithm.network_assignments.clone());
    let ffi_assignments = Arc::new(algorithm.ffi_assignments.clone());
    let computational_assignments = Arc::new(algorithm.computational_assignments.clone());
    let memory_assignments = Arc::new(algorithm.memory_assignments.clone());

    let gpu_mailboxes: Vec<_> = (0..algorithm.units.gpu_units)
        .map(|_| Arc::new(Mailbox::new()))
        .collect();
    let simd_mailboxes: Vec<_> = (0..algorithm.units.simd_units)
        .map(|_| Arc::new(Mailbox::new()))
        .collect();
    let file_mailboxes: Vec<_> = (0..algorithm.units.file_units)
        .map(|_| Arc::new(Mailbox::new()))
        .collect();
    let network_mailboxes: Vec<_> = (0..algorithm.units.network_units)
        .map(|_| Arc::new(Mailbox::new()))
        .collect();
    let ffi_mailboxes: Vec<_> = (0..algorithm.units.ffi_units)
        .map(|_| Arc::new(Mailbox::new()))
        .collect();
    let computational_mailboxes: Vec<_> = (0..algorithm.units.computational_units)
        .map(|_| Arc::new(Mailbox::new()))
        .collect();
    let memory_mailboxes: Vec<_> = (0..algorithm.units.memory_units)
        .map(|_| Arc::new(Mailbox::new()))
        .collect();

    let simd_broadcast = Arc::new(Broadcast::new(algorithm.units.simd_units as u32));
    let computational_broadcast =
        Arc::new(Broadcast::new(algorithm.units.computational_units as u32));
    let memory_broadcast = Arc::new(Broadcast::new(algorithm.units.memory_units as u32));

    let mut thread_handles = Vec::new();
    let mut async_handles = Vec::new();

    for (gpu_id, mailbox) in gpu_mailboxes.iter().enumerate() {
        if gpu_id < algorithm.state.gpu_shader_offsets.len() {
            let gpu_size = algorithm.state.gpu_size;
            let backends = Backends::from_bits(algorithm.units.backends_bits).unwrap_or(Backends::all());
            let offset = algorithm.state.gpu_shader_offsets[gpu_id];
            let shader_source =
                read_null_terminated_string_from_slice(&algorithm.payloads, offset, 8192);
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

    for mailbox in file_mailboxes.iter().cloned() {
        let actions = actions_arc.clone();
        let shared = shared.clone();
        let buffer_size = algorithm.state.file_buffer_size;
        async_handles.push(tokio::spawn(async move {
            file_unit_task_mailbox(mailbox, actions, shared, buffer_size).await;
        }));
    }

    for mailbox in network_mailboxes.iter().cloned() {
        let actions = actions_arc.clone();
        let shared = shared.clone();
        async_handles.push(tokio::spawn(async move {
            network_unit_task_mailbox(mailbox, actions, shared).await;
        }));
    }

    for mailbox in ffi_mailboxes.iter().cloned() {
        let actions = actions_arc.clone();
        let shared = shared.clone();
        thread_handles.push(std::thread::spawn(move || {
            ffi_unit_task_mailbox(mailbox, actions, shared);
        }));
    }

    for (worker_id, mailbox) in computational_mailboxes.iter().cloned().enumerate() {
        let actions = actions_arc.clone();
        let shared = shared.clone();
        let regs = algorithm.state.computational_regs;
        let broadcast = computational_broadcast.clone();
        thread_handles.push(std::thread::spawn(move || {
            computational_unit_task_mailbox(
                mailbox,
                broadcast,
                worker_id as u32,
                actions,
                shared,
                regs,
            );
        }));
    }

    for (worker_id, mailbox) in memory_mailboxes.iter().cloned().enumerate() {
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

    let mut pc: usize = 0;
    let actions = &algorithm.actions;
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
                    3 => {
                        if network_mailboxes.is_empty() {
                            pc += 1;
                            continue;
                        }

                        let assigned = network_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0
                        } else {
                            (assigned as usize).min(network_mailboxes.len() - 1)
                        };

                        network_mailboxes[unit_id].post(start, end, flag);
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
                    5 => {
                        if is_broadcast {
                            computational_broadcast.dispatch(start, end, flag);
                            pc += 1;
                            continue;
                        }

                        if computational_mailboxes.is_empty() {
                            pc += 1;
                            continue;
                        }

                        let assigned = computational_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0
                        } else {
                            (assigned as usize).min(computational_mailboxes.len() - 1)
                        };

                        computational_mailboxes[unit_id].post(start, end, flag);
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
                    _ => {}
                }

                pc += 1;
            }

            Kind::Wait => {
                loop {
                    let flag = unsafe { shared.load_u64(action.dst as usize, Ordering::Acquire) };
                    if flag != 0 {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                pc += 1;
            }

            _ => {
                pc += 1;
            }
        }
    }

    for mailbox in gpu_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in simd_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in file_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in network_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in ffi_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in computational_mailboxes.iter() {
        mailbox.shutdown();
    }
    for mailbox in memory_mailboxes.iter() {
        mailbox.shutdown();
    }
    simd_broadcast.shutdown();
    computational_broadcast.shutdown();
    memory_broadcast.shutdown();

    for handle in thread_handles {
        let _ = handle.join();
    }
    for handle in async_handles {
        let _ = handle.await;
    }

    Ok(())
}
