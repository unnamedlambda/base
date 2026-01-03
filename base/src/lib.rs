use std::{pin::Pin, sync::Arc, time::Duration};
use tokio::sync::mpsc;
use wgpu::Backends;

pub use base_types::{Action, Algorithm, Kind, QueueSpec, State, UnitSpec};

mod units;
mod validation;

use crate::units::{
    computational_unit_task, ffi_unit_task, file_unit_task, gpu_unit_task, memory_unit_task,
    network_unit_task, read_null_terminated_string_from_slice, simd_unit_task, SharedMemory,
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
                Kind::SimdLoad | Kind::SimdAdd | Kind::SimdMul | Kind::SimdStore => {
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
                Kind::ConditionalWrite | Kind::MemCopy | Kind::MemScan | Kind::AtomicCAS | Kind::Fence => {
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

    let gpu_channels: Vec<_> = (0..algorithm.units.gpu_units)
        .map(|_| mpsc::channel(algorithm.queues.capacity))
        .collect();
    let gpu_senders: Vec<_> = gpu_channels.iter().map(|(tx, _)| tx.clone()).collect();
    let gpu_receivers: Vec<_> = gpu_channels.into_iter().map(|(_, rx)| rx).collect();

    let simd_channels: Vec<_> = (0..algorithm.units.simd_units)
        .map(|_| mpsc::channel(algorithm.queues.capacity))
        .collect();
    let simd_senders: Vec<_> = simd_channels.iter().map(|(tx, _)| tx.clone()).collect();
    let simd_receivers: Vec<_> = simd_channels.into_iter().map(|(_, rx)| rx).collect();

    let file_channels: Vec<_> = (0..algorithm.units.file_units)
        .map(|_| mpsc::channel(algorithm.queues.capacity))
        .collect();
    let file_senders: Vec<_> = file_channels.iter().map(|(tx, _)| tx.clone()).collect();
    let file_receivers: Vec<_> = file_channels.into_iter().map(|(_, rx)| rx).collect();

    let network_channels: Vec<_> = (0..algorithm.units.network_units)
        .map(|_| mpsc::channel(algorithm.queues.capacity))
        .collect();
    let network_senders: Vec<_> = network_channels.iter().map(|(tx, _)| tx.clone()).collect();
    let network_receivers: Vec<_> = network_channels.into_iter().map(|(_, rx)| rx).collect();

    let ffi_channels: Vec<_> = (0..algorithm.units.ffi_units)
        .map(|_| mpsc::channel(algorithm.queues.capacity))
        .collect();
    let ffi_senders: Vec<_> = ffi_channels.iter().map(|(tx, _)| tx.clone()).collect();
    let ffi_receivers: Vec<_> = ffi_channels.into_iter().map(|(_, rx)| rx).collect();

    let computational_channels: Vec<_> = (0..algorithm.units.computational_units)
        .map(|_| mpsc::channel(algorithm.queues.capacity))
        .collect();
    let computational_senders: Vec<_> = computational_channels.iter().map(|(tx, _)| tx.clone()).collect();
    let computational_receivers: Vec<_> = computational_channels.into_iter().map(|(_, rx)| rx).collect();

    let memory_channels: Vec<_> = (0..algorithm.units.memory_units)
        .map(|_| mpsc::channel(algorithm.queues.capacity))
        .collect();
    let memory_senders: Vec<_> = memory_channels.iter().map(|(tx, _)| tx.clone()).collect();
    let memory_receivers: Vec<_> = memory_channels.into_iter().map(|(_, rx)| rx).collect();

    for (gpu_id, rx) in gpu_receivers.into_iter().enumerate() {
        if gpu_id < algorithm.state.gpu_shader_offsets.len() {
            let gpu_size = algorithm.state.gpu_size;
            let backends = Backends::from_bits(algorithm.units.backends_bits).unwrap_or(Backends::all());
            let offset = algorithm.state.gpu_shader_offsets[gpu_id];
            let shader_source = read_null_terminated_string_from_slice(&algorithm.payloads, offset, 8192);

            tokio::spawn(gpu_unit_task(
                rx,
                actions_arc.clone(),
                shared.clone(),
                shader_source,
                gpu_size,
                backends,
            ));
        }
    }

    for (_, rx) in simd_receivers.into_iter().enumerate() {
        tokio::spawn(simd_unit_task(
            rx,
            actions_arc.clone(),
            shared.clone(),
            algorithm.state.regs_per_unit,
        ));
    }

    for (_, rx) in file_receivers.into_iter().enumerate() {
        tokio::spawn(file_unit_task(
            rx,
            actions_arc.clone(),
            shared.clone(),
            algorithm.state.file_buffer_size,
        ));
    }

    for (_, rx) in network_receivers.into_iter().enumerate() {
        tokio::spawn(network_unit_task(
            rx,
            actions_arc.clone(),
            shared.clone(),
        ));
    }

    for (_, rx) in ffi_receivers.into_iter().enumerate() {
        tokio::spawn(ffi_unit_task(
            rx,
            actions_arc.clone(),
            shared.clone(),
        ));
    }

    for (_, rx) in computational_receivers.into_iter().enumerate() {
        tokio::spawn(computational_unit_task(
            rx,
            actions_arc.clone(),
            shared.clone(),
            algorithm.state.computational_regs,
        ));
    }

    for (_, rx) in memory_receivers.into_iter().enumerate() {
        tokio::spawn(memory_unit_task(
            rx,
            actions_arc.clone(),
            shared.clone(),
        ));
    }

    let mut memory_unit = units::MemoryUnit::new(shared.clone());
    let mut file_unit = units::FileUnit::new(0, shared.clone(), algorithm.state.file_buffer_size);

    let mut pc: usize = 0;
    let actions = &algorithm.actions;
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
            Kind::MemCopy => {
                unsafe { memory_unit.execute(action); }
                pc += 1;
            }

            Kind::ConditionalJump => {
                let cond_bytes = unsafe { shared.read(action.src as usize + action.offset as usize, 8) };
                let cond = f64::from_le_bytes(
                    cond_bytes[0..8].try_into().map_err(|_| Error::Execution("ConditionalJump: invalid condition".into()))?
                );

                if cond != 0.0 {
                    pc = action.dst as usize;  // Jump
                } else {
                    pc += 1;  // Fall through
                }
            }

            Kind::AsyncDispatch => {
                unsafe {
                    // Clear completion flag
                    let zero: u64 = 0;
                    shared.write(action.offset as usize, &zero.to_le_bytes());
                }

                let unit_type = action.dst;

                match unit_type {
                    0 => {
                        if gpu_senders.is_empty() {
                            // No GPU units configured - skip this action
                            pc += 1;
                            continue;
                        }

                        let assigned = gpu_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0  // Unassigned - use GPU 0
                        } else {
                            (assigned as usize).min(gpu_senders.len() - 1)
                        };

                        let item = units::QueueItem {
                            action_index: action.src,
                            offset: action.offset,
                        };
                        let _ = gpu_senders[unit_id].send(item).await;
                    }
                    1 => {
                        if simd_senders.is_empty() {
                            // No SIMD units configured - skip this action
                            pc += 1;
                            continue;
                        }

                        let assigned = simd_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0  // Unassigned - use unit 0
                        } else {
                            (assigned as usize).min(simd_senders.len() - 1)
                        };

                        let item = units::QueueItem {
                            action_index: action.src,
                            offset: action.offset,
                        };
                        let _ = simd_senders[unit_id].send(item).await;
                    }
                    2 => {
                        if file_senders.is_empty() {
                            // No file units configured - skip this action
                            pc += 1;
                            continue;
                        }

                        let assigned = file_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0  // Unassigned - use unit 0
                        } else {
                            (assigned as usize).min(file_senders.len() - 1)
                        };

                        let item = units::QueueItem {
                            action_index: action.src,
                            offset: action.offset,
                        };
                        let _ = file_senders[unit_id].send(item).await;
                    }
                    3 => {
                        if network_senders.is_empty() {
                            // No network units configured - skip this action
                            pc += 1;
                            continue;
                        }

                        let assigned = network_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0  // Unassigned - use interface 0
                        } else {
                            (assigned as usize).min(network_senders.len() - 1)
                        };

                        let item = units::QueueItem {
                            action_index: action.src,
                            offset: action.offset,
                        };
                        let _ = network_senders[unit_id].send(item).await;
                    }
                    4 => {
                        if ffi_senders.is_empty() {
                            // No FFI units configured - skip this action
                            pc += 1;
                            continue;
                        }

                        let assigned = ffi_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0  // Unassigned - use unit 0
                        } else {
                            (assigned as usize).min(ffi_senders.len() - 1)
                        };

                        let item = units::QueueItem {
                            action_index: action.src,
                            offset: action.offset,
                        };
                        let _ = ffi_senders[unit_id].send(item).await;
                    }
                    5 => {
                        if computational_senders.is_empty() {
                            // No computational units configured - skip this action
                            pc += 1;
                            continue;
                        }

                        let assigned = computational_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0  // Unassigned - use unit 0
                        } else {
                            (assigned as usize).min(computational_senders.len() - 1)
                        };

                        let item = units::QueueItem {
                            action_index: action.src,
                            offset: action.offset,
                        };
                        let _ = computational_senders[unit_id].send(item).await;
                    }
                    6 => {
                        if memory_senders.is_empty() {
                            // No memory units configured - skip this action
                            pc += 1;
                            continue;
                        }

                        let assigned = memory_assignments
                            .get(pc)
                            .copied()
                            .unwrap_or(0);

                        let unit_id = if assigned == 255 {
                            0  // Unassigned - use unit 0
                        } else {
                            (assigned as usize).min(memory_senders.len() - 1)
                        };

                        let item = units::QueueItem {
                            action_index: action.src,
                            offset: action.offset,
                        };
                        let _ = memory_senders[unit_id].send(item).await;
                    }
                    _ => {}
                }

                pc += 1;
            }

            Kind::Wait => {
                // Busy-wait on completion flag
                loop {
                    let flag_bytes = unsafe { shared.read(action.dst as usize, 8) };
                    let flag = u64::from_le_bytes(
                        flag_bytes[0..8].try_into().map_err(|_| Error::Execution("Wait: invalid flag".into()))?
                    );
                    if flag != 0 {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                pc += 1;
            }

            Kind::FileWrite => {
                file_unit.execute(action).await;
                pc += 1;
            }

            Kind::FileRead => {
                file_unit.execute(action).await;
                pc += 1;
            }

            Kind::MemWrite => {
                unsafe {
                    let data = shared.read(action.src as usize, action.size as usize);
                    shared.write(action.dst as usize, data);
                }
                pc += 1;
            }

            _ => {
                pc += 1;
            }
        }
    }

    Ok(())
}
