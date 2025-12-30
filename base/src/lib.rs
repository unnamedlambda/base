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
                Kind::ConditionalWrite | Kind::MemCopy | Kind::MemScan => {
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
        for (i, action) in algorithm.actions.iter().enumerate() {
            match action.kind {
                Kind::CreateBuffer | Kind::WriteBuffer | Kind::CreateShader 
                | Kind::CreatePipeline | Kind::Dispatch | Kind::ReadBuffer => {
                    algorithm.gpu_assignments[i] = 0;
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


    let (gpu_tx, gpu_rx) = mpsc::channel(algorithm.queues.capacity);
    let (simd_tx, simd_rx) = mpsc::channel(algorithm.queues.capacity);
    let (file_tx, file_rx) = mpsc::channel(algorithm.queues.capacity);
    let (network_tx, network_rx) = mpsc::channel(algorithm.queues.capacity);
    let (ffi_tx, ffi_rx) = mpsc::channel(algorithm.queues.capacity);
    let (computational_tx, computational_rx) = mpsc::channel(algorithm.queues.capacity);
    let (memory_tx, memory_rx) = mpsc::channel(algorithm.queues.capacity);

    if algorithm.units.gpu_enabled && !algorithm.state.gpu_shader_offsets.is_empty() {
        let gpu_size = algorithm.state.gpu_size;
        let backends = Backends::from_bits(algorithm.units.backends_bits).unwrap_or(Backends::all());
        let offset = algorithm.state.gpu_shader_offsets[0];
        let shader_source = read_null_terminated_string_from_slice(&algorithm.payloads, offset, 8192);

        tokio::spawn(gpu_unit_task(
            gpu_rx,
            actions_arc.clone(),
            shared.clone(),
            shader_source,
            gpu_size,
            backends,
        ));
    }

    if algorithm.units.simd_units > 0 {
        let scratch = Arc::new(vec![0u8; algorithm.state.unit_scratch_size]);
        tokio::spawn(simd_unit_task(
            simd_rx,
            actions_arc.clone(),
            shared.clone(),
            algorithm.state.regs_per_unit,
            scratch,
            algorithm.state.unit_scratch_offsets[0],
            algorithm.state.unit_scratch_size,
            0, // shared_offset
        ));
    }

    if algorithm.units.file_units > 0 {
        tokio::spawn(file_unit_task(
            file_rx,
            actions_arc.clone(),
            shared.clone(),
            algorithm.state.file_buffer_size,
        ));
    }


    tokio::spawn(network_unit_task(
        network_rx,
        actions_arc.clone(),
        shared.clone(),
    ));

    tokio::spawn(ffi_unit_task(
        ffi_rx,
        actions_arc.clone(),
        shared.clone(),
    ));

    if algorithm.units.computational_enabled {
        tokio::spawn(computational_unit_task(
            computational_rx,
            actions_arc.clone(),
            shared.clone(),
            algorithm.state.computational_regs,
        ));
    }

    tokio::spawn(memory_unit_task(
        memory_rx,
        actions_arc.clone(),
        shared.clone(),
    ));

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

                let item = units::QueueItem {
                    action_index: action.src,
                    offset: action.offset,
                    size: action.size as u16,
                    unit_id: action.dst as u8,
                    _pad: 0,
                };

                match action.dst {
                    0 => { let _ = gpu_tx.send(item).await; }
                    1 => { let _ = simd_tx.send(item).await; }
                    2 => { let _ = file_tx.send(item).await; }
                    3 => { let _ = network_tx.send(item).await; }
                    4 => { let _ = ffi_tx.send(item).await; }
                    5 => { let _ = computational_tx.send(item).await; }
                    6 => { let _ = memory_tx.send(item).await; }
                    _ => {
                    }
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
