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
    let memory_arc = Arc::new(memory.to_vec());

    let (tx, rx) = mpsc::channel(algorithm.queues.capacity);

    let timeout = algorithm.timeout_ms.map(Duration::from_millis);

    let mut simd_work: Vec<Vec<usize>> = vec![Vec::new(); algorithm.units.simd_units];
    for (i, &assignment) in algorithm.simd_assignments.iter().enumerate() {
        if assignment != 255 {
            simd_work[assignment as usize].push(i);
        }
    }

    let mut computational_work: Vec<usize> = Vec::new();
    for (i, &assignment) in algorithm.computational_assignments.iter().enumerate() {
        if assignment != 255 {
            computational_work.push(i);
        }
    }

    let mut memory_work: Vec<usize> = Vec::new();
    for (i, &assignment) in algorithm.memory_assignments.iter().enumerate() {
        if assignment != 255 {
            memory_work.push(i);
        }
    }

    let mut ffi_work: Vec<usize> = Vec::new();
    for (i, &assignment) in algorithm.ffi_assignments.iter().enumerate() {
        if assignment != 255 {
            ffi_work.push(i);
        }
    }

    let mut network_work: Vec<usize> = Vec::new();
    for (i, &assignment) in algorithm.network_assignments.iter().enumerate() {
        if assignment != 255 {
            network_work.push(i);
        }
    }

    let mut file_work: Vec<Vec<usize>> = vec![Vec::new(); algorithm.units.file_units];
    for (i, &assignment) in algorithm.file_assignments.iter().enumerate() {
        if assignment != 255 {
            file_work[assignment as usize].push(i);
        }
    }

    let actions = Arc::new(algorithm.actions.clone());
    let mut handles = vec![];

    for (unit_id, indices) in file_work.into_iter().enumerate() {
        if !indices.is_empty() {
            let tx = tx.clone();
            let actions = actions.clone();
            let shared = shared.clone();
            let buffer_size = algorithm.state.file_buffer_size;

            handles.push(tokio::spawn(file_unit_task(
                unit_id as u8,
                actions,
                indices,
                shared,
                buffer_size,
                tx,
            )));
        }
    }

    for (unit_id, indices) in simd_work.into_iter().enumerate() {
        if !indices.is_empty() {
            let tx = tx.clone();
            let actions = actions.clone();
            let scratch_offset = algorithm.state.unit_scratch_offsets[unit_id];
            let scratch_size = algorithm.state.unit_scratch_size;
            let shared = shared.clone();
            let shared_offset = unit_id * 1024;
            let regs = algorithm.state.regs_per_unit;
            let memory_arc = memory_arc.clone();

            handles.push(tokio::spawn(simd_unit_task(
                unit_id as u8,
                actions,
                indices,
                memory_arc,
                scratch_offset,
                scratch_size,
                shared,
                shared_offset,
                regs,
                tx,
            )));
        }
    }

    if !ffi_work.is_empty() {
        let actions = actions.clone();
        let shared = shared.clone();

        handles.push(tokio::spawn(ffi_unit_task(actions, ffi_work, shared)));
    }

    if !network_work.is_empty() {
        let tx = tx.clone();
        let actions = actions.clone();
        let shared = shared.clone();

        handles.push(tokio::spawn(network_unit_task(
            0, // network unit id
            actions,
            network_work,
            shared,
            tx,
        )));
    }

    drop(tx);

    if algorithm.units.computational_enabled && !computational_work.is_empty() {
        let actions = actions.clone();
        let regs = algorithm.state.computational_regs;

        handles.push(tokio::spawn(computational_unit_task(
            actions,
            computational_work,
            regs,
        )));
    }

    if !memory_work.is_empty() {
        let actions = actions.clone();
        let shared_clone = shared.clone();

        handles.push(tokio::spawn(memory_unit_task(
            actions,
            memory_work,
            shared_clone,
        )));
    }

    if algorithm.units.gpu_enabled && !algorithm.state.gpu_shader_offsets.is_empty() {
        let gpu_size = algorithm.state.gpu_size;
        let batch_size = algorithm.queues.batch_size;
        let backends =
            Backends::from_bits(algorithm.units.backends_bits).unwrap_or(Backends::all());

        let offset = algorithm.state.gpu_shader_offsets[0];
        let shader_source = read_null_terminated_string_from_slice(&algorithm.payloads, offset, 8192);

        handles.push(tokio::spawn(gpu_unit_task(
            rx, 
            shared, 
            shader_source,
            gpu_size, 
            batch_size, 
            backends,
        )));
    }

    if let Some(timeout) = timeout {
        match tokio::time::timeout(timeout, futures::future::join_all(&mut handles)).await {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::Execution("Timeout".into())),
        }
    } else {
        futures::future::join_all(handles).await;
        Ok(())
    }
}
