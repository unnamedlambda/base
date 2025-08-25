use std::{pin::Pin, sync::Arc, time::Duration};
use tokio::sync::mpsc;
use wgpu::Backends;

pub use crate::types::{Action, Algorithm, Kind, QueueSpec, State, UnitSpec};

mod types;
mod units;
mod validation;

use crate::units::{
    computational_unit_task, ffi_unit_task, file_unit_task, gpu_unit_task, memory_unit_task,
    network_unit_task, simd_unit_task, SharedMemory,
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

    let mut write_work: Vec<usize> = Vec::new();
    for (i, &assignment) in algorithm.memory_assignments.iter().enumerate() {
        if assignment != 255 {
            write_work.push(i);
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

    if !write_work.is_empty() {
        let actions = actions.clone();
        let shared_clone = shared.clone();

        handles.push(tokio::spawn(memory_unit_task(
            actions,
            write_work,
            shared_clone,
        )));
    }

    if algorithm.units.gpu_enabled {
        let gpu_size = algorithm.state.gpu_size;
        let batch_size = algorithm.queues.batch_size;
        let backends =
            Backends::from_bits(algorithm.units.backends_bits).unwrap_or(Backends::all());

        handles.push(tokio::spawn(gpu_unit_task(
            rx, shared, gpu_size, batch_size, backends,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_algorithm() {
        let alg = Algorithm::default();
        assert_eq!(alg.units.simd_units, 4);
        assert_eq!(alg.queues.capacity, 256);
        assert_eq!(alg.queues.batch_size, 16);
        assert!(alg.units.gpu_enabled);
        assert!(alg.units.computational_enabled);
        assert_eq!(alg.state.computational_regs, 32);
    }

    #[test]
    fn test_validate_memory_overlap() {
        let mut alg = Algorithm::default();
        // Force memory overlap
        alg.state.unit_scratch_offsets[1] = 1000;
        alg.state.unit_scratch_size = 5000; // This will overlap with offset 4096

        let result = validate(&alg);
        assert!(result.is_err());
        if let Err(Error::InvalidConfig(msg)) = result {
            assert!(msg.contains("overlap"));
        }
    }

    #[test]
    fn test_validate_invalid_assignments() {
        let mut alg = Algorithm::default();
        alg.simd_assignments = vec![0, 1, 2, 3, 4]; // 4 is invalid for 4 units (0-3)

        let result = validate(&alg);
        assert!(result.is_err());
    }

    #[test]
    fn test_simd_assignment_generation() {
        let mut alg = Algorithm::default();
        alg.actions = vec![
            Action {
                kind: Kind::SimdLoad,
                dst: 0,
                src: 0,
                offset: 0,
                size: 16,
            },
            Action {
                kind: Kind::SimdAdd,
                dst: 1,
                src: 0,
                offset: 0,
                size: 0,
            },
            Action {
                kind: Kind::SimdMul,
                dst: 2,
                src: 1,
                offset: 0,
                size: 0,
            },
            Action {
                kind: Kind::SimdStore,
                dst: 0,
                src: 2,
                offset: 0,
                size: 16,
            },
        ];

        assert!(alg.simd_assignments.is_empty());
        // Disable GPU for this test to avoid pipeline issues
        alg.units.gpu_enabled = false;
        let result = execute(alg);
        // Should auto-generate assignments without error
        assert!(result.is_ok());
    }

    #[test]
    fn test_computational_assignment_generation() {
        let mut alg = Algorithm::default();

        // Set up some initial values in registers
        alg.payloads = vec![0u8; 65536];
        // Store 100.0 in the first position (for Choose to use)
        let hundred_bytes = 100.0_f64.to_le_bytes();
        alg.payloads[0..8].copy_from_slice(&hundred_bytes);

        alg.actions = vec![
            Action {
                kind: Kind::Choose,
                dst: 0,
                src: 1,
                offset: 0,
                size: 0,
            },
            Action {
                kind: Kind::Approximate,
                dst: 1,
                src: 0,
                offset: 0,
                size: 0,
            },
        ];

        assert!(alg.computational_assignments.is_empty());
        alg.units.gpu_enabled = false; // Disable GPU for test
                                       // Keep SIMD units as default instead of setting to 0

        let result = execute(alg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mixed_actions() {
        let mut alg = Algorithm::default();
        alg.actions = vec![
            Action {
                kind: Kind::SimdLoad,
                dst: 0,
                src: 0,
                offset: 0,
                size: 16,
            },
            Action {
                kind: Kind::Choose,
                dst: 1,
                src: 0,
                offset: 0,
                size: 0,
            },
            Action {
                kind: Kind::Approximate,
                dst: 2,
                src: 1,
                offset: 0,
                size: 0,
            },
            Action {
                kind: Kind::SimdStore,
                dst: 0,
                src: 0,
                offset: 0,
                size: 16,
            },
        ];

        alg.units.gpu_enabled = false;
        let result = execute(alg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_empty_algorithm() {
        let alg = Algorithm::default();
        let result = execute(alg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_timeout() {
        let mut alg = Algorithm::default();
        alg.timeout_ms = Some(1); // 1ms timeout
        alg.actions = vec![Action {
            kind: Kind::SimdLoad,
            dst: 0,
            src: 0,
            offset: 0,
            size: 16,
        }];

        // This might or might not timeout depending on system speed
        // Just ensure it doesn't panic
        let _ = execute(alg);
    }

    #[test]
    fn test_computational_disabled() {
        let mut alg = Algorithm::default();
        alg.units.computational_enabled = false;
        alg.actions = vec![Action {
            kind: Kind::Choose,
            dst: 0,
            src: 1,
            offset: 0,
            size: 0,
        }];

        // Should still execute without error (action will be ignored)
        alg.units.gpu_enabled = false;
        let result = execute(alg);
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_algorithm_execution() {
        let mut alg = Algorithm::default();

        // Create a simple computation pipeline
        alg.actions = vec![
            // Load data into SIMD
            Action {
                kind: Kind::SimdLoad,
                dst: 0,
                src: 0,
                offset: 0,
                size: 16,
            },
            // Add two SIMD registers
            Action {
                kind: Kind::SimdAdd,
                dst: 2,
                src: 0,
                offset: 1,
                size: 0,
            },
            // Store result
            Action {
                kind: Kind::SimdStore,
                src: 2,
                dst: 0,
                offset: 100,
                size: 16,
            },
        ];

        // Initialize some data
        for i in 0..16 {
            alg.payloads[i] = i as u8;
        }

        alg.units.gpu_enabled = false; // Disable GPU for test
        alg.timeout_ms = Some(1000); // 1 second timeout

        let result = execute(alg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_algorithm_with_timeout() {
        let mut alg = Algorithm::default();
        alg.timeout_ms = Some(1); // Very short timeout

        // Add many actions to potentially trigger timeout
        for _ in 0..1000 {
            alg.actions.push(Action {
                kind: Kind::MemCopy,
                src: 0,
                dst: 100,
                offset: 0,
                size: 100,
            });
        }

        alg.units.gpu_enabled = false;

        // This might timeout or complete depending on system speed
        let _ = execute(alg);
    }

    #[test]
    fn test_mixed_unit_coordination() {
        let mut alg = Algorithm::default();

        // Create actions that use different units
        alg.actions = vec![
            // Computational unit: generate random number
            Action {
                kind: Kind::Choose,
                dst: 0,
                src: 1,
                offset: 0,
                size: 0,
            },
            // SIMD unit: process data
            Action {
                kind: Kind::SimdLoad,
                dst: 0,
                src: 0,
                offset: 0,
                size: 16,
            },
            // Write unit: copy memory
            Action {
                kind: Kind::MemCopy,
                src: 100,
                dst: 200,
                offset: 0,
                size: 50,
            },
        ];

        alg.units.gpu_enabled = false;

        let result = execute(alg);
        assert!(result.is_ok());
    }
}
