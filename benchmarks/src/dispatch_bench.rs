use base::Algorithm;
use base_types::{Action, Kind, State, UnitSpec};
use crossbeam_channel::unbounded;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use crate::harness::BenchResult;

const MEMORY_UNIT_TYPE: u32 = 6;

fn align64(v: usize) -> usize {
    (v + 63) & !63
}

fn packed_packet(start: u32, end: u32, flag: u32) -> u64 {
    ((start as u64) << 43) | ((end as u64) << 22) | (flag as u64)
}

fn decode_packet(packet: u64) -> (u32, u32, u32) {
    let start = (packet >> 43) as u32;
    let end = ((packet >> 22) & 0x1F_FFFF) as u32;
    let flag = (packet & 0x3F_FFFF) as u32;
    (start, end, flag)
}

fn synthetic_degree(node: usize) -> u64 {
    let base = 1 + (1024 / (1 + (node % 1024))) as u64;
    let jitter = ((node as u64).wrapping_mul(1_103_515_245).wrapping_add(12_345) % 11) as u64;
    let hotspot = if node % 97 == 0 { 32 } else { 1 };
    (base + jitter) * hotspot
}

fn build_frontier_chunk_sums(num_nodes: usize, chunk: usize) -> (Vec<u64>, u64) {
    let mut frontier_costs = Vec::with_capacity(num_nodes * 2 / 3);
    for i in 0..num_nodes {
        if i % 5 != 0 {
            frontier_costs.push(synthetic_degree(i));
        }
    }

    let mut chunk_sums = Vec::with_capacity((frontier_costs.len() + chunk - 1) / chunk);
    let mut expected = 0u64;

    let mut pos = 0usize;
    while pos < frontier_costs.len() {
        let end = (pos + chunk).min(frontier_costs.len());
        let mut s = 0u64;
        for &v in &frontier_costs[pos..end] {
            s = s.wrapping_add(v);
        }
        expected = expected.wrapping_add(s);
        chunk_sums.push(s);
        pos = end;
    }

    (chunk_sums, expected)
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}

fn median(times: &mut [f64]) -> f64 {
    if times.is_empty() {
        return f64::NAN;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times[times.len() / 2]
}

// ---------------------------------------------------------------------------
// Single-phase frontier sum
// ---------------------------------------------------------------------------

fn build_frontier_algorithm(
    num_nodes: usize,
    coarse_chunk: usize,
    workers: usize,
) -> (Algorithm, Vec<u64>, Vec<u64>, u64, u32) {
    let (chunk_sums, expected_sum) = build_frontier_chunk_sums(num_nodes, coarse_chunk);

    let chunk_count = chunk_sums.len();

    let max_action_idx = workers + 1 + chunk_count;
    assert!(
        max_action_idx <= 0x1F_FFFF,
        "Action count {} exceeds 21-bit packet encoding limit (n={}, chunk={})",
        max_action_idx, num_nodes, coarse_chunk
    );

    let addend_base = align64(64);
    let checksum_addr = align64(addend_base + chunk_count * 8);
    let expected_sum_addr = checksum_addr + 8;
    let dynamic_end = align64(expected_sum_addr + 8);
    let mut payloads = vec![0u8; dynamic_end];

    payloads[checksum_addr..checksum_addr + 8].copy_from_slice(&0u64.to_le_bytes());
    payloads[expected_sum_addr..expected_sum_addr + 8].copy_from_slice(&expected_sum.to_le_bytes());

    let worker_base = (workers + 1) as u32;
    let mut actions = Vec::with_capacity(workers + 1 + chunk_count);

    let packets_per_worker = (chunk_count + workers - 1) / workers;
    for w in 0..workers {
        let start_idx = (w * packets_per_worker).min(chunk_count);
        let end_idx = ((w + 1) * packets_per_worker).min(chunk_count);
        let count = (end_idx - start_idx) as u32;
        if count == 0 {
            actions.push(Action {
                kind: Kind::AsyncDispatch,
                dst: u32::MAX,
                src: 0,
                offset: 0,
                size: 0,
            });
        } else {
            let start = worker_base + start_idx as u32;
            actions.push(Action {
                kind: Kind::AsyncDispatch,
                dst: MEMORY_UNIT_TYPE,
                src: start,
                offset: 0,
                size: count,
            });
        }
    }

    actions.push(Action {
        kind: Kind::WaitUntil,
        dst: checksum_addr as u32,
        src: expected_sum_addr as u32,
        offset: 0,
        size: 8,
    });

    for (idx, &sum) in chunk_sums.iter().enumerate() {
        payloads[addend_base + idx * 8..addend_base + idx * 8 + 8].copy_from_slice(&sum.to_le_bytes());
        actions.push(Action {
            kind: Kind::AtomicFetchAdd,
            dst: checksum_addr as u32,
            src: 0,
            offset: (addend_base + idx * 8) as u32,
            size: 8,
        });
    }

    let mut packets = Vec::with_capacity(workers);
    for w in 0..workers {
        let start_idx = (w * packets_per_worker).min(chunk_count);
        let end_idx = ((w + 1) * packets_per_worker).min(chunk_count);
        if start_idx < end_idx {
            let start = worker_base + start_idx as u32;
            let end = worker_base + end_idx as u32;
            packets.push(packed_packet(start, end, 0));
        }
    }

    let num_actions = actions.len();
    let mut memory_assignments = vec![0u8; num_actions];
    for w in 0..workers {
        memory_assignments[w] = w as u8;
    }

    let algorithm = Algorithm {
        actions,
        payloads,
        state: State {
            gpu_size: 0,
            file_buffer_size: 0,
            gpu_shader_offsets: vec![],
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            gpu_units: 0,
            file_units: 0,
            memory_units: workers,
            ffi_units: 0,
            cranelift_units: 0,
            backends_bits: 0,
        },
        memory_assignments,
        file_assignments: vec![],
        ffi_assignments: vec![],
        gpu_assignments: vec![],
        cranelift_assignments: vec![],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(256 * 1024),
        timeout_ms: Some(30_000),
        thread_name_prefix: Some("frontier-bench".to_string()),
    };

    (algorithm, packets, chunk_sums, expected_sum, worker_base)
}

fn rust_frontier_sum(packets: &[u64], chunk_sums: &[u64], worker_base: u32, workers: usize) -> u64 {
    let chunk_sums = Arc::new(chunk_sums.to_vec());
    let sum = Arc::new(AtomicU64::new(0));
    let (tx, rx) = unbounded::<u64>();

    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let rx = rx.clone();
        let chunk_sums = chunk_sums.clone();
        let sum = sum.clone();
        handles.push(thread::spawn(move || {
            while let Ok(packet) = rx.recv() {
                let (start, end, _flag) = decode_packet(packet);
                let mut idx = (start - worker_base) as usize;
                let limit = (end - worker_base) as usize;
                while idx < limit {
                    sum.fetch_add(chunk_sums[idx], Ordering::Relaxed);
                    idx += 1;
                }
            }
        }));
    }

    for &p in packets {
        let _ = tx.send(p);
    }
    drop(tx);

    for h in handles {
        let _ = h.join();
    }

    sum.load(Ordering::Relaxed)
}

fn bench_frontier(
    n: usize,
    chunk: usize,
    workers: usize,
    rounds: usize,
    label: &str,
) -> BenchResult {
    let (_, packets, chunk_sums, expected, worker_base) =
        build_frontier_algorithm(n, chunk, workers);
    let action_count = workers + 1 + chunk_sums.len();

    // Warmup
    let _ = rust_frontier_sum(&packets, &chunk_sums, worker_base, workers);
    let _ = base::execute(build_frontier_algorithm(n, chunk, workers).0);

    let mut rust_times = Vec::with_capacity(rounds);
    let mut base_times = Vec::with_capacity(rounds);
    let mut verified = true;

    for _ in 0..rounds {
        let t = std::time::Instant::now();
        let rust_got = rust_frontier_sum(&packets, &chunk_sums, worker_base, workers);
        rust_times.push(t.elapsed().as_secs_f64() * 1000.0);
        if rust_got != expected {
            eprintln!("WARNING: Rust checksum {} != expected {}", rust_got, expected);
            verified = false;
        }

        let (alg, _, _, _, _) = build_frontier_algorithm(n, chunk, workers);
        let t = std::time::Instant::now();
        let ok = base::execute(alg).is_ok();
        base_times.push(t.elapsed().as_secs_f64() * 1000.0);
        if !ok {
            eprintln!("WARNING: Base execution failed (n={}, chunk={})", n, chunk);
            verified = false;
        }
    }

    BenchResult {
        name: label.to_string(),
        python_ms: None,
        actions: Some(action_count),
        rust_ms: Some(median(&mut rust_times)),
        base_ms: median(&mut base_times),
        verified: Some(verified),
    }
}

// ---------------------------------------------------------------------------
// Multi-phase infrastructure
// ---------------------------------------------------------------------------

fn build_multi_phase_algorithm(phases: &[Vec<u64>], workers: usize) -> Algorithm {
    let num_phases = phases.len();
    let total_chunks: usize = phases.iter().map(|p| p.len()).sum();

    let addend_base = align64(64);
    let acc_base = align64(addend_base + total_chunks * 8);
    let exp_base = acc_base + num_phases * 8;
    let dynamic_end = align64(exp_base + num_phases * 8);
    let mut payloads = vec![0u8; dynamic_end];

    let mut addend_offset = addend_base;
    for (k, phase) in phases.iter().enumerate() {
        let mut phase_expected = 0u64;
        for &val in phase {
            payloads[addend_offset..addend_offset + 8].copy_from_slice(&val.to_le_bytes());
            addend_offset += 8;
            phase_expected = phase_expected.wrapping_add(val);
        }
        payloads[acc_base + k * 8..acc_base + k * 8 + 8].copy_from_slice(&0u64.to_le_bytes());
        payloads[exp_base + k * 8..exp_base + k * 8 + 8]
            .copy_from_slice(&phase_expected.to_le_bytes());
    }

    let main_section_len = num_phases * (workers + 1);
    let total_actions = main_section_len + total_chunks;
    let mut actions = Vec::with_capacity(total_actions);

    let mut phase_worker_starts = Vec::with_capacity(num_phases);
    let mut worker_action_idx = main_section_len;
    for phase in phases {
        phase_worker_starts.push(worker_action_idx);
        worker_action_idx += phase.len();
    }

    for (k, phase) in phases.iter().enumerate() {
        let chunk_count = phase.len();
        let base_idx = phase_worker_starts[k];
        let packets_per_worker = if chunk_count == 0 { 0 } else { (chunk_count + workers - 1) / workers };

        for w in 0..workers {
            let start_idx = (w * packets_per_worker).min(chunk_count);
            let end_idx = ((w + 1) * packets_per_worker).min(chunk_count);
            let count = (end_idx - start_idx) as u32;
            if count == 0 {
                // No work for this worker — dispatch to non-existent unit type
                // so the engine's `_ => {}` branch makes it a no-op.
                // (size=0 gets remapped to 1 in the engine, which would
                // corrupt the next phase's accumulator.)
                actions.push(Action {
                    kind: Kind::AsyncDispatch,
                    dst: u32::MAX,
                    src: 0,
                    offset: 0,
                    size: 0,
                });
            } else {
                let start = (base_idx + start_idx) as u32;
                actions.push(Action {
                    kind: Kind::AsyncDispatch,
                    dst: MEMORY_UNIT_TYPE,
                    src: start,
                    offset: 0,
                    size: count,
                });
            }
        }

        actions.push(Action {
            kind: Kind::WaitUntil,
            dst: (acc_base + k * 8) as u32,
            src: (exp_base + k * 8) as u32,
            offset: 0,
            size: 8,
        });
    }

    let mut addend_off = addend_base;
    for (k, phase) in phases.iter().enumerate() {
        let acc_addr = (acc_base + k * 8) as u32;
        for _ in 0..phase.len() {
            actions.push(Action {
                kind: Kind::AtomicFetchAdd,
                dst: acc_addr,
                src: 0,
                offset: addend_off as u32,
                size: 8,
            });
            addend_off += 8;
        }
    }

    assert_eq!(actions.len(), total_actions);

    let mut memory_assignments = vec![0u8; total_actions];
    for k in 0..num_phases {
        for w in 0..workers {
            let action_idx = k * (workers + 1) + w;
            memory_assignments[action_idx] = (w % workers) as u8;
        }
    }

    Algorithm {
        actions,
        payloads,
        state: State {
            gpu_size: 0,
            file_buffer_size: 0,
            gpu_shader_offsets: vec![],
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            gpu_units: 0,
            file_units: 0,
            memory_units: workers,
            ffi_units: 0,
            cranelift_units: 0,
            backends_bits: 0,
        },
        memory_assignments,
        file_assignments: vec![],
        ffi_assignments: vec![],
        gpu_assignments: vec![],
        cranelift_assignments: vec![],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(256 * 1024),
        timeout_ms: Some(30_000),
        thread_name_prefix: Some("multi-phase-bench".to_string()),
    }
}

fn rust_multi_phase_sum(phases: &[Vec<u64>], workers: usize) -> u64 {
    let sum = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(workers));
    let phases = Arc::new(phases.to_vec());

    let mut handles = Vec::with_capacity(workers);
    for w in 0..workers {
        let sum = sum.clone();
        let barrier = barrier.clone();
        let phases = phases.clone();
        handles.push(thread::spawn(move || {
            for phase in phases.iter() {
                let per_worker = if phase.is_empty() { 0 } else { (phase.len() + workers - 1) / workers };
                let start = (w * per_worker).min(phase.len());
                let end = ((w + 1) * per_worker).min(phase.len());
                for i in start..end {
                    sum.fetch_add(phase[i], Ordering::Relaxed);
                }
                barrier.wait();
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }

    sum.load(Ordering::Relaxed)
}

fn bench_multi_phase(
    phases: &[Vec<u64>],
    expected: u64,
    workers: usize,
    rounds: usize,
    label: &str,
) -> BenchResult {
    let alg = build_multi_phase_algorithm(phases, workers);
    let action_count = alg.actions.len();

    // Warmup
    let _ = rust_multi_phase_sum(phases, workers);
    let _ = base::execute(build_multi_phase_algorithm(phases, workers));

    let mut rust_times = Vec::with_capacity(rounds);
    let mut base_times = Vec::with_capacity(rounds);
    let mut verified = true;

    for _ in 0..rounds {
        let t = std::time::Instant::now();
        let rust_got = rust_multi_phase_sum(phases, workers);
        rust_times.push(t.elapsed().as_secs_f64() * 1000.0);
        if rust_got != expected {
            eprintln!("WARNING: Rust multi-phase sum {} != expected {}", rust_got, expected);
            verified = false;
        }

        let alg = build_multi_phase_algorithm(phases, workers);
        let t = std::time::Instant::now();
        let ok = base::execute(alg).is_ok();
        base_times.push(t.elapsed().as_secs_f64() * 1000.0);
        if !ok {
            eprintln!("WARNING: Base multi-phase execution failed");
            verified = false;
        }
    }

    BenchResult {
        name: label.to_string(),
        python_ms: None,
        actions: Some(action_count),
        rust_ms: Some(median(&mut rust_times)),
        base_ms: median(&mut base_times),
        verified: Some(verified),
    }
}

// ---------------------------------------------------------------------------
// Prefix sum (multi-phase reduction)
// ---------------------------------------------------------------------------

fn prefix_sum_phases(num_nodes: usize, chunk: usize, num_phases: usize) -> (Vec<Vec<u64>>, u64) {
    let (chunk_sums, expected) = build_frontier_chunk_sums(num_nodes, chunk);
    let per_phase = (chunk_sums.len() + num_phases - 1) / num_phases;
    let phases: Vec<Vec<u64>> = chunk_sums
        .chunks(per_phase)
        .map(|c| c.to_vec())
        .collect();
    (phases, expected)
}

fn bench_prefix_sum(
    n: usize,
    chunk: usize,
    num_phases: usize,
    workers: usize,
    rounds: usize,
    label: &str,
) -> BenchResult {
    let (phases, expected) = prefix_sum_phases(n, chunk, num_phases);
    bench_multi_phase(&phases, expected, workers, rounds, label)
}

// ---------------------------------------------------------------------------
// BFS levels (variable-size phases)
// ---------------------------------------------------------------------------

fn synthetic_bfs_profile(total_nodes: usize) -> Vec<usize> {
    let peak = total_nodes * 2 / 5;
    let mut levels = Vec::new();
    let mut frontier = 1usize;
    let mut visited = 0usize;

    while visited < total_nodes {
        let size = frontier.min(total_nodes - visited);
        levels.push(size);
        visited += size;
        if frontier < peak {
            frontier = (frontier * 6).min(peak);
        } else {
            frontier = (frontier / 3).max(1);
        }
    }
    levels
}

fn bfs_level_phases(num_nodes: usize, chunk: usize) -> (Vec<Vec<u64>>, u64) {
    let level_sizes = synthetic_bfs_profile(num_nodes);
    let mut phases = Vec::with_capacity(level_sizes.len());
    let mut total_expected = 0u64;
    let mut node_offset = 0usize;

    for &level_size in &level_sizes {
        let mut costs = Vec::with_capacity(level_size);
        for i in 0..level_size {
            costs.push(synthetic_degree(node_offset + i));
        }
        node_offset += level_size;

        let mut chunk_sums = Vec::new();
        let mut pos = 0;
        while pos < costs.len() {
            let end = (pos + chunk).min(costs.len());
            let s: u64 = costs[pos..end].iter().copied().sum();
            chunk_sums.push(s);
            total_expected = total_expected.wrapping_add(s);
            pos = end;
        }
        phases.push(chunk_sums);
    }

    (phases, total_expected)
}

fn bench_bfs_levels(
    n: usize,
    chunk: usize,
    workers: usize,
    rounds: usize,
    label: &str,
) -> BenchResult {
    let (phases, expected) = bfs_level_phases(n, chunk);
    bench_multi_phase(&phases, expected, workers, rounds, label)
}

// ---------------------------------------------------------------------------
// Kernel queue dispatch (KernelStart/Submit/Wait/Stop)
// ---------------------------------------------------------------------------

const QUEUE_MASK_OFF: usize = 8;
const QUEUE_BASE_OFF: usize = 12;
const QUEUE_DESC_SIZE: usize = 24;
const KERNEL_DESC_QUEUE_OFF: usize = 0;
const KERNEL_DESC_UNIT_TYPE_OFF: usize = 4;
const KERNEL_DESC_UNIT_ID_OFF: usize = 8;
const KERNEL_DESC_PROGRESS_OFF: usize = 16;
const KERNEL_DESC_THREAD_COUNT_OFF: usize = 20;
const KERNEL_DESC_SIZE: usize = 24;
const KERNEL_KIND_QUEUE_ROUTER: u32 = 1;

fn build_kernel_queue_algorithm(
    num_nodes: usize,
    coarse_chunk: usize,
    workers: usize,
) -> (Algorithm, Vec<u64>, Vec<u64>, u64, u32) {
    let (chunk_sums, expected_sum) = build_frontier_chunk_sums(num_nodes, coarse_chunk);
    let chunk_count = chunk_sums.len();
    let packets_per_worker = (chunk_count + workers - 1) / workers;

    let main_action_count: usize = 5;
    let worker_base = main_action_count as u32;

    assert!(
        (worker_base as usize + chunk_count) <= 0x1F_FFFF,
        "Action count exceeds 21-bit packet encoding limit"
    );

    // Build packets (same split as frontier benchmark)
    let mut packets = Vec::new();
    for w in 0..workers {
        let start_idx = (w * packets_per_worker).min(chunk_count);
        let end_idx = ((w + 1) * packets_per_worker).min(chunk_count);
        if start_idx < end_idx {
            packets.push(packed_packet(
                worker_base + start_idx as u32,
                worker_base + end_idx as u32,
                0,
            ));
        }
    }
    let num_packets = packets.len();
    let queue_capacity = num_packets.next_power_of_two().max(16);

    // --- Memory layout ---
    let queue_desc_addr = align64(64);
    let queue_slots_addr = queue_desc_addr + QUEUE_DESC_SIZE;
    let kernel_desc_addr = align64(queue_slots_addr + queue_capacity * 8);
    let handle_id_addr = align64(kernel_desc_addr + KERNEL_DESC_SIZE);
    let start_status_addr = align64(handle_id_addr + 8);
    let submit_status_addr = start_status_addr + 8;
    let wait_status_addr = submit_status_addr + 8;
    let stop_status_addr = wait_status_addr + 8;
    let progress_addr = stop_status_addr + 8;
    let packet_data_addr = align64(progress_addr + 8);
    let addend_base = align64(packet_data_addr + num_packets * 8);
    let checksum_addr = align64(addend_base + chunk_count * 8);
    let expected_sum_addr = checksum_addr + 8;
    let dynamic_end = align64(expected_sum_addr + 8);

    let mut payloads = vec![0u8; dynamic_end];

    // Queue descriptor (head, tail, reserve start at 0 — already zeroed)
    payloads[queue_desc_addr + QUEUE_MASK_OFF..][..4]
        .copy_from_slice(&((queue_capacity - 1) as u32).to_le_bytes());
    payloads[queue_desc_addr + QUEUE_BASE_OFF..][..4]
        .copy_from_slice(&(queue_slots_addr as u32).to_le_bytes());

    // Kernel descriptor
    payloads[kernel_desc_addr + KERNEL_DESC_QUEUE_OFF..][..4]
        .copy_from_slice(&(queue_desc_addr as u32).to_le_bytes());
    payloads[kernel_desc_addr + KERNEL_DESC_UNIT_TYPE_OFF..][..4]
        .copy_from_slice(&MEMORY_UNIT_TYPE.to_le_bytes());
    payloads[kernel_desc_addr + KERNEL_DESC_UNIT_ID_OFF..][..4]
        .copy_from_slice(&(u32::MAX - 1).to_le_bytes()); // Pool (round-robin)
    // stop_flag = 0 (already zeroed)
    payloads[kernel_desc_addr + KERNEL_DESC_PROGRESS_OFF..][..4]
        .copy_from_slice(&(progress_addr as u32).to_le_bytes());
    payloads[kernel_desc_addr + KERNEL_DESC_THREAD_COUNT_OFF..][..4]
        .copy_from_slice(&(workers as u32).to_le_bytes());

    // Packet data
    for (i, &pkt) in packets.iter().enumerate() {
        let off = packet_data_addr + i * 8;
        payloads[off..off + 8].copy_from_slice(&pkt.to_le_bytes());
    }

    // Checksum and expected
    payloads[checksum_addr..checksum_addr + 8].copy_from_slice(&0u64.to_le_bytes());
    payloads[expected_sum_addr..expected_sum_addr + 8]
        .copy_from_slice(&expected_sum.to_le_bytes());

    // --- Actions ---
    let total_actions = main_action_count + chunk_count;
    let mut actions = Vec::with_capacity(total_actions);

    // 0: KernelStart
    actions.push(Action {
        kind: Kind::KernelStart,
        dst: kernel_desc_addr as u32,
        src: handle_id_addr as u32,
        offset: start_status_addr as u32,
        size: KERNEL_KIND_QUEUE_ROUTER,
    });

    // 1: KernelSubmit
    actions.push(Action {
        kind: Kind::KernelSubmit,
        dst: handle_id_addr as u32,
        src: packet_data_addr as u32,
        offset: submit_status_addr as u32,
        size: num_packets as u32,
    });

    // 2: KernelWait (wait for all packets drained)
    actions.push(Action {
        kind: Kind::KernelWait,
        dst: handle_id_addr as u32,
        src: num_packets as u32,
        offset: wait_status_addr as u32,
        size: 30_000,
    });

    // 3: WaitUntil (checksum == expected)
    actions.push(Action {
        kind: Kind::WaitUntil,
        dst: checksum_addr as u32,
        src: expected_sum_addr as u32,
        offset: 0,
        size: 8,
    });

    // 4: KernelStop
    actions.push(Action {
        kind: Kind::KernelStop,
        dst: handle_id_addr as u32,
        src: 0,
        offset: stop_status_addr as u32,
        size: 0,
    });

    // 5..5+chunk_count: AtomicFetchAdd (worker actions dispatched via queue)
    for (idx, &sum) in chunk_sums.iter().enumerate() {
        payloads[addend_base + idx * 8..addend_base + idx * 8 + 8]
            .copy_from_slice(&sum.to_le_bytes());
        actions.push(Action {
            kind: Kind::AtomicFetchAdd,
            dst: checksum_addr as u32,
            src: 0,
            offset: (addend_base + idx * 8) as u32,
            size: 8,
        });
    }

    let mut memory_assignments = vec![0u8; total_actions];
    for i in 0..chunk_count {
        memory_assignments[main_action_count + i] = (i % workers) as u8;
    }

    let algorithm = Algorithm {
        actions,
        payloads,
        state: State {
            gpu_size: 0,
            file_buffer_size: 0,
            gpu_shader_offsets: vec![],
            cranelift_ir_offsets: vec![],
        },
        units: UnitSpec {
            gpu_units: 0,
            file_units: 0,
            memory_units: workers,
            ffi_units: 0,
            cranelift_units: 0,
            backends_bits: 0,
        },
        memory_assignments,
        file_assignments: vec![],
        ffi_assignments: vec![],
        gpu_assignments: vec![],
        cranelift_assignments: vec![],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(256 * 1024),
        timeout_ms: Some(30_000),
        thread_name_prefix: Some("kernel-queue-bench".to_string()),
    };

    (algorithm, packets, chunk_sums, expected_sum, worker_base)
}

fn bench_kernel_queue(
    n: usize,
    chunk: usize,
    workers: usize,
    rounds: usize,
    label: &str,
) -> BenchResult {
    let (_, packets, chunk_sums, expected, worker_base) =
        build_kernel_queue_algorithm(n, chunk, workers);
    let action_count = 5 + chunk_sums.len();

    // Warmup
    let _ = rust_frontier_sum(&packets, &chunk_sums, worker_base, workers);
    let _ = base::execute(build_kernel_queue_algorithm(n, chunk, workers).0);

    let mut rust_times = Vec::with_capacity(rounds);
    let mut base_times = Vec::with_capacity(rounds);
    let mut verified = true;

    for _ in 0..rounds {
        let t = std::time::Instant::now();
        let rust_got = rust_frontier_sum(&packets, &chunk_sums, worker_base, workers);
        rust_times.push(t.elapsed().as_secs_f64() * 1000.0);
        if rust_got != expected {
            eprintln!("WARNING: Rust checksum {} != expected {}", rust_got, expected);
            verified = false;
        }

        let (alg, _, _, _, _) = build_kernel_queue_algorithm(n, chunk, workers);
        let t = std::time::Instant::now();
        let ok = base::execute(alg).is_ok();
        base_times.push(t.elapsed().as_secs_f64() * 1000.0);
        if !ok {
            eprintln!("WARNING: Base kernel queue failed (n={}, chunk={})", n, chunk);
            verified = false;
        }
    }

    BenchResult {
        name: label.to_string(),
        python_ms: None,
        actions: Some(action_count),
        rust_ms: Some(median(&mut rust_times)),
        base_ms: median(&mut base_times),
        verified: Some(verified),
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub struct Config {
    pub profile: String,
    pub rounds: usize,
    pub chunk: usize,
    pub workers: usize,
}

pub fn run(cfg: &Config) -> Vec<BenchResult> {
    let mut results = Vec::new();

    println!(
        "Workers: {}, Chunk: {}, Rounds: {}, Profile: {}",
        cfg.workers, cfg.chunk, cfg.rounds, cfg.profile
    );

    // --- Frontier sum: size sweep ---
    let sizes: &[usize] = match cfg.profile.as_str() {
        "quick" => &[1_000_000, 5_000_000],
        "full" => &[500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000],
        _ => &[1_000_000, 2_000_000, 5_000_000, 10_000_000],
    };

    for &n in sizes {
        let label = format!("n={} c={}", format_count(n), format_count(cfg.chunk));
        results.push(bench_frontier(n, cfg.chunk, cfg.workers, cfg.rounds, &label));
    }

    // --- Chunk sweep at fixed size (full profile only) ---
    if cfg.profile == "full" {
        let sweep_n: usize = 10_000_000;
        let chunks: &[usize] = &[256, 1024, 4096, 16384, 65536];
        for &c in chunks {
            if c == cfg.chunk {
                continue;
            }
            let label = format!("n={} c={}", format_count(sweep_n), format_count(c));
            results.push(bench_frontier(sweep_n, c, cfg.workers, cfg.rounds, &label));
        }
    }

    // --- Multi-phase prefix sum ---
    println!("\n--- Multi-phase reduction (prefix-sum style) ---");
    let phase_n: usize = match cfg.profile.as_str() {
        "quick" => 5_000_000,
        "full" => 10_000_000,
        _ => 5_000_000,
    };
    let phase_counts: &[usize] = match cfg.profile.as_str() {
        "quick" => &[1, 10],
        "full" => &[1, 5, 10, 20, 50],
        _ => &[1, 10, 20],
    };
    for &k in phase_counts {
        let label = format!("phase k={} n={}", k, format_count(phase_n));
        results.push(bench_prefix_sum(phase_n, cfg.chunk, k, cfg.workers, cfg.rounds, &label));
    }

    // --- BFS-level dispatch ---
    println!("\n--- BFS-level dispatch (variable frontier sizes) ---");
    let bfs_sizes: &[usize] = match cfg.profile.as_str() {
        "quick" => &[1_000_000],
        "full" => &[1_000_000, 5_000_000, 10_000_000],
        _ => &[1_000_000, 5_000_000],
    };
    for &n in bfs_sizes {
        let label = format!("bfs n={} c={}", format_count(n), format_count(cfg.chunk));
        results.push(bench_bfs_levels(n, cfg.chunk, cfg.workers, cfg.rounds, &label));
    }

    // --- Kernel queue dispatch ---
    println!("\n--- Kernel queue dispatch (KernelStart/Submit/Wait/Stop) ---");
    let kq_sizes: &[usize] = match cfg.profile.as_str() {
        "quick" => &[1_000_000, 5_000_000],
        "full" => &[1_000_000, 5_000_000, 10_000_000, 20_000_000],
        _ => &[1_000_000, 5_000_000, 10_000_000],
    };
    for &n in kq_sizes {
        let label = format!("kq n={} c={}", format_count(n), format_count(cfg.chunk));
        results.push(bench_kernel_queue(n, cfg.chunk, cfg.workers, cfg.rounds, &label));
    }

    results
}
