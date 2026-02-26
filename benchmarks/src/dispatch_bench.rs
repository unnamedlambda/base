use base_types::{Action, Kind};
use crossbeam_channel::unbounded;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use crate::harness::BenchResult;

fn align64(v: usize) -> usize {
    (v + 63) & !63
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

fn packed_packet(start: u32, end: u32, flag: u32) -> u64 {
    ((start as u64) << 43) | ((end as u64) << 22) | (flag as u64)
}

fn decode_packet(packet: u64) -> (u32, u32, u32) {
    let start = (packet >> 43) as u32;
    let end = ((packet >> 22) & 0x1F_FFFF) as u32;
    let flag = (packet & 0x3F_FFFF) as u32;
    (start, end, flag)
}

// ---------------------------------------------------------------------------
// CLIF IR: raw atomic_rmw, no FFI
// ---------------------------------------------------------------------------

/// Generate CLIF IR for the dispatch bench.
///
/// fn0: unused placeholder
/// fn1: atomic-add worker — reads addend + relative acc offset from descriptor,
///      computes absolute accumulator address, does atomic_rmw add.
fn gen_dispatch_clif_ir() -> String {
    // Each Describe action has dst = descriptor offset.
    // Worker passes ptr = base + desc.dst to the function.
    //
    // Descriptor layout (16 bytes):
    //   +0: addend (u64)
    //   +8: signed relative offset from this descriptor to the accumulator (i64)
    //
    // The function computes: acc_ptr = ptr + rel_offset, then atomic adds.
    r#"function u0:0(i64) system_v {
block0(v0: i64):
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    v1 = load.i64 notrap aligned v0
    v2 = load.i64 notrap aligned v0+8
    v3 = iadd v0, v2
    v4 = atomic_rmw.i64 little add v3, v1
    return
}"#
    .to_string()
}

// ---------------------------------------------------------------------------
// Single-phase frontier sum (CLIF + AsyncDispatch)
// ---------------------------------------------------------------------------

fn build_frontier_algorithm(
    num_nodes: usize,
    coarse_chunk: usize,
    workers: usize,
) -> (base::BaseConfig, base::Algorithm, u64) {
    let (chunk_sums, expected_sum) = build_frontier_chunk_sums(num_nodes, coarse_chunk);
    let chunk_count = chunk_sums.len();

    // Memory layout:
    //   0-63:          reserved (HT context etc)
    //   64:            accumulator (u64, init 0)
    //   72:            expected_sum (u64)
    //   80..80+W*8:    per-dispatch flags
    //   256:           CLIF IR (null-terminated)
    //   desc_base:     descriptor array, 16 bytes each
    let acc_addr = 64usize;
    let expected_addr = 72usize;
    let flags_base = 80usize;
    let clif_off = 256usize;

    let clif_ir = gen_dispatch_clif_ir();
    let clif_bytes = format!("{}\0", clif_ir);
    let desc_base = align64(clif_off + clif_bytes.len());
    let desc_end = desc_base + chunk_count * 16;
    let payload_size = align64(desc_end);

    let mut payloads = vec![0u8; payload_size];

    payloads[acc_addr..acc_addr + 8].copy_from_slice(&0u64.to_le_bytes());
    payloads[expected_addr..expected_addr + 8].copy_from_slice(&expected_sum.to_le_bytes());
    payloads[clif_off..clif_off + clif_bytes.len()].copy_from_slice(clif_bytes.as_bytes());

    for (i, &sum) in chunk_sums.iter().enumerate() {
        let desc_off = desc_base + i * 16;
        payloads[desc_off..desc_off + 8].copy_from_slice(&sum.to_le_bytes());
        let acc_rel = (acc_addr as i64) - (desc_off as i64);
        payloads[desc_off + 8..desc_off + 16].copy_from_slice(&acc_rel.to_le_bytes());
    }

    // Actions:
    //   [0..W-1]:          ClifCallAsync (one per worker)
    //   [W]:               WaitUntil (acc == expected)
    //   [W+1..W+1+chunks]: Describe worker actions (dispatched by ClifCallAsync)
    let worker_base_action = (workers + 1) as u32;
    let mut actions = Vec::with_capacity(workers + 1 + chunk_count);

    let chunks_per_worker = (chunk_count + workers - 1) / workers;
    for w in 0..workers {
        let start_idx = (w * chunks_per_worker).min(chunk_count);
        let end_idx = ((w + 1) * chunks_per_worker).min(chunk_count);
        let count = (end_idx - start_idx) as u32;
        let start_action = worker_base_action + start_idx as u32;
        let flag_addr = (flags_base + w * 8) as u32;

        if count == 0 {
            actions.push(Action {
                kind: Kind::Describe,
                dst: 0,
                src: 0,
                offset: 0,
                size: 0,
            });
        } else {
            actions.push(Action {
                kind: Kind::ClifCallAsync,
                dst: w as u32,
                src: start_action,
                offset: flag_addr,
                size: count,
            });
        }
    }

    actions.push(Action {
        kind: Kind::WaitUntil,
        dst: acc_addr as u32,
        src: expected_addr as u32,
        offset: 0,
        size: 8,
    });

    for i in 0..chunk_count {
        let desc_off = desc_base + i * 16;
        actions.push(Action {
            kind: Kind::Describe,
            dst: desc_off as u32,
            src: 1,
            offset: 0,
            size: 0,
        });
    }

    let config = base::BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: payloads.len(),
        context_offset: 0,
    };
    let algorithm = base::Algorithm {
        actions,
        payloads,
        cranelift_units: workers,
        timeout_ms: Some(30_000),
        output: vec![],
    };

    (config, algorithm, expected_sum)
}

// ---------------------------------------------------------------------------
// Rust baseline: crossbeam channel + atomic
// ---------------------------------------------------------------------------

fn rust_frontier_sum(
    packets: &[u64],
    chunk_sums: &[u64],
    worker_base: u32,
    workers: usize,
) -> u64 {
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
    let (chunk_sums_vec, expected) = build_frontier_chunk_sums(n, chunk);
    let chunk_count = chunk_sums_vec.len();
    let action_count = workers + 1 + chunk_count;

    // Build packets for Rust baseline
    let chunks_per_worker = (chunk_count + workers - 1) / workers;
    let worker_base = 0u32;
    let mut packets = Vec::new();
    for w in 0..workers {
        let start_idx = (w * chunks_per_worker).min(chunk_count);
        let end_idx = ((w + 1) * chunks_per_worker).min(chunk_count);
        if start_idx < end_idx {
            packets.push(packed_packet(
                worker_base + start_idx as u32,
                worker_base + end_idx as u32,
                0,
            ));
        }
    }

    // Warmup
    let _ = rust_frontier_sum(&packets, &chunk_sums_vec, worker_base, workers);
    {
        let (cfg, alg, _) = build_frontier_algorithm(n, chunk, workers);
        let _ = base::run(cfg, alg);
    }

    let mut rust_times = Vec::with_capacity(rounds);
    let mut base_times = Vec::with_capacity(rounds);
    let mut verified = true;

    for _ in 0..rounds {
        let t = std::time::Instant::now();
        let rust_got = rust_frontier_sum(&packets, &chunk_sums_vec, worker_base, workers);
        rust_times.push(t.elapsed().as_secs_f64() * 1000.0);
        if rust_got != expected {
            eprintln!("WARNING: Rust checksum {} != expected {}", rust_got, expected);
            verified = false;
        }

        let (cfg, alg, _) = build_frontier_algorithm(n, chunk, workers);
        let t = std::time::Instant::now();
        let ok = base::run(cfg, alg).is_ok();
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

fn build_multi_phase_algorithm(phases: &[Vec<u64>], workers: usize) -> (base::BaseConfig, base::Algorithm) {
    let num_phases = phases.len();
    let total_chunks: usize = phases.iter().map(|p| p.len()).sum();

    let acc_base_addr = 64usize;
    let exp_base_addr = acc_base_addr + num_phases * 8;
    let flags_base = align64(exp_base_addr + num_phases * 8);
    let clif_off = align64(flags_base + workers * 8);

    let clif_ir = gen_dispatch_clif_ir();
    let clif_bytes = format!("{}\0", clif_ir);
    let desc_base = align64(clif_off + clif_bytes.len());
    let desc_end = desc_base + total_chunks * 16;
    let payload_size = align64(desc_end);

    let mut payloads = vec![0u8; payload_size];
    payloads[clif_off..clif_off + clif_bytes.len()].copy_from_slice(clif_bytes.as_bytes());

    // Fill accumulator and expected values per phase
    let mut phase_expected = Vec::with_capacity(num_phases);
    for (k, phase) in phases.iter().enumerate() {
        let mut pe = 0u64;
        for &val in phase {
            pe = pe.wrapping_add(val);
        }
        phase_expected.push(pe);
        payloads[acc_base_addr + k * 8..acc_base_addr + k * 8 + 8]
            .copy_from_slice(&0u64.to_le_bytes());
        payloads[exp_base_addr + k * 8..exp_base_addr + k * 8 + 8]
            .copy_from_slice(&pe.to_le_bytes());
    }

    // Fill descriptors
    let mut desc_idx = 0usize;
    for phase in phases {
        for &val in phase {
            let desc_off = desc_base + desc_idx * 16;
            payloads[desc_off..desc_off + 8].copy_from_slice(&val.to_le_bytes());
            // acc_rel for this chunk's phase — we need to know which phase this chunk belongs to
            // We'll fix this below after building actions
            desc_idx += 1;
        }
    }

    // Actions layout:
    //   main section: num_phases * (workers + 1) actions
    //     per phase: W ClifCallAsyncs + 1 WaitUntil
    //   worker section: total_chunks Describe actions
    let main_section_len = num_phases * (workers + 1);
    let total_actions = main_section_len + total_chunks;
    let mut actions = Vec::with_capacity(total_actions);

    // Track where each phase's worker actions start
    let mut phase_worker_starts = Vec::with_capacity(num_phases);
    let mut worker_action_idx = main_section_len;
    for phase in phases {
        phase_worker_starts.push(worker_action_idx);
        worker_action_idx += phase.len();
    }

    // Main section: dispatches + waits
    for (k, phase) in phases.iter().enumerate() {
        let chunk_count = phase.len();
        let base_action_idx = phase_worker_starts[k];
        let chunks_per_worker = if chunk_count == 0 {
            0
        } else {
            (chunk_count + workers - 1) / workers
        };

        for w in 0..workers {
            let start_idx = (w * chunks_per_worker).min(chunk_count);
            let end_idx = ((w + 1) * chunks_per_worker).min(chunk_count);
            let count = (end_idx - start_idx) as u32;
            if count == 0 {
                actions.push(Action {
                    kind: Kind::Describe,
                    dst: 0,
                    src: 0,
                    offset: 0,
                    size: 0,
                });
            } else {
                let start = (base_action_idx + start_idx) as u32;
                let flag_addr = (flags_base + w * 8) as u32;
                actions.push(Action {
                    kind: Kind::ClifCallAsync,
                    dst: w as u32,
                    src: start,
                    offset: flag_addr,
                    size: count,
                });
            }
        }

        actions.push(Action {
            kind: Kind::WaitUntil,
            dst: (acc_base_addr + k * 8) as u32,
            src: (exp_base_addr + k * 8) as u32,
            offset: 0,
            size: 8,
        });
    }

    // Worker section: Describe actions with descriptors
    // Also fix up descriptor acc_rel offsets now that we know positions
    desc_idx = 0;
    for (k, phase) in phases.iter().enumerate() {
        let acc_addr = acc_base_addr + k * 8;
        for _ in 0..phase.len() {
            let desc_off = desc_base + desc_idx * 16;
            let acc_rel = (acc_addr as i64) - (desc_off as i64);
            payloads[desc_off + 8..desc_off + 16].copy_from_slice(&acc_rel.to_le_bytes());
            actions.push(Action {
                kind: Kind::Describe,
                dst: desc_off as u32,
                src: 1,
                offset: 0,
                size: 0,
            });
            desc_idx += 1;
        }
    }

    assert_eq!(actions.len(), total_actions);

    let config = base::BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: payloads.len(),
        context_offset: 0,
    };
    let algorithm = base::Algorithm {
        actions,
        payloads,
        cranelift_units: workers,
        timeout_ms: Some(30_000),
        output: vec![],
    };
    (config, algorithm)
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
                let per_worker = if phase.is_empty() {
                    0
                } else {
                    (phase.len() + workers - 1) / workers
                };
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
    let (_, alg_for_count) = build_multi_phase_algorithm(phases, workers);
    let action_count = alg_for_count.actions.len();

    // Warmup
    let _ = rust_multi_phase_sum(phases, workers);
    {
        let (cfg, alg) = build_multi_phase_algorithm(phases, workers);
        let _ = base::run(cfg, alg);
    }

    let mut rust_times = Vec::with_capacity(rounds);
    let mut base_times = Vec::with_capacity(rounds);
    let mut verified = true;

    for _ in 0..rounds {
        let t = std::time::Instant::now();
        let rust_got = rust_multi_phase_sum(phases, workers);
        rust_times.push(t.elapsed().as_secs_f64() * 1000.0);
        if rust_got != expected {
            eprintln!(
                "WARNING: Rust multi-phase sum {} != expected {}",
                rust_got, expected
            );
            verified = false;
        }

        let (cfg, alg) = build_multi_phase_algorithm(phases, workers);
        let t = std::time::Instant::now();
        let ok = base::run(cfg, alg).is_ok();
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
    let phases: Vec<Vec<u64>> = chunk_sums.chunks(per_phase).map(|c| c.to_vec()).collect();
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
// Table printer
// ---------------------------------------------------------------------------

pub fn print_dispatch_table(results: &[BenchResult]) {
    let name_w = 26;
    let col_w = 14;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>8} {:>6}",
        "Dispatch",
        "Rust",
        "Base (CLIF)",
        "Actions",
        "Check",
        name_w = name_w,
        col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 2 + 8 + 6 + 4));

    for r in results {
        let rust_str = match r.rust_ms {
            Some(ms) => format!("{:.1}ms", ms),
            None => "N/A".to_string(),
        };
        let base_str = format!("{:.1}ms", r.base_ms);
        let actions_str = match r.actions {
            Some(a) => format!("{}", a),
            None => "—".to_string(),
        };
        let check_str = match r.verified {
            Some(true) => "\u{2713}",
            Some(false) => "\u{2717}",
            None => "\u{2014}",
        };

        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>8} {:>6}",
            r.name,
            rust_str,
            base_str,
            actions_str,
            check_str,
            name_w = name_w,
            col_w = col_w
        );
    }
    println!();
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
        "full" => &[
            500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000,
            100_000_000,
        ],
        _ => &[1_000_000, 2_000_000, 5_000_000, 10_000_000],
    };

    for &n in sizes {
        let label = format!("n={} c={}", format_count(n), format_count(cfg.chunk));
        results.push(bench_frontier(n, cfg.chunk, cfg.workers, cfg.rounds, &label));
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
        results.push(bench_prefix_sum(
            phase_n,
            cfg.chunk,
            k,
            cfg.workers,
            cfg.rounds,
            &label,
        ));
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
        results.push(bench_bfs_levels(
            n,
            cfg.chunk,
            cfg.workers,
            cfg.rounds,
            &label,
        ));
    }

    results
}
