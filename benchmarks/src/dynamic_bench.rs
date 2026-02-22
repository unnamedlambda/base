use base::Algorithm;
use base_types::{Action, Kind, State, UnitSpec};
use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;

use crate::harness::BenchResult;

fn align64(v: usize) -> usize {
    (v + 63) & !63
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn synthetic_degree(node: usize) -> u64 {
    let base = 1 + (1024 / (1 + (node % 1024))) as u64;
    let jitter = ((node as u64).wrapping_mul(1_103_515_245).wrapping_add(12_345) % 11) as u64;
    let hotspot = if node % 97 == 0 { 32 } else { 1 };
    (base + jitter) * hotspot
}

// =========================================================================
// Problem 1: Divide-and-conquer parallel sum
//
// Recursively split an array in half, spawn a thread for the left half,
// process the right half in the current thread. When below threshold,
// sum sequentially. Thread count is data-dependent.
// =========================================================================

// --- Rust baseline ---

fn rust_recursive_sum(data: &Arc<Vec<u64>>, start: usize, end: usize, threshold: usize) -> u64 {
    let count = end - start;
    if count <= threshold {
        data[start..end].iter().sum()
    } else {
        let mid = (start + end) / 2;
        let data_l = data.clone();
        let handle = thread::spawn(move || rust_recursive_sum(&data_l, start, mid, threshold));
        let right = rust_recursive_sum(data, mid, end, threshold);
        let left = handle.join().unwrap();
        left + right
    }
}

// --- Base (CLIF) ---

const NODE_SIZE: usize = 128;

fn gen_recursive_clif_ir() -> String {
    r#"function u0:0(i64) system_v {
block0(v0: i64):
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    return
}

function u0:2(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64, i64) -> i64 system_v
    fn2 = %cl_thread_join sig2
    sig3 = (i64) system_v
    fn3 = %cl_thread_cleanup sig3
block0(v0: i64):
    ; v0 = orchestrator control block: +0 root_node_rel, +8 thread_ctx
    v1 = load.i64 notrap aligned v0
    v2 = iadd v0, v1
    v3 = iadd_imm v0, 8
    call fn0(v3)
    v4 = iconst.i64 3
    v5 = call fn1(v3, v4, v2)
    v6 = call fn2(v3, v5)
    call fn3(v3)
    return
}

function u0:3(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64, i64) -> i64 system_v
    fn2 = %cl_thread_join sig2
    sig3 = (i64) system_v
    fn3 = %cl_thread_cleanup sig3
    fn4 = %cl_thread_call sig1
block0(v0: i64):
    ; Node layout: +0 data_ptr, +8 start, +16 end, +24 threshold, +32 result
    ;              +40 thread_ctx, +48 left_child_rel, +56 right_child_rel
    v1 = load.i64 notrap aligned v0
    v2 = load.i64 notrap aligned v0+8
    v3 = load.i64 notrap aligned v0+16
    v4 = load.i64 notrap aligned v0+24
    v5 = isub v3, v2
    v6 = icmp ule v5, v4
    brif v6, block1(v1, v2, v3), block4(v1, v2, v3, v4)

block1(v10: i64, v11: i64, v12: i64):
    ; Leaf: sequential sum
    v13 = iconst.i64 0
    v14 = icmp eq v11, v12
    brif v14, block3(v13), block2(v10, v11, v12, v13)

block2(v20: i64, v21: i64, v22: i64, v23: i64):
    v24 = ishl_imm v21, 3
    v25 = iadd v20, v24
    v26 = load.i64 notrap aligned v25
    v27 = iadd v23, v26
    v28 = iadd_imm v21, 1
    v29 = icmp eq v28, v22
    brif v29, block3(v27), block2(v20, v28, v22, v27)

block3(v30: i64):
    store.i64 notrap aligned v30, v0+32
    return

block4(v40: i64, v41: i64, v42: i64, v43: i64):
    ; Internal node: fork left child, call right child, join
    v44 = iadd v41, v42
    v45 = ushr_imm v44, 1
    v46 = load.i64 notrap aligned v0+48
    v47 = load.i64 notrap aligned v0+56
    v48 = iadd v0, v46
    v49 = iadd v0, v47
    ; Write left child descriptor
    store.i64 notrap aligned v40, v48
    store.i64 notrap aligned v41, v48+8
    store.i64 notrap aligned v45, v48+16
    store.i64 notrap aligned v43, v48+24
    ; Write right child descriptor
    store.i64 notrap aligned v40, v49
    store.i64 notrap aligned v45, v49+8
    store.i64 notrap aligned v42, v49+16
    store.i64 notrap aligned v43, v49+24
    ; Fork left, call right inline
    v52 = iadd_imm v0, 40
    call fn0(v52)
    v53 = iconst.i64 3
    v54 = call fn1(v52, v53, v48)
    v55 = call fn4(v52, v53, v49)
    v56 = call fn2(v52, v54)
    call fn3(v52)
    ; Sum children results
    v58 = load.i64 notrap aligned v48+32
    v59 = load.i64 notrap aligned v49+32
    v60 = iadd v58, v59
    store.i64 notrap aligned v60, v0+32
    return
}"#
    .to_string()
}

fn build_recursive_algorithm(data: &[u64], threshold: usize) -> (Algorithm, u64) {
    let n = data.len();
    let expected: u64 = data.iter().sum();

    struct NodeInfo {
        start: usize,
        end: usize,
        is_leaf: bool,
    }

    let mut nodes = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back((0usize, n));
    while let Some((start, end)) = queue.pop_front() {
        let count = end - start;
        if count <= threshold {
            nodes.push(NodeInfo { start, end, is_leaf: true });
        } else {
            let mid = (start + end) / 2;
            nodes.push(NodeInfo { start, end, is_leaf: false });
            queue.push_back((start, mid));
            queue.push_back((mid, end));
        }
    }
    let num_nodes = nodes.len();

    let flag_addr = 64usize;
    let exp_addr = 56usize;
    let ctrl_off = 72usize;
    let nodes_off = align64(ctrl_off + 16);
    let nodes_end = nodes_off + num_nodes * NODE_SIZE;
    let clif_off = align64(nodes_end);

    let clif_ir = gen_recursive_clif_ir();
    let clif_bytes = format!("{}\0", clif_ir);
    let payload_size = align64(clif_off + clif_bytes.len());

    let mut payloads = vec![0u8; payload_size];
    payloads[clif_off..clif_off + clif_bytes.len()].copy_from_slice(clif_bytes.as_bytes());
    payloads[exp_addr..exp_addr + 8].copy_from_slice(&expected.to_le_bytes());

    let root_rel = (nodes_off as i64) - (ctrl_off as i64);
    payloads[ctrl_off..ctrl_off + 8].copy_from_slice(&root_rel.to_le_bytes());

    let data_ptr = data.as_ptr() as u64;

    for (i, node) in nodes.iter().enumerate() {
        let node_off = nodes_off + i * NODE_SIZE;
        payloads[node_off..node_off + 8].copy_from_slice(&data_ptr.to_le_bytes());
        payloads[node_off + 8..node_off + 16].copy_from_slice(&(node.start as u64).to_le_bytes());
        payloads[node_off + 16..node_off + 24].copy_from_slice(&(node.end as u64).to_le_bytes());
        payloads[node_off + 24..node_off + 32].copy_from_slice(&(threshold as u64).to_le_bytes());

        if !node.is_leaf {
            let left_idx = 2 * i + 1;
            let right_idx = 2 * i + 2;
            if left_idx < num_nodes && right_idx < num_nodes {
                let left_off = nodes_off + left_idx * NODE_SIZE;
                let right_off = nodes_off + right_idx * NODE_SIZE;
                let left_rel = (left_off as i64) - (node_off as i64);
                let right_rel = (right_off as i64) - (node_off as i64);
                payloads[node_off + 48..node_off + 56].copy_from_slice(&left_rel.to_le_bytes());
                payloads[node_off + 56..node_off + 64].copy_from_slice(&right_rel.to_le_bytes());
            }
        }
    }

    let root_result_off = nodes_off + 32;

    let actions = vec![
        Action {
            kind: Kind::AsyncDispatch,
            dst: 0,
            src: 2,
            offset: flag_addr as u32,
            size: 1,
        },
        Action {
            kind: Kind::WaitUntil,
            dst: root_result_off as u32,
            src: exp_addr as u32,
            offset: 0,
            size: 8,
        },
        Action {
            kind: Kind::Noop,
            dst: ctrl_off as u32,
            src: 2,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = Algorithm {
        actions,
        payloads,
        state: State { cranelift_ir_offsets: vec![clif_off] },
        units: UnitSpec { cranelift_units: 1 },
        cranelift_assignments: vec![0u8; 3],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(256 * 1024),
        timeout_ms: Some(30_000),
        thread_name_prefix: None,
    };

    (algorithm, expected)
}

// =========================================================================
// Problem 2: Parallel tree reduction
//
// Given a tree (adjacency list), compute sum of all node values.
// At each internal node, spawn threads for children subtrees.
// Thread count depends on tree shape (branching factor, depth).
// =========================================================================

struct Tree {
    children: Vec<Vec<usize>>,
    values: Vec<u64>,
}

fn build_random_tree(num_nodes: usize, max_children: usize) -> Tree {
    let mut children = vec![Vec::new(); num_nodes];
    let mut values = Vec::with_capacity(num_nodes);

    for i in 0..num_nodes {
        values.push(synthetic_degree(i));
    }

    // Build tree: each node gets assigned as child of a "parent" node
    // using a deterministic pattern that creates variable branching
    let mut next_child = 1usize;
    for parent in 0..num_nodes {
        if next_child >= num_nodes {
            break;
        }
        let branch = 1 + (parent * 7 + 3) % max_children;
        for _ in 0..branch {
            if next_child >= num_nodes {
                break;
            }
            children[parent].push(next_child);
            next_child += 1;
        }
    }

    Tree { children, values }
}

// --- Rust baseline: recursive parallel tree sum ---

fn rust_tree_sum(tree: &Arc<Tree>, node: usize, threshold: usize) -> u64 {
    let mut sum = tree.values[node];

    if tree.children[node].is_empty() {
        return sum;
    }

    // Count total descendants to decide if we should parallelize
    let child_count = tree.children[node].len();

    if child_count <= threshold {
        // Sequential: not enough children to justify spawning
        for &child in &tree.children[node] {
            sum += rust_tree_sum(tree, child, threshold);
        }
    } else {
        // Parallel: spawn a thread per child
        let handles: Vec<_> = tree.children[node]
            .iter()
            .map(|&child| {
                let tree = tree.clone();
                thread::spawn(move || rust_tree_sum(&tree, child, threshold))
            })
            .collect();

        for h in handles {
            sum += h.join().unwrap();
        }
    }

    sum
}

// --- Base (CLIF): parallel tree sum via cl_thread_spawn ---

fn gen_tree_clif_ir() -> String {
    // fn0: noop
    // fn1: noop
    // fn2: orchestrator - init ctx, spawn fn3 on root, join, cleanup
    // fn3: recursive tree worker
    //   Node descriptor: +0 value, +8 num_children, +16 threshold,
    //                    +24 children_rel, +32 result, +40 thread_ctx
    r#"function u0:0(i64) system_v {
block0(v0: i64):
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    return
}

function u0:2(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64, i64) -> i64 system_v
    fn2 = %cl_thread_join sig2
    sig3 = (i64) system_v
    fn3 = %cl_thread_cleanup sig3
block0(v0: i64):
    v1 = load.i64 notrap aligned v0
    v2 = iadd v0, v1
    v3 = iadd_imm v0, 8
    call fn0(v3)
    v4 = iconst.i64 3
    v5 = call fn1(v3, v4, v2)
    v6 = call fn2(v3, v5)
    call fn3(v3)
    return
}

function u0:3(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64, i64) -> i64 system_v
    fn2 = %cl_thread_join sig2
    sig3 = (i64) system_v
    fn3 = %cl_thread_cleanup sig3
    fn4 = %cl_thread_call sig1
block0(v0: i64):
    ; Node: +0 value, +8 num_children, +16 threshold, +24 children_rel, +32 result
    ;       +40 thread_ctx, +48 handles_rel
    v1 = load.i64 notrap aligned v0
    v2 = load.i64 notrap aligned v0+8
    v3 = load.i64 notrap aligned v0+16
    v4 = iconst.i64 0
    v5 = icmp eq v2, v4
    brif v5, block10(v1), block1(v2, v3, v1)

block1(v60: i64, v61: i64, v62: i64):
    ; v60=num_children, v61=threshold, v62=accumulated sum
    v63 = icmp ule v60, v61
    brif v63, block2(v60, v62), block5(v60, v62)

block2(v70: i64, v71: i64):
    ; Sequential: call fn3 inline for each child
    v72 = load.i64 notrap aligned v0+24
    v73 = iadd v0, v72
    v74 = iadd_imm v0, 40
    call fn0(v74)
    v75 = iconst.i64 0
    v76 = iconst.i64 3
    jump block3(v75, v71, v73, v70)

block3(v80: i64, v81: i64, v82: i64, v83: i64):
    v84 = ishl_imm v80, 3
    v85 = iadd v82, v84
    v86 = load.i64 notrap aligned v85
    v87 = iadd v0, v86
    v88 = iadd_imm v0, 40
    v89 = iconst.i64 3
    v90 = call fn4(v88, v89, v87)
    v91 = load.i64 notrap aligned v87+32
    v92 = iadd v81, v91
    v93 = iadd_imm v80, 1
    v94 = icmp eq v93, v83
    brif v94, block4(v92), block3(v93, v92, v82, v83)

block4(v100: i64):
    v101 = iadd_imm v0, 40
    call fn3(v101)
    jump block10(v100)

block5(v110: i64, v111: i64):
    ; Parallel: spawn a thread per child, then join all
    v112 = load.i64 notrap aligned v0+24
    v113 = iadd v0, v112
    v114 = load.i64 notrap aligned v0+48
    v115 = iadd v0, v114
    v116 = iadd_imm v0, 40
    call fn0(v116)
    v117 = iconst.i64 0
    v118 = iconst.i64 3
    jump block6(v117, v110, v113, v115)

block6(v120: i64, v121: i64, v122: i64, v123: i64):
    v124 = ishl_imm v120, 3
    v125 = iadd v122, v124
    v126 = load.i64 notrap aligned v125
    v127 = iadd v0, v126
    v128 = iadd_imm v0, 40
    v129 = iconst.i64 3
    v130 = call fn1(v128, v129, v127)
    v131 = iadd v123, v124
    store.i64 notrap aligned v130, v131
    v132 = iadd_imm v120, 1
    v133 = icmp eq v132, v121
    brif v133, block7(v121, v122, v123, v111), block6(v132, v121, v122, v123)

block7(v140: i64, v141: i64, v142: i64, v143: i64):
    v144 = iconst.i64 0
    jump block8(v144, v140, v141, v142, v143)

block8(v150: i64, v151: i64, v152: i64, v153: i64, v154: i64):
    v155 = ishl_imm v150, 3
    v156 = iadd v153, v155
    v157 = load.i64 notrap aligned v156
    v158 = iadd_imm v0, 40
    v159 = call fn2(v158, v157)
    v160 = iadd v152, v155
    v161 = load.i64 notrap aligned v160
    v162 = iadd v0, v161
    v163 = load.i64 notrap aligned v162+32
    v164 = iadd v154, v163
    v165 = iadd_imm v150, 1
    v166 = icmp eq v165, v151
    brif v166, block9(v164), block8(v165, v151, v152, v153, v164)

block9(v170: i64):
    v171 = iadd_imm v0, 40
    call fn3(v171)
    jump block10(v170)

block10(v180: i64):
    store.i64 notrap aligned v180, v0+32
    return
}"#
    .to_string()
}

const TREE_NODE_SIZE: usize = 128; // value, num_children, threshold, children_rel, result, thread_ctx, handles_rel, padding

fn build_tree_algorithm(tree: &Tree, threshold: usize) -> (Algorithm, u64) {
    let num_nodes = tree.values.len();
    let expected: u64 = tree.values.iter().sum();

    let flag_addr = 64usize;
    let exp_addr = 56usize;
    let ctrl_off = 72usize;
    let nodes_off = align64(ctrl_off + 16);
    let nodes_end = nodes_off + num_nodes * TREE_NODE_SIZE;

    // Children offset arrays: for each node, store relative offsets to child nodes
    let children_off = align64(nodes_end);
    let max_children: usize = tree.children.iter().map(|c| c.len()).sum();
    let children_end = children_off + max_children * 8;

    // Handle storage: one i64 per child for thread handles
    let handles_off = align64(children_end);
    let handles_end = handles_off + max_children * 8;

    let clif_off = align64(handles_end);
    let clif_ir = gen_tree_clif_ir();
    let clif_bytes = format!("{}\0", clif_ir);
    let payload_size = align64(clif_off + clif_bytes.len());

    let mut payloads = vec![0u8; payload_size];
    payloads[clif_off..clif_off + clif_bytes.len()].copy_from_slice(clif_bytes.as_bytes());
    payloads[exp_addr..exp_addr + 8].copy_from_slice(&expected.to_le_bytes());

    let root_rel = (nodes_off as i64) - (ctrl_off as i64);
    payloads[ctrl_off..ctrl_off + 8].copy_from_slice(&root_rel.to_le_bytes());

    let mut child_slot = 0usize;
    for (i, children) in tree.children.iter().enumerate() {
        let node_off = nodes_off + i * TREE_NODE_SIZE;

        // +0: value
        payloads[node_off..node_off + 8].copy_from_slice(&tree.values[i].to_le_bytes());
        // +8: num_children
        payloads[node_off + 8..node_off + 16].copy_from_slice(&(children.len() as u64).to_le_bytes());
        // +16: threshold
        payloads[node_off + 16..node_off + 24].copy_from_slice(&(threshold as u64).to_le_bytes());

        if !children.is_empty() {
            // +24: children_rel (offset from node to children array)
            let my_children_off = children_off + child_slot * 8;
            let children_rel = (my_children_off as i64) - (node_off as i64);
            payloads[node_off + 24..node_off + 32].copy_from_slice(&children_rel.to_le_bytes());

            // +48: handles_rel (offset from node to handles array)
            let my_handles_off = handles_off + child_slot * 8;
            let handles_rel = (my_handles_off as i64) - (node_off as i64);
            payloads[node_off + 48..node_off + 56].copy_from_slice(&handles_rel.to_le_bytes());

            // Write child node offsets (relative to this node)
            for &child_id in children {
                let child_node_off = nodes_off + child_id * TREE_NODE_SIZE;
                let child_rel = (child_node_off as i64) - (node_off as i64);
                let slot_off = children_off + child_slot * 8;
                payloads[slot_off..slot_off + 8].copy_from_slice(&child_rel.to_le_bytes());
                child_slot += 1;
            }
        }
    }

    let root_result_off = nodes_off + 32;

    let actions = vec![
        Action {
            kind: Kind::AsyncDispatch,
            dst: 0,
            src: 2,
            offset: flag_addr as u32,
            size: 1,
        },
        Action {
            kind: Kind::WaitUntil,
            dst: root_result_off as u32,
            src: exp_addr as u32,
            offset: 0,
            size: 8,
        },
        Action {
            kind: Kind::Noop,
            dst: ctrl_off as u32,
            src: 2,
            offset: 0,
            size: 0,
        },
    ];

    let algorithm = Algorithm {
        actions,
        payloads,
        state: State { cranelift_ir_offsets: vec![clif_off] },
        units: UnitSpec { cranelift_units: 1 },
        cranelift_assignments: vec![0u8; 3],
        worker_threads: Some(1),
        blocking_threads: Some(1),
        stack_size: Some(256 * 1024),
        timeout_ms: Some(30_000),
        thread_name_prefix: None,
    };

    (algorithm, expected)
}

// =========================================================================
// Benchmark runners
// =========================================================================

fn bench_recursive(
    n: usize,
    threshold: usize,
    rounds: usize,
    label: &str,
) -> BenchResult {
    let data: Vec<u64> = (0..n).map(|i| synthetic_degree(i)).collect();
    let expected: u64 = data.iter().sum();
    let data_arc = Arc::new(data.clone());

    // Warmup
    let _ = rust_recursive_sum(&data_arc, 0, n, threshold);
    let _ = base::execute(build_recursive_algorithm(&data, threshold).0);

    let mut rust_times = Vec::with_capacity(rounds);
    let mut base_times = Vec::with_capacity(rounds);
    let mut verified = true;

    for _ in 0..rounds {
        let t = std::time::Instant::now();
        let got = rust_recursive_sum(&data_arc, 0, n, threshold);
        rust_times.push(t.elapsed().as_secs_f64() * 1000.0);
        if got != expected { verified = false; }

        let (alg, _) = build_recursive_algorithm(&data, threshold);
        let t = std::time::Instant::now();
        if !base::execute(alg).is_ok() { verified = false; }
        base_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    BenchResult {
        name: label.to_string(),
        python_ms: None,
        rust_ms: Some(median(&mut rust_times)),
        base_ms: median(&mut base_times),
        actions: None,
        verified: Some(verified),
    }
}

fn bench_tree(
    num_nodes: usize,
    max_branch: usize,
    threshold: usize,
    rounds: usize,
    label: &str,
) -> BenchResult {
    let tree = build_random_tree(num_nodes, max_branch);
    let expected: u64 = tree.values.iter().sum();
    let tree_arc = Arc::new(tree);

    // Warmup
    let _ = rust_tree_sum(&tree_arc, 0, threshold);
    let tree_ref: &Tree = &*tree_arc;
    let _ = base::execute(build_tree_algorithm(tree_ref, threshold).0);

    let mut rust_times = Vec::with_capacity(rounds);
    let mut base_times = Vec::with_capacity(rounds);
    let mut verified = true;

    for _ in 0..rounds {
        let t = std::time::Instant::now();
        let got = rust_tree_sum(&tree_arc, 0, threshold);
        rust_times.push(t.elapsed().as_secs_f64() * 1000.0);
        if got != expected { verified = false; }

        let (alg, _) = build_tree_algorithm(tree_ref, threshold);
        let t = std::time::Instant::now();
        if !base::execute(alg).is_ok() { verified = false; }
        base_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    BenchResult {
        name: label.to_string(),
        python_ms: None,
        rust_ms: Some(median(&mut rust_times)),
        base_ms: median(&mut base_times),
        actions: None,
        verified: Some(verified),
    }
}

// =========================================================================
// Public entry point
// =========================================================================

pub struct Config {
    pub profile: String,
    pub rounds: usize,
    pub workers: usize,
}

pub fn run(cfg: &Config) -> Vec<BenchResult> {
    let mut results = Vec::new();

    println!("Workers: {}, Rounds: {}, Profile: {}", cfg.workers, cfg.rounds, cfg.profile);
    println!("\n=== Dynamic Thread Spawning: Rust vs Base ===");
    println!("Both sides spawn/join threads dynamically based on workload.\n");

    // --- Divide-and-conquer ---
    println!("--- Divide-and-conquer parallel sum ---");
    let rec_configs: &[(usize, usize)] = match cfg.profile.as_str() {
        "quick" => &[(1_000_000, 100_000)],
        "full" => &[
            (1_000_000, 100_000),
            (5_000_000, 500_000),
            (10_000_000, 1_000_000),
        ],
        _ => &[(1_000_000, 100_000), (5_000_000, 500_000)],
    };
    for &(n, thresh) in rec_configs {
        let n_str = if n >= 1_000_000 { format!("{}M", n / 1_000_000) } else { format!("{}K", n / 1_000) };
        let t_str = if thresh >= 1_000_000 { format!("{}M", thresh / 1_000_000) } else { format!("{}K", thresh / 1_000) };
        let label = format!("rec n={} t={}", n_str, t_str);
        results.push(bench_recursive(n, thresh, cfg.rounds, &label));
    }

    // --- Tree traversal ---
    println!("\n--- Parallel tree reduction ---");
    let tree_configs: &[(usize, usize, usize)] = match cfg.profile.as_str() {
        // (num_nodes, max_branch, threshold)
        "quick" => &[(10_000, 4, 2)],
        "full" => &[
            (10_000, 4, 2),
            (50_000, 4, 2),
            (10_000, 8, 3),
            (50_000, 8, 3),
        ],
        _ => &[(10_000, 4, 2), (50_000, 4, 2)],
    };
    for &(nodes, branch, thresh) in tree_configs {
        let n_str = if nodes >= 1_000 { format!("{}K", nodes / 1_000) } else { format!("{}", nodes) };
        let label = format!("tree n={} b={} t={}", n_str, branch, thresh);
        results.push(bench_tree(nodes, branch, thresh, cfg.rounds, &label));
    }

    results
}
