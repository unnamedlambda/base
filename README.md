# Base

Base is an execution system that delivers performance comparable to idiomatic Rust while allowing control logic to be specified in a language with a strong type system — without introducing type checking or interpreter overhead at runtime.

Programs are defined as a configuration pair: a `BaseConfig` (Cranelift IR + memory layout) and an `Algorithm` (actions, payloads, execution parameters). This pair is data — serializable as JSON, transportable, and buildable from any language. Lean 4 is used here as the specification language, where dependent types can verify program structure at Rust build time, but the execution at runtime has no knowledge of or usage of Lean.

The system is completely portable — all dependencies build from `cargo` with no manual system library installation, and only portable Rust features are used. Cranelift provides a JIT compiler similar to LLVM but without the system dependency — it is pure Rust, built from Cargo. Backend code quality is comparable to LLVM, which means optimization lives in Lean: emit good IR, and the generated code is fast. CPU, GPU, file, network, and database primitives are exposed through a shared memory space and directly callable from the JIT-compiled IR.

## How it works

A program in Base is two things:

```
BaseConfig { cranelift_ir, memory_size, context_offset }
Algorithm  { actions, payloads, cranelift_units, timeout_ms, output }
```

**BaseConfig** defines the compiled code (Cranelift IR text) and the memory region it operates on. `context_offset` splits memory into a payload region `[0..context_offset)` that gets overwritten each call and a persistent region `[context_offset..memory_size)` that survives across calls.

**Algorithm** defines what to execute: a sequence of actions (call a compiled function, dispatch async work, conditional jump, park/wake), initial memory contents (payloads), and optionally an output schema for returning Arrow RecordBatches.

At build time, Lean 4 generates this pair as JSON. The Rust build script deserializes it into typed structs and embeds the binary. At runtime, Cranelift JIT-compiles the IR and executes actions against the shared memory — no interpreter, no GC, no serialization layer in the hot path.

## Example: GPU path tracer

The [raytrace](applications/raytrace/) application renders a 4096x4096 Cornell box with path tracing on the GPU. The entire program — WGSL shader, Cranelift IR orchestration, BMP header, memory layout — is defined in a single Lean file.

![Cornell box rendered by Base](cornell_box.png)

*4096x4096, 32 samples per pixel, 5 bounces. Rendered in ~735ms. (downsized and converted to PNG offline for README inclusion)*

**[MakeAlgorithm.lean](applications/raytrace/lean/MakeAlgorithm.lean)** defines:

1. A WGSL compute shader implementing a full path tracer — 5 planes, 2 AABBs, an area light, Lambertian diffuse with direct lighting and cosine-weighted indirect sampling (dispatched as 256x256 workgroups of 16x16 threads)

2. Cranelift IR that orchestrates: GPU init -> buffer creation -> shader upload -> dispatch -> download pixel data -> file write as BMP

3. A payload that packs the BMP header, shader source, binding descriptors, output filename, and CLIF IR into a flat byte layout

4. A configuration that wires it all together:

```lean
def raytraceConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := payloads.length + pixelBytes,
  context_offset := 0
}

def raytraceAlgorithm : Algorithm :=
  let clifCallAction : Action :=
    { kind := .ClifCall, dst := u32 0, src := u32 1, offset := u32 0, size := u32 0 }
  {
    actions := [clifCallAction],
    payloads := payloads,
    cranelift_units := 0,
    timeout_ms := some 300000
  }
```

The Rust side is minimal — deserialize the pair, call `run`:

```rust
let (config, alg): (BaseConfig, Algorithm) = bincode::deserialize(ALGORITHM_BINARY)
    .expect("Failed to deserialize");

match run(config, alg) { ... }
```

## Interactive execution with `Base` struct

For workloads that benefit from persistent state, the `Base` struct provides compile-once, execute-many semantics:

```rust
let mut base = Base::new(config)?;          // JIT compile once

// First call: set up GPU pipeline, open DB handles, etc.
let _ = base.execute(setup_algorithm)?;     // State persists above context_offset

// Hot path: just pass new data, everything else is already initialized
let results = base.execute(work_algorithm)?; // Payloads overwrite [0..context_offset)
```

GPU handles, database connections, and computed state survive across `execute()` calls in the persistent memory region. New payloads (file paths, parameters, input data) are written into the payload region without disturbing persistent state.

## Type system design

The `(BaseConfig, Algorithm)` pair is plain data. The entire assembly-like control surface — memory layout, action sequencing, FFI calls, GPU dispatch — is exposed to the specification language. This enables freedom to bolt on type systems that constrain effects in a bottom-up way: start with the raw primitives, then layer on whatever invariants the application needs.

Lean 4 is used here because its dependent type system can express constraints on program structure (memory layout invariants, offset arithmetic, action sequencing) and verify them at build time. But this is a property of the specification language, not the Rust executor. Any language that can produce the right JSON structure can target Base. The Cranelift JIT sees only IR text and a byte array.

## Benchmark results

All benchmarks run on the same machine, 10 rounds with median taken, comparing Python, idiomatic Rust, and Base (Cranelift JIT). GPU benchmarks compare against Burn (wgpu backend).

### CPU workloads

```
Benchmark                  Python         Rust         Base
------------------------------------------------------------
CSV (10K)                  19.8ms        0.2ms        1.7ms
CSV (1M)                  547.9ms       58.1ms       36.3ms
CSV (5M)                 2697.5ms      320.6ms      176.0ms
JSON (1K)                  16.5ms        0.0ms        1.3ms
JSON (500K)               357.4ms       16.2ms       10.0ms
Regex (10K)                15.1ms        0.1ms        1.4ms
Regex (1M)                105.3ms        7.1ms        5.2ms
StrSearch (1M)             20.3ms        2.3ms        1.7ms
WordCount (1M)            162.6ms       32.1ms       34.4ms
```

Base has ~1.3ms JIT compilation overhead visible at small sizes. At scale, direct CLIF IR manipulation 
including SIMD usage can overtake raw Rust.

### GPU workloads

```
GPU Benchmark              Burn(wgpu)         Base
-----------------------------------------------------
VecAdd 256K                     4.3ms          5.0ms
VecAdd 500K                     5.7ms          9.1ms
MatMul 256x256                  3.0ms          2.3ms
MatMul 512x512                  4.4ms          6.9ms
Reduction 256K                  2.9ms          1.5ms
Reduction 512K                  3.3ms          1.9ms
Reduction 896K                  4.0ms          2.6ms
```

Results vary by kernel shape. Burn wins on simple element-wise operations (VecAdd) and larger MatMul. Base wins on reductions and small MatMul.

### GPU iterative (data stays GPU-resident)

```
GPU Iterative              Raw wgpu         Burn         Base
----------------------------------------------------------------
1x scale 1M                  3.8ms        3.9ms          3.3ms
10x scale 1M                 3.9ms        4.2ms          3.3ms
100x scale 1M                8.8ms        5.8ms          9.1ms
1000x scale 1M              51.7ms       34.4ms         54.3ms
```

At low iteration counts Base matches or beats raw wgpu. At high iteration counts Burn pulls ahead through kernel fusion.

### Parallel dispatch and threading

```
Dispatch                         Rust    Base     Actions
----------------------------------------------------------
bfs n=5M c=100K                 0.4ms   0.4ms        214
phase k=20 n=5M                 0.5ms   0.4ms        300

Benchmark                        Rust         Base
----------------------------------------------------
rec n=5M t=500K                 1.4ms        2.0ms
tree n=50K b=4 t=2           1010.2ms     1015.1ms
```

### Histogram (parallel, 256 bins)

```
Benchmark                        Rust         Base
----------------------------------------------------
hist n=1M w=4                   1.0ms        1.8ms
hist n=10M w=4                  2.1ms        2.8ms
```

### Network and memory pipeline

```
Network                         Raw TCP         Base
------------------------------------------------------
TCP Echo 100KB                  51.0ms        50.6ms

Memory Pipeline              Rust(wgpu)         Base
------------------------------------------------------
File->GPU->File (1.0MB)         4.6ms          4.6ms
File->GPU->File (3.8MB)        10.7ms        14.3ms
```

## FFI primitives

Cranelift IR can call these directly via `%cl_*` function references:

| Category | Functions |
|----------|-----------|
| **File** | `cl_file_read`, `cl_file_write` |
| **GPU** | `cl_gpu_init`, `cl_gpu_create_buffer`, `cl_gpu_create_pipeline`, `cl_gpu_upload`, `cl_gpu_dispatch`, `cl_gpu_download`, `cl_gpu_cleanup` |
| **Network** | `cl_net_init`, `cl_net_listen`, `cl_net_connect`, `cl_net_accept`, `cl_net_send`, `cl_net_recv`, `cl_net_cleanup` |
| **Database** | `cl_lmdb_init`, `cl_lmdb_open`, `cl_lmdb_begin_write_txn`, `cl_lmdb_commit_write_txn`, `cl_lmdb_put`, `cl_lmdb_get`, `cl_lmdb_delete`, `cl_lmdb_cursor_scan`, `cl_lmdb_sync`, `cl_lmdb_cleanup` |
| **Threading** | `cl_thread_init`, `cl_thread_spawn`, `cl_thread_join`, `cl_thread_call`, `cl_thread_cleanup` |
| **Hash table** | `ht_create`, `ht_insert`, `ht_lookup`, `ht_count`, `ht_get_entry`, `ht_increment` |

## Building

The `base` crate requires only Rust. Applications and benchmarks additionally require [Lean 4](https://leanprover.github.io/lean4/doc/setup.html) to generate their algorithm definitions. GPU workloads use wgpu, which runs on Vulkan, Metal, DX12, and WebGPU.

```bash
# Run benchmarks
cargo run --release -p benchmarks -- --rounds 5

# Build an application
cargo build --release -p raytrace

# Run
./target/release/raytrace
```
