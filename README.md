# Base

Base is an execution system that delivers performance comparable to idiomatic Rust while allowing control logic to be specified in a language with a strong type system — without introducing type checking or interpreter overhead at runtime.

Programs are defined as a configuration pair: a `BaseConfig` (Cranelift IR + memory layout) and an `Algorithm` (actions, execution parameters). This pair is data — serializable as JSON, transportable, and buildable from any language. Lean 4 is used here as the specification language, where dependent types can verify program structure at Rust build time, but the execution at runtime has no knowledge of or usage of Lean.

The system is completely portable — all dependencies build from `cargo` with no manual system library installation, and only portable Rust features are used. Cranelift provides a JIT compiler similar to LLVM but without the system dependency — it is pure Rust, built from Cargo. Backend code quality is comparable to LLVM, which means optimization lives in Lean: emit good IR, and the generated code is fast. CPU, GPU, file, network, and database primitives are exposed through a shared memory space and directly callable from the JIT-compiled IR.

GPU compute uses [wgpu](https://wgpu.rs/), a portable abstraction over Vulkan, Metal, DX12, and WebGPU — no vendor-specific SDK required. For NVIDIA GPUs, Base also provides direct CUDA support via [cudarc](https://github.com/coreylowman/cudarc), which dynamically loads the CUDA driver at runtime (`libcuda.so` / `nvcuda.dll`). PTX kernel source is embedded in the algorithm's initial memory by Lean at build time and loaded into the CUDA driver at execution time — no `nvcc` compilation step, no `-sys` crate, and no build-time CUDA SDK dependency beyond having the driver installed.

## How it works

A program in Base is two things:

```
BaseConfig { cranelift_ir, memory_size, context_offset, initial_memory }
Algorithm  { actions, cranelift_units, timeout_ms, output }
```

**BaseConfig** defines the compiled code (Cranelift IR text), the memory region it operates on, and static initial memory contents generated at build time (shader sources, binding descriptors, PTX kernels, etc.).

**Algorithm** defines what to execute: a sequence of actions (call a compiled function, dispatch async work, conditional jump, park/wake), and optionally an output schema for returning Arrow RecordBatches.

At build time, Lean 4 generates this pair as JSON. The Rust build script deserializes it into typed structs and embeds the binary. At runtime, Cranelift JIT-compiles the IR once and executes actions against shared memory — no interpreter, no GC, no serialization layer in the hot path.

## Execution patterns

### One-shot execution

```rust
let (config, alg): (BaseConfig, Algorithm) = bincode::deserialize(ALGORITHM_BINARY)?;
base::run(config, alg)?;
```

### Compile-once, execute-many with payloads

For workloads that benefit from persistent state and dynamic data, the `Base` struct provides JIT-once semantics with zero-copy data passing:

```rust
let mut base = Base::new(config)?;              // JIT compile once

// Pass dynamic data via pointer — no copying into shared memory
let results = base.execute(&algorithm, &data)?;

// Or with an output buffer for zero-copy results
base.execute_into(&algorithm, &payload, &mut output)?;
```

The CLIF code reads `data_ptr`, `data_len`, `out_ptr`, and `out_len` from a reserved region at offsets 0x08-0x27. GPU uploads/downloads use `cl_gpu_upload_ptr` / `cl_gpu_download_ptr` to transfer directly between caller buffers and GPU memory — no intermediate copies through shared memory.

## Example: CUDA Scene Renderer

The [scene](applications/scene/) application renders a small physically based scene on the GPU using raw PTX launched through the CUDA driver API. The entire program — PTX kernel source, Cranelift IR orchestration, BMP header, memory layout, and output filename — is defined in a single Lean file.

![CUDA scene render by Base](scene.png)

*1280x720, 128 samples per pixel, up to 5 bounces. Rendered in ~415ms from raw PTX and converted to PNG offline for README inclusion.*

**[MakeAlgorithm.lean](applications/scene/lean/MakeAlgorithm.lean)** defines:

1. A raw PTX Monte Carlo renderer with diffuse, mirror-like metal, and glass materials; multiple direct lights; fog; depth of field; tone mapping; and stochastic sampling.

2. Cranelift IR that orchestrates: CUDA init -> device buffer creation -> PTX launch -> pixel download -> file write as BMP.

3. Initial memory that packs the BMP header, PTX source, binding descriptors, output filename, and CLIF IR into a flat byte layout.

## Type system design

The `(BaseConfig, Algorithm)` pair is plain data. The entire assembly-like control surface — memory layout, action sequencing, FFI calls, GPU dispatch — is exposed to the specification language. This enables freedom to bolt on type systems that constrain effects in a bottom-up way: start with the raw primitives, then layer on whatever invariants the application needs.

Lean 4 is used here because its dependent type system can express constraints on program structure (memory layout invariants, offset arithmetic, action sequencing) and verify them at build time. But this is a property of the specification language, not the Rust executor. Any language that can produce the right JSON structure can target Base. The Cranelift JIT sees only IR text and a byte array.

## Benchmark results

The benchmarks demonstrate that Base matches idiomatic Rust performance — the key point being that programs specified in Lean and compiled through Cranelift do not pay the runtime overhead you would see from a Lean, Python, or other high-level language runtime. The specification language's type system is a build-time concern only; at execution time, it is just native code operating on flat memory.

All benchmarks run on the same machine, median of 10 rounds. CPU benchmarks compare Python, idiomatic Rust, and Base. GPU benchmarks compare Burn (wgpu/cuda backends) and Base. All results are verified for correctness. JIT compilation happens once before timing (`Base::new`), so only execution cost is measured.

### CPU workloads

```
Benchmark                  Python         Rust         Base
------------------------------------------------------------
CSV (1M)                  542.6ms       55.8ms       30.0ms
CSV (5M)                 2713.3ms      314.6ms      153.9ms
JSON (100K)                81.5ms        2.6ms        2.4ms
JSON (500K)               350.1ms       15.5ms        9.4ms
Regex (500K)               59.4ms        4.1ms        2.8ms
Regex (1M)                103.7ms        8.1ms        4.5ms
StrSearch (1M)             19.2ms        2.1ms        1.5ms
WordCount (1M)            155.7ms       36.0ms       37.1ms
```

The benchmarks measure pure execution time — JIT compilation happens once before timing via `Base::new()`. All file I/O benchmarks (CSV, JSON, Regex, StrSearch, WC) include file read/write in the timing for all three implementations. Base can exceed Rust on workloads like CSV and Regex because Cranelift IR can emit SIMD instructions directly, avoiding the overhead of Rust's standard library abstractions. WordCount is a regression — Base uses FFI calls to a hash table (`ht_*` primitives) which Cranelift cannot inline, whereas LLVM can inline Rust's equivalent `HashMap` operations.

### Sort (radix sort vs Rust pdqsort)

```
Benchmark                    Rust         Base
---------------------------------------------
Sort (100K)                 1.1ms        1.0ms
Sort (1M)                  12.4ms        8.4ms
Sort (5M)                  63.8ms       44.7ms
```

Base uses LSD radix sort (O(n), Lean-generated CLIF) vs Rust's `sort_unstable` (pdqsort, O(n log n)). This benchmark demonstrates that Base can execute a memory-intensive workload competitively. Rust would likely match given a native radix sort implementation — the comparison is less about Rust vs Base and more about showing Base handles hot memory access patterns well.

### Burn CPU workloads (Rust vs Burn vs Base)

```
Benchmark                    Rust         Burn         Base
------------------------------------------------------------
MatMul (256x256)            1.5ms        0.2ms        1.0ms
MatMul (1024x1024)         96.6ms        8.0ms       59.3ms
VecAdd (10M)                7.1ms       22.5ms        2.4ms
VecAdd (50M)               35.9ms      127.1ms       12.6ms
Sum (50M)                  35.7ms        9.1ms        7.9ms
Sum (100M)                 71.3ms       18.1ms       16.1ms
```

Burn wins at MatMul (uses optimized BLAS kernels). Base wins at simple operations (VecAdd, Sum) where allocation overhead dominates the trivial O(n) compute.

### GPU workloads (Burn wgpu vs Base wgpu)

```
Benchmark              Burn(wgpu)         Base
---------------------------------------------
VecAdd 256K                 3.7ms        0.6ms
MatMul 256x256              3.0ms        0.5ms
MatMul 512x512              3.8ms        1.9ms
Reduction 256K              3.1ms        0.4ms
Reduction 896K              4.2ms        1.1ms
```

Both sides use [wgpu](https://wgpu.rs/) — the same portable GPU library, the same GPU hardware. The 3-5x gap is entirely Burn's host-side framework overhead (tensor allocation, cubecl scheduling, backend abstraction layer). Base calls the wgpu API directly through thin FFI wrappers from JIT-compiled CLIF.

### GPU iterative (data stays GPU-resident)

```
Benchmark                Raw wgpu         Burn         Base
------------------------------------------------------------
1x scale 1M                 4.5ms        5.7ms        2.2ms
100x scale 1M               9.6ms        8.9ms        9.3ms
1000x scale 1M             70.2ms       44.1ms       69.9ms
```

Base wins at low iteration counts (minimal setup overhead). At high iteration counts Burn pulls ahead through kernel fusion.

### CUDA (Burn cuda-jit vs Base PTX)

```
Benchmark              Burn(cuda)         Base
---------------------------------------------
SAXPY 262K                  0.5ms        0.4ms
SAXPY 1M                    1.8ms        1.3ms
```

Both sides go through similar CUDA driver paths — Base with less runtime abstraction overhead.

### Histogram (parallel, 256 bins)

```
Benchmark                    Rust         Base
---------------------------------------------
hist n=10M workers=1        8.8ms        9.1ms
hist n=10M workers=4        6.9ms        7.1ms
```

## FFI primitives

Cranelift IR can call these directly via `%cl_*` function references:

| Category | Functions |
|----------|-----------|
| **File** | `cl_file_read`, `cl_file_write` |
| **GPU** | `cl_gpu_init`, `cl_gpu_create_buffer`, `cl_gpu_create_pipeline`, `cl_gpu_upload`, `cl_gpu_upload_ptr`, `cl_gpu_dispatch`, `cl_gpu_download`, `cl_gpu_download_ptr`, `cl_gpu_cleanup` |
| **CUDA** | `cl_cuda_init`, `cl_cuda_create_buffer`, `cl_cuda_launch`, `cl_cuda_upload`, `cl_cuda_upload_ptr`, `cl_cuda_download`, `cl_cuda_download_ptr`, `cl_cuda_cleanup` |
| **Network** | `cl_net_init`, `cl_net_listen`, `cl_net_connect`, `cl_net_accept`, `cl_net_send`, `cl_net_recv`, `cl_net_cleanup` |
| **Database** | `cl_lmdb_init`, `cl_lmdb_open`, `cl_lmdb_begin_write_txn`, `cl_lmdb_commit_write_txn`, `cl_lmdb_put`, `cl_lmdb_get`, `cl_lmdb_delete`, `cl_lmdb_cursor_scan`, `cl_lmdb_sync`, `cl_lmdb_cleanup` |
| **Threading** | `cl_thread_init`, `cl_thread_spawn`, `cl_thread_join`, `cl_thread_call`, `cl_thread_cleanup` |
| **Hash table** | `ht_create`, `ht_insert`, `ht_lookup`, `ht_count`, `ht_get_entry`, `ht_increment` |

The `_ptr` variants (`cl_gpu_upload_ptr`, `cl_gpu_download_ptr`, `cl_cuda_upload_ptr`, `cl_cuda_download_ptr`) transfer data directly between caller-provided pointers and GPU/CUDA buffers, enabling zero-copy integration with the `execute_into` payload pattern.

## Building

The `base` crate requires only Rust. Applications and benchmarks additionally require [Lean 4](https://leanprover.github.io/lean4/doc/setup.html) to generate their algorithm definitions. GPU workloads require a Vulkan, Metal, DX12, or WebGPU capable system. CUDA workloads require an NVIDIA GPU with a CUDA-capable driver installed (libraries are loaded dynamically at runtime — see above).

```bash
# Run all benchmarks
cargo run --release -p benchmarks -- --rounds 5

# Run a specific benchmark
cargo run --release -p benchmarks -- --bench sort --rounds 5

# Run tests
cargo test -p base

# Build an application
cargo build --release -p scene
./target/release/scene
```
