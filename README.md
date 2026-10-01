# Base

Base is an interface and system for programming at the scope of an entire
board. The CPU, GPU, network and file system are all controlled through a single
artifact loaded by a driver.

An artifact holds a program and the memory it starts with. The program is CLIF,
the IR of the Cranelift code generator: it runs on the CPU and calls a fixed set
of primitives for everything else, such as GPU dispatch, file reads and network
sends. The driver compiles the program with Cranelift when it loads the artifact,
then runs its entry points by name.

Artifacts are produced by generators. The generators here are written in Lean,
where theorems prove, at build time, properties of an artifact's run-time
behavior. See the [paper](paper/main.pdf) for the design.

## Usage

### Writing a generator

A generator is a Lean program that builds an artifact. Each `←` emits CLIF
instructions and binds their result, much as an IR builder does, and `forLoop`
emits a loop into the artifact rather than looping in Lean.

The generator below copies a 4096-byte buffer, replacing every NUL byte with a
space. It is shipped as the `main` entry point of the `byte_scrub` artifact
([source](lean/algorithms/Demo/ByteScrub.lean)).

```lean4
-- Sixteen bytes at a time: where a byte equals `nul`, take `space`.
def blend (v nul space : V .i8x16) : Prog V L (V .i8x16) := do
  bitselect (← icmp .eq v nul) space v

def scrub (vectors : Nat) (src dst : V .i64) : Prog V L Unit := do
  let nul ← splat .i8x16 (← iconst .i8 0)
  let space ← splat .i8x16 (← iconst .i8 32)
  forLoop (← iconst64 vectors) fun i => do
    let off ← ishlImm i 4
    let v ← loadI8x16 (← iadd src off)
    store (← blend v nul space) (← iadd dst off)

-- 256 vectors of 16 bytes, from the caller's input to its output.
def code : Prog V L Unit := do
  scrub 256 (← dataPtr) (← outPtr)
```

### Running an artifact

The `lean-artifacts` crate builds every generator with Lake and embeds the
resulting artifacts. Loading one compiles it once; each `execute` then calls an
entry point by name with the caller's input and output buffers.

```rust
use base::{Artifact, Driver};

let artifact = Artifact::from_bytes(lean_artifacts::BYTE_SCRUB)?;
let mut driver = Driver::load(artifact)?;

let input = [0u8; 4096];
let mut output = [0u8; 4096];
driver.execute("main", &input, &mut output)?;
assert!(!output.contains(&0));
```

The driver is also exported as a C API, which the Python and Lean bindings use.

## Performance

In both tables a ratio below 1 means Base is faster.

**CPU**, against the same kernel in Rust on an AMD Ryzen 5 5500. The CLIF column
is the generator's code compiled by Cranelift. The two workloads that need
instructions CLIF lacks (256-bit vectors, non-temporal stores) also carry native
assembly, assembled in Lean, in the asm column.

| Workload | Rust (µs) | CLIF / Rust | asm / Rust |
|---|---:|---:|---:|
| Mandelbrot set | 1800.4 | 1.00 | |
| Pointer chase | 2295.5 | 1.00 | |
| Byte histogram | 320.5 | 1.00 | |
| Polynomial, f32 | 6.1 | 2.04 | 1.01 |
| Copy, 32 MB | 2296.8 | 1.56 | 1.00 |

**GPU**, against PyTorch on an RTX 3060, taking whichever of eager and compiled
mode is faster. Small workloads are faster in Base through fusion and a lower
cost of dispatch.

| Workload | PyTorch (µs) | Base / PyTorch |
|---|---:|---:|
| Vector add 50M | 1810 | 0.99 |
| Scale-and-add 50M | 1860 | 0.97 |
| Scale-and-add 1M | 90 | 0.52 |
| GEMV 11008 × 4096 | 555 | 0.98 |
| RMSNorm 4096 | 57 | 0.19 |
| Softmax 32000 | 30 | 0.37 |
| Decode attention, 2048 keys | 83 | 0.77 |
| Decode attention, 128 keys | 67 | 0.22 |

To reproduce, `./benchmarks/run.sh` runs both suites, and
`./benchmarks/cpu/run.sh` runs the CPU suite alone.
