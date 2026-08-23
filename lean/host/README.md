# Lean as a host

A third way to run an artifact, beside the Rust `build.rs` path and `py-base`:
the Lean program that *builds* an artifact runs it, in the same process.

```
lake exe upcasehost
```

```
runtime memory: 1049152 bytes
in memory: A LEAN HOST, RUNNING ITS OWN ARTIFACT
artifact wrote output.txt: A LEAN HOST, RUNNING ITS OWN ARTIFACT
```

`input.txt` was read, upper-cased and written to `output.txt` by the artifact.
Nothing in `UpcaseHost.lean` did any of it.

## Build

`libbase.so` must exist first:

```
cargo build -p base            # or --release
cd lean/host && lake exe upcasehost
```

The lakefile prefers `target/release` when it holds a `libbase.so` and falls
back to `target/debug`; `BASE_LIB_DIR` overrides both. It is read when lake
loads the configuration, so a first `cargo build --release` afterwards wants a
`lake clean` to be picked up.

## What this is for

The point is not speed. The runtime does exactly what it did before — the same
cranelift JIT over the same CLIF — and building an artifact is a one-off either
way. The point is that the effectful part of a Lean program becomes a *value*.

`Upcase.setup` is a `Setup`. It reads a file, transforms every byte and writes a
file, and it is data: a `main` that uses it does its work through the artifact,
and its own `IO` is three lines of open/run/close. Lean's `IO` is opaque and
nothing can be proved about it; a CLIF program has a semantics — `HProgSem`
gives every FFI call a contract, `HProgFrames` says which memory it may write —
so this is not just a different way to plumb effects, it is effects you can
state something about.

That has limits, and they are the interesting part:

* **The menu is closed.** `AlgorithmLib.IR.Ffi` has 80 constructors. What is not
  in it cannot be done, and adding one is Rust work plus a table entry, not a
  Lean import.
* **Top-level `IO` does not vanish.** Something has to open the runtime. It
  shrinks to a fixed prelude; it does not go away.
* **Only compilable control flow stays pure.** A program that calls the runtime,
  inspects the result and decides what to call next is `IO` with extra steps.
  The escape is to compile the decision into the artifact, which is what
  `forLoop` and `callVoid` are for — but it is real work each time.

## The layers

| | |
|---|---|
| `base/src/capi.rs` | the runtime behind a C calling convention |
| `c/shim.c` | that, in Lean's `IO` convention |
| `BaseHost.lean` | `Runtime`, `execute`, `readMemory`, `withRuntime` |
| `Upcase.lean` | the demo artifact — a value, and buildable without any of the above |
| `UpcaseHost.lean` | 15 lines that run it |

The dependency arrow points Lean → base and never back. `base` links nothing of
Lean's, so an embedder shipping a Rust or Python binary with a bincode artifact
in it is unaffected by any of this.

## What a host reads back

The Rust and Python surfaces answer Arrow `RecordBatch`es. That is a second ABI
and this one does not carry it, so results come back as the out buffer
`execute` answers, or through `readMemory` at the offsets an output schema
names. An artifact whose effects are files, sockets or the GPU needs neither.

## Known gaps

* **Arrow.** As above — a host wanting `RecordBatch`es should use the Rust or
  Python surface.
* **One thread.** A `Runtime` is bound to the thread that opened it, because the
  FFI entry points a program calls find their compiled functions in a
  thread-local. `Runtime.bindThread` is what makes it callable from another, and
  nothing yet stops a caller from forgetting.
* **Two workspaces over one package.** `lake` here and `build-support`'s
  `lake` in `lean/algorithms` build the same package directory. `build-support`
  takes a lock; this does not. Do not run both at once.
* **Coverage.** `HProgFFI` wraps 33 of the 80 `Ffi` constructors in
  `HProg.Sur`. The other 47 are used, but inline in whichever generator needed
  them. Lifting them into one uniform layer is what would make this an API
  rather than a working host.
