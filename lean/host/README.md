# Lean as a host

A third way to run an artifact, beside the Rust `build.rs` path and `py-base`:
the Lean program that *builds* an artifact runs it, in the same process.

```
lake exe upcasehost
```

```
runtime memory: 1049160 bytes
artifact transformed 38 bytes
in memory: A LEAN HOST, RUNNING ITS OWN ARTIFACT
artifact wrote output.txt: A LEAN HOST, RUNNING ITS OWN ARTIFACT
```

`input.txt` was read, upper-cased and written to `output.txt` by the artifact.
Nothing in `UpcaseHost.lean` did any of it.

## Build

```
cd lean/host && lake exe upcasehost
```

That is the whole of it. Lake builds `libbase.so` itself — the `libbase` target
runs `cargo build -p base` — which is the mirror of `build-support` running
lake from a cargo build script. Each ecosystem's tool drives the other, so a
Lean caller never types `cargo` and a Rust caller never types `lake`.

`BASE_PROFILE=release` builds and links the release runtime. `BASE_LIB_DIR`
says the library is yours to manage: lake links what is there and runs no
cargo.

Both are read when lake *elaborates* this configuration, and lake caches that
by content — so changing either afterwards does nothing until `lake clean`.
Touching the lakefile is not enough.

## What this is for

The point is not speed. The runtime does exactly what it did before — the same
cranelift JIT over the same CLIF — and building an artifact is a one-off either
way. The point is that the effectful part of a Lean program becomes a *value*.

`Upcase.setup` is an `Artifact`. It reads a file, transforms every byte and writes a
file, and it is data: a `main` that uses it does its work through the artifact,
and its own `IO` is three lines of open/run/close. Lean's `IO` is opaque and
nothing can be proved about it; a CLIF program has a semantics — `HProgSem`
gives every FFI call a contract, `HProgFrames` says which memory it may write —
so this is not just a different way to plumb effects, it is effects you can
state something about.

That has limits, and they are the interesting part:

* **The menu is closed.** `AlgorithmLib.IR.Ffi` has 85 constructors. What is not
  in it cannot be done, and adding one is Rust work plus a table entry, not a
  Lean import.
* **Top-level `IO` does not vanish.** Something has to open the runtime. It
  shrinks to a fixed prelude; it does not go away.
* **Only compilable control flow stays pure.** A program that calls the runtime,
  inspects the result and decides what to call next is `IO` with extra steps.
  The escape is to compile the decision into the artifact, which is what
  `forLoop` and `callVoid` are for — but it is real work each time.

## Using it from another Lake package

```lean
require base from git "https://github.com/<you>/base.git" @ "main" / "lean" / "host"
```

or, for a checkout beside yours, `require base from "../base/lean/host"`. Then
`import BaseHost`, build an `Artifact` and run it — this was checked from a package
outside the repository, which builds, links and executes an artifact with no
`cargo` step of its own.

Your `lean-toolchain` must match this one exactly. Lake has no version ranges,
so a mismatch is a hard error rather than a negotiation.

The external library is built *shared* on purpose. Lake gives an executable
only its own package's `moreLinkArgs` (`Lake/Config/LeanExe.lean`), so a
dependent's binary never sees the `-lbase` written here — with a static shim it
fails at link with `undefined symbol: base_new`, and `precompileModules` does
not change that. A shared shim carries the runtime as its own `DT_NEEDED` and
resolves transitively, which is what makes `require` work at all.

## The layers

| | |
|---|---|
| `base/src/capi.rs` | the runtime behind a C calling convention |
| `c/shim.c` | that, in Lean's `IO` convention |
| `BaseHost.lean` | `Runtime`, `execute`, `readMemory`, `withRuntime` |
| `Upcase.lean` | the demo artifact — a value, and buildable without any of the above |
| `UpcaseHost.lean` | the ~15 lines that run it |

The layer it rests on is in `lean/lib`, shared with every generator:
`AlgorithmLib/ProgFFI.lean`, the call side of all 85 entry points.

The dependency arrow points Lean → base and never back. `base` links nothing of
Lean's, so an embedder shipping a Rust or Python binary with a bincode artifact
in it is unaffected by any of this.

## What a host reads back

Results come back three ways:

* `readField`, given the same `Fld` the artifact was built from — the offset
  and the width both come from the layout, so the host never writes either
  down. `Upcase` stores the byte count it read to a `size` field and
  `UpcaseHost` reads it back; neither end knows the number ahead of time.
* the out buffer `execute` answers, for a program that writes through its
  `out_ptr`/`out_len` offsets.
* `readMemory`, for an address no field describes.

An artifact whose effects are files, sockets or the GPU needs none of them.

## Known gaps

* **Window programs and the main thread.** `execute` installs the runtime's
  compiled functions on whichever thread calls it, so a `Runtime` is usable
  from any of them. A program that opens a window is the exception: on macOS
  its event loop has to be created on the main thread.
* **Two workspaces over one package.** `lake` here and `build-support`'s
  `lake` in `lean/algorithms` build the same package directory. `build-support`
  takes a lock; this does not. Do not run both at once.
* **Conventions, not coverage.** Every entry point is callable: `ffi f args`
  takes `Vals V f.params` and returns `ResV V f.result`, both read off `Ffi`,
  so arity and types are the call's type and there is nothing to keep in step.
  What is still uneven is the layer *above* that: `ProgFFI` gives wgpu, cuda,
  files, the window and the hash table a wrapper that reads the context pointer
  out of its slot for you; lmdb and threads have no such convention, because no
  two generators agree on one and `ContextSlots` has no entry for either.
  Inventing one here would mean shipping a convention no artifact uses, so they
  stay at the bare `ffi` call until a real one exists to lift.
* **`fileWrite` fsyncs, and its contract does not say so.** `cl_file_write`
  calls `sync_all()` after every write — a durability decision worth ~0.8 ms
  per call on an SSD, constant in the write's size. Lean's `IO.FS.writeBinFile`
  buffers (Lean exposes no fsync at all), so a benchmark of the two compares
  durable against buffered and reports the artifact slower; at equal semantics
  they are equal within noise, and the emitted loop itself measures 2 us/KB.
  The fact is invisible in `HProgSem`'s `fileWrite` contract, which is exactly
  where it belongs — whether to keep the fsync, drop it, or split the entry
  point is a semantics decision, not a performance tweak.
