import AlgorithmLib.HProgSem

/-!
# What a call may write

`HProgSem` gives the file, stdio, hash-table and libm families an executable
contract, because the corpus runs them. The rest are device drivers, windows
and database handles: nothing here will ever compute what they return. What can
still be said about them — and what a proof about a program containing one
actually needs — is *which memory they may disturb*.

A `Frame` is that statement, and only that. It says a call writes at an address
built from its own arguments and leaves the rest of memory alone. It does not
say what it writes.

**These frames are assumptions.** Each one is read off the Rust implementation
in `base/src/ffi/`, and nothing checks that it stays true when that code
changes. They are named here so a proof that leans on one is leaning on
something written down rather than on silence.
-/

namespace AlgorithmLib.HProg

open AlgorithmLib.HProg.Sem

/-- The memory a call may write. Argument positions are zero-based and count
    the callee's own parameters, so they line up with the `args` list. -/
inductive Frame where
  /-- Writes no addressable memory. The result, a file, a socket or the device
      is the whole effect. -/
  | none
  /-- Writes `[args[dst], args[dst] + args[len])`. -/
  | at (dst len : Nat)
  /-- Writes `[args[base] + args[off], … + args[len])` — the shared-memory form,
      where the caller passes an offset rather than a pointer. -/
  | atOff (base off len : Nat)
  /-- Writes at `args[base] + args[off]` for as many bytes as it returns. -/
  | atOffRet (base off : Nat)
  /-- Writes exactly `n` bytes at `args[dst]`. -/
  | fixed (dst n : Nat)
  /-- Writes at `args[dst]`, for a length decided by state the call reads
      rather than by any argument. -/
  | dataDependent (dst : Nat)
  /-- Writes at `args[a]` and `args[b]`, both data-dependent. -/
  | dataDependent2 (a b : Nat)
  deriving Repr, BEq

/-- Context slots are one pointer wide, and every `*_init`/`*_cleanup` writes
    exactly that at the slot it is handed. -/
private def ctxSlot : Frame := .fixed 0 8

/-- The memory each entry point may write, ordered by module to match
    `base/src/ffi/`.

    A total function of `Ffi`, so every entry point has a frame and none is
    named that no declaration uses. Keyed by symbol name this was neither: the
    six colocated hash-table symbols were spelled `cl_ht_*` while the JIT
    resolves `ht_*`, so every program using the table silently had no frame for
    it, and twenty-one names belonged to no declaration at all. -/
def frame : IR.Ffi → Frame
  -- file.rs — the read family writes into shared memory, the write family only
  -- touches the file system.
  | .fileRead => .atOffRet 0 2
  | .fileReadToPtr => .at 1 3
  | .fileWrite | .fileWriteFromPtr => .none

  -- stdio.rs
  | .stdinReadline => .atOff 0 1 2
  | .stdoutWrite => .none

  -- mod.rs — the libm shims are pure.
  | .sinf | .cosf | .powf => .none

  -- ht.rs — the table lives outside shared memory; only the reads write back
  -- into it, and for as many bytes as the stored value happens to be.
  | .htInit | .htCleanup => ctxSlot
  | .htCreate | .htCount | .htInsert | .htIncrement => .none
  | .htLookup => .dataDependent 3
  | .htGetEntry => .dataDependent2 2 3

  -- cuda.rs — uploads and launches move data the other way or stay on the
  -- device; only the downloads write host memory.
  | .cudaInit | .cudaCleanup => ctxSlot
  | .cudaDownload | .cudaDownloadAsync => .at 2 3
  | .cudaDownloadOffset => .at 3 4
  | .cudaCreateBuffer | .cudaFreeBuffer
  | .cudaUpload | .cudaUploadAsync | .cudaUploadOffset | .cudaUploadOffsetAsync
  | .cudaLaunch | .cudaLaunchNamed | .cudaLaunchOnStream | .cudaLaunchNamedOnStream
  | .cudaSync
  | .cudaStreamCreate | .cudaStreamSync | .cudaStreamDestroy | .cudaStreamWaitEvent
  | .cudaEventCreate | .cudaEventRecord | .cudaEventElapsedMsBits | .cudaEventDestroy
  | .cudaGraphBeginCapture | .cudaGraphEndCapture | .cudaGraphUpload
  | .cudaGraphLaunch | .cudaGraphDestroy
  | .cudaPinnedAlloc | .cudaPinnedPtr | .cudaPinnedFree => .none
  -- Both return a number about memory rather than writing any: the pinned
  -- pool's checked address, and what the device has free.
  | .cudaPinnedPtrAt | .cudaMemInfoFree | .cudaMemInfoTotal => .none

  -- cuda.rs, cuBLAS — the operands and the result are all device buffers.
  | .cublasSgemv | .cublasSgemvOnStream | .cublasSgemm | .cublasSgemmOnStream
  | .cublasPtrArray | .cublasSgemmBatchedOnStream
  | .cublasGemmExBf16 => .none
  | .cublasGemmStridedBatchedExBf16 => .none

  -- wgpu.rs
  | .gpuInit | .gpuCleanup => ctxSlot
  | .gpuDownload => .at 2 3
  | .gpuDownloadPtr => .at 3 4
  | .gpuCreateBuffer | .gpuCreatePipeline | .gpuDispatch
  | .gpuUpload | .gpuUploadPtr => .none

  -- lmdb.rs — reads write their result where the caller asked, for as long as
  -- the stored value is.
  | .lmdbInit | .lmdbCleanup => ctxSlot
  | .lmdbCursorScan => .dataDependent 5
  | .lmdbOpen | .lmdbPut | .lmdbBeginWriteTxn | .lmdbCommitWriteTxn => .none

  -- thread.rs — a spawned body runs against the arena it is handed, so what it
  -- writes is the callee's frame, not this one's.
  | .threadInit | .threadCleanup => ctxSlot
  | .threadSpawn => .dataDependent 2
  | .threadJoin => .none

  -- window.rs
  | .windowInit | .windowCleanup => ctxSlot
  | .windowPoll => .dataDependent 1
  | .windowOpen | .windowPresentGpuBuffer => .none

/-- The frame declared for a symbol, or `none` when it is not an entry point —
    which is the honest answer for a program's own colocated functions, and the
    one a proof has to handle. -/
def frameOf (name : String) : Option Frame := (IR.Ffi.ofCname name).map frame

-- ---------------------------------------------------------------------------
-- What one program assumes
-- ---------------------------------------------------------------------------

/-- The symbol a callee index resolves to, when it is an import. -/
def calleeName (env : FnEnv) (fn : Nat) : Option String := do
  let d ← env.fns.find? (·.ref.id == fn)
  match d.callee with
  | .import n => some n
  | .local _ => none

/-- **The FFI a program actually assumes.**

    Not every entry point that exists — only the symbols this body calls, which
    `callsOf` already reads off the term. A program that computes and stores
    assumes nothing at all; the histogram assumes two.

    This is what makes the FFI part of the trusted base per-program and usually
    near-empty, instead of a fixed eighty-item liability every artifact
    carries. -/
def footprint (env : FnEnv) (c : Code) : List (String × Option Frame) :=
  (callsOf c).foldl
    (fun acc fn =>
      match calleeName env fn with
      | none => acc
      | some n => if acc.any (·.1 == n) then acc else acc ++ [(n, frameOf n)])
    []

/-- A program whose every callee has a declared frame — otherwise something it
    calls has no statement about what it may write, and no proof about that
    program can be complete. -/
def footprintComplete (env : FnEnv) (c : Code) : Bool :=
  (footprint env c).all (·.2.isSome)

/-- Rendered for the build log, so each artifact reports its own assumptions. -/
def footprintReport (env : FnEnv) (c : Code) : String :=
  match footprint env c with
  | [] => "assumes no FFI"
  | fs => s!"assumes {fs.length} FFI frame(s): " ++
          String.intercalate ", " (fs.map fun (n, f) =>
            match f with | some _ => n | none => n ++ " (NO FRAME)")

end AlgorithmLib.HProg
