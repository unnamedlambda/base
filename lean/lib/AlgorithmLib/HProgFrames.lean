import AlgorithmLib.HProgSem

/-!
# What a call may write

`HProgSem` gives four symbols an executable contract, because the corpus needs
to run them. The other eighty-nine are device drivers, sockets, windows and
database handles: nothing here will ever compute what they return. What can
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

/-- The frame of every symbol the JIT resolves.

    Ordered by module, matching `base/src/ffi/`. A symbol absent from this
    table has no frame, and `frameOf` says so rather than guessing. -/
def frames : List (String × Frame) :=
  [ -- file.rs — the read family writes into shared memory, the write family
    -- only touches the file system.
    ("cl_file_read", .atOffRet 0 2),
    ("cl_file_read_to_ptr", .at 1 3),
    ("cl_file_write", .none),
    ("cl_file_write_from_ptr", .none),

    -- stdio.rs
    ("cl_stdin_readline", .atOff 0 1 2),
    ("cl_stdout_write", .none),

    -- mod.rs — the libm shims are pure.
    ("cl_sinf", .none), ("cl_cosf", .none), ("cl_powf", .none),

    -- ht.rs — the table lives outside shared memory; only the lookups write
    -- back into it, and for as many bytes as the stored value happens to be.
    ("cl_ht_init", ctxSlot), ("cl_ht_cleanup", ctxSlot),
    ("cl_ht_create", .none), ("cl_ht_count", .none),
    ("cl_ht_insert", .none), ("cl_ht_increment", .none),
    ("cl_ht_lookup", .dataDependent 3),
    ("cl_ht_get_entry", .dataDependent2 2 3),

    -- cuda.rs — uploads and launches move data the other way or stay on the
    -- device; only the downloads write host memory.
    ("cl_cuda_init", ctxSlot), ("cl_cuda_cleanup", ctxSlot),
    ("cl_cuda_create_buffer", .none), ("cl_cuda_free_buffer", .none),
    ("cl_cuda_upload", .none), ("cl_cuda_upload_ptr", .none),
    ("cl_cuda_upload_ptr_async", .none),
    ("cl_cuda_upload_ptr_offset", .none),
    ("cl_cuda_upload_ptr_offset_async", .none),
    ("cl_cuda_download", .at 2 3),
    ("cl_cuda_download_ptr", .at 2 3),
    ("cl_cuda_download_ptr_async", .at 2 3),
    ("cl_cuda_download_ptr_offset", .at 3 4),
    ("cl_cuda_launch", .none), ("cl_cuda_launch_named", .none),
    ("cl_cuda_launch_on_stream", .none),
    ("cl_cuda_launch_named_on_stream", .none),
    ("cl_cuda_sync", .none),
    ("cl_cuda_stream_create", .none), ("cl_cuda_stream_sync", .none),
    ("cl_cuda_stream_destroy", .none), ("cl_cuda_stream_wait_event", .none),
    ("cl_cuda_event_create", .none), ("cl_cuda_event_record", .none),
    ("cl_cuda_event_elapsed_ms_bits", .none), ("cl_cuda_event_destroy", .none),
    ("cl_cuda_graph_begin_capture", .none), ("cl_cuda_graph_end_capture", .none),
    ("cl_cuda_graph_upload", .none), ("cl_cuda_graph_launch", .none),
    ("cl_cuda_graph_destroy", .none),
    ("cl_cuda_pinned_alloc", .none), ("cl_cuda_pinned_ptr", .none),
    ("cl_cuda_pinned_free", .none),

    -- cuda.rs, cuBLAS — the operands and the result are all device buffers.
    ("cl_cublas_sgemm", .none), ("cl_cublas_sgemv", .none),
    ("cl_cublas_sgemv_on_stream", .none),
    ("cl_cublas_sgemm_strided_batched", .none),
    ("cl_cublas_sgemm_strided_batched_on_stream", .none),

    -- wgpu.rs
    ("cl_gpu_init", ctxSlot), ("cl_gpu_cleanup", ctxSlot),
    ("cl_gpu_create_buffer", .none), ("cl_gpu_create_pipeline", .none),
    ("cl_gpu_dispatch", .none),
    ("cl_gpu_upload", .none), ("cl_gpu_upload_ptr", .none),
    ("cl_gpu_download", .at 2 3),
    ("cl_gpu_download_ptr", .at 3 4),

    -- lmdb.rs — reads write their result where the caller asked, for as long
    -- as the stored value is.
    ("cl_lmdb_init", ctxSlot), ("cl_lmdb_cleanup", ctxSlot),
    ("cl_lmdb_open", .none), ("cl_lmdb_sync", .none),
    ("cl_lmdb_put", .none), ("cl_lmdb_delete", .none),
    ("cl_lmdb_begin_write_txn", .none), ("cl_lmdb_commit_write_txn", .none),
    ("cl_lmdb_get", .dataDependent 4),
    ("cl_lmdb_cursor_scan", .dataDependent 5),

    -- net.rs
    ("cl_net_init", ctxSlot), ("cl_net_cleanup", ctxSlot),
    ("cl_net_listen", .none), ("cl_net_accept", .none),
    ("cl_net_connect", .none), ("cl_net_listener_port", .none),
    ("cl_net_send", .none),
    ("cl_net_recv", .at 2 3),

    -- thread.rs — a spawned body runs against the arena it is handed, so what
    -- it writes is the callee's frame, not this one's.
    ("cl_thread_init", ctxSlot), ("cl_thread_cleanup", ctxSlot),
    ("cl_thread_spawn", .dataDependent 2),
    ("cl_thread_call", .dataDependent 2),
    ("cl_thread_join", .none),

    -- window.rs
    ("cl_window_init", ctxSlot), ("cl_window_cleanup", ctxSlot),
    ("cl_window_open", .none),
    ("cl_window_poll", .dataDependent 1),
    ("cl_window_present_gpu_buffer", .none) ]

/-- The frame declared for `name`, or `none` when the table does not mention
    it — which is the honest answer, and the one a proof has to handle. -/
def frameOf (name : String) : Option Frame :=
  (frames.find? (·.1 == name)).map (·.2)

/-- Every symbol is named once, so `frameOf` cannot pick between two claims. -/
theorem frames_nodup :
    frames.all (fun e => (frames.filter (·.1 == e.1)).length == 1) = true := by
  native_decide

/-- The symbols `HProgSem.callFile` can actually run. Every other name in
    `frames` has a frame and no definition. -/
def executable : List String := ["cl_file_read", "cl_file_write"]

/-- Nothing claims to be executable without a frame to go with it. -/
theorem executable_have_frames :
    executable.all (fun n => (frameOf n).isSome) = true := by native_decide

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

    Not the 93 in `frames` — only the symbols this body calls, which `callsOf`
    already reads off the term. A program that computes and stores assumes
    nothing at all; the histogram assumes two.

    This is what makes the FFI part of the trusted base per-program and usually
    near-empty, instead of a fixed 93-item liability every artifact carries. -/
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
