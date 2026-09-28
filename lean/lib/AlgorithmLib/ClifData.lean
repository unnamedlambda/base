import Lean
import AlgorithmLib.Cbor

/-!
# The CLIF program an artifact carries

The instruction set the generators emit, as data, together with the CBOR
`base_types::clif` reads. Separate from `IR.lean` because `Core.Setup`
carries a `Program` and `IR` builds one — both need these types and neither
should import the other.
-/

namespace AlgorithmLib

namespace IR

/-- CLIF value types -/
inductive ClifTy where
  | i8 | i16 | i32 | i64
  | f32 | f64
  | f32x4 | i8x16
  deriving Repr, BEq, Lean.ToExpr

/-- An SSA value reference -/
structure Val where
  id : Nat
  deriving Repr, BEq

/-- A block reference -/
structure BlockRef where
  id : Nat
  deriving Repr, BEq

/-- Comparison condition codes -/
inductive ICmpCond where
  | eq | ne | uge | ugt | ule | ult | slt | sle | sgt | sge
  deriving Repr, BEq

/-- Float comparison conditions -/
inductive FloatCC where
  | eq | ne | lt | le | gt | ge
  deriving Repr, BEq

/-- Which load instruction, independent of the type it yields. -/
inductive LoadKind where
  | plain | uload8 | uload32 | sload8
  deriving Repr, BEq

/-- A load: what to read, as what type, under which memory flags. -/
structure LoadOp where
  kind : LoadKind := .plain
  ty : ClifTy
  /-- `notrap aligned`; the float and vector accessors set it. -/
  notrapAligned : Bool := false
  deriving Repr, BEq

/-- One runtime entry point. -/
inductive Ffi where
  | fileRead | fileWrite | fileReadToPtr | fileWriteFromPtr
  | stdinReadline | stdoutWrite
  | gpuInit | gpuCreateBuffer | gpuCreatePipeline | gpuUpload | gpuDownload
  | gpuDispatch | gpuCleanup | gpuUploadPtr | gpuDownloadPtr
  | windowInit | windowOpen | windowPoll | windowPresentGpuBuffer | windowCleanup
  | lmdbInit | lmdbOpen | lmdbBeginWriteTxn | lmdbPut | lmdbCommitWriteTxn
  | lmdbCursorScan | lmdbCleanup
  | htCreate | htLookup | htInsert | htIncrement | htCount | htGetEntry
  | htCleanup | htInit
  | sinf | cosf | powf
  | threadInit | threadSpawn | threadJoin | threadCleanup
  | cudaInit | cudaCreateBuffer | cudaUpload | cudaUploadOffset | cudaUploadAsync
  | cudaUploadOffsetAsync | cudaDownload | cudaDownloadOffset | cudaDownloadAsync
  | cudaFreeBuffer | cudaStreamCreate | cudaStreamSync | cudaStreamDestroy
  | cudaEventCreate | cudaEventRecord | cudaStreamWaitEvent | cudaEventElapsedMsBits
  | cudaEventDestroy | cudaGraphBeginCapture | cudaGraphEndCapture | cudaGraphUpload
  | cudaGraphLaunch | cudaGraphDestroy | cudaPinnedAlloc | cudaPinnedPtr
  | cudaPinnedFree | cudaLaunch | cudaLaunchNamed | cudaLaunchOnStream
  | cudaLaunchNamedOnStream | cudaSync | cudaCleanup
  | cublasSgemv | cublasSgemvOnStream | cublasSgemm | cublasSgemmOnStream
  | cublasPtrArray | cublasSgemmBatchedOnStream
  -- Appended, and appending is the rule: a callee's id is its position in
  -- `all`, so inserting one renumbers every shipped artifact's calls.
  | cudaPinnedPtrAt | cudaMemInfoFree | cudaMemInfoTotal | cublasGemmExBf16
  | cublasGemmStridedBatchedExBf16
  | nativeLoad | nativeFree | nativeArch | cpuHas
  deriving Repr, BEq, DecidableEq, Inhabited, Lean.ToExpr

namespace Ffi

/-- The C symbol the JIT resolves. -/
def cname : Ffi → String
  | .fileRead => "cl_file_read"
  | .fileWrite => "cl_file_write"
  | .fileReadToPtr => "cl_file_read_to_ptr"
  | .fileWriteFromPtr => "cl_file_write_from_ptr"
  | .stdinReadline => "cl_stdin_readline"
  | .stdoutWrite => "cl_stdout_write"
  | .gpuInit => "cl_gpu_init"
  | .gpuCreateBuffer => "cl_gpu_create_buffer"
  | .gpuCreatePipeline => "cl_gpu_create_pipeline"
  | .gpuUpload => "cl_gpu_upload"
  | .gpuDownload => "cl_gpu_download"
  | .gpuDispatch => "cl_gpu_dispatch"
  | .gpuCleanup => "cl_gpu_cleanup"
  | .gpuUploadPtr => "cl_gpu_upload_ptr"
  | .gpuDownloadPtr => "cl_gpu_download_ptr"
  | .windowInit => "cl_window_init"
  | .windowOpen => "cl_window_open"
  | .windowPoll => "cl_window_poll"
  | .windowPresentGpuBuffer => "cl_window_present_gpu_buffer"
  | .windowCleanup => "cl_window_cleanup"
  | .lmdbInit => "cl_lmdb_init"
  | .lmdbOpen => "cl_lmdb_open"
  | .lmdbBeginWriteTxn => "cl_lmdb_begin_write_txn"
  | .lmdbPut => "cl_lmdb_put"
  | .lmdbCommitWriteTxn => "cl_lmdb_commit_write_txn"
  | .lmdbCursorScan => "cl_lmdb_cursor_scan"
  | .lmdbCleanup => "cl_lmdb_cleanup"
  | .htCreate => "ht_create"
  | .htLookup => "ht_lookup"
  | .htInsert => "ht_insert"
  | .htIncrement => "ht_increment"
  | .htCount => "ht_count"
  | .htGetEntry => "ht_get_entry"
  | .htCleanup => "cl_ht_cleanup"
  | .htInit => "cl_ht_init"
  | .sinf => "cl_sinf"
  | .cosf => "cl_cosf"
  | .powf => "cl_powf"
  | .threadInit => "cl_thread_init"
  | .threadSpawn => "cl_thread_spawn"
  | .threadJoin => "cl_thread_join"
  | .threadCleanup => "cl_thread_cleanup"
  | .cudaInit => "cl_cuda_init"
  | .cudaCreateBuffer => "cl_cuda_create_buffer"
  | .cudaUpload => "cl_cuda_upload_ptr"
  | .cudaUploadOffset => "cl_cuda_upload_ptr_offset"
  | .cudaUploadAsync => "cl_cuda_upload_ptr_async"
  | .cudaUploadOffsetAsync => "cl_cuda_upload_ptr_offset_async"
  | .cudaDownload => "cl_cuda_download_ptr"
  | .cudaDownloadOffset => "cl_cuda_download_ptr_offset"
  | .cudaDownloadAsync => "cl_cuda_download_ptr_async"
  | .cudaFreeBuffer => "cl_cuda_free_buffer"
  | .cudaStreamCreate => "cl_cuda_stream_create"
  | .cudaStreamSync => "cl_cuda_stream_sync"
  | .cudaStreamDestroy => "cl_cuda_stream_destroy"
  | .cudaEventCreate => "cl_cuda_event_create"
  | .cudaEventRecord => "cl_cuda_event_record"
  | .cudaStreamWaitEvent => "cl_cuda_stream_wait_event"
  | .cudaEventElapsedMsBits => "cl_cuda_event_elapsed_ms_bits"
  | .cudaEventDestroy => "cl_cuda_event_destroy"
  | .cudaGraphBeginCapture => "cl_cuda_graph_begin_capture"
  | .cudaGraphEndCapture => "cl_cuda_graph_end_capture"
  | .cudaGraphUpload => "cl_cuda_graph_upload"
  | .cudaGraphLaunch => "cl_cuda_graph_launch"
  | .cudaGraphDestroy => "cl_cuda_graph_destroy"
  | .cudaPinnedAlloc => "cl_cuda_pinned_alloc"
  | .cudaPinnedPtr => "cl_cuda_pinned_ptr"
  | .cudaPinnedFree => "cl_cuda_pinned_free"
  | .cudaLaunch => "cl_cuda_launch"
  | .cudaLaunchNamed => "cl_cuda_launch_named"
  | .cudaLaunchOnStream => "cl_cuda_launch_on_stream"
  | .cudaLaunchNamedOnStream => "cl_cuda_launch_named_on_stream"
  | .cudaSync => "cl_cuda_sync"
  | .cudaCleanup => "cl_cuda_cleanup"
  | .cublasSgemv => "cl_cublas_sgemv"
  | .cublasSgemvOnStream => "cl_cublas_sgemv_on_stream"
  | .cublasSgemm => "cl_cublas_sgemm_strided_batched"
  | .cublasSgemmOnStream => "cl_cublas_sgemm_strided_batched_on_stream"
  | .cublasPtrArray => "cl_cublas_ptr_array"
  | .cublasSgemmBatchedOnStream => "cl_cublas_sgemm_batched_on_stream"
  | .cudaPinnedPtrAt => "cl_cuda_pinned_ptr_at"
  | .cudaMemInfoFree => "cl_cuda_mem_info_free"
  | .cudaMemInfoTotal => "cl_cuda_mem_info_total"
  | .cublasGemmExBf16 => "cl_cublas_gemm_ex_bf16"
  | .cublasGemmStridedBatchedExBf16 => "cl_cublas_gemm_strided_batched_ex_bf16"
  | .nativeLoad => "cl_native_load"
  | .nativeFree => "cl_native_free"
  | .nativeArch => "cl_native_arch"
  | .cpuHas => "cl_cpu_has"

/-- Parameters and result, exactly as `base/src/ffi/` takes them. -/
def sig : Ffi → List ClifTy × Option ClifTy
  | .fileRead => ([.i64, .i64, .i64, .i64, .i64], some .i64)
  | .fileWrite => ([.i64, .i64, .i64, .i64, .i64], some .i64)
  | .fileReadToPtr => ([.i64, .i64, .i64, .i64], some .i64)
  | .fileWriteFromPtr => ([.i64, .i64, .i64, .i64], some .i64)
  | .stdinReadline => ([.i64, .i64, .i64], some .i64)
  | .stdoutWrite => ([.i64, .i64, .i64], some .i64)
  | .gpuInit => ([.i64], none)
  | .gpuCreateBuffer => ([.i64, .i64], some .i32)
  | .gpuCreatePipeline => ([.i64, .i64, .i64, .i32], some .i32)
  | .gpuUpload => ([.i64, .i32, .i64, .i64], some .i32)
  | .gpuDownload => ([.i64, .i32, .i64, .i64], some .i32)
  | .gpuDispatch => ([.i64, .i32, .i32, .i32, .i32], some .i32)
  | .gpuCleanup => ([.i64], none)
  | .gpuUploadPtr => ([.i64, .i32, .i64, .i64], some .i32)
  | .gpuDownloadPtr => ([.i64, .i32, .i64, .i64, .i64], some .i32)
  | .windowInit => ([.i64], none)
  | .windowOpen => ([.i64, .i64, .i64, .i64, .i64, .i64, .i64], some .i32)
  | .windowPoll => ([.i64, .i64, .i32], some .i32)
  | .windowPresentGpuBuffer => ([.i64, .i64, .i32], some .i32)
  | .windowCleanup => ([.i64], none)
  | .lmdbInit => ([.i64], none)
  | .lmdbOpen => ([.i64, .i64, .i32], some .i32)
  | .lmdbBeginWriteTxn => ([.i64, .i32], some .i32)
  | .lmdbPut => ([.i64, .i32, .i64, .i32, .i64, .i32], some .i32)
  | .lmdbCommitWriteTxn => ([.i64, .i32], some .i32)
  | .lmdbCursorScan => ([.i64, .i32, .i64, .i32, .i32, .i64], some .i32)
  | .lmdbCleanup => ([.i64], none)
  | .htCreate => ([.i64], some .i32)
  | .htLookup => ([.i64, .i64, .i32, .i64], some .i32)
  | .htInsert => ([.i64, .i64, .i32, .i64, .i32], none)
  | .htIncrement => ([.i64, .i64, .i32, .i64], some .i64)
  | .htCount => ([.i64], some .i32)
  | .htGetEntry => ([.i64, .i32, .i64, .i64], some .i32)
  | .htCleanup => ([.i64], none)
  | .htInit => ([.i64], none)
  | .sinf => ([.f32], some .f32)
  | .cosf => ([.f32], some .f32)
  | .powf => ([.f32, .f32], some .f32)
  | .threadInit => ([.i64], none)
  | .threadSpawn => ([.i64, .i64, .i64], some .i64)
  | .threadJoin => ([.i64, .i64], some .i64)
  | .threadCleanup => ([.i64], none)
  | .cudaInit => ([.i64], none)
  | .cudaCreateBuffer => ([.i64, .i64], some .i32)
  | .cudaUpload => ([.i64, .i32, .i64, .i64], some .i32)
  | .cudaUploadOffset => ([.i64, .i32, .i64, .i64, .i64], some .i32)
  | .cudaUploadAsync => ([.i64, .i32, .i64, .i64, .i32], some .i32)
  | .cudaUploadOffsetAsync => ([.i64, .i32, .i64, .i64, .i64, .i32], some .i32)
  | .cudaDownload => ([.i64, .i32, .i64, .i64], some .i32)
  | .cudaDownloadOffset => ([.i64, .i32, .i64, .i64, .i64], some .i32)
  | .cudaDownloadAsync => ([.i64, .i32, .i64, .i64, .i32], some .i32)
  | .cudaFreeBuffer => ([.i64, .i32], some .i32)
  | .cudaStreamCreate => ([.i64], some .i32)
  | .cudaStreamSync => ([.i64, .i32], some .i32)
  | .cudaStreamDestroy => ([.i64, .i32], some .i32)
  | .cudaEventCreate => ([.i64], some .i32)
  | .cudaEventRecord => ([.i64, .i32, .i32], some .i32)
  | .cudaStreamWaitEvent => ([.i64, .i32, .i32], some .i32)
  | .cudaEventElapsedMsBits => ([.i64, .i32, .i32], some .i32)
  | .cudaEventDestroy => ([.i64, .i32], some .i32)
  | .cudaGraphBeginCapture => ([.i64, .i32], some .i32)
  | .cudaGraphEndCapture => ([.i64, .i32], some .i32)
  | .cudaGraphUpload => ([.i64, .i32, .i32], some .i32)
  | .cudaGraphLaunch => ([.i64, .i32, .i32], some .i32)
  | .cudaGraphDestroy => ([.i64, .i32], some .i32)
  | .cudaPinnedAlloc => ([.i64, .i64], some .i32)
  | .cudaPinnedPtr => ([.i64, .i32], some .i64)
  | .cudaPinnedFree => ([.i64, .i32], some .i32)
  | .cudaLaunch => ([.i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  | .cudaLaunchNamed =>
      ([.i64, .i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  | .cudaLaunchOnStream =>
      ([.i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  | .cudaLaunchNamedOnStream =>
      ([.i64, .i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  | .cudaSync => ([.i64], some .i32)
  | .cudaCleanup => ([.i64], none)
  | .cublasSgemv => ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  | .cublasSgemvOnStream =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32], some .i32)
  -- The three `.i64`s after the batch count are element offsets into the A, B
  -- and C operands; a zero trailing `ld_*` asks for the default leading
  -- dimension. An offset moves the pointer and leaves the matrix the call
  -- contracts alone, so one buffer can hold several operands without touching
  -- what `Law.cublasIsMatvec` says a contraction computes.
  | .cublasSgemm =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32,
        .i32, .i64, .i32, .i64, .i64, .i64, .i32, .i32, .i32], some .i32)
  | .cublasSgemmOnStream =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32,
        .i32, .i64, .i32, .i32, .i64, .i64, .i64, .i32, .i32, .i32], some .i32)
  | .cublasPtrArray => ([.i64, .i32, .i32, .i32, .i64], some .i32)
  | .cublasSgemmBatchedOnStream =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32,
        .i32], some .i32)
  -- `(ctx, pinned_id, off, len)`: the pinned pool's host address at `off`, or
  -- `-1` when `off + len` runs past the allocation. The bound is the point --
  -- `cudaUploadOffsetAsync` checks the device range it writes but takes its
  -- source as a bare address, so an unchecked offset into a multi-gigabyte pool
  -- uploads whatever the process has there and calls it a weight.
  | .cudaPinnedPtrAt => ([.i64, .i32, .i64, .i64], some .i64)
  -- Free and total device memory. What a card has left after weights, caches
  -- and the driver's own reservations is not a number that can be written down
  -- ahead of the machine, so the sizes that depend on it are read here.
  | .cudaMemInfoFree => ([.i64], some .i64)
  | .cudaMemInfoTotal => ([.i64], some .i64)
  -- `cublasGemmEx` over bf16 operands with an f32 accumulator and result.
  -- Same argument shape as `.cublasSgemm` minus the batching: transposes, the
  -- three dimensions, alpha/A/B, beta/C, then the three element offsets and the
  -- three leading dimensions. BOTH inputs are bf16 -- cuBLAS rejects a mixed
  -- (bf16, f32) pair -- so `off_a` and `off_b` count 2-byte elements while
  -- `off_c` counts 4-byte ones.
  | .cublasGemmExBf16 =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32,
        .i64, .i64, .i64, .i32, .i32, .i32], some .i32)
  -- `.cublasSgemmBatchedOnStream`'s shape with `.cublasGemmExBf16`'s element
  -- types: transposes, dimensions, then (alpha, A, strideA), (B, strideB),
  -- (beta, C, strideC), the batch count, and the offsets and leading
  -- dimensions. Strides and offsets count elements, and an element is two
  -- bytes on both inputs and four on the result.
  | .cublasGemmStridedBatchedExBf16 =>
      ([.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64,
        .i32, .i32, .i64, .i32, .i64, .i64, .i64, .i32, .i32, .i32], some .i32)
  -- ffi/native.rs: machine code the program carries as data. `load` takes the
  -- bytes' address and length and answers where they now run (0 if they could
  -- not be placed); `arch` and `cpuHas` (a NUL-terminated feature name) decide
  -- which bytes to carry in. Running them is not an import: it is a call with
  -- `Callee.native`.
  | .nativeLoad => ([.i64, .i64], some .i64)
  | .nativeFree => ([.i64], some .i32)
  | .nativeArch => ([], some .i32)
  | .cpuHas => ([.i64], some .i32)

def params (f : Ffi) : List ClifTy := f.sig.1
def result (f : Ffi) : Option ClifTy := f.sig.2

/-- Every entry point, in the order that fixes the ids. -/
def all : List Ffi :=
  [.fileRead, .fileWrite, .fileReadToPtr, .fileWriteFromPtr,
   .stdinReadline, .stdoutWrite,
   .gpuInit, .gpuCreateBuffer, .gpuCreatePipeline, .gpuUpload, .gpuDownload,
   .gpuDispatch, .gpuCleanup, .gpuUploadPtr, .gpuDownloadPtr,
   .windowInit, .windowOpen, .windowPoll, .windowPresentGpuBuffer, .windowCleanup,
   .lmdbInit, .lmdbOpen, .lmdbBeginWriteTxn, .lmdbPut, .lmdbCommitWriteTxn,
   .lmdbCursorScan, .lmdbCleanup,
   .htCreate, .htLookup, .htInsert, .htIncrement, .htCount, .htGetEntry,
   .htCleanup, .htInit,
   .sinf, .cosf, .powf,
   .threadInit, .threadSpawn, .threadJoin, .threadCleanup,
   .cudaInit, .cudaCreateBuffer, .cudaUpload, .cudaUploadOffset, .cudaUploadAsync,
   .cudaUploadOffsetAsync, .cudaDownload, .cudaDownloadOffset, .cudaDownloadAsync,
   .cudaFreeBuffer, .cudaStreamCreate, .cudaStreamSync, .cudaStreamDestroy,
   .cudaEventCreate, .cudaEventRecord, .cudaStreamWaitEvent, .cudaEventElapsedMsBits,
   .cudaEventDestroy, .cudaGraphBeginCapture, .cudaGraphEndCapture, .cudaGraphUpload,
   .cudaGraphLaunch, .cudaGraphDestroy, .cudaPinnedAlloc, .cudaPinnedPtr,
   .cudaPinnedFree, .cudaLaunch, .cudaLaunchNamed, .cudaLaunchOnStream,
   .cudaLaunchNamedOnStream, .cudaSync, .cudaCleanup,
   .cublasSgemv, .cublasSgemvOnStream, .cublasSgemm, .cublasSgemmOnStream,
   .cublasPtrArray, .cublasSgemmBatchedOnStream,
   .cudaPinnedPtrAt, .cudaMemInfoFree, .cudaMemInfoTotal, .cublasGemmExBf16,
   .cublasGemmStridedBatchedExBf16,
   .nativeLoad, .nativeFree, .nativeArch, .cpuHas]

/-- The callee id every artifact carries for `f`. -/
def id (f : Ffi) : Nat := all.idxOf f

/-- The entry point a name resolves to, if it is one. -/
def ofCname (s : String) : Option Ffi := all.find? (·.cname == s)

end Ffi

/-- What a call names.

    Each arm is the identity its owner gives it: an import is named by the
    engine's own entry point, a defined function by its place in this artifact,
    which is closed. Machine code the program placed itself has no name at
    all: its address is a value, the call's first argument. Nothing here is an
    index into a table this format invented, which is what a call carrying a
    position into a per-function callee list was.

    An import that the engine does not provide is unrepresentable rather than
    refused at load: `Ffi` is the only way to name one. -/
inductive Callee where
  /-- An entry point of the engine, resolved by its `cname` through the JIT's
      symbol table. -/
  | ffi (f : Ffi)
  /-- Another function of this same program, by its `u0:N` position. -/
  | local (index : Nat)
  /-- Machine code at the address in the call's first argument --- one
      `nativeLoad` answered --- called on the other four under the
      architecture's C convention (System V on x86-64 whatever the OS), and
      answering an `i64`. A plain indirect call: nothing of the engine's runs
      between the caller and the code. -/
  | native
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

/-- A single CLIF instruction -/
inductive Inst where
  | iconst (dst : Val) (ty : ClifTy) (value : Int)
  | iadd (dst : Val) (a b : Val)
  | isub (dst : Val) (a b : Val)
  | imul (dst : Val) (a b : Val)
  | udiv (dst : Val) (a b : Val)
  | ineg (dst : Val) (a : Val)
  | ishl (dst : Val) (a b : Val)
  | ushr (dst : Val) (a b : Val)
  | band (dst : Val) (a b : Val)
  | bandNot (dst : Val) (a b : Val)
  | bor (dst : Val) (a b : Val)
  | bxor (dst : Val) (a b : Val)
  | ireduce32 (dst : Val) (a : Val)
  | uextend64 (dst : Val) (a : Val)
  | sextend64 (dst : Val) (a : Val)
  | store (val addr : Val)
  | istore8 (val addr : Val)
  | load (dst : Val) (op : LoadOp) (addr : Val)
  | icmp (dst : Val) (cond : ICmpCond) (a b : Val)
  | select (dst : Val) (cond a b : Val)
  | call (dst : Option Val) (callee : Callee) (args : List Val)
  | jump (target : BlockRef) (args : List Val)
  | brif (cond : Val) (thenBlk : BlockRef) (thenArgs : List Val)
         (elseBlk : BlockRef) (elseArgs : List Val)
  /-- `return v`, or `return` when a function answers nothing. A body whose
      `ret` carries a value is a body whose signature returns an `i64`: the
      runtime reads the signature off the body rather than being told. -/
  | ret (value : Option Val)
  -- Float / SIMD
  | fconst (dst : Val) (ty : ClifTy) (bits : UInt64)
  | fadd (dst a b : Val)
  | fsub (dst a b : Val)
  | fmul (dst a b : Val)
  | fmax (dst a b : Val)
  | fmin (dst a b : Val)
  | fpromote (dst a : Val)
  | splat (dst : Val) (ty : ClifTy) (src : Val)
  | extractlane (dst : Val) (src : Val) (lane : Nat)
  | storeTyped (ty : ClifTy) (val addr : Val)
  -- Additional float / int ops
  | fneg (dst a : Val)
  | fcvtFromSint (dst : Val) (ty : ClifTy) (src : Val)
  /-- Saturating float-to-unsigned conversion.  Saturating rather than trapping
      so an out-of-range value clamps instead of aborting the process — the
      only consumer is a token id read back from a kernel that computed it as
      an exactly-representable integer. -/
  | fcvtToUint (dst : Val) (ty : ClifTy) (src : Val)
  | fcmp (dst : Val) (cond : FloatCC) (a b : Val)
  | bitcast (dst : Val) (ty : ClifTy) (src : Val)
  /-- Lane-wise `c ? a : b` on the *bits* of `c`, which is how a vector
      comparison's all-ones/all-zeros mask is consumed. -/
  | bitselect (dst c a b : Val)
  | ctz (dst a : Val)
  | popcnt (dst a : Val)
  | vhighBits (dst a : Val)
  deriving BEq

/-- A finalized block -/
structure BlockData where
  ref : BlockRef
  params : List (Val × ClifTy)
  insts : List Inst


-- ---------------------------------------------------------------------------
-- The emitted program
-- ---------------------------------------------------------------------------

/-- One function of the emitted program.

    `index` is the `u0:N` it was compiled at, which is its position in the
    artifact and is not written out: `Prog.program` checks the two agree.
    `entryName` is the name a host calls it by, and a function without one is
    the program's own. -/
structure FuncData where
  index : Nat
  blocks : List BlockData
  entryName : Option String := none

-- ---------------------------------------------------------------------------
-- Serialization
--
-- The CBOR `base_types::clif` reads (see `Cbor`): externally tagged enums,
-- tuple variants whose fields are in constructor order, newtypes as their
-- number. Field names and their order are the Rust ones; a disagreement is a
-- build failure, because the build re-encodes every artifact it decodes.
-- ---------------------------------------------------------------------------

open Cbor

instance : ToCbor Val where
  cbor v := nat v.id
instance : ToCbor BlockRef where
  cbor b := nat b.id

instance : ToCbor ClifTy where
  cbor t := text <| match t with
    | .i8 => "I8" | .i16 => "I16" | .i32 => "I32" | .i64 => "I64"
    | .f32 => "F32" | .f64 => "F64" | .f32x4 => "F32x4" | .i8x16 => "I8x16"

instance : ToCbor ICmpCond where
  cbor c := text <| match c with
    | .eq => "Eq" | .ne => "Ne" | .uge => "Uge" | .ugt => "Ugt" | .ule => "Ule"
    | .ult => "Ult" | .slt => "Slt" | .sle => "Sle" | .sgt => "Sgt" | .sge => "Sge"

instance : ToCbor FloatCC where
  cbor c := text <| match c with
    | .eq => "Eq" | .ne => "Ne" | .lt => "Lt" | .le => "Le" | .gt => "Gt" | .ge => "Ge"

instance : ToCbor LoadKind where
  cbor k := text <| match k with
    | .plain => "Plain" | .uload8 => "Uload8"
    | .uload32 => "Uload32" | .sload8 => "Sload8"

instance : ToCbor LoadOp where
  cbor op := struct
    [("kind", cbor op.kind),
     ("ty", cbor op.ty),
     ("notrap_aligned", bool op.notrapAligned)]

/-- An import travels as the symbol the engine resolves, which is the name
    `Ffi` already fixes; a defined function as its position. The constructor
    does not travel: the wire says what the engine needs to resolve, and the
    restriction to entry points the engine has belongs on this side of it. -/
instance : ToCbor Callee where
  cbor
    | .ffi f   => newtypeVariant "Import" (text f.cname)
    | .local i => newtypeVariant "Local" (nat i)
    | .native  => text "Native"

/-- One instruction, in the shape `base_types::clif::Inst` reads.

    Stores and loads carry a byte offset on the Rust side that no builder here
    emits yet, so it is written as zero. -/
def Inst.toCbor : Inst → W Unit
  | .iconst d t v => variant "Iconst" [cbor d, cbor t, int v]
  | .iadd d a b => variant "Iadd" [cbor d, cbor a, cbor b]
  | .isub d a b => variant "Isub" [cbor d, cbor a, cbor b]
  | .imul d a b => variant "Imul" [cbor d, cbor a, cbor b]
  | .udiv d a b => variant "Udiv" [cbor d, cbor a, cbor b]
  | .ineg d a => variant "Ineg" [cbor d, cbor a]
  | .ishl d a b => variant "Ishl" [cbor d, cbor a, cbor b]
  | .ushr d a b => variant "Ushr" [cbor d, cbor a, cbor b]
  | .band d a b => variant "Band" [cbor d, cbor a, cbor b]
  | .bandNot d a b => variant "BandNot" [cbor d, cbor a, cbor b]
  | .bor d a b => variant "Bor" [cbor d, cbor a, cbor b]
  | .bxor d a b => variant "Bxor" [cbor d, cbor a, cbor b]
  | .ireduce32 d a => variant "Ireduce32" [cbor d, cbor a]
  | .uextend64 d a => variant "Uextend64" [cbor d, cbor a]
  | .sextend64 d a => variant "Sextend64" [cbor d, cbor a]
  | .store v a => variant "Store" [cbor v, cbor a, nat 0]
  | .istore8 v a => variant "Istore8" [cbor v, cbor a, nat 0]
  | .load d op a => variant "Load" [cbor d, cbor op, cbor a, nat 0]
  | .icmp d c a b => variant "Icmp" [cbor d, cbor c, cbor a, cbor b]
  | .select d c a b => variant "Select" [cbor d, cbor c, cbor a, cbor b]
  | .call d f args =>
    variant "Call" [option cbor d,
                   cbor f, array args cbor]
  | .jump t args => variant "Jump" [cbor t, array args cbor]
  | .brif c tb ta eb ea =>
    variant "Brif" [cbor c, cbor tb, array ta cbor,
                   cbor eb, array ea cbor]
  -- A newtype variant, so the payload sits directly under the tag rather than
  -- in an array the way the tuple variants above do.
  | .ret v => newtypeVariant "Ret" (option cbor v)
  | .fconst d t bits => variant "Fconst" [cbor d, cbor t, nat bits.toNat]
  | .fadd d a b => variant "Fadd" [cbor d, cbor a, cbor b]
  | .fsub d a b => variant "Fsub" [cbor d, cbor a, cbor b]
  | .fmul d a b => variant "Fmul" [cbor d, cbor a, cbor b]
  | .fmax d a b => variant "Fmax" [cbor d, cbor a, cbor b]
  | .fmin d a b => variant "Fmin" [cbor d, cbor a, cbor b]
  | .fpromote d a => variant "Fpromote" [cbor d, cbor a]
  | .splat d t s => variant "Splat" [cbor d, cbor t, cbor s]
  | .extractlane d s lane => variant "Extractlane" [cbor d, cbor s, nat lane]
  | .storeTyped t v a => variant "StoreTyped" [cbor t, cbor v, cbor a, nat 0]
  | .fneg d a => variant "Fneg" [cbor d, cbor a]
  | .fcvtFromSint d t s => variant "FcvtFromSint" [cbor d, cbor t, cbor s]
  | .fcvtToUint d t s => variant "FcvtToUint" [cbor d, cbor t, cbor s]
  | .fcmp d c a b => variant "Fcmp" [cbor d, cbor c, cbor a, cbor b]
  | .bitcast d t s => variant "Bitcast" [cbor d, cbor t, cbor s]
  | .bitselect d c a b => variant "Bitselect" [cbor d, cbor c, cbor a, cbor b]
  | .ctz d a => variant "Ctz" [cbor d, cbor a]
  | .popcnt d a => variant "Popcnt" [cbor d, cbor a]
  | .vhighBits d a => variant "VhighBits" [cbor d, cbor a]



instance : ToCbor Inst where
  cbor := Inst.toCbor

instance : ToCbor BlockData where
  cbor b := struct
    [("reference", cbor b.ref),
     ("params", array b.params fun (v, t) => do head 4 2; cbor v; cbor t),
     ("insts", array b.insts cbor)]


instance : ToCbor FuncData where
  cbor f := struct
    [("entry_name", option text f.entryName),
     ("blocks", array f.blocks cbor)]

end IR

end AlgorithmLib
