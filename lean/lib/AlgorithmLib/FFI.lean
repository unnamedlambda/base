import AlgorithmLib.Core
import AlgorithmLib.Layout
import AlgorithmLib.IR

/-!
# The runtime surface, as data

Every entry point the runtime exports, one constructor each. What a callee is —
its C symbol, its signature, how it resolves — are total functions of the
constructor, so a signature is written in exactly one place and two files
cannot describe the same symbol differently.

`all` fixes the ids: a callee's id is its position in that list, and every
artifact's call instructions carry these ids. Appending a new entry point is
compatible with every shipped artifact; reordering renumbers every call.

The executable contracts — what a call *does*, transcribed from
`base/src/ffi/` — are in `HProgSem`; which memory a call may write is in
`HProgFrames`. This file is only who exists and how to call them.
-/

namespace AlgorithmLib.IR

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
  deriving Repr, BEq, DecidableEq, Inhabited

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
   .cublasGemmStridedBatchedExBf16]

/-- The callee id every artifact carries for `f`. -/
def id (f : Ffi) : Nat := all.idxOf f

def ref (f : Ffi) : FnRef := ⟨f.id⟩

/-- The entry point a name resolves to, if it is one. -/
def ofCname (s : String) : Option Ffi := all.find? (·.cname == s)

def sigDecl (f : Ffi) : SigDecl :=
  { ref := ⟨f.id⟩, params := f.params, result := f.result }

/-- Every entry point is a host symbol the JIT resolves by name. Whether a
    call may be PC-relative is the runtime's decision, not the declaration's:
    it knows where it placed the program's code and where the loader put the
    host's. -/
def fnDecl (f : Ffi) : FnDecl :=
  { ref := ⟨f.id⟩, callee := .import f.cname, sig := ⟨f.id⟩ }

end Ffi

/-- The table a list of entry points declares. Ids come from `Ffi.all`, so a
    selection keeps the ids the full table hands out and `FnEnv.sigOf` cannot
    land two names on one id. -/
def envFromFfi (fs : List Ffi) : FnEnv :=
  { sigs := fs.map Ffi.sigDecl, fns := fs.map Ffi.fnDecl }

namespace FFI

/-- A named group of entry points, so a body can be checked against the part of
    the table it uses. -/
inductive Bundle where
  | fileIO | gpu | window | lmdb | ht | math | thread | cuda | cublas
  deriving Repr, BEq

end FFI

/-- The bundle an entry point belongs to. -/
def Ffi.bundle : Ffi → FFI.Bundle
  | .fileRead | .fileWrite | .fileReadToPtr | .fileWriteFromPtr
  | .stdinReadline | .stdoutWrite => .fileIO
  | .gpuInit | .gpuCreateBuffer | .gpuCreatePipeline | .gpuUpload | .gpuDownload
  | .gpuDispatch | .gpuCleanup | .gpuUploadPtr | .gpuDownloadPtr => .gpu
  | .windowInit | .windowOpen | .windowPoll | .windowPresentGpuBuffer
  | .windowCleanup => .window
  | .lmdbInit | .lmdbOpen | .lmdbBeginWriteTxn | .lmdbPut | .lmdbCommitWriteTxn
  | .lmdbCursorScan | .lmdbCleanup => .lmdb
  | .htCreate | .htLookup | .htInsert | .htIncrement | .htCount | .htGetEntry
  | .htCleanup | .htInit => .ht
  | .sinf | .cosf | .powf => .math
  | .threadInit | .threadSpawn | .threadJoin | .threadCleanup => .thread
  | .cublasSgemv | .cublasSgemvOnStream | .cublasSgemm | .cublasSgemmOnStream
  | .cublasPtrArray | .cublasSgemmBatchedOnStream => .cublas
  | _ => .cuda

/-- The fn_idx of the main entry point that every application emits as `u0:1`
    (with `u0:0` reserved as a no-op stub). Use this in `Algorithm.fn_idx`. -/
def mainFnIdx : UInt32 := u32 1

end AlgorithmLib.IR
