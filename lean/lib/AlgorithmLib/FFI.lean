import AlgorithmLib.Core
import AlgorithmLib.Layout
import AlgorithmLib.IR

namespace AlgorithmLib

namespace IR

/-- Declare cl_file_read: (ptr, fname_off, data_off, file_offset, size) -> bytes_read -/
def declareFileRead : IRBuilder FnRef :=
  declareFFI "cl_file_read" [.i64, .i64, .i64, .i64, .i64] (some .i64)

/-- Declare cl_file_write: (ptr, fname_off, src_off, file_offset, size) -> bytes_written -/
def declareFileWrite : IRBuilder FnRef :=
  declareFFI "cl_file_write" [.i64, .i64, .i64, .i64, .i64] (some .i64)

/-- Declare cl_file_read_to_ptr: (ptr, fname_off, dst_ptr, size) -> bytes_read -/
def declareFileReadToPtr : IRBuilder FnRef :=
  declareFFI "cl_file_read_to_ptr" [.i64, .i64, .i64, .i64] (some .i64)

/-- Declare cl_file_write_from_ptr: (ptr, fname_off, src_ptr, size) -> bytes_written -/
def declareFileWriteFromPtr : IRBuilder FnRef :=
  declareFFI "cl_file_write_from_ptr" [.i64, .i64, .i64, .i64] (some .i64)

/-- Declare cl_stdin_readline: (ptr, dst_off, max_len) -> bytes_read -/
def declareStdinReadline : IRBuilder FnRef :=
  declareFFI "cl_stdin_readline" [.i64, .i64, .i64] (some .i64)

/-- Declare cl_stdout_write: (ptr, src_off, size) -> bytes_written -/
def declareStdoutWrite : IRBuilder FnRef :=
  declareFFI "cl_stdout_write" [.i64, .i64, .i64] (some .i64)

/-- GPU FFI function bundle -/
structure GpuSetup where
  fnInit : FnRef
  fnCreateBuffer : FnRef
  fnUpload : FnRef
  fnDownload : FnRef
  fnCreatePipeline : FnRef
  fnDispatch : FnRef
  fnCleanup : FnRef
  /-- `(ctx, buf, src_ptr, size)`: upload from a raw host pointer. -/
  fnUploadPtr : FnRef
  /-- `(ctx, buf, dst_ptr, size, buf_offset)`: download to a raw host pointer. -/
  fnDownloadPtr : FnRef
  deriving Inhabited, Lean.ToExpr

/-- Declare every GPU entry point. -/
def declareGpuFFI : IRBuilder GpuSetup := do
  let fnInit ← declareFFI "cl_gpu_init" [.i64] none
  let fnCreateBuffer ← declareFFI "cl_gpu_create_buffer" [.i64, .i64] (some .i32)
  let fnCreatePipeline ← declareFFI "cl_gpu_create_pipeline" [.i64, .i64, .i64, .i32] (some .i32)
  let fnUpload ← declareFFI "cl_gpu_upload" [.i64, .i32, .i64, .i64] (some .i32)
  let fnDownload ← declareFFI "cl_gpu_download" [.i64, .i32, .i64, .i64] (some .i32)
  let fnDispatch ← declareFFI "cl_gpu_dispatch" [.i64, .i32, .i32, .i32, .i32] (some .i32)
  let fnCleanup ← declareFFI "cl_gpu_cleanup" [.i64] none
  let fnUploadPtr ← declareFFI "cl_gpu_upload_ptr" [.i64, .i32, .i64, .i64] (some .i32)
  let fnDownloadPtr ← declareFFI "cl_gpu_download_ptr" [.i64, .i32, .i64, .i64, .i64] (some .i32)
  pure { fnInit, fnCreateBuffer, fnUpload, fnDownload, fnCreatePipeline, fnDispatch, fnCleanup,
         fnUploadPtr, fnDownloadPtr }


/-- Window / input / present FFI bundle. The window shares the wgpu device, so
    `fnPresentGpuBuffer` blits a game framebuffer straight from a wgpu storage
    buffer to the swapchain with no host round trip. -/
structure WindowSetup where
  fnInit : FnRef
  fnOpen : FnRef
  fnPoll : FnRef
  fnPresentGpuBuffer : FnRef
  fnCleanup : FnRef
  deriving Inhabited, Lean.ToExpr

/-- Declare the minimal window FFI (init/open/poll/present/cleanup). -/
def declareWindowFFI : IRBuilder WindowSetup := do
  let fnInit ← declareFFI "cl_window_init" [.i64] none
  let fnOpen ← declareFFI "cl_window_open" [.i64, .i64, .i64, .i64, .i64, .i64, .i64] (some .i32)
  let fnPoll ← declareFFI "cl_window_poll" [.i64, .i64, .i32] (some .i32)
  let fnPresentGpuBuffer ← declareFFI "cl_window_present_gpu_buffer"
    [.i64, .i64, .i32] (some .i32)
  let fnCleanup ← declareFFI "cl_window_cleanup" [.i64] none
  pure { fnInit, fnOpen, fnPoll, fnPresentGpuBuffer, fnCleanup }


/-- LMDB FFI function bundle -/
structure LmdbSetup where
  fnInit : FnRef
  fnOpen : FnRef
  fnBeginWriteTxn : FnRef
  fnPut : FnRef
  fnCommitWriteTxn : FnRef
  fnCursorScan : FnRef
  fnCleanup : FnRef
  deriving Inhabited, Lean.ToExpr

/-- Declare all 7 LMDB FFI functions -/
def declareLmdbFFI : IRBuilder LmdbSetup := do
  let fnInit ← declareFFI "cl_lmdb_init" [.i64] none
  let fnOpen ← declareFFI "cl_lmdb_open" [.i64, .i64, .i32] (some .i32)
  let fnBeginWriteTxn ← declareFFI "cl_lmdb_begin_write_txn" [.i64, .i32] (some .i32)
  let fnPut ← declareFFI "cl_lmdb_put" [.i64, .i32, .i64, .i32, .i64, .i32] (some .i32)
  let fnCommitWriteTxn ← declareFFI "cl_lmdb_commit_write_txn" [.i64, .i32] (some .i32)
  let fnCursorScan ← declareFFI "cl_lmdb_cursor_scan" [.i64, .i32, .i64, .i32, .i32, .i64] (some .i32)
  let fnCleanup ← declareFFI "cl_lmdb_cleanup" [.i64] none
  pure { fnInit, fnOpen, fnBeginWriteTxn, fnPut, fnCommitWriteTxn, fnCursorScan, fnCleanup }

-- ---------------------------------------------------------------------------
-- Hash-table FFI wrappers
-- ---------------------------------------------------------------------------

/-- Hash-table FFI bundle (colocated: resolved within the same JIT module) -/
structure HtSetup where
  fnCreate : FnRef
  fnLookup : FnRef
  fnInsert : FnRef
  /-- `(ptr, ht, key_off, key_len, delta) -> new_count`, one call for the
      read-modify-write a word counter would otherwise spell out. -/
  fnIncrement : FnRef
  /-- `(ptr, ht) -> entries` -/
  fnCount : FnRef
  /-- `(ptr, ht, index, out_off) -> found`: entry `index` in table order. -/
  fnGetEntry : FnRef
  /-- `(ptr)`: release every table this context allocated. -/
  fnCleanup : FnRef
  /-- `(ptr)`: allocate the table context. -/
  fnInit : FnRef
  deriving Inhabited, Lean.ToExpr

/-- Declare the hash table as colocated FFI (resolved within the JIT module). -/
def declareHtFFI : IRBuilder HtSetup := do
  let fnCreate ← declareColocatedFFI "ht_create" [.i64] (some .i32)
  let fnLookup ← declareColocatedFFI "ht_lookup" [.i64, .i64, .i32, .i64] (some .i32)
  let fnInsert ← declareColocatedFFI "ht_insert" [.i64, .i64, .i32, .i64, .i32] none
  let fnIncrement ← declareColocatedFFI "ht_increment" [.i64, .i64, .i32, .i64] (some .i64)
  let fnCount ← declareColocatedFFI "ht_count" [.i64] (some .i32)
  let fnGetEntry ← declareColocatedFFI "ht_get_entry" [.i64, .i32, .i64, .i64] (some .i32)
  let fnCleanup ← declareFFI "cl_ht_cleanup" [.i64] none
  let fnInit ← declareFFI "cl_ht_init" [.i64] none
  pure { fnCreate, fnLookup, fnInsert, fnIncrement, fnCount, fnGetEntry, fnCleanup, fnInit }

/-- The libm entry points the runtime re-exports. -/
structure MathSetup where
  fnSinf : FnRef
  fnCosf : FnRef
  fnPowf : FnRef
  deriving Inhabited, Lean.ToExpr

def declareMathFFI : IRBuilder MathSetup := do
  let fnSinf ← declareFFI "cl_sinf" [.f32] (some .f32)
  let fnCosf ← declareFFI "cl_cosf" [.f32] (some .f32)
  let fnPowf ← declareFFI "cl_powf" [.f32, .f32] (some .f32)
  pure { fnSinf, fnCosf, fnPowf }

/-- Host threads: spawn runs one of this program's own functions. -/
structure ThreadSetup where
  fnInit : FnRef
  /-- `(ptr, fn_idx, arg) -> handle` -/
  fnSpawn : FnRef
  /-- `(ptr, handle) -> status` -/
  fnJoin : FnRef
  fnCleanup : FnRef
  deriving Inhabited, Lean.ToExpr

def declareThreadFFI : IRBuilder ThreadSetup := do
  let fnInit ← declareFFI "cl_thread_init" [.i64] none
  let fnSpawn ← declareFFI "cl_thread_spawn" [.i64, .i64, .i64] (some .i64)
  let fnJoin ← declareFFI "cl_thread_join" [.i64, .i64] (some .i64)
  let fnCleanup ← declareFFI "cl_thread_cleanup" [.i64] none
  pure { fnInit, fnSpawn, fnJoin, fnCleanup }


/-- CUDA FFI function bundle -/
structure CudaSetup where
  fnInit : FnRef
  fnCreateBuffer : FnRef
  fnUpload : FnRef
  fnUploadOffset : FnRef   -- cl_cuda_upload_ptr_offset: (ctx, buf_id, buf_offset, src_ptr, size) → i32
  fnUploadAsync : FnRef
  fnUploadOffsetAsync : FnRef
  fnDownload : FnRef
  fnDownloadOffset : FnRef -- cl_cuda_download_ptr_offset: (ctx, buf_id, buf_offset, dst_ptr, size) → i32
  fnDownloadAsync : FnRef
  fnFreeBuffer : FnRef
  fnStreamCreate : FnRef
  fnStreamSync : FnRef
  fnStreamDestroy : FnRef
  fnEventCreate : FnRef
  fnEventRecord : FnRef
  fnStreamWaitEvent : FnRef
  fnEventElapsedMsBits : FnRef
  fnEventDestroy : FnRef
  fnGraphBeginCapture : FnRef
  fnGraphEndCapture : FnRef
  fnGraphUpload : FnRef
  fnGraphLaunch : FnRef
  fnGraphDestroy : FnRef
  fnPinnedAlloc : FnRef
  fnPinnedPtr : FnRef
  fnPinnedFree : FnRef
  fnLaunch : FnRef
  fnLaunchNamed : FnRef    -- cl_cuda_launch_named: adds name_ptr arg between kernel and n_bufs
  fnLaunchOnStream : FnRef
  fnLaunchNamedOnStream : FnRef
  fnSync : FnRef           -- cl_cuda_sync: (ctx) → i32
  fnCleanup : FnRef
  deriving Inhabited, Lean.ToExpr

/-- Declare all CUDA FFI functions. -/
def declareCudaFFI : IRBuilder CudaSetup := do
  let fnInit         ← declareFFI "cl_cuda_init"              [.i64]                               none
  let fnCreateBuffer ← declareFFI "cl_cuda_create_buffer"     [.i64, .i64]                         (some .i32)
  let fnUpload       ← declareFFI "cl_cuda_upload_ptr"        [.i64, .i32, .i64, .i64]             (some .i32)
  let fnUploadOffset ← declareFFI "cl_cuda_upload_ptr_offset" [.i64, .i32, .i64, .i64, .i64]      (some .i32)
  let fnUploadAsync  ← declareFFI "cl_cuda_upload_ptr_async"  [.i64, .i32, .i64, .i64, .i32]       (some .i32)
  let fnUploadOffsetAsync ← declareFFI "cl_cuda_upload_ptr_offset_async"
    [.i64, .i32, .i64, .i64, .i64, .i32] (some .i32)
  let fnDownload     ← declareFFI "cl_cuda_download_ptr"      [.i64, .i32, .i64, .i64]             (some .i32)
  let fnDownloadOffset ← declareFFI "cl_cuda_download_ptr_offset" [.i64, .i32, .i64, .i64, .i64]   (some .i32)
  let fnDownloadAsync ← declareFFI "cl_cuda_download_ptr_async" [.i64, .i32, .i64, .i64, .i32]     (some .i32)
  let fnFreeBuffer   ← declareFFI "cl_cuda_free_buffer"       [.i64, .i32]                         (some .i32)
  let fnStreamCreate ← declareFFI "cl_cuda_stream_create"     [.i64]                               (some .i32)
  let fnStreamSync   ← declareFFI "cl_cuda_stream_sync"       [.i64, .i32]                         (some .i32)
  let fnStreamDestroy ← declareFFI "cl_cuda_stream_destroy"   [.i64, .i32]                         (some .i32)
  let fnEventCreate  ← declareFFI "cl_cuda_event_create"      [.i64]                               (some .i32)
  let fnEventRecord  ← declareFFI "cl_cuda_event_record"      [.i64, .i32, .i32]                   (some .i32)
  let fnStreamWaitEvent ← declareFFI "cl_cuda_stream_wait_event" [.i64, .i32, .i32]                (some .i32)
  let fnEventElapsedMsBits ← declareFFI "cl_cuda_event_elapsed_ms_bits" [.i64, .i32, .i32]         (some .i32)
  let fnEventDestroy ← declareFFI "cl_cuda_event_destroy"     [.i64, .i32]                         (some .i32)
  let fnGraphBeginCapture ← declareFFI "cl_cuda_graph_begin_capture" [.i64, .i32]                  (some .i32)
  let fnGraphEndCapture ← declareFFI "cl_cuda_graph_end_capture" [.i64, .i32]                      (some .i32)
  let fnGraphUpload ← declareFFI "cl_cuda_graph_upload" [.i64, .i32, .i32]                         (some .i32)
  let fnGraphLaunch ← declareFFI "cl_cuda_graph_launch" [.i64, .i32, .i32]                         (some .i32)
  let fnGraphDestroy ← declareFFI "cl_cuda_graph_destroy" [.i64, .i32]                             (some .i32)
  let fnPinnedAlloc  ← declareFFI "cl_cuda_pinned_alloc"      [.i64, .i64]                         (some .i32)
  let fnPinnedPtr    ← declareFFI "cl_cuda_pinned_ptr"        [.i64, .i32]                         (some .i64)
  let fnPinnedFree   ← declareFFI "cl_cuda_pinned_free"       [.i64, .i32]                         (some .i32)
  let fnLaunch       ← declareFFI "cl_cuda_launch"
    [.i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  let fnLaunchNamed  ← declareFFI "cl_cuda_launch_named"
    [.i64, .i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  let fnLaunchOnStream ← declareFFI "cl_cuda_launch_on_stream"
    [.i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  let fnLaunchNamedOnStream ← declareFFI "cl_cuda_launch_named_on_stream"
    [.i64, .i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  let fnSync         ← declareFFI "cl_cuda_sync"              [.i64]                               (some .i32)
  let fnCleanup      ← declareFFI "cl_cuda_cleanup"           [.i64]                               none
  pure { fnInit, fnCreateBuffer, fnUpload, fnUploadOffset, fnUploadAsync, fnUploadOffsetAsync,
         fnDownload, fnDownloadOffset, fnDownloadAsync, fnFreeBuffer, fnStreamCreate, fnStreamSync, fnStreamDestroy,
         fnEventCreate, fnEventRecord, fnStreamWaitEvent, fnEventElapsedMsBits, fnEventDestroy,
         fnGraphBeginCapture, fnGraphEndCapture, fnGraphUpload, fnGraphLaunch, fnGraphDestroy,
         fnPinnedAlloc, fnPinnedPtr, fnPinnedFree, fnLaunch, fnLaunchNamed, fnLaunchOnStream,
         fnLaunchNamedOnStream, fnSync, fnCleanup }

/-- cuBLAS FFI function bundle -/
structure CuBlasSetup where
  fnSgemv : FnRef   -- (ctx, trans, m, n, alpha_bits, a_buf, x_buf, beta_bits, y_buf) → i32
  fnSgemvOnStream : FnRef
  /-- `(ctx, transa, transb, m, n, k, alpha_bits, a_buf, stride_a, b_buf,
      stride_b, beta_bits, c_buf, stride_c, batch, off_a, off_b, off_c,
      ld_a, ld_b, ld_c) → i32`. A zero `ld_*` asks for the default leading
      dimension; `off_*` are element offsets into the operands. -/
  fnSgemm : FnRef
  fnSgemmOnStream : FnRef
  /-- `(ctx, arr_buf, slot, src_buf, off) → i32`: store one buffer's device
      pointer into an array of pointers. -/
  fnPtrArray : FnRef
  /-- `(ctx, transa, transb, m, n, k, alpha_bits, a_arr, b_arr, beta_bits,
      c_arr, batch, stream) → i32`: a batch whose members are named by pointer
      rather than by stride, so they need not share an allocation. -/
  fnSgemmBatchedOnStream : FnRef
  deriving Inhabited, Lean.ToExpr

def declareCuBlasFFI : IRBuilder CuBlasSetup := do
  let fnSgemv ← declareFFI "cl_cublas_sgemv"
    [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  let fnSgemvOnStream ← declareFFI "cl_cublas_sgemv_on_stream"
    [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  -- The three trailing `.i64`s are element offsets into the A, B and C
  -- operands.  An offset moves the pointer and leaves the matrix the call
  -- contracts alone, so it lets one buffer hold several operands without
  -- touching what `Law.cublasIsMatvec` says a contraction computes.
  let fnSgemm ← declareFFI "cl_cublas_sgemm_strided_batched"
    [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32, .i32, .i64, .i32,
     .i64, .i64, .i64, .i32, .i32, .i32] (some .i32)
  let fnSgemmOnStream ← declareFFI "cl_cublas_sgemm_strided_batched_on_stream"
    [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32, .i32, .i64, .i32, .i32,
     .i64, .i64, .i64, .i32, .i32, .i32] (some .i32)
  let fnPtrArray ← declareFFI "cl_cublas_ptr_array"
    [.i64, .i32, .i32, .i32, .i64] (some .i32)
  let fnSgemmBatchedOnStream ← declareFFI "cl_cublas_sgemm_batched_on_stream"
    [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  pure { fnSgemv, fnSgemvOnStream, fnSgemm, fnSgemmOnStream, fnPtrArray,
         fnSgemmBatchedOnStream }


/-- The fn_idx of the main entry point that every application emits as `u0:1`
    (with `u0:0` reserved as a no-op stub). Use this in `Algorithm.fn_idx`. -/
def mainFnIdx : UInt32 := u32 1


end IR

end AlgorithmLib
