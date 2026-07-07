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

/-- Declare all 7 GPU FFI functions -/
def declareGpuFFI : IRBuilder GpuSetup := do
  let fnInit ← declareFFI "cl_gpu_init" [.i64] none
  let fnCreateBuffer ← declareFFI "cl_gpu_create_buffer" [.i64, .i64] (some .i32)
  let fnCreatePipeline ← declareFFI "cl_gpu_create_pipeline" [.i64, .i64, .i64, .i32] (some .i32)
  let fnUpload ← declareFFI "cl_gpu_upload" [.i64, .i32, .i64, .i64] (some .i32)
  let fnDownload ← declareFFI "cl_gpu_download" [.i64, .i32, .i64, .i64] (some .i32)
  let fnDispatch ← declareFFI "cl_gpu_dispatch" [.i64, .i32, .i32, .i32, .i32] (some .i32)
  let fnCleanup ← declareFFI "cl_gpu_cleanup" [.i64] none
  pure { fnInit, fnCreateBuffer, fnUpload, fnDownload, fnCreatePipeline, fnDispatch, fnCleanup }

def gpuCtxSlotPtr (ptr : Val) (slotOffset : Nat := ContextSlots.wgpu) : IRBuilder Val :=
  absAddr ptr slotOffset

def gpuCtxPtr (ptr : Val) (slotOffset : Nat := ContextSlots.wgpu) : IRBuilder Val := do
  let slotPtr ← gpuCtxSlotPtr ptr slotOffset
  load64 slotPtr

def gpuInit (gpu : GpuSetup) (ptr : Val) (slotOffset : Nat := ContextSlots.wgpu) : IRBuilder Unit := do
  let slotPtr ← gpuCtxSlotPtr ptr slotOffset
  callVoid gpu.fnInit [slotPtr]

def gpuCreateBuffer (gpu : GpuSetup) (ptr size : Val)
    (slotOffset : Nat := ContextSlots.wgpu) : IRBuilder Val := do
  let ctxPtr ← gpuCtxPtr ptr slotOffset
  call gpu.fnCreateBuffer [ctxPtr, size]

def gpuCreatePipeline (gpu : GpuSetup) (ptr shaderOff bindOff nBindings : Val)
    (slotOffset : Nat := ContextSlots.wgpu) : IRBuilder Val := do
  let ctxPtr ← gpuCtxPtr ptr slotOffset
  let shaderPtr ← iadd ptr shaderOff
  let bindPtr ← iadd ptr bindOff
  call gpu.fnCreatePipeline [ctxPtr, shaderPtr, bindPtr, nBindings]

def gpuUpload (gpu : GpuSetup) (ptr bufId srcOff size : Val)
    (slotOffset : Nat := ContextSlots.wgpu) : IRBuilder Val := do
  let ctxPtr ← gpuCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call gpu.fnUpload [ctxPtr, bufId, srcPtr, size]

def gpuDownload (gpu : GpuSetup) (ptr bufId dstOff size : Val)
    (slotOffset : Nat := ContextSlots.wgpu) : IRBuilder Val := do
  let ctxPtr ← gpuCtxPtr ptr slotOffset
  let dstPtr ← iadd ptr dstOff
  call gpu.fnDownload [ctxPtr, bufId, dstPtr, size]

def gpuDispatch (gpu : GpuSetup) (ptr pipelineId wgX wgY wgZ : Val)
    (slotOffset : Nat := ContextSlots.wgpu) : IRBuilder Val := do
  let ctxPtr ← gpuCtxPtr ptr slotOffset
  call gpu.fnDispatch [ctxPtr, pipelineId, wgX, wgY, wgZ]

def gpuCleanup (gpu : GpuSetup) (ptr : Val) (slotOffset : Nat := ContextSlots.wgpu) : IRBuilder Unit := do
  let slotPtr ← gpuCtxSlotPtr ptr slotOffset
  callVoid gpu.fnCleanup [slotPtr]

/-- Read a file into shared memory. Returns bytes read.
    readFile ptr fnRead filenameOff dataOff => call fnRead(ptr, filenameOff, dataOff, 0, 0) -/
def readFile (ptr : Val) (fnRead : FnRef) (filenameOff dataOff : Nat) : IRBuilder Val := do
  let fnOff ← iconst64 filenameOff
  let dOff ← iconst64 dataOff
  let zero ← iconst64 0
  call fnRead [ptr, fnOff, dOff, zero, zero]

/-- Write a region of shared memory to a file. Returns bytes written.
    writeFile ptr fnWrite filenameOff srcOff fileOffset size -/
def writeFile (ptr : Val) (fnWrite : FnRef) (filenameOff srcOff : Nat)
    (fileOffset size : Val) : IRBuilder Val := do
  let fnOff ← iconst64 filenameOff
  let sOff ← iconst64 srcOff
  call fnWrite [ptr, fnOff, sOff, fileOffset, size]

/-- Write a region of shared memory to a file starting at file offset 0. -/
def writeFile0 (ptr : Val) (fnWrite : FnRef) (filenameOff srcOff : Nat)
    (size : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  writeFile ptr fnWrite filenameOff srcOff zero size

/-- Align a value up to the next multiple of 4 (wgpu COPY_BUFFER_ALIGNMENT).
    alignUp4 x = (x + 3) & ~3 -/
def alignUp4 (v : Val) : IRBuilder Val := do
  let c3 ← iconst64 3
  let sum ← iadd v c3
  let negFour ← iconst64 (-4)
  band sum negFour

/-- Window / input / present FFI bundle. The window shares the wgpu device, so
    `fnPresentGpuBuffer` blits a game framebuffer straight from a wgpu storage
    buffer to the swapchain with no host round trip. -/
structure WindowSetup where
  fnInit : FnRef
  fnOpen : FnRef
  fnPoll : FnRef
  fnPresentGpuBuffer : FnRef
  fnCleanup : FnRef

/-- Declare the minimal window FFI (init/open/poll/present/cleanup). -/
def declareWindowFFI : IRBuilder WindowSetup := do
  let fnInit ← declareFFI "cl_window_init" [.i64] none
  let fnOpen ← declareFFI "cl_window_open" [.i64, .i64, .i64, .i64, .i64, .i64, .i64] (some .i32)
  let fnPoll ← declareFFI "cl_window_poll" [.i64, .i64, .i32] (some .i32)
  let fnPresentGpuBuffer ← declareFFI "cl_window_present_gpu_buffer"
    [.i64, .i64, .i32] (some .i32)
  let fnCleanup ← declareFFI "cl_window_cleanup" [.i64] none
  pure { fnInit, fnOpen, fnPoll, fnPresentGpuBuffer, fnCleanup }

def windowCtxSlotPtr (ptr : Val) (slotOffset : Nat := ContextSlots.window) : IRBuilder Val :=
  absAddr ptr slotOffset

def windowCtxPtr (ptr : Val) (slotOffset : Nat := ContextSlots.window) : IRBuilder Val := do
  let slotPtr ← windowCtxSlotPtr ptr slotOffset
  load64 slotPtr

def windowInit (win : WindowSetup) (ptr : Val) (slotOffset : Nat := ContextSlots.window) : IRBuilder Unit := do
  let slotPtr ← windowCtxSlotPtr ptr slotOffset
  callVoid win.fnInit [slotPtr]

/-- Open the (single) window and build the blit pipeline from the WGSL at
    `blitOff` (a byte offset into shared memory, `blitLen` bytes). The blit
    shader must expose one read-only storage buffer at `@group(0) @binding(0)`
    (the packed-RGBA framebuffer) with entry points `vs_main`/`fs_main`, and bake
    in the framebuffer dimensions. `titleOff`/`titleLen` name the window title.
    Returns 0 on success, -1 on error. -/
def windowOpen (win : WindowSetup) (ptr width height titleOff titleLen blitOff blitLen : Val)
    (slotOffset : Nat := ContextSlots.window) : IRBuilder Val := do
  let ctxPtr ← windowCtxPtr ptr slotOffset
  let titlePtr ← iadd ptr titleOff
  let blitPtr ← iadd ptr blitOff
  call win.fnOpen [ctxPtr, width, height, titlePtr, titleLen, blitPtr, blitLen]

/-- Drain up to `maxEvents` input events into the buffer at `eventsOff` (32 bytes
    each: kind, a, b, c as i64). Returns the number of events written, or -1. -/
def windowPoll (win : WindowSetup) (ptr eventsOff maxEvents : Val)
    (slotOffset : Nat := ContextSlots.window) : IRBuilder Val := do
  let ctxPtr ← windowCtxPtr ptr slotOffset
  let eventsPtr ← iadd ptr eventsOff
  call win.fnPoll [ctxPtr, eventsPtr, maxEvents]

/-- Present a game framebuffer (wgpu storage buffer `bufId` of packed RGBA u32)
    to the window, zero-copy. The framebuffer size is baked into the blit shader,
    so no dimensions are passed. Reads the gpu context pointer from its slot. -/
def windowPresentGpuBuffer (win : WindowSetup) (ptr bufId : Val)
    (slotOffset : Nat := ContextSlots.window) (gpuSlotOffset : Nat := ContextSlots.wgpu)
    : IRBuilder Val := do
  let ctxPtr ← windowCtxPtr ptr slotOffset
  let gpuCtxPtr ← gpuCtxPtr ptr gpuSlotOffset
  call win.fnPresentGpuBuffer [ctxPtr, gpuCtxPtr, bufId]

def windowCleanup (win : WindowSetup) (ptr : Val) (slotOffset : Nat := ContextSlots.window) : IRBuilder Unit := do
  let slotPtr ← windowCtxSlotPtr ptr slotOffset
  callVoid win.fnCleanup [slotPtr]

/-- LMDB FFI function bundle -/
structure LmdbSetup where
  fnInit : FnRef
  fnOpen : FnRef
  fnBeginWriteTxn : FnRef
  fnPut : FnRef
  fnCommitWriteTxn : FnRef
  fnCursorScan : FnRef
  fnCleanup : FnRef

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

/-- Declare ht_create / ht_lookup / ht_insert as colocated FFI functions. -/
def declareHtFFI : IRBuilder HtSetup := do
  let fnCreate ← declareColocatedFFI "ht_create" [.i64] (some .i32)
  let fnLookup ← declareColocatedFFI "ht_lookup" [.i64, .i64, .i32, .i64] (some .i32)
  let fnInsert ← declareColocatedFFI "ht_insert" [.i64, .i64, .i32, .i64, .i32] none
  pure { fnCreate, fnLookup, fnInsert }

/-- Initialise the HT context; pass `ptr` (shared-memory base) — context ptr written to ptr[0]. -/
def htInit (ptr : Val) : IRBuilder Unit := do
  let fnInit ← declareFFI "cl_ht_init" [.i64] none
  callVoid fnInit [ptr]

-- ---------------------------------------------------------------------------
-- Layout-aware IR combinators (typed field handles)
-- ---------------------------------------------------------------------------

/-- Get a typed field's offset as an iconst64 value -/
def fldOffset (f : Layout.Fld t) : IRBuilder Val :=
  iconst64 f.offset

/-- Get a typed field's absolute address: base + offset -/
def fldAddr (base : Val) (f : Layout.Fld t) : IRBuilder Val :=
  absAddr base f.offset

/-- Proof witness that a FieldTy is a scalar (u8, i32, i64) — not a byte region.
    Carries store/load implementations so the match is done per-instance, not generically. -/
class Layout.IsScalar (t : Layout.FieldTy) where
  scalarStore : Val → Val → IRBuilder Unit
  scalarLoad  : Val → IRBuilder Val

instance : Layout.IsScalar .u8 where
  scalarStore val addr := istore8 val addr
  scalarLoad  addr     := uload8_64 addr

instance : Layout.IsScalar .i32 where
  scalarStore val addr := store val addr
  scalarLoad  addr     := uload32_64 addr

instance : Layout.IsScalar .i64 where
  scalarStore val addr := store val addr
  scalarLoad  addr     := load64 addr

/-- Store a value to a scalar field. Rejects `.bytes n` at the type level. -/
def fldStore (base : Val) (f : Layout.Fld t) [inst : Layout.IsScalar t] (val : Val) : IRBuilder Unit := do
  let addr ← fldAddr base f
  inst.scalarStore val addr

/-- Load a value from a scalar field. Rejects `.bytes n` at the type level. -/
def fldLoad (base : Val) (f : Layout.Fld t) [inst : Layout.IsScalar t] : IRBuilder Val := do
  let addr ← fldAddr base f
  inst.scalarLoad addr

/-- Store an i64 value at byte offset `i` within a `.bytes n` field.
    Requires a proof that `i + 8 ≤ n` (the 8-byte store fits). -/
def fldStoreAt (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat) (val : Val)
    (_h : i + 8 ≤ n := by omega) : IRBuilder Unit := do
  let addr ← absAddr base (f.offset + i)
  store val addr

/-- Store a u8 value at byte offset `i` within a `.bytes n` field.
    Requires a proof that `i < n` (the 1-byte store fits). -/
def fldStore8At (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat) (val : Val)
    (_h : i + 1 ≤ n := by omega) : IRBuilder Unit := do
  let addr ← absAddr base (f.offset + i)
  istore8 val addr

/-- Store an i32 value at byte offset `i` within a `.bytes n` field.
    Requires a proof that `i + 4 ≤ n` (the 4-byte store fits). -/
def fldStore32At (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat) (val : Val)
    (_h : i + 4 ≤ n := by omega) : IRBuilder Unit := do
  let addr ← absAddr base (f.offset + i)
  store val addr

/-- Load an i64 value from byte offset `i` within a `.bytes n` field.
    Requires a proof that `i + 8 ≤ n`. -/
def fldLoadAt (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 8 ≤ n := by omega) : IRBuilder Val := do
  let addr ← absAddr base (f.offset + i)
  load64 addr

/-- Load a u8 value (zero-extended to i64) from byte offset `i` within a `.bytes n` field.
    Requires a proof that `i < n`. -/
def fldLoad8At (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 1 ≤ n := by omega) : IRBuilder Val := do
  let addr ← absAddr base (f.offset + i)
  uload8_64 addr

/-- Load an i32 value (zero-extended to i64) from byte offset `i` within a `.bytes n` field.
    Requires a proof that `i + 4 ≤ n`. -/
def fldLoad32At (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 4 ≤ n := by omega) : IRBuilder Val := do
  let addr ← absAddr base (f.offset + i)
  uload32_64 addr

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
  fnSgemm : FnRef   -- (ctx, transa, transb, m, n, k, alpha_bits, a_buf, stride_a, b_buf, stride_b, beta_bits, c_buf, stride_c, batch) → i32
  fnSgemmOnStream : FnRef

def declareCuBlasFFI : IRBuilder CuBlasSetup := do
  let fnSgemv ← declareFFI "cl_cublas_sgemv"
    [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  let fnSgemvOnStream ← declareFFI "cl_cublas_sgemv_on_stream"
    [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  let fnSgemm ← declareFFI "cl_cublas_sgemm_strided_batched"
    [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32, .i32, .i64, .i32] (some .i32)
  let fnSgemmOnStream ← declareFFI "cl_cublas_sgemm_strided_batched_on_stream"
    [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32, .i32, .i64, .i32, .i32] (some .i32)
  pure { fnSgemv, fnSgemvOnStream, fnSgemm, fnSgemmOnStream }

def cudaCtxSlotPtr (ptr : Val) (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val :=
  absAddr ptr slotOffset

def cudaCtxPtr (ptr : Val) (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let slotPtr ← cudaCtxSlotPtr ptr slotOffset
  load64 slotPtr

def cudaInit (cuda : CudaSetup) (ptr : Val) (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Unit := do
  let slotPtr ← cudaCtxSlotPtr ptr slotOffset
  callVoid cuda.fnInit [slotPtr]

def cudaCreateBuffer (cuda : CudaSetup) (ptr size : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnCreateBuffer [ctxPtr, size]

def cudaUpload (cuda : CudaSetup) (ptr bufId srcOff size : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call cuda.fnUpload [ctxPtr, bufId, srcPtr, size]

/-- Upload to a specific offset within a GPU buffer. -/
def cudaUploadOffset (cuda : CudaSetup) (ptr bufId bufOff srcOff size : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call cuda.fnUploadOffset [ctxPtr, bufId, bufOff, srcPtr, size]

def cudaUploadAsync (cuda : CudaSetup) (ptr bufId srcOff size streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call cuda.fnUploadAsync [ctxPtr, bufId, srcPtr, size, streamId]

def cudaUploadOffsetAsync (cuda : CudaSetup) (ptr bufId bufOff srcOff size streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call cuda.fnUploadOffsetAsync [ctxPtr, bufId, bufOff, srcPtr, size, streamId]

def cudaDownload (cuda : CudaSetup) (ptr bufId dstOff size : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  let dstPtr ← iadd ptr dstOff
  call cuda.fnDownload [ctxPtr, bufId, dstPtr, size]

def cudaDownloadAsync (cuda : CudaSetup) (ptr bufId dstOff size streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  let dstPtr ← iadd ptr dstOff
  call cuda.fnDownloadAsync [ctxPtr, bufId, dstPtr, size, streamId]

def cudaSync (cuda : CudaSetup) (ptr : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnSync [ctxPtr]

def cudaFreeBuffer (cuda : CudaSetup) (ptr bufId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnFreeBuffer [ctxPtr, bufId]

def cudaStreamCreate (cuda : CudaSetup) (ptr : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnStreamCreate [ctxPtr]

def cudaStreamSync (cuda : CudaSetup) (ptr streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnStreamSync [ctxPtr, streamId]

def cudaStreamDestroy (cuda : CudaSetup) (ptr streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnStreamDestroy [ctxPtr, streamId]

def cudaEventCreate (cuda : CudaSetup) (ptr : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnEventCreate [ctxPtr]

def cudaEventRecord (cuda : CudaSetup) (ptr eventId streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnEventRecord [ctxPtr, eventId, streamId]

def cudaStreamWaitEvent (cuda : CudaSetup) (ptr streamId eventId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnStreamWaitEvent [ctxPtr, streamId, eventId]

def cudaEventElapsedMsBits (cuda : CudaSetup) (ptr startEventId endEventId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnEventElapsedMsBits [ctxPtr, startEventId, endEventId]

def cudaEventDestroy (cuda : CudaSetup) (ptr eventId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnEventDestroy [ctxPtr, eventId]

def cudaGraphBeginCapture (cuda : CudaSetup) (ptr streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnGraphBeginCapture [ctxPtr, streamId]

def cudaGraphEndCapture (cuda : CudaSetup) (ptr streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnGraphEndCapture [ctxPtr, streamId]

def cudaGraphUpload (cuda : CudaSetup) (ptr graphId streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnGraphUpload [ctxPtr, graphId, streamId]

def cudaGraphLaunch (cuda : CudaSetup) (ptr graphId streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnGraphLaunch [ctxPtr, graphId, streamId]

def cudaGraphDestroy (cuda : CudaSetup) (ptr graphId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnGraphDestroy [ctxPtr, graphId]

def cudaPinnedAlloc (cuda : CudaSetup) (ptr size : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnPinnedAlloc [ctxPtr, size]

def cudaPinnedPtr (cuda : CudaSetup) (ptr pinnedId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnPinnedPtr [ctxPtr, pinnedId]

def cudaPinnedFree (cuda : CudaSetup) (ptr pinnedId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cuda.fnPinnedFree [ctxPtr, pinnedId]

def cudaLaunch (cuda : CudaSetup) (ptr kernelOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let bindPtr ← iadd ptr bindOff
  call cuda.fnLaunch [ctxPtr, kernelPtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ]

def cudaLaunchNamed (cuda : CudaSetup) (ptr kernelOff nameOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let namePtr ← iadd ptr nameOff
  let bindPtr ← iadd ptr bindOff
  call cuda.fnLaunchNamed [ctxPtr, kernelPtr, namePtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ]

def cudaLaunchOnStream (cuda : CudaSetup) (ptr kernelOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let bindPtr ← iadd ptr bindOff
  call cuda.fnLaunchOnStream [ctxPtr, kernelPtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ, streamId]

def cudaLaunchNamedOnStream (cuda : CudaSetup) (ptr kernelOff nameOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let namePtr ← iadd ptr nameOff
  let bindPtr ← iadd ptr bindOff
  call cuda.fnLaunchNamedOnStream [ctxPtr, kernelPtr, namePtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ, streamId]

def cudaCleanup (cuda : CudaSetup) (ptr : Val) (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Unit := do
  let slotPtr ← cudaCtxSlotPtr ptr slotOffset
  callVoid cuda.fnCleanup [slotPtr]

def cublasSgemv (cublas : CuBlasSetup) (ptr trans m n alphaBits aBuf xBuf betaBits yBuf : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cublas.fnSgemv [ctxPtr, trans, m, n, alphaBits, aBuf, xBuf, betaBits, yBuf]

def cublasSgemvOnStream (cublas : CuBlasSetup)
    (ptr trans m n alphaBits aBuf xBuf betaBits yBuf streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cublas.fnSgemvOnStream [ctxPtr, trans, m, n, alphaBits, aBuf, xBuf, betaBits, yBuf, streamId]

def cublasSgemmStridedBatched (cublas : CuBlasSetup)
    (ptr transA transB m n k alphaBits aBuf strideA bBuf strideB betaBits cBuf strideC batchCount : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cublas.fnSgemm
    [ctxPtr, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB, betaBits, cBuf, strideC, batchCount]

def cublasSgemmStridedBatchedOnStream (cublas : CuBlasSetup)
    (ptr transA transB m n k alphaBits aBuf strideA bBuf strideB betaBits cBuf strideC batchCount streamId : Val)
    (slotOffset : Nat := ContextSlots.cuda) : IRBuilder Val := do
  let ctxPtr ← cudaCtxPtr ptr slotOffset
  call cublas.fnSgemmOnStream
    [ctxPtr, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB, betaBits, cBuf, strideC, batchCount, streamId]

/-- Read a file using typed field handles for filename and data regions -/
def fldReadFile (ptr : Val) (fnRead : FnRef)
    (filenameFld : Layout.Fld ft) (dataFld : Layout.Fld dt) : IRBuilder Val :=
  readFile ptr fnRead filenameFld.offset dataFld.offset

/-- Write a file using typed field handles for filename and source regions -/
def fldWriteFile0 (ptr : Val) (fnWrite : FnRef)
    (filenameFld : Layout.Fld ft) (srcFld : Layout.Fld st) (size : Val) : IRBuilder Val :=
  writeFile0 ptr fnWrite filenameFld.offset srcFld.offset size

/-- The fn_idx of the main entry point that every application emits as `u0:1`
    (with `u0:0` reserved as a no-op stub). Use this in `Algorithm.fn_idx`. -/
def mainFnIdx : UInt32 := u32 1


end IR

end AlgorithmLib
