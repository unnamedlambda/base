import AlgorithmLib.HProg

/-!
# One named wrapper per runtime entry point

`FFI.lean` says who the runtime exports and, in `sig`, exactly what each one
takes and returns. `ffi` and `ffiVoid` can call any of them today, but only as a
*positional list*: a call with the wrong number of arguments is a well-typed
Lean term, and what catches it is `wf` at compile time or the runtime at run
time, neither of which points at the call site.

Every entry point here has a wrapper whose arity is its signature's, so that
mistake is a type error where it is made. One `def` per constructor of
`IR.Ffi`, no exceptions -- `FFIRawScan` fails the build if this file and
`Ffi.sig` ever disagree, or if an appended entry point has no wrapper.

Parameter names are the ones `base/src/ffi/` gives them, so a reader sees what
the runtime calls its arguments rather than a position. What they *mean* --
which is a context pointer read from its slot, which an offset rather than an
address -- is a convention no signature can express, and lives in `HProgFFI`
with the wrappers built for each subsystem. Those are what a generator should
normally reach for; this is the total, mechanical layer beneath them.

Qualify these rather than opening the namespace: several share a name with the
convention wrapper in `HProgFFI` that supersedes them.
-/

namespace AlgorithmLib.HProg.Sur.Raw

open AlgorithmLib.IR

/-- `cl_file_read(ptr : i64, pathOff : i64, dstOff : i64, fileOffset : i64,
    size : i64) -> i64` -/
def fileRead
    (ptr pathOff dstOff fileOffset size : R)
    : M R :=
  ffi .fileRead
    [ptr, pathOff, dstOff, fileOffset, size]

/-- `cl_file_write(ptr : i64, pathOff : i64, srcOff : i64, fileOffset : i64,
    size : i64) -> i64` -/
def fileWrite
    (ptr pathOff srcOff fileOffset size : R)
    : M R :=
  ffi .fileWrite
    [ptr, pathOff, srcOff, fileOffset, size]

/-- `cl_file_read_to_ptr(pathPtr : i64, dstPtr : i64, fileOffset : i64,
    size : i64) -> i64` -/
def fileReadToPtr
    (pathPtr dstPtr fileOffset size : R)
    : M R :=
  ffi .fileReadToPtr
    [pathPtr, dstPtr, fileOffset, size]

/-- `cl_file_write_from_ptr(pathPtr : i64, srcPtr : i64, fileOffset : i64,
    size : i64) -> i64` -/
def fileWriteFromPtr
    (pathPtr srcPtr fileOffset size : R)
    : M R :=
  ffi .fileWriteFromPtr
    [pathPtr, srcPtr, fileOffset, size]

/-- `cl_stdin_readline(ptr : i64, dstOff : i64, maxLen : i64) -> i64` -/
def stdinReadline
    (ptr dstOff maxLen : R)
    : M R :=
  ffi .stdinReadline
    [ptr, dstOff, maxLen]

/-- `cl_stdout_write(ptr : i64, srcOff : i64, size : i64) -> i64` -/
def stdoutWrite
    (ptr srcOff size : R)
    : M R :=
  ffi .stdoutWrite
    [ptr, srcOff, size]

/-- `cl_gpu_init(ctxSlotPtr : i64) -> ()` -/
def gpuInit (ctxSlotPtr : R) : M Unit := ffiVoid .gpuInit [ctxSlotPtr]

/-- `cl_gpu_create_buffer(ctxPtr : i64, size : i64) -> i32` -/
def gpuCreateBuffer
    (ctxPtr size : R)
    : M R :=
  ffi .gpuCreateBuffer
    [ctxPtr, size]

/-- `cl_gpu_create_pipeline(ctxPtr : i64, shaderPtr : i64, bindPtr : i64,
    nBindings : i32) -> i32` -/
def gpuCreatePipeline
    (ctxPtr shaderPtr bindPtr nBindings : R)
    : M R :=
  ffi .gpuCreatePipeline
    [ctxPtr, shaderPtr, bindPtr, nBindings]

/-- `cl_gpu_upload(ctxPtr : i64, bufId : i32, srcPtr : i64,
    size : i64) -> i32` -/
def gpuUpload
    (ctxPtr bufId srcPtr size : R)
    : M R :=
  ffi .gpuUpload
    [ctxPtr, bufId, srcPtr, size]

/-- `cl_gpu_download(ctxPtr : i64, bufId : i32, dstPtr : i64,
    size : i64) -> i32` -/
def gpuDownload
    (ctxPtr bufId dstPtr size : R)
    : M R :=
  ffi .gpuDownload
    [ctxPtr, bufId, dstPtr, size]

/-- `cl_gpu_dispatch(ctxPtr : i64, pipelineId : i32, wgX : i32, wgY : i32,
    wgZ : i32) -> i32` -/
def gpuDispatch
    (ctxPtr pipelineId wgX wgY wgZ : R)
    : M R :=
  ffi .gpuDispatch
    [ctxPtr, pipelineId, wgX, wgY, wgZ]

/-- `cl_gpu_cleanup(ctxSlotPtr : i64) -> ()` -/
def gpuCleanup (ctxSlotPtr : R) : M Unit := ffiVoid .gpuCleanup [ctxSlotPtr]

/-- `cl_gpu_upload_ptr(ctxPtr : i64, bufId : i32, srcPtr : i64,
    size : i64) -> i32` -/
def gpuUploadPtr
    (ctxPtr bufId srcPtr size : R)
    : M R :=
  ffi .gpuUploadPtr
    [ctxPtr, bufId, srcPtr, size]

/-- `cl_gpu_download_ptr(ctxPtr : i64, bufId : i32, bufOffset : i64,
    dstPtr : i64, size : i64) -> i32` -/
def gpuDownloadPtr
    (ctxPtr bufId bufOffset dstPtr size : R)
    : M R :=
  ffi .gpuDownloadPtr
    [ctxPtr, bufId, bufOffset, dstPtr, size]

/-- `cl_window_init(ctxSlotPtr : i64) -> ()` -/
def windowInit (ctxSlotPtr : R) : M Unit := ffiVoid .windowInit [ctxSlotPtr]

/-- `cl_window_open(ctxPtr : i64, width : i64, height : i64, titlePtr : i64,
    titleLen : i64, blitPtr : i64, blitLen : i64) -> i32` -/
def windowOpen
    (ctxPtr width height titlePtr titleLen blitPtr blitLen : R)
    : M R :=
  ffi .windowOpen
    [ctxPtr, width, height, titlePtr, titleLen, blitPtr, blitLen]

/-- `cl_window_poll(ctxPtr : i64, eventsPtr : i64, maxEvents : i32) -> i32` -/
def windowPoll
    (ctxPtr eventsPtr maxEvents : R)
    : M R :=
  ffi .windowPoll
    [ctxPtr, eventsPtr, maxEvents]

/-- `cl_window_present_gpu_buffer(ctxPtr : i64, gpuCtxPtr : i64,
    bufId : i32) -> i32` -/
def windowPresentGpuBuffer
    (ctxPtr gpuCtxPtr bufId : R)
    : M R :=
  ffi .windowPresentGpuBuffer
    [ctxPtr, gpuCtxPtr, bufId]

/-- `cl_window_cleanup(ctxSlotPtr : i64) -> ()` -/
def windowCleanup
    (ctxSlotPtr : R)
    : M Unit :=
  ffiVoid .windowCleanup
    [ctxSlotPtr]

/-- `cl_lmdb_init(ctxSlotPtr : i64) -> ()` -/
def lmdbInit (ctxSlotPtr : R) : M Unit := ffiVoid .lmdbInit [ctxSlotPtr]

/-- `cl_lmdb_open(ctxPtr : i64, pathPtr : i64, mapSizeMb : i32) -> i32` -/
def lmdbOpen
    (ctxPtr pathPtr mapSizeMb : R)
    : M R :=
  ffi .lmdbOpen
    [ctxPtr, pathPtr, mapSizeMb]

/-- `cl_lmdb_begin_write_txn(ctxPtr : i64, handle : i32) -> i32` -/
def lmdbBeginWriteTxn
    (ctxPtr handle : R)
    : M R :=
  ffi .lmdbBeginWriteTxn
    [ctxPtr, handle]

/-- `cl_lmdb_put(ctxPtr : i64, handle : i32, keyPtr : i64, keyLen : i32,
    valPtr : i64, valLen : i32) -> i32` -/
def lmdbPut
    (ctxPtr handle keyPtr keyLen valPtr valLen : R)
    : M R :=
  ffi .lmdbPut
    [ctxPtr, handle, keyPtr, keyLen, valPtr, valLen]

/-- `cl_lmdb_commit_write_txn(ctxPtr : i64, handle : i32) -> i32` -/
def lmdbCommitWriteTxn
    (ctxPtr handle : R)
    : M R :=
  ffi .lmdbCommitWriteTxn
    [ctxPtr, handle]

/-- `cl_lmdb_cursor_scan(ctxPtr : i64, handle : i32, keyPtr : i64,
    keyLen : i32, maxEntries : i32, resultPtr : i64) -> i32` -/
def lmdbCursorScan
    (ctxPtr handle keyPtr keyLen maxEntries resultPtr : R)
    : M R :=
  ffi .lmdbCursorScan
    [ctxPtr, handle, keyPtr, keyLen, maxEntries, resultPtr]

/-- `cl_lmdb_cleanup(ctxSlotPtr : i64) -> ()` -/
def lmdbCleanup (ctxSlotPtr : R) : M Unit := ffiVoid .lmdbCleanup [ctxSlotPtr]

/-- `ht_create(ctx : i64) -> i32` -/
def htCreate (ctx : R) : M R := ffi .htCreate [ctx]

/-- `ht_lookup(ctx : i64, key : i64, keyLen : i32, result : i64) -> i32` -/
def htLookup
    (ctx key keyLen result : R)
    : M R :=
  ffi .htLookup
    [ctx, key, keyLen, result]

/-- `ht_insert(ctx : i64, key : i64, keyLen : i32, val : i64,
    valLen : i32) -> ()` -/
def htInsert
    (ctx key keyLen val valLen : R)
    : M Unit :=
  ffiVoid .htInsert
    [ctx, key, keyLen, val, valLen]

/-- `ht_increment(ctx : i64, key : i64, keyLen : i32, addend : i64) -> i64` -/
def htIncrement
    (ctx key keyLen addend : R)
    : M R :=
  ffi .htIncrement
    [ctx, key, keyLen, addend]

/-- `ht_count(ctx : i64) -> i32` -/
def htCount (ctx : R) : M R := ffi .htCount [ctx]

/-- `ht_get_entry(ctx : i64, index : i32, keyOut : i64, valOut : i64) -> i32` -/
def htGetEntry
    (ctx index keyOut valOut : R)
    : M R :=
  ffi .htGetEntry
    [ctx, index, keyOut, valOut]

/-- `cl_ht_cleanup(ctxSlotPtr : i64) -> ()` -/
def htCleanup (ctxSlotPtr : R) : M Unit := ffiVoid .htCleanup [ctxSlotPtr]

/-- `cl_ht_init(ctxSlotPtr : i64) -> ()` -/
def htInit (ctxSlotPtr : R) : M Unit := ffiVoid .htInit [ctxSlotPtr]

/-- `cl_sinf(x : f32) -> f32` -/
def sinf (x : R) : M R := ffi .sinf [x]

/-- `cl_cosf(x : f32) -> f32` -/
def cosf (x : R) : M R := ffi .cosf [x]

/-- `cl_powf(base : f32, exp : f32) -> f32` -/
def powf (base exp : R) : M R := ffi .powf [base, exp]

/-- `cl_thread_init(ctxSlotPtr : i64) -> ()` -/
def threadInit (ctxSlotPtr : R) : M Unit := ffiVoid .threadInit [ctxSlotPtr]

/-- `cl_thread_spawn(ctxPtr : i64, fnIndex : i64, threadPtr : i64) -> i64` -/
def threadSpawn
    (ctxPtr fnIndex threadPtr : R)
    : M R :=
  ffi .threadSpawn
    [ctxPtr, fnIndex, threadPtr]

/-- `cl_thread_join(ctxPtr : i64, handle : i64) -> i64` -/
def threadJoin (ctxPtr handle : R) : M R := ffi .threadJoin [ctxPtr, handle]

/-- `cl_thread_cleanup(ctxSlotPtr : i64) -> ()` -/
def threadCleanup
    (ctxSlotPtr : R)
    : M Unit :=
  ffiVoid .threadCleanup
    [ctxSlotPtr]

/-- `cl_cuda_init(ctxSlotPtr : i64) -> ()` -/
def cudaInit (ctxSlotPtr : R) : M Unit := ffiVoid .cudaInit [ctxSlotPtr]

/-- `cl_cuda_create_buffer(ctxPtr : i64, size : i64) -> i32` -/
def cudaCreateBuffer
    (ctxPtr size : R)
    : M R :=
  ffi .cudaCreateBuffer
    [ctxPtr, size]

/-- `cl_cuda_upload_ptr(ctxPtr : i64, bufId : i32, srcPtr : i64,
    size : i64) -> i32` -/
def cudaUpload
    (ctxPtr bufId srcPtr size : R)
    : M R :=
  ffi .cudaUpload
    [ctxPtr, bufId, srcPtr, size]

/-- `cl_cuda_upload_ptr_offset(ctxPtr : i64, bufId : i32, bufOffset : i64,
    srcPtr : i64, size : i64) -> i32` -/
def cudaUploadOffset
    (ctxPtr bufId bufOffset srcPtr size : R)
    : M R :=
  ffi .cudaUploadOffset
    [ctxPtr, bufId, bufOffset, srcPtr, size]

/-- `cl_cuda_upload_ptr_async(ctxPtr : i64, bufId : i32, srcPtr : i64,
    size : i64, streamId : i32) -> i32` -/
def cudaUploadAsync
    (ctxPtr bufId srcPtr size streamId : R)
    : M R :=
  ffi .cudaUploadAsync
    [ctxPtr, bufId, srcPtr, size, streamId]

/-- `cl_cuda_upload_ptr_offset_async(ctxPtr : i64, bufId : i32,
    bufOffset : i64, srcPtr : i64, size : i64, streamId : i32) -> i32` -/
def cudaUploadOffsetAsync
    (ctxPtr bufId bufOffset srcPtr size streamId : R)
    : M R :=
  ffi .cudaUploadOffsetAsync
    [ctxPtr, bufId, bufOffset, srcPtr, size, streamId]

/-- `cl_cuda_download_ptr(ctxPtr : i64, bufId : i32, dstPtr : i64,
    size : i64) -> i32` -/
def cudaDownload
    (ctxPtr bufId dstPtr size : R)
    : M R :=
  ffi .cudaDownload
    [ctxPtr, bufId, dstPtr, size]

/-- `cl_cuda_download_ptr_offset(ctxPtr : i64, bufId : i32, bufOffset : i64,
    dstPtr : i64, size : i64) -> i32` -/
def cudaDownloadOffset
    (ctxPtr bufId bufOffset dstPtr size : R)
    : M R :=
  ffi .cudaDownloadOffset
    [ctxPtr, bufId, bufOffset, dstPtr, size]

/-- `cl_cuda_download_ptr_async(ctxPtr : i64, bufId : i32, dstPtr : i64,
    size : i64, streamId : i32) -> i32` -/
def cudaDownloadAsync
    (ctxPtr bufId dstPtr size streamId : R)
    : M R :=
  ffi .cudaDownloadAsync
    [ctxPtr, bufId, dstPtr, size, streamId]

/-- `cl_cuda_free_buffer(ctxPtr : i64, bufId : i32) -> i32` -/
def cudaFreeBuffer
    (ctxPtr bufId : R)
    : M R :=
  ffi .cudaFreeBuffer
    [ctxPtr, bufId]

/-- `cl_cuda_stream_create(ctxPtr : i64) -> i32` -/
def cudaStreamCreate (ctxPtr : R) : M R := ffi .cudaStreamCreate [ctxPtr]

/-- `cl_cuda_stream_sync(ctxPtr : i64, streamId : i32) -> i32` -/
def cudaStreamSync
    (ctxPtr streamId : R)
    : M R :=
  ffi .cudaStreamSync
    [ctxPtr, streamId]

/-- `cl_cuda_stream_destroy(ctxPtr : i64, streamId : i32) -> i32` -/
def cudaStreamDestroy
    (ctxPtr streamId : R)
    : M R :=
  ffi .cudaStreamDestroy
    [ctxPtr, streamId]

/-- `cl_cuda_event_create(ctxPtr : i64) -> i32` -/
def cudaEventCreate (ctxPtr : R) : M R := ffi .cudaEventCreate [ctxPtr]

/-- `cl_cuda_event_record(ctxPtr : i64, eventId : i32,
    streamId : i32) -> i32` -/
def cudaEventRecord
    (ctxPtr eventId streamId : R)
    : M R :=
  ffi .cudaEventRecord
    [ctxPtr, eventId, streamId]

/-- `cl_cuda_stream_wait_event(ctxPtr : i64, streamId : i32,
    eventId : i32) -> i32` -/
def cudaStreamWaitEvent
    (ctxPtr streamId eventId : R)
    : M R :=
  ffi .cudaStreamWaitEvent
    [ctxPtr, streamId, eventId]

/-- `cl_cuda_event_elapsed_ms_bits(ctxPtr : i64, startEventId : i32,
    endEventId : i32) -> i32` -/
def cudaEventElapsedMsBits
    (ctxPtr startEventId endEventId : R)
    : M R :=
  ffi .cudaEventElapsedMsBits
    [ctxPtr, startEventId, endEventId]

/-- `cl_cuda_event_destroy(ctxPtr : i64, eventId : i32) -> i32` -/
def cudaEventDestroy
    (ctxPtr eventId : R)
    : M R :=
  ffi .cudaEventDestroy
    [ctxPtr, eventId]

/-- `cl_cuda_graph_begin_capture(ctxPtr : i64, streamId : i32) -> i32` -/
def cudaGraphBeginCapture
    (ctxPtr streamId : R)
    : M R :=
  ffi .cudaGraphBeginCapture
    [ctxPtr, streamId]

/-- `cl_cuda_graph_end_capture(ctxPtr : i64, streamId : i32) -> i32` -/
def cudaGraphEndCapture
    (ctxPtr streamId : R)
    : M R :=
  ffi .cudaGraphEndCapture
    [ctxPtr, streamId]

/-- `cl_cuda_graph_upload(ctxPtr : i64, graphId : i32,
    streamId : i32) -> i32` -/
def cudaGraphUpload
    (ctxPtr graphId streamId : R)
    : M R :=
  ffi .cudaGraphUpload
    [ctxPtr, graphId, streamId]

/-- `cl_cuda_graph_launch(ctxPtr : i64, graphId : i32,
    streamId : i32) -> i32` -/
def cudaGraphLaunch
    (ctxPtr graphId streamId : R)
    : M R :=
  ffi .cudaGraphLaunch
    [ctxPtr, graphId, streamId]

/-- `cl_cuda_graph_destroy(ctxPtr : i64, graphId : i32) -> i32` -/
def cudaGraphDestroy
    (ctxPtr graphId : R)
    : M R :=
  ffi .cudaGraphDestroy
    [ctxPtr, graphId]

/-- `cl_cuda_pinned_alloc(ctxPtr : i64, size : i64) -> i32` -/
def cudaPinnedAlloc
    (ctxPtr size : R)
    : M R :=
  ffi .cudaPinnedAlloc
    [ctxPtr, size]

/-- `cl_cuda_pinned_ptr(ctxPtr : i64, pinnedId : i32) -> i64` -/
def cudaPinnedPtr
    (ctxPtr pinnedId : R)
    : M R :=
  ffi .cudaPinnedPtr
    [ctxPtr, pinnedId]

/-- `cl_cuda_pinned_free(ctxPtr : i64, pinnedId : i32) -> i32` -/
def cudaPinnedFree
    (ctxPtr pinnedId : R)
    : M R :=
  ffi .cudaPinnedFree
    [ctxPtr, pinnedId]

/-- `cl_cuda_launch(ctxPtr : i64, kernelPtr : i64, nBufs : i32,
    bindPtr : i64, gridX : i32, gridY : i32, gridZ : i32, blockX : i32,
    blockY : i32, blockZ : i32) -> i32` -/
def cudaLaunch
    (ctxPtr kernelPtr nBufs bindPtr gridX gridY gridZ blockX blockY blockZ :
     R)
    : M R :=
  ffi .cudaLaunch
    [ctxPtr, kernelPtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY,
     blockZ]

/-- `cl_cuda_launch_named(ctxPtr : i64, kernelPtr : i64, namePtr : i64,
    nBufs : i32, bindPtr : i64, gridX : i32, gridY : i32, gridZ : i32,
    blockX : i32, blockY : i32, blockZ : i32) -> i32` -/
def cudaLaunchNamed
    (ctxPtr kernelPtr namePtr nBufs bindPtr gridX gridY gridZ blockX blockY
     blockZ : R)
    : M R :=
  ffi .cudaLaunchNamed
    [ctxPtr, kernelPtr, namePtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX,
     blockY, blockZ]

/-- `cl_cuda_launch_on_stream(ctxPtr : i64, kernelPtr : i64, nBufs : i32,
    bindPtr : i64, gridX : i32, gridY : i32, gridZ : i32, blockX : i32,
    blockY : i32, blockZ : i32, streamId : i32) -> i32` -/
def cudaLaunchOnStream
    (ctxPtr kernelPtr nBufs bindPtr gridX gridY gridZ blockX blockY blockZ
     streamId : R)
    : M R :=
  ffi .cudaLaunchOnStream
    [ctxPtr, kernelPtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY,
     blockZ, streamId]

/-- `cl_cuda_launch_named_on_stream(ctxPtr : i64, kernelPtr : i64,
    namePtr : i64, nBufs : i32, bindPtr : i64, gridX : i32, gridY : i32,
    gridZ : i32, blockX : i32, blockY : i32, blockZ : i32,
    streamId : i32) -> i32` -/
def cudaLaunchNamedOnStream
    (ctxPtr kernelPtr namePtr nBufs bindPtr gridX gridY gridZ blockX blockY
     blockZ streamId : R)
    : M R :=
  ffi .cudaLaunchNamedOnStream
    [ctxPtr, kernelPtr, namePtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX,
     blockY, blockZ, streamId]

/-- `cl_cuda_sync(ctxPtr : i64) -> i32` -/
def cudaSync (ctxPtr : R) : M R := ffi .cudaSync [ctxPtr]

/-- `cl_cuda_cleanup(ctxSlotPtr : i64) -> ()` -/
def cudaCleanup (ctxSlotPtr : R) : M Unit := ffiVoid .cudaCleanup [ctxSlotPtr]

/-- `cl_cublas_sgemv(ctxPtr : i64, trans : i32, m : i32, n : i32,
    alphaBits : i32, aBuf : i32, xBuf : i32, betaBits : i32,
    yBuf : i32) -> i32` -/
def cublasSgemv
    (ctxPtr trans m n alphaBits aBuf xBuf betaBits yBuf : R)
    : M R :=
  ffi .cublasSgemv
    [ctxPtr, trans, m, n, alphaBits, aBuf, xBuf, betaBits, yBuf]

/-- `cl_cublas_sgemv_on_stream(ctxPtr : i64, trans : i32, m : i32, n : i32,
    alphaBits : i32, aBuf : i32, xBuf : i32, betaBits : i32, yBuf : i32,
    streamId : i32) -> i32` -/
def cublasSgemvOnStream
    (ctxPtr trans m n alphaBits aBuf xBuf betaBits yBuf streamId : R)
    : M R :=
  ffi .cublasSgemvOnStream
    [ctxPtr, trans, m, n, alphaBits, aBuf, xBuf, betaBits, yBuf, streamId]

/-- `cl_cublas_sgemm_strided_batched(ctxPtr : i64, transa : i32,
    transb : i32, m : i32, n : i32, k : i32, alphaBits : i32, aBuf : i32,
    strideA : i64, bBuf : i32, strideB : i64, betaBits : i32, cBuf : i32,
    strideC : i64, batchCount : i32, offA : i64, offB : i64, offC : i64,
    ldA : i32, ldB : i32, ldC : i32) -> i32` -/
def cublasSgemm
    (ctxPtr transa transb m n k alphaBits aBuf strideA bBuf strideB betaBits
     cBuf strideC batchCount offA offB offC ldA ldB ldC : R)
    : M R :=
  ffi .cublasSgemm
    [ctxPtr, transa, transb, m, n, k, alphaBits, aBuf, strideA, bBuf,
     strideB, betaBits, cBuf, strideC, batchCount, offA, offB, offC, ldA,
     ldB, ldC]

/-- `cl_cublas_sgemm_strided_batched_on_stream(ctxPtr : i64, transa : i32,
    transb : i32, m : i32, n : i32, k : i32, alphaBits : i32, aBuf : i32,
    strideA : i64, bBuf : i32, strideB : i64, betaBits : i32, cBuf : i32,
    strideC : i64, batchCount : i32, streamId : i32, offA : i64, offB : i64,
    offC : i64, ldA : i32, ldB : i32, ldC : i32) -> i32` -/
def cublasSgemmOnStream
    (ctxPtr transa transb m n k alphaBits aBuf strideA bBuf strideB betaBits
     cBuf strideC batchCount streamId offA offB offC ldA ldB ldC : R)
    : M R :=
  ffi .cublasSgemmOnStream
    [ctxPtr, transa, transb, m, n, k, alphaBits, aBuf, strideA, bBuf,
     strideB, betaBits, cBuf, strideC, batchCount, streamId, offA, offB,
     offC, ldA, ldB, ldC]

/-- `cl_cublas_ptr_array(ctxPtr : i64, arrBuf : i32, slot : i32,
    srcBuf : i32, off : i64) -> i32` -/
def cublasPtrArray
    (ctxPtr arrBuf slot srcBuf off : R)
    : M R :=
  ffi .cublasPtrArray
    [ctxPtr, arrBuf, slot, srcBuf, off]

/-- `cl_cublas_sgemm_batched_on_stream(ctxPtr : i64, transa : i32,
    transb : i32, m : i32, n : i32, k : i32, alphaBits : i32, aArr : i32,
    bArr : i32, betaBits : i32, cArr : i32, batchCount : i32,
    streamId : i32) -> i32` -/
def cublasSgemmBatchedOnStream
    (ctxPtr transa transb m n k alphaBits aArr bArr betaBits cArr batchCount
     streamId : R)
    : M R :=
  ffi .cublasSgemmBatchedOnStream
    [ctxPtr, transa, transb, m, n, k, alphaBits, aArr, bArr, betaBits, cArr,
     batchCount, streamId]

/-- `cl_cuda_pinned_ptr_at(ctxPtr : i64, pinnedId : i32, off : i64,
    len : i64) -> i64` -/
def cudaPinnedPtrAt
    (ctxPtr pinnedId off len : R)
    : M R :=
  ffi .cudaPinnedPtrAt
    [ctxPtr, pinnedId, off, len]

/-- `cl_cuda_mem_info_free(ctxPtr : i64) -> i64` -/
def cudaMemInfoFree (ctxPtr : R) : M R := ffi .cudaMemInfoFree [ctxPtr]

/-- `cl_cuda_mem_info_total(ctxPtr : i64) -> i64` -/
def cudaMemInfoTotal (ctxPtr : R) : M R := ffi .cudaMemInfoTotal [ctxPtr]

/-- `cl_cublas_gemm_ex_bf16(ctxPtr : i64, transa : i32, transb : i32,
    m : i32, n : i32, k : i32, alphaBits : i32, aBuf : i32, bBuf : i32,
    betaBits : i32, cBuf : i32, offA : i64, offB : i64, offC : i64,
    ldA : i32, ldB : i32, ldC : i32) -> i32` -/
def cublasGemmExBf16
    (ctxPtr transa transb m n k alphaBits aBuf bBuf betaBits cBuf offA offB
     offC ldA ldB ldC : R)
    : M R :=
  ffi .cublasGemmExBf16
    [ctxPtr, transa, transb, m, n, k, alphaBits, aBuf, bBuf, betaBits, cBuf,
     offA, offB, offC, ldA, ldB, ldC]

/-- `cl_cublas_gemm_strided_batched_ex_bf16(ctxPtr : i64, transa : i32,
    transb : i32, m : i32, n : i32, k : i32, alphaBits : i32, aBuf : i32,
    strideA : i64, bBuf : i32, strideB : i64, betaBits : i32, cBuf : i32,
    strideC : i64, batchCount : i32, offA : i64, offB : i64, offC : i64,
    ldA : i32, ldB : i32, ldC : i32) -> i32` -/
def cublasGemmStridedBatchedExBf16
    (ctxPtr transa transb m n k alphaBits aBuf strideA bBuf strideB betaBits
     cBuf strideC batchCount offA offB offC ldA ldB ldC : R)
    : M R :=
  ffi .cublasGemmStridedBatchedExBf16
    [ctxPtr, transa, transb, m, n, k, alphaBits, aBuf, strideA, bBuf,
     strideB, betaBits, cBuf, strideC, batchCount, offA, offB, offC, ldA,
     ldB, ldC]

end AlgorithmLib.HProg.Sur.Raw
