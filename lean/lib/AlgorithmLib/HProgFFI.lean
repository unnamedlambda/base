import AlgorithmLib.FFI
import AlgorithmLib.FFIRaw
import AlgorithmLib.HProg

/-!
# The FFI call wrappers, over `HProg.Sur`

`FFI.lean` says who the runtime exports and what each one's signature is;
`FFIStd.lean` turns that into the table every body is checked and compiled
against.

What lives here is the *call* side: reading a context pointer out of its slot,
turning offsets into addresses, and issuing the call, in the surface a term is
written in.
-/

namespace AlgorithmLib.HProg.Sur

open AlgorithmLib.IR

-- ---------------------------------------------------------------------------
-- Contexts
--
-- Every subsystem keeps its context pointer in a slot of shared memory, written
-- by its `init` and read by everything after. The three functions below are the
-- whole pattern; the wrappers differ only in which slot and which callee.
-- ---------------------------------------------------------------------------

/-- The address of the slot holding a subsystem's context pointer. -/
def ctxSlotPtr (ptr : R) (slotOffset : Nat) : M R := absAddr ptr slotOffset

/-- The context pointer itself. -/
def ctxPtr (ptr : R) (slotOffset : Nat) : M R := do
  load64 (← ctxSlotPtr ptr slotOffset)

/-- `init` takes the *slot* — it writes the context there. -/
private def initAt (f : IR.Ffi) (ptr : R) (slotOffset : Nat) : M Unit := do
  ffiVoid f [← ctxSlotPtr ptr slotOffset]

-- ---------------------------------------------------------------------------
-- wgpu
-- ---------------------------------------------------------------------------

def gpuCtxSlotPtr (ptr : R) (slotOffset : Nat := ContextSlots.wgpu) : M R :=
  ctxSlotPtr ptr slotOffset

def gpuCtxPtr (ptr : R) (slotOffset : Nat := ContextSlots.wgpu) : M R :=
  ctxPtr ptr slotOffset

def gpuInit (ptr : R) (slotOffset : Nat := ContextSlots.wgpu) : M Unit :=
  initAt IR.Ffi.gpuInit ptr slotOffset

def gpuCreateBuffer (ptr size : R)
    (slotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← gpuCtxPtr ptr slotOffset
  call IR.Ffi.gpuCreateBuffer.id [c, size]

def gpuCreatePipeline (ptr shaderOff bindOff nBindings : R)
    (slotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← gpuCtxPtr ptr slotOffset
  let shaderPtr ← iadd ptr shaderOff
  let bindPtr ← iadd ptr bindOff
  call IR.Ffi.gpuCreatePipeline.id [c, shaderPtr, bindPtr, nBindings]

def gpuUpload (ptr bufId srcOff size : R)
    (slotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← gpuCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call IR.Ffi.gpuUpload.id [c, bufId, srcPtr, size]

def gpuDownload (ptr bufId dstOff size : R)
    (slotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← gpuCtxPtr ptr slotOffset
  let dstPtr ← iadd ptr dstOff
  call IR.Ffi.gpuDownload.id [c, bufId, dstPtr, size]

def gpuDispatch (ptr pipelineId wgX wgY wgZ : R)
    (slotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← gpuCtxPtr ptr slotOffset
  call IR.Ffi.gpuDispatch.id [c, pipelineId, wgX, wgY, wgZ]

def gpuCleanup (ptr : R) (slotOffset : Nat := ContextSlots.wgpu) : M Unit :=
  initAt IR.Ffi.gpuCleanup ptr slotOffset

-- ---------------------------------------------------------------------------
-- Hash table
--
-- The same shape as wgpu: `init` and `cleanup` take the *slot*, everything
-- else the context pointer read out of it. The offset is a parameter because
-- generators do not agree on one -- `WordCountAlgorithm` keeps it at 0 and
-- `HProgCorpus` at 0x80 -- and `ContextSlots.ht` is only the default.
-- ---------------------------------------------------------------------------

def htCtxSlotPtr (ptr : R) (slotOffset : Nat := ContextSlots.ht) : M R :=
  ctxSlotPtr ptr slotOffset

def htCtxPtr (ptr : R) (slotOffset : Nat := ContextSlots.ht) : M R :=
  ctxPtr ptr slotOffset

def htInit (ptr : R) (slotOffset : Nat := ContextSlots.ht) : M Unit :=
  initAt IR.Ffi.htInit ptr slotOffset

def htCleanup (ptr : R) (slotOffset : Nat := ContextSlots.ht) : M Unit :=
  initAt IR.Ffi.htCleanup ptr slotOffset

/-- Allocate the table itself. The context must already hold one. -/
def htCreate (ptr : R) (slotOffset : Nat := ContextSlots.ht) : M R := do
  Raw.htCreate (← htCtxPtr ptr slotOffset)

/-- Look `key` up and write the value to `resultOff`; the result says whether
    it was found. Both offsets are relative to the base pointer. -/
def htLookup (ptr keyOff keyLen resultOff : R)
    (slotOffset : Nat := ContextSlots.ht) : M R := do
  let c ← htCtxPtr ptr slotOffset
  Raw.htLookup c (← iadd ptr keyOff) keyLen (← iadd ptr resultOff)

def htInsert (ptr keyOff keyLen valOff valLen : R)
    (slotOffset : Nat := ContextSlots.ht) : M Unit := do
  let c ← htCtxPtr ptr slotOffset
  Raw.htInsert c (← iadd ptr keyOff) keyLen (← iadd ptr valOff) valLen

/-- Add `addend` to `key`'s value, inserting it if absent; the result is the
    value after the addition. -/
def htIncrement (ptr keyOff keyLen addend : R)
    (slotOffset : Nat := ContextSlots.ht) : M R := do
  let c ← htCtxPtr ptr slotOffset
  Raw.htIncrement c (← iadd ptr keyOff) keyLen addend

def htCount (ptr : R) (slotOffset : Nat := ContextSlots.ht) : M R := do
  Raw.htCount (← htCtxPtr ptr slotOffset)

/-- The `index`th entry, written to `keyOutOff` and `valOutOff`. Iterating to
    `htCount` is how a program reads the table back out. -/
def htGetEntry (ptr index keyOutOff valOutOff : R)
    (slotOffset : Nat := ContextSlots.ht) : M R := do
  let c ← htCtxPtr ptr slotOffset
  Raw.htGetEntry c index (← iadd ptr keyOutOff) (← iadd ptr valOutOff)

-- ---------------------------------------------------------------------------
-- Files
-- ---------------------------------------------------------------------------

/-- Read a file into shared memory; the result is the byte count. -/
def readFile (ptr : R) (fnRead : FnRef) (filenameOff dataOff : Nat) : M R := do
  let fnOff ← iconst64 filenameOff
  let dOff ← iconst64 dataOff
  let zero ← iconst64 0
  call fnRead.id [ptr, fnOff, dOff, zero, zero]

/-- Write a region of shared memory to a file; the result is the byte count. -/
def writeFile (ptr : R) (fnWrite : FnRef) (filenameOff srcOff : Nat)
    (fileOffset size : R) : M R := do
  let fnOff ← iconst64 filenameOff
  let sOff ← iconst64 srcOff
  call fnWrite.id [ptr, fnOff, sOff, fileOffset, size]

/-- Write starting at file offset 0. -/
def writeFile0 (ptr : R) (fnWrite : FnRef) (filenameOff srcOff : Nat) (size : R) : M R := do
  let zero ← iconst64 0
  writeFile ptr fnWrite filenameOff srcOff zero size

/-- `(x + 3) &&& ~3` — wgpu's `COPY_BUFFER_ALIGNMENT`. -/
def alignUp4 (v : R) : M R := do
  let c3 ← iconst64 3
  let sum ← iadd v c3
  let negFour ← iconst64 (-4)
  band sum negFour

-- ---------------------------------------------------------------------------
-- Typed fields
--
-- A `Layout.Fld` names a byte range in shared memory. The scalar forms reject
-- `.bytes n` through `IsScalar`; the `At` forms take a position inside a byte
-- region and a proof that the access fits, discharged by `omega` at the call
-- ---------------------------------------------------------------------------

/-- A field's offset as a constant. -/
def fldOffset (f : Layout.Fld t) : M R := iconst64 f.offset

/-- A field's address: `base + offset`. -/
def fldAddr (base : R) (f : Layout.Fld t) : M R := absAddr base f.offset

/-- How a scalar field is read and written, chosen per field type rather than by
    a match, so `.bytes n` has no instance and cannot be passed. -/
class IsScalarR (t : Layout.FieldTy) where
  scalarStore : R → R → M Unit
  scalarLoad  : R → M R

instance : IsScalarR .u8 where
  scalarStore val addr := istore8 val addr
  scalarLoad  addr     := uload8_64 addr

instance : IsScalarR .i32 where
  scalarStore val addr := storeUnaligned val addr
  scalarLoad  addr     := uload32_64 addr

instance : IsScalarR .i64 where
  scalarStore val addr := storeUnaligned val addr
  scalarLoad  addr     := load64 addr

def fldStore (base : R) (f : Layout.Fld t) [inst : IsScalarR t] (val : R) : M Unit := do
  inst.scalarStore val (← fldAddr base f)

def fldLoad (base : R) (f : Layout.Fld t) [inst : IsScalarR t] : M R := do
  inst.scalarLoad (← fldAddr base f)

def fldStoreAt (base : R) (f : Layout.Fld (.bytes n)) (i : Nat) (val : R)
    (_h : i + 8 ≤ n := by omega) : M Unit := do
  storeUnaligned val (← absAddr base (f.offset + i))


def fldStore32At (base : R) (f : Layout.Fld (.bytes n)) (i : Nat) (val : R)
    (_h : i + 4 ≤ n := by omega) : M Unit := do
  storeUnaligned val (← absAddr base (f.offset + i))

def fldLoadAt (base : R) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 8 ≤ n := by omega) : M R := do
  load64 (← absAddr base (f.offset + i))

def fldLoad8At (base : R) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 1 ≤ n := by omega) : M R := do
  uload8_64 (← absAddr base (f.offset + i))

def fldLoad32At (base : R) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 4 ≤ n := by omega) : M R := do
  uload32_64 (← absAddr base (f.offset + i))

/-- Read a file using typed field handles for the filename and data regions. -/
def fldReadFile (ptr : R) (fnRead : FnRef)
    (filenameFld : Layout.Fld ft) (dataFld : Layout.Fld dt) : M R :=
  readFile ptr fnRead filenameFld.offset dataFld.offset

/-- Write a whole field to a file, from offset 0. -/
def fldWriteFile0 (ptr : R) (fnWrite : FnRef)
    (filenameFld : Layout.Fld ft) (srcFld : Layout.Fld st) (size : R) : M R :=
  writeFile0 ptr fnWrite filenameFld.offset srcFld.offset size

-- ---------------------------------------------------------------------------
-- Window
-- ---------------------------------------------------------------------------

def windowCtxSlotPtr (ptr : R) (slotOffset : Nat := ContextSlots.window) : M R :=
  ctxSlotPtr ptr slotOffset

def windowCtxPtr (ptr : R) (slotOffset : Nat := ContextSlots.window) : M R :=
  ctxPtr ptr slotOffset

def windowInit (ptr : R)
    (slotOffset : Nat := ContextSlots.window) : M Unit :=
  initAt IR.Ffi.windowInit ptr slotOffset

def windowOpen (ptr width height titleOff titleLen blitOff blitLen : R)
    (slotOffset : Nat := ContextSlots.window) : M R := do
  let c ← windowCtxPtr ptr slotOffset
  let titlePtr ← iadd ptr titleOff
  let blitPtr ← iadd ptr blitOff
  call IR.Ffi.windowOpen.id [c, width, height, titlePtr, titleLen, blitPtr, blitLen]

def windowPoll (ptr eventsOff maxEvents : R)
    (slotOffset : Nat := ContextSlots.window) : M R := do
  let c ← windowCtxPtr ptr slotOffset
  let eventsPtr ← iadd ptr eventsOff
  call IR.Ffi.windowPoll.id [c, eventsPtr, maxEvents]

/-- Blit a wgpu storage buffer to the swapchain, so it takes both contexts. -/
def windowPresentGpuBuffer (ptr bufId : R)
    (slotOffset : Nat := ContextSlots.window)
    (gpuSlotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← windowCtxPtr ptr slotOffset
  let g ← gpuCtxPtr ptr gpuSlotOffset
  call IR.Ffi.windowPresentGpuBuffer.id [c, g, bufId]

def windowCleanup (ptr : R)
    (slotOffset : Nat := ContextSlots.window) : M Unit :=
  initAt IR.Ffi.windowCleanup ptr slotOffset

-- ---------------------------------------------------------------------------
-- CUDA
--
-- `Raw` names a host pointer the caller owns; the plain form takes an offset
-- into shared memory and turns it into one. `Offset` names a position *inside*
-- the device buffer, so the host pointee need not be the whole allocation.
-- ---------------------------------------------------------------------------

def cudaCtxSlotPtr (ptr : R) (slotOffset : Nat := ContextSlots.cuda) : M R :=
  ctxSlotPtr ptr slotOffset

def cudaCtxPtr (ptr : R) (slotOffset : Nat := ContextSlots.cuda) : M R :=
  ctxPtr ptr slotOffset

def cudaInit (ptr : R) (slotOffset : Nat := ContextSlots.cuda) : M Unit :=
  initAt IR.Ffi.cudaInit ptr slotOffset

def cudaCreateBuffer (ptr size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cudaCreateBuffer.id [c, size]

def cudaUpload (ptr bufId srcOff size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call IR.Ffi.cudaUpload.id [c, bufId, srcPtr, size]

def cudaUploadRaw (ptr bufId srcPtr size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cudaUpload.id [c, bufId, srcPtr, size]

def cudaUploadRawOffset (ptr bufId bufOff srcPtr size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cudaUploadOffset.id [c, bufId, bufOff, srcPtr, size]

def cudaUploadOffset (ptr bufId bufOff srcOff size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call IR.Ffi.cudaUploadOffset.id [c, bufId, bufOff, srcPtr, size]

def cudaUploadAsync (ptr bufId srcOff size streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call IR.Ffi.cudaUploadAsync.id [c, bufId, srcPtr, size, streamId]

def cudaUploadOffsetAsync (ptr bufId bufOff srcOff size streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call IR.Ffi.cudaUploadOffsetAsync.id [c, bufId, bufOff, srcPtr, size, streamId]

def cudaDownload (ptr bufId dstOff size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let dstPtr ← iadd ptr dstOff
  call IR.Ffi.cudaDownload.id [c, bufId, dstPtr, size]

def cudaDownloadRaw (ptr bufId dstPtr size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cudaDownload.id [c, bufId, dstPtr, size]

def cudaDownloadRawOffset (ptr bufId bufOff dstPtr size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cudaDownloadOffset.id [c, bufId, bufOff, dstPtr, size]

def cudaDownloadAsync (ptr bufId dstOff size streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let dstPtr ← iadd ptr dstOff
  call IR.Ffi.cudaDownloadAsync.id [c, bufId, dstPtr, size, streamId]

/-- The pinned pool's host address at `off`, or `-1` when `off + len` runs past
    the allocation.

    Use this wherever a source offset is computed rather than fixed: the async
    upload checks the device range it writes but takes its source as a bare
    address, so an unchecked offset into a large pool uploads whatever the
    process happens to hold there. -/
def cudaPinnedPtrAt (ptr pinnedId off len : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cudaPinnedPtrAt.id [c, pinnedId, off, len]

/-- Free device memory in bytes. What a card has left after weights, caches and
    the driver's own reservations is not a number that can be written down ahead
    of the machine, so sizes that depend on it are read here. -/
def cudaMemInfoFree (ptr : R) (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cudaMemInfoFree.id [c]

/-- Total device memory in bytes. -/
def cudaMemInfoTotal (ptr : R) (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cudaMemInfoTotal.id [c]

def cudaSync (ptr : R) (slotOffset : Nat := ContextSlots.cuda) : M R := do
  call IR.Ffi.cudaSync.id [← cudaCtxPtr ptr slotOffset]

def cudaFreeBuffer (ptr bufId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cudaFreeBuffer.id [c, bufId]

def cudaCleanup (ptr : R)
    (slotOffset : Nat := ContextSlots.cuda) : M Unit :=
  initAt IR.Ffi.cudaCleanup ptr slotOffset

def cudaLaunch (ptr kernelOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let bindPtr ← iadd ptr bindOff
  call IR.Ffi.cudaLaunch.id
    [c, kernelPtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ]

def cudaLaunchNamed (ptr kernelOff nameOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let namePtr ← iadd ptr nameOff
  let bindPtr ← iadd ptr bindOff
  call IR.Ffi.cudaLaunchNamed.id
    [c, kernelPtr, namePtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ]

def cudaLaunchOnStream (ptr kernelOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let bindPtr ← iadd ptr bindOff
  call IR.Ffi.cudaLaunchOnStream.id
    [c, kernelPtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ, streamId]

/-- `y ← alpha·op(A)·x + beta·y`. -/
def cublasSgemv (ptr trans m n alphaBits aBuf xBuf betaBits yBuf : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cublasSgemv.id [c, trans, m, n, alphaBits, aBuf, xBuf, betaBits, yBuf]

/-- `C ← alpha·op(A)·op(B) + beta·C`, strided-batched. The trailing offsets and
    leading dimensions default to zero, which the wrapper reads as "no offset,
    default leading dimension". -/
def cublasSgemmStridedBatched (ptr transA transB m n k alphaBits aBuf strideA bBuf strideB betaBits
     cBuf strideC batchCount : R)
    (slotOffset : Nat := ContextSlots.cuda)
    (offA offB offC : Nat := 0) (ldA ldB ldC : Nat := 0) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let oa ← iconst64 offA
  let ob ← iconst64 offB
  let oc ← iconst64 offC
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  call IR.Ffi.cublasSgemm.id
    [c, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB,
     betaBits, cBuf, strideC, batchCount, oa, ob, oc, la, lb, lc]

/-- The same contraction, issued on a created stream so a capture records it.
    cuBLAS is stream-bound through its handle, so the FFI keeps one handle per
    stream rather than retargeting the default. -/
def cublasSgemmStridedBatchedOnStream (ptr transA transB m n k alphaBits aBuf strideA bBuf strideB betaBits
     cBuf strideC batchCount streamId : R)
    (slotOffset : Nat := ContextSlots.cuda)
    (offA offB offC : Nat := 0) (ldA ldB ldC : Nat := 0) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let oa ← iconst64 offA
  let ob ← iconst64 offB
  let oc ← iconst64 offC
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  call IR.Ffi.cublasSgemmOnStream.id
    [c, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB,
     betaBits, cBuf, strideC, batchCount, streamId, oa, ob, oc, la, lb, lc]

/-- The same contraction with **bf16 operands** and an f32 accumulator and
    result: `C ← alpha·op(A)·op(B) + beta·C`, unbatched.

    Both inputs are bf16 because cuBLAS refuses a mixed pair, so `offA` and
    `offB` count 2-byte elements while `offC` counts 4-byte ones. Offsets and
    leading dimensions default to zero, read as "no offset, default leading
    dimension", as in the strided form. -/
def cublasGemmExBf16 (ptr transA transB m n k alphaBits aBuf bBuf betaBits cBuf : R)
    (slotOffset : Nat := ContextSlots.cuda)
    (offA offB offC : Nat := 0) (ldA ldB ldC : Nat := 0) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let oa ← iconst64 offA
  let ob ← iconst64 offB
  let oc ← iconst64 offC
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  call IR.Ffi.cublasGemmExBf16.id
    [c, transA, transB, m, n, k, alphaBits, aBuf, bBuf, betaBits, cBuf,
     oa, ob, oc, la, lb, lc]

/-- **The bf16 contraction, once per batch member at a fixed stride.**

    Attention with a bf16 key cache: one member per key head, the query narrowed
    to match because cuBLAS refuses a mixed pair. Strides and offsets count
    elements, and an element is two bytes on both inputs and four on the result
    -- so a stride that is right for the `Float32` form is twice what this one
    wants for its inputs and exactly right for its output.

    -/
def cublasGemmStridedBatchedExBf16 (ptr transA transB m n k alphaBits aBuf strideA
     bBuf strideB betaBits cBuf strideC batchCount : R)
    (slotOffset : Nat := ContextSlots.cuda)
    (offA offB offC : Nat := 0) (ldA ldB ldC : Nat := 0) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let oa ← iconst64 offA
  let ob ← iconst64 offB
  let oc ← iconst64 offC
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  call IR.Ffi.cublasGemmStridedBatchedExBf16.id
    [c, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB,
     betaBits, cBuf, strideC, batchCount, oa, ob, oc, la, lb, lc]

/-- The same, with the operand offsets held in **registers** rather than fixed
    at emission.

    A sliding window moves with the position, so the offset that names its first
    key is not a number the generator knows. Nothing else changes: an offset
    still moves a pointer and leaves the matrix contracted alone, which is why
    this costs no law the constant form does not already cost. -/
def cublasGemmExBf16At (ptr transA transB m n k alphaBits aBuf bBuf betaBits cBuf
     offA offB offC : R)
    (slotOffset : Nat := ContextSlots.cuda) (ldA ldB ldC : Nat := 0) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  call IR.Ffi.cublasGemmExBf16.id
    [c, transA, transB, m, n, k, alphaBits, aBuf, bBuf, betaBits, cBuf,
     offA, offB, offC, la, lb, lc]

/-- The strided-batched contraction with its operand offsets in registers.

    Attention over a sliding window is the reason: the window's first key is a
    function of the position, so `offA`/`offB` are computed at run time. The
    constant-offset form above is the same call with the offsets frozen. -/
def cublasSgemmStridedBatchedOnStreamAt (ptr transA transB m n k alphaBits aBuf strideA
     bBuf strideB betaBits cBuf strideC batchCount streamId offA offB offC : R)
    (slotOffset : Nat := ContextSlots.cuda) (ldA ldB ldC : Nat := 0) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  call IR.Ffi.cublasSgemmOnStream.id
    [c, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB,
     betaBits, cBuf, strideC, batchCount, streamId, offA, offB, offC, la, lb, lc]

/-- Store `srcBuf`'s device pointer, advanced by `off` f32 elements, into entry
    `slot` of the pointer array held in `arrBuf`. -/
def cublasPtrArray (ptr arrBuf slot srcBuf off : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cublasPtrArray.id [c, arrBuf, slot, srcBuf, off]

/-- A batch of contractions of one shape whose members are named by pointer,
    so they need not sit at a uniform stride inside one allocation. -/
def cublasSgemmBatchedOnStream (ptr transA transB m n k alphaBits aArr bArr betaBits cArr batchCount
     streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call IR.Ffi.cublasSgemmBatchedOnStream.id
    [c, transA, transB, m, n, k, alphaBits, aArr, bArr, betaBits, cArr,
     batchCount, streamId]

def cudaLaunchNamedOnStream (ptr kernelOff nameOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let namePtr ← iadd ptr nameOff
  let bindPtr ← iadd ptr bindOff
  call IR.Ffi.cudaLaunchNamedOnStream.id
    [c, kernelPtr, namePtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ, streamId]

end AlgorithmLib.HProg.Sur

namespace AlgorithmLib.IR

/-- A function at `u0:wrapperIdx` that calls each of `callees` in order.

    Composes stages without the caller having to build the call sequence
    itself; the callees are named by index, so nothing here resolves a symbol.

    The callee table is the wrapper's own — one declaration per distinct index
    — so it is built here rather than passed in. -/
private def wrapperUnique (callees : List Nat) : List Nat :=
  callees.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []

private def wrapperDecls (callees : List Nat) : List FnRef × FnEnv :=
  (wrapperUnique callees).foldl (fun (refs, e) c =>
    let (r, e) := e.declareLocal c [ClifTy.i64] none
    (refs ++ [r], e)) (([] : List FnRef), ({ sigs := [], fns := [] } : FnEnv))

/-- The callee table the wrapper ships, one declaration per distinct index. -/
def sequenceWrapperEnv (callees : List Nat) : FnEnv := (wrapperDecls callees).2

/-- The wrapper's body: each callee, in the order given. -/
def sequenceWrapperBody (callees : List Nat) : HProg.Code :=
  let (refs, env) := wrapperDecls callees
  let unique := wrapperUnique callees
  HProg.Sur.build (env := env) do
    for c in callees do
      let slot := (unique.idxOf? c).getD 0
      HProg.Sur.callVoid (refs[slot]!).id [HProg.Sur.basePtr]

/-- The wrapper is generic in its callees, so the obligation is a parameter
    rather than an auto-param: `decide` at a call site has to reduce the whole
    builder, which exhausts 10 GB at the depths this ships at. A proof by
    induction on `callees` would discharge every site at once and needs lemmas
    about `FnEnv.declare` that do not exist yet. -/
def clifSequenceWrapper (wrapperIdx : Nat) (callees : List Nat)
    (hwf : HProg.wf (sequenceWrapperEnv callees) HProg.ptrParams
             (sequenceWrapperBody callees) = true) : FuncData :=
  HProg.compileFn wrapperIdx (sequenceWrapperBody callees)
    (sequenceWrapperEnv callees) (hwf := hwf)

end AlgorithmLib.IR
