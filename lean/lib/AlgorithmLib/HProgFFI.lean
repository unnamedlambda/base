import AlgorithmLib.FFI
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
def clifSequenceWrapper (wrapperIdx : Nat) (callees : List Nat) : FuncData :=
  let unique : List Nat :=
    callees.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []
  let (refs, env) :=
    unique.foldl (fun (refs, e) c =>
      let (r, e) := e.declareLocal c [ClifTy.i64] none
      (refs ++ [r], e)) (([] : List FnRef), ({ sigs := [], fns := [] } : FnEnv))
  HProg.compileBody wrapperIdx
    (HProg.Sur.build (env := env) do
      for c in callees do
        let slot := (unique.idxOf? c).getD 0
        HProg.Sur.callVoid (refs[slot]!).id [HProg.Sur.basePtr])
    env

end AlgorithmLib.IR
