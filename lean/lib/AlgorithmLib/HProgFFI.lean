import AlgorithmLib.FFI
import AlgorithmLib.HProg

/-!
# The FFI call wrappers, over `HProg.Sur`

`FFI.lean` declares every entry point the runtime exposes and `FFIStd.lean`
runs those declarations once, into the table every body is checked and compiled
against. So a signature is written in exactly one place and two generators
cannot describe the same C symbol differently.

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
private def initAt (fn : FnRef) (ptr : R) (slotOffset : Nat) : M Unit := do
  callVoid fn.id [← ctxSlotPtr ptr slotOffset]

-- ---------------------------------------------------------------------------
-- wgpu
-- ---------------------------------------------------------------------------

def gpuCtxSlotPtr (ptr : R) (slotOffset : Nat := ContextSlots.wgpu) : M R :=
  ctxSlotPtr ptr slotOffset

def gpuCtxPtr (ptr : R) (slotOffset : Nat := ContextSlots.wgpu) : M R :=
  ctxPtr ptr slotOffset

def gpuInit (gpu : GpuSetup) (ptr : R) (slotOffset : Nat := ContextSlots.wgpu) : M Unit :=
  initAt gpu.fnInit ptr slotOffset

def gpuCreateBuffer (gpu : GpuSetup) (ptr size : R)
    (slotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← gpuCtxPtr ptr slotOffset
  call gpu.fnCreateBuffer.id [c, size]

def gpuCreatePipeline (gpu : GpuSetup) (ptr shaderOff bindOff nBindings : R)
    (slotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← gpuCtxPtr ptr slotOffset
  let shaderPtr ← iadd ptr shaderOff
  let bindPtr ← iadd ptr bindOff
  call gpu.fnCreatePipeline.id [c, shaderPtr, bindPtr, nBindings]

def gpuUpload (gpu : GpuSetup) (ptr bufId srcOff size : R)
    (slotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← gpuCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call gpu.fnUpload.id [c, bufId, srcPtr, size]

def gpuDownload (gpu : GpuSetup) (ptr bufId dstOff size : R)
    (slotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← gpuCtxPtr ptr slotOffset
  let dstPtr ← iadd ptr dstOff
  call gpu.fnDownload.id [c, bufId, dstPtr, size]

def gpuDispatch (gpu : GpuSetup) (ptr pipelineId wgX wgY wgZ : R)
    (slotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← gpuCtxPtr ptr slotOffset
  call gpu.fnDispatch.id [c, pipelineId, wgX, wgY, wgZ]

def gpuCleanup (gpu : GpuSetup) (ptr : R) (slotOffset : Nat := ContextSlots.wgpu) : M Unit :=
  initAt gpu.fnCleanup ptr slotOffset

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
-- site. Both are the same guarantees the `IRBuilder` forms give.
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

def fldStore8At (base : R) (f : Layout.Fld (.bytes n)) (i : Nat) (val : R)
    (_h : i + 1 ≤ n := by omega) : M Unit := do
  istore8 val (← absAddr base (f.offset + i))

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

def windowInit (win : WindowSetup) (ptr : R)
    (slotOffset : Nat := ContextSlots.window) : M Unit :=
  initAt win.fnInit ptr slotOffset

def windowOpen (win : WindowSetup) (ptr width height titleOff titleLen blitOff blitLen : R)
    (slotOffset : Nat := ContextSlots.window) : M R := do
  let c ← windowCtxPtr ptr slotOffset
  let titlePtr ← iadd ptr titleOff
  let blitPtr ← iadd ptr blitOff
  call win.fnOpen.id [c, width, height, titlePtr, titleLen, blitPtr, blitLen]

def windowPoll (win : WindowSetup) (ptr eventsOff maxEvents : R)
    (slotOffset : Nat := ContextSlots.window) : M R := do
  let c ← windowCtxPtr ptr slotOffset
  let eventsPtr ← iadd ptr eventsOff
  call win.fnPoll.id [c, eventsPtr, maxEvents]

/-- Blit a wgpu storage buffer to the swapchain, so it takes both contexts. -/
def windowPresentGpuBuffer (win : WindowSetup) (ptr bufId : R)
    (slotOffset : Nat := ContextSlots.window)
    (gpuSlotOffset : Nat := ContextSlots.wgpu) : M R := do
  let c ← windowCtxPtr ptr slotOffset
  let g ← gpuCtxPtr ptr gpuSlotOffset
  call win.fnPresentGpuBuffer.id [c, g, bufId]

def windowCleanup (win : WindowSetup) (ptr : R)
    (slotOffset : Nat := ContextSlots.window) : M Unit :=
  initAt win.fnCleanup ptr slotOffset

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

def cudaInit (cuda : CudaSetup) (ptr : R) (slotOffset : Nat := ContextSlots.cuda) : M Unit :=
  initAt cuda.fnInit ptr slotOffset

def cudaCreateBuffer (cuda : CudaSetup) (ptr size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call cuda.fnCreateBuffer.id [c, size]

def cudaUpload (cuda : CudaSetup) (ptr bufId srcOff size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call cuda.fnUpload.id [c, bufId, srcPtr, size]

def cudaUploadRaw (cuda : CudaSetup) (ptr bufId srcPtr size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call cuda.fnUpload.id [c, bufId, srcPtr, size]

def cudaUploadRawOffset (cuda : CudaSetup) (ptr bufId bufOff srcPtr size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call cuda.fnUploadOffset.id [c, bufId, bufOff, srcPtr, size]

def cudaUploadOffset (cuda : CudaSetup) (ptr bufId bufOff srcOff size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call cuda.fnUploadOffset.id [c, bufId, bufOff, srcPtr, size]

def cudaUploadAsync (cuda : CudaSetup) (ptr bufId srcOff size streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call cuda.fnUploadAsync.id [c, bufId, srcPtr, size, streamId]

def cudaUploadOffsetAsync (cuda : CudaSetup) (ptr bufId bufOff srcOff size streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  call cuda.fnUploadOffsetAsync.id [c, bufId, bufOff, srcPtr, size, streamId]

def cudaDownload (cuda : CudaSetup) (ptr bufId dstOff size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let dstPtr ← iadd ptr dstOff
  call cuda.fnDownload.id [c, bufId, dstPtr, size]

def cudaDownloadRaw (cuda : CudaSetup) (ptr bufId dstPtr size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call cuda.fnDownload.id [c, bufId, dstPtr, size]

def cudaDownloadRawOffset (cuda : CudaSetup) (ptr bufId bufOff dstPtr size : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call cuda.fnDownloadOffset.id [c, bufId, bufOff, dstPtr, size]

def cudaDownloadAsync (cuda : CudaSetup) (ptr bufId dstOff size streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let dstPtr ← iadd ptr dstOff
  call cuda.fnDownloadAsync.id [c, bufId, dstPtr, size, streamId]

def cudaSync (cuda : CudaSetup) (ptr : R) (slotOffset : Nat := ContextSlots.cuda) : M R := do
  call cuda.fnSync.id [← cudaCtxPtr ptr slotOffset]

def cudaFreeBuffer (cuda : CudaSetup) (ptr bufId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call cuda.fnFreeBuffer.id [c, bufId]

def cudaCleanup (cuda : CudaSetup) (ptr : R)
    (slotOffset : Nat := ContextSlots.cuda) : M Unit :=
  initAt cuda.fnCleanup ptr slotOffset

def cudaLaunch (cuda : CudaSetup)
    (ptr kernelOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let bindPtr ← iadd ptr bindOff
  call cuda.fnLaunch.id
    [c, kernelPtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ]

def cudaLaunchNamed (cuda : CudaSetup)
    (ptr kernelOff nameOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let namePtr ← iadd ptr nameOff
  let bindPtr ← iadd ptr bindOff
  call cuda.fnLaunchNamed.id
    [c, kernelPtr, namePtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ]

def cudaLaunchOnStream (cuda : CudaSetup)
    (ptr kernelOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let bindPtr ← iadd ptr bindOff
  call cuda.fnLaunchOnStream.id
    [c, kernelPtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ, streamId]

/-- `y ← alpha·op(A)·x + beta·y`. -/
def cublasSgemv (cublas : IR.CuBlasSetup)
    (ptr trans m n alphaBits aBuf xBuf betaBits yBuf : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call cublas.fnSgemv.id [c, trans, m, n, alphaBits, aBuf, xBuf, betaBits, yBuf]

/-- `C ← alpha·op(A)·op(B) + beta·C`, strided-batched. The trailing offsets and
    leading dimensions default to zero, which the wrapper reads as "no offset,
    default leading dimension". -/
def cublasSgemmStridedBatched (cublas : IR.CuBlasSetup)
    (ptr transA transB m n k alphaBits aBuf strideA bBuf strideB betaBits
     cBuf strideC batchCount : R)
    (slotOffset : Nat := ContextSlots.cuda)
    (offA offB offC : Nat := 0) (ldA ldB ldC : Nat := 0) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let oa ← iconst64 offA
  let ob ← iconst64 offB
  let oc ← iconst64 offC
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  call cublas.fnSgemm.id
    [c, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB,
     betaBits, cBuf, strideC, batchCount, oa, ob, oc, la, lb, lc]

/-- The same contraction, issued on a created stream so a capture records it.
    cuBLAS is stream-bound through its handle, so the FFI keeps one handle per
    stream rather than retargeting the default. -/
def cublasSgemmStridedBatchedOnStream (cublas : IR.CuBlasSetup)
    (ptr transA transB m n k alphaBits aBuf strideA bBuf strideB betaBits
     cBuf strideC batchCount streamId : R)
    (slotOffset : Nat := ContextSlots.cuda)
    (offA offB offC : Nat := 0) (ldA ldB ldC : Nat := 0) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let oa ← iconst64 offA
  let ob ← iconst64 offB
  let oc ← iconst64 offC
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  call cublas.fnSgemmOnStream.id
    [c, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB,
     betaBits, cBuf, strideC, batchCount, streamId, oa, ob, oc, la, lb, lc]

/-- Store `srcBuf`'s device pointer, advanced by `off` f32 elements, into entry
    `slot` of the pointer array held in `arrBuf`. -/
def cublasPtrArray (cublas : IR.CuBlasSetup) (ptr arrBuf slot srcBuf off : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call cublas.fnPtrArray.id [c, arrBuf, slot, srcBuf, off]

/-- A batch of contractions of one shape whose members are named by pointer,
    so they need not sit at a uniform stride inside one allocation. -/
def cublasSgemmBatchedOnStream (cublas : IR.CuBlasSetup)
    (ptr transA transB m n k alphaBits aArr bArr betaBits cArr batchCount
     streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  call cublas.fnSgemmBatchedOnStream.id
    [c, transA, transB, m, n, k, alphaBits, aArr, bArr, betaBits, cArr,
     batchCount, streamId]

def cudaLaunchNamedOnStream (cuda : CudaSetup)
    (ptr kernelOff nameOff nBufs bindOff gridX gridY gridZ blockX blockY blockZ streamId : R)
    (slotOffset : Nat := ContextSlots.cuda) : M R := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let namePtr ← iadd ptr nameOff
  let bindPtr ← iadd ptr bindOff
  call cuda.fnLaunchNamedOnStream.id
    [c, kernelPtr, namePtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ, streamId]

end AlgorithmLib.HProg.Sur

namespace AlgorithmLib.IR

/-- A function at `u0:wrapperIdx` that calls each of `callees` in order.

    Composes stages without the caller having to build the call sequence
    itself; the callees are named by index, so nothing here resolves a symbol.

    The callee table is the wrapper's own — `declareLocal` for each distinct
    index — so it is built here rather than passed in. -/
def clifSequenceWrapper (wrapperIdx : Nat) (callees : List Nat) : FuncData :=
  let unique : List Nat :=
    callees.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []
  let (refs, env) :=
    HProg.envOf (unique.mapM fun c => declareLocal c [ClifTy.i64] none)
  HProg.compileBody wrapperIdx
    (HProg.Sur.build (env := env) do
      for c in callees do
        let slot := (unique.idxOf? c).getD 0
        HProg.Sur.callVoid (refs[slot]!).id [HProg.Sur.basePtr])
    env

end AlgorithmLib.IR
