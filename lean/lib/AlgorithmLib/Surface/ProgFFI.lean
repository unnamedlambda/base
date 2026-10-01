import AlgorithmLib.Surface.Prog
import AlgorithmLib.Surface.Layout

/-!
# The FFI call wrappers, over `Prog`

`FFI.lean` says who the runtime exports and what each one's signature is. What
lives here is the *call* side: reading a context pointer out of its slot,
turning offsets into addresses, and issuing the call.

The arguments are typed: a buffer id is an `i32` because the signature says
so, and passing an `i64` where one belongs is a type error at the line that
writes it rather than a `wf` failure about a slot number.

No callee table is threaded. `Ffi` carries the id and the signature, so
`Prog.compileProg` derives the declarations a body needs from the calls it
makes.
-/

namespace AlgorithmLib.Prog

open AlgorithmLib.IR


-- ---------------------------------------------------------------------------
-- Contexts
--
-- Every subsystem keeps its context pointer in a slot of shared memory, written
-- by its `init` and read by everything after. The three functions below are the
-- whole pattern; the wrappers differ only in which slot and which callee.
-- ---------------------------------------------------------------------------

/-- The address of the slot holding a subsystem's context pointer. -/
def ctxSlotPtr (ptr : V .i64) (slotOffset : Nat) : Prog V L (V .i64) :=
  absAddr ptr slotOffset

/-- The context pointer itself. -/
def ctxPtr (ptr : V .i64) (slotOffset : Nat) : Prog V L (V .i64) := do
  load64 (← ctxSlotPtr ptr slotOffset)

/-- `init` takes the *slot* --- it writes the context there. Every subsystem's
    `init` and `cleanup` has this one signature, which the index states. -/
private def initAt (f : Ffi) (hp : f.params = [ClifTy.i64] := by rfl)
    (ptr : V .i64) (slotOffset : Nat) : Prog V L Unit := do
  ffiVoid f (hp ▸ %[← ctxSlotPtr ptr slotOffset])

-- ---------------------------------------------------------------------------
-- wgpu
-- ---------------------------------------------------------------------------

def gpuCtxSlotPtr (ptr : V .i64) (slotOffset : Nat := ContextSlots.wgpu) :
    Prog V L (V .i64) := ctxSlotPtr ptr slotOffset

def gpuCtxPtr (ptr : V .i64) (slotOffset : Nat := ContextSlots.wgpu) :
    Prog V L (V .i64) := ctxPtr ptr slotOffset

def gpuInit (ptr : V .i64) (slotOffset : Nat := ContextSlots.wgpu) : Prog V L Unit :=
  initAt .gpuInit rfl ptr slotOffset

def gpuCreateBuffer (ptr size : V .i64) (slotOffset : Nat := ContextSlots.wgpu) :
    Prog V L (V .i32) := do
  ffi .gpuCreateBuffer %[← gpuCtxPtr ptr slotOffset, size]

def gpuCreatePipeline (ptr shaderOff bindOff : V .i64) (nBindings : V .i32)
    (slotOffset : Nat := ContextSlots.wgpu) : Prog V L (V .i32) := do
  let c ← gpuCtxPtr ptr slotOffset
  let shaderPtr ← iadd ptr shaderOff
  let bindPtr ← iadd ptr bindOff
  ffi .gpuCreatePipeline %[c, shaderPtr, bindPtr, nBindings]

def gpuUpload (ptr : V .i64) (bufId : V .i32) (srcOff size : V .i64)
    (slotOffset : Nat := ContextSlots.wgpu) : Prog V L (V .i32) := do
  let c ← gpuCtxPtr ptr slotOffset
  let srcPtr ← iadd ptr srcOff
  ffi .gpuUpload %[c, bufId, srcPtr, size]

def gpuDownload (ptr : V .i64) (bufId : V .i32) (dstOff size : V .i64)
    (slotOffset : Nat := ContextSlots.wgpu) : Prog V L (V .i32) := do
  let c ← gpuCtxPtr ptr slotOffset
  let dstPtr ← iadd ptr dstOff
  ffi .gpuDownload %[c, bufId, dstPtr, size]

def gpuDispatch (ptr : V .i64) (pipelineId wgX wgY wgZ : V .i32)
    (slotOffset : Nat := ContextSlots.wgpu) : Prog V L (V .i32) := do
  ffi .gpuDispatch %[← gpuCtxPtr ptr slotOffset, pipelineId, wgX, wgY, wgZ]

def gpuCleanup (ptr : V .i64) (slotOffset : Nat := ContextSlots.wgpu) : Prog V L Unit :=
  initAt .gpuCleanup rfl ptr slotOffset

-- ---------------------------------------------------------------------------
-- Hash table
--
-- The same shape as wgpu: `init` and `cleanup` take the *slot*, everything
-- else the context pointer read out of it. The offset is a parameter because
-- generators do not agree on one -- `WordCountAlgorithm` keeps it at 0 and
-- `HProgCorpus` at 0x80 -- and `ContextSlots.ht` is only the default.
-- ---------------------------------------------------------------------------

def htCtxSlotPtr (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) :
    Prog V L (V .i64) := ctxSlotPtr ptr slotOffset

def htCtxPtr (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) :
    Prog V L (V .i64) := ctxPtr ptr slotOffset

def htInit (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) : Prog V L Unit :=
  initAt .htInit rfl ptr slotOffset

def htCleanup (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) : Prog V L Unit :=
  initAt .htCleanup rfl ptr slotOffset

/-- Allocate the table itself. The context must already hold one. -/
def htCreate (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) :
    Prog V L (V .i32) := do
  ffi .htCreate %[← htCtxPtr ptr slotOffset]

/-- Look `key` up and write the value to `resultOff`; the result says whether
    it was found. Both offsets are relative to the base pointer. -/
def htLookup (ptr keyOff : V .i64) (keyLen : V .i32) (resultOff : V .i64)
    (slotOffset : Nat := ContextSlots.ht) : Prog V L (V .i32) := do
  let c ← htCtxPtr ptr slotOffset
  ffi .htLookup %[c, ← iadd ptr keyOff, keyLen, ← iadd ptr resultOff]

def htInsert (ptr keyOff : V .i64) (keyLen : V .i32) (valOff : V .i64)
    (valLen : V .i32) (slotOffset : Nat := ContextSlots.ht) : Prog V L Unit := do
  let c ← htCtxPtr ptr slotOffset
  ffiVoid .htInsert %[c, ← iadd ptr keyOff, keyLen, ← iadd ptr valOff, valLen]

/-- Add `addend` to `key`'s value, inserting it if absent; the result is the
    value after the addition. -/
def htIncrement (ptr keyOff : V .i64) (keyLen : V .i32) (addend : V .i64)
    (slotOffset : Nat := ContextSlots.ht) : Prog V L (V .i64) := do
  let c ← htCtxPtr ptr slotOffset
  ffi .htIncrement %[c, ← iadd ptr keyOff, keyLen, addend]

def htCount (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) :
    Prog V L (V .i32) := do
  ffi .htCount %[← htCtxPtr ptr slotOffset]

/-- The `index`th entry, written to `keyOutOff` and `valOutOff`. Iterating to
    `htCount` is how a program reads the table back out. -/
def htGetEntry (ptr : V .i64) (index : V .i32) (keyOutOff valOutOff : V .i64)
    (slotOffset : Nat := ContextSlots.ht) : Prog V L (V .i32) := do
  let c ← htCtxPtr ptr slotOffset
  ffi .htGetEntry %[c, index, ← iadd ptr keyOutOff, ← iadd ptr valOutOff]

-- ---------------------------------------------------------------------------
-- Files
-- ---------------------------------------------------------------------------

/-- Read a file into shared memory; the result is the byte count. -/
def readFile (ptr : V .i64) (filenameOff dataOff : Nat) : Prog V L (V .i64) := do
  let fnOff ← iconst64 filenameOff
  let dOff ← iconst64 dataOff
  let zero ← iconst64 0
  ffi .fileRead %[ptr, fnOff, dOff, zero, zero]

/-- Write a region of shared memory to a file; the result is the byte count. -/
def writeFile (ptr : V .i64) (filenameOff srcOff : Nat) (fileOffset size : V .i64) :
    Prog V L (V .i64) := do
  let fnOff ← iconst64 filenameOff
  let sOff ← iconst64 srcOff
  ffi .fileWrite %[ptr, fnOff, sOff, fileOffset, size]

/-- Write starting at file offset 0. -/
def writeFile0 (ptr : V .i64) (filenameOff srcOff : Nat) (size : V .i64) :
    Prog V L (V .i64) := do
  let zero ← iconst64 0
  writeFile ptr filenameOff srcOff zero size

/-- Read a line from standard input into shared memory. -/
def stdinReadline (ptr : V .i64) (dstOff maxLen : V .i64) : Prog V L (V .i64) := do
  ffi .stdinReadline %[ptr, dstOff, maxLen]

/-- `(x + 3) &&& ~3` --- wgpu's `COPY_BUFFER_ALIGNMENT`. -/
def alignUp4 (v : V .i64) : Prog V L (V .i64) := do
  let c3 ← iconst64 3
  let sum ← iadd v c3
  let negFour ← iconst64 (-4)
  band sum negFour

-- ---------------------------------------------------------------------------
-- Typed fields
--
-- A `Layout.Fld` names a byte range in shared memory. The scalar forms reject
-- `.bytes n` through `IsScalar`; the `At` forms take a position inside a byte
-- region and a proof that the access fits, discharged by `omega` at the call.
-- ---------------------------------------------------------------------------

/-- A field's offset as a constant. -/
def fldOffset {t} (f : Layout.Fld t) : Prog V L (V .i64) := iconst64 f.offset

/-- A field's address: `base + offset`. -/
def fldAddr {t} (base : V .i64) (f : Layout.Fld t) : Prog V L (V .i64) :=
  absAddr base f.offset

/-- How a scalar field is read and written, chosen per field type rather than by
    a match, so `.bytes n` has no instance and cannot be passed.

    The stored value is an `i64` in every case: `u8` and `i32` fields are
    narrowed by the store and widened by the load, which is what the emitted
    `istore8`/`uload32` pair already did. -/
class IsScalarV (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type)
    (t : Layout.FieldTy) where
  scalarStore : V .i64 → V .i64 → Prog V L Unit
  scalarLoad  : V .i64 → Prog V L (V .i64)

instance : IsScalarV V L .u8 where
  scalarStore val addr := istore8 val addr
  scalarLoad  addr     := uload8_64 addr

instance : IsScalarV V L .i32 where
  scalarStore val addr := storeUnaligned val addr
  scalarLoad  addr     := uload32_64 addr

instance : IsScalarV V L .i64 where
  scalarStore val addr := storeUnaligned val addr
  scalarLoad  addr     := load64 addr

def fldStore {t} (base : V .i64) (f : Layout.Fld t) [inst : IsScalarV V L t]
    (val : V .i64) : Prog V L Unit := do
  inst.scalarStore val (← fldAddr base f)

def fldLoad {t} (base : V .i64) (f : Layout.Fld t) [inst : IsScalarV V L t] :
    Prog V L (V .i64) := do
  inst.scalarLoad (← fldAddr base f)

def fldStoreAt {n ty} (base : V .i64) (f : Layout.Fld (.bytes n)) (i : Nat)
    (val : V ty) (_h : i + 8 ≤ n := by omega) : Prog V L Unit := do
  storeUnaligned val (← absAddr base (f.offset + i))

def fldStore32At {n ty} (base : V .i64) (f : Layout.Fld (.bytes n)) (i : Nat)
    (val : V ty) (_h : i + 4 ≤ n := by omega) : Prog V L Unit := do
  storeUnaligned val (← absAddr base (f.offset + i))

def fldLoadAt {n} (base : V .i64) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 8 ≤ n := by omega) : Prog V L (V .i64) := do
  load64 (← absAddr base (f.offset + i))

def fldLoad8At {n} (base : V .i64) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 1 ≤ n := by omega) : Prog V L (V .i64) := do
  uload8_64 (← absAddr base (f.offset + i))

def fldLoad32At {n} (base : V .i64) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 4 ≤ n := by omega) : Prog V L (V .i64) := do
  uload32_64 (← absAddr base (f.offset + i))

/-- Read a file using typed field handles for the filename and data regions. -/
def fldReadFile {ft dt} (ptr : V .i64) (filenameFld : Layout.Fld ft)
    (dataFld : Layout.Fld dt) : Prog V L (V .i64) :=
  readFile ptr filenameFld.offset dataFld.offset

/-- Write a whole field to a file, from offset 0. -/
def fldWriteFile0 {ft st} (ptr : V .i64) (filenameFld : Layout.Fld ft)
    (srcFld : Layout.Fld st) (size : V .i64) : Prog V L (V .i64) :=
  writeFile0 ptr filenameFld.offset srcFld.offset size

-- ---------------------------------------------------------------------------
-- Window
-- ---------------------------------------------------------------------------

def windowCtxSlotPtr (ptr : V .i64) (slotOffset : Nat := ContextSlots.window) :
    Prog V L (V .i64) := ctxSlotPtr ptr slotOffset

def windowCtxPtr (ptr : V .i64) (slotOffset : Nat := ContextSlots.window) :
    Prog V L (V .i64) := ctxPtr ptr slotOffset

def windowInit (ptr : V .i64) (slotOffset : Nat := ContextSlots.window) :
    Prog V L Unit := initAt .windowInit rfl ptr slotOffset

def windowOpen (ptr width height titleOff titleLen blitOff blitLen : V .i64)
    (slotOffset : Nat := ContextSlots.window) : Prog V L (V .i32) := do
  let c ← windowCtxPtr ptr slotOffset
  let titlePtr ← iadd ptr titleOff
  let blitPtr ← iadd ptr blitOff
  ffi .windowOpen %[c, width, height, titlePtr, titleLen, blitPtr, blitLen]

def windowPoll (ptr eventsOff : V .i64) (maxEvents : V .i32)
    (slotOffset : Nat := ContextSlots.window) : Prog V L (V .i32) := do
  let c ← windowCtxPtr ptr slotOffset
  ffi .windowPoll %[c, ← iadd ptr eventsOff, maxEvents]

/-- Blit a wgpu storage buffer to the swapchain, so it takes both contexts. -/
def windowPresentGpuBuffer (ptr : V .i64) (bufId : V .i32)
    (slotOffset : Nat := ContextSlots.window)
    (gpuSlotOffset : Nat := ContextSlots.wgpu) : Prog V L (V .i32) := do
  let c ← windowCtxPtr ptr slotOffset
  let g ← gpuCtxPtr ptr gpuSlotOffset
  ffi .windowPresentGpuBuffer %[c, g, bufId]

def windowCleanup (ptr : V .i64) (slotOffset : Nat := ContextSlots.window) :
    Prog V L Unit := initAt .windowCleanup rfl ptr slotOffset

-- ---------------------------------------------------------------------------
-- CUDA
--
-- `Raw` names a host pointer the caller owns; the plain form takes an offset
-- into shared memory and turns it into one. `Offset` names a position *inside*
-- the device buffer, so the host pointee need not be the whole allocation.
-- ---------------------------------------------------------------------------

def cudaCtxSlotPtr (ptr : V .i64) (slotOffset : Nat := ContextSlots.cuda) :
    Prog V L (V .i64) := ctxSlotPtr ptr slotOffset

def cudaCtxPtr (ptr : V .i64) (slotOffset : Nat := ContextSlots.cuda) :
    Prog V L (V .i64) := ctxPtr ptr slotOffset

def cudaInit (ptr : V .i64) (slotOffset : Nat := ContextSlots.cuda) : Prog V L Unit :=
  initAt .cudaInit rfl ptr slotOffset

def cudaCreateBuffer (ptr size : V .i64) (slotOffset : Nat := ContextSlots.cuda) :
    Prog V L (V .i32) := do
  ffi .cudaCreateBuffer %[← cudaCtxPtr ptr slotOffset, size]

def cudaUpload (ptr : V .i64) (bufId : V .i32) (srcOff size : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  ffi .cudaUpload %[c, bufId, ← iadd ptr srcOff, size]

def cudaUploadRaw (ptr : V .i64) (bufId : V .i32) (srcPtr size : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  ffi .cudaUpload %[← cudaCtxPtr ptr slotOffset, bufId, srcPtr, size]

def cudaUploadRawOffset (ptr : V .i64) (bufId : V .i32) (bufOff srcPtr size : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  ffi .cudaUploadOffset %[← cudaCtxPtr ptr slotOffset, bufId, bufOff, srcPtr, size]

def cudaUploadOffset (ptr : V .i64) (bufId : V .i32) (bufOff srcOff size : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  ffi .cudaUploadOffset %[c, bufId, bufOff, ← iadd ptr srcOff, size]

def cudaUploadAsync (ptr : V .i64) (bufId : V .i32) (srcOff size : V .i64)
    (streamId : V .i32) (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  ffi .cudaUploadAsync %[c, bufId, ← iadd ptr srcOff, size, streamId]

def cudaUploadOffsetAsync (ptr : V .i64) (bufId : V .i32) (bufOff srcOff size : V .i64)
    (streamId : V .i32) (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  ffi .cudaUploadOffsetAsync %[c, bufId, bufOff, ← iadd ptr srcOff, size, streamId]

def cudaDownload (ptr : V .i64) (bufId : V .i32) (dstOff size : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  ffi .cudaDownload %[c, bufId, ← iadd ptr dstOff, size]

def cudaDownloadRaw (ptr : V .i64) (bufId : V .i32) (dstPtr size : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  ffi .cudaDownload %[← cudaCtxPtr ptr slotOffset, bufId, dstPtr, size]

def cudaDownloadRawOffset (ptr : V .i64) (bufId : V .i32) (bufOff dstPtr size : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  ffi .cudaDownloadOffset %[← cudaCtxPtr ptr slotOffset, bufId, bufOff, dstPtr, size]

def cudaDownloadAsync (ptr : V .i64) (bufId : V .i32) (dstOff size : V .i64)
    (streamId : V .i32) (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  ffi .cudaDownloadAsync %[c, bufId, ← iadd ptr dstOff, size, streamId]

/-- The pinned pool's host address at `off`, or `-1` when `off + len` runs past
    the allocation.

    Use this wherever a source offset is computed rather than fixed: the async
    upload checks the device range it writes but takes its source as a bare
    address, so an unchecked offset into a large pool uploads whatever the
    process happens to hold there. -/
def cudaPinnedPtrAt (ptr : V .i64) (pinnedId : V .i32) (off len : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i64) := do
  ffi .cudaPinnedPtrAt %[← cudaCtxPtr ptr slotOffset, pinnedId, off, len]

/-- Free device memory in bytes. What a card has left after weights, caches and
    the driver's own reservations is not a number that can be written down ahead
    of the machine, so sizes that depend on it are read here. -/
def cudaMemInfoFree (ptr : V .i64) (slotOffset : Nat := ContextSlots.cuda) :
    Prog V L (V .i64) := do
  ffi .cudaMemInfoFree %[← cudaCtxPtr ptr slotOffset]

/-- Total device memory in bytes. -/
def cudaMemInfoTotal (ptr : V .i64) (slotOffset : Nat := ContextSlots.cuda) :
    Prog V L (V .i64) := do
  ffi .cudaMemInfoTotal %[← cudaCtxPtr ptr slotOffset]

def cudaSync (ptr : V .i64) (slotOffset : Nat := ContextSlots.cuda) :
    Prog V L (V .i32) := do
  ffi .cudaSync %[← cudaCtxPtr ptr slotOffset]

def cudaFreeBuffer (ptr : V .i64) (bufId : V .i32)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  ffi .cudaFreeBuffer %[← cudaCtxPtr ptr slotOffset, bufId]

def cudaCleanup (ptr : V .i64) (slotOffset : Nat := ContextSlots.cuda) :
    Prog V L Unit := initAt .cudaCleanup rfl ptr slotOffset

def cudaStreamCreate (ptr : V .i64) (slotOffset : Nat := ContextSlots.cuda) :
    Prog V L (V .i32) := do ffi .cudaStreamCreate %[← cudaCtxPtr ptr slotOffset]

def cudaStreamSync (ptr : V .i64) (streamId : V .i32)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  ffi .cudaStreamSync %[← cudaCtxPtr ptr slotOffset, streamId]

def cudaStreamDestroy (ptr : V .i64) (streamId : V .i32)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  ffi .cudaStreamDestroy %[← cudaCtxPtr ptr slotOffset, streamId]

def cudaLaunch (ptr kernelOff : V .i64) (nBufs : V .i32) (bindOff : V .i64)
    (gridX gridY gridZ blockX blockY blockZ : V .i32)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let bindPtr ← iadd ptr bindOff
  ffi .cudaLaunch
    %[c, kernelPtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ]

def cudaLaunchNamed (ptr kernelOff nameOff : V .i64) (nBufs : V .i32) (bindOff : V .i64)
    (gridX gridY gridZ blockX blockY blockZ : V .i32)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let namePtr ← iadd ptr nameOff
  let bindPtr ← iadd ptr bindOff
  ffi .cudaLaunchNamed
    %[c, kernelPtr, namePtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ]

def cudaLaunchOnStream (ptr kernelOff : V .i64) (nBufs : V .i32) (bindOff : V .i64)
    (gridX gridY gridZ blockX blockY blockZ streamId : V .i32)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let bindPtr ← iadd ptr bindOff
  ffi .cudaLaunchOnStream
    %[c, kernelPtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY, blockZ, streamId]

def cudaLaunchNamedOnStream (ptr kernelOff nameOff : V .i64) (nBufs : V .i32)
    (bindOff : V .i64) (gridX gridY gridZ blockX blockY blockZ streamId : V .i32)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  let kernelPtr ← iadd ptr kernelOff
  let namePtr ← iadd ptr nameOff
  let bindPtr ← iadd ptr bindOff
  ffi .cudaLaunchNamedOnStream
    %[c, kernelPtr, namePtr, nBufs, bindPtr, gridX, gridY, gridZ, blockX, blockY,
      blockZ, streamId]

/-- `y ← alpha·op(A)·x + beta·y`. -/
def cublasSgemv (ptr : V .i64)
    (trans m n alphaBits aBuf xBuf betaBits yBuf : V .i32)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  ffi .cublasSgemv %[c, trans, m, n, alphaBits, aBuf, xBuf, betaBits, yBuf]

/-- `C ← alpha·op(A)·op(B) + beta·C`, strided-batched. The trailing offsets and
    leading dimensions default to zero, which the wrapper reads as "no offset,
    default leading dimension". -/
def cublasSgemmStridedBatched (ptr : V .i64)
    (transA transB m n k alphaBits aBuf : V .i32) (strideA : V .i64) (bBuf : V .i32)
    (strideB : V .i64) (betaBits cBuf : V .i32) (strideC : V .i64)
    (batchCount : V .i32)
    (slotOffset : Nat := ContextSlots.cuda)
    (offA offB offC : Nat := 0) (ldA ldB ldC : Nat := 0) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  let oa ← iconst64 offA
  let ob ← iconst64 offB
  let oc ← iconst64 offC
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  ffi .cublasSgemm
    %[c, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB,
      betaBits, cBuf, strideC, batchCount, oa, ob, oc, la, lb, lc]

/-- The same contraction, issued on a created stream so a capture records it.
    cuBLAS is stream-bound through its handle, so the FFI keeps one handle per
    stream rather than retargeting the default. -/
def cublasSgemmStridedBatchedOnStream (ptr : V .i64)
    (transA transB m n k alphaBits aBuf : V .i32) (strideA : V .i64) (bBuf : V .i32)
    (strideB : V .i64) (betaBits cBuf : V .i32) (strideC : V .i64)
    (batchCount streamId : V .i32)
    (slotOffset : Nat := ContextSlots.cuda)
    (offA offB offC : Nat := 0) (ldA ldB ldC : Nat := 0) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  let oa ← iconst64 offA
  let ob ← iconst64 offB
  let oc ← iconst64 offC
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  ffi .cublasSgemmOnStream
    %[c, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB,
      betaBits, cBuf, strideC, batchCount, streamId, oa, ob, oc, la, lb, lc]

/-- The same contraction with **bf16 operands** and an f32 accumulator and
    result: `C ← alpha·op(A)·op(B) + beta·C`, unbatched.

    Both inputs are bf16 because cuBLAS refuses a mixed pair, so `offA` and
    `offB` count 2-byte elements while `offC` counts 4-byte ones. Offsets and
    leading dimensions default to zero, read as "no offset, default leading
    dimension", as in the strided form. -/
def cublasGemmExBf16 (ptr : V .i64)
    (transA transB m n k alphaBits aBuf bBuf betaBits cBuf : V .i32)
    (slotOffset : Nat := ContextSlots.cuda)
    (offA offB offC : Nat := 0) (ldA ldB ldC : Nat := 0) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  let oa ← iconst64 offA
  let ob ← iconst64 offB
  let oc ← iconst64 offC
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  ffi .cublasGemmExBf16
    %[c, transA, transB, m, n, k, alphaBits, aBuf, bBuf, betaBits, cBuf,
      oa, ob, oc, la, lb, lc]

/-- **The bf16 contraction, once per batch member at a fixed stride.**

    Attention with a bf16 key cache: one member per key head, the query narrowed
    to match because cuBLAS refuses a mixed pair. Strides and offsets count
    elements, and an element is two bytes on both inputs and four on the result
    -- so a stride that is right for the `Float32` form is twice what this one
    wants for its inputs and exactly right for its output. -/
def cublasGemmStridedBatchedExBf16 (ptr : V .i64)
    (transA transB m n k alphaBits aBuf : V .i32) (strideA : V .i64) (bBuf : V .i32)
    (strideB : V .i64) (betaBits cBuf : V .i32) (strideC : V .i64)
    (batchCount : V .i32)
    (slotOffset : Nat := ContextSlots.cuda)
    (offA offB offC : Nat := 0) (ldA ldB ldC : Nat := 0) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  let oa ← iconst64 offA
  let ob ← iconst64 offB
  let oc ← iconst64 offC
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  ffi .cublasGemmStridedBatchedExBf16
    %[c, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB,
      betaBits, cBuf, strideC, batchCount, oa, ob, oc, la, lb, lc]

/-- The same, with the operand offsets held in **registers** rather than fixed
    at emission.

    A sliding window moves with the position, so the offset that names its first
    key is not a number the generator knows. Nothing else changes: an offset
    still moves a pointer and leaves the matrix contracted alone, which is why
    this costs no law the constant form does not already cost. -/
def cublasGemmExBf16At (ptr : V .i64)
    (transA transB m n k alphaBits aBuf bBuf betaBits cBuf : V .i32)
    (offA offB offC : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) (ldA ldB ldC : Nat := 0) :
    Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  ffi .cublasGemmExBf16
    %[c, transA, transB, m, n, k, alphaBits, aBuf, bBuf, betaBits, cBuf,
      offA, offB, offC, la, lb, lc]

/-- The strided-batched contraction with its operand offsets in registers.

    Attention over a sliding window is the reason: the window's first key is a
    function of the position, so `offA`/`offB` are computed at run time. The
    constant-offset form above is the same call with the offsets frozen. -/
def cublasSgemmStridedBatchedOnStreamAt (ptr : V .i64)
    (transA transB m n k alphaBits aBuf : V .i32) (strideA : V .i64) (bBuf : V .i32)
    (strideB : V .i64) (betaBits cBuf : V .i32) (strideC : V .i64)
    (batchCount streamId : V .i32) (offA offB offC : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) (ldA ldB ldC : Nat := 0) :
    Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  let la ← iconst32 ldA; let lb ← iconst32 ldB; let lc ← iconst32 ldC
  ffi .cublasSgemmOnStream
    %[c, transA, transB, m, n, k, alphaBits, aBuf, strideA, bBuf, strideB,
      betaBits, cBuf, strideC, batchCount, streamId, offA, offB, offC, la, lb, lc]

/-- Store `srcBuf`'s device pointer, advanced by `off` f32 elements, into entry
    `slot` of the pointer array held in `arrBuf`. -/
def cublasPtrArray (ptr : V .i64) (arrBuf slot srcBuf : V .i32) (off : V .i64)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  ffi .cublasPtrArray %[← cudaCtxPtr ptr slotOffset, arrBuf, slot, srcBuf, off]

/-- A batch of contractions of one shape whose members are named by pointer,
    so they need not sit at a uniform stride inside one allocation. -/
def cublasSgemmBatchedOnStream (ptr : V .i64)
    (transA transB m n k alphaBits aArr bArr betaBits cArr batchCount streamId : V .i32)
    (slotOffset : Nat := ContextSlots.cuda) : Prog V L (V .i32) := do
  let c ← cudaCtxPtr ptr slotOffset
  ffi .cublasSgemmBatchedOnStream
    %[c, transA, transB, m, n, k, alphaBits, aArr, bArr, betaBits, cArr,
      batchCount, streamId]

-- ---------------------------------------------------------------------------
-- Calling this program's own functions
-- ---------------------------------------------------------------------------

/-- A function that calls each of `callees` in order.

    Composes stages without the caller having to build the call sequence
    itself; the callees are named by `u0:N`, so nothing here resolves a symbol.
    The fold gives each its place in the table the first time it is called, so
    calling one twice declares it once. Each stage is an entry point, so each
    is handed exactly what the wrapper was. -/
def sequenceWrapper (callees : List Nat) : Prog V L Unit := do
  let args ← entryArgs
  for c in callees do
    callLocalVoid (ps := HProg.ptrParams) (res := none) { callee := .local c } args

-- ---------------------------------------------------------------------------
-- Native code
--
-- Machine code the program carries as data, the way it carries a PTX kernel:
-- placed by `nativeLoad`, run by `nativeCall`. `HProgSem` has no definition for
-- any of these, so a proof about a program that calls one does not reach past
-- the call; the CLIF path it would otherwise take is what the code is tested
-- against.
-- ---------------------------------------------------------------------------

/-- Place `len` bytes of machine code at `src` where they can run. Answers the
    address to call, or 0 if they could not be placed. -/
def nativeLoad (src len : V .i64) : Prog V L (V .i64) :=
  ffi .nativeLoad %[src, len]

/-- Unmap what `nativeLoad` answered. Answers 0, or -1 for an address it did
    not answer. -/
def nativeFree (fn : V .i64) : Prog V L (V .i32) :=
  ffi .nativeFree %[fn]

/-- Run the code at `fn` on four arguments and answer what it returns: an
    indirect call, with nothing of the engine's between. `fn` must be an
    address `nativeLoad` answered and did not answer 0; nothing checks that
    when the call runs, so a generator calls only what it read back from
    where it kept `nativeLoad`'s answer. -/
def nativeCall (fn a b c d : V .i64) : Prog V L (V .i64) :=
  callLocal ({ callee := .native } : LocalRef [.i64, .i64, .i64, .i64, .i64] (some .i64))
    %[fn, a, b, c, d]

/-- 1 on x86-64, 2 on AArch64, 0 on anything else. -/
def nativeArch : Prog V L (V .i32) :=
  ffi .nativeArch %[]

/-- Whether the CPU has the feature named by the NUL-terminated string at
    `name`: 1, 0, or -1 for a name the runtime does not know. -/
def cpuHas (name : V .i64) : Prog V L (V .i32) :=
  ffi .cpuHas %[name]

end AlgorithmLib.Prog
