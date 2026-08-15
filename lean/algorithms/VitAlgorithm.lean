import VitModel
import AlgorithmLib.HProgCuda

/-!
# The functions the ViT artifact ships

The model, its kernels and its memory map are `VitModel`; this is the CLIF that
drives them. Splitting the two keeps the proofs about the model clear of the
generator, so a change to the CLIF surface does not rebuild them.
-/

namespace Vit
open AlgorithmLib AlgorithmLib.ML AlgorithmLib.IR


open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- Every emitted function declares the same externals in the same order, so a
    slot index means the same thing in all of them. -/
def ffiEnv : (IR.CudaSetup × IR.CuBlasSetup) × FnEnv := (Id.run (do
  let c := IR.FFI.std.cuda
  let bl := IR.FFI.std.cublas
  pure (c, bl)), env% [.cuda, .cublas])
def cuda : IR.CudaSetup := ffiEnv.1.1
def blas : IR.CuBlasSetup := ffiEnv.1.2
def env : FnEnv := ffiEnv.2

def vGemmStep (ptr : R) (tA tB m n k a b c : Nat) :
    M Unit := do
  let ta ← iconst32 tA
  let tb ← iconst32 tB
  let vm ← iconst32 m
  let vn ← iconst32 n
  let vk ← iconst32 k
  let alpha ← iconst32 F32_ONE
  let beta ← iconst32 F32_ZERO
  let zero64 ← iconst64 0
  let one32 ← iconst32 1
  let aId ← load32 (← absAddr ptr (vBindOff a))
  let bId ← load32 (← absAddr ptr (vBindOff b))
  let cId ← load32 (← absAddr ptr (vBindOff c))
  let _ ← cublasSgemmStridedBatched blas ptr ta tb vm vn vk alpha aId zero64 bId zero64
            beta cId zero64 one32
  pure ()

/-- The same contraction, issued on a created stream so a capture records it.
    cuBLAS is stream-bound through its handle, so the FFI keeps one handle per
    stream rather than retargeting the default. -/
def vGemmStepOn (ptr : R) (tA tB m n k a b c : Nat) (sid : R) :
    M Unit := do
  let ta ← iconst32 tA
  let tb ← iconst32 tB
  let vm ← iconst32 m
  let vn ← iconst32 n
  let vk ← iconst32 k
  let alpha ← iconst32 F32_ONE
  let beta ← iconst32 F32_ZERO
  let zero64 ← iconst64 0
  let one32 ← iconst32 1
  let aId ← load32 (← absAddr ptr (vBindOff a))
  let bId ← load32 (← absAddr ptr (vBindOff b))
  let cId ← load32 (← absAddr ptr (vBindOff c))
  let _ ← cublasSgemmStridedBatchedOnStream blas ptr ta tb vm vn vk alpha aId zero64
            bId zero64 beta cId zero64 one32 sid
  pure ()

/-- **A batch of contractions, issued as one call.**

    The three pointer arrays name the members; the dimensions are shared, which
    is what `vBatchGroups` selects for.  What this rests on is
    `Law.cublasBatchedIsSomeReassoc` — that member `p` sums `p`'s own products
    in some association — and `VendorKernel.assumes` bills it there.  Not the
    closed form: a batch is measurably not bit-equal to the calls it replaces. -/
def vGemmBatchOn (ptr : R) (tA tB m n k : Nat) (pi cnt : Nat)
    (sid : R) : M Unit := do
  let ta ← iconst32 tA
  let tb ← iconst32 tB
  let vm ← iconst32 m
  let vn ← iconst32 n
  let vk ← iconst32 k
  let alpha ← iconst32 F32_ONE
  let beta ← iconst32 F32_ZERO
  let nb ← iconst32 cnt
  let aArr ← load32 (← absAddr ptr (vParrOff (3 * pi)))
  let bArr ← load32 (← absAddr ptr (vParrOff (3 * pi + 1)))
  let cArr ← load32 (← absAddr ptr (vParrOff (3 * pi + 2)))
  let _ ← cublasSgemmBatchedOnStream blas ptr ta tb vm vn vk alpha aArr bArr beta
            cArr nb sid
  pure ()

/-- One launch's contraction work: a single call, or a batch of them. -/
def vGemmUnitOn (ptr : R) (k : Nat) (sid : R) :
    M Unit := do
  let ops := vUnitOps (vUnitArr.getD k [])
  match ops.head? >>= vGemmOf with
  | none => pure ()
  | some (tA, tB, m, n, kk, a, b, c) =>
    if vBatchOf k ≥ 2 then
      vGemmBatchOn ptr tA tB m n kk (vParrIx k) (vBatchOf k) sid
    else
      vGemmStepOn ptr tA tB m n kk a b c sid


def vLoadFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit cuda ptr
  let ctxPtr ← cudaCtxPtr ptr
  for (i, nb) in (List.range VNBUF).zip vBufBytes do
    let sz ← iconst64 nb
    let id ← cudaCreateBuffer cuda ptr sz
    store id (← absAddr ptr (vBindOff i))
  for i in List.range VBASE do
    -- The mask is the artifact's, so its source is this program's own memory
    -- rather than the host blob.
    let src ← if i == VMASK_BUF then absAddr ptr VMASK_OFF
              else iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt vHostIn i)
    let id ← load32 (← absAddr ptr (vBindOff i))
    let bytes ← iconst64 (vInBytes.getD i 0)
    let _ ← call cuda.fnUpload.id [ctxPtr, id, src, bytes]
  -- Force the cuBLAS handle to exist here, before any kernel launch.  Creating
  -- it lazily mid-sequence is what fails.
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  -- The pointer arrays a batched contraction reads: one per operand per batched
  -- launch, filled once.  A device pointer does not move, so this is the whole
  -- of what a batch costs at run time -- the launch itself reads three buffer
  -- ids and nothing else.
  for (k, pi) in vBatchUnits.zip (List.range vBatchUnits.length) do
    let ops := vUnitOps (vUnitArr.getD k [])
    let cnt := ops.length
    let sz ← iconst64 (8 * cnt)
    for j in List.range 3 do
      let id ← cudaCreateBuffer cuda ptr sz
      store id (← absAddr ptr (vParrOff (3 * pi + j)))
    let zoff ← iconst64 0
    for (op, q) in ops.zip (List.range cnt) do
      match vGemmOf op with
      | none => pure ()
      | some (_, _, _, _, _, a, b, c) =>
        let slot ← iconst32 q
        for (j, r) in (List.range 3).zip [a, b, c] do
          let arr ← load32 (← absAddr ptr (vParrOff (3 * pi + j)))
          let src ← load32 (← absAddr ptr (vBindOff r))
          let _ ← cublasPtrArray blas ptr arr slot src zoff
          pure ()
  -- The stream pool, created once.  Each also gets its cuBLAS handle here:
  -- cuBLAS binds a handle to a stream, and allocating one mid-capture loads.
  for s in List.range VNSTRM do
    let sid ← call cuda.fnStreamCreate.id [ctxPtr]
    store sid (← absAddr ptr (vPoolOff s))
    vGemmStepOn ptr 0 0 1 1 1 0 0 VBASE sid
    let _ ← call cuda.fnStreamSync.id [ctxPtr, sid]
  for e in List.range VNEVENT do
    let ev ← call cuda.fnEventCreate.id [ctxPtr]
    store ev (← absAddr ptr (vEventOff e))

def vBindLocal (ptr : R) (bs : List Buf) : M Unit := do
  for (j, gb) in (List.range bs.length).zip bs do
    let id ← load32 (← absAddr ptr (vBindOff gb))
    store id (← absAddr ptr (VLOCAL_OFF + 4 * j))

/-- The tape's operations issued once, from `lo` up to but not including `hi`.

    `sid` selects the stream: `none` is the default stream, `some s` the created
    one a capture records from.  Capture and run differ only in that argument,
    so what is captured is the sequence that runs. -/
def vIssue (ptr : R) (sid : Option R)
    (lo hi : Nat) : M Unit := do
  for (k, u) in (List.range VNUNIT).zip vUnits do
    if lo ≤ vUnitLo u && vUnitHi u < hi then
      match (vUnitOps u).head? >>= vGemmOf with
      | some _ =>
        match sid with
        -- The eager pass performs a batch's members one at a time.  It runs
        -- before the capture, to force every module load, and the two agree by
        -- what `Law.cublasBatchedIsSomeReassoc` says a batch member sums.
        | none   =>
          for op in vUnitOps u do
            match vGemmOf op with
            | none => pure ()
            | some (tA, tB, m, n, kk, a, b, c) => vGemmStep ptr tA tB m n kk a b c
        | some s => vGemmUnitOn ptr k s
      | none =>
        let bs := vUnitBufs u
        vBindLocal ptr bs
        let ptxOff ← iconst64 (vSlotOff (vSlotIx k))
        let nBufs ← iconst32 bs.length
        let bindBase ← iconst64 VLOCAL_OFF
        let one ← iconst32 1
        let w := vWarpsOf (vUnitGrid u)
        let warp ← iconst32 (32 * w)
        let grid ← iconst32 (vUnitGrid u / w)
        match sid with
        | none   => let _ ← cudaLaunch cuda ptr ptxOff nBufs bindBase grid one one warp one one
                    pure ()
        | some s => let _ ← cudaLaunchOnStream cuda ptr ptxOff nBufs bindBase
                              grid one one warp one one s
                    pure ()

/-- The pooled streams, and the events a fork and a join use. -/
def vPool (ptr : R) : M (List R) :=
  (List.range VNSTRM).mapM (fun s => do load32 (← absAddr ptr (vPoolOff s)))

/-- **Every pooled stream made to follow `src`.**  One event, recorded once and
    waited on by each, so the pool starts from what `src` has already done. -/
def vFork (ptr src : R) (sids : List R) : M Unit := do
  let ctxPtr ← cudaCtxPtr ptr
  let ev ← load32 (← absAddr ptr (vEventOff 0))
  let _ ← call cuda.fnEventRecord.id [ctxPtr, ev, src]
  for sid in sids do
    let _ ← call cuda.fnStreamWaitEvent.id [ctxPtr, sid, ev]
  pure ()

/-- **`src` made to follow every pooled stream.**  One event each, so the range
    is finished on `src` exactly when the last of the pool is. -/
def vJoin (ptr src : R) (sids : List R) : M Unit := do
  let ctxPtr ← cudaCtxPtr ptr
  for (k, sid) in (List.range sids.length).zip sids do
    let ev ← load32 (← absAddr ptr (vEventOff (VNSTRM + k)))
    let _ ← call cuda.fnEventRecord.id [ctxPtr, ev, sid]
    let _ ← call cuda.fnStreamWaitEvent.id [ctxPtr, src, ev]
  pure ()

/-- **The tape issued on the streams `vDag` assigned it, with that schedule's
    events.**

    Each operation waits on the events its predecessors on other streams
    published, launches, and publishes its own if a later operation needs it.
    A predecessor before `lo` needs no edge: the fork already put every pooled
    stream behind everything the capture stream had done.

    The launches themselves are the same launches, from the same slots, with the
    same bindings, as the one-stream issue — only the stream differs. -/
def vIssueDag (ptr : R) (sids : List R)
    (lo hi : Nat) (only : Option Bool := none) : M Unit := do
  let ctxPtr ← cudaCtxPtr ptr
  for (k, u) in (List.range VNUNIT).zip vUnits do
    let isBlas := ((vUnitOps u).head? >>= vGemmOf).isSome
    if lo ≤ vUnitLo u && vUnitHi u < hi && (only.all (fun c => c == isBlas)) then
      match (sids.drop (vDag.strm.getD k 0)).head? with
      | none => pure ()
      | some sv =>
        -- An event whose owner this capture skipped is never recorded, and
        -- waiting on one invalidates the whole capture.  So a single-class
        -- graph keeps only its own class's edges, which is also the right
        -- question to ask of it: how long these launches take on their own.
        for e in (vDag.ewait.getD k []).filter (fun e =>
                   let ow := vUnitArr.getD (vDag.eown.getD e 0) []
                   lo ≤ vUnitLo ow
                     && only.all (fun c => c == ((vUnitOps ow).head? >>= vGemmOf).isSome)) do
          let ev ← load32 (← absAddr ptr (vEventOff (vDagEvent e)))
          let _ ← call cuda.fnStreamWaitEvent.id [ctxPtr, sv, ev]
          pure ()
        match (vUnitOps u).head? >>= vGemmOf with
        | some _ => vGemmUnitOn ptr k sv
        | none =>
          let bs := vUnitBufs u
          vBindLocal ptr bs
          let ptxOff ← iconst64 (vSlotOff (vSlotIx k))
          let nBufs ← iconst32 bs.length
          let bindBase ← iconst64 VLOCAL_OFF
          let one ← iconst32 1
          let w := vWarpsOf (vUnitGrid u)
          let warp ← iconst32 (32 * w)
          let grid ← iconst32 (vUnitGrid u / w)
          let _ ← cudaLaunchOnStream cuda ptr ptxOff nBufs bindBase grid one one warp one one sv
          pure ()
        match vDag.erec.getD k none with
        | none => pure ()
        | some e =>
          let ev ← load32 (← absAddr ptr (vEventOff (vDagEvent e)))
          let _ ← call cuda.fnEventRecord.id [ctxPtr, ev, sv]
          pure ()


/-- **A range captured as the graph its dependences allow**, rather than as the
    chain its listing is.  The fork and join make the pool's work part of the
    capture stream's sequence, so `endCapture` returns a graph whose branches
    are the streams.

    The eager pass first, as every capture here does: a module load is not
    something a stream capture may perform. -/
def vCaptureDagAt (lo hi gOff : Nat) : HProg.Code :=
  HProg.Sur.build do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  vIssue ptr none lo hi
  let _ ← cudaSync cuda ptr
  let sid ← load32 (← absAddr ptr VSTREAM_OFF)
  let sids ← vPool ptr
  let _ ← call cuda.fnStreamSync.id [ctxPtr, sid]
  let _ ← call cuda.fnGraphBeginCapture.id [ctxPtr, sid]
  vFork ptr sid sids
  vIssueDag ptr sids lo hi
  vJoin ptr sid sids
  let gid ← call cuda.fnGraphEndCapture.id [ctxPtr, sid]
  store gid (← absAddr ptr gOff)
  let _ ← call cuda.fnGraphUpload.id [ctxPtr, gid, sid]
  let _ ← call cuda.fnStreamSync.id [ctxPtr, sid]



/-- **One class of launch, captured on its own.**

    The graph records only the contractions, or only the proven kernels.  What
    it computes is meaningless — the other half never ran — but what it *takes*
    is the real thing: the same launches, the same shapes, the same streams and
    the same dependence edges among them.

    It exists because there is no working profiler here — `ncu` needs GPU counter
    permissions this user does not have, and `nsys`'s importer is absent — and
    the split between vendor and proven time is the one number that decides
    where the remaining gap is. -/
def vCaptureClassAt (isBlas : Bool) (gOff : Nat) : HProg.Code :=
  HProg.Sur.build do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  vIssue ptr none 0 VSTEP_N
  let _ ← cudaSync cuda ptr
  let sid ← load32 (← absAddr ptr VSTREAM_OFF)
  let sids ← vPool ptr
  let _ ← call cuda.fnStreamSync.id [ctxPtr, sid]
  let _ ← call cuda.fnGraphBeginCapture.id [ctxPtr, sid]
  vFork ptr sid sids
  vIssueDag ptr sids 0 VSTEP_N (some isBlas)
  vJoin ptr sid sids
  let gid ← call cuda.fnGraphEndCapture.id [ctxPtr, sid]
  store gid (← absAddr ptr gOff)
  let _ ← call cuda.fnGraphUpload.id [ctxPtr, gid, sid]
  let _ ← call cuda.fnStreamSync.id [ctxPtr, sid]

def vRunFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  vIssue ptr none 0 VFWD_N
  let _ ← cudaSync cuda ptr



/-- A range of the tape, issued from the host and synced. -/
def vRangeFn (lo hi : Nat) : HProg.Code :=
  HProg.Sur.build do
  let ptr := basePtr
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  vIssue ptr none lo hi
  let _ ← cudaSync cuda ptr

/-- **A prefix of the tape, captured once as a graph.**

    The sequence runs twice.  The first pass is on the default stream, so every
    kernel's module is resident: a module load is not something a stream capture
    may do, and a cold cache under capture fails rather than records.  The
    second pass issues the same launches on a created stream between
    `beginCapture` and `endCapture`, which records them.

    Hundreds of launches become one driver call.  What replay performs is a
    declared fact about the driver, in the same standing as the rest of
    `VendorKernel.cudaGraphLaunch`'s withholding.

    The stream is created once and reused, so the forward capture and the step
    capture replay on the same stream and never interleave. -/
def vCaptureAt (lo hi gOff : Nat) : HProg.Code :=
  HProg.Sur.build do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  vIssue ptr none lo hi
  let _ ← cudaSync cuda ptr
  let sid ← call cuda.fnStreamCreate.id [ctxPtr]
  store sid (← absAddr ptr VSTREAM_OFF)
  -- The stream's own cuBLAS handle, created before capture opens: allocating
  -- one mid-capture is a load, and would fail the same way a cold module does.
  vGemmStepOn ptr 0 0 1 1 1 0 0 VBASE sid
  let _ ← call cuda.fnStreamSync.id [ctxPtr, sid]
  let _ ← call cuda.fnGraphBeginCapture.id [ctxPtr, sid]
  vIssue ptr (some sid) lo hi
  let gid ← call cuda.fnGraphEndCapture.id [ctxPtr, sid]
  store gid (← absAddr ptr gOff)
  let _ ← call cuda.fnGraphUpload.id [ctxPtr, gid, sid]
  let _ ← call cuda.fnStreamSync.id [ctxPtr, sid]

/-- **A captured sequence, replayed `k` times.**  One driver call per pass
    instead of hundreds.  The bind array is untouched: a graph holds the
    parameters it was captured with, which is why the training step can replay
    at all — every buffer it touches is the same one every step. -/
def vReplayAt (gOff k : Nat) : HProg.Code :=
  HProg.Sur.build do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let sid ← load32 (← absAddr ptr VSTREAM_OFF)
  let gid ← load32 (← absAddr ptr gOff)
  for _ in List.range k do
    let _ ← call cuda.fnGraphLaunch.id [ctxPtr, gid, sid]
  let _ ← call cuda.fnStreamSync.id [ctxPtr, sid]

/-- Re-upload the parameters into the buffers that already hold them.

    The updates are in place, so a training run leaves the weights it changed;
    this puts them back without recreating a buffer.  Recreating is not an
    option once a graph has been captured — a capture holds the addresses it
    recorded, and new buffers would not be them. -/
def vReloadFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  for i in List.range VBASE do
    -- The mask is the artifact's, so its source is this program's own memory
    -- rather than the host blob.
    let src ← if i == VMASK_BUF then absAddr ptr VMASK_OFF
              else iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt vHostIn i)
    let id ← load32 (← absAddr ptr (vBindOff i))
    let bytes ← iconst64 (vInBytes.getD i 0)
    let _ ← call cuda.fnUpload.id [ctxPtr, id, src, bytes]

/-- Upload `dL/dlogits` into the buffer the backward is seeded from. -/
def vSeedFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let id ← load32 (← absAddr ptr (vBindOff VSEED))
  let bytes ← iconst64 (SQ * NC * 4)
  let _ ← call cuda.fnUpload.id [ctxPtr, id, dataPtr, bytes]

/-- Download any buffer, named at run time: the input carries the buffer index
    and the byte count.  A gradient check reads a few hundred buffers, and one
    emitted function per buffer would be a CLIF function per parameter. -/
def vFetchAnyFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let outPtr ← load64 (← absAddr ptr 0x28)
  let idx ← load32 dataPtr
  let nb ← load32 (← iaddImm dataPtr 4)
  let base ← absAddr ptr VBIND_OFF
  let off ← ishlImm (← uextend64 idx) 2
  let id ← load32 (← iadd base off)
  let _ ← call cuda.fnDownload.id [ctxPtr, id, outPtr, (← uextend64 nb)]

def vFetchFn (b n : Nat) : HProg.Code :=
  HProg.Sur.build do
  let ptr := basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let id ← load32 (← absAddr ptr (vBindOff b))
  let bytes ← iconst64 n
  let _ ← call cuda.fnDownload.id [ctxPtr, id, outPtr, bytes]

/-- Per-buffer fetches, for bisecting a mismatch.  Only at the geometries small
    enough that one function per buffer is worth emitting. -/
def VDBG : Nat := if NL <= 2 then VNBUF else 0

/-- The entries after the debug fetches, so the CLIF function indices stay
    contiguous at either geometry. -/
def VFN : Nat := 4 + VDBG

end Vit

