import Vit.Model
import AlgorithmLib.Surface.ProgCuda

/-!
# The functions the ViT artifact ships

The model, its kernels and its memory map are `VitModel`; this is the CLIF that
drives them. Splitting the two keeps the proofs about the model clear of the
generator, so a change to the CLIF surface does not rebuild them.
-/

namespace Vit
open AlgorithmLib AlgorithmLib.ML AlgorithmLib.IR


open AlgorithmLib.Prog

def vGemmStep (ptr : V .i64) (tA tB m n k a b c : Nat) :
    Prog V L Unit := do
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
  let _ ← cublasSgemmStridedBatched ptr ta tb vm vn vk alpha aId zero64 bId zero64
            beta cId zero64 one32
  pure ()

/-- The same contraction, issued on a created stream so a capture records it.
    cuBLAS is stream-bound through its handle, so the FFI keeps one handle per
    stream rather than retargeting the default. -/
def vGemmStepOn (ptr : V .i64) (tA tB m n k a b c : Nat) (sid : V .i32) :
    Prog V L Unit := do
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
  let _ ← cublasSgemmStridedBatchedOnStream ptr ta tb vm vn vk alpha aId zero64
            bId zero64 beta cId zero64 one32 sid
  pure ()

/-- **A batch of contractions, issued as one call.**

    The three pointer arrays name the members; the dimensions are shared, which
    is what `vBatchGroups` selects for.  What this rests on is
    `Law.cublasBatchedIsSomeReassoc` — that member `p` sums `p`'s own products
    in some association — and `VendorKernel.assumes` bills it there.  Not the
    closed form: a batch is measurably not bit-equal to the calls it replaces. -/
def vGemmBatchOn (ptr : V .i64) (tA tB m n k : Nat) (pi cnt : Nat)
    (sid : V .i32) : Prog V L Unit := do
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
  let _ ← cublasSgemmBatchedOnStream ptr ta tb vm vn vk alpha aArr bArr beta
            cArr nb sid
  pure ()

/-- One launch's contraction work: a single call, or a batch of them. -/
def vGemmUnitOn (ptr : V .i64) (k : Nat) (sid : V .i32) :
    Prog V L Unit := do
  let ops := vUnitOps (vUnitArr.getD k [])
  match ops.head? >>= vGemmOf with
  | none => pure ()
  | some (tA, tB, m, n, kk, a, b, c) =>
    if vBatchOf k ≥ 2 then
      vGemmBatchOn ptr tA tB m n kk (vParrIx k) (vBatchOf k) sid
    else
      vGemmStepOn ptr tA tB m n kk a b c sid


def vLoadFn : Prog V L Unit :=
  do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  cudaInit ptr
  let ctxPtr ← cudaCtxPtr ptr
  for (i, nb) in (List.range VNBUF).zip vBufBytes do
    let sz ← iconst64 nb
    let id ← cudaCreateBuffer ptr sz
    store id (← absAddr ptr (vBindOff i))
  for i in List.range VBASE do
    -- The mask is the artifact's, so its source is this program's own memory
    -- rather than the host blob.
    let src ← if i == VMASK_BUF then absAddr ptr VMASK_OFF
              else iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt vHostIn i)
    let id ← load32 (← absAddr ptr (vBindOff i))
    let bytes ← iconst64 (vInBytes.getD i 0)
    let _ ← ffi .cudaUpload %[ctxPtr, id, src, bytes]
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
      let id ← cudaCreateBuffer ptr sz
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
          let _ ← cublasPtrArray ptr arr slot src zoff
          pure ()
  -- The stream pool, created once.  Each also gets its cuBLAS handle here:
  -- cuBLAS binds a handle to a stream, and allocating one mid-capture loads.
  for s in List.range VNSTRM do
    let sid ← ffi .cudaStreamCreate %[ctxPtr]
    store sid (← absAddr ptr (vPoolOff s))
    vGemmStepOn ptr 0 0 1 1 1 0 0 VBASE sid
    let _ ← ffi .cudaStreamSync %[ctxPtr, sid]
  for e in List.range VNEVENT do
    let ev ← ffi .cudaEventCreate %[ctxPtr]
    store ev (← absAddr ptr (vEventOff e))

def vBindLocal (ptr : V .i64) (bs : List Buf) : Prog V L Unit := do
  for (j, gb) in (List.range bs.length).zip bs do
    let id ← load32 (← absAddr ptr (vBindOff gb))
    store id (← absAddr ptr (VLOCAL_OFF + 4 * j))

/-- The tape's operations issued once, from `lo` up to but not including `hi`.

    `sid` selects the stream: `none` is the default stream, `some s` the created
    one a capture records from.  Capture and run differ only in that argument,
    so what is captured is the sequence that runs. -/
def vIssue (ptr : V .i64) (sid : Option (V .i32))
    (lo hi : Nat) : Prog V L Unit := do
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
        | none   => let _ ← cudaLaunch ptr ptxOff nBufs bindBase grid one one warp one one
                    pure ()
        | some s => let _ ← cudaLaunchOnStream ptr ptxOff nBufs bindBase
                              grid one one warp one one s
                    pure ()

/-- The pooled streams, and the events a fork and a join use. -/
def vPool (ptr : V .i64) : Prog V L (List (V .i32)) :=
  (List.range VNSTRM).mapM (fun s => do load32 (← absAddr ptr (vPoolOff s)))

/-- **Every pooled stream made to follow `src`.**  One event, recorded once and
    waited on by each, so the pool starts from what `src` has already done. -/
def vFork (ptr : V .i64) (src : V .i32) (sids : List (V .i32)) : Prog V L Unit := do
  let ctxPtr ← cudaCtxPtr ptr
  let ev ← load32 (← absAddr ptr (vEventOff 0))
  let _ ← ffi .cudaEventRecord %[ctxPtr, ev, src]
  for sid in sids do
    let _ ← ffi .cudaStreamWaitEvent %[ctxPtr, sid, ev]
  pure ()

/-- **`src` made to follow every pooled stream.**  One event each, so the range
    is finished on `src` exactly when the last of the pool is. -/
def vJoin (ptr : V .i64) (src : V .i32) (sids : List (V .i32)) : Prog V L Unit := do
  let ctxPtr ← cudaCtxPtr ptr
  for (k, sid) in (List.range sids.length).zip sids do
    let ev ← load32 (← absAddr ptr (vEventOff (VNSTRM + k)))
    let _ ← ffi .cudaEventRecord %[ctxPtr, ev, sid]
    let _ ← ffi .cudaStreamWaitEvent %[ctxPtr, src, ev]
  pure ()

/-- **The tape issued on the streams `vDag` assigned it, with that schedule's
    events.**

    Each operation waits on the events its predecessors on other streams
    published, launches, and publishes its own if a later operation needs it.
    A predecessor before `lo` needs no edge: the fork already put every pooled
    stream behind everything the capture stream had done.

    The launches themselves are the same launches, from the same slots, with the
    same bindings, as the one-stream issue — only the stream differs. -/
def vIssueDag (ptr : V .i64) (sids : List (V .i32))
    (lo hi : Nat) (only : Option Bool := none) : Prog V L Unit := do
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
          let _ ← ffi .cudaStreamWaitEvent %[ctxPtr, sv, ev]
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
          let _ ← cudaLaunchOnStream ptr ptxOff nBufs bindBase grid one one warp one one sv
          pure ()
        match vDag.erec.getD k none with
        | none => pure ()
        | some e =>
          let ev ← load32 (← absAddr ptr (vEventOff (vDagEvent e)))
          let _ ← ffi .cudaEventRecord %[ctxPtr, ev, sv]
          pure ()


/-- **A range captured as the graph its dependences allow**, rather than as the
    chain its listing is.  The fork and join make the pool's work part of the
    capture stream's sequence, so `endCapture` returns a graph whose branches
    are the streams.

    The eager pass first, as every capture here does: a module load is not
    something a stream capture may perform. -/
def vCaptureDagAt (lo hi gOff : Nat) : Prog V L Unit :=
  do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  vIssue ptr none lo hi
  let _ ← cudaSync ptr
  let sid ← load32 (← absAddr ptr VSTREAM_OFF)
  let sids ← vPool ptr
  let _ ← ffi .cudaStreamSync %[ctxPtr, sid]
  let _ ← ffi .cudaGraphBeginCapture %[ctxPtr, sid]
  vFork ptr sid sids
  vIssueDag ptr sids lo hi
  vJoin ptr sid sids
  let gid ← ffi .cudaGraphEndCapture %[ctxPtr, sid]
  store gid (← absAddr ptr gOff)
  let _ ← ffi .cudaGraphUpload %[ctxPtr, gid, sid]
  let _ ← ffi .cudaStreamSync %[ctxPtr, sid]

/-- **One class of launch, captured on its own.**

    The graph records only the contractions, or only the proven kernels.  What
    it computes is meaningless — the other half never ran — but what it *takes*
    is the real thing: the same launches, the same shapes, the same streams and
    the same dependence edges among them.

    It exists because there is no working profiler here — `ncu` needs GPU counter
    permissions this user does not have, and `nsys`'s importer is absent — and
    the split between vendor and proven time is the one number that decides
    where the remaining gap is. -/
def vCaptureClassAt (isBlas : Bool) (gOff : Nat) : Prog V L Unit :=
  do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  vIssue ptr none 0 VSTEP_N
  let _ ← cudaSync ptr
  let sid ← load32 (← absAddr ptr VSTREAM_OFF)
  let sids ← vPool ptr
  let _ ← ffi .cudaStreamSync %[ctxPtr, sid]
  let _ ← ffi .cudaGraphBeginCapture %[ctxPtr, sid]
  vFork ptr sid sids
  vIssueDag ptr sids 0 VSTEP_N (some isBlas)
  vJoin ptr sid sids
  let gid ← ffi .cudaGraphEndCapture %[ctxPtr, sid]
  store gid (← absAddr ptr gOff)
  let _ ← ffi .cudaGraphUpload %[ctxPtr, gid, sid]
  let _ ← ffi .cudaStreamSync %[ctxPtr, sid]

def vRunFn : Prog V L Unit :=
  do
  let ptr ← basePtr
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  vIssue ptr none 0 VFWD_N
  let _ ← cudaSync ptr

/-- A range of the tape, issued from the host and synced. -/
def vRangeFn (lo hi : Nat) : Prog V L Unit :=
  do
  let ptr ← basePtr
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  vIssue ptr none lo hi
  let _ ← cudaSync ptr

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
def vCaptureAt (lo hi gOff : Nat) : Prog V L Unit :=
  do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  vGemmStep ptr 0 0 1 1 1 0 0 VBASE
  vIssue ptr none lo hi
  let _ ← cudaSync ptr
  let sid ← ffi .cudaStreamCreate %[ctxPtr]
  store sid (← absAddr ptr VSTREAM_OFF)
  -- The stream's own cuBLAS handle, created before capture opens: allocating
  -- one mid-capture is a load, and would fail the same way a cold module does.
  vGemmStepOn ptr 0 0 1 1 1 0 0 VBASE sid
  let _ ← ffi .cudaStreamSync %[ctxPtr, sid]
  let _ ← ffi .cudaGraphBeginCapture %[ctxPtr, sid]
  vIssue ptr (some sid) lo hi
  let gid ← ffi .cudaGraphEndCapture %[ctxPtr, sid]
  store gid (← absAddr ptr gOff)
  let _ ← ffi .cudaGraphUpload %[ctxPtr, gid, sid]
  let _ ← ffi .cudaStreamSync %[ctxPtr, sid]

/-- **A captured sequence, replayed `k` times.**  One driver call per pass
    instead of hundreds.  The bind array is untouched: a graph holds the
    parameters it was captured with, which is why the training step can replay
    at all — every buffer it touches is the same one every step. -/
def vReplayAt (gOff k : Nat) : Prog V L Unit :=
  do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let sid ← load32 (← absAddr ptr VSTREAM_OFF)
  let gid ← load32 (← absAddr ptr gOff)
  for _ in List.range k do
    let _ ← ffi .cudaGraphLaunch %[ctxPtr, gid, sid]
  let _ ← ffi .cudaStreamSync %[ctxPtr, sid]

/-- Re-upload the parameters into the buffers that already hold them.

    The updates are in place, so a training run leaves the weights it changed;
    this puts them back without recreating a buffer.  Recreating is not an
    option once a graph has been captured — a capture holds the addresses it
    recorded, and new buffers would not be them. -/
def vReloadFn : Prog V L Unit :=
  do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← dataPtr
  for i in List.range VBASE do
    -- The mask is the artifact's, so its source is this program's own memory
    -- rather than the host blob.
    let src ← if i == VMASK_BUF then absAddr ptr VMASK_OFF
              else iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt vHostIn i)
    let id ← load32 (← absAddr ptr (vBindOff i))
    let bytes ← iconst64 (vInBytes.getD i 0)
    let _ ← ffi .cudaUpload %[ctxPtr, id, src, bytes]

/-- Upload `dL/dlogits` into the buffer the backward is seeded from. -/
def vSeedFn : Prog V L Unit :=
  do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← dataPtr
  let id ← load32 (← absAddr ptr (vBindOff VSEED))
  let bytes ← iconst64 (SQ * NC * 4)
  let _ ← ffi .cudaUpload %[ctxPtr, id, dataPtr, bytes]

/-- Download any buffer, named at run time: the input carries the buffer index
    and the byte count.  A gradient check reads a few hundred buffers, and one
    emitted function per buffer would be a CLIF function per parameter. -/
def vFetchAnyFn : Prog V L Unit :=
  do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let dataPtr ← dataPtr
  let outPtr ← outPtr
  let idx ← load32 dataPtr
  let nb ← load32 (← iaddImm dataPtr 4)
  let base ← absAddr ptr VBIND_OFF
  let off ← ishlImm (← uextend64 idx) 2
  let id ← load32 (← iadd base off)
  let _ ← ffi .cudaDownload %[ctxPtr, id, outPtr, (← uextend64 nb)]

def vFetchFn (b n : Nat) : Prog V L Unit :=
  do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← outPtr
  let id ← load32 (← absAddr ptr (vBindOff b))
  let bytes ← iconst64 n
  let _ ← ffi .cudaDownload %[ctxPtr, id, outPtr, bytes]

/-- Per-buffer fetches, for bisecting a mismatch.  Only at the geometries small
    enough that one function per buffer is worth emitting. -/
def VDBG : Nat := if NL <= 2 then VNBUF else 0

/-- The entries after the debug fetches, so the CLIF function indices stay
    contiguous at either geometry. -/
def VFN : Nat := 4 + VDBG

end Vit

