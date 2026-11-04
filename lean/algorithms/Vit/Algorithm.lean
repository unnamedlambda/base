module
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import Vit.Model
meta import Vit.Model
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The functions the ViT artifact ships

The model, its kernels and its schedule are `VitModel`; this is the CLIF that
drives them, calling the CUDA driver and cuBLAS directly in the standard
layout (`Drv.App`): every launch that is not a contraction is a kernel of the
layout, handed its group's buffers, and every contraction is a cuBLAS product
on the handle of the stream it runs on.
-/

namespace Vit
open AlgorithmLib AlgorithmLib.ML AlgorithmLib.IR AlgorithmLib.Prog

-- ---------------------------------------------------------------------------
-- The layout
-- ---------------------------------------------------------------------------

/-- Whether launch `k` is a contraction. -/
def vIsGemm (k : Nat) : Bool := ((vUnitOps (vUnitArr.getD k [])).head? >>= vGemmOf).isSome

/-- The launches that are kernels, in launch order: kernel `j` of the layout
    is launch `vKernUnits[j]`. -/
def vKernUnits : List Nat := (List.range VNUNIT).filter (fun k => !vIsGemm k)

/-- Each launch's kernel index, for the launches that are kernels. -/
def vKernIx : Array Nat := Id.run do
  let mut a := Array.replicate VNUNIT 0
  for (k, j) in vKernUnits.zip (List.range vKernUnits.length) do
    a := a.set! k j
  pure a

/-- Launch `k` as a kernel of the layout: its slot's PTX, its group's
    buffers, and `w` warps per block over its chunks. -/
def vKSpec (k : Nat) : Drv.KSpec :=
  let u := vUnitArr.getD k []
  let w := vWarpsOf (vUnitGrid u)
  { ptxOff := vSlotOff (vSlotIx k), ptxLen := vSlotSizes.getD (vSlotIx k) 0, args := vUnitBufs u,
    grid := (vUnitGrid u / w, 1, 1), block := (32 * w, 1, 1) }

/-- Parameters a block holds: the widest group's buffers. -/
def VCELLS : Nat := (vKernUnits.map (fun k => (vUnitBufs (vUnitArr.getD k [])).length)).foldl max 16

def vAppAt (org mem : Nat) : Drv.App :=
  { bufs := vBufBytes, kernels := vKernUnits.map vKSpec, memSize := mem, blas := true, org := org,
    cells := VCELLS }

/-- The layout's state starts past the mask. -/
def VORG : Nat := (VMASK_OFF + 4 * SK + 255) / 256 * 256

/-- Past the layout's state: the stream a capture begins on, the pooled
    streams with a cuBLAS handle each, the events, and each capture's graph and
    instantiated graph. -/
def VX_OFF : Nat := ((vAppAt VORG 0).stateEnd + 7) / 8 * 8
def VSTREAM_OFF : Nat := VX_OFF
def vPoolOff (s : Nat) : Nat := VX_OFF + 8 + 8 * s
def vHandleOff (s : Nat) : Nat := vPoolOff VNSTRM + 8 * s
def vEventOff (e : Nat) : Nat := vHandleOff VNSTRM + 8 * e
def VGRAPH_FWD : Nat := vEventOff VNEVENT
def VEXEC_FWD : Nat := VGRAPH_FWD + 8
def VGRAPH_STEP : Nat := VGRAPH_FWD + 16
def VEXEC_STEP : Nat := VGRAPH_FWD + 24

/-- Each buffer's size in bytes, one `u64` each, for a fetch that names its
    buffer at run time. -/
def VSIZES_OFF : Nat := VGRAPH_FWD + 32

/-- Where each input's gradient landed, one `u32` per input, `0` for the
    constants and the patch embedding that are not trained. It is the last
    region, which is the whole of how a host finds it: `4 * VBASE` bytes ending
    `0x100` before the end of memory. -/
def VGMAP_OFF : Nat := VSIZES_OFF + 8 * VNBUF
def VMEM_SIZE : Nat := VGMAP_OFF + 4 * VBASE + 0x100

def vApp : Drv.App := vAppAt VORG VMEM_SIZE

/-- **Every byte of memory this program names**, so a collision between two
    regions is a build error rather than a silent overwrite. -/
def vMemMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"host_len", VHOST_LEN_OFF, 8⟩,
   ⟨"ptx",      VPTX_OFF, VPTX_BYTES⟩,
   ⟨"mask",     VMASK_OFF, 4 * SK⟩,
   ⟨"driver",   VORG, vApp.stateEnd - VORG⟩,
   ⟨"stream",   VSTREAM_OFF, 8⟩,
   ⟨"pool",     vPoolOff 0, VGRAPH_FWD - vPoolOff 0⟩,
   ⟨"graphs",   VGRAPH_FWD, 32⟩,
   ⟨"sizes",    VSIZES_OFF, 8 * VNBUF⟩,
   ⟨"gmap",     VGMAP_OFF, 4 * VBASE⟩]

#eval LayoutScan.check "Vit.Algorithm" [``vMemMap]
theorem vMemMap_ok :
    vMemMap.okB = true ∧ vMemMap.withinB VMEM_SIZE = true := by native_decide

-- ---------------------------------------------------------------------------
-- Issuing the tape
-- ---------------------------------------------------------------------------

/-- A contraction as a cuBLAS product on handle `h`. cuBLAS is column-major
    and a row-major `(r x c)` is a column-major `(c x r)` of the same bytes, so
    each contraction is one transpose choice; the leading dimensions are the
    packed ones. -/
def vGemm (base h : V .i64) (tA tB m n k a b c : Nat) : Prog V L Unit := do
  let z ← iconst64 0
  let _ ← ext (.cublas .sgemmStridedBatched) %[h, ← iconst32 tA, ← iconst32 tB, ← iconst32 m, ← iconst32 n,
    ← iconst32 k, ← iaddImm base vApp.ONE, ← load64 (← iaddImm base (vApp.bufSlot a)),
    ← iconst32 (if tA = 1 then k else m), z, ← load64 (← iaddImm base (vApp.bufSlot b)),
    ← iconst32 (if tB = 1 then n else k), z, ← iaddImm base vApp.ZERO,
    ← load64 (← iaddImm base (vApp.bufSlot c)), ← iconst32 m, z, ← iconst32 1]

/-- Launch `k` on a stream, its contraction on that stream's handle `h`. -/
def vLaunch (base stream h : V .i64) (k : Nat) : Prog V L Unit :=
  if vIsGemm k then
    match (vUnitOps (vUnitArr.getD k [])).head? >>= vGemmOf with
    | some (tA, tB, m, n, kk, a, b, c) => vGemm base h tA tB m n kk a b c
    | none => pure ()
  else vApp.runOn base stream (vKernIx.getD k 0)

/-- The tape's launches from `lo` up to but not including `hi`, on the
    default stream. -/
def vIssue (base : V .i64) (lo hi : Nat) : Prog V L Unit := do
  for (k, u) in (List.range VNUNIT).zip vUnits do
    if lo ≤ vUnitLo u && vUnitHi u < hi then
      vLaunch base (← iconst64 0) (← load64 (← iaddImm base vApp.BLAS)) k

/-- **Every pooled stream made to follow `src`.** One event, recorded once and
    waited on by each, so the pool starts from what `src` has already done. -/
def vFork (base src : V .i64) : Prog V L Unit := do
  let ev ← load64 (← iaddImm base (vEventOff 0))
  let _ ← ext (.cuda .eventRecord) %[ev, src]
  for s in List.range VNSTRM do
    let _ ← ext (.cuda .streamWaitEvent) %[← load64 (← iaddImm base (vPoolOff s)), ev, ← iconst32 0]

/-- **`src` made to follow every pooled stream.** One event each, so the range
    is finished on `src` exactly when the last of the pool is. -/
def vJoin (base src : V .i64) : Prog V L Unit := do
  for s in List.range VNSTRM do
    let ev ← load64 (← iaddImm base (vEventOff (VNSTRM + s)))
    let _ ← ext (.cuda .eventRecord) %[ev, ← load64 (← iaddImm base (vPoolOff s))]
    let _ ← ext (.cuda .streamWaitEvent) %[src, ev, ← iconst32 0]

/-- **The tape issued on the streams `vStrm` assigned it, with the events of
    the range's own schedule (`vDagIn`).** Each launch waits on the events its
    predecessors on other streams published, launches, and publishes its own if
    a later launch needs it. A predecessor before the range needs no edge: the
    fork already put every pooled stream behind everything the capture stream
    had done. -/
def vIssueDag (base : V .i64) (lo hi : Nat) : Prog V L Unit := do
  let d := vDagIn lo hi
  for (k, u) in (List.range VNUNIT).zip vUnits do
    if lo ≤ vUnitLo u && vUnitHi u < hi then
      let s := vStrm.getD k 0
      let sv ← load64 (← iaddImm base (vPoolOff s))
      for e in d.ewait.getD k [] do
        let _ ← ext (.cuda .streamWaitEvent) %[sv, ← load64 (← iaddImm base (vEventOff (vDagEvent e))),
          ← iconst32 0]
      vLaunch base sv (← load64 (← iaddImm base (vHandleOff s))) k
      match d.erec.getD k none with
      | none => pure ()
      | some e =>
        let _ ← ext (.cuda .eventRecord) %[← load64 (← iaddImm base (vEventOff (vDagEvent e))), sv]

-- ---------------------------------------------------------------------------
-- The entries
-- ---------------------------------------------------------------------------

/-- Copy the inputs from the caller's data, the mask from this program's own
    memory. -/
def vUploadInputs (base data : V .i64) : Prog V L Unit := do
  for i in List.range VBASE do
    let src ← if i == VMASK_BUF then iaddImm base VMASK_OFF
              else iaddImm data (AlgorithmLib.Layout.RegionMap.offAt vHostIn i)
    vApp.upload base i src (vInBytes.getD i 0)

/-- Bring the driver and cuBLAS up, make every buffer, load every kernel and
    write its parameters, upload the inputs, and make the pooled streams,
    each with its own cuBLAS handle, and the events. Each handle runs one
    product before any capture: cuBLAS readies a handle on its first call, and
    readying one is not something a capture may record. -/
def vLoadFn : Prog V L Unit := do
  let base ← basePtr
  let data ← dataPtr
  vApp.setup base
  vUploadInputs base data
  let one ← iconst32 1
  let z32 ← iconst32 0
  vGemm base (← load64 (← iaddImm base vApp.BLAS)) 0 0 1 1 1 0 0 VBASE
  let _ ← ext (.cuda .streamCreate) %[← iaddImm base VSTREAM_OFF, one]
  for s in List.range VNSTRM do
    let _ ← ext (.cuda .streamCreate) %[← iaddImm base (vPoolOff s), one]
    let _ ← ext (.cublas .create) %[← iaddImm base (vHandleOff s)]
    let _ ← ext (.cublas .setStream) %[← load64 (← iaddImm base (vHandleOff s)),
      ← load64 (← iaddImm base (vPoolOff s))]
    vGemm base (← load64 (← iaddImm base (vHandleOff s))) 0 0 1 1 1 0 0 VBASE
  for e in List.range VNEVENT do
    let _ ← ext (.cuda .eventCreate) %[← iaddImm base (vEventOff e), z32]
  Drv.sync

/-- **A range captured as the graph its dependences allow**, rather than as the
    chain its listing is, then instantiated and uploaded. The fork and join
    make the pool's work part of the capture stream's sequence, so
    `endCapture` returns a graph whose branches are the streams.

    The range runs once on the default stream first: a module load is not
    something a capture may perform. -/
def vCaptureAt (lo hi gSlot xSlot : Nat) : Prog V L Unit := do
  let base ← basePtr
  vIssue base lo hi
  Drv.sync
  let s ← load64 (← iaddImm base VSTREAM_OFF)
  let _ ← ext (.cuda .beginCapture) %[s, ← iconst32 2]
  vFork base s
  vIssueDag base lo hi
  vJoin base s
  let _ ← ext (.cuda .endCapture) %[s, ← iaddImm base gSlot]
  let _ ← ext (.cuda .graphInstantiate) %[← iaddImm base xSlot, ← load64 (← iaddImm base gSlot), ← iconst64 0]
  let _ ← ext (.cuda .graphUpload) %[← load64 (← iaddImm base xSlot), s]
  let _ ← ext (.cuda .streamSynchronize) %[s]

/-- **A captured range, replayed `k` times.** One driver call per pass
    instead of hundreds. A graph holds the parameters it was captured with,
    which is why the training step can replay at all: every buffer it touches
    is the same one every step. -/
def vReplayAt (xSlot k : Nat) : Prog V L Unit := do
  let base ← basePtr
  let s ← load64 (← iaddImm base VSTREAM_OFF)
  for _ in List.range k do
    let _ ← ext (.cuda .graphLaunch) %[← load64 (← iaddImm base xSlot), s]
  let _ ← ext (.cuda .streamSynchronize) %[s]

/-- A range of the tape, issued from the host and waited for. -/
def vRangeFn (lo hi : Nat) : Prog V L Unit := do
  let base ← basePtr
  vIssue base lo hi
  Drv.sync

/-- Re-upload the parameters into the buffers that already hold them. The
    updates are in place, so a training run leaves the weights it changed;
    this puts them back without remaking a buffer, which a captured graph
    could not follow.

    A copy from pageable memory returns once its bytes are staged, not once
    they land, and the graphs replay on a stream that does not wait for the
    default one: so each upload here is waited for before the entry returns. -/
def vReloadFn : Prog V L Unit := do
  vUploadInputs (← basePtr) (← dataPtr)
  Drv.sync

/-- Upload `dL/dlogits` into the buffer the backward is seeded from, and wait
    for it to land (see `vReloadFn`). -/
def vSeedFn : Prog V L Unit := do
  vApp.upload (← basePtr) VSEED (← dataPtr) (SQ * NC * 4)
  Drv.sync

/-- Download `n` bytes of buffer `b` into the caller's output. -/
def vFetchFn (b n : Nat) : Prog V L Unit := do
  vApp.download (← basePtr) (← outPtr) b n

/-- Download any buffer, named at run time: the input carries the buffer index
    and the byte count, both `u32`. It reads them only when the caller handed
    over their eight bytes, and downloads only when the index names a buffer
    and the count fits both that buffer, by the sizes table, and the caller's
    output. -/
def vFetchAnyFn : Prog V L Unit := do
  let base ← basePtr
  let data ← dataPtr
  let out ← outPtr
  Prog.when .ule (← iconst64 8) (← dataLen) do
    let idx ← uextend64 (← load32 data)
    let nb ← uextend64 (← load32 (← iaddImm data 4))
    Prog.when .ult idx (← iconst64 VNBUF) do
      Prog.when .ule nb (← outLen) do
        let off ← ishlImm idx 3
        let size ← load64 (← iadd (← iaddImm base VSIZES_OFF) off)
        Prog.when .ule nb size do
          let ptr ← load64 (← iadd (← iaddImm base (vApp.bufSlot 0)) off)
          let _ ← ext (.cuda .memcpyDtoH) %[out, ptr, nb]

end Vit
