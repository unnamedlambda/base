module
public import AlgorithmLib.Host.Contracts
meta import AlgorithmLib.Host.Contracts
public import AlgorithmLib.Host.StaticCong
meta import AlgorithmLib.Host.StaticCong
public import AlgorithmLib.Host.Frames
meta import AlgorithmLib.Host.Frames
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The contracts of the C library functions

One entry per `Ext`: what a call needs (`ExtPre`), the memory it may write
(`Ext.frame`, declared with the other frames), and where each need comes from
(`Ext.sources`). Two theorems make the table true of the model:

* `ext_pre_safe` — under its precondition a call answers: it is not misuse;
* `ext_respects_frame` — an answer changes no host byte outside the frame.

A library that did not load answers every call with `-1` (the probe with `0`)
and changes nothing, so its precondition asks nothing of an absent library.

**Sufficient, not exact.** A precondition is at least as strong as the
library's own rules: where the documentation leaves a call undefined, the
model gives it no answer and the precondition excludes it. Where the model
states less than the library defines, the precondition asks for more than the
library does, and the tag says so (`Source.unmodelled`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.DevSpec AlgorithmLib.HProg.Static AlgorithmLib.HProg.Contracts

namespace AlgorithmLib.HProg.ExtContracts

-- ---------------------------------------------------------------------------
-- Where a need comes from
-- ---------------------------------------------------------------------------

/-- Where one clause of a precondition comes from. -/
inductive Source where
  /-- The vendor states it: without it the call is undefined. This side cannot
      be tested, since undefined behaviour is not observable. -/
  | documented (ref : String)
  /-- A corpus run against the library shows it. -/
  | observed (corpus : String)
  /-- The library defines the call without it, but the model does not state
      that behaviour: the clause narrows the model, not the library. -/
  | unmodelled (why : String)
  deriving Repr

-- ---------------------------------------------------------------------------
-- The vocabulary
-- ---------------------------------------------------------------------------

/-- A four-byte slot the program may write. -/
def Slot4 (m : Mem) (a : UInt64) : Prop := ∀ v, (m.store a 4 v).isSome = true

/-- A C string at `a`: readable up to and including its terminating zero. -/
def Terminated (m : Mem) (a : UInt64) : Prop := (strlenAt m a).isSome = true

/-- `n` bytes at device pointer `p` lie in one live allocation, which the
    default stream may access (`write` for a write) without a race. -/
def DevReady (d : Dev) (p : UInt64) (n : Nat) (write : Bool) : Prop :=
  ∃ id off b, d.range? p n = some (id, off, b) ∧ Ready d.race defaultParty id write

/-- A loaded module's handle. -/
def ModuleLive (d : Dev) (h : UInt64) : Prop :=
  ∃ mi ptx, handleOf kModule h = some mi ∧ d.modules[mi]? = some (some ptx)

/-- A live stream's handle, or `0` for the default stream. -/
def StreamLive (d : Dev) (h : UInt64) : Prop := (d.streamParty? h).isSome = true

/-- What `cuLaunchKernel` needs of its function, stream and parameters: the
    function comes from a loaded module, the stream is live, every parameter
    pointer and the value it points at can be read, and each allocation a
    parameter points into is ready for the stream to write. -/
def LaunchOk (w : World) (hf hstream params : UInt64) : Prop :=
  ∃ fi mi entry ptx p sizes args,
    handleOf kFunc hf = some fi ∧ w.dev.funcs[fi]? = some (mi, entry) ∧
    w.dev.modules[mi]? = some (some ptx) ∧ w.dev.streamParty? hstream = some p ∧
    ptxParamSizes ptx entry = some sizes ∧ kernelArgs w.mem params sizes = some args ∧
    ∀ a ∈ args, ∀ id off b, w.dev.range? a 0 = some (id, off, b) → Ready (w.dev.raceFor p) p id true

/-- A live event's handle. -/
def EventLive (d : Dev) (h : UInt64) : Prop := (d.eventOf? h).isSome = true

/-- A live event's handle whose last recorded point, if any, is outside a
    capture. -/
def EventPlain (d : Dev) (h : UInt64) : Prop :=
  ∃ id ev, d.eventOf? h = some (id, ev) ∧ ∀ c, ev ≠ some (c, true)

/-- Two events recorded outside a capture that the host has waited for. -/
def ElapsedReady (d : Dev) (hs he : UInt64) : Prop :=
  ∃ si ei cs ce, d.eventOf? hs = some (si, some (cs, false)) ∧
    d.eventOf? he = some (ei, some (ce, false)) ∧
    d.race.hostSaw cs = true ∧ d.race.hostSaw ce = true

/-- `n` bytes at device pointer `p` lie in one live allocation that stream
    party `p` may access without a race. -/
def DevReadyOn (d : Dev) (q : Nat) (p : UInt64) (n : Nat) (write : Bool) : Prop :=
  ∃ id off b, d.range? p n = some (id, off, b) ∧ Ready d.race q id write

/-- A stream may wait for an event: both are live and the wait is one the
    model states (`Dev.waitOn`). -/
def WaitOk (d : Dev) (hstream hev : UInt64) : Prop :=
  ∃ p id ev, d.streamParty? hstream = some p ∧ d.eventOf? hev = some (id, ev) ∧ (d.waitOn p ev).isSome = true

/-- A live graph's handle, and a live instantiated graph's. -/
def GraphLive (d : Dev) (h : UInt64) : Prop := ∃ i ops, handleOf kGraph h = some i ∧ d.graphs[i]? = some (some ops)
def ExecLive (d : Dev) (h : UInt64) : Prop := ∃ i ops, handleOf kExec h = some i ∧ d.execs[i]? = some (some ops)

/-- An instantiated graph that runs on a stream: live, and its nodes run
    without a race against what was issued before and within the oracles. -/
def GraphRunsOn (w : World) (hexec hstream : UInt64) : Prop :=
  ∃ i ops p, handleOf kExec hexec = some i ∧ w.dev.execs[i]? = some (some ops) ∧
    w.dev.streamParty? hstream = some p ∧ (w.dev.runGraph w ops p).isSome = true

/-- A live cuBLAS handle. -/
def BlasLive (d : Dev) (h : UInt64) : Prop := ∃ i s, handleOf kBlas h = some i ∧ d.blas[i]? = some (some s)

/-- A product cuBLAS either refuses or runs: the handle's stream is live, and
    unless cuBLAS refuses the arguments, what the model states of them holds,
    the operands lie in their allocations with the output apart from the
    inputs, `α` and `β` can be read, and the stream may access them. -/
def GemmRuns (w : World) (ea eb ec : Nat)
    (h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch : UInt64) : Prop :=
  ∃ p, w.dev.blasParty? h = some p ∧
    (gemmInvalid ta tb m n k lda ldb ldc batch = false →
      gemmUnstated m n k batch sa sb sc ldc = false ∧
      ∃ ia oa ib ob ic oc,
        w.dev.operand? A ea (gemmSpans ta tb m n k lda ldb ldc sa sb sc batch).1 = some (ia, oa) ∧
        w.dev.operand? B eb (gemmSpans ta tb m n k lda ldb ldc sa sb sc batch).2.1 = some (ib, ob) ∧
        w.dev.operand? C ec (gemmSpans ta tb m n k lda ldb ldc sa sb sc batch).2.2 = some (ic, oc) ∧
        ic ≠ ia ∧ ic ≠ ib ∧ (w.mem.load pa 4).isSome = true ∧ (w.mem.load pb 4).isSome = true ∧
        OpReady (w.dev.raceFor p) p [ia, ib, ic] [ic])

/-- The same for an `sgemv`, whose operands start their allocations. -/
def SgemvRuns (w : World) (h trans m n pa A lda x incx pb y incy : UInt64) : Prop :=
  ∃ p, w.dev.blasParty? h = some p ∧
    (sgemvInvalid trans m n lda incx incy = false →
      sgemvUnstated m n lda incx incy = false ∧
      ∃ ia ix iy,
        w.dev.operand? A 4 ((asI32 m).toNat * (asI32 n).toNat) = some (ia, 0) ∧
        w.dev.operand? x 4 (sgemvLens trans m n).1 = some (ix, 0) ∧
        w.dev.operand? y 4 (sgemvLens trans m n).2 = some (iy, 0) ∧
        iy ≠ ia ∧ iy ≠ ix ∧ (w.mem.load pa 4).isSome = true ∧ (w.mem.load pb 4).isSome = true ∧
        OpReady (w.dev.raceFor p) p [ia, ix, iy] [iy])

/-- Each member of a pointer-array batch may run once those before it have:
    its operands are live and it is ready on `p`. -/
def BatchRuns (w : World) (p : Nat) (ta tb m n k al be : UInt64) (arrs : List Nat) :
    List ((Nat × Nat) × (Nat × Nat) × (Nat × Nat)) → Dev → Prop
  | [], _ => True
  | (a, b, c) :: ms, d =>
      (∃ A, (d.bufs[a.1]?).join = some A) ∧ (∃ B, (d.bufs[b.1]?).join = some B) ∧
      (∃ C, (d.bufs[c.1]?).join = some C) ∧
      OpReady (d.raceFor p) p (arrs ++ [a.1, b.1, c.1]) [c.1] ∧
      ∀ d', d.devOp w p (.vendor ⟨"sgemmBatchedMember",
          [ta, tb, m, n, k, al, be, UInt64.ofNat (4 * a.2), UInt64.ofNat (4 * b.2),
           UInt64.ofNat (4 * c.2)]⟩ [a.1, b.1, c.1] c.1) (arrs ++ [a.1, b.1, c.1]) [c.1] = some d' →
        BatchRuns w p ta tb m n k al be arrs ms d'

/-- A pointer-array batch cuBLAS either refuses or runs: its handle's stream is
    live, and unless cuBLAS refuses the arguments, the model states them, the
    arrays lie in their allocations, every member's operands lie in theirs
    with the outputs apart, `α` and `β` can be read, and each member may run
    in turn. -/
def SgemmBatchedRuns (w : World) (h ta tb m n k pa PA lda PB ldb pb PC ldc batch : UInt64) : Prop :=
  ∃ p, w.dev.blasParty? h = some p ∧
    (gemmInvalid ta tb m n k lda ldb ldc batch = false →
      batchUnstated ta tb m n k lda ldb ldc batch = false ∧
      ∃ iA oA bA iB oB bB iC oC bC ms al be,
        w.dev.range? PA (8 * (asI32 batch).toNat) = some (iA, oA, bA) ∧
        w.dev.range? PB (8 * (asI32 batch).toNat) = some (iB, oB, bB) ∧
        w.dev.range? PC (8 * (asI32 batch).toNat) = some (iC, oC, bC) ∧
        (List.range (asI32 batch).toNat).mapM
          (blasMember w.dev ta tb m n k bA bB bC oA oB oC) = some ms ∧
        batchApart ms [iA, iB, iC] = true ∧
        w.mem.load pa 4 = some al ∧ w.mem.load pb 4 = some be ∧
        BatchRuns w p ta tb m n k al be [iA, iB, iC] ms w.dev)

-- ---------------------------------------------------------------------------
-- The preconditions
-- ---------------------------------------------------------------------------

set_option maxHeartbeats 4000000 in
/-- What a driver call needs, by function and argument bits. -/
def cudaStepPre (f : CudaFn) (bits : List UInt64) : World → Prop :=
  match f with
  | .init => match bits with
    | [_] => fun _ => True
    | _ => fun _ => False
  | .deviceGet => match bits with
    | [pdev, _] => fun w => w.dev.drvInit = true ∧ Slot4 w.mem pdev
    | _ => fun _ => False
  | .primaryCtxRetain => match bits with
    | [pctx, dev] => fun w =>
      w.dev.drvInit = true ∧ asI32 dev = 0 ∧ Slot8 w.mem pctx
    | _ => fun _ => False
  | .primaryCtxRelease => match bits with
    | [dev] => fun w =>
      w.dev.drvInit = true ∧ asI32 dev = 0 ∧ 0 < w.dev.retained
    | _ => fun _ => False
  | .ctxSetCurrent => match bits with
    | [ctx] => fun w => ctx = 0 ∨ (ctx = cudaCtx ∧ 0 < w.dev.retained)
    | _ => fun _ => False
  | .ctxSynchronize => match bits with
    | [] => fun w => w.dev.live = true
    | _ => fun _ => False
  | .memGetInfo => match bits with
    | [pfree, ptotal] => fun w =>
      w.dev.live = true ∧ Slot8 w.mem pfree ∧ Slot8 w.mem ptotal
    | _ => fun _ => False
  | .memAlloc => match bits with
    | [pdptr, _] => fun w => w.dev.live = true ∧ Slot8 w.mem pdptr
    | _ => fun _ => False
  | .memFree => match bits with
    | [p] => fun w =>
      w.dev.live = true ∧
      ∃ id b, w.dev.range? p 0 = some (id, 0, b) ∧ Ready w.dev.race defaultParty id true
    | _ => fun _ => False
  | .memcpyHtoD => match bits with
    | [dst, src, n] => fun w =>
      w.dev.live = true ∧ DevReady w.dev dst n.toNat true ∧ Readable w.mem src n.toNat
    | _ => fun _ => False
  | .memcpyDtoH => match bits with
    | [dst, src, n] => fun w =>
      w.dev.live = true ∧ DevReady w.dev src n.toNat false ∧ Writable w.mem dst n.toNat
    | _ => fun _ => False
  | .memcpyDtoD => match bits with
    | [dst, src, n] => fun w =>
      w.dev.live = true ∧ DevReady w.dev src n.toNat false ∧ DevReady w.dev dst n.toNat true
    | _ => fun _ => False
  | .memsetD8 => match bits with
    | [dst, _, n] => fun w => w.dev.live = true ∧ DevReady w.dev dst n.toNat true
    | _ => fun _ => False
  | .moduleLoadData => match bits with
    | [pmod, image] => fun w =>
      w.dev.live = true ∧ CStr w.mem image ∧ Slot8 w.mem pmod
    | _ => fun _ => False
  | .moduleGetFunction => match bits with
    | [pfunc, hmod, pname] => fun w =>
      w.dev.live = true ∧ ModuleLive w.dev hmod ∧ CStr w.mem pname ∧ Slot8 w.mem pfunc
    | _ => fun _ => False
  | .moduleUnload => match bits with
    | [hmod] => fun w => w.dev.live = true ∧ ModuleLive w.dev hmod
    | _ => fun _ => False
  | .launchKernel => match bits with
    | [hf, _, _, _, _, _, _, _, hstream, params, extra] => fun w =>
      w.dev.live = true ∧ extra = 0 ∧ KeepsSize w.kernel ∧ LaunchOk w hf hstream params
    | _ => fun _ => False
  | .streamCreate => match bits with
    | [pstream, flags] => fun w =>
      w.dev.live = true ∧ flags &&& 0xffffffff = 1 ∧ Slot8 w.mem pstream
    | _ => fun _ => False
  | .streamSynchronize => match bits with
    | [h] => fun w =>
      w.dev.live = true ∧ w.dev.capture = none ∧ StreamLive w.dev h
    | _ => fun _ => False
  | .streamDestroy => match bits with
    | [h] => fun w =>
      w.dev.live = true ∧ ∃ id, handleOf kStream h = some id ∧ w.dev.streams.getD id false = true
    | _ => fun _ => False
  | .eventCreate => match bits with
    | [pev, flags] => fun w =>
      w.dev.live = true ∧ flags &&& 0xffffffff = 0 ∧ Slot8 w.mem pev
    | _ => fun _ => False
  | .eventRecord => match bits with
    | [hev, hstream] => fun w =>
      w.dev.live = true ∧ EventLive w.dev hev ∧ StreamLive w.dev hstream
    | _ => fun _ => False
  | .streamWaitEvent => match bits with
    | [hstream, hev, flags] => fun w =>
      w.dev.live = true ∧ flags &&& 0xffffffff = 0 ∧ WaitOk w.dev hstream hev
    | _ => fun _ => False
  | .eventSynchronize => match bits with
    | [hev] => fun w => w.dev.live = true ∧ EventPlain w.dev hev
    | _ => fun _ => False
  | .eventDestroy => match bits with
    | [hev] => fun w => w.dev.live = true ∧ EventLive w.dev hev
    | _ => fun _ => False
  | .eventElapsedTime => match bits with
    | [pms, hs, he] => fun w =>
      w.dev.live = true ∧ Slot4 w.mem pms ∧ ElapsedReady w.dev hs he
    | _ => fun _ => False
  | .memAllocHost => match bits with
    | [pp, size] => fun w =>
      w.dev.live = true ∧ size ≠ 0 ∧ w.mem.pinnedNext + size.toNat ≤ regionSpan.toNat ∧ Slot8 w.mem pp
    | _ => fun _ => False
  | .memFreeHost => match bits with
    | [p] => fun w =>
      w.dev.live = true ∧
      ∃ id off n, w.dev.hostAllocAt? p = some (id, off, n) ∧ ∀ b ∈ w.mem.busy, b.clear off n = true
    | _ => fun _ => False
  | .memcpyHtoDAsync => match bits with
    | [dst, src, n, hstream] => fun w =>
      w.dev.live = true ∧ w.dev.capture = none ∧
      ∃ q, w.dev.streamParty? hstream = some q ∧ DevReadyOn w.dev q dst n.toNat true ∧
        Readable (w.mem.forParty q) src n.toNat
    | _ => fun _ => False
  | .memcpyDtoHAsync => match bits with
    | [dst, src, n, hstream] => fun w =>
      w.dev.live = true ∧ w.dev.capture = none ∧
      ∃ q, w.dev.streamParty? hstream = some q ∧ DevReadyOn w.dev q src n.toNat false ∧
        Writable (w.mem.forParty q) dst n.toNat
    | _ => fun _ => False
  | .beginCapture => match bits with
    | [hstream, mode] => fun w =>
      w.dev.live = true ∧ mode &&& 0xffffffff = 2 ∧ w.dev.capture = none ∧
      ∃ p, w.dev.streamParty? hstream = some p ∧ p ≠ defaultParty
    | _ => fun _ => False
  | .endCapture => match bits with
    | [hstream, pgraph] => fun w =>
      w.dev.live = true ∧ Slot8 w.mem pgraph ∧
      ∃ p c, w.dev.streamParty? hstream = some p ∧ w.dev.capture = some c ∧ c.origin = p
    | _ => fun _ => False
  | .graphInstantiate => match bits with
    | [pexec, hgraph, flags] => fun w =>
      w.dev.live = true ∧ flags = 0 ∧ GraphLive w.dev hgraph ∧ Slot8 w.mem pexec
    | _ => fun _ => False
  | .graphLaunch => match bits with
    | [hexec, hstream] => fun w => w.dev.live = true ∧ GraphRunsOn w hexec hstream
    | _ => fun _ => False
  | .graphExecDestroy => match bits with
    | [hexec] => fun w => w.dev.live = true ∧ ExecLive w.dev hexec
    | _ => fun _ => False
  | .graphDestroy => match bits with
    | [hgraph] => fun w => w.dev.live = true ∧ GraphLive w.dev hgraph
    | _ => fun _ => False

/-- What a driver call needs: its own precondition, and no capture in
    progress unless it is one that keeps its meaning under one. -/
def cudaPre (f : CudaFn) (bits : List UInt64) (w : World) : Prop :=
  (w.dev.capture = none ∨ f.underCapture = true) ∧ cudaStepPre f bits w

/-- What a cuBLAS call needs. -/
def cublasPre : CublasFn → List UInt64 → World → Prop
  | .create, [ph] => fun w => w.dev.live = true ∧ w.dev.capture = none ∧ Slot8 w.mem ph
  | .destroy, [h] => fun w => w.dev.live = true ∧ w.dev.capture = none ∧ BlasLive w.dev h
  | .setStream, [h, s] => fun w =>
      w.dev.live = true ∧ w.dev.capture = none ∧ BlasLive w.dev h ∧ StreamLive w.dev s
  | .sgemv, [h, trans, m, n, pa, A, lda, x, incx, pb, y, incy] => fun w =>
      w.dev.live = true ∧ VendorKeeps w.vendor ∧
      SgemvRuns w h trans m n pa A lda x incx pb y incy
  | .sgemm, [h, ta, tb, m, n, k, pa, A, lda, B, ldb, pb, C, ldc] => fun w =>
      w.dev.live = true ∧ VendorKeeps w.vendor ∧
      GemmRuns w 4 4 4 h ta tb m n k pa A lda 0 B ldb 0 pb C ldc 0 1
  | .sgemmStridedBatched, [h, ta, tb, m, n, k, pa, A, lda, sa, B, ldb, sb, pb, C, ldc, sc, batch] =>
      fun w => w.dev.live = true ∧ VendorKeeps w.vendor ∧
        GemmRuns w 4 4 4 h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch
  | .gemmEx, [h, ta, tb, m, n, k, pa, A, at_, lda, B, bt, ldb, pb, C, ct, ldc, compute, algo] =>
      fun w => w.dev.live = true ∧ VendorKeeps w.vendor ∧
        bf16In32Out at_ bt ct compute algo = true ∧
        GemmRuns w 2 2 4 h ta tb m n k pa A lda 0 B ldb 0 pb C ldc 0 1
  | .gemmStridedBatchedEx,
      [h, ta, tb, m, n, k, pa, A, at_, lda, sa, B, bt, ldb, sb, pb, C, ct, ldc, sc, batch, compute, algo] =>
      fun w => w.dev.live = true ∧ VendorKeeps w.vendor ∧
        bf16In32Out at_ bt ct compute algo = true ∧
        GemmRuns w 2 2 4 h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch
  | .sgemmBatched, [h, ta, tb, m, n, k, pa, PA, lda, PB, ldb, pb, PC, ldc, batch] => fun w =>
      w.dev.live = true ∧ VendorKeeps w.vendor ∧
        SgemmBatchedRuns w h ta tb m n k pa PA lda PB ldb pb PC ldc batch
  | _, _ => fun _ => False

/-- What a wgpu call needs: its arguments, and that they decode — every handle
    one the program holds, of its parameter's kind and in the state the call
    needs, every descriptor laid out as the model states (`wgDecode`). -/
def wgpuPre (f : WgpuFn) (bits : List UInt64) (w : World) : Prop :=
  bits.length = (Ext.wgpu f).sig.1.length ∧ (wgDecode f bits w).isSome

/-- What a CPU library call needs: its arguments; every one answers. -/
def cpuPre (f : CpuFn) (bits : List UInt64) : Prop :=
  bits.length = (Ext.cpu f).sig.1.length

/-- What a serial library call needs: its arguments, and that they decode
    (`serDecode`). -/
def serialPre (f : SerialFn) (bits : List UInt64) (w : World) : Prop :=
  bits.length = (Ext.serial f).sig.1.length ∧ (serDecode f bits w).isSome

/-- What a USB library call needs: its arguments, and that they decode
    (`usbDecode`). -/
def usbPre (f : UsbFn) (bits : List UInt64) (w : World) : Prop :=
  bits.length = (Ext.usb f).sig.1.length ∧ (usbDecode f bits w).isSome

/-- What a window library call needs: its arguments, and that they decode
    (`wlDecode`). -/
def windowPre (f : WindowFn) (bits : List UInt64) (w : World) : Prop :=
  bits.length = (Ext.window f).sig.1.length ∧ (wlDecode f bits w).isSome

/-- **What a C library call needs**, from argument bits: nothing of a library
    that did not load. -/
def ExtPre (e : Ext) (bits : List UInt64) (w : World) : Prop :=
  w.has e.lib = true →
  match e, bits with
  | .present _, [] => True
  | .c .memcpy, [d, s, n] =>
      Readable w.mem s n.toNat ∧ Writable w.mem d n.toNat ∧
      (n.toNat = 0 ∨ s.toNat + n.toNat ≤ d.toNat ∨ d.toNat + n.toNat ≤ s.toNat)
  | .c .memmove, [d, s, n] => Readable w.mem s n.toNat ∧ Writable w.mem d n.toNat
  | .c .memset, [d, _, n] => Writable w.mem d n.toNat
  | .c .strlen, [a] => Terminated w.mem a
  | .c .calloc, [n, sz] => 0 < n.toNat * sz.toNat
  | .c .free, [p] => p = 0 ∨ (w.heapAt? p).isSome = true
  | .cuda f, bits => cudaPre f bits w
  | .cublas f, bits => cublasPre f bits w
  | .wgpu f, bits => wgpuPre f bits w
  | .window f, bits => windowPre f bits w
  | .cpu f, bits => cpuPre f bits
  | .serial f, bits => serialPre f bits w
  | .usb f, bits => usbPre f bits w
  | _, _ => False

/-- **Where each clause comes from.** The references are the C standard and
    the CUDA Driver API reference; the corpora are
    `Host/Corpus.lean` (`casesExt`) and `Host/DriverCorpus.lean`. -/
def _root_.AlgorithmLib.IR.Ext.sources : Ext → List (String × Source)
  | .present _ => [("none", .observed "Corpus casesExt: the probe answers 1 for libc")]
  | .c .memcpy =>
      [("source readable, destination writable", .documented "C11 7.24.2.1"),
       ("ranges disjoint", .documented "C11 7.24.2.1: copying between overlapping objects is undefined")]
  | .c .memmove => [("source readable, destination writable", .documented "C11 7.24.2.2")]
  | .c .memset => [("destination writable", .documented "C11 7.24.6.1")]
  | .c .strlen => [("terminated string", .documented "C11 7.24.6.3")]
  | .c .calloc =>
      [("size nonzero", .unmodelled "C11 7.22.3: a zero-size request may answer null or a unique pointer")]
  | .c .free =>
      [("null, or the start of a live allocation", .documented "C11 7.22.3.3")]
  | .cuda .init => [("none", .observed "DriverCorpus: cuInit(0) answers 0, flags ≠ 0 answer 1")]
  | .cuda .deviceGet =>
      [("cuInit called", .documented "Driver API: every call but cuInit before cuInit is undefined"),
       ("slot writable", .documented "cuDeviceGet writes *device")]
  | .cuda .primaryCtxRetain =>
      [("cuInit called", .documented "Driver API: initialization"),
       ("device 0", .unmodelled "the model has one device context"),
       ("slot writable", .documented "cuDevicePrimaryCtxRetain writes *pctx")]
  | .cuda .primaryCtxRelease =>
      [("cuInit called", .documented "Driver API: initialization"),
       ("device 0", .unmodelled "the model has one device context"),
       ("retained", .documented "releasing more often than retained is undefined")]
  | .cuda .ctxSetCurrent =>
      [("null or the retained context", .documented "cuCtxSetCurrent on a destroyed context is undefined")]
  | .cuda .ctxSynchronize => [("context current", .documented "Driver API: calls need a current context")]
  | .cuda .memGetInfo =>
      [("context current", .documented "Driver API: current context"),
       ("slots writable", .documented "cuMemGetInfo writes *free and *total")]
  | .cuda .memAlloc =>
      [("context current", .documented "Driver API: current context"),
       ("slot writable", .documented "cuMemAlloc writes *dptr")]
  | .cuda .memFree =>
      [("context current", .documented "Driver API: current context"),
       ("a live allocation's base", .documented "cuMemFree: dptr must come from cuMemAlloc"),
       ("no pending access", .documented "freeing memory the device still uses is a race")]
  | .cuda .memcpyHtoD | .cuda .memcpyDtoH | .cuda .memcpyDtoD | .cuda .memsetD8 =>
      [("context current", .documented "Driver API: current context"),
       ("inside one allocation", .documented "Driver API: copies outside an allocation are undefined"),
       ("no race", .documented "CUDA memory model: unordered conflicting accesses are a race"),
       ("host side readable or writable", .documented "the host pointer must cover the count")]
  | .cpu .count =>
      [("none", .documented "cpu.rs: sysconf(_SC_NPROCESSORS_ONLN), GetActiveProcessorCount")]
  | .cpu .core | .cpu .package =>
      [("what the system reports", .unmodelled "the world's cpus"),
       ("a CPU not there answers -1", .observed "CpuCorpus: -1 and the count itself")]
  | .cpu .pin =>
      [("granted or declined", .unmodelled "the world's cpuPins"),
       ("a CPU not there is declined", .observed "CpuCorpus: -1 and the count itself")]
  | .cpu .unpin => [("granted or declined", .unmodelled "the world's cpuPins")]
  | .serial .count => [("the ports listed", .unmodelled "the world's serialNames")]
  | .serial .name =>
      [("cap bytes writable", .documented "serial.rs: base_serial_name writes at most cap bytes"),
       ("a port not listed answers -1", .observed "SerialCorpus: -1 and the count itself")]
  | .serial .open =>
      [("path a C string", .documented "serial.rs: base_serial_open reads path to its NUL"),
       ("baud 0 or a negative timeout refused", .observed "SerialCorpus"),
       ("held exclusively", .observed "SerialCorpus: a second open of a port held answers null"),
       ("which paths open", .unmodelled "the world's serialDevices")]
  | .serial .close => [("an open port", .documented "serial.rs: base_serial_close drops the port")]
  | .serial .read =>
      [("an open port", .documented "serial.rs: base_serial_close drops the port"),
       ("len bytes writable", .documented "serial.rs: base_serial_read writes at most len bytes"),
       ("what has arrived, up to len; 0 when nothing did", .observed "SerialCorpus: a terminal's greeting in two pieces, then 0"),
       ("a negative length answers -1", .observed "SerialCorpus")]
  | .serial .write =>
      [("an open port", .documented "serial.rs: base_serial_close drops the port"),
       ("len bytes readable", .documented "serial.rs: base_serial_write sends len bytes"),
       ("every byte, or -1", .observed "SerialCorpus: the terminal receives exactly what was sent")]
  | .serial .pending =>
      [("an open port", .documented "serial.rs: base_serial_close drops the port"),
       ("what has arrived", .observed "SerialCorpus: the greeting's length, then 0")]
  | .usb .count | .usb .info =>
      [("the devices listed", .unmodelled "the world's usbDevices"),
       ("a device not listed, or a question not asked, answers -1", .observed "UsbCorpus")]
  | .usb .open =>
      [("whether this process may open it", .unmodelled "the world's usbDevices"),
       ("a device not listed answers null", .observed "UsbCorpus")]
  | .usb .close => [("an open device", .documented "usb.rs: base_usb_close drops the device")]
  | .usb .claim | .usb .release =>
      [("an open device", .documented "usb.rs: base_usb_close drops the device"),
       ("which interfaces it lets be claimed", .unmodelled "the world's usbDevices"),
       ("releasing one not claimed answers -1", .documented "usb.rs: base_usb_release")]
  | .usb .control =>
      [("an open device", .documented "usb.rs: base_usb_close drops the device"),
       ("len bytes of data", .documented "USB 2.0 9.3: wLength bytes in the data stage"),
       ("a kind or recipient USB lacks refused", .documented "USB 2.0 9.3: bmRequestType"),
       ("on Windows through a claimed interface", .documented "nusb: Device::control_in is not on Windows"),
       ("what the device answers", .unmodelled "the world's usbReply")]
  | .usb .bulk | .usb .interrupt =>
      [("an open device, the interface claimed", .documented "nusb: endpoints belong to a claimed interface"),
       ("from the device, whole packets", .documented "nusb: an IN transfer asks for a multiple of the packet size"),
       ("len bytes of data", .documented "usb.rs: base_usb_bulk moves at most len bytes"),
       ("what the device answers", .unmodelled "the world's usbReply")]
  | .wgpu .createInstance => []
  | .wgpu .requestAdapter =>
      [("an instance held", .documented "wgpu.rs: a handle of another kind is refused"),
       ("an adapter, or 0 without one", .unmodelled "the world's wgAdapter")]
  | .wgpu .requestDevice =>
      [("an adapter held", .documented "wgpu.rs: a handle of another kind is refused"),
       ("a device with no scope pushed", .documented "WebGPU: a new device's error scope stack is empty")]
  | .wgpu (.release _) =>
      [("an object the program holds, of the kind", .documented "wgpu.rs: a release frees the object's box; a use after it is undefined")]
  | .wgpu .deviceGetQueue => [("device held", .documented "wgpu.rs: a device holds its one queue")]
  | .wgpu .deviceCreateBuffer =>
      [("device held", .documented "wgpu.rs: a handle of another kind is refused"),
       ("usage nonzero, MAP_READ only with COPY_DST", .documented "WebGPU: createBuffer validation"),
       ("zeroed", .documented "WebGPU: buffers are zero-initialized"),
       ("room", .unmodelled "running out of memory is not modelled")]
  | .wgpu .deviceCreateShaderModule =>
      [("len bytes of UTF-8 at src", .documented "wgpu.rs: the source is the bytes it is handed, as UTF-8"),
       ("compiles or records an error", .observed "WgpuCorpus: a shader wgpu rejects is caught by an error scope")]
  | .wgpu .deviceCreateBindGroupLayout =>
      [("n words at ents, distinct bindings", .documented "WebGPU: createBindGroupLayout validation"),
       ("storage buffers seen by compute", .documented "wgpu.rs: every entry is a storage buffer")]
  | .wgpu .deviceCreatePipelineLayout =>
      [("n layouts the program holds", .documented "WebGPU: createPipelineLayout validation")]
  | .wgpu .deviceCreateComputePipeline =>
      [("a layout and a module held, the entry point's name UTF-8", .documented "WebGPU: createComputePipeline validation"),
       ("valid when the module is", .observed "WgpuCorpus: a shader wgpu rejects is caught by an error scope")]
  | .wgpu .deviceCreateBindGroup =>
      [("whole storage buffers the program holds", .documented "WebGPU: createBindGroup validation"),
       ("exactly the layout's bindings", .documented "WebGPU: every layout entry has one bind group entry")]
  | .wgpu .deviceCreateCommandEncoder => [("device held", .documented "wgpu.rs: a handle of another kind is refused")]
  | .wgpu .deviceCreateRenderPipeline =>
      [("a module held, entry points UTF-8", .documented "WebGPU: createRenderPipeline validation"),
       ("RGBA8 or BGRA8", .documented "wgpu.rs: the formats a target takes")]
  | .wgpu .devicePushErrorScope => [("validation errors", .documented "wgpu.rs: a scope catches validation errors")]
  | .wgpu .errorScopePop =>
      [("the innermost scope of its device", .documented "WebGPU: popErrorScope pops the top of the stack"),
       ("used up", .documented "wgpu: ErrorScopeGuard::pop consumes the guard")]
  | .wgpu .renderPipelineGetBindGroupLayout =>
      [("a valid pipeline, group 0", .documented "WebGPU: getBindGroupLayout on an index the layout has")]
  | .wgpu .encoderBeginComputePass | .wgpu .encoderBeginRenderPass =>
      [("encoder open, no pass open on it", .documented "WebGPU: an encoder is locked while a pass is open")]
  | .wgpu .computePassSetPipeline | .wgpu .computePassSetBindGroup
  | .wgpu .renderPassSetPipeline | .wgpu .renderPassSetBindGroup =>
      [("pass not ended, a valid pipeline", .documented "WebGPU: a pass is used until end()")]
  | .wgpu .computePassEnd | .wgpu .renderPassEnd =>
      [("pass not ended", .documented "WebGPU: a pass is used until end()"),
       ("used up, its encoder unlocked", .documented "wgpu: dropping a pass ends it")]
  | .wgpu .computePassDispatch =>
      [("pipeline set, every group of its layout bound", .documented "WebGPU: dispatch validation"),
       ("counts at most 65535", .documented "WebGPU: maxComputeWorkgroupsPerDimension"),
       ("oracle keeps sizes", .unmodelled "the shader is read by its oracle, whose answers are cut to each buffer's length")]
  | .wgpu .renderPassDraw => [("pipeline set", .documented "WebGPU: draw validation")]
  | .wgpu .encoderCopyBufferToBuffer =>
      [("COPY_SRC to COPY_DST, distinct", .documented "WebGPU: copyBufferToBuffer validation"),
       ("offsets and size multiples of 4, inside both", .documented "WebGPU: copyBufferToBuffer validation")]
  | .wgpu .encoderFinish =>
      [("encoder open", .documented "WebGPU: finish() with a pass open is an error"),
       ("used up", .documented "wgpu: CommandEncoder::finish consumes the encoder")]
  | .wgpu .queueSubmit =>
      [("a command buffer not yet submitted, used up", .documented "wgpu: Queue::submit consumes its command buffers"),
       ("runs in order", .observed "WgpuCorpus: copies and dispatches read back in submission order")]
  | .wgpu .queueWriteBuffer =>
      [("COPY_DST, offset and size multiples of 4, inside", .documented "WebGPU: writeBuffer validation"),
       ("source readable", .documented "wgpu.rs: n bytes at src")]
  | .wgpu .bufferRead =>
      [("MAP_READ, offset a multiple of 8, size of 4, inside", .documented "WebGPU: mapAsync validation"),
       ("room at dst", .documented "wgpu.rs: n bytes copied to dst"),
       ("waits for the queue", .observed "WgpuCorpus: bytes read back after a submit")]
  | .wgpu .instanceCreateSurface =>
      [("a window the window library holds open", .documented "wgpu.rs: a surface of the window's handle keeps it alive"),
       ("a surface", .observed "WindowCorpus, with a display")]
  | .wgpu .surfaceConfigure =>
      [("device held, RGBA8 or BGRA8, nonzero size", .documented "WebGPU: surface configuration"),
       ("no texture outstanding", .documented "wgpu: a surface texture must be presented or dropped before reconfiguring")]
  | .wgpu .surfaceGetCurrentTexture =>
      [("configured, last texture presented", .documented "wgpu: one texture outstanding"),
       ("a texture", .unmodelled "a texture that times out or is outdated answers 0, which the model does not state")]
  | .wgpu .surfacePresent =>
      [("the texture its surface holds, used up", .documented "wgpu: SurfaceTexture::present consumes the texture")]
  | .wgpu .textureCreateView => [("a texture held", .documented "wgpu.rs: the texture's default view")]
  | .window .init =>
      [("a display", .observed "WindowCorpus: without one base_window_init answers false"),
       ("one event loop, on the first thread to ask", .documented "window.rs: winit makes one per process")]
  | .window .open =>
      [("title a C string", .documented "window.rs: base_window_open reads title to its NUL"),
       ("a size that is not positive answers null", .documented "window.rs: base_window_open")]
  | .window .close | .window .pixels =>
      [("an open window", .documented "window.rs: base_window_close drops the window")]
  | .window .pump => []
  | .window .poll =>
      [("an open window", .documented "window.rs: base_window_close drops the window"),
       ("room for a record", .documented "window.rs: base_window_poll writes four i64s"),
       ("what arrives", .unmodelled "the world's wlInput")]
  | .cuda .moduleLoadData =>
      [("context current", .documented "Driver API: current context"),
       ("image a terminated string", .documented "cuModuleLoadData: PTX images are NUL-terminated"),
       ("slot writable", .documented "cuModuleLoadData writes *module")]
  | .cuda .moduleGetFunction =>
      [("context current", .documented "Driver API: current context"),
       ("module loaded", .documented "a handle after cuModuleUnload is undefined"),
       ("name a terminated string", .documented "cuModuleGetFunction: name is a C string"),
       ("slot writable", .documented "cuModuleGetFunction writes *hfunc")]
  | .cuda .moduleUnload =>
      [("context current", .documented "Driver API: current context"),
       ("module loaded", .documented "unloading twice is undefined")]
  | .cuda .launchKernel =>
      [("context current", .documented "Driver API: current context"),
       ("extra null", .unmodelled "the model reads parameters from kernelParams only"),
       ("oracle keeps sizes", .unmodelled "the kernel is read by its oracle, which must keep buffer lengths"),
       ("function, stream, parameters", .documented "cuLaunchKernel: kernelParams holds one pointer per parameter"),
       ("no race", .documented "CUDA memory model: unordered conflicting accesses are a race")]
  | .cuda .streamCreate =>
      [("context current", .documented "Driver API: current context"),
       ("non-blocking", .unmodelled "a blocking stream's implicit order with the default stream is not modelled"),
       ("slot writable", .documented "cuStreamCreate writes *phStream")]
  | .cuda .streamSynchronize =>
      [("context current", .documented "Driver API: current context"),
       ("not capturing", .documented "synchronizing a capturing stream is invalid"),
       ("stream live", .documented "a destroyed stream's handle is undefined")]
  | .cuda .streamDestroy =>
      [("context current", .documented "Driver API: current context"),
       ("stream live", .documented "destroying twice is undefined")]
  | .cuda .eventCreate =>
      [("context current", .documented "Driver API: current context"),
       ("default flags", .unmodelled "timing-disabled and blocking-sync events are not admitted until a corpus shows them"),
       ("slot writable", .documented "cuEventCreate writes *phEvent")]
  | .cuda .eventRecord =>
      [("context current", .documented "Driver API: current context"),
       ("event live", .documented "an event after cuEventDestroy is undefined"),
       ("stream live", .documented "a destroyed stream's handle is undefined")]
  | .cuda .streamWaitEvent =>
      [("context current", .documented "Driver API: current context"),
       ("flags zero", .documented "cuStreamWaitEvent: Flags must be 0"),
       ("stream and event live", .documented "destroyed handles are undefined"),
       ("a capture's point only while it is in progress", .unmodelled "waiting outside a capture on a point inside it is not modelled"),
       ("a capturing stream waits only on its capture's points", .unmodelled "a capture depending on outside work is not modelled")]
  | .cuda .eventSynchronize =>
      [("context current", .documented "Driver API: current context"),
       ("event live, recorded outside a capture", .documented "a destroyed event is undefined")]
  | .cuda .eventDestroy =>
      [("context current", .documented "Driver API: current context"),
       ("event live", .documented "destroying twice is undefined")]
  | .cuda .eventElapsedTime =>
      [("context current", .documented "Driver API: current context"),
       ("slot writable", .documented "cuEventElapsedTime writes *pMilliseconds"),
       ("both recorded and waited for", .unmodelled "before completion the driver may answer or report not-ready")]
  | .cuda .memAllocHost =>
      [("context current", .documented "Driver API: current context"),
       ("size nonzero", .unmodelled "a zero-byte page-locked allocation is not modelled"),
       ("room", .unmodelled "running out of page-locked memory is not modelled"),
       ("slot writable", .documented "cuMemAllocHost writes *pp")]
  | .cuda .memFreeHost =>
      [("context current", .documented "Driver API: current context"),
       ("an allocation's start", .documented "cuMemFreeHost: p must come from cuMemAllocHost"),
       ("no copy in flight", .documented "freeing memory an async copy still uses is a race")]
  | .cublas .create =>
      [("context current", .documented "cuBLAS: a handle is created in the current context"),
       ("no capture", .unmodelled "handles are managed outside a capture"),
       ("slot writable", .documented "cublasCreate writes *handle")]
  | .cublas .destroy =>
      [("handle live", .documented "destroying a handle twice is undefined"),
       ("no capture", .unmodelled "handles are managed outside a capture")]
  | .cublas .setStream =>
      [("handle and stream live", .documented "destroyed handles and streams are undefined"),
       ("no capture", .unmodelled "handles are managed outside a capture")]
  | .cublas .sgemv | .cublas .sgemm | .cublas .sgemmStridedBatched | .cublas .gemmEx
  | .cublas .gemmStridedBatchedEx =>
      [("context current", .documented "cuBLAS: a handle's context must be current"),
       ("oracle keeps the output's size", .unmodelled "the routine is read by its oracle"),
       ("handle and its stream live", .documented "destroyed handles and streams are undefined"),
       ("refused, or: operands inside their allocations", .documented "cuBLAS does not check operand extents"),
       ("element-aligned operands", .documented "cuBLAS: pointers are aligned to their element type"),
       ("output apart from the inputs", .documented "cuBLAS: C must not overlap A or B"),
       ("output in its own allocation", .unmodelled "an output sharing an allocation with an input is not modelled"),
       ("nonempty, unit strides, packed sgemv", .unmodelled "empty products and strided vectors are not modelled"),
       ("batches apart", .documented "cuBLAS: overlapping outputs across a batch are undefined"),
       ("bf16 in, f32 out, 32-bit compute", .unmodelled "the one gemmEx combination the model states"),
       ("α and β readable", .documented "pointer mode host: α and β are read from host memory"),
       ("no race", .documented "CUDA memory model: unordered conflicting accesses are a race")]
  | .cublas .sgemmBatched =>
      [("context current", .documented "cuBLAS: a handle's context must be current"),
       ("oracle keeps the output's size", .unmodelled "the routine is read by its oracle"),
       ("handle and its stream live", .documented "destroyed handles and streams are undefined"),
       ("refused, or: arrays and operands inside their allocations", .documented "cuBLAS does not check operand extents"),
       ("element-aligned operands", .documented "cuBLAS: pointers are aligned to their element type"),
       ("outputs apart from each other and every input", .documented "cuBLAS: C must not overlap A or B"),
       ("nonempty, packed operands", .unmodelled "empty batches and padded operands are not modelled"),
       ("arrays as they are when the call is made", .unmodelled "cuBLAS reads the arrays on the device, when the call runs"),
       ("α and β readable", .documented "pointer mode host: α and β are read from host memory"),
       ("no race", .documented "CUDA memory model: unordered conflicting accesses are a race")]
  | .cuda .beginCapture =>
      [("context current, no capture", .unmodelled "one capture at a time"),
       ("relaxed mode", .unmodelled "the other modes forbid calls the model does not tell apart"),
       ("a created stream", .documented "the legacy default stream cannot be captured")]
  | .cuda .endCapture =>
      [("the capture's origin stream", .documented "cuStreamEndCapture: hStream must be the capturing stream"),
       ("slot writable", .documented "cuStreamEndCapture writes *phGraph")]
  | .cuda .graphInstantiate =>
      [("flags zero", .unmodelled "instantiation flags are not modelled"),
       ("graph live", .documented "a destroyed graph is undefined"),
       ("slot writable", .documented "cuGraphInstantiate writes *phGraphExec")]
  | .cuda .graphLaunch =>
      [("graph and stream live", .documented "destroyed handles are undefined"),
       ("its nodes run without a race", .documented "CUDA memory model: unordered conflicting accesses are a race")]
  | .cuda .graphExecDestroy | .cuda .graphDestroy =>
      [("live", .documented "destroying twice is undefined")]
  | .cuda .memcpyHtoDAsync | .cuda .memcpyDtoHAsync =>
      [("context current", .documented "Driver API: current context"),
       ("not capturing", .unmodelled "a captured copy reads host memory when the graph runs"),
       ("stream live", .documented "a destroyed stream's handle is undefined"),
       ("inside one allocation", .documented "copies outside an allocation are undefined"),
       ("no race", .documented "CUDA memory model: unordered conflicting accesses are a race"),
       ("host side readable or writable", .documented "the host pointer must cover the count")]

-- ---------------------------------------------------------------------------
-- Calls from bits
-- ---------------------------------------------------------------------------

/-- The argument values a call with these bits passes, typed by its signature. -/
def _root_.AlgorithmLib.IR.Ext.args (e : Ext) (bits : List UInt64) : List V :=
  (e.sig.1.zip bits).map fun (t, b) => V.sc t b

theorem mapM_asBits_zip : ∀ (ts : List ClifTy) (bits : List UInt64), bits.length = ts.length →
    ((ts.zip bits).map fun (t, b) => V.sc t b).mapM asBits = some bits
  | [], [], _ => rfl
  | t :: ts, b :: bits, h => by
      simp only [List.zip_cons_cons, List.map_cons, List.mapM_cons, asBits]
      rw [mapM_asBits_zip ts bits (by simpa using h)]; rfl
  | [], _ :: _, h | _ :: _, [], h => by simp at h

set_option maxHeartbeats 1000000 in
/-- A driver call with its argument count. -/
theorem cudaStepPre_length {f : CudaFn} {bits : List UInt64} {w : World} (h : cudaStepPre f bits w) :
    bits.length = (Ext.cuda f).sig.1.length := by
  revert h
  cases f <;> simp only [cudaStepPre] <;> split <;> intro h <;> first | exact False.elim h | rfl


/-- A cuBLAS call with its argument count. -/
theorem cublasPre_length {f : CublasFn} {bits : List UInt64} {w : World} (h : cublasPre f bits w) :
    bits.length = (Ext.cublas f).sig.1.length := by
  unfold cublasPre at h
  split at h <;> first | exact h.elim | rfl

-- ---------------------------------------------------------------------------
-- Preconditions suffice
-- ---------------------------------------------------------------------------

theorem slot4_ne_none {m : Mem} {a v : UInt64} (h : Slot4 m a) (e : m.store a 4 v = none) : False := by
  have := h v; rw [e] at this; cases this

/-- A store leaves every other slot storable. -/
theorem slot8_after_store {m m' : Mem} {a b : UInt64} {n : Nat} {v : UInt64}
    (hs : m.store a n v = some m') (h : Slot8 m b) : Slot8 m' b := by
  obtain ⟨_, _, _, hsame, _⟩ := store_left (K := []) (MemSame.refl m) (fun _ _ _ h => by cases h) hs
  intro x
  rw [store_isSome hsame b 8 x x]
  exact h x

theorem op_isSome_rw {r : Race} {p s d : Nat} (hs : Ready r p s false) (hd : Ready r p d true) :
    (r.op p [s] [d]).isSome = true := by
  rw [op_some (fun x hx => by
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl
      · exact hs.1
      · exact hd.1)
    (fun x hx => by simp at hx; subst hx; exact hd.2 rfl)]
  rfl

theorem op_isSome_r {r : Race} {p s : Nat} (hs : Ready r p s false) : (r.op p [s] []).isSome = true := by
  have := op_isSome hs; simpa using this

theorem op_isSome_w {r : Race} {p d : Nat} (hd : Ready r p d true) : (r.op p [] [d]).isSome = true := by
  have := op_isSome hd; simpa using this

theorem op_isSome_ws {r : Race} {p : Nat} {ids : List Nat} (h : ∀ id ∈ ids, Ready r p id true) :
    (r.op p [] ids).isSome = true := by
  rw [op_some (fun x hx => (h x (by simpa using hx)).1) (fun x hx => (h x hx).2 rfl)]
  rfl

theorem range?_live {d : Dev} {p : UInt64} {n id off : Nat} {b : ByteArray}
    (h : d.range? p n = some (id, off, b)) : (d.bufs[id]?).join = some b ∧ off + n ≤ b.size := by
  unfold Dev.range? at h
  cases hp : devPtrOf p with
  | none => simp [hp] at h
  | some x =>
    obtain ⟨i, o⟩ := x
    cases hb : (d.bufs[i]?).join with
    | none => simp [hp, hb] at h
    | some bb =>
      simp only [hp, hb, Option.bind_eq_bind, Option.bind_some] at h
      split at h
      · rename_i hle
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl⟩ := h
        exact ⟨hb, hle⟩
      · cases h

theorem mapM_get_of_live {d : Dev} : ∀ {ids : List Nat},
    (∀ id ∈ ids, ((d.bufs[id]?).join).isSome = true) →
    (ids.mapM fun i => d.get? (Int.ofNat i)).isSome = true
  | [], _ => rfl
  | id :: ids, h => by
      obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (h id (List.mem_cons_self ..))
      have ih := mapM_get_of_live (ids := ids) (fun x hx => h x (List.mem_cons_of_mem _ hx))
      obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp ih
      have hg : d.get? (id : Int) = some b := by simp [Dev.get?, hb]
      simp only [Int.ofNat_eq_natCast] at hbs ⊢
      simp [List.mapM_cons, hg, hbs]

theorem runLaunch_isSome_nat {d : Dev} {k : Launch → List ByteArray → List ByteArray} (hk : KeepsSize k)
    (l : Launch) {ids : List Nat} (h : ∀ id ∈ ids, ((d.bufs[id]?).join).isSome = true) :
    (d.runLaunch k l ids).isSome = true := by
  obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp (mapM_get_of_live h)
  unfold Dev.runLaunch
  simp only [bind, Option.bind, hbs, keeps_ok hk, Bool.false_eq_true, if_false, Option.isSome_some]

theorem bind_some_isSome {α β : Type} {x : Option α} {g : α → β} (h : x.isSome = true) :
    (x.bind fun a => some (g a)).isSome = true := by
  cases x <;> simp_all

theorem mem_of_mem_eraseDups {α} [BEq α] {x : α} : ∀ {l : List α}, x ∈ l.eraseDups → x ∈ l
  | [], h => by simp at h
  | a :: as, h => by
      rw [List.eraseDups_cons] at h
      rcases List.mem_cons.mp h with rfl | h
      · exact List.mem_cons_self ..
      · exact List.mem_cons_of_mem _ ((List.mem_filter.mp (mem_of_mem_eraseDups h)).1)
termination_by l => l.length
decreasing_by exact Nat.lt_succ_of_le (List.length_filter_le _ _)

theorem mem_launch_ids {d : Dev} {args : List UInt64} {id : Nat}
    (h : id ∈ (args.filterMap fun a => (d.range? a 0).map (·.1)).eraseDups) :
    ∃ a ∈ args, ∃ off b, d.range? a 0 = some (id, off, b) := by
  have h := mem_of_mem_eraseDups h
  obtain ⟨a, ha, e⟩ := List.mem_filterMap.mp h
  obtain ⟨⟨i, off, b⟩, hr, rfl⟩ := Option.map_eq_some_iff.mp e
  exact ⟨a, ha, off, b, hr⟩

theorem handleOf_stream_zero : handleOf kStream 0 = none := by decide

/-- A device operation answers when it is not a race in the order it is
    checked against, and, if it runs now, running it answers. -/
theorem raceFor_of_not_capturing {d : Dev} {p : Nat} (h : d.capturing p = false) :
    d.raceFor p = d.race := by
  unfold Dev.raceFor
  cases hc : d.capture with
  | none => rfl
  | some c => simp [h]

theorem devOp_isSome_for {d : Dev} {w : World} {p : Nat} {op : DevOp} {rs ws : List Nat}
    (hr : OpReady (d.raceFor p) p rs ws)
    (hrun : d.capturing p = false → (Dev.devOp.run d w p op rs ws).isSome = true) :
    (d.devOp w p op rs ws).isSome = true := by
  unfold Dev.devOp
  cases hc : d.capture with
  | none => exact hrun (by simp [Dev.capturing, hc])
  | some c =>
      by_cases hp : d.capturing p = true
      · have hr' : OpReady c.race p rs ws := by simpa [Dev.raceFor, hc, hp] using hr
        obtain ⟨r, e⟩ := Option.isSome_iff_exists.mp hr'.op
        simp [hp, e]
      · have hp' : d.capturing p = false := by simpa using hp
        simp only [hp', Bool.false_eq_true, if_false]
        exact hrun hp'

/-- Running an operation now answers when it is not a race and its routine
    or kernel answers on the device after it is recorded. -/
theorem devOp_run_isSome {d : Dev} {w : World} {p : Nat} {op : DevOp} {rs ws : List Nat}
    (hr : OpReady d.race p rs ws)
    (hrun : ∀ r, (match op with
        | .launch l ids => ({ d with race := r } : Dev).runLaunch w.kernel l ids
        | .vendor c ins out => ({ d with race := r } : Dev).runVendor w.vendor c ins out).isSome = true) :
    (Dev.devOp.run d w p op rs ws).isSome = true := by
  unfold Dev.devOp.run
  obtain ⟨r, e⟩ := Option.isSome_iff_exists.mp hr.op
  simp only [e, bind, Option.bind]
  have := hrun r
  cases op <;> exact this

set_option maxHeartbeats 4000000

/-- The driver's preconditions suffice, capture aside. -/
theorem cudaStepPre_safe (f : CudaFn) (bits : List UInt64) (w : World) (h : cudaStepPre f bits w) :
    (cudaStep f bits w).isSome = true := by
  revert h
  cases f <;> simp only [cudaStepPre] <;> split <;> intro h <;> simp only [cudaStep]
  all_goals try exact False.elim h
  -- init
  · split <;> rfl
  -- deviceGet
  · rename_i pdev ord
    obtain ⟨hi, hs⟩ := h
    simp only [hi, Bool.not_true, Bool.false_eq_true, if_false]
    split
    · rfl
    · obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs (ord &&& 0xffffffff))
      simp [hm]
  -- primaryCtxRetain
  · obtain ⟨hi, hd, hs⟩ := h
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs cudaCtx)
    simp [hi, hd, hm]
  -- primaryCtxRelease
  · obtain ⟨hi, hd, hr⟩ := h
    simp only [hi, hd, bne_self_eq_false, Bool.not_true, Bool.false_or, Bool.or_false]
    split
    · rename_i hc; simp at hc; omega
    · split <;> rfl
  -- ctxSetCurrent
  · rcases h with rfl | ⟨rfl, hr⟩
    · rfl
    · simp [cudaCtx_ne_zero, hr]
  -- ctxSynchronize
  · simp [h, devOnly_isSome, cuRes]
  -- memGetInfo
  · obtain ⟨hl, hf, ht⟩ := h
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hf w.memInfo.1)
    obtain ⟨m', hm'⟩ := Option.isSome_iff_exists.mp (slot8_after_store hm ht w.memInfo.2)
    simp [hl, hm, hm']
  -- memAlloc
  · obtain ⟨hl, hs⟩ := h
    simp only [hl, Bool.not_true, Bool.false_eq_true, if_false]
    split
    · rfl
    · obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs (devPtr w.dev.bufs.size))
      simp [hm]
  -- memFree
  · obtain ⟨hl, id, b, hr, hready⟩ := h
    obtain ⟨r, hop⟩ := Option.isSome_iff_exists.mp (op_isSome_w hready)
    simp [hl, hr, hop, devOnly_isSome, cuRes]
  -- memcpyHtoD
  · obtain ⟨hl, ⟨id, off, b, hr, hready⟩, hread⟩ := h
    obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp hread
    obtain ⟨d', hd'⟩ := Option.isSome_iff_exists.mp (syncWrite_isSome (overwrite b off bs) hready)
    simp [hl, hr, hbs, hd', devOnly_isSome, cuRes]
  -- memcpyDtoH
  · rename_i dst src n
    obtain ⟨hl, ⟨id, off, b, hr, hready⟩, hwr⟩ := h
    obtain ⟨d', hd'⟩ := Option.isSome_iff_exists.mp (syncRead_isSome hready)
    have hsz : (b.extract off (off + n.toNat)).size = n.toNat := by
      have := (range?_live hr).2
      simp only [ByteArray.size_extract]; omega
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hwr _ hsz)
    simp [hl, hr, hd', hm]
  -- memcpyDtoD
  · obtain ⟨hl, ⟨sid, soff, sb, hsr, hsready⟩, ⟨did, doff, db, hdr, hdready⟩⟩ := h
    obtain ⟨r, hop⟩ := Option.isSome_iff_exists.mp (op_isSome_rw hsready hdready)
    simp [hl, hsr, hdr, hop, (range?_live hdr).1, devOnly_isSome, cuRes]
  -- memsetD8
  · obtain ⟨hl, ⟨id, off, b, hr, hready⟩⟩ := h
    obtain ⟨r, hop⟩ := Option.isSome_iff_exists.mp (op_isSome_w hready)
    simp [hl, hr, hop, devOnly_isSome, cuRes]
  -- moduleLoadData
  · obtain ⟨hl, hc, hs⟩ := h
    obtain ⟨ptx, hptx⟩ := Option.isSome_iff_exists.mp hc
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs (drvHandle kModule w.dev.modules.size))
    simp [hl, readCStrAt, hptx, hm]
  -- moduleGetFunction
  · obtain ⟨hl, ⟨mi, ptx, hh, hmod⟩, hc, hs⟩ := h
    obtain ⟨name, hname⟩ := Option.isSome_iff_exists.mp hc
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs (drvHandle kFunc w.dev.funcs.size))
    by_cases hn : ptxHasEntry ptx name = true <;> simp [hl, hh, hmod, readCStrAt, hname, hn, hm]
  -- moduleUnload
  · obtain ⟨hl, ⟨mi, ptx, hh, hmod⟩⟩ := h
    simp [hl, hh, hmod]
  -- launchKernel
  · obtain ⟨hl, he, hk, hlo⟩ := h
    obtain ⟨fi, mi, entry, ptx, p, sizes, args, h1, h2, h3, h4, h5, h6, hr⟩ := hlo
    simp only [hl, he, Bool.not_true, bne_self_eq_false, Bool.or_false, Bool.false_eq_true, if_false,
      h1, h2, h3, h4, h5, h6, Option.bind_eq_bind, Option.bind_some, Option.join_some]
    split
    · rfl
    · rw [devOnly_isSome]
      simp only [cuRes, Option.bind_eq_bind]
      have hready : OpReady (w.dev.raceFor p) p []
          (args.filterMap fun a => (w.dev.range? a 0).map (·.1)).eraseDups := by
        refine ⟨fun b hb => ?_, fun b hb => ?_⟩
        · simp only [List.nil_append] at hb
          obtain ⟨a, ha, off, bb, hra⟩ := mem_launch_ids hb
          exact (hr a ha b off bb hra).1
        · obtain ⟨a, ha, off, bb, hra⟩ := mem_launch_ids hb
          exact (hr a ha b off bb hra).2 rfl
      refine bind_some_isSome (devOp_isSome_for hready fun hp => ?_)
      rw [raceFor_of_not_capturing hp] at hready
      refine devOp_run_isSome hready fun r => ?_
      refine runLaunch_isSome_nat hk _ ?_
      intro id hid
      obtain ⟨a, _, off, b, hra⟩ := mem_launch_ids hid
      simp [(range?_live hra).1]
  -- streamCreate
  · obtain ⟨hl, hf, hs⟩ := h
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs (drvHandle kStream w.dev.streams.size))
    simp [hl, hf, hm]
  -- streamSynchronize
  · obtain ⟨hl, hcap, hs⟩ := h
    obtain ⟨p, hp⟩ := Option.isSome_iff_exists.mp hs
    simp [hl, hp, Dev.capturing, hcap, devOnly_isSome, cuRes]
  -- streamDestroy
  · rename_i hs
    obtain ⟨hl, id, hid, hlive⟩ := h
    have hne : hs ≠ 0 := by
      rintro rfl; rw [handleOf_stream_zero] at hid; cases hid
    simp [hl, hne, hid]
    simpa using hlive
  -- eventCreate
  · obtain ⟨hl, hf, hs⟩ := h
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs (drvHandle kEvent w.dev.events.size))
    simp [hl, hf, hm]
  -- eventRecord
  · obtain ⟨hl, he, hs⟩ := h
    obtain ⟨⟨id, ev⟩, hev⟩ := Option.isSome_iff_exists.mp he
    obtain ⟨p, hp⟩ := Option.isSome_iff_exists.mp hs
    simp [hl, hev, hp, devOnly_isSome, cuRes]
  -- streamWaitEvent
  · obtain ⟨hl, hf, p, id, ev, hp, hev, hw⟩ := h
    obtain ⟨d', hd'⟩ := Option.isSome_iff_exists.mp hw
    simp [hl, hf, hp, hev, hd', devOnly_isSome, cuRes]
  -- eventSynchronize
  · obtain ⟨hl, id, ev, hev, hnc⟩ := h
    rcases ev with _ | ⟨clk, _ | _⟩
    · simp [hl, hev, devOnly_isSome, cuRes]
    · simp [hl, hev, devOnly_isSome, cuRes]
    · exact absurd rfl (hnc _)
  -- eventDestroy
  · obtain ⟨hl, he⟩ := h
    obtain ⟨⟨id, ev⟩, hev⟩ := Option.isSome_iff_exists.mp he
    simp [hl, hev]
  -- eventElapsedTime
  · obtain ⟨hl, hs4, si, ei, cs, ce, h1, h2, h3, h4⟩ := h
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs4 (w.elapsed si ei).toUInt64)
    simp [hl, h1, h2, h3, h4, hm]
  -- memAllocHost
  · obtain ⟨hl, hz, hroom, hs⟩ := h
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs (addrOf .pinned w.mem.pinnedNext))
    simp [hl, hz, hroom, hm]
  -- memFreeHost
  · obtain ⟨hl, id, off, n, ha, hb⟩ := h
    simp [hl, ha]
    exact hb
  -- memcpyHtoDAsync
  · obtain ⟨hl, hcap, q, hq, ⟨id, off, b, hr, hready⟩, hread⟩ := h
    obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp hread
    obtain ⟨r, hop⟩ := Option.isSome_iff_exists.mp (op_isSome_w hready)
    simp [hl, hq, hr, hbs, asyncCopy, hcap, hop]
  -- memcpyDtoHAsync
  · rename_i dst src n hs
    obtain ⟨hl, hcap, q, hq, ⟨id, off, b, hr, hready⟩, hwr⟩ := h
    obtain ⟨r, hop⟩ := Option.isSome_iff_exists.mp (op_isSome_r hready)
    have hsz : (b.extract off (off + n.toNat)).size = n.toNat := by
      have := (range?_live hr).2
      simp only [ByteArray.size_extract]; omega
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hwr _ hsz)
    simp [hl, hq, hr, asyncCopy, hcap, hop, hm]
  -- beginCapture
  · obtain ⟨hl, hm, hcap, p, hp, hd⟩ := h
    simp [hl, hm, hcap, hp, hd, devOnly_isSome, cuRes]
  -- endCapture
  · obtain ⟨hl, hs, p, c, hp, hc, ho⟩ := h
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs (drvHandle kGraph w.dev.graphs.size))
    by_cases hj : c.rejoined = true
    · simp [hl, hp, hc, ho, hj, hm]
    · simp [hl, hp, hc, ho, hj, devOnly_isSome, cuRes]
  -- graphInstantiate
  · obtain ⟨hl, hf, ⟨i, ops, hi, hg⟩, hs⟩ := h
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs (drvHandle kExec w.dev.execs.size))
    simp [hl, hf, hi, hg, hm]
  -- graphLaunch
  · obtain ⟨hl, i, ops, p, hi, he, hp, hrun⟩ := h
    obtain ⟨d', hd'⟩ := Option.isSome_iff_exists.mp hrun
    simp [hl, hi, he, hp, hd', devOnly_isSome, cuRes]
  -- graphExecDestroy
  · obtain ⟨hl, i, ops, hi, he⟩ := h
    simp [hl, hi, he]
  -- graphDestroy
  · obtain ⟨hl, i, ops, hi, hg⟩ := h
    simp [hl, hi, hg]

theorem operand?_live {d : Dev} {p : UInt64} {e s id o : Nat} (h : d.operand? p e s = some (id, o)) :
    ∃ b, (d.bufs[id]?).join = some b := by
  unfold Dev.operand? at h
  cases hr : d.range? p (e * s) with
  | none => simp [hr] at h
  | some x =>
    obtain ⟨i, off, b⟩ := x
    simp only [hr, Option.bind_eq_bind, Option.bind_some] at h
    split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -⟩ := h
      exact ⟨b, (range?_live hr).1⟩
    · cases h

/-- A routine over three live allocations, ready on its stream, answers. -/
theorem vendor3_isSome {w : World} {p ia ib ic : Nat} {c : VendorCall}
    (hv : VendorKeeps w.vendor) (hr : OpReady (w.dev.raceFor p) p [ia, ib, ic] [ic])
    (ha : ∃ b, (w.dev.bufs[ia]?).join = some b) (hb : ∃ b, (w.dev.bufs[ib]?).join = some b)
    (hc : ∃ b, (w.dev.bufs[ic]?).join = some b) :
    (devOnly w (do cuRes (← w.dev.devOp w p (.vendor c [ia, ib, ic] ic) [ia, ib, ic] [ic]) blasOk)).isSome
      = true := by
  obtain ⟨A, hA⟩ := ha
  obtain ⟨B, hB⟩ := hb
  obtain ⟨C, hC⟩ := hc
  rw [devOnly_isSome]
  have hnn : ∀ i : Nat, ¬ ((i : Int) < 0) := fun i => by omega
  have hins : [ia, ib, ic].mapM (fun i => w.dev.get? (Int.ofNat i)) = some [A, B, C] := by
    simp [Dev.get?, hA, hB, hC, hnn]
  have hout : w.dev.get? (Int.ofNat ic) = some C := by simp [Dev.get?, hC]
  simp only [cuRes, Option.bind_eq_bind]
  refine bind_some_isSome (devOp_isSome_for hr fun hp => ?_)
  rw [raceFor_of_not_capturing hp] at hr
  exact devOp_run_isSome hr fun r => runVendor_isSome (d := { w.dev with race := r }) hv hins hout rfl

theorem gemmCall_isSome {w : World} {op : String} {ea eb ec : Nat}
    {h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch : UInt64}
    (hv : VendorKeeps w.vendor)
    (hg : GemmRuns w ea eb ec h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch) :
    (gemmCall w op ea eb ec h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch).isSome = true := by
  obtain ⟨p, hp, hrun⟩ := hg
  unfold gemmCall
  cases hi : gemmInvalid ta tb m n k lda ldb ldc batch
  · obtain ⟨hun, ia, oa, ib, ob, ic, oc, hA, hB, hC, hca, hcb, hpa, hpb, hr⟩ := hrun hi
    obtain ⟨al, hal⟩ := Option.isSome_iff_exists.mp hpa
    obtain ⟨be, hbe⟩ := Option.isSome_iff_exists.mp hpb
    have e1 : (ic == ia || ic == ib) = false := by simp [hca, hcb]
    simp only [hp, hun, hA, hB, hC, e1, hal, hbe, Option.bind_eq_bind, Option.bind_some,
      Bool.false_eq_true, if_false]
    exact vendor3_isSome hv hr (operand?_live hA) (operand?_live hB) (operand?_live hC)
  · simp [hp, hi]

theorem sgemvCall_isSome {w : World} {h trans m n pa A lda x incx pb y incy : UInt64}
    (hv : VendorKeeps w.vendor)
    (hg : SgemvRuns w h trans m n pa A lda x incx pb y incy) :
    (sgemvCall w h trans m n pa A lda x incx pb y incy).isSome = true := by
  obtain ⟨p, hp, hrun⟩ := hg
  unfold sgemvCall
  cases hi : sgemvInvalid trans m n lda incx incy
  · obtain ⟨hun, ia, ix, iy, hA, hX, hY, hya, hyx, hpa, hpb, hr⟩ := hrun hi
    obtain ⟨al, hal⟩ := Option.isSome_iff_exists.mp hpa
    obtain ⟨be, hbe⟩ := Option.isSome_iff_exists.mp hpb
    have e1 : ((0 : Nat) != 0 || (0 : Nat) != 0 || (0 : Nat) != 0 || iy == ia || iy == ix) = false := by
      simp [hya, hyx]
    simp only [hp, hi, hun, hA, hX, hY, e1, hal, hbe, Option.bind_eq_bind, Option.bind_some,
      Bool.false_eq_true, if_false]
    exact vendor3_isSome hv hr (operand?_live hA) (operand?_live hX) (operand?_live hY)
  · simp [hp, hi]

/-- A routine over three live allocations, ready on its stream, is a device
    operation that answers, whatever else it reads. -/
theorem vendorOp_isSome {d : Dev} {w : World} {p ia ib ic : Nat} {c : VendorCall} {rs : List Nat}
    (hv : VendorKeeps w.vendor) (hr : OpReady (d.raceFor p) p rs [ic])
    (ha : ∃ b, (d.bufs[ia]?).join = some b) (hb : ∃ b, (d.bufs[ib]?).join = some b)
    (hc : ∃ b, (d.bufs[ic]?).join = some b) :
    (d.devOp w p (.vendor c [ia, ib, ic] ic) rs [ic]).isSome = true := by
  obtain ⟨A, hA⟩ := ha
  obtain ⟨B, hB⟩ := hb
  obtain ⟨C, hC⟩ := hc
  have hnn : ∀ i : Nat, ¬ ((i : Int) < 0) := fun i => by omega
  have hins : [ia, ib, ic].mapM (fun i => d.get? (Int.ofNat i)) = some [A, B, C] := by
    simp [Dev.get?, hA, hB, hC, hnn]
  have hout : d.get? (Int.ofNat ic) = some C := by simp [Dev.get?, hC]
  refine devOp_isSome_for hr fun hp => ?_
  rw [raceFor_of_not_capturing hp] at hr
  exact devOp_run_isSome hr fun r => runVendor_isSome (d := { d with race := r }) hv hins hout rfl

theorem batchOps_isSome {w : World} {p : Nat} {ta tb m n k al be : UInt64} {arrs : List Nat}
    (hv : VendorKeeps w.vendor) :
    ∀ (ms : List ((Nat × Nat) × (Nat × Nat) × (Nat × Nat))) (d : Dev),
      BatchRuns w p ta tb m n k al be arrs ms d →
      (batchOps w p ta tb m n k al be arrs ms d).isSome = true
  | [], _, _ => rfl
  | (a, b, c) :: ms, d, ⟨ha, hb, hc, hr, hk⟩ => by
      obtain ⟨d', e⟩ := Option.isSome_iff_exists.mp (vendorOp_isSome hv hr ha hb hc)
      unfold batchOps
      rw [e]
      exact batchOps_isSome hv ms d' (hk d' e)

theorem sgemmBatchedCall_isSome {w : World} {h ta tb m n k pa PA lda PB ldb pb PC ldc batch : UInt64}
    (hv : VendorKeeps w.vendor) (hg : SgemmBatchedRuns w h ta tb m n k pa PA lda PB ldb pb PC ldc batch) :
    (sgemmBatchedCall w h ta tb m n k pa PA lda PB ldb pb PC ldc batch).isSome = true := by
  obtain ⟨p, hp, hrun⟩ := hg
  unfold sgemmBatchedCall
  cases hi : gemmInvalid ta tb m n k lda ldb ldc batch
  · obtain ⟨hun, iA, oA, bA, iB, oB, bB, iC, oC, bC, ms, al, be, hA, hB, hC, hms, hap, hal, hbe,
      hrun⟩ := hrun hi
    obtain ⟨d', e⟩ := Option.isSome_iff_exists.mp (batchOps_isSome hv ms w.dev hrun)
    simp [hp, hun, hA, hB, hC, hms, hap, hal, hbe, e, devOnly, cuRes]
  · simp [hp, hi]

/-- **cuBLAS's preconditions suffice.** -/
theorem cublasPre_safe (f : CublasFn) (bits : List UInt64) (w : World) (h : cublasPre f bits w) :
    (cublasCall f bits w).isSome = true := by
  revert h
  unfold cublasPre
  split <;> intro h <;> simp only [cublasCall]
  -- create
  · obtain ⟨hl, hc, hs⟩ := h
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs (drvHandle kBlas w.dev.blas.size))
    simp [hl, hc, hm]
  -- destroy
  · obtain ⟨hl, hc, i, s, hi, hb⟩ := h
    simp [hl, hc, hi, hb]
  -- setStream
  · obtain ⟨hl, hc, ⟨i, s', hi, hb⟩, hs⟩ := h
    obtain ⟨p, hp⟩ := Option.isSome_iff_exists.mp hs
    simp [hl, hc, hi, hb, hp]
  -- sgemv
  · obtain ⟨hl, hv, hr⟩ := h
    simp only [hl, Bool.not_true, Bool.false_eq_true, if_false]
    exact sgemvCall_isSome hv hr
  -- sgemm, sgemmStridedBatched
  iterate 2
    · obtain ⟨hl, hv, hr⟩ := h
      simp only [hl, Bool.not_true, Bool.false_eq_true, if_false]
      exact gemmCall_isSome hv hr
  -- gemmEx, gemmStridedBatchedEx
  iterate 2
    · obtain ⟨hl, hv, hb, hr⟩ := h
      simp only [hl, hb, Bool.not_true, Bool.or_false, Bool.false_eq_true, if_false]
      exact gemmCall_isSome hv hr
  -- sgemmBatched
  · obtain ⟨hl, hv, hr⟩ := h
    simp only [hl, Bool.not_true, Bool.false_eq_true, if_false]
    exact sgemmBatchedCall_isSome hv hr
  · exact h.elim

/-- **The driver's preconditions suffice**: a call that meets its own and is
    not left without an answer by a capture in progress answers. -/
theorem cudaPre_safe (f : CudaFn) (bits : List UInt64) (w : World) (h : cudaPre f bits w) :
    (cudaDrv f bits w).isSome = true := by
  obtain ⟨hc, h⟩ := h
  unfold cudaDrv
  rw [if_neg (by rcases hc with hc | hc <;> simp [hc])]
  exact cudaStepPre_safe f bits w h

theorem cudaPre_length {f : CudaFn} {bits : List UInt64} {w : World} (h : cudaPre f bits w) :
    bits.length = (Ext.cuda f).sig.1.length := cudaStepPre_length h.2

/-- **Every contract's precondition suffices**: a call that meets it answers. -/
theorem ext_pre_safe (e : Ext) (bits : List UInt64) (w : World) (h : ExtPre e bits w) :
    (extCall e (e.args bits) w).isSome = true := by
  unfold extCall
  cases hp : w.has e.lib with
  | false => cases e <;> rfl
  | true =>
    have h := h hp
    simp only [Bool.not_true, Bool.false_eq_true, if_false]
    cases e with
    | present l =>
        match bits, h with
        | [], _ => rfl
    | c f =>
        cases f with
        | memcpy =>
            match bits, h with
            | [d, s, n], ⟨hr, hw, hdis⟩ =>
                simp only [Ext.args, bind, Option.bind]
                rw [mapM_asBits_zip _ _ rfl]
                simp only [cCall]
                obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp hr
                have hsz := copyOut_size hbs
                obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hw bs hsz)
                have hno : (decide (n.toNat > 0) && decide (d.toNat < s.toNat + n.toNat) &&
                    decide (s.toNat < d.toNat + n.toNat)) = false := by
                  simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]; omega
                simp [hno, readBytes] at hbs ⊢
                simp [hbs, hm]
        | memmove =>
            match bits, h with
            | [d, s, n], ⟨hr, hw⟩ =>
                simp only [Ext.args, bind, Option.bind]
                rw [mapM_asBits_zip _ _ rfl]
                simp only [cCall]
                obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp hr
                have hsz := copyOut_size hbs
                obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hw bs hsz)
                simp [readBytes] at hbs
                simp [hbs, hm]
        | memset =>
            match bits, h with
            | [d, c, n], hw =>
                simp only [Ext.args, bind, Option.bind]
                rw [mapM_asBits_zip _ _ rfl]
                simp only [cCall]
                obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp
                  (hw (ByteArray.mk (Array.replicate n.toNat (c &&& 0xff).toUInt8)) (by simp [ByteArray.size]))
                simp at hm
                simp [hm]
        | strlen =>
            match bits, h with
            | [a], ht =>
                simp only [Ext.args, bind, Option.bind]
                rw [mapM_asBits_zip _ _ rfl]
                simp only [cCall]
                obtain ⟨i, hi⟩ := Option.isSome_iff_exists.mp ht
                simp [hi]
        | calloc =>
            match bits, h with
            | [n, sz], hz =>
                simp only [Ext.args, bind, Option.bind]
                rw [mapM_asBits_zip _ _ rfl]
                simp only [cCall]
                have hc : (n.toNat * sz.toNat == 0) = false := by
                  simp only [beq_eq_false_iff_ne]; omega
                split
                · simp_all
                · split <;> simp
        | free =>
            match bits, h with
            | [p], h =>
                simp only [Ext.args, bind, Option.bind]
                rw [mapM_asBits_zip _ _ rfl]
                simp only [cCall]
                rcases h with rfl | ha
                · simp
                · by_cases hp : p = 0
                  · simp [hp]
                  · obtain ⟨⟨off, n⟩, ha⟩ := Option.isSome_iff_exists.mp ha
                    simp [hp, ha]
    | cuda f =>
        simp only [Ext.args, bind, Option.bind]
        rw [mapM_asBits_zip _ _ (cudaPre_length h)]
        exact cudaPre_safe f bits w h
    | cublas f =>
        simp only [Ext.args, bind, Option.bind]
        rw [mapM_asBits_zip _ _ (cublasPre_length h)]
        exact cublasPre_safe f bits w h
    | cpu f =>
        have hl : bits.length = (Ext.cpu f).sig.1.length := h
        simp only [Ext.args, bind, Option.bind]
        rw [mapM_asBits_zip _ _ hl]
        simp only [cpuCall, hl, ne_eq, not_true_eq_false, if_false, Option.isSome_some]
    | wgpu f =>
        simp only [Ext.args, bind, Option.bind]
        rw [mapM_asBits_zip _ _ h.1]
        simp only [wgpuCall, Option.isSome_map]; exact h.2
    | window f =>
        simp only [Ext.args, bind, Option.bind]
        rw [mapM_asBits_zip _ _ h.1]
        simp only [wlCall, Option.isSome_map]; exact h.2
    | serial f =>
        simp only [Ext.args, bind, Option.bind]
        rw [mapM_asBits_zip _ _ h.1]
        simp only [serialCall, Option.isSome_map]; exact h.2
    | usb f =>
        simp only [Ext.args, bind, Option.bind]
        rw [mapM_asBits_zip _ _ h.1]
        simp only [usbCall, Option.isSome_map]; exact h.2

-- ---------------------------------------------------------------------------
-- Answers respect the frames
-- ---------------------------------------------------------------------------

set_option hygiene false in
/-- An answer that leaves memory as it was. -/
macro "same_mem" : tactic => `(tactic|
  (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact Unchanged.refl _ _))

set_option hygiene false in
/-- An answer whose memory is one store at a slot the frame names. -/
macro "one_store" hm:ident : tactic => `(tactic|
  (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
   exact Unchanged.of_eq (Mem.store_other $hm a (fun i hi e => hout ⟨i, hi, e⟩))))

set_option maxHeartbeats 4000000

/-- A driver call writes host memory only inside its frame, capture aside. -/
theorem cudaStep_respects_frame (f : CudaFn) (bits : List UInt64) (w : World) (r : Option V)
    (w' : World) (h : cudaStep f bits w = some (r, w')) (a : UInt64)
    (hout : ¬ (Ext.cuda f).frame.allows bits (r.bind asBits) a) : Unchanged w.mem w'.mem a := by
  revert h hout
  cases f <;> simp only [cudaStep] <;> split <;> intro h hout
  all_goals try (cases h; done)
  all_goals try dsimp only at h
  all_goals
    simp only [Ext.frame, Frame.allows, List.getD, List.getElem?_cons_zero,
      List.getElem?_cons_succ, Option.getD_some] at hout
  -- init
  · split at h <;> same_mem
  -- deviceGet
  · split at h
    · cases h
    split at h
    · same_mem
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    one_store hm
  -- primaryCtxRetain
  · split at h
    · cases h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    one_store hm
  -- primaryCtxRelease: the last release drops pinned allocations, no byte
  · split at h
    · cases h
    split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
      intro x y hx hy
      exact Unchanged.refl w.mem a x y hx (Mem.load_shrink (by simp) hy)
    · same_mem
  -- ctxSetCurrent
  · split at h
    · same_mem
    split at h
    · same_mem
    · cases h
  -- ctxSynchronize
  · split at h
    · cases h
    exact Unchanged.of_region (devOnly_mem h) a
  -- memGetInfo
  · split at h
    · cases h
    obtain ⟨m1, hm1, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨m2, hm2, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    exact Unchanged.of_eq ((Mem.store_other hm2 a (fun i hi e => hout (Or.inr ⟨i, hi, e⟩))).trans
      (Mem.store_other hm1 a (fun i hi e => hout (Or.inl ⟨i, hi, e⟩))))
  -- memAlloc
  · split at h
    · cases h
    split at h
    · same_mem
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    one_store hm
  -- memFree
  · split at h
    · cases h
    obtain ⟨⟨id, off, b⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    split at h
    · cases h
    exact Unchanged.of_region (devOnly_mem h) a
  -- memcpyHtoD
  · split at h
    · cases h
    obtain ⟨⟨id, off, b⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨bs, -, h⟩ := Option.bind_eq_some_iff.mp h
    exact Unchanged.of_region (devOnly_mem h) a
  -- memcpyDtoH
  · rename_i dst src n
    split at h
    · cases h
    obtain ⟨⟨id, off, b⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨d', -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    apply Unchanged.of_eq
    apply copyIn_other hm
    intro i hi e
    simp only [ByteArray.size_extract] at hi
    exact hout ⟨i, by omega, e⟩
  -- memcpyDtoD
  · split at h
    · cases h
    obtain ⟨⟨sid, soff, sb⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨did, doff, db'⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨r', -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨db, -, h⟩ := Option.bind_eq_some_iff.mp h
    exact Unchanged.of_region (devOnly_mem h) a
  -- memsetD8
  · split at h
    · cases h
    obtain ⟨⟨id, off, b⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨r', -, h⟩ := Option.bind_eq_some_iff.mp h
    exact Unchanged.of_region (devOnly_mem h) a
  -- moduleLoadData
  · split at h
    · cases h
    obtain ⟨ptx, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    one_store hm
  -- moduleGetFunction
  · split at h
    · cases h
    obtain ⟨mi, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨ptx, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨name, -, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · same_mem
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    one_store hm
  -- moduleUnload
  · split at h
    · cases h
    obtain ⟨mi, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨ptx, -, h⟩ := Option.bind_eq_some_iff.mp h
    same_mem
  -- launchKernel
  · split at h
    · cases h
    obtain ⟨fi, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨mi, entry⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    obtain ⟨ptx, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · same_mem
    obtain ⟨sizes, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨args, -, h⟩ := Option.bind_eq_some_iff.mp h
    exact Unchanged.of_region (devOnly_mem h) a
  -- streamCreate
  · split at h
    · cases h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    one_store hm
  -- streamSynchronize
  · split at h
    · cases h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · cases h
    exact Unchanged.of_region (devOnly_mem h) a
  -- streamDestroy
  · split at h
    · cases h
    obtain ⟨id, -, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · cases h
    same_mem
  -- eventCreate
  · split at h
    · cases h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    one_store hm
  -- eventRecord
  · split at h
    · cases h
    obtain ⟨⟨id, ev⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    exact Unchanged.of_region (devOnly_mem h) a
  -- streamWaitEvent
  · split at h
    · cases h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨id, ev⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    obtain ⟨d', -, h⟩ := Option.bind_eq_some_iff.mp h
    exact Unchanged.of_region (devOnly_mem h) a
  -- eventSynchronize
  · split at h
    · cases h
    obtain ⟨⟨id, ev⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    split at h
    · exact Unchanged.of_region (devOnly_mem h) a
    · exact Unchanged.of_region (devOnly_mem h) a
    · cases h
  -- eventDestroy
  · split at h
    · cases h
    obtain ⟨⟨id, ev⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    same_mem
  -- eventElapsedTime
  · split at h
    · cases h
    obtain ⟨⟨si, sev⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨ei, eev⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    split at h
    · split at h
      · obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
        one_store hm
      · cases h
    · same_mem
    · same_mem
    · cases h
  -- memAllocHost
  · split at h
    · cases h
    split at h
    · cases h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    intro x y hx hy
    have h1 : m.load a 1 = some x := (Mem.store_other hm a (fun i hi e => hout ⟨i, hi, e⟩)).trans hx
    rw [Mem.load_grow _ _ h1] at hy
    exact Option.some.inj hy
  -- memFreeHost
  · split at h
    · cases h
    obtain ⟨⟨id, off, n⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    dsimp only at h
    split at h
    · cases h
    simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    intro x y hx hy
    rw [Mem.load_shrink (fun r hr => List.mem_of_mem_erase hr) hy] at hx
    exact (Option.some.inj hx).symm
  -- memcpyHtoDAsync
  · split at h
    · cases h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨id, off, b⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨bs, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨_, _, m, hk, hreg, -, -, -⟩ := asyncCopy_region h
    simp only [Option.some.injEq, Prod.mk.injEq] at hk
    obtain ⟨-, rfl⟩ := hk
    exact Unchanged.of_region hreg a
  -- memcpyDtoHAsync
  · split at h
    · cases h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨id, off, b⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨_, _, m, hk, hreg, hlive, hfz, hsub⟩ := asyncCopy_region h
    obtain ⟨m0, hm0, hk⟩ := Option.bind_eq_some_iff.mp hk
    simp only [Option.some.injEq, Prod.mk.injEq] at hk
    obtain ⟨-, rfl⟩ := hk
    obtain ⟨hl0, hb0, hf0⟩ := copyIn_meta hm0
    intro x y hx hy
    have hy0 : m0.load a 1 = some y :=
      Mem.load_congr hreg (Mem.readable_busy_sub (hlive.trans rfl) hfz
        (fun b hb => hsub b (by
          rw [hb0] at hb
          exact (List.mem_filter.mp hb).1))) hy
    rw [copyIn_other hm0 a (by
      intro i hi e
      rw [ByteArray.size_extract] at hi
      exact hout ⟨i, by omega, e⟩)] at hy0
    exact Mem.load_agree (m := w.mem) (m' := w.mem.forParty _) (fun reg => by cases reg <;> rfl)
      hx hy0
  -- beginCapture
  · split at h
    · cases h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · cases h
    exact Unchanged.of_region (devOnly_mem h) a
  -- endCapture
  · split at h
    · cases h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨c, -, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · cases h
    split at h
    · exact Unchanged.of_region (devOnly_mem h) a
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    one_store hm
  -- graphInstantiate
  · split at h
    · cases h
    obtain ⟨gi, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨ops, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    one_store hm
  -- graphLaunch
  · split at h
    · cases h
    obtain ⟨ei, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨ops, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨p, -, h⟩ := Option.bind_eq_some_iff.mp h
    exact Unchanged.of_region (devOnly_mem h) a
  -- graphExecDestroy, graphDestroy
  iterate 2
    · split at h
      · cases h
      obtain ⟨i, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      same_mem

/-- **A driver call writes host memory only inside its frame.** -/
theorem cudaDrv_respects_frame (f : CudaFn) (bits : List UInt64) (w : World) (r : Option V)
    (w' : World) (h : cudaDrv f bits w = some (r, w')) (a : UInt64)
    (hout : ¬ (Ext.cuda f).frame.allows bits (r.bind asBits) a) : Unchanged w.mem w'.mem a := by
  unfold cudaDrv at h
  split at h
  · cases h
  · exact cudaStep_respects_frame f bits w r w' h a hout

theorem gemmCall_mem {w : World} {op : String} {ea eb ec : Nat}
    {h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch : UInt64} {r : Option V} {w' : World}
    (e : gemmCall w op ea eb ec h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch = some (r, w')) :
    ∀ reg, w'.mem.region reg = w.mem.region reg := by
  unfold gemmCall at e
  obtain ⟨p, -, e⟩ := Option.bind_eq_some_iff.mp e
  split at e
  · simp only [Option.some.injEq, Prod.mk.injEq] at e; obtain ⟨-, rfl⟩ := e; exact fun _ => rfl
  split at e
  · cases e
  obtain ⟨⟨ia, oa⟩, -, e⟩ := Option.bind_eq_some_iff.mp e
  obtain ⟨⟨ib, ob⟩, -, e⟩ := Option.bind_eq_some_iff.mp e
  obtain ⟨⟨ic, oc⟩, -, e⟩ := Option.bind_eq_some_iff.mp e
  try dsimp only at e
  split at e
  · cases e
  obtain ⟨al, -, e⟩ := Option.bind_eq_some_iff.mp e
  obtain ⟨be, -, e⟩ := Option.bind_eq_some_iff.mp e
  exact devOnly_mem e

theorem sgemvCall_mem {w : World} {h trans m n pa A lda x incx pb y incy : UInt64} {r : Option V}
    {w' : World} (e : sgemvCall w h trans m n pa A lda x incx pb y incy = some (r, w')) :
    ∀ reg, w'.mem.region reg = w.mem.region reg := by
  unfold sgemvCall at e
  obtain ⟨p, -, e⟩ := Option.bind_eq_some_iff.mp e
  split at e
  · simp only [Option.some.injEq, Prod.mk.injEq] at e; obtain ⟨-, rfl⟩ := e; exact fun _ => rfl
  split at e
  · cases e
  obtain ⟨⟨ia, oa⟩, -, e⟩ := Option.bind_eq_some_iff.mp e
  obtain ⟨⟨ix, ox⟩, -, e⟩ := Option.bind_eq_some_iff.mp e
  obtain ⟨⟨iy, oy⟩, -, e⟩ := Option.bind_eq_some_iff.mp e
  try dsimp only at e
  split at e
  · cases e
  obtain ⟨al, -, e⟩ := Option.bind_eq_some_iff.mp e
  obtain ⟨be, -, e⟩ := Option.bind_eq_some_iff.mp e
  exact devOnly_mem e

theorem sgemmBatchedCall_mem {w : World} {h ta tb m n k pa PA lda PB ldb pb PC ldc batch : UInt64}
    {r : Option V} {w' : World}
    (e : sgemmBatchedCall w h ta tb m n k pa PA lda PB ldb pb PC ldc batch = some (r, w')) :
    ∀ reg, w'.mem.region reg = w.mem.region reg := by
  unfold sgemmBatchedCall at e
  obtain ⟨p, -, e⟩ := Option.bind_eq_some_iff.mp e
  split at e
  · simp only [Option.some.injEq, Prod.mk.injEq] at e; obtain ⟨-, rfl⟩ := e; exact fun _ => rfl
  split at e
  · cases e
  obtain ⟨⟨iA, oA, bA⟩, -, e⟩ := Option.bind_eq_some_iff.mp e
  obtain ⟨⟨iB, oB, bB⟩, -, e⟩ := Option.bind_eq_some_iff.mp e
  obtain ⟨⟨iC, oC, bC⟩, -, e⟩ := Option.bind_eq_some_iff.mp e
  obtain ⟨ms, -, e⟩ := Option.bind_eq_some_iff.mp e
  try dsimp only at e
  split at e
  · cases e
  obtain ⟨al, -, e⟩ := Option.bind_eq_some_iff.mp e
  obtain ⟨be, -, e⟩ := Option.bind_eq_some_iff.mp e
  exact devOnly_mem e

/-- **A cuBLAS call writes host memory only inside its frame**: a handle's
    creation writes its slot, and every routine writes only the device. -/
theorem cublasCall_respects_frame (f : CublasFn) (bits : List UInt64) (w : World) (r : Option V)
    (w' : World) (h : cublasCall f bits w = some (r, w')) (a : UInt64)
    (hout : ¬ (Ext.cublas f).frame.allows bits (r.bind asBits) a) : Unchanged w.mem w'.mem a := by
  revert h hout
  unfold cublasCall
  split <;> intro h hout <;> dsimp only at h <;>
    simp only [Ext.frame, Frame.allows, List.getD, List.getElem?_cons_zero,
      List.getElem?_cons_succ, Option.getD_some] at hout
  -- create
  · split at h
    · cases h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    one_store hm
  -- destroy
  · split at h
    · cases h
    obtain ⟨i, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    same_mem
  -- setStream
  · split at h
    · cases h
    obtain ⟨i, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    same_mem
  -- sgemv
  · split at h
    · cases h
    exact Unchanged.of_region (sgemvCall_mem h) a
  -- the products
  iterate 4
    · split at h
      · cases h
      exact Unchanged.of_region (gemmCall_mem h) a
  -- sgemmBatched
  · split at h
    · cases h
    exact Unchanged.of_region (sgemmBatchedCall_mem h) a
  · cases h

theorem copyIn_getD_other {m : Mem} {a : UInt64} {src : ByteArray} (a' : UInt64)
    (hout : ∀ i < src.size, a' ≠ a + UInt64.ofNat i) :
    ((copyIn m a src).getD m).load a' 1 = m.load a' 1 := by
  cases h : copyIn m a src with
  | none => rfl
  | some m' => exact copyIn_other h a' hout

theorem store_getD_other {m : Mem} {a : UInt64} {n : Nat} {v : UInt64} (a' : UInt64)
    (hout : ∀ i < n, a' ≠ a + UInt64.ofNat i) : ((m.store a n v).getD m).load a' 1 = m.load a' 1 := by
  cases h : m.store a n v with
  | none => rfl
  | some m' => exact Mem.store_other h a' hout

/-- **A wgpu call writes host memory only inside its frame**: the one that
    writes any, a buffer read, writes at most the bytes it is asked for. -/
theorem wgpuCall_respects_frame (f : WgpuFn) (bits : List UInt64) (w : World) (r : Option V)
    (w' : World) (h : wgpuCall f bits w = some (r, w')) (a : UInt64)
    (hout : ¬ (Ext.wgpu f).frame.allows bits (r.bind asBits) a) : Unchanged w.mem w'.mem a := by
  simp only [wgpuCall] at h
  obtain ⟨c, -, h⟩ := Option.map_eq_some_iff.mp h
  apply Unchanged.of_eq
  split at h
  · simp only [Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    simp only [Ext.frame, Frame.allows, List.getD, List.getElem?_cons_zero,
      List.getElem?_cons_succ, Option.getD_some] at hout
    apply copyIn_getD_other
    intro i hi e
    refine hout ⟨i, ?_, e⟩
    split at hi
    · simp only [ByteArray.size_extract] at hi; omega
    · simp at hi
  · rw [show w' = (wgEffect c w).2 by rw [h], wgEffect_mem]

/-- **A CPU library call writes no host memory.** -/
theorem cpuCall_respects_frame (f : CpuFn) (bits : List UInt64) (w : World) (r : Option V)
    (w' : World) (h : cpuCall f bits w = some (r, w')) (a : UInt64) : Unchanged w.mem w'.mem a := by
  simp only [cpuCall] at h
  split at h
  · cases h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    exact Unchanged.refl _ _

/-- **A serial library call writes host memory only inside its frame**: a
    port's name, and what a read takes, into the buffer and length handed in. -/
theorem serialCall_respects_frame (f : SerialFn) (bits : List UInt64) (w : World) (r : Option V)
    (w' : World) (h : serialCall f bits w = some (r, w')) (a : UInt64)
    (hout : ¬ (Ext.serial f).frame.allows bits (r.bind asBits) a) : Unchanged w.mem w'.mem a := by
  simp only [serialCall] at h
  obtain ⟨c, -, h⟩ := Option.map_eq_some_iff.mp h
  apply Unchanged.of_eq
  split at h
  · simp only [Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    simp only [Ext.frame, Frame.allows, List.getD, List.getElem?_cons_zero,
      List.getElem?_cons_succ, Option.getD_some] at hout
    exact copyIn_getD_other a fun i hi e => hout ⟨i, Nat.lt_of_lt_of_le hi (serNameBytes_size ..), e⟩
  · simp only [Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    simp only [Ext.frame, Frame.allows, List.getD, List.getElem?_cons_zero,
      List.getElem?_cons_succ, Option.getD_some] at hout
    exact copyIn_getD_other a fun i hi e => hout ⟨i, Nat.lt_of_lt_of_le hi (serReadBytes_size ..), e⟩
  · rw [show w' = (serEffect c w).2 by rw [h], serEffect_mem]

/-- **A USB library call writes host memory only inside its frame**: what a
    transfer brings back, into the buffer and length handed in. -/
theorem usbCall_respects_frame (f : UsbFn) (bits : List UInt64) (w : World) (r : Option V)
    (w' : World) (h : usbCall f bits w = some (r, w')) (a : UInt64)
    (hout : ¬ (Ext.usb f).frame.allows bits (r.bind asBits) a) : Unchanged w.mem w'.mem a := by
  simp only [usbCall] at h
  obtain ⟨c, -, h⟩ := Option.map_eq_some_iff.mp h
  apply Unchanged.of_eq
  split at h
  all_goals first
    | (rw [show w' = (usbEffect c w).2 by rw [h], usbEffect_mem])
    | (simp only [Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
       simp only [Ext.frame, Frame.allows, List.getD, List.getElem?_cons_zero,
         List.getElem?_cons_succ, Option.getD_some] at hout
       exact copyIn_getD_other a fun i hi e => hout ⟨i, Nat.lt_of_lt_of_le hi (usbBack_size ..), e⟩)

/-- **A window library call writes host memory only inside its frame**: a
    record polled lands in the 32 bytes handed in. -/
theorem wlCall_respects_frame (f : WindowFn) (bits : List UInt64) (w : World) (r : Option V)
    (w' : World) (h : wlCall f bits w = some (r, w')) (a : UInt64)
    (hout : ¬ (Ext.window f).frame.allows bits (r.bind asBits) a) : Unchanged w.mem w'.mem a := by
  simp only [wlCall] at h
  obtain ⟨c, -, h⟩ := Option.map_eq_some_iff.mp h
  apply Unchanged.of_eq
  split at h
  · simp only [Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    simp only [Ext.frame, Frame.allows, List.getD, List.getElem?_cons_zero,
      List.getElem?_cons_succ, Option.getD_some] at hout
    dsimp only
    split
    · apply copyIn_getD_other
      intro i hi e
      rw [exactly_size] at hi
      exact hout ⟨i, hi, e⟩
    · rfl
  · rw [show w' = (wlEffect c w).2 by rw [h], wlEffect_mem]

/-- A C library function writes host memory only inside its frame. -/
theorem cCall_respects_frame (f : CFn) (bits : List UInt64) (w : World) (r : Option V) (w' : World)
    (h : cCall f bits w = some (r, w')) (a : UInt64)
    (hout : ¬ (Ext.c f).frame.allows bits (r.bind asBits) a) : Unchanged w.mem w'.mem a := by
  unfold cCall at h
  revert hout; revert h
  split <;> intro h hout
  -- memcpy, memmove: a copy of `n` bytes to `d`
  iterate 2
    · simp only [Ext.frame, Frame.allows, List.getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some] at hout
      try dsimp only at h
      try (split at h; · cases h)
      obtain ⟨bs, hbs, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
      apply Unchanged.of_eq
      apply copyIn_other hm
      intro i hi e
      rw [copyOut_size hbs] at hi
      exact hout ⟨i, hi, e⟩
  -- memset
  · simp only [Ext.frame, Frame.allows, List.getD, List.getElem?_cons_zero,
      List.getElem?_cons_succ, Option.getD_some] at hout
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    apply Unchanged.of_eq
    apply copyIn_other hm
    intro i hi e
    exact hout ⟨i, by simpa [ByteArray.size] using hi, e⟩
  -- strlen
  · obtain ⟨i, -, h⟩ := Option.bind_eq_some_iff.mp h
    same_mem
  -- calloc: fresh memory past everything that was there, or nothing
  · dsimp only at h
    split at h
    · cases h
    split at h
    · same_mem
    simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    intro x y hx hy
    rw [Mem.load_grow _ _ hx] at hy
    exact Option.some.inj hy
  -- free
  · split at h
    · same_mem
    obtain ⟨⟨off, n⟩, -, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    intro x y hx hy
    rw [Mem.load_shrink (fun r hr => List.mem_of_mem_erase hr) hy] at hx
    exact (Option.some.inj hx).symm
  · cases h

/-- **A C library call writes host memory only inside its frame**: an absent
    library's stub writes nothing. -/
theorem ext_respects_frame (e : Ext) (vs : List V) (w : World) (r : Option V) (w' : World)
    (h : extCall e vs w = some (r, w')) (bits : List UInt64) (hb : vs.mapM asBits = some bits)
    (a : UInt64) (hout : ¬ e.frame.allows bits (r.bind asBits) a) : Unchanged w.mem w'.mem a := by
  unfold extCall at h
  split at h
  · split at h <;> same_mem
  revert hout; revert hb; revert h
  split <;> intro h hb hout
  -- present
  · same_mem
  -- the C library
  · rename_i f vs'
    obtain ⟨bits', hb', h⟩ := Option.bind_eq_some_iff.mp h
    rw [hb] at hb'; cases hb'
    exact cCall_respects_frame f bits w r w' h a hout
  -- the driver
  · rename_i f vs'
    obtain ⟨bits', hb', h⟩ := Option.bind_eq_some_iff.mp h
    rw [hb] at hb'; cases hb'
    exact cudaDrv_respects_frame f bits w r w' h a hout
  -- cuBLAS
  · rename_i f vs'
    obtain ⟨bits', hb', h⟩ := Option.bind_eq_some_iff.mp h
    rw [hb] at hb'; cases hb'
    exact cublasCall_respects_frame f bits w r w' h a hout
  -- wgpu
  · rename_i f vs'
    obtain ⟨bits', hb', h⟩ := Option.bind_eq_some_iff.mp h
    rw [hb] at hb'; cases hb'
    exact wgpuCall_respects_frame f bits w r w' h a hout
  -- the window library
  · rename_i f vs'
    obtain ⟨bits', hb', h⟩ := Option.bind_eq_some_iff.mp h
    rw [hb] at hb'; cases hb'
    exact wlCall_respects_frame f bits w r w' h a hout
  -- the CPU library
  · rename_i f vs'
    obtain ⟨bits', hb', h⟩ := Option.bind_eq_some_iff.mp h
    rw [hb] at hb'; cases hb'
    exact cpuCall_respects_frame f bits w r w' h a
  -- the serial library
  · rename_i f vs'
    obtain ⟨bits', hb', h⟩ := Option.bind_eq_some_iff.mp h
    rw [hb] at hb'; cases hb'
    exact serialCall_respects_frame f bits w r w' h a hout
  -- the USB library
  · rename_i f vs'
    obtain ⟨bits', hb', h⟩ := Option.bind_eq_some_iff.mp h
    rw [hb] at hb'; cases hb'
    exact usbCall_respects_frame f bits w r w' h a hout
  · cases h

end AlgorithmLib.HProg.ExtContracts
