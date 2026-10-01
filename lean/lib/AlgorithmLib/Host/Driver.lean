module
public import AlgorithmLib.Host.Ffi
meta import AlgorithmLib.Host.Ffi
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The CUDA driver, called directly

What each `CudaFn` does to the world, over the same device state `Dev` the
engine's own CUDA entry points act on: buffers, streams, the ordering of work
on them, and the kernel oracle. Only identity and failure differ.

**Identity.** The driver names its objects with pointers and handles of its
choosing. The model chooses too, deterministically, in ranges that cannot meet:
buffer `n` starts at `devPtr n`, and a stream, module or function is its kind
and index. A program that treats a handle as a name, and a device pointer as a
base plus an offset inside its allocation, computes the same on the machine,
whose values differ.

**Failure.** Where the engine's entry points refuse with `-1`, the driver's
behaviour is undefined: freeing a pointer that is not a live allocation's base,
copying outside an allocation, using a module after unloading it. The model
answers none of those, so a program that does one is not given a meaning.
What the driver does report — an entry point a module does not have — is
answered with the driver's own code.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- `CUDA_SUCCESS`, `CUDA_ERROR_INVALID_VALUE`, `CUDA_ERROR_INVALID_DEVICE`,
    `CUDA_ERROR_NOT_FOUND`. -/
def cuSuccess : Int := 0
def cuInvalidValue : Int := 1
def cuInvalidDevice : Int := 101
def cuNotFound : Int := 500
/-- `CUDA_ERROR_INVALID_HANDLE`: an event never recorded, asked for a time. -/
def cuInvalidHandle : Int := 400
/-- `CUDA_ERROR_STREAM_CAPTURE_UNJOINED`: a capture ended with a stream that
    joined it not joined back. -/
def cuCaptureUnjoined : Int := 904

/-- Where device buffer `n` starts. Allocations lie `2^36` bytes apart, so an
    offset inside one never reaches the next. -/
def devPtr (n : Nat) : UInt64 := UInt64.ofNat (2 ^ 52 + n * 2 ^ 36)

/-- The buffer a device pointer falls in, and the offset into it. -/
def devPtrOf (p : UInt64) : Option (Nat × Nat) :=
  let x := p.toNat
  if 2 ^ 52 ≤ x && x < 2 ^ 53 then some ((x - 2 ^ 52) / 2 ^ 36, (x - 2 ^ 52) % 2 ^ 36)
  else none

/-- A stream, module or function handle: its kind and index. -/
def drvHandle (kind i : Nat) : UInt64 := UInt64.ofNat (2 ^ 56 + kind * 2 ^ 48 + i + 1)

def handleOf (kind : Nat) (h : UInt64) : Option Nat :=
  let x := h.toNat
  let lo := 2 ^ 56 + kind * 2 ^ 48
  if lo < x && x ≤ lo + 2 ^ 48 then some (x - lo - 1) else none

def kStream : Nat := 1
def kModule : Nat := 2
def kFunc : Nat := 3
def kEvent : Nat := 4
def kGraph : Nat := 6
def kExec : Nat := 7

/-- The race an operation by `p` is checked against: the capture's own, if `p`
    is capturing, and the device's otherwise. -/
def Dev.raceFor (d : Dev) (p : Nat) : Race :=
  match d.capture with
  | some c => if d.capturing p then c.race else d.race
  | none => d.race

/-- `p` records an event: a point in the capture if `p` is capturing, and in
    the device's order otherwise. -/
def Dev.recordAt (d : Dev) (id p : Nat) : Dev :=
  match d.capture with
  | some c =>
      if d.capturing p then
        let (r, clk) := c.race.issue p
        { d with capture := some { c with race := r }, events := d.events.set! id (some (some (clk, true))) }
      else
        let (r, clk) := d.race.issue p
        { d with race := r, events := d.events.set! id (some (some (clk, false))) }
  | none =>
      let (r, clk) := d.race.issue p
      { d with race := r, events := d.events.set! id (some (some (clk, false))) }

/-- `p` waits for an event's point: never recorded, the wait is satisfied; a
    point inside the capture makes `p` join it; a point outside any capture
    orders `p` after it, unless `p` is capturing, which the model does not
    state. A capture's point with no capture in progress is not either. -/
def Dev.waitOn (d : Dev) (p : Nat) : Option (Clock × Bool) → Option Dev
  | none => some d
  | some (clk, true) =>
      match d.capture with
      | some c =>
          let (r, _) := c.race.issue p
          let joined := if c.origin == p || c.joined.contains p then c.joined else p :: c.joined
          some { d with capture := some { c with race := r.joinInto p clk, joined := joined } }
      | none => none
  | some (clk, false) =>
      if d.capturing p then none
      else
        let (r, _) := d.race.issue p
        some { d with race := r.joinInto p clk }

/-- A graph's nodes run on `p`, ordered as one operation against everything
    else, in the order they were recorded. -/
def Dev.runGraph (d : Dev) (w : World) (ops : Array DevOp) (p : Nat) : Option Dev := do
  let (rs, ws) := graphAccess ops
  let r ← d.race.op p rs ws
  ops.foldlM (fun d op => match op with
    | .launch l ids => d.runLaunch w.kernel l ids
    | .vendor c ins out => d.runVendor w.vendor c ins out) { d with race := r }

/-- A live event's index and the point it last recorded, if any. -/
def Dev.eventOf? (d : Dev) (h : UInt64) : Option (Nat × Option (Clock × Bool)) := do
  let id ← handleOf kEvent h
  let ev ← (d.events[id]?).join
  some (id, ev)

/-- The live page-locked allocation `p` is the start of: its index, offset in
    the pinned region and length. -/
def Dev.hostAllocAt? (d : Dev) (p : UInt64) : Option (Nat × Nat × Nat) :=
  match decodeAddr p with
  | some (.pinned, off) =>
      (List.range d.pinned.size).findSome? fun i =>
        match (d.pinned[i]?).join with
        | some (o, n) => if o == off then some (i, o, n) else none
        | none => none
  | _ => none

/-- The heap allocation `p` is the start of, as `(offset, length)`. -/
def World.heapAt? (w : World) (p : UInt64) : Option (Nat × Nat) :=
  match decodeAddr p with
  | some (.pinned, off) => w.heap.find? (·.1 == off)
  | _ => none

/-- Where the next page-locked allocation starts: on a 64-byte boundary, the
    way page-locked memory is at least aligned. -/
def Mem.pinnedNext (m : Mem) : Nat := m.pinned.size + (64 - m.pinned.size % 64) % 64

/-- The live buffer `n` bytes at `p` lie inside, and where they start in it. -/
def Dev.range? (d : Dev) (p : UInt64) (n : Nat) : Option (Nat × Nat × ByteArray) := do
  let (id, off) ← devPtrOf p
  let b ← (d.bufs[id]?).join
  if off + n ≤ b.size then some (id, off, b) else none

/-- The party a stream handle names: `0` is the default stream. -/
def Dev.streamParty? (d : Dev) (h : UInt64) : Option Nat :=
  if h == 0 then some defaultParty
  else do
    let id ← handleOf kStream h
    if d.streams.getD id false then some (streamParty id) else none

/-- A driver call's result code, the device as it leaves it. -/
def cuRes (d : Dev) (code : Int) : Option (Option V × Dev) := some (some (ofInt .i32 code), d)

/-- The byte sizes of an entry point's parameters, read from its PTX
    declaration: `.param .u64 p0, .param .u32 p1, …`. -/
def ptxParamSizes (ptx entry : String) : Option (List Nat) := do
  let decl := (ptx.splitOn (".entry " ++ entry ++ "(")).drop 1
  let rest ← decl.head?
  let params ← (rest.splitOn ")").head?
  let items := (params.splitOn ",").map String.trim |>.filter (· ≠ "")
  items.mapM fun item =>
    let ws := (item.splitOn " ").filter (· ≠ "")
    match ws with
    | ".param" :: ty :: _ =>
        if ty == ".u64" || ty == ".s64" || ty == ".b64" || ty == ".f64" then some 8
        else if ty == ".u32" || ty == ".s32" || ty == ".b32" || ty == ".f32" then some 4
        else if ty == ".u16" || ty == ".s16" || ty == ".b16" then some 2
        else if ty == ".u8" || ty == ".s8" || ty == ".b8" then some 1
        else none
    | _ => none

/-- Whether a module's PTX declares an entry point of that name. -/
def ptxHasEntry (ptx entry : String) : Bool :=
  (ptx.splitOn (".entry " ++ entry ++ "(")).length > 1

/-- The parameter values `kernelParams` points at: one pointer per parameter,
    each to a value of the size the entry declares. -/
def kernelArgs (m : Mem) (params : UInt64) (sizes : List Nat) : Option (List UInt64) :=
  sizes.zipIdx.mapM fun (sz, i) => do
    let at_ ← m.load (params + UInt64.ofNat (8 * i)) 8
    m.load at_ sz

/-- Every party the device has a clock for, waited for by the host. -/
def Race.syncAll (r : Race) : Race :=
  (List.range r.clocks.size).foldl (fun r p => r.sync p) r

set_option maxHeartbeats 4000000 in
/-- What a driver call does, capture aside: `cudaDrv` says which calls keep
    this meaning while a capture is in progress. -/
def cudaStep (f : CudaFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  let d := w.dev
  match f with
  | .init => match bits with
    | [flags] =>
      if flags &&& 0xffffffff != 0 then some (some (ofInt .i32 cuInvalidValue), w)
      else some (some (ofInt .i32 cuSuccess), { w with dev := { d with drvInit := true } })
    | _ => none
  | .deviceGet => match bits with
    | [pdev, ordinal] =>
      if !d.drvInit then none
      else if (asI32 ordinal) < 0 || (asI32 ordinal).toNat ≥ w.devices then
        some (some (ofInt .i32 cuInvalidDevice), w)
      else do
        let m ← w.mem.store pdev 4 (ordinal &&& 0xffffffff)
        some (some (ofInt .i32 cuSuccess), { w with mem := m })
    | _ => none
  | .primaryCtxRetain => match bits with
    | [pctx, dev] =>
      if !d.drvInit || asI32 dev != 0 then none
      else do
        let m ← w.mem.store pctx 8 cudaCtx
        some (some (ofInt .i32 cuSuccess),
              { w with mem := m, dev := { d with retained := d.retained + 1 } })
    | _ => none
  | .primaryCtxRelease => match bits with
    | [dev] =>
      if !d.drvInit || asI32 dev != 0 || d.retained == 0 then none
      else if d.retained == 1 then
        -- The last release destroys the context and everything in it.
        some (some (ofInt .i32 cuSuccess),
              { w with dev := { drvInit := true }, mem := { w.mem with pinnedLive := [] } })
      else some (some (ofInt .i32 cuSuccess), { w with dev := { d with retained := d.retained - 1 } })
    | _ => none
  | .ctxSetCurrent => match bits with
    | [ctx] =>
      if ctx == 0 then some (some (ofInt .i32 cuSuccess), { w with dev := { d with live := false } })
      else if ctx == cudaCtx && d.retained > 0 then
        some (some (ofInt .i32 cuSuccess), { w with dev := { d with live := true } })
      else none
    | _ => none
  | .ctxSynchronize => match bits with
    | [] =>
      if !d.live then none
      else devOnly w (cuRes { d with race := d.race.syncAll } cuSuccess)
    | _ => none
  | .memGetInfo => match bits with
    | [pfree, ptotal] =>
      if !d.live then none
      else do
        let m ← w.mem.store pfree 8 w.memInfo.1
        let m ← m.store ptotal 8 w.memInfo.2
        some (some (ofInt .i32 cuSuccess), { w with mem := m })
    | _ => none
  | .memAlloc => match bits with
    | [pdptr, size] =>
      if !d.live then none
      else if size == 0 || size.toNat ≥ 2 ^ 36 then some (some (ofInt .i32 cuInvalidValue), w)
      else do
        let n := d.bufs.size
        let m ← w.mem.store pdptr 8 (devPtr n)
        some (some (ofInt .i32 cuSuccess),
              { w with mem := m, dev := { d with bufs := d.bufs.push (some (w.devFill n size.toNat)) } })
    | _ => none
  | .memFree => match bits with
    | [p] =>
      if !d.live then none
      else do
        let (id, off, _) ← d.range? p 0
        if off != 0 then none
        else devOnly w (do
          let r ← d.race.op defaultParty [] [id]
          cuRes { (d.put id none) with race := r } cuSuccess)
    | _ => none
  | .memcpyHtoD => match bits with
    | [dst, src, n] =>
      if !d.live then none
      else do
        let (id, off, b) ← d.range? dst n.toNat
        let bytes ← readBytes w.mem src n.toNat
        devOnly w (do cuRes (← d.syncWrite id (overwrite b off bytes)) cuSuccess)
    | _ => none
  | .memcpyDtoH => match bits with
    | [dst, src, n] =>
      if !d.live then none
      else do
        let (id, off, b) ← d.range? src n.toNat
        let d' ← d.syncRead id
        let m ← copyIn w.mem dst (b.extract off (off + n.toNat))
        some (some (ofInt .i32 cuSuccess), { w with mem := m, dev := d' })
    | _ => none
  | .memcpyDtoD => match bits with
    | [dst, src, n] =>
      if !d.live then none
      else do
        let (sid, soff, sb) ← d.range? src n.toNat
        let (did, doff, _) ← d.range? dst n.toNat
        -- Ordered on the default stream, without the host waiting.
        let r ← d.race.op defaultParty [sid] [did]
        let d := { d with race := r }
        let db ← (d.bufs[did]?).join
        devOnly w (cuRes (d.put did (some (overwrite db doff (sb.extract soff (soff + n.toNat)))))
          cuSuccess)
    | _ => none
  | .memsetD8 => match bits with
    | [dst, uc, n] =>
      if !d.live then none
      else do
        let (id, off, b) ← d.range? dst n.toNat
        let r ← d.race.op defaultParty [] [id]
        let fill := ByteArray.mk (Array.replicate n.toNat (uc &&& 0xff).toUInt8)
        devOnly w (cuRes ({ d with race := r }.put id (some (overwrite b off fill))) cuSuccess)
    | _ => none
  | .moduleLoadData => match bits with
    | [pmod, image] =>
      if !d.live then none
      else do
        let ptx ← readCStrAt w.mem image
        let i := d.modules.size
        let m ← w.mem.store pmod 8 (drvHandle kModule i)
        some (some (ofInt .i32 cuSuccess),
              { w with mem := m, dev := { d with modules := d.modules.push (some ptx) } })
    | _ => none
  | .moduleGetFunction => match bits with
    | [pfunc, hmod, pname] =>
      if !d.live then none
      else do
        let mi ← handleOf kModule hmod
        let ptx ← (d.modules[mi]?).join
        let name ← readCStrAt w.mem pname
        if !ptxHasEntry ptx name then some (some (ofInt .i32 cuNotFound), w)
        else
          let i := d.funcs.size
          let m ← w.mem.store pfunc 8 (drvHandle kFunc i)
          some (some (ofInt .i32 cuSuccess),
                { w with mem := m, dev := { d with funcs := d.funcs.push (mi, name) } })
    | _ => none
  | .moduleUnload => match bits with
    | [hmod] =>
      if !d.live then none
      else do
        let mi ← handleOf kModule hmod
        let _ ← (d.modules[mi]?).join
        some (some (ofInt .i32 cuSuccess),
              { w with dev := { d with modules := d.modules.set! mi none } })
    | _ => none
  | .launchKernel => match bits with
    | [hf, gx, gy, gz, bx, by_, bz, shmem, hstream, params, extra] =>
      if !d.live || extra != 0 then none
      else do
        let fi ← handleOf kFunc hf
        let (mi, entry) ← d.funcs[fi]?
        let ptx ← (d.modules[mi]?).join
        let p ← d.streamParty? hstream
        let dims := [gx, gy, gz, bx, by_, bz].map (· &&& 0xffffffff)
        if dims.any (· == 0) then some (some (ofInt .i32 cuInvalidValue), w)
        else
        let _ := shmem
        let sizes ← ptxParamSizes ptx entry
        let args ← kernelArgs w.mem params sizes
        -- A parameter that points into a live allocation binds it.
        let ids := (args.filterMap fun a => (d.range? a 0).map (·.1)).eraseDups
        let l : Launch := ⟨ptx, entry, ids, dims, args⟩
        devOnly w (do cuRes (← d.devOp w p (.launch l ids) [] ids) cuSuccess)
    | _ => none
  | .streamCreate => match bits with
    | [pstream, flags] =>
      -- Only a non-blocking stream: a blocking one waits on the default stream
      -- implicitly, which this model does not state.
      if !d.live || flags &&& 0xffffffff != 1 then none
      else do
        let id := d.streams.size
        let m ← w.mem.store pstream 8 (drvHandle kStream id)
        let r := d.race.joinInto (streamParty id)
          ((d.race.clock defaultParty).join (d.race.clock hostParty))
        some (some (ofInt .i32 cuSuccess),
              { w with mem := m, dev := { d with streams := d.streams.push true, race := r } })
    | _ => none
  | .streamSynchronize => match bits with
    | [hstream] =>
      if !d.live then none
      else do
        let p ← d.streamParty? hstream
        if d.capturing p then none
        else devOnly w (cuRes { d with race := d.race.sync p } cuSuccess)
    | _ => none
  | .streamDestroy => match bits with
    | [hstream] =>
      if !d.live || hstream == 0 then none
      else do
        let id ← handleOf kStream hstream
        if !d.streams.getD id false then none
        else some (some (ofInt .i32 cuSuccess),
                   { w with dev := { d with streams := d.streams.set! id false } })
    | _ => none
  | .eventCreate => match bits with
    | [pev, flags] =>
      -- Only default events: the flags that disable timing or make the host
      -- block rather than spin change nothing the model states, but are not
      -- admitted until a corpus shows them.
      if !d.live || flags &&& 0xffffffff != 0 then none
      else do
        let id := d.events.size
        let m ← w.mem.store pev 8 (drvHandle kEvent id)
        some (some (ofInt .i32 cuSuccess), { w with mem := m, dev := { d with events := d.events.push (some none) } })
    | _ => none
  | .eventRecord => match bits with
    | [hev, hstream] =>
      if !d.live then none
      else do
        let (id, _) ← d.eventOf? hev
        let p ← d.streamParty? hstream
        devOnly w (cuRes (d.recordAt id p) cuSuccess)
    | _ => none
  | .streamWaitEvent => match bits with
    | [hstream, hev, flags] =>
      if !d.live || flags &&& 0xffffffff != 0 then none
      else do
        let p ← d.streamParty? hstream
        let (_, ev) ← d.eventOf? hev
        let d' ← d.waitOn p ev
        devOnly w (cuRes d' cuSuccess)
    | _ => none
  | .eventSynchronize => match bits with
    | [hev] =>
      if !d.live then none
      else do
        let (_, ev) ← d.eventOf? hev
        match ev with
        | none => devOnly w (cuRes d cuSuccess)
        | some (clk, false) => devOnly w (cuRes { d with race := d.race.joinInto hostParty clk } cuSuccess)
        | some (_, true) => none
    | _ => none
  | .eventDestroy => match bits with
    | [hev] =>
      if !d.live then none
      else do
        let (id, _) ← d.eventOf? hev
        some (some (ofInt .i32 cuSuccess), { w with dev := { d with events := d.events.set! id none } })
    | _ => none
  | .eventElapsedTime => match bits with
    | [pms, hs, he] =>
      if !d.live then none
      else do
        let (si, sev) ← d.eventOf? hs
        let (ei, eev) ← d.eventOf? he
        match sev, eev with
        | some (cs, false), some (ce, false) =>
            -- Before the host has waited for both the driver may answer or
            -- report "not ready", which the model does not choose between.
            if d.race.hostSaw cs && d.race.hostSaw ce then do
              let m ← w.mem.store pms 4 (w.elapsed si ei).toUInt64
              some (some (ofInt .i32 cuSuccess), { w with mem := m })
            else none
        | none, _ | _, none => some (some (ofInt .i32 cuInvalidHandle), w)
        | _, _ => none
    | _ => none
  | .memAllocHost => match bits with
    | [pp, size] =>
      if !d.live || size == 0 then none
      else
        let off := w.mem.pinnedNext
        let n := size.toNat
        -- Running out of page-locked memory is not modelled.
        if off + n > regionSpan.toNat then none
        else do
          let m ← w.mem.store pp 8 (addrOf .pinned off)
          let bytes := exactly (w.pinnedFill d.pinned.size n) n
          let m := { m with pinned := m.pinned ++ (ByteArray.mk (Array.replicate (off - m.pinned.size) 0) ++ bytes)
                            pinnedLive := (off, n) :: m.pinnedLive }
          some (some (ofInt .i32 cuSuccess), { w with mem := m, dev := { d with pinned := d.pinned.push (some (off, n)) } })
    | _ => none
  | .memFreeHost => match bits with
    | [p] =>
      if !d.live then none
      else do
        let (id, off, n) ← d.hostAllocAt? p
        -- Freeing memory a copy still uses is undefined.
        if w.mem.busy.any (fun b => !b.clear off n) then none
        else some (some (ofInt .i32 cuSuccess),
          { w with mem := { w.mem with pinnedLive := w.mem.pinnedLive.erase (off, n) },
                   dev := { d with pinned := d.pinned.set! id none } })
    | _ => none
  | .memcpyHtoDAsync => match bits with
    | [dst, src, n, hstream] =>
      if !d.live then none
      else do
        let p ← d.streamParty? hstream
        let (id, off, b) ← d.range? dst n.toNat
        let bytes ← readBytes (w.mem.forParty p) src n.toNat
        asyncCopy w p src n.toNat false [] [id] fun r =>
          some ({ (d.put id (some (overwrite b off bytes))) with race := r }, w.mem)
    | _ => none
  | .memcpyDtoHAsync => match bits with
    | [dst, src, n, hstream] =>
      if !d.live then none
      else do
        let p ← d.streamParty? hstream
        let (id, off, b) ← d.range? src n.toNat
        asyncCopy w p dst n.toNat true [id] [] fun r => do
          let m ← copyIn (w.mem.forParty p) dst (b.extract off (off + n.toNat))
          -- to pageable memory the call returns once the copy is done
          let r := if (pinnedSpan dst n.toNat).isSome then r else r.sync p
          some ({ d with race := r }, m)
    | _ => none
  | .beginCapture => match bits with
    | [hstream, mode] =>
      -- Only relaxed capture: the other modes forbid calls this model does
      -- not tell apart.
      if !d.live || mode &&& 0xffffffff != 2 || d.capture.isSome then none
      else do
        let p ← d.streamParty? hstream
        -- the legacy default stream cannot be captured
        if p == defaultParty then none
        else devOnly w (cuRes { d with capture := some ⟨p, [], {}, #[]⟩ } cuSuccess)
    | _ => none
  | .endCapture => match bits with
    | [hstream, pgraph] =>
      if !d.live then none
      else do
        let p ← d.streamParty? hstream
        let c ← d.capture
        if c.origin != p then none
        -- a stream that joined and was not joined back invalidates the capture
        else if !c.rejoined then devOnly w (cuRes d.endCapture cuCaptureUnjoined)
        else do
          let i := d.graphs.size
          let m ← w.mem.store pgraph 8 (drvHandle kGraph i)
          some (some (ofInt .i32 cuSuccess),
                { w with mem := m, dev := { d.endCapture with graphs := d.graphs.push (some c.ops) } })
    | _ => none
  | .graphInstantiate => match bits with
    | [pexec, hgraph, flags] =>
      if !d.live || flags != 0 then none
      else do
        let gi ← handleOf kGraph hgraph
        let ops ← (d.graphs[gi]?).join
        let i := d.execs.size
        let m ← w.mem.store pexec 8 (drvHandle kExec i)
        some (some (ofInt .i32 cuSuccess), { w with mem := m, dev := { d with execs := d.execs.push (some ops) } })
    | _ => none
  | .graphLaunch => match bits with
    | [hexec, hstream] =>
      if !d.live then none
      else do
        let ei ← handleOf kExec hexec
        let ops ← (d.execs[ei]?).join
        let p ← d.streamParty? hstream
        devOnly w (do cuRes (← d.runGraph w ops p) cuSuccess)
    | _ => none
  | .graphExecDestroy => match bits with
    | [hexec] =>
      if !d.live then none
      else do
        let ei ← handleOf kExec hexec
        let _ ← (d.execs[ei]?).join
        some (some (ofInt .i32 cuSuccess), { w with dev := { d with execs := d.execs.set! ei none } })
    | _ => none
  | .graphDestroy => match bits with
    | [hgraph] =>
      if !d.live then none
      else do
        let gi ← handleOf kGraph hgraph
        let _ ← (d.graphs[gi]?).join
        some (some (ofInt .i32 cuSuccess), { w with dev := { d with graphs := d.graphs.set! gi none } })
    | _ => none

/-- Whether a driver call keeps its meaning while a capture is in progress:
    work on a stream, which a capturing stream records, and ending the
    capture. Every other call is left without an answer then — relaxed
    capture allows many of them, but the model does not state which. -/
def _root_.AlgorithmLib.IR.CudaFn.underCapture : CudaFn → Bool
  | .launchKernel | .eventRecord | .streamWaitEvent | .endCapture => true
  | _ => false

/-- **What a driver call does.** `none` where the driver's behaviour is
    undefined or the model does not state it. -/
def cudaDrv (f : CudaFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  if w.dev.capture.isSome && !f.underCapture then none else cudaStep f bits w

end AlgorithmLib.HProg.Sem
