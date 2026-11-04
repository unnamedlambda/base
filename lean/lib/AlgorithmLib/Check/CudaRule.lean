module
public import AlgorithmLib.Check.Calls
meta import AlgorithmLib.Check.Calls
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.CudaRule` — the CUDA driver and cuBLAS, called directly

The checker reads a call into either library by its rows: the driver's
(`cudaRow`) and cuBLAS's (`blasRow`). A cuBLAS handle is a
number the typestate counts (`nblas`) with the stream it runs on (`blasOn`);
a product is one operation on that stream, which must know of every device
access, over operands in live allocations with the output in its own.

What the checker takes of a call into the CUDA driver: whether the driver's
own precondition (`ExtContracts.cudaPre`) holds, decided from the typestate
and the arguments' known bits, what the call leaves of the typestate, and
what it answers. A call this file has no row for is refused.

Every row needs the driver loaded (`has cuda`), and no capture in progress
but the rows a capture records: an event recorded or waited on, a kernel's
launch, a cuBLAS product, and the capture's end. A capture's fact (`cap`)
holds its ordering exactly, so a launch or product on any stream the capture
has joined is checked against what it has recorded the way the driver checks
it, and the capture ends only once every stream that joined it is joined
back. The device's allocations are numbered in the order they are made, so a
typestate that counts them (`nbufs`) knows the pointer the next one answers;
a pointer the program loads back from where it was stored is then a known
number the rows can place in its allocation (`alloc`).

Streams and events are handles the driver numbers the same way (`streams`,
`events`, `evLive`). Ordering is stated as who knows of every device access
(`readyAll p`, or `devSeq` for the default stream): an event recorded on a
stream that knows of every access covers them (`evAll`), and a stream that
waits on such an event, or the host that waits on such a stream, knows of
every access in turn.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState Part afterStore storeKeeps roomAt readyIn streamsOf eventsOf)

/-- How many allocations the typestate says the device has made. -/
def nbufsOf (S : TState) : Option Nat :=
  S.findSome? fun | .nbufs n => some n | _ => none

/-- How many times the typestate says the primary context is retained. -/
def retainedOf (S : TState) : Option Nat :=
  S.findSome? fun | .retained n => some n | _ => none

/-- How many modules the typestate says the driver has loaded. -/
def nmodsOf (S : TState) : Option Nat :=
  S.findSome? fun | .nmods n => some n | _ => none

/-- How many functions the typestate says the driver has looked up. -/
def nfuncsOf (S : TState) : Option Nat :=
  S.findSome? fun | .nfuncs n => some n | _ => none

/-- The C string the typestate says memory holds at `a`. -/
def textOf (S : TState) (a : UInt64) : Option String :=
  S.findSome? fun | .text a' (some t) _ _ => if a' == a then some t else none | _ => none

/-- The entry points loaded module `i` declares, as the typestate has them. -/
def modOf (S : TState) (i : Nat) : Option (List (String × List Nat × List Bool)) :=
  S.findSome? fun | .mod i' sg => if i' == i then some sg else none | _ => none

/-- What the typestate says of the C string at `a`: the string, where it
    names it, and the entry points it declares. -/
def sigsOf (S : TState) (a : UInt64) : Option (Option String × List (String × List Nat × List Bool)) :=
  S.findSome? fun | .text a' t _ sg => if a' == a then some (t, sg) else none | _ => none

/-- The text loaded module `i` holds, as the typestate has it. -/
def modTextOf (S : TState) (i : Nat) : Option String :=
  S.findSome? fun | .modText i' t => if i' == i then some t else none | _ => none

/-- The parameter sizes of function `i`, as the typestate has them. -/
def fnOf (S : TState) (i : Nat) : Option (List Nat) :=
  S.findSome? fun | .fn i' sz _ => if i' == i then some sz else none | _ => none

/-- Which parameters function `i` writes, as the typestate has them. -/
def fnWritesOf (S : TState) (i : Nat) : Option (List Bool) :=
  S.findSome? fun | .fn i' _ fl => if i' == i then some fl else none | _ => none

/-- The eight bytes the typestate says memory holds at `a`. -/
def cellOf (S : TState) (a : UInt64) : Option UInt64 :=
  S.findSome? fun | .cell a' v => if a' == a then some v else none | _ => none

/-- The party a stream handle names, by the streams the typestate knows: `0`
    is the default stream. -/
def drvParty (S : TState) (h : UInt64) : Option Nat :=
  if h == 0 then some defaultParty
  else match handleOf kStream h, streamsOf S with
    | some id, some l => if l.getD id false then some (streamParty id) else none
    | _, _ => none

/-- The event a handle names, where the typestate holds it live. -/
def evOf (S : TState) (h : UInt64) : Option Nat :=
  match handleOf kEvent h with
  | some e => if S.contains (.evLive e) then some e else none
  | none => none

/-- Every parameter a launch passes can be read: the pointer to it in the
    array at `params`, and the bytes of the size its entry declares. -/
def argsReadable (S : TState) (params : UInt64) (sizes : List Nat) : Bool :=
  sizes.zipIdx.all fun (sz, i) =>
    match cellOf S (params + UInt64.ofNat (8 * i)) with
    | some a => roomAt S a sz
    | none => false

/-- `p` is the start of an allocation the typestate holds live: its index. -/
def allocBase (S : TState) (p : UInt64) : Option Nat :=
  match devPtrOf p with
  | some (id, 0) => if S.any (fun | .alloc i _ => i == id | _ => false) then some id else none
  | _ => none

/-- A fact about a captured or instantiated graph, or a capture in progress:
    what a step that frees an allocation drops. -/
def _root_.AlgorithmLib.HProg.Contracts.Fact.isGraph : Fact → Bool
  | .graph _ | .exec _ | .cap _ _ _ _ => true
  | _ => false

/-- The facts a step on the device's tracker keeps: every fact but those
    that read the tracker's clocks. -/
def raceKeeps : Fact → Bool
  | .readyAll _ | .evAll _ => false
  | _ => true

/-- The facts a copy of `n` bytes into host memory at `a` keeps: what a store
    of as many bytes there keeps, but a hash table's. -/
def copyKeeps (a : UInt64) (n : Nat) (x : Fact) : Bool :=
  storeKeeps a n x && !(x matches .htAt _ _ _)

/-- What a copy of `n` bytes into host memory leaves of a typestate: the
    facts it cannot reach where its address is known, and every fact about
    anything but memory's contents where it is not. -/
def afterCopy (a : Option UInt64) (n : Nat) (S : TState) : TState :=
  match a with
  | some a => S.filter (copyKeeps a n)
  | none => S.filter (!·.isCell)

/-- What one operation on the device by stream `q` leaves: no event or
    party is known to cover it but `q`, which keeps knowing of every access
    where it did; the default stream's order holds only if `q` is it. -/
def afterOp (S : TState) (q : Nat) : TState :=
  (S.filter raceKeeps).filter (fun x => q == defaultParty || x != .devSeq) ++
    (if readyIn S q then [.readyAll q] else [])

/-- What every driver call needs of the typestate: the driver loaded and no
    capture in progress. -/
def cudaBase (S : TState) : Bool :=
  S.contains (.part (.has .cuda) true) && S.contains (.part .capturing false)

/-- What a driver call recorded by a capture needs of the typestate: the
    driver loaded and a capture in progress. -/
def capBase (S : TState) : Bool :=
  S.contains (.part (.has .cuda) true) && S.contains (.part .capturing true)

/-- How many graphs the typestate says the driver has captured. -/
def ngraphsOf (S : TState) : Option Nat :=
  S.findSome? fun | .ngraphs n => some n | _ => none

/-- How many graphs the typestate says the driver has instantiated. -/
def nexecsOf (S : TState) : Option Nat :=
  S.findSome? fun | .nexecs n => some n | _ => none

/-- The graph a handle names, where the typestate holds it live. -/
def graphOf (S : TState) (h : UInt64) : Option Nat :=
  match handleOf kGraph h with
  | some g => if S.contains (.graph g) then some g else none
  | none => none

/-- The capture the typestate says is in progress: the party it began on,
    the streams that joined it, the ordering among what it records, and the
    events recorded inside it with their clocks. -/
def capOf (S : TState) : Option (Nat × List Nat × Race × List (Nat × Clock)) :=
  S.findSome? fun | .cap q j r evs => some (q, j, r, evs) | _ => none

/-- Whether a capture begun on `q` and joined by `j` records what `p` issues. -/
def capIn (q : Nat) (j : List Nat) (p : Nat) : Bool := q == p || j.contains p

/-- The typestate with its capture's fact replaced. -/
def withCap (S : TState) (q : Nat) (j : List Nat) (r : Race) (evs : List (Nat × Clock)) : TState :=
  .cap q j r evs :: S.filter (fun | .cap _ _ _ _ => false | _ => true)

/-- The allocation a device pointer lies in, as the typestate decides it:
    `some (some id)` inside live allocation `id`, `some none` inside none,
    `none` where the typestate cannot tell. -/
def allocAt? (S : TState) (x : UInt64) : Option (Option Nat) :=
  match devPtrOf x with
  | none => some none
  | some (id, off) =>
      match S.findSome? (fun | .alloc i n => if i == id then some n else none | _ => none) with
      | some n => some (if off ≤ n then some id else none)
      | none =>
          match nbufsOf S with
          | some k => if k ≤ id then some none else none
          | none => none

/-- The allocation each parameter of a launch through the array at `params`
    points into, where the typestate knows each parameter's eight bytes:
    `none` for one pointing into none. -/
def launchOs (S : TState) (params : UInt64) (sizes : List Nat) : Option (List (Option Nat)) :=
  sizes.zipIdx.mapM fun (sz, i) => do
    let a ← cellOf S (params + UInt64.ofNat (8 * i))
    if sz == 8 then allocAt? S (← cellOf S a) else none

/-- The allocations a launch through the array at `params` binds, where the
    typestate knows each parameter's eight bytes: the allocations they point
    into, each once. -/
def launchIds (S : TState) (params : UInt64) (sizes : List Nat) : Option (List Nat) := do
  let os ← launchOs S params sizes
  some (os.filterMap id).eraseDups

/-- The allocations such a launch writes: those the parameters `writes`
    flags point into (a missing flag is written), each once. -/
def launchWrites (S : TState) (params : UInt64) (sizes : List Nat) (writes : List Bool) :
    Option (List Nat) := do
  let os ← launchOs S params sizes
  some (os.zipIdx.filterMap fun (o, i) => if writes.getD i true then o else none).eraseDups

/-- The instantiated graph a handle names, where the typestate holds it live. -/
def execOf (S : TState) (h : UInt64) : Option Nat :=
  match handleOf kExec h with
  | some e => if S.contains (.exec e) then some e else none
  | none => none

/-- What a successful driver call answers. -/
def cuOk : AV := .known (.sc .i32 0)

/-- **A driver call, decided**: the typestate after it and its answer, where
    the call's precondition is shown; `none` otherwise. The values `cudaVal`
    reads are decided apart. -/
def cudaRow (S : TState) : CudaFn → List AV → Option (TState × AV)
  | .init, [fl] =>
      if cudaBase S && fl.bits == some 0 then
        some (.drvInit :: S.filter (· != .drvInit), cuOk)
      else none
  | .primaryCtxRetain, [p, dev] =>
      match retainedOf S with
      | some k =>
          if cudaBase S && S.contains .drvInit && dev.bits == some 0 && memN S p 8 then
            some (.retained (k + 1) ::
              (afterStore p.bits (some cudaCtx) 8 S).filter (fun | .retained _ => false | _ => true), cuOk)
          else none
      | none => none
  | .ctxSetCurrent, [c] =>
      match retainedOf S with
      | some k =>
          if cudaBase S && c.bits == some cudaCtx && decide (0 < k) then
            some (.part .cuda true :: S.filter (fun | .part .cuda _ => false | _ => true), cuOk)
          else none
      | none => none
  | .memAlloc, [p, sz] =>
      -- a size known only by a range the driver accepts all of: the
      -- allocation is made, and its size stated where the range is one number
      match nbufsOf S, sz.num with
      | some n, some z =>
          if cudaBase S && S.contains (.part .cuda true) && memN S p 8 &&
              decide (0 < z.lo) && decide (z.hi < 2 ^ 36) then
            some (.nbufs (n + 1) :: ((if z.lo = z.hi then [.alloc n z.lo] else []) ++
              (afterStore p.bits (some (devPtr n)) 8 S).filter (fun | .nbufs _ => false | _ => true)), cuOk)
          else none
      | _, _ => none
  | .memcpyHtoD, [_, src, n] =>
      -- the host waits for the copy, after the default stream's knowing of
      -- every access: so after it, the host does
      if cudaBase S && S.contains (.part .cuda true) && readyIn S defaultParty && memB S src n then
        some (S.filter raceKeeps ++ [.readyAll hostParty], cuOk)
      else none
  | .memcpyDtoH, [dst, _, n] =>
      if cudaBase S && S.contains (.part .cuda true) && readyIn S defaultParty && memB S dst n then
        -- where the length is not known, every fact about memory's contents goes
        some ((match n.bits with
          | some k => afterCopy dst.bits k.toNat S
          | none => afterCopy none 0 S).filter raceKeeps ++ [.readyAll hostParty], cuOk)
      else none
  | .memsetD8, [dst, _, n] =>
      match dst.bits, n.bits with
      | some d, some k =>
          if cudaBase S && S.contains (.part .cuda true) && S.contains .devSeq && allocFor S d k.toNat then
            some (S.filter raceKeeps, cuOk)
          else none
      | _, _ => none
  | .ctxSynchronize, [] =>
      if cudaBase S && S.contains (.part .cuda true) then
        some (S.filter raceKeeps, cuOk)
      else none
  | .memFree, [p] =>
      match p.bits.bind (allocBase S) with
      | some id =>
          if cudaBase S && S.contains (.part .cuda true) && S.contains .devSeq then
            some ((S.filter raceKeeps).filter (fun | .alloc i _ => i != id | .graph _ | .exec _ | .cap _ _ _ _ => false | _ => true), cuOk)
          else none
      | none => none
  | .memcpyDtoD, [_, _, _] =>
      if cudaBase S && S.contains (.part .cuda true) && S.contains .devSeq then
        some (S.filter raceKeeps, cuOk)
      else none
  | .memGetInfo, [pf, pt] =>
      if cudaBase S && S.contains (.part .cuda true) && memN S pf 8 && memN S pt 8 then
        some (afterStore pt.bits none 8 (afterStore pf.bits none 8 S), cuOk)
      else none
  | .moduleLoadData, [p, img] =>
      match nmodsOf S, img.bits.bind (sigsOf S) with
      | some k, some (t, sg) =>
          if cudaBase S && S.contains (.part .cuda true) && memN S p 8 then
            some (.nmods (k + 1) :: .mod k sg :: (t.map (.modText k)).toList ++
              (afterStore p.bits (some (drvHandle kModule k)) 8 S).filter (fun | .nmods _ => false | _ => true), cuOk)
          else none
      | _, _ => none
  | .moduleGetFunction, [p, hm, pn] =>
      match nfuncsOf S, hm.bits.bind (handleOf kModule), pn.bits.bind (textOf S) with
      | some k, some mi, some name =>
          match (modOf S mi).bind (·.lookup name) with
          | some (sizes, fl) =>
              if cudaBase S && S.contains (.part .cuda true) && memN S p 8 then
                some (.nfuncs (k + 1) :: .fn k sizes fl :: ((modTextOf S mi).map (.fnText k · name)).toList ++
                  (afterStore p.bits (some (drvHandle kFunc k)) 8 S).filter (fun | .nfuncs _ => false | _ => true), cuOk)
              else none
          | none => none
      | _, _, _ => none
  | .launchKernel, [hf, gx, gy, gz, bx, by_, bz, _, hs, pr, ex] =>
      match hf.bits.bind (handleOf kFunc), pr.bits with
      | some fi, some P =>
          match fnOf S fi, hs.bits.bind (drvParty S) with
          | some sizes, some q =>
              if cudaBase S && S.contains (.part .cuda true) && readyIn S q && S.contains .oracles &&
                  ex.bits == some 0 && argsReadable S P sizes then
                some (afterOp S q, cuOk)
              -- on a stream the capture records: recorded, not run, ordered
              -- against what the capture has recorded
              else if capBase S && S.contains (.part .cuda true) && S.contains .oracles &&
                  ex.bits == some 0 && argsReadable S P sizes then
                match capOf S, launchIds S P sizes, (fnWritesOf S fi).bind (launchWrites S P sizes) with
                | some (cq, j, r, evs), some ids, some ws =>
                    if capIn cq j q then
                      match r.op q ids ws with
                      | some r' => some (withCap S cq j r' evs, cuOk)
                      | none => none
                    else none
                | _, _, _ => none
              else none
          | _, _ => none
      | _, _ => none
  | .streamCreate, [p, fl] =>
      match streamsOf S with
      | some l =>
          if cudaBase S && S.contains (.part .cuda true) && fl.bits == some 1 && memN S p 8 then
            some (.streams (l ++ [true]) ::
              ((afterStore p.bits (some (drvHandle kStream l.length)) 8 S).filter
                (fun | .streams _ => false | _ => true)) ++
              (if readyIn S defaultParty then [.readyAll (streamParty l.length)] else []), cuOk)
          else none
      | none => none
  | .streamSynchronize, [h] =>
      match h.bits.bind (drvParty S) with
      | some q =>
          if cudaBase S && S.contains (.part .cuda true) then
            some (S ++ (if readyIn S q then [.readyAll hostParty] else []), cuOk)
          else none
      | none => none
  | .streamDestroy, [h] =>
      match h.bits, streamsOf S with
      | some x, some l =>
          match handleOf kStream x with
          | some id =>
              if cudaBase S && S.contains (.part .cuda true) && x != 0 && l.getD id false then
                some (.streams (l.set id false) :: S.filter (fun | .streams _ => false | _ => true), cuOk)
              else none
          | none => none
      | _, _ => none
  | .eventCreate, [p, fl] =>
      match eventsOf S with
      | some n =>
          if cudaBase S && S.contains (.part .cuda true) && fl.bits == some 0 && memN S p 8 then
            some (.events (n + 1) :: .evLive n ::
              (afterStore p.bits (some (drvHandle kEvent n)) 8 S).filter (fun | .events _ => false | _ => true),
              cuOk)
          else none
      | none => none
  | .eventRecord, [hev, h] =>
      match hev.bits.bind (evOf S), h.bits.bind (drvParty S) with
      | some e, some q =>
          if cudaBase S && S.contains (.part .cuda true) then
            some (S.filter (· != .evAll e) ++ (if readyIn S q then [.evAll e] else []), cuOk)
          -- on a stream the capture records: a point inside the capture
          else if capBase S && S.contains (.part .cuda true) then
            match capOf S with
            | some (cq, j, r, evs) =>
                if capIn cq j q then
                  some (withCap (S.filter (· != .evAll e)) cq j (r.issue q).1
                    ((e, (r.issue q).2) :: evs.filter (·.1 != e)), cuOk)
                else none
            | none => none
          else none
      | _, _ => none
  | .streamWaitEvent, [h, hev, fl] =>
      match h.bits.bind (drvParty S), hev.bits.bind (evOf S) with
      | some q, some e =>
          if cudaBase S && S.contains (.part .cuda true) && fl.bits == some 0 && S.contains (.evAll e) then
            some (S ++ [.readyAll q], cuOk)
          -- on a point inside the capture: the stream joins it, ordered after
          -- that point
          else if capBase S && S.contains (.part .cuda true) && fl.bits == some 0 && q != defaultParty then
            match capOf S with
            | some (cq, j, r, evs) =>
                match evs.lookup e with
                | some clk =>
                    some (withCap S cq (if capIn cq j q then j else q :: j)
                      ((r.issue q).1.joinInto q clk) evs, cuOk)
                | none => none
            | none => none
          else none
      | _, _ => none
  | .eventDestroy, [hev] =>
      match hev.bits.bind (evOf S) with
      | some e =>
          if cudaBase S && S.contains (.part .cuda true) then
            some (S.filter (fun x => x != .evLive e && x != .evAll e), cuOk)
          else none
      | none => none
  | .beginCapture, [h, mode] =>
      match h.bits.bind (drvParty S), mode.bits with
      | some q, some m =>
          if cudaBase S && S.contains (.part .cuda true) && m &&& 0xffffffff == 2 && q != defaultParty then
            some (.cap q [] {} [] :: .part .capturing true :: S.filter (· != .part .capturing false), cuOk)
          else none
      | _, _ => none
  | .endCapture, [h, pg] =>
      match h.bits.bind (drvParty S), ngraphsOf S, capOf S with
      | some q, some n, some (cq, j, r, _) =>
          -- every stream that joined has been joined back
          if capBase S && S.contains (.part .cuda true) && q == cq && (Capture.mk cq j r #[]).rejoined &&
              memN S pg 8 then
            some (.graph n :: .ngraphs (n + 1) :: .part .capturing false ::
              (afterStore pg.bits (some (drvHandle kGraph n)) 8 S).filter
                (fun | .ngraphs _ | .cap _ _ _ _ | .part .capturing _ => false | _ => true), cuOk)
          else none
      | _, _, _ => none
  | .graphInstantiate, [pe, hg, fl] =>
      match hg.bits.bind (graphOf S), nexecsOf S with
      | some _, some n =>
          if cudaBase S && S.contains (.part .cuda true) && fl.bits == some 0 && memN S pe 8 then
            some (.exec n :: .nexecs (n + 1) ::
              (afterStore pe.bits (some (drvHandle kExec n)) 8 S).filter (fun | .nexecs _ => false | _ => true), cuOk)
          else none
      | _, _ => none
  | .graphUpload, [he, h] =>
      match he.bits.bind (execOf S), h.bits.bind (drvParty S) with
      | some _, some _ => if cudaBase S && S.contains (.part .cuda true) then some (S, cuOk) else none
      | _, _ => none
  | .graphLaunch, [he, h] =>
      match he.bits.bind (execOf S), h.bits.bind (drvParty S) with
      | some _, some q =>
          if cudaBase S && S.contains (.part .cuda true) && readyIn S q && S.contains .oracles then
            some (afterOp S q, cuOk)
          else none
      | _, _ => none
  | .graphExecDestroy, [he] =>
      match he.bits.bind (execOf S) with
      | some e => if cudaBase S && S.contains (.part .cuda true) then some (S.filter (· != .exec e), cuOk) else none
      | none => none
  | .graphDestroy, [hg] =>
      match hg.bits.bind (graphOf S) with
      | some g => if cudaBase S && S.contains (.part .cuda true) then some (S.filter (· != .graph g), cuOk) else none
      | none => none
  | _, _ => none


-- ---------------------------------------------------------------------------
-- cuBLAS
-- ---------------------------------------------------------------------------

/-- How many handles the typestate says cuBLAS has made. -/
def nblasOf (S : TState) : Option Nat :=
  S.findSome? fun | .nblas n => some n | _ => none

/-- The stream handle `i` runs on, where the typestate holds it live. -/
def blasOnOf (S : TState) (i : Nat) : Option UInt64 :=
  S.findSome? fun | .blasOn j s => if j == i then some s else none | _ => none

/-- The party a cuBLAS handle runs its routines on. -/
def blasParty (S : TState) (h : UInt64) : Option Nat := do
  let i ← handleOf kBlas h
  let s ← blasOnOf S i
  drvParty S s

/-- What every cuBLAS call needs: cuBLAS and the driver loaded, the context
    current, and no capture in progress. -/
def blasBase (S : TState) : Bool :=
  S.contains (.part (.has .cublas) true) && cudaBase S && S.contains (.part .cuda true)

/-- What a cuBLAS call a capture records needs: cuBLAS and the driver
    loaded, the context current, and a capture in progress. -/
def blasCapBase (S : TState) : Bool :=
  S.contains (.part (.has .cublas) true) && capBase S && S.contains (.part .cuda true)

/-- **A product, decided**: `y = α op(A) x + β y` or `C = α op(A) op(B) + β C`.
    Kernels and cuBLAS keep the sizes of the buffers they write, and `α` and
    `β` can be read. Outside a capture its handle's stream knows of every
    device access; on a stream a capture records, it is recorded, ordered
    against what the capture has recorded by the allocations `ids` it reads,
    its output last. What it needs of its other arguments' values is
    `blasVal`'s. -/
def productRow (S : TState) (h pa pb : UInt64) (ids : Option (List Nat)) : Option (TState × AV) :=
  match blasParty S h with
  | some q =>
      if blasBase S && readyIn S q && S.contains .oracles && roomAt S pa 4 && roomAt S pb 4 then
        some (afterOp S q, cuOk)
      else if blasCapBase S && S.contains .oracles && roomAt S pa 4 && roomAt S pb 4 then
        match capOf S, ids with
        | some (cq, j, r, evs), some is =>
            if capIn cq j q then
              match r.op q is is.getLast?.toList with
              | some r' => some (withCap S cq j r' evs, cuOk)
              | none => none
            else none
        | _, _ => none
      else none
  | none => none

/-- A cuBLAS call on a handle alone, decided; its arguments must be known. -/
def blasHandleRow (S : TState) (f : CublasFn) (as : List AV) : Option (TState × AV) :=
  match f, as.mapM AV.bits with
  | .create, some [p] =>
      match nblasOf S with
      | some k =>
          if blasBase S && roomAt S p 8 then
            some (.nblas (k + 1) :: .blasOn k 0 ::
              (afterStore (some p) (some (drvHandle kBlas k)) 8 S).filter (fun | .nblas _ => false | _ => true),
              cuOk)
          else none
      | none => none
  | .destroy, some [h] =>
      match handleOf kBlas h with
      | some i =>
          if blasBase S && (blasOnOf S i).isSome then
            some (S.filter (fun | .blasOn j _ => j != i | _ => true), cuOk)
          else none
      | none => none
  | .setStream, some [h, s] =>
      match handleOf kBlas h with
      | some i =>
          if blasBase S && (blasOnOf S i).isSome && (drvParty S s).isSome then
            some (.blasOn i s :: S.filter (fun | .blasOn j _ => j != i | _ => true), cuOk)
          else none
      | none => none
  | _, _ => none

/-- **A cuBLAS call, decided**: the typestate after it and its answer, where
    the call's precondition is shown; `none` otherwise. A product's handle and
    scalars must be known numbers, and what it needs of its other arguments'
    values is `blasVal`'s; every other call's arguments must be known. -/
def blasRow (S : TState) (f : CublasFn) (as : List AV) : Option (TState × AV) :=
  match f, as with
  | .sgemv, [h, _, _, _, pa, _, _, _, _, pb, _, _]
  | .sgemm, [h, _, _, _, _, _, pa, _, _, _, _, pb, _, _]
  | .sgemmStridedBatched, [h, _, _, _, _, _, pa, _, _, _, _, _, _, pb, _, _, _, _]
  | .gemmEx, [h, _, _, _, _, _, pa, _, _, _, _, _, _, pb, _, _, _, _, _]
  | .gemmStridedBatchedEx, [h, _, _, _, _, _, pa, _, _, _, _, _, _, _, _, pb, _, _, _, _, _, _, _] =>
      match h.bits, pa.bits, pb.bits with
      | some h, some pa, some pb => productRow S h pa pb (blasIds S f (as.map AV.bits))
      | _, _, _ => none
  | _, _ => blasHandleRow S f as

/-- The driver calls that keep every allocation: allocations, copies,
    fills, launches and waits. -/
def _root_.AlgorithmLib.IR.CudaFn.keepsAllocs : CudaFn → Bool
  | .memAlloc | .memcpyHtoD | .memcpyDtoH | .memcpyDtoD | .memsetD8 | .launchKernel | .ctxSynchronize
  | .streamSynchronize => true
  | _ => false

/-- Whether a vendor call writes no host memory: its frame is empty. -/
def writesNone (e : Ext) : Bool :=
  match e.frame with
  | .none => true
  | _ => false

/-- The vendor calls that keep every allocation: those driver calls, and
    every cuBLAS call (products write their output at its size; handle calls
    change only the handle table). -/
def _root_.AlgorithmLib.IR.Ext.keepsAllocs : Ext → Bool
  | .cuda f => f.keepsAllocs
  | .cublas _ => true
  | _ => false

end AlgorithmLib.Check
