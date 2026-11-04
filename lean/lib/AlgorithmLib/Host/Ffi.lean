module
public import AlgorithmLib.Host.World
meta import AlgorithmLib.Host.World
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# What the foreign calls do

The models the engine's base functions, the device driver and the vendor
libraries are read by (`Ext`): one definition per base function, and the
device's steps. The corpora check them against the real symbols.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

-- ---------------------------------------------------------------------------
-- The FFI
-- ---------------------------------------------------------------------------

/-- The C string at `a`: the bytes up to the first zero, which are UTF-8.
    One that does not end inside its region, or is not UTF-8, is not a
    C string, and a call given it is undefined. -/
def readCStr (m : Mem) (a : UInt64) : Option String := do
  let (r, off) ← decodeAddr a
  let bs := m.region r
  let n := (List.range (bs.size - off)).find? (fun i => bs.get! (off + i) == 0)
  let n ← n
  String.fromUTF8? ((List.range n).map (fun i => bs.get! (off + i))).toByteArray

def copyIn (m : Mem) (a : UInt64) (src : ByteArray) : Option Mem :=
  (List.range src.size).foldlM
    (fun mm i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) m

def copyOut (m : Mem) (a : UInt64) (n : Nat) : Option ByteArray :=
  (List.range n).foldlM
    (fun (acc : ByteArray) i => do
      let b ← m.load (a + UInt64.ofNat i) 1
      pure (acc.push b.toUInt8))
    ByteArray.empty

/-- An `i64` as the eight bytes the implementation stores it in. -/
def le64 (x : UInt64) : ByteArray :=
  ((List.range 8).map (fun i => ((x >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)).toByteArray

/-- The `i64` the first eight bytes of `b` encode. -/
def ofLe64 (b : ByteArray) : UInt64 :=
  (List.range 8).foldr (fun (i : Nat) (acc : UInt64) => (acc <<< 8) ||| (b.get! i).toUInt64) 0

/-- The bytes from `a` up to the first zero. -/
def readCStrAt (m : Mem) (a : UInt64) : Option String := readCStr m a

/-- `n` bytes at `a` as a `ByteArray`, or `none` if they are not all mapped. -/
def readBytes (m : Mem) (a : UInt64) (n : Nat) : Option ByteArray := copyOut m a n

/-- The string in the `cap`-byte field at `a`, as base's own calls take a path
    or a name: its bytes up to the first zero, or all `cap` if there is none
    (`sun_path`'s and `strnlen`'s form). Inner `none` for bytes that are not
    UTF-8, which a call answers as its refusal; outer `none` when the field's
    bytes are not all there. -/
def readField (m : Mem) (a cap : UInt64) : Option (Option String) := do
  let bs ← readBytes m a cap.toNat
  let n := ((List.range bs.size).find? fun i => bs.get! i == 0).getD bs.size
  pure (String.fromUTF8? (bs.extract 0 n))

/-- Writing `bytes` into a file at `at_`, as `write_all` after a seek leaves it:
    a seek past the end leaves a hole that reads back as zeros, and a write
    shorter than what is there does *not* truncate the rest. -/
def spliceAt (prev bytes : ByteArray) (at_ : Nat) : ByteArray :=
  let pad := if at_ > prev.size then at_ - prev.size else 0
  let head := ((List.range (min at_ prev.size)).map prev.get!).toByteArray
  let tail :=
    if at_ + bytes.size < prev.size then
      ((List.range (prev.size - at_ - bytes.size)).map
        (fun i => prev.get! (at_ + bytes.size + i))).toByteArray
    else ByteArray.empty
  head ++ ByteArray.mk (Array.replicate pad 0) ++ bytes ++ tail

/-- The `i64` a `UInt64` denotes, for the arguments the Rust reads as signed. -/
def asI64 (x : UInt64) : Int := signed .i64 x

/-- An `i32` argument as the Rust reads it. -/
def asI32 (x : UInt64) : Int := signed .i32 x

/-- What a context argument is: the live context (`some true`); null
    (`some false`, which the Rust's `read_ctx` refuses, so the call returns
    `-1`); or anything else — a pointer the Rust would dereference, which the
    model does not guess at. -/
def cudaCtxOk (w : World) (ctx : UInt64) : Option Bool :=
  if ctx == 0 then some false
  else if ctx == cudaCtx && w.dev.live then some true
  else none

/-- `src` written over `b` from `off`. -/
def overwrite (b : ByteArray) (off : Nat) (src : ByteArray) : ByteArray :=
  b.extract 0 off ++ src ++ b.extract (off + src.size) b.size

/-- The `i32` result `-1`, the world unchanged. -/
def cudaFail (w : World) : Option (Option V × World) := some (some (ofInt .i32 (-1)), w)

/-- The floats a column-major `rows × cols` matrix with leading dimension `ld`
    spans. -/
def colMajorSpan (rows cols ld : Nat) : Nat :=
  if rows = 0 || cols = 0 then 0 else ld * (cols - 1) + rows

-- file.rs — a path and a buffer, each a pointer.
def ffiFileReadToPtr (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [pathPtr, pathCap, dstPtr, fileOff, size] =>
      if asI64 size ≤ 0 then some (some (ofInt .i64 (-1)), w)
      else do
        let path ← readField w.mem pathPtr pathCap
        match (match path with | some p => if p == "" then none else w.fs.get p | none => none) with
        | none => some (some (ofInt .i64 (-1)), w)
        | some content =>
            let from_ := fileOff.toNat
            let avail := content.size - min from_ content.size
            let n := min size.toNat avail
            let slice := ((List.range n).map (fun i => content.get! (from_ + i))).toByteArray
            let m ← copyIn w.mem dstPtr slice
            some (some (ofInt .i64 n), { w with mem := m })
  | _ => none

def ffiFileWriteFromPtr (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [pathPtr, pathCap, srcPtr, fileOff, size] =>
      if asI64 size ≤ 0 || asI64 fileOff < 0 then some (some (ofInt .i64 (-1)), w)
      else do
        let some path ← readField w.mem pathPtr pathCap | some (some (ofInt .i64 (-1)), w)
        if path == "" then some (some (ofInt .i64 (-1)), w) else
        let bytes ← readBytes w.mem srcPtr size.toNat
        let prev := (w.fs.get path).getD ByteArray.empty
        some (some (ofInt .i64 bytes.size),
              { w with fs := w.fs.set path (spliceAt prev bytes fileOff.toNat) })
  | _ => none

/-- The length of the file at a path, `-1` for none. -/
def ffiFileSize (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [pathPtr, pathCap] => do
      let path ← readField w.mem pathPtr pathCap
      match (match path with | some p => if p == "" then none else w.fs.get p | none => none) with
      | none => some (some (ofInt .i64 (-1)), w)
      | some c => some (some (ofInt .i64 c.size), w)
  | _ => none

/-- The file at a path made `len` bytes long, created if there is none: cut
    short, or extended with zeros. `-1` for a negative length. -/
def ffiFileSetLen (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [pathPtr, pathCap, len] => do
      let some path ← readField w.mem pathPtr pathCap | some (some (ofInt .i64 (-1)), w)
      if path == "" || asI64 len < 0 then some (some (ofInt .i64 (-1)), w) else
      let prev := (w.fs.get path).getD ByteArray.empty
      let n := len.toNat
      let next := if n ≤ prev.size then prev.extract 0 n
        else prev ++ ByteArray.mk (Array.replicate (n - prev.size) 0)
      some (some (ofInt .i64 0), { w with fs := w.fs.set path next })
  | _ => none

/-- `create_dir_all`. Directories are not modelled, so it changes nothing the
    model holds: `-1` when a file stands at the path or at a directory on the
    way to it, `0` otherwise. -/
def ffiFileCreateDirAll (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [pathPtr, pathCap] => do
      let some path ← readField w.mem pathPtr pathCap | some (some (ofInt .i64 (-1)), w)
      if path == "" then some (some (ofInt .i64 0), w) else
      if w.fs.files.any (fun (f, _) => f == path || path.startsWith (f ++ "/")) then
        some (some (ofInt .i64 (-1)), w)
      else some (some (ofInt .i64 0), w)
  | _ => none

/-- A performance control: the system's answer, `0` granted or `-1` declined,
    and nothing else changed. -/
def ffiGrant (key : String) (n : Nat) (bits : List UInt64) (w : World) :
    Option (Option V × World) :=
  if bits.length == n then some (some (ofInt .i32 (if w.grants key then 0 else -1)), w) else none

-- sync.rs — the system's address wait. A wait reads its word and may sleep
-- while it holds what was expected, a wake wakes sleepers on it, and neither
-- answers: a caller rereads the word. Both need a word: four bytes there, at
-- an address a multiple of four.
def ffiMemWait (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [addr, _, _] =>
      if addr % 4 != 0 then none
      else match w.mem.load addr 4 with
        | none => none
        | some _ => some (none, w)
  | _ => none

def ffiMemWake (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [addr, _] =>
      if addr % 4 != 0 then none
      else match w.mem.load addr 4 with
        | none => none
        | some _ => some (none, w)
  | _ => none

def ffiClockNow (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [] => some (some (ofInt .i64 w.clock), w)
  | _ => none

-- stdio.rs — a line comes back with its newline, or as much of it as `cap`
-- holds, the rest left for the next read, as `fgets` leaves it; end of input
-- is `0` and writes nothing, and a `cap` that is not positive is `-1`.
def ffiStdinReadline (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [dst, cap] =>
      if asI64 cap ≤ 0 then some (some (ofInt .i64 (-1)), w)
      else match w.stdin with
        | [] => some (some (ofInt .i64 0), w)
        | line :: rest => do
            let n := min line.size cap.toNat
            let m ← copyIn w.mem dst (((List.range n).map line.get!).toByteArray)
            let left := ((List.range (line.size - n)).map fun i => line.get! (n + i)).toByteArray
            some (some (ofInt .i64 n),
              { w with mem := m, stdin := if n < line.size then left :: rest else rest })
  | _ => none

def ffiStdoutWrite (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [src, size] =>
      if asI64 size < 0 then some (some (ofInt .i64 (-1)), w)
      else do
        let bytes ← readBytes w.mem src size.toNat
        some (some (ofInt .i64 size.toNat), { w with stdout := w.stdout ++ bytes })
  | _ => none

-- cuda — the default stream and created streams, with a race check. Every
-- operation runs when it is issued; the tracker in `Dev.race` makes that the
-- only order the device could have chosen, or the call is stuck. Two things
-- the model cannot know are taken as given: that an allocation succeeds, and
-- that a kernel loads.

/-- The in-flight copies the host has now waited for, dropped: their bytes are
    the host's again. -/
def Mem.retire (m : Mem) (host : Clock) : Mem :=
  { m with busy := m.busy.filter fun b => !decide (b.tick ≤ host.get b.party) }

/-- Memory as a copy on party `p` sees it: that party's own earlier copies are
    ordered before it by the stream, so they do not stand in its way. -/
def Mem.forParty (m : Mem) (p : Nat) : Mem := { m with busy := m.busy.filter (·.party != p) }

/-- A contract that changes only the device. `k` computes the result and the
    new device from the world, which it may read and does not write. -/
def devOnly (w : World) (k : Option (Option V × Dev)) : Option (Option V × World) :=
  k.map fun (r, d) => (r, { w with dev := d, mem := w.mem.retire (d.race.clock hostParty) })

/-- `-1`, the device unchanged. -/
def devFail (d : Dev) : Option (Option V × Dev) := some (some (ofInt .i32 (-1)), d)

/-- `0` and a new device. -/
def devOk (d : Dev) : Option (Option V × Dev) := some (some (ofInt .i32 0), d)

/-- An `i64` result, the device unchanged. -/
def devI64 (d : Dev) (x : Int) : Option (Option V × Dev) := some (some (ofInt .i64 x), d)

/-- The party a stream id names: `-1` and below the default stream, otherwise a
    live created stream. -/
def Dev.party? (d : Dev) (sid : Int) : Option Nat :=
  if sid < 0 then some defaultParty
  else if d.streams.getD sid.toNat false then some (streamParty sid.toNat) else none

/-- Whether `p` is recording into the capture in progress. -/
def Dev.capturing (d : Dev) (p : Nat) : Bool :=
  match d.capture with
  | some c => c.origin == p || c.joined.contains p
  | none => false

/-- Run a kernel launch now: its bound buffers get the oracle's contents, kept
    at their lengths. -/
def Dev.runLaunch (d : Dev) (kernel : Launch → List ByteArray → List ByteArray)
    (l : Launch) (ids : List Nat) : Option Dev := do
  let ins ← ids.mapM (fun i => d.get? (Int.ofNat i))
  let outs := kernel l ins
  if outs.length != ins.length || (outs.zip ins).any (fun (o, i) => o.size != i.size) then none
  else some ((ids.zip outs).foldl (fun d (id, o) => d.put id (some o)) d)

/-- Run a vendor routine now: its output buffer gets the oracle's contents, kept
    at its length. -/
def Dev.runVendor (d : Dev) (vendor : VendorCall → List ByteArray → ByteArray)
    (c : VendorCall) (ins : List Nat) (out : Nat) : Option Dev := do
  let bs ← ins.mapM (fun i => d.get? (Int.ofNat i))
  let old ← d.get? (Int.ofNat out)
  let new := vendor c bs
  if new.size != old.size then none else some (d.put out (some new))

/-- One device operation on `p` that reads `rs` and writes `ws`: recorded if `p`
    is capturing, checked against the capture's own order; otherwise checked
    against everything issued so far and run. -/
def Dev.devOp (d : Dev) (w : World) (p : Nat) (op : DevOp) (rs ws : List Nat) : Option Dev := do
  match d.capture with
  | some c =>
      if d.capturing p then
        let r ← c.race.op p rs ws
        some { d with capture := some { c with race := r, ops := c.ops.push op } }
      else run
  | none => run
where
  run : Option Dev := do
    let r ← d.race.op p rs ws
    let d := { d with race := r }
    match op with
    | .launch l ids => d.runLaunch w.kernel l ids
    | .vendor c ins out => d.runVendor w.vendor c ins out

/-- A synchronous upload of `bytes` over buffer `id` from `off`: a write on the
    default stream that the host then waits for. -/
def Dev.syncWrite (d : Dev) (id : Nat) (b : ByteArray) : Option Dev := do
  let r ← d.race.op defaultParty [] [id]
  some { (d.put id (some b)) with race := r.sync defaultParty }

/-- A synchronous download reads buffer `id` on the default stream, and the host
    waits for it. -/
def Dev.syncRead (d : Dev) (id : Nat) : Option Dev := do
  let r ← d.race.op defaultParty [id] []
  some { d with race := r.sync defaultParty }

/-- An `i64` result. -/
def cudaI64 (w : World) (x : Int) : Option (Option V × World) := some (some (ofInt .i64 x), w)

/-- Exactly `n` bytes of `b`, padded with zeros. -/
def exactly (b : ByteArray) (n : Nat) : ByteArray :=
  b.extract 0 n ++ ByteArray.mk (Array.replicate (n - b.size) 0)

/-- `exactly b n` has `n` bytes. -/
theorem exactly_size (b : ByteArray) (n : Nat) : (exactly b n).size = n := by
  rw [exactly, ByteArray.size_append, ByteArray.size_extract]
  have : (ByteArray.mk (Array.replicate (n - b.size) 0)).size = n - b.size := Array.size_replicate ..
  rw [this]; omega

/-- `cuMemGetInfo`'s free or total, from the world; `-1` when the device reports
    no memory at all, as the Rust does. -/
def memInfoOf (w : World) (ctx : UInt64) (pick : UInt64 × UInt64 → UInt64) :
    Option (Option V × Dev) := do
  let ok ← cudaCtxOk w ctx
  if !ok || w.memInfo.2 == 0 then devI64 w.dev (-1)
  else devI64 w.dev (pick w.memInfo).toNat

/-- The buffer ids `n` `i32` bindings name, read from memory. -/
def readIds (m : Mem) (bindPtr : UInt64) (n : Nat) : Option (List Int) := do
  let raw ← (List.range n).mapM (fun i => m.load (bindPtr + UInt64.ofNat (4 * i)) 4)
  some (raw.map asI32)

/-- The pinned bytes a host address range covers, if it lies in pinned memory. -/
def pinnedSpan (a : UInt64) (n : Nat) : Option (Nat × Nat) :=
  match decodeAddr a with
  | some (.pinned, off) => some (off, n)
  | _ => none

/-- An asynchronous copy on party `p` of `n` host bytes at `hostAddr`, once the
    device side is known good: `k` does the device side and the host side
    given the race state after the operation is issued. From pinned memory the
    copy runs while the host goes on, so its bytes stay busy until the host has
    waited for it; from pageable memory the driver stages or finishes the copy
    before returning. During a capture the copy would become a graph node that
    reads host memory when the graph runs: stuck. -/
def asyncCopy (w : World) (p : Nat) (hostAddr : UInt64) (n : Nat) (writes : Bool)
    (rs ws : List Nat) (k : Race → Option (Dev × Mem)) : Option (Option V × World) := do
  if w.dev.capture.isSome && w.dev.capturing p then none
  else
    let r ← w.dev.race.op p rs ws
    let (d, m) ← k r
    let busy := match pinnedSpan hostAddr n with
      | some (off, len) => { off, len, party := p, tick := (r.clock p).get p, writes } :: w.mem.busy
      | none => w.mem.busy
    some (some (ofInt .i32 0), { w with dev := d, mem := { m with busy } })

-- The asynchronous copies. From pageable memory an upload is staged before the
-- call returns and a download has finished; from pinned memory neither has, and
-- until the host waits on the stream it may not write the bytes, nor read what
-- a download is writing. A later copy on the same stream is ordered after them.
/-- The event an `i32` id names, if it exists. -/
def Dev.event? (d : Dev) (id : Int) : Option (Option (Clock × Bool)) :=
  if id < 0 then none else (d.events[id.toNat]?).join

/-- Every stream that joined the capture has been joined back by the origin:
    the origin's clock covers each one's recorded work. -/
def Capture.rejoined (c : Capture) : Bool :=
  c.joined.all fun q => decide ((c.race.clock q).get q ≤ (c.race.clock c.origin).get q)

/-- An event recorded inside a capture names a node of that capture; once the
    capture ends its clock belongs to no history a later wait could join, so it
    is emptied and such a wait orders nothing. -/
def Dev.endCapture (d : Dev) : Dev :=
  { d with capture := none,
           events := d.events.map fun e => match e with
             | some (some (_, true)) => some (some (#[], true))
             | e => e }

/-- What a graph touches, for ordering it against everything else: the vendor
    routines' inputs are read; every kernel binding and every output written. -/
def graphAccess (ops : Array DevOp) : List Nat × List Nat :=
  ops.foldl (fun (rs, ws) op => match op with
    | .launch _ ids => (rs, ws ++ ids)
    | .vendor _ ins out => (rs ++ ins, ws ++ [out])) ([], [])

/-- Whether an `sgemv` over an `M × N` column-major `A` stays inside its three
    buffers. -/
def sgemvFits (trans : UInt64) (M N : Nat) (A X Y : ByteArray) : Bool :=
  let xl := if asI32 trans != 0 then M else N
  let yl := if asI32 trans != 0 then N else M
  decide (4 * (M * N) ≤ A.size) && decide (4 * xl ≤ X.size) && decide (4 * yl ≤ Y.size)

/-- A `sgemv` on party `p`, once the context is known live. -/
def sgemvOn (w : World) (p : Nat) (trans m n alpha a x beta y : UInt64) : Option (Option V × Dev) :=
  match w.dev.get? (asI32 a), w.dev.get? (asI32 x), w.dev.get? (asI32 y) with
  | some A, some X, some Y =>
      if asI32 m ≤ 0 || asI32 n ≤ 0
          || !sgemvFits trans (asI32 m).toNat (asI32 n).toNat A X Y then none
      else do
        let (ia, ix, iy) := ((asI32 a).toNat, (asI32 x).toNat, (asI32 y).toNat)
        let d ← w.dev.devOp w p
          (.vendor ⟨"sgemv", [trans, m, n, alpha, beta]⟩ [ia, ix, iy] iy) [ia, ix, iy] [iy]
        devOk d
  | _, _, _ => devFail w.dev

/-- Whether a strided-batched GEMM stays inside its three buffers: every
    batch's slice of each operand, at its offset and stride, with the leading
    dimensions the entry point defaults. Offsets, strides and spans count
    elements of `ea`, `eb` and `ec` bytes. -/
def gemmStridedFits (ea eb ec : Nat) (ta tb m n k sa sb sc batch oa ob oc la lb lc : UInt64)
    (A B C : ByteArray) : Bool :=
  let M := (asI32 m).toNat
  let N := (asI32 n).toNat
  let K := (asI32 k).toNat
  let ra := if asI32 ta != 0 then K else M
  let ca := if asI32 ta != 0 then M else K
  let rb := if asI32 tb != 0 then N else K
  let cb := if asI32 tb != 0 then K else N
  let lda := if asI32 la != 0 then (asI32 la).toNat else ra
  let ldb := if asI32 lb != 0 then (asI32 lb).toNat else rb
  let ldc := if asI32 lc != 0 then (asI32 lc).toNat else M
  let last := (asI32 batch).toNat - 1
  let fits (e off stride span : Nat) (buf : ByteArray) : Bool :=
    decide (e * (off + last * stride + span) ≤ buf.size)
  decide (ra ≤ lda) && decide (rb ≤ ldb) && decide (M ≤ ldc)
    && fits ea (asI64 oa).toNat (asI64 sa).toNat (colMajorSpan ra ca lda) A
    && fits eb (asI64 ob).toNat (asI64 sb).toNat (colMajorSpan rb cb ldb) B
    && fits ec (asI64 oc).toNat (asI64 sc).toNat (colMajorSpan M N ldc) C

/-- `gemmStridedFits` for `f32` operands. -/
def sgemmFits (ta tb m n k sa sb sc batch oa ob oc la lb lc : UInt64) (A B C : ByteArray) : Bool :=
  gemmStridedFits 4 4 4 ta tb m n k sa sb sc batch oa ob oc la lb lc A B C

/-- A strided-batched `sgemm` on party `p`, once its buffers are resolved. -/
def sgemmOn (w : World) (p : Nat)
    (ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc : UInt64) :
    Option (Option V × Dev) :=
  match w.dev.get? (asI32 a), w.dev.get? (asI32 b), w.dev.get? (asI32 c) with
  | some A, some B, some C =>
      if !sgemmFits ta tb m n k sa sb sc batch oa ob oc la lb lc A B C then none
      else do
        let (ia, ib, ic) := ((asI32 a).toNat, (asI32 b).toNat, (asI32 c).toNat)
        let d ← w.dev.devOp w p
          (.vendor ⟨"sgemmStridedBatched",
              [ta, tb, m, n, k, alpha, beta, sa, sb, sc, batch, oa, ob, oc, la, lb, lc]⟩ [ia, ib, ic] ic)
          [ia, ib, ic] [ic]
        devOk d
  | _, _, _ => devFail w.dev

/-- Whether a single `gemmEx` with `bf16` operands and an `f32` result stays
    inside its buffers: `A` and `B` at two bytes an element, `C` at four. -/
def gemmBf16Fits (ta tb m n k oa ob oc la lb lc : UInt64) (A B C : ByteArray) : Bool :=
  let M := (asI32 m).toNat
  let N := (asI32 n).toNat
  let K := (asI32 k).toNat
  let ra := if asI32 ta != 0 then K else M
  let ca := if asI32 ta != 0 then M else K
  let rb := if asI32 tb != 0 then N else K
  let cb := if asI32 tb != 0 then K else N
  let lda := if asI32 la != 0 then (asI32 la).toNat else ra
  let ldb := if asI32 lb != 0 then (asI32 lb).toNat else rb
  let ldc := if asI32 lc != 0 then (asI32 lc).toNat else M
  decide (ra ≤ lda) && decide (rb ≤ ldb) && decide (M ≤ ldc)
    && decide (2 * ((asI64 oa).toNat + colMajorSpan ra ca lda) ≤ A.size)
    && decide (2 * ((asI64 ob).toNat + colMajorSpan rb cb ldb) ≤ B.size)
    && decide (4 * ((asI64 oc).toNat + colMajorSpan M N ldc) ≤ C.size)

/-- Whether the host has waited for everything up to clock `c`. -/
def Race.hostSaw (r : Race) (c : Clock) : Bool :=
  (List.range c.size).all fun q => c.get q ≤ (r.clock hostParty).get q

-- native.rs — placing code and asking about the CPU. That a mapping is
-- granted is taken as given, as a device allocation is.

def ffiNativeLoad (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [src, len] =>
      if src == 0 || asI64 len ≤ 0 then some (some (ofInt .i64 0), w)
      else do
        let code ← readBytes w.mem src (asI64 len).toNat
        some (some (.sc .i64 (nativeAddr w.native.size)), { w with native := w.native.push (some code) })
  | _ => none

/-- The live load at `addr`, if it is one. -/
def World.nativeAt? (w : World) (addr : UInt64) : Option Nat :=
  (List.range w.native.size).find? fun k => nativeAddr k == addr && (w.native[k]?).join.isSome

/-- A live load unmapped. Any other address is undefined: the engine keeps
    no record of what it mapped, so it cannot tell. -/
def ffiNativeFree (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [addr] =>
      match w.nativeAt? addr with
      | none => none
      | some k => some (none, { w with native := w.native.set! k none })
  | _ => none

def ffiNativeArch (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [] => some (some (ofInt .i32 w.arch), w)
  | _ => none

def featureCode : Option Bool → Int
  | some true => 1
  | some false => 0
  | none => -1

def ffiCpuHas (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [name, cap] => do
      let some s ← readField w.mem name cap | some (some (ofInt .i32 (-1)), w)
      some (some (ofInt .i32 (featureCode (w.cpuHas s))), w)
  | _ => none

end AlgorithmLib.HProg.Sem
