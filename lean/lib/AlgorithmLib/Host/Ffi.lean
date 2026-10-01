module
public import AlgorithmLib.Host.World
meta import AlgorithmLib.Host.World
public import AlgorithmLib.Vocab.Libm
meta import AlgorithmLib.Vocab.Libm
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# What the foreign calls do

One definition per entry point with an executable contract, and the dispatch
`callBits`. `Spec.lean` gathers these with the signatures and frames into one
table and proves each writes only inside its frame; the corpora check them
against the real symbols.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

-- ---------------------------------------------------------------------------
-- The FFI
-- ---------------------------------------------------------------------------

/-- The bytes from `a` up to the first zero, as a path. -/
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

/-- The hash-table context, which is a Rust allocation rather than anything in
    the three regions. `decodeAddr` rejects it, so it can be stored and passed
    but never loaded through. -/
def htCtx : UInt64 := 0x5000000000

/-- An `i64` as the eight bytes the implementation stores it in. -/
def le64 (x : UInt64) : ByteArray :=
  ((List.range 8).map (fun i => ((x >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)).toByteArray

/-- The `i64` the first eight bytes of `b` encode. -/
def ofLe64 (b : ByteArray) : UInt64 :=
  (List.range 8).foldr (fun (i : Nat) (acc : UInt64) => (acc <<< 8) ||| (b.get! i).toUInt64) 0

/-- The longest path the file shims scan for its terminating zero. -/
def pathMax : Nat := 4096

/-- A path as the file shims read it: the bytes up to the first zero among the
    first `pathMax`. No zero among them, or bytes that are not UTF-8, give the
    empty path, which every file shim refuses. A region that ends before either
    is a read out of bounds. -/
def readPath (m : Mem) (a : UInt64) : Option String := do
  let (r, off) ← decodeAddr a
  let bs := m.region r
  let avail := bs.size - off
  match (List.range (min avail pathMax)).find? (fun i => bs.get! (off + i) == 0) with
  | some n => some ((String.fromUTF8? ((List.range n).map (fun i => bs.get! (off + i))).toByteArray).getD "")
  | none => if avail < pathMax then none else some ""

/-- The bytes from `a` up to the first zero. -/
def readCStrAt (m : Mem) (a : UInt64) : Option String := readCStr m a

/-- `n` bytes at `a` as a `ByteArray`, or `none` if they are not all mapped. -/
def readBytes (m : Mem) (a : UInt64) (n : Nat) : Option ByteArray := copyOut m a n

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

-- file.rs
def ffiFileRead (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [base, pathOff, dstOff, fileOff, size] => do
      let path ← readPath w.mem (base + pathOff)
      match (if path == "" then none else w.fs.get path) with
      | none => some (some (ofInt .i64 (-1)), w)
      | some content =>
          let from_ := fileOff.toNat
          let avail := content.size - min from_ content.size
          let want := if size == 0 then avail else size.toNat
          let n := min want avail
          let slice := ((List.range n).map (fun i => content.get! (from_ + i))).toByteArray
          let m ← copyIn w.mem (base + dstOff) slice
          some (some (ofInt .i64 n), { w with mem := m })
  | _ => none

def ffiFileWrite (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [base, pathOff, srcOff, fileOff, size] => do
      let path ← readPath w.mem (base + pathOff)
      -- `size == 0` means "the bytes up to the first NUL", and writing at
      -- offset 0 goes through `File::create`, which truncates.
      if path == "" then some (some (ofInt .i64 (-1)), w) else
      let bytes ←
        if size == 0 then (readCStrAt w.mem (base + srcOff)).map (·.toUTF8)
        else readBytes w.mem (base + srcOff) size.toNat
      let at_ := fileOff.toNat
      let prev := if at_ == 0 then ByteArray.empty else (w.fs.get path).getD ByteArray.empty
      some (some (ofInt .i64 bytes.size), { w with fs := w.fs.set path (spliceAt prev bytes at_) })
  | _ => none

-- These two take the path and the buffer as raw pointers, so neither is
-- relative to the shared arena.
def ffiFileReadToPtr (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [pathPtr, dstPtr, fileOff, size] =>
      if asI64 size ≤ 0 then some (some (ofInt .i64 (-1)), w)
      else do
        let path ← readPath w.mem pathPtr
        match (if path == "" then none else w.fs.get path) with
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
  | [pathPtr, srcPtr, fileOff, size] =>
      if asI64 size ≤ 0 || asI64 fileOff < 0 then some (some (ofInt .i64 (-1)), w)
      else do
        let path ← readPath w.mem pathPtr
        if path == "" then some (some (ofInt .i64 (-1)), w) else
        let bytes ← readBytes w.mem srcPtr size.toNat
        let prev := (w.fs.get path).getD ByteArray.empty
        some (some (ofInt .i64 bytes.size),
              { w with fs := w.fs.set path (spliceAt prev bytes fileOff.toNat) })
  | _ => none

/-- `create_dir_all`. Directories are not modelled, so it changes nothing the
    model holds: `-1` when a file stands at the path or at a directory on the
    way to it, `0` otherwise. -/
def ffiFileCreateDirAll (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [pathPtr] => do
      let path ← readPath w.mem pathPtr
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

-- stdio.rs — a line comes back with its newline, capped at `max_len - 1` and
-- terminated; end of input is `0` and writes nothing.
def ffiStdinReadline (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [base, dstOff, maxLen] =>
      if asI64 maxLen ≤ 0 then some (some (ofInt .i64 0), w)
      else match w.stdin with
        | [] => some (some (ofInt .i64 0), w)
        | line :: rest => do
            let n := min line.size (maxLen.toNat - 1)
            let m ← copyIn w.mem (base + dstOff) (((List.range n).map line.get!).toByteArray)
            let m ← m.store (base + dstOff + UInt64.ofNat n) 1 0
            some (some (ofInt .i64 n), { w with mem := m, stdin := rest })
  | _ => none

def ffiStdoutWrite (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [base, srcOff, size] =>
      if asI64 size < 0 then some (some (ofInt .i64 (-1)), w)
      else do
        let bytes ← readBytes w.mem (base + srcOff) size.toNat
        some (some (ofInt .i64 size.toNat), { w with stdout := w.stdout ++ bytes })
  | _ => none

-- `Libm` states the three functions as IEEE operations, which `Lib.Math`
-- performs instruction for instruction: the same bits on every host.
def ffiSinf (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [x] => some (some (.sc .f32 (Libm.sinf x.toUInt32).toUInt64), w)

  | _ => none

def ffiCosf (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [x] => some (some (.sc .f32 (Libm.cosf x.toUInt32).toUInt64), w)

  | _ => none

def ffiPowf (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [b, e] => some (some (.sc .f32 (Libm.powf b.toUInt32 e.toUInt32).toUInt64), w)

  | _ => none

-- ht.rs — the context is not addressable memory, so `init` writes a sentinel
-- that no region decodes: a program that dereferences the handle or does
-- arithmetic on it goes stuck here rather than quietly agreeing with a run
-- that would have faulted.
def ffiHtInit (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] => do
      let m ← w.mem.store slot 8 htCtx
      some (none, { w with mem := m, ht := {} })
  | _ => none

def ffiHtCleanup (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] => do
      let m ← w.mem.store slot 8 0
      some (none, { w with mem := m, ht := {} })
  | _ => none

def ffiHtCreate (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx] =>
      if ctx != htCtx then some (some (ofInt .i32 0xFFFFFFFF), w)
      else
        let handle := w.ht.handles.length
        some (some (ofInt .i32 handle),
              { w with ht := { w.ht with handles := w.ht.handles ++ [handle] } })
  | _ => none

def ffiHtCount (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx] =>
      if ctx != htCtx || !w.ht.hasTable then some (some (ofInt .i32 0), w)
      else some (some (ofInt .i32 w.ht.entries.length), w)
  | _ => none

def ffiHtLookup (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, keyPtr, keyLen, resultPtr] => do
      let key ← readBytes w.mem keyPtr keyLen.toNat
      if ctx != htCtx || !w.ht.hasTable then some (some (ofInt .i32 0xFFFFFFFF), w)
      else match w.ht.get key with
        | none => some (some (ofInt .i32 0xFFFFFFFF), w)
        | some val => do
            let m ← copyIn w.mem resultPtr val
            some (some (ofInt .i32 val.size), { w with mem := m })
  | _ => none

def ffiHtInsert (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, keyPtr, keyLen, valPtr, valLen] => do
      let key ← readBytes w.mem keyPtr keyLen.toNat
      let val ← readBytes w.mem valPtr valLen.toNat
      if ctx != htCtx || !w.ht.hasTable then some (none, w)
      else some (none, { w with ht := w.ht.set key val })
  | _ => none

def ffiHtIncrement (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, keyPtr, keyLen, addend] => do
      let key ← readBytes w.mem keyPtr keyLen.toNat
      if ctx != htCtx || !w.ht.hasTable then some (some (.sc .i64 addend), w)
      else match w.ht.get key with
        | none =>
            some (some (.sc .i64 addend), { w with ht := w.ht.set key (le64 addend) })
        | some existing =>
            -- The counter is the first eight bytes; anything the value carries
            -- past them is left alone. A value shorter than eight bytes indexes
            -- out of range in the Rust, so there is nothing here to transcribe.
            if existing.size < 8 then none
            else
              let next := ofLe64 existing + addend
              let updated := le64 next ++
                ((List.range (existing.size - 8)).map (fun i => existing.get! (8 + i))).toByteArray
              some (some (.sc .i64 next), { w with ht := w.ht.set key updated })
  | _ => none

def ffiHtGetEntry (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, index, keyOut, valOut] =>
      if ctx != htCtx || !w.ht.hasTable then some (some (ofInt .i32 (-1)), w)
      else match w.ht.entries[index.toNat]? with
        | none => some (some (ofInt .i32 (-1)), w)
        | some (k, v) => do
            let m ← copyIn w.mem keyOut k
            let m ← copyIn m valOut v
            some (some (ofInt .i32 k.size), { w with mem := m })
  | _ => none

-- lmdb.rs — each environment is one directory holding one unnamed database.
-- While a write transaction is open on a handle, reads through it see the
-- transaction and puts go into it; commit makes it what the directory holds,
-- and cleanup drops it. The model takes three things as given: the map never
-- fills, a commit succeeds, and no directory is open twice at once — LMDB
-- forbids that within a process, and the model is stuck instead.

/-- LMDB's default key order: byte by byte, a key before every longer key it
    begins. -/
def lmdbKeyLt : List UInt8 → List UInt8 → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | a :: as, b :: bs => a < b || (a == b && lmdbKeyLt as bs)

/-- `k` stored as `v`: overwritten in place, or inserted in key order. -/
def lmdbPutIn (t : LmdbTable) (k v : ByteArray) : LmdbTable :=
  match t with
  | [] => [(k, v)]
  | e :: rest =>
      if e.1.toList == k.toList then (k, v) :: rest
      else if lmdbKeyLt k.toList e.1.toList then (k, v) :: e :: rest
      else e :: lmdbPutIn rest k v

/-- A key LMDB accepts: one to 511 bytes. Any other size is `MDB_BAD_VALSIZE`. -/
def lmdbKeyOk (k : ByteArray) : Bool := 0 < k.size && k.size ≤ 511

/-- What a directory holds, committed. -/
def World.lmdbTable (w : World) (path : String) : LmdbTable :=
  ((w.lmdbDisk.find? (·.1 == path)).map (·.2)).getD []

def World.setLmdbTable (w : World) (path : String) (t : LmdbTable) : World :=
  { w with lmdbDisk := (path, t) :: w.lmdbDisk.filter (·.1 != path) }

/-- The context argument, as `cudaCtxOk` reads its own. -/
def lmdbCtxOk (w : World) (ctx : UInt64) : Option Bool :=
  if ctx == 0 then some false
  else if ctx == lmdbCtx && w.lmdb.live then some true
  else none

/-- The environment a handle names; the Rust reads the handle as `u32`. -/
def lmdbEnv? (w : World) (handle : UInt64) : Option (Nat × LmdbEnv) :=
  let h := handle.toUInt32.toNat
  (w.lmdb.envs[h]?).map (h, ·)

def World.setLmdbEnv (w : World) (h : Nat) (e : LmdbEnv) : World :=
  { w with lmdb := { w.lmdb with envs := w.lmdb.envs.set! h e } }

/-- What reads through an environment see. -/
def lmdbView (w : World) (e : LmdbEnv) : LmdbTable := e.txn.getD (w.lmdbTable e.path)

def lmdbI32 (w : World) (x : Int) : Option (Option V × World) := some (some (ofInt .i32 x), w)

/-- `x` as `n` little-endian bytes. -/
def leBytes (n x : Nat) : ByteArray :=
  ((List.range n).map (fun i => (x / 256 ^ i % 256).toUInt8)).toByteArray

/-- A scan's answer as the runtime lays it out: the count as `u32`, then per
    entry the key and value lengths as `u16` and the key and value bytes. -/
def lmdbScanBytes (rows : LmdbTable) : ByteArray :=
  rows.foldl (fun acc r => acc ++ leBytes 2 r.1.size ++ leBytes 2 r.2.size ++ r.1 ++ r.2)
    (leBytes 4 rows.length)

/-- The rows that fit in `room` bytes after the count, each taking its
    lengths and bytes: the longest prefix that does. -/
def lmdbFit (room : Nat) : LmdbTable → LmdbTable
  | [] => []
  | r :: rs =>
      let n := 4 + r.1.size + r.2.size
      if n ≤ room then r :: lmdbFit (room - n) rs else []

-- A second `init` while a context is live leaks it with its environments open,
-- and reopening one of them is what LMDB forbids; the model is stuck.
def ffiLmdbInit (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] =>
      if w.lmdb.live then none
      else do
        let m ← w.mem.store slot 8 lmdbCtx
        some (none, { w with mem := m, lmdb := { live := true } })
  | _ => none

-- Open write transactions are aborted: what they held is gone.
def ffiLmdbCleanup (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] => do
      let c ← w.mem.load slot 8
      if c == 0 then some (none, w)
      else if c == lmdbCtx && w.lmdb.live then do
        let m ← w.mem.store slot 8 0
        some (none, { w with mem := m, lmdb := {} })
      else none
  | _ => none

-- The directory is made with every one missing on the way, which fails where a
-- file stands at the path or on the way to it; a path of 4096 bytes or more,
-- and an empty one, open nothing.
def ffiLmdbOpen (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, pathPtr, _] =>
      match lmdbCtxOk w ctx with
      | none => none
      | some false => lmdbI32 w (-1)
      | some true => do
          let path ← readCStr w.mem pathPtr
          if path.isEmpty || 4096 ≤ path.utf8ByteSize ||
              w.fs.files.any (fun (f, _) => f == path || path.startsWith (f ++ "/")) then lmdbI32 w (-1)
          else if w.lmdb.envs.any (·.path == path) then none
          else
            some (some (ofInt .i32 w.lmdb.envs.size),
                  { w with lmdb := { w.lmdb with envs := w.lmdb.envs.push { path } } })
  | _ => none

-- An open transaction on the handle is aborted first.
def ffiLmdbBeginWriteTxn (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, handle] =>
      match lmdbCtxOk w ctx with
      | none => none
      | some false => lmdbI32 w (-1)
      | some true =>
          match lmdbEnv? w handle with
          | none => lmdbI32 w (-1)
          | some (h, e) =>
              some (some (ofInt .i32 0), w.setLmdbEnv h { e with txn := some (w.lmdbTable e.path) })
  | _ => none

-- Into the open transaction, or into the directory in one of its own.
def ffiLmdbPut (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, handle, keyPtr, keyLen, valPtr, valLen] =>
      if asI32 keyLen < 0 || asI32 valLen < 0 || keyPtr == 0 || valPtr == 0 then lmdbI32 w (-1)
      else match lmdbCtxOk w ctx with
      | none => none
      | some false => lmdbI32 w (-1)
      | some true =>
          match lmdbEnv? w handle with
          | none => lmdbI32 w (-1)
          | some (h, e) => do
              let key ← readBytes w.mem keyPtr (asI32 keyLen).toNat
              let val ← readBytes w.mem valPtr (asI32 valLen).toNat
              if !lmdbKeyOk key then lmdbI32 w (-1)
              else match e.txn with
                | some t => some (some (ofInt .i32 0), w.setLmdbEnv h { e with txn := some (lmdbPutIn t key val) })
                | none => some (some (ofInt .i32 0), w.setLmdbTable e.path (lmdbPutIn (w.lmdbTable e.path) key val))
  | _ => none

def ffiLmdbCommitWriteTxn (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, handle] =>
      match lmdbCtxOk w ctx with
      | none => none
      | some false => lmdbI32 w (-1)
      | some true =>
          match lmdbEnv? w handle with
          | none => lmdbI32 w (-1)
          | some (h, e) =>
              match e.txn with
              | none => lmdbI32 w (-1)
              | some t => some (some (ofInt .i32 0), (w.setLmdbEnv h { e with txn := none }).setLmdbTable e.path t)
  | _ => none

-- Entries from the first key at or after the start key (every entry when the
-- length is not positive), at most `max` of them read as `u32`, stopping
-- before any longer than a `u16` can count and before the first that does not
-- fit in the `cap` bytes at the result. Less than four bytes, room for no
-- count, answers `-1` and writes nothing. A start key LMDB would refuse is an
-- error code the model does not guess at.
def ffiLmdbCursorScan (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, handle, keyPtr, keyLen, maxEntries, resultPtr, cap] =>
      match lmdbCtxOk w ctx with
      | none => none
      | some false => lmdbI32 w 0
      | some true =>
          if cap.toNat < 4 then lmdbI32 w (-1) else
          match lmdbEnv? w handle with
          | none => do
              let m ← copyIn w.mem resultPtr (leBytes 4 0)
              some (some (ofInt .i32 0), { w with mem := m })
          | some (_, e) => do
              let start ←
                if 0 < asI32 keyLen then (readBytes w.mem keyPtr (asI32 keyLen).toNat).map some
                else some none
              if start.any (!lmdbKeyOk ·) then none
              else
                let rows := (lmdbView w e).filter fun r => start.all fun s => !lmdbKeyLt r.1.toList s.toList
                let rows := lmdbFit (cap.toNat - 4) ((rows.take maxEntries.toUInt32.toNat).takeWhile
                  fun r => r.1.size ≤ 65535 && r.2.size ≤ 65535)
                let m ← copyIn w.mem resultPtr (lmdbScanBytes rows)
                some (some (ofInt .i32 rows.length), { w with mem := m })
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

/-- A context, on a machine with a CUDA device; null without one. -/
def ffiCudaInit (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] =>
      if w.cudaDevice then do
        let m ← w.mem.store slot 8 cudaCtx
        some (none, { w with mem := m, dev := { live := true } })
      else do
        let m ← w.mem.store slot 8 0
        some (none, { w with mem := m, dev := {} })
  | _ => none

/-- Dropping the context frees its pinned allocations too. -/
def ffiCudaCleanup (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] => do
      let m ← w.mem.store slot 8 0
      some (none, { w with mem := { m with pinnedLive := [] }, dev := {} })
  | _ => none

def ffiCudaCreateBuffer (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, size] =>
      if asI64 size ≤ 0 then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else
          some (some (ofInt .i32 w.dev.bufs.size),
                { w.dev with bufs := w.dev.bufs.push (some (ByteArray.mk (Array.replicate size.toNat 0))) })
  | _ => none

/-- A synchronous upload of `bytes` over buffer `id` from `off`: a write on the
    default stream that the host then waits for. -/
def Dev.syncWrite (d : Dev) (id : Nat) (b : ByteArray) : Option Dev := do
  let r ← d.race.op defaultParty [] [id]
  some { (d.put id (some b)) with race := r.sync defaultParty }

def ffiCudaUpload (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, buf, src, size] =>
      if asI32 buf < 0 || asI64 size ≤ 0 || src == 0 then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else match w.dev.get? (asI32 buf) with
          | none => devFail w.dev
          | some b =>
              -- A length other than the buffer's is refused.
              if b.size != size.toNat then devFail w.dev
              else do
                let bytes ← readBytes w.mem src size.toNat
                let d ← w.dev.syncWrite (asI32 buf).toNat bytes
                devOk d
  | _ => none

def ffiCudaUploadOffset (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, buf, off, src, size] =>
      if asI32 buf < 0 || asI64 off < 0 || asI64 size ≤ 0 || src == 0 then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else match w.dev.get? (asI32 buf) with
          | none => devFail w.dev
          | some b =>
              if off.toNat + size.toNat > b.size then devFail w.dev
              else do
                let bytes ← readBytes w.mem src size.toNat
                let d ← w.dev.syncWrite (asI32 buf).toNat (overwrite b off.toNat bytes)
                devOk d
  | _ => none

/-- A synchronous download reads buffer `id` on the default stream, and the host
    waits for it. -/
def Dev.syncRead (d : Dev) (id : Nat) : Option Dev := do
  let r ← d.race.op defaultParty [id] []
  some { d with race := r.sync defaultParty }

def ffiCudaDownload (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, buf, dst, size] =>
      if asI32 buf < 0 || asI64 size ≤ 0 || dst == 0 then cudaFail w
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then cudaFail w
        else match w.dev.get? (asI32 buf) with
          | none => cudaFail w
          | some b =>
              if b.size != size.toNat then cudaFail w
              else do
                let d ← w.dev.syncRead (asI32 buf).toNat
                let m ← copyIn w.mem dst b
                some (some (ofInt .i32 0), { w with mem := m, dev := d })
  | _ => none

def ffiCudaDownloadOffset (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, buf, off, dst, size] =>
      if asI32 buf < 0 || asI64 off < 0 || asI64 size ≤ 0 || dst == 0 then cudaFail w
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then cudaFail w
        else match w.dev.get? (asI32 buf) with
          | none => cudaFail w
          | some b =>
              if off.toNat + size.toNat > b.size then cudaFail w
              else do
                let d ← w.dev.syncRead (asI32 buf).toNat
                let m ← copyIn w.mem dst (b.extract off.toNat (off.toNat + size.toNat))
                some (some (ofInt .i32 0), { w with mem := m, dev := d })
  | _ => none

/-- Freeing is a write on the default stream: work still using the buffer on
    another stream, unordered with it, is a race. -/
def ffiCudaFreeBuffer (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, buf] =>
      if asI32 buf < 0 then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else match w.dev.get? (asI32 buf) with
          | none => devFail w.dev
          | some _ => do
              let r ← w.dev.race.op defaultParty [] [(asI32 buf).toNat]
              devOk { (w.dev.put (asI32 buf).toNat none) with race := r }
  | _ => none

/-- `cl_cuda_sync` waits for the default stream, and only for it: created
    streams are non-blocking. -/
def ffiCudaSync (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else devOk { w.dev with race := w.dev.race.sync defaultParty }
  | _ => none

/-- An `i64` result. -/
def cudaI64 (w : World) (x : Int) : Option (Option V × World) := some (some (ofInt .i64 x), w)

/-- Exactly `n` bytes of `b`, padded with zeros. -/
def exactly (b : ByteArray) (n : Nat) : ByteArray :=
  b.extract 0 n ++ ByteArray.mk (Array.replicate (n - b.size) 0)

def ffiCudaPinnedAlloc (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, size] =>
      if asI64 size ≤ 0 then cudaFail w
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then cudaFail w
        else
          -- Allocations start on a 64-byte boundary, the way page-locked
          -- memory is at least aligned; one past the pinned span is stuck.
          let pad := (64 - w.mem.pinned.size % 64) % 64
          let off := w.mem.pinned.size + pad
          let n := size.toNat
          if off + n > regionSpan.toNat then none
          else
            let bytes := exactly (w.pinnedFill w.dev.pinned.size n) n
            let m := { w.mem with
              pinned := w.mem.pinned ++ (ByteArray.mk (Array.replicate pad 0) ++ bytes)
              pinnedLive := (off, n) :: w.mem.pinnedLive }
            some (some (ofInt .i32 w.dev.pinned.size),
                  { w with mem := m, dev := { w.dev with pinned := w.dev.pinned.push (some (off, n)) } })
  | _ => none

/-- The live pinned allocation an `i32` id names. -/
def Dev.pinnedAt? (d : Dev) (id : Int) : Option (Nat × Nat) :=
  if id < 0 then none else (d.pinned[id.toNat]?).join

def ffiCudaPinnedPtr (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, id] =>
      if asI32 id < 0 then devI64 w.dev (-1)
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devI64 w.dev (-1)
        else match w.dev.pinnedAt? (asI32 id) with
          | none => devI64 w.dev (-1)
          | some (off, _) => devI64 w.dev (addrOf .pinned off).toNat
  | _ => none

def ffiCudaPinnedPtrAt (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, id, off, len] =>
      if asI32 id < 0 || asI64 off < 0 || asI64 len < 0 then devI64 w.dev (-1)
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devI64 w.dev (-1)
        else match w.dev.pinnedAt? (asI32 id) with
          | none => devI64 w.dev (-1)
          | some (o, n) =>
              if off.toNat + len.toNat > n then devI64 w.dev (-1)
              else devI64 w.dev (addrOf .pinned (o + off.toNat)).toNat
  | _ => none

def ffiCudaPinnedFree (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, id] =>
      if asI32 id < 0 then cudaFail w
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then cudaFail w
        else match w.dev.pinnedAt? (asI32 id) with
          | none => cudaFail w
          | some r =>
              some (some (ofInt .i32 0),
                    { w with mem := { w.mem with pinnedLive := w.mem.pinnedLive.erase r },
                             dev := { w.dev with pinned := w.dev.pinned.setIfInBounds (asI32 id).toNat none } })
  | _ => none

/-- `cuMemGetInfo`'s free or total, from the world; `-1` when the device reports
    no memory at all, as the Rust does. -/
def memInfoOf (w : World) (ctx : UInt64) (pick : UInt64 × UInt64 → UInt64) :
    Option (Option V × Dev) := do
  let ok ← cudaCtxOk w ctx
  if !ok || w.memInfo.2 == 0 then devI64 w.dev (-1)
  else devI64 w.dev (pick w.memInfo).toNat

def ffiCudaMemInfoFree (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx] => memInfoOf w ctx Prod.fst
  | _ => none

def ffiCudaMemInfoTotal (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx] => memInfoOf w ctx Prod.snd
  | _ => none

/-- The buffer ids `n` `i32` bindings name, read from memory. -/
def readIds (m : Mem) (bindPtr : UInt64) (n : Nat) : Option (List Int) := do
  let raw ← (List.range n).mapM (fun i => m.load (bindPtr + UInt64.ofNat (4 * i)) 4)
  some (raw.map asI32)

/-- A launch on party `p` once its kernel text and entry point are read: the
    bindings must name live buffers, which the kernel reads and writes. -/
def cudaLaunchOn (w : World) (p : Nat) (kernel entry : String) (nBufs bindPtr : UInt64)
    (dims : List UInt64) : Option (Option V × Dev) := do
  let ids ← readIds w.mem bindPtr (asI32 nBufs).toNat
  if ids.any (fun id => (w.dev.get? id).isNone) then devFail w.dev
  else
    let ns := ids.map Int.toNat
    let d ← w.dev.devOp w p (.launch ⟨kernel, entry, ns, dims, []⟩ ns) [] ns
    devOk d

def launchDimsBad (nBufs : UInt64) (dims : List UInt64) : Bool :=
  asI32 nBufs < 0 || dims.any (fun d => asI32 d ≤ 0)

def ffiCudaLaunch (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] =>
      if launchDimsBad nBufs [gx, gy, gz, bx, by_, bz] then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else do
          let kernel ← readCStrAt w.mem kptr
          cudaLaunchOn w defaultParty kernel "main" nBufs bindPtr [gx, gy, gz, bx, by_, bz]
  | _ => none

/-- The same launch with the entry point named in memory rather than `main`. -/
def ffiCudaLaunchNamed (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, kptr, namePtr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] =>
      if launchDimsBad nBufs [gx, gy, gz, bx, by_, bz] then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else do
          let kernel ← readCStrAt w.mem kptr
          let entry ← readCStrAt w.mem namePtr
          cudaLaunchOn w defaultParty kernel entry nBufs bindPtr [gx, gy, gz, bx, by_, bz]
  | _ => none

/-- A launch on a created stream (or the default one, for `-1`). -/
def ffiCudaLaunchOnStream (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid] =>
      if launchDimsBad nBufs [gx, gy, gz, bx, by_, bz] then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else do
          let kernel ← readCStrAt w.mem kptr
          match w.dev.party? (asI32 sid) with
          | none => devFail w.dev
          | some p => cudaLaunchOn w p kernel "main" nBufs bindPtr [gx, gy, gz, bx, by_, bz]
  | _ => none

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
def ffiCudaUploadAsyncAt (w : World) (ctx buf off src size sid : UInt64) :
    Option (Option V × World) :=
  if asI32 buf < 0 || asI64 off < 0 || asI64 size ≤ 0 || src == 0 then cudaFail w
  else do
    let ok ← cudaCtxOk w ctx
    if !ok then cudaFail w
    else match w.dev.party? (asI32 sid) with
      | none => cudaFail w
      | some p =>
          match w.dev.get? (asI32 buf) with
          | none => cudaFail w
          | some b =>
              if off.toNat + size.toNat > b.size then cudaFail w
              else do
                let bytes ← readBytes (w.mem.forParty p) src size.toNat
                let id := (asI32 buf).toNat
                asyncCopy w p src size.toNat false [] [id] fun r =>
                  some ({ (w.dev.put id (some (overwrite b off.toNat bytes))) with race := r },
                        w.mem)

def ffiCudaUploadAsync (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, buf, src, size, sid] => ffiCudaUploadAsyncAt w ctx buf 0 src size sid
  | _ => none

def ffiCudaUploadOffsetAsync (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, buf, off, src, size, sid] => ffiCudaUploadAsyncAt w ctx buf off src size sid
  | _ => none

def ffiCudaDownloadAsyncAt (w : World) (ctx buf dst size sid : UInt64) :
    Option (Option V × World) :=
  if asI32 buf < 0 || asI64 size ≤ 0 || dst == 0 then cudaFail w
  else do
    let ok ← cudaCtxOk w ctx
    if !ok then cudaFail w
    else match w.dev.party? (asI32 sid) with
      | none => cudaFail w
      | some p =>
          match w.dev.get? (asI32 buf) with
          | none => cudaFail w
          | some b =>
              if size.toNat > b.size then cudaFail w
              else
                asyncCopy w p dst size.toNat true [(asI32 buf).toNat] [] fun r => do
                  let m ← copyIn (w.mem.forParty p) dst (b.extract 0 size.toNat)
                  -- to pageable memory the call returns once the copy is done
                  let r := if (pinnedSpan dst size.toNat).isSome then r else r.sync p
                  some ({ w.dev with race := r }, m)

def ffiCudaDownloadAsync (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, buf, dst, size, sid] => ffiCudaDownloadAsyncAt w ctx buf dst size sid
  | _ => none

-- streams, events, graphs

def ffiCudaStreamCreate (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else
        -- The runtime fences a new stream behind the default stream's work.
        let id := w.dev.streams.size
        let r := w.dev.race.joinInto (streamParty id)
          ((w.dev.race.clock defaultParty).join (w.dev.race.clock hostParty))
        some (some (ofInt .i32 id), { w.dev with streams := w.dev.streams.push true, race := r })
  | _ => none

def ffiCudaStreamSync (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, sid] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else match w.dev.party? (asI32 sid) with
        | none => devFail w.dev
        | some p =>
            -- Synchronising a capturing stream invalidates the capture.
            if w.dev.capturing p then none
            else devOk { w.dev with race := w.dev.race.sync p }
  | _ => none

def ffiCudaStreamDestroy (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, sid] =>
      if asI32 sid < 0 then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else if !w.dev.streams.getD (asI32 sid).toNat false then devFail w.dev
        else
          let p := streamParty (asI32 sid).toNat
          if w.dev.capturing p then none
          else
            -- The runtime fences the default stream behind the stream's work.
            devOk { w.dev with streams := w.dev.streams.set! (asI32 sid).toNat false,
                               race := w.dev.race.joinInto defaultParty (w.dev.race.clock p) }
  | _ => none

def ffiCudaEventCreate (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else some (some (ofInt .i32 w.dev.events.size), { w.dev with events := w.dev.events.push (some none) })
  | _ => none

/-- The event an `i32` id names, if it exists. -/
def Dev.event? (d : Dev) (id : Int) : Option (Option (Clock × Bool)) :=
  if id < 0 then none else (d.events[id.toNat]?).join

def ffiCudaEventRecord (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, eid, sid] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else match w.dev.event? (asI32 eid), w.dev.party? (asI32 sid) with
        | some _, some p =>
            let e := (asI32 eid).toNat
            match w.dev.capture with
            | some c =>
                if w.dev.capturing p then
                  let (r, clk) := c.race.issue p
                  devOk { w.dev with capture := some { c with race := r },
                                     events := w.dev.events.set! e (some (some (clk, true))) }
                else
                  let (r, clk) := w.dev.race.issue p
                  devOk { w.dev with race := r, events := w.dev.events.set! e (some (some (clk, false))) }
            | none =>
                let (r, clk) := w.dev.race.issue p
                devOk { w.dev with race := r, events := w.dev.events.set! e (some (some (clk, false))) }
        | _, _ => devFail w.dev
  | _ => none

def ffiCudaStreamWaitEvent (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, sid, eid] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else match w.dev.party? (asI32 sid), w.dev.event? (asI32 eid) with
        | some p, some ev =>
            match ev, w.dev.capture with
            -- never recorded: the wait is already satisfied
            | none, _ => devOk w.dev
            -- a point inside the capture: the waiting stream joins it
            | some (clk, true), some c =>
                let (r, _) := c.race.issue p
                let r := r.joinInto p clk
                let joined := if c.origin == p || c.joined.contains p then c.joined else p :: c.joined
                devOk { w.dev with capture := some { c with race := r, joined := joined } }
            -- a point outside any capture, waited on outside one
            | some (clk, false), _ =>
                if w.dev.capturing p then none
                else
                  let (r, _) := w.dev.race.issue p
                  devOk { w.dev with race := r.joinInto p clk }
            -- a capture's point with no capture in progress
            | some (_, true), none => none
        | _, _ => devFail w.dev
  | _ => none

def ffiCudaEventDestroy (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, eid] =>
      if asI32 eid < 0 then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else match w.dev.event? (asI32 eid) with
          | none => devFail w.dev
          | some _ => devOk { w.dev with events := w.dev.events.set! (asI32 eid).toNat none }
  | _ => none

def ffiCudaGraphBeginCapture (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, sid] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else match w.dev.party? (asI32 sid) with
        | none => devFail w.dev
        | some p =>
            -- The legacy default stream cannot be captured.
            if p == defaultParty then devFail w.dev
            else if w.dev.capture.isSome then none
            else devOk { w.dev with capture := some ⟨p, [], {}, #[]⟩ }
  | _ => none

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

def ffiCudaGraphEndCapture (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, sid] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else match w.dev.party? (asI32 sid), w.dev.capture with
        | some p, some c =>
            if c.origin != p then none
            else if !c.rejoined then devFail w.dev.endCapture
            else
              some (some (ofInt .i32 w.dev.graphs.size),
                    { w.dev.endCapture with graphs := w.dev.graphs.push (some c.ops) })
        | _, _ => devFail w.dev
  | _ => none

/-- The captured graph an `i32` id names. -/
def Dev.graph? (d : Dev) (id : Int) : Option (Array DevOp) :=
  if id < 0 then none else (d.graphs[id.toNat]?).join

def ffiCudaGraphUpload (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, gid, sid] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else match w.dev.graph? (asI32 gid), w.dev.party? (asI32 sid) with
        | some _, some _ => devOk w.dev
        | _, _ => devFail w.dev
  | _ => none

/-- What a graph touches, for ordering it against everything else: the vendor
    routines' inputs are read; every kernel binding and every output written. -/
def graphAccess (ops : Array DevOp) : List Nat × List Nat :=
  ops.foldl (fun (rs, ws) op => match op with
    | .launch _ ids => (rs, ws ++ ids)
    | .vendor _ ins out => (rs ++ ins, ws ++ [out])) ([], [])

def ffiCudaGraphLaunch (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, gid, sid] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else match w.dev.graph? (asI32 gid), w.dev.party? (asI32 sid) with
        | some ops, some p =>
            if w.dev.capturing p then none
            else do
              let (rs, ws) := graphAccess ops
              let r ← w.dev.race.op p rs ws
              -- Its nodes in the order they were recorded — one the graph's
              -- own edges allow, and the only result, since the capture
              -- checked they do not race.
              let d ← ops.foldlM (fun d op => match op with
                | .launch l ids => d.runLaunch w.kernel l ids
                | .vendor c ins out => d.runVendor w.vendor c ins out) { w.dev with race := r }
              devOk d
        | _, _ => devFail w.dev
  | _ => none

def ffiCudaGraphDestroy (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, gid] =>
      if asI32 gid < 0 then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else match w.dev.graph? (asI32 gid) with
          | none => devFail w.dev
          | some _ => devOk { w.dev with graphs := w.dev.graphs.set! (asI32 gid).toNat none }
  | _ => none

-- cublas — cuBLAS reads and writes wherever its dimensions say and the entry
-- points do not bound them by the buffers, so a call whose extents run past a
-- buffer, or that cuBLAS itself refuses, is not one a program can go on from:
-- stuck. What the routine computes is the vendor oracle's.

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

def ffiCublasSgemv (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, trans, m, n, alpha, a, x, beta, y] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else sgemvOn w defaultParty trans m n alpha a x beta y
  | _ => none

/-- The same routine on a created stream's own cuBLAS handle. -/
def ffiCublasSgemvOnStream (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, trans, m, n, alpha, a, x, beta, y, sid] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else match w.dev.party? (asI32 sid) with
        | none => devFail w.dev
        | some p => sgemvOn w p trans m n alpha a x beta y
  | _ => none

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

def sgemmSignsBad (m n k sa sb sc batch oa ob oc la lb lc : UInt64) : Bool :=
  asI32 m ≤ 0 || asI32 n ≤ 0 || asI32 k ≤ 0 || asI64 sa < 0 || asI64 sb < 0
    || asI64 sc < 0 || asI32 batch ≤ 0 || asI64 oa < 0 || asI64 ob < 0 || asI64 oc < 0
    || asI32 la < 0 || asI32 lb < 0 || asI32 lc < 0

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

def ffiCublasSgemm (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, ta, tb, m, n, k, alpha, a, sa, b, sb, beta, c, sc, batch, oa, ob, oc, la, lb, lc] =>
      if sgemmSignsBad m n k sa sb sc batch oa ob oc la lb lc then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else sgemmOn w defaultParty ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc
  | _ => none

/-- The same routine on a created stream's own cuBLAS handle. -/
def ffiCublasSgemmOnStream (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, ta, tb, m, n, k, alpha, a, sa, b, sb, beta, c, sc, batch, sid,
     oa, ob, oc, la, lb, lc] =>
      if sgemmSignsBad m n k sa sb sc batch oa ob oc la lb lc then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else match w.dev.party? (asI32 sid) with
          | none => devFail w.dev
          | some p => sgemmOn w p ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc
  | _ => none

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

def ffiCublasGemmExBf16 (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, ta, tb, m, n, k, alpha, a, b, beta, c, oa, ob, oc, la, lb, lc] =>
      if asI32 m ≤ 0 || asI32 n ≤ 0 || asI32 k ≤ 0 || asI64 oa < 0 || asI64 ob < 0
          || asI64 oc < 0 || asI32 la < 0 || asI32 lb < 0 || asI32 lc < 0 then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else match w.dev.get? (asI32 a), w.dev.get? (asI32 b), w.dev.get? (asI32 c) with
          | some A, some B, some C =>
              if !gemmBf16Fits ta tb m n k oa ob oc la lb lc A B C then none
              else do
                let (ia, ib, ic) := ((asI32 a).toNat, (asI32 b).toNat, (asI32 c).toNat)
                let d ← w.dev.devOp w defaultParty
                  (.vendor ⟨"gemmExBf16", [ta, tb, m, n, k, alpha, beta, oa, ob, oc, la, lb, lc]⟩
                    [ia, ib, ic] ic) [ia, ib, ic] [ic]
                devOk d
          | _, _, _ => devFail w.dev
  | _ => none

/-- The strided-batched `gemmEx`, `bf16` in and `f32` out, on the default
    stream. -/
def ffiCublasGemmStridedBatchedExBf16 (bits : List UInt64) (w : World) :
    Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, ta, tb, m, n, k, alpha, a, sa, b, sb, beta, c, sc, batch, oa, ob, oc, la, lb, lc] =>
      if sgemmSignsBad m n k sa sb sc batch oa ob oc la lb lc then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else match w.dev.get? (asI32 a), w.dev.get? (asI32 b), w.dev.get? (asI32 c) with
          | some A, some B, some C =>
              if !gemmStridedFits 2 2 4 ta tb m n k sa sb sc batch oa ob oc la lb lc A B C then none
              else do
                let (ia, ib, ic) := ((asI32 a).toNat, (asI32 b).toNat, (asI32 c).toNat)
                let d ← w.dev.devOp w defaultParty
                  (.vendor ⟨"gemmStridedBatchedExBf16",
                      [ta, tb, m, n, k, alpha, beta, sa, sb, sc, batch, oa, ob, oc, la, lb, lc]⟩
                    [ia, ib, ic] ic) [ia, ib, ic] [ic]
                devOk d
          | _, _, _ => devFail w.dev
  | _ => none

/-- Entry `slot` of the pointer array in buffer `arr` set to where buffer `src`
    lives, `off` floats in: a synchronous copy on the default stream. -/
def ffiCublasPtrArray (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, arr, slot, src, off] =>
      if asI32 arr < 0 || asI32 slot < 0 || asI32 src < 0 || asI64 off < 0 then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else match w.dev.get? (asI32 src), w.dev.get? (asI32 arr) with
          | some _, some A =>
              let s := (asI32 slot).toNat
              if A.size < 8 * (s + 1) then devFail w.dev
              else do
                let v := le64 (w.devAddr (asI32 src).toNat + 4 * UInt64.ofNat (asI64 off).toNat)
                let d ← w.dev.syncWrite (asI32 arr).toNat (overwrite A (8 * s) v)
                devOk d
          | _, _ => devFail w.dev
  | _ => none

/-- The live buffer a device pointer lands in, and how far into it. -/
def World.devBufAt? (w : World) (ptr : UInt64) : Option (Nat × Nat) :=
  (List.range w.dev.bufs.size).findSome? fun id =>
    match (w.dev.bufs[id]?).join with
    | some b =>
        if w.devAddr id ≤ ptr && (ptr - w.devAddr id).toNat < b.size then
          some (id, (ptr - w.devAddr id).toNat)
        else none
    | none => none

/-- One member of a pointer-array batch: its A, B and C as (buffer, byte
    offset), read from entry `i` of the three arrays. -/
def batchMember (w : World) (PA PB PC : ByteArray) (i : Nat) :
    Option ((Nat × Nat) × (Nat × Nat) × (Nat × Nat)) := do
  let at_ (P : ByteArray) : Option (Nat × Nat) :=
    if P.size < 8 * (i + 1) then none else w.devBufAt? (ofLe64 (P.extract (8 * i) (8 * i + 8)))
  let a ← at_ PA
  let b ← at_ PB
  let c ← at_ PC
  some (a, b, c)

-- Every member's operands must resolve to live buffers and lie inside them,
-- or the call is stuck: cuBLAS would read whatever is there. Members run as
-- one call, in no order, so a member that writes a buffer another member reads
-- or writes is stuck too; with that excluded, running them one after another
-- is what the call does.
def ffiCublasSgemmBatchedOnStream (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, ta, tb, m, n, k, alpha, aArr, bArr, beta, cArr, batch, sid] =>
      if asI32 m ≤ 0 || asI32 n ≤ 0 || asI32 k ≤ 0 || asI32 batch ≤ 0
          || asI32 aArr < 0 || asI32 bArr < 0 || asI32 cArr < 0 then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else match w.dev.party? (asI32 sid) with
        | none => devFail w.dev
        | some p =>
          match w.dev.get? (asI32 aArr), w.dev.get? (asI32 bArr), w.dev.get? (asI32 cArr) with
          | some PA, some PB, some PC => do
              let (M, N, K) := ((asI32 m).toNat, (asI32 n).toNat, (asI32 k).toNat)
              let (ra, ca) := if asI32 ta != 0 then (K, M) else (M, K)
              let (rb, cb) := if asI32 tb != 0 then (N, K) else (K, N)
              let ms ← (List.range (asI32 batch).toNat).mapM (batchMember w PA PB PC)
              let inside (x : Nat × Nat) (span : Nat) : Bool :=
                match w.dev.get? x.1 with
                | some buf => decide (x.2 % 4 = 0) && decide (x.2 + 4 * span ≤ buf.size)
                | none => false
              if !ms.all (fun (a, b, c) => inside a (colMajorSpan ra ca ra)
                  && inside b (colMajorSpan rb cb rb) && inside c (colMajorSpan M N M)) then none
              else
                let outs := ms.map (·.2.2.1)
                let ins := ms.flatMap (fun (a, b, _) => [a.1, b.1])
                if !outs.Nodup || outs.any (ins.contains ·) then none
                else
                  let arrs := [(asI32 aArr).toNat, (asI32 bArr).toNat, (asI32 cArr).toNat]
                  let d ← ms.foldlM (fun (d : Dev) (a, b, c) =>
                    d.devOp w p (.vendor ⟨"sgemmBatchedMember",
                        [ta, tb, m, n, k, alpha, beta, UInt64.ofNat a.2, UInt64.ofNat b.2,
                         UInt64.ofNat c.2]⟩ [a.1, b.1, c.1] c.1)
                      (arrs ++ [a.1, b.1, c.1]) [c.1]) w.dev
                  devOk d
          | _, _, _ => devFail w.dev
  | _ => none

/-- A launch by entry-point name on a created stream (or the default one). -/
def ffiCudaLaunchNamedOnStream (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, kptr, namePtr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid] =>
      if launchDimsBad nBufs [gx, gy, gz, bx, by_, bz] then devFail w.dev
      else do
        let ok ← cudaCtxOk w ctx
        if !ok then devFail w.dev
        else do
          let kernel ← readCStrAt w.mem kptr
          let entry ← readCStrAt w.mem namePtr
          match w.dev.party? (asI32 sid) with
          | none => devFail w.dev
          | some p => cudaLaunchOn w p kernel entry nBufs bindPtr [gx, gy, gz, bx, by_, bz]
  | _ => none

/-- Whether the host has waited for everything up to clock `c`. -/
def Race.hostSaw (r : Race) (c : Clock) : Bool :=
  (List.range c.size).all fun q => c.get q ≤ (r.clock hostParty).get q

-- Elapsed time between two recorded events, once the host has waited for
-- both. Before that the driver may answer or say "not ready", which the model
-- does not guess at; an event recorded into a capture is a graph node, not a
-- time. A never-recorded event is refused.
def ffiCudaEventElapsedMsBits (bits : List UInt64) (w : World) : Option (Option V × World) :=
  devOnly w <| match bits with
  | [ctx, s, e] => do
      let ok ← cudaCtxOk w ctx
      if !ok then devFail w.dev
      else match w.dev.event? (asI32 s), w.dev.event? (asI32 e) with
        | some (some (cs, false)), some (some (ce, false)) =>
            if w.dev.race.hostSaw cs && w.dev.race.hostSaw ce then
              some (some (ofInt .i32 (w.elapsed (asI32 s).toNat (asI32 e).toNat).toNat), w.dev)
            else none
        | some none, some _ | some _, some none => devFail w.dev
        | some _, some _ => none
        | _, _ => devFail w.dev
  | _ => none

-- wgpu — the one queue. Uploads land at once; dispatches wait in `pending`
-- and run, in order, at the next dispatch or download. A length that is not a
-- multiple of four, runs past a buffer, or (for a download) is not the whole
-- buffer answers `-1`: `Lib.Wgpu` checks it against the size it recorded
-- before wgpu sees it. That a shader compiles is taken as given.

/-- Whether `ctx` is the live wgpu context (`some true`), null (`some false`:
    `-1`), or anything else (stuck). -/
def gpuCtxOk (w : World) (ctx : UInt64) : Option Bool :=
  if ctx == 0 then some false
  else if ctx == gpuCtx && w.gpu.live then some true
  else none

/-- Run one recorded dispatch: every binding gets the shader's contents for it,
    or keeps its own if it is read-only. A buffer's length is fixed when it is
    made, so contents of another length are not what the device computes: the
    buffer keeps its own. A dispatch names a pipeline that exists, and a
    pipeline buffers that exist, so the rest is never reached; there the queue
    is left as it was. -/
def Wgpu.runOne (sh : Dispatch → List ByteArray → List ByteArray) (g : Wgpu)
    (d : Nat × List UInt64) : Option Wgpu :=
  match g.pipes[d.1]? with
  | none => some g
  | some p =>
      match p.binds.mapM (fun b => g.bufs[b.1]?) with
      | none => some g
      | some ins =>
          let outs := sh ⟨p.shader, p.binds, d.2⟩ ins
          let bufs := (p.binds.zip (ins.zipIdx)).foldl
            (fun bs (b, i, k) =>
              let o := (outs[k]?).getD i
              bs.set! b.1 (if b.2 || o.size != i.size then i else o)) g.bufs
          some { g with bufs := bufs }

/-- Submit what is pending. -/
def Wgpu.flush (sh : Dispatch → List ByteArray → List ByteArray) (g : Wgpu) : Option Wgpu := do
  let g' ← g.pending.foldlM (Wgpu.runOne sh) g
  some { g' with pending := [] }

/-- A context, where wgpu finds an adapter; null, as `Lib.Wgpu` stores,
    without one. -/
def ffiGpuInit (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] =>
      if w.gpuAdapter then do
        let m ← w.mem.store slot 8 gpuCtx
        some (none, { w with mem := m, gpu := { live := true } })
      else do
        let m ← w.mem.store slot 8 0
        some (none, { w with mem := m, gpu := {} })
  | _ => none

def ffiGpuCleanup (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] => do
      let m ← w.mem.store slot 8 0
      some (none, { w with mem := m, gpu := {} })
  | _ => none

def ffiGpuCreateBuffer (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, size] =>
      if asI64 size ≤ 0 then cudaFail w
      else do
        let ok ← gpuCtxOk w ctx
        if !ok then cudaFail w
        else
          some (some (ofInt .i32 w.gpu.bufs.size),
                { w with gpu := { w.gpu with
                    bufs := w.gpu.bufs.push (ByteArray.mk (Array.replicate size.toNat 0)) } })
  | _ => none

/-- The bindings a pipeline asks for: `n` descriptors of an `i32` buffer id and
    an `i32` read-only flag, or `none` for a descriptor out of memory. -/
def readBinds (m : Mem) (at_ : UInt64) (n : Nat) : Option (List (Int × Bool)) :=
  (List.range n).mapM fun i => do
    let id ← m.load (at_ + UInt64.ofNat (8 * i)) 4
    let ro ← m.load (at_ + UInt64.ofNat (8 * i + 4)) 4
    some (asI32 id, ro != 0)

def ffiGpuCreatePipeline (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, shaderPtr, bindPtr, n] =>
      if asI32 n < 0 then cudaFail w
      else do
        let ok ← gpuCtxOk w ctx
        if !ok then cudaFail w
        else do
          let src ← readCStrAt w.mem shaderPtr
          let bs ← readBinds w.mem bindPtr (asI32 n).toNat
          -- The Rust reads the id as `usize`, so a negative one is out of range too.
          if bs.any (fun (id, _) => id < 0 || id.toNat ≥ w.gpu.bufs.size) then cudaFail w
          else
            some (some (ofInt .i32 w.gpu.pipes.size),
                  { w with gpu := { w.gpu with
                      pipes := w.gpu.pipes.push ⟨src, bs.map (fun (id, ro) => (id.toNat, ro))⟩ } })
  | _ => none

/-- `cl_gpu_upload` and `cl_gpu_upload_ptr`, which are one function. -/
def gpuUploadAt (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, buf, src, size] =>
      if asI32 buf < 0 || asI64 size ≤ 0 || src == 0 then cudaFail w
      else do
        let ok ← gpuCtxOk w ctx
        if !ok then cudaFail w
        else match w.gpu.bufs[(asI32 buf).toNat]? with
          | none => cudaFail w
          | some b =>
              if size.toNat > b.size || size.toNat % 4 != 0 then cudaFail w
              else do
                let bytes ← readBytes w.mem src size.toNat
                some (some (ofInt .i32 0),
                      { w with gpu := { w.gpu with
                          bufs := w.gpu.bufs.set! (asI32 buf).toNat (overwrite b 0 bytes) } })
  | _ => none

def ffiGpuUpload (bits : List UInt64) (w : World) : Option (Option V × World) := gpuUploadAt bits w
def ffiGpuUploadPtr (bits : List UInt64) (w : World) : Option (Option V × World) := gpuUploadAt bits w

def ffiGpuDispatch (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, pipe, x, y, z] =>
      if asI32 pipe < 0 || asI32 x ≤ 0 || asI32 y ≤ 0 || asI32 z ≤ 0 then cudaFail w
      else do
        let ok ← gpuCtxOk w ctx
        if !ok then cudaFail w
        else if (asI32 pipe).toNat ≥ w.gpu.pipes.size then cudaFail w
        else do
          let g ← w.gpu.flush w.shader
          some (some (ofInt .i32 0),
                { w with gpu := { g with pending := [((asI32 pipe).toNat, [x, y, z])] } })
  | _ => none

def ffiGpuDownload (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, buf, dst, size] =>
      if asI32 buf < 0 || asI64 size ≤ 0 || dst == 0 then cudaFail w
      else do
        let ok ← gpuCtxOk w ctx
        if !ok then cudaFail w
        else if (asI32 buf).toNat ≥ w.gpu.bufs.size then cudaFail w
        else do
          let g ← w.gpu.flush w.shader
          match g.bufs[(asI32 buf).toNat]? with
          | none => cudaFail w
          | some b =>
              if size.toNat != b.size || size.toNat % 4 != 0 then cudaFail w
              else do
                let m ← copyIn w.mem dst b
                some (some (ofInt .i32 0), { w with mem := m, gpu := g })
  | _ => none

def ffiGpuDownloadPtr (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, buf, off, dst, size] =>
      if asI32 buf < 0 || asI64 size ≤ 0 || dst == 0 || asI64 off < 0 then cudaFail w
      else do
        let ok ← gpuCtxOk w ctx
        if !ok then cudaFail w
        else if (asI32 buf).toNat ≥ w.gpu.bufs.size then cudaFail w
        else do
          let g ← w.gpu.flush w.shader
          match g.bufs[(asI32 buf).toNat]? with
          | none => cudaFail w
          | some b =>
              if off.toNat + size.toNat > b.size || off.toNat % 4 != 0 || size.toNat % 4 != 0 then cudaFail w
              else do
                let m ← copyIn w.mem dst (b.extract off.toNat (off.toNat + size.toNat))
                some (some (ofInt .i32 0), { w with mem := m, gpu := g })
  | _ => none


-- The window — one per context, and events from an oracle. Every call that
-- pumps the event loop moves the next batch of `winInput` into the pending
-- events; a present records the buffer it showed, and fails while the blit
-- shader does not compile. The model takes a present as succeeding otherwise:
-- a surface that times out drops the frame and still returns 0.

def winCtxOk (w : World) (ctx : UInt64) : Option Bool :=
  if ctx == 0 then some false
  else if ctx == winCtx && w.win.live then some true
  else none

/-- The events that have arrived by the next pump. -/
def World.pump (w : World) : World :=
  match w.winInput with
  | [] => w
  | batch :: rest => { w with win := { w.win with pending := w.win.pending ++ batch }, winInput := rest }

/-- Events as the runtime writes them: four little-endian `i64`s each. -/
def winEventBytes (es : List WinEvent) : ByteArray :=
  es.foldl (fun acc e => acc ++ le64 e.kind ++ le64 e.a ++ le64 e.b ++ le64 e.c) ByteArray.empty

-- Without a display the window library's event loop does not start, and `init`
-- stores null. A second `init` while one is live is stuck: its context would leak.
def ffiWindowInit (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] =>
      if w.win.live then none
      else if w.display then do
        let m ← w.mem.store slot 8 winCtx
        some (none, { w with mem := m, win := { live := true } })
      else do
        let m ← w.mem.store slot 8 0
        some (none, { w with mem := m })
  | _ => none

def ffiWindowCleanup (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] => do
      let c ← w.mem.load slot 8
      if c == 0 then some (none, w)
      else if c == winCtx && w.win.live then do
        let m ← w.mem.store slot 8 0
        some (none, { w with mem := m, win := {} })
      else none
  | _ => none

-- The title is bytes, handed on as a C string. A blit shader that is not UTF-8,
-- which wgpu requires of WGSL, answers `-1`; whether it compiles is decided at
-- the first present.
def ffiWindowOpen (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, width, height, titlePtr, titleLen, blitPtr, blitLen] =>
      if asI64 width ≤ 0 || asI64 height ≤ 0 || asI64 titleLen < 0 || titlePtr == 0
          || asI64 blitLen ≤ 0 || blitPtr == 0 then cudaFail w
      else match winCtxOk w ctx with
      | none => none
      | some false => cudaFail w
      | some true =>
          if w.win.isOpen then cudaFail w
          else do
            let _ ← readBytes w.mem titlePtr (asI64 titleLen).toNat
            let blit ← readBytes w.mem blitPtr (asI64 blitLen).toNat
            match String.fromUTF8? blit with
            | none => cudaFail w
            | some src => some (some (ofInt .i32 0), { w with win := { w.win with isOpen := true, blit := src } })
  | _ => none

def ffiWindowPoll (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, eventsPtr, maxEvents] =>
      if asI32 maxEvents < 0 || eventsPtr == 0 then cudaFail w
      else match winCtxOk w ctx with
      | none => none
      | some false => cudaFail w
      | some true => do
          let w := w.pump
          let n := min w.win.pending.length (asI32 maxEvents).toNat
          let m ← copyIn w.mem eventsPtr (winEventBytes (w.win.pending.take n))
          some (some (ofInt .i32 n),
                { w with mem := m, win := { w.win with pending := w.win.pending.drop n } })
  | _ => none

def ffiWindowPresentGpuBuffer (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, gctx, buf] =>
      if asI32 buf < 0 then cudaFail w
      else match winCtxOk w ctx with
      | none => none
      | some false => cudaFail w
      | some true => do
          let w := w.pump
          let ok ← gpuCtxOk w gctx
          if !ok then cudaFail w
          else do
            let g ← w.gpu.flush w.shader
            let w := { w with gpu := g }
            match g.bufs[(asI32 buf).toNat]? with
            | none => cudaFail w
            | some b =>
                if !w.win.isOpen || !w.wgslOk w.win.blit then cudaFail w
                else some (some (ofInt .i32 0), { w with shown := w.shown ++ [b] })
  | _ => none


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

def ffiNativeFree (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [addr] =>
      match w.nativeAt? addr with
      | none => some (some (ofInt .i32 (-1)), w)
      | some k => some (some (ofInt .i32 0), { w with native := w.native.set! k none })
  | _ => none

def ffiNativeArch (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [] => some (some (ofInt .i32 w.arch), w)
  | _ => none

-- A null or unterminated name reads as empty, which no feature is called.
def readName (m : Mem) (a : UInt64) : Option String :=
  if a == 0 then some ""
  else (readCStr m a).map fun s => if 4096 ≤ s.utf8ByteSize then "" else s

def featureCode : Option Bool → Int
  | some true => 1
  | some false => 0
  | none => -1

def ffiCpuHas (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [name] => do
      let s ← readName w.mem name
      some (some (ofInt .i32 (featureCode (w.cpuHas s))), w)
  | _ => none


-- thread.rs — the context and joining. Spawning runs one of the program's own
-- functions, so its meaning is in `Sem.callOf`, beside local calls. The model
-- lets one worker be outstanding and freezes host memory until it is joined:
-- the worker has already run in the model, and with the host touching nothing
-- meanwhile, running it first is what any interleaving computes.

def threadCtxOk (w : World) (ctx : UInt64) : Option Bool :=
  if ctx == 0 then some false
  else if ctx == threadCtx && w.thread.live then some true
  else none

def ffiThreadInit (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] =>
      if w.thread.live then none
      else do
        let m ← w.mem.store slot 8 threadCtx
        some (none, { w with mem := m, thread := { w.thread with live := true } })
  | _ => none

-- Cleanup joins what is outstanding, so memory is the host's again.
def ffiThreadCleanup (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [slot] => do
      let m := { w.mem with frozen := false }
      let c ← m.load slot 8
      if c == 0 then some (none, { w with mem := m })
      else if c == threadCtx && w.thread.live then do
        let m ← m.store slot 8 0
        some (none, { w with mem := m, thread := { w.thread with live := false, outstanding := none } })
      else none
  | _ => none

def ffiThreadJoin (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match bits with
  | [ctx, h] =>
      match threadCtxOk w ctx with
      | none => none
      | some false => some (some (ofInt .i64 (-1)), w)
      | some true =>
          if w.thread.outstanding == some h.toUInt32.toNat then
            some (some (ofInt .i64 0),
                  { w with mem := { w.mem with frozen := false },
                           thread := { w.thread with outstanding := none } })
          else some (some (ofInt .i64 (-1)), w)
  | _ => none

/-- **What each entry point does, as a function of the world.**

    One definition per entry point with a contract, and this dispatch. The
    families are the ones whose behavior the model can state: files, standard
    streams, the hash table and the libm shims, whose results the bytes
    determine; and CUDA with cuBLAS on the default stream and wgpu on its one
    queue, whose data
    movement the bytes determine and whose arithmetic comes from the world's
    kernel and vendor oracles. Each is a transcription of `base/src/ffi/`, and
    like `evalOp` a transcription is an assumption — what makes it more than
    that is the corpora, which run these against the real symbols and compare.

    Everything else — CUDA streams, events and graphs, windows, databases — has
    a `Frame` and no definition, and calling one goes `stuck` rather than
    guessing. -/
def callBits (f : IR.Ffi) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match f with
  | .fileRead => ffiFileRead bits w
  | .fileWrite => ffiFileWrite bits w
  | .fileReadToPtr => ffiFileReadToPtr bits w
  | .fileWriteFromPtr => ffiFileWriteFromPtr bits w
  | .stdinReadline => ffiStdinReadline bits w
  | .stdoutWrite => ffiStdoutWrite bits w
  | .sinf => ffiSinf bits w
  | .cosf => ffiCosf bits w
  | .powf => ffiPowf bits w
  | .htInit => ffiHtInit bits w
  | .htCleanup => ffiHtCleanup bits w
  | .htCreate => ffiHtCreate bits w
  | .htCount => ffiHtCount bits w
  | .htLookup => ffiHtLookup bits w
  | .htInsert => ffiHtInsert bits w
  | .htIncrement => ffiHtIncrement bits w
  | .htGetEntry => ffiHtGetEntry bits w
  | .cudaInit => ffiCudaInit bits w
  | .cudaCleanup => ffiCudaCleanup bits w
  | .cudaCreateBuffer => ffiCudaCreateBuffer bits w
  | .cudaUpload => ffiCudaUpload bits w
  | .cudaUploadOffset => ffiCudaUploadOffset bits w
  | .cudaDownload => ffiCudaDownload bits w
  | .cudaDownloadOffset => ffiCudaDownloadOffset bits w
  | .cudaFreeBuffer => ffiCudaFreeBuffer bits w
  | .cudaSync => ffiCudaSync bits w
  | .cudaLaunch => ffiCudaLaunch bits w
  | .cublasSgemv => ffiCublasSgemv bits w
  | .cublasSgemm => ffiCublasSgemm bits w
  | .cublasGemmExBf16 => ffiCublasGemmExBf16 bits w
  | .cublasSgemvOnStream => ffiCublasSgemvOnStream bits w
  | .cublasGemmStridedBatchedExBf16 => ffiCublasGemmStridedBatchedExBf16 bits w
  | .cublasPtrArray => ffiCublasPtrArray bits w
  | .cublasSgemmBatchedOnStream => ffiCublasSgemmBatchedOnStream bits w
  | .cudaLaunchNamedOnStream => ffiCudaLaunchNamedOnStream bits w
  | .cudaEventElapsedMsBits => ffiCudaEventElapsedMsBits bits w
  | .cudaUploadAsync => ffiCudaUploadAsync bits w
  | .cudaUploadOffsetAsync => ffiCudaUploadOffsetAsync bits w
  | .cudaDownloadAsync => ffiCudaDownloadAsync bits w
  | .cublasSgemmOnStream => ffiCublasSgemmOnStream bits w
  | .cudaLaunchOnStream => ffiCudaLaunchOnStream bits w
  | .cudaStreamCreate => ffiCudaStreamCreate bits w
  | .cudaStreamSync => ffiCudaStreamSync bits w
  | .cudaStreamDestroy => ffiCudaStreamDestroy bits w
  | .cudaEventCreate => ffiCudaEventCreate bits w
  | .cudaEventRecord => ffiCudaEventRecord bits w
  | .cudaStreamWaitEvent => ffiCudaStreamWaitEvent bits w
  | .cudaEventDestroy => ffiCudaEventDestroy bits w
  | .cudaGraphBeginCapture => ffiCudaGraphBeginCapture bits w
  | .cudaGraphEndCapture => ffiCudaGraphEndCapture bits w
  | .cudaGraphUpload => ffiCudaGraphUpload bits w
  | .cudaGraphLaunch => ffiCudaGraphLaunch bits w
  | .cudaGraphDestroy => ffiCudaGraphDestroy bits w
  | .cudaLaunchNamed => ffiCudaLaunchNamed bits w
  | .cudaPinnedAlloc => ffiCudaPinnedAlloc bits w
  | .cudaPinnedPtr => ffiCudaPinnedPtr bits w
  | .cudaPinnedPtrAt => ffiCudaPinnedPtrAt bits w
  | .cudaPinnedFree => ffiCudaPinnedFree bits w
  | .cudaMemInfoFree => ffiCudaMemInfoFree bits w
  | .cudaMemInfoTotal => ffiCudaMemInfoTotal bits w
  | .gpuInit => ffiGpuInit bits w
  | .gpuCleanup => ffiGpuCleanup bits w
  | .gpuCreateBuffer => ffiGpuCreateBuffer bits w
  | .gpuCreatePipeline => ffiGpuCreatePipeline bits w
  | .gpuUpload => ffiGpuUpload bits w
  | .gpuUploadPtr => ffiGpuUploadPtr bits w
  | .gpuDispatch => ffiGpuDispatch bits w
  | .gpuDownload => ffiGpuDownload bits w
  | .gpuDownloadPtr => ffiGpuDownloadPtr bits w
  | .lmdbInit => ffiLmdbInit bits w
  | .lmdbCleanup => ffiLmdbCleanup bits w
  | .lmdbOpen => ffiLmdbOpen bits w
  | .lmdbBeginWriteTxn => ffiLmdbBeginWriteTxn bits w
  | .lmdbPut => ffiLmdbPut bits w
  | .lmdbCommitWriteTxn => ffiLmdbCommitWriteTxn bits w
  | .lmdbCursorScan => ffiLmdbCursorScan bits w
  | .windowInit => ffiWindowInit bits w
  | .windowCleanup => ffiWindowCleanup bits w
  | .windowOpen => ffiWindowOpen bits w
  | .windowPoll => ffiWindowPoll bits w
  | .windowPresentGpuBuffer => ffiWindowPresentGpuBuffer bits w
  | .nativeLoad => ffiNativeLoad bits w
  | .threadInit => ffiThreadInit bits w
  | .threadCleanup => ffiThreadCleanup bits w
  | .threadJoin => ffiThreadJoin bits w
  | .nativeFree => ffiNativeFree bits w
  | .nativeArch => ffiNativeArch bits w
  | .cpuHas => ffiCpuHas bits w
  | .fileCreateDirAll => ffiFileCreateDirAll bits w
  | .memLock => ffiGrant "lock" 2 bits w
  | .memUnlock => ffiGrant "unlock" 2 bits w
  | .memAdviseHuge => ffiGrant "hugepages" 2 bits w
  | .threadPriority => ffiGrant "priority" 1 bits w
  | _ => none

/-- A call with its arguments as values: every argument a scalar, read as bits,
    then `callBits`. -/
def callFfi (f : IR.Ffi) (args : List V) (w : World) : Option (Option V × World) :=
  (args.mapM asBits).bind fun bits => callBits f bits w

/-- The entry point a symbol names, and what it does. A name that is not an
    entry point — a program's own colocated function — has no contract. -/
def callImport (name : String) (args : List V) (w : World) : Option (Option V × World) := do
  let f ← IR.Ffi.ofCname name
  callFfi f args w

end AlgorithmLib.HProg.Sem
