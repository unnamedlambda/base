module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace HashTable

/-!
  # base's hash table: a SwissTable in CLIF

  `benchmarks/hashtable` measures it beside hashbrown and abseil's and
  boost's flat maps: `u64` keys to `u64` values, open addressing over
  power-of-two buckets, each bucket's control byte its key's top 7 hash bits
  (`0x80` empty), probed 16 control bytes at a time by one vector compare
  (`vhighBits` of the bytes equal to the key's tag), the same group width
  hashbrown probes at on x86. The first 16 control bytes are mirrored past
  the end, so a group read at any bucket never wraps. A table full to 7/8
  doubles, every key placed again in fresh memory.

  Everything lives in the arena, which is mapped lazily: the state words,
  then the tables, each made past the one before (a bump pointer that only
  `clear` takes back), so the arena holds every size up to the largest: room
  for `2^25` buckets, 29 million keys. An insert that would need a table past
  that is not made.

  Its functions (the caller's data, then what each does):

  * `clear`: the table emptied to 16 buckets (the first, offering the
    tables' memory huge pages);
  * `insert`: a count (`u64`), the keys, the values (`u64` each): each key
    set to its value;
  * `get`: a count, the keys: the output's first word gets the sum of the
    values found, its second how many were found.
-/

/-- Arena words: the table's control bytes and slots (offsets in the arena),
    its bucket mask, items, room before it must grow, and the bump pointer. -/
def CTRL : Nat := 0x100
def SLOTS : Nat := 0x108
def MASK : Nat := 0x110
def ITEMS : Nat := 0x118
def ROOM : Nat := 0x120
def BUMP : Nat := 0x128
/-- Whether the tables' memory has been offered huge pages yet. -/
def ADVISED : Nat := 0x130
/-- Where tables start, and the arena's size: room for every table from 16
    buckets to `2^25` (17 bytes a bucket, 16 more each). -/
def HEAP : Nat := 0x1000
def MAX_BUCKETS : Nat := 1 <<< 25
def MEM : Nat := HEAP + 2 * MAX_BUCKETS * 17 + (1 <<< 20)

def EMPTY : Nat := 0x80

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def word (i : Nat) : Prog V L (V .i64) := do iaddImm (← basePtr) i
def getW (i : Nat) : Prog V L (V .i64) := do load64 (← word i)
def setW (i : Nat) (v : V .i64) : Prog V L Unit := do storeI64 v (← word i)

/-- Fibonacci hashing: `k` times an odd constant (2^64 over the golden
    ratio), its top bits the ones a table reads (`homeOf`, `tagOf`), each of
    them depending on every bit of the key. -/
def hash (k : V .i64) : Prog V L (V .i64) := do
  imul k (← iconst64 (0x9E3779B97F4A7C15 - 2 ^ 64))

/-- A table of `buckets` (a power of two, at least 16) made at the bump
    pointer: control bytes all empty, slots after them. -/
def makeTable (buckets : V .i64) : Prog V L Unit := do
  let base ← basePtr
  let at_ ← getW BUMP
  let ctrlBytes ← iaddImm buckets 16
  let slots ← iadd at_ (← band (← iaddImm ctrlBytes 15) (← iconst64 (-16)))
  let empty ← splat .i8x16 (← ireduce .i8 (← iconst64 EMPTY))
  forLoop (← ushrImm ctrlBytes 4) fun i => do
    storeUnaligned empty (← iadd base (← iadd at_ (← ishlImm i 4)))
  setW CTRL at_
  setW SLOTS slots
  setW MASK (← iaddImm buckets (-1))
  setW ITEMS (← iconst64 0)
  setW ROOM (← isub buckets (← ushrImm buckets 3))
  setW BUMP (← iadd slots (← ishlImm buckets 4))

/-- The table as a probe reads it: its control bytes and slots (addresses)
    and its bucket mask, loaded once for many probes. -/
structure Tab (V : ClifTy → Type) where
  ctrl : V .i64
  slots : V .i64
  mask : V .i64

def tab : Prog V L (Tab V) := do
  let base ← basePtr
  pure ⟨← iadd base (← getW CTRL), ← iadd base (← getW SLOTS), ← getW MASK⟩

/-- The home bucket of hash `h` in table `t`: its top bits, as many as the
    table's buckets take. -/
def homeOf (t : Tab V) (h : V .i64) : Prog V L (V .i64) := do
  ushr h (← clz t.mask)

/-- The 7-bit tag of hash `h` in table `t`: the bits just below its home's. -/
def tagOf (t : Tab V) (h : V .i64) : Prog V L (V .i64) := do
  band (← ushr h (← iaddImm (← clz t.mask) (-7))) (← iconst64 0x7F)

/-- The bucket of `k` in table `t`: where it is, or the first empty one its
    probe meets, and whether it was found (`1`) or not (`0`). A group's tag
    matches are checked in turn, leaving the probe at the first key equal to
    `k`; a group with an empty byte ends it; otherwise the next group, a
    triangular step on. -/
def probeH (t : Tab V) (h k : V .i64) : Prog V L (V .i64 × V .i64) := do
  let tag ← splat .i8x16 (← ireduce .i8 (← tagOf t h))
  let zero ← iconst64 0
  let one ← iconst64 1
  -- a loop left only by `brk`: its head always goes on
  let r ← wloopL %[← homeOf t h, zero]
    (head := fun _ _ => pure (contIf .eq zero zero, %[zero, zero], ()))
    (body := fun out cs _ => do
      let pos := cs.head
      let g ← load { ty := .i8x16 } (← iadd t.ctrl pos)
      let hits ← uextend64 (← vhighBits (← icmp .eq g tag))
      let _ ← wloopL %[hits]
        (head := fun _ ms => pure (exitIfEq ms.head zero, %[], ()))
        (body := fun _ ms _ => do
          let m := ms.head
          let b ← band (← iadd pos (← ctz m)) t.mask
          when .eq (← load64 (← iadd t.slots (← ishlImm b 4))) k do brk out %[b, one]
          pure %[← band m (← iaddImm m (-1))])
      let empties ← uextend64 (← vhighBits g)
      when .ne empties zero do brk out %[← band (← iadd pos (← ctz empties)) t.mask, zero]
      let st ← iaddImm cs.snd 16
      pure %[← band (← iadd pos st) t.mask, st])
  pure (r.head, r.snd)

/-- `probeH` from the key's own hash. -/
def probe (t : Tab V) (k : V .i64) : Prog V L (V .i64 × V .i64) := do probeH t (← hash k) k

/-- `k ↦ v` placed in bucket `b` of table `t` (empty, from `probe`), `h` the
    key's hash: its tag, mirrored past the end where `b < 16`. The caller
    counts it. -/
def placeAt (t : Tab V) (h b k v : V .i64) : Prog V L Unit := do
  let tag ← tagOf t h
  istore8 tag (← iadd t.ctrl b)
  when .ult b (← iconst64 16) do istore8 tag (← iadd (← iadd t.ctrl (← iaddImm t.mask 1)) b)
  let slot ← iadd t.slots (← ishlImm b 4)
  storeI64 k slot
  storeI64 v (← iaddImm slot 8)

/-- The first empty bucket of the probe from hash `h` in table `t`. -/
def firstEmpty (t : Tab V) (h : V .i64) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  let r ← wloopL %[← homeOf t h, zero]
    (head := fun _ _ => pure (contIf .eq zero zero, %[zero], ()))
    (body := fun out cs _ => do
      let pos := cs.head
      let empties ← uextend64 (← vhighBits (← load { ty := .i8x16 } (← iadd t.ctrl pos)))
      when .ne empties zero do brk out %[← band (← iadd pos (← ctz empties)) t.mask]
      let st ← iaddImm cs.snd 16
      pure %[← band (← iadd pos st) t.mask, st])
  pure r.head

/-- The table doubled: a new one past the old, every key of the old placed
    again. The keys are distinct and the new table has nothing but them, so
    each goes to the first empty bucket of its probe, its hash taken once,
    and the counts are set once at the end. -/
def grow : Prog V L Unit := do
  let base ← basePtr
  let oldCtrl ← getW CTRL
  let oldSlots ← getW SLOTS
  let items ← getW ITEMS
  let buckets ← iaddImm (← getW MASK) 1
  makeTable (← ishlImm buckets 1)
  let t ← tab
  let top ← iadd t.ctrl (← iadd t.mask (← iconst64 1))
  let zero ← iconst64 0
  -- a group of 16 old buckets at a time, its full ones by their high bits
  forLoop (← ushrImm buckets 4) fun gi => do
    let g0 ← ishlImm gi 4
    let full ← bxor (← uextend64 (← vhighBits (← load { ty := .i8x16 }
      (← iadd base (← iadd oldCtrl g0))))) (← iconst64 0xFFFF)
    let _ ← wloopL %[full]
      (head := fun _ ms => pure (exitIfEq ms.head zero, %[], ()))
      (body := fun _ ms _ => do
        let m := ms.head
        let i ← iadd g0 (← ctz m)
        let slot ← iadd base (← iadd oldSlots (← ishlImm i 4))
        let k ← load64 slot
        let h ← hash k
        let b ← firstEmpty t h
        let tag ← tagOf t h
        istore8 tag (← iadd t.ctrl b)
        when .ult b (← iconst64 16) do istore8 tag (← iadd top b)
        let dst ← iadd t.slots (← ishlImm b 4)
        storeI64 k dst
        storeI64 (← load64 (← iaddImm slot 8)) (← iaddImm dst 8)
        pure %[← band m (← iaddImm m (-1))])
  setW ITEMS items
  setW ROOM (← isub (← getW ROOM) items)

def clearBody : Prog V L Unit := do
  -- the tables on huge pages where the system grants them, asked once: a
  -- large table's probes otherwise land on more pages than the translation
  -- cache holds
  when .eq (← getW ADVISED) (← iconst64 0) do
    let _ ← memAdviseHuge (← word HEAP) (← iconst64 (MEM - HEAP))
    setW ADVISED (← iconst64 1)
  setW BUMP (← iconst64 HEAP)
  makeTable (← iconst64 16)

/-- Keys a lookup or an insert takes together. -/
def BATCH : Nat := 8

/-- The fewest buckets at which lookups go in batches: 64 Ki buckets' slots
    and control bytes are past the second-level cache. -/
def SMALL_BUCKETS : Nat := 65536

/-- The table's counts back in memory from `room`, the buckets it still has
    before it is 7/8 full: `ITEMS` is the rest of those 7/8. -/
def putRoom (t : Tab V) (room : V .i64) : Prog V L Unit := do
  let buckets ← iaddImm t.mask 1
  setW ITEMS (← isub (← isub buckets (← ushrImm buckets 3)) room)
  setW ROOM room

/-- `k ↦ v` set in table `t` with `room` left, `h` the key's hash: its value
    replaced where it is there, else placed at the empty bucket its probe
    meets, the table doubled first where full (only while the next size fits
    the arena); the table and the room after. -/
def insertOne (t : Tab V) (room k v h : V .i64) : Prog V L (Vals V [.i64, .i64, .i64, .i64]) := do
  let (b, found) ← probeH t h k
  ifte (jTys := [.i64, .i64, .i64, .i64]) .ne found (← iconst64 0)
    (thn := do
      storeI64 v (← iaddImm (← iadd t.slots (← ishlImm b 4)) 8)
      pure %[t.ctrl, t.slots, t.mask, room])
    (els := ifte .eq room (← iconst64 0)
      (thn := do
        let r ← ifte (jTys := [.i64, .i64, .i64, .i64]) .ult t.mask (← iconst64 (MAX_BUCKETS / 2))
          (thn := do
            putRoom t room
            grow
            let t2 ← tab
            let (b2, _) ← probeH t2 h k
            placeAt t2 h b2 k v
            pure %[t2.ctrl, t2.slots, t2.mask, ← iaddImm (← getW ROOM) (-1)])
          (els := do pure %[t.ctrl, t.slots, t.mask, room])
        pure %[r.head, r.snd, r.thd, r.fth])
      (els := do
        placeAt t h b k v
        pure %[t.ctrl, t.slots, t.mask, ← iaddImm room (-1)]))

def insertBody : Prog V L Unit := do
  let dp ← dataPtr
  let n ← load64 dp
  when .ule (← iaddImm (← ishlImm n 4) 8) (← dataLen) do
    let keys ← iaddImm dp 8
    let vals ← iadd keys (← ishlImm n 3)
    -- batches of `BATCH` keys: their hashes taken and home groups read
    -- first, so their cache misses overlap, then each set in order
    let whole ← band n (← iconst64 (-(BATCH : Int)))
    -- the table and its room kept in registers, in memory again at the end
    let t0 ← tab
    let st0 := %[← iconst64 0, t0.ctrl, t0.slots, t0.mask, ← getW ROOM]
    let st1 ← wloop st0 (fun cs => pure (contIfULt cs.head whole, (cs.drop 1 : Vals V _), ()))
      (fun cs _ => do
        let mut t : Tab V := ⟨(cs.drop 1).head, (cs.drop 2).head, (cs.drop 3).head⟩
        let mut room := (cs.drop 4).head
        let mut batch := #[]
        for j in List.range BATCH do
          let i ← iaddImm cs.head j
          let k ← load64 (← iadd keys (← ishlImm i 3))
          let h ← hash k
          let _ ← load { ty := .i8x16 } (← iadd t.ctrl (← homeOf t h))
          batch := batch.push (k, ← load64 (← iadd vals (← ishlImm i 3)), h)
        for (k, v, h) in batch do
          let r ← insertOne t room k v h
          t := ⟨r.head, r.snd, r.thd⟩
          room := r.fth
        pure %[← iaddImm cs.head BATCH, t.ctrl, t.slots, t.mask, room])
    let st2 ← wloop (Vals.append %[whole] st1) (fun cs => pure (contIfULt cs.head n, (cs.drop 1 : Vals V _), ()))
      (fun cs _ => do
        let t : Tab V := ⟨(cs.drop 1).head, (cs.drop 2).head, (cs.drop 3).head⟩
        let k ← load64 (← iadd keys (← ishlImm cs.head 3))
        let r ← insertOne t (cs.drop 4).head k (← load64 (← iadd vals (← ishlImm cs.head 3))) (← hash k)
        pure %[← iaddImm cs.head 1, r.head, r.snd, r.thd, r.fth])
    putRoom ⟨st2.head, st2.snd, st2.thd⟩ st2.fth

/-- Keys `[i, e)` of `keys` looked up in table `t`, `b` at a time, `sum` and
    `cnt` carried: the values found added to the one, their number to the
    other. A batch's hashes and home groups are read first, so its cache
    misses are under way together; then each key's group matched: the first
    tag match holding the key, or no match and an empty byte, settle it;
    anything else takes the full probe. -/
def lookups (t : Tab V) (keys : V .i64) (b : Nat) (i e sum cnt : V .i64) :
    Prog V L (Vals V [.i64, .i64]) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  -- the value read only where the full probe found the key
  let slow (k : V .i64) : Prog V L (Vals V [.i64, .i64]) := do
    let (b, found) ← probe t k
    let v ← ifte (jTys := [.i64]) .ne found zero
      (thn := do pure %[← load64 (← iaddImm (← iadd t.slots (← ishlImm b 4)) 8)])
      (els := pure %[zero])
    pure %[v.head, found]
  wloop %[i, sum, cnt] (fun cs => pure (contIfULt cs.head e, %[cs.snd, cs.thd], ()))
    (fun cs _ => do
      let mut home := #[]
      for j in List.range b do
        let k ← load64 (← iadd keys (← ishlImm (← iaddImm cs.head j) 3))
        let h ← hash k
        let pos ← homeOf t h
        home := home.push (k, h, pos, ← load { ty := .i8x16 } (← iadd t.ctrl pos))
      let mut sum := cs.snd
      let mut cnt := cs.thd
      for (k, h, pos, g) in home do
        let hits ← uextend64 (← vhighBits (← icmp .eq g (← splat .i8x16 (← ireduce .i8 (← tagOf t h)))))
        let r ← ifte (jTys := [.i64, .i64]) .ne hits zero
          (thn := do
            let slot ← iadd t.slots (← ishlImm (← band (← iadd pos (← ctz hits)) t.mask) 4)
            ifte .eq (← load64 slot) k
              (thn := do pure %[← load64 (← iaddImm slot 8), one])
              (els := slow k))
          (els := do
            ifte .ne (← uextend64 (← vhighBits g)) zero
              (thn := do pure %[zero, zero])
              (els := slow k))
        sum ← iadd sum r.head
        cnt ← iadd cnt r.snd
      pure %[← iaddImm cs.head b, sum, cnt])

def getBody : Prog V L Unit := do
  let dp ← dataPtr
  let n ← load64 dp
  when .ule (← iaddImm (← ishlImm n 3) 8) (← dataLen) do
    when .uge (← outLen) (← iconst64 16) do
      let keys ← iaddImm dp 8
      let zero ← iconst64 0
      let t ← tab
      -- a table past the cache in batches of `BATCH`, the misses of more
      -- keys overlapped; one that fits it in batches of 4, fewer registers
      -- held across the batch; the last few keys one at a time
      let big ← icmp .uge t.mask (← iconst64 (SMALL_BUCKETS - 1))
      let w8 ← select big (← band n (← iconst64 (-(BATCH : Int)))) zero
      let w4 ← band n (← iconst64 (-4))
      let r8 ← lookups t keys BATCH zero w8 zero zero
      let r4 ← lookups t keys 4 w8 w4 r8.head r8.snd
      let r ← lookups t keys 1 w4 n r4.head r4.snd
      storeI64 r.head (← outPtr)
      storeI64 r.snd (← iaddImm (← outPtr) 8)

end

def clifIR : Except String (List FuncData) :=
  Prog.program
    [Prog.entry "clear" (Prog.compileProg 0 clearBody),
     Prog.entry "insert" (Prog.compileProg 1 insertBody),
     Prog.entry "get" (Prog.compileProg 2 getBody)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "hashtable" { functions := clif, required_memory := MEM, initial_memory := [] }]

end HashTable

def Bench.HashTable.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie HashTable.clifIR
  emitArtifacts outDir (HashTable.artifacts clif)

#eval ShipScan.check "Bench.HashTable" `Bench.HashTable.main
