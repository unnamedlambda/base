module
public import AlgorithmLib.Surface.LibBase
meta import AlgorithmLib.Surface.LibBase
public import AlgorithmLib.Core.HtLayout
meta import AlgorithmLib.Core.HtLayout
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lib.Ht` — the hash table, as CLIF over the C heap

The engine's hash-table entry points (`htInit` … `htGetEntry`), stated in
`Host.Ffi`, as functions of the program itself. One table per context, keyed
and valued by byte strings; `htCreate` counts handles and the table exists once
one has been made, as the model says.

**Layout.** `htInit` allocates a block from the heap and writes its address
where the engine wrote its context pointer. The block holds the handle count,
the entries in the order they were first inserted, one arena every key and
value is copied into, and an open-addressed index over the entries, kept at
most half full, whose slots carry each key's hash tag so that a probe compares
keys only where the tags agree. A value rewritten at its old length is
overwritten in place; one of another length is copied to the arena's end.
`htGetEntry` walks the entries in insertion order, which is the order the model
states.

**Limits.** One thread. Running out of heap is not modelled: an allocation
that fails leaves the table as it was, and the call does nothing further.
`htInit` that cannot allocate the context or one of its arrays frees what it
allocated and leaves null in the slot.
-/

namespace AlgorithmLib.LibHt

open AlgorithmLib.IR
open AlgorithmLib.Prog
open AlgorithmLib.Lib

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

-- The block's layout and the hash's multipliers are `Core.HtLayout`'s.

/-- A hash of `n` bytes at `p`: eight bytes a step, the tail as one word,
    then a finishing mix, so the low bits an index takes and the high bits a
    slot keeps as its tag both depend on every byte. -/
def hash (p n : V .i64) : Prog V L (V .i64) := do
  let mix ← iconst64 MIX
  let words ← ushrImm n 3
  let h ← forLoopAcc words (← imul n mix) fun i h => do
    imul (← bxor h (← load64 (← iadd p (← ishlImm i 3)))) mix
  let at_ ← ishlImm words 3
  let tail ← forLoopAcc (← isub n at_) (← iconst64 0) fun j t => do
    let b ← uload8_64 (← iadd p (← iadd at_ j))
    bor t (← ishl b (← ishlImm j 3))
  let h ← imul (← bxor h tail) mix
  let h ← bxor h (← ushrImm h 32)
  let h ← imul h (← iconst64 FIN)
  bxor h (← ushrImm h 29)

/-- Whether `n` bytes at `a` and at `b` agree: `1` or `0`. -/
def bytesEq (a b n : V .i64) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  let words ← ushrImm n 3
  let d ← forLoopAcc words zero fun i d => do
    let o ← ishlImm i 3
    bor d (← bxor (← load64 (← iadd a o)) (← load64 (← iadd b o)))
  let at_ ← ishlImm words 3
  let d ← forLoopAcc (← isub n at_) d fun j d => do
    let o ← iadd at_ j
    bor d (← bxor (← uload8_64 (← iadd a o)) (← uload8_64 (← iadd b o)))
  uextend64 (← icmp .eq d zero)

/-- Copy `n` bytes to the end of the arena, doubling it first when they do
    not fit: where they now start, or `-1` with the arena as it was where it
    could not be grown. -/
def append (s p n : V .i64) : Prog V L (V .i64) := do
  let usedA ← at_ s AUSED
  let capA ← at_ s ACAP
  let used ← load64 usedA
  let need ← iadd used n
  let zero ← iconst64 0
  let room ← ifte (jTys := [.i64]) .ule need (← load64 capA) (do pure %[← iconst64 1]) (do
    let cap ← load64 capA
    let nc ← umax (← ishlImm cap 1) need
    let na ← calloc nc
    let r ← ifte (jTys := [.i64]) .eq na zero (pure %[zero]) (do
      let old ← load64 (← at_ s ARENA)
      let _ ← memcpy na old used
      free old
      storeI64 na (← at_ s ARENA)
      storeI64 nc capA
      pure %[← iconst64 1])
    pure %[r.head])
  let r ← ifte (jTys := [.i64]) .eq room.head zero (do pure %[← iconst64 (-1)]) (do
    let _ ← memcpy (← iadd (← load64 (← at_ s ARENA)) used) p n
    storeI64 need usedA
    pure %[used])
  pure r.head

/-- An entry's key, as an address in the arena. -/
def keyAt (s e : V .i64) : Prog V L (V .i64) := do
  iadd (← load64 (← at_ s ARENA)) (← load64 e)

def valAt (s e : V .i64) : Prog V L (V .i64) := do
  iadd (← load64 (← at_ s ARENA)) (← load64 (← iaddImm e 16))

/-- The index slot for a key of hash `h`: the slot holding its entry, with
    that entry's position plus one, or the empty slot it would go in, with
    `0`. A slot holds the entry's position plus one below and the hash's high
    half above; only a slot whose half agrees is compared by key. The index is
    never more than half full, so the probe ends. -/
def find (s key len h : V .i64) : Prog V L (V .i64 × V .i64) := do
  let mask ← iaddImm (← load64 (← at_ s ICAP)) (-1)
  let index ← load64 (← at_ s INDEX)
  let entries ← load64 (← at_ s ENTRIES)
  let tag ← ushrImm h 32
  let lo ← iconst64 0xffffffff
  let zero ← iconst64 0
  let r ← wloop1 (← band h mask)
    (head := fun i => do
      let slot ← iadd index (← ishlImm i 3)
      let v ← load64 slot
      let pos ← band v lo
      let found ← ifte (jTys := [.i64]) .ne (← ushrImm v 32) tag (pure %[zero]) (do
        let r ← ifte (jTys := [.i64]) .eq v zero (pure %[zero]) (do
          let e ← iadd entries (← imul (← iaddImm pos (-1)) (← iconst64 ENTRY))
          let same ← ifte (jTys := [.i64]) .ne (← load64 (← iaddImm e 8)) len (pure %[zero])
            (do pure %[← bytesEq (← keyAt s e) key len])
          pure %[same.head])
        pure %[r.head])
      let stop ← bor (← icmp .eq v zero) (← icmp .ne found.head zero)
      return (exitIf .ne stop (← iconst .i8 0), %[slot, ← select (← icmp .eq v zero) zero pos], ()))
    (body := fun i _ => do return %[← band (← iaddImm i 1) mask])
  pure (r.head, r.snd)

/-- Rebuild the index at twice its size. -/
def grow (s : V .i64) : Prog V L Unit := do
  let icapA ← at_ s ICAP
  let ncap ← ishlImm (← load64 icapA) 1
  let nidx ← calloc (← ishlImm ncap 3)
  when .ne nidx (← iconst64 0) (do
    free (← load64 (← at_ s INDEX))
    storeI64 nidx (← at_ s INDEX)
    storeI64 ncap icapA
    let entries ← load64 (← at_ s ENTRIES)
    forLoop (← load64 (← at_ s COUNT)) fun i => do
      let e ← iadd entries (← imul i (← iconst64 ENTRY))
      let len ← load64 (← iaddImm e 8)
      let key ← keyAt s e
      let h ← hash key len
      let (slot, _) ← find s key len h
      storeI64 (← bor (← ishlImm (← ushrImm h 32) 32) (← iaddImm i 1)) slot)

/-- Append a new entry for `key` at index slot `slot`: the key and the value
    copied into the arena, the entry array doubled first if it is full. Where
    the entry array or the arena cannot be grown, the table's entries are as
    they were. -/
def insertAt (s slot key klen val vlen h : V .i64) : Prog V L Unit := do
  let countA ← at_ s COUNT
  let ecapA ← at_ s ECAP
  let count ← load64 countA
  let entryW ← iconst64 ENTRY
  let zero ← iconst64 0
  let failed ← iconst64 (-1)
  let room ← ifte (jTys := [.i64]) .ult count (← load64 ecapA) (do pure %[← iconst64 1]) (do
    let ncap ← ishlImm (← load64 ecapA) 1
    let ne ← calloc (← imul ncap entryW)
    let r ← ifte (jTys := [.i64]) .eq ne zero (pure %[zero]) (do
      let old ← load64 (← at_ s ENTRIES)
      let _ ← memcpy ne old (← imul count entryW)
      free old
      storeI64 ne (← at_ s ENTRIES)
      storeI64 ncap ecapA
      pure %[← iconst64 1])
    pure %[r.head])
  when .ne room.head zero (do
    let ko ← append s key klen
    when .ne ko failed (do
      let vo ← append s val vlen
      when .ne vo failed (do
        let e ← iadd (← load64 (← at_ s ENTRIES)) (← imul count entryW)
        storeI64 ko e
        storeI64 klen (← iaddImm e 8)
        storeI64 vo (← iaddImm e 16)
        storeI64 vlen (← iaddImm e 24)
        let n ← iaddImm count 1
        storeI64 (← bor (← ishlImm (← ushrImm h 32) 32) n) slot
        storeI64 n countA
        when .ugt (← ishlImm n 1) (← load64 (← at_ s ICAP)) (grow s))))

/-- The entry at position `pos` plus one. -/
def entryOf (s pos : V .i64) : Prog V L (V .i64) := do
  iadd (← load64 (← at_ s ENTRIES)) (← imul (← iaddImm pos (-1)) (← iconst64 ENTRY))

/-- Set `key` to `val`: the value overwritten in place when it is as long as
    the old one, copied to the arena's end otherwise, or a new entry. -/
def set (s key klen val vlen : V .i64) : Prog V L Unit := do
  let h ← hash key klen
  let (slot, pos) ← find s key klen h
  let _ ← ifte (jTys := []) .eq pos (← iconst64 0)
    (do insertAt s slot key klen val vlen h; pure %[])
    (do
      let e ← entryOf s pos
      let _ ← ifte (jTys := []) .eq (← load64 (← iaddImm e 24)) vlen
        (do let _ ← memcpy (← valAt s e) val vlen; pure %[])
        (do
          let vo ← append s val vlen
          when .ne vo (← iconst64 (-1)) (do
            let e ← entryOf s pos
            storeI64 vo (← iaddImm e 16)
            storeI64 vlen (← iaddImm e 24))
          pure %[])
      pure %[])
  pure ()

/-- `k` for a live context whose table has been made; `dflt` otherwise. -/
def withTable {ty} (s : V .i64) (dflt : Prog V L (V ty)) (k : Prog V L (V ty)) :
    Prog V L (V ty) := do
  let zero ← iconst64 0
  let r ← ifte (jTys := [ty]) .eq s zero (do pure %[← dflt]) (do
    let r ← ifte (jTys := [ty]) .eq (← load64 (← at_ s CREATED)) zero (do pure %[← dflt])
      (do pure %[← k])
    pure %[r.head])
  pure r.head

-- ---------------------------------------------------------------------------
-- The entry points
-- ---------------------------------------------------------------------------

def init : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let zero ← iconst64 0
  let s ← calloc (← iconst64 SIZE)
  let r ← ifte (jTys := [.i64]) .eq s zero (pure %[zero]) (do
    let e ← calloc (← iconst64 (64 * ENTRY))
    let r ← ifte (jTys := [.i64]) .eq e zero (do free s; pure %[zero]) (do
      let ix ← calloc (← iconst64 (128 * 8))
      let r ← ifte (jTys := [.i64]) .eq ix zero (do free e; free s; pure %[zero]) (do
        let ar ← calloc (← iconst64 4096)
        let r ← ifte (jTys := [.i64]) .eq ar zero (do free ix; free e; free s; pure %[zero]) (do
          storeI64 e (← at_ s ENTRIES)
          storeI64 (← iconst64 64) (← at_ s ECAP)
          storeI64 ix (← at_ s INDEX)
          storeI64 (← iconst64 128) (← at_ s ICAP)
          storeI64 ar (← at_ s ARENA)
          storeI64 (← iconst64 4096) (← at_ s ACAP)
          pure %[s])
        pure %[r.head])
      pure %[r.head])
    pure %[r.head])
  storeI64 r.head slot

def cleanup : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let s ← load64 slot
  let zero ← iconst64 0
  when .ne s zero (do
    free (← load64 (← at_ s ENTRIES))
    free (← load64 (← at_ s INDEX))
    free (← load64 (← at_ s ARENA))
    free s)
  storeI64 zero slot

def create : StatusBody := do
  let (s ::ᵥ .nil) ← entryParams [.i64]
  let r ← ifte (jTys := [.i64]) .eq s (← iconst64 0) (do pure %[← iconst64 0xFFFFFFFF]) (do
    let a ← at_ s CREATED
    let n ← load64 a
    storeI64 (← iaddImm n 1) a
    pure %[n])
  pure r.head

def lookup : StatusBody := do
  let (s ::ᵥ key ::ᵥ klen ::ᵥ out ::ᵥ .nil) ← entryParams [.i64, .i64, .i32, .i64]
  withTable s (iconst64 0xFFFFFFFF) (do
    let n ← uextend64 klen
    let (_, pos) ← find s key n (← hash key n)
    let r ← ifte (jTys := [.i64]) .eq pos (← iconst64 0) (do pure %[← iconst64 0xFFFFFFFF]) (do
      let e ← entryOf s pos
      let vlen ← load64 (← iaddImm e 24)
      let _ ← memcpy out (← valAt s e) vlen
      pure %[vlen])
    pure r.head)

def insert : Body := do
  let (s ::ᵥ key ::ᵥ klen ::ᵥ val ::ᵥ vlen ::ᵥ .nil) ← entryParams [.i64, .i64, .i32, .i64, .i32]
  let _ ← withTable s (iconst64 0) (do
    set s key (← uextend64 klen) val (← uextend64 vlen)
    iconst64 0)
  pure ()

/-- Add to the counter in a value's first eight bytes, or insert the addend as
    a new eight-byte value; the counter's new value either way. -/
def increment : StatusBody := do
  let (s ::ᵥ key ::ᵥ klen ::ᵥ addend ::ᵥ .nil) ← entryParams [.i64, .i64, .i32, .i64]
  withTable s (pure addend) (do
    let n ← uextend64 klen
    let h ← hash key n
    let (slot, pos) ← find s key n h
    let r ← ifte (jTys := [.i64]) .eq pos (← iconst64 0)
      (do
        let sc ← at_ s SCRATCH
        storeI64 addend sc
        insertAt s slot key n sc (← iconst64 8) h
        pure %[addend])
      (do
        let vp ← valAt s (← entryOf s pos)
        let next ← iadd (← load64 vp) addend
        storeI64 next vp
        pure %[next])
    pure r.head)

def count : StatusBody := do
  let (s ::ᵥ .nil) ← entryParams [.i64]
  withTable s (iconst64 0) (load64 (← at_ s COUNT))

/-- The entry at `index` in insertion order: its key and value copied out, and
    the key's length; `-1` past the end. -/
def getEntry : StatusBody := do
  let (s ::ᵥ index ::ᵥ kout ::ᵥ vout ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64]
  withTable s (iconst64 (-1)) (do
    let i ← uextend64 index
    failIf .uge i (← load64 (← at_ s COUNT)) (do
      let e ← entryOf s (← iaddImm i 1)
      let klen ← load64 (← iaddImm e 8)
      let _ ← memcpy kout (← keyAt s e) klen
      let _ ← memcpy vout (← valAt s e) (← load64 (← iaddImm e 24))
      pure klen))

/-- The function implementing a hash-table entry point. -/
def implOf : Ffi → Option Impl
  | .htInit => some (.void init)
  | .htCleanup => some (.void cleanup)
  | .htCreate => some (.status create)
  | .htLookup => some (.status lookup)
  | .htInsert => some (.void insert)
  | .htIncrement => some (.status increment)
  | .htCount => some (.status count)
  | .htGetEntry => some (.status getEntry)
  | _ => none

end AlgorithmLib.LibHt
