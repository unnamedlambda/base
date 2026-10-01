module
public import AlgorithmLib.Surface.LibBase
meta import AlgorithmLib.Surface.LibBase
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lib.Lmdb` — the key-value store, as CLIF

The engine's LMDB entry points (`lmdbInit` … `lmdbCleanup`), stated in
`Host.Ffi`: environments by directory, each one ordered table of byte keys,
a write transaction per environment that reads see and a commit makes durable,
and scans in key order from a starting key. Here that is a function of the
program itself: nothing of LMDB is linked, and what the model states is what
this keeps.

**Tables.** A table is an array of entries sorted by key in LMDB's default
order (byte by byte, a key before every longer key it begins), found by binary
search, over one arena its keys and values are copied into; a value rewritten
at its old length is overwritten in place. A write transaction is a copy of
the committed table — two buffers, copied whole — and a commit replaces the
committed table with it. Opening reads the data file's body as the arena.

**Durability.** The committed table of the environment at directory `path`
lives in the file `path/data.tbl`: a header (a magic word and the file's length), then per entry
the key and value lengths as `u32` and their bytes. Opening reads it; every
change to the committed table rewrites it. The header's length is what makes a
shorter rewrite of a longer file read correctly.

**Limits.** One thread. Running out of heap is not modelled.
-/

namespace AlgorithmLib.LibLmdb

open AlgorithmLib.IR
open AlgorithmLib.Prog
open AlgorithmLib.Lib

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- The context: the environments, as a table of `ENV`-byte entries. -/
def ENVS : Nat := 0x00
def NENV : Nat := 0x08
def ECAP : Nat := 0x10
def CTX_SIZE : Nat := 0x20

/-- An environment: its data file's name (a heap C string, never null once
    open), then the committed table and the transaction's, and whether a
    transaction is open. -/
def E_FILE : Nat := 0x00
def E_COMMITTED : Nat := 0x08
def E_TXN : Nat := 0x40
def E_OPEN : Nat := 0x78
def ENV : Nat := 0x80

/-- A table's header, at an offset in its environment: the entry array, its
    length and capacity, the arena its keys and values live in, with the
    bytes used and its capacity, and the position after the last put. An
    entry is where its key and its value sit in the arena, their lengths, and
    the key's first eight bytes as a big-endian word, zero-padded, so a search
    reads the arena only where two keys begin alike. -/
def T_ARR : Nat := 0x00
def T_LEN : Nat := 0x08
def T_CAP : Nat := 0x10
def T_ARENA : Nat := 0x18
def T_AUSED : Nat := 0x20
def T_ACAP : Nat := 0x28
def T_HINT : Nat := 0x30
def TABLE : Nat := 0x38
def E_PFX : Nat := 32
def ENTRY : Nat := 40

def MAGIC : Int := 0x31424D4C46494C43  -- "CLIFLMB1"

-- ---------------------------------------------------------------------------
-- Keys and tables
-- ---------------------------------------------------------------------------

/-- LMDB's order on two keys: `-1`, `0` or `1`. Eight bytes at a time, read
    big-endian so that comparing the words compares the bytes in order, then
    the tail byte by byte, then the lengths. -/
def cmp (a la b lb : V .i64) : Prog V L (V .i64) := do
  let n ← umin la lb
  let zero ← iconst64 0
  let sign (x y : V .i64) : Prog V L (V .i64) := do
    select (← icmp .ult x y) (← iconst64 (-1)) (← select (← icmp .ugt x y) (← iconst64 1) zero)
  let words ← ushrImm n 3
  let w ← wloop2 zero zero
    (head := fun i r => do
      let stop ← bor (← icmp .uge i words) (← icmp .ne r zero)
      return (exitIf .ne stop (← iconst .i8 0), %[r], ()))
    (body := fun i _ _ => do
      let o ← ishlImm i 3
      let x ← bswap (← load64 (← iadd a o))
      let y ← bswap (← load64 (← iadd b o))
      return %[← iaddImm i 1, ← sign x y])
  let at_ ← ishlImm words 3
  let r ← wloop2 at_ w.head
    (head := fun i r => do
      let stop ← bor (← icmp .uge i n) (← icmp .ne r zero)
      return (exitIf .ne stop (← iconst .i8 0), %[r], ()))
    (body := fun i _ _ => do
      return %[← iaddImm i 1, ← sign (← uload8_64 (← iadd a i)) (← uload8_64 (← iadd b i))])
  select (← icmp .eq r.head zero) (← sign la lb) r.head

/-- The address of entry `i` of the table whose header is at `t`. -/
def entryAt (t i : V .i64) : Prog V L (V .i64) := do
  iadd (← load64 t) (← imul i (← iconst64 ENTRY))

/-- An entry's key and value, as addresses in its table's arena. -/
def keyAt (t e : V .i64) : Prog V L (V .i64) := do
  iadd (← load64 (← iaddImm t T_ARENA)) (← load64 e)

def valAt (t e : V .i64) : Prog V L (V .i64) := do
  iadd (← load64 (← iaddImm t T_ARENA)) (← load64 (← iaddImm e 16))

/-- A key's first eight bytes as a big-endian word, zero-padded. Two keys
    whose words differ are in the order of their words; LMDB's order puts a
    key before every longer key it begins, and a zero pad sorts first. -/
def prefixOf (p len : V .i64) : Prog V L (V .i64) := do
  let r ← ifte (jTys := [.i64]) .uge len (← iconst64 8) (do pure %[← bswap (← load64 p)]) (do
    let acc ← forLoopAcc len (← iconst64 0) fun i acc => do
      let b ← uload8_64 (← iadd p i)
      bor acc (← ishl b (← isub (← iconst64 56) (← ishlImm i 3)))
    pure %[acc])
  pure r.head

/-- Whether the key of entry `i` is below `key`, whose word is `kp`, as `1`
    or `0`. -/
def below (t i key klen kp : V .i64) : Prog V L (V .i64) := do
  let e ← entryAt t i
  let ep ← load64 (← iaddImm e E_PFX)
  let r ← ifte (jTys := [.i64]) .ne ep kp (do pure %[← uextend64 (← icmp .ult ep kp)]) (do
    pure %[← uextend64 (← icmp .slt (← cmp (← keyAt t e) (← load64 (← iaddImm e 8)) key klen)
      (← iconst64 0))])
  pure r.head

/-- The first position whose key is not below `key`. The position after the
    last put is tried first, as keys written in order find it; any other key
    is found by binary search. -/
def lowerBound (t key klen : V .i64) : Prog V L (V .i64) := do
  let len ← load64 (← iaddImm t T_LEN)
  let h ← load64 (← iaddImm t T_HINT)
  let kp ← prefixOf key klen
  let z ← iconst64 0
  -- the hint holds when entry `h` is not below the key (or is past the end)
  -- and entry `h - 1` is below it (or there is none)
  let atH ← ifte (jTys := [.i64]) .uge h len (do pure %[← uextend64 (← icmp .eq h len)]) (do
    pure %[← bxor (← below t h key klen kp) (← iconst64 1)])
  let ok ← ifte (jTys := [.i64]) .eq atH.head z (do pure %[z]) (do
    let r ← ifte (jTys := [.i64]) .eq h z (do pure %[← iconst64 1]) (do
      pure %[← below t (← iaddImm h (-1)) key klen kp])
    pure %[r.head])
  let r ← ifte (jTys := [.i64]) .ne ok.head z (do pure %[h]) (do
    let r ← wloop2 z len
      (head := fun lo hi => return (exitIf .uge lo hi, %[lo], ()))
      (body := fun lo hi _ => do
        let mid ← ushrImm (← iadd lo hi) 1
        let b ← icmp .ne (← below t mid key klen kp) z
        return %[← select b (← iaddImm mid 1) lo, ← select b hi mid])
    pure %[r.head])
  pure r.head

/-- Room for `n` more units of `unit` bytes in the buffer whose address,
    length in units and capacity in units sit at those offsets of `t`: the
    buffer doubled, or grown to fit, and its contents kept. The entry array's
    unit is an entry, the arena's a byte. -/
def reserveUnits (t : V .i64) (ptrOff usedOff capOff unit : Nat) (n : V .i64) : Prog V L Unit := do
  let used ← load64 (← iaddImm t usedOff)
  let capA ← iaddImm t capOff
  let need ← iadd used n
  when .ugt need (← load64 capA) (do
    let nc ← umax (← ishlImm (← load64 capA) 1) (← umax need (← iconst64 512))
    let w ← iconst64 unit
    let na ← calloc (← imul nc w)
    let old ← load64 (← iaddImm t ptrOff)
    when .ne old (← iconst64 0) (do
      let _ ← memcpy na old (← imul used w)
      free old)
    storeI64 na (← iaddImm t ptrOff)
    storeI64 nc capA)

/-- Copy `n` bytes to the end of the table's arena: where they now start. -/
def append (t p n : V .i64) : Prog V L (V .i64) := do
  reserveUnits t T_ARENA T_AUSED T_ACAP 1 n
  let usedA ← iaddImm t T_AUSED
  let used ← load64 usedA
  let _ ← memcpy (← iadd (← load64 (← iaddImm t T_ARENA)) used) p n
  storeI64 (← iadd used n) usedA
  pure used

/-- Store `val` under `key`: the value overwritten in place when it is as long
    as the old one and copied to the arena's end otherwise, or a new entry in
    order. -/
def put (t key klen val vlen : V .i64) : Prog V L Unit := do
  let i ← lowerBound t key klen
  let lenA ← iaddImm t T_LEN
  let len ← load64 lenA
  let same ← ifte (jTys := [.i64]) .uge i len (do pure %[← iconst64 0]) (do
    let e ← entryAt t i
    let c ← cmp (← keyAt t e) (← load64 (← iaddImm e 8)) key klen
    pure %[← uextend64 (← icmp .eq c (← iconst64 0))])
  let _ ← ifte (jTys := []) .ne same.head (← iconst64 0)
    (do
      let e ← entryAt t i
      let _ ← ifte (jTys := []) .eq (← load64 (← iaddImm e 24)) vlen
        (do let _ ← memcpy (← valAt t e) val vlen; pure %[])
        (do
          let vo ← append t val vlen
          let e ← entryAt t i
          storeI64 vo (← iaddImm e 16)
          storeI64 vlen (← iaddImm e 24)
          pure %[])
      pure %[])
    (do
      let ko ← append t key klen
      let vo ← append t val vlen
      reserveUnits t T_ARR T_LEN T_CAP ENTRY (← iconst64 1)
      let e ← entryAt t i
      let w ← iconst64 ENTRY
      let _ ← ext (.c .memmove) %[← iadd e w, e, ← imul (← isub len i) w]
      storeI64 ko e
      storeI64 klen (← iaddImm e 8)
      storeI64 vo (← iaddImm e 16)
      storeI64 vlen (← iaddImm e 24)
      storeI64 (← prefixOf key klen) (← iaddImm e E_PFX)
      storeI64 (← iaddImm len 1) lenA
      pure %[])
  storeI64 (← iaddImm i 1) (← iaddImm t T_HINT)

/-- Release a table's entries and arena, leaving it empty. -/
def clear (t : V .i64) : Prog V L Unit := do
  free (← load64 (← iaddImm t T_ARR))
  free (← load64 (← iaddImm t T_ARENA))
  let _ ← memset t (← iconst32 0) (← iconst64 TABLE)

/-- Copy table `src` into the empty table `dst`: its entries and its arena,
    each in one piece. -/
def copy (dst src : V .i64) : Prog V L Unit := do
  for (p, n, c) in [(T_ARR, T_LEN, T_CAP), (T_ARENA, T_AUSED, T_ACAP)] do
    let bytes ← if p == T_ARR then imul (← load64 (← iaddImm src n)) (← iconst64 ENTRY)
      else load64 (← iaddImm src n)
    let cap ← load64 (← iaddImm src c)
    let sz ← if p == T_ARR then imul cap (← iconst64 ENTRY) else pure cap
    let buf ← calloc (← umax sz (← iconst64 1))
    when .ne bytes (← iconst64 0) (do
      let _ ← memcpy buf (← load64 (← iaddImm src p)) bytes
      pure ())
    storeI64 buf (← iaddImm dst p)
    storeI64 (← load64 (← iaddImm src n)) (← iaddImm dst n)
    storeI64 cap (← iaddImm dst c)

-- ---------------------------------------------------------------------------
-- The data file
-- ---------------------------------------------------------------------------

/-- Write table `t` to `file`: header, then each entry's key and value lengths
    as `u32` and its bytes, in key order. -/
def persist (file t : V .i64) : Prog V L Unit := do
  let len ← load64 (← iaddImm t T_LEN)
  let size ← forLoopAcc len (← iconst64 16) fun i acc => do
    let e ← entryAt t i
    iadd acc (← iaddImm (← iadd (← load64 (← iaddImm e 8)) (← load64 (← iaddImm e 24))) 8)
  let buf ← calloc size
  storeI64 (← iconst64 MAGIC) buf
  storeI64 size (← iaddImm buf 8)
  let _ ← forLoopAcc len (← iconst64 16) fun i at_ => do
    let e ← entryAt t i
    let klen ← load64 (← iaddImm e 8)
    let vlen ← load64 (← iaddImm e 24)
    let p ← iadd buf at_
    storeI32 (← ireduce32 klen) p
    storeI32 (← ireduce32 vlen) (← iaddImm p 4)
    let _ ← memcpy (← iaddImm p 8) (← keyAt t e) klen
    let _ ← memcpy (← iadd (← iaddImm p 8) klen) (← valAt t e) vlen
    iadd at_ (← iaddImm (← iadd klen vlen) 8)
  let _ ← ffi .fileWriteFromPtr %[file, buf, ← iconst64 0, size]
  free buf

/-- Read the table in `file` into the empty table `t`, when there is one: the
    file's body becomes the arena, and the entries point into it, already in
    key order. -/
def load (file t : V .i64) : Prog V L Unit := do
  let hdr ← calloc (← iconst64 16)
  let got ← ffi .fileReadToPtr %[file, hdr, ← iconst64 0, ← iconst64 16]
  let ok ← band (← icmp .eq got (← iconst64 16)) (← icmp .eq (← load64 hdr) (← iconst64 MAGIC))
  when .ne ok (← iconst .i8 0) (do
    let size ← load64 (← iaddImm hdr 8)
    let buf ← calloc size
    let _ ← ffi .fileReadToPtr %[file, buf, ← iconst64 0, size]
    storeI64 buf (← iaddImm t T_ARENA)
    storeI64 size (← iaddImm t T_AUSED)
    storeI64 size (← iaddImm t T_ACAP)
    let _ ← wloop1 (← iconst64 16)
      (head := fun at_ => return (exitIf .uge at_ size, %[], ()))
      (body := fun at_ _ => do
        let p ← iadd buf at_
        let klen ← uextend64 (← load32 p)
        let vlen ← uextend64 (← load32 (← iaddImm p 4))
        reserveUnits t T_ARR T_LEN T_CAP ENTRY (← iconst64 1)
        let lenA ← iaddImm t T_LEN
        let len ← load64 lenA
        let e ← entryAt t len
        storeI64 (← iaddImm at_ 8) e
        storeI64 klen (← iaddImm e 8)
        storeI64 (← iadd (← iaddImm at_ 8) klen) (← iaddImm e 16)
        storeI64 vlen (← iaddImm e 24)
        storeI64 (← prefixOf (← iaddImm p 8) klen) (← iaddImm e E_PFX)
        storeI64 (← iaddImm len 1) lenA
        return %[← iadd at_ (← iaddImm (← iadd klen vlen) 8)]))
  free hdr

-- ---------------------------------------------------------------------------
-- The entry points
-- ---------------------------------------------------------------------------

/-- The environment a handle names, read as `u32`, or `0`. -/
def envOf (s : V .i64) (h : V .i32) : Prog V L (V .i64) := do
  let i ← uextend64 h
  let r ← ifte (jTys := [.i64]) .uge i (← load64 (← at_ s NENV)) (do pure %[← iconst64 0]) (do
    pure %[← iadd (← load64 (← at_ s ENVS)) (← imul i (← iconst64 ENV))])
  pure r.head

/-- `-1` for a null context or a handle naming no environment, `k` otherwise. -/
def withEnv (s : V .i64) (h : V .i32) (k : V .i64 → Prog V L (V .i64)) : Prog V L (V .i64) := do
  failIf .eq s (← iconst64 0) (do
    let e ← envOf s h
    failIf .eq e (← iconst64 0) (k e))

def init : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  storeI64 (← calloc (← iconst64 CTX_SIZE)) slot

def cleanup : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let s ← load64 slot
  when .ne s (← iconst64 0) (do
    let envs ← load64 (← at_ s ENVS)
    forLoop (← load64 (← at_ s NENV)) fun i => do
      let e ← iadd envs (← imul i (← iconst64 ENV))
      clear (← iaddImm e E_COMMITTED)
      clear (← iaddImm e E_TXN)
      free (← load64 e)
    free envs
    free s)
  storeI64 (← iconst64 0) slot

/-- Open the environment at a path, making the directory and any missing on the
    way: `-1` for an empty or overlong path, or one a directory cannot be made
    at; its handle otherwise, with what the directory holds read. -/
def open_ : StatusBody := do
  let (s ::ᵥ path ::ᵥ _ ::ᵥ .nil) ← entryParams [.i64, .i64, .i32]
  failIf .eq s (← iconst64 0) (do
    let n ← ext (.c .strlen) %[path]
    let bad ← bor (← icmp .eq n (← iconst64 0)) (← icmp .uge n (← iconst64 4096))
    failUnless0 bad (do
      let made ← ffi .fileCreateDirAll %[path]
      failIf .slt made (← iconst64 0) (do
        -- the data file's name: the path and "/data.tbl"
        let file ← calloc (← iaddImm n 10)
        let _ ← memcpy file path n
        let tail ← iadd file n
        storeI64 (← iconst64 0x6c62742e61746164 ) (← iaddImm tail 1)
        istore8 (← iconst32 0x2f) tail
        -- room for one more environment
        let nA ← at_ s NENV
        let capA ← at_ s ECAP
        let cnt ← load64 nA
        when .uge cnt (← load64 capA) (do
          let nc ← select (← icmp .eq (← load64 capA) (← iconst64 0)) (← iconst64 8)
            (← ishlImm (← load64 capA) 1)
          let na ← calloc (← imul nc (← iconst64 ENV))
          let old ← load64 (← at_ s ENVS)
          when .ne old (← iconst64 0) (do
            let _ ← memcpy na old (← imul cnt (← iconst64 ENV))
            free old)
          storeI64 na (← at_ s ENVS)
          storeI64 nc capA)
        let e ← iadd (← load64 (← at_ s ENVS)) (← imul cnt (← iconst64 ENV))
        let _ ← memset e (← iconst32 0) (← iconst64 ENV)
        storeI64 file e
        load file (← iaddImm e E_COMMITTED)
        storeI64 (← iaddImm cnt 1) nA
        pure cnt)))

/-- Open a write transaction on a copy of what is committed, aborting any
    transaction already open. -/
def beginWriteTxn : StatusBody := do
  let (s ::ᵥ h ::ᵥ .nil) ← entryParams [.i64, .i32]
  withEnv s h fun e => do
    let t ← iaddImm e E_TXN
    clear t
    copy t (← iaddImm e E_COMMITTED)
    storeI64 (← iconst64 1) (← iaddImm e E_OPEN)
    iconst64 0

def putE : StatusBody := do
  let (s ::ᵥ h ::ᵥ key ::ᵥ klen ::ᵥ val ::ᵥ vlen ::ᵥ .nil) ←
    entryParams [.i64, .i32, .i64, .i32, .i64, .i32]
  let z64 ← iconst64 0
  let bad ← bor (← bor (← icmp .slt klen (← iconst32 0)) (← icmp .slt vlen (← iconst32 0)))
    (← bor (← icmp .eq key z64) (← icmp .eq val z64))
  failUnless0 bad (withEnv s h fun e => do
    let kl ← uextend64 klen
    let vl ← uextend64 vlen
    -- a key of one to 511 bytes
    failIf .ugt (← iaddImm kl (-1)) (← iconst64 510) (do
      let txn ← load64 (← iaddImm e E_OPEN)
      let _ ← ifte (jTys := []) .ne txn z64
        (do put (← iaddImm e E_TXN) key kl val vl; pure %[])
        (do
          let c ← iaddImm e E_COMMITTED
          put c key kl val vl
          persist (← load64 e) c
          pure %[])
      iconst64 0))

/-- Make the open transaction what the environment holds. -/
def commitWriteTxn : StatusBody := do
  let (s ::ᵥ h ::ᵥ .nil) ← entryParams [.i64, .i32]
  withEnv s h fun e => do
    let openA ← iaddImm e E_OPEN
    failIf .eq (← load64 openA) (← iconst64 0) (do
      let c ← iaddImm e E_COMMITTED
      let t ← iaddImm e E_TXN
      clear c
      let _ ← memcpy c t (← iconst64 TABLE)
      let _ ← memset t (← iconst32 0) (← iconst64 TABLE)
      storeI64 (← iconst64 0) openA
      persist (← load64 e) c
      iconst64 0)

/-- Up to `max` entries in key order from `key` (from the first, for a
    non-positive length), as the model lays them out, stopping at one whose key
    or value is longer than a `u16` holds and at the first that does not fit in
    the `cap` bytes at `out`. Less room than the count takes answers `-1`. -/
def cursorScan : StatusBody := do
  let (s ::ᵥ h ::ᵥ key ::ᵥ klen ::ᵥ max ::ᵥ out ::ᵥ room ::ᵥ .nil) ←
    entryParams [.i64, .i32, .i64, .i32, .i32, .i64, .i64]
  let z64 ← iconst64 0
  let r ← ifte (jTys := [.i64]) .eq s z64 (pure %[z64]) (do
   let r ← ifte (jTys := [.i64]) .ult room (← iconst64 4) (do pure %[← iconst64 (-1)]) (do
    let e ← envOf s h
    let r ← ifte (jTys := [.i64]) .eq e z64
      (do
        storeI32 (← iconst32 0) out
        pure %[z64])
      (do
        let t ← select (← icmp .ne (← load64 (← iaddImm e E_OPEN)) z64)
          (← iaddImm e E_TXN) (← iaddImm e E_COMMITTED)
        let start ← ifte (jTys := [.i64]) .sgt klen (← iconst32 0)
          (do pure %[← lowerBound t key (← uextend64 klen)]) (pure %[z64])
        let len ← load64 (← iaddImm t T_LEN)
        let cap ← uextend64 max
        let lim ← iconst64 65535
        let res ← wloop %[start.head, z64, ← iconst64 4]
          (head := fun c => do
            let i := c.head
            let n := c.snd
            let inRange ← band (← icmp .ult i len) (← icmp .ult n cap)
            let fits ← ifte (jTys := [.i64]) .eq inRange (← iconst .i8 0) (do pure %[z64]) (do
              let e ← entryAt t i
              let kl ← load64 (← iaddImm e 8)
              let vl ← load64 (← iaddImm e 24)
              let ok ← band (← icmp .ule kl lim) (← icmp .ule vl lim)
              let fitsRoom ← icmp .ule (← iadd (← iaddImm c.thd 4) (← iadd kl vl)) room
              pure %[← uextend64 (← band ok fitsRoom)])
            return (exitIf .eq fits.head z64, %[n], ()))
          (body := fun c _ => do
            let e ← entryAt t c.head
            let klen ← load64 (← iaddImm e 8)
            let vlen ← load64 (← iaddImm e 24)
            let p ← iadd out c.thd
            istore16 klen p
            istore16 vlen (← iaddImm p 2)
            let _ ← memcpy (← iaddImm p 4) (← keyAt t e) klen
            let _ ← memcpy (← iadd (← iaddImm p 4) klen) (← valAt t e) vlen
            return %[← iaddImm c.head 1, ← iaddImm c.snd 1,
              ← iadd c.thd (← iaddImm (← iadd klen vlen) 4)])
        storeI32 (← ireduce32 res.head) out
        pure %[res.head])
    pure %[r.head])
   pure %[r.head])
  pure r.head

/-- The function implementing an LMDB entry point. -/
def implOf : Ffi → Option Impl
  | .lmdbInit => some (.void init)
  | .lmdbCleanup => some (.void cleanup)
  | .lmdbOpen => some (.status open_)
  | .lmdbBeginWriteTxn => some (.status beginWriteTxn)
  | .lmdbPut => some (.status putE)
  | .lmdbCommitWriteTxn => some (.status commitWriteTxn)
  | .lmdbCursorScan => some (.status cursorScan)
  | _ => none

end AlgorithmLib.LibLmdb
