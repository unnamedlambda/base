module
public import Lean
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace SharedSum

/-!
  # One memory, many callers

  A program whose state is meant to be shared: host threads call `add` on the
  same driver at once, and everything a call leaves in memory it leaves with an
  atomic read-modify-write, so no call's update is lost to another's.

  - `add` sums its input bytes, counts each byte into a histogram, adds the sum
    to a running total and counts itself. It answers the sum, and writes the
    total as it stood just before its own addition to the first eight bytes of
    its output when there is room. The calls' `[before, before + sum)` ranges
    then tile `[0, total)` exactly, which is how a host sees that each addition
    was applied once, in some single order.
  - `total` and `calls` answer the running total and the number of `add` calls.

  The histogram is 256 little-endian `u64`s, which the host reads from memory.

  ## A ring buffer, with a lock

  `push`, `pop` and `close` share a ring of 64 `u64` items between any number
  of producers and consumers, guarded by a lock built from the hardware's
  compare-and-swap and the runtime's wait and wake: a thread that finds the
  lock taken, the ring full, or the ring empty sleeps until another changes
  the word it is waiting on, rather than spinning.

  - The lock word is `0` free, `1` held, `2` held with threads waiting.
  - The head word counts items pushed, in its low 31 bits; its top bit says the
    ring is closed. The tail word counts items popped. Item `n` is in slot
    `n % 64`. Producers sleep on the tail word, consumers on the head word, so
    each sleeps on the word the other side changes.
  - `push` takes an item from the first eight input bytes and answers `1`, or
    `0` once the ring is closed. `pop` writes an item to the first eight output
    bytes and answers `1`, or `0` once the ring is closed and empty. `close`
    sets the top bit, which changes the word every consumer sleeps on, so none
    sleeps through it.

  A host can take part directly, through the same words: the lock is a
  compare-and-swap and the same wait and wake, which the C API offers too.
-/

/-- The running total, past the 64 bytes reserved for the runtime. -/
def TOTAL_AT : Nat := 0x40
/-- How many times `add` ran. -/
def CALLS_AT : Nat := 0x48
/-- 256 counters, one per byte value. -/
def HIST_AT : Nat := 0x50

/-- The ring's lock: `0` free, `1` held, `2` held with threads waiting. -/
def LOCK_AT : Nat := HIST_AT + 8 * 256
/-- Items pushed (low 31 bits), and whether the ring is closed (top bit). -/
def HEAD_AT : Nat := LOCK_AT + 4
/-- Items popped. -/
def TAIL_AT : Nat := LOCK_AT + 8
/-- The ring's slots, 64 `u64`s. -/
def SLOTS_AT : Nat := LOCK_AT + 16
def CAP : Nat := 64

def MEM_SIZE : Nat := SLOTS_AT + 8 * CAP

/-- Take the lock at `lk`: a compare-and-swap from free, and otherwise mark it
    contended and sleep until it is free. -/
def lock (lk : V .i64) : Prog V L Unit := do
  let zero ← iconst32 0
  let two ← iconst32 2
  let first ← atomicCas lk zero (← iconst32 1)
  let _ ← wloop1 first
    (fun c => pure (exitIfEq c zero, %[], ()))
    (fun _ _ => do
      let old ← atomicRmw .xchg lk two
      when .ne old zero do
        memWait lk two (← iconst64 (-1))
        pure ()
      pure %[old])
  pure ()

/-- Release the lock at `lk`, waking one sleeper if any marked it contended. -/
def unlock (lk : V .i64) : Prog V L Unit := do
  let old ← atomicRmw .xchg lk (← iconst32 0)
  when .eq old (← iconst32 2) do
    memWake lk (← iconst32 1)
    pure ()

/-- The ring's words, from the memory's base. -/
structure Ring (V : ClifTy → Type) where
  lk : V .i64
  head : V .i64
  tail : V .i64
  slots : V .i64

def ring : Prog V L (Ring V) := do
  let base ← basePtr
  pure { lk := ← absAddr base LOCK_AT, head := ← absAddr base HEAD_AT,
         tail := ← absAddr base TAIL_AT, slots := ← absAddr base SLOTS_AT }

/-- Where item `n` lives. -/
def slotOf (r : Ring V) (n : V .i32) : Prog V L (V .i64) := do
  iadd r.slots (← ishlImm (← uextend64 (← band n (← iconst32 (CAP - 1)))) 3)

def pushCode : Prog V L (V .i64) := do
  let r ← ring
  let zero ← iconst32 0
  let closed ← iconst32 0x80000000
  let count ← iconst32 0x7FFFFFFF
  let cap ← iconst32 CAP
  let item ← ifte (jTys := [.i64]) .uge (← dataLen) (← iconst64 8)
    (do pure %[← load64 (← dataPtr)]) (do pure %[← iconst64 0])
  let out ← wloop1L (← iconst64 1)
    (fun _ go => do pure (exitIfEq go (← iconst64 0), %[go], ()))
    (fun l go _ => do
      lock r.lk
      let h ← atomicLoad .i32 r.head
      let t ← atomicLoad .i32 r.tail
      when .ne (← band h closed) zero do
        unlock r.lk
        brk l %[← iconst64 0]
      when .uge (← band (← isub h t) count) cap do
        unlock r.lk
        memWait r.tail t (← iconst64 (-1))
        continueWith l %[go]
      storeI64 item.head (← slotOf r h)
      atomicStore (← iadd h (← iconst32 1)) r.head
      unlock r.lk
      memWake r.head (← iconst32 (-1))
      brk l %[← iconst64 1])
  pure out.head

def popCode : Prog V L (V .i64) := do
  let r ← ring
  let zero ← iconst32 0
  let closed ← iconst32 0x80000000
  let count ← iconst32 0x7FFFFFFF
  let out ← wloop1L (← iconst64 1)
    (fun _ go => do pure (exitIfEq go (← iconst64 0), %[go], ()))
    (fun l go _ => do
      lock r.lk
      let h ← atomicLoad .i32 r.head
      let t ← atomicLoad .i32 r.tail
      when .eq (← band (← isub h t) count) zero do
        unlock r.lk
        when .ne (← band h closed) zero do
          brk l %[← iconst64 0]
        memWait r.head h (← iconst64 (-1))
        continueWith l %[go]
      let item ← load64 (← slotOf r t)
      atomicStore (← iadd t (← iconst32 1)) r.tail
      unlock r.lk
      memWake r.tail (← iconst32 (-1))
      when .ule (← iconst64 8) (← outLen) do
        storeAt (← outPtr) 0 item
      brk l %[← iconst64 1])
  pure out.head

def closeCode : Prog V L (V .i64) := do
  let r ← ring
  lock r.lk
  let h ← atomicLoad .i32 r.head
  atomicStore (← bor h (← iconst32 0x80000000)) r.head
  unlock r.lk
  memWake r.head (← iconst32 (-1))
  iconst64 0

def addCode : Prog V L (V .i64) := do
  let base ← basePtr
  let data ← dataPtr
  let len ← dataLen
  let hist ← absAddr base HIST_AT
  let one ← iconst64 1
  let sum ← forLoopAcc len (← iconst64 0) fun i acc => do
    let b ← uload8_64 (← iadd data i)
    let _ ← atomicRmw .add (← iadd hist (← ishlImm b 3)) one
    iadd acc b
  let before ← atomicRmw .add (← absAddr base TOTAL_AT) sum
  let _ ← atomicRmw .add (← absAddr base CALLS_AT) one
  when .ule (← iconst64 8) (← outLen) do
    storeAt (← outPtr) 0 before
  pure sum

def totalCode : Prog V L (V .i64) := do
  atomicLoad .i64 (← absAddr (← basePtr) TOTAL_AT)

def callsCode : Prog V L (V .i64) := do
  atomicLoad .i64 (← absAddr (← basePtr) CALLS_AT)

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction,
    Prog.entry "add" (Prog.compileProgStatus 1 addCode),
    Prog.entry "total" (Prog.compileProgStatus 2 totalCode),
    Prog.entry "calls" (Prog.compileProgStatus 3 callsCode),
    Prog.entry "push" (Prog.compileProgStatus 4 pushCode),
    Prog.entry "pop" (Prog.compileProgStatus 5 popCode),
    Prog.entry "close" (Prog.compileProgStatus 6 closeCode)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "shared_sum" {
    functions := clif,
    required_memory := MEM_SIZE
  }]

end SharedSum

def Demo.SharedSum.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie SharedSum.clifIR
  emitArtifacts outDir (SharedSum.artifacts clif)

#eval ShipScan.check "Demo.SharedSum" `Demo.SharedSum.main
