module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# System benchmarks: the libraries a program links, at work

Two workloads through the surface a program calls, each answering one `u64`
at the start of the caller's output so the harness can check it against the
Rust it is timed against (`benchmarks/system`):

* `ht` — a table keyed by byte strings (`Lib.Ht`): insert `n` eight-byte keys
  with eight-byte values, then look every one up and sum the values.
* `wc` — count the newlines of the file the input names, read whole through
  the file adapter and counted sixteen bytes at a time.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace SystemBench

/-- The arena: the table's context slot, scratch for a key and a value, and
    room for the file `wc` reads. -/
def S_HT : Nat := 0x00
def KEY : Nat := 0x40
def VAL : Nat := 0x80
def BUF : Nat := 0x1000
/-- The largest file `wc` reads. -/
def WC_MAX : Nat := 64 * 1024 * 1024
def MEM : Nat := BUF + WC_MAX + 64

/-- A key from a counter: the counter spread over all 64 bits. -/
def keyOf (i : V .i64) : Prog V L (V .i64) := do
  imul (← iaddImm i 1) (← iconst64 (0x9E3779B97F4A7C15 - 2 ^ 64))

def ht : Prog V L Unit := do
  let base ← basePtr
  let n ← load64 (← dataPtr)
  let slot ← iaddImm base S_HT
  callLibVoid .htInit %[slot]
  let c ← load64 slot
  let _ ← callLib .htCreate %[c]
  let key ← iaddImm base KEY
  let val ← iaddImm base VAL
  let eight ← iconst32 8
  forLoop n fun i => do
    storeI64 (← keyOf i) key
    storeI64 i val
    callLibVoid .htInsert %[c, key, eight, val, eight]
  let sum ← forLoopAcc n (← iconst64 0) fun i acc => do
    storeI64 (← keyOf i) key
    let _ ← callLib .htLookup %[c, key, eight, val]
    iadd acc (← load64 val)
  callLibVoid .htCleanup %[slot]
  storeI64 sum (← outPtr)

/-- Newlines in `n` bytes from `p`: sixteen at a time, four vectors a trip,
    then the tail byte by byte. -/
def countNewlines (p n : V .i64) : Prog V L (V .i64) := do
  let nl ← splat .i8x16 (← iconst .i8 10)
  let trips ← ushrImm n 6
  let vecs ← forLoopAcc trips (← iconst64 0) fun t acc => do
    let q ← iadd p (← ishlImm t 6)
    let mut acc := acc
    for k in List.range 4 do
      let v ← loadI8x16 (← iaddImm q (16 * k))
      let bits ← vhighBits (← icmp .eq v nl)
      acc ← iadd acc (← uextend64 (← popcnt bits))
    pure acc
  let done ← ishlImm trips 6
  forLoopAcc (← isub n done) vecs fun i acc => do
    let b ← uload8_64 (← iadd p (← iadd done i))
    iadd acc (← uextend64 (← icmp .eq b (← iconst64 10)))

def wc : Prog V L Unit := do
  let base ← basePtr
  let buf ← iaddImm base BUF
  -- the path is the caller's whole data buffer
  let got ← ext (.base .fileReadToPtr) %[← dataPtr, ← dataLen, buf, ← iconst64 0, ← iconst64 WC_MAX]
  let n ← select (← icmp .slt got (← iconst64 0)) (← iconst64 0) got
  storeI64 (← countNewlines buf n) (← outPtr)

def clifIR : Except String (List FuncData) :=
  Prog.program
    [Prog.entry "noop" (Prog.compileProg 0 (pure ())),
     Prog.entry "ht" (Prog.compileProg 1 ht),
     Prog.entry "wc" (Prog.compileProg 2 wc)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "system_bench" { functions := clif, required_memory := MEM }]

end SystemBench

def Bench.System.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie SystemBench.clifIR
  emitArtifacts outDir (SystemBench.artifacts clif)

#eval ShipScan.check "Bench.System" `Bench.System.main
