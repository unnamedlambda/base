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

Three workloads through the surface a program calls, each answering one `u64`
at the start of the caller's output so the harness can check it against the
Rust it is timed against (`benchmarks/system`):

* `ht` — a table keyed by byte strings (`Lib.Ht`): insert `n` eight-byte keys
  with eight-byte values, then look every one up and sum the values.
* `kv` — an ordered store (`Lib.Lmdb`): open the directory the input names,
  write `n` records in one transaction, in scrambled key order, and commit it,
  then scan them all in key order into the output; answers the count.
* `wc` — count the newlines of the file the input names, read whole through
  the file adapter and counted sixteen bytes at a time.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace SystemBench

/-- The arena: context slots, scratch for a key and a value, and room for the
    file `wc` reads. -/
def S_HT : Nat := 0x00
def S_KV : Nat := 0x08
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
  ffiVoid .htInit %[slot]
  let c ← load64 slot
  let _ ← ffi .htCreate %[c]
  let key ← iaddImm base KEY
  let val ← iaddImm base VAL
  let eight ← iconst32 8
  forLoop n fun i => do
    storeI64 (← keyOf i) key
    storeI64 i val
    ffiVoid .htInsert %[c, key, eight, val, eight]
  let sum ← forLoopAcc n (← iconst64 0) fun i acc => do
    storeI64 (← keyOf i) key
    let _ ← ffi .htLookup %[c, key, eight, val]
    iadd acc (← load64 val)
  ffiVoid .htCleanup %[slot]
  storeI64 sum (← outPtr)

/-- Records: an eight-byte big-endian key, `7919 i mod n` for the `i`th, so the
    keys are `0 … n-1` written in scrambled order, and 32 bytes of value. -/
def kv : Prog V L Unit := do
  let base ← basePtr
  let data ← dataPtr
  let out ← outPtr
  let n ← load64 data
  let path ← iaddImm data 8
  let slot ← iaddImm base S_KV
  ffiVoid .lmdbInit %[slot]
  let c ← load64 slot
  let h ← ffi .lmdbOpen %[c, path, ← iconst32 0]
  let _ ← ffi .lmdbBeginWriteTxn %[c, h]
  let key ← iaddImm base KEY
  let val ← iaddImm base VAL
  let eight ← iconst32 8
  let vlen ← iconst32 32
  forLoop n fun i => do
    storeI64 (← bswap (← urem (← imul i (← iconst64 7919)) n)) key
    let v ← keyOf i
    for k in List.range 4 do
      storeI64 v (← iaddImm val (8 * k))
    let _ ← ffi .lmdbPut %[c, h, key, eight, val, vlen]
    pure ()
  let _ ← ffi .lmdbCommitWriteTxn %[c, h]
  let count ← ffi .lmdbCursorScan
    %[c, h, key, ← iconst32 0, ← ireduce32 n, ← iaddImm out 64, ← iaddImm (← outLen) (-64)]
  ffiVoid .lmdbCleanup %[slot]
  storeI64 (← uextend64 count) out

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
  let got ← ffi .fileReadToPtr %[← dataPtr, buf, ← iconst64 0, ← iconst64 WC_MAX]
  let n ← select (← icmp .slt got (← iconst64 0)) (← iconst64 0) got
  storeI64 (← countNewlines buf n) (← outPtr)

def clifIR : Except String (List FuncData) :=
  Prog.program
    [Prog.entry "noop" (.ok noopFunction),
     Prog.entry "ht" (Prog.compileProg 1 ht),
     Prog.entry "kv" (Prog.compileProg 2 kv),
     Prog.entry "wc" (Prog.compileProg 3 wc)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "system_bench" { functions := clif, required_memory := MEM }]

end SystemBench

def Bench.System.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie SystemBench.clifIR
  emitArtifacts outDir (SystemBench.artifacts clif)

#eval ShipScan.check "Bench.System" `Bench.System.main
