import Lean
import Std
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace SortBench


/-
  Sort benchmark algorithm — LSD radix sort (4 passes, 8 bits each) via Cranelift JIT.

  Payload (via execute data arg): [i32 values: N]
  Output (via execute_into out arg): [i32 values: N] (sorted ascending)

  Memory layout (shared memory):
    0x0000  RESERVED  (40 bytes, runtime-managed)
    0x0028  scratch   (1024 bytes for 256 × i32 counters)
-/

def COUNTERS_OFF : Nat := 0x28  -- 256 × 4 = 1024 bytes for counting
def MEM_SIZE : Nat := 0x0428    -- 0x28 + 1024

/-- Zero 1024 bytes of counters at COUNTERS_OFF using 8-byte stores. -/
def zeroCounters : Prog V L Unit := do
  let zero ← iconst64 0
  let counterBase ← absAddr (← basePtr) COUNTERS_OFF
  let lim ← iconst64 128
  let _ ← wloop1 (← iconst64 0)
    (head := fun i => return (contIfULt i lim, %[], ()))
    (body := fun i _ => do
      let off ← ishlImm i 3
      let addr ← iadd counterBase off
      storeUnaligned zero addr
      return %[← iaddImm i 1])

/-- One pass of LSD radix sort: scatter from `src` to `dst` on byte `shift`
    (0, 8, 16, 24). `n` is the element count. The final pass flips bit 7 so the
    sign byte orders signed values correctly. -/
def radixPass (src dst n shift : V .i64) (isSignedPass : Bool) : Prog V L Unit := do
  let counterBase ← absAddr (← basePtr) COUNTERS_OFF

  -- Step 1: zero counters
  zeroCounters

  -- Step 2: count occurrences
  let _ ← wloop1 (← iconst64 0)
    (head := fun ci => return (contIfULt ci n, %[], ()))
    (body := fun ci _ => do
      let srcAddr ← iadd src (← ishlImm ci 2)
      let val ← uload32_64 srcAddr
      let byte ← band (← ushr val shift) (← iconst64 0xFF)
      let byte2 ← if isSignedPass then bxor byte (← iconst64 0x80) else pure byte
      let cntAddr ← iadd counterBase (← ishlImm byte2 2)
      let cnt ← uload32_64 cntAddr
      let cnt1_32 ← ireduce32 (← iadd cnt (← iconst64 1))
      storeUnaligned cnt1_32 cntAddr
      return %[← iaddImm ci 1])

  -- Step 3: prefix sum (exclusive) over 256 counters; accumulator = running sum
  let plim ← iconst64 256
  let psum0 ← iconst64 0
  let _ ← wloop2 (← iconst64 0) psum0
    (head := fun pi psum => return (contIfULt pi plim, %[psum], ()))
    (body := fun pi psum _ => do
      let pcntAddr ← iadd counterBase (← ishlImm pi 2)
      let pcnt ← uload32_64 pcntAddr
      storeUnaligned (← ireduce32 psum) pcntAddr
      let nextAcc ← iadd psum pcnt
      return %[← iaddImm pi 1, nextAcc])

  -- Step 4: scatter
  let _ ← wloop1 (← iconst64 0)
    (head := fun si => return (contIfULt si n, %[], ()))
    (body := fun si _ => do
      let ssrcAddr ← iadd src (← ishlImm si 2)
      let sval ← uload32_64 ssrcAddr
      let sval32 ← ireduce32 sval
      let sbyte ← band (← ushr sval shift) (← iconst64 0xFF)
      let sbyte2 ← if isSignedPass then bxor sbyte (← iconst64 0x80) else pure sbyte
      let scntAddr ← iadd counterBase (← ishlImm sbyte2 2)
      let destIdx ← uload32_64 scntAddr
      storeUnaligned (← ireduce32 (← iadd destIdx (← iconst64 1))) scntAddr
      storeUnaligned sval32 (← iadd dst (← ishlImm destIdx 2))
      return %[← iaddImm si 1])

def code : Prog V L Unit := do
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let outPtr  ← outPtr
  -- out_len at 0x20 — caller provides 2*data_len, second half is temp
  let n ← ushr dataLen (← iconst64 2)
  let tempPtr ← iadd outPtr dataLen  -- second half of out buffer

  -- Copy payload to outPtr (4-byte stores; data is i32-aligned)
  let _ ← wloop1 (← iconst64 0)
    (head := fun ci => return (contIfULt ci n, %[], ()))
    (body := fun ci _ => do
      let off ← ishlImm ci 2
      let sv ← load32 (← iadd dataPtr off)
      storeUnaligned sv (← iadd outPtr off)
      return %[← iaddImm ci 1])

  -- 4 radix passes (LSD, 8 bits each)
  -- Pass 0: bits 0-7, out → temp
  let shift0 ← iconst64 0
  radixPass outPtr tempPtr n shift0 false
  -- Pass 1: bits 8-15, temp → out
  let shift1 ← iconst64 8
  radixPass tempPtr outPtr n shift1 false
  -- Pass 2: bits 16-23, out → temp
  let shift2 ← iconst64 16
  radixPass outPtr tempPtr n shift2 false
  -- Pass 3: bits 24-31 (sign byte), temp → out
  let shift3 ← iconst64 24
  radixPass tempPtr outPtr n shift3 true


def clifIR : Except String Program :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

def buildInitialMemory : List UInt8 := zeros MEM_SIZE

def buildSetup (clif : Program) : Setup := {
  clif,
  memory_size := MEM_SIZE,
  initial_memory := buildInitialMemory
}

def buildAlgorithm : Algorithm := {
  fn_idx := u32 1
}

def artifacts (clif : Program) : Array Json :=
  #[toJsonEntry "sort_algorithm" (buildSetup clif) buildAlgorithm]

end SortBench
