import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

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
def TIMEOUT_MS : Nat := 300000

-- Zero 1024 bytes of counters at COUNTERS_OFF using 8-byte stores
def emitZeroCounters (ptr : Val) : IRBuilder Unit := do
  let zero ← iconst64 0
  let counterBase ← absAddr ptr COUNTERS_OFF
  forLoop .i64 (← iconst64 128) fun i => do  -- 1024/8 = 128 iterations
    let off ← ishl i =<< iconst64 3
    let addr ← iadd counterBase off
    store zero addr

-- One pass of LSD radix sort: scatter from src to dst based on byte `shift` (0,8,16,24)
-- n = number of i32 elements
-- ptr = shared memory base (for counter access)
-- For the final pass (shift=24, sign bit), we flip bit 7 to handle signed ordering.
def emitRadixPass (ptr src dst n shift : Val) (isSignedPass : Bool) : IRBuilder Unit := do
  let counterBase ← absAddr ptr COUNTERS_OFF

  -- Step 1: zero counters
  emitZeroCounters ptr

  -- Step 2: count occurrences
  forLoop .i64 n fun ci => do
    let srcAddr ← iadd src (←ishl ci =<< iconst64 2)
    let val ← uload32_64 srcAddr
    let byte ← band (← ushr val shift) =<< iconst64 0xFF
    let byte2 ← if isSignedPass then bxor byte =<< iconst64 0x80 else pure byte
    let cntAddr ← iadd counterBase (←ishl byte2 =<< iconst64 2)
    let cnt ← uload32_64 cntAddr
    let cnt1_32 ← ireduce32 (← iadd cnt =<< iconst64 1)
    store cnt1_32 cntAddr

  -- Step 3: prefix sum (exclusive) over 256 counters; accumulator = running sum
  let _ ← forLoopAcc .i64 .i64 (← iconst64 256) (← iconst64 0) fun pi psum => do
    let pcntAddr ← iadd counterBase (←ishl pi =<< iconst64 2)
    let pcnt ← uload32_64 pcntAddr
    store (← ireduce32 psum) pcntAddr
    iadd psum pcnt

  -- Step 4: scatter
  forLoop .i64 n fun si => do
    let ssrcAddr ← iadd src (←ishl si =<< iconst64 2)
    let sval ← uload32_64 ssrcAddr
    let sval32 ← ireduce32 sval
    let sbyte ← band (← ushr sval shift) =<< iconst64 0xFF
    let sbyte2 ← if isSignedPass then bxor sbyte =<< iconst64 0x80 else pure sbyte
    let scntAddr ← iadd counterBase (←ishl sbyte2 =<< iconst64 2)
    let destIdx ← uload32_64 scntAddr
    store (← ireduce32 (← iadd destIdx =<< iconst64 1)) scntAddr
    store sval32 (← iadd dst (←ishl destIdx =<< iconst64 2))
  return ()

def mainFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let dataPtr ← load64 =<< absAddr ptr 0x18
  let dataLen ← load64 =<< absAddr ptr 0x20
  let outPtr  ← load64 =<< absAddr ptr 0x28
  -- out_len at 0x20 — caller provides 2*data_len, second half is temp
  let n ← ushr dataLen =<< iconst64 2
  let tempPtr ← iadd outPtr dataLen  -- second half of out buffer

  -- Copy payload to outPtr (4-byte stores; data is i32-aligned)
  forLoop .i64 n fun ci => do
    let off ← ishl ci =<< iconst64 2
    let sv ← load32 (←iadd dataPtr off)
    store sv (←iadd outPtr off)

  -- 4 radix passes (LSD, 8 bits each)
  -- Pass 0: bits 0-7, out → temp
  let shift0 ← iconst64 0
  emitRadixPass ptr outPtr tempPtr n shift0 false
  -- Pass 1: bits 8-15, temp → out
  let shift1 ← iconst64 8
  emitRadixPass ptr tempPtr outPtr n shift1 false
  -- Pass 2: bits 16-23, out → temp
  let shift2 ← iconst64 16
  emitRadixPass ptr outPtr tempPtr n shift2 false
  -- Pass 3: bits 24-31 (sign byte), temp → out
  let shift3 ← iconst64 24
  emitRadixPass ptr tempPtr outPtr n shift3 true

  ret

def clifIR : String := buildProgram mainFn

def buildInitialMemory : List UInt8 := zeros MEM_SIZE

def controlActions : List Action :=
  [{ kind := .ClifCall, dst := 0, src := 1, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0,
  initial_memory := buildInitialMemory
}

def buildAlgorithm : Algorithm := {
  actions := controlActions,
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

def artifacts : Array Json :=
  #[toJsonEntry "sort_algorithm" buildConfig buildAlgorithm]

end SortBench
