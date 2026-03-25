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
  let loopBlk ← declareBlock [.i64]
  let bodyBlk ← declareBlock [.i64]
  let doneBlk ← declareBlock []

  jump loopBlk.ref [←iconst64 0]

  startBlock loopBlk
  let i := loopBlk.param 0
  let limit ← iconst64 128  -- 1024/8 = 128 iterations
  let ok ← icmp .ult i limit
  brif ok bodyBlk.ref [i] doneBlk.ref []

  startBlock bodyBlk
  let i2 := bodyBlk.param 0
  let off ← ishl i2 =<< iconst64 3
  let addr ← iadd counterBase off
  store zero addr
  let i3 ← iadd i2 =<< iconst64 1
  jump loopBlk.ref [i3]

  startBlock doneBlk
  return ()

-- One pass of LSD radix sort: scatter from src to dst based on byte `shift` (0,8,16,24)
-- n = number of i32 elements
-- ptr = shared memory base (for counter access)
-- For the final pass (shift=24, sign bit), we flip bit 7 to handle signed ordering.
def emitRadixPass (ptr src dst n shift : Val) (isSignedPass : Bool) : IRBuilder Unit := do
  let counterBase ← absAddr ptr COUNTERS_OFF

  -- Step 1: zero counters
  emitZeroCounters ptr

  -- Step 2: count occurrences
  let countLoop ← declareBlock [.i64]
  let countBody ← declareBlock [.i64]
  let countDone ← declareBlock []

  jump countLoop.ref [←iconst64 0]

  startBlock countLoop
  let ci := countLoop.param 0
  let cok ← icmp .ult ci n
  brif cok countBody.ref [ci] countDone.ref []

  startBlock countBody
  let ci2 := countBody.param 0
  let srcAddr ← iadd src (←ishl ci2 =<< iconst64 2)
  let val ← uload32_64 srcAddr
  -- Extract byte: (val >> shift) & 0xFF
  let shifted ← ushr val shift
  let byte ← band shifted =<< iconst64 0xFF
  -- For signed pass, flip bit 7 so negative values sort before positive
  let byte2 ← if isSignedPass then bxor byte =<< iconst64 0x80 else pure byte
  -- counters[byte] += 1
  let cntAddr ← iadd counterBase (←ishl byte2 =<< iconst64 2)
  let cnt ← uload32_64 cntAddr
  let cnt1 ← iadd cnt =<< iconst64 1
  let cnt1_32 ← ireduce32 cnt1
  store cnt1_32 cntAddr
  let ci3 ← iadd ci2 =<< iconst64 1
  jump countLoop.ref [ci3]

  startBlock countDone

  -- Step 3: prefix sum (exclusive)
  let prefLoop ← declareBlock [.i64, .i64]  -- (i, running_sum)
  let prefBody ← declareBlock [.i64, .i64]
  let prefDone ← declareBlock []

  jump prefLoop.ref [←iconst64 0, ←iconst64 0]

  startBlock prefLoop
  let pi := prefLoop.param 0
  let psum := prefLoop.param 1
  let plimit ← iconst64 256
  let pok ← icmp .ult pi plimit
  brif pok prefBody.ref [pi, psum] prefDone.ref []

  startBlock prefBody
  let pi2 := prefBody.param 0
  let psum2 := prefBody.param 1
  let pcntAddr ← iadd counterBase (←ishl pi2 =<< iconst64 2)
  let pcnt ← uload32_64 pcntAddr
  -- Write prefix sum, then add count
  let psum2_32 ← ireduce32 psum2
  store psum2_32 pcntAddr
  let psum3 ← iadd psum2 pcnt
  let pi3 ← iadd pi2 =<< iconst64 1
  jump prefLoop.ref [pi3, psum3]

  startBlock prefDone

  -- Step 4: scatter
  let scatLoop ← declareBlock [.i64]
  let scatBody ← declareBlock [.i64]
  let scatDone ← declareBlock []

  jump scatLoop.ref [←iconst64 0]

  startBlock scatLoop
  let si := scatLoop.param 0
  let sok ← icmp .ult si n
  brif sok scatBody.ref [si] scatDone.ref []

  startBlock scatBody
  let si2 := scatBody.param 0
  let ssrcAddr ← iadd src (←ishl si2 =<< iconst64 2)
  let sval ← uload32_64 ssrcAddr
  let sval32 ← ireduce32 sval
  let sshifted ← ushr sval shift
  let sbyte ← band sshifted =<< iconst64 0xFF
  let sbyte2 ← if isSignedPass then bxor sbyte =<< iconst64 0x80 else pure sbyte
  -- dest_idx = counters[byte]; counters[byte]++
  let scntAddr ← iadd counterBase (←ishl sbyte2 =<< iconst64 2)
  let destIdx ← uload32_64 scntAddr
  let destIdx1 ← iadd destIdx =<< iconst64 1
  let destIdx1_32 ← ireduce32 destIdx1
  store destIdx1_32 scntAddr
  -- dst[dest_idx] = val
  let dstAddr ← iadd dst (←ishl destIdx =<< iconst64 2)
  store sval32 dstAddr
  let si3 ← iadd si2 =<< iconst64 1
  jump scatLoop.ref [si3]

  startBlock scatDone
  return ()

def mainFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let dataPtr ← load64 =<< absAddr ptr 0x08
  let dataLen ← load64 =<< absAddr ptr 0x10
  let outPtr  ← load64 =<< absAddr ptr 0x18
  -- out_len at 0x20 — caller provides 2*data_len, second half is temp
  let n ← ushr dataLen =<< iconst64 2
  let tempPtr ← iadd outPtr dataLen  -- second half of out buffer

  -- Copy payload to outPtr
  -- (simple 4-byte copy since data is i32-aligned)
  let copyLoop ← declareBlock [.i64]
  let copyBody ← declareBlock [.i64]
  let copyDone ← declareBlock []

  jump copyLoop.ref [←iconst64 0]

  startBlock copyLoop
  let ci := copyLoop.param 0
  let cok ← icmp .ult ci n
  brif cok copyBody.ref [ci] copyDone.ref []

  startBlock copyBody
  let ci2 := copyBody.param 0
  let off ← ishl ci2 =<< iconst64 2
  let sv ← load32 (←iadd dataPtr off)
  store sv (←iadd outPtr off)
  let ci3 ← iadd ci2 =<< iconst64 1
  jump copyLoop.ref [ci3]

  startBlock copyDone

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

end SortBench

def main : IO Unit := do
  let json := toJsonPair SortBench.buildConfig SortBench.buildAlgorithm
  IO.println (Json.compress json)
