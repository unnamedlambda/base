import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace HistogramBench1

/-
  Single-threaded histogram: 256-bin u32 histogram, written to file.
  Payload: "input_path\0output_path\0"
  4x-unrolled scan, 8x-unrolled zero loop.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def HIST_OFF        : Nat := 0x0400
def BINS            : Nat := 256
def HIST_BYTES      : Nat := BINS * 8
def DATA_OFF        : Nat := HIST_OFF + HIST_BYTES
def MAX_DATA_BYTES  : Nat := 64 * 1024 * 1024
def MEM_SIZE        : Nat := DATA_OFF + MAX_DATA_BYTES

set_option maxRecDepth 2048 in
def orchFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let fnRead  ← declareFileRead
  let fnWrite ← declareFileWrite
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let zero    ← iconst64 0

  let cpIn    ← declareBlock [.i64]
  let cpOut1  ← declareBlock [.i64]
  let cpOut   ← declareBlock [.i64, .i64]
  let readBlk ← declareBlock []
  let zeroLp  ← declareBlock [.i64]
  let scanPre ← declareBlock []
  let scan4L  ← declareBlock [.i64]
  let scan4B  ← declareBlock [.i64]
  let scalarL ← declareBlock [.i64]
  let scalarB ← declareBlock [.i64]
  let writeOut← declareBlock []

  jump cpIn.ref [zero]

  startBlock cpIn
  let si1 := cpIn.param 0
  let ch1  ← uload8_64 (← iadd dataPtr si1)
  istore8 ch1 (← iadd (← absAddr ptr INPUT_PATH_OFF) si1)
  let si1' ← iaddImm si1 1
  brif (← icmpImm .eq ch1 0) cpOut1.ref [si1'] cpIn.ref [si1']

  startBlock cpOut1
  jump cpOut.ref [cpOut1.param 0, zero]

  startBlock cpOut
  let si3 := cpOut.param 0; let di3 := cpOut.param 1
  let ch3  ← uload8_64 (← iadd dataPtr si3)
  istore8 ch3 (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di3)
  let si3' ← iaddImm si3 1; let di3' ← iaddImm di3 1
  brif (← icmpImm .eq ch3 0) readBlk.ref [] cpOut.ref [si3', di3']

  startBlock readBlk
  let fileSize ← readFile ptr fnRead INPUT_PATH_OFF DATA_OFF
  let n        ← ushrImm fileSize 2    -- n = bytes / 4
  let histPtr  ← absAddr ptr HIST_OFF
  let histEnd  ← iadd histPtr (← iconst64 HIST_BYTES)
  jump zeroLp.ref [histPtr]

  -- Zero histogram (8x unrolled, 64 bytes/iter)
  startBlock zeroLp
  let hp := zeroLp.param 0
  storeI64 zero hp
  storeI64 zero (← iaddImm hp 8)
  storeI64 zero (← iaddImm hp 16)
  storeI64 zero (← iaddImm hp 24)
  storeI64 zero (← iaddImm hp 32)
  storeI64 zero (← iaddImm hp 40)
  storeI64 zero (← iaddImm hp 48)
  storeI64 zero (← iaddImm hp 56)
  let hp' ← iaddImm hp 64
  brif (← icmp .ult hp' histEnd) zeroLp.ref [hp'] scanPre.ref []

  startBlock scanPre
  let dataPtr2 ← absAddr ptr DATA_OFF
  let dataEnd  ← iadd dataPtr2 (← ishlImm n 2)
  let n4       ← bandImm n (-4)
  let dataEnd4 ← iadd dataPtr2 (← ishlImm n4 2)
  brif (← icmp .ult dataPtr2 dataEnd4) scan4L.ref [dataPtr2]
                                         scalarL.ref [dataPtr2]

  -- 4x unrolled histogram scan
  startBlock scan4L
  let dp := scan4L.param 0
  brif (← icmp .ult dp dataEnd4) scan4B.ref [dp] scalarL.ref [dataEnd4]

  startBlock scan4B
  let dp2 := scan4B.param 0
  -- slot 0
  let v0   ← uload32_64 dp2
  let a0   ← iadd histPtr (← ishlImm v0 3)
  let c0   ← load64 a0
  storeI64 (← iaddImm c0 1) a0
  -- slot 1
  let v1   ← uload32_64 (← iaddImm dp2 4)
  let a1   ← iadd histPtr (← ishlImm v1 3)
  let c1   ← load64 a1
  storeI64 (← iaddImm c1 1) a1
  -- slot 2
  let v2   ← uload32_64 (← iaddImm dp2 8)
  let a2   ← iadd histPtr (← ishlImm v2 3)
  let c2   ← load64 a2
  storeI64 (← iaddImm c2 1) a2
  -- slot 3
  let v3   ← uload32_64 (← iaddImm dp2 12)
  let a3   ← iadd histPtr (← ishlImm v3 3)
  let c3   ← load64 a3
  storeI64 (← iaddImm c3 1) a3
  jump scan4L.ref [← iaddImm dp2 16]

  -- Scalar tail
  startBlock scalarL
  let dp3 := scalarL.param 0
  brif (← icmp .ult dp3 dataEnd) scalarB.ref [dp3] writeOut.ref []

  startBlock scalarB
  let dp4 := scalarB.param 0
  let vS  ← uload32_64 dp4
  let aS  ← iadd histPtr (← ishlImm vS 3)
  let cS  ← load64 aS
  storeI64 (← iaddImm cS 1) aS
  jump scalarL.ref [← iaddImm dp4 4]

  startBlock writeOut
  let _ ← call fnWrite [ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 HIST_OFF,
                         zero, ← iconst64 HIST_BYTES]
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++ noopAt 1 ++ "\n" ++ buildFunction 2 orchFn

def artifacts : Array Json :=
  #[toJsonEntry "hist1_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 2
  }]

end HistogramBench1
