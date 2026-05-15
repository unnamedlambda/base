import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace StringSearchBench

/-
  String search benchmark algorithm — Cranelift JIT version.

  Payload (via execute data arg): "input_path\0output_path\0"

  Memory layout (shared memory):
    0x0000  RESERVED        (56 bytes, runtime-managed)
    0x0100  INPUT_PATH      (256 bytes, copied from payload by CLIF)
    0x0200  OUTPUT_PATH     (256 bytes, copied from payload by CLIF)
    0x0350  OUTPUT_BUF      (64 bytes, itoa result for FileWrite)
    0x4000  INPUT_DATA      (variable, populated by FileRead)

  SIMD 4-byte pattern match for "that" across 16 positions per iteration.
  popcnt bitmask to count all occurrences.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def OUTPUT_BUF      : Nat := 0x0350
def INPUT_DATA      : Nat := 0x4000
def MAX_TEXT_BYTES  : Nat := 512 * 1024 * 1024
def MEM_SIZE        : Nat := INPUT_DATA + MAX_TEXT_BYTES

set_option maxRecDepth 2048 in
def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let fnRead  ← declareFileRead
  let fnWrite ← declareFileWrite
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let zero    ← iconst64 0

  let cpIn      ← declareBlock [.i64]
  let cpOut1    ← declareBlock [.i64]
  let cpOut     ← declareBlock [.i64, .i64]
  let readBlk   ← declareBlock []
  let searchL   ← declareBlock [.i64, .i64]   -- (pos, count)
  let searchB   ← declareBlock [.i64, .i64]
  let itoaStart ← declareBlock [.i64]          -- (total)
  let itoaWr    ← declareBlock [.i64, .i64, .i64]  -- (value, div, wpos): write digits
  let itoaNL    ← declareBlock [.i64]

  jump cpIn.ref [zero]

  -- Copy input path
  startBlock cpIn
  let si1 := cpIn.param 0
  let ch1  ← uload8_64 (← iadd dataPtr si1)
  istore8 ch1 (← iadd (← absAddr ptr INPUT_PATH_OFF) si1)
  let si1' ← iaddImm si1 1
  brif (← icmpImm .eq ch1 0) cpOut1.ref [si1'] cpIn.ref [si1']

  startBlock cpOut1
  jump cpOut.ref [cpOut1.param 0, zero]

  -- Copy output path
  startBlock cpOut
  let si3 := cpOut.param 0; let di3 := cpOut.param 1
  let ch3  ← uload8_64 (← iadd dataPtr si3)
  istore8 ch3 (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di3)
  let si3' ← iaddImm si3 1; let di3' ← iaddImm di3 1
  brif (← icmpImm .eq ch3 0) readBlk.ref [] cpOut.ref [si3', di3']

  -- Read file, set up SIMD vectors
  startBlock readBlk
  let fileSize ← readFile ptr fnRead INPUT_PATH_OFF INPUT_DATA
  let dataBase ← absAddr ptr INPUT_DATA
  let endPos   ← isub fileSize (← iconst64 4)
  -- "that": t=116, h=104, a=97, t=116
  let tVec ← splat .i8x16 (← iconst8 116)
  let hVec ← splat .i8x16 (← iconst8 104)
  let aVec ← splat .i8x16 (← iconst8 97)
  jump searchL.ref [zero, zero]

  -- SIMD search loop: 16 positions per iteration
  startBlock searchL
  let pos := searchL.param 0; let cnt := searchL.param 1
  brif (← icmp .sgt pos endPos) itoaStart.ref [cnt] searchB.ref [pos, cnt]

  startBlock searchB
  let pos2 := searchB.param 0; let cnt2 := searchB.param 1
  let base  ← iadd dataBase pos2
  let v0    ← loadI8x16 base
  let v1    ← loadI8x16 (← iaddImm base 1)
  let v2    ← loadI8x16 (← iaddImm base 2)
  let v3    ← loadI8x16 (← iaddImm base 3)
  let m0    ← icmp .eq v0 tVec
  let m1    ← icmp .eq v1 hVec
  let m2    ← icmp .eq v2 aVec
  let m3    ← icmp .eq v3 tVec
  let m01   ← band m0 m1
  let m23   ← band m2 m3
  let mAll  ← band m01 m23
  let bits  ← vhighBits mAll
  let hits  ← uextend64 (← popcnt32 bits)
  jump searchL.ref [← iaddImm pos2 16, ← iadd cnt2 hits]

  -- itoa: start with divisor=1, scale up while div*10 <= val
  startBlock itoaStart
  let total := itoaStart.param 0
  let finalDiv ← whileLoop1 .i64 (← iconst64 1)
    (fun div => do icmp .ule (← imul div (← iconst64 10)) total)
    (fun div => do imul div (← iconst64 10))
  jump itoaWr.ref [total, finalDiv, ← iconst64 OUTPUT_BUF]

  -- Write digits loop
  startBlock itoaWr
  let valW := itoaWr.param 0; let divW := itoaWr.param 1; let wpos := itoaWr.param 2
  let ten2 ← iconst64 10
  let dig  ← udiv valW divW
  let digB ← iadd dig (← iconst64 48)
  istore8 digB (← iadd ptr wpos)
  let rem  ← isub valW (← imul dig divW)
  let div' ← udiv divW ten2
  let wpos'← iaddImm wpos 1
  brif (← icmpImm .eq div' 0) itoaNL.ref [wpos'] itoaWr.ref [rem, div', wpos']

  startBlock itoaNL
  let wp := itoaNL.param 0
  istore8 (← iconst64 10) (← iadd ptr wp)
  istore8 (← iconst32 0) (← iadd ptr (← iaddImm wp 1))
  let _ ← call fnWrite [ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 OUTPUT_BUF,
                         zero, zero]
  ret

def clifIR : String := buildProgram mainFn

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0
}

def buildAlgorithm : Algorithm := {
  fn_idx := u32 1
}

def artifacts : Array Json :=
  #[toJsonEntry "strsearch_algorithm" buildConfig buildAlgorithm]

end StringSearchBench
