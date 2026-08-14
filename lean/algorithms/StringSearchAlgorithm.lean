import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

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

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- `cl_file_read` as fn0, `cl_file_write` as fn1. -/
def env : FnEnv := env% [.fileIO]

def fnRead : Nat := IR.FFI.std.fileRead.id
def fnWrite : Nat := IR.FFI.std.fileWrite.id

def code : HProg.Code := clif% do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let zero    ← iconst64 0

  -- Copy the input path until NUL; no guard, because the test reads the byte
  -- the body just loaded.
  let inEnd ← dwloop [zero] .eq zero (contOnTrue := false) [0]
    (body := fun c => do
      let si := c.headD 0
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr INPUT_PATH_OFF) si)
      let si' ← iaddImm si 1
      return (ch, [si']))

  -- Copy the output path, from where that stopped.
  let _ ← dwloop [inEnd.headD 0, zero] .eq zero (contOnTrue := false) []
    (body := fun c => do
      let si := c.headD 0; let di := c.getD 1 0
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di)
      let si' ← iaddImm si 1
      let di' ← iaddImm di 1
      return (ch, [si', di']))

  -- Read the file and set up the comparison vectors
  let fileSize ← call fnRead
    [ptr, ← iconst64 INPUT_PATH_OFF, ← iconst64 INPUT_DATA, zero, zero]
  let dataBase ← absAddr ptr INPUT_DATA
  let endPos   ← isub fileSize (← iconst64 4)
  -- "that": t=116, h=104, a=97, t=116
  let tVec ← splat .i8x16 (← iconst .i8 116)
  let hVec ← splat .i8x16 (← iconst .i8 104)
  let aVec ← splat .i8x16 (← iconst .i8 97)

  -- SIMD search: 16 positions a trip
  let found ← wloop2 zero zero
    (head := fun pos cnt => return (exitIfSGt pos endPos, [cnt], ()))
    (body := fun pos cnt _ => do
      let base  ← iadd dataBase pos
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
      let hits  ← uextend64 (← popcnt bits)
      return [← iaddImm pos 16, ← iadd cnt hits])
  let total := found.headD 0

  -- itoa: scale the divisor up while `div * 10 <= val`
  let scaled ← wloop1 (← iconst64 1)
    (head := fun div => do
      let d10 ← imul div (← iconst64 10)
      return (contIfULe d10 total, [div], ()))
    (body := fun div _ => return [← imul div (← iconst64 10)])

  -- then write one digit a trip until the divisor runs out
  let written ← dwloop [total, scaled.headD 0, ← iconst64 OUTPUT_BUF]
      .eq zero (contOnTrue := false) [2]
    (body := fun c => do
      let valW := c.headD 0; let divW := c.getD 1 0; let wpos := c.getD 2 0
      let ten2 ← iconst64 10
      let dig  ← udiv valW divW
      let digB ← iadd dig (← iconst64 48)
      istore8 digB (← iadd ptr wpos)
      let rem  ← isub valW (← imul dig divW)
      let div' ← udiv divW ten2
      let wpos'← iaddImm wpos 1
      return (div', [rem, div', wpos']))

  let wp := written.headD 0
  istore8 (← iconst64 10) (← iadd ptr wp)
  istore8 (← iconst32 0) (← iadd ptr (← iaddImm wp 1))
  let _ ← call fnWrite [ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 OUTPUT_BUF,
                        zero, zero]

theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 code]

def buildSetup : Setup := {
  clif := clifIR,
  memory_size := MEM_SIZE
}

def buildAlgorithm : Algorithm := {
  fn_idx := u32 1
}

def artifacts : Array Json :=
  #[toJsonEntry "strsearch_algorithm" buildSetup buildAlgorithm]

end StringSearchBench
