import Lean
import AlgorithmLib.Gen
import LayoutScan

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

open AlgorithmLib.Prog


abbrev fnRead : Ffi := .fileRead
abbrev fnWrite : Ffi := .fileWrite

def code : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let zero    ← iconst64 0

  -- Copy the input path until NUL; no guard, because the test reads the byte
  -- the body just loaded.
  let inEnd ← dwloop %[zero] .eq zero (contOnTrue := false) [0]
    (body := fun c => do
      let si := c.head
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr INPUT_PATH_OFF) si)
      let si' ← iaddImm si 1
      return (ch, %[si']))

  -- Copy the output path, from where that stopped.
  let _ ← dwloop %[inEnd.head, zero] .eq zero (contOnTrue := false) []
    (body := fun c => do
      let si := c.head; let di := c.snd
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di)
      let si' ← iaddImm si 1
      let di' ← iaddImm di 1
      return (ch, %[si', di']))

  -- Read the file and set up the comparison vectors
  let fileSize ← ffi fnRead
    %[ptr, ← iconst64 INPUT_PATH_OFF, ← iconst64 INPUT_DATA, zero, zero]
  let dataBase ← absAddr ptr INPUT_DATA
  let endPos   ← isub fileSize (← iconst64 4)
  -- "that": t=116, h=104, a=97, t=116
  let tVec ← splat .i8x16 (← iconst .i8 116)
  let hVec ← splat .i8x16 (← iconst .i8 104)
  let aVec ← splat .i8x16 (← iconst .i8 97)

  -- SIMD search: 16 positions a trip
  let found ← wloop2 zero zero
    (head := fun pos cnt => return (exitIfSGt pos endPos, %[cnt], ()))
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
      return %[← iaddImm pos 16, ← iadd cnt hits])
  let total := found.head

  -- itoa: scale the divisor up while `div * 10 <= val`
  let scaled ← wloop1 (← iconst64 1)
    (head := fun div => do
      let d10 ← imul div (← iconst64 10)
      return (contIfULe d10 total, %[div], ()))
    (body := fun div _ => return %[← imul div (← iconst64 10)])

  -- then write one digit a trip until the divisor runs out
  let written ← dwloop %[total, scaled.head, ← iconst64 OUTPUT_BUF]
      .eq zero (contOnTrue := false) [2]
    (body := fun c => do
      let valW := c.head; let divW := c.snd; let wpos := c.thd
      let ten2 ← iconst64 10
      let dig  ← udiv valW divW
      let digB ← iadd dig (← iconst64 48)
      istore8 digB (← iadd ptr wpos)
      let rem  ← isub valW (← imul dig divW)
      let div' ← udiv divW ten2
      let wpos'← iaddImm wpos 1
      return (div', %[rem, div', wpos']))

  let wp := written.head
  istore8 (← iconst64 10) (← iadd ptr wp)
  istore8 (← iconst32 0) (← iadd ptr (← iaddImm wp 1))
  let _ ← ffi fnWrite %[ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 OUTPUT_BUF,
                        zero, zero]


def clifIR : Except String Program :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the runtime fills and `0x18`-`0x38` the
    input and output descriptors, so naming those is what stops an offset being
    placed where the runtime will overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"io_offsets",  0x18, 0x20⟩,
   ⟨"input_path",  INPUT_PATH_OFF, OUTPUT_PATH_OFF - INPUT_PATH_OFF⟩,
   ⟨"output_path", OUTPUT_PATH_OFF, OUTPUT_BUF - OUTPUT_PATH_OFF⟩,
   ⟨"output_buf",  OUTPUT_BUF, INPUT_DATA - OUTPUT_BUF⟩,
   ⟨"input_data",  INPUT_DATA, MAX_TEXT_BYTES⟩]


#eval LayoutScan.check "StringSearchAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def buildSetup (clif : Program) : Setup := {
  clif,
  memory_size := MEM_SIZE
}

def buildAlgorithm : Algorithm := {
  fn_idx := u32 1
}

def artifacts (clif : Program) : Array Json :=
  #[toJsonEntry "strsearch_algorithm" (buildSetup clif) buildAlgorithm]

end StringSearchBench
