import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

namespace RegexBench

/-
  Regex benchmark: count [a-z]+ing words (len>=4, all lowercase, ends "ing").
  Payload: "input_path\0output_path\0"
  Scalar byte scan with branchless "ing" match at word end.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def OUTPUT_BUF      : Nat := 0x0350
def INPUT_DATA      : Nat := 0x4000
def MAX_TEXT_BYTES  : Nat := 512 * 1024 * 1024
def MEM_SIZE        : Nat := INPUT_DATA + MAX_TEXT_BYTES

set_option maxRecDepth 2048 in
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- `cl_file_read` as fn0, `cl_file_write` as fn1. -/
def env : FnEnv := env% [.fileIO]

def fnRead : Nat := IR.FFI.std.fileRead.id
def fnWrite : Nat := IR.FFI.std.fileWrite.id

def code : HProg.Code := clif% do
  let ptr := basePtr
  let dataPtr← load64 (← absAddr ptr 0x18)
  let zero   ← iconst64 0

  let inEnd ← dwloop [zero] .eq zero (contOnTrue := false) [0]
    (body := fun c => do
      let si := c.headD 0
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr INPUT_PATH_OFF) si)
      let si' ← iaddImm si 1
      return (ch, [si']))

  let _ ← dwloop [inEnd.headD 0, zero] .eq zero (contOnTrue := false) []
    (body := fun c => do
      let si := c.headD 0; let di := c.getD 1 0
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di)
      let si' ← iaddImm si 1
      let di' ← iaddImm di 1
      return (ch, [si', di']))

  -- Read the file once; `fileSize` and `dataBase` are in scope for every scan
  let fileSize ← call fnRead
    [ptr, ← iconst64 INPUT_PATH_OFF, ← iconst64 INPUT_DATA, zero, zero]
  let dataBase ← absAddr ptr INPUT_DATA

  -- One position a trip. Whitespace advances; a word runs the inner scan, and
  -- both arms take the back edge themselves.
  let scanned ← wloop2 zero zero
    (head := fun pos cnt => return (exitIfSGe pos fileSize, [cnt], ()))
    (body := fun pos cnt _ => do
      let byte2 ← uload8_64 (← iadd dataBase pos)
      let space ← iconst64 32
      let one64 ← iconst64 1
      let _ ← ifte .ule byte2 space
        (thn := do
          continueWith [← iaddImm pos 1, cnt]
          pure [])
        (els := do
          -- scan to the end of the word, tracking whether it is all lowercase
          let w ← wloop2 pos one64
            (head := fun p allL => do
              let b ← uload8_64 (← iadd dataBase p)
              let sp ← iconst64 32
              return (exitIf .ule b sp, [p, allL], ()))
            (body := fun p allL _ => do
              let aLow  ← iconst64 97
              let r25   ← iconst64 25
              let bt ← uload8_64 (← iadd dataBase p)
              let shifted ← isub bt aLow
              let isLow   ← uextend64 (← icmp .ule shifted r25)
              return [← iaddImm p 1, ← band allL isLow])
          let pos5 := w.headD 0; let allL3 := w.getD 1 0
          let minLen ← iconst64 4
          let mask24 ← iconst64 16777215
          let ingLE  ← iconst64 6778473
          let len    ← isub pos5 pos
          let lenOk  ← uextend64 (← icmp .sge len minLen)
          let both   ← band allL3 lenOk
          let m3pos  ← iaddImm pos5 (-3)
          let raw4   ← uextend64 (← load32 (← iadd dataBase m3pos))
          let last3  ← band raw4 mask24
          let isIng  ← uextend64 (← icmp .eq last3 ingLE)
          let match1 ← band both isIng
          continueWith [← iaddImm pos5 1, ← iadd cnt match1]
          pure [])
      return [pos, cnt])
  let total := scanned.headD 0

  -- itoa: scale the divisor up, then write one digit a trip
  let ten   ← iconst64 10
  let one64 ← iconst64 1
  let scaled ← wloop1 one64
    (head := fun div => do
      let d10 ← imul div ten
      return (exitIf .ugt d10 total, [div], ()))
    (body := fun div _ => return [← imul div ten])

  let written ← dwloop [total, scaled.headD 0, ← iconst64 OUTPUT_BUF]
      .eq zero (contOnTrue := false) [2]
    (body := fun c => do
      let valW := c.headD 0; let divW := c.getD 1 0; let wposW := c.getD 2 0
      let dig  ← udiv valW divW
      let digB ← iadd dig (← iconst64 48)
      istore8 digB (← iadd ptr wposW)
      let rem  ← isub valW (← imul dig divW)
      let divW'← udiv divW ten
      let wpos'← iaddImm wposW 1
      return (divW', [rem, divW', wpos']))

  let wp := written.headD 0
  istore8 (← iconst64 10) (← iadd ptr wp)
  istore8 (← iconst32 0) (← iadd ptr (← iaddImm wp 1))
  let outOff ← iconst64 OUTPUT_PATH_OFF
  let bufOff ← iconst64 OUTPUT_BUF
  let _ ← call fnWrite [ptr, outOff, bufOff, zero, zero]

theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 code]

def artifacts : Array Json :=
  #[toJsonEntry "regex_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end RegexBench
