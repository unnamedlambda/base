import Lean
import AlgorithmLib.Gen
import LayoutScan

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

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


def code : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let zero   ← iconst64 0

  let inEnd ← dwloop %[zero] .eq zero (contOnTrue := false) [0]
    (body := fun c => do
      let si := c.head
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr INPUT_PATH_OFF) si)
      let si' ← iaddImm si 1
      return (ch, %[si']))

  let _ ← dwloop %[inEnd.head, zero] .eq zero (contOnTrue := false) []
    (body := fun c => do
      let si := c.head; let di := c.snd
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di)
      let si' ← iaddImm si 1
      let di' ← iaddImm di 1
      return (ch, %[si', di']))

  -- Read the file once; `fileSize` and `dataBase` are in scope for every scan
  let fileSize ← ffi .fileRead
    %[ptr, ← iconst64 INPUT_PATH_OFF, ← iconst64 INPUT_DATA, zero, zero]
  let dataBase ← absAddr ptr INPUT_DATA

  -- One position a trip. Whitespace advances; a word runs the inner scan, and
  -- both arms take the back edge themselves.
  let scanned ← wloopL %[zero, zero]
    (head := fun _ c => return (exitIfSGe c.head fileSize, %[c.snd], ()))
    (body := fun outer c _ => do
      let pos := c.head; let cnt := c.snd
      let byte2 ← uload8_64 (← iadd dataBase pos)
      let space ← iconst64 32
      let one64 ← iconst64 1
      let _ ← ifte (jTys := []) .ule byte2 space
        (thn := do
          continueWith outer %[← iaddImm pos 1, cnt])
        (els := do
          -- scan to the end of the word, tracking whether it is all lowercase
          -- The head already loaded the byte to test for whitespace; it hands
          -- it to the body rather than the body loading the same address a
          -- second time.
          let w ← wloop2 pos one64
            (head := fun p allL => do
              let b ← uload8_64 (← iadd dataBase p)
              let sp ← iconst64 32
              return (exitIf .ule b sp, %[p, allL], b))
            (body := fun p allL b => do
              let aLow  ← iconst64 97
              let r25   ← iconst64 25
              let shifted ← isub b aLow
              let isLow   ← uextend64 (← icmp .ule shifted r25)
              return %[← iaddImm p 1, ← band allL isLow])
          let pos5 := w.head; let allL3 := w.snd
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
          continueWith outer %[← iaddImm pos5 1, ← iadd cnt match1])
      return %[pos, cnt])
  let total := scanned.head

  -- itoa: scale the divisor up, then write one digit a trip
  let ten   ← iconst64 10
  let one64 ← iconst64 1
  let scaled ← wloop1 one64
    (head := fun div => do
      let d10 ← imul div ten
      return (exitIf .ugt d10 total, %[div], ()))
    (body := fun div _ => return %[← imul div ten])

  let written ← dwloop %[total, scaled.head, ← iconst64 OUTPUT_BUF]
      .eq zero (contOnTrue := false) [2]
    (body := fun c => do
      let valW := c.head; let divW := c.snd; let wposW := c.thd
      let dig  ← udiv valW divW
      let digB ← iadd dig (← iconst64 48)
      istore8 digB (← iadd ptr wposW)
      let rem  ← isub valW (← imul dig divW)
      let divW'← udiv divW ten
      let wpos'← iaddImm wposW 1
      return (divW', %[rem, divW', wpos']))

  let wp := written.head
  istore8 (← iconst64 10) (← iadd ptr wp)
  istore8 (← iconst32 0) (← iadd ptr (← iaddImm wp 1))
  let outOff ← iconst64 OUTPUT_PATH_OFF
  let bufOff ← iconst64 OUTPUT_BUF
  let _ ← ffi .fileWrite %[ptr, outOff, bufOff, zero, zero]

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the FFI's init calls fill, so naming
    those is what stops an offset being placed where one of those calls will
    overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"input_path",  INPUT_PATH_OFF, OUTPUT_PATH_OFF - INPUT_PATH_OFF⟩,
   ⟨"output_path", OUTPUT_PATH_OFF, OUTPUT_BUF - OUTPUT_PATH_OFF⟩,
   ⟨"output_buf",  OUTPUT_BUF, INPUT_DATA - OUTPUT_BUF⟩,
   ⟨"input_data",  INPUT_DATA, MAX_TEXT_BYTES⟩]


#eval LayoutScan.check "RegexBenchAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "regex_algorithm" {
    functions := clif,
    required_memory := MEM_SIZE
  }]

end RegexBench
