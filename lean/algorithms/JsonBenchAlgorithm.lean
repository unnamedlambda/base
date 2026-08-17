import Lean
import AlgorithmLib.Gen
import LayoutScan

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

namespace JsonBench

/-
  JSON benchmark: sum "value": <N> integers.
  Payload: "input_path\0output_path\0"
  SIMD '\"v' prefix scan then 8-byte needle verify, then digit accumulate.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def OUTPUT_BUF      : Nat := 0x0350
def INPUT_DATA      : Nat := 0x4000
def MAX_JSON_BYTES  : Nat := 512 * 1024 * 1024
def MEM_SIZE        : Nat := INPUT_DATA + MAX_JSON_BYTES

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- `cl_file_read` as fn0, `cl_file_write` as fn1. -/
def env : FnEnv := env% [.fileIO]

def fnRead : Nat := IR.Ffi.fileRead.id
def fnWrite : Nat := IR.Ffi.fileWrite.id

def code : HProg.Code := clif% do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let zero    ← iconst64 0

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

  let fileSize ← call fnRead
    [ptr, ← iconst64 INPUT_PATH_OFF, ← iconst64 INPUT_DATA, zero, zero]
  let dataBase ← absAddr ptr INPUT_DATA
  let nine     ← iconst64 9
  let endPos   ← isub fileSize nine      -- scan until pos > fileSize-9
  let quot34   ← iconst .i8 34           -- '"'
  let vOf34    ← splat .i8x16 quot34
  let vv       ← iconst .i8 118          -- 'v'
  let vOfV     ← splat .i8x16 vv
  let needle   ← iconst64 2322206377019990390  -- "value\": " as LE i64

  -- 16 bytes a trip. A chunk with no candidate advances; a chunk with
  -- candidates runs the mask loop, which either finds a match, runs out of
  -- bits (back to the next chunk), or walks off the end (out of both loops).
  let scanned ← wloop2 zero zero
    (head := fun pos tot => return (exitIfSGt pos endPos, [tot], ()))
    (body := fun pos tot _ => do
      let p2   ← iadd dataBase pos
      let row0 ← loadI8x16 p2
      let row1 ← loadI8x16 (← iaddImm p2 1)
      let eq0  ← icmp .eq row0 vOf34
      let eq1  ← icmp .eq row1 vOfV
      let both ← band eq0 eq1
      let mask ← vhighBits both
      let zero32 ← iconst32 0
      let _ ← ifte .ne mask zero32
        (thn := do
          let found ← wloop1 mask
            (head := fun msk => do
              let off32 ← ctz msk
              let off  ← uextend64 off32
              let abs  ← iadd pos off
              -- past the end: leave the position scan as well
              let _ ← ifte .sgt abs endPos
                (thn := do brkTo 1 [tot]; pure [])
                (els := pure [])
              let bytes8 ← load64 (← iaddImm (← iadd dataBase abs) 1)
              let isNeedle ← icmp .eq bytes8 needle
              let z8 ← iconst .i8 0
              return (exitIf .ne isNeedle z8, [abs], ()))
            (body := fun msk _ => do
              let m1 ← iconst32 (-1)
              let newMask ← band msk (← iadd msk m1)
              let z32 ← iconst32 0
              -- no candidates left in this chunk: take the outer back edge
              let _ ← ifte .eq newMask z32
                (thn := do contTo 1 [← iaddImm pos 16, tot]; pure [])
                (els := pure [])
              return [newMask])
          -- a match: skip the needle and accumulate the digits that follow
          let ap := found.headD 0
          let d ← wloop2 (← iaddImm ap 9) zero
            (head := fun dp acc => do
              let byte ← uload8_64 (← iadd dataBase dp)
              let dg   ← isub byte (← iconst64 48)
              let nine2 ← iconst64 9
              return (exitIf .ugt dg nine2, [dp, acc], ()))
            (body := fun dp acc _ => do
              let byte ← uload8_64 (← iadd dataBase dp)
              let dg   ← isub byte (← iconst64 48)
              return [← iaddImm dp 1, ← iadd (← imul acc (← iconst64 10)) dg])
          continueWith [d.headD 0, ← iadd tot (d.getD 1 0)]
          pure [])
        (els := do
          continueWith [← iaddImm pos 16, tot]
          pure [])
      return [pos, tot])
  let total := scanned.headD 0

  -- itoa + write
  let ten ← iconst64 10
  let scaled ← wloop1 (← iconst64 1)
    (head := fun div => do
      let d10 ← imul div ten
      return (contIfULe d10 total, [div], ()))
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
  let _ ← call fnWrite [ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 OUTPUT_BUF,
                        zero, zero]

theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 code]

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
   ⟨"input_data",  INPUT_DATA, MAX_JSON_BYTES⟩]


#eval LayoutScan.check "JsonBenchAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts : Array Json :=
  #[toJsonEntry "json_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end JsonBench
