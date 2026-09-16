import Lean
import AlgorithmLib.Gen
import LayoutScan

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

namespace CsvBench

/-
  CSV benchmark: sum salary column (field 6) across all rows.
  Payload: "input_path\0output_path\0"
  SIMD newline scan to skip header; SIMD comma scan to find field 6; digit accumulate.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def LEFT_VAL        : Nat := 0x0350
def CSV_DATA        : Nat := 0x2000
def MAX_CSV_BYTES   : Nat := 512 * 1024 * 1024
def MEM_SIZE        : Nat := CSV_DATA + MAX_CSV_BYTES

set_option maxRecDepth 4096
open AlgorithmLib.Prog


abbrev fnRead : Ffi := .fileRead
abbrev fnWrite : Ffi := .fileWrite

def code : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let zero    ← iconst64 0
  let zero32  ← iconst32 0

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

  let fileSize ← ffi fnRead
    %[ptr, ← iconst64 INPUT_PATH_OFF, ← iconst64 CSV_DATA, zero, zero]
  let csvBase  ← absAddr ptr CSV_DATA
  let nlVec    ← splat .i8x16 (← iconst .i8 10)
  let cmVec    ← splat .i8x16 (← iconst .i8 44)

  -- Skip the header row. Dispatches between a SIMD chunk and a scalar byte;
  -- every path either leaves with the newline's position or takes the edge, so
  -- the loop needs no test of its own.
  let hdr ← dwloopL %[zero] .eq zero (contOnTrue := true) [0]
    (body := fun hdrLoop c => do
      let hp := c.head
      let _ ← ifte (jTys := []) .sle (← iaddImm hp 16) fileSize
        (thn := do
          let row  ← loadI8x16 (← iadd csvBase hp)
          let eq   ← icmp .eq row nlVec
          let mask ← vhighBits eq
          let _ ← ifte (jTys := []) .ne mask zero32
            (thn := do
              let off ← uextend64 (← ctz mask)
              brk hdrLoop %[← iadd hp off]
              pure %[])
            (els := do continueWith hdrLoop %[← iaddImm hp 16]; pure %[])
          pure %[])
        (els := do
          let b ← uload8_64 (← iadd csvBase hp)
          let _ ← ifte (jTys := []) .eq b (← iconst64 10)
            (thn := do brk hdrLoop %[hp]; pure %[])
            (els := do continueWith hdrLoop %[← iaddImm hp 1]; pure %[])
          pure %[])
      return (zero, %[hp]))

  -- Sum the sixth field of every row. Same shape: a dispatch with several back
  -- edges and one way out, when the scan passes the end of the file.
  let scanned ← dwloopL %[← iaddImm hdr.head 1, zero, zero]
      .eq zero (contOnTrue := true) [1]
    (body := fun fieldLoop c => do
      let cp := c.head; let tot := c.snd; let cc := c.thd
      let _ ← ifte (jTys := []) .sle (← iaddImm cp 16) fileSize
        (thn := do
          let row2 ← loadI8x16 (← iadd csvBase cp)
          let eq2  ← icmp .eq row2 cmVec
          let msk2 ← vhighBits eq2
          let _ ← ifte (jTys := []) .ne msk2 zero32
            (thn := do
              let off2 ← uextend64 (← ctz msk2)
              let abs  ← iadd cp off2
              let cc'  ← iaddImm cc 1
              let _ ← ifte (jTys := []) .eq cc' (← iconst64 5)
                (thn := do
                  let d ← wloop2 (← iaddImm abs 1) zero
                    (head := fun dp acc => do
                      let b3 ← uload8_64 (← iadd csvBase dp)
                      return (exitIf .eq b3 (← iconst64 10), %[dp, acc], ()))
                    (body := fun dp acc _ => do
                      let b4 ← uload8_64 (← iadd csvBase dp)
                      return %[← iaddImm dp 1,
                              ← iadd (← imul acc (← iconst64 10))
                                     (← isub b4 (← iconst64 48))])
                  let dp' ← iaddImm (d.head) 1
                  let tot' ← iadd tot (d.snd)
                  let _ ← ifte (jTys := []) .sge dp' fileSize
                    (thn := do brk fieldLoop %[tot']; pure %[])
                    (els := do continueWith fieldLoop %[dp', tot', zero]; pure %[])
                  pure %[])
                (els := do continueWith fieldLoop %[← iaddImm abs 1, tot, cc']; pure %[])
              pure %[])
            (els := do continueWith fieldLoop %[← iaddImm cp 16, tot, cc]; pure %[])
          pure %[])
        (els := do
          let b2 ← uload8_64 (← iadd csvBase cp)
          let _ ← ifte (jTys := []) .eq b2 (← iconst64 44)
            (thn := do
              let cc' ← iaddImm cc 1
              let _ ← ifte (jTys := []) .eq cc' (← iconst64 5)
                (thn := do
                  let d ← wloop2 (← iaddImm cp 1) zero
                    (head := fun dp acc => do
                      let b3 ← uload8_64 (← iadd csvBase dp)
                      return (exitIf .eq b3 (← iconst64 10), %[dp, acc], ()))
                    (body := fun dp acc _ => do
                      let b4 ← uload8_64 (← iadd csvBase dp)
                      return %[← iaddImm dp 1,
                              ← iadd (← imul acc (← iconst64 10))
                                     (← isub b4 (← iconst64 48))])
                  let dp' ← iaddImm (d.head) 1
                  let tot' ← iadd tot (d.snd)
                  let _ ← ifte (jTys := []) .sge dp' fileSize
                    (thn := do brk fieldLoop %[tot']; pure %[])
                    (els := do continueWith fieldLoop %[dp', tot', zero]; pure %[])
                  pure %[])
                (els := do continueWith fieldLoop %[← iaddImm cp 1, tot, cc']; pure %[])
              pure %[])
            (els := do continueWith fieldLoop %[← iaddImm cp 1, tot, cc]; pure %[])
          pure %[])
      return (zero, %[cp, tot, cc]))
  let total := scanned.head

  -- itoa + write
  let ten ← iconst64 10
  let scaled ← wloop1 (← iconst64 1)
    (head := fun div => do
      let d10 ← imul div ten
      return (exitIf .ugt d10 total, %[div], ()))
    (body := fun div _ => return %[← imul div ten])

  let written ← dwloop %[total, scaled.head, ← iconst64 LEFT_VAL]
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
  let _ ← ffi fnWrite %[ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 LEFT_VAL,
                        zero, zero]

set_option maxRecDepth 1000000 in

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the FFI's init calls fill, so naming
    those is what stops an offset being placed where one of those calls will
    overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"input_path",  INPUT_PATH_OFF, OUTPUT_PATH_OFF - INPUT_PATH_OFF⟩,
   ⟨"output_path", OUTPUT_PATH_OFF, LEFT_VAL - OUTPUT_PATH_OFF⟩,
   ⟨"left_val",    LEFT_VAL, CSV_DATA - LEFT_VAL⟩,
   ⟨"csv_data",    CSV_DATA, MAX_CSV_BYTES⟩]


#eval LayoutScan.check "CsvBenchAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "csv_algorithm" {
    functions := clif,
    memory_size := MEM_SIZE
  }]

end CsvBench
