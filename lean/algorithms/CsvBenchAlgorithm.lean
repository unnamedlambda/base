import Lean
import AlgorithmLib.Gen

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

set_option maxRecDepth 4096 in
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
  let zero32  ← iconst32 0

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
    [ptr, ← iconst64 INPUT_PATH_OFF, ← iconst64 CSV_DATA, zero, zero]
  let csvBase  ← absAddr ptr CSV_DATA
  let nlVec    ← splat .i8x16 (← iconst .i8 10)
  let cmVec    ← splat .i8x16 (← iconst .i8 44)

  -- Skip the header row. Dispatches between a SIMD chunk and a scalar byte;
  -- every path either leaves with the newline's position or takes the edge, so
  -- the loop needs no test of its own.
  let hdr ← dwloop [zero] .eq zero (contOnTrue := true) [0]
    (body := fun c => do
      let hp := c.headD 0
      let _ ← ifte .sle (← iaddImm hp 16) fileSize
        (thn := do
          let row  ← loadI8x16 (← iadd csvBase hp)
          let eq   ← icmp .eq row nlVec
          let mask ← vhighBits eq
          let _ ← ifte .ne mask zero32
            (thn := do
              let off ← uextend64 (← ctz mask)
              brk [← iadd hp off]
              pure [])
            (els := do continueWith [← iaddImm hp 16]; pure [])
          pure [])
        (els := do
          let b ← uload8_64 (← iadd csvBase hp)
          let _ ← ifte .eq b (← iconst64 10)
            (thn := do brk [hp]; pure [])
            (els := do continueWith [← iaddImm hp 1]; pure [])
          pure [])
      return (zero, [hp]))

  -- Sum the sixth field of every row. Same shape: a dispatch with several back
  -- edges and one way out, when the scan passes the end of the file.
  let scanned ← dwloop [← iaddImm (hdr.headD 0) 1, zero, zero]
      .eq zero (contOnTrue := true) [1]
    (body := fun c => do
      let cp := c.headD 0; let tot := c.getD 1 0; let cc := c.getD 2 0
      let _ ← ifte .sle (← iaddImm cp 16) fileSize
        (thn := do
          let row2 ← loadI8x16 (← iadd csvBase cp)
          let eq2  ← icmp .eq row2 cmVec
          let msk2 ← vhighBits eq2
          let _ ← ifte .ne msk2 zero32
            (thn := do
              let off2 ← uextend64 (← ctz msk2)
              let abs  ← iadd cp off2
              let cc'  ← iaddImm cc 1
              let _ ← ifte .eq cc' (← iconst64 5)
                (thn := do
                  let d ← wloop2 (← iaddImm abs 1) zero
                    (head := fun dp acc => do
                      let b3 ← uload8_64 (← iadd csvBase dp)
                      return (exitIf .eq b3 (← iconst64 10), [dp, acc], ()))
                    (body := fun dp acc _ => do
                      let b4 ← uload8_64 (← iadd csvBase dp)
                      return [← iaddImm dp 1,
                              ← iadd (← imul acc (← iconst64 10))
                                     (← isub b4 (← iconst64 48))])
                  let dp' ← iaddImm (d.headD 0) 1
                  let tot' ← iadd tot (d.getD 1 0)
                  let _ ← ifte .sge dp' fileSize
                    (thn := do brk [tot']; pure [])
                    (els := do continueWith [dp', tot', zero]; pure [])
                  pure [])
                (els := do continueWith [← iaddImm abs 1, tot, cc']; pure [])
              pure [])
            (els := do continueWith [← iaddImm cp 16, tot, cc]; pure [])
          pure [])
        (els := do
          let b2 ← uload8_64 (← iadd csvBase cp)
          let _ ← ifte .eq b2 (← iconst64 44)
            (thn := do
              let cc' ← iaddImm cc 1
              let _ ← ifte .eq cc' (← iconst64 5)
                (thn := do
                  let d ← wloop2 (← iaddImm cp 1) zero
                    (head := fun dp acc => do
                      let b3 ← uload8_64 (← iadd csvBase dp)
                      return (exitIf .eq b3 (← iconst64 10), [dp, acc], ()))
                    (body := fun dp acc _ => do
                      let b4 ← uload8_64 (← iadd csvBase dp)
                      return [← iaddImm dp 1,
                              ← iadd (← imul acc (← iconst64 10))
                                     (← isub b4 (← iconst64 48))])
                  let dp' ← iaddImm (d.headD 0) 1
                  let tot' ← iadd tot (d.getD 1 0)
                  let _ ← ifte .sge dp' fileSize
                    (thn := do brk [tot']; pure [])
                    (els := do continueWith [dp', tot', zero]; pure [])
                  pure [])
                (els := do continueWith [← iaddImm cp 1, tot, cc']; pure [])
              pure [])
            (els := do continueWith [← iaddImm cp 1, tot, cc]; pure [])
          pure [])
      return (zero, [cp, tot, cc]))
  let total := scanned.headD 0

  -- itoa + write
  let ten ← iconst64 10
  let scaled ← wloop1 (← iconst64 1)
    (head := fun div => do
      let d10 ← imul div ten
      return (exitIf .ugt d10 total, [div], ()))
    (body := fun div _ => return [← imul div ten])

  let written ← dwloop [total, scaled.headD 0, ← iconst64 LEFT_VAL]
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
  let _ ← call fnWrite [ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 LEFT_VAL,
                        zero, zero]

set_option maxRecDepth 1000000 in
theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  IR.program [noopFunction, HProg.compileFn 1 code env (hwf := code_wf)]

def artifacts : Array Json :=
  #[toJsonEntry "csv_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end CsvBench
