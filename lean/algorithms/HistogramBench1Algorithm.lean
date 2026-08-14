import Lean
import AlgorithmLib.Gen

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

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

open AlgorithmLib.HProg.Sur


/-- `cl_file_read` as fn0, `cl_file_write` as fn1. -/
def env : FnEnv := env% [.fileIO]

def fnRead : Nat := IR.FFI.std.fileRead.id
def fnWrite : Nat := IR.FFI.std.fileWrite.id

def code : HProg.Code := clif% do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let zero    ← iconst64 0

  -- Copy the input path until NUL. Entered unconditionally, and the test reads
  -- the byte the body just loaded, so there is no guard to read it at.
  let inEnd ← dwloop [zero] .eq zero (contOnTrue := false) [0]
    (body := fun c => do
      let si := c.headD 0
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr INPUT_PATH_OFF) si)
      let si' ← iaddImm si 1
      return (ch, [si']))
    (guardIdx := none)

  -- And the output path, from where that stopped.
  let _ ← dwloop [inEnd.headD 0, zero] .eq zero (contOnTrue := false) []
    (body := fun c => do
      let si := c.headD 0; let di := c.getD 1 0
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di)
      let si' ← iaddImm si 1
      let di' ← iaddImm di 1
      return (ch, [si', di']))
    (guardIdx := none)

  let fileSize ← call fnRead
    [ptr, ← iconst64 INPUT_PATH_OFF, ← iconst64 DATA_OFF, zero, zero]
  let n        ← ushrImm fileSize 2    -- n = bytes / 4
  let histPtr  ← absAddr ptr HIST_OFF
  let histEnd  ← iadd histPtr (← iconst64 HIST_BYTES)

  -- Zero the histogram, eight words a trip. The region is a fixed size, so the
  -- loop always runs and needs no guard.
  let _ ← dwloop [histPtr] .ult histEnd (contOnTrue := true) []
    (body := fun c => do
      let hp := c.headD 0
      store zero hp
      for k in [1:8] do store zero (← iaddImm hp (8 * k))
      let hp' ← iaddImm hp 64
      return (hp', [hp']))
    (guardIdx := none)

  let dataPtr2 ← absAddr ptr DATA_OFF
  let dataEnd  ← iadd dataPtr2 (← ishlImm n 2)
  let n4       ← band n (← iconst64 (-4))
  let dataEnd4 ← iadd dataPtr2 (← ishlImm n4 2)

  -- Four counts a trip while a whole group is left, then one at a time.
  let mid ← ifte .ult dataPtr2 dataEnd4
    (thn := do
      let _ ← wloop1 dataPtr2
        (head := fun dp => return (contIfULt dp dataEnd4, ([] : List R), ()))
        (body := fun dp _ => do
          for k in [0:4] do
            let v ← uload32_64 (← iaddImm dp (4 * k))
            let a ← iadd histPtr (← ishlImm v 3)
            let c ← load64 a
            store (← iaddImm c 1) a
          return [← iaddImm dp 16])
      pure [dataEnd4])
    (els := pure [dataPtr2])

  let _ ← wloop1 (mid.headD 0)
    (head := fun dp => return (contIfULt dp dataEnd, ([] : List R), ()))
    (body := fun dp _ => do
      let v ← uload32_64 dp
      let a ← iadd histPtr (← ishlImm v 3)
      let c ← load64 a
      store (← iaddImm c 1) a
      return [← iaddImm dp 4])

  let _ ← call fnWrite [ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 HIST_OFF,
                        zero, ← iconst64 HIST_BYTES]

theorem code_wf : HProg.wf env HProg.ptrParams code = true := by decide

def clifIR : Program :=
  program
    [noopFunction,
     noopAt 1,
     HProg.compileFn 2 code]

def artifacts : Array Json :=
  #[toJsonEntry "hist1_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 2
  }]

end HistogramBench1
