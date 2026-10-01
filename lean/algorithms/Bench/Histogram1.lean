module
public import Lean
public import Scan.Layout
meta import Scan.Layout
public import AlgorithmLib.Core.Artifact
meta import AlgorithmLib.Core.Artifact
public import AlgorithmLib.Surface.Layout
meta import AlgorithmLib.Surface.Layout
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

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
def PATH_MAX_IN     : Nat := OUTPUT_PATH_OFF - INPUT_PATH_OFF
def PATH_MAX_OUT    : Nat := HIST_OFF - OUTPUT_PATH_OFF
def BINS            : Nat := 256
def HIST_BYTES      : Nat := BINS * 8
def DATA_OFF        : Nat := HIST_OFF + HIST_BYTES
def MAX_DATA_BYTES  : Nat := 64 * 1024 * 1024
def MEM_SIZE        : Nat := DATA_OFF + MAX_DATA_BYTES

open AlgorithmLib.Prog


abbrev fnRead : Ffi := .fileRead
abbrev fnWrite : Ffi := .fileWrite

def code : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let zero    ← iconst64 0

  -- Copy the input path until NUL, within the data handed over and its
  -- region less one byte, which a NUL ends where the path ran out first; then
  -- the output path, from just past that NUL, the same way.
  let dl ← dataLen
  let inLim ← umin dl (← iconst64 (PATH_MAX_IN - 1))
  let inEnd ← wloop1L zero
    (head := fun _ si => return (contIfULt si inLim, %[si], ()))
    (body := fun l si _ => do
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr INPUT_PATH_OFF) si)
      when .eq ch zero (brk l %[← iaddImm si 1])
      return %[← iaddImm si 1])
  istore8 zero (← iadd (← absAddr ptr INPUT_PATH_OFF) inEnd.head)
  let outSrc ← iadd dataPtr inEnd.head
  let outLim ← umin (← isub dl inEnd.head) (← iconst64 (PATH_MAX_OUT - 1))
  let outEnd ← wloop1L zero
    (head := fun _ di => return (contIfULt di outLim, %[di], ()))
    (body := fun l di _ => do
      let ch ← uload8_64 (← iadd outSrc di)
      istore8 ch (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di)
      when .eq ch zero (brk l %[di])
      return %[← iaddImm di 1])
  istore8 zero (← iadd (← absAddr ptr OUTPUT_PATH_OFF) outEnd.head)

  -- at most the data region: a read of size 0 would take the whole file
  let fileSize ← ffi fnRead
    %[ptr, ← iconst64 INPUT_PATH_OFF, ← iconst64 DATA_OFF, zero, ← iconst64 MAX_DATA_BYTES]
  let n        ← ushrImm fileSize 2    -- n = bytes / 4
  let histPtr  ← absAddr ptr HIST_OFF
  let histEnd  ← iadd histPtr (← iconst64 HIST_BYTES)

  -- Zero the histogram, eight words a trip. The region is a fixed size, so the
  -- loop always runs and needs no guard.
  let _ ← dwloop %[histPtr] .ult histEnd (contOnTrue := true) []
    (body := fun c => do
      let hp := c.head
      store zero hp
      for k in List.range 7 do store zero (← iaddImm hp (8 * (k + 1)))
      let hp' ← iaddImm hp 64
      return (hp', %[hp']))
    (guardIdx := none)

  -- a value past the last bin counts in the bin its low bits name
  let binMask  ← iconst64 (BINS - 1)
  let dataPtr2 ← absAddr ptr DATA_OFF
  let dataEnd  ← iadd dataPtr2 (← ishlImm n 2)
  let n4       ← band n (← iconst64 (-4))
  let dataEnd4 ← iadd dataPtr2 (← ishlImm n4 2)

  -- Four counts a trip while a whole group is left, then one at a time.
  let mid ← ifte .ult dataPtr2 dataEnd4
    (thn := do
      let _ ← wloop1 dataPtr2
        (head := fun dp => return (contIfULt dp dataEnd4, %[], ()))
        (body := fun dp _ => do
          for k in List.range 4 do
            let v ← band (← uload32_64 (← iaddImm dp (4 * k))) binMask
            let a ← iadd histPtr (← ishlImm v 3)
            let c ← load64 a
            store (← iaddImm c 1) a
          return %[← iaddImm dp 16])
      pure %[dataEnd4])
    (els := pure %[dataPtr2])

  let _ ← wloop1 (mid.head)
    (head := fun dp => return (contIfULt dp dataEnd, %[], ()))
    (body := fun dp _ => do
      let v ← band (← uload32_64 dp) binMask
      let a ← iadd histPtr (← ishlImm v 3)
      let c ← load64 a
      store (← iaddImm c 1) a
      return %[← iaddImm dp 4])

  let _ ← ffi fnWrite %[ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 HIST_OFF,
                        zero, ← iconst64 HIST_BYTES]


def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     .ok (noopAt 1),
     Prog.entry "main" (Prog.compileProg 2 code)]

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the FFI's init calls fill, so naming
    those is what stops an offset being placed where one of those calls will
    overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"input_path",  INPUT_PATH_OFF, OUTPUT_PATH_OFF - INPUT_PATH_OFF⟩,
   ⟨"output_path", OUTPUT_PATH_OFF, HIST_OFF - OUTPUT_PATH_OFF⟩,
   ⟨"hist",        HIST_OFF, HIST_BYTES⟩,
   ⟨"data",        DATA_OFF, MAX_DATA_BYTES⟩]


#eval LayoutScan.check "Bench.Histogram1" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "hist1_algorithm" {
    functions := clif,
    required_memory := MEM_SIZE
  }]

end HistogramBench1
