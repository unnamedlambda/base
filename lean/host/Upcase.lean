import AlgorithmLib.Gen
import ShipScan

/-!
# An artifact that reads a file, changes it, and writes it

Three effects and a loop, and all four of them are in the artifact: the read,
the byte-by-byte transform, the write. Nothing here runs. `setup` and
`algorithm` are values, and what performs them is whichever host is handed
them — in-process from Lean by `UpcaseHost`, a Rust `build.rs`, or Python.

This is the smallest program that makes the point, which is why it is a
transform and not a copy: the loop over the file's bytes is compiled into the
same CLIF function as the two file calls, so a host never sees the middle of the
work.
-/

open AlgorithmLib
open AlgorithmLib.Layout
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

namespace Upcase

/-- Bigger than the demo needs, and small enough that the runtime's memory is
still a rounding error. -/
def maxFileSize : Nat := 1 <<< 20

structure Fields where
  /-- The context-pointer slots and the `IoOffsets` region, which every program
  reserves whether or not it uses them. -/
  reserved       : Fld (.bytes 64)
  inputFilename  : Fld (.bytes 256)
  outputFilename : Fld (.bytes 256)
  fileData       : Fld (.bytes maxFileSize)

def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved       ← field (.bytes 64)
  let inputFilename  ← field (.bytes 256)
  let outputFilename ← field (.bytes 256)
  let fileData       ← field (.bytes maxFileSize)
  pure { reserved, inputFilename, outputFilename, fileData }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

def fnRead : FnRef := IR.Ffi.fileRead.ref
def fnWrite : FnRef := IR.Ffi.fileWrite.ref
def env : FnEnv := env% [.fileIO]

/-- Read `inputFilename`, upper-case its ASCII letters in place, write
`outputFilename`.

The transform is branch-free on purpose: `b - 'a'` compared *unsigned* against
26 is both bounds at once, because a byte below `'a'` wraps to something far
above 26. So the loop body is a load, three integer operations, a select and a
store, with no control flow of its own. -/
def mainCode : HProg.Code :=
  clif%(env, HProg.ptrParams) do
  let ptr := basePtr

  let size ← fldReadFile ptr fnRead f.inputFilename f.fileData
  let dataAddr ← fldAddr ptr f.fileData

  let lowerA ← iconst64 97
  let letters ← iconst64 26
  let toUpper ← iconst64 32

  forLoop size fun i => do
    let addr ← iadd dataAddr i
    let b ← uload8_64 addr
    let isLower ← icmp .ult (← isub b lowerA) letters
    let upper ← select isLower (← isub b toUpper) b
    istore8 upper addr

  let _ ← fldWriteFile0 ptr fnWrite f.outputFilename f.fileData size

def clifIrSource : Program :=
  IR.program [IR.noopFunction, HProg.compileFn 1 mainCode env]

/-- The filenames the program reads from memory, laid into the region the
layout reserved for them. -/
def payloads : List UInt8 :=
  mkPayload f.fileData.offset [
    f.inputFilename.init (stringToBytes "input.txt"),
    f.outputFilename.init (stringToBytes "output.txt")
  ]

def setup : Setup := {
  clif := clifIrSource,
  memory_size := layoutMeta.totalSize,
  initial_memory := payloads
}

def algorithm : Algorithm := { fn_idx := IR.mainFnIdx }

end Upcase

-- The same check every shipped artifact in this repository is held to: walk
-- from `setup` and fail the build if any body it carries reached the runtime
-- through a door other than `HProg.compileFn`.
#eval ShipScan.check "Upcase" (root := `Upcase.setup)
