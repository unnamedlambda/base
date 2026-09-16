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
open AlgorithmLib.Prog

namespace Upcase

/-- Bigger than the demo needs, and small enough that the runtime's memory is
still a rounding error. -/
def maxFileSize : Nat := 1 <<< 20

structure Fields where
  /-- The context-pointer slots and the unused gap after them, which every
  program reserves whether or not it uses them. -/
  reserved       : Fld (.bytes 64)
  /-- How many bytes the program read, and therefore transformed and wrote.

  The program knows this the moment `fileRead` returns, and a host cannot
  know it at all -- the file is read inside the artifact. Storing it is what
  lets `UpcaseHost` read the result back without being told its length.

  The entry also *answers* this number, as its status. Two routes to one
  count, on purpose: a scalar a host wants immediately needs no address, and
  anything wider than a scalar still does. -/
  size           : Fld .i64
  inputFilename  : Fld (.bytes 256)
  outputFilename : Fld (.bytes 256)
  fileData       : Fld (.bytes maxFileSize)

def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved       ← field (.bytes 64)
  let size           ← field .i64
  let inputFilename  ← field (.bytes 256)
  let outputFilename ← field (.bytes 256)
  let fileData       ← field (.bytes maxFileSize)
  pure { reserved, size, inputFilename, outputFilename, fileData }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

abbrev fnRead : Ffi := .fileRead
abbrev fnWrite : Ffi := .fileWrite

/-- Read `inputFilename`, upper-case its ASCII letters in place, write
`outputFilename`, and record how many bytes that was.

The transform is branch-free on purpose: `b - 'a'` compared *unsigned* against
26 is both bounds at once, because a byte below `'a'` wraps to something far
above 26. So the loop body is a load, three integer operations, a select and a
store, with no control flow of its own. -/
def mainCode : Prog V L (V .i64) := do
  let ptr ← basePtr

  let size ← fldReadFile ptr f.inputFilename f.fileData
  fldStore ptr f.size size
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

  let _ ← fldWriteFile0 ptr f.outputFilename f.fileData size
  pure size

def clifIrSource : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.compileProgStatus 1 mainCode]

/-- The filenames the program reads from memory, laid into the region the
layout reserved for them. -/
def payloads : List UInt8 :=
  mkPayload f.fileData.offset [
    f.inputFilename.init (stringToBytes "input.txt"),
    f.outputFilename.init (stringToBytes "output.txt")
  ]

def setup (clif : List FuncData) : Artifact := {
  functions := clif,
  memory_size := layoutMeta.totalSize,
  initial_memory := payloads
}

def algorithm : UInt32 := IR.mainFnIdx

/-- What a host runs: the artifact, once the body it carries has been checked. -/
def shipped : Except String Artifact := do return setup (← clifIrSource)

end Upcase

-- The same check every shipped artifact in this repository is held to: walk
-- from the program and fail the build if any body it carries reached the
-- runtime through a door other than the checked one.
#eval ShipScan.check "Upcase" (root := `Upcase.shipped)
