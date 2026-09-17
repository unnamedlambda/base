import Lean
import AlgorithmLib.Gen
import ShipScan

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace ByteScrub

/-!
  # Scrubbing NUL out of a buffer, sixteen bytes at a time

  A sanitizing copy: every NUL byte becomes a space, and every other byte is
  passed through. It is the pass run before handing a buffer to something that
  stops at the first NUL, such as a C string interface or a line-oriented log.

  Each trip loads sixteen bytes, compares all sixteen lanes against a vector of
  NULs, and blends in a vector of spaces where they matched. No branch, no
  table. This is the loop one would write in C with intrinsics, and the emitted
  CLIF is that loop.

  `ByteScrub.blend_scrubs`, in `ByteScrubProof.lean`, compiles `blend` and says
  what the result computes, for every sixteen bytes, under `HProgSem`: the
  executable CLIF semantics this repository checks its artifacts against.
-/

/-- Where the sixteen lanes of `v` equal `nul`, take `space`; elsewhere keep `v`. -/
def blend (v nul space : V .i8x16) : Prog V L (V .i8x16) := do
  bitselect (← icmp .eq v nul) space v

/-- Copy `vectors` vectors of bytes from `src` to `dst`, every NUL made a space. -/
def scrub (vectors : Nat) (src dst : V .i64) : Prog V L Unit := do
  let nul ← splat .i8x16 (← iconst .i8 0)
  let space ← splat .i8x16 (← iconst .i8 32)
  forLoop (← iconst64 vectors) fun i => do
    let off ← ishlImm i 4
    let v ← loadI8x16 (← iadd src off)
    store (← blend v nul space) (← iadd dst off)

/-- **What ships.** 256 vectors, so 4096 bytes, from the caller's input to its
    output buffer. -/
def code : Prog V L Unit := do
  scrub 256 (← dataPtr) (← outPtr)

-- ---------------------------------------------------------------------------

/-- This artifact declares no memory of its own --- it reads through the
    caller's input pointer and writes through the caller's output buffer --- so
    the only bytes it names are the runtime's own header, which the engine
    already sizes the arena to cover. -/
def MEM_SIZE : Nat := 0

/-- `compileProg` folds the term, derives the callee table it needs --- empty,
    since this body calls nothing --- and refuses a body `wf` rejects. -/
def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code)]

def artifacts (clif : List FuncData) : Array Json :=
  #[toJsonArtifact "byte_scrub" {
    functions := clif,
    memory_size := MEM_SIZE
  }]

end ByteScrub

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie ByteScrub.clifIR
  emitArtifacts outDir (ByteScrub.artifacts clif)

#eval ShipScan.check "ByteScrubAlgorithm"
