module
public import Lean
public import AlgorithmLib.Gen
meta import AlgorithmLib.Gen
public import Scan.Ship
meta import Scan.Ship
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace ByteCount

/-!
  # Counting several bytes at once, and what can be said about the loop

  Sixteen bytes a trip through `i8x16`: compare every lane against a splatted
  needle, take the lane mask to a scalar, count its bits. One pass per needle,
  one counter written per needle. This is the scanning half of a CSV or line
  reader, and the emitted body is the loop one would write in C with intrinsics.

  ## The interface

  A caller gives two things that have to agree: the table of bytes to look for,
  and the buffer the counts are written into. `countEach` takes its output as
  `Counters V ns.length` --- a run of counters whose extent *is* the length of
  the needle table --- so the two cannot disagree. Add a needle and leave the
  buffer at three slots and the build stops, naming both sides:

      ... has type Counters V SLOTS
      but is expected to have type Counters V needles.length

  No tactic runs and nothing is proven: the extents are indices in a type, and a
  mismatch is ordinary unification failing. `needles.length` is a fact about a
  table written in Lean; `SLOTS` mirrors the `[u64; 3]` the Rust host allocates
  in `applications/bytecount/tests/count.rs`. A `static_assert` over two
  constants reaches this far, and the paper says so.

  ## What the instructions compute

  `ByteCountProof.trip_counts` is the claim a systems language has no place to
  put: that the `icmp`/`vhighBits`/`popcnt` sequence emitted below counts
  exactly the matching bytes, for **every** sixteen bytes and every needle. The
  domain is 2^128 inputs, so no test reaches it, and no `constexpr` or
  `requires` clause can state it --- those range over values a compiler knows,
  and this ranges over values the program will meet.

  That theorem is stated over `HProgSem`, the executable semantics of CLIF this
  repository carries, which is differentially tested against the real
  JIT-compiled artifact by `base/tests/hprog_corpus.rs`. It is a claim about
  the instructions under that semantics, not about the silicon directly, and it
  covers one trip --- the loop repeating it is ordinary accumulation, and the
  `applications/bytecount` test is what exercises the whole artifact.
-/

/-- A run of `n` 64-bit counters, at a base address the caller supplied. -/
structure Counters (V : ClifTy → Type) (n : Nat) where
  base : V .i64

/-- Count each of `ns` in `vectors` vectors' worth of bytes at `data`, writing
    one 64-bit count per needle, in the order given.

    The run is counted in vectors rather than bytes, which makes the loop's
    bound and its step the same number by construction. The output extent is
    not that kind of condition: it is `ns.length`, and the caller's buffer
    either has that many slots or the program is not built. -/
def countEach (ns : List Nat) (vectors : Nat) (data : V .i64)
    (out : Counters V ns.length) : Prog V L Unit := do
  for (needle, idx) in ns.zipIdx do
    let nVec ← splat .i8x16 (← iconst .i8 (Int.ofNat needle))
    let total ← forLoopAcc (← iconst64 vectors) (← iconst64 0) fun i acc => do
      let v ← loadI8x16 (← iadd data (← ishlImm i 4))
      let m ← icmp .eq v nVec
      iadd acc (← uextend64 (← popcnt (← vhighBits m)))
    storeAt out.base (8 * idx) total

/-- The bytes this artifact looks for: comma, newline, space. -/
def needles : List Nat := [44, 10, 32]

/-- How many counters the caller's output buffer holds. -/
def SLOTS : Nat := 3

/-- Vectors examined, so 4096 bytes. -/
def VECTORS : Nat := 256

/-- **What ships.** -/
def code : Prog V L Unit := do
  let data ← dataPtr
  let out : Counters _ SLOTS := ⟨← outPtr⟩
  countEach needles VECTORS data out

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

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "byte_count" {
    functions := clif,
    required_memory := MEM_SIZE
  }]

end ByteCount

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie ByteCount.clifIR
  emitArtifacts outDir (ByteCount.artifacts clif)

#eval ShipScan.check "Demo.ByteCount"
