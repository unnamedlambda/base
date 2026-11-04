module
public import Lean
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace ParSum

/-!
  # Summing on threads the program starts

  The caller writes 4096 bytes into the arena (`base_driver_arena`) and calls
  `main`, which starts four threads on its own function `u0:2`, each owning
  one 1024-byte quarter, and joins them. Each thread answers the wrapping sum
  of its quarter's 128 little-endian words, and `main` writes the sum of the
  four answers to the first eight bytes of its output. While a thread runs
  only it may read or write its quarter (`Mem.lent`), and it sees nothing
  else (`Thread.view`). `Ship.Threads` proves that `main` and `u0:2` never
  misuse a call and never fault, and `Ship.ParSumSpec` that what `main` writes
  is the wrapping sum of the arena's 512 words (`main_computes`).
-/

/-- Threads, and the bytes each sums. -/
def WORKERS : Nat := 4
def CHUNK : Nat := 1024
/-- The arena, which holds the input. -/
def MEM : Nat := WORKERS * CHUNK

/-- `u0:2`: the wrapping sum of the 128 words at `p`. -/
def worker : StatusBody := do
  let (p ::ᵥ .nil) ← entryParams [.i64]
  forLoopAcc (← iconst64 (CHUNK / 8)) (← iconst64 0) fun i acc => do
    iadd acc (← load64 (← iadd p (← ishlImm i 3)))

/-- **What ships**: when the caller has room for the sum; nothing
    otherwise. -/
def code : Prog V L Unit := do
  let out ← outPtr
  let base ← basePtr
  when .ule (← iconst64 8) (← outLen) do
    let f ← op (.funcAddr 2)
    let hs ← (List.range WORKERS).mapM fun k => do
      threadSpawn f (← iaddImm base ((CHUNK * k : Nat) : Int)) (← iconst64 0)
    let mut total ← iconst64 0
    for h in hs do
      total ← iadd total (← threadJoin h)
    storeI64 total out

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 code),
    Prog.compileProgStatus 2 worker [.i64]]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "par_sum" { functions := clif, required_memory := MEM }]

end ParSum

def Demo.ParSum.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie ParSum.clifIR
  emitArtifacts outDir (ParSum.artifacts clif)

#eval ShipScan.check "Demo.ParSum" `Demo.ParSum.main
