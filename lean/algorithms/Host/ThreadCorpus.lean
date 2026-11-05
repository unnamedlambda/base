module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Host.Threads
meta import AlgorithmLib.Host.Threads
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Host.Program
meta import AlgorithmLib.Host.Program
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# Threads, checked against the JIT

A program that starts four threads on one of its own functions and joins
them, run through `base/tests/hprog_thread_corpus.rs` and compared byte for
byte with what the interpreter computed here. Each thread owns 32 bytes of
the arena: it sums the four words there, writes the sum over the first, and
answers twice the sum. The interpreter runs a thread's function in its view
of those bytes (`Thread.runOut`), the machine `ThreadHonest` names, so the
bytes agreeing check that `base_thread_spawn` and `base_thread_join` do what
`Sem` says they do.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgThreadCorpus

def MEM : Nat := 0x100
/-- Threads, and the bytes each owns. -/
def K : Nat := 4
def C : Nat := 32
def OUT : Nat := 16 * K

/-- `u0:2`: sums the four words at `p`, writes the sum at `p`, and answers
    twice the sum. -/
def worker : StatusBody := do
  let (p ::ᵥ .nil) ← entryParams [.i64]
  let a ← load64 p
  let b ← load64 (← iaddImm p 8)
  let c ← load64 (← iaddImm p 16)
  let d ← load64 (← iaddImm p 24)
  let s ← iadd (← iadd a b) (← iadd c d)
  storeI64 s p
  iadd s s

/-- Word `j` of chunk `k` is `10·k + j + 1`; thread `k` owns chunk `k`. Out:
    for each thread, the word it left at its chunk and what its join answered. -/
def body : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  for k in List.range K do
    for j in List.range 4 do
      storeI64 (← iconst64 (10 * k + j + 1)) (← iaddImm base ((C * k + 8 * j : Nat) : Int))
  let f ← op (.funcAddr 2)
  let hs ← (List.range K).mapM fun k => do
    threadSpawn f (← iaddImm base ((C * k : Nat) : Int)) (← iconst64 0)
  for (h, k) in hs.zip (List.range K) do
    storeI64 (← threadJoin h) (← iaddImm out ((16 * k + 8 : Nat) : Int))
  for k in List.range K do
    storeI64 (← load64 (← iaddImm base ((C * k : Nat) : Int))) (← iaddImm out ((16 * k : Nat) : Int))

def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body),
    Prog.compileProgStatus 2 worker [.i64]]

/-- The program's own function as a term, for `termLocals`. -/
def fns : Nat → Option Fn
  | 2 => let (a, c, e, _) := runAns worker [.i64]
         some { params := [.i64], code := c, status := a, env := e }
  | _ => none

def code : Code := Prog.emit body
def checked : Except String Code := Prog.emitChecked body
def env : FnEnv := (Prog.run body).2.1

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .data),
   .sc .i64 0, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

/-- The span a start of `u0:2` on `a` owns: chunk `k` for `a` its address. -/
def lendOf (i : Nat) (a : UInt64) : Option (Nat × Nat) :=
  if i = 2 && Sem.regionBase .arena ≤ a then
    let off := (a - Sem.regionBase .arena).toNat
    if off % C = 0 && off < C * K then some (off, C) else none
  else none

def startWorld (L : Sem.Locals) : Sem.World :=
  { mem := { arena := ByteArray.mk (Array.replicate MEM 0), data := ByteArray.empty,
             out := ByteArray.mk (Array.replicate OUT 0) },
    lend := lendOf, threadOut := Sem.Thread.runOut L }

def expected : Except String ByteArray :=
  let steps := 100000000
  let L := termLocals fns env steps 4
  match Sem.run { env, steps, locals := L } entryArgVals (startWorld L) code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

end HProgThreadCorpus

open AlgorithmLib in
def Host.ThreadCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgThreadCorpus.checked with
  | .error e => throw (IO.userError s!"the thread corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgThreadCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the thread corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgThreadCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_thread_corpus" {
        functions := clif, required_memory := HProgThreadCorpus.MEM
      }]
      let sideDir := System.FilePath.mk dir / "hprog_thread_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat)))]).compress
      IO.println s!"thread corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.ThreadCorpus" `Host.ThreadCorpus.main
