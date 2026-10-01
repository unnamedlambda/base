module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Host.Program
meta import AlgorithmLib.Host.Program
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# Threads, checked against the runtime

What the model says `cl_thread_*` do, run through `base/tests/hprog_thread_corpus.rs`
and compared byte for byte. A worker is one of the program's own functions,
taking one pointer; the model runs it at the spawn and keeps host memory frozen
until the join, so `main` records what it learned only after joining. Refused
spawns --- a null context, a function that answers a value --- answer `-1` on
both sides; a second join of the same handle does too.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgThreadCorpus

def SLOT : Nat := 0
def MEM : Nat := 0x100
def OUT : Nat := 96

/-- `u0:2`: the worker. Writes `77` through its pointer, then reads it back and
    writes `82` beside it. -/
def worker : Body := do
  let (arg ::ᵥ .nil) ← entryParams [.i64]
  storeI64 (← iconst64 77) arg
  storeI64 (← iadd (← load64 arg) (← iconst64 5)) (← iadd arg (← iconst64 8))

/-- `u0:3`: answers a value, so it is no worker. -/
def answers : StatusBody := do
  let (_ ::ᵥ .nil) ← entryParams [.i64]
  iconst64 1

def body : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  let slot ← iadd ptr (← iconst64 SLOT)
  ffiVoid .threadInit %[slot]
  let c ← load64 slot
  let arg ← iadd out (← iconst64 64)
  let h ← ffi .threadSpawn %[c, ← iconst64 2, arg]
  let j ← ffi .threadJoin %[c, h]
  -- memory is the host's again
  storeI64 h out
  storeI64 j (← iadd out (← iconst64 8))
  storeI64 (← ffi .threadJoin %[c, h]) (← iadd out (← iconst64 16))
  storeI64 (← ffi .threadSpawn %[c, ← iconst64 3, arg]) (← iadd out (← iconst64 24))
  storeI64 (← ffi .threadSpawn %[← iconst64 0, ← iconst64 2, arg]) (← iadd out (← iconst64 32))
  ffiVoid .threadCleanup %[slot]

def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body),
    Prog.compileProg 2 worker [.i64], Prog.compileProgStatus 3 answers [.i64]]

def fns : Nat → Option Fn
  | 2 => let (_, c, e, _) := runAns worker [.i64]
         some { params := [.i64], code := c, status := none, env := e }
  | 3 => let (a, c, e, _) := runAns answers [.i64]
         some { params := [.i64], code := c, status := a, env := e }
  | _ => none

def code : Code := Prog.emit body
def checked : Except String Code := Prog.emitChecked body
def env : FnEnv := (Prog.run body).2.1

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
   .sc .i64 0, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

def startWorld : Sem.World :=
  { mem := { arena := ByteArray.mk (Array.replicate MEM 0), data := ByteArray.empty,
             out := ByteArray.mk (Array.replicate OUT 0) } }

def expected : Except String ByteArray :=
  let steps := 100000000
  match Sem.run { env, steps, locals := termLocals fns env steps 4 } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

/-- Touching memory between the spawn and the join is refused. -/
def racy : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  let slot ← iadd ptr (← iconst64 SLOT)
  ffiVoid .threadInit %[slot]
  let c ← load64 slot
  let h ← ffi .threadSpawn %[c, ← iconst64 2, ← iadd out (← iconst64 64)]
  storeI64 h out
  let _ ← ffi .threadJoin %[c, h]

#guard (match Sem.run { env := (Prog.run racy).2.1, locals := termLocals fns env 1000 4 } entryArgVals
    startWorld (Prog.emit racy) with
  | .stuck _ | .fault _ => true
  | _ => false)

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
