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
# Local calls, checked against the JIT

What `Sem` says a call to one of the program's own functions does, run through
`base/tests/hprog_local_corpus.rs` and compared byte for byte with what the
interpreter computed here, where each local call runs its callee's term
(`termLocals`). `main` calls a function that answers, twice, and a function
that answers nothing and itself calls the first --- a call two deep. What the
compiled program does with the same calls is `program_run_sound`; this checks
that the term semantics it is stated over is the one Cranelift runs.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgLocalCorpus

def MEM : Nat := 0x100
def OUT : Nat := 64

/-- `u0:2`: `3·x + 1`. -/
def tripleRef : LocalRef [.i64] (some .i64) := { callee := .local 2 }
/-- `u0:3`: stores `v`, `v + 1` and `triple v` from `addr`. -/
def spreadRef : LocalRef [.i64, .i64] none := { callee := .local 3 }

def triple : StatusBody := do
  let (x ::ᵥ .nil) ← entryParams [.i64]
  iadd (← imul x (← iconst64 3)) (← iconst64 1)

def spread : Body := do
  let (addr ::ᵥ v ::ᵥ .nil) ← entryParams [.i64, .i64]
  storeI64 v addr
  storeI64 (← iadd v (← iconst64 1)) (← iadd addr (← iconst64 8))
  storeI64 (← callLocal tripleRef %[v]) (← iadd addr (← iconst64 16))

def body : Prog V L Unit := do
  let out ← outPtr
  storeI64 (← callLocal tripleRef %[← iconst64 5]) out
  storeI64 (← callLocal tripleRef %[← iconst64 (-7)]) (← iadd out (← iconst64 8))
  callLocalVoid spreadRef %[← iadd out (← iconst64 16), ← iconst64 40]

def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body),
    Prog.compileProgStatus 2 triple [.i64], Prog.compileProg 3 spread [.i64, .i64]]

/-- The program's own functions as terms, by index, for `termLocals`. -/
def fns : Nat → Option Fn
  | 2 => let (a, c, e, _) := runAns triple [.i64]
         some { params := [.i64], code := c, status := a, env := e }
  | 3 => let (_, c, e, _) := runAns spread [.i64, .i64]
         some { params := [.i64, .i64], code := c, status := none, env := e }
  | _ => none

/-- Both callees pass the door `program_sound` asks of them. -/
theorem fns_retOk : ∀ i f, fns i = some f → retOk f.params f.code f.status = true := by
  intro i f h
  match i, h with
  | 2, h => cases h; decide
  | 3, h => cases h; decide

/-- The functions the artifact ships at `u0:2` and `u0:3` are the compiled table's. -/
theorem shipped_are_compiled :
    Prog.compileProgStatus 2 triple [.i64] = .ok ((compiledFns fns 2).get rfl) ∧
    Prog.compileProg 3 spread [.i64, .i64] = .ok ((compiledFns fns 3).get rfl) := by
  constructor <;> rfl

/-- **This program's local calls, compiled, answer as their terms do**, at every
    call depth. -/
theorem locals_sound (env : FnEnv) (steps : Nat) :
    ∀ k, (termLocals fns env steps k).le (blockLocals (compiledFns fns) k) :=
  program_sound fns env steps fns_retOk

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

end HProgLocalCorpus

open AlgorithmLib in
def Host.LocalCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgLocalCorpus.checked with
  | .error e => throw (IO.userError s!"the local-call corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgLocalCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the local-call corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgLocalCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_local_corpus" {
        functions := clif, required_memory := HProgLocalCorpus.MEM
      }]
      let sideDir := System.FilePath.mk dir / "hprog_local_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat)))]).compress
      IO.println s!"local-call corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.LocalCorpus" `Host.LocalCorpus.main
