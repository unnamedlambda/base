module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# The CPU library, checked against the system

`Sem.cpuCall` states what each CPU library call answers. This body asks each
of them, storing only what holds on any Linux or Windows machine: at least
one CPU; a CPU number below zero or at the count names none, so its core and
package answer `-1` and pinning to it is declined; the first CPU has a core
and a package; pinning the thread to it and releasing it are granted.
`base/tests/hprog_cpu_corpus.rs` runs it and compares.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgCpuCorpus

def N : Nat := 11
def OUT : Nat := 4 * N
def MEM : Nat := 0x40

def cpu (f : CpuFn) (args : Vals V (Ext.cpu f).sig.1) : Prog V L (ResV V (Ext.cpu f).sig.2) :=
  ext (.cpu f) args

def body : Prog V L Unit := do
  let out ← outPtr
  let put (k : Nat) (v : V .i32) : Prog V L Unit := do storeI32 v (← iadd out (← iconst64 (4 * k)))
  let flag (k : Nat) (c : V .i8) : Prog V L Unit := do put k (← ireduce32 (← uextend64 c))
  let z ← iconst32 0
  let n ← cpu .count %[]
  flag 0 (← icmp .sgt n z)
  let none_ ← iconst32 (-1)
  put 1 (← cpu .core %[none_])
  put 2 (← cpu .core %[n])
  put 3 (← cpu .package %[n])
  put 4 (← cpu .pin %[n])
  put 5 (← cpu .pin %[none_])
  flag 6 (← icmp .sge (← cpu .core %[z]) z)
  flag 7 (← icmp .sge (← cpu .package %[z]) z)
  put 8 (← cpu .pin %[z])
  put 9 (← cpu .unpin %[])
  put 10 (← iconst32 1)

def codeNames : List String :=
  [ "some CPU online", "core of CPU -1", "core of the CPU at the count", "package of the CPU at the count",
    "pinned to the CPU at the count", "pinned to CPU -1", "CPU 0 has a core", "CPU 0 has a package",
    "pinned to CPU 0", "released", "finished" ]

def checked : Except String Code := Prog.emitChecked body
def code : Code := Prog.emit body
def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]
def env : FnEnv := (Prog.run body).2.1

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
   .sc .i64 8, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

def startWorld : Sem.World :=
  { mem := { arena := ByteArray.mk (Array.replicate MEM 0), data := ByteArray.mk (Array.replicate 8 0),
             out := ByteArray.mk (Array.replicate OUT 0) } }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

end HProgCpuCorpus

open AlgorithmLib in
def Host.CpuCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgCpuCorpus.checked with
  | .error e => throw (IO.userError s!"the CPU corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgCpuCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the CPU corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgCpuCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_cpu_corpus" {
        functions := clif, required_memory := HProgCpuCorpus.MEM,
        initial_memory := List.replicate HProgCpuCorpus.MEM 0 }]
      let sideDir := System.FilePath.mk dir / "hprog_cpu_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat))),
                          ("names", Lean.toJson HProgCpuCorpus.codeNames)]).compress
      IO.println s!"CPU corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.CpuCorpus" `Host.CpuCorpus.main
