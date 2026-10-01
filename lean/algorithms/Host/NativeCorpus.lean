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
# The native-code contracts, checked against the runtime

What `Sem.callFfi` says `cl_native_load`, `cl_native_free`, `cl_native_arch`
and `cl_cpu_has` do, run through `base/tests/hprog_native_corpus.rs` on an
x86-64 host and compared byte for byte with what the interpreter computed here,
with the oracles set to that host: architecture 1, and `sse2`, which every
x86-64 CPU has, present. The loaded bytes are never called: that is
`Callee.native`, which the model does not read.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgNativeCorpus

def CODE : Nat := 0x40
def SSE2 : Nat := 0x80
def UNKNOWN : Nat := 0xA0
def MEM : Nat := 0x100
def OUT : Nat := 64

/-- `mov rax, rdi; add rax, rsi; ret`. -/
def add : List UInt8 := [0x48, 0x89, 0xf8, 0x48, 0x01, 0xf0, 0xc3]

def image : List UInt8 :=
  let pad (xs : List UInt8) (n : Nat) := xs ++ List.replicate (n - xs.length) 0
  pad (pad (pad (pad [] CODE ++ add) SSE2 ++ stringToBytes "sse2") UNKNOWN
    ++ stringToBytes "no-such-feature") MEM

def body : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  let code (k : Nat) (r : V .i32) : Prog V L Unit := do
    storeI32 r (← iadd out (← iconst64 (4 * k)))
  let at_ (o : Nat) : Prog V L (V .i64) := do iadd ptr (← iconst64 o)
  -- Nothing to load answers 0; `free` of 0 is refused.
  code 0 (← ffi .nativeFree %[← ffi .nativeLoad %[← iconst64 0, ← iconst64 7]])
  code 1 (← ffi .nativeFree %[← ffi .nativeLoad %[← at_ CODE, ← iconst64 0]])
  let f ← ffi .nativeLoad %[← at_ CODE, ← iconst64 add.length]
  code 2 (← ffi .nativeFree %[f])
  code 3 (← ffi .nativeFree %[f])
  code 4 (← ffi .nativeFree %[← at_ CODE])
  code 5 (← ffi .nativeArch %[])
  code 6 (← ffi .cpuHas %[← at_ SSE2])
  code 7 (← ffi .cpuHas %[← at_ UNKNOWN])
  code 8 (← ffi .cpuHas %[← iconst64 0])

def code : Code := Prog.emit body
def checked : Except String Code := Prog.emitChecked body
def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]
def env : FnEnv := (Prog.run body).2.1

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
   .sc .i64 0, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

def startWorld : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ByteArray.empty,
             out := ByteArray.mk (Array.replicate OUT 0) },
    arch := 1, cpuHas := fun s => if s == "sse2" then some true else none }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

end HProgNativeCorpus

open AlgorithmLib in
def Host.NativeCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgNativeCorpus.checked with
  | .error e => throw (IO.userError s!"the native corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgNativeCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the native corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgNativeCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_native_corpus" {
        functions := clif, required_memory := HProgNativeCorpus.MEM,
        initial_memory := HProgNativeCorpus.image
      }]
      let sideDir := System.FilePath.mk dir / "hprog_native_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat)))]).compress
      IO.println s!"native corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.NativeCorpus" `Host.NativeCorpus.main
