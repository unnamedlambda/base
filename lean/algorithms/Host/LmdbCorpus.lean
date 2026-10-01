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
# The LMDB contracts, checked against the library

What `Sem.callFfi` says the seven LMDB entry points do, run through
`base/tests/hprog_lmdb_corpus.rs` against real LMDB directories and compared
byte for byte with what the interpreter computed here.

The cases are there for what a reader of the contract could get wrong: puts
inside a write transaction are what a scan through the same handle sees before
the commit; beginning a second transaction discards the first; keys come back
in LMDB's order, where a key sorts before every longer key it begins; a scan
from a start key begins at the first key not before it; and what was committed
is still there after the context is cleaned up and a new one opens the same
directory.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgLmdbCorpus

def SLOT : Nat := 0
def STR : Nat := 0x100
def MEM : Nat := 0x400
def OUT : Nat := 320

/-- The directories. The test removes them before and after the run. -/
def dirA : String := "/tmp/base-hprog-lmdb-corpus/a"
def dirB : String := "/tmp/base-hprog-lmdb-corpus/b"

/-- Every string the body passes, NUL-terminated, laid out from `STR`. -/
def strings : List String := [dirA, dirB, "", "a", "b", "c", "ab", "x", "1", "2", "3", "one", "y"]

def offsets : List Nat :=
  (strings.foldl (fun (acc : List Nat × Nat) s => (acc.1 ++ [acc.2], acc.2 + s.utf8ByteSize + 1))
    ([], STR)).1

def at_ (s : String) : Nat := (offsets[strings.idxOf s]?).getD 0

def image : List UInt8 :=
  let bytes := strings.foldl (fun acc s => acc ++ stringToBytes s) []
  let a := List.replicate STR 0 ++ bytes
  a ++ List.replicate (MEM - a.length) 0

def body : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  let code (k : Nat) (r : V .i32) : Prog V L Unit := do
    storeI32 r (← iadd out (← iconst64 (4 * k)))
  let str (s : String) : Prog V L (V .i64) := do iadd ptr (← iconst64 (at_ s))
  let len (s : String) : Prog V L (V .i32) := iconst32 s.utf8ByteSize
  let slot ← iadd ptr (← iconst64 SLOT)
  ffiVoid .lmdbInit %[slot]
  let c ← load64 slot
  let zero ← iconst64 0
  let mb ← iconst32 16
  let hA ← ffi .lmdbOpen %[c, ← str dirA, mb]
  code 0 hA
  let hB ← ffi .lmdbOpen %[c, ← str dirB, mb]
  code 1 hB
  code 2 (← ffi .lmdbOpen %[zero, ← str dirA, mb])
  code 3 (← ffi .lmdbOpen %[c, ← str "", mb])
  let put (h : V .i32) (k v : String) : Prog V L (V .i32) := do
    ffi .lmdbPut %[c, h, ← str k, ← len k, ← str v, ← len v]
  -- Outside a transaction each put commits on its own.
  code 4 (← put hA "b" "2")
  code 5 (← put hA "a" "1")
  code 6 (← put hA "" "1")
  code 7 (← ffi .lmdbPut %[c, hA, ← str "a", ← iconst32 (-1), ← str "1", ← len "1"])
  code 8 (← put (← iconst32 7) "a" "1")
  -- Inside one, a scan through the handle sees what it holds.
  code 9 (← ffi .lmdbBeginWriteTxn %[c, hA])
  code 10 (← put hA "c" "3")
  code 11 (← put hA "a" "one")
  code 12 (← put hA "ab" "x")
  -- Each scan has the room up to the next one's result.
  let scan (ctx : V .i64) (h : V .i32) (start : Option String) (max : Int) (dst room : Nat) :
      Prog V L (V .i32) := do
    let (kp, kl) ← match start with
      | some s => do pure (← str s, ← len s)
      | none => do pure (ptr, ← iconst32 0)
    ffi .lmdbCursorScan %[ctx, h, kp, kl, ← iconst32 max, ← iadd out (← iconst64 dst), ← iconst64 room]
  code 13 (← scan c hA none 10 128 64)
  code 14 (← ffi .lmdbCommitWriteTxn %[c, hA])
  code 15 (← ffi .lmdbCommitWriteTxn %[c, hA])
  -- A second begin discards the first transaction.
  code 16 (← ffi .lmdbBeginWriteTxn %[c, hB])
  code 17 (← put hB "x" "y")
  code 18 (← ffi .lmdbBeginWriteTxn %[c, hB])
  code 19 (← ffi .lmdbCommitWriteTxn %[c, hB])
  code 20 (← scan c hB none 10 192 8)
  code 21 (← scan c hA (some "b") 10 200 24)
  code 22 (← scan c hA none 1 224 16)
  -- A handle that names nothing answers an empty scan; a null context writes nothing.
  storeI32 (← iconst32 (-1)) (← iadd out (← iconst64 240))
  code 23 (← scan c (← iconst32 9) none 10 240 8)
  storeI32 (← iconst32 0x2AAAAAAA) (← iadd out (← iconst64 248))
  code 24 (← scan zero hA none 10 248 8)
  ffiVoid .lmdbCleanup %[slot]
  -- What was committed outlives the context.
  ffiVoid .lmdbInit %[slot]
  let c ← load64 slot
  let hA ← ffi .lmdbOpen %[c, ← str dirA, mb]
  code 25 hA
  code 26 (← scan c hA none (-1) 256 64)
  ffiVoid .lmdbCleanup %[slot]

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
             out := ByteArray.mk (Array.replicate OUT 0) } }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

end HProgLmdbCorpus

open AlgorithmLib in
def Host.LmdbCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgLmdbCorpus.checked with
  | .error e => throw (IO.userError s!"the LMDB corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgLmdbCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the LMDB corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgLmdbCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_lmdb_corpus" {
        functions := clif, required_memory := HProgLmdbCorpus.MEM,
        initial_memory := HProgLmdbCorpus.image
      }]
      let sideDir := System.FilePath.mk dir / "hprog_lmdb_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat)))]).compress
      IO.println s!"LMDB corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.LmdbCorpus" `Host.LmdbCorpus.main
