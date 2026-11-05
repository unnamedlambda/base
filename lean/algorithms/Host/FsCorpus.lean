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
# The file library, checked against the system

`Sem.fsCall` states what each file library call does. This body is handed four
paths in a directory of its own, `x`, `x/f`, `x/g` and `x/h`, and makes the
first a directory; opens a file the ways that are refused, and the way that
creates it; writes it with a gap, reads it back, cuts it, syncs it; finds a
read-only handle refusing writes; renames the file, removes it while a handle
is open on it and reads through that handle; lists the directory; and removes
the directory once it is empty. It also takes sixteen bytes of randomness,
reads the wall clock, and writes nothing to standard error. Every answer is
stored; `base/tests/hprog_fs_corpus.rs` hands it a temporary directory's paths
and compares.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgFsCorpus

def A_HELLO : Nat := 0x00
def A_XY : Nat := 0x10
def A_BUF : Nat := 0x40
def A_NAME : Nat := 0x80
def MEM : Nat := 0x100

/-- Each path's field, in the data handed over. -/
def FIELD : Nat := 256

def N : Nat := 39
def OUT : Nat := 4 * N

def image : List UInt8 :=
  let pad (xs : List UInt8) (n : Nat) := xs ++ List.replicate (n - xs.length) 0
  pad (pad ("hello world".toUTF8.toList) A_XY ++ "XY".toUTF8.toList) MEM

def fs (f : FsFn) (args : Vals V (Ext.fs f).sig.1) : Prog V L (ResV V (Ext.fs f).sig.2) :=
  ext (.fs f) args

set_option maxRecDepth 4096 in
def body : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  let d ← dataPtr
  let z ← iconst64 0
  let cap ← iconst64 FIELD
  let path (k : Nat) : Prog V L (V .i64) := iaddImm d (k * FIELD)
  let at_ (k : Nat) : Prog V L (V .i64) := do iadd out (← iconst64 (4 * k))
  let put (k : Nat) (v : V .i32) : Prog V L Unit := do storeI32 v (← at_ k)
  let put64 (k : Nat) (v : V .i64) : Prog V L Unit := do put k (← ireduce32 v)
  let nz (k : Nat) (v : V .i64) : Prog V L Unit := do put k (← ireduce32 (← uextend64 (← icmp .ne v z)))
  let mode (m : Int) : Prog V L (V .i32) := iconst32 m
  put64 0 (← fs .createDir %[← path 0, cap])
  put64 1 (← fs .createDir %[← path 0, cap])
  nz 2 (← fs .open %[← path 0, cap, ← mode 1])
  nz 3 (← fs .open %[← path 1, cap, ← mode 5])
  let f ← fs .open %[← path 1, cap, ← mode 7]
  nz 4 f
  put64 5 (← fs .write %[f, ← iaddImm base A_HELLO, ← iconst64 11, z])
  put64 6 (← fs .write %[f, ← iaddImm base A_XY, ← iconst64 2, ← iconst64 20])
  put64 7 (← fs .size %[f])
  let buf ← iaddImm base A_BUF
  put64 8 (← fs .read %[f, buf, ← iconst64 64, z])
  storeI32 (← load32 buf) (← at_ 9)
  storeI32 (← load32 (← iaddImm buf 12)) (← at_ 10)
  storeI32 (← load32 (← iaddImm buf 20)) (← at_ 11)
  put64 12 (← fs .read %[f, buf, ← iconst64 4, ← iconst64 100])
  put64 13 (← fs .setLen %[f, ← iconst64 5])
  put64 14 (← fs .size %[f])
  put64 15 (← fs .sync %[f])
  put64 16 (← fs .read %[f, buf, ← iconst64 (-1), z])
  fs .close %[f]
  let r ← fs .open %[← path 1, cap, ← mode 1]
  put64 17 (← fs .write %[r, buf, ← iconst64 1, z])
  put64 18 (← fs .sync %[r])
  put64 19 (← fs .setLen %[r, z])
  fs .close %[r]
  put64 20 (← fs .rename %[← path 1, cap, ← path 2, cap])
  nz 21 (← fs .open %[← path 1, cap, ← mode 1])
  let g ← fs .open %[← path 2, cap, ← mode 3]
  nz 22 g
  put64 23 (← fs .removeFile %[← path 2, cap])
  put64 24 (← fs .read %[g, buf, ← iconst64 16, z])
  fs .close %[g]
  put64 25 (← fs .removeFile %[← path 2, cap])
  let h ← fs .open %[← path 3, cap, ← mode 14]
  fs .close %[h]
  let dd ← fs .dirOpen %[← path 0, cap]
  nz 26 dd
  let name ← iaddImm base A_NAME
  put64 27 (← fs .dirNext %[dd, name, ← iconst64 16])
  storeI32 (← load32 name) (← at_ 28)
  put64 29 (← fs .dirNext %[dd, name, ← iconst64 16])
  fs .dirClose %[dd]
  put64 30 (← fs .removeDir %[← path 0, cap])
  put64 31 (← fs .removeFile %[← path 3, cap])
  put64 32 (← fs .removeDir %[← path 0, cap])
  nz 33 (← fs .dirOpen %[← path 0, cap])
  put64 34 (← ext (.base .random) %[buf, ← iconst64 16])
  put64 35 (← ext (.base .random) %[buf, ← iconst64 (-1)])
  let wall ← ext (.base .clockWall) %[]
  put 36 (← ireduce32 (← uextend64 (← icmp .sgt wall (← iconst64 1600000000000000000))))
  put64 37 (← ext (.base .stderrWrite) %[buf, z])
  put 38 (← iconst32 1)

def codeNames : List String :=
  [ "made the directory", "made it again", "opened the directory as a file",
    "opened to create without write", "opened to create", "wrote hello world", "wrote past a gap",
    "its size", "read it back", "its first bytes", "the gap", "the bytes past the gap",
    "read past the end", "cut it", "its size after", "synced it", "read a negative length",
    "wrote through a read-only handle", "synced a read-only handle", "cut a read-only handle",
    "renamed it", "opened the old name", "opened the new name", "removed it while open",
    "read through the open handle", "removed it again", "listed the directory", "the entry's name length",
    "the entry's name", "listed past the end", "removed the directory while not empty",
    "removed its file", "removed the empty directory", "listed the removed directory",
    "took sixteen bytes of randomness", "took a negative length", "the wall clock is past 2020",
    "wrote nothing to standard error", "finished" ]

def checked : Except String Code := Prog.emitChecked body
def code : Code := Prog.emit body
def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]
def env : FnEnv := (Prog.run body).2.1

/-- The directory is the test's; any path stands for it here. -/
def DIR : String := "/corpus"

def paths : List String := ["x", "x/f", "x/g", "x/h"].map (DIR ++ "/" ++ ·)

def data : List UInt8 :=
  paths.flatMap fun p => p.toUTF8.toList ++ List.replicate (FIELD - p.utf8ByteSize) 0

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .data), .sc .i64 data.length.toUInt64,
   .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

def startWorld : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ⟨data.toArray⟩, out := ByteArray.mk (Array.replicate OUT 0) },
    fs := { dirs := [DIR] },
    wall := 1700000000000000000 }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

end HProgFsCorpus

open AlgorithmLib in
def Host.FsCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgFsCorpus.checked with
  | .error e => throw (IO.userError s!"the file corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgFsCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the file corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgFsCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_fs_corpus" {
        functions := clif, required_memory := HProgFsCorpus.MEM,
        initial_memory := HProgFsCorpus.image }]
      let sideDir := System.FilePath.mk dir / "hprog_fs_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat))),
                          ("names", Lean.toJson HProgFsCorpus.codeNames),
                          ("field", Lean.toJson HProgFsCorpus.FIELD)]).compress
      IO.println s!"file corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.FsCorpus" `Host.FsCorpus.main
