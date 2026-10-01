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
# The serial library, checked against a terminal

`Sem.serialCall` states what each serial library call does. This body opens
the port whose path it is handed, which `base/tests/hprog_serial_corpus.rs`
makes a pseudo-terminal that has already sent `hello`; it asks how much is
waiting, reads it in two pieces and once more with nothing left, writes
`ping`, and takes the refusals — a baud rate of `0`, a negative timeout, a
second open of a port held, negative lengths, a port past the listed ones.
Every answer is stored, and the two pieces read; the test compares them,
and what the terminal received with what the model says was sent. The
ports the system lists are machine-dependent, so only the refusals past
them are stored.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgSerialCorpus

def A_PING : Nat := 0x00
def A_BUF : Nat := 0x10
def A_NAME : Nat := 0x40
def MEM : Nat := 0x80

def N : Nat := 18
def OUT : Nat := 4 * N

/-- The greeting the terminal sends before the program runs. -/
def GREETING : String := "hello"

def image : List UInt8 :=
  let pad (xs : List UInt8) (n : Nat) := xs ++ List.replicate (n - xs.length) 0
  pad ("ping".toUTF8.toList) MEM

def ser (f : SerialFn) (args : Vals V (Ext.serial f).sig.1) : Prog V L (ResV V (Ext.serial f).sig.2) :=
  ext (.serial f) args

def body : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  let path ← dataPtr
  let z ← iconst64 0
  let put (k : Nat) (v : V .i32) : Prog V L Unit := do storeI32 v (← iadd out (← iconst64 (4 * k)))
  let put64 (k : Nat) (v : V .i64) : Prog V L Unit := do put k (← ireduce32 v)
  let nz (k : Nat) (v : V .i64) : Prog V L Unit := do put k (← ireduce32 (← uextend64 (← icmp .ne v z)))
  let n ← ser .count %[]
  put 0 (← ireduce32 (← uextend64 (← icmp .sge n (← iconst32 0))))
  let name ← absAddr base A_NAME
  put64 1 (← ser .name %[n, name, ← iconst64 16])
  put64 2 (← ser .name %[← iconst32 (-1), name, ← iconst64 16])
  nz 3 (← ser .open %[path, ← iconst32 0, ← iconst32 100])
  nz 4 (← ser .open %[path, ← iconst32 9600, ← iconst32 (-1)])
  let p ← ser .open %[path, ← iconst32 115200, ← iconst32 200]
  nz 5 p
  nz 6 (← ser .open %[path, ← iconst32 115200, ← iconst32 200])
  put64 7 (← ser .pending %[p])
  let buf ← absAddr base A_BUF
  put64 8 (← ser .read %[p, buf, ← iconst64 3])
  storeI32 (← load32 buf) (← iadd out (← iconst64 (4 * 9)))
  let buf2 ← iaddImm buf 8
  put64 10 (← ser .read %[p, buf2, ← iconst64 16])
  storeI32 (← load32 buf2) (← iadd out (← iconst64 (4 * 11)))
  put64 12 (← ser .read %[p, buf, ← iconst64 16])
  put64 13 (← ser .read %[p, buf, ← iconst64 (-1)])
  put64 14 (← ser .write %[p, ← absAddr base A_PING, ← iconst64 4])
  put64 15 (← ser .write %[p, buf, ← iconst64 (-1)])
  put64 16 (← ser .pending %[p])
  ser .close %[p]
  put 17 (← iconst32 1)

def codeNames : List String :=
  [ "listed: some number", "name past the listed ports", "name of port -1", "opened at baud 0",
    "opened with a negative timeout", "opened", "opened again while held", "waiting", "first read",
    "its bytes", "second read", "its bytes", "read with nothing left", "read of a negative length",
    "wrote ping", "write of a negative length", "waiting after", "finished" ]

def checked : Except String Code := Prog.emitChecked body
def code : Code := Prog.emit body
def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]
def env : FnEnv := (Prog.run body).2.1

/-- The path's bytes are the test's; any path stands for them here, since
    the device at it is the one terminal. -/
def PATH : String := "/dev/pts/corpus"

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
   .sc .i64 (PATH.utf8ByteSize + 1).toUInt64, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

def startWorld : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ⟨(PATH.toUTF8.toList ++ [0]).toArray⟩,
             out := ByteArray.mk (Array.replicate OUT 0) },
    serialDevices := fun _ => some GREETING.toUTF8 }

def run : Except String Sem.World :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w

end HProgSerialCorpus

open AlgorithmLib in
def Host.SerialCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgSerialCorpus.checked with
  | .error e => throw (IO.userError s!"the serial corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgSerialCorpus.run with
  | .error e => throw (IO.userError s!"interpreting the serial corpus: {e}")
  | .ok w =>
      let clif ← AlgorithmLib.Prog.orDie HProgSerialCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_serial_corpus" {
        functions := clif, required_memory := HProgSerialCorpus.MEM,
        initial_memory := HProgSerialCorpus.image }]
      let sideDir := System.FilePath.mk dir / "hprog_serial_corpus"
      IO.FS.createDirAll sideDir
      let sent := ((w.ser[0]?).map (·.sent.toList.map (·.toNat))).getD []
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (w.mem.out.toList.map (·.toNat))),
                          ("names", Lean.toJson HProgSerialCorpus.codeNames),
                          ("greeting", Lean.toJson HProgSerialCorpus.GREETING),
                          ("sent", Lean.toJson sent)]).compress
      IO.println s!"serial corpus: {w.mem.out.size} expected bytes"

#eval ShipScan.check "Host.SerialCorpus" `Host.SerialCorpus.main
