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
# The USB library, checked against the system

`Sem.usbCall` states what each USB library call does. What devices a machine
has, and which a process may open, are the machine's, so this body stores
only what holds on any: the devices listed are some number; a place past
them, or below zero, names no device, so what it is answers `-1` and it does
not open; a question the library does not have answers `-1`; and a listed
device is on a numbered bus. `base/tests/hprog_usb_corpus.rs` runs it and
compares. Transfers need a device this process may open, which a machine
without one granted does not have: their clauses stand on the documentation
(`Ext.sources`), not on this corpus.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgUsbCorpus

def N : Nat := 8
def OUT : Nat := 4 * N
def MEM : Nat := 0x40

def usb (f : UsbFn) (args : Vals V (Ext.usb f).sig.1) : Prog V L (ResV V (Ext.usb f).sig.2) :=
  ext (.usb f) args

def body : Prog V L Unit := do
  let out ← outPtr
  let z ← iconst64 0
  let put (k : Nat) (v : V .i32) : Prog V L Unit := do storeI32 v (← iadd out (← iconst64 (4 * k)))
  let flag (k : Nat) (c : V .i8) : Prog V L Unit := do put k (← ireduce32 (← uextend64 c))
  let n ← usb .count %[]
  flag 0 (← icmp .sge n (← iconst32 0))
  put 1 (← ireduce32 (← usb .info %[n, ← iconst32 0]))
  put 2 (← ireduce32 (← usb .info %[← iconst32 (-1), ← iconst32 2]))
  put 3 (← ireduce32 (← usb .info %[← iconst32 0, ← iconst32 9]))
  flag 4 (← icmp .eq (← usb .open %[n]) z)
  flag 5 (← icmp .eq (← usb .open %[← iconst32 (-1)]) z)
  let none_ ← icmp .eq n (← iconst32 0)
  let bus ← usb .info %[← iconst32 0, ← iconst32 0]
  flag 6 (← bor none_ (← icmp .sge bus (← iconst64 0)))
  put 7 (← iconst32 1)

def codeNames : List String :=
  [ "listed: some number", "bus of the device past the listed ones", "vendor of device -1",
    "question 9 of device 0", "the device past the listed ones opened", "device -1 opened",
    "a listed device is on a numbered bus", "finished" ]

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

end HProgUsbCorpus

open AlgorithmLib in
def Host.UsbCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgUsbCorpus.checked with
  | .error e => throw (IO.userError s!"the USB corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgUsbCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the USB corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgUsbCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_usb_corpus" {
        functions := clif, required_memory := HProgUsbCorpus.MEM,
        initial_memory := List.replicate HProgUsbCorpus.MEM 0 }]
      let sideDir := System.FilePath.mk dir / "hprog_usb_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat))),
                          ("names", Lean.toJson HProgUsbCorpus.codeNames)]).compress
      IO.println s!"USB corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.UsbCorpus" `Host.UsbCorpus.main
