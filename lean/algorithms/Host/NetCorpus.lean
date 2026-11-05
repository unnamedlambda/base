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
# The network library, checked against the system

`Sem.netCall` states what each network library call does. This body works on
the loopback address alone: it resolves a numeric name; binds a listener to a
port the system picks, asks its address, and finds nothing waiting; connects
to it and accepts the connection; sends both ways, receives in pieces, runs
out a receive's timeout, shuts a stream for sending and receives its end;
finds a connection to a closed listener refused; and sends a datagram between
two UDP sockets, receiving it with its sender. Ports are the machine's, so
only whether the sender's is the socket's own is stored. Every other answer
is stored; `base/tests/hprog_net_corpus.rs` runs it and compares.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgNetCorpus

def A_LOOP : Nat := 0x00
def A_BAD : Nat := 0x18
def A_BOUND : Nat := 0x30
def A_UA : Nat := 0x48
def A_VA : Nat := 0x60
def A_PING : Nat := 0x80
def A_PONG : Nat := 0x84
def A_HEY : Nat := 0x88
def A_NAME : Nat := 0x90
def A_BUF : Nat := 0xA0
def A_RES : Nat := 0xE0
def MEM : Nat := 0x120

def N : Nat := 31
def OUT : Nat := 4 * N

/-- The loopback address, port `0`, as a record. -/
def loopback : List UInt8 := [4, 0, 0, 0, 0, 0, 0, 0, 127, 0, 0, 1] ++ List.replicate 12 0

def image : List UInt8 :=
  let at_ (xs : List UInt8) (k : Nat) (bytes : List UInt8) := xs ++ List.replicate (k - xs.length) 0 ++ bytes
  let img := at_ [] A_LOOP loopback
  let img := at_ img A_BAD (9 :: List.replicate 23 0)
  let img := at_ img A_PING "ping".toUTF8.toList
  let img := at_ img A_PONG "pong".toUTF8.toList
  let img := at_ img A_HEY "hey".toUTF8.toList
  let img := at_ img A_NAME ("127.0.0.1".toUTF8.toList ++ [0])
  img ++ List.replicate (MEM - img.length) 0

def net (f : NetFn) (args : Vals V (Ext.net f).sig.1) : Prog V L (ResV V (Ext.net f).sig.2) :=
  ext (.net f) args

set_option maxRecDepth 4096 in
def body : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  let z ← iconst64 0
  let at_ (k : Nat) : Prog V L (V .i64) := do iadd out (← iconst64 (4 * k))
  let put (k : Nat) (v : V .i32) : Prog V L Unit := do storeI32 v (← at_ k)
  let put64 (k : Nat) (v : V .i64) : Prog V L Unit := do put k (← ireduce32 v)
  let nz (k : Nat) (v : V .i64) : Prog V L Unit := do put k (← ireduce32 (← uextend64 (← icmp .ne v z)))
  let addr (a : Nat) : Prog V L (V .i64) := iaddImm base a
  let buf ← addr A_BUF
  let len (n : Int) : Prog V L (V .i64) := iconst64 n
  let ms (n : Int) : Prog V L (V .i32) := iconst32 n
  put64 0 (← net .resolve %[← addr A_NAME, ← len 0, ← addr A_RES, ← len 48])
  put64 1 (← net .resolve %[← addr A_NAME, ← len 16, ← addr A_RES, ← len 48])
  storeI32 (← load32 (← addr A_RES)) (← at_ 2)
  let l ← net .tcpListen %[← addr A_LOOP]
  nz 3 l
  put64 4 (← net .tcpLocal %[l, ← addr A_BOUND])
  nz 5 (← uload16_64 (← addr (A_BOUND + 2)))
  nz 6 (← net .tcpAccept %[l, ← ms 1000])
  let c ← net .tcpConnect %[← addr A_BOUND, ← ms 1000]
  nz 7 c
  let s ← net .tcpAccept %[l, ← ms 100]
  nz 8 s
  put64 9 (← net .tcpSend %[c, ← addr A_PING, ← len 4])
  put64 10 (← net .tcpRecv %[s, buf, ← len 2])
  storeI32 (← load32 buf) (← at_ 11)
  put64 12 (← net .tcpRecv %[s, buf, ← len 16])
  put64 13 (← net .tcpRecv %[s, buf, ← len 16])
  put64 14 (← net .tcpShutdown %[c])
  put64 15 (← net .tcpRecv %[s, buf, ← len 16])
  put64 16 (← net .tcpSend %[s, ← addr A_PONG, ← len 4])
  put64 17 (← net .tcpRecv %[c, buf, ← len 16])
  put64 18 (← net .tcpRecv %[c, buf, ← len (-1)])
  net .tcpClose %[c]
  put64 19 (← net .tcpRecv %[s, buf, ← len 16])
  net .tcpClose %[s]
  net .tcpListenerClose %[l]
  nz 20 (← net .tcpConnect %[← addr A_BOUND, ← ms 1000])
  let u ← net .udpBind %[← addr A_LOOP, ← ms 1000]
  nz 21 u
  let v ← net .udpBind %[← addr A_LOOP, ← ms 100]
  nz 22 v
  put64 23 (← net .udpLocal %[u, ← addr A_UA])
  let _ ← net .udpLocal %[v, ← addr A_VA]
  put64 24 (← net .udpSend %[v, ← addr A_HEY, ← len 3, ← addr A_UA])
  put64 25 (← net .udpRecv %[u, buf, ← len 26])
  let same ← icmp .eq (← uload16_64 (← iaddImm buf 2)) (← uload16_64 (← addr (A_VA + 2)))
  put 26 (← ireduce32 (← uextend64 same))
  storeI32 (← load32 (← iaddImm buf 24)) (← at_ 27)
  put64 28 (← net .udpRecv %[v, buf, ← len 64])
  put64 29 (← net .udpRecv %[u, buf, ← len 10])
  net .udpClose %[u]
  net .udpClose %[v]
  put 30 (← ireduce32 (← uextend64 (← icmp .eq (← net .tcpConnect %[← addr A_BAD, ← ms 1000]) z)))

def codeNames : List String :=
  [ "resolved an empty name", "resolved 127.0.0.1", "its record's first word", "bound a listener",
    "asked its address", "its port is one", "accepted with none waiting", "connected to it",
    "accepted the connection", "sent ping", "received two bytes", "their bytes", "received the rest",
    "received with nothing sent", "shut the stream for sending", "received its end", "sent pong back",
    "received pong", "received a negative length", "received after the far end closed",
    "connected to the closed listener", "bound a UDP socket", "bound another", "asked its address",
    "sent a datagram", "received it", "its sender is the other socket", "its first bytes",
    "received with nothing sent", "received into room under a record", "an unknown family refused" ]

def checked : Except String Code := Prog.emitChecked body
def code : Code := Prog.emit body
def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]
def env : FnEnv := (Prog.run body).2.1

/-- The loopback address, as the model holds one. -/
def loopAddr : Sem.NetAddr := { family := 4, port := 0, scope := 0, ip := [127, 0, 0, 1] ++ List.replicate 12 0 }

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .data), .sc .i64 0, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

def startWorld : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ByteArray.empty, out := ByteArray.mk (Array.replicate OUT 0) },
    resolves := fun n => if n == "127.0.0.1" then some [loopAddr] else none }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

end HProgNetCorpus

open AlgorithmLib in
def Host.NetCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgNetCorpus.checked with
  | .error e => throw (IO.userError s!"the network corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgNetCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the network corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgNetCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_net_corpus" {
        functions := clif, required_memory := HProgNetCorpus.MEM,
        initial_memory := HProgNetCorpus.image }]
      let sideDir := System.FilePath.mk dir / "hprog_net_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat))),
                          ("names", Lean.toJson HProgNetCorpus.codeNames)]).compress
      IO.println s!"network corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.NetCorpus" `Host.NetCorpus.main
