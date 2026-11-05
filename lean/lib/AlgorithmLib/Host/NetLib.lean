module
public import AlgorithmLib.Host.FsLib
meta import AlgorithmLib.Host.FsLib
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The network library, called directly

What each `NetFn` does: the engine's functions over `std::net`
(`base/src/builtin/net.rs`). Sockets of this run are the world's `socks`; a
connection to a listener among them makes both its ends there, the far one
waiting in the listener's queue until it is accepted, and what one end sends
arrives at the other. A stream connected outside the run reaches a server
the world names (`remote`), and what it sends there is kept. A socket lives
outside every region (`netSockAddr`), so a program passes it and never loads
through it.

An address is a 24-byte record (`NetAddr.ofBytes`). Port `0` binds a port
the system picks (`netPort`). An accept never waits: with none queued it
answers null. A receive with nothing arrived answers `0` once the other end
has shut sending or closed, and `-2` — its timeout run out — otherwise.

**Undefined.** A socket not held or of another kind, an address record not
all there, a buffer without room for what is received or bytes to send.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- The bytes of an address record. -/
def addrBytes : Nat := 24

/-- The address a record holds, `none` for a family other than `4` and `6`. -/
def NetAddr.ofBytes (b : ByteArray) : Option NetAddr :=
  let port := (b.get! 2).toNat + 256 * (b.get! 3).toNat
  if b.get! 0 == 4 then
    some { family := 4, port, scope := 0,
           ip := (List.range 16).map fun i => if i < 4 then b.get! (8 + i) else 0 }
  else if b.get! 0 == 6 then
    some { family := 6, port,
           scope := (b.get! 4).toNat + 256 * (b.get! 5).toNat + 65536 * (b.get! 6).toNat
             + 16777216 * (b.get! 7).toNat,
           ip := (List.range 16).map fun i => b.get! (8 + i) }
  else none

/-- Byte `i` of an address's record. -/
def NetAddr.byte (a : NetAddr) (i : Nat) : UInt8 :=
  if i = 0 then a.family.toUInt8
  else if i = 1 then 0
  else if i < 4 then (a.port / 256 ^ (i - 2) % 256).toUInt8
  else if i < 8 then (a.scope / 256 ^ (i - 4) % 256).toUInt8
  else a.ip.getD (i - 8) 0

/-- An address as its record. -/
def NetAddr.toBytes (a : NetAddr) : ByteArray := ⟨((List.range addrBytes).map a.byte).toArray⟩

theorem NetAddr.toBytes_size (a : NetAddr) : a.toBytes.size = addrBytes := by
  simp [NetAddr.toBytes, ByteArray.size]

/-- Whether a socket bound to `b` is the one an address `a` reaches: the same
    family and port, and the same IP or none in particular. -/
def NetAddr.reaches (b a : NetAddr) : Bool :=
  b.family == a.family && b.port == a.port && (b.ip == a.ip || b.ip.all (· == 0))

inductive NetCall where
  | resolve (name : String) (room : Nat)
  | connect (a : NetAddr)
  | listen (a : NetAddr)
  | accept (k : Nat)
  | send (k : Nat) (data : ByteArray)
  | recv (k : Nat) (len : Nat)
  | shutdown (k : Nat)
  | close (k : Nat)
  | bind (a : NetAddr)
  | udpSend (k : Nat) (data : ByteArray) (dst : NetAddr)
  | udpRecv (k : Nat)
  /-- An answer that changes nothing. -/
  | answer (v : V)

/-- The socket of kind `kind` the program holds at `a`. -/
def World.sock? (w : World) (kind : SockKind) (a : UInt64) : Option Nat :=
  (List.range w.socks.size).find? fun k =>
    netSockAddr k == a && (w.socks[k]?.map fun s => s.live && s.held && s.kind == kind).getD false

def netNeg : NetCall := .answer (ofInt .i64 (-1))
def netNull : NetCall := .answer (.sc .i64 0)

section
variable (w : World)

/-- The address in the record at `a`, read whole: `none` inside for a family
    the library does not know. -/
def addrAt (a : UInt64) : Option (Option NetAddr) :=
  (readBytes w.mem a addrBytes).map NetAddr.ofBytes

def netDResolve : List UInt64 → Option NetCall
  | [name, cap, out, ocap] => do
      let p ← readField w.mem name cap
      match p with
      | some p =>
          if p == "" then some netNeg else
          match w.resolves p with
          | some as =>
              let room := lenOf ocap / addrBytes
              (copyIn w.mem out (ByteArray.mk (Array.replicate (addrBytes * min as.length room) 0))).map
                fun _ => .resolve p room
          | none => some netNeg
      | none => some netNeg
  | _ => none

/-- An address and a timeout: null for an unknown family or a timeout that is
    not positive. -/
def netDTimed (k : NetAddr → NetCall) : List UInt64 → Option NetCall
  | [a, t] => do
      let a ← addrAt w a
      match a with
      | some a => if asI32 t ≤ 0 then some netNull else some (k a)
      | none => some netNull
  | _ => none

def netDListen : List UInt64 → Option NetCall
  | [a] => do
      let a ← addrAt w a
      match a with
      | some a => some (.listen a)
      | none => some netNull
  | _ => none

/-- The address a held socket of kind `kind` is bound to, with room for it. -/
def netDLocal (kind : SockKind) : List UInt64 → Option NetCall
  | [h, out] => do
      let _ ← w.sock? kind h
      let _ ← copyIn w.mem out (ByteArray.mk (Array.replicate addrBytes 0))
      some (.answer (ofInt .i64 0))
  | _ => none

def netDAccept : List UInt64 → Option NetCall
  | [l, t] => do
      let k ← w.sock? .listener l
      if asI32 t ≤ 0 then some netNull else some (.accept k)
  | _ => none

def netDSend : List UInt64 → Option NetCall
  | [s, buf, len] => do
      let k ← w.sock? .stream s
      if negI64 len then some netNeg else do
        let bs ← readBytes w.mem buf len.toNat
        some (.send k bs)
  | _ => none

/-- Room for `len` bytes; a negative length answers `-1`, and none `0`. -/
def netDRecv : List UInt64 → Option NetCall
  | [s, buf, len] => do
      let k ← w.sock? .stream s
      if negI64 len then some netNeg
      else if len == 0 then some (.answer (ofInt .i64 0))
      else do
        let _ ← copyIn w.mem buf (ByteArray.mk (Array.replicate len.toNat 0))
        some (.recv k len.toNat)
  | _ => none

def netDHeld (kind : SockKind) (c : Nat → NetCall) : List UInt64 → Option NetCall
  | [h] => (w.sock? kind h).map c
  | _ => none

/-- The destination is read before the length is looked at. -/
def netDUdpSend : List UInt64 → Option NetCall
  | [u, buf, len, a] => do
      let k ← w.sock? .udp u
      let a ← addrAt w a
      match a with
      | some a =>
          if negI64 len then some netNeg else do
            let bs ← readBytes w.mem buf len.toNat
            some (.udpSend k bs a)
      | none => some netNeg
  | _ => none

/-- Room for `len` bytes, at least a record's. -/
def netDUdpRecv : List UInt64 → Option NetCall
  | [u, buf, len] => do
      let k ← w.sock? .udp u
      if negI64 len || lenOf len < addrBytes then some netNeg else do
        let _ ← copyIn w.mem buf (ByteArray.mk (Array.replicate len.toNat 0))
        some (.udpRecv k)
  | _ => none

end

/-- **What a network library call decodes to**, or `none` where the library
    does not define it. -/
def netDecode (f : NetFn) (bits : List UInt64) (w : World) : Option NetCall :=
  match f with
  | .resolve => netDResolve w bits
  | .tcpConnect => netDTimed w .connect bits
  | .tcpListen => netDListen w bits
  | .tcpLocal => netDLocal w .listener bits
  | .tcpAccept => netDAccept w bits
  | .tcpSend => netDSend w bits
  | .tcpRecv => netDRecv w bits
  | .tcpShutdown => netDHeld w .stream .shutdown bits
  | .tcpClose => netDHeld w .stream .close bits
  | .tcpListenerClose => netDHeld w .listener .close bits
  | .udpBind => netDTimed w .bind bits
  | .udpLocal => netDLocal w .udp bits
  | .udpSend => netDUdpSend w bits
  | .udpRecv => netDUdpRecv w bits
  | .udpClose => netDHeld w .udp .close bits

/-- The live socket of a kind an address reaches. -/
def World.reached (w : World) (kind : SockKind) (a : NetAddr) : Option Nat :=
  (List.range w.socks.size).find? fun k =>
    (w.socks[k]?.map fun s => s.live && s.kind == kind && s.addr.reaches a).getD false

/-- A socket bound to `a`, a port picked for port `0`: `none` where a live
    socket of the kind already has the port. -/
def World.bound (w : World) (kind : SockKind) (a : NetAddr) : Option NetAddr :=
  let a := if a.port == 0 then { a with port := w.netPort w.socks.size } else a
  if (w.reached kind a).isSome then none else some a

/-- The answer naming socket `k`. -/
def sockAns (k : Nat) : Option V := some (.sc .i64 (netSockAddr k))

/-- **What a decoded network call does to the sockets.** It always answers,
    and it leaves program memory alone: what a call writes is written in
    `netCall`, where the caller asked. -/
def netEffect (c : NetCall) (w : World) : Option V × World :=
  match c with
  | .resolve p _ => (some (ofInt .i64 ((w.resolves p).getD []).length), w)
  | .connect a =>
      let c := w.socks.size
      let mine := { a with port := w.netPort c }
      match w.reached .listener a with
      | some l =>
          let ls := (w.socks.push { kind := .stream, addr := mine, peer := some (c + 1) }).push
            { held := false, kind := .stream, addr := a, peer := some c }
          (sockAns c, { w with socks := ls.modify l fun s => { s with queue := s.queue ++ [c + 1] } })
      | none =>
          match w.remote a with
          | some bytes => (sockAns c, { w with socks := w.socks.push { kind := .stream, addr := mine, incoming := bytes } })
          | none => (some (.sc .i64 0), w)
  | .listen a =>
      match w.bound .listener a with
      | some b => (sockAns w.socks.size, { w with socks := w.socks.push { kind := .listener, addr := b } })
      | none => (some (.sc .i64 0), w)
  | .accept k =>
      match (w.socks[k]?).map (·.queue) with
      | some (j :: rest) =>
          let ss := w.socks.modify k fun s => { s with queue := rest }
          match ss[j]? with
          | some s =>
              if s.live && s.kind == .stream then
                (sockAns j, { w with socks := ss.modify j fun s => { s with held := true } })
              else (some (.sc .i64 0), { w with socks := ss })
          | none => (some (.sc .i64 0), { w with socks := ss })
      | _ => (some (.sc .i64 0), w)
  | .send k data =>
      let s := (w.socks[k]?).getD { kind := .stream, addr := default }
      if s.wshut then (some (ofInt .i64 (-1)), w) else
      match s.peer with
      | some j =>
          (some (ofInt .i64 data.size),
           { w with socks := w.socks.modify j fun p => if p.live then { p with incoming := p.incoming ++ data } else p })
      | none => (some (ofInt .i64 data.size), { w with socks := w.socks.modify k fun s => { s with sent := s.sent ++ data } })
  | .recv k len =>
      let s := (w.socks[k]?).getD { kind := .stream, addr := default }
      if s.incoming.size > 0 then
        let n := min len s.incoming.size
        (some (ofInt .i64 n),
         { w with socks := w.socks.modify k fun s => { s with incoming := s.incoming.extract n s.incoming.size } })
      else
        let gone := match s.peer with
          | some j => !((w.socks[j]?.map (·.live)).getD false)
          | none => false
        (some (ofInt .i64 (if s.shut || gone then 0 else -2)), w)
  | .shutdown k =>
      let ss := w.socks.modify k fun s => { s with wshut := true }
      (some (ofInt .i64 0), { w with socks := match ((w.socks[k]?).map (·.peer)).join with
        | some j => ss.modify j fun p => { p with shut := true }
        | none => ss })
  | .close k =>
      let ss := w.socks.modify k fun s => { s with live := false }
      (none, { w with socks := match ((w.socks[k]?).map (·.peer)).join with
        | some j => ss.modify j fun p => { p with shut := true }
        | none => ss })
  | .bind a =>
      match w.bound .udp a with
      | some b => (sockAns w.socks.size, { w with socks := w.socks.push { kind := .udp, addr := b } })
      | none => (some (.sc .i64 0), w)
  | .udpSend k data dst =>
      let s := (w.socks[k]?).getD { kind := .udp, addr := default }
      let most := if dst.family == 4 then 65507 else 65527
      if s.addr.family != dst.family || data.size > most then (some (ofInt .i64 (-1)), w) else
      let src := if s.addr.ip.all (· == 0) then { s.addr with ip := dst.ip } else s.addr
      (some (ofInt .i64 data.size),
       match w.reached .udp dst with
       | some j => { w with socks := w.socks.modify j fun r => { r with dgrams := r.dgrams ++ [(src, data)] } }
       | none => w)
  | .udpRecv k =>
      match (w.socks[k]?).map (·.dgrams) with
      | some ((_, data) :: rest) =>
          (some (ofInt .i64 data.size), { w with socks := w.socks.modify k fun s => { s with dgrams := rest } })
      | _ => (some (ofInt .i64 (-2)), w)
  | .answer v => (some v, w)

/-- What a resolve writes: the records of as many addresses as fit. -/
def World.netResolveBytes (w : World) (name cap ocap : UInt64) : ByteArray :=
  match readField w.mem name cap with
  | some (some p) =>
      let as := (w.resolves p).getD []
      (as.take (lenOf ocap / addrBytes)).foldl (fun acc a => acc ++ a.toBytes) ByteArray.empty
  | _ => ByteArray.empty

/-- What asking a socket of a kind its address writes: its record. -/
def World.netLocalBytes (w : World) (kind : SockKind) (h : UInt64) : ByteArray :=
  match (w.sock? kind h).bind (w.socks[·]?) with
  | some s => s.addr.toBytes
  | none => ByteArray.empty

/-- What a stream's receive writes: the bytes it takes. -/
def World.netRecvBytes (w : World) (h len : UInt64) : ByteArray :=
  match (w.sock? .stream h).bind (w.socks[·]?) with
  | some s => if negI64 len then ByteArray.empty else s.incoming.extract 0 (min len.toNat s.incoming.size)
  | none => ByteArray.empty

/-- What a UDP receive writes: the sender's record, then as much of the
    datagram as fits after it. -/
def World.netUdpBytes (w : World) (h len : UInt64) : ByteArray :=
  match (w.sock? .udp h).bind (w.socks[·]?) with
  | some s =>
      if negI64 len || lenOf len < addrBytes then ByteArray.empty else
      match s.dgrams with
      | (src, data) :: _ => src.toBytes ++ data.extract 0 (lenOf len - addrBytes)
      | [] => ByteArray.empty
  | none => ByteArray.empty

theorem netResolveBytes_size (w : World) (name cap ocap : UInt64) :
    (w.netResolveBytes name cap ocap).size ≤ ocap.toNat := by
  unfold World.netResolveBytes
  split
  · rename_i p _
    suffices h : ∀ (l : List NetAddr) (acc : ByteArray),
        (l.foldl (fun acc a => acc ++ a.toBytes) acc).size = acc.size + addrBytes * l.length by
      rw [h]; simp only [ByteArray.size_empty, Nat.zero_add]
      have := List.length_take_le (lenOf ocap / addrBytes) ((w.resolves p).getD [])
      have h2 := Nat.div_mul_le_self (lenOf ocap) addrBytes
      have h3 := lenOf_le ocap
      calc addrBytes * (List.take (lenOf ocap / addrBytes) ((w.resolves p).getD [])).length
          ≤ addrBytes * (lenOf ocap / addrBytes) := Nat.mul_le_mul_left _ (by
            rw [List.length_take]; exact Nat.min_le_left _ _)
        _ ≤ ocap.toNat := by rw [Nat.mul_comm]; omega
    intro l
    induction l with
    | nil => intro acc; simp
    | cons a l ih =>
        intro acc
        rw [List.foldl_cons, ih, ByteArray.size_append, NetAddr.toBytes_size, List.length_cons]
        rw [Nat.mul_succ]; omega
  · simp

theorem netLocalBytes_size (w : World) (kind : SockKind) (h : UInt64) :
    (w.netLocalBytes kind h).size ≤ addrBytes := by
  unfold World.netLocalBytes; split
  · rw [NetAddr.toBytes_size]; exact Nat.le_refl _
  · simp

theorem netRecvBytes_size (w : World) (h len : UInt64) : (w.netRecvBytes h len).size ≤ len.toNat := by
  unfold World.netRecvBytes; split
  · split
    · simp
    · rw [ByteArray.size_extract]; omega
  · simp

theorem netUdpBytes_size (w : World) (h len : UInt64) : (w.netUdpBytes h len).size ≤ len.toNat := by
  unfold World.netUdpBytes
  split
  · split
    · simp
    · rename_i hl
      split
      · rw [ByteArray.size_append, NetAddr.toBytes_size, ByteArray.size_extract]
        have := lenOf_le len
        simp only [Bool.or_eq_true, not_or, decide_eq_true_eq] at hl
        have h2 := hl.2
        omega
      · simp
  · simp

/-- **What a network library call does**: decoded, then carried out; what a
    call hands back is written where the caller asked. -/
def netCall (f : NetFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  (netDecode f bits w).map fun c =>
    let e := netEffect c w
    let put (a : UInt64) (bs : ByteArray) := (e.1, { e.2 with mem := (copyIn w.mem a bs).getD w.mem })
    match f, bits with
    | .resolve, [name, cap, out, ocap] => put out (w.netResolveBytes name cap ocap)
    | .tcpLocal, [h, out] => put out (w.netLocalBytes .listener h)
    | .udpLocal, [h, out] => put out (w.netLocalBytes .udp h)
    | .tcpRecv, [h, buf, len] => put buf (w.netRecvBytes h len)
    | .udpRecv, [h, buf, len] => put buf (w.netUdpBytes h len)
    | _, _ => e

theorem netEffect_mem (c : NetCall) (w : World) : (netEffect c w).2.mem = w.mem := by
  cases c <;> simp only [netEffect]
  all_goals (repeat' split) <;> rfl

end AlgorithmLib.HProg.Sem
