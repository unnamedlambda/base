module
public import AlgorithmLib.Host.Ffi
meta import AlgorithmLib.Host.Ffi
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The serial library, called directly

What each `SerialFn` does: the engine's seven functions over the
`serialport` crate (`base/src/ffi/serial.rs`). The ports the system lists are
the world's `serialNames`; a device that opens at a path is its
`serialDevices`, with the bytes it has sent and not yet been read. A port
lives outside every region (`serPortAddr`), so a program passes it and never
loads through it.

A port is held exclusively: opening a path already open answers null. A
read takes what has arrived, up to what it asks for, and answers `0` when
nothing has — the port's timeout having run out; a write sends every byte or
answers `-1`. What the program has written to each port is kept, so a corpus
can compare it with what the device received.

**Undefined.** A port not open, a path that is not a C string, a buffer
without room for what is read or bytes to write.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

inductive SerCall where
  | name (i : Nat) (out : UInt64) (cap : Nat)
  | open (path : String)
  | close (k : Nat)
  | read (k : Nat) (buf : UInt64) (len : Nat)
  | write (k : Nat) (data : ByteArray)
  /-- An answer that changes nothing. -/
  | answer (v : V)

/-- A length the C side reads as an `int64_t`: a negative one is none. -/
def lenOf (x : UInt64) : Nat := if x < 0x8000000000000000 then x.toNat else 0

theorem lenOf_le (x : UInt64) : lenOf x ≤ x.toNat := by
  unfold lenOf; split <;> omega

/-- The open port at `a`. -/
def World.serPort? (w : World) (a : UInt64) : Option Nat :=
  (List.range w.ser.size).find? fun k => serPortAddr k == a && (w.ser[k]?.map (·.live)).getD false

section
variable (w : World)

/-- A port's name: room for as many of its bytes as fit in `cap`. -/
def serDName : List UInt64 → Option SerCall
  | [i, out, cap] =>
      let n := asI32 i
      match (if n < 0 then none else w.serialNames[n.toNat]?) with
      | none => some (.answer (.sc .i64 (UInt64.ofInt (-1))))
      | some name =>
          let c := min name.utf8ByteSize (lenOf cap)
          (copyIn w.mem out (ByteArray.mk (Array.replicate c 0))).map fun _ => .name n.toNat out c
  | _ => none

/-- A baud rate that is not positive, or a negative timeout, is refused
    after the path is read. -/
def serDOpen : List UInt64 → Option SerCall
  | [path, baud, t] => do
      let p ← readCStr w.mem path
      if asI32 baud ≤ 0 || asI32 t < 0 then some (.answer (.sc .i64 0)) else some (.open p)
  | _ => none

def serDClose : List UInt64 → Option SerCall
  | [p] => (w.serPort? p).map .close
  | _ => none

/-- Room for `len` bytes; a negative length answers `-1`. -/
def serDRead : List UInt64 → Option SerCall
  | [p, buf, len] => do
      let k ← w.serPort? p
      if len ≥ 0x8000000000000000 then some (.answer (.sc .i64 (UInt64.ofInt (-1))))
      else do
        let _ ← copyIn w.mem buf (ByteArray.mk (Array.replicate (lenOf len) 0))
        some (.read k buf (lenOf len))
  | _ => none

def serDWrite : List UInt64 → Option SerCall
  | [p, buf, len] => do
      let k ← w.serPort? p
      if len ≥ 0x8000000000000000 then some (.answer (.sc .i64 (UInt64.ofInt (-1))))
      else do
        let bs ← readBytes w.mem buf (lenOf len)
        some (.write k bs)
  | _ => none

def serDPending : List UInt64 → Option SerCall
  | [p] => do
      let k ← w.serPort? p
      let waiting := ((w.ser[k]?).map (·.incoming.size)).getD 0
      some (.answer (.sc .i64 (UInt64.ofNat waiting)))
  | _ => none

end

/-- **What a serial library call decodes to**, or `none` where the library
    does not define it. -/
def serDecode (f : SerialFn) (bits : List UInt64) (w : World) : Option SerCall :=
  match f with
  | .count => if bits.isEmpty then some (.answer (ofInt .i32 w.serialNames.length)) else none
  | .name => serDName w bits
  | .open => serDOpen w bits
  | .close => serDClose w bits
  | .read => serDRead w bits
  | .write => serDWrite w bits
  | .pending => serDPending w bits

/-- The bytes a read of `len` takes from port `k`. -/
def World.serTake (w : World) (k : Nat) (len : Nat) : ByteArray :=
  let inc := ((w.ser[k]?).map (·.incoming)).getD ByteArray.empty
  inc.extract 0 (min len inc.size)

/-- **What a decoded serial call does to the ports.** It always answers, and
    it leaves program memory alone: what a name or a read writes is written in
    `serialCall`, where the caller asked. -/
def serEffect (c : SerCall) (w : World) : Option V × World :=
  match c with
  | .name i _ _ =>
      (some (.sc .i64 (UInt64.ofNat (((w.serialNames[i]?).map (·.utf8ByteSize)).getD 0))), w)
  | .open p =>
      let held := w.ser.any fun s => s.live && s.path == p
      match (if held then none else w.serialDevices p) with
      | some input =>
          (some (.sc .i64 (serPortAddr w.ser.size)), { w with ser := w.ser.push { path := p, incoming := input } })
      | none => (some (.sc .i64 0), w)
  | .close k => (none, { w with ser := w.ser.modify k fun s => { s with live := false } })
  | .read k _ len =>
      let got := w.serTake k len
      (some (.sc .i64 (UInt64.ofNat got.size)),
       { w with ser := w.ser.modify k fun s => { s with incoming := s.incoming.extract got.size s.incoming.size } })
  | .write k data =>
      (some (.sc .i64 (UInt64.ofNat data.size)),
       { w with ser := w.ser.modify k fun s => { s with sent := s.sent ++ data } })
  | .answer v => (some v, w)

/-- What a name call writes: as much of port `i`'s name as fits in `cap`. -/
def World.serNameBytes (w : World) (i cap : UInt64) : ByteArray :=
  let n := asI32 i
  match (if n < 0 then none else w.serialNames[n.toNat]?) with
  | some name => name.toUTF8.extract 0 (lenOf cap)
  | none => ByteArray.empty

/-- What a read call writes: the bytes it takes from the port at `p`. -/
def World.serReadBytes (w : World) (p len : UInt64) : ByteArray :=
  match w.serPort? p with
  | some k => w.serTake k (lenOf len)
  | none => ByteArray.empty

theorem extract_zero_size_le (b : ByteArray) (n : Nat) : (b.extract 0 n).size ≤ n := by
  rw [ByteArray.size_extract]; omega

theorem serNameBytes_size (w : World) (i cap : UInt64) : (w.serNameBytes i cap).size ≤ cap.toNat := by
  unfold World.serNameBytes; dsimp only
  split
  · exact Nat.le_trans (extract_zero_size_le _ _) (lenOf_le cap)
  · simp

theorem serReadBytes_size (w : World) (p len : UInt64) : (w.serReadBytes p len).size ≤ len.toNat := by
  unfold World.serReadBytes World.serTake; split
  · exact Nat.le_trans (by rw [ByteArray.size_extract]; omega) (lenOf_le len)
  · simp

/-- **What a serial library call does**: decoded, then carried out; a port's
    name and the bytes a read takes are written where the caller asked. -/
def serialCall (f : SerialFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  (serDecode f bits w).map fun c =>
    let e := serEffect c w
    match f, bits with
    | .name, [i, out, cap] => (e.1, { e.2 with mem := (copyIn w.mem out (w.serNameBytes i cap)).getD w.mem })
    | .read, [p, buf, len] => (e.1, { e.2 with mem := (copyIn w.mem buf (w.serReadBytes p len)).getD w.mem })
    | _, _ => e

theorem serEffect_mem (c : SerCall) (w : World) : (serEffect c w).2.mem = w.mem := by
  cases c <;> simp only [serEffect]
  all_goals first | rfl | (split <;> rfl)

end AlgorithmLib.HProg.Sem
