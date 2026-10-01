module
public import AlgorithmLib.Host.SerialLib
meta import AlgorithmLib.Host.SerialLib
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The USB library, called directly

What each `UsbFn` does: the engine's nine functions over the `nusb` crate
(`base/src/ffi/usb.rs`). The devices the system lists are the world's
`usbDevices`, each with what identifies it, whether this process may open
it, which interfaces it lets be claimed and the packet size of each
endpoint; what a device answers a transfer is its `usbReply`, `none` for a
transfer that fails. A device lives outside every region (`usbDevAddr`), so
a program passes it and never loads through it.

Every refusal answers `-1` and writes nothing: a device or interface not
there, an interface not claimed, a request type with a kind or recipient
USB does not have, a length out of range, and a transfer from the device
that asks for part of a packet. On Windows a control transfer goes through a
claimed interface, and without one is refused (`usbControlNeedsClaim`).

**Undefined.** A device not open, a buffer without room for what comes from
the device or bytes to send.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

inductive UsbCall where
  | open (i : Nat)
  | close (k : Nat)
  | claim (k n : Nat)
  | release (k n : Nat)
  /-- A transfer on device `k`: what is asked, and whether its data comes
      from the device. -/
  | transfer (k : Nat) (x : UsbXfer) (fromDevice : Bool)
  /-- An answer that changes nothing. -/
  | answer (v : V)

def usbNo : UsbCall := .answer (.sc .i64 (UInt64.ofInt (-1)))
def usbNo32 : UsbCall := .answer (ofInt .i32 (-1))

/-- A control transfer's length, a `u16`; one out of range is none. -/
def len16 (x : UInt64) : Nat := if x < 65536 then x.toNat else 0

theorem len16_le (x : UInt64) : len16 x ≤ x.toNat := by
  unfold len16; split <;> omega

/-- The open device at `a`. -/
def World.usbDev? (w : World) (a : UInt64) : Option Nat :=
  (List.range w.usb.size).find? fun k => usbDevAddr k == a && (w.usb[k]?.map (·.live)).getD false

/-- The device listed at `i`, by its place. -/
def World.usbListed? (w : World) (i : UInt64) : Option UsbDevice :=
  let n := asI32 i
  if n < 0 then none else w.usbDevices[n.toNat]?

/-- Whether device `k` has interface `n` claimed. -/
def World.usbClaimed (w : World) (k n : Nat) : Bool :=
  ((w.usb[k]?).map (·.claimed.contains n)).getD false

/-- The device open device `k` is. -/
def World.usbDevOf (w : World) (k : Nat) : Option UsbDevice :=
  (w.usb[k]?).bind fun o => w.usbDevices[o.device]?

section
variable (w : World)

def usbDInfo : List UInt64 → Option UsbCall
  | [i, which] =>
      let v : Int := match w.usbListed? i, asI32 which with
        | some d, 0 => d.bus | some d, 1 => d.address
        | some d, 2 => d.vendor | some d, 3 => d.product
        | _, _ => -1
      some (.answer (.sc .i64 (UInt64.ofInt v)))
  | _ => none

def usbDOpen : List UInt64 → Option UsbCall
  | [i] => match w.usbListed? i with
    | some d => if d.openable then some (.open (asI32 i).toNat) else some (.answer (.sc .i64 0))
    | none => some (.answer (.sc .i64 0))
  | _ => none

def usbDClose : List UInt64 → Option UsbCall
  | [d] => (w.usbDev? d).map .close
  | _ => none

def usbDClaim (release : Bool) : List UInt64 → Option UsbCall
  | [d, iface] => do
      let k ← w.usbDev? d
      let n := asI32 iface
      if n < 0 || n > 255 then some usbNo32
      else if release then
        some (if w.usbClaimed k n.toNat then .release k n.toNat else usbNo32)
      else if w.usbClaimed k n.toNat then some (.answer (ofInt .i32 0))
      else if ((w.usbDevOf k).map (·.claimable.contains n.toNat)).getD false then some (.claim k n.toNat)
      else some usbNo32
  | _ => none

/-- A control transfer: its data stage `len` bytes at `data`, read for one to
    the device and with room for one from it. -/
def usbDControl : List UInt64 → Option UsbCall
  | [d, rt, rq, val, idx, data, len, t] => do
      let k ← w.usbDev? d
      let r := rt &&& 0xff
      let noClaim := ((w.usb[k]?).map (·.claimed.isEmpty)).getD true
      if len ≥ 65536 || asI32 t < 0 || (r >>> 5) &&& 3 == 3 || r &&& 0x1f > 3
          || (w.usbControlNeedsClaim && noClaim) then some usbNo
      else
        let n := len16 len
        let fromDevice := r &&& 0x80 != 0
        let out ← if fromDevice then do
            let _ ← copyIn w.mem data (ByteArray.mk (Array.replicate n 0)); some ByteArray.empty
          else readBytes w.mem data n
        some (.transfer k (.control r.toNat (rq &&& 0xff).toNat (val &&& 0xffff).toNat
          (idx &&& 0xffff).toNat n out) fromDevice)
  | _ => none

/-- A bulk or interrupt transfer of `len` bytes at `data` on an endpoint of
    a claimed interface. -/
def usbDData (interrupt : Bool) : List UInt64 → Option UsbCall
  | [d, iface, ep, data, len, t] => do
      let k ← w.usbDev? d
      let n := asI32 iface
      let e := asI32 ep
      if n < 0 || n > 255 || e < 0 || e > 255 || len ≥ 0x8000000000000000 || asI32 t < 0
          || !w.usbClaimed k n.toNat then some usbNo
      else
        let l := lenOf len
        let fromDevice := e.toNat &&& 0x80 != 0
        let packet := ((w.usbDevOf k).map fun dv => dv.packet e.toNat).getD 0
        if fromDevice && (l == 0 || packet == 0 || l % packet != 0) then some usbNo
        else do
          let out ← if fromDevice then do
              let _ ← copyIn w.mem data (ByteArray.mk (Array.replicate l 0)); some ByteArray.empty
            else readBytes w.mem data l
          let x := if interrupt then UsbXfer.interrupt e.toNat l out else UsbXfer.bulk e.toNat l out
          some (.transfer k x fromDevice)
  | _ => none

end

/-- **What a USB library call decodes to**, or `none` where the library does
    not define it. -/
def usbDecode (f : UsbFn) (bits : List UInt64) (w : World) : Option UsbCall :=
  match f with
  | .count => if bits.isEmpty then some (.answer (ofInt .i32 w.usbDevices.length)) else none
  | .info => usbDInfo w bits
  | .open => usbDOpen w bits
  | .close => usbDClose w bits
  | .claim => usbDClaim w false bits
  | .release => usbDClaim w true bits
  | .control => usbDControl w bits
  | .bulk => usbDData w false bits
  | .interrupt => usbDData w true bits

/-- What device `k` answers transfer `x`. -/
def World.usbAnswer (w : World) (k : Nat) (x : UsbXfer) : Option ByteArray :=
  ((w.usb[k]?).map (·.device)).bind fun i => w.usbReply i x

/-- **What a decoded USB call does to the devices.** It always answers, and
    it leaves program memory alone: what comes from a device is written in
    `usbCall`, where the caller asked. -/
def usbEffect (c : UsbCall) (w : World) : Option V × World :=
  match c with
  | .open i => (some (.sc .i64 (usbDevAddr w.usb.size)), { w with usb := w.usb.push { device := i } })
  | .close k => (none, { w with usb := w.usb.modify k fun o => { o with live := false, claimed := [] } })
  | .claim k n =>
      (some (ofInt .i32 0), { w with usb := w.usb.modify k fun o => { o with claimed := n :: o.claimed } })
  | .release k n =>
      (some (ofInt .i32 0), { w with usb := w.usb.modify k fun o => { o with claimed := o.claimed.erase n } })
  | .transfer k x fromDevice =>
      let moved : Int := match w.usbAnswer k x with
        | none => -1
        | some back => if fromDevice then ((min back.size x.len : Nat) : Int) else (x.len : Int)
      (some (.sc .i64 (UInt64.ofInt moved)), w)
  | .answer v => (some v, w)

/-- What a transfer from the device writes: what the device sent, up to the
    length asked for. -/
def World.usbBack (w : World) (f : UsbFn) (bits : List UInt64) (len : UInt64) : ByteArray :=
  match usbDecode f bits w with
  | some (.transfer k x true) => ((w.usbAnswer k x).getD ByteArray.empty).extract 0 (min x.len len.toNat)
  | _ => ByteArray.empty

theorem usbBack_size (w : World) (f : UsbFn) (bits : List UInt64) (len : UInt64) :
    (w.usbBack f bits len).size ≤ len.toNat := by
  unfold World.usbBack; split
  · rw [ByteArray.size_extract]; omega
  · simp

/-- **What a USB library call does**: decoded, then carried out; what comes
    from the device is written where the caller asked. -/
def usbCall (f : UsbFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  (usbDecode f bits w).map fun c =>
    let e := usbEffect c w
    match f, bits with
    | .control, [_, _, _, _, _, data, len, _] =>
        (e.1, { e.2 with mem := (copyIn w.mem data (w.usbBack f bits len)).getD w.mem })
    | .bulk, [_, _, _, data, len, _] | .interrupt, [_, _, _, data, len, _] =>
        (e.1, { e.2 with mem := (copyIn w.mem data (w.usbBack f bits len)).getD w.mem })
    | _, _ => e

theorem usbEffect_mem (c : UsbCall) (w : World) : (usbEffect c w).2.mem = w.mem := by
  cases c <;> rfl

end AlgorithmLib.HProg.Sem
