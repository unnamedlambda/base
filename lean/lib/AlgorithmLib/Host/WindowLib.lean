module
public import AlgorithmLib.Host.Ffi
meta import AlgorithmLib.Host.Ffi
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The window library, called directly

What each `WindowFn` does: the engine's seven functions over winit
(`base/src/ffi/window.rs`). As with wgpu, a call is decoded — its window one
the program opened and has not closed, its out-parameter with room — and
then carried out, always answering (`wlDecode`, `wlEffect`).

A window lives outside every region (`wlWinAddr`), so a program passes it and
never loads through it. Events come from the world's `wlInput`, a batch each
time the library pumps: at `base_window_pump` and at every
`base_window_poll`; each record goes to the window it names, and one for a
window since closed is dropped. Without a display the event loop does not
start, `base_window_init` answers `false`, and no window opens. A window's
size in pixels is the world's `wlPixels` of the size it was opened at. A
wgpu surface is made from the window's handle (`Host/Wgpu.lean`).

**Undefined.** A window not open (closed, or never opened), a title that is
not a C string where the size is positive, a poll with no room for its
record.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

inductive WlCall where
  | init
  | open (wd ht : Int)
  | close (k : Nat)
  | pump
  | poll (k : Nat)
  /-- An answer that changes nothing. -/
  | answer (v : V)

/-- The open window at `a`. -/
def World.wlWin? (w : World) (a : UInt64) : Option Nat :=
  (List.range w.wl.windows.size).find? fun k =>
    wlWinAddr k == a && (w.wl.windows[k]?.map (·.live)).getD false

/-- The size in pixels `base_window_pixels` answers for window `k`: the width,
    and the height in the upper 32 bits. -/
def World.wlPixelsOf (w : World) (k : Nat) : UInt64 :=
  let win := w.wl.windows[k]?.getD { width := 0, height := 0 }
  let (x, y) := w.wlPixels (win.width, win.height)
  let mask : UInt64 := 0xffffffff
  (UInt64.ofInt x &&& mask) ||| ((UInt64.ofInt y &&& mask) <<< (32 : UInt64))

section
variable (w : World)

/-- A size that is not positive answers null before the title is read; a
    window opens only on a thread whose event loop is up. -/
def wlDOpen : List UInt64 → Option WlCall
  | [title, wd, ht] =>
      if asI32 wd ≤ 0 || asI32 ht ≤ 0 then some (.answer (.sc .i64 0))
      else do
        let _ ← readCStr w.mem title
        if w.wl.up then some (.open (asI32 wd) (asI32 ht)) else some (.answer (.sc .i64 0))
  | _ => none

def wlDClose : List UInt64 → Option WlCall
  | [win] => (w.wlWin? win).map .close
  | _ => none

/-- Room for a record: 32 bytes. -/
def wlDPoll : List UInt64 → Option WlCall
  | [win, out] => do
      let k ← w.wlWin? win
      let _ ← copyIn w.mem out (ByteArray.mk (Array.replicate 32 0))
      some (.poll k)
  | _ => none

def wlDPixels : List UInt64 → Option WlCall
  | [win] => do let k ← w.wlWin? win; some (.answer (.sc .i64 (w.wlPixelsOf k)))
  | _ => none

end

/-- **What a window library call decodes to**, or `none` where the library
    does not define it. -/
def wlDecode (f : WindowFn) (bits : List UInt64) (w : World) : Option WlCall :=
  match f with
  | .init => if bits.isEmpty then some .init else none
  | .open => wlDOpen w bits
  | .close => wlDClose w bits
  | .pump => if bits.isEmpty then some .pump else none
  | .poll => wlDPoll w bits
  | .pixels => wlDPixels w bits

/-- What has arrived by the next pump, onto the events of the windows open:
    nothing while the event loop is down. -/
def World.wlPump (w : World) : World :=
  if !w.wl.up then w else
  match w.wlInput with
  | [] => w
  | batch :: rest =>
      let windows := batch.foldl (fun ws (k, r) =>
        ws.modify k fun win => if win.live then { win with pending := win.pending ++ [r] } else win)
        w.wl.windows
      { w with wl := { w.wl with windows }, wlInput := rest }

/-- The next record of window `k`, after a pump. -/
def World.wlNext (w : World) (k : Nat) : Option ByteArray :=
  ((w.wlPump.wl.windows[k]?).map (·.pending)).bind List.head?

def wlTrue : Option V := some (ofInt .i8 1)
def wlFalse : Option V := some (ofInt .i8 0)

/-- **What a decoded window library call does to it.** It always answers,
    and it leaves program memory alone: the record a poll takes is written in
    `wlCall`, where the caller asked. -/
def wlEffect (c : WlCall) (w : World) : Option V × World :=
  match c with
  | .init =>
      if w.wl.up then (wlTrue, w)
      else if w.display then (wlTrue, { w with wl := { w.wl with up := true } })
      else (wlFalse, w)
  | .open wd ht =>
      (some (.sc .i64 (wlWinAddr w.wl.windows.size)),
       { w with wl := { w.wl with windows := w.wl.windows.push { width := wd, height := ht } } })
  | .close k =>
      (none, { w with wl := { w.wl with windows := w.wl.windows.modify k fun win =>
        { win with live := false, pending := [] } } })
  | .pump => (none, w.wlPump)
  | .poll k =>
      let w := w.wlPump
      match (w.wl.windows[k]?).map (·.pending) with
      | some (_ :: rest) =>
          (wlTrue, { w with wl := { w.wl with windows := w.wl.windows.modify k fun win =>
            { win with pending := rest } } })
      | _ => (wlFalse, w)
  | .answer v => (some v, w)

/-- **What a window library call does**: decoded, then carried out; the
    record a poll takes is written where the caller asked. -/
def wlCall (f : WindowFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  (wlDecode f bits w).map fun c =>
    let e := wlEffect c w
    match f, bits with
    | .poll, [win, out] =>
        (e.1, { e.2 with mem := match (w.wlWin? win).bind w.wlNext with
          | some r => (copyIn w.mem out (exactly r 32)).getD w.mem
          | none => w.mem })
    | _, _ => e

theorem exactly_size (b : ByteArray) (n : Nat) : (exactly b n).size = n := by
  rw [exactly, ByteArray.size_append, ByteArray.size_extract]
  have : (ByteArray.mk (Array.replicate (n - b.size) 0)).size = n - b.size := Array.size_replicate ..
  rw [this]; omega

theorem wlPump_mem (w : World) : w.wlPump.mem = w.mem := by
  unfold World.wlPump; split
  · rfl
  · split <;> rfl

theorem wlEffect_mem (c : WlCall) (w : World) : (wlEffect c w).2.mem = w.mem := by
  cases c <;> simp only [wlEffect]
  all_goals first
    | rfl
    | (split <;> (try split) <;> rfl)
    | exact wlPump_mem w
    | (split <;> simp only [wlPump_mem])

end AlgorithmLib.HProg.Sem
