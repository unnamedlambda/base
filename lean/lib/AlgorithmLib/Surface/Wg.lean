module
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
import all Init.Data.Repr
@[expose] public section

/-!
# `Wg` — GPU work as calls into wgpu

A program that runs compute shaders calls wgpu itself (`Ext.wgpu`), its
objects' handles at fixed offsets of its arena (`App`): the instance, adapter,
device and queue; per buffer, the storage buffer and the buffer it is read
back through; per pipeline, its bind group layout, pipeline layout, pipeline
and bind group, and the words those were made from. Each helper is a few
wgpu calls at those offsets; the checker reads the calls themselves
(`Check.WgpuRule`), so a theorem about a program states what it needs of wgpu
in wgpu's own terms.
-/

namespace AlgorithmLib.Wg

open Lean Elab Term Meta in
/-- `utf8Len% s`: the length in bytes of the string `s`, evaluated when the
    program is elaborated and written as a number. A program's layout states
    its shaders' lengths this way, so a proof about the program reads a number
    rather than evaluating the code that generates the shader. -/
elab "utf8Len% " t:term : term => do
  let e ← instantiateMVars (← elabTerm t (some (mkConst ``String)))
  let s ← unsafe evalExpr String (mkConst ``String) e
  return mkNatLit s.utf8ByteSize

open AlgorithmLib.IR AlgorithmLib.Prog

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- `WGPUBufferUsage`: a storage buffer copied to and from; a buffer mapped
    for reading, copied to. -/
def USAGE_STORAGE : Nat := 128 + 8 + 4
def USAGE_STAGING : Nat := 1 + 8

/-- A compute pipeline: its WGSL's offset in the arena and length in bytes,
    and its bindings in order, each a buffer by index and whether the shader
    only reads it. Its entry point is `main`. -/
structure PSpec where
  shaderOff : Nat
  shaderLen : Nat
  binds : List (Nat × Bool)

/-- A window a program draws a buffer of its own to: the WGSL of the render
    pipeline that blits it (entry points `vs_main` and `fs_main`, its one
    binding the buffer), the buffer by index, and the window's title, a C
    string in a field of `titleCap` bytes. -/
structure Blit where
  shaderOff : Nat
  shaderLen : Nat
  buf : Nat
  titleOff : Nat
  titleCap : Nat

/-- **A program in the standard layout**: its buffers' sizes in bytes, its
    pipelines, its arena's size, and where in the arena wgpu's state starts
    (`org`). From `org` lie the instance, adapter, device and queue, the entry
    point's name `main` at `0x20`, a storage and a staging slot per buffer, five
    slots per pipeline (its bind group layout, pipeline layout, pipeline and
    bind group, and the one-handle array its pipeline layout is made of), and
    per pipeline `cells` words of layout entries and as many pairs of words of
    bind group entries, and for a program with a window, the window's state
    (`winBase`). The shaders lie wherever `shaderOff` says, outside all of it. -/
structure App where
  bufs : List Nat
  pipes : List PSpec
  memSize : Nat
  org : Nat := 0
  /-- How many bindings a pipeline may have. -/
  cells : Nat := 8
  /-- The window it draws to, if any. -/
  blit : Option Blit := none

namespace App

def INST (a : App) : Nat := a.org
def ADAPTER (a : App) : Nat := a.org + 0x08
def DEVICE (a : App) : Nat := a.org + 0x10
def QUEUE (a : App) : Nat := a.org + 0x18
def NAME (a : App) : Nat := a.org + 0x20

def bufSlot (a : App) (i : Nat) : Nat := a.org + 0x28 + 16 * i
def stageSlot (a : App) (i : Nat) : Nat := a.bufSlot i + 8

def pipeSlots (a : App) (j : Nat) : Nat := a.org + 0x28 + 16 * a.bufs.length + 40 * j
def BGL (a : App) (j : Nat) : Nat := a.pipeSlots j
def PL (a : App) (j : Nat) : Nat := a.pipeSlots j + 8
def PIPE (a : App) (j : Nat) : Nat := a.pipeSlots j + 16
def BG (a : App) (j : Nat) : Nat := a.pipeSlots j + 24
def ARR (a : App) (j : Nat) : Nat := a.pipeSlots j + 32

/-- Pipeline `j`'s layout entries, a word each, then its bind group entries,
    two words each. -/
def bglEnts (a : App) (j : Nat) : Nat := a.pipeSlots a.pipes.length + 24 * a.cells * j
def bgEnts (a : App) (j : Nat) : Nat := a.bglEnts j + 8 * a.cells

/-- The window's state, past the pipelines': the window, its surface, the
    blit pipeline, its layout and bind group, the size the surface is
    configured at (`i32`s), a record the window library polls, the bind
    group's entry, and the blit's entry points' names. -/
def winBase (a : App) : Nat := a.bglEnts a.pipes.length
def WIN (a : App) : Nat := a.winBase
def SURF (a : App) : Nat := a.winBase + 0x08
def RPIPE (a : App) : Nat := a.winBase + 0x10
def RBGL (a : App) : Nat := a.winBase + 0x18
def RBG (a : App) : Nat := a.winBase + 0x20
def PW (a : App) : Nat := a.winBase + 0x28
def PH (a : App) : Nat := a.winBase + 0x2c
def EV (a : App) : Nat := a.winBase + 0x30
def RBGENT (a : App) : Nat := a.winBase + 0x50
def VS (a : App) : Nat := a.winBase + 0x60
def FS (a : App) : Nat := a.winBase + 0x68

/-- Where wgpu's state ends. -/
def stateEnd (a : App) : Nat := if a.blit.isSome then a.winBase + 0x70 else a.winBase

/-- Each pipeline with its index. -/
def indexed (a : App) : List (Nat × PSpec) := (List.range a.pipes.length).zip a.pipes

/-- A handle kept at `off`. -/
def at_ (base : V .i64) (off : Nat) : Prog V L (V .i64) := do load64 (← iaddImm base off)

/-- **Bring wgpu up**: an instance, its adapter, a device and its queue; each
    buffer with its staging buffer; each pipeline, its layouts and its bind
    group. The shader modules and pipeline layouts are released once their
    pipelines are made. -/
def setup (a : App) (base : V .i64) : Prog V L Unit := do
  let inst ← ext (.wgpu .createInstance) %[]
  storeI64 inst (← iaddImm base a.INST)
  let ad ← ext (.wgpu .requestAdapter) %[inst]
  storeI64 ad (← iaddImm base a.ADAPTER)
  let dev ← ext (.wgpu .requestDevice) %[ad]
  storeI64 dev (← iaddImm base a.DEVICE)
  let q ← ext (.wgpu .deviceGetQueue) %[dev]
  storeI64 q (← iaddImm base a.QUEUE)
  for (n, i) in a.bufs.zip (List.range a.bufs.length) do
    let b ← ext (.wgpu .deviceCreateBuffer) %[dev, ← iconst64 USAGE_STORAGE, ← iconst64 n]
    storeI64 b (← iaddImm base (a.bufSlot i))
    let st ← ext (.wgpu .deviceCreateBuffer) %[dev, ← iconst64 USAGE_STAGING, ← iconst64 n]
    storeI64 st (← iaddImm base (a.stageSlot i))
  for (j, p) in a.indexed do
    let sh ← ext (.wgpu .deviceCreateShaderModule) %[dev, ← iaddImm base p.shaderOff, ← iconst64 p.shaderLen]
    for ((_, ro), k) in p.binds.zip (List.range p.binds.length) do
      storeI64 (← iconst64 (k + (if ro then 2 ^ 32 else 0))) (← iaddImm base (a.bglEnts j + 8 * k))
    let bgl ← ext (.wgpu .deviceCreateBindGroupLayout) %[dev, ← iaddImm base (a.bglEnts j), ← iconst64 p.binds.length]
    storeI64 bgl (← iaddImm base (a.BGL j))
    storeI64 bgl (← iaddImm base (a.ARR j))
    let pl ← ext (.wgpu .deviceCreatePipelineLayout) %[dev, ← iaddImm base (a.ARR j), ← iconst64 1]
    let pipe ← ext (.wgpu .deviceCreateComputePipeline) %[dev, pl, sh, ← iaddImm base a.NAME, ← iconst64 4]
    storeI64 pipe (← iaddImm base (a.PIPE j))
    for ((bi, _), k) in p.binds.zip (List.range p.binds.length) do
      storeI64 (← iconst64 k) (← iaddImm base (a.bgEnts j + 16 * k))
      storeI64 (← at_ base (a.bufSlot bi)) (← iaddImm base (a.bgEnts j + 16 * k + 8))
    let bg ← ext (.wgpu .deviceCreateBindGroup) %[dev, bgl, ← iaddImm base (a.bgEnts j), ← iconst64 p.binds.length]
    storeI64 bg (← iaddImm base (a.BG j))
    let _ ← ext (.wgpu (.release .shaderModule)) %[sh]
    let _ ← ext (.wgpu (.release .pipelineLayout)) %[pl]

/-- Record a dispatch of pipeline `j` over `gx × gy × gz` workgroups and
    submit it. -/
def dispatch (a : App) (base : V .i64) (j : Nat) (gx gy gz : V .i32) : Prog V L Unit := do
  let enc ← ext (.wgpu .deviceCreateCommandEncoder) %[← at_ base a.DEVICE]
  let pass ← ext (.wgpu .encoderBeginComputePass) %[enc]
  let _ ← ext (.wgpu .computePassSetPipeline) %[pass, ← at_ base (a.PIPE j)]
  let _ ← ext (.wgpu .computePassSetBindGroup) %[pass, ← iconst32 0, ← at_ base (a.BG j)]
  let _ ← ext (.wgpu .computePassDispatch) %[pass, gx, gy, gz]
  let _ ← ext (.wgpu .computePassEnd) %[pass]
  let cb ← ext (.wgpu .encoderFinish) %[enc]
  let _ ← ext (.wgpu .queueSubmit) %[← at_ base a.QUEUE, cb]

/-- `bytes` bytes from `src` into buffer `i`, landing before any later
    submit. -/
def upload (a : App) (base : V .i64) (i : Nat) (src bytes : V .i64) : Prog V L Unit := do
  let _ ← ext (.wgpu .queueWriteBuffer) %[← at_ base a.QUEUE, ← at_ base (a.bufSlot i), ← iconst64 0, src, bytes]

/-- The first `bytes` bytes of buffer `i` to `dst`, through its staging
    buffer, once every earlier submit has run. -/
def download (a : App) (base : V .i64) (i : Nat) (dst bytes : V .i64) : Prog V L Unit := do
  let z ← iconst64 0
  let enc ← ext (.wgpu .deviceCreateCommandEncoder) %[← at_ base a.DEVICE]
  let _ ← ext (.wgpu .encoderCopyBufferToBuffer) %[enc, ← at_ base (a.bufSlot i), z, ← at_ base (a.stageSlot i), z,
    bytes]
  let cb ← ext (.wgpu .encoderFinish) %[enc]
  let _ ← ext (.wgpu .queueSubmit) %[← at_ base a.QUEUE, cb]
  let _ ← ext (.wgpu .bufferRead) %[← at_ base a.DEVICE, ← at_ base (a.stageSlot i), z, dst, bytes]

/-- Release everything `setup` made. -/
def cleanup (a : App) (base : V .i64) : Prog V L Unit := do
  for (j, _) in a.indexed do
    let _ ← ext (.wgpu (.release .bindGroup)) %[← at_ base (a.BG j)]
    let _ ← ext (.wgpu (.release .computePipeline)) %[← at_ base (a.PIPE j)]
    let _ ← ext (.wgpu (.release .bindGroupLayout)) %[← at_ base (a.BGL j)]
  for (_, i) in a.bufs.zip (List.range a.bufs.length) do
    let _ ← ext (.wgpu (.release .buffer)) %[← at_ base (a.bufSlot i)]
    let _ ← ext (.wgpu (.release .buffer)) %[← at_ base (a.stageSlot i)]
  let _ ← ext (.wgpu (.release .queue)) %[← at_ base a.QUEUE]
  let _ ← ext (.wgpu (.release .device)) %[← at_ base a.DEVICE]
  let _ ← ext (.wgpu (.release .adapter)) %[← at_ base a.ADAPTER]
  let _ ← ext (.wgpu (.release .instance)) %[← at_ base a.INST]

-- ---------------------------------------------------------------------------
-- A window
-- ---------------------------------------------------------------------------

/-- The surfaces' format: `BGRA8Unorm`. -/
def FORMAT : Nat := 0x1B

/-- **A window of `wd × ht` pixels to present to**, open while `body` runs on
    its handle: the window library brought up, the window opened, its surface
    configured, and the blit pipeline and its bind group over the presented
    buffer made; all of it released after. Without a display, or where the
    window does not open, `body` does not run. -/
def withWindow (a : App) (base : V .i64) (wd ht : Nat) (body : V .i64 → Prog V L Unit) : Prog V L Unit := do
  match a.blit with
  | none => pure ()
  | some b => do
    let z ← iconst64 0
    let up ← ext (.window .init) %[]
    when .ne up (← iconst .i8 0) (do
      let win ← ext (.window .open) %[← iaddImm base b.titleOff, ← iconst32 wd, ← iconst32 ht]
      when .ne win z (do
        let dev ← at_ base a.DEVICE
        let surf ← ext (.wgpu .instanceCreateSurface) %[← at_ base a.INST, win]
        storeI64 surf (← iaddImm base a.SURF)
        let _ ← ext (.wgpu .surfaceConfigure) %[surf, dev, ← iconst32 FORMAT, ← iconst32 wd, ← iconst32 ht]
        storeI32 (← iconst32 wd) (← iaddImm base a.PW)
        storeI32 (← iconst32 ht) (← iaddImm base a.PH)
        let m ← ext (.wgpu .deviceCreateShaderModule) %[dev, ← iaddImm base b.shaderOff, ← iconst64 b.shaderLen]
        let seven ← iconst64 7
        let rp ← ext (.wgpu .deviceCreateRenderPipeline) %[dev, m, ← iaddImm base a.VS, seven, ← iaddImm base a.FS,
          seven, ← iconst32 FORMAT]
        storeI64 rp (← iaddImm base a.RPIPE)
        let _ ← ext (.wgpu (.release .shaderModule)) %[m]
        let bgl ← ext (.wgpu .renderPipelineGetBindGroupLayout) %[rp, ← iconst32 0]
        storeI64 bgl (← iaddImm base a.RBGL)
        storeI64 z (← iaddImm base a.RBGENT)
        storeI64 (← at_ base (a.bufSlot b.buf)) (← iaddImm base (a.RBGENT + 8))
        let bg ← ext (.wgpu .deviceCreateBindGroup) %[dev, bgl, ← iaddImm base a.RBGENT, ← iconst64 1]
        storeI64 bg (← iaddImm base a.RBG)
        body win
        let _ ← ext (.wgpu (.release .bindGroup)) %[← at_ base a.RBG]
        let _ ← ext (.wgpu (.release .renderPipeline)) %[← at_ base a.RPIPE]
        let _ ← ext (.wgpu (.release .bindGroupLayout)) %[← at_ base a.RBGL]
        let _ ← ext (.wgpu (.release .surface)) %[← at_ base a.SURF]
        let _ ← ext (.window .close) %[win]))

/-- The engine's id for a key's USB HID usage: `0` for a key it does not name. -/
def keyId (usage : V .i64) : Prog V L (V .i64) := do
  let ids : List (Int × Int) :=
    [(41, 1), (44, 2), (80, 3), (79, 4), (82, 5), (81, 6), (26, 10), (4, 11), (22, 12), (7, 13)]
  ids.foldlM (fun acc (k, id) => do select (← icmp .eq usage (← iconst64 k)) (← iconst64 id) acc)
    (← iconst64 0)

/-- A coordinate, the bits of an `f64`, rounded to the nearest pixel; `0` for
    one no pixel is at. -/
def roundCoord (bits : V .i64) : Prog V L (V .i64) := do
  let x ← bitcast .f64 bits
  let ok ← fcmp .lt (← fabs x) (← fconst64 2147483648.0)
  let r ← ifte (jTys := [.i64]) .ne ok (← iconst .i8 0)
    (do pure %[← fcvtToSint .i64 (← nearest x)]) (do pure %[← iconst64 0])
  pure r.head

/-- The engine's record of the library's record at `ev`, written at `out`:
    its kind, `0` for a key the engine does not name. The kinds and the
    buttons are numbered alike; keys are mapped, the pointer rounded. -/
def translate (ev out : V .i64) : Prog V L (V .i64) := do
  let z ← iconst64 0
  let ty ← load64 ev
  let a ← load64 (← iaddImm ev 8)
  let b ← load64 (← iaddImm ev 16)
  let isKey ← bor (← icmp .eq ty (← iconst64 3)) (← icmp .eq ty (← iconst64 4))
  let motion ← icmp .eq ty (← iconst64 5)
  let key ← keyId a
  let dropped ← band isKey (← icmp .eq key z)
  let kind ← select dropped z ty
  let a' ← select isKey key (← select motion (← roundCoord a) a)
  let b' ← select motion (← roundCoord b) b
  storeI64 kind out
  storeI64 a' (← iaddImm out 8)
  storeI64 b' (← iaddImm out 16)
  storeI64 z (← iaddImm out 24)
  pure kind

/-- Up to `max` of the window's events written at `events`, 32 bytes each, in
    the engine's records: how many. -/
def pollEvents (a : App) (base win events : V .i64) (max : Nat) : Prog V L (V .i64) := do
  let z ← iconst64 0
  let m ← iconst64 max
  let ev ← iaddImm base a.EV
  let e ← wloop1L z
    (head := fun _ n => return (contIfULt n m, %[n], ()))
    (body := fun l n _ => do
      let got ← ext (.window .poll) %[win, ev]
      let r ← ifte (jTys := [.i64]) .eq got (← iconst .i8 0) (brk l %[n]) (do
        let kind ← translate ev (← iadd events (← ishlImm n 5))
        pure %[← select (← icmp .ne kind z) (← iaddImm n 1) n])
      return %[r.head])
  pure e.head

/-- **One frame**: the presented buffer drawn to the window and shown, the
    surface reconfigured first where the window's size in pixels changed; a
    size no surface takes keeps the one it has. -/
def present (a : App) (base win : V .i64) : Prog V L Unit := do
  let z32 ← iconst32 0
  let _ ← ext (.window .pump) %[]
  let px ← ext (.window .pixels) %[win]
  let pw ← ireduce32 px
  let ph ← ireduce32 (← ushrImm px 32)
  let big ← iconst32 0x80000000
  when .ugt pw z32 (when .ult pw big (when .ugt ph z32 (when .ult ph big (do
    let changed ← bor (← icmp .ne pw (← load32 (← iaddImm base a.PW))) (← icmp .ne ph (← load32 (← iaddImm base a.PH)))
    when .ne changed (← iconst .i8 0) (do
      let _ ← ext (.wgpu .surfaceConfigure) %[← at_ base a.SURF, ← at_ base a.DEVICE, ← iconst32 FORMAT, pw, ph]
      storeI32 pw (← iaddImm base a.PW)
      storeI32 ph (← iaddImm base a.PH))))))
  let tex ← ext (.wgpu .surfaceGetCurrentTexture) %[← at_ base a.SURF]
  let view ← ext (.wgpu .textureCreateView) %[tex]
  let enc ← ext (.wgpu .deviceCreateCommandEncoder) %[← at_ base a.DEVICE]
  let pass ← ext (.wgpu .encoderBeginRenderPass) %[enc, view, ← iconst32 1]
  let _ ← ext (.wgpu .renderPassSetPipeline) %[pass, ← at_ base a.RPIPE]
  let _ ← ext (.wgpu .renderPassSetBindGroup) %[pass, z32, ← at_ base a.RBG]
  let _ ← ext (.wgpu .renderPassDraw) %[pass, ← iconst32 3, ← iconst32 1, z32, z32]
  let _ ← ext (.wgpu .renderPassEnd) %[pass]
  let cb ← ext (.wgpu .encoderFinish) %[enc]
  let _ ← ext (.wgpu .queueSubmit) %[← at_ base a.QUEUE, cb]
  let _ ← ext (.wgpu (.release .textureView)) %[view]
  let _ ← ext (.wgpu .surfacePresent) %[tex]

/-- What wgpu's state holds from `org` before the program starts: the entry
    points' names at their places, zero elsewhere. -/
def stateInit (a : App) : List UInt8 :=
  let names : List (Nat × String) := [(a.NAME, "main"), (a.VS, "vs_main"), (a.FS, "fs_main")]
  (names.foldl (fun m (o, n) => n.toUTF8.toList.zipIdx.foldl (fun m (b, i) => m.set! (o - a.org + i) b) m)
    (Array.replicate (a.stateEnd - a.org) (0 : UInt8))).toList

/-- The arena's first bytes: the name `main` at its place, the blit's entry
    points' names at theirs, each pipeline's shader at its own, `shaders` in
    the pipelines' order, and each of `extra`'s bytes at its offset. -/
def initialMemory (a : App) (shaders : List String) (extra : List (Nat × List UInt8) := []) : List UInt8 :=
  let placed := (a.NAME, "main".toUTF8.toList) :: (a.VS, "vs_main".toUTF8.toList) ::
    (a.FS, "fs_main".toUTF8.toList) ::
    (a.pipes.zip shaders).map (fun (p, s) => (p.shaderOff, s.toUTF8.toList)) ++ extra
  (placed.foldl (fun m (o, bs) => bs.zipIdx.foldl (fun m (b, i) => m.set! (o + i) b) m)
    (Array.replicate a.memSize (0 : UInt8))).toList

end App

end AlgorithmLib.Wg
