module
public import AlgorithmLib.Surface.LibWgpu
meta import AlgorithmLib.Surface.LibWgpu
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lib.Window` — the engine's window entry points, as CLIF over its window library

One window per context, its input as event records, and a present that draws
a wgpu storage buffer with the blit shader the window was opened with — the
`window*` entry points stated in `Host.Ffi`, as functions of the program
itself over the engine's window library (`Ext.window`, winit underneath) and
wgpu (`Ext.wgpu`).

**The surface join.** A wgpu surface is made from the window's handle, which
it shares. The surface, and the pipeline the blit shader compiles to, are made
at the first present, on the device of the wgpu context the buffer belongs to,
so the buffer is drawn where it already is. A window then presents that
context's buffers only.

**Events.** The library's records are read one at a time and written as the
engine's — kind, then three operands — for a close request, a change of the
window's size in pixels, the keys the engine names (by USB HID usage), the
pointer's position (rounded to the nearest pixel) and its buttons; a key the
engine does not name is read and dropped.

**Frames.** Each present reads the window's size in pixels and reconfigures
the surface when it changed. A surface texture that is not ready — timed out,
occluded, outdated or lost — drops the frame, answers `0`, and has the surface
reconfigured at the next.

**Without a display or wgpu** `windowInit` stores null.
-/

namespace AlgorithmLib.LibWindow

open AlgorithmLib.IR
open AlgorithmLib.Prog
open AlgorithmLib.Lib
open AlgorithmLib.LibWgpu (wg rel)

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def WIN : Nat := 0x00
def SURF : Nat := 0x08
/-- The wgpu context whose device the surface and pipeline are on. -/
def GPU : Nat := 0x10
def PIPE : Nat := 0x18
def BGL : Nat := 0x20
/-- The size the surface is configured at, `i32`s; z until it is. -/
def PW : Nat := 0x28
def PH : Nat := 0x2c
def BLIT : Nat := 0x30
def BLITLEN : Nat := 0x38
def OPEN : Nat := 0x48
/-- A record the window library polls. -/
def EV : Nat := 0x80
/-- A bind group's one entry. -/
def SCRATCH : Nat := 0x100
/-- A name: a property's, or an entry point's. -/
def NAME : Nat := 0x280
def SIZE : Nat := 0x300

/-- `WGPUTextureFormat_BGRA8Unorm`: written as the shader computes it, no
    gamma applied. -/
def FORMAT : Int := 27

def win (f : WindowFn) (args : Vals V (Ext.window f).sig.1) : Prog V L (ResV V (Ext.window f).sig.2) :=
  ext (.window f) args

/-- `str` and a z byte at `a`, eight bytes at a time. -/
def putStr (a : V .i64) (str : String) : Prog V L Unit := do
  let bs := str.toUTF8.toList ++ [0]
  for k in List.range ((bs.length + 7) / 8) do
    let word : Nat := (List.range 8).foldl (fun acc j => acc + (bs.getD (8 * k + j) 0).toNat * 256 ^ j) 0
    storeI64 (← iconst64 word) (← at_ a (8 * k))

def orAll (xs : List (V .i8)) : Prog V L (V .i8) := do
  xs.foldlM (fun acc x => or8 acc x) (← iconst .i8 0)

-- ---------------------------------------------------------------------------
-- The context
-- ---------------------------------------------------------------------------

/-- The thread's event loop and a context; null without wgpu or a display. -/
def init : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let z ← iconst64 0
  storeI64 z slot
  let has ← band (← libPresent .window) (← libPresent .wgpu)
  when .ne has (← iconst32 0) (do
    let ok ← win .init %[]
    when .ne ok (← iconst .i8 0) (do
      let s ← calloc (← iconst64 SIZE)
      when .ne s z (storeI64 s slot)))

/-- One byte of UTF-8 read in state `s`: `0` between characters, `1`–`3` for
    that many continuation bytes still owed, `4`–`7` for a first continuation
    byte with a narrower range (after `E0`, `ED`, `F0`, `F4`: no overlong
    forms, surrogates or code points past `U+10FFFF`), `8` once invalid. The
    same strings `String.fromUTF8?` accepts end in `0`. -/
def utf8Step (s b : V .i64) : Prog V L (V .i64) := do
  let c (k : Int) : Prog V L (V .i64) := iconst64 k
  let inR (x : V .i64) (lo hi : Int) : Prog V L (V .i8) := do
    band (← icmp .uge x (← c lo)) (← icmp .ule x (← c hi))
  let bad ← c 8
  let r ← select (← icmp .ult b (← c 0x80)) (← c 0) bad
  let r ← select (← inR b 0xC2 0xDF) (← c 1) r
  let r ← select (← inR b 0xE1 0xEF) (← c 2) r
  let r ← select (← icmp .eq b (← c 0xE0)) (← c 4) r
  let r ← select (← icmp .eq b (← c 0xED)) (← c 5) r
  let r ← select (← inR b 0xF1 0xF3) (← c 3) r
  let r ← select (← icmp .eq b (← c 0xF0)) (← c 6) r
  let lead ← select (← icmp .eq b (← c 0xF4)) (← c 7) r
  let lo ← select (← icmp .eq s (← c 4)) (← c 0xA0)
    (← select (← icmp .eq s (← c 6)) (← c 0x90) (← c 0x80))
  let hi ← select (← icmp .eq s (← c 5)) (← c 0x9F)
    (← select (← icmp .eq s (← c 7)) (← c 0x8F) (← c 0xBF))
  let ok ← band (← icmp .uge b lo) (← icmp .ule b hi)
  let next ← select (← icmp .ule s (← c 3)) (← isub s (← c 1))
    (← select (← icmp .ule s (← c 5)) (← c 1) (← c 2))
  let r ← select (← icmp .eq s (← c 0)) lead (← select ok next bad)
  select (← icmp .eq s bad) bad r

/-- Whether the `n` bytes at `p` are not UTF-8: nonzero when they are not. -/
def utf8Bad (p n : V .i64) : Prog V L (V .i8) := do
  let s ← forLoopAcc n (← iconst64 0) fun i s => do
    utf8Step s (← uload8_64 (← iadd p i))
  icmp .ne s (← iconst64 0)

/-- The window, titled with `titleLen` bytes, and the blit shader kept for the
    first present: `0`, or `-1` — also for a shader that is not UTF-8, which
    wgpu takes as a string. -/
def open_ : StatusBody := do
  let (s ::ᵥ wd ::ᵥ ht ::ᵥ title ::ᵥ tlen ::ᵥ blit ::ᵥ blen ::ᵥ .nil) ←
    entryParams [.i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let z ← iconst64 0
  let bad ← orAll [← icmp .sle wd z, ← icmp .sle ht z, ← icmp .slt tlen z,
    ← icmp .eq title z, ← icmp .sle blen z, ← icmp .eq blit z]
  failUnless0 bad (do
    failIf .eq s z (do
      failIf .ne (← load64 (← at_ s OPEN)) z (do
       failUnless0 (← utf8Bad blit blen) (do
        let t ← calloc (← iaddImm tlen 1)
        failIf .eq t z (do
          let _ ← memcpy t title tlen
          let w ← win .open %[t, ← ireduce32 wd, ← ireduce32 ht]
          free t
          failIf .eq w z (do
            let b ← calloc blen
            let r ← ifte (jTys := [.i64]) .eq b z
              (do win .close %[w]; pure %[← iconst64 (-1)]) (do
              let _ ← memcpy b blit blen
              storeI64 w (← at_ s WIN)
              storeI64 b (← at_ s BLIT)
              storeI64 blen (← at_ s BLITLEN)
              storeI64 (← iconst64 1) (← at_ s OPEN)
              pure %[z])
            pure r.head))))))

-- ---------------------------------------------------------------------------
-- Events
-- ---------------------------------------------------------------------------

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
  let a ← load64 (← at_ ev 8)
  let b ← load64 (← at_ ev 16)
  let isKey ← or8 (← icmp .eq ty (← iconst64 3)) (← icmp .eq ty (← iconst64 4))
  let motion ← icmp .eq ty (← iconst64 5)
  let key ← keyId a
  let dropped ← band isKey (← icmp .eq key z)
  let kind ← select dropped z ty
  let a' ← select isKey key (← select motion (← roundCoord a) a)
  let b' ← select motion (← roundCoord b) b
  storeI64 kind out
  storeI64 a' (← at_ out 8)
  storeI64 b' (← at_ out 16)
  storeI64 z (← at_ out 24)
  pure kind

/-- Up to `max` events written at `events`, 32 bytes each: how many. -/
def poll : StatusBody := do
  let (s ::ᵥ events ::ᵥ max ::ᵥ .nil) ← entryParams [.i64, .i64, .i32]
  let z ← iconst64 0
  failUnless0 (← or8 (← anyNeg32 [max]) (← icmp .eq events z)) (do
    failIf .eq s z (do
      let m ← sextend64 max
      let ev ← at_ s EV
      let w ← load64 (← at_ s WIN)
      let (n, _) ← whileLoop2 z (← select (← icmp .ne w z) (← iconst64 1) z)
        (fun n more => do band (← icmp .ult n m) (← icmp .ne more z))
        (fun n _ => do
          let got ← win .poll %[w, ev]
          let r ← ifte (jTys := [.i64, .i64]) .eq got (← iconst .i8 0) (do pure %[n, z]) (do
            let kind ← translate ev (← iadd events (← ishlImm n 5))
            pure %[← select (← icmp .ne kind z) (← iaddImm n 1) n, ← iconst64 1])
          pure (r.head, r.snd))
      pure n))

-- ---------------------------------------------------------------------------
-- Presenting
-- ---------------------------------------------------------------------------

/-- The window's surface on the instance of wgpu context `g`. Leaves `SURF`
    z when wgpu cannot make one. -/
def makeSurface (s g : V .i64) : Prog V L Unit := do
  let surf ← wg .instanceCreateSurface %[← load64 (← at_ g LibWgpu.INST), ← load64 (← at_ s WIN)]
  storeI64 surf (← at_ s SURF)
  storeI64 g (← at_ s GPU)

/-- The blit pipeline on `g`'s device, and the layout of its one group; both
    left z when wgpu rejects the shader. -/
def makePipeline (s g : V .i64) : Prog V L Unit := do
  let dev ← load64 (← at_ g LibWgpu.DEVICE)
  let scope ← wg .devicePushErrorScope %[dev]
  let module ← wg .deviceCreateShaderModule %[dev, ← load64 (← at_ s BLIT), ← load64 (← at_ s BLITLEN)]
  let nm ← at_ s NAME
  putStr nm "vs_main"
  let fsName ← at_ nm 16
  putStr fsName "fs_main"
  let seven ← iconst64 7
  let pipe ← wg .deviceCreateRenderPipeline %[dev, module, nm, seven, fsName, seven, ← iconst32 FORMAT]
  rel .shaderModule module
  let err ← wg .errorScopePop %[scope]
  let _ ← ifte (jTys := []) .ne err (← iconst32 0) (do rel .renderPipeline pipe; pure %[]) (do
    storeI64 (← wg .renderPipelineGetBindGroupLayout %[pipe, ← iconst32 0]) (← at_ s BGL)
    storeI64 pipe (← at_ s PIPE)
    pure %[])
  pure ()

/-- Whether the surface and the blit pipeline are made on `g`'s device: `1`
    or `0`. -/
def ready (s g : V .i64) : Prog V L (V .i64) := do
  let z ← iconst64 0
  let gs ← load64 (← at_ s GPU)
  let other ← band (← icmp .ne gs z) (← icmp .ne gs g)
  let r ← ifte (jTys := [.i64]) .ne other (← iconst .i8 0) (do pure %[z]) (do
    when .eq (← load64 (← at_ s SURF)) z (makeSurface s g)
    let r ← ifte (jTys := [.i64]) .eq (← load64 (← at_ s SURF)) z (do pure %[z]) (do
      when .eq (← load64 (← at_ s PIPE)) z (makePipeline s g)
      pure %[← select (← icmp .ne (← load64 (← at_ s PIPE)) z) (← iconst64 1) z])
    pure %[r.head])
  pure r.head

/-- One frame: the buffer at entry `e` of wgpu context `g`, drawn with the
    blit pipeline and presented. `0`, also when the surface has no texture
    ready and the frame is dropped. -/
def frame (s g e : V .i64) : Prog V L (V .i64) := do
  let z ← iconst64 0
  let z32 ← iconst32 0
  let surf ← load64 (← at_ s SURF)
  let dev ← load64 (← at_ g LibWgpu.DEVICE)
  let px ← win .pixels %[← load64 (← at_ s WIN)]
  let pw ← ireduce32 px
  let ph ← ireduce32 (← ushrImm px 32)
  let r ← ifte (jTys := [.i64]) .ne (← anyNonPos [pw, ph]) (← iconst .i8 0) (do pure %[z]) (do
    let pwA ← at_ s PW
    let phA ← at_ s PH
    let changed ← or8 (← icmp .ne pw (← load32 pwA)) (← icmp .ne ph (← load32 phA))
    when .ne changed (← iconst .i8 0) (do
      let _ ← wg .surfaceConfigure %[surf, dev, ← iconst32 FORMAT, pw, ph]
      storeI32 pw pwA
      storeI32 ph phA)
    let tex ← wg .surfaceGetCurrentTexture %[surf]
    let _ ← ifte (jTys := []) .eq tex z (do
        -- not ready: the frame is dropped, and the surface reconfigured next
        storeI32 z32 pwA
        pure %[]) (do
      let view ← wg .textureCreateView %[tex]
      -- the group's one entry: binding 0, the buffer
      let be ← at_ s SCRATCH
      storeI64 z be
      storeI64 (← load64 e) (← at_ be 8)
      let bg ← wg .deviceCreateBindGroup %[dev, ← load64 (← at_ s BGL), be, ← iconst64 1]
      let enc ← wg .deviceCreateCommandEncoder %[dev]
      let pass ← wg .encoderBeginRenderPass %[enc, view, ← iconst32 1]
      let _ ← wg .renderPassSetPipeline %[pass, ← load64 (← at_ s PIPE)]
      let _ ← wg .renderPassSetBindGroup %[pass, z32, bg]
      let _ ← wg .renderPassDraw %[pass, ← iconst32 3, ← iconst32 1, z32, z32]
      let _ ← wg .renderPassEnd %[pass]
      let _ ← wg .queueSubmit %[← load64 (← at_ g LibWgpu.QUEUE), ← wg .encoderFinish %[enc]]
      rel .bindGroup bg
      rel .textureView view
      let _ ← wg .surfacePresent %[tex]
      pure %[])
    pure %[z])
  pure r.head

/-- Buffer `buf` of wgpu context `g` presented: `0`, or `-1`. -/
def present : StatusBody := do
  let (s ::ᵥ g ::ᵥ buf ::ᵥ .nil) ← entryParams [.i64, .i64, .i32]
  let z ← iconst64 0
  failUnless0 (← anyNeg32 [buf]) (do
    failIf .eq s z (do
      win .pump %[]
      failIf .eq g z (do
        LibWgpu.flush g
        let e ← LibWgpu.entry g LibWgpu.BUFS LibWgpu.BUF_W buf
        failIf .eq e z (do
          failIf .eq (← load64 (← at_ s OPEN)) z (do
            failIf .eq (← ready s g) z (frame s g e))))))

/-- The window, its surface and pipeline released. The thread's event loop
    stays, as winit keeps it for the life of the thread. -/
def cleanup : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let s ← load64 slot
  let z ← iconst64 0
  when .ne s z (do
    let pipe ← load64 (← at_ s PIPE)
    when .ne pipe z (rel .renderPipeline pipe)
    let bgl ← load64 (← at_ s BGL)
    when .ne bgl z (rel .bindGroupLayout bgl)
    let surf ← load64 (← at_ s SURF)
    when .ne surf z (rel .surface surf)
    let w ← load64 (← at_ s WIN)
    when .ne w z (win .close %[w])
    free (← load64 (← at_ s BLIT))
    free s)
  storeI64 z slot

/-- The function implementing a window entry point. -/
def implOf : Ffi → Option Impl
  | .windowInit => some (.void init)
  | .windowOpen => some (.status open_)
  | .windowPoll => some (.status poll)
  | .windowPresentGpuBuffer => some (.status present)
  | .windowCleanup => some (.void cleanup)
  | _ => none

end AlgorithmLib.LibWindow
