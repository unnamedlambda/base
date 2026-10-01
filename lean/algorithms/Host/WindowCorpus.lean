module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Surface.LibWindow
meta import AlgorithmLib.Surface.LibWindow
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# The window contracts, checked without a display

What `Sem.callFfi` says the five window entry points do when there is no
display, run through `base/tests/hprog_window_corpus.rs` with the display
variables removed and compared byte for byte with what the interpreter computed
here. Without a display `init` stores null over whatever the slot held, and
every call after it takes a refusal path; those paths, and the argument checks
that come before the context is read, are what this checks. What happens with a
window open depends on the oracles (`display`, `windowOpens`, `winInput`) and
on a person at the keyboard, and is not checked here.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgWindowCorpus

def SLOT : Nat := 0
def TITLE : Nat := 0x40
def MEM : Nat := 0x100
def OUT : Nat := 160

def image : List UInt8 :=
  let a := List.replicate TITLE 0 ++ stringToBytes "corpus"
  a ++ List.replicate (MEM - a.length) 0

def body : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  let code (k : Nat) (r : V .i32) : Prog V L Unit := do
    storeI32 r (← iadd out (← iconst64 (4 * k)))
  let slot ← iadd ptr (← iconst64 SLOT)
  store (← iconst64 0x1234) slot
  ffiVoid .windowInit %[slot]
  let c ← load64 slot
  store c (← iadd out (← iconst64 128))
  let n64 (k : Int) : Prog V L (V .i64) := iconst64 k
  let title ← iadd ptr (← iconst64 TITLE)
  code 0 (← ffi .windowOpen %[c, ← n64 64, ← n64 64, title, ← n64 6, title, ← n64 6])
  code 1 (← ffi .windowOpen %[c, ← n64 0, ← n64 64, title, ← n64 6, title, ← n64 6])
  code 2 (← ffi .windowOpen %[c, ← n64 64, ← n64 64, title, ← n64 6, title, ← n64 0])
  let events ← iadd out (← iconst64 64)
  code 3 (← ffi .windowPoll %[c, events, ← iconst32 1])
  code 4 (← ffi .windowPoll %[c, events, ← iconst32 (-1)])
  code 5 (← ffi .windowPoll %[c, ← n64 0, ← iconst32 1])
  code 6 (← ffi .windowPresentGpuBuffer %[c, ← n64 0, ← iconst32 (-1)])
  code 7 (← ffi .windowPresentGpuBuffer %[c, ← n64 0, ← iconst32 0])
  ffiVoid .windowCleanup %[slot]
  store (← iconst64 0x5678) slot
  ffiVoid .windowInit %[slot]
  store (← load64 slot) (← iadd out (← iconst64 136))
  ffiVoid .windowCleanup %[slot]

def code : Code := Prog.emit body
def checked : Except String Code := Prog.emitChecked body
def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]
def env : FnEnv := (Prog.run body).2.1

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
   .sc .i64 0, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

def startWorld : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ByteArray.empty,
             out := ByteArray.mk (Array.replicate OUT 0) } }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

-- ---------------------------------------------------------------------------
-- With a display
-- ---------------------------------------------------------------------------

/-- A window of `W × H` pixels showing a gradient: at `(x, y)` red `4x`,
    green `5y`, blue `128`, opaque. The test reads the window back from the
    X server's framebuffer while it is still open, which is why this body
    leaves it open. -/
def W : Nat := 64
def H : Nat := 48

def D_WSLOT : Nat := 0x00
def D_GSLOT : Nat := 0x08
def D_TITLE : Nat := 0x40
def D_BLIT : Nat := 0x100
def D_PIXELS : Nat := 0x1000
def D_MEM : Nat := D_PIXELS + 4 * W * H
def D_OUT : Nat := 4 * 11

def blit : String :=
  "@group(0) @binding(0) var<storage, read> pixels: array<u32>;\n" ++
  "struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };\n" ++
  "@vertex\nfn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {\n" ++
  "  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));\n" ++
  "  var u = array<vec2<f32>, 3>(vec2<f32>(0.0, 2.0), vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0));\n" ++
  "  var o: VsOut;\n  o.pos = vec4<f32>(p[idx], 0.0, 1.0);\n  o.uv = u[idx];\n  return o;\n}\n" ++
  "@fragment\nfn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n" ++
  "  let px = min(u32(in.uv.x * " ++ toString W ++ ".0), " ++ toString (W - 1) ++ "u);\n" ++
  "  let py = min(u32(in.uv.y * " ++ toString H ++ ".0), " ++ toString (H - 1) ++ "u);\n" ++
  "  let c = pixels[py * " ++ toString W ++ "u + px];\n" ++
  "  return vec4<f32>(f32(c & 0xFFu), f32((c >> 8u) & 0xFFu), f32((c >> 16u) & 0xFFu), 255.0) / 255.0;\n}\n"

/-- A shader that is not UTF-8: `C0 AF` is an overlong form. -/
def D_BADBLIT : Nat := 0x80
def badBlit : List UInt8 := [0x66, 0xC0, 0xAF, 0x0A]

def displayImage : List UInt8 :=
  let pad (xs : List UInt8) (n : Nat) := xs ++ List.replicate (n - xs.length) 0
  let px := (List.range H).flatMap fun y => (List.range W).flatMap fun x =>
    [(4 * x).toUInt8, (5 * y).toUInt8, 128, 255]
  pad (pad (pad (List.replicate D_TITLE 0 ++ stringToBytes "corpus") D_BADBLIT ++ badBlit) D_BLIT
    ++ stringToBytes blit) D_PIXELS ++ px

def displayBody : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  let code (k : Nat) (r : V .i32) : Prog V L Unit := do
    storeI32 r (← iadd out (← iconst64 (4 * k)))
  let flag (k : Nat) (c : V .i8) : Prog V L Unit := do code k (← ireduce32 (← uextend64 c))
  let n64 (k : Int) : Prog V L (V .i64) := iconst64 k
  let wslot ← iadd ptr (← n64 D_WSLOT)
  let gslot ← iadd ptr (← n64 D_GSLOT)
  ffiVoid .windowInit %[wslot]
  ffiVoid .gpuInit %[gslot]
  let wc ← load64 wslot
  let gc ← load64 gslot
  flag 0 (← icmp .ne wc (← n64 0))
  flag 1 (← icmp .ne gc (← n64 0))
  let title ← iadd ptr (← n64 D_TITLE)
  let blitAt ← iadd ptr (← n64 D_BLIT)
  code 10 (← ffi .windowOpen %[wc, ← n64 W, ← n64 H, title, ← n64 6, ← iadd ptr (← n64 D_BADBLIT),
    ← n64 badBlit.length])
  code 2 (← ffi .windowOpen %[wc, ← n64 W, ← n64 H, title, ← n64 6, blitAt, ← n64 blit.utf8ByteSize])
  let size ← n64 (4 * W * H)
  let buf ← ffi .gpuCreateBuffer %[gc, size]
  code 3 buf
  code 4 (← ffi .gpuUpload %[gc, buf, ← iadd ptr (← n64 D_PIXELS), size])
  code 5 (← ffi .windowPresentGpuBuffer %[wc, gc, buf])
  code 6 (← ffi .windowPresentGpuBuffer %[wc, gc, buf])
  code 7 (← ffi .windowPresentGpuBuffer %[wc, gc, buf])
  let events ← iadd out (← n64 D_OUT)
  flag 8 (← icmp .sge (← ffi .windowPoll %[wc, events, ← iconst32 0]) (← iconst32 0))
  code 9 (← ffi .windowOpen %[wc, ← n64 W, ← n64 H, title, ← n64 6, blitAt, ← n64 blit.utf8ByteSize])

def displayProgram : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 displayBody)]
def displayChecked : Except String Code := Prog.emitChecked displayBody

def displayExpected : Except String ByteArray :=
  let env := (Prog.run displayBody).2.1
  let args : List Sem.V :=
    [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
     .sc .i64 0, .sc .i64 (Sem.regionBase .out), .sc .i64 D_OUT.toUInt64]
  let w : Sem.World :=
    { mem := { arena := ⟨displayImage.toArray⟩, data := ByteArray.empty,
               out := ByteArray.mk (Array.replicate D_OUT 0) },
      display := true }
  match Sem.run { env } args w (Prog.emit displayBody) with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

/-- `Lib.Window`'s UTF-8 check, run by the interpreter over `n` bytes at the
    arena: whether it finds them not UTF-8. -/
def utf8Body (n : Nat) : Prog V L Unit := do
  let b ← LibWindow.utf8Bad (← basePtr) (← iconst64 n)
  storeI32 (← select b (← iconst32 1) (← iconst32 0)) (← outPtr)

def utf8Refuses (bs : List UInt8) : Option Bool :=
  let body := utf8Body bs.length
  let args : List Sem.V := [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
     .sc .i64 0, .sc .i64 (Sem.regionBase .out), .sc .i64 4]
  let arena : ByteArray := ⟨(bs ++ List.replicate 8 0).toArray⟩
  let w : Sem.World := { mem := { arena := arena, data := ByteArray.empty, out := ByteArray.mk (Array.replicate 4 0) } }
  match Sem.run { env := (Prog.run body).2.1 } args w (Prog.emit body) with
  | .ok _ w => some (w.mem.out.get! 0 != 0)
  | _ => none

/-- Every single byte, and sequences of two to four around each boundary of
    the lead and continuation ranges. -/
def utf8Cases : List (List UInt8) :=
  let conts : List UInt8 := [0x00, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0, 0xFF]
  let leads : List UInt8 := [0x00, 0x41, 0x7F, 0x80, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xE1, 0xEC,
    0xED, 0xEE, 0xEF, 0xF0, 0xF1, 0xF3, 0xF4, 0xF5, 0xFF]
  (List.range 256).map (fun i => [i.toUInt8]) ++
  leads.flatMap (fun a => conts.map fun b => [a, b]) ++
  leads.flatMap (fun a => conts.flatMap fun b => conts.map fun c => [a, b, c]) ++
  [0xF0, 0xF1, 0xF4, 0xF5].flatMap (fun a => conts.flatMap fun b => conts.flatMap fun c =>
    [0x80, 0xBF, 0xC0].map fun d => [a, b, c, d]) ++
  [[0x41, 0xE2, 0x82], [0xE2, 0x82, 0xAC, 0x41], [0xF0, 0x9F, 0x98, 0x80, 0x41]]

/-- The cases where the check and `String.fromUTF8?`, which the model uses,
    disagree. -/
def utf8Disagreements : List (List UInt8) :=
  utf8Cases.filter fun c => utf8Refuses c != some (String.fromUTF8? ⟨c.toArray⟩).isNone

end HProgWindowCorpus

open AlgorithmLib in
def Host.WindowCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgWindowCorpus.utf8Disagreements with
  | [] => pure ()
  | c :: _ => throw (IO.userError s!"Lib.Window's UTF-8 check disagrees with the model on {c}")
  match HProgWindowCorpus.checked with
  | .error e => throw (IO.userError s!"the window corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgWindowCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the window corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgWindowCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_window_corpus" {
        functions := clif, required_memory := HProgWindowCorpus.MEM,
        initial_memory := HProgWindowCorpus.image
      }]
      let sideDir := System.FilePath.mk dir / "hprog_window_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat)))]).compress
      IO.println s!"window corpus: {bytes.size} expected bytes"
  match HProgWindowCorpus.displayChecked, HProgWindowCorpus.displayExpected with
  | .error e, _ => throw (IO.userError s!"the window display corpus body is not well-formed: {e}")
  | _, .error e => throw (IO.userError s!"interpreting the window display corpus: {e}")
  | .ok _, .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgWindowCorpus.displayProgram
      emitArtifacts dir #[artifactEntry "hprog_window_display_corpus" {
        functions := clif, required_memory := HProgWindowCorpus.D_MEM,
        initial_memory := HProgWindowCorpus.displayImage
      }]
      let sideDir := System.FilePath.mk dir / "hprog_window_display_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat))),
                          ("width", Lean.toJson HProgWindowCorpus.W),
                          ("height", Lean.toJson HProgWindowCorpus.H)]).compress
      IO.println s!"window display corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.WindowCorpus" `Host.WindowCorpus.main
