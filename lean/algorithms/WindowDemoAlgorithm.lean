import AlgorithmLib

set_option maxRecDepth 8192

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout
open AlgorithmLib.WGSL

namespace Algorithm

-- Screen / framebuffer -------------------------------------------------------
def imageWidth  : Nat := 640
def imageHeight : Nat := 360
def pixelBytes  : Nat := imageWidth * imageHeight * 4
def paramsBytes : Nat := 32
def eventSlots  : Nat := 64
def eventBytes  : Nat := eventSlots * 32
def wgX : Nat := (imageWidth + 7) / 8
def wgY : Nat := (imageHeight + 7) / 8

-- Player ---------------------------------------------------------------------
def playerSpeed  : Int := 4
def playerStartX : Int := (imageWidth / 2 : Nat)
def playerStartY : Int := (imageHeight / 2 : Nat)
def minX : Int := 0
def maxX : Int := imageWidth - 1
def minY : Int := 0
def maxY : Int := imageHeight - 1

-- Event kinds + key ids (mirror base/src/ffi/window.rs) ----------------------
def evClose   : Int := 1
def evKeyDown : Int := 3
def evKeyUp   : Int := 4
def keyEscape : Int := 1
def keyLeft   : Int := 3
def keyRight  : Int := 4
def keyUp     : Int := 5
def keyDown   : Int := 6
-- Held-key bitmask (persisted across frames in `keyMask`)
def bitLeft  : Int := 1
def bitRight : Int := 2
def bitUp    : Int := 4
def bitDown  : Int := 8

def titleText : String := "Base Game"

def shaderSource : String :=
  let pixels : Expr (.arr .u32) := ⟨"pixels"⟩
  let params : Expr (.arr .u32) := ⟨"params"⟩
  let tau := litF "6.2831853"
  buildShader
    [ { binding := 0, name := "pixels", ty := .arr .u32, ro := false },
      { binding := 1, name := "params", ty := .arr .u32, ro := true } ]
    []
    []
    { name := "main", wgX := 8, wgY := 8 }
    (do
      let width ← letV "width" (arrIdx params (litU 1))
      let height ← letV "height" (arrIdx params (litU 2))
      let x ← letV "x" gidX
      let y ← letV "y" gidY
      ifB ((x .>= width) .|| (y .>= height)) retV
      let playerX ← letV "player_x" (i32OfU (arrIdx params (litU 3)))
      let playerY ← letV "player_y" (i32OfU (arrIdx params (litU 4)))
      let frame ← letV "frame" (f32OfU (arrIdx params (litU 0)) * litF "0.02")
      let fx ← letV "fx" (f32OfU x / f32OfU width)
      let fy ← letV "fy" (f32OfU y / f32OfU height)
      let wave ← letV "wave" (litF "0.5" + litF "0.5" * wSin (tau * (fx * litF "0.8" + frame)))
      let r ← varV "r" (litF "0.08" + litF "0.15" * wave)
      let g ← varV "g" (litF "0.10" + litF "0.20" *
        (litF "0.5" + litF "0.5" * wSin (tau * (fy + frame * litF "0.7"))))
      let b ← varV "b" (litF "0.18" + litF "0.25" *
        (litF "0.5" + litF "0.5" * wSin (tau * (fx + fy + frame * litF "0.35"))))
      let dx ← letV "dx" (i32OfU x - playerX)
      let dy ← letV "dy" (i32OfU y - playerY)
      ifB ((wAbsI dx .<= litI 10) .&& (wAbsI dy .<= litI 10)) (do
        assign r (litF "0.95"); assign g (litF "0.92"); assign b (litF "0.25")
        ifB ((wAbsI dx .<= litI 7) .&& (wAbsI dy .<= litI 7)) (do
          assign r (litF "0.99"); assign g (litF "0.35"); assign b (litF "0.18")))
      let ri ← letV "ri" (u32OfF (wClamp (r * litF "255.0") (litF "0.0") (litF "255.0")))
      let gi ← letV "gi" (u32OfF (wClamp (g * litF "255.0") (litF "0.0") (litF "255.0")))
      let bi ← letV "bi" (u32OfF (wClamp (b * litF "255.0") (litF "0.0") (litF "255.0")))
      let idx ← letV "idx" (y * width + x)
      assign (arrIdx pixels idx) (((ri .| (gi .<< litU 8)) .| (bi .<< litU 16)) .| (litU 0xFF .<< litU 24)))

/-- Present blit shader (Lean-generated, so *all* WGSL lives here). A fullscreen
    triangle samples the packed-RGBA framebuffer at `@binding(0)` and stretches it
    to the window. The framebuffer dimensions are baked in as literals (known at
    generation time), so there is no `dims` uniform — the only contract with the
    Rust present appliance is: one storage buffer at binding 0, `vs_main`/`fs_main`. -/
def blitShaderSource : String :=
  let w := toString imageWidth
  let h := toString imageHeight
  "@group(0) @binding(0) var<storage, read> pixels: array<u32>;\n\n" ++
  "struct VsOut {\n" ++
  "  @builtin(position) pos: vec4<f32>,\n" ++
  "  @location(0) uv: vec2<f32>,\n" ++
  "};\n\n" ++
  "@vertex\n" ++
  "fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {\n" ++
  "  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));\n" ++
  "  var u = array<vec2<f32>, 3>(vec2<f32>(0.0, 2.0), vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0));\n" ++
  "  var o: VsOut;\n" ++
  "  o.pos = vec4<f32>(p[idx], 0.0, 1.0);\n" ++
  "  o.uv = u[idx];\n" ++
  "  return o;\n" ++
  "}\n\n" ++
  "@fragment\n" ++
  "fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n" ++
  "  let w: u32 = " ++ w ++ "u;\n" ++
  "  let h: u32 = " ++ h ++ "u;\n" ++
  "  let px = min(u32(in.uv.x * f32(w)), w - 1u);\n" ++
  "  let py = min(u32(in.uv.y * f32(h)), h - 1u);\n" ++
  "  let packed = pixels[py * w + px];\n" ++
  "  let r = f32(packed & 0xFFu) / 255.0;\n" ++
  "  let g = f32((packed >> 8u) & 0xFFu) / 255.0;\n" ++
  "  let b = f32((packed >> 16u) & 0xFFu) / 255.0;\n" ++
  "  let a = f32((packed >> 24u) & 0xFFu) / 255.0;\n" ++
  "  return vec4<f32>(r, g, b, a);\n" ++
  "}\n"

-- Memory layout --------------------------------------------------------------
-- `reserved` covers 0x00..0x40 (ht/wgpu/cuda ctx slots, the IoOffsets region,
-- and the window ctx slot at 0x38). All game state is i64 for clean 8-byte
-- load/store; only `params` (the GPU uniform) is packed u32.
structure Fields where
  reserved    : Fld (.bytes 64)
  bindDesc    : Fld (.bytes 16)
  shader      : Fld (.bytes 4096)
  blitShader  : Fld (.bytes 2048)
  title       : Fld (.bytes 64)
  events      : Fld (.bytes eventBytes)
  params      : Fld (.bytes paramsBytes)
  keyMask     : Fld .i64
  quit        : Fld .i64
  playerX     : Fld .i64
  playerY     : Fld .i64
  nEvents     : Fld .i64
  rowCount    : Fld .i64
  outPass     : Fld .i64
  outActual   : Fld .i64
  outExpected : Fld .i64
  pixels      : Fld (.bytes pixelBytes)

def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved    ← field (.bytes 64)
  let bindDesc    ← field (.bytes 16)
  let shader      ← field (.bytes 4096)
  let blitShader  ← field (.bytes 2048)
  let title       ← field (.bytes 64)
  let events      ← field (.bytes eventBytes)
  let params      ← field (.bytes paramsBytes)
  let keyMask     ← field .i64
  let quit        ← field .i64
  let playerX     ← field .i64
  let playerY     ← field .i64
  let nEvents     ← field .i64
  let rowCount    ← field .i64
  let outPass     ← field .i64
  let outActual   ← field .i64
  let outExpected ← field .i64
  let pixels      ← field (.bytes pixelBytes)
  pure { reserved, bindDesc, shader, blitShader, title, events, params, keyMask, quit,
         playerX, playerY, nEvents, rowCount, outPass, outActual,
         outExpected, pixels }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

open AlgorithmLib.IR

-- Game-logic helpers (emit inline CLIF; shared by the live loop and tests) ---

/-- Put the player at (x, y) and clear held-keys / quit / frame. -/
def clearState (ptr : Val) (x y : Int) : IRBuilder Unit := do
  let z ← iconst64 0
  fldStore ptr f.keyMask z
  fldStore ptr f.quit z
  fldStore ptr f.playerX (← iconst64 x)
  fldStore ptr f.playerY (← iconst64 y)

/-- Scan events [0, nEvents): update the held-key bitmask + quit flag. Branchless
    (no per-event blocks): each key contributes a bit that is OR'd in on key-down
    and cleared on key-up, so holding a key persists across frames. -/
def processEvents (ptr : Val) : IRBuilder Unit := do
  let evBase ← iadd ptr (← fldOffset f.events)
  let n ← fldLoad ptr f.nEvents
  let recSz ← iconst64 32
  forLoop .i64 n (fun i => do
    let base ← iadd evBase (← imul i recSz)
    let kind ← load64 base
    let keyCode ← load64 (← iaddImm base 8)
    let isDown  ← sextend64 (← icmp .eq kind (← iconst64 evKeyDown))
    let isUp    ← sextend64 (← icmp .eq kind (← iconst64 evKeyUp))
    let isClose ← sextend64 (← icmp .eq kind (← iconst64 evClose))
    let isL ← sextend64 (← icmp .eq keyCode (← iconst64 keyLeft))
    let isR ← sextend64 (← icmp .eq keyCode (← iconst64 keyRight))
    let isU ← sextend64 (← icmp .eq keyCode (← iconst64 keyUp))
    let isD ← sextend64 (← icmp .eq keyCode (← iconst64 keyDown))
    let isEsc ← sextend64 (← icmp .eq keyCode (← iconst64 keyEscape))
    let bit ← iadd (← iadd (← imul isL (← iconst64 bitLeft)) (← imul isR (← iconst64 bitRight)))
                   (← iadd (← imul isU (← iconst64 bitUp)) (← imul isD (← iconst64 bitDown)))
    let mask0 ← fldLoad ptr f.keyMask
    let mask1 ← bor mask0 (← imul bit isDown)
    let mask2 ← bandNot mask1 (← imul bit isUp)
    fldStore ptr f.keyMask mask2
    let q0 ← fldLoad ptr f.quit
    let q1 ← bor q0 (← bor isClose (← imul isDown isEsc))
    fldStore ptr f.quit q1)

/-- Clamp field `fld` into [lo, hi] in place. Branchless. -/
def clampField (ptr : Val) (fld : Fld .i64) (lo hi : Int) : IRBuilder Unit := do
  let x ← fldLoad ptr fld
  let over ← sextend64 (← icmp .sgt x (← iconst64 hi))
  let x1 ← iadd x (← imul over (← isub (← iconst64 hi) x))
  let under ← sextend64 (← icmp .slt x1 (← iconst64 lo))
  let x2 ← iadd x1 (← imul under (← isub (← iconst64 lo) x1))
  fldStore ptr fld x2

/-- Move the player one step from the held-key mask, then clamp to the screen. -/
def applyMovement (ptr : Val) : IRBuilder Unit := do
  let mask ← fldLoad ptr f.keyMask
  let speed ← iconst64 playerSpeed
  let leftOn  ← sextend64 (← icmp .ne (← band mask (← iconst64 bitLeft)) (← iconst64 0))
  let rightOn ← sextend64 (← icmp .ne (← band mask (← iconst64 bitRight)) (← iconst64 0))
  let upOn    ← sextend64 (← icmp .ne (← band mask (← iconst64 bitUp)) (← iconst64 0))
  let downOn  ← sextend64 (← icmp .ne (← band mask (← iconst64 bitDown)) (← iconst64 0))
  let dx ← imul (← isub rightOn leftOn) speed
  let dy ← imul (← isub downOn upOn) speed
  fldStore ptr f.playerX (← iadd (← fldLoad ptr f.playerX) dx)
  fldStore ptr f.playerY (← iadd (← fldLoad ptr f.playerY) dy)
  clampField ptr f.playerX minX maxX
  clampField ptr f.playerY minY maxY

/-- Write the per-frame uniform (frame, player x/y) into the packed-u32 region.
    Width/height stay from the initial payload. -/
def writeParams (ptr : Val) (frame : Val) : IRBuilder Unit := do
  fldStore32At ptr f.params 0 (← ireduce32 frame) (by decide)
  fldStore32At ptr f.params 12 (← ireduce32 (← fldLoad ptr f.playerX)) (by decide)
  fldStore32At ptr f.params 16 (← ireduce32 (← fldLoad ptr f.playerY)) (by decide)

/-- Write a synthetic input event into event `slot` (for tests). -/
def writeEvent (ptr : Val) (slot : Nat) (kind keyCode : Int) : IRBuilder Unit := do
  let base ← absAddr ptr (f.events.offset + slot * 32)
  store (← iconst64 kind) base
  store (← iconst64 keyCode) (← iaddImm base 8)

/-- Emit one output row: pass (1/0), actual, expected. -/
def writeOutput (ptr : Val) (passV actualV expectedV : Val) : IRBuilder Unit := do
  fldStore ptr f.rowCount (← iconst64 1)
  fldStore ptr f.outPass passV
  fldStore ptr f.outActual actualV
  fldStore ptr f.outExpected expectedV

/-- Run the shared logic `steps` times (each step re-scans events, so a held key
    keeps moving — exactly what the live loop does with real poll results). -/
def stepN (ptr : Val) (steps : Int) : IRBuilder Unit := do
  let limit ← iconst64 steps
  forLoop .i64 limit (fun _ => do
    processEvents ptr
    applyMovement ptr)

/-- Assert `actual == expected`; store pass/actual/expected as the output row. -/
def assertEq (ptr actual : Val) (expected : Int) : IRBuilder Unit := do
  let exp ← iconst64 expected
  writeOutput ptr (← sextend64 (← icmp .eq actual exp)) actual exp

-- Live entry (fn 1): open window, then loop poll → logic → render → present ----
def mainBody : IRBuilder Unit := do
  let gpu ← declareGpuFFI
  let win ← declareWindowFFI
  let ptr ← entryBlock
  windowInit win ptr
  gpuInit gpu ptr
  let pixelBuf ← gpuCreateBuffer gpu ptr (← iconst64 pixelBytes)
  let paramBuf ← gpuCreateBuffer gpu ptr (← iconst64 paramsBytes)
  let pipeId ← gpuCreatePipeline gpu ptr (← fldOffset f.shader) (← fldOffset f.bindDesc) (← iconst32 2)
  let w64 ← iconst64 imageWidth
  let h64 ← iconst64 imageHeight
  let _ ← windowOpen win ptr w64 h64 (← fldOffset f.title) (← iconst64 (titleText.length : Int))
                      (← fldOffset f.blitShader) (← iconst64 (blitShaderSource.length : Int))
  clearState ptr playerStartX playerStartY
  let _ ← whileLoop1 .i64 (← iconst64 0)
    (fun _ => do icmp .eq (← fldLoad ptr f.quit) (← iconst64 0))
    (fun frame => do
      let n ← windowPoll win ptr (← fldOffset f.events) (← iconst32 eventSlots)
      fldStore ptr f.nEvents (← sextend64 n)
      processEvents ptr
      applyMovement ptr
      writeParams ptr frame
      let _ ← gpuUpload gpu ptr paramBuf (← fldOffset f.params) (← iconst64 paramsBytes)
      let _ ← gpuDispatch gpu ptr pipeId (← iconst32 wgX) (← iconst32 wgY) (← iconst32 1)
      let _ ← windowPresentGpuBuffer win ptr pixelBuf
      iaddImm frame 1)
  windowCleanup win ptr
  gpuCleanup gpu ptr
  ret

-- Headless test scenarios (fn 2+): inject events, step, assert player state ----
def testMoveRight : IRBuilder Unit := do
  let ptr ← entryBlock
  clearState ptr playerStartX playerStartY
  writeEvent ptr 0 evKeyDown keyRight
  fldStore ptr f.nEvents (← iconst64 1)
  stepN ptr 10
  assertEq ptr (← fldLoad ptr f.playerX) (playerStartX + 10 * playerSpeed)
  ret

def testMoveLeft : IRBuilder Unit := do
  let ptr ← entryBlock
  clearState ptr playerStartX playerStartY
  writeEvent ptr 0 evKeyDown keyLeft
  fldStore ptr f.nEvents (← iconst64 1)
  stepN ptr 10
  assertEq ptr (← fldLoad ptr f.playerX) (playerStartX - 10 * playerSpeed)
  ret

def testMoveUpClamp : IRBuilder Unit := do
  let ptr ← entryBlock
  clearState ptr playerStartX playerStartY
  writeEvent ptr 0 evKeyDown keyUp
  fldStore ptr f.nEvents (← iconst64 1)
  stepN ptr 100   -- 100*speed = 400 > startY 180 ⇒ clamps at the top edge (0)
  assertEq ptr (← fldLoad ptr f.playerY) minY
  ret

def testQuitOnClose : IRBuilder Unit := do
  let ptr ← entryBlock
  clearState ptr playerStartX playerStartY
  writeEvent ptr 0 evClose 0
  fldStore ptr f.nEvents (← iconst64 1)
  processEvents ptr
  assertEq ptr (← fldLoad ptr f.quit) 1
  ret

-- Headless render test (fn 6): dispatch the real WGSL kernel, download the frame
-- into memory, and assert the player pixel is player-coloured. GPU, no window.
def testRenderPixel : IRBuilder Unit := do
  let gpu ← declareGpuFFI
  let ptr ← entryBlock
  gpuInit gpu ptr
  let pixelBuf ← gpuCreateBuffer gpu ptr (← iconst64 pixelBytes)
  let paramBuf ← gpuCreateBuffer gpu ptr (← iconst64 paramsBytes)
  let pipeId ← gpuCreatePipeline gpu ptr (← fldOffset f.shader) (← fldOffset f.bindDesc) (← iconst32 2)
  clearState ptr 100 100
  writeParams ptr (← iconst64 0)
  let _ ← gpuUpload gpu ptr paramBuf (← fldOffset f.params) (← iconst64 paramsBytes)
  let _ ← gpuDispatch gpu ptr pipeId (← iconst32 wgX) (← iconst32 wgY) (← iconst32 1)
  let _ ← gpuDownload gpu ptr pixelBuf (← fldOffset f.pixels) (← iconst64 pixelBytes)
  gpuCleanup gpu ptr
  -- red byte of pixel (100,100): (y*width + x)*4
  let red ← uload8_64 (← absAddr ptr (f.pixels.offset + (100 * imageWidth + 100) * 4))
  writeOutput ptr (← sextend64 (← icmp .uge red (← iconst64 250))) red (← iconst64 252)
  ret

-- Program assembly -----------------------------------------------------------
def clifIrSource : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 mainBody ++ "\n" ++
  buildFunction 2 testMoveRight ++ "\n" ++
  buildFunction 3 testMoveLeft ++ "\n" ++
  buildFunction 4 testMoveUpClamp ++ "\n" ++
  buildFunction 5 testQuitOnClose ++ "\n" ++
  buildFunction 6 testRenderPixel

def payloads : List UInt8 :=
  mkPayload layoutMeta.totalSize [
    -- binding0 → buf 0 (pixels, read_write); binding1 → buf 1 (params, read_only)
    f.bindDesc.init (uint32ToBytes 0 ++ uint32ToBytes 0 ++ uint32ToBytes 1 ++ uint32ToBytes 1),
    f.shader.init (stringToBytes shaderSource),
    f.blitShader.init (stringToBytes blitShaderSource),
    f.title.init (stringToBytes titleText),
    -- params: frame, width, height, player x/y (x/y are overwritten each frame)
    f.params.init (
      uint32ToBytes 0 ++
      uint32ToBytes (UInt32.ofNat imageWidth) ++
      uint32ToBytes (UInt32.ofNat imageHeight) ++
      uint32ToBytes (UInt32.ofNat (imageWidth / 2)) ++
      uint32ToBytes (UInt32.ofNat (imageHeight / 2)))
  ]

def gameSetup : Setup := {
  cranelift_ir := clifIrSource,
  memory_size := layoutMeta.totalSize,
  initial_memory := payloads
}

-- Every test writes the same 3-column output row.
def testSchema : List Json :=
  [Output.schema
    [ Output.column "pass" .i64 f.outPass.offset,
      Output.column "actual" .i64 f.outActual.offset,
      Output.column "expected" .i64 f.outExpected.offset ]
    f.rowCount.offset]

def mainAlgorithm : Algorithm := { fn_idx := IR.mainFnIdx }
def moveRightAlg  : Algorithm := { fn_idx := u32 2, output := testSchema }
def moveLeftAlg   : Algorithm := { fn_idx := u32 3, output := testSchema }
def moveUpClampAlg : Algorithm := { fn_idx := u32 4, output := testSchema }
def quitOnCloseAlg : Algorithm := { fn_idx := u32 5, output := testSchema }
def renderPixelAlg : Algorithm := { fn_idx := u32 6, output := testSchema }

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← AlgorithmLib.requireOutputDir args
  AlgorithmLib.emitArtifacts outDir #[
    AlgorithmLib.toJsonArtifact "window_demo" Algorithm.gameSetup Algorithm.mainAlgorithm [
      ("test_move_right",   Algorithm.moveRightAlg),
      ("test_move_left",    Algorithm.moveLeftAlg),
      ("test_move_up_clamp", Algorithm.moveUpClampAlg),
      ("test_quit_on_close", Algorithm.quitOnCloseAlg),
      ("test_render_pixel", Algorithm.renderPixelAlg)
    ]]
