import AlgorithmLib

set_option maxRecDepth 8192

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout
open AlgorithmLib.WGSL

namespace Algorithm

def imageWidth  : Nat := 640
def imageHeight : Nat := 360
def pixelBytes  : Nat := imageWidth * imageHeight * 4
def cellPx : Nat := 1
def gw : Nat := imageWidth / cellPx
def gh : Nat := imageHeight / cellPx
def gridCells : Nat := gw * gh
def gridBytes : Nat := gridCells * 4

def EMPTY : Nat := 0
def SAND  : Nat := 1
def WALL  : Nat := 2

def sandX0 : Nat := 120
def sandX1 : Nat := 200
def sandY0 : Nat := 16
def sandY1 : Nat := 60
def shelfY : Nat := 120
def gapX0  : Nat := 148
def gapX1  : Nat := 172
def brushR : Nat := 8

-- params buffer word indices
def pPARITY : Nat := 0
def pFRAME  : Nat := 1
def pMOUSEX : Nat := 2
def pMOUSEY : Nat := 3
def pBDOWN  : Nat := 4
def pBMAT   : Nat := 5
def pBR     : Nat := 6

def gridWgX : Nat := (gw + 7) / 8
def gridWgY : Nat := (gh + 7) / 8
def renderWgX : Nat := (imageWidth + 7) / 8
def renderWgY : Nat := (imageHeight + 7) / 8

def eventSlots : Nat := 64
def eventBytes : Nat := eventSlots * 32
def evClose   : Int := 1
def evKeyDown : Int := 3
def evMouseMove : Int := 5
def evMouseDown : Int := 6
def evMouseUp   : Int := 7
def keyEscape : Int := 1
def titleText : String := "Base Sand"

def stepShader : String :=
  let gridIn  : Expr (.arr .u32) := ⟨"gridIn"⟩
  let gridOut : Expr (.arr .u32) := ⟨"gridOut"⟩
  let params  : Expr (.arr .u32) := ⟨"params"⟩
  let gwE : Expr .u32 := ⟨"GW"⟩
  let ghE : Expr .u32 := ⟨"GH"⟩
  let readAt := fun (cx cy : Expr .u32) =>
    let inb := (cx .< gwE) .&& (cy .< ghE)
    let idx := wSelect (litU 0) (cy * gwE + cx) inb
    wSelect (litU WALL) (arrIdx gridIn idx) inb
  let hashbit := fun (a b c : Expr .u32) =>
    let h0 := (a * litU 1597334677) + (b * litU 3812015801) + (c * litU 2654435761)
    let h1 := (bxorU h0 (shrU h0 (litU 15))) * litU 2246822519
    bandU (bxorU h1 (shrU h1 (litU 13))) (litU 1)
  buildShader
    [ { binding := 0, name := "gridIn",  ty := .arr .u32, ro := true },
      { binding := 1, name := "gridOut", ty := .arr .u32, ro := false },
      { binding := 2, name := "params",  ty := .arr .u32, ro := true } ]
    []
    [.constU "GW" gw, .constU "GH" gh]
    { name := "main", wgX := 8, wgY := 8 }
    (do
      let x ← letV gidX
      let y ← letV gidY
      ifB ((x .>= gwE) .|| (y .>= ghE)) retV
      let par ← letV (arrIdx params (litU pPARITY))
      let frame ← letV (arrIdx params (litU pFRAME))
      let idx ← letV (y * gwE + x)
      ifElse ((x .< par) .|| (y .< par))
        (assign (arrIdx gridOut idx) (arrIdx gridIn idx))
        (do
          let lx ← letV ((x - par) % litU 2)
          let ly ← letV ((y - par) % litU 2)
          let cx0 ← letV (x - lx)
          let cy0 ← letV (y - ly)
          let tl ← varV (readAt cx0 cy0)
          let tr ← varV (readAt (cx0 + litU 1) cy0)
          let bl ← varV (readAt cx0 (cy0 + litU 1))
          let br ← varV (readAt (cx0 + litU 1) (cy0 + litU 1))
          let rnd ← letV (hashbit cx0 cy0 frame)
          ifB ((tl .== litU SAND) .&& (bl .== litU EMPTY))
            (ifElse ((rnd .== litU 1) .&& (br .== litU EMPTY))
              (do assign tl (litU EMPTY); assign br (litU SAND))
              (do assign tl (litU EMPTY); assign bl (litU SAND)))
          ifB ((tr .== litU SAND) .&& (br .== litU EMPTY))
            (ifElse ((rnd .== litU 0) .&& (bl .== litU EMPTY))
              (do assign tr (litU EMPTY); assign bl (litU SAND))
              (do assign tr (litU EMPTY); assign br (litU SAND)))
          ifB ((tl .== litU SAND) .&& (neE bl (litU EMPTY)) .&& (br .== litU EMPTY))
            (do assign tl (litU EMPTY); assign br (litU SAND))
          ifB ((tr .== litU SAND) .&& (neE br (litU EMPTY)) .&& (bl .== litU EMPTY))
            (do assign tr (litU EMPTY); assign bl (litU SAND))
          let top := wSelect tr tl (lx .== litU 0)
          let bot := wSelect br bl (lx .== litU 0)
          assign (arrIdx gridOut idx) (wSelect bot top (ly .== litU 0))))

def paintShader : String :=
  let grid   : Expr (.arr .u32) := ⟨"grid"⟩
  let params : Expr (.arr .u32) := ⟨"params"⟩
  let gwE  : Expr .u32 := ⟨"GW"⟩
  let ghE  : Expr .u32 := ⟨"GH"⟩
  let cell : Expr .u32 := ⟨"CELL"⟩
  buildShader
    [ { binding := 0, name := "grid",   ty := .arr .u32, ro := false },
      { binding := 1, name := "params", ty := .arr .u32, ro := true } ]
    []
    [.constU "GW" gw, .constU "GH" gh, .constU "CELL" cellPx]
    { name := "main", wgX := 8, wgY := 8 }
    (do
      let x ← letV gidX
      let y ← letV gidY
      ifB ((x .>= gwE) .|| (y .>= ghE)) retV
      ifB (arrIdx params (litU pBDOWN) .== litU 0) retV
      let mcx ← letV (arrIdx params (litU pMOUSEX) / cell)
      let mcy ← letV (arrIdx params (litU pMOUSEY) / cell)
      let r ← letV (i32OfU (arrIdx params (litU pBR)))
      let ddx ← letV (wAbsI (i32OfU x - i32OfU mcx))
      let ddy ← letV (wAbsI (i32OfU y - i32OfU mcy))
      ifB ((leE ddx r) .&& (leE ddy r))
        (assign (arrIdx grid (y * gwE + x)) (arrIdx params (litU pBMAT))))

def seedShader : String :=
  let grid : Expr (.arr .u32) := ⟨"grid"⟩
  let gwE : Expr .u32 := ⟨"GW"⟩
  let ghE : Expr .u32 := ⟨"GH"⟩
  buildShader
    [ { binding := 0, name := "grid", ty := .arr .u32, ro := false } ]
    []
    [.constU "GW" gw, .constU "GH" gh]
    { name := "main", wgX := 8, wgY := 8 }
    (do
      let x ← letV gidX
      let y ← letV gidY
      ifB ((x .>= gwE) .|| (y .>= ghE)) retV
      assign (arrIdx grid (y * gwE + x)) (litU EMPTY))

def renderShader : String :=
  let grid   : Expr (.arr .u32) := ⟨"grid"⟩
  let pixels : Expr (.arr .u32) := ⟨"pixels"⟩
  let imgW : Expr .u32 := ⟨"IMG_W"⟩
  let imgH : Expr .u32 := ⟨"IMG_H"⟩
  let cell : Expr .u32 := ⟨"CELL"⟩
  let gwE  : Expr .u32 := ⟨"GW"⟩
  buildShader
    [ { binding := 0, name := "grid",   ty := .arr .u32, ro := true },
      { binding := 1, name := "pixels", ty := .arr .u32, ro := false } ]
    []
    [.constU "IMG_W" imageWidth, .constU "IMG_H" imageHeight,
     .constU "CELL" cellPx, .constU "GW" gw]
    { name := "main", wgX := 8, wgY := 8 }
    (do
      let px ← letV gidX
      let py ← letV gidY
      ifB ((px .>= imgW) .|| (py .>= imgH)) retV
      let v ← letV (arrIdx grid ((py / cell) * gwE + (px / cell)))
      let isSand := v .== litU SAND
      let isWall := v .== litU WALL
      let r ← letV (wSelect (wSelect (litF "0.05") (litF "0.45") isWall) (litF "0.85") isSand)
      let g ← letV (wSelect (wSelect (litF "0.06") (litF "0.45") isWall) (litF "0.72") isSand)
      let b ← letV (wSelect (wSelect (litF "0.09") (litF "0.50") isWall) (litF "0.35") isSand)
      let ri ← letV (u32OfF (r * litF "255.0"))
      let gi ← letV (u32OfF (g * litF "255.0"))
      let bi ← letV (u32OfF (b * litF "255.0"))
      assign (arrIdx pixels (py * imgW + px))
        (((ri .| (gi .<< litU 8)) .| (bi .<< litU 16)) .| (litU 0xFF .<< litU 24)))

def blitShaderSource : String :=
  let w := toString imageWidth
  let h := toString imageHeight
  "@group(0) @binding(0) var<storage, read> pixels: array<u32>;\n\n" ++
  "struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };\n\n" ++
  "@vertex\n" ++
  "fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {\n" ++
  "  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));\n" ++
  "  var u = array<vec2<f32>, 3>(vec2<f32>(0.0, 2.0), vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0));\n" ++
  "  var o: VsOut; o.pos = vec4<f32>(p[idx], 0.0, 1.0); o.uv = u[idx]; return o;\n" ++
  "}\n\n" ++
  "@fragment\n" ++
  "fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n" ++
  "  let w: u32 = " ++ w ++ "u; let h: u32 = " ++ h ++ "u;\n" ++
  "  let px = min(u32(in.uv.x * f32(w)), w - 1u);\n" ++
  "  let py = min(u32(in.uv.y * f32(h)), h - 1u);\n" ++
  "  let packed = pixels[py * w + px];\n" ++
  "  let r = f32(packed & 0xFFu) / 255.0;\n" ++
  "  let g = f32((packed >> 8u) & 0xFFu) / 255.0;\n" ++
  "  let b = f32((packed >> 16u) & 0xFFu) / 255.0;\n" ++
  "  return vec4<f32>(r, g, b, 1.0);\n" ++
  "}\n"

structure Fields where
  reserved    : Fld (.bytes 64)
  stepSh      : Fld (.bytes 8192)
  paintSh     : Fld (.bytes 2048)
  renderSh    : Fld (.bytes 2048)
  seedSh      : Fld (.bytes 2048)
  blitSh      : Fld (.bytes 2048)
  title       : Fld (.bytes 64)
  events      : Fld (.bytes eventBytes)
  paramsMem   : Fld (.bytes 32)
  bindSeed    : Fld (.bytes 8)
  bindPaintA  : Fld (.bytes 16)
  bindPaintB  : Fld (.bytes 16)
  bindStepAB  : Fld (.bytes 24)
  bindStepBA  : Fld (.bytes 24)
  bindRenderA : Fld (.bytes 16)
  bindRenderB : Fld (.bytes 16)
  quit        : Fld .i64
  nEvents     : Fld .i64
  frame       : Fld .i64
  mouseX      : Fld .i64
  mouseY      : Fld .i64
  brushDown   : Fld .i64
  brushMat    : Fld .i64
  gridInit    : Fld (.bytes gridBytes)
  gridOut     : Fld (.bytes gridBytes)
  rowCount    : Fld .i64
  outPass     : Fld .i64
  outActual   : Fld .i64
  outExpected : Fld .i64
  pixels      : Fld (.bytes pixelBytes)

def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved    ← field (.bytes 64)
  let stepSh      ← field (.bytes 8192)
  let paintSh     ← field (.bytes 2048)
  let renderSh    ← field (.bytes 2048)
  let seedSh      ← field (.bytes 2048)
  let blitSh      ← field (.bytes 2048)
  let title       ← field (.bytes 64)
  let events      ← field (.bytes eventBytes)
  let paramsMem   ← field (.bytes 32)
  let bindSeed    ← field (.bytes 8)
  let bindPaintA  ← field (.bytes 16)
  let bindPaintB  ← field (.bytes 16)
  let bindStepAB  ← field (.bytes 24)
  let bindStepBA  ← field (.bytes 24)
  let bindRenderA ← field (.bytes 16)
  let bindRenderB ← field (.bytes 16)
  let quit        ← field .i64
  let nEvents     ← field .i64
  let frame       ← field .i64
  let mouseX      ← field .i64
  let mouseY      ← field .i64
  let brushDown   ← field .i64
  let brushMat    ← field .i64
  let gridInit    ← field (.bytes gridBytes)
  let gridOut     ← field (.bytes gridBytes)
  let rowCount    ← field .i64
  let outPass     ← field .i64
  let outActual   ← field .i64
  let outExpected ← field .i64
  let pixels      ← field (.bytes pixelBytes)
  pure { reserved, stepSh, paintSh, renderSh, seedSh, blitSh, title, events,
         paramsMem, bindSeed, bindPaintA, bindPaintB, bindStepAB, bindStepBA,
         bindRenderA, bindRenderB, quit, nEvents, frame, mouseX, mouseY,
         brushDown, brushMat, gridInit, gridOut, rowCount, outPass, outActual,
         outExpected, pixels }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

open AlgorithmLib.IR

-- Scan events: set quit on close/escape, track mouse position + brush state.
def processEvents (ptr : Val) : IRBuilder Unit := do
  let evBase ← iadd ptr (← fldOffset f.events)
  let n ← fldLoad ptr f.nEvents
  let recSz ← iconst64 32
  forLoop .i64 n (fun i => do
    let base ← iadd evBase (← imul i recSz)
    let kind ← load64 base
    let a ← load64 (← iaddImm base 8)
    let b ← load64 (← iaddImm base 16)
    let isClose ← sextend64 (← icmp .eq kind (← iconst64 evClose))
    let isDownKey ← sextend64 (← icmp .eq kind (← iconst64 evKeyDown))
    let isEsc ← sextend64 (← icmp .eq a (← iconst64 keyEscape))
    let isMove ← sextend64 (← icmp .eq kind (← iconst64 evMouseMove))
    let isMDown ← sextend64 (← icmp .eq kind (← iconst64 evMouseDown))
    let isMUp ← sextend64 (← icmp .eq kind (← iconst64 evMouseUp))
    -- quit |= close | (keydown & escape)
    let q0 ← fldLoad ptr f.quit
    fldStore ptr f.quit (← bor q0 (← bor isClose (← imul isDownKey isEsc)))
    -- mouse position (a=x, b=y on move)
    let mx ← fldLoad ptr f.mouseX
    fldStore ptr f.mouseX (← iadd mx (← imul isMove (← isub a mx)))
    let my ← fldLoad ptr f.mouseY
    fldStore ptr f.mouseY (← iadd my (← imul isMove (← isub b my)))
    -- brush material on mouse-down (a=button: 1→sand, 2→wall, else erase)
    let matSand ← sextend64 (← icmp .eq a (← iconst64 1))
    let matWall ← sextend64 (← icmp .eq a (← iconst64 2))
    let newMat ← iadd (← imul matSand (← iconst64 SAND)) (← imul matWall (← iconst64 WALL))
    let bm ← fldLoad ptr f.brushMat
    fldStore ptr f.brushMat (← iadd bm (← imul isMDown (← isub newMat bm)))
    -- brush held: set on down, clear on up
    let bd ← fldLoad ptr f.brushDown
    let bd1 ← iadd bd (← imul isMDown (← isub (← iconst64 1) bd))
    fldStore ptr f.brushDown (← isub bd1 (← imul isMUp bd1)))

def pollAndProcess (win : WindowSetup) (ptr : Val) : IRBuilder Unit := do
  let n ← windowPoll win ptr (← fldOffset f.events) (← iconst32 eventSlots)
  fldStore ptr f.nEvents (← sextend64 n)
  processEvents ptr

def putParam (ptr : Val) (idx : Nat) (v : Val) : IRBuilder Unit := do
  store (← ireduce32 v) (← absAddr ptr (f.paramsMem.offset + idx * 4))

-- Fill the params staging region for a given parity, then it's uploaded.
def writeParams (ptr parity : Val) : IRBuilder Unit := do
  putParam ptr pPARITY parity
  putParam ptr pFRAME (← fldLoad ptr f.frame)
  putParam ptr pMOUSEX (← fldLoad ptr f.mouseX)
  putParam ptr pMOUSEY (← fldLoad ptr f.mouseY)
  putParam ptr pBDOWN (← fldLoad ptr f.brushDown)
  putParam ptr pBMAT (← fldLoad ptr f.brushMat)
  putParam ptr pBR (← iconst64 brushR)

def clearGrid (ptr : Val) : IRBuilder Unit := do
  let base ← iadd ptr (← fldOffset f.gridInit)
  let z ← iconst32 0
  let four ← iconst64 4
  forLoop .i64 (← iconst64 gridCells) (fun i => do
    store z (← iadd base (← imul i four)))

def setCell (ptr : Val) (cx cy val : Nat) : IRBuilder Unit := do
  store (← iconst32 val) (← absAddr ptr (f.gridInit.offset + (cy * gw + cx) * 4))

def readOut (ptr : Val) (cx cy : Nat) : IRBuilder Val := do
  uload32_64 (← absAddr ptr (f.gridOut.offset + (cy * gw + cx) * 4))

def writeOutput (ptr : Val) (passV actualV expectedV : Val) : IRBuilder Unit := do
  fldStore ptr f.rowCount (← iconst64 1)
  fldStore ptr f.outPass passV
  fldStore ptr f.outActual actualV
  fldStore ptr f.outExpected expectedV

-- Create the 4 buffers (gridA=0, gridB=1, pixels=2, params=3).
def mkBuffers (ptr : Val) (gpu : GpuSetup) : IRBuilder (Val × Val × Val × Val) := do
  gpuInit gpu ptr
  let gridA ← gpuCreateBuffer gpu ptr (← iconst64 gridBytes)
  let gridB ← gpuCreateBuffer gpu ptr (← iconst64 gridBytes)
  let pixels ← gpuCreateBuffer gpu ptr (← iconst64 pixelBytes)
  let params ← gpuCreateBuffer gpu ptr (← iconst64 32)
  pure (gridA, gridB, pixels, params)

def mainBody : IRBuilder Unit := do
  let gpu ← declareGpuFFI
  let win ← declareWindowFFI
  let ptr ← entryBlock
  windowInit win ptr
  let (_gridA, _gridB, pixels, params) ← mkBuffers ptr gpu
  let seedP ← gpuCreatePipeline gpu ptr (← fldOffset f.seedSh) (← fldOffset f.bindSeed) (← iconst32 1)
  let paintA ← gpuCreatePipeline gpu ptr (← fldOffset f.paintSh) (← fldOffset f.bindPaintA) (← iconst32 2)
  let stepAB ← gpuCreatePipeline gpu ptr (← fldOffset f.stepSh) (← fldOffset f.bindStepAB) (← iconst32 3)
  let stepBA ← gpuCreatePipeline gpu ptr (← fldOffset f.stepSh) (← fldOffset f.bindStepBA) (← iconst32 3)
  let renderA ← gpuCreatePipeline gpu ptr (← fldOffset f.renderSh) (← fldOffset f.bindRenderA) (← iconst32 2)
  let w64 ← iconst64 imageWidth
  let h64 ← iconst64 imageHeight
  let _ ← windowOpen win ptr w64 h64 (← fldOffset f.title) (← iconst64 (titleText.length : Int))
                      (← fldOffset f.blitSh) (← iconst64 (blitShaderSource.length : Int))
  let gwg ← iconst32 gridWgX
  let ghg ← iconst32 gridWgY
  let rwx ← iconst32 renderWgX
  let rwy ← iconst32 renderWgY
  let one32 ← iconst32 1
  let paramsOff ← fldOffset f.paramsMem
  let p32 ← iconst64 32
  let _ ← gpuDispatch gpu ptr seedP gwg ghg one32
  fldStore ptr f.quit (← iconst64 0)
  fldStore ptr f.frame (← iconst64 0)
  fldStore ptr f.brushDown (← iconst64 0)
  fldStore ptr f.brushMat (← iconst64 SAND)
  let bumpFrame : IRBuilder Unit := do
    fldStore ptr f.frame (← iadd (← fldLoad ptr f.frame) (← iconst64 1))
  let subStep : Val → Val → IRBuilder Unit := fun parity pipe => do
    writeParams ptr parity
    let _ ← gpuUpload gpu ptr params paramsOff p32
    let _ ← gpuDispatch gpu ptr pipe gwg ghg one32
    bumpFrame
  let zero64 ← iconst64 0
  let one64 ← iconst64 1
  let loopHdr ← declareBlock []
  let cont ← declareBlock []
  let done ← declareBlock []
  jump loopHdr.ref []
  startBlock loopHdr
  pollAndProcess win ptr
  brif (← icmp .ne (← fldLoad ptr f.quit) zero64) done.ref [] cont.ref []
  startBlock cont
  -- stamp the brush into A once, then run 8 Margolus sub-steps (4 ping-pong
  -- pairs, parity alternating) so sand advances fast; render A and present once.
  writeParams ptr zero64
  let _ ← gpuUpload gpu ptr params paramsOff p32
  let _ ← gpuDispatch gpu ptr paintA gwg ghg one32
  for _ in List.range 4 do
    subStep zero64 stepAB
    subStep one64 stepBA
  let _ ← gpuDispatch gpu ptr renderA rwx rwy one32
  let _ ← windowPresentGpuBuffer win ptr pixels
  jump loopHdr.ref []
  startBlock done
  windowCleanup win ptr
  gpuCleanup gpu ptr
  ret

-- Shared test setup: buffers + stepAB pipeline, params zeroed (parity 0, brush off).
def testSetup (ptr : Val) (gpu : GpuSetup) : IRBuilder (Val × Val × Val) := do
  let (gridA, gridB, _pixels, params) ← mkBuffers ptr gpu
  let stepAB ← gpuCreatePipeline gpu ptr (← fldOffset f.stepSh) (← fldOffset f.bindStepAB) (← iconst32 3)
  fldStore ptr f.frame (← iconst64 0)
  fldStore ptr f.mouseX (← iconst64 0)
  fldStore ptr f.mouseY (← iconst64 0)
  fldStore ptr f.brushDown (← iconst64 0)
  fldStore ptr f.brushMat (← iconst64 0)
  writeParams ptr (← iconst64 0)
  let _ ← gpuUpload gpu ptr params (← fldOffset f.paramsMem) (← iconst64 32)
  pure (gridA, gridB, stepAB)

-- A lone grain with empty below drops to the next row (straight or scattered).
def testGrainFalls : IRBuilder Unit := do
  let gpu ← declareGpuFFI
  let ptr ← entryBlock
  let (gridA, gridB, stepAB) ← testSetup ptr gpu
  clearGrid ptr
  setCell ptr 10 10 SAND
  let _ ← gpuUpload gpu ptr gridA (← fldOffset f.gridInit) (← iconst64 gridBytes)
  let _ ← gpuDispatch gpu ptr stepAB (← iconst32 gridWgX) (← iconst32 gridWgY) (← iconst32 1)
  let _ ← gpuDownload gpu ptr gridB (← fldOffset f.gridOut) (← iconst64 gridBytes)
  gpuCleanup gpu ptr
  let bl ← readOut ptr 10 11
  let br ← readOut ptr 11 11
  let orig ← readOut ptr 10 10
  let landed ← bor (← sextend64 (← icmp .eq bl (← iconst64 SAND)))
                   (← sextend64 (← icmp .eq br (← iconst64 SAND)))
  let vacated ← sextend64 (← icmp .eq orig (← iconst64 EMPTY))
  writeOutput ptr (← band landed vacated) (← iadd bl br) (← iconst64 SAND)
  ret

-- Sand is conserved: a 4×4 blob keeps its 16 grains after one step.
def testConservation : IRBuilder Unit := do
  let gpu ← declareGpuFFI
  let ptr ← entryBlock
  let (gridA, gridB, stepAB) ← testSetup ptr gpu
  clearGrid ptr
  for cy in [20, 21, 22, 23] do
    for cx in [40, 41, 42, 43] do
      setCell ptr cx cy SAND
  let _ ← gpuUpload gpu ptr gridA (← fldOffset f.gridInit) (← iconst64 gridBytes)
  let _ ← gpuDispatch gpu ptr stepAB (← iconst32 gridWgX) (← iconst32 gridWgY) (← iconst32 1)
  let _ ← gpuDownload gpu ptr gridB (← fldOffset f.gridOut) (← iconst64 gridBytes)
  gpuCleanup gpu ptr
  let gridOutBase ← iadd ptr (← fldOffset f.gridOut)
  let four ← iconst64 4
  let sandC ← iconst64 SAND
  let count ← forLoopAcc .i64 .i64 (← iconst64 gridCells) (← iconst64 0) (fun i acc => do
    let cell ← uload32_64 (← iadd gridOutBase (← imul i four))
    iadd acc (← sextend64 (← icmp .eq cell sandC)))
  let expected ← iconst64 16
  writeOutput ptr (← sextend64 (← icmp .eq count expected)) count expected
  ret

def clifIrSource : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 mainBody ++ "\n" ++
  buildFunction 2 testGrainFalls ++ "\n" ++
  buildFunction 3 testConservation

def bindBytes (pairs : List (Nat × Nat)) : List UInt8 :=
  pairs.foldl (fun acc (b, ro) => acc ++ uint32ToBytes (UInt32.ofNat b) ++ uint32ToBytes (UInt32.ofNat ro)) []

def payloads : List UInt8 :=
  mkPayload layoutMeta.totalSize [
    f.stepSh.init (stringToBytes stepShader),
    f.paintSh.init (stringToBytes paintShader),
    f.renderSh.init (stringToBytes renderShader),
    f.seedSh.init (stringToBytes seedShader),
    f.blitSh.init (stringToBytes blitShaderSource),
    f.title.init (stringToBytes titleText),
    f.bindSeed.init    (bindBytes [(0, 0)]),
    f.bindPaintA.init  (bindBytes [(0, 0), (3, 1)]),
    f.bindPaintB.init  (bindBytes [(1, 0), (3, 1)]),
    f.bindStepAB.init  (bindBytes [(0, 1), (1, 0), (3, 1)]),
    f.bindStepBA.init  (bindBytes [(1, 1), (0, 0), (3, 1)]),
    f.bindRenderA.init (bindBytes [(0, 1), (2, 0)]),
    f.bindRenderB.init (bindBytes [(1, 1), (2, 0)])
  ]

def gameSetup : Setup := {
  cranelift_ir := clifIrSource,
  memory_size := layoutMeta.totalSize,
  initial_memory := payloads
}

def testSchema : List Json :=
  [Output.schema
    [ Output.column "pass" .i64 f.outPass.offset,
      Output.column "actual" .i64 f.outActual.offset,
      Output.column "expected" .i64 f.outExpected.offset ]
    f.rowCount.offset]

def mainAlgorithm   : Algorithm := { fn_idx := IR.mainFnIdx }
def grainFallsAlg   : Algorithm := { fn_idx := u32 2, output := testSchema }
def conservationAlg : Algorithm := { fn_idx := u32 3, output := testSchema }

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← AlgorithmLib.requireOutputDir args
  AlgorithmLib.emitArtifacts outDir #[
    AlgorithmLib.toJsonArtifact "falling_sand" Algorithm.gameSetup Algorithm.mainAlgorithm [
      ("test_grain_falls",  Algorithm.grainFallsAlg),
      ("test_conservation", Algorithm.conservationAlg)
    ]]
