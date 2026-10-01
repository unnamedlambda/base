module
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Vocab.WGSL
meta import AlgorithmLib.Vocab.WGSL
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

set_option maxRecDepth 8192

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout
open AlgorithmLib.WGSL

namespace FallingSand

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

inductive Cell | empty | sand | wall
  deriving DecidableEq

def Cell.code : Cell → Nat
  | .empty => EMPTY | .sand => SAND | .wall => WALL

def Cell.isSand : Cell → Nat
  | .sand => 1 | _ => 0

inductive CName | tl | tr | bl | br

structure Block where
  tl : Cell
  tr : Cell
  bl : Cell
  br : Cell

def Block.sandCount (b : Block) : Nat :=
  b.tl.isSand + b.tr.isSand + b.bl.isSand + b.br.isSand

def Block.get : Block → CName → Cell
  | b, .tl => b.tl | b, .tr => b.tr | b, .bl => b.bl | b, .br => b.br

def Block.set : Block → CName → Cell → Block
  | b, .tl, v => { b with tl := v } | b, .tr, v => { b with tr := v }
  | b, .bl, v => { b with bl := v } | b, .br, v => { b with br := v }

inductive GAtom | is (c : CName) (v : Cell) | isnt (c : CName) (v : Cell) | rnd (v : Bool)
abbrev Guard := List GAtom

structure Move where
  guard : Guard
  src : CName
  dst : CName

def GAtom.eval : GAtom → Block → Bool → Bool
  | .is c v, b, _   => b.get c == v
  | .isnt c v, b, _ => !(b.get c == v)
  | .rnd v, _, r    => r == v

def Guard.eval (g : Guard) (b : Block) (r : Bool) : Bool := g.all (fun a => a.eval b r)

/-- Semantics of one move: if its guard holds, move a grain src→dst. -/
def Move.apply (m : Move) (r : Bool) (b : Block) : Block :=
  cond (m.guard.eval b r) ((b.set m.src .empty).set m.dst .sand) b

abbrev Program := List Move

def Program.eval (ms : Program) (r : Bool) (b : Block) : Block :=
  ms.foldl (fun acc m => m.apply r acc) b

/-- The Margolus sand rule as data: straight fall, RNG scatter, then diagonal. -/
def sandProgram : Program :=
  [ { guard := [.is .tl .sand, .is .bl .empty, .rnd true, .is .br .empty], src := .tl, dst := .br },
    { guard := [.is .tl .sand, .is .bl .empty],                             src := .tl, dst := .bl },
    { guard := [.is .tr .sand, .is .br .empty, .rnd false, .is .bl .empty], src := .tr, dst := .bl },
    { guard := [.is .tr .sand, .is .br .empty],                             src := .tr, dst := .br },
    { guard := [.is .tl .sand, .isnt .bl .empty, .is .br .empty],           src := .tl, dst := .br },
    { guard := [.is .tr .sand, .isnt .br .empty, .is .bl .empty],           src := .tr, dst := .bl } ]

theorem sandProgram_conserves (b : Block) (r : Bool) :
    (sandProgram.eval r b).sandCount = b.sandCount := by
  obtain ⟨tl, tr, bl, br⟩ := b
  cases tl <;> cases tr <;> cases bl <;> cases br <;> cases r <;> rfl

-- The WGSL renderer for the same `sandProgram` (used by stepShader). `cv` maps
-- each cell name to its shader `var` handle; `rndE` is the RNG bit expression.
def renderAtom (cv : CName → Expr .u32) (rndE : Expr .u32) : GAtom → Expr .bool
  | .is c v   => (cv c) .== litU v.code
  | .isnt c v => neE (cv c) (litU v.code)
  | .rnd v    => rndE .== litU (if v then 1 else 0)

def renderGuard (cv : CName → Expr .u32) (rndE : Expr .u32) (g : Guard) : Expr .bool :=
  g.foldl (fun acc a => acc .&& renderAtom cv rndE a) litTrue

def renderProgram (cv : CName → Expr .u32) (rndE : Expr .u32) (ms : Program) : WB Unit :=
  ms.forM fun m =>
    ifB (renderGuard cv rndE m.guard)
      (do assign (cv m.src) (litU EMPTY); assign (cv m.dst) (litU SAND))

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
          renderProgram (fun c => match c with
              | .tl => tl | .tr => tr | .bl => bl | .br => br) rnd sandProgram
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
  let pixels      ← field (.bytes pixelBytes)
  pure { reserved, stepSh, paintSh, renderSh, seedSh, blitSh, title, events,
         paramsMem, bindSeed, bindPaintA, bindPaintB, bindStepAB, bindStepBA,
         bindRenderA, bindRenderB, quit, nEvents, frame, mouseX, mouseY,
         brushDown, brushMat, gridInit, gridOut, pixels }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

open AlgorithmLib.IR
open AlgorithmLib.HProg

-- Scan events: set quit on close/escape, track mouse position + brush state.
open AlgorithmLib.Prog


def processEvents (ptr : V .i64) : Prog V L Unit := do
  let evBase ← iadd ptr (← fldOffset f.events)
  let n ← fldLoad ptr f.nEvents
  let recSz ← iconst64 32
  -- a count past the slots, as a failed poll's -1 is, reads no events
  when .ule n (← iconst64 eventSlots) do
    forLoop n (fun i => do
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

def pollAndProcess (ptr : V .i64) : Prog V L Unit := do
  let n ← windowPoll ptr (← fldOffset f.events) (← iconst32 eventSlots)
  fldStore ptr f.nEvents (← sextend64 n)
  processEvents ptr

def putParam (ptr : V .i64) (idx : Nat) (v : V .i64) : Prog V L Unit := do
  storeUnaligned (← ireduce32 v) (← absAddr ptr (f.paramsMem.offset + idx * 4))

-- Fill the params staging region for a given parity, then it's uploaded.
def writeParams (ptr parity : V .i64) : Prog V L Unit := do
  putParam ptr pPARITY parity
  putParam ptr pFRAME (← fldLoad ptr f.frame)
  putParam ptr pMOUSEX (← fldLoad ptr f.mouseX)
  putParam ptr pMOUSEY (← fldLoad ptr f.mouseY)
  putParam ptr pBDOWN (← fldLoad ptr f.brushDown)
  putParam ptr pBMAT (← fldLoad ptr f.brushMat)
  putParam ptr pBR (← iconst64 brushR)

def clearGrid (ptr : V .i64) : Prog V L Unit := do
  let base ← iadd ptr (← fldOffset f.gridInit)
  let z ← iconst32 0
  let four ← iconst64 4
  forLoop (← iconst64 gridCells) (fun i => do
    store z (← iadd base (← imul i four)))

def setCell (ptr : V .i64) (cx cy val : Nat) : Prog V L Unit := do
  storeUnaligned (← iconst32 val) (← absAddr ptr (f.gridInit.offset + (cy * gw + cx) * 4))

def readOut (ptr : V .i64) (cx cy : Nat) : Prog V L (V .i64) := do
  uload32_64 (← absAddr ptr (f.gridOut.offset + (cy * gw + cx) * 4))

/-- Answer one result row in the caller's out buffer: pass (1/0), actual and
    expected, eight bytes each. A buffer too short for the row is left alone,
    so a host that passes none learns nothing rather than having memory past
    its buffer written. -/
def writeOutput (passV actualV expectedV : V .i64) : Prog V L Unit := do
  let out ← outPtr
  when .uge (← outLen) (← iconst64 24) do
    storeAt out 0 passV
    storeAt out 8 actualV
    storeAt out 16 expectedV

-- Create the 4 buffers (gridA=0, gridB=1, pixels=2, params=3).
def mkBuffers (ptr : V .i64) : Prog V L (V .i32 × V .i32 × V .i32 × V .i32) := do
  gpuInit ptr
  let gridA ← gpuCreateBuffer ptr (← iconst64 gridBytes)
  let gridB ← gpuCreateBuffer ptr (← iconst64 gridBytes)
  let pixels ← gpuCreateBuffer ptr (← iconst64 pixelBytes)
  let params ← gpuCreateBuffer ptr (← iconst64 32)
  pure (gridA, gridB, pixels, params)

def mainBody : Prog V L Unit := do
  let ptr ← basePtr
  windowInit ptr
  let (_gridA, _gridB, pixels, params) ← mkBuffers ptr
  let seedP ← gpuCreatePipeline ptr (← fldOffset f.seedSh) (← fldOffset f.bindSeed) (← iconst32 1)
  let paintA ← gpuCreatePipeline ptr (← fldOffset f.paintSh) (← fldOffset f.bindPaintA) (← iconst32 2)
  let stepAB ← gpuCreatePipeline ptr (← fldOffset f.stepSh) (← fldOffset f.bindStepAB) (← iconst32 3)
  let stepBA ← gpuCreatePipeline ptr (← fldOffset f.stepSh) (← fldOffset f.bindStepBA) (← iconst32 3)
  let renderA ← gpuCreatePipeline ptr (← fldOffset f.renderSh) (← fldOffset f.bindRenderA) (← iconst32 2)
  let w64 ← iconst64 imageWidth
  let h64 ← iconst64 imageHeight
  let _ ← windowOpen ptr w64 h64 (← fldOffset f.title) (← iconst64 (titleText.utf8ByteSize : Int))
                      (← fldOffset f.blitSh) (← iconst64 (blitShaderSource.utf8ByteSize : Int))
  let gwg ← iconst32 gridWgX
  let ghg ← iconst32 gridWgY
  let rwx ← iconst32 renderWgX
  let rwy ← iconst32 renderWgY
  let one32 ← iconst32 1
  let paramsOff ← fldOffset f.paramsMem
  let p32 ← iconst64 32
  let _ ← gpuDispatch ptr seedP gwg ghg one32
  fldStore ptr f.quit (← iconst64 0)
  fldStore ptr f.frame (← iconst64 0)
  fldStore ptr f.brushDown (← iconst64 0)
  fldStore ptr f.brushMat (← iconst64 SAND)
  let bumpFrame : Prog V L Unit := do
    fldStore ptr f.frame (← iadd (← fldLoad ptr f.frame) (← iconst64 1))
  let subStep : V .i64 → V .i32 → Prog V L Unit := fun parity pipe => do
    writeParams ptr parity
    let _ ← gpuUpload ptr params paramsOff p32
    let _ ← gpuDispatch ptr pipe gwg ghg one32
    bumpFrame
  let zero64 ← iconst64 0
  let one64 ← iconst64 1
  let _ ← wloop %[] (head := fun _ => do
      pollAndProcess ptr
      let q ← fldLoad ptr f.quit
      return (exitIf .ne q zero64, %[], ()))
    (body := fun _ _ => do
      -- stamp the brush into A once, then run 8 Margolus sub-steps (4 ping-pong
      -- pairs, parity alternating) so sand advances fast; render A and present once.
      writeParams ptr zero64
      let _ ← gpuUpload ptr params paramsOff p32
      let _ ← gpuDispatch ptr paintA gwg ghg one32
      for _ in List.range 4 do
        subStep zero64 stepAB
        subStep one64 stepBA
      let _ ← gpuDispatch ptr renderA rwx rwy one32
      let _ ← windowPresentGpuBuffer ptr pixels
      return %[])
  windowCleanup ptr
  gpuCleanup ptr

-- Shared test setup: buffers + stepAB pipeline, params zeroed (parity 0, brush off).
def testSetup (ptr : V .i64) : Prog V L (V .i32 × V .i32 × V .i32) := do
  let (gridA, gridB, _pixels, params) ← mkBuffers ptr
  let stepAB ← gpuCreatePipeline ptr (← fldOffset f.stepSh) (← fldOffset f.bindStepAB) (← iconst32 3)
  fldStore ptr f.frame (← iconst64 0)
  fldStore ptr f.mouseX (← iconst64 0)
  fldStore ptr f.mouseY (← iconst64 0)
  fldStore ptr f.brushDown (← iconst64 0)
  fldStore ptr f.brushMat (← iconst64 0)
  writeParams ptr (← iconst64 0)
  let _ ← gpuUpload ptr params (← fldOffset f.paramsMem) (← iconst64 32)
  pure (gridA, gridB, stepAB)

-- A lone grain with empty below drops to the next row (straight or scattered).
def testGrainFalls : Prog V L Unit := do
  let ptr ← basePtr
  let (gridA, gridB, stepAB) ← testSetup ptr
  clearGrid ptr
  setCell ptr 10 10 SAND
  let _ ← gpuUpload ptr gridA (← fldOffset f.gridInit) (← iconst64 gridBytes)
  let _ ← gpuDispatch ptr stepAB (← iconst32 gridWgX) (← iconst32 gridWgY) (← iconst32 1)
  let _ ← gpuDownload ptr gridB (← fldOffset f.gridOut) (← iconst64 gridBytes)
  gpuCleanup ptr
  let bl ← readOut ptr 10 11
  let br ← readOut ptr 11 11
  let orig ← readOut ptr 10 10
  let landed ← bor (← sextend64 (← icmp .eq bl (← iconst64 SAND)))
                   (← sextend64 (← icmp .eq br (← iconst64 SAND)))
  let vacated ← sextend64 (← icmp .eq orig (← iconst64 EMPTY))
  writeOutput (← band landed vacated) (← iadd bl br) (← iconst64 SAND)

-- Sand is conserved: a 4×4 blob keeps its 16 grains after one step.
def testConservation : Prog V L Unit := do
  let ptr ← basePtr
  let (gridA, gridB, stepAB) ← testSetup ptr
  clearGrid ptr
  for cy in [20, 21, 22, 23] do
    for cx in [40, 41, 42, 43] do
      setCell ptr cx cy SAND
  let _ ← gpuUpload ptr gridA (← fldOffset f.gridInit) (← iconst64 gridBytes)
  let _ ← gpuDispatch ptr stepAB (← iconst32 gridWgX) (← iconst32 gridWgY) (← iconst32 1)
  let _ ← gpuDownload ptr gridB (← fldOffset f.gridOut) (← iconst64 gridBytes)
  gpuCleanup ptr
  let gridOutBase ← iadd ptr (← fldOffset f.gridOut)
  let four ← iconst64 4
  let sandC ← iconst64 SAND
  let count ← forLoopAcc (← iconst64 gridCells) (← iconst64 0) (fun i acc => do
    let cell ← uload32_64 (← iadd gridOutBase (← imul i four))
    iadd acc (← sextend64 (← icmp .eq cell sandC)))
  let expected ← iconst64 16
  writeOutput (← sextend64 (← icmp .eq count expected)) count expected


def clifIrSource : Except String (List IR.FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 mainBody),
     Prog.entry "test_grain_falls" (Prog.compileProg 2 testGrainFalls),
     Prog.entry "test_conservation" (Prog.compileProg 3 testConservation)]

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

def gameSetup (clif : List IR.FuncData) : Artifact := {
  functions := clif,
  required_memory := layoutMeta.totalSize,
  initial_memory := payloads
}

end FallingSand

def Demo.FallingSand.main (args : List String) : IO Unit := do
  let outDir ← AlgorithmLib.requireOutputDir args
  let clif ← AlgorithmLib.Prog.orDie FallingSand.clifIrSource
  AlgorithmLib.emitArtifacts outDir #[
    AlgorithmLib.artifactEntry "falling_sand" (FallingSand.gameSetup clif)]

#eval ShipScan.check "Demo.FallingSand" `Demo.FallingSand.main
