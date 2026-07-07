import AlgorithmLib

set_option maxRecDepth 8192

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout

namespace Algorithm

def imageWidth  : Nat := 640
def imageHeight : Nat := 360
def pixelBytes  : Nat := imageWidth * imageHeight * 4
def paramsBytes : Nat := 32
def eventSlots  : Nat := 64
def eventBytes  : Nat := eventSlots * 32
def wgX : Nat := (imageWidth + 7) / 8
def wgY : Nat := (imageHeight + 7) / 8

def moveSpeed : Int := 8
def camStartX : Int := 0
def camStartY : Int := 200
def camStartZ : Int := 0
def minXZ : Int := -4000
def maxXZ : Int := 4000
def minY : Int := 50
def maxY : Int := 800

def evClose   : Int := 1
def evKeyDown : Int := 3
def evKeyUp   : Int := 4
def keyEscape : Int := 1
def keyLeft   : Int := 3
def keyRight  : Int := 4
def keyFwd    : Int := 5
def keyBack   : Int := 6
def keyUpK    : Int := 10
def keyDownK  : Int := 12
def bitLeft  : Int := 1
def bitRight : Int := 2
def bitFwd   : Int := 4
def bitBack  : Int := 8
def bitUp    : Int := 16
def bitDown  : Int := 32

def titleText : String := "Base Raymarch"

def centerX : Nat := imageWidth / 2
def skyByteOff : Nat := (10 * imageWidth + centerX) * 4 + 2
def groundByteOff : Nat := ((imageHeight - 10) * imageWidth + centerX) * 4 + 2

def shaderSource : String :=
  "@group(0) @binding(0)\n" ++
  "var<storage, read_write> pixels: array<u32>;\n" ++
  "@group(0) @binding(1)\n" ++
  "var<storage, read> params: array<u32>;\n\n" ++
  "fn sdSphere(p: vec3<f32>, c: vec3<f32>, r: f32) -> f32 { return length(p - c) - r; }\n" ++
  "fn sdBox(p: vec3<f32>, c: vec3<f32>, b: vec3<f32>) -> f32 {\n" ++
  "  let q = abs(p - c) - b;\n" ++
  "  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);\n" ++
  "}\n" ++
  "fn scene(p: vec3<f32>) -> f32 {\n" ++
  "  let ground = p.y;\n" ++
  "  let s1 = sdSphere(p, vec3<f32>(0.0, 1.0, -6.0), 1.0);\n" ++
  "  let s2 = sdSphere(p, vec3<f32>(-2.5, 0.7, -8.0), 0.7);\n" ++
  "  let b1 = sdBox(p, vec3<f32>(2.5, 1.0, -9.0), vec3<f32>(1.0, 1.0, 1.0));\n" ++
  "  return min(ground, min(s1, min(s2, b1)));\n" ++
  "}\n" ++
  "fn calcNormal(p: vec3<f32>) -> vec3<f32> {\n" ++
  "  let e = vec2<f32>(0.001, 0.0);\n" ++
  "  return normalize(vec3<f32>(\n" ++
  "    scene(p + e.xyy) - scene(p - e.xyy),\n" ++
  "    scene(p + e.yxy) - scene(p - e.yxy),\n" ++
  "    scene(p + e.yyx) - scene(p - e.yyx)));\n" ++
  "}\n\n" ++
  "@compute @workgroup_size(8, 8)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  "  let width = params[1];\n" ++
  "  let height = params[2];\n" ++
  "  let x = gid.x;\n" ++
  "  let y = gid.y;\n" ++
  "  if (x >= width || y >= height) { return; }\n" ++
  "  let camX = f32(bitcast<i32>(params[3])) / 100.0;\n" ++
  "  let camY = f32(bitcast<i32>(params[4])) / 100.0;\n" ++
  "  let camZ = f32(bitcast<i32>(params[5])) / 100.0;\n" ++
  "  let res = vec2<f32>(f32(width), f32(height));\n" ++
  "  let frag = vec2<f32>(f32(x), f32(height - 1u - y));\n" ++
  "  let uv = (frag - 0.5 * res) / f32(height);\n" ++
  "  let ro = vec3<f32>(camX, camY, camZ);\n" ++
  "  let rd = normalize(vec3<f32>(uv.x, uv.y, -1.0));\n" ++
  "  var t = 0.0;\n" ++
  "  var hit = false;\n" ++
  "  var p = ro;\n" ++
  "  for (var i = 0; i < 96; i = i + 1) {\n" ++
  "    p = ro + rd * t;\n" ++
  "    let d = scene(p);\n" ++
  "    if (d < 0.001) { hit = true; break; }\n" ++
  "    t = t + d;\n" ++
  "    if (t > 60.0) { break; }\n" ++
  "  }\n" ++
  "  var col: vec3<f32>;\n" ++
  "  if (hit) {\n" ++
  "    let n = calcNormal(p);\n" ++
  "    let ld = normalize(vec3<f32>(0.6, 0.8, 0.3));\n" ++
  "    let diff = max(dot(n, ld), 0.0);\n" ++
  "    var base: vec3<f32>;\n" ++
  "    if (p.y < 0.01) {\n" ++
  "      let checker = f32((i32(floor(p.x)) + i32(floor(p.z))) & 1);\n" ++
  "      let c = 0.25 + 0.15 * checker;\n" ++
  "      base = vec3<f32>(c, c * 1.1, c * 0.7);\n" ++
  "    } else {\n" ++
  "      base = 0.5 + 0.5 * n;\n" ++
  "    }\n" ++
  "    col = base * (0.2 + 0.8 * diff);\n" ++
  "  } else {\n" ++
  "    col = mix(vec3<f32>(0.5, 0.7, 1.0), vec3<f32>(0.1, 0.2, 0.5), clamp(rd.y * 0.5 + 0.5, 0.0, 1.0));\n" ++
  "  }\n" ++
  "  col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));\n" ++
  "  let ri = u32(col.x * 255.0);\n" ++
  "  let gi = u32(col.y * 255.0);\n" ++
  "  let bi = u32(col.z * 255.0);\n" ++
  "  pixels[y * width + x] = ri | (gi << 8u) | (bi << 16u) | (0xFFu << 24u);\n" ++
  "}\n"

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

structure Fields where
  reserved    : Fld (.bytes 64)
  bindDesc    : Fld (.bytes 16)
  shader      : Fld (.bytes 8192)
  blitShader  : Fld (.bytes 2048)
  title       : Fld (.bytes 64)
  events      : Fld (.bytes eventBytes)
  params      : Fld (.bytes paramsBytes)
  keyMask     : Fld .i64
  quit        : Fld .i64
  camX        : Fld .i64
  camY        : Fld .i64
  camZ        : Fld .i64
  nEvents     : Fld .i64
  rowCount    : Fld .i64
  outPass     : Fld .i64
  outActual   : Fld .i64
  outExpected : Fld .i64
  pixels      : Fld (.bytes pixelBytes)

def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved    ← field (.bytes 64)
  let bindDesc    ← field (.bytes 16)
  let shader      ← field (.bytes 8192)
  let blitShader  ← field (.bytes 2048)
  let title       ← field (.bytes 64)
  let events      ← field (.bytes eventBytes)
  let params      ← field (.bytes paramsBytes)
  let keyMask     ← field .i64
  let quit        ← field .i64
  let camX        ← field .i64
  let camY        ← field .i64
  let camZ        ← field .i64
  let nEvents     ← field .i64
  let rowCount    ← field .i64
  let outPass     ← field .i64
  let outActual   ← field .i64
  let outExpected ← field .i64
  let pixels      ← field (.bytes pixelBytes)
  pure { reserved, bindDesc, shader, blitShader, title, events, params, keyMask,
         quit, camX, camY, camZ, nEvents, rowCount, outPass, outActual,
         outExpected, pixels }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

open AlgorithmLib.IR

def clearState (ptr : Val) : IRBuilder Unit := do
  let z ← iconst64 0
  fldStore ptr f.keyMask z
  fldStore ptr f.quit z
  fldStore ptr f.camX (← iconst64 camStartX)
  fldStore ptr f.camY (← iconst64 camStartY)
  fldStore ptr f.camZ (← iconst64 camStartZ)

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
    let isF ← sextend64 (← icmp .eq keyCode (← iconst64 keyFwd))
    let isB ← sextend64 (← icmp .eq keyCode (← iconst64 keyBack))
    let isU ← sextend64 (← icmp .eq keyCode (← iconst64 keyUpK))
    let isD ← sextend64 (← icmp .eq keyCode (← iconst64 keyDownK))
    let isEsc ← sextend64 (← icmp .eq keyCode (← iconst64 keyEscape))
    let bit01 ← iadd (← imul isL (← iconst64 bitLeft)) (← imul isR (← iconst64 bitRight))
    let bit23 ← iadd (← imul isF (← iconst64 bitFwd)) (← imul isB (← iconst64 bitBack))
    let bit45 ← iadd (← imul isU (← iconst64 bitUp)) (← imul isD (← iconst64 bitDown))
    let bit ← iadd (← iadd bit01 bit23) bit45
    let mask0 ← fldLoad ptr f.keyMask
    let mask1 ← bor mask0 (← imul bit isDown)
    let mask2 ← bandNot mask1 (← imul bit isUp)
    fldStore ptr f.keyMask mask2
    let q0 ← fldLoad ptr f.quit
    let q1 ← bor q0 (← bor isClose (← imul isDown isEsc))
    fldStore ptr f.quit q1)

def clampField (ptr : Val) (fld : Fld .i64) (lo hi : Int) : IRBuilder Unit := do
  let x ← fldLoad ptr fld
  let over ← sextend64 (← icmp .sgt x (← iconst64 hi))
  let x1 ← iadd x (← imul over (← isub (← iconst64 hi) x))
  let under ← sextend64 (← icmp .slt x1 (← iconst64 lo))
  let x2 ← iadd x1 (← imul under (← isub (← iconst64 lo) x1))
  fldStore ptr fld x2

def heldAxis (mask : Val) (posBit negBit : Int) : IRBuilder Val := do
  let pos ← sextend64 (← icmp .ne (← band mask (← iconst64 posBit)) (← iconst64 0))
  let neg ← sextend64 (← icmp .ne (← band mask (← iconst64 negBit)) (← iconst64 0))
  isub pos neg

def applyMovement (ptr : Val) : IRBuilder Unit := do
  let mask ← fldLoad ptr f.keyMask
  let speed ← iconst64 moveSpeed
  let dx ← imul (← heldAxis mask bitRight bitLeft) speed
  let dz ← imul (← heldAxis mask bitBack bitFwd) speed
  let dy ← imul (← heldAxis mask bitUp bitDown) speed
  fldStore ptr f.camX (← iadd (← fldLoad ptr f.camX) dx)
  fldStore ptr f.camZ (← iadd (← fldLoad ptr f.camZ) dz)
  fldStore ptr f.camY (← iadd (← fldLoad ptr f.camY) dy)
  clampField ptr f.camX minXZ maxXZ
  clampField ptr f.camZ minXZ maxXZ
  clampField ptr f.camY minY maxY

def writeParams (ptr : Val) (frame : Val) : IRBuilder Unit := do
  fldStore32At ptr f.params 0 (← ireduce32 frame) (by decide)
  fldStore32At ptr f.params 12 (← ireduce32 (← fldLoad ptr f.camX)) (by decide)
  fldStore32At ptr f.params 16 (← ireduce32 (← fldLoad ptr f.camY)) (by decide)
  fldStore32At ptr f.params 20 (← ireduce32 (← fldLoad ptr f.camZ)) (by decide)

def writeEvent (ptr : Val) (slot : Nat) (kind keyCode : Int) : IRBuilder Unit := do
  let base ← absAddr ptr (f.events.offset + slot * 32)
  store (← iconst64 kind) base
  store (← iconst64 keyCode) (← iaddImm base 8)

def writeOutput (ptr : Val) (passV actualV expectedV : Val) : IRBuilder Unit := do
  fldStore ptr f.rowCount (← iconst64 1)
  fldStore ptr f.outPass passV
  fldStore ptr f.outActual actualV
  fldStore ptr f.outExpected expectedV

def stepN (ptr : Val) (steps : Int) : IRBuilder Unit := do
  let limit ← iconst64 steps
  forLoop .i64 limit (fun _ => do
    processEvents ptr
    applyMovement ptr)

def assertEq (ptr actual : Val) (expected : Int) : IRBuilder Unit := do
  let exp ← iconst64 expected
  writeOutput ptr (← sextend64 (← icmp .eq actual exp)) actual exp

def dispatchScene (gpu : GpuSetup) (ptr paramBuf pipeId : Val) : IRBuilder Unit := do
  let _ ← gpuUpload gpu ptr paramBuf (← fldOffset f.params) (← iconst64 paramsBytes)
  let _ ← gpuDispatch gpu ptr pipeId (← iconst32 wgX) (← iconst32 wgY) (← iconst32 1)
  pure ()

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
  clearState ptr
  let _ ← whileLoop1 .i64 (← iconst64 0)
    (fun _ => do icmp .eq (← fldLoad ptr f.quit) (← iconst64 0))
    (fun frame => do
      let n ← windowPoll win ptr (← fldOffset f.events) (← iconst32 eventSlots)
      fldStore ptr f.nEvents (← sextend64 n)
      processEvents ptr
      applyMovement ptr
      writeParams ptr frame
      dispatchScene gpu ptr paramBuf pipeId
      let _ ← windowPresentGpuBuffer win ptr pixelBuf
      iaddImm frame 1)
  windowCleanup win ptr
  gpuCleanup gpu ptr
  ret

def testMoveForward : IRBuilder Unit := do
  let ptr ← entryBlock
  clearState ptr
  writeEvent ptr 0 evKeyDown keyFwd
  fldStore ptr f.nEvents (← iconst64 1)
  stepN ptr 10
  assertEq ptr (← fldLoad ptr f.camZ) (camStartZ - 10 * moveSpeed)
  ret

def testStrafeRight : IRBuilder Unit := do
  let ptr ← entryBlock
  clearState ptr
  writeEvent ptr 0 evKeyDown keyRight
  fldStore ptr f.nEvents (← iconst64 1)
  stepN ptr 10
  assertEq ptr (← fldLoad ptr f.camX) (camStartX + 10 * moveSpeed)
  ret

def testRiseClamp : IRBuilder Unit := do
  let ptr ← entryBlock
  clearState ptr
  writeEvent ptr 0 evKeyDown keyUpK
  fldStore ptr f.nEvents (← iconst64 1)
  stepN ptr 200
  assertEq ptr (← fldLoad ptr f.camY) maxY
  ret

def testQuitOnClose : IRBuilder Unit := do
  let ptr ← entryBlock
  clearState ptr
  writeEvent ptr 0 evClose 0
  fldStore ptr f.nEvents (← iconst64 1)
  processEvents ptr
  assertEq ptr (← fldLoad ptr f.quit) 1
  ret

def testRenderScene : IRBuilder Unit := do
  let gpu ← declareGpuFFI
  let ptr ← entryBlock
  gpuInit gpu ptr
  let pixelBuf ← gpuCreateBuffer gpu ptr (← iconst64 pixelBytes)
  let paramBuf ← gpuCreateBuffer gpu ptr (← iconst64 paramsBytes)
  let pipeId ← gpuCreatePipeline gpu ptr (← fldOffset f.shader) (← fldOffset f.bindDesc) (← iconst32 2)
  clearState ptr
  writeParams ptr (← iconst64 0)
  dispatchScene gpu ptr paramBuf pipeId
  let _ ← gpuDownload gpu ptr pixelBuf (← fldOffset f.pixels) (← iconst64 pixelBytes)
  gpuCleanup gpu ptr
  let skyB ← uload8_64 (← absAddr ptr (f.pixels.offset + skyByteOff))
  let groundB ← uload8_64 (← absAddr ptr (f.pixels.offset + groundByteOff))
  let thresh ← iadd groundB (← iconst64 20)
  writeOutput ptr (← sextend64 (← icmp .ugt skyB thresh)) groundB skyB
  ret

def clifIrSource : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 mainBody ++ "\n" ++
  buildFunction 2 testMoveForward ++ "\n" ++
  buildFunction 3 testStrafeRight ++ "\n" ++
  buildFunction 4 testRiseClamp ++ "\n" ++
  buildFunction 5 testQuitOnClose ++ "\n" ++
  buildFunction 6 testRenderScene

def payloads : List UInt8 :=
  mkPayload layoutMeta.totalSize [
    f.bindDesc.init (uint32ToBytes 0 ++ uint32ToBytes 0 ++ uint32ToBytes 1 ++ uint32ToBytes 1),
    f.shader.init (stringToBytes shaderSource),
    f.blitShader.init (stringToBytes blitShaderSource),
    f.title.init (stringToBytes titleText),
    f.params.init (
      uint32ToBytes 0 ++
      uint32ToBytes (UInt32.ofNat imageWidth) ++
      uint32ToBytes (UInt32.ofNat imageHeight) ++
      uint32ToBytes 0 ++
      uint32ToBytes (UInt32.ofNat camStartY.toNat) ++
      uint32ToBytes 0)
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

def mainAlgorithm  : Algorithm := { fn_idx := IR.mainFnIdx }
def moveFwdAlg     : Algorithm := { fn_idx := u32 2, output := testSchema }
def strafeAlg      : Algorithm := { fn_idx := u32 3, output := testSchema }
def riseClampAlg   : Algorithm := { fn_idx := u32 4, output := testSchema }
def quitOnCloseAlg : Algorithm := { fn_idx := u32 5, output := testSchema }
def renderSceneAlg : Algorithm := { fn_idx := u32 6, output := testSchema }

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← AlgorithmLib.requireOutputDir args
  AlgorithmLib.emitArtifacts outDir #[
    AlgorithmLib.toJsonArtifact "raymarch_demo" Algorithm.gameSetup Algorithm.mainAlgorithm [
      ("test_move_forward",  Algorithm.moveFwdAlg),
      ("test_strafe_right",  Algorithm.strafeAlg),
      ("test_rise_clamp",    Algorithm.riseClampAlg),
      ("test_quit_on_close", Algorithm.quitOnCloseAlg),
      ("test_render_scene",  Algorithm.renderSceneAlg)
    ]]
