import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace Algorithm

-- ---------------------------------------------------------------------------
-- 4096×4096 Mandelbrot set rendered on GPU, written to BMP
--
-- Single CLIF function orchestrates: GPU compute → file write.
-- 2D dispatch with workgroup_size(16, 16) = 256 threads per group.
-- dispatch(4096/16, 4096/16, 1) = (256, 256, 1) — well under 65535 limit.
-- ---------------------------------------------------------------------------

def imageWidth : Nat := 4096
def imageHeight : Nat := 4096
def maxIter : Nat := 1000

def pixelCount : Nat := imageWidth * imageHeight
def pixelBytes : Nat := pixelCount * 4   -- BGRA, 4 bytes per pixel

-- ---------------------------------------------------------------------------
-- BMP header (54 bytes) for 32-bit BGRA image
-- ---------------------------------------------------------------------------

def bmpHeader : List UInt8 :=
  let fileSize : Nat := 54 + pixelBytes
  let bfType := [0x42, 0x4D]
  let bfSize := uint32ToBytes (UInt32.ofNat fileSize)
  let bfReserved := [0, 0, 0, 0]
  let bfOffBits := uint32ToBytes 54
  let biSize := uint32ToBytes 40
  let biWidth := uint32ToBytes (UInt32.ofNat imageWidth)
  -- Negative height = top-down (row 0 is top of image)
  let biHeight := int32ToBytes (Int.negSucc (imageHeight - 1))
  let biPlanes := [1, 0]
  let biBitCount := [32, 0]
  let biCompression := uint32ToBytes 0
  let biSizeImage := uint32ToBytes (UInt32.ofNat pixelBytes)
  let biXPelsPerMeter := uint32ToBytes 2835
  let biYPelsPerMeter := uint32ToBytes 2835
  let biClrUsed := uint32ToBytes 0
  let biClrImportant := uint32ToBytes 0
  bfType ++ bfSize ++ bfReserved ++ bfOffBits ++
  biSize ++ biWidth ++ biHeight ++ biPlanes ++ biBitCount ++
  biCompression ++ biSizeImage ++ biXPelsPerMeter ++ biYPelsPerMeter ++
  biClrUsed ++ biClrImportant

-- ---------------------------------------------------------------------------
-- WGSL compute shader: Mandelbrot iteration + smooth colorization
--
-- 2D dispatch with workgroup_size(16, 16). gid.x = pixel column, gid.y = row.
-- No integer div/mod needed — 2D indexing maps directly to pixel coordinates.
-- Smooth escape: iter + 1 - log2(log2(|z|²)) / log2(2)
-- Bernstein polynomial palette for coloring.
-- ---------------------------------------------------------------------------

def mandelbrotShader : String :=
  let w := toString imageWidth
  let h := toString imageHeight
  let mi := toString maxIter
  "@group(0) @binding(0)\n" ++
  "var<storage, read_write> pixels: array<u32>;\n\n" ++
  "@compute @workgroup_size(16, 16)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  s!"    let width: u32 = {w}u;\n" ++
  s!"    let height: u32 = {h}u;\n" ++
  s!"    let max_iter: u32 = {mi}u;\n" ++
  "    let px: u32 = gid.x;\n" ++
  "    let py: u32 = gid.y;\n" ++
  "    if (px >= width || py >= height) { return; }\n" ++
  "    let idx: u32 = py * width + px;\n" ++
  -- Map pixel to complex plane: Re [-2.2, 0.8], Im [-1.2, 1.2]
  "    let x0: f32 = -2.2 + f32(px) * 3.0 / f32(width - 1u);\n" ++
  "    let y0: f32 = -1.2 + f32(py) * 2.4 / f32(height - 1u);\n" ++
  -- z = z^2 + c
  "    var zr: f32 = 0.0;\n" ++
  "    var zi: f32 = 0.0;\n" ++
  "    var iter: u32 = 0u;\n" ++
  "    var zr2: f32 = 0.0;\n" ++
  "    var zi2: f32 = 0.0;\n" ++
  "    loop {\n" ++
  "        if (iter >= max_iter) { break; }\n" ++
  "        zr2 = zr * zr;\n" ++
  "        zi2 = zi * zi;\n" ++
  "        if (zr2 + zi2 > 4.0) { break; }\n" ++
  "        zi = 2.0 * zr * zi + y0;\n" ++
  "        zr = zr2 - zi2 + x0;\n" ++
  "        iter = iter + 1u;\n" ++
  "    }\n" ++
  -- Smooth coloring with continuous escape value
  "    var r: f32 = 0.0;\n" ++
  "    var g: f32 = 0.0;\n" ++
  "    var b: f32 = 0.0;\n" ++
  "    if (iter < max_iter) {\n" ++
  "        let mag2: f32 = zr2 + zi2;\n" ++
  "        let smooth_iter: f32 = f32(iter) + 1.0 - log2(log2(mag2)) / log2(2.0);\n" ++
  "        let t: f32 = smooth_iter / f32(max_iter);\n" ++
  "        r = 9.0 * (1.0 - t) * t * t * t;\n" ++
  "        g = 15.0 * (1.0 - t) * (1.0 - t) * t * t;\n" ++
  "        b = 8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t;\n" ++
  "    }\n" ++
  "    let ri: u32 = u32(clamp(r * 255.0, 0.0, 255.0));\n" ++
  "    let gi: u32 = u32(clamp(g * 255.0, 0.0, 255.0));\n" ++
  "    let bi: u32 = u32(clamp(b * 255.0, 0.0, 255.0));\n" ++
  "    pixels[idx] = bi | (gi << 8u) | (ri << 16u) | (0xFFu << 24u);\n" ++
  "}\n"

-- ---------------------------------------------------------------------------
-- CLIF IR: GPU pipeline + file write
--
-- fn u0:0: noop (required by CraneliftUnit, never called)
-- fn u0:1: orchestrator
--   1. cl_gpu_init
--   2. cl_gpu_create_buffer (pixelBytes)
--   3. cl_gpu_create_pipeline (mandelbrot shader, 1 rw binding)
--   4. cl_gpu_dispatch (wg_x, wg_y, wg_z) — 2D dispatch
--   5. cl_gpu_download (pixels into payload)
--   6. cl_gpu_cleanup
--   7. cl_file_write (BMP header + pixel data)
--
-- No upload needed — shader generates all pixel data from scratch.
-- All sizes and offsets are CLIF iconst immediates (no payload loads).
-- ---------------------------------------------------------------------------

-- Payload layout
def hdrBase : Nat := 0x40          -- skip CraneliftHashTableContext at offset 0
def bindDesc_off : Nat := 0x100    -- [buf_id=0 (i32), read_only=0 (i32)]
def shader_off : Nat := 0x200      -- WGSL shader (null-terminated)
def shaderRegionSize : Nat := 8192
def filename_off : Nat := shader_off + shaderRegionSize
def filenameRegionSize : Nat := 256
def flag_off : Nat := filename_off + filenameRegionSize
def clifIr_off : Nat := flag_off + 64
def clifIrRegionSize : Nat := 4096
def bmpHeader_off : Nat := clifIr_off + clifIrRegionSize
def pixels_off : Nat := bmpHeader_off + 54
def totalPayload : Nat := pixels_off + pixelBytes

def wgX : Nat := imageWidth / 16    -- 256
def wgY : Nat := imageHeight / 16   -- 256

def clifIrSource : String :=
  let sh := toString shader_off
  let bd := toString bindDesc_off
  let fn_out := toString filename_off
  let bmp := toString bmpHeader_off
  let bmpTotalSize := toString (54 + pixelBytes)
  let px := toString pixels_off
  let dataSz := toString pixelBytes
  let wgXStr := toString wgX
  let wgYStr := toString wgY
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n\n" ++
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++
  "    sig1 = (i64, i64) -> i32 system_v\n" ++
  "    sig2 = (i64, i64, i64, i32) -> i32 system_v\n" ++
  "    sig3 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v\n" ++
  "    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "    fn0 = %cl_gpu_init sig0\n" ++
  "    fn1 = %cl_gpu_create_buffer sig1\n" ++
  "    fn2 = %cl_gpu_create_pipeline sig2\n" ++
  "    fn3 = %cl_gpu_dispatch sig4\n" ++
  "    fn4 = %cl_gpu_download sig3\n" ++
  "    fn5 = %cl_gpu_cleanup sig0\n" ++
  "    fn6 = %cl_file_write sig5\n" ++
  "\n" ++
  "block0(v0: i64):\n" ++
  "    call fn0(v0)\n" ++
  s!"    v1 = iconst.i64 {dataSz}\n" ++
  "    v2 = call fn1(v0, v1)\n" ++
  s!"    v3 = iconst.i64 {sh}\n" ++
  s!"    v4 = iconst.i64 {bd}\n" ++
  "    v5 = iconst.i32 1\n" ++
  "    v6 = call fn2(v0, v3, v4, v5)\n" ++
  s!"    v7 = iconst.i32 {wgXStr}\n" ++
  s!"    v8 = iconst.i32 {wgYStr}\n" ++
  "    v9 = call fn3(v0, v6, v7, v8, v5)\n" ++
  s!"    v10 = iconst.i64 {px}\n" ++
  "    v11 = call fn4(v0, v2, v10, v1)\n" ++
  "    call fn5(v0)\n" ++
  s!"    v12 = iconst.i64 {fn_out}\n" ++
  s!"    v13 = iconst.i64 {bmp}\n" ++
  "    v14 = iconst.i64 0\n" ++
  s!"    v15 = iconst.i64 {bmpTotalSize}\n" ++
  "    v16 = call fn6(v0, v12, v13, v14, v15)\n" ++
  "    return\n" ++
  "}\n"

-- ---------------------------------------------------------------------------
-- Payload construction
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  let reserved := zeros hdrBase
  let hdrPad := zeros (bindDesc_off - hdrBase)
  -- Binding descriptor at bindDesc_off: [buf_id=0 (i32), read_only=0 (i32)]
  let bindDesc := uint32ToBytes 0 ++ uint32ToBytes 0
  let bindPad := zeros (shader_off - bindDesc_off - 8)
  let shaderBytes := padTo (stringToBytes mandelbrotShader) shaderRegionSize
  let filenameBytes := padTo (stringToBytes "mandelbrot.bmp") filenameRegionSize
  let flagBytes := uint64ToBytes 0
  let flagPad := zeros (clifIr_off - flag_off - 8)
  let clifPad := zeros clifIrRegionSize
  let bmpBytes := bmpHeader
  reserved ++ hdrPad ++
  bindDesc ++ bindPad ++
  shaderBytes ++ filenameBytes ++ flagBytes ++ flagPad ++
  clifPad ++ bmpBytes

-- ---------------------------------------------------------------------------
-- Algorithm definition
--
-- Actions:
--   [0] ClifCall: synchronous call to CLIF fn 1
-- ---------------------------------------------------------------------------

def drawConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := payloads.length + pixelBytes,
  context_offset := 0
}

def drawAlgorithm : Algorithm :=
  let clifCallAction : Action :=
    { kind := .ClifCall, dst := u32 0, src := u32 1, offset := u32 0, size := u32 0 }
  {
    actions := [clifCallAction],
    payloads := payloads,
    cranelift_units := 0,
    timeout_ms := some 120000
  }

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.drawConfig Algorithm.drawAlgorithm
  IO.println (Json.compress json)
