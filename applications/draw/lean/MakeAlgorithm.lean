import AlgorithmLib

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout

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
-- Memory layout (typed field handles)
-- ---------------------------------------------------------------------------

structure Fields where
  reserved   : Fld (.bytes 64)
  bindDesc   : Fld (.bytes 8)
  shader     : Fld (.bytes 8192)
  filename   : Fld (.bytes 256)
  flag       : Fld (.bytes 64)
  clifIr     : Fld (.bytes 4096)
  bmpHeader  : Fld (.bytes 54)
  pixels     : Fld (.bytes pixelBytes)

def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved  ← field (.bytes 64)       -- [0x00, 0x40)  GPU ctx ptr
  skip 192                                 -- pad to 0x100
  let bindDesc  ← field (.bytes 8)        -- [buf_id=0 (i32), read_only=0 (i32)]
  skip 248                                 -- pad to 0x200
  let shader    ← field (.bytes 8192)     -- WGSL shader (null-terminated)
  let filename  ← field (.bytes 256)      -- output filename
  let flag      ← field (.bytes 64)       -- sync flag
  let clifIr    ← field (.bytes 4096)     -- CLIF IR region
  let bmpHeader ← field (.bytes 54)       -- BMP file header
  let pixels    ← field (.bytes pixelBytes)
  pure { reserved, bindDesc, shader, filename, flag, clifIr, bmpHeader, pixels }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

def wgX : Nat := imageWidth / 16    -- 256
def wgY : Nat := imageHeight / 16   -- 256

-- ---------------------------------------------------------------------------
-- CLIF IR: GPU pipeline + file write
-- ---------------------------------------------------------------------------

open AlgorithmLib.IR in
def clifIrSource : String := buildProgram do
  let gpu ← declareGpuFFI
  let fnWr ← declareFileWrite
  let ptr ← entryBlock
  callVoid gpu.fnInit [ptr]
  let dataSz ← iconst64 pixelBytes
  let bufId  ← call gpu.fnCreateBuffer [ptr, dataSz]
  let shOff  ← fldOffset f.shader
  let bdOff  ← fldOffset f.bindDesc
  let one32  ← iconst32 1
  let pipeId ← call gpu.fnCreatePipeline [ptr, shOff, bdOff, one32]
  let wgx    ← iconst32 wgX
  let wgy    ← iconst32 wgY
  let _      ← call gpu.fnDispatch [ptr, pipeId, wgx, wgy, one32]
  let pxOff  ← fldOffset f.pixels
  let _      ← call gpu.fnDownload [ptr, bufId, pxOff, dataSz]
  callVoid gpu.fnCleanup [ptr]
  let total  ← iconst64 (54 + pixelBytes)
  let _      ← fldWriteFile0 ptr fnWr f.filename f.bmpHeader total
  ret

-- ---------------------------------------------------------------------------
-- Payload & config
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  mkPayload (f.bmpHeader.offset + 54) [
    f.bindDesc.init (uint32ToBytes 0 ++ uint32ToBytes 0),
    f.shader.init (stringToBytes mandelbrotShader),
    f.filename.init (stringToBytes "mandelbrot.bmp"),
    f.bmpHeader.init bmpHeader
  ]

def drawConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := layoutMeta.totalSize,
  context_offset := 0
}

def drawAlgorithm : Algorithm := {
    actions := [IR.clifCallAction],
    payloads := payloads,
    cranelift_units := 0,
    timeout_ms := some 120000
  }

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.drawConfig Algorithm.drawAlgorithm
  IO.println (Json.compress json)
