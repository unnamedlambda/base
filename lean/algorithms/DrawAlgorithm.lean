import AlgorithmLib

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout
open AlgorithmLib.WGSL

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
  let pixels : AlgorithmLib.WGSL.Expr (.arr .u32) := ⟨"pixels"⟩
  buildShader
    [{ binding := 0, name := "pixels", ty := .arr .u32 }]
    []
    [.constU "IMG_W" imageWidth,
     .constU "IMG_H" imageHeight,
     .constU "MAX_ITER" maxIter]
    { wgX := 16, wgY := 16 }
    do
      let px   ← letV "px" gidX
      let py   ← letV "py" gidY
      ifB ((px .>= ⟨"IMG_W"⟩) .|| (py .>= ⟨"IMG_H"⟩)) retV
      let idx  ← letV "idx" (py * ⟨"IMG_W"⟩ + px)
      let x0   ← letV "x0" (litF "-2.2" + f32OfU px * litF "3.0" / f32OfU (⟨"IMG_W"⟩ - litU 1))
      let y0   ← letV "y0" (litF "-1.2" + f32OfU py * litF "2.4" / f32OfU (⟨"IMG_H"⟩ - litU 1))
      let zr   ← varV "zr" (litF "0.0")
      let zi   ← varV "zi" (litF "0.0")
      let iter ← varV "iter" (litU 0)
      let zr2  ← varV "zr2" (litF "0.0")
      let zi2  ← varV "zi2" (litF "0.0")
      loopB do
        ifB (iter .>= ⟨"MAX_ITER"⟩) breakS
        assign zr2 (zr * zr)
        assign zi2 (zi * zi)
        ifB (zr2 + zi2 .> litF "4.0") breakS
        assign zi (litF "2.0" * zr * zi + y0)
        assign zr (zr2 - zi2 + x0)
        assign iter (iter + litU 1)
      let r ← varV "r" (litF "0.0")
      let g ← varV "g" (litF "0.0")
      let b ← varV "b" (litF "0.0")
      ifB (iter .< ⟨"MAX_ITER"⟩) do
        let mag2       ← letV "mag2" (zr2 + zi2)
        let smoothIter ← letV "smooth_iter" (f32OfU iter + litF "1.0" - wLog2 (wLog2 mag2) / wLog2 (litF "2.0"))
        let t          ← letV "t" (smoothIter / f32OfU ⟨"MAX_ITER"⟩)
        assign r (litF "9.0" * (litF "1.0" - t) * t * t * t)
        assign g (litF "15.0" * (litF "1.0" - t) * (litF "1.0" - t) * t * t)
        assign b (litF "8.5" * (litF "1.0" - t) * (litF "1.0" - t) * (litF "1.0" - t) * t)
      let ri ← letV "ri" (u32OfF (wClamp (r * litF "255.0") (litF "0.0") (litF "255.0")))
      let gi ← letV "gi" (u32OfF (wClamp (g * litF "255.0") (litF "0.0") (litF "255.0")))
      let bi ← letV "bi" (u32OfF (wClamp (b * litF "255.0") (litF "0.0") (litF "255.0")))
      assign (arrIdx pixels idx) (((bi .| (gi .<< litU 8)) .| (ri .<< litU 16)) .| (litU 0xFF .<< litU 24))

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
  gpuInit gpu ptr
  let dataSz ← iconst64 pixelBytes
  let bufId  ← gpuCreateBuffer gpu ptr dataSz
  let shOff  ← fldOffset f.shader
  let bdOff  ← fldOffset f.bindDesc
  let one32  ← iconst32 1
  let pipeId ← gpuCreatePipeline gpu ptr shOff bdOff one32
  let wgx    ← iconst32 wgX
  let wgy    ← iconst32 wgY
  let _      ← gpuDispatch gpu ptr pipeId wgx wgy one32
  let pxOff  ← fldOffset f.pixels
  let _      ← gpuDownload gpu ptr bufId pxOff dataSz
  gpuCleanup gpu ptr
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

def drawConfig : Setup := {
  cranelift_ir := clifIrSource,
  memory_size := layoutMeta.totalSize,
  context_offset := 0,
  initial_memory := payloads
}

def drawAlgorithm : Algorithm := {
    fn_idx := IR.mainFnIdx
  }

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[toJsonEntry "draw_app" Algorithm.drawConfig Algorithm.drawAlgorithm]
