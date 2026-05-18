import AlgorithmLib

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout
open AlgorithmLib.PTX

namespace Algorithm

-- ---------------------------------------------------------------------------
-- SAXPY: y[i] = 2.0 * x[i] + y[i]
--
-- CUDA PTX kernel launched via CLIF FFI.
-- Scalar a=2.0 is hardcoded in PTX for simplicity (no scalar-param FFI).
-- The harness patches N, x[], y[] data and output filename at runtime.
-- ---------------------------------------------------------------------------

-- PTX kernel: y[i] = 2.0 * x[i] + y[i]
-- Block size 256, grid = N/256 (harness ensures N is multiple of 256).
def saxpyPtx : String := buildModuleWith { version := "7.0", target := "sm_50" } [{
  name := "main", params := ["x_ptr", "y_ptr"], body := do
  let xPtr ← ldParam "x_ptr"
  let yPtr ← ldParam "y_ptr"
  let bid ← freshR; movR bid ctaX
  let tid ← freshR; movR tid tidX
  let gid ← freshR; madLoRC gid bid 256 tid
  let off ← freshRd; cvtU64 off gid; shlRd off off 2
  let xa ← freshRd; addRd xa xPtr off
  let ya ← freshRd; addRd ya yPtr off
  let fx ← freshF; ldGlobalF fx xa
  let fy ← freshF; ldGlobalF fy ya
  let fa ← freshF; movFC fa 0x40000000  -- 2.0f
  fmaRn fy fa fx fy
  stGlobalF ya fy
  ptxRet }]

-- ---------------------------------------------------------------------------
-- Memory layout
-- ---------------------------------------------------------------------------

def maxElems : Nat := 1048576   -- 1M elements max
def maxDataBytes : Nat := maxElems * 4 * 2  -- x[] then y[]

structure Fields where
  reserved   : Fld (.bytes 64)     -- CUDA context pointer at offset 0
  nElems     : Fld .i32            -- element count N (patched by harness)
  ptxSrc     : Fld (.bytes 4096)   -- PTX source (null-terminated)
  bindDesc   : Fld (.bytes 8)      -- 2 × i32 buf_ids: [x=0, y=1]
  filename   : Fld (.bytes 256)    -- output filename (null-terminated)
  dataRegion : Fld (.bytes maxDataBytes)  -- x[0..N] then y[0..N]

def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved   ← field (.bytes 64)
  skip 192                               -- pad to 0x100
  let nElems     ← field .i32
  skip 252                               -- pad to 0x200
  let ptxSrc     ← field (.bytes 4096)
  let bindDesc   ← field (.bytes 8)
  let filename   ← field (.bytes 256)
  skip 256                               -- pad before data
  let dataRegion ← field (.bytes maxDataBytes)
  pure { reserved, nElems, ptxSrc, bindDesc, filename, dataRegion }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

-- ---------------------------------------------------------------------------
-- CLIF IR: CUDA SAXPY pipeline
--
-- init → create 2 buffers → upload x,y → launch PTX → download y → file write
-- ---------------------------------------------------------------------------

open AlgorithmLib.IR in
def clifIrSource : String := buildProgram do
  let cuda ← declareCudaFFI
  let fnWr ← declareFileWrite
  let ptr  ← entryBlock

  -- Init CUDA context
  cudaInit cuda ptr

  -- Load N (i32) from memory, compute buffer size = N * 4
  let nAddr ← absAddr ptr f.nElems.offset
  let nVal  ← load32 nAddr
  let nVal64 ← sextend64 nVal
  let four  ← iconst64 4
  let bufSz ← imul nVal64 four

  -- Create 2 device buffers
  let xBuf ← cudaCreateBuffer cuda ptr bufSz
  let yBuf ← cudaCreateBuffer cuda ptr bufSz

  -- Upload x from dataRegion, y from dataRegion + bufSz
  let dataOff ← iconst64 f.dataRegion.offset
  let _  ← cudaUpload cuda ptr xBuf dataOff bufSz
  let yOff ← iadd dataOff bufSz
  let _  ← cudaUpload cuda ptr yBuf yOff bufSz

  -- Compute grid dimensions: ceil(N / 256)
  let n255  ← iconst32 255
  let nPlus ← iadd nVal n255
  let c256  ← iconst32 256
  let gridX ← udiv nPlus c256
  let one32 ← iconst32 1

  -- Launch PTX: 2 buffer bindings, grid(gridX,1,1), block(256,1,1)
  let ptxOff  ← iconst64 f.ptxSrc.offset
  let two32   ← iconst32 2
  let bindOff ← iconst64 f.bindDesc.offset
  let _ ← cudaLaunch cuda ptr ptxOff two32 bindOff
                               gridX one32 one32
                               c256 one32 one32

  -- Download y result
  let _ ← cudaDownload cuda ptr yBuf yOff bufSz

  -- Write y to verify file
  let fnameOff ← iconst64 f.filename.offset
  let zero64   ← iconst64 0
  let _ ← call fnWr [ptr, fnameOff, yOff, zero64, bufSz]

  -- Cleanup
  cudaCleanup cuda ptr
  ret

-- ---------------------------------------------------------------------------
-- Payloads
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  mkPayload layoutMeta.totalSize [
    f.ptxSrc.init (stringToBytes saxpyPtx),
    f.bindDesc.init (uint32ToBytes 0 ++ uint32ToBytes 1)
  ]

def saxpyConfig : Setup := {
  cranelift_ir := clifIrSource,
  memory_size := layoutMeta.totalSize,
  initial_memory := payloads
}

def saxpyAlgorithm : Algorithm := {
  fn_idx := IR.mainFnIdx
}

def artifacts : Array Json :=
  #[toJsonEntry "saxpy_algorithm" saxpyConfig saxpyAlgorithm]

end Algorithm
