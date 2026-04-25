import AlgorithmLib

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout

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
def saxpyPtx : String :=
  ".version 7.0\n" ++
  ".target sm_50\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 x_ptr,\n" ++
  "    .param .u64 y_ptr\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .u32 %r0, %r1;\n" ++
  "    .reg .u64 %rx, %ry, %off;\n" ++
  "    .reg .f32 %fx, %fy, %fa;\n" ++
  "\n" ++
  "    mov.u32 %r0, %ctaid.x;\n" ++
  "    mov.u32 %r1, %tid.x;\n" ++
  "    mad.lo.u32 %r0, %r0, 256, %r1;\n" ++
  "\n" ++
  "    cvt.u64.u32 %off, %r0;\n" ++
  "    shl.b64 %off, %off, 2;\n" ++
  "\n" ++
  "    ld.param.u64 %rx, [x_ptr];\n" ++
  "    ld.param.u64 %ry, [y_ptr];\n" ++
  "\n" ++
  "    add.u64 %rx, %rx, %off;\n" ++
  "    add.u64 %ry, %ry, %off;\n" ++
  "\n" ++
  "    ld.global.f32 %fx, [%rx];\n" ++
  "    ld.global.f32 %fy, [%ry];\n" ++
  "    mov.f32 %fa, 0f40000000;\n" ++     -- 2.0f IEEE 754
  "    fma.rn.f32 %fy, %fa, %fx, %fy;\n" ++
  "    st.global.f32 [%ry], %fy;\n" ++
  "\n" ++
  "    ret;\n" ++
  "}\n"

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
  callVoid cuda.fnInit [ptr]

  -- Load N (i32) from memory, compute buffer size = N * 4
  let nAddr ← absAddr ptr f.nElems.offset
  let nVal  ← load32 nAddr
  let nVal64 ← sextend64 nVal
  let four  ← iconst64 4
  let bufSz ← imul nVal64 four

  -- Create 2 device buffers
  let xBuf ← call cuda.fnCreateBuffer [ptr, bufSz]
  let yBuf ← call cuda.fnCreateBuffer [ptr, bufSz]

  -- Upload x from dataRegion, y from dataRegion + bufSz
  let dataOff ← iconst64 f.dataRegion.offset
  let _  ← call cuda.fnUpload [ptr, xBuf, dataOff, bufSz]
  let yOff ← iadd dataOff bufSz
  let _  ← call cuda.fnUpload [ptr, yBuf, yOff, bufSz]

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
  let _ ← call cuda.fnLaunch [ptr, ptxOff, two32, bindOff,
                               gridX, one32, one32,
                               c256, one32, one32]

  -- Download y result
  let _ ← call cuda.fnDownload [ptr, yBuf, yOff, bufSz]

  -- Write y to verify file
  let fnameOff ← iconst64 f.filename.offset
  let zero64   ← iconst64 0
  let _ ← call fnWr [ptr, fnameOff, yOff, zero64, bufSz]

  -- Cleanup
  callVoid cuda.fnCleanup [ptr]
  ret

-- ---------------------------------------------------------------------------
-- Payloads
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  mkPayload layoutMeta.totalSize [
    f.ptxSrc.init (stringToBytes saxpyPtx),
    f.bindDesc.init (uint32ToBytes 0 ++ uint32ToBytes 1)
  ]

def saxpyConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := layoutMeta.totalSize,
  context_offset := 0,
  initial_memory := payloads
}

def saxpyAlgorithm : Algorithm := {
  actions := [IR.clifCallAction],
  cranelift_units := 0,
  timeout_ms := some 120000
}

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.saxpyConfig Algorithm.saxpyAlgorithm
  IO.println (Json.compress json)
