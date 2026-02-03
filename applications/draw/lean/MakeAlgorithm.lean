import Lean
import Std

open Lean

namespace Algorithm

instance : ToJson UInt8 where
  toJson n := toJson n.toNat

instance : ToJson (List UInt8) where
  toJson lst := toJson (lst.map (·.toNat))

instance : ToJson UInt32 where
  toJson n := toJson n.toNat

instance : ToJson UInt64 where
  toJson n := toJson n.toNat

inductive Kind where
  | SimdLoad
  | SimdAdd
  | SimdMul
  | SimdStore
  | SimdLoadI32
  | SimdAddI32
  | SimdMulI32
  | SimdStoreI32
  | MemCopy
  | FileRead
  | FileWrite
  | Approximate
  | Choose
  | Compare
  | Timestamp
  | ConditionalJump
  | Fence
  | AsyncDispatch
  | Wait
  | MemWrite
  | Dispatch
  deriving Repr

instance : ToJson Kind where
  toJson
    | .SimdLoad => "simd_load"
    | .SimdAdd => "simd_add"
    | .SimdMul => "simd_mul"
    | .SimdStore => "simd_store"
    | .SimdLoadI32 => "simd_load_i32"
    | .SimdAddI32 => "simd_add_i32"
    | .SimdMulI32 => "simd_mul_i32"
    | .SimdStoreI32 => "simd_store_i32"
    | .MemCopy => "mem_copy"
    | .FileRead => "file_read"
    | .FileWrite => "file_write"
    | .Approximate => "approximate"
    | .Choose => "choose"
    | .Compare => "compare"
    | .Timestamp => "timestamp"
    | .ConditionalJump => "conditional_jump"
    | .Fence => "fence"
    | .AsyncDispatch => "async_dispatch"
    | .Wait => "wait"
    | .MemWrite => "mem_write"
    | .Dispatch => "dispatch"

structure Action where
  kind : Kind
  dst : UInt32
  src : UInt32
  offset : UInt32
  size : UInt32
  deriving Repr

instance : ToJson Action where
  toJson a := Json.mkObj [
    ("kind", toJson a.kind),
    ("dst", toJson a.dst),
    ("src", toJson a.src),
    ("offset", toJson a.offset),
    ("size", toJson a.size)
  ]

structure State where
  regs_per_unit : Nat
  unit_scratch_offsets : List Nat
  unit_scratch_size : Nat
  shared_data_offset : Nat
  shared_data_size : Nat
  gpu_offset : Nat
  gpu_size : Nat
  computational_regs : Nat
  file_buffer_size : Nat
  gpu_shader_offsets : List Nat
  deriving Repr

instance : ToJson State where
  toJson s := Json.mkObj [
    ("regs_per_unit", toJson s.regs_per_unit),
    ("unit_scratch_offsets", toJson s.unit_scratch_offsets),
    ("unit_scratch_size", toJson s.unit_scratch_size),
    ("shared_data_offset", toJson s.shared_data_offset),
    ("shared_data_size", toJson s.shared_data_size),
    ("gpu_offset", toJson s.gpu_offset),
    ("gpu_size", toJson s.gpu_size),
    ("computational_regs", toJson s.computational_regs),
    ("file_buffer_size", toJson s.file_buffer_size),
    ("gpu_shader_offsets", toJson s.gpu_shader_offsets)
  ]

structure QueueSpec where
  capacity : Nat
  deriving Repr

instance : ToJson QueueSpec where
  toJson q := Json.mkObj [
    ("capacity", toJson q.capacity)
  ]

structure UnitSpec where
  simd_units : Nat
  gpu_units : Nat
  computational_units : Nat
  file_units : Nat
  network_units : Nat
  memory_units : Nat
  ffi_units : Nat
  backends_bits : UInt32
  features_bits : UInt64
  deriving Repr

instance : ToJson UnitSpec where
  toJson u := Json.mkObj [
    ("simd_units", toJson u.simd_units),
    ("gpu_units", toJson u.gpu_units),
    ("computational_units", toJson u.computational_units),
    ("file_units", toJson u.file_units),
    ("network_units", toJson u.network_units),
    ("memory_units", toJson u.memory_units),
    ("ffi_units", toJson u.ffi_units),
    ("backends_bits", toJson u.backends_bits),
    ("features_bits", toJson u.features_bits)
  ]

structure Algorithm where
  actions : List Action
  payloads : List UInt8
  state : State
  queues : QueueSpec
  units : UnitSpec
  simd_assignments : List UInt8
  computational_assignments : List UInt8
  memory_assignments : List UInt8
  file_assignments : List UInt8
  network_assignments : List UInt8
  ffi_assignments : List UInt8
  gpu_assignments : List UInt8
  worker_threads : Option Nat
  blocking_threads : Option Nat
  stack_size : Option Nat
  timeout_ms : Option Nat
  thread_name_prefix : Option String
  deriving Repr

instance : ToJson Algorithm where
  toJson alg := Json.mkObj [
    ("actions", toJson alg.actions),
    ("payloads", toJson alg.payloads),
    ("state", toJson alg.state),
    ("queues", toJson alg.queues),
    ("units", toJson alg.units),
    ("simd_assignments", toJson alg.simd_assignments),
    ("computational_assignments", toJson alg.computational_assignments),
    ("memory_assignments", toJson alg.memory_assignments),
    ("file_assignments", toJson alg.file_assignments),
    ("network_assignments", toJson alg.network_assignments),
    ("ffi_assignments", toJson alg.ffi_assignments),
    ("gpu_assignments", toJson alg.gpu_assignments),
    ("worker_threads", toJson alg.worker_threads),
    ("blocking_threads", toJson alg.blocking_threads),
    ("stack_size", toJson alg.stack_size),
    ("timeout_ms", toJson alg.timeout_ms),
    ("thread_name_prefix", toJson alg.thread_name_prefix)
  ]

def u32 (n : Nat) : UInt32 := UInt32.ofNat n

def stringToBytes (s : String) : List UInt8 :=
  s.toUTF8.toList ++ [0]

def padTo (bytes : List UInt8) (targetLen : Nat) : List UInt8 :=
  bytes ++ List.replicate (targetLen - bytes.length) 0

def zeros (n : Nat) : List UInt8 :=
  List.replicate n 0

def uint32ToBytes (n : UInt32) : List UInt8 :=
  let b0 := UInt8.ofNat (n.toNat &&& 0xFF)
  let b1 := UInt8.ofNat ((n.toNat >>> 8) &&& 0xFF)
  let b2 := UInt8.ofNat ((n.toNat >>> 16) &&& 0xFF)
  let b3 := UInt8.ofNat ((n.toNat >>> 24) &&& 0xFF)
  [b0, b1, b2, b3]

def uint64ToBytes (n : UInt64) : List UInt8 :=
  let b0 := UInt8.ofNat (n.toNat &&& 0xFF)
  let b1 := UInt8.ofNat ((n.toNat >>> 8) &&& 0xFF)
  let b2 := UInt8.ofNat ((n.toNat >>> 16) &&& 0xFF)
  let b3 := UInt8.ofNat ((n.toNat >>> 24) &&& 0xFF)
  let b4 := UInt8.ofNat ((n.toNat >>> 32) &&& 0xFF)
  let b5 := UInt8.ofNat ((n.toNat >>> 40) &&& 0xFF)
  let b6 := UInt8.ofNat ((n.toNat >>> 48) &&& 0xFF)
  let b7 := UInt8.ofNat ((n.toNat >>> 56) &&& 0xFF)
  [b0, b1, b2, b3, b4, b5, b6, b7]

def int32ToBytes (n : Int) : List UInt8 :=
  let two32 : Int := Int.ofNat ((2:Nat) ^ 32)
  let u : UInt32 :=
    if n >= 0 then
      UInt32.ofNat n.toNat
    else
      let m : Int := n + two32
      UInt32.ofNat m.toNat
  uint32ToBytes u

-- Image and batching configuration

def imageWidth : Nat := 512

def imageHeight : Nat := 512

def batchRows : Nat := 64

-- 32-bit BMP header (54 bytes) for imageWidth x imageHeight
-- File size = 54 + imageWidth * imageHeight * 4
-- BMP is bottom-up, we place rows accordingly in memory

def bmpHeader32 : List UInt8 :=
  let pixelDataSize : Nat := imageWidth * imageHeight * 4
  let fileSize : Nat := 54 + pixelDataSize

  let bfType := [0x42, 0x4D]
  let bfSize := uint32ToBytes (UInt32.ofNat fileSize)
  let bfReserved := [0, 0, 0, 0]
  let bfOffBits := uint32ToBytes 54

  let biSize := uint32ToBytes 40
  let biWidth := uint32ToBytes (UInt32.ofNat imageWidth)
  let biHeight := uint32ToBytes (UInt32.ofNat imageHeight)
  let biPlanes := [1, 0]
  let biBitCount := [32, 0]
  let biCompression := uint32ToBytes 0
  let biSizeImage := uint32ToBytes (UInt32.ofNat pixelDataSize)
  let biXPelsPerMeter := uint32ToBytes 2835
  let biYPelsPerMeter := uint32ToBytes 2835
  let biClrUsed := uint32ToBytes 0
  let biClrImportant := uint32ToBytes 0

  bfType ++ bfSize ++ bfReserved ++ bfOffBits ++
  biSize ++ biWidth ++ biHeight ++ biPlanes ++ biBitCount ++
  biCompression ++ biSizeImage ++ biXPelsPerMeter ++ biYPelsPerMeter ++
  biClrUsed ++ biClrImportant

-- GPU shader: read i32 heightmap and output BGRA pixels

def heightmapShader : String :=
  let widthStr := toString imageWidth
  let heightStr := toString batchRows
  "@group(0) @binding(0)\n" ++
  "var<storage, read_write> data: array<u32>;\n\n" ++
  "@compute @workgroup_size(64)\n" ++
  "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n" ++
  s!"    let width: u32 = {widthStr}u;\n" ++
  s!"    let height: u32 = {heightStr}u;\n" ++
  "    let total: u32 = width * height;\n" ++
  "    let idx: u32 = id.x;\n" ++
  "    if (idx >= total) { return; }\n" ++
  "    let h_i: i32 = bitcast<i32>(data[idx]);\n" ++
  "    let h: f32 = f32(h_i) / 131072.0;\n" ++
  "    let t: f32 = clamp(h, 0.0, 1.0);\n" ++
  "    let x: f32 = f32(idx % width) / f32(width - 1u);\n" ++
  "    let y: f32 = f32(idx / width) / f32(height - 1u);\n" ++
  "    let r: f32 = t;\n" ++
  "    let g: f32 = t * t * (0.5 + 0.5 * x);\n" ++
  "    let b: f32 = (1.0 - t) * (0.5 + 0.5 * y);\n" ++
  "    let ri: u32 = u32(clamp(r * 255.0, 0.0, 255.0));\n" ++
  "    let gi: u32 = u32(clamp(g * 255.0, 0.0, 255.0));\n" ++
  "    let bi: u32 = u32(clamp(b * 255.0, 0.0, 255.0));\n" ++
  "    let ai: u32 = 255u;\n" ++
  "    data[idx] = bi | (gi << 8u) | (ri << 16u) | (ai << 24u);\n" ++
  "}\n"

-- Table of x values (i32) in range [-256, 255]
def buildXTable (i width : Nat) : List UInt8 :=
  if _h : i < width then
    let v : Int := Int.ofNat i - Int.ofNat (width / 2)
    int32ToBytes v ++ buildXTable (i + 1) width
  else
    []

-- Table of y vectors, each row repeated 4x (i32)
def buildYTable (i height : Nat) : List UInt8 :=
  if _h : i < height then
    let v : Int := Int.ofNat i - Int.ofNat (height / 2)
    let row := int32ToBytes v ++ int32ToBytes v ++ int32ToBytes v ++ int32ToBytes v
    row ++ buildYTable (i + 1) height
  else
    []

-- Dimensions and batch sizes

def batches : Nat := imageHeight / batchRows

def blocksPerRow : Nat := imageWidth / 4

def pixelsPerBatch : Nat := imageWidth * batchRows

def batchBytes : Nat := pixelsPerBatch * 4

def pixelBytes : Nat := imageWidth * imageHeight * 4

-- Layout offsets

def shaderOffset : Nat := 0

def shaderSize : Nat := 4096

def filenameOffset : Nat := shaderOffset + shaderSize

def filenameSize : Nat := 256

def flagsOffset : Nat := filenameOffset + filenameSize

def flagCount : Nat := 1 + 3 * batches + 1

def flagsSize : Nat := flagCount * 8

def headerOffset : Nat := flagsOffset + flagsSize

def headerSize : Nat := 54

def pixelsOffset : Nat := headerOffset + headerSize

def hm0Offset : Nat := pixelsOffset + pixelBytes

def hm1Offset : Nat := hm0Offset + batchBytes

def out0Offset : Nat := hm1Offset + batchBytes

def out1Offset : Nat := out0Offset + batchBytes

def xTableOffset : Nat := out1Offset + batchBytes

def xTableSize : Nat := imageWidth * 4

def yTableOffset : Nat := xTableOffset + xTableSize

def yTableSize : Nat := imageHeight * 16

def dataSize : Nat := yTableOffset + yTableSize

def simdScratchSize : Nat := 4096

def simdScratchTotal : Nat := simdScratchSize * 4

def fileBufferSize : Nat := 2_000_000

-- Flags

def flagAlways : Nat := flagsOffset

def simdFlagOffset (i : Nat) : Nat := flagsOffset + 8 * (1 + i)

def gpuFlagOffset (i : Nat) : Nat := flagsOffset + 8 * (1 + batches + i)

def memFlagOffset (i : Nat) : Nat := flagsOffset + 8 * (1 + batches + batches + i)

def fileFlagOffset : Nat := flagsOffset + 8 * (1 + 3 * batches)

-- Offsets for batch buffers

def hmOffset (i : Nat) : Nat := if i % 2 == 0 then hm0Offset else hm1Offset

def outOffset (i : Nat) : Nat := if i % 2 == 0 then out0Offset else out1Offset

def destOffset (i : Nat) : Nat :=
  let rowStart := (imageHeight - (i + 1) * batchRows) * imageWidth * 4
  pixelsOffset + rowStart

-- SIMD action generation

def simdActionsForBatch (batchIndex : Nat) : List Action :=
  let baseRow := batchIndex * batchRows
  let hmBase := hmOffset batchIndex
  let rec rowLoop (r : Nat) : List Action :=
    if _h : r < batchRows then
      let yOffset := yTableOffset + (baseRow + r) * 16
      let rowBase := hmBase + (r * imageWidth * 4)
      let rec colLoop (c : Nat) : List Action :=
        if _h2 : c < blocksPerRow then
          let xOffset := xTableOffset + (c * 16)
          let dstOffset := rowBase + (c * 16)
          let a0 : Action := { kind := .SimdLoadI32, dst := u32 0, src := u32 xOffset, offset := 0, size := 16 }
          let a1 : Action := { kind := .SimdLoadI32, dst := u32 1, src := u32 yOffset, offset := 0, size := 16 }
          let a2 : Action := { kind := .SimdMulI32, dst := u32 2, src := u32 0, offset := u32 0, size := 0 }
          let a3 : Action := { kind := .SimdMulI32, dst := u32 3, src := u32 1, offset := u32 1, size := 0 }
          let a4 : Action := { kind := .SimdAddI32, dst := u32 4, src := u32 2, offset := u32 3, size := 0 }
          let a5 : Action := { kind := .SimdStoreI32, dst := u32 0, src := u32 4, offset := u32 dstOffset, size := 16 }
          a0 :: a1 :: a2 :: a3 :: a4 :: a5 :: colLoop (c + 1)
        else
          []
      colLoop 0 ++ rowLoop (r + 1)
    else
      []
  rowLoop 0

-- GPU and memory actions

def gpuActionForBatch (i : Nat) : Action :=
  { kind := .Dispatch, dst := u32 (outOffset i), src := u32 (hmOffset i), offset := 0, size := u32 batchBytes }

def memActionForBatch (i : Nat) : Action :=
  { kind := .MemCopy, dst := u32 (destOffset i), src := u32 (outOffset i), offset := 0, size := u32 batchBytes }

-- Control actions (AsyncDispatch + Wait), with overlap

def broadcastMask : UInt32 := UInt32.ofNat 0x80000000

def broadcastSize (n : Nat) : UInt32 := broadcastMask + UInt32.ofNat n

def simdActionsPerBatch : Nat := batchRows * blocksPerRow * 6

def simdStartIndex (i : Nat) : Nat := 1 + i * simdActionsPerBatch

def gpuStartIndex (simdCount : Nat) (i : Nat) : Nat := 1 + simdCount + i

def memStartIndex (simdCount gpuCount : Nat) (i : Nat) : Nat := 1 + simdCount + gpuCount + i

def fileWriteIndex (simdCount gpuCount memCount : Nat) : Nat := 1 + simdCount + gpuCount + memCount


def controlActions (simdCount gpuCount memCount : Nat) : List Action :=
  let asyncSimd (i : Nat) : Action :=
    { kind := .AsyncDispatch, dst := u32 1, src := u32 (simdStartIndex i), offset := u32 (simdFlagOffset i), size := broadcastSize simdActionsPerBatch }
  let waitFlag (off : Nat) : Action :=
    { kind := .Wait, dst := u32 off, src := 0, offset := 0, size := 0 }
  let asyncGpu (i : Nat) : Action :=
    { kind := .AsyncDispatch, dst := u32 0, src := u32 (gpuStartIndex simdCount i), offset := u32 (gpuFlagOffset i), size := 1 }
  let asyncMem (i : Nat) : Action :=
    { kind := .AsyncDispatch, dst := u32 6, src := u32 (memStartIndex simdCount gpuCount i), offset := u32 (memFlagOffset i), size := 1 }
  let rec loop (i : Nat) : List Action :=
    if _h : i < batches then
      let base := [
        waitFlag (simdFlagOffset i),
        asyncGpu i
      ]
      let nextSimd := if i + 1 < batches then [asyncSimd (i + 1)] else []
      let tail := [
        waitFlag (gpuFlagOffset i),
        asyncMem i,
        waitFlag (memFlagOffset i)
      ]
      base ++ nextSimd ++ tail ++ loop (i + 1)
    else
      []
  let fileFlag := fileFlagOffset
  let fileDispatch : Action :=
    { kind := .AsyncDispatch, dst := u32 2, src := u32 (fileWriteIndex simdCount gpuCount memCount), offset := u32 fileFlag, size := 1 }
  let fileWait : Action :=
    { kind := .Wait, dst := u32 fileFlag, src := 0, offset := 0, size := 0 }
  asyncSimd 0 :: loop 0 ++ [fileDispatch, fileWait]

-- Payload layout

def payloads : List UInt8 :=
  let shaderBytes := padTo (stringToBytes heightmapShader) shaderSize
  let filename := padTo (stringToBytes "heightmap.bmp") filenameSize
  let flags := uint64ToBytes 1 ++ zeros (flagsSize - 8)
  let header := bmpHeader32
  let pixelData := zeros pixelBytes
  let hm0 := zeros batchBytes
  let hm1 := zeros batchBytes
  let out0 := zeros batchBytes
  let out1 := zeros batchBytes
  let xTable := buildXTable 0 imageWidth
  let yTable := buildYTable 0 imageHeight
  let data := shaderBytes ++ filename ++ flags ++ header ++ pixelData ++ hm0 ++ hm1 ++ out0 ++ out1 ++ xTable ++ yTable
  let simdScratch := zeros simdScratchTotal
  let gpuRegion := zeros batchBytes
  data ++ simdScratch ++ gpuRegion


def drawAlgorithm : Algorithm :=
  let simdActions := (List.range batches).foldr (fun i acc => simdActionsForBatch i ++ acc) []
  let gpuActions := List.map gpuActionForBatch (List.range batches)
  let memActions := List.map memActionForBatch (List.range batches)
  let fileAction : Action := { kind := .FileWrite, dst := u32 filenameOffset, src := u32 headerOffset, offset := u32 filenameSize, size := u32 (headerSize + pixelBytes) }
  let workerActions := simdActions ++ gpuActions ++ memActions ++ [fileAction]
  let dispatchStart := 1 + workerActions.length
  let jumpAction : Action := { kind := .ConditionalJump, dst := u32 dispatchStart, src := u32 flagAlways, offset := 0, size := 1 }
  let simdCount := simdActions.length
  let gpuCount := gpuActions.length
  let memCount := memActions.length
  let control := controlActions simdCount gpuCount memCount
  {
    actions := jumpAction :: workerActions ++ control,
    payloads := payloads,
    state := {
      regs_per_unit := 8,
      unit_scratch_offsets := [dataSize, dataSize + simdScratchSize, dataSize + simdScratchSize * 2, dataSize + simdScratchSize * 3],
      unit_scratch_size := simdScratchSize,
      shared_data_offset := 0,
      shared_data_size := dataSize,
      gpu_offset := dataSize + 4096 * 4,
      gpu_size := batchBytes,
      computational_regs := 32,
      file_buffer_size := fileBufferSize,
      gpu_shader_offsets := [0]
    },
    queues := {
      capacity := 256
    },
    units := {
      simd_units := 4,
      gpu_units := 1,
      computational_units := 1,
      file_units := 1,
      network_units := 1,
      memory_units := 1,
      ffi_units := 1,
      backends_bits := 0xFFFFFFFF,
      features_bits := 0
    },
    simd_assignments := [],
    computational_assignments := [],
    memory_assignments := [],
    file_assignments := [],
    network_assignments := [],
    ffi_assignments := [],
    gpu_assignments := [],
    worker_threads := none,
    blocking_threads := none,
    stack_size := none,
    timeout_ms := some 30000,
    thread_name_prefix := some "heightmap"
  }

end Algorithm

def main : IO Unit := do
  let json := toJson Algorithm.drawAlgorithm
  IO.println (Json.compress json)
