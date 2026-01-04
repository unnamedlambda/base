import Lean

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
  | MemCopy
  | FileRead
  | FileWrite
  | Approximate
  | Choose
  | Compare
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
    | .MemCopy => "mem_copy"
    | .FileRead => "file_read"
    | .FileWrite => "file_write"
    | .Approximate => "approximate"
    | .Choose => "choose"
    | .Compare => "compare"
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

def addIntShader : String :=
  "@group(0) @binding(0)\n" ++
  "var<storage, read_write> data: array<u32>;\n\n" ++
  "@compute @workgroup_size(1)\n" ++
  "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n" ++
  "    if (id.x == 0u) {\n" ++
  "        let num1 = data[0];\n" ++
  "        let num2 = data[1];\n" ++
  "        let result = num1 + num2;\n" ++
  "        \n" ++
  "        var temp = result;\n" ++
  "        var digits = 0u;\n" ++
  "        var digit_values: array<u32, 10>;\n" ++
  "        \n" ++
  "        if (temp == 0u) {\n" ++
  "            digit_values[0] = 0u;\n" ++
  "            digits = 1u;\n" ++
  "        } else {\n" ++
  "            while (temp > 0u) {\n" ++
  "                digit_values[digits] = temp % 10u;\n" ++
  "                temp = temp / 10u;\n" ++
  "                digits = digits + 1u;\n" ++
  "            }\n" ++
  "        }\n" ++
  "        \n" ++
  "        var packed_idx = 3u;\n" ++
  "        for (var i = 0u; i < digits; i = i + 1u) {\n" ++
  "            let ascii_val = 48u + digit_values[digits - 1u - i];\n" ++
  "            let byte_pos = i % 4u;\n" ++
  "            \n" ++
  "            if (byte_pos == 0u) {\n" ++
  "                data[packed_idx] = ascii_val;\n" ++
  "            } else {\n" ++
  "                data[packed_idx] = data[packed_idx] | (ascii_val << (byte_pos * 8u));\n" ++
  "            }\n" ++
  "            \n" ++
  "            if (byte_pos == 3u && i < digits - 1u) {\n" ++
  "                packed_idx = packed_idx + 1u;\n" ++
  "            }\n" ++
  "        }\n" ++
  "        \n" ++
  "        data[2] = digits;\n" ++
  "    }\n" ++
  "}\n"

-- Gradient shader: generates 512x512 BMP gradient image
-- BMP format: BGR order, bottom-to-top rows, rows padded to 4 bytes
-- 512 pixels * 3 bytes = 1536 bytes per row (already 4-byte aligned)
-- Each thread writes one u32 (4 bytes) of output
def gradientShader : String :=
  "@group(0) @binding(0)\n" ++
  "var<storage, read_write> data: array<u32>;\n\n" ++
  "@compute @workgroup_size(64)\n" ++
  "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n" ++
  "    let width = 512u;\n" ++
  "    let height = 512u;\n" ++
  "    let row_size = width * 3u;  // 1536, already 4-byte aligned\n" ++
  "    let total_bytes = row_size * height;\n" ++
  "    let total_words = (total_bytes + 3u) / 4u;\n" ++
  "    let word_idx = id.x;\n" ++
  "    \n" ++
  "    if (word_idx >= total_words) { return; }\n" ++
  "    \n" ++
  "    // Compute 4 bytes for this u32\n" ++
  "    var result: u32 = 0u;\n" ++
  "    \n" ++
  "    for (var i = 0u; i < 4u; i = i + 1u) {\n" ++
  "        let byte_idx = word_idx * 4u + i;\n" ++
  "        if (byte_idx >= total_bytes) { break; }\n" ++
  "        \n" ++
  "        // Which row and column\n" ++
  "        let row = byte_idx / row_size;\n" ++
  "        let col_byte = byte_idx % row_size;\n" ++
  "        let x = col_byte / 3u;\n" ++
  "        // BMP is bottom-to-top, so flip y\n" ++
  "        let y = height - 1u - row;\n" ++
  "        // BMP is BGR order: 0=B, 1=G, 2=R\n" ++
  "        let channel = col_byte % 3u;\n" ++
  "        \n" ++
  "        // Calculate color value for this channel\n" ++
  "        var value: u32;\n" ++
  "        if (channel == 2u) {\n" ++
  "            // Red: gradient left to right\n" ++
  "            value = (x * 255u) / (width - 1u);\n" ++
  "        } else if (channel == 1u) {\n" ++
  "            // Green: gradient top to bottom\n" ++
  "            value = (y * 255u) / (height - 1u);\n" ++
  "        } else {\n" ++
  "            // Blue: diagonal gradient\n" ++
  "            value = ((x + y) * 128u) / (width + height - 2u);\n" ++
  "        }\n" ++
  "        \n" ++
  "        result = result | (value << (i * 8u));\n" ++
  "    }\n" ++
  "    \n" ++
  "    data[word_idx] = result;\n" ++
  "}\n"

def gpuDualPayloads (shader : String) (file1 : String) (file2 : String) (a1 : UInt32) (b1 : UInt32) (a2 : UInt32) (b2 : UInt32) : List UInt8 :=
  let shaderBytes := padTo (stringToBytes shader) 2048
  let file1Bytes := padTo (stringToBytes file1) 256
  let file2Bytes := padTo (stringToBytes file2) 256
  let flag1Bytes := zeros 8
  let flag2Bytes := zeros 8
  let input1Bytes := uint32ToBytes a1 ++ uint32ToBytes b1
  let workspace1 := zeros 56
  let input2Bytes := uint32ToBytes a2 ++ uint32ToBytes b2
  let workspace2 := zeros 56
  shaderBytes ++ file1Bytes ++ file2Bytes ++ flag1Bytes ++ flag2Bytes ++ input1Bytes ++ workspace1 ++ input2Bytes ++ workspace2

def f64ToBytes (f : Float) : List UInt8 :=
  let bits := f.toBits
  let b0 := UInt8.ofNat (bits.toNat &&& 0xFF)
  let b1 := UInt8.ofNat ((bits.toNat >>> 8) &&& 0xFF)
  let b2 := UInt8.ofNat ((bits.toNat >>> 16) &&& 0xFF)
  let b3 := UInt8.ofNat ((bits.toNat >>> 24) &&& 0xFF)
  let b4 := UInt8.ofNat ((bits.toNat >>> 32) &&& 0xFF)
  let b5 := UInt8.ofNat ((bits.toNat >>> 40) &&& 0xFF)
  let b6 := UInt8.ofNat ((bits.toNat >>> 48) &&& 0xFF)
  let b7 := UInt8.ofNat ((bits.toNat >>> 56) &&& 0xFF)
  [b0, b1, b2, b3, b4, b5, b6, b7]

def complexPayloads (shader : String) : List UInt8 :=
  -- 0-2047: GPU shader
  let shaderBytes := padTo (stringToBytes shader) 2048

  -- 2048-2303: filename "result1.txt"
  let file1 := padTo (stringToBytes "result1.txt") 256

  -- 2304-2559: filename "result2.txt"
  let file2 := padTo (stringToBytes "result2.txt") 256

  -- 2560-2567: completion flag 1
  let flag1 := zeros 8

  -- 2568-2575: completion flag 2
  let flag2 := zeros 8

  -- 2576-2639: gpu_data_1 (7,9 + workspace)
  let gpuData1 := uint32ToBytes 7 ++ uint32ToBytes 9 ++ zeros 56

  -- 2640-2703: gpu_data_2 (3,5 + workspace)
  let gpuData2 := uint32ToBytes 3 ++ uint32ToBytes 5 ++ zeros 56

  -- 2704-2711: comparison flag
  let compFlag := zeros 8

  -- 2712-2719: compare_area_a (will hold 16 as f64)
  let compareA := zeros 8

  -- 2720-2727: compare_area_b (will hold 8 as f64)
  let compareB := zeros 8

  -- 2728-2735: condition1 (1 = true, will take path A)
  let condition1 := uint64ToBytes 1

  -- 2736-2743: condition2 (0 = false, will take path B)
  let condition2 := uint64ToBytes 0

  -- 2744-2807: gpu_data_3 for doubling (16, workspace)
  let gpuData3 := uint32ToBytes 16 ++ uint32ToBytes 16 ++ zeros 56

  -- 2808-2815: completion flag 3
  let flag3 := zeros 8

  -- 2816-2823: completion flag 4
  let flag4 := zeros 8

  -- 2824-2879: filename "path_a.txt"
  let filePathA := padTo (stringToBytes "path_a.txt") 56

  -- 2880-2935: filename "path_b.txt"
  let filePathB := padTo (stringToBytes "path_b.txt") 56

  -- 2936-2991: filename "doubled.txt"
  let fileDoubled := padTo (stringToBytes "doubled.txt") 56

  -- 2992-3047: text "TOOK PATH A"
  let textA := padTo (stringToBytes "TOOK PATH A") 56

  -- 3048-3103: text "TOOK PATH B"
  let textB := padTo (stringToBytes "TOOK PATH B") 56

  shaderBytes ++ file1 ++ file2 ++ flag1 ++ flag2 ++ gpuData1 ++ gpuData2 ++
  compFlag ++ compareA ++ compareB ++ condition1 ++ condition2 ++ gpuData3 ++ flag3 ++ flag4 ++
  filePathA ++ filePathB ++ fileDoubled ++ textA ++ textB

-- BMP file header (14 bytes) + DIB header (40 bytes) = 54 bytes total
-- For 512x512 24-bit image: file size = 54 + 512*512*3 = 786486 bytes
def bmpHeader : List UInt8 :=
  let width : Nat := 512
  let height : Nat := 512
  let rowSize : Nat := width * 3  -- 1536, already 4-byte aligned
  let pixelDataSize : Nat := rowSize * height  -- 786432
  let fileSize : Nat := 54 + pixelDataSize  -- 786486

  -- BMP File Header (14 bytes)
  let bfType := [0x42, 0x4D]  -- "BM"
  let bfSize := uint32ToBytes (UInt32.ofNat fileSize)
  let bfReserved := [0, 0, 0, 0]
  let bfOffBits := uint32ToBytes 54  -- pixel data starts at offset 54

  -- DIB Header - BITMAPINFOHEADER (40 bytes)
  let biSize := uint32ToBytes 40
  let biWidth := uint32ToBytes (UInt32.ofNat width)
  let biHeight := uint32ToBytes (UInt32.ofNat height)  -- positive = bottom-up
  let biPlanes := [1, 0]  -- always 1
  let biBitCount := [24, 0]  -- 24 bits per pixel
  let biCompression := uint32ToBytes 0  -- BI_RGB (uncompressed)
  let biSizeImage := uint32ToBytes (UInt32.ofNat pixelDataSize)
  let biXPelsPerMeter := uint32ToBytes 2835  -- 72 DPI
  let biYPelsPerMeter := uint32ToBytes 2835
  let biClrUsed := uint32ToBytes 0
  let biClrImportant := uint32ToBytes 0

  bfType ++ bfSize ++ bfReserved ++ bfOffBits ++
  biSize ++ biWidth ++ biHeight ++ biPlanes ++ biBitCount ++
  biCompression ++ biSizeImage ++ biXPelsPerMeter ++ biYPelsPerMeter ++
  biClrUsed ++ biClrImportant

-- Gradient image payloads for 512x512 BMP output
def gradientPayloads (shader : String) : List UInt8 :=
  -- 0-4095: GPU shader (4KB)
  let shaderBytes := padTo (stringToBytes shader) 4096

  -- 4096-4351: filename "gradient.bmp" (256 bytes)
  let filename := padTo (stringToBytes "gradient.bmp") 256

  -- 4352-4359: completion flag (8 bytes)
  let flag := zeros 8

  -- 4360-4361: 2 bytes padding
  let padding := zeros 2

  -- 4362-4415: BMP header (54 bytes) - positioned right before pixels
  let header := bmpHeader  -- exactly 54 bytes

  -- 4416+: pixel data area (512*512*3 = 786432 bytes)
  -- GPU will write here, we just reserve space
  let pixelData := zeros 786432

  shaderBytes ++ filename ++ flag ++ padding ++ header ++ pixelData

def exampleAlgorithm : Algorithm := {
  actions := [
    -- Generate 512x512 gradient BMP image using GPU
    -- Layout: shader(4096) + filename(256) + flag(8) + padding(2) + header(54) + pixels(786432)
    -- GPU writes pixel data at offset 4416
    -- BMP header is at 4362-4415 (54 bytes) - already positioned before pixels
    -- File output: header(54) + pixels(786432) = 786486 bytes

    -- Action 0: Dispatch GPU to generate gradient pixels
    -- size = pixel data bytes (786432)
    { kind := .Dispatch, dst := 4416, src := 4416, offset := 4352, size := 786432 },

    -- Action 1: Queue to GPU unit
    { kind := .AsyncDispatch, dst := 0, src := 0, offset := 4352, size := 0 },

    -- Action 2: Wait for GPU completion
    { kind := .Wait, dst := 4352, src := 0, offset := 0, size := 0 },

    -- Action 3: FileWrite the complete BMP (header + pixels)
    -- filename at 4096, data starts at 4362, size = 54 + 786432 = 786486
    { kind := .FileWrite, dst := 4096, src := 4362, offset := 256, size := 786486 }
  ],
  payloads := gradientPayloads gradientShader,
  state := {
    regs_per_unit := 16,
    unit_scratch_offsets := [900000, 904096, 908192, 912288],
    unit_scratch_size := 4096,
    shared_data_offset := 920000,
    shared_data_size := 16384,
    gpu_offset := 4416,
    gpu_size := 786432,
    computational_regs := 32,
    file_buffer_size := 800000,
    gpu_shader_offsets := [0]
  },
  queues := {
    capacity := 256
  },
  units := {
    simd_units := 4,
    gpu_units := 1,
    computational_units := 1,
    file_units := 2,
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
  thread_name_prefix := none
}

end Algorithm

def main : IO Unit := do
  let json := toJson Algorithm.exampleAlgorithm
  IO.println (Json.compress json)
