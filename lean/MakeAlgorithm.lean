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
  batch_size : Nat
  deriving Repr

instance : ToJson QueueSpec where
  toJson q := Json.mkObj [
    ("capacity", toJson q.capacity),
    ("batch_size", toJson q.batch_size)
  ]

structure UnitSpec where
  simd_units : Nat
  gpu_enabled : Bool
  computational_enabled : Bool
  file_units : Nat
  backends_bits : UInt32
  features_bits : UInt64
  deriving Repr

instance : ToJson UnitSpec where
  toJson u := Json.mkObj [
    ("simd_units", toJson u.simd_units),
    ("gpu_enabled", toJson u.gpu_enabled),
    ("computational_enabled", toJson u.computational_enabled),
    ("file_units", toJson u.file_units),
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

  -- 2728-2735: condition1 (1.0 = true, will take path A)
  let condition1 := f64ToBytes 1.0

  -- 2736-2743: condition2 (0.0 = false, will take path B)
  let condition2 := f64ToBytes 0.0

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

def exampleAlgorithm : Algorithm := {
  actions := [
    -- Action 0-3: First GPU computation: 7 + 9 = 16
    { kind := .Dispatch, dst := 2576, src := 2576, offset := 2560, size := 64 },
    { kind := .AsyncDispatch, dst := 0, src := 0, offset := 2560, size := 0 },
    { kind := .Wait, dst := 2560, src := 0, offset := 0, size := 0 },
    { kind := .FileWrite, dst := 2048, src := 2588, offset := 256, size := 2 },

    -- Action 4-7: Second GPU computation: 3 + 5 = 8
    { kind := .Dispatch, dst := 2640, src := 2640, offset := 2568, size := 64 },
    { kind := .AsyncDispatch, dst := 0, src := 4, offset := 2568, size := 0 },
    { kind := .Wait, dst := 2568, src := 0, offset := 0, size := 0 },
    { kind := .FileWrite, dst := 2304, src := 2652, offset := 256, size := 1 },

    -- Action 8: Simple test - write a marker file to prove we reach here
    { kind := .FileWrite, dst := 2824, src := 2992, offset := 56, size := 11 },

    -- TEST 1: Condition = 1.0 (true) - should take path A
    -- Action 9: ConditionalJump with condition1 (1.0 != 0 → jump to action 11)
    { kind := .ConditionalJump, src := 2728, dst := 11, offset := 0, size := 0 },

    -- Action 10: Path B for test 1 (SKIPPED because jump happened)
    { kind := .FileWrite, dst := 2880, src := 3048, offset := 56, size := 11 },

    -- TEST 2: Condition = 0.0 (false) - should take path B
    -- Action 11: ConditionalJump with condition2 (0.0 == 0 → fall through to action 12)
    { kind := .ConditionalJump, src := 2736, dst := 14, offset := 0, size := 0 },

    -- Action 12: Path B for test 2 (EXECUTED - fell through from action 11)
    { kind := .FileWrite, dst := 2880, src := 3048, offset := 56, size := 11 },

    -- Action 13: Skip past the path A write (unconditional jump - reads from shader which is non-zero)
    { kind := .ConditionalJump, src := 0, dst := 15, offset := 0, size := 0 },

    -- Action 14: Path A for test 2 (SKIPPED - never reached)
    { kind := .FileWrite, dst := 2824, src := 2992, offset := 56, size := 11 },

    -- Action 15-18: Final GPU computation (double: 16+16=32)
    { kind := .Dispatch, dst := 2744, src := 2744, offset := 2808, size := 64 },
    { kind := .AsyncDispatch, dst := 0, src := 15, offset := 2808, size := 0 },
    { kind := .Wait, dst := 2808, src := 0, offset := 0, size := 0 },
    { kind := .FileWrite, dst := 2936, src := 2756, offset := 56, size := 2 }
  ],
  payloads := complexPayloads addIntShader,
  state := {
    regs_per_unit := 16,
    unit_scratch_offsets := [4096, 8192, 12288, 16384],
    unit_scratch_size := 4096,
    shared_data_offset := 20480,
    shared_data_size := 16384,
    gpu_offset := 2312,
    gpu_size := 128,
    computational_regs := 32,
    file_buffer_size := 65536,
    gpu_shader_offsets := [0]
  },
  queues := {
    capacity := 256,
    batch_size := 1
  },
  units := {
    simd_units := 4,
    gpu_enabled := true,
    computational_enabled := true,
    file_units := 2,
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
