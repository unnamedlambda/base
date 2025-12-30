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

def exampleAlgorithm : Algorithm := {
  actions := [
    -- First GPU computation: 7 + 9 = 16
    { kind := .Dispatch, dst := 2576, src := 2576, offset := 2560, size := 64 },
    { kind := .AsyncDispatch, dst := 0, src := 0, offset := 2560, size := 0 },
    { kind := .Wait, dst := 2560, src := 0, offset := 0, size := 0 },
    { kind := .FileWrite, dst := 2048, src := 2588, offset := 256, size := 2 },

    -- Second GPU computation: 3 + 5 = 8
    { kind := .Dispatch, dst := 2640, src := 2640, offset := 2568, size := 64 },
    { kind := .AsyncDispatch, dst := 0, src := 4, offset := 2568, size := 0 },
    { kind := .Wait, dst := 2568, src := 0, offset := 0, size := 0 },
    { kind := .FileWrite, dst := 2304, src := 2652, offset := 256, size := 1 }
  ],
  payloads := gpuDualPayloads addIntShader "output1.txt" "output2.txt" 7 9 3 5,
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
  timeout_ms := some 5000,
  thread_name_prefix := none
}

end Algorithm

def main : IO Unit := do
  let json := toJson Algorithm.exampleAlgorithm
  IO.println (Json.compress json)
