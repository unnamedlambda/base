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
  | SimdLoad | SimdAdd | SimdMul | SimdStore
  | MemCopy | FileRead | FileWrite
  | Approximate | Choose | Compare
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
    ("file_buffer_size", toJson s.file_buffer_size)
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
    ("worker_threads", toJson alg.worker_threads),
    ("blocking_threads", toJson alg.blocking_threads),
    ("stack_size", toJson alg.stack_size),
    ("timeout_ms", toJson alg.timeout_ms),
    ("thread_name_prefix", toJson alg.thread_name_prefix)
  ]

def exampleAlgorithm : Algorithm := {
  actions := [
    { kind := .SimdLoad, dst := 0, src := 0, offset := 0, size := 16 },
    { kind := .SimdLoad, dst := 1, src := 0, offset := 16, size := 16 },
    { kind := .SimdAdd, dst := 2, src := 0, offset := 1, size := 0 },
    { kind := .SimdStore, dst := 0, src := 2, offset := 32, size := 16 }
  ],
  payloads := List.replicate 65536 0,
  state := {
    regs_per_unit := 16,
    unit_scratch_offsets := [0, 4096, 8192, 12288],
    unit_scratch_size := 4096,
    shared_data_offset := 16384,
    shared_data_size := 16384,
    gpu_offset := 32768,
    gpu_size := 32768,
    computational_regs := 32,
    file_buffer_size := 65536
  },
  queues := {
    capacity := 256,
    batch_size := 16
  },
  units := {
    simd_units := 4,
    gpu_enabled := false,
    computational_enabled := true,
    file_units := 2,
    backends_bits := 0xFFFFFFFF,
    features_bits := 0
  },
  simd_assignments := [0, 0, 0, 0],
  computational_assignments := [],
  memory_assignments := [],
  file_assignments := [],
  network_assignments := [],
  ffi_assignments := [],
  worker_threads := none,
  blocking_threads := none,
  stack_size := none,
  timeout_ms := none,
  thread_name_prefix := none
}

end Algorithm

def main : IO Unit := do
  let json := toJson Algorithm.exampleAlgorithm
  IO.println (Json.compress json)
