import Lean
import Std

open Lean

namespace AlgorithmLib

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
  | SimdDivI32
  | SimdSubI32
  | MemCopy
  | MemCopyIndirect
  | MemStoreIndirect
  | MemWrite
  | MemScan
  | FileRead
  | FileWrite
  | Compare
  | ConditionalJump
  | Fence
  | AsyncDispatch
  | Wait
  | Dispatch
  | FFICall
  | HashTableCreate
  | HashTableInsert
  | HashTableLookup
  | HashTableDelete
  | LmdbOpen
  | LmdbPut
  | LmdbGet
  | LmdbDelete
  | LmdbCursorScan
  | LmdbSync
  | LmdbBeginWriteTxn
  | LmdbCommitWriteTxn
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
    | .SimdDivI32 => "simd_div_i32"
    | .SimdSubI32 => "simd_sub_i32"
    | .MemCopy => "mem_copy"
    | .MemCopyIndirect => "mem_copy_indirect"
    | .MemStoreIndirect => "mem_store_indirect"
    | .MemWrite => "mem_write"
    | .MemScan => "mem_scan"
    | .FileRead => "file_read"
    | .FileWrite => "file_write"
    | .Compare => "compare"
    | .ConditionalJump => "conditional_jump"
    | .Fence => "fence"
    | .AsyncDispatch => "async_dispatch"
    | .Wait => "wait"
    | .Dispatch => "dispatch"
    | .FFICall => "f_f_i_call"
    | .HashTableCreate => "hash_table_create"
    | .HashTableInsert => "hash_table_insert"
    | .HashTableLookup => "hash_table_lookup"
    | .HashTableDelete => "hash_table_delete"
    | .LmdbOpen => "lmdb_open"
    | .LmdbPut => "lmdb_put"
    | .LmdbGet => "lmdb_get"
    | .LmdbDelete => "lmdb_delete"
    | .LmdbCursorScan => "lmdb_cursor_scan"
    | .LmdbSync => "lmdb_sync"
    | .LmdbBeginWriteTxn => "lmdb_begin_write_txn"
    | .LmdbCommitWriteTxn => "lmdb_commit_write_txn"

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
  gpu_size : Nat
  file_buffer_size : Nat
  gpu_shader_offsets : List Nat
  cranelift_ir_offsets : List Nat
  deriving Repr

instance : ToJson State where
  toJson s := Json.mkObj [
    ("regs_per_unit", toJson s.regs_per_unit),
    ("gpu_size", toJson s.gpu_size),
    ("file_buffer_size", toJson s.file_buffer_size),
    ("gpu_shader_offsets", toJson s.gpu_shader_offsets),
    ("cranelift_ir_offsets", toJson s.cranelift_ir_offsets),
  ]

structure UnitSpec where
  simd_units : Nat
  gpu_units : Nat
  file_units : Nat
  memory_units : Nat
  ffi_units : Nat
  hash_table_units : Nat
  lmdb_units : Nat
  cranelift_units : Nat
  backends_bits : UInt32
  deriving Repr

instance : ToJson UnitSpec where
  toJson u := Json.mkObj [
    ("simd_units", toJson u.simd_units),
    ("gpu_units", toJson u.gpu_units),
    ("file_units", toJson u.file_units),
    ("memory_units", toJson u.memory_units),
    ("ffi_units", toJson u.ffi_units),
    ("hash_table_units", toJson u.hash_table_units),
    ("lmdb_units", toJson u.lmdb_units),
    ("cranelift_units", toJson u.cranelift_units),
    ("backends_bits", toJson u.backends_bits)
  ]

structure Algorithm where
  actions : List Action
  payloads : List UInt8
  state : State
  units : UnitSpec
  simd_assignments : List UInt8
  memory_assignments : List UInt8
  file_assignments : List UInt8
  ffi_assignments : List UInt8
  hash_table_assignments : List UInt8
  lmdb_assignments : List UInt8
  gpu_assignments : List UInt8
  cranelift_assignments : List UInt8
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
    ("units", toJson alg.units),
    ("simd_assignments", toJson alg.simd_assignments),
    ("memory_assignments", toJson alg.memory_assignments),
    ("file_assignments", toJson alg.file_assignments),
    ("ffi_assignments", toJson alg.ffi_assignments),
    ("hash_table_assignments", toJson alg.hash_table_assignments),
    ("lmdb_assignments", toJson alg.lmdb_assignments),
    ("gpu_assignments", toJson alg.gpu_assignments),
    ("cranelift_assignments", toJson alg.cranelift_assignments),
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

end AlgorithmLib
