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
  | FFICall
  | MemScan
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
    | .FFICall => "f_f_i_call"
    | .MemScan => "mem_scan"

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

-- Pattern to skip at file start
def EVAL_PATTERN : String := "#eval "
def EVAL_PATTERN_LEN : Nat := EVAL_PATTERN.length

-- Buffer sizes
def FN_PTR_SIZE : Nat := 8
def STDOUT_PATH_SIZE : Nat := 16
def PADDING_1_SIZE : Nat := 8
def ARGV_PARAMS_SIZE : Nat := 8
def FILENAME_BUF_SIZE : Nat := 256
def FFI_RESULT_SIZE : Nat := 8
def PADDING_2_SIZE : Nat := 176
def FLAGS_SIZE : Nat := 32       -- 4 completion flags (8 bytes each)
def SOURCE_BUF_SIZE : Nat := 512
def OUTPUT_BUF_SIZE : Nat := 256

-- Memory addresses (aligned for main.rs patching at 0x0010)
def FFI_GET_ARGV : UInt32 := 0
def OUTPUT_PATH : UInt32 := 0x0010  -- main.rs patches output path here
def ARGV_PARAMS : UInt32 := (OUTPUT_PATH.toNat + STDOUT_PATH_SIZE + PADDING_1_SIZE).toUInt32
def FILENAME_BUF : UInt32 := (ARGV_PARAMS.toNat + ARGV_PARAMS_SIZE).toUInt32
def FFI_RESULT : UInt32 := (FILENAME_BUF.toNat + FILENAME_BUF_SIZE).toUInt32
-- Completion flags after FFI_RESULT
def FLAG_FFI : UInt32 := (FFI_RESULT.toNat + FFI_RESULT_SIZE + PADDING_2_SIZE).toUInt32
def FLAG_FILEREAD : UInt32 := (FLAG_FFI.toNat + 8).toUInt32
def FLAG_MEMCOPY : UInt32 := (FLAG_FILEREAD.toNat + 8).toUInt32
def FLAG_FILEWRITE : UInt32 := (FLAG_MEMCOPY.toNat + 8).toUInt32
def SOURCE_BUF : UInt32 := (FLAG_FILEWRITE.toNat + 8).toUInt32
def OUTPUT_BUF : UInt32 := (SOURCE_BUF.toNat + SOURCE_BUF_SIZE).toUInt32

def leanEvalPayloads : List UInt8 :=
  let fnPtrPlaceholder := zeros FN_PTR_SIZE
  let padding0 := zeros 8  -- Padding to align OUTPUT_PATH at 0x0010
  let outputPath := padTo (stringToBytes "/dev/stdout") STDOUT_PATH_SIZE
  let padding1 := zeros PADDING_1_SIZE
  let argvParams := uint32ToBytes 1 ++ uint32ToBytes (FILENAME_BUF_SIZE - 8).toUInt32
  let filenameBuf := zeros FILENAME_BUF_SIZE
  let ffiResult := zeros FFI_RESULT_SIZE
  let padding2 := zeros PADDING_2_SIZE
  let flags := zeros FLAGS_SIZE
  let sourceBuf := zeros SOURCE_BUF_SIZE
  let outputBuf := zeros OUTPUT_BUF_SIZE
  fnPtrPlaceholder ++ padding0 ++ outputPath ++ padding1 ++ argvParams ++
    filenameBuf ++ ffiResult ++ padding2 ++ flags ++ sourceBuf ++ outputBuf

def leanEvalAlgorithm : Algorithm := {
  actions := [
    -- Action 0: FFICall - Get filename from argv[1]
    { kind := .FFICall, src := FFI_GET_ARGV, dst := ARGV_PARAMS, offset := FFI_RESULT, size := 0 },
    -- Action 1: FileRead - Read source file
    { kind := .FileRead, src := FILENAME_BUF, dst := SOURCE_BUF,
      offset := FILENAME_BUF_SIZE.toUInt32, size := SOURCE_BUF_SIZE.toUInt32 },
    -- Action 2: MemCopy - Copy bytes after "#eval " to output
    { kind := .MemCopy, src := SOURCE_BUF + EVAL_PATTERN_LEN.toUInt32, dst := OUTPUT_BUF,
      offset := 0, size := 32 },
    -- Action 3: FileWrite - Write output (size=0 means null-terminated)
    { kind := .FileWrite, dst := OUTPUT_PATH, src := OUTPUT_BUF,
      offset := STDOUT_PATH_SIZE.toUInt32, size := 0 },
    -- Action 4: AsyncDispatch FFICall to FFI unit (type 4)
    { kind := .AsyncDispatch, dst := 4, src := 0, offset := FLAG_FFI, size := 0 },
    -- Action 5: Wait for FFICall
    { kind := .Wait, dst := FLAG_FFI, src := 0, offset := 0, size := 0 },
    -- Action 6: AsyncDispatch FileRead to File unit (type 2)
    { kind := .AsyncDispatch, dst := 2, src := 1, offset := FLAG_FILEREAD, size := 0 },
    -- Action 7: Wait for FileRead
    { kind := .Wait, dst := FLAG_FILEREAD, src := 0, offset := 0, size := 0 },
    -- Action 8: AsyncDispatch MemCopy to Memory unit (type 6)
    { kind := .AsyncDispatch, dst := 6, src := 2, offset := FLAG_MEMCOPY, size := 0 },
    -- Action 9: Wait for MemCopy
    { kind := .Wait, dst := FLAG_MEMCOPY, src := 0, offset := 0, size := 0 },
    -- Action 10: AsyncDispatch FileWrite to File unit (type 2)
    { kind := .AsyncDispatch, dst := 2, src := 3, offset := FLAG_FILEWRITE, size := 0 },
    -- Action 11: Wait for FileWrite
    { kind := .Wait, dst := FLAG_FILEWRITE, src := 0, offset := 0, size := 0 }
  ],
  payloads := leanEvalPayloads,
  state := {
    regs_per_unit := 32,
    unit_scratch_offsets := [],
    unit_scratch_size := 0,
    shared_data_offset := 0x100000,
    shared_data_size := 0x100000,
    gpu_offset := 0,
    gpu_size := 0,
    computational_regs := 32,
    file_buffer_size := 0x100000,
    gpu_shader_offsets := []
  },
  queues := { capacity := 64 },
  units := {
    simd_units := 0,
    gpu_units := 0,
    computational_units := 0,
    file_units := 1,
    network_units := 0,
    memory_units := 1,
    ffi_units := 1,
    backends_bits := 0,
    features_bits := 0
  },
  simd_assignments := [],
  computational_assignments := [],
  memory_assignments := [],
  file_assignments := [],
  network_assignments := [],
  ffi_assignments := [],
  gpu_assignments := [],
  worker_threads := some 2,
  blocking_threads := some 2,
  stack_size := none,
  timeout_ms := some 5000,
  thread_name_prefix := some "lean-eval"
}

end Algorithm

def main : IO Unit := do
  let json := toJson Algorithm.leanEvalAlgorithm
  IO.println (Json.compress json)
