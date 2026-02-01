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
  | SimdLoadI32
  | SimdAddI32
  | SimdMulI32
  | SimdStoreI32
  | SimdDivI32
  | MemCopyIndirect
  | SimdSubI32
  | MemStoreIndirect
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
    | .SimdLoadI32 => "simd_load_i32"
    | .SimdAddI32 => "simd_add_i32"
    | .SimdMulI32 => "simd_mul_i32"
    | .SimdStoreI32 => "simd_store_i32"
    | .SimdDivI32 => "simd_div_i32"
    | .MemCopyIndirect => "mem_copy_indirect"
    | .SimdSubI32 => "simd_sub_i32"
    | .MemStoreIndirect => "mem_store_indirect"

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

def int32ToBytes16 (n : Int) : List UInt8 :=
  let n32 := if n < 0 then (0x100000000 + n).toNat else n.toNat
  let b0 := UInt8.ofNat (n32 &&& 0xFF)
  let b1 := UInt8.ofNat ((n32 >>> 8) &&& 0xFF)
  let b2 := UInt8.ofNat ((n32 >>> 16) &&& 0xFF)
  let b3 := UInt8.ofNat ((n32 >>> 24) &&& 0xFF)
  [b0, b1, b2, b3] ++ zeros 12

-- Memory Layout (16-byte aligned for SIMD)
-- 0x0000 (16):  FFI_PTR
-- 0x0010 (16):  OUTPUT_PATH "/tmp/output.txt\0"
-- 0x0020 (16):  ARGV_PARAMS [index=1, maxlen=248]
-- 0x0030 (256): FILENAME_BUF
-- 0x0130 (16):  FFI_RESULT
-- 0x0140 (16):  FLAG_FFI
-- 0x0150 (16):  FLAG_FILE
-- 0x0160 (16):  FLAG_SIMD
-- 0x0170 (16):  FLAG_MEM
-- 0x0180 (16):  CONST_ZERO [0]
-- 0x0190 (512): SOURCE_BUF
-- 0x0390 (16):  CONST_48 [48] (ASCII '0')
-- 0x03A0 (16):  CONST_10 [10]
-- 0x03B0 (16):  CONST_ONE [1]
-- 0x03C0 (16):  CONST_SPACE [32] (ASCII ' ')
-- 0x03D0 (16):  CONST_PLUS [43] (ASCII '+')
-- 0x03E0 (16):  POS [current position, starts at 6]
-- 0x03F0 (16):  CHAR_BUF [current char]
-- 0x0400 (16):  ACCUM [accumulator for number parsing]
-- 0x0410 (16):  LEFT_VAL [first operand]
-- 0x0420 (16):  RIGHT_VAL [second operand]
-- 0x0430 (16):  RESULT [computation result]
-- 0x0440 (16):  DIGIT_COUNT [number of digits in output]
-- 0x0450 (16):  OUTPUT_POS [current output position, counts down from 15]
-- 0x0460 (32):  OUTPUT_BUF [output string buffer, up to 32 chars]

def FFI_PTR : UInt32 := 0x0000
def OUTPUT_PATH : UInt32 := 0x0010
def ARGV_PARAMS : UInt32 := 0x0020
def FILENAME_IN_ARGV : UInt32 := 0x0028
def FILENAME_BUF : UInt32 := 0x0030
def FILENAME_BUF_SIZE : Nat := 256
def FFI_RESULT : UInt32 := 0x0130
def FLAG_FFI : UInt32 := 0x0140
def FLAG_FILE : UInt32 := 0x0150
def FLAG_SIMD : UInt32 := 0x0160
def FLAG_MEM : UInt32 := 0x0170
def CONST_ZERO : UInt32 := 0x0180
def SOURCE_BUF : UInt32 := 0x0190
def SOURCE_BUF_SIZE : Nat := 512
def CONST_48 : UInt32 := 0x0390
def CONST_10 : UInt32 := 0x03A0
def CONST_ONE : UInt32 := 0x03B0
def CONST_SPACE : UInt32 := 0x03C0
def CONST_PLUS : UInt32 := 0x03D0
def POS : UInt32 := 0x03E0
def CHAR_BUF : UInt32 := 0x03F0
def ACCUM : UInt32 := 0x0400
def LEFT_VAL : UInt32 := 0x0410
def RIGHT_VAL : UInt32 := 0x0420
def RESULT : UInt32 := 0x0430
def DIGIT_COUNT : UInt32 := 0x0440
def OUTPUT_POS : UInt32 := 0x0450
def OUTPUT_BUF : UInt32 := 0x0460

def leanEvalPayloads : List UInt8 :=
  let ffiPtr := zeros 16
  let outputPath := padTo (stringToBytes "/tmp/output.txt") 16
  let argvParams := uint32ToBytes 1 ++ uint32ToBytes (FILENAME_BUF_SIZE - 8).toUInt32 ++ zeros 8
  let filenameBuf := zeros FILENAME_BUF_SIZE
  let ffiResult := zeros 16
  let flagFfi := zeros 16
  let flagFile := zeros 16
  let flagSimd := zeros 16
  let flagMem := zeros 16
  let constZero := int32ToBytes16 0
  let sourceBuf := zeros SOURCE_BUF_SIZE
  let const48 := int32ToBytes16 48
  let const10 := int32ToBytes16 10
  let constOne := int32ToBytes16 1
  let constSpace := int32ToBytes16 32
  let constPlus := int32ToBytes16 43
  let pos := int32ToBytes16 6  -- Start after "#eval "
  let charBuf := zeros 16
  let accum := zeros 16
  let leftVal := zeros 16
  let rightVal := zeros 16
  let result := zeros 16
  let digitCount := zeros 16
  let outputPos := int32ToBytes16 30  -- Start at end of buffer for right-to-left
  let outputBuf := zeros 32

  ffiPtr ++ outputPath ++ argvParams ++ filenameBuf ++ ffiResult ++
    flagFfi ++ flagFile ++ flagSimd ++ flagMem ++ constZero ++ sourceBuf ++
    const48 ++ const10 ++ constOne ++ constSpace ++ constPlus ++
    pos ++ charBuf ++ accum ++ leftVal ++ rightVal ++ result ++
    digitCount ++ outputPos ++ outputBuf

def fence : Action := { kind := .Fence, dst := 0, src := 0, offset := 0, size := 0 }


def workActions : List Action := [
  { kind := .FFICall, src := FFI_PTR, dst := ARGV_PARAMS, offset := FFI_RESULT, size := 0 },
  { kind := .FileRead, src := FILENAME_IN_ARGV, dst := SOURCE_BUF,
    offset := FILENAME_BUF_SIZE.toUInt32, size := SOURCE_BUF_SIZE.toUInt32 },
  { kind := .MemCopyIndirect, src := POS, dst := CHAR_BUF, offset := SOURCE_BUF, size := 1 },
  { kind := .SimdLoadI32, src := CHAR_BUF, dst := 0, offset := 0, size := 0 },
  { kind := .SimdLoadI32, src := CONST_SPACE, dst := 1, offset := 0, size := 0 },
  { kind := .SimdSubI32, src := 0, dst := 2, offset := 1, size := 0 },
  { kind := .SimdStoreI32, src := 2, dst := 0, offset := DIGIT_COUNT, size := 0 },
  { kind := .SimdLoadI32, src := CONST_48, dst := 3, offset := 0, size := 0 },
  { kind := .SimdSubI32, src := 0, dst := 4, offset := 3, size := 0 },
  { kind := .SimdLoadI32, src := CONST_10, dst := 5, offset := 0, size := 0 },
  { kind := .SimdLoadI32, src := ACCUM, dst := 6, offset := 0, size := 0 },
  { kind := .SimdMulI32, src := 6, dst := 7, offset := 5, size := 0 },
  { kind := .SimdAddI32, src := 7, dst := 8, offset := 4, size := 0 },
  { kind := .SimdStoreI32, src := 8, dst := 0, offset := ACCUM, size := 0 },
  { kind := .SimdLoadI32, src := POS, dst := 9, offset := 0, size := 0 },
  { kind := .SimdLoadI32, src := CONST_ONE, dst := 10, offset := 0, size := 0 },
  { kind := .SimdAddI32, src := 9, dst := 11, offset := 10, size := 0 },
  { kind := .SimdStoreI32, src := 11, dst := 0, offset := POS, size := 0 },
  { kind := .SimdLoadI32, src := ACCUM, dst := 12, offset := 0, size := 0 },
  { kind := .SimdStoreI32, src := 12, dst := 0, offset := LEFT_VAL, size := 0 },
  { kind := .MemWrite, dst := ACCUM, src := 0, offset := 0, size := 4 },
  { kind := .SimdLoadI32, src := CONST_PLUS, dst := 13, offset := 0, size := 0 },
  { kind := .SimdSubI32, src := 0, dst := 14, offset := 13, size := 0 },
  { kind := .SimdStoreI32, src := 14, dst := 0, offset := DIGIT_COUNT, size := 0 },
  { kind := .SimdStoreI32, src := 0, dst := 0, offset := OUTPUT_POS, size := 0 },
  { kind := .SimdLoadI32, src := CONST_10, dst := 15, offset := 0, size := 0 },
  { kind := .SimdSubI32, src := 0, dst := 16, offset := 15, size := 0 },
  { kind := .SimdLoadI32, src := ACCUM, dst := 17, offset := 0, size := 0 },
  { kind := .SimdStoreI32, src := 17, dst := 0, offset := RIGHT_VAL, size := 0 },
  { kind := .SimdLoadI32, src := LEFT_VAL, dst := 18, offset := 0, size := 0 },
  { kind := .SimdLoadI32, src := RIGHT_VAL, dst := 19, offset := 0, size := 0 },
  { kind := .SimdAddI32, src := 18, dst := 20, offset := 19, size := 0 },
  { kind := .SimdStoreI32, src := 20, dst := 0, offset := RESULT, size := 0 },
  { kind := .MemWrite, dst := OUTPUT_POS, src := 30, offset := 0, size := 4 },
  { kind := .MemWrite, dst := OUTPUT_BUF + 31, src := 10, offset := 0, size := 1 },
  { kind := .SimdLoadI32, src := RESULT, dst := 21, offset := 0, size := 0 },
  { kind := .SimdDivI32, src := 21, dst := 22, offset := 5, size := 0 },
  { kind := .SimdMulI32, src := 22, dst := 23, offset := 5, size := 0 },
  { kind := .SimdSubI32, src := 21, dst := 24, offset := 23, size := 0 },
  { kind := .SimdAddI32, src := 24, dst := 25, offset := 3, size := 0 },
  { kind := .SimdStoreI32, src := 25, dst := 0, offset := CHAR_BUF, size := 0 },
  { kind := .MemStoreIndirect, src := CHAR_BUF, dst := OUTPUT_POS, offset := OUTPUT_BUF, size := 1 },
  { kind := .SimdLoadI32, src := OUTPUT_POS, dst := 26, offset := 0, size := 0 },
  { kind := .SimdSubI32, src := 26, dst := 27, offset := 10, size := 0 },
  { kind := .SimdStoreI32, src := 27, dst := 0, offset := OUTPUT_POS, size := 0 },
  { kind := .SimdStoreI32, src := 22, dst := 0, offset := RESULT, size := 0 },
  { kind := .SimdLoadI32, src := OUTPUT_POS, dst := 28, offset := 0, size := 0 },
  { kind := .SimdAddI32, src := 28, dst := 29, offset := 10, size := 0 },
  { kind := .SimdStoreI32, src := 29, dst := 0, offset := DIGIT_COUNT, size := 0 },
  { kind := .MemCopyIndirect, src := DIGIT_COUNT, dst := LEFT_VAL, offset := OUTPUT_BUF, size := 16 },
  { kind := .FileWrite, dst := OUTPUT_PATH, src := LEFT_VAL, offset := 16, size := 0 },
  { kind := .SimdStoreI32, src := 18, dst := 0, offset := RESULT, size := 0 },
  { kind := .SimdStoreI32, src := 0, dst := 0, offset := CHAR_BUF, size := 0 },
  { kind := .SimdLoadI32, src := CONST_10, dst := 30, offset := 0, size := 0 },
  { kind := .SimdSubI32, src := 0, dst := 31, offset := 30, size := 0 },
  { kind := .SimdStoreI32, src := 31, dst := 0, offset := DIGIT_COUNT, size := 0 }
]

def WORK_BASE : UInt32 := 208

def actualMainFlow : List Action :=
  -- ========== SETUP (0-3) ==========
  [{ kind := .AsyncDispatch, dst := 4, src := WORK_BASE + 0, offset := FLAG_FFI, size := 0 },   -- 0
   { kind := .Wait, dst := FLAG_FFI, src := 0, offset := 0, size := 0 },                        -- 1
   { kind := .AsyncDispatch, dst := 2, src := WORK_BASE + 1, offset := FLAG_FILE, size := 0 }, -- 2
   { kind := .Wait, dst := FLAG_FILE, src := 0, offset := 0, size := 0 },                       -- 3

   -- ========== PARSE_NUM1_LOOP (4-55) ==========
   -- Action 4: Load char at SOURCE_BUF[POS]
   { kind := .AsyncDispatch, dst := 6, src := WORK_BASE + 2, offset := FLAG_MEM, size := 0 },  -- 4
   { kind := .Wait, dst := FLAG_MEM, src := 0, offset := 0, size := 0 },                        -- 5
   -- r0 = char
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 3, offset := FLAG_SIMD, size := 0 }, -- 6
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 7
   -- Store r0 (char) -> CHAR_BUF to check for null
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 52, offset := FLAG_SIMD, size := 0 }, -- 8
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 9
   -- if char != 0 (CHAR_BUF != 0), jump to CHECK_NEWLINE (action 14)
   { kind := .ConditionalJump, src := CHAR_BUF, dst := 14, offset := 0, size := 4 },            -- 10
   -- char == 0 (null), jump to AFTER_NUM1 (action 58)
   { kind := .ConditionalJump, src := CONST_ONE, dst := 58, offset := 0, size := 4 },           -- 11
   fence, fence,  -- 12, 13 padding

   -- CHECK_NEWLINE (action 14) - check if char is newline (ASCII 10)
   -- r30 = CONST_10 (value 10, also newline ASCII)
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 53, offset := FLAG_SIMD, size := 0 }, -- 14
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 15
   -- r31 = r0 - r30 (char - 10)
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 54, offset := FLAG_SIMD, size := 0 }, -- 16
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 17
   -- Store r31 -> DIGIT_COUNT (temp)
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 55, offset := FLAG_SIMD, size := 0 }, -- 18
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 19
   -- if char != '\n' (DIGIT_COUNT != 0), jump to CHECK_SPACE (action 24)
   { kind := .ConditionalJump, src := DIGIT_COUNT, dst := 24, offset := 0, size := 4 },         -- 20
   -- char == '\n' (newline), jump to AFTER_NUM1 (action 58)
   { kind := .ConditionalJump, src := CONST_ONE, dst := 58, offset := 0, size := 4 },           -- 21
   fence, fence,  -- 22, 23 padding

   -- CHECK_SPACE (action 24) - check if char is space
   -- r1 = ' '
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 4, offset := FLAG_SIMD, size := 0 }, -- 24
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 25
   -- r2 = char - ' '
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 5, offset := FLAG_SIMD, size := 0 }, -- 26
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 27
   -- Store r2 -> DIGIT_COUNT (temp)
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 6, offset := FLAG_SIMD, size := 0 }, -- 28
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 29
   -- if char != ' ' (DIGIT_COUNT != 0), jump to ACCUMULATE (action 34)
   { kind := .ConditionalJump, src := DIGIT_COUNT, dst := 34, offset := 0, size := 4 },        -- 30
   -- char == ' ', jump to AFTER_NUM1 (action 58)
   { kind := .ConditionalJump, src := CONST_ONE, dst := 58, offset := 0, size := 4 },          -- 31
   fence, fence,  -- 32, 33 padding

   -- ACCUMULATE_NUM1 (action 34)
   -- r3 = '0'
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 7, offset := FLAG_SIMD, size := 0 }, -- 34
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 35
   -- r4 = char - '0'
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 8, offset := FLAG_SIMD, size := 0 }, -- 36
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 37
   -- r5 = 10
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 9, offset := FLAG_SIMD, size := 0 }, -- 38
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 39
   -- r6 = ACCUM
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 10, offset := FLAG_SIMD, size := 0 }, -- 40
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 41
   -- r7 = accum * 10
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 11, offset := FLAG_SIMD, size := 0 }, -- 42
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 43
   -- r8 = accum*10 + digit
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 12, offset := FLAG_SIMD, size := 0 }, -- 44
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 45
   -- Store accum
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 13, offset := FLAG_SIMD, size := 0 }, -- 46
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 47
   -- r9 = POS
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 14, offset := FLAG_SIMD, size := 0 }, -- 48
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 49
   -- r10 = 1
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 15, offset := FLAG_SIMD, size := 0 }, -- 50
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 51
   -- r11 = pos + 1
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 16, offset := FLAG_SIMD, size := 0 }, -- 52
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 53
   -- Store pos
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 17, offset := FLAG_SIMD, size := 0 }, -- 54
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 55
   -- Jump back to PARSE_NUM1_LOOP (action 4)
   { kind := .ConditionalJump, src := CONST_ONE, dst := 4, offset := 0, size := 4 },            -- 56
   fence,  -- 57 padding

   -- ========== AFTER_NUM1 (action 58) ==========
   -- Save ACCUM to LEFT_VAL
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 18, offset := FLAG_SIMD, size := 0 }, -- 58
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 59
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 19, offset := FLAG_SIMD, size := 0 }, -- 60
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 61
   -- Reset ACCUM
   { kind := .AsyncDispatch, dst := 6, src := WORK_BASE + 20, offset := FLAG_MEM, size := 0 },  -- 62
   { kind := .Wait, dst := FLAG_MEM, src := 0, offset := 0, size := 0 },                         -- 63

   -- ========== SKIP_LOOP (action 64) ==========
   -- Load char
   { kind := .AsyncDispatch, dst := 6, src := WORK_BASE + 2, offset := FLAG_MEM, size := 0 },  -- 64
   { kind := .Wait, dst := FLAG_MEM, src := 0, offset := 0, size := 0 },                        -- 65
   -- r0 = char
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 3, offset := FLAG_SIMD, size := 0 }, -- 66
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 67
   -- r1 = ' '
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 4, offset := FLAG_SIMD, size := 0 }, -- 68
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 69
   -- r2 = char - ' '
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 5, offset := FLAG_SIMD, size := 0 }, -- 70
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 71
   -- Store r2
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 6, offset := FLAG_SIMD, size := 0 }, -- 72
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 73
   -- if char != ' ', jump to CHECK_PLUS (action 84)
   { kind := .ConditionalJump, src := DIGIT_COUNT, dst := 84, offset := 0, size := 4 },        -- 74
   -- char == ' ', increment POS and loop
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 14, offset := FLAG_SIMD, size := 0 }, -- 75
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 76
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 15, offset := FLAG_SIMD, size := 0 }, -- 77
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 78
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 16, offset := FLAG_SIMD, size := 0 }, -- 79
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 80
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 17, offset := FLAG_SIMD, size := 0 }, -- 81
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 82
   { kind := .ConditionalJump, src := CONST_ONE, dst := 64, offset := 0, size := 4 },           -- 83

   -- CHECK_PLUS (action 84)
   -- First check if char == 0 (literal case - no '+' operator)
   -- Store r0 (char) to CHAR_BUF
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 52, offset := FLAG_SIMD, size := 0 }, -- 84
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 85
   -- if char != 0, check for newline (jump to action 94)
   { kind := .ConditionalJump, src := CHAR_BUF, dst := 94, offset := 0, size := 4 },            -- 86

   -- LITERAL_OUTPUT (action 87): char == 0 OR char == newline, treat as literal
   -- char == 0 (literal): copy LEFT_VAL to RESULT and jump to ITOA_INIT
   -- r18 already has LEFT_VAL from W+29, but we need to load it first
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 29, offset := FLAG_SIMD, size := 0 }, -- 87: r18 = LEFT_VAL
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 88
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 51, offset := FLAG_SIMD, size := 0 }, -- 89: store r18 -> RESULT
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 90
   -- Jump to ITOA_INIT (action 170)
   { kind := .ConditionalJump, src := CONST_ONE, dst := 170, offset := 0, size := 4 },          -- 91
   fence, fence, -- 92, 93 padding

   -- CHECK_NEWLINE_PLUS (action 94) - check if char is newline (also means literal)
   -- r30 = CONST_10 (newline)
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 53, offset := FLAG_SIMD, size := 0 }, -- 94
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 95
   -- r31 = char - 10
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 54, offset := FLAG_SIMD, size := 0 }, -- 96
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 97
   -- Store r31 -> DIGIT_COUNT (temp)
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 55, offset := FLAG_SIMD, size := 0 }, -- 98
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 99
   -- if char != newline, continue to check for '+' (jump to action 104)
   { kind := .ConditionalJump, src := DIGIT_COUNT, dst := 104, offset := 0, size := 4 },        -- 100
   -- char == newline (literal): jump to LITERAL_OUTPUT (action 87)
   { kind := .ConditionalJump, src := CONST_ONE, dst := 87, offset := 0, size := 4 },           -- 101
   fence, fence, -- 102, 103 padding

   -- CHECK_PLUS_CONTINUE (action 104) - check for '+' operator
   -- r13 = '+'
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 21, offset := FLAG_SIMD, size := 0 }, -- 104
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 105
   -- r14 = char - '+'
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 22, offset := FLAG_SIMD, size := 0 }, -- 106
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 107
   -- Store r14 to DIGIT_COUNT
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 23, offset := FLAG_SIMD, size := 0 }, -- 108
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 109
   -- if char != '+', jump to PARSE_NUM2 (action 120)
   { kind := .ConditionalJump, src := DIGIT_COUNT, dst := 120, offset := 0, size := 4 },        -- 110
   -- char == '+', increment POS and go to SKIP_LOOP
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 14, offset := FLAG_SIMD, size := 0 }, -- 111
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 112
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 15, offset := FLAG_SIMD, size := 0 }, -- 113
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 114
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 16, offset := FLAG_SIMD, size := 0 }, -- 115
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 116
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 17, offset := FLAG_SIMD, size := 0 }, -- 117
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 118
   { kind := .ConditionalJump, src := CONST_ONE, dst := 64, offset := 0, size := 4 },            -- 119

   -- ========== PARSE_NUM2_LOOP (action 120) ==========
   -- Load char
   { kind := .AsyncDispatch, dst := 6, src := WORK_BASE + 2, offset := FLAG_MEM, size := 0 },  -- 120
   { kind := .Wait, dst := FLAG_MEM, src := 0, offset := 0, size := 0 },                        -- 121
   -- r0 = char
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 3, offset := FLAG_SIMD, size := 0 }, -- 122
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 123
   -- Check for null terminator (char == 0) to end NUM2
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 24, offset := FLAG_SIMD, size := 0 }, -- 124: store r0 to OUTPUT_POS
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 125
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 25, offset := FLAG_SIMD, size := 0 }, -- 126: padding
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 127
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 26, offset := FLAG_SIMD, size := 0 }, -- 128: padding
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 129
   -- if char != 0, jump to ACCUMULATE_NUM2 (action 134)
   { kind := .ConditionalJump, src := OUTPUT_POS, dst := 134, offset := 0, size := 4 },         -- 130
   -- char == 0, jump to AFTER_NUM2 (action 158)
   { kind := .ConditionalJump, src := CONST_ONE, dst := 158, offset := 0, size := 4 },          -- 131
   fence, fence,  -- 132, 133 padding

   -- ACCUMULATE_NUM2 (action 134)
   -- r3 = '0'
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 7, offset := FLAG_SIMD, size := 0 }, -- 134
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 135
   -- r4 = char - '0'
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 8, offset := FLAG_SIMD, size := 0 }, -- 136
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 137
   -- r5 = 10
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 9, offset := FLAG_SIMD, size := 0 }, -- 138
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                       -- 139
   -- r6 = ACCUM
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 10, offset := FLAG_SIMD, size := 0 }, -- 140
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 141
   -- r7 = accum * 10
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 11, offset := FLAG_SIMD, size := 0 }, -- 142
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 143
   -- r8 = accum*10 + digit
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 12, offset := FLAG_SIMD, size := 0 }, -- 144
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 145
   -- Store accum
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 13, offset := FLAG_SIMD, size := 0 }, -- 146
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 147
   -- r9 = POS
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 14, offset := FLAG_SIMD, size := 0 }, -- 148
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 149
   -- r10 = 1
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 15, offset := FLAG_SIMD, size := 0 }, -- 150
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 151
   -- r11 = pos + 1
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 16, offset := FLAG_SIMD, size := 0 }, -- 152
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 153
   -- Store pos
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 17, offset := FLAG_SIMD, size := 0 }, -- 154
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 155
   -- Jump back to PARSE_NUM2_LOOP (action 120)
   { kind := .ConditionalJump, src := CONST_ONE, dst := 120, offset := 0, size := 4 },          -- 156
   fence,  -- 157 padding

   -- ========== AFTER_NUM2 (action 158) ==========
   -- Save ACCUM to RIGHT_VAL
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 27, offset := FLAG_SIMD, size := 0 }, -- 158
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 159
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 28, offset := FLAG_SIMD, size := 0 }, -- 160
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 161

   -- ========== COMPUTE (action 162) ==========
   -- r18 = LEFT_VAL
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 29, offset := FLAG_SIMD, size := 0 }, -- 162
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 163
   -- r19 = RIGHT_VAL
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 30, offset := FLAG_SIMD, size := 0 }, -- 164
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 165
   -- r20 = left + right
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 31, offset := FLAG_SIMD, size := 0 }, -- 166
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 167
   -- Store RESULT
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 32, offset := FLAG_SIMD, size := 0 }, -- 168
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 169

   -- ========== ITOA_INIT (action 170) ==========
   -- Init OUTPUT_POS to 30
   { kind := .AsyncDispatch, dst := 6, src := WORK_BASE + 33, offset := FLAG_MEM, size := 0 },  -- 170
   { kind := .Wait, dst := FLAG_MEM, src := 0, offset := 0, size := 0 },                         -- 171
   -- Write '\n' at OUTPUT_BUF+31
   { kind := .AsyncDispatch, dst := 6, src := WORK_BASE + 34, offset := FLAG_MEM, size := 0 },  -- 172
   { kind := .Wait, dst := FLAG_MEM, src := 0, offset := 0, size := 0 },                         -- 173

   -- ========== ITOA_LOOP (action 174) ==========
   -- r21 = RESULT
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 35, offset := FLAG_SIMD, size := 0 }, -- 174
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 175
   -- r22 = result / 10
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 36, offset := FLAG_SIMD, size := 0 }, -- 176
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 177
   -- r23 = quotient * 10
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 37, offset := FLAG_SIMD, size := 0 }, -- 178
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 179
   -- r24 = result - quotient*10 (remainder)
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 38, offset := FLAG_SIMD, size := 0 }, -- 180
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 181
   -- r25 = remainder + '0'
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 39, offset := FLAG_SIMD, size := 0 }, -- 182
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 183
   -- Store r25 -> CHAR_BUF
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 40, offset := FLAG_SIMD, size := 0 }, -- 184
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 185
   -- MemStoreIndirect: CHAR_BUF -> OUTPUT_BUF[OUTPUT_POS]
   { kind := .AsyncDispatch, dst := 6, src := WORK_BASE + 41, offset := FLAG_MEM, size := 0 },  -- 186
   { kind := .Wait, dst := FLAG_MEM, src := 0, offset := 0, size := 0 },                         -- 187
   -- r26 = OUTPUT_POS
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 42, offset := FLAG_SIMD, size := 0 }, -- 188
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 189
   -- r27 = output_pos - 1
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 43, offset := FLAG_SIMD, size := 0 }, -- 190
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 191
   -- Store OUTPUT_POS
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 44, offset := FLAG_SIMD, size := 0 }, -- 192
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 193
   -- Store quotient -> RESULT
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 45, offset := FLAG_SIMD, size := 0 }, -- 194
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 195
   -- if RESULT != 0, loop back to ITOA_LOOP (action 174)
   { kind := .ConditionalJump, src := RESULT, dst := 174, offset := 0, size := 4 },             -- 196
   fence, -- 197 padding

   -- ========== OUTPUT (action 198) ==========
   -- Compute OUTPUT_POS + 1 -> DIGIT_COUNT
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 46, offset := FLAG_SIMD, size := 0 }, -- 198
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 199
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 47, offset := FLAG_SIMD, size := 0 }, -- 200
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 201
   { kind := .AsyncDispatch, dst := 1, src := WORK_BASE + 48, offset := FLAG_SIMD, size := 0 }, -- 202
   { kind := .Wait, dst := FLAG_SIMD, src := 0, offset := 0, size := 0 },                        -- 203
   -- MemCopyIndirect: OUTPUT_BUF[DIGIT_COUNT] -> LEFT_VAL
   { kind := .AsyncDispatch, dst := 6, src := WORK_BASE + 49, offset := FLAG_MEM, size := 0 },  -- 204
   { kind := .Wait, dst := FLAG_MEM, src := 0, offset := 0, size := 0 },                         -- 205
   -- FileWrite
   { kind := .AsyncDispatch, dst := 2, src := WORK_BASE + 50, offset := FLAG_FILE, size := 0 }, -- 206
   { kind := .Wait, dst := FLAG_FILE, src := 0, offset := 0, size := 0 }                         -- 207
  ]

def leanEvalAlgorithm : Algorithm := {
  actions := actualMainFlow ++ workActions,
  payloads := leanEvalPayloads,
  state := {
    regs_per_unit := 32,
    unit_scratch_offsets := [0],
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
    simd_units := 1,
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
  timeout_ms := some 10000,
  thread_name_prefix := some "lean-eval"
}

end Algorithm

def main : IO Unit := do
  let json := toJson Algorithm.leanEvalAlgorithm
  IO.println (Json.compress json)
