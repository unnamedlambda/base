import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace Algorithm

namespace C

def ZERO_N : Nat := 0
def SIZE_BYTE_N : Nat := 1
def UNIT_STEP_N : Nat := 1
def SINGLE_ACTION_N : Nat := 1
def SIZE_WORD_N : Nat := 4
def SIZE_16_N : Nat := 16
def SIZE_32_N : Nat := 32
def SIZE_256_N : Nat := 256
def SIZE_512_N : Nat := 512
def OUTPUT_BUF_SIZE_N : Nat := 32
def OUTPUT_POS_INIT_N : Nat := 30
def POS_INIT_N : Nat := 6
def OUTPUT_COPY_BYTES_U32 : UInt32 := 16
def OUTPUT_BUF_PAD_SIZE_N : Nat := 16
def ONE_MIB_N : Nat := 0x100000
def TIMEOUT_MS_N : Nat := 10000
def REGS_PER_UNIT_N : Nat := 32
def WORKER_THREADS_N : Nat := 2
def BLOCKING_THREADS_N : Nat := 2
def ARG_INDEX_N : Nat := 1
def ARG_PARAMS_PADDING_N : Nat := 8
def LAST_INDEX_DELTA_N : Nat := 1

def SIMD_STEP_ACTIONS_N : Nat := 2
def INC_POS_SIMD_STEPS_N : Nat := 4
def JUMP_PAD_ACTIONS_N : Nat := 4
def SKIP_SPACES_SIMD_STEPS_N : Nat := 3
def PARSE_DIGIT_SIMD_STEPS_N : Nat := 7
def ADD_RESULT_SIMD_STEPS_N : Nat := 6
def CHECK_PLUS_SIMD_STEPS_N : Nat := 3
def ITOA_INIT_MEM_STEPS_N : Nat := 2
def ITOA_LOOP_DIV_SIMD_STEPS_N : Nat := 6
def ITOA_LOOP_POST_SIMD_STEPS_N : Nat := 4
def OUTPUT_SIMD_STEPS_N : Nat := 3
def STORE_RESULT_SIMD_STEPS_N : Nat := 2

def SIMD_UNIT_ID_N : Nat := 1
def FILE_UNIT_ID_N : Nat := 2
def FFI_UNIT_ID_N : Nat := 4
def MEM_UNIT_ID_N : Nat := 6

def SIMD_UNIT_COUNT_N : Nat := 1
def GPU_UNIT_COUNT_N : Nat := 0
def COMPUTE_UNIT_COUNT_N : Nat := 0
def FILE_UNIT_COUNT_N : Nat := 1
def NETWORK_UNIT_COUNT_N : Nat := 0
def MEMORY_UNIT_COUNT_N : Nat := 1
def FFI_UNIT_COUNT_N : Nat := 1

def BYTE_MASK_N : Nat := 0xFF
def SHIFT_8_N : Nat := 8
def SHIFT_16_N : Nat := 16
def SHIFT_24_N : Nat := 24

def U32_WRAP_I : Int := 0x100000000

def ZERO_U8 : UInt8 := 0

def ASCII_ZERO_I : Int := 48
def ASCII_SPACE_I : Int := 32
def ASCII_PLUS_I : Int := 43
def ASCII_NEWLINE_I : Int := 10
def ASCII_LPAREN_I : Int := 40
def ASCII_RPAREN_I : Int := 41

def OFFSET_FFI_PTR_U32 : UInt32 := 0x0000
def OFFSET_OUTPUT_PATH_U32 : UInt32 := 0x0010
def OFFSET_ARGV_PARAMS_U32 : UInt32 := 0x0020
def OFFSET_FILENAME_IN_ARGV_U32 : UInt32 := 0x0028
def OFFSET_FILENAME_BUF_U32 : UInt32 := 0x0030
def OFFSET_FFI_RESULT_U32 : UInt32 := 0x0130
def OFFSET_FLAG_FFI_U32 : UInt32 := 0x0140
def OFFSET_FLAG_FILE_U32 : UInt32 := 0x0150
def OFFSET_FLAG_SIMD_U32 : UInt32 := 0x0160
def OFFSET_FLAG_MEM_U32 : UInt32 := 0x0170
def OFFSET_CONST_ZERO_U32 : UInt32 := 0x0180
def OFFSET_SOURCE_BUF_U32 : UInt32 := 0x0190
def OFFSET_CONST_48_U32 : UInt32 := 0x0390
def OFFSET_CONST_10_U32 : UInt32 := 0x03A0
def OFFSET_CONST_ONE_U32 : UInt32 := 0x03B0
def OFFSET_CONST_SPACE_U32 : UInt32 := 0x03C0
def OFFSET_CONST_PLUS_U32 : UInt32 := 0x03D0
def OFFSET_POS_U32 : UInt32 := 0x03E0
def OFFSET_CHAR_BUF_U32 : UInt32 := 0x03F0
def OFFSET_ACCUM_U32 : UInt32 := 0x0400
def OFFSET_LEFT_VAL_U32 : UInt32 := 0x0410
def OFFSET_RIGHT_VAL_U32 : UInt32 := 0x0420
def OFFSET_RESULT_U32 : UInt32 := 0x0430
def OFFSET_DIGIT_COUNT_U32 : UInt32 := 0x0440
def OFFSET_OUTPUT_POS_U32 : UInt32 := 0x0450
def OFFSET_OUTPUT_BUF_U32 : UInt32 := 0x0460
def OFFSET_CONST_RPAREN_U32 : UInt32 := 0x0490
def OFFSET_CONST_LPAREN_U32 : UInt32 := 0x04A0

end C

def i (n : Nat) : Int := Int.ofNat n

def int32ToBytes16 (n : Int) : List UInt8 :=
  let n32 := if n < i C.ZERO_N then (C.U32_WRAP_I + n).toNat else n.toNat
  let b0 := UInt8.ofNat (n32 &&& C.BYTE_MASK_N)
  let b1 := UInt8.ofNat ((n32 >>> C.SHIFT_8_N) &&& C.BYTE_MASK_N)
  let b2 := UInt8.ofNat ((n32 >>> C.SHIFT_16_N) &&& C.BYTE_MASK_N)
  let b3 := UInt8.ofNat ((n32 >>> C.SHIFT_24_N) &&& C.BYTE_MASK_N)
  [b0, b1, b2, b3] ++ zeros (C.SIZE_16_N - C.SIZE_WORD_N)

structure Layout where
  FFI_PTR : UInt32
  OUTPUT_PATH : UInt32
  ARGV_PARAMS : UInt32
  FILENAME_IN_ARGV : UInt32
  FILENAME_BUF : UInt32
  FILENAME_BUF_SIZE : Nat
  FFI_RESULT : UInt32
  FLAG_FFI : UInt32
  FLAG_FILE : UInt32
  FLAG_SIMD : UInt32
  FLAG_MEM : UInt32
  CONST_ZERO : UInt32
  SOURCE_BUF : UInt32
  SOURCE_BUF_SIZE : Nat
  CONST_48 : UInt32
  CONST_10 : UInt32
  CONST_ONE : UInt32
  CONST_SPACE : UInt32
  CONST_PLUS : UInt32
  POS : UInt32
  CHAR_BUF : UInt32
  ACCUM : UInt32
  LEFT_VAL : UInt32
  RIGHT_VAL : UInt32
  RESULT : UInt32
  DIGIT_COUNT : UInt32
  OUTPUT_POS : UInt32
  OUTPUT_BUF : UInt32
  CONST_RPAREN : UInt32
  CONST_LPAREN : UInt32


def layout : Layout := {
  FFI_PTR := C.OFFSET_FFI_PTR_U32,
  OUTPUT_PATH := C.OFFSET_OUTPUT_PATH_U32,
  ARGV_PARAMS := C.OFFSET_ARGV_PARAMS_U32,
  FILENAME_IN_ARGV := C.OFFSET_FILENAME_IN_ARGV_U32,
  FILENAME_BUF := C.OFFSET_FILENAME_BUF_U32,
  FILENAME_BUF_SIZE := C.SIZE_256_N,
  FFI_RESULT := C.OFFSET_FFI_RESULT_U32,
  FLAG_FFI := C.OFFSET_FLAG_FFI_U32,
  FLAG_FILE := C.OFFSET_FLAG_FILE_U32,
  FLAG_SIMD := C.OFFSET_FLAG_SIMD_U32,
  FLAG_MEM := C.OFFSET_FLAG_MEM_U32,
  CONST_ZERO := C.OFFSET_CONST_ZERO_U32,
  SOURCE_BUF := C.OFFSET_SOURCE_BUF_U32,
  SOURCE_BUF_SIZE := C.SIZE_512_N,
  CONST_48 := C.OFFSET_CONST_48_U32,
  CONST_10 := C.OFFSET_CONST_10_U32,
  CONST_ONE := C.OFFSET_CONST_ONE_U32,
  CONST_SPACE := C.OFFSET_CONST_SPACE_U32,
  CONST_PLUS := C.OFFSET_CONST_PLUS_U32,
  POS := C.OFFSET_POS_U32,
  CHAR_BUF := C.OFFSET_CHAR_BUF_U32,
  ACCUM := C.OFFSET_ACCUM_U32,
  LEFT_VAL := C.OFFSET_LEFT_VAL_U32,
  RIGHT_VAL := C.OFFSET_RIGHT_VAL_U32,
  RESULT := C.OFFSET_RESULT_U32,
  DIGIT_COUNT := C.OFFSET_DIGIT_COUNT_U32,
  OUTPUT_POS := C.OFFSET_OUTPUT_POS_U32,
  OUTPUT_BUF := C.OFFSET_OUTPUT_BUF_U32,
  CONST_RPAREN := C.OFFSET_CONST_RPAREN_U32,
  CONST_LPAREN := C.OFFSET_CONST_LPAREN_U32
}

def L : Layout := layout

def ZERO : UInt32 := u32 C.ZERO_N
def SIZE_NONE : UInt32 := u32 C.ZERO_N
def SIZE_BYTE : UInt32 := u32 C.SIZE_BYTE_N
def SIZE_I32 : UInt32 := u32 C.SIZE_WORD_N
def OUTPUT_BUF_SIZE : Nat := C.OUTPUT_BUF_SIZE_N
def OUTPUT_BUF_LAST : UInt32 := L.OUTPUT_BUF + u32 (OUTPUT_BUF_SIZE - C.LAST_INDEX_DELTA_N)
def OUTPUT_POS_INIT : UInt32 := u32 C.OUTPUT_POS_INIT_N
def OUTPUT_COPY_BYTES : UInt32 := C.OUTPUT_COPY_BYTES_U32


def leanEvalPayloads : List UInt8 :=
  let ffiPtr := zeros C.SIZE_16_N
  let outputPath := padTo (stringToBytes "/tmp/output.txt") C.SIZE_16_N
  let argvParams := uint32ToBytes (u32 C.ARG_INDEX_N) ++ uint32ToBytes (u32 (L.FILENAME_BUF_SIZE - C.ARG_PARAMS_PADDING_N)) ++ zeros C.ARG_PARAMS_PADDING_N
  let filenameBuf := zeros L.FILENAME_BUF_SIZE
  let ffiResult := zeros C.SIZE_16_N
  let flagFfi := zeros C.SIZE_16_N
  let flagFile := zeros C.SIZE_16_N
  let flagSimd := zeros C.SIZE_16_N
  let flagMem := zeros C.SIZE_16_N
  let constZero := int32ToBytes16 (i C.ZERO_N)
  let sourceBuf := zeros L.SOURCE_BUF_SIZE
  let const48 := int32ToBytes16 C.ASCII_ZERO_I
  let const10 := int32ToBytes16 C.ASCII_NEWLINE_I
  let constOne := int32ToBytes16 (i C.UNIT_STEP_N)
  let constSpace := int32ToBytes16 C.ASCII_SPACE_I
  let constPlus := int32ToBytes16 C.ASCII_PLUS_I
  let pos := int32ToBytes16 (i C.POS_INIT_N)
  let charBuf := zeros C.SIZE_16_N
  let accum := zeros C.SIZE_16_N
  let leftVal := zeros C.SIZE_16_N
  let rightVal := zeros C.SIZE_16_N
  let result := zeros C.SIZE_16_N
  let digitCount := zeros C.SIZE_16_N
  let outputPos := int32ToBytes16 (i C.OUTPUT_POS_INIT_N)
  let outputBuf := zeros C.OUTPUT_BUF_SIZE_N
  let outputPad := zeros C.OUTPUT_BUF_PAD_SIZE_N
  let constRParen := int32ToBytes16 C.ASCII_RPAREN_I
  let constLParen := int32ToBytes16 C.ASCII_LPAREN_I

  ffiPtr ++ outputPath ++ argvParams ++ filenameBuf ++ ffiResult ++
    flagFfi ++ flagFile ++ flagSimd ++ flagMem ++ constZero ++ sourceBuf ++
    const48 ++ const10 ++ constOne ++ constSpace ++ constPlus ++
    pos ++ charBuf ++ accum ++ leftVal ++ rightVal ++ result ++
    digitCount ++ outputPos ++ outputBuf ++ outputPad ++ constRParen ++ constLParen

def fence : Action := { kind := .Fence, dst := ZERO, src := ZERO, offset := ZERO, size := ZERO }

def w (kind : Kind) (src dst offset size : UInt32) : Action :=
  { kind := kind, src := src, dst := dst, offset := offset, size := size }

inductive WorkOp where
  | FfiCall
  | FileRead
  | MemCopyChar
  | LoadChar
  | LoadRParen
  | SubCharRParen
  | StoreDigitCountFromRParen
  | LoadLParen
  | SubCharLParen
  | StoreDigitCountFromLParen
  | LoadSpace
  | SubCharSpace
  | StoreDigitCountFromSpace
  | LoadAsciiZero
  | SubCharZero
  | LoadBase
  | LoadAccum
  | MulAccumBase
  | AddAccumDigit
  | StoreAccum
  | LoadPos
  | LoadOne
  | AddPosOne
  | StorePos
  | LoadAccumForResult
  | StoreResultFromAccum
  | ClearAccum
  | LoadPlus
  | SubCharPlus
  | StoreDigitCountFromPlus
  | StoreOutputPosZero
  | LoadBaseForDigitCheck
  | SubCharBase
  | LoadAccumForRight
  | StoreRightVal
  | LoadResultForAdd
  | LoadRightVal
  | AddResultRight
  | StoreResultAfterAdd
  | InitOutputPos
  | WriteOutputNewline
  | LoadResultForItoa
  | DivResultBase
  | MulQuotBase
  | SubResultQuot
  | AddRemAsciiZero
  | StoreCharBuf
  | StoreCharToOutput
  | LoadOutputPos
  | SubOutputPosOne
  | StoreOutputPos
  | StoreQuotToResult
  | LoadOutputPosForDigitCount
  | AddOutputPosOne
  | StoreDigitCountFinal
  | CopyOutputToLeft
  | FileWriteOutput
  | StoreResultFromLeft
  | ClearCharBuf
  | LoadBaseForNewlineCheck
  | SubCharNewline
  | StoreDigitCountFromNewline
  | ClearResult
  deriving BEq, DecidableEq, Repr

open WorkOp

inductive Reg where
  | RA | RB | RC | RD | RE | RF | RG | RH | RI | RJ | RK | RL | RM | RN | RO | RP
  | RQ | RR | RS | RT | RU | RV | RW | RX | RY | RZ
  | RAA | RAB | RAC | RAD | RAE | RAF
  deriving BEq, DecidableEq, Repr

open Reg

def regs : List Reg := [
  RA, RB, RC, RD, RE, RF, RG, RH, RI, RJ, RK, RL, RM, RN, RO, RP,
  RQ, RR, RS, RT, RU, RV, RW, RX, RY, RZ, RAA, RAB, RAC, RAD, RAE, RAF
]

def indexOf [BEq α] (xs : List α) (x : α) : Nat :=
  let rec go (i : Nat) (ys : List α) : Nat :=
    match ys with
    | [] => i
    | y :: ys' => if y == x then i else go (Nat.succ i) ys'
  go C.ZERO_N xs

def regIndex (r : Reg) : Nat := indexOf regs r

def r (reg : Reg) : UInt32 := UInt32.ofNat (regIndex reg)

def workOps : List WorkOp := [
  FfiCall,
  FileRead,
  MemCopyChar,
  LoadChar,
  LoadRParen,
  SubCharRParen,
  StoreDigitCountFromRParen,
  LoadLParen,
  SubCharLParen,
  StoreDigitCountFromLParen,
  LoadSpace,
  SubCharSpace,
  StoreDigitCountFromSpace,
  LoadAsciiZero,
  SubCharZero,
  LoadBase,
  LoadAccum,
  MulAccumBase,
  AddAccumDigit,
  StoreAccum,
  LoadPos,
  LoadOne,
  AddPosOne,
  StorePos,
  LoadAccumForResult,
  StoreResultFromAccum,
  ClearAccum,
  LoadPlus,
  SubCharPlus,
  StoreDigitCountFromPlus,
  StoreOutputPosZero,
  LoadBaseForDigitCheck,
  SubCharBase,
  LoadAccumForRight,
  StoreRightVal,
  LoadResultForAdd,
  LoadRightVal,
  AddResultRight,
  StoreResultAfterAdd,
  InitOutputPos,
  WriteOutputNewline,
  LoadResultForItoa,
  DivResultBase,
  MulQuotBase,
  SubResultQuot,
  AddRemAsciiZero,
  StoreCharBuf,
  StoreCharToOutput,
  LoadOutputPos,
  SubOutputPosOne,
  StoreOutputPos,
  StoreQuotToResult,
  LoadOutputPosForDigitCount,
  AddOutputPosOne,
  StoreDigitCountFinal,
  CopyOutputToLeft,
  FileWriteOutput,
  StoreResultFromLeft,
  ClearCharBuf,
  LoadBaseForNewlineCheck,
  SubCharNewline,
  StoreDigitCountFromNewline,
  ClearResult
]

def workIndex (op : WorkOp) : Nat :=
  indexOf workOps op


def jumpPad (a b : Action) : List Action := [a, b, fence, fence]

def opToAction : WorkOp -> Action
  | FfiCall => w .FFICall L.FFI_PTR L.ARGV_PARAMS L.FFI_RESULT SIZE_NONE
  | FileRead => w .FileRead L.FILENAME_IN_ARGV L.SOURCE_BUF (L.FILENAME_BUF_SIZE.toUInt32) (L.SOURCE_BUF_SIZE.toUInt32)
  | MemCopyChar => w .MemCopyIndirect L.POS L.CHAR_BUF L.SOURCE_BUF SIZE_BYTE
  | LoadChar => w .SimdLoadI32 L.CHAR_BUF (r RA) ZERO SIZE_NONE
  | LoadRParen => w .SimdLoadI32 L.CONST_RPAREN (r RB) ZERO SIZE_NONE
  | SubCharRParen => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromRParen => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | LoadLParen => w .SimdLoadI32 L.CONST_LPAREN (r RB) ZERO SIZE_NONE
  | SubCharLParen => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromLParen => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | LoadSpace => w .SimdLoadI32 L.CONST_SPACE (r RB) ZERO SIZE_NONE
  | SubCharSpace => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromSpace => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | LoadAsciiZero => w .SimdLoadI32 L.CONST_48 (r RD) ZERO SIZE_NONE
  | SubCharZero => w .SimdSubI32 (r RA) (r RE) (r RD) SIZE_NONE
  | LoadBase => w .SimdLoadI32 L.CONST_10 (r RF) ZERO SIZE_NONE
  | LoadAccum => w .SimdLoadI32 L.ACCUM (r RG) ZERO SIZE_NONE
  | MulAccumBase => w .SimdMulI32 (r RG) (r RH) (r RF) SIZE_NONE
  | AddAccumDigit => w .SimdAddI32 (r RH) (r RI) (r RE) SIZE_NONE
  | StoreAccum => w .SimdStoreI32 (r RI) (r RA) L.ACCUM SIZE_NONE
  | LoadPos => w .SimdLoadI32 L.POS (r RJ) ZERO SIZE_NONE
  | LoadOne => w .SimdLoadI32 L.CONST_ONE (r RK) ZERO SIZE_NONE
  | AddPosOne => w .SimdAddI32 (r RJ) (r RL) (r RK) SIZE_NONE
  | StorePos => w .SimdStoreI32 (r RL) (r RA) L.POS SIZE_NONE
  | LoadAccumForResult => w .SimdLoadI32 L.ACCUM (r RM) ZERO SIZE_NONE
  | StoreResultFromAccum => w .SimdStoreI32 (r RM) (r RA) L.RESULT SIZE_NONE
  | ClearAccum => w .MemWrite ZERO L.ACCUM ZERO SIZE_I32
  | LoadPlus => w .SimdLoadI32 L.CONST_PLUS (r RN) ZERO SIZE_NONE
  | SubCharPlus => w .SimdSubI32 (r RA) (r RO) (r RN) SIZE_NONE
  | StoreDigitCountFromPlus => w .SimdStoreI32 (r RO) (r RA) L.DIGIT_COUNT SIZE_NONE
  | StoreOutputPosZero => w .SimdStoreI32 (r RA) (r RA) L.OUTPUT_POS SIZE_NONE
  | LoadBaseForDigitCheck => w .SimdLoadI32 L.CONST_10 (r RP) ZERO SIZE_NONE
  | SubCharBase => w .SimdSubI32 (r RA) (r RQ) (r RP) SIZE_NONE
  | LoadAccumForRight => w .SimdLoadI32 L.ACCUM (r RR) ZERO SIZE_NONE
  | StoreRightVal => w .SimdStoreI32 (r RR) (r RA) L.RIGHT_VAL SIZE_NONE
  | LoadResultForAdd => w .SimdLoadI32 L.RESULT (r RS) ZERO SIZE_NONE
  | LoadRightVal => w .SimdLoadI32 L.RIGHT_VAL (r RT) ZERO SIZE_NONE
  | AddResultRight => w .SimdAddI32 (r RS) (r RU) (r RT) SIZE_NONE
  | StoreResultAfterAdd => w .SimdStoreI32 (r RU) (r RA) L.RESULT SIZE_NONE
  | InitOutputPos => w .MemWrite OUTPUT_POS_INIT L.OUTPUT_POS ZERO SIZE_I32
  | WriteOutputNewline => w .MemCopy L.CONST_10 OUTPUT_BUF_LAST ZERO SIZE_BYTE
  | LoadResultForItoa => w .SimdLoadI32 L.RESULT (r RV) ZERO SIZE_NONE
  | DivResultBase => w .SimdDivI32 (r RV) (r RW) (r RF) SIZE_NONE
  | MulQuotBase => w .SimdMulI32 (r RW) (r RX) (r RF) SIZE_NONE
  | SubResultQuot => w .SimdSubI32 (r RV) (r RY) (r RX) SIZE_NONE
  | AddRemAsciiZero => w .SimdAddI32 (r RY) (r RZ) (r RD) SIZE_NONE
  | StoreCharBuf => w .SimdStoreI32 (r RZ) (r RA) L.CHAR_BUF SIZE_NONE
  | StoreCharToOutput => w .MemStoreIndirect L.CHAR_BUF L.OUTPUT_POS L.OUTPUT_BUF SIZE_BYTE
  | LoadOutputPos => w .SimdLoadI32 L.OUTPUT_POS (r RAA) ZERO SIZE_NONE
  | SubOutputPosOne => w .SimdSubI32 (r RAA) (r RAB) (r RK) SIZE_NONE
  | StoreOutputPos => w .SimdStoreI32 (r RAB) (r RA) L.OUTPUT_POS SIZE_NONE
  | StoreQuotToResult => w .SimdStoreI32 (r RW) (r RA) L.RESULT SIZE_NONE
  | LoadOutputPosForDigitCount => w .SimdLoadI32 L.OUTPUT_POS (r RAC) ZERO SIZE_NONE
  | AddOutputPosOne => w .SimdAddI32 (r RAC) (r RAD) (r RK) SIZE_NONE
  | StoreDigitCountFinal => w .SimdStoreI32 (r RAD) (r RA) L.DIGIT_COUNT SIZE_NONE
  | CopyOutputToLeft => w .MemCopyIndirect L.DIGIT_COUNT L.LEFT_VAL L.OUTPUT_BUF OUTPUT_COPY_BYTES
  | FileWriteOutput => w .FileWrite L.LEFT_VAL L.OUTPUT_PATH OUTPUT_COPY_BYTES SIZE_NONE
  | StoreResultFromLeft => w .SimdStoreI32 (r RS) (r RA) L.RESULT SIZE_NONE
  | ClearCharBuf => w .SimdStoreI32 (r RA) (r RA) L.CHAR_BUF SIZE_NONE
  | LoadBaseForNewlineCheck => w .SimdLoadI32 L.CONST_10 (r RAE) ZERO SIZE_NONE
  | SubCharNewline => w .SimdSubI32 (r RA) (r RAF) (r RAE) SIZE_NONE
  | StoreDigitCountFromNewline => w .SimdStoreI32 (r RAF) (r RA) L.DIGIT_COUNT SIZE_NONE
  | ClearResult => w .MemWrite ZERO L.RESULT ZERO SIZE_I32


def workActions : List Action := workOps.map opToAction

def SIMD_STEP_LEN : Nat := C.SIMD_STEP_ACTIONS_N
def MEM_STEP_LEN : Nat := C.SIMD_STEP_ACTIONS_N
def FILE_STEP_LEN : Nat := C.SIMD_STEP_ACTIONS_N
def FFI_STEP_LEN : Nat := C.SIMD_STEP_ACTIONS_N
def LOAD_CHAR_LEN : Nat := MEM_STEP_LEN + SIMD_STEP_LEN
def INC_POS_LEN : Nat := SIMD_STEP_LEN * C.INC_POS_SIMD_STEPS_N
def JUMP_PAD_LEN : Nat := C.JUMP_PAD_ACTIONS_N
def SKIP_SPACES_LEN : Nat := LOAD_CHAR_LEN + SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N + C.SINGLE_ACTION_N + INC_POS_LEN + C.SINGLE_ACTION_N
def PARSE_NUMBER_LEN : Nat :=
  let p0 := LOAD_CHAR_LEN + SIMD_STEP_LEN
  let p1 := JUMP_PAD_LEN
  let p2 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N
  let p3 := JUMP_PAD_LEN
  let p4 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N
  let p5 := JUMP_PAD_LEN
  let p6 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N
  let p7 := JUMP_PAD_LEN
  let p8 := SIMD_STEP_LEN * C.PARSE_DIGIT_SIMD_STEPS_N
  let p9 := INC_POS_LEN + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
def PARSE_ADDEND_LEN : Nat := PARSE_NUMBER_LEN
def ADD_TO_RESULT_LEN : Nat := SIMD_STEP_LEN * C.ADD_RESULT_SIMD_STEPS_N + MEM_STEP_LEN
def CHECK_PLUS_ONLY_LEN : Nat := LOAD_CHAR_LEN + SIMD_STEP_LEN * C.CHECK_PLUS_SIMD_STEPS_N + C.SINGLE_ACTION_N + INC_POS_LEN + C.SINGLE_ACTION_N
def CHECK_PLUS_LEN : Nat := SKIP_SPACES_LEN + CHECK_PLUS_ONLY_LEN
def ITOA_INIT_LEN : Nat := MEM_STEP_LEN * C.ITOA_INIT_MEM_STEPS_N
def ITOA_LOOP_LEN : Nat := SIMD_STEP_LEN * C.ITOA_LOOP_DIV_SIMD_STEPS_N + MEM_STEP_LEN + SIMD_STEP_LEN * C.ITOA_LOOP_POST_SIMD_STEPS_N + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
def OUTPUT_LEN : Nat := SIMD_STEP_LEN * C.OUTPUT_SIMD_STEPS_N + MEM_STEP_LEN + FILE_STEP_LEN
def SETUP_LEN : Nat := FFI_STEP_LEN + FILE_STEP_LEN + MEM_STEP_LEN + MEM_STEP_LEN
def LAMBDA_CHECK_LEN : Nat := LOAD_CHAR_LEN + SIMD_STEP_LEN * C.CHECK_PLUS_SIMD_STEPS_N + C.SINGLE_ACTION_N
def FIND_PLUS_LEN : Nat := LOAD_CHAR_LEN + SIMD_STEP_LEN * C.CHECK_PLUS_SIMD_STEPS_N + JUMP_PAD_LEN + INC_POS_LEN + C.SINGLE_ACTION_N
def LAMBDA_PATH_LEN : Nat :=
  FIND_PLUS_LEN + INC_POS_LEN + SKIP_SPACES_LEN + PARSE_ADDEND_LEN +
  (SIMD_STEP_LEN * C.STORE_RESULT_SIMD_STEPS_N) + MEM_STEP_LEN +
  INC_POS_LEN + SKIP_SPACES_LEN + PARSE_NUMBER_LEN + ADD_TO_RESULT_LEN + C.SINGLE_ACTION_N
def SKIP_PARSE_LEN : Nat := SKIP_SPACES_LEN + PARSE_NUMBER_LEN
def MAIN_FLOW_LEN : Nat :=
  SETUP_LEN + LAMBDA_CHECK_LEN + LAMBDA_PATH_LEN + SKIP_PARSE_LEN + ADD_TO_RESULT_LEN + CHECK_PLUS_LEN +
  ITOA_INIT_LEN + ITOA_LOOP_LEN + OUTPUT_LEN

def WORK_BASE : UInt32 := UInt32.ofNat MAIN_FLOW_LEN

def workSrc (op : WorkOp) : UInt32 := WORK_BASE + u32 (workIndex op)

def async (unit : UInt32) (op : WorkOp) (flag : UInt32) : Action :=
  { kind := .AsyncDispatch, dst := unit, src := workSrc op, offset := flag, size := SIZE_NONE }

def SIMD_UNIT_ID : UInt32 := u32 C.SIMD_UNIT_ID_N
def FILE_UNIT_ID : UInt32 := u32 C.FILE_UNIT_ID_N
def FFI_UNIT_ID : UInt32 := u32 C.FFI_UNIT_ID_N
def MEM_UNIT_ID : UInt32 := u32 C.MEM_UNIT_ID_N

def simd (op : WorkOp) : Action := async SIMD_UNIT_ID op L.FLAG_SIMD

def mem (op : WorkOp) : Action := async MEM_UNIT_ID op L.FLAG_MEM

def file (op : WorkOp) : Action := async FILE_UNIT_ID op L.FLAG_FILE

def ffi (op : WorkOp) : Action := async FFI_UNIT_ID op L.FLAG_FFI

def wait (flag : UInt32) : Action :=
  { kind := .Wait, dst := flag, src := ZERO, offset := ZERO, size := SIZE_NONE }

def jumpIf (src : UInt32) (dst : UInt32) : Action :=
  { kind := .ConditionalJump, src := src, dst := dst, offset := ZERO, size := SIZE_I32 }

def jumpIfN (src : UInt32) (dst : Nat) : Action := jumpIf src (u32 dst)

def simdStep (op : WorkOp) : List Action := [simd op, wait L.FLAG_SIMD]

def memStep (op : WorkOp) : List Action := [mem op, wait L.FLAG_MEM]

def fileStep (op : WorkOp) : List Action := [file op, wait L.FLAG_FILE]

def ffiStep (op : WorkOp) : List Action := [ffi op, wait L.FLAG_FFI]

def incPos : List Action := simdStep LoadPos ++ simdStep LoadOne ++ simdStep AddPosOne ++ simdStep StorePos

def loadChar : List Action := memStep MemCopyChar ++ simdStep LoadChar

def skipSpacesBlock (loopStart done : Nat) : List Action :=
  loadChar ++
  simdStep LoadSpace ++ simdStep SubCharSpace ++ simdStep StoreDigitCountFromSpace ++
  [jumpIfN L.DIGIT_COUNT done] ++
  incPos ++
  [jumpIfN L.CONST_ONE loopStart]

def parseNumberBlock (loopStart done : Nat) : List Action :=
  let p0 := loadChar ++ simdStep ClearCharBuf
  let newlineCheck := loopStart + (p0.length + JUMP_PAD_LEN)
  let p1 := jumpPad (jumpIfN L.CHAR_BUF newlineCheck) (jumpIfN L.CONST_ONE done)
  let p2 := simdStep LoadBaseForNewlineCheck ++ simdStep SubCharNewline ++ simdStep StoreDigitCountFromNewline
  let spaceCheck := loopStart + (p0.length + p1.length + p2.length + JUMP_PAD_LEN)
  let p3 := jumpPad (jumpIfN L.DIGIT_COUNT spaceCheck) (jumpIfN L.CONST_ONE done)
  let p4 := simdStep LoadSpace ++ simdStep SubCharSpace ++ simdStep StoreDigitCountFromSpace
  let plusCheck := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + JUMP_PAD_LEN)
  let p5 := jumpPad (jumpIfN L.DIGIT_COUNT plusCheck) (jumpIfN L.CONST_ONE done)
  let p6 := simdStep LoadPlus ++ simdStep SubCharPlus ++ simdStep StoreDigitCountFromPlus
  let accumulate := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + JUMP_PAD_LEN)
  let p7 := jumpPad (jumpIfN L.DIGIT_COUNT accumulate) (jumpIfN L.CONST_ONE done)
  let p8 := simdStep LoadAsciiZero ++ simdStep SubCharZero ++ simdStep LoadBase ++ simdStep LoadAccum ++ simdStep MulAccumBase ++ simdStep AddAccumDigit ++ simdStep StoreAccum
  let p9 := incPos ++ [jumpIfN L.CONST_ONE loopStart, fence]
  p0 ++ p1 ++ p2 ++ p3 ++ p4 ++ p5 ++ p6 ++ p7 ++ p8 ++ p9

def addToResultBlock : List Action :=
  simdStep LoadAccumForRight ++ simdStep StoreRightVal ++
  simdStep LoadResultForAdd ++ simdStep LoadRightVal ++ simdStep AddResultRight ++ simdStep StoreResultAfterAdd ++
  memStep ClearAccum

def findPlusBlock (loopStart done : Nat) : List Action :=
  let continueStart := loopStart + (LOAD_CHAR_LEN + SIMD_STEP_LEN * C.CHECK_PLUS_SIMD_STEPS_N + JUMP_PAD_LEN)
  loadChar ++
  simdStep LoadPlus ++ simdStep SubCharPlus ++ simdStep StoreDigitCountFromPlus ++
  jumpPad (jumpIfN L.DIGIT_COUNT continueStart) (jumpIfN L.CONST_ONE done) ++
  incPos ++
  [jumpIfN L.CONST_ONE loopStart]

def parseAddendBlock (loopStart done : Nat) : List Action :=
  let p0 := loadChar ++ simdStep ClearCharBuf
  let rparenCheck := loopStart + (p0.length + JUMP_PAD_LEN)
  let p1 := jumpPad (jumpIfN L.CHAR_BUF rparenCheck) (jumpIfN L.CONST_ONE done)
  let p2 := simdStep LoadRParen ++ simdStep SubCharRParen ++ simdStep StoreDigitCountFromRParen
  let newlineCheck := loopStart + (p0.length + p1.length + p2.length + JUMP_PAD_LEN)
  let p3 := jumpPad (jumpIfN L.DIGIT_COUNT newlineCheck) (jumpIfN L.CONST_ONE done)
  let p4 := simdStep LoadBaseForNewlineCheck ++ simdStep SubCharNewline ++ simdStep StoreDigitCountFromNewline
  let spaceCheck := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + JUMP_PAD_LEN)
  let p5 := jumpPad (jumpIfN L.DIGIT_COUNT spaceCheck) (jumpIfN L.CONST_ONE done)
  let p6 := simdStep LoadSpace ++ simdStep SubCharSpace ++ simdStep StoreDigitCountFromSpace
  let accumulate := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + JUMP_PAD_LEN)
  let p7 := jumpPad (jumpIfN L.DIGIT_COUNT accumulate) (jumpIfN L.CONST_ONE done)
  let p8 := simdStep LoadAsciiZero ++ simdStep SubCharZero ++ simdStep LoadBase ++ simdStep LoadAccum ++ simdStep MulAccumBase ++ simdStep AddAccumDigit ++ simdStep StoreAccum
  let p9 := incPos ++ [jumpIfN L.CONST_ONE loopStart, fence]
  p0 ++ p1 ++ p2 ++ p3 ++ p4 ++ p5 ++ p6 ++ p7 ++ p8 ++ p9

def checkPlusBlock (loopStart skipParseStart outputStart : Nat) : List Action :=
  let afterSkip := loopStart + SKIP_SPACES_LEN
  let skip := skipSpacesBlock loopStart afterSkip
  let plusCheck :=
    loadChar ++
    simdStep LoadPlus ++ simdStep SubCharPlus ++ simdStep StoreDigitCountFromPlus ++
    [jumpIfN L.DIGIT_COUNT outputStart] ++
    incPos ++
  [jumpIfN L.CONST_ONE skipParseStart]
  skip ++ plusCheck

def lambdaCheckBlock (normalStart : Nat) : List Action :=
  loadChar ++
  simdStep LoadLParen ++ simdStep SubCharLParen ++ simdStep StoreDigitCountFromLParen ++
  [jumpIfN L.DIGIT_COUNT normalStart]

def lambdaPathBlock (loopStart outputStart : Nat) : List Action :=
  let findPlusDone := loopStart + FIND_PLUS_LEN
  let findPlus := findPlusBlock loopStart findPlusDone
  let skipPlus := incPos
  let skipToAddendStart := loopStart + findPlus.length + skipPlus.length
  let addendStart := skipToAddendStart + SKIP_SPACES_LEN
  let addendDone := addendStart + PARSE_ADDEND_LEN
  let skipToAddend := skipSpacesBlock skipToAddendStart addendStart
  let parseAddend := parseAddendBlock addendStart addendDone
  let storeAddend := simdStep LoadAccumForResult ++ simdStep StoreResultFromAccum
  let clearAccum := memStep ClearAccum
  let skipRParen := incPos
  let skipToArgStart := addendDone + storeAddend.length + clearAccum.length + skipRParen.length
  let argStart := skipToArgStart + SKIP_SPACES_LEN
  let argDone := argStart + PARSE_NUMBER_LEN
  let skipToArg := skipSpacesBlock skipToArgStart argStart
  let parseArg := parseNumberBlock argStart argDone
  let addToResult := addToResultBlock
  let jumpOutput := [jumpIfN L.CONST_ONE outputStart]
  findPlus ++ skipPlus ++ skipToAddend ++ parseAddend ++ storeAddend ++ clearAccum ++ skipRParen ++ skipToArg ++ parseArg ++ addToResult ++ jumpOutput

def itoaLoopBlock (loopStart : Nat) : List Action :=
  simdStep LoadResultForItoa ++ simdStep DivResultBase ++ simdStep MulQuotBase ++ simdStep SubResultQuot ++
  simdStep AddRemAsciiZero ++ simdStep StoreCharBuf ++
  memStep StoreCharToOutput ++
  simdStep LoadOutputPos ++ simdStep SubOutputPosOne ++ simdStep StoreOutputPos ++ simdStep StoreQuotToResult ++
  [jumpIfN L.RESULT loopStart, fence]

def skipParseBlock (start done : Nat) : List Action :=
  let parseStart := start + SKIP_SPACES_LEN
  skipSpacesBlock start parseStart ++ parseNumberBlock parseStart done

def itoaInitBlock : List Action := memStep InitOutputPos ++ memStep WriteOutputNewline

def outputBlock : List Action :=
  simdStep LoadOutputPosForDigitCount ++ simdStep AddOutputPosOne ++ simdStep StoreDigitCountFinal ++
  memStep CopyOutputToLeft ++
  fileStep FileWriteOutput

def actualMainFlow : List Action :=
  let setup := ffiStep FfiCall ++ fileStep FileRead ++ memStep ClearAccum ++ memStep ClearResult

  let lambdaCheckStart := SETUP_LEN
  let lambdaPathStart := lambdaCheckStart + LAMBDA_CHECK_LEN
  let normalStart := lambdaPathStart + LAMBDA_PATH_LEN

  let skipParseStart := normalStart
  let addStart := skipParseStart + SKIP_PARSE_LEN
  let checkPlusStart := addStart + ADD_TO_RESULT_LEN
  let outputStart := checkPlusStart + CHECK_PLUS_LEN
  let itoaStart := outputStart
  let itoaLoopStart := itoaStart + ITOA_INIT_LEN

  let lambdaCheck := lambdaCheckBlock normalStart
  let lambdaPath := lambdaPathBlock lambdaPathStart outputStart
  let skipParse := skipParseBlock skipParseStart addStart
  let addToResult := addToResultBlock
  let checkPlus := checkPlusBlock checkPlusStart skipParseStart outputStart
  let itoaInit := itoaInitBlock
  let itoaLoop := itoaLoopBlock itoaLoopStart
  let output := outputBlock

  setup ++
  lambdaCheck ++
  lambdaPath ++
  skipParse ++
  addToResult ++
  checkPlus ++
  itoaInit ++
  itoaLoop ++
  output

def leanEvalAlgorithm : Algorithm := {
  actions := actualMainFlow ++ workActions,
  payloads := leanEvalPayloads,
  state := {
    regs_per_unit := C.REGS_PER_UNIT_N,
    unit_scratch_offsets := [C.ZERO_N],
    unit_scratch_size := C.ZERO_N,
    shared_data_offset := C.ONE_MIB_N,
    shared_data_size := C.ONE_MIB_N,
    gpu_offset := C.ZERO_N,
    gpu_size := C.ZERO_N,
    computational_regs := C.REGS_PER_UNIT_N,
    file_buffer_size := C.ONE_MIB_N,
    gpu_shader_offsets := []
  },
  units := {
    simd_units := C.SIMD_UNIT_COUNT_N,
    gpu_units := C.GPU_UNIT_COUNT_N,
    computational_units := C.COMPUTE_UNIT_COUNT_N,
    file_units := C.FILE_UNIT_COUNT_N,
    network_units := C.NETWORK_UNIT_COUNT_N,
    memory_units := C.MEMORY_UNIT_COUNT_N,
    ffi_units := C.FFI_UNIT_COUNT_N,
    backends_bits := ZERO
  },
  simd_assignments := [],
  computational_assignments := [],
  memory_assignments := [],
  file_assignments := [],
  network_assignments := [],
  ffi_assignments := [],
  gpu_assignments := [],
  worker_threads := some C.WORKER_THREADS_N,
  blocking_threads := some C.BLOCKING_THREADS_N,
  stack_size := none,
  timeout_ms := some C.TIMEOUT_MS_N,
  thread_name_prefix := some "lean-eval"
}

end Algorithm

def main : IO Unit := do
  let json := toJson Algorithm.leanEvalAlgorithm
  IO.println (Json.compress json)
