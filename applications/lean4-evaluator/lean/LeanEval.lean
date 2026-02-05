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
def ASCII_STAR_I : Int := 42
def ASCII_MINUS_I : Int := 45
def ASCII_L_I : Int := 108
def ASCII_E_I : Int := 101
def ASCII_COLON_I : Int := 58
def ASCII_EQUALS_I : Int := 61
def ASCII_I_CHAR_I : Int := 105
def ASCII_N_CHAR_I : Int := 110
def ASCII_X_I : Int := 120
def ASCII_SEMICOLON_I : Int := 59
def ASCII_F_I : Int := 102

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
def OFFSET_TERM_U32 : UInt32 := 0x04B0
def OFFSET_CONST_STAR_U32 : UInt32 := 0x04C0
def OFFSET_CONST_MINUS_U32 : UInt32 := 0x04D0
def OFFSET_CONST_L_U32 : UInt32 := 0x04E0
def OFFSET_CONST_E_U32 : UInt32 := 0x04F0
def OFFSET_CONST_COLON_U32 : UInt32 := 0x0500
def OFFSET_CONST_EQUALS_U32 : UInt32 := 0x0510
def OFFSET_CONST_I_CHAR_U32 : UInt32 := 0x0520
def OFFSET_CONST_N_CHAR_U32 : UInt32 := 0x0530
def OFFSET_CONST_X_U32 : UInt32 := 0x0540
def OFFSET_VAR_VAL_U32 : UInt32 := 0x0550
def OFFSET_CONST_SEMICOLON_U32 : UInt32 := 0x0560
def OFFSET_CONST_F_U32 : UInt32 := 0x0570
def OFFSET_SAVED_RESULT_U32 : UInt32 := 0x0580
def OFFSET_SAVED_TERM_U32 : UInt32 := 0x0590

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
  TERM : UInt32
  CONST_STAR : UInt32
  CONST_MINUS : UInt32
  CONST_L : UInt32
  CONST_E : UInt32
  CONST_COLON : UInt32
  CONST_EQUALS : UInt32
  CONST_I_CHAR : UInt32
  CONST_N_CHAR : UInt32
  CONST_X : UInt32
  VAR_VAL : UInt32
  CONST_SEMICOLON : UInt32
  CONST_F : UInt32
  SAVED_RESULT : UInt32
  SAVED_TERM : UInt32


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
  CONST_LPAREN := C.OFFSET_CONST_LPAREN_U32,
  TERM := C.OFFSET_TERM_U32,
  CONST_STAR := C.OFFSET_CONST_STAR_U32,
  CONST_MINUS := C.OFFSET_CONST_MINUS_U32,
  CONST_L := C.OFFSET_CONST_L_U32,
  CONST_E := C.OFFSET_CONST_E_U32,
  CONST_COLON := C.OFFSET_CONST_COLON_U32,
  CONST_EQUALS := C.OFFSET_CONST_EQUALS_U32,
  CONST_I_CHAR := C.OFFSET_CONST_I_CHAR_U32,
  CONST_N_CHAR := C.OFFSET_CONST_N_CHAR_U32,
  CONST_X := C.OFFSET_CONST_X_U32,
  VAR_VAL := C.OFFSET_VAR_VAL_U32,
  CONST_SEMICOLON := C.OFFSET_CONST_SEMICOLON_U32,
  CONST_F := C.OFFSET_CONST_F_U32,
  SAVED_RESULT := C.OFFSET_SAVED_RESULT_U32,
  SAVED_TERM := C.OFFSET_SAVED_TERM_U32
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
  let term := zeros C.SIZE_16_N
  let constStar := int32ToBytes16 C.ASCII_STAR_I
  let constMinus := int32ToBytes16 C.ASCII_MINUS_I
  let constL := int32ToBytes16 C.ASCII_L_I
  let constE := int32ToBytes16 C.ASCII_E_I
  let constColon := int32ToBytes16 C.ASCII_COLON_I
  let constEquals := int32ToBytes16 C.ASCII_EQUALS_I
  let constIChar := int32ToBytes16 C.ASCII_I_CHAR_I
  let constNChar := int32ToBytes16 C.ASCII_N_CHAR_I
  let constX := int32ToBytes16 C.ASCII_X_I
  let varVal := zeros C.SIZE_16_N
  let constSemicolon := int32ToBytes16 C.ASCII_SEMICOLON_I
  let constF := int32ToBytes16 C.ASCII_F_I
  let savedResult := zeros C.SIZE_16_N
  let savedTerm := zeros C.SIZE_16_N

  ffiPtr ++ outputPath ++ argvParams ++ filenameBuf ++ ffiResult ++
    flagFfi ++ flagFile ++ flagSimd ++ flagMem ++ constZero ++ sourceBuf ++
    const48 ++ const10 ++ constOne ++ constSpace ++ constPlus ++
    pos ++ charBuf ++ accum ++ leftVal ++ rightVal ++ result ++
    digitCount ++ outputPos ++ outputBuf ++ outputPad ++ constRParen ++ constLParen ++
    term ++ constStar ++ constMinus ++ constL ++ constE ++ constColon ++
    constEquals ++ constIChar ++ constNChar ++ constX ++ varVal ++ constSemicolon ++
    constF ++ savedResult ++ savedTerm

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
  | LoadStar
  | SubCharStar
  | StoreDigitCountFromStar
  | LoadTerm
  | StoreTerm
  | StoreTermFromAccum
  | MulTermAccum
  | AddResultTerm
  | StoreResultAfterTermAdd
  | ClearTerm
  | MulResultRight
  | StoreResultAfterMul
  | ClearLeftVal
  | SetLeftVal
  | LoadMinus
  | SubCharMinus
  | StoreDigitCountFromMinus
  | SubResultTerm
  | StoreResultAfterSub
  | SetLeftValTwo
  | SubResultRight
  | StoreResultAfterSubRight
  | LoadLeftValForCheck
  | SubLeftValOne
  | StoreDigitCountFromLeftCheck
  | LoadZeroForNegate
  | LoadAccumForNegate
  | SubZeroAccum
  | StoreTermFromNegate
  | LoadL
  | SubCharL
  | StoreDigitCountFromL
  | LoadVarVal
  | StoreVarValFromAccum
  | AddVarValAccum
  | StoreResultFromVarValAdd
  | MulVarValAccum
  | StoreResultFromVarValMul
  | SubVarValAccum
  | StoreResultFromVarValSub
  | LoadSemicolon
  | SubCharSemicolon
  | StoreDigitCountFromSemicolon
  | LoadF
  | SubCharF
  | StoreDigitCountFromF
  | SaveResult
  | RestoreResult
  | SaveTerm
  | RestoreTerm
  | StoreResultFromTerm
  | StoreTermFromResult
  | StoreAccumFromResult
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
  ClearResult,
  LoadStar,
  SubCharStar,
  StoreDigitCountFromStar,
  LoadTerm,
  StoreTerm,
  StoreTermFromAccum,
  MulTermAccum,
  AddResultTerm,
  StoreResultAfterTermAdd,
  ClearTerm,
  MulResultRight,
  StoreResultAfterMul,
  ClearLeftVal,
  SetLeftVal,
  LoadMinus,
  SubCharMinus,
  StoreDigitCountFromMinus,
  SubResultTerm,
  StoreResultAfterSub,
  SetLeftValTwo,
  SubResultRight,
  StoreResultAfterSubRight,
  LoadLeftValForCheck,
  SubLeftValOne,
  StoreDigitCountFromLeftCheck,
  LoadZeroForNegate,
  LoadAccumForNegate,
  SubZeroAccum,
  StoreTermFromNegate,
  LoadL,
  SubCharL,
  StoreDigitCountFromL,
  LoadVarVal,
  StoreVarValFromAccum,
  AddVarValAccum,
  StoreResultFromVarValAdd,
  MulVarValAccum,
  StoreResultFromVarValMul,
  SubVarValAccum,
  StoreResultFromVarValSub,
  LoadSemicolon,
  SubCharSemicolon,
  StoreDigitCountFromSemicolon,
  LoadF,
  SubCharF,
  StoreDigitCountFromF,
  SaveResult,
  RestoreResult,
  SaveTerm,
  RestoreTerm,
  StoreResultFromTerm,
  StoreTermFromResult,
  StoreAccumFromResult
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
  | LoadStar => w .SimdLoadI32 L.CONST_STAR (r RB) ZERO SIZE_NONE
  | SubCharStar => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromStar => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | LoadTerm => w .SimdLoadI32 L.TERM (r RT) ZERO SIZE_NONE
  | StoreTerm => w .SimdStoreI32 (r RU) (r RA) L.TERM SIZE_NONE
  | StoreTermFromAccum => w .SimdStoreI32 (r RM) (r RA) L.TERM SIZE_NONE
  | MulTermAccum => w .SimdMulI32 (r RT) (r RU) (r RR) SIZE_NONE
  | AddResultTerm => w .SimdAddI32 (r RS) (r RU) (r RT) SIZE_NONE
  | StoreResultAfterTermAdd => w .SimdStoreI32 (r RU) (r RA) L.RESULT SIZE_NONE
  | ClearTerm => w .MemWrite ZERO L.TERM ZERO SIZE_I32
  | MulResultRight => w .SimdMulI32 (r RS) (r RU) (r RT) SIZE_NONE
  | StoreResultAfterMul => w .SimdStoreI32 (r RU) (r RA) L.RESULT SIZE_NONE
  | ClearLeftVal => w .MemWrite ZERO L.LEFT_VAL ZERO SIZE_I32
  | SetLeftVal => w .MemWrite (u32 1) L.LEFT_VAL ZERO SIZE_I32
  | LoadMinus => w .SimdLoadI32 L.CONST_MINUS (r RB) ZERO SIZE_NONE
  | SubCharMinus => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromMinus => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | SubResultTerm => w .SimdSubI32 (r RS) (r RU) (r RT) SIZE_NONE
  | StoreResultAfterSub => w .SimdStoreI32 (r RU) (r RA) L.RESULT SIZE_NONE
  | SetLeftValTwo => w .MemWrite (u32 2) L.LEFT_VAL ZERO SIZE_I32
  | SubResultRight => w .SimdSubI32 (r RT) (r RU) (r RS) SIZE_NONE  -- RU = RT - RS = RIGHT_VAL - RESULT = ACCUM - RESULT
  | StoreResultAfterSubRight => w .SimdStoreI32 (r RU) (r RA) L.RESULT SIZE_NONE
  | LoadLeftValForCheck => w .SimdLoadI32 L.LEFT_VAL (r RN) ZERO SIZE_NONE
  | SubLeftValOne => w .SimdSubI32 (r RN) (r RO) (r RK) SIZE_NONE
  | StoreDigitCountFromLeftCheck => w .SimdStoreI32 (r RO) (r RA) L.DIGIT_COUNT SIZE_NONE
  | LoadZeroForNegate => w .SimdLoadI32 L.CONST_ZERO (r RS) ZERO SIZE_NONE
  | LoadAccumForNegate => w .SimdLoadI32 L.ACCUM (r RT) ZERO SIZE_NONE
  | SubZeroAccum => w .SimdSubI32 (r RS) (r RU) (r RT) SIZE_NONE
  | StoreTermFromNegate => w .SimdStoreI32 (r RU) (r RA) L.TERM SIZE_NONE
  | LoadL => w .SimdLoadI32 L.CONST_L (r RB) ZERO SIZE_NONE
  | SubCharL => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromL => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | LoadVarVal => w .SimdLoadI32 L.VAR_VAL (r RS) ZERO SIZE_NONE
  | StoreVarValFromAccum => w .SimdStoreI32 (r RR) (r RA) L.VAR_VAL SIZE_NONE
  | AddVarValAccum => w .SimdAddI32 (r RS) (r RU) (r RR) SIZE_NONE  -- RU = VAR_VAL + ACCUM
  | StoreResultFromVarValAdd => w .SimdStoreI32 (r RU) (r RA) L.RESULT SIZE_NONE
  | MulVarValAccum => w .SimdMulI32 (r RS) (r RU) (r RR) SIZE_NONE  -- RU = VAR_VAL * ACCUM
  | StoreResultFromVarValMul => w .SimdStoreI32 (r RU) (r RA) L.RESULT SIZE_NONE
  | SubVarValAccum => w .SimdSubI32 (r RS) (r RU) (r RR) SIZE_NONE  -- RU = VAR_VAL - ACCUM
  | StoreResultFromVarValSub => w .SimdStoreI32 (r RU) (r RA) L.RESULT SIZE_NONE
  | LoadSemicolon => w .SimdLoadI32 L.CONST_SEMICOLON (r RB) ZERO SIZE_NONE
  | SubCharSemicolon => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromSemicolon => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | LoadF => w .SimdLoadI32 L.CONST_F (r RB) ZERO SIZE_NONE
  | SubCharF => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromF => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | SaveResult => w .MemCopy L.RESULT L.SAVED_RESULT ZERO SIZE_I32
  | RestoreResult => w .MemCopy L.SAVED_RESULT L.RESULT ZERO SIZE_I32
  | SaveTerm => w .MemCopy L.TERM L.SAVED_TERM ZERO SIZE_I32
  | RestoreTerm => w .MemCopy L.SAVED_TERM L.TERM ZERO SIZE_I32
  | StoreResultFromTerm => w .SimdStoreI32 (r RT) (r RA) L.RESULT SIZE_NONE
  | StoreTermFromResult => w .SimdStoreI32 (r RV) (r RA) L.TERM SIZE_NONE
  | StoreAccumFromResult => w .SimdStoreI32 (r RV) (r RA) L.ACCUM SIZE_NONE


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
  let p2 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N  -- newline check
  let p3 := JUMP_PAD_LEN
  let p4 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N  -- space check
  let p5 := JUMP_PAD_LEN
  let p6 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N  -- plus check
  let p7 := JUMP_PAD_LEN
  let p8 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N  -- star check
  let p9 := JUMP_PAD_LEN
  let p8b := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N -- minus check
  let p9b := JUMP_PAD_LEN
  let p8c := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N -- semicolon check
  let p9c := JUMP_PAD_LEN
  let p10 := SIMD_STEP_LEN * C.PARSE_DIGIT_SIMD_STEPS_N
  let p11 := INC_POS_LEN + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p8b + p9b + p8c + p9c + p10 + p11
def PARSE_ADDEND_LEN : Nat :=
  let p0 := LOAD_CHAR_LEN + SIMD_STEP_LEN
  let p1 := JUMP_PAD_LEN
  let p2 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N  -- rparen check
  let p3 := JUMP_PAD_LEN
  let p4 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N  -- newline check
  let p5 := JUMP_PAD_LEN
  let p6 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N  -- space check
  let p7 := JUMP_PAD_LEN
  let p8 := SIMD_STEP_LEN * C.SKIP_SPACES_SIMD_STEPS_N  -- star check
  let p9 := JUMP_PAD_LEN
  let p10 := SIMD_STEP_LEN * C.PARSE_DIGIT_SIMD_STEPS_N
  let p11 := INC_POS_LEN + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11
def ADD_TO_RESULT_LEN : Nat := SIMD_STEP_LEN * C.ADD_RESULT_SIMD_STEPS_N + MEM_STEP_LEN
def CHECK_PLUS_ONLY_LEN : Nat := LOAD_CHAR_LEN + SIMD_STEP_LEN * C.CHECK_PLUS_SIMD_STEPS_N + C.SINGLE_ACTION_N + INC_POS_LEN + C.SINGLE_ACTION_N
def CHECK_PLUS_LEN : Nat := SKIP_SPACES_LEN + CHECK_PLUS_ONLY_LEN
def ITOA_INIT_LEN : Nat := MEM_STEP_LEN * C.ITOA_INIT_MEM_STEPS_N
def ITOA_LOOP_LEN : Nat := SIMD_STEP_LEN * C.ITOA_LOOP_DIV_SIMD_STEPS_N + MEM_STEP_LEN + SIMD_STEP_LEN * C.ITOA_LOOP_POST_SIMD_STEPS_N + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
def OUTPUT_LEN : Nat := SIMD_STEP_LEN * C.OUTPUT_SIMD_STEPS_N + MEM_STEP_LEN + FILE_STEP_LEN
def SETUP_LEN : Nat := FFI_STEP_LEN + FILE_STEP_LEN + MEM_STEP_LEN + MEM_STEP_LEN + MEM_STEP_LEN
def LAMBDA_CHECK_LEN : Nat := LOAD_CHAR_LEN + SIMD_STEP_LEN * C.CHECK_PLUS_SIMD_STEPS_N + C.SINGLE_ACTION_N
def FIND_PLUS_LEN : Nat := LOAD_CHAR_LEN + SIMD_STEP_LEN * C.CHECK_PLUS_SIMD_STEPS_N + JUMP_PAD_LEN + INC_POS_LEN + C.SINGLE_ACTION_N
def FIND_OPERATOR_LEN : Nat :=
  LOAD_CHAR_LEN +
  SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N +  -- plus check
  MEM_STEP_LEN + C.SINGLE_ACTION_N +       -- plus found
  SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N +  -- star check
  MEM_STEP_LEN + C.SINGLE_ACTION_N +       -- star found
  SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N +  -- minus check
  MEM_STEP_LEN + C.SINGLE_ACTION_N +       -- minus found
  INC_POS_LEN + C.SINGLE_ACTION_N
def LAMBDA_PATH_LEN : Nat :=
  MEM_STEP_LEN +
  FIND_OPERATOR_LEN +
  INC_POS_LEN +
  SKIP_SPACES_LEN + PARSE_ADDEND_LEN +
  (SIMD_STEP_LEN * C.STORE_RESULT_SIMD_STEPS_N) + MEM_STEP_LEN +
  INC_POS_LEN + SKIP_SPACES_LEN + PARSE_NUMBER_LEN +
  C.SINGLE_ACTION_N +                              -- checkFlag1
  ADD_TO_RESULT_LEN + C.SINGLE_ACTION_N +          -- addPath
  SIMD_STEP_LEN * 4 + C.SINGLE_ACTION_N +          -- checkFlag2 (LoadOne, LoadLeftVal, Sub, Store, CondJump)
  ADD_TO_RESULT_LEN + C.SINGLE_ACTION_N +          -- mulPath
  ADD_TO_RESULT_LEN + C.SINGLE_ACTION_N            -- subPath
def SKIP_PARSE_LEN : Nat := SKIP_SPACES_LEN + PARSE_NUMBER_LEN
def TERM_FROM_ACCUM_LEN : Nat := SIMD_STEP_LEN * 2 + MEM_STEP_LEN
def MULTIPLY_TERM_LEN : Nat := SIMD_STEP_LEN * 4 + MEM_STEP_LEN
def ADD_TERM_TO_RESULT_LEN : Nat := SIMD_STEP_LEN * 4
def SUB_TERM_FROM_RESULT_LEN : Nat := SIMD_STEP_LEN * 4
def NEGATE_ACCUM_TO_TERM_LEN : Nat := SIMD_STEP_LEN * 4 + MEM_STEP_LEN
def CHECK_STAR_SIMD_STEPS_N : Nat := 3
def CHECK_STAR_LEN : Nat := SIMD_STEP_LEN * CHECK_STAR_SIMD_STEPS_N + C.SINGLE_ACTION_N
def CHECK_MINUS_LEN : Nat := SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N
def HANDLE_MULTIPLY_LEN : Nat := INC_POS_LEN + SKIP_SPACES_LEN + PARSE_NUMBER_LEN + MULTIPLY_TERM_LEN + C.SINGLE_ACTION_N
def HANDLE_ADD_LEN : Nat := ADD_TERM_TO_RESULT_LEN + INC_POS_LEN + SKIP_SPACES_LEN + PARSE_NUMBER_LEN + TERM_FROM_ACCUM_LEN + C.SINGLE_ACTION_N
def HANDLE_SUBTRACT_LEN : Nat := ADD_TERM_TO_RESULT_LEN + INC_POS_LEN + SKIP_SPACES_LEN + PARSE_NUMBER_LEN + NEGATE_ACCUM_TO_TERM_LEN + C.SINGLE_ACTION_N
def FINALIZE_LEN : Nat := ADD_TERM_TO_RESULT_LEN + C.SINGLE_ACTION_N
def CHECK_OPERATOR_LEN : Nat := SKIP_SPACES_LEN + LOAD_CHAR_LEN + SIMD_STEP_LEN * 6 + 4 + CHECK_MINUS_LEN + C.SINGLE_ACTION_N
def LET_CHECK_LEN : Nat := LOAD_CHAR_LEN + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N
def SKIP_LET_KEYWORD_LEN : Nat := INC_POS_LEN * 9  -- skip "let x := "
def STORE_ACCUM_TO_VAR_VAL_LEN : Nat := SIMD_STEP_LEN * 2 + MEM_STEP_LEN
def LET_COMPUTE_LEN : Nat :=
  C.SINGLE_ACTION_N +                              -- checkFlag1
  SIMD_STEP_LEN * 4 + C.SINGLE_ACTION_N +          -- addPath
  SIMD_STEP_LEN * 4 + C.SINGLE_ACTION_N +          -- checkFlag2
  SIMD_STEP_LEN * 4 + C.SINGLE_ACTION_N +          -- mulPath
  SIMD_STEP_LEN * 4 + C.SINGLE_ACTION_N            -- subPath
def LET_PATH_LEN : Nat :=
  SKIP_LET_KEYWORD_LEN +
  SKIP_SPACES_LEN + PARSE_NUMBER_LEN +             -- parse binding value
  STORE_ACCUM_TO_VAR_VAL_LEN +
  MEM_STEP_LEN +                                   -- clear LEFT_VAL
  FIND_OPERATOR_LEN +
  INC_POS_LEN +                                    -- skip operator
  SKIP_SPACES_LEN + PARSE_NUMBER_LEN +             -- parse operand
  LET_COMPUTE_LEN

def PAREN_CHECK_LEN : Nat :=
  LOAD_CHAR_LEN + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N +
  INC_POS_LEN +
  LOAD_CHAR_LEN + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N

def PARSE_GROUPED_NUMBER_LEN : Nat :=
  let p0 := LOAD_CHAR_LEN + SIMD_STEP_LEN
  let p1 := JUMP_PAD_LEN
  let p2 := SIMD_STEP_LEN * 3
  let p3 := JUMP_PAD_LEN
  let p4 := SIMD_STEP_LEN * 3
  let p5 := JUMP_PAD_LEN
  let p6 := SIMD_STEP_LEN * 3
  let p7 := JUMP_PAD_LEN
  let p8 := SIMD_STEP_LEN * 3
  let p9 := JUMP_PAD_LEN
  let p8b := SIMD_STEP_LEN * 3
  let p9b := JUMP_PAD_LEN
  let p8c := SIMD_STEP_LEN * 3
  let p9c := JUMP_PAD_LEN
  let p10 := SIMD_STEP_LEN * 7
  let p11 := INC_POS_LEN + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p8b + p9b + p8c + p9c + p10 + p11

def GROUPED_SKIP_PARSE_LEN : Nat := SKIP_SPACES_LEN + PARSE_GROUPED_NUMBER_LEN

def GROUPED_CHECK_OPERATOR_LEN : Nat :=
  SKIP_SPACES_LEN + LOAD_CHAR_LEN +
  SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N +
  SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N +
  SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N +
  SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N

def GROUPED_HANDLE_MULTIPLY_LEN : Nat := INC_POS_LEN + SKIP_SPACES_LEN + PARSE_GROUPED_NUMBER_LEN + MULTIPLY_TERM_LEN + C.SINGLE_ACTION_N
def GROUPED_HANDLE_ADD_LEN : Nat := ADD_TERM_TO_RESULT_LEN + INC_POS_LEN + SKIP_SPACES_LEN + PARSE_GROUPED_NUMBER_LEN + TERM_FROM_ACCUM_LEN + C.SINGLE_ACTION_N
def GROUPED_HANDLE_SUBTRACT_LEN : Nat := ADD_TERM_TO_RESULT_LEN + INC_POS_LEN + SKIP_SPACES_LEN + PARSE_GROUPED_NUMBER_LEN + NEGATE_ACCUM_TO_TERM_LEN + C.SINGLE_ACTION_N
def GROUPED_FINALIZE_LEN : Nat := ADD_TERM_TO_RESULT_LEN + SIMD_STEP_LEN * 2 + MEM_STEP_LEN + INC_POS_LEN + C.SINGLE_ACTION_N

def GROUPING_PATH_LEN : Nat :=
  GROUPED_SKIP_PARSE_LEN + TERM_FROM_ACCUM_LEN +
  GROUPED_CHECK_OPERATOR_LEN +
  GROUPED_HANDLE_MULTIPLY_LEN + GROUPED_HANDLE_ADD_LEN + GROUPED_HANDLE_SUBTRACT_LEN +
  GROUPED_FINALIZE_LEN

def PAREN_CHECK_BEFORE_PARSE_LEN : Nat := LOAD_CHAR_LEN + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
def MID_GROUPED_FINALIZE_LEN : Nat := ADD_TERM_TO_RESULT_LEN
def MID_GROUPED_EVAL_LEN : Nat :=
  GROUPED_SKIP_PARSE_LEN + TERM_FROM_ACCUM_LEN +
  GROUPED_CHECK_OPERATOR_LEN +
  GROUPED_HANDLE_MULTIPLY_LEN + GROUPED_HANDLE_ADD_LEN + GROUPED_HANDLE_SUBTRACT_LEN +
  MID_GROUPED_FINALIZE_LEN
def ACCUM_FROM_RESULT_LEN : Nat := SIMD_STEP_LEN * 2 + MEM_STEP_LEN

def HANDLE_MULTIPLY_NORMAL_PATH_LEN : Nat := PARSE_NUMBER_LEN + MULTIPLY_TERM_LEN + C.SINGLE_ACTION_N
def HANDLE_MULTIPLY_PAREN_PATH_LEN : Nat :=
  MEM_STEP_LEN + INC_POS_LEN + MID_GROUPED_EVAL_LEN + ACCUM_FROM_RESULT_LEN + INC_POS_LEN + MEM_STEP_LEN + MULTIPLY_TERM_LEN + C.SINGLE_ACTION_N
def NEW_HANDLE_MULTIPLY_LEN : Nat :=
  INC_POS_LEN + SKIP_SPACES_LEN + PAREN_CHECK_BEFORE_PARSE_LEN +
  HANDLE_MULTIPLY_NORMAL_PATH_LEN + HANDLE_MULTIPLY_PAREN_PATH_LEN

def HANDLE_ADD_NORMAL_PATH_LEN : Nat := PARSE_NUMBER_LEN + TERM_FROM_ACCUM_LEN + C.SINGLE_ACTION_N
def HANDLE_ADD_PAREN_PATH_LEN : Nat :=
  MEM_STEP_LEN + INC_POS_LEN + MID_GROUPED_EVAL_LEN + ACCUM_FROM_RESULT_LEN + INC_POS_LEN + MEM_STEP_LEN + TERM_FROM_ACCUM_LEN + C.SINGLE_ACTION_N
def NEW_HANDLE_ADD_LEN : Nat :=
  ADD_TERM_TO_RESULT_LEN + INC_POS_LEN + SKIP_SPACES_LEN + PAREN_CHECK_BEFORE_PARSE_LEN +
  HANDLE_ADD_NORMAL_PATH_LEN + HANDLE_ADD_PAREN_PATH_LEN

def HANDLE_SUBTRACT_NORMAL_PATH_LEN : Nat := PARSE_NUMBER_LEN + NEGATE_ACCUM_TO_TERM_LEN + C.SINGLE_ACTION_N
def HANDLE_SUBTRACT_PAREN_PATH_LEN : Nat :=
  MEM_STEP_LEN + INC_POS_LEN + MID_GROUPED_EVAL_LEN + ACCUM_FROM_RESULT_LEN + INC_POS_LEN + MEM_STEP_LEN + NEGATE_ACCUM_TO_TERM_LEN + C.SINGLE_ACTION_N
def NEW_HANDLE_SUBTRACT_LEN : Nat :=
  ADD_TERM_TO_RESULT_LEN + INC_POS_LEN + SKIP_SPACES_LEN + PAREN_CHECK_BEFORE_PARSE_LEN +
  HANDLE_SUBTRACT_NORMAL_PATH_LEN + HANDLE_SUBTRACT_PAREN_PATH_LEN

def MAIN_FLOW_LEN : Nat :=
  SETUP_LEN + PAREN_CHECK_LEN + LAMBDA_PATH_LEN + GROUPING_PATH_LEN + LET_CHECK_LEN + LET_PATH_LEN + SKIP_PARSE_LEN + TERM_FROM_ACCUM_LEN + CHECK_OPERATOR_LEN +
  NEW_HANDLE_MULTIPLY_LEN + NEW_HANDLE_ADD_LEN + NEW_HANDLE_SUBTRACT_LEN + FINALIZE_LEN + ITOA_INIT_LEN + ITOA_LOOP_LEN + OUTPUT_LEN

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
  let starCheck := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + JUMP_PAD_LEN)
  let p7 := jumpPad (jumpIfN L.DIGIT_COUNT starCheck) (jumpIfN L.CONST_ONE done)
  let p8 := simdStep LoadStar ++ simdStep SubCharStar ++ simdStep StoreDigitCountFromStar
  let minusCheck := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + p7.length + p8.length + JUMP_PAD_LEN)
  let p9 := jumpPad (jumpIfN L.DIGIT_COUNT minusCheck) (jumpIfN L.CONST_ONE done)
  let p8b := simdStep LoadMinus ++ simdStep SubCharMinus ++ simdStep StoreDigitCountFromMinus
  let semicolonCheck := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + p7.length + p8.length + p9.length + p8b.length + JUMP_PAD_LEN)
  let p9b := jumpPad (jumpIfN L.DIGIT_COUNT semicolonCheck) (jumpIfN L.CONST_ONE done)
  let p8c := simdStep LoadSemicolon ++ simdStep SubCharSemicolon ++ simdStep StoreDigitCountFromSemicolon
  let accumulate := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + p7.length + p8.length + p9.length + p8b.length + p9b.length + p8c.length + JUMP_PAD_LEN)
  let p9c := jumpPad (jumpIfN L.DIGIT_COUNT accumulate) (jumpIfN L.CONST_ONE done)
  let p10 := simdStep LoadAsciiZero ++ simdStep SubCharZero ++ simdStep LoadBase ++ simdStep LoadAccum ++ simdStep MulAccumBase ++ simdStep AddAccumDigit ++ simdStep StoreAccum
  let p11 := incPos ++ [jumpIfN L.CONST_ONE loopStart, fence]
  p0 ++ p1 ++ p2 ++ p3 ++ p4 ++ p5 ++ p6 ++ p7 ++ p8 ++ p9 ++ p8b ++ p9b ++ p8c ++ p9c ++ p10 ++ p11

def addToResultBlock : List Action :=
  simdStep LoadAccumForRight ++ simdStep StoreRightVal ++
  simdStep LoadResultForAdd ++ simdStep LoadRightVal ++ simdStep AddResultRight ++ simdStep StoreResultAfterAdd ++
  memStep ClearAccum

def mulResultBlock : List Action :=
  simdStep LoadAccumForRight ++ simdStep StoreRightVal ++
  simdStep LoadResultForAdd ++ simdStep LoadRightVal ++ simdStep MulResultRight ++ simdStep StoreResultAfterMul ++
  memStep ClearAccum

def termFromAccumBlock : List Action :=
  simdStep LoadAccumForResult ++ simdStep StoreTermFromAccum ++
  memStep ClearAccum

def multiplyTermBlock : List Action :=
  simdStep LoadTerm ++ simdStep LoadAccumForRight ++
  simdStep MulTermAccum ++ simdStep StoreTerm ++
  memStep ClearAccum

def addTermToResultBlock : List Action :=
  simdStep LoadResultForAdd ++ simdStep LoadTerm ++
  simdStep AddResultTerm ++ simdStep StoreResultAfterTermAdd

def subTermFromResultBlock : List Action :=
  simdStep LoadResultForAdd ++ simdStep LoadTerm ++
  simdStep SubResultTerm ++ simdStep StoreResultAfterSub

def negateAccumToTermBlock : List Action :=
  simdStep LoadZeroForNegate ++ simdStep LoadAccumForNegate ++
  simdStep SubZeroAccum ++ simdStep StoreTermFromNegate ++
  memStep ClearAccum

def subResultBlock : List Action :=
  simdStep LoadAccumForRight ++ simdStep StoreRightVal ++
  simdStep LoadResultForAdd ++ simdStep LoadRightVal ++ simdStep SubResultRight ++ simdStep StoreResultAfterSubRight ++
  memStep ClearAccum

def findPlusBlock (loopStart done : Nat) : List Action :=
  let continueStart := loopStart + (LOAD_CHAR_LEN + SIMD_STEP_LEN * C.CHECK_PLUS_SIMD_STEPS_N + JUMP_PAD_LEN)
  loadChar ++
  simdStep LoadPlus ++ simdStep SubCharPlus ++ simdStep StoreDigitCountFromPlus ++
  jumpPad (jumpIfN L.DIGIT_COUNT continueStart) (jumpIfN L.CONST_ONE done) ++
  incPos ++
  [jumpIfN L.CONST_ONE loopStart]

-- Find '+', '*', or '-', set LEFT_VAL flag (0 for +, 1 for *, 2 for -)
def findOperatorBlock (loopStart done : Nat) : List Action :=
  let loadCharEnd := loopStart + LOAD_CHAR_LEN
  let plusCheckEnd := loadCharEnd + SIMD_STEP_LEN * 3
  let plusFoundEnd := plusCheckEnd + C.SINGLE_ACTION_N + MEM_STEP_LEN + C.SINGLE_ACTION_N
  let starCheckEnd := plusFoundEnd + SIMD_STEP_LEN * 3
  let starFoundEnd := starCheckEnd + C.SINGLE_ACTION_N + MEM_STEP_LEN + C.SINGLE_ACTION_N
  let minusCheckEnd := starFoundEnd + SIMD_STEP_LEN * 3
  let minusFoundEnd := minusCheckEnd + C.SINGLE_ACTION_N + MEM_STEP_LEN + C.SINGLE_ACTION_N

  loadChar ++
  -- Check for '+'
  simdStep LoadPlus ++ simdStep SubCharPlus ++ simdStep StoreDigitCountFromPlus ++
  [jumpIfN L.DIGIT_COUNT plusFoundEnd] ++  -- if not '+', skip to star check
  -- '+' found: clear flag (0), jump to done
  memStep ClearLeftVal ++
  [jumpIfN L.CONST_ONE done] ++
  -- Check for '*'
  simdStep LoadStar ++ simdStep SubCharStar ++ simdStep StoreDigitCountFromStar ++
  [jumpIfN L.DIGIT_COUNT starFoundEnd] ++  -- if not '*', skip to minus check
  -- '*' found: set flag (1), jump to done
  memStep SetLeftVal ++
  [jumpIfN L.CONST_ONE done] ++
  -- Check for '-'
  simdStep LoadMinus ++ simdStep SubCharMinus ++ simdStep StoreDigitCountFromMinus ++
  [jumpIfN L.DIGIT_COUNT minusFoundEnd] ++  -- if not '-', skip to incPos
  -- '-' found: set flag (2), jump to done
  memStep SetLeftValTwo ++
  [jumpIfN L.CONST_ONE done] ++
  -- Neither found: increment position and loop
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
  let starCheck := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + JUMP_PAD_LEN)
  let p7 := jumpPad (jumpIfN L.DIGIT_COUNT starCheck) (jumpIfN L.CONST_ONE done)
  let p8 := simdStep LoadStar ++ simdStep SubCharStar ++ simdStep StoreDigitCountFromStar
  let accumulate := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + p7.length + p8.length + JUMP_PAD_LEN)
  let p9 := jumpPad (jumpIfN L.DIGIT_COUNT accumulate) (jumpIfN L.CONST_ONE done)
  let p10 := simdStep LoadAsciiZero ++ simdStep SubCharZero ++ simdStep LoadBase ++ simdStep LoadAccum ++ simdStep MulAccumBase ++ simdStep AddAccumDigit ++ simdStep StoreAccum
  let p11 := incPos ++ [jumpIfN L.CONST_ONE loopStart, fence]
  p0 ++ p1 ++ p2 ++ p3 ++ p4 ++ p5 ++ p6 ++ p7 ++ p8 ++ p9 ++ p10 ++ p11

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

def parenCheckBlock (lambdaStart groupingStart normalStart : Nat) : List Action :=
  loadChar ++
  simdStep LoadLParen ++ simdStep SubCharLParen ++ simdStep StoreDigitCountFromLParen ++
  [jumpIfN L.DIGIT_COUNT normalStart] ++
  incPos ++
  loadChar ++
  simdStep LoadF ++ simdStep SubCharF ++ simdStep StoreDigitCountFromF ++
  [jumpIfN L.DIGIT_COUNT groupingStart] ++
  [jumpIfN L.CONST_ONE lambdaStart]

def lambdaPathBlock (loopStart outputStart : Nat) : List Action :=
  let clearFlag := memStep ClearLeftVal
  let findOpStart := loopStart + clearFlag.length
  let findOpDone := findOpStart + FIND_OPERATOR_LEN
  let findOp := findOperatorBlock findOpStart findOpDone
  let skipOp := incPos
  let skipToAddendStart := findOpDone + skipOp.length
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
  -- Check flag: 0=add, 1=mul, 2=sub
  -- checkFlag1: if LEFT_VAL != 0, skip addPath
  -- checkFlag2: load LEFT_VAL, subtract 1, if != 0 skip mulPath (go to subPath)
  let checkFlag1Len := C.SINGLE_ACTION_N
  let addPathLen := ADD_TO_RESULT_LEN + C.SINGLE_ACTION_N
  let checkFlag2Len := SIMD_STEP_LEN * 4 + C.SINGLE_ACTION_N  -- LoadOne, LoadLeftVal, Sub, Store, CondJump
  let mulPathLen := ADD_TO_RESULT_LEN + C.SINGLE_ACTION_N
  let notAddStart := argDone + checkFlag1Len + addPathLen
  let subStart := notAddStart + checkFlag2Len + mulPathLen
  let checkFlag1 := [jumpIfN L.LEFT_VAL notAddStart]
  let addPath := addToResultBlock ++ [jumpIfN L.CONST_ONE outputStart]
  -- At notAddStart: check if (LEFT_VAL - 1) != 0 to distinguish mul (1) from sub (2)
  let checkFlag2 := simdStep LoadOne ++ simdStep LoadLeftValForCheck ++ simdStep SubLeftValOne ++ simdStep StoreDigitCountFromLeftCheck ++ [jumpIfN L.DIGIT_COUNT subStart]
  let mulPath := mulResultBlock ++ [jumpIfN L.CONST_ONE outputStart]
  let subPath := subResultBlock ++ [jumpIfN L.CONST_ONE outputStart]
  clearFlag ++ findOp ++ skipOp ++ skipToAddend ++ parseAddend ++ storeAddend ++ clearAccum ++ skipRParen ++ skipToArg ++ parseArg ++ checkFlag1 ++ addPath ++ checkFlag2 ++ mulPath ++ subPath

def letCheckBlock (normalStart : Nat) : List Action :=
  loadChar ++
  simdStep LoadL ++ simdStep SubCharL ++ simdStep StoreDigitCountFromL ++
  [jumpIfN L.DIGIT_COUNT normalStart]

def storeAccumToVarValBlock : List Action :=
  simdStep LoadAccumForRight ++ simdStep StoreVarValFromAccum ++
  memStep ClearAccum

def letPathBlock (loopStart outputStart : Nat) : List Action :=
  -- Skip "let x := " (9 chars)
  let skipKeyword := incPos ++ incPos ++ incPos ++ incPos ++ incPos ++ incPos ++ incPos ++ incPos ++ incPos
  let skipKeywordDone := loopStart + SKIP_LET_KEYWORD_LEN

  -- Skip spaces and parse binding value
  let parseBindingStart := skipKeywordDone + SKIP_SPACES_LEN
  let parseBindingDone := parseBindingStart + PARSE_NUMBER_LEN
  let skipToBinding := skipSpacesBlock skipKeywordDone parseBindingStart
  let parseBinding := parseNumberBlock parseBindingStart parseBindingDone

  -- Store ACCUM to VAR_VAL and clear ACCUM
  let storeVarVal := storeAccumToVarValBlock
  let storeVarValDone := parseBindingDone + STORE_ACCUM_TO_VAR_VAL_LEN

  -- Clear LEFT_VAL flag
  let clearFlag := memStep ClearLeftVal
  let clearFlagDone := storeVarValDone + MEM_STEP_LEN

  -- Find operator (will loop until it finds +, *, or - and set LEFT_VAL)
  let findOpStart := clearFlagDone
  let findOpDone := findOpStart + FIND_OPERATOR_LEN
  let findOp := findOperatorBlock findOpStart findOpDone

  -- Skip operator
  let skipOp := incPos
  let skipOpDone := findOpDone + INC_POS_LEN

  -- Skip spaces and parse operand
  let parseOperandStart := skipOpDone + SKIP_SPACES_LEN
  let parseOperandDone := parseOperandStart + PARSE_NUMBER_LEN
  let skipToOperand := skipSpacesBlock skipOpDone parseOperandStart
  let parseOperand := parseNumberBlock parseOperandStart parseOperandDone

  -- Compute result based on LEFT_VAL flag
  -- LEFT_VAL = 0: add, LEFT_VAL = 1: mul, LEFT_VAL = 2: sub
  let computeStart := parseOperandDone
  let addPathLen := SIMD_STEP_LEN * 4 + C.SINGLE_ACTION_N
  let checkFlag2Len := SIMD_STEP_LEN * 4 + C.SINGLE_ACTION_N
  let mulPathLen := SIMD_STEP_LEN * 4 + C.SINGLE_ACTION_N
  let notAddStart := computeStart + C.SINGLE_ACTION_N + addPathLen
  let subStart := notAddStart + checkFlag2Len + mulPathLen
  let checkFlag1 := [jumpIfN L.LEFT_VAL notAddStart]
  let addPath :=
    simdStep LoadVarVal ++ simdStep LoadAccumForRight ++
    simdStep AddVarValAccum ++ simdStep StoreResultFromVarValAdd ++
    [jumpIfN L.CONST_ONE outputStart]
  let checkFlag2 :=
    simdStep LoadOne ++ simdStep LoadLeftValForCheck ++
    simdStep SubLeftValOne ++ simdStep StoreDigitCountFromLeftCheck ++
    [jumpIfN L.DIGIT_COUNT subStart]
  let mulPath :=
    simdStep LoadVarVal ++ simdStep LoadAccumForRight ++
    simdStep MulVarValAccum ++ simdStep StoreResultFromVarValMul ++
    [jumpIfN L.CONST_ONE outputStart]
  let subPath :=
    simdStep LoadVarVal ++ simdStep LoadAccumForRight ++
    simdStep SubVarValAccum ++ simdStep StoreResultFromVarValSub ++
    [jumpIfN L.CONST_ONE outputStart]

  skipKeyword ++ skipToBinding ++ parseBinding ++ storeVarVal ++ clearFlag ++
  findOp ++ skipOp ++ skipToOperand ++ parseOperand ++
  checkFlag1 ++ addPath ++ checkFlag2 ++ mulPath ++ subPath

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

def parseGroupedNumberBlock (loopStart done : Nat) : List Action :=
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
  let starCheck := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + JUMP_PAD_LEN)
  let p7 := jumpPad (jumpIfN L.DIGIT_COUNT starCheck) (jumpIfN L.CONST_ONE done)
  let p8 := simdStep LoadStar ++ simdStep SubCharStar ++ simdStep StoreDigitCountFromStar
  let minusCheck := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + p7.length + p8.length + JUMP_PAD_LEN)
  let p9 := jumpPad (jumpIfN L.DIGIT_COUNT minusCheck) (jumpIfN L.CONST_ONE done)
  let p8b := simdStep LoadMinus ++ simdStep SubCharMinus ++ simdStep StoreDigitCountFromMinus
  let rparenCheck := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + p7.length + p8.length + p9.length + p8b.length + JUMP_PAD_LEN)
  let p9b := jumpPad (jumpIfN L.DIGIT_COUNT rparenCheck) (jumpIfN L.CONST_ONE done)
  let p8c := simdStep LoadRParen ++ simdStep SubCharRParen ++ simdStep StoreDigitCountFromRParen
  let accumulate := loopStart + (p0.length + p1.length + p2.length + p3.length + p4.length + p5.length + p6.length + p7.length + p8.length + p9.length + p8b.length + p9b.length + p8c.length + JUMP_PAD_LEN)
  let p9c := jumpPad (jumpIfN L.DIGIT_COUNT accumulate) (jumpIfN L.CONST_ONE done)
  let p10 := simdStep LoadAsciiZero ++ simdStep SubCharZero ++ simdStep LoadBase ++ simdStep LoadAccum ++ simdStep MulAccumBase ++ simdStep AddAccumDigit ++ simdStep StoreAccum
  let p11 := incPos ++ [jumpIfN L.CONST_ONE loopStart, fence]
  p0 ++ p1 ++ p2 ++ p3 ++ p4 ++ p5 ++ p6 ++ p7 ++ p8 ++ p9 ++ p8b ++ p9b ++ p8c ++ p9c ++ p10 ++ p11

def groupedSkipParseBlock (start done : Nat) : List Action :=
  let parseStart := start + SKIP_SPACES_LEN
  skipSpacesBlock start parseStart ++ parseGroupedNumberBlock parseStart done

def accumFromResultBlock : List Action :=
  simdStep LoadResultForItoa ++ simdStep StoreAccumFromResult ++ memStep ClearResult

def midGroupedEvalBlock (start : Nat) : List Action :=
  let skipParseDone := start + GROUPED_SKIP_PARSE_LEN
  let termFromAccumDone := skipParseDone + TERM_FROM_ACCUM_LEN
  let checkOpDone := termFromAccumDone + GROUPED_CHECK_OPERATOR_LEN
  let handleMulStart := checkOpDone
  let handleMulDone := handleMulStart + GROUPED_HANDLE_MULTIPLY_LEN
  let handleAddStart := handleMulDone
  let handleAddDone := handleAddStart + GROUPED_HANDLE_ADD_LEN
  let handleSubStart := handleAddDone
  let handleSubDone := handleSubStart + GROUPED_HANDLE_SUBTRACT_LEN
  let finalizeStart := handleSubDone
  let skipParse := groupedSkipParseBlock start skipParseDone
  let termFromAccum := termFromAccumBlock
  let checkOpSkipEnd := termFromAccumDone + SKIP_SPACES_LEN
  let rparenStart := checkOpSkipEnd + LOAD_CHAR_LEN
  let starStart := rparenStart + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  let plusStart := starStart + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  let minusStart := plusStart + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  let checkOp :=
    skipSpacesBlock termFromAccumDone checkOpSkipEnd ++
    loadChar ++
    simdStep LoadRParen ++ simdStep SubCharRParen ++ simdStep StoreDigitCountFromRParen ++
    [jumpIfN L.DIGIT_COUNT starStart] ++ [jumpIfN L.CONST_ONE finalizeStart] ++
    simdStep LoadStar ++ simdStep SubCharStar ++ simdStep StoreDigitCountFromStar ++
    [jumpIfN L.DIGIT_COUNT plusStart] ++ [jumpIfN L.CONST_ONE handleMulStart] ++
    simdStep LoadPlus ++ simdStep SubCharPlus ++ simdStep StoreDigitCountFromPlus ++
    [jumpIfN L.DIGIT_COUNT minusStart] ++ [jumpIfN L.CONST_ONE handleAddStart] ++
    simdStep LoadMinus ++ simdStep SubCharMinus ++ simdStep StoreDigitCountFromMinus ++
    [jumpIfN L.DIGIT_COUNT finalizeStart] ++ [jumpIfN L.CONST_ONE handleSubStart]
  let handleMulSkipStart := handleMulStart + INC_POS_LEN
  let handleMulParseDone := handleMulSkipStart + SKIP_SPACES_LEN + PARSE_GROUPED_NUMBER_LEN
  let handleMul :=
    incPos ++
    groupedSkipParseBlock handleMulSkipStart handleMulParseDone ++
    multiplyTermBlock ++
    [jumpIfN L.CONST_ONE termFromAccumDone]
  let handleAddIncStart := handleAddStart + ADD_TERM_TO_RESULT_LEN
  let handleAddSkipStart := handleAddIncStart + INC_POS_LEN
  let handleAddParseDone := handleAddSkipStart + SKIP_SPACES_LEN + PARSE_GROUPED_NUMBER_LEN
  let handleAdd :=
    addTermToResultBlock ++
    incPos ++
    groupedSkipParseBlock handleAddSkipStart handleAddParseDone ++
    termFromAccumBlock ++
    [jumpIfN L.CONST_ONE termFromAccumDone]
  let handleSubIncStart := handleSubStart + ADD_TERM_TO_RESULT_LEN
  let handleSubSkipStart := handleSubIncStart + INC_POS_LEN
  let handleSubParseDone := handleSubSkipStart + SKIP_SPACES_LEN + PARSE_GROUPED_NUMBER_LEN
  let handleSub :=
    addTermToResultBlock ++
    incPos ++
    groupedSkipParseBlock handleSubSkipStart handleSubParseDone ++
    negateAccumToTermBlock ++
    [jumpIfN L.CONST_ONE termFromAccumDone]
  let finalize := addTermToResultBlock
  skipParse ++ termFromAccum ++ checkOp ++ handleMul ++ handleAdd ++ handleSub ++ finalize

def groupingPathBlock (start checkOpStart : Nat) : List Action :=
  let skipParseDone := start + GROUPED_SKIP_PARSE_LEN
  let termFromAccumDone := skipParseDone + TERM_FROM_ACCUM_LEN
  let checkOpDone := termFromAccumDone + GROUPED_CHECK_OPERATOR_LEN
  let handleMulStart := checkOpDone
  let handleMulDone := handleMulStart + GROUPED_HANDLE_MULTIPLY_LEN
  let handleAddStart := handleMulDone
  let handleAddDone := handleAddStart + GROUPED_HANDLE_ADD_LEN
  let handleSubStart := handleAddDone
  let handleSubDone := handleSubStart + GROUPED_HANDLE_SUBTRACT_LEN
  let finalizeStart := handleSubDone
  let skipParse := groupedSkipParseBlock start skipParseDone
  let termFromAccum := termFromAccumBlock
  let checkOpSkipEnd := termFromAccumDone + SKIP_SPACES_LEN
  let rparenStart := checkOpSkipEnd + LOAD_CHAR_LEN
  let starStart := rparenStart + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  let plusStart := starStart + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  let minusStart := plusStart + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  let checkOp :=
    skipSpacesBlock termFromAccumDone checkOpSkipEnd ++
    loadChar ++
    simdStep LoadRParen ++ simdStep SubCharRParen ++ simdStep StoreDigitCountFromRParen ++
    [jumpIfN L.DIGIT_COUNT starStart] ++ [jumpIfN L.CONST_ONE finalizeStart] ++
    simdStep LoadStar ++ simdStep SubCharStar ++ simdStep StoreDigitCountFromStar ++
    [jumpIfN L.DIGIT_COUNT plusStart] ++ [jumpIfN L.CONST_ONE handleMulStart] ++
    simdStep LoadPlus ++ simdStep SubCharPlus ++ simdStep StoreDigitCountFromPlus ++
    [jumpIfN L.DIGIT_COUNT minusStart] ++ [jumpIfN L.CONST_ONE handleAddStart] ++
    simdStep LoadMinus ++ simdStep SubCharMinus ++ simdStep StoreDigitCountFromMinus ++
    [jumpIfN L.DIGIT_COUNT finalizeStart] ++ [jumpIfN L.CONST_ONE handleSubStart]
  let handleMulSkipStart := handleMulStart + INC_POS_LEN
  let handleMulParseDone := handleMulSkipStart + SKIP_SPACES_LEN + PARSE_GROUPED_NUMBER_LEN
  let handleMul :=
    incPos ++
    groupedSkipParseBlock handleMulSkipStart handleMulParseDone ++
    multiplyTermBlock ++
    [jumpIfN L.CONST_ONE termFromAccumDone]
  let handleAddIncStart := handleAddStart + ADD_TERM_TO_RESULT_LEN
  let handleAddSkipStart := handleAddIncStart + INC_POS_LEN
  let handleAddParseDone := handleAddSkipStart + SKIP_SPACES_LEN + PARSE_GROUPED_NUMBER_LEN
  let handleAdd :=
    addTermToResultBlock ++
    incPos ++
    groupedSkipParseBlock handleAddSkipStart handleAddParseDone ++
    termFromAccumBlock ++
    [jumpIfN L.CONST_ONE termFromAccumDone]
  let handleSubIncStart := handleSubStart + ADD_TERM_TO_RESULT_LEN
  let handleSubSkipStart := handleSubIncStart + INC_POS_LEN
  let handleSubParseDone := handleSubSkipStart + SKIP_SPACES_LEN + PARSE_GROUPED_NUMBER_LEN
  let handleSub :=
    addTermToResultBlock ++
    incPos ++
    groupedSkipParseBlock handleSubSkipStart handleSubParseDone ++
    negateAccumToTermBlock ++
    [jumpIfN L.CONST_ONE termFromAccumDone]
  let finalize :=
    addTermToResultBlock ++
    simdStep LoadResultForItoa ++ simdStep StoreTermFromResult ++
    memStep ClearResult ++
    incPos ++
    [jumpIfN L.CONST_ONE checkOpStart]
  skipParse ++ termFromAccum ++ checkOp ++ handleMul ++ handleAdd ++ handleSub ++ finalize

def parenCheckBeforeParseBlock (normalStart parenStart : Nat) : List Action :=
  loadChar ++
  simdStep LoadLParen ++ simdStep SubCharLParen ++ simdStep StoreDigitCountFromLParen ++
  [jumpIfN L.DIGIT_COUNT normalStart] ++ [jumpIfN L.CONST_ONE parenStart]

def actualMainFlow : List Action :=
  let setup := ffiStep FfiCall ++ fileStep FileRead ++ memStep ClearAccum ++ memStep ClearResult ++ memStep ClearTerm

  let parenCheckStart := SETUP_LEN
  let lambdaPathStart := parenCheckStart + PAREN_CHECK_LEN
  let groupingPathStart := lambdaPathStart + LAMBDA_PATH_LEN
  let letCheckStart := groupingPathStart + GROUPING_PATH_LEN
  let letPathStart := letCheckStart + LET_CHECK_LEN
  let normalStart := letPathStart + LET_PATH_LEN

  let skipParseStart := normalStart
  let termFromAccumStart := skipParseStart + SKIP_PARSE_LEN
  let checkOperatorStart := termFromAccumStart + TERM_FROM_ACCUM_LEN
  let checkOperatorSkipEnd := checkOperatorStart + SKIP_SPACES_LEN
  let checkOperatorLoadCharEnd := checkOperatorSkipEnd + LOAD_CHAR_LEN
  let checkOperatorStarEnd := checkOperatorLoadCharEnd + SIMD_STEP_LEN * 3

  let handleMultiplyStart := checkOperatorStart + CHECK_OPERATOR_LEN
  let handleAddStart := handleMultiplyStart + NEW_HANDLE_MULTIPLY_LEN
  let handleSubtractStart := handleAddStart + NEW_HANDLE_ADD_LEN
  let finalizeStart := handleSubtractStart + NEW_HANDLE_SUBTRACT_LEN
  let outputStart := finalizeStart + FINALIZE_LEN
  let itoaStart := outputStart
  let itoaLoopStart := itoaStart + ITOA_INIT_LEN

  let parenCheck := parenCheckBlock lambdaPathStart groupingPathStart letCheckStart
  let lambdaPath := lambdaPathBlock lambdaPathStart outputStart
  let groupingPath := groupingPathBlock groupingPathStart checkOperatorStart
  let letCheck := letCheckBlock normalStart
  let letPath := letPathBlock letPathStart outputStart
  let skipParse := skipParseBlock skipParseStart termFromAccumStart
  let termFromAccum := termFromAccumBlock

  let plusCheckStart := checkOperatorStarEnd + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  let minusCheckStart := plusCheckStart + SIMD_STEP_LEN * 3 + C.SINGLE_ACTION_N + C.SINGLE_ACTION_N
  let checkOperator :=
    skipSpacesBlock checkOperatorStart checkOperatorSkipEnd ++
    loadChar ++
    simdStep LoadStar ++ simdStep SubCharStar ++ simdStep StoreDigitCountFromStar ++
    [jumpIfN L.DIGIT_COUNT plusCheckStart] ++ [jumpIfN L.CONST_ONE handleMultiplyStart] ++
    simdStep LoadPlus ++ simdStep SubCharPlus ++ simdStep StoreDigitCountFromPlus ++
    [jumpIfN L.DIGIT_COUNT minusCheckStart] ++ [jumpIfN L.CONST_ONE handleAddStart] ++
    simdStep LoadMinus ++ simdStep SubCharMinus ++ simdStep StoreDigitCountFromMinus ++
    [jumpIfN L.DIGIT_COUNT finalizeStart] ++ [jumpIfN L.CONST_ONE handleSubtractStart]

  let handleMulIncStart := handleMultiplyStart + INC_POS_LEN
  let handleMulSkipStart := handleMulIncStart + SKIP_SPACES_LEN
  let handleMulCheckParenStart := handleMulSkipStart + PAREN_CHECK_BEFORE_PARSE_LEN
  let handleMulNormalPathStart := handleMulCheckParenStart
  let handleMulNormalParseDone := handleMulNormalPathStart + PARSE_NUMBER_LEN
  let handleMulParenPathStart := handleMulCheckParenStart + HANDLE_MULTIPLY_NORMAL_PATH_LEN
  let handleMulMidGroupedStart := handleMulParenPathStart + MEM_STEP_LEN + INC_POS_LEN
  let handleMultiply :=
    incPos ++
    skipSpacesBlock handleMulIncStart handleMulSkipStart ++
    parenCheckBeforeParseBlock handleMulNormalPathStart handleMulParenPathStart ++
    parseNumberBlock handleMulNormalPathStart handleMulNormalParseDone ++
    multiplyTermBlock ++
    [jumpIfN L.CONST_ONE checkOperatorStart] ++
    memStep SaveTerm ++
    incPos ++
    midGroupedEvalBlock handleMulMidGroupedStart ++
    accumFromResultBlock ++
    incPos ++
    memStep RestoreTerm ++
    multiplyTermBlock ++
    [jumpIfN L.CONST_ONE checkOperatorStart]

  let handleAddIncStart := handleAddStart + ADD_TERM_TO_RESULT_LEN + INC_POS_LEN
  let handleAddSkipStart := handleAddIncStart + SKIP_SPACES_LEN
  let handleAddCheckParenStart := handleAddSkipStart + PAREN_CHECK_BEFORE_PARSE_LEN
  let handleAddNormalPathStart := handleAddCheckParenStart
  let handleAddNormalParseDone := handleAddNormalPathStart + PARSE_NUMBER_LEN
  let handleAddParenPathStart := handleAddCheckParenStart + HANDLE_ADD_NORMAL_PATH_LEN
  let handleAddMidGroupedStart := handleAddParenPathStart + MEM_STEP_LEN + INC_POS_LEN
  let handleAdd :=
    addTermToResultBlock ++
    incPos ++
    skipSpacesBlock (handleAddStart + ADD_TERM_TO_RESULT_LEN + INC_POS_LEN) handleAddSkipStart ++
    parenCheckBeforeParseBlock handleAddNormalPathStart handleAddParenPathStart ++
    parseNumberBlock handleAddNormalPathStart handleAddNormalParseDone ++
    termFromAccumBlock ++
    [jumpIfN L.CONST_ONE checkOperatorStart] ++
    memStep SaveTerm ++
    incPos ++
    midGroupedEvalBlock handleAddMidGroupedStart ++
    accumFromResultBlock ++
    incPos ++
    memStep RestoreTerm ++
    termFromAccumBlock ++
    [jumpIfN L.CONST_ONE checkOperatorStart]

  let handleSubIncStart := handleSubtractStart + ADD_TERM_TO_RESULT_LEN + INC_POS_LEN
  let handleSubSkipStart := handleSubIncStart + SKIP_SPACES_LEN
  let handleSubCheckParenStart := handleSubSkipStart + PAREN_CHECK_BEFORE_PARSE_LEN
  let handleSubNormalPathStart := handleSubCheckParenStart
  let handleSubNormalParseDone := handleSubNormalPathStart + PARSE_NUMBER_LEN
  let handleSubParenPathStart := handleSubCheckParenStart + HANDLE_SUBTRACT_NORMAL_PATH_LEN
  let handleSubMidGroupedStart := handleSubParenPathStart + MEM_STEP_LEN + INC_POS_LEN
  let handleSubtract :=
    addTermToResultBlock ++
    incPos ++
    skipSpacesBlock (handleSubtractStart + ADD_TERM_TO_RESULT_LEN + INC_POS_LEN) handleSubSkipStart ++
    parenCheckBeforeParseBlock handleSubNormalPathStart handleSubParenPathStart ++
    parseNumberBlock handleSubNormalPathStart handleSubNormalParseDone ++
    negateAccumToTermBlock ++
    [jumpIfN L.CONST_ONE checkOperatorStart] ++
    memStep SaveTerm ++
    incPos ++
    midGroupedEvalBlock handleSubMidGroupedStart ++
    accumFromResultBlock ++
    incPos ++
    memStep RestoreTerm ++
    negateAccumToTermBlock ++
    [jumpIfN L.CONST_ONE checkOperatorStart]

  let finalize := addTermToResultBlock ++ [jumpIfN L.CONST_ONE itoaStart]
  let itoaInit := itoaInitBlock
  let itoaLoop := itoaLoopBlock itoaLoopStart
  let output := outputBlock

  setup ++
  parenCheck ++
  lambdaPath ++
  groupingPath ++
  letCheck ++
  letPath ++
  skipParse ++
  termFromAccum ++
  checkOperator ++
  handleMultiply ++
  handleAdd ++
  handleSubtract ++
  finalize ++
  itoaInit ++
  itoaLoop ++
  output

def leanEvalAlgorithm : Algorithm := {
  actions := actualMainFlow ++ workActions,
  payloads := leanEvalPayloads,
  state := {
    regs_per_unit := C.REGS_PER_UNIT_N,
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
