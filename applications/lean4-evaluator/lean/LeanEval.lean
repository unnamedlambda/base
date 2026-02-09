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
def ASCII_LT_I : Int := 60
def ASCII_GT_I : Int := 62

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
def OFFSET_FLAG_HASH_U32 : UInt32 := 0x05A0
def OFFSET_HT_HANDLE_U32 : UInt32 := 0x05B0
def OFFSET_IDENTIFIER_BUF_U32 : UInt32 := 0x05C0
def OFFSET_HT_VAL_BUF_U32 : UInt32 := 0x05E0
def OFFSET_HT_RESULT_BUF_U32 : UInt32 := 0x05F0
def OFFSET_HT_RESULT_VAL_U32 : UInt32 := 0x05F4
def OFFSET_IDENT_WRITE_PTR_U32 : UInt32 := 0x0600
def OFFSET_SAVED_IDENTIFIER_BUF_U32 : UInt32 := 0x0610
def OFFSET_CONST_LT_U32 : UInt32 := 0x0630
def OFFSET_BOOL_FLAG_U32 : UInt32 := 0x0640
def OFFSET_TRUE_STRING_U32 : UInt32 := 0x0650
def OFFSET_FALSE_STRING_U32 : UInt32 := 0x0660
def OFFSET_CONST_GT_U32 : UInt32 := 0x0670
def OFFSET_IF_FLAG_U32 : UInt32 := 0x0680
def OFFSET_SAVED_CONDITION_U32 : UInt32 := 0x0690
def OFFSET_IF_RESULT_A_U32 : UInt32 := 0x06A0
def OFFSET_IF_RESULT_B_U32 : UInt32 := 0x06B0

def IDENT_BUF_SIZE_N : Nat := 32
def HASH_TABLE_UNIT_ID_N : Nat := 7
def HASH_TABLE_UNIT_COUNT_N : Nat := 1
def HASH_STEP_ACTIONS_N : Nat := 2

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
  FLAG_HASH : UInt32
  HT_HANDLE : UInt32
  IDENTIFIER_BUF : UInt32
  HT_VAL_BUF : UInt32
  HT_RESULT_BUF : UInt32
  HT_RESULT_VAL : UInt32
  IDENT_WRITE_PTR : UInt32
  SAVED_IDENTIFIER_BUF : UInt32
  CONST_LT : UInt32
  BOOL_FLAG : UInt32
  TRUE_STRING : UInt32
  FALSE_STRING : UInt32
  CONST_GT : UInt32
  IF_FLAG : UInt32
  SAVED_CONDITION : UInt32
  IF_RESULT_A : UInt32
  IF_RESULT_B : UInt32


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
  SAVED_TERM := C.OFFSET_SAVED_TERM_U32,
  FLAG_HASH := C.OFFSET_FLAG_HASH_U32,
  HT_HANDLE := C.OFFSET_HT_HANDLE_U32,
  IDENTIFIER_BUF := C.OFFSET_IDENTIFIER_BUF_U32,
  HT_VAL_BUF := C.OFFSET_HT_VAL_BUF_U32,
  HT_RESULT_BUF := C.OFFSET_HT_RESULT_BUF_U32,
  HT_RESULT_VAL := C.OFFSET_HT_RESULT_VAL_U32,
  IDENT_WRITE_PTR := C.OFFSET_IDENT_WRITE_PTR_U32,
  SAVED_IDENTIFIER_BUF := C.OFFSET_SAVED_IDENTIFIER_BUF_U32,
  CONST_LT := C.OFFSET_CONST_LT_U32,
  BOOL_FLAG := C.OFFSET_BOOL_FLAG_U32,
  TRUE_STRING := C.OFFSET_TRUE_STRING_U32,
  FALSE_STRING := C.OFFSET_FALSE_STRING_U32,
  CONST_GT := C.OFFSET_CONST_GT_U32,
  IF_FLAG := C.OFFSET_IF_FLAG_U32,
  SAVED_CONDITION := C.OFFSET_SAVED_CONDITION_U32,
  IF_RESULT_A := C.OFFSET_IF_RESULT_A_U32,
  IF_RESULT_B := C.OFFSET_IF_RESULT_B_U32
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
  let flagHash := zeros C.SIZE_16_N
  let htHandle := zeros C.SIZE_16_N
  let identBuf := zeros C.IDENT_BUF_SIZE_N
  let htValBuf := zeros C.SIZE_16_N
  let htResultBuf := zeros C.SIZE_16_N
  let identWritePtr := zeros C.SIZE_16_N
  let savedIdentBuf := zeros C.IDENT_BUF_SIZE_N
  let constLT := int32ToBytes16 C.ASCII_LT_I
  let boolFlag := zeros C.SIZE_16_N
  let trueString := padTo (stringToBytes "true\n") C.SIZE_16_N
  let falseString := padTo (stringToBytes "false\n") C.SIZE_16_N
  let constGT := int32ToBytes16 C.ASCII_GT_I
  let ifFlag := zeros C.SIZE_16_N
  let savedCondition := zeros C.SIZE_16_N
  let ifResultA := zeros C.SIZE_16_N
  let ifResultB := zeros C.SIZE_16_N

  ffiPtr ++ outputPath ++ argvParams ++ filenameBuf ++ ffiResult ++
    flagFfi ++ flagFile ++ flagSimd ++ flagMem ++ constZero ++ sourceBuf ++
    const48 ++ const10 ++ constOne ++ constSpace ++ constPlus ++
    pos ++ charBuf ++ accum ++ leftVal ++ rightVal ++ result ++
    digitCount ++ outputPos ++ outputBuf ++ outputPad ++ constRParen ++ constLParen ++
    term ++ constStar ++ constMinus ++ constL ++ constE ++ constColon ++
    constEquals ++ constIChar ++ constNChar ++ constX ++ varVal ++ constSemicolon ++
    constF ++ savedResult ++ savedTerm ++
    flagHash ++ htHandle ++ identBuf ++ htValBuf ++ htResultBuf ++ identWritePtr ++ savedIdentBuf ++
    constLT ++ boolFlag ++ trueString ++ falseString ++
    constGT ++
    ifFlag ++ savedCondition ++ ifResultA ++ ifResultB

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
  | SaveIdentBuf
  | RestoreIdentBuf
  | StoreResultFromTerm
  | StoreTermFromResult
  | StoreAccumFromResult
  | ClearIdentBuf
  | InitIdentWritePtr
  | StoreCharToIdent
  | LoadIdentWritePtr
  | AddIdentWritePtrOne
  | StoreIdentWritePtr
  | StoreAccumToHtVal
  | LoadHtResultLen
  | AddHtResultLenOne
  | StoreHtCheckResult
  | LoadHtResultVal
  | StoreAccumFromHtVal
  | LoadLT
  | SubCharLT
  | StoreDigitCountFromLT
  | CompareGT
  | SetBoolFlag
  | ClearBoolFlag
  | CopyTrueToLeft
  | CopyFalseToLeft
  | LoadGT
  | SubCharGT
  | StoreDigitCountFromGT
  | LoadEqualsChar
  | SubCharEquals
  | StoreDigitCountFromEquals
  | CompareGTReversed
  | CompareGE
  | CompareGEReversed
  | LoadIChar
  | SubCharIChar
  | StoreDigitCountFromIChar
  | LoadEChar
  | SubCharEChar
  | StoreDigitCountFromEChar
  | SetIfFlag
  | ClearIfFlag
  | SaveCondition
  | SaveIfResultA
  | SaveIfResultB
  | RestoreIfResultA
  | RestoreIfResultB
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
  SaveIdentBuf,
  RestoreIdentBuf,
  StoreResultFromTerm,
  StoreTermFromResult,
  StoreAccumFromResult,
  ClearIdentBuf,
  InitIdentWritePtr,
  StoreCharToIdent,
  LoadIdentWritePtr,
  AddIdentWritePtrOne,
  StoreIdentWritePtr,
  StoreAccumToHtVal,
  LoadHtResultLen,
  AddHtResultLenOne,
  StoreHtCheckResult,
  LoadHtResultVal,
  StoreAccumFromHtVal,
  LoadLT,
  SubCharLT,
  StoreDigitCountFromLT,
  CompareGT,
  SetBoolFlag,
  ClearBoolFlag,
  CopyTrueToLeft,
  CopyFalseToLeft,
  LoadGT,
  SubCharGT,
  StoreDigitCountFromGT,
  LoadEqualsChar,
  SubCharEquals,
  StoreDigitCountFromEquals,
  CompareGTReversed,
  CompareGE,
  CompareGEReversed,
  LoadIChar,
  SubCharIChar,
  StoreDigitCountFromIChar,
  LoadEChar,
  SubCharEChar,
  StoreDigitCountFromEChar,
  SetIfFlag,
  ClearIfFlag,
  SaveCondition,
  SaveIfResultA,
  SaveIfResultB,
  RestoreIfResultA,
  RestoreIfResultB
]

def workIndex (op : WorkOp) : Nat :=
  indexOf workOps op


def opToAction : WorkOp -> Action
  | FfiCall => w .FFICall L.FFI_PTR L.ARGV_PARAMS L.FFI_RESULT SIZE_NONE
  | FileRead => w .FileRead L.FILENAME_IN_ARGV L.SOURCE_BUF ZERO (L.SOURCE_BUF_SIZE.toUInt32)
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
  | FileWriteOutput => w .FileWrite L.LEFT_VAL L.OUTPUT_PATH ZERO SIZE_NONE
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
  | SaveIdentBuf => w .MemCopy L.IDENTIFIER_BUF L.SAVED_IDENTIFIER_BUF ZERO (u32 C.IDENT_BUF_SIZE_N)
  | RestoreIdentBuf => w .MemCopy L.SAVED_IDENTIFIER_BUF L.IDENTIFIER_BUF ZERO (u32 C.IDENT_BUF_SIZE_N)
  | StoreResultFromTerm => w .SimdStoreI32 (r RT) (r RA) L.RESULT SIZE_NONE
  | StoreTermFromResult => w .SimdStoreI32 (r RV) (r RA) L.TERM SIZE_NONE
  | StoreAccumFromResult => w .SimdStoreI32 (r RV) (r RA) L.ACCUM SIZE_NONE
  | ClearIdentBuf => w .MemWrite ZERO L.IDENTIFIER_BUF ZERO (u32 C.IDENT_BUF_SIZE_N)
  | InitIdentWritePtr => w .MemWrite ZERO L.IDENT_WRITE_PTR ZERO SIZE_I32
  | StoreCharToIdent => w .MemStoreIndirect L.CHAR_BUF L.IDENT_WRITE_PTR L.IDENTIFIER_BUF SIZE_BYTE
  | LoadIdentWritePtr => w .SimdLoadI32 L.IDENT_WRITE_PTR (r RJ) ZERO SIZE_NONE
  | AddIdentWritePtrOne => w .SimdAddI32 (r RJ) (r RL) (r RK) SIZE_NONE
  | StoreIdentWritePtr => w .SimdStoreI32 (r RL) (r RA) L.IDENT_WRITE_PTR SIZE_NONE
  | StoreAccumToHtVal => w .MemCopy L.ACCUM L.HT_VAL_BUF ZERO SIZE_I32
  | LoadHtResultLen => w .SimdLoadI32 L.HT_RESULT_BUF (r RS) ZERO SIZE_NONE
  | AddHtResultLenOne => w .SimdAddI32 (r RS) (r RU) (r RK) SIZE_NONE
  | StoreHtCheckResult => w .SimdStoreI32 (r RU) (r RA) L.DIGIT_COUNT SIZE_NONE
  | LoadHtResultVal => w .SimdLoadI32 L.HT_RESULT_VAL (r RS) ZERO SIZE_NONE
  | StoreAccumFromHtVal => w .SimdStoreI32 (r RS) (r RA) L.ACCUM SIZE_NONE
  | LoadLT => w .SimdLoadI32 L.CONST_LT (r RB) ZERO SIZE_NONE
  | SubCharLT => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromLT => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | CompareGT => w .Compare L.TERM L.ACCUM L.RESULT SIZE_I32
  | SetBoolFlag => w .MemWrite (u32 1) L.BOOL_FLAG ZERO SIZE_I32
  | ClearBoolFlag => w .MemWrite ZERO L.BOOL_FLAG ZERO SIZE_I32
  | CopyTrueToLeft => w .MemCopy L.TRUE_STRING L.LEFT_VAL ZERO (u32 16)
  | CopyFalseToLeft => w .MemCopy L.FALSE_STRING L.LEFT_VAL ZERO (u32 16)
  | LoadGT => w .SimdLoadI32 L.CONST_GT (r RB) ZERO SIZE_NONE
  | SubCharGT => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromGT => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | LoadEqualsChar => w .SimdLoadI32 L.CONST_EQUALS (r RB) ZERO SIZE_NONE
  | SubCharEquals => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromEquals => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | CompareGTReversed => w .Compare L.RESULT L.ACCUM L.TERM SIZE_I32
  | CompareGE => w .Compare L.TERM L.ACCUM L.RESULT (u32 5)
  | CompareGEReversed => w .Compare L.RESULT L.ACCUM L.TERM (u32 5)
  | LoadIChar => w .SimdLoadI32 L.CONST_I_CHAR (r RB) ZERO SIZE_NONE
  | SubCharIChar => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromIChar => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | LoadEChar => w .SimdLoadI32 L.CONST_E (r RB) ZERO SIZE_NONE
  | SubCharEChar => w .SimdSubI32 (r RA) (r RC) (r RB) SIZE_NONE
  | StoreDigitCountFromEChar => w .SimdStoreI32 (r RC) (r RA) L.DIGIT_COUNT SIZE_NONE
  | SetIfFlag => w .MemWrite (u32 1) L.IF_FLAG ZERO SIZE_I32
  | ClearIfFlag => w .MemWrite ZERO L.IF_FLAG ZERO SIZE_I32
  | SaveCondition => w .MemCopy L.RESULT L.SAVED_CONDITION ZERO SIZE_I32
  | SaveIfResultA => w .MemCopy L.RESULT L.IF_RESULT_A ZERO SIZE_I32
  | SaveIfResultB => w .MemCopy L.RESULT L.IF_RESULT_B ZERO SIZE_I32
  | RestoreIfResultA => w .MemCopy L.IF_RESULT_A L.RESULT ZERO SIZE_I32
  | RestoreIfResultB => w .MemCopy L.IF_RESULT_B L.RESULT ZERO SIZE_I32


def workActions : List Action := workOps.map opToAction

def htWorkActions : List Action := [
  w .HashTableCreate ZERO L.HT_HANDLE ZERO SIZE_NONE,
  w .HashTableInsert L.HT_VAL_BUF L.IDENTIFIER_BUF ZERO (u32 (32 * 65536 + 4)),
  w .HashTableLookup L.HT_RESULT_BUF L.IDENTIFIER_BUF ZERO (u32 (32 * 65536 + 8))
]

inductive Instr where
  | simd : WorkOp → Instr
  | mem : WorkOp → Instr
  | file : WorkOp → Instr
  | ffi : WorkOp → Instr
  | hash : Nat → Instr      -- 0=create, 1=insert, 2=lookup
  | label : String → Instr
  | jumpTo : UInt32 → String → Instr
  | jumpAlways : String → Instr
  | fence : Instr

def resolve (instrs : List Instr) : List Action :=
  -- First pass: count actions to determine workBase
  let actionCount := instrs.foldl (fun n i => n + match i with
    | .simd _ | .mem _ | .file _ | .ffi _ | .hash _ => 2
    | .jumpTo _ _ | .jumpAlways _ | .fence => 1
    | .label _ => 0) 0
  let workBase := actionCount
  let hashWorkBase := actionCount + workOps.length
  let wSrc (op : WorkOp) : UInt32 := u32 (workBase + workIndex op)
  let simdUnit := u32 C.SIMD_UNIT_ID_N
  let memUnit := u32 C.MEM_UNIT_ID_N
  let fileUnit := u32 C.FILE_UNIT_ID_N
  let ffiUnit := u32 C.FFI_UNIT_ID_N
  let htUnit := u32 C.HASH_TABLE_UNIT_ID_N
  let dispatch (unit src flag : UInt32) : Action :=
    { kind := .AsyncDispatch, dst := unit, src := src, offset := flag, size := SIZE_NONE }
  let waitA (flag : UInt32) : Action :=
    { kind := .Wait, dst := flag, src := ZERO, offset := ZERO, size := SIZE_NONE }
  let fenceA : Action :=
    { kind := .Fence, dst := ZERO, src := ZERO, offset := ZERO, size := ZERO }
  let jumpA (flag : UInt32) (target : Nat) : Action :=
    { kind := .ConditionalJump, src := flag, dst := u32 target, offset := ZERO, size := SIZE_I32 }
  let init : Array Action × List (String × Nat) × List (Nat × UInt32 × String) := (#[], [], [])
  let (actions, labels, patches) := instrs.foldl (fun state instr =>
    let (actions, labels, patches) := state
    match instr with
    | .simd op => (actions.push (dispatch simdUnit (wSrc op) L.FLAG_SIMD) |>.push (waitA L.FLAG_SIMD), labels, patches)
    | .mem op => (actions.push (dispatch memUnit (wSrc op) L.FLAG_MEM) |>.push (waitA L.FLAG_MEM), labels, patches)
    | .file op => (actions.push (dispatch fileUnit (wSrc op) L.FLAG_FILE) |>.push (waitA L.FLAG_FILE), labels, patches)
    | .ffi op => (actions.push (dispatch ffiUnit (wSrc op) L.FLAG_FFI) |>.push (waitA L.FLAG_FFI), labels, patches)
    | .hash idx => (actions.push (dispatch htUnit (u32 (hashWorkBase + idx)) L.FLAG_HASH) |>.push (waitA L.FLAG_HASH), labels, patches)
    | .label name => (actions, (name, actions.size) :: labels, patches)
    | .jumpTo flag name =>
      let idx := actions.size
      (actions.push (jumpA flag 0), labels, (idx, flag, name) :: patches)
    | .jumpAlways name =>
      let idx := actions.size
      (actions.push (jumpA L.CONST_ONE 0), labels, (idx, L.CONST_ONE, name) :: patches)
    | .fence => (actions.push fenceA, labels, patches)
  ) init
  let patched := patches.foldl (fun (a : Array Action) (patch : Nat × UInt32 × String) =>
    let (idx, flag, name) := patch
    let target := match labels.find? (fun p => p.1 == name) with
      | some (_, pos) => pos
      | none => 0
    a.set! idx (jumpA flag target)
  ) actions
  patched.toList

def simdStepI (op : WorkOp) : List Instr := [.simd op]
def memStepI (op : WorkOp) : List Instr := [.mem op]
def fileStepI (op : WorkOp) : List Instr := [.file op]
def ffiStepI (op : WorkOp) : List Instr := [.ffi op]
def hashStepCreateI : List Instr := [.hash 0]
def hashStepInsertI : List Instr := [.hash 1]
def hashStepLookupI : List Instr := [.hash 2]
def incPosI : List Instr := simdStepI LoadPos ++ simdStepI LoadOne ++ simdStepI AddPosOne ++ simdStepI StorePos
def incIdentWritePtrI : List Instr := simdStepI LoadIdentWritePtr ++ simdStepI LoadOne ++ simdStepI AddIdentWritePtrOne ++ simdStepI StoreIdentWritePtr
def loadCharI : List Instr := memStepI MemCopyChar ++ simdStepI LoadChar
def fenceI : Instr := .fence
def jumpPadI (a b : Instr) : List Instr := [a, b, .fence, .fence]
def charCheckI (l s st : WorkOp) : List Instr := simdStepI l ++ simdStepI s ++ simdStepI st

def operatorChainI (pfx : String) (mulLabel addLabel subLabel fallLabel : String) : List Instr :=
  [Instr.label (pfx ++ "_starCheck")] ++
  charCheckI LoadStar SubCharStar StoreDigitCountFromStar ++
  [Instr.jumpTo L.DIGIT_COUNT (pfx ++ "_plusCheck"), Instr.jumpAlways mulLabel] ++
  [Instr.label (pfx ++ "_plusCheck")] ++
  charCheckI LoadPlus SubCharPlus StoreDigitCountFromPlus ++
  [Instr.jumpTo L.DIGIT_COUNT (pfx ++ "_minusCheck"), Instr.jumpAlways addLabel] ++
  [Instr.label (pfx ++ "_minusCheck")] ++
  charCheckI LoadMinus SubCharMinus StoreDigitCountFromMinus ++
  [Instr.jumpTo L.DIGIT_COUNT fallLabel, Instr.jumpAlways subLabel]

def addToResultI : List Instr :=
  simdStepI LoadAccumForRight ++ simdStepI StoreRightVal ++
  simdStepI LoadResultForAdd ++ simdStepI LoadRightVal ++ simdStepI AddResultRight ++ simdStepI StoreResultAfterAdd ++
  memStepI ClearAccum

def mulResultI : List Instr :=
  simdStepI LoadAccumForRight ++ simdStepI StoreRightVal ++
  simdStepI LoadResultForAdd ++ simdStepI LoadRightVal ++ simdStepI MulResultRight ++ simdStepI StoreResultAfterMul ++
  memStepI ClearAccum

def termFromAccumI : List Instr :=
  simdStepI LoadAccumForResult ++ simdStepI StoreTermFromAccum ++
  memStepI ClearAccum

def multiplyTermI : List Instr :=
  simdStepI LoadTerm ++ simdStepI LoadAccumForRight ++
  simdStepI MulTermAccum ++ simdStepI StoreTerm ++
  memStepI ClearAccum

def addTermToResultI : List Instr :=
  simdStepI LoadResultForAdd ++ simdStepI LoadTerm ++
  simdStepI AddResultTerm ++ simdStepI StoreResultAfterTermAdd

def negateAccumToTermI : List Instr :=
  simdStepI LoadZeroForNegate ++ simdStepI LoadAccumForNegate ++
  simdStepI SubZeroAccum ++ simdStepI StoreTermFromNegate ++
  memStepI ClearAccum

def subResultI : List Instr :=
  simdStepI LoadAccumForRight ++ simdStepI StoreRightVal ++
  simdStepI LoadResultForAdd ++ simdStepI LoadRightVal ++ simdStepI SubResultRight ++ simdStepI StoreResultAfterSubRight ++
  memStepI ClearAccum

def skipSpacesI (loop done : String) : List Instr :=
  [Instr.label loop] ++
  loadCharI ++
  charCheckI LoadSpace SubCharSpace StoreDigitCountFromSpace ++
  [Instr.jumpTo L.DIGIT_COUNT done] ++
  incPosI ++
  [Instr.jumpAlways loop]

def skipI (pfx : String) : List Instr :=
  skipSpacesI (pfx ++ "_loop") (pfx ++ "_done") ++ [Instr.label (pfx ++ "_done")]
def incPosNI (n : Nat) : List Instr := (List.replicate n ()).flatMap fun _ => incPosI

structure TerminatorCheck where
  loadOp : WorkOp
  subOp : WorkOp
  storeOp : WorkOp

structure ParseLoopConfig where
  pfx : String
  terminators : List TerminatorCheck
  includeIdentBuf : Bool
  exitLabel : String

def parseLoopI (cfg : ParseLoopConfig) : List Instr :=
  let p := cfg.pfx
  let n := cfg.terminators.length
  let tLabel (idx : Nat) : String := p ++ "_t" ++ toString idx
  let accumLabel := p ++ "_accum"
  let firstCheck := if n > 0 then tLabel 0 else accumLabel
  let (termInstrs, _) := cfg.terminators.foldl (fun (acc : List Instr × Nat) tc =>
    let (instrs, idx) := acc
    let next := if idx + 1 < n then tLabel (idx + 1) else accumLabel
    (instrs ++
      [Instr.label (tLabel idx)] ++
      charCheckI tc.loadOp tc.subOp tc.storeOp ++
      jumpPadI (Instr.jumpTo L.DIGIT_COUNT next) (Instr.jumpAlways cfg.exitLabel),
     idx + 1)
  ) ([], 0)
  [Instr.label (p ++ "_loop")] ++
  loadCharI ++ simdStepI ClearCharBuf ++
  jumpPadI (Instr.jumpTo L.CHAR_BUF firstCheck) (Instr.jumpAlways cfg.exitLabel) ++
  termInstrs ++
  [Instr.label accumLabel] ++
  simdStepI LoadAsciiZero ++ simdStepI SubCharZero ++ simdStepI LoadBase ++
  simdStepI LoadAccum ++ simdStepI MulAccumBase ++ simdStepI AddAccumDigit ++ simdStepI StoreAccum ++
  (if cfg.includeIdentBuf then memStepI StoreCharToIdent ++ incIdentWritePtrI else []) ++
  incPosI ++ [Instr.jumpAlways (p ++ "_loop"), fenceI]

def numberTerminators : List TerminatorCheck := [
  ⟨LoadBaseForNewlineCheck, SubCharNewline, StoreDigitCountFromNewline⟩,
  ⟨LoadSpace, SubCharSpace, StoreDigitCountFromSpace⟩,
  ⟨LoadPlus, SubCharPlus, StoreDigitCountFromPlus⟩,
  ⟨LoadStar, SubCharStar, StoreDigitCountFromStar⟩,
  ⟨LoadMinus, SubCharMinus, StoreDigitCountFromMinus⟩,
  ⟨LoadSemicolon, SubCharSemicolon, StoreDigitCountFromSemicolon⟩]

def groupedNumberTerminators : List TerminatorCheck := [
  ⟨LoadBaseForNewlineCheck, SubCharNewline, StoreDigitCountFromNewline⟩,
  ⟨LoadSpace, SubCharSpace, StoreDigitCountFromSpace⟩,
  ⟨LoadPlus, SubCharPlus, StoreDigitCountFromPlus⟩,
  ⟨LoadStar, SubCharStar, StoreDigitCountFromStar⟩,
  ⟨LoadMinus, SubCharMinus, StoreDigitCountFromMinus⟩,
  ⟨LoadRParen, SubCharRParen, StoreDigitCountFromRParen⟩]

def addendTerminators : List TerminatorCheck := [
  ⟨LoadRParen, SubCharRParen, StoreDigitCountFromRParen⟩,
  ⟨LoadBaseForNewlineCheck, SubCharNewline, StoreDigitCountFromNewline⟩,
  ⟨LoadSpace, SubCharSpace, StoreDigitCountFromSpace⟩,
  ⟨LoadStar, SubCharStar, StoreDigitCountFromStar⟩]

def parseNumberI (pfx doneLabel : String) : List Instr :=
  parseLoopI { pfx, terminators := numberTerminators, includeIdentBuf := false, exitLabel := doneLabel }

def parseGroupedNumberI (pfx doneLabel : String) : List Instr :=
  parseLoopI { pfx, terminators := groupedNumberTerminators, includeIdentBuf := false, exitLabel := doneLabel }

def parseAddendI (pfx doneLabel : String) : List Instr :=
  parseLoopI { pfx, terminators := addendTerminators, includeIdentBuf := false, exitLabel := doneLabel }

def parseAtomI (pfx doneLabel : String) : List Instr :=
  let lookupLabel := pfx ++ "_lookup"
  let htFoundLabel := pfx ++ "_htFound"
  memStepI ClearIdentBuf ++ memStepI InitIdentWritePtr ++
  parseLoopI { pfx := pfx ++ "_inner", terminators := numberTerminators, includeIdentBuf := true, exitLabel := lookupLabel } ++
  [Instr.label lookupLabel] ++
  hashStepLookupI ++
  simdStepI LoadOne ++ simdStepI LoadHtResultLen ++ simdStepI AddHtResultLenOne ++ simdStepI StoreHtCheckResult ++
  [Instr.jumpTo L.DIGIT_COUNT htFoundLabel, Instr.jumpAlways doneLabel] ++
  [Instr.label htFoundLabel] ++
  simdStepI LoadHtResultVal ++ simdStepI StoreAccumFromHtVal



def itoaInitI : List Instr := memStepI InitOutputPos ++ memStepI WriteOutputNewline

def outputI : List Instr :=
  simdStepI LoadOutputPosForDigitCount ++ simdStepI AddOutputPosOne ++ simdStepI StoreDigitCountFinal ++
  memStepI CopyOutputToLeft ++
  fileStepI FileWriteOutput

def accumFromResultI : List Instr :=
  simdStepI LoadResultForItoa ++ simdStepI StoreAccumFromResult ++ memStepI ClearResult

def itoaLoopI (loop : String) : List Instr :=
  [Instr.label loop] ++
  simdStepI LoadResultForItoa ++ simdStepI DivResultBase ++ simdStepI MulQuotBase ++ simdStepI SubResultQuot ++
  simdStepI AddRemAsciiZero ++ simdStepI StoreCharBuf ++
  memStepI StoreCharToOutput ++
  simdStepI LoadOutputPos ++ simdStepI SubOutputPosOne ++ simdStepI StoreOutputPos ++ simdStepI StoreQuotToResult ++
  [Instr.jumpTo L.RESULT loop, fenceI]

def parenCheckBeforeParseI (normalLabel parenLabel : String) : List Instr :=
  loadCharI ++
  charCheckI LoadLParen SubCharLParen StoreDigitCountFromLParen ++
  [Instr.jumpTo L.DIGIT_COUNT normalLabel, Instr.jumpAlways parenLabel]

structure OpCheck where
  ops : List WorkOp     -- load, sub, store triple (or single for ClearCharBuf NUL check)
  flag : UInt32         -- flag to check after ops
  matchLabel : String   -- where to jump on match

inductive ParseKind where
  | Number
  | GroupedNumber
  | Atom

def emitParseI (kind : ParseKind) (pfx doneLabel : String) : List Instr :=
  match kind with
  | .Number => parseNumberI pfx doneLabel
  | .GroupedNumber => parseGroupedNumberI pfx doneLabel
  | .Atom => parseAtomI pfx doneLabel

def emitSkipParseI (kind : ParseKind) (pfx : String) : List Instr :=
  let skipLabel := pfx ++ "_skip"
  let parseDone := pfx ++ "_parseDone"
  skipI skipLabel ++
  emitParseI kind pfx parseDone ++
  [Instr.label parseDone]

def parenSubEvalI (pfx : String) : List Instr :=
  let p := pfx
  let checkOp := p ++ "_checkOp"
  let handleMul := p ++ "_handleMul"
  let handleAdd := p ++ "_handleAdd"
  let handleSub := p ++ "_handleSub"
  let finalize := p ++ "_finalize"
  -- init: skip spaces, parse grouped number, termFromAccum
  emitSkipParseI .GroupedNumber (p ++ "_init") ++
  termFromAccumI ++
  -- checkOp: skip spaces, check rparen/star/plus/minus
  [Instr.label checkOp] ++
  skipI (p ++ "_opSkip") ++
  loadCharI ++
  charCheckI LoadRParen SubCharRParen StoreDigitCountFromRParen ++
  [Instr.jumpTo L.DIGIT_COUNT (p ++ "_starCheck"), Instr.jumpAlways finalize] ++
  operatorChainI p handleMul handleAdd handleSub finalize ++
  -- handleMul
  [Instr.label handleMul] ++
  incPosI ++
  emitSkipParseI .GroupedNumber (p ++ "_mul") ++
  multiplyTermI ++
  [Instr.jumpAlways checkOp] ++
  -- handleAdd
  [Instr.label handleAdd] ++
  addTermToResultI ++
  incPosI ++
  emitSkipParseI .GroupedNumber (p ++ "_add") ++
  termFromAccumI ++
  [Instr.jumpAlways checkOp] ++
  -- handleSub
  [Instr.label handleSub] ++
  addTermToResultI ++
  incPosI ++
  emitSkipParseI .GroupedNumber (p ++ "_sub") ++
  negateAccumToTermI ++
  [Instr.jumpAlways checkOp] ++
  -- finalize
  [Instr.label finalize] ++
  addTermToResultI

def exprEvalI
    (pfx : String)
    (parseKind : ParseKind)
    (terminators : List OpCheck)
    (initHasParen : Bool)
    (handlersHasParen : Bool)
    (finalizeTail : List Instr)
    (initSkipSpaces : Bool := true)
    (afterMinusFallthrough : Option String := none)
    : List Instr :=
  let p := pfx
  let checkOp := p ++ "_checkOp"
  let handleMul := p ++ "_handleMul"
  let handleAdd := p ++ "_handleAdd"
  let handleSub := p ++ "_handleSub"
  let finalize := p ++ "_finalize"
  let minusFallLabel := match afterMinusFallthrough with
    | some l => l
    | none => finalize
  -- init section
  let initSkip :=
    if initSkipSpaces then
      skipI (p ++ "_initSkip")
    else []
  let initInstrs :=
    if initHasParen then
      initSkip ++
      parenCheckBeforeParseI (p ++ "_initNormal") (p ++ "_initParen") ++
      [Instr.label (p ++ "_initNormal")] ++
      emitParseI parseKind (p ++ "_initParse") (p ++ "_initParseDone") ++
      [Instr.label (p ++ "_initParseDone")] ++
      termFromAccumI ++
      [Instr.jumpAlways checkOp] ++
      [Instr.label (p ++ "_initParen")] ++
      incPosI ++
      parenSubEvalI (p ++ "_initMGE") ++
      accumFromResultI ++
      incPosI ++
      termFromAccumI
    else if initSkipSpaces then
      emitSkipParseI parseKind (p ++ "_init") ++
      termFromAccumI
    else
      emitParseI parseKind (p ++ "_initParse") (p ++ "_initParseDone") ++
      [Instr.label (p ++ "_initParseDone")] ++
      termFromAccumI
  -- checkOp section
  let n := terminators.length
  let termChecks := terminators.foldl (fun (acc : List Instr × Nat) tc =>
    let (instrs, idx) := acc
    let currentLabel := p ++ "_termNext_" ++ toString idx
    let jumpTarget := if idx + 1 < n
                      then p ++ "_termNext_" ++ toString (idx + 1)
                      else p ++ "_starCheck"
    let checkInstrs :=
      (tc.ops.map fun op => simdStepI op).foldl (· ++ ·) [] ++
      [Instr.jumpTo tc.flag jumpTarget, Instr.jumpAlways tc.matchLabel]
    (instrs ++ [Instr.label currentLabel] ++ checkInstrs, idx + 1)
  ) ([], 0)
  let checkOpInstrs :=
    [Instr.label checkOp] ++
    skipI (p ++ "_opSkip") ++
    loadCharI ++
    termChecks.1 ++
    operatorChainI p handleMul handleAdd handleSub minusFallLabel
  initInstrs ++
  checkOpInstrs ++
  -- handleMul
  (if handlersHasParen then
    [Instr.label handleMul] ++
    incPosI ++
    skipI (p ++ "_MulSkip") ++
    parenCheckBeforeParseI (p ++ "_MulNormal") (p ++ "_MulParen") ++
    [Instr.label (p ++ "_MulNormal")] ++
    emitParseI parseKind (p ++ "_MulParse") (p ++ "_MulParseDone") ++
    [Instr.label (p ++ "_MulParseDone")] ++
    multiplyTermI ++
    [Instr.jumpAlways checkOp] ++
    [Instr.label (p ++ "_MulParen")] ++
    memStepI SaveTerm ++ incPosI ++
    parenSubEvalI (p ++ "_MulMGE") ++
    accumFromResultI ++ incPosI ++
    memStepI RestoreTerm ++
    multiplyTermI ++
    [Instr.jumpAlways checkOp]
  else
    [Instr.label handleMul] ++
    incPosI ++
    emitSkipParseI parseKind (p ++ "_Mul") ++
    multiplyTermI ++
    [Instr.jumpAlways checkOp]) ++
  -- handleAdd
  (if handlersHasParen then
    [Instr.label handleAdd] ++
    addTermToResultI ++
    incPosI ++
    skipI (p ++ "_AddSkip") ++
    parenCheckBeforeParseI (p ++ "_AddNormal") (p ++ "_AddParen") ++
    [Instr.label (p ++ "_AddNormal")] ++
    emitParseI parseKind (p ++ "_AddParse") (p ++ "_AddParseDone") ++
    [Instr.label (p ++ "_AddParseDone")] ++
    termFromAccumI ++
    [Instr.jumpAlways checkOp] ++
    [Instr.label (p ++ "_AddParen")] ++
    memStepI SaveTerm ++ incPosI ++
    parenSubEvalI (p ++ "_AddMGE") ++
    accumFromResultI ++ incPosI ++
    memStepI RestoreTerm ++
    termFromAccumI ++
    [Instr.jumpAlways checkOp]
  else
    [Instr.label handleAdd] ++
    addTermToResultI ++
    incPosI ++
    emitSkipParseI parseKind (p ++ "_Add") ++
    termFromAccumI ++
    [Instr.jumpAlways checkOp]) ++
  -- handleSub
  (if handlersHasParen then
    [Instr.label handleSub] ++
    addTermToResultI ++
    incPosI ++
    skipI (p ++ "_SubSkip") ++
    parenCheckBeforeParseI (p ++ "_SubNormal") (p ++ "_SubParen") ++
    [Instr.label (p ++ "_SubNormal")] ++
    emitParseI parseKind (p ++ "_SubParse") (p ++ "_SubParseDone") ++
    [Instr.label (p ++ "_SubParseDone")] ++
    negateAccumToTermI ++
    [Instr.jumpAlways checkOp] ++
    [Instr.label (p ++ "_SubParen")] ++
    memStepI SaveTerm ++ incPosI ++
    parenSubEvalI (p ++ "_SubMGE") ++
    accumFromResultI ++ incPosI ++
    memStepI RestoreTerm ++
    negateAccumToTermI ++
    [Instr.jumpAlways checkOp]
  else
    [Instr.label handleSub] ++
    addTermToResultI ++
    incPosI ++
    emitSkipParseI parseKind (p ++ "_Sub") ++
    negateAccumToTermI ++
    [Instr.jumpAlways checkOp]) ++
  -- finalize
  [Instr.label finalize] ++
  addTermToResultI ++
  finalizeTail

def setupI : List Instr :=
  ffiStepI FfiCall ++ fileStepI FileRead ++
  memStepI ClearAccum ++ memStepI ClearResult ++ memStepI ClearTerm ++
  memStepI ClearBoolFlag ++ memStepI ClearIfFlag

def parenCheckI : List Instr :=
  [Instr.label "parenCheck"] ++
  loadCharI ++
  charCheckI LoadLParen SubCharLParen StoreDigitCountFromLParen ++
  [Instr.jumpTo L.DIGIT_COUNT "letCheck"] ++
  incPosI ++
  loadCharI ++
  charCheckI LoadF SubCharF StoreDigitCountFromF ++
  [Instr.jumpTo L.DIGIT_COUNT "groupingPath", Instr.jumpAlways "lambdaPath"]

def findOperatorI : List Instr :=
  let loop := "lam_findOp_loop"
  let done := "lam_findOp_done"
  [Instr.label loop] ++
  loadCharI ++
  charCheckI LoadPlus SubCharPlus StoreDigitCountFromPlus ++
  [Instr.jumpTo L.DIGIT_COUNT "lam_findOp_starCheck"] ++
  memStepI ClearLeftVal ++
  [Instr.jumpAlways done] ++
  [Instr.label "lam_findOp_starCheck"] ++
  charCheckI LoadStar SubCharStar StoreDigitCountFromStar ++
  [Instr.jumpTo L.DIGIT_COUNT "lam_findOp_minusCheck"] ++
  memStepI SetLeftVal ++
  [Instr.jumpAlways done] ++
  [Instr.label "lam_findOp_minusCheck"] ++
  charCheckI LoadMinus SubCharMinus StoreDigitCountFromMinus ++
  [Instr.jumpTo L.DIGIT_COUNT "lam_findOp_next"] ++
  memStepI SetLeftValTwo ++
  [Instr.jumpAlways done] ++
  [Instr.label "lam_findOp_next"] ++
  incPosI ++
  [Instr.jumpAlways loop] ++
  [Instr.label done]

def lambdaPathI : List Instr :=
  [Instr.label "lambdaPath"] ++
  memStepI ClearLeftVal ++
  findOperatorI ++
  incPosI ++
  skipI "lam_skip1" ++
  parseAddendI "lam_addend" "lam_addendDone" ++
  [Instr.label "lam_addendDone"] ++
  simdStepI LoadAccumForResult ++ simdStepI StoreResultFromAccum ++
  memStepI ClearAccum ++
  incPosI ++
  skipI "lam_skip2" ++
  parseNumberI "lam_arg" "lam_argDone" ++
  [Instr.label "lam_argDone"] ++
  [Instr.jumpTo L.LEFT_VAL "lam_notAdd"] ++
  addToResultI ++
  [Instr.jumpAlways "output"] ++
  [Instr.label "lam_notAdd"] ++
  simdStepI LoadOne ++ simdStepI LoadLeftValForCheck ++ simdStepI SubLeftValOne ++ simdStepI StoreDigitCountFromLeftCheck ++
  [Instr.jumpTo L.DIGIT_COUNT "lam_subPath"] ++
  mulResultI ++
  [Instr.jumpAlways "output"] ++
  [Instr.label "lam_subPath"] ++
  subResultI ++
  [Instr.jumpAlways "output"]

def groupingPathI : List Instr :=
  [Instr.label "groupingPath"] ++
  exprEvalI "gp" .GroupedNumber
    [⟨[LoadRParen, SubCharRParen, StoreDigitCountFromRParen], L.DIGIT_COUNT, "gp_finalize"⟩]
    false false
    (simdStepI LoadResultForItoa ++ simdStepI StoreTermFromResult ++
     memStepI ClearResult ++ incPosI ++
     [Instr.jumpAlways "main_checkOp"])

def letCheckI : List Instr :=
  [Instr.label "letCheck"] ++
  loadCharI ++
  charCheckI LoadL SubCharL StoreDigitCountFromL ++
  [Instr.jumpTo L.DIGIT_COUNT "ifCheck"]

def parseBindingExprI (pfx doneLabel : String) : List Instr :=
  exprEvalI pfx .Atom
    [⟨[ClearCharBuf], L.CHAR_BUF, (pfx ++ "_finalize")⟩,
     ⟨[LoadSemicolon, SubCharSemicolon, StoreDigitCountFromSemicolon], L.DIGIT_COUNT, (pfx ++ "_finalize")⟩,
     ⟨[LoadBaseForNewlineCheck, SubCharNewline, StoreDigitCountFromNewline], L.DIGIT_COUNT, (pfx ++ "_finalize")⟩]
    false false
    (simdStepI LoadResultForItoa ++ simdStepI StoreAccumFromResult ++
     memStepI ClearResult ++
     [Instr.jumpAlways doneLabel])
    (initSkipSpaces := false)

def identLoopI : List Instr :=
  [Instr.label "let_identLoop"] ++
  loadCharI ++
  charCheckI LoadSpace SubCharSpace StoreDigitCountFromSpace ++
  [Instr.jumpTo L.DIGIT_COUNT "let_identContinue", Instr.jumpAlways "let_identDone"] ++
  [Instr.label "let_identContinue"] ++
  memStepI StoreCharToIdent ++ incIdentWritePtrI ++ incPosI ++
  [Instr.jumpAlways "let_identLoop"] ++
  [Instr.label "let_identDone"]

def letBodyEvalI : List Instr :=
  exprEvalI "letBody" .Atom
    [⟨[ClearCharBuf], L.CHAR_BUF, "letBody_finalize"⟩,
     ⟨[LoadBaseForNewlineCheck, SubCharNewline, StoreDigitCountFromNewline], L.DIGIT_COUNT, "letBody_finalize"⟩]
    false false
    [Instr.jumpAlways "output"]
    (afterMinusFallthrough := some "letBody_compRedirect")

def letBodyCompRedirectI : List Instr :=
  [Instr.label "letBody_compRedirect"] ++
  addTermToResultI ++
  memStepI ClearTerm ++
  [Instr.jumpAlways "main_checkOp"]

def letPathI : List Instr :=
  [Instr.label "letPath"] ++
  hashStepCreateI ++
  -- letBindingLoop:
  [Instr.label "let_bindLoop"] ++
  incPosNI 4 ++  -- skip "let "
  memStepI ClearIdentBuf ++ memStepI InitIdentWritePtr ++
  identLoopI ++
  -- skip " := "
  incPosI ++ incPosI ++ incPosI ++ incPosI ++
  skipI "let_skipBeforeExpr" ++
  memStepI SaveIdentBuf ++
  parseBindingExprI "letBind" "let_afterBind" ++
  [Instr.label "let_afterBind"] ++
  memStepI RestoreIdentBuf ++
  memStepI StoreAccumToHtVal ++ hashStepInsertI ++ memStepI ClearAccum ++
  -- semicolon check
  loadCharI ++
  charCheckI LoadSemicolon SubCharSemicolon StoreDigitCountFromSemicolon ++
  [Instr.jumpTo L.DIGIT_COUNT "let_bodyStart"] ++
  -- semicolon found: skip ; then spaces, check for 'let'
  incPosI ++
  skipI "let_afterSemi" ++
  loadCharI ++
  charCheckI LoadL SubCharL StoreDigitCountFromL ++
  [Instr.jumpTo L.DIGIT_COUNT "let_bodyStart", Instr.jumpAlways "let_bindLoop"] ++
  -- bodyStart: check for 'if' before body
  [Instr.label "let_bodyStart"] ++
  loadCharI ++
  charCheckI LoadIChar SubCharIChar StoreDigitCountFromIChar ++
  [Instr.jumpTo L.DIGIT_COUNT "let_bodyNormal"] ++
  incPosI ++ incPosI ++ incPosI ++
  memStepI SetIfFlag ++
  skipI "let_bodyIfSkip" ++
  [Instr.jumpAlways "let_bodyNormal"] ++
  -- body evaluation
  [Instr.label "let_bodyNormal"] ++
  letBodyEvalI ++
  letBodyCompRedirectI

def ifCheckI : List Instr :=
  [Instr.label "ifCheck"] ++
  loadCharI ++
  charCheckI LoadIChar SubCharIChar StoreDigitCountFromIChar ++
  [Instr.jumpTo L.DIGIT_COUNT "normalStart"] ++
  incPosI ++ incPosI ++ incPosI ++
  memStepI SetIfFlag ++
  skipI "ifCheck_skip" ++
  [Instr.jumpAlways "parenCheck"]

def mainExprEvalI : List Instr :=
  [Instr.label "normalStart"] ++
  exprEvalI "main" .Number [] false true
    [Instr.jumpAlways "boolCheck"]
    (afterMinusFallthrough := some "main_ltCheck")

def handleComparisonI (pfx : String) (strictOp eqOp : WorkOp) : List Instr :=
  [Instr.label ("main_handle" ++ pfx)] ++
  addTermToResultI ++
  memStepI ClearLeftVal ++
  incPosI ++
  loadCharI ++
  charCheckI LoadEqualsChar SubCharEquals StoreDigitCountFromEquals ++
  [Instr.jumpTo L.DIGIT_COUNT (pfx ++ "_afterEq")] ++
  incPosI ++
  memStepI SetLeftVal ++
  [Instr.label (pfx ++ "_afterEq")] ++
  skipI (pfx ++ "_skip") ++
  parseNumberI (pfx ++ "_parse") (pfx ++ "_parseDone") ++
  [Instr.label (pfx ++ "_parseDone")] ++
  termFromAccumI ++
  [Instr.jumpTo L.LEFT_VAL (pfx ++ "_eqPath")] ++
  memStepI strictOp ++
  [Instr.jumpAlways (pfx ++ "_afterCompare")] ++
  [Instr.label (pfx ++ "_eqPath")] ++
  memStepI eqOp ++
  [Instr.label (pfx ++ "_afterCompare")] ++
  memStepI ClearResult ++
  memStepI SetBoolFlag ++
  memStepI ClearLeftVal ++
  termFromAccumI ++
  [Instr.jumpAlways "main_checkOp"]

def handleLTI : List Instr :=
  [Instr.label "main_ltCheck"] ++
  charCheckI LoadLT SubCharLT StoreDigitCountFromLT ++
  [Instr.jumpTo L.DIGIT_COUNT "main_gtCheck", Instr.jumpAlways "main_handleLT"] ++
  [Instr.label "main_gtCheck"] ++
  charCheckI LoadGT SubCharGT StoreDigitCountFromGT ++
  [Instr.jumpTo L.DIGIT_COUNT "main_finalize", Instr.jumpAlways "main_handleGT"] ++
  handleComparisonI "LT" CompareGT CompareGE ++
  handleComparisonI "GT" CompareGTReversed CompareGEReversed

def boolCheckI : List Instr :=
  [Instr.label "boolCheck"] ++
  [Instr.jumpTo L.BOOL_FLAG "boolCheck_ifFlag", Instr.jumpAlways "itoa"] ++
  [Instr.label "boolCheck_ifFlag"] ++
  [Instr.jumpTo L.IF_FLAG "ifThenElse", Instr.jumpAlways "resultCheck"]

def resultCheckI : List Instr :=
  [Instr.label "resultCheck"] ++
  [Instr.jumpTo L.RESULT "trueOutput", Instr.jumpAlways "falseOutput"]

def trueOutputI : List Instr :=
  [Instr.label "trueOutput"] ++
  memStepI CopyTrueToLeft ++
  fileStepI FileWriteOutput ++
  [Instr.jumpAlways "end"]

def falseOutputI : List Instr :=
  [Instr.label "falseOutput"] ++
  memStepI CopyFalseToLeft ++
  fileStepI FileWriteOutput ++
  [Instr.jumpAlways "end"]

def branchEvalI (pfx doneLabel : String) (termOps : List WorkOp) : List Instr :=
  exprEvalI pfx .Atom
    [⟨[ClearCharBuf], L.CHAR_BUF, (pfx ++ "_finalize")⟩,
     ⟨termOps, L.DIGIT_COUNT, (pfx ++ "_finalize")⟩]
    true true
    [Instr.jumpAlways doneLabel]

def ifThenElseI : List Instr :=
  [Instr.label "ifThenElse"] ++
  memStepI SaveCondition ++
  memStepI ClearBoolFlag ++ memStepI ClearIfFlag ++ memStepI ClearResult ++ memStepI ClearAccum ++ memStepI ClearTerm ++
  incPosNI 5 ++  -- skip "then "
  skipI "ite_thenSkip" ++
  branchEvalI "then" "ite_afterThen" [LoadEChar, SubCharEChar, StoreDigitCountFromEChar] ++
  [Instr.label "ite_afterThen"] ++
  memStepI SaveIfResultA ++
  memStepI ClearResult ++ memStepI ClearAccum ++ memStepI ClearTerm ++
  incPosNI 5 ++  -- skip "else "
  skipI "ite_elseSkip" ++
  branchEvalI "else" "ite_afterElse" [LoadBaseForNewlineCheck, SubCharNewline, StoreDigitCountFromNewline] ++
  [Instr.label "ite_afterElse"] ++
  memStepI SaveIfResultB ++
  [Instr.jumpTo L.SAVED_CONDITION "ite_restoreA"] ++
  memStepI RestoreIfResultB ++
  [Instr.jumpAlways "itoa"] ++
  [Instr.label "ite_restoreA"] ++
  memStepI RestoreIfResultA ++
  [Instr.jumpAlways "itoa"]

def mainProgramI : List Instr :=
  setupI ++
  parenCheckI ++
  lambdaPathI ++
  groupingPathI ++
  letCheckI ++
  letPathI ++
  ifCheckI ++
  mainExprEvalI ++
  handleLTI ++
  boolCheckI ++
  resultCheckI ++
  trueOutputI ++
  falseOutputI ++
  ifThenElseI ++
  [Instr.label "itoa", Instr.label "output"] ++
  itoaInitI ++
  itoaLoopI "itoaLoop" ++
  outputI ++
  [Instr.label "end"]


def leanEvalAlgorithm : Algorithm := {
  actions := resolve mainProgramI ++ workActions ++ htWorkActions,
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
    hash_table_units := C.HASH_TABLE_UNIT_COUNT_N,
    lmdb_units := 0,
    backends_bits := ZERO
  },
  simd_assignments := [],
  computational_assignments := [],
  memory_assignments := [],
  file_assignments := [],
  network_assignments := [],
  ffi_assignments := [],
  hash_table_assignments := [],
  lmdb_assignments := [],
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
