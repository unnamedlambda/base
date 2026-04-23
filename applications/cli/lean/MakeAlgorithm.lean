import AlgorithmLib

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout

namespace Algorithm

structure Fields where
  reserved : Fld (.bytes 64)
  input : Fld (.bytes 256)
  scratch : Fld (.bytes 32)
  output : Fld (.bytes 32)
  inputLen : Fld .i64
  firstVal : Fld .i64
  secondVal : Fld .i64
  result : Fld .i64
  outputLen : Fld .i64

def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved ← field (.bytes 64)
  let input ← field (.bytes 256)
  let scratch ← field (.bytes 32)
  let output ← field (.bytes 32)
  let inputLen ← field .i64
  let firstVal ← field .i64
  let secondVal ← field .i64
  let result ← field .i64
  let outputLen ← field .i64
  pure { reserved, input, scratch, output, inputLen, firstVal, secondVal, result, outputLen }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

open AlgorithmLib.IR

def asciiSpace : Int := 32
def asciiNewline : Int := 10
def asciiMinus : Int := 45
def asciiPlus : Int := 43
def asciiZero : Int := 48
def asciiNine : Int := 57

def loadByteAt (ptr : Val) (baseOff : Nat) (idx : Val) : IRBuilder Val := do
  let base ← iconst64 baseOff
  let rel ← iadd base idx
  let addr ← iadd ptr rel
  uload8_64 addr

def storeByteAt (ptr : Val) (baseOff : Nat) (idx : Val) (value : Val) : IRBuilder Unit := do
  let base ← iconst64 baseOff
  let rel ← iadd base idx
  let addr ← iadd ptr rel
  istore8 value addr

def loadInputByte (ptr idx : Val) : IRBuilder Val :=
  loadByteAt ptr f.input.offset idx

def loadScratchByte (ptr idx : Val) : IRBuilder Val :=
  loadByteAt ptr f.scratch.offset idx

def storeScratchByte (ptr idx value : Val) : IRBuilder Unit :=
  storeByteAt ptr f.scratch.offset idx value

def storeOutputByte (ptr idx value : Val) : IRBuilder Unit :=
  storeByteAt ptr f.output.offset idx value

def emitSkipWs (ptr pos len : Val) : IRBuilder Val := do
  let one ← iconst64 1
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline

  let hdr ← declareBlock [.i64]
  let i := hdr.param 0
  let body ← declareBlock []
  let nlCheck ← declareBlock [.i64]
  let advance ← declareBlock []
  let done ← declareBlock [.i64]

  jump hdr.ref [pos]

  startBlock hdr
  let atEnd ← icmp .uge i len
  brif atEnd done.ref [i] body.ref []

  startBlock body
  let ch ← loadInputByte ptr i
  let isSp ← icmp .eq ch sp
  brif isSp advance.ref [] nlCheck.ref [ch]

  startBlock nlCheck
  let ch2 := nlCheck.param 0
  let isNl ← icmp .eq ch2 nl
  brif isNl advance.ref [] done.ref [i]

  startBlock advance
  let nextI ← iadd i one
  jump hdr.ref [nextI]

  startBlock done
  pure (done.param 0)

def emitParseInt (ptr start len : Val) : IRBuilder (Val × Val) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let ten ← iconst64 10
  let minus ← iconst64 asciiMinus
  let plus ← iconst64 asciiPlus
  let zeroCh ← iconst64 asciiZero
  let nineCh ← iconst64 asciiNine

  let pos0 ← emitSkipWs ptr start len

  let signHdr ← declareBlock [.i64]
  let signPos := signHdr.param 0
  let signRead ← declareBlock []
  let signPlus ← declareBlock [.i64]
  let signCh := signPlus.param 0
  let beginDigits ← declareBlock [.i64, .i64]
  let digitStart := beginDigits.param 0
  let negFlag := beginDigits.param 1
  let digitsHdr ← declareBlock [.i64, .i64, .i64]
  let i := digitsHdr.param 0
  let acc := digitsHdr.param 1
  let neg := digitsHdr.param 2
  let digitRead ← declareBlock []
  let digitUpper ← declareBlock [.i64]
  let digitCh := digitUpper.param 0
  let digitStep ← declareBlock [.i64]
  let digitCh2 := digitStep.param 0
  let done ← declareBlock [.i64, .i64, .i64]

  jump signHdr.ref [pos0]

  startBlock signHdr
  let atEndSign ← icmp .uge signPos len
  brif atEndSign done.ref [zero, signPos, zero] signRead.ref []

  startBlock signRead
  let ch ← loadInputByte ptr signPos
  let isMinus ← icmp .eq ch minus
  let signPosNext ← iadd signPos one
  brif isMinus beginDigits.ref [signPosNext, one] signPlus.ref [ch]

  startBlock signPlus
  let isPlus ← icmp .eq signCh plus
  let signPosNext ← iadd signPos one
  brif isPlus beginDigits.ref [signPosNext, zero] beginDigits.ref [signPos, zero]

  startBlock beginDigits
  jump digitsHdr.ref [digitStart, zero, negFlag]

  startBlock digitsHdr
  let atEndDigits ← icmp .uge i len
  brif atEndDigits done.ref [acc, i, neg] digitRead.ref []

  startBlock digitRead
  let ch ← loadInputByte ptr i
  let belowZero ← icmp .ult ch zeroCh
  brif belowZero done.ref [acc, i, neg] digitUpper.ref [ch]

  startBlock digitUpper
  let aboveNine ← icmp .ugt digitCh nineCh
  brif aboveNine done.ref [acc, i, neg] digitStep.ref [digitCh]

  startBlock digitStep
  let digit ← isub digitCh2 zeroCh
  let acc10 ← imul acc ten
  let nextAcc ← iadd acc10 digit
  let nextI ← iadd i one
  jump digitsHdr.ref [nextI, nextAcc, neg]

  startBlock done
  let magnitude := done.param 0
  let endPos := done.param 1
  let negResult := done.param 2
  let negated ← ineg magnitude
  let isNeg ← icmp .eq negResult one
  let finalVal ← select' isNeg negated magnitude
  pure (finalVal, endPos)

def emitFormatSigned (ptr value : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
  let ten ← iconst64 10
  let minusCh ← iconst64 asciiMinus
  let zeroCh ← iconst64 asciiZero
  let nlCh ← iconst64 asciiNewline
  let scratchLast ← iconst64 31

  let isNeg ← icmp .slt value zero
  let negValue ← ineg value
  let absVal ← select' isNeg negValue value
  let isZero ← icmp .eq absVal zero

  let zeroBlk ← declareBlock []
  let nonZeroBlk ← declareBlock []
  let divHdr ← declareBlock [.i64, .i64]
  let cur := divHdr.param 0
  let idx := divHdr.param 1
  let divBody ← declareBlock []
  let divDone ← declareBlock [.i64]
  let firstIdx := divDone.param 0
  let signBlk ← declareBlock [.i64]
  let signFirstIdx := signBlk.param 0
  let copyHdr ← declareBlock [.i64, .i64]
  let copyIdx := copyHdr.param 0
  let outPos := copyHdr.param 1
  let copyBody ← declareBlock []
  let newlineBlk ← declareBlock [.i64]
  let newlinePos := newlineBlk.param 0
  let done ← declareBlock [.i64]

  brif isZero zeroBlk.ref [] nonZeroBlk.ref []

  startBlock zeroBlk
  storeOutputByte ptr zero zeroCh
  storeOutputByte ptr one nlCh
  jump done.ref [two]

  startBlock nonZeroBlk
  jump divHdr.ref [absVal, scratchLast]

  startBlock divHdr
  let doneDiv ← icmp .eq cur zero
  brif doneDiv divDone.ref [idx] divBody.ref []

  startBlock divBody
  let q ← udiv cur ten
  let q10 ← imul q ten
  let rem ← isub cur q10
  let digitCh ← iadd zeroCh rem
  storeScratchByte ptr idx digitCh
  let nextIdx ← isub idx one
  jump divHdr.ref [q, nextIdx]

  startBlock divDone
  let firstDigitIdx ← iadd firstIdx one
  brif isNeg signBlk.ref [firstDigitIdx] copyHdr.ref [firstDigitIdx, zero]

  startBlock signBlk
  storeOutputByte ptr zero minusCh
  jump copyHdr.ref [signFirstIdx, one]

  startBlock copyHdr
  let copiedAll ← icmp .ugt copyIdx scratchLast
  brif copiedAll newlineBlk.ref [outPos] copyBody.ref []

  startBlock copyBody
  let ch ← loadScratchByte ptr copyIdx
  storeOutputByte ptr outPos ch
  let nextIdx ← iadd copyIdx one
  let nextOutPos ← iadd outPos one
  jump copyHdr.ref [nextIdx, nextOutPos]

  startBlock newlineBlk
  storeOutputByte ptr newlinePos nlCh
  let totalLen ← iadd newlinePos one
  jump done.ref [totalLen]

  startBlock done
  pure (done.param 0)

def emitParseAddChain (ptr len : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let plus ← iconst64 asciiPlus
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline

  let (first, pos1) ← emitParseInt ptr zero len
  fldStore ptr f.firstVal first
  fldStore ptr f.secondVal zero

  let loopHdr ← declareBlock [.i64, .i64]
  let acc := loopHdr.param 0
  let pos := loopHdr.param 1
  let loopBody ← declareBlock []
  let loopCheckNl ← declareBlock [.i64, .i64, .i64]
  let checkNlAcc := loopCheckNl.param 0
  let checkNlPos := loopCheckNl.param 1
  let chNl := loopCheckNl.param 2
  let loopCheckPlus ← declareBlock [.i64, .i64, .i64]
  let checkPlusAcc := loopCheckPlus.param 0
  let checkPlusPos := loopCheckPlus.param 1
  let chPlus := loopCheckPlus.param 2
  let plusBlk ← declareBlock [.i64, .i64]
  let plusAcc := plusBlk.param 0
  let plusPos := plusBlk.param 1
  let done ← declareBlock [.i64]

  jump loopHdr.ref [first, pos1]

  startBlock loopHdr
  let atEnd ← icmp .uge pos len
  brif atEnd done.ref [acc] loopBody.ref []

  startBlock loopBody
  let ch ← loadInputByte ptr pos
  let isSp ← icmp .eq ch sp
  let nextPos ← iadd pos one
  brif isSp loopHdr.ref [acc, nextPos] loopCheckNl.ref [acc, pos, ch]

  startBlock loopCheckNl
  let isNl ← icmp .eq chNl nl
  let nextPos ← iadd checkNlPos one
  brif isNl loopHdr.ref [checkNlAcc, nextPos] loopCheckPlus.ref [checkNlAcc, checkNlPos, chNl]

  startBlock loopCheckPlus
  let isPlus ← icmp .eq chPlus plus
  brif isPlus plusBlk.ref [checkPlusAcc, checkPlusPos] done.ref [checkPlusAcc]

  startBlock plusBlk
  let nextPos ← iadd plusPos one
  let (term, termEnd) ← emitParseInt ptr nextPos len
  fldStore ptr f.secondVal term
  let nextAcc ← iadd plusAcc term
  jump loopHdr.ref [nextAcc, termEnd]

  startBlock done
  pure (done.param 0)

def clifIrSource : String := buildProgram do
  let fnRead ← AlgorithmLib.IR.declareStdinReadline
  let fnWrite ← AlgorithmLib.IR.declareStdoutWrite
  let ptr ← entryBlock
  let inputOff ← fldOffset f.input
  let inputMax ← iconst64 256
  let bytesRead ← call fnRead [ptr, inputOff, inputMax]
  fldStore ptr f.inputLen bytesRead
  let sum ← emitParseAddChain ptr bytesRead
  fldStore ptr f.result sum
  let outLen ← emitFormatSigned ptr sum
  fldStore ptr f.outputLen outLen
  let outOff ← fldOffset f.output
  let _ ← call fnWrite [ptr, outOff, outLen]
  ret

def payloads : List UInt8 := mkPayload layoutMeta.totalSize []

def cliConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := layoutMeta.totalSize,
  context_offset := 0,
  initial_memory := payloads
}

def cliAlgorithm : Algorithm := {
  actions := [IR.clifCallAction],
  cranelift_units := 0,
  timeout_ms := some 1000
}

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.cliConfig Algorithm.cliAlgorithm
  IO.println (Json.compress json)
