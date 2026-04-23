import AlgorithmLib

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout

namespace Algorithm

structure Fields where
  reserved : Fld (.bytes 64)
  ptx : Fld (.bytes 512)
  bindDesc : Fld (.bytes 12)
  zeroBuf : Fld .i32
  litBuf : Fld .i32
  accBuf : Fld .i32
  outBuf : Fld .i32
  varPresent : Fld (.bytes 26)
  varBufIds : Fld (.bytes 104)
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
  let ptx ← field (.bytes 512)
  let bindDesc ← field (.bytes 12)
  let zeroBuf ← field .i32
  let litBuf ← field .i32
  let accBuf ← field .i32
  let outBuf ← field .i32
  let varPresent ← field (.bytes 26)
  let varBufIds ← field (.bytes 104)
  let input ← field (.bytes 256)
  let scratch ← field (.bytes 32)
  let output ← field (.bytes 32)
  let inputLen ← field .i64
  let firstVal ← field .i64
  let secondVal ← field .i64
  let result ← field .i64
  let outputLen ← field .i64
  pure { reserved, ptx, bindDesc, zeroBuf, litBuf, accBuf, outBuf, varPresent, varBufIds, input, scratch, output, inputLen, firstVal, secondVal, result, outputLen }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

open AlgorithmLib.IR

def asciiSpace : Int := 32
def asciiNewline : Int := 10
def asciiMinus : Int := 45
def asciiPlus : Int := 43
def asciiEq : Int := 61
def asciiZero : Int := 48
def asciiNine : Int := 57
def asciiA : Int := 97
def asciiZ : Int := 122

def scalarAddPtx : String :=
  ".version 7.0\n" ++
  ".target sm_50\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 lhs_ptr,\n" ++
  "    .param .u64 rhs_ptr,\n" ++
  "    .param .u64 out_ptr\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .u64 %lhs, %rhs, %out;\n" ++
  "    .reg .u64 %a, %b, %sum;\n" ++
  "\n" ++
  "    ld.param.u64 %lhs, [lhs_ptr];\n" ++
  "    ld.param.u64 %rhs, [rhs_ptr];\n" ++
  "    ld.param.u64 %out, [out_ptr];\n" ++
  "\n" ++
  "    ld.global.u64 %a, [%lhs];\n" ++
  "    ld.global.u64 %b, [%rhs];\n" ++
  "    add.s64 %sum, %a, %b;\n" ++
  "    st.global.u64 [%out], %sum;\n" ++
  "\n" ++
  "    ret;\n" ++
  "}\n"

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

def emitIsVarChar (ch : Val) : IRBuilder Val := do
  let a ← iconst64 asciiA
  let z ← iconst64 asciiZ
  let geA ← icmp .uge ch a
  let leZ ← icmp .ule ch z
  band geA leZ

def loadVarPresent (ptr idx : Val) : IRBuilder Val :=
  loadByteAt ptr f.varPresent.offset idx

def storeVarPresent (ptr idx value : Val) : IRBuilder Unit :=
  storeByteAt ptr f.varPresent.offset idx value

def loadVarBufId (ptr idx : Val) : IRBuilder Val := do
  let four ← iconst64 4
  let base ← iconst64 f.varBufIds.offset
  let byteOff ← imul idx four
  let rel ← iadd base byteOff
  let addr ← iadd ptr rel
  uload32_64 addr

def storeVarBufId (ptr idx bufId32 : Val) : IRBuilder Unit := do
  let four ← iconst64 4
  let base ← iconst64 f.varBufIds.offset
  let byteOff ← imul idx four
  let rel ← iadd base byteOff
  let addr ← iadd ptr rel
  store bufId32 addr

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

def emitCudaLaunchAdd (cuda : CudaSetup) (ptr lhsBuf rhsBuf outBuf : Val) : IRBuilder Unit := do
  let ptxOff ← fldOffset f.ptx
  let bindOff ← fldOffset f.bindDesc
  let nBufs ← iconst32 3
  let one32 ← iconst32 1
  fldStore32At ptr f.bindDesc 0 lhsBuf
  fldStore32At ptr f.bindDesc 4 rhsBuf
  fldStore32At ptr f.bindDesc 8 outBuf
  let _ ← call cuda.fnLaunch [ptr, ptxOff, nBufs, bindOff, one32, one32, one32, one32, one32, one32]
  pure ()

def emitUploadLiteralToBuf (cuda : CudaSetup) (ptr bufId value : Val) : IRBuilder Unit := do
  fldStore ptr f.firstVal value
  let size8 ← iconst64 8
  let valOff ← fldOffset f.firstVal
  let _ ← call cuda.fnUpload [ptr, bufId, valOff, size8]
  pure ()

def emitAccFromLiteral (cuda : CudaSetup) (ptr value : Val) : IRBuilder Unit := do
  let bufAcc64 ← fldLoad ptr f.accBuf
  let bufAcc ← ireduce32 bufAcc64
  emitUploadLiteralToBuf cuda ptr bufAcc value

def emitAccFromVar (cuda : CudaSetup) (ptr varIdx : Val) : IRBuilder Unit := do
  let zeroBuf ← ireduce32 (← fldLoad ptr f.zeroBuf)
  let accBuf ← ireduce32 (← fldLoad ptr f.accBuf)
  let varBuf ← ireduce32 (← loadVarBufId ptr varIdx)
  emitCudaLaunchAdd cuda ptr zeroBuf varBuf accBuf

def emitTermToLiteralBuf (cuda : CudaSetup) (ptr value : Val) : IRBuilder Val := do
  let litBuf64 ← fldLoad ptr f.litBuf
  let litBuf ← ireduce32 litBuf64
  emitUploadLiteralToBuf cuda ptr litBuf value
  pure litBuf

def emitTermParse (ptr start len : Val) : IRBuilder (Val × Val × Val) := do
  let zero ← iconst64 0
  let pos0 ← emitSkipWs ptr start len

  let hdr ← declareBlock [.i64]
  let pos := hdr.param 0
  let readBlk ← declareBlock []
  let varBlk ← declareBlock [.i64]
  let varIdx := varBlk.param 0
  let intBlk ← declareBlock []
  let done ← declareBlock [.i64, .i64, .i64]

  jump hdr.ref [pos0]

  startBlock hdr
  let atEnd ← icmp .uge pos len
  brif atEnd done.ref [zero, zero, pos] readBlk.ref []

  startBlock readBlk
  let ch ← loadInputByte ptr pos
  let isVar ← emitIsVarChar ch
  let a ← iconst64 asciiA
  let idx ← isub ch a
  brif isVar varBlk.ref [idx] intBlk.ref []

  startBlock varBlk
  let one ← iconst64 1
  let nextPos ← iadd pos one
  jump done.ref [one, varIdx, nextPos]

  startBlock intBlk
  let (v, p) ← emitParseInt ptr pos len
  jump done.ref [zero, v, p]

  startBlock done
  pure (done.param 0, done.param 1, done.param 2)

def emitParseAddChain (cuda : CudaSetup) (ptr start len : Val) : IRBuilder Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let plus ← iconst64 asciiPlus
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline

  let (firstIsVar, firstPayload, pos1) ← emitTermParse ptr start len
  let firstIsVarNonzero ← icmp .ne firstIsVar zero
  let firstVarBlk ← declareBlock [.i64]
  let firstVarIdx := firstVarBlk.param 0
  let firstLitBlk ← declareBlock [.i64]
  let firstLitVal := firstLitBlk.param 0

  let loopHdr ← declareBlock [.i64, .i64]
  let haveAcc := loopHdr.param 0
  let pos := loopHdr.param 1
  let loopBody ← declareBlock []
  let loopCheckNl ← declareBlock [.i64, .i64, .i64]
  let checkNlHave := loopCheckNl.param 0
  let checkNlPos := loopCheckNl.param 1
  let chNl := loopCheckNl.param 2
  let loopCheckPlus ← declareBlock [.i64, .i64, .i64]
  let checkPlusHave := loopCheckPlus.param 0
  let checkPlusPos := loopCheckPlus.param 1
  let chPlus := loopCheckPlus.param 2
  let plusBlk ← declareBlock [.i64, .i64]
  let plusHave := plusBlk.param 0
  let plusPos := plusBlk.param 1
  let done ← declareBlock []

  brif firstIsVarNonzero firstVarBlk.ref [firstPayload] firstLitBlk.ref [firstPayload]

  startBlock firstVarBlk
  emitAccFromVar cuda ptr firstVarIdx
  jump loopHdr.ref [one, pos1]

  startBlock firstLitBlk
  emitAccFromLiteral cuda ptr firstLitVal
  jump loopHdr.ref [one, pos1]

  startBlock loopHdr
  let atEnd ← icmp .uge pos len
  brif atEnd done.ref [] loopBody.ref []

  startBlock loopBody
  let ch ← loadInputByte ptr pos
  let isSp ← icmp .eq ch sp
  let nextPos ← iadd pos one
  brif isSp loopHdr.ref [haveAcc, nextPos] loopCheckNl.ref [haveAcc, pos, ch]

  startBlock loopCheckNl
  let isNl ← icmp .eq chNl nl
  let nextPos ← iadd checkNlPos one
  brif isNl loopHdr.ref [checkNlHave, nextPos] loopCheckPlus.ref [checkNlHave, checkNlPos, chNl]

  startBlock loopCheckPlus
  let isPlus ← icmp .eq chPlus plus
  brif isPlus plusBlk.ref [checkPlusHave, checkPlusPos] done.ref []

  startBlock plusBlk
  let nextPos ← iadd plusPos one
  let (termIsVar, termPayload, termEnd) ← emitTermParse ptr nextPos len
  let termIsVarNonzero ← icmp .ne termIsVar zero
  let termVarBlk ← declareBlock [.i64]
  let termVarIdx := termVarBlk.param 0
  let termLitBlk ← declareBlock [.i64]
  let termLitVal := termLitBlk.param 0

  brif termIsVarNonzero termVarBlk.ref [termPayload] termLitBlk.ref [termPayload]

  startBlock termVarBlk
  let accBuf ← ireduce32 (← fldLoad ptr f.accBuf)
  let outBuf ← ireduce32 (← fldLoad ptr f.outBuf)
  let rhsBuf ← ireduce32 (← loadVarBufId ptr termVarIdx)
  emitCudaLaunchAdd cuda ptr accBuf rhsBuf outBuf
  let zeroBuf ← ireduce32 (← fldLoad ptr f.zeroBuf)
  emitCudaLaunchAdd cuda ptr zeroBuf outBuf accBuf
  jump loopHdr.ref [plusHave, termEnd]

  startBlock termLitBlk
  let litBuf ← emitTermToLiteralBuf cuda ptr termLitVal
  let accBuf ← ireduce32 (← fldLoad ptr f.accBuf)
  let outBuf ← ireduce32 (← fldLoad ptr f.outBuf)
  emitCudaLaunchAdd cuda ptr accBuf litBuf outBuf
  let zeroBuf ← ireduce32 (← fldLoad ptr f.zeroBuf)
  emitCudaLaunchAdd cuda ptr zeroBuf outBuf accBuf
  jump loopHdr.ref [plusHave, termEnd]

  startBlock done
  pure ()

def emitDownloadAccToResult (cuda : CudaSetup) (ptr : Val) : IRBuilder Val := do
  let accBuf64 ← fldLoad ptr f.accBuf
  let accBuf ← ireduce32 accBuf64
  let size8 ← iconst64 8
  let outOff ← fldOffset f.result
  let _ ← call cuda.fnDownload [ptr, accBuf, outOff, size8]
  fldLoad ptr f.result

def emitEvalLine (cuda : CudaSetup) (ptr len : Val) : IRBuilder (Val × Val) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let eqCh ← iconst64 asciiEq

  let pos0 ← emitSkipWs ptr zero len

  let hdr ← declareBlock [.i64]
  let pos := hdr.param 0
  let readBlk ← declareBlock []
  let exprBlk ← declareBlock [.i64]
  let exprStart := exprBlk.param 0
  let varStartBlk ← declareBlock [.i64, .i64]
  let varIdx := varStartBlk.param 0
  let nameEnd := varStartBlk.param 1
  let bareVarBlk ← declareBlock [.i64]
  let bareVarIdx := bareVarBlk.param 0
  let eqReadBlk ← declareBlock [.i64, .i64]
  let eqVarIdx := eqReadBlk.param 0
  let eqPos := eqReadBlk.param 1
  let assignBlk ← declareBlock [.i64, .i64]
  let assignVarIdx := assignBlk.param 0
  let assignEqPos := assignBlk.param 1
  let done ← declareBlock [.i64, .i64]

  jump hdr.ref [pos0]

  startBlock hdr
  let atEnd ← icmp .uge pos len
  brif atEnd done.ref [zero, zero] readBlk.ref []

  startBlock readBlk
  let ch ← loadInputByte ptr pos
  let isVar ← emitIsVarChar ch
  let a ← iconst64 asciiA
  let idx ← isub ch a
  let nextPos ← iadd pos one
  brif isVar varStartBlk.ref [idx, nextPos] exprBlk.ref [pos]

  startBlock exprBlk
  emitParseAddChain cuda ptr exprStart len
  let value ← emitDownloadAccToResult cuda ptr
  jump done.ref [value, one]

  startBlock varStartBlk
  let eqPos2 ← emitSkipWs ptr nameEnd len
  let eqAtEnd ← icmp .uge eqPos2 len
  brif eqAtEnd bareVarBlk.ref [varIdx] eqReadBlk.ref [varIdx, eqPos2]

  startBlock bareVarBlk
  emitAccFromVar cuda ptr bareVarIdx
  let value ← emitDownloadAccToResult cuda ptr
  jump done.ref [value, one]

  startBlock eqReadBlk
  let ch2 ← loadInputByte ptr eqPos
  let isEq ← icmp .eq ch2 eqCh
  brif isEq assignBlk.ref [eqVarIdx, eqPos] exprBlk.ref [pos0]

  startBlock assignBlk
  let exprPos ← iadd assignEqPos one
  emitParseAddChain cuda ptr exprPos len
  let accBuf ← ireduce32 (← fldLoad ptr f.accBuf)
  let zeroBuf ← ireduce32 (← fldLoad ptr f.zeroBuf)
  let varBuf ← ireduce32 (← loadVarBufId ptr assignVarIdx)
  emitCudaLaunchAdd cuda ptr zeroBuf accBuf varBuf
  storeVarPresent ptr assignVarIdx one
  let value ← emitDownloadAccToResult cuda ptr
  jump done.ref [value, zero]

  startBlock done
  pure (done.param 0, done.param 1)

def clifIrSource : String := buildProgram do
  let fnRead ← AlgorithmLib.IR.declareStdinReadline
  let fnWrite ← AlgorithmLib.IR.declareStdoutWrite
  let cuda ← declareCudaFFI
  let ptr ← entryBlock
  let inputOff ← fldOffset f.input
  let inputMax ← iconst64 256
  let outOff ← fldOffset f.output
  let size8 ← iconst64 8
  let zero ← iconst64 0
  let one ← iconst64 1

  callVoid cuda.fnInit [ptr]
  let zeroBuf ← call cuda.fnCreateBuffer [ptr, size8]
  let litBuf ← call cuda.fnCreateBuffer [ptr, size8]
  let accBuf ← call cuda.fnCreateBuffer [ptr, size8]
  let outBuf ← call cuda.fnCreateBuffer [ptr, size8]
  fldStore ptr f.zeroBuf (← sextend64 zeroBuf)
  fldStore ptr f.litBuf (← sextend64 litBuf)
  fldStore ptr f.accBuf (← sextend64 accBuf)
  fldStore ptr f.outBuf (← sextend64 outBuf)

  let setupHdr ← declareBlock [.i64]
  let vi := setupHdr.param 0
  let setupBody ← declareBlock []
  let afterSetup ← declareBlock []
  let twentySix ← iconst64 26
  jump setupHdr.ref [zero]

  startBlock setupHdr
  let doneSetup ← icmp .uge vi twentySix
  brif doneSetup afterSetup.ref [] setupBody.ref []

  startBlock setupBody
  let vbuf ← call cuda.fnCreateBuffer [ptr, size8]
  storeVarBufId ptr vi vbuf
  storeVarPresent ptr vi zero
  let nextI ← iadd vi one
  jump setupHdr.ref [nextI]

  startBlock afterSetup
  let loopHdr ← declareBlock []
  let loopBody ← declareBlock [.i64]
  let bytesRead := loopBody.param 0
  let printBlk ← declareBlock [.i64]
  let lineValue := printBlk.param 0
  let done ← declareBlock []

  jump loopHdr.ref []

  startBlock loopHdr
  let readLen ← call fnRead [ptr, inputOff, inputMax]
  let isEof ← icmp .eq readLen zero
  brif isEof done.ref [] loopBody.ref [readLen]

  startBlock loopBody
  fldStore ptr f.inputLen bytesRead
  let (lineResult, shouldPrint) ← emitEvalLine cuda ptr bytesRead
  fldStore ptr f.result lineResult
  let shouldPrintNow ← icmp .eq shouldPrint one
  brif shouldPrintNow printBlk.ref [lineResult] loopHdr.ref []

  startBlock printBlk
  let outLen ← emitFormatSigned ptr lineValue
  fldStore ptr f.outputLen outLen
  let _ ← call fnWrite [ptr, outOff, outLen]
  jump loopHdr.ref []

  startBlock done
  callVoid cuda.fnCleanup [ptr]
  ret

def payloads : List UInt8 :=
  mkPayload layoutMeta.totalSize [
    f.ptx.init (stringToBytes scalarAddPtx)
  ]

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
