import AlgorithmLib

set_option maxRecDepth 4096

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout

namespace Algorithm

structure Fields where
  reserved : Fld (.bytes 64)
  ptx : Fld (.bytes 512)
  arrayScalePtx : Fld (.bytes 1536)
  arrayAddPtx : Fld (.bytes 1536)
  bindDesc : Fld (.bytes 16)
  zeroBuf : Fld .i32
  litBuf : Fld .i32
  accBuf : Fld .i32
  outBuf : Fld .i32
  tmpArrayBufA : Fld .i32
  tmpArrayBufB : Fld .i32
  tmpArrayBufOut : Fld .i32
  tmpArrayParamBuf : Fld .i32
  varPresent : Fld (.bytes 26)
  varBufIds : Fld (.bytes 104)
  input : Fld (.bytes 256)
  scratch : Fld (.bytes 512)
  output : Fld (.bytes 512)
  varKind : Fld (.bytes 26)
  varTextLens : Fld (.bytes 208)
  varTexts : Fld (.bytes 6656)
  arrayLhsData : Fld (.bytes 2048)
  arrayRhsData : Fld (.bytes 2048)
  arrayOutData : Fld (.bytes 2048)
  arrayParamData : Fld (.bytes 16)
  inputLen : Fld .i64
  firstVal : Fld .i64
  secondVal : Fld .i64
  result : Fld .i64
  outputLen : Fld .i64

def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved ← field (.bytes 64)
  let ptx ← field (.bytes 512)
  let arrayScalePtx ← field (.bytes 1536)
  let arrayAddPtx ← field (.bytes 1536)
  let bindDesc ← field (.bytes 16)
  let zeroBuf ← field .i32
  let litBuf ← field .i32
  let accBuf ← field .i32
  let outBuf ← field .i32
  let tmpArrayBufA ← field .i32
  let tmpArrayBufB ← field .i32
  let tmpArrayBufOut ← field .i32
  let tmpArrayParamBuf ← field .i32
  let varPresent ← field (.bytes 26)
  let varBufIds ← field (.bytes 104)
  let input ← field (.bytes 256)
  let scratch ← field (.bytes 512)
  let output ← field (.bytes 512)
  let varKind ← field (.bytes 26)
  let varTextLens ← field (.bytes 208)
  let varTexts ← field (.bytes 6656)
  let arrayLhsData ← field (.bytes 2048)
  let arrayRhsData ← field (.bytes 2048)
  let arrayOutData ← field (.bytes 2048)
  let arrayParamData ← field (.bytes 16)
  let inputLen ← field .i64
  let firstVal ← field .i64
  let secondVal ← field .i64
  let result ← field .i64
  let outputLen ← field .i64
  pure { reserved, ptx, arrayScalePtx, arrayAddPtx, bindDesc, zeroBuf, litBuf, accBuf, outBuf, tmpArrayBufA, tmpArrayBufB, tmpArrayBufOut, tmpArrayParamBuf, varPresent, varBufIds, input, scratch, output, varKind, varTextLens, varTexts, arrayLhsData, arrayRhsData, arrayOutData, arrayParamData, inputLen, firstVal, secondVal, result, outputLen }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

open AlgorithmLib.IR

def asciiSpace : Int := 32
def asciiNewline : Int := 10
def asciiMinus : Int := 45
def asciiPlus : Int := 43
def asciiEq : Int := 61
def asciiStar : Int := 42
def asciiComma : Int := 44
def asciiLBracket : Int := 91
def asciiRBracket : Int := 93
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

def arrayScalePtxSource : String :=
  ".version 7.0\n" ++
  ".target sm_50\n" ++
  ".address_size 64\n\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 in_ptr,\n" ++
  "    .param .u64 param_ptr,\n" ++
  "    .param .u64 out_ptr\n" ++
  ")\n{\n" ++
  "    .reg .u32 %idx;\n" ++
  "    .reg .u64 %byte, %x, %s, %y;\n" ++
  "    .reg .u64 %in, %param, %out, %addr;\n" ++
  "    ld.param.u64 %in, [in_ptr];\n" ++
  "    ld.param.u64 %param, [param_ptr];\n" ++
  "    ld.param.u64 %out, [out_ptr];\n" ++
  "    mov.u32 %idx, %ctaid.x;\n" ++
  "    cvt.u64.u32 %byte, %idx;\n" ++
  "    shl.b64 %byte, %byte, 3;\n" ++
  "    add.u64 %addr, %in, %byte;\n" ++
  "    ld.global.s64 %x, [%addr];\n" ++
  "    ld.global.s64 %s, [%param];\n" ++
  "    mul.lo.s64 %y, %x, %s;\n" ++
  "    add.u64 %addr, %out, %byte;\n" ++
  "    st.global.s64 [%addr], %y;\n" ++
  "    ret;\n" ++
  "}\n"

def arrayAddPtxSource : String :=
  ".version 7.0\n" ++
  ".target sm_50\n" ++
  ".address_size 64\n\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 lhs_ptr,\n" ++
  "    .param .u64 rhs_ptr,\n" ++
  "    .param .u64 param_ptr,\n" ++
  "    .param .u64 out_ptr\n" ++
  ")\n{\n" ++
  "    .reg .u32 %idx;\n" ++
  "    .reg .u64 %byte, %a, %b, %y;\n" ++
  "    .reg .u64 %lhs, %rhs, %param, %out, %addr;\n" ++
  "    ld.param.u64 %lhs, [lhs_ptr];\n" ++
  "    ld.param.u64 %rhs, [rhs_ptr];\n" ++
  "    ld.param.u64 %param, [param_ptr];\n" ++
  "    ld.param.u64 %out, [out_ptr];\n" ++
  "    mov.u32 %idx, %ctaid.x;\n" ++
  "    cvt.u64.u32 %byte, %idx;\n" ++
  "    shl.b64 %byte, %byte, 3;\n" ++
  "    add.u64 %addr, %lhs, %byte;\n" ++
  "    ld.global.s64 %a, [%addr];\n" ++
  "    add.u64 %addr, %rhs, %byte;\n" ++
  "    ld.global.s64 %b, [%addr];\n" ++
  "    add.s64 %y, %a, %b;\n" ++
  "    add.u64 %addr, %out, %byte;\n" ++
  "    st.global.s64 [%addr], %y;\n" ++
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

def scratchAddr (ptr idx : Val) : IRBuilder Val := do
  let base ← iconst64 f.scratch.offset
  let rel ← iadd base idx
  iadd ptr rel

def scratchI64Addr (ptr baseIdx depth : Val) : IRBuilder Val := do
  let eight ← iconst64 8
  let byteOff ← imul depth eight
  let idx ← iadd baseIdx byteOff
  scratchAddr ptr idx

def loadScratchI64 (ptr baseIdx depth : Val) : IRBuilder Val := do
  let addr ← scratchI64Addr ptr baseIdx depth
  load64 addr

def storeScratchI64 (ptr baseIdx depth value : Val) : IRBuilder Unit := do
  let addr ← scratchI64Addr ptr baseIdx depth
  store value addr

def loadScratchByteDyn (ptr baseIdx depth : Val) : IRBuilder Val := do
  let idx ← iadd baseIdx depth
  loadScratchByte ptr idx

def storeScratchByteDyn (ptr baseIdx depth value : Val) : IRBuilder Unit := do
  let idx ← iadd baseIdx depth
  storeScratchByte ptr idx value

def storeOutputByte (ptr idx value : Val) : IRBuilder Unit :=
  storeByteAt ptr f.output.offset idx value

def storeOutputByteFrom (ptr base idx value : Val) : IRBuilder Unit := do
  let outIdx ← iadd base idx
  storeOutputByte ptr outIdx value

def loadOutputByte (ptr idx : Val) : IRBuilder Val :=
  loadByteAt ptr f.output.offset idx

def loadVarKind (ptr idx : Val) : IRBuilder Val :=
  loadByteAt ptr f.varKind.offset idx

def storeVarKind (ptr idx value : Val) : IRBuilder Unit :=
  storeByteAt ptr f.varKind.offset idx value

def varTextAddr (ptr varIdx textIdx : Val) : IRBuilder Val := do
  let stride ← iconst64 256
  let base ← iconst64 f.varTexts.offset
  let varOff ← imul varIdx stride
  let rel0 ← iadd base varOff
  let rel ← iadd rel0 textIdx
  iadd ptr rel

def loadVarTextByte (ptr varIdx textIdx : Val) : IRBuilder Val := do
  let addr ← varTextAddr ptr varIdx textIdx
  uload8_64 addr

def storeVarTextByte (ptr varIdx textIdx value : Val) : IRBuilder Unit := do
  let addr ← varTextAddr ptr varIdx textIdx
  istore8 value addr

def varTextLenAddr (ptr varIdx : Val) : IRBuilder Val := do
  let eight ← iconst64 8
  let base ← iconst64 f.varTextLens.offset
  let byteOff ← imul varIdx eight
  let rel ← iadd base byteOff
  iadd ptr rel

def loadVarTextLen (ptr varIdx : Val) : IRBuilder Val := do
  let addr ← varTextLenAddr ptr varIdx
  load64 addr

def storeVarTextLen (ptr varIdx value : Val) : IRBuilder Unit := do
  let addr ← varTextLenAddr ptr varIdx
  store value addr

def dataI64Addr (ptr : Val) (baseOff : Nat) (idx : Val) : IRBuilder Val := do
  let eight ← iconst64 8
  let base ← iconst64 baseOff
  let byteOff ← imul idx eight
  let rel ← iadd base byteOff
  iadd ptr rel

def storeDataI64 (ptr : Val) (baseOff : Nat) (idx value : Val) : IRBuilder Unit := do
  let addr ← dataI64Addr ptr baseOff idx
  store value addr

def loadDataI64 (ptr : Val) (baseOff : Nat) (idx : Val) : IRBuilder Val := do
  let addr ← dataI64Addr ptr baseOff idx
  load64 addr

def emitArrayParam (ptr scalar count : Val) : IRBuilder Unit := do
  storeDataI64 ptr f.arrayParamData.offset (← iconst64 0) scalar
  storeDataI64 ptr f.arrayParamData.offset (← iconst64 1) count

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

def emitFormatSignedAt (ptr value startPos : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let ten ← iconst64 10
  let minusCh ← iconst64 asciiMinus
  let zeroCh ← iconst64 asciiZero
  let scratchLast ← iconst64 511

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
  let done ← declareBlock [.i64]

  brif isZero zeroBlk.ref [] nonZeroBlk.ref []

  startBlock zeroBlk
  storeOutputByte ptr startPos zeroCh
  let next ← iadd startPos one
  jump done.ref [next]

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
  brif isNeg signBlk.ref [firstDigitIdx] copyHdr.ref [firstDigitIdx, startPos]

  startBlock signBlk
  storeOutputByte ptr startPos minusCh
  let afterSign ← iadd startPos one
  jump copyHdr.ref [signFirstIdx, afterSign]

  startBlock copyHdr
  let copiedAll ← icmp .ugt copyIdx scratchLast
  brif copiedAll done.ref [outPos] copyBody.ref []

  startBlock copyBody
  let ch ← loadScratchByte ptr copyIdx
  storeOutputByte ptr outPos ch
  let nextIdx ← iadd copyIdx one
  let nextOutPos ← iadd outPos one
  jump copyHdr.ref [nextIdx, nextOutPos]

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

def emitCudaLaunchArrayScale (cuda : CudaSetup) (ptr inBuf paramBuf outBuf count : Val) : IRBuilder Unit := do
  let ptxOff ← fldOffset f.arrayScalePtx
  let bindOff ← fldOffset f.bindDesc
  let nBufs ← iconst32 3
  let count32 ← ireduce32 count
  let one32 ← iconst32 1
  fldStore32At ptr f.bindDesc 0 inBuf
  fldStore32At ptr f.bindDesc 4 paramBuf
  fldStore32At ptr f.bindDesc 8 outBuf
  let _ ← call cuda.fnLaunch [ptr, ptxOff, nBufs, bindOff, count32, one32, one32, one32, one32, one32]
  pure ()

def emitCudaLaunchArrayAdd (cuda : CudaSetup) (ptr lhsBuf rhsBuf paramBuf outBuf count : Val) : IRBuilder Unit := do
  let ptxOff ← fldOffset f.arrayAddPtx
  let bindOff ← fldOffset f.bindDesc
  let nBufs ← iconst32 4
  let count32 ← ireduce32 count
  let one32 ← iconst32 1
  fldStore32At ptr f.bindDesc 0 lhsBuf
  fldStore32At ptr f.bindDesc 4 rhsBuf
  fldStore32At ptr f.bindDesc 8 paramBuf
  fldStore32At ptr f.bindDesc 12 outBuf
  let _ ← call cuda.fnLaunch [ptr, ptxOff, nBufs, bindOff, count32, one32, one32, one32, one32, one32]
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

def emitScalarTermValue (cuda : CudaSetup) (ptr start len : Val) : IRBuilder (Val × Val) := do
  let zero ← iconst64 0
  let pos0 ← emitSkipWs ptr start len

  let hdr ← declareBlock [.i64]
  let pos := hdr.param 0
  let readBlk ← declareBlock []
  let varBlk ← declareBlock [.i64]
  let varIdx := varBlk.param 0
  let intBlk ← declareBlock []
  let done ← declareBlock [.i64, .i64]

  jump hdr.ref [pos0]

  startBlock hdr
  let atEnd ← icmp .uge pos len
  brif atEnd done.ref [zero, pos] readBlk.ref []

  startBlock readBlk
  let ch ← loadInputByte ptr pos
  let isVar ← emitIsVarChar ch
  let a ← iconst64 asciiA
  let idx ← isub ch a
  brif isVar varBlk.ref [idx] intBlk.ref []

  startBlock varBlk
  emitAccFromVar cuda ptr varIdx
  let value ← emitDownloadAccToResult cuda ptr
  let one ← iconst64 1
  let nextPos ← iadd pos one
  jump done.ref [value, nextPos]

  startBlock intBlk
  let (v, p) ← emitParseInt ptr pos len
  jump done.ref [v, p]

  startBlock done
  pure (done.param 0, done.param 1)

def emitParseScalarExpr (cuda : CudaSetup) (ptr start len : Val) : IRBuilder Val := do
  let one ← iconst64 1
  let plus ← iconst64 asciiPlus
  let star ← iconst64 asciiStar
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline

  let (firstVal, firstEnd) ← emitScalarTermValue cuda ptr start len

  let mulHdr ← declareBlock [.i64, .i64]
  let mulAcc := mulHdr.param 0
  let mulPos := mulHdr.param 1
  let mulReadBlk ← declareBlock []
  let mulNlCheckBlk ← declareBlock [.i64, .i64]
  let mulNlPos := mulNlCheckBlk.param 0
  let mulNlCh := mulNlCheckBlk.param 1
  let mulStarCheckBlk ← declareBlock [.i64, .i64]
  let mulStarPos := mulStarCheckBlk.param 0
  let mulStarCh := mulStarCheckBlk.param 1
  let mulStepBlk ← declareBlock [.i64]
  let mulStepPos := mulStepBlk.param 0
  let mulDone ← declareBlock [.i64, .i64]

  let addHdr ← declareBlock [.i64, .i64]
  let addAcc := addHdr.param 0
  let addPos := addHdr.param 1
  let addReadBlk ← declareBlock []
  let addNlCheckBlk ← declareBlock [.i64, .i64]
  let addNlPos := addNlCheckBlk.param 0
  let addNlCh := addNlCheckBlk.param 1
  let addPlusCheckBlk ← declareBlock [.i64, .i64]
  let addPlusPos := addPlusCheckBlk.param 0
  let addPlusCh := addPlusCheckBlk.param 1
  let addStepBlk ← declareBlock [.i64]
  let addStepPos := addStepBlk.param 0
  let done ← declareBlock [.i64]

  jump mulHdr.ref [firstVal, firstEnd]

  startBlock mulHdr
  let pos0 ← emitSkipWs ptr mulPos len
  let atEnd ← icmp .uge pos0 len
  brif atEnd mulDone.ref [mulAcc, pos0] mulReadBlk.ref []

  startBlock mulReadBlk
  let ch ← loadInputByte ptr pos0
  let isSp ← icmp .eq ch sp
  let nextPos ← iadd pos0 one
  brif isSp mulHdr.ref [mulAcc, nextPos] mulNlCheckBlk.ref [pos0, ch]

  startBlock mulNlCheckBlk
  let isNl ← icmp .eq mulNlCh nl
  let nextPos ← iadd mulNlPos one
  brif isNl mulDone.ref [mulAcc, nextPos] mulStarCheckBlk.ref [mulNlPos, mulNlCh]

  startBlock mulStarCheckBlk
  let isStar ← icmp .eq mulStarCh star
  brif isStar mulStepBlk.ref [mulStarPos] mulDone.ref [mulAcc, mulStarPos]

  startBlock mulStepBlk
  let rhsStart ← iadd mulStepPos one
  let (rhs, rhsEnd) ← emitScalarTermValue cuda ptr rhsStart len
  let product ← imul mulAcc rhs
  jump mulHdr.ref [product, rhsEnd]

  startBlock mulDone
  jump addHdr.ref [mulDone.param 0, mulDone.param 1]

  startBlock addHdr
  let pos0 ← emitSkipWs ptr addPos len
  let atEnd ← icmp .uge pos0 len
  brif atEnd done.ref [addAcc] addReadBlk.ref []

  startBlock addReadBlk
  let ch ← loadInputByte ptr pos0
  let isSp ← icmp .eq ch sp
  let nextPos ← iadd pos0 one
  brif isSp addHdr.ref [addAcc, nextPos] addNlCheckBlk.ref [pos0, ch]

  startBlock addNlCheckBlk
  let isNl ← icmp .eq addNlCh nl
  let nextPos ← iadd addNlPos one
  brif isNl done.ref [addAcc] addPlusCheckBlk.ref [addNlPos, addNlCh]

  startBlock addPlusCheckBlk
  let isPlus ← icmp .eq addPlusCh plus
  brif isPlus addStepBlk.ref [addPlusPos] done.ref [addAcc]

  startBlock addStepBlk
  let rhsStart ← iadd addStepPos one
  let (rhsFirstVal, rhsFirstEnd) ← emitScalarTermValue cuda ptr rhsStart len
  let rhsMulHdr ← declareBlock [.i64, .i64]
  let rhsMulAcc := rhsMulHdr.param 0
  let rhsMulPos := rhsMulHdr.param 1
  let rhsMulReadBlk ← declareBlock []
  let rhsMulNlCheckBlk ← declareBlock [.i64, .i64]
  let rhsMulNlPos := rhsMulNlCheckBlk.param 0
  let rhsMulNlCh := rhsMulNlCheckBlk.param 1
  let rhsMulStarCheckBlk ← declareBlock [.i64, .i64]
  let rhsMulStarPos := rhsMulStarCheckBlk.param 0
  let rhsMulStarCh := rhsMulStarCheckBlk.param 1
  let rhsMulStepBlk ← declareBlock [.i64]
  let rhsMulStepPos := rhsMulStepBlk.param 0
  let rhsMulDone ← declareBlock [.i64, .i64]

  jump rhsMulHdr.ref [rhsFirstVal, rhsFirstEnd]

  startBlock rhsMulHdr
  let pos0 ← emitSkipWs ptr rhsMulPos len
  let atEnd ← icmp .uge pos0 len
  brif atEnd rhsMulDone.ref [rhsMulAcc, pos0] rhsMulReadBlk.ref []

  startBlock rhsMulReadBlk
  let ch ← loadInputByte ptr pos0
  let isSp ← icmp .eq ch sp
  let nextPos ← iadd pos0 one
  brif isSp rhsMulHdr.ref [rhsMulAcc, nextPos] rhsMulNlCheckBlk.ref [pos0, ch]

  startBlock rhsMulNlCheckBlk
  let isNl ← icmp .eq rhsMulNlCh nl
  let nextPos ← iadd rhsMulNlPos one
  brif isNl rhsMulDone.ref [rhsMulAcc, nextPos] rhsMulStarCheckBlk.ref [rhsMulNlPos, rhsMulNlCh]

  startBlock rhsMulStarCheckBlk
  let isStar ← icmp .eq rhsMulStarCh star
  brif isStar rhsMulStepBlk.ref [rhsMulStarPos] rhsMulDone.ref [rhsMulAcc, rhsMulStarPos]

  startBlock rhsMulStepBlk
  let nextTermStart ← iadd rhsMulStepPos one
  let (nextTerm, nextTermEnd) ← emitScalarTermValue cuda ptr nextTermStart len
  let product ← imul rhsMulAcc nextTerm
  jump rhsMulHdr.ref [product, nextTermEnd]

  startBlock rhsMulDone
  let sum ← iadd addAcc (rhsMulDone.param 0)
  jump addHdr.ref [sum, rhsMulDone.param 1]

  startBlock done
  pure (done.param 0)

def emitCopyInputToVarText (ptr varIdx start len : Val) : IRBuilder Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let maxText ← iconst64 255
  let nl ← iconst64 asciiNewline
  let pos0 ← emitSkipWs ptr start len

  let hdr ← declareBlock [.i64, .i64]
  let inPos := hdr.param 0
  let outPos := hdr.param 1
  let readBlk ← declareBlock []
  let copyBlk ← declareBlock [.i64]
  let ch := copyBlk.param 0
  let done ← declareBlock [.i64]

  jump hdr.ref [pos0, zero]

  startBlock hdr
  let atEnd ← icmp .uge inPos len
  let full ← icmp .uge outPos maxText
  let stop ← bor atEnd full
  brif stop done.ref [outPos] readBlk.ref []

  startBlock readBlk
  let c ← loadInputByte ptr inPos
  let isNl ← icmp .eq c nl
  brif isNl done.ref [outPos] copyBlk.ref [c]

  startBlock copyBlk
  storeVarTextByte ptr varIdx outPos ch
  let nextIn ← iadd inPos one
  let nextOut ← iadd outPos one
  jump hdr.ref [nextIn, nextOut]

  startBlock done
  storeVarTextLen ptr varIdx (done.param 0)

def emitCopyInputToOutput (ptr start len : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let nl ← iconst64 asciiNewline
  let pos0 ← emitSkipWs ptr start len

  let hdr ← declareBlock [.i64, .i64]
  let inPos := hdr.param 0
  let outPos := hdr.param 1
  let readBlk ← declareBlock []
  let copyBlk ← declareBlock [.i64]
  let ch := copyBlk.param 0
  let done ← declareBlock [.i64]

  jump hdr.ref [pos0, zero]

  startBlock hdr
  let atEnd ← icmp .uge inPos len
  brif atEnd done.ref [outPos] readBlk.ref []

  startBlock readBlk
  let c ← loadInputByte ptr inPos
  let isNl ← icmp .eq c nl
  brif isNl done.ref [outPos] copyBlk.ref [c]

  startBlock copyBlk
  storeOutputByte ptr outPos ch
  let nextIn ← iadd inPos one
  let nextOut ← iadd outPos one
  jump hdr.ref [nextIn, nextOut]

  startBlock done
  storeOutputByte ptr (done.param 0) nl
  iadd (done.param 0) one

def emitCopyVarTextToOutput (ptr varIdx : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let nl ← iconst64 asciiNewline
  let textLen ← loadVarTextLen ptr varIdx

  let hdr ← declareBlock [.i64]
  let i := hdr.param 0
  let body ← declareBlock []
  let done ← declareBlock []

  jump hdr.ref [zero]

  startBlock hdr
  let atEnd ← icmp .uge i textLen
  brif atEnd done.ref [] body.ref []

  startBlock body
  let ch ← loadVarTextByte ptr varIdx i
  storeOutputByte ptr i ch
  let nextI ← iadd i one
  jump hdr.ref [nextI]

  startBlock done
  storeOutputByte ptr textLen nl
  iadd textLen one

def emitCopyOutputToVarText (ptr varIdx outLen : Val) : IRBuilder Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let nl ← iconst64 asciiNewline
  let maxText ← iconst64 255

  let hdr ← declareBlock [.i64]
  let i := hdr.param 0
  let body ← declareBlock []
  let copyBlk ← declareBlock [.i64]
  let ch := copyBlk.param 0
  let done ← declareBlock [.i64]

  jump hdr.ref [zero]

  startBlock hdr
  let atEnd ← icmp .uge i outLen
  let full ← icmp .uge i maxText
  let stop ← bor atEnd full
  brif stop done.ref [i] body.ref []

  startBlock body
  let ch ← loadOutputByte ptr i
  let isNl ← icmp .eq ch nl
  brif isNl done.ref [i] copyBlk.ref [ch]

  startBlock copyBlk
  storeVarTextByte ptr varIdx i ch
  let next ← iadd i one
  jump hdr.ref [next]

  startBlock done
  storeVarTextLen ptr varIdx (done.param 0)

def emitFindInputChar (ptr start len target : Val) : IRBuilder Val := do
  let one ← iconst64 1
  let hdr ← declareBlock [.i64]
  let pos := hdr.param 0
  let readBlk ← declareBlock []
  let done ← declareBlock [.i64]

  jump hdr.ref [start]

  startBlock hdr
  let atEnd ← icmp .uge pos len
  brif atEnd done.ref [len] readBlk.ref []

  startBlock readBlk
  let ch ← loadInputByte ptr pos
  let found ← icmp .eq ch target
  let next ← iadd pos one
  brif found done.ref [pos] hdr.ref [next]

  startBlock done
  pure (done.param 0)

def emitCopyVarTextToInput (ptr varIdx : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let textLen ← loadVarTextLen ptr varIdx

  let hdr ← declareBlock [.i64]
  let i := hdr.param 0
  let body ← declareBlock []
  let done ← declareBlock []

  jump hdr.ref [zero]

  startBlock hdr
  let atEnd ← icmp .uge i textLen
  brif atEnd done.ref [] body.ref []

  startBlock body
  let ch ← loadVarTextByte ptr varIdx i
  storeByteAt ptr f.input.offset i ch
  let next ← iadd i one
  jump hdr.ref [next]

  startBlock done
  pure textLen

def emitClearArrayValidationScratch (ptr : Val) : IRBuilder Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let maxDepth ← iconst64 32
  let countsBase ← iconst64 0
  let expectedBase ← iconst64 128
  let kindBase ← iconst64 256
  let expectedSetBase ← iconst64 288

  let hdr ← declareBlock [.i64]
  let depth := hdr.param 0
  let body ← declareBlock []
  let done ← declareBlock []

  jump hdr.ref [zero]

  startBlock hdr
  let finished ← icmp .uge depth maxDepth
  brif finished done.ref [] body.ref []

  startBlock body
  storeScratchI64 ptr countsBase depth zero
  storeScratchI64 ptr expectedBase depth zero
  storeScratchByteDyn ptr kindBase depth zero
  storeScratchByteDyn ptr expectedSetBase depth zero
  let next ← iadd depth one
  jump hdr.ref [next]

  startBlock done
  pure ()

def emitValidateInputArray (ptr start len : Val) : IRBuilder Val := do
  emitClearArrayValidationScratch ptr

  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
  let maxDepth ← iconst64 31
  let countsBase ← iconst64 0
  let expectedBase ← iconst64 128
  let kindBase ← iconst64 256
  let expectedSetBase ← iconst64 288
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline
  let plus ← iconst64 asciiPlus
  let comma ← iconst64 asciiComma
  let lb ← iconst64 asciiLBracket
  let rb ← iconst64 asciiRBracket
  let star ← iconst64 asciiStar
  let minus ← iconst64 asciiMinus
  let zeroCh ← iconst64 asciiZero
  let nineCh ← iconst64 asciiNine

  let scanHdr ← declareBlock [.i64, .i64]
  let pos := scanHdr.param 0
  let depth := scanHdr.param 1
  let readBlk ← declareBlock []
  let nonSpaceBlk ← declareBlock []
  let nonCommaBlk ← declareBlock []
  let nonLbBlk ← declareBlock []
  let nonRbBlk ← declareBlock []
  let digitUpperBlk ← declareBlock [.i64]
  let digitCh := digitUpperBlk.param 0
  let skipBlk ← declareBlock []

  let openBlk ← declareBlock []
  let openParentBlk ← declareBlock []
  let openParentKindCheck ← declareBlock [.i64]
  let openParentKind := openParentKindCheck.param 0
  let openParentKindExisting ← declareBlock [.i64]
  let openParentKind2 := openParentKindExisting.param 0
  let openParentSetKind ← declareBlock []
  let openParentCountBlk ← declareBlock []
  let openEnterBlk ← declareBlock []

  let closeBlk ← declareBlock []
  let closeExpectedUnsetBlk ← declareBlock [.i64]
  let closeCountA := closeExpectedUnsetBlk.param 0
  let closeExpectedSetBlk ← declareBlock [.i64, .i64]
  let closeCountB := closeExpectedSetBlk.param 0
  let closeExpected := closeExpectedSetBlk.param 1
  let closeAdvanceBlk ← declareBlock []

  let numberStartBlk ← declareBlock []
  let numberKindCheckBlk ← declareBlock [.i64]
  let numberKind := numberKindCheckBlk.param 0
  let numberKindExisting ← declareBlock [.i64]
  let numberKind2 := numberKindExisting.param 0
  let numberSetKind ← declareBlock []
  let numberCountBlk ← declareBlock []
  let numberParseBlk ← declareBlock []

  let afterOuter ← declareBlock [.i64]
  let afterOuterPos := afterOuter.param 0
  let afterOuterRead ← declareBlock []
  let afterOuterNonNl ← declareBlock [.i64]
  let afterOuterCh := afterOuterNonNl.param 0
  let invalid ← declareBlock []
  let valid ← declareBlock []
  let done ← declareBlock [.i64]

  jump scanHdr.ref [start, zero]

  startBlock scanHdr
  let atEnd ← icmp .uge pos len
  brif atEnd invalid.ref [] readBlk.ref []

  startBlock readBlk
  let ch ← loadInputByte ptr pos
  let isSpace ← icmp .eq ch sp
  let isComma ← icmp .eq ch comma
  let isLb ← icmp .eq ch lb
  let isRb ← icmp .eq ch rb
  let isMinus ← icmp .eq ch minus
  brif isSpace skipBlk.ref [] nonSpaceBlk.ref []

  startBlock nonSpaceBlk
  brif isComma skipBlk.ref [] nonCommaBlk.ref []

  startBlock nonCommaBlk
  brif isLb openBlk.ref [] nonLbBlk.ref []

  startBlock nonLbBlk
  brif isRb closeBlk.ref [] nonRbBlk.ref []

  startBlock nonRbBlk
  brif isMinus numberStartBlk.ref [] digitUpperBlk.ref [ch]

  startBlock digitUpperBlk
  let belowZero ← icmp .ult digitCh zeroCh
  let aboveNine ← icmp .ugt digitCh nineCh
  let digitOkBlk ← declareBlock []
  brif belowZero invalid.ref [] digitOkBlk.ref []

  startBlock digitOkBlk
  brif aboveNine invalid.ref [] numberStartBlk.ref []

  startBlock skipBlk
  let next ← iadd pos one
  jump scanHdr.ref [next, depth]

  startBlock openBlk
  let tooDeep ← icmp .uge depth maxDepth
  let atOuter ← icmp .eq depth zero
  let openDepthOkBlk ← declareBlock []
  brif tooDeep invalid.ref [] openDepthOkBlk.ref []

  startBlock openDepthOkBlk
  brif atOuter openEnterBlk.ref [] openParentBlk.ref []

  startBlock openParentBlk
  let k ← loadScratchByteDyn ptr kindBase depth
  jump openParentKindCheck.ref [k]

  startBlock openParentKindCheck
  let isNone ← icmp .eq openParentKind zero
  brif isNone openParentSetKind.ref [] openParentKindExisting.ref [openParentKind]

  startBlock openParentKindExisting
  let isArrayKind ← icmp .eq openParentKind2 two
  brif isArrayKind openParentCountBlk.ref [] invalid.ref []

  startBlock openParentSetKind
  storeScratchByteDyn ptr kindBase depth two
  jump openParentCountBlk.ref []

  startBlock openParentCountBlk
  let c ← loadScratchI64 ptr countsBase depth
  let c1 ← iadd c one
  storeScratchI64 ptr countsBase depth c1
  jump openEnterBlk.ref []

  startBlock openEnterBlk
  let nextDepth ← iadd depth one
  storeScratchI64 ptr countsBase nextDepth zero
  storeScratchByteDyn ptr kindBase nextDepth zero
  let nextPos ← iadd pos one
  jump scanHdr.ref [nextPos, nextDepth]

  startBlock numberStartBlk
  let atTop ← icmp .eq depth zero
  brif atTop invalid.ref [] numberKindCheckBlk.ref [(← loadScratchByteDyn ptr kindBase depth)]

  startBlock numberKindCheckBlk
  let isNone ← icmp .eq numberKind zero
  brif isNone numberSetKind.ref [] numberKindExisting.ref [numberKind]

  startBlock numberKindExisting
  let isScalarKind ← icmp .eq numberKind2 one
  brif isScalarKind numberCountBlk.ref [] invalid.ref []

  startBlock numberSetKind
  storeScratchByteDyn ptr kindBase depth one
  jump numberCountBlk.ref []

  startBlock numberCountBlk
  let c ← loadScratchI64 ptr countsBase depth
  let c1 ← iadd c one
  storeScratchI64 ptr countsBase depth c1
  jump numberParseBlk.ref []

  startBlock numberParseBlk
  let (_, nextPos) ← emitParseInt ptr pos len
  jump scanHdr.ref [nextPos, depth]

  startBlock closeBlk
  let atTop ← icmp .eq depth zero
  let closeDepthOkBlk ← declareBlock []
  brif atTop invalid.ref [] closeDepthOkBlk.ref []

  startBlock closeDepthOkBlk
  let count ← loadScratchI64 ptr countsBase depth
  let expectedSet ← loadScratchByteDyn ptr expectedSetBase depth
  let setAlready ← icmp .eq expectedSet one
  brif setAlready closeExpectedSetBlk.ref [count, (← loadScratchI64 ptr expectedBase depth)] closeExpectedUnsetBlk.ref [count]

  startBlock closeExpectedUnsetBlk
  storeScratchI64 ptr expectedBase depth closeCountA
  storeScratchByteDyn ptr expectedSetBase depth one
  jump closeAdvanceBlk.ref []

  startBlock closeExpectedSetBlk
  let same ← icmp .eq closeCountB closeExpected
  brif same closeAdvanceBlk.ref [] invalid.ref []

  startBlock closeAdvanceBlk
  let newDepth ← isub depth one
  let nextPos ← iadd pos one
  let doneOuter ← icmp .eq newDepth zero
  brif doneOuter afterOuter.ref [nextPos] scanHdr.ref [nextPos, newDepth]

  startBlock afterOuter
  let atEnd ← icmp .uge afterOuterPos len
  brif atEnd valid.ref [] afterOuterRead.ref []

  startBlock afterOuterRead
  let ch ← loadInputByte ptr afterOuterPos
  let isSpace ← icmp .eq ch sp
  let isNl ← icmp .eq ch nl
  let isStar ← icmp .eq ch star
  let isPlus ← icmp .eq ch plus
  let nextPos ← iadd afterOuterPos one
  brif isSpace afterOuter.ref [nextPos] afterOuterNonNl.ref [ch]

  startBlock afterOuterNonNl
  let afterOuterNotNl ← declareBlock []
  brif isNl valid.ref [] afterOuterNotNl.ref []

  startBlock afterOuterNotNl
  let isArrayOp ← bor isStar isPlus
  brif isArrayOp valid.ref [] invalid.ref []

  startBlock invalid
  jump done.ref [zero]

  startBlock valid
  jump done.ref [one]

  startBlock done
  pure (done.param 0)

def emitFinishOutputLineTrimSpaces (ptr outPos : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline

  let hdr ← declareBlock [.i64]
  let pos := hdr.param 0
  let checkBlk ← declareBlock [.i64]
  let prev := checkBlk.param 0
  let done ← declareBlock [.i64]

  jump hdr.ref [outPos]

  startBlock hdr
  let atStart ← icmp .eq pos zero
  let prevPos ← isub pos one
  brif atStart done.ref [pos] checkBlk.ref [prevPos]

  startBlock checkBlk
  let ch ← loadOutputByte ptr prev
  let isSpace ← icmp .eq ch sp
  brif isSpace hdr.ref [prev] done.ref [pos]

  startBlock done
  storeOutputByte ptr (done.param 0) nl
  iadd (done.param 0) one

def emitParseInputArrayNumbersToData (ptr start len : Val) (baseOff : Nat) : IRBuilder Val := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let nl ← iconst64 asciiNewline
  let plus ← iconst64 asciiPlus
  let star ← iconst64 asciiStar
  let minus ← iconst64 asciiMinus
  let zeroCh ← iconst64 asciiZero
  let nineCh ← iconst64 asciiNine

  let hdr ← declareBlock [.i64, .i64]
  let pos := hdr.param 0
  let count := hdr.param 1
  let readBlk ← declareBlock []
  let digitCheckBlk ← declareBlock [.i64]
  let chDigit := digitCheckBlk.param 0
  let numberBlk ← declareBlock []
  let skipBlk ← declareBlock []
  let done ← declareBlock [.i64]

  jump hdr.ref [start, zero]

  startBlock hdr
  let atEnd ← icmp .uge pos len
  brif atEnd done.ref [count] readBlk.ref []

  startBlock readBlk
  let ch ← loadInputByte ptr pos
  let isNl ← icmp .eq ch nl
  let isPlus ← icmp .eq ch plus
  let isStar ← icmp .eq ch star
  let stop0 ← bor isNl isPlus
  let stop ← bor stop0 isStar
  let isMinus ← icmp .eq ch minus
  brif stop done.ref [count] digitCheckBlk.ref [ch]

  startBlock digitCheckBlk
  let belowZero ← icmp .ult chDigit zeroCh
  let aboveNine ← icmp .ugt chDigit nineCh
  let maybeDigitBlk ← declareBlock []
  brif isMinus numberBlk.ref [] maybeDigitBlk.ref []

  startBlock maybeDigitBlk
  let notDigit ← bor belowZero aboveNine
  brif notDigit skipBlk.ref [] numberBlk.ref []

  startBlock numberBlk
  let (value, nextPos) ← emitParseInt ptr pos len
  storeDataI64 ptr baseOff count value
  let nextCount ← iadd count one
  jump hdr.ref [nextPos, nextCount]

  startBlock skipBlk
  let nextPos ← iadd pos one
  jump hdr.ref [nextPos, count]

  startBlock done
  pure (done.param 0)

def emitFormatInputArrayShapeFromOutData (ptr start len count : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let nl ← iconst64 asciiNewline
  let plus ← iconst64 asciiPlus
  let star ← iconst64 asciiStar
  let minus ← iconst64 asciiMinus
  let zeroCh ← iconst64 asciiZero
  let nineCh ← iconst64 asciiNine

  let hdr ← declareBlock [.i64, .i64, .i64]
  let inPos := hdr.param 0
  let outPos := hdr.param 1
  let dataIdx := hdr.param 2
  let readBlk ← declareBlock []
  let digitCheckBlk ← declareBlock [.i64]
  let chDigit := digitCheckBlk.param 0
  let numberBlk ← declareBlock []
  let copyBlk ← declareBlock [.i64]
  let copyCh := copyBlk.param 0
  let done ← declareBlock [.i64]

  jump hdr.ref [start, zero, zero]

  startBlock hdr
  let atEnd ← icmp .uge inPos len
  brif atEnd done.ref [outPos] readBlk.ref []

  startBlock readBlk
  let ch ← loadInputByte ptr inPos
  let isNl ← icmp .eq ch nl
  let isPlus ← icmp .eq ch plus
  let isStar ← icmp .eq ch star
  let stop0 ← bor isNl isPlus
  let stop ← bor stop0 isStar
  let isMinus ← icmp .eq ch minus
  brif stop done.ref [outPos] digitCheckBlk.ref [ch]

  startBlock digitCheckBlk
  let belowZero ← icmp .ult chDigit zeroCh
  let aboveNine ← icmp .ugt chDigit nineCh
  let maybeDigitBlk ← declareBlock []
  brif isMinus numberBlk.ref [] maybeDigitBlk.ref []

  startBlock maybeDigitBlk
  let notDigit ← bor belowZero aboveNine
  brif notDigit copyBlk.ref [chDigit] numberBlk.ref []

  startBlock numberBlk
  let dataDone ← icmp .uge dataIdx count
  let badBlk ← declareBlock []
  let numberWriteBlk ← declareBlock []
  brif dataDone badBlk.ref [] numberWriteBlk.ref []

  startBlock numberWriteBlk
  let value ← loadDataI64 ptr f.arrayOutData.offset dataIdx
  let nextOut ← emitFormatSignedAt ptr value outPos
  let (_, nextIn) ← emitParseInt ptr inPos len
  let nextDataIdx ← iadd dataIdx one
  jump hdr.ref [nextIn, nextOut, nextDataIdx]

  startBlock badBlk
  jump done.ref [zero]

  startBlock copyBlk
  storeOutputByte ptr outPos copyCh
  let nextIn ← iadd inPos one
  let nextOut ← iadd outPos one
  jump hdr.ref [nextIn, nextOut, dataIdx]

  startBlock done
  emitFinishOutputLineTrimSpaces ptr (done.param 0)

def emitUploadInputArrayToVarBuffer (cuda : CudaSetup) (ptr varIdx start len : Val) : IRBuilder Unit := do
  let _count ← emitParseInputArrayNumbersToData ptr start len f.arrayLhsData.offset
  let bytes ← iconst64 2048
  let dataOff ← fldOffset f.arrayLhsData
  let varBuf ← ireduce32 (← loadVarBufId ptr varIdx)
  let _ ← call cuda.fnUpload [ptr, varBuf, dataOff, bytes]
  pure ()

def emitUploadOutDataToVarBuffer (cuda : CudaSetup) (ptr varIdx : Val) : IRBuilder Unit := do
  let bytes ← iconst64 2048
  let dataOff ← fldOffset f.arrayOutData
  let varBuf ← ireduce32 (← loadVarBufId ptr varIdx)
  let _ ← call cuda.fnUpload [ptr, varBuf, dataOff, bytes]
  pure ()

def emitScaleInputArrayToOutput (cuda : CudaSetup) (ptr arrStart len scalar : Val) : IRBuilder Val := do
  let count ← emitParseInputArrayNumbersToData ptr arrStart len f.arrayLhsData.offset
  emitArrayParam ptr scalar count
  let bytes ← iconst64 2048
  let lhsOff ← fldOffset f.arrayLhsData
  let paramOff ← fldOffset f.arrayParamData
  let outOff ← fldOffset f.arrayOutData
  let lhsBuf ← ireduce32 (← fldLoad ptr f.tmpArrayBufA)
  let paramBuf ← ireduce32 (← fldLoad ptr f.tmpArrayParamBuf)
  let outBuf ← ireduce32 (← fldLoad ptr f.tmpArrayBufOut)
  let paramBytes ← iconst64 16
  let _ ← call cuda.fnUpload [ptr, lhsBuf, lhsOff, bytes]
  let _ ← call cuda.fnUpload [ptr, paramBuf, paramOff, paramBytes]
  emitCudaLaunchArrayScale cuda ptr lhsBuf paramBuf outBuf count
  let _ ← call cuda.fnDownload [ptr, outBuf, outOff, bytes]
  emitFormatInputArrayShapeFromOutData ptr arrStart len count

def emitAddInputArraysToOutput (cuda : CudaSetup) (ptr lhsStart rhsStart len : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  let lhsCount ← emitParseInputArrayNumbersToData ptr lhsStart len f.arrayLhsData.offset
  let rhsCount ← emitParseInputArrayNumbersToData ptr rhsStart len f.arrayRhsData.offset
  let sameCount ← icmp .eq lhsCount rhsCount
  let runBlk ← declareBlock [.i64]
  let count := runBlk.param 0
  let done ← declareBlock [.i64]
  brif sameCount runBlk.ref [lhsCount] done.ref [zero]

  startBlock runBlk
  emitArrayParam ptr zero count
  let bytes ← iconst64 2048
  let lhsOff ← fldOffset f.arrayLhsData
  let rhsOff ← fldOffset f.arrayRhsData
  let paramOff ← fldOffset f.arrayParamData
  let outOff ← fldOffset f.arrayOutData
  let lhsBuf ← ireduce32 (← fldLoad ptr f.tmpArrayBufA)
  let rhsBuf ← ireduce32 (← fldLoad ptr f.tmpArrayBufB)
  let paramBuf ← ireduce32 (← fldLoad ptr f.tmpArrayParamBuf)
  let outBuf ← ireduce32 (← fldLoad ptr f.tmpArrayBufOut)
  let paramBytes ← iconst64 16
  let _ ← call cuda.fnUpload [ptr, lhsBuf, lhsOff, bytes]
  let _ ← call cuda.fnUpload [ptr, rhsBuf, rhsOff, bytes]
  let _ ← call cuda.fnUpload [ptr, paramBuf, paramOff, paramBytes]
  emitCudaLaunchArrayAdd cuda ptr lhsBuf rhsBuf paramBuf outBuf count
  let _ ← call cuda.fnDownload [ptr, outBuf, outOff, bytes]
  let outLen ← emitFormatInputArrayShapeFromOutData ptr lhsStart len count
  jump done.ref [outLen]

  startBlock done
  pure (done.param 0)

def emitEvalLine (cuda : CudaSetup) (ptr len : Val) : IRBuilder (Val × Val) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
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
  let bareArrayBlk ← declareBlock [.i64]
  let bareArrayIdx := bareArrayBlk.param 0
  let bareScalarBlk ← declareBlock [.i64]
  let bareScalarIdx := bareScalarBlk.param 0
  let assignArrayCheckBlk ← declareBlock [.i64, .i64, .i64]
  let assignArrayVarIdx := assignArrayCheckBlk.param 0
  let assignArrayExprPos := assignArrayCheckBlk.param 1
  let assignArrayFirstCh := assignArrayCheckBlk.param 2
  let assignArrayBlk ← declareBlock [.i64, .i64]
  let assignArrayIdx := assignArrayBlk.param 0
  let assignArrayPos := assignArrayBlk.param 1
  let assignScalarBlk ← declareBlock [.i64, .i64]
  let assignScalarIdx := assignScalarBlk.param 0
  let assignScalarPos := assignScalarBlk.param 1
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
  let exprPos0 ← emitSkipWs ptr exprStart len
  let exprAtEnd ← icmp .uge exprPos0 len
  let exprFirstCh ← loadInputByte ptr exprPos0
  let lb ← iconst64 asciiLBracket
  let isArrayLiteral ← icmp .eq exprFirstCh lb
  let exprIsVar ← emitIsVarChar exprFirstCh
  let exprArrayLiteralBlk ← declareBlock [.i64]
  let exprArrayStart := exprArrayLiteralBlk.param 0
  let exprVarCheckBlk ← declareBlock [.i64, .i64]
  let exprVarStart := exprVarCheckBlk.param 0
  let exprVarIdx := exprVarCheckBlk.param 1
  let exprScalarBlk ← declareBlock [.i64]
  let exprScalarStart := exprScalarBlk.param 0
  let exprVarArrayBlk ← declareBlock [.i64, .i64]
  let exprArrayVarStart := exprVarArrayBlk.param 0
  let exprArrayVarIdx := exprVarArrayBlk.param 1
  brif isArrayLiteral exprArrayLiteralBlk.ref [exprPos0] exprVarCheckBlk.ref [exprPos0, (← isub exprFirstCh (← iconst64 asciiA))]

  startBlock exprArrayLiteralBlk
  let isValidArray ← emitValidateInputArray ptr exprArrayStart len
  let isValidArrayNow ← icmp .eq isValidArray one
  let exprArrayValidatedBlk ← declareBlock [.i64]
  let exprArrayValidatedStart := exprArrayValidatedBlk.param 0
  brif isValidArrayNow exprArrayValidatedBlk.ref [exprArrayStart] done.ref [zero, zero]

  startBlock exprArrayValidatedBlk
  let star ← iconst64 asciiStar
  let plus ← iconst64 asciiPlus
  let plusPos ← emitFindInputChar ptr exprArrayValidatedStart len plus
  let starPos ← emitFindInputChar ptr exprArrayValidatedStart len star
  let hasPlus ← icmp .ult plusPos len
  let hasStar ← icmp .ult starPos len
  let addLiteralBlk ← declareBlock [.i64, .i64]
  let addLiteralStart := addLiteralBlk.param 0
  let addLiteralPlus := addLiteralBlk.param 1
  let scaleLiteralBlk ← declareBlock [.i64, .i64]
  let scaleLiteralStart := scaleLiteralBlk.param 0
  let scaleLiteralStar := scaleLiteralBlk.param 1
  let copyLiteralBlk ← declareBlock [.i64]
  let copyLiteralStart := copyLiteralBlk.param 0
  let noPlusLiteralBlk ← declareBlock [.i64]
  let noPlusLiteralStart := noPlusLiteralBlk.param 0
  brif hasPlus addLiteralBlk.ref [exprArrayValidatedStart, plusPos] noPlusLiteralBlk.ref [exprArrayValidatedStart]

  startBlock noPlusLiteralBlk
  brif hasStar scaleLiteralBlk.ref [noPlusLiteralStart, starPos] copyLiteralBlk.ref [noPlusLiteralStart]

  startBlock addLiteralBlk
  let rhsStart0 ← iadd addLiteralPlus one
  let rhsStart ← emitSkipWs ptr rhsStart0 len
  let rhsValid ← emitValidateInputArray ptr rhsStart len
  let rhsValidNow ← icmp .eq rhsValid one
  let addLiteralDoBlk ← declareBlock [.i64, .i64]
  let addLiteralDoLhs := addLiteralDoBlk.param 0
  let addLiteralDoRhs := addLiteralDoBlk.param 1
  brif rhsValidNow addLiteralDoBlk.ref [addLiteralStart, rhsStart] done.ref [zero, zero]

  startBlock addLiteralDoBlk
  let outLen ← emitAddInputArraysToOutput cuda ptr addLiteralDoLhs addLiteralDoRhs len
  fldStore ptr f.outputLen outLen
  let dummy ← iconst64 0
  jump done.ref [dummy, two]

  startBlock scaleLiteralBlk
  let scalarPos ← iadd scaleLiteralStar one
  let (scalar, _) ← emitParseInt ptr scalarPos len
  let outLen ← emitScaleInputArrayToOutput cuda ptr scaleLiteralStart len scalar
  fldStore ptr f.outputLen outLen
  let dummy ← iconst64 0
  jump done.ref [dummy, two]

  startBlock copyLiteralBlk
  let outLen ← emitCopyInputToOutput ptr copyLiteralStart len
  fldStore ptr f.outputLen outLen
  let dummy ← iconst64 0
  jump done.ref [dummy, two]

  startBlock exprVarCheckBlk
  let kind ← loadVarKind ptr exprVarIdx
  let isArrayVar ← icmp .eq kind one
  let useArrayVar ← band exprIsVar isArrayVar
  brif useArrayVar exprVarArrayBlk.ref [exprVarStart, exprVarIdx] exprScalarBlk.ref [exprVarStart]

  startBlock exprVarArrayBlk
  let star ← iconst64 asciiStar
  let starPos ← emitFindInputChar ptr exprArrayVarStart len star
  let hasStar ← icmp .ult starPos len
  let scaleVarBlk ← declareBlock [.i64, .i64]
  let scaleVarIdx := scaleVarBlk.param 0
  let scaleVarStar := scaleVarBlk.param 1
  let copyVarBlk ← declareBlock [.i64]
  let copyVarIdx := copyVarBlk.param 0
  brif hasStar scaleVarBlk.ref [exprArrayVarIdx, starPos] copyVarBlk.ref [exprArrayVarIdx]

  startBlock scaleVarBlk
  let scalarPos ← iadd scaleVarStar one
  let (scalar, _) ← emitParseInt ptr scalarPos len
  let varLen ← emitCopyVarTextToInput ptr scaleVarIdx
  let inputStart ← iconst64 0
  let outLen ← emitScaleInputArrayToOutput cuda ptr inputStart varLen scalar
  fldStore ptr f.outputLen outLen
  let dummy ← iconst64 0
  jump done.ref [dummy, two]

  startBlock copyVarBlk
  let outLen ← emitCopyVarTextToOutput ptr copyVarIdx
  fldStore ptr f.outputLen outLen
  let dummy ← iconst64 0
  jump done.ref [dummy, two]

  startBlock exprScalarBlk
  let (lhsValue, lhsEnd) ← emitScalarTermValue cuda ptr exprScalarStart len
  let lhsNext ← emitSkipWs ptr lhsEnd len
  let lhsAtEnd ← icmp .uge lhsNext len
  let exprScalarEvalBlk ← declareBlock [.i64]
  let exprScalarEvalStart := exprScalarEvalBlk.param 0
  let exprScalarMaybeStarBlk ← declareBlock [.i64, .i64]
  let exprScalarLhsValue := exprScalarMaybeStarBlk.param 0
  let exprScalarStarPos := exprScalarMaybeStarBlk.param 1
  brif lhsAtEnd exprScalarEvalBlk.ref [exprScalarStart] exprScalarMaybeStarBlk.ref [lhsValue, lhsNext]

  startBlock exprScalarMaybeStarBlk
  let ch ← loadInputByte ptr exprScalarStarPos
  let star ← iconst64 asciiStar
  let isStar ← icmp .eq ch star
  let exprScalarStarBlk ← declareBlock [.i64, .i64]
  let scalarLeftValue := exprScalarStarBlk.param 0
  let scalarLeftStarPos := exprScalarStarBlk.param 1
  brif isStar exprScalarStarBlk.ref [exprScalarLhsValue, exprScalarStarPos] exprScalarEvalBlk.ref [exprScalarStart]

  startBlock exprScalarStarBlk
  let rhsStart0 ← iadd scalarLeftStarPos one
  let rhsStart ← emitSkipWs ptr rhsStart0 len
  let rhsAtEnd ← icmp .uge rhsStart len
  let scalarLeftRhsCheckBlk ← declareBlock [.i64, .i64]
  let scalarLeftValue2 := scalarLeftRhsCheckBlk.param 0
  let scalarLeftRhsStart := scalarLeftRhsCheckBlk.param 1
  brif rhsAtEnd exprScalarEvalBlk.ref [exprScalarStart] scalarLeftRhsCheckBlk.ref [scalarLeftValue, rhsStart]

  startBlock scalarLeftRhsCheckBlk
  let rhsCh ← loadInputByte ptr scalarLeftRhsStart
  let lb ← iconst64 asciiLBracket
  let isArrayLiteral ← icmp .eq rhsCh lb
  let scalarLeftArrayLiteralBlk ← declareBlock [.i64, .i64]
  let scalarLeftArrayScalar := scalarLeftArrayLiteralBlk.param 0
  let scalarLeftArrayStart := scalarLeftArrayLiteralBlk.param 1
  let scalarLeftVarCheckBlk ← declareBlock [.i64, .i64, .i64]
  let scalarLeftVarScalar := scalarLeftVarCheckBlk.param 0
  let scalarLeftVarStart := scalarLeftVarCheckBlk.param 1
  let scalarLeftVarIdx := scalarLeftVarCheckBlk.param 2
  let a ← iconst64 asciiA
  let rhsVarIdx ← isub rhsCh a
  brif isArrayLiteral scalarLeftArrayLiteralBlk.ref [scalarLeftValue2, scalarLeftRhsStart] scalarLeftVarCheckBlk.ref [scalarLeftValue2, scalarLeftRhsStart, rhsVarIdx]

  startBlock scalarLeftArrayLiteralBlk
  let isValidArray ← emitValidateInputArray ptr scalarLeftArrayStart len
  let isValidArrayNow ← icmp .eq isValidArray one
  let scalarLeftArrayScaleBlk ← declareBlock [.i64, .i64]
  let scalarLeftArrayScaleValue := scalarLeftArrayScaleBlk.param 0
  let scalarLeftArrayScaleStart := scalarLeftArrayScaleBlk.param 1
  brif isValidArrayNow scalarLeftArrayScaleBlk.ref [scalarLeftArrayScalar, scalarLeftArrayStart] done.ref [zero, zero]

  startBlock scalarLeftArrayScaleBlk
  let outLen ← emitScaleInputArrayToOutput cuda ptr scalarLeftArrayScaleStart len scalarLeftArrayScaleValue
  fldStore ptr f.outputLen outLen
  let dummy ← iconst64 0
  jump done.ref [dummy, two]

  startBlock scalarLeftVarCheckBlk
  let isVar ← emitIsVarChar (← loadInputByte ptr scalarLeftVarStart)
  let kind ← loadVarKind ptr scalarLeftVarIdx
  let isArrayVar ← icmp .eq kind one
  let useArrayVar ← band isVar isArrayVar
  let scalarLeftArrayVarBlk ← declareBlock [.i64, .i64]
  let scalarLeftArrayVarScalar := scalarLeftArrayVarBlk.param 0
  let scalarLeftArrayVarIdx := scalarLeftArrayVarBlk.param 1
  brif useArrayVar scalarLeftArrayVarBlk.ref [scalarLeftVarScalar, scalarLeftVarIdx] exprScalarEvalBlk.ref [exprScalarStart]

  startBlock scalarLeftArrayVarBlk
  let varLen ← emitCopyVarTextToInput ptr scalarLeftArrayVarIdx
  let inputStart ← iconst64 0
  let outLen ← emitScaleInputArrayToOutput cuda ptr inputStart varLen scalarLeftArrayVarScalar
  fldStore ptr f.outputLen outLen
  let dummy ← iconst64 0
  jump done.ref [dummy, two]

  startBlock exprScalarEvalBlk
  let value ← emitParseScalarExpr cuda ptr exprScalarEvalStart len
  jump done.ref [value, one]

  startBlock varStartBlk
  let eqPos2 ← emitSkipWs ptr nameEnd len
  let eqAtEnd ← icmp .uge eqPos2 len
  brif eqAtEnd bareVarBlk.ref [varIdx] eqReadBlk.ref [varIdx, eqPos2]

  startBlock bareVarBlk
  let kind ← loadVarKind ptr bareVarIdx
  let isArray ← icmp .eq kind one
  brif isArray bareArrayBlk.ref [bareVarIdx] bareScalarBlk.ref [bareVarIdx]

  startBlock bareArrayBlk
  let outLen ← emitCopyVarTextToOutput ptr bareArrayIdx
  fldStore ptr f.outputLen outLen
  let dummy ← iconst64 0
  jump done.ref [dummy, two]

  startBlock bareScalarBlk
  emitAccFromVar cuda ptr bareScalarIdx
  let value ← emitDownloadAccToResult cuda ptr
  jump done.ref [value, one]

  startBlock eqReadBlk
  let ch2 ← loadInputByte ptr eqPos
  let isEq ← icmp .eq ch2 eqCh
  brif isEq assignBlk.ref [eqVarIdx, eqPos] exprBlk.ref [pos0]

  startBlock assignBlk
  let exprPos ← iadd assignEqPos one
  let exprPos0 ← emitSkipWs ptr exprPos len
  let exprAtEnd ← icmp .uge exprPos0 len
  let exprFirstCh ← loadInputByte ptr exprPos0
  brif exprAtEnd assignScalarBlk.ref [assignVarIdx, exprPos0] assignArrayCheckBlk.ref [assignVarIdx, exprPos0, exprFirstCh]

  startBlock assignArrayCheckBlk
  let lb ← iconst64 asciiLBracket
  let isArrayExpr ← icmp .eq assignArrayFirstCh lb
  brif isArrayExpr assignArrayBlk.ref [assignArrayVarIdx, assignArrayExprPos] assignScalarBlk.ref [assignArrayVarIdx, assignArrayExprPos]

  startBlock assignArrayBlk
  let isValidArray ← emitValidateInputArray ptr assignArrayPos len
  let isValidArrayNow ← icmp .eq isValidArray one
  let assignArrayValidatedBlk ← declareBlock [.i64, .i64]
  let assignArrayValidatedIdx := assignArrayValidatedBlk.param 0
  let assignArrayValidatedPos := assignArrayValidatedBlk.param 1
  brif isValidArrayNow assignArrayValidatedBlk.ref [assignArrayIdx, assignArrayPos] done.ref [zero, zero]

  startBlock assignArrayValidatedBlk
  let star ← iconst64 asciiStar
  let plus ← iconst64 asciiPlus
  let plusPos ← emitFindInputChar ptr assignArrayValidatedPos len plus
  let starPos ← emitFindInputChar ptr assignArrayValidatedPos len star
  let hasPlus ← icmp .ult plusPos len
  let hasStar ← icmp .ult starPos len
  let assignArrayAddStoreBlk ← declareBlock [.i64, .i64, .i64]
  let assignArrayAddStoreIdx := assignArrayAddStoreBlk.param 0
  let assignArrayAddStorePos := assignArrayAddStoreBlk.param 1
  let assignArrayAddStorePlus := assignArrayAddStoreBlk.param 2
  let assignArrayScaleStoreBlk ← declareBlock [.i64, .i64, .i64]
  let assignArrayScaleStoreIdx := assignArrayScaleStoreBlk.param 0
  let assignArrayScaleStorePos := assignArrayScaleStoreBlk.param 1
  let assignArrayScaleStoreStar := assignArrayScaleStoreBlk.param 2
  let assignArrayCopyStoreBlk ← declareBlock [.i64, .i64]
  let assignArrayCopyStoreIdx := assignArrayCopyStoreBlk.param 0
  let assignArrayCopyStorePos := assignArrayCopyStoreBlk.param 1
  let assignArrayNoPlusStoreBlk ← declareBlock [.i64, .i64]
  let assignArrayNoPlusStoreIdx := assignArrayNoPlusStoreBlk.param 0
  let assignArrayNoPlusStorePos := assignArrayNoPlusStoreBlk.param 1
  brif hasPlus assignArrayAddStoreBlk.ref [assignArrayValidatedIdx, assignArrayValidatedPos, plusPos] assignArrayNoPlusStoreBlk.ref [assignArrayValidatedIdx, assignArrayValidatedPos]

  startBlock assignArrayNoPlusStoreBlk
  brif hasStar assignArrayScaleStoreBlk.ref [assignArrayNoPlusStoreIdx, assignArrayNoPlusStorePos, starPos] assignArrayCopyStoreBlk.ref [assignArrayNoPlusStoreIdx, assignArrayNoPlusStorePos]

  startBlock assignArrayAddStoreBlk
  let rhsStart0 ← iadd assignArrayAddStorePlus one
  let rhsStart ← emitSkipWs ptr rhsStart0 len
  let rhsValid ← emitValidateInputArray ptr rhsStart len
  let rhsValidNow ← icmp .eq rhsValid one
  let assignArrayAddDoStoreBlk ← declareBlock [.i64, .i64, .i64]
  let assignArrayAddDoStoreIdx := assignArrayAddDoStoreBlk.param 0
  let assignArrayAddDoStoreLhs := assignArrayAddDoStoreBlk.param 1
  let assignArrayAddDoStoreRhs := assignArrayAddDoStoreBlk.param 2
  brif rhsValidNow assignArrayAddDoStoreBlk.ref [assignArrayAddStoreIdx, assignArrayAddStorePos, rhsStart] done.ref [zero, zero]

  startBlock assignArrayAddDoStoreBlk
  let outLen ← emitAddInputArraysToOutput cuda ptr assignArrayAddDoStoreLhs assignArrayAddDoStoreRhs len
  emitCopyOutputToVarText ptr assignArrayAddDoStoreIdx outLen
  emitUploadOutDataToVarBuffer cuda ptr assignArrayAddDoStoreIdx
  storeVarKind ptr assignArrayAddDoStoreIdx one
  storeVarPresent ptr assignArrayAddDoStoreIdx one
  let dummy ← iconst64 0
  jump done.ref [dummy, zero]

  startBlock assignArrayScaleStoreBlk
  let scalarPos ← iadd assignArrayScaleStoreStar one
  let (scalar, _) ← emitParseInt ptr scalarPos len
  let outLen ← emitScaleInputArrayToOutput cuda ptr assignArrayScaleStorePos len scalar
  emitCopyOutputToVarText ptr assignArrayScaleStoreIdx outLen
  emitUploadOutDataToVarBuffer cuda ptr assignArrayScaleStoreIdx
  storeVarKind ptr assignArrayScaleStoreIdx one
  storeVarPresent ptr assignArrayScaleStoreIdx one
  let dummy ← iconst64 0
  jump done.ref [dummy, zero]

  startBlock assignArrayCopyStoreBlk
  emitCopyInputToVarText ptr assignArrayCopyStoreIdx assignArrayCopyStorePos len
  emitUploadInputArrayToVarBuffer cuda ptr assignArrayCopyStoreIdx assignArrayCopyStorePos len
  storeVarKind ptr assignArrayCopyStoreIdx one
  storeVarPresent ptr assignArrayCopyStoreIdx one
  let dummy ← iconst64 0
  jump done.ref [dummy, zero]

  startBlock assignScalarBlk
  let rhsStart ← emitSkipWs ptr assignScalarPos len
  let rhsAtEnd ← icmp .uge rhsStart len
  let assignScalarEvalStoreBlk ← declareBlock [.i64, .i64]
  let assignScalarEvalIdx := assignScalarEvalStoreBlk.param 0
  let assignScalarEvalPos := assignScalarEvalStoreBlk.param 1
  let assignScalarFirstCheckBlk ← declareBlock [.i64, .i64]
  let assignScalarFirstIdx := assignScalarFirstCheckBlk.param 0
  let assignScalarFirstPos := assignScalarFirstCheckBlk.param 1
  brif rhsAtEnd assignScalarEvalStoreBlk.ref [assignScalarIdx, assignScalarPos] assignScalarFirstCheckBlk.ref [assignScalarIdx, rhsStart]

  startBlock assignScalarFirstCheckBlk
  let firstCh ← loadInputByte ptr assignScalarFirstPos
  let firstIsVar ← emitIsVarChar firstCh
  let a ← iconst64 asciiA
  let firstVarIdx ← isub firstCh a
  let firstKind ← loadVarKind ptr firstVarIdx
  let firstIsArrayVar ← icmp .eq firstKind one
  let useFirstArrayVar ← band firstIsVar firstIsArrayVar
  let assignFirstArrayVarBlk ← declareBlock [.i64, .i64, .i64]
  let assignFirstArrayTargetIdx := assignFirstArrayVarBlk.param 0
  let assignFirstArrayStart := assignFirstArrayVarBlk.param 1
  let assignFirstArrayVarIdx := assignFirstArrayVarBlk.param 2
  let assignMaybeScalarLeftBlk ← declareBlock [.i64, .i64]
  let assignMaybeScalarLeftIdx := assignMaybeScalarLeftBlk.param 0
  let assignMaybeScalarLeftPos := assignMaybeScalarLeftBlk.param 1
  brif useFirstArrayVar assignFirstArrayVarBlk.ref [assignScalarFirstIdx, assignScalarFirstPos, firstVarIdx] assignMaybeScalarLeftBlk.ref [assignScalarFirstIdx, assignScalarFirstPos]

  startBlock assignFirstArrayVarBlk
  let star ← iconst64 asciiStar
  let starPos ← emitFindInputChar ptr assignFirstArrayStart len star
  let hasStar ← icmp .ult starPos len
  let assignFirstArrayScaleBlk ← declareBlock [.i64, .i64, .i64]
  let assignFirstArrayScaleTarget := assignFirstArrayScaleBlk.param 0
  let assignFirstArrayScaleVar := assignFirstArrayScaleBlk.param 1
  let assignFirstArrayScaleStar := assignFirstArrayScaleBlk.param 2
  let assignFirstArrayCopyBlk ← declareBlock [.i64, .i64]
  let assignFirstArrayCopyTarget := assignFirstArrayCopyBlk.param 0
  let assignFirstArrayCopyVar := assignFirstArrayCopyBlk.param 1
  brif hasStar assignFirstArrayScaleBlk.ref [assignFirstArrayTargetIdx, assignFirstArrayVarIdx, starPos] assignFirstArrayCopyBlk.ref [assignFirstArrayTargetIdx, assignFirstArrayVarIdx]

  startBlock assignFirstArrayScaleBlk
  let scalarPos ← iadd assignFirstArrayScaleStar one
  let (scalar, _) ← emitParseInt ptr scalarPos len
  let varLen ← emitCopyVarTextToInput ptr assignFirstArrayScaleVar
  let inputStart ← iconst64 0
  let outLen ← emitScaleInputArrayToOutput cuda ptr inputStart varLen scalar
  emitCopyOutputToVarText ptr assignFirstArrayScaleTarget outLen
  emitUploadOutDataToVarBuffer cuda ptr assignFirstArrayScaleTarget
  storeVarKind ptr assignFirstArrayScaleTarget one
  storeVarPresent ptr assignFirstArrayScaleTarget one
  let dummy ← iconst64 0
  jump done.ref [dummy, zero]

  startBlock assignFirstArrayCopyBlk
  let outLen ← emitCopyVarTextToOutput ptr assignFirstArrayCopyVar
  emitCopyOutputToVarText ptr assignFirstArrayCopyTarget outLen
  storeVarKind ptr assignFirstArrayCopyTarget one
  storeVarPresent ptr assignFirstArrayCopyTarget one
  let dummy ← iconst64 0
  jump done.ref [dummy, zero]

  startBlock assignMaybeScalarLeftBlk
  let (lhsValue, lhsEnd) ← emitScalarTermValue cuda ptr assignMaybeScalarLeftPos len
  let lhsNext ← emitSkipWs ptr lhsEnd len
  let lhsAtEnd ← icmp .uge lhsNext len
  let assignScalarMaybeStarBlk ← declareBlock [.i64, .i64, .i64]
  let assignScalarMaybeStarIdx := assignScalarMaybeStarBlk.param 0
  let assignScalarMaybeStarValue := assignScalarMaybeStarBlk.param 1
  let assignScalarMaybeStarPos := assignScalarMaybeStarBlk.param 2
  brif lhsAtEnd assignScalarEvalStoreBlk.ref [assignMaybeScalarLeftIdx, assignMaybeScalarLeftPos] assignScalarMaybeStarBlk.ref [assignMaybeScalarLeftIdx, lhsValue, lhsNext]

  startBlock assignScalarMaybeStarBlk
  let ch ← loadInputByte ptr assignScalarMaybeStarPos
  let star ← iconst64 asciiStar
  let isStar ← icmp .eq ch star
  let assignScalarLeftStarBlk ← declareBlock [.i64, .i64, .i64]
  let assignScalarLeftTarget := assignScalarLeftStarBlk.param 0
  let assignScalarLeftValue := assignScalarLeftStarBlk.param 1
  let assignScalarLeftStar := assignScalarLeftStarBlk.param 2
  brif isStar assignScalarLeftStarBlk.ref [assignScalarMaybeStarIdx, assignScalarMaybeStarValue, assignScalarMaybeStarPos] assignScalarEvalStoreBlk.ref [assignScalarMaybeStarIdx, assignScalarPos]

  startBlock assignScalarLeftStarBlk
  let rhsStart0 ← iadd assignScalarLeftStar one
  let rhsStart ← emitSkipWs ptr rhsStart0 len
  let rhsAtEnd ← icmp .uge rhsStart len
  let assignScalarLeftRhsBlk ← declareBlock [.i64, .i64, .i64]
  let assignScalarLeftRhsTarget := assignScalarLeftRhsBlk.param 0
  let assignScalarLeftRhsValue := assignScalarLeftRhsBlk.param 1
  let assignScalarLeftRhsStart := assignScalarLeftRhsBlk.param 2
  brif rhsAtEnd assignScalarEvalStoreBlk.ref [assignScalarLeftTarget, assignScalarPos] assignScalarLeftRhsBlk.ref [assignScalarLeftTarget, assignScalarLeftValue, rhsStart]

  startBlock assignScalarLeftRhsBlk
  let rhsCh ← loadInputByte ptr assignScalarLeftRhsStart
  let lb ← iconst64 asciiLBracket
  let isArrayLiteral ← icmp .eq rhsCh lb
  let assignScalarLeftArrayLiteralBlk ← declareBlock [.i64, .i64, .i64]
  let assignScalarLeftArrayLiteralTarget := assignScalarLeftArrayLiteralBlk.param 0
  let assignScalarLeftArrayLiteralValue := assignScalarLeftArrayLiteralBlk.param 1
  let assignScalarLeftArrayLiteralStart := assignScalarLeftArrayLiteralBlk.param 2
  let assignScalarLeftVarCheckBlk ← declareBlock [.i64, .i64, .i64, .i64]
  let assignScalarLeftVarTarget := assignScalarLeftVarCheckBlk.param 0
  let assignScalarLeftVarValue := assignScalarLeftVarCheckBlk.param 1
  let assignScalarLeftVarStart := assignScalarLeftVarCheckBlk.param 2
  let assignScalarLeftVarIdx := assignScalarLeftVarCheckBlk.param 3
  let rhsVarIdx ← isub rhsCh a
  brif isArrayLiteral assignScalarLeftArrayLiteralBlk.ref [assignScalarLeftRhsTarget, assignScalarLeftRhsValue, assignScalarLeftRhsStart] assignScalarLeftVarCheckBlk.ref [assignScalarLeftRhsTarget, assignScalarLeftRhsValue, assignScalarLeftRhsStart, rhsVarIdx]

  startBlock assignScalarLeftArrayLiteralBlk
  let isValidArray ← emitValidateInputArray ptr assignScalarLeftArrayLiteralStart len
  let isValidArrayNow ← icmp .eq isValidArray one
  let assignScalarLeftArrayScaleBlk ← declareBlock [.i64, .i64, .i64]
  let assignScalarLeftArrayScaleTarget := assignScalarLeftArrayScaleBlk.param 0
  let assignScalarLeftArrayScaleValue := assignScalarLeftArrayScaleBlk.param 1
  let assignScalarLeftArrayScaleStart := assignScalarLeftArrayScaleBlk.param 2
  brif isValidArrayNow assignScalarLeftArrayScaleBlk.ref [assignScalarLeftArrayLiteralTarget, assignScalarLeftArrayLiteralValue, assignScalarLeftArrayLiteralStart] done.ref [zero, zero]

  startBlock assignScalarLeftArrayScaleBlk
  let outLen ← emitScaleInputArrayToOutput cuda ptr assignScalarLeftArrayScaleStart len assignScalarLeftArrayScaleValue
  emitCopyOutputToVarText ptr assignScalarLeftArrayScaleTarget outLen
  emitUploadOutDataToVarBuffer cuda ptr assignScalarLeftArrayScaleTarget
  storeVarKind ptr assignScalarLeftArrayScaleTarget one
  storeVarPresent ptr assignScalarLeftArrayScaleTarget one
  let dummy ← iconst64 0
  jump done.ref [dummy, zero]

  startBlock assignScalarLeftVarCheckBlk
  let rhsIsVar ← emitIsVarChar (← loadInputByte ptr assignScalarLeftVarStart)
  let kind ← loadVarKind ptr assignScalarLeftVarIdx
  let isArrayVar ← icmp .eq kind one
  let useArrayVar ← band rhsIsVar isArrayVar
  let assignScalarLeftArrayVarBlk ← declareBlock [.i64, .i64, .i64]
  let assignScalarLeftArrayVarTarget := assignScalarLeftArrayVarBlk.param 0
  let assignScalarLeftArrayVarValue := assignScalarLeftArrayVarBlk.param 1
  let assignScalarLeftArrayVarIdx := assignScalarLeftArrayVarBlk.param 2
  brif useArrayVar assignScalarLeftArrayVarBlk.ref [assignScalarLeftVarTarget, assignScalarLeftVarValue, assignScalarLeftVarIdx] assignScalarEvalStoreBlk.ref [assignScalarLeftVarTarget, assignScalarPos]

  startBlock assignScalarLeftArrayVarBlk
  let varLen ← emitCopyVarTextToInput ptr assignScalarLeftArrayVarIdx
  let inputStart ← iconst64 0
  let outLen ← emitScaleInputArrayToOutput cuda ptr inputStart varLen assignScalarLeftArrayVarValue
  emitCopyOutputToVarText ptr assignScalarLeftArrayVarTarget outLen
  emitUploadOutDataToVarBuffer cuda ptr assignScalarLeftArrayVarTarget
  storeVarKind ptr assignScalarLeftArrayVarTarget one
  storeVarPresent ptr assignScalarLeftArrayVarTarget one
  let dummy ← iconst64 0
  jump done.ref [dummy, zero]

  startBlock assignScalarEvalStoreBlk
  let scalarValue ← emitParseScalarExpr cuda ptr assignScalarEvalPos len
  let accBuf ← ireduce32 (← fldLoad ptr f.accBuf)
  let zeroBuf ← ireduce32 (← fldLoad ptr f.zeroBuf)
  let varBuf ← ireduce32 (← loadVarBufId ptr assignScalarEvalIdx)
  emitAccFromLiteral cuda ptr scalarValue
  emitCudaLaunchAdd cuda ptr zeroBuf accBuf varBuf
  storeVarKind ptr assignScalarEvalIdx zero
  storeVarPresent ptr assignScalarEvalIdx one
  jump done.ref [scalarValue, zero]

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
  let arrayBytes ← iconst64 2048
  let paramBytes ← iconst64 16
  let zero ← iconst64 0
  let one ← iconst64 1

  callVoid cuda.fnInit [ptr]
  let zeroBuf ← call cuda.fnCreateBuffer [ptr, size8]
  let litBuf ← call cuda.fnCreateBuffer [ptr, size8]
  let accBuf ← call cuda.fnCreateBuffer [ptr, size8]
  let outBuf ← call cuda.fnCreateBuffer [ptr, size8]
  let tmpArrayBufA ← call cuda.fnCreateBuffer [ptr, arrayBytes]
  let tmpArrayBufB ← call cuda.fnCreateBuffer [ptr, arrayBytes]
  let tmpArrayBufOut ← call cuda.fnCreateBuffer [ptr, arrayBytes]
  let tmpArrayParamBuf ← call cuda.fnCreateBuffer [ptr, paramBytes]
  fldStore ptr f.zeroBuf (← sextend64 zeroBuf)
  fldStore ptr f.litBuf (← sextend64 litBuf)
  fldStore ptr f.accBuf (← sextend64 accBuf)
  fldStore ptr f.outBuf (← sextend64 outBuf)
  fldStore ptr f.tmpArrayBufA (← sextend64 tmpArrayBufA)
  fldStore ptr f.tmpArrayBufB (← sextend64 tmpArrayBufB)
  fldStore ptr f.tmpArrayBufOut (← sextend64 tmpArrayBufOut)
  fldStore ptr f.tmpArrayParamBuf (← sextend64 tmpArrayParamBuf)

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
  let vbuf ← call cuda.fnCreateBuffer [ptr, arrayBytes]
  storeVarBufId ptr vi vbuf
  storeVarPresent ptr vi zero
  storeVarKind ptr vi zero
  storeVarTextLen ptr vi zero
  let nextI ← iadd vi one
  jump setupHdr.ref [nextI]

  startBlock afterSetup
  let loopHdr ← declareBlock []
  let loopBody ← declareBlock [.i64]
  let bytesRead := loopBody.param 0
  let printBlk ← declareBlock [.i64]
  let lineValue := printBlk.param 0
  let rawPrintBlk ← declareBlock []
  let rawDoPrintBlk ← declareBlock [.i64]
  let rawOutLen := rawDoPrintBlk.param 0
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
  brif shouldPrintNow printBlk.ref [lineResult] rawPrintBlk.ref []

  startBlock printBlk
  let outLen ← emitFormatSigned ptr lineValue
  fldStore ptr f.outputLen outLen
  let _ ← call fnWrite [ptr, outOff, outLen]
  jump loopHdr.ref []

  startBlock rawPrintBlk
  let two ← iconst64 2
  let shouldRawPrint ← icmp .eq shouldPrint two
  let rawLen ← fldLoad ptr f.outputLen
  brif shouldRawPrint rawDoPrintBlk.ref [rawLen] loopHdr.ref []

  startBlock rawDoPrintBlk
  let _ ← call fnWrite [ptr, outOff, rawOutLen]
  jump loopHdr.ref []

  startBlock done
  callVoid cuda.fnCleanup [ptr]
  ret

def payloads : List UInt8 :=
  mkPayload layoutMeta.totalSize [
    f.ptx.init (stringToBytes scalarAddPtx),
    f.arrayScalePtx.init (stringToBytes arrayScalePtxSource),
    f.arrayAddPtx.init (stringToBytes arrayAddPtxSource)
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
