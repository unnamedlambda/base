import AlgorithmLib


open Lean (Json)
open AlgorithmLib
open AlgorithmLib.Layout
open AlgorithmLib.PTX

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
def asciiGreater : Int := 62

def scalarAddPtx : String := buildModuleWith { version := "7.0", target := "sm_50" } [{
  name := "main", params := ["lhs_ptr", "rhs_ptr", "out_ptr"], body := do
  let lhs ← ldParam "lhs_ptr"
  let rhs ← ldParam "rhs_ptr"
  let out ← ldParam "out_ptr"
  let a ← freshRd; ldGlobalU64 a lhs
  let b ← freshRd; ldGlobalU64 b rhs
  let sum ← freshRd; addS64 sum a b
  stGlobalU64 out sum
  ptxRet }]

def arrayScalePtxSource : String := buildModuleWith { version := "7.0", target := "sm_50" } [{
  name := "main", params := ["in_ptr", "param_ptr", "out_ptr"], body := do
  let inPtr    ← ldParam "in_ptr"
  let paramPtr ← ldParam "param_ptr"
  let outPtr   ← ldParam "out_ptr"
  let idx  ← freshR;  movR idx ctaX
  let byte ← freshRd; cvtU64 byte idx; shlRd byte byte 3
  let addr ← freshRd; addRd addr inPtr byte
  let x    ← freshRd; ldGlobalS64 x addr
  let s    ← freshRd; ldGlobalS64 s paramPtr
  let y    ← freshRd; mulLoS64 y x s
  let outAddr ← freshRd; addRd outAddr outPtr byte
  stGlobalS64 outAddr y
  ptxRet }]

def arrayAddPtxSource : String := buildModuleWith { version := "7.0", target := "sm_50" } [{
  name := "main", params := ["lhs_ptr", "rhs_ptr", "param_ptr", "out_ptr"], body := do
  let lhs   ← ldParam "lhs_ptr"
  let rhs   ← ldParam "rhs_ptr"
  let _     ← ldParam "param_ptr"  -- loaded but unused
  let out   ← ldParam "out_ptr"
  let idx   ← freshR;  movR idx ctaX
  let byte  ← freshRd; cvtU64 byte idx; shlRd byte byte 3
  let addrA ← freshRd; addRd addrA lhs byte
  let a     ← freshRd; ldGlobalS64 a addrA
  let addrB ← freshRd; addRd addrB rhs byte
  let b     ← freshRd; ldGlobalS64 b addrB
  let y     ← freshRd; addS64 y a b
  let addrO ← freshRd; addRd addrO out byte
  stGlobalS64 addrO y
  ptxRet }]

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- The externals every emitted function declares, in one order, so a slot
    index means the same thing in all of them. -/
def ffiEnv : ((IR.FnRef × IR.FnRef) × IR.CudaSetup) × FnEnv := (Id.run (do
  let rd := IR.FFI.std.stdinReadline
  let wr := IR.FFI.std.stdoutWrite
  let c := IR.FFI.std.cuda
  pure ((rd, wr), c)), env% [.cuda, .fileIO])
def fnRead : IR.FnRef := ffiEnv.1.1.1
def fnWrite : IR.FnRef := ffiEnv.1.1.2
def cuda : IR.CudaSetup := ffiEnv.1.2
def env : FnEnv := ffiEnv.2

def loadByteAt (ptr : R) (baseOff : Nat) (idx : R) : M R := do
  let base ← iconst64 baseOff
  let rel ← iadd base idx
  let addr ← iadd ptr rel
  uload8_64 addr

def storeByteAt (ptr : R) (baseOff : Nat) (idx : R) (value : R) : M Unit := do
  let base ← iconst64 baseOff
  let rel ← iadd base idx
  let addr ← iadd ptr rel
  istore8 value addr

def loadInputByte (ptr idx : R) : M R :=
  loadByteAt ptr f.input.offset idx

def loadScratchByte (ptr idx : R) : M R :=
  loadByteAt ptr f.scratch.offset idx

def storeScratchByte (ptr idx value : R) : M Unit :=
  storeByteAt ptr f.scratch.offset idx value

def scratchAddr (ptr idx : R) : M R := do
  let base ← iconst64 f.scratch.offset
  let rel ← iadd base idx
  iadd ptr rel

def scratchI64Addr (ptr baseIdx depth : R) : M R := do
  let eight ← iconst64 8
  let byteOff ← imul depth eight
  let idx ← iadd baseIdx byteOff
  scratchAddr ptr idx

def loadScratchI64 (ptr baseIdx depth : R) : M R := do
  let addr ← scratchI64Addr ptr baseIdx depth
  load64 addr

def storeScratchI64 (ptr baseIdx depth value : R) : M Unit := do
  let addr ← scratchI64Addr ptr baseIdx depth
  store value addr

def loadScratchByteDyn (ptr baseIdx depth : R) : M R := do
  let idx ← iadd baseIdx depth
  loadScratchByte ptr idx

def storeScratchByteDyn (ptr baseIdx depth value : R) : M Unit := do
  let idx ← iadd baseIdx depth
  storeScratchByte ptr idx value

def storeOutputByte (ptr idx value : R) : M Unit :=
  storeByteAt ptr f.output.offset idx value

def storeOutputByteFrom (ptr base idx value : R) : M Unit := do
  let outIdx ← iadd base idx
  storeOutputByte ptr outIdx value

def loadOutputByte (ptr idx : R) : M R :=
  loadByteAt ptr f.output.offset idx

def loadVarKind (ptr idx : R) : M R :=
  loadByteAt ptr f.varKind.offset idx

def storeVarKind (ptr idx value : R) : M Unit :=
  storeByteAt ptr f.varKind.offset idx value

def varTextAddr (ptr varIdx textIdx : R) : M R := do
  let stride ← iconst64 256
  let base ← iconst64 f.varTexts.offset
  let varOff ← imul varIdx stride
  let rel0 ← iadd base varOff
  let rel ← iadd rel0 textIdx
  iadd ptr rel

def loadVarTextByte (ptr varIdx textIdx : R) : M R := do
  let addr ← varTextAddr ptr varIdx textIdx
  uload8_64 addr

def storeVarTextByte (ptr varIdx textIdx value : R) : M Unit := do
  let addr ← varTextAddr ptr varIdx textIdx
  istore8 value addr

def varTextLenAddr (ptr varIdx : R) : M R := do
  let eight ← iconst64 8
  let base ← iconst64 f.varTextLens.offset
  let byteOff ← imul varIdx eight
  let rel ← iadd base byteOff
  iadd ptr rel

def loadVarTextLen (ptr varIdx : R) : M R := do
  let addr ← varTextLenAddr ptr varIdx
  load64 addr

def storeVarTextLen (ptr varIdx value : R) : M Unit := do
  let addr ← varTextLenAddr ptr varIdx
  store value addr

def dataI64Addr (ptr : R) (baseOff : Nat) (idx : R) : M R := do
  let eight ← iconst64 8
  let base ← iconst64 baseOff
  let byteOff ← imul idx eight
  let rel ← iadd base byteOff
  iadd ptr rel

def storeDataI64 (ptr : R) (baseOff : Nat) (idx value : R) : M Unit := do
  let addr ← dataI64Addr ptr baseOff idx
  store value addr

def loadDataI64 (ptr : R) (baseOff : Nat) (idx : R) : M R := do
  let addr ← dataI64Addr ptr baseOff idx
  load64 addr

def emitArrayParam (ptr scalar count : R) : M Unit := do
  storeDataI64 ptr f.arrayParamData.offset (← iconst64 0) scalar
  storeDataI64 ptr f.arrayParamData.offset (← iconst64 1) count

def emitIsVarChar (ch : R) : M R := do
  let a ← iconst64 asciiA
  let z ← iconst64 asciiZ
  let geA ← icmp .uge ch a
  let leZ ← icmp .ule ch z
  band geA leZ

def loadVarPresent (ptr idx : R) : M R :=
  loadByteAt ptr f.varPresent.offset idx

def storeVarPresent (ptr idx value : R) : M Unit :=
  storeByteAt ptr f.varPresent.offset idx value

def loadVarBufId (ptr idx : R) : M R := do
  let four ← iconst64 4
  let base ← iconst64 f.varBufIds.offset
  let byteOff ← imul idx four
  let rel ← iadd base byteOff
  let addr ← iadd ptr rel
  uload32_64 addr

def storeVarBufId (ptr idx bufId32 : R) : M Unit := do
  let four ← iconst64 4
  let base ← iconst64 f.varBufIds.offset
  let byteOff ← imul idx four
  let rel ← iadd base byteOff
  let addr ← iadd ptr rel
  store bufId32 addr

def emitSkipWs (ptr pos len : R) : M R := do
  let one ← iconst64 1
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline

  let e ← wloop1 pos
    (head := fun i => return (contIfULt i len, [i], ()))
    (body := fun i _ => do
      let ch ← loadInputByte ptr i
      when .ne ch sp (when .ne ch nl (brk [i]))
      return [← iadd i one])
  return e.headD 0

def emitParseInt (ptr start len : R) : M (R × R) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let ten ← iconst64 10
  let minus ← iconst64 asciiMinus
  let plus ← iconst64 asciiPlus
  let zeroCh ← iconst64 asciiZero
  let nineCh ← iconst64 asciiNine

  let pos0 ← emitSkipWs ptr start len

  -- The sign, then the digits from wherever the sign left off. An input that
  -- ends before either is a zero at the position it ended.
  let r ← ifte .uge pos0 len (pure [zero, pos0, zero]) do
    let ch ← loadInputByte ptr pos0
    let posNext ← iadd pos0 one
    let sd ← ifte .eq ch minus (pure [posNext, one])
      (ifte .eq ch plus (pure [posNext, zero]) (pure [pos0, zero]))
    let digitStart := sd.headD 0
    let negFlag := sd.getD 1 0
    wloop [digitStart, zero, negFlag]
      (head := fun c => return (contIfULt (c.headD 0) len, [c.getD 1 0, c.headD 0, c.getD 2 0], ()))
      (body := fun c _ => do
        let i := c.headD 0; let acc := c.getD 1 0; let neg := c.getD 2 0
        let ch ← loadInputByte ptr i
        when .ult ch zeroCh (brk [acc, i, neg])
        when .ugt ch nineCh (brk [acc, i, neg])
        let digit ← isub ch zeroCh
        let acc10 ← imul acc ten
        return [← iadd i one, ← iadd acc10 digit, neg])

  let magnitude := r.headD 0
  let endPos := r.getD 1 0
  let negResult := r.getD 2 0
  let negated ← ineg magnitude
  let isNeg ← icmp .eq negResult one
  let finalVal ← select isNeg negated magnitude
  pure (finalVal, endPos)

def emitFormatSigned (ptr value : R) : M R := do
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
  let absVal ← select isNeg negValue value
  let isZero ← icmp .eq absVal zero

  let z8 ← iconst .i8 0

  -- The digits are produced least-significant first into the tail of scratch,
  -- so the copy that follows walks forward from wherever the division stopped.
  let r ← ifte .ne isZero z8
    (do
      storeOutputByte ptr zero zeroCh
      storeOutputByte ptr one nlCh
      pure [two])
    (do
      let dv ← wloop2 absVal scratchLast
        (head := fun cur idx => return (contIf .ne cur zero, [idx], ()))
        (body := fun cur idx _ => do
          let q ← udiv cur ten
          let q10 ← imul q ten
          let rem ← isub cur q10
          let digitCh ← iadd zeroCh rem
          storeScratchByte ptr idx digitCh
          return [q, ← isub idx one])
      let firstDigitIdx ← iadd (dv.headD 0) one
      let op ← ifte .ne isNeg z8
        (do storeOutputByte ptr zero minusCh; pure [one])
        (pure [zero])
      let cp ← wloop2 firstDigitIdx (op.headD 0)
        (head := fun ci outPos => return (contIfULe ci scratchLast, [outPos], ()))
        (body := fun ci outPos _ => do
          let ch ← loadScratchByte ptr ci
          storeOutputByte ptr outPos ch
          return [← iadd ci one, ← iadd outPos one])
      let newlinePos := cp.headD 0
      storeOutputByte ptr newlinePos nlCh
      pure [← iadd newlinePos one])
  return r.headD 0

def emitFormatSignedAt (ptr value startPos : R) : M R := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let ten ← iconst64 10
  let minusCh ← iconst64 asciiMinus
  let zeroCh ← iconst64 asciiZero
  let scratchLast ← iconst64 511

  let isNeg ← icmp .slt value zero
  let negValue ← ineg value
  let absVal ← select isNeg negValue value
  let isZero ← icmp .eq absVal zero

  let z8 ← iconst .i8 0

  let r ← ifte .ne isZero z8
    (do
      storeOutputByte ptr startPos zeroCh
      pure [← iadd startPos one])
    (do
      let dv ← wloop2 absVal scratchLast
        (head := fun cur idx => return (contIf .ne cur zero, [idx], ()))
        (body := fun cur idx _ => do
          let q ← udiv cur ten
          let q10 ← imul q ten
          let rem ← isub cur q10
          let digitCh ← iadd zeroCh rem
          storeScratchByte ptr idx digitCh
          return [q, ← isub idx one])
      let firstDigitIdx ← iadd (dv.headD 0) one
      let op ← ifte .ne isNeg z8
        (do
          storeOutputByte ptr startPos minusCh
          pure [← iadd startPos one])
        (pure [startPos])
      wloop2 firstDigitIdx (op.headD 0)
        (head := fun ci outPos => return (contIfULe ci scratchLast, [outPos], ()))
        (body := fun ci outPos _ => do
          let ch ← loadScratchByte ptr ci
          storeOutputByte ptr outPos ch
          return [← iadd ci one, ← iadd outPos one]))
  return r.headD 0

def emitCudaLaunchAdd (ptr lhsBuf rhsBuf outBuf : R) : M Unit := do
  let ptxOff ← fldOffset f.ptx
  let bindOff ← fldOffset f.bindDesc
  let nBufs ← iconst32 3
  let one32 ← iconst32 1
  fldStore32At ptr f.bindDesc 0 lhsBuf
  fldStore32At ptr f.bindDesc 4 rhsBuf
  fldStore32At ptr f.bindDesc 8 outBuf
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff one32 one32 one32 one32 one32 one32
  pure ()

def emitCudaLaunchArrayScale (ptr inBuf paramBuf outBuf count : R) : M Unit := do
  let ptxOff ← fldOffset f.arrayScalePtx
  let bindOff ← fldOffset f.bindDesc
  let nBufs ← iconst32 3
  let count32 ← ireduce32 count
  let one32 ← iconst32 1
  fldStore32At ptr f.bindDesc 0 inBuf
  fldStore32At ptr f.bindDesc 4 paramBuf
  fldStore32At ptr f.bindDesc 8 outBuf
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff count32 one32 one32 one32 one32 one32
  pure ()

def emitCudaLaunchArrayAdd (ptr lhsBuf rhsBuf paramBuf outBuf count : R) : M Unit := do
  let ptxOff ← fldOffset f.arrayAddPtx
  let bindOff ← fldOffset f.bindDesc
  let nBufs ← iconst32 4
  let count32 ← ireduce32 count
  let one32 ← iconst32 1
  fldStore32At ptr f.bindDesc 0 lhsBuf
  fldStore32At ptr f.bindDesc 4 rhsBuf
  fldStore32At ptr f.bindDesc 8 paramBuf
  fldStore32At ptr f.bindDesc 12 outBuf
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff count32 one32 one32 one32 one32 one32
  pure ()

def emitUploadLiteralToBuf (ptr bufId value : R) : M Unit := do
  fldStore ptr f.firstVal value
  let size8 ← iconst64 8
  let valOff ← fldOffset f.firstVal
  let _ ← cudaUpload cuda ptr bufId valOff size8
  pure ()

def emitAccFromLiteral (ptr value : R) : M Unit := do
  let bufAcc64 ← fldLoad ptr f.accBuf
  let bufAcc ← ireduce32 bufAcc64
  emitUploadLiteralToBuf ptr bufAcc value

def emitAccFromVar (ptr varIdx : R) : M Unit := do
  let zeroBuf ← ireduce32 (← fldLoad ptr f.zeroBuf)
  let accBuf ← ireduce32 (← fldLoad ptr f.accBuf)
  let varBuf ← ireduce32 (← loadVarBufId ptr varIdx)
  emitCudaLaunchAdd ptr zeroBuf varBuf accBuf

def emitTermToLiteralBuf (ptr value : R) : M R := do
  let litBuf64 ← fldLoad ptr f.litBuf
  let litBuf ← ireduce32 litBuf64
  emitUploadLiteralToBuf ptr litBuf value
  pure litBuf

/-- A term: `(isVar, value-or-variable-index, position after it)`. -/
def emitTermParse (ptr start len : R) : M (R × R × R) := do
  let zero ← iconst64 0
  let z8 ← iconst .i8 0
  let pos0 ← emitSkipWs ptr start len

  let r ← ifte .uge pos0 len (pure [zero, zero, pos0]) do
    let ch ← loadInputByte ptr pos0
    let isVar ← emitIsVarChar ch
    let a ← iconst64 asciiA
    let idx ← isub ch a
    ifte .ne isVar z8
      (do
        let one ← iconst64 1
        return [one, idx, ← iadd pos0 one])
      (do
        let (v, p) ← emitParseInt ptr pos0 len
        return [zero, v, p])
  return (r.headD 0, r.getD 1 0, r.getD 2 0)

def emitParseAddChain (ptr start len : R) : M Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let plus ← iconst64 asciiPlus
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline

  let (firstIsVar, firstPayload, pos1) ← emitTermParse ptr start len
  let _ ← ifte .ne firstIsVar zero
    (do emitAccFromVar ptr firstPayload; pure [])
    (do emitAccFromLiteral ptr firstPayload; pure [])

  -- Separators are skipped one at a time; a `+` takes the next term into the
  -- accumulator, and anything else ends the chain.
  let _ ← wloop2 one pos1
    (head := fun _ pos => return (contIfULt pos len, ([] : List R), ()))
    (body := fun haveAcc pos _ => do
      let ch ← loadInputByte ptr pos
      let nextPos ← iadd pos one
      when .eq ch sp (continueWith [haveAcc, nextPos])
      when .eq ch nl (continueWith [haveAcc, nextPos])
      when .ne ch plus (brk [])
      let (termIsVar, termPayload, termEnd) ← emitTermParse ptr nextPos len
      let _ ← ifte .ne termIsVar zero
        (do
          let accBuf ← ireduce32 (← fldLoad ptr f.accBuf)
          let outBuf ← ireduce32 (← fldLoad ptr f.outBuf)
          let rhsBuf ← ireduce32 (← loadVarBufId ptr termPayload)
          emitCudaLaunchAdd ptr accBuf rhsBuf outBuf
          let zeroBuf ← ireduce32 (← fldLoad ptr f.zeroBuf)
          emitCudaLaunchAdd ptr zeroBuf outBuf accBuf
          pure [])
        (do
          let litBuf ← emitTermToLiteralBuf ptr termPayload
          let accBuf ← ireduce32 (← fldLoad ptr f.accBuf)
          let outBuf ← ireduce32 (← fldLoad ptr f.outBuf)
          emitCudaLaunchAdd ptr accBuf litBuf outBuf
          let zeroBuf ← ireduce32 (← fldLoad ptr f.zeroBuf)
          emitCudaLaunchAdd ptr zeroBuf outBuf accBuf
          pure [])
      return [haveAcc, termEnd])
  pure ()

def emitDownloadAccToResult (ptr : R) : M R := do
  let accBuf64 ← fldLoad ptr f.accBuf
  let accBuf ← ireduce32 accBuf64
  let size8 ← iconst64 8
  let outOff ← fldOffset f.result
  let _ ← cudaDownload cuda ptr accBuf outOff size8
  fldLoad ptr f.result

def emitScalarTermValue (ptr start len : R) : M (R × R) := do
  let zero ← iconst64 0
  let pos0 ← emitSkipWs ptr start len

  let z8 ← iconst .i8 0

  let r ← ifte .uge pos0 len (pure [zero, pos0]) do
    let ch ← loadInputByte ptr pos0
    let isVar ← emitIsVarChar ch
    let a ← iconst64 asciiA
    let idx ← isub ch a
    ifte .ne isVar z8
      (do
        emitAccFromVar ptr idx
        let value ← emitDownloadAccToResult ptr
        let one ← iconst64 1
        return [value, ← iadd pos0 one])
      (do
        let (v, p) ← emitParseInt ptr pos0 len
        return [v, p])
  return (r.headD 0, r.getD 1 0)

/-- A chain of `*` from `acc0` at `pos0`: the product, and where it stopped. -/
def emitMulChain (ptr acc0 pos0 len : R) : M (R × R) := do
  let one ← iconst64 1
  let star ← iconst64 asciiStar
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline
  let e ← wloop2 acc0 pos0
    (head := fun acc p => do
      let q ← emitSkipWs ptr p len
      return (exitIf .uge q len, [acc, q], q))
    (body := fun acc _ q => do
      let ch ← loadInputByte ptr q
      let nextPos ← iadd q one
      when .eq ch sp (continueWith [acc, nextPos])
      when .eq ch nl (brk [acc, nextPos])
      when .ne ch star (brk [acc, q])
      let (rhs, rhsEnd) ← emitScalarTermValue ptr nextPos len
      return [← imul acc rhs, rhsEnd])
  return (e.headD 0, e.getD 1 0)

def emitParseScalarExpr (ptr start len : R) : M R := do
  let one ← iconst64 1
  let plus ← iconst64 asciiPlus
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline

  let (firstVal, firstEnd) ← emitScalarTermValue ptr start len
  let (mulVal, mulEnd) ← emitMulChain ptr firstVal firstEnd len

  -- The sum, each of whose terms is itself a product chain.
  let e ← wloop2 mulVal mulEnd
    (head := fun acc p => do
      let q ← emitSkipWs ptr p len
      return (exitIf .uge q len, [acc], q))
    (body := fun acc _ q => do
      let ch ← loadInputByte ptr q
      let nextPos ← iadd q one
      when .eq ch sp (continueWith [acc, nextPos])
      when .eq ch nl (brk [acc])
      when .ne ch plus (brk [acc])
      let (rhsVal, rhsEnd) ← emitScalarTermValue ptr nextPos len
      let (rhsMul, rhsMulEnd) ← emitMulChain ptr rhsVal rhsEnd len
      return [← iadd acc rhsMul, rhsMulEnd])
  return e.headD 0

def emitCopyInputToVarText (ptr varIdx start len : R) : M Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let maxText ← iconst64 255
  let nl ← iconst64 asciiNewline
  let pos0 ← emitSkipWs ptr start len

  let z8 ← iconst .i8 0
  let e ← wloop2 pos0 zero
    (head := fun inPos outPos => do
      let atEnd ← icmp .uge inPos len
      let full ← icmp .uge outPos maxText
      let stop ← bor atEnd full
      return (contIf .eq stop z8, [outPos], ()))
    (body := fun inPos outPos _ => do
      let c ← loadInputByte ptr inPos
      when .eq c nl (brk [outPos])
      storeVarTextByte ptr varIdx outPos c
      return [← iadd inPos one, ← iadd outPos one])
  storeVarTextLen ptr varIdx (e.headD 0)

def emitCopyInputToOutput (ptr start len : R) : M R := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let nl ← iconst64 asciiNewline
  let pos0 ← emitSkipWs ptr start len

  let e ← wloop2 pos0 zero
    (head := fun inPos outPos => return (contIfULt inPos len, [outPos], ()))
    (body := fun inPos outPos _ => do
      let c ← loadInputByte ptr inPos
      when .eq c nl (brk [outPos])
      storeOutputByte ptr outPos c
      return [← iadd inPos one, ← iadd outPos one])
  let outLen := e.headD 0
  storeOutputByte ptr outLen nl
  iadd outLen one

def emitCopyVarTextToOutput (ptr varIdx : R) : M R := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let nl ← iconst64 asciiNewline
  let textLen ← loadVarTextLen ptr varIdx

  let _ ← wloop1 zero
    (head := fun i => return (contIfULt i textLen, ([] : List R), ()))
    (body := fun i _ => do
      let ch ← loadVarTextByte ptr varIdx i
      storeOutputByte ptr i ch
      return [← iadd i one])
  storeOutputByte ptr textLen nl
  iadd textLen one

def emitCopyOutputToVarText (ptr varIdx outLen : R) : M Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let nl ← iconst64 asciiNewline
  let maxText ← iconst64 255

  let z8 ← iconst .i8 0
  let e ← wloop1 zero
    (head := fun i => do
      let atEnd ← icmp .uge i outLen
      let full ← icmp .uge i maxText
      let stop ← bor atEnd full
      return (contIf .eq stop z8, [i], ()))
    (body := fun i _ => do
      let ch ← loadOutputByte ptr i
      when .eq ch nl (brk [i])
      storeVarTextByte ptr varIdx i ch
      return [← iadd i one])
  storeVarTextLen ptr varIdx (e.headD 0)

def emitFindInputChar (ptr start len target : R) : M R := do
  let one ← iconst64 1
  let e ← wloop1 start
    (head := fun pos => return (contIfULt pos len, [len], ()))
    (body := fun pos _ => do
      let ch ← loadInputByte ptr pos
      when .eq ch target (brk [pos])
      return [← iadd pos one])
  return e.headD 0

def emitCopyVarTextToInput (ptr varIdx : R) : M R := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let textLen ← loadVarTextLen ptr varIdx

  let _ ← wloop1 zero
    (head := fun i => return (contIfULt i textLen, ([] : List R), ()))
    (body := fun i _ => do
      let ch ← loadVarTextByte ptr varIdx i
      storeByteAt ptr f.input.offset i ch
      return [← iadd i one])
  pure textLen

def emitClearArrayValidationScratch (ptr : R) : M Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let maxDepth ← iconst64 32
  let countsBase ← iconst64 0
  let expectedBase ← iconst64 128
  let kindBase ← iconst64 256
  let expectedSetBase ← iconst64 288

  let _ ← wloop1 zero
    (head := fun depth => return (contIfULt depth maxDepth, ([] : List R), ()))
    (body := fun depth _ => do
      storeScratchI64 ptr countsBase depth zero
      storeScratchI64 ptr expectedBase depth zero
      storeScratchByteDyn ptr kindBase depth zero
      storeScratchByteDyn ptr expectedSetBase depth zero
      return [← iadd depth one])

def emitValidateInputArray (ptr start len : R) : M R := do
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

  let z8 ← iconst .i8 0

  -- One scan over the text. The loop leaves with `0` for a malformed array, or
  -- with `1` and the position just past the outermost `]`.
  let e ← wloop2 start zero
    (head := fun pos _ => return (contIfULt pos len, [zero, pos], ()))
    (body := fun pos depth _ => do
      let ch ← loadInputByte ptr pos
      let nextPos ← iadd pos one
      when .eq ch sp (continueWith [nextPos, depth])
      when .eq ch comma (continueWith [nextPos, depth])
      when .eq ch lb (do
        -- A nested array counts as one element of its parent, whose kind must
        -- then be the array kind.
        when .uge depth maxDepth (brk [zero, pos])
        when .ne depth zero (do
          let k ← loadScratchByteDyn ptr kindBase depth
          let _ ← ifte .eq k zero
            (do storeScratchByteDyn ptr kindBase depth two; pure [])
            (do when .ne k two (brk [zero, pos]); pure [])
          let c ← loadScratchI64 ptr countsBase depth
          storeScratchI64 ptr countsBase depth (← iadd c one))
        let nextDepth ← iadd depth one
        storeScratchI64 ptr countsBase nextDepth zero
        storeScratchByteDyn ptr kindBase nextDepth zero
        continueWith [nextPos, nextDepth])
      when .eq ch rb (do
        -- The first sibling at this depth fixes the length every later one
        -- must have.
        when .eq depth zero (brk [zero, pos])
        let count ← loadScratchI64 ptr countsBase depth
        let expectedSet ← loadScratchByteDyn ptr expectedSetBase depth
        let _ ← ifte .eq expectedSet one
          (do
            let expected ← loadScratchI64 ptr expectedBase depth
            when .ne count expected (brk [zero, pos])
            pure [])
          (do
            storeScratchI64 ptr expectedBase depth count
            storeScratchByteDyn ptr expectedSetBase depth one
            pure [])
        let newDepth ← isub depth one
        when .eq newDepth zero (brk [one, nextPos])
        continueWith [nextPos, newDepth])
      when .ne ch minus (do
        when .ult ch zeroCh (brk [zero, pos])
        when .ugt ch nineCh (brk [zero, pos]))
      when .eq depth zero (brk [zero, pos])
      let k ← loadScratchByteDyn ptr kindBase depth
      let _ ← ifte .eq k zero
        (do storeScratchByteDyn ptr kindBase depth one; pure [])
        (do when .ne k one (brk [zero, pos]); pure [])
      let c ← loadScratchI64 ptr countsBase depth
      storeScratchI64 ptr countsBase depth (← iadd c one)
      let (_, numEnd) ← emitParseInt ptr pos len
      return [numEnd, depth])

  -- What follows the array may only be blank, or an operator that takes one.
  let r ← ifte .eq (e.headD 0) zero (pure [zero]) do
    wloop1 (e.getD 1 0)
      (head := fun p => return (contIfULt p len, [one], ()))
      (body := fun p _ => do
        let ch ← loadInputByte ptr p
        let nextPos ← iadd p one
        when .ne ch sp (do
          when .eq ch nl (brk [one])
          let isStar ← icmp .eq ch star
          let isPlus ← icmp .eq ch plus
          let isArrayOp ← bor isStar isPlus
          when .eq isArrayOp z8 (brk [zero])
          brk [one])
        return [nextPos])
  return r.headD 0

def emitFinishOutputLineTrimSpaces (ptr outPos : R) : M R := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let sp ← iconst64 asciiSpace
  let nl ← iconst64 asciiNewline

  let e ← wloop1 outPos
    (head := fun pos => return (contIf .ne pos zero, [pos], ()))
    (body := fun pos _ => do
      let prevPos ← isub pos one
      let ch ← loadOutputByte ptr prevPos
      when .ne ch sp (brk [pos])
      return [prevPos])
  let end_ := e.headD 0
  storeOutputByte ptr end_ nl
  iadd end_ one

def emitParseInputArrayNumbersToData (ptr start len : R) (baseOff : Nat) : M R := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let nl ← iconst64 asciiNewline
  let plus ← iconst64 asciiPlus
  let star ← iconst64 asciiStar
  let minus ← iconst64 asciiMinus
  let zeroCh ← iconst64 asciiZero
  let nineCh ← iconst64 asciiNine

  let z8 ← iconst .i8 0
  -- Brackets and separators are skipped; every number goes to the next slot.
  let e ← wloop2 start zero
    (head := fun pos count => return (contIfULt pos len, [count], ()))
    (body := fun pos count _ => do
      let ch ← loadInputByte ptr pos
      let isNl ← icmp .eq ch nl
      let isPlus ← icmp .eq ch plus
      let isStar ← icmp .eq ch star
      let stop0 ← bor isNl isPlus
      let stop ← bor stop0 isStar
      when .ne stop z8 (brk [count])
      when .ne ch minus (do
        let belowZero ← icmp .ult ch zeroCh
        let aboveNine ← icmp .ugt ch nineCh
        let notDigit ← bor belowZero aboveNine
        when .ne notDigit z8 (continueWith [← iadd pos one, count]))
      let (value, nextPos) ← emitParseInt ptr pos len
      storeDataI64 ptr baseOff count value
      return [nextPos, ← iadd count one])
  return e.headD 0

def emitFormatInputArrayShapeFromOutData (ptr start len count : R) : M R := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let nl ← iconst64 asciiNewline
  let plus ← iconst64 asciiPlus
  let star ← iconst64 asciiStar
  let minus ← iconst64 asciiMinus
  let zeroCh ← iconst64 asciiZero
  let nineCh ← iconst64 asciiNine

  let z8 ← iconst .i8 0
  -- The input's own punctuation is copied through, and each number it holds is
  -- replaced by the result at the matching index.
  let e ← wloop [start, zero, zero]
    (head := fun c => return (contIfULt (c.headD 0) len, [c.getD 1 0], ()))
    (body := fun c _ => do
      let inPos := c.headD 0
      let outPos := c.getD 1 0
      let dataIdx := c.getD 2 0
      let ch ← loadInputByte ptr inPos
      let isNl ← icmp .eq ch nl
      let isPlus ← icmp .eq ch plus
      let isStar ← icmp .eq ch star
      let stop0 ← bor isNl isPlus
      let stop ← bor stop0 isStar
      when .ne stop z8 (brk [outPos])
      when .ne ch minus (do
        let belowZero ← icmp .ult ch zeroCh
        let aboveNine ← icmp .ugt ch nineCh
        let notDigit ← bor belowZero aboveNine
        when .ne notDigit z8 (do
          storeOutputByte ptr outPos ch
          continueWith [← iadd inPos one, ← iadd outPos one, dataIdx]))
      when .uge dataIdx count (brk [zero])
      let value ← loadDataI64 ptr f.arrayOutData.offset dataIdx
      let nextOut ← emitFormatSignedAt ptr value outPos
      let (_, nextIn) ← emitParseInt ptr inPos len
      return [nextIn, nextOut, ← iadd dataIdx one])
  emitFinishOutputLineTrimSpaces ptr (e.headD 0)

def emitUploadInputArrayToVarBuffer (ptr varIdx start len : R) : M Unit := do
  let _count ← emitParseInputArrayNumbersToData ptr start len f.arrayLhsData.offset
  let bytes ← iconst64 2048
  let dataOff ← fldOffset f.arrayLhsData
  let varBuf ← ireduce32 (← loadVarBufId ptr varIdx)
  let _ ← cudaUpload cuda ptr varBuf dataOff bytes
  pure ()

def emitUploadOutDataToVarBuffer (ptr varIdx : R) : M Unit := do
  let bytes ← iconst64 2048
  let dataOff ← fldOffset f.arrayOutData
  let varBuf ← ireduce32 (← loadVarBufId ptr varIdx)
  let _ ← cudaUpload cuda ptr varBuf dataOff bytes
  pure ()

def emitScaleInputArrayToOutput (ptr arrStart len scalar : R) : M R := do
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
  let _ ← cudaUpload cuda ptr lhsBuf lhsOff bytes
  let _ ← cudaUpload cuda ptr paramBuf paramOff paramBytes
  emitCudaLaunchArrayScale ptr lhsBuf paramBuf outBuf count
  let _ ← cudaDownload cuda ptr outBuf outOff bytes
  emitFormatInputArrayShapeFromOutData ptr arrStart len count

def emitAddInputArraysToOutput (ptr lhsStart rhsStart len : R) : M R := do
  let zero ← iconst64 0
  let lhsCount ← emitParseInputArrayNumbersToData ptr lhsStart len f.arrayLhsData.offset
  let rhsCount ← emitParseInputArrayNumbersToData ptr rhsStart len f.arrayRhsData.offset
  -- Elementwise addition needs the shapes to agree; a mismatch is an empty
  -- line rather than a result.
  let r ← ifte .eq lhsCount rhsCount
    (do
      emitArrayParam ptr zero lhsCount
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
      let _ ← cudaUpload cuda ptr lhsBuf lhsOff bytes
      let _ ← cudaUpload cuda ptr rhsBuf rhsOff bytes
      let _ ← cudaUpload cuda ptr paramBuf paramOff paramBytes
      emitCudaLaunchArrayAdd ptr lhsBuf rhsBuf paramBuf outBuf lhsCount
      let _ ← cudaDownload cuda ptr outBuf outOff bytes
      let outLen ← emitFormatInputArrayShapeFromOutData ptr lhsStart len lhsCount
      pure [outLen])
      (pure [zero])
  return r.headD 0

/-- A scalar expression evaluated for its value; the caller prints it. -/
def emitScalarLine (ptr start len : R) : M (R × R) := do
  let one ← iconst64 1
  let value ← emitParseScalarExpr ptr start len
  return (value, one)

/-- `scalar * <array>` on the right of a `*`: `[status, value, mode]`, with
    `status = 0` meaning it was not that, so the caller evaluates a scalar. -/
def emitScaleByRhs (ptr rhsStart len lhsValue : R) : M (List R) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
  let a ← iconst64 asciiA
  let lb ← iconst64 asciiLBracket
  let z8 ← iconst .i8 0
  let rhsCh ← loadInputByte ptr rhsStart
  ifte .eq rhsCh lb
    (do
      let ok ← emitValidateInputArray ptr rhsStart len
      ifte .eq ok one
        (do
          let outLen ← emitScaleInputArrayToOutput ptr rhsStart len lhsValue
          return [one, zero, two, outLen])
        (pure [one, zero, zero, zero]))
    (do
      let rhsVarIdx ← isub rhsCh a
      let rhsIsVar ← emitIsVarChar rhsCh
      let rhsKind ← loadVarKind ptr rhsVarIdx
      let rhsIsArray ← icmp .eq rhsKind one
      let useRhsArray ← band rhsIsVar rhsIsArray
      ifte .ne useRhsArray z8
        (do
          let varLen ← emitCopyVarTextToInput ptr rhsVarIdx
          let inputStart ← iconst64 0
          let outLen ← emitScaleInputArrayToOutput ptr inputStart varLen lhsValue
          return [one, zero, two, outLen])
        (pure [zero, zero, zero, zero]))

/-- A scalar term followed by `* <array>`, or not: `[status, value, mode,
    outLen]`. `status = 0` leaves the line to a plain scalar evaluation. -/
def emitScalarStarChain (ptr p len : R) : M (List R) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let star ← iconst64 asciiStar
  let (lhsValue, lhsEnd) ← emitScalarTermValue ptr p len
  let lhsNext ← emitSkipWs ptr lhsEnd len
  ifte .uge lhsNext len (pure [zero, zero, zero, zero])
    (do
      let ch ← loadInputByte ptr lhsNext
      ifte .ne ch star (pure [zero, zero, zero, zero])
        (do
          let rhsStart ← emitSkipWs ptr (← iadd lhsNext one) len
          ifte .uge rhsStart len (pure [zero, zero, zero, zero])
            (emitScaleByRhs ptr rhsStart len lhsValue)))

/-- An array literal or array variable used on its own: `[status, mode,
    outLen]`, `status = 0` when the text is neither. -/
def emitArrayExpr (ptr p len : R) : M (List R) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
  let a ← iconst64 asciiA
  let lb ← iconst64 asciiLBracket
  let star ← iconst64 asciiStar
  let plus ← iconst64 asciiPlus
  let z8 ← iconst .i8 0
  let firstCh ← loadInputByte ptr p
  ifte .eq firstCh lb
    (do
      let ok ← emitValidateInputArray ptr p len
      ifte .eq ok one
        (do
          let plusPos ← emitFindInputChar ptr p len plus
          let starPos ← emitFindInputChar ptr p len star
          ifte .ult plusPos len
            (do
              let rhsStart ← emitSkipWs ptr (← iadd plusPos one) len
              let rhsOk ← emitValidateInputArray ptr rhsStart len
              ifte .eq rhsOk one
                (do
                  let outLen ← emitAddInputArraysToOutput ptr p rhsStart len
                  return [one, two, outLen])
                (pure [one, zero, zero]))
            (ifte .ult starPos len
              (do
                let (scalar, _) ← emitParseInt ptr (← iadd starPos one) len
                let outLen ← emitScaleInputArrayToOutput ptr p len scalar
                return [one, two, outLen])
              (do
                let outLen ← emitCopyInputToOutput ptr p len
                return [one, two, outLen])))
        (pure [one, zero, zero]))
    (do
      let varIdx ← isub firstCh a
      let isVar ← emitIsVarChar firstCh
      let kind ← loadVarKind ptr varIdx
      let isArrayVar ← icmp .eq kind one
      let useArrayVar ← band isVar isArrayVar
      ifte .ne useArrayVar z8
        (do
          let starPos ← emitFindInputChar ptr p len star
          ifte .ult starPos len
            (do
              let (scalar, _) ← emitParseInt ptr (← iadd starPos one) len
              let varLen ← emitCopyVarTextToInput ptr varIdx
              let inputStart ← iconst64 0
              let outLen ← emitScaleInputArrayToOutput ptr inputStart varLen scalar
              return [one, two, outLen])
            (do
              let outLen ← emitCopyVarTextToOutput ptr varIdx
              return [one, two, outLen]))
        (pure [zero, zero, zero]))

/-- The right-hand side of a line with no assignment. The pair is the value and
    how the caller should print it -- `1` a number, `2` the output buffer,
    `0` nothing. -/
def emitExprLine (ptr exprStart len : R) : M (R × R) := do
  let zero ← iconst64 0
  let p ← emitSkipWs ptr exprStart len
  let arr ← emitArrayExpr ptr p len
  let r ← ifte .ne (arr.headD 0) zero
    (do
      fldStore ptr f.outputLen (arr.getD 2 0)
      return [zero, arr.getD 1 0])
    (do
      -- Each way of not being an array scaling lands on the same evaluation,
      -- which is why the decision is carried out rather than duplicated.
      let st ← emitScalarStarChain ptr p len
      ifte .ne (st.headD 0) zero
        (do
          fldStore ptr f.outputLen (st.getD 3 0)
          return [st.getD 1 0, st.getD 2 0])
        (do
          let (v, m) ← emitScalarLine ptr p len
          return [v, m]))
  return (r.headD 0, r.getD 1 0)

/-- An array result in the output buffer copied into a variable's text and its
    device buffer. -/
def emitStoreArrayToVar (ptr varIdx outLen : R) : M Unit := do
  let one ← iconst64 1
  emitCopyOutputToVarText ptr varIdx outLen
  emitUploadOutDataToVarBuffer ptr varIdx
  storeVarKind ptr varIdx one
  storeVarPresent ptr varIdx one

/-- `x = [ ... ]`, possibly with `+` or `*`. A malformed array leaves the
    variable alone. -/
def emitAssignArrayLiteral (ptr varIdx p len : R) : M Unit := do
  let one ← iconst64 1
  let star ← iconst64 asciiStar
  let plus ← iconst64 asciiPlus
  let ok ← emitValidateInputArray ptr p len
  let _ ← ifte .eq ok one
    (do
      let plusPos ← emitFindInputChar ptr p len plus
      let starPos ← emitFindInputChar ptr p len star
      ifte .ult plusPos len
        (do
          let rhsStart ← emitSkipWs ptr (← iadd plusPos one) len
          let rhsOk ← emitValidateInputArray ptr rhsStart len
          ifte .eq rhsOk one
            (do
              let outLen ← emitAddInputArraysToOutput ptr p rhsStart len
              emitStoreArrayToVar ptr varIdx outLen
              pure [])
            (pure []))
        (ifte .ult starPos len
          (do
            let (scalar, _) ← emitParseInt ptr (← iadd starPos one) len
            let outLen ← emitScaleInputArrayToOutput ptr p len scalar
            emitStoreArrayToVar ptr varIdx outLen
            pure [])
          (do
            emitCopyInputToVarText ptr varIdx p len
            emitUploadInputArrayToVarBuffer ptr varIdx p len
            storeVarKind ptr varIdx one
            storeVarPresent ptr varIdx one
            pure [])))
    (pure [])
  pure ()

/-- `x = y` or `x = y * n`, where `y` holds an array. The plain copy shares the
    other's text without re-uploading, since the buffer already holds it. -/
def emitAssignFromArrayVar (ptr varIdx rhsStart rhsVarIdx len : R) : M Unit := do
  let one ← iconst64 1
  let star ← iconst64 asciiStar
  let starPos ← emitFindInputChar ptr rhsStart len star
  let _ ← ifte .ult starPos len
    (do
      let (scalar, _) ← emitParseInt ptr (← iadd starPos one) len
      let varLen ← emitCopyVarTextToInput ptr rhsVarIdx
      let inputStart ← iconst64 0
      let outLen ← emitScaleInputArrayToOutput ptr inputStart varLen scalar
      emitStoreArrayToVar ptr varIdx outLen
      pure [])
    (do
      let outLen ← emitCopyVarTextToOutput ptr rhsVarIdx
      emitCopyOutputToVarText ptr varIdx outLen
      storeVarKind ptr varIdx one
      storeVarPresent ptr varIdx one
      pure [])
  pure ()

/-- `x = ...`: the right-hand side evaluated and stored in variable `varIdx`.
    `status = 0` from the tree below asks for the scalar store, which every arm
    that declines an array shares. -/
def emitAssignLine (ptr varIdx eqPos len : R) : M (R × R) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
  let a ← iconst64 asciiA
  let lb ← iconst64 asciiLBracket
  let z8 ← iconst .i8 0

  let exprPos0 ← emitSkipWs ptr (← iadd eqPos one) len

  let r ← ifte .uge exprPos0 len (pure [zero])
    (do
      let firstCh ← loadInputByte ptr exprPos0
      ifte .eq firstCh lb
        (do
          emitAssignArrayLiteral ptr varIdx exprPos0 len
          pure [one])
        (do
          let rhsStart ← emitSkipWs ptr exprPos0 len
          ifte .uge rhsStart len (pure [zero])
            (do
              let rhsCh ← loadInputByte ptr rhsStart
              let rhsVarIdx ← isub rhsCh a
              let rhsIsVar ← emitIsVarChar rhsCh
              let rhsKind ← loadVarKind ptr rhsVarIdx
              let rhsIsArray ← icmp .eq rhsKind one
              let useRhsArray ← band rhsIsVar rhsIsArray
              ifte .ne useRhsArray z8
                (do
                  emitAssignFromArrayVar ptr varIdx rhsStart rhsVarIdx len
                  pure [one])
                (do
                  let st ← emitScalarStarChain ptr rhsStart len
                  ifte .ne (st.headD 0) zero
                    (do
                      let _ ← ifte .eq (st.getD 2 0) two
                        (do emitStoreArrayToVar ptr varIdx (st.getD 3 0); pure [])
                        (pure [])
                      pure [one])
                    (pure [zero])))))

  let out ← ifte .ne (r.headD 0) zero (pure [zero, zero])
    (do
      let scalarValue ← emitParseScalarExpr ptr exprPos0 len
      let accBuf ← ireduce32 (← fldLoad ptr f.accBuf)
      let zeroBuf ← ireduce32 (← fldLoad ptr f.zeroBuf)
      let varBuf ← ireduce32 (← loadVarBufId ptr varIdx)
      emitAccFromLiteral ptr scalarValue
      emitCudaLaunchAdd ptr zeroBuf accBuf varBuf
      storeVarKind ptr varIdx zero
      storeVarPresent ptr varIdx one
      return [scalarValue, zero])
  return (out.headD 0, out.getD 1 0)

/-- One line of input: its value, and how the caller should print it. -/
def emitEvalLine (ptr len : R) : M (R × R) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
  let eqCh ← iconst64 asciiEq
  let a ← iconst64 asciiA
  let z8 ← iconst .i8 0

  let pos0 ← emitSkipWs ptr zero len

  let r ← ifte .uge pos0 len (pure [zero, zero])
    (do
      let ch ← loadInputByte ptr pos0
      let isVar ← emitIsVarChar ch
      let idx ← isub ch a
      let nextPos ← iadd pos0 one
      ifte .ne isVar z8
        (do
          let eqPos ← emitSkipWs ptr nextPos len
          ifte .uge eqPos len
            (do
              -- A name on its own prints what it holds.
              let kind ← loadVarKind ptr idx
              ifte .eq kind one
                (do
                  let outLen ← emitCopyVarTextToOutput ptr idx
                  fldStore ptr f.outputLen outLen
                  return [zero, two])
                (do
                  emitAccFromVar ptr idx
                  let value ← emitDownloadAccToResult ptr
                  return [value, one]))
            (do
              let ch2 ← loadInputByte ptr eqPos
              ifte .eq ch2 eqCh
                (do
                  let (v, m) ← emitAssignLine ptr idx eqPos len
                  return [v, m])
                (do
                  let (v, m) ← emitExprLine ptr pos0 len
                  return [v, m])))
        (do
          let (v, m) ← emitExprLine ptr pos0 len
          return [v, m]))
  return (r.headD 0, r.getD 1 0)

def clifCode : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let inputOff ← fldOffset f.input
  let inputMax ← iconst64 256
  let outOff ← fldOffset f.output
  let size8 ← iconst64 8
  let arrayBytes ← iconst64 2048
  let paramBytes ← iconst64 16
  let zero ← iconst64 0
  let one ← iconst64 1

  cudaInit cuda ptr
  let zeroBuf ← cudaCreateBuffer cuda ptr size8
  let litBuf ← cudaCreateBuffer cuda ptr size8
  let accBuf ← cudaCreateBuffer cuda ptr size8
  let outBuf ← cudaCreateBuffer cuda ptr size8
  let tmpArrayBufA ← cudaCreateBuffer cuda ptr arrayBytes
  let tmpArrayBufB ← cudaCreateBuffer cuda ptr arrayBytes
  let tmpArrayBufOut ← cudaCreateBuffer cuda ptr arrayBytes
  let tmpArrayParamBuf ← cudaCreateBuffer cuda ptr paramBytes
  fldStore ptr f.zeroBuf (← sextend64 zeroBuf)
  fldStore ptr f.litBuf (← sextend64 litBuf)
  fldStore ptr f.accBuf (← sextend64 accBuf)
  fldStore ptr f.outBuf (← sextend64 outBuf)
  fldStore ptr f.tmpArrayBufA (← sextend64 tmpArrayBufA)
  fldStore ptr f.tmpArrayBufB (← sextend64 tmpArrayBufB)
  fldStore ptr f.tmpArrayBufOut (← sextend64 tmpArrayBufOut)
  fldStore ptr f.tmpArrayParamBuf (← sextend64 tmpArrayParamBuf)

  -- One device buffer per variable, made once.
  let twentySix ← iconst64 26
  let _ ← wloop1 zero
    (head := fun vi => return (contIfULt vi twentySix, ([] : List R), ()))
    (body := fun vi _ => do
      let vbuf ← cudaCreateBuffer cuda ptr arrayBytes
      storeVarBufId ptr vi vbuf
      storeVarPresent ptr vi zero
      storeVarKind ptr vi zero
      storeVarTextLen ptr vi zero
      return [← iadd vi one])

  -- The prompt and the read are the loop's condition: an empty read is the end
  -- of input.
  let _ ← wloop []
    (head := fun _ => do
      storeOutputByte ptr zero (← iconst64 asciiGreater)
      storeOutputByte ptr one (← iconst64 asciiSpace)
      let two ← iconst64 2
      let _ ← call fnWrite.id [ptr, outOff, two]
      let readLen ← call fnRead.id [ptr, inputOff, inputMax]
      return (contIf .ne readLen zero, ([] : List R), readLen))
    (body := fun _ readLen => do
      fldStore ptr f.inputLen readLen
      let (lineResult, shouldPrint) ← emitEvalLine ptr readLen
      fldStore ptr f.result lineResult
      let _ ← ifte .eq shouldPrint one
        (do
          let outLen ← emitFormatSigned ptr lineResult
          fldStore ptr f.outputLen outLen
          let _ ← call fnWrite.id [ptr, outOff, outLen]
          pure [])
        (do
          let two ← iconst64 2
          let _ ← ifte .eq shouldPrint two
            (do
              let rawLen ← fldLoad ptr f.outputLen
              let _ ← call fnWrite.id [ptr, outOff, rawLen]
              pure [])
            (pure [])
          pure [])
      return [])

  cudaCleanup cuda ptr

/-- The body is well formed against the two bundles it calls.

    `clifCode` is a `Sur.build` rather than a `clif%` splice, so the builder run
    is still part of the term and the kernel would have to reduce it before it
    could look at a single statement. The compiler evaluates the same check
    directly. -/
theorem clif_wf : HProg.wf env HProg.ptrParams clifCode = true := by native_decide

def clifIrSource : Program :=
  IR.program [IR.noopFunction, HProg.compileFn 1 clifCode env (hwf := clif_wf)]

def payloads : List UInt8 :=
  mkPayload layoutMeta.totalSize [
    f.ptx.init (stringToBytes scalarAddPtx),
    f.arrayScalePtx.init (stringToBytes arrayScalePtxSource),
    f.arrayAddPtx.init (stringToBytes arrayAddPtxSource)
  ]

def cliConfig : Setup := {
  clif := clifIrSource,
  memory_size := layoutMeta.totalSize,
  initial_memory := payloads
}

def cliAlgorithm : Algorithm := {
  fn_idx := IR.mainFnIdx
}

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[toJsonEntry "cli_app" Algorithm.cliConfig Algorithm.cliAlgorithm]
