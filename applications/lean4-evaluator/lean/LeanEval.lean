import AlgorithmLib

open Lean (Json)
open AlgorithmLib

namespace LeanEval

-- Payload layout
def OUTPUT_PATH    : Nat := 0x0020
def INPUT_PATH     : Nat := 0x0060
def SOURCE_BUF     : Nat := 0x0160
def SOURCE_BUF_SZ  : Nat := 4096
def TRUE_STR       : Nat := 0x1160
def FALSE_STR      : Nat := 0x1168
def IDENT_BUF      : Nat := 0x1170
def IDENT_BUF_SZ   : Nat := 64
def HT_VAL_BUF     : Nat := 0x11B0
def OUTPUT_BUF     : Nat := 0x11B8
def OUTPUT_BUF_SZ  : Nat := 64
def STACK_BASE     : Nat := 0x11F8
def STACK_SZ       : Nat := 512

def TIMEOUT_MS : Nat := 30000

-- ---------------------------------------------------------------------------
-- CLIF IR via DSL
-- ---------------------------------------------------------------------------

open AlgorithmLib.IR

-- Shared constants threaded across sub-functions
structure K where
  ptr : Val           -- shared memory base pointer
  zero : Val          -- iconst 0
  one : Val           -- iconst 1
  srcBase : Val       -- iconst SOURCE_BUF
  identOff : Val      -- iconst IDENT_BUF
  htValOff : Val      -- iconst HT_VAL_BUF
  outBufOff : Val     -- iconst OUTPUT_BUF
  stackOff : Val      -- iconst STACK_BASE
  frameSize : Val     -- iconst 24
  space : Val         -- iconst 32 (' ')
  newline : Val       -- iconst 10 ('\n')
  fnHtCreate : FnRef
  fnHtInsert : FnRef
  fnHtLookup : FnRef
  fnFileRead : FnRef
  fnFileWrite : FnRef

-- All declared blocks, bundled to reduce parameter count
structure B where
  -- skip_spaces
  skipSpHead        : IR.DeclaredBlock
  skipSpCheck       : IR.DeclaredBlock
  -- parse_expr
  parseExprEntry    : IR.DeclaredBlock
  exprOpCheck       : IR.DeclaredBlock
  exprCheckPlus     : IR.DeclaredBlock
  exprCheckMinus    : IR.DeclaredBlock
  exprCheckLt       : IR.DeclaredBlock
  exprCheckGt       : IR.DeclaredBlock
  exprAdd           : IR.DeclaredBlock
  exprSub           : IR.DeclaredBlock
  exprLt            : IR.DeclaredBlock
  exprGt            : IR.DeclaredBlock
  -- parse_term
  parseTermEntry    : IR.DeclaredBlock
  termOpCheck       : IR.DeclaredBlock
  termCheckMul      : IR.DeclaredBlock
  termMul           : IR.DeclaredBlock
  -- parse_atom dispatch
  parseAtomDispatch : IR.DeclaredBlock
  atomAfterSpaces   : IR.DeclaredBlock
  checkParen        : IR.DeclaredBlock
  checkLet          : IR.DeclaredBlock
  verifyLet         : IR.DeclaredBlock
  checkIf           : IR.DeclaredBlock
  verifyIf          : IR.DeclaredBlock
  checkTrue         : IR.DeclaredBlock
  verifyTrue        : IR.DeclaredBlock
  checkFalse        : IR.DeclaredBlock
  verifyFalse       : IR.DeclaredBlock
  -- literals and numbers
  trueLiteral       : IR.DeclaredBlock
  falseLiteral      : IR.DeclaredBlock
  parseNumber       : IR.DeclaredBlock
  numAccum          : IR.DeclaredBlock
  -- variable
  varIdent          : IR.DeclaredBlock
  storeIdentChar    : IR.DeclaredBlock
  varLookup         : IR.DeclaredBlock
  -- paren / let
  parenOrLambda     : IR.DeclaredBlock
  parenExpr         : IR.DeclaredBlock
  letBinding        : IR.DeclaredBlock
  letReadName       : IR.DeclaredBlock
  letStoreNameChar  : IR.DeclaredBlock
  letSkipAssign     : IR.DeclaredBlock
  letStoreAndContinue : IR.DeclaredBlock
  -- if-then-else
  ifEntry           : IR.DeclaredBlock
  ifSkipThen        : IR.DeclaredBlock
  ifSkipElse        : IR.DeclaredBlock
  ifSelect          : IR.DeclaredBlock
  -- lambda
  lambdaEntry          : IR.DeclaredBlock
  lambdaReadParam      : IR.DeclaredBlock
  lambdaStoreParamChar : IR.DeclaredBlock
  lambdaSkipArrow      : IR.DeclaredBlock
  lambdaScanBody       : IR.DeclaredBlock
  lambdaCheckOpen      : IR.DeclaredBlock
  lambdaFoundClose     : IR.DeclaredBlock
  lambdaDecrDepth      : IR.DeclaredBlock
  lambdaParseArg       : IR.DeclaredBlock
  lambdaBindAndEval    : IR.DeclaredBlock
  -- return dispatch
  returnFromSub    : IR.DeclaredBlock
  checkExprOp      : IR.DeclaredBlock
  exprOpReturn     : IR.DeclaredBlock
  exprAddOrSub     : IR.DeclaredBlock
  checkTermOp      : IR.DeclaredBlock
  termOpReturn     : IR.DeclaredBlock
  termMulResult    : IR.DeclaredBlock
  checkLetBind     : IR.DeclaredBlock
  checkLetBody     : IR.DeclaredBlock
  checkIfCond      : IR.DeclaredBlock
  checkIfThen      : IR.DeclaredBlock
  checkIfElse      : IR.DeclaredBlock
  rdCheckParen     : IR.DeclaredBlock
  parenClose       : IR.DeclaredBlock
  checkLambdaArg   : IR.DeclaredBlock
  checkLambdaBody  : IR.DeclaredBlock
  checkCmpRhs      : IR.DeclaredBlock
  -- comparisons
  cmpResult        : IR.DeclaredBlock
  cmpLt            : IR.DeclaredBlock
  cmpCheckLe       : IR.DeclaredBlock
  cmpLe            : IR.DeclaredBlock
  cmpCheckGt       : IR.DeclaredBlock
  cmpGt            : IR.DeclaredBlock
  cmpGe            : IR.DeclaredBlock
  -- output
  output           : IR.DeclaredBlock
  checkBoolVal     : IR.DeclaredBlock
  writeTrueBlk     : IR.DeclaredBlock
  writeFalseBlk    : IR.DeclaredBlock
  itoaFindDiv      : IR.DeclaredBlock
  itoaDivLoop      : IR.DeclaredBlock
  itoaWriteDigit   : IR.DeclaredBlock
  itoaDone         : IR.DeclaredBlock

-- ---- Sub-function 1: Setup, file read, skip_spaces ----
def emitSetup (k : K) (b : B) : IRBuilder Val := do
  -- Read source file
  let _ ← readFile k.ptr k.fnFileRead INPUT_PATH SOURCE_BUF
  -- Create hash table
  let ctxPtr ← load64 k.ptr  -- HT ctx ptr at offset 0
  let _ ← call k.fnHtCreate [ctxPtr]
  -- Push initial stack frame: tag=255 (done), val=0, extra=0
  let stackAddr ← iadd k.ptr k.stackOff
  let doneTag ← iconst64 255
  store doneTag stackAddr
  let c8 ← iconst64 8
  let addr_8 ← iadd stackAddr c8
  store k.zero addr_8
  let c16 ← iconst64 16
  let addr_16 ← iadd stackAddr c16
  store k.zero addr_16
  -- sp = 24 (one frame pushed), pos = 6 (skip "#eval ")
  let pos0 ← iconst64 6
  jump b.skipSpHead.ref [pos0, k.frameSize, ctxPtr, k.zero]
  pure ctxPtr

def emitSkipSpaces (k : K) (b : B) : IRBuilder Unit := do
  -- block1: skip_spaces head (pos, sp, ctx, is_bool)
  startBlock b.skipSpHead
  let pos := b.skipSpHead.param 0
  let sp := b.skipSpHead.param 1
  let ctx := b.skipSpHead.param 2
  let isBool := b.skipSpHead.param 3
  let srcAddr ← iadd k.ptr k.srcBase
  let byteAddr ← iadd srcAddr pos
  let ch ← uload8_64 byteAddr
  let isEnd ← icmp .eq ch k.zero
  brif isEnd b.parseExprEntry.ref [pos, sp, ctx, isBool]
              b.skipSpCheck.ref [pos, sp, ctx, isBool, ch]

  -- block2: skip_spaces check (pos, sp, ctx, is_bool, ch)
  startBlock b.skipSpCheck
  let pos2 := b.skipSpCheck.param 0
  let sp2 := b.skipSpCheck.param 1
  let ctx2 := b.skipSpCheck.param 2
  let isBool2 := b.skipSpCheck.param 3
  let ch2 := b.skipSpCheck.param 4
  let isSpace ← icmp .eq ch2 k.space
  let isNl ← icmp .eq ch2 k.newline
  let isWs ← bor isSpace isNl
  let nextPos ← iadd pos2 k.one
  brif isWs b.skipSpHead.ref [nextPos, sp2, ctx2, isBool2]
            b.parseExprEntry.ref [pos2, sp2, ctx2, isBool2]

-- ---- Sub-function 2: parse_expr (blocks 3-4, 70-73, 109) ----
def emitParseExpr (k : K) (b : B) : IRBuilder Unit := do
  -- block3: parse_expr entry (pos, sp, ctx, is_bool)
  startBlock b.parseExprEntry
  let pos := b.parseExprEntry.param 0
  let sp := b.parseExprEntry.param 1
  let ctx := b.parseExprEntry.param 2
  let isBool := b.parseExprEntry.param 3
  -- Push frame: tag=0 (expr_op check), val=0, extra=is_bool
  let frameAddr ← iadd k.ptr k.stackOff
  let spAddr ← iadd frameAddr sp
  store k.zero spAddr                         -- tag=0
  let c8 ← iconst64 8
  let spAddr8 ← iadd spAddr c8
  store k.zero spAddr8                        -- val=0
  let c16 ← iconst64 16
  let spAddr16 ← iadd spAddr c16
  store isBool spAddr16                       -- extra=is_bool
  let newSp ← iadd sp k.frameSize
  jump b.parseTermEntry.ref [pos, newSp, ctx]

  -- block4: parse_expr operator check (pos, sp, ctx, value, is_bool)
  startBlock b.exprOpCheck
  let ePos := b.exprOpCheck.param 0
  let eSp := b.exprOpCheck.param 1
  let eCtx := b.exprOpCheck.param 2
  let eVal := b.exprOpCheck.param 3
  let eIsBool := b.exprOpCheck.param 4
  let eSrcAddr ← iadd k.ptr k.srcBase
  let eByteAddr ← iadd eSrcAddr ePos
  let eCh ← uload8_64 eByteAddr
  let eIsSp ← icmp .eq eCh k.space
  let eIsNl ← icmp .eq eCh k.newline
  let eIsWs ← bor eIsSp eIsNl
  let eNext ← iadd ePos k.one
  brif eIsWs b.exprOpCheck.ref [eNext, eSp, eCtx, eVal, eIsBool]
             b.exprCheckPlus.ref [ePos, eSp, eCtx, eVal, eIsBool, eCh]

  -- block70: check '+'
  startBlock b.exprCheckPlus
  let p0 := b.exprCheckPlus.param 0
  let s0 := b.exprCheckPlus.param 1
  let c0 := b.exprCheckPlus.param 2
  let v0 := b.exprCheckPlus.param 3
  let b0 := b.exprCheckPlus.param 4
  let ch0 := b.exprCheckPlus.param 5
  let plus ← iconst64 43  -- '+'
  let isPlus ← icmp .eq ch0 plus
  brif isPlus b.exprAdd.ref [p0, s0, c0, v0]
              b.exprCheckMinus.ref [p0, s0, c0, v0, b0, ch0]

  -- block71: check '-'
  startBlock b.exprCheckMinus
  let p1 := b.exprCheckMinus.param 0
  let s1 := b.exprCheckMinus.param 1
  let c1 := b.exprCheckMinus.param 2
  let v1 := b.exprCheckMinus.param 3
  let b1 := b.exprCheckMinus.param 4
  let ch1 := b.exprCheckMinus.param 5
  let minus ← iconst64 45  -- '-'
  let isMinus ← icmp .eq ch1 minus
  brif isMinus b.exprSub.ref [p1, s1, c1, v1]
               b.exprCheckLt.ref [p1, s1, c1, v1, b1, ch1]

  -- block72: check '<'
  startBlock b.exprCheckLt
  let p2 := b.exprCheckLt.param 0
  let s2 := b.exprCheckLt.param 1
  let c2 := b.exprCheckLt.param 2
  let v2 := b.exprCheckLt.param 3
  let b2 := b.exprCheckLt.param 4
  let ch2 := b.exprCheckLt.param 5
  let ltChar ← iconst64 60  -- '<'
  let isLt ← icmp .eq ch2 ltChar
  brif isLt b.exprLt.ref [p2, s2, c2, v2]
            b.exprCheckGt.ref [p2, s2, c2, v2, b2, ch2]

  -- block109: check '>'
  startBlock b.exprCheckGt
  let p3 := b.exprCheckGt.param 0
  let s3 := b.exprCheckGt.param 1
  let c3 := b.exprCheckGt.param 2
  let v3 := b.exprCheckGt.param 3
  let b3 := b.exprCheckGt.param 4
  let ch3 := b.exprCheckGt.param 5
  let gtChar ← iconst64 62  -- '>'
  let isGt ← icmp .eq ch3 gtChar
  brif isGt b.exprGt.ref [p3, s3, c3, v3]
            b.returnFromSub.ref [p3, s3, c3, v3, b3]

-- ---- Sub-function 3: expr add/sub operators ----
def emitExprAddSub (k : K) (b : B) : IRBuilder Unit := do
  -- block5: parse_expr add (pos, sp, ctx, left_value)
  startBlock b.exprAdd
  let pos := b.exprAdd.param 0
  let sp := b.exprAdd.param 1
  let ctx := b.exprAdd.param 2
  let left := b.exprAdd.param 3
  let pos1 ← iadd pos k.one  -- skip '+'
  -- skip space after '+'
  let srcAddr ← iadd k.ptr k.srcBase
  let chAddr ← iadd srcAddr pos1
  let ch ← uload8_64 chAddr
  let isSp ← icmp .eq ch k.space
  let pos2 ← iadd pos1 k.one
  let adjPos ← select' isSp pos2 pos1
  -- Push frame: tag=0, val=left, extra=1 (add)
  let frameAddr ← iadd k.ptr k.stackOff
  let spAddr ← iadd frameAddr sp
  store k.zero spAddr                         -- tag=0
  let c8 ← iconst64 8
  let spAddr8 ← iadd spAddr c8
  store left spAddr8                          -- saved left
  let c16 ← iconst64 16
  let spAddr16 ← iadd spAddr c16
  store k.one spAddr16                        -- extra=1 (add)
  let newSp ← iadd sp k.frameSize
  jump b.parseTermEntry.ref [adjPos, newSp, ctx]

  -- block6: parse_expr sub (pos, sp, ctx, left_value)
  startBlock b.exprSub
  let sPos := b.exprSub.param 0
  let sSp := b.exprSub.param 1
  let sCtx := b.exprSub.param 2
  let sLeft := b.exprSub.param 3
  let sPos1 ← iadd sPos k.one  -- skip '-'
  let sSrcAddr ← iadd k.ptr k.srcBase
  let sChAddr ← iadd sSrcAddr sPos1
  let sCh ← uload8_64 sChAddr
  let sIsSp ← icmp .eq sCh k.space
  let sPos2 ← iadd sPos1 k.one
  let sAdjPos ← select' sIsSp sPos2 sPos1
  -- Push frame: tag=0, val=left, extra=2 (sub)
  let sFrameAddr ← iadd k.ptr k.stackOff
  let sSpAddr ← iadd sFrameAddr sSp
  store k.zero sSpAddr
  let sc8 ← iconst64 8
  let sSpAddr8 ← iadd sSpAddr sc8
  store sLeft sSpAddr8
  let two ← iconst64 2
  let sc16 ← iconst64 16
  let sSpAddr16 ← iadd sSpAddr sc16
  store two sSpAddr16                         -- extra=2 (sub)
  let sNewSp ← iadd sSp k.frameSize
  jump b.parseTermEntry.ref [sAdjPos, sNewSp, sCtx]

-- ---- Sub-function 4: expr comparison operators (< <= > >=) ----
def emitExprCmp (k : K) (b : B) : IRBuilder Unit := do
  -- block7: parse_expr '<' comparison (pos, sp, ctx, left_value)
  startBlock b.exprLt
  let pos := b.exprLt.param 0
  let sp := b.exprLt.param 1
  let ctx := b.exprLt.param 2
  let left := b.exprLt.param 3
  let pos1 ← iadd pos k.one  -- skip '<'
  -- check for '<='
  let srcAddr ← iadd k.ptr k.srcBase
  let chAddr ← iadd srcAddr pos1
  let ch ← uload8_64 chAddr
  let eqChar ← iconst64 61  -- '='
  let isEq ← icmp .eq ch eqChar
  let pos2 ← iadd pos1 k.one
  let adjPos ← select' isEq pos2 pos1  -- skip '=' if present
  -- skip space
  let chAddr2 ← iadd srcAddr adjPos
  let ch2 ← uload8_64 chAddr2
  let isSp ← icmp .eq ch2 k.space
  let adjPos2 ← iadd adjPos k.one
  let finalPos ← select' isSp adjPos2 adjPos
  -- Push frame: tag=10 (cmp_rhs), val=left, extra=op (0=lt, 1=le)
  let frameAddr ← iadd k.ptr k.stackOff
  let spAddr ← iadd frameAddr sp
  let tag10 ← iconst64 10
  store tag10 spAddr
  let c8 ← iconst64 8
  let spAddr8 ← iadd spAddr c8
  store left spAddr8
  let leOp ← iconst64 1
  let ltOp ← iconst64 0
  let op ← select' isEq leOp ltOp
  let c16 ← iconst64 16
  let spAddr16 ← iadd spAddr c16
  store op spAddr16
  let newSp ← iadd sp k.frameSize
  jump b.parseTermEntry.ref [finalPos, newSp, ctx]

  -- block8: parse_expr '>' comparison (pos, sp, ctx, left_value)
  startBlock b.exprGt
  let gPos := b.exprGt.param 0
  let gSp := b.exprGt.param 1
  let gCtx := b.exprGt.param 2
  let gLeft := b.exprGt.param 3
  let gPos1 ← iadd gPos k.one  -- skip '>'
  let gSrcAddr ← iadd k.ptr k.srcBase
  let gChAddr ← iadd gSrcAddr gPos1
  let gCh ← uload8_64 gChAddr
  let gEqChar ← iconst64 61
  let gIsEq ← icmp .eq gCh gEqChar
  let gPos2 ← iadd gPos1 k.one
  let gAdjPos ← select' gIsEq gPos2 gPos1
  let gChAddr2 ← iadd gSrcAddr gAdjPos
  let gCh2 ← uload8_64 gChAddr2
  let gIsSp ← icmp .eq gCh2 k.space
  let gAdjPos2 ← iadd gAdjPos k.one
  let gFinalPos ← select' gIsSp gAdjPos2 gAdjPos
  let gFrameAddr ← iadd k.ptr k.stackOff
  let gSpAddr ← iadd gFrameAddr gSp
  let gTag10 ← iconst64 10
  store gTag10 gSpAddr
  let gc8 ← iconst64 8
  let gSpAddr8 ← iadd gSpAddr gc8
  store gLeft gSpAddr8
  let geOp ← iconst64 3
  let gtOp ← iconst64 2
  let gOp ← select' gIsEq geOp gtOp
  let gc16 ← iconst64 16
  let gSpAddr16 ← iadd gSpAddr gc16
  store gOp gSpAddr16
  let gNewSp ← iadd gSp k.frameSize
  jump b.parseTermEntry.ref [gFinalPos, gNewSp, gCtx]

-- ---- Sub-function 5: parse_term (blocks 10-12, 110) ----
def emitParseTerm (k : K) (b : B) : IRBuilder Unit := do
  -- block10: parse_term entry (pos, sp, ctx)
  startBlock b.parseTermEntry
  let pos := b.parseTermEntry.param 0
  let sp := b.parseTermEntry.param 1
  let ctx := b.parseTermEntry.param 2
  -- Push frame: tag=1 (term_op), val=0, extra=0
  let frameAddr ← iadd k.ptr k.stackOff
  let spAddr ← iadd frameAddr sp
  store k.one spAddr                          -- tag=1
  let c8 ← iconst64 8
  let spAddr8 ← iadd spAddr c8
  store k.zero spAddr8                        -- val=0
  let c16 ← iconst64 16
  let spAddr16 ← iadd spAddr c16
  store k.zero spAddr16                       -- extra=0
  let newSp ← iadd sp k.frameSize
  jump b.parseAtomDispatch.ref [pos, newSp, ctx]

  -- block11: parse_term operator check (pos, sp, ctx, value, is_bool)
  startBlock b.termOpCheck
  let tPos := b.termOpCheck.param 0
  let tSp := b.termOpCheck.param 1
  let tCtx := b.termOpCheck.param 2
  let tVal := b.termOpCheck.param 3
  let tIsBool := b.termOpCheck.param 4
  let tSrcAddr ← iadd k.ptr k.srcBase
  let tByteAddr ← iadd tSrcAddr tPos
  let tCh ← uload8_64 tByteAddr
  let tIsSp ← icmp .eq tCh k.space
  let tIsNl ← icmp .eq tCh k.newline
  let tIsWs ← bor tIsSp tIsNl
  let tNext ← iadd tPos k.one
  brif tIsWs b.termOpCheck.ref [tNext, tSp, tCtx, tVal, tIsBool]
             b.termCheckMul.ref [tPos, tSp, tCtx, tVal, tCh, tIsBool]

  -- block110: check '*'
  startBlock b.termCheckMul
  let mPos := b.termCheckMul.param 0
  let mSp := b.termCheckMul.param 1
  let mCtx := b.termCheckMul.param 2
  let mVal := b.termCheckMul.param 3
  let mCh := b.termCheckMul.param 4
  let mIsBool := b.termCheckMul.param 5
  let star ← iconst64 42  -- '*'
  let isStar ← icmp .eq mCh star
  brif isStar b.termMul.ref [mPos, mSp, mCtx, mVal]
              b.returnFromSub.ref [mPos, mSp, mCtx, mVal, mIsBool]

  -- block12: parse_term mul (pos, sp, ctx, left_value)
  startBlock b.termMul
  let muPos := b.termMul.param 0
  let muSp := b.termMul.param 1
  let muCtx := b.termMul.param 2
  let muLeft := b.termMul.param 3
  let muPos1 ← iadd muPos k.one  -- skip '*'
  let muSrcAddr ← iadd k.ptr k.srcBase
  let muChAddr ← iadd muSrcAddr muPos1
  let muCh ← uload8_64 muChAddr
  let muIsSp ← icmp .eq muCh k.space
  let muPos2 ← iadd muPos1 k.one
  let muAdjPos ← select' muIsSp muPos2 muPos1
  -- Push frame: tag=1 (term_op), val=left, extra=1 (mul)
  let muFrameAddr ← iadd k.ptr k.stackOff
  let muSpAddr ← iadd muFrameAddr muSp
  store k.one muSpAddr                        -- tag=1
  let mc8 ← iconst64 8
  let muSpAddr8 ← iadd muSpAddr mc8
  store muLeft muSpAddr8
  let mc16 ← iconst64 16
  let muSpAddr16 ← iadd muSpAddr mc16
  store k.one muSpAddr16                      -- extra=1 (mul)
  let muNewSp ← iadd muSp k.frameSize
  jump b.parseAtomDispatch.ref [muAdjPos, muNewSp, muCtx]

-- ---- Sub-function 6: parse_atom dispatch (block15, 111, 73-77, 112-115) ----
set_option maxRecDepth 4096 in
def emitAtomDispatch (k : K) (b : B) : IRBuilder Unit := do
  -- block15: parse_atom dispatch — skip spaces first (pos, sp, ctx)
  startBlock b.parseAtomDispatch
  let pos := b.parseAtomDispatch.param 0
  let sp := b.parseAtomDispatch.param 1
  let ctx := b.parseAtomDispatch.param 2
  let srcAddr ← iadd k.ptr k.srcBase
  let byteAddr ← iadd srcAddr pos
  let ch ← uload8_64 byteAddr
  let isSp ← icmp .eq ch k.space
  let isNl ← icmp .eq ch k.newline
  let isWs ← bor isSp isNl
  let nextPos ← iadd pos k.one
  brif isWs b.parseAtomDispatch.ref [nextPos, sp, ctx]
            b.atomAfterSpaces.ref [pos, sp, ctx, ch]

  -- block111: dispatch after space skip (pos, sp, ctx, ch)
  startBlock b.atomAfterSpaces
  let aPos := b.atomAfterSpaces.param 0
  let aSp := b.atomAfterSpaces.param 1
  let aCtx := b.atomAfterSpaces.param 2
  let aCh := b.atomAfterSpaces.param 3
  let d0 ← iconst64 48  -- '0'
  let d9 ← iconst64 57  -- '9'
  let geD0 ← icmp .uge aCh d0
  let leD9 ← icmp .ule aCh d9
  let isDigit ← band geD0 leD9
  brif isDigit b.parseNumber.ref [aPos, aSp, aCtx, k.zero]
               b.checkParen.ref [aPos, aSp, aCtx, aCh]

  -- block73: check '(' for paren/lambda
  startBlock b.checkParen
  let cpPos := b.checkParen.param 0
  let cpSp := b.checkParen.param 1
  let cpCtx := b.checkParen.param 2
  let cpCh := b.checkParen.param 3
  let openParen ← iconst64 40  -- '('
  let isParen ← icmp .eq cpCh openParen
  brif isParen b.parenOrLambda.ref [cpPos, cpSp, cpCtx]
               b.checkLet.ref [cpPos, cpSp, cpCtx, cpCh]

  -- block74: check 'l' for "let"
  startBlock b.checkLet
  let clPos := b.checkLet.param 0
  let clSp := b.checkLet.param 1
  let clCtx := b.checkLet.param 2
  let clCh := b.checkLet.param 3
  let lChar ← iconst64 108  -- 'l'
  let isL ← icmp .eq clCh lChar
  brif isL b.verifyLet.ref [clPos, clSp, clCtx]
           b.checkIf.ref [clPos, clSp, clCtx, clCh]

  -- block112: verify "et " after 'l'
  startBlock b.verifyLet
  let vlPos := b.verifyLet.param 0
  let vlSp := b.verifyLet.param 1
  let vlCtx := b.verifyLet.param 2
  let vlSrc ← iadd k.ptr k.srcBase
  let vlPos1 ← iadd vlPos k.one
  let vlAddr1 ← iadd vlSrc vlPos1
  let vlCh1 ← uload8_64 vlAddr1
  let eChar ← iconst64 101  -- 'e'
  let isE ← icmp .eq vlCh1 eChar
  let c2 ← iconst64 2
  let vlPos2 ← iadd vlPos c2
  let vlAddr2 ← iadd vlSrc vlPos2
  let vlCh2 ← uload8_64 vlAddr2
  let tChar ← iconst64 116  -- 't'
  let isT ← icmp .eq vlCh2 tChar
  let c3 ← iconst64 3
  let vlPos3 ← iadd vlPos c3
  let vlAddr3 ← iadd vlSrc vlPos3
  let vlCh3 ← uload8_64 vlAddr3
  let isSpc ← icmp .eq vlCh3 k.space
  let et ← band isE isT
  let etSp ← band et isSpc
  brif etSp b.letBinding.ref [vlPos, vlSp, vlCtx]
             b.varIdent.ref [vlPos, vlSp, vlCtx, k.zero]

  -- block75: check 'i' for "if"
  startBlock b.checkIf
  let ciPos := b.checkIf.param 0
  let ciSp := b.checkIf.param 1
  let ciCtx := b.checkIf.param 2
  let ciCh := b.checkIf.param 3
  let iChar ← iconst64 105  -- 'i'
  let isI ← icmp .eq ciCh iChar
  brif isI b.verifyIf.ref [ciPos, ciSp, ciCtx]
           b.checkTrue.ref [ciPos, ciSp, ciCtx, ciCh]

  -- block113: verify "f " after 'i'
  startBlock b.verifyIf
  let viPos := b.verifyIf.param 0
  let viSp := b.verifyIf.param 1
  let viCtx := b.verifyIf.param 2
  let viSrc ← iadd k.ptr k.srcBase
  let viPos1 ← iadd viPos k.one
  let viAddr1 ← iadd viSrc viPos1
  let viCh1 ← uload8_64 viAddr1
  let fChar ← iconst64 102  -- 'f'
  let isF ← icmp .eq viCh1 fChar
  let vic2 ← iconst64 2
  let viPos2 ← iadd viPos vic2
  let viAddr2 ← iadd viSrc viPos2
  let viCh2 ← uload8_64 viAddr2
  let viIsSpc ← icmp .eq viCh2 k.space
  let viMatch ← band isF viIsSpc
  brif viMatch b.ifEntry.ref [viPos, viSp, viCtx]
               b.varIdent.ref [viPos, viSp, viCtx, k.zero]

  -- block76: check 't' for "true"
  startBlock b.checkTrue
  let ctPos := b.checkTrue.param 0
  let ctSp := b.checkTrue.param 1
  let ctCtx := b.checkTrue.param 2
  let ctCh := b.checkTrue.param 3
  let tCh ← iconst64 116  -- 't'
  let isTC ← icmp .eq ctCh tCh
  brif isTC b.verifyTrue.ref [ctPos, ctSp, ctCtx]
            b.checkFalse.ref [ctPos, ctSp, ctCtx, ctCh]

  -- block114: verify "rue" after 't' and not followed by ident char
  startBlock b.verifyTrue
  let vtPos := b.verifyTrue.param 0
  let vtSp := b.verifyTrue.param 1
  let vtCtx := b.verifyTrue.param 2
  let vtSrc ← iadd k.ptr k.srcBase
  let vtPos1 ← iadd vtPos k.one
  let vtAddr1 ← iadd vtSrc vtPos1
  let vtCh1 ← uload8_64 vtAddr1
  let rChar ← iconst64 114  -- 'r'
  let isR ← icmp .eq vtCh1 rChar
  let vtc2 ← iconst64 2
  let vtPos2 ← iadd vtPos vtc2
  let vtAddr2 ← iadd vtSrc vtPos2
  let vtCh2 ← uload8_64 vtAddr2
  let uChar ← iconst64 117  -- 'u'
  let isU ← icmp .eq vtCh2 uChar
  let vtc3 ← iconst64 3
  let vtPos3 ← iadd vtPos vtc3
  let vtAddr3 ← iadd vtSrc vtPos3
  let vtCh3 ← uload8_64 vtAddr3
  let veChar ← iconst64 101  -- 'e'
  let isVE ← icmp .eq vtCh3 veChar
  -- Check char after "true" is not alphanumeric
  let vtc4 ← iconst64 4
  let vtPos4 ← iadd vtPos vtc4
  let vtAddr4 ← iadd vtSrc vtPos4
  let vtCh4 ← uload8_64 vtAddr4
  let aChar ← iconst64 97  -- 'a'
  let notAlpha ← icmp .ult vtCh4 aChar
  let ru ← band isR isU
  let rue ← band ru isVE
  let trueMatch ← band rue notAlpha
  brif trueMatch b.trueLiteral.ref [vtPos, vtSp, vtCtx]
                 b.varIdent.ref [vtPos, vtSp, vtCtx, k.zero]

  -- block77: check 'f' for "false"
  startBlock b.checkFalse
  let cfPos := b.checkFalse.param 0
  let cfSp := b.checkFalse.param 1
  let cfCtx := b.checkFalse.param 2
  let cfCh := b.checkFalse.param 3
  let fCh ← iconst64 102  -- 'f'
  let isFC ← icmp .eq cfCh fCh
  brif isFC b.verifyFalse.ref [cfPos, cfSp, cfCtx]
            b.varIdent.ref [cfPos, cfSp, cfCtx, k.zero]

  -- block115: verify "alse" after 'f' and not followed by ident char
  startBlock b.verifyFalse
  let vfPos := b.verifyFalse.param 0
  let vfSp := b.verifyFalse.param 1
  let vfCtx := b.verifyFalse.param 2
  let vfSrc ← iadd k.ptr k.srcBase
  let vfPos1 ← iadd vfPos k.one
  let vfAddr1 ← iadd vfSrc vfPos1
  let vfCh1 ← uload8_64 vfAddr1
  let vaChar ← iconst64 97  -- 'a'
  let isA ← icmp .eq vfCh1 vaChar
  let vfc2 ← iconst64 2
  let vfPos2 ← iadd vfPos vfc2
  let vfAddr2 ← iadd vfSrc vfPos2
  let vfCh2 ← uload8_64 vfAddr2
  let vlChar ← iconst64 108  -- 'l'
  let isVL ← icmp .eq vfCh2 vlChar
  let vfc3 ← iconst64 3
  let vfPos3 ← iadd vfPos vfc3
  let vfAddr3 ← iadd vfSrc vfPos3
  let vfCh3 ← uload8_64 vfAddr3
  let sChar ← iconst64 115  -- 's'
  let isS ← icmp .eq vfCh3 sChar
  let vfc4 ← iconst64 4
  let vfPos4 ← iadd vfPos vfc4
  let vfAddr4 ← iadd vfSrc vfPos4
  let vfCh4 ← uload8_64 vfAddr4
  let veChar2 ← iconst64 101  -- 'e'
  let isVE2 ← icmp .eq vfCh4 veChar2
  -- Check char after "false" is not alphanumeric
  let vfc5 ← iconst64 5
  let vfPos5 ← iadd vfPos vfc5
  let vfAddr5 ← iadd vfSrc vfPos5
  let vfCh5 ← uload8_64 vfAddr5
  let vaChar2 ← iconst64 97
  let vfNotAlpha ← icmp .ult vfCh5 vaChar2
  let al ← band isA isVL
  let als ← band al isS
  let alse ← band als isVE2
  let falseMatch ← band alse vfNotAlpha
  brif falseMatch b.falseLiteral.ref [vfPos, vfSp, vfCtx]
                  b.varIdent.ref [vfPos, vfSp, vfCtx, k.zero]

-- ---- Sub-function 7: literals, number, variable ----
set_option maxRecDepth 4096 in
def emitLiteralsAndVariable (k : K) (b : B) : IRBuilder Unit := do
  -- block63: true literal (pos, sp, ctx)
  startBlock b.trueLiteral
  let tPos := b.trueLiteral.param 0
  let tSp := b.trueLiteral.param 1
  let tCtx := b.trueLiteral.param 2
  let c4 ← iconst64 4
  let tNewPos ← iadd tPos c4  -- skip "true"
  jump b.returnFromSub.ref [tNewPos, tSp, tCtx, k.one, k.one]  -- val=1, is_bool=1

  -- block64: false literal (pos, sp, ctx)
  startBlock b.falseLiteral
  let fPos := b.falseLiteral.param 0
  let fSp := b.falseLiteral.param 1
  let fCtx := b.falseLiteral.param 2
  let c5 ← iconst64 5
  let fNewPos ← iadd fPos c5  -- skip "false"
  jump b.returnFromSub.ref [fNewPos, fSp, fCtx, k.zero, k.one]  -- val=0, is_bool=1

  -- block16: parse_number loop (pos, sp, ctx, accum)
  startBlock b.parseNumber
  let nPos := b.parseNumber.param 0
  let nSp := b.parseNumber.param 1
  let nCtx := b.parseNumber.param 2
  let nAcc := b.parseNumber.param 3
  let nSrcAddr ← iadd k.ptr k.srcBase
  let nByteAddr ← iadd nSrcAddr nPos
  let nCh ← uload8_64 nByteAddr
  let nd0 ← iconst64 48
  let nd9 ← iconst64 57
  let nGeD0 ← icmp .uge nCh nd0
  let nLeD9 ← icmp .ule nCh nd9
  let nIsDigit ← band nGeD0 nLeD9
  brif nIsDigit b.numAccum.ref [nPos, nSp, nCtx, nAcc, nCh]
                b.returnFromSub.ref [nPos, nSp, nCtx, nAcc, k.zero]  -- is_bool=0

  -- block17: parse_number accumulate (pos, sp, ctx, accum, ch)
  startBlock b.numAccum
  let naPos := b.numAccum.param 0
  let naSp := b.numAccum.param 1
  let naCtx := b.numAccum.param 2
  let naAcc := b.numAccum.param 3
  let naCh := b.numAccum.param 4
  let ten ← iconst64 10
  let scaled ← imul naAcc ten
  let nd0b ← iconst64 48
  let digit ← isub naCh nd0b
  let newAcc ← iadd scaled digit
  let naNext ← iadd naPos k.one
  jump b.parseNumber.ref [naNext, naSp, naCtx, newAcc]

  -- block18: variable — read identifier (pos, sp, ctx, id_len)
  startBlock b.varIdent
  let vPos := b.varIdent.param 0
  let vSp := b.varIdent.param 1
  let vCtx := b.varIdent.param 2
  let vIdLen := b.varIdent.param 3
  let vSrcAddr ← iadd k.ptr k.srcBase
  let vByteAddr ← iadd vSrcAddr vPos
  let vCh ← uload8_64 vByteAddr
  -- Check if ident char: a-z(97-122), A-Z(65-90), 0-9(48-57), _(95)
  let va ← iconst64 97
  let vz ← iconst64 122
  let vGeA ← icmp .uge vCh va
  let vLeZ ← icmp .ule vCh vz
  let vIsLower ← band vGeA vLeZ
  let vA ← iconst64 65
  let vZ ← iconst64 90
  let vGeUA ← icmp .uge vCh vA
  let vLeUZ ← icmp .ule vCh vZ
  let vIsUpper ← band vGeUA vLeUZ
  let v0 ← iconst64 48
  let v9 ← iconst64 57
  let vGe0 ← icmp .uge vCh v0
  let vLe9 ← icmp .ule vCh v9
  let vIsDigit ← band vGe0 vLe9
  let vUnder ← iconst64 95
  let vIsUnder ← icmp .eq vCh vUnder
  let vLU ← bor vIsLower vIsUpper
  let vLUD ← bor vLU vIsDigit
  let vIsIdent ← bor vLUD vIsUnder
  brif vIsIdent b.storeIdentChar.ref [vPos, vSp, vCtx, vIdLen, vCh]
                b.varLookup.ref [vPos, vSp, vCtx, vIdLen]

  -- block78: store ident char (pos, sp, ctx, id_len, ch)
  startBlock b.storeIdentChar
  let siPos := b.storeIdentChar.param 0
  let siSp := b.storeIdentChar.param 1
  let siCtx := b.storeIdentChar.param 2
  let siIdLen := b.storeIdentChar.param 3
  let siCh := b.storeIdentChar.param 4
  let siBufAddr ← iadd k.ptr k.identOff
  let siCharAddr ← iadd siBufAddr siIdLen
  istore8 siCh siCharAddr
  let siNewLen ← iadd siIdLen k.one
  let siNextPos ← iadd siPos k.one
  jump b.varIdent.ref [siNextPos, siSp, siCtx, siNewLen]

  -- block19: variable lookup (pos, sp, ctx, id_len)
  startBlock b.varLookup
  let vlPos := b.varLookup.param 0
  let vlSp := b.varLookup.param 1
  let vlCtx := b.varLookup.param 2
  let vlIdLen := b.varLookup.param 3
  let vlBufAddr ← iadd k.ptr k.identOff
  let vlIdLen32 ← ireduce32 vlIdLen
  let vlValAddr ← iadd k.ptr k.htValOff
  let _ ← call k.fnHtLookup [vlCtx, vlBufAddr, vlIdLen32, vlValAddr]
  let vlValue ← load64 vlValAddr
  jump b.returnFromSub.ref [vlPos, vlSp, vlCtx, vlValue, k.zero]  -- is_bool=0

-- ---- Sub-function 8: paren and let binding ----
def emitParenAndLet (k : K) (b : B) : IRBuilder Unit := do
  -- block20: paren or lambda (pos, sp, ctx)
  startBlock b.parenOrLambda
  let pos := b.parenOrLambda.param 0
  let sp := b.parenOrLambda.param 1
  let ctx := b.parenOrLambda.param 2
  let pos1 ← iadd pos k.one  -- after '('
  let srcAddr ← iadd k.ptr k.srcBase
  let chAddr ← iadd srcAddr pos1
  let ch ← uload8_64 chAddr
  let fChar ← iconst64 102  -- 'f'
  let isF ← icmp .eq ch fChar
  brif isF b.lambdaEntry.ref [pos, sp, ctx]
           b.parenExpr.ref [pos1, sp, ctx]

  -- block21: paren expr (pos_after_open, sp, ctx)
  startBlock b.parenExpr
  let pePos := b.parenExpr.param 0
  let peSp := b.parenExpr.param 1
  let peCtx := b.parenExpr.param 2
  -- Push frame: tag=7 (paren)
  let peFrameAddr ← iadd k.ptr k.stackOff
  let peSpAddr ← iadd peFrameAddr peSp
  let tag7 ← iconst64 7
  store tag7 peSpAddr
  let pec8 ← iconst64 8
  let peSpAddr8 ← iadd peSpAddr pec8
  store k.zero peSpAddr8
  let pec16 ← iconst64 16
  let peSpAddr16 ← iadd peSpAddr pec16
  store k.zero peSpAddr16
  let peNewSp ← iadd peSp k.frameSize
  jump b.skipSpHead.ref [pePos, peNewSp, peCtx, k.zero]

  -- block25: let binding (pos, sp, ctx) — pos at 'l', skip "let "
  startBlock b.letBinding
  let lPos := b.letBinding.param 0
  let lSp := b.letBinding.param 1
  let lCtx := b.letBinding.param 2
  let c4 ← iconst64 4
  let lPos1 ← iadd lPos c4  -- skip "let "
  jump b.letReadName.ref [lPos1, lSp, lCtx, k.zero]  -- id_len=0

  -- block26: let read name (pos, sp, ctx, id_len)
  startBlock b.letReadName
  let lnPos := b.letReadName.param 0
  let lnSp := b.letReadName.param 1
  let lnCtx := b.letReadName.param 2
  let lnIdLen := b.letReadName.param 3
  let lnSrcAddr ← iadd k.ptr k.srcBase
  let lnByteAddr ← iadd lnSrcAddr lnPos
  let lnCh ← uload8_64 lnByteAddr
  let lnIsSpc ← icmp .eq lnCh k.space
  brif lnIsSpc b.letSkipAssign.ref [lnPos, lnSp, lnCtx, lnIdLen]
               b.letStoreNameChar.ref [lnPos, lnSp, lnCtx, lnIdLen, lnCh]

  -- block79: store name char
  startBlock b.letStoreNameChar
  let lsPos := b.letStoreNameChar.param 0
  let lsSp := b.letStoreNameChar.param 1
  let lsCtx := b.letStoreNameChar.param 2
  let lsIdLen := b.letStoreNameChar.param 3
  let lsCh := b.letStoreNameChar.param 4
  let lsBufAddr ← iadd k.ptr k.identOff
  let lsCharAddr ← iadd lsBufAddr lsIdLen
  istore8 lsCh lsCharAddr
  let lsNewLen ← iadd lsIdLen k.one
  let lsNextPos ← iadd lsPos k.one
  jump b.letReadName.ref [lsNextPos, lsSp, lsCtx, lsNewLen]

  -- block27: let skip assign (pos, sp, ctx, id_len) — skip " := "
  startBlock b.letSkipAssign
  let laPos := b.letSkipAssign.param 0
  let laSp := b.letSkipAssign.param 1
  let laCtx := b.letSkipAssign.param 2
  let laIdLen := b.letSkipAssign.param 3
  let c4b ← iconst64 4
  let laPos1 ← iadd laPos c4b  -- skip " := "
  -- Push frame: tag=2 (let_bind), val=id_len, extra=0
  let laFrameAddr ← iadd k.ptr k.stackOff
  let laSpAddr ← iadd laFrameAddr laSp
  let tag2 ← iconst64 2
  store tag2 laSpAddr
  let lac8 ← iconst64 8
  let laSpAddr8 ← iadd laSpAddr lac8
  store laIdLen laSpAddr8  -- saved id_len
  let lac16 ← iconst64 16
  let laSpAddr16 ← iadd laSpAddr lac16
  store k.zero laSpAddr16
  let laNewSp ← iadd laSp k.frameSize
  jump b.skipSpHead.ref [laPos1, laNewSp, laCtx, k.zero]

-- ---- Sub-function 9: let store + continue, if-then-else ----
def emitLetStoreAndIf (k : K) (b : B) : IRBuilder Unit := do
  -- block29: let store and continue (pos, sp, ctx, bind_value, id_len)
  startBlock b.letStoreAndContinue
  let pos := b.letStoreAndContinue.param 0
  let sp := b.letStoreAndContinue.param 1
  let ctx := b.letStoreAndContinue.param 2
  let bindVal := b.letStoreAndContinue.param 3
  let idLen := b.letStoreAndContinue.param 4
  -- Store bind_value to HT_VAL_BUF
  let valAddr ← iadd k.ptr k.htValOff
  store bindVal valAddr
  let identAddr ← iadd k.ptr k.identOff
  let idLen32 ← ireduce32 idLen
  let valLen8 ← iconst32 8
  callVoid k.fnHtInsert [ctx, identAddr, idLen32, valAddr, valLen8]
  -- Skip "; " or ";\n"
  let srcAddr ← iadd k.ptr k.srcBase
  let chAddr ← iadd srcAddr pos
  let ch ← uload8_64 chAddr
  let semi ← iconst64 59  -- ';'
  let isSemi ← icmp .eq ch semi
  let nextPos ← iadd pos k.one
  let adjPos ← select' isSemi nextPos pos
  -- Push frame: tag=3 (let_body), val=0, extra=0
  let frameAddr ← iadd k.ptr k.stackOff
  let spAddr ← iadd frameAddr sp
  let tag3 ← iconst64 3
  store tag3 spAddr
  let c8 ← iconst64 8
  let spAddr8 ← iadd spAddr c8
  store k.zero spAddr8
  let c16 ← iconst64 16
  let spAddr16 ← iadd spAddr c16
  store k.zero spAddr16
  let newSp ← iadd sp k.frameSize
  jump b.skipSpHead.ref [adjPos, newSp, ctx, k.zero]

  -- block30: if-then-else entry (pos, sp, ctx) — pos at 'i', skip "if "
  startBlock b.ifEntry
  let iPos := b.ifEntry.param 0
  let iSp := b.ifEntry.param 1
  let iCtx := b.ifEntry.param 2
  let c3 ← iconst64 3
  let iPos1 ← iadd iPos c3  -- skip "if "
  -- Push frame: tag=4 (if_cond)
  let iFrameAddr ← iadd k.ptr k.stackOff
  let iSpAddr ← iadd iFrameAddr iSp
  let tag4 ← iconst64 4
  store tag4 iSpAddr
  let ic8 ← iconst64 8
  let iSpAddr8 ← iadd iSpAddr ic8
  store k.zero iSpAddr8
  let ic16 ← iconst64 16
  let iSpAddr16 ← iadd iSpAddr ic16
  store k.zero iSpAddr16
  let iNewSp ← iadd iSp k.frameSize
  jump b.skipSpHead.ref [iPos1, iNewSp, iCtx, k.zero]

  -- block32: if skip "then " (pos, sp, ctx, cond_value)
  startBlock b.ifSkipThen
  let stPos := b.ifSkipThen.param 0
  let stSp := b.ifSkipThen.param 1
  let stCtx := b.ifSkipThen.param 2
  let stCond := b.ifSkipThen.param 3
  -- skip leading space then "then "
  let stSrcAddr ← iadd k.ptr k.srcBase
  let stChAddr ← iadd stSrcAddr stPos
  let stCh ← uload8_64 stChAddr
  let stIsSp ← icmp .eq stCh k.space
  let stPos1 ← iadd stPos k.one
  let stAdjPos ← select' stIsSp stPos1 stPos
  let c5 ← iconst64 5
  let stPos2 ← iadd stAdjPos c5  -- skip "then "
  -- Push frame: tag=5 (if_then), val=cond
  let stFrameAddr ← iadd k.ptr k.stackOff
  let stSpAddr ← iadd stFrameAddr stSp
  let tag5 ← iconst64 5
  store tag5 stSpAddr
  let stc8 ← iconst64 8
  let stSpAddr8 ← iadd stSpAddr stc8
  store stCond stSpAddr8
  let stc16 ← iconst64 16
  let stSpAddr16 ← iadd stSpAddr stc16
  store k.zero stSpAddr16
  let stNewSp ← iadd stSp k.frameSize
  jump b.skipSpHead.ref [stPos2, stNewSp, stCtx, k.zero]

  -- block34: if skip "else " (pos, sp, ctx, cond, then_val)
  startBlock b.ifSkipElse
  let sePos := b.ifSkipElse.param 0
  let seSp := b.ifSkipElse.param 1
  let seCtx := b.ifSkipElse.param 2
  let seCond := b.ifSkipElse.param 3
  let seThenVal := b.ifSkipElse.param 4
  let seSrcAddr ← iadd k.ptr k.srcBase
  let seChAddr ← iadd seSrcAddr sePos
  let seCh ← uload8_64 seChAddr
  let seIsSp ← icmp .eq seCh k.space
  let sePos1 ← iadd sePos k.one
  let seAdjPos ← select' seIsSp sePos1 sePos
  let sec5 ← iconst64 5
  let sePos2 ← iadd seAdjPos sec5  -- skip "else "
  -- Push frame: tag=6 (if_else), val=cond, extra=then_val
  let seFrameAddr ← iadd k.ptr k.stackOff
  let seSpAddr ← iadd seFrameAddr seSp
  let tag6 ← iconst64 6
  store tag6 seSpAddr
  let sec8 ← iconst64 8
  let seSpAddr8 ← iadd seSpAddr sec8
  store seCond seSpAddr8
  let sec16 ← iconst64 16
  let seSpAddr16 ← iadd seSpAddr sec16
  store seThenVal seSpAddr16
  let seNewSp ← iadd seSp k.frameSize
  jump b.skipSpHead.ref [sePos2, seNewSp, seCtx, k.zero]

  -- block36: if select (pos, sp, ctx, cond, then_val, else_val)
  startBlock b.ifSelect
  let isPos := b.ifSelect.param 0
  let isSp := b.ifSelect.param 1
  let isCtx := b.ifSelect.param 2
  let isCond := b.ifSelect.param 3
  let isThen := b.ifSelect.param 4
  let isElse := b.ifSelect.param 5
  let condNZ ← icmp .ne isCond k.zero
  let result ← select' condNZ isThen isElse
  jump b.returnFromSub.ref [isPos, isSp, isCtx, result, k.zero]

-- ---- Sub-function 10: lambda ----
set_option maxRecDepth 4096 in
def emitLambda (k : K) (b : B) : IRBuilder Unit := do
  -- block40: lambda entry (pos, sp, ctx) — skip "(fun "
  startBlock b.lambdaEntry
  let pos := b.lambdaEntry.param 0
  let sp := b.lambdaEntry.param 1
  let ctx := b.lambdaEntry.param 2
  let c5 ← iconst64 5
  let pos1 ← iadd pos c5  -- skip "(fun "
  jump b.lambdaReadParam.ref [pos1, sp, ctx, k.zero]

  -- block41: lambda read param (pos, sp, ctx, id_len)
  startBlock b.lambdaReadParam
  let rpPos := b.lambdaReadParam.param 0
  let rpSp := b.lambdaReadParam.param 1
  let rpCtx := b.lambdaReadParam.param 2
  let rpIdLen := b.lambdaReadParam.param 3
  let rpSrcAddr ← iadd k.ptr k.srcBase
  let rpByteAddr ← iadd rpSrcAddr rpPos
  let rpCh ← uload8_64 rpByteAddr
  let rpIsSpc ← icmp .eq rpCh k.space
  brif rpIsSpc b.lambdaSkipArrow.ref [rpPos, rpSp, rpCtx, rpIdLen]
               b.lambdaStoreParamChar.ref [rpPos, rpSp, rpCtx, rpIdLen, rpCh]

  -- block80: store param char
  startBlock b.lambdaStoreParamChar
  let spPos := b.lambdaStoreParamChar.param 0
  let spSp := b.lambdaStoreParamChar.param 1
  let spCtx := b.lambdaStoreParamChar.param 2
  let spIdLen := b.lambdaStoreParamChar.param 3
  let spCh := b.lambdaStoreParamChar.param 4
  let spBufAddr ← iadd k.ptr k.identOff
  let spCharAddr ← iadd spBufAddr spIdLen
  istore8 spCh spCharAddr
  let spNewLen ← iadd spIdLen k.one
  let spNextPos ← iadd spPos k.one
  jump b.lambdaReadParam.ref [spNextPos, spSp, spCtx, spNewLen]

  -- block42: lambda skip " => " (pos, sp, ctx, id_len)
  startBlock b.lambdaSkipArrow
  let saPos := b.lambdaSkipArrow.param 0
  let saSp := b.lambdaSkipArrow.param 1
  let saCtx := b.lambdaSkipArrow.param 2
  let saIdLen := b.lambdaSkipArrow.param 3
  let c4 ← iconst64 4
  let saPos1 ← iadd saPos c4  -- skip " => "
  -- Push frame: tag=8 (lambda_arg), val=body_start_pos, extra=id_len
  let saFrameAddr ← iadd k.ptr k.stackOff
  let saSpAddr ← iadd saFrameAddr saSp
  let tag8 ← iconst64 8
  store tag8 saSpAddr
  let sac8 ← iconst64 8
  let saSpAddr8 ← iadd saSpAddr sac8
  store saPos1 saSpAddr8  -- body_start_pos
  let sac16 ← iconst64 16
  let saSpAddr16 ← iadd saSpAddr sac16
  store saIdLen saSpAddr16  -- id_len
  let saNewSp ← iadd saSp k.frameSize
  jump b.lambdaScanBody.ref [saPos1, saNewSp, saCtx, k.zero]

  -- block43: lambda scan body — scan to ')' (pos, sp, ctx, depth)
  startBlock b.lambdaScanBody
  let sbPos := b.lambdaScanBody.param 0
  let sbSp := b.lambdaScanBody.param 1
  let sbCtx := b.lambdaScanBody.param 2
  let sbDepth := b.lambdaScanBody.param 3
  let sbSrcAddr ← iadd k.ptr k.srcBase
  let sbByteAddr ← iadd sbSrcAddr sbPos
  let sbCh ← uload8_64 sbByteAddr
  let sbNext ← iadd sbPos k.one
  let closeParen ← iconst64 41  -- ')'
  let isClose ← icmp .eq sbCh closeParen
  brif isClose b.lambdaFoundClose.ref [sbNext, sbSp, sbCtx, sbDepth]
               b.lambdaCheckOpen.ref [sbNext, sbSp, sbCtx, sbDepth, sbCh]

  -- block100: not ')' — check '(' to increment depth
  startBlock b.lambdaCheckOpen
  let coPos := b.lambdaCheckOpen.param 0
  let coSp := b.lambdaCheckOpen.param 1
  let coCtx := b.lambdaCheckOpen.param 2
  let coDepth := b.lambdaCheckOpen.param 3
  let coCh := b.lambdaCheckOpen.param 4
  let openParen ← iconst64 40  -- '('
  let isOpen ← icmp .eq coCh openParen
  let coNewDepth ← iadd coDepth k.one
  brif isOpen b.lambdaScanBody.ref [coPos, coSp, coCtx, coNewDepth]
              b.lambdaScanBody.ref [coPos, coSp, coCtx, coDepth]

  -- block81: found ')' (pos, sp, ctx, depth) — check if depth==0
  startBlock b.lambdaFoundClose
  let fcPos := b.lambdaFoundClose.param 0
  let fcSp := b.lambdaFoundClose.param 1
  let fcCtx := b.lambdaFoundClose.param 2
  let fcDepth := b.lambdaFoundClose.param 3
  let isZero ← icmp .eq fcDepth k.zero
  brif isZero b.lambdaParseArg.ref [fcPos, fcSp, fcCtx]
              b.lambdaDecrDepth.ref [fcPos, fcSp, fcCtx, fcDepth]

  -- block101: ')' but depth > 0 — decrement and continue
  startBlock b.lambdaDecrDepth
  let ddPos := b.lambdaDecrDepth.param 0
  let ddSp := b.lambdaDecrDepth.param 1
  let ddCtx := b.lambdaDecrDepth.param 2
  let ddDepth := b.lambdaDecrDepth.param 3
  let ddNewDepth ← isub ddDepth k.one
  jump b.lambdaScanBody.ref [ddPos, ddSp, ddCtx, ddNewDepth]

  -- block44: lambda parse arg (pos_after_close, sp, ctx)
  startBlock b.lambdaParseArg
  let paPos := b.lambdaParseArg.param 0
  let paSp := b.lambdaParseArg.param 1
  let paCtx := b.lambdaParseArg.param 2
  -- skip space after ')'
  let paSrcAddr ← iadd k.ptr k.srcBase
  let paChAddr ← iadd paSrcAddr paPos
  let paCh ← uload8_64 paChAddr
  let paIsSp ← icmp .eq paCh k.space
  let paPos1 ← iadd paPos k.one
  let paAdjPos ← select' paIsSp paPos1 paPos
  -- Push frame: tag=9 (lambda_body)
  let paFrameAddr ← iadd k.ptr k.stackOff
  let paSpAddr ← iadd paFrameAddr paSp
  let tag9 ← iconst64 9
  store tag9 paSpAddr
  let pac8 ← iconst64 8
  let paSpAddr8 ← iadd paSpAddr pac8
  store k.zero paSpAddr8
  let pac16 ← iconst64 16
  let paSpAddr16 ← iadd paSpAddr pac16
  store k.zero paSpAddr16
  let paNewSp ← iadd paSp k.frameSize
  jump b.parseTermEntry.ref [paAdjPos, paNewSp, paCtx]

  -- block45: lambda bind and eval (pos, sp, ctx, arg_val, body_pos, id_len)
  startBlock b.lambdaBindAndEval
  let _bePos := b.lambdaBindAndEval.param 0
  let beSp := b.lambdaBindAndEval.param 1
  let beCtx := b.lambdaBindAndEval.param 2
  let beArgVal := b.lambdaBindAndEval.param 3
  let beBodyPos := b.lambdaBindAndEval.param 4
  let beIdLen := b.lambdaBindAndEval.param 5
  -- Store arg_val to HT_VAL_BUF, then ht_insert
  let beValAddr ← iadd k.ptr k.htValOff
  store beArgVal beValAddr
  let beIdentAddr ← iadd k.ptr k.identOff
  let beIdLen32 ← ireduce32 beIdLen
  let beValLen8 ← iconst32 8
  callVoid k.fnHtInsert [beCtx, beIdentAddr, beIdLen32, beValAddr, beValLen8]
  -- Push frame: tag=3 (let_body — reuse)
  let beFrameAddr ← iadd k.ptr k.stackOff
  let beSpAddr ← iadd beFrameAddr beSp
  let tag3 ← iconst64 3
  store tag3 beSpAddr
  let bec8 ← iconst64 8
  let beSpAddr8 ← iadd beSpAddr bec8
  store k.zero beSpAddr8
  let bec16 ← iconst64 16
  let beSpAddr16 ← iadd beSpAddr bec16
  store k.zero beSpAddr16
  let beNewSp ← iadd beSp k.frameSize
  jump b.skipSpHead.ref [beBodyPos, beNewSp, beCtx, k.zero]

-- ---- Sub-function 11: return dispatch (block60, 82-97) ----
set_option maxRecDepth 4096 in
def emitReturnDispatch (k : K) (b : B) : IRBuilder Unit := do
  -- block60: return from subexpr (pos, sp, ctx, value, is_bool)
  startBlock b.returnFromSub
  let pos := b.returnFromSub.param 0
  let sp := b.returnFromSub.param 1
  let ctx := b.returnFromSub.param 2
  let value := b.returnFromSub.param 3
  let isBool := b.returnFromSub.param 4
  -- Pop stack frame
  let newSp ← isub sp k.frameSize
  let frameAddr ← iadd k.ptr k.stackOff
  let fAddr ← iadd frameAddr newSp
  let tag ← load64 fAddr
  let c8 ← iconst64 8
  let fAddr8 ← iadd fAddr c8
  let savedVal ← load64 fAddr8
  let c16 ← iconst64 16
  let fAddr16 ← iadd fAddr c16
  let savedExtra ← load64 fAddr16
  -- Dispatch on tag
  let doneTag ← iconst64 255
  let isDone ← icmp .eq tag doneTag
  brif isDone b.output.ref [value, isBool]
              b.checkExprOp.ref [pos, newSp, ctx, value, isBool, tag, savedVal, savedExtra]

  -- block82: check tag=0 (expr_op)
  startBlock b.checkExprOp
  let p82 := b.checkExprOp.param 0
  let s82 := b.checkExprOp.param 1
  let c82 := b.checkExprOp.param 2
  let v82 := b.checkExprOp.param 3
  let b82 := b.checkExprOp.param 4
  let t82 := b.checkExprOp.param 5
  let sv82 := b.checkExprOp.param 6
  let se82 := b.checkExprOp.param 7
  let isTag0 ← icmp .eq t82 k.zero
  brif isTag0 b.exprOpReturn.ref [p82, s82, c82, v82, b82, sv82, se82]
              b.checkTermOp.ref [p82, s82, c82, v82, b82, t82, sv82, se82]

  -- block83: expr_op return — extra: 0=first_term, 1=add, 2=sub
  startBlock b.exprOpReturn
  let p83 := b.exprOpReturn.param 0
  let s83 := b.exprOpReturn.param 1
  let c83 := b.exprOpReturn.param 2
  let v83 := b.exprOpReturn.param 3
  let b83 := b.exprOpReturn.param 4
  let sv83 := b.exprOpReturn.param 5
  let se83 := b.exprOpReturn.param 6
  let isFirst ← icmp .eq se83 k.zero
  brif isFirst b.exprOpCheck.ref [p83, s83, c83, v83, b83]
               b.exprAddOrSub.ref [p83, s83, c83, v83, sv83, se83]

  -- block85: add or sub
  startBlock b.exprAddOrSub
  let p85 := b.exprAddOrSub.param 0
  let s85 := b.exprAddOrSub.param 1
  let c85 := b.exprAddOrSub.param 2
  let v85 := b.exprAddOrSub.param 3
  let sv85 := b.exprAddOrSub.param 4
  let se85 := b.exprAddOrSub.param 5
  let isAdd ← icmp .eq se85 k.one
  let addResult ← iadd sv85 v85
  let subResult ← isub sv85 v85
  let result ← select' isAdd addResult subResult
  jump b.exprOpCheck.ref [p85, s85, c85, result, k.zero]

  -- block84: check tag=1 (term_op)
  startBlock b.checkTermOp
  let p84 := b.checkTermOp.param 0
  let s84 := b.checkTermOp.param 1
  let c84 := b.checkTermOp.param 2
  let v84 := b.checkTermOp.param 3
  let b84 := b.checkTermOp.param 4
  let t84 := b.checkTermOp.param 5
  let sv84 := b.checkTermOp.param 6
  let se84 := b.checkTermOp.param 7
  let isTag1 ← icmp .eq t84 k.one
  brif isTag1 b.termOpReturn.ref [p84, s84, c84, v84, b84, sv84, se84]
              b.checkLetBind.ref [p84, s84, c84, v84, b84, t84, sv84, se84]

  -- block86: term_op return — extra: 0=first_atom, 1=mul
  startBlock b.termOpReturn
  let p86 := b.termOpReturn.param 0
  let s86 := b.termOpReturn.param 1
  let c86 := b.termOpReturn.param 2
  let v86 := b.termOpReturn.param 3
  let b86 := b.termOpReturn.param 4
  let sv86 := b.termOpReturn.param 5
  let se86 := b.termOpReturn.param 6
  let isFirst86 ← icmp .eq se86 k.zero
  brif isFirst86 b.termOpCheck.ref [p86, s86, c86, v86, b86]
                  b.termMulResult.ref [p86, s86, c86, v86, sv86]

  -- block88: multiply
  startBlock b.termMulResult
  let p88 := b.termMulResult.param 0
  let s88 := b.termMulResult.param 1
  let c88 := b.termMulResult.param 2
  let v88 := b.termMulResult.param 3
  let sv88 := b.termMulResult.param 4
  let mulResult ← imul sv88 v88
  jump b.termOpCheck.ref [p88, s88, c88, mulResult, k.zero]

  -- block87: check tag=2 (let_bind)
  startBlock b.checkLetBind
  let p87 := b.checkLetBind.param 0
  let s87 := b.checkLetBind.param 1
  let c87 := b.checkLetBind.param 2
  let v87 := b.checkLetBind.param 3
  let b87 := b.checkLetBind.param 4
  let t87 := b.checkLetBind.param 5
  let sv87 := b.checkLetBind.param 6
  let _se87 := b.checkLetBind.param 7
  let tag2 ← iconst64 2
  let isTag2 ← icmp .eq t87 tag2
  brif isTag2 b.letStoreAndContinue.ref [p87, s87, c87, v87, sv87]
              b.checkLetBody.ref [p87, s87, c87, v87, b87, t87, sv87, _se87]

  -- block89: check tag=3 (let_body / lambda_body_done)
  startBlock b.checkLetBody
  let p89 := b.checkLetBody.param 0
  let s89 := b.checkLetBody.param 1
  let c89 := b.checkLetBody.param 2
  let v89 := b.checkLetBody.param 3
  let b89 := b.checkLetBody.param 4
  let t89 := b.checkLetBody.param 5
  let sv89 := b.checkLetBody.param 6
  let se89 := b.checkLetBody.param 7
  let tag3 ← iconst64 3
  let isTag3 ← icmp .eq t89 tag3
  brif isTag3 b.returnFromSub.ref [p89, s89, c89, v89, b89]
              b.checkIfCond.ref [p89, s89, c89, v89, b89, t89, sv89, se89]

  -- block90: check tag=4 (if_cond)
  startBlock b.checkIfCond
  let p90 := b.checkIfCond.param 0
  let s90 := b.checkIfCond.param 1
  let c90 := b.checkIfCond.param 2
  let v90 := b.checkIfCond.param 3
  let b90 := b.checkIfCond.param 4
  let t90 := b.checkIfCond.param 5
  let sv90 := b.checkIfCond.param 6
  let se90 := b.checkIfCond.param 7
  let tag4 ← iconst64 4
  let isTag4 ← icmp .eq t90 tag4
  brif isTag4 b.ifSkipThen.ref [p90, s90, c90, v90]
              b.checkIfThen.ref [p90, s90, c90, v90, b90, t90, sv90, se90]

  -- block91: check tag=5 (if_then)
  startBlock b.checkIfThen
  let p91 := b.checkIfThen.param 0
  let s91 := b.checkIfThen.param 1
  let c91 := b.checkIfThen.param 2
  let v91 := b.checkIfThen.param 3
  let b91 := b.checkIfThen.param 4
  let t91 := b.checkIfThen.param 5
  let sv91 := b.checkIfThen.param 6
  let se91 := b.checkIfThen.param 7
  let tag5 ← iconst64 5
  let isTag5 ← icmp .eq t91 tag5
  brif isTag5 b.ifSkipElse.ref [p91, s91, c91, sv91, v91]
              b.checkIfElse.ref [p91, s91, c91, v91, b91, t91, sv91, se91]

  -- block92: check tag=6 (if_else)
  startBlock b.checkIfElse
  let p92 := b.checkIfElse.param 0
  let s92 := b.checkIfElse.param 1
  let c92 := b.checkIfElse.param 2
  let v92 := b.checkIfElse.param 3
  let b92 := b.checkIfElse.param 4
  let t92 := b.checkIfElse.param 5
  let sv92 := b.checkIfElse.param 6
  let se92 := b.checkIfElse.param 7
  let tag6 ← iconst64 6
  let isTag6 ← icmp .eq t92 tag6
  brif isTag6 b.ifSelect.ref [p92, s92, c92, sv92, se92, v92]
              b.rdCheckParen.ref [p92, s92, c92, v92, b92, t92, sv92, se92]

  -- block93: check tag=7 (paren)
  startBlock b.rdCheckParen
  let p93 := b.rdCheckParen.param 0
  let s93 := b.rdCheckParen.param 1
  let c93 := b.rdCheckParen.param 2
  let v93 := b.rdCheckParen.param 3
  let b93 := b.rdCheckParen.param 4
  let t93 := b.rdCheckParen.param 5
  let sv93 := b.rdCheckParen.param 6
  let se93 := b.rdCheckParen.param 7
  let tag7 ← iconst64 7
  let isTag7 ← icmp .eq t93 tag7
  brif isTag7 b.parenClose.ref [p93, s93, c93, v93, b93]
              b.checkLambdaArg.ref [p93, s93, c93, v93, b93, t93, sv93, se93]

  -- block94: paren close — skip ')' and return
  startBlock b.parenClose
  let p94 := b.parenClose.param 0
  let s94 := b.parenClose.param 1
  let c94 := b.parenClose.param 2
  let v94 := b.parenClose.param 3
  let b94 := b.parenClose.param 4
  let pcSrcAddr ← iadd k.ptr k.srcBase
  let pcChAddr ← iadd pcSrcAddr p94
  let pcCh ← uload8_64 pcChAddr
  let pcClose ← iconst64 41  -- ')'
  let pcIsClose ← icmp .eq pcCh pcClose
  let pcPos1 ← iadd p94 k.one
  let pcAdjPos ← select' pcIsClose pcPos1 p94
  jump b.returnFromSub.ref [pcAdjPos, s94, c94, v94, b94]

  -- block95: check tag=8 (lambda_arg)
  startBlock b.checkLambdaArg
  let p95 := b.checkLambdaArg.param 0
  let s95 := b.checkLambdaArg.param 1
  let c95 := b.checkLambdaArg.param 2
  let v95 := b.checkLambdaArg.param 3
  let b95 := b.checkLambdaArg.param 4
  let t95 := b.checkLambdaArg.param 5
  let sv95 := b.checkLambdaArg.param 6
  let se95 := b.checkLambdaArg.param 7
  let tag8 ← iconst64 8
  let isTag8 ← icmp .eq t95 tag8
  brif isTag8 b.lambdaBindAndEval.ref [p95, s95, c95, v95, sv95, se95]
              b.checkLambdaBody.ref [p95, s95, c95, v95, b95, t95, sv95, se95]

  -- block96: check tag=9 (lambda_body)
  startBlock b.checkLambdaBody
  let p96 := b.checkLambdaBody.param 0
  let s96 := b.checkLambdaBody.param 1
  let c96 := b.checkLambdaBody.param 2
  let v96 := b.checkLambdaBody.param 3
  let b96 := b.checkLambdaBody.param 4
  let t96 := b.checkLambdaBody.param 5
  let sv96 := b.checkLambdaBody.param 6
  let se96 := b.checkLambdaBody.param 7
  let tag9 ← iconst64 9
  let isTag9 ← icmp .eq t96 tag9
  brif isTag9 b.returnFromSub.ref [p96, s96, c96, v96, b96]
              b.checkCmpRhs.ref [p96, s96, c96, v96, b96, t96, sv96, se96]

  -- block97: check tag=10 (cmp_rhs)
  startBlock b.checkCmpRhs
  let p97 := b.checkCmpRhs.param 0
  let s97 := b.checkCmpRhs.param 1
  let c97 := b.checkCmpRhs.param 2
  let v97 := b.checkCmpRhs.param 3
  let b97 := b.checkCmpRhs.param 4
  let _t97 := b.checkCmpRhs.param 5
  let sv97 := b.checkCmpRhs.param 6
  let se97 := b.checkCmpRhs.param 7
  let tag10 ← iconst64 10
  let isTag10 ← icmp .eq _t97 tag10
  brif isTag10 b.cmpResult.ref [p97, s97, c97, sv97, v97, se97]
               b.returnFromSub.ref [p97, s97, c97, v97, b97]  -- fallback

-- ---- Sub-function 12: comparisons ----
def emitComparisons (k : K) (b : B) : IRBuilder Unit := do
  -- block61: comparison result (pos, sp, ctx, left, right, op)
  startBlock b.cmpResult
  let pos := b.cmpResult.param 0
  let sp := b.cmpResult.param 1
  let ctx := b.cmpResult.param 2
  let left := b.cmpResult.param 3
  let right := b.cmpResult.param 4
  let op := b.cmpResult.param 5
  let isLtOp ← icmp .eq op k.zero
  brif isLtOp b.cmpLt.ref [pos, sp, ctx, left, right]
              b.cmpCheckLe.ref [pos, sp, ctx, left, right, op]

  -- block102: lt
  startBlock b.cmpLt
  let ltPos := b.cmpLt.param 0
  let ltSp := b.cmpLt.param 1
  let ltCtx := b.cmpLt.param 2
  let ltLeft := b.cmpLt.param 3
  let ltRight := b.cmpLt.param 4
  let ltResult ← icmp .slt ltLeft ltRight
  brif ltResult b.returnFromSub.ref [ltPos, ltSp, ltCtx, k.one, k.one]
                b.returnFromSub.ref [ltPos, ltSp, ltCtx, k.zero, k.one]

  -- block103: check le
  startBlock b.cmpCheckLe
  let clePos := b.cmpCheckLe.param 0
  let cleSp := b.cmpCheckLe.param 1
  let cleCtx := b.cmpCheckLe.param 2
  let cleLeft := b.cmpCheckLe.param 3
  let cleRight := b.cmpCheckLe.param 4
  let cleOp := b.cmpCheckLe.param 5
  let isLeOp ← icmp .eq cleOp k.one
  brif isLeOp b.cmpLe.ref [clePos, cleSp, cleCtx, cleLeft, cleRight]
              b.cmpCheckGt.ref [clePos, cleSp, cleCtx, cleLeft, cleRight, cleOp]

  -- block104: le
  startBlock b.cmpLe
  let lePos := b.cmpLe.param 0
  let leSp := b.cmpLe.param 1
  let leCtx := b.cmpLe.param 2
  let leLeft := b.cmpLe.param 3
  let leRight := b.cmpLe.param 4
  let leResult ← icmp .sle leLeft leRight
  brif leResult b.returnFromSub.ref [lePos, leSp, leCtx, k.one, k.one]
                b.returnFromSub.ref [lePos, leSp, leCtx, k.zero, k.one]

  -- block105: check gt
  startBlock b.cmpCheckGt
  let cgtPos := b.cmpCheckGt.param 0
  let cgtSp := b.cmpCheckGt.param 1
  let cgtCtx := b.cmpCheckGt.param 2
  let cgtLeft := b.cmpCheckGt.param 3
  let cgtRight := b.cmpCheckGt.param 4
  let cgtOp := b.cmpCheckGt.param 5
  let c2 ← iconst64 2
  let isGtOp ← icmp .eq cgtOp c2
  brif isGtOp b.cmpGt.ref [cgtPos, cgtSp, cgtCtx, cgtLeft, cgtRight]
              b.cmpGe.ref [cgtPos, cgtSp, cgtCtx, cgtLeft, cgtRight]

  -- block106: gt
  startBlock b.cmpGt
  let gtPos := b.cmpGt.param 0
  let gtSp := b.cmpGt.param 1
  let gtCtx := b.cmpGt.param 2
  let gtLeft := b.cmpGt.param 3
  let gtRight := b.cmpGt.param 4
  let gtResult ← icmp .sgt gtLeft gtRight
  brif gtResult b.returnFromSub.ref [gtPos, gtSp, gtCtx, k.one, k.one]
                b.returnFromSub.ref [gtPos, gtSp, gtCtx, k.zero, k.one]

  -- block107: ge
  startBlock b.cmpGe
  let gePos := b.cmpGe.param 0
  let geSp := b.cmpGe.param 1
  let geCtx := b.cmpGe.param 2
  let geLeft := b.cmpGe.param 3
  let geRight := b.cmpGe.param 4
  let geResult ← icmp .sge geLeft geRight
  brif geResult b.returnFromSub.ref [gePos, geSp, geCtx, k.one, k.one]
                b.returnFromSub.ref [gePos, geSp, geCtx, k.zero, k.one]

-- ---- Sub-function 13: output (bool/int formatting) ----
set_option maxRecDepth 4096 in
def emitOutput (k : K) (b : B) : IRBuilder Unit := do
  -- block50: output (value, is_bool)
  startBlock b.output
  let value := b.output.param 0
  let isBool := b.output.param 1
  let isBoolNZ ← icmp .ne isBool k.zero
  brif isBoolNZ b.checkBoolVal.ref [value] b.itoaFindDiv.ref [value, k.zero]

  -- block98: check true/false
  startBlock b.checkBoolVal
  let bVal := b.checkBoolVal.param 0
  let isTrue ← icmp .ne bVal k.zero
  brif isTrue b.writeTrueBlk.ref [] b.writeFalseBlk.ref []

  -- block51: write "true\n"
  startBlock b.writeTrueBlk
  let tBufAddr ← iadd k.ptr k.outBufOff
  let ct ← iconst64 116   -- 't'
  istore8 ct tBufAddr
  let cr ← iconst64 114   -- 'r'
  let tAddr1 ← iadd tBufAddr k.one
  istore8 cr tAddr1
  let cu ← iconst64 117   -- 'u'
  let c2 ← iconst64 2
  let tAddr2 ← iadd tBufAddr c2
  istore8 cu tAddr2
  let ce ← iconst64 101   -- 'e'
  let c3 ← iconst64 3
  let tAddr3 ← iadd tBufAddr c3
  istore8 ce tAddr3
  let c4 ← iconst64 4
  let tAddr4 ← iadd tBufAddr c4
  istore8 k.newline tAddr4
  let c5 ← iconst64 5
  let tAddr5 ← iadd tBufAddr c5
  istore8 k.zero tAddr5  -- null terminator
  -- Write output file (size=0 → null-terminated write)
  let _ ← writeFile0 k.ptr k.fnFileWrite OUTPUT_PATH OUTPUT_BUF k.zero
  ret

  -- block52: write "false\n"
  startBlock b.writeFalseBlk
  let fBufAddr ← iadd k.ptr k.outBufOff
  let cf ← iconst64 102   -- 'f'
  istore8 cf fBufAddr
  let ca ← iconst64 97    -- 'a'
  let fAddr1 ← iadd fBufAddr k.one
  istore8 ca fAddr1
  let cl ← iconst64 108   -- 'l'
  let fc2 ← iconst64 2
  let fAddr2 ← iadd fBufAddr fc2
  istore8 cl fAddr2
  let cs ← iconst64 115   -- 's'
  let fc3 ← iconst64 3
  let fAddr3 ← iadd fBufAddr fc3
  istore8 cs fAddr3
  let fce ← iconst64 101  -- 'e'
  let fc4 ← iconst64 4
  let fAddr4 ← iadd fBufAddr fc4
  istore8 fce fAddr4
  let fc5 ← iconst64 5
  let fAddr5 ← iadd fBufAddr fc5
  istore8 k.newline fAddr5
  let fc6 ← iconst64 6
  let fAddr6 ← iadd fBufAddr fc6
  istore8 k.zero fAddr6
  -- Write output file (size=0 → null-terminated write)
  let _ ← writeFile0 k.ptr k.fnFileWrite OUTPUT_PATH OUTPUT_BUF k.zero
  ret

  -- block53: itoa — find highest power of 10 (value, out_pos)
  startBlock b.itoaFindDiv
  let iVal := b.itoaFindDiv.param 0
  let iOutPos := b.itoaFindDiv.param 1
  let iBufAddr ← iadd k.ptr k.outBufOff
  let initDiv ← iconst64 1
  jump b.itoaDivLoop.ref [iVal, iOutPos, initDiv, iBufAddr]

  -- block99: find divisor loop (value, out_pos, divisor, buf_addr)
  startBlock b.itoaDivLoop
  let dlVal := b.itoaDivLoop.param 0
  let dlOutPos := b.itoaDivLoop.param 1
  let dlDiv := b.itoaDivLoop.param 2
  let dlBufAddr := b.itoaDivLoop.param 3
  let ten ← iconst64 10
  let nextDiv ← imul dlDiv ten
  let tooBig ← icmp .ugt nextDiv dlVal
  brif tooBig b.itoaWriteDigit.ref [dlVal, dlOutPos, dlDiv, dlBufAddr]
              b.itoaDivLoop.ref [dlVal, dlOutPos, nextDiv, dlBufAddr]

  -- block54: itoa write digit (remainder, out_pos, divisor, buf_addr)
  startBlock b.itoaWriteDigit
  let wdRem := b.itoaWriteDigit.param 0
  let wdOutPos := b.itoaWriteDigit.param 1
  let wdDiv := b.itoaWriteDigit.param 2
  let wdBufAddr := b.itoaWriteDigit.param 3
  let digit ← udiv wdRem wdDiv
  let ascii0 ← iconst64 48
  let asciiDigit ← iadd digit ascii0
  let digitAddr ← iadd wdBufAddr wdOutPos
  istore8 asciiDigit digitAddr
  let digitTimesDiv ← imul digit wdDiv
  let newRem ← isub wdRem digitTimesDiv
  let wdTen ← iconst64 10
  let newDiv ← udiv wdDiv wdTen
  let nextOutPos ← iadd wdOutPos k.one
  let divDone ← icmp .eq newDiv k.zero
  brif divDone b.itoaDone.ref [nextOutPos, wdBufAddr]
               b.itoaWriteDigit.ref [newRem, nextOutPos, newDiv, wdBufAddr]

  -- block55: itoa done — write newline + null, then write file
  startBlock b.itoaDone
  let dOutPos := b.itoaDone.param 0
  let dBufAddr := b.itoaDone.param 1
  let nlAddr ← iadd dBufAddr dOutPos
  istore8 k.newline nlAddr
  let nextPos ← iadd dOutPos k.one
  let nullAddr ← iadd dBufAddr nextPos
  istore8 k.zero nullAddr
  -- Write output file (size=0 → null-terminated write, stops before null byte)
  let _ ← writeFile0 k.ptr k.fnFileWrite OUTPUT_PATH OUTPUT_BUF k.zero
  ret

-- ---------------------------------------------------------------------------
-- Main program builder
-- ---------------------------------------------------------------------------

set_option maxRecDepth 4096 in
def clifIrSource : String := buildProgram do
  -- Declare FFI
  let fnFileRead ← declareFileRead
  let fnFileWrite ← declareFileWrite
  let fnHtCreate ← declareFFI "ht_create" [.i64] (some .i32)
  let fnHtInsert ← declareFFI "ht_insert" [.i64, .i64, .i32, .i64, .i32] none
  let fnHtLookup ← declareFFI "ht_lookup" [.i64, .i64, .i32, .i64] (some .i32)

  -- Entry block (block0) — must be first
  let ptr ← entryBlock

  -- Forward-declare ALL blocks
  let skipSpHead    ← declareBlock [.i64, .i64, .i64, .i64]
  let skipSpCheck   ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let parseExprEntry ← declareBlock [.i64, .i64, .i64, .i64]
  let exprOpCheck    ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let exprCheckPlus  ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let exprCheckMinus ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let exprCheckLt    ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let exprCheckGt    ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let exprAdd        ← declareBlock [.i64, .i64, .i64, .i64]
  let exprSub        ← declareBlock [.i64, .i64, .i64, .i64]
  let exprLt         ← declareBlock [.i64, .i64, .i64, .i64]
  let exprGt         ← declareBlock [.i64, .i64, .i64, .i64]
  let parseTermEntry ← declareBlock [.i64, .i64, .i64]
  let termOpCheck    ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let termCheckMul   ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let termMul        ← declareBlock [.i64, .i64, .i64, .i64]
  let parseAtomDispatch ← declareBlock [.i64, .i64, .i64]
  let atomAfterSpaces   ← declareBlock [.i64, .i64, .i64, .i64]
  let checkParen        ← declareBlock [.i64, .i64, .i64, .i64]
  let checkLet          ← declareBlock [.i64, .i64, .i64, .i64]
  let verifyLet         ← declareBlock [.i64, .i64, .i64]
  let checkIf           ← declareBlock [.i64, .i64, .i64, .i64]
  let verifyIf          ← declareBlock [.i64, .i64, .i64]
  let checkTrue         ← declareBlock [.i64, .i64, .i64, .i64]
  let verifyTrue        ← declareBlock [.i64, .i64, .i64]
  let checkFalse        ← declareBlock [.i64, .i64, .i64, .i64]
  let verifyFalse       ← declareBlock [.i64, .i64, .i64]
  let trueLiteral   ← declareBlock [.i64, .i64, .i64]
  let falseLiteral  ← declareBlock [.i64, .i64, .i64]
  let parseNumber   ← declareBlock [.i64, .i64, .i64, .i64]
  let numAccum      ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let varIdent        ← declareBlock [.i64, .i64, .i64, .i64]
  let storeIdentChar  ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let varLookup       ← declareBlock [.i64, .i64, .i64, .i64]
  let parenOrLambda    ← declareBlock [.i64, .i64, .i64]
  let parenExpr        ← declareBlock [.i64, .i64, .i64]
  let letBinding       ← declareBlock [.i64, .i64, .i64]
  let letReadName      ← declareBlock [.i64, .i64, .i64, .i64]
  let letStoreNameChar ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let letSkipAssign    ← declareBlock [.i64, .i64, .i64, .i64]
  let letStoreAndContinue ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let ifEntry     ← declareBlock [.i64, .i64, .i64]
  let ifSkipThen  ← declareBlock [.i64, .i64, .i64, .i64]
  let ifSkipElse  ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let ifSelect    ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let lambdaEntry          ← declareBlock [.i64, .i64, .i64]
  let lambdaReadParam      ← declareBlock [.i64, .i64, .i64, .i64]
  let lambdaStoreParamChar ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let lambdaSkipArrow      ← declareBlock [.i64, .i64, .i64, .i64]
  let lambdaScanBody       ← declareBlock [.i64, .i64, .i64, .i64]
  let lambdaCheckOpen      ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let lambdaFoundClose     ← declareBlock [.i64, .i64, .i64, .i64]
  let lambdaDecrDepth      ← declareBlock [.i64, .i64, .i64, .i64]
  let lambdaParseArg       ← declareBlock [.i64, .i64, .i64]
  let lambdaBindAndEval    ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let returnFromSub  ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let checkExprOp    ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let exprOpReturn   ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let exprAddOrSub   ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let checkTermOp    ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let termOpReturn   ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let termMulResult  ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let checkLetBind   ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let checkLetBody   ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let checkIfCond    ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let checkIfThen    ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let checkIfElse    ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let rdCheckParen   ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let parenClose     ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let checkLambdaArg  ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let checkLambdaBody ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let checkCmpRhs     ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let cmpResult  ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let cmpLt      ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let cmpCheckLe ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let cmpLe      ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let cmpCheckGt ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let cmpGt      ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let cmpGe      ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let output       ← declareBlock [.i64, .i64]
  let checkBoolVal ← declareBlock [.i64]
  let writeTrueBlk ← declareBlock []
  let writeFalseBlk ← declareBlock []
  let itoaFindDiv  ← declareBlock [.i64, .i64]
  let itoaDivLoop  ← declareBlock [.i64, .i64, .i64, .i64]
  let itoaWriteDigit ← declareBlock [.i64, .i64, .i64, .i64]
  let itoaDone     ← declareBlock [.i64, .i64]

  -- Build shared constants
  let zero ← iconst64 0
  let one ← iconst64 1
  let srcBase ← iconst64 SOURCE_BUF
  let identOff ← iconst64 IDENT_BUF
  let htValOff ← iconst64 HT_VAL_BUF
  let outBufOff ← iconst64 OUTPUT_BUF
  let stackOff ← iconst64 STACK_BASE
  let frameSize ← iconst64 24
  let space ← iconst64 32
  let newline ← iconst64 10

  let k : K := {
    ptr, zero, one, srcBase, identOff, htValOff, outBufOff, stackOff,
    frameSize, space, newline,
    fnHtCreate, fnHtInsert, fnHtLookup, fnFileRead, fnFileWrite
  }

  let b : B := {
    skipSpHead, skipSpCheck, parseExprEntry, exprOpCheck,
    exprCheckPlus, exprCheckMinus, exprCheckLt, exprCheckGt,
    exprAdd, exprSub, exprLt, exprGt,
    parseTermEntry, termOpCheck, termCheckMul, termMul,
    parseAtomDispatch, atomAfterSpaces,
    checkParen, checkLet, verifyLet, checkIf, verifyIf,
    checkTrue, verifyTrue, checkFalse, verifyFalse,
    trueLiteral, falseLiteral, parseNumber, numAccum,
    varIdent, storeIdentChar, varLookup,
    parenOrLambda, parenExpr,
    letBinding, letReadName, letStoreNameChar, letSkipAssign, letStoreAndContinue,
    ifEntry, ifSkipThen, ifSkipElse, ifSelect,
    lambdaEntry, lambdaReadParam, lambdaStoreParamChar,
    lambdaSkipArrow, lambdaScanBody,
    lambdaCheckOpen, lambdaFoundClose, lambdaDecrDepth,
    lambdaParseArg, lambdaBindAndEval,
    returnFromSub, checkExprOp, exprOpReturn, exprAddOrSub,
    checkTermOp, termOpReturn, termMulResult,
    checkLetBind, checkLetBody, checkIfCond, checkIfThen, checkIfElse,
    rdCheckParen, parenClose,
    checkLambdaArg, checkLambdaBody, checkCmpRhs,
    cmpResult, cmpLt, cmpCheckLe, cmpLe, cmpCheckGt, cmpGt, cmpGe,
    output, checkBoolVal, writeTrueBlk, writeFalseBlk,
    itoaFindDiv, itoaDivLoop, itoaWriteDigit, itoaDone
  }

  -- Emit setup (file read, HT create, initial stack frame)
  let _ ← emitSetup k b

  -- Emit all sub-functions
  emitSkipSpaces k b
  emitParseExpr k b
  emitExprAddSub k b
  emitExprCmp k b
  emitParseTerm k b
  emitAtomDispatch k b
  emitLiteralsAndVariable k b
  emitParenAndLet k b
  emitLetStoreAndIf k b
  emitLambda k b
  emitReturnDispatch k b
  emitComparisons k b
  emitOutput k b

-- ---------------------------------------------------------------------------
-- Payload construction
-- ---------------------------------------------------------------------------

def buildPayload : List UInt8 :=
  let htCtxPtr    := zeros 8                               -- 0x0000
  let flagFile    := zeros 8                               -- 0x0008
  let flagCl      := zeros 8                               -- 0x0010
  let reserved    := zeros 8                               -- 0x0018
  let outputPath  := padTo (stringToBytes "output.txt") 64 -- 0x0020
  let inputPath   := zeros 256                             -- 0x0060
  let sourceBuf   := zeros SOURCE_BUF_SZ                   -- 0x0160
  let trueStr     := padTo (stringToBytes "true") 8        -- 0x1160
  let falseStr    := padTo (stringToBytes "false") 8       -- 0x1168
  let identBuf    := zeros IDENT_BUF_SZ                    -- 0x1170
  let htValBuf    := zeros 8                               -- 0x11B0
  let outputBuf   := zeros OUTPUT_BUF_SZ                   -- 0x11B8
  let stackRegion := zeros STACK_SZ                        -- 0x11F8
  htCtxPtr ++ flagFile ++ flagCl ++ reserved ++
    outputPath ++ inputPath ++ sourceBuf ++
    trueStr ++ falseStr ++ identBuf ++ htValBuf ++ outputBuf ++
    stackRegion

def buildConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := buildPayload.length,
  context_offset := 0,
  initial_memory := buildPayload
}

def buildAlgorithm : Algorithm := {
  actions := [IR.clifCallAction],
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

end LeanEval

def main : IO Unit := do
  let json := toJsonPair LeanEval.buildConfig LeanEval.buildAlgorithm
  IO.println (Json.compress json)
