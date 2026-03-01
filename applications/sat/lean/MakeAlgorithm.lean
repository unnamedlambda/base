import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib

namespace Algorithm

-- ---------------------------------------------------------------------------
-- DPLL SAT solver for DIMACS CNF
-- ---------------------------------------------------------------------------

def maxVars : Nat := 10000
def maxClauses : Nat := 50000
def maxClauseWords : Nat := 500000
def maxCnfFileSize : Nat := 4 * 1024 * 1024

def reserved_off : Nat := 0x00
def numVars_off : Nat := 0x40
def numClauses_off : Nat := 0x48
def clauseCount_off : Nat := 0x50
def resultFlag_off : Nat := 0x58
def outLen_off : Nat := 0x60
def inputFilename_off : Nat := 0x100
def outputFilename_off : Nat := 0x200
def outputStr_off : Nat := 0x300
def cnf_off : Nat := 0x1000
def db_off : Nat := cnf_off + maxCnfFileSize
def assign_off : Nat := db_off + maxClauseWords * 4
def trail_off : Nat := assign_off + maxVars + 16
def out_off : Nat := trail_off + maxVars * 4 + 16
def clauseIndex_off : Nat := out_off
def decStack_off : Nat := clauseIndex_off + maxClauses * 8
def solver_scratch_off : Nat := decStack_off + maxVars * 8
def totalMemory : Nat := solver_scratch_off + 0x10000

-- ---------------------------------------------------------------------------
-- CLIF IR via DSL — split into sub-functions
-- ---------------------------------------------------------------------------

open AlgorithmLib.IR

-- Shared constants
structure K where
  ptr : Val
  c0 : Val
  c1 : Val
  c4 : Val
  c8 : Val
  c10 : Val
  c32 : Val
  c45 : Val
  c48 : Val
  c58 : Val
  assignBase : Val
  cnfOffV : Val
  dbOffV : Val
  clIdxOffV : Val
  trailOffV : Val
  outOffV : Val
  decStackV : Val
  numVarsAddr : Val
  clauseCountAddr : Val
  resultFlagAddr : Val
  bytesRead : Val
  fnWrite : FnRef

-- Phase 1: Zero assignment array
def emitZeroAssign (k : K) (doneBlk : DeclaredBlock) : IRBuilder Unit := do
  let hdr ← declareBlock [.i64]
  let body ← declareBlock []
  let maxV ← iconst64 maxVars
  jump hdr.ref [k.c0]
  startBlock hdr
  let i := hdr.param 0
  let done ← icmp .uge i maxV
  brif done doneBlk.ref [] body.ref []
  startBlock body
  let addr ← iadd k.assignBase i
  istore8 k.c0 addr
  let next ← iadd i k.c1
  jump hdr.ref [next]

-- Phase 2a: Skip to end of line
def emitSkipLine (k : K) (skipLineBlk newLineBlk doneBlk : DeclaredBlock) : IRBuilder Unit := do
  let body ← declareBlock []
  startBlock skipLineBlk
  let pos := skipLineBlk.param 0
  let dbp := skipLineBlk.param 1
  let eof ← icmp .uge pos k.bytesRead
  brif eof doneBlk.ref [] body.ref []
  startBlock body
  let relOff ← iadd k.cnfOffV pos
  let addr ← iadd k.ptr relOff
  let byte ← uload8_64 addr
  let isNl ← icmp .eq byte k.c10
  let pos1 ← iadd pos k.c1
  brif isNl newLineBlk.ref [pos1, dbp] skipLineBlk.ref [pos1, dbp]

-- Phase 2b: New line dispatcher
def emitNewLine (k : K) (newLineBlk skipLineBlk headerBlk clauseBlk doneBlk : DeclaredBlock) : IRBuilder Unit := do
  let checkByte ← declareBlock []
  let chkP ← declareBlock []
  let chkNl ← declareBlock []
  let chkCr ← declareBlock []
  let chkSp ← declareBlock []
  let chkTab ← declareBlock []

  startBlock newLineBlk
  let pos := newLineBlk.param 0
  let dbp := newLineBlk.param 1
  let eof ← icmp .uge pos k.bytesRead
  brif eof doneBlk.ref [] checkByte.ref []

  startBlock checkByte
  let relOff ← iadd k.cnfOffV pos
  let addr ← iadd k.ptr relOff
  let byte ← uload8_64 addr
  let c99 ← iconst64 99
  let isC ← icmp .eq byte c99
  brif isC skipLineBlk.ref [pos, dbp] chkP.ref []

  startBlock chkP
  let c112 ← iconst64 112
  let isP ← icmp .eq byte c112
  brif isP headerBlk.ref [pos, dbp] chkNl.ref []

  startBlock chkNl
  let isNl ← icmp .eq byte k.c10
  let pos1 ← iadd pos k.c1
  brif isNl newLineBlk.ref [pos1, dbp] chkCr.ref []

  startBlock chkCr
  let c13 ← iconst64 13
  let isCr ← icmp .eq byte c13
  brif isCr newLineBlk.ref [pos1, dbp] chkSp.ref []

  startBlock chkSp
  let isSp ← icmp .eq byte k.c32
  brif isSp newLineBlk.ref [pos1, dbp] chkTab.ref []

  startBlock chkTab
  let c9 ← iconst64 9
  let isTab ← icmp .eq byte c9
  brif isTab newLineBlk.ref [pos1, dbp] clauseBlk.ref [pos, dbp]

-- Phase 2c: Parse 'p cnf' header
def emitHeader (k : K) (headerBlk skipLineBlk doneBlk : DeclaredBlock) : IRBuilder Unit := do
  -- Skip non-digits to find nvars
  let skipToNvars ← declareBlock [.i64, .i64]  -- pos, db_ptr
  let skipNvBody ← declareBlock []
  let skipNvAdvance ← declareBlock []
  let parseNvars ← declareBlock [.i64, .i64, .i64]  -- pos, db_ptr, accum
  let parseNvBody ← declareBlock []
  let parseNvDigit ← declareBlock []
  let nvarsDone ← declareBlock [.i64, .i64, .i64]
  -- Skip non-digits to find nclauses
  let skipToNcl ← declareBlock [.i64, .i64]
  let skipNclBody ← declareBlock []
  let skipNclAdvance ← declareBlock []
  let parseNcl ← declareBlock [.i64, .i64, .i64]
  let parseNclBody ← declareBlock []
  let parseNclDigit ← declareBlock []
  let nclDone ← declareBlock [.i64, .i64, .i64]

  startBlock headerBlk
  let hPos := headerBlk.param 0
  let hDbp := headerBlk.param 1
  let hPos1 ← iadd hPos k.c1  -- skip 'p'
  jump skipToNvars.ref [hPos1, hDbp]

  -- Skip non-digits for nvars
  startBlock skipToNvars
  let snPos := skipToNvars.param 0
  let snDbp := skipToNvars.param 1
  let snEof ← icmp .uge snPos k.bytesRead
  brif snEof doneBlk.ref [] skipNvBody.ref []
  startBlock skipNvBody
  let snRelOff ← iadd k.cnfOffV snPos
  let snAddr ← iadd k.ptr snRelOff
  let snByte ← uload8_64 snAddr
  let snGe0 ← icmp .uge snByte k.c48
  let snLt10 ← icmp .ult snByte k.c58
  let snIsDigit ← band snGe0 snLt10
  brif snIsDigit parseNvars.ref [snPos, snDbp, k.c0] skipNvAdvance.ref []
  startBlock skipNvAdvance
  let snPos1 ← iadd snPos k.c1
  jump skipToNvars.ref [snPos1, snDbp]

  -- Parse nvars digits
  startBlock parseNvars
  let pnPos := parseNvars.param 0
  let pnDbp := parseNvars.param 1
  let pnAcc := parseNvars.param 2
  let pnEof ← icmp .uge pnPos k.bytesRead
  brif pnEof nvarsDone.ref [pnPos, pnDbp, pnAcc] parseNvBody.ref []
  startBlock parseNvBody
  let pnRelOff ← iadd k.cnfOffV pnPos
  let pnAddr ← iadd k.ptr pnRelOff
  let pnByte ← uload8_64 pnAddr
  let pnGe0 ← icmp .uge pnByte k.c48
  let pnLt10 ← icmp .ult pnByte k.c58
  let pnIsDigit ← band pnGe0 pnLt10
  brif pnIsDigit parseNvDigit.ref [] nvarsDone.ref [pnPos, pnDbp, pnAcc]
  startBlock parseNvDigit
  let pnAcc10 ← imul pnAcc k.c10
  let pnDv ← isub pnByte k.c48
  let pnAccNew ← iadd pnAcc10 pnDv
  let pnPos1 ← iadd pnPos k.c1
  jump parseNvars.ref [pnPos1, pnDbp, pnAccNew]

  -- Store nvars, skip to nclauses
  startBlock nvarsDone
  let ndPos := nvarsDone.param 0
  let ndDbp := nvarsDone.param 1
  let ndVal := nvarsDone.param 2
  store ndVal k.numVarsAddr
  jump skipToNcl.ref [ndPos, ndDbp]

  -- Skip non-digits for nclauses
  startBlock skipToNcl
  let scPos := skipToNcl.param 0
  let scDbp := skipToNcl.param 1
  let scEof ← icmp .uge scPos k.bytesRead
  brif scEof doneBlk.ref [] skipNclBody.ref []
  startBlock skipNclBody
  let scRelOff ← iadd k.cnfOffV scPos
  let scAddr ← iadd k.ptr scRelOff
  let scByte ← uload8_64 scAddr
  let scGe0 ← icmp .uge scByte k.c48
  let scLt10 ← icmp .ult scByte k.c58
  let scIsDigit ← band scGe0 scLt10
  brif scIsDigit parseNcl.ref [scPos, scDbp, k.c0] skipNclAdvance.ref []
  startBlock skipNclAdvance
  let scPos1 ← iadd scPos k.c1
  jump skipToNcl.ref [scPos1, scDbp]

  -- Parse nclauses digits
  startBlock parseNcl
  let ncPos := parseNcl.param 0
  let ncDbp := parseNcl.param 1
  let ncAcc := parseNcl.param 2
  let ncEof ← icmp .uge ncPos k.bytesRead
  brif ncEof nclDone.ref [ncPos, ncDbp, ncAcc] parseNclBody.ref []
  startBlock parseNclBody
  let ncRelOff ← iadd k.cnfOffV ncPos
  let ncAddr ← iadd k.ptr ncRelOff
  let ncByte ← uload8_64 ncAddr
  let ncGe0 ← icmp .uge ncByte k.c48
  let ncLt10 ← icmp .ult ncByte k.c58
  let ncIsDigit ← band ncGe0 ncLt10
  brif ncIsDigit parseNclDigit.ref [] nclDone.ref [ncPos, ncDbp, ncAcc]
  startBlock parseNclDigit
  let ncAcc10 ← imul ncAcc k.c10
  let ncDv ← isub ncByte k.c48
  let ncAccNew ← iadd ncAcc10 ncDv
  let ncPos1 ← iadd ncPos k.c1
  jump parseNcl.ref [ncPos1, ncDbp, ncAccNew]

  -- Store nclauses, skip to end of header line
  startBlock nclDone
  let ncdPos := nclDone.param 0
  let ncdDbp := nclDone.param 1
  let ncdVal := nclDone.param 2
  let numClAddr ← absAddr k.ptr numClauses_off
  store ncdVal numClAddr
  jump skipLineBlk.ref [ncdPos, ncdDbp]

-- Phase 2d: Clause parser
def emitClauseParser (k : K) (clauseBlk : DeclaredBlock)
    (newLineBlk : DeclaredBlock) : IRBuilder Unit := do
  -- Blocks
  let wsHdr ← declareBlock [.i64, .i64, .i64, .i64]  -- pos, db_start, lit_ptr, lit_count
  let wsBody ← declareBlock []
  let endClause ← declareBlock [.i64, .i64, .i64, .i64]
  let chkCr ← declareBlock []
  let chkSp ← declareBlock []
  let chkTab ← declareBlock []
  let parseIntBlk ← declareBlock [.i64, .i64, .i64, .i64]
  let digitLoop ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]  -- pos,db,lit,cnt,accum,is_neg
  let digitBody ← declareBlock []
  let digitAccum ← declareBlock []
  let intDone ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let storeLit ← declareBlock []
  let finalizeClause ← declareBlock []

  startBlock clauseBlk
  let cPos := clauseBlk.param 0
  let cDbp := clauseBlk.param 1
  let litPtr ← iadd cDbp k.c4
  jump wsHdr.ref [cPos, cDbp, litPtr, k.c0]

  -- Skip whitespace
  startBlock wsHdr
  let wPos := wsHdr.param 0
  let wDb := wsHdr.param 1
  let wLit := wsHdr.param 2
  let wCnt := wsHdr.param 3
  let eof ← icmp .uge wPos k.bytesRead
  brif eof endClause.ref [wPos, wDb, wLit, wCnt] wsBody.ref []

  startBlock wsBody
  let relOff ← iadd k.cnfOffV wPos
  let addr ← iadd k.ptr relOff
  let byte ← uload8_64 addr
  let pos1 ← iadd wPos k.c1
  let isNl ← icmp .eq byte k.c10
  brif isNl endClause.ref [pos1, wDb, wLit, wCnt] chkCr.ref []

  startBlock chkCr
  let c13 ← iconst64 13
  let isCr ← icmp .eq byte c13
  brif isCr endClause.ref [pos1, wDb, wLit, wCnt] chkSp.ref []

  startBlock chkSp
  let isSp ← icmp .eq byte k.c32
  brif isSp wsHdr.ref [pos1, wDb, wLit, wCnt] chkTab.ref []

  startBlock chkTab
  let c9 ← iconst64 9
  let isTab ← icmp .eq byte c9
  brif isTab wsHdr.ref [pos1, wDb, wLit, wCnt] parseIntBlk.ref [wPos, wDb, wLit, wCnt]

  -- Parse signed integer
  startBlock parseIntBlk
  let piPos := parseIntBlk.param 0
  let piDb := parseIntBlk.param 1
  let piLit := parseIntBlk.param 2
  let piCnt := parseIntBlk.param 3
  let piRelOff ← iadd k.cnfOffV piPos
  let piAddr ← iadd k.ptr piRelOff
  let piByte ← uload8_64 piAddr
  let isMinus ← icmp .eq piByte k.c45
  let posAfterSign ← iadd piPos k.c1
  brif isMinus digitLoop.ref [posAfterSign, piDb, piLit, piCnt, k.c0, k.c1]
              digitLoop.ref [piPos, piDb, piLit, piCnt, k.c0, k.c0]

  -- Digit loop
  startBlock digitLoop
  let dPos := digitLoop.param 0
  let dDb := digitLoop.param 1
  let dLit := digitLoop.param 2
  let dCnt := digitLoop.param 3
  let dAcc := digitLoop.param 4
  let dNeg := digitLoop.param 5
  let dEof ← icmp .uge dPos k.bytesRead
  brif dEof intDone.ref [dPos, dDb, dLit, dCnt, dAcc, dNeg] digitBody.ref []

  startBlock digitBody
  let dRelOff ← iadd k.cnfOffV dPos
  let dAddr ← iadd k.ptr dRelOff
  let dByte ← uload8_64 dAddr
  let ge0 ← icmp .uge dByte k.c48
  let lt10 ← icmp .ult dByte k.c58
  let isDigit ← band ge0 lt10
  brif isDigit digitAccum.ref [] intDone.ref [dPos, dDb, dLit, dCnt, dAcc, dNeg]

  startBlock digitAccum
  let acc10 ← imul dAcc k.c10
  let dv ← isub dByte k.c48
  let accNew ← iadd acc10 dv
  let dPos1 ← iadd dPos k.c1
  jump digitLoop.ref [dPos1, dDb, dLit, dCnt, accNew, dNeg]

  -- Integer parsed
  startBlock intDone
  let idPos := intDone.param 0
  let idDb := intDone.param 1
  let idLit := intDone.param 2
  let idCnt := intDone.param 3
  let idAcc := intDone.param 4
  let idNeg := intDone.param 5
  let isZero ← icmp .eq idAcc k.c0
  brif isZero endClause.ref [idPos, idDb, idLit, idCnt] storeLit.ref []

  -- Store literal
  startBlock storeLit
  let negAcc ← ineg idAcc
  let isNegFlag ← icmp .eq idNeg k.c1
  let litVal ← select' isNegFlag negAcc idAcc
  let litI32 ← ireduce32 litVal
  let litRelAddr ← iadd k.dbOffV idLit
  let litAbsAddr ← iadd k.ptr litRelAddr
  store litI32 litAbsAddr
  let litNext ← iadd idLit k.c4
  let cntNext ← iadd idCnt k.c1
  jump wsHdr.ref [idPos, idDb, litNext, cntNext]

  -- End clause
  startBlock endClause
  let ecPos := endClause.param 0
  let ecDb := endClause.param 1
  let ecLit := endClause.param 2
  let ecCnt := endClause.param 3
  let isEmpty ← icmp .eq ecCnt k.c0
  brif isEmpty newLineBlk.ref [ecPos, ecDb] finalizeClause.ref []

  startBlock finalizeClause
  let lenI32 ← ireduce32 ecCnt
  let clRelAddr ← iadd k.dbOffV ecDb
  let clAbsAddr ← iadd k.ptr clRelAddr
  store lenI32 clAbsAddr
  let cc ← load64 k.clauseCountAddr
  let ccTimes8 ← imul cc k.c8
  let idxOff ← iadd k.clIdxOffV ccTimes8
  let idxAddr ← iadd k.ptr idxOff
  store ecDb idxAddr
  let cc1 ← iadd cc k.c1
  store cc1 k.clauseCountAddr
  jump newLineBlk.ref [ecPos, ecLit]

-- Phase 3a: Unit propagation scan
def emitUnitPropScan (k : K) (scanHdr : DeclaredBlock)
    (decideBlk conflictBlk upEntry : DeclaredBlock) : IRBuilder Unit := do
  let scanDone ← declareBlock [.i64, .i64, .i64]
  let evalClause ← declareBlock [.i64, .i64, .i64, .i64]
  let litScan ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let litBody ← declareBlock []
  let litDone ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let unsetBlk ← declareBlock []
  let setBlk ← declareBlock []
  let chkConflict ← declareBlock []
  let chkUnit ← declareBlock []
  let assignUnit ← declareBlock []
  let skipClause ← declareBlock []

  startBlock scanHdr
  let sTd := scanHdr.param 0
  let sDd := scanHdr.param 1
  let sIdx := scanHdr.param 2
  let sFu := scanHdr.param 3
  let cc ← load64 k.clauseCountAddr
  let done ← icmp .uge sIdx cc
  brif done scanDone.ref [sTd, sDd, sFu] evalClause.ref [sTd, sDd, sIdx, sFu]

  -- Evaluate clause
  startBlock evalClause
  let eTd := evalClause.param 0
  let eDd := evalClause.param 1
  let eIdx := evalClause.param 2
  let eFu := evalClause.param 3
  let idxOff ← imul eIdx k.c8
  let idxRelAddr ← iadd k.clIdxOffV idxOff
  let idxAbsAddr ← iadd k.ptr idxRelAddr
  let dbPtrOff ← load64 idxAbsAddr
  let clRelAddr ← iadd k.dbOffV dbPtrOff
  let clAbsAddr ← iadd k.ptr clRelAddr
  let clLen32 ← load32 clAbsAddr
  let clLen ← uextend64 clLen32
  let firstLit ← iadd dbPtrOff k.c4
  jump litScan.ref [eTd, eDd, eIdx, eFu, firstLit, clLen, k.c0, k.c0, k.c0, k.c0]

  -- Scan literal
  startBlock litScan
  let lTd := litScan.param 0
  let lDd := litScan.param 1
  let lIdx := litScan.param 2
  let lFu := litScan.param 3
  let lOff := litScan.param 4
  let lRem := litScan.param 5
  let lCF := litScan.param 6
  let lCU := litScan.param 7
  let lLast := litScan.param 8
  let lSat := litScan.param 9
  let remZero ← icmp .eq lRem k.c0
  brif remZero litDone.ref [lTd, lDd, lIdx, lFu, lCF, lCU, lLast, lSat] litBody.ref []

  startBlock litBody
  let litRelAddr ← iadd k.dbOffV lOff
  let litAbsAddr ← iadd k.ptr litRelAddr
  let litI32 ← load32 litAbsAddr
  let litI64 ← sextend64 litI32
  let isNeg ← icmp .slt litI64 k.c0
  let negLit ← ineg litI64
  let absLit ← select' isNeg negLit litI64
  let varIdx ← isub absLit k.c1
  let assignAddr ← iadd k.assignBase varIdx
  let assignVal ← sload8_64 assignAddr
  let isUnset ← icmp .eq assignVal k.c0
  brif isUnset unsetBlk.ref [] setBlk.ref []

  startBlock unsetBlk
  let cuNew ← iadd lCU k.c1
  let offNext ← iadd lOff k.c4
  let remNext ← isub lRem k.c1
  jump litScan.ref [lTd, lDd, lIdx, lFu, offNext, remNext, lCF, cuNew, litI64, lSat]

  startBlock setBlk
  let posOne ← iconst64 1
  let negOne ← iconst64 (-1)
  let litPos ← icmp .sgt litI64 k.c0
  let sign ← select' litPos posOne negOne
  let isSatisfied ← icmp .eq sign assignVal
  let newSat ← select' isSatisfied k.c1 lSat
  let litFalse ← select' isSatisfied k.c0 k.c1
  let newCF ← iadd lCF litFalse
  let offNext2 ← iadd lOff k.c4
  let remNext2 ← isub lRem k.c1
  jump litScan.ref [lTd, lDd, lIdx, lFu, offNext2, remNext2, newCF, lCU, lLast, newSat]

  -- Clause evaluation
  startBlock litDone
  let dTd := litDone.param 0
  let dDd := litDone.param 1
  let dIdx := litDone.param 2
  let dFu := litDone.param 3
  let dCF := litDone.param 4
  let dCU := litDone.param 5
  let dLast := litDone.param 6
  let dSat := litDone.param 7
  let nextIdx ← iadd dIdx k.c1
  let clauseSat ← icmp .eq dSat k.c1
  brif clauseSat scanHdr.ref [dTd, dDd, nextIdx, dFu] chkConflict.ref []

  startBlock chkConflict
  let noUnset ← icmp .eq dCU k.c0
  brif noUnset conflictBlk.ref [dTd, dDd] chkUnit.ref []

  startBlock chkUnit
  let isUnit ← icmp .eq dCU k.c1
  brif isUnit assignUnit.ref [] skipClause.ref []

  startBlock assignUnit
  let uNeg ← icmp .slt dLast k.c0
  let uNegLit ← ineg dLast
  let uAbs ← select' uNeg uNegLit dLast
  let uVarIdx ← isub uAbs k.c1
  let uPosOne ← iconst64 1
  let uNegOne ← iconst64 (-1)
  let uIsPos ← icmp .sgt dLast k.c0
  let uAssign ← select' uIsPos uPosOne uNegOne
  let uAddr ← iadd k.assignBase uVarIdx
  istore8 uAssign uAddr
  let tOff ← imul dTd k.c4
  let tRelAddr ← iadd k.trailOffV tOff
  let tAbsAddr ← iadd k.ptr tRelAddr
  let uLitI32 ← ireduce32 dLast
  store uLitI32 tAbsAddr
  let newTd ← iadd dTd k.c1
  jump scanHdr.ref [newTd, dDd, nextIdx, k.c1]

  startBlock skipClause
  jump scanHdr.ref [dTd, dDd, nextIdx, dFu]

  -- Scan complete
  startBlock scanDone
  let sdTd := scanDone.param 0
  let sdDd := scanDone.param 1
  let sdFu := scanDone.param 2
  let foundAny ← icmp .eq sdFu k.c1
  brif foundAny upEntry.ref [sdTd, sdDd] decideBlk.ref [sdTd, sdDd]

-- Phase 3b: Decide
def emitDecide (k : K) (decideBlk satBlk upEntry : DeclaredBlock) : IRBuilder Unit := do
  let searchHdr ← declareBlock [.i64, .i64, .i64, .i64]
  let checkVar ← declareBlock []
  let tryTrue ← declareBlock []
  let nextVar ← declareBlock []

  startBlock decideBlk
  let dTd := decideBlk.param 0
  let dDd := decideBlk.param 1
  let nv ← load64 k.numVarsAddr
  jump searchHdr.ref [dTd, dDd, k.c0, nv]

  startBlock searchHdr
  let sTd := searchHdr.param 0
  let sDd := searchHdr.param 1
  let sVi := searchHdr.param 2
  let sNv := searchHdr.param 3
  let allDone ← icmp .uge sVi sNv
  brif allDone satBlk.ref [sTd] checkVar.ref []

  startBlock checkVar
  let aAddr ← iadd k.assignBase sVi
  let aVal ← sload8_64 aAddr
  let isUnset ← icmp .eq aVal k.c0
  brif isUnset tryTrue.ref [] nextVar.ref []

  startBlock tryTrue
  let dsOff ← imul sDd k.c8
  let dsRelAddr ← iadd k.decStackV dsOff
  let dsAbsAddr ← iadd k.ptr dsRelAddr
  store sTd dsAbsAddr
  let aAddr2 ← iadd k.assignBase sVi
  istore8 k.c1 aAddr2
  let lit ← iadd sVi k.c1
  let tOff ← imul sTd k.c4
  let tRelAddr ← iadd k.trailOffV tOff
  let tAbsAddr ← iadd k.ptr tRelAddr
  let litI32 ← ireduce32 lit
  store litI32 tAbsAddr
  let newTd ← iadd sTd k.c1
  let newDd ← iadd sDd k.c1
  jump upEntry.ref [newTd, newDd]

  startBlock nextVar
  let vi1 ← iadd sVi k.c1
  jump searchHdr.ref [sTd, sDd, vi1, sNv]

-- Phase 3c: Conflict/backtrack
def emitConflict (k : K) (conflictBlk unsatBlk upEntry : DeclaredBlock) : IRBuilder Unit := do
  let backtrack ← declareBlock []
  let undoHdr ← declareBlock [.i64, .i64, .i64]
  let undoBody ← declareBlock []
  let undoDone ← declareBlock [.i64, .i64]
  let tryFalseBlk ← declareBlock [.i64, .i64, .i64]

  startBlock conflictBlk
  let cTd := conflictBlk.param 0
  let cDd := conflictBlk.param 1
  let noDecisions ← icmp .eq cDd k.c0
  brif noDecisions unsatBlk.ref [] backtrack.ref []

  startBlock backtrack
  let ddM1 ← isub cDd k.c1
  let dsOff ← imul ddM1 k.c8
  let dsRelAddr ← iadd k.decStackV dsOff
  let dsAbsAddr ← iadd k.ptr dsRelAddr
  let savedTd ← load64 dsAbsAddr
  let undoStart ← isub cTd k.c1
  jump undoHdr.ref [undoStart, savedTd, ddM1]

  startBlock undoHdr
  let uI := undoHdr.param 0
  let uSaved := undoHdr.param 1
  let uDd := undoHdr.param 2
  let undoComplete ← icmp .slt uI uSaved
  brif undoComplete undoDone.ref [uSaved, uDd] undoBody.ref []

  startBlock undoBody
  let tOff ← imul uI k.c4
  let tRelAddr ← iadd k.trailOffV tOff
  let tAbsAddr ← iadd k.ptr tRelAddr
  let litI32 ← load32 tAbsAddr
  let litI64 ← sextend64 litI32
  let isNeg ← icmp .slt litI64 k.c0
  let negLit ← ineg litI64
  let absLit ← select' isNeg negLit litI64
  let varIdx ← isub absLit k.c1
  let aAddr ← iadd k.assignBase varIdx
  istore8 k.c0 aAddr
  let uIM1 ← isub uI k.c1
  jump undoHdr.ref [uIM1, uSaved, uDd]

  startBlock undoDone
  let udSaved := undoDone.param 0
  let udDd := undoDone.param 1
  let tOff2 ← imul udSaved k.c4
  let tRelAddr2 ← iadd k.trailOffV tOff2
  let tAbsAddr2 ← iadd k.ptr tRelAddr2
  let decLitI32 ← load32 tAbsAddr2
  let decLitI64 ← sextend64 decLitI32
  let isNeg2 ← icmp .slt decLitI64 k.c0
  let negDecLit ← ineg decLitI64
  let absDecLit ← select' isNeg2 negDecLit decLitI64
  let decVarIdx ← isub absDecLit k.c1
  let aAddr2 ← iadd k.assignBase decVarIdx
  istore8 k.c0 aAddr2
  let wasPositive ← icmp .sgt decLitI64 k.c0
  brif wasPositive tryFalseBlk.ref [udSaved, udDd, decVarIdx] conflictBlk.ref [udSaved, udDd]

  startBlock tryFalseBlk
  let tfSaved := tryFalseBlk.param 0
  let tfDd := tryFalseBlk.param 1
  let tfVar := tryFalseBlk.param 2
  let negOneV ← iconst64 (-1)
  let tfAddr ← iadd k.assignBase tfVar
  istore8 negOneV tfAddr
  let posLit ← iadd tfVar k.c1
  let negLitV ← ineg posLit
  let tOff3 ← imul tfSaved k.c4
  let tRelAddr3 ← iadd k.trailOffV tOff3
  let tAbsAddr3 ← iadd k.ptr tRelAddr3
  let negLitI32 ← ireduce32 negLitV
  store negLitI32 tAbsAddr3
  let newTd ← iadd tfSaved k.c1
  let newDd2 ← iadd tfDd k.c1
  jump upEntry.ref [newTd, newDd2]

-- Helper: emit a string as istore8 sequence
def emitStringBytes (base : Val) (s : String) : IRBuilder Unit := do
  let bytes := s.toList.map (·.toNat)
  let mut addr := base
  let c1 ← iconst64 1
  for b in bytes do
    let bv ← iconst64 b
    istore8 bv addr
    addr ← iadd addr c1

-- Phase 4: Output formatting
def emitOutput (k : K) (outputBlk : DeclaredBlock) : IRBuilder Unit := do
  let writeFileBlk ← declareBlock [.i64]
  let satOutBlk ← declareBlock []
  let unsatOutBlk ← declareBlock []
  let varLoop ← declareBlock [.i64, .i64, .i64]
  let varDone ← declareBlock [.i64]
  let writeVar ← declareBlock []
  let writeNeg ← declareBlock [.i64, .i64, .i64, .i64]
  let writeDigits ← declareBlock [.i64, .i64, .i64, .i64]
  let digitLoop ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let digitBody ← declareBlock []
  let emitDigit ← declareBlock []
  let skipDigit ← declareBlock []
  let writeSpace ← declareBlock [.i64, .i64, .i64]

  startBlock outputBlk
  let resFlag ← load64 k.resultFlagAddr
  let isSat ← icmp .eq resFlag k.c1
  brif isSat satOutBlk.ref [] unsatOutBlk.ref []

  -- UNSAT
  startBlock unsatOutBlk
  let outBase ← iadd k.ptr k.outOffV
  emitStringBytes outBase "s UNSATISFIABLE\n"
  let unsatLen ← iconst64 16
  jump writeFileBlk.ref [unsatLen]

  -- SAT
  startBlock satOutBlk
  let outBase2 ← iadd k.ptr k.outOffV
  emitStringBytes outBase2 "s SATISFIABLE\nv "
  let nv ← load64 k.numVarsAddr
  let c16 ← iconst64 16
  jump varLoop.ref [c16, k.c0, nv]

  -- Variable loop
  startBlock varLoop
  let vOff := varLoop.param 0
  let vI := varLoop.param 1
  let vNv := varLoop.param 2
  let allDone ← icmp .uge vI vNv
  brif allDone varDone.ref [vOff] writeVar.ref []

  startBlock writeVar
  let aAddr ← iadd k.assignBase vI
  let aVal ← sload8_64 aAddr
  let posLit ← iadd vI k.c1
  let negLit ← ineg posLit
  let isPos ← icmp .sgt aVal k.c0
  let litVal ← select' isPos posLit negLit
  let isNegLit ← icmp .slt litVal k.c0
  brif isNegLit writeNeg.ref [vOff, vI, vNv, posLit] writeDigits.ref [vOff, vI, vNv, posLit]

  -- Write '-'
  startBlock writeNeg
  let wnOff := writeNeg.param 0
  let wnI := writeNeg.param 1
  let wnNv := writeNeg.param 2
  let wnAbs := writeNeg.param 3
  let outAddr ← iadd k.outOffV wnOff
  let outAbsAddr ← iadd k.ptr outAddr
  istore8 k.c45 outAbsAddr
  let wnOff1 ← iadd wnOff k.c1
  jump writeDigits.ref [wnOff1, wnI, wnNv, wnAbs]

  -- Write digits
  startBlock writeDigits
  let wdOff := writeDigits.param 0
  let wdI := writeDigits.param 1
  let wdNv := writeDigits.param 2
  let wdAbs := writeDigits.param 3
  let c10000 ← iconst64 10000
  jump digitLoop.ref [wdOff, wdI, wdNv, wdAbs, c10000, k.c0]

  startBlock digitLoop
  let dlOff := digitLoop.param 0
  let dlI := digitLoop.param 1
  let dlNv := digitLoop.param 2
  let dlRem := digitLoop.param 3
  let dlDiv := digitLoop.param 4
  let dlStarted := digitLoop.param 5
  let divZero ← icmp .eq dlDiv k.c0
  brif divZero writeSpace.ref [dlOff, dlI, dlNv] digitBody.ref []

  startBlock digitBody
  let digit ← udiv dlRem dlDiv
  let digitTimesDiv ← imul digit dlDiv
  let remainder ← isub dlRem digitTimesDiv
  let digitNZ ← icmp .ne digit k.c0
  let dnzExt ← uextend64 digitNZ
  let startedOrNZ ← bor dlStarted dnzExt
  let divIs1 ← icmp .eq dlDiv k.c1
  let di1Ext ← uextend64 divIs1
  let shouldWrite ← bor startedOrNZ di1Ext
  brif shouldWrite emitDigit.ref [] skipDigit.ref []

  startBlock emitDigit
  let ascii ← iadd digit k.c48
  let dOutAddr ← iadd k.outOffV dlOff
  let dOutAbsAddr ← iadd k.ptr dOutAddr
  istore8 ascii dOutAbsAddr
  let dlOff1 ← iadd dlOff k.c1
  let nextDiv ← udiv dlDiv k.c10
  jump digitLoop.ref [dlOff1, dlI, dlNv, remainder, nextDiv, k.c1]

  startBlock skipDigit
  let nextDiv2 ← udiv dlDiv k.c10
  jump digitLoop.ref [dlOff, dlI, dlNv, remainder, nextDiv2, k.c0]

  -- Write space
  startBlock writeSpace
  let wsOff := writeSpace.param 0
  let wsI := writeSpace.param 1
  let wsNv := writeSpace.param 2
  let spOutAddr ← iadd k.outOffV wsOff
  let spOutAbsAddr ← iadd k.ptr spOutAddr
  istore8 k.c32 spOutAbsAddr
  let wsOff1 ← iadd wsOff k.c1
  let wsI1 ← iadd wsI k.c1
  jump varLoop.ref [wsOff1, wsI1, wsNv]

  -- Finish SAT: write "0\n"
  startBlock varDone
  let fOff := varDone.param 0
  let fOutAddr ← iadd k.outOffV fOff
  let fOutAbsAddr ← iadd k.ptr fOutAddr
  istore8 k.c48 fOutAbsAddr
  let fOff1 ← iadd fOff k.c1
  let fOutAddr2 ← iadd k.outOffV fOff1
  let fOutAbsAddr2 ← iadd k.ptr fOutAddr2
  istore8 k.c10 fOutAbsAddr2
  let totalLen ← iadd fOff1 k.c1
  jump writeFileBlk.ref [totalLen]

  -- Write to file
  startBlock writeFileBlk
  let outLen := writeFileBlk.param 0
  let outFnameV ← iconst64 outputFilename_off
  let outStartV ← iconst64 out_off
  let _ ← call k.fnWrite [k.ptr, outFnameV, outStartV, k.c0, outLen]
  ret

-- ============================================================
-- Main builder
-- ============================================================

set_option maxRecDepth 4096 in
def clifIrSource : String := buildProgram do
  let fnRead ← declareFileRead
  let fnWrite ← declareFileWrite

  let ptr ← entryBlock
  let c0 ← iconst64 0
  let c1 ← iconst64 1
  let c4 ← iconst64 4
  let c8 ← iconst64 8
  let c10 ← iconst64 10
  let c32 ← iconst64 32
  let c45 ← iconst64 45
  let c48 ← iconst64 48
  let c58 ← iconst64 58

  let assignBase ← absAddr ptr assign_off
  let cnfOffV ← iconst64 cnf_off
  let dbOffV ← iconst64 db_off
  let clIdxOffV ← iconst64 clauseIndex_off
  let trailOffV ← iconst64 trail_off
  let outOffV ← iconst64 out_off
  let decStackV ← iconst64 decStack_off
  let numVarsAddr ← absAddr ptr numVars_off
  let clauseCountAddr ← absAddr ptr clauseCount_off
  let resultFlagAddr ← absAddr ptr resultFlag_off

  -- Read CNF file
  let inFnameV ← iconst64 inputFilename_off
  let cnfStartV ← iconst64 cnf_off
  let bytesRead ← call fnRead [ptr, inFnameV, cnfStartV, c0, c0]

  -- Initialize scratch
  store c0 numVarsAddr
  let numClAddr ← absAddr ptr numClauses_off
  store c0 numClAddr
  store c0 clauseCountAddr
  store c0 resultFlagAddr

  let k : K := {
    ptr, c0, c1, c4, c8, c10, c32, c45, c48, c58,
    assignBase, cnfOffV, dbOffV, clIdxOffV, trailOffV, outOffV, decStackV,
    numVarsAddr, clauseCountAddr, resultFlagAddr,
    bytesRead, fnWrite
  }

  -- Phase 1: Zero assignment array
  let parserStartBlk ← declareBlock []
  emitZeroAssign k parserStartBlk

  -- Phase 2: Parser
  let solverBlk ← declareBlock []
  startBlock parserStartBlk
  let newLineBlk ← declareBlock [.i64, .i64]
  let skipLineBlk ← declareBlock [.i64, .i64]
  let headerBlk ← declareBlock [.i64, .i64]
  let clauseBlk ← declareBlock [.i64, .i64]
  jump newLineBlk.ref [c0, c0]

  emitSkipLine k skipLineBlk newLineBlk solverBlk
  emitNewLine k newLineBlk skipLineBlk headerBlk clauseBlk solverBlk
  emitHeader k headerBlk skipLineBlk solverBlk
  emitClauseParser k clauseBlk newLineBlk

  -- Phase 3: DPLL solver
  let outputBlk ← declareBlock []
  let upEntry ← declareBlock [.i64, .i64]
  let decideBlk ← declareBlock [.i64, .i64]
  let conflictBlk ← declareBlock [.i64, .i64]
  let satBlk ← declareBlock [.i64]
  let unsatBlk ← declareBlock []

  startBlock solverBlk
  jump upEntry.ref [c0, c0]

  startBlock upEntry
  let upTd := upEntry.param 0
  let upDd := upEntry.param 1
  let scanHdr ← declareBlock [.i64, .i64, .i64, .i64]
  jump scanHdr.ref [upTd, upDd, c0, c0]

  emitUnitPropScan k scanHdr decideBlk conflictBlk upEntry
  emitDecide k decideBlk satBlk upEntry
  emitConflict k conflictBlk unsatBlk upEntry

  startBlock satBlk
  let one ← iconst64 1
  store one k.resultFlagAddr
  jump outputBlk.ref []

  startBlock unsatBlk
  let two ← iconst64 2
  store two k.resultFlagAddr
  jump outputBlk.ref []

  -- Phase 4: Output
  emitOutput k outputBlk

-- ---------------------------------------------------------------------------
-- Payload / Config / Algorithm
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  let reserved := zeros inputFilename_off
  let inputFname := padTo (stringToBytes "input.cnf") (outputFilename_off - inputFilename_off)
  let outputFname := padTo (stringToBytes "sat_output.txt") (cnf_off - outputFilename_off)
  reserved ++ inputFname ++ outputFname

def satConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := totalMemory,
  context_offset := 0
}

def satAlgorithm : Algorithm :=
  let clifCallAction : Action :=
    { kind := .ClifCall, dst := u32 0, src := u32 1, offset := u32 0, size := u32 0 }
  {
    actions := [clifCallAction],
    payloads := payloads,
    cranelift_units := 0,
    timeout_ms := some 300000
  }

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.satConfig Algorithm.satAlgorithm
  IO.println (Json.compress json)
