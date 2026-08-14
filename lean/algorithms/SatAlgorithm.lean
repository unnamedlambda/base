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

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

def ffiEnv : (FnRef × FnRef) × FnEnv :=
  (Id.run (do
    let rd := IR.FFI.std.fileRead
    let wr := IR.FFI.std.fileWrite
    return (rd, wr)), env% [.fileIO])

def fnRead : FnRef := ffiEnv.1.1
def fnWrite : FnRef := ffiEnv.1.2
def env : FnEnv := ffiEnv.2

/-- The constants and base addresses the whole body reads, made once in the
    entry block. -/
structure K where
  ptr : R
  z8 : R
  c0 : R
  c1 : R
  c2 : R
  cM1 : R
  c4 : R
  c8 : R
  c9 : R
  c10 : R
  c13 : R
  c32 : R
  c45 : R
  c48 : R
  c58 : R
  c99 : R
  c112 : R
  assignBase : R
  cnfOffV : R
  dbOffV : R
  clIdxOffV : R
  trailOffV : R
  outOffV : R
  decStackV : R
  numVarsAddr : R
  numClausesAddr : R
  clauseCountAddr : R
  resultFlagAddr : R
  bytesRead : R

/-- The address of element `idx` of the region starting at relative `base`. -/
def atOff (k : K) (base idx : R) : M R := do iadd k.ptr (← iadd base idx)

/-- The CNF text byte at `pos`. -/
def cnfByte (k : K) (pos : R) : M R := do
  uload8_64 (← atOff k k.cnfOffV pos)

/-- Nonzero when `byte` is an ASCII digit. -/
def isDigitByte (k : K) (byte : R) : M R := do
  band (← icmp .uge byte k.c48) (← icmp .ult byte k.c58)

-- ---------------------------------------------------------------------------
-- Parser
-- ---------------------------------------------------------------------------

/-- Advance past the next newline, or to the end of the text. -/
def skipLine (k : K) (start : R) : M R := do
  let e ← wloop1 start
    (head := fun p => return (exitIf .uge p k.bytesRead, [p], ()))
    (body := fun p _ => do
      let byte ← cnfByte k p
      let p1 ← iaddImm p 1
      let _ ← ifte .eq byte k.c10 (do brk [p1]; pure []) (pure [])
      return [p1])
  return e.headD 0

/-- Advance to the next digit, or to the end of the text. -/
def skipToDigit (k : K) (start : R) : M R := do
  let e ← wloop1 start
    (head := fun p => return (exitIf .uge p k.bytesRead, [p], ()))
    (body := fun p _ => do
      let d ← isDigitByte k (← cnfByte k p)
      let _ ← ifte .ne d k.z8 (do brk [p]; pure []) (pure [])
      return [← iaddImm p 1])
  return e.headD 0

/-- Read a run of digits as an unsigned decimal; the result is the position of
    the first byte that is not one, and the value. -/
def parseDigits (k : K) (start : R) : M (R × R) := do
  let e ← wloop2 start k.c0
    (head := fun p a => return (exitIf .uge p k.bytesRead, [p, a], ()))
    (body := fun p a _ => do
      let byte ← cnfByte k p
      let d ← isDigitByte k byte
      let _ ← ifte .eq d k.z8 (do brk [p, a]; pure []) (pure [])
      let a' ← iadd (← imul a k.c10) (← isub byte k.c48)
      return [← iaddImm p 1, a'])
  return (e.headD 0, e.getD 1 0)

/-- `p cnf <vars> <clauses>`; `pos` is at the `p`. Text that runs out mid-header
    leaves the counts at the zero the entry block stored. -/
def parseHeader (k : K) (pos : R) : M R := do
  let pa ← skipToDigit k (← iaddImm pos 1)
  let (pb, nv) ← parseDigits k pa
  store nv k.numVarsAddr
  let pc ← skipToDigit k pb
  let (pd, nc) ← parseDigits k pc
  store nc k.numClausesAddr
  skipLine k pd

/-- One clause, terminated by `0`, a newline, or the end of the text. The
    result is the position after it and the next free database offset — the
    same offset when the clause held no literals. -/
def parseClause (k : K) (pos dbp : R) : M (R × R) := do
  let e ← wloop [pos, ← iadd dbp k.c4, k.c0]
    (head := fun cs => return (exitIf .uge (cs.headD 0) k.bytesRead, cs, ()))
    (body := fun cs _ => do
      let p := cs.headD 0
      let lit := cs.getD 1 0
      let cnt := cs.getD 2 0
      let byte ← cnfByte k p
      let p1 ← iaddImm p 1
      let _ ← ifte .eq byte k.c10 (do brk [p1, lit, cnt]; pure [])
        (ifte .eq byte k.c13 (do brk [p1, lit, cnt]; pure [])
          (ifte .eq byte k.c32 (do continueWith [p1, lit, cnt]; pure [])
            (ifte .eq byte k.c9 (do continueWith [p1, lit, cnt]; pure [])
              (do
                let isMinus ← icmp .eq byte k.c45
                let (pEnd, acc) ← parseDigits k (← select isMinus p1 p)
                let _ ← ifte .eq acc k.c0 (do brk [pEnd, lit, cnt]; pure []) (pure [])
                let litVal ← select isMinus (← ineg acc) acc
                storeI32 (← ireduce32 litVal) (← atOff k k.dbOffV lit)
                continueWith [pEnd, ← iaddImm lit 4, ← iaddImm cnt 1]
                pure []))))
      return cs)
  let cnt := e.getD 2 0
  let r ← ifte .eq cnt k.c0 (return [dbp])
    (do
      storeI32 (← ireduce32 cnt) (← atOff k k.dbOffV dbp)
      let cc ← load64 k.clauseCountAddr
      store dbp (← atOff k k.clIdxOffV (← imul cc k.c8))
      store (← iadd cc k.c1) k.clauseCountAddr
      return [e.getD 1 0])
  return (e.headD 0, r.headD 0)

/-- The whole file: one loop whose body dispatches on the first byte of a line.
    Every state the DIMACS grammar has returns here, so there is nothing else
    to carry but the position and the next free database offset. -/
def parseCnf (k : K) : M Unit := do
  let _ ← wloop2 k.c0 k.c0
    (head := fun p _ => return (exitIf .uge p k.bytesRead, ([] : List R), ()))
    (body := fun p dbp _ => do
      let byte ← cnfByte k p
      let p1 ← iaddImm p 1
      let _ ← ifte .eq byte k.c99
        (do continueWith [← skipLine k p, dbp]; pure [])
        (ifte .eq byte k.c112
          (do continueWith [← parseHeader k p, dbp]; pure [])
          (ifte .eq byte k.c10 (do continueWith [p1, dbp]; pure [])
            (ifte .eq byte k.c13 (do continueWith [p1, dbp]; pure [])
              (ifte .eq byte k.c32 (do continueWith [p1, dbp]; pure [])
                (ifte .eq byte k.c9 (do continueWith [p1, dbp]; pure [])
                  (do
                    let (np, ndb) ← parseClause k p dbp
                    continueWith [np, ndb]
                    pure []))))))
      return [p, dbp])

-- ---------------------------------------------------------------------------
-- DPLL solver
-- ---------------------------------------------------------------------------

/-- One pass over every clause, assigning what it forces. The result is a
    status — `1` when a clause came out false under the current assignment —
    the trail depth, and whether anything was assigned. -/
def unitPropagate (k : K) (td0 : R) : M (R × R × R) := do
  let e ← wloop [k.c0, td0, k.c0]
    (head := fun cs => do
      let cc ← load64 k.clauseCountAddr
      return (exitIf .uge (cs.headD 0) cc, [k.c0, cs.getD 1 0, cs.getD 2 0], ()))
    (body := fun cs _ => do
      let idx := cs.headD 0
      let td := cs.getD 1 0
      let fu := cs.getD 2 0
      let dbPtrOff ← load64 (← atOff k k.clIdxOffV (← imul idx k.c8))
      let clLen ← uextend64 (← load32 (← atOff k k.dbOffV dbPtrOff))
      -- Count the unassigned literals, remember the last of them, and note
      -- whether any literal is already true.
      let r ← wloop [← iadd dbPtrOff k.c4, clLen, k.c0, k.c0, k.c0]
        (head := fun ls =>
          return (exitIfEq (ls.getD 1 0) k.c0,
                  [ls.getD 2 0, ls.getD 3 0, ls.getD 4 0], ()))
        (body := fun ls _ => do
          let off := ls.headD 0
          let rem := ls.getD 1 0
          let cu := ls.getD 2 0
          let lastLit := ls.getD 3 0
          let sat := ls.getD 4 0
          let lit ← sextend64 (← load32 (← atOff k k.dbOffV off))
          let isNeg ← icmp .slt lit k.c0
          let absLit ← select isNeg (← ineg lit) lit
          let aVal ← sload8_64 (← iadd k.assignBase (← isub absLit k.c1))
          let off' ← iaddImm off 4
          let rem' ← isub rem k.c1
          let _ ← ifte .eq aVal k.c0
            (do continueWith [off', rem', ← iaddImm cu 1, lit, sat]; pure [])
            (do
              let sign ← select (← icmp .sgt lit k.c0) k.c1 k.cM1
              let sat' ← select (← icmp .eq sign aVal) k.c1 sat
              continueWith [off', rem', cu, lastLit, sat']
              pure [])
          return ls)
      let cu := r.headD 0
      let lastLit := r.getD 1 0
      let nextIdx ← iaddImm idx 1
      let _ ← ifte .eq (r.getD 2 0) k.c1
        (do continueWith [nextIdx, td, fu]; pure [])
        (ifte .eq cu k.c0
          (do brk [k.c1, td, fu]; pure [])
          (ifte .eq cu k.c1
            (do
              let uAbs ← select (← icmp .slt lastLit k.c0) (← ineg lastLit) lastLit
              let uAssign ← select (← icmp .sgt lastLit k.c0) k.c1 k.cM1
              istore8 uAssign (← iadd k.assignBase (← isub uAbs k.c1))
              storeI32 (← ireduce32 lastLit) (← atOff k k.trailOffV (← imul td k.c4))
              continueWith [nextIdx, ← iaddImm td 1, k.c1]
              pure [])
            (do continueWith [nextIdx, td, fu]; pure [])))
      return cs)
  return (e.headD 0, e.getD 1 0, e.getD 2 0)

/-- Assign the lowest unassigned variable true and push a decision level, or
    leave the solver loop with the satisfied flag when there is none. -/
def decide (k : K) (td dd : R) : M Unit := do
  let nv ← load64 k.numVarsAddr
  let e ← wloop1 k.c0
    (head := fun vi => return (exitIf .uge vi nv, [vi], ()))
    (body := fun vi _ => do
      let aVal ← sload8_64 (← iadd k.assignBase vi)
      let _ ← ifte .eq aVal k.c0 (do brk [vi]; pure []) (pure [])
      return [← iaddImm vi 1])
  let vi := e.headD 0
  let _ ← ifte .uge vi nv
    (do brk [k.c1]; pure [])
    (do
      store td (← atOff k k.decStackV (← imul dd k.c8))
      istore8 k.c1 (← iadd k.assignBase vi)
      storeI32 (← ireduce32 (← iaddImm vi 1)) (← atOff k k.trailOffV (← imul td k.c4))
      continueWith [← iaddImm td 1, ← iaddImm dd 1]
      pure [])

/-- Undo the trail entry at `i`, leaving its variable unassigned. -/
def undoTrail (k : K) (i : R) : M Unit := do
  let lit ← sextend64 (← load32 (← atOff k k.trailOffV (← imul i k.c4)))
  let absLit ← select (← icmp .slt lit k.c0) (← ineg lit) lit
  istore8 k.c0 (← iadd k.assignBase (← isub absLit k.c1))

/-- Pop decision levels until one is found that was only tried true, and flip
    it; with no level left the formula is unsatisfiable. The inner loop is the
    only place a state of this solver goes back to itself. -/
def handleConflict (k : K) (td dd : R) : M Unit := do
  let r ← wloop2 td dd
    (head := fun t d => return (exitIf .eq d k.c0, [k.c0, t, d], ()))
    (body := fun t d _ => do
      let ddM1 ← isub d k.c1
      let savedTd ← load64 (← atOff k k.decStackV (← imul ddM1 k.c8))
      let _ ← wloop1 (← isub t k.c1)
        (head := fun i => return (exitIf .slt i savedTd, ([] : List R), ()))
        (body := fun i _ => do
          undoTrail k i
          return [← isub i k.c1])
      let decLit ← sextend64 (← load32 (← atOff k k.trailOffV (← imul savedTd k.c4)))
      let decAbs ← select (← icmp .slt decLit k.c0) (← ineg decLit) decLit
      let decVar ← isub decAbs k.c1
      istore8 k.c0 (← iadd k.assignBase decVar)
      let _ ← ifte .sgt decLit k.c0
        (do
          istore8 k.cM1 (← iadd k.assignBase decVar)
          storeI32 (← ireduce32 (← ineg (← iaddImm decVar 1)))
                   (← atOff k k.trailOffV (← imul savedTd k.c4))
          brk [k.c1, ← iaddImm savedTd 1, ← iaddImm ddM1 1]
          pure [])
        (pure [])
      return [savedTd, ddM1])
  let _ ← ifte .eq (r.headD 0) k.c0
    (do brk [k.c2]; pure [])
    (do continueWith [r.getD 1 0, r.getD 2 0]; pure [])

/-- The search: propagate, then either backtrack, propagate again, or decide.
    The loop is left only by `brk`, carrying the flag the output phase reads. -/
def solve (k : K) : M Unit := do
  let e ← wloop2 k.c0 k.c0
    (head := fun _ _ => return (exitIf .ne k.c0 k.c0, [k.c0], ()))
    (body := fun td dd _ => do
      let (status, td', fu) ← unitPropagate k td
      let _ ← ifte .eq status k.c1
        (do handleConflict k td' dd; pure [])
        (ifte .eq fu k.c1
          (do continueWith [td', dd]; pure [])
          (do decide k td' dd; pure []))
      return [td, dd])
  store (e.headD 0) k.resultFlagAddr

-- ---------------------------------------------------------------------------
-- Output
-- ---------------------------------------------------------------------------

/-- A literal string, one `istore8` per byte. -/
def emitStringBytes (base : R) (s : String) : M Unit := do
  let one ← iconst64 1
  let mut addr := base
  for b in s.toList.map (·.toNat) do
    istore8 (← iconst64 b) addr
    addr := (← iadd addr one)

/-- The DIMACS answer line, written to the output file. -/
def emitOutput (k : K) : M Unit := do
  let resFlag ← load64 k.resultFlagAddr
  let outBase ← iadd k.ptr k.outOffV
  let len ← ifte .eq resFlag k.c1
    (do
      emitStringBytes outBase "s SATISFIABLE\nv "
      let nv ← load64 k.numVarsAddr
      let e ← wloop2 (← iconst64 16) k.c0
        (head := fun off vi => return (exitIf .uge vi nv, [off], ()))
        (body := fun off vi _ => do
          let aVal ← sload8_64 (← iadd k.assignBase vi)
          let signed ← ifte .sgt aVal k.c0 (return [off])
            (do
              istore8 k.c45 (← atOff k k.outOffV off)
              return [← iaddImm off 1])
          -- Decimal, most significant digit first, suppressing leading zeros.
          let d ← wloop [signed.headD 0, ← iaddImm vi 1, ← iconst64 10000, k.c0]
            (head := fun cs => return (exitIf .eq (cs.getD 2 0) k.c0, [cs.headD 0], ()))
            (body := fun cs _ => do
              let o := cs.headD 0
              let rem := cs.getD 1 0
              let dv := cs.getD 2 0
              let started := cs.getD 3 0
              let digit ← udiv rem dv
              let rem' ← isub rem (← imul digit dv)
              let nz ← uextend64 (← icmp .ne digit k.c0)
              let last ← uextend64 (← icmp .eq dv k.c1)
              let write ← bor (← bor started nz) last
              let dv' ← udiv dv k.c10
              let _ ← ifte .ne write k.c0
                (do
                  istore8 (← iadd digit k.c48) (← atOff k k.outOffV o)
                  continueWith [← iaddImm o 1, rem', dv', k.c1]
                  pure [])
                (do continueWith [o, rem', dv', k.c0]; pure [])
              return cs)
          let o := d.headD 0
          istore8 k.c32 (← atOff k k.outOffV o)
          return [← iaddImm o 1, ← iaddImm vi 1])
      let fOff := e.headD 0
      istore8 k.c48 (← atOff k k.outOffV fOff)
      istore8 k.c10 (← atOff k k.outOffV (← iaddImm fOff 1))
      return [← iaddImm fOff 2])
    (do
      emitStringBytes outBase "s UNSATISFIABLE\n"
      return [← iconst64 16])
  let _ ← writeFile k.ptr fnWrite outputFilename_off out_off k.c0 (len.headD 0)

-- ---------------------------------------------------------------------------
-- Main body
-- ---------------------------------------------------------------------------

def mainCode : HProg.Code :=
  clif%(env, HProg.ptrParams) do
  let ptr := basePtr
  let z8 ← iconst .i8 0
  let c0 ← iconst64 0
  let c1 ← iconst64 1
  let c2 ← iconst64 2
  let cM1 ← iconst64 (-1)
  let c4 ← iconst64 4
  let c8 ← iconst64 8
  let c9 ← iconst64 9
  let c10 ← iconst64 10
  let c13 ← iconst64 13
  let c32 ← iconst64 32
  let c45 ← iconst64 45
  let c48 ← iconst64 48
  let c58 ← iconst64 58
  let c99 ← iconst64 99
  let c112 ← iconst64 112

  let assignBase ← absAddr ptr assign_off
  let cnfOffV ← iconst64 cnf_off
  let dbOffV ← iconst64 db_off
  let clIdxOffV ← iconst64 clauseIndex_off
  let trailOffV ← iconst64 trail_off
  let outOffV ← iconst64 out_off
  let decStackV ← iconst64 decStack_off
  let numVarsAddr ← absAddr ptr numVars_off
  let numClausesAddr ← absAddr ptr numClauses_off
  let clauseCountAddr ← absAddr ptr clauseCount_off
  let resultFlagAddr ← absAddr ptr resultFlag_off

  let bytesRead ← readFile ptr fnRead inputFilename_off cnf_off

  store c0 numVarsAddr
  store c0 numClausesAddr
  store c0 clauseCountAddr
  store c0 resultFlagAddr

  let k : K := {
    ptr := ptr, z8 := z8,
    c0 := c0, c1 := c1, c2 := c2, cM1 := cM1, c4 := c4, c8 := c8, c9 := c9,
    c10 := c10, c13 := c13, c32 := c32, c45 := c45, c48 := c48, c58 := c58,
    c99 := c99, c112 := c112,
    assignBase := assignBase, cnfOffV := cnfOffV, dbOffV := dbOffV,
    clIdxOffV := clIdxOffV, trailOffV := trailOffV, outOffV := outOffV,
    decStackV := decStackV, numVarsAddr := numVarsAddr,
    numClausesAddr := numClausesAddr, clauseCountAddr := clauseCountAddr,
    resultFlagAddr := resultFlagAddr, bytesRead := bytesRead }

  forLoop (← iconst64 maxVars) (fun i => do istore8 c0 (← iadd assignBase i))
  parseCnf k
  solve k
  emitOutput k

-- Deciding `wf` walks the whole body, which is deeper than the default budget.
def clifIrSource : Program :=
  IR.program [IR.noopFunction, HProg.compileFn 1 mainCode env]

-- ---------------------------------------------------------------------------
-- Payload / Config / Algorithm
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  let reserved := zeros inputFilename_off
  let inputFname := padTo (stringToBytes "input.cnf") (outputFilename_off - inputFilename_off)
  let outputFname := padTo (stringToBytes "sat_output.txt") (cnf_off - outputFilename_off)
  reserved ++ inputFname ++ outputFname

def satConfig : Setup := {
  clif := clifIrSource,
  memory_size := totalMemory,
  initial_memory := payloads
}

def satAlgorithm : Algorithm := {
    fn_idx := IR.mainFnIdx
  }

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[toJsonEntry "sat_app" Algorithm.satConfig Algorithm.satAlgorithm]
