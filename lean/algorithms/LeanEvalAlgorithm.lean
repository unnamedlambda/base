import AlgorithmLib.Gen
import LayoutScan
import ShipScan

open Lean (Json)
open AlgorithmLib

namespace LeanEval

-- Payload layout (app fields start at 0x0038 = 56, past the reserved header)
def OUTPUT_PATH    : Nat := 0x0038
def INPUT_PATH     : Nat := 0x0078
def SOURCE_BUF     : Nat := 0x0178
def SOURCE_BUF_SZ  : Nat := 4096
def TRUE_STR       : Nat := 0x1178
def FALSE_STR      : Nat := 0x1180
def IDENT_BUF      : Nat := 0x1188
def IDENT_BUF_SZ   : Nat := 64
def HT_VAL_BUF     : Nat := 0x11C8
def OUTPUT_BUF     : Nat := 0x11D0
def OUTPUT_BUF_SZ  : Nat := 64
def STACK_BASE     : Nat := 0x1210
def STACK_SZ       : Nat := 512

open AlgorithmLib.IR
open AlgorithmLib.Prog

abbrev fnFileRead : Ffi := .fileRead
abbrev fnFileWrite : Ffi := .fileWrite
abbrev fnHtInit : Ffi := .htInit
abbrev fnHtCleanup : Ffi := .htCleanup
abbrev fnHtCreate : Ffi := .htCreate
abbrev fnHtInsert : Ffi := .htInsert
abbrev fnHtLookup : Ffi := .htLookup

/-- The constants and base addresses the evaluator reads, made once in the
    entry block. `ctx` is the binding table, created before the machine runs. -/
structure K (V : ClifTy → Type) where
  ptr : V .i64
  z8 : V .i8
  zero : V .i64
  one : V .i64
  c2 : V .i64
  c3 : V .i64
  c4 : V .i64
  c5 : V .i64
  c6 : V .i64
  c7 : V .i64
  c8 : V .i64
  c9 : V .i64
  c10 : V .i64
  ten : V .i64
  frameSize : V .i64
  space : V .i64
  newline : V .i64
  doneTag : V .i64
  srcAddr : V .i64
  identAddr : V .i64
  htValAddr : V .i64
  outBufAddr : V .i64
  stackAddr : V .i64
  ctx : V .i64

-- ---------------------------------------------------------------------------
-- The machine
--
-- Evaluation is a recursive descent whose recursion depth is the source's, so
-- it runs as a trampoline over an explicit frame stack:
--
--   the outer loop *descends* — skip space, push an expression frame, push a
--   term frame, parse an atom. An atom that opens a subexpression pushes its
--   own frame and goes round again; an atom that is a value falls through.
--
--   the inner loop *returns* — pop a frame and read its tag. A tag that only
--   combines values keeps popping; a tag that needs another subexpression
--   pushes a frame and re-enters the outer loop; the bottom frame leaves both.
--
-- `level` says where a descent starts, so the three entry points share one
-- body: 0 enters at the expression, 1 at the term, 2 at the atom.
-- ---------------------------------------------------------------------------

/-- The source byte at `pos`. -/
def srcByte (k : K V) (pos : V .i64) : Prog V L (V .i64) := do uload8_64 (← iadd k.srcAddr pos)

/-- The source byte `d` bytes past `pos`. -/
def srcByteAt (k : K V) (pos : V .i64) (d : Int) : Prog V L (V .i64) := do
  uload8_64 (← iadd k.srcAddr (← iaddImm pos d))

/-- Push a frame — tag, saved value, extra — and yield the new stack pointer. -/
def pushFrame (k : K V) (sp tag val extra : V .i64) : Prog V L (V .i64) := do
  let a ← iadd k.stackAddr sp
  store tag a
  store val (← iaddImm a 8)
  store extra (← iaddImm a 16)
  iadd sp k.frameSize

/-- Advance past spaces and newlines. -/
def skipWs (k : K V) (pos : V .i64) : Prog V L (V .i64) := do
  let e ← wloop1 pos
    (head := fun p => do
      let ch ← srcByte k p
      let isWs ← bor (← icmp .eq ch k.space) (← icmp .eq ch k.newline)
      return (contIf .ne isWs k.z8, %[p], ()))
    (body := fun p _ => return %[← iaddImm p 1])
  return e.head

/-- Advance one byte when the next is a space. -/
def skipOptSpace (k : K V) (pos : V .i64) : Prog V L (V .i64) := do
  select (← icmp .eq (← srcByte k pos) k.space) (← iaddImm pos 1) pos

/-- The evaluator's two loops, as labels.

    `Eval` is the outer one --- it carries the level, the position and the
    stack pointer, and going round it is how the machine descends. `Ret` is the
    inner one, which pops frames. The helpers below leave or re-enter one of
    them, and now say so in their types: under the old surface that was a
    `contTo 1` whose depth no reader could check. -/
abbrev EvalLbl (L : List ClifTy → List ClifTy → Type) :=
  L [ClifTy.i64, ClifTy.i64] [ClifTy.i64, ClifTy.i64, ClifTy.i64]
abbrev RetLbl (L : List ClifTy → List ClifTy → Type) :=
  L [ClifTy.i64, ClifTy.i64] [ClifTy.i64, ClifTy.i64, ClifTy.i64, ClifTy.i64]

/-- Nonzero when `ch` is a decimal digit. -/
def isDigitCh (k : K V) (ch : V .i64) : Prog V L (V .i8) := do
  band (← icmp .uge ch (← iconst64 48)) (← icmp .ule ch (← iconst64 57))

/-- Nonzero when `ch` may appear in an identifier. -/
def isIdentCh (k : K V) (ch : V .i64) : Prog V L (V .i8) := do
  let lower ← band (← icmp .uge ch (← iconst64 97)) (← icmp .ule ch (← iconst64 122))
  let upper ← band (← icmp .uge ch (← iconst64 65)) (← icmp .ule ch (← iconst64 90))
  let digit ← isDigitCh k ch
  let under ← icmp .eq ch (← iconst64 95)
  bor (← bor (← bor lower upper) digit) under

/-- A decimal literal; the result is the position after it and its value. -/
def parseNumber (k : K V) (start : V .i64) : Prog V L (V .i64 × V .i64) := do
  let e ← wloop2 start k.zero
    (head := fun p a => do
      let ch ← srcByte k p
      return (contIf .ne (← isDigitCh k ch) k.z8, %[p, a], ch))
    (body := fun p a ch => do
      let a' ← iadd (← imul a k.ten) (← isub ch (← iconst64 48))
      return %[← iaddImm p 1, a'])
  return (e.head, e.snd)

/-- Copy a name into the identifier buffer, stopping at the first space; the
    result is the position of that space and the length written. -/
def readName (k : K V) (start : V .i64) : Prog V L (V .i64 × V .i64) := do
  let e ← wloop2 start k.zero
    (head := fun p n => do
      let ch ← srcByte k p
      return (exitIf .eq ch k.space, %[p, n], ch))
    (body := fun p n ch => do
      istore8 ch (← iadd k.identAddr n)
      return %[← iaddImm p 1, ← iaddImm n 1])
  return (e.head, e.snd)

/-- An identifier, looked up in the binding table. -/
def readVar (k : K V) (start : V .i64) : Prog V L (V .i64 × V .i64) := do
  let e ← wloop2 start k.zero
    (head := fun p n => do
      let ch ← srcByte k p
      return (contIf .ne (← isIdentCh k ch) k.z8, %[p, n], ch))
    (body := fun p n ch => do
      istore8 ch (← iadd k.identAddr n)
      return %[← iaddImm p 1, ← iaddImm n 1])
  let _ ← ffi fnHtLookup
    %[k.ctx, k.identAddr, ← ireduce32 (e.snd), k.htValAddr]
  return (e.head, ← load64 k.htValAddr)

/-- Bind the identifier buffer's first `len` bytes to `value`. -/
def bindName (k : K V) (len value : V .i64) : Prog V L Unit := do
  store value k.htValAddr
  ffiVoid fnHtInsert
    %[k.ctx, k.identAddr, ← ireduce32 len, k.htValAddr, ← iconst32 8]

/-- Scan a lambda body to its matching `)`, and yield the position after it. -/
def scanToClose (k : K V) (start : V .i64) : Prog V L (V .i64) := do
  let e ← wloop2L start k.zero
    (head := fun _ p _ => return (exitIf .ne k.zero k.zero, %[p], ()))
    (body := fun lbl p depth _ => do
      let ch ← srcByte k p
      let next ← iaddImm p 1
      let _ ← ifte .eq ch (← iconst64 41)
        (do
          let _ ← ifte .eq depth k.zero (do brk lbl %[next]; pure %[]) (pure %[])
          continueWith lbl %[next, ← isub depth k.one]
          pure %[])
        (do
          let deeper ← select (← icmp .eq ch (← iconst64 40)) (← iadd depth k.one) depth
          continueWith lbl %[next, deeper]
          pure %[])
      return %[p, depth])
  return e.head

-- ---------------------------------------------------------------------------
-- Descend: classify the atom, then handle it exactly once
-- ---------------------------------------------------------------------------

/-- Which atom starts at `pos`: 0 number, 1 identifier, 2 parenthesis,
    3 lambda, 4 `let`, 5 `if`, 6 `true`, 7 `false`.

    The keyword tests are nested so a source byte is read only when the
    preceding one matched, as a hand-written chain of branches would. Naming
    the answer keeps the handlers below from being duplicated into every leaf
    that falls back to an identifier. -/
def classifyAtom (k : K V) (pos : V .i64) : Prog V L (V .i64) := do
  let ch ← srcByte k pos
  let cls ← ifte .ne (← isDigitCh k ch) k.z8 (return %[k.zero])
    (ifte .eq ch (← iconst64 40)
      (do
        let ch1 ← srcByteAt k pos 1
        ifte .eq ch1 (← iconst64 102) (return %[k.c3]) (return %[k.c2]))
      (ifte .eq ch (← iconst64 108)
        (do
          let ok ← band
            (← band (← icmp .eq (← srcByteAt k pos 1) (← iconst64 101))
                    (← icmp .eq (← srcByteAt k pos 2) (← iconst64 116)))
            (← icmp .eq (← srcByteAt k pos 3) k.space)
          ifte .ne ok k.z8 (return %[k.c4]) (return %[k.one]))
        (ifte .eq ch (← iconst64 105)
          (do
            let ok ← band (← icmp .eq (← srcByteAt k pos 1) (← iconst64 102))
                          (← icmp .eq (← srcByteAt k pos 2) k.space)
            ifte .ne ok k.z8 (return %[k.c5]) (return %[k.one]))
          (ifte .eq ch (← iconst64 116)
            (do
              let rue ← band
                (← band (← icmp .eq (← srcByteAt k pos 1) (← iconst64 114))
                        (← icmp .eq (← srcByteAt k pos 2) (← iconst64 117)))
                (← icmp .eq (← srcByteAt k pos 3) (← iconst64 101))
              let ok ← band rue (← icmp .ult (← srcByteAt k pos 4) (← iconst64 97))
              ifte .ne ok k.z8 (return %[k.c6]) (return %[k.one]))
            (ifte .eq ch (← iconst64 102)
              (do
                let alse ← band
                  (← band (← icmp .eq (← srcByteAt k pos 1) (← iconst64 97))
                          (← icmp .eq (← srcByteAt k pos 2) (← iconst64 108)))
                  (← band (← icmp .eq (← srcByteAt k pos 3) (← iconst64 115))
                          (← icmp .eq (← srcByteAt k pos 4) (← iconst64 101)))
                let ok ← band alse (← icmp .ult (← srcByteAt k pos 5) (← iconst64 97))
                ifte .ne ok k.z8 (return %[k.c7]) (return %[k.one]))
              (return %[k.one]))))))
  return cls.head

/-- One descent, from `level`, with `sp` as the live frame stack.

    An atom that opens a subexpression pushes its frame and re-enters the outer
    loop; the exports are the position, the live stack pointer, the value and
    whether it is a boolean, for the atoms that are values already. -/
def descend (k : K V) (evalLbl : EvalLbl L) (level pos sp : V .i64) :
    Prog V L (Vals V [ClifTy.i64, ClifTy.i64, ClifTy.i64, ClifTy.i64]) := do
  -- level 0 enters at the expression, level 1 at the term, level 2 at the atom
  let atExpr ← ifte .eq level k.zero
    (do
      let p ← skipWs k pos
      return %[p, ← pushFrame k sp k.zero k.zero k.zero])
    (return %[pos, sp])
  let pos1 := atExpr.head
  let atTerm ← ifte .ule level k.one
    (do return %[← pushFrame k (atExpr.snd) k.one k.zero k.zero])
    (return %[atExpr.snd])
  let sp1 := atTerm.head
  let pos2 ← skipWs k pos1
  let cls ← classifyAtom k pos2
  ifte .eq cls k.zero
    (do
      let (p, v) ← parseNumber k pos2
      return %[p, sp1, v, k.zero])
    (ifte .eq cls k.one
      (do
        let (p, v) ← readVar k pos2
        return %[p, sp1, v, k.zero])
      (ifte .eq cls k.c2
        (do
          let sp2 ← pushFrame k sp1 k.c7 k.zero k.zero
          continueWith evalLbl %[k.zero, ← iaddImm pos2 1, sp2])
        (ifte .eq cls k.c3
          (do
            -- (fun <name> => <body>) <arg>
            let (pName, len) ← readName k (← iaddImm pos2 5)
            let pBody ← iaddImm pName 4
            let sp2 ← pushFrame k sp1 k.c8 pBody len
            let pArg ← skipOptSpace k (← scanToClose k pBody)
            let sp3 ← pushFrame k sp2 k.c9 k.zero k.zero
            continueWith evalLbl %[k.one, pArg, sp3])
          (ifte .eq cls k.c4
            (do
              let (pName, len) ← readName k (← iaddImm pos2 4)
              let sp2 ← pushFrame k sp1 k.c2 len k.zero
              continueWith evalLbl %[k.zero, ← iaddImm pName 4, sp2])
            (ifte .eq cls k.c5
              (do
                let sp2 ← pushFrame k sp1 k.c4 k.zero k.zero
                continueWith evalLbl %[k.zero, ← iaddImm pos2 3, sp2])
              (ifte .eq cls k.c6
                (return %[← iaddImm pos2 4, sp1, k.one, k.one])
                (return %[← iaddImm pos2 5, sp1, k.zero, k.one])))))))

-- ---------------------------------------------------------------------------
-- Return: pop a frame and act on its tag
-- ---------------------------------------------------------------------------

/-- After a term: look for `+`, `-`, `<` or `>`. Finding one pushes the frame
    that will combine it and descends into the right-hand side; finding none
    means this expression is finished, so the return loop pops again. -/
def exprOperator (k : K V) (evalLbl : EvalLbl L) (retLbl : RetLbl L)
    (pos sp value isBool : V .i64) : Prog V L Unit := do
  let p ← skipWs k pos
  let ch ← srcByte k p
  let _ ← ifte .eq ch (← iconst64 43)
    (do
      let sp' ← pushFrame k sp k.zero value k.one
      continueWith evalLbl %[k.one, ← skipOptSpace k (← iaddImm p 1), sp'])
    (ifte .eq ch (← iconst64 45)
      (do
        let sp' ← pushFrame k sp k.zero value k.c2
        continueWith evalLbl %[k.one, ← skipOptSpace k (← iaddImm p 1), sp'])
      (ifte .eq ch (← iconst64 60)
        (do
          let after ← iaddImm p 1
          let isEq ← icmp .eq (← srcByte k after) (← iconst64 61)
          let p1 ← select isEq (← iaddImm after 1) after
          let sp' ← pushFrame k sp k.c10 value (← select isEq k.one k.zero)
          continueWith evalLbl %[k.one, ← skipOptSpace k p1, sp'])
        (ifte .eq ch (← iconst64 62)
          (do
            let after ← iaddImm p 1
            let isEq ← icmp .eq (← srcByte k after) (← iconst64 61)
            let p1 ← select isEq (← iaddImm after 1) after
            let sp' ← pushFrame k sp k.c10 value (← select isEq k.c3 k.c2)
            continueWith evalLbl %[k.one, ← skipOptSpace k p1, sp'])
          (do continueWith retLbl %[p, sp, value, isBool]; pure %[]))))

/-- After an atom: look for `*`, and otherwise pop again. -/
def termOperator (k : K V) (evalLbl : EvalLbl L) (retLbl : RetLbl L)
    (pos sp value isBool : V .i64) : Prog V L Unit := do
  let p ← skipWs k pos
  let _ ← ifte .eq (← srcByte k p) (← iconst64 42)
    (do
      let sp' ← pushFrame k sp k.one value k.one
      continueWith evalLbl %[k.c2, ← skipOptSpace k (← iaddImm p 1), sp'])
    (do continueWith retLbl %[p, sp, value, isBool]; pure %[])

/-- The comparison a `cmp_rhs` frame was pushed for: 0 `<`, 1 `<=`, 2 `>`,
    3 `>=`. -/
def compareBy (k : K V) (op left right : V .i64) : Prog V L (V .i64) := do
  let r ← ifte .eq op k.zero (do return %[← uextend64 (← icmp .slt left right)])
    (ifte .eq op k.one (do return %[← uextend64 (← icmp .sle left right)])
      (ifte .eq op k.c2 (do return %[← uextend64 (← icmp .sgt left right)])
        (do return %[← uextend64 (← icmp .sge left right)])))
  return r.head

-- ---------------------------------------------------------------------------
-- Output
-- ---------------------------------------------------------------------------

/-- A literal string and its terminator, at the start of the output buffer. -/
def writeCStr (k : K V) (s : String) : Prog V L Unit := do
  let mut i : Int := 0
  for b in s.toList.map (·.toNat) do
    istore8 (← iconst64 b) (← iaddImm k.outBufAddr i)
    i := i + 1
  istore8 k.zero (← iaddImm k.outBufAddr i)

/-- The decimal form of `value`, most significant digit first. -/
def writeDecimal (k : K V) (value : V .i64) : Prog V L Unit := do
  let top ← wloop1 k.one
    (head := fun d => do
      let next ← imul d k.ten
      return (exitIf .ugt next value, %[d], next))
    (body := fun _ next => return %[next])
  let e ← wloopL %[value, k.zero, top.head]
    (head := fun _ cs => return (exitIf .ne k.zero k.zero, %[cs.snd], ()))
    (body := fun lbl cs _ => do
      let rem := cs.head
      let outPos := cs.snd
      let div := cs.thd
      let digit ← udiv rem div
      istore8 (← iadd digit (← iconst64 48)) (← iadd k.outBufAddr outPos)
      let rem' ← isub rem (← imul digit div)
      let div' ← udiv div k.ten
      let outPos' ← iaddImm outPos 1
      let _ ← ifte .eq div' k.zero (do brk lbl %[outPos']; pure %[]) (pure %[])
      return %[rem', outPos', div'])
  let outPos := e.head
  istore8 k.newline (← iadd k.outBufAddr outPos)
  istore8 k.zero (← iadd k.outBufAddr (← iaddImm outPos 1))

-- ---------------------------------------------------------------------------
-- Main body
-- ---------------------------------------------------------------------------

set_option maxRecDepth 8192 in
def mainCode : Prog V L Unit := do
  let ptr ← basePtr
  let z8 ← iconst .i8 0
  let zero ← iconst64 0
  let one ← iconst64 1
  let c2 ← iconst64 2
  let c3 ← iconst64 3
  let c4 ← iconst64 4
  let c5 ← iconst64 5
  let c6 ← iconst64 6
  let c7 ← iconst64 7
  let c8 ← iconst64 8
  let c9 ← iconst64 9
  let c10 ← iconst64 10
  let ten ← iconst64 10
  let frameSize ← iconst64 24
  let space ← iconst64 32
  let newline ← iconst64 10
  let doneTag ← iconst64 255
  let srcAddr ← absAddr ptr SOURCE_BUF
  let identAddr ← absAddr ptr IDENT_BUF
  let htValAddr ← absAddr ptr HT_VAL_BUF
  let outBufAddr ← absAddr ptr OUTPUT_BUF
  let stackAddr ← absAddr ptr STACK_BASE

  let _ ← readFile ptr INPUT_PATH SOURCE_BUF
  let htSlotPtr ← absAddr ptr AlgorithmLib.ContextSlots.ht
  ffiVoid fnHtInit %[htSlotPtr]
  let ctx ← load64 htSlotPtr
  let _ ← ffi fnHtCreate %[ctx]

  -- The bottom frame: reaching it means the whole expression is evaluated.
  store doneTag stackAddr
  store zero (← iaddImm stackAddr 8)
  store zero (← iaddImm stackAddr 16)

  let k : K V := {
    ptr := ptr, z8 := z8, zero := zero, one := one,
    c2 := c2, c3 := c3, c4 := c4, c5 := c5, c6 := c6, c7 := c7,
    c8 := c8, c9 := c9, c10 := c10, ten := ten,
    frameSize := frameSize, space := space, newline := newline,
    doneTag := doneTag, srcAddr := srcAddr, identAddr := identAddr,
    htValAddr := htValAddr, outBufAddr := outBufAddr, stackAddr := stackAddr,
    ctx := ctx }

  -- `#eval ` is six bytes; one frame is already pushed.
  let answer ← wloopL %[zero, ← iconst64 6, frameSize]
    (head := fun _ _ => return (exitIf .ne zero zero, %[zero, zero], ()))
    (body := fun evalLbl cs _ => do
      let atom ← descend k evalLbl cs.head cs.snd cs.thd
      let done ← wloopL %[atom.head, atom.snd, atom.thd, atom.fth]
        (head := fun _ rs => do
          let sp ← isub (rs.snd) k.frameSize
          let fa ← iadd k.stackAddr sp
          let tag ← load64 fa
          let savedVal ← load64 (← iaddImm fa 8)
          let savedExtra ← load64 (← iaddImm fa 16)
          return (exitIf .eq tag k.doneTag, %[rs.thd, rs.fth],
                  (sp, tag, savedVal, savedExtra)))
        (body := fun retLbl rs x => do
          let pos := rs.head
          let value := rs.thd
          let isBool := rs.fth
          let (sp, tag, savedVal, savedExtra) := x
          let _ ← ifte .eq tag k.zero
            (do
              -- expression frame: extra 0 is the first term, 1 add, 2 subtract
              let j ← ifte .eq savedExtra k.zero (return %[value, isBool])
                (do
                  let sum ← iadd savedVal value
                  let diff ← isub savedVal value
                  return %[← select (← icmp .eq savedExtra k.one) sum diff, k.zero])
              exprOperator k evalLbl retLbl pos sp j.head j.snd
              pure %[])
            (ifte .eq tag k.one
              (do
                -- term frame: extra 0 is the first atom, 1 multiply
                let j ← ifte .eq savedExtra k.zero (return %[value, isBool])
                  (do return %[← imul savedVal value, k.zero])
                termOperator k evalLbl retLbl pos sp j.head j.snd
                pure %[])
              (ifte .eq tag k.c2
                (do
                  -- let: bind the name, then evaluate the body
                  bindName k savedVal value
                  let semi ← icmp .eq (← srcByte k pos) (← iconst64 59)
                  let p ← select semi (← iaddImm pos 1) pos
                  let sp' ← pushFrame k sp k.c3 k.zero k.zero
                  continueWith evalLbl %[k.zero, p, sp'])
                (ifte .eq tag k.c3
                  (do continueWith retLbl %[pos, sp, value, isBool]; pure %[])
                  (ifte .eq tag k.c4
                    (do
                      -- if: the condition is in hand, evaluate the then-branch
                      let p ← iaddImm (← skipOptSpace k pos) 5
                      let sp' ← pushFrame k sp k.c5 value k.zero
                      continueWith evalLbl %[k.zero, p, sp'])
                    (ifte .eq tag k.c5
                      (do
                        let p ← iaddImm (← skipOptSpace k pos) 5
                        let sp' ← pushFrame k sp k.c6 savedVal value
                        continueWith evalLbl %[k.zero, p, sp'])
                      (ifte .eq tag k.c6
                        (do
                          let taken ← select (← icmp .ne savedVal k.zero) savedExtra value
                          continueWith retLbl %[pos, sp, taken, k.zero]
                          pure %[])
                        (ifte .eq tag k.c7
                          (do
                            let close ← icmp .eq (← srcByte k pos) (← iconst64 41)
                            let p ← select close (← iaddImm pos 1) pos
                            continueWith retLbl %[p, sp, value, isBool]
                            pure %[])
                          (ifte .eq tag k.c8
                            (do
                              -- the lambda's argument is evaluated: bind it and
                              -- run the body the frame remembered
                              bindName k savedExtra value
                              let sp' ← pushFrame k sp k.c3 k.zero k.zero
                              continueWith evalLbl %[k.zero, savedVal, sp'])
                            (ifte .eq tag k.c9
                              (do continueWith retLbl %[pos, sp, value, isBool]; pure %[])
                              (ifte .eq tag k.c10
                                (do
                                  let r ← compareBy k savedExtra savedVal value
                                  continueWith retLbl %[pos, sp, r, k.one]
                                  pure %[])
                                (do
                                  continueWith retLbl %[pos, sp, value, isBool]
                                  pure %[])))))))))))
          return rs)
      brk evalLbl %[done.head, done.snd]
      return cs)

  let value := answer.head
  let isBool := answer.snd
  let _ ← ifte .ne isBool zero
    (do
      let _ ← ifte .ne value zero
        (do
          writeCStr k "true\n"
          let _ ← writeFile0 ptr OUTPUT_PATH OUTPUT_BUF zero
          pure %[])
        (do
          writeCStr k "false\n"
          let _ ← writeFile0 ptr OUTPUT_PATH OUTPUT_BUF zero
          pure %[])
      pure %[])
    (do
      writeDecimal k value
      let _ ← writeFile0 ptr OUTPUT_PATH OUTPUT_BUF zero
      ffiVoid fnHtCleanup %[htSlotPtr]
      pure %[])

-- Deciding `wf` walks the whole body: deeper than the default recursion budget,
-- and long enough that the kernel does not finish inside the default heartbeats.
set_option maxRecDepth 100000 in
set_option maxHeartbeats 2000000 in
def clifIrSource : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 mainCode)]

-- ---------------------------------------------------------------------------
-- Payload construction
-- ---------------------------------------------------------------------------

def buildPayload : List UInt8 :=
  let reserved    := zeros 56                              -- 0x0000-0x0037: reserved header (context slots, then unused)
  let outputPath  := padTo (stringToBytes "output.txt") 64 -- 0x0038
  let inputPath   := zeros 256                             -- 0x0078
  let sourceBuf   := zeros SOURCE_BUF_SZ                   -- 0x0178
  let trueStr     := padTo (stringToBytes "true") 8        -- 0x1178
  let falseStr    := padTo (stringToBytes "false") 8       -- 0x1180
  let identBuf    := zeros IDENT_BUF_SZ                    -- 0x1188
  let htValBuf    := zeros 8                               -- 0x11C8
  let outputBuf   := zeros OUTPUT_BUF_SZ                   -- 0x11D0
  let stackRegion := zeros STACK_SZ                        -- 0x1210
  reserved ++
    outputPath ++ inputPath ++ sourceBuf ++
    trueStr ++ falseStr ++ identBuf ++ htValBuf ++ outputBuf ++
    stackRegion

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the FFI's init calls fill, so naming
    those is what stops an offset being placed where one of those calls will
    overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"output_path", OUTPUT_PATH, INPUT_PATH - OUTPUT_PATH⟩,
   ⟨"input_path",  INPUT_PATH, SOURCE_BUF - INPUT_PATH⟩,
   ⟨"source_buf",  SOURCE_BUF, SOURCE_BUF_SZ⟩,
   ⟨"true_str",    TRUE_STR, FALSE_STR - TRUE_STR⟩,
   ⟨"false_str",   FALSE_STR, IDENT_BUF - FALSE_STR⟩,
   ⟨"ident_buf",   IDENT_BUF, IDENT_BUF_SZ⟩,
   ⟨"ht_val_buf",  HT_VAL_BUF, OUTPUT_BUF - HT_VAL_BUF⟩,
   ⟨"output_buf",  OUTPUT_BUF, OUTPUT_BUF_SZ⟩,
   ⟨"stack",       STACK_BASE, STACK_SZ⟩]


#eval LayoutScan.check "LeanEvalAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

-- The memory this ships is sized from the payload it builds, so there is no
-- constant to bound the regions against; `okB` is the whole check here.

def buildSetup (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := buildPayload.length,
  initial_memory := buildPayload
}

def buildAlgorithm : UInt32 := IR.mainFnIdx

end LeanEval

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie LeanEval.clifIrSource
  emitArtifacts outDir #[artifactEntry "lean_eval_app" (LeanEval.buildSetup clif)]

#eval ShipScan.check "LeanEvalAlgorithm"
