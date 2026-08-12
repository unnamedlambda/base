import AlgorithmLib.ClifData
import AlgorithmLib.Core
import AlgorithmLib.Bytes

namespace AlgorithmLib

namespace IR

/-- A declared block with its parameter values -/
structure DeclaredBlock where
  ref : BlockRef
  params : List (Val × ClifTy)

/-- Access the i-th parameter value of a declared block -/
def DeclaredBlock.param (blk : DeclaredBlock) (i : Nat) : Val :=
  match blk.params[i]? with
  | some (v, _) => v
  | none => { id := 0 }

/-- IR builder state -/
structure IRState where
  nextVal : Nat := 0
  nextBlock : Nat := 0
  nextSig : Nat := 0
  nextFn : Nat := 0
  currentBlock : Option BlockRef := none
  currentBlockParams : List (Val × ClifTy) := []
  currentInsts : List Inst := []  -- reverse order for O(1) prepend
  sigs : List SigDecl := []
  fns : List FnDecl := []
  blocks : List BlockData := []

/-- The IR builder monad -/
abbrev IRBuilder := StateM IRState

-- ---------------------------------------------------------------------------
-- Core operations
-- ---------------------------------------------------------------------------

/-- Allocate a fresh SSA value -/
def freshVal : IRBuilder Val := do
  let s ← get
  let v : Val := { id := s.nextVal }
  set { s with nextVal := s.nextVal + 1 }
  pure v

/-- Append an instruction to the current block (O(1) prepend, reversed at finalize) -/
private def emit (inst : Inst) : IRBuilder Unit :=
  modify fun s => { s with currentInsts := inst :: s.currentInsts }

/-- Finalize the current block, pushing it to the blocks list -/
private def finalizeCurrentBlock : IRBuilder Unit := do
  let s ← get
  match s.currentBlock with
  | none => pure ()
  | some bref =>
    let blk : BlockData := {
      ref := bref
      params := s.currentBlockParams
      insts := s.currentInsts.reverse
    }
    set { s with
      blocks := s.blocks ++ [blk]
      currentBlock := none
      currentBlockParams := []
      currentInsts := []
    }

/-- **Every block, including the one still being emitted.**

    `finalizeCurrentBlock` runs only from `startBlock`, so a builder's *last*
    block sits in `currentInsts` and never reaches `blocks`.  `buildFunction`
    finalizes before rendering, so the emitted CLIF is complete — but anything
    reading `IRState.blocks` directly sees a program with its final block
    missing.  Every analysis must go through this, not through `.blocks`. -/
def IRState.allBlocks (s : IRState) : List BlockData :=
  match s.currentBlock with
  | none      => s.blocks
  | some bref => s.blocks ++ [{ ref := bref
                                params := s.currentBlockParams
                                insts := s.currentInsts.reverse }]

/-- Declare a block with typed parameters. Returns the block and its param Vals.
    Does not start emitting into it yet. -/
def declareBlock (paramTys : List ClifTy) : IRBuilder DeclaredBlock := do
  let s ← get
  let bref : BlockRef := { id := s.nextBlock }
  let mut paramDecls : List (Val × ClifTy) := []
  let mut nextV := s.nextVal
  for ty in paramTys do
    paramDecls := paramDecls ++ [({ id := nextV : Val }, ty)]
    nextV := nextV + 1
  set { s with nextBlock := s.nextBlock + 1, nextVal := nextV }
  pure { ref := bref, params := paramDecls }

/-- Start emitting into a previously declared block. Finalizes the current block first. -/
def startBlock (blk : DeclaredBlock) : IRBuilder Unit := do
  finalizeCurrentBlock
  modify fun s => { s with
    currentBlock := some blk.ref
    currentBlockParams := blk.params
    currentInsts := []
  }

/-- Start the entry block: block0(v0: i64). Returns v0 (shared memory pointer). -/
def entryBlock : IRBuilder Val := do
  let blk ← declareBlock [.i64]
  startBlock blk
  pure (blk.param 0)

-- ---------------------------------------------------------------------------
-- FFI declarations
-- ---------------------------------------------------------------------------

/-- Declare a CLIF signature -/
def declareSig (params : List ClifTy) (result : Option ClifTy) : IRBuilder SigRef := do
  let s ← get
  let ref : SigRef := { id := s.nextSig }
  let decl : SigDecl := { ref := ref, params := params, result := result }
  set { s with
    nextSig := s.nextSig + 1
    sigs := s.sigs ++ [decl]
  }
  pure ref

/-- Declare an FFI function with a new signature -/
def declareFFI (name : String) (params : List ClifTy) (result : Option ClifTy) : IRBuilder FnRef := do
  let sig ← declareSig params result
  let s ← get
  let ref : FnRef := { id := s.nextFn }
  set { s with
    nextFn := s.nextFn + 1
    fns := s.fns ++ [{ ref := ref, callee := .import name, sig := sig : FnDecl }]
  }
  pure ref

-- ---------------------------------------------------------------------------
-- Instruction emitters — arithmetic
-- ---------------------------------------------------------------------------

def iconst64 (value : Int) : IRBuilder Val := do
  let v ← freshVal; emit (.iconst v .i64 value); pure v

def iconst32 (value : Int) : IRBuilder Val := do
  let v ← freshVal; emit (.iconst v .i32 value); pure v

def iadd (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.iadd v a b); pure v

def isub (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.isub v a b); pure v

def imul (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.imul v a b); pure v

def udiv (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.udiv v a b); pure v

def ineg (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.ineg v a); pure v

-- ---------------------------------------------------------------------------
-- Instruction emitters — bitwise
-- ---------------------------------------------------------------------------

def ishl (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.ishl v a b); pure v

def ushr (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.ushr v a b); pure v

def band (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.band v a b); pure v

def bandImm (a : Val) (imm : Int) : IRBuilder Val := do
  let c ← iconst64 imm; band a c

def bandNot (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.bandNot v a b); pure v

def bor (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.bor v a b); pure v

def bxor (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.bxor v a b); pure v

-- ---------------------------------------------------------------------------
-- Instruction emitters — immediate forms (convenience: emit iconst + op)
-- ---------------------------------------------------------------------------

def iaddImm (a : Val) (imm : Int) : IRBuilder Val := do
  let c ← iconst64 imm; iadd a c

def ishlImm (a : Val) (imm : Int) : IRBuilder Val := do
  let c ← iconst64 imm; ishl a c

def ushrImm (a : Val) (imm : Int) : IRBuilder Val := do
  let c ← iconst64 imm; ushr a c

-- ---------------------------------------------------------------------------
-- Instruction emitters — float / SIMD
-- ---------------------------------------------------------------------------

/-- Emit a 32-bit float constant. The value is carried as its IEEE 754 bit
    pattern, so a constant that cannot be spelled is a type error here rather
    than a parse failure at compile time. -/
def fconst32 (x : Float) : IRBuilder Val := do
  let v ← freshVal; emit (.fconst v .f32 x.toFloat32.toBits.toUInt64); pure v

/-- Emit a 64-bit float constant. -/
def fconst64 (x : Float) : IRBuilder Val := do
  let v ← freshVal; emit (.fconst v .f64 x.toBits); pure v

def f32Zero : Float := 0.0
def f64Zero : Float := 0.0

def fadd (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fadd v a b); pure v

def fsub (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fsub v a b); pure v

def fmul (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fmul v a b); pure v

def fmax (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fmax v a b); pure v

def fmin (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fmin v a b); pure v

def fpromote (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fpromote v a); pure v

def splat (ty : ClifTy) (src : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.splat v ty src); pure v

def extractlane (src : Val) (lane : Nat) : IRBuilder Val := do
  let v ← freshVal; emit (.extractlane v src lane); pure v

def loadF32 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { ty := .f32, notrapAligned := true } addr); pure v

def loadF64 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { ty := .f64, notrapAligned := true } addr); pure v

def loadF32x4 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { ty := .f32x4, notrapAligned := true } addr); pure v

def loadI8x16 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { ty := .i8x16, notrapAligned := true } addr); pure v

def storeF32 (val addr : Val) : IRBuilder Unit :=
  emit (.storeTyped .f32 val addr)

def storeF64 (val addr : Val) : IRBuilder Unit :=
  emit (.storeTyped .f64 val addr)

def storeI64 (val addr : Val) : IRBuilder Unit :=
  emit (.storeTyped .i64 val addr)

def storeI32 (val addr : Val) : IRBuilder Unit :=
  emit (.storeTyped .i32 val addr)

def iconst8 (value : Int) : IRBuilder Val := do
  let v ← freshVal; emit (.iconst v .i8 value); pure v

def fneg (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fneg v a); pure v

def fcvtFromSint (ty : ClifTy) (src : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fcvtFromSint v ty src); pure v

def fcvtToUint (ty : ClifTy) (src : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fcvtToUint v ty src); pure v

def fcmpGt (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fcmp v .gt a b); pure v

def fcmpLt (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fcmp v .lt a b); pure v

def bitcastTo (ty : ClifTy) (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.bitcast v ty a); pure v

def bitselect (c a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.bitselect v c a b); pure v

/-- `min(a, b)` with the hardware's NaN behaviour rather than IEEE
    `minimumNumber`. Cranelift lowers exactly this shape — the wasm `pmin`
    pattern — to a single `minps`, where `fmin` costs a NaN-correct sequence of
    about eight instructions. Use this when the data cannot be NaN.

    **Vector types only**: a scalar `fcmp` yields a one-bit mask, which cannot
    be bitcast to the operand width. Scalar tails should use `fmin`. -/
def pmin (ty : ClifTy) (a b : Val) : IRBuilder Val := do
  bitselect (← bitcastTo ty (← fcmpLt a b)) a b

/-- `max(a, b)`, likewise a single `maxps`. Note the reversed compare: the rule
    Cranelift matches is `bitselect(fcmp lt b a, a, b)`. -/
def pmax (ty : ClifTy) (a b : Val) : IRBuilder Val := do
  bitselect (← bitcastTo ty (← fcmpLt b a)) a b

def bitcastI64 (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.bitcast v .i64 a); pure v

def bitcastF64 (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.bitcast v .f64 a); pure v

def ctz32 (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.ctz v a); pure v

def popcnt32 (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.popcnt v a); pure v

def vhighBits (src : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.vhighBits v src); pure v


-- ---------------------------------------------------------------------------
-- Instruction emitters — type conversion
-- ---------------------------------------------------------------------------

def ireduce32 (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.ireduce32 v a); pure v

def uextend64 (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.uextend64 v a); pure v

def sextend64 (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.sextend64 v a); pure v

-- ---------------------------------------------------------------------------
-- Instruction emitters — memory
-- ---------------------------------------------------------------------------

def store (val addr : Val) : IRBuilder Unit :=
  emit (.store val addr)

def istore8 (val addr : Val) : IRBuilder Unit :=
  emit (.istore8 val addr)

def load64 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { ty := .i64 } addr); pure v

def load32 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { ty := .i32 } addr); pure v

def uload8_64 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { kind := .uload8, ty := .i64 } addr); pure v

def uload32_64 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { kind := .uload32, ty := .i64 } addr); pure v

def sload8_64 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { kind := .sload8, ty := .i64 } addr); pure v

def load_i8 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { ty := .i8 } addr); pure v

def load_i16 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v { ty := .i16 } addr); pure v

-- ---------------------------------------------------------------------------
-- Instruction emitters — comparison and selection
-- ---------------------------------------------------------------------------

def icmp (cond : ICmpCond) (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.icmp v cond a b); pure v

def icmpImm (cond : ICmpCond) (a : Val) (imm : Int) : IRBuilder Val := do
  let c ← iconst64 imm; icmp cond a c

def select' (cond a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.select v cond a b); pure v

-- ---------------------------------------------------------------------------
-- Instruction emitters — calls
-- ---------------------------------------------------------------------------

/-- Call a function that returns a value -/
def call (fn : FnRef) (args : List Val) : IRBuilder Val := do
  let v ← freshVal; emit (.call (some v) fn args); pure v

/-- Call a void function -/
def callVoid (fn : FnRef) (args : List Val) : IRBuilder Unit :=
  emit (.call none fn args)

-- ---------------------------------------------------------------------------
-- Instruction emitters — terminators
-- ---------------------------------------------------------------------------

def jump (target : BlockRef) (args : List Val := []) : IRBuilder Unit :=
  emit (.jump target args)

def brif (cond : Val) (thenBlk : BlockRef) (thenArgs : List Val := [])
         (elseBlk : BlockRef) (elseArgs : List Val := []) : IRBuilder Unit :=
  emit (.brif cond thenBlk thenArgs elseBlk elseArgs)

def ret : IRBuilder Unit :=
  emit .ret

-- ---------------------------------------------------------------------------
-- Loop combinators
--
-- Hide the canonical CLIF three-block loop choreography
-- (loopHdr / loopBody / loopExit).  `LoopTy` selects the counter width;
-- the same combinator emits either i32 or i64 loops.
-- ---------------------------------------------------------------------------

inductive LoopTy where | i32 | i64
  deriving BEq, Repr

def LoopTy.clif : LoopTy → ClifTy
  | .i32 => .i32
  | .i64 => .i64

def LoopTy.iconst (ty : LoopTy) (n : Int) : IRBuilder Val :=
  match ty with
  | .i32 => iconst32 n
  | .i64 => iconst64 n

/-- `forLoop ty limit body` — emit a counter loop from 0 to `limit` (exclusive),
    step 1. `body i` runs with the loop counter `i` bound. No carry. -/
def forLoop (ty : LoopTy) (limit : Val) (body : Val → IRBuilder Unit) : IRBuilder Unit := do
  let hdr  ← declareBlock [ty.clif]
  let bdy  ← declareBlock [ty.clif]
  let exit ← declareBlock []
  jump hdr.ref [← ty.iconst 0]
  startBlock hdr
  let iHdr := hdr.param 0
  let cond ← icmp .ult iHdr limit
  brif cond bdy.ref [iHdr] exit.ref []
  startBlock bdy
  let iBdy := bdy.param 0
  body iBdy
  let inc ← iaddImm iBdy 1
  jump hdr.ref [inc]
  startBlock exit

/-- `forLoopFromTo ty start limit body` — counter from `start` to `limit`. -/
def forLoopFromTo (ty : LoopTy) (start limit : Val)
    (body : Val → IRBuilder Unit) : IRBuilder Unit := do
  let hdr  ← declareBlock [ty.clif]
  let bdy  ← declareBlock [ty.clif]
  let exit ← declareBlock []
  jump hdr.ref [start]
  startBlock hdr
  let iHdr := hdr.param 0
  let cond ← icmp .ult iHdr limit
  brif cond bdy.ref [iHdr] exit.ref []
  startBlock bdy
  let iBdy := bdy.param 0
  body iBdy
  let inc ← iaddImm iBdy 1
  jump hdr.ref [inc]
  startBlock exit

/-- `forLoopAcc ty accTy limit acc0 body` — counter loop with a single Val
    accumulator. `body i acc` returns the next `acc`. After the loop, the
    final accumulator is returned. -/
def forLoopAcc (ty : LoopTy) (accTy : ClifTy)
    (limit acc0 : Val) (body : Val → Val → IRBuilder Val) : IRBuilder Val := do
  let hdr  ← declareBlock [ty.clif, accTy]
  let bdy  ← declareBlock [ty.clif, accTy]
  let exit ← declareBlock [accTy]
  jump hdr.ref [← ty.iconst 0, acc0]
  startBlock hdr
  let iHdr := hdr.param 0
  let aHdr := hdr.param 1
  let cond ← icmp .ult iHdr limit
  brif cond bdy.ref [iHdr, aHdr] exit.ref [aHdr]
  startBlock bdy
  let iBdy := bdy.param 0
  let aBdy := bdy.param 1
  let nextAcc ← body iBdy aBdy
  let inc ← iaddImm iBdy 1
  jump hdr.ref [inc, nextAcc]
  startBlock exit
  return exit.param 0

/-- `whileLoop1 carryTy init cond body` — while-loop with a single Val carry.
    `cond c` returns the loop-continue bool; `body c` returns the next carry.
    The final carry value is returned. -/
def whileLoop1 (carryTy : ClifTy) (init : Val)
    (cond : Val → IRBuilder Val)
    (body : Val → IRBuilder Val) : IRBuilder Val := do
  let hdr  ← declareBlock [carryTy]
  let bdy  ← declareBlock [carryTy]
  let exit ← declareBlock [carryTy]
  jump hdr.ref [init]
  startBlock hdr
  let cHdr := hdr.param 0
  let ok ← cond cHdr
  brif ok bdy.ref [cHdr] exit.ref [cHdr]
  startBlock bdy
  let cBdy := bdy.param 0
  let next ← body cBdy
  jump hdr.ref [next]
  startBlock exit
  return exit.param 0

/-- `whileLoop2 a b ia ib cond body` — while loop with two Val carries. -/
def whileLoop2 (a b : ClifTy) (ia ib : Val)
    (cond : Val → Val → IRBuilder Val)
    (body : Val → Val → IRBuilder (Val × Val)) : IRBuilder (Val × Val) := do
  let hdr  ← declareBlock [a, b]
  let bdy  ← declareBlock [a, b]
  let exit ← declareBlock [a, b]
  jump hdr.ref [ia, ib]
  startBlock hdr
  let x := hdr.param 0; let y := hdr.param 1
  let ok ← cond x y
  brif ok bdy.ref [x, y] exit.ref [x, y]
  startBlock bdy
  let xb := bdy.param 0; let yb := bdy.param 1
  let (nx, ny) ← body xb yb
  jump hdr.ref [nx, ny]
  startBlock exit
  return (exit.param 0, exit.param 1)

/-- `forLoopAcc2 ty aTy bTy limit ia ib body` — counter loop with two
    accumulator carries.  Body returns `(nextA, nextB)`. -/
def forLoopAcc2 (ty : LoopTy) (aTy bTy : ClifTy)
    (limit ia ib : Val)
    (body : Val → Val → Val → IRBuilder (Val × Val)) : IRBuilder (Val × Val) := do
  let hdr  ← declareBlock [ty.clif, aTy, bTy]
  let bdy  ← declareBlock [ty.clif, aTy, bTy]
  let exit ← declareBlock [aTy, bTy]
  jump hdr.ref [← ty.iconst 0, ia, ib]
  startBlock hdr
  let i := hdr.param 0; let x := hdr.param 1; let y := hdr.param 2
  let cond ← icmp .ult i limit
  brif cond bdy.ref [i, x, y] exit.ref [x, y]
  startBlock bdy
  let iBdy := bdy.param 0
  let xBdy := bdy.param 1
  let yBdy := bdy.param 2
  let (nx, ny) ← body iBdy xBdy yBdy
  let inc ← iaddImm iBdy 1
  jump hdr.ref [inc, nx, ny]
  startBlock exit
  return (exit.param 0, exit.param 1)

-- ---------------------------------------------------------------------------
-- Top-level builders
-- ---------------------------------------------------------------------------

/-- Run an IR builder and produce one function of the program. -/
def buildFunction (funcIdx : Nat) (builder : IRBuilder Unit) : FuncData :=
  let (_, st) := builder.run {}
  -- Finalize last block if still open
  let (_, st) := finalizeCurrentBlock.run st
  { index := funcIdx, sigs := st.sigs, fns := st.fns, blocks := st.blocks }

/-- A function that does nothing, at a given index. -/
def noopAt (funcIdx : Nat) : FuncData :=
  buildFunction funcIdx do
    let _ ← entryBlock
    ret

/-- The standard noop function u0:0 -/
def noopFunction : FuncData := noopAt 0

/-- Assemble functions into a program. They must be in `u0:N` order: the
    runtime resolves call targets by treating the index as a `FuncId`. -/
def program (fs : List FuncData) : Program := { functions := fs }

/-- A two-function program: the noop slot, then the entry function. -/
def buildProgram (mainBuilder : IRBuilder Unit) : Program :=
  program [noopFunction, buildFunction 1 mainBuilder]

/-- Declare a colocated FFI function (intra-module call, e.g. colocated %ht_create) -/
def declareColocatedFFI (name : String) (params : List ClifTy) (result : Option ClifTy) : IRBuilder FnRef := do
  let sig ← declareSig params result
  let s ← get
  let ref : FnRef := { id := s.nextFn }
  set { s with
    nextFn := s.nextFn + 1
    fns := s.fns ++ [{ ref := ref, callee := .import name, sig := sig, colocated := true : FnDecl }]
  }
  pure ref

/-- Declare a call to another function of this same program, by `u0:N` index. -/
def declareLocal (index : Nat) (params : List ClifTy) (result : Option ClifTy) : IRBuilder FnRef := do
  let sig ← declareSig params result
  let s ← get
  let ref : FnRef := { id := s.nextFn }
  set { s with
    nextFn := s.nextFn + 1
    fns := s.fns ++ [{ ref := ref, callee := .local index, sig := sig, colocated := true : FnDecl }]
  }
  pure ref

/-- A function at `u0:wrapperIdx` that calls each of `callees` in order.

    Composes stages without the caller having to build the call sequence
    itself; the callees are named by index, so nothing here resolves a symbol. -/
def clifSequenceWrapper (wrapperIdx : Nat) (callees : List Nat) : FuncData :=
  let unique : List Nat :=
    callees.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []
  buildFunction wrapperIdx do
    let refs ← unique.mapM fun c => declareLocal c [ClifTy.i64] none
    let arg ← entryBlock
    for c in callees do
      let slot := (unique.idxOf? c).getD 0
      callVoid (refs[slot]!) [arg]
    ret

-- ---------------------------------------------------------------------------
-- High-level combinators
-- ---------------------------------------------------------------------------

/-- Compute absolute address: base + constant offset -/
def absAddr (base : Val) (offset : Nat) : IRBuilder Val := do
  let off ← iconst64 offset
  iadd base off

/-- Store a value at base + offset -/
def storeAt (base : Val) (offset : Nat) (val : Val) : IRBuilder Unit := do
  let addr ← absAddr base offset
  store val addr


end IR

end AlgorithmLib
