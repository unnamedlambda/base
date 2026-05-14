import AlgorithmLib.Core
import AlgorithmLib.Bytes

namespace AlgorithmLib

namespace IR

/-- CLIF value types -/
inductive ClifTy where
  | i8 | i32 | i64
  | f32 | f64
  | f32x4 | i8x16
  deriving Repr, BEq

/-- An SSA value reference -/
structure Val where
  id : Nat
  deriving Repr, BEq

/-- A block reference -/
structure BlockRef where
  id : Nat
  deriving Repr, BEq

/-- A signature reference -/
structure SigRef where
  id : Nat
  deriving Repr, BEq

/-- An FFI function reference -/
structure FnRef where
  id : Nat
  deriving Repr, BEq

/-- Comparison condition codes -/
inductive ICmpCond where
  | eq | ne | uge | ugt | ule | ult | slt | sle | sgt | sge
  deriving Repr

/-- A single CLIF instruction -/
inductive Inst where
  | iconst (dst : Val) (ty : ClifTy) (value : Int)
  | iadd (dst : Val) (a b : Val)
  | isub (dst : Val) (a b : Val)
  | imul (dst : Val) (a b : Val)
  | udiv (dst : Val) (a b : Val)
  | ineg (dst : Val) (a : Val)
  | ishl (dst : Val) (a b : Val)
  | ushr (dst : Val) (a b : Val)
  | band (dst : Val) (a b : Val)
  | bandNot (dst : Val) (a b : Val)
  | bor (dst : Val) (a b : Val)
  | bxor (dst : Val) (a b : Val)
  | ireduce32 (dst : Val) (a : Val)
  | uextend64 (dst : Val) (a : Val)
  | sextend64 (dst : Val) (a : Val)
  | store (val addr : Val)
  | istore8 (val addr : Val)
  | load (dst : Val) (loadOp : String) (addr : Val)
  | icmp (dst : Val) (cond : ICmpCond) (a b : Val)
  | select (dst : Val) (cond a b : Val)
  | call (dst : Option Val) (fn : FnRef) (args : List Val)
  | jump (target : BlockRef) (args : List Val)
  | brif (cond : Val) (thenBlk : BlockRef) (thenArgs : List Val)
         (elseBlk : BlockRef) (elseArgs : List Val)
  | ret
  -- Float / SIMD
  | fconst (dst : Val) (ty : ClifTy) (hexBits : String)
  | fadd (dst a b : Val)
  | fsub (dst a b : Val)
  | fmul (dst a b : Val)
  | fmax (dst a b : Val)
  | fmin (dst a b : Val)
  | fpromote (dst a : Val)
  | splat (dst : Val) (ty : ClifTy) (src : Val)
  | extractlane (dst : Val) (src : Val) (lane : Nat)
  | storeTyped (ty : ClifTy) (val addr : Val)
  | rawInst (s : String)
  -- Additional float / int ops
  | fneg (dst a : Val)
  | fcvtFromSint (dst : Val) (ty : ClifTy) (src : Val)
  | fcmp (dst : Val) (cond : String) (a b : Val)
  | bitcast (dst : Val) (ty : ClifTy) (src : Val)
  | ctz (dst a : Val)
  | popcnt (dst a : Val)
  | vhighBits (dst a : Val)

/-- A declared block with its parameter values -/
structure DeclaredBlock where
  ref : BlockRef
  params : List (Val × ClifTy)

/-- Access the i-th parameter value of a declared block -/
def DeclaredBlock.param (blk : DeclaredBlock) (i : Nat) : Val :=
  match blk.params[i]? with
  | some (v, _) => v
  | none => { id := 0 }

/-- A finalized block -/
structure BlockData where
  ref : BlockRef
  params : List (Val × ClifTy)
  insts : List Inst

/-- A signature declaration -/
structure SigDecl where
  ref : SigRef
  params : List ClifTy
  result : Option ClifTy

/-- An FFI function declaration -/
structure FnDecl where
  ref : FnRef
  name : String
  sig : SigRef
  colocated : Bool := false

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
    fns := s.fns ++ [{ ref := ref, name := name, sig := sig : FnDecl }]
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

/-- Emit a 32-bit float constant; hexBits is the IEEE 754 bit pattern, e.g. "0x00000000" for 0.0 -/
def fconst32 (hexBits : String) : IRBuilder Val := do
  let v ← freshVal; emit (.fconst v .f32 hexBits); pure v

/-- Emit a 64-bit float constant; hexBits is the IEEE 754 bit pattern, e.g. "0x0000000000000000" for 0.0 -/
def fconst64 (hexBits : String) : IRBuilder Val := do
  let v ← freshVal; emit (.fconst v .f64 hexBits); pure v

-- Common float constants (CLIF accepts decimal and C99 hex-float, e.g. "0x1.000000p-1" for 0.5)
def f32Zero : String := "0.0"
def f64Zero : String := "0.0"

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
  let v ← freshVal; emit (.load v "load.f32 notrap aligned" addr); pure v

def loadF64 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v "load.f64 notrap aligned" addr); pure v

def loadF32x4 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v "load.f32x4 notrap aligned" addr); pure v

def loadI8x16 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v "load.i8x16 notrap aligned" addr); pure v

def storeF32 (val addr : Val) : IRBuilder Unit :=
  emit (.storeTyped .f32 val addr)

def storeF64 (val addr : Val) : IRBuilder Unit :=
  emit (.storeTyped .f64 val addr)

def storeI64 (val addr : Val) : IRBuilder Unit :=
  emit (.storeTyped .i64 val addr)

def storeI32 (val addr : Val) : IRBuilder Unit :=
  emit (.storeTyped .i32 val addr)

def rawInst (s : String) : IRBuilder Unit :=
  emit (.rawInst s)

def iconst8 (value : Int) : IRBuilder Val := do
  let v ← freshVal; emit (.iconst v .i8 value); pure v

def fneg (a : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fneg v a); pure v

def fcvtFromSint (ty : ClifTy) (src : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fcvtFromSint v ty src); pure v

def fcmpGt (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.fcmp v "gt" a b); pure v

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
  let v ← freshVal; emit (.load v "load.i64" addr); pure v

def load32 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v "load.i32" addr); pure v

def uload8_64 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v "uload8.i64" addr); pure v

def uload32_64 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v "uload32.i64" addr); pure v

def sload8_64 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v "sload8.i64" addr); pure v

def load_i8 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v "load.i8" addr); pure v

def load_i16 (addr : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.load v "load.i16" addr); pure v

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
-- String renderer
-- ---------------------------------------------------------------------------

def renderClifTy : ClifTy → String
  | .i8  => "i8"
  | .i32 => "i32"
  | .i64 => "i64"
  | .f32 => "f32"
  | .f64 => "f64"
  | .f32x4 => "f32x4"
  | .i8x16 => "i8x16"

def renderVal (v : Val) : String := s!"v{v.id}"

def renderBlockRef (b : BlockRef) : String := s!"block{b.id}"

def renderICmpCond : ICmpCond → String
  | .eq => "eq"
  | .ne => "ne"
  | .uge => "uge"
  | .ugt => "ugt"
  | .ule => "ule"
  | .ult => "ult"
  | .slt => "slt"
  | .sle => "sle"
  | .sgt => "sgt"
  | .sge => "sge"

def renderArgs (vals : List Val) : String :=
  String.intercalate ", " (vals.map renderVal)

def renderInst : Inst → String
  | .iconst dst ty val =>
    s!"    {renderVal dst} = iconst.{renderClifTy ty} {val}"
  | .iadd dst a b =>
    s!"    {renderVal dst} = iadd {renderVal a}, {renderVal b}"
  | .isub dst a b =>
    s!"    {renderVal dst} = isub {renderVal a}, {renderVal b}"
  | .imul dst a b =>
    s!"    {renderVal dst} = imul {renderVal a}, {renderVal b}"
  | .udiv dst a b =>
    s!"    {renderVal dst} = udiv {renderVal a}, {renderVal b}"
  | .ineg dst a =>
    s!"    {renderVal dst} = ineg {renderVal a}"
  | .ishl dst a b =>
    s!"    {renderVal dst} = ishl {renderVal a}, {renderVal b}"
  | .ushr dst a b =>
    s!"    {renderVal dst} = ushr {renderVal a}, {renderVal b}"
  | .band dst a b =>
    s!"    {renderVal dst} = band {renderVal a}, {renderVal b}"
  | .bandNot dst a b =>
    s!"    {renderVal dst} = band_not {renderVal a}, {renderVal b}"
  | .bor dst a b =>
    s!"    {renderVal dst} = bor {renderVal a}, {renderVal b}"
  | .bxor dst a b =>
    s!"    {renderVal dst} = bxor {renderVal a}, {renderVal b}"
  | .ireduce32 dst a =>
    s!"    {renderVal dst} = ireduce.i32 {renderVal a}"
  | .uextend64 dst a =>
    s!"    {renderVal dst} = uextend.i64 {renderVal a}"
  | .sextend64 dst a =>
    s!"    {renderVal dst} = sextend.i64 {renderVal a}"
  | .store val addr =>
    s!"    store {renderVal val}, {renderVal addr}"
  | .istore8 val addr =>
    s!"    istore8 {renderVal val}, {renderVal addr}"
  | .load dst loadOp addr =>
    s!"    {renderVal dst} = {loadOp} {renderVal addr}"
  | .icmp dst cond a b =>
    s!"    {renderVal dst} = icmp {renderICmpCond cond} {renderVal a}, {renderVal b}"
  | .select dst cond a b =>
    s!"    {renderVal dst} = select {renderVal cond}, {renderVal a}, {renderVal b}"
  | .call dst fn args =>
    let argStr := renderArgs args
    match dst with
    | some v => s!"    {renderVal v} = call fn{fn.id}({argStr})"
    | none => s!"    call fn{fn.id}({argStr})"
  | .jump target args =>
    if args.isEmpty then s!"    jump {renderBlockRef target}"
    else s!"    jump {renderBlockRef target}({renderArgs args})"
  | .brif cond tb ta eb ea =>
    let tStr := if ta.isEmpty then renderBlockRef tb
                else s!"{renderBlockRef tb}({renderArgs ta})"
    let eStr := if ea.isEmpty then renderBlockRef eb
                else s!"{renderBlockRef eb}({renderArgs ea})"
    s!"    brif {renderVal cond}, {tStr}, {eStr}"
  | .ret => "    return"
  | .fconst dst ty hexBits =>
    let op := if ty == .f32 then "f32const" else "f64const"
    s!"    {renderVal dst} = {op} {hexBits}"
  | .fadd dst a b => s!"    {renderVal dst} = fadd {renderVal a}, {renderVal b}"
  | .fsub dst a b => s!"    {renderVal dst} = fsub {renderVal a}, {renderVal b}"
  | .fmul dst a b => s!"    {renderVal dst} = fmul {renderVal a}, {renderVal b}"
  | .fmax dst a b => s!"    {renderVal dst} = fmax {renderVal a}, {renderVal b}"
  | .fmin dst a b => s!"    {renderVal dst} = fmin {renderVal a}, {renderVal b}"
  | .fpromote dst a => s!"    {renderVal dst} = fpromote.f64 {renderVal a}"
  | .splat dst ty src => s!"    {renderVal dst} = splat.{renderClifTy ty} {renderVal src}"
  | .extractlane dst src lane => s!"    {renderVal dst} = extractlane {renderVal src}, {lane}"
  | .storeTyped ty val addr =>
    s!"    store.{renderClifTy ty} notrap aligned {renderVal val}, {renderVal addr}"
  | .rawInst s => s!"    {s}"
  | .fneg dst a => s!"    {renderVal dst} = fneg {renderVal a}"
  | .fcvtFromSint dst ty src =>
    s!"    {renderVal dst} = fcvt_from_sint.{renderClifTy ty} {renderVal src}"
  | .fcmp dst cond a b => s!"    {renderVal dst} = fcmp {cond} {renderVal a}, {renderVal b}"
  | .bitcast dst ty src => s!"    {renderVal dst} = bitcast.{renderClifTy ty} {renderVal src}"
  | .ctz dst a => s!"    {renderVal dst} = ctz {renderVal a}"
  | .popcnt dst a => s!"    {renderVal dst} = popcnt {renderVal a}"
  | .vhighBits dst a => s!"    {renderVal dst} = vhigh_bits.i32 {renderVal a}"

def renderSigDecl (s : SigDecl) : String :=
  let params := String.intercalate ", " (s.params.map renderClifTy)
  let retPart := match s.result with
    | some t => s!" -> {renderClifTy t}"
    | none => ""
  s!"    sig{s.ref.id} = ({params}){retPart} system_v"

def renderFnDecl (f : FnDecl) : String :=
  s!"    fn{f.ref.id} = {if f.colocated then "colocated %" else "%"}{f.name} sig{f.sig.id}"

def renderBlock (b : BlockData) : String :=
  let paramStr := if b.params.isEmpty then ""
    else "(" ++ String.intercalate ", " (b.params.map fun (v, t) =>
      s!"{renderVal v}: {renderClifTy t}") ++ ")"
  let header := s!"{renderBlockRef b.ref}{paramStr}:"
  let body := String.intercalate "\n" (b.insts.map renderInst)
  if b.insts.isEmpty then header
  else header ++ "\n" ++ body

/-- Finalize and render a CLIF function from the builder state -/
def renderFunction (funcIdx : Nat) (st : IRState) : String :=
  let sigLines := st.sigs.map renderSigDecl
  let fnLines := st.fns.map renderFnDecl
  let prologueLines := sigLines ++ fnLines
  let prologueStr := if prologueLines.isEmpty then ""
    else String.intercalate "\n" prologueLines ++ "\n\n"
  let blockStr := String.intercalate "\n" (st.blocks.map renderBlock)
  s!"function u0:{funcIdx}(i64) system_v \{\n" ++
  prologueStr ++ blockStr ++ "\n}\n"

-- ---------------------------------------------------------------------------
-- Top-level builders
-- ---------------------------------------------------------------------------

/-- Run an IR builder and produce a CLIF function string -/
def buildFunction (funcIdx : Nat) (builder : IRBuilder Unit) : String :=
  let (_, st) := builder.run {}
  -- Finalize last block if still open
  let (_, st) := finalizeCurrentBlock.run st
  renderFunction funcIdx st

/-- The standard noop function u0:0 -/
def noopFunction : String :=
  buildFunction 0 do
    let _ ← entryBlock
    ret

/-- Build a complete two-function CLIF program (noop + main) -/
def buildProgram (mainBuilder : IRBuilder Unit) : String :=
  noopFunction ++ "\n" ++ buildFunction 1 mainBuilder

/-- Build a noop function at a given function index (for multi-function programs) -/
def noopAt (funcIdx : Nat) : String :=
  buildFunction funcIdx do
    let _ ← entryBlock
    ret

/-- Declare a colocated FFI function (intra-module call, e.g. colocated %ht_create) -/
def declareColocatedFFI (name : String) (params : List ClifTy) (result : Option ClifTy) : IRBuilder FnRef := do
  let sig ← declareSig params result
  let s ← get
  let ref : FnRef := { id := s.nextFn }
  set { s with
    nextFn := s.nextFn + 1
    fns := s.fns ++ [{ ref := ref, name := name, sig := sig, colocated := true : FnDecl }]
  }
  pure ref

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
