import Lean
import Std

open Lean

namespace AlgorithmLib

instance : ToJson UInt8 where
  toJson n := toJson n.toNat

instance : ToJson (List UInt8) where
  toJson lst := toJson (lst.map (·.toNat))

instance : ToJson UInt32 where
  toJson n := toJson n.toNat

instance : ToJson UInt64 where
  toJson n := toJson n.toNat

inductive Kind where
  | Describe
  | ClifCall
  | ConditionalJump
  | ClifCallAsync
  | Wait
  | WaitUntil
  | Park
  | Wake
  deriving Repr

instance : ToJson Kind where
  toJson
    | .Describe => "describe"
    | .ClifCall => "clif_call"
    | .ConditionalJump => "conditional_jump"
    | .ClifCallAsync => "clif_call_async"
    | .Wait => "wait"
    | .WaitUntil => "wait_until"
    | .Park => "park"
    | .Wake => "wake"

structure Action where
  kind : Kind
  dst : UInt32
  src : UInt32
  offset : UInt32
  size : UInt32
  deriving Repr

instance : ToJson Action where
  toJson a := Json.mkObj [
    ("kind", toJson a.kind),
    ("dst", toJson a.dst),
    ("src", toJson a.src),
    ("offset", toJson a.offset),
    ("size", toJson a.size)
  ]

structure BaseConfig where
  cranelift_ir : String
  memory_size : Nat
  context_offset : Nat
  deriving Repr

instance : ToJson BaseConfig where
  toJson c := Json.mkObj [
    ("cranelift_ir", toJson c.cranelift_ir),
    ("memory_size", toJson c.memory_size),
    ("context_offset", toJson c.context_offset)
  ]

structure Algorithm where
  actions : List Action
  payloads : List UInt8
  cranelift_units : Nat
  timeout_ms : Option Nat
  output : List Json := []

instance : ToJson Algorithm where
  toJson alg := Json.mkObj [
    ("actions", toJson alg.actions),
    ("payloads", toJson alg.payloads),
    ("cranelift_units", toJson alg.cranelift_units),
    ("timeout_ms", toJson alg.timeout_ms),
    ("output", Json.arr alg.output.toArray)
  ]

/-- Serialize a (BaseConfig, Algorithm) pair as a JSON tuple (array of two elements). -/
def toJsonPair (config : BaseConfig) (algorithm : Algorithm) : Json :=
  Json.arr #[toJson config, toJson algorithm]

def u32 (n : Nat) : UInt32 := UInt32.ofNat n

def stringToBytes (s : String) : List UInt8 :=
  s.toUTF8.toList ++ [0]

def padTo (bytes : List UInt8) (targetLen : Nat) : List UInt8 :=
  bytes ++ List.replicate (targetLen - bytes.length) 0

def zeros (n : Nat) : List UInt8 :=
  List.replicate n 0

def uint32ToBytes (n : UInt32) : List UInt8 :=
  let b0 := UInt8.ofNat (n.toNat &&& 0xFF)
  let b1 := UInt8.ofNat ((n.toNat >>> 8) &&& 0xFF)
  let b2 := UInt8.ofNat ((n.toNat >>> 16) &&& 0xFF)
  let b3 := UInt8.ofNat ((n.toNat >>> 24) &&& 0xFF)
  [b0, b1, b2, b3]

def uint64ToBytes (n : UInt64) : List UInt8 :=
  let b0 := UInt8.ofNat (n.toNat &&& 0xFF)
  let b1 := UInt8.ofNat ((n.toNat >>> 8) &&& 0xFF)
  let b2 := UInt8.ofNat ((n.toNat >>> 16) &&& 0xFF)
  let b3 := UInt8.ofNat ((n.toNat >>> 24) &&& 0xFF)
  let b4 := UInt8.ofNat ((n.toNat >>> 32) &&& 0xFF)
  let b5 := UInt8.ofNat ((n.toNat >>> 40) &&& 0xFF)
  let b6 := UInt8.ofNat ((n.toNat >>> 48) &&& 0xFF)
  let b7 := UInt8.ofNat ((n.toNat >>> 56) &&& 0xFF)
  [b0, b1, b2, b3, b4, b5, b6, b7]

def int32ToBytes (n : Int) : List UInt8 :=
  let two32 : Int := Int.ofNat ((2:Nat) ^ 32)
  let u : UInt32 :=
    if n >= 0 then
      UInt32.ofNat n.toNat
    else
      let m : Int := n + two32
      UInt32.ofNat m.toNat
  uint32ToBytes u

-- ===========================================================================
-- CLIF IR Builder DSL
-- ===========================================================================

namespace IR

/-- CLIF value types -/
inductive ClifTy where
  | i32
  | i64
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
  | eq | ne | uge | ugt | ult | slt | sgt
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

def bandNot (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.bandNot v a b); pure v

def bor (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.bor v a b); pure v

def bxor (a b : Val) : IRBuilder Val := do
  let v ← freshVal; emit (.bxor v a b); pure v

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
-- String renderer
-- ---------------------------------------------------------------------------

def renderClifTy : ClifTy → String
  | .i32 => "i32"
  | .i64 => "i64"

def renderVal (v : Val) : String := s!"v{v.id}"

def renderBlockRef (b : BlockRef) : String := s!"block{b.id}"

def renderICmpCond : ICmpCond → String
  | .eq => "eq"
  | .ne => "ne"
  | .uge => "uge"
  | .ugt => "ugt"
  | .ult => "ult"
  | .slt => "slt"
  | .sgt => "sgt"

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

def renderSigDecl (s : SigDecl) : String :=
  let params := String.intercalate ", " (s.params.map renderClifTy)
  let retPart := match s.result with
    | some t => s!" -> {renderClifTy t}"
    | none => ""
  s!"    sig{s.ref.id} = ({params}){retPart} system_v"

def renderFnDecl (f : FnDecl) : String :=
  s!"    fn{f.ref.id} = %{f.name} sig{f.sig.id}"

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

/-- Load i64 from base + offset -/
def load64At (base : Val) (offset : Nat) : IRBuilder Val := do
  let addr ← absAddr base offset
  load64 addr

/-- Load i32 from base + offset -/
def load32At (base : Val) (offset : Nat) : IRBuilder Val := do
  let addr ← absAddr base offset
  load32 addr

/-- Declare cl_file_read: (ptr, fname_off, data_off, file_offset, size) -> bytes_read -/
def declareFileRead : IRBuilder FnRef :=
  declareFFI "cl_file_read" [.i64, .i64, .i64, .i64, .i64] (some .i64)

/-- Declare cl_file_write: (ptr, fname_off, src_off, file_offset, size) -> bytes_written -/
def declareFileWrite : IRBuilder FnRef :=
  declareFFI "cl_file_write" [.i64, .i64, .i64, .i64, .i64] (some .i64)

/-- GPU FFI function bundle -/
structure GpuSetup where
  fnInit : FnRef
  fnCreateBuffer : FnRef
  fnUpload : FnRef
  fnDownload : FnRef
  fnCreatePipeline : FnRef
  fnDispatch : FnRef
  fnCleanup : FnRef

/-- Declare all 7 GPU FFI functions -/
def declareGpuFFI : IRBuilder GpuSetup := do
  let fnInit ← declareFFI "cl_gpu_init" [.i64] none
  let fnCreateBuffer ← declareFFI "cl_gpu_create_buffer" [.i64, .i64] (some .i32)
  let fnCreatePipeline ← declareFFI "cl_gpu_create_pipeline" [.i64, .i64, .i64, .i32] (some .i32)
  let fnUpload ← declareFFI "cl_gpu_upload" [.i64, .i32, .i64, .i64] (some .i32)
  let fnDownload ← declareFFI "cl_gpu_download" [.i64, .i32, .i64, .i64] (some .i32)
  let fnDispatch ← declareFFI "cl_gpu_dispatch" [.i64, .i32, .i32, .i32, .i32] (some .i32)
  let fnCleanup ← declareFFI "cl_gpu_cleanup" [.i64] none
  pure { fnInit, fnCreateBuffer, fnUpload, fnDownload, fnCreatePipeline, fnDispatch, fnCleanup }

/-- Build a counting for-loop: for (i = init; i < limit; i += step) { body i }.
    Returns the declared exit block so the caller can continue after the loop. -/
def forLoop (init limit step : Val) (body : Val → IRBuilder Unit) : IRBuilder DeclaredBlock := do
  let hdrBlk ← declareBlock [.i64]
  let i := hdrBlk.param 0
  let exitBlk ← declareBlock []
  let bodyBlk ← declareBlock []
  jump hdrBlk.ref [init]
  startBlock hdrBlk
  let done ← icmp .uge i limit
  brif done exitBlk.ref [] bodyBlk.ref []
  startBlock bodyBlk
  body i
  let next ← iadd i step
  jump hdrBlk.ref [next]
  startBlock exitBlk
  pure exitBlk

end IR

end AlgorithmLib
