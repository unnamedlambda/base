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
  initial_memory : List UInt8 := []
  deriving Repr

instance : ToJson BaseConfig where
  toJson c := Json.mkObj [
    ("cranelift_ir", toJson c.cranelift_ir),
    ("memory_size", toJson c.memory_size),
    ("context_offset", toJson c.context_offset),
    ("initial_memory", toJson c.initial_memory)
  ]

structure Algorithm where
  actions : List Action
  cranelift_units : Nat
  timeout_ms : Option Nat
  output : List Json := []

instance : ToJson Algorithm where
  toJson alg := Json.mkObj [
    ("actions", toJson alg.actions),
    ("cranelift_units", toJson alg.cranelift_units),
    ("timeout_ms", toJson alg.timeout_ms),
    ("output", Json.arr alg.output.toArray)
  ]

/-- Serialize a (BaseConfig, Algorithm) pair as a JSON tuple (array of two elements). -/
def toJsonPair (config : BaseConfig) (algorithm : Algorithm) : Json :=
  Json.arr #[toJson config, toJson algorithm]

/-- Common single-call action list used by most benchmark artifacts. -/
def mkCallActions (src : UInt32) : List Action :=
  [{ kind := .ClifCall, dst := 0, src := src, offset := 0, size := 0 }]

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
-- Memory Layout DSL (with typed field handles)
-- ===========================================================================

namespace Layout

/-- Field types with statically known sizes -/
inductive FieldTy where
  | u8                          -- 1 byte
  | i32                         -- 4 bytes
  | i64                         -- 8 bytes
  | bytes (n : Nat)             -- fixed-size byte region
  deriving Repr, BEq

/-- Size in bytes of a field type -/
def FieldTy.size : FieldTy → Nat
  | .u8 => 1
  | .i32 => 4
  | .i64 => 8
  | .bytes n => n

/-- A typed field handle. The FieldTy parameter is carried at the type level,
    so you can't accidentally use an i32 field where i64 is expected.
    The offset is computed by the layout builder — no manual hex constants. -/
structure Fld (t : FieldTy) where
  offset : Nat
  deriving Repr

/-- Type-erased field handle for payload initialization -/
structure AnyFld where
  offset : Nat
  ty : FieldTy

/-- Erase the type parameter from a field handle -/
def Fld.toAny (f : Fld t) : AnyFld := { offset := f.offset, ty := t }

/-- Layout builder state -/
structure BuilderState where
  fields : List AnyFld := []    -- accumulated fields (in order)
  cursor : Nat := 0             -- next free byte offset

/-- Layout builder monad -/
abbrev LayoutBuilder := StateM BuilderState

/-- Skip to an absolute offset. Fails silently if cursor is already past it. -/
def skipTo (offset : Nat) : LayoutBuilder Unit :=
  modify fun s => { s with cursor := max s.cursor offset }

/-- Skip forward by n bytes -/
def skip (n : Nat) : LayoutBuilder Unit :=
  modify fun s => { s with cursor := s.cursor + n }

/-- Add a field at the current cursor position. Returns a typed handle. -/
def field (ty : FieldTy) : LayoutBuilder (Fld ty) :=
  modifyGet fun s =>
    let f : Fld ty := { offset := s.cursor }
    (f, { fields := s.fields ++ [f.toAny], cursor := s.cursor + ty.size })

/-- Add a field at a specific absolute offset. Returns a typed handle. -/
def fieldAt (ty : FieldTy) (offset : Nat) : LayoutBuilder (Fld ty) :=
  modifyGet fun s =>
    let f : Fld ty := { offset }
    (f, { fields := s.fields ++ [f.toAny], cursor := max s.cursor (offset + ty.size) })

-- -------------------------------------------------------------------------
-- Layout finalization
-- -------------------------------------------------------------------------

/-- A finalized memory layout with total size -/
structure LayoutMeta where
  totalSize : Nat
  deriving Repr

/-- Run a layout builder, return both the builder's result and the layout metadata -/
def build (builder : LayoutBuilder α) : α × LayoutMeta :=
  let (a, st) := builder.run {}
  (a, { totalSize := st.cursor })

-- -------------------------------------------------------------------------
-- Payload generation
-- -------------------------------------------------------------------------

/-- A payload initializer: typed field handle + byte content -/
structure FieldInit where
  offset : Nat
  bytes : List UInt8
  deriving Inhabited

/-- Create a payload initializer from a typed field handle.
    Panics at build time if `bytes` exceeds the field's declared size. -/
def Fld.init (_f : Fld t) (bytes : List UInt8) : FieldInit :=
  if bytes.length > t.size then
    panic! s!"Fld.init: {bytes.length} bytes exceeds field size {t.size}"
  else
    { offset := _f.offset, bytes }

/-- Write bytes into a list at a given offset -/
private def writeBytesAux (buf : List UInt8) (pos : Nat) (bytes : List UInt8) (limit : Nat) : List UInt8 :=
  match bytes with
  | [] => buf
  | b :: rest =>
    if pos < limit then
      writeBytesAux (buf.set pos b) (pos + 1) rest limit
    else buf

/-- Generate a payload byte list from a total size and field initializers.
    Fields without initializers are zero-filled. -/
def mkPayload (size : Nat) (inits : List FieldInit) : List UInt8 :=
  let buf : List UInt8 := List.replicate size (0 : UInt8)
  inits.foldl (init := buf) fun acc init =>
    writeBytesAux acc init.offset init.bytes size

end Layout

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

/-- Declare cl_file_read: (ptr, fname_off, data_off, file_offset, size) -> bytes_read -/
def declareFileRead : IRBuilder FnRef :=
  declareFFI "cl_file_read" [.i64, .i64, .i64, .i64, .i64] (some .i64)

/-- Declare cl_file_write: (ptr, fname_off, src_off, file_offset, size) -> bytes_written -/
def declareFileWrite : IRBuilder FnRef :=
  declareFFI "cl_file_write" [.i64, .i64, .i64, .i64, .i64] (some .i64)

/-- Declare cl_stdin_readline: (ptr, dst_off, max_len) -> bytes_read -/
def declareStdinReadline : IRBuilder FnRef :=
  declareFFI "cl_stdin_readline" [.i64, .i64, .i64] (some .i64)

/-- Declare cl_stdout_write: (ptr, src_off, size) -> bytes_written -/
def declareStdoutWrite : IRBuilder FnRef :=
  declareFFI "cl_stdout_write" [.i64, .i64, .i64] (some .i64)

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

/-- Read a file into shared memory. Returns bytes read.
    readFile ptr fnRead filenameOff dataOff => call fnRead(ptr, filenameOff, dataOff, 0, 0) -/
def readFile (ptr : Val) (fnRead : FnRef) (filenameOff dataOff : Nat) : IRBuilder Val := do
  let fnOff ← iconst64 filenameOff
  let dOff ← iconst64 dataOff
  let zero ← iconst64 0
  call fnRead [ptr, fnOff, dOff, zero, zero]

/-- Write a region of shared memory to a file. Returns bytes written.
    writeFile ptr fnWrite filenameOff srcOff fileOffset size -/
def writeFile (ptr : Val) (fnWrite : FnRef) (filenameOff srcOff : Nat)
    (fileOffset size : Val) : IRBuilder Val := do
  let fnOff ← iconst64 filenameOff
  let sOff ← iconst64 srcOff
  call fnWrite [ptr, fnOff, sOff, fileOffset, size]

/-- Write a region of shared memory to a file starting at file offset 0. -/
def writeFile0 (ptr : Val) (fnWrite : FnRef) (filenameOff srcOff : Nat)
    (size : Val) : IRBuilder Val := do
  let zero ← iconst64 0
  writeFile ptr fnWrite filenameOff srcOff zero size

/-- Align a value up to the next multiple of 4 (wgpu COPY_BUFFER_ALIGNMENT).
    alignUp4 x = (x + 3) & ~3 -/
def alignUp4 (v : Val) : IRBuilder Val := do
  let c3 ← iconst64 3
  let sum ← iadd v c3
  let negFour ← iconst64 (-4)
  band sum negFour

/-- LMDB FFI function bundle -/
structure LmdbSetup where
  fnInit : FnRef
  fnOpen : FnRef
  fnBeginWriteTxn : FnRef
  fnPut : FnRef
  fnCommitWriteTxn : FnRef
  fnCursorScan : FnRef
  fnCleanup : FnRef

/-- Declare all 7 LMDB FFI functions -/
def declareLmdbFFI : IRBuilder LmdbSetup := do
  let fnInit ← declareFFI "cl_lmdb_init" [.i64] none
  let fnOpen ← declareFFI "cl_lmdb_open" [.i64, .i64, .i32] (some .i32)
  let fnBeginWriteTxn ← declareFFI "cl_lmdb_begin_write_txn" [.i64, .i32] (some .i32)
  let fnPut ← declareFFI "cl_lmdb_put" [.i64, .i32, .i64, .i32, .i64, .i32] (some .i32)
  let fnCommitWriteTxn ← declareFFI "cl_lmdb_commit_write_txn" [.i64, .i32] (some .i32)
  let fnCursorScan ← declareFFI "cl_lmdb_cursor_scan" [.i64, .i32, .i64, .i32, .i32, .i64] (some .i32)
  let fnCleanup ← declareFFI "cl_lmdb_cleanup" [.i64] none
  pure { fnInit, fnOpen, fnBeginWriteTxn, fnPut, fnCommitWriteTxn, fnCursorScan, fnCleanup }

-- ---------------------------------------------------------------------------
-- Layout-aware IR combinators (typed field handles)
-- ---------------------------------------------------------------------------

/-- Get a typed field's offset as an iconst64 value -/
def fldOffset (f : Layout.Fld t) : IRBuilder Val :=
  iconst64 f.offset

/-- Get a typed field's absolute address: base + offset -/
def fldAddr (base : Val) (f : Layout.Fld t) : IRBuilder Val :=
  absAddr base f.offset

/-- Proof witness that a FieldTy is a scalar (u8, i32, i64) — not a byte region.
    Carries store/load implementations so the match is done per-instance, not generically. -/
class Layout.IsScalar (t : Layout.FieldTy) where
  scalarStore : Val → Val → IRBuilder Unit
  scalarLoad  : Val → IRBuilder Val

instance : Layout.IsScalar .u8 where
  scalarStore val addr := istore8 val addr
  scalarLoad  addr     := uload8_64 addr

instance : Layout.IsScalar .i32 where
  scalarStore val addr := store val addr
  scalarLoad  addr     := uload32_64 addr

instance : Layout.IsScalar .i64 where
  scalarStore val addr := store val addr
  scalarLoad  addr     := load64 addr

/-- Store a value to a scalar field. Rejects `.bytes n` at the type level. -/
def fldStore (base : Val) (f : Layout.Fld t) [inst : Layout.IsScalar t] (val : Val) : IRBuilder Unit := do
  let addr ← fldAddr base f
  inst.scalarStore val addr

/-- Load a value from a scalar field. Rejects `.bytes n` at the type level. -/
def fldLoad (base : Val) (f : Layout.Fld t) [inst : Layout.IsScalar t] : IRBuilder Val := do
  let addr ← fldAddr base f
  inst.scalarLoad addr

/-- Store an i64 value at byte offset `i` within a `.bytes n` field.
    Requires a proof that `i + 8 ≤ n` (the 8-byte store fits). -/
def fldStoreAt (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat) (val : Val)
    (_h : i + 8 ≤ n := by omega) : IRBuilder Unit := do
  let addr ← absAddr base (f.offset + i)
  store val addr

/-- Store a u8 value at byte offset `i` within a `.bytes n` field.
    Requires a proof that `i < n` (the 1-byte store fits). -/
def fldStore8At (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat) (val : Val)
    (_h : i + 1 ≤ n := by omega) : IRBuilder Unit := do
  let addr ← absAddr base (f.offset + i)
  istore8 val addr

/-- Store an i32 value at byte offset `i` within a `.bytes n` field.
    Requires a proof that `i + 4 ≤ n` (the 4-byte store fits). -/
def fldStore32At (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat) (val : Val)
    (_h : i + 4 ≤ n := by omega) : IRBuilder Unit := do
  let addr ← absAddr base (f.offset + i)
  store val addr

/-- Load an i64 value from byte offset `i` within a `.bytes n` field.
    Requires a proof that `i + 8 ≤ n`. -/
def fldLoadAt (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 8 ≤ n := by omega) : IRBuilder Val := do
  let addr ← absAddr base (f.offset + i)
  load64 addr

/-- Load a u8 value (zero-extended to i64) from byte offset `i` within a `.bytes n` field.
    Requires a proof that `i < n`. -/
def fldLoad8At (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 1 ≤ n := by omega) : IRBuilder Val := do
  let addr ← absAddr base (f.offset + i)
  uload8_64 addr

/-- Load an i32 value (zero-extended to i64) from byte offset `i` within a `.bytes n` field.
    Requires a proof that `i + 4 ≤ n`. -/
def fldLoad32At (base : Val) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 4 ≤ n := by omega) : IRBuilder Val := do
  let addr ← absAddr base (f.offset + i)
  uload32_64 addr

/-- CUDA FFI function bundle -/
structure CudaSetup where
  fnInit : FnRef
  fnCreateBuffer : FnRef
  fnUpload : FnRef
  fnDownload : FnRef
  fnFreeBuffer : FnRef
  fnLaunch : FnRef
  fnCleanup : FnRef

/-- Declare all 7 CUDA FFI functions.
    `cl_cuda_launch` takes: ptr, kernel_off, n_bufs, bind_off,
    grid_x, grid_y, grid_z, block_x, block_y, block_z → i32 -/
def declareCudaFFI : IRBuilder CudaSetup := do
  let fnInit ← declareFFI "cl_cuda_init" [.i64] none
  let fnCreateBuffer ← declareFFI "cl_cuda_create_buffer" [.i64, .i64] (some .i32)
  let fnUpload ← declareFFI "cl_cuda_upload" [.i64, .i32, .i64, .i64] (some .i32)
  let fnDownload ← declareFFI "cl_cuda_download" [.i64, .i32, .i64, .i64] (some .i32)
  let fnFreeBuffer ← declareFFI "cl_cuda_free_buffer" [.i64, .i32] (some .i32)
  let fnLaunch ← declareFFI "cl_cuda_launch"
    [.i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  let fnCleanup ← declareFFI "cl_cuda_cleanup" [.i64] none
  pure { fnInit, fnCreateBuffer, fnUpload, fnDownload, fnFreeBuffer, fnLaunch, fnCleanup }

/-- Read a file using typed field handles for filename and data regions -/
def fldReadFile (ptr : Val) (fnRead : FnRef)
    (filenameFld : Layout.Fld ft) (dataFld : Layout.Fld dt) : IRBuilder Val :=
  readFile ptr fnRead filenameFld.offset dataFld.offset

/-- Write a file using typed field handles for filename and source regions -/
def fldWriteFile0 (ptr : Val) (fnWrite : FnRef)
    (filenameFld : Layout.Fld ft) (srcFld : Layout.Fld st) (size : Val) : IRBuilder Val :=
  writeFile0 ptr fnWrite filenameFld.offset srcFld.offset size

/-- The standard ClifCall action used by all applications -/
def clifCallAction : Action := {
  kind := .ClifCall, dst := u32 0, src := u32 1, offset := u32 0, size := u32 0
}

end IR

-- ===========================================================================
-- CUDA tensor front-end DSL
-- ===========================================================================

namespace CudaTensor

inductive Expr : Nat → Type where
  | input : Fin n → Expr n
  | const : String → Expr n
  | add : Expr n → Expr n → Expr n
  | mul : Expr n → Expr n → Expr n

structure PersistentKernel (inputs : Nat) where
  expr : Expr inputs
  output : Fin inputs
  blockSize : Nat := 256
  timeoutMs : Nat := 30000
  ptxSourceOff : Nat := 0x0100
  bindDescOff : Nat := 0x1400

structure CompileResult where
  config : BaseConfig
  loadAlgorithm : Algorithm
  prepAlgorithm : Algorithm
  inferAlgorithm : Algorithm

instance : ToJson CompileResult where
  toJson r := Json.arr #[
    toJson r.config,
    toJson r.loadAlgorithm,
    toJson r.prepAlgorithm,
    toJson r.inferAlgorithm
  ]

def input {n : Nat} (i : Fin n) : Expr n := .input i
def const {n : Nat} (bits : String) : Expr n := .const bits
def add {n : Nat} (a b : Expr n) : Expr n := .add a b
def mul {n : Nat} (a b : Expr n) : Expr n := .mul a b

scoped infixl:65 " + " => add
scoped infixl:70 " * " => mul

private def paramName (i : Nat) : String :=
  s!"in{i}_ptr"

private def ptrReg (i : Nat) : String :=
  s!"%rd{1 + i}"

private structure EmitState where
  nextAddr : Nat := 10
  nextF : Nat := 0
  lines : List String := []

private def emitLine (line : String) : StateM EmitState Unit :=
  modify fun s => { s with lines := s.lines ++ [line] }

private def freshAddr : StateM EmitState String := do
  let s ← get
  let reg := s!"%rd{s.nextAddr}"
  set { s with nextAddr := s.nextAddr + 1 }
  pure reg

private def freshF : StateM EmitState String := do
  let s ← get
  let reg := s!"%f{s.nextF}"
  set { s with nextF := s.nextF + 1 }
  pure reg

private partial def emitExpr {n : Nat} (expr : Expr n) (offReg : String) : StateM EmitState String := do
  match expr with
  | .input idx =>
      let addr ← freshAddr
      let freg ← freshF
      emitLine s!"    add.u64 {addr}, {ptrReg idx.val}, {offReg};"
      emitLine s!"    ld.global.f32 {freg}, [{addr}];"
      pure freg
  | .const bits =>
      let freg ← freshF
      emitLine s!"    mov.f32 {freg}, {bits};"
      pure freg
  | .add a b =>
      let fa ← emitExpr a offReg
      let fb ← emitExpr b offReg
      let freg ← freshF
      emitLine s!"    add.f32 {freg}, {fa}, {fb};"
      pure freg
  | .mul a b =>
      let fa ← emitExpr a offReg
      let fb ← emitExpr b offReg
      let freg ← freshF
      emitLine s!"    mul.f32 {freg}, {fa}, {fb};"
      pure freg

private def ptxParams (inputs : Nat) : String :=
  let metaParam := "    .param .u64 meta_ptr"
  let ins := List.range inputs |>.map (fun i => s!"    .param .u64 {paramName i}")
  String.intercalate ",\n" (metaParam :: ins)

private def ptxLoadParams (inputs : Nat) : List String :=
  let metaLoad := "    ld.param.u64 %rd0, [meta_ptr];"
  let ins := List.range inputs |>.map (fun i => s!"    ld.param.u64 {ptrReg i}, [{paramName i}];")
  metaLoad :: ins

private def ptxSource {inputs : Nat} (spec : PersistentKernel inputs) : String :=
  let (resultF, exprState) := (emitExpr spec.expr "%rd9").run {}
  let outAddrReg := s!"%rd{exprState.nextAddr}"
  let outPtr := ptrReg spec.output.val
  let preLines := [
    ".version 8.0",
    ".target sm_86",
    ".address_size 64",
    "",
    ".visible .entry main(",
    ptxParams inputs,
    ")",
    "{",
    "    .reg .pred %p;",
    "    .reg .u32 %r<4>;",
    "    .reg .u64 %rd<32>;",
    "    .reg .f32 %f<32>;",
    ""
  ]
  let bodyLines :=
    ptxLoadParams inputs ++
    [
      "    ld.global.u32 %r0, [%rd0];",
      "    mov.u32 %r1, %ctaid.x;",
      "    mov.u32 %r2, %tid.x;",
      s!"    mad.lo.u32 %r1, %r1, {spec.blockSize}, %r2;",
      "    setp.ge.u32 %p, %r1, %r0;",
      "    @%p bra DONE;",
      "    cvt.u64.u32 %rd8, %r1;",
      "    shl.b64 %rd9, %rd8, 2;"
    ] ++
    exprState.lines ++
    [
      s!"    add.u64 {outAddrReg}, {outPtr}, %rd9;",
      s!"    st.global.f32 [{outAddrReg}], {resultF};",
      "DONE:",
      "    ret;",
      "}"
    ]
  String.intercalate "\n" (preLines ++ bodyLines) ++ "\n"

private def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "  return\n" ++
  "}\n"

private def clifLoadFn (inputs : Nat) : String :=
  let mkCreates :=
    String.intercalate "" <|
      (List.range inputs).map fun i =>
        let callIx := 21 + i
        let off := 0x34 + (4 * i)
        s!"  v{callIx} = call fn1(v0, v11)\n" ++
        s!"  store notrap aligned v{callIx}, v0+{off}\n"
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++
  "    sig1 = (i64, i64) -> i32 system_v\n" ++
  "    sig2 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_init sig0\n" ++
  "    fn1 = %cl_cuda_create_buffer sig1\n" ++
  "    fn2 = %cl_cuda_upload sig2\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  call fn0(v0)\n" ++
  "  v10 = load.i64 notrap aligned v1\n" ++
  "  store notrap aligned v10, v0+0x28\n" ++
  "  v11 = ishl_imm v10, 2\n" ++
  "  v12 = iconst.i64 8\n" ++
  "  v20 = call fn1(v0, v12)\n" ++
  "  store notrap aligned v20, v0+0x30\n" ++
  mkCreates ++
  "  v40 = iconst.i64 0x28\n" ++
  "  v41 = call fn2(v0, v20, v40, v12)\n" ++
  "  return\n" ++
  "}\n"

private def clifPrepFn (inputs : Nat) : String :=
  let uploads :=
    String.intercalate "" <|
      (List.range inputs).map fun i =>
        let bufOff := 0x34 + (4 * i)
        let bufVar := 10 + (3 * i)
        let ptrVar := 11 + (3 * i)
        let callVar := 12 + (3 * i)
        if i == 0 then
          s!"  v{bufVar} = load.i32 notrap aligned v0+{bufOff}\n" ++
          s!"  v{callVar} = call fn0(v0, v{bufVar}, v1, v5)\n"
        else
          let prevPtr := if i == 1 then 1 else 11 + (3 * (i - 1))
          s!"  v{bufVar} = load.i32 notrap aligned v0+{bufOff}\n" ++
          s!"  v{ptrVar} = iadd v{prevPtr}, v5\n" ++
          s!"  v{callVar} = call fn0(v0, v{bufVar}, v{ptrVar}, v5)\n"
  "function u0:2(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_upload_ptr sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x08\n" ++
  "  v2 = load.i64 notrap aligned v0+0x28\n" ++
  "  v5 = ishl_imm v2, 2\n" ++
  uploads ++
  "  return\n" ++
  "}\n"

private def clifInferFn (inputs : Nat) (spec : PersistentKernel inputs) : String :=
  let outBufOff := 0x34 + (4 * spec.output.val)
  "function u0:3(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    sig1 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v\n" ++
  "    sig2 = (i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_download_ptr sig0\n" ++
  "    fn1 = %cl_cuda_launch sig1\n" ++
  "    fn2 = %cl_cuda_sync sig2\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x18\n" ++
  "  v2 = load.i64 notrap aligned v0+0x20\n" ++
  "  v3 = load.i64 notrap aligned v0+0x28\n" ++
  s!"  v4 = iadd_imm v3, {spec.blockSize - 1}\n" ++
  "  v5 = ushr_imm v4, 8\n" ++
  "  v6 = ireduce.i32 v5\n" ++
  s!"  v7 = iconst.i64 {spec.ptxSourceOff}\n" ++
  s!"  v8 = iconst.i32 {inputs + 1}\n" ++
  s!"  v9 = iconst.i64 {spec.bindDescOff}\n" ++
  "  v10 = iconst.i32 1\n" ++
  s!"  v11 = iconst.i32 {spec.blockSize}\n" ++
  "  v12 = call fn1(v0, v7, v8, v9, v6, v10, v10, v11, v10, v10)\n" ++
  "  v13 = call fn2(v0)\n" ++
  "  v14 = iconst.i64 0\n" ++
  "  v15 = icmp eq v2, v14\n" ++
  "  brif v15, block1, block2\n" ++
  "block1:\n" ++
  "  return\n" ++
  "block2:\n" ++
  s!"  v16 = load.i32 notrap aligned v0+{outBufOff}\n" ++
  "  v17 = call fn0(v0, v16, v1, v2)\n" ++
  "  return\n" ++
  "}\n"

def compile {inputs : Nat} (spec : PersistentKernel inputs) : CompileResult :=
  let ptx := ptxSource spec
  let ptxBytes := ptx.toUTF8.toList ++ [0]
  let bindDesc := ((List.range (inputs + 1)).map fun i => uint32ToBytes (UInt32.ofNat i)).foldr (· ++ ·) []
  let memSize := spec.bindDescOff + bindDesc.length + 0x100
  let buildInitialMemory : List UInt8 :=
    let reserved := zeros spec.ptxSourceOff
    let ptxBlock := ptxBytes ++ zeros (spec.bindDescOff - spec.ptxSourceOff - ptxBytes.length)
    let bind := bindDesc ++ zeros (memSize - spec.bindDescOff - bindDesc.length)
    reserved ++ ptxBlock ++ bind
  let mkAlg (src : UInt32) : Algorithm := {
    actions := mkCallActions src
    cranelift_units := 0
    timeout_ms := some spec.timeoutMs
  }
  {
    config := {
      cranelift_ir := clifNoopFn ++ "\n" ++ clifLoadFn inputs ++ "\n" ++ clifPrepFn inputs ++ "\n" ++ clifInferFn inputs spec
      memory_size := memSize
      context_offset := 0
      initial_memory := buildInitialMemory
    }
    loadAlgorithm := mkAlg 1
    prepAlgorithm := mkAlg 2
    inferAlgorithm := mkAlg 3
  }

end CudaTensor

end AlgorithmLib
