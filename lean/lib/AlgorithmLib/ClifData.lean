import Lean
import AlgorithmLib.Cbor

/-!
# The CLIF program an artifact carries

The instruction set the generators emit, as data, together with the CBOR
`base_types::clif` reads. Separate from `IR.lean` because `Core.Setup`
carries a `Program` and `IR` builds one — both need these types and neither
should import the other.
-/

namespace AlgorithmLib

namespace IR

/-- CLIF value types -/
inductive ClifTy where
  | i8 | i16 | i32 | i64
  | f32 | f64
  | f32x4 | i8x16
  deriving Repr, BEq, Lean.ToExpr

/-- An SSA value reference -/
structure Val where
  id : Nat
  deriving Repr, BEq

/-- A block reference -/
structure BlockRef where
  id : Nat
  deriving Repr, BEq

/-- A callee: its position in the function's `callees`. -/
structure FnRef where
  id : Nat
  deriving Repr, BEq, Inhabited, Lean.ToExpr

/-- Comparison condition codes -/
inductive ICmpCond where
  | eq | ne | uge | ugt | ule | ult | slt | sle | sgt | sge
  deriving Repr, BEq

/-- Float comparison conditions -/
inductive FloatCC where
  | eq | ne | lt | le | gt | ge
  deriving Repr, BEq

/-- Which load instruction, independent of the type it yields. -/
inductive LoadKind where
  | plain | uload8 | uload32 | sload8
  deriving Repr, BEq

/-- A load: what to read, as what type, under which memory flags. -/
structure LoadOp where
  kind : LoadKind := .plain
  ty : ClifTy
  /-- `notrap aligned`; the float and vector accessors set it. -/
  notrapAligned : Bool := false
  deriving Repr, BEq

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
  | load (dst : Val) (op : LoadOp) (addr : Val)
  | icmp (dst : Val) (cond : ICmpCond) (a b : Val)
  | select (dst : Val) (cond a b : Val)
  | call (dst : Option Val) (fn : FnRef) (args : List Val)
  | jump (target : BlockRef) (args : List Val)
  | brif (cond : Val) (thenBlk : BlockRef) (thenArgs : List Val)
         (elseBlk : BlockRef) (elseArgs : List Val)
  /-- `return v`, or `return` when a function answers nothing. A body whose
      `ret` carries a value is a body whose signature returns an `i64`: the
      runtime reads the signature off the body rather than being told. -/
  | ret (value : Option Val)
  -- Float / SIMD
  | fconst (dst : Val) (ty : ClifTy) (bits : UInt64)
  | fadd (dst a b : Val)
  | fsub (dst a b : Val)
  | fmul (dst a b : Val)
  | fmax (dst a b : Val)
  | fmin (dst a b : Val)
  | fpromote (dst a : Val)
  | splat (dst : Val) (ty : ClifTy) (src : Val)
  | extractlane (dst : Val) (src : Val) (lane : Nat)
  | storeTyped (ty : ClifTy) (val addr : Val)
  -- Additional float / int ops
  | fneg (dst a : Val)
  | fcvtFromSint (dst : Val) (ty : ClifTy) (src : Val)
  /-- Saturating float-to-unsigned conversion.  Saturating rather than trapping
      so an out-of-range value clamps instead of aborting the process — the
      only consumer is a token id read back from a kernel that computed it as
      an exactly-representable integer. -/
  | fcvtToUint (dst : Val) (ty : ClifTy) (src : Val)
  | fcmp (dst : Val) (cond : FloatCC) (a b : Val)
  | bitcast (dst : Val) (ty : ClifTy) (src : Val)
  /-- Lane-wise `c ? a : b` on the *bits* of `c`, which is how a vector
      comparison's all-ones/all-zeros mask is consumed. -/
  | bitselect (dst c a b : Val)
  | ctz (dst a : Val)
  | popcnt (dst a : Val)
  | vhighBits (dst a : Val)
  deriving BEq

/-- A finalized block -/
structure BlockData where
  ref : BlockRef
  params : List (Val × ClifTy)
  insts : List Inst

/-- What a call names: a host symbol, or another function of this program. -/
inductive Callee where
  /-- A symbol resolved through the JIT's symbol table. -/
  | import (name : String)
  /-- Another function of this same program, by its `u0:N` index. -/
  | local (index : Nat)
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

-- ---------------------------------------------------------------------------
-- The emitted program
-- ---------------------------------------------------------------------------

/-- One function of the emitted program.

    `index` is the `u0:N` it was compiled at, which is its position in the
    artifact and is not written out: `Prog.program` checks the two agree.
    `entryName` is the name a host calls it by, and a function without one is
    the program's own. -/
structure FuncData where
  index : Nat
  /-- What the body may call. A `call` names one by its position here, so the
      order is the numbering and there is nothing to disagree with it. -/
  callees : List Callee
  blocks : List BlockData
  entryName : Option String := none

-- ---------------------------------------------------------------------------
-- Serialization
--
-- The CBOR `base_types::clif` reads (see `Cbor`): externally tagged enums,
-- tuple variants whose fields are in constructor order, newtypes as their
-- number. Field names and their order are the Rust ones; a disagreement is a
-- build failure, because the build re-encodes every artifact it decodes.
-- ---------------------------------------------------------------------------

open Cbor

instance : ToCbor Val where
  cbor v := nat v.id
instance : ToCbor BlockRef where
  cbor b := nat b.id
instance : ToCbor FnRef where
  cbor f := nat f.id

instance : ToCbor ClifTy where
  cbor t := text <| match t with
    | .i8 => "I8" | .i16 => "I16" | .i32 => "I32" | .i64 => "I64"
    | .f32 => "F32" | .f64 => "F64" | .f32x4 => "F32x4" | .i8x16 => "I8x16"

instance : ToCbor ICmpCond where
  cbor c := text <| match c with
    | .eq => "Eq" | .ne => "Ne" | .uge => "Uge" | .ugt => "Ugt" | .ule => "Ule"
    | .ult => "Ult" | .slt => "Slt" | .sle => "Sle" | .sgt => "Sgt" | .sge => "Sge"

instance : ToCbor FloatCC where
  cbor c := text <| match c with
    | .eq => "Eq" | .ne => "Ne" | .lt => "Lt" | .le => "Le" | .gt => "Gt" | .ge => "Ge"

instance : ToCbor LoadKind where
  cbor k := text <| match k with
    | .plain => "Plain" | .uload8 => "Uload8"
    | .uload32 => "Uload32" | .sload8 => "Sload8"

instance : ToCbor LoadOp where
  cbor op := struct
    [("kind", cbor op.kind),
     ("ty", cbor op.ty),
     ("notrap_aligned", bool op.notrapAligned)]

/-- One instruction, in the shape `base_types::clif::Inst` reads.

    Stores and loads carry a byte offset on the Rust side that no builder here
    emits yet, so it is written as zero. -/
def Inst.toCbor : Inst → W Unit
  | .iconst d t v => variant "Iconst" [cbor d, cbor t, int v]
  | .iadd d a b => variant "Iadd" [cbor d, cbor a, cbor b]
  | .isub d a b => variant "Isub" [cbor d, cbor a, cbor b]
  | .imul d a b => variant "Imul" [cbor d, cbor a, cbor b]
  | .udiv d a b => variant "Udiv" [cbor d, cbor a, cbor b]
  | .ineg d a => variant "Ineg" [cbor d, cbor a]
  | .ishl d a b => variant "Ishl" [cbor d, cbor a, cbor b]
  | .ushr d a b => variant "Ushr" [cbor d, cbor a, cbor b]
  | .band d a b => variant "Band" [cbor d, cbor a, cbor b]
  | .bandNot d a b => variant "BandNot" [cbor d, cbor a, cbor b]
  | .bor d a b => variant "Bor" [cbor d, cbor a, cbor b]
  | .bxor d a b => variant "Bxor" [cbor d, cbor a, cbor b]
  | .ireduce32 d a => variant "Ireduce32" [cbor d, cbor a]
  | .uextend64 d a => variant "Uextend64" [cbor d, cbor a]
  | .sextend64 d a => variant "Sextend64" [cbor d, cbor a]
  | .store v a => variant "Store" [cbor v, cbor a, nat 0]
  | .istore8 v a => variant "Istore8" [cbor v, cbor a, nat 0]
  | .load d op a => variant "Load" [cbor d, cbor op, cbor a, nat 0]
  | .icmp d c a b => variant "Icmp" [cbor d, cbor c, cbor a, cbor b]
  | .select d c a b => variant "Select" [cbor d, cbor c, cbor a, cbor b]
  | .call d f args =>
    variant "Call" [option cbor d,
                   cbor f, array args cbor]
  | .jump t args => variant "Jump" [cbor t, array args cbor]
  | .brif c tb ta eb ea =>
    variant "Brif" [cbor c, cbor tb, array ta cbor,
                   cbor eb, array ea cbor]
  -- A newtype variant, so the payload sits directly under the tag rather than
  -- in an array the way the tuple variants above do.
  | .ret v => newtypeVariant "Ret" (option cbor v)
  | .fconst d t bits => variant "Fconst" [cbor d, cbor t, nat bits.toNat]
  | .fadd d a b => variant "Fadd" [cbor d, cbor a, cbor b]
  | .fsub d a b => variant "Fsub" [cbor d, cbor a, cbor b]
  | .fmul d a b => variant "Fmul" [cbor d, cbor a, cbor b]
  | .fmax d a b => variant "Fmax" [cbor d, cbor a, cbor b]
  | .fmin d a b => variant "Fmin" [cbor d, cbor a, cbor b]
  | .fpromote d a => variant "Fpromote" [cbor d, cbor a]
  | .splat d t s => variant "Splat" [cbor d, cbor t, cbor s]
  | .extractlane d s lane => variant "Extractlane" [cbor d, cbor s, nat lane]
  | .storeTyped t v a => variant "StoreTyped" [cbor t, cbor v, cbor a, nat 0]
  | .fneg d a => variant "Fneg" [cbor d, cbor a]
  | .fcvtFromSint d t s => variant "FcvtFromSint" [cbor d, cbor t, cbor s]
  | .fcvtToUint d t s => variant "FcvtToUint" [cbor d, cbor t, cbor s]
  | .fcmp d c a b => variant "Fcmp" [cbor d, cbor c, cbor a, cbor b]
  | .bitcast d t s => variant "Bitcast" [cbor d, cbor t, cbor s]
  | .bitselect d c a b => variant "Bitselect" [cbor d, cbor c, cbor a, cbor b]
  | .ctz d a => variant "Ctz" [cbor d, cbor a]
  | .popcnt d a => variant "Popcnt" [cbor d, cbor a]
  | .vhighBits d a => variant "VhighBits" [cbor d, cbor a]


instance : ToCbor Inst where
  cbor := Inst.toCbor

instance : ToCbor BlockData where
  cbor b := struct
    [("reference", cbor b.ref),
     ("params", array b.params fun (v, t) => do head 4 2; cbor v; cbor t),
     ("insts", array b.insts cbor)]

instance : ToCbor Callee where
  cbor
    | .import n => newtypeVariant "Import" (text n)
    | .local i => newtypeVariant "Local" (nat i)

instance : ToCbor FuncData where
  cbor f := struct
    [("entry_name", option text f.entryName),
     ("callees", array f.callees cbor),
     ("blocks", array f.blocks cbor)]

end IR

end AlgorithmLib
