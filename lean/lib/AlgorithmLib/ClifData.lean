import Lean

/-!
# The CLIF program an artifact carries

The instruction set the generators emit, as data, together with the JSON shape
`base_types::clif` deserializes. Separate from `IR.lean` because `Core.Setup`
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

/-- A signature reference -/
structure SigRef where
  id : Nat
  deriving Repr, BEq, Lean.ToExpr

/-- An FFI function reference -/
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
  | ret
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

/-- A signature declaration -/
structure SigDecl where
  ref : SigRef
  params : List ClifTy
  result : Option ClifTy
  deriving Lean.ToExpr

/-- What a `fn` declaration names. -/
inductive Callee where
  /-- A symbol resolved through the JIT's symbol table. -/
  | import (name : String)
  /-- Another function of this same program, by its `u0:N` index. -/
  | local (index : Nat)
  deriving Repr, BEq, DecidableEq, Lean.ToExpr

/-- A callee declaration -/
structure FnDecl where
  ref : FnRef
  callee : Callee
  sig : SigRef
  deriving Lean.ToExpr

-- ---------------------------------------------------------------------------
-- The emitted program
-- ---------------------------------------------------------------------------

/-- One function of the emitted program. -/
structure FuncData where
  index : Nat
  sigs : List SigDecl
  fns : List FnDecl
  blocks : List BlockData

-- ---------------------------------------------------------------------------
-- Serialization
--
-- The shape serde reads on the Rust side: externally-tagged enums, tuple
-- variants whose fields are in constructor order, newtypes as bare numbers.
-- Field names are the Rust ones, so a rename there is a build failure here
-- rather than a silent mismatch.
-- ---------------------------------------------------------------------------

private def tagged (tag : String) (args : List Lean.Json) : Lean.Json :=
  Lean.Json.mkObj [(tag, Lean.Json.arr args.toArray)]

private def jNat (n : Nat) : Lean.Json := Lean.Json.num (n : Int)
private def jInt (n : Int) : Lean.Json := Lean.Json.num n

instance : Lean.ToJson Val where
  toJson v := jNat v.id
instance : Lean.ToJson BlockRef where
  toJson b := jNat b.id
instance : Lean.ToJson SigRef where
  toJson s := jNat s.id
instance : Lean.ToJson FnRef where
  toJson f := jNat f.id

instance : Lean.ToJson ClifTy where
  toJson
    | .i8 => "I8" | .i16 => "I16" | .i32 => "I32" | .i64 => "I64"
    | .f32 => "F32" | .f64 => "F64" | .f32x4 => "F32x4" | .i8x16 => "I8x16"

instance : Lean.ToJson ICmpCond where
  toJson
    | .eq => "Eq" | .ne => "Ne" | .uge => "Uge" | .ugt => "Ugt" | .ule => "Ule"
    | .ult => "Ult" | .slt => "Slt" | .sle => "Sle" | .sgt => "Sgt" | .sge => "Sge"

instance : Lean.ToJson FloatCC where
  toJson
    | .eq => "Eq" | .ne => "Ne" | .lt => "Lt" | .le => "Le" | .gt => "Gt" | .ge => "Ge"

instance : Lean.ToJson LoadKind where
  toJson
    | .plain => "Plain" | .uload8 => "Uload8"
    | .uload32 => "Uload32" | .sload8 => "Sload8"

instance : Lean.ToJson LoadOp where
  toJson op := Lean.Json.mkObj
    [("kind", Lean.toJson op.kind),
     ("ty", Lean.toJson op.ty),
     ("notrap_aligned", Lean.Json.bool op.notrapAligned)]

open Lean (toJson) in
/-- One instruction, in the shape `base_types::clif::Inst` deserializes.

    Stores and loads carry a byte offset on the Rust side that no builder here
    emits yet, so it is written as zero. -/
def Inst.json : Inst → Lean.Json
  | .iconst d t v => tagged "Iconst" [toJson d, toJson t, jInt v]
  | .iadd d a b => tagged "Iadd" [toJson d, toJson a, toJson b]
  | .isub d a b => tagged "Isub" [toJson d, toJson a, toJson b]
  | .imul d a b => tagged "Imul" [toJson d, toJson a, toJson b]
  | .udiv d a b => tagged "Udiv" [toJson d, toJson a, toJson b]
  | .ineg d a => tagged "Ineg" [toJson d, toJson a]
  | .ishl d a b => tagged "Ishl" [toJson d, toJson a, toJson b]
  | .ushr d a b => tagged "Ushr" [toJson d, toJson a, toJson b]
  | .band d a b => tagged "Band" [toJson d, toJson a, toJson b]
  | .bandNot d a b => tagged "BandNot" [toJson d, toJson a, toJson b]
  | .bor d a b => tagged "Bor" [toJson d, toJson a, toJson b]
  | .bxor d a b => tagged "Bxor" [toJson d, toJson a, toJson b]
  | .ireduce32 d a => tagged "Ireduce32" [toJson d, toJson a]
  | .uextend64 d a => tagged "Uextend64" [toJson d, toJson a]
  | .sextend64 d a => tagged "Sextend64" [toJson d, toJson a]
  | .store v a => tagged "Store" [toJson v, toJson a, jNat 0]
  | .istore8 v a => tagged "Istore8" [toJson v, toJson a, jNat 0]
  | .load d op a => tagged "Load" [toJson d, toJson op, toJson a, jNat 0]
  | .icmp d c a b => tagged "Icmp" [toJson d, toJson c, toJson a, toJson b]
  | .select d c a b => tagged "Select" [toJson d, toJson c, toJson a, toJson b]
  | .call d f args =>
    tagged "Call" [match d with | some v => toJson v | none => Lean.Json.null,
                   toJson f, Lean.Json.arr ((args.map toJson).toArray)]
  | .jump t args => tagged "Jump" [toJson t, Lean.Json.arr ((args.map toJson).toArray)]
  | .brif c tb ta eb ea =>
    tagged "Brif" [toJson c, toJson tb, Lean.Json.arr ((ta.map toJson).toArray),
                   toJson eb, Lean.Json.arr ((ea.map toJson).toArray)]
  | .ret => Lean.Json.str "Ret"
  | .fconst d t bits => tagged "Fconst" [toJson d, toJson t, jNat bits.toNat]
  | .fadd d a b => tagged "Fadd" [toJson d, toJson a, toJson b]
  | .fsub d a b => tagged "Fsub" [toJson d, toJson a, toJson b]
  | .fmul d a b => tagged "Fmul" [toJson d, toJson a, toJson b]
  | .fmax d a b => tagged "Fmax" [toJson d, toJson a, toJson b]
  | .fmin d a b => tagged "Fmin" [toJson d, toJson a, toJson b]
  | .fpromote d a => tagged "Fpromote" [toJson d, toJson a]
  | .splat d t s => tagged "Splat" [toJson d, toJson t, toJson s]
  | .extractlane d s lane => tagged "Extractlane" [toJson d, toJson s, jNat lane]
  | .storeTyped t v a => tagged "StoreTyped" [toJson t, toJson v, toJson a, jNat 0]
  | .fneg d a => tagged "Fneg" [toJson d, toJson a]
  | .fcvtFromSint d t s => tagged "FcvtFromSint" [toJson d, toJson t, toJson s]
  | .fcvtToUint d t s => tagged "FcvtToUint" [toJson d, toJson t, toJson s]
  | .fcmp d c a b => tagged "Fcmp" [toJson d, toJson c, toJson a, toJson b]
  | .bitcast d t s => tagged "Bitcast" [toJson d, toJson t, toJson s]
  | .bitselect d c a b => tagged "Bitselect" [toJson d, toJson c, toJson a, toJson b]
  | .ctz d a => tagged "Ctz" [toJson d, toJson a]
  | .popcnt d a => tagged "Popcnt" [toJson d, toJson a]
  | .vhighBits d a => tagged "VhighBits" [toJson d, toJson a]

instance : Lean.ToJson Inst where
  toJson := Inst.json

instance : Lean.ToJson BlockData where
  toJson b := Lean.Json.mkObj
    [("reference", Lean.toJson b.ref),
     ("params", Lean.Json.arr ((b.params.map fun (v, t) =>
        Lean.Json.arr #[Lean.toJson v, Lean.toJson t]).toArray)),
     ("insts", Lean.Json.arr ((b.insts.map Lean.toJson).toArray))]

instance : Lean.ToJson SigDecl where
  toJson s := Lean.Json.mkObj
    [("reference", Lean.toJson s.ref),
     ("params", Lean.Json.arr ((s.params.map Lean.toJson).toArray)),
     ("result", match s.result with
        | some t => Lean.toJson t
        | none => Lean.Json.null)]

instance : Lean.ToJson Callee where
  toJson
    | .import n => Lean.Json.mkObj [("Import", Lean.Json.str n)]
    | .local i => Lean.Json.mkObj [("Local", jNat i)]

instance : Lean.ToJson FnDecl where
  toJson f := Lean.Json.mkObj
    [("reference", Lean.toJson f.ref),
     ("callee", Lean.toJson f.callee),
     ("sig", Lean.toJson f.sig)]

instance : Lean.ToJson FuncData where
  toJson f := Lean.Json.mkObj
    [("index", jNat f.index),
     ("sigs", Lean.Json.arr ((f.sigs.map Lean.toJson).toArray)),
     ("fns", Lean.Json.arr ((f.fns.map Lean.toJson).toArray)),
     ("blocks", Lean.Json.arr ((f.blocks.map Lean.toJson).toArray))]

end IR

end AlgorithmLib
