module
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # `evalOp` answers exactly when `Op.check` accepts

  `evalOp`'s own docstring states this — "`none` exactly when `Op.check` would
  reject it or an address is unmapped" — and nothing tested it. The differential
  corpus cannot: `HProgCorpus` writes its cases as `Prog`s, whose types admit
  no ill-typed operation, so by construction it only ever produces well-typed
  terms. The one claim it structurally cannot reach is this one.

  Both directions are checked, and both had failures. An evaluator more defined
  than the checker answers for a program Cranelift will not build, which is how
  `stepPure` and this semantics came to disagree on a mixed-width `iadd`. An
  evaluator less defined gets stuck on a body `wf` accepts, which makes a
  theorem about that body vacuous rather than wrong.

  The type environment is *derived* from the value environment, so the statement
  carries no well-typedness hypothesis: whatever the values are, the checker is
  asked about exactly their types.
-/

namespace AlgorithmLib.HProg.Conform

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

/-- The types a value environment assigns, which is what `Op.check` is asked
    about. -/
def tyEnvOf (Γ : Sem.Env) : TyEnv := TyEnv.ofList (Γ.toList.map V.ty)

/-- Every type an operand can have. -/
def tys : List ClifTy := [.i8, .i16, .i32, .i64, .f32, .f64, .f32x4, .i8x16]

/-- One value per type. Vectors carry their declared lane count so `V.ty` does
    not lie, and an `i64` is a mapped address so that a `load` reporting `none`
    means the checker disagreed rather than that nothing was there. -/
def valOf (t : ClifTy) : V :=
  match t with
  | .f32x4 => .vec .f32x4 #[1, 2, 3, 4]
  | .i8x16 => .vec .i8x16 (Array.replicate 16 7)
  | .i64   => .sc .i64 (regionBase .arena)
  | _      => .sc t 3

/-- A region with room for any load the check admits. -/
def mem0 : Mem :=
  { arena := ⟨Array.replicate 256 0⟩, data := ByteArray.empty, out := ByteArray.empty }

/-- Whether the two agree at one operand typing: both refuse, or both answer
    and at the type the checker predicted. -/
def agrees (ats : List ClifTy) (o : Op) : Bool :=
  let Γ : Sem.Env := (ats.map valOf).toArray
  match Op.check (tyEnvOf Γ) o, (evalOp mem0 Γ o).map V.ty with
  | none,   none   => true
  | some c, some g => c == g
  | _,      _      => false

/-- Whether both sides answered, so the space can be shown to hold cases that
    exercise the agreement rather than only cases both refuse. -/
def bothAnswer (ats : List ClifTy) (o : Op) : Bool :=
  let Γ : Sem.Env := (ats.map valOf).toArray
  (Op.check (tyEnvOf Γ) o).isSome && (evalOp mem0 Γ o).isSome

def unOps : List (R → Op) :=
  [ (.ineg ·), (.ctz ·), (.popcnt ·), (.ireduce32 ·), (.uextend64 ·), (.sextend64 ·)
  , (.fneg ·), (.fpromote ·), (.vhighBits ·), (Op.extractlane · 0)
  , (Op.fcvtFromSint .f32 ·), (Op.fcvtFromSint .f64 ·), (Op.fcvtToUint .i32 ·)
  , (Op.splat .f32x4 ·), (Op.splat .i8x16 ·), (Op.bitcast .i64 ·), (Op.bitcast .f32x4 ·)
  , (Op.load { ty := .i64 } ·), (Op.load { ty := .f32 } ·) ]

def binOps : List (R → R → Op) :=
  [ (.iadd · ·), (.isub · ·), (.imul · ·), (.udiv · ·), (.ishl · ·), (.ushr · ·)
  , (.band · ·), (.bandNot · ·), (.bor · ·), (.bxor · ·)
  , (Op.icmp .eq · ·), (Op.icmp .slt · ·), (Op.icmp .ult · ·)
  , (.fadd · ·), (.fsub · ·), (.fmul · ·), (.fmax · ·), (.fmin · ·)
  , (Op.fcmp .lt · ·), (Op.fcmp .eq · ·) ]

def terOps : List (R → R → R → Op) :=
  [ (.select · · ·), (.bitselect · · ·) ]

def unOk : Bool := unOps.all fun f => tys.all fun a => agrees [a] (f 0)

def binOk : Bool :=
  binOps.all fun f => tys.all fun a => tys.all fun b => agrees [a, b] (f 0 1)

def terOk : Bool :=
  terOps.all fun f => tys.all fun a => tys.all fun b => tys.all fun c =>
    agrees [a, b, c] (f 0 1 2)

/-- **The evaluator is defined exactly where the checker accepts**, over every
    operation at every operand typing, and produces the type the checker
    predicted.

    Both halves matter and both were violated. `Sem.bin` discarded the second
    operand's type; `select` compared neither of its arms; `ireduce32`,
    `ctz`, `popcnt`, `ineg` and `fcvtFromSint` accepted floats; the bitwise and
    comparison zips accepted them too. In the other direction `Op.check` allowed
    float arithmetic on `i8x16`, and `bitselect` was stuck on integer operands
    the checker admits. -/
theorem evalOp_agrees_with_check : (unOk && binOk && terOk) = true := by native_decide

def liveCount : Nat :=
  (unOps.flatMap fun f => tys.filter fun a => bothAnswer [a] (f 0)).length
  + (binOps.flatMap fun f => tys.flatMap fun a => tys.filter fun b =>
      bothAnswer [a, b] (f 0 1)).length
  + (terOps.flatMap fun f => tys.flatMap fun a => tys.flatMap fun b =>
      tys.filter fun c => bothAnswer [a, b, c] (f 0 1 2)).length

/-- **…and the agreement is carried by cases where both answer**, not only by
    cases both refuse. An `evalOp` that returned `none` everywhere would satisfy
    the theorem above and fail this. -/
theorem conformance_is_live : (150 < liveCount) = true := by native_decide

end AlgorithmLib.HProg.Conform
