import AlgorithmLib.HProg

/-!
# `HProgSem` — what an `HProg` term does

A concrete, executable semantics for the terms `HProg` defines: real bit
patterns, real memory, real floating point. Two things use it.

* **As an oracle.** The differential harness runs a term here and the artifact
  it compiles to through the real JIT, and compares the bytes. A wrong decode
  arm on the Rust side, a field-order drift between Lean and serde, or a
  Cranelift semantic change on upgrade all show up as a mismatch.
* **As the left-hand side of `compile_sound`.** Execution records an
  observation trace — the calls it makes and the stores it performs, in order —
  which is the statement the compilation proof is against.

Everything here is a pure function of the state, including the file system and
the FFI: `Sem.run` takes a `World` and returns one. That is what lets a term's
behavior be a *value* rather than an effect.

Unlike `HProg.wf`, none of this is meant to reduce in the kernel — it runs on
`ByteArray`, which the compiler makes fast and the kernel does not. Statements
about executions are proved from the equations, not by `decide`.

## What this assumes, and what checks it

`evalOp` says `iadd` wraps at the operand's width, `fmax` returns the other
operand when one is NaN, `fcvt_to_uint` saturates rather than traps, `ushr`
takes its shift amount modulo the width. **None of that is proven.** Cranelift
publishes no formal semantics to prove against; those meanings live in its
instruction documentation and its lowering rules. What is written here is a
transcription of them, and a transcription is an assumption — the same grade of
assumption as the frames in `HProgFrames`.

Two things keep that from being load-bearing in the wrong place.

*The compilation proof does not rest on it.* `compile_sound` relates a term's
execution to its compiled form's, and both sides call the same `evalOp`. It
therefore says *compiling preserves meaning*, not *the meaning is Cranelift's*.
A wrong arm here would not make that theorem false; it would make this file a
bad oracle, which is a different failure and a detectable one.

*Differential execution is what detects it.* `HProgCorpus` runs every operation
over operands chosen to separate the arms that are easy to get wrong — zero,
±1, `i64::MIN`/`MAX`, NaN, ±∞, subnormals, the saturation boundaries — in this
interpreter and through the real JIT, and compares the bytes. That check is not
circular: the Rust decoder says only *which* Cranelift instruction to emit, this
file says what that instruction *computes*, and the machine says who was right.
It is also the check that runs again on the next Cranelift upgrade.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

-- ---------------------------------------------------------------------------
-- Values
-- ---------------------------------------------------------------------------

/-- A runtime value. Integers narrower than 64 bits are held zero-extended;
    floats are held as their bit pattern, so a value is always its bytes. -/
inductive V where
  | sc  (ty : ClifTy) (bits : UInt64)
  | vec (ty : ClifTy) (lanes : Array UInt64)
  deriving Repr, BEq, Inhabited

/-- The mask that keeps a value inside its type's width. -/
def widthMask : ClifTy → UInt64
  | .i8 => 0xff
  | .i16 => 0xffff
  | .i32 | .f32 => 0xffffffff
  | _ => 0xffffffffffffffff

def norm (ty : ClifTy) (x : UInt64) : V := .sc ty (x &&& widthMask ty)

/-- The value read as a two's-complement integer of its own width. -/
def signed (ty : ClifTy) (x : UInt64) : Int :=
  let w := ty.width
  let m := x &&& widthMask ty
  if w ≥ 64 then
    if m ≥ 0x8000000000000000 then (m.toNat : Int) - (1 <<< 64) else m.toNat
  else
    let half : UInt64 := 1 <<< (UInt64.ofNat (w - 1))
    if m ≥ half then (m.toNat : Int) - (1 <<< w) else m.toNat

def ofInt (ty : ClifTy) (i : Int) : V :=
  norm ty (UInt64.ofNat (i.emod (1 <<< 64)).toNat)

/-- Branches read a condition as "any bit set", which is how `brif` behaves. -/
def isTrue : V → Bool
  | .sc _ b => b != 0
  | .vec _ ls => ls.any (· != 0)

def asBits : V → Option UInt64
  | .sc _ b => some b
  | _ => none

private def f32 (b : UInt64) : Float32 := Float32.ofBits b.toUInt32
private def f64 (b : UInt64) : Float := Float.ofBits b
private def ofF32 (x : Float32) : UInt64 := x.toBits.toUInt64
private def ofF64 (x : Float) : UInt64 := x.toBits

-- ---------------------------------------------------------------------------
-- Memory
-- ---------------------------------------------------------------------------

/-- The three regions CLIF code can address: the shared arena it is handed a
    pointer to, and the caller's input and output buffers, which it reaches
    through pointers the runtime writes into the arena. -/
inductive Region where
  | arena | data | out
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Region bases are far apart and aligned to a power of two, so `base + off`
    arithmetic stays inside its region and a stray address is detectable
    instead of silently landing somewhere valid. -/
def regionBase : Region → UInt64
  | .arena => 0x1000000000
  | .data  => 0x2000000000
  | .out   => 0x3000000000

def regionSpan : UInt64 := 0x1000000000

def addrOf (r : Region) (off : Nat) : UInt64 := regionBase r + UInt64.ofNat off

/-- Which region an address names, and how far into it. -/
def decodeAddr (a : UInt64) : Option (Region × Nat) :=
  [Region.arena, .data, .out].findSome? fun r =>
    let b := regionBase r
    if a ≥ b && a - b < regionSpan then some (r, (a - b).toNat) else none

structure Mem where
  arena : ByteArray
  data  : ByteArray
  out   : ByteArray
  deriving Inhabited

def Mem.region : Mem → Region → ByteArray
  | m, .arena => m.arena
  | m, .data => m.data
  | m, .out => m.out

def Mem.setRegion (m : Mem) : Region → ByteArray → Mem
  | .arena, b => { m with arena := b }
  | .data, b => { m with data := b }
  | .out, b => { m with out := b }

/-- `n` bytes at `a`, little-endian, or `none` if they are not all in one
    region — a load the real program would fault on. -/
def Mem.load (m : Mem) (a : UInt64) (n : Nat) : Option UInt64 := do
  let (r, off) ← decodeAddr a
  let bs := m.region r
  if off + n > bs.size then none
  else
    some <| (List.range n).foldr
      (fun i acc => (acc <<< 8) ||| (bs.get! (off + i)).toUInt64) 0

def Mem.store (m : Mem) (a : UInt64) (n : Nat) (v : UInt64) : Option Mem := do
  let (r, off) ← decodeAddr a
  let bs := m.region r
  if off + n > bs.size then none
  else
    let bs := (List.range n).foldl
      (fun (b : ByteArray) i => b.set! (off + i) (((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8))
      bs
    some (m.setRegion r bs)

/-- How many bytes a value of this type occupies. -/
def tyBytes (t : ClifTy) : Nat := t.width / 8

-- ---------------------------------------------------------------------------
-- Operations
-- ---------------------------------------------------------------------------

abbrev Env := Array V

private def get (Γ : Env) (r : R) : Option V := Γ[r]?

private def bin (Γ : Env) (a b : R) (f : ClifTy → UInt64 → UInt64 → Option V) : Option V := do
  let (.sc ta x) ← get Γ a | none
  let (.sc _ y) ← get Γ b | none
  f ta x y

/-- Elementwise on a vector pair, or on two scalars — how the float operations
    apply to both `f32`/`f64` and `f32x4`. -/
private def zipF (u v : V) (f32op : Float32 → Float32 → Float32)
    (f64op : Float → Float → Float) : Option V :=
  match u, v with
  | .sc .f32 x, .sc .f32 y => some (.sc .f32 (ofF32 (f32op (f32 x) (f32 y))))
  | .sc .f64 x, .sc .f64 y => some (.sc .f64 (ofF64 (f64op (f64 x) (f64 y))))
  | .vec .f32x4 xs, .vec .f32x4 ys =>
      if xs.size == ys.size then
        some (.vec .f32x4 (xs.zipWith (fun x y => ofF32 (f32op (f32 x) (f32 y))) ys))
      else none
  | _, _ => none

/-- Elementwise on the raw bits, for the operations that have to see a sign or
    a NaN payload that `Float` arithmetic would not preserve. -/
private def zipBits (u v : V) (f : ClifTy → UInt64 → UInt64 → UInt64) : Option V :=
  match u, v with
  | .sc t x, .sc t' y => if t == t' && t.isFloat then some (.sc t (f t x y)) else none
  | .vec t xs, .vec t' ys =>
      match t.lanes with
      | some (lane, _) =>
          if t == t' && xs.size == ys.size then
            some (.vec t (xs.zipWith (f lane) ys))
          else none
      | none => none
  | _, _ => none

private def isNaNBits (t : ClifTy) (x : UInt64) : Bool :=
  match t with
  | .f32 => (x &&& 0x7f800000) == 0x7f800000 && (x &&& 0x007fffff) != 0
  | .f64 => (x &&& 0x7ff0000000000000) == 0x7ff0000000000000 &&
            (x &&& 0x000fffffffffffff) != 0
  | _ => false

private def signBit (t : ClifTy) (x : UInt64) : Bool :=
  match t with
  | .f32 => x &&& 0x80000000 != 0
  | _ => x &&& 0x8000000000000000 != 0

private def ltBits (t : ClifTy) (x y : UInt64) : Bool :=
  match t with
  | .f32 => f32 x < f32 y
  | _ => f64 x < f64 y

private def eqBits (t : ClifTy) (x y : UInt64) : Bool :=
  match t with
  | .f32 => f32 x == f32 y
  | _ => f64 x == f64 y

/-- `fmax` (`wantMax`) and `fmin`, sharing everything but which end they take.
    NaN wins outright; otherwise equal values are split by sign, which is how
    the two zeros get ordered. -/
private def pickExtreme (wantMax : Bool) (t : ClifTy) (x y : UInt64) : UInt64 :=
  if isNaNBits t x then x
  else if isNaNBits t y then y
  else if eqBits t x y then
    (if signBit t x == wantMax then y else x)
  else if ltBits t x y then (if wantMax then y else x)
  else (if wantMax then x else y)

private def cmpInt : ICmpCond → ClifTy → UInt64 → UInt64 → Bool
  | .eq, _, x, y => x == y
  | .ne, _, x, y => x != y
  | .ult, _, x, y => x < y
  | .ule, _, x, y => x ≤ y
  | .ugt, _, x, y => x > y
  | .uge, _, x, y => x ≥ y
  | .slt, t, x, y => signed t x < signed t y
  | .sle, t, x, y => signed t x ≤ signed t y
  | .sgt, t, x, y => signed t x > signed t y
  | .sge, t, x, y => signed t x ≥ signed t y

private def cmpF32 : FloatCC → Float32 → Float32 → Bool
  | .eq, x, y => x == y | .ne, x, y => !(x == y)
  | .lt, x, y => x < y  | .le, x, y => x ≤ y
  | .gt, x, y => y < x  | .ge, x, y => y ≤ x

private def cmpF64 : FloatCC → Float → Float → Bool
  | .eq, x, y => x == y | .ne, x, y => !(x == y)
  | .lt, x, y => x < y  | .le, x, y => x ≤ y
  | .gt, x, y => y < x  | .ge, x, y => y ≤ x

private def boolV (b : Bool) : V := .sc .i8 (if b then 1 else 0)

/-- Saturating float-to-unsigned: NaN and everything below zero give zero, and
    everything above the type's range gives its maximum. -/
private def satToUint (ty : ClifTy) (x : Float) : V :=
  let hi := widthMask ty
  if x != x || x ≤ 0.0 then .sc ty 0
  else if x ≥ Float.ofNat hi.toNat then .sc ty hi
  else norm ty (UInt64.ofNat x.toUInt64.toNat)

/-- What an operation computes, given the slots in scope. `none` exactly when
    `Op.check` would reject it or an address is unmapped. -/
def evalOp (m : Mem) (Γ : Env) : Op → Option V
  | .iconst ty k => some (ofInt ty k)
  | .fconst ty b => some (.sc ty (b &&& widthMask ty))
  | .iadd a b => bin Γ a b fun t x y => some (norm t (x + y))
  | .isub a b => bin Γ a b fun t x y => some (norm t (x - y))
  | .imul a b => bin Γ a b fun t x y => some (norm t (x * y))
  | .udiv a b => bin Γ a b fun t x y => if y == 0 then none else some (norm t (x / y))
  -- The shift amount is taken modulo the *shifted operand's* width, not 64.
  | .ishl a b => bin Γ a b fun t x y => some (norm t (x <<< (y % UInt64.ofNat t.width)))
  | .ushr a b => bin Γ a b fun t x y => some (norm t ((x &&& widthMask t) >>> (y % UInt64.ofNat t.width)))
  | .band a b => bin Γ a b fun t x y => some (norm t (x &&& y))
  | .bandNot a b => bin Γ a b fun t x y => some (norm t (x &&& ~~~y))
  | .bor a b => bin Γ a b fun t x y => some (norm t (x ||| y))
  | .bxor a b => bin Γ a b fun t x y => some (norm t (x ^^^ y))
  | .ineg a => do
      let (.sc t x) ← get Γ a | none
      some (norm t (0 - x))
  | .ctz a => do
      let (.sc t x) ← get Γ a | none
      some (norm t (UInt64.ofNat (((List.range t.width).find? (fun i =>
        (x >>> UInt64.ofNat i) &&& 1 == 1)).getD t.width)))
  | .popcnt a => do
      let (.sc t x) ← get Γ a | none
      some (norm t (UInt64.ofNat (((List.range t.width).filter (fun i =>
        (x >>> UInt64.ofNat i) &&& 1 == 1)).length)))
  | .ireduce32 a => do
      let (.sc _ x) ← get Γ a | none
      some (norm .i32 x)
  | .uextend64 a => do
      let (.sc t x) ← get Γ a | none
      some (.sc .i64 (x &&& widthMask t))
  | .sextend64 a => do
      let (.sc t x) ← get Γ a | none
      some (ofInt .i64 (signed t x))
  | .icmp c a b => bin Γ a b fun t x y => some (boolV (cmpInt c t x y))
  | .select c a b => do
      let cv ← get Γ c
      if isTrue cv then get Γ a else get Γ b
  -- Bit-for-bit, per lane: the mask decides each bit, not each lane as a whole.
  -- That is what `bitselect` means and why a comparison result has to be
  -- bitcast to the operand width before it can be used here.
  | .bitselect c a b => do
      let cv ← get Γ c
      let av ← get Γ a
      let bv ← get Γ b
      let masked ← zipBits cv av (fun _ m x => m &&& x)
      let other ← zipBits cv bv (fun _ m y => (~~~m) &&& y)
      zipBits masked other (fun _ x y => x ||| y)
  | .fadd a b => do zipF (← get Γ a) (← get Γ b) (· + ·) (· + ·)
  | .fsub a b => do zipF (← get Γ a) (← get Γ b) (· - ·) (· - ·)
  | .fmul a b => do zipF (← get Γ a) (← get Γ b) (· * ·) (· * ·)
  -- `fmax`/`fmin` propagate NaN rather than returning the other operand, and
  -- they order the two zeros: `fmin (+0) (-0) = -0`, `fmax (+0) (-0) = +0`.
  -- Both were measured against the machine; see `HProgCorpus`.
  | .fmax a b => do zipBits (← get Γ a) (← get Γ b) (pickExtreme true)
  | .fmin a b => do zipBits (← get Γ a) (← get Γ b) (pickExtreme false)
  -- Negation is a sign-bit flip, so it applies to NaN too and cannot go
  -- through `Float` arithmetic, which does not preserve a NaN's sign.
  | .fneg a => do
      match ← get Γ a with
      | .sc .f32 x => some (.sc .f32 (x ^^^ 0x80000000))
      | .sc .f64 x => some (.sc .f64 (x ^^^ 0x8000000000000000))
      | .vec .f32x4 xs => some (.vec .f32x4 (xs.map (· ^^^ 0x80000000)))
      | _ => none
  | .fpromote a => do
      let (.sc .f32 x) ← get Γ a | none
      some (.sc .f64 (ofF64 (f32 x).toFloat))
  | .fcmp c a b => do
      match ← get Γ a, ← get Γ b with
      | .sc .f32 x, .sc .f32 y => some (boolV (cmpF32 c (f32 x) (f32 y)))
      | .sc .f64 x, .sc .f64 y => some (boolV (cmpF64 c (f64 x) (f64 y)))
      -- On vectors the result is a per-lane mask of all-ones or all-zeros, at
      -- the lane's own width — which is why `bitselect` consumes it directly
      -- once it has been bitcast to the operand type.
      | .vec .f32x4 xs, .vec .f32x4 ys =>
          if xs.size == ys.size then
            some (.vec .f32x4 (xs.zipWith
              (fun x y => if cmpF32 c (f32 x) (f32 y) then 0xffffffff else 0) ys))
          else none
      | _, _ => none
  | .fcvtFromSint ty a => do
      let (.sc t x) ← get Γ a | none
      let f : Float := Float.ofInt (signed t x)
      match ty with
      | .f32 => some (.sc .f32 (ofF32 f.toFloat32))
      | .f64 => some (.sc .f64 (ofF64 f))
      | _ => none
  | .fcvtToUint ty a => do
      match ← get Γ a with
      | .sc .f32 x => some (satToUint ty (f32 x).toFloat)
      | .sc .f64 x => some (satToUint ty (f64 x))
      | _ => none
  | .splat ty a => do
      let (lane, n) ← ty.lanes
      let (.sc t x) ← get Γ a | none
      if t == lane then some (.vec ty (Array.replicate n (x &&& widthMask lane))) else none
  | .extractlane a l => do
      let (.vec t xs) ← get Γ a | none
      let (lane, _) ← t.lanes
      let x ← xs[l]?
      some (.sc lane x)
  | .vhighBits a => do
      let (.vec t xs) ← get Γ a | none
      let (lane, _) ← t.lanes
      let top : UInt64 := 1 <<< UInt64.ofNat (lane.width - 1)
      some (.sc .i32 (UInt64.ofNat ((xs.zipIdx.filter (fun (x, _) => x &&& top != 0)).foldl
        (fun acc (_, i) => acc ||| (1 <<< i)) 0)))
  -- Bitcast is a reinterpretation: the bits are already what they are.
  | .bitcast ty a => do
      match ← get Γ a with
      | .sc t x => if t.width == ty.width then some (.sc ty x) else none
      | .vec t xs => if t.width == ty.width then some (.vec ty xs) else none
  | .load op a => do
      let (.sc _ addr) ← get Γ a | none
      match op.kind with
      | .uload8 => do
          let b ← m.load addr 1
          some (norm op.ty b)
      | .uload32 => do
          let b ← m.load addr 4
          some (norm op.ty b)
      | .sload8 => do
          let b ← m.load addr 1
          some (ofInt op.ty (signed .i8 b))
      | .plain =>
          match op.ty.lanes with
          | some (lane, n) => do
              let w := tyBytes lane
              let ls ← (List.range n).foldlM
                (fun acc i => do let b ← m.load (addr + UInt64.ofNat (i * w)) w; pure (acc.push b))
                (#[] : Array UInt64)
              some (.vec op.ty ls)
          | none => do
              let b ← m.load addr (tyBytes op.ty)
              some (.sc op.ty b)

-- ---------------------------------------------------------------------------
-- Relocation
-- ---------------------------------------------------------------------------

/-!
`evalOp` reaches into `Γ` only through `get`, so an operation cannot tell the
difference between the environment it was written against and a two-slot
environment holding just its operands. That is the fact the block interpreter
needs: `evalInst` evaluates `f 0 1` against `#[x, y]`, and the term evaluates
`f a b` against the whole of `Γ`.

Stated as a property of the operation rather than proved inline at each use, so
the compilation theorem can quote one name per shape instead of re-deriving it.
-/

/-- A two-operand shape reads its arguments and nothing else. -/
def Reloc2 (f : R → R → Op) : Prop :=
  ∀ (m : Mem) (Γ : Env) (a b : R) (x y : V),
    Γ[a]? = some x → Γ[b]? = some y →
    evalOp m Γ (f a b) = evalOp m #[x, y] (f 0 1)

/-- A one-operand shape reads its argument and nothing else. -/
def Reloc1 (f : R → Op) : Prop :=
  ∀ (m : Mem) (Γ : Env) (a : R) (x : V),
    Γ[a]? = some x → evalOp m Γ (f a) = evalOp m #[x] (f 0)

/-- A three-operand shape reads its arguments and nothing else. -/
def Reloc3 (f : R → R → R → Op) : Prop :=
  ∀ (m : Mem) (Γ : Env) (a b c : R) (x y z : V),
    Γ[a]? = some x → Γ[b]? = some y → Γ[c]? = some z →
    evalOp m Γ (f a b c) = evalOp m #[x, y, z] (f 0 1 2)

theorem reloc2_iadd : Reloc2 .iadd := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_isub : Reloc2 .isub := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_imul : Reloc2 .imul := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_udiv : Reloc2 .udiv := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_ishl : Reloc2 .ishl := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_ushr : Reloc2 .ushr := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_band : Reloc2 .band := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_bandNot : Reloc2 .bandNot := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_bor : Reloc2 .bor := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_bxor : Reloc2 .bxor := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_icmp (c : ICmpCond) : Reloc2 (Op.icmp c) := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]

theorem reloc2_fadd : Reloc2 .fadd := by
  intro m Γ a b x y ha hb; simp [evalOp, get, ha, hb]
theorem reloc2_fsub : Reloc2 .fsub := by
  intro m Γ a b x y ha hb; simp [evalOp, get, ha, hb]
theorem reloc2_fmul : Reloc2 .fmul := by
  intro m Γ a b x y ha hb; simp [evalOp, get, ha, hb]
theorem reloc2_fmax : Reloc2 .fmax := by
  intro m Γ a b x y ha hb; simp [evalOp, get, ha, hb]
theorem reloc2_fmin : Reloc2 .fmin := by
  intro m Γ a b x y ha hb; simp [evalOp, get, ha, hb]
theorem reloc2_fcmp (c : FloatCC) : Reloc2 (Op.fcmp c) := by
  intro m Γ a b x y ha hb; simp [evalOp, get, ha, hb]

theorem reloc1_ineg : Reloc1 .ineg := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_ctz : Reloc1 .ctz := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_popcnt : Reloc1 .popcnt := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_ireduce32 : Reloc1 .ireduce32 := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_uextend64 : Reloc1 .uextend64 := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_sextend64 : Reloc1 .sextend64 := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_fneg : Reloc1 .fneg := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_fpromote : Reloc1 .fpromote := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_vhighBits : Reloc1 .vhighBits := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_fcvtFromSint (ty : ClifTy) : Reloc1 (Op.fcvtFromSint ty) := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_fcvtToUint (ty : ClifTy) : Reloc1 (Op.fcvtToUint ty) := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_splat (ty : ClifTy) : Reloc1 (Op.splat ty) := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_bitcast (ty : ClifTy) : Reloc1 (Op.bitcast ty) := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_extractlane (l : Nat) : Reloc1 (Op.extractlane · l) := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_load (op : LoadOp) : Reloc1 (Op.load op) := by
  intro m Γ a x ha; simp [evalOp, get, ha]

theorem reloc3_select : Reloc3 .select := by
  intro m Γ a b c x y z ha hb hc; simp [evalOp, get, ha, hb, hc]

-- ---------------------------------------------------------------------------
-- The world a term runs in
-- ---------------------------------------------------------------------------

/-- The files a run may read and write, by absolute path. Modelling them as
    state is what keeps the semantics a pure function. -/
structure FS where
  files : List (String × ByteArray) := []
  deriving Inhabited

def FS.get (fs : FS) (p : String) : Option ByteArray :=
  (fs.files.find? (·.1 == p)).map (·.2)

def FS.set (fs : FS) (p : String) (b : ByteArray) : FS :=
  { files := (p, b) :: fs.files.filter (·.1 != p) }

/-- One entry of the observation trace. -/
inductive Obs where
  | call  (fn : Nat) (args : List V)
  | store (addr : UInt64) (width : Nat) (bits : UInt64)
  deriving Repr, BEq

structure World where
  mem : Mem
  fs  : FS := {}
  /-- Reversed while running; `Sem.run` hands it back in order. -/
  obs : List Obs := []
  deriving Inhabited

/-- Execution either finishes, or stops because the term did something the
    semantics does not define — an unmapped address, an undeclared callee, a
    loop that outran its budget. Stopping is never silent. -/
inductive Outcome (α : Type) where
  | ok (a : α) (w : World)
  | stuck (why : String)

instance : Inhabited (Outcome α) := ⟨.stuck "unreachable"⟩

/-- Outcomes print as their shape, not their memory. -/
instance [Repr α] : Repr (Outcome α) where
  reprPrec o _ := match o with
    | .ok a _ => "ok " ++ repr a
    | .stuck m => "stuck: " ++ m

-- ---------------------------------------------------------------------------
-- The FFI
-- ---------------------------------------------------------------------------

/-- The bytes from `a` up to the first zero, as a path. -/
def readCStr (m : Mem) (a : UInt64) : Option String := do
  let (r, off) ← decodeAddr a
  let bs := m.region r
  let n := (List.range (bs.size - off)).find? (fun i => bs.get! (off + i) == 0)
  let n ← n
  String.fromUTF8? ((List.range n).map (fun i => bs.get! (off + i))).toByteArray

private def copyIn (m : Mem) (a : UInt64) (src : ByteArray) : Option Mem :=
  (List.range src.size).foldlM
    (fun mm i => mm.store (a + UInt64.ofNat i) 1 (src.get! i).toUInt64) m

private def copyOut (m : Mem) (a : UInt64) (n : Nat) : Option ByteArray :=
  (List.range n).foldlM
    (fun (acc : ByteArray) i => do
      let b ← m.load (a + UInt64.ofNat i) 1
      pure (acc.push b.toUInt8))
    ByteArray.empty

/-- The file family, as a function of the world.

    These four are the only symbols with an executable contract: they are what
    the corpus uses, and their behavior is fully determined by the bytes.
    Every other symbol has a `Frame` and no definition — see `HProgFrames`. -/
def callFile (name : String) (args : List V) (w : World) : Option (Option V × World) := do
  let bits ← args.mapM asBits
  match name, bits with
  | "cl_file_read", [base, pathOff, dstOff, fileOff, size] => do
      let path ← readCStr w.mem (base + pathOff)
      match w.fs.get path with
      | none => some (some (ofInt .i64 (-1)), w)
      | some content =>
          let from_ := fileOff.toNat
          let want := if size == 0 then content.size - min from_ content.size else size.toNat
          let n := min want (content.size - min from_ content.size)
          let slice := ((List.range n).map (fun i => content.get! (from_ + i))).toByteArray
          let m ← copyIn w.mem (base + dstOff) slice
          some (some (ofInt .i64 n), { w with mem := m })
  | "cl_file_write", [base, pathOff, srcOff, fileOff, size] => do
      let path ← readCStr w.mem (base + pathOff)
      let bytes ← copyOut w.mem (base + srcOff) size.toNat
      let prev := (w.fs.get path).getD ByteArray.empty
      let at_ := fileOff.toNat
      let pad := if at_ > prev.size then at_ - prev.size else 0
      let head := ((List.range (min at_ prev.size)).map prev.get!).toByteArray
      let tail :=
        if at_ + bytes.size < prev.size then
          ((List.range (prev.size - at_ - bytes.size)).map
            (fun i => prev.get! (at_ + bytes.size + i))).toByteArray
        else ByteArray.empty
      let merged := head ++ ByteArray.mk (Array.replicate pad 0) ++ bytes ++ tail
      some (some (ofInt .i64 bytes.size), { w with fs := w.fs.set path merged })
  | _, _ => none

-- ---------------------------------------------------------------------------
-- Slot accounting
--
-- A loop's exit block and a branch's join block bind their parameters after
-- every slot the regions before them defined, whether or not those regions
-- ran. So the interpreter needs the *static* count, exactly as `compileFn`
-- and `wf` compute it.
-- ---------------------------------------------------------------------------

def stmtsSlots (n : Nat) (ss : List Stmt) : Nat :=
  ss.foldl (fun n s => match s with
    | .op _ | .call _ _ => n + 1
    | _ => n) n

def slotsGo : Nat → Nat → List Piece → Nat
  | 0, n, _ => n
  | _ + 1, n, [] => n
  | fuel + 1, n, .straight ss :: ps => slotsGo fuel (stmtsSlots n ss) ps
  | fuel + 1, n, .loop l pre body :: ps =>
      let afterBody := slotsGo fuel (slotsGo fuel (n + l.pTys.length) pre) body
      slotsGo fuel (afterBody + l.exitTys.length) ps
  | fuel + 1, n, .ite m thn els _ _ :: ps =>
      let afterEls := slotsGo fuel (slotsGo fuel n thn) els
      slotsGo fuel (afterEls + m.jTys.length) ps

/-- The number of slots in scope after `c`, starting from `n`. -/
def slotsOf (n : Nat) (c : Code) : Nat := slotsGo fuel n c

-- ---------------------------------------------------------------------------
-- Execution
-- ---------------------------------------------------------------------------

/-- Grow `Γ` to `n` slots so a binder lands at the index the compiled block
    parameter has, then append `vs`. -/
private def bindAt (Γ : Env) (n : Nat) (vs : List V) : Env :=
  (Γ.take n ++ Array.replicate (n - Γ.size) default) ++ vs.toArray

def obsCall (w : World) (fn : Nat) (args : List V) : World :=
  { w with obs := .call fn args :: w.obs }

def obsStore (w : World) (a : UInt64) (n : Nat) (v : UInt64) : World :=
  { w with obs := .store a n v :: w.obs }

structure Cfg where
  env : FnEnv
  /-- How many loop iterations one run may take in total. -/
  steps : Nat := 100000000

def runStmt (cfg : Cfg) (Γ : Env) (w : World) : Stmt → Outcome Env
  | .op o =>
      match evalOp w.mem Γ o with
      | some v => .ok (Γ.push v) w
      | none => .stuck s!"operation is undefined here: {repr o}"
  | .store ty v a =>
      match get Γ v, get Γ a with
      | some val, some (.sc _ addr) =>
          let n := tyBytes ty
          match val with
          | .sc _ b =>
              match w.mem.store addr n b with
              | some m => .ok Γ { obsStore w addr n b with mem := m }
              | none => .stuck s!"store to unmapped address {addr}"
          | .vec t ls =>
              let lw := ((t.lanes.map (·.1.width)).getD 8) / 8
              match (ls.zipIdx.foldlM
                  (fun mm (x, i) => mm.store (addr + UInt64.ofNat (i * lw)) lw x) w.mem) with
              | some m => .ok Γ { obsStore w addr n 0 with mem := m }
              | none => .stuck s!"vector store to unmapped address {addr}"
      | _, _ => .stuck "store operand is not in scope"
  | .storeUnaligned v a =>
      match get Γ v, get Γ a with
      | some (.sc t b), some (.sc _ addr) =>
          match w.mem.store addr (tyBytes t) b with
          | some m => .ok Γ { obsStore w addr (tyBytes t) b with mem := m }
          | none => .stuck s!"store to unmapped address {addr}"
      | _, _ => .stuck "store operand is not in scope"
  | .istore8 v a =>
      match get Γ v, get Γ a with
      | some (.sc _ b), some (.sc _ addr) =>
          match w.mem.store addr 1 (b &&& 0xff) with
          | some m => .ok Γ { obsStore w addr 1 (b &&& 0xff) with mem := m }
          | none => .stuck s!"istore8 to unmapped address {addr}"
      | _, _ => .stuck "istore8 operand is not in scope"
  | .call fn args => runCall cfg Γ w fn args true
  | .callVoid fn args => runCall cfg Γ w fn args false
where
  runCall (cfg : Cfg) (Γ : Env) (w : World) (fn : Nat) (args : List R) (binds : Bool) :
      Outcome Env :=
    match args.mapM (get Γ) with
    | none => .stuck s!"call argument to fn{fn} is not in scope"
    | some vs =>
        match cfg.env.fns.find? (·.ref.id == fn) with
        | none => .stuck s!"fn{fn} is not declared"
        | some d =>
            match d.callee with
            | .local i => .stuck s!"fn{fn} calls u0:{i}; only imports have contracts"
            | .import name =>
                match callFile name vs (obsCall w fn vs) with
                | none => .stuck s!"{name} has no executable contract"
                | some (res, w') =>
                    if binds then
                      match res with
                      | some v => .ok (Γ.push v) w'
                      | none => .stuck s!"{name} returned nothing to bind"
                    else .ok Γ w'

def runStmts (cfg : Cfg) : Env → World → List Stmt → Outcome Env
  | Γ, w, [] => .ok Γ w
  | Γ, w, s :: ss =>
      match runStmt cfg Γ w s with
      | .ok Γ' w' => runStmts cfg Γ' w' ss
      | .stuck m => .stuck m

mutual

/-- Fuel bounds every loop iteration and every nested region, **and is the
    structural argument the recursion decreases on**. That second role is the
    reason it is written this way: a `partial def` runs the same programs and
    proves nothing, because Lean gives partial definitions no equations. These
    have equations, so `compile_sound` is a statement one can actually work on. -/
def runPiece : Nat → Cfg → Env → World → Piece → Outcome Env
  | 0, _, _, _, _ => .stuck "step budget exhausted"
  | _ + 1, cfg, Γ, w, .straight ss => runStmts cfg Γ w ss
  | fuel + 1, cfg, Γ, w, .loop l pre body =>
      let n0 := Γ.size
      let afterBody := slotsOf (slotsOf (n0 + l.pTys.length) pre) body
      match l.init.mapM (get Γ) with
      | none => .stuck "loop initializer is not in scope"
      | some inits => iter fuel cfg Γ w l pre body n0 afterBody inits
  | fuel + 1, cfg, Γ, w, .ite m thn els thnR elsR =>
      match get Γ m.ca, get Γ m.cb with
      | some (.sc t x), some (.sc _ y) =>
          -- Both arms are numbered as if the one before it had run, because
          -- that is how `emitIte` numbers them. Only one arm executes, so the
          -- else arm starts from an environment padded past the then arm's
          -- slots; otherwise its own slots land at the then arm's indices.
          let thnEnd := slotsOf Γ.size thn
          let joinAt := slotsOf thnEnd els
          let (arm, exports, Γ0) :=
            if cmpInt m.cc t x y then (thn, thnR, Γ) else (els, elsR, bindAt Γ thnEnd [])
          match runCode fuel cfg Γ0 w arm with
          | .stuck s => .stuck s
          | .ok Γ' w' =>
              match exports.mapM (get Γ') with
              | none => .stuck "branch export is not in scope"
              | some vs => .ok (bindAt Γ' joinAt vs) w'
      | _, _ => .stuck "branch condition is not in scope"

/-- One trip of a loop: bind the carries, run the condition prefix, test, then
    either leave with the exit values or run the body and go round again. -/
def iter : Nat → Cfg → Env → World → Loop → List Piece → List Piece →
    Nat → Nat → List V → Outcome Env
  | 0, _, _, _, _, _, _, _, _, _ => .stuck "loop exceeded its step budget"
  | fuel + 1, cfg, Γ, w, l, pre, body, n0, afterBody, carries =>
      match runCode fuel cfg (bindAt Γ n0 carries) w pre with
      | .stuck s => .stuck s
      | .ok Γ1 w1 =>
          match get Γ1 l.ca, get Γ1 l.cb with
          | some (.sc t x), some (.sc _ y) =>
              if cmpInt l.cc t x y == l.exitOnTrue then
                match l.exitR.mapM (get Γ1) with
                | none => .stuck "loop exit value is not in scope"
                | some vs => .ok (bindAt Γ1 afterBody vs) w1
              else
                match runCode fuel cfg Γ1 w1 body with
                | .stuck s => .stuck s
                | .ok Γ2 w2 =>
                    match l.cont.mapM (get Γ2) with
                    | none => .stuck "loop carry is not in scope"
                    | some next => iter fuel cfg Γ w2 l pre body n0 afterBody next
          | _, _ => .stuck "loop condition is not in scope"

def runCode : Nat → Cfg → Env → World → List Piece → Outcome Env
  | 0, _, _, _, _ => .stuck "step budget exhausted"
  | _ + 1, _, Γ, w, [] => .ok Γ w
  | fuel + 1, cfg, Γ, w, p :: ps =>
      match runPiece fuel cfg Γ w p with
      | .stuck s => .stuck s
      | .ok Γ' w' => runCode fuel cfg Γ' w' ps
end

/-- Run a body with `params` bound to `args`, and hand back the world it
    leaves, with the observation trace in program order. -/
def run (cfg : Cfg) (args : List V) (w : World) (c : Code) : Outcome (List Obs) :=
  match runCode cfg.steps cfg args.toArray w c with
  | .stuck m => .stuck m
  | .ok _ w' => .ok w'.obs.reverse w'


/-- `runStmt`'s `istore8` arm, phrased without the private `get` so proofs in
    other files can rewrite with it. Holds by definition. -/
theorem runStmt_istore8 (cfg : Cfg) (Γ : Env) (w : World) (v a : R) :
    runStmt cfg Γ w (.istore8 v a)
      = match Γ[v]?, Γ[a]? with
        | some (.sc _ b), some (.sc _ addr) =>
            match w.mem.store addr 1 (b &&& 0xff) with
            | some m => .ok Γ { obsStore w addr 1 (b &&& 0xff) with mem := m }
            | none => .stuck s!"istore8 to unmapped address {addr}"
        | _, _ => .stuck "istore8 operand is not in scope" := rfl

/-- `runStmt`'s `storeUnaligned` arm, phrased without the private `get`. -/
theorem runStmt_storeUnaligned (cfg : Cfg) (Γ : Env) (w : World) (v a : R) :
    runStmt cfg Γ w (.storeUnaligned v a)
      = match Γ[v]?, Γ[a]? with
        | some (.sc t b), some (.sc _ addr) =>
            match w.mem.store addr (tyBytes t) b with
            | some m => .ok Γ { obsStore w addr (tyBytes t) b with mem := m }
            | none => .stuck s!"store to unmapped address {addr}"
        | _, _ => .stuck "store operand is not in scope" := rfl

/-- `runStmt`'s typed-`store` arm, phrased without the private `get`. -/
theorem runStmt_store (cfg : Cfg) (Γ : Env) (w : World) (ty : ClifTy) (v a : R) :
    runStmt cfg Γ w (.store ty v a)
      = match Γ[v]?, Γ[a]? with
        | some val, some (.sc _ addr) =>
            let n := tyBytes ty
            match val with
            | .sc _ b =>
                match w.mem.store addr n b with
                | some m => .ok Γ { obsStore w addr n b with mem := m }
                | none => .stuck s!"store to unmapped address {addr}"
            | .vec t ls =>
                let lw := ((t.lanes.map (·.1.width)).getD 8) / 8
                match (ls.zipIdx.foldlM
                    (fun mm (x, i) => mm.store (addr + UInt64.ofNat (i * lw)) lw x) w.mem) with
                | some m => .ok Γ { obsStore w addr n 0 with mem := m }
                | none => .stuck s!"vector store to unmapped address {addr}"
        | _, _ => .stuck "store operand is not in scope" := rfl

/-- `runStmt`'s `call` arm, phrased without the private `get`. -/
theorem runStmt_call (cfg : Cfg) (Γ : Env) (w : World) (fn : Nat) (args : List R) :
    runStmt cfg Γ w (.call fn args)
      = match args.mapM (fun r => Γ[r]?) with
        | none => .stuck s!"call argument to fn{fn} is not in scope"
        | some vs =>
            match cfg.env.fns.find? (·.ref.id == fn) with
            | none => .stuck s!"fn{fn} is not declared"
            | some d =>
                match d.callee with
                | .local i => .stuck s!"fn{fn} calls u0:{i}; only imports have contracts"
                | .import name =>
                    match callFile name vs (obsCall w fn vs) with
                    | none => .stuck s!"{name} has no executable contract"
                    | some (res, w') =>
                        match res with
                        | some v => .ok (Γ.push v) w'
                        | none => .stuck s!"{name} returned nothing to bind" := rfl

/-- `runStmt`'s `callVoid` arm, phrased without the private `get`. -/
theorem runStmt_callVoid (cfg : Cfg) (Γ : Env) (w : World) (fn : Nat) (args : List R) :
    runStmt cfg Γ w (.callVoid fn args)
      = match args.mapM (fun r => Γ[r]?) with
        | none => .stuck s!"call argument to fn{fn} is not in scope"
        | some vs =>
            match cfg.env.fns.find? (·.ref.id == fn) with
            | none => .stuck s!"fn{fn} is not declared"
            | some d =>
                match d.callee with
                | .local i => .stuck s!"fn{fn} calls u0:{i}; only imports have contracts"
                | .import name =>
                    match callFile name vs (obsCall w fn vs) with
                    | none => .stuck s!"{name} has no executable contract"
                    | some (_, w') => .ok Γ w' := rfl

end AlgorithmLib.HProg.Sem
