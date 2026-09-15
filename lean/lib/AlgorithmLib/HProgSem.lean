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

/-- The type a value carries, which is the type `Op.check` predicts for the
    operation that produced it. -/
def V.ty : V → ClifTy
  | .sc t _  => t
  | .vec t _ => t

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

def get (Γ : Env) (r : R) : Option V := Γ[r]?

/-- An arithmetic operation on two integer scalars of **one** type, which is
    what `Op.check` requires of the operations reached through here and what
    Cranelift's verifier requires of the instruction. Taking the left operand's
    type and evaluating anyway would answer for a program that cannot be built:
    `iadd (i64 0) (i32 (-1))` would wrap at 64 bits and give `2^32 - 1` where
    the operand denotes `-1`. Refusing is what keeps the two semantics over
    `Inst` — this one and `Clif.stepPure` — from disagreeing there. -/
def bin (Γ : Env) (a b : R) (f : ClifTy → UInt64 → UInt64 → Option V) : Option V := do
  let (.sc ta x) ← get Γ a | none
  let (.sc tb y) ← get Γ b | none
  if ta == tb && ta.isInt then f ta x y else none

/-- The shift rule, which is the one binary integer operation whose operands
    may differ: Cranelift takes the amount at any integer width and the result
    at the shifted operand's. -/
def shiftBin (Γ : Env) (a b : R) (f : ClifTy → UInt64 → UInt64 → Option V) :
    Option V := do
  let (.sc ta x) ← get Γ a | none
  let (.sc tb y) ← get Γ b | none
  if ta.isInt && tb.isInt then f ta x y else none

/-- A unary operation on one scalar whose type `ok` admits — the condition
    `Op.check`'s arm for it states. -/
def un (Γ : Env) (a : R) (ok : ClifTy → Bool) (f : ClifTy → UInt64 → Option V) :
    Option V := do
  let (.sc t x) ← get Γ a | none
  if ok t then f t x else none

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
def zipBitsIf (ok : ClifTy → Bool) (u v : V)
    (f : ClifTy → UInt64 → UInt64 → UInt64) : Option V :=
  match u, v with
  | .sc t x, .sc t' y => if t == t' && ok t then some (.sc t (f t x y)) else none
  | .vec t xs, .vec t' ys =>
      match t.lanes with
      | some (lane, _) =>
          if t == t' && ok lane && xs.size == ys.size then
            some (.vec t (xs.zipWith (f lane) ys))
          else none
      | none => none
  | _, _ => none

/-- The float-shaped uses: `fmax`, `fmin` and `bitselect`'s mask arithmetic. -/
private def zipBits : V → V → (ClifTy → UInt64 → UInt64 → UInt64) → Option V :=
  zipBitsIf (·.isFloat)

/-- Bit-for-bit on two values of one type, whatever that type is. `bitselect`'s
    check constrains only that its three operands agree, so an integer mask is
    as well formed as a float one. -/
def zipAnyBits : V → V → (ClifTy → UInt64 → UInt64 → UInt64) → Option V :=
  zipBitsIf (fun _ => true)

/-- A bitwise operation applied to two integers or lane-wise to two vectors.

    `zipBits` is the float-shaped twin: it rejects integer scalars because the
    operations that use it are float ones. The bitwise operations apply to both,
    and `Op.check` accepts a vector for every one of them, so the semantics has
    to as well or a term can pass the checker and get stuck here. -/
private def zipIntBits (u v : V) (f : ClifTy → UInt64 → UInt64 → UInt64) : Option V :=
  match u, v with
  | .sc t x, .sc t' y =>
      if t == t' && (t.isInt || t.isVec) then some (norm t (f t x y)) else none
  | .vec t xs, .vec t' ys =>
      match t.lanes with
      | some (lane, _) =>
          if t == t' && xs.size == ys.size then
            some (.vec t (xs.zipWith (fun x y => f lane x y &&& widthMask lane) ys))
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

def cmpInt : ICmpCond → ClifTy → UInt64 → UInt64 → Bool
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

def boolV (b : Bool) : V := .sc .i8 (if b then 1 else 0)

/-- A comparison applied to two integers, or lane-wise to two vectors, where a
    true lane is all ones at the lane's own width — the mask `vhighBits` and
    `bitselect` read. -/
def zipIntCmp (c : ICmpCond) (u v : V) : Option V :=
  match u, v with
  | .sc t x, .sc t' y =>
      if t == t' && t.isInt then some (boolV (cmpInt c t x y)) else none
  | .vec t xs, .vec t' ys =>
      match t.lanes with
      | some (lane, _) =>
          if t == t' && xs.size == ys.size then
            some (.vec t (xs.zipWith
              (fun x y => if cmpInt c lane x y then widthMask lane else 0) ys))
          else none
      | none => none
  | _, _ => none


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
  | .ishl a b => shiftBin Γ a b fun t x y => some (norm t (x <<< (y % UInt64.ofNat t.width)))
  | .ushr a b => shiftBin Γ a b fun t x y =>
      some (norm t ((x &&& widthMask t) >>> (y % UInt64.ofNat t.width)))
  | .band a b => do zipIntBits (← get Γ a) (← get Γ b) (fun _ x y => x &&& y)
  | .bandNot a b => do zipIntBits (← get Γ a) (← get Γ b) (fun _ x y => x &&& ~~~y)
  | .bor a b => do zipIntBits (← get Γ a) (← get Γ b) (fun _ x y => x ||| y)
  | .bxor a b => do zipIntBits (← get Γ a) (← get Γ b) (fun _ x y => x ^^^ y)
  | .ineg a => un Γ a (·.isInt) fun t x => some (norm t (0 - x))
  | .ctz a => un Γ a (·.isInt) fun t x =>
      some (norm t (UInt64.ofNat (((List.range t.width).find? (fun i =>
        (x >>> UInt64.ofNat i) &&& 1 == 1)).getD t.width)))
  | .popcnt a => un Γ a (·.isInt) fun t x =>
      some (norm t (UInt64.ofNat (((List.range t.width).filter (fun i =>
        (x >>> UInt64.ofNat i) &&& 1 == 1)).length)))
  | .ireduce32 a => un Γ a (fun t => t.isInt && t.width > 32) fun _ x => some (norm .i32 x)
  | .uextend64 a => un Γ a (fun t => t.isInt && t.width < 64) fun t x =>
      some (.sc .i64 (x &&& widthMask t))
  | .sextend64 a => un Γ a (fun t => t.isInt && t.width < 64) fun t x =>
      some (ofInt .i64 (signed t x))
  | .icmp c a b => do zipIntCmp c (← get Γ a) (← get Γ b)
  | .select c a b => do
      let cv ← get Γ c
      let av ← get Γ a
      let bv ← get Γ b
      -- `Op.check` requires an integer condition and one type on the arms; a
      -- select that returned whichever operand it was handed would answer for
      -- a program Cranelift refuses to build.
      let (.sc tc _) := cv | none
      if tc.isInt && av.ty == bv.ty then (if isTrue cv then some av else some bv) else none
  -- Bit-for-bit, per lane: the mask decides each bit, not each lane as a whole.
  -- That is what `bitselect` means and why a comparison result has to be
  -- bitcast to the operand width before it can be used here.
  | .bitselect c a b => do
      let cv ← get Γ c
      let av ← get Γ a
      let bv ← get Γ b
      let masked ← zipAnyBits cv av (fun _ m x => m &&& x)
      let other ← zipAnyBits cv bv (fun _ m y => (~~~m) &&& y)
      zipAnyBits masked other (fun _ x y => x ||| y)
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
      if !t.isInt then none else
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
  intro m Γ a b x y ha hb; simp [evalOp, shiftBin, get, ha, hb]
theorem reloc2_ushr : Reloc2 .ushr := by
  intro m Γ a b x y ha hb; simp [evalOp, shiftBin, get, ha, hb]
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
  intro m Γ a x ha; simp [evalOp, un, get, ha]
theorem reloc1_ctz : Reloc1 .ctz := by
  intro m Γ a x ha; simp [evalOp, un, get, ha]
theorem reloc1_popcnt : Reloc1 .popcnt := by
  intro m Γ a x ha; simp [evalOp, un, get, ha]
theorem reloc1_ireduce32 : Reloc1 .ireduce32 := by
  intro m Γ a x ha; simp [evalOp, un, get, ha]
theorem reloc1_uextend64 : Reloc1 .uextend64 := by
  intro m Γ a x ha; simp [evalOp, un, get, ha]
theorem reloc1_sextend64 : Reloc1 .sextend64 := by
  intro m Γ a x ha; simp [evalOp, un, get, ha]
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

/-- The hash table `ht.rs` keeps, which lives outside addressable memory.

    Faithful to the Rust in two respects that read as defects and are not
    modelled away: every accessor hardcodes table `0`, so the handle `ht_create`
    returns is decorative and only the *first* create makes the accessors work;
    and entries are stored in a `HashMap`, so `ht_get_entry` iterates in an
    order the implementation does not fix. Insertion order is what this models,
    and a program whose answer depends on that order is relying on something the
    real thing does not promise. -/
structure Ht where
  /-- Handles handed out by `ht_create`, so the table `0` the accessors read is
      present exactly when one of them was `0`. -/
  handles : List Nat := []
  entries : List (ByteArray × ByteArray) := []
  deriving Inhabited

def Ht.hasTable (h : Ht) : Bool := h.handles.contains 0

def Ht.get (h : Ht) (k : ByteArray) : Option ByteArray :=
  (h.entries.find? (·.1.toList == k.toList)).map (·.2)

/-- Overwrite in place, keeping the entry's position, or append. -/
def Ht.set (h : Ht) (k v : ByteArray) : Ht :=
  if (h.get k).isSome then
    { h with entries := h.entries.map (fun e => if e.1.toList == k.toList then (e.1, v) else e) }
  else { h with entries := h.entries ++ [(k, v)] }

structure World where
  mem : Mem
  fs  : FS := {}
  /-- Lines `cl_stdin_readline` will return, in order, each including its
      newline as `read_line` leaves it. An empty list is end of input. -/
  stdin : List ByteArray := []
  /-- What `cl_stdout_write` has written so far. -/
  stdout : ByteArray := ByteArray.empty
  ht : Ht := {}
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

/-- The hash-table context, which is a Rust allocation rather than anything in
    the three regions. `decodeAddr` rejects it, so it can be stored and passed
    but never loaded through. -/
def htCtx : UInt64 := 0x5000000000

/-- An `i64` as the eight bytes the implementation stores it in. -/
private def le64 (x : UInt64) : ByteArray :=
  ((List.range 8).map (fun i => ((x >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)).toByteArray

/-- The `i64` the first eight bytes of `b` encode. -/
private def ofLe64 (b : ByteArray) : UInt64 :=
  (List.range 8).foldr (fun (i : Nat) (acc : UInt64) => (acc <<< 8) ||| (b.get! i).toUInt64) 0

/-- The bytes from `a` up to the first zero. -/
private def readCStrAt (m : Mem) (a : UInt64) : Option String := readCStr m a

/-- `n` bytes at `a` as a `ByteArray`, or `none` if they are not all mapped. -/
private def readBytes (m : Mem) (a : UInt64) (n : Nat) : Option ByteArray := copyOut m a n

/-- Writing `bytes` into a file at `at_`, as `write_all` after a seek leaves it:
    a seek past the end leaves a hole that reads back as zeros, and a write
    shorter than what is there does *not* truncate the rest. -/
private def spliceAt (prev bytes : ByteArray) (at_ : Nat) : ByteArray :=
  let pad := if at_ > prev.size then at_ - prev.size else 0
  let head := ((List.range (min at_ prev.size)).map prev.get!).toByteArray
  let tail :=
    if at_ + bytes.size < prev.size then
      ((List.range (prev.size - at_ - bytes.size)).map
        (fun i => prev.get! (at_ + bytes.size + i))).toByteArray
    else ByteArray.empty
  head ++ ByteArray.mk (Array.replicate pad 0) ++ bytes ++ tail

/-- The `i64` a `UInt64` denotes, for the arguments the Rust reads as signed. -/
private def asI64 (x : UInt64) : Int := signed .i64 x

/-- What each entry point does, as a function of the world.

    The families here are the ones whose behavior is fully determined by the
    bytes: files, standard streams, the hash table and the libm shims. Each is a
    transcription of `base/src/ffi/`, and like `evalOp` a transcription is an
    assumption — what makes it more than that is the corpus, which runs these
    against the real symbols through the JIT and compares.

    Everything else — devices, windows, database handles — has a `Frame` and no
    definition, and calling one goes `stuck` rather than guessing. -/
def callFfi (f : IR.Ffi) (args : List V) (w : World) : Option (Option V × World) := do
  let bits ← args.mapM asBits
  match f, bits with
  -- file.rs
  | .fileRead, [base, pathOff, dstOff, fileOff, size] => do
      let path ← readCStrAt w.mem (base + pathOff)
      match w.fs.get path with
      | none => some (some (ofInt .i64 (-1)), w)
      | some content =>
          let from_ := fileOff.toNat
          let avail := content.size - min from_ content.size
          let want := if size == 0 then avail else size.toNat
          let n := min want avail
          let slice := ((List.range n).map (fun i => content.get! (from_ + i))).toByteArray
          let m ← copyIn w.mem (base + dstOff) slice
          some (some (ofInt .i64 n), { w with mem := m })
  | .fileWrite, [base, pathOff, srcOff, fileOff, size] => do
      let path ← readCStrAt w.mem (base + pathOff)
      -- `size == 0` means "the bytes up to the first NUL", and writing at
      -- offset 0 goes through `File::create`, which truncates.
      let bytes ←
        if size == 0 then (readCStrAt w.mem (base + srcOff)).map (·.toUTF8)
        else readBytes w.mem (base + srcOff) size.toNat
      let at_ := fileOff.toNat
      let prev := if at_ == 0 then ByteArray.empty else (w.fs.get path).getD ByteArray.empty
      some (some (ofInt .i64 bytes.size), { w with fs := w.fs.set path (spliceAt prev bytes at_) })
  -- These two take the path and the buffer as raw pointers, so neither is
  -- relative to the shared arena.
  | .fileReadToPtr, [pathPtr, dstPtr, fileOff, size] =>
      if asI64 size ≤ 0 then some (some (ofInt .i64 (-1)), w)
      else do
        let path ← readCStrAt w.mem pathPtr
        match w.fs.get path with
        | none => some (some (ofInt .i64 (-1)), w)
        | some content =>
            let from_ := fileOff.toNat
            let avail := content.size - min from_ content.size
            let n := min size.toNat avail
            let slice := ((List.range n).map (fun i => content.get! (from_ + i))).toByteArray
            let m ← copyIn w.mem dstPtr slice
            some (some (ofInt .i64 n), { w with mem := m })
  | .fileWriteFromPtr, [pathPtr, srcPtr, fileOff, size] =>
      if asI64 size ≤ 0 || asI64 fileOff < 0 then some (some (ofInt .i64 (-1)), w)
      else do
        let path ← readCStrAt w.mem pathPtr
        let bytes ← readBytes w.mem srcPtr size.toNat
        let prev := (w.fs.get path).getD ByteArray.empty
        some (some (ofInt .i64 bytes.size),
              { w with fs := w.fs.set path (spliceAt prev bytes fileOff.toNat) })

  -- stdio.rs — a line comes back with its newline, capped at `max_len - 1` and
  -- terminated; end of input is `0` and writes nothing.
  | .stdinReadline, [base, dstOff, maxLen] =>
      if asI64 maxLen ≤ 0 then some (some (ofInt .i64 0), w)
      else match w.stdin with
        | [] => some (some (ofInt .i64 0), w)
        | line :: rest => do
            let n := min line.size (maxLen.toNat - 1)
            let m ← copyIn w.mem (base + dstOff) (((List.range n).map line.get!).toByteArray)
            let m ← m.store (base + dstOff + UInt64.ofNat n) 1 0
            some (some (ofInt .i64 n), { w with mem := m, stdin := rest })
  | .stdoutWrite, [base, srcOff, size] =>
      if asI64 size < 0 then some (some (ofInt .i64 (-1)), w)
      else do
        let bytes ← readBytes w.mem (base + srcOff) size.toNat
        some (some (ofInt .i64 size.toNat), { w with stdout := w.stdout ++ bytes })

  -- mod.rs — `Float32.sin`/`cos` are `@[extern "sinf"]`/`"cosf"`, the same libm
  -- symbols these shims call, so on one host the two agree by construction.
  -- Across hosts libm is free to differ in the last ulp, and then this is a
  -- transcription like any other.
  | .sinf, [x] => some (some (.sc .f32 (ofF32 (f32 x).sin)), w)
  | .cosf, [x] => some (some (.sc .f32 (ofF32 (f32 x).cos)), w)
  | .powf, [b, e] => some (some (.sc .f32 (ofF32 ((f32 b).pow (f32 e)))), w)

  -- ht.rs — the context is not addressable memory, so `init` writes a sentinel
  -- that no region decodes: a program that dereferences the handle or does
  -- arithmetic on it goes stuck here rather than quietly agreeing with a run
  -- that would have faulted.
  | .htInit, [slot] => do
      let m ← w.mem.store slot 8 htCtx
      some (none, { w with mem := m, ht := {} })
  | .htCleanup, [slot] => do
      let m ← w.mem.store slot 8 0
      some (none, { w with mem := m, ht := {} })
  | .htCreate, [ctx] =>
      if ctx != htCtx then some (some (ofInt .i32 0xFFFFFFFF), w)
      else
        let handle := w.ht.handles.length
        some (some (ofInt .i32 handle),
              { w with ht := { w.ht with handles := w.ht.handles ++ [handle] } })
  | .htCount, [ctx] =>
      if ctx != htCtx || !w.ht.hasTable then some (some (ofInt .i32 0), w)
      else some (some (ofInt .i32 w.ht.entries.length), w)
  | .htLookup, [ctx, keyPtr, keyLen, resultPtr] => do
      let key ← readBytes w.mem keyPtr keyLen.toNat
      if ctx != htCtx || !w.ht.hasTable then some (some (ofInt .i32 0xFFFFFFFF), w)
      else match w.ht.get key with
        | none => some (some (ofInt .i32 0xFFFFFFFF), w)
        | some val => do
            let m ← copyIn w.mem resultPtr val
            some (some (ofInt .i32 val.size), { w with mem := m })
  | .htInsert, [ctx, keyPtr, keyLen, valPtr, valLen] => do
      let key ← readBytes w.mem keyPtr keyLen.toNat
      let val ← readBytes w.mem valPtr valLen.toNat
      if ctx != htCtx || !w.ht.hasTable then some (none, w)
      else some (none, { w with ht := w.ht.set key val })
  | .htIncrement, [ctx, keyPtr, keyLen, addend] => do
      let key ← readBytes w.mem keyPtr keyLen.toNat
      if ctx != htCtx || !w.ht.hasTable then some (some (.sc .i64 addend), w)
      else match w.ht.get key with
        | none =>
            some (some (.sc .i64 addend), { w with ht := w.ht.set key (le64 addend) })
        | some existing =>
            -- The counter is the first eight bytes; anything the value carries
            -- past them is left alone. A value shorter than eight bytes indexes
            -- out of range in the Rust, so there is nothing here to transcribe.
            if existing.size < 8 then none
            else
              let next := ofLe64 existing + addend
              let updated := le64 next ++
                ((List.range (existing.size - 8)).map (fun i => existing.get! (8 + i))).toByteArray
              some (some (.sc .i64 next), { w with ht := w.ht.set key updated })
  | .htGetEntry, [ctx, index, keyOut, valOut] =>
      if ctx != htCtx || !w.ht.hasTable then some (some (ofInt .i32 (-1)), w)
      else match w.ht.entries[index.toNat]? with
        | none => some (some (ofInt .i32 (-1)), w)
        | some (k, v) => do
            let m ← copyIn w.mem keyOut k
            let m ← copyIn m valOut v
            some (some (ofInt .i32 k.size), { w with mem := m })

  | _, _ => none

/-- The entry point a symbol names, and what it does. A name that is not an
    entry point — a program's own colocated function — has no contract. -/
def callImport (name : String) (args : List V) (w : World) : Option (Option V × World) := do
  let f ← IR.Ffi.ofCname name
  callFfi f args w

-- ---------------------------------------------------------------------------
-- Slot accounting
--
-- A loop's exit block and a branch's join block bind their parameters after
-- every slot the regions before them defined, whether or not those regions
-- ran. So the interpreter needs the *static* count, exactly as the compiler
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
      let jn := if termsGo fuel thn && termsGo fuel els then 0 else m.jTys.length
      slotsGo fuel (afterEls + jn) ps
  | fuel + 1, n, .dloop l body :: ps =>
      let afterBody := slotsGo fuel (n + l.pTys.length) body
      slotsGo fuel (afterBody + l.exitTys.length) ps
  | fuel + 1, n, .br _ _ :: ps | fuel + 1, n, .cont _ _ :: ps => slotsGo fuel n ps

/-- The number of slots in scope after `c`, starting from `n`. -/
def slotsOf (n : Nat) (c : Code) : Nat := slotsGo fuel n c

-- ---------------------------------------------------------------------------
-- Execution
-- ---------------------------------------------------------------------------

/-- Grow `Γ` to `n` slots so a binder lands at the index the compiled block
    parameter has, then append `vs`. -/
def bindAt (Γ : Env) (n : Nat) (vs : List V) : Env :=
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
                match callImport name vs (obsCall w fn vs) with
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

/-- What running a region can do: finish, leave an enclosing loop, or get stuck.
    `brk` carries the environment it left from, because the exit block binds its
    parameters at a static index that may be past what the abandoned path had. -/
inductive CodeRes where
  | ok    (Γ : Env) (w : World)
  | brk   (depth : Nat) (Γ : Env) (vals : List V) (w : World)
  /-- Go round the `depth`-th enclosing top-tested loop again. -/
  | cont  (depth : Nat) (vals : List V) (w : World)
  | stuck (why : String)

mutual

/-- Fuel bounds every loop iteration and every nested region, **and is the
    structural argument the recursion decreases on**. That second role is the
    reason it is written this way: a `partial def` runs the same programs and
    proves nothing, because Lean gives partial definitions no equations. These
    have equations, so `compile_sound` is a statement one can actually work on. -/
def runPiece : Nat → Cfg → Env → World → Piece → CodeRes
  | 0, _, _, _, _ => .stuck "step budget exhausted"
  | _ + 1, cfg, Γ, w, .straight ss =>
      match runStmts cfg Γ w ss with
      | .ok Γ' w' => .ok Γ' w'
      | .stuck m => .stuck m
  | _ + 1, _, Γ, w, .br depth args =>
      match args.mapM (get Γ) with
      | none => .stuck "branch-out value is not in scope"
      | some vs => .brk depth Γ vs w
  | _ + 1, _, Γ, w, .cont depth args =>
      match args.mapM (get Γ) with
      | none => .stuck "loop carry is not in scope"
      | some vs => .cont depth vs w
  | fuel + 1, cfg, Γ, w, .dloop l body =>
      let n0 := Γ.size
      let afterBody := slotsOf (n0 + l.pTys.length) body
      match l.init.mapM (get Γ) with
      | none => .stuck "loop initializer is not in scope"
      | some inits => dtrip fuel cfg Γ w l body n0 afterBody inits l.guardIdx.isSome
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
          | .brk d Γb vs w' => .brk d Γb vs w'
          | .cont d vs w' => .cont d vs w'
          | .ok Γ' w' =>
              match exports.mapM (get Γ') with
              | none => .stuck "branch export is not in scope"
              | some vs => .ok (bindAt Γ' joinAt vs) w'
      | _, _ => .stuck "branch condition is not in scope"

/-- One trip of a loop: bind the carries, run the condition prefix, test, then
    either leave with the exit values or run the body and go round again.

    A `br 0` from either region leaves here, binding its values where the normal
    exit would have; a deeper one is passed out with its depth reduced. -/
def iter : Nat → Cfg → Env → World → Loop → List Piece → List Piece →
    Nat → Nat → List V → CodeRes
  | 0, _, _, _, _, _, _, _, _, _ => .stuck "loop exceeded its step budget"
  | fuel + 1, cfg, Γ, w, l, pre, body, n0, afterBody, carries =>
      match runCode fuel cfg (bindAt Γ n0 carries) w pre with
      | .stuck s => .stuck s
      | .brk 0 Γb vs w' => .ok (bindAt Γb afterBody vs) w'
      | .brk (d + 1) Γb vs w' => .brk d Γb vs w'
      | .cont 0 vs w' => iter fuel cfg Γ w' l pre body n0 afterBody vs
      | .cont (d + 1) vs w' => .cont d vs w'
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
                | .brk 0 Γb vs w2 => .ok (bindAt Γb afterBody vs) w2
                | .brk (d + 1) Γb vs w2 => .brk d Γb vs w2
                | .cont 0 vs w2 => iter fuel cfg Γ w2 l pre body n0 afterBody vs
                | .cont (d + 1) vs w2 => .cont d vs w2
                | .ok Γ2 w2 =>
                    match l.cont.mapM (get Γ2) with
                    | none => .stuck "loop carry is not in scope"
                    | some next => iter fuel cfg Γ w2 l pre body n0 afterBody next
          | _, _ => .stuck "loop condition is not in scope"

/-- One trip of a bottom-tested loop. `first` says the test has not run yet, so
    the guard is made before the body rather than after it; every later call
    arrives with the test already passed. -/
def dtrip : Nat → Cfg → Env → World → DLoop → List Piece →
    Nat → Nat → List V → Bool → CodeRes
  | 0, _, _, _, _, _, _, _, _, _ => .stuck "loop exceeded its step budget"
  | fuel + 1, cfg, Γ, w, l, body, n0, afterBody, carries, first =>
      let leave (Γ' : Env) (vs : List V) (w' : World) : CodeRes :=
        match l.exitIdx.mapM (fun i => vs[i]?) with
        | none => .stuck "loop exit value is not a carry"
        | some outs => .ok (bindAt Γ' afterBody outs) w'
      -- At the guard the tested value is a carry; at the back edge it is a slot
      -- the body defined, so the two reads differ.
      let testAt (Γ' : Env) (v : Option V) : Option Bool := do
        let (.sc t x) ← v | none
        let (.sc _ y) ← get Γ' l.cb | none
        pure (cmpInt l.cc t x y)
      if first then
        match testAt Γ (l.guardIdx.bind (carries[·]?)) with
        | none => .stuck "loop condition is not in scope"
        | some c =>
            if c == l.contOnTrue then
              dtrip fuel cfg Γ w l body n0 afterBody carries false
            else leave Γ carries w
      else
        match runCode fuel cfg (bindAt Γ n0 carries) w body with
        | .stuck s => .stuck s
        | .brk 0 Γb vs w' => .ok (bindAt Γb afterBody vs) w'
        | .brk (d + 1) Γb vs w' => .brk d Γb vs w'
        | .cont 0 vs w2 => dtrip fuel cfg Γ w2 l body n0 afterBody vs false
        | .cont (d + 1) vs w' => .cont d vs w'
        | .ok Γ2 w2 =>
            -- Reaching here means the body fell through, so it did supply
            -- carries and a value for the back-edge test.
            match l.cont.mapM (get Γ2) with
            | none => .stuck "loop carry is not in scope"
            | some next =>
                match testAt Γ2 (get Γ2 l.ca) with
                | none => .stuck "loop condition is not in scope"
                | some c =>
                    if c == l.contOnTrue then
                      dtrip fuel cfg Γ w2 l body n0 afterBody next false
                    else leave Γ2 next w2

def runCode : Nat → Cfg → Env → World → List Piece → CodeRes
  | 0, _, _, _, _ => .stuck "step budget exhausted"
  | _ + 1, _, Γ, w, [] => .ok Γ w
  | fuel + 1, cfg, Γ, w, p :: ps =>
      match runPiece fuel cfg Γ w p with
      | .stuck s => .stuck s
      | .brk d Γb vs w' => .brk d Γb vs w'
      | .cont d vs w' => .cont d vs w'
      | .ok Γ' w' => runCode fuel cfg Γ' w' ps
end

/-- Run a body with `params` bound to `args`, and hand back the world it
    leaves, with the observation trace in program order. -/
def run (cfg : Cfg) (args : List V) (w : World) (c : Code) : Outcome (List Obs) :=
  match runCode cfg.steps cfg args.toArray w c with
  | .stuck m => .stuck m
  | .brk d _ _ _ => .stuck s!"br {d} leaves the function body"
  | .cont d _ _ => .stuck s!"continue {d} leaves the function body"
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
                    match callImport name vs (obsCall w fn vs) with
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
                    match callImport name vs (obsCall w fn vs) with
                    | none => .stuck s!"{name} has no executable contract"
                    | some (_, w') => .ok Γ w' := rfl

end AlgorithmLib.HProg.Sem
