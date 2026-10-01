module
public import AlgorithmLib.Host.Term
meta import AlgorithmLib.Host.Term
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The world a term runs in

Values, memory, the operations `evalOp` defines, and the `World` that holds
memory, files, standard streams, the hash table and the devices. `Ffi.lean`
says what the foreign calls do to it and `Sem.lean` runs terms over it.
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

def f32 (b : UInt64) : Float32 := Float32.ofBits b.toUInt32
def f64 (b : UInt64) : Float := Float.ofBits b
def ofF32 (x : Float32) : UInt64 := x.toBits.toUInt64
def ofF64 (x : Float) : UInt64 := x.toBits

-- ---------------------------------------------------------------------------
-- Memory
-- ---------------------------------------------------------------------------

/-- The three regions CLIF code can address: the shared arena and the caller's
    input and output buffers, each reached through a pointer an entry point is
    called with. -/
inductive Region where
  | arena | data | out
  /-- Page-locked host memory `cl_cuda_pinned_alloc` hands out: addressable
      only inside a live allocation. -/
  | pinned
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Region bases are far apart and aligned to a power of two, so `base + off`
    arithmetic stays inside its region and a stray address is detectable
    instead of silently landing somewhere valid. -/
def regionBase : Region → UInt64
  | .arena => 0x1000000000
  | .data  => 0x2000000000
  | .out   => 0x3000000000
  | .pinned => 0x4000000000

def regionSpan : UInt64 := 0x1000000000

def addrOf (r : Region) (off : Nat) : UInt64 := regionBase r + UInt64.ofNat off

/-- Which region an address names, and how far into it. -/
def decodeAddr (a : UInt64) : Option (Region × Nat) :=
  [Region.arena, .data, .out, .pinned].findSome? fun r =>
    let b := regionBase r
    if a ≥ b && a - b < regionSpan then some (r, (a - b).toNat) else none

/-- An asynchronous copy still in flight over pinned bytes: which bytes, the
    party and tick the host must wait for, and whether the copy writes them (a
    download) or only reads them (an upload). -/
structure Busy where
  off : Nat
  len : Nat
  party : Nat
  tick : Nat
  writes : Bool
  deriving Inhabited

/-- `n` bytes at `off` are clear of the copy. -/
def Busy.clear (b : Busy) (off n : Nat) : Bool := off + n ≤ b.off || b.off + b.len ≤ off

structure Mem where
  arena : ByteArray
  data  : ByteArray
  out   : ByteArray
  pinned : ByteArray := ByteArray.empty
  /-- The live pinned allocations, as `(offset, length)` in `pinned`. A freed
      allocation's bytes stay, unreachable, so a use after free is stuck. -/
  pinnedLive : List (Nat × Nat) := []
  /-- In-flight asynchronous copies that touch pinned memory: until the host
      has waited for one, it may not write the bytes it covers, nor read them if
      the copy writes them. -/
  busy : List Busy := []
  /-- A spawned worker has not been joined: the host may touch no memory until
      it is, because the worker may be reading or writing any of it. -/
  frozen : Bool := false
  deriving Inhabited

def Mem.region : Mem → Region → ByteArray
  | m, .arena => m.arena
  | m, .data => m.data
  | m, .out => m.out
  | m, .pinned => m.pinned

def Mem.setRegion (m : Mem) : Region → ByteArray → Mem
  | .arena, b => { m with arena := b }
  | .data, b => { m with data := b }
  | .out, b => { m with out := b }
  | .pinned, b => { m with pinned := b }

/-- Whether `n` bytes at `off` in region `r` may be written: always, except in
    pinned memory, where they must lie inside one live allocation and clear of
    every copy still in flight. -/
def Mem.reachable (m : Mem) (r : Region) (off n : Nat) : Bool :=
  !m.frozen && (r != .pinned || (m.pinnedLive.any (fun (o, len) => o ≤ off && off + n ≤ o + len)
    && m.busy.all (·.clear off n)))

/-- Whether they may be read: the same, except that a copy that only reads
    them (an upload) does not stand in the way. -/
def Mem.readable (m : Mem) (r : Region) (off n : Nat) : Bool :=
  !m.frozen && (r != .pinned || (m.pinnedLive.any (fun (o, len) => o ≤ off && off + n ≤ o + len)
    && m.busy.all (fun b => !b.writes || b.clear off n)))

/-- `n` bytes at `a`, little-endian, or `none` if they are not all in one
    region — a load the real program would fault on. -/
def Mem.load (m : Mem) (a : UInt64) (n : Nat) : Option UInt64 := do
  let (r, off) ← decodeAddr a
  let bs := m.region r
  if off + n > bs.size || !m.readable r off n then none
  else
    some <| (List.range n).foldr
      (fun i acc => (acc <<< 8) ||| (bs.get! (off + i)).toUInt64) 0

def Mem.store (m : Mem) (a : UInt64) (n : Nat) (v : UInt64) : Option Mem := do
  let (r, off) ← decodeAddr a
  let bs := m.region r
  if off + n > bs.size || !m.reachable r off n then none
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
def zipF (u v : V) (f32op : Float32 → Float32 → Float32)
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
def zipBits : V → V → (ClifTy → UInt64 → UInt64 → UInt64) → Option V :=
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
def zipIntBits (u v : V) (f : ClifTy → UInt64 → UInt64 → UInt64) : Option V :=
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

def isNaNBits (t : ClifTy) (x : UInt64) : Bool :=
  match t with
  | .f32 => (x &&& 0x7f800000) == 0x7f800000 && (x &&& 0x007fffff) != 0
  | .f64 => (x &&& 0x7ff0000000000000) == 0x7ff0000000000000 &&
            (x &&& 0x000fffffffffffff) != 0
  | _ => false

def signBit (t : ClifTy) (x : UInt64) : Bool :=
  match t with
  | .f32 => x &&& 0x80000000 != 0
  | _ => x &&& 0x8000000000000000 != 0

def ltBits (t : ClifTy) (x y : UInt64) : Bool :=
  match t with
  | .f32 => f32 x < f32 y
  | _ => f64 x < f64 y

def eqBits (t : ClifTy) (x y : UInt64) : Bool :=
  match t with
  | .f32 => f32 x == f32 y
  | _ => f64 x == f64 y

/-- `fmax` (`wantMax`) and `fmin`, sharing everything but which end they take.
    NaN wins outright; otherwise equal values are split by sign, which is how
    the two zeros get ordered. -/
def pickExtreme (wantMax : Bool) (t : ClifTy) (x y : UInt64) : UInt64 :=
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

def cmpF32 : FloatCC → Float32 → Float32 → Bool
  | .eq, x, y => x == y | .ne, x, y => !(x == y)
  | .lt, x, y => x < y  | .le, x, y => x ≤ y
  | .gt, x, y => y < x  | .ge, x, y => y ≤ x

def cmpF64 : FloatCC → Float → Float → Bool
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
def satToUint (ty : ClifTy) (x : Float) : V :=
  let hi := widthMask ty
  if x != x || x ≤ 0.0 then .sc ty 0
  else if x ≥ Float.ofNat hi.toNat then .sc ty hi
  else norm ty (UInt64.ofNat x.toUInt64.toNat)

/-- The integer operations of `IBin` on two values of one width. `none` where
    the instruction traps: a zero divisor, and `sdiv` of the most negative value
    by `-1`, whose quotient does not fit. `srem` of that pair is `0`, which is
    what the instruction answers rather than a trap. -/
def ibinOp (k : IBin) (t : ClifTy) (x y : UInt64) : Option V :=
  let w := t.width
  let sx := signed t x
  let sy := signed t y
  let ux := (x &&& widthMask t).toNat
  let uy := (y &&& widthMask t).toNat
  match k with
  | .sdiv =>
      if sy == 0 then none
      else if sy == -1 && sx == -((1 : Int) <<< (w - 1)) then none
      else some (ofInt t (sx.tdiv sy))
  | .urem => if uy == 0 then none else some (norm t (UInt64.ofNat (ux % uy)))
  | .srem => if sy == 0 then none else some (ofInt t (sx.tmod sy))
  | .smin => some (ofInt t (min sx sy))
  | .smax => some (ofInt t (max sx sy))
  | .umin => some (norm t (UInt64.ofNat (min ux uy)))
  | .umax => some (norm t (UInt64.ofNat (max ux uy)))
  | .umulhi => some (norm t (UInt64.ofNat ((ux * uy) >>> w)))
  | .smulhi => some (ofInt t ((sx * sy) >>> w))

/-- The shift-shaped operations, the amount taken modulo the width. -/
def ishiftOp (k : IShift) (t : ClifTy) (x y : UInt64) : V :=
  let w := t.width
  let n := (y.toNat) % w
  let ux := (x &&& widthMask t).toNat
  match k with
  | .sshr => ofInt t (signed t x >>> n)
  | .rotl => norm t (UInt64.ofNat ((ux <<< n) % (1 <<< w) ||| (ux >>> (w - n))))
  | .rotr => norm t (UInt64.ofNat ((ux >>> n) ||| (ux <<< (w - n)) % (1 <<< w)))

/-- The bits of `x` from the top of a `w`-bit word down, most significant first. -/
def bitsDown (w : Nat) (x : Nat) : List Bool :=
  (List.range w).reverse.map fun i => x.testBit i

/-- The one-operand integer operations. `bswap` has no 8-bit form. -/
def iunOp (k : IUn) (t : ClifTy) (x : UInt64) : Option V :=
  let w := t.width
  let ux := (x &&& widthMask t).toNat
  match k with
  | .bnot => some (norm t (~~~x))
  | .iabs => some (ofInt t (signed t x).natAbs)
  | .clz => some (norm t (UInt64.ofNat ((bitsDown w ux).takeWhile (· == false)).length))
  | .bswap =>
      if w < 16 then none else
      let bytes := (List.range (w / 8)).map fun i => (ux >>> (8 * i)) % 256
      some (norm t (UInt64.ofNat (bytes.foldl (fun acc b => acc * 256 + b) 0)))
  | .bitrev =>
      some (norm t (UInt64.ofNat ((List.range w).foldl
        (fun acc i => if ux.testBit i then acc ||| (1 <<< (w - 1 - i)) else acc) 0)))

/-- The sign bit of a float lane of type `t`. -/
def fsignMask (t : ClifTy) : UInt64 :=
  if t == .f64 then 0x8000000000000000 else 0x80000000

/-- A float operation applied to a scalar, or lane-wise to an `f32x4`, given
    how it acts on one lane's bits. -/
def mapFBits (u : V) (f : ClifTy → UInt64 → UInt64) : Option V :=
  match u with
  | .sc .f32 x => some (.sc .f32 (f .f32 x &&& widthMask .f32))
  | .sc .f64 x => some (.sc .f64 (f .f64 x))
  | .vec .f32x4 xs => some (.vec .f32x4 (xs.map (fun x => f .f32 x &&& widthMask .f32)))
  | _ => none

/-- Round half to even, from the half-away-from-zero `round` the C library has:
    at a tie, twice the rounding of the half is the even neighbour. -/
def nearest64 (x : Float) : Float :=
  let t := if x < 0 then x.ceil else x.floor
  if (x - t).abs == 0.5 then 2.0 * (x / 2.0).round else x.round

def nearest32 (x : Float32) : Float32 :=
  let t := if x < 0 then x.ceil else x.floor
  if (x - t).abs == 0.5 then 2.0 * (x / 2.0).round else x.round

/-- The rounding operations on one lane. `trunc` is `ceil` below zero and
    `floor` elsewhere, which keeps the sign of a zero it produces. -/
def funOp (k : FUn) (t : ClifTy) (x : UInt64) : UInt64 :=
  let sm := fsignMask t
  match k with
  | .fabs => x &&& ~~~sm
  | _ =>
    if t == .f64 then
      let v := f64 x
      ofF64 <| match k with
        | .sqrt => v.sqrt | .ceil => v.ceil | .floor => v.floor
        | .trunc => if v < 0 then v.ceil else v.floor
        | .nearest => nearest64 v | .fabs => v
    else
      let v := f32 x
      ofF32 <| match k with
        | .sqrt => v.sqrt | .ceil => v.ceil | .floor => v.floor
        | .trunc => if v < 0 then v.ceil else v.floor
        | .nearest => nearest32 v | .fabs => v

/-- A binary IEEE format by its exponent and fraction widths. -/
structure FFmt where
  ebits : Nat
  mbits : Nat

def FFmt.bias (f : FFmt) : Nat := 2 ^ (f.ebits - 1) - 1
def FFmt.expAll (f : FFmt) : Nat := 2 ^ f.ebits - 1
def FFmt.signAt (f : FFmt) : Nat := 2 ^ (f.ebits + f.mbits)

def fmtOf (t : ClifTy) : FFmt := if t == .f64 then ⟨11, 52⟩ else ⟨8, 23⟩

/-- A float's bits read exactly: NaN, an infinity, or `±m · 2^e`. -/
inductive FVal where
  | nan
  | inf (neg : Bool)
  | fin (neg : Bool) (m : Nat) (e : Int)

def FFmt.decode (f : FFmt) (x : Nat) : FVal :=
  let neg := (x / f.signAt) % 2 == 1
  let ex := (x / 2 ^ f.mbits) % 2 ^ f.ebits
  let man := x % 2 ^ f.mbits
  if ex == f.expAll then (if man == 0 then .inf neg else .nan)
  else if ex == 0 then .fin neg man (1 - (f.bias : Int) - f.mbits)
  else .fin neg (man + 2 ^ f.mbits) ((ex : Int) - f.bias - f.mbits)

/-- `±n · 2^e` rounded to nearest, ties to even, into the format's bits:
    subnormals, overflow to infinity and underflow to a signed zero included. -/
def FFmt.round (f : FFmt) (neg : Bool) (n : Nat) (e : Int) : Nat :=
  let sgn := if neg then f.signAt else 0
  if n == 0 then sgn else
  let top : Int := (n.log2 : Int) + e
  let emin : Int := 1 - (f.bias : Int)
  -- the exponent of the result's last place
  let q : Int := max (top - f.mbits) (emin - f.mbits)
  let (m, q) :=
    if q ≤ e then (n * 2 ^ (e - q).toNat, q)
    else
      let sh := (q - e).toNat
      let m := n / 2 ^ sh
      let r := n % 2 ^ sh
      let half := 2 ^ sh / 2
      let m := if r > half || (r == half && m % 2 == 1) then m + 1 else m
      if m == 2 ^ (f.mbits + 1) then (m / 2, q + 1) else (m, q)
  if m < 2 ^ f.mbits then sgn + m
  else
    let be := q + f.mbits + f.bias
    if be ≥ f.expAll then sgn + f.expAll * 2 ^ f.mbits
    else sgn + be.toNat * 2 ^ f.mbits + (m - 2 ^ f.mbits)

/-- `a · b + c` computed exactly and rounded once, on one lane's bits. A NaN
    operand gives that NaN quietened; `∞ · 0` and `∞ - ∞` give the machine's
    invalid-operation NaN, the one its own `0 / 0` produces. -/
def fmaLane (t : ClifTy) (a b c : UInt64) : UInt64 :=
  let f := fmtOf t
  let quiet (x : UInt64) : UInt64 := x ||| UInt64.ofNat (2 ^ (f.mbits - 1))
  let invalid : UInt64 :=
    if t == .f64 then ofF64 (0.0 / 0.0) else ofF32 ((0.0 : Float32) / 0.0)
  match f.decode a.toNat, f.decode b.toNat, f.decode c.toNat with
  | .nan, _, _ => quiet a
  | _, .nan, _ => quiet b
  | _, _, .nan => quiet c
  | .inf _, .fin _ 0 _, _ | .fin _ 0 _, .inf _, _ => invalid
  | .inf sa, .inf sb, cv | .inf sa, .fin sb _ _, cv | .fin sa _ _, .inf sb, cv =>
      let sp := sa != sb
      let infP := UInt64.ofNat ((if sp then f.signAt else 0) + f.expAll * 2 ^ f.mbits)
      match cv with
      | .inf sc => if sc != sp then invalid else infP
      | _ => infP
  | .fin _ _ _, .fin _ _ _, .inf _ => c
  | .fin sa ma ea, .fin sb mb eb, .fin sc mc ec =>
      let sp := sa != sb
      let ep := ea + eb
      let e := min ep ec
      let p : Int := ((ma * mb * 2 ^ (ep - e).toNat : Nat) : Int)
      let cc : Int := ((mc * 2 ^ (ec - e).toNat : Nat) : Int)
      let sum := (if sp then -p else p) + (if sc then -cc else cc)
      if sum == 0 then
        let neg := ma * mb == 0 && mc == 0 && sp == sc && sp
        UInt64.ofNat (if neg then f.signAt else 0)
      else UInt64.ofNat (f.round (sum < 0) sum.natAbs e)

/-- Saturating float-to-signed: NaN gives zero, and everything outside the
    type's range its nearest end. -/
def satToSint (ty : ClifTy) (x : Float) : V :=
  let w := ty.width
  let lo : Int := -((1 : Int) <<< (w - 1))
  let hi : Int := ((1 : Int) <<< (w - 1)) - 1
  if x != x then .sc ty 0
  else if x ≤ Float.ofInt lo then ofInt ty lo
  else if x ≥ Float.ofInt (hi + 1) then ofInt ty hi
  else if x ≥ 0 then ofInt ty x.toUInt64.toNat
  else ofInt ty (-((-x).toUInt64.toNat : Int))

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
      -- Rounded once, straight to the target width: going through `f64`
      -- first rounds an `i64` twice.
      match ty with
      | .f32 => some (.sc .f32 (ofF32 (Float32.ofInt (signed t x))))
      | .f64 => some (.sc .f64 (ofF64 (Float.ofInt (signed t x))))
      | _ => none
  | .fcvtToUint ty a => do
      if !(ty.isInt && ty.width ≥ 32) then none else
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
  | .ibin k a b => bin Γ a b (ibinOp k)
  | .ishift k a b => shiftBin Γ a b fun t x y => some (ishiftOp k t x y)
  | .iun k a => un Γ a (·.isInt) (iunOp k)
  | .fbin k a b => do
      match k with
      | .fdiv => zipF (← get Γ a) (← get Γ b) (· / ·) (· / ·)
      | .fcopysign => zipBits (← get Γ a) (← get Γ b) fun t x y =>
          let sm := fsignMask t
          (x &&& ~~~sm) ||| (y &&& sm)
  | .fun1 k a => do mapFBits (← get Γ a) (funOp k)
  | .fma a b c => do
      match ← get Γ a, ← get Γ b, ← get Γ c with
      | .sc t x, .sc t' y, .sc t'' z =>
          if t.isFloat && t == t' && t == t'' then some (.sc t (fmaLane t x y z)) else none
      | .vec .f32x4 xs, .vec .f32x4 ys, .vec .f32x4 zs =>
          if xs.size == ys.size && xs.size == zs.size then
            some (.vec .f32x4 ((xs.zip (ys.zip zs)).map fun (x, y, z) => fmaLane .f32 x y z))
          else none
      | _, _, _ => none
  | .iext k ty a => do
      let (.sc t x) ← get Γ a | none
      if !k.admits t ty then none else
      match k with
      | .reduce => some (norm ty x)
      | .uextend => some (.sc ty (x &&& widthMask t))
      | .sextend => some (ofInt ty (signed t x))
  | .fconv k ty a => do
      match k, ← get Γ a with
      | .toSint, .sc .f32 x =>
          if ty.isInt && ty.width ≥ 32 then some (satToSint ty (f32 x).toFloat) else none
      | .toSint, .sc .f64 x =>
          if ty.isInt && ty.width ≥ 32 then some (satToSint ty (f64 x)) else none
      | .fromUint, .sc t x =>
          if !(t.isInt && t.width ≥ 32) then none else
          let n := (x &&& widthMask t).toNat
          match ty with
          | .f32 => some (.sc .f32 (ofF32 (Float32.ofNat n)))
          | .f64 => some (.sc .f64 (ofF64 (Float.ofNat n)))
          | _ => none
      | .demote, .sc .f64 x => if ty == .f32 then some (.sc .f32 (ofF32 (f64 x).toFloat32)) else none
      | _, _ => none
  | .load op a => do
      let (.sc _ addr) ← get Γ a | none
      match op.kind with
      | .uload8 => do
          let b ← m.load addr 1
          some (norm op.ty b)
      | .uload16 => do
          let b ← m.load addr 2
          some (norm op.ty b)
      | .uload32 => do
          let b ← m.load addr 4
          some (norm op.ty b)
      | .sload8 => do
          let b ← m.load addr 1
          some (ofInt op.ty (signed .i8 b))
      | .sload16 => do
          let b ← m.load addr 2
          some (ofInt op.ty (signed .i16 b))
      | .sload32 => do
          let b ← m.load addr 4
          some (ofInt op.ty (signed .i32 b))
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

theorem reloc3_bitselect : Reloc3 .bitselect := by
  intro m Γ a b c x y z ha hb hc; simp [evalOp, get, ha, hb, hc]

theorem reloc3_fma : Reloc3 .fma := by
  intro m Γ a b c x y z ha hb hc; simp [evalOp, get, ha, hb, hc]

theorem reloc2_ibin (k : IBin) : Reloc2 (Op.ibin k) := by
  intro m Γ a b x y ha hb; simp [evalOp, bin, get, ha, hb]
theorem reloc2_ishift (k : IShift) : Reloc2 (Op.ishift k) := by
  intro m Γ a b x y ha hb; simp [evalOp, shiftBin, get, ha, hb]
theorem reloc2_fbin (k : FBin) : Reloc2 (Op.fbin k) := by
  intro m Γ a b x y ha hb; simp [evalOp, get, ha, hb]
theorem reloc1_iun (k : IUn) : Reloc1 (Op.iun k) := by
  intro m Γ a x ha; simp [evalOp, un, get, ha]
theorem reloc1_fun1 (k : FUn) : Reloc1 (Op.fun1 k) := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_fconv (k : FConv) (ty : ClifTy) : Reloc1 (Op.fconv k ty) := by
  intro m Γ a x ha; simp [evalOp, get, ha]
theorem reloc1_iext (k : IExt) (ty : ClifTy) : Reloc1 (Op.iext k ty) := by
  intro m Γ a x ha; simp [evalOp, get, ha]

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
  | call  (callee : Callee) (args : List V)
  | store (addr : UInt64) (width : Nat) (bits : UInt64)
  deriving Repr, BEq

/-- The hash table `Lib.Ht` keeps, which the program reaches only through its
    entry points. Every accessor reads table `0`, so the handle `ht_create`
    returns only counts, and the accessors work once the first has been made;
    entries are kept, and `ht_get_entry` walks them, in the order they were
    first inserted. -/
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

/-- The CUDA context: a Rust allocation, like the hash table's, so it can be
    stored and passed but never loaded through. -/
def cudaCtx : UInt64 := 0x6000000000

/-- One kernel launch as the host requests it: the kernel text, its entry
    point, the buffers bound to its parameters in order, and the grid and
    block dimensions. -/
structure Launch where
  kernel : String
  entry : String
  binds : List Nat
  dims : List UInt64
  /-- Every parameter word, in order, for a launch through the driver: the
      pointers that bind buffers and the scalars alike. -/
  args : List UInt64 := []

/-- One vendor routine call as the model sees it: which routine, and its scalar
    arguments in the order the entry point takes them. -/
structure VendorCall where
  op : String
  scalars : List UInt64

-- ---------------------------------------------------------------------------
-- Streams: when a buffer may be touched
--
-- A created stream is non-blocking, so nothing orders it against the default
-- stream or another created stream except what the program says: an event
-- recorded on one and waited on by another, a stream synchronised by the host,
-- and the fences the runtime puts at a stream's creation and destruction. The
-- model runs every operation when it is issued, which is one of the orders the
-- device may choose; it is the only order when conflicting accesses are
-- ordered, and a conflict they leave unordered is a race — stuck.
-- ---------------------------------------------------------------------------

/-- A vector clock over the parties that do device work: entry `p` is how many
    of `p`'s operations are known to have happened. -/
abbrev Clock := Array Nat

def Clock.get (c : Clock) (p : Nat) : Nat := c.getD p 0

def Clock.join (a b : Clock) : Clock :=
  (Array.range (max a.size b.size)).map fun i => max (a.get i) (b.get i)

def Clock.bump (c : Clock) (p : Nat) : Clock :=
  let c := if p < c.size then c else c ++ Array.replicate (p + 1 - c.size) 0
  c.set! p (c.get p + 1)

/-- The host, which issues everything and waits when it synchronises. -/
def hostParty : Nat := 0
/-- The default stream: `cl_cuda_launch`, the synchronous copies, the default
    cuBLAS handle, and buffer frees. -/
def defaultParty : Nat := 1
/-- Created stream `id`. -/
def streamParty (id : Nat) : Nat := 2 + id

/-- Grow an array to hold index `i`. -/
def growTo {α : Type} (xs : Array α) (i : Nat) (d : α) : Array α :=
  if i < xs.size then xs else xs ++ Array.replicate (i + 1 - xs.size) d

/-- Every party's clock, and for every buffer the last write and the reads
    since, each as `(party, count)`. -/
structure Race where
  clocks : Array Clock := #[]
  lastW : Array (Option (Nat × Nat)) := #[]
  reads : Array (List (Nat × Nat)) := #[]
  deriving Inhabited

def Race.clock (r : Race) (p : Nat) : Clock := r.clocks.getD p #[]

def Race.setClock (r : Race) (p : Nat) (c : Clock) : Race :=
  { r with clocks := (growTo r.clocks p #[]).set! p c }

/-- `p` learns everything `c` knows. -/
def Race.joinInto (r : Race) (p : Nat) (c : Clock) : Race := r.setClock p ((r.clock p).join c)

/-- The host enqueues one operation on `p`: it follows everything the host has
    seen and `p`'s own earlier work. The operation's clock. -/
def Race.issue (r : Race) (p : Nat) : Race × Clock :=
  let c := ((r.clock p).join (r.clock hostParty)).bump p
  (r.setClock p c, c)

/-- An access by `p` knowing `c` comes after the earlier access `e`. -/
def ordered (p : Nat) (c : Clock) (e : Nat × Nat) : Bool := e.1 == p || decide (e.2 ≤ c.get e.1)

/-- Check an access by `p` knowing `c` that reads `rs` and writes `ws` against
    every earlier access, and record it: `none` is a race. -/
def Race.access (r : Race) (p : Nat) (c : Clock) (rs ws : List Nat) : Option Race :=
  let ok := (rs ++ ws).all (fun b => ((r.lastW.getD b none).map (ordered p c)).getD true)
    && ws.all (fun b => (r.reads.getD b []).all (ordered p c))
  if !ok then none
  else
    let e := (p, c.get p)
    let r := rs.foldl (fun r b => { r with reads := (growTo r.reads b []).modify b (e :: ·) }) r
    some (ws.foldl (fun r b => { r with lastW := (growTo r.lastW b none).set! b (some e),
                                         reads := (growTo r.reads b []).set! b [] }) r)

/-- One device operation on `p`, issued and checked. -/
def Race.op (r : Race) (p : Nat) (rs ws : List Nat) : Option Race :=
  let (r, c) := r.issue p
  r.access p c rs ws

/-- The host waits for `p`. -/
def Race.sync (r : Race) (p : Nat) : Race := r.joinInto hostParty (r.clock p)

/-- What a captured graph does when it runs, in the order it was recorded. -/
inductive DevOp where
  | launch (l : Launch) (binds : List Nat)
  | vendor (c : VendorCall) (ins : List Nat) (out : Nat)

/-- A capture in progress: the stream it began on, the streams that have joined
    it by waiting on its events, the ordering among what they recorded — checked
    as it is recorded, so a graph that ran its nodes in another order could not
    tell — and the operations. -/
structure Capture where
  origin : Nat
  joined : List Nat
  race : Race
  ops : Array DevOp

/-- The device as the host observes it: whether a context is live, and every
    buffer `cl_cuda_create_buffer` has handed out, at the index it returned. A
    freed buffer keeps its slot, empty, so an id is never reused. -/
structure Dev where
  live : Bool := false
  bufs : Array (Option ByteArray) := #[]
  /-- Pinned host allocations by the id `cl_cuda_pinned_alloc` returned:
      `(offset, length)` in `Mem.pinned`, empty once freed. -/
  pinned : Array (Option (Nat × Nat)) := #[]
  /-- Created streams, alive or destroyed, by id. -/
  streams : Array Bool := #[]
  /-- Created events by id: never recorded, or recorded with the clock of the
      point they mark and whether that point is inside the capture. -/
  events : Array (Option (Option (Clock × Bool))) := #[]
  race : Race := {}
  capture : Option Capture := none
  graphs : Array (Option (Array DevOp)) := #[]
  /-- Through the driver: whether `cuInit` has run, how many times the
      primary context is retained, the loaded modules' PTX by handle index,
      and each function as its module's index and entry name. -/
  drvInit : Bool := false
  retained : Nat := 0
  modules : Array (Option String) := #[]
  funcs : Array (Nat × String) := #[]
  /-- Live cuBLAS handles by index, each with the stream it runs on (`0` for
      the default stream). -/
  blas : Array (Option UInt64) := #[]
  /-- Instantiated graphs by handle index, each the operations it runs. -/
  execs : Array (Option (Array DevOp)) := #[]
  deriving Inhabited

/-- The live buffer an `i32` id names, if any. -/
def Dev.get? (d : Dev) (id : Int) : Option ByteArray :=
  if id < 0 then none else (d.bufs[id.toNat]?).join

/-- Replace buffer `id`'s contents. -/
def Dev.put (d : Dev) (id : Nat) (b : Option ByteArray) : Dev :=
  { d with bufs := d.bufs.setIfInBounds id b }

/-- The wgpu context: a Rust allocation, stored and passed, never loaded. -/
def gpuCtx : UInt64 := 0x7000000000

/-- A wgpu compute pipeline: its WGSL source and, binding by binding, the
    buffer it binds and whether the binding is read-only. -/
structure Pipe where
  shader : String
  binds : List (Nat × Bool)

/-- One dispatch as the host requests it: the pipeline's shader and bindings,
    and the workgroup counts. -/
structure Dispatch where
  shader : String
  binds : List (Nat × Bool)
  groups : List UInt64

/-- The wgpu device as the host observes it. `pending` is what `cl_gpu_dispatch`
    has recorded and not yet submitted: the runtime submits it at the next
    dispatch or download, and `queue.write_buffer` lands before that submit —
    so an upload made after a dispatch and before the flush is one the dispatch
    sees. -/
structure Wgpu where
  live : Bool := false
  bufs : Array ByteArray := #[]
  pipes : Array Pipe := #[]
  pending : List (Nat × List UInt64) := []
  deriving Inhabited

/-- The LMDB context: a Rust allocation, stored and passed, never loaded. -/
def lmdbCtx : UInt64 := 0x8000000000

/-- What an LMDB database holds, in the order LMDB's default comparison sorts
    keys: byte by byte, a key before every longer key it begins. -/
abbrev LmdbTable := List (ByteArray × ByteArray)

/-- One environment the context opened: its directory and, while a write
    transaction is open on it, what that transaction sees. -/
structure LmdbEnv where
  path : String
  txn : Option LmdbTable := none

/-- The LMDB context as the host observes it: the environments by the handle
    `cl_lmdb_open` returned, which is their position. -/
structure Lmdb where
  live : Bool := false
  envs : Array LmdbEnv := #[]
  deriving Inhabited

/-- Where the `k`th piece of loaded machine code lives: outside every region,
    so the program can call it (`Callee.native`) and never load through it. -/
def nativeAddr (k : Nat) : UInt64 := 0xA000000000 + 0x10000 * UInt64.ofNat k

/-- The thread context: a Rust allocation, stored and passed, never loaded. -/
def threadCtx : UInt64 := 0xB000000000

/-- The thread context as the host observes it: whether one is live, the handle
    the next spawn answers, and the one worker spawned and not yet joined. -/
structure Threads where
  live : Bool := false
  next : Nat := 1
  outstanding : Option Nat := none
  deriving Inhabited

/-- The window context: a Rust allocation, stored and passed, never loaded. -/
def winCtx : UInt64 := 0x9000000000

/-- One input event as `cl_window_poll` writes it: kind, then three operands. -/
structure WinEvent where
  kind : UInt64
  a : UInt64 := 0
  b : UInt64 := 0
  c : UInt64 := 0

/-- The window context as the host observes it: whether one is live, whether
    its one window is open and with which blit shader, and the events that
    have arrived and not been polled. -/
structure Win where
  live : Bool := false
  isOpen : Bool := false
  /-- The blit shader the window was opened with, compiled at the first
      present. -/
  blit : String := ""
  pending : List WinEvent := []
  deriving Inhabited

-- ---------------------------------------------------------------------------
-- wgpu and the window library, called directly
-- ---------------------------------------------------------------------------

/-- A handle wgpu hands out: object `i` of the program's, never `0` and never
    an address in a region. -/
def wgHandle (i : Nat) : UInt64 := UInt64.ofNat (2 ^ 56 + 10 * 2 ^ 48 + i + 1)

/-- The object a handle names, if it is one of wgpu's. -/
def wgIndex (h : UInt64) : Option Nat :=
  if 2 ^ 56 + 10 * 2 ^ 48 + 1 ≤ h.toNat ∧ h.toNat < 2 ^ 56 + 11 * 2 ^ 48 then
    some (h.toNat - (2 ^ 56 + 10 * 2 ^ 48 + 1))
  else none

/-- A command an encoder has recorded, by the objects it names. -/
inductive WgCmd where
  /-- `n` bytes from buffer `src` at `soff` to buffer `dst` at `doff`. -/
  | copy (src soff dst doff n : Nat)
  /-- A dispatch of `pipe` with bind groups `(index, group)`, of `groups` workgroups. -/
  | dispatch (pipe : Nat) (bound : List (Nat × Nat)) (groups : List UInt64)
  /-- A draw into `view` with bind group `group` at index 0. -/
  | draw (view : Nat) (group : Option Nat)

/-- What a wgpu object is, as the program made it. Objects of one device are
    not told apart from another's: the program is taken to use one. -/
inductive WgData where
  | instance
  | adapter
  /-- The error scopes pushed and not popped, innermost first: whether each
      has caught an error. -/
  | device (scopes : List Bool)
  | queue
  | buffer (usage : UInt64) (bytes : ByteArray)
  | shader (src : String) (valid : Bool)
  /-- Bindings as `(binding, read-only)`; `none` for a layout a render
      pipeline derived from its shader, which the model does not read. -/
  | bgl (entries : Option (List (Nat × Bool)))
  | playout (groups : List (List (Nat × Bool)))
  | cpipe (src : String) (groups : List (List (Nat × Bool))) (valid : Bool)
  /-- Bindings as `(binding, buffer)`. -/
  | bgroup (entries : List (Nat × Nat))
  /-- `open` while it accepts commands: no pass is open on it and it is not
      finished. -/
  | encoder (cmds : List WgCmd) (isOpen : Bool)
  | cpass (enc : Nat) (pipe : Option Nat) (bound : List (Nat × Nat)) (ended : Bool)
  | cmdbuf (cmds : List WgCmd) (used : Bool)
  /-- Configured or not, and the texture acquired and not yet presented. -/
  | surface (configured : Bool) (current : Option Nat)
  /-- The buffer a draw last put on it. -/
  | texture (surf : Nat) (drawn : Option ByteArray)
  | view (tex : Nat)
  | rpipe (valid : Bool)
  | rpass (enc view : Nat) (pipe : Option Nat) (group : Option Nat) (ended : Bool)
  /-- An error scope of device `dev`, the `level`th pushed on it and not
      popped. -/
  | scope (dev level : Nat)
  deriving Inhabited

def WgData.kind : WgData → IR.WgObj
  | .instance => .instance | .adapter => .adapter | .device _ => .device | .queue => .queue
  | .buffer .. => .buffer | .shader .. => .shaderModule | .bgl _ => .bindGroupLayout
  | .playout _ => .pipelineLayout | .cpipe .. => .computePipeline | .bgroup _ => .bindGroup
  | .encoder .. => .commandEncoder | .cpass .. => .computePass | .cmdbuf .. => .commandBuffer
  | .surface .. => .surface | .texture .. => .texture | .view _ => .textureView
  | .rpipe _ => .renderPipeline | .rpass .. => .renderPass | .scope .. => .errorScope

/-- wgpu as the program has made it: every object by the handle it was
    answered as, and whether the program still holds it. A released object
    stays while another holds it — a bind group its buffers — as wgpu keeps
    it. -/
structure WgState where
  objs : Array (Bool × WgData) := #[]
  deriving Inhabited

/-- The object `h` names, if the program holds it and it is of kind `k`. -/
def WgState.get? (s : WgState) (k : IR.WgObj) (h : UInt64) : Option (Nat × WgData) := do
  let i ← wgIndex h
  let (live, d) ← s.objs[i]?
  if live && d.kind == k then some (i, d) else none

/-- A new object, and its handle. -/
def WgState.push (s : WgState) (d : WgData) : UInt64 × WgState :=
  (wgHandle s.objs.size, { objs := s.objs.push (true, d) })

def WgState.set (s : WgState) (i : Nat) (d : WgData) : WgState :=
  { objs := s.objs.modify i fun (l, _) => (l, d) }

def WgState.data? (s : WgState) (i : Nat) : Option WgData := (s.objs[i]?).map (·.2)

/-- Where the `k`th window the window library opens lives: outside every
    region, so the program can pass it and never load through it. -/
def wlWinAddr (k : Nat) : UInt64 := 0xC000000000 + 0x10000 * UInt64.ofNat k

/-- A window the window library opened: whether it is open, the size it was
    opened at, and its events pumped and not yet polled, as 32-byte records. -/
structure WlWin where
  live : Bool := true
  width : Int
  height : Int
  pending : List ByteArray := []
  deriving Inhabited

/-- Where the `k`th serial port opened lives: outside every region. -/
def serPortAddr (k : Nat) : UInt64 := 0xC800000000 + 0x10000 * UInt64.ofNat k

/-- A serial port the program opened: whether it is open, the path it was
    opened at, what the device has sent and not yet been read, and what the
    program has written to it. -/
structure SerPort where
  live : Bool := true
  path : String
  incoming : ByteArray
  sent : ByteArray := ByteArray.empty

/-- Where the `k`th USB device opened lives: outside every region. -/
def usbDevAddr (k : Nat) : UInt64 := 0xD000000000 + 0x10000 * UInt64.ofNat k

/-- A USB transfer as the device sees it: a control transfer's SETUP fields
    and data length, or a bulk or interrupt transfer's endpoint and length,
    with the bytes sent to the device (empty for one from it). -/
inductive UsbXfer where
  | control (reqType request value index len : Nat) (out : ByteArray)
  | bulk (endpoint len : Nat) (out : ByteArray)
  | interrupt (endpoint len : Nat) (out : ByteArray)

def UsbXfer.len : UsbXfer → Nat
  | .control _ _ _ _ l _ | .bulk _ l _ | .interrupt _ l _ => l

/-- A USB device the system lists: what identifies it, whether this process
    may open it, which interfaces it lets be claimed, and each endpoint's
    packet size. -/
structure UsbDevice where
  bus : Int
  address : Int
  vendor : Int
  product : Int
  openable : Bool := false
  claimable : List Nat := []
  packet : Nat → Nat := fun _ => 64

/-- A USB device the program opened: whether it is open, which listed device
    it is, and the interfaces claimed through it. -/
structure UsbOpen where
  live : Bool := true
  device : Nat
  claimed : List Nat := []

/-- The window library as the program has made it: whether this thread's
    event loop is up, and the windows by opening order. -/
structure WlState where
  up : Bool := false
  windows : Array WlWin := #[]
  deriving Inhabited

structure World where
  mem : Mem
  fs  : FS := {}
  /-- Lines `cl_stdin_readline` will return, in order, each including its
      newline as `read_line` leaves it. An empty list is end of input. -/
  stdin : List ByteArray := []
  /-- What `cl_stdout_write` has written so far. -/
  stdout : ByteArray := ByteArray.empty
  ht : Ht := {}
  /-- The C library's live heap allocations, as `(offset, length)` in the
      host-allocation region, which holds every allocation the host makes,
      page-locked or not: these are the ones `free` may release. -/
  heap : List (Nat × Nat) := []
  lmdb : Lmdb := {}
  /-- What each LMDB directory holds, committed: the part that outlives the
      context. -/
  lmdbDisk : List (String × LmdbTable) := []
  dev : Dev := {}
  /-- **What a kernel computes**: the new contents of the buffers bound to a
      launch, from their contents before it. The model does not run PTX, so
      this is the oracle a claim about a program that launches kernels is
      stated over; a kernel's own proof is what pins it down. A launch can
      change only the buffers bound to it, by construction. -/
  kernel : Launch → List ByteArray → List ByteArray := fun _ bs => bs
  /-- **What a vendor routine computes**: the new contents of its one output
      buffer, from the buffers it reads, the output last since a `beta` term
      reads it. The laws a claim assumes (`Law.cublasIsMatvec`, …) are what pin
      it down; the model only says which buffer it may change. -/
  vendor : VendorCall → List ByteArray → ByteArray := fun _ bs => bs.getLastD ByteArray.empty
  /-- **What a fresh pinned allocation holds**, by its index and length: the
      bytes are whatever the host page held, which nothing here computes. -/
  pinnedFill : Nat → Nat → ByteArray := fun _ n => ByteArray.mk (Array.replicate n 0)
  /-- **Device memory, free and total**, as `cuMemGetInfo` would report it. -/
  memInfo : UInt64 × UInt64 := (0, 0)
  /-- **Which C libraries loaded**: a program runs on machines with and
      without each, so a claim about one that calls a library is stated for
      whichever this is. -/
  present : IR.Lib → Bool := fun _ => true
  /-- How many CUDA devices the driver reports. -/
  devices : Nat := 1
  /-- **What a fresh device allocation holds**, by its index and length:
      `cuMemAlloc` does not clear it, so nothing here computes it. -/
  devFill : Nat → Nat → ByteArray := fun _ n => ByteArray.mk (Array.replicate n 0)
  /-- **Where each device buffer lives**, by id: what `cl_cublas_ptr_array`
      writes and `cl_cublas_sgemm_batched_on_stream` follows. Live buffers do
      not overlap, as the driver's allocations do not. -/
  devAddr : Nat → UInt64 := fun id => (0x700000000000 : UInt64) + (UInt64.ofNat id <<< 32)
  /-- **Time between two completed events**, as the bits of an `f32` count of
      milliseconds, by their ids. -/
  elapsed : Nat → Nat → UInt32 := fun _ _ => 0
  gpu : Wgpu := {}
  win : Win := {}
  thread : Threads := {}
  /-- **Whether there is a display**: without one, the window library's event
      loop does not start. -/
  display : Bool := false
  /-- **Whether there is a CUDA device**: without one, `cudaInit` stores null. -/
  cudaDevice : Bool := true
  /-- **Whether wgpu finds an adapter**: without one, `gpuInit` stores null. -/
  gpuAdapter : Bool := true
  /-- **Whether WGSL compiles**: shader sources the device accepts. -/
  wgslOk : String → Bool := fun _ => true
  /-- **What arrives**: each time the runtime pumps the event loop — at a poll
      and at a present — the next batch here joins the pending events. -/
  winInput : List (List WinEvent) := []
  /-- The buffers presented to the window, oldest first. -/
  shown : List ByteArray := []
  /-- Machine code `cl_native_load` placed, by load order, empty once freed.
      The `k`th load answers `nativeAddr k`. -/
  native : Array (Option ByteArray) := #[]
  /-- **The host's architecture**, as `cl_native_arch` numbers it: 1 for
      x86-64, 2 for AArch64, 0 for any other. -/
  arch : Int := 1
  /-- **What the CPU has**, by Rust's spelling of the feature: `none` for a
      name the runtime does not know. -/
  cpuHas : String → Option Bool := fun _ => none
  /-- Whether the system grants a performance control — `"lock"`, `"unlock"`,
      `"hugepages"`, `"priority"` — which depends on its limits and
      privileges, not on the program. -/
  grants : String → Bool := fun _ => true
  /-- **The logical CPUs online**, each with its physical core and package
      as the system reports them, `-1` where it does not. -/
  cpus : List (Int × Int) := [(0, 0)]
  /-- **Whether the system pins a thread to a CPU** when asked. -/
  cpuPins : Bool := true
  /-- **The serial ports the system lists**, by name. -/
  serialNames : List String := []
  /-- **The device at a path**, if one opens there: what it has sent. -/
  serialDevices : String → Option ByteArray := fun _ => none
  ser : Array SerPort := #[]
  /-- **The USB devices the system lists**, in order. -/
  usbDevices : List UsbDevice := []
  /-- **What listed device `i` answers a transfer**: the bytes it sends back
      (none asked for, for one to it), or `none` where the transfer fails. -/
  usbReply : Nat → UsbXfer → Option ByteArray := fun _ _ => none
  /-- **Whether a control transfer needs a claimed interface**: on Windows. -/
  usbControlNeedsClaim : Bool := false
  usb : Array UsbOpen := #[]
  /-- **What a compute shader computes**: the new contents of the buffers bound
      to a dispatch, from their contents before it. A read-only binding keeps
      its contents whatever this says, by construction. -/
  shader : Dispatch → List ByteArray → List ByteArray := fun _ bs => bs
  wg : WgState := {}
  /-- **Whether there is a GPU adapter** for wgpu to hand out. -/
  wgAdapter : Bool := true
  wl : WlState := {}
  /-- **What arrives through the window library**: each pump appends the next
      batch, each record to the events of the window it names by opening
      order. -/
  wlInput : List (List (Nat × ByteArray)) := []
  /-- **A window's size in pixels**, from the size it was opened at. -/
  wlPixels : Int × Int → Int × Int := id
  /-- Reversed while running; `Sem.run` hands it back in order. -/
  obs : List Obs := []
  deriving Inhabited

/-- Execution either finishes, or stops because the term did something the
    semantics does not define — an unmapped address, an undeclared callee, a
    loop that outran its budget. Stopping is never silent. -/
inductive Outcome (α : Type) where
  | ok (a : α) (w : World)
  | stuck (why : String)
  /-- A foreign call refused its arguments or the world it was made in: a use
      the program could have avoided, which `effectsOk` rules out statically. -/
  | misuse (why : String)
  /-- The program's own fault: an operation undefined on its operands (a load
      from memory it does not have, a division by zero) or a store to memory
      it does not have. -/
  | fault (why : String)

instance : Inhabited (Outcome α) := ⟨.stuck "unreachable"⟩

/-- Outcomes print as their shape, not their memory. -/
instance [Repr α] : Repr (Outcome α) where
  reprPrec o _ := match o with
    | .ok a _ => "ok " ++ repr a
    | .stuck m => "stuck: " ++ m
    | .misuse m => "misuse: " ++ m
    | .fault m => "fault: " ++ m

end AlgorithmLib.HProg.Sem
