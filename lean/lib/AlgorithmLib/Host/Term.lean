module
public import Lean
public import AlgorithmLib.Core.IR
meta import AlgorithmLib.Core.IR
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `HProg` — a function body as a first-order term

A function body as *data* — one inductive term per body, which `compileBody`
turns into the shipped `IR.FuncData`. A body that exists only while a builder
runs is a `StateM` action, and nothing can be stated about one of those.

Two things follow from the term being first-order:

* **It can be checked.** `wf` decides slot scoping *and* types by kernel
  computation. `IR.Val` carries no type, and until this check existed an
  `f32`/`i64` mix-up surfaced only when Cranelift rejected the finished
  artifact.
* **It can be observed.** `callsOf` reads the FFI calls in program order —
  the observation the compilation proof is stated against.

Nobody writes one of these by hand. `Prog` is the surface: a body with binders
and types, which `Prog.emit` folds into a term here by a total function. This
is the form that ships and the form the proofs are about, and there is only
the one.

## Control flow

Four constructs: `loop` (top-tested), `dloop` (bottom-tested, so the body block
branches to itself and there is no header), `ite`, and `br`, which leaves the
`depth`-th enclosing loop. Measured over every emitted artifact, all 455
functions have reducible control flow, and these cover the shapes the generators
emit. A loop's condition prefix is a whole `Code`, so a test that follows an
inner loop is expressible too.

## Types

Slots are typed, and the two constructs that bind block parameters (`loop`
carries and exits, `ite` joins) carry their types in the term. So `compileBody`
performs no inference: it reads the annotation. `wf` is where inference lives,
checking each annotation against the types it computes for the operands. At the
surface the annotations are the `Prog` term's own indices, so a generator never
writes one by hand and cannot write a wrong one.
-/

namespace AlgorithmLib.HProg

open AlgorithmLib.IR

/-- A slot: the n-th value defined in the function, counting its parameters.
    `dst` is the position, so the term carries no value numbering to get wrong. -/
abbrev R := Nat

-- ---------------------------------------------------------------------------
-- The term
-- ---------------------------------------------------------------------------

/-- Pure operations — one result slot each.

    A type appears only where the CLIF instruction carries one; every other
    result type follows from the operands. -/
inductive Op where
  | iconst  (ty : ClifTy) (k : Int)
  | iadd    (a b : R)
  | isub    (a b : R)
  | imul    (a b : R)
  | udiv    (a b : R)
  | ineg    (a : R)
  | ishl    (a b : R)
  | ushr    (a b : R)
  | band    (a b : R)
  | bandNot (a b : R)
  | bor     (a b : R)
  | bxor    (a b : R)
  | ireduce32 (a : R)
  | uextend64 (a : R)
  | sextend64 (a : R)
  | icmp    (cond : ICmpCond) (a b : R)
  | select  (c a b : R)
  /-- Lane-wise `c ? a : b` on the bits of `c`. -/
  | bitselect (c a b : R)
  | ctz     (a : R)
  | popcnt  (a : R)
  | fconst  (ty : ClifTy) (bits : UInt64)
  | fadd    (a b : R)
  | fsub    (a b : R)
  | fmul    (a b : R)
  | fmax    (a b : R)
  | fmin    (a b : R)
  | fneg    (a : R)
  | fpromote (a : R)
  | fcmp    (cond : FloatCC) (a b : R)
  | fcvtFromSint (ty : ClifTy) (a : R)
  | fcvtToUint (ty : ClifTy) (a : R)
  | splat   (ty : ClifTy) (a : R)
  | extractlane (a : R) (lane : Nat)
  | vhighBits (a : R)
  | bitcast (ty : ClifTy) (a : R)
  | load    (op : LoadOp) (a : R)
  | ibin    (k : IBin) (a b : R)
  | ishift  (k : IShift) (a b : R)
  | iun     (k : IUn) (a : R)
  | fbin    (k : FBin) (a b : R)
  | fun1    (k : FUn) (a : R)
  | fconv   (k : FConv) (ty : ClifTy) (a : R)
  /-- `a * b + c`, rounded once. -/
  | fma     (a b c : R)
  | iext    (k : IExt) (ty : ClifTy) (a : R)
  deriving Repr

/-- The slots an operation reads, in the order its CLIF instruction takes them.

    Used to say "every operand is in scope" without appealing to `wf`, which
    also checks types and is therefore stronger than the dynamic step needs. -/
def Op.regs : Op → List R
  | .iconst _ _ | .fconst _ _ => []
  | .iadd a b | .isub a b | .imul a b | .udiv a b
  | .ishl a b | .ushr a b | .band a b | .bandNot a b
  | .bor a b | .bxor a b | .icmp _ a b
  | .fadd a b | .fsub a b | .fmul a b | .fmax a b | .fmin a b
  | .fcmp _ a b | .ibin _ a b | .ishift _ a b | .fbin _ a b => [a, b]
  | .iun _ a | .fun1 _ a | .fconv _ _ a | .iext _ _ a
  | .ineg a | .ctz a | .popcnt a | .ireduce32 a | .uextend64 a
  | .sextend64 a | .fneg a | .fpromote a | .vhighBits a
  | .fcvtFromSint _ a | .fcvtToUint _ a | .splat _ a
  | .extractlane a _ | .bitcast _ a | .load _ a => [a]
  | .select c a b | .bitselect c a b | .fma c a b => [c, a, b]

/-- An operation's name, for diagnostics.

    A plain `String` rather than `repr`: `Repr` renders through `Std.Format`,
    whose internals are `@[extern]`, and a generator's *error path* has no
    business enlarging what a trust scan has to account for. -/
def Op.name : Op → String
  | .iconst _ _ => "iconst" | .fconst _ _ => "fconst"
  | .iadd _ _ => "iadd" | .isub _ _ => "isub" | .imul _ _ => "imul"
  | .udiv _ _ => "udiv" | .ineg _ => "ineg"
  | .ishl _ _ => "ishl" | .ushr _ _ => "ushr"
  | .band _ _ => "band" | .bandNot _ _ => "bandNot"
  | .bor _ _ => "bor" | .bxor _ _ => "bxor"
  | .ireduce32 _ => "ireduce32" | .uextend64 _ => "uextend64"
  | .sextend64 _ => "sextend64"
  | .icmp _ _ _ => "icmp" | .select _ _ _ => "select"
  | .bitselect _ _ _ => "bitselect"
  | .ctz _ => "ctz" | .popcnt _ => "popcnt"
  | .fadd _ _ => "fadd" | .fsub _ _ => "fsub" | .fmul _ _ => "fmul"
  | .fmax _ _ => "fmax" | .fmin _ _ => "fmin" | .fneg _ => "fneg"
  | .fpromote _ => "fpromote" | .fcmp _ _ _ => "fcmp"
  | .fcvtFromSint _ _ => "fcvtFromSint" | .fcvtToUint _ _ => "fcvtToUint"
  | .splat _ _ => "splat" | .extractlane _ _ => "extractlane"
  | .vhighBits _ => "vhighBits" | .bitcast _ _ => "bitcast"
  | .load _ _ => "load"
  | .ibin _ _ _ => "ibin" | .ishift _ _ _ => "ishift" | .iun _ _ => "iun"
  | .fbin _ _ _ => "fbin" | .fun1 _ _ => "fun1" | .fconv _ _ _ => "fconv"
  | .fma _ _ _ => "fma"
  | .iext _ _ _ => "iext"

/-- Statements. `op` and `call` define the next slot; the rest define nothing. -/
inductive Stmt where
  | op       (o : Op)
  /-- Stores under `notrap aligned`; `ty` documents the width the value carries. -/
  | store    (ty : ClifTy) (v a : R)
  /-- Stores under default memory flags. -/
  | storeUnaligned (v a : R)
  | istore8  (v a : R)
  /-- A call whose result binds the next slot. -/
  | call     (callee : Callee) (args : List R)
  /-- A call that answers nothing. -/
  | callVoid (callee : Callee) (args : List R)
  deriving Repr

/-- One top-tested loop.

    Entry binds `pTys.length` carried slots from `init`. Each iteration runs
    `pre` — a whole `Code`, so the test may follow an inner loop — and reads
    `flag`, the slot `pre` last bound: the exit side leaves with `exitR`, each
    becoming a fresh slot for the code after the loop, and the continue side
    binds the carries again, as the slots after `pre`'s, runs `body` and loops
    back with `cont` as the next carries. The second binding is the body block's
    parameters, so the body's carries are slots of its own.

    The test is a statement of `pre` rather than a comparison this structure
    carries, so it spends a slot like any other operation and `wf` types it by
    the rule every `icmp` gets. That is what keeps the emitter's slot and value
    counters equal through a loop, which is what lets a claim about a body
    inside one be stated in terms of its slots. -/
structure Loop where
  pTys       : List ClifTy
  init       : List R
  /-- The `i8` the exit test reads, bound by the last statement of `pre`. -/
  flag       : R
  exitOnTrue : Bool
  cont       : List R
  exitR      : List R
  exitTys    : List ClifTy
  deriving Repr

/-- A bottom-tested loop: the test is made once before entry and again at the
    end of each trip, so the body block branches to itself and there is no
    header. One fewer block on the hot path, and an empty input never enters.

    The test is read in two scopes, so it is two statements, each an ordinary
    comparison that spends a slot: the *guard*, the statement before the loop,
    comparing the initial carries; and the *back-edge test*, the body's last
    statement, comparing what the trip produced. As with `Loop` and `IteMeta`,
    the emitter reads both rather than building them, which is what keeps its
    slot and value counters equal through the loop.

    The exit values are *carry positions* rather than slots, for the same
    reason: at the guard they are the initial carries, at the back edge the
    next ones. -/
structure DLoop where
  pTys    : List ClifTy
  init    : List R
  /-- The `i8` the entry guard reads, bound by the statement before the loop, or
      `none` for a loop entered unconditionally. A loop whose trip count is known
      non-zero — zeroing a fixed-size region, say — needs no guard, and one whose
      test reads a value the body produces cannot have one: there is nothing to
      read yet. -/
  guard   : Option R
  /-- The `i8` the back-edge test reads, bound by the body's last statement. -/
  flag    : R
  contOnTrue : Bool
  cont    : List R
  /-- Which carries leave, in the order the exit block binds them. -/
  exitIdx : List Nat
  exitTys : List ClifTy
  deriving Repr

/-- A two-way branch: `flag` is read in the current block, each arm ends with
    its export list, and the join binds one fresh slot per export.

    As with `Loop`, the test is a statement of the code before the branch rather
    than a comparison carried here. -/
structure IteMeta where
  /-- The `i8` the branch reads, bound by the statement before it. -/
  flag : R
  jTys : List ClifTy
  deriving Repr

inductive Piece where
  | straight (stmts : List Stmt)
  /-- Both the condition prefix and the body are whole `Code`, so loops nest. -/
  | loop (l : Loop) (pre body : List Piece)
  | ite (m : IteMeta) (thn els : List Piece) (thnR elsR : List R)
  /-- A bottom-tested loop; `body` is the whole trip, test included. -/
  | dloop (l : DLoop) (body : List Piece)
  /-- Leave the `depth`-th enclosing loop, innermost first, carrying `args` as
      that loop's exit values.

      Labels are loop exits and nothing else: a branch is not a label, so the
      depth counts only the loops a piece sits inside. Backward branches need no
      construct — that is the loop's own back edge. -/
  | br (depth : Nat) (args : List R)
  /-- Go round the `depth`-th enclosing loop again, with `args` as the next
      carries — the back edge, taken from somewhere other than the end of the
      body. For a `dloop` the target is its body block, which is where its own
      self-branch goes. -/
  | cont (depth : Nat) (args : List R)
  deriving Repr

/-- A function body; compilation appends the final `ret`. -/
abbrev Code := List Piece

/-- Whether a piece leaves its region unconditionally, so the block it ends is
    already closed and nothing may follow it.

    A branch counts only when *both* arms leave — one arm leaving still falls
    through on the other. A loop never counts: its head always has an exit edge. -/
def termsGo : Nat → List Piece → Bool
  | 0, _ => false
  | _ + 1, [] => false
  | _ + 1, [.br _ _] | _ + 1, [.cont _ _] => true
  | fuel + 1, [.ite _ thn els _ _] => termsGo fuel thn && termsGo fuel els
  | _ + 1, [_] => false
  | fuel + 1, _ :: ps => termsGo fuel ps

/-- Fuel large enough for every body written against this library. -/
def fuel : Nat := 1000

/-- `c` leaves its region unconditionally. -/
def terminates (c : Code) : Bool := termsGo fuel c

-- `FnEnv` and `envOf` are declared in `IR` so the standard table can be built
-- before this module; both names remain reachable as `HProg.*`.
export _root_.AlgorithmLib.IR (FnEnv)

-- ---------------------------------------------------------------------------
-- Types
-- ---------------------------------------------------------------------------

/-- A map from slot numbers, holding the value for `k` at the tree position
    whose children are `2k+1` and `2k+2`.

    Slots are keyed by a number that only grows, so a sequence container is the
    obvious choice — and it is the wrong one, whichever end it makes cheap.
    Slot `0` is the base pointer and nearly every statement reads it, while the
    slots a statement's other operands name are the most recent ones. A tree
    costs `log k` either way, so no slot is a bad slot, and bodies here run to
    tens of thousands of them.

    Both the checker and the compiler key their slot maps on this, and both are
    reduced by the kernel: `wf` under the `decide` `emitChecked` runs,
    `emitStmt` under the `rfl` steps `Host.Blocks` takes. So the recursion is
    structural throughout — `get` on the tree, `set` on a fuel of `k + 1`,
    which is past what halving `k` can consume. -/
inductive Trie (α : Type) where
  | nil
  | node (v : Option α) (l r : Trie α)
  deriving Inhabited, BEq

/-- What is recorded at key `k`. -/
def Trie.get {α : Type} : Trie α → Nat → Option α
  | .nil, _ => none
  | .node v _ _, 0 => v
  | .node _ l r, k + 1 =>
      match k % 2 with
      | 0 => l.get (k / 2)
      | _ => r.get (k / 2)

/-- `set` with its measure exposed, for the lemmas to run their induction on. -/
def Trie.setGo {α : Type} : Nat → Trie α → Nat → α → Trie α
  | _, .nil, 0, x => .node (some x) .nil .nil
  | _, .node _ l r, 0, x => .node (some x) l r
  | 0, t, _, _ => t
  | d + 1, .nil, k + 1, x =>
      match k % 2 with
      | 0 => .node none (Trie.setGo d .nil (k / 2) x) .nil
      | _ => .node none .nil (Trie.setGo d .nil (k / 2) x)
  | d + 1, .node v l r, k + 1, x =>
      match k % 2 with
      | 0 => .node v (Trie.setGo d l (k / 2) x) r
      | _ => .node v l (Trie.setGo d r (k / 2) x)

/-- Record `x` at key `k`, growing the tree along the way. -/
def Trie.set {α : Type} (t : Trie α) (k : Nat) (x : α) : Trie α :=
  Trie.setGo (k + 1) t k x

theorem Trie.get_setGo_self {α : Type} : ∀ (d : Nat) (t : Trie α) (k : Nat) (x : α), k < d →
    (Trie.setGo d t k x).get k = some x
  | 0, _, _, _, h => absurd h (by omega)
  | _ + 1, .nil, 0, _, _ => rfl
  | _ + 1, .node _ _ _, 0, _, _ => rfl
  | d + 1, .nil, k + 1, x, h => by
      have hd : k / 2 < d := by omega
      by_cases hk : k % 2 = 0 <;>
        simp [Trie.setGo, Trie.get, hk, Trie.get_setGo_self d .nil (k / 2) x hd]
  | d + 1, .node _ l r, k + 1, x, h => by
      have hd : k / 2 < d := by omega
      by_cases hk : k % 2 = 0 <;>
        simp [Trie.setGo, Trie.get, hk, Trie.get_setGo_self d l (k / 2) x hd,
              Trie.get_setGo_self d r (k / 2) x hd]

theorem Trie.get_setGo_ne {α : Type} : ∀ (d : Nat) (t : Trie α) (k j : Nat) (x : α), j ≠ k →
    (Trie.setGo d t k x).get j = t.get j
  | _, .nil, 0, 0, _, h => absurd rfl h
  | _, .node _ _ _, 0, 0, _, h => absurd rfl h
  | _, .nil, 0, j + 1, _, _ => by
      by_cases hj : j % 2 = 0 <;> simp [Trie.setGo, Trie.get, hj]
  | _, .node _ _ _, 0, j + 1, _, _ => by
      by_cases hj : j % 2 = 0 <;> simp [Trie.setGo, Trie.get, hj]
  | 0, .nil, _ + 1, _, _, _ => rfl
  | 0, .node _ _ _, _ + 1, _, _, _ => rfl
  | d + 1, .nil, k + 1, 0, _, _ => by
      by_cases hk : k % 2 = 0 <;> simp [Trie.setGo, Trie.get, hk]
  | d + 1, .node _ _ _, k + 1, 0, _, _ => by
      by_cases hk : k % 2 = 0 <;> simp [Trie.setGo, Trie.get, hk]
  -- Matching parity sends both to the same child, at halves that stay apart;
  -- differing parity sends `j` to the child `set` left alone.
  | d + 1, .nil, k + 1, j + 1, x, h => by
      have hne : j ≠ k := by omega
      by_cases hk : k % 2 = 0 <;> by_cases hj : j % 2 = 0 <;>
        simp [Trie.setGo, Trie.get, hk, hj] <;>
        exact Trie.get_setGo_ne d .nil (k / 2) (j / 2) x (by omega)
  | d + 1, .node _ l r, k + 1, j + 1, x, h => by
      have hne : j ≠ k := by omega
      by_cases hk : k % 2 = 0 <;> by_cases hj : j % 2 = 0 <;>
        simp [Trie.setGo, Trie.get, hk, hj] <;>
        first
          | exact Trie.get_setGo_ne d l (k / 2) (j / 2) x (by omega)
          | exact Trie.get_setGo_ne d r (k / 2) (j / 2) x (by omega)

/-- The key just written reads back. -/
theorem Trie.get_set_self {α : Type} (t : Trie α) (k : Nat) (x : α) :
    (t.set k x).get k = some x :=
  Trie.get_setGo_self (k + 1) t k x (by omega)

/-- And every other key is untouched. -/
theorem Trie.get_set_ne {α : Type} (t : Trie α) (k j : Nat) (x : α) (h : j ≠ k) :
    (t.set k x).get j = t.get j :=
  Trie.get_setGo_ne (k + 1) t k j x h

/-- Slot types with the count carried, so the next slot's number is at hand and
    a lookup can reject an out-of-scope slot without searching for it. -/
structure TyEnv where
  /-- Slot `k`'s type is at key `k`. -/
  slots : Trie ClifTy
  /-- How many slots are bound; they are `0` to `n - 1`. -/
  n : Nat
  deriving Inhabited, BEq

/-- The type of slot `r`, or `none` when `r` is out of scope. -/
def TyEnv.get (Γ : TyEnv) (r : Nat) : Option ClifTy :=
  if r < Γ.n then Γ.slots.get r else none

/-- Bind the next slot. -/
def TyEnv.push (Γ : TyEnv) (t : ClifTy) : TyEnv := ⟨Γ.slots.set Γ.n t, Γ.n + 1⟩

/-- Bind several, in order. -/
def TyEnv.pushAll (Γ : TyEnv) (ts : List ClifTy) : TyEnv := ts.foldl TyEnv.push Γ

/-- The environment binding `ts` as slots `0..`. -/
def TyEnv.ofList (ts : List ClifTy) : TyEnv := TyEnv.pushAll ⟨.nil, 0⟩ ts

-- What the checker needs to know about a CLIF type, declared where dot
-- notation finds it.

/-- A type's name, for diagnostics; see `Op.name` for why not `repr`. -/
def _root_.AlgorithmLib.IR.ClifTy.name : ClifTy → String
  | .i8 => "i8" | .i16 => "i16" | .i32 => "i32" | .i64 => "i64"
  | .f32 => "f32" | .f64 => "f64" | .f32x4 => "f32x4" | .i8x16 => "i8x16"

def _root_.AlgorithmLib.IR.ClifTy.isInt : ClifTy → Bool
  | .i8 | .i16 | .i32 | .i64 => true
  | _ => false

def _root_.AlgorithmLib.IR.ClifTy.isFloat : ClifTy → Bool
  | .f32 | .f64 => true
  | _ => false

def _root_.AlgorithmLib.IR.ClifTy.isVec : ClifTy → Bool
  | .f32x4 | .i8x16 => true
  | _ => false

/-- A vector whose lanes are floats. The float operations apply to `f32x4` and
    not to `i8x16`, which `isVec` alone does not separate. -/
def _root_.AlgorithmLib.IR.ClifTy.isFloatVec : ClifTy → Bool
  | .f32x4 => true
  | _ => false

/-- The lane type of a vector, and how many lanes it has. -/
def _root_.AlgorithmLib.IR.ClifTy.lanes : ClifTy → Option (ClifTy × Nat)
  | .f32x4 => some (.f32, 4)
  | .i8x16 => some (.i8, 16)
  | _ => none

def _root_.AlgorithmLib.IR.ClifTy.width : ClifTy → Nat
  | .i8 => 8 | .i16 => 16 | .i32 => 32 | .i64 => 64
  | .f32 => 32 | .f64 => 64
  | .f32x4 => 128 | .i8x16 => 128

instance : Inhabited ClifTy := ⟨.i64⟩

/-- Guard reading as a predicate on the checked type. -/
def need (b : Bool) (t : ClifTy) : Option ClifTy := if b then some t else none

/-- The operand and result types a conversion of `FConv` admits.

    The integer side is `i32` or `i64`: Cranelift's verifier accepts a
    saturating conversion to `i8`, and its x64 emitter then reaches
    `unreachable!` on it, so the narrower widths would pass this check and
    abort the JIT. -/
def _root_.AlgorithmLib.IR.FConv.admits : FConv → ClifTy → ClifTy → Bool
  | .toSint, ta, ty => ta.isFloat && ty.isInt && decide (ty.width ≥ 32)
  | .fromUint, ta, ty => ta.isInt && decide (ta.width ≥ 32) && ty.isFloat
  | .demote, ta, ty => ta == .f64 && ty == .f32

/-- The operand and result types a width change admits: both integers, the
    result strictly narrower for `reduce` and strictly wider otherwise. -/
def _root_.AlgorithmLib.IR.IExt.admits : IExt → ClifTy → ClifTy → Bool
  | .reduce, ta, ty => ta.isInt && ty.isInt && decide (ty.width < ta.width)
  | _, ta, ty => ta.isInt && ty.isInt && decide (ta.width < ty.width)

/-- The integer types a one-operand integer operation admits: every one but
    `bswap`, which has no 8-bit form. -/
def _root_.AlgorithmLib.IR.IUn.admits (k : IUn) (ta : ClifTy) : Bool :=
  ta.isInt && (k != .bswap || decide (ta.width ≥ 16))

/-- The type `o` yields, or `none` when an operand is out of scope, the operand
    types disagree, or the operation does not apply to them. -/
def Op.check (Γ : TyEnv) : Op → Option ClifTy
  | .iconst ty _ => need ty.isInt ty
  | .fconst ty _ => need ty.isFloat ty
  | .iadd a b | .isub a b | .imul a b | .udiv a b => do
      let ta ← Γ.get a; let tb ← Γ.get b
      need (ta == tb && ta.isInt) ta
  | .band a b | .bandNot a b | .bor a b | .bxor a b => do
      let ta ← Γ.get a; let tb ← Γ.get b
      need (ta == tb && (ta.isInt || ta.isVec)) ta
  | .ineg a | .ctz a | .popcnt a => do
      let ta ← Γ.get a; need ta.isInt ta
  -- Cranelift lets the shift amount be any integer type.
  | .ishl a b | .ushr a b => do
      let ta ← Γ.get a; let tb ← Γ.get b
      need (ta.isInt && tb.isInt) ta
  | .ireduce32 a => do
      let ta ← Γ.get a; need (ta.isInt && ta.width > 32) .i32
  | .uextend64 a | .sextend64 a => do
      let ta ← Γ.get a; need (ta.isInt && ta.width < 64) .i64
  -- On vectors the result is a per-lane mask at the lane's own width, the same
  -- shape `fcmp` produces and the one `bitselect` and `vhighBits` consume.
  | .icmp _ a b => do
      let ta ← Γ.get a; let tb ← Γ.get b
      if ta.isVec then need (ta == tb) ta
      else need (ta == tb && ta.isInt) .i8
  | .select c a b => do
      let tc ← Γ.get c; let ta ← Γ.get a; let tb ← Γ.get b
      need (tc.isInt && ta == tb) ta
  -- The mask is the operands' own type: it comes from a comparison that was
  -- bitcast to that width, which is what makes the lane-wise select expressible.
  | .bitselect c a b => do
      let tc ← Γ.get c; let ta ← Γ.get a; let tb ← Γ.get b
      need (tc == ta && ta == tb) ta
  | .fadd a b | .fsub a b | .fmul a b | .fmax a b | .fmin a b => do
      let ta ← Γ.get a; let tb ← Γ.get b
      need (ta == tb && (ta.isFloat || ta.isFloatVec)) ta
  | .fneg a => do
      let ta ← Γ.get a; need (ta.isFloat || ta.isFloatVec) ta
  | .fpromote a => do
      let ta ← Γ.get a; need (ta == .f32) .f64
  -- On vectors the result is a per-lane mask at the lane's own width, which is
  -- the type `evalOp` produces and the width `bitselect` needs. Cranelift calls
  -- that type `i32x4`; there is no such `ClifTy`, and `Inst.fcmp` carries no
  -- result annotation for one to disagree with, so the mask keeps the operand
  -- type and `bitcast` — which compares widths — accepts it either way.
  | .fcmp _ a b => do
      let ta ← Γ.get a; let tb ← Γ.get b
      if ta.isFloatVec then need (ta == tb) ta
      else need (ta == tb && ta.isFloat) .i8
  | .fcvtFromSint ty a => do
      let ta ← Γ.get a; need (ta.isInt && ty.isFloat) ty
  -- `i32` or `i64` only, for the reason `FConv.admits` gives.
  | .fcvtToUint ty a => do
      let ta ← Γ.get a; need (ta.isFloat && ty.isInt && decide (ty.width ≥ 32)) ty
  | .splat ty a => do
      let ta ← Γ.get a; let (lane, _) ← ty.lanes
      need (ta == lane) ty
  | .extractlane a lane => do
      let ta ← Γ.get a; let (lt, n) ← ta.lanes
      need (lane < n) lt
  | .vhighBits a => do
      let ta ← Γ.get a; need ta.isVec .i32
  | .bitcast ty a => do
      let ta ← Γ.get a; need (ta.width == ty.width) ty
  | .load op a => do
      let ta ← Γ.get a; need (ta == .i64) op.ty
  | .ibin _ a b => do
      let ta ← Γ.get a; let tb ← Γ.get b
      need (ta == tb && ta.isInt) ta
  | .ishift _ a b => do
      let ta ← Γ.get a; let tb ← Γ.get b
      need (ta.isInt && tb.isInt) ta
  | .iun k a => do
      let ta ← Γ.get a; need (k.admits ta) ta
  | .fbin _ a b => do
      let ta ← Γ.get a; let tb ← Γ.get b
      need (ta == tb && (ta.isFloat || ta.isFloatVec)) ta
  | .fun1 _ a => do
      let ta ← Γ.get a; need (ta.isFloat || ta.isFloatVec) ta
  | .fma a b c => do
      let ta ← Γ.get a; let tb ← Γ.get b; let tc ← Γ.get c
      need (ta == tb && ta == tc && (ta.isFloat || ta.isFloatVec)) ta
  | .fconv k ty a => do
      let ta ← Γ.get a; need (k.admits ta ty) ty
  | .iext k ty a => do
      let ta ← Γ.get a; need (k.admits ta ty) ty

/-- Every argument in scope and typed as the signature declares. -/
def argsOk (Γ : TyEnv) (d : CalleeSig) (args : List R) : Bool :=
  args.length == d.params.length &&
    (List.zip args d.params).all fun (r, t) => Γ.get r == some t

/-- The type a statement appends to the environment, or `none` for statements
    that bind nothing. `ok` is the check itself. -/
def Stmt.check (env : FnEnv) (Γ : TyEnv) : Stmt → Bool × Option ClifTy
  | .op o => match o.check Γ with
      | some t => (true, some t)
      | none => (false, none)
  | .store ty v a =>
      (Γ.get v == some ty && Γ.get a == some .i64, none)
  | .storeUnaligned v a =>
      ((Γ.get v).isSome && Γ.get a == some .i64, none)
  -- Any integer: `istore8` narrows to the low byte whatever width it is given,
  -- which is what Cranelift does and what `runStmt` already accepted.
  | .istore8 v a =>
      (((Γ.get v).map (·.isInt)).getD false && Γ.get a == some .i64, none)
  | .call c args => match env.sigOf c with
      | some d => match d.result with
          | some t => (argsOk Γ d args, some t)
          | none => (false, none)
      | none => (false, none)
  | .callVoid c args => match env.sigOf c with
      | some d => (argsOk Γ d args && d.result.isNone, none)
      | none => (false, none)

/-- How many slots a statement defines. -/
def Stmt.binds : Stmt → Nat
  | .op _ | .call _ _ => 1
  | _ => 0

/-- The slots a statement reads, the way `Op.regs` gives an operation's.

    `emitStmt` resolves every one of them through `CS.get`, so "each is below
    the slot count" is exactly the condition under which that resolution is the
    identity — which is what the straight-line simulation runs on. Weaker than
    `Stmt.check`, which also types them. -/
def Stmt.regs : Stmt → List R
  | .op o => o.regs
  | .store _ v a | .storeUnaligned v a | .istore8 v a => [v, a]
  | .call _ args | .callVoid _ args => args

/-- Check a statement list, extending the environment as it goes.

    The verdict is carried down rather than combined on the way back up, so the
    recursion is a tail call and a straight-line body costs one stack frame
    instead of one per statement. Bodies here reach ten thousand statements,
    which is past what the runtime stack holds. -/
def wfStmts (env : FnEnv) (Γ : TyEnv) (ss : List Stmt) : Bool × TyEnv :=
  go true Γ ss
where
  go (acc : Bool) : TyEnv → List Stmt → Bool × TyEnv
    | Γ, [] => (acc, Γ)
    | Γ, s :: ss =>
        let (ok, t) := s.check env Γ
        let Γ' := match t with | some t => Γ.push t | none => Γ
        go (acc && ok) Γ' ss

/-- The slots `rs` name, in order, if all are in scope. -/
def tysOf (Γ : TyEnv) : List R → Option (List ClifTy)
  | [] => some []
  | r :: rs => do let t ← Γ.get r; let ts ← tysOf Γ rs; pure (t :: ts)

def tysAre (Γ : TyEnv) (rs : List R) (ts : List ClifTy) : Bool :=
  tysOf Γ rs == some ts

/-- The carry types of the `depth`-th enclosing loop, when it has a header to go
    back to. `none` marks a `dloop`, which does not. -/
def carryOf (lbl : List (List ClifTy × Option (List ClifTy))) (depth : Nat) :
    Option (List ClifTy) :=
  (lbl[depth]?).bind (·.2)

/-- Fuel-driven so the kernel reduces it (`decide`); nested-inductive mutual
    recursion compiles to well-founded fix, which does not. Fuel bounds the
    piece count along one chain, not the program size. -/
def wfGo (env : FnEnv) : Nat → List (List ClifTy × Option (List ClifTy)) → TyEnv →
    List Piece → Bool × TyEnv
  | 0, _, Γ, _ => (false, Γ)
  | _ + 1, _, Γ, [] => (true, Γ)
  | fuel + 1, lbl, Γ, .straight ss :: ps =>
      let r := wfStmts env Γ ss
      let r' := wfGo env fuel lbl r.2 ps
      (r.1 && r'.1, r'.2)
  | fuel + 1, lbl, Γ, .loop l pre body :: ps =>
      let lbl' := (l.exitTys, some l.pTys) :: lbl
      let Γcarry := Γ.pushAll l.pTys
      let rPre := wfGo env fuel lbl' Γcarry pre
      let Γhead := rPre.2
      let rBody := wfGo env fuel lbl' (Γhead.pushAll l.pTys) body
      let ok :=
        tysAre Γ l.init l.pTys &&
        rPre.1 && !termsGo fuel pre &&
        (Γhead.get l.flag == some .i8) &&
        tysAre Γhead l.exitR l.exitTys &&
        rBody.1 &&
        (termsGo fuel body || tysAre rBody.2 l.cont l.pTys)
      -- The exit block binds its parameters after everything the head and body
      -- defined, which is the numbering `emitLoop` uses.
      let r := wfGo env fuel lbl (rBody.2.pushAll l.exitTys) ps
      (ok && r.1, r.2)
  | fuel + 1, lbl, Γ, .ite m thn els thnR elsR :: ps =>
      let rT := wfGo env fuel lbl Γ thn
      let rE := wfGo env fuel lbl rT.2 els
      let tT := termsGo fuel thn
      let tE := termsGo fuel els
      -- An arm that leaves never reaches the join, so its exports are not a
      -- constraint; when neither arm reaches it there is no join block at all.
      let jTys := if tT && tE then [] else m.jTys
      let ok :=
        (Γ.get m.flag == some .i8) &&
        rT.1 && rE.1 &&
        (tT || tysAre rT.2 thnR jTys) &&
        (tE || tysAre rE.2 elsR jTys)
      let r := wfGo env fuel lbl (rE.2.pushAll jTys) ps
      (ok && r.1, r.2)
  | fuel + 1, lbl, Γ, .dloop l body :: ps =>
      let Γcarry := Γ.pushAll l.pTys
      let lbl' := (l.exitTys, some l.pTys) :: lbl
      let rBody := wfGo env fuel lbl' Γcarry body
      -- `exitIdx` names carries, so it is checked against the carry types once
      -- and holds at both the guard and the back edge.
      let carryTy (i : Nat) : Option ClifTy := l.pTys[i]?
      let guardOk := match l.guard with
        | none => true
        | some g => Γ.get g == some .i8
      let ok :=
        tysAre Γ l.init l.pTys && guardOk &&
        (termsGo fuel body || rBody.2.get l.flag == some .i8) &&
        l.exitIdx.length == l.exitTys.length &&
        (List.zip l.exitIdx l.exitTys).all (fun (i, t) => carryTy i == some t) &&
        rBody.1 &&
        -- A body where every path leaves or takes the edge itself never
        -- reaches the back-edge test, so it need not supply carries for one.
        (termsGo fuel body || tysAre rBody.2 l.cont l.pTys)
      let r := wfGo env fuel lbl (rBody.2.pushAll l.exitTys) ps
      (ok && r.1, r.2)
  | _ + 1, lbl, Γ, .br depth args :: ps =>
      -- Nothing may follow: the block is closed and the slots stop here.
      (ps.isEmpty && (lbl[depth]?).isSome && tysAre Γ args (((lbl[depth]?).map (·.1)).getD []),
       Γ)
  | _ + 1, lbl, Γ, .cont depth args :: ps =>
      (ps.isEmpty && (lbl[depth]?).isSome &&
        (carryOf lbl depth).isSome && tysAre Γ args ((carryOf lbl depth).getD []),
       Γ)

def callsIn (ss : List Stmt) : List Callee :=
  ss.filterMap fun
    | .call c _ => some c
    | .callVoid c _ => some c
    | _ => none

def callsGo : Nat → List Piece → List Callee
  | 0, _ => []
  | _ + 1, [] => []
  | fuel + 1, .straight ss :: ps => callsIn ss ++ callsGo fuel ps
  | fuel + 1, .loop _ pre body :: ps =>
      callsGo fuel pre ++ callsGo fuel body ++ callsGo fuel ps
  | fuel + 1, .ite _ thn els _ _ :: ps =>
      callsGo fuel thn ++ callsGo fuel els ++ callsGo fuel ps
  | fuel + 1, .dloop _ body :: ps => callsGo fuel body ++ callsGo fuel ps
  | fuel + 1, .br _ _ :: ps | fuel + 1, .cont _ _ :: ps => callsGo fuel ps

/-- The FFI calls a body performs, in program order — one iteration of each
    loop, both arms of each branch. -/
def callsOf (c : Code) : List Callee := callsGo fuel c

/-- Every reference names a slot that exists, with the type the use demands,
    and every annotation matches what its operands compute.

    Deliberately weaker than dominance: a reference from inside a loop to a slot
    the loop defined, used after the loop, satisfies `wf`. `scopeOk` is the
    check that refuses it. -/
def wf (env : FnEnv) (params : List ClifTy) (c : Code) : Bool :=
  (wfGo env fuel [] (TyEnv.ofList params) c).1

-- ---------------------------------------------------------------------------
-- Scope
-- ---------------------------------------------------------------------------

/-- Slots in scope, as half-open ranges `[a, b)`.

    A slot is in scope where every path that reaches the point has bound it,
    and bound it to the same value in the term and in the compiled blocks. That
    is less than "below the slot count": the else arm is numbered past the then
    arm's slots, which it never ran, and the code after a loop past the body's,
    which the last trip may not have reached. Those slots exist in both
    interpreters' stores and hold different things there. -/
abbrev Scope := List (Nat × Nat)

def Scope.mem (S : Scope) (i : Nat) : Bool := S.any fun (a, b) => a ≤ i && i < b

/-- Add `[a, b)`, merging it into the most recent range when they touch, which
    is what a straight-line run does one slot at a time. -/
def Scope.add (S : Scope) (a b : Nat) : Scope :=
  match S with
  | (c, d) :: rest => if c ≤ a && d == a && a ≤ b then (c, b) :: rest else (a, b) :: S
  | [] => [(a, b)]

/-- Every slot of `S` is in `T`. -/
def Scope.sub (S T : Scope) : Bool :=
  S.all fun (a, b) => (List.range' a (b - a)).all T.mem

/-- What a loop's `br` and `cont` are checked against: how many values its exit
    block and its back-edge target take, and what has to be in scope where the
    loop is left, which is what stays in scope after it. -/
structure SLbl where
  exitN  : Nat
  carryN : Nat
  need   : Scope

/-- `r` is a slot bound so far, and in scope. -/
def inS (S : Scope) (n : Nat) (r : R) : Bool := r < n && S.mem r

def allIn (S : Scope) (n : Nat) (rs : List R) : Bool := rs.all (inS S n)

/-- A straight-line run: each statement reads slots in scope and puts the ones
    it binds in scope. -/
def scStmts : Scope → Nat → List Stmt → Option (Scope × Nat)
  | S, n, [] => some (S, n)
  | S, n, st :: ss =>
      if allIn S n st.regs then scStmts (S.add n (n + st.binds)) (n + st.binds) ss else none

/-- The scope check, over the same fuel `emitCode` spends, so the two agree
    on every `termsGo` they ask. Answers the scope and the slot count after `c`.

    Where control joins, the scope is what every incoming path has: after a
    branch, what was in scope before it and the join's parameters; after a loop,
    what was in scope before it, its carries --- the head's parameters, which
    every exit edge leaves from under --- and its exit values; after a
    bottom-tested loop, what was in scope before it and its exit values. A `br`
    has to have the loop's in scope; so does the edge that leaves from the head
    or the back edge.

    Nothing may follow a piece that leaves on every path: its block is closed,
    and whatever came next would be emitted into a block that does not exist. -/
def scGo : Nat → List SLbl → Scope → Nat → List Piece → Option (Scope × Nat)
  | 0, _, _, _, _ => none
  | _ + 1, _, S, n, [] => some (S, n)
  | f + 1, lb, S, n, .straight ss :: ps => do
      let (S1, n1) ← scStmts S n ss
      scGo f lb S1 n1 ps
  | f + 1, lb, S, n, .loop l pre body :: ps => do
      let len := l.pTys.length
      let need := S.add n (n + len)
      let lb' : List SLbl := ⟨l.exitTys.length, len, need⟩ :: lb
      let (Sp, np) ← scGo f lb' need (n + len) pre
      let (Sb, nb) ← scGo f lb' (Sp.add np (np + len)) (np + len) body
      if allIn S n l.init && l.init.length == len && !termsGo f pre &&
          inS Sp np l.flag && allIn Sp np l.exitR &&
          l.exitR.length == l.exitTys.length && need.sub Sp &&
          (termsGo f body || (allIn Sb nb l.cont && l.cont.length == len))
      then scGo f lb (need.add nb (nb + l.exitTys.length)) (nb + l.exitTys.length) ps
      else none
  | f + 1, lb, S, n, .ite m thn els thnR elsR :: ps => do
      let (St, nt) ← scGo f lb S n thn
      let (Se, ne) ← scGo f lb S nt els
      let tT := termsGo f thn
      let tE := termsGo f els
      if inS S n m.flag &&
          (tT || (allIn St nt thnR && thnR.length == m.jTys.length && S.sub St)) &&
          (tE || (allIn Se ne elsR && elsR.length == m.jTys.length && S.sub Se))
      then
        if tT && tE then (if ps.isEmpty then some (S, ne) else none)
        else scGo f lb (S.add ne (ne + m.jTys.length)) (ne + m.jTys.length) ps
      else none
  | f + 1, lb, S, n, .dloop l body :: ps => do
      let len := l.pTys.length
      let lb' : List SLbl := ⟨l.exitTys.length, len, S⟩ :: lb
      let (Sb, nb) ← scGo f lb' (S.add n (n + len)) (n + len) body
      let guardOk := match l.guard with
        | none => true
        | some g => inS S n g
      if allIn S n l.init && l.init.length == len && guardOk &&
          l.exitIdx.length == l.exitTys.length && l.exitIdx.all (· < len) &&
          (termsGo f body ||
            (inS Sb nb l.flag && allIn Sb nb l.cont && l.cont.length == len && S.sub Sb))
      then scGo f lb (S.add nb (nb + l.exitTys.length)) (nb + l.exitTys.length) ps
      else none
  | _ + 1, lb, S, n, .br d args :: ps =>
      match lb[d]? with
      | some L =>
          if ps.isEmpty && allIn S n args && args.length == L.exitN && L.need.sub S
          then some (S, n) else none
      | none => none
  | _ + 1, lb, S, n, .cont d args :: ps =>
      match lb[d]? with
      | some L =>
          if ps.isEmpty && allIn S n args && args.length == L.carryN then some (S, n)
          else none
      | none => none

/-- **Every slot a body reads is bound, to the same value, on every path to the
    read.** What `wf` deliberately leaves to Cranelift's verifier, stated so
    that `compile_sound` can assume it: in scope is where the term's slot and
    the compiled block's value are the same thing. -/
def scopeOk (params : List ClifTy) (c : Code) : Bool :=
  (scGo fuel [] [(0, params.length)] params.length c).isSome

/-- `scopeOk`, and the slot a body answers with is in scope where it ends. -/
def retOk (params : List ClifTy) (c : Code) (status : Option R) : Bool :=
  match scGo fuel [] [(0, params.length)] params.length c with
  | none => false
  | some (S', n') => status.all fun r => S'.mem r && decide (r < n')

theorem retOk_none (params : List ClifTy) (c : Code) : retOk params c none = scopeOk params c := by
  unfold retOk scopeOk; split <;> simp_all

-- ---------------------------------------------------------------------------
-- Observations
-- ---------------------------------------------------------------------------


-- ---------------------------------------------------------------------------
-- Compilation
-- ---------------------------------------------------------------------------

/-- Blocks are numbered as reserved, values as created, and `env` maps slots to
    the `Val` carrying them in the region being emitted. A slot written twice
    keeps the later value, which is how a loop body's parameters shadow the
    head's. -/
structure CS where
  nextVal : Nat
  nextBlk : Nat
  slots   : Nat
  env     : Trie Val
  curRef  : Nat
  curPars : List (Val × ClifTy)
  cur     : List Inst              -- reversed
  done    : List BlockData
  /-- Per enclosing loop, innermost first: the block it leaves to, and the block
      its back edge returns to when it has one. `br` resolves against the first,
      `cont` against the second. -/
  labels  : List (Nat × Option Nat) := []

def CS.fresh (s : CS) : Val × CS :=
  (⟨s.nextVal⟩, { s with nextVal := s.nextVal + 1 })

/-- An unresolvable slot yields a value nothing defines: visible in a dump and
    rejected by Cranelift's verifier. `wf` is what rules it out. -/
def CS.get (s : CS) (r : R) : Val :=
  match s.env.get r with
  | some v => v
  | none => ⟨1000000 + r⟩

def CS.close (s : CS) (term : Inst) : CS :=
  { s with
    done := s.done ++ [{ ref := ⟨s.curRef⟩, params := s.curPars,
                         insts := (term :: s.cur).reverse }]
    cur := [], curPars := [] }

/-- Open block `ref` with one fresh parameter per entry of `tys`, bound to
    slots `firstSlot..`. -/
def CS.open'.go (firstSlot : Nat) : CS → Nat → List ClifTy → CS
  | st, _, [] => st
  | st, i, t :: ts =>
      let (v, st') := st.fresh
      go firstSlot
        { st' with curPars := st'.curPars ++ [(v, t)],
                   env := st'.env.set (firstSlot + i) v } (i + 1) ts

/-- Structural rather than a `for` loop, so the numbering it establishes can be
    reasoned about: `open'` is where a block's parameters become slots, and the
    carry and join correspondence is a statement about exactly this function. -/
def CS.open' (s : CS) (ref : Nat) (tys : List ClifTy) (firstSlot : Nat) : CS :=
  CS.open'.go firstSlot { s with curRef := ref, curPars := [] } 0 tys

def emitStmt (s : CS) : Stmt → CS
  | .op o =>
      let (v, s) := s.fresh
      let inst : Inst :=
        match o with
        | .iconst ty k  => .iconst v ty k
        | .iadd a b     => .iadd v (s.get a) (s.get b)
        | .isub a b     => .isub v (s.get a) (s.get b)
        | .imul a b     => .imul v (s.get a) (s.get b)
        | .udiv a b     => .udiv v (s.get a) (s.get b)
        | .ineg a       => .ineg v (s.get a)
        | .ishl a b     => .ishl v (s.get a) (s.get b)
        | .ushr a b     => .ushr v (s.get a) (s.get b)
        | .band a b     => .band v (s.get a) (s.get b)
        | .bandNot a b  => .bandNot v (s.get a) (s.get b)
        | .bor a b      => .bor v (s.get a) (s.get b)
        | .bxor a b     => .bxor v (s.get a) (s.get b)
        | .ireduce32 a  => .ireduce32 v (s.get a)
        | .uextend64 a  => .uextend64 v (s.get a)
        | .sextend64 a  => .sextend64 v (s.get a)
        | .icmp c a b   => .icmp v c (s.get a) (s.get b)
        | .select c a b => .select v (s.get c) (s.get a) (s.get b)
        | .bitselect c a b => .bitselect v (s.get c) (s.get a) (s.get b)
        | .ctz a        => .ctz v (s.get a)
        | .popcnt a     => .popcnt v (s.get a)
        | .fconst ty b  => .fconst v ty b
        | .fadd a b     => .fadd v (s.get a) (s.get b)
        | .fsub a b     => .fsub v (s.get a) (s.get b)
        | .fmul a b     => .fmul v (s.get a) (s.get b)
        | .fmax a b     => .fmax v (s.get a) (s.get b)
        | .fmin a b     => .fmin v (s.get a) (s.get b)
        | .fneg a       => .fneg v (s.get a)
        | .fpromote a   => .fpromote v (s.get a)
        | .fcmp c a b   => .fcmp v c (s.get a) (s.get b)
        | .ibin k a b   => .ibin v k (s.get a) (s.get b)
        | .ishift k a b => .ishift v k (s.get a) (s.get b)
        | .iun k a      => .iun v k (s.get a)
        | .fbin k a b   => .fbin v k (s.get a) (s.get b)
        | .fun1 k a     => .fun1 v k (s.get a)
        | .fconv k t a  => .fconv v k t (s.get a)
        | .fma a b c    => .fma v (s.get a) (s.get b) (s.get c)
        | .iext k t a   => .iext v k t (s.get a)
        | .fcvtFromSint ty a => .fcvtFromSint v ty (s.get a)
        | .fcvtToUint ty a   => .fcvtToUint v ty (s.get a)
        | .splat ty a   => .splat v ty (s.get a)
        | .extractlane a l => .extractlane v (s.get a) l
        | .vhighBits a  => .vhighBits v (s.get a)
        | .bitcast ty a => .bitcast v ty (s.get a)
        | .load op a    => .load v op (s.get a)
      { s with cur := inst :: s.cur, env := s.env.set s.slots v, slots := s.slots + 1 }
  | .store ty v a => { s with cur := .storeTyped ty (s.get v) (s.get a) :: s.cur }
  | .storeUnaligned v a => { s with cur := .store (s.get v) (s.get a) :: s.cur }
  | .istore8 v a => { s with cur := .istore8 (s.get v) (s.get a) :: s.cur }
  | .call c args =>
      let (v, s) := s.fresh
      { s with cur := .call (some v) c (args.map s.get) :: s.cur,
               env := s.env.set s.slots v, slots := s.slots + 1 }
  | .callVoid c args =>
      { s with cur := .call none c (args.map s.get) :: s.cur }

def emitStmts (s : CS) (ss : List Stmt) : CS := ss.foldl emitStmt s

mutual
def emitPiece (fuel : Nat) (s : CS) : Piece → CS
  | .straight ss => emitStmts s ss
  | .ite m thn els thnR elsR => emitIte fuel s m thn els thnR elsR
  | .loop l pre body => emitLoop fuel s l pre body
  | .dloop l body => emitDLoop fuel s l body
  -- An out-of-range depth leaves a jump to a block nothing defines: visible in
  -- a dump and rejected by Cranelift's verifier. `wf` is what rules it out.
  | .br depth args =>
      s.close (.jump ⟨((s.labels[depth]?).map (·.1)).getD 1000000⟩ (args.map s.get))
  | .cont depth args =>
      s.close (.jump ⟨((s.labels[depth]?).bind (·.2)).getD 1000000⟩ (args.map s.get))

def emitLoop (fuel : Nat) (s : CS) (l : Loop) (pre body : List Piece) : CS :=
  -- Reserve ids so every branch can name its target before it exists.
  let headId := s.nextBlk
  let bodyId := headId + 1
  let exitId := headId + 2
  let outerLabels := s.labels
  let s := { s with nextBlk := headId + 3 }
  let firstCarry := s.slots
  -- the current block ends by entering the loop
  let s := s.close (.jump ⟨headId⟩ (l.init.map s.get))
  -- head: carries as parameters, then the condition prefix, then the test
  let s := { s.open' headId l.pTys firstCarry with
             slots := firstCarry + l.pTys.length,
             labels := (exitId, some headId) :: outerLabels }
  let s := emitCode fuel s pre
  -- The test is the last statement `pre` emitted, so the flag is a slot to read
  -- rather than a value to allocate here: nothing advances the value counter
  -- past the slot counter, and every slot in the head is still `Val` of itself.
  let sHead := s
  let flag := sHead.get l.flag
  let carryVals := (List.range l.pTys.length).map (fun i => sHead.get (firstCarry + i))
  let exitArgs := l.exitR.map sHead.get
  let te :=
    if l.exitOnTrue then (exitId, exitArgs, bodyId, carryVals)
    else (bodyId, carryVals, exitId, exitArgs)
  let s := sHead.close (.brif flag ⟨te.1⟩ te.2.1 ⟨te.2.2.1⟩ te.2.2.2)
  -- body: the carries again, as its own parameters, bound to the slots after the
  -- prefix's --- which is where the term numbers the body's carries, so slot `i`
  -- stays `Val i`. The back edge carries only `cont`.
  let bodyFirst := s.slots
  let s := { s.open' bodyId l.pTys bodyFirst with slots := bodyFirst + l.pTys.length }
  let s := emitCode fuel s body
  -- A body that leaves on every path has already closed its block; the back
  -- edge would be a second terminator for a block that no longer exists.
  let s := if termsGo fuel body then s
           else s.close (.jump ⟨headId⟩ (l.cont.map s.get))
  -- exit: the code after the loop starts here, with the exit values as params
  let exitFirst := s.slots
  let s := s.open' exitId l.exitTys exitFirst
  { s with slots := exitFirst + l.exitTys.length, labels := outerLabels }

def emitDLoop (fuel : Nat) (s : CS) (l : DLoop) (body : List Piece) : CS :=
  let outerLabels := s.labels
  let bodyId := s.nextBlk
  let exitId := bodyId + 1
  let s := { s with nextBlk := bodyId + 2 }
  -- The guard: the statement before the loop compared the initial carries, and
  -- the block already open branches on it.
  let initVals := l.init.map s.get
  let s :=
    match l.guard with
    | some g =>
      let exit0 := l.exitIdx.map (fun i => (initVals[i]?).getD (⟨1000000⟩ : Val))
      let te0 :=
        if l.contOnTrue then (bodyId, initVals, exitId, exit0)
        else (exitId, exit0, bodyId, initVals)
      s.close (.brif (s.get g) ⟨te0.1⟩ te0.2.1 ⟨te0.2.2.1⟩ te0.2.2.2)
    | none => s.close (.jump ⟨bodyId⟩ initVals)
  -- The body: the carries as its parameters, the trip, then the same test.
  let firstCarry := s.slots
  let s := { s.open' bodyId l.pTys firstCarry with
             slots := firstCarry + l.pTys.length,
             labels := (exitId, some bodyId) :: outerLabels }
  let s := emitCode fuel s body
  -- A body where every path leaves or takes the edge itself has already closed
  -- its block; the back-edge test would be a second terminator for it.
  let s :=
    if termsGo fuel body then s
    else
      let contVals := l.cont.map s.get
      let exitN := l.exitIdx.map (fun i => (contVals[i]?).getD ⟨1000000⟩)
      let teN :=
        if l.contOnTrue then (bodyId, contVals, exitId, exitN)
        else (exitId, exitN, bodyId, contVals)
      s.close (.brif (s.get l.flag) ⟨teN.1⟩ teN.2.1 ⟨teN.2.2.1⟩ teN.2.2.2)
  let exitFirst := s.slots
  let s := s.open' exitId l.exitTys exitFirst
  { s with slots := exitFirst + l.exitTys.length, labels := outerLabels }

def emitIte (fuel : Nat) (s : CS) (m : IteMeta) (thn els : List Piece)
    (thnR elsR : List R) : CS :=
  let thnId := s.nextBlk
  let elsId := thnId + 1
  let joinId := thnId + 2
  let s := { s with nextBlk := thnId + 3 }
  -- The test is the statement before the branch, so the flag is read here.
  let flag := s.get m.flag
  let s := s.close (.brif flag ⟨thnId⟩ [] ⟨elsId⟩ [])
  let tT := termsGo fuel thn
  let tE := termsGo fuel els
  let s := s.open' thnId [] s.slots
  let s := emitCode fuel s thn
  let s := if tT then s else s.close (.jump ⟨joinId⟩ (thnR.map s.get))
  let s := s.open' elsId [] s.slots
  let s := emitCode fuel s els
  let s := if tE then s else s.close (.jump ⟨joinId⟩ (elsR.map s.get))
  -- Neither arm falls through: there is no join, and the branch is itself a
  -- terminator, so `joinId` stays reserved and unused.
  if tT && tE then s
  else
    let joinFirst := s.slots
    let s := s.open' joinId m.jTys joinFirst
    { s with slots := joinFirst + m.jTys.length }

def emitCode : Nat → CS → List Piece → CS
  | 0, s, _ => s
  | _ + 1, s, [] => s
  | fuel + 1, s, p :: ps => emitCode fuel (emitPiece fuel s p) ps
end

/-- What an entry point's entry block takes: the memory base pointer, then the
caller's input buffer and its length, then the caller's output buffer and its
length. The runtime passes the caller's buffers as these arguments, so there is
no place in memory the runtime and a program have to agree on. -/
def ptrParams : List ClifTy := [.i64, .i64, .i64, .i64, .i64]

/-- Compile a body, well-formed or not.

    This is what the semantics is stated against: `CompileSound` relates the two
    runs of an *arbitrary* body, so the compiler it names cannot demand one that
    passes `wf`. Anything headed for an artifact goes through
    `Prog.compileProg`, which runs `wf` first and refuses a body that fails.

    `params` types the entry block, whose parameters are slots `0..`; an entry
    point takes `ptrParams`, a thread worker its one spawn argument.

    `status` is the slot the function answers with, when it answers: its
    presence is what gives the compiled function an `i64` return, because the
    runtime reads a signature off the body. -/
def compileBody (idx : Nat) (c : Code) (env : FnEnv)
    (params : List ClifTy := ptrParams) (status : Option R := none) :
    FuncData :=
  Id.run do
    let s0 : CS := { nextVal := 0, nextBlk := 1, slots := 0, env := .nil,
                     curRef := 0, curPars := [], cur := [], done := [] }
    let s := { s0.open' 0 params 0 with slots := params.length }
    let s := emitCode fuel s c
    let s := s.close (.ret (status.map s.get))
    -- The table is what the body called, in the order it first called each, so
    -- there is nothing to filter and nothing to renumber: what ships is the
    -- table the body was checked against, with the signatures dropped.
    return {
      index := idx
      blocks := s.done.mergeSort (fun a b => a.ref.id ≤ b.ref.id)
    }

end AlgorithmLib.HProg
