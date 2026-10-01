module
public import AlgorithmLib.Host.Term
meta import AlgorithmLib.Host.Term
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Prog` — a function body as a typed term with binders

`HProg.Code` is the first-order body an artifact ships: slots are numbers, and
whether a number names a slot of the right type is a question `wf` answers by
running a checker. This module is the same body written with *binders* and
*types*, so that both questions are the elaborator's.

A value is a `V ty`, where `V` is a parameter the user never instantiates: the
only way to obtain one is to have bound it, and its index is its CLIF type. A
loop label is an `L exitTys carryTys`, obtained the same way, so a `brk` names
a loop that encloses it and carries what that loop exits with. `Prog` is a free
monad over these, which is what makes `do`-notation the surface: `bind` is the
continuation the constructors already hold.

`emit` instantiates `V` at the slot number and `L` at the nesting level and
folds the term into a `Code`. It is a total function, and it is the only way a
body reaches an artifact, so there is one form of a body to prove about and it
is the one that ships. Nothing is spliced and nothing is reified: a body is an
ordinary Lean term, `countEach needles VECTORS` is an ordinary application, and a
generator that computes its own shape is an ordinary program.

## What is checked, and where

* **Operand types, slot scope, vector lane counts, store widths** --- by the
  elaborator, at the line that writes them. An ill-typed body is not a `Prog`.
* **Loop and branch shape** --- by the types of `loop`, `ite` and the labels.
  A `brk` carrying the wrong list is a type error.
* **Whatever the generator itself computes** --- indices, divisibility,
  alignment --- by whatever the author put in their own types.
* **The residue** --- that a body assembled from all of that satisfies `wf` ---
  by `emitChecked`, when the generator runs. `emit_wf` is the theorem that
  would retire it; until it lands, the check is a gate rather than an
  obligation, and no generator carries one.

## `for` is not a loop

`for x in xs do ...` inside a `Prog` is Lean's `ForIn`: it runs while the
*generator* runs and unrolls into straight-line statements. `forLoop` is the
loop the artifact performs. This is the same distinction `List.forM` and
`forLoop` already drew at the old surface, and it is the only place where
reading a body as ordinary Lean gives the wrong answer.
-/

namespace AlgorithmLib.Prog

open AlgorithmLib.IR
open AlgorithmLib.HProg (Op Stmt Loop DLoop IteMeta Piece Code R fuel terminates
  termsGo callsOf wf scopeOk retOk ptrParams)

-- ---------------------------------------------------------------------------
-- Value lists
-- ---------------------------------------------------------------------------

/-- A list of values, typed by the list of their types. Loop carries, loop
    exits and branch joins are all of this shape, and it is what puts the
    annotation the first-order term carries beyond the author's reach. -/
inductive Vals (V : ClifTy → Type) : List ClifTy → Type where
  | nil : Vals V []
  | cons {ty tys} (v : V ty) (vs : Vals V tys) : Vals V (ty :: tys)

infixr:67 " ::ᵥ " => Vals.cons

/-- `%[a, b, c]`, the way a list of values is written. -/
syntax "%[" term,* "]" : term
macro_rules
  | `(%[]) => `(Vals.nil)
  | `(%[$x:term]) => `(Vals.cons $x Vals.nil)
  | `(%[$x:term, $xs:term,*]) => `(Vals.cons $x %[$xs,*])

namespace Vals

def head {V ty tys} : Vals V (ty :: tys) → V ty
  | .cons v _ => v

def tail {V ty tys} : Vals V (ty :: tys) → Vals V tys
  | .cons _ vs => vs

/-- The second, third and fourth, for the loops that carry that many. -/
def snd {V a b tys} (vs : Vals V (a :: b :: tys)) : V b := vs.tail.head
def thd {V a b c tys} (vs : Vals V (a :: b :: c :: tys)) : V c := vs.tail.tail.head
def fth {V a b c d tys} (vs : Vals V (a :: b :: c :: d :: tys)) : V d :=
  vs.tail.tail.tail.head
def fif {V a b c d e tys} (vs : Vals V (a :: b :: c :: d :: e :: tys)) : V e :=
  vs.tail.tail.tail.tail.head

/-- The `i`th value, typed by the `i`th entry of the index. The bound is an
    auto-param, so a position past the end is an error where it is written. -/
def get {V} : {tys : List ClifTy} → Vals V tys → (i : Nat) →
    (h : i < tys.length := by decide) → V tys[i]
  | _ :: _, .cons v _, 0, _ => v
  | _ :: _, .cons _ vs, i + 1, h => get vs i (by simpa using h)

/-- Everything past the first `k`. -/
def drop {V} : (k : Nat) → {tys : List ClifTy} → Vals V tys → Vals V (tys.drop k)
  | 0, _, vs => vs
  | _ + 1, [], .nil => .nil
  | k + 1, _ :: _, .cons _ vs => drop k vs

def append {V} : {as bs : List ClifTy} → Vals V as → Vals V bs → Vals V (as ++ bs)
  | [], _, .nil, ys => ys
  | _ :: _, _, .cons x xs, ys => .cons x (append xs ys)

-- A loop whose *width* the generator computed carries a run of one type. The
-- four functions below are that case: `List.replicate n ty` is the index, and
-- `n + 1` unfolds to `ty :: List.replicate n ty` definitionally, so `head` and
-- `tail` apply without a cast.

/-- `n` values of one type, from a function of the position. -/
def ofFn {V ty} : {n : Nat} → (Fin n → V ty) → Vals V (List.replicate n ty)
  | 0, _ => .nil
  | _ + 1, f => .cons (f 0) (ofFn (fun i => f i.succ))

/-- A run of one type as an ordinary list. -/
def uniformToList {V ty} : {n : Nat} → Vals V (List.replicate n ty) → List (V ty)
  | 0, _ => []
  | _ + 1, vs => vs.head :: uniformToList vs.tail

/-- The `i`th of a run --- total, because the index is bounded. -/
def uniformGet {V ty} : {n : Nat} → Vals V (List.replicate n ty) → Fin n → V ty
  | _ + 1, vs, ⟨0, _⟩ => vs.head
  | _ + 1, vs, ⟨i + 1, h⟩ => uniformGet vs.tail ⟨i, by omega⟩

/-- Rebuild a run, position by position, in position order --- which is the
    order the operations are emitted in. -/
def uniformMapIdxM.{u} {m : Type → Type u} [Monad m] {V ty ty'} :
    {n : Nat} → Vals V (List.replicate n ty) → (Nat → V ty → m (V ty')) →
    m (Vals V (List.replicate n ty'))
  | 0, _, _ => pure .nil
  | _ + 1, vs, f => do
      let x ← f 0 vs.head
      let rest ← uniformMapIdxM vs.tail (fun i v => f (i + 1) v)
      return .cons x rest

/-- The slots a value list names, at the emitting instantiation. -/
def slots {tys : List ClifTy} : Vals (fun _ => R) tys → List R
  | .nil => []
  | .cons v vs => v :: slots vs

end Vals

/-- The types a list of carry positions names. `default` for a position past
    the end, which the side condition on `dloop` rules out. -/
def idxTys (tys : List ClifTy) (idx : List Nat) : List ClifTy :=
  idx.map (fun i => (tys[i]?).getD default)

/-- The result a call binds: nothing, for a signature without one. -/
def ResV (V : ClifTy → Type) : Option ClifTy → Type
  | some t => V t
  | none => PUnit

-- ---------------------------------------------------------------------------
-- Operations
-- ---------------------------------------------------------------------------

/-- The lane type of a vector, and itself for anything else. -/
def laneTy (ty : ClifTy) : ClifTy :=
  match ty.lanes with | some (l, _) => l | none => ty

/-- How many lanes a type has; `0` for a scalar, so a lane index into one is
    unsatisfiable. -/
def laneCount (ty : ClifTy) : Nat :=
  match ty.lanes with | some (_, n) => n | none => 0

/-- What a comparison of `ty` yields: a per-lane mask on a vector, `i8` on a
    scalar. -/
def cmpTy (ty : ClifTy) : ClifTy := if ty.isVec then ty else .i8

/-- Pure operations, indexed by the type of the value they bind.

    One constructor per `HProg.Op` constructor, with `Op.check`'s conditions as
    the indices and the side conditions as auto-params: an operation this
    accepts is one `Op.check` types, which is what makes `wf`'s statement
    clause a theorem about the fold rather than a check on the output. -/
inductive Op' (V : ClifTy → Type) : ClifTy → Type where
  | iconst (ty : ClifTy) (k : Int) (h : ty.isInt = true := by decide) : Op' V ty
  | fconst (ty : ClifTy) (bits : UInt64) (h : ty.isFloat = true := by decide) : Op' V ty
  | iadd {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Op' V ty
  | isub {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Op' V ty
  | imul {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Op' V ty
  | udiv {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Op' V ty
  | band {ty} (a b : V ty) (h : (ty.isInt || ty.isVec) = true := by decide) : Op' V ty
  | bandNot {ty} (a b : V ty) (h : (ty.isInt || ty.isVec) = true := by decide) : Op' V ty
  | bor {ty} (a b : V ty) (h : (ty.isInt || ty.isVec) = true := by decide) : Op' V ty
  | bxor {ty} (a b : V ty) (h : (ty.isInt || ty.isVec) = true := by decide) : Op' V ty
  | ineg {ty} (a : V ty) (h : ty.isInt = true := by decide) : Op' V ty
  | ctz {ty} (a : V ty) (h : ty.isInt = true := by decide) : Op' V ty
  | popcnt {ty} (a : V ty) (h : ty.isInt = true := by decide) : Op' V ty
  -- Cranelift lets the shift amount be any integer type, so the two operands
  -- need not agree and the result is the shifted value's own type.
  | ishl {ta tb} (a : V ta) (b : V tb)
      (h : (ta.isInt && tb.isInt) = true := by decide) : Op' V ta
  | ushr {ta tb} (a : V ta) (b : V tb)
      (h : (ta.isInt && tb.isInt) = true := by decide) : Op' V ta
  | ireduce32 {ta} (a : V ta)
      (h : (ta.isInt && decide (ta.width > 32)) = true := by decide) : Op' V .i32
  | uextend64 {ta} (a : V ta)
      (h : (ta.isInt && decide (ta.width < 64)) = true := by decide) : Op' V .i64
  | sextend64 {ta} (a : V ta)
      (h : (ta.isInt && decide (ta.width < 64)) = true := by decide) : Op' V .i64
  | icmp (cond : ICmpCond) {ty} (a b : V ty)
      (h : (ty.isInt || ty.isVec) = true := by decide) : Op' V (cmpTy ty)
  | select {tc ty} (c : V tc) (a b : V ty)
      (h : tc.isInt = true := by decide) : Op' V ty
  -- The mask is the operands' own type: it comes from a comparison that was
  -- bitcast to that width, which is what makes the lane-wise select expressible.
  | bitselect {ty} (c a b : V ty) : Op' V ty
  | fadd {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Op' V ty
  | fsub {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Op' V ty
  | fmul {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Op' V ty
  | fmax {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Op' V ty
  | fmin {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Op' V ty
  | fneg {ty} (a : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Op' V ty
  | fpromote (a : V .f32) : Op' V .f64
  | fcmp (cond : FloatCC) {ty} (a b : V ty)
      (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Op' V (cmpTy ty)
  | fcvtFromSint (ty : ClifTy) {ta} (a : V ta)
      (h : (ta.isInt && ty.isFloat) = true := by decide) : Op' V ty
  | fcvtToUint (ty : ClifTy) {ta} (a : V ta)
      (h : (ta.isFloat && ty.isInt && decide (ty.width ≥ 32)) = true := by decide) : Op' V ty
  | splat (ty : ClifTy) (a : V (laneTy ty)) (h : ty.isVec = true := by decide) : Op' V ty
  | extractlane {ta} (a : V ta) (lane : Nat)
      (h : lane < laneCount ta := by decide) : Op' V (laneTy ta)
  | vhighBits {ta} (a : V ta) (h : ta.isVec = true := by decide) : Op' V .i32
  | bitcast (ty : ClifTy) {ta} (a : V ta)
      (h : (ta.width == ty.width) = true := by decide) : Op' V ty
  | load (op : LoadOp) (a : V .i64) : Op' V op.ty
  | ibin (k : IBin) {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Op' V ty
  | ishift (k : IShift) {ta tb} (a : V ta) (b : V tb)
      (h : (ta.isInt && tb.isInt) = true := by decide) : Op' V ta
  | iun (k : IUn) {ty} (a : V ty) (h : k.admits ty = true := by decide) : Op' V ty
  | fbin (k : FBin) {ty} (a b : V ty)
      (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Op' V ty
  | fun1 (k : FUn) {ty} (a : V ty)
      (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Op' V ty
  | fma {ty} (a b c : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Op' V ty
  | fconv (k : FConv) (ty : ClifTy) {ta} (a : V ta)
      (h : k.admits ta ty = true := by decide) : Op' V ty
  | iext (k : IExt) (ty : ClifTy) {ta} (a : V ta)
      (h : k.admits ta ty = true := by decide) : Op' V ty

/-- The first-order operation, once slots are what the values are. -/
def Op'.erase {ty : ClifTy} : Op' (fun _ => R) ty → Op
  | .iconst t k _ => .iconst t k
  | .fconst t b _ => .fconst t b
  | .iadd a b _ => .iadd a b
  | .isub a b _ => .isub a b
  | .imul a b _ => .imul a b
  | .udiv a b _ => .udiv a b
  | .band a b _ => .band a b
  | .bandNot a b _ => .bandNot a b
  | .bor a b _ => .bor a b
  | .bxor a b _ => .bxor a b
  | .ineg a _ => .ineg a
  | .ctz a _ => .ctz a
  | .popcnt a _ => .popcnt a
  | .ishl a b _ => .ishl a b
  | .ushr a b _ => .ushr a b
  | .ireduce32 a _ => .ireduce32 a
  | .uextend64 a _ => .uextend64 a
  | .sextend64 a _ => .sextend64 a
  | .icmp c a b _ => .icmp c a b
  | .select c a b _ => .select c a b
  | .bitselect c a b => .bitselect c a b
  | .fadd a b _ => .fadd a b
  | .fsub a b _ => .fsub a b
  | .fmul a b _ => .fmul a b
  | .fmax a b _ => .fmax a b
  | .fmin a b _ => .fmin a b
  | .fneg a _ => .fneg a
  | .fpromote a => .fpromote a
  | .fcmp c a b _ => .fcmp c a b
  | .fcvtFromSint t a _ => .fcvtFromSint t a
  | .fcvtToUint t a _ => .fcvtToUint t a
  | .splat t a _ => .splat t a
  | .extractlane a l _ => .extractlane a l
  | .vhighBits a _ => .vhighBits a
  | .bitcast t a _ => .bitcast t a
  | .load op a => .load op a
  | .ibin k a b _ => .ibin k a b
  | .ishift k a b _ => .ishift k a b
  | .iun k a _ => .iun k a
  | .fbin k a b _ => .fbin k a b
  | .fun1 k a _ => .fun1 k a
  | .fma a b c _ => .fma a b c
  | .fconv k t a _ => .fconv k t a
  | .iext k t a _ => .iext k t a

-- ---------------------------------------------------------------------------
-- The term
-- ---------------------------------------------------------------------------

/-- A loop test, with the polarity of the branch that leaves. Both operands
    have one type, which is the agreement `wf` checks by comparison. -/
structure Cond (V : ClifTy → Type) where
  {ty : ClifTy}
  cc : ICmpCond
  a : V ty
  b : V ty
  exitOnTrue : Bool

/-- A call to a function of this same program, or to a symbol resolved inside
    it: what `Ffi` does not cover. The callee travels with the reference, so
    `emit` can collect the table a body needs rather than being handed one.

    No id: the fold gives each callee its position in the table the first time
    the body calls it, so there is nothing to allocate and nothing to collide. -/
structure LocalRef (params : List ClifTy) (result : Option ClifTy) where
  callee : Callee

/-- A function body.

    `V ty` is a value of CLIF type `ty`; `L ex ca` is a label on a loop that
    exits with `ex` and carries `ca`. Both are parameters, so a body cannot
    name a value it did not bind or a loop it is not inside, and `emit` is free
    to choose what they are. -/
inductive Prog (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) :
    Type → Type 1 where
  | ret {α} (a : α) : Prog V L α
  | op {ty α} (o : Op' V ty) (k : V ty → Prog V L α) : Prog V L α
  /-- Stores `ty` bytes under `notrap aligned`; the width is the value's own. -/
  | store {ty α} (v : V ty) (a : V .i64) (k : Prog V L α) : Prog V L α
  /-- Stores under default memory flags. -/
  | storeUnaligned {ty α} (v : V ty) (a : V .i64) (k : Prog V L α) : Prog V L α
  | istore8 {ty α} (v : V ty) (a : V .i64) (h : ty.isInt = true := by decide)
      (k : Prog V L α) : Prog V L α
  | call {α} (f : Ffi) (args : Vals V f.params)
      (k : ResV V f.result → Prog V L α) : Prog V L α
  | callLocal {ps res α} (r : LocalRef ps res) (args : Vals V ps)
      (k : ResV V res → Prog V L α) : Prog V L α
  /-- A top-tested loop. `head` runs each iteration before the test and may
      itself contain loops; it yields the test, what to export on exit, and
      anything `body` needs. `body` yields the next carries. -/
  | loop {tys exitTys β α} (init : Vals V tys)
      (head : L exitTys tys → Vals V tys → Prog V L (Cond V × Vals V exitTys × β))
      (body : L exitTys tys → Vals V tys → β → Prog V L (Vals V tys))
      (k : Vals V exitTys → Prog V L α) : Prog V L α
  /-- A bottom-tested loop: the test is made on the initial carries before
      entry (when `guardIdx` names one) and again at the end of each trip, so
      the body block branches to itself. `body` yields the slot the back-edge
      test reads and the next carries; `exitIdx` says which carries leave. -/
  | dloop {tys α tb} (init : Vals V tys) (cc : ICmpCond) (cb : V tb)
      (guardIdx : Option Nat)
      (hguard : guardIdx.all (fun i => (tys[i]?).getD default == tb) = true := by decide)
      (contOnTrue : Bool) (exitIdx : List Nat)
      (body : L (idxTys tys exitIdx) tys → Vals V tys → Prog V L (V tb × Vals V tys))
      (k : Vals V (idxTys tys exitIdx) → Prog V L α) : Prog V L α
  | ite {jTys α} (c : Cond V) (thn els : Prog V L (Vals V jTys))
      (k : Vals V jTys → Prog V L α) : Prog V L α
  /-- Read the entry block's parameters. They are the one thing a body does
      not bind for itself, so this is where they enter --- and reading them is
      not a statement, so it spends no slot. -/
  | params {α} (tys : List ClifTy) (k : Vals V tys → Prog V L α) : Prog V L α
  /-- Leave the labelled loop with its exit values. Nothing follows. -/
  | br {ex ca α} (l : L ex ca) (args : Vals V ex) : Prog V L α
  /-- Go round the labelled loop again with the next carries. Nothing follows. -/
  | cont {ex ca α} (l : L ex ca) (args : Vals V ca) : Prog V L α


/-- Bind is the continuation each constructor already holds. -/
protected def bind {α β} : Prog V L α → (α → Prog V L β) → Prog V L β
  | .ret a, f => f a
  | .op o k, f => .op o (fun v => Prog.bind (k v) f)
  | .store v a k, f => .store v a (Prog.bind k f)
  | .storeUnaligned v a k, f => .storeUnaligned v a (Prog.bind k f)
  | .istore8 v a h k, f => .istore8 v a h (Prog.bind k f)
  | .call fn args k, f => .call fn args (fun r => Prog.bind (k r) f)
  | .callLocal r args k, f => .callLocal r args (fun x => Prog.bind (k x) f)
  | .loop init head body k, f => .loop init head body (fun vs => Prog.bind (k vs) f)
  | .dloop init cc cb g hg c e body k, f =>
      .dloop init cc cb g hg c e body (fun vs => Prog.bind (k vs) f)
  | .ite c thn els k, f => .ite c thn els (fun vs => Prog.bind (k vs) f)
  | .params tys k, f => .params tys (fun vs => Prog.bind (k vs) f)
  | .br l args, _ => .br l args
  | .cont l args, _ => .cont l args

instance : Monad (Prog V L) where
  pure := .ret
  bind := Prog.bind

-- ---------------------------------------------------------------------------
-- `emit` — the fold to the first-order term
-- ---------------------------------------------------------------------------

/-- What a value is while the body is being emitted: the slot it will occupy. -/
abbrev Slot : ClifTy → Type := fun _ => R

/-- What a label is while the body is being emitted: how many loops enclose the
    one it names. A `br` inside `d` loops targets `d - level - 1`, which is the
    depth the first-order term counts. -/
abbrev Lvl : List ClifTy → List ClifTy → Type := fun _ _ => Nat

/-- A body, ready to emit. Generators write `Prog V L Unit` with both
    parameters bound, so nothing in one can look at a slot number; this is that
    term at the instantiation `emit` chooses. -/
abbrev Body : Type 1 := Prog Slot Lvl Unit

/-- The emitter's state. `n` is the next slot, which is the whole of the
    numbering: everything else accumulates output. -/
structure St where
  n      : Nat
  depth  : Nat
  pieces : List Piece := []      -- reversed
  cur    : List Stmt := []       -- reversed: the open straight-line run
  /-- What the body calls, in the order it first calls each. Position is the
      reference, so this list is both the numbering and the table. -/
  callees : FnEnv := []
  /-- The first thing that went wrong, which only a loop whose head leaves can
      be. Kept so `emitChecked` can refuse the body and say why. -/
  err    : Option String := none

/-- Close the open straight-line run, if there is one. -/
def St.flush (s : St) : St :=
  if s.cur.isEmpty then s
  else { s with pieces := .straight s.cur.reverse :: s.pieces, cur := [] }

def St.note (s : St) (msg : String) : St :=
  if s.err.isSome then s else { s with err := some msg }

/-- Add a statement to the open run. -/
def St.stmt (s : St) (st : Stmt) : St := { s with cur := st :: s.cur }

/-- Hand out the next slot. -/
def St.bind1 (s : St) (st : Stmt) : R × St :=
  (s.n, { s with n := s.n + 1, cur := st :: s.cur })

/-- Record a local callee's signature, which is what the checker reads a
    `local` call's arity from. An import needs no record: its signature is a
    function of the `Ffi` the call already carries. -/
def St.useLocal (s : St) (index : Nat) (params : List ClifTy)
    (result : Option ClifTy) : St :=
  { s with callees := s.callees.withLocal index params result }

/-- The value a call hands its continuation: the slot it bound, or nothing. -/
def resSlot : (res : Option ClifTy) → R → ResV Slot res
  | some _, r => r
  | none, _ => ⟨⟩

/-- The `br`/`cont` depth a label denotes from inside `depth` loops. -/
def labelDepth (depth level : Nat) : Nat := depth - level - 1

/-- `n` consecutive slots from `first`, typed as the list says: the block
    parameters a loop entry, a loop exit or a branch join binds. -/
def carriesFrom (first : Nat) : (tys : List ClifTy) -> Vals Slot tys
  | [] => .nil
  | _ :: ts => .cons first (carriesFrom (first + 1) ts)

/-- Start capturing whole pieces, so a captured region may itself loop. The
    slot counter flows through --- which is what numbers an `ite`'s else arm
    after its then arm. -/
def St.enter (s : St) : St := { s with cur := [], pieces := [] }

/-- Close a captured region and restore what enclosed it. -/
def St.leave (outer inner : St) : List Piece × St :=
  let inner := inner.flush
  (inner.pieces.reverse, { inner with cur := outer.cur, pieces := outer.pieces })

-- Each construct's fold, given how to fold what it contains. `emitGo` and
-- `emitGoS` differ only in how they recurse, so both are these. They are
-- inlined so that in compiled code the continuation is `emitGoS`'s own tail
-- call: through a closure, every foreign call in a body costs a stack frame,
-- and a body with thousands of them overflows.

/-- A call to a foreign function: its answer binds the next slot, if it has one. -/
@[inline] def emitCall {α : Type} (f : Ffi) (args : Vals Slot f.params) (k : ResV Slot f.result → St → Option α × St)
    (s : St) : Option α × St :=
  let c : Callee := .ffi f
  let as := args.slots
  if f.result.isSome then
    let (r, s) := s.bind1 (.call c as)
    k (resSlot f.result r) s
  else
    k (resSlot f.result 0) (s.stmt (.callVoid c as))

/-- A call to one of the program's own functions, recorded in the callee table. -/
@[inline] def emitCallLocal {α : Type} {ps : List ClifTy} {res : Option ClifTy} (r : LocalRef ps res)
    (args : Vals Slot ps) (k : ResV Slot res → St → Option α × St) (s : St) : Option α × St :=
  let s := match r.callee with
    | .local i => s.useLocal i ps res
    | .ffi _ | .native | .atomic _ | .ext _ => s
  let as := args.slots
  if res.isSome then
    let (v, s) := s.bind1 (.call r.callee as)
    k (resSlot res v) s
  else
    k (resSlot res 0) (s.stmt (.callVoid r.callee as))

/-- A top-tested loop. -/
@[inline] def emitLoop {tys exitTys : List ClifTy} {β α : Type} (init : Vals Slot tys)
    (head : Lvl exitTys tys → Vals Slot tys → St → Option (Cond Slot × Vals Slot exitTys × β) × St)
    (body : Lvl exitTys tys → Vals Slot tys → β → St → Option (Vals Slot tys) × St)
    (k : Vals Slot exitTys → St → Option α × St) (s : St) : Option α × St :=
  let s := s.flush
  let firstCarry := s.n
  let carries := carriesFrom firstCarry tys
  let lvl := s.depth
  let s := { s with n := firstCarry + tys.length, depth := s.depth + 1 }
  let (hd, sH) := head lvl carries s.enter
  match hd with
  | none =>
      let (_, s) := St.leave s sH
      (none, { s with depth := lvl }.note "a loop's head leaves the loop")
  | some (c, exitR, x) =>
    -- The exit test is the head's last statement rather than something the
    -- `Loop` carries, so it spends a slot like any operation and the
    -- emitter's slot and value counters stay equal across the loop.
    let (flag, sH) := sH.bind1 (.op (.icmp c.cc c.a c.b))
    let (preCode, s) := St.leave s sH
    -- The body's carries are its block's parameters, the slots after the
    -- head's.
    let bodyCarries := carriesFrom s.n tys
    let s := { s with n := s.n + tys.length }
    let (bd, sB) := body lvl bodyCarries x s.enter
    let (bodyCode, s) := St.leave s sB
    let s := { s with depth := lvl }
    let cont := match bd with | some vs => vs.slots | none => []
    let exits := carriesFrom s.n exitTys
    let s := { s with n := s.n + exitTys.length }
    let l : Loop :=
      { pTys := tys, init := init.slots, flag,
        exitOnTrue := c.exitOnTrue, cont, exitR := exitR.slots, exitTys }
    k exits { s with pieces := .loop l preCode bodyCode :: s.pieces }

/-- A bottom-tested loop. -/
@[inline] def emitDloop {tys : List ClifTy} {α : Type} {tb : ClifTy} (init : Vals Slot tys) (cc : ICmpCond)
    (cb : Slot tb) (guardIdx : Option Nat) (contOnTrue : Bool) (exitIdx : List Nat)
    (body : Lvl (idxTys tys exitIdx) tys → Vals Slot tys → St → Option (Slot tb × Vals Slot tys) × St)
    (k : Vals Slot (idxTys tys exitIdx) → St → Option α × St) (s : St) : Option α × St :=
  -- The guard is the statement before the loop, comparing the initial
  -- carries, and the back-edge test the body's last, comparing what the
  -- trip produced: two ordinary comparisons, each spending a slot, for the
  -- same reason a loop's test is the last statement of its head.
  let (guard, s) := match guardIdx with
    | some gi =>
        let (g, s) := s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))
        (some g, s)
    | none => (none, s)
  let s := s.flush
  let exitTys := idxTys tys exitIdx
  let firstCarry := s.n
  let carries := carriesFrom firstCarry tys
  let lvl := s.depth
  let s := { s with n := firstCarry + tys.length, depth := s.depth + 1 }
  let (bd, sB) := body lvl carries s.enter
  -- A body that answers after a branch whose arms both leave answers from
  -- unreachable code: nothing reaches the back edge, so there is no test to
  -- make.
  let (flag, cont, sB) := match bd with
    | some (ca, vs) =>
        if terminates (St.leave s sB).1 then (0, [], sB)
        else
          let (f, sB) := sB.bind1 (.op (.icmp cc ca cb))
          (f, vs.slots, sB)
    | none => (0, [], sB)
  let (bodyCode, s) := St.leave s sB
  let s := { s with depth := lvl }
  let exits := carriesFrom s.n exitTys
  let s := { s with n := s.n + exitTys.length }
  let l : DLoop :=
    { pTys := tys, init := init.slots, guard, flag, contOnTrue, cont, exitIdx, exitTys }
  k exits { s with pieces := .dloop l bodyCode :: s.pieces }

/-- A branch. -/
@[inline] def emitIte {jTys : List ClifTy} {α : Type} (c : Cond Slot)
    (thn els : St → Option (Vals Slot jTys) × St) (k : Vals Slot jTys → St → Option α × St)
    (s : St) : Option α × St :=
  -- The test is the statement before the branch, for the same reason a
  -- loop's is the last statement of its head.
  let (flag, s) := s.bind1 (.op (.icmp c.cc c.a c.b))
  let s := s.flush
  let (tR, sT) := thn s.enter
  let (thnC, s) := St.leave s sT
  let (eR, sE) := els s.enter
  let (elsC, s) := St.leave s sE
  -- An arm that leaves never reaches the join; when neither reaches it
  -- there is no join block at all.
  let jTys' := if !terminates thnC || !terminates elsC then jTys else []
  -- When neither arm falls through the join block is never entered, so no
  -- slots are spent on it; the continuation is unreachable code and gets
  -- the numbers that block would have bound.
  let joins := carriesFrom s.n jTys
  let s := { s with n := s.n + jTys'.length }
  let thnR := match tR with | some vs => vs.slots | none => []
  let elsR := match eR with | some vs => vs.slots | none => []
  k joins { s with pieces := .ite ⟨flag, jTys'⟩ thnC elsC thnR elsR :: s.pieces }

/-- Leave a loop, or go round it: nothing follows. -/
@[inline] def emitJump (piece : Nat → Piece) (s : St) : Option α × St :=
  let s := s.flush
  (none, { s with pieces := piece s.depth :: s.pieces })

/-- `emitGo` by structural recursion: what compiled code runs (`emitGo_eq`). -/
def emitGoS : {α : Type} → Prog Slot Lvl α → St → Option α × St
  | _, .ret a, s => (some a, s)
  | _, .op o k, s =>
      let (r, s) := s.bind1 (.op o.erase)
      emitGoS (k r) s
  | _, .store (ty := ty) v a k, s => emitGoS k (s.stmt (.store ty v a))
  | _, .storeUnaligned v a k, s => emitGoS k (s.stmt (.storeUnaligned v a))
  | _, .istore8 v a _ k, s => emitGoS k (s.stmt (.istore8 v a))
  | _, .call f args k, s => emitCall f args (fun r => emitGoS (k r)) s
  | _, .callLocal r args k, s => emitCallLocal r args (fun v => emitGoS (k v)) s
  | _, .loop init head body k, s =>
      emitLoop init (fun l cs => emitGoS (head l cs)) (fun l cs x => emitGoS (body l cs x))
        (fun vs => emitGoS (k vs)) s
  | _, .dloop init cc cb guardIdx _ contOnTrue exitIdx body k, s =>
      emitDloop init cc cb guardIdx contOnTrue exitIdx (fun l cs => emitGoS (body l cs))
        (fun vs => emitGoS (k vs)) s
  | _, .ite c thn els k, s => emitIte c (emitGoS thn) (emitGoS els) (fun vs => emitGoS (k vs)) s
  | _, .params tys k, s => emitGoS (k (carriesFrom 0 tys)) s
  -- Nothing follows a `br` or a `cont`: they are where the fold stops.
  | _, .br l args, s => emitJump (fun d => .br (labelDepth d l) args.slots) s
  | _, .cont l args, s => emitJump (fun d => .cont (labelDepth d l) args.slots) s

/-- Fold a body into pieces, handing out slots in the order they are bound.

    The result is `none` exactly when the term left its region --- a `br` or a
    `cont` --- which is what tells `ite` that an arm has no exports and a loop
    body that it needs no back edge.

    Written with the recursor rather than by pattern matching: the kernel
    reduces `Prog.rec` on a constructor in one step, where structural recursion
    goes through `brecOn`, which is slow enough on a real body to exhaust
    memory in a `rfl`. Compiled code runs `emitGoS`, which `emitGo_eq` proves
    the same function. -/
noncomputable def emitGo {α : Type} (p : Prog Slot Lvl α) : St → Option α × St :=
  Prog.rec (motive := fun β _ => St → Option β × St)
    (ret := fun a s => (some a, s))
    (op := fun o _ ih s =>
      let (r, s) := s.bind1 (.op o.erase)
      ih r s)
    (store := fun {ty _} v a _ ih s => ih (s.stmt (.store ty v a)))
    (storeUnaligned := fun v a _ ih s => ih (s.stmt (.storeUnaligned v a)))
    (istore8 := fun v a _ _ ih s => ih (s.stmt (.istore8 v a)))
    (call := fun f args _ ih s => emitCall f args ih s)
    (callLocal := fun r args _ ih s => emitCallLocal r args ih s)
    (loop := fun init _ _ _ ihHead ihBody ihK s => emitLoop init ihHead ihBody ihK s)
    (dloop := fun {_ _ tb} init cc cb guardIdx _ contOnTrue exitIdx _ _ ihBody ihK s =>
      emitDloop (tb := tb) init cc cb guardIdx contOnTrue exitIdx ihBody ihK s)
    (ite := fun c _ _ _ ihThn ihEls ihK s => emitIte c ihThn ihEls ihK s)
    (params := fun tys _ ih s => ih (carriesFrom 0 tys) s)
    (br := fun l args s => emitJump (fun d => .br (labelDepth d l) args.slots) s)
    (cont := fun l args s => emitJump (fun d => .cont (labelDepth d l) args.slots) s)
    p

-- What `emitGo` does to each construct, by definition.
section emitGo_eqns
variable {α : Type}

theorem emitGo_ret (a : α) : emitGo (.ret a) = fun s => (some a, s) := rfl
theorem emitGo_op {ty} (o : Op' Slot ty) (k : Slot ty → Prog Slot Lvl α) :
    emitGo (.op o k) = fun s => let (r, s) := s.bind1 (.op o.erase); emitGo (k r) s := rfl
theorem emitGo_store {ty} (v : Slot ty) (a : Slot .i64) (k : Prog Slot Lvl α) :
    emitGo (.store v a k) = fun s => emitGo k (s.stmt (.store ty v a)) := rfl
theorem emitGo_storeUnaligned {ty} (v : Slot ty) (a : Slot .i64) (k : Prog Slot Lvl α) :
    emitGo (.storeUnaligned v a k) = fun s => emitGo k (s.stmt (.storeUnaligned v a)) := rfl
theorem emitGo_istore8 {ty} (v : Slot ty) (a : Slot .i64) (h : ty.isInt = true) (k : Prog Slot Lvl α) :
    emitGo (.istore8 v a h k) = fun s => emitGo k (s.stmt (.istore8 v a)) := rfl
theorem emitGo_call (f : Ffi) (args : Vals Slot f.params) (k : ResV Slot f.result → Prog Slot Lvl α) :
    emitGo (.call f args k) = emitCall f args (fun r => emitGo (k r)) := rfl
theorem emitGo_callLocal {ps res} (r : LocalRef ps res) (args : Vals Slot ps)
    (k : ResV Slot res → Prog Slot Lvl α) :
    emitGo (.callLocal r args k) = emitCallLocal r args (fun v => emitGo (k v)) := rfl
theorem emitGo_loop {tys exitTys β} (init : Vals Slot tys)
    (head : Lvl exitTys tys → Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × β))
    (body : Lvl exitTys tys → Vals Slot tys → β → Prog Slot Lvl (Vals Slot tys))
    (k : Vals Slot exitTys → Prog Slot Lvl α) :
    emitGo (.loop init head body k) =
      emitLoop init (fun l cs => emitGo (head l cs)) (fun l cs x => emitGo (body l cs x))
        (fun vs => emitGo (k vs)) := rfl
theorem emitGo_dloop {tys tb} (init : Vals Slot tys) (cc : ICmpCond) (cb : Slot tb) (guardIdx : Option Nat)
    (hg : guardIdx.all (fun i => (tys[i]?).getD default == tb) = true) (contOnTrue : Bool)
    (exitIdx : List Nat)
    (body : Lvl (idxTys tys exitIdx) tys → Vals Slot tys → Prog Slot Lvl (Slot tb × Vals Slot tys))
    (k : Vals Slot (idxTys tys exitIdx) → Prog Slot Lvl α) :
    emitGo (.dloop init cc cb guardIdx hg contOnTrue exitIdx body k) =
      emitDloop init cc cb guardIdx contOnTrue exitIdx (fun l cs => emitGo (body l cs))
        (fun vs => emitGo (k vs)) := rfl
theorem emitGo_ite {jTys} (c : Cond Slot) (thn els : Prog Slot Lvl (Vals Slot jTys))
    (k : Vals Slot jTys → Prog Slot Lvl α) :
    emitGo (.ite c thn els k) = emitIte c (emitGo thn) (emitGo els) (fun vs => emitGo (k vs)) := rfl
theorem emitGo_params (tys : List ClifTy) (k : Vals Slot tys → Prog Slot Lvl α) :
    emitGo (.params tys k) = emitGo (k (carriesFrom 0 tys)) := rfl
theorem emitGo_br {ex ca} (l : Lvl ex ca) (args : Vals Slot ex) :
    emitGo (α := α) (.br l args) = emitJump (fun d => .br (labelDepth d l) args.slots) := rfl
theorem emitGo_cont {ex ca} (l : Lvl ex ca) (args : Vals Slot ca) :
    emitGo (α := α) (.cont l args) = emitJump (fun d => .cont (labelDepth d l) args.slots) := rfl

end emitGo_eqns

/-- **The fold compiled code runs is the fold proofs reduce.** -/
@[csimp] theorem emitGo_eq : @emitGo = @emitGoS := by
  funext α p
  induction p <;> funext s <;>
    simp only [emitGo_ret, emitGo_op, emitGo_store, emitGo_storeUnaligned, emitGo_istore8,
      emitGo_call, emitGo_callLocal, emitGo_loop, emitGo_dloop, emitGo_ite, emitGo_params,
      emitGo_br, emitGo_cont, emitGoS, *]

/-- The term a body denotes. Total: an ill-formed body compiles to pieces `wf`
    rejects, which is what `emitChecked` is for. -/
def emit (p : Body) (params : List ClifTy := ptrParams) : Code :=
  let (_, s) := emitGo p { n := params.length, depth := 0 }
  s.flush.pieces.reverse

instance {ps res} : Inhabited (LocalRef ps res) := ⟨{ callee := .local 0 }⟩

/-- A float vector satisfies the condition `fadd` and friends ask for. -/
theorem floatVec_isFloat {ty : ClifTy} (h : ty.isFloatVec = true) :
    (ty.isFloat || ty.isFloatVec) = true := by simp [h]

/-- Comparing a float vector yields a mask of the operands' own width, which is
    what makes `pmin`'s bitcast well-typed. -/
theorem cmpTy_width {ty : ClifTy} (h : ty.isFloatVec = true) :
    ((cmpTy ty).width == ty.width) = true := by
  revert h; cases ty <;> decide

-- ---------------------------------------------------------------------------
-- The surface: the vocabulary a generator writes
--
-- One smart constructor per operation, named as the old builder named it and
-- emitting the same statement, so a body reads the same and the artifact is
-- the same. What has changed is that the arguments are typed.
-- ---------------------------------------------------------------------------

section Surface
variable {α : Type}

/-- The entry block's base pointer. Reading it is not a statement. -/
def basePtr : Prog V L (V .i64) := .params ptrParams (fun vs => .ret vs.head)

/-- Every argument of the entry block, in order, for a call that hands a callee
    exactly what this function was handed. -/
def entryArgs : Prog V L (Vals V ptrParams) := .params ptrParams .ret

/-- The caller's input buffer. -/
def dataPtr : Prog V L (V .i64) :=
  .params ptrParams (fun vs => .ret vs.snd)

/-- How many bytes of input the caller supplied. -/
def dataLen : Prog V L (V .i64) :=
  .params ptrParams (fun vs => .ret vs.thd)

/-- The caller's output buffer. -/
def outPtr : Prog V L (V .i64) :=
  .params ptrParams (fun vs => .ret vs.fth)

/-- How much room the caller left for the answer. -/
def outLen : Prog V L (V .i64) :=
  .params ptrParams (fun vs => .ret vs.fif)

/-- Read the entry block's parameters, for a body whose signature is not an
    entry point's `ptrParams` — a thread worker's spawn argument, say. -/
def entryParams (tys : List ClifTy) : Prog V L (Vals V tys) :=
  .params tys .ret

@[inline] def op {ty} (o : Op' V ty) : Prog V L (V ty) := .op o .ret

def iconst (ty : ClifTy) (k : Int) (h : ty.isInt = true := by decide) :
    Prog V L (V ty) := op (.iconst ty k h)
def iconst64 (k : Int) : Prog V L (V .i64) := iconst .i64 k
def iconst32 (k : Int) : Prog V L (V .i32) := iconst .i32 k
def fconst (ty : ClifTy) (bits : UInt64) (h : ty.isFloat = true := by decide) :
    Prog V L (V ty) := op (.fconst ty bits h)
def fconst32 (x : Float) : Prog V L (V .f32) := fconst .f32 (x.toFloat32.toBits.toUInt64)
def fconst64 (x : Float) : Prog V L (V .f64) := fconst .f64 x.toBits

def iadd {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  op (.iadd a b h)
def isub {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  op (.isub a b h)
def imul {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  op (.imul a b h)
def udiv {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  op (.udiv a b h)
def ineg {ty} (a : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  op (.ineg a h)
def ctz {ty} (a : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  op (.ctz a h)
def popcnt {ty} (a : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  op (.popcnt a h)
def ishl {ta tb} (a : V ta) (b : V tb)
    (h : (ta.isInt && tb.isInt) = true := by decide) : Prog V L (V ta) := op (.ishl a b h)
def ushr {ta tb} (a : V ta) (b : V tb)
    (h : (ta.isInt && tb.isInt) = true := by decide) : Prog V L (V ta) := op (.ushr a b h)
def band {ty} (a b : V ty) (h : (ty.isInt || ty.isVec) = true := by decide) :
    Prog V L (V ty) := op (.band a b h)
def bandNot {ty} (a b : V ty) (h : (ty.isInt || ty.isVec) = true := by decide) :
    Prog V L (V ty) := op (.bandNot a b h)
def bor {ty} (a b : V ty) (h : (ty.isInt || ty.isVec) = true := by decide) :
    Prog V L (V ty) := op (.bor a b h)
def bxor {ty} (a b : V ty) (h : (ty.isInt || ty.isVec) = true := by decide) :
    Prog V L (V ty) := op (.bxor a b h)
def ireduce32 {ta} (a : V ta)
    (h : (ta.isInt && decide (ta.width > 32)) = true := by decide) : Prog V L (V .i32) :=
  op (.ireduce32 a h)
def uextend64 {ta} (a : V ta)
    (h : (ta.isInt && decide (ta.width < 64)) = true := by decide) : Prog V L (V .i64) :=
  op (.uextend64 a h)
def sextend64 {ta} (a : V ta)
    (h : (ta.isInt && decide (ta.width < 64)) = true := by decide) : Prog V L (V .i64) :=
  op (.sextend64 a h)
def icmp (c : ICmpCond) {ty} (a b : V ty)
    (h : (ty.isInt || ty.isVec) = true := by decide) : Prog V L (V (cmpTy ty)) :=
  op (.icmp c a b h)
def select {tc ty} (c : V tc) (a b : V ty) (h : tc.isInt = true := by decide) :
    Prog V L (V ty) := op (.select c a b h)
def bitselect {ty} (c a b : V ty) : Prog V L (V ty) := op (.bitselect c a b)
def fadd {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := op (.fadd a b h)
def fsub {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := op (.fsub a b h)
def fmul {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := op (.fmul a b h)
def fmax {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := op (.fmax a b h)
def fmin {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := op (.fmin a b h)
def fneg {ty} (a : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := op (.fneg a h)
def fpromote (a : V .f32) : Prog V L (V .f64) := op (.fpromote a)
def fcmp (c : FloatCC) {ty} (a b : V ty)
    (h : (ty.isFloat || ty.isFloatVec) = true := by decide) : Prog V L (V (cmpTy ty)) :=
  op (.fcmp c a b h)
def fcvtFromSint (ty : ClifTy) {ta} (a : V ta)
    (h : (ta.isInt && ty.isFloat) = true := by decide) : Prog V L (V ty) :=
  op (.fcvtFromSint ty a h)
def fcvtToUint (ty : ClifTy) {ta} (a : V ta)
    (h : (ta.isFloat && ty.isInt && decide (ty.width ≥ 32)) = true := by decide) :
    Prog V L (V ty) :=
  op (.fcvtToUint ty a h)
def splat (ty : ClifTy) (a : V (laneTy ty)) (h : ty.isVec = true := by decide) :
    Prog V L (V ty) := op (.splat ty a h)
def extractlane {ta} (a : V ta) (lane : Nat) (h : lane < laneCount ta := by decide) :
    Prog V L (V (laneTy ta)) := op (.extractlane a lane h)
def vhighBits {ta} (a : V ta) (h : ta.isVec = true := by decide) : Prog V L (V .i32) :=
  op (.vhighBits a h)
def bitcast (ty : ClifTy) {ta} (a : V ta) (h : (ta.width == ty.width) = true := by decide) :
    Prog V L (V ty) := op (.bitcast ty a h)

def ibin (k : IBin) {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  op (.ibin k a b h)
def sdiv {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  ibin .sdiv a b h
def urem {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  ibin .urem a b h
def srem {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  ibin .srem a b h
def smin {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  ibin .smin a b h
def smax {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  ibin .smax a b h
def umin {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  ibin .umin a b h
def umax {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  ibin .umax a b h
def umulhi {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  ibin .umulhi a b h
def smulhi {ty} (a b : V ty) (h : ty.isInt = true := by decide) : Prog V L (V ty) :=
  ibin .smulhi a b h
def ishift (k : IShift) {ta tb} (a : V ta) (b : V tb)
    (h : (ta.isInt && tb.isInt) = true := by decide) : Prog V L (V ta) := op (.ishift k a b h)
def sshr {ta tb} (a : V ta) (b : V tb)
    (h : (ta.isInt && tb.isInt) = true := by decide) : Prog V L (V ta) :=
  ishift .sshr a b h
def rotl {ta tb} (a : V ta) (b : V tb)
    (h : (ta.isInt && tb.isInt) = true := by decide) : Prog V L (V ta) :=
  ishift .rotl a b h
def rotr {ta tb} (a : V ta) (b : V tb)
    (h : (ta.isInt && tb.isInt) = true := by decide) : Prog V L (V ta) :=
  ishift .rotr a b h
def iun (k : IUn) {ty} (a : V ty) (h : k.admits ty = true := by decide) : Prog V L (V ty) :=
  op (.iun k a h)
def bnot {ty} (a : V ty) (h : IUn.bnot.admits ty = true := by decide) : Prog V L (V ty) :=
  iun .bnot a h
def iabs {ty} (a : V ty) (h : IUn.iabs.admits ty = true := by decide) : Prog V L (V ty) :=
  iun .iabs a h
def clz {ty} (a : V ty) (h : IUn.clz.admits ty = true := by decide) : Prog V L (V ty) :=
  iun .clz a h
def bswap {ty} (a : V ty) (h : IUn.bswap.admits ty = true := by decide) : Prog V L (V ty) :=
  iun .bswap a h
def bitrev {ty} (a : V ty) (h : IUn.bitrev.admits ty = true := by decide) : Prog V L (V ty) :=
  iun .bitrev a h
def fbin (k : FBin) {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := op (.fbin k a b h)
def fdiv {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := fbin .fdiv a b h
def fcopysign {ty} (a b : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := fbin .fcopysign a b h
def fun1 (k : FUn) {ty} (a : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := op (.fun1 k a h)
def sqrt {ty} (a : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := fun1 .sqrt a h
def fabs {ty} (a : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := fun1 .fabs a h
def ceil {ty} (a : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := fun1 .ceil a h
def floor {ty} (a : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := fun1 .floor a h
def trunc {ty} (a : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := fun1 .trunc a h
def nearest {ty} (a : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := fun1 .nearest a h
def fma {ty} (a b c : V ty) (h : (ty.isFloat || ty.isFloatVec) = true := by decide) :
    Prog V L (V ty) := op (.fma a b c h)
def fconv (k : FConv) (ty : ClifTy) {ta} (a : V ta) (h : k.admits ta ty = true := by decide) :
    Prog V L (V ty) := op (.fconv k ty a h)
def fcvtToSint (ty : ClifTy) {ta} (a : V ta)
    (h : FConv.toSint.admits ta ty = true := by decide) : Prog V L (V ty) :=
  fconv .toSint ty a h
def fcvtFromUint (ty : ClifTy) {ta} (a : V ta)
    (h : FConv.fromUint.admits ta ty = true := by decide) : Prog V L (V ty) :=
  fconv .fromUint ty a h
def iext (k : IExt) (ty : ClifTy) {ta} (a : V ta) (h : k.admits ta ty = true := by decide) :
    Prog V L (V ty) := op (.iext k ty a h)
def ireduce (ty : ClifTy) {ta} (a : V ta) (h : IExt.reduce.admits ta ty = true := by decide) :
    Prog V L (V ty) := iext .reduce ty a h
def uextend (ty : ClifTy) {ta} (a : V ta) (h : IExt.uextend.admits ta ty = true := by decide) :
    Prog V L (V ty) := iext .uextend ty a h
def sextend (ty : ClifTy) {ta} (a : V ta) (h : IExt.sextend.admits ta ty = true := by decide) :
    Prog V L (V ty) := iext .sextend ty a h
def fdemote (a : V .f64) : Prog V L (V .f32) := fconv .demote .f32 a

def load (o : LoadOp) (a : V .i64) : Prog V L (V o.ty) := op (.load o a)
def uload16_64 (a : V .i64) : Prog V L (V .i64) := load { kind := .uload16, ty := .i64 } a
def sload16_64 (a : V .i64) : Prog V L (V .i64) := load { kind := .sload16, ty := .i64 } a
def sload32_64 (a : V .i64) : Prog V L (V .i64) := load { kind := .sload32, ty := .i64 } a
def load64 (a : V .i64) : Prog V L (V .i64) := load { ty := .i64 } a
def load32 (a : V .i64) : Prog V L (V .i32) := load { ty := .i32 } a
def load_i8 (a : V .i64) : Prog V L (V .i8) := load { ty := .i8 } a
def load_i16 (a : V .i64) : Prog V L (V .i16) := load { ty := .i16 } a
def uload8_64 (a : V .i64) : Prog V L (V .i64) := load { kind := .uload8, ty := .i64 } a
def sload8_64 (a : V .i64) : Prog V L (V .i64) := load { kind := .sload8, ty := .i64 } a
def uload32_64 (a : V .i64) : Prog V L (V .i64) := load { kind := .uload32, ty := .i64 } a
def loadF32 (a : V .i64) : Prog V L (V .f32) := load { ty := .f32, notrapAligned := true } a
def loadF32x4 (a : V .i64) : Prog V L (V .f32x4) :=
  load { ty := .f32x4, notrapAligned := true } a
def loadI8x16 (a : V .i64) : Prog V L (V .i8x16) :=
  load { ty := .i8x16, notrapAligned := true } a

/-- Stores under `notrap aligned`, typed from the value. -/
def store {ty} (v : V ty) (a : V .i64) : Prog V L Unit := .store v a (.ret ())
/-- Stores exactly `ty` bytes; the value must already carry that width, which
    is now the type rather than a check on the emitted term. -/
def storeW (ty : ClifTy) (v : V ty) (a : V .i64) : Prog V L Unit := .store v a (.ret ())
def storeI32 (v : V .i32) (a : V .i64) : Prog V L Unit := storeW .i32 v a
def storeI64 (v : V .i64) (a : V .i64) : Prog V L Unit := storeW .i64 v a
def storeUnaligned {ty} (v : V ty) (a : V .i64) : Prog V L Unit :=
  .storeUnaligned v a (.ret ())
def istore8 {ty} (v : V ty) (a : V .i64) (h : ty.isInt = true := by decide) :
    Prog V L Unit := .istore8 v a h (.ret ())
/-- The low 16 or 32 bits of a wider integer, stored: a narrowing and a store
    at the narrow type, which Cranelift emits as one narrow store. -/
def istore16 {ty} (v : V ty) (a : V .i64)
    (h : IExt.reduce.admits ty .i16 = true := by decide) : Prog V L Unit := do
  storeW .i16 (← ireduce .i16 v h) a
def istore32 {ty} (v : V ty) (a : V .i64)
    (h : IExt.reduce.admits ty .i32 = true := by decide) : Prog V L Unit := do
  storeW .i32 (← ireduce .i32 v h) a

/-- Call an entry point. The signature says what it takes and whether it binds
    a result, so a generator cannot pass the wrong list or read a result that
    does not exist. -/
def ffi (f : Ffi) (args : Vals V f.params) : Prog V L (ResV V f.result) :=
  .call f args .ret
/-- The same, for the effect alone. -/
def ffiVoid (f : Ffi) (args : Vals V f.params) : Prog V L Unit := do
  let _ ← ffi (L := L) f args
  pure ()

/-- Call another function of this program, or a symbol resolved inside it. -/
def callLocal {ps res} (r : LocalRef ps res) (args : Vals V ps) :
    Prog V L (ResV V res) := .callLocal r args .ret
def callLocalVoid {ps res} (r : LocalRef ps res) (args : Vals V ps) : Prog V L Unit := do
  let _ ← callLocal (L := L) r args
  pure ()

/-- **Atomics.** Each travels as a call to `Callee.atomic`, which the engine
    emits as the instruction itself; the address comes first. An integer width
    and a naturally aligned address are what the model admits. -/
def atomicLoad (ty : ClifTy) (p : V .i64) : Prog V L (V ty) :=
  callLocal (⟨.atomic (.load ty)⟩ : LocalRef [.i64] (some ty)) %[p]
def atomicStore {ty} (v : V ty) (p : V .i64) : Prog V L Unit :=
  callLocalVoid (⟨.atomic (.store ty)⟩ : LocalRef [ty, .i64] none) %[v, p]
/-- The old value, having left `k old x` in its place. -/
def atomicRmw (k : AtomicRmw) {ty} (p : V .i64) (x : V ty) : Prog V L (V ty) :=
  callLocal (⟨.atomic (.rmw ty k)⟩ : LocalRef [.i64, ty] (some ty)) %[p, x]
/-- The old value, having replaced it with `n` exactly when it was `e`. -/
def atomicCas {ty} (p : V .i64) (e n : V ty) : Prog V L (V ty) :=
  callLocal (⟨.atomic (.cas ty)⟩ : LocalRef [.i64, ty, ty] (some ty)) %[p, e, n]
def fence : Prog V L Unit :=
  callLocalVoid (⟨.atomic .fence⟩ : LocalRef [] none) %[]

/-- **A C library function, called directly**, at the signature its
    declaration gives. -/
def ext (e : Ext) (args : Vals V e.sig.1) : Prog V L (ResV V e.sig.2) :=
  callLocal (⟨.ext e⟩ : LocalRef e.sig.1 e.sig.2) args
/-- Whether a library loaded on this machine: `1` or `0`. -/
def libPresent (l : Lib) : Prog V L (V .i32) := ext (.present l) %[]

/-- `min(a, b)` with the hardware's NaN behaviour rather than IEEE
    `minimumNumber`. Cranelift lowers exactly this shape --- the wasm `pmin`
    pattern --- to a single `minps`, where `fmin` costs a NaN-correct sequence. -/
def pmin (ty : ClifTy) (a b : V ty) (h : ty.isFloatVec = true := by decide) :
    Prog V L (V ty) := do
  bitselect (← bitcast ty (← fcmp .lt a b (floatVec_isFloat h)) (cmpTy_width h)) a b

/-- The address of `base + off`, the shape every fixed-offset access takes. -/
def absAddr (base : V .i64) (off : Int) : Prog V L (V .i64) := do
  iadd base (← iconst64 off)

/-- An `i64` immediate and the operation that consumes it. -/
def iaddImm (a : V .i64) (imm : Int) : Prog V L (V .i64) := do iadd a (← iconst64 imm)

/-- Store at `base + offset`, under default memory flags. -/
def storeAt {ty} (base : V .i64) (offset : Nat) (val : V ty) : Prog V L Unit := do
  storeUnaligned val (← absAddr base offset)
/-- Shift by a constant. The amount is an `i64` whatever the value's width is,
    which is what CLIF's shifts take. -/
def ishlImm {ta} (a : V ta) (imm : Int) (h : ta.isInt = true := by decide) :
    Prog V L (V ta) := do ishl a (← iconst64 imm) (by rw [Bool.and_eq_true]; exact ⟨h, rfl⟩)
def ushrImm {ta} (a : V ta) (imm : Int) (h : ta.isInt = true := by decide) :
    Prog V L (V ta) := do ushr a (← iconst64 imm) (by rw [Bool.and_eq_true]; exact ⟨h, rfl⟩)

-- ---------------------------------------------------------------------------
-- Control flow
-- ---------------------------------------------------------------------------

/-- Leave the loop when `cc a b` holds. -/
def exitIf {ty} (cc : ICmpCond) (a b : V ty) : Cond V :=
  { cc, a, b, exitOnTrue := true }
/-- Stay in the loop while `cc a b` holds. -/
def contIf {ty} (cc : ICmpCond) (a b : V ty) : Cond V :=
  { cc, a, b, exitOnTrue := false }

def exitIfEq {ty} (a b : V ty) : Cond V := exitIf .eq a b
def exitIfSGe {ty} (a b : V ty) : Cond V := exitIf .sge a b
def contIfULt {ty} (a b : V ty) : Cond V := contIf .ult a b
def contIfULe {ty} (a b : V ty) : Cond V := contIf .ule a b

/-- Leave the labelled loop with its exit values. Nothing follows, which is why
    this is typed at every result: it is the end of its region. -/
def brk {ex ca} (l : L ex ca) (args : Vals V ex) : Prog V L α := .br l args
/-- Go round the labelled loop again with the next carries. -/
def continueWith {ex ca} (l : L ex ca) (args : Vals V ca) : Prog V L α := .cont l args

/-- A top-tested loop, with the label its body may leave by.

    `head` receives the carries and yields the condition, the values to export
    on exit, and anything `body` needs; `body` yields the next carries. -/
def wloopL {tys exitTys β} (init : Vals V tys)
    (head : L exitTys tys → Vals V tys → Prog V L (Cond V × Vals V exitTys × β))
    (body : L exitTys tys → Vals V tys → β → Prog V L (Vals V tys)) :
    Prog V L (Vals V exitTys) := .loop init head body .ret

/-- The same, where nothing leaves the loop early. -/
def wloop {tys exitTys β} (init : Vals V tys)
    (head : Vals V tys → Prog V L (Cond V × Vals V exitTys × β))
    (body : Vals V tys → β → Prog V L (Vals V tys)) :
    Prog V L (Vals V exitTys) := wloopL init (fun _ => head) (fun _ => body)

/-- One-carry loop, with the carry as a plain binder. -/
def wloop1 {t exitTys β} (init : V t)
    (head : V t → Prog V L (Cond V × Vals V exitTys × β))
    (body : V t → β → Prog V L (Vals V (t :: []))) :
    Prog V L (Vals V exitTys) :=
  wloop %[init] (fun cs => head cs.head) (fun cs x => body cs.head x)

/-- One-carry loop, with the label its body may leave by. -/
def wloop1L {t exitTys β} (init : V t)
    (head : L exitTys [t] → V t → Prog V L (Cond V × Vals V exitTys × β))
    (body : L exitTys [t] → V t → β → Prog V L (Vals V [t])) :
    Prog V L (Vals V exitTys) :=
  wloopL %[init] (fun l cs => head l cs.head) (fun l cs x => body l cs.head x)

/-- Two-carry loop. -/
def wloop2 {t u exitTys β} (a : V t) (b : V u)
    (head : V t → V u → Prog V L (Cond V × Vals V exitTys × β))
    (body : V t → V u → β → Prog V L (Vals V [t, u])) :
    Prog V L (Vals V exitTys) :=
  wloop %[a, b] (fun cs => head cs.head cs.snd) (fun cs x => body cs.head cs.snd x)

/-- Two-carry loop, with the label its body may leave by. -/
def wloop2L {t u exitTys β} (a : V t) (b : V u)
    (head : L exitTys [t, u] → V t → V u → Prog V L (Cond V × Vals V exitTys × β))
    (body : L exitTys [t, u] → V t → V u → β → Prog V L (Vals V [t, u])) :
    Prog V L (Vals V exitTys) :=
  wloopL %[a, b] (fun l cs => head l cs.head cs.snd) (fun l cs x => body l cs.head cs.snd x)

/-- A bottom-tested loop: the test is made once before entry and again at the
    end of each trip, so the body block branches to itself and an empty input
    never enters. -/
def dwloopL {tys tb} (init : Vals V tys) (cc : ICmpCond) (cb : V tb)
    (contOnTrue : Bool) (exitIdx : List Nat)
    (body : L (idxTys tys exitIdx) tys → Vals V tys → Prog V L (V tb × Vals V tys))
    (guardIdx : Option Nat := none)
    (hguard : guardIdx.all (fun i => (tys[i]?).getD default == tb) = true := by decide) :
    Prog V L (Vals V (idxTys tys exitIdx)) :=
  .dloop init cc cb guardIdx hguard contOnTrue exitIdx body .ret

/-- The same, where nothing leaves the loop early. -/
def dwloop {tys tb} (init : Vals V tys) (cc : ICmpCond) (cb : V tb)
    (contOnTrue : Bool) (exitIdx : List Nat)
    (body : Vals V tys → Prog V L (V tb × Vals V tys))
    (guardIdx : Option Nat := none)
    (hguard : guardIdx.all (fun i => (tys[i]?).getD default == tb) = true := by decide) :
    Prog V L (Vals V (idxTys tys exitIdx)) :=
  dwloopL init cc cb contOnTrue exitIdx (fun _ => body) guardIdx hguard

/-- A counted loop over `[0, limit)`, carrying nothing else --- the shape most
    generators want. The counter's `iconst 0` is emitted just before entry and
    the increment after everything the body did, which fixes the emission order
    here rather than at each call site. -/
def forLoop (limit : V .i64) (body : V .i64 → Prog V L Unit) : Prog V L Unit := do
  let _ ← wloop1 (← iconst64 0)
    (head := fun i => return (contIfULt i limit, %[], ()))
    (body := fun i _ => do body i; return %[← iaddImm i 1])
  pure ()

/-- A counted loop from `start`. -/
def forLoopFromTo (start limit : V .i64) (body : V .i64 → Prog V L Unit) :
    Prog V L Unit := do
  let _ ← wloop1 start
    (head := fun i => return (contIfULt i limit, %[], ()))
    (body := fun i _ => do body i; return %[← iaddImm i 1])
  pure ()

/-- A counted loop threading one accumulator. The accumulator's initial value
    is a value the caller made, so it is emitted before the counter's. -/
def forLoopAcc {t} (limit : V .i64) (acc0 : V t)
    (body : V .i64 → V t → Prog V L (V t)) : Prog V L (V t) := do
  let e ← wloop2 (← iconst64 0) acc0
    (head := fun i a => return (contIfULt i limit, %[a], ()))
    (body := fun i a _ => do
      let nextAcc ← body i a
      return %[← iaddImm i 1, nextAcc])
  return e.head

/-- A counted loop threading two accumulators. -/
def forLoopAcc2 {t u} (limit : V .i64) (ia : V t) (ib : V u)
    (body : V .i64 → V t → V u → Prog V L (V t × V u)) : Prog V L (V t × V u) := do
  let e ← wloop %[← iconst64 0, ia, ib]
    (head := fun c => return (contIfULt c.head limit, %[c.snd, c.thd], ()))
    (body := fun c _ => do
      let (nx, ny) ← body c.head c.snd c.thd
      return %[← iaddImm c.head 1, nx, ny])
  return (e.head, e.snd)

/-- A while loop over two carries: the condition is computed at the head and
    both carries leave. -/
def whileLoop2 {t u} (ia : V t) (ib : V u) (cond : V t → V u → Prog V L (V .i8))
    (body : V t → V u → Prog V L (V t × V u)) : Prog V L (V t × V u) := do
  let e ← wloop2 ia ib
    (head := fun x y => do
      let ok ← cond x y
      let z ← iconst .i8 0
      return (contIf .ne ok z, %[x, y], ()))
    (body := fun x y _ => do
      let (nx, ny) ← body x y
      return %[nx, ny])
  return (e.head, e.snd)

/-- Two-way branch with join values: each arm yields its exports, and the
    result carries whichever arm ran. -/
def ifte {jTys ty} (cc : ICmpCond) (a b : V ty)
    (thn els : Prog V L (Vals V jTys)) : Prog V L (Vals V jTys) :=
  .ite { cc, a, b, exitOnTrue := true } thn els .ret

/-- A branch taken for effect, joining no values. -/
def when {ty} (cc : ICmpCond) (a b : V ty) (thn : Prog V L Unit) : Prog V L Unit := do
  let _ ← ifte (jTys := []) cc a b (do thn; pure %[]) (pure %[])
  pure ()

end Surface

-- ---------------------------------------------------------------------------
-- Folding a body, and the door it passes
-- ---------------------------------------------------------------------------

/-- Which top-level piece first makes the body ill-formed, found by checking
    growing prefixes. Only ever run on a body already known to be bad. -/
def firstBadPiece (env : FnEnv) (params : List ClifTy) (c : Code) : Nat :=
  (List.range c.length).find?
      (fun k => !(HProg.wfGo env fuel [] (HProg.TyEnv.ofList params) (c.take (k + 1))).1)
    |>.getD c.length

/-- Everything one fold of a body yields: its answer, its term, the table it
    needs, and what went wrong, if anything.

    Generic in the answer because a body that names a status is the same fold
    as one that names nothing --- only the terminator differs. -/
def runAns {α} (p : Prog Slot Lvl α) (params : List ClifTy := ptrParams) :
    Option α × Code × FnEnv × Option String :=
  let (a, s) := emitGo p { n := params.length, depth := 0 }
  let s := s.flush
  (a, s.pieces.reverse, s.callees, s.err)

/-- Everything one fold of a body yields, for a body that answers nothing. -/
def run (p : Body) (params : List ClifTy := ptrParams) :
    Code × FnEnv × Option String :=
  (runAns p params).2

/-- The callees a body calls, in the order it calls them.

    A `call` carries its callee, so a claim about *what* a body calls reads it
    off the term with nothing to resolve. The names are the symbols the JIT
    resolves, which is what such a claim is about; a call to one of the
    program's own functions is its `u0:N`. -/
def callNames (p : Body) (params : List ClifTy := ptrParams) : List String :=
  let (c, _, _) := run p params
  (HProg.callsOf c).map fun
    | .ffi f   => f.cname
    | .local k => s!"u0:{k}"
    | .native  => "native"
    | .atomic _ => "atomic"
    | .ext e => s!"{e.lib.name}!{e.symbol}"

/-- The term a body denotes, or why it is not one.

    This is the gate every shipped body passes: `wf` names what the types do
    not --- that a body assembled from well-typed parts is itself well-formed
    --- and a generator that fails it stops rather than writing an artifact. -/
def emitChecked (p : Body) (params : List ClifTy := ptrParams) :
    Except String Code :=
  let (c, env, err) := run p params
  match err with
  | some e => .error e
  | none =>
      if !wf env params c then .error
        s!"the body is not well-formed, from piece {firstBadPiece env params c} on"
      else if !scopeOk params c then .error
        "the body reads a slot that is not bound on every path to the read"
      else .ok c

/-- **The function a body compiles to, and why it must not ship, if it must
    not.**

    One computation, projected two ways. `stateOf` is its function and
    `compileProg` is its verdict, so a claim stated over the one is a claim
    about the other: there is a single compiled form, not a shipped one and a
    proven one that a theorem has to hold together. The two names denote the
    same subterm rather than being provably equal, so they cannot drift. -/
def compile (idx : Nat) (p : Body) (params : List ClifTy := ptrParams) :
    FuncData × Option String :=
  let (c, env, err) := run p params
  (HProg.compileBody idx c env params,
   match err with
   | some e => some s!"function {idx}: {e}"
   | none =>
       if !wf env params c then some s!"function {idx} is not well-formed, from piece \
         {firstBadPiece env params c} on"
       else if !scopeOk params c then some s!"function {idx} reads a slot that is not \
         bound on every path to the read"
       else none)

/-- The compiled form of a body, against the table it derives, without the
    check.

    A *view*: what a claim about the emitted CLIF is stated over, applied to
    whatever body the claim is about. Nothing reaches an artifact through it
    --- `ShipScan` fails a generator whose `main` does --- because on its own
    it says nothing about whether the body was fit to ship. -/
def stateOf (idx : Nat) (p : Body) (params : List ClifTy := ptrParams) : FuncData :=
  (compile idx p params).1

/-- Compile a body to the function an artifact ships.

    Every path from a `Prog` to an artifact runs through here, which is what
    `ShipScan` checks. What it returns is `stateOf` --- the same subterm, not a
    second copy of it --- so the function that ships is the function the
    theorems name, by definition rather than by argument. -/
def compileProg (idx : Nat) (p : Body) (params : List ClifTy := ptrParams) :
    Except String FuncData :=
  let r := compile idx p params
  match r.2 with
  | none   => .ok r.1
  | some e => .error e

/-- A body that answers.

    The `i64` it names is the status `execute` hands back to a host --- the one
    value a program returns without agreeing on a place in memory to leave it.
    Bodies that answer nothing stay `Body`, which is why adopting this costs a
    generator nothing until it wants to. -/
abbrev StatusBody : Type 1 := Prog Slot Lvl (Slot .i64)

/-- `compile`, for a body that answers: the slot its fold ends on becomes the
    function's `return`, and with it the `i64` in its signature. -/
def compileStatus (idx : Nat) (p : StatusBody) (params : List ClifTy := ptrParams) :
    FuncData × Option String :=
  let (a, c, env, err) := runAns p params
  (HProg.compileBody idx c env params a,
   match err with
   | some e => some s!"function {idx}: {e}"
   | none =>
       if a.isNone then some s!"function {idx} leaves its own body --- by a \
         `br` or a `cont` --- rather than reaching a status"
       else if !wf env params c then some s!"function {idx} is not well-formed, from piece \
         {firstBadPiece env params c} on"
       else if !retOk params c a then some s!"function {idx} reads a slot that is not \
         bound on every path to the read, or answers with one"
       else none)

/-- Compile a body that answers to the function an artifact ships. The checked
    door for `StatusBody`, beside `compileProg`. -/
def compileProgStatus (idx : Nat) (p : StatusBody)
    (params : List ClifTy := ptrParams) : Except String FuncData :=
  let r := compileStatus idx p params
  match r.2 with
  | none   => .ok r.1
  | some e => .error e

/-- The functions of an artifact, in `u0:N` order, or the first failure.
/-- **The view is the shipped function.** True by `rfl` on the projection;
    stated because it is the property the single-form discipline exists to
    have, and a reader should be able to find it named. -/
theorem compileProg_eq_stateOf {idx : Nat} {p : Body} {params : List ClifTy}
    {fd : FuncData} (h : compileProg idx p params = .ok fd) :
    fd = stateOf idx p params := by
  simp only [compileProg, stateOf] at h ⊢
  split at h
  · exact (Except.ok.inj h).symm
  · cases h

/-- **The function index is only the index.**

    A body's blocks and its callee table do not depend on the position it is
    compiled at; `idx` reaches nothing but the `index` field.

    This is stated because of how it is used. A theorem about an emitted body
    is written over `stateOf i p`, and the artifact may ship that body at some
    other position `j` — `MlpCifar` states at 1 and ships at 20. Without this,
    the two are related by an argument nobody wrote down, which is the shape of
    defect the single-form discipline above exists to prevent; it would be an
    odd thing to close for `compileProg` and leave open one line away. `rfl`
    each, so the cost of having it is nothing and the cost of not having it was
    a reader's trust. -/
theorem stateOf_index {i j : Nat} {p : Body} {params : List ClifTy} :
    { stateOf i p params with index := j } = stateOf j p params := rfl

theorem stateOf_blocks_index {i j : Nat} {p : Body} {params : List ClifTy} :
    (stateOf i p params).blocks = (stateOf j p params).blocks := rfl


    Refused: a function whose position is not the index it was compiled at ---
    the artifact carries only the position, and a call names a callee by it ---
    and a name exported twice. -/
def program (fs : List (Except String FuncData)) : Except String (List FuncData) := do
  let fs ← fs.mapM id
  for (f, i) in fs.zipIdx do
    if f.index != i then
      throw s!"the function at position {i} was compiled as u0:{f.index}"
  let names := fs.filterMap (·.entryName)
  match names.find? (fun n => names.count n > 1) with
  | some n => throw s!"more than one function is exported as {n}"
  | none => pure fs

/-- `f`, exported as `name`: the name a host calls it by, which stays put
    wherever the function is placed. -/
def entry (name : String) (f : Except String FuncData) : Except String FuncData :=
  f.map fun d => { d with entryName := some name }

/-- Unwrap in a generator's `main`: an ill-formed body is a build failure with
    a message, not an artifact. -/
def orDie {α} : Except String α → IO α
  | .ok a => pure a
  | .error e => throw (IO.userError e)

end AlgorithmLib.Prog
