import Lean
import AlgorithmLib.IR

/-!
# `HProg` — a function body as a first-order term

A generator written against `IR.IRBuilder` is a `StateM` action: it exists only
while it runs, so nothing can be stated about it. `HProg` is the same programs
as *data* — one inductive term per function body, which `compileFn` turns into
the shipped `IR.FuncData`.

Three things follow from the term being first-order:

* **It can be checked.** `wf` decides slot scoping *and* types by kernel
  computation, so `by decide` proves a generator well-typed. `IR.Val` carries no
  type, and until this check existed an `f32`/`i64` mix-up surfaced only when
  Cranelift rejected the finished artifact.
* **It can be observed.** `callsOf` reads the FFI calls in program order —
  the observation the compilation proof is stated against.
* **It can be written with binders anyway.** `Sur` is an ordinary builder over
  the same term, and `clif%` runs it *while the file elaborates*, splicing the
  first-order literal it produced. Binders at the surface, data in the artifact.

## Control flow

Four constructs: `loop` (top-tested), `dloop` (bottom-tested, so the body block
branches to itself and there is no header), `ite`, and `br`, which leaves the
`depth`-th enclosing loop. Measured over every emitted artifact, all 300
functions have reducible control flow, and these cover the shapes the generators
emit. A loop's condition prefix is a whole `Code`, so a test that follows an
inner loop is expressible too.

## Types

Slots are typed, and the two constructs that bind block parameters (`loop`
carries and exits, `ite` joins) carry their types in the term. So `compileFn`
performs no inference: it reads the annotation. `wf` is where inference lives,
checking each annotation against the types it computes for the operands. The
`Sur` builder fills the annotations in from the types it tracks, so a generator
written at the surface never writes one by hand.
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
  | .fcmp _ a b => [a, b]
  | .ineg a | .ctz a | .popcnt a | .ireduce32 a | .uextend64 a
  | .sextend64 a | .fneg a | .fpromote a | .vhighBits a
  | .fcvtFromSint _ a | .fcvtToUint _ a | .splat _ a
  | .extractlane a _ | .bitcast _ a | .load _ a => [a]
  | .select c a b | .bitselect c a b => [c, a, b]

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

/-- Statements. `op` and `call` define the next slot; the rest define nothing. -/
inductive Stmt where
  | op       (o : Op)
  /-- Stores under `notrap aligned`; `ty` documents the width the value carries. -/
  | store    (ty : ClifTy) (v a : R)
  /-- Stores under default memory flags. -/
  | storeUnaligned (v a : R)
  | istore8  (v a : R)
  /-- A call whose result binds the next slot. -/
  | call     (fn : Nat) (args : List R)
  /-- A call to a signature with no result. -/
  | callVoid (fn : Nat) (args : List R)
  deriving Repr

/-- One top-tested loop.

    Entry binds `pTys.length` carried slots from `init`. Each iteration runs
    `pre` — a whole `Code`, so the test may follow an inner loop — then tests
    `cc ca cb`: the exit side leaves with `exitR`, each becoming a fresh slot
    for the code after the loop, and the continue side runs `body` and loops
    back with `cont` as the next carries. -/
structure Loop where
  pTys       : List ClifTy
  init       : List R
  cc         : ICmpCond
  ca         : R
  cb         : R
  exitOnTrue : Bool
  cont       : List R
  exitR      : List R
  exitTys    : List ClifTy
  deriving Repr

/-- A bottom-tested loop: the test is made once before entry and again at the
    end of each trip, so the body block branches to itself and there is no
    header. One fewer block on the hot path, and an empty input never enters.

    The test and the exit values are given as *carry positions* rather than
    slots, because they are read in two different scopes — from `init` at the
    guard, and from `cont` at the end of the body — and a slot number could
    only ever name one of them. -/
structure DLoop where
  pTys    : List ClifTy
  init    : List R
  cc      : ICmpCond
  /-- The slot the back-edge test reads, in the body's scope — so it may be
      something the body computed rather than a carry. -/
  ca      : R
  /-- Which carry the *entry* guard tests, or `none` for a loop entered
      unconditionally. A loop whose trip count is known non-zero — zeroing a
      fixed-size region, say — needs no guard, and one whose test reads a value
      the body produces cannot have one: there is nothing to read yet. -/
  guardIdx : Option Nat
  /-- The test's right operand, which the loop does not change. -/
  cb      : R
  contOnTrue : Bool
  cont    : List R
  /-- Which carries leave, in the order the exit block binds them. -/
  exitIdx : List Nat
  exitTys : List ClifTy
  deriving Repr

/-- A two-way branch: `cc ca cb` is tested in the current block, each arm ends
    with its export list, and the join binds one fresh slot per export. -/
structure IteMeta where
  cc   : ICmpCond
  ca   : R
  cb   : R
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

/-- The callee table a body is checked and compiled against — exactly the
    `sigs` and `fns` the emitted function will declare. -/
structure FnEnv where
  sigs : List SigDecl
  fns  : List FnDecl
  deriving Inhabited

/-- The signature `fn` resolves to, or `none` when nothing declares it. -/
def FnEnv.sigOf (env : FnEnv) (fn : Nat) : Option SigDecl := do
  let d ← env.fns.find? (·.ref.id == fn)
  env.sigs.find? (·.ref.id == d.sig.id)

/-- The callee table a declaration-only `IRBuilder` action produces, so a body
    written here names its FFI through the same `declare*` helpers as every
    other generator and cannot drift from their signatures. -/
def envOf (decls : IRBuilder α) : α × FnEnv :=
  let (a, st) := decls.run {}
  (a, { sigs := st.sigs, fns := st.fns })

-- ---------------------------------------------------------------------------
-- Types
-- ---------------------------------------------------------------------------

/-- Slot types in definition order: `Γ[r]?` is slot `r`, `Γ.length` the next. -/
abbrev TyEnv := List ClifTy

-- What the checker needs to know about a CLIF type, declared where dot
-- notation finds it.

/-- A type's name, for diagnostics; see `Op.name` for why not `repr`. -/
def _root_.AlgorithmLib.IR.ClifTy.name : ClifTy → String
  | .i8 => "i8" | .i16 => "i16" | .i32 => "i32" | .i64 => "i64"
  | .f32 => "f32" | .f64 => "f64" | .f32x4 => "f32x4" | .i8x16 => "i8x16"

/-- A type list, as diagnostics print it. -/
def tyListName (ts : List ClifTy) : String :=
  "[" ++ String.intercalate ", " (ts.map (·.name)) ++ "]"

def _root_.AlgorithmLib.IR.ClifTy.isInt : ClifTy → Bool
  | .i8 | .i16 | .i32 | .i64 => true
  | _ => false

def _root_.AlgorithmLib.IR.ClifTy.isFloat : ClifTy → Bool
  | .f32 | .f64 => true
  | _ => false

def _root_.AlgorithmLib.IR.ClifTy.isVec : ClifTy → Bool
  | .f32x4 | .i8x16 => true
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
private def need (b : Bool) (t : ClifTy) : Option ClifTy := if b then some t else none

/-- The type `o` yields, or `none` when an operand is out of scope, the operand
    types disagree, or the operation does not apply to them. -/
def Op.check (Γ : TyEnv) : Op → Option ClifTy
  | .iconst ty _ => need ty.isInt ty
  | .fconst ty _ => need ty.isFloat ty
  | .iadd a b | .isub a b | .imul a b | .udiv a b => do
      let ta ← Γ[a]?; let tb ← Γ[b]?
      need (ta == tb && ta.isInt) ta
  | .band a b | .bandNot a b | .bor a b | .bxor a b => do
      let ta ← Γ[a]?; let tb ← Γ[b]?
      need (ta == tb && (ta.isInt || ta.isVec)) ta
  | .ineg a | .ctz a | .popcnt a => do
      let ta ← Γ[a]?; need ta.isInt ta
  -- Cranelift lets the shift amount be any integer type.
  | .ishl a b | .ushr a b => do
      let ta ← Γ[a]?; let tb ← Γ[b]?
      need (ta.isInt && tb.isInt) ta
  | .ireduce32 a => do
      let ta ← Γ[a]?; need (ta.isInt && ta.width > 32) .i32
  | .uextend64 a | .sextend64 a => do
      let ta ← Γ[a]?; need (ta.isInt && ta.width < 64) .i64
  -- On vectors the result is a per-lane mask at the lane's own width, the same
  -- shape `fcmp` produces and the one `bitselect` and `vhighBits` consume.
  | .icmp _ a b => do
      let ta ← Γ[a]?; let tb ← Γ[b]?
      if ta.isVec then need (ta == tb) ta
      else need (ta == tb && ta.isInt) .i8
  | .select c a b => do
      let tc ← Γ[c]?; let ta ← Γ[a]?; let tb ← Γ[b]?
      need (tc.isInt && ta == tb) ta
  -- The mask is the operands' own type: it comes from a comparison that was
  -- bitcast to that width, which is what makes the lane-wise select expressible.
  | .bitselect c a b => do
      let tc ← Γ[c]?; let ta ← Γ[a]?; let tb ← Γ[b]?
      need (tc == ta && ta == tb) ta
  | .fadd a b | .fsub a b | .fmul a b | .fmax a b | .fmin a b => do
      let ta ← Γ[a]?; let tb ← Γ[b]?
      need (ta == tb && (ta.isFloat || ta.isVec)) ta
  | .fneg a => do
      let ta ← Γ[a]?; need (ta.isFloat || ta.isVec) ta
  | .fpromote a => do
      let ta ← Γ[a]?; need (ta == .f32) .f64
  -- On vectors the result is a per-lane mask at the lane's own width, which is
  -- the type `evalOp` produces and the width `bitselect` needs. Cranelift calls
  -- that type `i32x4`; there is no such `ClifTy`, and `Inst.fcmp` carries no
  -- result annotation for one to disagree with, so the mask keeps the operand
  -- type and `bitcast` — which compares widths — accepts it either way.
  | .fcmp _ a b => do
      let ta ← Γ[a]?; let tb ← Γ[b]?
      if ta.isVec then need (ta == tb) ta
      else need (ta == tb && ta.isFloat) .i8
  | .fcvtFromSint ty a => do
      let ta ← Γ[a]?; need (ta.isInt && ty.isFloat) ty
  | .fcvtToUint ty a => do
      let ta ← Γ[a]?; need (ta.isFloat && ty.isInt) ty
  | .splat ty a => do
      let ta ← Γ[a]?; let (lane, _) ← ty.lanes
      need (ta == lane) ty
  | .extractlane a lane => do
      let ta ← Γ[a]?; let (lt, n) ← ta.lanes
      need (lane < n) lt
  | .vhighBits a => do
      let ta ← Γ[a]?; need ta.isVec .i32
  | .bitcast ty a => do
      let ta ← Γ[a]?; need (ta.width == ty.width) ty
  | .load op a => do
      let ta ← Γ[a]?; need (ta == .i64) op.ty

/-- Every argument in scope and typed as the signature declares. -/
private def argsOk (Γ : TyEnv) (sig : SigDecl) (args : List R) : Bool :=
  args.length == sig.params.length &&
    (List.zip args sig.params).all fun (r, t) => Γ[r]? == some t

/-- The type a statement appends to the environment, or `none` for statements
    that bind nothing. `ok` is the check itself. -/
def Stmt.check (env : FnEnv) (Γ : TyEnv) : Stmt → Bool × Option ClifTy
  | .op o => match o.check Γ with
      | some t => (true, some t)
      | none => (false, none)
  | .store ty v a =>
      (Γ[v]? == some ty && Γ[a]? == some .i64, none)
  | .storeUnaligned v a =>
      ((Γ[v]?).isSome && Γ[a]? == some .i64, none)
  -- Any integer: `istore8` narrows to the low byte whatever width it is given,
  -- which is what Cranelift does and what `runStmt` already accepted.
  | .istore8 v a =>
      (((Γ[v]?).map (·.isInt)).getD false && Γ[a]? == some .i64, none)
  | .call fn args => match env.sigOf fn with
      | some sig => match sig.result with
          | some t => (argsOk Γ sig args, some t)
          | none => (false, none)
      | none => (false, none)
  | .callVoid fn args => match env.sigOf fn with
      | some sig => (argsOk Γ sig args && sig.result.isNone, none)
      | none => (false, none)

/-- How many slots a statement defines. -/
def Stmt.binds : Stmt → Nat
  | .op _ | .call _ _ => 1
  | _ => 0

/-- Check a statement list, extending the environment as it goes. -/
def wfStmts (env : FnEnv) : TyEnv → List Stmt → Bool × TyEnv
  | Γ, [] => (true, Γ)
  | Γ, s :: ss =>
      let (ok, t) := s.check env Γ
      let Γ' := match t with | some t => Γ ++ [t] | none => Γ
      let r := wfStmts env Γ' ss
      (ok && r.1, r.2)

/-- The slots `rs` name, in order, if all are in scope. -/
private def tysOf (Γ : TyEnv) : List R → Option (List ClifTy)
  | [] => some []
  | r :: rs => do let t ← Γ[r]?; let ts ← tysOf Γ rs; pure (t :: ts)

private def tysAre (Γ : TyEnv) (rs : List R) (ts : List ClifTy) : Bool :=
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
      let Γcarry := Γ ++ l.pTys
      let rPre := wfGo env fuel lbl' Γcarry pre
      let Γhead := rPre.2
      let rBody := wfGo env fuel lbl' Γhead body
      let ok :=
        tysAre Γ l.init l.pTys &&
        rPre.1 && !termsGo fuel pre &&
        (Γhead[l.ca]?).isSome && (Γhead[l.ca]? == Γhead[l.cb]?) &&
        tysAre Γhead l.exitR l.exitTys &&
        rBody.1 &&
        (termsGo fuel body || tysAre rBody.2 l.cont l.pTys)
      -- The exit block binds its parameters after everything the head and body
      -- defined, which is the numbering `emitLoop` uses.
      let r := wfGo env fuel lbl (rBody.2 ++ l.exitTys) ps
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
        (Γ[m.ca]?).isSome && (Γ[m.ca]? == Γ[m.cb]?) &&
        rT.1 && rE.1 &&
        (tT || tysAre rT.2 thnR jTys) &&
        (tE || tysAre rE.2 elsR jTys)
      let r := wfGo env fuel lbl (rE.2 ++ jTys) ps
      (ok && r.1, r.2)
  | fuel + 1, lbl, Γ, .dloop l body :: ps =>
      let Γcarry := Γ ++ l.pTys
      let lbl' := (l.exitTys, some l.pTys) :: lbl
      let rBody := wfGo env fuel lbl' Γcarry body
      -- `caIdx` and `exitIdx` name carries, so they are checked against the
      -- carry types once and hold at both the guard and the back edge.
      let carryTy (i : Nat) : Option ClifTy := l.pTys[i]?
      let guardOk := match l.guardIdx with
        | none => true
        | some i => (carryTy i).isSome && (carryTy i == Γ[l.cb]?)
      let ok :=
        tysAre Γ l.init l.pTys && guardOk &&
        (rBody.2[l.ca]?).isSome && (rBody.2[l.ca]? == Γ[l.cb]?) &&
        l.exitIdx.length == l.exitTys.length &&
        (List.zip l.exitIdx l.exitTys).all (fun (i, t) => carryTy i == some t) &&
        rBody.1 &&
        -- A body where every path leaves or takes the edge itself never
        -- reaches the back-edge test, so it need not supply carries for one.
        (termsGo fuel body || tysAre rBody.2 l.cont l.pTys)
      let r := wfGo env fuel lbl (rBody.2 ++ l.exitTys) ps
      (ok && r.1, r.2)
  | _ + 1, lbl, Γ, .br depth args :: ps =>
      -- Nothing may follow: the block is closed and the slots stop here.
      (ps.isEmpty && (lbl[depth]?).isSome && tysAre Γ args (((lbl[depth]?).map (·.1)).getD []),
       Γ)
  | _ + 1, lbl, Γ, .cont depth args :: ps =>
      (ps.isEmpty && (lbl[depth]?).isSome &&
        (carryOf lbl depth).isSome && tysAre Γ args ((carryOf lbl depth).getD []),
       Γ)

/-- Every reference names a slot that exists, with the type the use demands,
    and every annotation matches what its operands compute.

    Deliberately weaker than dominance: a reference from inside a loop to a slot
    the loop defined, used after the loop, satisfies `wf` and is caught by
    Cranelift's verifier instead. -/
def wf (env : FnEnv) (params : List ClifTy) (c : Code) : Bool :=
  (wfGo env fuel [] params c).1

-- ---------------------------------------------------------------------------
-- Observations
-- ---------------------------------------------------------------------------

private def callsIn (ss : List Stmt) : List Nat :=
  ss.filterMap fun
    | .call f _ => some f
    | .callVoid f _ => some f
    | _ => none

def callsGo : Nat → List Piece → List Nat
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
def callsOf (c : Code) : List Nat := callsGo fuel c

-- ---------------------------------------------------------------------------
-- Compilation
-- ---------------------------------------------------------------------------

/-- Blocks are numbered as reserved, values as created, and `env` maps slots to
    the `Val` carrying them in the region being emitted — an assoc list, newest
    binding wins, which is how a loop body's parameters shadow the head's. -/
structure CS where
  nextVal : Nat
  nextBlk : Nat
  slots   : Nat
  env     : List (Nat × Val)
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
  match s.env.lookup r with
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
                   env := (firstSlot + i, v) :: st'.env } (i + 1) ts

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
        | .fcvtFromSint ty a => .fcvtFromSint v ty (s.get a)
        | .fcvtToUint ty a   => .fcvtToUint v ty (s.get a)
        | .splat ty a   => .splat v ty (s.get a)
        | .extractlane a l => .extractlane v (s.get a) l
        | .vhighBits a  => .vhighBits v (s.get a)
        | .bitcast ty a => .bitcast v ty (s.get a)
        | .load op a    => .load v op (s.get a)
      { s with cur := inst :: s.cur, env := (s.slots, v) :: s.env, slots := s.slots + 1 }
  | .store ty v a => { s with cur := .storeTyped ty (s.get v) (s.get a) :: s.cur }
  | .storeUnaligned v a => { s with cur := .store (s.get v) (s.get a) :: s.cur }
  | .istore8 v a => { s with cur := .istore8 (s.get v) (s.get a) :: s.cur }
  | .call fn args =>
      let (v, s) := s.fresh
      { s with cur := .call (some v) ⟨fn⟩ (args.map s.get) :: s.cur,
               env := (s.slots, v) :: s.env, slots := s.slots + 1 }
  | .callVoid fn args =>
      { s with cur := .call none ⟨fn⟩ (args.map s.get) :: s.cur }

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
  let fr := s.fresh
  let flag := fr.1
  let sHead := { fr.2 with
                 cur := .icmp flag l.cc (fr.2.get l.ca) (fr.2.get l.cb) :: fr.2.cur }
  let carryVals := (List.range l.pTys.length).map (fun i => sHead.get (firstCarry + i))
  let exitArgs := l.exitR.map sHead.get
  let te :=
    if l.exitOnTrue then (exitId, exitArgs, bodyId, carryVals)
    else (bodyId, carryVals, exitId, exitArgs)
  let s := sHead.close (.brif flag ⟨te.1⟩ te.2.1 ⟨te.2.2.1⟩ te.2.2.2)
  -- body: the carries re-bound as its own parameters, shadowing the head's.
  -- Slots the prefix defined are re-run per iteration, so the body starts its
  -- numbering after them and the back edge carries only `cont`.
  let bodySlots := s.slots
  let s := s.open' bodyId l.pTys firstCarry
  let s := { s with slots := bodySlots }
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
  -- The guard: tested on the initial carries, in the block already open.
  let initVals := l.init.map s.get
  let s :=
    match l.guardIdx with
    | some gi =>
      let ca0 := (l.init[gi]?).getD 0
      let fr := s.fresh
      let flag := fr.1
      let sG := { fr.2 with cur := .icmp flag l.cc (fr.2.get ca0) (fr.2.get l.cb) :: fr.2.cur }
      let exit0 := l.exitIdx.map (fun i => (initVals[i]?).getD (⟨1000000⟩ : Val))
      let te0 :=
        if l.contOnTrue then (bodyId, initVals, exitId, exit0)
        else (exitId, exit0, bodyId, initVals)
      sG.close (.brif flag ⟨te0.1⟩ te0.2.1 ⟨te0.2.2.1⟩ te0.2.2.2)
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
      let fr2 := s.fresh
      let flag2 := fr2.1
      let sB := { fr2.2 with
                  cur := .icmp flag2 l.cc (fr2.2.get l.ca) (fr2.2.get l.cb) :: fr2.2.cur }
      let exitN := l.exitIdx.map (fun i => (contVals[i]?).getD ⟨1000000⟩)
      let teN :=
        if l.contOnTrue then (bodyId, contVals, exitId, exitN)
        else (exitId, exitN, bodyId, contVals)
      sB.close (.brif flag2 ⟨teN.1⟩ teN.2.1 ⟨teN.2.2.1⟩ teN.2.2.2)
  let exitFirst := s.slots
  let s := s.open' exitId l.exitTys exitFirst
  { s with slots := exitFirst + l.exitTys.length, labels := outerLabels }

def emitIte (fuel : Nat) (s : CS) (m : IteMeta) (thn els : List Piece)
    (thnR elsR : List R) : CS :=
  let thnId := s.nextBlk
  let elsId := thnId + 1
  let joinId := thnId + 2
  let s := { s with nextBlk := thnId + 3 }
  let fr := s.fresh
  let flag := fr.1
  let s := { fr.2 with cur := .icmp flag m.cc (fr.2.get m.ca) (fr.2.get m.cb) :: fr.2.cur }
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

/-- Compile a body to the function the artifact ships.

    `params` types the entry block, whose parameters are slots `0..`; every
    generator here takes the shared-memory base pointer alone. -/
def compileFn (idx : Nat) (env : FnEnv) (params : List ClifTy) (c : Code) : FuncData :=
  Id.run do
    let s0 : CS := { nextVal := 0, nextBlk := 1, slots := 0, env := [],
                     curRef := 0, curPars := [], cur := [], done := [] }
    let s := { s0.open' 0 params 0 with slots := params.length }
    let s := emitCode fuel s c
    let s := s.close .ret
    return {
      index := idx
      sigs := env.sigs
      fns := env.fns
      blocks := s.done.mergeSort (fun a b => a.ref.id ≤ b.ref.id)
    }

/-- The base pointer every generator's entry block takes. -/
def ptrParams : List ClifTy := [.i64]

/-- A compiled function, viewed as the builder state `Clif.lean`'s extractors
    read.

    Those extractors — `launchesOf`, `loopsOf`, `bindsOf`, `callsOf` — use only
    `sigs`, `fns` and `blocks`, and a `FuncData` has all three with the same
    types. So a claim about an emitted program keeps its exact statement when the
    generator behind it moves from `IRBuilder` to a term: what changes is how the
    program is *produced*, not what is being read. -/
def _root_.AlgorithmLib.IR.FuncData.asState (f : FuncData) : IRState :=
  { sigs := f.sigs, fns := f.fns, blocks := f.blocks }

-- ---------------------------------------------------------------------------
-- `Sur` — a binder surface evaluated at elaboration time
--
-- An ordinary `StateM` builder over the same term: slots handed out in order,
-- statements accumulated, so `do`-notation and plain Lean lambdas supply every
-- binder including the loop carries. Nothing about it survives to runtime —
-- `clif%` runs it while the file elaborates and splices the first-order `Code`
-- literal it produced.
-- ---------------------------------------------------------------------------

namespace Sur

structure St where
  env    : FnEnv
  tys    : TyEnv          -- slot types, in definition order
  pieces : List Piece     -- reversed
  cur    : List Stmt      -- reversed: the open straight-line run
  /-- The exit types of each enclosing loop, innermost first. A loop's own entry
      is `none` while its condition prefix runs, because the exit types are what
      that prefix returns — so `brk` from a prefix is the one thing the surface
      cannot type, and says so. -/
  labels : List (Option (List ClifTy) × Option (List ClifTy)) := []
  /-- The first thing that went wrong, kept so `clif%` can refuse the term.
      Only the first: everything after a failure is built against a scope the
      program never meant to have, so later complaints are noise. -/
  err    : Option String := none
  deriving Inhabited

abbrev M := StateM St

/-- Record a failure, keeping the first. The state is otherwise left alone: a
    surface that gave up must still hand back slot numbers its caller can use,
    or the numbering shifts and one mistake reads as a hundred. -/
private def note (s : St) (msg : String) : St :=
  if s.err.isSome then s else { s with err := some msg }

/-- The type the surface computed for a slot, and the types of a list of them.
    An unknown slot yields `i64` and a recorded failure. -/
private def tysAt (s : St) (rs : List R) : List ClifTy × St :=
  rs.foldl (fun (ts, s) r =>
    match s.tys[r]? with
    | some t => (ts ++ [t], s)
    | none => (ts ++ [default], note s s!"slot {r} is not in scope")) ([], s)

private def tyOf (s : St) (r : R) : ClifTy := (s.tys[r]?).getD default

private def bindOp (o : Op) : M R := fun s =>
  match o.check s.tys with
  | some t => (s.tys.length, { s with tys := s.tys ++ [t], cur := .op o :: s.cur })
  | none =>
      -- The slot is still handed out, typed `i64`, so the caller's numbering
      -- survives and the message names the one operation that failed.
      (s.tys.length,
       note { s with tys := s.tys ++ [default], cur := .op o :: s.cur }
         s!"operation is not well-typed: {o.name}")

private def bind0 (st : Stmt) : M Unit := fun s => ((), { s with cur := st :: s.cur })

def iconst (ty : ClifTy) (k : Int) : M R := bindOp (.iconst ty k)
def iconst64 (k : Int) : M R := iconst .i64 k
def iconst32 (k : Int) : M R := iconst .i32 k
def iadd (a b : R) : M R := bindOp (.iadd a b)
def isub (a b : R) : M R := bindOp (.isub a b)
def imul (a b : R) : M R := bindOp (.imul a b)
def udiv (a b : R) : M R := bindOp (.udiv a b)
def ineg (a : R) : M R := bindOp (.ineg a)
def ishl (a b : R) : M R := bindOp (.ishl a b)
def ushr (a b : R) : M R := bindOp (.ushr a b)
def band (a b : R) : M R := bindOp (.band a b)
def bandNot (a b : R) : M R := bindOp (.bandNot a b)
def bor (a b : R) : M R := bindOp (.bor a b)
def bxor (a b : R) : M R := bindOp (.bxor a b)
def ireduce32 (a : R) : M R := bindOp (.ireduce32 a)
def uextend64 (a : R) : M R := bindOp (.uextend64 a)
def sextend64 (a : R) : M R := bindOp (.sextend64 a)
def icmp (c : ICmpCond) (a b : R) : M R := bindOp (.icmp c a b)
def select (c a b : R) : M R := bindOp (.select c a b)
def bitselect (c a b : R) : M R := bindOp (.bitselect c a b)
def ctz (a : R) : M R := bindOp (.ctz a)
def popcnt (a : R) : M R := bindOp (.popcnt a)
def fconst (ty : ClifTy) (bits : UInt64) : M R := bindOp (.fconst ty bits)
def fconst32 (x : Float) : M R := fconst .f32 (x.toFloat32.toBits.toUInt64)
def fconst64 (x : Float) : M R := fconst .f64 x.toBits
def fadd (a b : R) : M R := bindOp (.fadd a b)
def fsub (a b : R) : M R := bindOp (.fsub a b)
def fmul (a b : R) : M R := bindOp (.fmul a b)
def fmax (a b : R) : M R := bindOp (.fmax a b)
def fmin (a b : R) : M R := bindOp (.fmin a b)
def fneg (a : R) : M R := bindOp (.fneg a)
def fpromote (a : R) : M R := bindOp (.fpromote a)
def fcmp (c : FloatCC) (a b : R) : M R := bindOp (.fcmp c a b)
def fcvtFromSint (ty : ClifTy) (a : R) : M R := bindOp (.fcvtFromSint ty a)
def fcvtToUint (ty : ClifTy) (a : R) : M R := bindOp (.fcvtToUint ty a)
def splat (ty : ClifTy) (a : R) : M R := bindOp (.splat ty a)
def extractlane (a : R) (lane : Nat) : M R := bindOp (.extractlane a lane)
def vhighBits (a : R) : M R := bindOp (.vhighBits a)
def bitcast (ty : ClifTy) (a : R) : M R := bindOp (.bitcast ty a)

def load (op : LoadOp) (a : R) : M R := bindOp (.load op a)
def load64 (a : R) : M R := load { ty := .i64 } a
def load32 (a : R) : M R := load { ty := .i32 } a
def load_i8 (a : R) : M R := load { ty := .i8 } a
def load_i16 (a : R) : M R := load { ty := .i16 } a
def uload8_64 (a : R) : M R := load { kind := .uload8, ty := .i64 } a
def sload8_64 (a : R) : M R := load { kind := .sload8, ty := .i64 } a
def uload32_64 (a : R) : M R := load { kind := .uload32, ty := .i64 } a
def loadF32 (a : R) : M R := load { ty := .f32, notrapAligned := true } a
def loadF64 (a : R) : M R := load { ty := .f64, notrapAligned := true } a
def loadF32x4 (a : R) : M R := load { ty := .f32x4, notrapAligned := true } a
def loadI8x16 (a : R) : M R := load { ty := .i8x16, notrapAligned := true } a

/-- Stores under `notrap aligned`, typed from the value's own slot. -/
def store (v a : R) : M Unit := fun s =>
  let (ts, s) := tysAt s [v]
  bind0 (.store (ts.headD default) v a) s
/-- Stores exactly `ty` bytes, whatever width the value's slot carries. Use
    these where the width is the point — `store` reads it off the value, so a
    value that widened upstream would silently write past the field. -/
def storeW (ty : ClifTy) (v a : R) : M Unit := bind0 (.store ty v a)
def storeI32 (v a : R) : M Unit := storeW .i32 v a
def storeI64 (v a : R) : M Unit := storeW .i64 v a

def storeUnaligned (v a : R) : M Unit := bind0 (.storeUnaligned v a)
def istore8 (v a : R) : M Unit := bind0 (.istore8 v a)

/-- A call binding its result; the callee's signature says whether there is
    one, so `callVoid` is the form for signatures without. -/
def call (fn : Nat) (args : List R) : M R := fun s =>
  match (s.env.sigOf fn).bind (·.result) with
  | some t => (s.tys.length, { s with tys := s.tys ++ [t], cur := .call fn args :: s.cur })
  | none =>
      (s.tys.length,
       note { s with tys := s.tys ++ [default], cur := .call fn args :: s.cur }
         s!"fn{fn} is undeclared or has no result")

def callVoid (fn : Nat) (args : List R) : M Unit := bind0 (.callVoid fn args)

/-- `min(a, b)` with the hardware's NaN behaviour rather than IEEE
    `minimumNumber`. Cranelift lowers exactly this shape — the wasm `pmin`
    pattern — to a single `minps`, where `fmin` costs a NaN-correct sequence. -/
def pmin (ty : ClifTy) (a b : R) : M R := do
  bitselect (← bitcast ty (← fcmp .lt a b)) a b

/-- `max(a, b)`, likewise a single `maxps`. Note the reversed compare: the rule
    Cranelift matches is `bitselect(fcmp lt b a, a, b)`. -/
def pmax (ty : ClifTy) (a b : R) : M R := do
  bitselect (← bitcast ty (← fcmp .lt b a)) a b

/-- The address of `base + off`, the shape every fixed-offset access takes. -/
def absAddr (base : R) (off : Int) : M R := do iadd base (← iconst64 off)

/-- An `i64` immediate and the operation that consumes it, which is how every
    constant operand reaches an instruction. -/
def iaddImm (a : R) (imm : Int) : M R := do iadd a (← iconst64 imm)

/-- Store at `base + offset`, under default memory flags. -/
def storeAt (base : R) (offset : Nat) (val : R) : M Unit := do
  storeUnaligned val (← absAddr base offset)
def ishlImm (a : R) (imm : Int) : M R := do ishl a (← iconst64 imm)
def ushrImm (a : R) (imm : Int) : M R := do ushr a (← iconst64 imm)

/-- The entry block's base pointer. -/
def basePtr : R := 0

/-- A loop condition, with the polarity of the branch that leaves. -/
structure Cond where
  cc : ICmpCond
  a : R
  b : R
  exitOnTrue : Bool

/-- Leave the loop when `cc a b` holds. -/
def exitIf (cc : ICmpCond) (a b : R) : Cond := ⟨cc, a, b, true⟩

/-- Stay in the loop while `cc a b` holds. -/
def contIf (cc : ICmpCond) (a b : R) : Cond := ⟨cc, a, b, false⟩

def exitIfEq (a b : R) : Cond := exitIf .eq a b
def exitIfSGe (a b : R) : Cond := exitIf .sge a b
def contIfULt (a b : R) : Cond := contIf .ult a b
def contIfSLt (a b : R) : Cond := contIf .slt a b
def exitIfSGt (a b : R) : Cond := exitIf .sgt a b
def contIfULe (a b : R) : Cond := contIf .ule a b

private def flushAux : M Unit := fun s =>
  ((), if s.cur.isEmpty then s
       else { s with pieces := .straight s.cur.reverse :: s.pieces, cur := [] })

/-- Run `m` capturing whole pieces, so a captured region may itself loop. -/
private def regionC (m : M α) : M (α × List Piece) := fun s =>
  let (a, s') := (do let a ← m; flushAux; pure a) { s with cur := [], pieces := [] }
  ((a, s'.pieces.reverse), { s' with cur := s.cur, pieces := s.pieces })

/-- Leave the `depth`-th enclosing loop, innermost first, with `args` as its
    exit values. Their types must be the ones the loop's normal exit exports. -/
def brkTo (depth : Nat) (args : List R) : M Unit := fun s0 =>
  let (_, s) := flushAux s0
  let (argTys, s) := tysAt s args
  let s := { s with pieces := .br depth args :: s.pieces }
  match (s.labels[depth]?).map (·.1) with
  | none => ((), note s s!"brk {depth} is outside that many loops")
  | some none => ((), note s "brk from a loop's condition prefix")
  | some (some tys) =>
      if argTys == tys then ((), s)
      else ((), note s s!"brk carries {tyListName argTys}, but the loop exits with {tyListName tys}")

/-- Go round the `depth`-th enclosing top-tested loop again. -/
def contTo (depth : Nat) (args : List R) : M Unit := fun s0 =>
  let (_, s) := flushAux s0
  let (argTys, s) := tysAt s args
  let s := { s with pieces := .cont depth args :: s.pieces }
  match (s.labels[depth]?).map (·.2) with
  | none => ((), note s s!"continue {depth} is outside that many loops")
  | some none => ((), note s "continue names a loop with no header to return to")
  | some (some tys) =>
      if argTys == tys then ((), s)
      else ((), note s s!"continue carries {tyListName argTys}, but the loop carries {tyListName tys}")

/-- Go round the innermost enclosing loop again. -/
def continueWith (args : List R) : M Unit := contTo 0 args

/-- Leave the innermost enclosing loop. -/
def brk (args : List R) : M Unit := brkTo 0 args

/-- A top-tested loop over `n` carried slots.

    `head` receives the carries and yields the condition, the slots to export
    on exit, and anything `body` needs; `body` yields the next carries. The
    result is the exit slots. Carry and exit types come from the surface's own
    type environment, so the term's annotations are never written by hand. -/
def wloop (inits : List R) (head : List R → M (Cond × List R × α))
    (body : List R → α → M (List R)) : M (List R) := fun s0 => Id.run do
  let (_, s) := flushAux s0
  let (pTys, s) := tysAt s inits
  let firstCarry := s.tys.length
  let carries := (List.range inits.length).map (firstCarry + ·)
  let outerLabels := s.labels
  let s := { s with tys := s.tys ++ pTys, labels := (none, some pTys) :: outerLabels }
  let (((c, exitR, x), preCode), s) := regionC (head carries) s
  let (exitTys, s) := tysAt s exitR
  let s := { s with labels := (some exitTys, some pTys) :: outerLabels }
  let ((cont, bodyCode), s) := regionC (body carries x) s
  let s := { s with labels := outerLabels }
  -- The code after the loop resumes at the exit block, whose parameters are
  -- numbered after every slot the head and body defined.
  let exits := (List.range exitR.length).map (s.tys.length + ·)
  let sAfter := { s with tys := s.tys ++ exitTys }
  let l : Loop := { pTys, init := inits, cc := c.cc, ca := c.a, cb := c.b,
                    exitOnTrue := c.exitOnTrue, cont, exitR, exitTys }
  return (exits, { sAfter with pieces := .loop l preCode bodyCode :: s.pieces })

/-- A bottom-tested loop over `n` carried slots.

    `guardIdx` defaults to *no* entry guard, because that is the failure that
    shows: a loop that should have been skipped runs once and the answer is
    wrong in a way a test sees. Defaulting to a guard on carry 0 fails the other
    way — a copy loop whose counter starts at zero and whose bound is zero
    exits before doing anything, and every shape check still passes.

    The test reads carry `caIdx` against `cb`, which the loop does not change,
    and `exitIdx` says which carries the code after the loop receives. `body`
    yields the next carries; the test is made on those, so the body always runs
    at least the work of one trip before the loop can be left — and never runs
    at all if the guard fails on `inits`. -/
def dwloop (inits : List R) (cc : ICmpCond) (cb : R)
    (contOnTrue : Bool) (exitIdx : List Nat)
    (body : List R → M (R × List R)) (guardIdx : Option Nat := none) :
    M (List R) := fun s0 => Id.run do
  let (_, s) := flushAux s0
  let (pTys, s) := tysAt s inits
  let exitTys := exitIdx.map (fun i => (pTys[i]?).getD default)
  let firstCarry := s.tys.length
  let carries := (List.range inits.length).map (firstCarry + ·)
  let outerLabels := s.labels
  let s := { s with tys := s.tys ++ pTys,
                    labels := (some exitTys, some pTys) :: outerLabels }
  let (((ca, cont), bodyCode), s) := regionC (body carries) s
  let s := { s with labels := outerLabels }
  let exits := (List.range exitIdx.length).map (s.tys.length + ·)
  let sAfter := { s with tys := s.tys ++ exitTys }
  let l : DLoop := { pTys, init := inits, cc, ca, guardIdx, cb, contOnTrue,
                     cont, exitIdx, exitTys }
  return (exits, { sAfter with pieces := .dloop l bodyCode :: s.pieces })

/-- One-carry loop, with the carry as a plain binder. -/
def wloop1 (init : R) (head : R → M (Cond × List R × α))
    (body : R → α → M (List R)) : M (List R) :=
  wloop [init] (fun cs => head (cs.headD 0)) (fun cs x => body (cs.headD 0) x)

/-- Two-carry loop. -/
def wloop2 (a b : R) (head : R → R → M (Cond × List R × α))
    (body : R → R → α → M (List R)) : M (List R) :=
  wloop [a, b] (fun cs => head (cs.headD 0) (cs.getD 1 0))
        (fun cs x => body (cs.headD 0) (cs.getD 1 0) x)

/-- A counted loop over `[0, limit)`, carrying nothing else — the shape most
    generators want, and the one `IR.forLoop` has.

    Writing it here rather than at each call site also fixes the emission order
    once: the limit is a slot the caller already made, the counter's `iconst 0`
    is emitted just before entry, and the increment comes after everything the
    body did. -/
def forLoop (limit : R) (body : R → M Unit) : M Unit := do
  let _ ← wloop1 (← iconst64 0)
    (head := fun i => return (contIfULt i limit, ([] : List R), ()))
    (body := fun i _ => do body i; return [← iaddImm i 1])

/-- A counted loop from `start`, as `IR.forLoopFromTo` does. -/
def forLoopFromTo (start limit : R) (body : R → M Unit) : M Unit := do
  let _ ← wloop1 start
    (head := fun i => return (contIfULt i limit, ([] : List R), ()))
    (body := fun i _ => do body i; return [← iaddImm i 1])

/-- A counted loop threading one accumulator, as `IR.forLoopAcc` does. The
    accumulator's initial value is a slot the caller made, so it is emitted
    before the counter's. -/
def forLoopAcc (limit acc0 : R) (body : R → R → M R) : M R := do
  let e ← wloop2 (← iconst64 0) acc0
    (head := fun i a => return (contIfULt i limit, [a], ()))
    (body := fun i a _ => do
      let nextAcc ← body i a
      return [← iaddImm i 1, nextAcc])
  return e.headD 0

/-- A counted loop threading two accumulators, as `IR.forLoopAcc2` does. -/
def forLoopAcc2 (limit ia ib : R) (body : R → R → R → M (R × R)) : M (R × R) := do
  let e ← wloop [← iconst64 0, ia, ib]
    (head := fun c => return (contIfULt (c.headD 0) limit, c.drop 1, ()))
    (body := fun c _ => do
      let (nx, ny) ← body (c.headD 0) (c.getD 1 0) (c.getD 2 0)
      return [← iaddImm (c.headD 0) 1, nx, ny])
  return (e.headD 0, e.getD 1 0)

/-- A while loop over two carries, as `IR.whileLoop2` does: the condition is
    computed at the head and both carries leave. -/
def whileLoop2 (ia ib : R) (cond : R → R → M R) (body : R → R → M (R × R)) :
    M (R × R) := do
  let e ← wloop2 ia ib
    (head := fun x y => do
      let ok ← cond x y
      let z ← iconst .i8 0
      return (contIf .ne ok z, [x, y], ()))
    (body := fun x y _ => do
      let (nx, ny) ← body x y
      return [nx, ny])
  return (e.headD 0, e.getD 1 0)

/-- Two-way branch with join values: each arm yields its exports, and the
    result slots carry whichever arm ran. -/
def ifte (cc : ICmpCond) (a b : R) (thn els : M (List R)) : M (List R) :=
  fun s0 => Id.run do
    let (_, s) := flushAux s0
    let ((thnR, thnC), s) := regionC thn s
    let (thnTys, s) := tysAt s thnR
    let ((elsR, elsC), s) := regionC els s
    let (elsTys, s) := tysAt s elsR
    -- The join carries what the arms that reach it export; an arm that leaves
    -- exports nothing, and when neither reaches it there is no join.
    let jTys :=
      if !terminates thnC then thnTys
      else if !terminates elsC then elsTys
      else []
    let joins := (List.range jTys.length).map (s.tys.length + ·)
    let sAfter := { s with tys := s.tys ++ jTys }
    return (joins, { sAfter with pieces := .ite ⟨cc, a, b, jTys⟩ thnC elsC thnR elsR :: s.pieces })

/-- A branch taken for effect, joining no values. -/
def when (cc : ICmpCond) (a b : R) (thn : M Unit) : M Unit := do
  let _ ← ifte cc a b (do thn; pure []) (pure [])
  pure ()

/-- Run a builder to the term it denotes. -/
def build (env : FnEnv) (params : List ClifTy) (m : M Unit) : Code :=
  let (_, s) := (do m; flushAux)
    { env, tys := params, pieces := [], cur := [] }
  s.pieces.reverse

/-- Which top-level piece first makes the body ill-formed, found by checking
    growing prefixes. Only ever run on a body already known to be bad, so its
    quadratic shape costs nothing on the path everything takes. -/
private def firstBadPiece (env : FnEnv) (params : List ClifTy) (c : Code) : Nat :=
  (List.range c.length).find? (fun k => !(wfGo env fuel [] params (c.take (k + 1))).1)
    |>.getD c.length

/-- The term a builder denotes, or why it is not one.

    Both failures a body can have are reported here rather than reaching an
    artifact: the surface's own — an operation whose operands do not typecheck,
    a slot out of scope, a `brk` that does not match its loop — and `wf`, which
    catches what the surface hands out correctly but assembles wrongly. -/
def buildChecked (env : FnEnv) (params : List ClifTy) (m : M Unit) :
    Except String Code :=
  let (_, s) := (do m; flushAux)
    { env, tys := params, pieces := [], cur := [] }
  let c := s.pieces.reverse
  match s.err with
  | some e => .error e
  | none =>
      if wf env params c then .ok c
      else .error s!"the body is not well-formed, from piece {firstBadPiece env params c} on"

end Sur

-- ---------------------------------------------------------------------------
-- Reification
-- ---------------------------------------------------------------------------

open Lean in
instance : ToExpr ClifTy where
  toTypeExpr := mkConst ``ClifTy
  toExpr t := mkConst <| match t with
    | .i8 => ``ClifTy.i8 | .i16 => ``ClifTy.i16 | .i32 => ``ClifTy.i32
    | .i64 => ``ClifTy.i64 | .f32 => ``ClifTy.f32 | .f64 => ``ClifTy.f64
    | .f32x4 => ``ClifTy.f32x4 | .i8x16 => ``ClifTy.i8x16

open Lean in
instance : ToExpr ICmpCond where
  toTypeExpr := mkConst ``ICmpCond
  toExpr c := mkConst <| match c with
    | .eq => ``ICmpCond.eq | .ne => ``ICmpCond.ne | .uge => ``ICmpCond.uge
    | .ugt => ``ICmpCond.ugt | .ule => ``ICmpCond.ule | .ult => ``ICmpCond.ult
    | .slt => ``ICmpCond.slt | .sle => ``ICmpCond.sle | .sgt => ``ICmpCond.sgt
    | .sge => ``ICmpCond.sge

open Lean in
instance : ToExpr FloatCC where
  toTypeExpr := mkConst ``FloatCC
  toExpr c := mkConst <| match c with
    | .eq => ``FloatCC.eq | .ne => ``FloatCC.ne | .lt => ``FloatCC.lt
    | .le => ``FloatCC.le | .gt => ``FloatCC.gt | .ge => ``FloatCC.ge

open Lean in
instance : ToExpr LoadKind where
  toTypeExpr := mkConst ``LoadKind
  toExpr k := mkConst <| match k with
    | .plain => ``LoadKind.plain | .uload8 => ``LoadKind.uload8
    | .uload32 => ``LoadKind.uload32 | .sload8 => ``LoadKind.sload8

open Lean in
instance : ToExpr LoadOp where
  toTypeExpr := mkConst ``LoadOp
  toExpr o := mkApp3 (mkConst ``LoadOp.mk) (toExpr o.kind) (toExpr o.ty)
                (toExpr o.notrapAligned)

open Lean in
instance : ToExpr Op where
  toTypeExpr := mkConst ``Op
  toExpr o := match o with
    | .iconst ty k => mkApp2 (mkConst ``Op.iconst) (toExpr ty) (toExpr k)
    | .iadd a b => mkApp2 (mkConst ``Op.iadd) (toExpr a) (toExpr b)
    | .isub a b => mkApp2 (mkConst ``Op.isub) (toExpr a) (toExpr b)
    | .imul a b => mkApp2 (mkConst ``Op.imul) (toExpr a) (toExpr b)
    | .udiv a b => mkApp2 (mkConst ``Op.udiv) (toExpr a) (toExpr b)
    | .ineg a => mkApp (mkConst ``Op.ineg) (toExpr a)
    | .ishl a b => mkApp2 (mkConst ``Op.ishl) (toExpr a) (toExpr b)
    | .ushr a b => mkApp2 (mkConst ``Op.ushr) (toExpr a) (toExpr b)
    | .band a b => mkApp2 (mkConst ``Op.band) (toExpr a) (toExpr b)
    | .bandNot a b => mkApp2 (mkConst ``Op.bandNot) (toExpr a) (toExpr b)
    | .bor a b => mkApp2 (mkConst ``Op.bor) (toExpr a) (toExpr b)
    | .bxor a b => mkApp2 (mkConst ``Op.bxor) (toExpr a) (toExpr b)
    | .ireduce32 a => mkApp (mkConst ``Op.ireduce32) (toExpr a)
    | .uextend64 a => mkApp (mkConst ``Op.uextend64) (toExpr a)
    | .sextend64 a => mkApp (mkConst ``Op.sextend64) (toExpr a)
    | .icmp c a b => mkApp3 (mkConst ``Op.icmp) (toExpr c) (toExpr a) (toExpr b)
    | .select c a b => mkApp3 (mkConst ``Op.select) (toExpr c) (toExpr a) (toExpr b)
    | .bitselect c a b => mkApp3 (mkConst ``Op.bitselect) (toExpr c) (toExpr a) (toExpr b)
    | .ctz a => mkApp (mkConst ``Op.ctz) (toExpr a)
    | .popcnt a => mkApp (mkConst ``Op.popcnt) (toExpr a)
    | .fconst ty b => mkApp2 (mkConst ``Op.fconst) (toExpr ty) (toExpr b)
    | .fadd a b => mkApp2 (mkConst ``Op.fadd) (toExpr a) (toExpr b)
    | .fsub a b => mkApp2 (mkConst ``Op.fsub) (toExpr a) (toExpr b)
    | .fmul a b => mkApp2 (mkConst ``Op.fmul) (toExpr a) (toExpr b)
    | .fmax a b => mkApp2 (mkConst ``Op.fmax) (toExpr a) (toExpr b)
    | .fmin a b => mkApp2 (mkConst ``Op.fmin) (toExpr a) (toExpr b)
    | .fneg a => mkApp (mkConst ``Op.fneg) (toExpr a)
    | .fpromote a => mkApp (mkConst ``Op.fpromote) (toExpr a)
    | .fcmp c a b => mkApp3 (mkConst ``Op.fcmp) (toExpr c) (toExpr a) (toExpr b)
    | .fcvtFromSint ty a => mkApp2 (mkConst ``Op.fcvtFromSint) (toExpr ty) (toExpr a)
    | .fcvtToUint ty a => mkApp2 (mkConst ``Op.fcvtToUint) (toExpr ty) (toExpr a)
    | .splat ty a => mkApp2 (mkConst ``Op.splat) (toExpr ty) (toExpr a)
    | .extractlane a l => mkApp2 (mkConst ``Op.extractlane) (toExpr a) (toExpr l)
    | .vhighBits a => mkApp (mkConst ``Op.vhighBits) (toExpr a)
    | .bitcast ty a => mkApp2 (mkConst ``Op.bitcast) (toExpr ty) (toExpr a)
    | .load op a => mkApp2 (mkConst ``Op.load) (toExpr op) (toExpr a)

open Lean in
instance : ToExpr Stmt where
  toTypeExpr := mkConst ``Stmt
  toExpr s := match s with
    | .op o => mkApp (mkConst ``Stmt.op) (toExpr o)
    | .store ty v a => mkApp3 (mkConst ``Stmt.store) (toExpr ty) (toExpr v) (toExpr a)
    | .storeUnaligned v a => mkApp2 (mkConst ``Stmt.storeUnaligned) (toExpr v) (toExpr a)
    | .istore8 v a => mkApp2 (mkConst ``Stmt.istore8) (toExpr v) (toExpr a)
    | .call f args => mkApp2 (mkConst ``Stmt.call) (toExpr f) (toExpr args)
    | .callVoid f args => mkApp2 (mkConst ``Stmt.callVoid) (toExpr f) (toExpr args)

open Lean in
instance : ToExpr Loop where
  toTypeExpr := mkConst ``Loop
  toExpr l :=
    mkAppN (mkConst ``Loop.mk)
      #[toExpr l.pTys, toExpr l.init, toExpr l.cc, toExpr l.ca, toExpr l.cb,
        toExpr l.exitOnTrue, toExpr l.cont, toExpr l.exitR, toExpr l.exitTys]

open Lean in
instance : ToExpr DLoop where
  toTypeExpr := mkConst ``DLoop
  toExpr l :=
    mkAppN (mkConst ``DLoop.mk)
      #[toExpr l.pTys, toExpr l.init, toExpr l.cc, toExpr l.ca, toExpr l.guardIdx,
        toExpr l.cb, toExpr l.contOnTrue, toExpr l.cont, toExpr l.exitIdx,
        toExpr l.exitTys]

open Lean in
instance : ToExpr IteMeta where
  toTypeExpr := mkConst ``IteMeta
  toExpr m :=
    mkAppN (mkConst ``IteMeta.mk)
      #[toExpr m.cc, toExpr m.ca, toExpr m.cb, toExpr m.jTys]

open Lean in
/-- `Piece` nests through `List Piece`, so the reifier recurses explicitly. -/
partial def pieceToExpr : Piece → Expr
  | .straight ss => mkApp (mkConst ``Piece.straight) (toExpr ss)
  | .loop l pre body =>
      mkApp3 (mkConst ``Piece.loop) (toExpr l) (codeToExpr pre) (codeToExpr body)
  | .ite m thn els thnR elsR =>
      mkAppN (mkConst ``Piece.ite)
        #[toExpr m, codeToExpr thn, codeToExpr els, toExpr thnR, toExpr elsR]
  | .dloop l body => mkApp2 (mkConst ``Piece.dloop) (toExpr l) (codeToExpr body)
  | .br depth args => mkApp2 (mkConst ``Piece.br) (toExpr depth) (toExpr args)
  | .cont depth args => mkApp2 (mkConst ``Piece.cont) (toExpr depth) (toExpr args)
where
  codeToExpr (body : List Piece) : Expr :=
    body.foldr (fun p acc =>
        mkApp2 (mkApp (mkConst ``List.cons [levelZero]) (mkConst ``Piece))
          (pieceToExpr p) acc)
      (mkApp (mkConst ``List.nil [levelZero]) (mkConst ``Piece))

open Lean in
instance : ToExpr Piece where
  toTypeExpr := mkConst ``Piece
  toExpr := pieceToExpr

-- ---------------------------------------------------------------------------
-- The `clif%` elaborator
-- ---------------------------------------------------------------------------

open Lean in
private def codeResultType : Expr :=
  mkApp2 (mkConst ``Except [levelZero, levelZero]) (mkConst ``String) (mkConst ``Code)

open Lean Meta Elab Term in
unsafe def evalCodeUnsafe (e : Expr) : TermElabM (Except String Code) :=
  evalExpr (Except String Code) codeResultType e

instance : Inhabited (Lean.Elab.TermElabM (Except String Code)) := ⟨pure (.ok [])⟩

@[implemented_by evalCodeUnsafe]
opaque evalCode (e : Lean.Expr) : Lean.Elab.TermElabM (Except String Code)

open Lean Elab Term in
/-- `clif% env params do …` — evaluate a `Sur.M Unit` builder while elaborating
    and splice the `Code` literal it builds. The binder surface is ordinary
    do-notation; what the artifact carries is first-order data.

    The term is checked here, so a body that does not typecheck or does not
    satisfy `wf` is an error on the block that wrote it. Nobody writes that
    check and there is nothing to forget: every use of this syntax has it, and
    none of it survives elaboration. -/
elab "clif% " env:term:max params:term:max b:term : term => do
  let e ← elabTermEnsuringType (← ``(Sur.buildChecked $env $params $b)) codeResultType
  synthesizeSyntheticMVarsNoPostponing
  match ← evalCode (← instantiateMVars e) with
  | .error msg => throwError "clif%: {msg}"
  | .ok code => return Lean.toExpr code

end AlgorithmLib.HProg
