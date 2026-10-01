module
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # The frontend composes — checked by the build

  `Prog` is meant to be the surface a *library* is written against, not just the
  one a body is written in. This module is the patterns someone reaches for when
  they write high-level code in a language — class hierarchies, instances that
  recurse on the shape of a type, recursion over their own syntax, containers
  with bounds in their types, obligations carried in records, notation — each
  written against `Prog` and each elaborated by the build.

  It is here because the claim is easy to assert and easy to have quietly stop
  being true: every one of these is an ordinary Lean construct whose values
  happen to be programs, and any of them could be broken by a change to `Prog`'s
  type that still leaves every generator in the tree compiling. Each definition
  *is* its own check. If the pattern stops being expressible, this file stops
  elaborating and the build fails here rather than in whoever's library meets it
  next.

  ## Where this list comes from

  Two sources, and the difference matters to whoever extends it.

  Most of it is a **census of what the tree's own generators use** — the Lean
  constructs that actually appear across `lean/algorithms` and
  `lib/AlgorithmLib`, weighted by how often. That is evidence rather than taste:
  `for`/`let mut` (317/169 uses), `Fin` (890), `Array` (773), `match` (677),
  `structure`/`instance`/`inductive` (165/156/87), `deriving` (78), `if h :`
  (56), `variable` (31), `partial def` (11), `mutual` and `termination_by` (5
  each).

  The rest is **anticipation**, and is marked as such below. The census is
  ground truth for one kind of user — someone writing a low-level generator —
  and silent about the other, someone writing a high-level library on top, since
  no such library exists here yet. Notation, dependent-pair results and the
  monad transformers are in that second group; the census finds 0, 0 and 6 uses
  of them respectively.

  When a real pattern turns out to be missing, add it here. That is the point:
  the list is incomplete in ways nobody can see from here, and the way it stops
  being incomplete is that a gap gets found once and then held by the build.

  Two things `Prog` does not support are documented where they would be reached
  for, with the cause and the workaround, so that someone who hits one
  recognises it instead of assuming a bug.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace ProgIdioms

-- ---------------------------------------------------------------------------
-- 1. Abstraction: classes, hierarchies, instances that follow the type
-- ---------------------------------------------------------------------------

/-- A class whose methods emit. -/
class Emit (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) (α : Type) where
  emit : α → Prog V L (V .i64)

/-- A class extending it, with a default method written in terms of the parent
    --- so the derived operation comes for free at every instance. -/
class EmitTwice (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) (α : Type)
    extends Emit V L α where
  twice : α → Prog V L (V .i64) := fun a => do
    let x ← Emit.emit (V := V) (L := L) a
    iadd x x

instance : Emit V L Nat where emit n := iconst64 (Int.ofNat n)
instance : EmitTwice V L Nat where

/-- Instances that recurse on the structure of a type: the resolution, not the
    author, assembles the emitter for a nested value. This is how a generic
    layout or serialisation library is written. -/
instance [Emit V L α] [Emit V L β] : Emit V L (α × β) where
  emit p := do iadd (← Emit.emit p.1) (← Emit.emit p.2)

instance [Emit V L α] : Emit V L (List α) where
  emit xs := do
    let vs ← xs.mapM (fun a => Emit.emit (V := V) (L := L) a)
    vs.foldlM (fun a v => iadd a v) (← iconst64 0)

/-- Nothing below was written for pairs-of-lists-of-naturals. -/
def nested : Prog V L (V .i64) := Emit.emit [(1, 2), (3, 4)]

/-- A class indexed by the CLIF type, with an instance per type. -/
class Zeroable (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type)
    (ty : ClifTy) where
  zero : Prog V L (V ty)

instance : Zeroable V L .i64 where zero := iconst64 0
instance : Zeroable V L .i32 where zero := iconst32 0

-- A library organised with `section` and `variable`, which is how the tree's
-- own modules are laid out. `V` and `L` are normally auto-bound per
-- declaration; declaring them once as section variables works the same way,
-- and a further `variable` of the library's own is captured by the
-- definitions that mention it.
section Lib
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def viaVariable (a b : V .i64) : Prog V L (V .i64) := iadd a b

variable (scale : Nat)

def scaled (a : V .i64) : Prog V L (V .i64) := ishlImm a (Int.ofNat scale)
end Lib

/-- **Anticipated, not observed.** The census finds no notation declarations in
    the tree; a high-level library is where infix syntax would earn its place.
    It is a plain abbreviation for a plain function, which is what makes it work
    --- see the note on `HAdd` below for what does not. -/
infixl:65 " +ᵥ " => iadd
infixl:70 " *ᵥ " => imul

def viaInfix : Prog V L (V .i64) := do
  let a ← iconst64 2
  let b ← iconst64 3
  let c ← a +ᵥ b
  c *ᵥ c

/-
  **Not available: arithmetic via `HAdd`/`HMul`.** The obvious wish is to write
  `a + b` on generated values by giving an instance

      instance : HAdd (V .i64) (V .i64) (Prog V L (V .i64)) := ⟨iadd⟩

  and it is rejected --- "instance does not provide concrete values for
  (semi-)out-params". `HAdd`'s result is an `outParam`, so it must be determined
  by the argument types, and the loop-label parameter `L` appears *only* in the
  result. Pinning `L` with an explicit binder does not help, for the same
  reason. The infix notations above give the same reading with no instance to
  resolve, and are what a library should offer.
-/

-- ---------------------------------------------------------------------------
-- 2. Recursion: over a user's own syntax, mutually, and well-founded
-- ---------------------------------------------------------------------------

inductive E where
  | lit (k : Int)
  | add (a b : E)
  | shl (a : E) (k : Nat)

def E.emit : E → Prog V L (V .i64)
  | .lit k => iconst64 k
  | .add a b => do iadd (← a.emit) (← b.emit)
  | .shl a k => do ishlImm (← a.emit) (Int.ofNat k)

-- Two syntaxes defined together, and two emitters defined together.
mutual
  inductive Ex where | lit (k : Int) | blk (s : Stm)
  inductive Stm where | one (e : Ex) | seq (a b : Stm)
end

mutual
  def Ex.go : Ex → Prog V L (V .i64)
    | .lit k => iconst64 k
    | .blk s => do Stm.go s; iconst64 0
  def Stm.go : Stm → Prog V L Unit
    | .one e => do let _ ← Ex.go e; pure ()
    | .seq a b => do Stm.go a; Stm.go b
end

/-- Recursion that is not structural: the measure decreases, and the proof is
    the author's. A divide-and-conquer emitter is written this way. -/
def halving (n : Nat) : Prog V L (V .i64) :=
  if h : n = 0 then iconst64 0
  else do iadd (← iconst64 (Int.ofNat n)) (← halving (n / 2))
termination_by n
decreasing_by exact Nat.div_lt_self (Nat.pos_of_ne_zero h) (by decide)

/-- `partial def`: no termination argument, and so nothing to reason about, but
    it elaborates and emits. Eleven definitions in the tree are written this
    way. -/
partial def spin (n : Nat) (acc : V .i64) : Prog V L (V .i64) :=
  if n == 0 then pure acc else do spin (n - 1) (← iaddImm acc 1)

/-- A combinator taking a program-valued function. -/
def applyN (n : Nat) (f : V .i64 → Prog V L (V .i64)) (x : V .i64) :
    Prog V L (V .i64) :=
  match n with
  | 0 => pure x
  | k + 1 => do applyN k f (← f x)

-- ---------------------------------------------------------------------------
-- 3. Data: containers with bounds, dependent results, generation-time tables
-- ---------------------------------------------------------------------------

/-- A user's own container, with its extent in its type and an accessor whose
    index cannot be out of range. -/
structure Vec (V : ClifTy → Type) (n : Nat) where
  base : V .i64

def Vec.get (v : Vec V n) (i : Fin n) : Prog V L (V .i64) := do
  load64 (← iaddImm v.base (Int.ofNat (8 * i.val)))

def Vec.sum (v : Vec V n) : Prog V L (V .i64) := do
  (List.finRange n).foldlM (fun a i => do iadd a (← v.get i)) (← iconst64 0)

/-- A result whose CLIF *type* the library chose: a dependent pair, so the
    caller learns which type it got. -/
def anyVal (wide : Bool) : Prog V L ((ty : ClifTy) × V ty) :=
  if wide then do pure ⟨.i64, ← iconst64 0⟩ else do pure ⟨.i32, ← iconst32 0⟩

/-- `Array`, and a generation-time symbol table keyed by name --- the shape of
    the environment a front end threads while lowering. -/
def symtab (names : Array String) : Prog V L (Std.HashMap String (V .i64)) := do
  names.foldlM (init := ∅) fun m nm => do
    return m.insert nm (← iconst64 (Int.ofNat nm.length))

/-- Programs as record fields, and a list of them sequenced. -/
structure Stages (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) where
  load  : Prog V L Unit
  infer : Prog V L Unit

def Stages.run (s : Stages V L) : Prog V L Unit := do s.load; s.infer

def runAll (ps : List (Prog V L Unit)) : Prog V L Unit := ps.forM id

/-- `for` and `let mut`: generation-time unrolling, not a loop in the artifact. -/
def unrolled (base : V .i64) (offs : List Nat) : Prog V L (V .i64) := do
  let mut acc ← iconst64 0
  for o in offs do
    let a ← absAddr base (Int.ofNat o)
    let v ← load64 a
    acc ← iadd acc v
  pure acc

/-- The stdlib's monadic combinators, over programs. -/
def sumLoaded (base : V .i64) (offs : List Nat) : Prog V L (V .i64) := do
  let vs ← offs.mapM (fun o => do load64 (← absAddr base (Int.ofNat o)))
  vs.foldlM (fun a v => iadd a v) (← iconst64 0)

/-- Generation-time list processing chosen by the caller's data, then emitted. -/
def zipped (base : V .i64) (offs : List Nat) (keep : List Bool) :
    Prog V L (V .i64) := do
  let sel := (offs.zip keep).filterMap (fun (o, k) => if k then some o else none)
  let vs ← sel.mapM (fun o => do load64 (← absAddr base (Int.ofNat o)))
  vs.foldlM (fun a v => iadd a v) (← iconst64 0)

/-- `Option` and `Except` around emission rather than inside it: a library that
    refuses a configuration returns the refusal, and the program it would have
    built is an ordinary value in the other branch. -/
def maybeLoad (o : Option Nat) (base : V .i64) : Prog V L (V .i64) :=
  match o with
  | none => iconst64 0
  | some k => do load64 (← absAddr base (Int.ofNat k))

def checkedCfg (n : Nat) : Except String (Prog V L Unit) :=
  if n % 16 == 0 then .ok (do let _ ← iconst64 (Int.ofNat n); pure ())
  else .error s!"{n} is not a multiple of 16"

/-- A callee the generator chose, with its argument list typed by that choice. -/
def dynCall (f : Ffi) (args : Vals V f.params) : Prog V L Unit := do
  let _ ← ffi f args
  pure ()

/-- Polymorphic in the CLIF type, with the side condition as an auto-param. -/
def addTwice {ty} (a b : V ty) (h : ty.isInt = true := by decide) :
    Prog V L (V ty) := do
  let s ← iadd a b h
  iadd s s h

-- ---------------------------------------------------------------------------
-- 4. Control abstraction
-- ---------------------------------------------------------------------------

/-- A library fragment that leaves the *caller's* loop. The label carries what
    that loop exits with, so this cannot name a loop it is not inside. -/
def breakWhenZero (lbl : L [ClifTy.i64] [ClifTy.i64]) (v exitVal : V .i64) :
    Prog V L Unit := do
  let z ← iconst64 0
  Prog.when .eq v z do
    brk lbl %[exitVal]

def usesLabel : Prog V L Unit := do
  let ptr ← basePtr
  let n ← load64 (← absAddr ptr 0x18)
  let r ← wloop1L (← iconst64 0)
    (head := fun lbl i => do
      breakWhenZero lbl i i
      return (exitIfSGe i n, %[i], ()))
    (body := fun _ i _ => do return %[← iaddImm i 1])
  store r.head (← absAddr ptr 0x28)

/-- A loop whose *number* of carried values the caller computed. -/
def wideLoop (k : Nat) : Prog V L Unit := do
  let ptr ← basePtr
  let z ← iconst64 0
  let _ ← wloop (Vals.ofFn (n := k) (fun _ => z))
    (head := fun vs => do
      let lim ← iconst64 10
      return (exitIfSGe (vs.uniformToList.headD z) lim, %[], ()))
    (body := fun vs _ => do
      let stepped ← vs.uniformToList.mapM (fun v => do iaddImm v 1)
      return Vals.ofFn (n := k) (fun i => stepped[i.val]?.getD z))
  store (← iconst64 (Int.ofNat k)) (← absAddr ptr 0x28)

/-- Artifact, hand the caller a handle, teardown — at whatever result type the
    caller's fragment has. -/
def withBase {α : Type} (k : V .i64 → Prog V L α) : Prog V L α := do
  let ptr ← basePtr
  let r ← k (← load64 (← absAddr ptr 0x18))
  storeAt ptr 0x30 (← iconst64 1)
  pure r

-- ---------------------------------------------------------------------------
-- 5. Obligations: derived, branched on, and carried in a value
--
-- The `example`s are the check that the default proofs close at a *symbolic*
-- argument. An obligation that only ever meets numerals looks healthy until
-- someone tries to wrap it.
-- ---------------------------------------------------------------------------

def wid : Nat := 16

def layer0 (bytes : Nat) (_h : bytes % wid = 0 := by simp [wid]) :
    Prog V L Unit := do
  let _ ← iconst64 (Int.ofNat bytes)
  pure ()

def layer1 (words : Nat) (_h : words % 4 = 0 := by omega) : Prog V L Unit :=
  layer0 (words * 4) (by simp only [wid]; omega)

def layer2 (recs : Nat) : Prog V L Unit := layer1 (recs * 4) (by omega)

example (k : Nat) : Prog V L Unit := layer0 (k * wid)
example (k : Nat) : Prog V L Unit := layer1 (k * 4)
example (k : Nat) : Prog V L Unit := layer2 k

/-- `deriving`, and a derived decision procedure standing behind an obligation:
    the width is computed from the caller's own enumeration, and `decide` closes
    the condition through it. -/
inductive Kind where | lo | hi
  deriving DecidableEq, Repr, Inhabited

def widthOf : Kind → Nat | .lo => 8 | .hi => 16

def forKind (k : Kind) (bytes : Nat)
    (_h : bytes % widthOf k = 0 := by decide) : Prog V L Unit := do
  let _ ← iconst64 (Int.ofNat bytes)
  pure ()

/-- A dependent `if`: the test produces the evidence, and the branch consumes
    it, so a caller with a value it cannot vouch for still has a way through. -/
def pick (n : Nat) : Prog V L Unit :=
  if h : n % wid = 0 then layer0 n h else pure ()

/-- The obligation carried in a record, decided once where the value is made
    rather than at each use. -/
structure Window where
  bytes   : Nat
  aligned : bytes % wid = 0 := by simp [wid]

def useWindow (w : Window) : Prog V L Unit := layer0 w.bytes w.aligned
def w4096 : Window := { bytes := 4096 }

-- ---------------------------------------------------------------------------
-- 6. Generation-time effects, as ordinary transformers
--
-- A library wanting to thread a layout, a config or a counter alongside the
-- emission does not need anything of its own. `Prog V L` is a `Monad`, so the
-- stdlib's transformers apply; the explicit `liftM` ascriptions are needed only
-- because `V` and `L` are implicit and leave the lift's instance ambiguous.
-- ---------------------------------------------------------------------------

structure Cfg where
  base   : Nat
  stride : Nat

abbrev R (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) :=
  ReaderT Cfg (Prog V L)

def field (i : Nat) : R V L (V .i64) := do
  let c ← read
  let ptr ← liftM (basePtr : Prog V L _)
  let a ← liftM (absAddr ptr (Int.ofNat (c.base + i * c.stride)) : Prog V L _)
  liftM (load64 a : Prog V L _)

abbrev S (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) :=
  StateT Nat (Prog V L)

def fresh : S V L (V .i64) := do
  let n ← get
  set (n + 1)
  liftM (iconst64 (Int.ofNat n) : Prog V L _)

abbrev Ex' (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) :=
  ExceptT String (Prog V L)

-- ---------------------------------------------------------------------------
-- 7. Emitter closures, and the one result shape that is rejected
-- ---------------------------------------------------------------------------

/-- A function returning an emitter closed over a generated handle. -/
def mkGet (base : V .i64) : V .i64 → Prog V L (V .i64) :=
  fun i => do load64 (← iadd base i)

/-- The same thing built inside a body: a *pure* `let`, after binding the
    handle. This is the workaround for the rejected shape below, and it is one
    line. -/
def usesClosure : Prog V L Unit := do
  let ptr ← basePtr
  let base ← load64 (← absAddr ptr 0x18)
  let get := mkGet (L := L) base
  store (← iadd (← get (← iconst64 0)) (← get (← iconst64 8)))
    (← absAddr ptr 0x28)

/-
  **Not available: an emitter returned through a bind.** `Prog V L : Type →
  Type 1`, because every constructor takes `{α : Type}` and `Type` is itself a
  `Type 1`. So a value carrying an emitter cannot come back through a `←`:

      def r : Prog V L (V .i64 → Prog V L Unit) := do
        let b ← iconst64 0
        pure (fun i => iadd b i)
      -- `V .i64 → Prog V L Unit` is `Type 1`, but `Prog`'s index is `Type`.

  The same applies to `Prog V L (Prog V L Unit)` and to any structure with an
  emitter field. First-order results are unaffected: handles, tuples,
  `List (V .i64)`, records of handles, the dependent pair in `anyVal`.

  It is a packaging restriction, not a limit on what can be written: return the
  handle and build the closure with a pure `let` (`usesClosure` above), or take
  the continuation instead (`withBase`). A universe-polymorphic `Prog` would
  admit it, at the cost of the `Monad` instance --- and so of `do`-notation ---
  at the crossing point.
-/

-- ---------------------------------------------------------------------------
-- 8. One body built out of all of it
-- ---------------------------------------------------------------------------

def composed : Prog V L Unit := do
  let ptr ← basePtr
  let out ← load64 (← absAddr ptr 0x28)
  let e : E := .add (.lit 3) (.shl (.lit 5) 2)
  let a ← applyN 3 (fun x => iaddImm x 1) (← e.emit)
  let b ← unrolled ptr [0x18, 0x20]
  let c ← sumLoaded ptr [0x18, 0x20]
  let d ← Zeroable.zero (V := V) (L := L) (ty := .i64)
  let f ← withBase (fun bp => do pure (← bp +ᵥ bp))
  let (g, _) ← StateT.run (m := Prog V L) ((List.range 2).mapM (fun _ => fresh)) 0
  let h ← (field 0).run ⟨0x18, 8⟩
  let i ← nested
  let j ← EmitTwice.twice (V := V) (L := L) (5 : Nat)
  let k ← halving 8
  let l ← (⟨ptr⟩ : Vec V 2).sum
  let m ← Ex.go (.blk (.seq (.one (.lit 1)) (.one (.lit 2))))
  let tbl ← symtab #["x", "y"]
  let n := (tbl.get? "x").getD m
  Stages.run { load := runAll [pure (), useWindow w4096], infer := layer2 7 }
  pick 4096
  forKind .hi 32
  let _ ← viaVariable ptr ptr
  let _ ← scaled 1 ptr
  let _ ← spin 2 ptr
  let _ ← zipped ptr [0x18, 0x20] [true, false]
  let _ ← maybeLoad (some 0x18) ptr
  match (checkedCfg 32 : Except String (Prog V L Unit)) with
  | .ok p => p
  | .error _ => pure ()
  let s ← [a, b, c, d, f, h, i, j, k, l, n].append g
            |>.foldlM (fun x y => iadd x y) (← iconst64 0)
  store (← addTwice s s) out

/-- The patterns above elaborate; this checks that the body they build still
    reaches an artifact rather than being rejected by `wf`. -/
def check : CoreM Unit := do
  match Prog.compileProg 1 (composed : Body) with
  | .error e => throwError s!"PROG IDIOMS FAILED: the composed body was refused: {e}"
  | .ok fd =>
      if fd.blocks.isEmpty then
        throwError "PROG IDIOMS FAILED: the composed body emitted no blocks"
      IO.println s!"[ProgIdioms] the frontend patterns compose; \
        the body they build emits {fd.blocks.length} block(s)"

end ProgIdioms

#eval ProgIdioms.check
