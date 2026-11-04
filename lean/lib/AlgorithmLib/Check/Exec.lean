module
public import AlgorithmLib.Check.Trace
public import AlgorithmLib.Check.VendorRule
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.Exec` — the checker over a function body

`exec` runs a body (`Code`, the term that ships) over abstract states: each
statement by its transfer function, a branch by both arms and a join, a loop by
iterating one trip until the state it carries round stops changing. It answers
what the body may end with — a state for falling off its end, and the states
each `br` and `cont` leave with — and a pattern over the calls any run of it
makes, or why it refuses.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState roomAt afterStore storeKeeps cellOk readyIn)

-- ---------------------------------------------------------------------------
-- Joins
-- ---------------------------------------------------------------------------

def V.beqv : V → V → Bool
  | .sc t x, .sc t' y => decide (t = t') && x == y
  | .vec t xs, .vec t' ys => decide (t = t') && decide (xs = ys)
  | _, _ => false

def ICmpCond.beqc (a b : ICmpCond) : Bool := a == b

/-- Equality of abstract values, decided. -/
def AV.beq : AV → AV → Bool
  | .known v, .known v' => V.beqv v v'
  | .int n, .int n' => decide (n = n')
  | .ptr r n, .ptr r' n' => decide (r = r') && decide (n = n')
  | .cmp c a b, .cmp c' a' b' => c == c' && a == a' && b == b'
  | .sc t, .sc t' => decide (t = t')
  | .hand k b, .hand k' b' => decide (k = k') && b == b'
  | .blk n b, .blk n' b' => decide (n = n') && b == b'
  | .into r lo hi, .into r' lo' hi' => decide (r = r') && decide (lo = lo') && decide (hi = hi')
  | .any, .any => true
  | _, _ => false

/-- What two typed values both are. -/
def AV.tyJoin (a b : AV) : AV :=
  match a.ty, b.ty with
  | some t, some t' => if t = t' then .sc t else .any
  | _, _ => .any

/-- The least description covering both. -/
def AV.join (a b : AV) : AV :=
  if a.beq b then a else
  match a.asPtr, b.asPtr with
  | .ptr r n, .ptr r' n' =>
      if r = r' then
        match n.join n' with
        | some j => if j.hi ≤ span then .ptr r j else .sc .i64
        | none => .sc .i64
      else .sc .i64
  | _, _ =>
      match a.num, b.num with
      | some x, some y => match x.join y with
        | some n => .int n
        | none => a.tyJoin b
      | _, _ => a.tyJoin b

/-- A fact of one typestate that holds where the other does too: itself where
    the other names it, a bound on a hash table's values the larger of the
    two bounds, a party knowing every access where the other says it does. -/
def meetFact (S' : TState) (x : Fact) : Option Fact :=
  if S'.contains x then some x else
  match x with
  | .htAt a n k => (htAtOf a S').map fun m => .htAt a (max n m.1) (max k m.2)
  | .readyAll p => if readyIn S' p then some (.readyAll p) else none
  | _ => none

/-- A party the second typestate says knows every access, which the first,
    not naming it, says too. -/
def meetReady (S : TState) (x : Fact) : Bool :=
  match x with
  | .readyAll p => !S.contains x && readyIn S p
  | _ => false

/-- Facts holding wherever either typestate does: the one typestate where
    the two are the same, as they are where a branch or a trip round a loop
    changes none. -/
def factsMeet (S S' : TState) : TState :=
  if S = S' then S else S.filterMap (meetFact S') ++ S'.filter (meetReady S)

/-- The bound of `τ`'s with the same multiple and terms as `p`. -/
def Sym.like (τ : Sym) (p : Nat × Lin) : Option (Nat × Lin) :=
  τ.ubs.find? fun q => decide (q.1 = p.1 ∧ q.2.terms = p.2.terms)

/-- What two slots' linear facts both say: a bound both have up to its
    constant, with the larger constant. -/
def Sym.join (σ τ : Sym) : Sym :=
  { dfn := if σ.dfn = τ.dfn then σ.dfn else none,
    ubs := σ.ubs.filterMap fun p => (τ.like p).map fun q => (p.1, ⟨p.2.terms, max p.2.k q.2.k⟩) }

theorem Sym.join_ubs {σ τ : Sym} {p : Nat × Lin} (hp : p ∈ (σ.join τ).ubs) :
    ∃ p0 ∈ σ.ubs, ∃ q ∈ τ.ubs, q.1 = p0.1 ∧ q.2.terms = p0.2.terms ∧
      p = (p0.1, ⟨p0.2.terms, max p0.2.k q.2.k⟩) := by
  simp only [Sym.join, List.mem_filterMap, Option.map_eq_some_iff] at hp
  obtain ⟨p0, hp0, q, hq, rfl⟩ := hp
  have hm := List.mem_of_find?_eq_some hq
  have hc := List.find?_some hq
  simp only [decide_eq_true_eq] at hc
  exact ⟨p0, hp0, q, hm, hc.1, hc.2, rfl⟩

theorem Sym.join_left {σ τ : Sym} {Γ : Env} {w : World} {x : Int} {m : Nat} (h : σ.FitsAt Γ w x m) :
    (σ.join τ).FitsAt Γ w x m :=
  Sym.fits_weaken h (fun D e => by
      simp only [Sym.join] at e; split at e
      · exact e
      · cases e)
    (fun p hp => by
      obtain ⟨p0, hp0, q, -, -, -, rfl⟩ := Sym.join_ubs hp
      exact ⟨p0, hp0, rfl, rfl, Int.le_max_left _ _⟩)

theorem Sym.join_right {σ τ : Sym} {Γ : Env} {w : World} {x : Int} {m : Nat} (h : τ.FitsAt Γ w x m) :
    (σ.join τ).FitsAt Γ w x m :=
  Sym.fits_weaken h (fun D e => by
      simp only [Sym.join] at e; split at e
      · rename_i heq; rw [← heq]; exact e
      · cases e)
    (fun p hp => by
      obtain ⟨p0, -, q, hq, e1, e2, rfl⟩ := Sym.join_ubs hp
      exact ⟨q, hq, e1, e2, Int.le_max_right _ _⟩)

/-- The join of two states over environments of one length. -/
def AS.join (A B : AS) : Except String AS :=
  if A.len = B.len then
    .ok ⟨Slots.zip AV.join A.slots B.slots, A.len, factsMeet A.facts B.facts,
      Slots.zip Sym.join A.syms B.syms, A.par.join B.par⟩
  else .error s!"join of environments of {A.len} and {B.len} slots"

def joinOpt : Option AS → Option AS → Except String (Option AS)
  | none, b => .ok b
  | a, none => .ok a
  | some a, some b => do return some (← a.join b)

-- ---------------------------------------------------------------------------
-- Environments
-- ---------------------------------------------------------------------------

/-- A description kept in an environment cut to `n` slots: one that reads a
    slot past the cut no longer can. -/
def AV.cut (n : Nat) : AV → AV
  | .cmp c a b => if a < n ∧ b < n then .cmp c a b else .sc .i8
  | .into b lo hi => if b < n then .into b lo hi else .sc .i64
  | v => v

/-- Linear facts kept in an environment cut to `n` slots: those that read no
    slot past the cut. -/
def Sym.cut (n : Nat) (σ : Sym) : Sym :=
  { dfn := σ.dfn.filter (·.slots.all (· < n)), ubs := σ.ubs.filter (·.2.slots.all (· < n)) }

/-- What a slot's facts say in an environment cut to `n` slots and grown
    again, where the slots before the cut keep their bits. -/
theorem Sym.cut_fits {σ : Sym} {Γ Γ' : Env} {w : World} {r m n m' : Nat} (h : σ.Fits Γ w r m)
    (hΓ : ∀ x, x < n → slotVal Γ' x = slotVal Γ x) (hr : slotVal Γ' r = slotVal Γ r) (hm : n ≤ m') :
    (σ.cut n).Fits Γ' w r m' := by
  have h1 : (σ.cut n).Fits Γ w r m :=
    Sym.fits_sub h (fun D e => (Option.filter_eq_some_iff.mp e).1) (fun p hp => (List.mem_filter.mp hp).1)
  have h2 : (σ.cut n).Fits Γ w r n := Sym.fits_bound h1 (fun x hx => by
    simp only [Sym.slots, Sym.cut, List.mem_append, List.mem_flatten, List.mem_map] at hx
    rcases hx with hx | ⟨_, ⟨p, hp, rfl⟩, hx⟩
    · cases e : σ.dfn.filter (·.slots.all (· < n)) with
      | none => rw [e] at hx; simp at hx
      | some D =>
          rw [e] at hx
          have := (Option.filter_eq_some_iff.mp e).2
          simpa using List.all_eq_true.mp this x hx
    · have := (List.mem_filter.mp hp).2
      simpa using List.all_eq_true.mp this x hx)
  exact Sym.fits_congr h2 hΓ hr (fun _ _ => rfl) hm

/-- `k` slots from `i` on known as nothing. -/
def Slots.clear {α : Type} [Blank α] (t : Slots α) (i : Nat) : Nat → Slots α
  | 0 => t
  | k + 1 => (t.set i Blank.blank).clear (i + 1) k

/-- `bindAt`, abstractly: grow or cut to `n` slots, then append `vs`, the
    `i`-th with linear facts `ss[i]` where there are any. Growing leaves what
    the state knows; cutting forgets, in what the slots kept say, the slots cut
    away. -/
def AS.bindAt (A : AS) (n : Nat) (vs : List AV) (ss : List Sym := []) : AS :=
  let base : AS :=
    if A.len ≤ n then ⟨A.slots.clear A.len (n - A.len), n, A.facts, A.syms.clear A.len (n - A.len), A.par⟩
    else ⟨A.slots.map (AV.cut n), n, A.facts, A.syms.map (Sym.cut n), A.par.cut n⟩
  base.pushAll (vs.mapIdx fun i v => (v.portable, ss.getD i {}))

-- ---------------------------------------------------------------------------
-- Branch conditions
-- ---------------------------------------------------------------------------

/-- A slot as a number: an address as its offset into the region it points
    into, anything else as itself. -/
def AV.offs (a : AV) : Option (Option Region × Num) :=
  match a.asPtr with
  | .ptr r n => some (some r, n)
  | b => b.num.map fun n => (none, n)

/-- A number read back as `offs` reads it: an offset into `r`, or itself. -/
def offAV : Option Region → Num → AV
  | some r, n => .ptr r n
  | none, n => .int n

/-- A signed comparison of two slots both known not to be negative, read as
    the unsigned one it agrees with. -/
def unsignedCond (A : AS) (c : ICmpCond) (x y : R) : ICmpCond :=
  match c, (A.get x).num, (A.get y).num with
  | .slt, some nx, some ny | .sle, some nx, some ny | .sgt, some nx, some ny | .sge, some nx, some ny =>
      if nx.hi < 2 ^ (nx.t.width - 1) ∧ ny.hi < 2 ^ (ny.t.width - 1) then
        match c with | .slt => ICmpCond.ult | .sle => ICmpCond.ule | .sgt => ICmpCond.ugt | _ => ICmpCond.uge
      else c
  | c, _, _ => c

/-- The relation `x c y` comes to when its flag is `b`, the signed ones read as
    unsigned where both operands are known not to be negative. -/
def holdsRel (A : AS) (c : ICmpCond) (x y : R) (b : Bool) : Option ICmpCond :=
  match unsignedCond A c x y, b with
  | .ult, true | .uge, false => some .ult
  | .ule, true | .ugt, false => some .ule
  | .uge, true | .ult, false => some .uge
  | .ugt, true | .ule, false => some .ugt
  | .eq, true | .ne, false => some .eq
  | .eq, false | .ne, true => some .ne
  | _, _ => none

/-- What `x + e ≤ y` teaches, `e` zero or one: `x` is below what `y` may be and
    inside the bound `y` has, `y` above what `x` may be, and the region `y` is
    bounded by holds at least what `x` may be. Where both are multiples of
    some `g`, `x < y` gives `x + g ≤ y`. -/
def lessRange (A : AS) (x y : R) (e : Nat) : AS :=
  match (A.get x).offs, (A.get y).offs with
  | some (gx, nx), some (gy, ny) =>
      if gx = gy then
        let g := Nat.gcd nx.al ny.al
        let step := if e = 1 then max 1 g else 0
        let fx : Option Bound := match ny.fit with
          | some (r, s, d) => some (r, s, d + step * s)
          | none => none
        let A1 := A.setAV x (offAV gx { nx with hi := min nx.hi (ny.hi - step), fit := fx.or nx.fit })
        let A2 := A1.setAV y (offAV gx { ny with lo := max ny.lo (nx.lo + step) })
        match ny.fit with
        | some (r, s, d) =>
            if 0 < nx.lo + step then
              if r != .arena || arenaFree A2.facts then A2.addFact (.room r ((nx.lo + step) * s + d)) else A2
            else A2
        | none => A2
      else A
  | _, _ => A

/-- How far below `y` `x + e ≤ y` puts `x`: where both are multiples of some
    `g`, `x < y` puts it `g` below. -/
def alignE (A : AS) (x y : R) (e : Nat) : Nat :=
  let g := Nat.gcd (A.get x).absAl (A.get y).absAl
  if e = 1 ∧ 0 < g then g else e

/-- What `x + e ≤ y` teaches: by ranges, then by linear facts. -/
def lessThan (A : AS) (x y : R) (e : Nat) : AS :=
  relLo (relLess (lessRange A x y e) x y (alignE A x y e)) x

/-- What `x ≠ 0` teaches: `x` is at least one, and an open's answer a handle
    held. -/
def nonZero (A : AS) (x y : R) : AS :=
  match A.get y with
  | .known (.sc _ 0) =>
      match A.get x with
      | .hand k false => A.setAV x (.hand k true)
      | .blk n false => A.setAV x (.blk n true)
      | _ => A.tightenLo x 1 1
  | _ => A

/-- What `x rel y` teaches. -/
def learn (A : AS) (rel : ICmpCond) (x y : R) : AS :=
  match rel with
  | .ult => lessThan A x y 1
  | .ule => lessThan A x y 0
  | .ugt => lessThan A y x 1
  | .uge => lessThan A y x 0
  | .eq => lessThan (lessThan A x y 0) y x 0
  | .ne => nonZero (nonZero A x y) y x
  | _ => A

/-- An argument as the parameters' record reads it: a number computed from
    them, or bits known. -/
def parArg (A : AS) (r : R) : Option PArg :=
  match A.par.slotOf r with
  | some (_, e) => some (.sym e)
  | none => (A.get r).bits.map .known

/-- Relation `rel` between `x` and `y` as one between numbers computed from
    the parameters, where at least one is. -/
def condOf (A : AS) (rel : ICmpCond) (x y : R) : Option PCond :=
  match parArg A x, parArg A y with
  | some a, some b => if a.isSym || b.isSym then some ⟨rel, a.expr, b.expr⟩ else none
  | _, _ => none

/-- What a branch learns from the flag it reads being `b`, of the slots and
    of the parameters: `none` where the flag cannot be `b`. -/
def refine (A : AS) (flag : R) (b : Bool) : Option AS :=
  match A.get flag with
  | .known (.sc _ f) => if (f != 0) == b then some A else none
  | .cmp c x y => match holdsRel A c x y b with
    | some rel => some ((learn A rel x y).addCond (condOf A rel x y))
    | none => some A
  | _ => some A

-- ---------------------------------------------------------------------------
-- Statements
-- ---------------------------------------------------------------------------

/-- The numeric addresses a store of `n` bytes may touch, where `memN` shows
    them mapped. -/
def storeSpan (S : TState) (addr : AV) (n : Nat) : Option (Nat × Nat) :=
  if memN S addr n then
    match addr with
    | .known (.sc _ x) => some (x.toNat, x.toNat + n)
    | .ptr r m => some ((regionBase r).toNat + m.lo, (regionBase r).toNat + m.hi + n)
    | _ => none
  else none

/-- Whether a store into `[lo, hi)` keeps a fact. -/
def rangeKeeps (lo hi : Nat) : Fact → Bool
  | .cell c _ => cellOk c && (hi ≤ c.toNat || c.toNat + 8 ≤ lo)
  | .cstr _ => false
  | .cstrIn c k => hi ≤ c.toNat || c.toNat + k ≤ lo
  | .text c _ k _ | .wgText c k _ => hi ≤ c.toNat || c.toNat + k ≤ lo
  | .htAt c _ _ => decide (hi < (regionBase .pinned).toNat) && cellOk c && (hi ≤ c.toNat || c.toNat + 8 ≤ lo)
  | _ => true

/-- How many bytes a store of slot `v` writes: `n` for a scalar, its lanes
    for a vector. -/
def storeBytes (A : AS) (v : R) (n : Nat) : Option Nat :=
  match (A.get v).scTy with
  | some _ => some n
  | none => match (A.get v).ty with
    | some T => T.lanes.map fun (l, k) => k * tyBytes l
    | none => none

/-- What a store leaves, `none` where it may fault. A scalar store to a known
    address keeps what `afterStore` keeps; one to an address in a range, the
    facts about bytes outside it; a vector store, every fact but the cells.
    Of the parameters' cells and buffers, each keeps those outside the bytes
    it may write. -/
def absStore (A : AS) (v a : R) (n : Nat) : Except String AS :=
  match storeBytes A v n with
  | none => .error s!"store of a value of unknown type (slot {v})"
  | some n =>
      match A.get a, storeSpan A.facts (A.get a) n, (A.get v).scTy with
      | .known (.sc _ x), some (lo, hi), some ty =>
          .ok ((A.withFactsK (afterStore (some x) (A.get v).bits n A.facts) (rangeKeeps lo hi)).parCtx v
            (if n == 8 && ty == .i64 && cellOk x then some x else none))
      | _, some (lo, hi), some _ => .ok (A.withFactsK (A.facts.filter (rangeKeeps lo hi)) (rangeKeeps lo hi))
      | _, some _, none => .ok (A.withFactsK (A.facts.filter (!·.isCell)) (!·.isCell))
      | ga, none, _ => .error s!"store of {n} bytes to {repr ga} the typestate gives no room for"

/-- Whether `n` bytes at slot `a` lie inside a heap block the state names,
    with no copy in flight and no worker running. -/
def memBlk (A : AS) (a : R) (n : Nat) : Bool :=
  decide (0 < n) && A.facts.contains (.part .idle true) &&
    match A.get a with
    | .into b _ hi =>
        (match A.get b with
         | .blk sz true => decide (hi + n ≤ sz)
         | _ => false)
    | .blk sz true => decide (n ≤ sz)
    | _ => false

/-- A scalar load from inside a heap block. -/
def blkLoad (A : AS) : Op → Option AV
  | .load op a => if op.ty.lanes.isNone && memBlk A a (loadBytes op) then some (loadAV op) else none
  | _ => none

/-- A store `absStore` refuses: one the linear facts show inside a region,
    or a scalar one inside a heap block, keeps every fact but the cells; where
    `fo` says the body may fault, any may fault, or write anywhere, and keeps
    the same. -/
def absStoreF (fo : Bool) (A : AS) (v a : R) (n : Nat) : Except String AS :=
  match absStore A v a n with
  | .ok A' => .ok A'
  | .error e =>
      if (storeBytes A v n).any (memRel A a) || ((A.get v).scTy.isSome && memBlk A a n) || fo then
        .ok (A.withFacts (A.facts.filter (!·.isCell)))
      else .error e

/-- A load the ranges cannot place, placed by the linear facts. -/
def relLoad (A : AS) : Op → Option AV
  | .load op a =>
      if loadKindOk op && (A.get a).scTy.isSome && memRel A a (loadBytes op) then some (loadAV op) else none
  | _ => none

/-- An operation's value pushed, with its facts and what it teaches of the
    slots it reads, ranges narrowed to what those teach. -/
def AS.pushOp (A : AS) (o : Op) (av : AV) : AS :=
  let B := A.pushS (opAV A o av) (symOp A o)
  let C := (opFacts A o).foldl (fun C (r, p) => if ubOk B p then C.addUb r p else C) B
  (opFacts A o).foldl (fun D (r, p) => match B.ctx.hiOf p.2 with
    | some k => D.tightenHi r p.1 k
    | none => D) C

/-- A 64-bit number as the parameters' record reads it: computed from them,
    or known. -/
def parI64 (A : AS) (r : R) : Option PExpr :=
  match A.par.slotOf r with
  | some (.i64, e) => some e
  | some _ => none
  | none => match A.get r with
    | .known (.sc .i64 c) => some (.lit c)
    | _ => none

/-- The number an operation computes from the parameters, where it computes
    one: a read of a cell holding a parameter, the low half of such a number,
    such a number times a known one. -/
def parOp (A : AS) : Op → Option (ClifTy × PExpr)
  | .load op a =>
      match op.kind, op.ty, A.get a with
      | .plain, .i64, .known (.sc _ x) => (A.par.cells.find? (·.1 == x)).map fun c => (.i64, .param c.2)
      | _, _, _ => none
  | .ireduce32 a =>
      match A.par.slotOf a with
      | some (.i64, e) => some (.i32, .trunc32 e)
      | _ => none
  | .imul a b =>
      match A.par.slotOf a, A.get b with
      | some (.i64, e), .known (.sc .i64 c) => some (.i64, .mul e (.lit c))
      | _, _ => none
  | .iadd a b =>
      match parI64 A a, parI64 A b with
      | some x, some y =>
          if (A.par.slotOf a).isSome || (A.par.slotOf b).isSome then some (.i64, .add x y) else none
      | _, _ => none
  | .isub a b =>
      match parI64 A a, parI64 A b with
      | some x, some y =>
          if (A.par.slotOf a).isSome || (A.par.slotOf b).isSome then some (.i64, .sub x y) else none
      | _, _ => none
  | _ => none

/-- The obligation a vendor call handed `pas` leaves: that its arguments'
    values are ones it accepts, given the allocations the parameters size and
    the relations they stand in. -/
def eOblOf (A : AS) (e : Ext) (pas : List PArg) : Obl := ⟨e, pas, A.facts, A.par.allocs, A.par.conds⟩

/-- The allocation a call makes of a size the parameters give: `cuMemAlloc`
    handed a number the record computes makes allocation `k`, `k` the count
    of them before it. -/
def sizedAlloc (A : AS) (e : Ext) (args : List R) : List (Nat × PExpr) :=
  match e, args with
  | .cuda .memAlloc, [_, sz] =>
      match nbufsOf A.facts, A.par.slotOf sz with
      | some k, some (.i64, x) => [(k, x)]
      | _, _ => []
  | _, _ => []

/-- The state after a vendor call handed `args` its row leaves `S` of: of
    what the record says of the world, the allocations it sizes and the one
    the call makes of a size they give, where the call keeps every allocation
    and kernels keep the sizes of the buffers they write; and the cells it
    names, where the call writes no host memory. -/
def AS.afterVendor (A0 : AS) (e : Ext) (args : List R) (S : TState) : AS :=
  let A := A0.forgetKinds e.kills
  if e.keepsAllocs && A.facts.contains .oracles then
    let keep := writesNone e
    (A.withFacts S).withPar { A.par.clearWorld with
      allocs := sizedAlloc A e args ++ A.par.allocs,
      cells := if keep then A.par.cells.filter (cellOk ·.1) else [],
      ctxs := if keep then A.par.ctxs.filter (cellOk ·.2) else [] }
  else A.withFacts S

/-- The cell a load reads the context of a hash table from: a 64-bit load at
    a known address the typestate says holds one. -/
def ctxOp (A : AS) : Op → Option UInt64
  | .load op a =>
      match op.kind, op.ty, A.get a with
      | .plain, .i64, .known (.sc _ x) => if (htAtOf x A.facts).isSome then some x else none
      | _, _, _ => none
  | _ => none

/-- What a library call answers: for `calloc` asked for a known number of
    bytes, null or a heap block of that many; for whether a library is there,
    a flag; for an engine or window function, a number of the type it
    returns; for an open, null or a handle held; any value otherwise. -/
def extAnswer (e : Ext) (as : List AV) : AV :=
  match e, as.mapM AV.bits with
  | .c .calloc, some [n, sz] => .blk (n.toNat * sz.toNat) false
  | .present _, _ => .sc .i32
  | .base f, _ => .sc ((Ext.base f).sig.2.getD .i64)
  | .window f, _ => if f = .open then .hand .window false else .sc ((Ext.window f).sig.2.getD .i64)
  | _, _ =>
      match e.opens with
      | some k => .hand k false
      | none => .any

/-- Whether a statement that keeps a call's answer (`val`), or one that drops
    it, may name the callee: a library function answers as its signature says. -/
def answersAs (c : Callee) (val : Bool) : Bool :=
  match c with
  | .ext e => e.sig.2.isSome == val
  | _ => true

/-- What a call into a library keeps of a typestate, having written, outside
    the heap, at most the bytes from `lo` up to `hi`: that no copy is in flight, every region's room outside pinned memory, and the
    cells clear of those bytes. -/
def libKeeps (lo hi : Nat) : Fact → Bool
  | .part p b => p == .idle && b
  | .room r _ => r != .pinned
  | .roomArg r _ => r != .pinned
  | .cell c _ => cellOk c && (hi ≤ c.toNat || c.toNat + 8 ≤ lo)
  | _ => false

/-- The cell slot `r` was read from, where the state says it holds what one
    holds. -/
def ctxOf (A : AS) (r : R) : Option UInt64 := (A.par.ctxs.find? (·.1 == r)).map (·.2)

/-- The state after a call into a library that wrote, outside the heap, at
    most the bytes from `lo` up to `hi`, and left `new`. -/
def AS.libDone (A : AS) (lo hi : Nat) (new : TState) : AS :=
  (A.forgetHands.withFacts (new ++ A.facts.filter (libKeeps lo hi))).withPar (A.par.keepWorld (libKeeps lo hi))

/-- `htInit` at a known slot with room leaves an empty table there. -/
def AS.libInit (A : AS) (a : R) : Option (AS × AV) :=
  match A.get a with
  | .known (.sc .i64 x) =>
      if cellOk x && memN A.facts (A.get a) 8 then some (A.libDone x.toNat (x.toNat + 8) [.htAt x 0 0], .any)
      else none
  | _ => none

/-- `htCleanup` at a slot holding a table leaves null. -/
def AS.libCleanup (A : AS) (a : R) : Option (AS × AV) :=
  match A.get a with
  | .known (.sc .i64 x) =>
      if (htAtOf x A.facts).isSome then some (A.libDone x.toNat (x.toNat + 8) [.cell x 0], .any) else none
  | _ => none

/-- `htCreate`, handed what a slot holding a table was read to hold, leaves
    the table, made once more. -/
def AS.libCreate (A : AS) (c : R) : Option (AS × AV) :=
  match ctxOf A c with
  | some x =>
      match htAtOf x A.facts with
      | some (n, k) => if k + 1 < 2 ^ 64 then some (A.libDone 0 0 [.htAt x n (k + 1)], .sc .i32) else none
      | none => none
  | none => none

/-- `htInsert` of a key and a value of known lengths leaves the table, with
    values as long as the one stored. -/
def AS.libInsert (A : AS) (c kp kl vp vl : R) : Option (AS × AV) :=
  match ctxOf A c, A.get kl, A.get vl with
  | some x, .known (.sc .i32 a), .known (.sc .i32 b) =>
      match htAtOf x A.facts with
      | some (n, k) =>
          if decide (a.toNat < 2 ^ 32) && decide (b.toNat < 2 ^ 32) && (A.get kp).scTy == some .i64 &&
              (A.get vp).scTy == some .i64 && memN A.facts (A.get kp) a.toNat && memN A.facts (A.get vp) b.toNat then
            some (A.libDone 0 0 [.htAt x (max n b.toNat) k], .any)
          else none
      | none => none
  | _, _, _ => none

/-- `htLookup` of a key of known length, into room for the longest value
    clear of the table's slot, leaves the table. -/
def AS.libLookup (A : AS) (c kp kl rp : R) : Option (AS × AV) :=
  match ctxOf A c, A.get kl with
  | some x, .known (.sc .i32 a) =>
      match htAtOf x A.facts, storeSpan A.facts (A.get rp) 0 with
      | some (n, k), some (lo, hi) =>
          if decide (a.toNat < 2 ^ 32) && (A.get kp).scTy == some .i64 && (A.get rp).scTy == some .i64 &&
              memN A.facts (A.get kp) a.toNat && memN A.facts (A.get rp) n &&
              (hi + n ≤ x.toNat || x.toNat + 8 ≤ lo) then
            some (A.libDone lo (hi + n) [.htAt x n k], .sc .i32)
          else none
      | _, _ => none
  | _, _ => none

/-- A math function, handed singles, answers a single and leaves the world. -/
def AS.libMath (A : AS) (args : List R) : Option (AS × AV) :=
  if args.all fun r => (A.get r).scTy == some .f32 then some (A.libDone 0 0 [], .sc .f32) else none

/-- **The rule for a call into `Lib.Ht` or `Lib.Math`**, by the function
    called, with no worker running and no copy in flight. -/
def AS.libRule (A : AS) (f : LibFn) (args : List R) : Option (AS × AV) :=
  match f, args with
  | .htInit, [a] => A.libInit a
  | .htCleanup, [a] => A.libCleanup a
  | .htCreate, [c] => A.libCreate c
  | .htInsert, [c, kp, kl, vp, vl] => A.libInsert c kp kl vp vl
  | .htLookup, [c, kp, kl, rp] => A.libLookup c kp kl rp
  | .sinf, [x] => A.libMath [x]
  | .cosf, [x] => A.libMath [x]
  | .powf, [x, y] => A.libMath [x, y]
  | _, _ => none

def AS.libCall (A : AS) (f : LibFn) (args : List R) : Except String (AS × AV) :=
  if A.facts.contains (.part .idle true) then
    match A.libRule f args with
    | some r => .ok r
    | none => .error s!"call {f.cname}: precondition not decided (args {repr (args.map A.get)})"
  else .error s!"call {f.cname}: a worker may be running, or a copy in flight"

/-- What the state knows of a value, as a record's place: its bits, or the
    numbers they lie between where those say more than its type does. -/
def AV.pat (a : AV) : ArgPat :=
  match a.bits with
  | some b => .is b
  | none => match a.num with
    | some n => if n.lo = 0 ∧ 2 ^ n.t.width - 1 ≤ n.hi then .any else .within n.lo n.hi
    | none => .any

/-- What the state knows of the eight bytes at `q`: a cell it names, or a cell
    a slot is known to hold the value of. -/
def cellPat (A : AS) (q : UInt64) : ArgPat :=
  match cellOf A.facts q with
  | some v => .is v
  | none => match A.par.ctxs.find? (·.2 == q) with
    | some (r, _) => (A.get r).pat
    | none => .any

/-- What the state says a four-byte scalar at `p` holds: the low half of a
    cell at its address, or the high half of one four bytes before it. -/
def scalarPat (A : AS) (p : R) : ArgPat :=
  match (A.get p).bits with
  | some a =>
      match cellOf A.facts a with
      | some v => if cellOk a then .is (v &&& 0xffffffff) else .any
      | none =>
          match cellOf A.facts (a - 4) with
          | some v => if cellOk (a - 4) then .is (v >>> 32) else .any
          | none => .any
  | none => .any

/-- What a call reads from host memory, as the state records it. A kernel
    launch, its parameter values: one per parameter of the function it names,
    each what the state knows of the eight bytes the pointer to it points at,
    where it knows that pointer, and none at all where the state does not name
    the function or show every parameter readable. A strided-batched product,
    its two scalars (`scalarPat`). Any other call reads none. -/
def wordsOf (A : AS) : Callee → List R → Option (List ArgPat)
  | .ext (.cuda .launchKernel), [hf, _, _, _, _, _, _, _, _, pr, _] =>
      match (A.get hf).bits.bind (handleOf kFunc), (A.get pr).bits with
      | some fi, some P =>
          match fnOf A.facts fi with
          | some sizes =>
              if argsReadable A.facts P sizes then
                some (sizes.zipIdx.map fun (sz, i) =>
                  match cellOf A.facts (P + UInt64.ofNat (8 * i)) with
                  | some q => if sz == 8 then cellPat A q else .any
                  | none => .any)
              else none
          | none => none
      | _, _ => none
  | .ext (.cublas .sgemmStridedBatched), [_, _, _, _, _, _, pa, _, _, _, _, _, _, pb, _, _, _, _] =>
      some [scalarPat A pa, scalarPat A pb]
  | _, _ => some []

-- ---------------------------------------------------------------------------
-- Threads
-- ---------------------------------------------------------------------------

/-- The thread call a function is, if it is one. -/
def _root_.AlgorithmLib.IR.Ext.threadFn : Ext → Option BaseFn
  | .base f => if f.thread then some f else none
  | _ => none

theorem _root_.AlgorithmLib.IR.Ext.threadFn_vendor {e : Ext} (h : e.vendor = true) : e.threadFn = none := by
  cases e <;> first | rfl | simp [Ext.vendor] at h

theorem _root_.AlgorithmLib.IR.Ext.threadFn_aside {e : Ext} (h : e.aside = true) : e.threadFn = none := by
  cases e <;> first | rfl | (rename_i f; cases f <;> simp_all [Ext.aside, Ext.threadFn, BaseFn.thread])

/-- The typestate's threads: started, the arena's size, running, the spans
    starts own, and whether the system makes threads. -/
def threadsOf (S : TState) :
    Option (Nat × Nat × List (Nat × Nat × UInt64 × Nat × Nat) × List (Nat × UInt64 × Nat × Nat) × Bool) :=
  S.findSome? fun | .threads n sz live lends g => some (n, sz, live, lends, g) | _ => none

/-- `len` bytes at `off` clear of every running thread's span. -/
def spanFree (live : List (Nat × Nat × UInt64 × Nat × Nat)) (off len : Nat) : Bool :=
  live.all fun t => off + len ≤ t.2.2.2.1 || t.2.2.2.1 + t.2.2.2.2 ≤ off

/-- What starting a thread that owns the arena bytes `[off, off + len)`, at
    addresses `[lo, hi)`, leaves of a fact: none that says the arena is wholly
    the program's or that those bytes are, nor any reading them. -/
def spawnKeeps (lo hi off len : Nat) : Fact → Bool
  | .threads .. | .htAt .. | .cstr _ | .cstrIn .. | .text .. | .wgText .. => false
  | .room r _ => r != .arena
  | .owns o l => off + len ≤ o || o + l ≤ off
  | x => rangeKeeps lo hi x

/-- What joining a thread, which hands its bytes `[lo, hi)` back as it left
    them, leaves of a fact: none reading them. -/
def joinKeeps (lo hi : Nat) : Fact → Bool
  | .threads .. | .htAt .. | .cstr _ | .cstrIn .. | .text .. | .wgText .. => false
  | x => rangeKeeps lo hi x

/-- **Starting a thread**: on a function and word known, which a start the
    typestate names owns a span of, clear of every running thread's, where the
    system makes threads. The answer is the new thread's handle. -/
def AS.spawnRule (A : AS) (args : List R) : Option (AS × AV) :=
  match args with
  | [f, a, st] =>
    match (A.get f).bits, (A.get a).bits, threadsOf A.facts with
    | some x, some y, some (n, sz, live, lends, true) =>
      if fnBase ≤ x && (A.get st).scTy.isSome then
        match lends.find? (fun l => l.1 == (x - fnBase).toNat && l.2.1 == y) with
        | some (_, _, off, len) =>
          if 0 < len && off + len ≤ sz && spanFree live off len && n < 2 ^ 48 then
            let lo := (regionBase .arena).toNat + off
            let kf := spawnKeeps lo (lo + len) off len
            some (A.withFactsK
              (.threads (n + 1) sz ((n, (x - fnBase).toNat, y, off, len) :: live) lends true :: A.facts.filter kf) kf,
              .known (.sc .i64 (threadHandle n)))
          else none
        | none => none
      else none
    | _, _, _ => none
  | _ => none

/-- **Joining a thread**: of a handle known, of a thread running. Its bytes
    come back as it left them, which the typestate knows nothing of; once no
    thread runs, the arena is the program's again. -/
def AS.joinRule (A : AS) (args : List R) : Option (AS × AV) :=
  match args with
  | [h] =>
    match (A.get h).bits, threadsOf A.facts with
    | some x, some (n, sz, live, lends, g) =>
      match threadOf x with
      | some k =>
        match live.find? (·.1 == k) with
        | some (_, _, _, off, len) =>
          if off + len ≤ sz then
            let lo := (regionBase .arena).toNat + off
            let kf := joinKeeps lo (lo + len)
            let live' := live.filter (·.1 != k)
            let S := .threads n sz live' lends g :: A.facts.filter kf
            some (A.withFactsK (if live'.isEmpty then .room .arena sz :: S else S) kf, .sc .i64)
          else none
        | none => none
      | none => none
    | _, _ => none
  | _ => none

/-- **The call rule.** The state after a call the checker accepts, its answer
    not yet appended, and what it answers: a function of a library linked into
    the program by `AS.libCall`, where `lb` says library calls are read; a library function the typestate follows by what
    it needs (`preExt`) and keeps (`extKeepsA`); no other callee is
    accepted. -/
def AS.callee (A : AS) (lb : Bool) (c : Callee) (args : List R) : Except String (AS × AV) :=
  match c with
  | .lib f =>
      if lb then A.libCall f args else .error s!"call {f.cname}: a library's, and library calls are not read"
  | .ext e =>
    match e.threadFn with
    | some .threadSpawn =>
      match A.spawnRule args with
      | some r => .ok r
      | none => .error s!"threadSpawn {args}: not a start the typestate names, clear of every running thread's"
    | some _ =>
      match A.joinRule args with
      | some r => .ok r
      | none => .error s!"threadJoin {args}: not a running thread's handle"
    | none =>
      let as := args.map A.get
      if e.vendor then
        if A.facts.contains (.part .idle true) &&
            as.all (·.scTy.isSome) then
          match vendorRow A.facts e as with
          | some (S, av) =>
              if vendorVal A.facts e (as.map (·.bits)) || vendorValR A.facts e (as.map AV.rng) then
                .ok (A.afterVendor e args S, av)
              else match args.mapM (parArg A) with
                | some pas =>
                    if A.par.obls.contains (eOblOf A e pas) then .ok (A.afterVendor e args S, av)
                    else if A.par.collect then
                      .ok ((A.afterVendor e args S).withPar
                        { (A.afterVendor e args S).par with obls := A.par.obls ++ [eOblOf A e pas] }, av)
                    else .error s!"call {repr e}: what it needs of its arguments' values not decided (args {repr as})"
                | none => .error s!"call {repr e}: what it needs of its arguments' values not decided (args {repr as})"
          | none => .error s!"call {repr e}: precondition not decided (args {repr as})"
        else .error s!"call {repr e}: a worker may be running, or a copy in flight"
      else if e.aside && as.all (·.scTy.isSome) && preExt A.facts (relRb A args) e as then
        .ok ((A.forgetExt e).withFacts (A.facts.filter (extKeepsA e as)), extAnswer e as)
      else .error s!"call {repr e}: precondition not decided (args {repr as})"
  | c => .error s!"call of {repr c} not handled"

/-- What a statement leaves; where `fo` says the body may fault, an operation
    or a store the checker cannot show defined is let fault. -/
def absStmt (fo lb : Bool) (A : AS) : Stmt → Except String AS
  | .op o => match absOp A o with
      | some av => .ok (((A.pushOp o av).parSlot A.len (parOp A o)).parCtx A.len (ctxOp A o))
      | none =>
          match relLoad A o with
          | some av => .ok (A.push av)
          | none =>
              match blkLoad A o with
              | some av => .ok (A.push av)
              | none =>
                  if fo then .ok (A.push .any)
                  else .error s!"operation {o.name} not shown defined on {repr (o.regs.map A.get)}"
  | .store ty v a => absStoreF fo A v a (tyBytes ty)
  | .storeUnaligned v a => absStoreF fo A v a (tyBytes ((A.get v).scTy.getD .i64))
  | .istore8 v a => match (A.get v).scTy with
      | some _ => absStoreF fo A v a 1
      | none => if fo then .ok (A.withFacts (A.facts.filter (!·.isCell))) else .error "istore8 of a non-scalar"
  | .call c args =>
      match A.callee lb c args with
      | .ok (B, av) =>
          if answersAs c true then .ok (B.push av) else .error s!"call of {repr c}: it answers nothing"
      | .error e => .error e
  | .callVoid c args =>
      match A.callee lb c args with
      | .ok (B, _) => if answersAs c false then .ok B else .error s!"call of {repr c}: its answer is dropped"
      | .error e => .error e

/-- A statement, for a refusal's message. -/
def stmtStr (n : Nat) : Stmt → String
  | .op o => s!"v{n} = {o.name} {o.regs}"
  | .store ty v a => s!"store.{repr ty} v{v} -> [v{a}]"
  | .storeUnaligned v a => s!"store v{v} -> [v{a}]"
  | .istore8 v a => s!"istore8 v{v} -> [v{a}]"
  | .call c as => s!"v{n} = call {match c with | .lib f => f.cname | c => (repr c).pretty} {as}"
  | .callVoid c as => s!"call {match c with | .lib f => f.cname | c => (repr c).pretty} {as}"

/-- The slots a statement reads. -/
def stmtRegs : Stmt → List R
  | .op o => o.regs
  | .store _ v a | .storeUnaligned v a | .istore8 v a => [v, a]
  | .call _ as | .callVoid _ as => as

/-- The slots a slot's linear facts read. -/
def AS.symReads (A : AS) (r : R) : List R :=
  let σ := A.sym r
  ((σ.dfn.toList ++ σ.ubs.map (·.2)).map fun L => L.terms.filterMap fun
    | (.slot y, _) => some y | _ => none).flatten

/-- What the checker knows of each of `rs` and of the slots their facts read,
    two steps back, for a refusal's message. -/
def slotsStr (A : AS) (rs : List R) : String :=
  let rs1 := rs ++ (rs.map A.symReads).flatten
  let rs2 := rs1 ++ (rs1.map A.symReads).flatten
  String.intercalate "\n" (rs2.eraseDups.map fun r =>
    s!"    v{r}: {repr (A.get r)}" ++ (if A.sym r = {} then "" else s!"\n      {repr (A.sym r)}"))

def absStmtsH (fo lb : Bool) (hist : List (Nat × Stmt)) (A : AS) : List Stmt → Except String AS
  | [] => .ok A
  | s :: ss =>
      match absStmt fo lb A s with
      | .ok A' => absStmtsH fo lb (((A.len, s) :: hist).take 8) A' ss
      | .error e => .error (e ++ "\n  at: " ++ stmtStr A.len s ++ "\n  after: " ++
          toString (hist.reverse.map fun (n, s) => stmtStr n s) ++ "\n  slots:\n" ++
          slotsStr A (stmtRegs s ++ (hist.map fun (_, s) => stmtRegs s).flatten) ++
          "\n  facts but cells:\n    " ++ toString ((A.facts.filter (!·.isCell)).map fun x => repr x))

def absStmts (fo lb : Bool) (A : AS) (ss : List Stmt) : Except String AS := absStmtsH fo lb [] A ss

/-- Whether `[lo, hi)` misses every range of `keep`. -/
def spanOut (keep : List (Nat × Nat)) (lo hi : Nat) : Bool := keep.all fun p => hi ≤ p.1 || p.2 ≤ lo

/-- A store of `n` bytes of slot `v` to slot `a` that `keep` admits: a
    scalar's, to bytes the state places, all outside `keep`. -/
def storeOut (keep : List (Nat × Nat)) (A : AS) (v a : R) (n : Nat) : Bool :=
  (A.get v).scTy.isSome && match storeSpan A.facts (A.get a) n with
    | some (lo, hi) => spanOut keep lo hi
    | none => false

/-- A statement `keep` admits: where it names ranges, a store is one
    `storeOut` admits and a call is not into a library. -/
def stmtKept (keep : List (Nat × Nat)) (A : AS) : Stmt → Bool
  | .store ty v a => keep.isEmpty || storeOut keep A v a (tyBytes ty)
  | .storeUnaligned v a => keep.isEmpty || storeOut keep A v a (tyBytes ((A.get v).scTy.getD .i64))
  | .istore8 v a => keep.isEmpty || storeOut keep A v a 1
  | .call (.lib _) _ | .callVoid (.lib _) _ => keep.isEmpty
  | _ => true

/-- Every statement of a straight run is one `keep` admits, each from the
    state the run reaches it in. -/
def storesKept (keep : List (Nat × Nat)) (fo lb : Bool) (A : AS) : List Stmt → Bool
  | [] => true
  | s :: ss => stmtKept keep A s && match absStmt fo lb A s with
      | .ok A' => storesKept keep fo lb A' ss
      | .error _ => true

/-- The call a statement makes, as the state it runs from records it: each
    argument's bits where the state knows them. -/
def recOf (A : AS) : Stmt → List CallRec
  | .call c args | .callVoid c args => [⟨c, args.map (fun r => (A.get r).pat) ++ ((wordsOf A c args).getD [])⟩]
  | _ => []

/-- The calls a straight run makes, in order, each recorded from the state its
    statement runs from. -/
def callRecs (fo lb : Bool) (A : AS) : List Stmt → List CallRec
  | [] => []
  | s :: ss =>
      recOf A s ++ match absStmt fo lb A s with
        | .ok A' => callRecs fo lb A' ss
        | .error _ => []

/-- Where a call's answer is one of two the contract tells apart, the state
    after it for each, from `A'`, the state after it: `fileReadToPtr`, asked
    for a known positive number of bytes, answers `-1` or at most that many. -/
def answerCases (A A' : AS) : Stmt → Option (AS × AS)
  | .call (.ext (.base .fileReadToPtr)) [_, _, _, _, n] =>
      match A.get n with
      | .known (.sc _ s) =>
          if 0 < s.toNat then
            some (A'.setAV A.len (.known (.sc .i64 0xFFFFFFFFFFFFFFFF)),
              A'.setAV A.len (.int ⟨.i64, 0, s.toNat, 1, none⟩))
          else none
      | _ => none
  | _ => none

/-- A straight run with a call whose answer `answerCases` splits: the state it
    ends with in each case. -/
def splitCases (fo lb : Bool) (A : AS) : List Stmt → Option (AS × AS)
  | [] => none
  | s :: ss =>
      match absStmt fo lb A s with
      | .error _ => none
      | .ok A' =>
          match answerCases A A' s with
          | some (B1, B2) =>
              match absStmts fo lb B1 ss, absStmts fo lb B2 ss with
              | .ok C1, .ok C2 => some (C1, C2)
              | _, _ => none
          | none => splitCases fo lb A' ss

/-- A piece whose end `splitCases` splits. -/
def pieceCases (fo lb : Bool) (A : AS) : Piece → Option (AS × AS)
  | .straight ss => splitCases fo lb A ss
  | _ => none

-- ---------------------------------------------------------------------------
-- Control flow
-- ---------------------------------------------------------------------------

/-- An address known exactly, bounded by the room the facts give its region:
    what a loop entered with it carries round. -/
def AV.enrich (S : TState) (a : AV) : AV :=
  match a.asPtr with
  | .ptr r n =>
      match n.exact, n.fit with
      | some k, none =>
          if k ≤ roomOf S r ∧ r ≠ .pinned then .ptr r { n with fit := some (r, 1, roomOf S r - k) } else a
      | _, _ => a
  | _ => a

/-- Carries and facts a loop goes round with, and each carry's linear facts,
    which read only slots from before the loop. -/
structure Carry where
  vals : List AV
  facts : TState
  syms : List Sym := []
  /-- The slots the loop is entered with, and for each carry, where it moves
      in step with an earlier one: `(j, r)`, the carry less what it was
      entered with is `r` times carry `j` less what that was entered with,
      modulo `2 ^ 64`. -/
  inits : List R := []
  ivs : List (Option (Nat × Int)) := []
  /-- What each carry goes round with less what it came in with, where that
      is a number: what `ivs` are guessed from, no part of what the carry
      state describes. -/
  steps : List (Option Int) := []
  /-- Slots from before the loop that hold, all the way round, what a cell of
      memory holds: `(r, a)`, slot `r` and the cell at `a`. -/
  ctxs : List (R × UInt64) := []
  /-- Slots from before the loop that hold, all the way round, a live handle:
      `(r, k)`, slot `r` and the handle's kind. -/
  hands : List (R × Contracts.Held) := []

/-- Equality of slot trees, by an equality of what they hold. -/
def Slots.beqBy {α : Type} (f : α → α → Bool) : Slots α → Slots α → Bool
  | .leaf, .leaf => true
  | .node l x r, .node l' x' r' => f x x' && beqBy f l l' && beqBy f r r'
  | _, _ => false

/-- Equality of states, decided. -/
def AS.beq (A B : AS) : Bool :=
  A.len == B.len && Slots.beqBy AV.beq A.slots B.slots && decide (A.facts = B.facts) &&
    Slots.beqBy (decide <| · = ·) A.syms B.syms && decide (A.par = B.par)

/-- What a run of the checker found, handed to a later run: the invariant each
    loop settled at, and the state each piece started from, each named by
    where it sits in the body (the index of each piece on the way down, `0`
    for a branch's then-arm or a loop's prefix and `1` for an else-arm or a
    loop's body). A run handed them checks each loop once and starts each
    piece from the state written out. `fo`: whether the body may fault, so
    that only its calls are checked; `guess`: whether the run only guesses
    which carries step together, each loop guessing whether or not it is
    refused without. `tr`: whether the run records the calls of each straight
    run; one that does not says only that it may make any. -/
structure Cert where
  loops : List (List Nat × Carry) := []
  states : List (List Nat × AS) := []
  fo : Bool := false
  guess : Bool := false
  tr : Bool := false
  /-- Calls into the libraries a program is linked with are read by the
      libraries' statements. -/
  lib : Bool := false
  /-- Address ranges `[lo, hi)` no store of the body may touch: where any is
      given, every store is a scalar's to bytes the state places outside them,
      and no call is into a library. -/
  keep : List (Nat × Nat) := []

instance : Append Cert :=
  ⟨fun a b => ⟨a.loops ++ b.loops, a.states ++ b.states, a.fo, a.guess, a.tr, a.lib, a.keep⟩⟩

/-- What a region may end with, and the invariant each loop in it settled at:
    each `br` and `cont` with the state it leaves from and the slots it
    passes. `tr` is a pattern over the calls a run of the region makes,
    however it ends. `cnt`, of one trip of a loop: where the trip is counted
    (`tripCnt`), the carry that counts, its bound, and the calls of the
    condition prefix and of the body. -/
structure AOut where
  ok : Option AS
  brk : List (Nat × AS × List R)
  cont : List (Nat × AS × List R)
  certs : Cert := {}
  tr : Tr
  cnt : Option (Nat × UInt64 × Tr × Tr) := none

def AOut.none : AOut := ⟨Option.none, [], [], {}, .never, Option.none⟩

/-- Whether slot `r` is described as a live handle of kind `k`. -/
def AS.holdsAt (A : AS) (r : R) (k : Contracts.Held) : Bool :=
  match A.get r with
  | .hand k' true => decide (k' = k)
  | _ => false

/-- The slots described as live handles, with their kinds. -/
def AS.handsHeld (A : AS) : List (R × Contracts.Held) :=
  (List.range A.len).filterMap fun r =>
    match A.get r with
    | .hand k true => some (r, k)
    | _ => none

/-- The state describing each slot of `hs` as a live handle of its kind. -/
def AS.withHands (A : AS) (hs : List (R × Contracts.Held)) : AS :=
  hs.foldl (fun A h => A.setAV h.1 (.hand h.2 true)) A

def Carry.join (a b : Carry) : Carry :=
  ⟨List.zipWith AV.join a.vals b.vals, factsMeet a.facts b.facts, List.zipWith Sym.join a.syms b.syms,
    a.inits, if a.inits = b.inits then List.zipWith (fun x y => if x = y then x else none) a.ivs b.ivs else [],
    a.steps, a.ctxs.filter b.ctxs.contains, a.hands.filter b.hands.contains⟩

def Carry.beq (a b : Carry) : Bool :=
  a.vals.length == b.vals.length && (List.zipWith AV.beq a.vals b.vals).all id &&
    (decide (a.facts = b.facts) || a.facts.all (b.facts.contains ·) && b.facts.all (a.facts.contains ·)) &&
    decide (a.syms = b.syms) &&
    decide (a.inits = b.inits) && decide (a.ivs = b.ivs) && decide (a.ctxs = b.ctxs) &&
    decide (a.hands = b.hands)

/-- A description still changing, widened: a range growing first to the
    largest number its type reads as not negative, then to its type's, and a
    bound still moving dropped. -/
def Num.widen (o n : Num) (top : Nat) : Num :=
  { n with lo := if o.lo ≤ n.lo then n.lo else 0,
           hi := if n.hi ≤ o.hi then n.hi else max n.hi top,
           fit := if o.fit = n.fit then n.fit else none }

def AV.widen (old new : AV) : AV :=
  match old, new with
  | .int o, .int n =>
      .int (o.widen n (if n.hi < 2 ^ (n.t.width - 1) then 2 ^ (n.t.width - 1) - 1 else 2 ^ n.t.width - 1))
  | .ptr _ o, .ptr r n => .ptr r (o.widen n (span - if 0 < n.al ∧ n.al ≤ span then n.al else 1))
  | _, n => n

def Carry.widen (old new : Carry) : Carry :=
  ⟨List.zipWith AV.widen old.vals new.vals, new.facts, new.syms, new.inits, new.ivs, new.steps, new.ctxs, new.hands⟩

def depthDown (xs : List (Nat × AS × List R)) : List (Nat × AS × List R) :=
  xs.filterMap fun (d, A, args) => match d with | 0 => none | d + 1 => some (d, A, args)

/-- What a value a loop passes on is no larger than: those of the slots `Y`
    from before the loop the checker shows it is at most, less a number both
    are multiples of where it shows it is below. -/
def closeSym (A : AS) (Y : List R) (a : R) : Sym :=
  { ubs := Y.filterMap fun y =>
      let g : Int := max 1 (Nat.gcd (A.get a).absAl (A.get y).absAl)
      if A.ctx.prove proveFuel (((lin1 a 1).sub (lin1 y 1)).add (Lin.const g)) then some (1, ⟨[(.slot y, 1)], -g⟩)
      else if A.ctx.prove proveFuel ((lin1 a 1).sub (lin1 y 1)) then some (1, lin1 y 1) else none }

/-- The slots a region's comparisons read. -/
def cmpsGo : Nat → List Piece → List R
  | 0, _ => []
  | _ + 1, [] => []
  | fuel + 1, .straight ss :: ps =>
      (ss.filterMap fun | .op (.icmp _ a b) => some [a, b] | _ => none).flatten ++ cmpsGo fuel ps
  | fuel + 1, .loop _ pre body :: ps => cmpsGo fuel pre ++ cmpsGo fuel body ++ cmpsGo fuel ps
  | fuel + 1, .dloop _ body :: ps => cmpsGo fuel body ++ cmpsGo fuel ps
  | fuel + 1, .ite _ thn els _ _ :: ps => cmpsGo fuel thn ++ cmpsGo fuel els ++ cmpsGo fuel ps
  | fuel + 1, _ :: ps => cmpsGo fuel ps

/-- The slots from before a loop its comparisons read: what its carries are
    bounded by. -/
def loopCands (n0 : Nat) (c : Code) : List R := (cmpsGo HProg.fuel c).eraseDups.filter (· < n0)

/-- How deep a form is unfolded to compare carries. -/
def ivDepth : Nat := 12

/-- What each value a loop goes round with less the carry it stands for, `n0`
    the first carry's slot, where that is a number. -/
def stepsOf (A : AS) (n0 : Nat) (args : List R) : List (Option Int) :=
  (List.range args.length).map fun i =>
    let L := (A.ctx.foldKnown (A.ctx.deepen ivDepth ((lin1 (args.getD i 0) 1).sub (lin1 (n0 + i) 1)))).norm
    if L.terms = [] then some L.k else none

/-- Which of `c`'s steps-together the values a loop goes round with keep:
    each that the facts show, modulo `2 ^ 64`. -/
def ivKeep (A : AS) (n0 : Nat) (c : Carry) (args : List R) : List (Option (Nat × Int)) :=
  (List.range args.length).map fun i =>
    match c.ivs.getD i none with
    | some (j, r) =>
        let ii := c.inits.getD i 0
        let ij := c.inits.getD j 0
        let D : Lin := ((lin1 (args.getD i 0) 1).sub (lin1 ii 1)).sub
          (((lin1 (args.getD j 0) 1).sub (lin1 ij 1)).scale r)
        let L := (A.ctx.foldKnown (A.ctx.deepen ivDepth D)).norm
        if L.terms = [] ∧ L.k % 2 ^ 64 = 0 ∧ j < args.length ∧ ii < n0 ∧ ij < n0 then some (j, r) else none
    | none => none

/-- The carry state a value list goes round with, from a trip that started
    from `c`. -/
def backCarry (A : AS) (Y : List R) (c : Carry) (n0 : Nat) (args : List R) : Carry :=
  ⟨(args.map A.get).map AV.portable, A.facts, args.map (closeSym A Y), c.inits, ivKeep A n0 c args,
    stepsOf A n0 args, A.par.ctxs.filter (·.1 < n0), c.hands.filter fun h => decide (h.1 < n0) && A.holdsAt h.1 h.2⟩

/-- What a value leaving a loop is exactly, in the slots from before it. -/
def closeDfn (A : AS) (n0 : Nat) (a : R) : Option Lin :=
  let E := A.ctx.foldKnown (A.ctx.deepen ivDepth (lin1 a 1)).norm
  if E.slots.all (· < n0) ∧ E.regions.all (· ≠ .pinned) then some E else none

/-- Two linear facts together. -/
def Sym.merge (σ τ : Sym) : Sym := { dfn := σ.dfn.or τ.dfn, ubs := σ.ubs ++ τ.ubs }

/-- What a value leaving a loop is: exactly what `closeDfn` says, and no larger
    than `closeSym` says. -/
def exitSym (A : AS) (Y : List R) (n0 : Nat) (a : R) : Sym :=
  Sym.merge { dfn := closeDfn A n0 a } (closeSym A Y a)

def conts0 (xs : List (Nat × AS × List R)) (Y : List R) (c : Carry) (n0 : Nat) : List Carry :=
  xs.filterMap fun (d, A, args) =>
    if d == 0 then some (backCarry A Y c n0 args) else none

def brks0 (xs : List (Nat × AS × List R)) (ab : Nat) (Y : List R) (n0 : Nat) : List AS :=
  xs.filterMap fun (d, A, args) =>
    if d == 0 then some (A.bindAt ab (args.map A.get) (args.map (exitSym A Y n0))) else none

/-- What carry `i` steps with, as a linear fact on its slot. -/
def ivSym (n0 k : Nat) (inits : List R) (i : Nat) : Option (Nat × Int) → Sym
  | some (j, r) =>
      if j < k ∧ inits.getD i 0 < n0 ∧ inits.getD j 0 < n0 then
        { dfn := some ⟨[(.slot (inits.getD i 0), 1), (.slot (n0 + j), r), (.slot (inits.getD j 0), -r)], 0⟩ }
      else {}
  | none => {}

/-- The linear facts a loop's carries are bound with at its head, `n0` the
    first one's slot. -/
def Carry.headSyms (c : Carry) (n0 : Nat) : List Sym :=
  (List.range c.vals.length).map fun i =>
    Sym.merge (c.syms.getD i {}) (ivSym n0 c.vals.length c.inits i (c.ivs.getD i none))

/-- The steps every way round agrees on. -/
def stepsJoin (k : Nat) : List (List (Option Int)) → List (Option Int)
  | [] => List.replicate k none
  | [s] => s
  | s :: ss => List.zipWith (fun x y => if x = y then x else none) s (stepsJoin k ss)

/-- Carries that step together, guessed from their steps: each with a nonzero
    step paired with the first earlier carry whose step divides it. -/
def ivPropose (steps : List (Option Int)) : List (Option (Nat × Int)) :=
  (List.range steps.length).map fun i =>
    match steps.getD i none with
    | some di =>
        if di = 0 then none
        else
          match (List.range i).find? (fun j => match steps.getD j none with
              | some dj => decide (dj ≠ 0 ∧ di % dj = 0)
              | none => false) with
          | some j => (steps.getD j none).map fun dj => (j, di / dj)
          | none => none
    | none => none

def joinAll : List AS → Except String (Option AS)
  | [] => .ok none
  | A :: As => do
      match ← joinAll As with
      | none => return some A
      | some B => return some (← A.join B)

/-- Iterate a loop's trip from `c` until the carries it goes round with stop
    changing, widening ranges that keep growing. -/
def settle (path : List Nat) (trip : Carry → Except String (List Carry × List AS × AOut)) :
    Nat → Carry → Except String AOut
  | n, c =>
      match trip c with
      | .error e => .error e
      | .ok (backs, exits, out) =>
          if backs.all (·.vals.length == c.vals.length) then
            let c' := backs.foldl Carry.join c
            if c'.beq c then
              match joinAll exits with
              | .error e => .error e
              | .ok j => .ok ⟨j, out.brk, out.cont, { out.certs with loops := (path, c) :: out.certs.loops },
                  .star out.tr, out.cnt⟩
            else
              match n with
              | 0 => .error "loop invariant did not settle"
              | n + 1 => settle path trip n (if n < 9 then Carry.widen c c' else c')
          else .error "a loop goes round with a different number of carries"

/-- A loop checked against an invariant it was given: the invariant, joined
    with the carries the loop is entered with where it does not already cover
    them, and one trip from it going round only with states it covers. -/
def checkLoop (path : List Nat) (trip : Carry → Except String (List Carry × List AS × AOut))
    (c0 c : Carry) : Except String AOut :=
  if c0.vals.length = c.vals.length then
    let ci := if (c0.join c).beq c then c else c0.join c
    match trip ci with
    | .error e => .error e
    | .ok (backs, exits, out) =>
        if backs.all (·.vals.length == ci.vals.length) && (backs.foldl Carry.join ci).beq ci then
          match joinAll exits with
          | .error e => .error e
          | .ok j => .ok ⟨j, out.brk, out.cont, { out.certs with loops := (path, ci) :: out.certs.loops },
              .star out.tr, out.cnt⟩
        else .error "the loop goes round with a state its invariant does not cover"
  else .error "the invariant given has a different number of carries"

/-- A loop by its invariant in `cert` where there is one, iterated until it
    settles where there is not, and then narrowed where one more trip shows a
    smaller invariant. -/
def runLoop (cert : Cert) (path : List Nat) (trip : Carry → Except String (List Carry × List AS × AOut))
    (c0 : Carry) : Except String AOut :=
  match cert.loops.lookup path with
  | some c => checkLoop path trip c0 c
  | none =>
      match settle path trip 12 c0 with
      | .error e => .error e
      | .ok o =>
          -- narrowed: the carries entered with and what one trip from the
          -- settled invariant goes round with, where that is an invariant too
          match o.certs.loops with
          | (_, c) :: _ =>
              match trip c with
              | .ok (backs, _, _) =>
                  match checkLoop path trip c0 (backs.foldl Carry.join c0) with
                  | .ok o' => .ok o'
                  | .error _ => .ok o
              | .error _ => .ok o
          | [] => .ok o

/-- The steps-together the invariant `cert` gives a loop has. -/
def certIvs (cert : Cert) (path : List Nat) : List (Option (Nat × Int)) :=
  ((cert.loops.lookup path).map (·.ivs)).getD []

/-- The steps-together a run of a loop as one that may fault suggests: those
    of the steps one trip from where it settles goes round with. -/
def ivGuess (cert : Cert) (path : List Nat) (tripFo : Carry → Except String (List Carry × List AS × AOut))
    (c0 : Carry) : List (Option (Nat × Int)) :=
  match runLoop cert path tripFo c0 with
  | .ok oF =>
      match oF.certs.loops.lookup path with
      | some cF =>
          match tripFo cF with
          | .ok (backs, _, _) => ivPropose (stepsJoin cF.vals.length (backs.map (·.steps)))
          | .error _ => []
      | none => []
  | .error _ => []

/-- A loop by `runLoop`, its carries entered with the steps-together that a
    guessing run of it as one that may fault (`tripFo`) suggests, where it has
    two carries or more and some are suggested. -/
def withIvs (cert : Cert) (path : List Nat) (trip tripFo : Carry → Except String (List Carry × List AS × AOut))
    (c0 : Carry) : Except String AOut :=
  if c0.vals.length < 2 then .error "one carry"
  else
    let ivs := ivGuess cert path tripFo c0
    if ivs.any Option.isSome then runLoop cert path trip { c0 with ivs := ivs } else .error "none step together"

/-- A loop by `runLoop` against the invariant `cert` gives; without one, a
    run that guesses tries carries stepping together first and one that does
    not tries them where it is refused without. -/
def runLoopIv (cert : Cert) (path : List Nat) (trip tripFo : Carry → Except String (List Carry × List AS × AOut))
    (c0 : Carry) : Except String AOut :=
  match cert.loops.lookup path with
  | some _ => runLoop cert path trip c0
  | none =>
      if cert.guess then
        match withIvs cert path trip tripFo c0 with
        | .ok o => .ok o
        | .error _ => runLoop cert path trip c0
      else
        match runLoop cert path trip c0 with
        | .ok o => .ok o
        | .error e =>
            match withIvs cert path trip tripFo c0 with
            | .ok o => .ok o
            | .error _ => .error e

/-- The state a piece starts from: where `cert` writes one out, that one,
    once the state reached is shown to be it. -/
def pin (cert : Cert) (path : List Nat) (A : AS) : Except String AS :=
  match cert.states.lookup path with
  | some A' => if A.beq A' then .ok A' else .error "the state reached is not the one written out"
  | none => .ok A

/-- What a loop body's parameters are: each, exactly, the carry `k` slots
    from `n0` it is bound from. -/
def copySyms (n0 k : Nat) : List Sym := (List.range k).map fun i => { dfn := some (lin1 (n0 + i) 1) }

/-- Whether slot `a` is, wherever `A` holds, slot `x` and one more, modulo
    `2 ^ 64`. -/
def stepsOne (A : AS) (a x : R) : Bool :=
  let L := (A.ctx.foldKnown (A.ctx.deepen ivDepth (((lin1 a 1).sub (lin1 x 1)).sub (Lin.const 1)))).norm
  decide (L.terms = []) && decide (L.k % 2 ^ 64 = 0)

/-- Whether a loop that goes on while slot `x` is below a bound counts its
    trips with it: the test exits when it fails, `x` is one of the carries,
    neither the prefix nor the body leaves by a `br` or a `cont`, and the body
    goes round with that carry, a 64-bit number, one more. -/
def cntOk (l : Loop) (n0 k : Nat) (op ob : AOut) (x : R) : Bool :=
  !l.exitOnTrue && op.brk.isEmpty && op.cont.isEmpty && ob.brk.isEmpty && ob.cont.isEmpty &&
    decide (n0 ≤ x ∧ x < n0 + k) &&
    (match ob.ok with
     | some A2 => decide (x - n0 < l.cont.length) &&
         decide ((A2.get (l.cont.getD (x - n0) 0)).scTy = some .i64) &&
         stepsOne A2 (l.cont.getD (x - n0) 0) x
     | none => true)

/-- **A counted trip**: the loop goes on exactly while one carry is below a
    bound `n` the state knows, and counts its trips with it (`cntOk`). Answers
    the carry, the bound, and the calls of the prefix and of the body. -/
def tripCnt (l : Loop) (n0 k : Nat) (op : AOut) (Ap : AS) (ob : AOut) : Option (Nat × UInt64 × Tr × Tr) :=
  match Ap.get l.flag with
  | .cmp .ult x y =>
      match Ap.get y with
      | .known (.sc .i64 n) => if cntOk l n0 k op ob x then some (x - n0, n, op.tr, ob.tr) else none
      | _ => none
  | _ => none

/-- The calls of a loop whose trips are counted (`q`) and whose counting carry
    is entered with a known number no larger than its bound: the prefix's and
    the body's, as many times as the bound is above it, then the prefix's.
    Otherwise `t`. -/
def cntTr (A : AS) (l : Loop) (t : Tr) (q : Option (Nat × UInt64 × Tr × Tr)) : Tr :=
  match q with
  | some (j, n, P, B) =>
      match A.get (l.init.getD j 0) with
      | .known (.sc .i64 c0) =>
          if j < l.init.length ∧ c0.toNat ≤ n.toNat then .seq (.rep (n.toNat - c0.toNat) (.seq P B)) P else t
      | _ => t
  | none => t

/-- A loop's result with its calls counted where its trips are. -/
def AOut.counted (o : AOut) (A : AS) (l : Loop) : AOut :=
  ⟨o.ok, o.brk, o.cont, o.certs, cntTr A l o.tr o.cnt, Option.none⟩

/-- One trip of a top-tested loop, from what its condition prefix ended with
    (`op`) and a way to run its body: the carries that go round again, the
    states it leaves with, and the `br`s and `cont`s past it; its calls are
    the prefix's, then none or the body's. -/
def loopTrip (l : Loop) (n0 k preEnd ab : Nat) (Y : List R) (c : Carry) (op : AOut)
    (runBody : AS → Except String AOut) : Except String (List Carry × List AS × AOut) :=
  let backs := conts0 op.cont Y c n0
  let exits := brks0 op.brk ab Y n0
  let outB := depthDown op.brk
  let outC := depthDown op.cont
  match op.ok with
  | none => .ok (backs, exits, ⟨none, outB, outC, op.certs, op.tr, Option.none⟩)
  | some Ap =>
      let exitT : List AS := match refine Ap l.flag l.exitOnTrue with
        | some Ax => [Ax.bindAt ab (l.exitR.map Ax.get) (l.exitR.map (exitSym Ax Y n0))]
        | none => []
      match refine Ap l.flag (!l.exitOnTrue) with
      | none => .ok (backs, exits ++ exitT, ⟨none, outB, outC, op.certs, op.tr, Option.none⟩)
      | some Ag =>
          match runBody (Ag.bindAt preEnd ((List.range k).map fun i => Ag.get (n0 + i)) (copySyms n0 k)) with
          | .error e => .error e
          | .ok ob =>
              let fall : List Carry := match ob.ok with
                | some A2 => [backCarry A2 Y c n0 l.cont]
                | none => []
              .ok (backs ++ conts0 ob.cont Y c n0 ++ fall, exits ++ exitT ++ brks0 ob.brk ab Y n0,
                ⟨none, outB ++ depthDown ob.brk, outC ++ depthDown ob.cont, op.certs ++ ob.certs,
                  .seq op.tr (.alt (.lit []) ob.tr), tripCnt l n0 k op Ap ob⟩)

/-- One trip of a bottom-tested loop, from what its body ended with: the
    carries that go round again where the back-edge test can say so, the
    states it leaves with where the test can say stop or a `br` leaves, and the
    `br`s and `cont`s past it. -/
def dloopTrip (l : DLoop) (n0 ab : Nat) (Y : List R) (c : Carry) (ob : AOut) : List Carry × List AS × AOut :=
  let goes : List Carry := match ob.ok with
    | some A2 => match refine A2 l.flag l.contOnTrue with
      | some Ag => [backCarry Ag Y c n0 l.cont]
      | none => []
    | none => []
  let leaves : List AS := match ob.ok with
    | some A2 => match refine A2 l.flag (!l.contOnTrue) with
      | some Ax => [Ax.bindAt ab (l.exitIdx.map fun i => (l.cont.map Ax.get).getD i .any)
          (l.exitIdx.map fun i => (l.cont.map (exitSym Ax Y n0)).getD i {})]
      | none => []
    | none => []
  (conts0 ob.cont Y c n0 ++ goes, brks0 ob.brk ab Y n0 ++ leaves,
    ⟨none, depthDown ob.brk, depthDown ob.cont, ob.certs, ob.tr, Option.none⟩)

/-- The state a bottom-tested loop is entered with: where it has a guard, what
    the guard saying go teaches; `none` where it cannot say go. -/
def dEnter (A : AS) (l : DLoop) : Option AS :=
  match l.guard with
  | none => some A
  | some g => refine A g l.contOnTrue

/-- The state a bottom-tested loop is skipped with: where it has a guard that can
    say stop, the exits bound from the carries it would have been entered with. -/
def dSkip (A : AS) (l : DLoop) (ab : Nat) (Y : List R) : Option AS :=
  match l.guard with
  | none => none
  | some g => (refine A g (!l.contOnTrue)).map fun Ax =>
      Ax.bindAt ab (l.exitIdx.map fun i => (l.init.map Ax.get).getD i .any)
        (l.exitIdx.map fun i => (l.init.map (exitSym Ax Y A.len)).getD i {})

mutual

/-- What a piece at `path` may end with, from `A`. A loop with an invariant in
    `cert` is checked against it; one without is iterated until it settles. A
    straight run's calls are the records of its statements, a branch's either
    arm's, a loop's any number of its trips', or exactly as many as it makes
    where they are counted. -/
def execP (cert : Cert) (path : List Nat) (A : AS) : Piece → Except String AOut
  | .straight ss =>
      match absStmts cert.fo cert.lib A ss with
      | .error e => .error e
      | .ok A' =>
          if storesKept cert.keep cert.fo cert.lib A ss then
            .ok ⟨some A', [], [], {}, if cert.tr && !cert.lib then .lit (callRecs cert.fo cert.lib A ss) else .any,
              Option.none⟩
          else .error "a store may touch bytes the certificate keeps, or a call is into a library"
  | .br d args => .ok ⟨none, [(d, A, args)], [], {}, .lit [], Option.none⟩
  | .cont d args => .ok ⟨none, [], [(d, A, args)], {}, .lit [], Option.none⟩
  | .ite m thn els thnR elsR =>
      let thnEnd := slotsOf A.len thn
      let joinAt := slotsOf thnEnd els
      let armT : Except String AOut := match refine A m.flag true with
        | none => .ok AOut.none
        | some At => execC cert (0 :: path) 0 At thn
      let armE : Except String AOut := match refine A m.flag false with
        | none => .ok AOut.none
        | some Ae => execC cert (1 :: path) 0 (Ae.bindAt thnEnd []) els
      match armT, armE with
      | .ok ot, .ok oe =>
          match joinOpt (ot.ok.map fun A' => A'.bindAt joinAt (thnR.map A'.get))
              (oe.ok.map fun A' => A'.bindAt joinAt (elsR.map A'.get)) with
          | .ok j => .ok ⟨j, ot.brk ++ oe.brk, ot.cont ++ oe.cont, ot.certs ++ oe.certs, .alt ot.tr oe.tr, Option.none⟩
          | .error e => .error e
      | .error e, _ => .error e
      | _, .error e => .error e
  | .loop l pre body =>
      let n0 := A.len
      let k := l.pTys.length
      let preEnd := slotsOf (n0 + k) pre
      let ab := slotsOf (preEnd + k) body
      let Y := loopCands n0 (pre ++ body)
      let tripOf := fun (ct : Cert) (c : Carry) =>
        match execC ct (0 :: path) 0 ((((A.forgetHands.withFacts c.facts).withCtxs c.ctxs).withHands c.hands).bindAt n0 c.vals (c.headSyms n0)) pre with
        | .error e => .error e
        | .ok op => loopTrip l n0 k preEnd ab Y c op fun Ab => execC ct (1 :: path) 0 Ab body
      let c0 : Carry := ⟨((l.init.map A.get).map AV.portable).map (AV.enrich A.facts), A.facts,
        l.init.map (closeSym A Y), l.init, certIvs cert path, [], A.par.ctxs.filter (·.1 < n0), A.handsHeld⟩
      if l.init.length = k then
        match runLoopIv cert path (tripOf cert) (tripOf { cert with fo := true, guess := true }) c0 with
        | .error e => .error e
        | .ok o => .ok (o.counted A l)
      else .error "a loop entered with a different number of carries"
  | .dloop l body =>
      let n0 := A.len
      let k := l.pTys.length
      let ab := slotsOf (n0 + k) body
      let Y := loopCands n0 body
      if l.init.length = k then
        match dEnter A l with
        | none => .ok ⟨dSkip A l ab Y, [], [], {}, .lit [], Option.none⟩
        | some Ae =>
            let tripOf := fun (ct : Cert) (c : Carry) =>
              match execC ct (0 :: path) 0 ((((Ae.forgetHands.withFacts c.facts).withCtxs c.ctxs).withHands c.hands).bindAt n0 c.vals (c.headSyms n0)) body with
              | .error e => .error e
              | .ok ob => .ok (dloopTrip l n0 ab Y c ob)
            match runLoopIv cert path (tripOf cert) (tripOf { cert with fo := true, guess := true }) ⟨((l.init.map Ae.get).map AV.portable).map (AV.enrich Ae.facts), Ae.facts,
                l.init.map (closeSym Ae Y), l.init, certIvs cert path, [], Ae.par.ctxs.filter (·.1 < n0), Ae.handsHeld⟩ with
            | .error e => .error e
            | .ok o =>
                match joinOpt o.ok (dSkip A l ab Y) with
                | .ok j => .ok ⟨j, o.brk, o.cont, o.certs, o.tr, Option.none⟩
                | .error e => .error e
      else .error "a loop entered with a different number of carries"

/-- What a region whose pieces sit at `i :: base`, `(i + 1) :: base`, ... may
    end with, from `A`. Its calls are its first piece's and then the rest's;
    where that piece may leave by a `br` or a `cont`, its alone or then the
    rest's. -/
def execC (cert : Cert) (base : List Nat) (i : Nat) (A0 : AS) : List Piece → Except String AOut
  | [] => .ok ⟨some A0, [], [], {}, .lit [], Option.none⟩
  | p :: ps =>
      match pin cert (i :: base) A0 with
      | .error e => .error e
      | .ok A =>
          let here : Cert := { states := [(i :: base, A)] }
          match (if cert.keep.isEmpty then pieceCases cert.fo cert.lib A p else none) with
          | some (C1, C2) =>
              -- the rest of the region from each case, at paths of their own
              match execC cert (2 :: i :: base) (i + 1) C1 ps, execC cert (3 :: i :: base) (i + 1) C2 ps with
              | .ok o1, .ok o2 =>
                  match joinOpt o1.ok o2.ok with
                  | .ok j => .ok ⟨j, o1.brk ++ o2.brk, o1.cont ++ o2.cont, here ++ o1.certs ++ o2.certs,
                      .seq .any (.alt o1.tr o2.tr), Option.none⟩
                  | .error e => .error e
              | .error e, _ => .error e
              | _, .error e => .error e
          | none =>
              match execP cert (i :: base) A p with
              | .error e => .error e
              | .ok o =>
                  match o.ok with
                  | none => .ok ⟨none, o.brk, o.cont, here ++ o.certs, o.tr, Option.none⟩
                  | some A' =>
                      match execC cert base (i + 1) A' ps with
                      | .error e => .error e
                      | .ok o' => .ok ⟨o'.ok, o.brk ++ o'.brk, o.cont ++ o'.cont, here ++ o.certs ++ o'.certs,
                          if o.brk.isEmpty && o.cont.isEmpty then .seq o.tr o'.tr
                          else .seq o.tr (.alt (.lit []) o'.tr), Option.none⟩

end

end AlgorithmLib.Check
