module
public import AlgorithmLib.Check.Rel
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.Exec` — the checker over a function body

`exec` runs a body (`Code`, the term that ships) over abstract states: each
statement by its transfer function, a branch by both arms and a join, a loop by
iterating one trip until the state it carries round stops changing. It answers
what the body may end with — a state for falling off its end, and the states
each `br` and `cont` leave with — or why it refuses.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState after keeps roomAt afterStore storeKeeps cellOk readyIn)

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
    the other names it, a bound on the hash table's values the larger of the
    two bounds, a party knowing every access where the other says it does. -/
def meetFact (S' : TState) (x : Fact) : Option Fact :=
  if S'.contains x then some x else
  match x with
  | .htVals n => (htValsOf S').map fun m => .htVals (max n m)
  | .readyAll p => if readyIn S' p then some (.readyAll p) else none
  | _ => none

/-- A party the second typestate says knows every access, which the first,
    not naming it, says too. -/
def meetReady (S : TState) (x : Fact) : Bool :=
  match x with
  | .readyAll p => !S.contains x && readyIn S p
  | _ => false

/-- Facts holding wherever either typestate does. -/
def factsMeet (S S' : TState) : TState := S.filterMap (meetFact S') ++ S'.filter (meetReady S)

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
            if 0 < nx.lo + step then A2.addFact (.room r ((nx.lo + step) * s + d)) else A2
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

/-- What a branch learns from the flag it reads being `b`: `none` where the
    flag cannot be `b`. -/
def refine (A : AS) (flag : R) (b : Bool) : Option AS :=
  match A.get flag with
  | .known (.sc _ f) => if (f != 0) == b then some A else none
  | .cmp c x y => match holdsRel A c x y b with
    | some rel => some (learn A rel x y)
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
      | .known (.sc _ x), some (lo, hi), some _ =>
          .ok (A.withFactsK (afterStore (some x) (A.get v).bits n A.facts) (rangeKeeps lo hi))
      | _, some (lo, hi), some _ => .ok (A.withFactsK (A.facts.filter (rangeKeeps lo hi)) (rangeKeeps lo hi))
      | _, some _, none => .ok (A.withFactsK (A.facts.filter (!·.isCell)) (!·.isCell))
      | ga, none, _ => .error s!"store of {n} bytes to {repr ga} the typestate gives no room for"

/-- A store `absStore` refuses: one the linear facts show inside a region
    keeps every fact but the cells; where `fo` says the body may fault, any
    may fault, or write anywhere, and keeps the same. -/
def absStoreF (fo : Bool) (A : AS) (v a : R) (n : Nat) : Except String AS :=
  match absStore A v a n with
  | .ok A' => .ok A'
  | .error e =>
      if (storeBytes A v n).any (memRel A a) || fo then .ok (A.withFacts (A.facts.filter (!·.isCell)))
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
  | _ => none

/-- An argument as the parameters' record reads it: a number computed from
    them, or bits known. -/
def parArg (A : AS) (r : R) : Option PArg :=
  match A.par.slotOf r with
  | some (_, e) => some (.sym e)
  | none => (A.get r).bits.map .known

/-- The obligation a call of `f` handed `pas` leaves, in the state's facts and
    buffers. -/
def oblOf (A : AS) (f : Ffi) (pas : List PArg) : Obl := ⟨f, pas, A.facts, A.par.bufs⟩

/-- How a foreign call's precondition is shown, and the record after: decided
    by the facts; or an obligation on the parameters the record holds, or, in
    a run that gathers them, gathers. -/
def ffiPre (A : AS) (f : Ffi) (args : List R) : Option Par :=
  if preB f (args.map A.get) A.facts (relRb A args) then some A.par else
  match args.mapM (parArg A) with
  | some pas =>
      if A.par.obls.contains (oblOf A f pas) then some A.par
      else if A.par.collect then some { A.par with obls := A.par.obls ++ [oblOf A f pas] }
      else none
  | none => none

/-- The state after a foreign call the checker accepts, its answer not yet
    appended. -/
def AS.ffiCall (A : AS) (f : Ffi) (args : List R) : Except String AS :=
  let as := args.map A.get
  if f != .threadSpawn && A.facts.contains (.part .frozen false) && as.all (·.scTy.isSome) then
    match ffiPre A f args with
    | some P => .ok ((A.forgetHands.withFacts (postA f as A.facts)).withPar (P.keepWorld (keepA f as)))
    | none => .error s!"call {f.cname}: precondition not decided (args {repr as})"
  else .error s!"call {f.cname}: precondition not decided (args {repr as})"

/-- What a foreign call answers: the index `devAnswer` gives, or any number of
    its result type. -/
def ffiAnswer (S : TState) (f : Ffi) (as : List AV) : AV :=
  match devAnswer f as S with
  | some n => .known (ofInt .i32 n)
  | none => .sc (f.result.getD .i64)

/-- What a library call answers: for an open, null or a handle held; any value
    otherwise. -/
def extAnswer (e : Ext) : AV :=
  match e.opens with
  | some k => .hand k false
  | none => .any

/-- What a statement leaves; where `fo` says the body may fault, an operation
    or a store the checker cannot show defined is let fault. -/
def absStmt (fo : Bool) (A : AS) : Stmt → Except String AS
  | .op o => match absOp A o with
      | some av => .ok ((A.pushOp o av).parSlot A.len (parOp A o))
      | none =>
          match relLoad A o with
          | some av => .ok (A.push av)
          | none =>
              if fo then .ok (A.push .any)
              else .error s!"operation {o.name} not shown defined on {repr (o.regs.map A.get)}"
  | .store ty v a => absStoreF fo A v a (tyBytes ty)
  | .storeUnaligned v a => absStoreF fo A v a (tyBytes ((A.get v).scTy.getD .i64))
  | .istore8 v a => match (A.get v).scTy with
      | some _ => absStoreF fo A v a 1
      | none => if fo then .ok (A.withFacts (A.facts.filter (!·.isCell))) else .error "istore8 of a non-scalar"
  | .call (.ffi f) args =>
      match A.ffiCall f args with
      | .ok B => .ok (B.push (ffiAnswer A.facts f (args.map A.get)))
      | .error e => .error e
  | .callVoid (.ffi f) args => A.ffiCall f args
  | .call (.ext e) args =>
      let as := args.map A.get
      if e.aside && e.sig.2.isSome && as.all (·.scTy.isSome) && preExt A.facts e as then
        .ok (((A.forgetKind e.closes).withFacts (A.facts.filter (extKeepsA e as))).push (extAnswer e))
      else .error s!"call {repr e}: precondition not decided (args {repr as})"
  | .callVoid (.ext e) args =>
      let as := args.map A.get
      if e.aside && e.sig.2.isNone && as.all (·.scTy.isSome) && preExt A.facts e as then
        .ok ((A.forgetKind e.closes).withFacts (A.facts.filter (extKeepsA e as)))
      else .error s!"call {repr e}: precondition not decided (args {repr as})"
  | .call c _ | .callVoid c _ => .error s!"call of {repr c} not handled"

/-- A statement, for a refusal's message. -/
def stmtStr (n : Nat) : Stmt → String
  | .op o => s!"v{n} = {o.name} {o.regs}"
  | .store ty v a => s!"store.{repr ty} v{v} -> [v{a}]"
  | .storeUnaligned v a => s!"store v{v} -> [v{a}]"
  | .istore8 v a => s!"istore8 v{v} -> [v{a}]"
  | .call c as => s!"v{n} = call {match c with | .ffi f => f.cname | c => (repr c).pretty} {as}"
  | .callVoid c as => s!"call {match c with | .ffi f => f.cname | c => (repr c).pretty} {as}"

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

def absStmtsH (fo : Bool) (hist : List (Nat × Stmt)) (A : AS) : List Stmt → Except String AS
  | [] => .ok A
  | s :: ss =>
      match absStmt fo A s with
      | .ok A' => absStmtsH fo (((A.len, s) :: hist).take 8) A' ss
      | .error e => .error (e ++ "\n  at: " ++ stmtStr A.len s ++ "\n  after: " ++
          toString (hist.reverse.map fun (n, s) => stmtStr n s) ++ "\n  slots:\n" ++
          slotsStr A (stmtRegs s ++ (hist.map fun (_, s) => stmtRegs s).flatten))

def absStmts (fo : Bool) (A : AS) (ss : List Stmt) : Except String AS := absStmtsH fo [] A ss

/-- Where a call's answer is one of two the contract tells apart, the state
    after it for each, from `A'`, the state after it: `fileRead`, asked for a
    known positive number of bytes, answers `-1` or at most that many. -/
def answerCases (A A' : AS) : Stmt → Option (AS × AS)
  | .call (.ffi .fileRead) [_, _, _, _, n] =>
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
def splitCases (fo : Bool) (A : AS) : List Stmt → Option (AS × AS)
  | [] => none
  | s :: ss =>
      match absStmt fo A s with
      | .error _ => none
      | .ok A' =>
          match answerCases A A' s with
          | some (B1, B2) =>
              match absStmts fo B1 ss, absStmts fo B2 ss with
              | .ok C1, .ok C2 => some (C1, C2)
              | _, _ => none
          | none => splitCases fo A' ss

/-- A piece whose end `splitCases` splits. -/
def pieceCases (fo : Bool) (A : AS) : Piece → Option (AS × AS)
  | .straight ss => splitCases fo A ss
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
    refused without. -/
structure Cert where
  loops : List (List Nat × Carry) := []
  states : List (List Nat × AS) := []
  fo : Bool := false
  guess : Bool := false

instance : Append Cert := ⟨fun a b => ⟨a.loops ++ b.loops, a.states ++ b.states, a.fo, a.guess⟩⟩

/-- What a region may end with, and the invariant each loop in it settled at:
    each `br` and `cont` with the state it leaves from and the slots it
    passes. -/
structure AOut where
  ok : Option AS
  brk : List (Nat × AS × List R)
  cont : List (Nat × AS × List R)
  certs : Cert := {}

def AOut.none : AOut := ⟨Option.none, [], [], {}⟩

def Carry.join (a b : Carry) : Carry :=
  ⟨List.zipWith AV.join a.vals b.vals, factsMeet a.facts b.facts, List.zipWith Sym.join a.syms b.syms,
    a.inits, if a.inits = b.inits then List.zipWith (fun x y => if x = y then x else none) a.ivs b.ivs else [],
    a.steps⟩

def Carry.beq (a b : Carry) : Bool :=
  a.vals.length == b.vals.length && (List.zipWith AV.beq a.vals b.vals).all id &&
    a.facts.all (b.facts.contains ·) && b.facts.all (a.facts.contains ·) && decide (a.syms = b.syms) &&
    decide (a.inits = b.inits) && decide (a.ivs = b.ivs)

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
  ⟨List.zipWith AV.widen old.vals new.vals, new.facts, new.syms, new.inits, new.ivs, new.steps⟩

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
    stepsOf A n0 args⟩

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
              | .ok j => .ok ⟨j, out.brk, out.cont, { out.certs with loops := (path, c) :: out.certs.loops }⟩
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
          | .ok j => .ok ⟨j, out.brk, out.cont, { out.certs with loops := (path, ci) :: out.certs.loops }⟩
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

/-- One trip of a top-tested loop, from what its condition prefix ended with
    (`op`) and a way to run its body: the carries that go round again, the
    states it leaves with, and the `br`s and `cont`s past it. -/
def loopTrip (l : Loop) (n0 k preEnd ab : Nat) (Y : List R) (c : Carry) (op : AOut)
    (runBody : AS → Except String AOut) : Except String (List Carry × List AS × AOut) :=
  let backs := conts0 op.cont Y c n0
  let exits := brks0 op.brk ab Y n0
  let outB := depthDown op.brk
  let outC := depthDown op.cont
  match op.ok with
  | none => .ok (backs, exits, ⟨none, outB, outC, op.certs⟩)
  | some Ap =>
      let exitT : List AS := match refine Ap l.flag l.exitOnTrue with
        | some Ax => [Ax.bindAt ab (l.exitR.map Ax.get) (l.exitR.map (exitSym Ax Y n0))]
        | none => []
      match refine Ap l.flag (!l.exitOnTrue) with
      | none => .ok (backs, exits ++ exitT, ⟨none, outB, outC, op.certs⟩)
      | some Ag =>
          match runBody (Ag.bindAt preEnd ((List.range k).map fun i => Ag.get (n0 + i)) (copySyms n0 k)) with
          | .error e => .error e
          | .ok ob =>
              let fall : List Carry := match ob.ok with
                | some A2 => [backCarry A2 Y c n0 l.cont]
                | none => []
              .ok (backs ++ conts0 ob.cont Y c n0 ++ fall, exits ++ exitT ++ brks0 ob.brk ab Y n0,
                ⟨none, outB ++ depthDown ob.brk, outC ++ depthDown ob.cont, op.certs ++ ob.certs⟩)

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
    ⟨none, depthDown ob.brk, depthDown ob.cont, ob.certs⟩)

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
    `cert` is checked against it; one without is iterated until it settles. -/
def execP (cert : Cert) (path : List Nat) (A : AS) : Piece → Except String AOut
  | .straight ss =>
      match absStmts cert.fo A ss with
      | .error e => .error e
      | .ok A' => .ok ⟨some A', [], [], {}⟩
  | .br d args => .ok ⟨none, [(d, A, args)], [], {}⟩
  | .cont d args => .ok ⟨none, [], [(d, A, args)], {}⟩
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
          | .ok j => .ok ⟨j, ot.brk ++ oe.brk, ot.cont ++ oe.cont, ot.certs ++ oe.certs⟩
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
        match execC ct (0 :: path) 0 ((A.forgetHands.withFacts c.facts).bindAt n0 c.vals (c.headSyms n0)) pre with
        | .error e => .error e
        | .ok op => loopTrip l n0 k preEnd ab Y c op fun Ab => execC ct (1 :: path) 0 Ab body
      let c0 : Carry := ⟨((l.init.map A.get).map AV.portable).map (AV.enrich A.facts), A.facts,
        l.init.map (closeSym A Y), l.init, certIvs cert path, []⟩
      if l.init.length = k then runLoopIv cert path (tripOf cert) (tripOf { cert with fo := true, guess := true }) c0
      else .error "a loop entered with a different number of carries"
  | .dloop l body =>
      let n0 := A.len
      let k := l.pTys.length
      let ab := slotsOf (n0 + k) body
      let Y := loopCands n0 body
      if l.init.length = k then
        match dEnter A l with
        | none => .ok ⟨dSkip A l ab Y, [], [], {}⟩
        | some Ae =>
            let tripOf := fun (ct : Cert) (c : Carry) =>
              match execC ct (0 :: path) 0 ((Ae.forgetHands.withFacts c.facts).bindAt n0 c.vals (c.headSyms n0)) body with
              | .error e => .error e
              | .ok ob => .ok (dloopTrip l n0 ab Y c ob)
            match runLoopIv cert path (tripOf cert) (tripOf { cert with fo := true, guess := true }) ⟨((l.init.map Ae.get).map AV.portable).map (AV.enrich Ae.facts), Ae.facts,
                l.init.map (closeSym Ae Y), l.init, certIvs cert path, []⟩ with
            | .error e => .error e
            | .ok o =>
                match joinOpt o.ok (dSkip A l ab Y) with
                | .ok j => .ok ⟨j, o.brk, o.cont, o.certs⟩
                | .error e => .error e
      else .error "a loop entered with a different number of carries"

/-- What a region whose pieces sit at `i :: base`, `(i + 1) :: base`, ... may
    end with, from `A`. -/
def execC (cert : Cert) (base : List Nat) (i : Nat) (A0 : AS) : List Piece → Except String AOut
  | [] => .ok ⟨some A0, [], [], {}⟩
  | p :: ps =>
      match pin cert (i :: base) A0 with
      | .error e => .error e
      | .ok A =>
          let here : Cert := { states := [(i :: base, A)] }
          match pieceCases cert.fo A p with
          | some (C1, C2) =>
              -- the rest of the region from each case, at paths of their own
              match execC cert (2 :: i :: base) (i + 1) C1 ps, execC cert (3 :: i :: base) (i + 1) C2 ps with
              | .ok o1, .ok o2 =>
                  match joinOpt o1.ok o2.ok with
                  | .ok j => .ok ⟨j, o1.brk ++ o2.brk, o1.cont ++ o2.cont, here ++ o1.certs ++ o2.certs⟩
                  | .error e => .error e
              | .error e, _ => .error e
              | _, .error e => .error e
          | none =>
              match execP cert (i :: base) A p with
              | .error e => .error e
              | .ok o =>
                  match o.ok with
                  | none => .ok ⟨none, o.brk, o.cont, here ++ o.certs⟩
                  | some A' =>
                      match execC cert base (i + 1) A' ps with
                      | .error e => .error e
                      | .ok o' => .ok ⟨o'.ok, o.brk ++ o'.brk, o.cont ++ o'.cont, here ++ o.certs ++ o'.certs⟩

end

end AlgorithmLib.Check
