module
public import AlgorithmLib.Check.Calls
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

open AlgorithmLib.HProg.Contracts (Fact TState after keeps roomAt afterStore storeKeeps cellOk)

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
    two bounds. -/
def meetFact (S' : TState) (x : Fact) : Option Fact :=
  if S'.contains x then some x else
  match x with
  | .htVals n => (htValsOf S').map fun m => .htVals (max n m)
  | _ => none

/-- Facts holding wherever either typestate does. -/
def factsMeet (S S' : TState) : TState := S.filterMap (meetFact S')

/-- The join of two states over environments of one length. -/
def AS.join (A B : AS) : Except String AS :=
  if A.len = B.len then
    .ok ⟨Slots.zip AV.join A.slots B.slots, A.len, factsMeet A.facts B.facts⟩
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

/-- `k` slots from `i` on known as nothing. -/
def Slots.clear (t : Slots) (i : Nat) : Nat → Slots
  | 0 => t
  | k + 1 => (t.set i .any).clear (i + 1) k

/-- `bindAt`, abstractly: grow or cut to `n` slots, then append `vs`. Growing
    leaves what the state knows; cutting forgets, in what the slots kept say,
    the slots cut away. -/
def AS.bindAt (A : AS) (n : Nat) (vs : List AV) : AS :=
  let base : AS :=
    if A.len ≤ n then ⟨A.slots.clear A.len (n - A.len), n, A.facts⟩
    else ⟨A.slots.map (AV.cut n), n, A.facts⟩
  (vs.map AV.portable).foldl AS.push base

-- ---------------------------------------------------------------------------
-- Branch conditions
-- ---------------------------------------------------------------------------

/-- A slot replaced by a description known to hold of it too; one known
    exactly is kept. -/
def AS.setAV (A : AS) (r : R) (av : AV) : AS :=
  match A.get r with
  | .known _ => A
  | _ => if r < A.len then ⟨A.slots.set r av, A.len, A.facts⟩ else A

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
  | _, _ => none

/-- What `x + e ≤ y` teaches, `e` zero or one: `x` is below what `y` may be and
    inside the bound `y` has, `y` above what `x` may be, and where `x` is known,
    the region `y` is bounded by holds that much. Where both are multiples of
    some `g`, `x < y` gives `x + g ≤ y`. -/
def lessThan (A : AS) (x y : R) (e : Nat) : AS :=
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
        match nx.exact, ny.fit with
        | some k, some (r, s, d) => A2.withFacts (.room r ((k + step) * s + d) :: A2.facts)
        | _, _ => A2
      else A
  | _, _ => A

/-- What `x rel y` teaches. -/
def learn (A : AS) (rel : ICmpCond) (x y : R) : AS :=
  match rel with
  | .ult => lessThan A x y 1
  | .ule => lessThan A x y 0
  | .ugt => lessThan A y x 1
  | .uge => lessThan A y x 0
  | .eq => lessThan (lessThan A x y 0) y x 0
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
    facts about bytes outside it; a vector store, every fact but the cells. -/
def absStore (A : AS) (v a : R) (n : Nat) : Except String AS :=
  match storeBytes A v n with
  | none => .error s!"store of a value of unknown type (slot {v})"
  | some n =>
      match A.get a, storeSpan A.facts (A.get a) n, (A.get v).scTy with
      | .known (.sc _ x), some _, some _ => .ok (A.withFacts (afterStore (some x) (A.get v).bits n A.facts))
      | _, some (lo, hi), some _ => .ok (A.withFacts (A.facts.filter (rangeKeeps lo hi)))
      | _, some _, none => .ok (A.withFacts (A.facts.filter (!·.isCell)))
      | ga, none, _ => .error s!"store of {n} bytes to {repr ga} the typestate gives no room for"

/-- A store where `fo` says the body may fault: one `absStore` refuses may
    fault, or write anywhere, so every fact but the cells. -/
def absStoreF (fo : Bool) (A : AS) (v a : R) (n : Nat) : Except String AS :=
  match absStore A v a n with
  | .ok A' => .ok A'
  | .error e => if fo then .ok (A.withFacts (A.facts.filter (!·.isCell))) else .error e

/-- What a statement leaves; where `fo` says the body may fault, an operation
    or a store the checker cannot show defined is let fault. -/
def absStmt (fo : Bool) (A : AS) : Stmt → Except String AS
  | .op o => match absOp A o with
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
      let as := args.map A.get
      if f != .threadSpawn && A.facts.contains (.part .frozen false) && as.all (·.scTy.isSome) && preB f as A.facts then
        .ok ((A.withFacts (postA f as A.facts)).push (.sc (f.result.getD .i64)))
      else .error s!"call {f.cname}: precondition not decided (args {repr as})"
  | .callVoid (.ffi f) args =>
      let as := args.map A.get
      if f != .threadSpawn && A.facts.contains (.part .frozen false) && as.all (·.scTy.isSome) && preB f as A.facts then
        .ok (A.withFacts (postA f as A.facts))
      else .error s!"call {f.cname}: precondition not decided (args {repr as})"
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

/-- What the checker knows of each of `rs`, for a refusal's message. -/
def slotsStr (A : AS) (rs : List R) : String :=
  String.intercalate "\n" (rs.eraseDups.map fun r => s!"    v{r}: {repr (A.get r)}")

def absStmtsH (fo : Bool) (hist : List (Nat × Stmt)) (A : AS) : List Stmt → Except String AS
  | [] => .ok A
  | s :: ss =>
      match absStmt fo A s with
      | .ok A' => absStmtsH fo (((A.len, s) :: hist).take 8) A' ss
      | .error e => .error (e ++ "\n  at: " ++ stmtStr A.len s ++ "\n  after: " ++
          toString (hist.reverse.map fun (n, s) => stmtStr n s) ++ "\n  slots:\n" ++
          slotsStr A (stmtRegs s ++ (hist.map fun (_, s) => stmtRegs s).flatten))

def absStmts (fo : Bool) (A : AS) (ss : List Stmt) : Except String AS := absStmtsH fo [] A ss

-- ---------------------------------------------------------------------------
-- Control flow
-- ---------------------------------------------------------------------------

/-- The most room the facts give region `r`. -/
def roomOf (S : TState) (r : Region) : Nat :=
  S.foldl (fun m f => match f with | .room r' n => if r' = r then max m n else m | _ => m) 0

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

/-- Carries and facts a loop goes round with. -/
structure Carry where
  vals : List AV
  facts : TState

/-- Equality of slot trees, decided. -/
def Slots.beq : Slots → Slots → Bool
  | .leaf, .leaf => true
  | .node l x r, .node l' x' r' => x.beq x' && l.beq l' && r.beq r'
  | _, _ => false

/-- Equality of states, decided. -/
def AS.beq (A B : AS) : Bool :=
  A.len == B.len && A.slots.beq B.slots && decide (A.facts = B.facts)

/-- What a run of the checker found, handed to a later run: the invariant each
    loop settled at, and the state each piece started from, each named by
    where it sits in the body (the index of each piece on the way down, `0`
    for a branch's then-arm or a loop's prefix and `1` for an else-arm or a
    loop's body). A run handed them checks each loop once and starts each
    piece from the state written out. `fo`: whether the body may fault, so
    that only its calls are checked. -/
structure Cert where
  loops : List (List Nat × Carry) := []
  states : List (List Nat × AS) := []
  fo : Bool := false

instance : Append Cert := ⟨fun a b => ⟨a.loops ++ b.loops, a.states ++ b.states, a.fo⟩⟩

/-- What a region may end with, and the invariant each loop in it settled at. -/
structure AOut where
  ok : Option AS
  brk : List (Nat × AS × List AV)
  cont : List (Nat × List AV × TState)
  certs : Cert := {}

def AOut.none : AOut := ⟨Option.none, [], [], {}⟩

def Carry.join (a b : Carry) : Carry :=
  ⟨List.zipWith AV.join a.vals b.vals, factsMeet a.facts b.facts⟩

def Carry.beq (a b : Carry) : Bool :=
  a.vals.length == b.vals.length && (List.zipWith AV.beq a.vals b.vals).all id &&
    a.facts.all (b.facts.contains ·) && b.facts.all (a.facts.contains ·)

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
  ⟨List.zipWith AV.widen old.vals new.vals, new.facts⟩

def depthDown (xs : List (Nat × AS × List AV)) : List (Nat × AS × List AV) :=
  xs.filterMap fun (d, A, vs) => match d with | 0 => none | d + 1 => some (d, A, vs)

def depthDownC (xs : List (Nat × List AV × TState)) : List (Nat × List AV × TState) :=
  xs.filterMap fun (d, vs, S) => match d with | 0 => none | d + 1 => some (d, vs, S)

def conts0 (xs : List (Nat × List AV × TState)) : List Carry :=
  xs.filterMap fun (d, vs, S) => if d == 0 then some ⟨vs, S⟩ else none

def brks0 (xs : List (Nat × AS × List AV)) (ab : Nat) : List AS :=
  xs.filterMap fun (d, A, vs) => if d == 0 then some (A.bindAt ab vs) else none

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

/-- The state a piece starts from: where `cert` writes one out, that one,
    once the state reached is shown to be it. -/
def pin (cert : Cert) (path : List Nat) (A : AS) : Except String AS :=
  match cert.states.lookup path with
  | some A' => if A.beq A' then .ok A' else .error "the state reached is not the one written out"
  | none => .ok A

/-- One trip of a top-tested loop, from what its condition prefix ended with
    (`op`) and a way to run its body: the carries that go round again, the
    states it leaves with, and the `br`s and `cont`s past it. -/
def loopTrip (l : Loop) (n0 k preEnd ab : Nat) (op : AOut) (runBody : AS → Except String AOut) :
    Except String (List Carry × List AS × AOut) :=
  let backs := conts0 op.cont
  let exits := brks0 op.brk ab
  let outB := depthDown op.brk
  let outC := depthDownC op.cont
  match op.ok with
  | none => .ok (backs, exits, ⟨none, outB, outC, op.certs⟩)
  | some Ap =>
      let exitT : List AS := match refine Ap l.flag l.exitOnTrue with
        | some Ax => [Ax.bindAt ab (l.exitR.map Ax.get)]
        | none => []
      match refine Ap l.flag (!l.exitOnTrue) with
      | none => .ok (backs, exits ++ exitT, ⟨none, outB, outC, op.certs⟩)
      | some Ag =>
          match runBody (Ag.bindAt preEnd ((List.range k).map fun i => Ag.get (n0 + i))) with
          | .error e => .error e
          | .ok ob =>
              let fall : List Carry := match ob.ok with
                | some A2 => [⟨(l.cont.map A2.get).map AV.portable, A2.facts⟩]
                | none => []
              .ok (backs ++ conts0 ob.cont ++ fall, exits ++ exitT ++ brks0 ob.brk ab,
                ⟨none, outB ++ depthDown ob.brk, outC ++ depthDownC ob.cont, op.certs ++ ob.certs⟩)

/-- One trip of a bottom-tested loop, from what its body ended with: the
    carries that go round again where the back-edge test can say so, the
    states it leaves with where the test can say stop or a `br` leaves, and the
    `br`s and `cont`s past it. -/
def dloopTrip (l : DLoop) (ab : Nat) (ob : AOut) : List Carry × List AS × AOut :=
  let goes : List Carry := match ob.ok with
    | some A2 => match refine A2 l.flag l.contOnTrue with
      | some Ag => [⟨(l.cont.map Ag.get).map AV.portable, Ag.facts⟩]
      | none => []
    | none => []
  let leaves : List AS := match ob.ok with
    | some A2 => match refine A2 l.flag (!l.contOnTrue) with
      | some Ax => [Ax.bindAt ab (l.exitIdx.map fun i => (l.cont.map Ax.get).getD i .any)]
      | none => []
    | none => []
  (conts0 ob.cont ++ goes, brks0 ob.brk ab ++ leaves,
    ⟨none, depthDown ob.brk, depthDownC ob.cont, ob.certs⟩)

/-- The state a bottom-tested loop is entered with: where it has a guard, what
    the guard saying go teaches; `none` where it cannot say go. -/
def dEnter (A : AS) (l : DLoop) : Option AS :=
  match l.guard with
  | none => some A
  | some g => refine A g l.contOnTrue

/-- The state a bottom-tested loop is skipped with: where it has a guard that can
    say stop, the exits bound from the carries it would have been entered with. -/
def dSkip (A : AS) (l : DLoop) (ab : Nat) : Option AS :=
  match l.guard with
  | none => none
  | some g => (refine A g (!l.contOnTrue)).map fun Ax =>
      Ax.bindAt ab (l.exitIdx.map fun i => (l.init.map Ax.get).getD i .any)

mutual

/-- What a piece at `path` may end with, from `A`. A loop with an invariant in
    `cert` is checked against it; one without is iterated until it settles. -/
def execP (cert : Cert) (path : List Nat) (A : AS) : Piece → Except String AOut
  | .straight ss =>
      match absStmts cert.fo A ss with
      | .error e => .error e
      | .ok A' => .ok ⟨some A', [], [], {}⟩
  | .br d args => .ok ⟨none, [(d, A, args.map A.get)], [], {}⟩
  | .cont d args => .ok ⟨none, [], [(d, (args.map A.get).map AV.portable, A.facts)], {}⟩
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
      let trip := fun c : Carry =>
        match execC cert (0 :: path) 0 ((A.withFacts c.facts).bindAt n0 c.vals) pre with
        | .error e => .error e
        | .ok op => loopTrip l n0 k preEnd ab op fun Ab => execC cert (1 :: path) 0 Ab body
      let c0 : Carry := ⟨((l.init.map A.get).map AV.portable).map (AV.enrich A.facts), A.facts⟩
      if l.init.length = k then runLoop cert path trip c0
      else .error "a loop entered with a different number of carries"
  | .dloop l body =>
      let n0 := A.len
      let k := l.pTys.length
      let ab := slotsOf (n0 + k) body
      if l.init.length = k then
        match dEnter A l with
        | none => .ok ⟨dSkip A l ab, [], [], {}⟩
        | some Ae =>
            let trip := fun c : Carry =>
              match execC cert (0 :: path) 0 ((Ae.withFacts c.facts).bindAt n0 c.vals) body with
              | .error e => .error e
              | .ok ob => .ok (dloopTrip l ab ob)
            match runLoop cert path trip ⟨((l.init.map Ae.get).map AV.portable).map (AV.enrich Ae.facts), Ae.facts⟩ with
            | .error e => .error e
            | .ok o =>
                match joinOpt o.ok (dSkip A l ab) with
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
