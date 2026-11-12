module
public import AlgorithmLib.ML.Machine.WarpEmit
meta import AlgorithmLib.ML.Machine.WarpEmit
public import AlgorithmLib.Kernel.Block
meta import AlgorithmLib.Kernel.Block
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
public import AlgorithmLib.ML.Num.Bits
meta import AlgorithmLib.ML.Num.Bits
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.Body` — warp statements as kernel bodies

`body nbuf s` is the body a thread of a one-warp block runs for the warp
statement `s` (`EWStmt`): lane is the thread, the block is the CTA, buffer
`b` is the `b`-th pointer parameter, an index counts 32-bit elements. A float
register is a value the statements thread through; a loop carries the
registers its body writes.

Each operation computes what it means at the carrier `F32` (`F32.ops`):
`inv` is `1 / x` and `rsqrt` `1 / sqrt x`, both exactly rounded; `ex2` is the
hardware's approximation and `exp` is `ex2 (x · log₂ e)`; `geF` is `1.0` where
`a ≥ b` and `0.0` otherwise; a literal is its word.
`maxW` is `x` where `x ≥ y` and `y` otherwise, as `NumOps.max` defines it.

An integer register is a 64-bit value, or a predicate where only comparisons
set it (`flagRegs`); one known to be a 32-bit word zero-extended (a ballot's,
an index shuffle's, trailing zeros) keeps the word too, so its 32-bit uses and
comparisons need no conversion (`low32`, `narrow`). A test that is a
comparison branches on it directly (`condOf`). A branch joins the integer
registers its arms set, a `while` loop carries the ones it names, and
`locals` puts its registers back by rebinding alone. Where a `while` loop's
body ends in a branch or a loop, each arm makes the assignments after it and
jumps to the loop's head, with no join (`Tail`, `goRound`).
-/

namespace AlgorithmLib.ML.Body

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.ML AlgorithmLib.Kernel AlgorithmLib.HProg.Sem
open AlgorithmLib.Kernel.Block (shflBfly shflIdx barrier ex2Approx c32 ballotR)

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- What a statement sees: the thread's lane and block, the buffers'
    addresses, shared memory's, the innermost loop's counter, and zeros. -/
structure Ctx (V : ClifTy → Type) where
  lane : V .i32
  cta : V .i32
  bufs : List (V .i64)
  sm : V .i64
  i : V .i32
  zf : V .f32
  zi : V .i32
  z64 : V .i64
  /-- The integer registers held as predicates: those only comparisons set. -/
  fl : List Nat := []

/-- The registers' values: each float register's, each integer register's
    held as a 64-bit value and each held as a predicate, the latest binding
    first. A register never written reads zero. -/
structure Regs (V : ClifTy → Type) where
  f : List (Nat × V .f32) := []
  v : List (Nat × V .i64) := []
  p : List (Nat × V .i8) := []
  /-- Integer registers whose value is a 32-bit word zero-extended, with the word. -/
  w : List (Nat × V .i32) := []

def get (c : Ctx V) (r : Regs V) (n : Nat) : V .f32 := (r.f.lookup n).getD c.zf
def set (r : Regs V) (n : Nat) (v : V .f32) : Regs V := { r with f := (n, v) :: r.f.filter (·.1 != n) }
def setV (r : Regs V) (n : Nat) (x : V .i64) : Regs V :=
  { r with v := (n, x) :: r.v, w := r.w.filter (·.1 != n) }
/-- Integer register `n` := `x`, the zero extension of the 32-bit `x32`. -/
def setW (r : Regs V) (n : Nat) (x : V .i64) (x32 : V .i32) : Regs V :=
  { r with v := (n, x) :: r.v, w := (n, x32) :: r.w.filter (·.1 != n) }
def setP (r : Regs V) (n : Nat) (x : V .i8) : Regs V := { r with p := (n, x) :: r.p }

/-- Element `ix` of the buffer at `base`. -/
def at4 (base : V .i64) (ix : V .i32) : Prog V L (V .i64) := do
  iadd base (← ishlImm (← uextend64 ix) 2)

def buf (c : Ctx V) (b : Nat) : V .i64 := c.bufs.getD b c.z64

def idx (c : Ctx V) : IdxE → Prog V L (V .i32)
  | .laneId => pure c.lane
  | .loopI => pure c.i
  | .ctaId => pure c.cta
  | .lit n => c32 n
  | .ireg _ => pure c.zi
  | .ldIdx b off => do load32 (← at4 (buf c b) (← idx c off))
  | .add a b => do iadd (← idx c a) (← idx c b)
  | .mul a b => do imul (← idx c a) (← idx c b)

def fexp (c : Ctx V) (r : Regs V) : WFExp → Prog V L (V .f32)
  | .reg n => pure (get c r n)
  | .lit v => fconst .f32 (ofF32 v)
  | .add a b => do fadd (← fexp c r a) (← fexp c r b)
  | .mul a b => do fmul (← fexp c r a) (← fexp c r b)
  | .neg a => do fneg (← fexp c r a)
  | .inv a => do
      let x ← fexp c r a
      fdiv (← fconst .f32 oneBits) x
  | .rsqrt a => do
      let s ← sqrt (← fexp c r a)
      fdiv (← fconst .f32 oneBits) s
  | .ex2 a => do callLocal ex2Approx %[← fexp c r a]
  | .exp a => do
      let x ← fexp c r a
      callLocal ex2Approx %[← fmul x (← fconst .f32 log2eBits)]
  | .maxW a b => do
      let x ← fexp c r a
      let y ← fexp c r b
      select (← fcmp .ge x y) x y
  | .geF a b => do
      let x ← fexp c r a
      let y ← fexp c r b
      select (← fcmp .ge x y) (← fconst .f32 (ofF32 1.0)) (← fconst .f32 (ofF32 0.0))

/-- The float registers a statement may write. -/
def written : EWStmt → List Nat
  | .setR r _ | .shflXor r _ _ | .loadIdx r _ _ | .ldSm r _ | .cvtIF r _ => [r]
  | .loadV4 a b c d _ _ => [a, b, c, d]
  | .seq a b | .iif _ a b => (written a ++ written b).eraseDups
  | .forN _ b | .forM _ _ b | .iwhile _ _ _ b | .locals _ b => written b
  | _ => []

/-- The integer registers a statement may change: a loop's, those it carries. -/
def setsI : EWStmt → List Nat
  | .setI d _ | .ldB d _ | .ldH d _ | .ballot d _ | .shflIdx d _ _ => [d]
  | .seq a b | .iif _ a b => (setsI a ++ setsI b).eraseDups
  | .forN n b => if n = 0 then [] else setsI b
  | .forM _ _ b => setsI b
  | .iwhile _ _ cs _ => cs
  | .locals rs b => (setsI b).filter (· ∉ rs)
  | _ => []

/-- Each integer register a statement sets, and whether by a value that is
    `0` or `1` wherever `fl` are. -/
def assigns (fl : List Nat) : EWStmt → List (Nat × Bool)
  | .setI d e => [(d, e.isBool fl)]
  | .ldB d _ | .ldH d _ | .ballot d _ | .shflIdx d _ _ => [(d, false)]
  | .seq a b | .iif _ a b => assigns fl a ++ assigns fl b
  | .forN _ b | .forM _ _ b | .iwhile _ _ _ b | .locals _ b => assigns fl b
  | _ => []

/-- The integer registers held as predicates: of those only set to `0` or `1`,
    the largest set closed under that, each pass dropping a register some
    assignment does not keep so. -/
def flagStep (s : EWStmt) (fl : List Nat) : List Nat :=
  fl.filter fun r => (assigns fl s).all fun (d, b) => d != r || b

/-- Passes of `flagStep` from `fl` until one drops nothing. -/
def flagFix (s : EWStmt) (fl : List Nat) : List Nat :=
  if (flagStep s fl).length < fl.length then flagFix s (flagStep s fl) else fl
termination_by fl.length

def flagRegs (s : EWStmt) : List Nat := flagFix s ((assigns [] s).map (·.1) |>.eraseDups)

/-- `n` trips of `run`, carrying registers `ws`. -/
def loop (c : Ctx V) (r : Regs V) (n : V .i32) (ws : List Nat)
    (run : Ctx V → Regs V → Prog V L (Regs V)) : Prog V L (Regs V) := do
  let pick (r : Regs V) : Vals V (List.replicate ws.length .f32) :=
    Vals.ofFn fun j => get c r (ws.getD j.val 0)
  let back (r : Regs V) (vs : Vals V (List.replicate ws.length .f32)) : Regs V :=
    (ws.zip (Vals.uniformToList vs)).foldl (fun r (k, v) => set r k v) r
  let i0 ← c32 0
  let res ← wloop (tys := .i32 :: List.replicate ws.length .f32) (.cons i0 (pick r))
    (fun cs => pure (contIfULt cs.head n, cs.tail, ()))
    (fun cs _ => do
      let r' ← run { c with i := cs.head } (back r cs.tail)
      pure (.cons (← iadd cs.head (← c32 1)) (pick r')))
  pure (back r res)

/-! ### Integer registers -/

/-- A false predicate. -/
def no : Prog V L (V .i8) := iconst .i8 0

/-- Integer register `n` as a predicate, `n ≠ 0`. -/
def getP (c : Ctx V) (r : Regs V) (n : Nat) : Prog V L (V .i8) := do
  if c.fl.contains n then
    match r.p.lookup n with
    | some x => pure x
    | none => iconst .i8 0
  else
    let v ← match r.v.lookup n with
      | some x => pure x
      | none => iconst64 0
    icmp .ne v (← iconst64 0)

/-- Integer register `n`'s 64-bit value. -/
def getI (c : Ctx V) (r : Regs V) (n : Nat) : Prog V L (V .i64) := do
  if c.fl.contains n then select (← getP c r n) (← iconst64 1) (← iconst64 0)
  else
    match r.v.lookup n with
    | some x => pure x
    | none => iconst64 0

def opOf (o : IOp) (a b : V .i64) : Prog V L (V .i64) :=
  match o with
  | .add => iadd a b | .sub => isub a b | .mul => imul a b
  | .band => band a b | .bor => bor a b | .bxor => bxor a b
  | .shl => ishl a b | .shr => ushr a b | .umin => umin a b

def cmpOf : ICmp → ICmpCond
  | .eq => .eq | .ne => .ne | .ult => .ult | .ule => .ule | .uge => .uge

/-- `k` as a 32-bit constant, its bits those of `k mod 2^32`. -/
def c32w (k : Nat) : Prog V L (V .i32) := iconst .i32 ((k % 2 ^ 32 : Nat) : Int)

/-- Below `2^32` wherever it runs: a register held as a 32-bit word, a literal
    below `2^32`, trailing zeros. -/
def narrow (r : Regs V) : IExp → Bool
  | .reg n => (r.w.lookup n).isSome
  | .lit k => k < 2 ^ 32
  | .ctz _ => true
  | _ => false

mutual
def iexp (c : Ctx V) (r : Regs V) : IExp → Prog V L (V .i64)
  | .reg n => getI c r n
  | .lit n => iconst64 n
  | .lane => uextend64 c.lane
  | .cta => uextend64 c.cta
  | .ptr b => pure (buf c b)
  | .bin o a b => do opOf o (← iexp c r a) (← iexp c r b)
  | .cmp k a b => do
      let p ← if narrow r a && narrow r b then do icmp (cmpOf k) (← low32 c r a) (← low32 c r b)
        else do icmp (cmpOf k) (← iexp c r a) (← iexp c r b)
      select p (← iconst64 1) (← iconst64 0)
  | .ctz a => do uextend64 (← ctz (← low32 c r a))

/-- `e`'s low 32 bits, computed in 32 bits where truncation commutes. -/
def low32 (c : Ctx V) (r : Regs V) : IExp → Prog V L (V .i32)
  | .reg n => match r.w.lookup n with
    | some x => pure x
    | none => do ireduce32 (← getI c r n)
  | .lit k => c32w k
  | .bin .add a b => do iadd (← low32 c r a) (← low32 c r b)
  | .bin .sub a b => do isub (← low32 c r a) (← low32 c r b)
  | .bin .mul a b => do imul (← low32 c r a) (← low32 c r b)
  | .bin .band a b => do band (← low32 c r a) (← low32 c r b)
  | .bin .bor a b => do bor (← low32 c r a) (← low32 c r b)
  | .bin .bxor a b => do bxor (← low32 c r a) (← low32 c r b)
  | .ctz a => do ctz (← low32 c r a)
  | e => do ireduce32 (← iexp c r e)
end

/-- A comparison, of the 32-bit words where both sides are below `2^32`. -/
def icmpE (c : Ctx V) (r : Regs V) (k : ICmp) (a b : IExp) : Prog V L (V .i8) := do
  if narrow r a && narrow r b then icmp (cmpOf k) (← low32 c r a) (← low32 c r b)
  else icmp (cmpOf k) (← iexp c r a) (← iexp c r b)

/-- `e ≠ 0` as a predicate: a comparison's own, the and, or or xor of two
    values `0` or `1` the predicates', a predicate register's. -/
def ipred (c : Ctx V) (r : Regs V) : IExp → Prog V L (V .i8)
  | .cmp k a b => icmpE c r k a b
  | .reg n => getP c r n
  | e@(.bin o a b) =>
      if (o == .band || o == .bor || o == .bxor) && e.isBool c.fl then do
        let x ← ipred c r a
        let y ← ipred c r b
        match o with
        | .band => band x y
        | .bor => bor x y
        | _ => bxor x y
      else do icmp .ne (← iexp c r e) (← iconst64 0)
  | e => do icmp .ne (← iexp c r e) (← iconst64 0)

/-- The test `e ≠ 0` as a comparison of two values: a comparison's own
    operands, else `e`'s predicate against false. -/
def condOf (c : Ctx V) (r : Regs V) : IExp → Prog V L ((ty : ClifTy) × ICmpCond × V ty × V ty)
  | .cmp k a b => do
      if narrow r a && narrow r b then
        let x ← low32 c r a
        let y ← low32 c r b
        pure ⟨.i32, cmpOf k, x, y⟩
      else
        let x ← iexp c r a
        let y ← iexp c r b
        pure ⟨.i64, cmpOf k, x, y⟩
  | e => do
      let p ← ipred c r e
      pure ⟨.i8, .ne, p, ← no⟩

/-- `act` where `g ≠ 0`: always where `g` is the literal `1`. -/
def guarded (c : Ctx V) (r : Regs V) (g : IExp) (act : Prog V L Unit) : Prog V L Unit :=
  match g with
  | .lit 1 => act
  | g => do
      let ⟨_, k, x, y⟩ ← condOf c r g
      when k x y act

/-- The carried registers' types: floats, integer values, predicates. -/
abbrev carryTys (ws rv rf : List Nat) : List ClifTy :=
  ws.map (fun _ => ClifTy.f32) ++ (rv.map (fun _ => ClifTy.i64) ++ rf.map (fun _ => ClifTy.i8))

def appendV {a b : List ClifTy} : Vals V a → Vals V b → Vals V (a ++ b)
  | .nil, ys => ys
  | .cons x xs, ys => .cons x (appendV xs ys)

def splitV {b : List ClifTy} : (a : List ClifTy) → Vals V (a ++ b) → Vals V a × Vals V b
  | [], ys => (.nil, ys)
  | _ :: a, .cons x xs => let (l, r) := splitV a xs; (.cons x l, r)

def packF (c : Ctx V) (r : Regs V) : (ws : List Nat) → Vals V (ws.map fun _ => ClifTy.f32)
  | [] => %[]
  | n :: ws => get c r n ::ᵥ packF c r ws

def packV (c : Ctx V) (r : Regs V) : (rs : List Nat) → Prog V L (Vals V (rs.map fun _ => ClifTy.i64))
  | [] => pure %[]
  | n :: rs => do
      let v ← getI (L := L) c r n
      pure (v ::ᵥ (← packV c r rs))

def packP (c : Ctx V) (r : Regs V) : (rs : List Nat) → Prog V L (Vals V (rs.map fun _ => ClifTy.i8))
  | [] => pure %[]
  | n :: rs => do
      let v ← getP (L := L) c r n
      pure (v ::ᵥ (← packP c r rs))

def bindF (r : Regs V) : (ws : List Nat) → Vals V (ws.map fun _ => ClifTy.f32) → Regs V
  | [], _ => r
  | n :: ws, .cons v vs => bindF (set r n v) ws vs

def bindV (r : Regs V) : (rs : List Nat) → Vals V (rs.map fun _ => ClifTy.i64) → Regs V
  | [], _ => r
  | n :: rs, .cons v vs => bindV (setV r n v) rs vs

def bindP (r : Regs V) : (rs : List Nat) → Vals V (rs.map fun _ => ClifTy.i8) → Regs V
  | [], _ => r
  | n :: rs, .cons v vs => bindP (setP r n v) rs vs

def pack (c : Ctx V) (r : Regs V) (ws rv rf : List Nat) : Prog V L (Vals V (carryTys ws rv rf)) := do
  let a ← packV (L := L) c r rv
  let b ← packP (L := L) c r rf
  pure (appendV (packF c r ws) (appendV a b))

def unpack (r : Regs V) (ws rv rf : List Nat) (vs : Vals V (carryTys ws rv rf)) : Regs V :=
  let (f, iv) := splitV _ vs
  let (a, b) := splitV _ iv
  bindP (bindV (bindF r ws f) rv a) rf b

/-- After a statement with local integer registers `rs`, from `r0` before it:
    those bound as they were. -/
def restoreR (rs : List Nat) (r0 r : Regs V) : Regs V :=
  { r with v := r.v.filter (·.1 ∉ rs) ++ r0.v.filter (·.1 ∈ rs),
           p := r.p.filter (·.1 ∉ rs) ++ r0.p.filter (·.1 ∈ rs),
           w := r.w.filter (·.1 ∉ rs) ++ r0.w.filter (·.1 ∈ rs) }

/-- A statement of register assignments alone. -/
def straight : EWStmt → Bool
  | .skip | .setR .. | .setI .. => true
  | .seq a b => straight a && straight b
  | _ => false

/-- Where a loop's body ends: its carries, its label, and the register
    assignments still to make before going round. -/
structure Tail (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) where
  ws : List Nat
  rv : List Nat
  rf : List Nat
  lp : L (carryTys ws rv rf) (carryTys ws rv rf)
  post : EWStmt

/-- A statement holding no other. -/
def leaf (c : Ctx V) : Regs V → EWStmt → Prog V L (Regs V)
  | r, .setR n e => do pure (set r n (← fexp c r e))
  | r, .shflXor d s m => do
      let o ← callLocal shflBfly %[← bitcast .i32 (get c r s), ← c32 m]
      pure (set r d (← bitcast .f32 o))
  | r, .loadIdx d b ix => do pure (set r d (← loadF32 (← at4 (buf c b) (← idx c ix))))
  | r, .loadV4 d0 d1 d2 d3 b ix => do
      let a ← at4 (buf c b) (← idx c ix)
      let v0 ← loadF32 a
      let v1 ← loadF32 (← iaddImm a 4)
      let v2 ← loadF32 (← iaddImm a 8)
      let v3 ← loadF32 (← iaddImm a 12)
      pure (set (set (set (set r d0 v0) d1 v1) d2 v2) d3 v3)
  | r, .storeLane0 b ix n => do
      when .eq c.lane (← c32 0) do store (get c r n) (← at4 (buf c b) (← idx c ix))
      pure r
  | r, .storeLane b ix n => do
      store (get c r n) (← at4 (buf c b) (← idx c ix))
      pure r
  | r, .stSm ix n => do
      store (get c r n) (← at4 c.sm (← idx c ix))
      pure r
  | r, .ldSm d ix => do pure (set r d (← loadF32 (← at4 c.sm (← idx c ix))))
  | r, .barrier => do
      callLocalVoid barrier %[]
      pure r
  | r, .cvtIF d ix => do pure (set r d (← fcvtFromUint .f32 (← idx c ix)))
  | r, .setI d e => do
      if c.fl.contains d then pure (setP r d (← ipred c r e))
      else match e with
        | .ctz a => do
            let x32 ← ctz (← low32 c r a)
            pure (setW r d (← uextend64 x32) x32)
        | e => pure (setV r d (← iexp c r e))
  | r, .ldB d a => do pure (setV r d (← uload8_64 (← iexp c r a)))
  | r, .stB a v g => do
      guarded c r g do istore8 (← iexp c r v) (← iexp c r a)
      pure r
  | r, .ldH d a => do pure (setV r d (← uload16_64 (← iadd c.sm (← iexp c r a))))
  | r, .stH a v g => do
      guarded c r g do
        let x ← iexp c r v
        istore16 x (← iadd c.sm (← iexp c r a))
      pure r
  | r, .ballot d p => do
      let m ← callLocal ballotR %[← ipred c r p]
      pure (setW r d (← uextend64 m) m)
  | r, .shflIdx d v s => do
      let x ← low32 c r v
      let o ← callLocal shflIdx %[x, ← low32 c r s]
      pure (setW r d (← uextend64 o) o)
  | r, _ => pure r

/-- Straight statements' code. -/
def straightCode (c : Ctx V) : Regs V → EWStmt → Prog V L (Regs V)
  | r, .seq a b => do straightCode c (← straightCode c r a) b
  | r, s => leaf c r s

/-- Going round a loop from its body's end: the tail's assignments, then a jump
    to the loop's head with the carries. -/
def goRound (c : Ctx V) (t : Tail V L) (r : Regs V) : Prog V L (Regs V) := do
  let r ← straightCode c r t.post
  .cont t.lp (← pack c r t.ws t.rv t.rf)

/-- Of integer registers `rs`, those held as values. -/
def vregs (c : Ctx V) (rs : List Nat) : List Nat := rs.filter (!c.fl.contains ·)

/-- Of integer registers `rs`, those held as predicates. -/
def pregs (c : Ctx V) (rs : List Nat) : List Nat := rs.filter (c.fl.contains ·)

/-- **A statement's code**, at the end of a loop's body where `tl` says so:
    there it goes round the loop itself, each arm of a branch it ends with
    making the assignments after it and jumping to the loop's head, with no
    join between them. A branch joins the registers its arms set; a counted
    loop carries the registers its body sets; a `while` loop carries its
    float registers and the integer registers it names. -/
def stmtT (c : Ctx V) : Option (Tail V L) → Regs V → EWStmt → Prog V L (Regs V)
  | none, r, .skip => pure r
  | none, r, .seq a b => do stmtT c none (← stmtT c none r a) b
  | some t, r, .seq a b =>
      if straight b then stmtT c (some { t with post := .seq b t.post }) r a
      else do stmtT c (some t) (← stmtT c none r a) b
  | none, r, .forN n body => do
      if setsI (.forN n body) = [] then loop c r (← c32 n) (written body) fun c' r' => stmtT c' none r' body
      else loopX c r (← c32 n) (written body) (setsI body) fun c' r' => stmtT c' none r' body
  | some t, r, .forN n body => do
      let r' ← if setsI (.forN n body) = [] then loop c r (← c32 n) (written body) fun c' r' => stmtT c' none r' body
        else loopX c r (← c32 n) (written body) (setsI body) fun c' r' => stmtT c' none r' body
      goRound c t r'
  | none, r, .forM b a body => do
      let n ← load32 (← at4 (buf c b) (← c32 a))
      if setsI body = [] then loop c r n (written body) fun c' r' => stmtT c' none r' body
      else loopX c r n (written body) (setsI body) fun c' r' => stmtT c' none r' body
  | some t, r, .forM b a body => do
      let n ← load32 (← at4 (buf c b) (← c32 a))
      let r' ← if setsI body = [] then loop c r n (written body) fun c' r' => stmtT c' none r' body
        else loopX c r n (written body) (setsI body) fun c' r' => stmtT c' none r' body
      goRound c t r'
  | none, r, .iif cnd a b => do
      let ws := (written a ++ written b).eraseDups
      let rv := vregs c (setsI a ++ setsI b).eraseDups
      let rf := pregs c (setsI a ++ setsI b).eraseDups
      let ⟨_, k, x, y⟩ ← condOf c r cnd
      let out ← ifte (jTys := carryTys ws rv rf) k x y
        (do pack c (← stmtT c none r a) ws rv rf)
        (do pack c (← stmtT c none r b) ws rv rf)
      pure (unpack r ws rv rf out)
  | some t, r, .iif cnd a b => do
      let ⟨_, k, x, y⟩ ← condOf c r cnd
      let _ ← ifte (jTys := []) k x y
        (do let _ ← stmtT c (some t) r a; pure %[])
        (do let _ ← stmtT c (some t) r b; pure %[])
      pure r
  | tl, r, .iwhile _ cnd cs body => do
      let ws := written body
      let rv := vregs c cs
      let rf := pregs c cs
      let init ← pack c r ws rv rf
      let out ← wloopL (exitTys := carryTys ws rv rf) init
        (fun _ vs => do
          let ⟨_, k, x, y⟩ ← condOf c (unpack r ws rv rf vs) cnd
          pure (contIf k x y, vs, ()))
        (fun lp vs _ => do
          let _ ← stmtT c (some ⟨ws, rv, rf, lp, .skip⟩) (unpack r ws rv rf vs) body
          pure vs)
      match tl with
      | none => pure (unpack r ws rv rf out)
      | some t => goRound c t (unpack r ws rv rf out)
  | none, r, .locals rs b => do pure (restoreR rs r (← stmtT c none r b))
  | some t, r, .locals rs b => do goRound c t (restoreR rs r (← stmtT c none r b))
  | none, r, s => leaf c r s
  | some t, r, s => do goRound c t (← leaf c r s)
where
  /-- `n` trips of `run`, carrying float registers `ws` and integer registers `rs`. -/
  loopX (c : Ctx V) (r : Regs V) (n : V .i32) (ws rs : List Nat)
      (run : Ctx V → Regs V → Prog V L (Regs V)) : Prog V L (Regs V) := do
    let rv := vregs c rs
    let rf := pregs c rs
    let init ← pack c r ws rv rf
    let i0 ← c32 0
    let res ← wloop (tys := .i32 :: carryTys ws rv rf) (.cons i0 init)
      (fun cs => pure (contIfULt cs.head n, cs.tail, ()))
      (fun cs _ => do
        let r' ← run { c with i := cs.head } (unpack r ws rv rf cs.tail)
        pure (.cons (← iadd cs.head (← c32 1)) (← pack c r' ws rv rf)))
    pure (unpack r ws rv rf res)

def stmt (c : Ctx V) : Regs V → EWStmt → Prog V L (Regs V) := stmtT c none

/-- A warp statement run as lane `lane` of block `cta`, over buffers `bufs`
    and shared memory `sm`. -/
def run (s : EWStmt) (lane cta : V .i32) (bufs : List (V .i64)) (sm : V .i64) : Prog V L Unit := do
  let c : Ctx V := { lane, cta, bufs, sm, i := ← c32 0, zf := ← fconst .f32 0, zi := ← c32 0, z64 := ← iconst64 0, fl := flagRegs s }
  let _ ← stmt c {} s
  pure ()

/-- **A warp statement's body**, over `nbuf` buffers, for blocks of `warps`
    warps that do not meet: warp `w` of block `b` runs the statement as block
    `b·warps + w`, its lanes as the warp's. -/
def body (nbuf : Nat) (s : EWStmt) (warps : Nat := 1) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate nbuf ClifTy.i64)
  let bufs := Vals.uniformToList (ps.drop 4 : Vals V (List.replicate nbuf .i64))
  let (lane, cta) ← if warps ≤ 1 then pure (ps.snd, ps.head) else do
    let lane ← band ps.snd (← c32 31)
    let w ← ushrImm ps.snd 5
    pure (lane, ← iadd (← imul ps.head (← c32 warps)) w)
  run s lane cta bufs (ps.drop 3).head
end

/-- The kernel: `nbuf` pointer parameters, `smem` bytes of shared memory, and
    `warps` warps a block. -/
def kernel (name : String) (nbuf smem : Nat) (s : EWStmt) (warps : Nat := 1) : Kernel :=
  { name, params := List.replicate nbuf .i64,
    code := emit (body nbuf s warps) (idParams ++ List.replicate nbuf .i64), smem }

/-- **A statement's kernel as PTX text**: a module of its own, or the printer's
    error as a comment, which no driver loads. -/
def text (name : String) (nbuf smem : Nat) (s : EWStmt) (warps : Nat := 1) : String :=
  match (kernel name nbuf smem s warps).ptx with
  | .ok t => t
  | .error e => s!"// {e}"

end AlgorithmLib.ML.Body
