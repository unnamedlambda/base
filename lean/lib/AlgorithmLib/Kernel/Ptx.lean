module
public import AlgorithmLib.Host.Term
meta import AlgorithmLib.Host.Term
public import AlgorithmLib.Kernel.PtxIr
meta import AlgorithmLib.Kernel.PtxIr
public import AlgorithmLib.Kernel.Coalesce
meta import AlgorithmLib.Kernel.Coalesce
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Ptx` — a kernel body compiled to PTX

A GPU kernel is a body like any other (`Code`), run once per thread. Its first
four parameters say which thread: `blockIdx.x`, `threadIdx.x`, `blockDim.x`
(each `i32`) and the address of the block's shared memory (`i64`); the rest
are the kernel's own. What one thread does is what `Code` means; what threads
exchange goes through collectives (`Callee.intr`), whose answers
`Kernel.Run` states.

A body is compiled as the host's are, into blocks with parameters
(`compileChecked`: `compileBody`, which `compile_sound` proves keeps its
meaning, on a body that passes what that proof assumes), and the
blocks are written as PTX data (`PtxIr`): each block a label, each value a
register of its type's class, each instruction the PTX instruction that
computes it with the same meaning, each edge a parallel move into the
target's parameters then a branch. Where PTX and the semantics differ it
writes what makes them agree:

* a shift amount is taken modulo the width (PTX clamps), unless it is a
  constant already below it;
* float `add`, `sub`, `mul` and `fma` round to nearest (`.rn`), so `ptxas`
  does not contract them, and `div` and `sqrt` are the exactly rounded ones;
* `fmax` and `fmin` propagate NaN (`.NaN`);
* a float `ne` holds when unordered (`neu`);
* a float negation flips the sign bit (`xor`), as `neg.f32` does not of a NaN,
  and an absolute value clears it (`and`).

Four choices keep the code as fast as hand-written PTX, each local to one
instruction: an address computed from the shared-memory base is held as a
32-bit offset in the shared window and accessed with `ld.shared`/`st.shared`
(unless the body uses one as a value, when every shared address is generic);
a load through a pointer parameter no store goes through takes the read-only
path (`ld.global.nc`); a comparison writes its negation beside its flag,
which a branch on the flag tests: `ptxas` unrolls a loop whose exit is so
tested and not one tested on a negated predicate; and, in a kernel that asks
(`Kernel.mulWide`), a 64-bit product of a value within 32 bits by a constant
within 32 bits is `mul.wide.u32` of the value's low half (`Facts.wideMul`).
That last one is the kernel's to choose because `ptxas` schedules some
kernels faster each way: the fused attention tile keeps its loop control on
the uniform datapath only with `mul.wide`, and the MXFP4 GEMVs are faster
with a 64-bit `mul.lo`.
-/

namespace AlgorithmLib.Kernel

open AlgorithmLib.IR AlgorithmLib.HProg

/-- The parameters every kernel body takes first: block, thread, block size,
    and the address of the block's shared memory. -/
def idParams : List ClifTy := [.i32, .i32, .i32, .i64]

/-- A kernel: its body, the parameters after `idParams`, how many bytes of
    shared memory a block has, whether each block is one warp, its barrier
    then the warp's (`bar.warp.sync`), whether its 64-bit products of a
    value within 32 bits by a constant are `mul.wide.u32` (`Facts.wideMul`),
    and its launch bounds. -/
structure Kernel where
  name : String
  params : List ClifTy
  code : Code
  smem : Nat := 0
  warp : Bool := false
  mulWide : Bool := false
  /-- The entry's launch bounds (`PtxIr.Entry.bounds`). -/
  bounds : Option (Nat × Nat) := none

namespace Ptx

open PtxIr (Line Instr Opnd Reg Lbl)

/-- A register class per type; `i8` is a predicate, which is what the
    comparisons that bind one make. -/
def cls : ClifTy → String
  | .i8 => "p" | .i16 => "h" | .i32 => "r" | .i64 => "rd" | .f32 => "f" | .f64 => "fd" | _ => "v"
/-- A type's register type, and its untyped bits. -/
def bits : ClifTy → PtxIr.Ty
  | .i8 => .pred | .i16 => .b16 | .i32 => .b32 | .i64 => .b64 | .f32 => .f32 | .f64 => .f64
  | _ => .b64
def bTy : Nat → PtxIr.Ty
  | 8 => .b8 | 16 => .b16 | 32 => .b32 | _ => .b64
def uTy : Nat → PtxIr.Ty
  | 8 => .u8 | 16 => .u16 | 32 => .u32 | _ => .u64
def sTy : Nat → PtxIr.Ty
  | 16 => .s16 | 32 => .s32 | _ => .s64

def icc : ICmpCond → PtxIr.Cmp × Bool
  | .eq => (.eq, false) | .ne => (.ne, false)
  | .uge => (.ge, false) | .ugt => (.gt, false) | .ule => (.le, false) | .ult => (.lt, false)
  | .slt => (.lt, true) | .sle => (.le, true) | .sgt => (.gt, true) | .sge => (.ge, true)

def fcc : FloatCC → PtxIr.Cmp
  | .eq => .eq | .ne => .neu | .lt => .lt | .le => .le | .gt => .gt | .ge => .ge

/-- The comparison that holds exactly when `cc` does not. -/
def icneg : ICmpCond → ICmpCond
  | .eq => .ne | .ne => .eq | .uge => .ult | .ult => .uge | .ugt => .ule | .ule => .ugt
  | .sge => .slt | .slt => .sge | .sgt => .sle | .sle => .sgt

/-- The same for floats: the unordered comparisons, true on NaN. -/
def fcneg : FloatCC → PtxIr.Cmp
  | .eq => .neu | .ne => .eq | .lt => .geu | .le => .gtu | .gt => .leu | .ge => .ltu

/-- A float constant's literal: its bits, cut to its width. -/
def lit (ty : ClifTy) (b : UInt64) : Opnd :=
  if ty == .f32 then .f32 (b.toNat % 2 ^ 32) else .f64 b.toNat

/-- A value's register. -/
def regOf (t : ClifTy) (v : Val) : Reg := .val (cls t) v.id

/-- The register holding the negation of a comparison's flag. -/
def negReg (v : Val) : Reg := .val "q" v.id

/-- A shared address's register in the window. -/
def winReg (v : Val) : Reg := .val "rs" v.id

/-- Where an address points: into the block's shared memory, or into the
    buffer kernel parameter `p` points to. -/
inductive Origin where
  | shared
  | global (p : Nat)
  deriving DecidableEq, Inhabited

def Origin.ofSum : Option Origin → Option Origin → Option Origin
  | some .shared, _ | _, some .shared => some .shared
  | some (.global p), _ | _, some (.global p) => some (.global p)
  | _, _ => none

/-- The largest shared offset, in bits, a shared address may span: below the
    span of one region (`regionSpan`), so its sums stay in shared memory's
    region. -/
def sharedReach : Nat := 36

/-- How many bits a value spans that `mag` and `origin` say: a shared
    address's offset from the shared base, any other value's whole bits. -/
def offBits (origin : Std.HashMap Nat Origin) (mag : Std.HashMap Nat Nat) (v : Val) : Nat :=
  (mag.get? v.id).getD 64

/-- How many bits a value spans as a number: a shared address's are all 64,
    its region's base being far above its offset. -/
def valBits (origin : Std.HashMap Nat Origin) (mag : Std.HashMap Nat Nat) (v : Val) : Nat :=
  if origin.get? v.id == some .shared then 64 else offBits origin mag v

/-- The bits an instruction's value spans, from its operands': a constant its
    literal's, a widened value its source's and at most 32 (it widens an 8-,
    16- or 32-bit integer), a shift by a constant that many
    more, an and no more than its narrower operand, a sum one more than its
    wider operand, a shared address's sum its
    offset's and the other operand's, at 64 bits; `none` when the instruction
    bounds nothing. -/
def magRule (origin : Std.HashMap Nat Origin) (mag : Std.HashMap Nat Nat) (lits : Std.HashMap Nat Opnd)
    (tys : Std.HashMap Nat ClifTy) : Inst → Option Nat
  | .iconst _ ty k => some (k % (2 ^ ty.width : Int)).toNat.log2.succ
  | .uextend64 _ a => some (min 32 (valBits origin mag a))
  | .ishl _ a b =>
      let w : Nat := ((tys.get? a.id).map (·.width)).getD 64
      match lits.get? b.id with
      | some (.imm k) => if 0 ≤ k && k < (w : Int) then some (min w (valBits origin mag a + k.toNat)) else none
      | _ => none
  | .band _ a b =>
      if tys.get? a.id == some .i8 then none else some (min (valBits origin mag a) (valBits origin mag b))
  | .iadd d a b =>
      if origin.get? d.id == some .shared then
        if tys.get? d.id != some .i64 then none
        else if origin.get? a.id == some .shared then some (max (offBits origin mag a) (valBits origin mag b) + 1)
        else if origin.get? b.id == some .shared then some (max (offBits origin mag b) (valBits origin mag a) + 1)
        else none
      else some (min 64 (max (valBits origin mag a) (valBits origin mag b) + 1))
  | _ => none

/-- Where each value's address points and how many bits it spans (`magRule`):
    the shared base and the pointer parameters, sums with them (a shared one
    only while its offset spans at most `sharedReach` bits), and block
    parameters every edge into the block passes one buffer's address. -/
def addrFacts (f : FuncData) (k0 : List ClifTy) (tys : Std.HashMap Nat ClifTy) (lits : Std.HashMap Nat Opnd) :
    Std.HashMap Nat Origin × Std.HashMap Nat Nat := Id.run do
  let mut o : Std.HashMap Nat Origin := {}
  let mut m : Std.HashMap Nat Nat := {}
  if let some b0 := f.blocks.head? then
    for ((v, _), i) in b0.params.zipIdx do
      if i == 3 then o := o.insert v.id .shared; m := m.insert v.id 0
      else if i ≥ 4 && k0[i - 4]? == some ClifTy.i64 then o := o.insert v.id (.global (i - 4))
  let entry := o
  let insts := f.blocks.flatMap (·.insts)
  -- what each block parameter is passed, as the values themselves
  let mut incoming : Std.HashMap Nat (List Val) := {}
  for i in insts do
    let edges : List (BlockRef × List Val) := match i with
      | .jump t as => [(t, as)]
      | .brif _ t as e as' => [(t, as), (e, as')]
      | _ => []
    for (t, as) in edges do
      if let some tb := f.blocks.find? (·.ref.id == t.id) then
        for ((v, _), a) in tb.params.zip as do
          incoming := incoming.insert v.id (a :: (incoming.getD v.id []))
  -- A loop's parameter is passed its own successor, so a fixed point grown
  -- from the entry never reaches it. Assume each parameter holds the buffer
  -- every incoming value of known origin points into, then drop any
  -- assumption an incoming value fails, until none does: what is left is
  -- consistent, every parameter passed that buffer's addresses on every edge.
  let grow (seed : Std.HashMap Nat Origin) (strict : Bool) : Std.HashMap Nat Origin := Id.run do
    let mut o := seed
    for _ in [0:8] do
      let mut n := seed
      for i in insts do
        if let .iadd d a b := i then
          if let some x := Origin.ofSum (o.get? a.id) (o.get? b.id) then
            match x with
            | .shared => pure ()
            | g => n := n.insert d.id g
      for (v, as) in incoming.toList do
        let os := as.map (o.get? ·.id)
        match os.filterMap id with
        | .global p :: rest =>
            if rest.all (· == .global p) && (!strict || os.all (·.isSome)) then n := n.insert v (.global p)
        | _ => pure ()
      o := n
    return o
  let opt := grow entry false
  let mut assumed : Std.HashMap Nat Origin := {}
  for (v, _) in incoming.toList do
    if let some g@(.global _) := opt.get? v then assumed := assumed.insert v g
  for _ in [0:8] do
    let o' := grow (assumed.fold (fun s k x => s.insert k x) entry) true
    assumed := assumed.filter fun v g => (incoming.getD v []).all (o'.get? ·.id == some g)
  for (v, g) in assumed.toList do o := o.insert v g
  let entry := o
  for _ in [0:8] do
    let mut n := entry
    for i in insts do
      if let .iadd d a b := i then
        if let some x := Origin.ofSum (o.get? a.id) (o.get? b.id) then
          match x with
          | .shared =>
              let s := if o.get? a.id == some .shared then a else b
              let x := if o.get? a.id == some .shared then b else a
              if max (offBits o m s) (valBits o m x) + 1 ≤ sharedReach then n := n.insert d.id .shared
          | g => n := n.insert d.id g
    -- what each block parameter is passed, edge by edge
    let mut passed : Std.HashMap Nat (List (Option Origin)) := {}
    for i in insts do
      let edges : List (BlockRef × List Val) := match i with
        | .jump t args => [(t, args)]
        | .brif _ t args e args' => [(t, args), (e, args')]
        | _ => []
      for (t, args) in edges do
        if let some tb := f.blocks.find? (·.ref.id == t.id) then
          for ((v, _), a) in tb.params.zip args do
            passed := passed.insert v.id ((o.get? a.id) :: (passed.getD v.id []))
    for (v, xs) in passed.toList do
      match xs with
      | some (.global p) :: rest => if rest.all (· == some (.global p)) then n := n.insert v (.global p)
      | _ => pure ()
    o := n
    for i in insts do
      if let some d := Inst.dst? i then
        if let some k := magRule o m lits tys i then m := m.insert d.id k
  return (o, m)

/-- What the whole body tells each instruction: each value's type, origin,
    literal, defining instruction, which pointer parameters are
    read-only, and whether shared addresses are held in the window. -/
structure Facts where
  blocks : List BlockData
  tys : Std.HashMap Nat ClifTy
  origin : Std.HashMap Nat Origin
  /-- How many bits each value spans (`magRule`), a shared address its offset. -/
  mag : Std.HashMap Nat Nat
  /-- Values a constant bound: the literal PTX writes for each. -/
  lits : Std.HashMap Nat Opnd
  defs : Std.HashMap Nat Inst
  /-- Pointer parameters a load through may take the read-only path:
      no store goes through them, nor through an address of no known origin. -/
  ro : List Nat
  /-- Whether shared addresses are held as 32-bit offsets in the shared
      window (`%rs`): unless the body uses one as a value. -/
  window : Bool
  /-- Whether each block is one warp: a barrier is then the warp's. -/
  warp : Bool := false
  /-- Whether a product `Facts.wideMul` admits is `mul.wide.u32`. -/
  mulWide : Bool := false

/-- Whether `v` is a shared address held in the window. -/
def Facts.win (F : Facts) (v : Val) : Bool := F.window && F.origin.get? v.id == some .shared

/-- Whether a comparison bound `v` whose negation `negReg` holds. -/
def Facts.negated (F : Facts) (v : Val) : Bool :=
  match F.defs.get? v.id with
  | some (.icmp _ _ a _) => F.tys.get? a.id != some .i8
  | some (.fcmp ..) => true
  | _ => false

/-- Whether a 64-bit product `a * b` of a kernel that asks for it
    (`Kernel.mulWide`) is a value within 32 bits by a constant within 32 bits:
    it is then `mul.wide.u32` of `a`'s low half and the constant, which cannot
    overflow. -/
def Facts.wideMul (F : Facts) (a b : Val) : Bool :=
  F.mulWide && F.tys.get? a.id == some .i64 && F.lits.get? a.id == none &&
    valBits F.origin F.mag a ≤ 32 && valBits F.origin F.mag b ≤ 32 &&
    match F.lits.get? b.id with
    | some (.imm _) => true
    | _ => false

/-- The type of what an atomic answers, none for a store or a fence. -/
def atomicRes : Atomic → Option ClifTy
  | .load t _ | .rmw t _ _ | .cas t => some t
  | .store .. | .fence => none

/-- The value an instruction defines and its type, from its operands'. -/
def resTy (ty : Val → Option ClifTy) : Inst → Option (Val × ClifTy)
  | .iconst d t _ | .fconst d t _ | .bitcast d t _ | .fcvtFromSint d t _ | .fconv d _ t _ | .iext d _ t _ =>
      some (d, t)
  | .iadd d a _ | .isub d a _ | .imul d a _ | .udiv d a _ | .ibin d _ a _ | .band d a _ | .bor d a _
  | .bxor d a _
  | .fadd d a _ | .fsub d a _ | .fmul d a _ | .fmax d a _ | .fmin d a _ | .fbin d _ a _
  | .ishl d a _ | .ushr d a _ | .ishift d _ a _ | .select d _ a _ | .fneg d a | .fun1 d _ a
  | .fma d a _ _ | .ctz d a | .popcnt d a => (ty a).map (d, ·)
  | .uextend64 d _ | .sextend64 d _ => some (d, .i64)
  | .ireduce32 d _ => some (d, .i32)
  | .icmp d .. | .fcmp d .. => some (d, .i8)
  | .load d lo _ => some (d, lo.ty)
  | .call (some d) (.intr c) _ => c.sig.2.map (d, ·)
  | .call (some d) (.atomic a) _ => (atomicRes a).map (d, ·)
  | _ => none

/-- The facts of a body's blocks, `k0` the kernel's own parameters. -/
def facts (f : FuncData) (k0 : List ClifTy) (window : Bool) : Facts := Id.run do
  let insts := f.blocks.flatMap (·.insts)
  let mut tys : Std.HashMap Nat ClifTy := {}
  for b in f.blocks do
    for (v, t) in b.params do tys := tys.insert v.id t
  for i in insts do
    if let some (d, t) := resTy (tys.get? ·.id) i then tys := tys.insert d.id t
  let mut lits : Std.HashMap Nat Opnd := {}
  let mut defs : Std.HashMap Nat Inst := {}
  for i in insts do
    if let some (d, _) := resTy (tys.get? ·.id) i then defs := defs.insert d.id i
    match i with
    | .iconst d ty k =>
        if ty != .i8 then lits := lits.insert d.id (.imm (k % (2 ^ ty.width : Int)).toNat)
    | .fconst d ty b => lits := lits.insert d.id (lit ty b)
    -- a widened constant is a constant: its literal is already zero above
    -- 32 bits
    | .uextend64 d a =>
        if tys.get? a.id == some .i32 then
          if let some l := lits.get? a.id then lits := lits.insert d.id l
    | _ => pure ()
  let (origin, mag) := addrFacts f k0 tys lits
  let stores := insts.filterMap fun
    | .store _ a | .storeTyped _ _ a | .istore8 _ a => some (origin.get? a.id)
    | _ => none
  let ro := if stores.any (·.isNone) then [] else
    (List.range k0.length).filter fun p => !stores.any (· == some (.global p))
  return { blocks := f.blocks, tys, origin, mag, lits, defs, ro, window }

/-- What lowering one instruction gives: its lines, the registers of its own
    it declares, and what it could not take. -/
structure Out where
  lines : Array Line := #[]
  regs : Array (Reg × ClifTy) := #[]
  n : Nat := 0
  err : Option String := none

/-- Lowering one instruction, at position `tag`: its own registers are named
    after the position. -/
abbrev LM := ReaderT (Facts × Nat) (StateM Out)

def emitL (l : Line) : LM Unit := modify fun s => { s with lines := s.lines.push l }
def emit (i : Instr) : LM Unit := emitL (.ins none i)
def fail (m : String) : LM Unit := modify fun s => if s.err.isNone then { s with err := some m } else s
/-- A register of this instruction's own. -/
def own (p : String) (t : ClifTy) : LM Reg := do
  let tag := (← read).2
  let s ← get
  let nm := Reg.own p tag s.n
  set { s with n := s.n + 1, regs := s.regs.push (nm, t) }
  pure nm

def facts? : LM Facts := do pure (← read).1
def tyOf (v : Val) : LM ClifTy := do
  match (← facts?).tys.get? v.id with
  | some t => pure t
  | none => fail s!"v{v.id} has no type"; pure .i32

/-- Whether `v` is a shared address held in the window. -/
def isWin (v : Val) : LM Bool := do pure ((← facts?).win v)

/-- `v`'s register as an address, a sum's operand or an edge's argument. -/
def rw (v : Val) : LM Reg := do
  if ← isWin v then pure (winReg v) else pure (regOf (← tyOf v) v)

/-- `v`'s register as a value: a constant is put in a register of its own
    where it is used. -/
def r (v : Val) : LM Reg := do
  match (← facts?).lits.get? v.id with
  | some l =>
      let k ← own "k" (← tyOf v)
      emit (.mov (bits (← tyOf v)) k l); pure k
  | none => rw v
def o (v : Val) : LM Opnd := do pure (.reg (← r v))

/-- `v` as an operand that takes an immediate: a constant's literal. -/
def imm (v : Val) : LM Opnd := do
  match (← facts?).lits.get? v.id with
  | some l => pure l
  | none => o v

/-- A shift amount that is a constant below the width. -/
def shiftLit (F : Facts) (ta : ClifTy) (b : Val) : Option Int :=
  match F.lits.get? b.id with
  | some (.imm k) => if 0 ≤ k && k < ta.width then some k else none
  | _ => none

/-- A shift amount as PTX takes it: masked to the width unless a constant
    below it. -/
def shiftAmt (ta : ClifTy) (b : Val) : LM Opnd := do
  if let some k := shiftLit (← facts?) ta b then return .imm k
  let a ← own "sa" .i32
  if (← tyOf b) == .i64 then emit (.cvt none .u32 .u64 a (← o b)) else emit (.mov .b32 a (← o b))
  emit (.bin .and none .b32 a (.reg a) (.imm (ta.width - 1)))
  pure (.reg a)

/-- The state space and address an access through `a` takes: a shared
    address as its offset in the shared window, a buffer's as global. -/
def addr (a : Val) : LM (PtxIr.Space × PtxIr.Addr) := do
  if ← isWin a then return (.shared, { base := .reg (← rw a) })
  match (← facts?).origin.get? a.id with
  | some .shared =>
      let w ← own "sw" .i64
      emit (.cvta true .shared .u64 w (← o a))
      pure (.shared, { base := .reg w })
  | some (.global _) => pure (.global, { base := ← o a })
  | none => pure (.generic, { base := ← o a })

def approxOp : Intr → Option PtxIr.UnOp
  | .ex2Approx => some .ex2Approx | .rcpApprox => some .rcpApprox | .lg2Approx => some .lg2Approx
  | .sinApprox => some .sinApprox | .cosApprox => some .cosApprox | .sqrtApprox => some .sqrtApprox
  | .rsqrtApprox => some .rsqrtApprox
  | _ => none

/-- A sum of a shared address `sa` in the window and `other`: the offset
    fits 32 bits, so it is summed in 32. -/
def iaddWin (d sa other : Val) : LM Unit := do
  let o32 : Opnd ← match (← facts?).lits.get? other.id with
    | some l => pure l
    | none =>
        if (← tyOf other) == .i64 then do
          let t ← own "so" .i32
          emit (.cvt none .u32 .u64 t (← o other)); pure (.reg t)
        else o other
  emit (.bin .add none .u32 (winReg d) (.reg (← rw sa)) o32)

/-- The `atom` of a CLIF read-modify-write; none for one PTX has not
    (`sub`, `nand`). -/
def atomOp : AtomicRmw → Option PtxIr.AtomOp
  | .add => some .add | .and => some .and | .or => some .or | .xor => some .xor | .xchg => some .exch
  | .umin => some .umin | .umax => some .umax | .smin => some .smin | .smax => some .smax
  | .sub | .nand => none

/-- The type an `atom` of `o` names at the width of `t`. -/
def atomPty (o : PtxIr.AtomOp) (t : ClifTy) : PtxIr.Ty :=
  let w64 := t.width == 64
  match o with
  | .add | .umin | .umax => if w64 then .u64 else .u32
  | .smin | .smax => if w64 then .s64 else .s32
  | _ => if w64 then .b64 else .b32

/-- An atomic: its address's lines, its operand's, then one `atom`. -/
def atomic : Option Val → Atomic → List Val → LM Unit
  | some d, .rmw t k ord, [p, x] =>
      match atomOp k with
      | some op => do
          let (sp, ad) ← addr p
          emit (.atom sp op (ord == .seqCst) (atomPty op t) (some (regOf t d)) ad (← o x))
      | none => fail s!"atomic_rmw {k.clif}, which PTX has not"
  | some d, .load t ord, [p] => do
      let (sp, ad) ← addr p
      emit (.atom sp .load (ord == .seqCst) (atomPty .load t) (some (regOf t d)) ad (.imm 0))
  | none, .store t ord, [v, p] => do
      let (sp, ad) ← addr p
      emit (.atom sp .store (ord == .seqCst) (atomPty .store t) none ad (← o v))
  | _, _, _ => fail "an atomic the compiler does not take"

/-- An intrinsic that answers nothing: a barrier, or a shared add, its
    address's lines, its operand's, then one `red`. -/
def voidIntr : Intr → List Val → LM Unit
  | .barrier, [] => do emit (if (← facts?).warp then .barWarp (.imm 0xffffffff) else .bar 0)
  | .sharedAdd, [p, x] => do
      let (_, ad) ← addr p
      emit (.redShared ad (← o x))
  | c, _ => fail s!"{c.name} without a result, on arguments it does not take"

/-- The intrinsics of three words and a word: `dp4a`, `prmt`. -/
def word3 : Intr → Bool
  | .dp4a | .prmt => true
  | _ => false

/-- A three-argument intrinsic with a word result, into `d`: `dp4a` or
    `prmt`. -/
def intr3 (d : Val) (c : Intr) (a b e : Val) : LM Unit :=
  match c with
  | .dp4a => do emit (.dp4a (regOf .i32 d) (← o a) (← o b) (← o e))
  | .prmt => do emit (.prmt (regOf .i32 d) (← o a) (← o b) (← o e))
  | c => fail s!"{c.name} with three arguments"

/-- The intrinsics of six words and a 64-bit answer: the wide `mma`. -/
def word6 : Intr → Bool
  | .mmaLo | .mmaHi => true
  | _ => false

/-- A six-argument intrinsic, into `d`: a wide `mma`'s half. -/
def intr6 (d : Val) (c : Intr) (a0 a1 a2 a3 b0 b1 : Val) : LM Unit :=
  match c with
  | .mmaLo => do emit (.mmaW false (regOf .i64 d) (← o a0) (← o a1) (← o a2) (← o a3) (← o b0) (← o b1))
  | .mmaHi => do emit (.mmaW true (regOf .i64 d) (← o a0) (← o a1) (← o a2) (← o a3) (← o b0) (← o b1))
  | c => fail s!"{c.name} with six arguments"

/-- The intrinsics of six words, a 64-bit accumulator and a 64-bit answer:
    the bf16 `mma`. -/
def word7 : Intr → Bool
  | .mmaBfLo | .mmaBfHi => true
  | _ => false

/-- A seven-argument intrinsic, into `d`: a bf16 `mma`'s half. -/
def intr7 (d : Val) (c : Intr) (a0 a1 a2 a3 b0 b1 e : Val) : LM Unit :=
  match c with
  | .mmaBfLo => do
      emit (.mmaB false (regOf .i64 d) (← o a0) (← o a1) (← o a2) (← o a3) (← o b0) (← o b1) (← o e))
  | .mmaBfHi => do
      emit (.mmaB true (regOf .i64 d) (← o a0) (← o a1) (← o a2) (← o a3) (← o b0) (← o b1) (← o e))
  | c => fail s!"{c.name} with seven arguments"

/-- An intrinsic of three words or more, into `d`: three (`intr3`), six
    (`intr6`) or seven (`intr7`). -/
def intrN (d : Val) (c : Intr) (a b e : Val) : List Val → LM Unit
  | [] => intr3 d c a b e
  | [a3, b0, b1] => intr6 d c a b e a3 b0 b1
  | [a3, b0, b1, x] => intr7 d c a b e a3 b0 b1 x
  | _ => fail s!"{c.name} on arguments it does not take"

/-- One instruction that is not a terminator. -/
def inst (i : Inst) : LM Unit := do
  let bin2 (d a b : Val) (bo : PtxIr.BinOp) (rnd : Option PtxIr.Rnd) (ty : ClifTy → PtxIr.Ty) : LM Unit := do
    let t ← tyOf a
    let x ← o a
    emit (.bin bo rnd (ty t) (regOf t d) x (← imm b))
  match i with
  | .iconst d ty k =>
      if ty == .i8 then
        let t ← own "c" .i32
        emit (.mov .u32 t (.imm (k % 256)))
        emit (.setp .ne .u32 (regOf ty d) (.reg t) (.imm 0))
  | .fconst .. => pure ()
  | .iadd d a b => do
      if ← isWin a then iaddWin d a b
      else if ← isWin b then iaddWin d b a
      else bin2 d a b .add none (uTy ·.width)
  | .isub d a b => bin2 d a b .sub none (uTy ·.width)
  | .imul d a b => do
      if (← facts?).wideMul a b then
        let x ← r a
        let w ← own "w" .i32
        emit (.cvt none .u32 .u64 w (.reg x))
        let y ← imm b
        emit (.bin .mulWide none .u32 (regOf .i64 d) (.reg w) y)
      else bin2 d a b .mulLo none (uTy ·.width)
  | .udiv d a b => bin2 d a b .div none (uTy ·.width)
  | .ibin d .umin a b => bin2 d a b .min none (uTy ·.width)
  | .ibin d .umax a b => bin2 d a b .max none (uTy ·.width)
  | .iext d .reduce .i16 a => do
      let t ← tyOf a; emit (.cvt none .u16 (uTy t.width) (regOf .i16 d) (← o a))
  | .band d a b => bin2 d a b .and none (fun t => if t == .i8 then .pred else bTy t.width)
  | .bor d a b => bin2 d a b .or none (fun t => if t == .i8 then .pred else bTy t.width)
  | .bxor d a b => bin2 d a b .xor none (fun t => if t == .i8 then .pred else bTy t.width)
  | .fadd d a b => bin2 d a b .add (some .rn) bits
  | .fsub d a b => bin2 d a b .sub (some .rn) bits
  | .fmul d a b => bin2 d a b .mul (some .rn) bits
  | .fmax d a b => bin2 d a b .maxNaN none bits
  | .fmin d a b => bin2 d a b .minNaN none bits
  | .fbin d .fdiv a b => bin2 d a b .div (some .rn) bits
  | .ishl d a b | .ushr d a b | .ishift d .sshr a b => do
      let t ← tyOf a
      let amt ← shiftAmt t b
      let (bo, ty) : PtxIr.BinOp × PtxIr.Ty := match i with
        | .ishl .. => (.shl, bTy t.width) | .ushr .. => (.shr, uTy t.width) | _ => (.shr, sTy t.width)
      emit (.bin bo none ty (regOf t d) (← o a) amt)
  | .uextend64 d a => do
      let t ← tyOf a
      -- a widened constant is a literal, written where it is used
      if ((← facts?).lits.get? d.id).isSome then pure ()
      else if t == .i8 then emit (.selp .u64 (regOf .i64 d) (.imm 1) (.imm 0) (← o a))
      else emit (.cvt none .u64 (uTy t.width) (regOf .i64 d) (← o a))
  | .sextend64 d a => do
      let t ← tyOf a; emit (.cvt none .s64 (sTy t.width) (regOf .i64 d) (← o a))
  | .ireduce32 d a => do emit (.cvt none .u32 .u64 (regOf .i32 d) (← o a))
  | .icmp d cc a b => do
      let t ← tyOf a
      if t == .i8 then
        -- flags differ exactly when their exclusive or holds
        let x ← o a; let y ← o b
        match cc with
        | .ne => emit (.bin .xor none .pred (regOf .i8 d) x y)
        | .eq =>
            emit (.bin .xor none .pred (regOf .i8 d) x y)
            emit (.un .not none .pred (regOf .i8 d) (.reg (regOf .i8 d)))
        | _ => fail "flags compared other than for equality"
      else
        let (c, s) := icc cc
        let x ← o a; let y ← imm b
        let ty := if s then sTy t.width else uTy t.width
        emit (.setp c ty (regOf .i8 d) x y)
        -- its negation beside it, which a branch on the flag tests: ptxas
        -- unrolls a loop whose exit is so tested and not one tested on `@!p`
        emit (.setp (icc (icneg cc)).1 ty (negReg d) x y)
  | .fcmp d cc a b => do
      let t ← tyOf a
      let x ← o a; let y ← imm b
      emit (.setp (fcc cc) (bits t) (regOf .i8 d) x y)
      emit (.setp (fcneg cc) (bits t) (negReg d) x y)
  | .select d c a b => do
      let t ← tyOf a
      let x ← imm a; let y ← imm b
      emit (.selp (bits t) (regOf t d) x y (← o c))
  -- a sign flip, a NaN's included, which `neg.f32` does not keep
  | .fneg d a => do
      let t ← tyOf a
      emit (.bin .xor none (bTy t.width) (regOf t d) (← o a) (.imm (2 ^ (t.width - 1))))
  | .fun1 d .sqrt a => do let t ← tyOf a; emit (.un .sqrt (some .rn) (bits t) (regOf t d) (← o a))
  | .fun1 d .fabs a => do
      let t ← tyOf a
      emit (.bin .and none (bTy t.width) (regOf t d) (← o a) (.imm (2 ^ (t.width - 1) - 1)))
  | .fcvtFromSint d ty a => do
      let t ← tyOf a; emit (.cvt (some .rn) (bits ty) (sTy t.width) (regOf ty d) (← o a))
  | .fconv d .fromUint ty a => do
      let t ← tyOf a; emit (.cvt (some .rn) (bits ty) (uTy t.width) (regOf ty d) (← o a))
  | .fconv d .demote ty a => do
      emit (.cvt (some .rn) (bits ty) .f64 (regOf ty d) (← o a))
  | .fconv d .toSint ty a => do
      emit (.cvt (some .rzi) (sTy ty.width) .f32 (regOf ty d) (← o a))
  | .fma d a b c => do
      let t ← tyOf a
      let x ← o a; let y ← imm b
      emit (.fma (some .rn) (bits t) (regOf t d) x y (← imm c))
  | .ctz d a => do
      let t ← tyOf a
      let rb ← own "bv" t
      emit (.un .brev none (bTy t.width) rb (← o a))
      if t.width == 64 then
        let q ← own "cz" .i32
        emit (.un .clz none .b64 q (.reg rb))
        emit (.cvt none .u64 .u32 (regOf t d) (.reg q))
      else emit (.un .clz none .b32 (regOf t d) (.reg rb))
  | .popcnt d a => do emit (.un .popc none .b32 (regOf .i32 d) (← o a))
  | .bitcast d ty a => do emit (.mov (bTy ty.width) (regOf ty d) (← o a))
  | .load d lo a => do
      let pt : Option PtxIr.Ty := match lo.kind with
        | .plain => some (bits lo.ty) | .uload8 => some .u8 | .uload16 => some .u16
        | .uload32 => some .u32 | _ => none
      match pt with
      | some pt =>
          let (sp, ad) ← addr a
          let F ← facts?
          let nc := match F.origin.get? a.id with
            | some (.global p) => lo.kind == .plain && F.ro.contains p
            | _ => false
          emit (.ld sp nc pt (regOf lo.ty d) ad)
      | none => fail "a load kind the compiler does not take"
  | .storeTyped ty v a => do let (sp, ad) ← addr a; emit (.st sp (bits ty) ad (← o v))
  | .store v a => do let t ← tyOf v; let (sp, ad) ← addr a; emit (.st sp (bits t) ad (← o v))
  | .istore8 v a => do let (sp, ad) ← addr a; emit (.st sp .u8 ad (← o v))
  | .call (some d) (.intr .shflBfly) [v, m] => do
      let x ← o v; emit (.shfl .bfly (regOf .i32 d) x (← o m) (.imm 31) (.imm 0xffffffff))
  | .call (some d) (.intr .shflIdx) [v, m] => do
      let x ← o v; emit (.shfl .idx (regOf .i32 d) x (← o m) (.imm 31) (.imm 0xffffffff))
  | .call (some d) (.intr .ballot) [v] => do emit (.vote (regOf .i32 d) (← o v) (.imm 0xffffffff))
  | .call (some d) (.intr c) (a :: b :: e :: rest) => intrN d c a b e rest
  | .call none (.intr c) args => voidIntr c args
  | .call d (.atomic a) args => atomic d a args
  | .call (some d) (.intr c) [v] =>
      match approxOp c with
      | some u => do emit (.un u none .f32 (regOf .f32 d) (← o v))
      | none => fail s!"{c.name} with one argument"
  | _ => fail "an instruction the compiler does not take"

/-- The one instruction that puts value `a`, an edge's argument, into
    register `tmp` of type `t`: from its register, a windowed shared address
    from its window register, a constant from its literal. -/
def moveInstr (F : Facts) (tmp : Reg) (t : ClifTy) (a : Val) : Instr :=
  if F.win a then .mov .b32 tmp (.reg (winReg a))
  else match F.lits.get? a.id with
    | some l => .mov (bits t) tmp l
    | none => .mov (bits t) tmp (.reg (regOf ((F.tys.get? a.id).getD .i32) a))

/-- `moveInstr` as a line of the lowering. -/
def moveIn (tmp : Reg) (t : ClifTy) (a : Val) : LM Unit := do
  emit (moveInstr (← facts?) tmp t a)

/-- Each argument put into a register of the instruction's own; the
    registers, in order. -/
def copyOut : List ((Reg × ClifTy × Bool) × Val) → LM (List Reg)
  | [] => pure []
  | ((_, t, _), a) :: ps => do
      let tmp ← own "t" t
      moveIn tmp t a
      return tmp :: (← copyOut ps)

/-- Each copy moved into its destination. -/
def copyIn : List ((Reg × ClifTy × Bool) × Val) → List Reg → LM Unit
  | ((d, t, _), _) :: ps, tmp :: tmps => do
      emit (.mov (bits t) d (.reg tmp))
      copyIn ps tmps
  | _, _ => pure ()

def blkLbl (b : BlockRef) : Lbl := .blk b.id

/-- Where an edge's arguments go: each parameter's register, its type and
    whether it is a windowed shared address. -/
def dstsOf (F : Facts) (params : List (Val × ClifTy)) : List (Reg × ClifTy × Bool) :=
  params.map fun (v, t) => if F.win v then (winReg v, .i32, true) else (regOf t v, t, false)

/-- Moves into block `b`'s parameters then a branch there. -/
def edge (b : BlockRef) (args : List Val) : LM Unit := do
  match (← facts?).blocks.find? (·.ref.id == b.id) with
  | none => fail s!"block{b.id} does not exist"
  | some bd =>
      -- every argument is read before any parameter is written
      let ps := (dstsOf (← facts?) bd.params).zip args
      copyIn ps (← copyOut ps)
      emit (.bra true (blkLbl b))

def term : Inst → LM Unit
  | .jump t args => edge t args
  | .brif c tb ta eb ea => do
      let p ← r c
      let tag := (← read).2
      if (← facts?).negated c then emitL (.ins (some (false, negReg c)) (.bra false (.els tag)))
      else emitL (.ins (some (true, p)) (.bra false (.els tag)))
      edge tb ta
      emitL (.label (.els tag))
      edge eb ea
  | .ret none => emit .ret
  | _ => fail "a terminator the compiler does not take"

def isTerm : Inst → Bool
  | .jump .. | .brif .. | .ret .. => true
  | _ => false

/-- **One instruction's lowering**, at position `tag`: a function of the
    body's facts, the position and the instruction alone. -/
def lower (F : Facts) (tag : Nat) (i : Inst) : Out :=
  (((if isTerm i then term i else inst i).run (F, tag)).run {}).2

/-- The entry's own first lines: the thread's identity and the shared base
    into the entry block's first parameters, then its own parameters. -/
def prologueInstrs (F : Facts) (k0 : List ClifTy) (b0 : BlockData) : List Instr :=
  let pr := fun i => match b0.params[i]? with
    | some (v, t) => regOf t v
    | none => .name "none"
  let p3 : Reg := match b0.params[3]? with
    | some (v, _) => winReg v
    | none => .name "none"
  [.mov .u32 (pr 0) (.reg (.special "ctaid.x")), .mov .u32 (pr 1) (.reg (.special "tid.x")),
    .mov .u32 (pr 2) (.reg (.special "ntid.x"))] ++
  [if F.window then .mov .u32 p3 (.sym "smem") else .cvta false .shared .u64 (pr 3) (.sym "smem")] ++
  k0.zipIdx.map fun (t, i) => .ld .param false (bits t) (pr (i + 4)) { base := .sym s!"p{i}" }

def prologue (F : Facts) (k0 : List ClifTy) (b0 : BlockData) : Out :=
  { lines := ((prologueInstrs F k0 b0).map (.ins none)).toArray }

/-- Instructions' lowerings, each at its position, from `tag` on. -/
def instsBody (F : Facts) : Nat → List Inst → List Out
  | _, [] => []
  | tag, i :: is => lower F tag i :: instsBody F (tag + 1) is

/-- Blocks' labels and their instructions' lowerings, the first instruction
    after `start` others. -/
def blocksBody (F : Facts) : Nat → List BlockData → List Out
  | _, [] => []
  | start, b :: bs =>
      { lines := #[.label (blkLbl b.ref)] } :: instsBody F (start + 1) b.insts ++
        blocksBody F (start + b.insts.length) bs

/-- The body's lines: the prologue, then each block's label and its
    instructions' lowerings, each at its position in the body. -/
def body (F : Facts) (k0 : List ClifTy) : List Out :=
  match F.blocks with
  | [] => [{ err := some "no entry block" }]
  | b0 :: _ => prologue F k0 b0 :: blocksBody F 0 F.blocks

end Ptx

/-! ## Defined before use

What every run of a function has defined on entry to each block is found by
the usual fixpoint (`definedIn`), and checked (`definedOk`): each instruction
reads only values defined on entry, as the block's parameters, or by an
instruction before it, and what an edge's target counts on is defined at the
edge. A kernel that fails it is refused; none does, the blocks being
`compileBody`'s, but the proof that the PTX does what the blocks do rests on
the check rather than on how the blocks were made. -/

/-- The values an instruction reads. -/
def Inst.uses : Inst → List Val
  | .iconst .. | .fconst .. | .memory _ | .funcAddr .. => []
  | .iadd _ a b | .isub _ a b | .imul _ a b | .udiv _ a b | .ishl _ a b | .ushr _ a b
  | .band _ a b | .bandNot _ a b | .bor _ a b | .bxor _ a b | .fadd _ a b | .fsub _ a b
  | .fmul _ a b | .fmax _ a b | .fmin _ a b | .icmp _ _ a b | .fcmp _ _ a b
  | .ibin _ _ a b | .ishift _ _ a b | .fbin _ _ a b => [a, b]
  | .ineg _ a | .ireduce32 _ a | .uextend64 _ a | .sextend64 _ a | .fpromote _ a | .fneg _ a
  | .ctz _ a | .popcnt _ a | .vhighBits _ a | .splat _ _ a | .bitcast _ _ a | .fcvtFromSint _ _ a
  | .fcvtToUint _ _ a | .extractlane _ a _ | .load _ _ a | .iun _ _ a | .fun1 _ _ a
  | .fconv _ _ _ a | .iext _ _ _ a => [a]
  | .select _ c a b | .bitselect _ c a b => [c, a, b]
  | .fma _ a b c => [a, b, c]
  | .store v a | .istore8 v a | .storeTyped _ v a => [v, a]
  | .call _ _ args | .jump _ args => args
  | .brif c _ ta _ ea => c :: ta ++ ea
  | .ret v => v.toList

/-- The blocks an instruction may go to. -/
def Inst.targets : Inst → List BlockRef
  | .jump t _ => [t]
  | .brif _ tb _ eb _ => [tb, eb]
  | _ => []

/-- What a block has defined at its end, from what it had on entry. -/
def defOut (inn : Std.HashSet Nat) (b : BlockData) : Std.HashSet Nat :=
  b.insts.foldl (fun s i => match Inst.dst? i with
    | some d => s.insert d.id
    | none => s) (b.params.foldl (fun s (v, _) => s.insert v.id) inn)

/-- What every run has defined on entry to each block, by block id: nothing
    on entry to the first; elsewhere what every edge into the block has
    defined, found by iterating from "everything" (`none`) until a pass
    changes nothing (a set only shrinks, so that is the fixpoint). -/
def definedIn (f : FuncData) : Std.HashMap Nat (Std.HashSet Nat) := Id.run do
  let mut inn : Std.HashMap Nat (Option (Std.HashSet Nat)) :=
    (f.blocks.map fun b => (b.ref.id, if b.ref.id == 0 then some {} else none)).foldl
      (fun m (k, v) => m.insert k v) {}
  for _ in [0:f.blocks.length + 2] do
    let mut changed := false
    for b in f.blocks do
      match inn.getD b.ref.id none with
      | none => pure ()
      | some s =>
          let out := defOut s b
          for i in b.insts do
            for t in Inst.targets i do
              if t.id != 0 then
                match inn.getD t.id none with
                | none => inn := inn.insert t.id (some out); changed := true
                | some s' =>
                    let s'' := s'.filter out.contains
                    if s''.size != s'.size then
                      inn := inn.insert t.id (some s''); changed := true
    if !changed then break
  return inn.fold (fun m k v => match v with
    | some s => m.insert k s
    | none => m) {}

/-- One instruction of a block checked: it reads only what is defined, each
    of its edges' targets counts only on what is, and it defines its own. -/
def blockStep (inOf : Std.HashMap Nat (Std.HashSet Nat)) : Bool × Std.HashSet Nat → Inst → Bool × Std.HashSet Nat
  | (ok, s), i =>
    (ok && (Inst.uses i).all (fun v => s.contains v.id) &&
      (Inst.targets i).all (fun t => match inOf.get? t.id with
        | some it => it.toList.all s.contains
        | none => false),
     match Inst.dst? i with
     | some d => s.insert d.id
     | none => s)

/-- One block checked against what it has on entry: each instruction reads
    only what is defined, and each edge's target counts only on what is. -/
def blockOk (inOf : Std.HashMap Nat (Std.HashSet Nat)) (inn : Std.HashSet Nat) (b : BlockData) : Bool :=
  (b.insts.foldl (blockStep inOf) (true, b.params.foldl (fun s (v, _) => s.insert v.id) inn)).1

/-- **Defined before use**: what is claimed defined on entry to each block
    holds, the first block counting on nothing. -/
def definedOk (f : FuncData) (inOf : Std.HashMap Nat (Std.HashSet Nat)) : Bool :=
  (match inOf.get? 0 with
   | some s => s.isEmpty
   | none => false) &&
  f.blocks.all fun b => match inOf.get? b.ref.id with
    | some inn => blockOk inOf inn b
    | none => false

/-- **Edges are typed**: each edge passes as many arguments as its target
    has parameters, each of its parameter's type, and a shared address in the
    window exactly to a parameter in the window. -/
def edgesTyped (F : Ptx.Facts) : Bool :=
  F.blocks.all fun b => b.insts.all fun i =>
    let edge (t : BlockRef) (args : List Val) : Bool :=
      match F.blocks.find? (·.ref.id == t.id) with
      | some bd => bd.params.length == args.length &&
          (bd.params.zip args).all fun ((pv, pt), a) => F.tys.get? a.id == some pt && F.win a == F.win pv
      | none => false
    match i with
    | .jump t args => edge t args
    | .brif _ tb ta eb ea => edge tb ta && edge eb ea
    | _ => true

/-- The blocks an instruction branches to. -/
def jumpsTo : Inst → List Nat
  | .jump t _ => [t.id]
  | .brif _ t _ e _ => [t.id, e.id]
  | _ => []

/-- **Parameters are plain**: each block's parameters are distinct, each has
    the type it is declared with, a scalar one, and none is a constant or a
    comparison; only the entry's shared base is a shared address or has its
    bits bounded, and no edge passes the entry's parameters. -/
def paramsOk (F : Ptx.Facts) : Bool :=
  F.blocks.all (fun b =>
    (b.params.map (·.1.id)).Nodup &&
    b.params.all fun (v, t) =>
      F.lits.get? v.id == none && !F.negated v && F.tys.get? v.id == some t &&
      [ClifTy.i8, .i16, .i32, .i64, .f32, .f64].contains t) &&
  -- only the entry's shared base is a shared address, at offset 0; no other
  -- parameter's bits are bounded, and no edge passes the entry's
  (match F.blocks with
   | b0 :: _ => b0.params.zipIdx.all fun ((v, _), i) =>
       if i == 3 then F.origin.get? v.id == some .shared && F.mag.get? v.id == some 0
       else F.mag.get? v.id == none && F.origin.get? v.id != some .shared
   | [] => true) &&
  F.blocks.all fun b =>
    (b.ref.id == 0 || b.params.all fun (v, _) => F.mag.get? v.id == none && F.origin.get? v.id != some .shared) &&
    b.insts.all fun i => !(jumpsTo i).contains 0

/-- **Branches test flags**: each conditional branch's condition is an `i8`,
    neither a constant nor a windowed shared address. -/
def condsOk (F : Ptx.Facts) : Bool :=
  F.blocks.all fun b => b.insts.all fun
    | .brif c .. => F.tys.get? c.id == some .i8 && F.lits.get? c.id == none && !F.win c
    | _ => true

/-- The literal a constant-defining instruction gives its value, if one. -/
def litOf (F : Ptx.Facts) : Inst → Option PtxIr.Opnd
  | .iconst _ ty k => if ty == .i8 then none else some (.imm (k % (2 ^ ty.width : Int)).toNat)
  | .fconst _ ty b => some (Ptx.lit ty b)
  | .uextend64 _ a => if F.tys.get? a.id == some .i32 then F.lits.get? a.id else none
  | _ => none

/-- Whether an instruction's value is a comparison with its negation beside it. -/
def negOf (F : Ptx.Facts) : Inst → Bool
  | .icmp _ _ a _ => F.tys.get? a.id != some .i8
  | .fcmp .. => true
  | _ => false

/-- Whether a flag an instruction defines is 0 or 1: a comparison's, the
    constant 0 or 1, or the bitwise and, or or xor of flags. A flag's
    register is a predicate, which holds only whether it is zero. -/
def flagOf (F : Ptx.Facts) : Inst → Bool
  | .iconst _ ty k => ty != .i8 || k % 256 == 0 || k % 256 == 1
  | .icmp .. | .fcmp .. | .band .. | .bor .. | .bxor .. => true
  | i => (Ptx.resTy (F.tys.get? ·.id) i).map (·.2) != some .i8

/-- What else the facts must say of a value instruction `i` defines. -/
def defSide (F : Ptx.Facts) (d : Val) : Inst → Bool
  -- a sum is a windowed shared address exactly when an operand is
  | .iadd _ a b => F.win d == (F.win a || F.win b) && F.tys.get? a.id != some .i8
  -- integer arithmetic is not on flags, which are predicates
  | .isub _ a _ | .imul _ a _ | .udiv _ a _ | .ibin _ _ a _ | .ishl _ a _ | .ushr _ a _ | .ishift _ _ a _ =>
      !F.win d && F.tys.get? a.id != some .i8
  -- a flag is widened as 0 or 1, not sign-extended
  | .sextend64 _ a => !F.win d && F.tys.get? a.id != some .i8
  -- `selp` tests a predicate, and does not choose between predicates
  | .select _ c a _ => !F.win d && F.tys.get? c.id == some .i8 && F.tys.get? a.id != some .i8
  | _ => !F.win d

/-- **A value spans the bits the facts say**: no more than its definition
    bounds, and a shared address at most `sharedReach`. -/
def magOk (F : Ptx.Facts) (d : Val) (i : Inst) : Bool :=
  match F.mag.get? d.id with
  | none => F.origin.get? d.id != some .shared
  | some k => (Ptx.magRule F.origin F.mag F.lits F.tys i).any (· ≤ k) &&
      (F.origin.get? d.id != some .shared || k ≤ Ptx.sharedReach && i matches .iadd ..)

/-- **Values are as the facts say**: each value an instruction defines has the
    type it gives, is a negated comparison as it says, has the literal it gives
    if any, is windowed only as `defSide` allows, and is 0 or 1 if a flag. -/
def defsOk (F : Ptx.Facts) : Bool :=
  F.blocks.all fun b => b.insts.all fun i =>
    match Inst.dst? i with
    | none => true
    | some d =>
        F.tys.get? d.id == (Ptx.resTy (F.tys.get? ·.id) i).map (·.2) &&
        F.negated d == negOf F i &&
        F.lits.get? d.id == litOf F i &&
        defSide F d i && flagOf F i && magOk F d i

/-- The values an instruction reads as values: not an access's address (a
    shared add's included), nor a sum's shared address in the window, nor an
    edge's arguments, which `edgesTyped` takes. -/
def valUses (F : Ptx.Facts) : Inst → List Val
  | .iadd _ a b => if F.win a then [b] else if F.win b then [a] else [a, b]
  | .load .. => []
  | .store v _ | .storeTyped _ v _ | .istore8 v _ => [v]
  | .call none (.intr .sharedAdd) [_, x] => [x]
  | .jump .. => []
  | .brif c .. => [c]
  | i => Inst.uses i

/-- **Shared addresses in the window stay addresses**: each is 64 bits, and no
    instruction reads one as a value. -/
def winOk (F : Ptx.Facts) : Bool :=
  F.blocks.all fun b => b.insts.all fun i =>
    (valUses F i).all (!F.win ·) && (Inst.uses i).all fun v => !F.win v || F.tys.get? v.id == some .i64

/-- The scalar types a register holds. -/
def scalarTys : List ClifTy := [.i8, .i16, .i32, .i64, .f32, .f64]

/-- **Values are scalars**: each value's type is one a register holds. -/
def tysOk (F : Ptx.Facts) : Bool := F.tys.toList.all fun (_, t) => scalarTys.contains t

/-- Whether `v`'s type is one of `ts`. -/
def isTy (F : Ptx.Facts) (v : Val) (ts : List ClifTy) : Bool :=
  match F.tys.get? v.id with
  | some t => ts.contains t
  | none => false

/-- An atomic's address: 64 bits into one of the launch's buffers (never
    shared memory, whose block semantics answers a thread's reads from the
    last barrier), and its value a word or a double word. -/
def atomAt (F : Ptx.Facts) (p : Val) (t : ClifTy) : Bool :=
  isTy F p [.i64] && !F.win p && (t == .i32 || t == .i64) &&
    match F.origin.get? p.id with
    | some (.global _) => true
    | _ => false

/-- A shared add's address: 64 bits into the block's shared memory. -/
def sharedAt (F : Ptx.Facts) (p : Val) : Bool :=
  isTy F p [.i64] && F.origin.get? p.id == some .shared

/-- The intrinsics answering nothing the compiler takes: a barrier, and a
    shared add of a word at a shared address. -/
def takesVoid (F : Ptx.Facts) : Intr → List Val → Bool
  | .barrier, [] => true
  | .sharedAdd, [p, x] => sharedAt F p && isTy F x [.i32]
  | _, _ => false

/-- The atomics the compiler takes: at a word or a double word of a launch's
    buffer, a read-modify-write PTX has, a load or a store. -/
def takesAtomic (F : Ptx.Facts) : Option Val → Atomic → List Val → Bool
  | some _, .rmw t k _, [p, x] => atomAt F p t && isTy F x [t] && (Ptx.atomOp k).isSome
  | some _, .load t _, [p] => atomAt F p t
  | none, .store t _, [v, p] => atomAt F p t && isTy F v [t]
  | _, _, _ => false

/-- An intrinsic of words the compiler takes: three (`Ptx.word3`) or six
    (`Ptx.word6`), every one an `i32`, or six and a 64-bit accumulator
    (`Ptx.word7`). -/
def takesN (F : Ptx.Facts) (c : Intr) (a b e : Val) : List Val → Bool
  | [] => Ptx.word3 c && isTy F a [.i32] && isTy F b [.i32] && isTy F e [.i32]
  | [a3, b0, b1] => Ptx.word6 c && isTy F a [.i32] && isTy F b [.i32] && isTy F e [.i32] &&
      isTy F a3 [.i32] && isTy F b0 [.i32] && isTy F b1 [.i32]
  | [a3, b0, b1, x] => Ptx.word7 c && isTy F a [.i32] && isTy F b [.i32] && isTy F e [.i32] &&
      isTy F a3 [.i32] && isTy F b0 [.i32] && isTy F b1 [.i32] && isTy F x [.i64]
  | _ => false

/-- **What the compiler takes**: the instructions it lowers, each in the forms
    it lowers, on operands of the types it lowers them for; an access's address
    is 64 bits. -/
def takes (F : Ptx.Facts) : Inst → Bool
  | .iconst _ ty _ => [ClifTy.i8, .i16, .i32, .i64].contains ty
  | .fconst _ ty _ => ty == .f32 || ty == .f64
  | .iadd _ a b => isTy F a [.i16, .i32, .i64] && F.tys.get? b.id == F.tys.get? a.id
  | .isub _ a _ | .imul _ a _ | .udiv _ a _ => isTy F a [.i16, .i32, .i64]
  | .ibin _ k a _ => (k == .umin || k == .umax) && isTy F a [.i16, .i32, .i64]
  -- a narrowing to 16 bits, as a 16-bit store wants
  | .iext _ k ty a => k == .reduce && ty == .i16 && isTy F a [.i32, .i64]
  | .band _ a _ | .bor _ a _ | .bxor _ a _ => isTy F a [.i8, .i16, .i32, .i64]
  | .ishl _ a b | .ushr _ a b => isTy F a [.i16, .i32, .i64] && isTy F b [.i16, .i32, .i64]
  | .ishift _ k a b => k == .sshr && isTy F a [.i16, .i32, .i64] && isTy F b [.i16, .i32, .i64]
  | .uextend64 _ a => isTy F a [.i8, .i16, .i32]
  | .sextend64 _ a => isTy F a [.i16, .i32]
  | .ireduce32 _ a => isTy F a [.i64]
  | .select _ _ a _ => isTy F a [.i16, .i32, .i64, .f32, .f64]
  | .fneg _ a => isTy F a [.f32, .f64]
  -- `clz.b32` counts from bit 31, so a 16-bit count is not lowered
  | .ctz _ a => isTy F a [.i32, .i64]
  -- `popc` answers 32 bits, so a count of 64 is not lowered
  | .popcnt _ a => isTy F a [.i32]
  | .bitcast _ _ a => isTy F a [.i16, .i32, .i64, .f32, .f64]
  | .icmp _ cc a _ => isTy F a [.i16, .i32, .i64] || (isTy F a [.i8] && (cc == .eq || cc == .ne))
  -- PTX's float arithmetic is at f32
  | .fadd _ a _ | .fsub _ a _ | .fmul _ a _ | .fmax _ a _ | .fmin _ a _ | .fcmp _ _ a _ => isTy F a [.f32]
  | .fma _ a _ _ => isTy F a [.f32]
  | .fcvtFromSint _ ty a => ty == .f32 && isTy F a [.i16, .i32, .i64]
  | .fbin _ k a _ => k == .fdiv && isTy F a [.f32]
  | .fun1 _ k a => (k == .sqrt || k == .fabs) && isTy F a [.f32]
  | .fconv _ k ty a =>
      match k with
      | .fromUint => ty == .f32 && isTy F a [.i32, .i64]
      | .demote => ty == .f32 && isTy F a [.f64]
      | .toSint => ty == .i32 && isTy F a [.f32]
  | .load _ lo a => isTy F a [.i64] &&
      match lo.kind with
      | .plain => [ClifTy.i16, .i32, .i64, .f32, .f64].contains lo.ty
      | .uload8 | .uload16 => [ClifTy.i16, .i32, .i64].contains lo.ty
      | .uload32 => [ClifTy.i32, .i64].contains lo.ty
      | _ => false
  | .store v a | .istore8 v a => isTy F a [.i64] && isTy F v [.i16, .i32, .i64, .f32, .f64]
  -- a typed store of a value at its own type
  | .storeTyped ty v a => isTy F a [.i64] && F.tys.get? v.id == some ty && [ClifTy.i16, .i32, .i64, .f32, .f64].contains ty
  | .call (some _) (.intr .shflBfly) [v, m] | .call (some _) (.intr .shflIdx) [v, m] =>
      isTy F v [.i32] && isTy F m [.i32]
  | .call (some _) (.intr .ballot) [v] => isTy F v [.i8]
  | .call (some _) (.intr c) (a :: b :: e :: rest) => takesN F c a b e rest
  | .call none (.intr c) args => takesVoid F c args
  | .call d (.atomic a) args => takesAtomic F d a args
  | .call (some _) (.intr c) [v] => (Ptx.approxOp c).isSome && isTy F v [.f32]
  | .jump .. | .brif .. | .ret none => true
  | _ => false

/-- Each instruction one the compiler takes. -/
def takesAll (F : Ptx.Facts) : Bool := F.blocks.all fun b => b.insts.all (takes F)

/-- **The entry block is the thread's**: the first, where a run begins; its
    parameters the thread's identity,
    the shared base and the kernel's own, of their types, the kernel's not
    flags; only the shared base in the window, and it when the window is. -/
def entryOk (F : Ptx.Facts) (k0 : List ClifTy) : Bool :=
  match F.blocks with
  | b0 :: _ => b0.ref.id == 0 && b0.params.map (·.2) == idParams ++ k0 && !k0.contains .i8 &&
      b0.params.zipIdx.all fun ((v, _), i) => F.win v == (F.window && i == 3)
  | [] => false

/-- The labels a body's lines write, in order. -/
def labelsIn (body : List PtxIr.Line) : List PtxIr.Lbl :=
  body.filterMap fun
    | .label n => some n
    | _ => none

/-- The facts a kernel is lowered with: shared addresses held in the window
    unless an instruction reads one as a value, an edge passes one to a
    parameter that is not, or a sum with one leaves shared memory's reach. -/
def Kernel.factsOf (k : Kernel) (f : FuncData) : Ptx.Facts :=
  let F := Ptx.facts f k.params true
  { (if winOk F && edgesTyped F && defsOk F then F else Ptx.facts f k.params false) with warp := k.warp, mulWide := k.mulWide }

/-- The compiler's checks of function `f` of kernel `k`, each with what it
    says when it fails. -/
def Kernel.checks (k : Kernel) (f : FuncData) : List (Bool × String) :=
  let F := k.factsOf f
  [(decide (f.blocks.map (·.ref.id)).Nodup, "two blocks have one name"),
   (definedOk f (definedIn f), "a value is read where it may not be defined"),
   (edgesTyped F, "an edge passes a value of another type"),
   (winOk F, "a shared address in the window is read as a value"),
   (tysOk F, "a value is not a scalar"),
   (takesAll F, "an instruction the compiler does not take"),
   (entryOk F k.params, "the entry block is not the thread's"),
   (paramsOk F, "a block parameter is not plain"),
   (condsOk F, "a branch tests what is not a flag"),
   (defsOk F, "a value is not as the facts say"),
   (decide (k.params.zipIdx.map fun (_, i) => s!"p{i}").Nodup, "two parameters have one name"),
   (decide (k.smem < 2 ^ 32), "more shared memory than a GPU addresses")]

/-- What the compiler refuses in function `f` of kernel `k`, if anything: a
    check failed, a lowering it could not make, or a label written twice, so
    that a branch to it has two places to go. -/
def Kernel.refusal (k : Kernel) (f : FuncData) : Option String :=
  (((k.checks f).find? (!·.1)).map (·.2) <|> (Ptx.body (k.factsOf f) k.params).findSome? (·.err)) <|>
    if decide (labelsIn ((Ptx.body (k.factsOf f) k.params).flatMap (·.lines.toList))).Nodup then none
    else some "a label is written twice"

/-- A kernel's entry from the facts it is lowered with: its parameters, its
    registers' declarations, and its body's lines. -/
def Kernel.assemble (k : Kernel) (F : Ptx.Facts) : PtxIr.Entry :=
  let outs := Ptx.body F k.params
  let n := (F.tys.fold (fun m k _ => max m k) 0) + 1
  let decl : ClifTy → PtxIr.Ty := fun t => if t == .i8 then .pred else Ptx.bits t
  -- 16-bit registers only where a value is 16 bits
  let wide := if F.tys.fold (fun b _ t => b || t == .i16) false then [ClifTy.i16] else []
  let decls := ([ClifTy.i8] ++ wide ++ [ClifTy.i32, .i64, .f32, .f64]).map
      (fun t => PtxIr.RegDecl.range (decl t) s!"%{Ptx.cls t}" n)
    ++ [.range .b32 "%rs" n, .range .pred "%q" n]
  let regs := outs.flatMap (·.regs.toList)
  let extras := [ClifTy.i8, .i16, .i32, .i64, .f32, .f64].filterMap fun t =>
    match regs.filter (·.2 == t) with
    | [] => none
    | xs => some (PtxIr.RegDecl.names (decl t) (xs.map (·.1)))
  let params := k.params.zipIdx.map fun (t, i) =>
    ((if t == .f32 then .f32 else if t == .f64 then .f64 else Ptx.uTy t.width : PtxIr.Ty), s!"p{i}")
  { name := k.name, params, regs := decls ++ extras, body := outs.flatMap (·.lines.toList), bounds := k.bounds }

/-- **One kernel's entry**, or what the compiler does not take: its body
    compiled to blocks as the host's are, then each instruction lowered.
    Shared addresses are held in the window unless an instruction reads one as
    a value or an edge passes one to a parameter that is not. -/
def Kernel.entry (k : Kernel) : Except String PtxIr.Entry :=
  match compileChecked 0 k.code .nil (idParams ++ k.params) with
  | .error e => .error s!"{k.name}: {e}"
  | .ok f =>
    match k.refusal f with
    | some m => .error s!"{k.name}: {m}"
    | none => .ok (k.assemble (k.factsOf f))

/-- **The entry that ships**: the compiled entry with its registers merged
    where they never meet (`Coalesce.run`), which runs every thread to the
    same world (`PtxCoalesce.thread_run`). -/
def Kernel.shipped (k : Kernel) : Except String PtxIr.Entry := k.entry.map Coalesce.run

/-- **Kernels as one PTX module**: one shared array, as large as the largest
    any of them asks for, and each kernel's shipped entry. -/
def Kernel.ptxModule (ks : List Kernel) : Except String PtxIr.Module := do
  let entries ← ks.mapM Kernel.shipped
  let smem := ks.foldl (fun m k => max m k.smem) 16
  pure { version := "7.4", target := "sm_80", addressSize := 64,
         shared := [{ align := 16, sym := "smem", bytes := smem }], entries }

/-- The module's text. -/
def Kernel.module (ks : List Kernel) : Except String String := (Kernel.ptxModule ks).map (·.text)

/-- **The kernel as PTX**: a module of its own. -/
def Kernel.ptx (k : Kernel) : Except String String := Kernel.module [k]

end AlgorithmLib.Kernel
