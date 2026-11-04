module
public import AlgorithmLib.ML.Machine.Buf
meta import AlgorithmLib.ML.Machine.Buf
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # A warp-level machine — making real tuning provable

  `Stmt` is sequential: `skip`, `seq`, `setR`, `store`, `forN`.  Every kernel
  optimisation that actually matters — cross-lane reductions, shared-memory
  staging, tiling — is *inexpressible* in it.  This file is a 32-lane lockstep
  machine with `vote`/`shfl`/`barwarp`, carrying `Float32`, which is what ML
  kernels need.

  ## The reassociation trap

  A butterfly shuffle reduction does **not** compute `Expr.sum`.  `Expr.sum`
  commits to a *left fold*; a shuffle tree computes a *balanced tree fold*.
  Over `Float32` those differ — as measured earlier, left vs right folding of
  `[1.0, 1e16, -1e16]` gives `0.0` vs `1.0`.

  So a warp-reduction kernel cannot be proven against a left-fold spec at all.
  The fix is the same discipline used throughout: **commit the spec to the tree
  the hardware actually walks**.  `bflyFold` below is that commitment, and
  `warpReduceSum_spec` proves the shuffle sequence realises it exactly.
-/

namespace AlgorithmLib.ML

/-- Warp width. -/
abbrev W : Nat := 32
abbrev Lane := Fin W

/-- The thread `32 * wi + l` is lane `l` of warp `wi`. -/
theorem lane_div (wi : Nat) (l : Lane) : (32 * wi + l.val) / 32 = wi ∧ (32 * wi + l.val) % 32 = l.val := by
  have : l.val < 32 := l.isLt; constructor <;> omega

/-- Lane `l` paired with lane `l XOR m` — the butterfly partner. -/
def xorLane (l : Lane) (m : Nat) : Lane :=
  ⟨(l.val ^^^ m) % W, Nat.mod_lt _ (by decide)⟩

/-- Base address for lane `l` on iteration `i`: the warp covers `4*W`
    consecutive floats per step, each lane taking an aligned quad. -/
def v4BaseOff (off i : Nat) (l : Lane) : Nat := i * (4 * W) + l.val * 4 + off

/-- Addresses a kernel can actually *emit*.

    An address is an `IdxE` term, not a Lean function `Lane → Nat`: a closure
    is convenient for proving and impossible to compile, since PTX has no way to
    evaluate one.  `IdxE` is the syntactic fragment a real address computation
    uses — `laneid`, the loop counter, and affine arithmetic — so a kernel built
    from it lowers to `mad.lo`/`add` instructions.

    This is the same shallow-vs-deep distinction that forced `Expr` to be a
    deep embedding, arriving one level down. -/
inductive IdxE where
  | laneId : IdxE
  | loopI  : IdxE
  /-- `%ctaid.x` — which block this warp belongs to. -/
  | ctaId  : IdxE
  | lit    : Nat → IdxE
  /-- An index held in integer register `r` — a *data-dependent* address.

      This is what a gather needs: MoE routing reads which expert a token goes
      to, paged attention reads a block table, a sparse format reads its index
      array.  None of that is affine in `(lane, loop, cta)`, so none of it was
      expressible before.

      It is a **register** read rather than a nested memory load because that is
      what the hardware does: `ld.global.u32` lands an index in `%r`, and the
      address arithmetic then uses the register.  Modelling it any other way
      would put a load inside an address expression, which no PTX instruction
      corresponds to.

      Reading an *integer* register rather than reinterpreting a float is not a
      stylistic choice: offsets in a large model exceed `2^24`, above which
      `Float32` no longer represents consecutive integers, so a float-routed
      index would silently address the wrong expert. -/
  | ireg   : Nat → IdxE
  /-- An index *read from an integer buffer* at a computed offset — a gather's
      source: the MoE routing table, the paged-attention block table, a sparse
      format's index array.

      The buffer is **read-only**, which is why the integer memory is a
      parameter of `eval` rather than a field of the mutable state: no kernel
      writes its own routing table, so there is nothing to frame. -/
  | ldIdx  : Buf → IdxE → IdxE
  | add    : IdxE → IdxE → IdxE
  | mul    : IdxE → IdxE → IdxE

/-- `i` is the current loop counter, `l` the lane, `ir` the integer register
    file.

    `ir` defaults to the zero file so that every affine address — which is all
    of them, before `ireg` — reads exactly as it did.  The default is only ever
    consulted by an `ireg` node, so an `IdxE` built without one is unaffected;
    anything containing `ireg` must pass the real file, and the lowering
    theorems do (`emitIdx_sound` instantiates it with `m.ir`). -/
def IdxE.eval (cta i : Nat) (l : Lane) (ir : Nat → Lane → Nat := fun _ _ => 0)
    (im : Buf → Nat → Nat := fun _ _ => 0) : IdxE → Nat
  | .laneId  => l.val
  | .loopI   => i
  | .ctaId   => cta
  | .lit n   => n
  | .ireg r  => ir r l
  | .ldIdx b off => im b (off.eval cta i l ir im)
  | .add a b => a.eval cta i l ir im + b.eval cta i l ir im
  | .mul a b => a.eval cta i l ir im * b.eval cta i l ir im

/-- Every `ireg` an address mentions is below `n`.

    The emitter allocates fresh index registers from `n` upward, so this is the
    condition that a data-dependent index was loaded *before* the address code
    runs — and therefore is preserved by it (`emitIdx_frame`).  It is the exact
    analogue of `WFExp.regsIn` for the float side. -/
def IdxE.regsBelow (n : Nat) : IdxE → Prop
  | .laneId  => True
  | .loopI   => True
  | .ctaId   => True
  | .lit _   => True
  | .ireg r  => r < n
  | .ldIdx _ off => off.regsBelow n
  | .add a b => a.regsBelow n ∧ b.regsBelow n
  | .mul a b => a.regsBelow n ∧ b.regsBelow n

/-- No data-dependent index.  The lowering theorems are stated over exactly
    this fragment — see the note on `IdxE.ireg` lowering in `Assumptions`. -/
def IdxE.IregFree : IdxE → Prop
  | .laneId  => True
  | .loopI   => True
  | .ctaId   => True
  | .lit _   => True
  | .ireg _  => False
  | .ldIdx _ _ => False
  | .add a b => a.IregFree ∧ b.IregFree
  | .mul a b => a.IregFree ∧ b.IregFree

/-- Decidable, so a concrete kernel discharges the lowering precondition with
    `by decide` instead of a hand-built proof term. -/
instance IdxE.decIregFree : ∀ e : IdxE, Decidable e.IregFree
  | .laneId  => .isTrue trivial
  | .loopI   => .isTrue trivial
  | .ctaId   => .isTrue trivial
  | .lit _   => .isTrue trivial
  | .ireg _  => .isFalse (fun h => h)
  | .ldIdx _ _ => .isFalse (fun h => h)
  | .add a b => @instDecidableAnd _ _ (decIregFree a) (decIregFree b)
  | .mul a b => @instDecidableAnd _ _ (decIregFree a) (decIregFree b)

/-- On the `IregFree` fragment the register file is irrelevant, which is why
    the existing theorems keep their exact statements. -/
theorem IdxE.eval_iregFree (cta i : Nat) (l : Lane) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) :
    ∀ e : IdxE, e.IregFree → e.eval cta i l ir im = e.eval cta i l := by
  intro e
  induction e with
  | laneId => intro _; rfl
  | loopI  => intro _; rfl
  | ctaId  => intro _; rfl
  | lit _  => intro _; rfl
  | ireg r => intro h; exact h.elim
  | ldIdx _ _ _ => intro h; exact h.elim
  | add a b iha ihb => intro h; show _ + _ = _ + _; rw [iha h.1, ihb h.2]
  | mul a b iha ihb => intro h; show _ * _ = _ * _; rw [iha h.1, ihb h.2]

instance IdxE.decRegsBelow (n : Nat) : ∀ e : IdxE, Decidable (e.regsBelow n)
  | .laneId  => .isTrue trivial
  | .loopI   => .isTrue trivial
  | .ctaId   => .isTrue trivial
  | .lit _   => .isTrue trivial
  | .ireg r  => inferInstanceAs (Decidable (r < n))
  | .ldIdx _ off => decRegsBelow n off
  | .add a b => @instDecidableAnd _ _ (decRegsBelow n a) (decRegsBelow n b)
  | .mul a b => @instDecidableAnd _ _ (decRegsBelow n a) (decRegsBelow n b)

theorem IdxE.regsBelow_mono {n n' : Nat} (h : n ≤ n') :
    ∀ e : IdxE, e.regsBelow n → e.regsBelow n' := by
  intro e
  induction e with
  | laneId => intro _; trivial
  | loopI  => intro _; trivial
  | ctaId  => intro _; trivial
  | lit _  => intro _; trivial
  | ireg r => intro hr; exact Nat.lt_of_lt_of_le hr h
  | ldIdx _ _ ih => intro hr; exact ih hr
  | add a b iha ihb => intro hab; exact ⟨iha hab.1, ihb hab.2⟩
  | mul a b iha ihb => intro hab; exact ⟨iha hab.1, ihb hab.2⟩

/-- An address whose registers are all below `n` does not care how registers
    from `n` up are changed — the frame property, on the spec side. -/
theorem IdxE.eval_frame (cta i : Nat) (l : Lane) (n : Nat) (ir ir' : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (hir : ∀ x, x < n → ir' x = ir x) :
    ∀ e : IdxE, e.regsBelow n → e.eval cta i l ir' im = e.eval cta i l ir im := by
  intro e
  induction e with
  | laneId => intro _; rfl
  | loopI  => intro _; rfl
  | ctaId  => intro _; rfl
  | lit _  => intro _; rfl
  | ireg r => intro hr; show ir' r l = ir r l; rw [hir r hr]
  | ldIdx b off ih => intro hr; exact congrArg (im b) (ih hr)
  | add a b iha ihb =>
      intro hab; show _ + _ = _ + _; rw [iha hab.1, ihb hab.2]
  | mul a b iha ihb =>
      intro hab; show _ * _ = _ * _; rw [iha hab.1, ihb hab.2]

/-- Grid form: block `c` owns the chunk starting at `c * chunk`. -/
def strideIxV4Cta (chunk : Nat) : IdxE :=
  .add (.add (.mul .loopI (.lit (4 * W))) (.mul .laneId (.lit 4)))
       (.mul .ctaId (.lit chunk))

/-- A warp's integer state: each lane's 64-bit integer registers, global
    memory's bytes by address, shared memory's bytes by offset, and each
    buffer's address. Integer registers hold addresses as the hardware's do,
    so a byte's address is a register's value and not a buffer and an offset. -/
structure ISt where
  regs : Nat → Lane → UInt64 := fun _ _ => 0
  gmem : Nat → UInt8 := fun _ => 0
  smem : Nat → UInt8 := fun _ => 0
  ptr : Buf → UInt64 := fun _ => 0

/-- Warp state: a per-lane register file plus global memory, at a carrier
    (`Float32` unless said), and the integer state. Memory is shared across
    lanes; registers are not. -/
structure WSt (R : Type := Float32) where
  regs : Nat → Lane → R
  mem  : Buf → Nat → R
  /-- Shared memory: on-chip, block-scoped, ~100x lower latency than global.
      This is what tiling stages data through. -/
  smem : Nat → R
  int : ISt := {}

theorem WSt.ext {R : Type} {a b : WSt R} (hr : a.regs = b.regs) (hm : a.mem = b.mem)
    (hs : a.smem = b.smem) (hi : a.int = b.int) : a = b := by
  cases a; cases b; simp_all

namespace WSt

variable {R : Type}

/-- The state with integer state `i`. -/
def withInt (st : WSt R) (i : ISt) : WSt R := { st with int := i }

/-- Lockstep write: every lane updates register `r` simultaneously. -/
def setReg (st : WSt R) (r : Nat) (f : Lane → R) : WSt R :=
  { st with regs := fun x => if x = r then f else st.regs x }

/-- A single global store.  Reduction epilogues write one value from one lane,
    so this is the shape that matters. -/
def store1 (st : WSt R) (b : Buf) (i : Nat) (v : R) : WSt R :=
  { st with mem := fun c j => if c = b ∧ j = i then v else st.mem c j }

@[simp] theorem regs_setReg_same (st : WSt R) (r : Nat) (f : Lane → R) :
    (st.setReg r f).regs r = f := by simp [setReg]

@[simp] theorem regs_setReg_other (st : WSt R) (r r' : Nat) (f : Lane → R)
    (h : r' ≠ r) : (st.setReg r f).regs r' = st.regs r' := by simp [setReg, h]

@[simp] theorem mem_setReg (st : WSt R) (r : Nat) (f : Lane → R) :
    (st.setReg r f).mem = st.mem := rfl

@[simp] theorem mem_store1_same (st : WSt R) (b : Buf) (i : Nat) (v : R) :
    (st.store1 b i v).mem b i = v := by simp [store1]

end WSt

-- ---------------------------------------------------------------------------
-- Per-lane expressions
-- ---------------------------------------------------------------------------

/-- Expressions evaluated independently in each lane. -/
inductive WFExp where
  | reg   : Nat → WFExp
  | lit   : Float32 → WFExp
  | add   : WFExp → WFExp → WFExp
  | mul   : WFExp → WFExp → WFExp
  | neg   : WFExp → WFExp
  | inv   : WFExp → WFExp
  | exp   : WFExp → WFExp
  /-- `ex2.approx.f32`.  Present because the hardware has it; `exp` is present
      because the *spec* has it.  They are different functions, and the machine
      language says so. -/
  | ex2   : WFExp → WFExp
  | rsqrt : WFExp → WFExp
  /-- `max.f32` — one instruction, and the operation a softmax's stabilising
      pass is built from. -/
  | maxW  : WFExp → WFExp → WFExp
  /-- `set.ge.f32.f32` — `1.0` when `b ≤ a`, else `0.0`.  A *value*, not a
      predicate, which is why it needs no predicate file in the warp model and
      why an argmax can be written without one. -/
  | geF   : WFExp → WFExp → WFExp

def WFExp.eval {R : Type} [NumOps R] [NumLit R] (st : WSt R) (l : Lane) : WFExp → R
  | .reg r     => st.regs r l
  | .lit v     => NumLit.lit v
  | .add a b   => NumOps.add (a.eval st l) (b.eval st l)
  | .mul a b   => NumOps.mul (a.eval st l) (b.eval st l)
  | .neg a     => NumOps.neg (a.eval st l)
  | .inv a     => NumOps.inv (a.eval st l)
  | .exp a     => NumOps.exp (a.eval st l)
  | .ex2 a     => NumOps.ex2 (a.eval st l)
  | .rsqrt a   => NumOps.rsqrt (a.eval st l)
  | .maxW a b  => NumOps.max (a.eval st l) (b.eval st l)
  | .geF a b   => NumOps.ifGe (a.eval st l) (b.eval st l) (NumLit.lit 1.0) (NumLit.lit 0.0)

/-- Every lane stores its register `r` to its own address `ix l` of buffer `b`,
    lanes in order, so a colliding address resolves last-lane-wins. -/
def WSt.storeLanes {R : Type} (st : WSt R) (b : Buf) (ix : Lane → Nat) (r : Nat) : WSt R :=
  (List.finRange W).foldl (fun s l => s.store1 b (ix l) (st.regs r l)) st

/-- Integer operations on 64-bit words; shifts take their amount modulo 64. -/
inductive IOp where
  | add | sub | mul | band | bor | bxor | shl | shr | umin
  deriving Repr, DecidableEq

def IOp.run : IOp → UInt64 → UInt64 → UInt64
  | .add, a, b => a + b
  | .sub, a, b => a - b
  | .mul, a, b => a * b
  | .band, a, b => a &&& b
  | .bor, a, b => a ||| b
  | .bxor, a, b => a ^^^ b
  | .shl, a, b => a <<< b
  | .shr, a, b => a >>> b
  | .umin, a, b => if a ≤ b then a else b

/-- Unsigned comparisons. -/
inductive ICmp where
  | eq | ne | ult | ule | uge
  deriving Repr, DecidableEq

def ICmp.run : ICmp → UInt64 → UInt64 → Bool
  | .eq, a, b => a == b
  | .ne, a, b => a != b
  | .ult, a, b => a < b
  | .ule, a, b => a ≤ b
  | .uge, a, b => b ≤ a

/-- Trailing zeros of a 32-bit word, 32 where it is zero. -/
def ctz32 (x : UInt64) : UInt64 :=
  UInt64.ofNat ((List.range 32).find? (fun k => (x >>> UInt64.ofNat k) &&& 1 == 1) |>.getD 32)

/-- Integer expressions, evaluated in each lane on 64-bit words. -/
inductive IExp where
  | reg  : Nat → IExp
  | lit  : Nat → IExp
  | lane : IExp
  | cta  : IExp
  /-- Where buffer `b` begins. -/
  | ptr  : Buf → IExp
  | bin  : IOp → IExp → IExp → IExp
  /-- `1` where the comparison holds, `0` where not. -/
  | cmp  : ICmp → IExp → IExp → IExp
  /-- Trailing zeros of the low 32 bits, 32 where they are zero. -/
  | ctz  : IExp → IExp
  deriving Repr, DecidableEq

def IExp.eval (st : ISt) (cta : Nat) (l : Lane) : IExp → UInt64
  | .reg r => st.regs r l
  | .lit n => UInt64.ofNat n
  | .lane => UInt64.ofNat l.val
  | .cta => UInt64.ofNat cta
  | .ptr b => st.ptr b
  | .bin o a b => o.run (a.eval st cta l) (b.eval st cta l)
  | .cmp c a b => if c.run (a.eval st cta l) (b.eval st cta l) then 1 else 0
  | .ctz a => ctz32 ((a.eval st cta l) &&& 0xffffffff)

/-- Every buffer an integer expression names. -/
def IExp.bufsOf : IExp → List Buf
  | .ptr b => [b]
  | .bin _ a b | .cmp _ a b => a.bufsOf ++ b.bufsOf
  | .ctz a => a.bufsOf
  | _ => []

/-- `e` is a comparison, a register of `fl`, or their and, or or xor: `0` or
    `1` wherever it runs, the registers `fl` being so. -/
def IExp.isBool (fl : List Nat) : IExp → Bool
  | .cmp .. => true
  | .reg n => fl.contains n
  | .bin .band a b | .bin .bor a b | .bin .bxor a b => a.isBool fl && b.isBool fl
  | _ => false

namespace ISt

/-- Every lane updates integer register `r`. -/
def set (st : ISt) (r : Nat) (f : Lane → UInt64) : ISt :=
  { st with regs := fun x => if x = r then f else st.regs x }

/-- Global memory's byte at `a`. -/
def byte (st : ISt) (a : UInt64) : UInt64 := (st.gmem a.toNat).toUInt64

/-- Shared memory's little-endian 16-bit word at offset `a`. -/
def half (st : ISt) (a : UInt64) : UInt64 :=
  (st.smem a.toNat).toUInt64 ||| ((st.smem (a.toNat + 1)).toUInt64 <<< 8)

/-- The lanes where `g` holds store the low byte of `v` at `a`, in lane
    order, so a colliding address keeps the last lane's. -/
def storeB (st : ISt) (a v g : Lane → UInt64) : ISt :=
  (List.finRange W).foldl (fun s l => if g l ≠ 0 then
    { s with gmem := fun x => if x = (a l).toNat then (v l).toUInt8 else s.gmem x } else s) st

/-- The lanes where `g` holds store the low 16 bits of `v` at shared offset
    `a`, little-endian, in lane order. -/
def storeH (st : ISt) (a v g : Lane → UInt64) : ISt :=
  (List.finRange W).foldl (fun s l => if g l ≠ 0 then
    { s with smem := fun x =>
        if x = (a l).toNat then (v l).toUInt8
        else if x = (a l).toNat + 1 then ((v l) >>> 8).toUInt8 else s.smem x } else s) st

/-- The mask of the lanes where `p` holds. -/
def ballot (p : Lane → UInt64) : UInt64 :=
  (List.finRange W).foldl (fun m l => if p l ≠ 0 then m ||| ((1 : UInt64) <<< UInt64.ofNat l.val) else m) 0

/-- Lane `s mod 32`. -/
def srcLane (s : UInt64) : Lane := ⟨s.toNat % W, Nat.mod_lt _ (by decide)⟩

end ISt

/-- **The warp language**: what one warp runs, written so it can be both
    proven and emitted.  Loops carry a *syntactic* body and addresses are
    `IdxE` terms, because PTX cannot evaluate a Lean closure; `EWStmt.run` says
    what a statement does to the warp's state and `ML.Ptx.Body` lowers it. -/
inductive EWStmt where
  | skip       : EWStmt
  | seq        : EWStmt → EWStmt → EWStmt
  | setR       : Nat → WFExp → EWStmt
  | shflXor    : Nat → Nat → Nat → EWStmt
  | loadIdx    : Nat → Buf → IdxE → EWStmt
  | loadV4     : Nat → Nat → Nat → Nat → Buf → IdxE → EWStmt
  | storeLane0 : Buf → IdxE → Nat → EWStmt
  | storeLane  : Buf → IdxE → Nat → EWStmt
  /-- `st.shared.f32` — stage a register into on-chip memory.  With `barrier`,
      this is what makes a *block* kernel writable in the emittable language
      rather than only in the machine language. -/
  | stSm       : IdxE → Nat → EWStmt
  /-- `ld.shared.f32` — read a slot back, possibly one another warp wrote. -/
  | ldSm       : Nat → IdxE → EWStmt
  /-- `bar.sync` — a no-op within a warp, and the synchronisation point of a
      block's shared memory. -/
  | barrier    : EWStmt
  | forN       : Nat → EWStmt → EWStmt
  /-- A loop whose trip count is read from integer memory at a fixed address —
      lane-uniform by construction.  This is what a sequence-length-dependent
      kernel needs, and `forN` covers every static shape. -/
  | forM       : Buf → Nat → EWStmt → EWStmt
  /-- `cvt.rn.f32.u32` — put an *index* where a float expression can see it.
      The only bridge between `IdxE` and `WFExp`, and the reason an argmax is
      expressible: its arithmetic mixes an element's value with the element's
      own index. -/
  | cvtIF      : Nat → IdxE → EWStmt
  /-- Each lane's integer register `d` := `e`. -/
  | setI       : Nat → IExp → EWStmt
  /-- `ld.global.u8` — each lane's `d` := the byte at address `a`. -/
  | ldB        : Nat → IExp → EWStmt
  /-- `st.global.u8` — where `g` is not zero, the low byte of `v` at `a`. -/
  | stB        : IExp → IExp → IExp → EWStmt
  /-- `ld.shared.u16` — each lane's `d` := the 16-bit word at shared offset `a`. -/
  | ldH        : Nat → IExp → EWStmt
  /-- `st.shared.u16` — where `g` is not zero, the low 16 bits of `v` at `a`. -/
  | stH        : IExp → IExp → IExp → EWStmt
  /-- `vote.ballot` — every lane's `d` := the mask of lanes where `p` is not zero. -/
  | ballot     : Nat → IExp → EWStmt
  /-- `shfl.idx` — each lane's `d` := the low 32 bits of `v` at lane `s mod 32`. -/
  | shflIdx    : Nat → IExp → IExp → EWStmt
  /-- A branch on `c`, which every lane holds alike. -/
  | iif        : IExp → EWStmt → EWStmt → EWStmt
  /-- At most `n` trips of `body` while `c` (alike in every lane) is not zero.
      A trip starts with the integer registers `cs` as the last trip left them
      and every other integer register as before the loop, which is how the
      loop leaves them too: `cs` are what it carries. -/
  | iwhile     : Nat → IExp → List Nat → EWStmt → EWStmt
  /-- `s`, its integer registers `rs` afterwards as before it: registers
      local to `s`. -/
  | locals     : List Nat → EWStmt → EWStmt

/-- A trip of a loop carrying the integer registers `cs`, from before the loop
    `st0`: the other integer registers back as they were. -/
def ISt.keep (cs : List Nat) (st0 st : ISt) : ISt :=
  { st with regs := fun r => if r ∈ cs then st.regs r else st0.regs r }

/-- After a statement with local integer registers `rs`, from before it `st0`:
    those back as they were. -/
def ISt.restore (rs : List Nat) (st0 st : ISt) : ISt :=
  { st with regs := fun r => if r ∈ rs then st0.regs r else st.regs r }


/-- **What a statement does** to the warp's state, in block `cta`, with `i`
    the innermost loop's counter, `ir` the integer registers and `im` integer
    memory: each address is its `IdxE` evaluated there, a loop runs its body
    once per counter value. -/
def EWStmt.run {R : Type} [NumOps R] [NumLit R] (s : EWStmt) (st : WSt R) (cta i : Nat)
    (ir : Nat → Lane → Nat := fun _ _ => 0) (im : Buf → Nat → Nat := fun _ _ => 0) : WSt R :=
  match s with
  | .skip                 => st
  | .seq a b              => b.run (a.run st cta i ir im) cta i ir im
  | .setR r e             => st.setReg r (fun l => e.eval st l)
  | .shflXor d s m        => st.setReg d (fun l => st.regs s (xorLane l m))
  | .loadIdx d b ix       => st.setReg d (fun l => st.mem b (ix.eval cta i l ir im))
  | .loadV4 d0 d1 d2 d3 b ix =>
      (((st.setReg d0 (fun l => st.mem b (ix.eval cta i l ir im))).setReg d1
          (fun l => st.mem b (ix.eval cta i l ir im + 1))).setReg d2
          (fun l => st.mem b (ix.eval cta i l ir im + 2))).setReg d3
          (fun l => st.mem b (ix.eval cta i l ir im + 3))
  | .storeLane0 b ix r    => st.store1 b (ix.eval cta i ⟨0, by decide⟩ ir im) (st.regs r ⟨0, by decide⟩)
  | .storeLane b ix r     => st.storeLanes b (fun l => ix.eval cta i l ir im) r
  | .stSm ix r            =>
      -- lanes write in order; a colliding address resolves last-lane-wins
      let upd := (List.finRange W).foldl
        (fun m l => fun j => if j = ix.eval cta i l ir im then st.regs r l else m j) st.smem
      { st with smem := upd }
  | .ldSm d ix            => st.setReg d (fun l => st.smem (ix.eval cta i l ir im))
  | .barrier              => st
  | .forN n body          => (List.range n).foldl (fun s' j => body.run s' cta j ir im) st
  | .forM bu a body       => (List.range (im bu a)).foldl (fun s' j => body.run s' cta j ir im) st
  | .cvtIF d ix           => st.setReg d (fun l => NumOps.ofNat (ix.eval cta i l ir im))
  | .setI d e             => st.withInt (st.int.set d (fun l => e.eval st.int cta l))
  | .ldB d a              => st.withInt (st.int.set d (fun l => st.int.byte (a.eval st.int cta l)))
  | .stB a v g            => st.withInt (st.int.storeB (fun l => a.eval st.int cta l)
      (fun l => v.eval st.int cta l) (fun l => g.eval st.int cta l))
  | .ldH d a              => st.withInt (st.int.set d (fun l => st.int.half (a.eval st.int cta l)))
  | .stH a v g            => st.withInt (st.int.storeH (fun l => a.eval st.int cta l)
      (fun l => v.eval st.int cta l) (fun l => g.eval st.int cta l))
  | .ballot d p           => st.withInt (st.int.set d (fun _ => ISt.ballot (fun l => p.eval st.int cta l)))
  | .shflIdx d v s        => st.withInt (st.int.set d (fun l =>
      v.eval st.int cta (ISt.srcLane (s.eval st.int cta l)) &&& 0xffffffff))
  | .iif c t e            =>
      if c.eval st.int cta ⟨0, by decide⟩ ≠ 0 then t.run st cta i ir im else e.run st cta i ir im
  | .iwhile n c cs body   =>
      (List.range n).foldl (fun s _ =>
        if c.eval s.int cta ⟨0, by decide⟩ ≠ 0 then
          let s' := body.run s cta i ir im
          s'.withInt (ISt.keep cs st.int s'.int)
        else s) st
  | .locals rs body       =>
      let s' := body.run st cta i ir im
      s'.withInt (ISt.restore rs st.int s'.int)

/-- An invariant every step keeps holds after a fold. -/
theorem foldl_inv {α β : Type} (f : β → α → β) (P : β → Prop) (h : ∀ s a, P s → P (f s a)) :
    ∀ (L : List α) (s : β), P s → P (L.foldl f s) := by
  intro L
  induction L with
  | nil => intro s hs; exact hs
  | cons a L ih => intro s hs; exact ih _ (h s a hs)

section Run
variable {R : Type} [NumOps R] [NumLit R] (st : WSt R) (cta i : Nat) (ir : Nat → Lane → Nat)
  (im : Buf → Nat → Nat)

@[simp] theorem erun_skip : EWStmt.skip.run st cta i ir im = st := rfl
@[simp] theorem erun_seq (a b : EWStmt) :
    (EWStmt.seq a b).run st cta i ir im = b.run (a.run st cta i ir im) cta i ir im := rfl
@[simp] theorem erun_setR (r : Nat) (e : WFExp) :
    (EWStmt.setR r e).run st cta i ir im = st.setReg r (fun l => e.eval st l) := rfl
@[simp] theorem erun_shflXor (d s m : Nat) :
    (EWStmt.shflXor d s m).run st cta i ir im = st.setReg d (fun l => st.regs s (xorLane l m)) := rfl
@[simp] theorem erun_loadIdx (d : Nat) (b : Buf) (ix : IdxE) :
    (EWStmt.loadIdx d b ix).run st cta i ir im = st.setReg d (fun l => st.mem b (ix.eval cta i l ir im)) := rfl
@[simp] theorem erun_loadV4 (d0 d1 d2 d3 : Nat) (b : Buf) (ix : IdxE) :
    (EWStmt.loadV4 d0 d1 d2 d3 b ix).run st cta i ir im =
      (((st.setReg d0 (fun l => st.mem b (ix.eval cta i l ir im))).setReg d1
          (fun l => st.mem b (ix.eval cta i l ir im + 1))).setReg d2
          (fun l => st.mem b (ix.eval cta i l ir im + 2))).setReg d3
          (fun l => st.mem b (ix.eval cta i l ir im + 3)) := rfl
@[simp] theorem erun_storeLane0 (b : Buf) (ix : IdxE) (r : Nat) :
    (EWStmt.storeLane0 b ix r).run st cta i ir im =
      st.store1 b (ix.eval cta i ⟨0, by decide⟩ ir im) (st.regs r ⟨0, by decide⟩) := rfl
@[simp] theorem erun_storeLane (b : Buf) (ix : IdxE) (r : Nat) :
    (EWStmt.storeLane b ix r).run st cta i ir im = st.storeLanes b (fun l => ix.eval cta i l ir im) r := rfl
@[simp] theorem erun_ldSm (d : Nat) (ix : IdxE) :
    (EWStmt.ldSm d ix).run st cta i ir im = st.setReg d (fun l => st.smem (ix.eval cta i l ir im)) := rfl
@[simp] theorem erun_barrier : EWStmt.barrier.run st cta i ir im = st := rfl
@[simp] theorem erun_forN (n : Nat) (body : EWStmt) :
    (EWStmt.forN n body).run st cta i ir im =
      (List.range n).foldl (fun s' j => body.run s' cta j ir im) st := rfl
@[simp] theorem erun_forM (bu : Buf) (a : Nat) (body : EWStmt) :
    (EWStmt.forM bu a body).run st cta i ir im =
      (List.range (im bu a)).foldl (fun s' j => body.run s' cta j ir im) st := rfl
@[simp] theorem erun_cvtIF (d : Nat) (ix : IdxE) :
    (EWStmt.cvtIF d ix).run st cta i ir im = st.setReg d (fun l => NumOps.ofNat (ix.eval cta i l ir im)) := rfl

end Run


-- ---------------------------------------------------------------------------
-- Reading a store back
-- ---------------------------------------------------------------------------

/-!
  Every theorem so far describes a *register*.  A kernel's output is a *store*,
  and saying what memory holds afterwards means knowing which write landed last
  at that address.

  The obvious hypothesis is that the address map is injective, but that forces a
  `Nodup` obligation on the lane list which this Mathlib-free setting makes
  awkward.  **"Every writer to this address writes the same value" is weaker,
  sufficient, and follows from injectivity** — so it is the hypothesis used
  here, and instances discharge it from injectivity when they have it.

  This closes a gap that has been open since `compileWKernel_correct`: that
  theorem proves the *value* a kernel computes and says nothing about where it
  lands.
-/

/-- An address no store in the list touches is unchanged. -/
theorem storeFold_other {α R : Type} (b : Buf) (f : α → Nat) (v : α → R)
    (c : Buf) (j : Nat) :
    ∀ (L : List α) (st : WSt R), (∀ x ∈ L, ¬(c = b ∧ j = f x)) →
      ((L.foldl (fun s x => s.store1 b (f x) (v x)) st).mem c j) = st.mem c j := by
  intro L
  induction L with
  | nil => intro st _; rfl
  | cons x L ih =>
      intro st h
      show ((L.foldl _ (st.store1 b (f x) (v x))).mem c j) = _
      rw [ih _ (fun y hy => h y (by simp [hy]))]
      show (if c = b ∧ j = f x then _ else _) = _
      rw [if_neg (h x (by simp))]

/-- **Reading a store back.**  If some element writes address `a` and every
    element that writes `a` writes the same value, that is what memory holds. -/
theorem storeFold_at {α R : Type} (b : Buf) (f : α → Nat) (v : α → R)
    (a : Nat) (w : R) :
    ∀ (L : List α) (st : WSt R),
      (∀ x ∈ L, f x = a → v x = w) → (∃ x ∈ L, f x = a) →
      ((L.foldl (fun s x => s.store1 b (f x) (v x)) st).mem b a) = w := by
  intro L
  induction L with
  | nil => intro _ _ hex; obtain ⟨_, hm, _⟩ := hex; exact absurd hm (by simp)
  | cons x L ih =>
      intro st hall hex
      show ((L.foldl _ (st.store1 b (f x) (v x))).mem b a) = _
      by_cases hlater : ∃ y ∈ L, f y = a
      · exact ih _ (fun y hy => hall y (by simp [hy])) hlater
      · -- nothing later writes `a`, so `x` must, and its write survives
        have hx : f x = a := by
          obtain ⟨y, hy, hfy⟩ := hex
          rcases List.mem_cons.mp hy with h | h
          · exact h ▸ hfy
          · exact absurd ⟨y, h, hfy⟩ hlater
        rw [storeFold_other b f v b a L _
              (fun y hy => fun hc => hlater ⟨y, hy, hc.2.symm⟩)]
        rw [← hx]
        show (if b = b ∧ f x = f x then _ else _) = _
        rw [if_pos ⟨rfl, rfl⟩]
        exact hall x (by simp) hx

section Carrier
variable {R : Type} [NumOps R] [NumLit R]

/-- The per-lane store, read back at one lane's address. -/
theorem storeLane_at (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt R) (l0 : Lane)
    (hagree : ∀ l, ix l = ix l0 → st.regs r l = st.regs r l0) :
    (st.storeLanes b ix r).mem b (ix l0) = st.regs r l0 := by
  show ((List.finRange W).foldl (fun s l => s.store1 b (ix l) (st.regs r l)) st).mem b (ix l0) = _
  exact storeFold_at b ix (st.regs r) (ix l0) (st.regs r l0) (List.finRange W) st
    (fun l _ h => hagree l h) ⟨l0, List.mem_finRange l0, rfl⟩

/-- A store leaves every other buffer alone. -/
theorem storeLane_otherBuf (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt R)
    (c : Buf) (hc : c ≠ b) : (st.storeLanes b ix r).mem c = st.mem c := by
  funext j
  show ((List.finRange W).foldl (fun s l => s.store1 b (ix l) (st.regs r l)) st).mem c j = _
  exact storeFold_other b ix (st.regs r) c j (List.finRange W) st
    (fun _ _ => fun hcon => hc hcon.1)

/-- An address no lane writes is unchanged by the store. -/
theorem storeLane_otherAddr (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt R) (a : Nat)
    (h : ∀ l, ix l ≠ a) : (st.storeLanes b ix r).mem b a = st.mem b a := by
  show ((List.finRange W).foldl (fun s l => s.store1 b (ix l) (st.regs r l)) st).mem b a = _
  exact storeFold_other b ix (st.regs r) b a (List.finRange W) st
    (fun l _ => fun hcon => h l hcon.2.symm)

/-- A store leaves the register file alone. -/
theorem storeFold_regs {α R : Type} (b : Buf) (f : α → Nat) (v : α → R) :
    ∀ (L : List α) (st : WSt R),
      (L.foldl (fun s x => s.store1 b (f x) (v x)) st).regs = st.regs := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons x L ih =>
      intro st
      show (L.foldl _ (st.store1 b (f x) (v x))).regs = _
      rw [ih]
      rfl

@[simp] theorem storeLane_regs (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt R) :
    (st.storeLanes b ix r).regs = st.regs :=
  storeFold_regs b ix (st.regs r) (List.finRange W) st

/-- **Two stores to one buffer: reading back the first.**

    A kernel that writes two disjoint slices of the same buffer — RoPE writes
    the low and high halves of a head — needs to know the second store did not
    land on the first's addresses.  `hAB` is exactly that, and it is one `omega`
    when the two address maps differ by a constant offset. -/
theorem storeLane_two_first (b : Buf) (ixA ixB : Lane → Nat) (rA rB : Nat)
    (S : WSt R) (l0 : Lane) (hAB : ∀ l, ixB l ≠ ixA l0)
    (hAA : ∀ l, ixA l = ixA l0 → S.regs rA l = S.regs rA l0) :
    ((S.storeLanes b ixA rA).storeLanes b ixB rB).mem b (ixA l0) = S.regs rA l0 := by
  rw [storeLane_otherAddr b ixB rB _ (ixA l0) hAB]
  exact storeLane_at b ixA rA S l0 hAA

/-- …and reading back the second, which simply wins. -/
theorem storeLane_two_second (b : Buf) (ixA ixB : Lane → Nat) (rA rB : Nat)
    (S : WSt R) (l0 : Lane)
    (hBB : ∀ l, ixB l = ixB l0 → S.regs rB l = S.regs rB l0) :
    ((S.storeLanes b ixA rA).storeLanes b ixB rB).mem b (ixB l0) = S.regs rB l0 :=
  storeLane_at b ixB rB (S.storeLanes b ixA rA) l0 hBB



-- ---------------------------------------------------------------------------
-- The butterfly reduction, and the tree it commits to
-- ---------------------------------------------------------------------------


/-- One butterfly round at `mask`: every lane adds its partner's value.
    This is the *specification* of a shuffle round — note it is a tree step,
    not a sequential accumulation. -/
def bflyStep (m : Nat) (v : Lane → R) : Lane → R :=
  fun l => NumOps.add (v l) (v (xorLane l m))

/-- The full 5-round butterfly over a 32-lane warp.  Masks descend 16→1,
    matching `Kernel.Block.warpAll`.  Every lane ends holding the
    total, folded in this exact tree order. -/
def bflyFold (v : Lane → R) : Lane → R :=
  bflyStep 1 (bflyStep 2 (bflyStep 4 (bflyStep 8 (bflyStep 16 v))))

/-- One shuffle-and-add round: `tmp := shfl(acc, mask); acc := acc + tmp`. -/
def warpRound (acc tmp mask : Nat) : EWStmt :=
  .seq (.shflXor tmp acc mask)
       (.setR acc (.add (.reg acc) (.reg tmp)))

/-- The emitted warp reduction — five rounds, masks 16 down to 1. -/
def warpReduceSum (acc tmp : Nat) : EWStmt :=
  .seq (warpRound acc tmp 16)
    (.seq (warpRound acc tmp 8)
      (.seq (warpRound acc tmp 4)
        (.seq (warpRound acc tmp 2)
              (warpRound acc tmp 1))))

section Reduce
variable (st : WSt R) (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)

/-- …and so does the whole network.  Needed by the pipeline layer, where a
    reduction's *only* memory effect must be its final single store. -/
theorem warpReduceSum_mem (acc tmp : Nat) :
    ((warpReduceSum acc tmp).run st cta i ir im).mem = st.mem := rfl

theorem warpRound_spec (acc tmp mask : Nat) (h : tmp ≠ acc) :
    ((warpRound acc tmp mask).run st cta i ir im).regs acc
      = bflyStep mask (st.regs acc) := by
  simp only [warpRound, erun_seq, erun_setR, erun_shflXor, WSt.regs_setReg_same]
  funext l
  simp only [WFExp.eval, WSt.regs_setReg_other _ tmp acc _ (Ne.symm h),
             WSt.regs_setReg_same, bflyStep]

/-- **The warp reduction realises the butterfly tree exactly.**

    This is the theorem that makes a cross-lane reduction — the single most
    important primitive in a tuned GPU kernel — provable.  It is an exact
    equality at any carrier, achieved by committing the spec to the tree the shuffle
    network actually walks rather than to a sequential fold. -/
theorem warpReduceSum_spec (acc tmp : Nat) (h : tmp ≠ acc) :
    ((warpReduceSum acc tmp).run st cta i ir im).regs acc = bflyFold (st.regs acc) := by
  simp only [warpReduceSum, erun_seq]
  rw [warpRound_spec _ cta i ir im acc tmp 1 h, warpRound_spec _ cta i ir im acc tmp 2 h,
      warpRound_spec _ cta i ir im acc tmp 4 h, warpRound_spec _ cta i ir im acc tmp 8 h,
      warpRound_spec _ cta i ir im acc tmp 16 h]
  rfl

end Reduce
-- ---------------------------------------------------------------------------
-- A complete warp kernel: strided accumulate, then tree-reduce
-- ---------------------------------------------------------------------------

/-! There is no scalar sum-of-squares chain: the vectorised `warpAccSqV4`
    below covers the shipped kernel, and `Schema.sweep_fold` covers the general
    case — one load, one accumulate, any combining expression, proven once. -/

-- ---------------------------------------------------------------------------
-- Vectorised loads: same reduction, 4x fewer memory transactions
-- ---------------------------------------------------------------------------

/-! Scalar `ld.global.f32` issues one 32-bit transaction per lane per element.
    `ld.global.v4.f32` issues one 128-bit transaction for four.  On a
    bandwidth-bound decode kernel that is the difference between saturating the
    memory system and leaving most of it idle — a prime suspect for the measured
    32%-of-roofline gap.

    Registers are concrete (`acc=0, r0..r3 = 1..4`) exactly as a real kernel's
    allocation would be, which also makes every distinctness side condition
    `by decide`. -/

abbrev aR : Nat := 0
abbrev v0 : Nat := 1
abbrev v1 : Nat := 2
abbrev v2 : Nat := 3
abbrev v3 : Nat := 4

def sq (x : R) : R := NumOps.mul x x

/-- One vectorised step: a `v4` load at `ix`, then four squares accumulated. -/
def accSqStep (b : Buf) (ix : IdxE) : EWStmt :=
  .seq (.loadV4 v0 v1 v2 v3 b ix)
       (.setR aR (.add (.add (.add (.add (.reg aR) (.mul (.reg v0) (.reg v0)))
                                   (.mul (.reg v1) (.reg v1)))
                             (.mul (.reg v2) (.reg v2)))
                       (.mul (.reg v3) (.reg v3))))

/-- Vectorised accumulation: one `v4` load per lane per iteration, the quad's
    base at `ix`. -/
def warpAccSqV4 (b : Buf) (ix : IdxE) (k : Nat) : EWStmt :=
  .seq (.setR aR (.lit (NumOps.ofNat 0))) (.forN k (accSqStep b ix))

/-- The fold a vectorised lane performs — four elements per step, in load
    order.  Note this is a *different* order from the scalar kernel's, so the
    two have genuinely different specs.  Over `Float32` that distinction is
    observable, and pretending otherwise is exactly the bug this design
    prevents. -/
def laneFoldV4 (off : Nat) (mem : Nat → R) (k : Nat) (l : Lane) : R :=
  (List.range k).foldl
    (fun a i => NumOps.add (NumOps.add (NumOps.add
        (NumOps.add a (sq (mem (v4BaseOff off i l))))
        (sq (mem (v4BaseOff off i l + 1))))
        (sq (mem (v4BaseOff off i l + 2))))
        (sq (mem (v4BaseOff off i l + 3))))
    (NumOps.ofNat 0)

theorem warpAccSqV4_fold (off : Nat) (b : Buf) (ix : IdxE) (cta : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (hix : ∀ i l, ix.eval cta i l ir im = v4BaseOff off i l) :
    ∀ (L : List Nat) (st : WSt R) (l : Lane),
      ((L.foldl (fun s' i => (accSqStep b ix).run s' cta i ir im) st).regs aR l)
        = L.foldl (fun a i => NumOps.add (NumOps.add (NumOps.add
              (NumOps.add a (sq (st.mem b (v4BaseOff off i l))))
              (sq (st.mem b (v4BaseOff off i l + 1))))
              (sq (st.mem b (v4BaseOff off i l + 2))))
              (sq (st.mem b (v4BaseOff off i l + 3))))
            (st.regs aR l) := by
  intro L
  induction L with
  | nil => intro st l; rfl
  | cons i L ih =>
      intro st l
      have hstep : ((accSqStep b ix).run st cta i ir im).regs aR
          = fun l => NumOps.add (NumOps.add (NumOps.add
              (NumOps.add (st.regs aR l) (sq (st.mem b (v4BaseOff off i l))))
              (sq (st.mem b (v4BaseOff off i l + 1))))
              (sq (st.mem b (v4BaseOff off i l + 2))))
              (sq (st.mem b (v4BaseOff off i l + 3))) := by
        simp only [accSqStep, erun_seq, erun_loadV4, erun_setR, WSt.regs_setReg_same]
        funext l
        simp only [WFExp.eval, sq, hix,
          WSt.regs_setReg_other _ v3 aR _ (by decide),
          WSt.regs_setReg_other _ v2 aR _ (by decide),
          WSt.regs_setReg_other _ v1 aR _ (by decide),
          WSt.regs_setReg_other _ v0 aR _ (by decide),
          WSt.regs_setReg_other _ v3 v0 _ (by decide),
          WSt.regs_setReg_other _ v2 v0 _ (by decide),
          WSt.regs_setReg_other _ v1 v0 _ (by decide),
          WSt.regs_setReg_other _ v3 v1 _ (by decide),
          WSt.regs_setReg_other _ v2 v1 _ (by decide),
          WSt.regs_setReg_other _ v3 v2 _ (by decide),
          WSt.regs_setReg_same]
      have hmem : ((accSqStep b ix).run st cta i ir im).mem = st.mem := rfl
      show ((L.foldl _ ((accSqStep b ix).run st cta i ir im)).regs aR l) = _
      rw [ih _ l, hmem, hstep]
      rfl

/-- **The vectorised phase is correct.** -/
theorem warpAccSqV4_spec [LitNat R] (off : Nat) (b : Buf) (ix : IdxE) (k cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (hix : ∀ i l, ix.eval cta i l ir im = v4BaseOff off i l) (st : WSt R) :
    ((warpAccSqV4 b ix k).run st cta 0 ir im).regs aR = laneFoldV4 off (st.mem b) k := by
  funext l
  show ((List.range k).foldl _ ((EWStmt.setR aR (.lit (NumOps.ofNat 0))).run st cta 0 ir im)).regs aR l = _
  rw [warpAccSqV4_fold off b ix cta ir im hix (List.range k) _ l]
  simp only [erun_setR, WSt.regs_setReg_same, WFExp.eval, laneFoldV4, LitNat.ofNat]
  rfl

/-- The full vectorised reduction: `v4` coalesced accumulate, then butterfly. -/
def warpSumSqV4 (b : Buf) (ix : IdxE) (k : Nat) : EWStmt :=
  .seq (warpAccSqV4 b ix k) (warpReduceSum aR v0)

/-- **A vectorised, hand-tuned warp kernel, proven.**

    Four times fewer memory transactions than the scalar version, and its spec
    is stated honestly as a *different* fold — because it is one. -/
theorem warpSumSqV4_spec [LitNat R] (off : Nat) (b : Buf) (ix : IdxE) (k cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (hix : ∀ i l, ix.eval cta i l ir im = v4BaseOff off i l) (st : WSt R) :
    ((warpSumSqV4 b ix k).run st cta 0 ir im).regs aR = bflyFold (laneFoldV4 off (st.mem b) k) := by
  show ((warpReduceSum aR v0).run ((warpAccSqV4 b ix k).run st cta 0 ir im) cta 0 ir im).regs aR = _
  rw [warpReduceSum_spec _ cta 0 ir im aR v0 (by decide), warpAccSqV4_spec off b ix k cta ir im hix]

-- ---------------------------------------------------------------------------
-- End to end: load, reduce, store
-- ---------------------------------------------------------------------------

/-- A complete, writable kernel: vectorised coalesced load, sequential per-lane
    accumulation, butterfly cross-lane reduction, single-lane store at `oi`.

    This is the full shape of a production reduction kernel, and every stage
    of it is proven. -/
def warpSumSqV4Store (b out : Buf) (ix oi : IdxE) (k : Nat) : EWStmt :=
  .seq (warpSumSqV4 b ix k) (.storeLane0 out oi aR)

/-- **The kernel writes the right value to memory.**

    Not "computes it in a register" — actually lands it in the output buffer,
    as an exact equality, at any carrier, against a spec that commits to the two-level
    fold the hardware performs. -/
theorem warpSumSqV4Store_spec [LitNat R] (off : Nat) (b out : Buf) (ix oi : IdxE) (k cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (hix : ∀ i l, ix.eval cta i l ir im = v4BaseOff off i l) (st : WSt R) :
    ((warpSumSqV4Store b out ix oi k).run st cta 0 ir im).mem out (oi.eval cta 0 ⟨0, by decide⟩ ir im)
      = bflyFold (laneFoldV4 off (st.mem b) k) ⟨0, by decide⟩ := by
  show (((warpSumSqV4 b ix k).run st cta 0 ir im).store1 out _
          (((warpSumSqV4 b ix k).run st cta 0 ir im).regs aR ⟨0, by decide⟩)).mem out _ = _
  rw [WSt.mem_store1_same, warpSumSqV4_spec off b ix k cta ir im hix]

-- ---------------------------------------------------------------------------
-- Connecting the tuned kernel to a model spec
-- ---------------------------------------------------------------------------

/-! Everything above proves warp kernels against *warp-level folds*.  That is
    not yet the claim that matters — a user wants "my hand-tuned kernel computes
    **my model spec**".

    The bridge: the two-level fold the hardware walks is itself expressible as
    an ordinary `Expr`, so the spec and the kernel meet in `denote`. -/

variable {Γ : Nat}

/-- The per-lane sequential fold, as a spec-language term. -/
def laneFoldV4E (off : Nat) (be : Nat → Expr Γ) (k : Nat) (l : Lane) : Expr Γ :=
  (List.range k).foldl
    (fun a i => .add (.add (.add
        (.add a (.mul (be (v4BaseOff off i l)) (be (v4BaseOff off i l))))
        (.mul (be (v4BaseOff off i l + 1)) (be (v4BaseOff off i l + 1))))
        (.mul (be (v4BaseOff off i l + 2)) (be (v4BaseOff off i l + 2))))
        (.mul (be (v4BaseOff off i l + 3)) (be (v4BaseOff off i l + 3))))
    (.lit 0)

/-- One butterfly round, as a spec-language term. -/
def bflyStepE (m : Nat) (v : Lane → Expr Γ) : Lane → Expr Γ :=
  fun l => .add (v l) (v (xorLane l m))

/-- The whole two-level reduction as one `Expr`.  This is what a user writes
    (or, later, what a rewrite produces) when they want the warp-tree order. -/
def warpSumSqV4E (off : Nat) (be : Nat → Expr Γ) (k : Nat) : Expr Γ :=
  bflyStepE 1 (bflyStepE 2 (bflyStepE 4 (bflyStepE 8 (bflyStepE 16
    (laneFoldV4E off be k))))) ⟨0, by decide⟩

theorem denote_laneFoldV4E (off : Nat) (env : Fin Γ → R) (be : Nat → Expr Γ) (k : Nat)
    (mem : Nat → R) (hb : ∀ i, denote env (be i) = mem i) (l : Lane) :
    denote env (laneFoldV4E off be k l) = laneFoldV4 off mem k l := by
  show denote env ((List.range k).foldl _ (.lit 0)) = (List.range k).foldl _ (NumOps.ofNat 0)
  have key : ∀ (L : List Nat) (acc : Expr Γ),
      denote env (L.foldl (fun a i => Expr.add (.add (.add
          (.add a (.mul (be (v4BaseOff off i l)) (be (v4BaseOff off i l))))
          (.mul (be (v4BaseOff off i l + 1)) (be (v4BaseOff off i l + 1))))
          (.mul (be (v4BaseOff off i l + 2)) (be (v4BaseOff off i l + 2))))
          (.mul (be (v4BaseOff off i l + 3)) (be (v4BaseOff off i l + 3)))) acc)
        = L.foldl (fun a i => NumOps.add (NumOps.add (NumOps.add
            (NumOps.add a (sq (mem (v4BaseOff off i l))))
            (sq (mem (v4BaseOff off i l + 1))))
            (sq (mem (v4BaseOff off i l + 2))))
            (sq (mem (v4BaseOff off i l + 3)))) (denote env acc) := by
    intro L
    induction L with
    | nil => intro acc; rfl
    | cons i L ih =>
        intro acc
        rw [List.foldl_cons, List.foldl_cons, ih]
        simp only [denote_add, denote_mul, hb, sq]
  rw [key (List.range k) (.lit 0)]
  rfl

/-- **The hand-tuned, vectorised warp kernel computes the model spec.**

    `hb` is the layout link: it says buffer slot `i` holds what spec term
    `be i` denotes.  Given that, the kernel's *store* lands exactly
    `denote env spec` — an exact equality at any carrier.

    This is the sentence the whole architecture exists to support: a kernel
    written with `ld.global.v4`, coalesced striding and a butterfly shuffle
    network — none of which appear in the spec — provably computes the spec. -/
theorem warpSumSqV4Store_implements [LitNat R] (off : Nat) (b out : Buf) (ix oi : IdxE) (k cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (hix : ∀ i l, ix.eval cta i l ir im = v4BaseOff off i l) (st : WSt R)
    (env : Fin Γ → R) (be : Nat → Expr Γ)
    (hb : ∀ i, denote env (be i) = st.mem b i) :
    ((warpSumSqV4Store b out ix oi k).run st cta 0 ir im).mem out (oi.eval cta 0 ⟨0, by decide⟩ ir im)
      = denote env (warpSumSqV4E off be k) := by
  rw [warpSumSqV4Store_spec off b out ix oi k cta ir im hix st]
  show _ = denote env (bflyStepE 1 (bflyStepE 2 (bflyStepE 4 (bflyStepE 8
    (bflyStepE 16 (laneFoldV4E off be k))))) ⟨0, by decide⟩)
  simp only [bflyStepE, denote_add, bflyFold, bflyStep,
             denote_laneFoldV4E off env be k (st.mem b) hb]


end Carrier

-- ---------------------------------------------------------------------------
-- Delegating to a vendor kernel, with the assumption named
-- ---------------------------------------------------------------------------

/-! GEMM is the one kernel worth delegating: cuBLAS uses tensor cores, multi-level
    tiling and per-architecture autotuning, and is not realistically beatable at
    large dense shapes.  Decode-shaped (M=1) and quantized GEMM are a different
    story and stay in-house.

    Delegation introduces **no new class of assumption** — it is the same
    Float32-vs-ℝ gap at coarser granularity, one assumption about `cublasGemm`
    instead of `n` about `Float32.add`.  What it does cost is the *reduction-order
    commitment*: with split-K and atomics the vendor result is not reproducible,
    so `deterministic` must be set for any bit-exactness claim. -/

end AlgorithmLib.ML
