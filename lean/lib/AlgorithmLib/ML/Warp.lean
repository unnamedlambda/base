import AlgorithmLib.ML.Machine

/-!
  # A warp-level machine — making real tuning provable

  `Stmt` is sequential: `skip`, `seq`, `setR`, `store`, `forN`.  Every kernel
  optimisation that actually matters — cross-lane reductions, shared-memory
  staging, tiling — is *inexpressible* in it.  Meanwhile `AlgorithmLib.PTX`
  emits `shflBfly`, `voteBallot`, `warpReduceSum`, `warpReduceOnline` with no
  semantics attached.  So today the tuning knobs and the proofs live in disjoint
  languages, and "hand-tuned kernels stay provably correct" is aspirational for
  exactly the tuning that makes a kernel fast.

  This file starts closing that.  It mirrors `AlgorithmLib.LZ4Simt` — which
  already does 32-lane lockstep for *integers*, with `vote`/`shfl`/`barwarp` —
  but carries `Float32`, which is what ML kernels need.

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

/-- Lane `l` paired with lane `l XOR m` — the butterfly partner. -/
def xorLane (l : Lane) (m : Nat) : Lane :=
  ⟨(l.val ^^^ m) % W, Nat.mod_lt _ (by decide)⟩

/-- Base address for lane `l` on iteration `i`: the warp covers `4*W`
    consecutive floats per step, each lane taking an aligned quad. -/
def v4BaseOff (off i : Nat) (l : Lane) : Nat := i * (4 * W) + l.val * 4 + off

/-- The single-block case. -/
def v4Base (i : Nat) (l : Lane) : Nat := v4BaseOff 0 i l

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

/-- The address pattern of a coalesced stride-`W` sweep: `i*W + lane`. -/
def strideIx : IdxE := .add (.mul .loopI (.lit W)) .laneId

/-- The address pattern of a vectorised sweep: `i*4W + lane*4`. -/
def strideIxV4 : IdxE := .add (.mul .loopI (.lit (4 * W))) (.mul .laneId (.lit 4))

/-- Grid form: block `c` owns the chunk starting at `c * chunk`. -/
def strideIxV4Cta (chunk : Nat) : IdxE :=
  .add (.add (.mul .loopI (.lit (4 * W))) (.mul .laneId (.lit 4)))
       (.mul .ctaId (.lit chunk))

@[simp] theorem strideIx_eval (cta i : Nat) (l : Lane) :
    strideIx.eval cta i l = i * W + l.val := rfl

@[simp] theorem strideIxV4_eval (cta i : Nat) (l : Lane) :
    strideIxV4.eval cta i l = v4Base i l := rfl

@[simp] theorem strideIxV4Cta_eval (chunk cta i : Nat) (l : Lane) :
    (strideIxV4Cta chunk).eval cta i l = v4BaseOff (cta * chunk) i l := rfl

/-- An external operation, identified by name, with the contract it is assumed
    to satisfy.  `deterministic` records whether the vendor path is pinned to a
    reproducible mode — cuBLAS split-K with atomics is *not*, and a claim of
    bit-exact reproducibility depends on it. -/
structure ExternOp where
  name          : String
  deterministic : Bool
  /-- What the operation is assumed to compute: output memory from input. -/
  spec          : (Nat → Float32) → (Nat → Float32)

/-- Warp state: a per-lane float register file plus global memory.
    Memory is shared across lanes; registers are not. -/
structure WSt where
  regs : Nat → Lane → Float32
  mem  : Buf → Nat → Float32
  /-- Shared memory: on-chip, block-scoped, ~100x lower latency than global.
      This is what tiling stages data through. -/
  smem : Nat → Float32

theorem WSt.ext {a b : WSt} (hr : a.regs = b.regs) (hm : a.mem = b.mem)
    (hs : a.smem = b.smem) : a = b := by
  cases a; cases b; simp_all

namespace WSt

/-- Lockstep write: every lane updates register `r` simultaneously. -/
def setReg (st : WSt) (r : Nat) (f : Lane → Float32) : WSt :=
  { st with regs := fun x => if x = r then f else st.regs x }

/-- A single global store.  Reduction epilogues write one value from one lane,
    so this is the shape that matters. -/
def store1 (st : WSt) (b : Buf) (i : Nat) (v : Float32) : WSt :=
  { st with mem := fun c j => if c = b ∧ j = i then v else st.mem c j }

@[simp] theorem regs_setReg_same (st : WSt) (r : Nat) (f : Lane → Float32) :
    (st.setReg r f).regs r = f := by simp [setReg]

@[simp] theorem regs_setReg_other (st : WSt) (r r' : Nat) (f : Lane → Float32)
    (h : r' ≠ r) : (st.setReg r f).regs r' = st.regs r' := by simp [setReg, h]

@[simp] theorem mem_setReg (st : WSt) (r : Nat) (f : Lane → Float32) :
    (st.setReg r f).mem = st.mem := rfl

@[simp] theorem smem_setReg (st : WSt) (r : Nat) (f : Lane → Float32) :
    (st.setReg r f).smem = st.smem := rfl

@[simp] theorem mem_store1_same (st : WSt) (b : Buf) (i : Nat) (v : Float32) :
    (st.store1 b i v).mem b i = v := by simp [store1]

@[simp] theorem regs_store1 (st : WSt) (b : Buf) (i : Nat) (v : Float32) :
    (st.store1 b i v).regs = st.regs := rfl

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

def WFExp.eval (st : WSt) (l : Lane) : WFExp → Float32
  | .reg r     => st.regs r l
  | .lit v     => v
  | .add a b   => NumOps.add (a.eval st l) (b.eval st l)
  | .mul a b   => NumOps.mul (a.eval st l) (b.eval st l)
  | .neg a     => NumOps.neg (a.eval st l)
  | .inv a     => NumOps.inv (a.eval st l)
  | .exp a     => NumOps.exp (a.eval st l)
  | .ex2 a     => NumOps.ex2 (a.eval st l)
  | .rsqrt a   => NumOps.rsqrt (a.eval st l)
  | .maxW a b  => NumOps.max (a.eval st l) (b.eval st l)
  | .geF a b   => NumOps.ifGe (a.eval st l) (b.eval st l) 1.0 0.0

-- ---------------------------------------------------------------------------
-- Warp statements
-- ---------------------------------------------------------------------------

/-- The warp-parallel fragment.  `shflXor` is `shfl.sync.bfly.b32`: every lane
    simultaneously reads register `s` from lane `l XOR mask`.  This is the one
    instruction that makes cross-lane reduction possible, and the one `Stmt`
    could not express. -/
inductive WStmt where
  | skip    : WStmt
  | seq     : WStmt → WStmt → WStmt
  | setR    : Nat → WFExp → WStmt
  | shflXor : Nat → Nat → Nat → WStmt   -- dst, src, mask
  /-- Each lane loads from its own address — the coalesced access pattern. -/
  | loadIdx : Nat → Buf → (Lane → Nat) → WStmt
  /-- Set a float register from a per-lane *integer* value.  The one bridge
      between the index world and the float world; `cvt.rn.f32.u32` on the
      machine.  An argmax needs it because its arithmetic mixes an element's
      value with the element's *index*, and until now no instruction could put
      an index where a float expression could see it. -/
  | setLaneF : Nat → (Lane → Nat) → WStmt
  /-- `ld.global.v4.f32` — one 128-bit transaction per lane instead of four
      32-bit ones.  Four consecutive floats land in four registers. -/
  | loadV4  : Nat → Nat → Nat → Nat → Buf → (Lane → Nat) → WStmt
  | forN    : Nat → (Nat → WStmt) → WStmt
  /-- `@p st.global.f32` with `p = (laneid == 0)` — the reduction epilogue:
      one lane writes the warp's result. -/
  | storeLane0 : Buf → Nat → Nat → WStmt
  /-- `st.shared.f32` — stage a register into on-chip memory. -/
  | stSmem     : (Lane → Nat) → Nat → WStmt
  /-- `ld.shared.f32` — read it back, possibly from a different lane's slot.
      Together with `barrier` this is what makes tiling expressible. -/
  | ldSmem     : Nat → (Lane → Nat) → WStmt
  /-- `bar.warp.sync` — a no-op within a single warp, which executes in
      lockstep.  Present so tiled kernels are writable now and remain correct
      when the model is widened to a block. -/
  | barrier    : WStmt
  /-- An external kernel invocation — cuBLAS GEMM, cuDNN, a vendor library.
      Modelled as an opaque, deterministic memory transformer with a *named
      contract*: `spec` is what the vendor promises.  Everything composed
      around it stays proven; the one assumption is that the vendor meets its
      contract, and it is recorded rather than implicit. -/
  | extern     : ExternOp → Buf → Buf → WStmt
  /-- `st.global.f32` from every lane to its own address — the map epilogue.
      Lanes are applied in order, so a colliding address is resolved
      last-lane-wins (the same convention `LZ4Simt.storeBytes` uses). -/
  | storeLane  : Buf → (Lane → Nat) → Nat → WStmt

def WStmt.run (s : WStmt) (st : WSt) : WSt :=
  match s with
  | .skip           => st
  | .seq a b        => b.run (a.run st)
  | .setR r e       => st.setReg r (fun l => e.eval st l)
  | .shflXor d s m  => st.setReg d (fun l => st.regs s (xorLane l m))
  | .loadIdx d b ix => st.setReg d (fun l => st.mem b (ix l))
  | .setLaneF d f   => st.setReg d (fun l => NumOps.ofNat (f l))
  | .loadV4 d0 d1 d2 d3 b ix =>
      (((st.setReg d0 (fun l => st.mem b (ix l))).setReg d1 (fun l => st.mem b (ix l + 1))).setReg
          d2 (fun l => st.mem b (ix l + 2))).setReg d3 (fun l => st.mem b (ix l + 3))
  | .forN n f       => (List.range n).foldl (fun s' i => (f i).run s') st
  | .storeLane0 b i r => st.store1 b i (st.regs r ⟨0, by decide⟩)
  | .extern op inB outB =>
      { st with mem := fun c j => if c = outB then op.spec (st.mem inB) j else st.mem c j }
  | .stSmem ix r =>
      -- lanes write in order; a colliding address resolves last-lane-wins
      let upd := (List.finRange W).foldl
        (fun m l => fun j => if j = ix l then st.regs r l else m j) st.smem
      { st with smem := upd }
  | .ldSmem d ix => st.setReg d (fun l => st.smem (ix l))
  | .barrier => st
  | .storeLane b ix r =>
      (List.finRange W).foldl (fun s l => s.store1 b (ix l) (st.regs r l)) st

@[simp] theorem wrun_skip (st : WSt) : WStmt.run .skip st = st := rfl
@[simp] theorem wrun_seq (a b : WStmt) (st : WSt) :
    WStmt.run (.seq a b) st = b.run (a.run st) := rfl
@[simp] theorem wrun_setR (r : Nat) (e : WFExp) (st : WSt) :
    WStmt.run (.setR r e) st = st.setReg r (fun l => e.eval st l) := rfl
@[simp] theorem wrun_shflXor (d s m : Nat) (st : WSt) :
    WStmt.run (.shflXor d s m) st = st.setReg d (fun l => st.regs s (xorLane l m)) := rfl
@[simp] theorem wrun_loadIdx (d : Nat) (b : Buf) (ix : Lane → Nat) (st : WSt) :
    WStmt.run (.loadIdx d b ix) st = st.setReg d (fun l => st.mem b (ix l)) := rfl
@[simp] theorem wrun_loadV4 (d0 d1 d2 d3 : Nat) (b : Buf) (ix : Lane → Nat) (st : WSt) :
    WStmt.run (.loadV4 d0 d1 d2 d3 b ix)
      = fun st => (((st.setReg d0 (fun l => st.mem b (ix l))).setReg d1
          (fun l => st.mem b (ix l + 1))).setReg d2
          (fun l => st.mem b (ix l + 2))).setReg d3 (fun l => st.mem b (ix l + 3)) := rfl
@[simp] theorem wrun_forN (n : Nat) (f : Nat → WStmt) (st : WSt) :
    WStmt.run (.forN n f) st = (List.range n).foldl (fun s' i => (f i).run s') st := rfl
@[simp] theorem wrun_setLaneF (d : Nat) (f : Lane → Nat) (st : WSt) :
    WStmt.run (.setLaneF d f) st = st.setReg d (fun l => NumOps.ofNat (f l)) := rfl
@[simp] theorem wrun_storeLane0 (b : Buf) (i r : Nat) (st : WSt) :
    WStmt.run (.storeLane0 b i r) st = st.store1 b i (st.regs r ⟨0, by decide⟩) := rfl
@[simp] theorem wrun_barrier (st : WSt) : WStmt.run .barrier st = st := rfl
@[simp] theorem wrun_ldSmem (d : Nat) (ix : Lane → Nat) (st : WSt) :
    WStmt.run (.ldSmem d ix) st = st.setReg d (fun l => st.smem (ix l)) := rfl
@[simp] theorem wrun_storeLane (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt) :
    WStmt.run (.storeLane b ix r)
      = fun st => (List.finRange W).foldl
          (fun s l => s.store1 b (ix l) (st.regs r l)) st := rfl

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
theorem storeFold_other {α : Type} (b : Buf) (f : α → Nat) (v : α → Float32)
    (c : Buf) (j : Nat) :
    ∀ (L : List α) (st : WSt), (∀ x ∈ L, ¬(c = b ∧ j = f x)) →
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
theorem storeFold_at {α : Type} (b : Buf) (f : α → Nat) (v : α → Float32)
    (a : Nat) (w : Float32) :
    ∀ (L : List α) (st : WSt),
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

/-- The per-lane store, read back at one lane's address. -/
theorem storeLane_at (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt) (l0 : Lane)
    (hagree : ∀ l, ix l = ix l0 → st.regs r l = st.regs r l0) :
    ((WStmt.storeLane b ix r).run st).mem b (ix l0) = st.regs r l0 := by
  show ((List.finRange W).foldl (fun s l => s.store1 b (ix l) (st.regs r l)) st).mem b (ix l0) = _
  exact storeFold_at b ix (st.regs r) (ix l0) (st.regs r l0) (List.finRange W) st
    (fun l _ h => hagree l h) ⟨l0, List.mem_finRange l0, rfl⟩

/-- A store leaves every other buffer alone. -/
theorem storeLane_otherBuf (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt)
    (c : Buf) (hc : c ≠ b) : ((WStmt.storeLane b ix r).run st).mem c = st.mem c := by
  funext j
  show ((List.finRange W).foldl (fun s l => s.store1 b (ix l) (st.regs r l)) st).mem c j = _
  exact storeFold_other b ix (st.regs r) c j (List.finRange W) st
    (fun _ _ => fun hcon => hc hcon.1)

/-- An address no lane writes is unchanged by the store. -/
theorem storeLane_otherAddr (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt) (a : Nat)
    (h : ∀ l, ix l ≠ a) : ((WStmt.storeLane b ix r).run st).mem b a = st.mem b a := by
  show ((List.finRange W).foldl (fun s l => s.store1 b (ix l) (st.regs r l)) st).mem b a = _
  exact storeFold_other b ix (st.regs r) b a (List.finRange W) st
    (fun l _ => fun hcon => h l hcon.2.symm)

/-- A store leaves the register file alone. -/
theorem storeFold_regs {α : Type} (b : Buf) (f : α → Nat) (v : α → Float32) :
    ∀ (L : List α) (st : WSt),
      (L.foldl (fun s x => s.store1 b (f x) (v x)) st).regs = st.regs := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons x L ih =>
      intro st
      show (L.foldl _ (st.store1 b (f x) (v x))).regs = _
      rw [ih]
      rfl

@[simp] theorem storeLane_regs (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt) :
    ((WStmt.storeLane b ix r).run st).regs = st.regs :=
  storeFold_regs b ix (st.regs r) (List.finRange W) st

/-- **Two stores to one buffer: reading back the first.**

    A kernel that writes two disjoint slices of the same buffer — RoPE writes
    the low and high halves of a head — needs to know the second store did not
    land on the first's addresses.  `hAB` is exactly that, and it is one `omega`
    when the two address maps differ by a constant offset. -/
theorem storeLane_two_first (b : Buf) (ixA ixB : Lane → Nat) (rA rB : Nat)
    (S : WSt) (l0 : Lane) (hAB : ∀ l, ixB l ≠ ixA l0)
    (hAA : ∀ l, ixA l = ixA l0 → S.regs rA l = S.regs rA l0) :
    ((WStmt.storeLane b ixB rB).run ((WStmt.storeLane b ixA rA).run S)).mem b (ixA l0)
      = S.regs rA l0 := by
  rw [storeLane_otherAddr b ixB rB _ (ixA l0) hAB]
  exact storeLane_at b ixA rA S l0 hAA

/-- …and reading back the second, which simply wins. -/
theorem storeLane_two_second (b : Buf) (ixA ixB : Lane → Nat) (rA rB : Nat)
    (S : WSt) (l0 : Lane)
    (hBB : ∀ l, ixB l = ixB l0 → S.regs rB l = S.regs rB l0) :
    ((WStmt.storeLane b ixB rB).run ((WStmt.storeLane b ixA rA).run S)).mem b (ixB l0)
      = S.regs rB l0 :=
  storeLane_at b ixB rB ((WStmt.storeLane b ixA rA).run S) l0 hBB



-- ---------------------------------------------------------------------------
-- The butterfly reduction, and the tree it commits to
-- ---------------------------------------------------------------------------

/-- One butterfly round at `mask`: every lane adds its partner's value.
    This is the *specification* of a shuffle round — note it is a tree step,
    not a sequential accumulation. -/
def bflyStep (m : Nat) (v : Lane → Float32) : Lane → Float32 :=
  fun l => NumOps.add (v l) (v (xorLane l m))

/-- The full 5-round butterfly over a 32-lane warp.  Masks descend 16→1,
    matching `AlgorithmLib.PTX.warpReduceSum`.  Every lane ends holding the
    total, folded in this exact tree order. -/
def bflyFold (v : Lane → Float32) : Lane → Float32 :=
  bflyStep 1 (bflyStep 2 (bflyStep 4 (bflyStep 8 (bflyStep 16 v))))

/-- One shuffle-and-add round: `tmp := shfl(acc, mask); acc := acc + tmp`. -/
def warpRound (acc tmp mask : Nat) : WStmt :=
  .seq (.shflXor tmp acc mask)
       (.setR acc (.add (.reg acc) (.reg tmp)))

/-- The emitted warp reduction — five rounds, masks 16 down to 1.  This is the
    shape `AlgorithmLib.PTX.warpReduceSum` emits. -/
def warpReduceSum (acc tmp : Nat) : WStmt :=
  .seq (warpRound acc tmp 16)
    (.seq (warpRound acc tmp 8)
      (.seq (warpRound acc tmp 4)
        (.seq (warpRound acc tmp 2)
              (warpRound acc tmp 1))))

/-- …and so does the whole network.  Needed by the pipeline layer, where a
    reduction's *only* memory effect must be its final single store. -/
theorem warpReduceSum_mem (acc tmp : Nat) (st : WSt) :
    ((warpReduceSum acc tmp).run st).mem = st.mem := rfl

theorem warpRound_spec (acc tmp mask : Nat) (h : tmp ≠ acc) (st : WSt) :
    ((warpRound acc tmp mask).run st).regs acc
      = bflyStep mask (st.regs acc) := by
  simp only [warpRound, wrun_seq, wrun_setR, wrun_shflXor, WSt.regs_setReg_same]
  funext l
  simp only [WFExp.eval, WSt.regs_setReg_other _ tmp acc _ (Ne.symm h),
             WSt.regs_setReg_same, bflyStep]

/-- **The warp reduction realises the butterfly tree exactly.**

    This is the theorem that makes a cross-lane reduction — the single most
    important primitive in a tuned GPU kernel — provable.  It is an exact
    `Float32` equality, achieved by committing the spec to the tree the shuffle
    network actually walks rather than to a sequential fold. -/
theorem warpReduceSum_spec (acc tmp : Nat) (h : tmp ≠ acc) (st : WSt) :
    ((warpReduceSum acc tmp).run st).regs acc = bflyFold (st.regs acc) := by
  simp only [warpReduceSum, wrun_seq]
  rw [warpRound_spec acc tmp 1 h, warpRound_spec acc tmp 2 h,
      warpRound_spec acc tmp 4 h, warpRound_spec acc tmp 8 h,
      warpRound_spec acc tmp 16 h]
  rfl
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

private abbrev aR : Nat := 0
private abbrev v0 : Nat := 1
private abbrev v1 : Nat := 2
private abbrev v2 : Nat := 3
private abbrev v3 : Nat := 4

def sq (x : Float32) : Float32 := NumOps.mul x x

/-- Vectorised accumulation: one `v4` load per lane per iteration. -/
def warpAccSqV4 (off : Nat) (b : Buf) (k : Nat) : WStmt :=
  .seq (.setR aR (.lit (NumOps.ofNat 0)))
       (.forN k (fun i =>
          .seq (.loadV4 v0 v1 v2 v3 b (v4BaseOff off i))
               (.setR aR (.add (.add (.add (.add (.reg aR) (.mul (.reg v0) (.reg v0)))
                                           (.mul (.reg v1) (.reg v1)))
                                     (.mul (.reg v2) (.reg v2)))
                               (.mul (.reg v3) (.reg v3))))))

/-- The fold a vectorised lane performs — four elements per step, in load
    order.  Note this is a *different* order from the scalar kernel's, so the
    two have genuinely different specs.  Over `Float32` that distinction is
    observable, and pretending otherwise is exactly the bug this design
    prevents. -/
def laneFoldV4 (off : Nat) (mem : Nat → Float32) (k : Nat) (l : Lane) : Float32 :=
  (List.range k).foldl
    (fun a i => NumOps.add (NumOps.add (NumOps.add
        (NumOps.add a (sq (mem (v4BaseOff off i l))))
        (sq (mem (v4BaseOff off i l + 1))))
        (sq (mem (v4BaseOff off i l + 2))))
        (sq (mem (v4BaseOff off i l + 3))))
    (NumOps.ofNat 0)

theorem warpAccSqV4_fold (off : Nat) (b : Buf) :
    ∀ (L : List Nat) (st : WSt) (l : Lane),
      ((L.foldl (fun s' i =>
          (WStmt.seq (.loadV4 v0 v1 v2 v3 b (v4BaseOff off i))
            (.setR aR (.add (.add (.add (.add (.reg aR) (.mul (.reg v0) (.reg v0)))
                                        (.mul (.reg v1) (.reg v1)))
                                  (.mul (.reg v2) (.reg v2)))
                            (.mul (.reg v3) (.reg v3))))).run s') st).regs aR l)
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
      have hstep : ((WStmt.seq (.loadV4 v0 v1 v2 v3 b (v4BaseOff off i))
            (.setR aR (.add (.add (.add (.add (.reg aR) (.mul (.reg v0) (.reg v0)))
                                        (.mul (.reg v1) (.reg v1)))
                                  (.mul (.reg v2) (.reg v2)))
                            (.mul (.reg v3) (.reg v3))))).run st).regs aR
          = fun l => NumOps.add (NumOps.add (NumOps.add
              (NumOps.add (st.regs aR l) (sq (st.mem b (v4BaseOff off i l))))
              (sq (st.mem b (v4BaseOff off i l + 1))))
              (sq (st.mem b (v4BaseOff off i l + 2))))
              (sq (st.mem b (v4BaseOff off i l + 3))) := by
        simp only [wrun_seq, wrun_loadV4, wrun_setR, WSt.regs_setReg_same]
        funext l
        simp only [WFExp.eval, sq,
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
      have hmem : ((WStmt.seq (.loadV4 v0 v1 v2 v3 b (v4BaseOff off i))
            (.setR aR (.add (.add (.add (.add (.reg aR) (.mul (.reg v0) (.reg v0)))
                                        (.mul (.reg v1) (.reg v1)))
                                  (.mul (.reg v2) (.reg v2)))
                            (.mul (.reg v3) (.reg v3))))).run st).mem = st.mem := rfl
      show ((L.foldl _ ((WStmt.seq _ _).run st)).regs aR l) = _
      rw [ih _ l, hmem, hstep]
      rfl

/-- **The vectorised phase is correct.** -/
theorem warpAccSqV4_spec (off : Nat) (b : Buf) (k : Nat) (st : WSt) :
    ((warpAccSqV4 off b k).run st).regs aR = laneFoldV4 off (st.mem b) k := by
  funext l
  show ((List.range k).foldl _ ((WStmt.setR aR (.lit (NumOps.ofNat 0))).run st)).regs aR l = _
  rw [warpAccSqV4_fold off b (List.range k) _ l]
  simp only [wrun_setR, WSt.regs_setReg_same, WFExp.eval, laneFoldV4]
  rfl

/-- The full vectorised reduction: `v4` coalesced accumulate, then butterfly. -/
def warpSumSqV4 (off : Nat) (b : Buf) (k : Nat) : WStmt :=
  .seq (warpAccSqV4 off b k) (warpReduceSum aR v0)

/-- **A vectorised, hand-tuned warp kernel, proven.**

    Four times fewer memory transactions than the scalar version, and its spec
    is stated honestly as a *different* fold — because it is one. -/
theorem warpSumSqV4_spec (off : Nat) (b : Buf) (k : Nat) (st : WSt) :
    ((warpSumSqV4 off b k).run st).regs aR = bflyFold (laneFoldV4 off (st.mem b) k) := by
  show ((warpReduceSum aR v0).run ((warpAccSqV4 off b k).run st)).regs aR = _
  rw [warpReduceSum_spec aR v0 (by decide), warpAccSqV4_spec off b k]

-- ---------------------------------------------------------------------------
-- End to end: load, reduce, store
-- ---------------------------------------------------------------------------

/-- A complete, writable kernel: vectorised coalesced load, sequential per-lane
    accumulation, butterfly cross-lane reduction, single-lane store.

    This is the full shape of a production reduction kernel — the same shape
    `PTX.rmsNormBody` emits — and every stage of it is proven. -/
def warpSumSqV4Store (off : Nat) (b out : Buf) (oi : Nat) (k : Nat) : WStmt :=
  .seq (warpSumSqV4 off b k) (.storeLane0 out oi aR)

/-- **The kernel writes the right value to memory.**

    Not "computes it in a register" — actually lands it in the output buffer,
    as an exact `Float32` equality against a spec that commits to the two-level
    fold the hardware performs. -/
theorem warpSumSqV4Store_spec (off : Nat) (b out : Buf) (oi k : Nat) (st : WSt) :
    ((warpSumSqV4Store off b out oi k).run st).mem out oi
      = bflyFold (laneFoldV4 off (st.mem b) k) ⟨0, by decide⟩ := by
  show (((warpSumSqV4 off b k).run st).store1 out oi
          (((warpSumSqV4 off b k).run st).regs aR ⟨0, by decide⟩)).mem out oi = _
  rw [WSt.mem_store1_same, warpSumSqV4_spec off b k]

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

theorem denote_laneFoldV4E (off : Nat) (env : Fin Γ → Float32) (be : Nat → Expr Γ) (k : Nat)
    (mem : Nat → Float32) (hb : ∀ i, denote env (be i) = mem i) (l : Lane) :
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
    `denote env spec` — an exact `Float32` equality.

    This is the sentence the whole architecture exists to support: a kernel
    written with `ld.global.v4`, coalesced striding and a butterfly shuffle
    network — none of which appear in the spec — provably computes the spec. -/
theorem warpSumSqV4Store_implements (off : Nat) (b out : Buf) (oi k : Nat) (st : WSt)
    (env : Fin Γ → Float32) (be : Nat → Expr Γ)
    (hb : ∀ i, denote env (be i) = st.mem b i) :
    ((warpSumSqV4Store off b out oi k).run st).mem out oi
      = denote env (warpSumSqV4E off be k) := by
  rw [warpSumSqV4Store_spec off b out oi k st]
  show _ = denote env (bflyStepE 1 (bflyStepE 2 (bflyStepE 4 (bflyStepE 8
    (bflyStepE 16 (laneFoldV4E off be k))))) ⟨0, by decide⟩)
  simp only [bflyStepE, denote_add, bflyFold, bflyStep,
             denote_laneFoldV4E off env be k (st.mem b) hb]


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

/-- The value an extern leaves in the output buffer is exactly its contract. -/
@[simp] theorem extern_spec (op : ExternOp) (inB outB : Buf) (st : WSt) (j : Nat) :
    ((WStmt.extern op inB outB).run st).mem outB j = op.spec (st.mem inB) j := by
  show (if outB = outB then _ else _) = _
  simp

/-- An extern leaves every other buffer alone. -/
@[simp] theorem extern_frame (op : ExternOp) (inB outB : Buf) (st : WSt) (c : Buf)
    (h : c ≠ outB) (j : Nat) :
    ((WStmt.extern op inB outB).run st).mem c j = st.mem c j := by
  show (if c = outB then _ else _) = _
  simp [h]

/-- Registers are untouched by an extern — the warp is idle while it runs. -/
@[simp] theorem extern_regs (op : ExternOp) (inB outB : Buf) (st : WSt) :
    ((WStmt.extern op inB outB).run st).regs = st.regs := rfl

/-- Matrix-vector product as a vendor contract: `y i = Σₖ A[i·n+k]·x[k]`,
    committed to a left fold so a *deterministic* vendor mode can meet it
    exactly.  The `A` and `x` regions are laid out consecutively. -/
def gemvSpec (m n : Nat) (deterministic : Bool) : ExternOp where
  name := "cl_cublas_sgemv"
  deterministic := deterministic
  spec := fun mem i =>
    if i < m then
      (List.range n).foldl
        (fun acc k => NumOps.add acc (NumOps.mul (mem (i * n + k)) (mem (m * n + k))))
        NumOps.zero
    else NumOps.zero

end AlgorithmLib.ML
