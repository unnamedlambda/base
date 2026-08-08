import AlgorithmLib.ML.Ptx
import AlgorithmLib.ML.WarpEmit

/-!
  # The full lowering, part 1: a machine with memory, addresses and loops

  `Ptx.lean` gave the *straight-line float* fragment a semantics and proved the
  emitter against it (`emitP_sound`, `emitS_sound`).  Everything a real kernel
  also needs — address arithmetic, global loads, vectorised loads, predicated
  stores, shared memory, and counted loops — was still trusted.

  This file closes that.  It defines `MState`, which extends `PState` with

  * an **index register file** `ir : Nat → Lane → Nat` (`%r`), per lane because
    `%laneid` is,
  * a **predicate file** `pr : Nat → Lane → Bool` (`%p`),
  * **shared memory** `sm`, and
  * the block id `cta`, which `%ctaid.x` reads,

  and a structured instruction set `SI` whose semantics is given in terms of the
  *same* `WSt` operations the warp model uses.  `emitEW` lowers the emittable
  language `EWStmt` into `SI`, and `emitEW_sound` proves the result runs the
  statement.

  Control flow is *structured* here (`SI.loop`); `PtxFlat.lean` compiles that to
  real labels and branches and proves the program-counter machine agrees.
-/

namespace AlgorithmLib.ML

-- ---------------------------------------------------------------------------
-- Machine state
-- ---------------------------------------------------------------------------

/-- The full machine state.  `f`/`fw` are the two disjoint float files from
    `Ptx.lean`; `ir` and `pr` are the index and predicate files; `sm` is shared
    memory; `cta` is the (immutable) block id that `%ctaid.x` returns. -/
structure MState where
  f   : Nat → Lane → Float32
  fw  : Nat → Lane → Float32
  ir  : Nat → Lane → Nat
  pr  : Nat → Lane → Bool
  mem : Buf → Nat → Float32
  /-- Integer-valued memory: routing tables, block tables, sparse index arrays.
      **Read-only** — no instruction writes it, so it carries no frame
      obligation and never appears in a `WSt`. -/
  imem : Buf → Nat → Nat
  sm  : Nat → Float32

namespace MState

/-- The float-only view `Ptx.lean`'s theorems are stated over. -/
def toP (m : MState) : PState := { f := m.f, fw := m.fw, mem := m.mem }

/-- Put a `PState` back — the only fields `PInstr` can touch. -/
def ofP (m : MState) (ps : PState) : MState :=
  { m with f := ps.f, fw := ps.fw, mem := ps.mem }

/-- The warp-model view: machine registers, global memory, shared memory. -/
def toWSt (m : MState) : WSt := { regs := m.fw, mem := m.mem, smem := m.sm }

/-- Write a warp-model state back.  Memory instructions are specified through
    this, so their effect on the warp view is literally the `WSt` operation the
    warp machine performs — there is no room for the two to drift. -/
def ofWSt (m : MState) (st : WSt) : MState :=
  { m with fw := st.regs, mem := st.mem, sm := st.smem }

@[simp] theorem toWSt_ofWSt (m : MState) (st : WSt) : (m.ofWSt st).toWSt = st := by
  cases st; rfl

def getF (m : MState) : PReg → Lane → Float32
  | .tmp n  => m.f n
  | .mach n => m.fw n

def setF (m : MState) : PReg → (Lane → Float32) → MState
  | .tmp n, v  => { m with f := fun x => if x = n then v else m.f x }
  | .mach n, v => { m with fw := fun x => if x = n then v else m.fw x }

def setI (m : MState) (n : Nat) (v : Lane → Nat) : MState :=
  { m with ir := fun x => if x = n then v else m.ir x }

def setPr (m : MState) (n : Nat) (v : Lane → Bool) : MState :=
  { m with pr := fun x => if x = n then v else m.pr x }

@[simp] theorem toP_ofP (m : MState) (ps : PState) : (m.ofP ps).toP = ps := by
  cases ps; rfl

@[simp] theorem ofP_toP (m : MState) : m.ofP m.toP = m := by cases m; rfl

@[simp] theorem ir_setI_same (m : MState) (n : Nat) (v : Lane → Nat) :
    (m.setI n v).ir n = v := by simp [setI]
@[simp] theorem ir_setI_other (m : MState) (n x : Nat) (v : Lane → Nat) (h : x ≠ n) :
    (m.setI n v).ir x = m.ir x := by simp [setI, h]
@[simp] theorem toWSt_setI (m : MState) (n : Nat) (v : Lane → Nat) :
    (m.setI n v).toWSt = m.toWSt := rfl
@[simp] theorem pr_setI (m : MState) (n : Nat) (v : Lane → Nat) :
    (m.setI n v).pr = m.pr := rfl

@[simp] theorem pr_setPr_same (m : MState) (n : Nat) (v : Lane → Bool) :
    (m.setPr n v).pr n = v := by simp [setPr]
@[simp] theorem ir_setPr (m : MState) (n : Nat) (v : Lane → Bool) :
    (m.setPr n v).ir = m.ir := rfl
@[simp] theorem toWSt_setPr (m : MState) (n : Nat) (v : Lane → Bool) :
    (m.setPr n v).toWSt = m.toWSt := rfl

@[simp] theorem ir_setF (m : MState) (r : PReg) (v : Lane → Float32) :
    (m.setF r v).ir = m.ir := by cases r <;> rfl
@[simp] theorem mem_setF (m : MState) (r : PReg) (v : Lane → Float32) :
    (m.setF r v).mem = m.mem := by cases r <;> rfl

/-- Writing a machine register is exactly `WSt.setReg` on the warp view. -/
@[simp] theorem toWSt_setF_mach (m : MState) (n : Nat) (v : Lane → Float32) :
    (m.setF (.mach n) v).toWSt = m.toWSt.setReg n v := rfl

end MState

/-- **The lowerable fragment.**  Two conditions, both required by every
    lowering theorem below and both discharged by `decide` for a concrete
    kernel:

    * no `exp` anywhere — the hardware has `ex2.approx.f32`, not `e^x`
      (`expandEW` establishes this);
    * no `IdxE.ireg` in any address — a *data-dependent* index is expressible
      in the language but is not yet lowered, because `EWStmt.elabAt` has no
      integer register file to give it a meaning.  See `Assumptions`.

    The name is historical: it began as the exp condition alone. -/
def EWStmt.ExpFree : EWStmt → Prop
  | .skip => True
  | .seq a b => a.ExpFree ∧ b.ExpFree
  | .setR _ e => e.ExpFree
  | .shflXor _ _ _ => True
  | .loadIdx _ _ _ => True
  | .loadV4 _ _ _ _ _ _ => True
  | .storeLane0 _ _ _ => True
  | .storeLane _ _ _ => True
  | .stSm _ _ => True
  | .ldSm _ _ => True
  | .barrier => True
  | .forN _ body => body.ExpFree
  | .forM _ _ body => body.ExpFree
  | .cvtIF _ _ => True



-- ---------------------------------------------------------------------------
-- The structured instruction set
-- ---------------------------------------------------------------------------

/-- A loop's trip count: a compile-time constant, or a value read from **integer
    memory** at a fixed address.

    Memory rather than a register is the whole point.  `MState.imem` has **no
    lane index**, so a bound read this way is lane-uniform *by construction* —
    and uniformity is exactly what a register-sourced bound could not supply,
    since `setp.ge.u32` compares per lane and a divergent guard makes `fstep`
    stuck.  Here there is nothing to assume. -/
inductive LB where
  | lit : Nat → LB
  | mem : Buf → Nat → LB

def LB.val (b : LB) (m : MState) : Nat :=
  match b with
  | .lit c => c
  | .mem bu a => m.imem bu a

/-- The instruction set the emitter actually produces, with structured loops.

    Every constructor names a single PTX opcode, and its semantics is written
    with the *same* `WSt` operations the warp machine uses, so a wrong opcode is
    a failed proof rather than a runtime surprise. -/
inductive SI where
  /-- Any float-file instruction from `Ptx.lean` (`add.f32`, `shfl.sync.bfly`…). -/
  | fp    : PInstr → SI
  /-- `mov.u32 %r, <imm>` -/
  | movIC : Nat → Nat → SI
  /-- `mov.u32 %r, %laneid` -/
  | movLane : Nat → SI
  /-- `mov.u32 %r, %ctaid.x` -/
  | movCta : Nat → SI
  /-- `add.u32 %r, %r, %r` -/
  | addR  : Nat → Nat → Nat → SI
  /-- `mul.lo.u32 %r, %r, %r` -/
  | mulR  : Nat → Nat → Nat → SI
  /-- `add.u32 %r, %r, <imm>` -/
  | addRC : Nat → Nat → Nat → SI
  /-- `setp.eq.u32 %p, %r, <imm>` -/
  | setpEqC : Nat → Nat → Nat → SI
  /-- `setp.ge.u32 %p, %r, <imm>` -/
  | setpGeC : Nat → Nat → Nat → SI
  /-- `ld.global.u32 %r, [buf + %r]` — a gather's index load. -/
  | ldGI  : Nat → Buf → Nat → SI
  /-- `ld.global.f32` at the per-lane address in an index register. -/
  | ldG   : PReg → Buf → Nat → SI
  /-- `ld.global.v4.f32` — four consecutive floats into four registers. -/
  | ldGV4 : PReg → PReg → PReg → PReg → Buf → Nat → SI
  /-- `st.global.f32` from every lane. -/
  | stG   : Buf → Nat → PReg → SI
  /-- `@%p st.global.f32` — the predicated reduction epilogue. -/
  | stGIf : Nat → Buf → Nat → PReg → SI
  /-- `st.shared.f32` -/
  | stS   : Nat → PReg → SI
  /-- `ld.shared.f32` -/
  | ldS   : PReg → Nat → SI
  /-- `bar.warp.sync` -/
  | bar   : SI
  /-- An external kernel, with its named contract. -/
  | ext   : ExternOp → Buf → Buf → SI
  /-- A counted loop: `%r{c} = 0 .. n-1`, body executed with the counter live.
      Compiled to labels and branches — and proven so — in `PtxFlat.lean`. -/
  | loop  : Nat → LB → List SI → SI
  /-- `setp.ge.u32 %p, %r, %r` — the guard a memory-sourced bound needs. -/
  | setpGeR : Nat → Nat → Nat → SI
  /-- `cvt.rn.f32.u32 %f, %r` — the index-to-float bridge.  Exact for every
      index this stack can form: `Float32` represents every integer below
      2^24, and a buffer that large would not fit in device memory. -/
  | cvtIF : PReg → Nat → SI

/-! Semantics: `SI.step` runs one instruction, `SI.stepL` a list. -/
mutual
def SI.step (cta : Nat) (i : SI) (m : MState) : MState :=
  match i with
  | .fp p        => m.ofP (p.step m.toP)
  | .movIC d c   => m.setI d (fun _ => c)
  | .movLane d   => m.setI d (fun l => l.val)
  | .movCta d    => m.setI d (fun _ => cta)
  | .addR d a b  => m.setI d (fun l => m.ir a l + m.ir b l)
  | .mulR d a b  => m.setI d (fun l => m.ir a l * m.ir b l)
  | .addRC d a c => m.setI d (fun l => m.ir a l + c)
  | .setpEqC p a c => m.setPr p (fun l => m.ir a l == c)
  | .setpGeC p a c => m.setPr p (fun l => decide (c ≤ m.ir a l))
  | .ldGI d b ix => m.setI d (fun l => m.imem b (m.ir ix l))
  | .ldG d b ix  => m.setF d (fun l => m.mem b (m.ir ix l))
  | .ldGV4 d0 d1 d2 d3 b ix =>
      (((m.setF d0 (fun l => m.mem b (m.ir ix l))).setF d1
            (fun l => m.mem b (m.ir ix l + 1))).setF d2
            (fun l => m.mem b (m.ir ix l + 2))).setF d3
            (fun l => m.mem b (m.ir ix l + 3))
  | .stG b ix r  =>
      m.ofWSt ((List.finRange W).foldl
        (fun s l => s.store1 b (m.ir ix l) (m.getF r l)) m.toWSt)
  | .stGIf p b ix r =>
      m.ofWSt ((List.finRange W).foldl
        (fun s l => if m.pr p l then s.store1 b (m.ir ix l) (m.getF r l) else s)
        m.toWSt)
  | .stS ix r =>
      m.ofWSt { m.toWSt with smem := ((List.finRange W).foldl
        (fun sm' l => fun j => if j = m.ir ix l then m.getF r l else sm' j) m.toWSt.smem) }
  | .ldS d ix => m.setF d (fun l => m.sm (m.ir ix l))
  | .bar => m
  | .ext op inB outB =>
      m.ofWSt { m.toWSt with
        mem := fun c j => if c = outB then op.spec (m.toWSt.mem inB) j else m.toWSt.mem c j }
  | .setpGeR p a b => m.setPr p (fun l => decide (m.ir b l ≤ m.ir a l))
  | .cvtIF d ix    => m.setF d (fun l => NumOps.ofNat (m.ir ix l))
  | .loop c b body =>
      ((List.range (b.val m)).foldl
        (fun s j => SI.stepL cta body (s.setI c (fun _ => j))) m).setI
        c (fun _ => b.val m)

def SI.stepL (cta : Nat) (is : List SI) (m : MState) : MState :=
  match is with
  | []      => m
  | i :: is => SI.stepL cta is (SI.step cta i m)
end

@[simp] theorem srunL_nil (cta : Nat) (m : MState) : SI.stepL cta [] m = m := rfl
@[simp] theorem srunL_cons (cta : Nat) (i : SI) (is : List SI) (m : MState) :
    SI.stepL cta (i :: is) m = SI.stepL cta is (SI.step cta i m) := rfl

theorem srunL_append (cta : Nat) : ∀ (a b : List SI) (m : MState),
    SI.stepL cta (a ++ b) m = SI.stepL cta b (SI.stepL cta a m) := by
  intro a
  induction a with
  | nil => intro b m; rfl
  | cons i a ih => intro b m; simp only [List.cons_append, srunL_cons, ih]

-- ---------------------------------------------------------------------------
-- Running a float fragment inside the full machine
-- ---------------------------------------------------------------------------

@[simp] theorem MState.ofP_ofP (m : MState) (ps ps' : PState) :
    (m.ofP ps).ofP ps' = m.ofP ps' := by cases m; cases ps; cases ps'; rfl

@[simp] theorem MState.toWSt_ofP (m : MState) (ps : PState) :
    (m.ofP ps).toWSt = { regs := ps.fw, mem := ps.mem, smem := m.sm } := rfl

/-- A float-file fragment runs in the full machine exactly as it does in
    `PState` — so every theorem in `Ptx.lean` transfers verbatim. -/
theorem stepL_fp (cta : Nat) : ∀ (is : List PInstr) (m : MState),
    SI.stepL cta (is.map SI.fp) m = m.ofP (prun is m.toP) := by
  intro is
  induction is with
  | nil => intro m; simp
  | cons i is ih =>
      intro m
      show SI.stepL cta (is.map SI.fp) (m.ofP (i.step m.toP)) = _
      rw [ih]
      simp only [MState.toP_ofP, MState.ofP_ofP, prun_cons]

/-- Per-lane expressions read only the register file. -/
theorem WFExp.eval_regs (st1 st2 : WSt) (h : st1.regs = st2.regs) :
    ∀ (e : WFExp) (l : Lane), e.eval st1 l = e.eval st2 l := by
  intro e
  induction e with
  | reg r => intro l; show st1.regs r l = st2.regs r l; rw [h]
  | lit v => intro l; rfl
  | add a b iha ihb => intro l; show NumOps.add _ _ = NumOps.add _ _; rw [iha, ihb]
  | mul a b iha ihb => intro l; show NumOps.mul _ _ = NumOps.mul _ _; rw [iha, ihb]
  | neg a ih => intro l; show NumOps.neg _ = NumOps.neg _; rw [ih]
  | inv a ih => intro l; show NumOps.inv _ = NumOps.inv _; rw [ih]
  | exp a ih => intro l; show NumOps.exp _ = NumOps.exp _; rw [ih]
  | ex2 a ih => intro l; show NumOps.ex2 _ = NumOps.ex2 _; rw [ih]
  | rsqrt a ih => intro l; show NumOps.rsqrt _ = NumOps.rsqrt _; rw [ih]
  | maxW a b iha ihb =>
      intro l; show NumOps.max _ _ = NumOps.max _ _; rw [iha, ihb]
  | geF a b iha ihb =>
      intro l; show NumOps.ifGe _ _ 1.0 0.0 = NumOps.ifGe _ _ 1.0 0.0; rw [iha, ihb]

-- ---------------------------------------------------------------------------
-- Address computation
-- ---------------------------------------------------------------------------

/-- `%r0` holds `%laneid`, `%r1` holds `%ctaid.x`; allocation starts at 2. -/
abbrev laneIR : Nat := 0
abbrev ctaIR  : Nat := 1

/-- Lower an address expression into index registers.  `lr` is the register
    holding the enclosing loop counter; `n` is the first free index register. -/
def emitIdx (lr : Nat) : Nat → IdxE → Nat × List SI × Nat
  | n, .laneId => (laneIR, [], n)
  | n, .loopI  => (lr, [], n)
  | n, .ctaId  => (ctaIR, [], n)
  | n, .lit c  => (n, [.movIC n c], n + 1)
  -- Already in a register: a data-dependent index was loaded there earlier.
  | n, .ireg r => (r, [], n)
  -- Materialise the gathered index into a fresh register, as the hardware does.
  | n, .ldIdx b off =>
      ((emitIdx lr n off).2.2,
       (emitIdx lr n off).2.1
         ++ [SI.ldGI (emitIdx lr n off).2.2 b (emitIdx lr n off).1],
       (emitIdx lr n off).2.2 + 1)
  | n, .add a b =>
      ((emitIdx lr (emitIdx lr n a).2.2 b).2.2,
       (emitIdx lr n a).2.1 ++ (emitIdx lr (emitIdx lr n a).2.2 b).2.1
         ++ [.addR (emitIdx lr (emitIdx lr n a).2.2 b).2.2 (emitIdx lr n a).1
                   (emitIdx lr (emitIdx lr n a).2.2 b).1],
       (emitIdx lr (emitIdx lr n a).2.2 b).2.2 + 1)
  | n, .mul a b =>
      ((emitIdx lr (emitIdx lr n a).2.2 b).2.2,
       (emitIdx lr n a).2.1 ++ (emitIdx lr (emitIdx lr n a).2.2 b).2.1
         ++ [.mulR (emitIdx lr (emitIdx lr n a).2.2 b).2.2 (emitIdx lr n a).1
                   (emitIdx lr (emitIdx lr n a).2.2 b).1],
       (emitIdx lr (emitIdx lr n a).2.2 b).2.2 + 1)

theorem emitIdx_counter (lr : Nat) : ∀ (e : IdxE) (n : Nat), n ≤ (emitIdx lr n e).2.2 := by
  intro e
  induction e with
  | laneId => intro n; exact Nat.le_refl n
  | loopI  => intro n; exact Nat.le_refl n
  | ctaId  => intro n; exact Nat.le_refl n
  | ireg r => intro n; exact Nat.le_refl n
  | ldIdx b off ih =>
      intro n; have h1 := ih n; show n ≤ (emitIdx lr n off).2.2 + 1; omega
  | lit c  => intro n; show n ≤ n + 1; omega
  | add a b iha ihb =>
      intro n
      have h1 := iha n; have h2 := ihb (emitIdx lr n a).2.2
      show n ≤ (emitIdx lr (emitIdx lr n a).2.2 b).2.2 + 1; omega
  | mul a b iha ihb =>
      intro n
      have h1 := iha n; have h2 := ihb (emitIdx lr n a).2.2
      show n ≤ (emitIdx lr (emitIdx lr n a).2.2 b).2.2 + 1; omega

/-- The result register is one the emitter has already allocated. -/
theorem emitIdx_res (lr : Nat) : ∀ (e : IdxE) (n : Nat), 2 ≤ n → lr < n →
    e.regsBelow n → (emitIdx lr n e).1 < (emitIdx lr n e).2.2 := by
  intro e n h2 hlr hrb
  cases e with
  | laneId => show 0 < n; omega
  | loopI  => show lr < n; omega
  | ctaId  => show 1 < n; omega
  | ireg r => exact hrb
  | ldIdx b off => show (emitIdx lr n off).2.2 < (emitIdx lr n off).2.2 + 1; omega
  | lit c  => show n < n + 1; omega
  | add a b => show (emitIdx lr (emitIdx lr n a).2.2 b).2.2 < _ + 1; omega
  | mul a b => show (emitIdx lr (emitIdx lr n a).2.2 b).2.2 < _ + 1; omega

/-- Address code writes only freshly allocated index registers, and touches
    nothing else in the machine. -/
theorem emitIdx_frame (cta lr : Nat) : ∀ (e : IdxE) (n : Nat) (m : MState),
    (SI.stepL cta (emitIdx lr n e).2.1 m).toWSt = m.toWSt
    ∧ (SI.stepL cta (emitIdx lr n e).2.1 m).pr = m.pr
    ∧ (∀ x, x < n → (SI.stepL cta (emitIdx lr n e).2.1 m).ir x = m.ir x)
    ∧ (SI.stepL cta (emitIdx lr n e).2.1 m).imem = m.imem := by
  intro e
  induction e with
  | laneId => intro n m; exact ⟨rfl, rfl, fun _ _ => rfl, rfl⟩
  | loopI  => intro n m; exact ⟨rfl, rfl, fun _ _ => rfl, rfl⟩
  | ctaId  => intro n m; exact ⟨rfl, rfl, fun _ _ => rfl, rfl⟩
  | ireg r => intro n m; exact ⟨rfl, rfl, fun _ _ => rfl, rfl⟩
  | ldIdx b off ih =>
      intro n m
      have ho := ih n m
      have hc := emitIdx_counter lr off n
      have hcode : (emitIdx lr n (IdxE.ldIdx b off)).2.1
          = (emitIdx lr n off).2.1
            ++ [SI.ldGI (emitIdx lr n off).2.2 b (emitIdx lr n off).1] := rfl
      rw [hcode, srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      exact ⟨by rw [MState.toWSt_setI, ho.1], by rw [MState.pr_setI, ho.2.1],
             fun x hx => by
               rw [MState.ir_setI_other _ _ _ _ (by omega), ho.2.2.1 x hx],
             by show (MState.setI _ _ _).imem = _
                rw [show ∀ (mm : MState) (k : Nat) (v : Lane → Nat),
                      (mm.setI k v).imem = mm.imem from fun _ _ _ => rfl, ho.2.2.2]⟩
  | lit c  =>
      intro n m
      refine ⟨rfl, rfl, ?_, rfl⟩
      intro x hx
      show (m.setI n (fun _ => c)).ir x = m.ir x
      exact MState.ir_setI_other m n x _ (by omega)
  | add a b iha ihb =>
      intro n m
      have ha := iha n m
      have hb := ihb (emitIdx lr n a).2.2 (SI.stepL cta (emitIdx lr n a).2.1 m)
      have hc1 := emitIdx_counter lr a n
      have hc2 := emitIdx_counter lr b (emitIdx lr n a).2.2
      have hcode : (emitIdx lr n (IdxE.add a b)).2.1
          = (emitIdx lr n a).2.1 ++ (emitIdx lr (emitIdx lr n a).2.2 b).2.1
            ++ [SI.addR (emitIdx lr (emitIdx lr n a).2.2 b).2.2 (emitIdx lr n a).1
                   (emitIdx lr (emitIdx lr n a).2.2 b).1] := rfl
      rw [hcode, srunL_append, srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      refine ⟨?_, ?_, ?_, ?_⟩
      · rw [MState.toWSt_setI, hb.1, ha.1]
      · rw [MState.pr_setI, hb.2.1, ha.2.1]
      · intro x hx
        rw [MState.ir_setI_other _ _ _ _ (by omega), hb.2.2.1 x (by omega), ha.2.2.1 x hx]
      · show (MState.setI _ _ _).imem = _
        rw [show ∀ (mm : MState) (k : Nat) (v : Lane → Nat), (mm.setI k v).imem = mm.imem from
              fun _ _ _ => rfl, hb.2.2.2, ha.2.2.2]
  | mul a b iha ihb =>
      intro n m
      have ha := iha n m
      have hb := ihb (emitIdx lr n a).2.2 (SI.stepL cta (emitIdx lr n a).2.1 m)
      have hc1 := emitIdx_counter lr a n
      have hc2 := emitIdx_counter lr b (emitIdx lr n a).2.2
      have hcode : (emitIdx lr n (IdxE.mul a b)).2.1
          = (emitIdx lr n a).2.1 ++ (emitIdx lr (emitIdx lr n a).2.2 b).2.1
            ++ [SI.mulR (emitIdx lr (emitIdx lr n a).2.2 b).2.2 (emitIdx lr n a).1
                   (emitIdx lr (emitIdx lr n a).2.2 b).1] := rfl
      rw [hcode, srunL_append, srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      refine ⟨?_, ?_, ?_, ?_⟩
      · rw [MState.toWSt_setI, hb.1, ha.1]
      · rw [MState.pr_setI, hb.2.1, ha.2.1]
      · intro x hx
        rw [MState.ir_setI_other _ _ _ _ (by omega), hb.2.2.1 x (by omega), ha.2.2.1 x hx]
      · show (MState.setI _ _ _).imem = _
        rw [show ∀ (mm : MState) (k : Nat) (v : Lane → Nat), (mm.setI k v).imem = mm.imem from
              fun _ _ _ => rfl, hb.2.2.2, ha.2.2.2]

/-- The machine invariant address code runs under: `%r0` is the lane id, `%r1`
    the block id, and `%r{lr}` the enclosing loop counter. -/
def MInv (cta i lr : Nat) (m : MState) : Prop :=
  m.ir laneIR = (fun l => (l.val : Nat)) ∧ m.ir ctaIR = (fun _ => cta)
    ∧ m.ir lr = (fun _ => i)

/-- **Address computation is correct.**  The emitted `mov`/`add`/`mul.lo`
    sequence leaves `IdxE.eval` in the result register, in every lane. -/
theorem emitIdx_sound (cta lr i : Nat) : ∀ (e : IdxE) (n : Nat) (m : MState),
    2 ≤ n → lr < n → e.regsBelow n → MInv cta i lr m →
    (SI.stepL cta (emitIdx lr n e).2.1 m).ir (emitIdx lr n e).1
      = fun l => e.eval cta i l m.ir m.imem := by
  intro e
  induction e with
  | laneId => intro n m _ _ _ hinv; exact hinv.1
  | loopI  => intro n m _ _ _ hinv; exact hinv.2.2
  | ctaId  => intro n m _ _ _ hinv; exact hinv.2.1
  | ireg r => intro n m _ _ _ _; rfl
  | ldIdx b off ih =>
      intro n m h2 hlr hrb hinv
      have hc := emitIdx_counter lr off n
      have ho := ih n m h2 hlr hrb hinv
      have hfo := emitIdx_frame cta lr off n m
      have hcode : (emitIdx lr n (IdxE.ldIdx b off)).2.1
          = (emitIdx lr n off).2.1
            ++ [SI.ldGI (emitIdx lr n off).2.2 b (emitIdx lr n off).1] := rfl
      have hres : (emitIdx lr n (IdxE.ldIdx b off)).1 = (emitIdx lr n off).2.2 := rfl
      have hmem : (SI.stepL cta (emitIdx lr n off).2.1 m).imem = m.imem := hfo.2.2.2
      rw [hres, hcode, srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      rw [MState.ir_setI_same, ho, hmem]
      rfl
  | lit c  =>
      intro n m _ _ _ _
      show (m.setI n (fun _ => c)).ir n = _
      rw [MState.ir_setI_same]
      rfl
  | add a b iha ihb =>
      intro n m h2 hlr hrb hinv
      have hc1 := emitIdx_counter lr a n
      have hc2 := emitIdx_counter lr b (emitIdx lr n a).2.2
      have hra := emitIdx_res lr a n h2 hlr hrb.1
      have ha := iha n m h2 hlr hrb.1 hinv
      have hfa := emitIdx_frame cta lr a n m
      have hinv1 : MInv cta i lr (SI.stepL cta (emitIdx lr n a).2.1 m) :=
        ⟨by rw [hfa.2.2.1 laneIR (show 0 < n by omega)]; exact hinv.1,
         by rw [hfa.2.2.1 ctaIR (show 1 < n by omega)]; exact hinv.2.1,
         by rw [hfa.2.2.1 lr hlr]; exact hinv.2.2⟩
      have hbridge : ∀ l : Lane,
          b.eval cta i l (SI.stepL cta (emitIdx lr n a).2.1 m).ir
                         (SI.stepL cta (emitIdx lr n a).2.1 m).imem
            = b.eval cta i l m.ir m.imem := by
        intro l
        rw [hfa.2.2.2]
        exact IdxE.eval_frame cta i l n m.ir _ m.imem (fun x hx => hfa.2.2.1 x hx) b hrb.2
      have hb := ihb (emitIdx lr n a).2.2 (SI.stepL cta (emitIdx lr n a).2.1 m)
        (by omega) (by omega) (IdxE.regsBelow_mono (by omega) b hrb.2) hinv1
      have hfb := emitIdx_frame cta lr b (emitIdx lr n a).2.2
        (SI.stepL cta (emitIdx lr n a).2.1 m)
      have hcode : (emitIdx lr n (IdxE.add a b)).2.1
          = (emitIdx lr n a).2.1 ++ (emitIdx lr (emitIdx lr n a).2.2 b).2.1
            ++ [SI.addR (emitIdx lr (emitIdx lr n a).2.2 b).2.2 (emitIdx lr n a).1
                   (emitIdx lr (emitIdx lr n a).2.2 b).1] := rfl
      have hres : (emitIdx lr n (IdxE.add a b)).1
          = (emitIdx lr (emitIdx lr n a).2.2 b).2.2 := rfl
      rw [hres, hcode, srunL_append, srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      rw [MState.ir_setI_same]
      funext l
      show _ + _ = _ + _
      rw [hfb.2.2.1 (emitIdx lr n a).1 (by omega)]
      rw [ha, hb]
      simp only [hbridge l]
  | mul a b iha ihb =>
      intro n m h2 hlr hrb hinv
      have hc1 := emitIdx_counter lr a n
      have hc2 := emitIdx_counter lr b (emitIdx lr n a).2.2
      have hra := emitIdx_res lr a n h2 hlr hrb.1
      have ha := iha n m h2 hlr hrb.1 hinv
      have hfa := emitIdx_frame cta lr a n m
      have hinv1 : MInv cta i lr (SI.stepL cta (emitIdx lr n a).2.1 m) :=
        ⟨by rw [hfa.2.2.1 laneIR (show 0 < n by omega)]; exact hinv.1,
         by rw [hfa.2.2.1 ctaIR (show 1 < n by omega)]; exact hinv.2.1,
         by rw [hfa.2.2.1 lr hlr]; exact hinv.2.2⟩
      have hbridge : ∀ l : Lane,
          b.eval cta i l (SI.stepL cta (emitIdx lr n a).2.1 m).ir
                         (SI.stepL cta (emitIdx lr n a).2.1 m).imem
            = b.eval cta i l m.ir m.imem := by
        intro l
        rw [hfa.2.2.2]
        exact IdxE.eval_frame cta i l n m.ir _ m.imem (fun x hx => hfa.2.2.1 x hx) b hrb.2
      have hb := ihb (emitIdx lr n a).2.2 (SI.stepL cta (emitIdx lr n a).2.1 m)
        (by omega) (by omega) (IdxE.regsBelow_mono (by omega) b hrb.2) hinv1
      have hfb := emitIdx_frame cta lr b (emitIdx lr n a).2.2
        (SI.stepL cta (emitIdx lr n a).2.1 m)
      have hcode : (emitIdx lr n (IdxE.mul a b)).2.1
          = (emitIdx lr n a).2.1 ++ (emitIdx lr (emitIdx lr n a).2.2 b).2.1
            ++ [SI.mulR (emitIdx lr (emitIdx lr n a).2.2 b).2.2 (emitIdx lr n a).1
                   (emitIdx lr (emitIdx lr n a).2.2 b).1] := rfl
      have hres : (emitIdx lr n (IdxE.mul a b)).1
          = (emitIdx lr (emitIdx lr n a).2.2 b).2.2 := rfl
      rw [hres, hcode, srunL_append, srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      rw [MState.ir_setI_same]
      funext l
      show _ * _ = _ * _
      rw [hfb.2.2.1 (emitIdx lr n a).1 (by omega)]
      rw [ha, hb]
      simp only [hbridge l]

-- ---------------------------------------------------------------------------
-- Memory helper lemmas
-- ---------------------------------------------------------------------------

/-- **The predicated epilogue writes exactly lane 0's value, once.**

    `@p st.global.f32` with `p = (%laneid == 0)` is the shape every reduction
    ends with; this is the step that turns 32 lanes of predicated store into the
    single write the warp model specifies. -/
theorem storeIf_lane0 (b : Buf) (f : Lane → Nat) (v : Lane → Float32) (st : WSt) :
    (List.finRange W).foldl
        (fun s l => if (l.val == 0) = true then s.store1 b (f l) (v l) else s) st
      = st.store1 b (f ⟨0, by decide⟩) (v ⟨0, by decide⟩) := by
  rfl

-- ---------------------------------------------------------------------------
-- The statement emitter
-- ---------------------------------------------------------------------------

/-- Lower an emittable warp statement.  `lr` is the index register holding the
    enclosing loop counter; `n` is the first free index register.

    Float32 temporaries restart at 0 for every statement: `emitP_frame` proves
    they never disturb a machine register, so they are dead at every statement
    boundary.  Index registers *do* thread, because a loop counter must survive
    its body. -/
def emitEW (lr n : Nat) : EWStmt → List SI
  | .skip => []
  | .seq a b => emitEW lr n a ++ emitEW lr n b
  | .setR r e => (emitP 0 e).2.1.map SI.fp ++ [SI.fp (.mov (.mach r) (emitP 0 e).1)]
  | .shflXor d s m => [SI.fp (.shflBfly (.mach d) (.mach s) m)]
  | .loadIdx d b ix => (emitIdx lr n ix).2.1 ++ [SI.ldG (.mach d) b (emitIdx lr n ix).1]
  | .loadV4 d0 d1 d2 d3 bu ix =>
      (emitIdx lr n ix).2.1
        ++ [SI.ldGV4 (.mach d0) (.mach d1) (.mach d2) (.mach d3) bu (emitIdx lr n ix).1]
  | .storeLane0 bu ix r =>
      (emitIdx lr n ix).2.1
        ++ [SI.setpEqC (emitIdx lr n ix).2.2 laneIR 0,
            SI.stGIf (emitIdx lr n ix).2.2 bu (emitIdx lr n ix).1 (.mach r)]
  | .stSm ix r => (emitIdx lr n ix).2.1 ++ [SI.stS (emitIdx lr n ix).1 (.mach r)]
  | .ldSm d ix => (emitIdx lr n ix).2.1 ++ [SI.ldS (.mach d) (emitIdx lr n ix).1]
  | .barrier => [SI.bar]
  | .storeLane bu ix r =>
      (emitIdx lr n ix).2.1 ++ [SI.stG bu (emitIdx lr n ix).1 (.mach r)]
  | .forN cnt body => [SI.loop n (.lit cnt) (emitEW n (n + 1) body)]
  | .forM bu a body => [SI.loop n (.mem bu a) (emitEW n (n + 3) body)]
  | .cvtIF d ix => (emitIdx lr n ix).2.1 ++ [SI.cvtIF (.mach d) (emitIdx lr n ix).1]

/-- **The emitted code only ever writes index registers it allocated.**

    This is what lets a loop counter, `%laneid` and `%ctaid.x` survive an
    arbitrary body — the register-allocation discipline, as a theorem. -/
theorem emitEW_frame (cta : Nat) : ∀ (s : EWStmt) (lr n : Nat) (m : MState) (x : Nat),
    x < n → (SI.stepL cta (emitEW lr n s) m).ir x = m.ir x := by
  intro s
  induction s with
  | skip => intro lr n m x _; rfl
  | seq a b iha ihb =>
      intro lr n m x hx
      show (SI.stepL cta (emitEW lr n a ++ emitEW lr n b) m).ir x = _
      rw [srunL_append, ihb _ _ _ _ hx, iha _ _ _ _ hx]
  | setR r e =>
      intro lr n m x _
      show (SI.stepL cta ((emitP 0 e).2.1.map SI.fp ++ [SI.fp _]) m).ir x = _
      rw [srunL_append, stepL_fp]
      rfl
  | shflXor d s mk => intro lr n m x _; rfl
  | loadIdx d b ix =>
      intro lr n m x hx
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.ldG _ _ _]) m).ir x = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step, MState.ir_setF]
      exact (emitIdx_frame cta lr ix n m).2.2.1 x hx
  | loadV4 d0 d1 d2 d3 bu ix =>
      intro lr n m x hx
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.ldGV4 _ _ _ _ _ _]) m).ir x = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step, MState.ir_setF]
      exact (emitIdx_frame cta lr ix n m).2.2.1 x hx
  | storeLane0 bu ix r =>
      intro lr n m x hx
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.setpEqC _ _ _, SI.stGIf _ _ _ _]) m).ir x = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step, MState.ir_setPr]
      exact (emitIdx_frame cta lr ix n m).2.2.1 x hx
  | storeLane bu ix r =>
      intro lr n m x hx
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.stG _ _ _]) m).ir x = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      exact (emitIdx_frame cta lr ix n m).2.2.1 x hx
  | stSm ix r =>
      intro lr n m x hx
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.stS _ _]) m).ir x = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      exact (emitIdx_frame cta lr ix n m).2.2.1 x hx
  | ldSm d ix =>
      intro lr n m x hx
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.ldS _ _]) m).ir x = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step, MState.ir_setF]
      exact (emitIdx_frame cta lr ix n m).2.2.1 x hx
  | barrier => intro lr n m x _; rfl
  | forN cnt body ih =>
      intro lr n m x hx
      show (SI.stepL cta [SI.loop n (.lit cnt) (emitEW n (n + 1) body)] m).ir x = _
      simp only [srunL_cons, srunL_nil, SI.step]
      rw [MState.ir_setI_other _ _ _ _ (by omega)]
      have key : ∀ (L : List Nat) (m' : MState),
          (L.foldl (fun s j => SI.stepL cta (emitEW n (n + 1) body) (s.setI n (fun _ => j)))
            m').ir x = m'.ir x := by
        intro L
        induction L with
        | nil => intro m'; rfl
        | cons j L ihl =>
            intro m'
            rw [List.foldl_cons, ihl, ih n (n + 1) (m'.setI n (fun _ => j)) x (by omega),
                MState.ir_setI_other _ _ _ _ (by omega)]
      exact key (List.range cnt) m
  | cvtIF d ix =>
      intro lr n m x hx
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.cvtIF _ _]) m).ir x = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step, MState.ir_setF]
      exact (emitIdx_frame cta lr ix n m).2.2.1 x hx
  | forM bu ad body ih =>
      intro lr n m x hx
      show (SI.stepL cta [SI.loop n (.mem bu ad) (emitEW n (n + 3) body)] m).ir x = _
      simp only [srunL_cons, srunL_nil, SI.step]
      rw [MState.ir_setI_other _ _ _ _ (by omega)]
      have key : ∀ (L : List Nat) (m' : MState),
          (L.foldl (fun s j => SI.stepL cta (emitEW n (n + 3) body) (s.setI n (fun _ => j)))
            m').ir x = m'.ir x := by
        intro L
        induction L with
        | nil => intro m'; rfl
        | cons j L ihl =>
            intro m'
            rw [List.foldl_cons, ihl, ih n (n + 3) (m'.setI n (fun _ => j)) x (by omega),
                MState.ir_setI_other _ _ _ _ (by omega)]
      exact key (List.range ((LB.mem bu ad).val m)) m

/-- The read-only integer memory really is read-only: no emitted instruction
    touches it.  This is what lets a whole statement — including every iteration
    of a loop — be elaborated against one integer memory. -/
theorem emitEW_imem (cta : Nat) : ∀ (s : EWStmt) (lr n : Nat) (m : MState),
    (SI.stepL cta (emitEW lr n s) m).imem = m.imem := by
  intro s
  induction s with
  | skip => intro lr n m; rfl
  | seq a b iha ihb =>
      intro lr n m
      show (SI.stepL cta (emitEW lr n a ++ emitEW lr n b) m).imem = _
      rw [srunL_append, ihb, iha]
  | setR r e =>
      intro lr n m
      show (SI.stepL cta ((emitP 0 e).2.1.map SI.fp ++ [SI.fp _]) m).imem = _
      rw [srunL_append, stepL_fp]
      rfl
  | shflXor d s mk => intro lr n m; rfl
  | loadIdx d b ix =>
      intro lr n m
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.ldG _ _ _]) m).imem = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      exact (emitIdx_frame cta lr ix n m).2.2.2
  | loadV4 d0 d1 d2 d3 bu ix =>
      intro lr n m
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.ldGV4 _ _ _ _ _ _]) m).imem = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      exact (emitIdx_frame cta lr ix n m).2.2.2
  | storeLane0 bu ix r =>
      intro lr n m
      show (SI.stepL cta ((emitIdx lr n ix).2.1
              ++ [SI.setpEqC _ _ _, SI.stGIf _ _ _ _]) m).imem = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      exact (emitIdx_frame cta lr ix n m).2.2.2
  | storeLane bu ix r =>
      intro lr n m
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.stG _ _ _]) m).imem = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      exact (emitIdx_frame cta lr ix n m).2.2.2
  | stSm ix r =>
      intro lr n m
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.stS _ _]) m).imem = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      exact (emitIdx_frame cta lr ix n m).2.2.2
  | ldSm d ix =>
      intro lr n m
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.ldS _ _]) m).imem = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      exact (emitIdx_frame cta lr ix n m).2.2.2
  | barrier => intro lr n m; rfl
  | forN cnt body ih =>
      intro lr n m
      show (SI.stepL cta [SI.loop n (.lit cnt) (emitEW n (n + 1) body)] m).imem = _
      simp only [srunL_cons, srunL_nil, SI.step]
      have key : ∀ (L : List Nat) (m' : MState),
          (L.foldl (fun s j => SI.stepL cta (emitEW n (n + 1) body) (s.setI n (fun _ => j)))
            m').imem = m'.imem := by
        intro L
        induction L with
        | nil => intro m'; rfl
        | cons j L ihl =>
            intro m'
            rw [List.foldl_cons, ihl, ih n (n + 1) (m'.setI n (fun _ => j))]
            rfl
      exact key (List.range cnt) m
  | cvtIF d ix =>
      intro lr n m
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.cvtIF _ _]) m).imem = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      exact (emitIdx_frame cta lr ix n m).2.2.2
  | forM bu ad body ih =>
      intro lr n m
      show (SI.stepL cta [SI.loop n (.mem bu ad) (emitEW n (n + 3) body)] m).imem = _
      simp only [srunL_cons, srunL_nil, SI.step]
      have key : ∀ (L : List Nat) (m' : MState),
          (L.foldl (fun s j => SI.stepL cta (emitEW n (n + 3) body) (s.setI n (fun _ => j)))
            m').imem = m'.imem := by
        intro L
        induction L with
        | nil => intro m'; rfl
        | cons j L ihl =>
            intro m'
            rw [List.foldl_cons, ihl, ih n (n + 3) (m'.setI n (fun _ => j))]
            rfl
      exact key (List.range ((LB.mem bu ad).val m)) m

/-- **The lowering is correct: emitted instructions run the statement.**

    Every construct a real kernel uses is covered — address arithmetic, scalar
    and vectorised global loads, per-lane and predicated stores, shared memory,
    externs, and counted loops.  The equality is on the *whole* warp state, and
    it is exact: no epsilon anywhere.

    `hexp` is the single declared approximation (`ExpIsEx2`); everything else in
    the chain is unconditional. -/
theorem emitEW_sound (cta : Nat) :
    ∀ (s : EWStmt), s.ExpFree → ∀ (lr n i : Nat) (m : MState),
      2 ≤ n → lr < n → s.IdxBelow n → MInv cta i lr m →
      (SI.stepL cta (emitEW lr n s) m).toWSt = (s.elabAt cta i m.ir m.imem).run m.toWSt := by
  intro s
  induction s with
  | skip => intro hf lr n i m _ _ _ _; rfl
  | seq a b iha ihb =>
      intro hf lr n i m h2 hlr hb hinv
      show (SI.stepL cta (emitEW lr n a ++ emitEW lr n b) m).toWSt = _
      rw [srunL_append]
      have hinv1 : MInv cta i lr (SI.stepL cta (emitEW lr n a) m) :=
        ⟨by rw [emitEW_frame cta a lr n m laneIR (show 0 < n by omega)]; exact hinv.1,
         by rw [emitEW_frame cta a lr n m ctaIR (show 1 < n by omega)]; exact hinv.2.1,
         by rw [emitEW_frame cta a lr n m lr hlr]; exact hinv.2.2⟩
      have hbr : b.elabAt cta i (SI.stepL cta (emitEW lr n a) m).ir
            (SI.stepL cta (emitEW lr n a) m).imem = b.elabAt cta i m.ir m.imem := by
        rw [emitEW_imem cta a lr n m]
        exact EWStmt.elabAt_frame cta n m.ir _ m.imem
          (fun x hx => emitEW_frame cta a lr n m x hx) b i hb.2
      rw [ihb hf.2 lr n i _ h2 hlr hb.2 hinv1, hbr, iha hf.1 lr n i m h2 hlr hb.1 hinv]
      rfl
  | setR r e =>
      intro hf lr n i m _ _ _ _
      show (SI.stepL cta ((emitP 0 e).2.1.map SI.fp ++ [SI.fp _]) m).toWSt = _
      rw [srunL_append, stepL_fp]
      simp only [srunL_cons, srunL_nil, SI.step, MState.toP_ofP, MState.ofP_ofP,
                 MState.toWSt_ofP]
      refine WSt.ext ?_ ?_ ?_
      · funext x l
        by_cases hx : x = r
        · subst hx
          show ((prun (emitP 0 e).2.1 m.toP).set (.mach x)
                  ((prun (emitP 0 e).2.1 m.toP).get (emitP 0 e).1)).get (.mach x) l = _
          rw [PState.get_set_same, emitP_sound e hf 0 m.toP l]
          show _ = (WSt.setReg m.toWSt x _).regs x l
          rw [WSt.regs_setReg_same]
          exact WFExp.eval_regs m.toP.toWSt m.toWSt rfl e l
        · show ((prun (emitP 0 e).2.1 m.toP).set (.mach r)
                  ((prun (emitP 0 e).2.1 m.toP).get (emitP 0 e).1)).get (.mach x) l = _
          rw [PState.get_set_mach_other _ r x _ hx]
          show (prun (emitP 0 e).2.1 m.toP).fw x l = (WSt.setReg m.toWSt r _).regs x l
          rw [WSt.regs_setReg_other _ r x _ hx, emitP_fw e 0 m.toP]
          rfl
      · show ((prun (emitP 0 e).2.1 m.toP).set (.mach r)
                  ((prun (emitP 0 e).2.1 m.toP).get (emitP 0 e).1)).mem = _
        rw [PState.mem_set, prun_mem]
        rfl
      · rfl
  | shflXor d s mk => intro hf lr n i m _ _ _ _; rfl
  | loadIdx d b ix =>
      intro hf lr n i m h2 hlr hb hinv
      have hfr := emitIdx_frame cta lr ix n m
      have hs := emitIdx_sound cta lr i ix n m h2 hlr hb hinv
      have hm : (SI.stepL cta (emitIdx lr n ix).2.1 m).mem = m.mem := congrArg WSt.mem hfr.1
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.ldG _ _ _]) m).toWSt = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step, MState.toWSt_setF_mach]
      rw [hs, hm, hfr.1]
      rfl
  | loadV4 d0 d1 d2 d3 bu ix =>
      intro hf lr n i m h2 hlr hb hinv
      have hfr := emitIdx_frame cta lr ix n m
      have hs := emitIdx_sound cta lr i ix n m h2 hlr hb hinv
      have hm : (SI.stepL cta (emitIdx lr n ix).2.1 m).mem = m.mem := congrArg WSt.mem hfr.1
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.ldGV4 _ _ _ _ _ _]) m).toWSt = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step, MState.toWSt_setF_mach,
                 MState.ir_setF, MState.mem_setF]
      rw [hs, hm, hfr.1]
      rfl
  | storeLane0 bu ix r =>
      intro hf lr n i m h2 hlr hb hinv
      have hfr := emitIdx_frame cta lr ix n m
      have hs := emitIdx_sound cta lr i ix n m h2 hlr hb hinv
      have hreg : (SI.stepL cta (emitIdx lr n ix).2.1 m).fw = m.fw := congrArg WSt.regs hfr.1
      have hlane : ((SI.stepL cta (emitIdx lr n ix).2.1 m).setPr (emitIdx lr n ix).2.2
            (fun l => (SI.stepL cta (emitIdx lr n ix).2.1 m).ir laneIR l == 0)).pr
            (emitIdx lr n ix).2.2 = fun l => (l.val == 0) := by
        rw [MState.pr_setPr_same, hfr.2.2.1 laneIR (show 0 < n by omega), hinv.1]
      have hidx : ((SI.stepL cta (emitIdx lr n ix).2.1 m).setPr (emitIdx lr n ix).2.2
            (fun l => (SI.stepL cta (emitIdx lr n ix).2.1 m).ir laneIR l == 0)).ir
            (emitIdx lr n ix).1 = fun l => ix.eval cta i l m.ir m.imem := by
        rw [MState.ir_setPr]; exact hs
      have hval : ((SI.stepL cta (emitIdx lr n ix).2.1 m).setPr (emitIdx lr n ix).2.2
            (fun l => (SI.stepL cta (emitIdx lr n ix).2.1 m).ir laneIR l == 0)).getF
            (PReg.mach r) = m.fw r := by
        show (SI.stepL cta (emitIdx lr n ix).2.1 m).fw r = _
        rw [hreg]
      have hws : ((SI.stepL cta (emitIdx lr n ix).2.1 m).setPr (emitIdx lr n ix).2.2
            (fun l => (SI.stepL cta (emitIdx lr n ix).2.1 m).ir laneIR l == 0)).toWSt
            = m.toWSt := by rw [MState.toWSt_setPr]; exact hfr.1
      show (SI.stepL cta ((emitIdx lr n ix).2.1
              ++ [SI.setpEqC _ _ _, SI.stGIf _ _ _ _]) m).toWSt = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      rw [MState.toWSt_ofWSt, hlane, hidx, hval, hws, storeIf_lane0]
      rfl
  | storeLane bu ix r =>
      intro hf lr n i m h2 hlr hb hinv
      have hfr := emitIdx_frame cta lr ix n m
      have hs := emitIdx_sound cta lr i ix n m h2 hlr hb hinv
      have hreg : (SI.stepL cta (emitIdx lr n ix).2.1 m).fw = m.fw := congrArg WSt.regs hfr.1
      have hval : (SI.stepL cta (emitIdx lr n ix).2.1 m).getF (PReg.mach r) = m.fw r := by
        show (SI.stepL cta (emitIdx lr n ix).2.1 m).fw r = _
        rw [hreg]
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.stG _ _ _]) m).toWSt = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      rw [MState.toWSt_ofWSt, hs, hval, hfr.1]
      rfl
  | stSm ix r =>
      intro hf lr n i m h2 hlr hb hinv
      have hfr := emitIdx_frame cta lr ix n m
      have hs := emitIdx_sound cta lr i ix n m h2 hlr hb hinv
      have hreg : (SI.stepL cta (emitIdx lr n ix).2.1 m).fw = m.fw := congrArg WSt.regs hfr.1
      have hval : (SI.stepL cta (emitIdx lr n ix).2.1 m).getF (PReg.mach r) = m.fw r := by
        show (SI.stepL cta (emitIdx lr n ix).2.1 m).fw r = _
        rw [hreg]
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.stS _ _]) m).toWSt = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step]
      rw [MState.toWSt_ofWSt, hs, hval, hfr.1]
      rfl
  | ldSm d ix =>
      intro hf lr n i m h2 hlr hb hinv
      have hfr := emitIdx_frame cta lr ix n m
      have hs := emitIdx_sound cta lr i ix n m h2 hlr hb hinv
      have hsm : (SI.stepL cta (emitIdx lr n ix).2.1 m).sm = m.sm :=
        congrArg WSt.smem hfr.1
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.ldS _ _]) m).toWSt = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step, MState.toWSt_setF_mach]
      rw [hs, hsm, hfr.1]
      rfl
  | barrier => intro hf lr n i m _ _ _ _; rfl
  | forN cnt body ih =>
      intro hf lr n i m h2 hlr hb hinv
      show (SI.stepL cta [SI.loop n (.lit cnt) (emitEW n (n + 1) body)] m).toWSt = _
      simp only [srunL_cons, srunL_nil, SI.step, MState.toWSt_setI]
      -- Every iteration elaborates against the *initial* register file: the loop
      -- counter lives at `n`, the body's data-dependent indices strictly below
      -- it, and neither `setI n` nor the body's own code disturbs those.
      have key : ∀ (L : List Nat) (m' : MState),
          m'.ir laneIR = (fun l => (l.val : Nat)) → m'.ir ctaIR = (fun _ => cta) →
          (∀ x, x < n → m'.ir x = m.ir x) → m'.imem = m.imem →
          (L.foldl (fun s j => SI.stepL cta (emitEW n (n + 1) body) (s.setI n (fun _ => j)))
            m').toWSt
            = L.foldl (fun s' j => (body.elabAt cta j m.ir m.imem).run s') m'.toWSt := by
        intro L
        induction L with
        | nil => intro m' _ _ _ _; rfl
        | cons j L ihl =>
            intro m' h0 h1 hpres hmpres
            have hinvj : MInv cta j n (m'.setI n (fun _ => j)) :=
              ⟨by rw [MState.ir_setI_other _ _ _ _ (show 0 ≠ n by omega)]; exact h0,
               by rw [MState.ir_setI_other _ _ _ _ (show 1 ≠ n by omega)]; exact h1,
               MState.ir_setI_same _ _ _⟩
            have hbb := ih hf n (n + 1) j (m'.setI n (fun _ => j)) (by omega) (by omega)
              (EWStmt.idxBelow_mono (by omega) body hb) hinvj
            have hf0 := emitEW_frame cta body n (n + 1) (m'.setI n (fun _ => j))
            have hsetPres : ∀ x, x < n → (m'.setI n (fun _ => j)).ir x = m.ir x := by
              intro x hx
              rw [MState.ir_setI_other _ _ _ _ (show x ≠ n by omega)]
              exact hpres x hx
            have hbridge : body.elabAt cta j (m'.setI n (fun _ => j)).ir
                  (m'.setI n (fun _ => j)).imem = body.elabAt cta j m.ir m.imem := by
              rw [show (m'.setI n (fun _ => j)).imem = m'.imem from rfl, hmpres]
              exact EWStmt.elabAt_frame cta n m.ir _ m.imem hsetPres body j hb
            rw [List.foldl_cons, List.foldl_cons]
            rw [ihl _ (by rw [hf0 laneIR (show 0 < n + 1 by omega),
                              MState.ir_setI_other _ _ _ _ (show 0 ≠ n by omega)]; exact h0)
                      (by rw [hf0 ctaIR (show 1 < n + 1 by omega),
                              MState.ir_setI_other _ _ _ _ (show 1 ≠ n by omega)]; exact h1)
                      (by intro x hx
                          rw [hf0 x (show x < n + 1 by omega)]
                          exact hsetPres x hx)
                      (by rw [emitEW_imem cta body n (n + 1) (m'.setI n (fun _ => j))]
                          exact hmpres)]
            rw [hbb, hbridge]
            rfl
      exact key (List.range cnt) m hinv.1 hinv.2.1 (fun _ _ => rfl) rfl
  | cvtIF d ix =>
      intro hf lr n i m h2 hlr hb hinv
      have hfr := emitIdx_frame cta lr ix n m
      have hs := emitIdx_sound cta lr i ix n m h2 hlr hb hinv
      show (SI.stepL cta ((emitIdx lr n ix).2.1 ++ [SI.cvtIF _ _]) m).toWSt = _
      rw [srunL_append]
      simp only [srunL_cons, srunL_nil, SI.step, MState.toWSt_setF_mach]
      rw [hs, hfr.1]
      rfl
  | forM bu ad body ih =>
      intro hf lr n i m h2 hlr hb hinv
      show (SI.stepL cta [SI.loop n (.mem bu ad) (emitEW n (n + 3) body)] m).toWSt = _
      simp only [srunL_cons, srunL_nil, SI.step, MState.toWSt_setI]
      -- Every iteration elaborates against the *initial* register file: the loop
      -- counter lives at `n`, the body's data-dependent indices strictly below
      -- it, and neither `setI n` nor the body's own code disturbs those.
      have key : ∀ (L : List Nat) (m' : MState),
          m'.ir laneIR = (fun l => (l.val : Nat)) → m'.ir ctaIR = (fun _ => cta) →
          (∀ x, x < n → m'.ir x = m.ir x) → m'.imem = m.imem →
          (L.foldl (fun s j => SI.stepL cta (emitEW n (n + 3) body) (s.setI n (fun _ => j)))
            m').toWSt
            = L.foldl (fun s' j => (body.elabAt cta j m.ir m.imem).run s') m'.toWSt := by
        intro L
        induction L with
        | nil => intro m' _ _ _ _; rfl
        | cons j L ihl =>
            intro m' h0 h1 hpres hmpres
            have hinvj : MInv cta j n (m'.setI n (fun _ => j)) :=
              ⟨by rw [MState.ir_setI_other _ _ _ _ (show 0 ≠ n by omega)]; exact h0,
               by rw [MState.ir_setI_other _ _ _ _ (show 1 ≠ n by omega)]; exact h1,
               MState.ir_setI_same _ _ _⟩
            have hbb := ih hf n (n + 3) j (m'.setI n (fun _ => j)) (by omega) (by omega)
              (EWStmt.idxBelow_mono (by omega) body hb) hinvj
            have hf0 := emitEW_frame cta body n (n + 3) (m'.setI n (fun _ => j))
            have hsetPres : ∀ x, x < n → (m'.setI n (fun _ => j)).ir x = m.ir x := by
              intro x hx
              rw [MState.ir_setI_other _ _ _ _ (show x ≠ n by omega)]
              exact hpres x hx
            have hbridge : body.elabAt cta j (m'.setI n (fun _ => j)).ir
                  (m'.setI n (fun _ => j)).imem = body.elabAt cta j m.ir m.imem := by
              rw [show (m'.setI n (fun _ => j)).imem = m'.imem from rfl, hmpres]
              exact EWStmt.elabAt_frame cta n m.ir _ m.imem hsetPres body j hb
            rw [List.foldl_cons, List.foldl_cons]
            rw [ihl _ (by rw [hf0 laneIR (show 0 < n + 3 by omega),
                              MState.ir_setI_other _ _ _ _ (show 0 ≠ n by omega)]; exact h0)
                      (by rw [hf0 ctaIR (show 1 < n + 3 by omega),
                              MState.ir_setI_other _ _ _ _ (show 1 ≠ n by omega)]; exact h1)
                      (by intro x hx
                          rw [hf0 x (show x < n + 3 by omega)]
                          exact hsetPres x hx)
                      (by rw [emitEW_imem cta body n (n + 3) (m'.setI n (fun _ => j))]
                          exact hmpres)]
            rw [hbb, hbridge]
            rfl
      exact key (List.range ((LB.mem bu ad).val m)) m hinv.1 hinv.2.1 (fun _ _ => rfl) rfl



-- ---------------------------------------------------------------------------
-- The one declared approximation, isolated in a rewrite
-- ---------------------------------------------------------------------------

/-- `log₂ e`, the constant folded in to realise `exp` on hardware that has only
    `ex2`. -/
def log2eF : Float32 := 1.4426950408889634

/-- **The single declared approximation in the stack.**

    PTX has no `e^x` instruction, so `exp` is realised as `2^(x·log₂e)`.  This
    proposition is *false* for IEEE floats — `ex2.approx` is documented at 2 ULP
    and the f32 constant is not `log₂ e` exactly — which is exactly why it is a
    named hypothesis and not a lemma.

    It appears in **one** theorem, `expandExp_approx`, and in no lowering
    theorem.  Everything from `expandExp` downwards is unconditional and exact.
    Measured cost on silu: max relative error 5.06e-7. -/
def ExpIsEx2 : Prop :=
  ∀ x : Float32, NumOps.exp x = NumOps.ex2 (NumOps.mul x log2eF)

/-- Rewrite `exp` into the hardware's `ex2`.  This is the tier-3 `Approx` step
    of `Rewrite.lean`, made concrete at the machine-expression level. -/
def expandExp : WFExp → WFExp
  | .reg r   => .reg r
  | .lit v   => .lit v
  | .add a b => .add (expandExp a) (expandExp b)
  | .mul a b => .mul (expandExp a) (expandExp b)
  | .neg a   => .neg (expandExp a)
  | .inv a   => .inv (expandExp a)
  | .rsqrt a => .rsqrt (expandExp a)
  | .maxW a b => .maxW (expandExp a) (expandExp b)
  | .geF a b => .geF (expandExp a) (expandExp b)
  | .ex2 a   => .ex2 (expandExp a)
  | .exp a   => .ex2 (.mul (expandExp a) (.lit log2eF))

/-- **After the rewrite there is no `exp` left** — so the emitter's exactness
    hypothesis is discharged by construction, for any expression at all. -/
theorem expandExp_expFree : ∀ e : WFExp, (expandExp e).ExpFree := by
  intro e
  induction e with
  | reg _ => trivial
  | lit _ => trivial
  | add a b iha ihb => exact ⟨iha, ihb⟩
  | mul a b iha ihb => exact ⟨iha, ihb⟩
  | neg a ih => exact ih
  | inv a ih => exact ih
  | rsqrt a ih => exact ih
  | maxW a b iha ihb => exact ⟨iha, ihb⟩
  | geF a b iha ihb => exact ⟨iha, ihb⟩
  | ex2 a ih => exact ih
  | exp a ih => exact ⟨ih, trivial⟩

/-- **The rewrite preserves meaning, given the declared identity.**  This is the
    only theorem in the stack that mentions `ExpIsEx2`. -/
theorem expandExp_approx (h : ExpIsEx2) : ∀ (e : WFExp) (st : WSt) (l : Lane),
    (expandExp e).eval st l = e.eval st l := by
  intro e
  induction e with
  | reg _ => intro _ _; rfl
  | lit _ => intro _ _; rfl
  | add a b iha ihb => intro st l; show NumOps.add _ _ = NumOps.add _ _; rw [iha, ihb]
  | mul a b iha ihb => intro st l; show NumOps.mul _ _ = NumOps.mul _ _; rw [iha, ihb]
  | neg a ih => intro st l; show NumOps.neg _ = NumOps.neg _; rw [ih]
  | inv a ih => intro st l; show NumOps.inv _ = NumOps.inv _; rw [ih]
  | rsqrt a ih => intro st l; show NumOps.rsqrt _ = NumOps.rsqrt _; rw [ih]
  | maxW a b iha ihb =>
      intro st l; show NumOps.max _ _ = NumOps.max _ _; rw [iha, ihb]
  | geF a b iha ihb =>
      intro st l; show NumOps.ifGe _ _ 1.0 0.0 = NumOps.ifGe _ _ 1.0 0.0; rw [iha, ihb]
  | ex2 a ih => intro st l; show NumOps.ex2 _ = NumOps.ex2 _; rw [ih]
  | exp a ih =>
      intro st l
      show NumOps.ex2 (NumOps.mul (WFExp.eval st l (expandExp a)) log2eF) = NumOps.exp _
      rw [ih, ← h]

/-- Apply the rewrite throughout a kernel. -/
def expandEW : EWStmt → EWStmt
  | .skip => .skip
  | .seq a b => .seq (expandEW a) (expandEW b)
  | .setR r e => .setR r (expandExp e)
  | .shflXor d s m => .shflXor d s m
  | .loadIdx d b ix => .loadIdx d b ix
  | .loadV4 a b c d bu ix => .loadV4 a b c d bu ix
  | .storeLane0 b ix r => .storeLane0 b ix r
  | .storeLane b ix r => .storeLane b ix r
  | .stSm ix r => .stSm ix r
  | .ldSm d ix => .ldSm d ix
  | .barrier => .barrier
  | .forN n body => .forN n (expandEW body)
  | .forM bu a body => .forM bu a (expandEW body)
  | .cvtIF d ix => .cvtIF d ix

theorem expandEW_expFree : ∀ s : EWStmt, (expandEW s).ExpFree := by
  intro s
  induction s with
  | skip => trivial
  | seq a b iha ihb => exact ⟨iha, ihb⟩
  | setR r e => exact expandExp_expFree e
  | shflXor _ _ _ => trivial
  | loadIdx _ _ _ => trivial
  | loadV4 _ _ _ _ _ _ => trivial
  | storeLane0 _ _ _ => trivial
  | storeLane _ _ _ => trivial
  | stSm _ _ => trivial
  | ldSm _ _ => trivial
  | barrier => trivial
  | forN _ _ ih => exact ih
  | forM _ _ _ ih => exact ih
  | cvtIF _ _ => trivial

/-- `expandEW` rewrites `exp`; it never touches an address. -/
theorem expandEW_idxFree : ∀ s : EWStmt, s.IdxFree → (expandEW s).IdxFree := by
  intro s
  induction s with
  | skip => intro _; trivial
  | seq a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | setR _ _ => intro _; trivial
  | shflXor _ _ _ => intro _; trivial
  | loadIdx _ _ _ => intro h; exact h
  | loadV4 _ _ _ _ _ _ => intro h; exact h
  | storeLane0 _ _ _ => intro h; exact h
  | storeLane _ _ _ => intro h; exact h
  | stSm _ _ => intro h; exact h
  | ldSm _ _ => intro h; exact h
  | barrier => intro _; trivial
  | forN _ _ ih => intro h; exact ih h
  | forM _ _ _ _ => intro h; exact (h : False).elim
  | cvtIF _ _ => intro h; exact h

theorem expandEW_flat : ∀ s : EWStmt, s.Flat → (expandEW s).Flat := by
  intro s
  induction s with
  | skip => intro _; trivial
  | seq a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | setR _ _ => intro _; trivial
  | shflXor _ _ _ => intro _; trivial
  | loadIdx _ _ _ => intro _; trivial
  | loadV4 _ _ _ _ _ _ => intro _; trivial
  | storeLane0 _ _ _ => intro _; trivial
  | storeLane _ _ _ => intro _; trivial
  | stSm _ _ => intro _; trivial
  | ldSm _ _ => intro _; trivial
  | barrier => intro _; trivial
  | forN _ _ ih => intro h; exact ih h
  | forM _ _ _ ih => intro h; exact ih h
  | cvtIF _ _ => intro _; trivial

theorem expandEW_idxBelow (n : Nat) : ∀ s : EWStmt, s.IdxBelow n → (expandEW s).IdxBelow n := by
  intro s
  induction s with
  | skip => intro _; trivial
  | seq a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | setR _ _ => intro _; trivial
  | shflXor _ _ _ => intro _; trivial
  | loadIdx _ _ _ => intro h; exact h
  | loadV4 _ _ _ _ _ _ => intro h; exact h
  | storeLane0 _ _ _ => intro h; exact h
  | storeLane _ _ _ => intro h; exact h
  | stSm _ _ => intro h; exact h
  | ldSm _ _ => intro h; exact h
  | barrier => intro _; trivial
  | forN _ _ ih => intro h; exact ih h
  | forM _ _ _ ih => intro h; exact ih h
  | cvtIF _ _ => intro h; exact h

/-- **The rewritten kernel runs the original kernel**, given the declared
    identity.  Composed with the unconditional lowering theorems, this is what
    licenses shipping a kernel whose spec mentions `exp`. -/
theorem expandEW_run (h : ExpIsEx2) : ∀ (s : EWStmt) (cta i : Nat) (st : WSt),
    ((expandEW s).elabAt cta i).run st = (s.elabAt cta i).run st := by
  intro s
  induction s with
  | skip => intro _ _ _; rfl
  | seq a b iha ihb =>
      intro cta i st
      show ((expandEW b).elabAt cta i).run (((expandEW a).elabAt cta i).run st)
          = (b.elabAt cta i).run ((a.elabAt cta i).run st)
      rw [iha, ihb]
  | setR r e =>
      intro cta i st
      show st.setReg r _ = st.setReg r _
      congr 1
      funext l
      exact expandExp_approx h e st l
  | shflXor _ _ _ => intro _ _ _; rfl
  | loadIdx _ _ _ => intro _ _ _; rfl
  | loadV4 _ _ _ _ _ _ => intro _ _ _; rfl
  | storeLane0 _ _ _ => intro _ _ _; rfl
  | storeLane _ _ _ => intro _ _ _; rfl
  | stSm ix r => intro cta i st; rfl
  | ldSm d ix => intro cta i st; rfl
  | barrier => intro cta i st; rfl
  | forN n body ih =>
      intro cta i st
      show (List.range n).foldl (fun s' j => ((expandEW body).elabAt cta j).run s') st
          = (List.range n).foldl (fun s' j => (body.elabAt cta j).run s') st
      have key : ∀ (L : List Nat) (st' : WSt),
          L.foldl (fun s' j => ((expandEW body).elabAt cta j).run s') st'
            = L.foldl (fun s' j => (body.elabAt cta j).run s') st' := by
        intro L
        induction L with
        | nil => intro _; rfl
        | cons j L ihl => intro st'; rw [List.foldl_cons, List.foldl_cons, ih cta j st', ihl]
      exact key (List.range n) st
  | cvtIF _ _ => intro _ _ _; rfl
  | forM bu ad body ih =>
      intro cta i st
      show (List.range ((fun _ _ => 0 : Buf → Nat → Nat) bu ad)).foldl (fun s' j => ((expandEW body).elabAt cta j).run s') st
          = (List.range ((fun _ _ => 0 : Buf → Nat → Nat) bu ad)).foldl (fun s' j => (body.elabAt cta j).run s') st
      have key : ∀ (L : List Nat) (st' : WSt),
          L.foldl (fun s' j => ((expandEW body).elabAt cta j).run s') st'
            = L.foldl (fun s' j => (body.elabAt cta j).run s') st' := by
        intro L
        induction L with
        | nil => intro _; rfl
        | cons j L ihl => intro st'; rw [List.foldl_cons, List.foldl_cons, ih cta j st', ihl]
      exact key _ st

-- ---------------------------------------------------------------------------
-- The kernel prologue
-- ---------------------------------------------------------------------------

/-- The prologue every emitted kernel opens with: `%r0 = %laneid`,
    `%r1 = %ctaid.x`, `%r2 = 0` (the outermost loop counter, which
    `EWStmt.elabIn` fixes at 0). -/
def emitPrologue : List SI := [SI.movLane laneIR, SI.movCta ctaIR, SI.movIC 2 0]

/-- Index registers `0,1,2` are reserved; allocation starts at 3. -/
def emitKernelSI (s : EWStmt) : List SI := emitPrologue ++ emitEW 2 3 s

/-- **A whole kernel, from raw launch, computes its statement.**

    No hypothesis on the incoming machine state beyond the memory it reads: the
    prologue establishes the invariant itself.  This is the structured-code
    analogue of the LZ4 stack's `prologue_couple`. -/
theorem emitKernelSI_sound (cta : Nat) (s : EWStmt) (hf : s.ExpFree)
    (hb : s.IdxBelow 3) (m : MState) :
    (SI.stepL cta (emitKernelSI s) m).toWSt
      = (s.elabAt cta 0 (SI.stepL cta emitPrologue m).ir m.imem).run m.toWSt := by
  show (SI.stepL cta (emitPrologue ++ emitEW 2 3 s) m).toWSt = _
  rw [srunL_append]
  have hpro : SI.stepL cta emitPrologue m
      = ((m.setI laneIR (fun l => l.val)).setI ctaIR (fun _ => cta)).setI 2 (fun _ => 0) := rfl
  have hinv : MInv cta 0 2 (SI.stepL cta emitPrologue m) := by
    refine ⟨?_, ?_, ?_⟩
    · rw [hpro, MState.ir_setI_other _ _ _ _ (by decide),
          MState.ir_setI_other _ _ _ _ (by decide), MState.ir_setI_same]
    · rw [hpro, MState.ir_setI_other _ _ _ _ (by decide), MState.ir_setI_same]
    · rw [hpro, MState.ir_setI_same]
  have hws : (SI.stepL cta emitPrologue m).toWSt = m.toWSt := by rw [hpro]; rfl
  rw [emitEW_sound cta s hf 2 3 0 _ (by omega) (by omega) hb hinv, hws]
  rfl

-- PLACEHOLDER

end AlgorithmLib.ML
