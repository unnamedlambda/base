module
public import AlgorithmLib.Kernel.Ptx
public import AlgorithmLib.Kernel.PtxRun
public import AlgorithmLib.Host.Blocks
public import AlgorithmLib.Host.MemLaws
public import AlgorithmLib.Host.Heap
import all Init.Data.Repr
import all Init.Data.OfScientific
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Sound.PtxSound` — the PTX a body compiles to does what its blocks do

A kernel body is compiled to blocks (`compileBody`, proven by
`compile_sound`) and each block's instructions are lowered to PTX
(`Kernel.Ptx`). This proves the lowering keeps the blocks' meaning: the
registers hold the block run's values (`Sim`), an operand's lines leave its
value in the register they name (`r_sound`), a parallel move moves
(`steps_out`, `steps_in`), straight-line code, labels and branches run as
they are written (`go_steps`, `fold_label`, `go_bra`), and a block runs up to
its terminator as its instructions say (`prefix_go`), given that each
instruction meets its obligation (`InstOk`).
-/

namespace AlgorithmLib.Kernel.PtxSound

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem Ptx

/-- Running a lowering step from facts `F`, position `tag` and output `s`. -/
def runLM {α : Type} (m : LM α) (F : Facts) (tag : Nat) (s : Out) : α × Out := (m.run (F, tag)).run s

@[simp] theorem runLM_pure {α} (x : α) F tag s : runLM (pure x : LM α) F tag s = (x, s) := rfl
@[simp] theorem runLM_bind {α β} (m : LM α) (f : α → LM β) F tag s :
    runLM (m >>= f) F tag s = runLM (f (runLM m F tag s).1) F tag (runLM m F tag s).2 := rfl
@[simp] theorem runLM_map {α β} (f : α → β) (m : LM α) F tag s :
    runLM (f <$> m) F tag s = (f (runLM m F tag s).1, (runLM m F tag s).2) := rfl
@[simp] theorem runLM_read F tag s : runLM (read : LM (Facts × Nat)) F tag s = ((F, tag), s) := rfl
@[simp] theorem runLM_get F tag s : runLM (get : LM Out) F tag s = (s, s) := rfl
@[simp] theorem runLM_set (s' : Out) F tag s : runLM (set s' : LM Unit) F tag s = ((), s') := rfl
@[simp] theorem runLM_modify (g : Out → Out) F tag s : runLM (modify g : LM Unit) F tag s = ((), g s) := rfl
@[simp] theorem runLM_ite {α} (c : Prop) [Decidable c] (x y : LM α) F tag s :
    runLM (if c then x else y) F tag s = if c then runLM x F tag s else runLM y F tag s := by split <;> rfl

theorem lower_eq (F : Facts) (tag : Nat) (i : Inst) :
    lower F tag i = (runLM (if isTerm i then term i else inst i) F tag {}).2 := rfl

/-- An integer sum of two values that are neither constants, widenings nor
    windowed shared addresses lowers to the one `add`. -/
theorem lower_iadd (F : Facts) (tag : Nat) (d a b : Val) (t : ClifTy)
    (hta : F.tys.get? a.id = some t) (htb : F.tys.get? b.id = some t)
    (hwa : ¬ (F.window = true ∧ F.origin.get? a.id = some .shared))
    (hwb : ¬ (F.window = true ∧ F.origin.get? b.id = some .shared))
    (hla : F.lits.get? a.id = none) (hlb : F.lits.get? b.id = none)
    :
    (lower F tag (.iadd d a b)).lines =
      #[.ins none (.bin .add none (uTy t.width) (regOf t d) (.reg (regOf t a)) (.reg (regOf t b)))] := by
  simp only [Std.HashMap.get?_eq_getElem?] at *
  simp [lower_eq, isTerm, inst, isWin, Facts.win, tyOf, facts?, r, rw, o, imm, hta, htb, hla, hlb, hwa, hwb,
    emit, emitL]

/-! ## What the lines do -/

@[simp] theorem except_pure {α : Type} (x : α) : (pure x : Except String α) = .ok x := rfl
@[simp] theorem except_ok_bind {α β : Type} (x : α) (f : α → Except String β) :
    (Except.ok x >>= f) = f x := rfl
@[simp] theorem except_ok_map {α β : Type} (x : α) (f : α → β) :
    (f <$> (Except.ok x : Except String α)) = .ok (f x) := rfl

open PtxRun in
/-- A register not special holds what was last written to it, zero before. -/
theorem getReg_plain (c : PtxRun.Ctx) (r : PtxIr.Reg) (h : ∀ s, r ≠ .special s) :
    getReg c r = (do pure (((← get).regs.get? r).getD 0) : PtxRun.M UInt64) := by
  cases r with
  | special s => exact absurd rfl (h s)
  | _ => rfl

open PtxRun in
/-- `add` on two registers holding `x` and `y` writes their sum, cut to the
    width, into its destination. -/
theorem exec_add (c : PtxRun.Ctx) (ts : PtxRun.TS) (t : ClifTy) (d ra rb : PtxIr.Reg) (x y : UInt64)
    (hi : t = .i16 ∨ t = .i32 ∨ t = .i64)
    (hsa : ∀ s, ra ≠ .special s) (hsb : ∀ s, rb ≠ .special s)
    (hx : ts.regs.get? ra = some x) (hy : ts.regs.get? rb = some y) :
    (exec c (.bin .add none (uTy t.width) d (.reg ra) (.reg rb))).run ts =
      .ok (.step, { ts with regs := ts.regs.insert d ((x + y) &&& widthMask t) }) := by
  simp only [Std.HashMap.get?_eq_getElem?] at hx hy
  rcases hi with rfl | rfl | rfl <;>
  simp [exec, opnd, getReg_plain _ _ hsa, getReg_plain _ _ hsb, hx, hy, PtxRun.bin, PtxRun.binI, setReg, lift, uTy, ClifTy.width,
    PtxIr.Ty.width, PtxIr.Ty.isFloat, widthMask, mask] <;>
  first | rfl | exact congrArg _ (and_wide64 _).symm

/-! ## Straight-line code -/

open PtxRun in
/-- Instructions run one after another, each going on to the next line. -/
def steps (c : PtxRun.Ctx) : List PtxIr.Instr → PtxRun.M Unit
  | [] => pure ()
  | i :: is => do
      match ← exec c i with
      | .step => steps c is
      | _ => throw "not straight-line"

/-! ### What a thread's memory keeps

A PTX thread's world changes only by its stores and collectives: its memory
by a store, or a barrier's answer replacing shared memory. So a property of
memory that stores keep and every answer's shared memory has (`MemInv`) holds
through any run (`Keeps`, `exec_keeps`, `go_keeps`, `steps_keeps`): the
platform's NaN rule is one (`nanInv`). -/

/-- **A property of memory a thread keeps**: stores keep it, and so does
    shared memory becoming any of the thread's barrier answers. -/
structure MemInv (c : PtxRun.Ctx) (P : Mem → Prop) : Prop where
  store : ∀ {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64}, m.store a n v = some m' → P m → P m'
  smem : ∀ {k : Nat} {b : ByteArray} (m : Mem), c.ans[k]? = some (.smem b) → P m → P { m with arena := b }

/-- **What a thread's answers are**: a value answer an `i32`, and shared
    memory within what a GPU addresses, below `2 ^ 32` bytes. -/
def AnsOk (ans : List Answer) : Prop :=
  (∀ (k : Nat) (v : V), ans[k]? = some (Answer.val v) → ∃ x, v = V.sc .i32 x ∧ x &&& widthMask .i32 = x) ∧
  ∀ (k : Nat) (b : ByteArray), ans[k]? = some (Answer.smem b) → b.size < 2 ^ 32

/-- **A thread's memory as the platform has it**: its NaN rule, and shared
    memory below `2 ^ 32` bytes. -/
def MemOk (c : PtxRun.Ctx) (m : Mem) : Prop := m.nan = c.hw.nan ∧ m.arena.size < 2 ^ 32

/-- The platform's NaN rule is kept. -/
theorem nanInv (c : PtxRun.Ctx) (N : ClifTy → UInt64 → UInt64) : MemInv c (·.nan = N) :=
  ⟨fun h hm => (Mem.store_nan h).trans hm, fun _ _ hm => hm⟩

/-- The platform's memory is kept, the answers' shared memory within reach. -/
theorem memOkInv (c : PtxRun.Ctx) (hans : AnsOk c.ans) : MemInv c (MemOk c) :=
  ⟨fun h hm => ⟨(Mem.store_nan h).trans hm.1, (Contracts.store_sizes h .arena).symm ▸ hm.2⟩,
    fun _ hk hm => ⟨hm.1, hans.2 _ _ hk⟩⟩

/-- Running `x` keeps `P` of the world's memory. -/
def Keeps {α : Type} (P : Mem → Prop) (x : PtxRun.M α) : Prop :=
  ∀ (ts : PtxRun.TS) (r : α) (ts' : PtxRun.TS), x.run ts = .ok (r, ts') → P ts.w.mem → P ts'.w.mem

section Keeps
variable {P : Mem → Prop}

theorem kn_pure {α : Type} (a : α) : Keeps P (pure a : PtxRun.M α) := by
  intro ts r ts' h
  rw [← (Prod.mk.inj (Except.ok.inj h)).2]; exact id

theorem kn_bind {α β : Type} {x : PtxRun.M α} {f : α → PtxRun.M β} (hx : Keeps P x)
    (hf : ∀ a, Keeps P (f a)) : Keeps P (x >>= f) := by
  intro ts r ts' h hp
  simp only [StateT.run, bind, StateT.bind, Except.bind] at h
  split at h
  · cases h
  · rename_i p hp'
    obtain ⟨a, ts1⟩ := p
    exact hf a ts1 r ts' h (hx ts a ts1 hp' hp)

theorem kn_throw {α : Type} (m : String) : Keeps P (throw m : PtxRun.M α) := by
  intro ts r ts' h
  cases h

theorem kn_get : Keeps P (MonadState.get : PtxRun.M PtxRun.TS) := by
  intro ts r ts' h
  rw [← (Prod.mk.inj (Except.ok.inj h)).2]; exact id

theorem kn_modify (f : PtxRun.TS → PtxRun.TS) (hf : ∀ ts, (f ts).w.mem = ts.w.mem) :
    Keeps P (modify f : PtxRun.M Unit) := by
  intro ts r ts' h
  rw [← (Prod.mk.inj (Except.ok.inj h)).2, hf ts]; exact id

theorem kn_ite {α : Type} {p : Prop} [Decidable p] {x y : PtxRun.M α} (hx : Keeps P x) (hy : Keeps P y) :
    Keeps P (if p then x else y) := by
  split
  · exact hx
  · exact hy

theorem kn_setReg (n : PtxIr.Reg) (v : UInt64) : Keeps P (PtxRun.setReg n v) :=
  kn_modify _ fun _ => rfl

theorem kn_getReg (c : PtxRun.Ctx) (n : PtxIr.Reg) : Keeps P (PtxRun.getReg c n) := by
  unfold PtxRun.getReg
  split
  · exact kn_pure _
  · exact kn_bind kn_get fun _ => kn_pure _

theorem kn_opnd (c : PtxRun.Ctx) (o : PtxIr.Opnd) : Keeps P (PtxRun.opnd c o) := by
  unfold PtxRun.opnd
  split
  · exact kn_getReg c _
  · exact kn_pure _
  · exact kn_pure _
  · exact kn_pure _
  · split
    · exact kn_pure _
    · exact kn_throw _

theorem kn_lift {α : Type} (e : Except String α) : Keeps P (PtxRun.lift e) := by
  unfold PtxRun.lift
  split
  · exact kn_pure _
  · exact kn_throw _

theorem kn_load (c : PtxRun.Ctx) (sp : PtxIr.Space) (t : PtxIr.Ty) (ad : PtxIr.Addr) :
    Keeps P (PtxRun.load c sp t ad) := by
  unfold PtxRun.load
  split
  · split
    · split
      · exact kn_pure _
      · exact kn_throw _
    · exact kn_throw _
  · refine kn_bind (kn_opnd c _) fun _ => kn_bind kn_get fun _ => ?_
    split
    · exact kn_pure _
    · exact kn_throw _

/-- Reading the state, then going on from it. -/
theorem kn_get_bind {β : Type} {f : PtxRun.TS → PtxRun.M β}
    (hf : ∀ ts r ts', (f ts).run ts = .ok (r, ts') → P ts.w.mem → P ts'.w.mem) :
    Keeps P (MonadState.get >>= f) :=
  fun ts r ts' h => hf ts r ts' h

/-- Setting the state to one whose memory keeps `P`, then going on. -/
theorem kn_set_bind {β : Type} {g : PUnit → PtxRun.M β} (X ts0 : PtxRun.TS) (hg : Keeps P (g ⟨⟩))
    (hX : P ts0.w.mem → P X.w.mem) :
    ∀ r ts', ((MonadState.set X : PtxRun.M PUnit) >>= g).run ts0 = .ok (r, ts') → P ts0.w.mem → P ts'.w.mem :=
  fun r ts' h hp => hg X r ts' h (hX hp)

theorem kn_store {c : PtxRun.Ctx} (hP : MemInv c P) (c' : PtxRun.Ctx) (sp : PtxIr.Space) (t : PtxIr.Ty)
    (ad : PtxIr.Addr) (v : UInt64) : Keeps P (PtxRun.store c' sp t ad v) := by
  unfold PtxRun.store
  refine kn_bind (kn_opnd c' _) fun base => ?_
  dsimp only
  refine kn_get_bind fun ts0 r ts' h hp => ?_
  split at h
  · rename_i m hm
    rw [← (Prod.mk.inj (Except.ok.inj h)).2]
    exact hP.store hm hp
  · cases h

theorem kn_coll {c : PtxRun.Ctx} (hP : MemInv c P) (i : Intr) (args : List V) : Keeps P (PtxRun.coll c i args) := by
  unfold PtxRun.coll
  refine kn_get_bind fun ts0 r ts' h => ?_
  dsimp only at h
  refine kn_set_bind _ ts0 ?_ ?_ r ts' h
  rotate_left
  · exact id
  apply kn_ite
  · split
    · rename_i b hb
      refine kn_bind ?_ fun _ => kn_pure _
      intro ts r ts' h hp
      rw [← (Prod.mk.inj (Except.ok.inj h)).2]
      exact hP.smem _ hb hp
    · exact kn_ite (kn_pure _) (kn_throw _)
  · exact kn_pure _

/-- Compose `Keeps` through a computation built from the primitives. -/
macro "kn" hP:term : tactic => `(tactic| repeat' (first
  | exact kn_store $hP _ _ _ _ _ | exact kn_coll $hP _ _
  | exact kn_pure _ | exact kn_throw _ | exact kn_get | exact kn_setReg _ _
  | exact kn_opnd _ _ | exact kn_getReg _ _ | exact kn_lift _ | exact kn_load _ _ _ _
  | (refine kn_bind ?_ (fun _ => ?_)) | apply kn_ite | (apply kn_modify; intro; rfl) | split))

/-- **A PTX instruction keeps what its memory keeps.** -/
theorem exec_keeps {c : PtxRun.Ctx} (hP : MemInv c P) (i : PtxIr.Instr) : Keeps P (PtxRun.exec c i) := by
  cases i
  case fma r t d a b e => cases r <;> simp only [PtxRun.exec] <;> kn hP
  all_goals simp only [PtxRun.exec] <;> kn hP

theorem kn_guardOk (c : PtxRun.Ctx) (g : Option (Bool × PtxIr.Reg)) : Keeps P (PtxRun.guardOk c g) := by
  unfold PtxRun.guardOk
  split
  · exact kn_pure _
  · exact kn_bind (kn_getReg _ _) fun _ => kn_pure _

/-- **A PTX thread's run keeps what its memory keeps.** -/
theorem go_keeps {c : PtxRun.Ctx} (hP : MemInv c P) : ∀ fuel pc, Keeps P (PtxRun.go c fuel pc)
  | 0, _ => by simp only [PtxRun.go]; exact kn_throw _
  | fuel + 1, pc => by
    simp only [PtxRun.go]
    split
    · exact kn_pure _
    · exact go_keeps hP fuel _
    · refine kn_bind (kn_guardOk c _) fun b => ?_
      apply kn_ite
      · refine kn_bind (exec_keeps hP _) fun n => ?_
        split
        · exact go_keeps hP fuel _
        · exact go_keeps hP fuel _
        · exact kn_pure _
      · exact go_keeps hP fuel _

theorem steps_keeps {c : PtxRun.Ctx} (hP : MemInv c P) : ∀ L, Keeps P (steps c L)
  | [] => kn_pure _
  | i :: is => by
    simp only [steps]
    refine kn_bind (exec_keeps hP i) fun n => ?_
    split
    · exact steps_keeps hP is
    · exact kn_throw _

end Keeps

open PtxRun in
/-- **Straight-line code in the body**: instructions that sit unguarded at
    `pc` and each go on to the next line are run by `go` as `steps` runs them,
    arriving at the line after them. -/
theorem go_steps (c : PtxRun.Ctx) : ∀ (is : List PtxIr.Instr) (pc fuel : Nat) (ts ts' : PtxRun.TS),
    (∀ k (hk : k < is.length), c.body[pc + k]? = some (.ins none is[k])) →
    (steps c is).run ts = .ok ((), ts') →
    (PtxRun.go c (fuel + is.length) pc).run ts = (PtxRun.go c fuel (pc + is.length)).run ts'
  | [], pc, fuel, ts, ts', _, h => by
      simp [steps, StateT.run, pure, StateT.pure] at h
      cases h; rfl
  | i :: is, pc, fuel, ts, ts', hb, h => by
      have h0 := hb 0 (by simp)
      simp only [Nat.add_zero, List.getElem_cons_zero] at h0
      simp only [steps, StateT.run, bind, StateT.bind] at h
      cases he : exec c i ts with
      | error e => simp [he, Except.bind] at h
      | ok p =>
        obtain ⟨n, ts1⟩ := p
        cases n with
        | step =>
            simp [he, Except.bind] at h
            have ih := go_steps c is (pc + 1) fuel ts1 ts'
              (fun k hk => by
                have := hb (k + 1) (by simp; omega)
                simpa [Nat.add_assoc, Nat.add_comm 1 k] using this) h
            rw [List.length_cons, show fuel + (is.length + 1) = (fuel + is.length) + 1 by omega]
            simp only [PtxRun.go, StateT.run, h0, guardOk, bind, StateT.bind, pure, StateT.pure, Except.bind]
            simp only [StateT.run] at ih
            rw [show pc + (is.length + 1) = pc + 1 + is.length by omega]
            simp [Except.pure, StateT.bind, he, Except.bind, ih]
        | jump j => simp [he, Except.bind, throw, throwThe, MonadExceptOf.throw, StateT.lift, bind] at h
        | done => simp [he, Except.bind, throw, throwThe, MonadExceptOf.throw, StateT.lift, bind] at h

/-! ## Values in registers -/

/-- How a value of type `t` with bits `b` sits in its register: cut to the
    type's width, a flag as a predicate's one bit. -/
def enc (t : ClifTy) (b : UInt64) : UInt64 :=
  if t = .i8 then (if b &&& 0xff = 0 then 0 else 1) else b &&& widthMask t

theorem toNat_and_mask (a : UInt64) (w : Nat) (hw : w ≤ 64) :
    (a &&& (2 ^ w - 1 : Nat).toUInt64).toNat = a.toNat % 2 ^ w := by
  rw [UInt64.toNat_and, Nat.toUInt64_eq, UInt64.toNat_ofNat']
  rw [Nat.mod_eq_of_lt (by have := Nat.pow_le_pow_right (by decide : 0 < 2) hw; omega)]
  exact Nat.and_two_pow_sub_one_eq_mod _ _

/-- A sum cut to a width is the sum of the operands cut to it. -/
theorem add_mask (a b : UInt64) (w : Nat) (hw : w ≤ 64) :
    ((a &&& (2 ^ w - 1 : Nat).toUInt64) + (b &&& (2 ^ w - 1 : Nat).toUInt64)) &&& (2 ^ w - 1 : Nat).toUInt64 =
      (a + b) &&& (2 ^ w - 1 : Nat).toUInt64 := by
  apply UInt64.toNat_inj.mp
  rw [toNat_and_mask _ _ hw, toNat_and_mask _ _ hw, UInt64.toNat_add, UInt64.toNat_add,
    toNat_and_mask _ _ hw, toNat_and_mask _ _ hw]
  have h64 : 2 ^ w ∣ 2 ^ 64 := Nat.pow_dvd_pow 2 hw
  rw [Nat.mod_mod_of_dvd _ h64, Nat.mod_mod_of_dvd _ h64, ← Nat.add_mod]

theorem widthMask_int (t : ClifTy) (h : t = .i8 ∨ t = .i16 ∨ t = .i32 ∨ t = .i64) :
    widthMask t = (2 ^ t.width - 1 : Nat).toUInt64 := by
  rcases h with rfl | rfl | rfl | rfl <;> rfl

/-- What the block semantics says an `iadd` of two integers of type `t`
    computes. -/
theorem evalInst_iadd (m : Mem) (vs : Blocks.Vals) (d a b : Val) (t : ClifTy) (x y : UInt64)
    (hi : t.isInt = true) (ha : Blocks.getV vs a = some (.sc t x)) (hb : Blocks.getV vs b = some (.sc t y)) :
    Blocks.evalInst m vs (.iadd d a b) = some (d, .sc t ((x + y) &&& widthMask t)) := by
  simp [Blocks.evalInst, Blocks.evalInst.bin, ha, hb, Blocks.viaOp, evalOp, Sem.bin, Sem.get, hi, norm]

/-- **An integer sum, end to end**: two plain values of type `t` (16, 32 or
    64 bits) held in their registers as `enc` says; the block semantics
    computes `r`; the compiler writes lines whose run leaves `enc t r` in the
    destination's register. -/
theorem iadd_sound (F : Facts) (tag : Nat) (c : PtxRun.Ctx) (ts : PtxRun.TS) (m : Mem) (vs : Blocks.Vals)
    (d a b : Val) (t : ClifTy) (x y : UInt64)
    (hi : t = .i16 ∨ t = .i32 ∨ t = .i64)
    (hta : F.tys.get? a.id = some t) (htb : F.tys.get? b.id = some t)
    (hwa : ¬ (F.window = true ∧ F.origin.get? a.id = some .shared))
    (hwb : ¬ (F.window = true ∧ F.origin.get? b.id = some .shared))
    (hla : F.lits.get? a.id = none) (hlb : F.lits.get? b.id = none)
    (hva : Blocks.getV vs a = some (.sc t x)) (hvb : Blocks.getV vs b = some (.sc t y))
    (hra : ts.regs.get? (regOf t a) = some (enc t x)) (hrb : ts.regs.get? (regOf t b) = some (enc t y)) :
    ∃ r, Blocks.evalInst m vs (.iadd d a b) = some (d, .sc t r) ∧
      (steps c ((lower F tag (.iadd d a b)).lines.toList.filterMap fun
          | .ins none i => some i
          | _ => none)).run ts =
        .ok ((), { ts with regs := ts.regs.insert (regOf t d) (enc t r) }) := by
  have hint : t.isInt = true := by rcases hi with rfl | rfl | rfl <;> rfl
  refine ⟨(x + y) &&& widthMask t, evalInst_iadd m vs d a b t x y hint hva hvb, ?_⟩
  rw [lower_iadd F tag d a b t hta htb hwa hwb hla hlb]
  have hne8 : t ≠ .i8 := by rcases hi with rfl | rfl | rfl <;> decide
  have hm := widthMask_int t (by rcases hi with h | h | h <;> simp [h])
  simp only [enc, hne8, if_false] at hra hrb ⊢
  have hw : t.width ≤ 64 := by rcases hi with rfl | rfl | rfl <;> decide
  have hx := exec_add c ts t (regOf t d) (regOf t a) (regOf t b) _ _ hi
    (fun s h => by cases h) (fun s h => by cases h) hra hrb
  simp only [StateT.run] at hx
  simp [steps, StateT.run, hx, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  rw [hm, add_mask _ _ _ hw, ← hm, UInt64.and_assoc, UInt64.and_self]

/-! ## Operands -/

/-- Whether `v` is a shared address held in the window, as a proposition. -/
def Win (F : Facts) (v : Val) : Prop := F.window = true ∧ F.origin.get? v.id = some .shared

/-- A plain value is its own register, and the lowering writes nothing. -/
theorem runLM_r_plain (F : Facts) (tag : Nat) (s : Out) (v : Val) (t : ClifTy)
    (hw : ¬ Win F v) (ht : F.tys.get? v.id = some t)
    (hl : F.lits.get? v.id = none) :
    runLM (r v) F tag s = (regOf t v, s) := by
  simp only [Win, Std.HashMap.get?_eq_getElem?] at *
  simp [r, rw, isWin, Facts.win, tyOf, facts?, hw, ht, hl]

/-- A constant is put in a register of the instruction's own, by one `mov`. -/
theorem runLM_r_lit (F : Facts) (tag : Nat) (s : Out) (v : Val) (t : ClifTy) (l : PtxIr.Opnd)
    (hw : ¬ Win F v) (ht : F.tys.get? v.id = some t) (hl : F.lits.get? v.id = some l) :
    runLM (r v) F tag s = (.own "k" tag s.n,
      { s with n := s.n + 1, regs := s.regs.push (.own "k" tag s.n, t),
               lines := s.lines.push (.ins none (.mov (bits t) (.own "k" tag s.n) l)) }) := by
  simp only [Win, Std.HashMap.get?_eq_getElem?] at *
  simp [r, rw, isWin, Facts.win, tyOf, facts?, hw, ht, hl, own, emit, emitL]

open PtxRun in
/-- `mov` of an integer literal writes it cut to the type's width. -/
theorem exec_mov_imm (c : PtxRun.Ctx) (ts : PtxRun.TS) (ty : PtxIr.Ty) (d : PtxIr.Reg) (v : Int) :
    (exec c (.mov ty d (.imm v))).run ts =
      .ok (.step, { ts with regs := ts.regs.insert d (mask ty.width (UInt64.ofInt v)) }) := by
  simp [exec, opnd, setReg]

open PtxRun in
/-- `mov` of a float literal writes its bits cut to the type's width. -/
theorem exec_mov_f32 (c : PtxRun.Ctx) (ts : PtxRun.TS) (ty : PtxIr.Ty) (d : PtxIr.Reg) (b : Nat) :
    (exec c (.mov ty d (.f32 b))).run ts =
      .ok (.step, { ts with regs := ts.regs.insert d (mask ty.width (UInt64.ofNat b)) }) := by
  simp [exec, opnd, setReg]

open PtxRun in
theorem exec_mov_f64 (c : PtxRun.Ctx) (ts : PtxRun.TS) (ty : PtxIr.Ty) (d : PtxIr.Reg) (b : Nat) :
    (exec c (.mov ty d (.f64 b))).run ts =
      .ok (.step, { ts with regs := ts.regs.insert d (mask ty.width (UInt64.ofNat b)) }) := by
  simp [exec, opnd, setReg]

theorem ofInt_toNat (z : UInt64) : UInt64.ofInt (z.toNat : Int) = z := by
  unfold UInt64.ofInt
  have hz := z.toNat_lt
  rw [Int.emod_eq_of_lt (by omega) (by omega)]
  simp

/-- `mul.wide.u32` of the low half of a value within 32 bits and another value
    within 32 bits is their 64-bit product. -/
theorem mulWide_low (x y : UInt64) (hx : x.toNat < 2 ^ 32) (hy : y.toNat < 2 ^ 32) :
    PtxRun.mask (2 * PtxIr.Ty.u32.width)
      (UInt64.ofInt (PtxRun.toInt .u32 (PtxRun.ofIntW 32 (PtxRun.toInt .u64 x)) * PtxRun.toInt .u32 y)) = x * y := by
  have hm32 : ∀ z : UInt64, z.toNat < 2 ^ 32 → PtxRun.mask 32 z = z := by
    intro z hz
    apply UInt64.toNat_inj.mp
    simp only [PtxRun.mask]
    rw [if_neg (by decide), UInt64.toNat_and]
    have : ((1 : UInt64) <<< UInt64.ofNat 32 - 1).toNat = 2 ^ 32 - 1 := by decide
    rw [this]
    exact Nat.and_two_pow_sub_one_of_lt_two_pow hz
  have hm64 : ∀ z : UInt64, PtxRun.mask 64 z = z := by intro z; simp [PtxRun.mask]
  have hx64 : PtxRun.toInt .u64 x = (x.toNat : Int) := by
    simp only [PtxRun.toInt, show PtxIr.Ty.u64.width = 64 from rfl, show PtxIr.Ty.u64.isSigned = false from rfl,
      hm64, Bool.false_and, Bool.false_eq_true, if_false]
  have hlow : PtxRun.ofIntW 32 (PtxRun.toInt .u64 x) = x := by
    rw [hx64, PtxRun.ofIntW, ofInt_toNat]; exact hm32 x hx
  have hu32 : ∀ z : UInt64, z.toNat < 2 ^ 32 → PtxRun.toInt .u32 z = (z.toNat : Int) := by
    intro z hz
    simp only [PtxRun.toInt, show PtxIr.Ty.u32.width = 32 from rfl, show PtxIr.Ty.u32.isSigned = false from rfl,
      hm32 z hz, Bool.false_and, Bool.false_eq_true, if_false]
  rw [hlow, hu32 x hx, hu32 y hy]
  have hp : x.toNat * y.toNat < 2 ^ 64 :=
    Nat.lt_of_lt_of_le (Nat.mul_lt_mul'' hx hy) (by rw [← Nat.pow_add]; exact Nat.le_refl _)
  have hxy : (x * y).toNat = x.toNat * y.toNat := by rw [UInt64.toNat_mul, Nat.mod_eq_of_lt hp]
  rw [show 2 * PtxIr.Ty.u32.width = 64 from rfl, hm64, ← Int.natCast_mul, ← hxy, ofInt_toNat]

theorem zext32 (y : UInt64) : PtxRun.ofIntW 64 (PtxRun.toInt .u32 y) = y &&& 0xffffffff := by
  simp [PtxRun.ofIntW, PtxRun.toInt, PtxRun.mask, PtxIr.Ty.width, PtxIr.Ty.isSigned]
  have h : (y.toNat &&& (((1 : UInt64) <<< 32) - 1).toNat) = (y &&& 4294967295).toNat := by
    rw [UInt64.toNat_and]; rfl
  rw [h, ofInt_toNat]

open PtxRun in
/-- `cvt.u64.u32` of a register holding `y` writes its low 32 bits. -/
theorem exec_cvt_zext (c : PtxRun.Ctx) (ts : PtxRun.TS) (d src : PtxIr.Reg) (y : UInt64)
    (hs : ∀ s, src ≠ .special s) (hy : ts.regs.get? src = some y) :
    (exec c (.cvt none .u64 .u32 d (.reg src))).run ts =
      .ok (.step, { ts with regs := ts.regs.insert d (y &&& 0xffffffff) }) := by
  simp only [Std.HashMap.get?_eq_getElem?] at hy
  simp [exec, opnd, getReg_plain _ _ hs, hy, setReg, lift, PtxRun.cvt, PtxIr.Ty.isFloat]
  exact congrArg _ (zext32 y)

/-! ## The simulation -/

/-- The bits a `mov` of literal `l` into a register of type `t` writes. -/
def litBits (t : ClifTy) : PtxIr.Opnd → UInt64
  | .imm v => PtxRun.mask (bits t).width (UInt64.ofInt v)
  | .f32 b | .f64 b => PtxRun.mask (bits t).width (UInt64.ofNat b)
  | _ => 0

/-- A literal: an integer or a float's bits. -/
def IsLit : PtxIr.Opnd → Prop
  | .imm _ | .f32 _ | .f64 _ => True
  | _ => False

/-- The raw value a literal reads. -/
def litRaw : PtxIr.Opnd → UInt64
  | .imm v => UInt64.ofInt v
  | .f32 b | .f64 b => UInt64.ofNat b
  | _ => 0

theorem litBits_eq (t : ClifTy) (l : PtxIr.Opnd) (hl : IsLit l) :
    litBits t l = PtxRun.mask (bits t).width (litRaw l) := by
  cases l <;> simp_all [IsLit, litBits, litRaw]

/-- A scalar type a register holds. -/
def Scalar (t : ClifTy) : Prop := t = .i8 ∨ t = .i16 ∨ t = .i32 ∨ t = .i64 ∨ t = .f32 ∨ t = .f64

/-- A value's encoding is already cut to its register's width. -/
theorem mask_enc (t : ClifTy) (b : UInt64) :
    PtxRun.mask (bits t).width (enc t b) = enc t b := by
  by_cases ht : ¬ Scalar t
  · have : (bits t).width = 64 := by cases t <;> simp_all [Scalar, bits, PtxIr.Ty.width]
    simp [PtxRun.mask, this]
  rcases Classical.not_not.mp ht with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [PtxRun.mask, bits, PtxIr.Ty.width, enc, widthMask]
  · by_cases h : b &&& 255 = 0 <;> simp [h] <;> decide
  all_goals first
    | rfl
    | (simp only [show ¬ (64 ≥ 64) = False from by decide]; rfl)
    | (simp; rw [UInt64.and_assoc]; rfl)

/-- A literal that is a value's encoding exactly writes it into a register
    of the value's type. -/
theorem lit_bits (t : ClifTy) (l : PtxIr.Opnd) (b : UInt64) (hl : IsLit l) (h : litRaw l = enc t b) :
    litBits t l = enc t b := by
  rw [litBits_eq t l hl, h, mask_enc]

/-- A register an instruction made for itself. -/
def Own (R : PtxIr.Reg) : Prop := ∃ p t n, R = .own p t n

/-- One of the registers instruction `tag` makes for itself, numbered `n` or
    more. -/
def Fresh (tag n : Nat) (R : PtxIr.Reg) : Prop := ∃ p k, n ≤ k ∧ R = .own p tag k

/-- A windowed shared address's register holds its offset into the shared
    arena, modulo 2^32: the window's sums are 32-bit. -/
def WinOff (b off : UInt64) : Prop := (b - regionBase .arena).toNat % 2 ^ 32 = off.toNat

/-- A value as its bits are bounded: a shared address its offset from the
    shared base, any other value itself. -/
def Rel (F : Facts) (v : Val) (b : UInt64) : UInt64 :=
  if F.origin.get? v.id = some .shared then b - regionBase .arena else b

/-- A value spans no more bits than the facts say (`Facts.mag`), a shared
    address at most `sharedReach`. -/
def Bounded (F : Facts) (v : Val) (b : UInt64) : Prop :=
  (Rel F v b).toNat < 2 ^ offBits F.origin F.mag v ∧
    (F.origin.get? v.id = some .shared → offBits F.origin F.mag v ≤ sharedReach)

/-- A value whose bits the facts bound nothing of, and not a shared address. -/
def Unbounded (F : Facts) (v : Val) : Prop :=
  F.mag.get? v.id = none ∧ F.origin.get? v.id ≠ some .shared

theorem bounded_unb {F : Facts} {v : Val} (h : Unbounded F v) (b : UInt64) : Bounded F v b := by
  unfold Bounded offBits; rw [h.1]; exact ⟨UInt64.toNat_lt _, fun hs => absurd hs h.2⟩

/-- **The check's bound holds of a value** its definition's rule bounds. -/
theorem bounded_of (F : Facts) (d : Val) (i : Inst) (r : UInt64) (hmo : magOk F d i = true)
    (hrule : ∀ k, magRule F.origin F.mag F.lits F.tys i = some k → (Rel F d r).toNat < 2 ^ k) :
    Bounded F d r := by
  unfold Bounded offBits
  unfold magOk at hmo
  split at hmo
  · rename_i h; rw [h]; exact ⟨UInt64.toNat_lt _, fun hs => absurd hmo (by rw [hs]; decide)⟩
  · rename_i k' h
    rw [h]
    simp only [Bool.and_eq_true, Option.any_eq_true, Bool.or_eq_true, bne_iff_ne, ne_eq] at hmo
    obtain ⟨⟨k, hk, hle⟩, hsh⟩ := hmo
    refine ⟨Nat.lt_of_lt_of_le (hrule k hk) (Nat.pow_le_pow_right (by decide) (by simpa using hle)), fun hs => ?_⟩
    rcases hsh with h' | ⟨h', -⟩
    · exact absurd hs h'
    · simpa using h'

/-- A bounded value, as a number, spans its `valBits`. -/
theorem bnd_val {F : Facts} {v : Val} {b : UInt64} (h : Bounded F v b) : b.toNat < 2 ^ valBits F.origin F.mag v := by
  replace h := h.1
  unfold Rel at h; unfold valBits
  by_cases hs : F.origin.get? v.id = some .shared
  · rw [if_pos (by rw [hs]; rfl)]; exact UInt64.toNat_lt _
  · rw [if_neg hs] at h; rw [if_neg (by simp only [beq_iff_eq]; exact hs)]; exact h

/-- A bounded shared address's offset spans its `offBits`. -/
theorem bnd_off {F : Facts} {v : Val} {b : UInt64} (h : Bounded F v b) (hs : F.origin.get? v.id = some .shared) :
    (b - regionBase .arena).toNat < 2 ^ offBits F.origin F.mag v := by
  have h1 := h.1; unfold Rel at h1; rwa [if_pos hs] at h1

theorem rel_plain {F : Facts} {d : Val} {r : UInt64} (hs : F.origin.get? d.id ≠ some .shared) : Rel F d r = r := by
  unfold Rel; rw [if_neg hs]

/-- Only a sum is a shared address the check passes. -/
theorem not_shared_of {F : Facts} {d : Val} {i : Inst} (hmo : magOk F d i = true)
    (hi : ∀ d' a b, i ≠ .iadd d' a b) : F.origin.get? d.id ≠ some .shared := by
  unfold magOk at hmo
  split at hmo
  · simpa using hmo
  · simp only [Bool.and_eq_true, Bool.or_eq_true, bne_iff_ne, ne_eq] at hmo
    rcases hmo.2 with h | ⟨-, h⟩
    · exact h
    · exfalso; cases i <;> first | exact hi _ _ _ rfl | simp at h

theorem lt_min64 {x : UInt64} {p : Nat} (h : x.toNat < 2 ^ p) : x.toNat < 2 ^ (min 64 p) := by
  rcases Nat.le_total p 64 with hp | hp
  · rw [Nat.min_eq_right hp]; exact h
  · rw [Nat.min_eq_left hp]; exact x.toNat_lt

/-- A value cut to at most 32 bits spans at most 32 and at most what it already
    spans. -/
theorem lt_min32 {x m : UInt64} {p : Nat} (hm : m.toNat < 2 ^ 32) (h : x.toNat < 2 ^ p) :
    (x &&& m).toNat < 2 ^ (min 32 p) := by
  rw [UInt64.toNat_and]
  rcases Nat.le_total p 32 with hp | hp
  · rw [Nat.min_eq_right hp]; exact Nat.lt_of_le_of_lt Nat.and_le_left h
  · rw [Nat.min_eq_left hp]; exact Nat.lt_of_le_of_lt Nat.and_le_right hm

theorem add_lt_bits (x y : UInt64) (p q : Nat) (hx : x.toNat < 2 ^ p) (hy : y.toNat < 2 ^ q) :
    (x + y).toNat < 2 ^ (max p q + 1) := by
  rw [UInt64.toNat_add]
  have h1 : 2 ^ p ≤ 2 ^ max p q := Nat.pow_le_pow_right (by decide) (Nat.le_max_left _ _)
  have h2 : 2 ^ q ≤ 2 ^ max p q := Nat.pow_le_pow_right (by decide) (Nat.le_max_right _ _)
  rw [Nat.pow_succ]
  exact Nat.lt_of_le_of_lt (Nat.mod_le _ _) (by omega)

theorem and_le_toNat (x m : UInt64) : (x &&& m).toNat ≤ x.toNat := by
  rw [UInt64.toNat_and]; exact Nat.and_le_left


/-- **The registers hold the block's values**, those of them the run has
    defined (`D`): a plain value its bits, as `enc` says, in its register; a
    constant's literal its bits. -/
structure Sim (F : Facts) (D : Nat → Prop) (vs : Blocks.Vals) (regs : Std.HashMap PtxIr.Reg UInt64) :
    Prop where
  /-- A defined value is present, and of the type the facts predict. -/
  pres : ∀ (v : Val), D v.id → ∃ t b, Blocks.getV vs v = some (.sc t b) ∧ F.tys.get? v.id = some t
  /-- A defined value holds no bits above its type's width. -/
  normd : ∀ v t b, D v.id → Blocks.getV vs v = some (.sc t b) → b &&& widthMask t = b
  /-- A flag is 0 or 1. -/
  flag : ∀ v b, D v.id → Blocks.getV vs v = some (.sc .i8 b) → b ≤ 1
  plain : ∀ v t b, D v.id → Blocks.getV vs v = some (.sc t b) → F.tys.get? v.id = some t →
    F.lits.get? v.id = none → ¬ Win F v →
    regs.get? (regOf t v) = some (enc t b)
  lit : ∀ v t b l, D v.id → Blocks.getV vs v = some (.sc t b) → F.tys.get? v.id = some t →
    F.lits.get? v.id = some l → IsLit l ∧ litRaw l = enc t b
  /-- A windowed shared address is held as its offset in the window. -/
  win : ∀ v t b, D v.id → Blocks.getV vs v = some (.sc t b) → Win F v →
    ∃ off : UInt64, regs.get? (winReg v) = some off ∧ off.toNat < 2 ^ 32 ∧ WinOff b off
  /-- A comparison's flag has its negation beside it. -/
  neg : ∀ v b, D v.id → Blocks.getV vs v = some (.sc .i8 b) → F.negated v = true →
    regs.get? (negReg v) = some (if b &&& 0xff = 0 then 1 else 0)
  /-- A defined value spans no more bits than the facts say. -/
  bound : ∀ v t b, D v.id → Blocks.getV vs v = some (.sc t b) → Bounded F v b

theorem regOf_not_own (t : ClifTy) (v : Val) : ¬ Own (regOf t v) := by
  rintro ⟨p, t', n, h⟩; cases h

theorem not_special_own (p : String) (t n : Nat) : ∀ s, PtxIr.Reg.own p t n ≠ .special s := by
  intro s h; cases h

theorem not_special_val (c : String) (i : Nat) : ∀ s, PtxIr.Reg.val c i ≠ .special s := by
  intro s h; cases h

open PtxRun in
/-- **An operand**: the lines `r v` writes for a value in scope, not a
    windowed shared address, leave its encoding in the register it names,
    and change no register but the instruction's own. -/
theorem r_sound (F : Facts) (tag : Nat) (s : Out) (vs : Blocks.Vals) (c : PtxRun.Ctx) (ts : PtxRun.TS)
    (v : Val) (t : ClifTy) (b : UInt64)
    (D : Nat → Prop) (hS : Sim F D vs ts.regs) (hD : D v.id) (hv : Blocks.getV vs v = some (.sc t b))
    (ht : F.tys.get? v.id = some t)
    (hw : ¬ Win F v) :
    ∃ L : List PtxIr.Instr,
      (runLM (r v) F tag s).2.lines = s.lines ++ (L.map (.ins none)).toArray ∧
      s.n ≤ (runLM (r v) F tag s).2.n ∧ ¬ Fresh tag (runLM (r v) F tag s).2.n (runLM (r v) F tag s).1 ∧
      (∀ s', (runLM (r v) F tag s).1 ≠ .special s') ∧
      ∃ regs', (steps c L).run ts = .ok ((), { ts with regs := regs' }) ∧
        regs'.get? (runLM (r v) F tag s).1 = some (enc t b) ∧
        ∀ R, ¬ Fresh tag s.n R → regs'.get? R = ts.regs.get? R := by
  cases hl : F.lits.get? v.id with
  | some l =>
      obtain ⟨hil, hraw⟩ := hS.lit v t b l hD hv ht hl
      have hlb := lit_bits t l b hil hraw
      rw [runLM_r_lit F tag s v t l hw ht hl]
      refine ⟨[.mov (bits t) (.own "k" tag s.n) l], by simp, by simp,
        fun ⟨p, k, hk, h⟩ => by simp at hk h; omega, not_special_own _ _ _,
        ts.regs.insert (.own "k" tag s.n) (litBits t l), ?_, ?_, ?_⟩
      · have hx : (exec c (.mov (bits t) (.own "k" tag s.n) l)).run ts =
            .ok (.step, { ts with regs := ts.regs.insert (.own "k" tag s.n) (litBits t l) }) := by
          cases l with
          | imm k => exact exec_mov_imm c ts _ _ k
          | f32 k => exact exec_mov_f32 c ts _ _ k
          | f64 k => exact exec_mov_f64 c ts _ _ k
          | _ => cases hil
        simp only [StateT.run] at hx
        simp [steps, StateT.run, hx, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
      · simp [hlb]
      · intro R hR
        have : (PtxIr.Reg.own "k" tag s.n) ≠ R := fun h => hR ⟨_, _, Nat.le_refl _, h.symm⟩
        simp [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, this]
  | none =>
      rw [runLM_r_plain F tag s v t hw ht hl]
      have hnf : ¬ Fresh tag s.n (regOf t v) := fun ⟨p, k, hk, h⟩ => by cases h
      exact ⟨[], by simp, Nat.le_refl _, hnf, not_special_val _ _,
        ts.regs, by simp [steps, StateT.run, pure, StateT.pure, Except.pure], hS.plain v t b hD hv ht hl hw,
        fun _ _ => rfl⟩

/-! ## The block semantics' values -/

/-- Writing value `d` leaves every other value present as it was. -/
theorem getV_setV_ne (vs : Blocks.Vals) (d v : Val) (x : V) (h : v.id ≠ d.id)
    (hv : (Blocks.getV vs v).isSome) : Blocks.getV (Blocks.setV vs d x) v = Blocks.getV vs v := by
  simp only [Blocks.getV, Blocks.setV] at hv ⊢
  have hlt : v.id < vs.size := by
    rcases h' : vs[v.id]? with _ | y
    · simp [h'] at hv
    · exact (Array.getElem?_eq_some_iff.mp h').1
  rw [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_ne (Ne.symm h)]
  split
  · rfl
  · rw [Array.getElem?_append_left hlt]

/-- A block's parameters bound to arguments, as `runFrom` binds them. -/
def bindParams (vs : Blocks.Vals) (ps : List (Val × ClifTy)) (xs : List V) : Blocks.Vals :=
  (ps.zip xs).foldl (fun vs ((v, _), x) => Blocks.setV vs v x) vs

/-- Binding leaves every other value present as it was. -/
theorem getV_bind_other : ∀ (ps : List (Val × ClifTy)) (xs : List V) (vs : Blocks.Vals) (v : Val),
    (∀ p ∈ ps, p.1.id ≠ v.id) → (Blocks.getV vs v).isSome →
    Blocks.getV (bindParams vs ps xs) v = Blocks.getV vs v
  | [], _, _, _, _, _ => by simp [bindParams]
  | _ :: _, [], _, _, _, _ => by simp [bindParams]
  | (p, t) :: ps, x :: xs, vs, v, hne, hv => by
      have h1 : Blocks.getV (Blocks.setV vs p x) v = Blocks.getV vs v :=
        getV_setV_ne vs p v x (Ne.symm (hne (p, t) List.mem_cons_self)) hv
      have := getV_bind_other ps xs (Blocks.setV vs p x) v (fun q hq => hne q (List.mem_cons_of_mem _ hq))
        (by rw [h1]; exact hv)
      simp only [bindParams, List.zip_cons_cons, List.foldl_cons] at this ⊢
      rw [this, h1]

/-- Binding gives each parameter its argument, the parameters distinct. -/
theorem getV_bind_param : ∀ (ps : List (Val × ClifTy)) (xs : List V) (vs : Blocks.Vals) (k : Nat)
    (hk : k < ps.length) (hk' : k < xs.length), (ps.map (·.1.id)).Nodup →
    Blocks.getV (bindParams vs ps xs) ps[k].1 = some xs[k]
  | [], _, _, _, hk, _, _ => by simp at hk
  | _ :: _, [], _, _, _, hk', _ => by simp at hk'
  | (p, t) :: ps, x :: xs, vs, k, hk, hk', hnd => by
      simp only [List.map_cons, List.nodup_cons, List.mem_map] at hnd
      simp only [bindParams, List.zip_cons_cons, List.foldl_cons]
      cases k with
      | zero =>
          simp only [List.getElem_cons_zero]
          have := getV_bind_other ps xs (Blocks.setV vs p x) p
            (fun q hq h => hnd.1 ⟨q, hq, h⟩) (by rw [getV_setV_self]; rfl)
          simp only [bindParams] at this
          rw [this, getV_setV_self]
      | succ k =>
          simp only [List.getElem_cons_succ]
          exact getV_bind_param ps xs (Blocks.setV vs p x) k (by simp at hk; omega) (by simp at hk'; omega) hnd.2

/-- Arguments that all have values have one each, in order. -/
theorem mapM_getV (vs : Blocks.Vals) : ∀ (args : List Val) (xs : List V),
    args.mapM (Blocks.getV vs) = some xs →
    xs.length = args.length ∧ ∀ k (hk : k < args.length) (hk' : k < xs.length), Blocks.getV vs args[k] = some xs[k]
  | [], xs, h => by simp at h; subst h; simp
  | a :: args, xs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨x, hx, ys, hys, hxs⟩ := h
      simp only [Option.pure_def, Option.some.injEq] at hxs
      subst hxs
      obtain ⟨hl, hk⟩ := mapM_getV vs args ys hys
      refine ⟨by simp [hl], fun k hk1 hk2 => ?_⟩
      cases k with
      | zero => simpa using hx
      | succ k => simpa using hk k (by simp at hk1; omega) (by simp at hk2; omega)

/-! ## Edges -/

/-- The temporaries `copyOut` makes, from count `n`. -/
def tmpsFrom (tag n k : Nat) : List PtxIr.Reg := (List.range k).map fun i => .own "t" tag (n + i)

theorem tmpsFrom_succ (tag n k : Nat) :
    tmpsFrom tag n (k + 1) = .own "t" tag n :: tmpsFrom tag (n + 1) k := by
  simp only [tmpsFrom, List.range_succ_eq_map, List.map_cons, List.map_map, Nat.add_zero, List.cons.injEq,
    true_and]
  apply List.map_congr_left; intro i _
  simp only [Function.comp, Nat.succ_eq_add_one, PtxIr.Reg.own.injEq, true_and]; omega

theorem runLM_moveIn (F : Facts) (tag : Nat) (s : Out) (tmp : PtxIr.Reg) (t : ClifTy) (a : Val) :
    (runLM (moveIn tmp t a) F tag s).2.lines = s.lines.push (.ins none (moveInstr F tmp t a)) ∧
    (runLM (moveIn tmp t a) F tag s).2.n = s.n := by
  simp [moveIn, facts?, emit, emitL]

/-- The lines `copyOut` writes. -/
def outLines (F : Facts) (tag n : Nat) : List ((PtxIr.Reg × ClifTy × Bool) × Val) → List PtxIr.Instr
  | [] => []
  | ((_, t, _), a) :: ps => moveInstr F (.own "t" tag n) t a :: outLines F tag (n + 1) ps

/-- The lines `copyIn` writes. -/
def inLines : List ((PtxIr.Reg × ClifTy × Bool) × Val) → List PtxIr.Reg → List PtxIr.Instr
  | ((d, t, _), _) :: ps, tmp :: tmps => .mov (bits t) d (.reg tmp) :: inLines ps tmps
  | _, _ => []

theorem runLM_copyOut (F : Facts) (tag : Nat) :
    ∀ (ps : List ((PtxIr.Reg × ClifTy × Bool) × Val)) (s : Out),
      (runLM (Ptx.copyOut ps) F tag s).1 = tmpsFrom tag s.n ps.length ∧
      (runLM (Ptx.copyOut ps) F tag s).2.lines = s.lines ++ ((outLines F tag s.n ps).map (.ins none)).toArray ∧
      (runLM (Ptx.copyOut ps) F tag s).2.n = s.n + ps.length
  | [], s => by simp [Ptx.copyOut, tmpsFrom, outLines]
  | ((_, t, _), a) :: ps, s => by
      let s1 : Out := { s with n := s.n + 1, regs := s.regs.push (.own "t" tag s.n, t) }
      have hm := runLM_moveIn F tag s1 (.own "t" tag s.n) t a
      have ih := runLM_copyOut F tag ps (runLM (moveIn (.own "t" tag s.n) t a) F tag s1).2
      simp only [Ptx.copyOut, runLM_bind, runLM_pure, own, runLM_read, runLM_get, runLM_set] at ih ⊢
      rw [hm.2] at ih
      refine ⟨?_, ?_, ?_⟩
      · rw [ih.1, List.length_cons, tmpsFrom_succ]
      · rw [ih.2.1, hm.1]; simp [outLines, s1]
      · rw [ih.2.2]; simp [s1]; omega

theorem runLM_copyIn (F : Facts) (tag : Nat) :
    ∀ (ps : List ((PtxIr.Reg × ClifTy × Bool) × Val)) (tmps : List PtxIr.Reg) (s : Out),
      (runLM (Ptx.copyIn ps tmps) F tag s).2.lines = s.lines ++ ((inLines ps tmps).map (.ins none)).toArray ∧
      (runLM (Ptx.copyIn ps tmps) F tag s).2.n = s.n
  | ((d, t, _), _) :: ps, tmp :: tmps, s => by
      have ih := runLM_copyIn F tag ps tmps
        { s with lines := s.lines.push (.ins none (.mov (bits t) d (.reg tmp))) }
      simp only [Ptx.copyIn, runLM_bind, emit, emitL, runLM_modify] at ih ⊢
      refine ⟨?_, ?_⟩
      · rw [ih.1]; simp [inLines]
      · rw [ih.2]
  | [], _, s => by simp [Ptx.copyIn, inLines]
  | _ :: _, [], s => by simp [Ptx.copyIn, inLines]

open PtxRun in
/-- `mov` of a register holding `x` writes `x` cut to the type's width. -/
theorem exec_mov_reg (c : PtxRun.Ctx) (ts : PtxRun.TS) (ty : PtxIr.Ty) (d src : PtxIr.Reg) (x : UInt64)
    (hs : ∀ s, src ≠ .special s) (hx : ts.regs.get? src = some x) :
    (exec c (.mov ty d (.reg src))).run ts =
      .ok (.step, { ts with regs := ts.regs.insert d (mask ty.width x) }) := by
  simp only [Std.HashMap.get?_eq_getElem?] at hx
  simp [exec, opnd, getReg_plain _ _ hs, hx, setReg]

open PtxRun in
theorem steps_append (c : PtxRun.Ctx) (L1 L2 : List PtxIr.Instr) (ts ts1 : PtxRun.TS)
    (h1 : (steps c L1).run ts = .ok ((), ts1)) :
    (steps c (L1 ++ L2)).run ts = (steps c L2).run ts1 := by
  induction L1 generalizing ts with
  | nil => simp [steps, StateT.run, pure, StateT.pure, Except.pure] at h1; cases h1; rfl
  | cons i is ih =>
      simp only [steps, List.cons_append, StateT.run, bind, StateT.bind] at h1 ⊢
      cases he : exec c i ts with
      | error e => simp [he, Except.bind] at h1
      | ok p =>
          obtain ⟨n, ts2⟩ := p
          cases n with
          | step => simp [he, Except.bind] at h1 ⊢; exact ih ts2 h1
          | jump j => simp [he, Except.bind, throw, throwThe, MonadExceptOf.throw, StateT.lift, bind] at h1
          | done => simp [he, Except.bind, throw, throwThe, MonadExceptOf.throw, StateT.lift, bind] at h1

/-- What an edge's temporary holds for argument `a` with bits `b`: a windowed
    shared address its offset in the window, any other value its encoding. -/
def EdgeVal (F : Facts) (t : ClifTy) (a : Val) (b x : UInt64) : Prop :=
  (Win F a → x.toNat < 2 ^ 32 ∧ WinOff b x) ∧ (¬ Win F a → x = enc t b)

theorem mask32_lt (x : UInt64) (h : x.toNat < 2 ^ 32) : PtxRun.mask 32 x = x := by
  simp only [PtxRun.mask, show ¬ (32 ≥ 64) from by decide, if_false]
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, show ((1 : UInt64) <<< UInt64.ofNat 32 - 1).toNat = 2 ^ 32 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt h]

theorem win_iff (F : Facts) (v : Val) : F.win v = true ↔ Win F v := by
  simp [Facts.win, Win]

open PtxRun in
/-- **An edge's argument**: its one instruction leaves the edge value in its
    temporary. -/
theorem exec_moveInstr (c : PtxRun.Ctx) (ts : PtxRun.TS) (F : Facts) (vs : Blocks.Vals) (tmp : PtxIr.Reg)
    (t : ClifTy) (a : Val) (b : UInt64) (D : Nat → Prop) (hS : Sim F D vs ts.regs) (hD : D a.id)
    (hv : ∃ ta, Blocks.getV vs a = some (.sc ta b) ∧ (¬ Win F a → ta = t ∧ F.tys.get? a.id = some t))
    (hsc : Scalar t) :
    ∃ x, (exec c (moveInstr F tmp t a)).run ts = .ok (.step, { ts with regs := ts.regs.insert tmp x }) ∧
      EdgeVal F t a b x := by
  obtain ⟨ta, hva, hty⟩ := hv
  by_cases hw : Win F a
  · obtain ⟨off, hoff, hlt, hb⟩ := hS.win a ta b hD hva hw
    have hwin : F.win a = true := (win_iff F a).mpr hw
    refine ⟨off, ?_, fun _ => ⟨hlt, hb⟩, fun h => absurd hw h⟩
    simp only [moveInstr, hwin, if_true]
    rw [exec_mov_reg c ts .b32 tmp (winReg a) off (not_special_val _ _) hoff]
    simp only [PtxIr.Ty.width, mask32_lt off hlt]
  · obtain ⟨rfl, ht⟩ := hty hw
    have hwin : F.win a = false := by
      cases h : F.win a
      · rfl
      · exact absurd ((win_iff F a).mp h) hw
    simp only [moveInstr, hwin, Bool.false_eq_true, if_false]
    cases hl : F.lits.get? a.id with
    | some l =>
        obtain ⟨hil, hraw⟩ := hS.lit a ta b l hD hva ht hl
        have hlb := lit_bits ta l b hil hraw
        refine ⟨litBits ta l, ?_, fun h => absurd h hw, fun _ => hlb⟩
        simp only [Std.HashMap.get?_eq_getElem?] at hl
        simp only [hl]
        cases l with
        | imm k => exact exec_mov_imm c ts _ _ k
        | f32 k => exact exec_mov_f32 c ts _ _ k
        | f64 k => exact exec_mov_f64 c ts _ _ k
        | _ => cases hil
    | none =>
        have hrp := hS.plain a ta b hD hva ht hl hw
        refine ⟨enc ta b, ?_, fun h => absurd h hw, fun _ => rfl⟩
        have hg : (F.tys.get? a.id).getD .i32 = ta := by rw [ht]; rfl
        rw [hg]
        simp only [Std.HashMap.get?_eq_getElem?] at hl
        simp only [hl]
        rw [exec_mov_reg c ts (bits ta) tmp (regOf ta a) (enc ta b) (not_special_val _ _) hrp,
          mask_enc ta b]

/-- Writing an instruction's own register leaves the relation. -/
theorem Sim.own {F : Facts} {D : Nat → Prop} {vs : Blocks.Vals} {regs : Std.HashMap PtxIr.Reg UInt64}
    (h : Sim F D vs regs) (p : String) (t n : Nat) (x : UInt64) :
    Sim F D vs (regs.insert (.own p t n) x) := by
  have k : ∀ c i, (regs.insert (.own p t n) x).get? (.val c i) = regs.get? (.val c i) := by
    intro c i
    simp [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert]
  exact { pres := h.pres, normd := h.normd, flag := h.flag, bound := h.bound
          plain := fun v t b h0 h1 h2 h3 h4 => by rw [regOf, k]; exact h.plain v t b h0 h1 h2 h3 h4
          lit := h.lit
          win := fun v t b h0 h1 h2 => by
            obtain ⟨off, h3, h4, h5⟩ := h.win v t b h0 h1 h2
            exact ⟨off, by rw [winReg, k]; exact h3, h4, h5⟩
          neg := fun v b h0 h1 h2 => by rw [negReg, k]; exact h.neg v b h0 h1 h2 }

open PtxRun in
/-- **Copying out**: each argument's edge value lands in its temporary, and
    no register but the temporaries changes. -/
theorem steps_out (c : PtxRun.Ctx) (F : Facts) (D : Nat → Prop) (vs : Blocks.Vals) (tag : Nat) :
    ∀ (ps : List ((PtxIr.Reg × ClifTy × Bool) × Val)) (n : Nat) (ts : PtxRun.TS),
      Sim F D vs ts.regs →
      (∀ p ∈ ps, Scalar p.1.2.1 ∧ D p.2.id ∧ ∃ ta b, Blocks.getV vs p.2 = some (.sc ta b) ∧
        (¬ Win F p.2 → ta = p.1.2.1 ∧ F.tys.get? p.2.id = some p.1.2.1)) →
      ∃ regs', (steps c (outLines F tag n ps)).run ts = .ok ((), { ts with regs := regs' }) ∧
        Sim F D vs regs' ∧
        (∀ i (h : i < ps.length) ta b, Blocks.getV vs ps[i].2 = some (.sc ta b) →
          ∃ x, regs'.get? (.own "t" tag (n + i)) = some x ∧ EdgeVal F ps[i].1.2.1 ps[i].2 b x) ∧
        (∀ R, (∀ k, n ≤ k → R ≠ .own "t" tag k) → regs'.get? R = ts.regs.get? R)
  | [], n, ts, hS, _ => ⟨ts.regs, by simp [outLines, steps, StateT.run, pure, StateT.pure, Except.pure],
      hS, by simp, fun _ _ => rfl⟩
  | ((d, t, pw), a) :: ps, n, ts, hS, hp => by
      obtain ⟨hsc, hD, ta, b, hva, hty⟩ := hp _ List.mem_cons_self
      obtain ⟨x, hx, hev⟩ := exec_moveInstr c ts F vs (.own "t" tag n) t a b D hS hD ⟨ta, hva, hty⟩ hsc
      let ts1 : PtxRun.TS := { ts with regs := ts.regs.insert (.own "t" tag n) x }
      have keep : ∀ R, R ≠ .own "t" tag n → ts1.regs.get? R = ts.regs.get? R := by
        intro R hR
        simp only [ts1, Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert]
        simp [Ne.symm hR]
      obtain ⟨regs', hrun, hS', hval, hrest⟩ := steps_out c F D vs tag ps (n + 1) ts1 (hS.own "t" tag n x)
        (fun p hpm => hp p (List.mem_cons_of_mem _ hpm))
      refine ⟨regs', ?_, hS', ?_, ?_⟩
      · simp only [outLines, steps, StateT.run, bind, StateT.bind] at hrun ⊢
        simp only [StateT.run] at hx
        simp [hx, Except.bind]; exact hrun
      · intro i hi ta' b' hv'
        cases i with
        | zero =>
            simp only [List.getElem_cons_zero] at hv' ⊢
            rw [hva] at hv'; obtain ⟨-, rfl⟩ := V.sc.inj (Option.some.inj hv')
            refine ⟨x, ?_, hev⟩
            rw [hrest _ (fun k hk h => by cases h; omega)]
            simp [ts1, Std.HashMap.get?_eq_getElem?]
        | succ i =>
            simp only [List.getElem_cons_succ] at hv' ⊢
            obtain ⟨x', hx', hev'⟩ := hval i (by simp at hi; omega) ta' b' hv'
            exact ⟨x', by rw [show n + (i + 1) = n + 1 + i by omega]; exact hx', hev'⟩
      · intro R hR
        rw [hrest R (fun k hk => hR k (by omega)), keep R (hR n (Nat.le_refl _))]


/-- What an edge to a block writes: its arguments into temporaries, the
    temporaries into the block's parameters, and the branch. -/
theorem runLM_edge (F : Facts) (tag : Nat) (s : Out) (b : BlockRef) (args : List Val) (bd : BlockData)
    (hf : F.blocks.find? (·.ref.id == b.id) = some bd) :
    let ps := (dstsOf F bd.params).zip args
    (runLM (edge b args) F tag s).2.lines = s.lines ++
      ((outLines F tag s.n ps ++ inLines ps (tmpsFrom tag s.n ps.length) ++
        [PtxIr.Instr.bra true (.blk b.id)]).map (PtxIr.Line.ins none)).toArray := by
  intro ps
  have h1 := runLM_copyOut F tag ps s
  have h2 := runLM_copyIn F tag ps (runLM (Ptx.copyOut ps) F tag s).1 (runLM (Ptx.copyOut ps) F tag s).2
  simp only [edge, runLM_bind, facts?, runLM_read, hf, runLM_pure, emit, emitL, runLM_modify, blkLbl]
  simp only [ps] at h1 h2 ⊢
  rw [h2.1, h1.2.1, h1.1]
  simp

theorem dstsOf_length (F : Facts) (ps : List (Val × ClifTy)) : (dstsOf F ps).length = ps.length := by
  simp [dstsOf]

theorem dstsOf_get (F : Facts) (ps : List (Val × ClifTy)) (k : Nat) (hk : k < ps.length) :
    (dstsOf F ps)[k]'(by simp [dstsOf_length, hk]) =
      if F.win ps[k].1 then (winReg ps[k].1, .i32, true) else (regOf ps[k].2 ps[k].1, ps[k].2, false) := by
  simp [dstsOf]

/-- A destination is its parameter's register, of some class. -/
theorem dstsOf_reg (F : Facts) (ps : List (Val × ClifTy)) (k : Nat) (hk : k < ps.length) :
    ∃ c, ((dstsOf F ps)[k]'(by simp [dstsOf_length, hk])).1 = .val c ps[k].1.id := by
  rw [dstsOf_get F ps k hk]
  split
  · exact ⟨"rs", rfl⟩
  · exact ⟨_, rfl⟩

/-- Distinct parameters have distinct destinations. -/
theorem dstsOf_nodup (F : Facts) (ps : List (Val × ClifTy)) (hnd : (ps.map (·.1.id)).Nodup) :
    ((dstsOf F ps).map (·.1)).Nodup := by
  induction ps with
  | nil => simp [dstsOf]
  | cons p ps ih =>
      simp only [List.map_cons, List.nodup_cons, List.mem_map] at hnd
      have ih' := ih hnd.2
      simp only [dstsOf, List.map_cons, List.map_map, List.nodup_cons, List.mem_map, Function.comp] at ih' ⊢
      refine ⟨?_, ih'⟩
      rintro ⟨q, hq, hqe⟩
      apply hnd.1
      refine ⟨q, hq, ?_⟩
      split at hqe <;> split at hqe <;> simp_all [winReg, regOf]

/-- A register of a value no parameter is, is no destination. -/
theorem val_not_dst (F : Facts) (ps : List (Val × ClifTy)) (c : String) (n : Nat)
    (hn : ∀ p ∈ ps, p.1.id ≠ n) : PtxIr.Reg.val c n ∉ (dstsOf F ps).map (·.1) := by
  intro h
  simp only [dstsOf, List.map_map, List.mem_map, Function.comp] at h
  obtain ⟨p, hp, he⟩ := h
  apply hn p hp
  split at he <;> simp_all [winReg, regOf]

open PtxRun in
/-- The register `copyIn` writes for a parameter takes, cut to the
    parameter's width, what its temporary holds. -/
theorem steps_in (c : PtxRun.Ctx) :
    ∀ (ps : List ((PtxIr.Reg × ClifTy × Bool) × Val)) (tmps : List PtxIr.Reg) (ts : PtxRun.TS),
      ps.length = tmps.length →
      (∀ r ∈ tmps, (∀ s, r ≠ .special s) ∧ (ts.regs.get? r).isSome) →
      (ps.map (·.1.1)).Nodup →
      (∀ p ∈ ps, p.1.1 ∉ tmps) →
      ∃ regs', (steps c (inLines ps tmps)).run ts = .ok ((), { ts with regs := regs' }) ∧
        (∀ i (h : i < ps.length) (h' : i < tmps.length), regs'.get? ps[i].1.1 =
          (ts.regs.get? tmps[i]).map (mask (bits ps[i].1.2.1).width)) ∧
        (∀ R, R ∉ ps.map (·.1.1) → regs'.get? R = ts.regs.get? R)
  | [], _, ts, _, _, _, _ => ⟨ts.regs, by simp [inLines, steps, StateT.run, pure, StateT.pure, Except.pure],
      by simp, fun _ _ => rfl⟩
  | _ :: _, [], _, hl, _, _, _ => by simp at hl
  | ((d, t, pw), a) :: ps, tmp :: tmps, ts, hl, ht, hnd, hdt => by
      obtain ⟨hs, hsome⟩ := ht tmp List.mem_cons_self
      obtain ⟨x, hx⟩ := Option.isSome_iff_exists.mp hsome
      have he := exec_mov_reg c ts (bits t) d tmp x hs hx
      let ts1 : PtxRun.TS := { ts with regs := ts.regs.insert d (mask (bits t).width x) }
      have keep : ∀ R, R ≠ d → ts1.regs.get? R = ts.regs.get? R := by
        intro R hR
        simp only [ts1, Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert]
        simp [Ne.symm hR]
      have hd : d ∉ tmp :: tmps := hdt _ List.mem_cons_self
      simp only [List.map_cons, List.nodup_cons] at hnd
      obtain ⟨regs', hrun, hval, hrest⟩ := steps_in c ps tmps ts1 (by simpa using hl)
        (fun r hr => by
          obtain ⟨a, b⟩ := ht r (List.mem_cons_of_mem _ hr)
          refine ⟨a, ?_⟩
          rw [keep r (fun h => hd (h ▸ List.mem_cons_of_mem _ hr))]; exact b)
        hnd.2 (fun p hp h => hdt p (List.mem_cons_of_mem _ hp) (List.mem_cons_of_mem _ h))
      refine ⟨regs', ?_, ?_, ?_⟩
      · simp only [inLines, steps, StateT.run, bind, StateT.bind] at hrun ⊢
        simp only [StateT.run] at he
        simp [he, Except.bind]; exact hrun
      · intro i hi hi'
        cases i with
        | zero =>
            simp only [List.getElem_cons_zero]
            rw [hrest d hnd.1]
            simp only [ts1, Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert_self]
            simp only [Std.HashMap.get?_eq_getElem?] at hx; simp [hx]
        | succ i =>
            simp only [List.getElem_cons_succ]
            rw [hval i (by simp at hi; omega) (by simp at hi'; omega)]
            rw [keep _ (fun h => hd (h ▸ List.mem_cons_of_mem _ (List.getElem_mem _)))]
      · intro R hR
        simp only [List.map_cons, List.mem_cons, not_or] at hR
        rw [hrest R hR.2, keep R hR.1]

/-! ## Labels and branches -/

/-- The fold `labelsOf` makes, from any map and starting index. -/
def labelStep : Std.HashMap PtxIr.Lbl Nat → PtxIr.Line × Nat → Std.HashMap PtxIr.Lbl Nat :=
  fun m (l, i) => match l with
    | .label n => m.insert n i
    | _ => m

theorem labelStep_other (m : Std.HashMap PtxIr.Lbl Nat) (l : PtxIr.Line) (i : Nat) (L : PtxIr.Lbl)
    (h : l ≠ .label L) : (labelStep m (l, i)).get? L = m.get? L := by
  cases l with
  | label n =>
      have : n ≠ L := fun e => h (e ▸ rfl)
      simp [labelStep, Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, this]
  | ins _ _ => rfl

/-- A label written at one place only is found there. -/
theorem fold_label : ∀ (xs : List PtxIr.Line) (n j : Nat) (m : Std.HashMap PtxIr.Lbl Nat) (L : PtxIr.Lbl),
    xs[j]? = some (.label L) → (∀ k, xs[k]? = some (.label L) → k = j) →
    ((xs.zipIdx n).foldl labelStep m).get? L = some (n + j)
  | [], _, _, _, _, h, _ => by simp at h
  | x :: xs, n, j, m, L, h, hu => by
      simp only [List.zipIdx_cons, List.foldl_cons]
      cases j with
      | zero =>
          simp at h; subst h
          -- the rest holds no other `L`, so the map keeps what this line put
          have rest : ∀ (ys : List PtxIr.Line) (k : Nat) (m' : Std.HashMap PtxIr.Lbl Nat),
              (∀ i : Nat, ys[i]? ≠ some (PtxIr.Line.label L)) → ((ys.zipIdx k).foldl labelStep m').get? L = m'.get? L := by
            intro ys
            induction ys with
            | nil => intros; rfl
            | cons y ys ih =>
                intro k m' hn
                simp only [List.zipIdx_cons, List.foldl_cons]
                rw [ih _ _ (fun i => by simpa using hn (i + 1))]
                exact labelStep_other _ _ _ _ (by simpa using hn 0)
          rw [rest xs (n + 1) _ (fun i hi => absurd (hu (i + 1) (by simpa using hi)) (by simp))]
          simp [labelStep, Std.HashMap.get?_eq_getElem?]
      | succ j =>
          have := fold_label xs (n + 1) j (labelStep m (x, n)) L (by simpa using h)
            (fun k hk => by have := hu (k + 1) (by simpa using hk); omega)
          rw [this]; congr 1; omega

theorem labelsOf_eq (body : List PtxIr.Line) : PtxRun.labelsOf body = (body.zipIdx 0).foldl labelStep {} := rfl

open PtxRun in
/-- A label line is passed over. -/
theorem go_label (c : PtxRun.Ctx) (fuel pc : Nat) (l : PtxIr.Lbl) (ts : PtxRun.TS)
    (h : c.body[pc]? = some (.label l)) :
    (PtxRun.go c (fuel + 1) pc).run ts = (PtxRun.go c fuel (pc + 1)).run ts := by
  simp [PtxRun.go, h]

open PtxRun in
/-- A branch goes to its label's line. -/
theorem go_bra (c : PtxRun.Ctx) (fuel pc j : Nat) (u : Bool) (l : PtxIr.Lbl) (ts : PtxRun.TS)
    (h : c.body[pc]? = some (.ins none (.bra u l))) (hl : c.labels.get? l = some j) :
    (PtxRun.go c (fuel + 1) pc).run ts = (PtxRun.go c fuel j).run ts := by
  simp only [Std.HashMap.get?_eq_getElem?] at hl
  simp [PtxRun.go, h, guardOk, exec, hl, StateT.run, bind, StateT.bind, pure, StateT.pure, Except.pure,
    Except.bind]

open PtxRun in
/-- A branch under a guard that holds goes to its label's line. -/
theorem go_bra_taken (c : PtxRun.Ctx) (fuel pc j : Nat) (neg : Bool) (p : PtxIr.Reg) (l : PtxIr.Lbl)
    (ts : PtxRun.TS) (hp : ∀ s, p ≠ .special s)
    (h : c.body[pc]? = some (.ins (some (neg, p)) (.bra false l))) (hl : c.labels.get? l = some j)
    (hg : (if neg then ((ts.regs.get? p).getD 0) &&& 1 == 0 else ((ts.regs.get? p).getD 0) &&& 1 == 1) = true) :
    (PtxRun.go c (fuel + 1) pc).run ts = (PtxRun.go c fuel j).run ts := by
  simp only [Std.HashMap.get?_eq_getElem?] at hl hg
  simp only [PtxRun.go, h, guardOk, getReg_plain _ _ hp]
  simp [exec, hl, StateT.run, bind, StateT.bind, pure, StateT.pure, Except.pure, Except.bind, MonadState.get,
    getThe, MonadStateOf.get, StateT.get, hg]

open PtxRun in
/-- A branch under a guard that fails goes on to the next line. -/
theorem go_bra_not (c : PtxRun.Ctx) (fuel pc : Nat) (neg : Bool) (p : PtxIr.Reg) (l : PtxIr.Lbl)
    (ts : PtxRun.TS) (hp : ∀ s, p ≠ .special s)
    (h : c.body[pc]? = some (.ins (some (neg, p)) (.bra false l)))
    (hg : (if neg then ((ts.regs.get? p).getD 0) &&& 1 == 0 else ((ts.regs.get? p).getD 0) &&& 1 == 1) = false) :
    (PtxRun.go c (fuel + 1) pc).run ts = (PtxRun.go c fuel (pc + 1)).run ts := by
  simp only [Std.HashMap.get?_eq_getElem?] at hg
  simp only [PtxRun.go, h, guardOk, getReg_plain _ _ hp]
  simp [StateT.run, bind, StateT.bind, pure, StateT.pure, Except.pure, Except.bind, MonadState.get,
    getThe, MonadStateOf.get, StateT.get, hg]

/-! ## One instruction of a block -/

open Blocks in
/-- What one instruction other than a terminator leaves, when it goes on. -/
def step1 (lc : Sem.Locals) (s : BSt) : Inst → Option BSt
  | .store v a => (doStore s v a none).toOption
  | .storeTyped t v a => (doStore s v a (some t)).toOption
  | .istore8 v a =>
      match getV s.vals v, getV s.vals a with
      | some (.sc _ b), some (.sc _ addr) =>
          (s.world.mem.store addr 1 (b &&& 0xff)).map fun m =>
            { s with world := { obsStore s.world addr 1 (b &&& 0xff) with mem := m } }
      | _, _ => none
  | .call d c args => do
      let vs ← args.mapM (getV s.vals)
      let (res, w') ← callOf lc c vs (obsCall s.world c vs)
      match d, res with
      | some dv, some r => some { vals := setV s.vals dv r, world := w' }
      | none, _ => some { s with world := w' }
      | some _, none => none
  | i => (evalInst s.world.mem s.vals i).map fun (d, r) => { s with vals := setV s.vals d r }

open Blocks in
/-- A block run past an instruction other than a terminator is its step and
    then the rest. -/
theorem runInsts_step (lc : Sem.Locals) (s : BSt) (i : Inst) (rest : List Inst) (r : BSt × Blocks.Next)
    (w : World) (hi : isTerm i = false) (h : runInsts lc s (i :: rest) = .ok r w) :
    ∃ s1, step1 lc s i = some s1 ∧ runInsts lc s1 rest = .ok r w := by
  cases i <;> simp [isTerm] at hi <;> simp only [runInsts] at h <;>
    (repeat' split at h) <;>
    first
    | (simp at h)
    | (exact absurd h failOf_ne_ok)
    | (exact ⟨_, by simp [step1, Except.toOption, *], h⟩)
    | (exact ⟨_, by simp [step1, *], h⟩)

/-! ## The obligation each instruction meets -/

/-- What a ballot's argument says: whether it is non-zero. -/
def ballotBit (v : V) : V := .sc .i32 (if (asBits v).getD 0 != 0 then 1 else 0)

/-- What a launch reads of an observation: a store, or a collective with its
    arguments, a ballot's as whether non-zero; nothing of any other call. -/
def viewObs : Obs → Option Obs
  | .store a n v => some (.store a n v)
  | .call (.intr c) args =>
      if c.isColl then some (.call (.intr c) (if c == .ballot then args.map ballotBit else args)) else none
  | _ => none

/-- **The block run's world and the thread's**: the same apart from their
    observations, and those each made since it began (after the first `bs`,
    and the first `ps`) the same as far as a launch reads them. -/
def WRel (bs ps : Nat) (bw pw : World) : Prop :=
  { bw with obs := [] } = { pw with obs := [] } ∧ bs ≤ bw.obs.length ∧ ps ≤ pw.obs.length ∧
  (bw.obs.reverse.drop bs).filterMap viewObs = (pw.obs.reverse.drop ps).filterMap viewObs

/-- The instructions of straight-line lines. -/
def instrsOf (ls : Array PtxIr.Line) : List PtxIr.Instr :=
  ls.toList.filterMap fun
    | .ins none i => some i
    | _ => none

/-- Lines that run straight on: each an unguarded instruction. -/
def Straight (ls : Array PtxIr.Line) : Prop :=
  ls.toList = (instrsOf ls).map (.ins none)

/-- The values defined after instruction `i`, from `D`. -/
def extD (D : Nat → Prop) (i : Inst) : Nat → Prop := fun n => D n ∨ (Inst.dst? i).any (·.id == n)

/-- Each instruction reads only values defined: in `D`, or by an instruction
    before it. -/
def DefFrom (D : Nat → Prop) : List Inst → Prop
  | [] => True
  | i :: is => (∀ v ∈ Inst.uses i, D v.id) ∧ DefFrom (extD D i) is

/-- What each instruction's edges' targets count on is defined where it is. -/
def TargetsFrom (inOf : Std.HashMap Nat (Std.HashSet Nat)) (D : Nat → Prop) : List Inst → Prop
  | [] => True
  | i :: is => (∀ t ∈ Inst.targets i, ∃ it, inOf.get? t.id = some it ∧ ∀ n ∈ it.toList, D n) ∧
      TargetsFrom inOf (extD D i) is

/-- **The check is sound**: a block it accepts, from a set the run has
    defined, reads only defined values, and its edges' targets count only on
    what is defined. -/
theorem blockStep_sound (inOf : Std.HashMap Nat (Std.HashSet Nat)) :
    ∀ (is : List Inst) (ok : Bool) (s : Std.HashSet Nat) (D : Nat → Prop),
      (is.foldl (blockStep inOf) (ok, s)).1 = true → (∀ n, s.contains n = true → D n) →
      ok = true ∧ DefFrom D is ∧ TargetsFrom inOf D is
  | [], ok, s, D, h, _ => by simpa [DefFrom, TargetsFrom] using h
  | i :: is, ok, s, D, h, hs => by
      simp only [List.foldl_cons] at h
      have hs' : ∀ n, (blockStep inOf (ok, s) i).2.contains n = true → extD D i n := by
        intro n hn
        simp only [blockStep] at hn
        cases hd : Inst.dst? i with
        | none => rw [hd] at hn; exact Or.inl (hs n hn)
        | some d =>
            rw [hd, Std.HashSet.contains_insert] at hn
            rcases Bool.or_eq_true_iff.mp hn with h1 | h1
            · exact Or.inr (by simp [hd]; simpa using h1)
            · exact Or.inl (hs n h1)
      obtain ⟨hok, hdef, htar⟩ := blockStep_sound inOf is (blockStep inOf (ok, s) i).1
        (blockStep inOf (ok, s) i).2 (extD D i) h hs'
      simp only [blockStep, Bool.and_eq_true, List.all_eq_true] at hok
      obtain ⟨⟨hok0, huses⟩, htargets⟩ := hok
      refine ⟨hok0, ⟨fun v hv => hs _ (huses v hv), hdef⟩, ⟨fun t ht => ?_, htar⟩⟩
      have := htargets t ht
      cases hit : inOf.get? t.id with
      | none => rw [hit] at this; simp at this
      | some it =>
          rw [hit] at this
          refine ⟨it, rfl, fun n hn => hs n ?_⟩
          have := (List.all_eq_true.mp this) n hn
          exact this

/-- **The obligation an instruction meets**: from registers holding a block
    run's defined values, the instruction reading only defined ones, and a
    world related to its own whose NaN rule is the hardware's, the
    instruction's lowering is straight-line and
    runs to registers holding the values after its step, its own now defined,
    the worlds still related. -/
def InstOk (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start : Nat) (i : Inst) : Prop :=
  ∀ (tag : Nat) (D : Nat → Prop) (s s1 : Blocks.BSt) (ts : PtxRun.TS),
    step1 lc s i = some s1 → (∀ v ∈ Inst.uses i, D v.id) → Sim F D s.vals ts.regs → WRel start c.start s.world ts.w →
    (lower F tag i).err = none → MemOk c ts.w.mem →
    Straight (lower F tag i).lines ∧
    ∃ ts1, (steps c (instrsOf (lower F tag i).lines)).run ts = .ok ((), ts1) ∧
      Sim F (extD D i) s1.vals ts1.regs ∧ WRel start c.start s1.world ts1.w

/-- Lines `ls` sit in the body from line `pc` on. -/
def At (c : PtxRun.Ctx) (pc : Nat) (ls : List PtxIr.Line) : Prop :=
  ∀ k (h : k < ls.length), c.body[pc + k]? = some ls[k]

theorem At.left {c : PtxRun.Ctx} {pc : Nat} {a b : List PtxIr.Line} (h : At c pc (a ++ b)) : At c pc a :=
  fun k hk => by have := h k (by simp; omega); rwa [List.getElem_append_left hk] at this

theorem At.right {c : PtxRun.Ctx} {pc : Nat} {a b : List PtxIr.Line} (h : At c pc (a ++ b)) :
    At c (pc + a.length) b :=
  fun k hk => by
    have := h (a.length + k) (by simp; omega)
    rw [List.getElem_append_right (by omega)] at this
    simpa [Nat.add_assoc] using this

/-- A block's instructions' lowerings, one after another, from position
    `start`. -/
def linesOf (F : Facts) : Nat → List Inst → List PtxIr.Line
  | _, [] => []
  | start, i :: is => (lower F start i).lines.toList ++ linesOf F (start + 1) is

/-- **A block up to its terminator**: from its instructions' lines, `go`
    runs each instruction before the first terminator, arriving at the
    terminator's lines with the registers and world still related. -/
theorem prefix_go (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (inOf : Std.HashMap Nat (Std.HashSet Nat))
    (hinv : MemInv c (MemOk c)) :
    ∀ (is : List Inst) (start pc : Nat) (D : Nat → Prop) (s : Blocks.BSt) (ts : PtxRun.TS)
      (r : Blocks.BSt × Blocks.Next) (w : World),
      (∀ i ∈ is, isTerm i = false → InstOk F lc c start0 i) →
      (∀ j (hj : j < is.length), (lower F (start + j) is[j]).err = none) →
      DefFrom D is → TargetsFrom inOf D is →
      Blocks.runInsts lc s is = .ok r w →
      At c pc (linesOf F start is) → Sim F D s.vals ts.regs → WRel start0 c.start s.world ts.w →
      MemOk c ts.w.mem →
      ∃ (k : Nat) (D' : Nat → Prop) (s' : Blocks.BSt) (ts' : PtxRun.TS) (K : Nat),
        Blocks.runInsts lc s' (is.drop k) = .ok r w ∧
        (∀ i ∈ (is.drop k).head?, isTerm i = true) ∧
        At c (pc + K) (linesOf F (start + k) (is.drop k)) ∧
        DefFrom D' (is.drop k) ∧ TargetsFrom inOf D' (is.drop k) ∧ Sim F D' s'.vals ts'.regs ∧
        WRel start0 c.start s'.world ts'.w ∧ MemOk c ts'.w.mem ∧
        ∀ fuel, (PtxRun.go c (fuel + K) pc).run ts = (PtxRun.go c fuel (pc + K)).run ts'
  | [], _, _, _, _, _, _, _, _, _, _, _, h, _, _, _, _ => by simp [Blocks.runInsts] at h
  | i :: rest, start, pc, D, s, ts, r, w, hok, hlow, hdef, htar, h, hat, hS, hW, hN => by
      cases ht : isTerm i with
      | true => exact ⟨0, D, s, ts, 0, h, by simp [ht], by simpa using hat, hdef, htar, hS, hW, hN, fun _ => rfl⟩
      | false =>
          obtain ⟨s1, hs1, hrest⟩ := runInsts_step lc s i rest r w ht h
          have herr := hlow 0 (by simp)
          simp only [Nat.add_zero, List.getElem_cons_zero] at herr
          obtain ⟨hst, ts1, hrun, hS1, hW1⟩ :=
            hok i List.mem_cons_self ht start D s s1 ts hs1 hdef.1 hS hW herr hN
          simp only [linesOf] at hat
          have hL := hat.left
          obtain ⟨k, D', s', ts', K, h1, h2, h3, h0, h0', h4, h5, h5n, h6⟩ := prefix_go F lc c start0 inOf hinv rest (start + 1)
            (pc + (lower F start i).lines.toList.length) (extD D i) s1 ts1 r w
            (fun i' hi' => hok i' (List.mem_cons_of_mem _ hi'))
            (fun j hj => by have := hlow (j + 1) (by simp; omega); simpa [Nat.add_assoc, Nat.add_comm 1 j] using this)
            hdef.2 htar.2 hrest hat.right hS1 hW1 (steps_keeps hinv _ ts () ts1 hrun hN)
          refine ⟨k + 1, D', s', ts', (lower F start i).lines.toList.length + K, by simpa using h1,
            by simpa using h2, ?_, by simpa using h0, by simpa using h0', h4, h5, h5n, ?_⟩
          · simpa [Nat.add_assoc, Nat.add_comm 1 k] using h3
          · intro fuel
            have hlen : (lower F start i).lines.toList.length = (instrsOf (lower F start i).lines).length := by
              rw [hst]; simp
            have hgo := go_steps c (instrsOf (lower F start i).lines) pc (fuel + K) ts ts1
              (fun k' hk' => by
                have := hL k' (by rw [hlen]; exact hk')
                rw [this, ← List.getElem?_eq_getElem, hst, List.getElem?_map, List.getElem?_eq_getElem hk']
                rfl) hrun
            rw [← hlen] at hgo
            rw [show fuel + ((lower F start i).lines.toList.length + K) =
                fuel + K + (lower F start i).lines.toList.length by omega, hgo, h6,
              Nat.add_assoc]

/-! ## An edge, whole -/

open PtxRun in
/-- **An edge**: from registers holding the defined values, its moves leave
    registers holding the values with the target's parameters bound to the
    arguments, the parameters now defined. -/
theorem edge_sim (c : PtxRun.Ctx) (F : Facts) (D : Nat → Prop) (vs : Blocks.Vals) (ts : PtxRun.TS)
    (tag n : Nat) (bd : BlockData) (args : List Val) (xs : List V)
    (hS : Sim F D vs ts.regs)
    (hx : args.mapM (Blocks.getV vs) = some xs)
    (hlen : bd.params.length = args.length)
    (hD : ∀ a ∈ args, D a.id)
    (hesc : ∀ p ∈ (dstsOf F bd.params).zip args, F.win p.2 = p.1.2.2)
    (hpar : ∀ p ∈ bd.params, F.lits.get? p.1.id = none ∧ F.negated p.1 = false ∧
      F.tys.get? p.1.id = some p.2 ∧ Scalar p.2)
    (hnd : (bd.params.map (·.1.id)).Nodup)
    (hty : ∀ k (hk : k < bd.params.length) (hk' : k < args.length), F.tys.get? args[k].id = some bd.params[k].2)
    (hmag : ∀ p ∈ bd.params, Unbounded F p.1) :
    ∃ regs', (steps c (outLines F tag n ((dstsOf F bd.params).zip args) ++
        inLines ((dstsOf F bd.params).zip args) (tmpsFrom tag n args.length))).run ts =
        .ok ((), { ts with regs := regs' }) ∧
      Sim F (fun m => D m ∨ ∃ p ∈ bd.params, p.1.id = m) (bindParams vs bd.params xs) regs' := by
  obtain ⟨hxl, hxv⟩ := mapM_getV vs args xs hx
  have hpl : ((dstsOf F bd.params).zip args).length = args.length := by
    simp [dstsOf_length, hlen]
  have hdl : (dstsOf F bd.params).length = args.length := by simp [dstsOf_length, hlen]
  -- the arguments' values, typed as their parameters
  have harg : ∀ k (hk : k < args.length), ∃ b, Blocks.getV vs args[k] = some (.sc bd.params[k].2 b) := by
    intro k hk
    obtain ⟨ta, b, hva, hta⟩ := hS.pres args[k] (hD _ (List.getElem_mem _))
    rw [hty k (by omega) hk] at hta
    cases hta
    exact ⟨b, hva⟩
  -- copying out
  obtain ⟨regs1, hrun1, hS1, hval1, hrest1⟩ := steps_out c F D vs tag ((dstsOf F bd.params).zip args) n ts hS
    (by
      intro p hp
      obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp
      have hka : k < args.length := by omega
      have hkp : k < bd.params.length := by omega
      simp only [List.getElem_zip]
      rw [dstsOf_get F bd.params k hkp]
      obtain ⟨-, -, htp, hsc⟩ := hpar bd.params[k] (List.getElem_mem _)
      obtain ⟨b, hb⟩ := harg k hka
      have hw := hesc _ (List.getElem_mem (h := hk))
      simp only [List.getElem_zip] at hw
      rw [dstsOf_get F bd.params k hkp] at hw
      split
      · rename_i hpw
        simp only [hpw] at hw
        refine ⟨by simp [Scalar], hD _ (List.getElem_mem _), _, b, hb, fun hn => absurd ((win_iff F _).mp hw) hn⟩
      · refine ⟨hsc, hD _ (List.getElem_mem _), _, b, hb, fun _ => ⟨rfl, hty k hkp hka⟩⟩)
  -- copying in
  obtain ⟨regs2, hrun2, hval2, hrest2⟩ := steps_in c ((dstsOf F bd.params).zip args) (tmpsFrom tag n args.length)
    { ts with regs := regs1 } (by simp [tmpsFrom, hpl])
    (by
      intro r hr
      simp only [tmpsFrom, List.mem_map, List.mem_range] at hr
      obtain ⟨i, hi, rfl⟩ := hr
      refine ⟨not_special_own _ _ _, ?_⟩
      obtain ⟨b, hb⟩ := harg i hi
      obtain ⟨x, hx1, -⟩ := hval1 i (by omega) _ b (by simpa [List.getElem_zip] using hb)
      simp only [hx1, Option.isSome_some])
    (by
      have e : ((dstsOf F bd.params).zip args).map (fun x => x.1.1) = (dstsOf F bd.params).map (·.1) := by
        rw [show (fun x : (PtxIr.Reg × ClifTy × Bool) × Val => x.1.1) = (fun y => y.1) ∘ Prod.fst from rfl,
          ← List.map_map, List.map_fst_zip (by omega)]
      rw [e]; exact dstsOf_nodup F bd.params hnd)
    (by
      intro p hp hpt
      obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp
      simp only [List.getElem_zip] at hpt
      obtain ⟨c', hc⟩ := dstsOf_reg F bd.params k (by omega)
      rw [hc] at hpt
      simp [tmpsFrom] at hpt)
  refine ⟨regs2, ?_, ?_⟩
  · rw [steps_append c _ _ ts _ hrun1]; exact hrun2
  have e : ((dstsOf F bd.params).zip args).map (fun x => x.1.1) = (dstsOf F bd.params).map (·.1) := by
    rw [show (fun x : (PtxIr.Reg × ClifTy × Bool) × Val => x.1.1) = (fun y => y.1) ∘ Prod.fst from rfl,
      ← List.map_map, List.map_fst_zip (by omega)]
  -- a register of a value no parameter is keeps what it held
  have keep : ∀ (c' : String) (v : Val), (∀ p ∈ bd.params, p.1.id ≠ v.id) →
      regs2.get? (.val c' v.id) = ts.regs.get? (.val c' v.id) := by
    intro c' v hv
    rw [hrest2 _ (by rw [e]; exact val_not_dst F bd.params c' v.id hv), hrest1 _ (fun k _ h => by cases h)]
  -- a value no parameter is keeps its value
  have keepV : ∀ v : Val, (∀ p ∈ bd.params, p.1.id ≠ v.id) → D v.id →
      Blocks.getV (bindParams vs bd.params xs) v = Blocks.getV vs v := by
    intro v hv hd
    obtain ⟨t, b, hb, -⟩ := hS.pres v hd
    exact getV_bind_other bd.params xs vs v hv (by rw [hb]; rfl)
  -- a parameter: its value, and what its destination holds
  have param : ∀ (v : Val) (k : Nat) (hk : k < bd.params.length), bd.params[k].1.id = v.id →
      v = bd.params[k].1 ∧ ∃ b x, Blocks.getV (bindParams vs bd.params xs) v = some (.sc bd.params[k].2 b) ∧
        regs1.get? (.own "t" tag (n + k)) = some x ∧
        EdgeVal F ((dstsOf F bd.params)[k]'(by simp [dstsOf_length]; omega)).2.1 (args[k]'(by omega)) b x ∧
        regs2.get? ((dstsOf F bd.params)[k]'(by simp [dstsOf_length]; omega)).1 =
          some (PtxRun.mask (bits ((dstsOf F bd.params)[k]'(by simp [dstsOf_length]; omega)).2.1).width x) := by
    intro v k hk hid
    have hv : v = bd.params[k].1 := by cases v; cases h : bd.params[k].1; simp_all
    subst hv
    obtain ⟨b, hb⟩ := harg k (by omega)
    have hxk : xs[k]'(by omega) = .sc bd.params[k].2 b := by
      have := hxv k (by omega) (by omega); rw [hb] at this; exact (Option.some.inj this).symm
    obtain ⟨x, hx1, hev⟩ := hval1 k (by omega) _ b (by simpa [List.getElem_zip] using hb)
    refine ⟨rfl, b, x, ?_, hx1, by simpa [List.getElem_zip] using hev, ?_⟩
    · rw [getV_bind_param bd.params xs vs k hk (by omega) hnd, hxk]
    · have := hval2 k (by omega) (by simp [tmpsFrom]; omega)
      simp only [List.getElem_zip, tmpsFrom, List.getElem_map, List.getElem_range] at this
      rw [this, hx1]; rfl
  -- the argument of a parameter is windowed as the parameter is
  have winArg : ∀ k (hk : k < bd.params.length), F.win (args[k]'(by omega)) = F.win bd.params[k].1 := by
    intro k hk
    have hw := hesc _ (List.getElem_mem (h := (by omega : k < ((dstsOf F bd.params).zip args).length)))
    simp only [List.getElem_zip] at hw
    rw [dstsOf_get F bd.params k hk] at hw
    split at hw <;> simp_all
  exact {
    pres := fun v hd => by
      by_cases hp : ∃ k, ∃ hk : k < bd.params.length, bd.params[k].1.id = v.id
      · obtain ⟨k, hk, hid⟩ := hp
        obtain ⟨rfl, b, x, hv, -⟩ := param v k hk hid
        exact ⟨_, b, hv, (hpar _ (List.getElem_mem _)).2.2.1⟩
      · have hne : ∀ p ∈ bd.params, p.1.id ≠ v.id := fun p hp' h => by
          obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp'; exact hp ⟨k, hk, h⟩
        have hdv : D v.id := hd.resolve_right (fun ⟨p, hp', h⟩ => hne p hp' h)
        rw [keepV v hne hdv]; exact hS.pres v hdv
    normd := fun v t b hd hv => by
      by_cases hp : ∃ k, ∃ hk : k < bd.params.length, bd.params[k].1.id = v.id
      · obtain ⟨k, hk, hid⟩ := hp
        obtain ⟨rfl, b', x, hv', -⟩ := param v k hk hid
        rw [hv'] at hv
        obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hv)
        obtain ⟨b2, hb2⟩ := harg k (by omega)
        have hxk : xs[k]'(by omega) = .sc bd.params[k].2 b2 := by
          have := hxv k (by omega) (by omega); rw [hb2] at this; exact (Option.some.inj this).symm
        have := getV_bind_param bd.params xs vs k hk (by omega) hnd
        rw [hv', hxk] at this
        obtain ⟨-, rfl⟩ := V.sc.inj (Option.some.inj this)
        exact hS.normd _ _ _ (hD _ (List.getElem_mem _)) hb2
      · have hne : ∀ p ∈ bd.params, p.1.id ≠ v.id := fun p hp' h => by
          obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp'; exact hp ⟨k, hk, h⟩
        have hdv : D v.id := hd.resolve_right (fun ⟨p, hp', h⟩ => hne p hp' h)
        rw [keepV v hne hdv] at hv; exact hS.normd v t b hdv hv
    flag := fun v b hd hv => by
      by_cases hp : ∃ k, ∃ hk : k < bd.params.length, bd.params[k].1.id = v.id
      · obtain ⟨k, hk, hid⟩ := hp
        obtain ⟨rfl, b', x, hv', -⟩ := param v k hk hid
        rw [hv'] at hv
        obtain ⟨ht8, rfl⟩ := V.sc.inj (Option.some.inj hv)
        obtain ⟨b2, hb2⟩ := harg k (by omega)
        have hxk : xs[k]'(by omega) = .sc bd.params[k].2 b2 := by
          have := hxv k (by omega) (by omega); rw [hb2] at this; exact (Option.some.inj this).symm
        have := getV_bind_param bd.params xs vs k hk (by omega) hnd
        rw [hv', hxk] at this
        obtain ⟨-, rfl⟩ := V.sc.inj (Option.some.inj this)
        rw [ht8] at hb2
        exact hS.flag _ _ (hD _ (List.getElem_mem _)) hb2
      · have hne : ∀ p ∈ bd.params, p.1.id ≠ v.id := fun p hp' h => by
          obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp'; exact hp ⟨k, hk, h⟩
        have hdv : D v.id := hd.resolve_right (fun ⟨p, hp', h⟩ => hne p hp' h)
        rw [keepV v hne hdv] at hv; exact hS.flag v b hdv hv
    plain := fun v t b hd hv ht hl hw => by
      by_cases hp : ∃ k, ∃ hk : k < bd.params.length, bd.params[k].1.id = v.id
      · obtain ⟨k, hk, hid⟩ := hp
        obtain ⟨rfl, b', x, hv', hx1, hev, hreg⟩ := param v k hk hid
        rw [hv'] at hv
        obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hv)
        have hwf : F.win bd.params[k].1 = false := by
          cases h : F.win bd.params[k].1
          · rfl
          · exact absurd ((win_iff F _).mp h) hw
        have hwa : ¬ Win F (args[k]'(by omega)) := fun h => by
          have := (win_iff F _).mpr h; rw [winArg k hk, hwf] at this; cases this
        simp only [dstsOf_get F bd.params k hk, hwf, Bool.false_eq_true, if_false] at hev hreg
        rw [hreg, hev.2 hwa, mask_enc _ _]
      · have hne : ∀ p ∈ bd.params, p.1.id ≠ v.id := fun p hp' h => by
          obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp'; exact hp ⟨k, hk, h⟩
        have hdv : D v.id := hd.resolve_right (fun ⟨p, hp', h⟩ => hne p hp' h)
        rw [keepV v hne hdv] at hv
        rw [regOf, keep _ v hne]; exact hS.plain v t b hdv hv ht hl hw
    lit := fun v t b l hd hv ht hl => by
      by_cases hp : ∃ k, ∃ hk : k < bd.params.length, bd.params[k].1.id = v.id
      · obtain ⟨k, hk, hid⟩ := hp
        obtain ⟨rfl, -⟩ := param v k hk hid
        rw [(hpar _ (List.getElem_mem _)).1] at hl; cases hl
      · have hne : ∀ p ∈ bd.params, p.1.id ≠ v.id := fun p hp' h => by
          obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp'; exact hp ⟨k, hk, h⟩
        have hdv : D v.id := hd.resolve_right (fun ⟨p, hp', h⟩ => hne p hp' h)
        rw [keepV v hne hdv] at hv; exact hS.lit v t b l hdv hv ht hl
    win := fun v t b hd hv hw => by
      by_cases hp : ∃ k, ∃ hk : k < bd.params.length, bd.params[k].1.id = v.id
      · obtain ⟨k, hk, hid⟩ := hp
        obtain ⟨rfl, b', x, hv', hx1, hev, hreg⟩ := param v k hk hid
        rw [hv'] at hv
        obtain ⟨-, rfl⟩ := V.sc.inj (Option.some.inj hv)
        have hwt : F.win bd.params[k].1 = true := (win_iff F _).mpr hw
        have hwa : Win F (args[k]'(by omega)) := (win_iff F _).mp (by rw [winArg k hk, hwt])
        simp only [dstsOf_get F bd.params k hk, hwt, if_true] at hev hreg
        obtain ⟨hlt, hb⟩ := hev.1 hwa
        refine ⟨x, ?_, hlt, hb⟩
        rw [hreg]; show some (PtxRun.mask 32 x) = some x; rw [mask32_lt x hlt]
      · have hne : ∀ p ∈ bd.params, p.1.id ≠ v.id := fun p hp' h => by
          obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp'; exact hp ⟨k, hk, h⟩
        have hdv : D v.id := hd.resolve_right (fun ⟨p, hp', h⟩ => hne p hp' h)
        rw [keepV v hne hdv] at hv
        obtain ⟨off, h1, h2, h3⟩ := hS.win v t b hdv hv hw
        exact ⟨off, by rw [winReg, keep _ v hne]; exact h1, h2, h3⟩
    neg := fun v b hd hv hn => by
      by_cases hp : ∃ k, ∃ hk : k < bd.params.length, bd.params[k].1.id = v.id
      · obtain ⟨k, hk, hid⟩ := hp
        obtain ⟨rfl, -⟩ := param v k hk hid
        rw [(hpar _ (List.getElem_mem _)).2.1] at hn; cases hn
      · have hne : ∀ p ∈ bd.params, p.1.id ≠ v.id := fun p hp' h => by
          obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp'; exact hp ⟨k, hk, h⟩
        have hdv : D v.id := hd.resolve_right (fun ⟨p, hp', h⟩ => hne p hp' h)
        rw [keepV v hne hdv] at hv
        rw [negReg, keep _ v hne]; exact hS.neg v b hdv hv hn
    bound := fun v t b hd hv => by
      by_cases hp : ∃ k, ∃ hk : k < bd.params.length, bd.params[k].1.id = v.id
      · obtain ⟨k, hk, hid⟩ := hp
        obtain ⟨rfl, -⟩ := param v k hk hid
        exact bounded_unb (hmag _ (List.getElem_mem _)) _
      · have hne : ∀ p ∈ bd.params, p.1.id ≠ v.id := fun p hp' h => by
          obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp'; exact hp ⟨k, hk, h⟩
        have hdv : D v.id := hd.resolve_right (fun ⟨p, hp', h⟩ => hne p hp' h)
        rw [keepV v hne hdv] at hv
        exact hS.bound v t b hdv hv }

/-! ## Terminators -/

/-- What the parameters check says of one block. -/
theorem paramsOk_block (F : Facts) (h : paramsOk F = true) (bd : BlockData) (hb : bd ∈ F.blocks) :
    (bd.params.map (·.1.id)).Nodup ∧ ∀ p ∈ bd.params, F.lits.get? p.1.id = none ∧ F.negated p.1 = false ∧
      F.tys.get? p.1.id = some p.2 ∧ Scalar p.2 := by
  simp only [paramsOk, Bool.and_eq_true] at h
  replace h := h.1.1
  simp only [List.all_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨hnd, hps⟩ := h bd hb
  refine ⟨hnd, fun p hp => ?_⟩
  have := hps p hp
  simp only [beq_iff_eq, Bool.not_eq_true'] at this
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := this
  refine ⟨h1, h2, h3, ?_⟩
  simp [Scalar] at h4 ⊢; omega

/-- What the parameters check says of the entry's: the shared base at offset 0,
    no other bounded or shared. -/
theorem paramsOk_entry (F : Facts) (h : paramsOk F = true) (b0 : BlockData) (bs : List BlockData)
    (hb : F.blocks = b0 :: bs) (k : Nat) (hk : k < b0.params.length) :
    (k = 3 → F.origin.get? b0.params[k].1.id = some .shared ∧ F.mag.get? b0.params[k].1.id = some 0) ∧
    (k ≠ 3 → Unbounded F b0.params[k].1) := by
  simp only [paramsOk, Bool.and_eq_true, hb] at h
  have h2 := h.1.2
  simp only [List.all_eq_true] at h2
  have := h2 (b0.params[k], k) (List.mem_iff_getElem.mpr ⟨k, by simpa using hk, by simp⟩)
  by_cases h3 : k = 3
  · subst h3; simp at this; exact ⟨fun _ => this, fun h => absurd rfl h⟩
  · simp only [h3, if_false, Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at this
    exact ⟨fun h => absurd h h3, fun _ => this⟩

/-- What the parameters check says of a block not the entry's: its parameters
    bound nothing, and no edge goes to the entry. -/
theorem paramsOk_tail (F : Facts) (h : paramsOk F = true) (bd : BlockData) (hb : bd ∈ F.blocks) :
    (bd.ref.id ≠ 0 → ∀ p ∈ bd.params, Unbounded F p.1) ∧ ∀ i ∈ bd.insts, 0 ∉ jumpsTo i := by
  simp only [paramsOk, Bool.and_eq_true] at h
  have h2 := h.2
  simp only [List.all_eq_true, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at h2
  obtain ⟨h3, h4⟩ := h2 bd hb
  refine ⟨fun hne p hp => ?_, fun i hi => by simpa using h4 i hi⟩
  rcases h3 with h3 | h3
  · exact absurd h3 hne
  · exact h3 p hp

/-- **A branch's target bounds nothing**: it is not the entry, whose
    parameters alone are bounded. -/
theorem target_mag (F : Facts) (h : paramsOk F = true) (bd : BlockData) (hb : bd ∈ F.blocks) (i : Inst)
    (hi : i ∈ bd.insts) (t : BlockRef) (ht : t.id ∈ jumpsTo i) (bdt : BlockData)
    (hft : F.blocks.find? (·.ref.id == t.id) = some bdt) : ∀ p ∈ bdt.params, Unbounded F p.1 := by
  have hbt : bdt ∈ F.blocks := List.mem_of_find?_eq_some hft
  have hbid : bdt.ref.id = t.id := by have := List.find?_some hft; simpa using this
  refine (paramsOk_tail F h bdt hbt).1 ?_
  rw [hbid]; intro h0; rw [h0] at ht; exact (paramsOk_tail F h bd hb).2 i hi ht

/-- What a jump lowers to: its edge, from the instruction's first register. -/
theorem lower_jump (F : Facts) (tag : Nat) (t : BlockRef) (args : List Val) (bd : BlockData)
    (hf : F.blocks.find? (·.ref.id == t.id) = some bd) :
    (lower F tag (.jump t args)).lines =
      ((outLines F tag 0 ((dstsOf F bd.params).zip args) ++
        inLines ((dstsOf F bd.params).zip args) (tmpsFrom tag 0 ((dstsOf F bd.params).zip args).length) ++
        [PtxIr.Instr.bra true (.blk t.id)]).map (PtxIr.Line.ins none)).toArray := by
  have he := runLM_edge F tag {} t args bd hf
  simp only [lower_eq, isTerm, if_true, term] at he ⊢
  rw [he]; simp

/-- Instructions sitting unguarded from `pc` on, as `At` says of their lines. -/
theorem At.instrs {c : PtxRun.Ctx} {pc : Nat} {is : List PtxIr.Instr} {rest : List PtxIr.Line}
    (h : At c pc (is.map (PtxIr.Line.ins none) ++ rest)) :
    ∀ k (hk : k < is.length), c.body[pc + k]? = some (.ins none is[k]) := by
  intro k hk
  have := h.left k (by simpa using hk)
  simpa using this

open PtxRun in
/-- **A jump**: from its lines, `go` makes its edge's moves and arrives at its
    target's label, the registers holding the values with the target's
    parameters bound. -/
theorem jump_go (c : PtxRun.Ctx) (F : Facts) (D : Nat → Prop) (vs : Blocks.Vals) (ts : PtxRun.TS)
    (tag pc j : Nat) (t : BlockRef) (args : List Val) (xs : List V) (bd : BlockData)
    (hS : Sim F D vs ts.regs) (hx : args.mapM (Blocks.getV vs) = some xs)
    (hf : F.blocks.find? (·.ref.id == t.id) = some bd) (hpok : paramsOk F = true)
    (hlen : bd.params.length = args.length)
    (hty : ∀ k (hk : k < bd.params.length) (hk' : k < args.length), F.tys.get? args[k].id = some bd.params[k].2)
    (hD : ∀ a ∈ args, D a.id) (hesc : ∀ p ∈ (dstsOf F bd.params).zip args, F.win p.2 = p.1.2.2)
    (hat : At c pc (lower F tag (.jump t args)).lines.toList)
    (hl : c.labels.get? (.blk t.id) = some j) (hmag : ∀ p ∈ bd.params, Unbounded F p.1) :
    ∃ regs', Sim F (fun m => D m ∨ ∃ p ∈ bd.params, p.1.id = m) (bindParams vs bd.params xs) regs' ∧
      ∀ fuel, (PtxRun.go c (fuel + (lower F tag (.jump t args)).lines.size) pc).run ts =
        (PtxRun.go c fuel j).run { ts with regs := regs' } := by
  have hlines := lower_jump F tag t args bd hf
  obtain ⟨hnd, hpar⟩ := paramsOk_block F hpok bd (List.mem_of_find?_eq_some hf)
  have hpl : ((dstsOf F bd.params).zip args).length = args.length := by simp [dstsOf_length, hlen]
  obtain ⟨regs', hrun, hS'⟩ := edge_sim c F D vs ts tag 0 bd args xs hS hx hlen hD hesc hpar hnd hty hmag
  refine ⟨regs', hS', fun fuel => ?_⟩
  rw [hpl] at hlines
  let L := outLines F tag 0 ((dstsOf F bd.params).zip args) ++
    inLines ((dstsOf F bd.params).zip args) (tmpsFrom tag 0 args.length)
  have hls : (lower F tag (.jump t args)).lines.toList =
      L.map (PtxIr.Line.ins none) ++ [PtxIr.Line.ins none (.bra true (.blk t.id))] := by
    rw [hlines]; simp [L]
  have hsz : (lower F tag (.jump t args)).lines.size = L.length + 1 := by
    rw [← Array.length_toList, hls]; simp
  rw [hls] at hat
  have hsteps := go_steps c L pc (fuel + 1) ts { ts with regs := regs' } (At.instrs hat) hrun
  have hbra := go_bra c fuel (pc + L.length) j true (.blk t.id) { ts with regs := regs' }
    (by have := hat.right 0 (by simp); simpa using this) hl
  rw [hsz, show fuel + (L.length + 1) = fuel + 1 + L.length by omega, hsteps, hbra]

/-- An edge takes as many temporaries as it has arguments. -/
theorem runLM_edge_n (F : Facts) (tag : Nat) (s : Out) (b : BlockRef) (args : List Val) (bd : BlockData)
    (hf : F.blocks.find? (·.ref.id == b.id) = some bd) :
    (runLM (edge b args) F tag s).2.n = s.n + ((dstsOf F bd.params).zip args).length := by
  have h1 := runLM_copyOut F tag ((dstsOf F bd.params).zip args) s
  have h2 := runLM_copyIn F tag ((dstsOf F bd.params).zip args)
    (runLM (Ptx.copyOut ((dstsOf F bd.params).zip args)) F tag s).1
    (runLM (Ptx.copyOut ((dstsOf F bd.params).zip args)) F tag s).2
  simp only [edge, runLM_bind, facts?, runLM_read, hf, runLM_pure, emit, emitL, runLM_modify]
  rw [h2.2, h1.2.2]

/-- The guard a branch on flag `c` tests: the flag's negation where a
    comparison bound it, else the flag itself, negated. -/
def guardOf (F : Facts) (c : Val) : Bool × PtxIr.Reg :=
  if F.negated c then (false, negReg c) else (true, regOf .i8 c)

/-- The lines of an edge, from temporary count `n`. -/
def edgeLines (F : Facts) (tag n : Nat) (t : BlockRef) (bd : BlockData) (args : List Val) : List PtxIr.Line :=
  (outLines F tag n ((dstsOf F bd.params).zip args) ++
    inLines ((dstsOf F bd.params).zip args) (tmpsFrom tag n ((dstsOf F bd.params).zip args).length) ++
    [PtxIr.Instr.bra true (.blk t.id)]).map (PtxIr.Line.ins none)

open PtxRun in
/-- **An edge's lines**: `go` makes its moves and arrives at its target's
    label, the registers holding the values with the target's parameters
    bound. -/
theorem edge_go (c : PtxRun.Ctx) (F : Facts) (D : Nat → Prop) (vs : Blocks.Vals) (ts : PtxRun.TS)
    (tag n pc j : Nat) (t : BlockRef) (args : List Val) (xs : List V) (bd : BlockData)
    (hS : Sim F D vs ts.regs) (hx : args.mapM (Blocks.getV vs) = some xs)
    (hf : F.blocks.find? (·.ref.id == t.id) = some bd) (hpok : paramsOk F = true)
    (hlen : bd.params.length = args.length)
    (hty : ∀ k (hk : k < bd.params.length) (hk' : k < args.length), F.tys.get? args[k].id = some bd.params[k].2)
    (hD : ∀ a ∈ args, D a.id) (hesc : ∀ p ∈ (dstsOf F bd.params).zip args, F.win p.2 = p.1.2.2)
    (hat : At c pc (edgeLines F tag n t bd args))
    (hl : c.labels.get? (.blk t.id) = some j) (hmag : ∀ p ∈ bd.params, Unbounded F p.1) :
    ∃ regs', Sim F (fun m => D m ∨ ∃ p ∈ bd.params, p.1.id = m) (bindParams vs bd.params xs) regs' ∧
      ∀ fuel, (PtxRun.go c (fuel + (edgeLines F tag n t bd args).length) pc).run ts =
        (PtxRun.go c fuel j).run { ts with regs := regs' } := by
  obtain ⟨hnd, hpar⟩ := paramsOk_block F hpok bd (List.mem_of_find?_eq_some hf)
  have hpl : ((dstsOf F bd.params).zip args).length = args.length := by simp [dstsOf_length, hlen]
  obtain ⟨regs', hrun, hS'⟩ := edge_sim c F D vs ts tag n bd args xs hS hx hlen hD hesc hpar hnd hty hmag
  refine ⟨regs', hS', fun fuel => ?_⟩
  let L := outLines F tag n ((dstsOf F bd.params).zip args) ++
    inLines ((dstsOf F bd.params).zip args) (tmpsFrom tag n args.length)
  have hls : edgeLines F tag n t bd args =
      L.map (PtxIr.Line.ins none) ++ [PtxIr.Line.ins none (.bra true (.blk t.id))] := by
    simp [edgeLines, L, hpl]
  rw [hls] at hat ⊢
  have hsteps := go_steps c L pc (fuel + 1) ts { ts with regs := regs' } (At.instrs hat) hrun
  have hbra := go_bra c fuel (pc + L.length) j true (.blk t.id) { ts with regs := regs' }
    (by have := hat.right 0 (by simp); simpa using this) hl
  simp only [List.length_append, List.length_map, List.length_singleton]
  rw [show fuel + (L.length + 1) = fuel + 1 + L.length by omega, hsteps, hbra]

/-- What a conditional branch lowers to: the guarded branch to its other
    side, the first edge, the other side's label, the second edge. -/
theorem lower_brif (F : Facts) (tag : Nat) (cnd : Val) (tb eb : BlockRef) (ta ea : List Val)
    (bdT bdE : BlockData) (hwc : ¬ Win F cnd) (htc : F.tys.get? cnd.id = some .i8)
    (hlc : F.lits.get? cnd.id = none)
    (hT : F.blocks.find? (·.ref.id == tb.id) = some bdT) (hE : F.blocks.find? (·.ref.id == eb.id) = some bdE) :
    (lower F tag (.brif cnd tb ta eb ea)).lines.toList =
      [.ins (some (guardOf F cnd)) (.bra false (.els tag))] ++ edgeLines F tag 0 tb bdT ta ++
      [.label (.els tag)] ++ edgeLines F tag ((dstsOf F bdT.params).zip ta).length eb bdE ea := by
  have hr := runLM_r_plain F tag {} cnd .i8 hwc htc hlc
  simp only [lower_eq, isTerm, if_true, term, runLM_bind, hr, runLM_read, facts?, runLM_pure]
  have e1 := fun s => runLM_edge F tag s tb ta bdT hT
  have e2 := fun s => runLM_edge F tag s eb ea bdE hE
  have n1 := fun s => runLM_edge_n F tag s tb ta bdT hT
  simp only [guardOf]
  split <;> simp only [runLM_bind, emitL, runLM_modify] <;> rw [e2, n1, e1] <;> simp [edgeLines]

/-- **The guard holds exactly when the flag is zero**, whichever register
    it reads. -/
theorem guard_eval (F : Facts) (D : Nat → Prop) (vs : Blocks.Vals) (regs : Std.HashMap PtxIr.Reg UInt64)
    (cnd : Val) (b : UInt64) (hS : Sim F D vs regs) (hc : Blocks.getV vs cnd = some (.sc .i8 b))
    (hDc : D cnd.id) (htc : F.tys.get? cnd.id = some .i8) (hlc : F.lits.get? cnd.id = none) (hwc : ¬ Win F cnd) :
    ∃ x, regs.get? (guardOf F cnd).2 = some x ∧
      (if (guardOf F cnd).1 then x &&& 1 == 0 else x &&& 1 == 1) = (b == 0) := by
  have hn : b &&& 0xff = b := hS.normd cnd .i8 b hDc hc
  have hz : (b &&& 0xff = 0) ↔ b = 0 := by rw [hn]
  unfold guardOf
  split
  · rename_i hneg
    refine ⟨_, hS.neg cnd b hDc hc hneg, ?_⟩
    by_cases hb : b = 0
    · subst hb; simp
    · have h255 : ¬ (b &&& 255 = 0) := fun h => hb (hz.mp h)
      simp [h255, hb]
  · refine ⟨_, hS.plain cnd .i8 b hDc hc htc hlc hwc, ?_⟩
    by_cases hb : b = 0
    · subst hb; simp [enc]
    · have h255 : ¬ (b &&& 255 = 0) := fun h => hb (hz.mp h)
      simp [enc, h255, hb]

open PtxRun in
/-- **A conditional branch**: from its lines, `go` arrives at the label of
    the target the block semantics chooses, the registers holding the values
    with that target's parameters bound. -/
theorem brif_go (c : PtxRun.Ctx) (F : Facts) (D : Nat → Prop) (vs : Blocks.Vals) (ts : PtxRun.TS)
    (tag pc : Nat) (cnd : Val) (tb eb : BlockRef) (ta ea : List Val) (bdT bdE : BlockData) (b : UInt64)
    (hS : Sim F D vs ts.regs) (hc : Blocks.getV vs cnd = some (.sc .i8 b)) (hDc : D cnd.id)
    (htc : F.tys.get? cnd.id = some .i8) (hlc : F.lits.get? cnd.id = none) (hwc : ¬ Win F cnd)
    (hT : F.blocks.find? (·.ref.id == tb.id) = some bdT) (hE : F.blocks.find? (·.ref.id == eb.id) = some bdE)
    (hpok : paramsOk F = true)
    (escT : ∀ p ∈ (dstsOf F bdT.params).zip ta, F.win p.2 = p.1.2.2)
    (escE : ∀ p ∈ (dstsOf F bdE.params).zip ea, F.win p.2 = p.1.2.2)
    (hat : At c pc (lower F tag (.brif cnd tb ta eb ea)).lines.toList)
    (hlEls : c.labels.get? (.els tag) = some (pc + 1 + (edgeLines F tag 0 tb bdT ta).length))
    (tgt : BlockRef) (args : List Val) (xs : List V) (bd : BlockData) (j : Nat)
    (hsel : (if Sem.isTrue (.sc .i8 b) then (tb, ta) else (eb, ea)) = (tgt, args))
    (hx : args.mapM (Blocks.getV vs) = some xs) (hbd : F.blocks.find? (·.ref.id == tgt.id) = some bd)
    (hlen : bd.params.length = args.length)
    (hty : ∀ k (hk : k < bd.params.length) (hk' : k < args.length), F.tys.get? args[k].id = some bd.params[k].2)
    (hD : ∀ a ∈ args, D a.id) (hl : c.labels.get? (.blk tgt.id) = some j)
    (hmagT : ∀ p ∈ bdT.params, Unbounded F p.1) (hmagE : ∀ p ∈ bdE.params, Unbounded F p.1) :
    ∃ K regs', Sim F (fun m => D m ∨ ∃ p ∈ bd.params, p.1.id = m) (bindParams vs bd.params xs) regs' ∧
      ∀ fuel, (PtxRun.go c (fuel + K) pc).run ts = (PtxRun.go c fuel j).run { ts with regs := regs' } := by
  have hlines := lower_brif F tag cnd tb eb ta ea bdT bdE hwc htc hlc hT hE
  rw [hlines] at hat
  simp only [List.append_assoc] at hat
  obtain ⟨x, hx1, hg⟩ := guard_eval F D vs ts.regs cnd b hS hc hDc htc hlc hwc
  have hgl : c.body[pc]? = some (.ins (some (guardOf F cnd)) (.bra false (.els tag))) := by
    have := hat 0 (by simp); simpa using this
  have hgp : ∀ s, (guardOf F cnd).2 ≠ .special s := by
    intro s h; unfold guardOf at h; split at h <;> cases h
  have h1 := hat.right
  simp only [List.length_singleton] at h1
  have hatT := h1.left
  have h2 := h1.right
  have hlab : c.body[pc + 1 + (edgeLines F tag 0 tb bdT ta).length]? = some (.label (.els tag)) := by
    have := h2 0 (by simp); simpa using this
  have hatE := h2.right
  simp only [List.length_singleton] at hatE
  have hcond : (if (guardOf F cnd).1 then ((ts.regs.get? (guardOf F cnd).2).getD 0) &&& 1 == 0
      else ((ts.regs.get? (guardOf F cnd).2).getD 0) &&& 1 == 1) = (b == 0) := by
    rw [hx1]; exact hg
  by_cases hb : b = 0
  · -- the flag is zero: the guard holds, to the second edge
    subst hb
    simp only [Sem.isTrue, bne_self_eq_false, Bool.false_eq_true, if_false, Prod.mk.injEq] at hsel
    obtain ⟨rfl, rfl⟩ := hsel
    rw [hE] at hbd; cases hbd
    obtain ⟨regs', hS', hgo⟩ := edge_go c F D vs ts tag _ _ j eb ea xs bdE hS hx hE hpok hlen hty hD escE hatE hl hmagE
    refine ⟨1 + 1 + (edgeLines F tag ((dstsOf F bdT.params).zip ta).length eb bdE ea).length, regs', hS',
      fun fuel => ?_⟩
    have ht := go_bra_taken c (fuel + (edgeLines F tag ((dstsOf F bdT.params).zip ta).length eb bdE ea).length + 1)
      pc _ (guardOf F cnd).1 (guardOf F cnd).2 (.els tag) ts hgp hgl hlEls (by rw [hcond]; rfl)
    rw [show fuel + (1 + 1 + (edgeLines F tag ((dstsOf F bdT.params).zip ta).length eb bdE ea).length) =
      fuel + (edgeLines F tag ((dstsOf F bdT.params).zip ta).length eb bdE ea).length + 1 + 1 by omega, ht,
      go_label c _ _ _ ts hlab, hgo]
  · -- the flag is not zero: the guard fails, to the first edge
    have hT' : Sem.isTrue (.sc .i8 b) = true := by simp [Sem.isTrue, hb]
    simp only [hT', if_true, Prod.mk.injEq] at hsel
    obtain ⟨rfl, rfl⟩ := hsel
    rw [hT] at hbd; cases hbd
    obtain ⟨regs', hS', hgo⟩ := edge_go c F D vs ts tag 0 _ j tb ta xs bdT hS hx hT hpok hlen hty hD escT hatT hl hmagT
    refine ⟨(edgeLines F tag 0 tb bdT ta).length + 1, regs', hS', fun fuel => ?_⟩
    have hn := go_bra_not c (fuel + (edgeLines F tag 0 tb bdT ta).length) pc (guardOf F cnd).1
      (guardOf F cnd).2 (.els tag) ts hgp hgl (by rw [hcond]; simp [hb])
    rw [← Nat.add_assoc, hn, hgo]

/-- A return lowers to `ret`. -/
theorem lower_ret (F : Facts) (tag : Nat) : (lower F tag (.ret none)).lines.toList = [.ins none .ret] := by
  simp [lower_eq, isTerm, term, emit, emitL]

open PtxRun in
/-- `ret` ends the thread where it stands. -/
theorem go_ret (c : PtxRun.Ctx) (fuel pc : Nat) (ts : PtxRun.TS) (h : c.body[pc]? = some (.ins none .ret)) :
    (PtxRun.go c (fuel + 1) pc).run ts = .ok ((), ts) := by
  simp [PtxRun.go, h, guardOk, exec, StateT.run, bind, StateT.bind, pure, StateT.pure, Except.pure, Except.bind]

/-- A label written once in a body: its line is the only one writing it. -/
theorem label_unique (body : List PtxIr.Line) (h : (labelsIn body).Nodup) (L : PtxIr.Lbl) (i : Nat)
    (hi : body[i]? = some (.label L)) : ∀ k, body[k]? = some (.label L) → k = i := by
  induction body generalizing i with
  | nil => simp at hi
  | cons l body ih =>
      intro k hk
      cases l with
      | label n =>
          simp only [labelsIn, List.filterMap_cons, List.nodup_cons] at h
          cases i with
          | zero =>
              simp at hi; subst hi
              cases k with
              | zero => rfl
              | succ k =>
                  simp at hk
                  exact (h.1 (List.mem_filterMap.mpr ⟨.label n, List.mem_of_getElem? hk, rfl⟩)).elim
          | succ i =>
              cases k with
              | zero =>
                  simp at hk; subst hk; simp at hi
                  exact (h.1 (List.mem_filterMap.mpr ⟨.label n, List.mem_of_getElem? hi, rfl⟩)).elim
              | succ k => simp at hi hk; exact congrArg (· + 1) (ih h.2 i hi k hk)
      | ins g ins =>
          simp only [labelsIn, List.filterMap_cons] at h
          cases i with
          | zero => simp at hi
          | succ i =>
              cases k with
              | zero => simp at hk
              | succ k => simp at hi hk; exact congrArg (· + 1) (ih h i hi k hk)

/-- **Every label line is where its label goes**, the labels written once. -/
theorem labelsOf_line (body : List PtxIr.Line) (h : (labelsIn body).Nodup) (L : PtxIr.Lbl) (i : Nat)
    (hi : body[i]? = some (.label L)) : (PtxRun.labelsOf body).get? L = some i := by
  rw [labelsOf_eq]
  have := fold_label body 0 i {} L hi (label_unique body h L i hi)
  simpa using this

/-! ## What the checks say of one instruction -/

/-- What edge typing says of one edge. -/
def EdgeTyped (F : Facts) (t : BlockRef) (args : List Val) : Prop :=
  ∃ bd, F.blocks.find? (·.ref.id == t.id) = some bd ∧ bd.params.length = args.length ∧
    (∀ k (hk : k < bd.params.length) (hk' : k < args.length), F.tys.get? args[k].id = some bd.params[k].2) ∧
    ∀ p ∈ (dstsOf F bd.params).zip args, F.win p.2 = p.1.2.2

theorem edgeTyped_of (F : Facts) (t : BlockRef) (args : List Val)
    (h : (match F.blocks.find? (·.ref.id == t.id) with
      | some bd => bd.params.length == args.length &&
          (bd.params.zip args).all fun ((pv, pt), a) => F.tys.get? a.id == some pt && F.win a == F.win pv
      | none => false) = true) : EdgeTyped F t args := by
  cases hf : F.blocks.find? (·.ref.id == t.id) with
  | none => simp [hf] at h
  | some bd =>
      simp only [hf, Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at h
      have hk2 : ∀ k (hk : k < bd.params.length) (hk' : k < args.length),
          F.tys.get? args[k].id = some bd.params[k].2 ∧ F.win args[k] = F.win bd.params[k].1 := by
        intro k hk hk'
        have hm : (bd.params[k], args[k]) ∈ bd.params.zip args := by
          rw [← List.getElem_zip (h := by simp; omega)]; exact List.getElem_mem _
        have := h.2 _ hm
        simpa using this
      refine ⟨bd, hf, h.1, fun k hk hk' => (hk2 k hk hk').1, fun p hp => ?_⟩
      obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp
      have hk1 : k < bd.params.length := by simp [dstsOf_length] at hk; omega
      have hk3 : k < args.length := by simp at hk; omega
      simp only [List.getElem_zip, dstsOf_get F bd.params k hk1]
      rw [(hk2 k hk1 hk3).2]
      split <;> simp_all

theorem edgesTyped_jump (F : Facts) (h : edgesTyped F = true) (b : BlockData) (hb : b ∈ F.blocks)
    (t : BlockRef) (args : List Val) (hi : Inst.jump t args ∈ b.insts) : EdgeTyped F t args := by
  simp only [edgesTyped, List.all_eq_true] at h
  exact edgeTyped_of F t args (h b hb _ hi)

theorem edgesTyped_brif (F : Facts) (h : edgesTyped F = true) (b : BlockData) (hb : b ∈ F.blocks)
    (cnd : Val) (tb eb : BlockRef) (ta ea : List Val) (hi : Inst.brif cnd tb ta eb ea ∈ b.insts) :
    EdgeTyped F tb ta ∧ EdgeTyped F eb ea := by
  simp only [edgesTyped, List.all_eq_true] at h
  have := h b hb _ hi
  simp only [Bool.and_eq_true] at this
  exact ⟨edgeTyped_of F tb ta this.1, edgeTyped_of F eb ea this.2⟩

theorem condsOk_brif (F : Facts) (h : condsOk F = true) (b : BlockData) (hb : b ∈ F.blocks)
    (cnd : Val) (tb eb : BlockRef) (ta ea : List Val) (hi : Inst.brif cnd tb ta eb ea ∈ b.insts) :
    F.tys.get? cnd.id = some .i8 ∧ F.lits.get? cnd.id = none ∧ ¬ Win F cnd := by
  simp only [condsOk, List.all_eq_true] at h
  have := h b hb _ hi
  simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true'] at this
  exact ⟨this.1.1, this.1.2, fun hw => by rw [(win_iff F cnd).mpr hw] at this; exact absurd this.2 (by simp)⟩

/-- The set a block's check starts from: what is defined on entry, and its
    parameters. -/
theorem start_contains (inn : Std.HashSet Nat) : ∀ (ps : List (Val × ClifTy)) (n : Nat),
    (ps.foldl (fun s (v, _) => s.insert v.id) inn).contains n = true → inn.contains n = true ∨ ∃ p ∈ ps, p.1.id = n
  | [], n, h => Or.inl (by simpa using h)
  | (v, t) :: ps, n, h => by
      simp only [List.foldl_cons] at h
      rcases start_contains (inn.insert v.id) ps n h with h1 | ⟨p, hp, he⟩
      · rw [Std.HashSet.contains_insert] at h1
        rcases Bool.or_eq_true_iff.mp h1 with h2 | h2
        · exact Or.inr ⟨(v, t), List.mem_cons_self, by simpa using h2⟩
        · exact Or.inl h2
      · exact Or.inr ⟨p, List.mem_cons_of_mem _ hp, he⟩

/-! ## Blocks, one after another -/

/-- A return with a value is not lowered: kernels return nothing. -/
theorem lower_ret_some (F : Facts) (tag : Nat) (v : Val) : (lower F tag (.ret (some v))).err ≠ none := by
  simp [lower_eq, isTerm, term, fail]

/-- **Where the blocks sit** in a thread's body: each block's label line,
    then its instructions' lines; and every label line where its label goes. -/
structure Layout (F : Facts) (c : PtxRun.Ctx) (pos start : Nat → Nat) : Prop where
  lbl : ∀ L i, c.body[i]? = some (.label L) → c.labels.get? L = some i
  blk : ∀ b ∈ F.blocks, c.body[pos b.ref.id]? = some (.label (.blk b.ref.id)) ∧
    At c (pos b.ref.id + 1) (linesOf F (start b.ref.id) b.insts)

/-- **What a function's blocks have**: their layout, each instruction's
    obligation met, no lowering refused, and the compiler's checks passed. -/
structure Good (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (pos start : Nat → Nat)
    (inOf : Std.HashMap Nat (Std.HashSet Nat)) : Prop where
  lay : Layout F c pos start
  ok : ∀ b ∈ F.blocks, ∀ i ∈ b.insts, isTerm i = false → InstOk F lc c start0 i
  low : ∀ b ∈ F.blocks, ∀ j (hj : j < b.insts.length), (lower F (start b.ref.id + j) b.insts[j]).err = none
  inn : ∀ b ∈ F.blocks, ∃ inn, inOf.get? b.ref.id = some inn ∧ blockOk inOf inn b = true
  params : paramsOk F = true
  typed : edgesTyped F = true
  conds : condsOk F = true

theorem lbl_blk {F : Facts} {c : PtxRun.Ctx} {pos start : Nat → Nat} (hL : Layout F c pos start)
    (b : BlockData) (hb : b ∈ F.blocks) : c.labels.get? (.blk b.ref.id) = some (pos b.ref.id) :=
  hL.lbl _ _ (hL.blk b hb).1

open PtxRun in
/-- **A function's blocks, run one after another**: if the block program runs
    from a block to its end, then from that block's lines `go` runs to its
    end, the worlds related, given registers holding the block's values with
    its parameters bound and what the block counts on defined. -/
theorem run_sim (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (pos start : Nat → Nat)
    (inOf : Std.HashMap Nat (Std.HashSet Nat)) (hG : Good F lc c start0 pos start inOf) (f : FuncData)
    (hfb : F.blocks = f.blocks) (hinv : MemInv c (MemOk c)) :
    ∀ (steps : Nat) (st : Blocks.BSt) (blk : Nat) (args : List V) (r : Option V) (w : World),
      Blocks.runFrom lc f steps st blk args = .ok r w →
      ∀ (bd : BlockData) (D : Nat → Prop) (ts : PtxRun.TS) (inn : Std.HashSet Nat),
        f.blocks.find? (·.ref.id == blk) = some bd → inOf.get? blk = some inn →
        (∀ n, inn.contains n = true → D n) → (∀ p ∈ bd.params, D p.1.id) →
        Sim F D (bindParams st.vals bd.params args) ts.regs → WRel start0 c.start st.world ts.w →
        MemOk c ts.w.mem →
        ∃ N ts', WRel start0 c.start w ts'.w ∧ ∀ fuel, (PtxRun.go c (fuel + N) (pos blk + 1)).run ts = .ok ((), ts')
  | 0, _, _, _, _, _, h, _, _, _, _, _, _, _, _, _, _, _ => by simp [Blocks.runFrom] at h
  | steps + 1, st, blk, args, r, w, h, bd, D, ts, inn, hfind, hinn, hD, hDp, hS, hW, hN => by
      simp only [Blocks.runFrom, hfind] at h
      split at h
      · simp at h
      rename_i hlenb
      simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hlenb
      have hb : bd ∈ F.blocks := hfb ▸ List.mem_of_find?_eq_some hfind
      have hblk : bd.ref.id = blk := by
        have := List.find?_some hfind; simpa using this
      subst hblk
      obtain ⟨inn', hinn', hok⟩ := hG.inn bd hb
      rw [hinn] at hinn'; cases hinn'
      -- the block's check, from what is defined on entry and the parameters
      have hst := blockStep_sound inOf bd.insts true _ D hok (fun n hn => by
        rcases start_contains inn bd.params n hn with h1 | ⟨p, hp, rfl⟩
        · exact hD n h1
        · exact hDp p hp)
      obtain ⟨-, hdef, htar⟩ := hst
      cases hri : Blocks.runInsts lc { st with vals := bindParams st.vals bd.params args } bd.insts with
      | stuck _ => simp only [bindParams] at hri; simp [hri] at h
      | misuse _ => simp only [bindParams] at hri; simp [hri] at h
      | fault _ => simp only [bindParams] at hri; simp [hri] at h
      | ok p w1 =>
      obtain ⟨s1, nx⟩ := p
      have hri' : Blocks.runInsts lc { st with vals := bindParams st.vals bd.params args } bd.insts =
          .ok (s1, nx) w1 := hri
      simp only [bindParams] at hri
      rw [hri] at h
      obtain ⟨hlay, hatb⟩ := hG.lay.blk bd hb
      obtain ⟨k, D', s', ts', K, h1, h2, h3, h4, h4', h5, h6, h6n, h7⟩ := prefix_go F lc c start0 inOf hinv bd.insts
        (start bd.ref.id) (pos bd.ref.id + 1) D { st with vals := bindParams st.vals bd.params args } ts (s1, nx) w1
        (fun i hi ht => hG.ok bd hb i hi ht)
        (fun j hj => hG.low bd hb j hj) hdef htar hri' hatb hS hW hN
      cases hdk : bd.insts.drop k with
      | nil => rw [hdk] at h1; simp [Blocks.runInsts] at h1
      | cons i rest =>
      rw [hdk] at h1 h2 h3 h4 h4'
      have hik : k < bd.insts.length := by
        have := congrArg List.length hdk; simp at this; omega
      have hik' : bd.insts[k] = i := by
        have := congrArg (·[0]?) hdk; simp [List.getElem?_drop] at this
        obtain ⟨_, h⟩ := List.getElem?_eq_some_iff.mp this; exact h
      have hmem : i ∈ bd.insts := hik' ▸ List.getElem_mem _
      have herr := hG.low bd hb k hik
      rw [hik'] at herr
      simp only [linesOf] at h3
      have hatI := h3.left
      simp only [List.head?_cons, Option.mem_def, Option.some.injEq, forall_eq'] at h2
      cases i
      case ret v =>
        cases v with
        | some v => exact absurd herr (lower_ret_some F _ v)
        | none =>
          simp only [Blocks.runInsts, Sem.Outcome.ok.injEq, Prod.mk.injEq] at h1
          obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h1
          simp only [Sem.Outcome.ok.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          refine ⟨K + 1, ts', h6, fun fuel => ?_⟩
          rw [show fuel + (K + 1) = fuel + 1 + K by omega, h7]
          exact go_ret c fuel _ ts' (by have := hatI 0 (by simp [lower_ret]); simpa [lower_ret] using this)
      case jump t a =>
        simp only [Blocks.runInsts] at h1
        cases hm : a.mapM (Blocks.getV s'.vals) with
        | none => simp [hm] at h1
        | some xs =>
        simp only [hm, Sem.Outcome.ok.injEq, Prod.mk.injEq] at h1
        obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h1
        simp only at h
        obtain ⟨bdt, hft, hlen, hty, hesc⟩ := edgesTyped_jump F hG.typed bd hb t a hmem
        have hbt : bdt ∈ F.blocks := List.mem_of_find?_eq_some hft
        have hbid : bdt.ref.id = t.id := by have := List.find?_some hft; simpa using this
        obtain ⟨regs', hS', hgo⟩ := jump_go c F D' s'.vals ts' (start bd.ref.id + k) (pos bd.ref.id + 1 + K)
          (pos bdt.ref.id) t a xs bdt h5 hm hft hG.params hlen hty (fun v hv => h4.1 v hv) hesc hatI
          (hbid ▸ lbl_blk hG.lay bdt hbt) (target_mag F hG.params bd hb _ hmem t (by simp [jumpsTo]) bdt hft)
        obtain ⟨it, hit, hitD⟩ := h4'.1 t (by simp [Inst.targets])
        obtain ⟨N', ts'', hW', hgo'⟩ := run_sim F lc c start0 pos start inOf hG f hfb hinv steps { s' with world := s'.world }
          t.id xs r w h bdt (fun m => D' m ∨ ∃ p ∈ bdt.params, p.1.id = m) { ts' with regs := regs' } it
          (hfb ▸ hft) hit (fun n hn => Or.inl (hitD n (by simpa using hn))) (fun p hp => Or.inr ⟨p, hp, rfl⟩)
          hS' h6 h6n
        refine ⟨K + (lower F (start bd.ref.id + k) (.jump t a)).lines.size + 1 + N', ts'', hW', fun fuel => ?_⟩
        rw [show fuel + (K + (lower F (start bd.ref.id + k) (.jump t a)).lines.size + 1 + N') =
            fuel + N' + 1 + (lower F (start bd.ref.id + k) (.jump t a)).lines.size + K by omega, h7, hgo,
          go_label c _ _ _ _ (hG.lay.blk bdt hbt).1, hbid]
        exact hgo' fuel
      case brif cnd tb ta eb ea =>
        obtain ⟨htc, hlc, hwc⟩ := condsOk_brif F hG.conds bd hb cnd tb eb ta ea hmem
        have hDc : D' cnd.id := h4.1 cnd (by simp [Inst.uses])
        obtain ⟨tc, bc, hvc, htyc⟩ := h5.pres cnd hDc
        rw [htc] at htyc; cases htyc
        obtain ⟨⟨bdT, hT, hlenT, htyT, hwT⟩, ⟨bdE, hE, hlenE, htyE, hwE⟩⟩ :=
          edgesTyped_brif F hG.typed bd hb cnd tb eb ta ea hmem
        obtain ⟨tgt, args, hsel⟩ : ∃ tgt args,
            (if Sem.isTrue (.sc .i8 bc) then (tb, ta) else (eb, ea)) = (tgt, args) := ⟨_, _, rfl⟩
        -- the chosen edge: its target, typed, its arguments read
        have hch : (tgt = tb ∧ args = ta) ∨ (tgt = eb ∧ args = ea) := by
          split at hsel <;> simp_all
        obtain ⟨bdg, hbdg, hlen, hty, -⟩ : EdgeTyped F tgt args := by
          rcases hch with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
          · exact ⟨bdT, hT, hlenT, htyT, hwT⟩
          · exact ⟨bdE, hE, hlenE, htyE, hwE⟩
        have hDa : ∀ v ∈ args, D' v.id := fun v hv => h4.1 v (by
          rcases hch with ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> simp [Inst.uses, hv])
        simp only [Blocks.runInsts, hvc, hsel] at h1
        cases hm : args.mapM (Blocks.getV s'.vals) with
        | none => simp [hm] at h1
        | some xs =>
        simp only [hm, Sem.Outcome.ok.injEq, Prod.mk.injEq] at h1
        obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h1
        simp only at h
        have hbg : bdg ∈ F.blocks := List.mem_of_find?_eq_some hbdg
        have hbid : bdg.ref.id = tgt.id := by have := List.find?_some hbdg; simpa using this
        -- the branch's other side, found by its label
        have hlEls : c.labels.get? (.els (start bd.ref.id + k)) =
            some (pos bd.ref.id + 1 + K + 1 + (edgeLines F (start bd.ref.id + k) 0 tb bdT ta).length) := by
          apply hG.lay.lbl
          have hl := lower_brif F (start bd.ref.id + k) cnd tb eb ta ea bdT bdE hwc htc hlc hT hE
          rw [hl] at hatI
          simp only [List.append_assoc] at hatI
          have := hatI.right.right 0 (by simp)
          simpa [Nat.add_assoc] using this
        obtain ⟨K', regs', hS', hgo⟩ := brif_go c F D' s'.vals ts' (start bd.ref.id + k) (pos bd.ref.id + 1 + K)
          cnd tb eb ta ea bdT bdE bc h5 hvc hDc htc hlc hwc hT hE hG.params hwT hwE hatI hlEls tgt args xs bdg
          (pos bdg.ref.id) hsel hm hbdg hlen hty hDa (hbid ▸ lbl_blk hG.lay bdg hbg)
          (target_mag F hG.params bd hb _ hmem tb (by simp [jumpsTo]) bdT hT)
          (target_mag F hG.params bd hb _ hmem eb (by simp [jumpsTo]) bdE hE)
        obtain ⟨it, hit, hitD⟩ := h4'.1 tgt (by rcases hch with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> simp [Inst.targets])
        obtain ⟨N', ts'', hW', hgo'⟩ := run_sim F lc c start0 pos start inOf hG f hfb hinv steps { s' with world := s'.world }
          tgt.id xs r w h bdg (fun m => D' m ∨ ∃ p ∈ bdg.params, p.1.id = m) { ts' with regs := regs' } it
          (hfb ▸ hbdg) hit (fun n hn => Or.inl (hitD n (by simpa using hn))) (fun p hp => Or.inr ⟨p, hp, rfl⟩)
          hS' h6 h6n
        refine ⟨K + K' + 1 + N', ts'', hW', fun fuel => ?_⟩
        rw [show fuel + (K + K' + 1 + N') = fuel + N' + 1 + K' + K by omega, h7, hgo,
          go_label c _ _ _ _ (hG.lay.blk bdg hbg).1, hbid]
        exact hgo' fuel
      all_goals simp [isTerm] at h2

/-! ## An instruction that defines a value -/

/-- **Defining a value keeps the relation**: the registers after agree with
    those before on every other value's registers, and the new value's
    registers hold it as the relation says. -/
theorem sim_set (F : Facts) (D : Nat → Prop) (vs : Blocks.Vals) (regs regs' : Std.HashMap PtxIr.Reg UInt64)
    (i : Inst) (d : Val) (t : ClifTy) (r : UInt64) (hd : Inst.dst? i = some d) (hS : Sim F D vs regs)
    (hty : F.tys.get? d.id = some t) (hn : r &&& widthMask t = r)
    (hplain : F.lits.get? d.id = none → ¬ Win F d → regs'.get? (regOf t d) = some (enc t r))
    (hlit : ∀ l, F.lits.get? d.id = some l → IsLit l ∧ litRaw l = enc t r)
    (hwin : Win F d → ∃ off : UInt64, regs'.get? (winReg d) = some off ∧ off.toNat < 2 ^ 32 ∧
      WinOff r off)
    (hneg : t = .i8 → F.negated d = true → regs'.get? (negReg d) = some (if r &&& 0xff = 0 then 1 else 0))
    (hflag : t = .i8 → r ≤ 1)
    (hkeep : ∀ c n, n ≠ d.id → regs'.get? (.val c n) = regs.get? (.val c n))
    (hmo : magOk F d i = true)
    (hrule : ∀ k, magRule F.origin F.mag F.lits F.tys i = some k → (Rel F d r).toNat < 2 ^ k) :
    Sim F (extD D i) (Blocks.setV vs d (.sc t r)) regs' := by
  have hD : ∀ v : Val, extD D i v.id → v.id ≠ d.id → D v.id := by
    intro v h hne
    rcases h with h | h
    · exact h
    · simp [hd] at h; exact absurd h.symm hne
  have hvd : ∀ v : Val, v.id = d.id → v = d := by intro v h; cases v; cases d; simp_all
  have hold : ∀ v : Val, D v.id → v.id ≠ d.id → Blocks.getV (Blocks.setV vs d (.sc t r)) v = Blocks.getV vs v := by
    intro v hv hne
    obtain ⟨t', b', hb', -⟩ := hS.pres v hv
    exact getV_setV_ne vs d v _ hne (by rw [hb']; rfl)
  exact {
    pres := fun v h => by
      by_cases he : v.id = d.id
      · rw [hvd v he, getV_setV_self]; exact ⟨t, r, rfl, hty⟩
      · rw [hold v (hD v h he) he]; exact hS.pres v (hD v h he)
    normd := fun v t' b h hv => by
      by_cases he : v.id = d.id
      · rw [hvd v he, getV_setV_self] at hv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hv); exact hn
      · rw [hold v (hD v h he) he] at hv; exact hS.normd v t' b (hD v h he) hv
    flag := fun v b h hv => by
      by_cases he : v.id = d.id
      · rw [hvd v he, getV_setV_self] at hv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hv); exact hflag rfl
      · rw [hold v (hD v h he) he] at hv; exact hS.flag v b (hD v h he) hv
    plain := fun v t' b h hv ht hl hw => by
      by_cases he : v.id = d.id
      · have := hvd v he; subst this
        rw [getV_setV_self] at hv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hv); exact hplain hl hw
      · rw [hold v (hD v h he) he] at hv
        rw [regOf, hkeep _ _ he]; exact hS.plain v t' b (hD v h he) hv ht hl hw
    lit := fun v t' b l h hv ht hl => by
      by_cases he : v.id = d.id
      · have := hvd v he; subst this
        rw [getV_setV_self] at hv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hv); exact hlit l hl
      · rw [hold v (hD v h he) he] at hv; exact hS.lit v t' b l (hD v h he) hv ht hl
    win := fun v t' b h hv hw => by
      by_cases he : v.id = d.id
      · have := hvd v he; subst this
        rw [getV_setV_self] at hv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hv); exact hwin hw
      · rw [hold v (hD v h he) he] at hv
        obtain ⟨off, h1, h2, h3⟩ := hS.win v t' b (hD v h he) hv hw
        exact ⟨off, by rw [winReg, hkeep _ _ he]; exact h1, h2, h3⟩
    neg := fun v b h hv hng => by
      by_cases he : v.id = d.id
      · have := hvd v he; subst this
        rw [getV_setV_self] at hv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hv); exact hneg rfl hng
      · rw [hold v (hD v h he) he] at hv
        rw [negReg, hkeep _ _ he]; exact hS.neg v b (hD v h he) hv hng
    bound := fun v t' b h hv => by
      by_cases he : v.id = d.id
      · have := hvd v he; subst this
        rw [getV_setV_self] at hv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hv)
        exact bounded_of F v i r hmo hrule
      · rw [hold v (hD v h he) he] at hv; exact hS.bound v t' b (hD v h he) hv }

/-- The relation reads only values' registers: registers agreeing on those
    keep it. -/
theorem Sim.frame {F : Facts} {D : Nat → Prop} {vs : Blocks.Vals} {regs regs' : Std.HashMap PtxIr.Reg UInt64}
    (h : Sim F D vs regs) (hk : ∀ c n, regs'.get? (.val c n) = regs.get? (.val c n)) : Sim F D vs regs' :=
  { pres := h.pres, normd := h.normd, flag := h.flag, lit := h.lit, bound := h.bound
    plain := fun v t b h0 h1 h2 h3 h4 => by rw [regOf, hk]; exact h.plain v t b h0 h1 h2 h3 h4
    win := fun v t b h0 h1 h2 => by
      obtain ⟨off, h3, h4, h5⟩ := h.win v t b h0 h1 h2
      exact ⟨off, by rw [winReg, hk]; exact h3, h4, h5⟩
    neg := fun v b h0 h1 h2 => by rw [negReg, hk]; exact h.neg v b h0 h1 h2 }

/-! ## Operands as values -/

/-- The value an operand reads, the thread's state left as it was. -/
def OpVal (c : PtxRun.Ctx) (ts : PtxRun.TS) (o : PtxIr.Opnd) (x : UInt64) : Prop :=
  (PtxRun.opnd c o).run ts = .ok (x, ts)

theorem opVal_reg (c : PtxRun.Ctx) (ts : PtxRun.TS) (r : PtxIr.Reg) (x : UInt64) (hs : ∀ s, r ≠ .special s)
    (h : ts.regs.get? r = some x) : OpVal c ts (.reg r) x := by
  simp only [Std.HashMap.get?_eq_getElem?] at h
  simp [OpVal, PtxRun.opnd, getReg_plain _ _ hs, h]

theorem opVal_lit (c : PtxRun.Ctx) (ts : PtxRun.TS) (l : PtxIr.Opnd) (hl : IsLit l) : OpVal c ts l (litRaw l) := by
  cases l <;> simp_all [IsLit, OpVal, PtxRun.opnd, litRaw] <;> rfl

/-- **An immediate operand**: the lines `imm v` writes for a value in scope,
    not a windowed shared address, leave an operand reading a value whose cut
    to the register's width is the value's encoding, changing no register but
    the instruction's own. -/
theorem imm_sound (F : Facts) (tag : Nat) (s : Out) (vs : Blocks.Vals) (c : PtxRun.Ctx) (ts : PtxRun.TS)
    (v : Val) (t : ClifTy) (b : UInt64)
    (D : Nat → Prop) (hS : Sim F D vs ts.regs) (hD : D v.id) (hv : Blocks.getV vs v = some (.sc t b))
    (ht : F.tys.get? v.id = some t) (hw : ¬ Win F v) (hsc : Scalar t) :
    ∃ L : List PtxIr.Instr,
      (runLM (imm v) F tag s).2.lines = s.lines ++ (L.map (.ins none)).toArray ∧
      s.n ≤ (runLM (imm v) F tag s).2.n ∧
      ∃ regs' x, (steps c L).run ts = .ok ((), { ts with regs := regs' }) ∧
        -- the operand reads `x` from any registers agreeing on all but the fresh
        (∀ ts2 : PtxRun.TS, (∀ R, ¬ Fresh tag (runLM (imm v) F tag s).2.n R → ts2.regs.get? R = regs'.get? R) →
          OpVal c ts2 (runLM (imm v) F tag s).1 x) ∧
        PtxRun.mask (bits t).width x = enc t b ∧
        ∀ R, ¬ Fresh tag s.n R → regs'.get? R = ts.regs.get? R := by
  cases hl : F.lits.get? v.id with
  | some l =>
      obtain ⟨hil, hraw⟩ := hS.lit v t b l hD hv ht hl
      have hr : runLM (imm v) F tag s = (l, s) := by
        simp only [Std.HashMap.get?_eq_getElem?] at hl
        simp [imm, facts?, hl]
      rw [hr]
      refine ⟨[], by simp, Nat.le_refl _, ts.regs, litRaw l, ?_, fun ts2 _ => opVal_lit c ts2 l hil, ?_,
        fun _ _ => rfl⟩
      · simp [steps, StateT.run, pure, StateT.pure, Except.pure]
      · rw [hraw]; exact mask_enc t b
  | none =>
      have hr : runLM (imm v) F tag s = (.reg (runLM (r v) F tag s).1, (runLM (r v) F tag s).2) := by
        simp only [Std.HashMap.get?_eq_getElem?] at hl
        simp [imm, facts?, hl, o]
      obtain ⟨L, hL, hn, hnf, hns, regs', hrun, hreg, hkeep⟩ := r_sound F tag s vs c ts v t b D hS hD hv ht hw
      rw [hr]
      refine ⟨L, hL, hn, regs', enc t b, hrun, fun ts2 h2 => opVal_reg c _ _ _ hns (by rw [h2 _ hnf]; exact hreg),
        mask_enc t b, hkeep⟩

/-! ## Instructions -/

/-- What the values check says of one instruction defining a value. -/
theorem defsOk_inst (F : Facts) (h : defsOk F = true) (b : BlockData) (hb : b ∈ F.blocks) (i : Inst)
    (hi : i ∈ b.insts) (d : Val) (hd : Inst.dst? i = some d) :
    F.tys.get? d.id = (resTy (F.tys.get? ·.id) i).map (·.2) ∧ F.negated d = negOf F i ∧
    F.lits.get? d.id = litOf F i ∧ defSide F d i = true ∧ flagOf F i = true ∧ magOk F d i = true := by
  simp only [defsOk, List.all_eq_true] at h
  have := h b hb i hi
  rw [hd] at this
  simp only [Bool.and_eq_true, beq_iff_eq] at this
  obtain ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩ := this
  exact ⟨h1, h2, h3, h4, h5, h6⟩

/-- Cutting to an integer type's width, as PTX and the block semantics do it. -/
theorem mask_width (t : ClifTy) (h : t = .i16 ∨ t = .i32 ∨ t = .i64) (x : UInt64) :
    PtxRun.mask t.width x = x &&& widthMask t := by
  rcases h with rfl | rfl | rfl <;> simp [PtxRun.mask, ClifTy.width, widthMask] <;> try rfl
  exact (and_wide64 x).symm

/-- A sum cut to a width depends on its operands only as cut to it. -/
theorem add_cong (t : ClifTy) (h : t = .i16 ∨ t = .i32 ∨ t = .i64) (xa xb y : UInt64)
    (hy : y &&& widthMask t = xb &&& widthMask t) :
    ((xa &&& widthMask t) + y) &&& widthMask t = (xa + xb) &&& widthMask t := by
  have hw : t.width ≤ 64 := by rcases h with rfl | rfl | rfl <;> decide
  rw [widthMask_int t (by rcases h with h | h | h <;> simp [h])] at hy ⊢
  apply UInt64.toNat_inj.mp
  have hy' := congrArg UInt64.toNat hy
  rw [toNat_and_mask _ _ hw, toNat_and_mask _ _ hw] at hy'
  rw [toNat_and_mask _ _ hw, toNat_and_mask _ _ hw, UInt64.toNat_add, UInt64.toNat_add, toNat_and_mask _ _ hw]
  have h64 : 2 ^ t.width ∣ 2 ^ 64 := Nat.pow_dvd_pow 2 hw
  rw [Nat.mod_mod_of_dvd _ h64, Nat.mod_mod_of_dvd _ h64, Nat.add_mod, Nat.mod_mod, hy', ← Nat.add_mod]

/-- The congruence every integer operation here has: on operands cut to the
    width, the result cut to it is as on the operands. -/
def Cong (op : UInt64 → UInt64 → UInt64) (t : ClifTy) : Prop :=
  ∀ xa xb y : UInt64, y &&& widthMask t = xb &&& widthMask t →
    (op (xa &&& widthMask t) y) &&& widthMask t = (op xa xb) &&& widthMask t

theorem toNat_wm (t : ClifTy) (h : t = .i16 ∨ t = .i32 ∨ t = .i64) (x : UInt64) :
    (x &&& widthMask t).toNat = x.toNat % 2 ^ t.width := by
  rw [widthMask_int t (by rcases h with h | h | h <;> simp [h])]
  exact toNat_and_mask _ _ (by rcases h with rfl | rfl | rfl <;> decide)

theorem cong_add (t : ClifTy) (h : t = .i16 ∨ t = .i32 ∨ t = .i64) : Cong (· + ·) t :=
  fun xa xb y hy => add_cong t h xa xb y hy

theorem cong_sub (t : ClifTy) (h : t = .i16 ∨ t = .i32 ∨ t = .i64) : Cong (· - ·) t := by
  intro xa xb y hy
  dsimp only
  apply UInt64.toNat_inj.mp
  have hy' := congrArg UInt64.toNat hy
  rw [toNat_wm t h, toNat_wm t h] at hy' ⊢
  rw [UInt64.toNat_sub, UInt64.toNat_sub, toNat_wm t h]
  have := y.toNat_lt; have := xb.toNat_lt; have := xa.toNat_lt
  rcases h with rfl | rfl | rfl <;> simp only [ClifTy.width] at hy' ⊢ <;> omega

theorem cong_mul (t : ClifTy) (h : t = .i16 ∨ t = .i32 ∨ t = .i64) : Cong (· * ·) t := by
  intro xa xb y hy
  dsimp only
  apply UInt64.toNat_inj.mp
  have hy' := congrArg UInt64.toNat hy
  rw [toNat_wm t h, toNat_wm t h] at hy' ⊢
  rw [UInt64.toNat_mul, UInt64.toNat_mul, toNat_wm t h]
  have hw : t.width ≤ 64 := by rcases h with rfl | rfl | rfl <;> decide
  have h64 : 2 ^ t.width ∣ 2 ^ 64 := Nat.pow_dvd_pow 2 hw
  rw [Nat.mod_mod_of_dvd _ h64, Nat.mod_mod_of_dvd _ h64, Nat.mul_mod, Nat.mod_mod, hy', ← Nat.mul_mod]

theorem cong_and (t : ClifTy) (_h : t = .i16 ∨ t = .i32 ∨ t = .i64) : Cong (· &&& ·) t := by
  intro xa xb y hy
  dsimp only
  have e1 : xa &&& widthMask t &&& y &&& widthMask t = xa &&& (y &&& widthMask t) := by
    rw [show xa &&& widthMask t &&& y &&& widthMask t = xa &&& (y &&& (widthMask t &&& widthMask t)) by ac_rfl,
      UInt64.and_self]
  rw [e1, hy]; ac_rfl

theorem cong_or (t : ClifTy) (h : t = .i16 ∨ t = .i32 ∨ t = .i64) : Cong (· ||| ·) t := by
  intro xa xb y hy
  dsimp only
  apply UInt64.toNat_inj.mp
  have hy' := congrArg UInt64.toNat hy
  rw [toNat_wm t h, toNat_wm t h] at hy' ⊢
  rw [UInt64.toNat_or, UInt64.toNat_or, Nat.or_mod_two_pow, Nat.or_mod_two_pow, toNat_wm t h,
    Nat.mod_mod, hy']

theorem cong_xor (t : ClifTy) (h : t = .i16 ∨ t = .i32 ∨ t = .i64) : Cong (· ^^^ ·) t := by
  intro xa xb y hy
  dsimp only
  apply UInt64.toNat_inj.mp
  have hy' := congrArg UInt64.toNat hy
  rw [toNat_wm t h, toNat_wm t h] at hy' ⊢
  rw [UInt64.toNat_xor, UInt64.toNat_xor, Nat.xor_mod_two_pow, Nat.xor_mod_two_pow, toNat_wm t h,
    Nat.mod_mod, hy']

open PtxRun in
/-- An integer two-operand instruction on operands reading `x` and `y` writes
    what `bin` gives. -/
theorem exec_bin {rnd : Option PtxIr.Rnd} (c : PtxRun.Ctx) (ts : PtxRun.TS) (o : PtxIr.BinOp) (ty : PtxIr.Ty)
    (dr : PtxIr.Reg) (xo yo : PtxIr.Opnd) (x y v : UInt64) (hx : OpVal c ts xo x) (hy : OpVal c ts yo y)
    (hv : PtxRun.bin o ty x y = .ok v) :
    (exec c (.bin o rnd ty dr xo yo)).run ts = .ok (.step, { ts with regs := ts.regs.insert dr v }) := by
  simp only [OpVal] at hx hy
  simp only [exec, StateT.run, bind, StateT.bind] at hx hy ⊢
  simp [hx, hy, Except.bind, lift, hv, setReg, pure, StateT.pure, Except.pure, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet]

theorem instrsOf_map (L : List PtxIr.Instr) : instrsOf ((L.map (PtxIr.Line.ins none)).toArray) = L := by
  induction L with
  | nil => rfl
  | cons i L ih => simp only [instrsOf, List.map_cons, List.toList_toArray] at ih ⊢; simp [ih]

theorem straight_map (L : List PtxIr.Instr) : Straight ((L.map (PtxIr.Line.ins none)).toArray) := by
  simp [Straight, instrsOf_map]

/-- An integer's encoding is its bits. -/
theorem enc_int (t : ClifTy) (ht : t = .i16 ∨ t = .i32 ∨ t = .i64) (x : UInt64) (hx : x &&& widthMask t = x) :
    enc t x = x := by
  have hne8 : t ≠ .i8 := by rcases ht with rfl | rfl | rfl <;> decide
  simp only [enc, hne8, if_false, hx]

/-- **A sum's bound**: a shared address's offset plus the other operand, or
    the operands' numbers summed. -/
theorem iadd_rule (F : Facts) (d a b : Val) (t : ClifTy) (xa xb : UInt64) (k : Nat)
    (htd : F.tys.get? d.id = some t) (hba : Bounded F a xa) (hbb : Bounded F b xb)
    (hk : magRule F.origin F.mag F.lits F.tys (.iadd d a b) = some k) :
    (Rel F d ((xa + xb) &&& widthMask t)).toNat < 2 ^ k := by
  simp only [magRule] at hk
  by_cases hs : F.origin.get? d.id = some .shared
  · rw [if_pos (by rw [hs]; rfl), htd] at hk
    by_cases h64 : t = .i64
    · subst h64
      rw [if_neg (by simp)] at hk
      unfold Rel; rw [if_pos hs, and_wide64]
      by_cases ha : F.origin.get? a.id = some .shared
      · rw [if_pos (by rw [ha]; rfl)] at hk; cases hk
        rw [show xa + xb - regionBase .arena = (xa - regionBase .arena) + xb by
          apply UInt64.toBitVec_inj.mp; simp; bv_omega]
        exact add_lt_bits _ _ _ _ (bnd_off hba ha) (bnd_val hbb)
      · rw [if_neg (by simp only [beq_iff_eq]; exact ha)] at hk
        by_cases hb : F.origin.get? b.id = some .shared
        · rw [if_pos (by rw [hb]; rfl)] at hk; cases hk
          rw [show xa + xb - regionBase .arena = (xb - regionBase .arena) + xa by
            apply UInt64.toBitVec_inj.mp; simp; bv_omega]
          exact add_lt_bits _ _ _ _ (bnd_off hbb hb) (bnd_val hba)
        · rw [if_neg (by simp only [beq_iff_eq]; exact hb)] at hk; cases hk
    · rw [if_pos (by simp [h64])] at hk; cases hk
  · rw [if_neg (by simp only [beq_iff_eq]; exact hs)] at hk; cases hk
    rw [rel_plain hs]
    exact lt_min64 (Nat.lt_of_le_of_lt (and_le_toNat _ _) (add_lt_bits _ _ _ _ (bnd_val hba) (bnd_val hbb)))

/-- **A shift by a constant's bound**: its operand's number that many bits
    more, within the type's width. -/
theorem ishl_rule (F : Facts) (d a b : Val) (t : ClifTy) (ht : t = .i16 ∨ t = .i32 ∨ t = .i64)
    (hta : F.tys.get? a.id = some t) (xa yb : UInt64) (k : Nat)
    (hmo : magOk F d (.ishl d a b) = true) (hba : Bounded F a xa)
    (hlb : ∀ l, F.lits.get? b.id = some l → litRaw l = yb)
    (hk : magRule F.origin F.mag F.lits F.tys (.ishl d a b) = some k) :
    (Rel F d ((xa <<< (yb % UInt64.ofNat t.width)) &&& widthMask t)).toNat < 2 ^ k := by
  rw [rel_plain (not_shared_of hmo (fun _ _ _ h => by cases h)), toNat_wm t ht]
  simp only [magRule, hta, Option.map_some, Option.getD_some] at hk
  split at hk
  · rename_i k' hl
    split at hk
    · rename_i hc
      cases hk
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      obtain ⟨h0, hlt⟩ := hc
      have hy : yb = UInt64.ofInt k' := (hlb _ hl).symm
      have hw64 : t.width ≤ 64 := by rcases ht with rfl | rfl | rfl <;> decide
      have hyn : yb.toNat = k'.toNat := by
        subst hy; simp only [UInt64.ofInt, UInt64.toNat_ofNat']; omega
      have hmod : (yb % UInt64.ofNat t.width).toNat = k'.toNat := by
        rw [UInt64.toNat_mod, hyn, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega : t.width < 2 ^ 64)]
        exact Nat.mod_eq_of_lt (by omega)
      rw [UInt64.toNat_shiftLeft, hmod, Nat.shiftLeft_eq, Nat.mod_eq_of_lt (a := k'.toNat) (by omega)]
      have hva := bnd_val hba
      by_cases hle : valBits F.origin F.mag a + k'.toNat ≤ t.width
      · rw [Nat.min_eq_right hle]
        have h1 : xa.toNat * 2 ^ k'.toNat < 2 ^ (valBits F.origin F.mag a + k'.toNat) := by
          rw [Nat.pow_add]; exact Nat.mul_lt_mul_of_pos_right hva (Nat.two_pow_pos _)
        have h2 : 2 ^ (valBits F.origin F.mag a + k'.toNat) ≤ 2 ^ 64 :=
          Nat.pow_le_pow_right (by decide) (by omega)
        rw [Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le h1 h2)]
        exact Nat.lt_of_le_of_lt (Nat.mod_le _ _) h1
      · rw [Nat.min_eq_left (by omega)]
        exact Nat.mod_lt _ (Nat.two_pow_pos _)
    · cases hk
  · cases hk

/-- **An and's bound**: no more bits than its narrower operand's number. -/
theorem band_rule (F : Facts) (d a b : Val) (t : ClifTy) (xa xb : UInt64) (k : Nat)
    (hmo : magOk F d (.band d a b) = true) (hba : Bounded F a xa) (hbb : Bounded F b xb)
    (hk : magRule F.origin F.mag F.lits F.tys (.band d a b) = some k) :
    (Rel F d ((xa &&& xb) &&& widthMask t)).toNat < 2 ^ k := by
  rw [rel_plain (not_shared_of hmo (fun _ _ _ h => by cases h))]
  simp only [magRule] at hk
  split at hk
  · cases hk
  · cases hk
    have h1 := and_le_toNat (xa &&& xb) (widthMask t)
    have h2 : (xa &&& xb).toNat ≤ xa.toNat := by rw [UInt64.toNat_and]; exact Nat.and_le_left
    have h3 : (xa &&& xb).toNat ≤ xb.toNat := by rw [UInt64.toNat_and]; exact Nat.and_le_right
    have ha := bnd_val hba
    have hb := bnd_val hbb
    rcases Nat.le_total (valBits F.origin F.mag a) (valBits F.origin F.mag b) with h | h
    · rw [Nat.min_eq_left h]; omega
    · rw [Nat.min_eq_right h]; omega

open PtxRun in
/-- **A two-operand instruction** meets its obligation, given what the block
    semantics computes (`hstep`), what its lowering writes (its operands'
    lines, then one PTX instruction, `hlow`), and that the PTX instruction on
    the operands' encodings gives the result's (`hsem`). -/
theorem bin2_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (i : Inst) (d a b : Val)
    (bo : PtxIr.BinOp) (rnd : Option PtxIr.Rnd) (tyf : ClifTy → PtxIr.Ty) (f : ClifTy → UInt64 → UInt64 → UInt64)
    (P : ClifTy → Prop) (hP : ∀ t, P t → Scalar t) (Q : ClifTy → UInt64 → UInt64 → Prop)
    (hd : Inst.dst? i = some d) (hua : a ∈ Inst.uses i) (hub : b ∈ Inst.uses i)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → P t ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hstep : ∀ (s s1 : Blocks.BSt), step1 lc s i = some s1 → s.world.mem.nan = c.hw.nan →
      ∀ t xa, P t → Blocks.getV s.vals a = some (.sc t xa) →
      ∃ xb, Blocks.getV s.vals b = some (.sc t xb) ∧ Q t xa xb ∧
        s1 = { s with vals := Blocks.setV s.vals d (.sc t (f t xa xb)) })
    (hlow : ∀ tag t, F.tys.get? a.id = some t → (lower F tag i).lines =
      ((runLM (imm b) F tag (runLM (r a) F tag {}).2).2.lines).push
        (.ins none (.bin bo rnd (tyf t) (regOf t d) (.reg (runLM (r a) F tag {}).1)
          (runLM (imm b) F tag (runLM (r a) F tag {}).2).1)))
    (hng : ∀ t, P t → t = .i8 → F.negated d = false)
    (hflag : ∀ t xa xb, P t → t = .i8 → xa ≤ 1 → xb ≤ 1 → f t xa xb ≤ 1)
    (hsem : ∀ t xa xb y, P t → Q t xa xb → xa &&& widthMask t = xa → xb &&& widthMask t = xb →
      (t = .i8 → xa ≤ 1 ∧ xb ≤ 1) → PtxRun.mask (bits t).width y = enc t xb →
      ∀ (c : PtxRun.Ctx) (ts : PtxRun.TS) (xo yo : PtxIr.Opnd) (dr : PtxIr.Reg), OpVal c ts xo (enc t xa) →
        OpVal c ts yo y →
        (exec c (.bin bo rnd (tyf t) dr xo yo)).run ts = .ok (.step, { ts with regs := ts.regs.insert dr (enc t (f t xa xb)) }))
    (hnorm : ∀ t xa xb, P t → xa &&& widthMask t = xa → xb &&& widthMask t = xb →
      f t xa xb &&& widthMask t = f t xa xb)
    (hmo : magOk F d i = true) (hrule : ∀ t xa xb k, P t → Q t xa xb → F.tys.get? d.id = some t → Bounded F a xa → Bounded F b xb →
      magRule F.origin F.mag F.lits F.tys i = some k → (Rel F d (f t xa xb)).toNat < 2 ^ k) :
    InstOk F lc c start0 i := by
  intro tag D s s1 ts hst huses hS hW herr hnan
  obtain ⟨ta, xa, hva, hta⟩ := hS.pres a (huses a hua)
  obtain ⟨hPt, htd⟩ := hT ta hta
  have hsc := hP ta hPt
  have hsn : s.world.mem.nan = c.hw.nan := (congrArg (fun x : World => x.mem.nan) hW.1).trans hnan.1
  obtain ⟨xb, hvb, hQ, rfl⟩ := hstep s s1 hst hsn ta xa hPt hva
  have hfl : ta = .i8 → xa ≤ 1 ∧ xb ≤ 1 := fun h8 => by
    subst h8; exact ⟨hS.flag a xa (huses a hua) hva, hS.flag b xb (huses b hub) hvb⟩
  have htb : F.tys.get? b.id = some ta := by
    obtain ⟨tb, xb', hvb', htb⟩ := hS.pres b (huses b hub)
    rw [hvb] at hvb'; obtain ⟨rfl, -⟩ := V.sc.inj (Option.some.inj hvb'); exact htb
  let sa := (runLM (r a) F tag {}).2
  obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag {} s.vals c ts a ta xa D hS (huses a hua) hva hta hwa
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lb, hLb, hn2, regs2, yv, hrun2, hop2, hy, hkeep2⟩ :=
    imm_sound F tag sa s.vals c { ts with regs := regs1 } b ta xb D hS1 (huses b hub) hvb htb hwb hsc
  let core : PtxIr.Instr := .bin bo rnd (tyf ta) (regOf ta d) (.reg (runLM (r a) F tag {}).1)
    (runLM (imm b) F tag sa).1
  have hlist : (lower F tag i).lines = ((La ++ Lb ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow tag ta hta, hLb]
    show (sa.lines ++ _).push _ = _
    rw [hLa]; simp [core]; rfl
  have hins : instrsOf (lower F tag i).lines = La ++ Lb ++ [core] := by rw [hlist, instrsOf_map]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hxa : OpVal c { ts with regs := regs2 } (.reg (runLM (r a) F tag {}).1) (enc ta xa) :=
    opVal_reg c _ _ _ hns1 (by rw [hkeep2 _ hnf1]; exact hreg1)
  have hyb : OpVal c { ts with regs := regs2 } (runLM (imm b) F tag sa).1 yv := hop2 _ (fun _ _ => rfl)
  have hx : (exec c core).run { ts with regs := regs2 } =
      .ok (.step, { ts with regs := regs2.insert (regOf ta d) (enc ta (f ta xa xb)) }) :=
    hsem ta xa xb yv hPt hQ (hS.normd a ta xa (huses a hua) hva) (hS.normd b ta xb (huses b hub) hvb) hfl hy
      c _ _ _ _ hxa hyb
  refine ⟨{ ts with regs := regs2.insert (regOf ta d) (enc ta (f ta xa xb)) }, ?_, ?_, hW⟩
  · rw [hins, steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]
    simp only [StateT.run] at hx
    simp [steps, StateT.run, hx, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun k hk => hrule ta xa xb k hPt hQ htd (hS.bound a ta xa (huses a hua) hva) (hS.bound b ta xb (huses b hub) hvb) hk) F D s.vals ts.regs _ i d ta (f ta xa xb) hd hS htd
      (hnorm ta xa xb hPt (hS.normd a ta xa (huses a hua) hva) (hS.normd b ta xb (huses b hub) hvb))
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8 hn; rw [hng ta hPt h8] at hn; cases hn
    · intro h8; exact hflag ta xa xb hPt h8 (hfl h8).1 (hfl h8).2
    · intro c' n hn
      have hne : PtxIr.Reg.val c' n ≠ regOf ta d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h), hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]

open PtxRun in
/-- **An integer sum of values not windowed** meets its obligation. -/
theorem iadd_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hint : ∀ t, F.tys.get? a.id = some t → t = .i16 ∨ t = .i32 ∨ t = .i64)
    (hty : ∀ t, F.tys.get? a.id = some t → F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.iadd d a b) = true) :
    InstOk F lc c start0 (.iadd d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply bin2_ok (hmo := hmo) (hrule := fun t xa xb k _ _ htd hba hbb hk => iadd_rule F d a b t xa xb k htd hba hbb hk) F lc c start0 (.iadd d a b) d a b .add none (fun t => uTy t.width)
    (fun t x y => (x + y) &&& widthMask t) (fun t => t = .i16 ∨ t = .i32 ∨ t = .i64)
    (fun t h => by rcases h with rfl | rfl | rfl <;> simp [Scalar])
    (fun _ _ _ => True) rfl (by simp [Inst.uses]) (by simp [Inst.uses]) hwa hwb hwd
  · intro t ht
    exact ⟨hint t ht, hty t ht⟩
  · exact hld
  · intro s s1 hst _ t xa _ hva
    obtain ⟨tb, xb, hvb, -⟩ : ∃ tb xb, Blocks.getV s.vals b = some (.sc tb xb) ∧ True := by
      simp only [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, Blocks.viaOp, evalOp, Sem.bin, Sem.get] at hst
      cases hb : Blocks.getV s.vals b with
      | none => simp [hb] at hst
      | some vb =>
          cases vb with
          | sc tb xb => exact ⟨tb, xb, rfl, trivial⟩
          | vec _ _ => simp [hb] at hst
    have hsame : tb = t := by
      simp only [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hvb, Blocks.viaOp, evalOp, Sem.bin,
        Sem.get] at hst
      by_cases hne : tb = t
      · exact hne
      · simp [Ne.symm hne] at hst
    subst hsame
    have hint' : tb.isInt = true := by
      simp only [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hvb, Blocks.viaOp, evalOp, Sem.bin,
        Sem.get] at hst
      by_cases h : tb.isInt = true
      · exact h
      · simp [h] at hst
    have hev := evalInst_iadd s.world.mem s.vals d a b tb xa xb hint' hva hvb
    simp only [step1, hev, Option.map_some, Option.some.injEq] at hst
    exact ⟨xb, hvb, trivial, hst.symm⟩
  · intro tag t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', Bool.or_false, if_false, tyOf, ht, o, emit, emitL, runLM_modify]
  · intro t ht h8; rcases ht with rfl | rfl | rfl <;> cases h8
  · intro t _ _ ht h8; rcases ht with rfl | rfl | rfl <;> cases h8
  · intro t xa xb y hti _ _ _ _ hy c' ts' xo yo dr hxo hyo
    have hne8 : t ≠ .i8 := by rcases hti with rfl | rfl | rfl <;> decide
    have hbin : PtxRun.bin .add (uTy t.width) (enc t xa) y = .ok (PtxRun.mask t.width (enc t xa + y)) := by
      rcases hti with rfl | rfl | rfl <;> rfl
    rw [exec_bin c' ts' .add (uTy t.width) dr xo yo _ _ _ hxo hyo hbin]
    have hmb : PtxRun.mask (bits t).width y = y &&& widthMask t := by
      rw [show (bits t).width = t.width by rcases hti with rfl | rfl | rfl <;> rfl]; exact mask_width t hti y
    rw [hmb] at hy
    simp only [enc, hne8, if_false] at hy ⊢
    rw [mask_width t hti, UInt64.and_assoc, UInt64.and_self, add_cong t hti xa xb y hy]
  · intro t xa xb _ _ _; rw [UInt64.and_assoc, UInt64.and_self]

/-- An integer type a register holds as itself, not a flag. -/
def IntT (t : ClifTy) : Prop := t = .i16 ∨ t = .i32 ∨ t = .i64

open PtxRun in
/-- **An integer two-operand instruction** meets its obligation, given what
    the block semantics computes, its lowering, that PTX's operation is the
    integer one cut to the width, and that the operation commutes with the cut. -/
theorem intbin_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (i : Inst) (d a b : Val)
    (bo : PtxIr.BinOp) (tyf : ClifTy → PtxIr.Ty) (op : UInt64 → UInt64 → UInt64)
    (hd : Inst.dst? i = some d) (hua : a ∈ Inst.uses i) (hub : b ∈ Inst.uses i)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hstep : ∀ (s s1 : Blocks.BSt), step1 lc s i = some s1 → ∀ t xa, Blocks.getV s.vals a = some (.sc t xa) →
      ∃ xb, Blocks.getV s.vals b = some (.sc t xb) ∧
        s1 = { s with vals := Blocks.setV s.vals d (.sc t (op xa xb &&& widthMask t)) })
    (hlow : ∀ tag t, F.tys.get? a.id = some t → (lower F tag i).lines =
      ((runLM (imm b) F tag (runLM (r a) F tag {}).2).2.lines).push
        (.ins none (.bin bo none (tyf t) (regOf t d) (.reg (runLM (r a) F tag {}).1)
          (runLM (imm b) F tag (runLM (r a) F tag {}).2).1)))
    (hbin : ∀ t x y, IntT t → PtxRun.bin bo (tyf t) x y = .ok (PtxRun.mask t.width (op x y)))
    (hcong : ∀ t, IntT t → Cong op t)
    (hmo : magOk F d i = true)
    (hrule : ∀ t xa xb k, IntT t → F.tys.get? d.id = some t → Bounded F a xa → Bounded F b xb →
      magRule F.origin F.mag F.lits F.tys i = some k → (Rel F d (op xa xb &&& widthMask t)).toNat < 2 ^ k) :
    InstOk F lc c start0 i := by
  apply bin2_ok (hmo := hmo) (hrule := fun t xa xb k hP _ htd hba hbb hk => hrule t xa xb k hP htd hba hbb hk) F lc c start0 i d a b bo none tyf (fun t x y => op x y &&& widthMask t) IntT
    (fun t h => by rcases h with rfl | rfl | rfl <;> simp [Scalar])
    (fun _ _ _ => True) hd hua hub hwa hwb hwd hT hld
    (fun s s1 h _ t xa _ hv => let ⟨xb, h1, h2⟩ := hstep s s1 h t xa hv; ⟨xb, h1, trivial, h2⟩) hlow
  · intro t ht h8; rcases ht with rfl | rfl | rfl <;> cases h8
  · intro t _ _ ht h8; rcases ht with rfl | rfl | rfl <;> cases h8
  · intro t xa xb y hti _ _ _ _ hy c' ts' xo yo dr hxo hyo
    have hne8 : t ≠ .i8 := by rcases hti with rfl | rfl | rfl <;> decide
    rw [exec_bin c' ts' bo (tyf t) dr xo yo _ _ _ hxo hyo (hbin t _ _ hti)]
    have hmb : PtxRun.mask (bits t).width y = y &&& widthMask t := by
      rw [show (bits t).width = t.width by rcases hti with rfl | rfl | rfl <;> rfl]; exact mask_width t hti y
    rw [hmb] at hy
    simp only [enc, hne8, if_false] at hy ⊢
    rw [mask_width t hti, UInt64.and_assoc, UInt64.and_self, hcong t hti xa xb y hy]
  · intro t xa xb _ _ _; rw [UInt64.and_assoc, UInt64.and_self]

open PtxRun in
/-- **An integer difference of values not windowed** meets its obligation. -/
theorem isub_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.isub d a b) = true) :
    InstOk F lc c start0 (.isub d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply intbin_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 (.isub d a b) d a b .sub (fun t => uTy t.width) (· - ·) rfl
    (by simp [Inst.uses]) (by simp [Inst.uses]) hwa hwb hwd hT hld
  · intro s s1 hst t xa hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin,
        Sem.get, zipIntBits] at hst
    | sc tb xb =>
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin, Sem.get,
      zipIntBits] at hst
    by_cases hc : t = tb ∧ t.isInt = true
    · obtain ⟨rfl, hci⟩ := hc
      simp [hci] at hst
      subst hst
      exact ⟨xb, rfl, by simp [norm]⟩
    · simp [hc] at hst
  · intro tag t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify]
    all_goals rfl
  · intro t x y ht; rcases ht with rfl | rfl | rfl <;> rfl
  · exact cong_sub

/-- What an integer product computes: its operands at one type, their product
    cut to it. -/
theorem step_imul (lc : Sem.Locals) (s s1 : Blocks.BSt) (d a b : Val) (t : ClifTy) (xa : UInt64)
    (hst : step1 lc s (.imul d a b) = some s1) (hva : Blocks.getV s.vals a = some (.sc t xa)) :
    ∃ xb, Blocks.getV s.vals b = some (.sc t xb) ∧
      s1 = { s with vals := Blocks.setV s.vals d (.sc t (xa * xb &&& widthMask t)) } := by
  cases hb : Blocks.getV s.vals b with
  | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
  | some vb =>
  cases vb with
  | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin,
      Sem.get] at hst
  | sc tb xb =>
  simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin, Sem.get] at hst
  by_cases hc : t = tb ∧ t.isInt = true
  · obtain ⟨rfl, hci⟩ := hc
    simp [hci] at hst
    subst hst
    exact ⟨xb, rfl, by simp [norm]⟩
  · simp [hc] at hst

open PtxRun in
/-- **An integer product of values not windowed**, not a wide one
    (`Facts.wideMul`), meets its obligation. -/
theorem imul_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none) (hnw : F.wideMul a b = false)
    (hmo : magOk F d (.imul d a b) = true) :
    InstOk F lc c start0 (.imul d a b) := by
  apply intbin_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 (.imul d a b) d a b .mulLo (fun t => uTy t.width) (· * ·) rfl
    (by simp [Inst.uses]) (by simp [Inst.uses]) hwa hwb hwd hT hld
  · intro s s1 hst t xa hva
    exact step_imul lc s s1 d a b t xa hst hva
  · intro tag t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, facts?, runLM_bind, runLM_read,
      runLM_pure, hnw, tyOf, ht, o, emit, emitL, runLM_modify]
    all_goals rfl
  · intro t x y ht; rcases ht with rfl | rfl | rfl <;> rfl
  · exact cong_mul

theorem mask1 (z : UInt64) : (PtxRun.mask 1 z).toNat = z.toNat % 2 ^ 1 := by
  simp only [PtxRun.mask, show ¬ (1 ≥ 64) from by decide, if_false]
  rw [show ((1 : UInt64) <<< UInt64.ofNat 1 - 1) = ((2 ^ 1 - 1 : Nat).toUInt64) from rfl,
    toNat_and_mask _ _ (by decide)]

/-- A bitwise operation on flags: on a flag and any bits, its low bit depends
    only on the low bit of the bits, and on two flags it gives a flag. -/
def FlagOp (op : UInt64 → UInt64 → UInt64) : Prop :=
  (∀ x y : UInt64, x ≤ 1 → PtxRun.mask 1 (op x y) = op x (PtxRun.mask 1 y)) ∧
  ∀ x y : UInt64, x ≤ 1 → y ≤ 1 → op x y ≤ 1

theorem flagOp_of (op : UInt64 → UInt64 → UInt64) (nop : Nat → Nat → Nat)
    (htn : ∀ x y : UInt64, (op x y).toNat = nop x.toNat y.toNat)
    (hmod : ∀ a b, nop a b % 2 ^ 1 = nop (a % 2 ^ 1) (b % 2 ^ 1))
    (hb : ∀ x y : UInt64, x ≤ 1 → y ≤ 1 → op x y ≤ 1) : FlagOp op := by
  refine ⟨fun x y hx => ?_, hb⟩
  have hx' : x.toNat % 2 ^ 1 = x.toNat := Nat.mod_eq_of_lt (by have := UInt64.le_iff_toNat_le.mp hx; simp at this ⊢; omega)
  apply UInt64.toNat_inj.mp
  rw [mask1, htn, htn, mask1, hmod, hx']

theorem flagOp_and : FlagOp (· &&& ·) :=
  flagOp_of _ (· &&& ·) (fun x y => UInt64.toNat_and x y) (fun a b => Nat.and_mod_two_pow)
    (fun x y hx hy => by rcases flag_cases x hx with rfl | rfl <;> rcases flag_cases y hy with rfl | rfl <;> decide)

theorem flagOp_or : FlagOp (· ||| ·) :=
  flagOp_of _ (· ||| ·) (fun x y => UInt64.toNat_or x y) (fun a b => Nat.or_mod_two_pow)
    (fun x y hx hy => by rcases flag_cases x hx with rfl | rfl <;> rcases flag_cases y hy with rfl | rfl <;> decide)

theorem flagOp_xor : FlagOp (· ^^^ ·) :=
  flagOp_of _ (· ^^^ ·) (fun x y => UInt64.toNat_xor x y) (fun a b => Nat.xor_mod_two_pow)
    (fun x y hx hy => by rcases flag_cases x hx with rfl | rfl <;> rcases flag_cases y hy with rfl | rfl <;> decide)

theorem flag_byte (z : UInt64) (h : z ≤ 1) : z &&& widthMask .i8 = z := by
  rcases flag_cases z h with rfl | rfl <;> decide

theorem enc_flag (z : UInt64) (h : z ≤ 1) : enc .i8 z = z := by
  rcases flag_cases z h with rfl | rfl <;> decide

open PtxRun in
/-- **A bitwise operation on flags** meets its obligation: the same operation
    on their predicates. -/
theorem flagbin_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (i : Inst) (d a b : Val)
    (bo : PtxIr.BinOp) (tyf : ClifTy → PtxIr.Ty) (op : UInt64 → UInt64 → UInt64)
    (hd : Inst.dst? i = some d) (hua : a ∈ Inst.uses i) (hub : b ∈ Inst.uses i)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → t = .i8 ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none) (hng : F.negated d = false)
    (hstep : ∀ (s s1 : Blocks.BSt), step1 lc s i = some s1 → ∀ t xa, Blocks.getV s.vals a = some (.sc t xa) →
      ∃ xb, Blocks.getV s.vals b = some (.sc t xb) ∧
        s1 = { s with vals := Blocks.setV s.vals d (.sc t (op xa xb &&& widthMask t)) })
    (hlow : ∀ tag t, F.tys.get? a.id = some t → (lower F tag i).lines =
      ((runLM (imm b) F tag (runLM (r a) F tag {}).2).2.lines).push
        (.ins none (.bin bo none (tyf t) (regOf t d) (.reg (runLM (r a) F tag {}).1)
          (runLM (imm b) F tag (runLM (r a) F tag {}).2).1)))
    (hbin : ∀ x y, PtxRun.bin bo (tyf .i8) x y = .ok (PtxRun.mask 1 (op x y)))
    (hop : FlagOp op)
    (hmo : magOk F d i = true) (hnr : magRule F.origin F.mag F.lits F.tys i = none) :
    InstOk F lc c start0 i := by
  apply bin2_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ _ hk => by rw [hnr] at hk; cases hk) F lc c start0 i d a b bo none tyf (fun t x y => op x y &&& widthMask t) (· = .i8)
    (fun t h => by subst h; simp [Scalar]) (fun _ _ _ => True) hd hua hub hwa hwb hwd hT hld
    (fun s s1 h _ t xa _ hv => let ⟨xb, h1, h2⟩ := hstep s s1 h t xa hv; ⟨xb, h1, trivial, h2⟩) hlow
  · intro _ _ _; exact hng
  · intro t xa xb h8 _ hxa hxb; subst h8; dsimp only; rw [flag_byte _ (hop.2 xa xb hxa hxb)]; exact hop.2 xa xb hxa hxb
  · intro t xa xb y h8 _ _ _ hfl hy c' ts' xo yo dr hxo hyo
    subst h8
    obtain ⟨hxa, hxb⟩ := hfl rfl
    rw [enc_flag xa hxa] at hxo
    rw [enc_flag xb hxb] at hy
    have hy1 : PtxRun.mask 1 y = xb := hy
    dsimp only
    rw [exec_bin c' ts' bo (tyf .i8) dr xo yo _ _ _ hxo hyo (hbin xa y), hop.1 xa y hxa,
      hy1, flag_byte _ (hop.2 xa xb hxa hxb), enc_flag _ (hop.2 xa xb hxa hxb)]
  · intro t xa xb _ _ _; rw [UInt64.and_assoc, UInt64.and_self]

open PtxRun in
/-- A bitwise operation, on integers or flags, meets its obligation. -/
theorem logic_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (i : Inst) (d a b : Val)
    (bo : PtxIr.BinOp) (op : UInt64 → UInt64 → UInt64)
    (hd : Inst.dst? i = some d) (hua : a ∈ Inst.uses i) (hub : b ∈ Inst.uses i)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → (IntT t ∨ t = .i8) ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none) (hng : F.negated d = false)
    (hstep : ∀ (s s1 : Blocks.BSt), step1 lc s i = some s1 → ∀ t xa, Blocks.getV s.vals a = some (.sc t xa) →
      ∃ xb, Blocks.getV s.vals b = some (.sc t xb) ∧
        s1 = { s with vals := Blocks.setV s.vals d (.sc t (op xa xb &&& widthMask t)) })
    (hlow : ∀ tag t, F.tys.get? a.id = some t → (lower F tag i).lines =
      ((runLM (imm b) F tag (runLM (r a) F tag {}).2).2.lines).push
        (.ins none (.bin bo none (if t == .i8 then .pred else bTy t.width) (regOf t d) (.reg (runLM (r a) F tag {}).1)
          (runLM (imm b) F tag (runLM (r a) F tag {}).2).1)))
    (hbin : ∀ (ty : PtxIr.Ty), ty.isFloat = false → ∀ x y, PtxRun.bin bo ty x y = .ok (PtxRun.mask ty.width (op x y)))
    (hcong : ∀ t, IntT t → Cong op t) (hop : FlagOp op)
    (hmo : magOk F d i = true)
    (hrule : ∀ t xa xb k, IntT t → F.tys.get? d.id = some t → Bounded F a xa → Bounded F b xb →
      magRule F.origin F.mag F.lits F.tys i = some k → (Rel F d (op xa xb &&& widthMask t)).toNat < 2 ^ k)
    (hr8 : F.tys.get? a.id = some .i8 → magRule F.origin F.mag F.lits F.tys i = none) :
    InstOk F lc c start0 i := by
  by_cases h8 : F.tys.get? a.id = some .i8
  · apply flagbin_ok (hmo := hmo) (hnr := hr8 h8) F lc c start0 i d a b bo (fun t => if t == .i8 then .pred else bTy t.width) op hd hua hub
      hwa hwb hwd (fun t ht => by rw [h8] at ht; cases ht; exact ⟨rfl, (hT _ h8).2⟩) hld hng hstep hlow
      (fun x y => hbin .pred rfl x y) hop
  · apply intbin_ok (hmo := hmo) (hrule := hrule) F lc c start0 i d a b bo (fun t => if t == .i8 then .pred else bTy t.width) op hd hua hub
      hwa hwb hwd (fun t ht => ⟨((hT t ht).1.resolve_right (fun h => h8 (h ▸ ht))), (hT t ht).2⟩) hld hstep hlow
    · intro t x y ht
      have h : (if t == .i8 then PtxIr.Ty.pred else bTy t.width) = bTy t.width := by
        rcases ht with rfl | rfl | rfl <;> rfl
      dsimp only
      rw [h, hbin _ (by rcases ht with rfl | rfl | rfl <;> rfl)]
      rcases ht with rfl | rfl | rfl <;> rfl
    · exact hcong

open PtxRun in
/-- **A bitwise and** of integers or flags, not windowed, meets its obligation. -/
theorem band_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → (IntT t ∨ t = .i8) ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none) (hng : F.negated d = false)
    (hmo : magOk F d (.band d a b) = true) :
    InstOk F lc c start0 (.band d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply logic_ok (hmo := hmo) (hrule := fun t xa xb k _ _ hba hbb hk => band_rule F d a b t xa xb k hmo hba hbb hk)
    (hr8 := fun h8 => by simp only [magRule]; rw [if_pos (by rw [h8]; rfl)]) F lc c start0 (.band d a b) d a b .and (· &&& ·) rfl (by simp [Inst.uses]) (by simp [Inst.uses])
    hwa hwb hwd hT hld hng
  · intro s s1 hst t xa hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin,
        Sem.get, zipIntBits] at hst
    | sc tb xb =>
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin, Sem.get,
      zipIntBits] at hst
    by_cases hc : t = tb ∧ (t.isInt = true ∨ t.isVec = true)
    · obtain ⟨rfl, hci⟩ := hc
      simp [hci] at hst
      subst hst
      exact ⟨xb, rfl, by simp [norm]⟩
    · simp [hc] at hst
  · intro tag t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify]
    all_goals rfl
  · intro ty hf x y; simp [PtxRun.bin, hf, PtxRun.binI]
  · exact cong_and
  · exact flagOp_and

open PtxRun in
/-- **A bitwise or** of integers or flags, not windowed, meets its obligation. -/
theorem bor_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → (IntT t ∨ t = .i8) ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none) (hng : F.negated d = false)
    (hmo : magOk F d (.bor d a b) = true) :
    InstOk F lc c start0 (.bor d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply logic_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ hk => by simp [magRule] at hk) (hr8 := fun _ => rfl)
    F lc c start0 (.bor d a b) d a b .or (· ||| ·) rfl (by simp [Inst.uses]) (by simp [Inst.uses])
    hwa hwb hwd hT hld hng
  · intro s s1 hst t xa hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin,
        Sem.get, zipIntBits] at hst
    | sc tb xb =>
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin, Sem.get,
      zipIntBits] at hst
    by_cases hc : t = tb ∧ (t.isInt = true ∨ t.isVec = true)
    · obtain ⟨rfl, hci⟩ := hc
      simp [hci] at hst
      subst hst
      exact ⟨xb, rfl, by simp [norm]⟩
    · simp [hc] at hst
  · intro tag t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify]
    all_goals rfl
  · intro ty hf x y; simp [PtxRun.bin, hf, PtxRun.binI]
  · exact cong_or
  · exact flagOp_or

open PtxRun in
/-- **A bitwise exclusive or** of integers or flags, not windowed, meets its obligation. -/
theorem bxor_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → (IntT t ∨ t = .i8) ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none) (hng : F.negated d = false)
    (hmo : magOk F d (.bxor d a b) = true) :
    InstOk F lc c start0 (.bxor d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply logic_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ hk => by simp [magRule] at hk) (hr8 := fun _ => rfl)
    F lc c start0 (.bxor d a b) d a b .xor (· ^^^ ·) rfl (by simp [Inst.uses]) (by simp [Inst.uses])
    hwa hwb hwd hT hld hng
  · intro s s1 hst t xa hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin,
        Sem.get, zipIntBits] at hst
    | sc tb xb =>
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin, Sem.get,
      zipIntBits] at hst
    by_cases hc : t = tb ∧ (t.isInt = true ∨ t.isVec = true)
    · obtain ⟨rfl, hci⟩ := hc
      simp [hci] at hst
      subst hst
      exact ⟨xb, rfl, by simp [norm]⟩
    · simp [hc] at hst
  · intro tag t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify]
    all_goals rfl
  · intro ty hf x y; simp [PtxRun.bin, hf, PtxRun.binI]
  · exact cong_xor
  · exact flagOp_xor

/-! ## Constants -/

open PtxRun in
/-- **An instruction the lowering writes nothing for** meets its obligation:
    the value it defines is a constant, whose literal, written where it is
    used, is its encoding (`hstep`). -/
theorem lit_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (i : Inst) (d : Val)
    (hd : Inst.dst? i = some d) (hlow : ∀ tag, (lower F tag i).lines = #[])
    (hwd : F.win d = false) (hng : F.negated d = false)
    (hstep : ∀ (D : Nat → Prop) (s s1 : Blocks.BSt) (regs : Std.HashMap PtxIr.Reg UInt64),
      Sim F D s.vals regs → (∀ v ∈ Inst.uses i, D v.id) → step1 lc s i = some s1 →
      ∃ t r l, s1 = { s with vals := Blocks.setV s.vals d (.sc t r) } ∧ F.tys.get? d.id = some t ∧
        r &&& widthMask t = r ∧ F.lits.get? d.id = some l ∧ IsLit l ∧ litRaw l = enc t r ∧ t ≠ .i8 ∧
        ∀ k, magRule F.origin F.mag F.lits F.tys i = some k → (Rel F d r).toNat < 2 ^ k)
    (hmo : magOk F d i = true) :
    InstOk F lc c start0 i := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  obtain ⟨t, r, l, rfl, htd, hn, hl, hil, hraw, h8, hrl⟩ := hstep D s s1 ts.regs hS huses hst
  have h0 : (lower F tag i).lines = (([] : List PtxIr.Instr).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow]; rfl
  refine ⟨by rw [h0]; exact straight_map _, ts, ?_, ?_, hW⟩
  · rw [h0, instrsOf_map]; simp [steps, StateT.run, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := hrl) F D s.vals ts.regs ts.regs i d t r hd hS htd hn
    · intro h; rw [hl] at h; cases h
    · intro l' hl'; rw [hl] at hl'; cases hl'; exact ⟨hil, hraw⟩
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro _ h; rw [hng] at h; cases h
    · intro h; exact absurd h h8
    · intro _ _ _; rfl

/-- An integer constant's literal is its encoding. -/
theorem iconst_raw (t : ClifTy) (ht : IntT t) (k : Int) :
    litRaw (.imm ((k % (2 ^ t.width : Int)).toNat)) =
      enc t (UInt64.ofNat (k.emod (1 <<< 64)).toNat &&& widthMask t) := by
  have hne8 : t ≠ .i8 := by rcases ht with rfl | rfl | rfl <;> decide
  simp only [litRaw, enc, hne8, if_false, UInt64.and_assoc, UInt64.and_self]
  apply UInt64.toNat_inj.mp
  rw [toNat_wm t ht]
  simp only [UInt64.ofInt, UInt64.toNat_ofNat']
  rw [show (1 <<< 64 : Nat) = 2 ^ 64 from rfl, show k.emod ((2 ^ 64 : Nat) : Int) = k % ((2 ^ 64 : Nat) : Int) from rfl]
  rcases ht with rfl | rfl | rfl <;> simp only [ClifTy.width, Nat.reducePow] <;> omega

open PtxRun in
/-- **An integer constant** other than a flag meets its obligation. -/
theorem iconst_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d : Val) (ty : ClifTy) (k : Int)
    (hty : IntT ty) (htd : F.tys.get? d.id = some ty)
    (hl : F.lits.get? d.id = some (.imm ((k % (2 ^ ty.width : Int)).toNat)))
    (hwd : F.win d = false) (hng : F.negated d = false)
    (hmo : magOk F d (.iconst d ty k) = true) :
    InstOk F lc c start0 (.iconst d ty k) := by
  have hne8 : (ty == .i8) = false := by rcases hty with rfl | rfl | rfl <;> rfl
  apply lit_ok (hmo := hmo) F lc c start0 _ d rfl (fun tag => by simp [lower_eq, isTerm, inst, hne8]) hwd hng
  intro D s s1 regs _ _ hst
  simp [step1, Blocks.evalInst, Blocks.viaOp, evalOp, ofInt, norm] at hst
  refine ⟨ty, _, _, hst.symm, htd, by rw [UInt64.and_assoc, UInt64.and_self], hl, trivial, iconst_raw ty hty k,
    by rcases hty with rfl | rfl | rfl <;> decide, fun k' hk' => ?_⟩
  simp only [magRule] at hk'; cases hk'
  rw [rel_plain (not_shared_of hmo (fun _ _ _ h => by cases h))]
  have := iconst_raw ty hty k
  rw [enc_int ty hty _ (by rw [UInt64.and_assoc, UInt64.and_self]),
    show ((1 <<< 64 : Nat) : Int) = 18446744073709551616 from rfl] at this
  rw [← this]
  simp only [litRaw, UInt64.ofInt, UInt64.toNat_ofNat']
  have hw : (2 ^ ty.width : Int) ≤ 2 ^ 64 := by rcases hty with rfl | rfl | rfl <;> decide
  have hp : (0 : Int) < 2 ^ ty.width := by rcases hty with rfl | rfl | rfl <;> decide
  have h0 : 0 ≤ k % (2 ^ ty.width : Int) := Int.emod_nonneg _ (by omega)
  have h1 : k % (2 ^ ty.width : Int) < 2 ^ ty.width := Int.emod_lt_of_pos _ hp
  have h2 : k % (2 ^ ty.width : Int) < 2 ^ 64 := Int.lt_of_lt_of_le h1 hw
  have hN : ((((k % (2 ^ ty.width : Int)).toNat : Int) % 2 ^ 64).toNat % 2 ^ 64) = (k % (2 ^ ty.width : Int)).toNat := by
    generalize k % (2 ^ ty.width : Int) = N at h0 h2 ⊢
    omega
  rw [hN]
  exact Nat.lt_log2_self

open PtxRun in
/-- **A float constant** meets its obligation. -/
theorem fconst_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d : Val) (ty : ClifTy) (b : UInt64)
    (hty : ty = .f32 ∨ ty = .f64) (htd : F.tys.get? d.id = some ty) (hl : F.lits.get? d.id = some (lit ty b))
    (hwd : F.win d = false) (hng : F.negated d = false)
    (hmo : magOk F d (.fconst d ty b) = true) :
    InstOk F lc c start0 (.fconst d ty b) := by
  apply lit_ok (hmo := hmo) F lc c start0 _ d rfl (fun tag => by simp [lower_eq, isTerm, inst]) hwd hng
  intro D s s1 regs _ _ hst
  simp [step1, Blocks.evalInst, Blocks.viaOp, evalOp] at hst
  have hne8 : ty ≠ .i8 := by rcases hty with rfl | rfl <;> decide
  refine ⟨ty, _, _, hst.symm, htd, by rw [UInt64.and_assoc, UInt64.and_self], hl, ?_, ?_, hne8,
    fun _ hk => by simp [magRule] at hk⟩
  · simp only [lit]; split <;> trivial
  · simp only [enc, hne8, if_false, UInt64.and_assoc, UInt64.and_self]
    rcases hty with rfl | rfl
    · apply UInt64.toNat_inj.mp
      rw [show widthMask .f32 = ((2 ^ 32 - 1 : Nat).toUInt64) from rfl, toNat_and_mask _ 32 (by decide)]
      simp [lit, litRaw]; omega
    · simp [lit, litRaw, show widthMask .f64 = widthMask .i64 from rfl, and_wide64]

open PtxRun in
/-- **A widened 32-bit constant** meets its obligation: its literal is the
    constant's, already zero above 32 bits. -/
theorem uextend_lit_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val) (l : PtxIr.Opnd)
    (hta : F.tys.get? a.id = some .i32) (htd : F.tys.get? d.id = some .i64)
    (hla : F.lits.get? a.id = some l) (hl : F.lits.get? d.id = some l)
    (hwd : F.win d = false) (hng : F.negated d = false)
    (hmo : magOk F d (.uextend64 d a) = true) :
    InstOk F lc c start0 (.uextend64 d a) := by
  have hm : d.id ∈ F.lits := by
    rw [Std.HashMap.mem_iff_isSome_getElem?, ← Std.HashMap.get?_eq_getElem?, hl]; rfl
  apply lit_ok (hmo := hmo) F lc c start0 _ d rfl (fun tag => by
    have hta' : F.tys[a.id]? = some .i32 := by rw [← Std.HashMap.get?_eq_getElem?]; exact hta
    simp [lower_eq, isTerm, inst, facts?, tyOf, hm, hta']) hwd hng
  intro D s s1 regs hS huses hst
  have hDa : D a.id := huses a (by simp [Inst.uses])
  obtain ⟨t, x, hva, hta'⟩ := hS.pres a hDa
  rw [hta] at hta'; cases hta'
  obtain ⟨hil, hraw⟩ := hS.lit a .i32 x l hDa hva hta hla
  simp only [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.un, Sem.get] at hst
  simp [ClifTy.isInt, ClifTy.width] at hst
  refine ⟨.i64, _, l, hst.symm, htd, and_wide64 _, hl, hil, ?_, by decide, fun k hk => ?_⟩
  · rw [hraw]; simp [enc, and_wide64]
  · simp only [magRule] at hk; cases hk
    rw [rel_plain (not_shared_of hmo (fun _ _ _ h => by cases h))]
    exact lt_min32 (by decide) (bnd_val (hS.bound a .i32 x hDa hva))

/-! ## One-operand instructions -/

open PtxRun in
/-- **A one-operand instruction** meets its obligation, given what the block
    semantics computes (`hstep`), what its lowering writes (its operand's
    lines, then one PTX instruction, `hlow`), and that the PTX instruction on
    the operand's encoding gives the result's (`hsem`). -/
theorem un1_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (i : Inst) (d a : Val)
    (mk : ClifTy → PtxIr.Reg → PtxIr.Opnd → PtxIr.Instr) (td : ClifTy → ClifTy) (f : ClifTy → UInt64 → UInt64)
    (P : ClifTy → Prop) (htd8 : ∀ t, P t → td t ≠ .i8)
    (hd : Inst.dst? i = some d) (hua : a ∈ Inst.uses i) (hwa : ¬ Win F a) (hwd : F.win d = false)
    (htd : ∀ t, F.tys.get? a.id = some t → F.tys.get? d.id = some (td t))
    (hld : F.lits.get? d.id = none)
    (hstep : ∀ (s s1 : Blocks.BSt), step1 lc s i = some s1 → s.world.mem.nan = c.hw.nan →
      ∀ t xa, Blocks.getV s.vals a = some (.sc t xa) →
      F.tys.get? a.id = some t → P t ∧ s1 = { s with vals := Blocks.setV s.vals d (.sc (td t) (f t xa)) })
    (hlow : ∀ tag t, F.tys.get? a.id = some t → (lower F tag i).lines =
      ((runLM (r a) F tag {}).2.lines).push (.ins none (mk t (regOf (td t) d) (.reg (runLM (r a) F tag {}).1))))
    (hsem : ∀ t xa, P t → xa &&& widthMask t = xa → (t = .i8 → xa ≤ 1) →
      ∀ (c : PtxRun.Ctx) (ts : PtxRun.TS) (xo : PtxIr.Opnd) (dr : PtxIr.Reg), OpVal c ts xo (enc t xa) →
        (exec c (mk t dr xo)).run ts = .ok (.step, { ts with regs := ts.regs.insert dr (enc (td t) (f t xa)) }))
    (hnorm : ∀ t xa, P t → xa &&& widthMask t = xa → f t xa &&& widthMask (td t) = f t xa)
    (hmo : magOk F d i = true) (hrule : ∀ t xa k, P t → Bounded F a xa →
      magRule F.origin F.mag F.lits F.tys i = some k → (Rel F d (f t xa)).toNat < 2 ^ k) :
    InstOk F lc c start0 i := by
  intro tag D s s1 ts hst huses hS hW herr hnan
  have hsn : s.world.mem.nan = c.hw.nan := (congrArg (fun x : World => x.mem.nan) hW.1).trans hnan.1
  obtain ⟨ta, xa, hva, hta⟩ := hS.pres a (huses a hua)
  obtain ⟨hPt, rfl⟩ := hstep s s1 hst hsn ta xa hva hta
  obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag {} s.vals c ts a ta xa D hS (huses a hua) hva hta hwa
  let core : PtxIr.Instr := mk ta (regOf (td ta) d) (.reg (runLM (r a) F tag {}).1)
  have hlist : (lower F tag i).lines = ((La ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow tag ta hta, hLa]; simp [core]
  have hins : instrsOf (lower F tag i).lines = La ++ [core] := by rw [hlist, instrsOf_map]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hxa : OpVal c { ts with regs := regs1 } (.reg (runLM (r a) F tag {}).1) (enc ta xa) :=
    opVal_reg c _ _ _ hns1 hreg1
  have hx : (exec c core).run { ts with regs := regs1 } =
      .ok (.step, { ts with regs := regs1.insert (regOf (td ta) d) (enc (td ta) (f ta xa)) }) :=
    hsem ta xa hPt (hS.normd a ta xa (huses a hua) hva)
      (fun h8 => by subst h8; exact hS.flag a xa (huses a hua) hva) c { ts with regs := regs1 } _ (regOf (td ta) d) hxa
  refine ⟨{ ts with regs := regs1.insert (regOf (td ta) d) (enc (td ta) (f ta xa)) }, ?_, ?_, hW⟩
  · rw [hins, steps_append c _ _ ts _ hrun1]
    simp only [StateT.run] at hx
    simp [steps, StateT.run, hx, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun k hk => hrule ta xa k hPt (hS.bound a ta xa (huses a hua) hva) hk) F D s.vals ts.regs _ i d (td ta) (f ta xa) hd hS (htd ta hta) (hnorm ta xa hPt (hS.normd a ta xa (huses a hua) hva))
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; exact absurd h8 (htd8 ta hPt)
    · intro h8; exact absurd h8 (htd8 ta hPt)
    · intro c' n hn
      have hne : PtxIr.Reg.val c' n ≠ regOf (td ta) d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]

open PtxRun in
theorem exec_cvt (c : PtxRun.Ctx) (ts : PtxRun.TS) (rn : Option PtxIr.Rnd) (dt st : PtxIr.Ty) (dr : PtxIr.Reg)
    (xo : PtxIr.Opnd) (x v : UInt64) (hx : OpVal c ts xo x) (hv : PtxRun.cvt rn dt st x = .ok v) :
    (exec c (.cvt rn dt st dr xo)).run ts = .ok (.step, { ts with regs := ts.regs.insert dr v }) := by
  simp only [OpVal] at hx
  simp only [exec, StateT.run, bind, StateT.bind] at hx ⊢
  simp [hx, Except.bind, lift, hv, setReg, pure, StateT.pure, Except.pure, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet]

open PtxRun in
theorem exec_selp (c : PtxRun.Ctx) (ts : PtxRun.TS) (ty : PtxIr.Ty) (dr : PtxIr.Reg)
    (ao bo po : PtxIr.Opnd) (x y p : UInt64) (ha : OpVal c ts ao x) (hb : OpVal c ts bo y) (hp : OpVal c ts po p) :
    (exec c (.selp ty dr ao bo po)).run ts =
      .ok (.step, { ts with regs := ts.regs.insert dr (PtxRun.mask ty.width (if p &&& 1 == 1 then x else y)) }) := by
  simp only [OpVal] at ha hb hp
  simp only [exec, StateT.run, bind, StateT.bind] at ha hb hp ⊢
  simp [ha, hb, hp, Except.bind, setReg, pure, StateT.pure, Except.pure, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet]

/-- An integer conversion: the source read as an integer, cut to the
    destination's width. -/
theorem cvt_int (rn : Option PtxIr.Rnd) (dt st : PtxIr.Ty) (x : UInt64) (hd : dt.isFloat = false)
    (hs : st.isFloat = false) : PtxRun.cvt rn dt st x = .ok (PtxRun.ofIntW dt.width (PtxRun.toInt st x)) := by
  simp [PtxRun.cvt, hd, hs]

theorem mask64 (y : UInt64) : PtxRun.mask 64 y = y := by simp [PtxRun.mask]

open PtxRun in
/-- **A narrowing to 32 bits** meets its obligation. -/
theorem ireduce_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false) (htd : F.tys.get? d.id = some .i32)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.ireduce32 d a) = true) :
    InstOk F lc c start0 (.ireduce32 d a) := by
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a (fun _ dr xo => .cvt none .u32 .u64 dr xo) (fun _ => .i32)
    (fun _ x => x &&& widthMask .i32) (fun t => t = .i64) (fun _ _ h => nomatch h) rfl (by simp [Inst.uses]) hwa hwd
    (fun _ _ => htd) hld
  · intro s s1 hst _ t xa hva _
    simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.un, Sem.get, norm] at hst
    by_cases hc : t.isInt = true ∧ 32 < t.width
    · simp [hc] at hst
      refine ⟨?_, hst.symm⟩
      cases t <;> simp_all [ClifTy.isInt, ClifTy.width]
    · simp [hc] at hst
  · intro tag t ht
    simp [lower_eq, isTerm, inst, emit, emitL, o]
  · intro t xa ht _ _ c' ts' xo dr hxo
    subst ht
    rw [exec_cvt c' ts' none .u32 .u64 dr xo _ _ hxo (cvt_int none .u32 .u64 _ rfl rfl)]
    simp only [enc, show ClifTy.i64 ≠ .i8 by decide, show ClifTy.i32 ≠ .i8 by decide, if_false, and_wide64]
    simp only [PtxRun.toInt, PtxRun.ofIntW, PtxIr.Ty.isSigned, PtxIr.Ty.width, Bool.false_and,
      Bool.false_eq_true, if_false, mask64, ofInt_toNat]
    rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp), UInt64.and_assoc, UInt64.and_self]
  · intro t xa _ _; rw [UInt64.and_assoc, UInt64.and_self]

open PtxRun in
/-- **A zero-extension to 64 bits** of a value not a constant meets its
    obligation: a flag's by choosing 1 or 0, any other's by `cvt`. -/
theorem uextend_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false) (htd : F.tys.get? d.id = some .i64)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.uextend64 d a) = true) :
    InstOk F lc c start0 (.uextend64 d a) := by
  have hld' : d.id ∉ F.lits := by
    rw [Std.HashMap.mem_iff_isSome_getElem?, ← Std.HashMap.get?_eq_getElem?, hld]; simp
  apply un1_ok (hmo := hmo) (hrule := fun t xa k hP hba hk => by
      simp only [magRule] at hk; cases hk
      rw [rel_plain (not_shared_of hmo (fun _ _ _ h => by cases h))]
      exact lt_min32 (by rcases hP with rfl | rfl | rfl <;> decide) (bnd_val hba)) F lc c start0 _ d a
    (fun t dr xo => if t = .i8 then .selp .u64 dr (.imm 1) (.imm 0) xo else .cvt none .u64 (uTy t.width) dr xo)
    (fun _ => .i64) (fun t x => x &&& widthMask t) (fun t => t = .i8 ∨ t = .i16 ∨ t = .i32) (fun _ _ h => nomatch h) rfl
    (by simp [Inst.uses]) hwa hwd (fun _ _ => htd) hld
  · intro s s1 hst _ t xa hva _
    simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.un, Sem.get] at hst
    by_cases hc : t.isInt = true ∧ t.width < 64
    · simp [hc] at hst
      refine ⟨?_, hst.symm⟩
      cases t <;> simp_all [ClifTy.isInt, ClifTy.width]
    · simp [hc] at hst
  · intro tag t ht
    have ht' : F.tys[a.id]? = some t := by rw [← Std.HashMap.get?_eq_getElem?]; exact ht
    by_cases h8 : t = .i8
    · subst h8; simp [lower_eq, isTerm, inst, emit, emitL, o, facts?, tyOf, ht', hld']
    · simp [lower_eq, isTerm, inst, emit, emitL, o, facts?, tyOf, ht', hld', h8]
  · intro t xa ht hn hfl c' ts' xo dr hxo
    simp only [enc, show ClifTy.i64 ≠ .i8 by decide, if_false, and_wide64]
    rcases ht with rfl | rfl | rfl
    · have hx1 := hfl rfl
      simp only [if_true]
      have ho1 : OpVal c' ts' (.imm 1) 1 := opVal_lit c' ts' (.imm 1) trivial
      have ho0 : OpVal c' ts' (.imm 0) 0 := opVal_lit c' ts' (.imm 0) trivial
      rw [exec_selp c' ts' .u64 dr _ _ xo 1 0 _ ho1 ho0 hxo]
      rcases flag_cases xa hx1 with rfl | rfl <;> rfl
    · simp only [show ¬ (ClifTy.i16 = .i8) by decide, if_false]
      rw [exec_cvt c' ts' none .u64 _ dr xo _ _ hxo (cvt_int none .u64 _ _ rfl rfl)]
      simp only [enc, show ClifTy.i16 ≠ .i8 by decide, if_false, hn, PtxRun.toInt, PtxRun.ofIntW,
        PtxIr.Ty.isSigned, PtxIr.Ty.width, uTy, ClifTy.width, Bool.false_and, Bool.false_eq_true, if_false, mask64]
      rw [show (16 : Nat) = ClifTy.i16.width from rfl, mask_width .i16 (by simp), hn, ofInt_toNat]
    · simp only [show ¬ (ClifTy.i32 = .i8) by decide, if_false]
      rw [exec_cvt c' ts' none .u64 _ dr xo _ _ hxo (cvt_int none .u64 _ _ rfl rfl)]
      simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hn, PtxRun.toInt, PtxRun.ofIntW,
        PtxIr.Ty.isSigned, PtxIr.Ty.width, uTy, ClifTy.width, Bool.false_and, Bool.false_eq_true, if_false, mask64]
      rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp), hn, ofInt_toNat]
  · intro t xa _ hn; dsimp only; rw [hn, and_wide64]


/-- Reading a 16- or 32-bit integer as signed, as both semantics do it. -/
theorem toInt_signed (t : ClifTy) (ht : t = .i16 ∨ t = .i32) (x : UInt64) :
    PtxRun.toInt (sTy t.width) x = signed t x := by
  have hm : PtxRun.mask t.width x = x &&& widthMask t :=
    mask_width t (by rcases ht with rfl | rfl <;> simp) x
  have hlt : (x &&& widthMask t).toNat < 2 ^ t.width := by
    rw [toNat_wm t (by rcases ht with rfl | rfl <;> simp)]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  rcases ht with rfl | rfl
  all_goals
    simp only [ClifTy.width] at hm hlt
    simp only [PtxRun.toInt, signed, sTy, ClifTy.width, PtxIr.Ty.isSigned, PtxIr.Ty.width, Bool.true_and, hm]
    simp only [ge_iff_le, UInt64.le_iff_toNat_le]
    split <;> split <;> simp_all <;> omega

open PtxRun in
/-- **A sign-extension to 64 bits** of a 16- or 32-bit integer meets its
    obligation. -/
theorem sextend_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false) (htd : F.tys.get? d.id = some .i64)
    (hld : F.lits.get? d.id = none) (ha8 : F.tys.get? a.id ≠ some .i8)
    (hmo : magOk F d (.sextend64 d a) = true) :
    InstOk F lc c start0 (.sextend64 d a) := by
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a (fun t dr xo => .cvt none .s64 (sTy t.width) dr xo) (fun _ => .i64)
    (fun t x => UInt64.ofNat ((signed t x).emod ((1 <<< 64 : Nat) : Int)).toNat &&& widthMask .i64)
    (fun t => t = .i16 ∨ t = .i32) (fun _ _ h => nomatch h) rfl (by simp [Inst.uses]) hwa hwd (fun _ _ => htd) hld
  · intro s s1 hst _ t xa hva hta
    simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.un, Sem.get, ofInt,
      norm] at hst
    by_cases hc : t.isInt = true ∧ t.width < 64
    · simp [hc] at hst
      refine ⟨?_, hst.symm⟩
      rw [hta] at ha8
      cases t <;> simp_all [ClifTy.isInt, ClifTy.width]
    · simp [hc] at hst
  · intro tag t ht
    have ht' : F.tys[a.id]? = some t := by rw [← Std.HashMap.get?_eq_getElem?]; exact ht
    simp [lower_eq, isTerm, inst, emit, emitL, o, facts?, tyOf, ht']
  · intro t xa ht hn _ c' ts' xo dr hxo
    have hne8 : t ≠ .i8 := by rcases ht with rfl | rfl <;> decide
    rw [exec_cvt c' ts' none .s64 _ dr xo _ _ hxo
      (cvt_int none .s64 _ _ rfl (by rcases ht with rfl | rfl <;> rfl))]
    simp only [enc, hne8, show ClifTy.i64 ≠ .i8 by decide, if_false, and_wide64, hn]
    rw [show PtxIr.Ty.width .s64 = 64 from rfl, PtxRun.ofIntW, mask64, toInt_signed t ht xa]
    simp only [UInt64.ofInt, show (1 <<< 64 : Nat) = 2 ^ 64 from rfl]
    rfl
  · intro t xa _ _; rw [and_wide64]

/-! ## Comparisons -/

/-- Reading an operand survives writing a register it does not name. -/
theorem opVal_insert (c : PtxRun.Ctx) (ts : PtxRun.TS) (o : PtxIr.Opnd) (x : UInt64) (dr : PtxIr.Reg)
    (v : UInt64) (h : OpVal c ts o x) (hne : ∀ R, o = .reg R → R ≠ dr) :
    OpVal c { ts with regs := ts.regs.insert dr v } o x := by
  cases o with
  | reg R =>
      have hR := hne R rfl
      by_cases hs : ∃ s, R = .special s
      · obtain ⟨s, rfl⟩ := hs
        simpa [OpVal, PtxRun.opnd, PtxRun.getReg, StateT.run, pure, StateT.pure, Except.pure] using h
      · have hs' : ∀ s, R ≠ .special s := fun s h => hs ⟨s, h⟩
        simp [OpVal, PtxRun.opnd, getReg_plain _ _ hs'] at h ⊢
        rw [Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hR)]; exact h
  | sym n =>
      cases hl : c.syms.lookup n <;>
        simp_all [OpVal, PtxRun.opnd, StateT.run, pure, StateT.pure, Except.pure, throw, throwThe,
          MonadExceptOf.throw, StateT.lift, bind, Except.bind]
  | _ => simpa [OpVal, PtxRun.opnd] using h

/-- A value's register, unless a flag's, is not a flag's or a negation's. -/
theorem r_not_flag (F : Facts) (tag : Nat) (s : Out) (v : Val) (t : ClifTy) (hw : ¬ Win F v)
    (ht : F.tys.get? v.id = some t) (h8 : t ≠ .i8) (n : Nat) :
    (runLM (r v) F tag s).1 ≠ .val "p" n ∧ (runLM (r v) F tag s).1 ≠ .val "q" n := by
  cases hl : F.lits.get? v.id with
  | some l => rw [runLM_r_lit F tag s v t l hw ht hl]; simp
  | none =>
      rw [runLM_r_plain F tag s v t hw ht hl]
      cases t <;> simp_all [regOf, cls]

/-- What `imm v` gives: a constant's literal, or `v`'s register. -/
theorem imm_eq (F : Facts) (tag : Nat) (s : Out) (v : Val) :
    (F.lits.get? v.id = none → runLM (imm v) F tag s = (.reg (runLM (r v) F tag s).1, (runLM (r v) F tag s).2)) ∧
    ∀ l, F.lits.get? v.id = some l → (runLM (imm v) F tag s).1 = l := by
  refine ⟨fun hl => ?_, fun l hl => ?_⟩ <;> simp only [Std.HashMap.get?_eq_getElem?] at hl <;>
    simp [imm, facts?, hl, o]

open PtxRun in
theorem exec_setp (c : PtxRun.Ctx) (ts : PtxRun.TS) (cc : PtxIr.Cmp) (ty : PtxIr.Ty) (dr : PtxIr.Reg)
    (xo yo : PtxIr.Opnd) (x y : UInt64) (q : Bool) (hx : OpVal c ts xo x) (hy : OpVal c ts yo y)
    (hq : PtxRun.cmp cc ty x y = .ok q) :
    (exec c (.setp cc ty dr xo yo)).run ts = .ok (.step, { ts with regs := ts.regs.insert dr (if q then 1 else 0) }) := by
  simp only [OpVal] at hx hy
  simp only [exec, StateT.run, bind, StateT.bind] at hx hy ⊢
  simp [hx, hy, Except.bind, lift, hq, setReg, pure, StateT.pure, Except.pure, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet]

/-- Reading an integer as signed, as both semantics do it. -/
theorem toInt_signed' (t : ClifTy) (ht : IntT t) (x : UInt64) :
    PtxRun.toInt (sTy t.width) x = signed t x := by
  rcases ht with rfl | rfl | rfl
  · exact toInt_signed .i16 (Or.inl rfl) x
  · exact toInt_signed .i32 (Or.inr rfl) x
  · simp only [PtxRun.toInt, signed, sTy, ClifTy.width, PtxIr.Ty.isSigned, PtxIr.Ty.width, Bool.true_and, mask64,
      and_wide64, ge_iff_le, UInt64.le_iff_toNat_le]
    rw [show (1 <<< 64 : Nat) = 2 ^ 64 from rfl,
      show (9223372036854775808 : UInt64).toNat = 2 ^ 63 from rfl]
    simp only [Nat.le_refl, if_true]
    split <;> split <;> simp_all <;> omega

/-- An unsigned PTX type reads a value cut to its width as itself. -/
theorem toInt_u (t : ClifTy) (ht : IntT t) (x : UInt64) (hx : x &&& widthMask t = x) :
    PtxRun.toInt (uTy t.width) x = x.toNat := by
  have hm : PtxRun.mask (uTy t.width).width x = x := by
    rw [show (uTy t.width).width = t.width by rcases ht with rfl | rfl | rfl <;> rfl, mask_width t ht, hx]
  simp only [PtxRun.toInt, hm]
  rcases ht with rfl | rfl | rfl <;> simp [uTy, ClifTy.width, PtxIr.Ty.isSigned]

theorem toInt_mask (ty : PtxIr.Ty) (y : UInt64) :
    PtxRun.toInt ty (PtxRun.mask ty.width y) = PtxRun.toInt ty y := by
  have hm : ∀ w, PtxRun.mask w (PtxRun.mask w y) = PtxRun.mask w y := by
    intro w; unfold PtxRun.mask; split
    · rfl
    · rw [UInt64.and_assoc, UInt64.and_self]
  simp only [PtxRun.toInt, hm]

/-- **A comparison's PTX** on the operands' encodings is the block
    semantics' comparison. -/
theorem icmp_cmp (cc : ICmpCond) (t : ClifTy) (ht : IntT t) (xa xb y : UInt64)
    (hxa : xa &&& widthMask t = xa) (hxb : xb &&& widthMask t = xb)
    (hy : PtxRun.mask (bits t).width y = enc t xb) :
    PtxRun.cmp (icc cc).1 (if (icc cc).2 then sTy t.width else uTy t.width) (enc t xa) y =
      .ok (cmpInt cc t xa xb) := by
  have hne8 : t ≠ .i8 := by rcases ht with rfl | rfl | rfl <;> decide
  have hbw : (bits t).width = t.width := by rcases ht with rfl | rfl | rfl <;> rfl
  have huw : (uTy t.width).width = t.width := by rcases ht with rfl | rfl | rfl <;> rfl
  have hsw : (sTy t.width).width = t.width := by rcases ht with rfl | rfl | rfl <;> rfl
  have huf : (uTy t.width).isFloat = false := by rcases ht with rfl | rfl | rfl <;> rfl
  have hsf : (sTy t.width).isFloat = false := by rcases ht with rfl | rfl | rfl <;> rfl
  simp only [enc, hne8, if_false, hxa] at hy ⊢
  rw [hxb, hbw] at hy
  have hmU : PtxRun.mask (uTy t.width).width y = xb := by rw [huw, hy]
  have hmS : PtxRun.mask (sTy t.width).width y = xb := by rw [hsw, hy]
  have hxU : PtxRun.mask (uTy t.width).width xa = xa := by rw [huw, mask_width t ht, hxa]
  have hxS : PtxRun.mask (sTy t.width).width xa = xa := by rw [hsw, mask_width t ht, hxa]
  have hyU : PtxRun.toInt (uTy t.width) y = xb.toNat := by
    rw [← toInt_mask, hmU, toInt_u t ht xb hxb]
  have hyS : PtxRun.toInt (sTy t.width) y = signed t xb := by
    rw [← toInt_mask, hmS, toInt_signed' t ht xb]
  have haU := toInt_u t ht xa hxa
  have haS := toInt_signed' t ht xa
  cases cc <;>
    simp [icc, cmpInt, PtxRun.cmp, huf, hsf, hmU, hmS, hxU, hxS, hyU, hyS, haU, haS,
      UInt64.lt_iff_toNat_lt, UInt64.le_iff_toNat_le, UInt64.toNat_inj]
  all_goals
    rw [Bool.eq_iff_iff]
    have hc : ((xa.toNat : Int) = xb.toNat) ↔ xa = xb := ⟨fun h => UInt64.toNat_inj.mp (by omega), fun h => by rw [h]⟩
    simp [hc]

/-- The negated condition compares as the comparison's negation. -/
theorem cmpInt_neg (cc : ICmpCond) (t : ClifTy) (x y : UInt64) :
    cmpInt (icneg cc) t x y = !cmpInt cc t x y := by
  cases cc <;> simp [icneg, cmpInt, UInt64.lt_iff_toNat_lt, UInt64.le_iff_toNat_le]
  all_goals first
    | rfl
    | (rw [Bool.eq_iff_iff]; simp [Nat.not_le, Nat.not_lt, Int.not_le, Int.not_lt])

theorem icc_neg_sign (cc : ICmpCond) : (icc (icneg cc)).2 = (icc cc).2 := by cases cc <;> rfl

open PtxRun in
/-- **An integer comparison** of values not flags nor windowed meets its
    obligation: its flag and, beside it, its negation. -/
theorem icmp_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d : Val) (cc : ICmpCond)
    (a b : Val) (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t)
    (htd : F.tys.get? d.id = some .i8) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.icmp d cc a b) = true) :
    InstOk F lc c start0 (.icmp d cc a b) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have hua : a ∈ Inst.uses (.icmp d cc a b) := by simp [Inst.uses]
  have hub : b ∈ Inst.uses (.icmp d cc a b) := by simp [Inst.uses]
  obtain ⟨ta, xa, hva, hta⟩ := hS.pres a (huses a hua)
  have hit := hT ta hta
  have hne8 : ta ≠ .i8 := by rcases hit with rfl | rfl | rfl <;> decide
  -- the block semantics' step
  obtain ⟨xb, hvb, rfl⟩ : ∃ xb, Blocks.getV s.vals b = some (.sc ta xb) ∧
      s1 = { s with vals := Blocks.setV s.vals d (.sc .i8 (if cmpInt cc ta xa xb then 1 else 0)) } := by
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get,
        zipIntCmp] at hst
    | sc tb xb =>
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipIntCmp,
      boolV] at hst
    by_cases hc : ta = tb ∧ ta.isInt = true
    · obtain ⟨rfl, hci⟩ := hc
      simp [hci] at hst
      exact ⟨xb, rfl, hst.symm⟩
    · simp [hc] at hst
  have htb : F.tys.get? b.id = some ta := by
    obtain ⟨tb, xb', hvb', htb⟩ := hS.pres b (huses b hub)
    rw [hvb] at hvb'; obtain ⟨rfl, -⟩ := V.sc.inj (Option.some.inj hvb'); exact htb
  have hxa := hS.normd a ta xa (huses a hua) hva
  have hxb := hS.normd b ta xb (huses b hub) hvb
  -- the operands' lines
  let sa := (runLM (r a) F tag {}).2
  obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag {} s.vals c ts a ta xa D hS (huses a hua) hva hta hwa
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lb, hLb, hn2, regs2, yv, hrun2, hop2, hy, hkeep2⟩ :=
    imm_sound F tag sa s.vals c { ts with regs := regs1 } b ta xb D hS1 (huses b hub) hvb htb hwb
      (by rcases hit with rfl | rfl | rfl <;> simp [Scalar])
  let X : PtxIr.Opnd := .reg (runLM (r a) F tag {}).1
  let Y : PtxIr.Opnd := (runLM (imm b) F tag sa).1
  let ty : PtxIr.Ty := if (icc cc).2 then sTy ta.width else uTy ta.width
  let core1 : PtxIr.Instr := .setp (icc cc).1 ty (regOf .i8 d) X Y
  let core2 : PtxIr.Instr := .setp (icc (icneg cc)).1 ty (negReg d) X Y
  have hlow : (lower F tag (.icmp d cc a b)).lines =
      (((runLM (imm b) F tag sa).2.lines).push (.ins none core1)).push (.ins none core2) := by
    have ht' : F.tys[a.id]? = some ta := by rw [← Std.HashMap.get?_eq_getElem?]; exact hta
    have h8 : (ta == .i8) = false := by simpa using hne8
    rcases hcc : icc cc with ⟨c0, s0⟩
    simp [lower_eq, isTerm, inst, emit, emitL, facts?, tyOf, ht', h8, hne8, hcc, core1, core2, X, Y, ty, sa, o]
  have hlist : (lower F tag (.icmp d cc a b)).lines =
      ((La ++ Lb ++ [core1, core2]).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow, hLb]
    show ((sa.lines ++ _).push _).push _ = _
    rw [hLa]; simp
  have hins : instrsOf (lower F tag (.icmp d cc a b)).lines = La ++ Lb ++ [core1, core2] := by
    rw [hlist, instrsOf_map]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  -- the comparisons
  have hX : OpVal c { ts with regs := regs2 } X (enc ta xa) :=
    opVal_reg c _ _ _ hns1 (by rw [hkeep2 _ hnf1]; exact hreg1)
  have hY : OpVal c { ts with regs := regs2 } Y yv := hop2 _ (fun _ _ => rfl)
  have hq1 := icmp_cmp cc ta hit xa xb yv hxa hxb hy
  have hq2 := icmp_cmp (icneg cc) ta hit xa xb yv hxa hxb hy
  rw [icc_neg_sign, cmpInt_neg] at hq2
  -- the operands' registers are neither the flag's nor its negation's
  have hXne : ∀ R, X = .reg R → R ≠ regOf .i8 d := by
    intro R hR; simp only [X, PtxIr.Opnd.reg.injEq] at hR; subst hR
    exact (r_not_flag F tag {} a ta hwa hta hne8 d.id).1
  have hYne : ∀ R, Y = .reg R → R ≠ regOf .i8 d := by
    intro R hR
    cases hlb : F.lits.get? b.id with
    | some l =>
        obtain ⟨hil, -⟩ := hS.lit b ta xb l (huses b hub) hvb htb hlb
        have := (imm_eq F tag sa b).2 l hlb
        simp only [Y] at hR; rw [this] at hR; subst hR; cases hil
    | none =>
        have := (imm_eq F tag sa b).1 hlb
        simp only [Y, this, PtxIr.Opnd.reg.injEq] at hR; subst hR
        exact (r_not_flag F tag sa b ta hwb htb hne8 d.id).1
  let v1 : UInt64 := if cmpInt cc ta xa xb then 1 else 0
  let v2 : UInt64 := if !cmpInt cc ta xa xb then 1 else 0
  have he1 : (exec c core1).run { ts with regs := regs2 } =
      .ok (.step, { ts with regs := regs2.insert (regOf .i8 d) v1 }) :=
    exec_setp c _ _ _ _ _ _ _ _ _ hX hY hq1
  have he2 : (exec c core2).run { ts with regs := regs2.insert (regOf .i8 d) v1 } =
      .ok (.step, { ts with regs := (regs2.insert (regOf .i8 d) v1).insert (negReg d) v2 }) :=
    exec_setp c { ts with regs := regs2.insert (regOf .i8 d) v1 } _ _ _ _ _ _ _ _
      (opVal_insert c _ X _ _ v1 hX hXne) (opVal_insert c _ Y _ _ v1 hY hYne) hq2
  refine ⟨{ ts with regs := (regs2.insert (regOf .i8 d) v1).insert (negReg d) v2 }, ?_, ?_, hW⟩
  · rw [hins, steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]
    simp only [StateT.run] at he1 he2
    simp [steps, StateT.run, he1, he2, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · have hpq : negReg d ≠ regOf .i8 d := by simp [negReg, regOf, cls]
    apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ _ d .i8 v1 rfl hS htd
    · (by_cases h : cmpInt cc ta xa xb <;> simp [v1, h, widthMask]) <;> decide
    · intro _ _
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using hpq),
        Std.HashMap.getElem?_insert_self]
      (by_cases h : cmpInt cc ta xa xb <;> simp [v1, h, enc]) <;> decide
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro _ _
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert_self]
      (by_cases h : cmpInt cc ta xa xb <;> simp [v1, v2, h]) <;> decide
    · intro _; (by_cases h : cmpInt cc ta xa xb <;> simp [v1, h]) <;> decide
    · intro c' n hn
      have hne1 : PtxIr.Reg.val c' n ≠ regOf .i8 d := by simp [regOf]; omega
      have hne2 : PtxIr.Reg.val c' n ≠ negReg d := by simp [negReg]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne2),
        Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne1), ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h), hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]

/-! ## Choice -/

theorem fresh_mono {tag n m : Nat} {R : PtxIr.Reg} (h : n ≤ m) (hf : Fresh tag m R) : Fresh tag n R := by
  obtain ⟨p, k, hk, rfl⟩ := hf; exact ⟨p, k, by omega, rfl⟩

/-- An instruction other than a terminator lowers by `inst`. -/
theorem lower_inst (F : Facts) (tag : Nat) (i : Inst) (h : isTerm i = false) :
    lower F tag i = (runLM (inst i) F tag {}).2 := by
  rw [lower_eq, if_neg (by simp [h])]

theorem runLM_tyOf (F : Facts) (tag : Nat) (s : Out) (a : Val) (ta : ClifTy) (hta : F.tys.get? a.id = some ta) :
    runLM (tyOf a) F tag s = (ta, s) := by
  simp only [facts?, tyOf, hta, runLM_bind, runLM_read, runLM_pure]

open PtxRun in
/-- **A 64-bit product of a value within 32 bits and a constant within 32
    bits** (`Facts.wideMul`) meets its obligation: the value's low half into a
    32-bit register of the instruction's own, then `mul.wide.u32` of it and the
    constant, which cannot overflow. -/
theorem imulWide_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none) (hwm : F.wideMul a b = true)
    (hmo : magOk F d (.imul d a b) = true) :
    InstOk F lc c start0 (.imul d a b) := by
  intro tag D s s1 ts hst huses hS hW _ _
  have hwm' := hwm
  simp only [Facts.wideMul, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hwm'
  obtain ⟨⟨⟨⟨⟨-, hta⟩, hla⟩, hba⟩, hbb⟩, hlb⟩ := hwm'
  obtain ⟨k, hkb⟩ : ∃ k, F.lits.get? b.id = some (.imm k) := by
    revert hlb
    cases F.lits.get? b.id with
    | none => simp
    | some l => cases l <;> simp
  have htd : F.tys.get? d.id = some .i64 := (hT _ hta).2
  have hDa : D a.id := huses a (by simp [Inst.uses])
  have hDb : D b.id := huses b (by simp [Inst.uses])
  obtain ⟨ta, xa, hva, hta'⟩ := hS.pres a hDa
  rw [hta] at hta'; cases hta'
  obtain ⟨xb, hvb, rfl⟩ := step_imul lc s s1 d a b .i64 xa hst hva
  have htb : F.tys.get? b.id = some .i64 := by
    obtain ⟨tb, xb', hvb', htb⟩ := hS.pres b hDb
    rw [hvb] at hvb'; obtain ⟨rfl, -⟩ := V.sc.inj (Option.some.inj hvb'); exact htb
  obtain ⟨hil, hraw⟩ := hS.lit b .i64 xb (.imm k) hDb hvb htb hkb
  have hxa := hS.plain a .i64 xa hDa hva hta hla hwa
  have he64 : ∀ z : UInt64, enc .i64 z = z := fun z => by simp [enc, and_wide64]
  rw [he64] at hraw hxa
  have hxa32 : xa.toNat < 2 ^ 32 :=
    Nat.lt_of_lt_of_le (bnd_val (hS.bound a .i64 xa hDa hva)) (Nat.pow_le_pow_right (by decide) hba)
  have hxb32 : xb.toNat < 2 ^ 32 :=
    Nat.lt_of_lt_of_le (bnd_val (hS.bound b .i64 xb hDb hvb)) (Nat.pow_le_pow_right (by decide) hbb)
  let w : PtxIr.Reg := .own "w" tag 0
  let i1 : PtxIr.Instr := .cvt none .u32 .u64 w (.reg (regOf .i64 a))
  let i2 : PtxIr.Instr := .bin .mulWide none .u32 (regOf .i64 d) (.reg w) (.imm k)
  have hlow : (lower F tag (.imul d a b)).lines =
      (([i1, i2] : List PtxIr.Instr).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_inst F tag _ rfl]
    simp only [inst, facts?, runLM_bind, runLM_read, runLM_pure, hwm, if_true]
    rw [runLM_r_plain F tag _ a .i64 hwa hta hla]
    simp only [Std.HashMap.get?_eq_getElem?] at hkb
    simp [own, emit, emitL, imm, facts?, hkb, i1, i2, w]
  refine ⟨by rw [hlow]; exact straight_map _, ?_⟩
  let wv : UInt64 := PtxRun.ofIntW 32 (PtxRun.toInt .u64 xa)
  have he1 : (exec c i1).run ts = .ok (.step, { ts with regs := ts.regs.insert w wv }) :=
    exec_cvt c ts none .u32 .u64 w _ xa wv (opVal_reg c ts _ xa (not_special_val _ _) hxa)
      (cvt_int none .u32 .u64 xa rfl rfl)
  have hbin : PtxRun.bin .mulWide .u32 wv xb = .ok (xa * xb) := by
    rw [← mulWide_low xa xb hxa32 hxb32]; rfl
  have he2 : (exec c i2).run { ts with regs := ts.regs.insert w wv } =
      .ok (.step, { ts with regs := (ts.regs.insert w wv).insert (regOf .i64 d) (xa * xb) }) :=
    exec_bin c _ .mulWide .u32 (regOf .i64 d) (.reg w) (.imm k) wv xb _
      (opVal_reg c _ w wv (not_special_own _ _ _) (by simp [Std.HashMap.get?_eq_getElem?]))
      (by rw [← hraw]; exact opVal_lit c _ (.imm k) hil) hbin
  refine ⟨{ ts with regs := (ts.regs.insert w wv).insert (regOf .i64 d) (xa * xb) }, ?_, ?_, hW⟩
  · rw [hlow, instrsOf_map]
    simp only [StateT.run] at he1 he2
    simp [steps, StateT.run, he1, he2, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun k' hk' => by simp [magRule] at hk') F D s.vals ts.regs _ _ d .i64
      (xa * xb &&& widthMask .i64) rfl hS htd (by rw [and_wide64, and_wide64])
    · intro _ _
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert_self, he64, and_wide64]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; cases h8
    · intro h8; cases h8
    · intro c' n hn
      have hne : PtxIr.Reg.val c' n ≠ regOf .i64 d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        Std.HashMap.getElem?_insert, if_neg (by simp [w]), ← Std.HashMap.get?_eq_getElem?]

theorem inst_select (d cv a b : Val) :
    inst (.select d cv a b) = (do
      let t ← tyOf a
      let x ← imm a; let y ← imm b
      emit (.selp (bits t) (regOf t d) x y (← o cv))) := by simp only [inst]

/-- A choice lowers to its arms' and flag's lines, then `selp`. -/
theorem lower_select (F : Facts) (tag : Nat) (d cv a b : Val) (ta : ClifTy) (hta : F.tys.get? a.id = some ta) :
    (lower F tag (.select d cv a b)).lines =
      ((runLM (r cv) F tag (runLM (imm b) F tag (runLM (imm a) F tag {}).2).2).2.lines).push
        (.ins none (.selp (bits ta) (regOf ta d) (runLM (imm a) F tag {}).1
          (runLM (imm b) F tag (runLM (imm a) F tag {}).2).1
          (.reg (runLM (r cv) F tag (runLM (imm b) F tag (runLM (imm a) F tag {}).2).2).1))) := by
  rw [lower_inst F tag _ rfl, inst_select, runLM_bind, runLM_tyOf F tag _ a ta hta]
  simp only [runLM_bind, runLM_pure, o, emit, emitL, runLM_modify]

open PtxRun in
/-- **A choice** between values of one type, not flags nor windowed, on a
    flag meets its obligation: `selp` on the flag's predicate. -/
theorem select_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d cv a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwc : ¬ Win F cv) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → Scalar t ∧ t ≠ .i8 ∧ F.tys.get? d.id = some t)
    (htc : F.tys.get? cv.id = some .i8) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.select d cv a b) = true) :
    InstOk F lc c start0 (.select d cv a b) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have hua : a ∈ Inst.uses (.select d cv a b) := by simp [Inst.uses]
  have hub : b ∈ Inst.uses (.select d cv a b) := by simp [Inst.uses]
  have huc : cv ∈ Inst.uses (.select d cv a b) := by simp [Inst.uses]
  obtain ⟨ta, xa, hva, hta⟩ := hS.pres a (huses a hua)
  obtain ⟨hsc, hne8, htd⟩ := hT ta hta
  obtain ⟨tc, xc, hvc, htc'⟩ := hS.pres cv (huses cv huc)
  rw [htc] at htc'; cases htc'
  -- the block semantics' step
  obtain ⟨xb, hvb, rfl⟩ : ∃ xb, Blocks.getV s.vals b = some (.sc ta xb) ∧
      s1 = { s with vals := Blocks.setV s.vals d (.sc ta (if xc != 0 then xa else xb)) } := by
    obtain ⟨tb, xb, hvb, -⟩ := hS.pres b (huses b hub)
    simp [step1, Blocks.evalInst, hva, hvb, hvc, Blocks.viaOp, evalOp, Sem.get, Sem.isTrue, V.ty,
      ClifTy.isInt] at hst
    by_cases h : ta = tb
    · subst h
      refine ⟨xb, hvb, ?_⟩
      by_cases hx : xc = 0 <;> simp_all
    · simp [h] at hst
  have htb : F.tys.get? b.id = some ta := by
    obtain ⟨tb, xb', hvb', htb⟩ := hS.pres b (huses b hub)
    rw [hvb] at hvb'; obtain ⟨rfl, -⟩ := V.sc.inj (Option.some.inj hvb'); exact htb
  -- the operands' lines: the arms, then the flag
  let s0 : Out := {}
  let sa := (runLM (imm a) F tag s0).2
  let sb := (runLM (imm b) F tag sa).2
  obtain ⟨La, hLa, hn1, regs1, x, hrun1, hop1, hx, hkeep1⟩ :=
    imm_sound F tag s0 s.vals c ts a ta xa D hS (huses a hua) hva hta hwa hsc
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lb, hLb, hn2, regs2, y, hrun2, hop2, hy, hkeep2⟩ :=
    imm_sound F tag sa s.vals c { ts with regs := regs1 } b ta xb D hS1 (huses b hub) hvb htb hwb hsc
  have hS2 : Sim F D s.vals regs2 := hS1.frame (fun c' n => hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lc, hLc, hn3, hnf3, hns3, regs3, hrun3, hreg3, hkeep3⟩ :=
    r_sound F tag sb s.vals c { ts with regs := regs2 } cv .i8 xc D hS2 (huses cv huc) hvc htc hwc
  have hlow := lower_select F tag d cv a b ta hta
  let X := (runLM (imm a) F tag s0).1
  let Y := (runLM (imm b) F tag sa).1
  let P : PtxIr.Opnd := .reg (runLM (r cv) F tag sb).1
  let core : PtxIr.Instr := .selp (bits ta) (regOf ta d) X Y P
  have hlist : (lower F tag (.select d cv a b)).lines =
      ((La ++ Lb ++ Lc ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow, hLc, hLb, hLa]
    simp [core, X, Y, P, sa, sb, s0]
  have hins : instrsOf (lower F tag (.select d cv a b)).lines = La ++ Lb ++ Lc ++ [core] := by
    rw [hlist, instrsOf_map]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  -- the operands' values where `selp` reads them
  have hk31 : ∀ R, ¬ Fresh tag sa.n R → regs3.get? R = regs1.get? R := by
    intro R hR
    rw [hkeep3 R (fun h => hR (fresh_mono hn2 h)), hkeep2 R hR]
  have hX : OpVal c { ts with regs := regs3 } X x := hop1 _ hk31
  have hY : OpVal c { ts with regs := regs3 } Y y := hop2 _ (fun R hR => hkeep3 R hR)
  have hP : OpVal c { ts with regs := regs3 } P (enc .i8 xc) := opVal_reg c _ _ _ hns3 hreg3
  have hfl := hS.flag cv xc (huses cv huc) hvc
  let v : UInt64 := enc ta (if xc != 0 then xa else xb)
  have he : (exec c core).run { ts with regs := regs3 } =
      .ok (.step, { ts with regs := regs3.insert (regOf ta d) v }) := by
    rw [exec_selp c _ _ _ _ _ _ x y _ hX hY hP]
    rcases flag_cases xc hfl with rfl | rfl
    · have h0 : (enc .i8 0 &&& 1 == 1) = false := by decide
      simp only [h0, v, Bool.false_eq_true, if_false, show ((0 : UInt64) != 0) = false from rfl, hy]
    · have h1 : (enc .i8 1 &&& 1 == 1) = true := by decide
      simp only [h1, v, if_true, show ((1 : UInt64) != 0) = true from rfl, hx]
  refine ⟨{ ts with regs := regs3.insert (regOf ta d) v }, ?_, ?_, hW⟩
  · rw [hins, steps_append c _ _ ts _
      (by rw [steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]; exact hrun3)]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ _ d ta _ rfl hS htd
    · by_cases h : xc = 0 <;> simp [h, hS.normd a ta xa (huses a hua) hva, hS.normd b ta xb (huses b hub) hvb]
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?, v]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; exact absurd h8 hne8
    · intro h8; exact absurd h8 hne8
    · intro c' n hn
      have hne : PtxIr.Reg.val c' n ≠ regOf ta d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep3 _ (fun ⟨p, k, _, h⟩ => by cases h), hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h),
        hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]

/-! ## Shifts -/

/-- A shift amount masked in 32 bits to an integer width's bits is the
    amount modulo the width. -/
theorem amt_mask (t : ClifTy) (ht : IntT t) (y : UInt64) :
    (PtxRun.mask 32 (PtxRun.mask 32 y &&& UInt64.ofInt ((t.width : Int) - 1))).toNat = y.toNat % t.width := by
  have h32 : ∀ z : UInt64, PtxRun.mask 32 z = z &&& ((2 ^ 32 - 1 : Nat).toUInt64) := fun z => by
    simp [PtxRun.mask]; rfl
  rw [h32, h32]
  rcases ht with rfl | rfl | rfl
  all_goals
    simp only [ClifTy.width]
    rw [UInt64.toNat_and, UInt64.toNat_and, UInt64.toNat_and]
    simp only [show ((2 ^ 32 - 1 : Nat).toUInt64).toNat = 2 ^ 32 - 1 from rfl]
  · rw [show (UInt64.ofInt ((16 : Nat) - 1 : Int)).toNat = 2 ^ 4 - 1 from rfl]
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod]
    omega
  · rw [show (UInt64.ofInt ((32 : Nat) - 1 : Int)).toNat = 2 ^ 5 - 1 from rfl]
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod]
    omega
  · rw [show (UInt64.ofInt ((64 : Nat) - 1 : Int)).toNat = 2 ^ 6 - 1 from rfl]
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod]
    omega

theorem shiftLit_spec (F : Facts) (ta : ClifTy) (b : Val) (k : Int) (h : shiftLit F ta b = some k) :
    F.lits.get? b.id = some (.imm k) ∧ 0 ≤ k ∧ k < ta.width := by
  unfold shiftLit at h
  split at h
  · rename_i k' hk'
    split at h
    · cases h; simp_all [Std.HashMap.get?_eq_getElem?]
    · cases h
  · cases h

theorem runLM_shiftAmt_lit (F : Facts) (tag : Nat) (s : Out) (ta : ClifTy) (b : Val) (k : Int)
    (hk : shiftLit F ta b = some k) : runLM (shiftAmt ta b) F tag s = (.imm k, s) := by
  simp [shiftAmt, facts?, hk]

/-- The masked amount's lines: `b` into a 32-bit register of the instruction's
    own, then cut to the width's bits. -/
def amtMove (tb : ClifTy) (sa rb : PtxIr.Reg) : PtxIr.Instr :=
  if tb = .i64 then .cvt none .u32 .u64 sa (.reg rb) else .mov .b32 sa (.reg rb)

theorem runLM_shiftAmt_reg (F : Facts) (tag : Nat) (s : Out) (ta tb : ClifTy) (b : Val)
    (hk : shiftLit F ta b = none) (htb : F.tys.get? b.id = some tb) :
    runLM (shiftAmt ta b) F tag s =
      (.reg (.own "sa" tag s.n),
        { (runLM (r b) F tag { s with n := s.n + 1, regs := s.regs.push (.own "sa" tag s.n, .i32) }).2 with
          lines := (((runLM (r b) F tag { s with n := s.n + 1, regs := s.regs.push (.own "sa" tag s.n, .i32) }).2.lines).push
            (.ins none (amtMove tb (.own "sa" tag s.n)
              (runLM (r b) F tag { s with n := s.n + 1, regs := s.regs.push (.own "sa" tag s.n, .i32) }).1))).push
            (.ins none (.bin .and none .b32 (.own "sa" tag s.n) (.reg (.own "sa" tag s.n)) (.imm ((ta.width : Int) - 1)))) }) := by
  have htb' : F.tys[b.id]? = some tb := by rw [← Std.HashMap.get?_eq_getElem?]; exact htb
  by_cases h64 : tb = .i64
  · subst h64; simp [shiftAmt, facts?, hk, own, tyOf, htb', emit, emitL, o, amtMove]
  · have h64' : (tb == .i64) = false := by simpa using h64
    simp [shiftAmt, facts?, hk, own, tyOf, htb', emit, emitL, o, amtMove, h64, h64']

open PtxRun in
/-- **A shift amount**: the lines `shiftAmt ta b` writes for an integer value
    in scope leave an operand reading the amount modulo `ta`'s width, changing
    no register but the instruction's own. -/
theorem shiftAmt_sound (F : Facts) (tag : Nat) (s : Out) (vs : Blocks.Vals) (c : PtxRun.Ctx) (ts : PtxRun.TS)
    (b : Val) (ta tb : ClifTy) (yb : UInt64) (hta : IntT ta) (htbi : IntT tb)
    (D : Nat → Prop) (hS : Sim F D vs ts.regs) (hD : D b.id) (hv : Blocks.getV vs b = some (.sc tb yb))
    (htb : F.tys.get? b.id = some tb) (hw : ¬ Win F b) :
    ∃ L : List PtxIr.Instr,
      (runLM (shiftAmt ta b) F tag s).2.lines = s.lines ++ (L.map (.ins none)).toArray ∧
      s.n ≤ (runLM (shiftAmt ta b) F tag s).2.n ∧
      ∃ regs' A, (steps c L).run ts = .ok ((), { ts with regs := regs' }) ∧
        (∀ ts2 : PtxRun.TS, (∀ R, ¬ Fresh tag (runLM (shiftAmt ta b) F tag s).2.n R → ts2.regs.get? R = regs'.get? R) →
          OpVal c ts2 (runLM (shiftAmt ta b) F tag s).1 A) ∧
        A.toNat = yb.toNat % ta.width ∧
        ∀ R, ¬ Fresh tag s.n R → regs'.get? R = ts.regs.get? R := by
  have hne8 : tb ≠ .i8 := by rcases htbi with rfl | rfl | rfl <;> decide
  have hnb := hS.normd b tb yb hD hv
  cases hk : shiftLit F ta b with
  | some k =>
      obtain ⟨hl, hk0, hkw⟩ := shiftLit_spec F ta b k hk
      obtain ⟨-, hraw⟩ := hS.lit b tb yb (.imm k) hD hv htb hl
      simp only [litRaw, enc, hne8, if_false, hnb] at hraw
      rw [runLM_shiftAmt_lit F tag s ta b k hk]
      refine ⟨[], by simp, Nat.le_refl _, ts.regs, UInt64.ofInt k, ?_, fun ts2 _ => opVal_lit c ts2 (.imm k) trivial,
        ?_, fun _ _ => rfl⟩
      · simp [steps, StateT.run, pure, StateT.pure, Except.pure]
      · rw [hraw]
        have hw64 : (ta.width : Int) ≤ 64 := by rcases hta with rfl | rfl | rfl <;> decide
        have : yb.toNat = k.toNat := by
          rw [← hraw]; simp only [UInt64.ofInt, UInt64.toNat_ofNat']; omega
        have hkn : k.toNat < ta.width := by omega
        rw [this, Nat.mod_eq_of_lt hkn]
  | none =>
      rw [runLM_shiftAmt_reg F tag s ta tb b hk htb]
      let sa : PtxIr.Reg := .own "sa" tag s.n
      let s1 : Out := { s with n := s.n + 1, regs := s.regs.push (sa, .i32) }
      obtain ⟨Lb, hLb, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
        r_sound F tag s1 vs c ts b tb yb D hS hD hv htb hw
      let mv := amtMove tb sa (runLM (r b) F tag s1).1
      let an : PtxIr.Instr := .bin .and none .b32 sa (.reg sa) (.imm ((ta.width : Int) - 1))
      refine ⟨Lb ++ [mv, an], ?_, ?_, ?_⟩
      · show ((_ : Array PtxIr.Line).push _).push _ = _
        rw [hLb]; simp [s1, sa, mv, an]
      · show s.n ≤ (runLM (r b) F tag s1).2.n
        have : s1.n = s.n + 1 := rfl
        omega
      -- the move leaves the amount's low 32 bits
      have hxb : OpVal c { ts with regs := regs1 } (.reg (runLM (r b) F tag s1).1) yb := by
        have := opVal_reg c { ts with regs := regs1 } _ _ hns1 hreg1
        simpa [enc, hne8, hnb] using this
      let v : UInt64 := PtxRun.mask 32 yb
      have hmv : (exec c mv).run { ts with regs := regs1 } = .ok (.step, { ts with regs := regs1.insert sa v }) := by
        by_cases h64 : tb = .i64
        · subst h64
          simp only [mv, amtMove, if_true]
          rw [exec_cvt c _ none .u32 .u64 sa _ yb _ hxb (cvt_int none .u32 .u64 yb rfl rfl)]
          simp only [PtxRun.ofIntW, PtxRun.toInt, PtxIr.Ty.isSigned, PtxIr.Ty.width, Bool.false_and,
            Bool.false_eq_true, if_false, mask64, ofInt_toNat, v]
        · simp only [mv, amtMove, h64, if_false]
          rw [exec_mov_reg c _ .b32 sa _ yb hns1 (by simpa [enc, hne8, hnb] using hreg1)]
          rfl
      let A : UInt64 := PtxRun.mask 32 (v &&& UInt64.ofInt ((ta.width : Int) - 1))
      have hsa : OpVal c { ts with regs := regs1.insert sa v } (.reg sa) v :=
        opVal_reg c _ _ _ (not_special_own _ _ _) (by simp [Std.HashMap.get?_eq_getElem?])
      have han : (exec c an).run { ts with regs := regs1.insert sa v } =
          .ok (.step, { ts with regs := (regs1.insert sa v).insert sa A }) :=
        exec_bin c _ .and .b32 sa (.reg sa) _ v _ A hsa (opVal_lit c _ (.imm _) trivial) rfl
      refine ⟨(regs1.insert sa v).insert sa A, A, ?_, ?_, amt_mask ta hta yb, ?_⟩
      · rw [steps_append c _ _ ts _ hrun1]
        simp only [StateT.run] at hmv han
        simp [steps, StateT.run, hmv, han, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
      · intro ts2 h2
        have hnf : ¬ Fresh tag (runLM (r b) F tag s1).2.n sa := by
          rintro ⟨p, k, hk, hh⟩; simp [sa] at hh; have : s1.n = s.n + 1 := rfl; omega
        exact opVal_reg c _ _ _ (not_special_own _ _ _) (by rw [h2 _ hnf]; simp [Std.HashMap.get?_eq_getElem?])
      · intro R hR
        have hne : sa ≠ R := fun h => hR ⟨"sa", s.n, Nat.le_refl _, h.symm⟩
        rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using hne),
          Std.HashMap.getElem?_insert, if_neg (by simpa using hne), ← Std.HashMap.get?_eq_getElem?]
        exact hkeep1 R (fun h => hR (fresh_mono (by show s.n ≤ s.n + 1; omega) h))

open PtxRun in
/-- **A shift** of an integer by an integer meets its obligation, given what
    the block semantics computes, its lowering (the amount's lines, the
    operand's, then one PTX instruction), and that the PTX instruction on the
    operand's encoding and the amount modulo the width gives the result's. -/
theorem shift_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (i : Inst) (d a b : Val)
    (bo : PtxIr.BinOp) (tyf : ClifTy → PtxIr.Ty) (f : ClifTy → UInt64 → UInt64 → UInt64)
    (hd : Inst.dst? i = some d) (hua : a ∈ Inst.uses i) (hub : b ∈ Inst.uses i)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hTb : ∀ t, F.tys.get? b.id = some t → IntT t)
    (hld : F.lits.get? d.id = none)
    (hstep : ∀ (s s1 : Blocks.BSt), step1 lc s i = some s1 → ∀ t xa tb yb,
      Blocks.getV s.vals a = some (.sc t xa) → Blocks.getV s.vals b = some (.sc tb yb) →
      s1 = { s with vals := Blocks.setV s.vals d (.sc t (f t xa yb)) })
    (hlow : ∀ tag t, F.tys.get? a.id = some t → (lower F tag i).lines =
      ((runLM (r a) F tag (runLM (shiftAmt t b) F tag {}).2).2.lines).push
        (.ins none (.bin bo none (tyf t) (regOf t d) (.reg (runLM (r a) F tag (runLM (shiftAmt t b) F tag {}).2).1)
          (runLM (shiftAmt t b) F tag {}).1)))
    (hsem : ∀ t xa yb A, IntT t → xa &&& widthMask t = xa → A.toNat = yb.toNat % t.width →
      PtxRun.bin bo (tyf t) xa A = .ok (enc t (f t xa yb)))
    (hnorm : ∀ t xa yb, IntT t → f t xa yb &&& widthMask t = f t xa yb)
    (hmo : magOk F d i = true) (hrule : ∀ t xa yb k, IntT t → F.tys.get? a.id = some t → Bounded F a xa →
      (∀ l, F.lits.get? b.id = some l → litRaw l = yb) →
      magRule F.origin F.mag F.lits F.tys i = some k → (Rel F d (f t xa yb)).toNat < 2 ^ k) :
    InstOk F lc c start0 i := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  obtain ⟨ta, xa, hva, hta⟩ := hS.pres a (huses a hua)
  obtain ⟨hit, htd⟩ := hT ta hta
  obtain ⟨tb, yb, hvb, htb⟩ := hS.pres b (huses b hub)
  have hne8 : ta ≠ .i8 := by rcases hit with rfl | rfl | rfl <;> decide
  obtain rfl := hstep s s1 hst ta xa tb yb hva hvb
  have hxa := hS.normd a ta xa (huses a hua) hva
  let sm := (runLM (shiftAmt ta b) F tag {}).2
  obtain ⟨Lb, hLb, hn1, regs1, A, hrun1, hopA, hA, hkeep1⟩ :=
    shiftAmt_sound F tag {} s.vals c ts b ta tb yb hit (hTb tb htb) D hS (huses b hub) hvb htb hwb
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨La, hLa, hn2, hnf2, hns2, regs2, hrun2, hreg2, hkeep2⟩ :=
    r_sound F tag sm s.vals c { ts with regs := regs1 } a ta xa D hS1 (huses a hua) hva hta hwa
  let core : PtxIr.Instr := .bin bo none (tyf ta) (regOf ta d) (.reg (runLM (r a) F tag sm).1)
    (runLM (shiftAmt ta b) F tag {}).1
  have hlist : (lower F tag i).lines = ((Lb ++ La ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow tag ta hta, hLa]
    show (sm.lines ++ _).push _ = _
    rw [hLb]; simp [core]; rfl
  have hins : instrsOf (lower F tag i).lines = Lb ++ La ++ [core] := by rw [hlist, instrsOf_map]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hX : OpVal c { ts with regs := regs2 } (.reg (runLM (r a) F tag sm).1) xa := by
    have := opVal_reg c { ts with regs := regs2 } _ _ hns2 hreg2
    simpa [enc, hne8, hxa] using this
  have hY : OpVal c { ts with regs := regs2 } (runLM (shiftAmt ta b) F tag {}).1 A := hopA _ (fun R hR => hkeep2 R hR)
  have he : (exec c core).run { ts with regs := regs2 } =
      .ok (.step, { ts with regs := regs2.insert (regOf ta d) (enc ta (f ta xa yb)) }) :=
    exec_bin c _ bo (tyf ta) _ _ _ xa A _ hX hY (hsem ta xa yb A hit hxa hA)
  refine ⟨{ ts with regs := regs2.insert (regOf ta d) (enc ta (f ta xa yb)) }, ?_, ?_, hW⟩
  · rw [hins, steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun k hk => hrule ta xa yb k hit hta (hS.bound a ta xa (huses a hua) hva) (fun l hl => by
        rw [(hS.lit b tb yb l (huses b hub) hvb htb hl).2]
        exact enc_int tb (hTb tb htb) yb (hS.normd b tb yb (huses b hub) hvb)) hk) F D s.vals ts.regs _ i d ta (f ta xa yb) hd hS htd (hnorm ta xa yb hit)
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; exact absurd h8 hne8
    · intro h8; exact absurd h8 hne8
    · intro c' n hn
      have hne : PtxIr.Reg.val c' n ≠ regOf ta d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h), hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]

/-- The amount modulo the width, as the block semantics takes it. -/
theorem amt_eq (t : ClifTy) (ht : IntT t) (yb A : UInt64) (hA : A.toNat = yb.toNat % t.width) :
    A = yb % UInt64.ofNat t.width := by
  apply UInt64.toNat_inj.mp
  have hw : t.width < 2 ^ 64 := by rcases ht with rfl | rfl | rfl <;> decide
  rw [hA, UInt64.toNat_mod, UInt64.toNat_ofNat', Nat.mod_eq_of_lt hw]

theorem amt_lt (t : ClifTy) (ht : IntT t) (yb A : UInt64) (hA : A.toNat = yb.toNat % t.width) :
    (PtxRun.mask 32 A).toNat = A.toNat ∧ A.toNat < t.width := by
  have hw : 0 < t.width := by rcases ht with rfl | rfl | rfl <;> decide
  have hlt : A.toNat < t.width := by rw [hA]; exact Nat.mod_lt _ hw
  refine ⟨?_, hlt⟩
  have h32 : t.width ≤ 2 ^ 32 := by rcases ht with rfl | rfl | rfl <;> decide
  rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp), toNat_wm .i32 (by simp)]
  exact Nat.mod_eq_of_lt (by simp [ClifTy.width]; omega)

theorem inst_shift (i : Inst) (d a b : Val) (h : i = .ishl d a b ∨ i = .ushr d a b ∨ i = .ishift d .sshr a b) :
    inst i = (do
      let t ← tyOf a
      let amt ← shiftAmt t b
      let (bo, ty) : PtxIr.BinOp × PtxIr.Ty := match i with
        | .ishl .. => (.shl, bTy t.width) | .ushr .. => (.shr, uTy t.width) | _ => (.shr, sTy t.width)
      emit (.bin bo none ty (regOf t d) (← o a) amt)) := by
  rcases h with rfl | rfl | rfl <;> simp only [inst]

theorem lower_shift (F : Facts) (tag : Nat) (i : Inst) (d a b : Val) (ta : ClifTy) (bo : PtxIr.BinOp)
    (ty : PtxIr.Ty) (h : i = .ishl d a b ∨ i = .ushr d a b ∨ i = .ishift d .sshr a b)
    (hbo : (match i with
        | .ishl .. => ((.shl, bTy ta.width) : PtxIr.BinOp × PtxIr.Ty) | .ushr .. => (.shr, uTy ta.width)
        | _ => (.shr, sTy ta.width)) = (bo, ty))
    (hta : F.tys.get? a.id = some ta) :
    (lower F tag i).lines =
      ((runLM (r a) F tag (runLM (shiftAmt ta b) F tag {}).2).2.lines).push
        (.ins none (.bin bo none ty (regOf ta d) (.reg (runLM (r a) F tag (runLM (shiftAmt ta b) F tag {}).2).1)
          (runLM (shiftAmt ta b) F tag {}).1)) := by
  have hT : isTerm i = false := by rcases h with rfl | rfl | rfl <;> rfl
  rw [lower_inst F tag i hT, inst_shift i d a b h, runLM_bind, runLM_tyOf F tag _ a ta hta]
  simp only [runLM_bind, runLM_pure, o, emit, emitL, runLM_modify, hbo]

/-- A logical shift right of a value cut to its width stays within it. -/
theorem shr_norm (t : ClifTy) (ht : IntT t) (z A : UInt64) (hz : z &&& widthMask t = z) :
    (z >>> A) &&& widthMask t = z >>> A := by
  apply UInt64.toNat_inj.mp
  rw [toNat_wm t ht, UInt64.toNat_shiftRight]
  apply Nat.mod_eq_of_lt
  have h1 : z.toNat < 2 ^ t.width := by
    rw [← hz, toNat_wm t ht]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  exact Nat.lt_of_le_of_lt (by rw [Nat.shiftRight_eq_div_pow]; exact Nat.div_le_self _ _) h1

open PtxRun in
/-- **A shift left** meets its obligation. -/
theorem ishl_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hTb : ∀ t, F.tys.get? b.id = some t → IntT t) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.ishl d a b) = true) :
    InstOk F lc c start0 (.ishl d a b) := by
  apply shift_ok (hmo := hmo) (hrule := fun t xa yb k hit hta hba hlb hk => ishl_rule F d a b t hit hta xa yb k hmo hba hlb hk) F lc c start0 _ d a b .shl (fun t => bTy t.width)
    (fun t x y => (x <<< (y % UInt64.ofNat t.width)) &&& widthMask t) rfl (by simp [Inst.uses])
    (by simp [Inst.uses]) hwa hwb hwd hT hTb hld
  · intro s s1 hst t xa tb yb hva hvb
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hvb, Blocks.viaOp, evalOp, Sem.shiftBin, Sem.get,
      norm] at hst
    by_cases hc : t.isInt = true ∧ tb.isInt = true
    · simp [hc] at hst; exact hst.symm
    · simp [hc] at hst
  · intro tag t ht; exact lower_shift F tag _ d a b t .shl _ (Or.inl rfl) rfl ht
  · intro t xa yb A hit hxa hA
    obtain ⟨hm, hlt⟩ := amt_lt t hit yb A hA
    have hne8 : t ≠ .i8 := by rcases hit with rfl | rfl | rfl <;> decide
    have hbf : (bTy t.width).isFloat = false := by rcases hit with rfl | rfl | rfl <;> rfl
    have hbw : (bTy t.width).width = t.width := by rcases hit with rfl | rfl | rfl <;> rfl
    simp only [PtxRun.bin, hbf, Bool.false_eq_true, if_false, PtxRun.binI, hbw, hm]
    rw [if_neg (by omega), mask_width t hit, amt_eq t hit yb A hA]
    simp [enc, hne8, UInt64.and_assoc]
  · intro t xa yb _; rw [UInt64.and_assoc, UInt64.and_self]

open PtxRun in
/-- **A logical shift right** meets its obligation. -/
theorem ushr_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hTb : ∀ t, F.tys.get? b.id = some t → IntT t) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.ushr d a b) = true) :
    InstOk F lc c start0 (.ushr d a b) := by
  apply shift_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a b .shr (fun t => uTy t.width)
    (fun t x y => ((x &&& widthMask t) >>> (y % UInt64.ofNat t.width)) &&& widthMask t) rfl (by simp [Inst.uses])
    (by simp [Inst.uses]) hwa hwb hwd hT hTb hld
  · intro s s1 hst t xa tb yb hva hvb
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hvb, Blocks.viaOp, evalOp, Sem.shiftBin, Sem.get,
      norm] at hst
    by_cases hc : t.isInt = true ∧ tb.isInt = true
    · simp [hc] at hst; exact hst.symm
    · simp [hc] at hst
  · intro tag t ht; exact lower_shift F tag _ d a b t .shr _ (Or.inr (Or.inl rfl)) rfl ht
  · intro t xa yb A hit hxa hA
    obtain ⟨hm, hlt⟩ := amt_lt t hit yb A hA
    have hne8 : t ≠ .i8 := by rcases hit with rfl | rfl | rfl <;> decide
    have huf : (uTy t.width).isFloat = false := by rcases hit with rfl | rfl | rfl <;> rfl
    have hus : (uTy t.width).isSigned = false := by rcases hit with rfl | rfl | rfl <;> rfl
    have huw : (uTy t.width).width = t.width := by rcases hit with rfl | rfl | rfl <;> rfl
    simp only [PtxRun.bin, huf, Bool.false_eq_true, if_false, PtxRun.binI, huw, hm, hus]
    rw [if_neg (by omega), mask_width t hit, amt_eq t hit yb A hA]
    simp only [enc, hne8, if_false, UInt64.and_assoc, UInt64.and_self]
    rw [shr_norm t hit _ _ (by rw [UInt64.and_assoc, UInt64.and_self])]
    rfl
  · intro t xa yb _; rw [UInt64.and_assoc, UInt64.and_self]

open PtxRun in
/-- **An arithmetic shift right** meets its obligation. -/
theorem sshr_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hTb : ∀ t, F.tys.get? b.id = some t → IntT t) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.ishift d .sshr a b) = true) :
    InstOk F lc c start0 (.ishift d .sshr a b) := by
  apply shift_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a b .shr (fun t => sTy t.width)
    (fun t x y => UInt64.ofNat ((signed t x >>> (y.toNat % t.width)).emod ((1 <<< 64 : Nat) : Int)).toNat
      &&& widthMask t) rfl (by simp [Inst.uses]) (by simp [Inst.uses]) hwa hwb hwd hT hTb hld
  · intro s s1 hst t xa tb yb hva hvb
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hvb, Blocks.viaOp, evalOp, Sem.shiftBin, Sem.get,
      ishiftOp, ofInt, norm] at hst
    by_cases hc : t.isInt = true ∧ tb.isInt = true
    · simp [hc] at hst; exact hst.symm
    · simp [hc] at hst
  · intro tag t ht; exact lower_shift F tag _ d a b t .shr _ (Or.inr (Or.inr rfl)) rfl ht
  · intro t xa yb A hit hxa hA
    obtain ⟨hm, hlt⟩ := amt_lt t hit yb A hA
    have hne8 : t ≠ .i8 := by rcases hit with rfl | rfl | rfl <;> decide
    have hsf : (sTy t.width).isFloat = false := by rcases hit with rfl | rfl | rfl <;> rfl
    have hss : (sTy t.width).isSigned = true := by rcases hit with rfl | rfl | rfl <;> rfl
    have hsw : (sTy t.width).width = t.width := by rcases hit with rfl | rfl | rfl <;> rfl
    simp only [PtxRun.bin, hsf, Bool.false_eq_true, if_false, PtxRun.binI, hsw, hm, hss, if_true]
    rw [Nat.min_eq_left (by omega), toInt_signed' t hit, ← hA, PtxRun.ofIntW, mask_width t hit]
    simp only [enc, hne8, if_false, UInt64.and_assoc, UInt64.and_self, UInt64.ofInt,
      show (1 <<< 64 : Nat) = 2 ^ 64 from rfl]
    rfl
  · intro t xa yb _; rw [UInt64.and_assoc, UInt64.and_self]

open PtxRun in
/-- **An unsigned quotient** of values not windowed meets its obligation: the
    block semantics divides only by a divisor not zero, and so does PTX. -/
theorem udiv_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.udiv d a b) = true) :
    InstOk F lc c start0 (.udiv d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply bin2_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 (.udiv d a b) d a b .div none (fun t => uTy t.width)
    (fun t x y => (x / y) &&& widthMask t) IntT
    (fun t h => by rcases h with rfl | rfl | rfl <;> simp [Scalar])
    (fun _ _ xb => xb ≠ 0) rfl (by simp [Inst.uses]) (by simp [Inst.uses]) hwa hwb hwd hT hld
  · intro s s1 hst _ t xa _ hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin,
        Sem.get] at hst
    | sc tb xb =>
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin, Sem.get] at hst
    by_cases hc : t = tb ∧ t.isInt = true
    · obtain ⟨rfl, hci⟩ := hc
      by_cases h0 : xb = 0
      · simp [hci, h0] at hst
      · simp [hci, h0] at hst
        subst hst
        exact ⟨xb, rfl, h0, by simp [norm]⟩
    · simp [hc] at hst
  · intro tag t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify]
    all_goals rfl
  · intro t ht h8; rcases ht with rfl | rfl | rfl <;> cases h8
  · intro t _ _ ht h8; rcases ht with rfl | rfl | rfl <;> cases h8
  · intro t xa xb y hti h0 hxa hxb _ hy c' ts' xo yo dr hxo hyo
    have hne8 : t ≠ .i8 := by rcases hti with rfl | rfl | rfl <;> decide
    have huw : (uTy t.width).width = t.width := by rcases hti with rfl | rfl | rfl <;> rfl
    have huf : (uTy t.width).isFloat = false := by rcases hti with rfl | rfl | rfl <;> rfl
    have hus : (uTy t.width).isSigned = false := by rcases hti with rfl | rfl | rfl <;> rfl
    have hbw : (bits t).width = t.width := by rcases hti with rfl | rfl | rfl <;> rfl
    simp only [enc, hne8, if_false, hxb, hxa] at hy hxo ⊢
    rw [hbw] at hy
    have hmy : PtxRun.mask (uTy t.width).width y = xb := by rw [huw, hy]
    have hty : PtxRun.toInt (uTy t.width) y = xb.toNat := by
      rw [← toInt_mask, hmy, toInt_u t hti xb hxb]
    have hv : PtxRun.bin .div (uTy t.width) xa y = .ok ((xa / xb) &&& widthMask t) := by
      simp only [PtxRun.bin, huf, Bool.false_eq_true, if_false, PtxRun.binI, hmy, hus, hty,
        toInt_u t hti xa hxa, beq_iff_eq, h0]
      rw [PtxRun.ofIntW, huw, mask_width t hti, ← Int.natCast_ediv, ← UInt64.toNat_div, ofInt_toNat]
      rfl
    rw [exec_bin c' ts' .div (uTy t.width) dr xo yo _ _ _ hxo hyo hv, UInt64.and_assoc, UInt64.and_self]
  · intro t xa xb _ _ _; rw [UInt64.and_assoc, UInt64.and_self]

open PtxRun in
/-- **An unsigned minimum** of values not windowed meets its obligation: one `min` at the
    unsigned type. -/
theorem umin_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.ibin d .umin a b) = true) :
    InstOk F lc c start0 (.ibin d .umin a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply bin2_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0
    (.ibin d .umin a b) d a b .min none (fun t => uTy t.width)
    (fun t x y => UInt64.ofNat (min (x &&& widthMask t).toNat (y &&& widthMask t).toNat) &&& widthMask t) IntT
    (fun t h => by rcases h with rfl | rfl | rfl <;> simp [Scalar])
    (fun _ _ _ => True) rfl (by simp [Inst.uses]) (by simp [Inst.uses]) hwa hwb hwd hT hld
  · intro s s1 hst _ t xa _ hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin,
        Sem.get] at hst
    | sc tb xb =>
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin, Sem.get, ibinOp,
      norm] at hst
    by_cases hc : t = tb ∧ t.isInt = true
    · obtain ⟨rfl, hci⟩ := hc
      simp [hci] at hst
      subst hst
      exact ⟨xb, rfl, trivial, rfl⟩
    · simp [hc] at hst
  · intro tag t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify]
    all_goals rfl
  · intro t ht h8; rcases ht with rfl | rfl | rfl <;> cases h8
  · intro t _ _ ht h8; rcases ht with rfl | rfl | rfl <;> cases h8
  · intro t xa xb y hti _ hxa hxb _ hy c' ts' xo yo dr hxo hyo
    have hne8 : t ≠ .i8 := by rcases hti with rfl | rfl | rfl <;> decide
    have huw : (uTy t.width).width = t.width := by rcases hti with rfl | rfl | rfl <;> rfl
    have huf : (uTy t.width).isFloat = false := by rcases hti with rfl | rfl | rfl <;> rfl
    have hbw : (bits t).width = t.width := by rcases hti with rfl | rfl | rfl <;> rfl
    simp only [enc, hne8, if_false, hxb, hxa] at hy hxo ⊢
    rw [hbw] at hy
    have hmy : PtxRun.mask (uTy t.width).width y = xb := by rw [huw, hy]
    have hty : PtxRun.toInt (uTy t.width) y = xb.toNat := by
      rw [← toInt_mask, hmy, toInt_u t hti xb hxb]
    have hmx : PtxRun.mask (uTy t.width).width xa = xa := by rw [huw, mask_width t hti, hxa]
    have hv : PtxRun.bin .min (uTy t.width) xa y =
        .ok (UInt64.ofNat (min xa.toNat xb.toNat) &&& widthMask t) := by
      simp only [PtxRun.bin, huf, Bool.false_eq_true, if_false, PtxRun.binI, hty, toInt_u t hti xa hxa, hmx,
        hmy, Nat.min_def]
      split <;> split <;> (try (exfalso; omega)) <;>
        first
          | (simp only [UInt64.ofNat_toNat, hxa, hxb]; rfl)
          | (have he : xa = xb := UInt64.toNat_inj.mp (by omega)
             subst he; simp only [UInt64.ofNat_toNat, hxa]; rfl)
    rw [exec_bin c' ts' .min (uTy t.width) dr xo yo _ _ _ hxo hyo hv, UInt64.and_assoc, UInt64.and_self]
  · intro t xa xb _ _ _; rw [UInt64.and_assoc, UInt64.and_self]

open PtxRun in
/-- **An unsigned maximum** of values not windowed meets its obligation: one `max` at the
    unsigned type. -/
theorem umax_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.ibin d .umax a b) = true) :
    InstOk F lc c start0 (.ibin d .umax a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply bin2_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0
    (.ibin d .umax a b) d a b .max none (fun t => uTy t.width)
    (fun t x y => UInt64.ofNat (max (x &&& widthMask t).toNat (y &&& widthMask t).toNat) &&& widthMask t) IntT
    (fun t h => by rcases h with rfl | rfl | rfl <;> simp [Scalar])
    (fun _ _ _ => True) rfl (by simp [Inst.uses]) (by simp [Inst.uses]) hwa hwb hwd hT hld
  · intro s s1 hst _ t xa _ hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin,
        Sem.get] at hst
    | sc tb xb =>
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.bin, Sem.get, ibinOp,
      norm] at hst
    by_cases hc : t = tb ∧ t.isInt = true
    · obtain ⟨rfl, hci⟩ := hc
      simp [hci] at hst
      subst hst
      exact ⟨xb, rfl, trivial, rfl⟩
    · simp [hc] at hst
  · intro tag t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify]
    all_goals rfl
  · intro t ht h8; rcases ht with rfl | rfl | rfl <;> cases h8
  · intro t _ _ ht h8; rcases ht with rfl | rfl | rfl <;> cases h8
  · intro t xa xb y hti _ hxa hxb _ hy c' ts' xo yo dr hxo hyo
    have hne8 : t ≠ .i8 := by rcases hti with rfl | rfl | rfl <;> decide
    have huw : (uTy t.width).width = t.width := by rcases hti with rfl | rfl | rfl <;> rfl
    have huf : (uTy t.width).isFloat = false := by rcases hti with rfl | rfl | rfl <;> rfl
    have hbw : (bits t).width = t.width := by rcases hti with rfl | rfl | rfl <;> rfl
    simp only [enc, hne8, if_false, hxb, hxa] at hy hxo ⊢
    rw [hbw] at hy
    have hmy : PtxRun.mask (uTy t.width).width y = xb := by rw [huw, hy]
    have hty : PtxRun.toInt (uTy t.width) y = xb.toNat := by
      rw [← toInt_mask, hmy, toInt_u t hti xb hxb]
    have hmx : PtxRun.mask (uTy t.width).width xa = xa := by rw [huw, mask_width t hti, hxa]
    have hv : PtxRun.bin .max (uTy t.width) xa y =
        .ok (UInt64.ofNat (max xa.toNat xb.toNat) &&& widthMask t) := by
      simp only [PtxRun.bin, huf, Bool.false_eq_true, if_false, PtxRun.binI, hty, toInt_u t hti xa hxa, hmx,
        hmy, Nat.max_def]
      split <;> split <;> (try (exfalso; omega)) <;>
        first
          | (simp only [UInt64.ofNat_toNat, hxa, hxb]; rfl)
          | (have he : xa = xb := UInt64.toNat_inj.mp (by omega)
             subst he; simp only [UInt64.ofNat_toNat, hxa]; rfl)
    rw [exec_bin c' ts' .max (uTy t.width) dr xo yo _ _ _ hxo hyo hv, UInt64.and_assoc, UInt64.and_self]
  · intro t xa xb _ _ _; rw [UInt64.and_assoc, UInt64.and_self]

open PtxRun in
/-- **A narrowing to 16 bits** of a value not windowed meets its obligation:
    one `cvt.u16`. -/
theorem reduce16_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false) (hT : ∀ t, F.tys.get? a.id = some t → t = .i32 ∨ t = .i64)
    (htd : F.tys.get? d.id = some .i16) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.iext d .reduce .i16 a) = true) :
    InstOk F lc c start0 (.iext d .reduce .i16 a) := by
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a
    (fun t dr xo => .cvt none .u16 (uTy t.width) dr xo) (fun _ => .i16) (fun _ x => x &&& widthMask .i16)
    (fun t => t = .i32 ∨ t = .i64) (fun _ _ h => nomatch h) rfl (by simp [Inst.uses]) hwa hwd (fun _ _ => htd) hld
  · intro s s1 hst _ t xa hva hta
    have ht := hT t hta
    refine ⟨ht, ?_⟩
    rcases ht with rfl | rfl <;>
      simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.un, Sem.get, norm,
        IExt.admits, ClifTy.isInt, ClifTy.width] at hst <;> exact hst.symm
  · intro tag t ht
    have ht' : F.tys[a.id]? = some t := by rw [← Std.HashMap.get?_eq_getElem?]; exact ht
    simp [lower_eq, isTerm, inst, emit, emitL, o, facts?, tyOf, ht']
  · intro t xa ht hn _ c' ts' xo dr hxo
    have hit : IntT t := by rcases ht with rfl | rfl <;> simp [IntT]
    have hne8 : t ≠ .i8 := by rcases ht with rfl | rfl <;> decide
    rw [exec_cvt c' ts' none .u16 _ dr xo _ _ hxo (cvt_int none .u16 _ _ rfl (by rcases ht with rfl | rfl <;> rfl))]
    simp only [enc, hne8, show ClifTy.i16 ≠ .i8 by decide, if_false, hn]
    rw [toInt_u t hit xa hn, PtxRun.ofIntW, show PtxIr.Ty.width .u16 = ClifTy.i16.width from rfl,
      mask_width .i16 (by simp [IntT]), ofInt_toNat, UInt64.and_assoc, UInt64.and_self]
  · intro t xa _ _; rw [UInt64.and_assoc, UInt64.and_self]

/-! ## Sums in the shared window -/

/-- A windowed address plus a value is windowed at the 32-bit sum of the
    offset and the value. -/
theorem winOff_add (x y off y' : UInt64) (h : WinOff x off) (hy : y'.toNat % 2 ^ 32 = y.toNat % 2 ^ 32) :
    WinOff (x + y) (PtxRun.mask 32 (off + y')) := by
  unfold WinOff at h ⊢
  rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp), toNat_wm .i32 (by simp)]
  simp only [ClifTy.width]
  rw [UInt64.toNat_add, ← h]
  have := x.toNat_lt; have := y.toNat_lt; have := (regionBase .arena).toNat_lt
  simp only [UInt64.toNat_sub, UInt64.toNat_add]
  omega

theorem runLM_isWin (F : Facts) (tag : Nat) (s : Out) (v : Val) : runLM (isWin v) F tag s = (F.win v, s) := by
  simp [isWin, facts?]

theorem runLM_rw_win (F : Facts) (tag : Nat) (s : Out) (v : Val) (h : F.win v = true) :
    runLM (rw v) F tag s = (winReg v, s) := by
  simp [rw, isWin, facts?, h]

theorem runLM_isWin_eq (F : Facts) (tag : Nat) (s : Out) (v : Val) (w : Bool) (h : F.win v = w) :
    runLM (isWin v) F tag s = (w, s) := by
  simp [isWin, facts?, h]

/-- A sum with a windowed address lowers by `iaddWin`, the address first. -/
theorem lower_iadd_win (F : Facts) (tag : Nat) (d a b sa other : Val)
    (h : (F.win a = true ∧ sa = a ∧ other = b) ∨ (F.win a = false ∧ F.win b = true ∧ sa = b ∧ other = a)) :
    lower F tag (.iadd d a b) = (runLM (iaddWin d sa other) F tag {}).2 := by
  rw [lower_inst F tag _ rfl]
  rcases h with ⟨ha, rfl, rfl⟩ | ⟨ha, hb, rfl, rfl⟩
  · simp only [inst, runLM_bind, runLM_isWin_eq F tag _ _ _ ha, if_true]
  · simp only [inst, runLM_bind, runLM_isWin_eq F tag _ _ _ ha, runLM_isWin_eq F tag _ _ _ hb,
      Bool.false_eq_true, if_false, if_true]

/-- A windowed sum with a constant is one 32-bit add. -/
theorem runLM_iaddWin_lit (F : Facts) (tag : Nat) (d sa other : Val) (l : PtxIr.Opnd)
    (hsw : F.win sa = true) (hl : F.lits.get? other.id = some l) :
    (runLM (iaddWin d sa other) F tag {}).2.lines = #[.ins none (.bin .add none .u32 (winReg d) (.reg (winReg sa)) l)] := by
  have hl' : F.lits[other.id]? = some l := by rw [← Std.HashMap.get?_eq_getElem?]; exact hl
  simp [iaddWin, facts?, hl', rw, isWin, hsw, emit, emitL]

/-- A windowed sum with a 64-bit value: the value's lines, its low 32 bits,
    then one 32-bit add. -/
theorem runLM_iaddWin_reg (F : Facts) (tag : Nat) (d sa other : Val)
    (hsw : F.win sa = true) (hl : F.lits.get? other.id = none) (hto : F.tys.get? other.id = some .i64) :
    (runLM (iaddWin d sa other) F tag {}).2.lines =
      (((runLM (r other) F tag { n := 1, regs := #[(.own "so" tag 0, .i32)] }).2.lines).push
        (.ins none (.cvt none .u32 .u64 (.own "so" tag 0)
          (.reg (runLM (r other) F tag { n := 1, regs := #[(.own "so" tag 0, .i32)] }).1)))).push
        (.ins none (.bin .add none .u32 (winReg d) (.reg (winReg sa)) (.reg (.own "so" tag 0)))) := by
  have hl' : F.lits[other.id]? = none := by rw [← Std.HashMap.get?_eq_getElem?]; exact hl
  have hto' : F.tys[other.id]? = some .i64 := by rw [← Std.HashMap.get?_eq_getElem?]; exact hto
  simp [iaddWin, facts?, hl', tyOf, hto', own, o, rw, isWin, hsw, emit, emitL]

theorem mask32_mod (z : UInt64) : (PtxRun.mask 32 z).toNat % 2 ^ 32 = z.toNat % 2 ^ 32 := by
  rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp), toNat_wm .i32 (by simp)]
  simp [ClifTy.width]

open PtxRun in
/-- **A windowed sum's lines**: from registers holding the block's values,
    `iaddW sa other` leaves `d`'s window register at the 32-bit sum of `sa`'s
    offset and `other`, changing no other value's register. -/
theorem iaddWin_run (F : Facts) (tag : Nat) (c : PtxRun.Ctx) (ts : PtxRun.TS) (vs : Blocks.Vals) (D : Nat → Prop)
    (d sa other : Val) (xs xo : UInt64) (hS : Sim F D vs ts.regs) (hDs : D sa.id) (hDo : D other.id)
    (hvs : Blocks.getV vs sa = some (.sc .i64 xs)) (hvo : Blocks.getV vs other = some (.sc .i64 xo))
    (hws : Win F sa) (hwo : ¬ Win F other) (hto : F.tys.get? other.id = some .i64) :
    Straight (runLM (iaddWin d sa other) F tag {}).2.lines ∧
    ∃ regs', (steps c (instrsOf (runLM (iaddWin d sa other) F tag {}).2.lines)).run ts = .ok ((), { ts with regs := regs' }) ∧
      (∃ off', regs'.get? (winReg d) = some off' ∧ off'.toNat < 2 ^ 32 ∧ WinOff (xs + xo) off') ∧
      ∀ c' n, n ≠ d.id → regs'.get? (.val c' n) = ts.regs.get? (.val c' n) := by
  have hsw : F.win sa = true := (win_iff F sa).mpr hws
  obtain ⟨off, hoff, hlt, hwo'⟩ := hS.win sa .i64 xs hDs hvs hws
  have hxo := hS.normd other .i64 xo hDo hvo
  rw [and_wide64] at hxo
  have hS0 : OpVal c ts (.reg (winReg sa)) off := opVal_reg c _ _ _ (not_special_val _ _) hoff
  cases hl : F.lits.get? other.id with
  | some l =>
      obtain ⟨hil, hraw⟩ := hS.lit other .i64 xo l hDo hvo hto hl
      simp only [enc, show ClifTy.i64 ≠ .i8 by decide, if_false, and_wide64] at hraw
      rw [runLM_iaddWin_lit F tag d sa other l hsw hl]
      let core : PtxIr.Instr := .bin .add none .u32 (winReg d) (.reg (winReg sa)) l
      have hm : #[PtxIr.Line.ins none core] = (([core] : List PtxIr.Instr).map (PtxIr.Line.ins none)).toArray := rfl
      refine ⟨by rw [hm]; exact straight_map _, ts.regs.insert (winReg d) (PtxRun.mask 32 (off + xo)), ?_, ?_, ?_⟩
      · rw [hm, instrsOf_map]
        have he := exec_bin (rnd := none) c ts .add .u32 (winReg d) _ l off xo (PtxRun.mask 32 (off + xo)) hS0
          (by rw [← hraw]; exact opVal_lit c ts l hil) rfl
        simp only [StateT.run] at he
        simp [core, steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
      · exact ⟨_, by simp [Std.HashMap.get?_eq_getElem?], PtxRun.mask32_toNat_lt _, winOff_add xs xo off xo hwo' rfl⟩
      · intro c' n hn
        have hne : PtxIr.Reg.val c' n ≠ winReg d := by simp [winReg]; omega
        rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
          ← Std.HashMap.get?_eq_getElem?]
  | none =>
      rw [runLM_iaddWin_reg F tag d sa other hsw hl hto]
      let so : PtxIr.Reg := .own "so" tag 0
      let s1 : Out := { n := 1, regs := #[(so, .i32)] }
      obtain ⟨Lo, hLo, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
        r_sound F tag s1 vs c ts other .i64 xo D hS hDo hvo hto hwo
      simp only [enc, show ClifTy.i64 ≠ .i8 by decide, if_false, and_wide64] at hreg1
      let cv : PtxIr.Instr := .cvt none .u32 .u64 so (.reg (runLM (r other) F tag s1).1)
      let ad : PtxIr.Instr := .bin .add none .u32 (winReg d) (.reg (winReg sa)) (.reg so)
      have hlist : (((runLM (r other) F tag s1).2.lines).push (.ins none cv)).push (.ins none ad) =
          ((Lo ++ [cv, ad]).map (PtxIr.Line.ins none)).toArray := by
        rw [hLo]; simp [s1]
      rw [hlist]
      refine ⟨straight_map _, ?_⟩
      have hcv : (exec c cv).run { ts with regs := regs1 } =
          .ok (.step, { ts with regs := regs1.insert so (PtxRun.mask 32 xo) }) := by
        rw [exec_cvt c _ none .u32 .u64 so _ xo _ (opVal_reg c { ts with regs := regs1 } _ _ hns1 hreg1)
          (cvt_int none .u32 .u64 xo rfl rfl)]
        simp only [PtxRun.ofIntW, PtxRun.toInt, PtxIr.Ty.isSigned, PtxIr.Ty.width, Bool.false_and,
          Bool.false_eq_true, if_false, mask64, ofInt_toNat]
      have hoff1 : (regs1.insert so (PtxRun.mask 32 xo)).get? (winReg sa) = some off := by
        rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simp [so, winReg]),
          ← Std.HashMap.get?_eq_getElem?, hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]
        exact hoff
      let off' := PtxRun.mask 32 (off + PtxRun.mask 32 xo)
      have had : (exec c ad).run { ts with regs := regs1.insert so (PtxRun.mask 32 xo) } =
          .ok (.step, { ts with regs := (regs1.insert so (PtxRun.mask 32 xo)).insert (winReg d) off' }) :=
        exec_bin c _ .add .u32 (winReg d) _ _ off _ off'
          (opVal_reg c _ _ _ (not_special_val _ _) hoff1)
          (opVal_reg c _ _ _ (not_special_own _ _ _) (by simp [Std.HashMap.get?_eq_getElem?])) rfl
      refine ⟨(regs1.insert so (PtxRun.mask 32 xo)).insert (winReg d) off', ?_, ?_, ?_⟩
      · rw [instrsOf_map, steps_append c _ _ ts _ hrun1]
        simp only [StateT.run] at hcv had
        simp [steps, StateT.run, hcv, had, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
      · exact ⟨off', by simp [Std.HashMap.get?_eq_getElem?], PtxRun.mask32_toNat_lt _,
          winOff_add xs xo off _ hwo' (mask32_mod xo)⟩
      · intro c' n hn
        have hne : PtxIr.Reg.val c' n ≠ winReg d := by simp [winReg]; omega
        rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
          Std.HashMap.getElem?_insert, if_neg (by simp [so]), ← Std.HashMap.get?_eq_getElem?]
        exact hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)

open PtxRun in
/-- **A sum with a shared address in the window** meets its obligation: its
    offset, summed in 32 bits, in the sum's window register. -/
theorem iaddw_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val)
    (hw : Win F a ∨ Win F b) (hnw : ¬ (Win F a ∧ Win F b))
    (hta : F.tys.get? a.id = some .i64) (htb : F.tys.get? b.id = some .i64) (htd : F.tys.get? d.id = some .i64)
    (hwd : F.win d = true) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.iadd d a b) = true) :
    InstOk F lc c start0 (.iadd d a b) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have hua : a ∈ Inst.uses (.iadd d a b) := by simp [Inst.uses]
  have hub : b ∈ Inst.uses (.iadd d a b) := by simp [Inst.uses]
  obtain ⟨ta, xa, hva, hta'⟩ := hS.pres a (huses a hua)
  obtain ⟨tb, xb, hvb, htb'⟩ := hS.pres b (huses b hub)
  rw [hta] at hta'; cases hta'
  rw [htb] at htb'; cases htb'
  have hev := evalInst_iadd s.world.mem s.vals d a b .i64 xa xb rfl hva hvb
  simp only [step1, hev, Option.map_some, Option.some.injEq] at hst
  subst hst
  -- the windowed operand first
  have key : ∃ sa other xs xo, lower F tag (.iadd d a b) = (runLM (iaddWin d sa other) F tag {}).2 ∧
      D sa.id ∧ D other.id ∧ Blocks.getV s.vals sa = some (.sc .i64 xs) ∧
      Blocks.getV s.vals other = some (.sc .i64 xo) ∧ Win F sa ∧ ¬ Win F other ∧
      F.tys.get? other.id = some .i64 ∧ xs + xo = xa + xb := by
    by_cases ha : F.win a = true
    · have hwa : Win F a := (win_iff F a).mp ha
      exact ⟨a, b, xa, xb, lower_iadd_win F tag d a b a b (Or.inl ⟨ha, rfl, rfl⟩), huses a hua, huses b hub,
        hva, hvb, hwa, fun h => hnw ⟨hwa, h⟩, htb, rfl⟩
    · have ha' : F.win a = false := by simpa using ha
      have hwb : Win F b := hw.resolve_left (fun h => ha ((win_iff F a).mpr h))
      exact ⟨b, a, xb, xa, lower_iadd_win F tag d a b b a (Or.inr ⟨ha', (win_iff F b).mpr hwb, rfl, rfl⟩),
        huses b hub, huses a hua, hvb, hva, hwb, fun h => ha ((win_iff F a).mpr h), hta, UInt64.add_comm _ _⟩
  obtain ⟨sa, other, xs, xo, hlow, hDs, hDo, hvs, hvo, hws, hwo, hto, hsum⟩ := key
  obtain ⟨hstr, regs', hrun, ⟨off', hoff', hlt', hwoff⟩, hkeep⟩ :=
    iaddWin_run F tag c ts s.vals D d sa other xs xo hS hDs hDo hvs hvo hws hwo hto
  rw [hlow]
  refine ⟨hstr, { ts with regs := regs' }, hrun, ?_, hW⟩
  apply sim_set (hmo := hmo) (hrule := fun k hk => iadd_rule F d a b .i64 xa xb k htd (hS.bound a .i64 xa (huses a hua) hva) (hS.bound b .i64 xb (huses b hub) hvb) hk) F D s.vals ts.regs _ _ d .i64 _ rfl hS htd (by rw [and_wide64, and_wide64])
  · intro _ hnw'; exact absurd ((win_iff F d).mp hwd) hnw'
  · intro l hl; rw [hld] at hl; cases hl
  · intro _; exact ⟨off', hoff', hlt', by rw [and_wide64, ← hsum]; exact hwoff⟩
  · intro h8; cases h8
  · intro h8; cases h8
  · exact hkeep


/-! ## Float sign -/

theorem inst_fneg (d a : Val) :
    inst (.fneg d a) = (do
      let t ← tyOf a
      emit (.bin .xor none (bTy t.width) (regOf t d) (← o a) (.imm (2 ^ (t.width - 1))))) := by simp only [inst]

open PtxRun in
/-- **A float's sign flipped** meets its obligation: the exclusive or of its
    bits with the sign bit, which keeps a NaN's payload as the block semantics
    does. -/
theorem fneg_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false)
    (htd : ∀ t, F.tys.get? a.id = some t → F.tys.get? d.id = some t) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fneg d a) = true) :
    InstOk F lc c start0 (.fneg d a) := by
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a
    (fun t dr xo => .bin .xor none (bTy t.width) dr xo (.imm (2 ^ (t.width - 1)))) (fun t => t)
    (fun t x => x ^^^ fsignMask t) (fun t => t = .f32 ∨ t = .f64)
    (fun t h => by rcases h with rfl | rfl <;> decide) rfl (by simp [Inst.uses]) hwa hwd htd hld
  · intro s s1 hst _ t xa hva _
    cases t <;> simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.get] at hst <;>
      exact ⟨by simp, by rw [← hst]; rfl⟩
  · intro tag t ht
    rw [lower_inst F tag _ rfl, inst_fneg, runLM_bind, runLM_tyOf F tag _ a t ht]
    simp only [runLM_bind, runLM_pure, o, emit, emitL, runLM_modify]
  · intro t xa ht hn _ c' ts' xo dr hxo
    rcases ht with rfl | rfl
    · rw [exec_bin c' ts' .xor _ dr xo _ _ _ ((xa ^^^ 0x80000000) &&& widthMask .f32) hxo
        (opVal_lit c' ts' (.imm _) trivial)]
      · simp [enc, fsignMask]
      · simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hn]
        show PtxRun.bin .xor .b32 xa (UInt64.ofInt (2 ^ 31)) = _
        simp only [PtxRun.bin, PtxIr.Ty.isFloat, Bool.false_eq_true, if_false, PtxRun.binI, PtxIr.Ty.width]
        rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp)]
        rfl
    · rw [exec_bin c' ts' .xor _ dr xo _ _ _ (xa ^^^ 0x8000000000000000) hxo (opVal_lit c' ts' (.imm _) trivial)]
      · simp only [enc, show ClifTy.f64 ≠ .i8 by decide, if_false, fsignMask, hn,
          show widthMask .f64 = widthMask .i64 from rfl, and_wide64]; rfl
      · simp only [enc, show ClifTy.f64 ≠ .i8 by decide, if_false, hn, show widthMask .f64 = widthMask .i64 from rfl,
          and_wide64]
        rfl
  · intro t xa ht hn
    rcases ht with rfl | rfl
    · apply UInt64.toNat_inj.mp
      rw [show widthMask .f32 = ((2 ^ 32 - 1 : Nat).toUInt64) from rfl, toNat_and_mask _ _ (by decide : 32 ≤ 64),
        UInt64.toNat_xor, Nat.xor_mod_two_pow]
      have := congrArg UInt64.toNat hn
      rw [show widthMask .f32 = ((2 ^ 32 - 1 : Nat).toUInt64) from rfl, toNat_and_mask _ _ (by decide : 32 ≤ 64)] at this
      rw [this]; rfl
    · rw [show widthMask .f64 = widthMask .i64 from rfl, and_wide64]

theorem inst_fabs (d a : Val) :
    inst (.fun1 d .fabs a) = (do
      let t ← tyOf a
      emit (.bin .and none (bTy t.width) (regOf t d) (← o a) (.imm (2 ^ (t.width - 1) - 1)))) := by
  simp only [inst]

open PtxRun in
/-- **An f32's absolute value** meets its obligation: the and of its bits with
    all but the sign bit, which keeps a NaN's payload as the block semantics
    does. -/
theorem fabs_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false)
    (hta : F.tys.get? a.id = some .f32) (htd : F.tys.get? d.id = some .f32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fun1 d .fabs a) = true) :
    InstOk F lc c start0 (.fun1 d .fabs a) := by
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a
    (fun t dr xo => .bin .and none (bTy t.width) dr xo (.imm (2 ^ (t.width - 1) - 1))) (fun _ => .f32)
    (fun _ x => x &&& ~~~fsignMask .f32 &&& widthMask .f32) (· = .f32)
    (fun _ _ h => by cases h) rfl (by simp [Inst.uses]) hwa hwd (fun t ht => by rw [hta] at ht; cases ht; exact htd) hld
  · intro s s1 hst _ t xa hva ht
    rw [hta] at ht; cases ht
    refine ⟨rfl, ?_⟩
    simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.get, mapFBits, funOp] at hst
    rw [← hst]
  · intro tag t ht
    rw [hta] at ht; cases ht
    rw [lower_inst F tag _ rfl, inst_fabs, runLM_bind, runLM_tyOf F tag _ a .f32 hta]
    simp only [runLM_bind, runLM_pure, o, emit, emitL, runLM_modify]
  · intro t xa ht hn _ c' ts' xo dr hxo
    subst ht
    have hk : ~~~fsignMask .f32 &&& widthMask .f32 = UInt64.ofInt (2 ^ 31 - 1) &&& widthMask .i32 := by decide
    rw [exec_bin c' ts' .and _ dr xo _ _ _ (xa &&& ~~~fsignMask .f32 &&& widthMask .f32) hxo
      (opVal_lit c' ts' (.imm _) trivial)]
    · simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, UInt64.and_assoc, UInt64.and_self]
    · simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hn]
      show PtxRun.bin .and .b32 xa (UInt64.ofInt (2 ^ 31 - 1)) = _
      simp only [PtxRun.bin, PtxIr.Ty.isFloat, Bool.false_eq_true, if_false, PtxRun.binI, PtxIr.Ty.width]
      rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp), UInt64.and_assoc, ← hk,
        ← UInt64.and_assoc]
      rfl
  · intro t xa ht _
    subst ht
    simp only [UInt64.and_assoc, UInt64.and_self]

/-! ## Flags made and compared -/

open PtxRun in
theorem exec_un (c : PtxRun.Ctx) (ts : PtxRun.TS) (o : PtxIr.UnOp) (rn : Option PtxIr.Rnd) (ty : PtxIr.Ty)
    (dr : PtxIr.Reg) (xo : PtxIr.Opnd) (x v : UInt64) (hx : OpVal c ts xo x) (hv : PtxRun.un o c.hw ty x = .ok v) :
    (exec c (.un o rn ty dr xo)).run ts = .ok (.step, { ts with regs := ts.regs.insert dr v }) := by
  simp only [OpVal] at hx
  simp only [exec, StateT.run, bind, StateT.bind] at hx ⊢
  simp [hx, Except.bind, lift, hv, setReg, pure, StateT.pure, Except.pure, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet]

/-- A flag constant: its byte, 0 or 1, is the constant modulo 256. -/
theorem flag_const (k : Int) (hk : k % 256 = 0 ∨ k % 256 = 1) :
    UInt64.ofNat (k.emod 18446744073709551616).toNat &&& widthMask .i8 = UInt64.ofInt (k % 256) := by
  apply UInt64.toNat_inj.mp
  rw [show widthMask .i8 = ((2 ^ 8 - 1 : Nat).toUInt64) from rfl, toNat_and_mask _ _ (by decide)]
  simp only [UInt64.toNat_ofNat', UInt64.ofInt]
  rw [show k.emod 18446744073709551616 = k % 18446744073709551616 from rfl]
  simp only [Nat.reducePow]
  omega

open PtxRun in
/-- **A flag constant** meets its obligation: the constant into a 32-bit
    register of the instruction's own, then whether it is not zero. -/
theorem iconst8_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d : Val) (k : Int)
    (hk : k % 256 = 0 ∨ k % 256 = 1) (htd : F.tys.get? d.id = some .i8) (hld : F.lits.get? d.id = none)
    (hwd : F.win d = false) (hng : F.negated d = false)
    (hmo : magOk F d (.iconst d .i8 k) = true) :
    InstOk F lc c start0 (.iconst d .i8 k) := by
  intro tag D s s1 ts hst _ hS hW _ _hnan
  let kc : PtxIr.Reg := .own "c" tag 0
  let i1 : PtxIr.Instr := .mov .u32 kc (.imm (k % 256))
  let i2 : PtxIr.Instr := .setp .ne .u32 (regOf .i8 d) (.reg kc) (.imm 0)
  have hlow : (lower F tag (.iconst d .i8 k)).lines = (([i1, i2] : List PtxIr.Instr).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_inst F tag _ rfl]
    simp [inst, own, emit, emitL, i1, i2, kc]
  simp [step1, Blocks.evalInst, Blocks.viaOp, evalOp, ofInt, norm] at hst
  subst hst
  rw [flag_const k hk]
  let r : UInt64 := UInt64.ofInt (k % 256)
  have hr : r = 0 ∨ r = 1 := by rcases hk with h | h <;> simp only [r, h] <;> decide
  refine ⟨by rw [hlow]; exact straight_map _, ?_⟩
  have he1 : (exec c i1).run ts = .ok (.step, { ts with regs := ts.regs.insert kc r }) := by
    rw [exec_mov_imm]; rcases hr with h | h <;> simp only [r] at h ⊢ <;> rw [h] <;> rfl
  have hq : PtxRun.cmp .ne .u32 r 0 = .ok (r != 0) := by rcases hr with h | h <;> rw [h] <;> rfl
  have he2 : (exec c i2).run { ts with regs := ts.regs.insert kc r } =
      .ok (.step, { ts with regs := (ts.regs.insert kc r).insert (regOf .i8 d) (if r != 0 then 1 else 0) }) :=
    exec_setp c _ _ _ _ _ _ r 0 _ (opVal_reg c _ _ _ (not_special_own _ _ _) (by simp [Std.HashMap.get?_eq_getElem?]))
      (opVal_lit c _ (.imm 0) trivial) hq
  refine ⟨{ ts with regs := (ts.regs.insert kc r).insert (regOf .i8 d) (if r != 0 then 1 else 0) }, ?_, ?_, hW⟩
  · rw [hlow, instrsOf_map]
    simp only [StateT.run] at he1 he2
    simp [steps, StateT.run, he1, he2, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun k' hk' => by
        simp only [magRule] at hk'; cases hk'
        rw [rel_plain (not_shared_of hmo (fun _ _ _ h => by cases h))]
        have hrn : r.toNat = (k % (2 ^ ClifTy.i8.width : Int)).toNat := by
          rw [show (2 ^ ClifTy.i8.width : Int) = 256 from rfl]
          rcases hk with h | h <;> simp only [r, h] <;> rfl
        rw [hrn]; exact Nat.lt_log2_self) F D s.vals ts.regs _ _ d .i8 r rfl hS htd (by rcases hr with h | h <;> rw [h] <;> decide)
    · intro _ _
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert_self]
      rcases hr with h | h <;> rw [h] <;> rfl
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro _ h; rw [hng] at h; cases h
    · intro _; rcases hr with h | h <;> rw [h] <;> decide
    · intro c' n hn
      have hne : PtxIr.Reg.val c' n ≠ regOf .i8 d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        Std.HashMap.getElem?_insert, if_neg (by simp [kc]), ← Std.HashMap.get?_eq_getElem?]

theorem inst_icmp8 (F : Facts) (tag : Nat) (d a b : Val) (cc : ICmpCond) (ht : F.tys.get? a.id = some .i8) :
    runLM (inst (.icmp d cc a b)) F tag {} = runLM (do
      let x ← o a; let y ← o b
      match cc with
      | .ne => emit (.bin .xor none .pred (regOf .i8 d) x y)
      | .eq =>
          emit (.bin .xor none .pred (regOf .i8 d) x y)
          emit (.un .not none .pred (regOf .i8 d) (.reg (regOf .i8 d)))
      | _ => fail "flags compared other than for equality") F tag {} := by
  rw [inst]
  simp only [runLM_bind, runLM_tyOf F tag _ a .i8 ht, beq_self_eq_true, if_true]
  rfl

open PtxRun in
/-- **Flags compared for equality** meet their obligation: their exclusive
    or, negated for equality. -/
theorem icmp8_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d : Val) (cc : ICmpCond) (a b : Val)
    (hcc : cc = .eq ∨ cc = .ne) (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hta : F.tys.get? a.id = some .i8) (htd : F.tys.get? d.id = some .i8) (hld : F.lits.get? d.id = none)
    (hng : F.negated d = false)
    (hmo : magOk F d (.icmp d cc a b) = true) :
    InstOk F lc c start0 (.icmp d cc a b) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have hua : a ∈ Inst.uses (.icmp d cc a b) := by simp [Inst.uses]
  have hub : b ∈ Inst.uses (.icmp d cc a b) := by simp [Inst.uses]
  obtain ⟨ta, xa, hva, hta'⟩ := hS.pres a (huses a hua)
  rw [hta] at hta'; cases hta'
  obtain ⟨tb, xb, hvb, htb⟩ := hS.pres b (huses b hub)
  -- the block semantics' step
  have htb8 : tb = .i8 := by
    rcases hcc with rfl | rfl <;>
    · simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hvb, Blocks.viaOp, evalOp, Sem.get, zipIntCmp] at hst
      by_cases h : ClifTy.i8 = tb
      · exact h.symm
      · simp [h] at hst
  subst htb8
  have hfa := hS.flag a xa (huses a hua) hva
  have hfb := hS.flag b xb (huses b hub) hvb
  let rv : UInt64 := if cmpInt cc .i8 xa xb then 1 else 0
  have hs1 : s1 = { s with vals := Blocks.setV s.vals d (.sc .i8 rv) } := by
    simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hvb, Blocks.viaOp, evalOp, Sem.get, zipIntCmp,
      boolV, ClifTy.isInt] at hst
    exact hst.symm
  subst hs1
  -- the operands
  let sa := (runLM (r a) F tag {}).2
  obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag {} s.vals c ts a .i8 xa D hS (huses a hua) hva hta hwa
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lb, hLb, hn2, hnf2, hns2, regs2, hrun2, hreg2, hkeep2⟩ :=
    r_sound F tag sa s.vals c { ts with regs := regs1 } b .i8 xb D hS1 (huses b hub) hvb htb hwb
  rw [enc_flag xa hfa] at hreg1
  rw [enc_flag xb hfb] at hreg2
  let X : PtxIr.Opnd := .reg (runLM (r a) F tag {}).1
  let Y : PtxIr.Opnd := .reg (runLM (r b) F tag sa).1
  let P : PtxIr.Reg := regOf .i8 d
  let ix : PtxIr.Instr := .bin .xor none .pred P X Y
  let inot : PtxIr.Instr := .un .not none .pred P (.reg P)
  let core : List PtxIr.Instr := if cc = .eq then [ix, inot] else [ix]
  have hlow : (lower F tag (.icmp d cc a b)).lines = ((La ++ Lb ++ core).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_inst F tag _ rfl, inst_icmp8 F tag d a b cc hta]
    rcases hcc with rfl | rfl
    · simp only [runLM_bind, o, runLM_pure, emit, emitL, runLM_modify]
      rw [hLb]; show ((sa.lines ++ _).push _).push _ = _; rw [hLa]; simp [core, ix, inot, X, Y, P, sa]
    · simp only [runLM_bind, o, runLM_pure, emit, emitL, runLM_modify]
      rw [hLb]; show (sa.lines ++ _).push _ = _; rw [hLa]; simp [core, ix, X, Y, P, sa]
  refine ⟨by rw [hlow]; exact straight_map _, ?_⟩
  have hX : OpVal c { ts with regs := regs2 } X xa :=
    opVal_reg c _ _ _ hns1 (by rw [hkeep2 _ hnf1]; exact hreg1)
  have hY : OpVal c { ts with regs := regs2 } Y xb := opVal_reg c _ _ _ hns2 hreg2
  let v : UInt64 := PtxRun.mask 1 (xa ^^^ xb)
  have hv : v = if xa != xb then 1 else 0 := by
    rcases flag_cases xa hfa with rfl | rfl <;> rcases flag_cases xb hfb with rfl | rfl <;> rfl
  have hex : (exec c ix).run { ts with regs := regs2 } = .ok (.step, { ts with regs := regs2.insert P v }) :=
    exec_bin c _ .xor .pred P X Y xa xb v hX hY rfl
  have hrun : (steps c (La ++ Lb)).run ts = .ok ((), { ts with regs := regs2 }) := by
    rw [steps_append c _ _ ts _ hrun1]; exact hrun2
  have hkeepv : ∀ (c' : String) (n : Nat), n ≠ d.id → ∀ z w : UInt64,
      ((regs2.insert P z).insert P w).get? (.val c' n) = ts.regs.get? (.val c' n) ∧
      (regs2.insert P z).get? (.val c' n) = ts.regs.get? (.val c' n) := by
    intro c' n hn z w
    have hne : PtxIr.Reg.val c' n ≠ P := by simp [P, regOf]; omega
    constructor <;>
    · repeat rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne)]
      rw [← Std.HashMap.get?_eq_getElem?, hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h),
        hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]
      try simp [Std.HashMap.get?_eq_getElem?]
  rcases hcc with rfl | rfl
  · -- equality: the exclusive or, negated
    let w : UInt64 := PtxRun.mask 1 (~~~v)
    have hnot : (exec c inot).run { ts with regs := regs2.insert P v } =
        .ok (.step, { ts with regs := (regs2.insert P v).insert P w }) :=
      exec_un c _ .not none .pred P (.reg P) v w
        (opVal_reg c _ _ _ (not_special_val _ _) (by simp [Std.HashMap.get?_eq_getElem?])) rfl
    have hw : w = rv := by
      simp only [w, rv, hv, cmpInt]
      rcases flag_cases xa hfa with rfl | rfl <;> rcases flag_cases xb hfb with rfl | rfl <;> rfl
    refine ⟨{ ts with regs := (regs2.insert P v).insert P w }, ?_, ?_, hW⟩
    · rw [hlow, instrsOf_map, steps_append c _ _ ts _ hrun]
      simp only [StateT.run] at hex hnot
      simp [core, steps, StateT.run, hex, hnot, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
    · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ _ d .i8 rv rfl hS htd (by simp only [rv]; split <;> decide)
      · intro _ _
        rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert_self, hw]
        simp only [rv]; split <;> rfl
      · intro l hl; rw [hld] at hl; cases hl
      · intro hw'; rw [(win_iff F d).mpr hw'] at hwd; cases hwd
      · intro _ h; rw [hng] at h; cases h
      · intro _; simp only [rv]; split <;> decide
      · intro c' n hn; exact (hkeepv c' n hn v w).1
  · -- inequality: the exclusive or
    have hvr : v = rv := by
      simp only [rv, hv, cmpInt]
    refine ⟨{ ts with regs := regs2.insert P v }, ?_, ?_, hW⟩
    · rw [hlow, instrsOf_map, steps_append c _ _ ts _ hrun]
      simp only [StateT.run] at hex
      simp [core, steps, StateT.run, hex, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
    · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ _ d .i8 rv rfl hS htd (by simp only [rv]; split <;> decide)
      · intro _ _
        rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert_self, hvr]
        simp only [rv]; split <;> rfl
      · intro l hl; rw [hld] at hl; cases hl
      · intro hw'; rw [(win_iff F d).mpr hw'] at hwd; cases hwd
      · intro _ h; rw [hng] at h; cases h
      · intro _; simp only [rv]; split <;> decide
      · intro c' n hn; exact (hkeepv c' n hn v 0).2

/-! ## Loads through global and generic addresses -/

/-- The bytes `Mem.load` assembles, least significant first, fit their count. -/
theorem foldr_bytes_lt (f : Nat → UInt8) :
    ∀ l : List Nat, 8 * l.length ≤ 64 →
      (l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (f i).toUInt64) (0 : UInt64)).toNat < 2 ^ (8 * l.length)
  | [], _ => by simp
  | i :: l, h => by
      have hl : 8 * l.length ≤ 64 := by simp at h; omega
      have ih := foldr_bytes_lt f l hl
      simp only [List.foldr_cons, List.length_cons]
      rw [UInt64.toNat_or, UInt64.toNat_shiftLeft]
      have hb : (f i).toUInt64.toNat < 2 ^ 8 := by simpa using (f i).toNat_lt
      have h8 : (8 : UInt64).toNat % 64 = 8 := rfl
      rw [h8]
      have hk : 8 * (l.length + 1) ≤ 64 := by simpa using h
      have hsh : (l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (f i).toUInt64) (0 : UInt64)).toNat <<< 8 <
          2 ^ (8 * (l.length + 1)) := by
        rw [Nat.shiftLeft_eq, show 8 * (l.length + 1) = 8 * l.length + 8 by omega]
        rw [Nat.pow_add]; exact Nat.mul_lt_mul_of_pos_right ih (by decide)
      have h64 : 2 ^ (8 * (l.length + 1)) ≤ 2 ^ 64 := Nat.pow_le_pow_right (by decide) hk
      rw [Nat.mod_eq_of_lt (by omega)]
      exact Nat.or_lt_two_pow hsh (Nat.lt_of_lt_of_le hb (Nat.pow_le_pow_right (by decide) (by omega)))

/-- **A load's value fits its bytes.** -/
theorem load_lt (m : Mem) (a : UInt64) (n : Nat) (v : UInt64) (hn : n ≤ 8) (h : m.load a n = some v) :
    v.toNat < 2 ^ (8 * n) := by
  unfold Mem.load at h
  cases hd : decodeAddr a with
  | none => simp [hd] at h
  | some p =>
      simp [hd] at h
      obtain ⟨-, rfl⟩ := h
      have := foldr_bytes_lt (fun i => (m.region p.1).get! (p.2 + i)) (List.range n) (by simp; omega)
      simpa using this

/-! ## Addresses -/

/-- The space an access through `a`, not a shared address, takes. -/
def gspace (F : Facts) (a : Val) : PtxIr.Space :=
  match F.origin.get? a.id with
  | some (.global _) => .global
  | _ => .generic


/-- **Where an access's address reaches**: a shared one an offset into shared
    memory the Code's address is at, cut to the window's 32 bits; a global or
    generic one the Code's address itself. -/
def Reaches (sp : PtxIr.Space) (x xa : UInt64) : Prop :=
  (sp = .shared ∧ ∃ k, xa = regionBase .arena + UInt64.ofNat k ∧ k < 2 ^ 36 ∧ x.toNat % 2 ^ 32 = k % 2 ^ 32) ∨
  ((sp = .global ∨ sp = .generic) ∧ x = xa)

theorem reaches_sp {sp : PtxIr.Space} {x xa : UInt64} (h : Reaches sp x xa) : sp ≠ .param := by
  rcases h with ⟨rfl, -⟩ | ⟨rfl | rfl, -⟩ <;> intro h <;> cases h

/-- **An access's place is the Code's address** when an access at it into
    shared memory lies within 32 bits, as one the Code makes does. -/
theorem place_eq {sp : PtxIr.Space} {x xa : UInt64} (hr : Reaches sp x xa)
    (hfit : ∀ k, decodeAddr xa = some (.arena, k) → k < 2 ^ 32) : PtxRun.place sp x = xa := by
  rcases hr with ⟨rfl, k, rfl, hk, hx⟩ | ⟨hsp, rfl⟩
  · have hd := decodeAddr_base .arena k (by rw [regionSpan_toNat]; exact hk)
    have hk32 := hfit k hd
    simp only [PtxRun.place, addrOf]
    rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp [IntT]), toNat_wm .i32 (by simp),
      show ClifTy.i32.width = 32 from rfl, hx, Nat.mod_eq_of_lt hk32]
  · rcases hsp with rfl | rfl <;> rfl

/-- A load the Code makes into shared memory lies within it. -/
theorem fit_load {m : Mem} {xa : UInt64} {n : Nat} {v : UInt64} (hm : m.load xa n = some v)
    (hs : m.arena.size < 2 ^ 32) : ∀ k, decodeAddr xa = some (.arena, k) → k < 2 ^ 32 := by
  intro k hk
  unfold Mem.load at hm
  simp only [hk, bind, Option.bind, Mem.region] at hm
  split at hm
  · cases hm
  · rename_i h; simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, Bool.not_eq_true'] at h; omega

/-- A store the Code makes into shared memory lies within it. -/
theorem fit_store {m m' : Mem} {xa : UInt64} {n : Nat} {v : UInt64} (hm : m.store xa n v = some m')
    (hs : m.arena.size < 2 ^ 32) : ∀ k, decodeAddr xa = some (.arena, k) → k < 2 ^ 32 := by
  intro k hk
  unfold Mem.store at hm
  simp only [hk, bind, Option.bind, Mem.region] at hm
  split at hm
  · cases hm
  · rename_i h; simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, Bool.not_eq_true'] at h; omega

theorem runLM_addr_win (F : Facts) (tag : Nat) (s : Out) (a : Val) (h : F.win a = true) :
    runLM (addr a) F tag s = ((.shared, { base := .reg (winReg a) }), s) := by
  simp [addr, isWin, facts?, h, rw]

theorem runLM_addr_plain (F : Facts) (tag : Nat) (s : Out) (a : Val) (h : F.win a = false)
    (ho : F.origin.get? a.id ≠ some .shared) :
    runLM (addr a) F tag s = ((gspace F a, { base := .reg (runLM (r a) F tag s).1 }), (runLM (r a) F tag s).2) := by
  cases ho2 : F.origin.get? a.id with
  | none =>
      have ho' : F.origin[a.id]? = none := by rwa [← Std.HashMap.get?_eq_getElem?]
      simp [addr, isWin, facts?, h, ho', o, gspace, ho2]
  | some og =>
      have ho' : F.origin[a.id]? = some og := by rwa [← Std.HashMap.get?_eq_getElem?]
      cases og with
      | shared => exact absurd ho2 ho
      | global p => simp [addr, isWin, facts?, h, ho', o, gspace, ho2]

theorem runLM_addr_cvta (F : Facts) (tag : Nat) (s : Out) (a : Val) (h : F.win a = false)
    (ho : F.origin.get? a.id = some .shared) :
    runLM (addr a) F tag s =
      ((.shared, { base := .reg (.own "sw" tag s.n) }),
        { (runLM (r a) F tag { s with n := s.n + 1, regs := s.regs.push (.own "sw" tag s.n, .i64) }).2 with
          lines := (runLM (r a) F tag { s with n := s.n + 1, regs := s.regs.push (.own "sw" tag s.n, .i64) }).2.lines.push
            (.ins none (.cvta true .shared .u64 (.own "sw" tag s.n)
              (.reg (runLM (r a) F tag { s with n := s.n + 1, regs := s.regs.push (.own "sw" tag s.n, .i64) }).1))) }) := by
  have ho' : F.origin[a.id]? = some .shared := by rwa [← Std.HashMap.get?_eq_getElem?]
  simp [addr, isWin, facts?, h, ho', own, o, emit, emitL]

open PtxRun in
theorem exec_cvta_to (c : PtxRun.Ctx) (ts : PtxRun.TS) (w R : PtxIr.Reg) (x : UInt64) (k : Nat)
    (hR : OpVal c ts (.reg R) x) (hd : decodeAddr x = some (.arena, k)) :
    (exec c (.cvta true .shared .u64 w (.reg R))).run ts =
      .ok (.step, { ts with regs := ts.regs.insert w (UInt64.ofNat k) }) := by
  simp only [OpVal] at hR
  simp only [exec, StateT.run, bind, StateT.bind] at hR ⊢
  simp [hR, hd, Except.bind, setReg, pure, StateT.pure, Except.pure, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet]
  simp [StateT.bind, StateT.pure, StateT.modifyGet, Except.bind, Except.pure, bind, pure]

/-- A windowed value is a shared address. -/
theorem shared_of_win {F : Facts} {v : Val} (h : Win F v) : F.origin.get? v.id = some .shared := h.2

/-- A shared address is its region's base and its offset. -/
theorem shared_split (xa : UInt64) :
    xa = regionBase .arena + UInt64.ofNat (xa - regionBase .arena).toNat := by
  rw [UInt64.ofNat_toNat]; apply UInt64.toBitVec_inj.mp; simp; bv_omega

open PtxRun in
/-- **An access's address**: the lines `addr a` writes for an address in scope
    leave in a register of the access a value reaching the Code's address, as
    `Reaches` says: the window's offset, the offset `cvta` finds, or the
    address itself. -/
theorem addr_sound (F : Facts) (tag : Nat) (s : Out) (vs : Blocks.Vals) (c : PtxRun.Ctx) (ts : PtxRun.TS)
    (a : Val) (xa : UInt64) (D : Nat → Prop) (hS : Sim F D vs ts.regs) (hD : D a.id)
    (hv : Blocks.getV vs a = some (.sc .i64 xa)) (hta : F.tys.get? a.id = some .i64) :
    ∃ L : List PtxIr.Instr, (runLM (addr a) F tag s).2.lines = s.lines ++ (L.map (.ins none)).toArray ∧
      s.n ≤ (runLM (addr a) F tag s).2.n ∧
      ∃ R x, (runLM (addr a) F tag s).1.2 = { base := .reg R } ∧ (∀ s', R ≠ .special s') ∧
        ¬ Fresh tag (runLM (addr a) F tag s).2.n R ∧ Reaches (runLM (addr a) F tag s).1.1 x xa ∧
        ∃ regs', (steps c L).run ts = .ok ((), { ts with regs := regs' }) ∧ regs'.get? R = some x ∧
          ∀ R', ¬ Fresh tag s.n R' → regs'.get? R' = ts.regs.get? R' := by
  -- a shared address's offset, within one region
  have hoff : F.origin.get? a.id = some .shared → (xa - regionBase .arena).toNat < 2 ^ 36 := fun hsh => by
    have hb := hS.bound a .i64 xa hD hv
    have h1 := bnd_off hb hsh
    exact Nat.lt_of_lt_of_le h1 (Nat.pow_le_pow_right (by decide) (by simpa [sharedReach] using hb.2 hsh))
  by_cases hw : F.win a = true
  · rw [runLM_addr_win F tag s a hw]
    have hwa : Win F a := (win_iff F a).mp hw
    obtain ⟨off, hoffr, hlt, hwo⟩ := hS.win a .i64 xa hD hv hwa
    refine ⟨[], ?_, Nat.le_refl _, winReg a, off, rfl, not_special_val _ _, ?_, ?_, ts.regs, ?_, hoffr, fun _ _ => rfl⟩
    · simp
    · intro hf; obtain ⟨p, k, _, h⟩ := hf; cases h
    · refine Or.inl ⟨rfl, _, shared_split xa, hoff (shared_of_win hwa), ?_⟩
      unfold WinOff at hwo; rw [hwo, Nat.mod_eq_of_lt hlt]
    · simp [steps, StateT.run, pure, StateT.pure, Except.pure]
  have hwf : F.win a = false := by simpa using hw
  have hnw : ¬ Win F a := fun h => by rw [(win_iff F a).mpr h] at hwf; cases hwf
  by_cases hsh : F.origin.get? a.id = some .shared
  · rw [runLM_addr_cvta F tag s a hwf hsh]
    let s1 : Out := { s with n := s.n + 1, regs := s.regs.push (.own "sw" tag s.n, .i64) }
    obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
      r_sound F tag s1 vs c ts a .i64 xa D hS hD hv hta hnw
    simp only [enc, show ClifTy.i64 ≠ .i8 by decide, if_false, and_wide64] at hreg1
    have hk := hoff hsh
    have hd : decodeAddr xa = some (.arena, (xa - regionBase .arena).toNat) := by
      conv => lhs; rw [shared_split xa]
      exact decodeAddr_base .arena _ (by rw [regionSpan_toNat]; exact hk)
    let w : PtxIr.Reg := .own "sw" tag s.n
    let core : PtxIr.Instr := .cvta true .shared .u64 w (.reg (runLM (r a) F tag s1).1)
    refine ⟨La ++ [core], ?_, by have := hn1; simp [s1] at this ⊢; omega, w, UInt64.ofNat (xa - regionBase .arena).toNat,
      rfl, not_special_own _ _ _, fun hf => ?_,
      Or.inl ⟨rfl, _, shared_split xa, hk, ?_⟩, regs1.insert w (UInt64.ofNat (xa - regionBase .arena).toNat), ?_, ?_, ?_⟩
    · rw [hLa]; simp [s1, core, w]
    · obtain ⟨p, k, hk', h⟩ := hf
      simp only [w, PtxIr.Reg.own.injEq] at h
      obtain ⟨-, -, rfl⟩ := h
      have := hn1; simp only [s1] at this hk'; omega
    · rw [UInt64.toNat_ofNat_of_lt' (Nat.lt_of_lt_of_le hk (by decide))]
    · rw [steps_append c _ _ ts _ hrun1]
      have he := exec_cvta_to c { ts with regs := regs1 } w _ xa _ (opVal_reg c _ _ _ hns1 hreg1) hd
      simp only [StateT.run] at he
      simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure, core]
    · simp [Std.HashMap.get?_eq_getElem?]
    · intro R' hR'
      have hne : w ≠ R' := fun h => hR' ⟨"sw", s.n, Nat.le_refl _, h.symm⟩
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using hne),
        ← Std.HashMap.get?_eq_getElem?]
      exact hkeep1 R' (fun h => hR' (fresh_mono (by simp [s1]) h))
  · rw [runLM_addr_plain F tag s a hwf hsh]
    obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
      r_sound F tag s vs c ts a .i64 xa D hS hD hv hta hnw
    simp only [enc, show ClifTy.i64 ≠ .i8 by decide, if_false, and_wide64] at hreg1
    refine ⟨La, hLa, hn1, _, xa, rfl, hns1, hnf1, Or.inr ⟨?_, rfl⟩, regs1, hrun1, hreg1, hkeep1⟩
    unfold gspace; split <;> simp

/-- The PTX type a load of `lo` reads. -/
def ldTy (lo : LoadOp) : PtxIr.Ty :=
  match lo.kind with
  | .uload8 => .u8 | .uload16 => .u16 | .uload32 => .u32 | _ => bits lo.ty

/-- Whether a load through `a` takes the read-only path. -/
def ldNc (F : Facts) (lo : LoadOp) (a : Val) : Bool :=
  match F.origin.get? a.id with
  | some (.global p) => lo.kind == .plain && F.ro.contains p
  | _ => false

/-- A load: its address's lines, then one `ld`. -/
theorem lower_load (F : Facts) (tag : Nat) (d a : Val) (lo : LoadOp)
    (hk : lo.kind = .plain ∨ lo.kind = .uload8 ∨ lo.kind = .uload16 ∨ lo.kind = .uload32) :
    (lower F tag (.load d lo a)).lines =
      ((runLM (addr a) F tag {}).2.lines).push
        (.ins none (.ld (runLM (addr a) F tag {}).1.1 (ldNc F lo a) (ldTy lo) (regOf lo.ty d)
          (runLM (addr a) F tag {}).1.2)) := by
  rw [lower_inst F tag _ rfl, inst]
  rcases hk with hk | hk | hk | hk <;>
    simp only [hk, ldNc, ldTy, runLM_bind, facts?, runLM_read, runLM_pure, emit, emitL, runLM_modify] <;> rfl

open PtxRun in
/-- `ld` through a register holding `x`, in a space that takes the address as
    it is, reads the world's memory there. -/
theorem exec_ld (c : PtxRun.Ctx) (ts : PtxRun.TS) (sp : PtxIr.Space) (nc : Bool) (ty : PtxIr.Ty) (dr R : PtxIr.Reg)
    (x y v : UInt64) (hsp : sp ≠ .param) (hR : OpVal c ts (.reg R) x) (hp : place sp x = y)
    (hm : ts.w.mem.load y (ty.width / 8) = some v) :
    (exec c (.ld sp nc ty dr { base := .reg R })).run ts = .ok (.step, { ts with regs := ts.regs.insert dr v }) := by
  simp only [OpVal] at hR
  have hne : (sp == .param) = false := by cases sp <;> first | rfl | exact absurd rfl hsp
  simp only [exec, PtxRun.load, StateT.run, bind, StateT.bind, hne, Bool.false_eq_true, if_false] at hR ⊢
  simp [hR, Except.bind, ofInt_toNat, hp, hm, MonadState.get, getThe, MonadStateOf.get, StateT.get, setReg, pure,
    StateT.pure, Except.pure, modify, modifyGet, MonadStateOf.modifyGet, StateT.modifyGet]

/-- A scalar type other than a flag, as loads read it. -/
def LdTy (t : ClifTy) : Prop := t = .i16 ∨ t = .i32 ∨ t = .i64 ∨ t = .f32 ∨ t = .f64

open PtxRun in
/-- **A load** meets its obligation: one `ld` of the same bytes, through any
    address. -/
theorem load_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val) (lo : LoadOp)
    (hk : lo.kind = .plain ∨ lo.kind = .uload8 ∨ lo.kind = .uload16 ∨ lo.kind = .uload32)
    (hty : LdTy lo.ty) (hku : lo.kind ≠ .plain → IntT lo.ty ∧ (lo.kind = .uload32 → lo.ty ≠ .i16))
    (hta : F.tys.get? a.id = some .i64)
    (htd : F.tys.get? d.id = some lo.ty) (hld : F.lits.get? d.id = none) (hwd : F.win d = false)
    (hmo : magOk F d (.load d lo a) = true) :
    InstOk F lc c start0 (.load d lo a) := by
  intro tag D s s1 ts hst huses hS hW _ hmok
  have hua : a ∈ Inst.uses (.load d lo a) := by simp [Inst.uses]
  obtain ⟨ta, xa, hva, hta'⟩ := hS.pres a (huses a hua)
  rw [hta] at hta'; cases hta'
  have hne8 : lo.ty ≠ .i8 := by rcases hty with h | h | h | h | h <;> rw [h] <;> decide
  have hlanes : lo.ty.lanes = none := by rcases hty with h | h | h | h | h <;> rw [h] <;> rfl
  -- the bytes read, and the value the block semantics makes of them
  let n : Nat := (ldTy lo).width / 8
  obtain ⟨b, hb, rfl⟩ : ∃ b, s.world.mem.load xa n = some b ∧
      s1 = { s with vals := Blocks.setV s.vals d (.sc lo.ty (b &&& widthMask lo.ty)) } := by
    rcases hk with hk | hk | hk | hk <;>
    · simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.get, hk, hlanes, norm]
        at hst
      revert hst
      generalize hb : s.world.mem.load xa _ = ob
      intro hst
      cases ob with
      | none => simp at hst
      | some b =>
      simp at hst
      subst hst
      refine ⟨b, ?_, ?_⟩
      · first
          | (simp only [n, ldTy, hk]; exact hb)
          | (rcases hty with h | h | h | h | h <;> simp only [n, ldTy, hk, h, bits, PtxIr.Ty.width] <;>
              simp only [h, tyBytes, ClifTy.width] at hb <;> exact hb)
      · first
          | rfl
          | (have hlt := load_lt _ _ _ b (by rcases hty with h | h | h | h | h <;> simp [h, tyBytes, ClifTy.width]) hb
             congr 3
             rcases hty with h | h | h | h | h <;> simp only [h] at hlt ⊢ <;>
             · apply UInt64.toNat_inj.mp
               first
                 | rw [show widthMask .i64 = ((2 ^ 64 - 1 : Nat).toUInt64) from rfl]
                 | rw [show widthMask .f64 = ((2 ^ 64 - 1 : Nat).toUInt64) from rfl]
                 | rw [show widthMask .i32 = ((2 ^ 32 - 1 : Nat).toUInt64) from rfl]
                 | rw [show widthMask .f32 = ((2 ^ 32 - 1 : Nat).toUInt64) from rfl]
                 | rw [show widthMask .i16 = ((2 ^ 16 - 1 : Nat).toUInt64) from rfl]
               rw [toNat_and_mask _ _ (by decide)]
               simp [tyBytes, ClifTy.width] at hlt
               exact (Nat.mod_eq_of_lt hlt).symm)
  -- the bytes fit the value's type
  have hn8 : n ≤ 8 := by
    rcases hk with hk | hk | hk | hk <;> simp only [n, ldTy, hk] <;>
      (try rcases hty with h | h | h | h | h <;> rw [h]) <;> decide
  have hwn : 8 * n ≤ lo.ty.width := by
    rcases hk with hk | hk | hk | hk
    · simp only [n, ldTy, hk]; rcases hty with h | h | h | h | h <;> rw [h] <;> decide
    all_goals
      obtain ⟨hi, h32⟩ := hku (by rw [hk]; intro h; cases h)
      simp only [n, ldTy, hk]
      rcases hi with h | h | h <;> rw [h] <;> (try decide)
      all_goals exact absurd (h32 hk) (by rw [h]; decide)
  have hbw : b &&& widthMask lo.ty = b := by
    have hlt := load_lt _ _ _ b hn8 hb
    have hw64 : lo.ty.width ≤ 64 := by rcases hty with h | h | h | h | h <;> rw [h] <;> decide
    have hwm : widthMask lo.ty = ((2 ^ lo.ty.width - 1 : Nat).toUInt64) := by
      rcases hty with h | h | h | h | h <;> rw [h] <;> rfl
    apply UInt64.toNat_inj.mp
    rw [hwm, toNat_and_mask _ _ hw64]
    exact Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le hlt (Nat.pow_le_pow_right (by decide) hwn))
  rw [hbw]
  -- the address's lines, then the load
  obtain ⟨La, hLa, hn1, R, x, hbase, hns1, hnf1, hreach, regs1, hrun1, hreg1, hkeep1⟩ :=
    addr_sound F tag {} s.vals c ts a xa D hS (huses a hua) hva hta
  let core : PtxIr.Instr := .ld (runLM (addr a) F tag {}).1.1 (ldNc F lo a) (ldTy lo) (regOf lo.ty d) { base := .reg R }
  have hlist : (lower F tag (.load d lo a)).lines = ((La ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_load F tag d a lo hk, hLa, hbase]; simp [core]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hmem : ts.w.mem = s.world.mem := (congrArg World.mem hW.1).symm
  have hb' : ts.w.mem.load xa n = some b := by rw [hmem]; exact hb
  have he : (exec c core).run { ts with regs := regs1 } =
      .ok (.step, { ts with regs := regs1.insert (regOf lo.ty d) b }) :=
    exec_ld c { ts with regs := regs1 } _ _ _ _ _ x xa b (reaches_sp hreach) (opVal_reg c _ _ _ hns1 hreg1)
      (place_eq hreach (fit_load hb' hmok.2)) hb'

  refine ⟨{ ts with regs := regs1.insert (regOf lo.ty d) b }, ?_, ?_, hW⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _ hrun1]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ _ d lo.ty b rfl hS htd hbw
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?, enc, hne8, hbw]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; exact absurd h8 hne8
    · intro h8; exact absurd h8 hne8
    · intro c' n' hn
      have hne : PtxIr.Reg.val c' n' ≠ regOf lo.ty d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        ← Std.HashMap.get?_eq_getElem?]
      exact hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)

/-! ## Stores through global and generic addresses -/

/-- The relation over fewer defined values. -/
theorem Sim.sub {F : Facts} {D D' : Nat → Prop} {vs : Blocks.Vals} {regs : Std.HashMap PtxIr.Reg UInt64}
    (h : Sim F D vs regs) (hs : ∀ n, D' n → D n) : Sim F D' vs regs :=
  { pres := fun v hv => h.pres v (hs _ hv),
    normd := fun v t b hv => h.normd v t b (hs _ hv),
    flag := fun v b hv => h.flag v b (hs _ hv),
    plain := fun v t b hv => h.plain v t b (hs _ hv),
    lit := fun v t b l hv => h.lit v t b l (hs _ hv),
    win := fun v t b hv => h.win v t b (hs _ hv),
    neg := fun v b hv => h.neg v b (hs _ hv),
    bound := fun v t b hv => h.bound v t b (hs _ hv) }

/-- Both worlds take the same memory and the same observation: still related. -/
theorem wrel_cons (bs ps : Nat) (bw pw : World) (m : Mem) (o : Obs) (h : WRel bs ps bw pw) :
    WRel bs ps { bw with mem := m, obs := o :: bw.obs } { pw with mem := m, obs := o :: pw.obs } := by
  obtain ⟨h1, h3, h4, h5⟩ := h
  refine ⟨?_, by simp; omega, by simp; omega, ?_⟩
  · have := congrArg (fun w : World => { w with mem := m }) h1
    simpa using this
  · simp only [List.reverse_cons]
    rw [List.drop_append_of_le_length (by simpa using h3), List.drop_append_of_le_length (by simpa using h4),
      List.filterMap_append, List.filterMap_append, h5]

/-- A store: the address's lines, the value's, then one `st`. -/
theorem lower_store (F : Facts) (tag : Nat) (v a : Val) (t : ClifTy) (htv : F.tys.get? v.id = some t) :
    (lower F tag (.store v a)).lines =
      ((runLM (r v) F tag (runLM (addr a) F tag {}).2).2.lines).push
        (.ins none (.st (runLM (addr a) F tag {}).1.1 (bits t) (runLM (addr a) F tag {}).1.2
          (.reg (runLM (r v) F tag (runLM (addr a) F tag {}).2).1))) := by
  rw [lower_inst F tag _ rfl, inst]
  simp only [runLM_bind, runLM_tyOf F tag _ v t htv]
  rfl

open PtxRun in
/-- `st` through a register holding `x`, in a space that takes the address as
    it is, stores there and records the store. -/
theorem exec_st (c : PtxRun.Ctx) (ts : PtxRun.TS) (sp : PtxIr.Space) (ty : PtxIr.Ty) (R : PtxIr.Reg)
    (vo : PtxIr.Opnd) (x z y : UInt64) (m : Mem) (hR : OpVal c ts (.reg R) x) (hp : place sp x = z)
    (hv : OpVal c ts vo y) (hm : ts.w.mem.store z (ty.width / 8) (PtxRun.mask ty.width y) = some m) :
    (exec c (.st sp ty { base := .reg R } vo)).run ts =
      .ok (.step, { ts with w := { ts.w with mem := m, obs := (Obs.store z (ty.width / 8) (PtxRun.mask ty.width y)) :: ts.w.obs } }) := by
  simp only [OpVal] at hR hv
  simp only [exec, PtxRun.store, StateT.run, bind, StateT.bind] at hR hv ⊢
  simp [hR, hv, Except.bind, ofInt_toNat, hp, hm, MonadState.get, getThe, MonadStateOf.get, StateT.get, set,
    MonadStateOf.set, StateT.set, pure, StateT.pure, Except.pure]

open PtxRun in
/-- **A store** through a global or generic address meets its obligation,
    given what the block semantics stores (`nb t` bytes of `fv t b` for a
    value of type `t` with bits `b`), its lowering (the address's lines, the
    value's, then one `st` of PTX type `pty t`), and that the `st` stores the
    same. -/
theorem st_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (i : Inst) (v a : Val)
    (pty : ClifTy → PtxIr.Ty) (nb : ClifTy → Nat) (fv : ClifTy → UInt64 → UInt64)
    (hua : a ∈ Inst.uses i) (huv : v ∈ Inst.uses i) (hdst : Inst.dst? i = none)
    (hwv : ¬ Win F v)
    (hta : F.tys.get? a.id = some .i64) (htv : ∀ t, F.tys.get? v.id = some t → LdTy t)
    (hstep : ∀ (s s1 : Blocks.BSt), step1 lc s i = some s1 → ∀ t b xa, Blocks.getV s.vals v = some (.sc t b) →
      F.tys.get? v.id = some t → Blocks.getV s.vals a = some (.sc .i64 xa) →
      ∃ m, s.world.mem.store xa (nb t) (fv t b) = some m ∧
        s1 = { s with world := { obsStore s.world xa (nb t) (fv t b) with mem := m } })
    (hlow : ∀ tag t, F.tys.get? v.id = some t → (lower F tag i).lines =
      ((runLM (r v) F tag (runLM (addr a) F tag {}).2).2.lines).push
        (.ins none (.st (runLM (addr a) F tag {}).1.1 (pty t) (runLM (addr a) F tag {}).1.2
          (.reg (runLM (r v) F tag (runLM (addr a) F tag {}).2).1))))
    (hsem : ∀ t b, LdTy t → b &&& widthMask t = b →
      (pty t).width / 8 = nb t ∧ PtxRun.mask (pty t).width b = fv t b) :
    InstOk F lc c start0 i := by
  intro tag D s s1 ts hst huses hS hW _ hmok
  obtain ⟨ta, xa, hva, hta'⟩ := hS.pres a (huses a hua)
  rw [hta] at hta'; cases hta'
  obtain ⟨t, b, hvv, htv'⟩ := hS.pres v (huses v huv)
  have hlt := htv t htv'
  have hne8 : t ≠ .i8 := by rcases hlt with h | h | h | h | h <;> rw [h] <;> decide
  have hbn := hS.normd v t b (huses v huv) hvv
  obtain ⟨m, hm, rfl⟩ := hstep s s1 hst t b xa hvv htv' hva
  obtain ⟨hbytes, hval⟩ := hsem t b hlt hbn
  -- the lines
  let sa := (runLM (addr a) F tag {}).2
  obtain ⟨La, hLa, hn1, R, x, hbase, hns1, hnf1, hreach, regs1, hrun1, hreg1, hkeep1⟩ :=
    addr_sound F tag {} s.vals c ts a xa D hS (huses a hua) hva hta
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lv, hLv, hn2, hnf2, hns2, regs2, hrun2, hreg2, hkeep2⟩ :=
    r_sound F tag sa s.vals c { ts with regs := regs1 } v t b D hS1 (huses v huv) hvv htv' hwv
  simp only [enc, hne8, if_false, hbn] at hreg2
  let core : PtxIr.Instr := .st (runLM (addr a) F tag {}).1.1 (pty t) { base := .reg R }
    (.reg (runLM (r v) F tag sa).1)
  have hlist : (lower F tag i).lines = ((La ++ Lv ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow tag t htv', hLv]
    show (sa.lines ++ _).push _ = _
    rw [hLa, hbase]; simp [core, sa]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hmem : ts.w.mem = s.world.mem := (congrArg World.mem hW.1).symm
  have hm' : ts.w.mem.store xa (nb t) (fv t b) = some m := by rw [hmem]; exact hm
  have he : (exec c core).run { ts with regs := regs2 } =
      .ok (.step, { ts with regs := regs2, w := { ts.w with mem := m, obs := (Obs.store xa (nb t) (fv t b)) :: ts.w.obs } }) := by
    have := exec_st c { ts with regs := regs2 } _ (pty t) _ _ x xa b m
      (opVal_reg c _ _ _ hns1 (by rw [hkeep2 _ hnf1]; exact hreg1)) (place_eq hreach (fit_store hm' hmok.2))
      (opVal_reg c _ _ _ hns2 hreg2)
      (by rw [hbytes, hval]; exact hm')
    simpa [hbytes, hval] using this
  refine ⟨{ ts with regs := regs2, w := { ts.w with mem := m, obs := (Obs.store xa (nb t) (fv t b)) :: ts.w.obs } },
    ?_, ?_, ?_⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · have hS2 : Sim F D s.vals regs2 := hS1.frame (fun c' n => hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h))
    exact hS2.sub (fun n h => by simpa [extD, hdst] using h)
  · exact wrel_cons start0 _ s.world ts.w m _ hW

open PtxRun in
/-- **A store of a value's bytes** through a global or generic address meets
    its obligation. -/
theorem store_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (v a : Val)
    (hwv : ¬ Win F v)
    (hta : F.tys.get? a.id = some .i64) (htv : ∀ t, F.tys.get? v.id = some t → LdTy t) :
    InstOk F lc c start0 (.store v a) := by
  apply st_ok F lc c start0 _ v a bits tyBytes (fun _ b => b) (by simp [Inst.uses]) (by simp [Inst.uses]) rfl
    hwv hta htv
  · intro s s1 hst t b xa hvv htvt hva
    simp only [step1, Blocks.doStore, hvv, hva, Option.getD_none] at hst
    revert hst
    generalize s.world.mem.store xa (tyBytes t) b = om
    intro hst
    cases om with
    | none => simp [Except.toOption] at hst
    | some m => simp [Except.toOption] at hst; exact ⟨m, rfl, hst.symm⟩
  · intro tag t ht; exact lower_store F tag v a t ht
  · intro t b hlt hbn
    have hne8 : t ≠ .i8 := by rcases hlt with h | h | h | h | h <;> rw [h] <;> decide
    refine ⟨by rcases hlt with h | h | h | h | h <;> rw [h] <;> rfl, ?_⟩
    have := mask_enc t b; simpa [enc, hne8, hbn] using this

/-- A byte store: the address's lines, the value's, then one `st.u8`. -/
theorem lower_istore8 (F : Facts) (tag : Nat) (v a : Val) :
    (lower F tag (.istore8 v a)).lines =
      ((runLM (r v) F tag (runLM (addr a) F tag {}).2).2.lines).push
        (.ins none (.st (runLM (addr a) F tag {}).1.1 .u8 (runLM (addr a) F tag {}).1.2
          (.reg (runLM (r v) F tag (runLM (addr a) F tag {}).2).1))) := by
  rw [lower_inst F tag _ rfl, inst]
  rfl

open PtxRun in
/-- **A store of a value's low byte** through a global or generic address
    meets its obligation. -/
theorem istore8_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (v a : Val)
    (hwv : ¬ Win F v)
    (hta : F.tys.get? a.id = some .i64) (htv : ∀ t, F.tys.get? v.id = some t → LdTy t) :
    InstOk F lc c start0 (.istore8 v a) := by
  apply st_ok F lc c start0 _ v a (fun _ => .u8) (fun _ => 1) (fun _ b => b &&& 0xff) (by simp [Inst.uses])
    (by simp [Inst.uses]) rfl hwv hta htv
  · intro s s1 hst t b xa hvv htvt hva
    simp only [step1, hvv, hva] at hst
    revert hst
    generalize s.world.mem.store xa 1 (b &&& 0xff) = om
    intro hst
    cases om with
    | none => simp at hst
    | some m => simp at hst; exact ⟨m, rfl, hst.symm⟩
  · intro tag t _; exact lower_istore8 F tag v a
  · intro t b _ _; exact ⟨rfl, rfl⟩

/-! ## Bits reinterpreted -/

open PtxRun in
theorem exec_mov (c : PtxRun.Ctx) (ts : PtxRun.TS) (ty : PtxIr.Ty) (d : PtxIr.Reg) (xo : PtxIr.Opnd) (x : UInt64)
    (hx : OpVal c ts xo x) :
    (exec c (.mov ty d xo)).run ts = .ok (.step, { ts with regs := ts.regs.insert d (mask ty.width x) }) := by
  simp only [OpVal] at hx
  simp only [exec, StateT.run, bind, StateT.bind] at hx ⊢
  simp [hx, Except.bind, setReg, pure, StateT.pure, Except.pure, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet]

/-- Types of one width, flags apart, keep the same bits. -/
theorem wm_width (t u : ClifTy) (ht : LdTy t) (hu : LdTy u) (h : t.width = u.width) : widthMask t = widthMask u := by
  rcases ht with rfl | rfl | rfl | rfl | rfl <;> rcases hu with rfl | rfl | rfl | rfl | rfl <;>
    first | rfl | (simp [ClifTy.width] at h)

theorem mask_ldTy (t : ClifTy) (ht : LdTy t) (x : UInt64) (hx : x &&& widthMask t = x) :
    PtxRun.mask (bTy t.width).width x = x := by
  rcases ht with rfl | rfl | rfl | rfl | rfl
  · rw [show (bTy ClifTy.i16.width).width = ClifTy.i16.width from rfl, mask_width .i16 (by simp [IntT]), hx]
  · rw [show (bTy ClifTy.i32.width).width = ClifTy.i32.width from rfl, mask_width .i32 (by simp [IntT]), hx]
  · rw [show (bTy ClifTy.i64.width).width = ClifTy.i64.width from rfl, mask_width .i64 (by simp [IntT]), hx]
  · rw [show (bTy ClifTy.f32.width).width = ClifTy.i32.width from rfl, mask_width .i32 (by simp [IntT]),
      show widthMask .i32 = widthMask .f32 from rfl, hx]
  · rw [show (bTy ClifTy.f64.width).width = ClifTy.i64.width from rfl, mask_width .i64 (by simp [IntT]),
      show widthMask .i64 = widthMask .f64 from rfl, hx]

theorem inst_bitcast (d : Val) (ty : ClifTy) (a : Val) :
    inst (.bitcast d ty a) = (do emit (.mov (bTy ty.width) (regOf ty d) (← o a))) := by simp only [inst]

open PtxRun in
/-- **Bits reinterpreted** as another type of their width meet their
    obligation: one `mov` of the same bits. -/
theorem bitcast_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d : Val) (ty : ClifTy) (a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false) (hty : LdTy ty) (hTa : ∀ t, F.tys.get? a.id = some t → LdTy t)
    (htd : F.tys.get? d.id = some ty) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.bitcast d ty a) = true) :
    InstOk F lc c start0 (.bitcast d ty a) := by
  have hne8 : ty ≠ .i8 := by rcases hty with h | h | h | h | h <;> rw [h] <;> decide
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a (fun _ dr xo => .mov (bTy ty.width) dr xo) (fun _ => ty) (fun _ x => x)
    (fun t => LdTy t ∧ t.width = ty.width) (fun _ _ => hne8) rfl (by simp [Inst.uses]) hwa hwd (fun _ _ => htd) hld
  · intro s s1 hst _ t xa hva hta
    simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.get] at hst
    by_cases hw : t.width = ty.width
    · simp [hw] at hst
      exact ⟨⟨hTa t hta, hw⟩, hst.symm⟩
    · simp [hw] at hst
  · intro tag t ht
    rw [lower_inst F tag _ rfl, inst_bitcast]
    simp only [runLM_bind, runLM_pure, o, emit, emitL, runLM_modify]
  · intro t xa ⟨hlt, hw⟩ hn _ c' ts' xo dr hxo
    have hne8t : t ≠ .i8 := by rcases hlt with h | h | h | h | h <;> rw [h] <;> decide
    rw [exec_mov c' ts' _ dr xo _ hxo]
    simp only [enc, hne8t, hne8, if_false, hn]
    rw [← hw, mask_ldTy t hlt xa hn, ← wm_width t ty hlt hty hw, hn]
  · intro t xa ⟨hlt, hw⟩ hn
    rw [← wm_width t ty hlt hty hw, hn]

/-! ## Collectives -/

/-- Observations a launch reads keep their collectives: as many, in order. -/
theorem colls_view : ∀ l : List Obs, (l.filterMap collOf).length = ((l.filterMap viewObs).filterMap collOf).length
  | [] => rfl
  | o :: l => by
      have ih := colls_view l
      cases o with
      | call c args =>
          cases c with
          | intr i =>
              by_cases hc : i.isColl = true
              · simp [List.filterMap_cons, collOf, viewObs, hc, ih]
              · simp [List.filterMap_cons, collOf, viewObs, hc, ih]
          | _ => simp [List.filterMap_cons, collOf, viewObs, ih]
      | _ => simp [List.filterMap_cons, collOf, viewObs, ih]

/-- **Related worlds have made as many collectives** since the thread began. -/
theorem collCalls_wrel (bs ps : Nat) (bw pw : World) (h : WRel bs ps bw pw) :
    (collCalls bw bs).length = (collCalls pw ps).length := by
  unfold collCalls
  rw [colls_view, colls_view (pw.obs.reverse.drop ps), h.2.2.2]

/-- Recording one more observation adds its collective, if any, last. -/
theorem collCalls_push (w : World) (start : Nat) (o : Obs) (h : start ≤ w.obs.length) :
    collCalls { w with obs := o :: w.obs } start = collCalls w start ++ (collOf o).toList := by
  simp only [collCalls, List.reverse_cons]
  rw [List.drop_append_of_le_length (by simpa using h), List.filterMap_append]
  cases hc : collOf o <;> simp [hc]

/-- Worlds that each add an observation a launch reads alike, and are the
    same apart from their observations, stay related. -/
theorem wrel_step (bs ps : Nat) (bw pw bw' pw' : World) (o o' : Obs) (h : WRel bs ps bw pw)
    (hb : bw'.obs = o :: bw.obs) (hp : pw'.obs = o' :: pw.obs) (hv : viewObs o = viewObs o')
    (heq : { bw' with obs := [] } = { pw' with obs := [] }) : WRel bs ps bw' pw' := by
  obtain ⟨-, h3, h4, h5⟩ := h
  refine ⟨heq, by rw [hb]; simp; omega, by rw [hp]; simp; omega, ?_⟩
  rw [hb, hp]; simp only [List.reverse_cons]
  rw [List.drop_append_of_le_length (by simpa using h3), List.drop_append_of_le_length (by simpa using h4),
    List.filterMap_append, List.filterMap_append, h5]
  simp only [List.filterMap_cons, List.filterMap_nil, hv]

open PtxRun in
/-- **A barrier** meets its obligation: both semantics record it, and take the
    shared memory its answer gives, or, provisionally, none. Where each block is
    one warp it is the warp's barrier, the block's in a block of at most 32
    threads. -/
theorem barrier_ok (F : Facts) (c : PtxRun.Ctx) (start0 : Nat)
    (hwarp : F.warp = true → (c.special.lookup "ntid.x").getD 0 ≤ 32) :
    InstOk F (threadLocals c.ans start0 c.hw c.provisional) c start0 (.call none (.intr .barrier) []) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have hlow : (lower F tag (.call none (.intr .barrier) [])).lines =
      (([if F.warp then PtxIr.Instr.barWarp (.imm 0xffffffff) else .bar 0] : List PtxIr.Instr).map
        (PtxIr.Line.ins none)).toArray := by
    rw [lower_inst F tag _ rfl]; simp [inst, voidIntr, emit, emitL, facts?]
  have hbi : exec c (if F.warp then PtxIr.Instr.barWarp (.imm 0xffffffff) else .bar 0) = exec c (.bar 0) := by
    cases hw : F.warp
    · rfl
    · simp only [if_true, exec, if_pos (hwarp hw)]
  refine ⟨by rw [hlow]; exact straight_map _, ?_⟩
  rw [hlow, instrsOf_map]
  have hlen := collCalls_wrel start0 _ s.world ts.w hW
  have hle : start0 ≤ s.world.obs.reverse.length := by simpa using hW.2.1
  have hmem : ts.w.mem = s.world.mem := (congrArg World.mem hW.1).symm
  simp only [step1, List.mapM_nil, pure_bind, callOf, obsCall, callWords, List.append_nil, threadLocals,
    Intr.find_slot, Intr.isColl, if_true, collCalls, List.reverse_cons, List.drop_append_of_le_length hle,
    List.filterMap_append, collOf, List.filterMap_cons, List.filterMap_nil, List.length_append,
    List.length_singleton, Nat.add_sub_cancel] at hst
  have hk : ((s.world.obs.reverse.drop start0).filterMap collOf).length = (collCalls ts.w c.start).length := hlen
  rw [hk] at hst
  have hsub : Sim F (extD D (.call none (.intr .barrier) [])) s.vals ts.regs :=
    hS.sub (fun n h => by simpa [extD, Inst.dst?] using h)
  have hrun : ∀ ts1 : PtxRun.TS, (exec c (.bar 0)).run ts = .ok (.step, ts1) →
      (steps c [if F.warp then PtxIr.Instr.barWarp (.imm 0xffffffff) else .bar 0]).run ts = .ok ((), ts1) := by
    intro ts1 h
    rw [← hbi] at h
    simp only [StateT.run] at h
    simp [steps, StateT.run, h, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  generalize hA : c.ans[(collCalls ts.w c.start).length]? = A at hst
  have hex := fun (m : Mem) (h : (exec c (.bar 0)).run ts =
      .ok (.step, { ts with w := { ts.w with mem := m, obs := .call (.intr .barrier) [] :: ts.w.obs } })) => h
  cases A with
  | none =>
      cases hprov : c.provisional
      · simp [hprov, Except.toOption] at hst
      · simp [hprov, Except.toOption] at hst
        subst hst
        refine ⟨{ ts with w := { ts.w with obs := .call (.intr .barrier) [] :: ts.w.obs } }, hrun _ ?_, hsub,
          wrel_step start0 _ s.world ts.w _ _ (.call (.intr .barrier) []) (.call (.intr .barrier) []) hW rfl rfl rfl (by simpa using hW.1)⟩
        simp only [exec, coll, StateT.run, bind, StateT.bind, hA, hprov]
        simp [MonadState.get, getThe, MonadStateOf.get, StateT.get, set, MonadStateOf.set, StateT.set, pure,
          StateT.pure, Except.pure, Except.bind, hmem, hA, show (Intr.barrier == Intr.barrier) = true from rfl, hprov]
  | some a =>
      cases a with
      | smem b =>
          simp [Except.toOption] at hst
          subst hst
          refine ⟨{ ts with w := { ts.w with mem := { ts.w.mem with arena := b }, obs := .call (.intr .barrier) [] :: ts.w.obs } }, hrun _ ?_, hsub,
            wrel_step start0 _ s.world ts.w _ _ (.call (.intr .barrier) []) (.call (.intr .barrier) []) hW rfl rfl rfl ?_⟩
          · simp only [exec, coll, StateT.run, bind, StateT.bind]
            simp [MonadState.get, getThe, MonadStateOf.get, StateT.get, set, MonadStateOf.set, StateT.set, pure,
              StateT.pure, Except.pure, Except.bind, modify, modifyGet, MonadStateOf.modifyGet, StateT.modifyGet, hA,
              show (Intr.barrier == Intr.barrier) = true from rfl, StateT.bind]
          · have h2 := congrArg (fun w : World => { w with mem := { w.mem with arena := b }, obs := ([] : List Obs) })
              hW.1
            simpa using h2
      | val v =>
          cases hprov : c.provisional
          · simp [hprov, Except.toOption] at hst
          · simp [hprov, Except.toOption] at hst
            subst hst
            refine ⟨{ ts with w := { ts.w with obs := .call (.intr .barrier) [] :: ts.w.obs } }, hrun _ ?_, hsub,
              wrel_step start0 _ s.world ts.w _ _ (.call (.intr .barrier) []) (.call (.intr .barrier) []) hW rfl rfl rfl (by simpa using hW.1)⟩
            simp only [exec, coll, StateT.run, bind, StateT.bind, hA, hprov]
            simp [MonadState.get, getThe, MonadStateOf.get, StateT.get, set, MonadStateOf.set, StateT.set, pure,
              StateT.pure, Except.pure, Except.bind, hmem, hA, show (Intr.barrier == Intr.barrier) = true from rfl, hprov]

/-- **Answers a thread can take**: every value a 32-bit integer cut to its
    width, as the laws give a shuffle's or a ballot's. -/
theorem valAt_ok (ans : List Answer) (h : AnsOk ans) (k : Nat) :
    ∃ x, valAt ans k = .sc .i32 x ∧ x &&& widthMask .i32 = x := by
  unfold valAt
  split
  · rename_i v hv; exact h.1 k v hv
  · exact ⟨0, rfl, rfl⟩

open PtxRun in
/-- **A shuffle** meets its obligation: both semantics record it with the same
    arguments, and take the answer at the same place among the thread's. -/
theorem shfl_ok (F : Facts) (c : PtxRun.Ctx) (start0 : Nat) (hans : AnsOk c.ans)
    (d v m : Val) (i : Intr) (md : PtxIr.Shfl) (hi : (i = .shflBfly ∧ md = .bfly) ∨ (i = .shflIdx ∧ md = .idx))
    (hwv : ¬ Win F v) (hwm : ¬ Win F m) (hwd : F.win d = false)
    (htv : F.tys.get? v.id = some .i32) (htm : F.tys.get? m.id = some .i32) (htd : F.tys.get? d.id = some .i32)
    (hld : F.lits.get? d.id = none)
    (hlow : ∀ tag, (lower F tag (.call (some d) (.intr i) [v, m])).lines =
      ((runLM (r m) F tag (runLM (r v) F tag {}).2).2.lines).push
        (.ins none (.shfl md (regOf .i32 d) (.reg (runLM (r v) F tag {}).1) (.reg (runLM (r m) F tag (runLM (r v) F tag {}).2).1)
          (.imm 31) (.imm 0xffffffff))))
    (hmo : magOk F d (.call (some d) (.intr i) [v, m]) = true) :
    InstOk F (threadLocals c.ans start0 c.hw c.provisional) c start0 (.call (some d) (.intr i) [v, m]) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have huv : v ∈ Inst.uses (.call (some d) (.intr i) [v, m]) := by simp [Inst.uses]
  have hum : m ∈ Inst.uses (.call (some d) (.intr i) [v, m]) := by simp [Inst.uses]
  obtain ⟨tv, xv, hvv, htv'⟩ := hS.pres v (huses v huv)
  rw [htv] at htv'; cases htv'
  obtain ⟨tm, xm, hvm, htm'⟩ := hS.pres m (huses m hum)
  rw [htm] at htm'; cases htm'
  have hnv := hS.normd v .i32 xv (huses v huv) hvv
  have hnm := hS.normd m .i32 xm (huses m hum) hvm
  have hcoll : i.isColl = true := by rcases hi with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> rfl
  have hnb : (i == .barrier) = false := by rcases hi with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> rfl
  have hlen := collCalls_wrel start0 _ s.world ts.w hW
  have hle : start0 ≤ s.world.obs.reverse.length := by simpa using hW.2.1
  -- the block semantics' step
  let args : List V := [.sc .i32 xv, .sc .i32 xm]
  let o : Obs := .call (.intr i) args
  obtain ⟨x, hx, hxn⟩ := valAt_ok c.ans hans (collCalls ts.w c.start).length
  have hs1 : s1 = { vals := Blocks.setV s.vals d (.sc .i32 x), world := { s.world with obs := o :: s.world.obs } } := by
    simp only [step1, List.mapM_cons, List.mapM_nil, hvv, hvm, bind, Option.bind, pure, callOf, obsCall, callWords,
      List.append_nil, threadLocals, Intr.find_slot, hcoll, if_true, hnb, Bool.false_eq_true, if_false] at hst
    simp only [collCalls, List.reverse_cons, List.drop_append_of_le_length hle, List.filterMap_append, collOf, hcoll,
      if_true, List.filterMap_cons, List.filterMap_nil, List.length_append, List.length_singleton,
      Nat.add_sub_cancel] at hst
    have hk : ((s.world.obs.reverse.drop start0).filterMap collOf).length = (collCalls ts.w c.start).length := hlen
    rw [hk, hx] at hst
    have hnb' : i ≠ .barrier := by rcases hi with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> decide
    have hns' : i ≠ .sharedAdd := by rcases hi with ⟨rfl, -⟩ | ⟨rfl, -⟩ <;> decide
    simp [Except.toOption, hnb', hns', hx] at hst
    exact hst.symm
  subst hs1
  -- the lines
  let sa := (runLM (r v) F tag {}).2
  obtain ⟨Lv, hLv, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag {} s.vals c ts v .i32 xv D hS (huses v huv) hvv htv hwv
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lm, hLm, hn2, hnf2, hns2, regs2, hrun2, hreg2, hkeep2⟩ :=
    r_sound F tag sa s.vals c { ts with regs := regs1 } m .i32 xm D hS1 (huses m hum) hvm htm hwm
  simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hnv] at hreg1
  simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hnm] at hreg2
  let core : PtxIr.Instr := .shfl md (regOf .i32 d) (.reg (runLM (r v) F tag {}).1) (.reg (runLM (r m) F tag sa).1)
    (.imm 31) (.imm 0xffffffff)
  have hlist : (lower F tag (.call (some d) (.intr i) [v, m])).lines =
      ((Lv ++ Lm ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow tag, hLm]
    show (sa.lines ++ _).push _ = _
    rw [hLv]; simp [core, sa]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hX : OpVal c { ts with regs := regs2 } (.reg (runLM (r v) F tag {}).1) xv :=
    opVal_reg c _ _ _ hns1 (by rw [hkeep2 _ hnf1]; exact hreg1)
  have hY : OpVal c { ts with regs := regs2 } (.reg (runLM (r m) F tag sa).1) xm := opVal_reg c _ _ _ hns2 hreg2
  have hm32 : ∀ z : UInt64, z &&& widthMask .i32 = z → PtxRun.mask 32 z = z := fun z hz => by
    rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp [IntT]), hz]
  have he : (exec c core).run { ts with regs := regs2 } =
      .ok (.step, { ts with regs := regs2.insert (regOf .i32 d) x, w := { ts.w with obs := o :: ts.w.obs } }) := by
    have h1 : regs2[(runLM (r v) F tag {}).1]? = some xv := by
      rw [← Std.HashMap.get?_eq_getElem?, hkeep2 _ hnf1]; exact hreg1
    have h2 : regs2[(runLM (r m) F tag sa).1]? = some xm := by
      rw [← Std.HashMap.get?_eq_getElem?]; exact hreg2
    rcases hi with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;>
    simp [core, exec, coll, opnd, getReg_plain _ _ hns1, getReg_plain _ _ hns2, h1, h2,
      show UInt64.ofInt 31 = (31 : UInt64) from rfl, show (Intr.shflBfly == Intr.barrier) = false from rfl,
      show (Intr.shflIdx == Intr.barrier) = false from rfl, hx, hm32 xv hnv, hm32 xm hnm, hm32 x hxn,
      MonadState.get, getThe, MonadStateOf.get, StateT.get, set, MonadStateOf.set, StateT.set, setReg, pure,
      StateT.pure, Except.pure, modify, modifyGet, MonadStateOf.modifyGet, StateT.modifyGet, o, args,
      StateT.bind, StateT.run, bind, Except.bind]
  refine ⟨{ ts with regs := regs2.insert (regOf .i32 d) x, w := { ts.w with obs := o :: ts.w.obs } }, ?_, ?_,
    wrel_step start0 _ s.world ts.w _ _ o o hW rfl rfl rfl (by simpa using hW.1)⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ (.call (some d) (.intr i) [v, m]) d .i32 x rfl hS htd hxn
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?, enc, hxn]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; cases h8
    · intro h8; cases h8
    · intro c' n hn
      have hne : PtxIr.Reg.val c' n ≠ regOf .i32 d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h), hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]

/-- Not fresh at `n` is not fresh at any later `m`. -/
theorem not_fresh_mono {tag n m : Nat} {R : PtxIr.Reg} (h : n ≤ m) (hn : ¬ Fresh tag n R) : ¬ Fresh tag m R :=
  fun ⟨p, k, hk, hR⟩ => hn ⟨p, k, Nat.le_trans h hk, hR⟩

/-- An `mma` lowers to its three arguments' lines, then the `mma`. -/
theorem lower_mma (F : Facts) (tag : Nat) (d a b e : Val) :
    (lower F tag (.call (some d) (.intr .mma) [a, b, e])).lines =
      ((runLM (r e) F tag (runLM (r b) F tag (runLM (r a) F tag {}).2).2).2.lines).push
        (.ins none (.mma (regOf .i32 d) (.reg (runLM (r a) F tag {}).1)
          (.reg (runLM (r b) F tag (runLM (r a) F tag {}).2).1)
          (.reg (runLM (r e) F tag (runLM (r b) F tag (runLM (r a) F tag {}).2).2).1))) := by
  rw [lower_inst F tag _ rfl, inst]
  simp [intr3, o, emit, emitL]

open PtxRun in
/-- **An `mma`** meets its obligation: both semantics record it with the same
    three words, and take the answer at the same place among the thread's. -/
theorem mma_ok (F : Facts) (c : PtxRun.Ctx) (start0 : Nat) (hans : AnsOk c.ans)
    (d a b e : Val) (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwe : ¬ Win F e) (hwd : F.win d = false)
    (hta : F.tys.get? a.id = some .i32) (htb : F.tys.get? b.id = some .i32) (hte : F.tys.get? e.id = some .i32)
    (htd : F.tys.get? d.id = some .i32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.call (some d) (.intr .mma) [a, b, e]) = true) :
    InstOk F (threadLocals c.ans start0 c.hw c.provisional) c start0 (.call (some d) (.intr .mma) [a, b, e]) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have hua : a ∈ Inst.uses (.call (some d) (.intr .mma) [a, b, e]) := by simp [Inst.uses]
  have hub : b ∈ Inst.uses (.call (some d) (.intr .mma) [a, b, e]) := by simp [Inst.uses]
  have hue : e ∈ Inst.uses (.call (some d) (.intr .mma) [a, b, e]) := by simp [Inst.uses]
  obtain ⟨ta, xa, hva, hta'⟩ := hS.pres a (huses a hua)
  rw [hta] at hta'; cases hta'
  obtain ⟨tb, xb, hvb, htb'⟩ := hS.pres b (huses b hub)
  rw [htb] at htb'; cases htb'
  obtain ⟨te, xe, hve, hte'⟩ := hS.pres e (huses e hue)
  rw [hte] at hte'; cases hte'
  have hna := hS.normd a .i32 xa (huses a hua) hva
  have hnb := hS.normd b .i32 xb (huses b hub) hvb
  have hne := hS.normd e .i32 xe (huses e hue) hve
  have hlen := collCalls_wrel start0 _ s.world ts.w hW
  have hle : start0 ≤ s.world.obs.reverse.length := by simpa using hW.2.1
  -- the block semantics' step
  let o : Obs := .call (.intr .mma) [.sc .i32 xa, .sc .i32 xb, .sc .i32 xe]
  obtain ⟨x, hx, hxn⟩ := valAt_ok c.ans hans (collCalls ts.w c.start).length
  have hs1 : s1 = { vals := Blocks.setV s.vals d (.sc .i32 x), world := { s.world with obs := o :: s.world.obs } } := by
    simp only [step1, List.mapM_cons, List.mapM_nil, hva, hvb, hve, bind, Option.bind, pure, callOf, obsCall, callWords,
      List.append_nil, threadLocals, Intr.find_slot, Intr.isColl, if_true] at hst
    simp only [collCalls, List.reverse_cons, List.drop_append_of_le_length hle, List.filterMap_append, collOf,
      Intr.isColl, if_true, List.filterMap_cons, List.filterMap_nil, List.length_append, List.length_singleton,
      Nat.add_sub_cancel] at hst
    have hk : ((s.world.obs.reverse.drop start0).filterMap collOf).length = (collCalls ts.w c.start).length := hlen
    rw [hk, hx] at hst
    simp [Except.toOption, show Intr.mma ≠ .barrier by decide, show Intr.mma ≠ .sharedAdd by decide] at hst
    exact hst.symm
  subst hs1
  -- the lines: the three words', then one `mma`
  let sa := (runLM (r a) F tag {}).2
  let sb := (runLM (r b) F tag sa).2
  obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag {} s.vals c ts a .i32 xa D hS (huses a hua) hva hta hwa
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lb, hLb, hn2, hnf2, hns2, regs2, hrun2, hreg2, hkeep2⟩ :=
    r_sound F tag sa s.vals c { ts with regs := regs1 } b .i32 xb D hS1 (huses b hub) hvb htb hwb
  have hS2 : Sim F D s.vals regs2 := hS1.frame (fun c' n => hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Le, hLe, hn3, hnf3, hns3, regs3, hrun3, hreg3, hkeep3⟩ :=
    r_sound F tag sb s.vals c { ts with regs := regs2 } e .i32 xe D hS2 (huses e hue) hve hte hwe
  simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hna] at hreg1
  simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hnb] at hreg2
  simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hne] at hreg3
  let core : PtxIr.Instr := .mma (regOf .i32 d) (.reg (runLM (r a) F tag {}).1) (.reg (runLM (r b) F tag sa).1)
    (.reg (runLM (r e) F tag sb).1)
  have hlist : (lower F tag (.call (some d) (.intr .mma) [a, b, e])).lines =
      ((La ++ Lb ++ Le ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_mma F tag d a b e, hLe]
    show (sb.lines ++ _).push _ = _
    rw [hLb]
    show ((sa.lines ++ _) ++ _).push _ = _
    rw [hLa]; simp [core, sa, sb]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hm32 : ∀ z : UInt64, z &&& widthMask .i32 = z → PtxRun.mask 32 z = z := fun z hz => by
    rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp [IntT]), hz]
  have he : (exec c core).run { ts with regs := regs3 } =
      .ok (.step, { ts with regs := regs3.insert (regOf .i32 d) x, w := { ts.w with obs := o :: ts.w.obs } }) := by
    have h1 : regs3[(runLM (r a) F tag {}).1]? = some xa := by
      rw [← Std.HashMap.get?_eq_getElem?, hkeep3 _ (not_fresh_mono hn2 hnf1), hkeep2 _ hnf1]; exact hreg1
    have h2 : regs3[(runLM (r b) F tag sa).1]? = some xb := by
      rw [← Std.HashMap.get?_eq_getElem?, hkeep3 _ hnf2]; exact hreg2
    have h3 : regs3[(runLM (r e) F tag sb).1]? = some xe := by
      rw [← Std.HashMap.get?_eq_getElem?]; exact hreg3
    simp [core, exec, coll, opnd, getReg_plain _ _ hns1, getReg_plain _ _ hns2, getReg_plain _ _ hns3, h1, h2, h3,
      show (Intr.mma == Intr.barrier) = false from rfl, hx, hm32 xa hna, hm32 xb hnb, hm32 xe hne, hm32 x hxn,
      MonadState.get, getThe, MonadStateOf.get, StateT.get, set, MonadStateOf.set, StateT.set, setReg, pure,
      StateT.pure, Except.pure, modify, modifyGet, MonadStateOf.modifyGet, StateT.modifyGet, o,
      StateT.bind, StateT.run, bind, Except.bind]
  refine ⟨{ ts with regs := regs3.insert (regOf .i32 d) x, w := { ts.w with obs := o :: ts.w.obs } }, ?_, ?_,
    wrel_step start0 _ s.world ts.w _ _ o o hW rfl rfl rfl (by simpa using hW.1)⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _
      (by rw [steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]; exact hrun3)]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ (.call (some d) (.intr .mma) [a, b, e]) d .i32 x rfl hS htd hxn
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?, enc, hxn]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; cases h8
    · intro h8; cases h8
    · intro c' n hn
      have hne' : PtxIr.Reg.val c' n ≠ regOf .i32 d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne'),
        ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep3 _ (fun ⟨p, k, _, h⟩ => by cases h), hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h),
        hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]

/-- A ballot lowers to its argument's lines, whether it is not zero into a
    predicate of its own, then a vote on that predicate. -/
theorem lower_ballot (F : Facts) (tag : Nat) (d v : Val) :
    (lower F tag (.call (some d) (.intr .ballot) [v])).lines =
      ((runLM (r v) F tag {}).2.lines).push
        (.ins none (.vote (regOf .i32 d) (.reg (runLM (r v) F tag {}).1) (.imm 0xffffffff))) := by
  rw [lower_inst F tag _ rfl, inst]
  simp [o, emit, emitL]

open PtxRun in
/-- **A ballot** meets its obligation: both semantics record it, with
    arguments a launch reads alike, whether each is not zero, and take the
    answer at the same place among the thread's. -/
theorem ballot_ok (F : Facts) (c : PtxRun.Ctx) (start0 : Nat) (hans : AnsOk c.ans)
    (d v : Val) (hwv : ¬ Win F v) (hwd : F.win d = false)
    (htv : F.tys.get? v.id = some .i8) (htd : F.tys.get? d.id = some .i32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.call (some d) (.intr .ballot) [v]) = true) :
    InstOk F (threadLocals c.ans start0 c.hw c.provisional) c start0 (.call (some d) (.intr .ballot) [v]) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have huv : v ∈ Inst.uses (.call (some d) (.intr .ballot) [v]) := by simp [Inst.uses]
  obtain ⟨tv, xv, hvv, htv'⟩ := hS.pres v (huses v huv)
  rw [htv] at htv'; cases htv'
  have hfl := hS.flag v xv (huses v huv) hvv
  have hlen := collCalls_wrel start0 _ s.world ts.w hW
  have hle : start0 ≤ s.world.obs.reverse.length := by simpa using hW.2.1
  -- the block semantics' step
  let o : Obs := .call (.intr .ballot) [.sc .i8 xv]
  obtain ⟨x, hx, hxn⟩ := valAt_ok c.ans hans (collCalls ts.w c.start).length
  have hs1 : s1 = { vals := Blocks.setV s.vals d (.sc .i32 x), world := { s.world with obs := o :: s.world.obs } } := by
    simp only [step1, List.mapM_cons, List.mapM_nil, hvv, bind, Option.bind, pure, callOf, obsCall, callWords,
      List.append_nil, threadLocals, Intr.find_slot, Intr.isColl, if_true] at hst
    simp only [collCalls, List.reverse_cons, List.drop_append_of_le_length hle, List.filterMap_append, collOf,
      Intr.isColl, if_true, List.filterMap_cons, List.filterMap_nil, List.length_append, List.length_singleton,
      Nat.add_sub_cancel] at hst
    have hk : ((s.world.obs.reverse.drop start0).filterMap collOf).length = (collCalls ts.w c.start).length := hlen
    rw [hk, hx] at hst
    simp [Except.toOption, show Intr.ballot ≠ .barrier by decide, show Intr.ballot ≠ .sharedAdd by decide, hx] at hst
    exact hst.symm
  subst hs1
  -- the lines: the flag's predicate, then one `vote`
  obtain ⟨Lv, hLv, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag {} s.vals c ts v .i8 xv D hS (huses v huv) hvv htv hwv
  rw [enc_flag xv hfl] at hreg1
  let vt : PtxIr.Instr := .vote (regOf .i32 d) (.reg (runLM (r v) F tag {}).1) (.imm 0xffffffff)
  have hlist : (lower F tag (.call (some d) (.intr .ballot) [v])).lines =
      ((Lv ++ [vt]).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_ballot F tag d v, hLv]; simp [vt]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hm32 : PtxRun.mask 32 x = x := by
    rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp [IntT]), hxn]
  let o' : Obs := .call (.intr .ballot) [.sc .i32 (xv &&& 1)]
  have hR := hreg1
  simp only [Std.HashMap.get?_eq_getElem?] at hR
  have hvt : (exec c vt).run { ts with regs := regs1 } =
      .ok (.step, { ts with regs := regs1.insert (regOf .i32 d) x, w := { ts.w with obs := o' :: ts.w.obs } }) := by
    simp [vt, exec, coll, opnd, getReg_plain _ _ hns1, hR, hx, hm32,
      show (Intr.ballot == Intr.barrier) = false from rfl, MonadState.get, getThe, MonadStateOf.get, StateT.get, set,
      MonadStateOf.set, StateT.set, setReg, pure, StateT.pure, Except.pure, modify, modifyGet, MonadStateOf.modifyGet,
      StateT.modifyGet, o', StateT.bind, StateT.run, bind, Except.bind]
  refine ⟨{ ts with regs := regs1.insert (regOf .i32 d) x, w := { ts.w with obs := o' :: ts.w.obs } },
    ?_, ?_, wrel_step start0 _ s.world ts.w _ _ o o' hW rfl rfl ?_ (by simpa using hW.1)⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _ hrun1]
    simp only [StateT.run] at hvt
    simp [steps, StateT.run, hvt, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ (.call (some d) (.intr .ballot) [v]) d .i32 x rfl hS htd hxn
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?, enc, hxn]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; cases h8
    · intro h8; cases h8
    · intro c' n hn
      have hne : PtxIr.Reg.val c' n ≠ regOf .i32 d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        ← Std.HashMap.get?_eq_getElem?]
      exact hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)
  · rcases flag_cases xv hfl with rfl | rfl <;>
      simp [viewObs, o, o', ballotBit, asBits, Intr.isColl, show (Intr.ballot == Intr.ballot) = true from rfl]
    all_goals first | rfl | trivial | decide

/-- A block world adding an observation a launch does not read stays related
    to a thread world that adds none. -/
theorem wrel_skip (bs ps : Nat) (bw pw : World) (o : Obs) (h : WRel bs ps bw pw) (hv : viewObs o = none) :
    WRel bs ps { bw with obs := o :: bw.obs } pw := by
  obtain ⟨h1, h3, h4, h5⟩ := h
  refine ⟨by simpa using h1, by simp; omega, h4, ?_⟩
  simp only [List.reverse_cons]
  rw [List.drop_append_of_le_length (by simpa using h3), List.filterMap_append, h5]
  simp [hv]

/-- **The hardware is a GPU's**: each approximation gives 32 bits, and a NaN
    result at f32 is its one NaN. -/
structure HwOk (hw : Hw) : Prop where
  /-- Each approximation gives 32 bits. -/
  approx : ∀ c x, hw.approx c x &&& widthMask .f32 = hw.approx c x
  /-- A NaN result at f32 is the GPU's one NaN. -/
  nan : ∀ x, hw.nan .f32 x &&& widthMask .f32 = 0x7fffffff

/-! ### Float arithmetic -/

/-- **On a GPU, PTX's one NaN is the platform's**: an f32 result `canon`
    leaves is the one the NaN rule leaves. -/
theorem canon_nanFix {hw : Hw} (hhw : HwOk hw) (r : UInt64) (hr : r.toNat < 2 ^ 32) :
    PtxRun.canon r = nanFix hw.nan .f32 r := by
  unfold PtxRun.canon nanFix
  split
  · rw [hhw.nan]
  · rfl

/-- An f32 reads only its word's low 32 bits. -/
theorem f32_mask (y : UInt64) : f32 (PtxRun.mask 32 y) = f32 y := by
  simp only [f32]
  congr 1
  apply UInt32.toNat_inj.mp
  rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp)]
  simp only [UInt64.toNat_toUInt32, toNat_wm .i32 (by simp), ClifTy.width, Nat.mod_mod]

open PtxRun in
/-- **A float operation on two f32s** meets its obligation, on a GPU: both
    semantics give the operation's word `g`, a NaN the GPU's one. -/
theorem fbin_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (i : Inst) (d a b : Val)
    (bo : PtxIr.BinOp) (rnd : Option PtxIr.Rnd) (g : UInt64 → UInt64 → UInt64) (hhw : HwOk c.hw)
    (hg : ∀ x y, x.toNat < 2 ^ 32 → y.toNat < 2 ^ 32 → (g x y).toNat < 2 ^ 32)
    (hd : Inst.dst? i = some d) (hua : a ∈ Inst.uses i) (hub : b ∈ Inst.uses i)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → t = .f32 ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hstep : ∀ (s s1 : Blocks.BSt), step1 lc s i = some s1 → ∀ xa, Blocks.getV s.vals a = some (.sc .f32 xa) →
      ∃ xb, Blocks.getV s.vals b = some (.sc .f32 xb) ∧
        s1 = { s with vals := Blocks.setV s.vals d (.sc .f32 (s.world.mem.fl .f32 (g xa xb))) })
    (hlow : ∀ tag t, F.tys.get? a.id = some t → (lower F tag i).lines =
      ((runLM (imm b) F tag (runLM (r a) F tag {}).2).2.lines).push
        (.ins none (.bin bo rnd .f32 (regOf t d) (.reg (runLM (r a) F tag {}).1)
          (runLM (imm b) F tag (runLM (r a) F tag {}).2).1)))
    (hbin : ∀ x y, x.toNat < 2 ^ 32 → y.toNat < 2 ^ 32 → PtxRun.binF bo x y = .ok (PtxRun.canon (g x y)))
    (hmo : magOk F d i = true) (hnr : magRule F.origin F.mag F.lits F.tys i = none) :
    InstOk F lc c start0 i := by
  apply bin2_ok (hmo := hmo) (hrule := fun _ _ _ _ _ _ _ _ _ hk => by rw [hnr] at hk; cases hk) F lc c start0 i d a b bo rnd (fun _ => .f32)
    (fun _ x y => nanFix c.hw.nan .f32 (g x y)) (· = .f32) (fun t h => by subst h; simp [Scalar])
    (fun _ _ _ => True) hd hua hub hwa hwb hwd hT hld
  · intro s s1 hst hsn t xa ht hva
    subst ht
    obtain ⟨xb, hb, h1⟩ := hstep s s1 hst xa hva
    refine ⟨xb, hb, trivial, ?_⟩
    rw [h1]
    show _ = { s with vals := Blocks.setV s.vals d (.sc .f32 (nanFix c.hw.nan .f32 _)) }
    rw [← hsn]; rfl
  · exact hlow
  · intro t ht h8; subst ht; cases h8
  · intro t _ _ ht h8; subst ht; cases h8
  · intro t xa xb y ht _ hxa hxb _ hy c' ts' xo yo dr hxo hyo
    subst ht
    have hea : enc .f32 xa = xa := by simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hxa]
    have heb : PtxRun.mask 32 y = xb := by
      have := hy; simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hxb] at this; exact this
    rw [hea] at hxo
    have ha32 := lt32_of_norm hxa
    have hb32 := lt32_of_norm hxb
    have hr := hg xa xb ha32 hb32
    have hv : PtxRun.bin bo .f32 xa y = .ok (enc .f32 (nanFix c.hw.nan .f32 (g xa xb))) := by
      simp only [PtxRun.bin, PtxIr.Ty.isFloat, if_true, bne_self_eq_false, Bool.false_eq_true, if_false]
      rw [mask32_lt _ ha32, heb, hbin _ _ ha32 hb32, canon_nanFix hhw _ hr]
      simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false]
      rw [and_f32_lt _ (nanFix_lt _ hr)]
    exact exec_bin c' ts' bo .f32 dr xo yo xa y _ hxo hyo hv
  · intro t xa xb ht hxa hxb
    subst ht
    exact and_f32_lt _ (nanFix_lt _ (hg xa xb (lt32_of_norm hxa) (lt32_of_norm hxb)))

open PtxRun in
theorem fadd_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val) (hhw : HwOk c.hw)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → t = .f32 ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fadd d a b) = true) :
    InstOk F lc c start0 (.fadd d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply fbin_ok (hmo := hmo) (hnr := rfl) F lc c start0 (.fadd d a b) d a b .add (some .rn) (fun x y => ofF32 (f32 x + f32 y)) hhw (fun _ _ _ _ => ofF32_lt _) rfl (by simp [Inst.uses]) (by simp [Inst.uses])
    hwa hwb hwd hT hld
  · intro s s1 hst xa hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipF] at hst
    | sc tb xb =>
    cases tb <;> simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipF] at hst
    exact ⟨xb, rfl, hst.symm⟩
  · intro tag t ht
    obtain ⟨rfl, -⟩ := hT t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify, bits]
    all_goals rfl
  · intro x y _ _; rfl

open PtxRun in
theorem fsub_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val) (hhw : HwOk c.hw)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → t = .f32 ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fsub d a b) = true) :
    InstOk F lc c start0 (.fsub d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply fbin_ok (hmo := hmo) (hnr := rfl) F lc c start0 (.fsub d a b) d a b .sub (some .rn) (fun x y => ofF32 (f32 x - f32 y)) hhw (fun _ _ _ _ => ofF32_lt _) rfl (by simp [Inst.uses]) (by simp [Inst.uses])
    hwa hwb hwd hT hld
  · intro s s1 hst xa hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipF] at hst
    | sc tb xb =>
    cases tb <;> simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipF] at hst
    exact ⟨xb, rfl, hst.symm⟩
  · intro tag t ht
    obtain ⟨rfl, -⟩ := hT t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify, bits]
    all_goals rfl
  · intro x y _ _; rfl

open PtxRun in
theorem fmul_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val) (hhw : HwOk c.hw)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → t = .f32 ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fmul d a b) = true) :
    InstOk F lc c start0 (.fmul d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply fbin_ok (hmo := hmo) (hnr := rfl) F lc c start0 (.fmul d a b) d a b .mul (some .rn) (fun x y => ofF32 (f32 x * f32 y)) hhw (fun _ _ _ _ => ofF32_lt _) rfl (by simp [Inst.uses]) (by simp [Inst.uses])
    hwa hwb hwd hT hld
  · intro s s1 hst xa hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipF] at hst
    | sc tb xb =>
    cases tb <;> simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipF] at hst
    exact ⟨xb, rfl, hst.symm⟩
  · intro tag t ht
    obtain ⟨rfl, -⟩ := hT t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify, bits]
    all_goals rfl
  · intro x y _ _; rfl

open PtxRun in
theorem fdiv_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val) (hhw : HwOk c.hw)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → t = .f32 ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fbin d .fdiv a b) = true) :
    InstOk F lc c start0 (.fbin d .fdiv a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply fbin_ok (hmo := hmo) (hnr := rfl) F lc c start0 (.fbin d .fdiv a b) d a b .div (some .rn) (fun x y => ofF32 (f32 x / f32 y)) hhw (fun _ _ _ _ => ofF32_lt _) rfl (by simp [Inst.uses]) (by simp [Inst.uses])
    hwa hwb hwd hT hld
  · intro s s1 hst xa hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipF] at hst
    | sc tb xb =>
    cases tb <;> simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipF] at hst
    exact ⟨xb, rfl, hst.symm⟩
  · intro tag t ht
    obtain ⟨rfl, -⟩ := hT t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify, bits]
    all_goals rfl
  · intro x y _ _; rfl

/-- `fmax` and `fmin` give one of their operands, so f32 words stay f32 words. -/
theorem pick_lt (wantMax : Bool) (x y : UInt64) (hx : x.toNat < 2 ^ 32) (hy : y.toNat < 2 ^ 32) :
    (pickExtreme wantMax .f32 x y).toNat < 2 ^ 32 := by
  unfold pickExtreme; repeat' split
  all_goals assumption

open PtxRun in
theorem fmax_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val) (hhw : HwOk c.hw)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → t = .f32 ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fmax d a b) = true) :
    InstOk F lc c start0 (.fmax d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply fbin_ok (hmo := hmo) (hnr := rfl) F lc c start0 (.fmax d a b) d a b .maxNaN none (pickExtreme true .f32) hhw
    (fun x y => pick_lt true x y) rfl (by simp [Inst.uses]) (by simp [Inst.uses]) hwa hwb hwd hT hld
  · intro s s1 hst xa hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipBits, zipBitsIf] at hst
    | sc tb xb =>
    cases tb <;> simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipBits,
      zipBitsIf, ClifTy.isFloat] at hst
    exact ⟨xb, rfl, hst.symm⟩
  · intro tag t ht
    obtain ⟨rfl, -⟩ := hT t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify, bits]
    all_goals rfl
  · intro x y _ _; rfl

open PtxRun in
theorem fmin_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b : Val) (hhw : HwOk c.hw)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : ∀ t, F.tys.get? a.id = some t → t = .f32 ∧ F.tys.get? d.id = some t)
    (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fmin d a b) = true) :
    InstOk F lc c start0 (.fmin d a b) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  have hwb' : F.win b = false := by
    cases h : F.win b
    · rfl
    · exact absurd ((win_iff F b).mp h) hwb
  apply fbin_ok (hmo := hmo) (hnr := rfl) F lc c start0 (.fmin d a b) d a b .minNaN none (pickExtreme false .f32) hhw
    (fun x y => pick_lt false x y) rfl (by simp [Inst.uses]) (by simp [Inst.uses]) hwa hwb hwd hT hld
  · intro s s1 hst xa hva
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipBits, zipBitsIf] at hst
    | sc tb xb =>
    cases tb <;> simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get, zipBits,
      zipBitsIf, ClifTy.isFloat] at hst
    exact ⟨xb, rfl, hst.symm⟩
  · intro tag t ht
    obtain ⟨rfl, -⟩ := hT t ht
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, isWin, facts?, runLM_bind, runLM_read,
      runLM_pure, hwa', hwb', tyOf, ht, o, emit, emitL, runLM_modify, bits]
    all_goals rfl
  · intro x y _ _; rfl

open PtxRun in
/-- **A float square root** meets its obligation, on a GPU: both semantics give
    the f32 root, a NaN the GPU's one. -/
theorem sqrt_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val) (hhw : HwOk c.hw)
    (hwa : ¬ Win F a) (hwd : F.win d = false)
    (hta : F.tys.get? a.id = some .f32) (htd : F.tys.get? d.id = some .f32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fun1 d .sqrt a) = true) :
    InstOk F lc c start0 (.fun1 d .sqrt a) := by
  have hwa' : F.win a = false := by
    cases h : F.win a
    · rfl
    · exact absurd ((win_iff F a).mp h) hwa
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a (fun _ dr xo => .un .sqrt (some .rn) .f32 dr xo) (fun _ => .f32)
    (fun _ x => nanFix c.hw.nan .f32 (ofF32 (f32 x).sqrt)) (· = .f32) (fun _ _ h => by cases h) rfl
    (by simp [Inst.uses]) hwa hwd (fun t ht => by rw [hta] at ht; cases ht; exact htd) hld
  · intro s s1 hst hsn t xa hva ht
    rw [hta] at ht; cases ht
    refine ⟨rfl, ?_⟩
    simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.get, mapFBits, funOp,
      Mem.fl, hsn] at hst
    rw [← hst, and_f32_lt _ (nanFix_lt _ (ofF32_lt _))]
  · intro tag t ht
    rw [hta] at ht; cases ht
    have ht' : F.tys[a.id]? = some .f32 := by rw [← Std.HashMap.get?_eq_getElem?]; exact hta
    simp [lower_eq, isTerm, inst, emit, emitL, facts?, tyOf, ht', o, bits]
  · intro t xa ht hxa _ c' ts' xo dr hxo
    subst ht
    have hea : enc .f32 xa = xa := by simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hxa]
    refine exec_un c' ts' .sqrt (some .rn) .f32 dr xo _ _ hxo ?_
    rw [hea]
    simp only [PtxRun.un, PtxIr.Ty.width, beq_self_eq_true, if_true]
    show Except.ok (PtxRun.canon (ofF32 (f32 xa).sqrt)) = _
    rw [canon_nanFix hhw _ (ofF32_lt _)]
    simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false]
    rw [and_f32_lt _ (nanFix_lt _ (ofF32_lt _))]
  · intro t xa ht _
    subst ht
    exact and_f32_lt _ (nanFix_lt _ (ofF32_lt _))

open PtxRun in
theorem exec_fma {rnd : PtxIr.Rnd} (c : PtxRun.Ctx) (ts : PtxRun.TS) (dr : PtxIr.Reg) (xo yo zo : PtxIr.Opnd)
    (x y z : UInt64) (hx : OpVal c ts xo x) (hy : OpVal c ts yo y) (hz : OpVal c ts zo z) :
    (exec c (.fma (some rnd) .f32 dr xo yo zo)).run ts = .ok (.step, { ts with regs := ts.regs.insert dr (PtxRun.canon (fmaLane .f32 (PtxRun.mask 32 x) (PtxRun.mask 32 y) (PtxRun.mask 32 z))) }) := by
  simp only [OpVal] at hx hy hz
  simp only [exec, StateT.run, bind, StateT.bind] at hx hy hz ⊢
  simp [Except.bind, setReg, pure, StateT.pure, Except.pure, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet]
  simp [StateT.bind, StateT.pure, StateT.modifyGet, hx, hy, hz, Except.bind, Except.pure, bind, pure]

open PtxRun in
/-- **A fused multiply-add of f32s** meets its obligation, on a GPU: both
    semantics round `a · b + c` once, a NaN the GPU's one. -/
theorem fma_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a b e : Val) (hhw : HwOk c.hw)
    (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwe : ¬ Win F e) (hwd : F.win d = false)
    (hta : F.tys.get? a.id = some .f32) (htd : F.tys.get? d.id = some .f32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fma d a b e) = true) :
    InstOk F lc c start0 (.fma d a b e) := by
  intro tag D s s1 ts hst huses hS hW _ hnan
  have hua : a ∈ Inst.uses (.fma d a b e) := by simp [Inst.uses]
  have hub : b ∈ Inst.uses (.fma d a b e) := by simp [Inst.uses]
  have hue : e ∈ Inst.uses (.fma d a b e) := by simp [Inst.uses]
  have hsn : s.world.mem.nan = c.hw.nan := (congrArg (fun x : World => x.mem.nan) hW.1).trans hnan.1
  obtain ⟨ta, xa, hva, hta'⟩ := hS.pres a (huses a hua)
  rw [hta] at hta'; cases hta'
  -- the block semantics' step
  obtain ⟨xb, xe, hvb, hve, rfl⟩ : ∃ xb xe, Blocks.getV s.vals b = some (.sc .f32 xb) ∧
      Blocks.getV s.vals e = some (.sc .f32 xe) ∧
      s1 = { s with vals := Blocks.setV s.vals d (.sc .f32 (nanFix c.hw.nan .f32 (fmaLane .f32 xa xb xe))) } := by
    obtain ⟨tb, xb, hvb, -⟩ := hS.pres b (huses b hub)
    obtain ⟨te, xe, hve, -⟩ := hS.pres e (huses e hue)
    simp [step1, Blocks.evalInst, hva, hvb, hve, Blocks.viaOp, evalOp, Sem.get, Mem.fl, hsn] at hst
    cases tb <;> cases te <;> simp [ClifTy.isFloat] at hst
    exact ⟨xb, xe, hvb, hve, hst.symm⟩
  have htb : F.tys.get? b.id = some .f32 := by
    obtain ⟨tb, xb', hvb', htb⟩ := hS.pres b (huses b hub)
    rw [hvb] at hvb'; obtain ⟨rfl, -⟩ := V.sc.inj (Option.some.inj hvb'); exact htb
  have hte : F.tys.get? e.id = some .f32 := by
    obtain ⟨te, xe', hve', hte⟩ := hS.pres e (huses e hue)
    rw [hve] at hve'; obtain ⟨rfl, -⟩ := V.sc.inj (Option.some.inj hve'); exact hte
  have hsc : Scalar .f32 := by simp [Scalar]
  -- the operands' lines: `a` into a register, then `b` and `e`
  let s0 : Out := {}
  let sa := (runLM (r a) F tag s0).2
  let sb := (runLM (imm b) F tag sa).2
  obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag s0 s.vals c ts a .f32 xa D hS (huses a hua) hva hta hwa
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lb, hLb, hn2, regs2, y, hrun2, hop2, hy, hkeep2⟩ :=
    imm_sound F tag sa s.vals c { ts with regs := regs1 } b .f32 xb D hS1 (huses b hub) hvb htb hwb hsc
  have hS2 : Sim F D s.vals regs2 := hS1.frame (fun c' n => hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Le, hLe, hn3, regs3, z, hrun3, hop3, hz, hkeep3⟩ :=
    imm_sound F tag sb s.vals c { ts with regs := regs2 } e .f32 xe D hS2 (huses e hue) hve hte hwe hsc
  let X : PtxIr.Opnd := .reg (runLM (r a) F tag s0).1
  let Y := (runLM (imm b) F tag sa).1
  let Z := (runLM (imm e) F tag sb).1
  let core : PtxIr.Instr := .fma (some .rn) .f32 (regOf .f32 d) X Y Z
  have hlow : (lower F tag (.fma d a b e)).lines = ((runLM (imm e) F tag sb).2.lines).push (.ins none core) := by
    have ht' : F.tys[a.id]? = some .f32 := by rw [← Std.HashMap.get?_eq_getElem?]; exact hta
    simp only [lower_eq, isTerm, Bool.false_eq_true, if_false, inst, facts?, runLM_bind, runLM_read,
      runLM_pure, tyOf, hta, ht', o, emit, emitL, runLM_modify, bits]
    all_goals rfl
  have hlist : (lower F tag (.fma d a b e)).lines =
      ((La ++ Lb ++ Le ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow, hLe, hLb, hLa]
    simp [core, X, Y, Z, sa, sb, s0]
  have hins : instrsOf (lower F tag (.fma d a b e)).lines = La ++ Lb ++ Le ++ [core] := by
    rw [hlist, instrsOf_map]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  -- the operands' values where `fma` reads them
  have hna := hS.normd a .f32 xa (huses a hua) hva
  have hnb := hS.normd b .f32 xb (huses b hub) hvb
  have hne := hS.normd e .f32 xe (huses e hue) hve
  have hea : enc .f32 xa = xa := by simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hna]
  have hX : OpVal c { ts with regs := regs3 } X xa := by
    refine opVal_reg c _ _ _ hns1 ?_
    rw [hkeep3 _ (fun h => hnf1 (fresh_mono hn2 h)), hkeep2 _ hnf1, ← hea]; exact hreg1
  have hY : OpVal c { ts with regs := regs3 } Y y := hop2 _ (fun R hR => hkeep3 R hR)
  have hZ : OpVal c { ts with regs := regs3 } Z z := hop3 _ (fun _ _ => rfl)
  have hmy : PtxRun.mask 32 y = xb := by
    have := hy; simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hnb] at this; exact this
  have hmz : PtxRun.mask 32 z = xe := by
    have := hz; simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hne] at this; exact this
  have hr := fmaLane32_lt xa xb xe (lt32_of_norm hna) (lt32_of_norm hnb) (lt32_of_norm hne)
  let v : UInt64 := nanFix c.hw.nan .f32 (fmaLane .f32 xa xb xe)
  have hv : v &&& widthMask .f32 = v := and_f32_lt _ (nanFix_lt _ hr)
  have he : (exec c core).run { ts with regs := regs3 } =
      .ok (.step, { ts with regs := regs3.insert (regOf .f32 d) (enc .f32 v) }) := by
    rw [exec_fma c _ _ _ _ _ _ _ _ hX hY hZ, mask32_lt _ (lt32_of_norm hna), hmy, hmz, canon_nanFix hhw _ hr]
    simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, v, hv]
  refine ⟨{ ts with regs := regs3.insert (regOf .f32 d) (enc .f32 v) }, ?_, ?_, hW⟩
  · rw [hins, steps_append c _ _ ts _
      (by rw [steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]; exact hrun3)]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ (.fma d a b e) d .f32 v rfl hS htd hv
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; cases h8
    · intro h8; cases h8
    · intro c' n hn
      have hne' : PtxIr.Reg.val c' n ≠ regOf .f32 d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne'),
        ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep3 _ (fun ⟨p, k, _, h⟩ => by cases h), hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h),
        hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]

/-! ### Conversions to f32 -/

open PtxRun in
/-- **A signed integer to f32** meets its obligation: both semantics round the
    integer once. -/
theorem sitof_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false) (hint : ∀ t, F.tys.get? a.id = some t → IntT t)
    (htd : F.tys.get? d.id = some .f32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fcvtFromSint d .f32 a) = true) :
    InstOk F lc c start0 (.fcvtFromSint d .f32 a) := by
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a (fun t dr xo => .cvt (some .rn) .f32 (sTy t.width) dr xo) (fun _ => .f32)
    (fun t x => ofF32 (Float32.ofInt (signed t x))) IntT (fun _ _ h => by cases h) rfl (by simp [Inst.uses])
    hwa hwd (fun _ _ => htd) hld
  · intro s s1 hst _ t xa hva hta
    have hit := hint t hta
    refine ⟨hit, ?_⟩
    rcases hit with rfl | rfl | rfl <;>
      simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.get, ClifTy.isInt] at hst <;>
      exact hst.symm
  · intro tag t ht
    have ht' : F.tys[a.id]? = some t := by rw [← Std.HashMap.get?_eq_getElem?]; exact ht
    simp [lower_eq, isTerm, inst, emit, emitL, o, facts?, tyOf, ht', bits]
  · intro t xa ht hn _ c' ts' xo dr hxo
    have hne8 : t ≠ .i8 := by rcases ht with rfl | rfl | rfl <;> decide
    have hv : PtxRun.cvt (some .rn) .f32 (sTy t.width) (enc t xa) = .ok (enc .f32 (ofF32 (Float32.ofInt (signed t xa)))) := by
      simp only [enc, hne8, show ClifTy.f32 ≠ .i8 by decide, if_false, hn, and_f32_lt _ (ofF32_lt _)]
      rw [← toInt_signed' t ht xa]
      rcases ht with rfl | rfl | rfl <;> simp [PtxRun.cvt, sTy, ClifTy.width, PtxIr.Ty.isFloat]
    exact exec_cvt c' ts' _ _ _ dr xo _ _ hxo hv
  · intro t xa _ _; exact and_f32_lt _ (ofF32_lt _)

open PtxRun in
/-- **An unsigned integer to f32** meets its obligation: both semantics round
    the integer once. -/
theorem uitof_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false) (hint : ∀ t, F.tys.get? a.id = some t → t = .i32 ∨ t = .i64)
    (htd : F.tys.get? d.id = some .f32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fconv d .fromUint .f32 a) = true) :
    InstOk F lc c start0 (.fconv d .fromUint .f32 a) := by
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a (fun t dr xo => .cvt (some .rn) .f32 (uTy t.width) dr xo) (fun _ => .f32)
    (fun t x => ofF32 (Float32.ofNat (x &&& widthMask t).toNat)) (fun t => t = .i32 ∨ t = .i64)
    (fun _ _ h => by cases h) rfl (by simp [Inst.uses]) hwa hwd (fun _ _ => htd) hld
  · intro s s1 hst _ t xa hva hta
    have hit := hint t hta
    refine ⟨hit, ?_⟩
    rcases hit with rfl | rfl <;>
      simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.get, ClifTy.isInt,
        ClifTy.width] at hst <;>
      exact hst.symm
  · intro tag t ht
    have ht' : F.tys[a.id]? = some t := by rw [← Std.HashMap.get?_eq_getElem?]; exact ht
    simp [lower_eq, isTerm, inst, emit, emitL, o, facts?, tyOf, ht', bits]
  · intro t xa ht hn _ c' ts' xo dr hxo
    have hit : IntT t := by rcases ht with rfl | rfl <;> simp [IntT]
    have hne8 : t ≠ .i8 := by rcases ht with rfl | rfl <;> decide
    have hv : PtxRun.cvt (some .rn) .f32 (uTy t.width) (enc t xa) =
        .ok (enc .f32 (ofF32 (Float32.ofNat (xa &&& widthMask t).toNat))) := by
      have hu := toInt_u t hit xa hn
      simp only [enc, hne8, show ClifTy.f32 ≠ .i8 by decide, if_false, hn, and_f32_lt _ (ofF32_lt _)]
      rcases ht with rfl | rfl <;> simp [uTy, ClifTy.width] at hu <;>
        simp [PtxRun.cvt, uTy, ClifTy.width, PtxIr.Ty.isFloat, hu] <;> rfl
    exact exec_cvt c' ts' _ _ _ dr xo _ _ hxo hv
  · intro t xa _ _; exact and_f32_lt _ (ofF32_lt _)

open PtxRun in
/-- **An f64 to f32** meets its obligation, on a GPU: both semantics round the
    double once, a NaN the GPU's one. -/
theorem demote_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val) (hhw : HwOk c.hw)
    (hwa : ¬ Win F a) (hwd : F.win d = false) (hta : F.tys.get? a.id = some .f64)
    (htd : F.tys.get? d.id = some .f32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fconv d .demote .f32 a) = true) :
    InstOk F lc c start0 (.fconv d .demote .f32 a) := by
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a (fun _ dr xo => .cvt (some .rn) .f32 .f64 dr xo) (fun _ => .f32)
    (fun _ x => nanFix c.hw.nan .f32 (ofF32 (f64 x).toFloat32)) (· = .f64) (fun _ _ h => by cases h) rfl
    (by simp [Inst.uses]) hwa hwd (fun _ _ => htd) hld
  · intro s s1 hst hsn t xa hva ht
    rw [hta] at ht; cases ht
    refine ⟨rfl, ?_⟩
    simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.get, Mem.fl, hsn] at hst
    exact hst.symm
  · intro tag t ht
    rw [hta] at ht; cases ht
    have ht' : F.tys[a.id]? = some .f64 := by rw [← Std.HashMap.get?_eq_getElem?]; exact hta
    simp [lower_eq, isTerm, inst, emit, emitL, o, facts?, tyOf, ht', bits]
  · intro t xa ht hn _ c' ts' xo dr hxo
    subst ht
    have hr := ofF32_lt (f64 xa).toFloat32
    have hv : PtxRun.cvt (some .rn) .f32 .f64 (enc .f64 xa) = .ok (enc .f32 (nanFix c.hw.nan .f32 (ofF32 (f64 xa).toFloat32))) := by
      simp only [enc, show ClifTy.f64 ≠ .i8 by decide, show ClifTy.f32 ≠ .i8 by decide, if_false, hn]
      simp [PtxRun.cvt, PtxIr.Ty.isFloat]
      rw [show Float.ofBits xa = f64 xa from rfl, canon_nanFix hhw _ hr, and_f32_lt _ (nanFix_lt _ hr)]
    exact exec_cvt c' ts' _ _ _ dr xo _ _ hxo hv
  · intro t xa _ _; exact and_f32_lt _ (nanFix_lt _ (ofF32_lt _))

/-- The bits saturating truncation of an f32 to an i32 answers. -/
def ftosiBits (x : UInt64) : UInt64 :=
  match satToSint .i32 (f32 x).toFloat with
  | .sc _ v => v
  | .vec .. => 0

theorem satToSint_i32 (v : Float) : ∃ b, satToSint .i32 v = .sc .i32 b ∧ b &&& widthMask .i32 = b := by
  unfold satToSint; simp only [ofInt, norm]
  split <;> (try split) <;> (try split) <;> (try split) <;>
    exact ⟨_, rfl, by simp [UInt64.and_assoc]⟩

open PtxRun in
/-- **An f32 truncated to an i32** meets its obligation: both semantics are the
    saturating conversion, NaN to zero and out of range to the nearer end. -/
theorem ftosi_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false) (hta : F.tys.get? a.id = some .f32)
    (htd : F.tys.get? d.id = some .i32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fconv d .toSint .i32 a) = true) :
    InstOk F lc c start0 (.fconv d .toSint .i32 a) := by
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a
    (fun _ dr xo => .cvt (some .rzi) .s32 .f32 dr xo) (fun _ => .i32) (fun _ x => ftosiBits x) (· = .f32)
    (fun _ _ h => by cases h) rfl (by simp [Inst.uses]) hwa hwd (fun _ _ => htd) hld
  · intro s s1 hst _ t xa hva ht
    rw [hta] at ht; cases ht
    refine ⟨rfl, ?_⟩
    obtain ⟨b, hb, -⟩ := satToSint_i32 (f32 xa).toFloat
    simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.get, hb, ClifTy.isInt,
      ClifTy.width] at hst
    rw [← hst]; simp [ftosiBits, hb]
  · intro tag t ht
    rw [hta] at ht; cases ht
    have ht' : F.tys[a.id]? = some .f32 := by rw [← Std.HashMap.get?_eq_getElem?]; exact hta
    simp [lower_eq, isTerm, inst, emit, emitL, o, facts?, tyOf, ht', bits, sTy, ClifTy.width]
  · intro t xa ht hn _ c' ts' xo dr hxo
    subst ht
    obtain ⟨b, hb, hbn⟩ := satToSint_i32 (f32 xa).toFloat
    have hv : PtxRun.cvt (some .rzi) .s32 .f32 (enc .f32 xa) = .ok (enc .i32 (ftosiBits xa)) := by
      simp only [enc, show ClifTy.f32 ≠ .i8 by decide, show ClifTy.i32 ≠ .i8 by decide, if_false, hn]
      simp [PtxRun.cvt, PtxIr.Ty.isFloat, PtxIr.Ty.isSigned, PtxIr.Ty.width, PtxRun.f32b, ftosiBits, hb, hbn]
    exact exec_cvt c' ts' _ _ _ dr xo _ _ hxo hv
  · intro t xa _ _
    obtain ⟨b, hb, hbn⟩ := satToSint_i32 (f32 xa).toFloat
    simp [ftosiBits, hb, hbn]

/-! ### Float comparisons -/

theorem mask32_and (y : UInt64) : PtxRun.mask 32 y = y &&& 0xffffffff := by
  simp [PtxRun.mask]; rfl

/-- An f32's class and key read only its word's low 32 bits. -/
theorem f32_low (y : UInt64) :
    isNaNBits .f32 (PtxRun.mask 32 y) = isNaNBits .f32 y ∧ fkey .f32 (PtxRun.mask 32 y) = fkey .f32 y := by
  rw [mask32_and]
  have h1 : (y &&& 0xffffffff) &&& 0x7f800000 = y &&& 0x7f800000 := by rw [UInt64.and_assoc]; rfl
  have h2 : (y &&& 0xffffffff) &&& 0x007fffff = y &&& 0x007fffff := by rw [UInt64.and_assoc]; rfl
  have h3 : (y &&& 0xffffffff) &&& 0x7fffffff = y &&& 0x7fffffff := by rw [UInt64.and_assoc]; rfl
  have h4 : (y &&& 0xffffffff) &&& 0x80000000 = y &&& 0x80000000 := by rw [UInt64.and_assoc]; rfl
  refine ⟨by simp only [isNaNBits, h1, h2], ?_⟩
  simp only [fkey, signBit, h3, h4]

/-- **PTX's comparison is the Code's**: `fcc cc` on the bits gives the Code's
    comparison, `fcneg cc` its negation. -/
theorem fcmp_cmp (cc : FloatCC) (xa xb y : UInt64) (hy : PtxRun.mask 32 y = xb) :
    PtxRun.cmp (fcc cc) .f32 xa y = .ok (fcmpBits cc .f32 xa xb) ∧
      PtxRun.cmp (fcneg cc) .f32 xa y = .ok (!fcmpBits cc .f32 xa xb) := by
  obtain ⟨hn, hk⟩ := f32_low y
  rw [hy] at hn hk
  simp only [PtxRun.cmp, PtxIr.Ty.isFloat, if_true, ← hn, ← hk, fcmpBits]
  generalize isNaNBits ClifTy.f32 xa = p
  generalize isNaNBits ClifTy.f32 xb = q
  generalize fkey ClifTy.f32 xa = a
  generalize fkey ClifTy.f32 xb = b
  cases cc <;> cases p <;> cases q <;> simp [fcc, fcneg] <;> first | omega | rfl | (simp [bne]; done) | (cases a == b <;> rfl) | (by_cases h1 : a < b <;> by_cases h2 : b < a <;> by_cases h3 : a ≤ b <;> by_cases h4 : b ≤ a <;>
      (try simp [h1, h2, h3, h4]) <;> omega)

open PtxRun in
/-- **A float comparison** meets its obligation: both semantics compare the bits
    as IEEE 754 orders the values, the flag and its negation. -/
theorem fcmp_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d : Val) (cc : FloatCC)
    (a b : Val) (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwd : F.win d = false)
    (hT : F.tys.get? a.id = some .f32)
    (htd : F.tys.get? d.id = some .i8) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.fcmp d cc a b) = true) :
    InstOk F lc c start0 (.fcmp d cc a b) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have hua : a ∈ Inst.uses (.fcmp d cc a b) := by simp [Inst.uses]
  have hub : b ∈ Inst.uses (.fcmp d cc a b) := by simp [Inst.uses]
  obtain ⟨ta, xa, hva, hta⟩ := hS.pres a (huses a hua)
  rw [hT] at hta; cases hta
  have hta := hT
  have hne8 : ClifTy.f32 ≠ .i8 := by decide
  -- the block semantics' step
  obtain ⟨xb, hvb, rfl⟩ : ∃ xb, Blocks.getV s.vals b = some (.sc .f32 xb) ∧
      s1 = { s with vals := Blocks.setV s.vals d (.sc .i8 (if fcmpBits cc .f32 xa xb then 1 else 0)) } := by
    cases hb : Blocks.getV s.vals b with
    | none => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb] at hst
    | some vb =>
    cases vb with
    | vec _ _ => simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get] at hst
    | sc tb xb =>
    cases tb <;> simp [step1, Blocks.evalInst, Blocks.evalInst.bin, hva, hb, Blocks.viaOp, evalOp, Sem.get,
      boolV] at hst
    exact ⟨xb, rfl, hst.symm⟩
  have htb : F.tys.get? b.id = some .f32 := by
    obtain ⟨tb, xb', hvb', htb⟩ := hS.pres b (huses b hub)
    rw [hvb] at hvb'; obtain ⟨rfl, -⟩ := V.sc.inj (Option.some.inj hvb'); exact htb
  have hxa := hS.normd a .f32 xa (huses a hua) hva
  have hxb := hS.normd b .f32 xb (huses b hub) hvb
  -- the operands' lines
  let sa := (runLM (r a) F tag {}).2
  obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag {} s.vals c ts a .f32 xa D hS (huses a hua) hva hta hwa
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lb, hLb, hn2, regs2, yv, hrun2, hop2, hy, hkeep2⟩ :=
    imm_sound F tag sa s.vals c { ts with regs := regs1 } b .f32 xb D hS1 (huses b hub) hvb htb hwb
      (by simp [Scalar])
  let X : PtxIr.Opnd := .reg (runLM (r a) F tag {}).1
  let Y : PtxIr.Opnd := (runLM (imm b) F tag sa).1
  let core1 : PtxIr.Instr := .setp (fcc cc) .f32 (regOf .i8 d) X Y
  let core2 : PtxIr.Instr := .setp (fcneg cc) .f32 (negReg d) X Y
  have hlow : (lower F tag (.fcmp d cc a b)).lines =
      (((runLM (imm b) F tag sa).2.lines).push (.ins none core1)).push (.ins none core2) := by
    have ht' : F.tys[a.id]? = some .f32 := by rw [← Std.HashMap.get?_eq_getElem?]; exact hta
    simp [lower_eq, isTerm, inst, emit, emitL, facts?, tyOf, ht', core1, core2, X, Y, sa, o, bits]
  have hlist : (lower F tag (.fcmp d cc a b)).lines =
      ((La ++ Lb ++ [core1, core2]).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow, hLb]
    show ((sa.lines ++ _).push _).push _ = _
    rw [hLa]; simp
  have hins : instrsOf (lower F tag (.fcmp d cc a b)).lines = La ++ Lb ++ [core1, core2] := by
    rw [hlist, instrsOf_map]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  -- the comparisons
  have hX : OpVal c { ts with regs := regs2 } X (enc .f32 xa) :=
    opVal_reg c _ _ _ hns1 (by rw [hkeep2 _ hnf1]; exact hreg1)
  have hY : OpVal c { ts with regs := regs2 } Y yv := hop2 _ (fun _ _ => rfl)
  have hea : enc .f32 xa = xa := by simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hxa]
  have hmy : PtxRun.mask 32 yv = xb := by
    have := hy; simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hxb, bits] at this; exact this
  obtain ⟨hq1, hq2⟩ := fcmp_cmp cc xa xb yv hmy
  rw [hea] at hX
  -- the operands' registers are neither the flag's nor its negation's
  have hXne : ∀ R, X = .reg R → R ≠ regOf .i8 d := by
    intro R hR; simp only [X, PtxIr.Opnd.reg.injEq] at hR; subst hR
    exact (r_not_flag F tag {} a .f32 hwa hta hne8 d.id).1
  have hYne : ∀ R, Y = .reg R → R ≠ regOf .i8 d := by
    intro R hR
    cases hlb : F.lits.get? b.id with
    | some l =>
        obtain ⟨hil, -⟩ := hS.lit b .f32 xb l (huses b hub) hvb htb hlb
        have := (imm_eq F tag sa b).2 l hlb
        simp only [Y] at hR; rw [this] at hR; subst hR; cases hil
    | none =>
        have := (imm_eq F tag sa b).1 hlb
        simp only [Y, this, PtxIr.Opnd.reg.injEq] at hR; subst hR
        exact (r_not_flag F tag sa b .f32 hwb htb hne8 d.id).1
  let v1 : UInt64 := if fcmpBits cc .f32 xa xb then 1 else 0
  let v2 : UInt64 := if !fcmpBits cc .f32 xa xb then 1 else 0
  have he1 : (exec c core1).run { ts with regs := regs2 } =
      .ok (.step, { ts with regs := regs2.insert (regOf .i8 d) v1 }) :=
    exec_setp c _ _ _ _ _ _ _ _ _ hX hY hq1
  have he2 : (exec c core2).run { ts with regs := regs2.insert (regOf .i8 d) v1 } =
      .ok (.step, { ts with regs := (regs2.insert (regOf .i8 d) v1).insert (negReg d) v2 }) :=
    exec_setp c { ts with regs := regs2.insert (regOf .i8 d) v1 } _ _ _ _ _ _ _ _
      (opVal_insert c _ X _ _ v1 hX hXne) (opVal_insert c _ Y _ _ v1 hY hYne) hq2
  refine ⟨{ ts with regs := (regs2.insert (regOf .i8 d) v1).insert (negReg d) v2 }, ?_, ?_, hW⟩
  · rw [hins, steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]
    simp only [StateT.run] at he1 he2
    simp [steps, StateT.run, he1, he2, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · have hpq : negReg d ≠ regOf .i8 d := by simp [negReg, regOf, cls]
    apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ _ d .i8 v1 rfl hS htd
    · (by_cases h : fcmpBits cc .f32 xa xb <;> simp [v1, h, widthMask]) <;> decide
    · intro _ _
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using hpq),
        Std.HashMap.getElem?_insert_self]
      (by_cases h : fcmpBits cc .f32 xa xb <;> simp [v1, h, enc]) <;> decide
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro _ _
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert_self]
      (by_cases h : fcmpBits cc .f32 xa xb <;> simp [v1, v2, h]) <;> decide
    · intro _; (by_cases h : fcmpBits cc .f32 xa xb <;> simp [v1, h]) <;> decide
    · intro c' n hn
      have hne1 : PtxIr.Reg.val c' n ≠ regOf .i8 d := by simp [regOf]; omega
      have hne2 : PtxIr.Reg.val c' n ≠ negReg d := by simp [negReg]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne2),
        Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne1), ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h), hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]

open PtxRun in
/-- **`dp4a` or `prmt`** meets its obligation: both semantics give the law's
    value (`dp4aVal`, `prmtVal`) of the operands' words, and only the block
    semantics records the call, which a launch does not read. -/
theorem word3_ok (F : Facts) (c : PtxRun.Ctx) (start0 : Nat) (i : Intr) (f : UInt64 → UInt64 → UInt64 → UInt64)
    (mk : PtxIr.Reg → PtxIr.Opnd → PtxIr.Opnd → PtxIr.Opnd → PtxIr.Instr)
    (hi : (i = .dp4a ∧ f = Intr.dp4aVal ∧ mk = PtxIr.Instr.dp4a) ∨ (i = .prmt ∧ f = Intr.prmtVal ∧ mk = PtxIr.Instr.prmt))
    (d a b e : Val) (hwa : ¬ Win F a) (hwb : ¬ Win F b) (hwe : ¬ Win F e) (hwd : F.win d = false)
    (hta : F.tys.get? a.id = some .i32) (htb : F.tys.get? b.id = some .i32) (hte : F.tys.get? e.id = some .i32)
    (htd : F.tys.get? d.id = some .i32) (hld : F.lits.get? d.id = none)
    (hlow : ∀ tag, (lower F tag (.call (some d) (.intr i) [a, b, e])).lines =
      ((runLM (r e) F tag (runLM (r b) F tag (runLM (r a) F tag {}).2).2).2.lines).push
        (.ins none (mk (regOf .i32 d) (.reg (runLM (r a) F tag {}).1)
          (.reg (runLM (r b) F tag (runLM (r a) F tag {}).2).1)
          (.reg (runLM (r e) F tag (runLM (r b) F tag (runLM (r a) F tag {}).2).2).1))))
    (hmo : magOk F d (.call (some d) (.intr i) [a, b, e]) = true) :
    InstOk F (threadLocals c.ans start0 c.hw c.provisional) c start0 (.call (some d) (.intr i) [a, b, e]) := by
  have hm : ∀ y : UInt64, (y &&& 0xffffffff) &&& widthMask .i32 = y &&& 0xffffffff := fun y => by
    rw [show widthMask .i32 = (0xffffffff : UInt64) from rfl, UInt64.and_assoc, UInt64.and_self]
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have hua : a ∈ Inst.uses (.call (some d) (.intr i) [a, b, e]) := by simp [Inst.uses]
  have hub : b ∈ Inst.uses (.call (some d) (.intr i) [a, b, e]) := by simp [Inst.uses]
  have hue : e ∈ Inst.uses (.call (some d) (.intr i) [a, b, e]) := by simp [Inst.uses]
  obtain ⟨ta, xa, hva, hta'⟩ := hS.pres a (huses a hua)
  rw [hta] at hta'; cases hta'
  obtain ⟨tb, xb, hvb, htb'⟩ := hS.pres b (huses b hub)
  rw [htb] at htb'; cases htb'
  obtain ⟨te, xe, hve, hte'⟩ := hS.pres e (huses e hue)
  rw [hte] at hte'; cases hte'
  have hna := hS.normd a .i32 xa (huses a hua) hva
  have hnb := hS.normd b .i32 xb (huses b hub) hvb
  have hne := hS.normd e .i32 xe (huses e hue) hve
  -- the block semantics' step
  let ob : Obs := .call (.intr i) [.sc .i32 xa, .sc .i32 xb, .sc .i32 xe]
  have hs1 : s1 = { vals := Blocks.setV s.vals d (.sc .i32 (f xa xb xe)), world := { s.world with obs := ob :: s.world.obs } } := by
    rcases hi with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ <;>
    · simp [step1, hva, hvb, hve, callOf, obsCall, callWords, threadLocals, Intr.find_slot, Intr.isColl,
        Except.toOption] at hst
      exact hst.symm
  subst hs1
  -- the lines
  let sa := (runLM (r a) F tag {}).2
  let sb := (runLM (r b) F tag sa).2
  obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag {} s.vals c ts a .i32 xa D hS (huses a hua) hva hta hwa
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lb, hLb, hn2, hnf2, hns2, regs2, hrun2, hreg2, hkeep2⟩ :=
    r_sound F tag sa s.vals c { ts with regs := regs1 } b .i32 xb D hS1 (huses b hub) hvb htb hwb
  have hS2 : Sim F D s.vals regs2 := hS1.frame (fun c' n => hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Le, hLe, hn3, hnf3, hns3, regs3, hrun3, hreg3, hkeep3⟩ :=
    r_sound F tag sb s.vals c { ts with regs := regs2 } e .i32 xe D hS2 (huses e hue) hve hte hwe
  simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hna] at hreg1
  simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hnb] at hreg2
  simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hne] at hreg3
  let core : PtxIr.Instr := mk (regOf .i32 d) (.reg (runLM (r a) F tag {}).1) (.reg (runLM (r b) F tag sa).1)
    (.reg (runLM (r e) F tag sb).1)
  have hlist : (lower F tag (.call (some d) (.intr i) [a, b, e])).lines =
      ((La ++ Lb ++ Le ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [hlow tag, hLe]
    show (sb.lines ++ _).push _ = _
    rw [hLb]
    show ((sa.lines ++ _) ++ _).push _ = _
    rw [hLa]; simp [core, sa, sb]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have h1 : regs3[(runLM (r a) F tag {}).1]? = some xa := by
    rw [← Std.HashMap.get?_eq_getElem?, hkeep3 _ (fun h => hnf1 (fresh_mono hn2 h)), hkeep2 _ hnf1]; exact hreg1
  have h2 : regs3[(runLM (r b) F tag sa).1]? = some xb := by
    rw [← Std.HashMap.get?_eq_getElem?, hkeep3 _ hnf2]; exact hreg2
  have h3 : regs3[(runLM (r e) F tag sb).1]? = some xe := by
    rw [← Std.HashMap.get?_eq_getElem?]; exact hreg3
  let x := f xa xb xe
  have hxn : x &&& widthMask .i32 = x := by
    rcases hi with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ <;> exact hm _
  have he : (exec c core).run { ts with regs := regs3 } =
      .ok (.step, { ts with regs := regs3.insert (regOf .i32 d) x }) := by
    rcases hi with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ <;>
    simp [core, exec, opnd, getReg_plain _ _ hns1, getReg_plain _ _ hns2, getReg_plain _ _ hns3, h1, h2, h3, x,
      MonadState.get, getThe, MonadStateOf.get, StateT.get, set, MonadStateOf.set, StateT.set, setReg, pure,
      StateT.pure, Except.pure, modify, modifyGet, MonadStateOf.modifyGet, StateT.modifyGet,
      StateT.bind, StateT.run, bind, Except.bind]
  refine ⟨{ ts with regs := regs3.insert (regOf .i32 d) x }, ?_, ?_,
    wrel_skip start0 _ s.world ts.w ob hW (by
      rcases hi with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ <;> simp [viewObs, ob, Intr.isColl])⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _
      (by rw [steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]; exact hrun3)]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _
      (.call (some d) (.intr i) [a, b, e]) d .i32 x rfl hS htd hxn
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?, enc, hxn]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; cases h8
    · intro h8; cases h8
    · intro c' n hn
      have hne' : PtxIr.Reg.val c' n ≠ regOf .i32 d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne'),
        ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep3 _ (fun ⟨p, k, _, h⟩ => by cases h), hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h),
        hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]

open PtxRun in
/-- **One of the hardware's approximations** meets its obligation: both
    semantics give the hardware's bits on the argument's, and only the block
    semantics records the call, which a launch does not read. -/
theorem approx_ok (F : Facts) (c : PtxRun.Ctx) (start0 : Nat) (hhw : HwOk c.hw)
    (d v : Val) (i : Intr) (u : PtxIr.UnOp) (hu : Ptx.approxOp i = some u)
    (hwv : ¬ Win F v) (hwd : F.win d = false)
    (htv : F.tys.get? v.id = some .f32) (htd : F.tys.get? d.id = some .f32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.call (some d) (.intr i) [v]) = true) :
    InstOk F (threadLocals c.ans start0 c.hw c.provisional) c start0 (.call (some d) (.intr i) [v]) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have huv : v ∈ Inst.uses (.call (some d) (.intr i) [v]) := by simp [Inst.uses]
  obtain ⟨tv, xv, hvv, htv'⟩ := hS.pres v (huses v huv)
  rw [htv] at htv'; cases htv'
  have hnv := hS.normd v .f32 xv (huses v huv) hvv
  have hcoll : i.isColl = false := by revert hu; cases i <;> simp [Ptx.approxOp, Intr.isColl]
  -- the block semantics' step
  let ob : Obs := .call (.intr i) [.sc .f32 xv]
  have hs1 : s1 = { vals := Blocks.setV s.vals d (.sc .f32 (c.hw.approx i xv)), world := { s.world with obs := ob :: s.world.obs } } := by
    simp [step1, hvv, callOf, obsCall, callWords, threadLocals, Intr.find_slot, hcoll, Except.toOption] at hst
    exact hst.symm
  subst hs1
  -- the lines
  obtain ⟨Lv, hLv, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
    r_sound F tag {} s.vals c ts v .f32 xv D hS (huses v huv) hvv htv hwv
  simp only [enc, show ClifTy.f32 ≠ .i8 by decide, if_false, hnv] at hreg1
  let core : PtxIr.Instr := .un u none .f32 (regOf .f32 d) (.reg (runLM (r v) F tag {}).1)
  have hlist : (lower F tag (.call (some d) (.intr i) [v])).lines = ((Lv ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_inst F tag _ rfl]
    revert hu; cases i <;> intro hu <;> simp [Ptx.approxOp] at hu <;> subst hu <;>
      simp [inst, Ptx.approxOp, Ptx.o, emit, emitL, hLv, core]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hun : PtxRun.un u c.hw .f32 xv = .ok (c.hw.approx i xv) := by
    revert hu; cases i <;> intro hu <;> simp [Ptx.approxOp] at hu <;> subst hu <;> rfl
  have he : (exec c core).run { ts with regs := regs1 } =
      .ok (.step, { ts with regs := regs1.insert (regOf .f32 d) (c.hw.approx i xv) }) :=
    exec_un c _ u none .f32 _ _ xv _ (opVal_reg c _ _ _ hns1 hreg1) hun
  refine ⟨{ ts with regs := regs1.insert (regOf .f32 d) (c.hw.approx i xv) }, ?_, ?_,
    wrel_skip start0 _ s.world ts.w ob hW (by simp [viewObs, ob, hcoll])⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _ hrun1]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ (.call (some d) (.intr i) [v]) d .f32 _ rfl hS htd (hhw.approx i xv)
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?, enc, hhw.approx i xv]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; cases h8
    · intro h8; cases h8
    · intro c' n hn
      have hne : PtxIr.Reg.val c' n ≠ regOf .f32 d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        ← Std.HashMap.get?_eq_getElem?]
      exact hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)

/-- A typed store: the address's lines, the value's, then one `st` of its type. -/
theorem lower_storeTyped (F : Facts) (tag : Nat) (ty : ClifTy) (v a : Val) :
    (lower F tag (.storeTyped ty v a)).lines =
      ((runLM (r v) F tag (runLM (addr a) F tag {}).2).2.lines).push
        (.ins none (.st (runLM (addr a) F tag {}).1.1 (bits ty) (runLM (addr a) F tag {}).1.2
          (.reg (runLM (r v) F tag (runLM (addr a) F tag {}).2).1))) := by
  rw [lower_inst F tag _ rfl, inst]
  rfl

open PtxRun in
/-- **A typed store of a value at its own type** through a global or generic
    address meets its obligation, as a value's store does. -/
theorem storeTyped_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (ty : ClifTy) (v a : Val)
    (hwv : ¬ Win F v)
    (hta : F.tys.get? a.id = some .i64) (htv : F.tys.get? v.id = some ty) (hty : LdTy ty) :
    InstOk F lc c start0 (.storeTyped ty v a) := by
  apply st_ok F lc c start0 _ v a bits tyBytes (fun _ b => b) (by simp [Inst.uses]) (by simp [Inst.uses]) rfl
    hwv hta (fun t ht => by rw [htv] at ht; cases ht; exact hty)
  · intro s s1 hst t b xa hvv htvt hva
    rw [htv] at htvt; cases htvt
    simp only [step1, Blocks.doStore, hvv, hva, Option.getD_some] at hst
    revert hst
    generalize s.world.mem.store xa (tyBytes ty) b = om
    intro hst
    cases om with
    | none => simp [Except.toOption] at hst
    | some m => simp [Except.toOption] at hst; exact ⟨m, rfl, hst.symm⟩
  · intro tag t ht
    rw [htv] at ht; cases ht
    exact lower_storeTyped F tag ty v a
  · intro t b hlt hbn
    have hne8 : t ≠ .i8 := by rcases hlt with h | h | h | h | h <;> rw [h] <;> decide
    refine ⟨by rcases hlt with h | h | h | h | h <;> rw [h] <;> rfl, ?_⟩
    have := mask_enc t b; simpa [enc, hne8, hbn] using this

/-! ## Every instruction -/

/-- An instruction whose step fails from every state the registers hold meets
    its obligation: there is no step to follow. -/
theorem instOk_of_none (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (i : Inst)
    (h : ∀ (D : Nat → Prop) (s : Blocks.BSt) (regs : Std.HashMap PtxIr.Reg UInt64),
      Sim F D s.vals regs → (∀ v ∈ Inst.uses i, D v.id) → step1 lc s i = none) :
    InstOk F lc c start0 i := by
  intro tag D s s1 ts hst huses hS _ _ _hnan
  rw [h D s ts.regs hS huses] at hst; cases hst

/-- A value's type, by the types check, is a scalar. -/
theorem tysOk_scalar (F : Facts) (h : tysOk F = true) (v : Val) (t : ClifTy) (ht : F.tys.get? v.id = some t) :
    Scalar t := by
  simp only [tysOk, List.all_eq_true] at h
  have hm : (v.id, t) ∈ F.tys.toList := by
    rw [Std.HashMap.mem_toList_iff_getElem?_eq_some, ← Std.HashMap.get?_eq_getElem?]; exact ht
  have := h _ hm
  simp [scalarTys] at this
  rcases this with rfl | rfl | rfl | rfl | rfl | rfl <;> simp [Scalar]

theorem takes_inst (F : Facts) (h : takesAll F = true) (b : BlockData) (hb : b ∈ F.blocks) (i : Inst)
    (hi : i ∈ b.insts) : takes F i = true := by
  simp only [takesAll, List.all_eq_true] at h
  exact h b hb i hi

theorem winOk_inst (F : Facts) (h : winOk F = true) (b : BlockData) (hb : b ∈ F.blocks) (i : Inst)
    (hi : i ∈ b.insts) :
    (∀ v ∈ valUses F i, F.win v = false) ∧ (∀ v ∈ Inst.uses i, F.win v = true → F.tys.get? v.id = some .i64) := by
  simp only [winOk, List.all_eq_true, Bool.and_eq_true] at h
  obtain ⟨h1, h2⟩ := h b hb i hi
  refine ⟨fun v hv => by simpa using h1 v hv, fun v hv hw => ?_⟩
  have := h2 v hv
  simp [hw] at this
  exact this

theorem isTy_spec (F : Facts) (v : Val) (ts : List ClifTy) (h : isTy F v ts = true) :
    ∃ t, F.tys.get? v.id = some t ∧ t ∈ ts := by
  revert h
  unfold isTy
  cases ht : F.tys.get? v.id with
  | none => intro h; cases h
  | some t => intro h; exact ⟨t, rfl, by simpa using h⟩

theorem intT_of_isTy (F : Facts) (a : Val) (h : isTy F a [.i16, .i32, .i64] = true) :
    ∀ t, F.tys.get? a.id = some t → IntT t := by
  intro t ht
  obtain ⟨t', h1, h2⟩ := isTy_spec F a _ h
  rw [h1] at ht; cases ht
  simp at h2; rcases h2 with rfl | rfl | rfl <;> simp [IntT]

/-- An instruction whose value has its operand's type: so the facts say. -/
theorem resTy_same (F : Facts) (i : Inst) (d a : Val)
    (hres : resTy (F.tys.get? ·.id) i = (F.tys.get? a.id).map (d, ·))
    (hty : F.tys.get? d.id = (resTy (F.tys.get? ·.id) i).map (·.2)) :
    ∀ t, F.tys.get? a.id = some t → F.tys.get? d.id = some t := by
  intro t ht; rw [hty, hres, ht]; rfl

theorem logicT_of_isTy (F : Facts) (a : Val) (h : isTy F a [.i8, .i16, .i32, .i64] = true) :
    ∀ t, F.tys.get? a.id = some t → IntT t ∨ t = .i8 := by
  intro t ht
  obtain ⟨t', h1, h2⟩ := isTy_spec F a _ h
  rw [h1] at ht; cases ht
  simp at h2; rcases h2 with rfl | rfl | rfl | rfl <;> simp [IntT]

/-! ### Counting trailing zeros

`ctz` lowers to `brev` then `clz`: the leading zeros of the low `w` bits
reversed are the trailing zeros of the low `w`, `w` where none is set. -/

/-- The low `w` bits of `x` reversed, as `PtxRun.un`'s `brev` computes them. -/
def brevN (w x : Nat) : Nat :=
  (List.range w).foldl (fun acc i => if x.testBit i then acc ||| 2 ^ (w - 1 - i) else acc) 0

theorem brev_fold_bit (w x : Nat) : ∀ (n acc j : Nat), n ≤ w →
    ((List.range n).foldl (fun acc i => if x.testBit i then acc ||| 2 ^ (w - 1 - i) else acc) acc).testBit j =
      (acc.testBit j || (decide (j < w ∧ w - 1 - j < n) && x.testBit (w - 1 - j)))
  | 0, acc, j, _ => by simp
  | n + 1, acc, j, hn => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      have ih := brev_fold_bit w x n acc j (by omega)
      by_cases hb : x.testBit n = true
      · rw [if_pos hb, Nat.testBit_or, ih, Nat.testBit_two_pow]
        by_cases hj : w - 1 - n = j
        · have h1 : w - 1 - j = n := by omega
          have h2 : j < w := by omega
          simp [hj, h1, h2, hb]
        · have h3 : decide (j < w ∧ w - 1 - j < n + 1) = decide (j < w ∧ w - 1 - j < n) := by
            apply decide_eq_decide.mpr
            constructor
            · intro ⟨h4, h5⟩; exact ⟨h4, by omega⟩
            · intro ⟨h4, h5⟩; exact ⟨h4, by omega⟩
          simp [hj, h3]
      · rw [if_neg hb, ih]
        by_cases he : j < w ∧ w - 1 - j = n
        · have hx : x.testBit (w - 1 - j) = false := by rw [he.2]; simpa using hb
          simp [hx]
        · have h3 : decide (j < w ∧ w - 1 - j < n + 1) = decide (j < w ∧ w - 1 - j < n) := by
            apply decide_eq_decide.mpr
            constructor
            · intro ⟨h4, h5⟩; exact ⟨h4, by omega⟩
            · intro ⟨h4, h5⟩; exact ⟨h4, by omega⟩
          rw [h3]

/-- Bit `j` of the reversal is the input's bit at `j`'s mirror. -/
theorem brev_bit (w x j : Nat) : (brevN w x).testBit j = (decide (j < w) && x.testBit (w - 1 - j)) := by
  unfold brevN
  rw [brev_fold_bit w x w 0 j (Nat.le_refl _)]
  by_cases hj : j < w
  · simp [hj, show w - 1 - j < w by omega]
  · simp [hj]

theorem brev_lt (w x : Nat) : brevN w x < 2 ^ w :=
  Nat.lt_pow_two_of_testBit _ fun i hi => by rw [brev_bit]; simp [show ¬ i < w by omega]

theorem find?_range_congr {p q : Nat → Bool} (n : Nat) (h : ∀ i, i < n → p i = q i) :
    (List.range n).find? p = (List.range n).find? q := by
  cases hq : (List.range n).find? q with
  | none =>
      rw [List.find?_range_eq_none] at hq ⊢
      intro i hi; rw [h i hi]; exact hq i hi
  | some j =>
      rw [List.find?_range_eq_some] at hq ⊢
      obtain ⟨h1, h2, h3⟩ := hq
      have hj := List.mem_range.mp h2
      exact ⟨by rw [h j hj]; exact h1, h2, fun k hk => by rw [h k (by omega)]; exact h3 k hk⟩

/-- `brev`, as `PtxRun.un` computes it. -/
theorem un_brev (hw : Hw) (ty : PtxIr.Ty) (x : UInt64) :
    PtxRun.un .brev hw ty x = .ok (UInt64.ofNat (brevN ty.width x.toNat)) := rfl

/-- **`clz` of the reversal counts trailing zeros.** -/
theorem un_clz_brev (hw : Hw) (ty : PtxIr.Ty) (h64 : ty.width ≤ 64) (x : Nat) :
    PtxRun.un .clz hw ty (UInt64.ofNat (brevN ty.width x)) =
      .ok (UInt64.ofNat (((List.range ty.width).find? (fun i => x.testBit i)).getD ty.width)) := by
  have hy : (UInt64.ofNat (brevN ty.width x)).toNat = brevN ty.width x :=
    UInt64.toNat_ofNat_of_lt' (Nat.lt_of_lt_of_le (brev_lt _ x) (Nat.pow_le_pow_right (by decide) h64))
  have hr : (List.range ty.width).reverse = (List.range ty.width).map (fun i => ty.width - 1 - i) := by
    conv => lhs; rw [List.range_eq_range']
    rw [List.reverse_range']; simp
  have hc : (List.range ty.width).find? ((fun i => (brevN ty.width x).testBit i) ∘ (fun i => ty.width - 1 - i)) =
      (List.range ty.width).find? (fun i => x.testBit i) :=
    find?_range_congr _ fun i hi => by
      simp only [Function.comp, brev_bit, show ty.width - 1 - i < ty.width by omega, decide_true, Bool.true_and,
        show ty.width - 1 - (ty.width - 1 - i) = i by omega]
  unfold PtxRun.un
  simp only []
  rw [hy, hr, List.find?_map, hc]
  cases h : (List.range ty.width).find? (fun i => x.testBit i) with
  | none => rfl
  | some i =>
      have hi := List.mem_range.mp (List.find?_range_eq_some.mp h).2.1
      simp only [Option.map_some, Option.getD_some]
      rw [show ty.width - 1 - (ty.width - 1 - i) = i by omega]
      rfl

/-- Bit `i` of a 64-bit value, as the Code's `ctz` reads it. -/
theorem shr_bit (x : UInt64) (i : Nat) (hi : i < 64) :
    ((x >>> UInt64.ofNat i) &&& 1 == 1) = x.toNat.testBit i := by
  have h : ((x >>> UInt64.ofNat i) &&& 1).toNat = x.toNat / 2 ^ i % 2 := by
    rw [UInt64.toNat_and, UInt64.toNat_shiftRight, UInt64.toNat_ofNat_of_lt' (Nat.lt_trans hi (by decide)),
      Nat.mod_eq_of_lt hi, Nat.shiftRight_eq_div_pow, show (1 : UInt64).toNat = 1 from rfl, Nat.and_one_is_mod]
  rw [Nat.testBit_eq_decide_div_mod_eq]
  by_cases hm : x.toNat / 2 ^ i % 2 = 1
  · have e : (x >>> UInt64.ofNat i) &&& 1 = 1 := UInt64.toNat_inj.mp (by rw [h, hm]; rfl)
    simp [e, hm]
  · have e : (x >>> UInt64.ofNat i) &&& 1 ≠ 1 := fun he => hm (by rw [← h, he]; rfl)
    simp [e, hm]

/-- A count of at most 64 is its own at 32 or 64 bits. -/
theorem small_and (t : ClifTy) (ht : t = .i32 ∨ t = .i64) (N : Nat) (hN : N ≤ 64) :
    UInt64.ofNat N &&& widthMask t = UInt64.ofNat N := by
  rcases ht with rfl | rfl
  · apply UInt64.toNat_inj.mp
    rw [toNat_wm .i32 (by simp), UInt64.toNat_ofNat_of_lt' (Nat.lt_of_le_of_lt hN (by decide))]
    exact Nat.mod_eq_of_lt (by simp [ClifTy.width]; omega)
  · exact and_wide64 _

/-- A count of at most 64 widened from 32 bits to 64. -/
theorem cvt_small (N : Nat) (hN : N ≤ 64) :
    PtxRun.cvt none .u64 .u32 (UInt64.ofNat N) = .ok (UInt64.ofNat N) := by
  rw [cvt_int none .u64 .u32 _ rfl rfl]
  have hm : PtxRun.mask 32 (UInt64.ofNat N) = UInt64.ofNat N := by
    rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp)]
    exact small_and .i32 (Or.inl rfl) N hN
  simp only [PtxRun.toInt, PtxRun.ofIntW, PtxIr.Ty.isSigned, PtxIr.Ty.width, Bool.false_and,
    Bool.false_eq_true, if_false, hm, ofInt_toNat, mask64]

/-- A count of trailing zeros of 32 bits lowers to its operand's lines, a
    `brev` into a register of its own, then `clz`. -/
theorem lower_ctz32 (F : Facts) (tag : Nat) (d a : Val) (ht : F.tys.get? a.id = some .i32) :
    (lower F tag (.ctz d a)).lines =
      (((runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i32)] }).2.lines).push
        (.ins none (.un .brev none .b32 (.own "bv" tag 0)
          (.reg (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i32)] }).1)))).push
        (.ins none (.un .clz none .b32 (regOf .i32 d) (.reg (.own "bv" tag 0)))) := by
  simp only [Std.HashMap.get?_eq_getElem?] at ht
  rw [lower_inst F tag _ rfl, inst]
  simp [own, o, emit, emitL, tyOf, facts?, ht, ClifTy.width, bTy]

/-- A count of trailing zeros of 64 bits: `brev`, then `clz` into a 32-bit
    register of its own, widened. -/
theorem lower_ctz64 (F : Facts) (tag : Nat) (d a : Val) (ht : F.tys.get? a.id = some .i64) :
    (lower F tag (.ctz d a)).lines =
      ((((runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).2.lines).push
        (.ins none (.un .brev none .b64 (.own "bv" tag 0)
          (.reg (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).1)))).push
        (.ins none (.un .clz none .b64
          (.own "cz" tag (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).2.n)
          (.reg (.own "bv" tag 0))))).push
        (.ins none (.cvt none .u64 .u32 (regOf .i64 d)
          (.reg (.own "cz" tag (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).2.n)))) := by
  simp only [Std.HashMap.get?_eq_getElem?] at ht
  rw [lower_inst F tag _ rfl, inst]
  simp [own, o, emit, emitL, tyOf, facts?, ht, ClifTy.width, bTy]

open PtxRun in
/-- **A count of trailing zeros** meets its obligation: `brev` then `clz`
    give the Code's count, at 64 bits widened from the 32-bit count. -/
theorem ctz_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val) (t : ClifTy)
    (ht : t = .i32 ∨ t = .i64) (hwa : ¬ Win F a) (hwd : F.win d = false)
    (hta : F.tys.get? a.id = some t) (htd : F.tys.get? d.id = some t) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.ctz d a) = true) :
    InstOk F lc c start0 (.ctz d a) := by
  intro tag D s s1 ts hst huses hS hW _ _hnan
  have hua : a ∈ Inst.uses (.ctz d a) := by simp [Inst.uses]
  obtain ⟨ta, xa, hva, hta'⟩ := hS.pres a (huses a hua)
  rw [hta] at hta'; cases hta'
  have hna := hS.normd a t xa (huses a hua) hva
  have h64 : t.width ≤ 64 := by rcases ht with rfl | rfl <;> decide
  have hN : ((List.range t.width).find? (fun i => xa.toNat.testBit i)).getD t.width ≤ 64 := by
    cases h : (List.range t.width).find? (fun i => xa.toNat.testBit i) with
    | none => simpa using h64
    | some i => have := List.mem_range.mp (List.find?_range_eq_some.mp h).2.1; simp; omega
  have hNm := small_and t ht _ hN
  have hf : (List.range t.width).find? (fun i => (xa >>> UInt64.ofNat i) &&& 1 == 1) =
      (List.range t.width).find? (fun i => xa.toNat.testBit i) :=
    find?_range_congr _ fun i hi => shr_bit xa i (by omega)
  have hint : t.isInt = true := by rcases ht with rfl | rfl <;> rfl
  have hs1 : s1 = { s with vals := Blocks.setV s.vals d (.sc t (UInt64.ofNat (((List.range t.width).find? (fun i => xa.toNat.testBit i)).getD t.width))) } := by
    simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.un, Sem.get, norm, hint] at hst
    rw [← hst, hf, hNm]
  subst hs1
  have h8 : t ≠ .i8 := by rcases ht with rfl | rfl <;> decide
  have hfin : ∀ regs' : Std.HashMap PtxIr.Reg UInt64,
      regs'.get? (regOf t d) = some (UInt64.ofNat (((List.range t.width).find? (fun i => xa.toNat.testBit i)).getD t.width)) →
      (∀ c' n, n ≠ d.id → regs'.get? (.val c' n) = ts.regs.get? (.val c' n)) →
      Sim F (extD D (.ctz d a)) (Blocks.setV s.vals d
        (.sc t (UInt64.ofNat (((List.range t.width).find? (fun i => xa.toNat.testBit i)).getD t.width)))) regs' := by
    intro regs' hd hk
    apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs regs' (.ctz d a) d t _ rfl hS htd hNm
    · intro _ _; simp only [enc, if_neg h8, hNm]; exact hd
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8'; exact absurd h8' h8
    · intro h8'; exact absurd h8' h8
    · exact hk
  rcases ht with rfl | rfl
  · obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
      r_sound F tag { n := 1, regs := #[(.own "bv" tag 0, .i32)] } s.vals c ts a .i32 xa D hS (huses a hua) hva hta hwa
    simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hna] at hreg1
    have hlist : (lower F tag (.ctz d a)).lines =
        ((La ++ ([.un .brev none .b32 (.own "bv" tag 0)
            (.reg (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i32)] }).1),
          .un .clz none .b32 (regOf .i32 d) (.reg (.own "bv" tag 0))] : List PtxIr.Instr)).map (PtxIr.Line.ins none)).toArray := by
      rw [lower_ctz32 F tag d a hta, hLa]; simp
    refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
    have hX := opVal_reg c { ts with regs := regs1 } _ _ hns1 hreg1
    have hbr := exec_un c { ts with regs := regs1 } .brev none .b32 (.own "bv" tag 0) _ xa _ hX (un_brev _ _ _)
    have hY : OpVal c { ts with regs := regs1.insert (.own "bv" tag 0) (UInt64.ofNat (brevN 32 xa.toNat)) }
        (.reg (.own "bv" tag 0)) (UInt64.ofNat (brevN 32 xa.toNat)) :=
      opVal_reg c _ _ _ (not_special_own "bv" tag 0) (by simp)
    have hcz := exec_un c _ .clz none .b32 (regOf .i32 d) _ _ _ hY (un_clz_brev c.hw .b32 (by decide) xa.toNat)
    refine ⟨{ ts with regs := (regs1.insert (.own "bv" tag 0) (UInt64.ofNat (brevN 32 xa.toNat))).insert (regOf .i32 d) (UInt64.ofNat (((List.range 32).find? (fun i => xa.toNat.testBit i)).getD 32)) }, ?_, ?_, hW⟩
    · rw [hlist, instrsOf_map, steps_append c _ _ ts _ hrun1]
      simp only [StateT.run, PtxIr.Ty.width] at hbr hcz
      simp [steps, StateT.run, hbr, hcz, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
    · apply hfin
      · simp [Std.HashMap.get?_eq_getElem?, ClifTy.width]
      · intro c' n hn
        have hne : PtxIr.Reg.val c' n ≠ regOf .i32 d := by simp [regOf]; omega
        rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
          Std.HashMap.getElem?_insert, if_neg (by simp), ← Std.HashMap.get?_eq_getElem?]
        exact hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)
  · obtain ⟨La, hLa, hn1, hnf1, hns1, regs1, hrun1, hreg1, hkeep1⟩ :=
      r_sound F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] } s.vals c ts a .i64 xa D hS (huses a hua) hva hta hwa
    simp only [enc, show ClifTy.i64 ≠ .i8 by decide, if_false, hna] at hreg1
    have hlist : (lower F tag (.ctz d a)).lines =
        ((La ++ ([.un .brev none .b64 (.own "bv" tag 0)
            (.reg (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).1),
          .un .clz none .b64 (.own "cz" tag (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).2.n)
            (.reg (.own "bv" tag 0)),
          .cvt none .u64 .u32 (regOf .i64 d)
            (.reg (.own "cz" tag (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).2.n))] : List PtxIr.Instr)).map
          (PtxIr.Line.ins none)).toArray := by
      rw [lower_ctz64 F tag d a hta, hLa]; simp
    refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
    have hX := opVal_reg c { ts with regs := regs1 } _ _ hns1 hreg1
    have hbr := exec_un c { ts with regs := regs1 } .brev none .b64 (.own "bv" tag 0) _ xa _ hX (un_brev _ _ _)
    have hY : OpVal c { ts with regs := regs1.insert (.own "bv" tag 0) (UInt64.ofNat (brevN 64 xa.toNat)) }
        (.reg (.own "bv" tag 0)) (UInt64.ofNat (brevN 64 xa.toNat)) :=
      opVal_reg c _ _ _ (not_special_own "bv" tag 0) (by simp)
    have hcz := exec_un c _ .clz none .b64
      (.own "cz" tag (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).2.n) _ _ _ hY
      (un_clz_brev c.hw .b64 (by decide) xa.toNat)
    have hZ : OpVal c { ts with regs := (regs1.insert (.own "bv" tag 0) (UInt64.ofNat (brevN 64 xa.toNat))).insert (.own "cz" tag (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).2.n) (UInt64.ofNat (((List.range 64).find? (fun i => xa.toNat.testBit i)).getD 64)) }
        (.reg (.own "cz" tag (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).2.n))
        (UInt64.ofNat (((List.range 64).find? (fun i => xa.toNat.testBit i)).getD 64)) :=
      opVal_reg c _ _ _ (not_special_own _ _ _) (by simp)
    have hcv := exec_cvt c _ none .u64 .u32 (regOf .i64 d) _ _ _ hZ (cvt_small _ hN)
    refine ⟨{ ts with regs := ((regs1.insert (.own "bv" tag 0) (UInt64.ofNat (brevN 64 xa.toNat))).insert (.own "cz" tag (runLM (r a) F tag { n := 1, regs := #[(.own "bv" tag 0, .i64)] }).2.n) (UInt64.ofNat (((List.range 64).find? (fun i => xa.toNat.testBit i)).getD 64))).insert (regOf .i64 d) (UInt64.ofNat (((List.range 64).find? (fun i => xa.toNat.testBit i)).getD 64)) }, ?_, ?_, hW⟩
    · rw [hlist, instrsOf_map, steps_append c _ _ ts _ hrun1]
      simp only [StateT.run, PtxIr.Ty.width] at hbr hcz hcv
      simp [steps, StateT.run, hbr, hcz, hcv, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
    · apply hfin
      · simp [Std.HashMap.get?_eq_getElem?, ClifTy.width]
      · intro c' n hn
        have hne : PtxIr.Reg.val c' n ≠ regOf .i64 d := by simp [regOf]; omega
        rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
          Std.HashMap.getElem?_insert, if_neg (by simp), Std.HashMap.getElem?_insert, if_neg (by simp),
          ← Std.HashMap.get?_eq_getElem?]
        exact hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)

/-! ### Counting set bits

`popcnt` of 32 bits lowers to `popc.b32`, which counts the set bits of the
low 32. -/

theorem inst_popcnt (d a : Val) :
    inst (.popcnt d a) = (do emit (.un .popc none .b32 (regOf .i32 d) (← o a))) := by simp only [inst]

/-- A count of set bits among `w` is at most `w`. -/
theorem popc_count_le (x w : Nat) : ((List.range w).filter (fun i => x.testBit i)).length ≤ w := by
  simpa using List.length_filter_le (fun i => x.testBit i) (List.range w)

open PtxRun in
/-- **A count of set bits** meets its obligation: `popc.b32` counts the low
    32 bits as the Code's `popcnt` counts an `i32`'s. -/
theorem popcnt_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d a : Val)
    (hwa : ¬ Win F a) (hwd : F.win d = false)
    (hta : F.tys.get? a.id = some .i32) (htd : F.tys.get? d.id = some .i32) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.popcnt d a) = true) :
    InstOk F lc c start0 (.popcnt d a) := by
  have hN : ∀ x : UInt64,
      UInt64.ofNat ((List.range 32).filter (fun i => x.toNat.testBit i)).length &&& widthMask .i32 =
        UInt64.ofNat ((List.range 32).filter (fun i => x.toNat.testBit i)).length :=
    fun x => small_and .i32 (Or.inl rfl) _ (Nat.le_trans (popc_count_le x.toNat 32) (by decide))
  apply un1_ok (hmo := hmo) (hrule := fun _ _ _ _ _ hk => by simp [magRule] at hk) F lc c start0 _ d a
    (fun _ dr xo => .un .popc none .b32 dr xo) (fun _ => .i32)
    (fun _ x => UInt64.ofNat ((List.range 32).filter (fun i => x.toNat.testBit i)).length) (· = .i32)
    (fun _ _ h => by cases h) rfl (by simp [Inst.uses]) hwa hwd (fun t ht => by rw [hta] at ht; cases ht; exact htd) hld
  · intro s s1 hst _ t xa hva ht
    rw [hta] at ht; cases ht
    refine ⟨rfl, ?_⟩
    have hf : (List.range 32).filter (fun i => (xa >>> UInt64.ofNat i) &&& 1 == 1) =
        (List.range 32).filter (fun i => xa.toNat.testBit i) :=
      List.filter_congr fun i hi => shr_bit xa i (by have := List.mem_range.mp hi; omega)
    simp [step1, Blocks.evalInst, Blocks.evalInst.un, hva, Blocks.viaOp, evalOp, Sem.un, Sem.get, norm, ClifTy.isInt, ClifTy.width] at hst
    rw [← hst, hf, hN]
  · intro tag t ht
    rw [hta] at ht; cases ht
    rw [lower_inst F tag _ rfl, inst_popcnt]
    simp only [runLM_bind, runLM_pure, o, emit, emitL, runLM_modify]
  · intro t xa ht hn _ c' ts' xo dr hxo
    subst ht
    have hx : OpVal c' ts' xo xa := by simpa [enc, hn] using hxo
    rw [exec_un c' ts' .popc none .b32 dr xo xa _ hx rfl]
    simp [enc, PtxIr.Ty.width, hN]
  · intro t xa _ _
    exact hN xa


theorem not_win_of (F : Facts) (v : Val) (h : F.win v = false) : ¬ Win F v := by
  intro hw; rw [(win_iff F v).mpr hw] at h; cases h

open PtxRun in
/-- An `atom` read-modify-write through a register holding `x`: aligned there,
    it reads the old value, stores the new one, records the store, and the
    destination gets the old value. -/
theorem exec_atom_rmw (c : PtxRun.Ctx) (ts : PtxRun.TS) (sp : PtxIr.Space) (o : PtxIr.AtomOp) (od : Bool) (k : AtomicRmw)
    (pt : PtxIr.Ty) (dr R : PtxIr.Reg) (vo : PtxIr.Opnd) (x z y old : UInt64) (m : Mem)
    (hk : o.rmw? = some k) (hsp : sp ≠ .param) (hR : OpVal c ts (.reg R) x) (hp : place sp x = z)
    (hal : z % UInt64.ofNat (pt.width / 8) = 0) (hv : OpVal c ts vo y)
    (hl : ts.w.mem.load z (pt.width / 8) = some old)
    (hm : ts.w.mem.store z (pt.width / 8) (PtxRun.mask pt.width (rmwNew k (atomTy pt) old y)) = some m) :
    (exec c (.atom sp o od pt (some dr) { base := .reg R } vo)).run ts =
      .ok (.step, { ts with regs := ts.regs.insert dr old, w := { ts.w with mem := m, obs := (Obs.store z (pt.width / 8) (PtxRun.mask pt.width (rmwNew k (atomTy pt) old y))) :: ts.w.obs } }) := by
  simp only [OpVal] at hR hv
  have hne : (sp == .param) = false := by cases sp <;> first | rfl | exact absurd rfl hsp
  simp only [exec, PtxRun.atom, hk, aligned, PtxRun.load, PtxRun.store, StateT.run, bind, StateT.bind, hne,
    Bool.false_eq_true, if_false] at hR hv ⊢
  simp [hR, hv, Except.bind, ofInt_toNat, hp, hal, hl, hm, MonadState.get, getThe, MonadStateOf.get, StateT.get,
    set, MonadStateOf.set, StateT.set, setReg, pure, StateT.pure, Except.pure, modify, modifyGet,
    MonadStateOf.modifyGet, StateT.modifyGet]

/-- An atomic read-modify-write: the address's lines, the operand's, then one
    `atom`. -/
theorem lower_atom_rmw (F : Facts) (tag : Nat) (d p x : Val) (t : ClifTy) (k : AtomicRmw) (ord : AtomicOrder)
    (op : PtxIr.AtomOp)
    (hop : atomOp k = some op) :
    (lower F tag (.call (some d) (.atomic (.rmw t k ord)) [p, x])).lines =
      ((runLM (r x) F tag (runLM (addr p) F tag {}).2).2.lines).push
        (.ins none (.atom (runLM (addr p) F tag {}).1.1 op (ord == .seqCst) (atomPty op t) (some (regOf t d))
          (runLM (addr p) F tag {}).1.2 (.reg (runLM (r x) F tag (runLM (addr p) F tag {}).2).1))) := by
  rw [lower_inst F tag _ rfl, inst]
  simp only [atomic, hop, runLM_bind, o]
  rfl

theorem atomPty_width (op : PtxIr.AtomOp) (t : ClifTy) (ht : t = .i32 ∨ t = .i64) :
    (atomPty op t).width / 8 = tyBytes t ∧ PtxRun.atomTy (atomPty op t) = t ∧ (atomPty op t).width = t.width := by
  rcases ht with rfl | rfl <;> cases op <;> decide

theorem atomOp_rmw (k : AtomicRmw) (op : PtxIr.AtomOp) (h : atomOp k = some op) : op.rmw? = some k := by
  cases k <;> simp [atomOp] at h <;> subst h <;> rfl

theorem rmwNew_mask (k : AtomicRmw) (t : ClifTy) (old x : UInt64) :
    rmwNew k t old x &&& widthMask t = rmwNew k t old x := by
  simp only [rmwNew, UInt64.and_assoc, UInt64.and_self]

/-- What an atomic read-modify-write answers: aligned, the old value loaded,
    the new stored and recorded. -/
theorem atomicCall_rmw (w w' : World) (t : ClifTy) (k : AtomicRmw) (ord : AtomicOrder) (xp xx : UInt64) (r : Option V)
    (h : atomicCall (.rmw t k ord) [.sc .i64 xp, .sc t xx] w = some (r, w')) :
    ∃ old m, xp % UInt64.ofNat (tyBytes t) = 0 ∧ w.mem.load xp (tyBytes t) = some old ∧
      w.mem.store xp (tyBytes t) (rmwNew k t old xx) = some m ∧ r = some (.sc t old) ∧
      w' = { w with mem := m, obs := .store xp (tyBytes t) (rmwNew k t old xx) :: w.obs } := by
  simp only [atomicCall, bne_self_eq_false, Bool.false_or] at h
  by_cases ha : xp % UInt64.ofNat (tyBytes t) = 0
  · simp only [ha, obsStore] at h
    obtain ⟨old, hl, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨old, m, ha, hl, hm, rfl, rfl⟩
  · simp [ha] at h

/-- A word's or double word's bits, masked to its width, are themselves. -/
theorem mask_int (t : ClifTy) (ht : t = .i32 ∨ t = .i64) (v : UInt64) (hv : v &&& widthMask t = v) :
    PtxRun.mask t.width v = v := by
  rcases ht with rfl | rfl
  · rw [show widthMask .i32 = (1 <<< UInt64.ofNat 32) - 1 from rfl] at hv
    simpa [PtxRun.mask, ClifTy.width] using hv
  · simp [PtxRun.mask, ClifTy.width]

open PtxRun in
/-- **An atomic read-modify-write** at a buffer's word or double word meets
    its obligation: the block semantics records the call, which a launch does
    not read, then loads the old value, stores the new and answers the old;
    the `atom` does the same to the same bytes. -/
theorem atom_rmw_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d p x : Val) (t : ClifTy)
    (k : AtomicRmw) (ord : AtomicOrder) (op : PtxIr.AtomOp) (hop : atomOp k = some op) (ht : t = .i32 ∨ t = .i64)
    (hwx : ¬ Win F x) (hwd : F.win d = false)
    (htp : F.tys.get? p.id = some .i64) (htx : F.tys.get? x.id = some t)
    (htd : F.tys.get? d.id = some t) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.call (some d) (.atomic (.rmw t k ord)) [p, x]) = true) :
    InstOk F lc c start0 (.call (some d) (.atomic (.rmw t k ord)) [p, x]) := by
  intro tag D s s1 ts hst huses hS hW _ hmok
  have hup : p ∈ Inst.uses (.call (some d) (.atomic (.rmw t k ord)) [p, x]) := by simp [Inst.uses]
  have hux : x ∈ Inst.uses (.call (some d) (.atomic (.rmw t k ord)) [p, x]) := by simp [Inst.uses]
  obtain ⟨tp, xp, hvp, htp'⟩ := hS.pres p (huses p hup)
  rw [htp] at htp'; cases htp'
  obtain ⟨tx, xx, hvx, htx'⟩ := hS.pres x (huses x hux)
  rw [htx] at htx'; cases htx'
  have hnx := hS.normd x t xx (huses x hux) hvx
  have hne8 : t ≠ .i8 := by rcases ht with rfl | rfl <;> decide
  obtain ⟨hbytes, hty, hwid⟩ := atomPty_width op t ht
  -- the block semantics' step
  let ob : Obs := .call (.atomic (.rmw t k ord)) [.sc .i64 xp, .sc t xx]
  obtain ⟨old, m, hal, hold, hm, rfl⟩ : ∃ old m, xp % UInt64.ofNat (tyBytes t) = 0 ∧
      s.world.mem.load xp (tyBytes t) = some old ∧ s.world.mem.store xp (tyBytes t) (rmwNew k t old xx) = some m ∧
      s1 = { vals := Blocks.setV s.vals d (.sc t old), world := { s.world with mem := m, obs := .store xp (tyBytes t) (rmwNew k t old xx) :: ob :: s.world.obs } } := by
    simp only [step1, List.mapM_cons, List.mapM_nil, hvp, hvx, bind, Option.bind, pure] at hst
    rw [show callOf lc (.atomic (.rmw t k ord)) [.sc .i64 xp, .sc t xx]
        (obsCall s.world (.atomic (.rmw t k ord)) [.sc .i64 xp, .sc t xx]) =
        atomicCall (.rmw t k ord) [.sc .i64 xp, .sc t xx]
        (obsCall s.world (.atomic (.rmw t k ord)) [.sc .i64 xp, .sc t xx]) from rfl] at hst
    cases hc : atomicCall (.rmw t k ord) [.sc .i64 xp, .sc t xx]
        (obsCall s.world (.atomic (.rmw t k ord)) [.sc .i64 xp, .sc t xx]) with
    | none => rw [hc] at hst; cases hst
    | some rw' =>
      obtain ⟨r', w'⟩ := rw'
      rw [hc] at hst
      obtain ⟨old, m, hal, hl, hm, rfl, rfl⟩ := atomicCall_rmw _ _ t k ord xp xx r' hc
      simp only [Option.some.injEq] at hst
      refine ⟨old, m, hal, hl, hm, ?_⟩
      rw [← hst]
      have hcw' : callWords s.world (.atomic (.rmw t k ord)) [.sc .i64 xp, .sc t xx] = [] := rfl
      simp [obsCall, hcw', ob]
  -- the lines
  let sa := (runLM (addr p) F tag {}).2
  obtain ⟨La, hLa, hn1, R, xr, hbase, hns1, hnf1, hreach, regs1, hrun1, hreg1, hkeep1⟩ :=
    addr_sound F tag {} s.vals c ts p xp D hS (huses p hup) hvp htp
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lv, hLv, hn2, hnf2, hns2, regs2, hrun2, hreg2, hkeep2⟩ :=
    r_sound F tag sa s.vals c { ts with regs := regs1 } x t xx D hS1 (huses x hux) hvx htx hwx
  simp only [enc, hne8, if_false, hnx] at hreg2
  let core : PtxIr.Instr := .atom (runLM (addr p) F tag {}).1.1 op (ord == .seqCst) (atomPty op t) (some (regOf t d))
    { base := .reg R } (.reg (runLM (r x) F tag sa).1)
  have hlist : (lower F tag (.call (some d) (.atomic (.rmw t k ord)) [p, x])).lines =
      ((La ++ Lv ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_atom_rmw F tag d p x t k ord op hop, hLv]
    show (sa.lines ++ _).push _ = _
    rw [hLa, hbase]; simp [core, sa]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hmem : ts.w.mem = s.world.mem := (congrArg World.mem hW.1).symm
  have hl' : ts.w.mem.load xp (tyBytes t) = some old := by rw [hmem]; exact hold
  have hm' : ts.w.mem.store xp (tyBytes t) (rmwNew k t old xx) = some m := by rw [hmem]; exact hm
  have hmask : PtxRun.mask (atomPty op t).width (rmwNew k (atomTy (atomPty op t)) old xx) = rmwNew k t old xx := by
    rw [hty, hwid]; exact mask_int t ht _ (rmwNew_mask k t old xx)
  have he : (exec c core).run { ts with regs := regs2 } =
      .ok (.step, { ts with regs := regs2.insert (regOf t d) old, w := { ts.w with mem := m, obs := (Obs.store xp (tyBytes t) (rmwNew k t old xx)) :: ts.w.obs } }) := by
    have := exec_atom_rmw c { ts with regs := regs2 } _ op (ord == .seqCst) k (atomPty op t) (regOf t d) R _ xr xp xx old m
      (atomOp_rmw k op hop) (reaches_sp hreach)
      (opVal_reg c _ _ _ hns1 (by rw [hkeep2 _ hnf1]; exact hreg1))
      (place_eq hreach (fit_store hm' hmok.2)) (by rw [hbytes]; exact hal)
      (opVal_reg c _ _ _ hns2 hreg2) (by rw [hbytes]; exact hl') (by rw [hbytes, hmask]; exact hm')
    simpa [hbytes, hmask] using this
  -- the old value fits its type
  have hbw : old &&& widthMask t = old := by
    have hlt := load_lt _ _ _ old (by rcases ht with rfl | rfl <;> decide) hold
    apply UInt64.toNat_inj.mp
    rcases ht with rfl | rfl
    · rw [show widthMask .i32 = ((2 ^ 32 - 1 : Nat).toUInt64) from rfl, toNat_and_mask _ _ (by decide)]
      simp [tyBytes, ClifTy.width] at hlt
      exact Nat.mod_eq_of_lt hlt
    · rw [show widthMask .i64 = ((2 ^ 64 - 1 : Nat).toUInt64) from rfl, toNat_and_mask _ _ (by decide)]
      exact Nat.mod_eq_of_lt old.toNat_lt
  refine ⟨{ ts with regs := regs2.insert (regOf t d) old, w := { ts.w with mem := m, obs := (Obs.store xp (tyBytes t) (rmwNew k t old xx)) :: ts.w.obs } }, ?_, ?_, ?_⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ _ d t old rfl
      hS htd hbw
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?, enc, hne8, hbw]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; exact absurd h8 hne8
    · intro h8; exact absurd h8 hne8
    · intro c' n' hn
      have hne : PtxIr.Reg.val c' n' ≠ regOf t d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        ← Std.HashMap.get?_eq_getElem?]
      rw [hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h), hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)]
  · have := wrel_cons start0 _ { s.world with obs := ob :: s.world.obs } ts.w m
      (Obs.store xp (tyBytes t) (rmwNew k t old xx)) (wrel_skip start0 _ s.world ts.w ob hW (by simp [viewObs, ob]))
    simpa using this

open PtxRun in
/-- An `atom` load through a register holding `x`: aligned there, the
    destination gets the bytes. -/
theorem exec_atom_load (c : PtxRun.Ctx) (ts : PtxRun.TS) (sp : PtxIr.Space) (od : Bool) (pt : PtxIr.Ty) (dr R : PtxIr.Reg)
    (vo : PtxIr.Opnd) (x z v : UInt64) (hsp : sp ≠ .param) (hR : OpVal c ts (.reg R) x) (hp : place sp x = z)
    (hal : z % UInt64.ofNat (pt.width / 8) = 0) (hl : ts.w.mem.load z (pt.width / 8) = some v) :
    (exec c (.atom sp .load od pt (some dr) { base := .reg R } vo)).run ts =
      .ok (.step, { ts with regs := ts.regs.insert dr v }) := by
  simp only [OpVal] at hR
  have hne : (sp == .param) = false := by cases sp <;> first | rfl | exact absurd rfl hsp
  simp only [exec, PtxRun.atom, PtxIr.AtomOp.rmw?, aligned, PtxRun.load, StateT.run, bind, StateT.bind, hne,
    Bool.false_eq_true, if_false] at hR ⊢
  simp [hR, Except.bind, ofInt_toNat, hp, hal, hl, MonadState.get, getThe, MonadStateOf.get, StateT.get,
    setReg, pure, StateT.pure, Except.pure, modify, modifyGet, MonadStateOf.modifyGet, StateT.modifyGet]

open PtxRun in
/-- An `atom` store through a register holding `x`: aligned there, it stores
    and records the store. -/
theorem exec_atom_store (c : PtxRun.Ctx) (ts : PtxRun.TS) (sp : PtxIr.Space) (od : Bool) (pt : PtxIr.Ty) (R : PtxIr.Reg)
    (vo : PtxIr.Opnd) (x z y : UInt64) (m : Mem) (hR : OpVal c ts (.reg R) x) (hp : place sp x = z)
    (hal : z % UInt64.ofNat (pt.width / 8) = 0) (hv : OpVal c ts vo y)
    (hm : ts.w.mem.store z (pt.width / 8) (PtxRun.mask pt.width y) = some m) :
    (exec c (.atom sp .store od pt none { base := .reg R } vo)).run ts =
      .ok (.step, { ts with w := { ts.w with mem := m, obs := (Obs.store z (pt.width / 8) (PtxRun.mask pt.width y)) :: ts.w.obs } }) := by
  simp only [OpVal] at hR hv
  simp only [exec, PtxRun.atom, PtxIr.AtomOp.rmw?, aligned, PtxRun.store, StateT.run, bind, StateT.bind] at hR hv ⊢
  simp [hR, hv, Except.bind, ofInt_toNat, hp, hal, hm, MonadState.get, getThe, MonadStateOf.get, StateT.get, set,
    MonadStateOf.set, StateT.set, pure, StateT.pure, Except.pure]

/-- What an atomic load answers: aligned, the bytes there, the world as it
    was. -/
theorem atomicCall_load (w w' : World) (t : ClifTy) (ord : AtomicOrder) (xp : UInt64) (r : Option V)
    (h : atomicCall (.load t ord) [.sc .i64 xp] w = some (r, w')) :
    ∃ b, xp % UInt64.ofNat (tyBytes t) = 0 ∧ w.mem.load xp (tyBytes t) = some b ∧ r = some (.sc t b) ∧ w' = w := by
  simp only [atomicCall] at h
  by_cases ha : xp % UInt64.ofNat (tyBytes t) = 0
  · simp only [ha] at h
    obtain ⟨b, hl, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨b, ha, hl, rfl, rfl⟩
  · simp [ha] at h

/-- What an atomic store does: aligned, the bytes stored and recorded. -/
theorem atomicCall_store (w w' : World) (t : ClifTy) (ord : AtomicOrder) (xv xp : UInt64) (r : Option V)
    (h : atomicCall (.store t ord) [.sc t xv, .sc .i64 xp] w = some (r, w')) :
    ∃ m, xp % UInt64.ofNat (tyBytes t) = 0 ∧ w.mem.store xp (tyBytes t) xv = some m ∧ r = none ∧
      w' = { w with mem := m, obs := .store xp (tyBytes t) xv :: w.obs } := by
  simp only [atomicCall, bne_self_eq_false, Bool.false_or] at h
  by_cases ha : xp % UInt64.ofNat (tyBytes t) = 0
  · simp only [ha, obsStore] at h
    obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨m, ha, hm, rfl, rfl⟩
  · simp [ha] at h

theorem lower_atom_load (F : Facts) (tag : Nat) (d p : Val) (t : ClifTy) (ord : AtomicOrder) :
    (lower F tag (.call (some d) (.atomic (.load t ord)) [p])).lines =
      ((runLM (addr p) F tag {}).2.lines).push
        (.ins none (.atom (runLM (addr p) F tag {}).1.1 .load (ord == .seqCst) (atomPty .load t) (some (regOf t d))
          (runLM (addr p) F tag {}).1.2 (.imm 0))) := by
  rw [lower_inst F tag _ rfl, inst]
  simp only [atomic, runLM_bind]
  rfl

theorem lower_atom_store (F : Facts) (tag : Nat) (v p : Val) (t : ClifTy) (ord : AtomicOrder) :
    (lower F tag (.call none (.atomic (.store t ord)) [v, p])).lines =
      ((runLM (r v) F tag (runLM (addr p) F tag {}).2).2.lines).push
        (.ins none (.atom (runLM (addr p) F tag {}).1.1 .store (ord == .seqCst) (atomPty .store t) none
          (runLM (addr p) F tag {}).1.2 (.reg (runLM (r v) F tag (runLM (addr p) F tag {}).2).1))) := by
  rw [lower_inst F tag _ rfl, inst]
  simp only [atomic, runLM_bind, o]
  rfl

open PtxRun in
/-- **An atomic load** of a buffer's word or double word meets its
    obligation: the block semantics records the call, which a launch does not
    read, and answers the bytes; the `atom` reads the same. -/
theorem atom_load_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (d p : Val) (t : ClifTy)
    (ord : AtomicOrder)
    (ht : t = .i32 ∨ t = .i64) (hwd : F.win d = false)
    (htp : F.tys.get? p.id = some .i64) (htd : F.tys.get? d.id = some t) (hld : F.lits.get? d.id = none)
    (hmo : magOk F d (.call (some d) (.atomic (.load t ord)) [p]) = true) :
    InstOk F lc c start0 (.call (some d) (.atomic (.load t ord)) [p]) := by
  intro tag D s s1 ts hst huses hS hW _ hmok
  have hup : p ∈ Inst.uses (.call (some d) (.atomic (.load t ord)) [p]) := by simp [Inst.uses]
  obtain ⟨tp, xp, hvp, htp'⟩ := hS.pres p (huses p hup)
  rw [htp] at htp'; cases htp'
  have hne8 : t ≠ .i8 := by rcases ht with rfl | rfl <;> decide
  obtain ⟨hbytes, -, -⟩ := atomPty_width .load t ht
  let ob : Obs := .call (.atomic (.load t ord)) [.sc .i64 xp]
  obtain ⟨b, hal, hold, rfl⟩ : ∃ b, xp % UInt64.ofNat (tyBytes t) = 0 ∧ s.world.mem.load xp (tyBytes t) = some b ∧
      s1 = { vals := Blocks.setV s.vals d (.sc t b), world := { s.world with obs := ob :: s.world.obs } } := by
    simp only [step1, List.mapM_cons, List.mapM_nil, hvp, bind, Option.bind, pure] at hst
    rw [show callOf lc (.atomic (.load t ord)) [.sc .i64 xp] (obsCall s.world (.atomic (.load t ord)) [.sc .i64 xp]) =
        atomicCall (.load t ord) [.sc .i64 xp] (obsCall s.world (.atomic (.load t ord)) [.sc .i64 xp]) from rfl] at hst
    cases hc : atomicCall (.load t ord) [.sc .i64 xp] (obsCall s.world (.atomic (.load t ord)) [.sc .i64 xp]) with
    | none => rw [hc] at hst; cases hst
    | some rw' =>
      obtain ⟨r', w'⟩ := rw'
      rw [hc] at hst
      obtain ⟨b, hal, hl, rfl, rfl⟩ := atomicCall_load _ _ t ord xp r' hc
      simp only [Option.some.injEq] at hst
      refine ⟨b, hal, hl, ?_⟩
      rw [← hst]
      have hcw' : callWords s.world (.atomic (.load t ord)) [.sc .i64 xp] = [] := rfl
      simp [obsCall, hcw', ob]
  obtain ⟨La, hLa, hn1, R, xr, hbase, hns1, hnf1, hreach, regs1, hrun1, hreg1, hkeep1⟩ :=
    addr_sound F tag {} s.vals c ts p xp D hS (huses p hup) hvp htp
  let core : PtxIr.Instr := .atom (runLM (addr p) F tag {}).1.1 .load (ord == .seqCst) (atomPty .load t)
    (some (regOf t d))
    { base := .reg R } (.imm 0)
  have hlist : (lower F tag (.call (some d) (.atomic (.load t ord)) [p])).lines =
      ((La ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_atom_load F tag d p t ord, hLa, hbase]; simp [core]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hmem : ts.w.mem = s.world.mem := (congrArg World.mem hW.1).symm
  have hl' : ts.w.mem.load xp (tyBytes t) = some b := by rw [hmem]; exact hold
  have he : (exec c core).run { ts with regs := regs1 } = .ok (.step, { ts with regs := regs1.insert (regOf t d) b }) :=
    exec_atom_load c { ts with regs := regs1 } _ (ord == .seqCst) (atomPty .load t) (regOf t d) R _ xr xp b (reaches_sp hreach)
      (opVal_reg c _ _ _ hns1 hreg1) (place_eq hreach (fit_load hl' hmok.2)) (by rw [hbytes]; exact hal)
      (by rw [hbytes]; exact hl')
  have hbw : b &&& widthMask t = b := by
    have hlt := load_lt _ _ _ b (by rcases ht with rfl | rfl <;> decide) hold
    apply UInt64.toNat_inj.mp
    rcases ht with rfl | rfl
    · rw [show widthMask .i32 = ((2 ^ 32 - 1 : Nat).toUInt64) from rfl, toNat_and_mask _ _ (by decide)]
      simp [tyBytes, ClifTy.width] at hlt
      exact Nat.mod_eq_of_lt hlt
    · rw [show widthMask .i64 = ((2 ^ 64 - 1 : Nat).toUInt64) from rfl, toNat_and_mask _ _ (by decide)]
      exact Nat.mod_eq_of_lt b.toNat_lt
  refine ⟨{ ts with regs := regs1.insert (regOf t d) b }, ?_, ?_,
    wrel_skip start0 _ s.world ts.w ob hW (by simp [viewObs, ob])⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _ hrun1]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · apply sim_set (hmo := hmo) (hrule := fun _ hk => by simp [magRule] at hk) F D s.vals ts.regs _ _ d t b rfl
      hS htd hbw
    · intro _ _
      simp [Std.HashMap.get?_eq_getElem?, enc, hne8, hbw]
    · intro l hl; rw [hld] at hl; cases hl
    · intro hw; rw [(win_iff F d).mpr hw] at hwd; cases hwd
    · intro h8; exact absurd h8 hne8
    · intro h8; exact absurd h8 hne8
    · intro c' n' hn
      have hne : PtxIr.Reg.val c' n' ≠ regOf t d := by simp [regOf]; omega
      rw [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert, if_neg (by simpa using Ne.symm hne),
        ← Std.HashMap.get?_eq_getElem?]
      exact hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h)

open PtxRun in
/-- **An atomic store** to a buffer's word or double word meets its
    obligation: the block semantics records the call, which a launch does not
    read, then the store; the `atom` stores the same bytes. -/
theorem atom_store_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat) (v p : Val) (t : ClifTy)
    (ord : AtomicOrder)
    (ht : t = .i32 ∨ t = .i64) (hwv : ¬ Win F v)
    (htp : F.tys.get? p.id = some .i64) (htv : F.tys.get? v.id = some t) :
    InstOk F lc c start0 (.call none (.atomic (.store t ord)) [v, p]) := by
  intro tag D s s1 ts hst huses hS hW _ hmok
  have hup : p ∈ Inst.uses (.call none (.atomic (.store t ord)) [v, p]) := by simp [Inst.uses]
  have huv : v ∈ Inst.uses (.call none (.atomic (.store t ord)) [v, p]) := by simp [Inst.uses]
  obtain ⟨tp, xp, hvp, htp'⟩ := hS.pres p (huses p hup)
  rw [htp] at htp'; cases htp'
  obtain ⟨tv, xv, hvv, htv'⟩ := hS.pres v (huses v huv)
  rw [htv] at htv'; cases htv'
  have hnv := hS.normd v t xv (huses v huv) hvv
  have hne8 : t ≠ .i8 := by rcases ht with rfl | rfl <;> decide
  obtain ⟨hbytes, -, hwid⟩ := atomPty_width .store t ht
  let ob : Obs := .call (.atomic (.store t ord)) [.sc t xv, .sc .i64 xp]
  obtain ⟨m, hal, hm, rfl⟩ : ∃ m, xp % UInt64.ofNat (tyBytes t) = 0 ∧ s.world.mem.store xp (tyBytes t) xv = some m ∧
      s1 = { s with world := { s.world with mem := m, obs := .store xp (tyBytes t) xv :: ob :: s.world.obs } } := by
    simp only [step1, List.mapM_cons, List.mapM_nil, hvp, hvv, bind, Option.bind, pure] at hst
    rw [show callOf lc (.atomic (.store t ord)) [.sc t xv, .sc .i64 xp]
        (obsCall s.world (.atomic (.store t ord)) [.sc t xv, .sc .i64 xp]) =
        atomicCall (.store t ord) [.sc t xv, .sc .i64 xp]
        (obsCall s.world (.atomic (.store t ord)) [.sc t xv, .sc .i64 xp]) from rfl] at hst
    cases hc : atomicCall (.store t ord) [.sc t xv, .sc .i64 xp]
        (obsCall s.world (.atomic (.store t ord)) [.sc t xv, .sc .i64 xp]) with
    | none => rw [hc] at hst; cases hst
    | some rw' =>
      obtain ⟨r', w'⟩ := rw'
      rw [hc] at hst
      obtain ⟨m, hal, hm, rfl, rfl⟩ := atomicCall_store _ _ t ord xv xp r' hc
      simp only [Option.some.injEq] at hst
      refine ⟨m, hal, hm, ?_⟩
      rw [← hst]
      have hcw' : callWords s.world (.atomic (.store t ord)) [.sc t xv, .sc .i64 xp] = [] := rfl
      simp [obsCall, hcw', ob]
  let sa := (runLM (addr p) F tag {}).2
  obtain ⟨La, hLa, hn1, R, xr, hbase, hns1, hnf1, hreach, regs1, hrun1, hreg1, hkeep1⟩ :=
    addr_sound F tag {} s.vals c ts p xp D hS (huses p hup) hvp htp
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lv, hLv, hn2, hnf2, hns2, regs2, hrun2, hreg2, hkeep2⟩ :=
    r_sound F tag sa s.vals c { ts with regs := regs1 } v t xv D hS1 (huses v huv) hvv htv hwv
  simp only [enc, hne8, if_false, hnv] at hreg2
  let core : PtxIr.Instr := .atom (runLM (addr p) F tag {}).1.1 .store (ord == .seqCst) (atomPty .store t) none
    { base := .reg R } (.reg (runLM (r v) F tag sa).1)
  have hlist : (lower F tag (.call none (.atomic (.store t ord)) [v, p])).lines =
      ((La ++ Lv ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_atom_store F tag v p t ord, hLv]
    show (sa.lines ++ _).push _ = _
    rw [hLa, hbase]; simp [core, sa]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hmem : ts.w.mem = s.world.mem := (congrArg World.mem hW.1).symm
  have hm' : ts.w.mem.store xp (tyBytes t) xv = some m := by rw [hmem]; exact hm
  have hmask : PtxRun.mask (atomPty .store t).width xv = xv := by rw [hwid]; exact mask_int t ht xv hnv
  have he : (exec c core).run { ts with regs := regs2 } =
      .ok (.step, { ts with regs := regs2, w := { ts.w with mem := m, obs := (Obs.store xp (tyBytes t) xv) :: ts.w.obs } }) := by
    have := exec_atom_store c { ts with regs := regs2 } _ (ord == .seqCst) (atomPty .store t) R _ xr xp xv m
      (opVal_reg c _ _ _ hns1 (by rw [hkeep2 _ hnf1]; exact hreg1))
      (place_eq hreach (fit_store hm' hmok.2)) (by rw [hbytes]; exact hal)
      (opVal_reg c _ _ _ hns2 hreg2) (by rw [hbytes, hmask]; exact hm')
    simpa [hbytes, hmask] using this
  refine ⟨{ ts with regs := regs2, w := { ts.w with mem := m, obs := (Obs.store xp (tyBytes t) xv) :: ts.w.obs } }, ?_, ?_, ?_⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · have hS2 : Sim F D s.vals regs2 := hS1.frame (fun c' n => hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h))
    exact hS2.sub (fun n h => by simpa [extD, Inst.dst?] using h)
  · have := wrel_cons start0 _ { s.world with obs := ob :: s.world.obs } ts.w m
      (Obs.store xp (tyBytes t) xv) (wrel_skip start0 _ s.world ts.w ob hW (by simp [viewObs, ob]))
    simpa using this

/-- A shared address's lines: the space is shared memory. -/
theorem addr_shared (F : Facts) (tag : Nat) (s : Out) (p : Val) (h : F.origin.get? p.id = some .shared) :
    (runLM (addr p) F tag s).1.1 = .shared := by
  by_cases hw : F.win p = true
  · simp [addr, isWin, facts?, runLM_bind, runLM_read, hw]
  · simp only [Bool.not_eq_true] at hw
    have h' : F.origin[p.id]? = some .shared := by simpa [Std.HashMap.get?_eq_getElem?] using h
    simp [addr, isWin, facts?, runLM_bind, runLM_read, hw, h', own, emit, emitL, runLM_modify]

/-- A shared add: its address's lines, its operand's, then one `red`. -/
theorem lower_red (F : Facts) (tag : Nat) (p x : Val) :
    (lower F tag (.call none (.intr .sharedAdd) [p, x])).lines =
      ((runLM (r x) F tag (runLM (addr p) F tag {}).2).2.lines).push
        (.ins none (.redShared (runLM (addr p) F tag {}).1.2
          (.reg (runLM (r x) F tag (runLM (addr p) F tag {}).2).1))) := by
  rw [lower_inst F tag _ rfl, inst]
  simp only [voidIntr, runLM_bind, o]
  rfl

open PtxRun in
/-- **A shared add** meets its obligation: both semantics record it with the
    word's address and the word, the block semantics having found the address
    one of shared memory, and leave the rest to the laws. -/
theorem red_ok (F : Facts) (c : PtxRun.Ctx) (start0 : Nat) (p x : Val) (hwx : ¬ Win F x)
    (htp : F.tys.get? p.id = some .i64) (htx : F.tys.get? x.id = some .i32)
    (hsh : F.origin.get? p.id = some .shared) :
    InstOk F (threadLocals c.ans start0 c.hw c.provisional) c start0 (.call none (.intr .sharedAdd) [p, x]) := by
  intro tag D s s1 ts hst huses hS hW _ hmok
  have hup : p ∈ Inst.uses (.call none (.intr .sharedAdd) [p, x]) := by simp [Inst.uses]
  have hux : x ∈ Inst.uses (.call none (.intr .sharedAdd) [p, x]) := by simp [Inst.uses]
  obtain ⟨tp, xp, hvp, htp'⟩ := hS.pres p (huses p hup)
  rw [htp] at htp'; cases htp'
  obtain ⟨tx, xx, hvx, htx'⟩ := hS.pres x (huses x hux)
  rw [htx] at htx'; cases htx'
  have hnx := hS.normd x .i32 xx (huses x hux) hvx
  let o : Obs := .call (.intr .sharedAdd) [.sc .i64 xp, .sc .i32 xx]
  -- the block semantics' step: the word is shared memory's, and the call recorded
  obtain ⟨hword, rfl⟩ : sharedWord s.world.mem [.sc .i64 xp, .sc .i32 xx] = true ∧
      s1 = { s with world := { s.world with obs := o :: s.world.obs } } := by
    simp only [step1, List.mapM_cons, List.mapM_nil, hvp, hvx, bind, Option.bind, pure, callOf, obsCall, callWords,
      List.append_nil, threadLocals, Intr.find_slot, Intr.isColl, if_true, show Intr.sharedAdd ≠ .barrier by decide,
      if_false] at hst
    by_cases hw : sharedWord s.world.mem [.sc .i64 xp, .sc .i32 xx] = true
    · simp [hw, Except.toOption] at hst
      exact ⟨hw, hst.symm⟩
    · simp [hw, Except.toOption] at hst
  -- the word lies in shared memory, within 32 bits
  have hfit : ∀ k, decodeAddr xp = some (.arena, k) → k < 2 ^ 32 := by
    intro k hk
    simp only [sharedWord, hk, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hword
    have hmem : ts.w.mem = s.world.mem := (congrArg World.mem hW.1).symm
    have := hmok.2
    rw [hmem] at this
    omega
  -- the lines
  let sa := (runLM (addr p) F tag {}).2
  obtain ⟨La, hLa, hn1, R, xr, hbase, hns1, hnf1, hreach, regs1, hrun1, hreg1, hkeep1⟩ :=
    addr_sound F tag {} s.vals c ts p xp D hS (huses p hup) hvp htp
  have hS1 : Sim F D s.vals regs1 := hS.frame (fun c' n => hkeep1 _ (fun ⟨p, k, _, h⟩ => by cases h))
  obtain ⟨Lv, hLv, hn2, hnf2, hns2, regs2, hrun2, hreg2, hkeep2⟩ :=
    r_sound F tag sa s.vals c { ts with regs := regs1 } x .i32 xx D hS1 (huses x hux) hvx htx hwx
  simp only [enc, show ClifTy.i32 ≠ .i8 by decide, if_false, hnx] at hreg2
  have hsp := addr_shared F tag {} p hsh
  rw [hsp] at hreach
  have hpl : place .shared xr = xp := place_eq hreach hfit
  let core : PtxIr.Instr := .redShared { base := .reg R } (.reg (runLM (r x) F tag sa).1)
  have hlist : (lower F tag (.call none (.intr .sharedAdd) [p, x])).lines =
      ((La ++ Lv ++ [core]).map (PtxIr.Line.ins none)).toArray := by
    rw [lower_red F tag p x, hLv]
    show (sa.lines ++ _).push _ = _
    rw [hLa, hbase]; simp [core, sa]
  refine ⟨by rw [hlist]; exact straight_map _, ?_⟩
  have hm32 : PtxRun.mask 32 xx = xx := by
    rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp), hnx]
  have h1 : regs2.get? R = some xr := by rw [hkeep2 _ hnf1]; exact hreg1
  simp only [Std.HashMap.get?_eq_getElem?] at h1 hreg2
  have he : (exec c core).run { ts with regs := regs2 } =
      .ok (.step, { ts with regs := regs2, w := { ts.w with obs := o :: ts.w.obs } }) := by
    simp [core, exec, coll, opnd, getReg_plain _ _ hns1, getReg_plain _ _ hns2, h1, hreg2, hm32, ofInt_toNat, hpl,
      show (Intr.sharedAdd == Intr.barrier) = false from rfl, o, MonadState.get, getThe, MonadStateOf.get,
      StateT.get, set, MonadStateOf.set, StateT.set, pure, StateT.pure, Except.pure, StateT.bind, StateT.run, bind,
      Except.bind]
  refine ⟨{ ts with regs := regs2, w := { ts.w with obs := o :: ts.w.obs } }, ?_, ?_,
    wrel_step start0 _ s.world ts.w _ _ o o hW rfl rfl rfl (by simpa using hW.1)⟩
  · rw [hlist, instrsOf_map, steps_append c _ _ ts _ (by rw [steps_append c _ _ ts _ hrun1]; exact hrun2)]
    simp only [StateT.run] at he
    simp [steps, StateT.run, he, bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure]
  · have hS2 : Sim F D s.vals regs2 := hS1.frame (fun c' n => hkeep2 _ (fun ⟨p, k, _, h⟩ => by cases h))
    exact hS2.sub (fun n h => by simpa [extD, Inst.dst?] using h)

/-- The atomics the compiler takes, by shape. -/
theorem takesAtomic_cases (F : Facts) (d : Option Val) (a : Atomic) (args : List Val)
    (h : takesAtomic F d a args = true) :
    (∃ dv t k o op p x, d = some dv ∧ a = .rmw t k o ∧ args = [p, x] ∧ atomOp k = some op ∧
      atomAt F p t = true ∧ isTy F x [t] = true) ∨
    (∃ dv t o p, d = some dv ∧ a = .load t o ∧ args = [p] ∧ atomAt F p t = true) ∨
    (∃ t o v p, d = none ∧ a = .store t o ∧ args = [v, p] ∧ atomAt F p t = true ∧ isTy F v [t] = true) := by
  rcases d with _ | dv <;> rcases a with ⟨t, o⟩ | ⟨t, o⟩ | ⟨t, k, o⟩ | t | _ <;>
    rcases args with _ | ⟨p, _ | ⟨x, _ | ⟨y, rest⟩⟩⟩ <;>
    simp only [takesAtomic, Bool.and_eq_true, Option.isSome_iff_exists] at h <;>
    first
    | (obtain ⟨⟨hp, hx⟩, op, hop⟩ := h; exact Or.inl ⟨_, _, _, _, op, _, _, rfl, rfl, rfl, hop, hp, hx⟩)
    | exact Or.inr (Or.inl ⟨_, _, _, _, rfl, rfl, rfl, h⟩)
    | (obtain ⟨hp, hv⟩ := h; exact Or.inr (Or.inr ⟨_, _, _, _, rfl, rfl, rfl, hp, hv⟩))
    | (simp at h)

/-- An atomic's address is 64 bits and its type a word or a double word. -/
theorem atomAt_spec (F : Facts) (p : Val) (t : ClifTy) (h : atomAt F p t = true) :
    F.tys.get? p.id = some .i64 ∧ (t = .i32 ∨ t = .i64) := by
  simp only [atomAt, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨hp, -⟩, ht⟩, -⟩ := h
  obtain ⟨tp, htp, htp'⟩ := isTy_spec F p _ hp
  simp at htp'; subst htp'
  exact ⟨htp, ht⟩


set_option maxHeartbeats 1000000 in
/-- A call the compiler takes is a barrier, a shuffle, a ballot, a `dp4a`, a
    `prmt`, an `mma`, an approximation, an atomic or a shared add. -/
theorem takes_call_none (F : Facts) (d : Option Val) (cal : Callee) (args : List Val)
    (hat : ¬ ∃ a, cal = .atomic a)
    (hb : ¬ (d = none ∧ cal = .intr .barrier ∧ args = []))
    (hs : ¬ (∃ dv v m, d = some dv ∧ (cal = .intr .shflBfly ∨ cal = .intr .shflIdx) ∧ args = [v, m]))
    (hv : ¬ (∃ dv v, d = some dv ∧ cal = .intr .ballot ∧ args = [v]))
    (hd4 : ¬ (∃ dv a b e, d = some dv ∧ (cal = .intr .dp4a ∨ cal = .intr .prmt) ∧ args = [a, b, e]))
    (hmm : ¬ (∃ dv a b e, d = some dv ∧ cal = .intr .mma ∧ args = [a, b, e]))
    (ha : ¬ (∃ dv v i u, d = some dv ∧ cal = .intr i ∧ Ptx.approxOp i = some u ∧ args = [v]))
    (hsa : ¬ (∃ p x, d = none ∧ cal = .intr .sharedAdd ∧ args = [p, x])) :
    takes F (.call d cal args) = false := by
  by_cases hv0 : ∃ c', d = none ∧ cal = .intr c'
  · obtain ⟨c', rfl, rfl⟩ := hv0
    simp only [takes]
    rcases c' <;> rcases args with _ | ⟨p, _ | ⟨x, _ | ⟨y, rest⟩⟩⟩ <;> simp_all [takesVoid]
  · have hw3 : ∀ dv c a b e, d = some dv → cal = .intr c → args = [a, b, e] → Ptx.word3 c = false := by
      intro dv c a b e hd hc ha3
      cases c <;> simp_all [Ptx.word3]
    unfold takes
    split <;> simp_all [Option.isSome_iff_exists]

open PtxRun in
/-- **Every instruction the compiler takes meets its obligation.** -/
theorem insts_ok (F : Facts) (lc : Sem.Locals) (c : PtxRun.Ctx) (start0 : Nat)
    (hdefs : defsOk F = true) (hwin : winOk F = true) (htys : tysOk F = true) (htk : takesAll F = true)
    (hlc : lc = threadLocals c.ans start0 c.hw c.provisional) (hans : AnsOk c.ans)
    (hhw : HwOk c.hw) (hwarp : F.warp = true → (c.special.lookup "ntid.x").getD 0 ≤ 32) :
    ∀ b ∈ F.blocks, ∀ i ∈ b.insts, isTerm i = false → InstOk F lc c start0 i := by
  intro b hb i hi hterm
  have htk' := takes_inst F htk b hb i hi
  obtain ⟨hvu, hw64⟩ := winOk_inst F hwin b hb i hi
  have hd := fun d (h : Inst.dst? i = some d) => defsOk_inst F hdefs b hb i hi d h
  cases i with
  | iconst d ty k =>
      obtain ⟨hty, hng, hlit, hside, hfl, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [takes] at htk'
      simp only [negOf] at hng
      simp only [defSide, Bool.not_eq_true'] at hside
      by_cases h8 : ty = .i8
      · subst h8
        simp only [litOf, beq_self_eq_true, if_true] at hlit
        simp [flagOf] at hfl
        exact iconst8_ok (hmo := hmo) F lc c start0 d k (by omega) hty hlit hside hng
      · have hint : IntT ty := by
          simp at htk'
          rcases htk' with rfl | rfl | rfl | rfl
          · exact absurd rfl h8
          all_goals simp [IntT]
        have h8' : (ty == .i8) = false := by simpa using h8
        simp only [litOf, h8', Bool.false_eq_true, if_false] at hlit
        exact iconst_ok (hmo := hmo) F lc c start0 d ty k hint hty hlit hside hng
  | fconst d ty b0 =>
      obtain ⟨hty, hng, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [takes, Bool.or_eq_true, beq_iff_eq] at htk'
      simp only [negOf] at hng
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [litOf] at hlit
      exact fconst_ok (hmo := hmo) F lc c start0 d ty b0 htk' hty hlit hside hng
  | iadd d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      have htd : ∀ t, F.tys.get? a.id = some t → F.tys.get? d.id = some t := resTy_same F _ d a rfl hty
      simp only [takes, Bool.and_eq_true, beq_iff_eq] at htk'
      obtain ⟨hia, hsame⟩ := htk'
      simp only [defSide, Bool.and_eq_true, beq_iff_eq] at hside
      by_cases hw : F.win a = true ∨ F.win b = true
      · have hwd : F.win d = true := by rw [hside.1]; rcases hw with h | h <;> simp [h]
        rcases hw with ha | hb'
        · have hwb : F.win b = false := hvu b (by simp [valUses, ha])
          have hta := hw64 a (by simp [Inst.uses]) ha
          exact iaddw_ok (hmo := hmo) F lc c start0 d a b (Or.inl ((win_iff F a).mp ha))
            (fun ⟨_, h⟩ => not_win_of F b hwb h) hta (by rw [hsame, hta]) (htd _ hta) hwd hlit
        · have hwa : F.win a = false := by
            cases h : F.win a
            · rfl
            · have := hvu b (by simp [valUses, h]); rw [hb'] at this; cases this
          have htb := hw64 b (by simp [Inst.uses]) hb'
          have hta : F.tys.get? a.id = some .i64 := by rw [← hsame]; exact htb
          exact iaddw_ok (hmo := hmo) F lc c start0 d a b (Or.inr ((win_iff F b).mp hb'))
            (fun ⟨h, _⟩ => not_win_of F a hwa h) hta htb (htd _ hta) hwd hlit
      · have hwa : F.win a = false := by simpa using (not_or.mp hw).1
        have hwb : F.win b = false := by simpa using (not_or.mp hw).2
        have hwd : F.win d = false := by rw [hside.1, hwa, hwb]; rfl
        exact iadd_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a hwa) (not_win_of F b hwb) hwd
          (intT_of_isTy F a hia) htd hlit
  | isub d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.and_eq_true, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      exact isub_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside.1
        (fun t ht => ⟨intT_of_isTy F a htk' t ht, resTy_same F _ d a rfl hty t ht⟩) hlit
  | imul d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.and_eq_true, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      have hwa := not_win_of F a (hvu a (by simp [valUses, Inst.uses]))
      have hT : ∀ t, F.tys.get? a.id = some t → IntT t ∧ F.tys.get? d.id = some t :=
        fun t ht => ⟨intT_of_isTy F a htk' t ht, resTy_same F _ d a rfl hty t ht⟩
      cases hwm : F.wideMul a b
      · exact imul_ok (hmo := hmo) F lc c start0 d a b hwa
          (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside.1 hT hlit hwm
      · exact imulWide_ok (hmo := hmo) F lc c start0 d a b hwa hside.1 hT hlit hwm
  | udiv d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.and_eq_true, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      exact udiv_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside.1
        (fun t ht => ⟨intT_of_isTy F a htk' t ht, resTy_same F _ d a rfl hty t ht⟩) hlit
  | ibin d k a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.and_eq_true, Bool.not_eq_true'] at hside
      simp only [takes, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at htk'
      obtain ⟨hk, hia⟩ := htk'
      have hTa := fun t ht => (⟨intT_of_isTy F a hia t ht, resTy_same F _ d a rfl hty t ht⟩ :
        IntT t ∧ F.tys.get? d.id = some t)
      have hk' : k = .umin ∨ k = .umax := by revert hk; cases k <;> decide
      rcases hk' with rfl | rfl
      · exact umin_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
          (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside.1 hTa hlit
      · exact umax_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
          (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside.1 hTa hlit
  | iext d k ty a =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes, Bool.and_eq_true, beq_iff_eq] at htk'
      obtain ⟨⟨hkr, hty16⟩, hia⟩ := htk'
      obtain rfl : k = .reduce := by revert hkr; cases k <;> decide
      obtain rfl : ty = .i16 := by revert hty16; cases ty <;> decide
      simp only [resTy, Option.map_some] at hty
      exact reduce16_ok (hmo := hmo) F lc c start0 d a (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside
        (fun t ht => by obtain ⟨t', h1, h2⟩ := isTy_spec F a _ hia; rw [h1] at ht; cases ht; simpa using h2) hty hlit
  | band d a b =>
      obtain ⟨hty, hng, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [negOf] at hng
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      exact band_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside
        (fun t ht => ⟨logicT_of_isTy F a htk' t ht, resTy_same F _ d a rfl hty t ht⟩) hlit hng
  | bor d a b =>
      obtain ⟨hty, hng, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [negOf] at hng
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      exact bor_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside
        (fun t ht => ⟨logicT_of_isTy F a htk' t ht, resTy_same F _ d a rfl hty t ht⟩) hlit hng
  | bxor d a b =>
      obtain ⟨hty, hng, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [negOf] at hng
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      exact bxor_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside
        (fun t ht => ⟨logicT_of_isTy F a htk' t ht, resTy_same F _ d a rfl hty t ht⟩) hlit hng
  | ishl d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.and_eq_true, Bool.not_eq_true'] at hside
      simp only [takes, Bool.and_eq_true] at htk'
      exact ishl_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside.1
        (fun t ht => ⟨intT_of_isTy F a htk'.1 t ht, resTy_same F _ d a rfl hty t ht⟩)
        (intT_of_isTy F b htk'.2) hlit
  | ushr d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.and_eq_true, Bool.not_eq_true'] at hside
      simp only [takes, Bool.and_eq_true] at htk'
      exact ushr_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside.1
        (fun t ht => ⟨intT_of_isTy F a htk'.1 t ht, resTy_same F _ d a rfl hty t ht⟩)
        (intT_of_isTy F b htk'.2) hlit
  | ishift d k a b =>
      simp only [takes, Bool.and_eq_true] at htk'
      obtain ⟨⟨hk, hia⟩, hib⟩ := htk'
      obtain rfl : k = .sshr := by revert hk; cases k <;> decide
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.and_eq_true, Bool.not_eq_true'] at hside
      exact sshr_ok (hmo := hmo) F lc c start0 d a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside.1
        (fun t ht => ⟨intT_of_isTy F a hia t ht, resTy_same F _ d a rfl hty t ht⟩) (intT_of_isTy F b hib) hlit
  | uextend64 d a =>
      obtain ⟨hty, hng, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [negOf] at hng
      simp only [defSide, Bool.not_eq_true'] at hside
      have hwa := not_win_of F a (hvu a (by simp [valUses, Inst.uses]))
      cases hl : F.lits.get? d.id with
      | none => exact uextend_ok (hmo := hmo) F lc c start0 d a hwa hside hty hl
      | some l =>
          rw [hl] at hlit
          simp only [litOf] at hlit
          by_cases h32 : F.tys.get? a.id = some .i32
          · simp only [h32, beq_self_eq_true, if_true] at hlit
            exact uextend_lit_ok (hmo := hmo) F lc c start0 d a l h32 hty hlit.symm hl hside hng
          · have : (F.tys.get? a.id == some .i32) = false := by simpa using h32
            simp only [this, Bool.false_eq_true, if_false] at hlit; cases hlit
  | sextend64 d a =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [litOf] at hlit
      simp only [defSide, Bool.and_eq_true, Bool.not_eq_true', bne_iff_ne, ne_eq] at hside
      exact sextend_ok (hmo := hmo) F lc c start0 d a (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside.1 hty hlit
        hside.2
  | ireduce32 d a =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      exact ireduce_ok (hmo := hmo) F lc c start0 d a (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside hty hlit
  | icmp d cc a b =>
      obtain ⟨hty, hng, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      have hwa := not_win_of F a (hvu a (by simp [valUses, Inst.uses]))
      have hwb := not_win_of F b (hvu b (by simp [valUses, Inst.uses]))
      simp only [takes, Bool.or_eq_true, Bool.and_eq_true] at htk'
      rcases htk' with hi | ⟨h8, hcc⟩
      · exact icmp_ok (hmo := hmo) F lc c start0 d cc a b hwa hwb hside (intT_of_isTy F a hi) hty hlit
      · obtain ⟨t, ht, h8'⟩ := isTy_spec F a _ h8
        simp at h8'; subst h8'
        simp only [negOf, ht, bne_self_eq_false] at hng
        have hcc' : cc = .eq ∨ cc = .ne := by revert hcc; cases cc <;> decide
        exact icmp8_ok (hmo := hmo) F lc c start0 d cc a b hcc' hwa hwb hside ht hty hlit hng
  | fun1 d k a =>
      cases k with
      | sqrt =>
        obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
        simp only [litOf] at hlit
        simp only [defSide, Bool.not_eq_true'] at hside
        simp only [takes, Bool.and_eq_true] at htk'
        obtain ⟨t, ht, h32⟩ := isTy_spec F a _ htk'.2
        simp at h32; subst h32
        exact sqrt_ok (hmo := hmo) F lc c start0 d a hhw (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside ht
          (resTy_same F _ d a rfl hty _ ht) hlit
      | fabs =>
        obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
        simp only [litOf] at hlit
        simp only [defSide, Bool.not_eq_true'] at hside
        simp only [takes, Bool.and_eq_true] at htk'
        obtain ⟨t, ht, h32⟩ := isTy_spec F a _ htk'.2
        simp at h32; subst h32
        exact fabs_ok (hmo := hmo) F lc c start0 d a (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside ht
          (resTy_same F _ d a rfl hty _ ht) hlit
      | _ => simp only [takes, Bool.and_eq_true] at htk'; exact absurd htk'.1 (by decide)
  | fma d a b e =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      obtain ⟨t, ht, h32⟩ := isTy_spec F a _ htk'
      simp at h32; subst h32
      exact fma_ok (hmo := hmo) F lc c start0 d a b e hhw (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) (not_win_of F e (hvu e (by simp [valUses, Inst.uses])))
        hside ht (resTy_same F _ d a rfl hty _ ht) hlit
  | fcmp d cc a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      obtain ⟨t, ht, h32⟩ := isTy_spec F a _ htk'
      simp at h32; subst h32
      exact fcmp_ok (hmo := hmo) F lc c start0 d cc a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside ht hty hlit
  | select d cv a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq, bne_iff_ne, ne_eq] at hside
      exact select_ok (hmo := hmo) F lc c start0 d cv a b (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) (not_win_of F cv (hvu cv (by simp [valUses, Inst.uses])))
        hside.1.1 (fun t ht => ⟨tysOk_scalar F htys a t ht, fun h => hside.2 (h ▸ ht), resTy_same F _ d a rfl hty t ht⟩)
        hside.1.2 hlit
  | fneg d a =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      exact fneg_ok (hmo := hmo) F lc c start0 d a (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside
        (resTy_same F _ d a rfl hty) hlit
  | load d lo a =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes, Bool.and_eq_true] at htk'
      obtain ⟨hia, hkind⟩ := htk'
      obtain ⟨ta, hta, hta'⟩ := isTy_spec F a _ hia
      simp at hta'; subst hta'
      have go := fun (hk : lo.kind = .plain ∨ lo.kind = .uload8 ∨ lo.kind = .uload16 ∨ lo.kind = .uload32)
        (hlt : LdTy lo.ty) (hku : lo.kind ≠ .plain → IntT lo.ty ∧ (lo.kind = .uload32 → lo.ty ≠ .i16)) =>
        load_ok (hmo := hmo) F lc c start0 d a lo hk hlt hku hta hty hlit hside
      cases hk : lo.kind <;> simp only [hk] at hkind
      · simp at hkind
        exact go (Or.inl hk) (by rcases hkind with h | h | h | h | h <;> simp [LdTy, h])
          (fun h => absurd hk h)
      · simp at hkind
        exact go (Or.inr (Or.inl hk)) (by rcases hkind with h | h | h <;> simp [LdTy, h])
          (fun _ => ⟨by rcases hkind with h | h | h <;> simp [IntT, h], fun h => by rw [hk] at h; cases h⟩)
      · simp at hkind
        exact go (Or.inr (Or.inr (Or.inl hk))) (by rcases hkind with h | h | h <;> simp [LdTy, h])
          (fun _ => ⟨by rcases hkind with h | h | h <;> simp [IntT, h], fun h => by rw [hk] at h; cases h⟩)
      · simp at hkind
        exact go (Or.inr (Or.inr (Or.inr hk))) (by rcases hkind with h | h <;> simp [LdTy, h])
          (fun _ => ⟨by rcases hkind with h | h <;> simp [IntT, h], fun _ => by rcases hkind with h | h <;> simp [h]⟩)
      all_goals cases hkind
  | store v a =>
      simp only [takes, Bool.and_eq_true] at htk'
      obtain ⟨hia, hiv⟩ := htk'
      obtain ⟨ta, hta, hta'⟩ := isTy_spec F a _ hia
      simp at hta'; subst hta'
      exact store_ok F lc c start0 v a (not_win_of F v (hvu v (by simp [valUses])))
        hta (fun t ht => by
          obtain ⟨t', h1, h2⟩ := isTy_spec F v _ hiv
          rw [h1] at ht; cases ht
          simp at h2; rcases h2 with h | h | h | h | h <;> simp [LdTy, h])
  | istore8 v a =>
      simp only [takes, Bool.and_eq_true] at htk'
      obtain ⟨hia, hiv⟩ := htk'
      obtain ⟨ta, hta, hta'⟩ := isTy_spec F a _ hia
      simp at hta'; subst hta'
      exact istore8_ok F lc c start0 v a (not_win_of F v (hvu v (by simp [valUses])))
        hta (fun t ht => by
          obtain ⟨t', h1, h2⟩ := isTy_spec F v _ hiv
          rw [h1] at ht; cases ht
          simp at h2; rcases h2 with h | h | h | h | h <;> simp [LdTy, h])
  | bitcast d ty a =>
      obtain ⟨hty, -, hlit, hside, hfl, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [flagOf, resTy, Option.map_some, bne_iff_ne, ne_eq, Option.some.injEq] at hfl
      simp only [takes] at htk'
      have hld : LdTy ty := by
        have := tysOk_scalar F htys d ty hty
        rcases this with h | h | h | h | h | h
        · exact absurd h hfl
        all_goals simp [LdTy, h]
      exact bitcast_ok (hmo := hmo) F lc c start0 d ty a (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside hld
        (fun t ht => by
          obtain ⟨t', h1, h2⟩ := isTy_spec F a _ htk'
          rw [h1] at ht; cases ht
          simp at h2; rcases h2 with h | h | h | h | h <;> simp [LdTy, h]) hty hlit
  | call d cal args =>
      by_cases hb' : d = none ∧ cal = .intr .barrier ∧ args = []
      · obtain ⟨rfl, rfl, rfl⟩ := hb'
        rw [hlc]; exact barrier_ok F c start0 hwarp
      by_cases hs' : ∃ dv v m, d = some dv ∧ (cal = .intr .shflBfly ∨ cal = .intr .shflIdx) ∧ args = [v, m]
      · obtain ⟨dv, v, m, rfl, hcal, rfl⟩ := hs'
        obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd dv rfl
        simp only [defSide, Bool.not_eq_true'] at hside
        rw [hlc]
        rcases hcal with rfl | rfl
        · simp only [resTy, Intr.sig, Option.map_some] at hty
          simp only [takes, Bool.and_eq_true] at htk'
          obtain ⟨hv', hm'⟩ := htk'
          obtain ⟨tv, htv, htv'⟩ := isTy_spec F v _ hv'
          obtain ⟨tm, htm, htm'⟩ := isTy_spec F m _ hm'
          simp at htv' htm'; subst htv'; subst htm'
          exact shfl_ok (hmo := hmo) F c start0 hans dv v m .shflBfly .bfly (Or.inl ⟨rfl, rfl⟩)
            (not_win_of F v (hvu v (by simp [valUses, Inst.uses])))
            (not_win_of F m (hvu m (by simp [valUses, Inst.uses]))) hside htv htm hty hlit
            (fun tag => by rw [lower_inst F tag _ rfl, inst]; simp [o, emit, emitL])
        · simp only [resTy, Intr.sig, Option.map_some] at hty
          simp only [takes, Bool.and_eq_true] at htk'
          obtain ⟨hv', hm'⟩ := htk'
          obtain ⟨tv, htv, htv'⟩ := isTy_spec F v _ hv'
          obtain ⟨tm, htm, htm'⟩ := isTy_spec F m _ hm'
          simp at htv' htm'; subst htv'; subst htm'
          exact shfl_ok (hmo := hmo) F c start0 hans dv v m .shflIdx .idx (Or.inr ⟨rfl, rfl⟩)
            (not_win_of F v (hvu v (by simp [valUses, Inst.uses])))
            (not_win_of F m (hvu m (by simp [valUses, Inst.uses]))) hside htv htm hty hlit
            (fun tag => by rw [lower_inst F tag _ rfl, inst]; simp [o, emit, emitL])
      by_cases hv' : ∃ dv v, d = some dv ∧ cal = .intr .ballot ∧ args = [v]
      · obtain ⟨dv, v, rfl, rfl, rfl⟩ := hv'
        obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd dv rfl
        simp only [defSide, Bool.not_eq_true'] at hside
        simp only [resTy, Intr.sig, Option.map_some] at hty
        simp only [takes] at htk'
        obtain ⟨tv, htv, htv'⟩ := isTy_spec F v _ htk'
        simp at htv'; subst htv'
        rw [hlc]
        exact ballot_ok (hmo := hmo) F c start0 hans dv v (not_win_of F v (hvu v (by simp [valUses, Inst.uses]))) hside htv hty
          hlit
      by_cases hd4 : ∃ dv a b e, d = some dv ∧ (cal = .intr .dp4a ∨ cal = .intr .prmt) ∧ args = [a, b, e]
      · obtain ⟨dv, a, b, e, rfl, hcal, rfl⟩ := hd4
        have hw : ∃ ic f mk, cal = .intr ic ∧
            ((ic = .dp4a ∧ f = Intr.dp4aVal ∧ mk = PtxIr.Instr.dp4a) ∨ (ic = .prmt ∧ f = Intr.prmtVal ∧ mk = PtxIr.Instr.prmt)) := by
          rcases hcal with rfl | rfl
          · exact ⟨_, _, _, rfl, .inl ⟨rfl, rfl, rfl⟩⟩
          · exact ⟨_, _, _, rfl, .inr ⟨rfl, rfl, rfl⟩⟩
        obtain ⟨ic, f, mk, rfl, hw⟩ := hw
        obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd dv rfl
        simp only [defSide, Bool.not_eq_true'] at hside
        simp only [resTy, Intr.sig, Option.map_some] at hty
        simp only [takes, Bool.and_eq_true] at htk'
        obtain ⟨⟨⟨-, ha'⟩, hb'⟩, he'⟩ := htk'
        obtain ⟨ta, hta, hta'⟩ := isTy_spec F a _ ha'
        obtain ⟨tb, htb, htb'⟩ := isTy_spec F b _ hb'
        obtain ⟨te, hte, hte'⟩ := isTy_spec F e _ he'
        simp at hta' htb' hte'; subst hta'; subst htb'; subst hte'
        rw [hlc]
        exact word3_ok (hmo := hmo) F c start0 ic f mk hw dv a b e
          (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
          (not_win_of F b (hvu b (by simp [valUses, Inst.uses])))
          (not_win_of F e (hvu e (by simp [valUses, Inst.uses]))) hside hta htb hte
          (by rcases hw with ⟨rfl, -, -⟩ | ⟨rfl, -, -⟩ <;> exact hty) hlit
          (fun tag => by
            rw [lower_inst F tag _ rfl, inst]
            rcases hw with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ <;> simp [intr3, o, emit, emitL])
      by_cases hmm : ∃ dv a b e, d = some dv ∧ cal = .intr .mma ∧ args = [a, b, e]
      · obtain ⟨dv, a, b, e, rfl, rfl, rfl⟩ := hmm
        obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd dv rfl
        simp only [defSide, Bool.not_eq_true'] at hside
        simp only [resTy, Intr.sig, Option.map_some] at hty
        simp only [takes, Bool.and_eq_true] at htk'
        obtain ⟨⟨⟨-, ha'⟩, hb'⟩, he'⟩ := htk'
        obtain ⟨ta, hta, hta'⟩ := isTy_spec F a _ ha'
        obtain ⟨tb, htb, htb'⟩ := isTy_spec F b _ hb'
        obtain ⟨te, hte, hte'⟩ := isTy_spec F e _ he'
        simp at hta' htb' hte'; subst hta'; subst htb'; subst hte'
        rw [hlc]
        exact mma_ok (hmo := hmo) F c start0 hans dv a b e
          (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
          (not_win_of F b (hvu b (by simp [valUses, Inst.uses])))
          (not_win_of F e (hvu e (by simp [valUses, Inst.uses]))) hside hta htb hte hty hlit
      by_cases ha' : ∃ dv v i u, d = some dv ∧ cal = .intr i ∧ Ptx.approxOp i = some u ∧ args = [v]
      · obtain ⟨dv, v, i, u, rfl, rfl, hu, rfl⟩ := ha'
        obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd dv rfl
        simp only [defSide, Bool.not_eq_true'] at hside
        have hsig : i.sig.2 = some .f32 := by revert hu; cases i <;> simp [Ptx.approxOp, Intr.sig]
        simp only [resTy, hsig, Option.map_some] at hty
        have htk2 : isTy F v [.f32] = true := by
          revert htk' hu; cases i <;> simp [takes, Ptx.approxOp]
        obtain ⟨tv, htv, htv'⟩ := isTy_spec F v _ htk2
        simp at htv'; subst htv'
        rw [hlc]
        exact approx_ok (hmo := hmo) F c start0 hhw dv v i u hu (not_win_of F v (hvu v (by simp [valUses, Inst.uses]))) hside htv
          hty hlit
      by_cases hat : ∃ a, cal = .atomic a
      · obtain ⟨a, rfl⟩ := hat
        simp only [takes] at htk'
        rcases takesAtomic_cases F d a args htk' with
          ⟨dv, t, k, ord, op, p, x, rfl, rfl, rfl, hop, hat, hx⟩ | ⟨dv, t, ord, p, rfl, rfl, rfl, hat⟩ |
          ⟨t, ord, v, p, rfl, rfl, rfl, hat, hv⟩
        · obtain ⟨htp, ht⟩ := atomAt_spec F p t hat
          obtain ⟨tx, htx, htx'⟩ := isTy_spec F x _ hx
          simp at htx'; subst htx'
          obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd dv rfl
          simp only [defSide, Bool.not_eq_true'] at hside
          simp only [resTy, atomicRes, Option.map_some] at hty
          exact atom_rmw_ok F lc c start0 dv p x tx k ord op hop ht
            (not_win_of F x (hvu x (by simp [valUses, Inst.uses]))) hside htp htx hty hlit hmo
        · obtain ⟨htp, ht⟩ := atomAt_spec F p t hat
          obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd dv rfl
          simp only [defSide, Bool.not_eq_true'] at hside
          simp only [resTy, atomicRes, Option.map_some] at hty
          exact atom_load_ok F lc c start0 dv p t ord ht hside htp hty hlit hmo
        · obtain ⟨htp, ht⟩ := atomAt_spec F p t hat
          obtain ⟨tv, htv, htv'⟩ := isTy_spec F v _ hv
          simp at htv'; subst htv'
          exact atom_store_ok F lc c start0 v p tv ord ht
            (not_win_of F v (hvu v (by simp [valUses, Inst.uses]))) htp htv
      by_cases hsa : ∃ p x, d = none ∧ cal = .intr .sharedAdd ∧ args = [p, x]
      · obtain ⟨p, x, rfl, rfl, rfl⟩ := hsa
        simp only [takes, takesVoid, sharedAt, Bool.and_eq_true, beq_iff_eq] at htk'
        obtain ⟨⟨hp, hsh⟩, hx⟩ := htk'
        obtain ⟨tp, htp, htp'⟩ := isTy_spec F p _ hp
        obtain ⟨tx, htx, htx'⟩ := isTy_spec F x _ hx
        simp at htp' htx'; subst htp'; subst htx'
        rw [hlc]
        exact red_ok F c start0 p x (not_win_of F x (hvu x (by simp [valUses, Inst.uses]))) htp htx hsh
      rw [takes_call_none F d cal args hat hb' hs' hv' hd4 hmm ha' hsa] at htk'; cases htk'
  | storeTyped ty v a =>
      simp only [takes, Bool.and_eq_true, beq_iff_eq] at htk'
      obtain ⟨⟨hia, hvt⟩, hty⟩ := htk'
      obtain ⟨ta, hta, hta'⟩ := isTy_spec F a _ hia
      simp at hta'; subst hta'
      exact storeTyped_ok F lc c start0 ty v a (not_win_of F v (hvu v (by simp [valUses])))
        hta hvt (by simp at hty; rcases hty with h | h | h | h | h <;> simp [LdTy, h])
  | ctz d a =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      obtain ⟨t, h1, h2⟩ := isTy_spec F a _ htk'
      have ht : t = .i32 ∨ t = .i64 := by simpa using h2
      exact ctz_ok (hmo := hmo) F lc c start0 d a t ht (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside h1
        (resTy_same F _ d a rfl hty t h1) hlit
  | popcnt d a =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      obtain ⟨t, h1, h2⟩ := isTy_spec F a _ htk'
      have ht : t = .i32 := by simpa using h2
      subst ht
      exact popcnt_ok (hmo := hmo) F lc c start0 d a (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside h1
        (resTy_same F _ d a rfl hty .i32 h1) hlit
  | fadd d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      exact fadd_ok (hmo := hmo) F lc c start0 d a b hhw (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside
        (fun t ht => ⟨by obtain ⟨t', h1, h2⟩ := isTy_spec F a _ htk'; rw [h1] at ht; cases ht; simpa using h2,
          resTy_same F _ d a rfl hty t ht⟩) hlit
  | fsub d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      exact fsub_ok (hmo := hmo) F lc c start0 d a b hhw (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside
        (fun t ht => ⟨by obtain ⟨t', h1, h2⟩ := isTy_spec F a _ htk'; rw [h1] at ht; cases ht; simpa using h2,
          resTy_same F _ d a rfl hty t ht⟩) hlit
  | fmul d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      exact fmul_ok (hmo := hmo) F lc c start0 d a b hhw (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside
        (fun t ht => ⟨by obtain ⟨t', h1, h2⟩ := isTy_spec F a _ htk'; rw [h1] at ht; cases ht; simpa using h2,
          resTy_same F _ d a rfl hty t ht⟩) hlit
  | fmax d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      exact fmax_ok (hmo := hmo) F lc c start0 d a b hhw (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside
        (fun t ht => ⟨by obtain ⟨t', h1, h2⟩ := isTy_spec F a _ htk'; rw [h1] at ht; cases ht; simpa using h2,
          resTy_same F _ d a rfl hty t ht⟩) hlit
  | fmin d a b =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      exact fmin_ok (hmo := hmo) F lc c start0 d a b hhw (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
        (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside
        (fun t ht => ⟨by obtain ⟨t', h1, h2⟩ := isTy_spec F a _ htk'; rw [h1] at ht; cases ht; simpa using h2,
          resTy_same F _ d a rfl hty t ht⟩) hlit
  | fcvtFromSint d ty a =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes, Bool.and_eq_true, beq_iff_eq] at htk'
      obtain ⟨rfl, hk⟩ := htk'
      exact sitof_ok (hmo := hmo) F lc c start0 d a (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside
        (fun t ht => by obtain ⟨t', h1, h2⟩ := isTy_spec F a _ hk; rw [h1] at ht; cases ht; simpa [IntT] using h2)
        hty hlit
  | fconv d k ty a =>
      obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
      simp only [resTy, Option.map_some] at hty
      simp only [litOf] at hlit
      simp only [defSide, Bool.not_eq_true'] at hside
      simp only [takes] at htk'
      cases k with
      | fromUint =>
        simp only [Bool.and_eq_true, beq_iff_eq] at htk'
        obtain ⟨rfl, hk⟩ := htk'
        exact uitof_ok (hmo := hmo) F lc c start0 d a (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside
          (fun t ht => by obtain ⟨t', h1, h2⟩ := isTy_spec F a _ hk; rw [h1] at ht; cases ht; simpa using h2)
          hty hlit
      | demote =>
        simp only [Bool.and_eq_true, beq_iff_eq] at htk'
        obtain ⟨rfl, hk⟩ := htk'
        obtain ⟨t, ht, h64⟩ := isTy_spec F a _ hk
        simp at h64; subst h64
        exact demote_ok (hmo := hmo) F lc c start0 d a hhw (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside ht
          hty hlit
      | toSint =>
        simp only [Bool.and_eq_true, beq_iff_eq] at htk'
        obtain ⟨rfl, hk⟩ := htk'
        obtain ⟨t, ht, h32⟩ := isTy_spec F a _ hk
        simp at h32; subst h32
        exact ftosi_ok (hmo := hmo) F lc c start0 d a (not_win_of F a (hvu a (by simp [valUses, Inst.uses]))) hside ht
          hty hlit
  | fbin d k a b =>
      cases k with
      | fdiv =>
        obtain ⟨hty, -, hlit, hside, -, hmo⟩ := hd d rfl
        simp only [litOf] at hlit
        simp only [defSide, Bool.not_eq_true'] at hside
        simp only [takes, Bool.and_eq_true] at htk'
        exact fdiv_ok (hmo := hmo) F lc c start0 d a b hhw (not_win_of F a (hvu a (by simp [valUses, Inst.uses])))
          (not_win_of F b (hvu b (by simp [valUses, Inst.uses]))) hside
          (fun t ht => ⟨by obtain ⟨t', h1, h2⟩ := isTy_spec F a _ htk'.2; rw [h1] at ht; cases ht; simpa using h2,
            resTy_same F _ d a rfl hty t ht⟩) hlit
      | _ => simp only [takes, Bool.and_eq_true] at htk'; exact absurd htk'.1 (by decide)
  | jump t args => simp [isTerm] at hterm
  | brif cnd tb ta eb ea => simp [isTerm] at hterm
  | ret v => simp [isTerm] at hterm
  | _ => simp [takes] at htk'

/-! ## The body's layout -/

theorem instsBody_lines (F : Facts) : ∀ (tag : Nat) (is : List Inst),
    (instsBody F tag is).flatMap (·.lines.toList) = linesOf F tag is
  | _, [] => rfl
  | tag, i :: is => by simp [instsBody, linesOf, instsBody_lines F (tag + 1) is]

theorem instsBody_mem (F : Facts) : ∀ (tag : Nat) (is : List Inst) (j : Nat) (hj : j < is.length),
    lower F (tag + j) is[j] ∈ instsBody F tag is
  | _, [], j, hj => by simp at hj
  | tag, i :: is, 0, _ => by simp [instsBody]
  | tag, i :: is, j + 1, hj => by
      have := instsBody_mem F (tag + 1) is j (by simp at hj; omega)
      simp only [instsBody, List.getElem_cons_succ, List.mem_cons]
      right; rw [show tag + (j + 1) = tag + 1 + j by omega]; exact this

/-- Where each block's label is and its first instruction's position, in
    order: the label `pc` lines in, the instructions after `start` others. -/
def places (F : Facts) : Nat → Nat → List BlockData → List (Nat × Nat × Nat)
  | _, _, [] => []
  | pc, start, b :: bs => (b.ref.id, pc, start + 1) ::
      places F (pc + 1 + (linesOf F (start + 1) b.insts).length) (start + b.insts.length) bs

theorem places_keys (F : Facts) : ∀ (pc start : Nat) (bs : List BlockData),
    (places F pc start bs).map (·.1) = bs.map (·.ref.id)
  | _, _, [] => rfl
  | pc, start, b :: bs => by simp [places, places_keys F _ _ bs]

/-- **Each block sits where its place says**: its label, then its
    instructions' lines; each lowering is one of the body's. -/
theorem blocks_at (F : Facts) (c : PtxRun.Ctx) : ∀ (bs : List BlockData) (pc start : Nat),
    At c pc ((blocksBody F start bs).flatMap (·.lines.toList)) →
    ∀ b ∈ bs, ∃ p s, (b.ref.id, p, s) ∈ places F pc start bs ∧ c.body[p]? = some (.label (.blk b.ref.id)) ∧
      At c (p + 1) (linesOf F s b.insts) ∧
      ∀ j (hj : j < b.insts.length), lower F (s + j) b.insts[j] ∈ blocksBody F start bs
  | [], _, _, _ => by simp
  | b :: bs, pc, start, hat => by
      have hl : (blocksBody F start (b :: bs)).flatMap (·.lines.toList) =
          [PtxIr.Line.label (.blk b.ref.id)] ++ (linesOf F (start + 1) b.insts ++
            (blocksBody F (start + b.insts.length) bs).flatMap (·.lines.toList)) := by
        simp [blocksBody, blkLbl, instsBody_lines]
      rw [hl] at hat
      intro b' hb'
      rcases List.mem_cons.mp hb' with rfl | hb'
      · refine ⟨pc, start + 1, by simp [places], ?_, ?_, fun j hj => ?_⟩
        · have := hat 0 (by simp); simpa using this
        · exact (hat.right).left
        · simp [blocksBody, instsBody_mem F (start + 1) b'.insts j hj]
      · have hrest := (hat.right).right
        simp only [List.length_singleton] at hrest
        obtain ⟨p, s, hm, h1, h2, h3⟩ :=
          blocks_at F c bs (pc + 1 + (linesOf F (start + 1) b.insts).length) (start + b.insts.length)
            (by simpa [Nat.add_assoc] using hrest) b' hb'
        refine ⟨p, s, by simp [places, hm], h1, h2, fun j hj => ?_⟩
        simp [blocksBody, h3 j hj]

/-- A key list without repeats finds the entry of each key. -/
theorem find_key {α : Type} : ∀ (l : List (Nat × α)) (k : Nat) (x : α), (l.map (·.1)).Nodup → (k, x) ∈ l →
    l.find? (·.1 == k) = some (k, x)
  | [], _, _, _, h => by simp at h
  | (k', x') :: l, k, x, hnd, h => by
      simp only [List.map_cons, List.nodup_cons] at hnd
      rcases List.mem_cons.mp h with h | h
      · cases h; simp
      · have hne : k' ≠ k := fun he => hnd.1 (by rw [he]; exact List.mem_map.mpr ⟨(k, x), h, rfl⟩)
        simp [hne, find_key l k x hnd.2 h]

/-- A block's label's line, by its places. -/
def posOf (pl : List (Nat × Nat × Nat)) (id : Nat) : Nat := ((pl.find? (·.1 == id)).map (·.2.1)).getD 0

/-- A block's first instruction's position, by its places. -/
def startOf (pl : List (Nat × Nat × Nat)) (id : Nat) : Nat := ((pl.find? (·.1 == id)).map (·.2.2)).getD 0

/-- **The body's layout**: in a context whose body is the lowering's lines and
    whose labels are where those lines write them, the blocks are laid out as
    `run_sim` takes them, each instruction lowered at its position one of the
    body's lowerings. -/
theorem body_layout (F : Facts) (k0 : List ClifTy) (b0 : BlockData) (rest : List BlockData)
    (hb : F.blocks = b0 :: rest) (hnd : (F.blocks.map (·.ref.id)).Nodup)
    (c : PtxRun.Ctx) (hc : c.body = ((body F k0).flatMap (·.lines.toList)).toArray)
    (hlab : (labelsIn ((body F k0).flatMap (·.lines.toList))).Nodup)
    (hls : c.labels = PtxRun.labelsOf ((body F k0).flatMap (·.lines.toList))) :
    ∃ pos start, Layout F c pos start ∧ pos b0.ref.id = (prologue F k0 b0).lines.size ∧
      ∀ b ∈ F.blocks, ∀ j (hj : j < b.insts.length), lower F (start b.ref.id + j) b.insts[j] ∈ body F k0 := by
  let pl := places F (prologue F k0 b0).lines.size 0 F.blocks
  have hbody : (body F k0).flatMap (·.lines.toList) =
      (prologue F k0 b0).lines.toList ++ (blocksBody F 0 F.blocks).flatMap (·.lines.toList) := by
    simp [body, hb]
  have hat : At c (prologue F k0 b0).lines.size ((blocksBody F 0 F.blocks).flatMap (·.lines.toList)) := by
    intro k hk
    rw [hc, hbody]
    simp only [List.getElem?_toArray]
    rw [List.getElem?_append_right (by simp)]
    simp
  have hkeys : (pl.map (·.1)).Nodup := by rw [places_keys]; exact hnd
  refine ⟨posOf pl, startOf pl, ⟨fun L i hi => ?_, fun b hbm => ?_⟩, ?_, fun b hbm j hj => ?_⟩
  · rw [hls]
    apply labelsOf_line _ hlab
    rw [hc] at hi; simpa using hi
  · obtain ⟨p, s, hm, h1, h2, -⟩ := blocks_at F c F.blocks _ 0 hat b hbm
    have hf := find_key pl b.ref.id (p, s) hkeys hm
    simp only [posOf, startOf, hf, Option.map_some, Option.getD_some]
    exact ⟨h1, h2⟩
  · simp [posOf, pl, hb, places]
  · obtain ⟨p, s, hm, -, -, h3⟩ := blocks_at F c F.blocks _ 0 hat b hbm
    have hf := find_key pl b.ref.id (p, s) hkeys hm
    simp only [startOf, hf, Option.map_some, Option.getD_some]
    have := h3 j hj
    rw [hb] at this
    simp [body, hb, this]

/-! ## The prologue -/

open PtxRun in
theorem exec_mov_special (c : PtxRun.Ctx) (ts : PtxRun.TS) (ty : PtxIr.Ty) (d : PtxIr.Reg) (s : String) :
    (exec c (.mov ty d (.reg (.special s)))).run ts =
      .ok (.step, { ts with regs := ts.regs.insert d (mask ty.width ((c.special.lookup s).getD 0)) }) := by
  simp [exec, opnd, getReg, setReg]

open PtxRun in
theorem exec_mov_sym (c : PtxRun.Ctx) (ts : PtxRun.TS) (ty : PtxIr.Ty) (d : PtxIr.Reg) (n : String) (v : UInt64)
    (h : c.syms.lookup n = some v) :
    (exec c (.mov ty d (.sym n))).run ts = .ok (.step, { ts with regs := ts.regs.insert d (mask ty.width v) }) := by
  simp [exec, opnd, h, setReg]

open PtxRun in
theorem exec_ld_param (c : PtxRun.Ctx) (ts : PtxRun.TS) (nc : Bool) (ty : PtxIr.Ty) (d : PtxIr.Reg) (n : String)
    (v : UInt64) (h : c.params.lookup n = some v) :
    (exec c (.ld .param nc ty d { base := .sym n })).run ts = .ok (.step, { ts with regs := ts.regs.insert d v }) := by
  simp [exec, PtxRun.load, h, setReg]

open PtxRun in
theorem exec_cvta_shared (c : PtxRun.Ctx) (ts : PtxRun.TS) (d : PtxIr.Reg) (v : UInt64) (hd : ∀ s, d ≠ .special s)
    (h : ts.regs.get? d = some v) :
    (exec c (.cvta false .shared .u64 d (.reg d))).run ts =
      .ok (.step, { ts with regs := ts.regs.insert d (addrOf .arena (mask 32 v).toNat) }) := by
  simp only [Std.HashMap.get?_eq_getElem?] at h
  simp [exec, opnd, getReg_plain _ _ hd, h, setReg]

/-- Instructions each of which writes one register a value not read from the
    registers: their run leaves those writes, in order. -/
theorem steps_writes (c : PtxRun.Ctx) :
    ∀ (L : List (PtxIr.Instr × PtxIr.Reg × UInt64)) (ts : PtxRun.TS),
      (∀ e ∈ L, ∀ ts' : PtxRun.TS, (PtxRun.exec c e.1).run ts' =
        .ok (.step, { ts' with regs := ts'.regs.insert e.2.1 e.2.2 })) →
      (steps c (L.map (·.1))).run ts =
        .ok ((), { ts with regs := L.foldl (fun m e => m.insert e.2.1 e.2.2) ts.regs })
  | [], ts, _ => by simp [steps, StateT.run, pure, StateT.pure, Except.pure]
  | e :: L, ts, h => by
      have h1 := h e List.mem_cons_self ts
      have ih := steps_writes c L { ts with regs := ts.regs.insert e.2.1 e.2.2 }
        (fun e' he' => h e' (List.mem_cons_of_mem _ he'))
      simp only [StateT.run] at h1 ih
      simp [steps, StateT.run, h1, ih, bind, StateT.bind, Except.bind]

/-- After inserts with keys not repeated, a key inserted holds its value. -/
theorem foldl_insert_get : ∀ (L : List (PtxIr.Reg × UInt64)) (m : Std.HashMap PtxIr.Reg UInt64) (k : PtxIr.Reg)
    (v : UInt64), (L.map (·.1)).Nodup → (k, v) ∈ L → (L.foldl (fun m e => m.insert e.1 e.2) m).get? k = some v
  | [], _, _, _, _, h => by simp at h
  | (k', v') :: L, m, k, v, hnd, h => by
      simp only [List.map_cons, List.nodup_cons] at hnd
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp h with h | h
      · cases h
        rw [foldl_insert_keep L _ k' (fun hk => hnd.1 hk)]
        simp [Std.HashMap.get?_eq_getElem?]
      · exact foldl_insert_get L _ k v hnd.2 h
where
  foldl_insert_keep : ∀ (L : List (PtxIr.Reg × UInt64)) (m : Std.HashMap PtxIr.Reg UInt64) (k : PtxIr.Reg),
      k ∉ L.map (·.1) → (L.foldl (fun m e => m.insert e.1 e.2) m).get? k = m.get? k
    | [], _, _, _ => rfl
    | (k', v') :: L, m, k, hk => by
        simp only [List.map_cons, List.mem_cons, not_or] at hk
        simp only [List.foldl_cons]
        rw [foldl_insert_keep L _ k hk.2]
        simp only [Std.HashMap.get?_eq_getElem?, Std.HashMap.getElem?_insert]
        rw [if_neg (by simpa using Ne.symm hk.1)]

open PtxRun in
theorem exec_cvta_sym (c : PtxRun.Ctx) (ts : PtxRun.TS) (d : PtxIr.Reg) (n : String) (v : UInt64)
    (h : c.syms.lookup n = some v) :
    (exec c (.cvta false .shared .u64 d (.sym n))).run ts =
      .ok (.step, { ts with regs := ts.regs.insert d (addrOf .arena (mask 32 v).toNat) }) := by
  simp [exec, opnd, h, setReg]

/-- A map over a list of pairs' second parts with their indices is a map over
    the pairs with theirs. -/
theorem zipIdx_map_snd {α β γ : Type} (ps : List (α × β)) (f : β × Nat → γ) (g : (α × β) × Nat → γ)
    (h : ∀ i (hi : i < ps.length), f (ps[i].2, i) = g (ps[i], i)) :
    (ps.map (·.2)).zipIdx.map f = ps.zipIdx.map g := by
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_map, List.getElem_zipIdx, Nat.zero_add]
  exact h i (by simpa using h2)

theorem nodup_of_map {α β : Type} (f : α → β) : ∀ l : List α, (l.map f).Nodup → l.Nodup
  | [], _ => List.nodup_nil
  | a :: l, h => by
      simp only [List.map_cons, List.nodup_cons, List.mem_map, not_exists, not_and] at h ⊢
      exact ⟨fun ha => h.1 a ha rfl, nodup_of_map f l h.2⟩

/-- The register a parameter's value is held in: its window register if it is
    a windowed shared address, else its own. -/
def parReg (F : Facts) (p : Val × ClifTy) : PtxIr.Reg := if F.win p.1 then winReg p.1 else regOf p.2 p.1

open PtxRun in
/-- **The prologue**: from no registers, in a thread whose identity, shared
    base and parameters are the entry block's arguments, its lines leave each
    of the entry block's parameters held as the relation says. -/
theorem prologue_sim (F : Facts) (k0 : List ClifTy) (b0 : BlockData) (rest : List BlockData)
    (hb : F.blocks = b0 :: rest) (hent : entryOk F k0 = true) (hpok : paramsOk F = true)
    (c : PtxRun.Ctx) (w : World) (args : List V) (xs : List UInt64)
    (hlen : xs.length = b0.params.length)
    (hargs : args = (b0.params.zip xs).map fun (p, x) => .sc p.2 x)
    (hnorm : ∀ i (hi : i < b0.params.length) (hx : i < xs.length), xs[i] &&& widthMask b0.params[i].2 = xs[i])
    (h3 : xs[3]? = some (addrOf .arena 0))
    (hsp : ∀ (i : Nat) (s : String), [("ctaid.x", 0), ("tid.x", 1), ("ntid.x", 2)].lookup s = some i →
      (c.special.lookup s).getD 0 = xs.getD i 0)
    (hsm : c.syms.lookup "smem" = some 0)
    (hpl : ∀ i, i < k0.length → c.params.lookup s!"p{i}" = some (xs.getD (i + 4) 0)) :
    ∃ regs, (steps c (prologueInstrs F k0 b0)).run { regs := {}, w } = .ok ((), { regs, w }) ∧
      Sim F (fun n => ∃ p ∈ b0.params, p.1.id = n) (bindParams #[] b0.params args) regs := by
  have hent' := hent
  simp only [entryOk, hb, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true', List.all_eq_true] at hent'
  obtain ⟨⟨⟨-, hmap⟩, hk8⟩, hwz⟩ := hent'
  have hl4 : 4 ≤ b0.params.length := by
    have := congrArg List.length hmap; simp [idParams] at this; omega
  obtain ⟨p0, p1, p2, p3, ps, hps⟩ : ∃ p0 p1 p2 p3 ps, b0.params = p0 :: p1 :: p2 :: p3 :: ps := by
    rcases h : b0.params with _ | ⟨p0, _ | ⟨p1, _ | ⟨p2, _ | ⟨p3, ps⟩⟩⟩⟩ <;> rw [h] at hl4 <;> simp at hl4
    exact ⟨p0, p1, p2, p3, ps, rfl⟩
  rw [hps] at hmap
  simp only [List.map_cons, idParams, List.cons_append, List.nil_append, List.cons.injEq] at hmap
  obtain ⟨ht0, ht1, ht2, ht3, hk0⟩ := hmap
  -- which parameters are windowed
  have hwin : ∀ q i, (q, i) ∈ b0.params.zipIdx → F.win q.1 = (F.window && i == 3) := by
    intro q i hq; have := hwz _ hq; simpa using this
  have hwp : ∀ (k : Nat) (hk : k < b0.params.length), F.win b0.params[k].1 = (F.window && k == 3) := by
    intro k hk
    apply hwin
    have := List.getElem_mem (l := b0.params.zipIdx) (by simpa using hk)
    simpa [List.getElem_zipIdx] using this
  -- the prologue's instructions
  let i3 : PtxIr.Instr := if F.window then .mov .u32 (winReg p3.1) (.sym "smem")
    else .cvta false .shared .u64 (regOf p3.2 p3.1) (.sym "smem")
  have hI : prologueInstrs F k0 b0 =
      [.mov .u32 (regOf p0.2 p0.1) (.reg (.special "ctaid.x")), .mov .u32 (regOf p1.2 p1.1) (.reg (.special "tid.x")),
        .mov .u32 (regOf p2.2 p2.1) (.reg (.special "ntid.x")), i3] ++
      ps.zipIdx.map fun (p, i) => .ld .param false (bits p.2) (regOf p.2 p.1) { base := .sym s!"p{i}" } := by
    simp only [prologueInstrs, hps, i3]
    rw [← hk0]
    congr 1
    apply zipIdx_map_snd
    intro k hk
    simp [hk]
  -- the arguments' bits
  have hl4x : 4 ≤ xs.length := by rw [hlen]; exact hl4
  obtain ⟨x0, x1, x2, x3, xr, hxs⟩ : ∃ x0 x1 x2 x3 xr, xs = x0 :: x1 :: x2 :: x3 :: xr := by
    rcases h : xs with _ | ⟨x0, _ | ⟨x1, _ | ⟨x2, _ | ⟨x3, xr⟩⟩⟩⟩ <;> rw [h] at hl4x <;> simp at hl4x
    exact ⟨x0, x1, x2, x3, xr, rfl⟩
  have hx3 : x3 = addrOf .arena 0 := by rw [hxs] at h3; simpa using h3
  have hlr : xr.length = ps.length := by rw [hxs, hps] at hlen; simpa using hlen
  have hpar_lt : ∀ i, (∃ p, (p, i) ∈ ps.zipIdx) → i < ps.length := by
    rintro i ⟨p, hp⟩; have := List.mem_zipIdx_iff_getElem?.mp hp
    exact (List.getElem?_eq_some_iff.mp this).1
  have hnw : ∀ p i, (p, i) ∈ ps.zipIdx → F.win p.1 = false := by
    intro p i hp
    have hi := List.mem_zipIdx_iff_getElem?.mp hp
    obtain ⟨hlt, hpi⟩ := List.getElem?_eq_some_iff.mp hi
    have := hwp (i + 4) (by rw [hps]; simp; omega)
    simp only [hps, List.getElem_cons_succ] at this
    rw [hpi] at this; rw [this]; simp
  have hw0 : F.win p0.1 = false := by have := hwp 0 (by rw [hps]; simp); simpa [hps] using this
  have hw1 : F.win p1.1 = false := by have := hwp 1 (by rw [hps]; simp); simpa [hps] using this
  have hw2 : F.win p2.1 = false := by have := hwp 2 (by rw [hps]; simp); simpa [hps] using this
  have hw3 : F.win p3.1 = F.window := by have := hwp 3 (by rw [hps]; simp); simpa [hps] using this
  have hn0 : x0 &&& widthMask .i32 = x0 := by
    have := hnorm 0 (by rw [hps]; simp) (by rw [hxs]; simp); simpa [hps, hxs, ht0] using this
  have hn1 : x1 &&& widthMask .i32 = x1 := by
    have := hnorm 1 (by rw [hps]; simp) (by rw [hxs]; simp); simpa [hps, hxs, ht1] using this
  have hn2 : x2 &&& widthMask .i32 = x2 := by
    have := hnorm 2 (by rw [hps]; simp) (by rw [hxs]; simp); simpa [hps, hxs, ht2] using this
  have hm32 : ∀ x : UInt64, x &&& widthMask .i32 = x → PtxRun.mask 32 x = x := fun x hx => by
    rw [show (32 : Nat) = ClifTy.i32.width from rfl, mask_width .i32 (by simp), hx]
  -- the writes
  let v3 : UInt64 := if F.window then 0 else x3
  let W : List (PtxIr.Instr × PtxIr.Reg × UInt64) :=
    [(.mov .u32 (regOf p0.2 p0.1) (.reg (.special "ctaid.x")), parReg F p0, x0),
     (.mov .u32 (regOf p1.2 p1.1) (.reg (.special "tid.x")), parReg F p1, x1),
     (.mov .u32 (regOf p2.2 p2.1) (.reg (.special "ntid.x")), parReg F p2, x2), (i3, parReg F p3, v3)] ++
    ps.zipIdx.map fun (p, i) =>
      (.ld .param false (bits p.2) (regOf p.2 p.1) { base := .sym s!"p{i}" }, parReg F p, xr.getD i 0)
  have hWI : W.map (·.1) = prologueInstrs F k0 b0 := by
    rw [hI]; simp [W]
  have hex : ∀ e ∈ W, ∀ ts' : PtxRun.TS, (PtxRun.exec c e.1).run ts' =
      .ok (.step, { ts' with regs := ts'.regs.insert e.2.1 e.2.2 }) := by
    intro e he ts'
    simp only [W, List.mem_append, List.mem_cons, List.mem_map, List.not_mem_nil, or_false] at he
    rcases he with (rfl | rfl | rfl | rfl) | ⟨⟨p, i⟩, hp, rfl⟩
    · rw [exec_mov_special, hsp 0 "ctaid.x" rfl, show xs.getD 0 0 = x0 by simp [hxs],
        show PtxIr.Ty.width .u32 = 32 from rfl, hm32 x0 hn0]
      simp [parReg, hw0]
    · rw [exec_mov_special, hsp 1 "tid.x" rfl, show xs.getD 1 0 = x1 by simp [hxs],
        show PtxIr.Ty.width .u32 = 32 from rfl, hm32 x1 hn1]
      simp [parReg, hw1]
    · rw [exec_mov_special, hsp 2 "ntid.x" rfl, show xs.getD 2 0 = x2 by simp [hxs],
        show PtxIr.Ty.width .u32 = 32 from rfl, hm32 x2 hn2]
      simp [parReg, hw2]
    · by_cases hwd : F.window = true
      · simp only [i3, v3, hwd, if_true]
        rw [exec_mov_sym c ts' .u32 _ "smem" 0 hsm]
        simp [parReg, hw3, hwd]; rfl
      · simp only [i3, v3, hwd, Bool.false_eq_true, if_false]
        rw [exec_cvta_sym c ts' _ "smem" 0 hsm]
        simp [parReg, hw3, hwd, hx3]; rfl
    · have hi := hpar_lt i ⟨p, hp⟩
      rw [exec_ld_param c ts' false _ _ _ (xr.getD i 0)]
      · simp [parReg, hnw p i hp]
      · rw [hpl i (by rw [← hk0]; simpa using hi)]; simp [hxs]
  have hrun := steps_writes c W { regs := {}, w } hex
  rw [hWI] at hrun
  refine ⟨_, hrun, ?_⟩
  -- each parameter's register holds its value
  let regId : PtxIr.Reg → Nat := fun r => match r with | .val _ n => n | _ => 0
  have hrid : ∀ q : Val × ClifTy, regId (parReg F q) = q.1.id := by
    intro q; unfold parReg; split <;> rfl
  have hzm : ∀ {β : Type} (f : Val × ClifTy → β), (ps.zipIdx.map fun x => f x.1) = ps.map f := by
    intro β f
    apply List.ext_getElem (by simp)
    intro i h1 h2; simp [List.getElem_zipIdx]
  have hkeys : ((W.map (·.2)).map (·.1)).Nodup := by
    apply nodup_of_map regId
    have hnd := (paramsOk_block F hpok b0 (by rw [hb]; exact List.mem_cons_self)).1
    rw [hps] at hnd
    have : ((W.map (·.2)).map (·.1)).map regId = (p0 :: p1 :: p2 :: p3 :: ps).map (·.1.id) := by
      simp only [W, List.map_append, List.map_cons, List.map_nil, List.map_map, Function.comp_def, hrid]
      rw [hzm (fun q => q.1.id)]
      simp
    rw [this]; exact hnd
  let valOf : (k : Nat) → k < b0.params.length → UInt64 := fun k hk =>
    if F.win b0.params[k].1 then 0 else xs[k]'(by rw [hlen]; exact hk)
  have hmem : ∀ k (hk : k < b0.params.length), (parReg F b0.params[k], valOf k hk) ∈ W.map (·.2) := by
    intro k hk
    simp only [valOf, W, List.map_append, List.map_cons, List.mem_append, List.mem_cons, List.map_map,
      List.mem_map]
    match k, hk with
    | 0, _ => left; left; simp [hps, hw0, hxs]
    | 1, _ => left; right; left; simp [hps, hw1, hxs]
    | 2, _ => left; right; right; left; simp [hps, hw2, hxs]
    | 3, _ =>
        left; right; right; right; left
        by_cases hwd : F.window = true <;> simp [hps, hw3, hwd, hxs, v3, hx3]
    | i + 4, hk =>
        right
        have hi : i < ps.length := by rw [hps] at hk; simpa using hk
        refine ⟨(ps[i], i), List.mem_zipIdx_iff_getElem?.mpr (by simp [hi]), ?_⟩
        simp [hps, hxs, hnw ps[i] i (List.mem_zipIdx_iff_getElem?.mpr (by simp [hi])),
          List.getElem?_eq_getElem (by omega : i < xr.length)]
  have hget : ∀ k (hk : k < b0.params.length),
      (W.foldl (fun m e => m.insert e.2.1 e.2.2) ({ regs := {}, w } : PtxRun.TS).regs).get? (parReg F b0.params[k]) =
        some (valOf k hk) := by
    intro k hk
    rw [show W.foldl (fun m e => m.insert e.2.1 e.2.2) ({ regs := {}, w } : PtxRun.TS).regs =
        (W.map (·.2)).foldl (fun m e => m.insert e.1 e.2) {} by simp [List.foldl_map]]
    exact foldl_insert_get _ _ _ _ hkeys (hmem k hk)
  -- the relation
  have hnd := (paramsOk_block F hpok b0 (by rw [hb]; exact List.mem_cons_self))
  have hxl : ∀ k (hk : k < b0.params.length), k < xs.length := fun k hk => by rw [hlen]; exact hk
  have hbind : ∀ k (hk : k < b0.params.length),
      Blocks.getV (bindParams #[] b0.params args) b0.params[k].1 = some (.sc b0.params[k].2 (xs[k]'(hxl k hk))) := by
    intro k hk
    have hal : k < args.length := by rw [hargs]; simp; omega
    rw [getV_bind_param b0.params args #[] k hk hal hnd.1]
    congr 1; subst hargs; simp
  have hn8 : ∀ k (hk : k < b0.params.length), b0.params[k].2 ≠ .i8 := by
    intro k hk h
    have hmap' : b0.params.map (·.2) = idParams ++ k0 := by
      rw [hps]; simp [idParams, ht0, ht1, ht2, ht3, hk0]
    have := congrArg (fun l => l[k]?) hmap'
    simp only [List.getElem?_map, List.getElem?_eq_getElem hk, Option.map_some, h] at this
    match k, hk with
    | 0, _ => simp [idParams] at this
    | 1, _ => simp [idParams] at this
    | 2, _ => simp [idParams] at this
    | 3, _ => simp [idParams] at this
    | i + 4, _ =>
        simp [idParams] at this
        have : ClifTy.i8 ∈ k0 := List.mem_of_getElem? this.symm
        simp [List.contains_eq_mem, this] at hk8
  have hidx : ∀ v : Val, (∃ p ∈ b0.params, p.1.id = v.id) → ∃ k, ∃ hk : k < b0.params.length, b0.params[k].1 = v := by
    rintro v ⟨p, hp, hid⟩
    obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp
    exact ⟨k, hk, by cases v; cases h : b0.params[k].1; simp_all⟩
  exact {
    pres := fun v hv => by
      obtain ⟨k, hk, rfl⟩ := hidx v hv
      exact ⟨_, _, hbind k hk, (hnd.2 _ (List.getElem_mem hk)).2.2.1⟩
    normd := fun v t b hv hgv => by
      obtain ⟨k, hk, rfl⟩ := hidx v hv
      rw [hbind k hk] at hgv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hgv)
      exact hnorm k hk _
    flag := fun v b hv hgv => by
      obtain ⟨k, hk, rfl⟩ := hidx v hv
      rw [hbind k hk] at hgv; obtain ⟨h8, -⟩ := V.sc.inj (Option.some.inj hgv)
      exact absurd h8 (hn8 k hk)
    plain := fun v t b hv hgv ht hl hw => by
      obtain ⟨k, hk, rfl⟩ := hidx v hv
      rw [hbind k hk] at hgv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hgv)
      have hwf : F.win b0.params[k].1 = false := by
        cases h : F.win b0.params[k].1
        · rfl
        · exact absurd ((win_iff F _).mp h) hw
      have := hget k hk
      simp only [parReg, hwf, Bool.false_eq_true, if_false, valOf] at this
      rw [this, enc, if_neg (hn8 k hk), hnorm k hk _]
    lit := fun v t b l hv _ _ hl => by
      obtain ⟨k, hk, rfl⟩ := hidx v hv
      rw [(hnd.2 _ (List.getElem_mem hk)).1] at hl; cases hl
    win := fun v t b hv hgv hw => by
      obtain ⟨k, hk, rfl⟩ := hidx v hv
      rw [hbind k hk] at hgv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hgv)
      have hwt : F.win b0.params[k].1 = true := (win_iff F _).mpr hw
      have := hget k hk
      simp only [parReg, hwt, if_true, valOf] at this
      refine ⟨0, this, by decide, ?_⟩
      have hk3 : k = 3 ∧ F.window = true := by
        have := hwp k hk; rw [hwt] at this; simp at this; exact ⟨this.2, this.1⟩
      obtain ⟨rfl, -⟩ := hk3
      have : xs[3]'(hxl 3 hk) = addrOf .arena 0 := by
        have := h3; rw [List.getElem?_eq_getElem (hxl 3 hk)] at this; exact Option.some.inj this
      rw [this]; simp [WinOff, addrOf]
    neg := fun v b hv _ hng => by
      obtain ⟨k, hk, rfl⟩ := hidx v hv
      rw [(hnd.2 _ (List.getElem_mem hk)).2.1] at hng; cases hng
    bound := fun v t b hv hgv => by
      obtain ⟨k, hk, rfl⟩ := hidx v hv
      rw [hbind k hk] at hgv; obtain ⟨rfl, rfl⟩ := V.sc.inj (Option.some.inj hgv)
      have he := paramsOk_entry F hpok b0 _ hb k hk
      by_cases hk3 : k = 3
      · subst hk3
        obtain ⟨hs, hm⟩ := he.1 rfl
        have hx3 : xs[3]'(hxl 3 hk) = addrOf .arena 0 := by
          have := h3; rw [List.getElem?_eq_getElem (hxl 3 hk)] at this; exact Option.some.inj this
        unfold Bounded Rel offBits; rw [if_pos hs, hm, hx3]
        exact ⟨by simp [addrOf], fun _ => by decide⟩
      · exact bounded_unb (he.2 hk3) _ }

/-! ## A thread -/

/-- **What a kernel's entry checked**: its body compiled, the facts it was
    lowered with passing every check, no lowering refused, labels written once,
    and the entry assembled from those facts. -/
theorem entry_facts (k : Kernel) (e : PtxIr.Entry) (he : k.entry = .ok e) :
    ∃ f, compileChecked 0 k.code .nil (idParams ++ k.params) = .ok f ∧
      (f.blocks.map (·.ref.id)).Nodup ∧ definedOk f (definedIn f) = true ∧
      edgesTyped (k.factsOf f) = true ∧ winOk (k.factsOf f) = true ∧ tysOk (k.factsOf f) = true ∧
      takesAll (k.factsOf f) = true ∧ entryOk (k.factsOf f) k.params = true ∧
      paramsOk (k.factsOf f) = true ∧ condsOk (k.factsOf f) = true ∧ defsOk (k.factsOf f) = true ∧
      (∀ o ∈ Ptx.body (k.factsOf f) k.params, o.err = none) ∧
      (labelsIn ((Ptx.body (k.factsOf f) k.params).flatMap (·.lines.toList))).Nodup ∧
      (k.params.zipIdx.map fun (_, i) => s!"p{i}").Nodup ∧
      e = k.assemble (k.factsOf f) ∧ k.smem < 2 ^ 32 := by
  unfold Kernel.entry at he
  split at he
  · cases he
  rename_i f hf
  split at he
  · cases he
  rename_i hr
  cases he
  have hr' := hr
  simp only [Kernel.refusal, Option.orElse_eq_orElse, Option.orElse_eq_none_iff, Option.map_eq_none_iff] at hr'
  obtain ⟨⟨hc, herr⟩, hlab⟩ := hr'
  have hall : ∀ x ∈ k.checks f, x.1 = true := by
    intro x hx
    have := List.find?_eq_none.mp hc x hx
    simpa using this
  have c0 := hall (decide (f.blocks.map (·.ref.id)).Nodup, "two blocks have one name") (by simp [Kernel.checks])
  have c1 := hall (definedOk f (definedIn f), "a value is read where it may not be defined") (by simp [Kernel.checks])
  have c2 := hall (edgesTyped (k.factsOf f), "an edge passes a value of another type") (by simp [Kernel.checks])
  have c3 := hall (winOk (k.factsOf f), "a shared address in the window is read as a value") (by simp [Kernel.checks])
  have c4 := hall (tysOk (k.factsOf f), "a value is not a scalar") (by simp [Kernel.checks])
  have c5 := hall (takesAll (k.factsOf f), "an instruction the compiler does not take") (by simp [Kernel.checks])
  have c6 := hall (entryOk (k.factsOf f) k.params, "the entry block is not the thread's") (by simp [Kernel.checks])
  have c7 := hall (paramsOk (k.factsOf f), "a block parameter is not plain") (by simp [Kernel.checks])
  have c8 := hall (condsOk (k.factsOf f), "a branch tests what is not a flag") (by simp [Kernel.checks])
  have c9 := hall (defsOk (k.factsOf f), "a value is not as the facts say") (by simp [Kernel.checks])
  have c10 := hall (decide (k.params.zipIdx.map fun (_, i) => s!"p{i}").Nodup, "two parameters have one name")
    (by simp [Kernel.checks])
  have c11 := hall (decide (k.smem < 2 ^ 32), "more shared memory than a GPU addresses") (by simp [Kernel.checks])
  refine ⟨f, hf, by simpa using c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, fun o ho => ?_, ?_, by simpa using c10, rfl,
    by simpa using c11⟩
  · exact List.findSome?_eq_none_iff.mp herr o ho
  · by_cases hn : (labelsIn ((Ptx.body (k.factsOf f) k.params).flatMap (·.lines.toList))).Nodup
    · exact hn
    · simp [hn] at hlab

end AlgorithmLib.Kernel.PtxSound
