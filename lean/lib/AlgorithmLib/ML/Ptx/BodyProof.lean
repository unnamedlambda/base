module
public import AlgorithmLib.ML.Ptx.Body
meta import AlgorithmLib.ML.Ptx.Body
public import AlgorithmLib.Proof.LibSpecFloat
meta import AlgorithmLib.Proof.LibSpecFloat
public import AlgorithmLib.Proof.EndsRules
meta import AlgorithmLib.Proof.EndsRules
public import AlgorithmLib.Kernel.Host
meta import AlgorithmLib.Kernel.Host
public import AlgorithmLib.Kernel.Warp
meta import AlgorithmLib.Kernel.Warp
public import AlgorithmLib.Proof.LibStep
meta import AlgorithmLib.Proof.LibStep
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.BodyProof` — what a warp statement's body computes

`wpE_fexp`: the code `ML.Body` writes for an expression ends, with the
expression's value at the carrier `F32` (`evalL`, the lane's value) in a new
slot, memory as it was and no collective made, for every expression. At
`Float32` that value is the warp machine's (`evalL_float32`). The hardware's
`2^x` is a thread's locals' answer (`wpE_ex2`), so the statement holds for
every `Hw`.

`wpE_stmt`: the code it writes for a warp statement ends as one lane's run of
the statement (`runL`) does, wherever that run finishes, whatever words the
lane's shuffles are answered: with the lane's registers at `F32`, its memory
and the collectives it makes. Ending, it meets every triple too
(`Ends.triple`). An index or a loop count read from memory reads the words
`rd` reads, which memory holds (`RdOk`) and the run leaves alone
(`runL_rdOk`); `runL` is none where an access falls outside memory, an index
outside 32 bits, or a store outside the buffers or on a word `rd` reads. A
loop carries the registers its body writes and ends by its count
(`wpE_bodyLoop`). The index and address code is operations alone, so it ends
where its weakest precondition holds (`wpE_idx`, `wpE_at4`, from `wp_idx`,
`wp_at4`). Shared memory and `cvtIF` are not yet runs.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel

namespace AlgorithmLib.ML.BodyProof

/-- One lane's value of an expression at a carrier: registers `reg`, a
    literal as the carrier holds it (`NumLit`). -/
def evalL {R : Type} [NumOps R] [NumLit R] (reg : Nat → R) : WFExp → R
  | .reg n => reg n
  | .lit v => NumLit.lit v
  | .add a b => NumOps.add (evalL reg a) (evalL reg b)
  | .mul a b => NumOps.mul (evalL reg a) (evalL reg b)
  | .neg a => NumOps.neg (evalL reg a)
  | .inv a => NumOps.inv (evalL reg a)
  | .exp a => NumOps.exp (evalL reg a)
  | .ex2 a => NumOps.ex2 (evalL reg a)
  | .rsqrt a => NumOps.rsqrt (evalL reg a)
  | .maxW a b => NumOps.max (evalL reg a) (evalL reg b)
  | .geF a b => NumOps.ifGe (evalL reg a) (evalL reg b) (NumLit.lit 1.0) (NumLit.lit 0.0)

/-- At `Float32` it is the warp machine's. -/
theorem evalL_float32 (st : WSt) (l : Lane) (e : WFExp) :
    evalL (fun n => st.regs n l) e = e.eval st l := by
  induction e <;> simp [evalL, WFExp.eval, *]

section
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- The hardware's `2^x`, as a thread's locals answer it. -/
theorem wpE_ex2 {hw : Hw} {a : Slot .f32} {x : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (hex : ∀ x w, cfg.locals Intr.ex2Approx.slot [.sc .f32 x] w =
      .ok (some (.sc .f32 (hw.approx .ex2Approx x)), w))
    (ha : Γ[a]? = some (.sc .f32 x))
    (h : Q Γ.size (Γ.push (.sc .f32 (hw.approx .ex2Approx x))) (obsCall w (.intr .ex2Approx) [.sc .f32 x])) :
    wpE cfg (callLocal Body.ex2 %[a] : Prog Slot Lvl (Slot .f32)) Q Γ w :=
  wpE_callLocal (vs := [.sc .f32 x]) (v := .sc .f32 (hw.approx .ex2Approx x))
    (w' := obsCall w (.intr .ex2Approx) [.sc .f32 x]) rfl (by simp [Vals.slots, ha])
    (by simp [callOf, Body.ex2, hex, Except.toOption]) (wpE_ret h)

theorem fconst_mask (v : Float32) : ofF32 v &&& widthMask .f32 = ofF32 v := and_mask32 (ofF32_lt v)
theorem oneBits_mask : oneBits &&& widthMask .f32 = oneBits := by decide
theorem log2e_mask : log2eBits &&& widthMask .f32 = log2eBits := and_mask32 (ofF32_lt _)

theorem max_bits (hw : Hw) (A B : F32) : (@NumOps.max F32 (F32.ops hw) A B).bits =
    if fcmpBits .ge .f32 A.bits B.bits then A.bits else B.bits := by
  simp only [NumOps.max, NumOps.ifGe, F32.ops, fcmpBits, Bool.or_comm]
  split <;> simp_all <;> split <;> rfl

theorem ifGe_bits (hw : Hw) (A B X Y : F32) : (@NumOps.ifGe F32 (F32.ops hw) A B X Y).bits =
    if fcmpBits .ge .f32 A.bits B.bits then X.bits else Y.bits := by
  simp only [NumOps.ifGe, F32.ops, fcmpBits, Bool.or_comm]
  split <;> simp_all <;> split <;> rfl

/-- Memory that is a thread's, in a world whose memory has the hardware's NaN
    rule, leaves float results as the hardware does. -/
theorem fl_of (hw : Hw) {w' w : World} (h : w'.mem = w.mem) (hn : w.mem.nan = hw.nan) :
    w'.mem.fl = nanFix hw.nan := by
  show nanFix w'.mem.nan = _; rw [h, hn]

/-- **A body's expression computes its value at the carrier `F32`**: its code
    ends with that value's word in a slot of an environment that keeps the one
    it started from, memory as it was. -/
theorem wpE_fexp (hw : Hw) (c : Body.Ctx Slot) (r : Body.Regs Slot) (regv : Nat → F32)
    (hex : ∀ x w, cfg.locals Intr.ex2Approx.slot [.sc .f32 x] w =
      .ok (some (.sc .f32 (hw.approx .ex2Approx x)), w)) :
    ∀ (e : WFExp) (Γ : Env) (w : World) (Q : Slot .f32 → Env → World → Prop) (start : Nat),
      w.mem.nan = hw.nan → start ≤ w.obs.length →
      (∀ n, Γ[Body.get c r n]? = some (.sc .f32 (regv n).bits)) →
      (∀ v Γ' w', Extends Γ Γ' → w'.mem = w.mem → collCalls w' start = collCalls w start →
        start ≤ w'.obs.length →
        Γ'[v]? = some (.sc .f32 (@evalL F32 (F32.ops hw) _ regv e).bits) → Q v Γ' w') →
      wpE cfg (Body.fexp c r e : Prog Slot Lvl (Slot .f32)) Q Γ w := by
  intro e
  induction e with
  | reg n =>
      intro Γ w Q st hn hs hr hQ
      rw [Body.fexp]
      exact wpE_pure (hQ _ _ _ (Extends.refl _) rfl rfl hs (hr n))
  | lit v =>
      intro Γ w Q st hn hs _ hQ
      rw [Body.fexp]
      apply wpE_fconst32
      rw [fconst_mask]
      exact hQ _ _ _ (Extends.push _ _) rfl rfl hs get_push_size
  | add a b iha ihb =>
      intro Γ w Q st hn hs hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ st hn hs hr fun ra Γa wa hEa hwa hca hsa hra => ?_)
      refine wpE_bind (ihb Γa wa _ st (by rw [hwa]; exact hn) hsa (fun n => hEa.get (hr n)) fun rb Γb wb hEb hwb hcb hsb hrb => ?_)
      refine wpE_fadd32 (hEb.get hra) hrb ?_
      rw [fl_of hw (hwb.trans hwa) hn]
      exact (hQ _ _ _ ((hEa.trans hEb).push_of _) (hwb.trans hwa) (hcb.trans hca) hsb get_push_size)
  | mul a b iha ihb =>
      intro Γ w Q st hn hs hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ st hn hs hr fun ra Γa wa hEa hwa hca hsa hra => ?_)
      refine wpE_bind (ihb Γa wa _ st (by rw [hwa]; exact hn) hsa (fun n => hEa.get (hr n)) fun rb Γb wb hEb hwb hcb hsb hrb => ?_)
      refine wpE_fmul32 (hEb.get hra) hrb ?_
      rw [fl_of hw (hwb.trans hwa) hn]
      exact (hQ _ _ _ ((hEa.trans hEb).push_of _) (hwb.trans hwa) (hcb.trans hca) hsb get_push_size)
  | neg a iha =>
      intro Γ w Q st hn hs hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ st hn hs hr fun ra Γa wa hEa hwa hca hsa hra => ?_)
      exact wpE_fneg32 hra (hQ _ _ _ (hEa.push_of _) hwa hca hsa get_push_size)
  | inv a iha =>
      intro Γ w Q st hn hs hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ st hn hs hr fun ra Γa wa hEa hwa hca hsa hra => ?_)
      refine wpE_bind (wpE_fconst32 ?_)
      rw [oneBits_mask]
      refine wpE_fdiv32 get_push_size (get_push hra) ?_
      rw [fl_of hw hwa hn]
      exact (hQ _ _ _ ((hEa.push_of _).push_of _) hwa hca hsa get_push_size)
  | exp a iha =>
      intro Γ w Q st hn hs hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ st hn hs hr fun ra Γa wa hEa hwa hca hsa hra => ?_)
      refine wpE_bind (wpE_fconst32 ?_)
      rw [log2e_mask]
      refine wpE_bind (wpE_fmul32 (get_push hra) get_push_size ?_)
      rw [fl_of hw hwa hn]
      exact wpE_ex2 hex get_push_size
        (hQ _ _ _ (((hEa.push_of _).push_of _).push_of _) hwa
          (by rw [collCalls_call hsa]; simp [Intr.isColl, hca]) (by simp [obsCall]; omega) get_push_size)
  | ex2 a iha =>
      intro Γ w Q st hn hs hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ st hn hs hr fun ra Γa wa hEa hwa hca hsa hra => ?_)
      exact wpE_ex2 hex hra (hQ _ _ _ (hEa.push_of _) hwa
        (by rw [collCalls_call hsa]; simp [Intr.isColl, hca]) (by simp [obsCall]; omega) get_push_size)
  | rsqrt a iha =>
      intro Γ w Q st hn hs hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ st hn hs hr fun ra Γa wa hEa hwa hca hsa hra => ?_)
      refine wpE_bind (wpE_sqrt32 hra ?_)
      rw [fl_of hw hwa hn]
      refine wpE_bind (wpE_fconst32 ?_)
      rw [oneBits_mask]
      refine wpE_fdiv32 get_push_size (get_push get_push_size) ?_
      rw [fl_of hw hwa hn]
      exact (hQ _ _ _ (((hEa.push_of _).push_of _).push_of _) hwa hca hsa get_push_size)
  | maxW a b iha ihb =>
      intro Γ w Q st hn hs hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ st hn hs hr fun ra Γa wa hEa hwa hca hsa hra => ?_)
      refine wpE_bind (ihb Γa wa _ st (by rw [hwa]; exact hn) hsa (fun n => hEa.get (hr n)) fun rb Γb wb hEb hwb hcb hsb hrb => ?_)
      refine wpE_bind (wpE_fcmp32 (hEb.get hra) hrb ?_)
      refine wpE_selectF32 get_push_size (get_push (hEb.get hra)) (get_push hrb) ?_
      exact hQ _ _ _ (((hEa.trans hEb).push_of _).push_of _) (hwb.trans hwa) (hcb.trans hca) hsb
        (by simp only [evalL]; rw [max_bits]; exact get_push_size)
  | geF a b iha ihb =>
      intro Γ w Q st hn hs hr hQ
      simp only [Body.fexp]
      refine wpE_bind (iha Γ w _ st hn hs hr fun ra Γa wa hEa hwa hca hsa hra => ?_)
      refine wpE_bind (ihb Γa wa _ st (by rw [hwa]; exact hn) hsa (fun n => hEa.get (hr n)) fun rb Γb wb hEb hwb hcb hsb hrb => ?_)
      refine wpE_bind (wpE_fcmp32 (hEb.get hra) hrb ?_)
      refine wpE_bind (wpE_fconst32 ?_)
      refine wpE_bind (wpE_fconst32 ?_)
      rw [fconst_mask, fconst_mask]
      refine wpE_selectF32 (get_push (get_push get_push_size)) (get_push get_push_size) get_push_size ?_
      exact hQ _ _ _ ((((hEa.trans hEb).push_of _).push_of _).push_of _ |>.push_of _) (hwb.trans hwa)
        (hcb.trans hca) hsb (by simp only [evalL]; rw [ifGe_bits]; exact get_push_size)

end

/-! ## A warp statement's code, as one lane's run -/

/-- One lane's state: its registers, its memory, and the collectives it made. -/
structure LSt where
  reg : Nat → F32
  mem : Mem
  calls : List (Intr × List V)

def upd (f : Nat → F32) (n : Nat) (v : F32) : Nat → F32 := fun k => if k = n then v else f k

/-- An index's value, a word of memory read by `rd` (buffer, element), none
    where some part of it does not fit 32 bits. -/
def idxL (rd : Buf → Nat → Option Nat) (cta i l : Nat) : IdxE → Option Nat
  | .laneId => if l < 2 ^ 32 then some l else none
  | .loopI => if i < 2 ^ 32 then some i else none
  | .ctaId => if cta < 2 ^ 32 then some cta else none
  | .lit n => if n < 2 ^ 31 then some n else none
  | .ldIdx b off => do rd b (← idxL rd cta i l off)
  | .add a b => do
      let x ← idxL rd cta i l a
      let y ← idxL rd cta i l b
      if x + y < 2 ^ 32 then some (x + y) else none
  | .mul a b => do
      let x ← idxL rd cta i l a
      let y ← idxL rd cta i l b
      if x * y < 2 ^ 32 then some (x * y) else none
  | .ireg _ => none

/-- A statement's buffer `b`, among `nbuf`: its base. -/
def bufB (nbuf b : Nat) : UInt64 := if h : b < nbuf ∧ b < 268435449 then addrOf (.buf ⟨b, h.2⟩) 0 else 0

/-- Where a buffer's word `j` is. -/
def bufW (nbuf b j : Nat) : UInt64 := bufB nbuf b + off (UInt32.ofNat j)

/-- Word `j` of buffer `b` in `m`, as a number. -/
def rdM (nbuf : Nat) (m : Mem) (b : Buf) (j : Nat) : Option Nat := (m.load (bufW nbuf b j) 4).map UInt64.toNat

/-- **`m` holds what `rd` reads**: every word `rd` answers is a word of one
    of the `nbuf` buffers, and `m` holds it. -/
def RdOk (nbuf : Nat) (m : Mem) (rd : Buf → Nat → Option Nat) : Prop :=
  ∀ b j n, rd b j = some n → (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) ∧ rdM nbuf m b j = some n

section
variable (hw : Hw) (nbuf cta l : Nat) (o : Nat → UInt32) (rd : Buf → Nat → Option Nat)

/-- **One lane's run of a warp statement**, its `k`-th shuffle answered
    `o k` and the words its indexes and loop counts read `rd`'s: none where
    an access falls outside memory, an index outside 32 bits, or a store
    outside the buffers or on a word `rd` reads. -/
def runL : Nat → EWStmt → LSt → Option LSt
  | _, .skip, st => some st
  | i, .seq a b, st => do runL i b (← runL i a st)
  | _, .setR n e, st => some { st with reg := upd st.reg n (@evalL F32 (F32.ops hw) _ st.reg e) }
  | _, .shflXor d s m, st =>
      if m < 2 ^ 31 then
        some { st with reg := upd st.reg d ⟨s32 (o st.calls.length)⟩,
                       calls := st.calls ++ [(.shflBfly, [.sc .i32 (st.reg s).bits, .sc .i32 (s32 (UInt32.ofNat m))])] }
      else none
  | i, .loadIdx d b ix, st => do
      let j ← idxL rd cta i l ix
      let v ← st.mem.load (bufW nbuf b j) 4
      some { st with reg := upd st.reg d ⟨v⟩ }
  | i, .storeLane b ix n, st => do
      let j ← idxL rd cta i l ix
      if rd b j = none ∧ b < nbuf ∧ b < 268435449 then
        let m ← st.mem.store (bufW nbuf b j) 4 (st.reg n).bits
        some { st with mem := m }
      else none
  | i, .storeLane0 b ix n, st =>
      if l = 0 then do
        let j ← idxL rd cta i l ix
        if rd b j = none ∧ b < nbuf ∧ b < 268435449 then
          let m ← st.mem.store (bufW nbuf b j) 4 (st.reg n).bits
          some { st with mem := m }
        else none
      else some st
  | i, .loadV4 d0 d1 d2 d3 b ix, st => do
      let j ← idxL rd cta i l ix
      let v0 ← st.mem.load (bufW nbuf b j) 4
      let v1 ← st.mem.load (bufW nbuf b j + c64 4) 4
      let v2 ← st.mem.load (bufW nbuf b j + c64 8) 4
      let v3 ← st.mem.load (bufW nbuf b j + c64 12) 4
      some { st with reg := upd (upd (upd (upd st.reg d0 ⟨v0⟩) d1 ⟨v1⟩) d2 ⟨v2⟩) d3 ⟨v3⟩ }
  | _, .forN n body, st =>
      if n < 2 ^ 31 then (List.range n).foldlM (fun st j => runL j body st) st else none
  | _, .forM b a body, st =>
      if a < 2 ^ 31 then do
        let n ← rd b a
        if n < 2 ^ 31 then (List.range n).foldlM (fun st j => runL j body st) st else none
      else none
  | _, _, _ => none

end

theorem foldlM_inv {α β : Type} (f : β → α → Option β) (P : β → Prop)
    (hf : ∀ b a b', f b a = some b' → P b → P b') : ∀ (l : List α) (b b' : β), l.foldlM f b = some b' → P b → P b'
  | [], b, b', h, hb => by simp only [List.foldlM_nil] at h; cases h; exact hb
  | a :: as, b, b', h, hb => by
    simp only [List.foldlM_cons, bind, Option.bind] at h
    split at h
    · cases h
    rename_i b1 h1
    exact foldlM_inv f P hf as b1 b' h (hf b a b1 h1 hb)

section Runs
variable {hw : Hw} {nbuf cta l : Nat} {o : Nat → UInt32} {rd : Buf → Nat → Option Nat}

theorem runL_store {i : Nat} {b : Buf} {ix : IdxE} {n : Nat} {st st' : LSt}
    (h : runL hw nbuf cta l o rd i (.storeLane b ix n) st = some st') :
    ∃ j m', idxL rd cta i l ix = some j ∧ rd b j = none ∧ b < nbuf ∧ b < 268435449 ∧
      st.mem.store (bufW nbuf b j) 4 (st.reg n).bits = some m' ∧ st' = { st with mem := m' } := by
  simp only [runL, bind, Option.bind] at h
  split at h; · cases h
  rename_i j hj
  dsimp only at h
  split at h
  · rename_i hc
    split at h; · cases h
    rename_i m' hm'
    cases h
    exact ⟨j, m', hj, hc.1, hc.2.1, hc.2.2, hm', rfl⟩
  · cases h

theorem runL_store0 {i : Nat} {b : Buf} {ix : IdxE} {n : Nat} {st st' : LSt}
    (h : runL hw nbuf cta l o rd i (.storeLane0 b ix n) st = some st') :
    (l = 0 ∧ ∃ j m', idxL rd cta i l ix = some j ∧ rd b j = none ∧ b < nbuf ∧ b < 268435449 ∧
      st.mem.store (bufW nbuf b j) 4 (st.reg n).bits = some m' ∧ st' = { st with mem := m' }) ∨
    (l ≠ 0 ∧ st' = st) := by
  simp only [runL] at h
  split at h
  · rename_i hl
    left
    refine ⟨hl, ?_⟩
    simp only [bind, Option.bind] at h
    split at h; · cases h
    rename_i j hj
    dsimp only at h
    split at h
    · rename_i hc
      split at h; · cases h
      rename_i m' hm'
      cases h
      exact ⟨j, m', hj, hc.1, hc.2.1, hc.2.2, hm', rfl⟩
    · cases h
  · rename_i hl
    cases h
    exact Or.inr ⟨hl, rfl⟩

theorem runL_forM {i : Nat} {b : Buf} {a : Nat} {body : EWStmt} {st st' : LSt}
    (h : runL hw nbuf cta l o rd i (.forM b a body) st = some st') :
    a < 2 ^ 31 ∧ ∃ n, rd b a = some n ∧ n < 2 ^ 31 ∧
      (List.range n).foldlM (fun st j => runL hw nbuf cta l o rd j body st) st = some st' := by
  simp only [runL] at h
  split at h
  · rename_i ha
    simp only [bind, Option.bind] at h
    split at h; · cases h
    rename_i n hn
    dsimp only at h
    split at h
    · exact ⟨ha, n, hn, by assumption, h⟩
    · cases h
  · cases h

theorem runL_store_of {i : Nat} {b : Buf} {ix : IdxE} {n j : Nat} {st : LSt} {m' : Mem}
    (hj : idxL rd cta i l ix = some j) (hc : rd b j = none ∧ b < nbuf ∧ b < 268435449)
    (hm : st.mem.store (bufW nbuf b j) 4 (st.reg n).bits = some m') :
    runL hw nbuf cta l o rd i (.storeLane b ix n) st = some { st with mem := m' } := by
  simp only [runL, bind, Option.bind, hj, if_pos hc, hm]

theorem runL_store0_of {i : Nat} {b : Buf} {ix : IdxE} {n j : Nat} {st : LSt} {m' : Mem} (hl : l = 0)
    (hj : idxL rd cta i l ix = some j) (hc : rd b j = none ∧ b < nbuf ∧ b < 268435449)
    (hm : st.mem.store (bufW nbuf b j) 4 (st.reg n).bits = some m') :
    runL hw nbuf cta l o rd i (.storeLane0 b ix n) st = some { st with mem := m' } := by
  simp only [runL, if_pos hl, bind, Option.bind, hj, if_pos hc, hm]

theorem runL_store0_off {i : Nat} {b : Buf} {ix : IdxE} {n : Nat} {st : LSt} (hl : l ≠ 0) :
    runL hw nbuf cta l o rd i (.storeLane0 b ix n) st = some st := by
  simp only [runL, if_neg hl]

theorem runL_forM_of {i : Nat} {b : Buf} {a n : Nat} {body : EWStmt} {st st' : LSt} (ha : a < 2 ^ 31)
    (hn : rd b a = some n) (hn31 : n < 2 ^ 31)
    (hf : (List.range n).foldlM (fun st j => runL hw nbuf cta l o rd j body st) st = some st') :
    runL hw nbuf cta l o rd i (.forM b a body) st = some st' := by
  simp only [runL, if_pos ha, bind, Option.bind, hn, if_pos hn31]
  exact hf

theorem decode_bufW {nbuf b j : Nat} (hb : b < nbuf) (hb' : b < 268435449) (hj : j < 2 ^ 32) :
    decodeAddr (bufW nbuf b j) = some (.buf ⟨b, hb'⟩, 4 * j) := by
  unfold bufW bufB
  rw [dif_pos ⟨hb, hb'⟩, decode_elem, toNat_ofNat32 hj]

theorem apart_elems {nbuf b j b' j' : Nat} (hb : b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hb' : b' < nbuf ∧ b' < 268435449 ∧ j' < 2 ^ 32) (hne : ¬(b' = b ∧ j' = j)) {m m' : Mem} {v : UInt64}
    (h : m.store (bufW nbuf b j) 4 v = some m') :
    m'.load (bufW nbuf b' j') 4 = m.load (bufW nbuf b' j') 4 := by
  refine load_store_clear h (decode_bufW hb.1 hb.2.1 hb.2.2) (decode_bufW hb'.1 hb'.2.1 hb'.2.2) ?_
  by_cases hbb : b' = b
  · subst hbb
    right
    have hjj : j' ≠ j := fun e => hne ⟨rfl, e⟩
    omega
  · left; intro e; apply hbb; cases e; rfl

theorem idxL_lt {m : Mem} {i : Nat} (hrd : RdOk nbuf m rd) : ∀ (ix : IdxE) (j : Nat), idxL rd cta i l ix = some j → j < 2 ^ 32 := by
  intro ix
  induction ix with
  | laneId => intro j h; simp only [idxL] at h; split at h <;> cases h; assumption
  | loopI => intro j h; simp only [idxL] at h; split at h <;> cases h; assumption
  | ctaId => intro j h; simp only [idxL] at h; split at h <;> cases h; assumption
  | lit n => intro j h; simp only [idxL] at h; split at h <;> cases h; omega
  | ireg _ => intro j h; simp [idxL] at h
  | ldIdx b o _ =>
    intro j h
    simp only [idxL, bind, Option.bind] at h
    split at h; · cases h
    obtain ⟨_, hm⟩ := hrd b _ j h
    simp only [rdM, Option.map_eq_some_iff] at hm
    obtain ⟨v, hv, rfl⟩ := hm
    exact load4_lt hv
  | add a b _ _ =>
    intro j h
    simp only [idxL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h <;> cases h; assumption
  | mul a b _ _ =>
    intro j h
    simp only [idxL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h <;> cases h; assumption

/-- **A lane's run keeps what its stores keep**: memory `rd` reads as it says
    (`RdOk`), and any property every store of the run keeps. -/
theorem runL_inv (P : Mem → Prop)
    (hst : ∀ (m m' : Mem) (b : Buf) (j : Nat) (v : UInt64), rd b j = none → b < nbuf → b < 268435449 →
      j < 2 ^ 32 → m.store (bufW nbuf b j) 4 v = some m' → RdOk nbuf m rd → P m → P m') :
    ∀ (s : EWStmt) (i : Nat) (st st' : LSt), runL hw nbuf cta l o rd i s st = some st' →
      RdOk nbuf st.mem rd → P st.mem → RdOk nbuf st'.mem rd ∧ P st'.mem := by
  have store : ∀ (m m' : Mem) (b : Buf) (j : Nat) (v : UInt64), rd b j = none → b < nbuf → b < 268435449 →
      j < 2 ^ 32 → m.store (bufW nbuf b j) 4 v = some m' → RdOk nbuf m rd → RdOk nbuf m' rd := by
    intro m m' b j v hn hb hb' hj hs hrd b' j' n hr
    obtain ⟨hv, hm⟩ := hrd b' j' n hr
    refine ⟨hv, ?_⟩
    have hne : ¬(b' = b ∧ j' = j) := by
      rintro ⟨rfl, rfl⟩; rw [hn] at hr; cases hr
    simp only [rdM] at hm ⊢
    rw [apart_elems ⟨hb, hb', hj⟩ hv hne hs]
    exact hm
  intro s
  induction s with
  | skip => intro i st st' h hr hp; cases h; exact ⟨hr, hp⟩
  | seq a b iha ihb =>
    intro i st st' h hr hp
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    rename_i sa hsa
    obtain ⟨r1, p1⟩ := iha i st sa hsa hr hp
    exact ihb i sa st' h r1 p1
  | setR => intro i st st' h hr hp; cases h; exact ⟨hr, hp⟩
  | shflXor =>
    intro i st st' h hr hp
    simp only [runL] at h; split at h
    · cases h; exact ⟨hr, hp⟩
    · cases h
  | loadIdx =>
    intro i st st' h hr hp
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; exact ⟨hr, hp⟩
  | loadV4 =>
    intro i st st' h hr hp
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; exact ⟨hr, hp⟩
  | storeLane b ix n =>
    intro i st st' h hr hp
    obtain ⟨j, m', hj, hn, hb, hb', hs, rfl⟩ := runL_store h
    have hj' := idxL_lt hr ix j hj
    exact ⟨store _ _ b j _ hn hb hb' hj' hs hr, hst _ _ b j _ hn hb hb' hj' hs hr hp⟩
  | storeLane0 b ix n =>
    intro i st st' h hr hp
    rcases runL_store0 h with ⟨_, j, m', hj, hn, hb, hb', hs, rfl⟩ | ⟨_, rfl⟩
    · have hj' := idxL_lt hr ix j hj
      exact ⟨store _ _ b j _ hn hb hb' hj' hs hr, hst _ _ b j _ hn hb hb' hj' hs hr hp⟩
    · exact ⟨hr, hp⟩
  | forN n body ih =>
    intro i st st' h hr hp
    simp only [runL] at h
    split at h
    · exact foldlM_inv (fun st j => runL hw nbuf cta l o rd j body st)
        (fun st => RdOk nbuf st.mem rd ∧ P st.mem)
        (fun st j st' h h' => ih j st st' h h'.1 h'.2) _ _ _ h ⟨hr, hp⟩
    · cases h
  | forM b a body ih =>
    intro i st st' h hr hp
    obtain ⟨_, n, _, _, hf⟩ := runL_forM h
    exact foldlM_inv (fun st j => runL hw nbuf cta l o rd j body st)
      (fun st => RdOk nbuf st.mem rd ∧ P st.mem)
      (fun st j st' h h' => ih j st st' h h'.1 h'.2) _ _ _ hf ⟨hr, hp⟩
  | _ => intro i st st' h; simp [runL] at h

/-- **A lane's run leaves what `rd` reads alone.** -/
theorem runL_rdOk (s : EWStmt) (i : Nat) (st st' : LSt) (h : runL hw nbuf cta l o rd i s st = some st')
    (hr : RdOk nbuf st.mem rd) : RdOk nbuf st'.mem rd :=
  (runL_inv (fun _ => True) (fun _ _ _ _ _ _ _ _ _ _ _ _ => trivial) s i st st' h hr trivial).1

/-- A lane's run leaves the platform's NaN rule. -/
theorem runL_nan (s : EWStmt) (i : Nat) (st st' : LSt) (h : runL hw nbuf cta l o rd i s st = some st')
    (hr : RdOk nbuf st.mem rd) : st'.mem.nan = st.mem.nan :=
  (runL_inv (fun m => m.nan = st.mem.nan) (fun _ _ _ _ _ _ _ _ _ hs _ hp => (Mem.store_nan hs).trans hp)
    s i st st' h hr rfl).2

end Runs

section Steps
variable {cfg : Cfg} {d : Nat} {J : Post}

/-- What a statement's context holds: its lane, block and loop index as
    32-bit words, its buffers' bases, zeros. -/
structure CtxRel (c : Body.Ctx Slot) (Γ : Env) (nbuf cta l i : Nat) : Prop where
  lane : Γ[c.lane]? = some (.sc .i32 (s32 (UInt32.ofNat l)))
  cta : Γ[c.cta]? = some (.sc .i32 (s32 (UInt32.ofNat cta)))
  i : Γ[c.i]? = some (.sc .i32 (s32 (UInt32.ofNat i)))
  buf : ∀ b, Γ[Body.buf c b]? = some (.sc .i64 (bufB nbuf b))
  zf : Γ[c.zf]? = some (.sc .f32 0)

theorem CtxRel.ext {c : Body.Ctx Slot} {Γ Γ' : Env} {nbuf cta l i : Nat} (h : CtxRel c Γ nbuf cta l i)
    (hE : Extends Γ Γ') : CtxRel c Γ' nbuf cta l i :=
  ⟨hE.get h.lane, hE.get h.cta, hE.get h.i, fun b => hE.get (h.buf b), hE.get h.zf⟩

theorem wp_at4 {base : Slot .i64} {Γ : Env} {w : World} {B : UInt64} {ixs : Slot .i32} {j : Nat}
    {Q : Slot .i64 → Env → World → Prop}
    (hb : Γ[base]? = some (.sc .i64 B)) (hj : Γ[ixs]? = some (.sc .i32 (s32 (UInt32.ofNat j))))
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (B + off (UInt32.ofNat j))) → Q v Γ' w) :
    wp cfg d J (Body.at4 base ixs : Prog Slot Lvl (Slot .i64)) Q Γ w := by
  simp only [Body.at4, wp_bind]
  refine wp_uext32 hj ?_
  refine wp_ishlImm get_push_size ?_
  refine wp_iadd (get_push (get_push (get_push hb))) get_push_size ?_
  exact h _ _ ((((Extends.refl _).push_of _).push_of _).push_of _ |>.push_of _) get_push_size

theorem wp_loadI32 {a : Slot .i64} {p v : UInt64} {Γ : Env} {w : World} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 p)) (hl : w.mem.load p 4 = some v)
    (h : Q Γ.size (Γ.push (.sc .i32 v)) w) : wp cfg d J (load32 a) Q Γ w :=
  wp_val (by simp [load32, Op'.erase, evalOp, Sem.get, ha, hl, tyBytes, ClifTy.lanes, ClifTy.width] <;> op_fin) h

theorem s32_toNat {v : UInt64} (h : v.toNat < 2 ^ 32) : s32 (UInt32.ofNat v.toNat) = v := by
  apply UInt64.toNat_inj.mp
  simp [s32, Nat.mod_eq_of_lt h]

theorem wp_idx {c : Body.Ctx Slot} {nbuf cta l i : Nat} {rd : Buf → Nat → Option Nat} :
    ∀ (ix : IdxE) (Γ : Env) (w : World) (Q : Slot .i32 → Env → World → Prop) (j : Nat),
      CtxRel c Γ nbuf cta l i → RdOk nbuf w.mem rd → idxL rd cta i l ix = some j →
      (∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i32 (s32 (UInt32.ofNat j))) → Q v Γ' w) →
      wp cfg d J (Body.idx c ix : Prog Slot Lvl (Slot .i32)) Q Γ w := by
  intro ix
  induction ix with
  | laneId =>
    intro Γ w Q j hc hrd hj hQ
    simp only [idxL] at hj; split at hj <;> cases hj
    rw [Body.idx, wp_pure]; exact hQ _ _ (Extends.refl _) hc.lane
  | loopI =>
    intro Γ w Q j hc hrd hj hQ
    simp only [idxL] at hj; split at hj <;> cases hj
    rw [Body.idx, wp_pure]; exact hQ _ _ (Extends.refl _) hc.i
  | ctaId =>
    intro Γ w Q j hc hrd hj hQ
    simp only [idxL] at hj; split at hj <;> cases hj
    rw [Body.idx, wp_pure]; exact hQ _ _ (Extends.refl _) hc.cta
  | lit n =>
    intro Γ w Q j hc hrd hj hQ
    simp only [idxL] at hj; split at hj <;> cases hj
    rename_i hn
    rw [Body.idx, Body.c32]
    refine wp_iconst32u (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_
    exact hQ _ _ (Extends.push _ _) (by simpa using get_push_size)
  | add a b iha ihb =>
    intro Γ w Q j hc hrd hj hQ
    simp only [idxL, bind, Option.bind] at hj
    split at hj; · cases hj
    rename_i x hx
    dsimp only at hj
    split at hj; · cases hj
    rename_i y hy
    dsimp only at hj
    split at hj <;> cases hj
    rename_i hxy
    simp only [Body.idx, wp_bind]
    refine iha Γ w _ x hc hrd hx fun va Γa hEa hva => ?_
    refine ihb Γa w _ y (hc.ext hEa) hrd hy fun vb Γb hEb hvb => ?_
    refine wp_iadd32 (hEb.get hva) hvb ?_
    exact hQ _ _ ((hEa.trans hEb).push_of _) (by rw [UInt32.ofNat_add]; exact get_push_size)
  | mul a b iha ihb =>
    intro Γ w Q j hc hrd hj hQ
    simp only [idxL, bind, Option.bind] at hj
    split at hj; · cases hj
    rename_i x hx
    dsimp only at hj
    split at hj; · cases hj
    rename_i y hy
    dsimp only at hj
    split at hj <;> cases hj
    rename_i hxy
    simp only [Body.idx, wp_bind]
    refine iha Γ w _ x hc hrd hx fun va Γa hEa hva => ?_
    refine ihb Γa w _ y (hc.ext hEa) hrd hy fun vb Γb hEb hvb => ?_
    refine wp_imul32 (hEb.get hva) hvb ?_
    exact hQ _ _ ((hEa.trans hEb).push_of _) (by rw [UInt32.ofNat_mul]; exact get_push_size)
  | ireg n => intro Γ w Q j _ _ hj; simp [idxL] at hj
  | ldIdx b o ih =>
    intro Γ w Q j hc hrd hj hQ
    simp only [idxL, bind, Option.bind] at hj
    split at hj; · cases hj
    rename_i x hx
    have hm := (hrd b x j hj).2
    simp only [rdM, Option.map_eq_some_iff] at hm
    obtain ⟨v, hv, rfl⟩ := hm
    simp only [Body.idx, wp_bind]
    refine ih Γ w _ x hc hrd hx fun vo Γo hEo hvo => ?_
    refine wp_at4 ((hc.ext hEo).buf b) hvo fun va Γa hEa hva => ?_
    refine wp_loadI32 hva hv ?_
    exact hQ _ _ ((hEo.trans hEa).push_of _) (by rw [s32_toNat (load4_lt hv)]; exact get_push_size)

theorem at4_ops (base : Slot .i64) (ix : Slot .i32) : OpsOnly (Body.at4 base ix : Prog Slot Lvl (Slot .i64)) := by
  unfold Body.at4; ops_only

theorem idx_ops (c : Body.Ctx Slot) : ∀ ix : IdxE, OpsOnly (Body.idx c ix : Prog Slot Lvl (Slot .i32))
  | .laneId => by unfold Body.idx; exact .ret _
  | .loopI => by unfold Body.idx; exact .ret _
  | .ctaId => by unfold Body.idx; exact .ret _
  | .ireg _ => by unfold Body.idx; exact .ret _
  | .lit n => by unfold Body.idx Body.c32; ops_only
  | .add a b => by
      unfold Body.idx; exact OpsOnly.bind (idx_ops c a) fun _ => OpsOnly.bind (idx_ops c b) fun _ => by ops_only
  | .mul a b => by
      unfold Body.idx; exact OpsOnly.bind (idx_ops c a) fun _ => OpsOnly.bind (idx_ops c b) fun _ => by ops_only
  | .ldIdx b o => by
      unfold Body.idx
      exact OpsOnly.bind (idx_ops c o) fun _ => OpsOnly.bind (at4_ops _ _) fun _ => by unfold load32; ops_only

theorem wpE_at4 {base : Slot .i64} {Γ : Env} {w : World} {B : UInt64} {ixs : Slot .i32} {j : Nat}
    {Q : Slot .i64 → Env → World → Prop}
    (hb : Γ[base]? = some (.sc .i64 B)) (hj : Γ[ixs]? = some (.sc .i32 (s32 (UInt32.ofNat j))))
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (B + off (UInt32.ofNat j))) → Q v Γ' w) :
    wpE cfg (Body.at4 base ixs : Prog Slot Lvl (Slot .i64)) Q Γ w :=
  wpE_of_wp (at4_ops _ _) (wp_at4 (d := 0) (J := J0) hb hj h)

theorem wpE_loadI32 {a : Slot .i64} {p v : UInt64} {Γ : Env} {w : World} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 p)) (hl : w.mem.load p 4 = some v)
    (h : Q Γ.size (Γ.push (.sc .i32 v)) w) : wpE cfg (load32 a) Q Γ w :=
  wpE_of_wp (by unfold load32; ops_only) (wp_loadI32 (d := 0) (J := J0) ha hl h)

theorem wpE_idx {c : Body.Ctx Slot} {nbuf cta l i : Nat} {rd : Buf → Nat → Option Nat}
    (ix : IdxE) (Γ : Env) (w : World) (Q : Slot .i32 → Env → World → Prop) (j : Nat)
    (hc : CtxRel c Γ nbuf cta l i) (hrd : RdOk nbuf w.mem rd) (hj : idxL rd cta i l ix = some j)
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i32 (s32 (UInt32.ofNat j))) → Q v Γ' w) :
    wpE cfg (Body.idx c ix : Prog Slot Lvl (Slot .i32)) Q Γ w :=
  wpE_of_wp (idx_ops c ix) (wp_idx (d := 0) (J := J0) ix Γ w Q j hc hrd hj h)

/-- The registers a statement's code holds: register `n` in its slot. -/
def RegRel (c : Body.Ctx Slot) (r : Body.Regs Slot) (Γ : Env) (reg : Nat → F32) : Prop :=
  ∀ n, Γ[Body.get c r n]? = some (.sc .f32 (reg n).bits)

theorem RegRel.ext {c : Body.Ctx Slot} {r : Body.Regs Slot} {Γ Γ' : Env} {reg : Nat → F32}
    (h : RegRel c r Γ reg) (hE : Extends Γ Γ') : RegRel c r Γ' reg := fun n => hE.get (h n)

theorem get_set (c : Body.Ctx Slot) (r : Body.Regs Slot) (n : Nat) (v : Slot .f32) (k : Nat) :
    Body.get c (Body.set r n v) k = if k = n then v else Body.get c r k := by
  unfold Body.get Body.set
  by_cases h : k = n
  · subst h; simp [List.lookup]
  · simp only [if_neg h, List.lookup]
    rw [show (k == n) = false from by simpa using h]
    simp only [cond_false]
    congr 1
    induction r with
    | nil => rfl
    | cons p ps ih =>
      obtain ⟨a, x⟩ := p
      by_cases ha : a = n
      · subst ha; simp [List.filter, List.lookup, ih, show (k == a) = false from by simpa using h]
      · simp only [List.filter, show (a != n) = true from by simpa using ha, List.lookup]
        split <;> simp_all

theorem RegRel.set {c : Body.Ctx Slot} {r : Body.Regs Slot} {Γ : Env} {reg : Nat → F32}
    (h : RegRel c r Γ reg) (n : Nat) {v : Slot .f32} {x : F32} (hv : Γ[v]? = some (.sc .f32 x.bits)) :
    RegRel c (Body.set r n v) Γ (upd reg n x) := by
  intro k
  rw [get_set]
  by_cases hk : k = n
  · simp [hk, upd, hv]
  · simp [hk, upd, h k]


/-- Every register a 32-bit word. -/
def Regs32 (reg : Nat → F32) : Prop := ∀ n, (reg n).bits.toNat < 2 ^ 32

theorem evalL_32 (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) {reg : Nat → F32}
    (h : Regs32 reg) : ∀ e : WFExp, (@evalL F32 (F32.ops hw) _ reg e).bits.toNat < 2 ^ 32 := by
  intro e
  induction e with
  | reg n => exact h n
  | lit v => exact ofF32_lt v
  | add a b _ _ => exact nanFix_lt _ (ofF32_lt _)
  | mul a b _ _ => exact nanFix_lt _ (ofF32_lt _)
  | neg a ih =>
    show ((@evalL F32 (F32.ops hw) _ reg a).bits ^^^ 0x80000000).toNat < 2 ^ 32
    rw [UInt64.toNat_xor]
    exact Nat.xor_lt_two_pow ih (by decide)
  | inv a _ => exact nanFix_lt _ (ofF32_lt _)
  | exp a _ => exact hw32 _ _
  | ex2 a _ => exact hw32 _ _
  | rsqrt a _ => exact nanFix_lt _ (ofF32_lt _)
  | maxW a b iha ihb =>
    simp only [evalL]; rw [max_bits]; split <;> assumption
  | geF a b _ _ =>
    simp only [evalL]; rw [ifGe_bits]; split <;> exact ofF32_lt _

theorem Regs32.set {reg : Nat → F32} (h : Regs32 reg) (n : Nat) {x : F32} (hx : x.bits.toNat < 2 ^ 32) :
    Regs32 (upd reg n x) := by
  intro k; unfold upd; split
  · exact hx
  · exact h k

theorem runL_32 (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (nbuf cta l : Nat) (o : Nat → UInt32)
    (rd : Buf → Nat → Option Nat) :
    ∀ (s : EWStmt) (i : Nat) (st st' : LSt), runL hw nbuf cta l o rd i s st = some st' →
      Regs32 st.reg → Regs32 st'.reg := by
  intro s
  induction s with
  | skip => intro i st st' h h32; cases h; exact h32
  | seq a b iha ihb =>
    intro i st st' h h32
    simp only [runL, bind, Option.bind] at h
    split at h
    · cases h
    rename_i sa hsa
    exact ihb i sa st' h (iha i st sa hsa h32)
  | setR n e =>
    intro i st st' h h32; cases h
    exact h32.set n (evalL_32 hw hw32 h32 e)
  | shflXor d s m =>
    intro i st st' h h32
    simp only [runL] at h; split at h
    · cases h; exact h32.set d (by simp [s32]; exact UInt32.toNat_lt _)
    · cases h
  | loadIdx d b ix =>
    intro i st st' h h32
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    rename_i v hv
    cases h
    exact h32.set d (load4_lt hv)
  | storeLane b ix n =>
    intro i st st' h h32
    obtain ⟨_, _, _, _, _, _, _, rfl⟩ := runL_store h
    exact h32
  | forN n body ih =>
    intro i st st' h h32
    simp only [runL] at h
    split at h
    · exact foldlM_inv _ (fun st => Regs32 st.reg) (fun st j st' h h32 => ih j st st' h h32) _ _ _ h h32
    · cases h
  | forM b a body ih =>
    intro i st st' h h32
    obtain ⟨_, n, _, _, hf⟩ := runL_forM h
    exact foldlM_inv _ (fun st => Regs32 st.reg) (fun st j st' h h32 => ih j st st' h h32) _ _ _ hf h32
  | storeLane0 b ix n =>
    intro i st st' h h32
    rcases runL_store0 h with ⟨_, _, _, _, _, _, _, _, rfl⟩ | ⟨_, rfl⟩ <;> exact h32
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i st st' h h32
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    rename_i v0 h0
    dsimp only at h
    split at h; · cases h
    rename_i v1 h1
    dsimp only at h
    split at h; · cases h
    rename_i v2 h2
    dsimp only at h
    split at h; · cases h
    rename_i v3 h3
    cases h
    exact (((h32.set d0 (load4_lt h0)).set d1 (load4_lt h1)).set d2 (load4_lt h2)).set d3 (load4_lt h3)
  | _ => intro i st st' h; simp [runL] at h


/-- Registers set from slots, in order: where every slot holds its register's
    value and every register not set already holds its, all do. -/
theorem RegRel.back {c : Body.Ctx Slot} {Γ : Env} {reg : Nat → F32} :
    ∀ (ws : List Nat) (ss : List Nat) (r : Body.Regs Slot), ws.length = ss.length →
      (∀ (p k sl : Nat), ws[p]? = some k → ss[p]? = some sl → Γ[sl]? = some (V.sc .f32 (reg k).bits)) →
      (∀ k, k ∉ ws → Γ[Body.get c r k]? = some (.sc .f32 (reg k).bits)) →
      RegRel c ((ws.zip ss).foldl (fun r p => Body.set r p.1 p.2) r) Γ reg
  | [], [], r, _, _, hk => fun k => hk k (by simp)
  | w0 :: ws, s0 :: ss, r, hl, hp, hk => by
    rw [List.zip_cons_cons, List.foldl_cons]
    refine RegRel.back ws ss _ (by simpa using hl) (fun p k sl h1 h2 => hp (p + 1) k sl (by simpa using h1)
      (by simpa using h2)) ?_
    intro k hk'
    rw [get_set]
    by_cases h : k = w0
    · subst h; simpa using hp 0 k s0 rfl rfl
    · simp only [if_neg h]; exact hk k (by simp [h, hk'])
  | [], _ :: _, _, hl, _, _ => by simp at hl
  | _ :: _, [], _, hl, _, _ => by simp at hl

theorem slots_ofFn {ty : ClifTy} : ∀ (n : Nat) (f : Fin n → Nat),
    (Vals.ofFn (V := Slot) (ty := ty) f).slots = List.ofFn f
  | 0, _ => rfl
  | n + 1, f => by simp [Vals.ofFn, Vals.slots, slots_ofFn n, List.ofFn_succ]

theorem uniformToList_carries {ty : ClifTy} : ∀ (n k : Nat),
    Vals.uniformToList (carriesFrom k (List.replicate n ty)) = List.range' k n
  | 0, _ => rfl
  | n + 1, k => by
    simp [List.replicate_succ, carriesFrom, Vals.uniformToList, Vals.head, Vals.tail,
      uniformToList_carries n (k + 1), List.range'_succ]

/-- A loop's first `j` trips of a lane's run. -/
def trips (f : LSt → Nat → Option LSt) (st : LSt) (j : Nat) : LSt :=
  ((List.range j).foldlM f st).getD st

theorem trips_succ {f : LSt → Nat → Option LSt} {st st' : LSt} {n : Nat}
    (h : (List.range n).foldlM f st = some st') :
    ∀ j, j < n → f (trips f st j) j = some (trips f st (j + 1)) := by
  intro j hj
  have key : ∀ m, m ≤ n → ∃ s, (List.range m).foldlM f st = some s := by
    intro m hm
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hm
    rw [List.range_add, List.foldlM_append] at h
    cases hm' : (List.range m).foldlM f st with
    | none => rw [hm'] at h; cases h
    | some s => exact ⟨s, rfl⟩
  obtain ⟨s, hs⟩ := key j (by omega)
  obtain ⟨s', hs'⟩ := key (j + 1) hj
  rw [List.range_succ, List.foldlM_append, hs] at hs'
  simp only [List.foldlM_cons, List.foldlM_nil, bind, Option.bind] at hs'
  unfold trips
  rw [hs, List.range_succ, List.foldlM_append, hs]
  simp only [Option.getD_some, List.foldlM_cons, List.foldlM_nil, bind, Option.bind]
  cases hf : f s j with
  | none => rw [hf] at hs'; cases hs'
  | some s2 => rfl

theorem trips_end {f : LSt → Nat → Option LSt} {st st' : LSt} {n : Nat}
    (h : (List.range n).foldlM f st = some st') : trips f st n = st' := by
  unfold trips; rw [h]; rfl


theorem mem_eraseDups' : ∀ (l : List Nat) (a : Nat), a ∈ l.eraseDups ↔ a ∈ l
  | [], a => by simp
  | x :: xs, a => by
    rw [List.eraseDups_cons, List.mem_cons, mem_eraseDups' (xs.filter fun b => !b == x) a, List.mem_filter,
      List.mem_cons]
    by_cases h : a = x <;> simp [h]
termination_by l => l.length
decreasing_by
  simp only [List.length_cons]
  exact Nat.lt_succ_of_le (List.length_filter_le _ _)

/-- **A statement writes only its written registers.** -/
theorem runL_frame (hw : Hw) (nbuf cta l : Nat) (o : Nat → UInt32) (rd : Buf → Nat → Option Nat) :
    ∀ (s : EWStmt) (i : Nat) (st st' : LSt), runL hw nbuf cta l o rd i s st = some st' →
      ∀ k, k ∉ Body.written s → st'.reg k = st.reg k := by
  intro s
  induction s with
  | skip => intro i st st' h k _; cases h; rfl
  | seq a b iha ihb =>
    intro i st st' h k hk
    simp only [runL, bind, Option.bind] at h
    split at h
    · cases h
    rename_i sa hsa
    simp only [Body.written, mem_eraseDups', List.mem_append, not_or] at hk
    rw [ihb i sa st' h k hk.2, iha i st sa hsa k hk.1]
  | setR n e =>
    intro i st st' h k hk; cases h
    simp only [Body.written, List.mem_singleton] at hk
    simp [upd, hk]
  | shflXor d s m =>
    intro i st st' h k hk
    simp only [runL] at h; split at h
    · cases h; simp only [Body.written, List.mem_singleton] at hk; simp [upd, hk]
    · cases h
  | loadIdx d b ix =>
    intro i st st' h k hk
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h
    simp only [Body.written, List.mem_singleton] at hk; simp [upd, hk]
  | storeLane b ix n =>
    intro i st st' h k hk
    obtain ⟨_, _, _, _, _, _, _, rfl⟩ := runL_store h
    rfl
  | forN n body ih =>
    intro i st st' h k hk
    simp only [runL] at h
    split at h
    · simp only [Body.written] at hk
      exact foldlM_inv _ (fun s' => s'.reg k = st.reg k)
        (fun s1 j s2 h12 h1 => (ih j s1 s2 h12 k hk).trans h1) _ _ _ h rfl
    · cases h
  | forM b a body ih =>
    intro i st st' h k hk
    obtain ⟨_, n, _, _, hf⟩ := runL_forM h
    simp only [Body.written] at hk
    exact foldlM_inv _ (fun s' => s'.reg k = st.reg k)
      (fun s1 j s2 h12 h1 => (ih j s1 s2 h12 k hk).trans h1) _ _ _ hf rfl
  | storeLane0 b ix n =>
    intro i st st' h k _
    rcases runL_store0 h with ⟨_, _, _, _, _, _, _, _, rfl⟩ | ⟨_, rfl⟩ <;> rfl
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i st st' h k hk
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h
    simp only [Body.written, List.mem_cons, List.not_mem_nil, or_false, not_or] at hk
    simp [upd, hk.1, hk.2.1, hk.2.2.1, hk.2.2.2]
  | _ => intro i st st' h; simp [runL] at h


theorem ofFn_getD_map {β : Type} (ws : List Nat) (f : Nat → β) :
    List.ofFn (fun j : Fin ws.length => f (ws.getD j.val 0)) = ws.map f := by
  apply List.ext_getElem <;> simp

theorem lookups_map {c : Body.Ctx Slot} {r : Body.Regs Slot} {Γ : Env} {reg : Nat → F32}
    (h : RegRel c r Γ reg) (ws : List Nat) :
    (ws.map (Body.get c r)).mapM (fun s => Γ[s]?) = some (ws.map fun k => V.sc .f32 (reg k).bits) := by
  induction ws with
  | nil => rfl
  | cons k ks ih => simp [List.mapM_cons, h k, ih]

theorem mapM_range' (A : Env) : ∀ (vs : List V) (n : Nat), (∀ j, j < vs.length → A[n + j]? = vs[j]?) →
    (List.range' n vs.length).mapM (fun s => A[s]?) = some vs
  | [], _, _ => rfl
  | v :: vs, n, h => by
    simp only [List.length_cons, List.range'_succ, List.mapM_cons]
    rw [show A[n]? = some v by simpa using h 0 (by simp),
      mapM_range' A vs (n + 1) (fun j hj => by rw [Nat.add_assoc, Nat.add_comm 1 j]; simpa using h (j + 1) (by simp; omega))]
    rfl

theorem slots_carries {ty : ClifTy} : ∀ (m k : Nat), (carriesFrom k (List.replicate m ty)).slots = List.range' k m
  | 0, _ => rfl
  | m + 1, k => by simp [List.replicate_succ, carriesFrom, Vals.slots, slots_carries m (k + 1), List.range'_succ]

theorem s32_ofNat_toNat {j : Nat} (hj : j < 2 ^ 32) : (s32 (UInt32.ofNat j)).toNat = j := by
  simp only [s32, UInt32.toNat_toUInt64, toNat_ofNat32 hj]

/-- **A counted loop of a body ends**: trips `0 … n-1` of `run`, each ending as
    the lane's trip says, end with the lane's state after `n`. -/
theorem wpE_bodyLoop (nbuf cta l i : Nat) {start : Nat} {c : Body.Ctx Slot} {r : Body.Regs Slot} {Γ : Env}
    {w : World} {nS : Slot .i32} {n : Nat} (ws : List Nat)
    (run : Body.Ctx Slot → Body.Regs Slot → Prog Slot Lvl (Body.Regs Slot)) (stF : Nat → LSt)
    (hnS : Γ[nS]? = some (.sc .i32 (s32 (UInt32.ofNat n)))) (hn : n < 2 ^ 31)
    (hc : CtxRel c Γ nbuf cta l i) (hreg : RegRel c r Γ (stF 0).reg)
    (hm : w.mem = (stF 0).mem) (hcl : collCalls w start = (stF 0).calls) (hs : start ≤ w.obs.length)
    (hframe : ∀ j, j ≤ n → ∀ k, k ∉ ws → (stF j).reg k = (stF 0).reg k)
    (hrun : ∀ j, j < n → ∀ (c' : Body.Ctx Slot) (r' : Body.Regs Slot) (Γ' : Env)
      (w' : World) (Q' : Body.Regs Slot → Env → World → Prop),
      CtxRel c' Γ' nbuf cta l j → RegRel c' r' Γ' (stF j).reg → w'.mem = (stF j).mem →
      collCalls w' start = (stF j).calls → start ≤ w'.obs.length →
      (∀ r'' Γ'' w'', Extends Γ' Γ'' → RegRel c' r'' Γ'' (stF (j + 1)).reg → w''.mem = (stF (j + 1)).mem →
        collCalls w'' start = (stF (j + 1)).calls → start ≤ w''.obs.length → Q' r'' Γ'' w'') →
      wpE cfg (run c' r') Q' Γ' w')
    {Q : Body.Regs Slot → Env → World → Prop}
    (hQ : ∀ r' Γ' w', Extends Γ Γ' → RegRel c r' Γ' (stF n).reg → w'.mem = (stF n).mem →
      collCalls w' start = (stF n).calls → start ≤ w'.obs.length → Q r' Γ' w') :
    wpE cfg (Body.loop c r nS ws run) Q Γ w := by
  simp only [Body.loop, Body.c32]
  refine wpE_bind (wpE_iconst32u (by decide) ?_)
  have hnlt : nS < Γ.size := by
    rcases Nat.lt_or_ge nS Γ.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at hnS; cases hnS
  let Γ0 := Γ.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))
  let vals (j : Nat) : List V := ws.map fun k => V.sc .f32 ((stF j).reg k).bits
  let Inv (cs : List V) (wi : World) : Prop := ∃ j, j ≤ n ∧ cs = V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j ∧
    wi.mem = (stF j).mem ∧ collCalls wi start = (stF j).calls ∧ start ≤ wi.obs.length
  let μ (cs : List V) (_ : World) : Nat := match cs with
    | .sc _ x :: _ => n - x.toNat
    | _ => 0
  let Hd (cs : List V) (_ : World) (a : Cond Slot × Vals Slot (List.replicate ws.length .f32) × Unit) (Γ1 : Env) (w1 : World) :
      Prop := a = (contIfULt Γ0.size nS, carriesFrom (Γ0.size + 1) (List.replicate ws.length .f32), ()) ∧
    Γ1 = bindAt Γ0 Γ0.size cs ∧ Inv cs w1
  let X (vs : List V) (wx : World) : Prop := vs = vals n ∧ wx.mem = (stF n).mem ∧
    collCalls wx start = (stF n).calls ∧ start ≤ wx.obs.length
  have hlt0 : nS < Γ0.size := by
    show nS < (Γ.push _).size
    rw [Array.size_push]
    exact Nat.lt_succ_of_lt hnlt
  have hev : ∀ j (w1 : World), j ≤ n →
      evalOp w1.mem (bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j)) (.icmp .ult Γ0.size nS) =
        some (boolV (cmpInt .ult .i32 (s32 (UInt32.ofNat j)) (s32 (UInt32.ofNat n)))) := by
    intro j w1 _
    have h1 : (bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j))[Γ0.size]? =
        some (V.sc .i32 (s32 (UInt32.ofNat j))) := bindAt_head _ _ _ _
    have h2 : (bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j))[nS]? =
        some (V.sc .i32 (s32 (UInt32.ofNat n))) := by
      rw [bindAt_lt hlt0 hlt0]
      exact get_push hnS
    simp [evalOp, Sem.get, h1, h2, zipIntCmp, ClifTy.isInt]
  refine wpE_bind (wpE_wloopN Inv μ Hd X ?_ ?_ ?_ ?_ ?_ ?_)
  · refine ⟨_, ?_, 0, Nat.zero_le _, rfl, hm, hcl, hs⟩
    simp only [Vals.slots, slots_ofFn, ofFn_getD_map, List.mapM_cons]
    rw [get_push_size, lookups_map (hreg.ext (Extends.push _ _))]
    simp [vals]
  · intro wi cs hinv
    exact wpE_pure ⟨Extends.refl _, rfl, rfl, hinv⟩
  · rintro cs _ a Γ1 w1 ⟨rfl, rfl, j, hj, rfl, -, -, -⟩
    exact ⟨_, _, by simp only [contIfULt, contIf]; rw [hev j w1 hj]; rfl⟩
  · rintro cs _ a Γ1 w1 t f ⟨rfl, rfl, j, hj, rfl, hmj, hcj, hsj⟩ hev' hexit
    simp only [contIfULt, contIf] at hev' hexit
    rw [hev j w1 hj] at hev'
    have hjn : j = n := by
      rw [cmp32_ult] at hev'
      by_cases hlt : UInt32.ofNat j < UInt32.ofNat n
      · simp [hlt, boolV] at hev'; obtain ⟨-, rfl⟩ := hev'; simp at hexit
      · rw [UInt32.lt_iff_toNat_lt, toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)] at hlt
        omega
    subst hjn
    refine ⟨vals j, ?_, rfl, hmj, hcj, hsj⟩
    simp only [slots_carries]
    have hlen : (vals j).length = ws.length := by simp [vals]
    rw [← hlen]
    rw [mapM_range' _ (vals j) (Γ0.size + 1) (fun p hp => by
      rw [show Γ0.size + 1 + p = Γ0.size + (p + 1) by omega, Array.getElem?_push,
        if_neg (by rw [bindAt_size]; simp; omega), bindAt_at]
      simp)]
  · rintro cs _ a Γ1 wb t f ⟨rfl, rfl, j, hj, rfl, hmj, hcj, hsj⟩ _ hev' hexit
    simp only [contIfULt, contIf] at hev' hexit
    rw [hev j wb hj] at hev'
    have hjn : j < n := by
      rw [cmp32_ult] at hev'
      by_cases hlt : UInt32.ofNat j < UInt32.ofNat n
      · rw [UInt32.lt_iff_toNat_lt, toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)] at hlt
        exact hlt
      · simp [hlt, boolV] at hev'; obtain ⟨-, rfl⟩ := hev'; simp at hexit
    have hE1 : Extends Γ (bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j)) :=
      (Extends.push _ _).trans ((Extends.refl _).bindAt_of (Nat.le_refl _) _)
    generalize hΓ1 : bindAt Γ0 Γ0.size (V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j) = Γ1 at hE1
    have hE2 : Extends Γ (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) (V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j)) :=
      (hE1.push_of _).trans ((Extends.refl _).bindAt_of (by simp) _)
    generalize hΓ2 : bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) (V.sc .i32 (s32 (UInt32.ofNat j)) :: vals j) = Γ2 at hE2
    have hci : Γ2[Γ1.size + 1]? = some (V.sc .i32 (s32 (UInt32.ofNat j))) := by rw [← hΓ2]; exact bindAt_head _ _ _ _
    have hcv : ∀ p, Γ2[Γ1.size + 1 + (p + 1)]? = (vals j)[p]? := by
      intro p; rw [← hΓ2, bindAt_at]; simp
    simp only [carriesFrom, Vals.head, Vals.tail]
    have hc' : CtxRel { c with i := Γ1.size + 1 } Γ2 nbuf cta l j :=
      ⟨hE2.get hc.lane, hE2.get hc.cta, hci, fun b => hE2.get (hc.buf b), hE2.get hc.zf⟩
    have hreg' : RegRel { c with i := Γ1.size + 1 } (List.foldl (fun r x => Body.set r x.fst x.snd) r
        (ws.zip (carriesFrom (Γ1.size + 1 + 1) (List.replicate ws.length ClifTy.f32)).uniformToList)) Γ2
        (stF j).reg := by
      rw [uniformToList_carries]
      refine RegRel.back ws (List.range' (Γ1.size + 1 + 1) ws.length) r (by simp) ?_ ?_
      · intro p k sl hk hsl
        have hp : p < ws.length := by
          rcases Nat.lt_or_ge p ws.length with h | h
          · exact h
          · rw [List.getElem?_eq_none h] at hk; cases hk
        rw [List.getElem?_range' hp] at hsl
        simp only [Nat.one_mul, Option.some.injEq] at hsl
        subst hsl
        rw [show Γ1.size + 1 + 1 + p = Γ1.size + 1 + (p + 1) by omega, hcv p]
        simp [vals, List.getElem?_map, hk]
      · intro k hk
        rw [hframe j (by omega) k hk]
        exact hE2.get (hreg k)
    refine wpE_bind (hrun j hjn _ _ Γ2 wb _ hc' hreg' hmj hcj hsj fun r'' Γ'' w'' hE hreg'' hm'' hc'' hs'' => ?_)
    refine wpE_bind (wpE_iconst32u (by decide) ?_)
    refine wpE_bind (wpE_iadd32 (x := UInt32.ofNat j) (get_push (hE.get hci)) get_push_size ?_)
    have hr1 : RegRel c r'' Γ'' (stF (j + 1)).reg := hreg''
    refine wpE_pure ⟨V.sc .i32 (s32 (UInt32.ofNat (j + 1))) :: vals (j + 1), ?_,
      ⟨j + 1, by omega, rfl, hm'', hc'', hs''⟩, ?_⟩
    · simp only [Vals.slots, slots_ofFn, ofFn_getD_map, List.mapM_cons, get_push_size]
      rw [lookups_map (hr1.ext ((Extends.push _ _).push_of _))]
      simp [vals, UInt32.ofNat_add]
    · simp only [μ]
      rw [s32_ofNat_toNat (by omega), s32_ofNat_toNat (by omega)]
      omega
  · rintro Γb vs w' nE hEb hnE ⟨rfl, hmn, hcn, hsn⟩
    have hE : Extends Γ (bindAt Γb nE (vals n)) :=
      ((Extends.push _ _).trans hEb).bindAt (by simp at hnE; omega) _
    refine wpE_pure (hQ _ _ _ hE ?_ hmn hcn hsn)
    rw [uniformToList_carries]
    refine RegRel.back ws (List.range' nE ws.length) r (by simp) ?_ ?_
    · intro p k sl hk hsl
      have hp : p < ws.length := by
        rcases Nat.lt_or_ge p ws.length with h | h
        · exact h
        · rw [List.getElem?_eq_none h] at hk; cases hk
      rw [List.getElem?_range' hp] at hsl
      simp only [Nat.one_mul, Option.some.injEq] at hsl
      subst hsl
      rw [bindAt_at]
      simp [vals, List.getElem?_map, hk]
    · intro k hk
      rw [hframe n (Nat.le_refl n) k hk]
      exact hE.get (hreg k)

theorem bits32 {x : F32} (h : x.bits.toNat < 2 ^ 32) : x.bits = s32 x.bits.toUInt32 := by
  apply UInt64.toNat_inj.mp
  simp [s32, Nat.mod_eq_of_lt h]

/-- **A warp statement's code ends as its lane's run**: for any answers of
    words, where the lane's run (`runL`) finishes, the code ends with the
    lane's registers, memory and collectives. -/
theorem wpE_stmt (hw : Hw) {ans : List Answer} {start : Nat} (hA : AnswersFrom ans start cfg)
    (hW : Words ans) (hex : ∀ x w, cfg.locals Intr.ex2Approx.slot [.sc .f32 x] w =
      .ok (some (.sc .f32 (hw.approx .ex2Approx x)), w))
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (nbuf cta l : Nat) (hl : l < 2 ^ 32)
    (rd : Buf → Nat → Option Nat) :
    ∀ (s : EWStmt) (i : Nat) (c : Body.Ctx Slot) (r : Body.Regs Slot) (Γ : Env) (w : World) (st st' : LSt)
      (Q : Body.Regs Slot → Env → World → Prop),
      runL hw nbuf cta l (wordAt ans) rd i s st = some st' →
      CtxRel c Γ nbuf cta l i → RegRel c r Γ st.reg → Regs32 st.reg → RdOk nbuf st.mem rd →
      st.mem.nan = hw.nan → w.mem = st.mem → collCalls w start = st.calls → start ≤ w.obs.length →
      (∀ r' Γ' w', Extends Γ Γ' → RegRel c r' Γ' st'.reg → w'.mem = st'.mem →
        collCalls w' start = st'.calls → start ≤ w'.obs.length → Q r' Γ' w') →
      wpE cfg (Body.stmt c r s : Prog Slot Lvl (Body.Regs Slot)) Q Γ w := by
  intro s
  induction s with
  | skip =>
    intro i c r Γ w st st' Q hr hc hreg h32 hrd hnan hm hcl hs hQ
    cases hr
    rw [Body.stmt]
    exact wpE_pure (hQ _ _ _ (Extends.refl _) hreg hm hcl hs)
  | seq a b iha ihb =>
    intro i c r Γ w st st' Q hr hc hreg h32 hrd hnan hm hcl hs hQ
    simp only [runL, bind, Option.bind] at hr
    split at hr
    · cases hr
    rename_i sa hsa
    rw [Body.stmt]
    refine wpE_bind (iha i c r Γ w st sa _ hsa hc hreg h32 hrd hnan hm hcl hs fun ra Γa wa hEa hrega hma hcla hsa' => ?_)
    exact ihb i c ra Γa wa sa st' Q hr (hc.ext hEa) hrega (runL_32 hw hw32 nbuf cta l _ _ a i st sa hsa h32)
      (runL_rdOk a i st sa hsa hrd) ((runL_nan a i st sa hsa hrd).trans hnan) hma hcla hsa'
      fun r' Γ' w' hE' => hQ r' Γ' w' (hEa.trans hE')
  | setR n e =>
    intro i c r Γ w st st' Q hr hc hreg h32 hrd hnan hm hcl hs hQ
    cases hr
    simp only [Body.stmt]
    refine wpE_bind (wpE_fexp hw c r st.reg hex e Γ w _ start (by rw [hm]; exact hnan) hs hreg fun v Γ' w' hE hm' hc' hs' hv => ?_)
    exact wpE_pure (hQ _ _ _ hE ((hreg.ext hE).set n hv) (hm'.trans hm) (hc'.trans hcl) hs')
  | shflXor dd ss m =>
    intro i c r Γ w st st' Q hr hc hreg h32 hrd hnan hm hcl hs hQ
    simp only [runL] at hr
    split at hr
    · rename_i hm31
      cases hr
      simp only [Body.stmt]
      have hss := hreg ss
      rw [bits32 (h32 ss)] at hss
      refine wpE_bind (wpE_bitcastI32 hss ?_)
      refine wpE_bind (wpE_iconst32u (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_)
      refine wpE_bind (wpE_coll2 hA (c := .shflBfly) rfl Body.shfl rfl (get_push get_push_size) get_push_size hs
        (k := st.calls.length) (by rw [hcl]) (valAt_words hW _) ?_)
      refine wpE_bind (wpE_bitcastF32 get_push_size ?_)
      refine wpE_pure (hQ _ _ _ (((Extends.push _ _).push_of _).push_of _ |>.push_of _)
        (((hreg.ext (((Extends.push _ _).push_of _).push_of _ |>.push_of _))).set dd get_push_size) hm ?_ ?_)
      · rw [collCalls_call hs, hcl, ← bits32 (h32 ss)]
        simp [Intr.isColl]
      · simp [obsCall]; omega
    · cases hr
  | loadIdx dd b ix =>
    intro i c r Γ w st st' Q hr hc hreg h32 hrd hnan hm hcl hs hQ
    simp only [runL, bind, Option.bind] at hr
    split at hr; · cases hr
    rename_i j hj
    dsimp only at hr
    split at hr; · cases hr
    rename_i v hv
    cases hr
    simp only [Body.stmt]
    refine wpE_bind (wpE_idx ix Γ w _ j hc (by rw [hm]; exact hrd) hj fun vi Γi hEi hvi => ?_)
    refine wpE_bind (wpE_at4 ((hc.ext hEi).buf b) hvi fun va Γa hEa hva => ?_)
    refine wpE_bind (wpE_loadF32 hva (by rw [hm]; exact hv) ?_)
    exact wpE_pure (hQ _ _ _ ((hEi.trans hEa).push_of _)
      ((hreg.ext ((hEi.trans hEa).push_of _)).set dd get_push_size) hm hcl hs)
  | storeLane b ix n =>
    intro i c r Γ w st st' Q hr hc hreg h32 hrd hnan hm hcl hs hQ
    obtain ⟨j, m', hj, -, -, -, hm', rfl⟩ := runL_store hr
    simp only [Body.stmt]
    refine wpE_bind (wpE_idx ix Γ w _ j hc (by rw [hm]; exact hrd) hj fun vi Γi hEi hvi => ?_)
    refine wpE_bind (wpE_at4 ((hc.ext hEi).buf b) hvi fun va Γa hEa hva => ?_)
    refine wpE_bind (wpE_storeF32 ((hreg.ext (hEi.trans hEa)) n) hva (by rw [hm]; exact hm') ?_)
    exact wpE_pure (hQ _ _ _ (hEi.trans hEa) (hreg.ext (hEi.trans hEa)) rfl
      (by change collCalls (obsStore _ _ _ _) _ = _; rw [collCalls_obsStore hs]; exact hcl)
      (by simp [obsStore]; omega))
  | forN n body ih =>
    intro i c r Γ w st st' Q hr hc hreg h32 hrd hnan hm hcl hs hQ
    simp only [runL] at hr
    split at hr
    · rename_i hn
      let f : LSt → Nat → Option LSt := fun st j => runL hw nbuf cta l (wordAt ans) rd j body st
      have hsucc := trips_succ (f := f) hr
      have h0 : trips f st 0 = st := rfl
      have hrdj : ∀ j, j ≤ n → RdOk nbuf (trips f st j).mem rd := by
        intro j hj
        induction j with
        | zero => exact hrd
        | succ j ihj => exact runL_rdOk body j _ _ (hsucc j (by omega)) (ihj (by omega))
      have hnanj : ∀ j, j ≤ n → (trips f st j).mem.nan = hw.nan := by
        intro j hj
        induction j with
        | zero => exact hnan
        | succ j ihj =>
          exact (runL_nan body j _ _ (hsucc j (by omega)) (hrdj j (by omega))).trans (ihj (by omega))
      have hframe : ∀ j, j ≤ n → ∀ k, k ∉ Body.written body → (trips f st j).reg k = (trips f st 0).reg k := by
        intro j hj k hk
        induction j with
        | zero => rfl
        | succ j ihj => rw [runL_frame hw nbuf cta l _ _ body j _ _ (hsucc j (by omega)) k hk, ihj (by omega)]
      have h32j : ∀ j, j ≤ n → Regs32 (trips f st j).reg := by
        intro j hj
        induction j with
        | zero => exact h32
        | succ j ihj => exact runL_32 hw hw32 nbuf cta l _ _ body j _ _ (hsucc j (by omega)) (ihj (by omega))
      simp only [Body.stmt, Body.c32]
      refine wpE_bind (wpE_iconst32u (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_)
      refine wpE_bodyLoop nbuf cta l i (Body.written body) (fun c' r' => Body.stmt c' r' body) (trips f st)
        (by simpa using get_push_size) hn (hc.ext (Extends.push _ _)) (h0 ▸ hreg.ext (Extends.push _ _))
        (h0 ▸ hm) (h0 ▸ hcl) (by simpa using hs) hframe ?_ ?_
      · intro j hj c' r' Γ' w' Q' hc' hreg' hm' hcl' hs' hQ'
        exact ih j c' r' Γ' w' _ _ Q' (hsucc j hj) hc' hreg' (h32j j (by omega)) (hrdj j (by omega)) (hnanj j (by omega)) hm' hcl'
          hs' hQ'
      · intro r' Γ' w' hE hreg' hm' hcl' hs'
        rw [trips_end hr] at hreg' hm' hcl'
        exact hQ r' Γ' w' ((Extends.push _ _).trans hE) hreg' hm' hcl' hs'
    · cases hr
  | forM bb a body ih =>
    intro i c r Γ w st st' Q hr hc hreg h32 hrd hnan hm hcl hs hQ
    obtain ⟨ha, n, hnr, hn, hr⟩ := runL_forM hr
    have hnm := (hrd bb a n hnr).2
    simp only [rdM, Option.map_eq_some_iff] at hnm
    obtain ⟨nv, hnv, rfl⟩ := hnm
    let f : LSt → Nat → Option LSt := fun st j => runL hw nbuf cta l (wordAt ans) rd j body st
    have hsucc := trips_succ (f := f) hr
    have h0 : trips f st 0 = st := rfl
    have hrdj : ∀ j, j ≤ nv.toNat → RdOk nbuf (trips f st j).mem rd := by
      intro j hj
      induction j with
      | zero => exact hrd
      | succ j ihj => exact runL_rdOk body j _ _ (hsucc j (by omega)) (ihj (by omega))
    have hnanj : ∀ j, j ≤ nv.toNat → (trips f st j).mem.nan = hw.nan := by
      intro j hj
      induction j with
      | zero => exact hnan
      | succ j ihj =>
        exact (runL_nan body j _ _ (hsucc j (by omega)) (hrdj j (by omega))).trans (ihj (by omega))
    have hframe : ∀ j, j ≤ nv.toNat → ∀ k, k ∉ Body.written body → (trips f st j).reg k = (trips f st 0).reg k := by
      intro j hj k hk
      induction j with
      | zero => rfl
      | succ j ihj => rw [runL_frame hw nbuf cta l _ _ body j _ _ (hsucc j (by omega)) k hk, ihj (by omega)]
    have h32j : ∀ j, j ≤ nv.toNat → Regs32 (trips f st j).reg := by
      intro j hj
      induction j with
      | zero => exact h32
      | succ j ihj => exact runL_32 hw hw32 nbuf cta l _ _ body j _ _ (hsucc j (by omega)) (ihj (by omega))
    simp only [Body.stmt, Body.c32]
    refine wpE_bind (wpE_iconst32u (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_)
    refine wpE_bind (wpE_at4 (j := a) (get_push (hc.buf bb)) (by simpa using get_push_size) fun va Γa hEa hva => ?_)
    refine wpE_bind (wpE_loadI32 hva (by rw [hm]; exact hnv) ?_)
    have hE : Extends Γ (Γa.push (.sc .i32 nv)) := ((Extends.push _ _).trans hEa).push_of _
    refine wpE_bodyLoop nbuf cta l i (Body.written body) (fun c' r' => Body.stmt c' r' body) (trips f st)
      (by rw [s32_toNat (load4_lt hnv)]; exact get_push_size) hn (hc.ext hE) (h0 ▸ hreg.ext hE)
      (h0 ▸ hm) (h0 ▸ hcl) hs hframe ?_ ?_
    · intro j hj c' r' Γ' w' Q' hc' hreg' hm' hcl' hs' hQ'
      exact ih j c' r' Γ' w' _ _ Q' (hsucc j hj) hc' hreg' (h32j j (by omega)) (hrdj j (by omega)) (hnanj j (by omega)) hm' hcl'
        hs' hQ'
    · intro r' Γ' w' hE' hreg' hm' hcl' hs'
      rw [trips_end hr] at hreg' hm' hcl'
      exact hQ r' Γ' w' (hE.trans hE') hreg' hm' hcl' hs'
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i c r Γ w st st' Q hr hc hreg h32 hrd hnan hm hcl hs hQ
    simp only [runL, bind, Option.bind] at hr
    split at hr; · cases hr
    rename_i j hj
    dsimp only at hr
    split at hr; · cases hr
    rename_i v0 h0
    dsimp only at hr
    split at hr; · cases hr
    rename_i v1 h1
    dsimp only at hr
    split at hr; · cases hr
    rename_i v2 h2
    dsimp only at hr
    split at hr; · cases hr
    rename_i v3 h3
    cases hr
    simp only [Body.stmt]
    refine wpE_bind (wpE_idx ix Γ w _ j hc (by rw [hm]; exact hrd) hj fun vi Γi hEi hvi => ?_)
    refine wpE_bind (wpE_at4 ((hc.ext hEi).buf b) hvi fun va Γa hEa hva => ?_)
    refine wpE_bind (wpE_loadF32 hva (by rw [hm]; exact h0) ?_)
    refine wpE_bind (wpE_iaddImm (get_push hva) ?_)
    refine wpE_bind (wpE_loadF32 get_push_size (by rw [hm]; exact h1) ?_)
    refine wpE_bind (wpE_iaddImm (get_push (get_push (get_push (get_push hva)))) ?_)
    refine wpE_bind (wpE_loadF32 get_push_size (by rw [hm]; exact h2) ?_)
    refine wpE_bind (wpE_iaddImm (get_push (get_push (get_push (get_push (get_push (get_push (get_push hva))))))) ?_)
    refine wpE_bind (wpE_loadF32 get_push_size (by rw [hm]; exact h3) ?_)
    have hE : Extends Γ Γa := hEi.trans hEa
    refine wpE_pure (hQ _ _ _ (by extc) ?_ hm hcl hs)
    exact ((((hreg.ext (by extc)).set d0 (by look)).set d1 (by look)).set d2 (by look)).set d3 (by look)
  | storeLane0 b ix n =>
    intro i c r Γ w st st' Q hr hc hreg h32 hrd hnan hm hcl hs hQ
    simp only [Body.stmt, Body.c32]
    refine wpE_bind (wpE_iconst32u (by decide) ?_)
    refine wpE_bind ?_
    simp only [when]
    refine wpE_bind (wpE_ifteT (Qj := fun _ Γ' u => Extends Γ Γ' ∧ u.mem = st'.mem ∧ collCalls u start = st'.calls ∧
      start ≤ u.obs.length) rfl (get_push hc.lane) get_push_size ?_ ?_ (fun jv _ _ _ => ⟨[], by cases jv; rfl⟩) ?_)
    · intro hcmp
      rw [cmp32_eq] at hcmp
      have hl0 : l = 0 := by
        have h := beq_iff_eq.mp hcmp
        have := congrArg UInt32.toNat h
        rw [toNat_ofNat32 hl] at this
        simpa using this
      obtain ⟨_, j, m', hj, -, -, -, hm', rfl⟩ := (runL_store0 hr).resolve_right (fun h => h.1 hl0)
      refine wpE_bind ?_
      refine wpE_bind (wpE_idx ix _ w _ j (hc.ext (Extends.push _ _) |>.ext (Extends.push _ _)) (by rw [hm]; exact hrd) hj
        fun vi Γi hEi hvi => ?_)
      refine wpE_bind (wpE_at4 ((hc.ext ((((Extends.push _ _).push_of _)).trans hEi)).buf b) hvi fun va Γa hEa hva => ?_)
      refine wpE_storeF32 ((hreg.ext ((((Extends.push _ _).push_of _).trans hEi).trans hEa)) n) hva
        (by rw [hm]; exact hm') ?_
      exact wpE_pure ⟨((((Extends.push _ _).push_of _).trans hEi).trans hEa), rfl,
        by change collCalls (obsStore _ _ _ _) _ = _; rw [collCalls_obsStore hs]; exact hcl,
        by simp [obsStore]; omega⟩
    · intro hcmp ne hne
      rw [cmp32_eq] at hcmp
      have hl0 : l ≠ 0 := by
        intro h; subst h; exact absurd hcmp (by decide)
      obtain ⟨_, rfl⟩ := (runL_store0 hr).resolve_left (fun h => hl0 h.1)
      exact wpE_pure ⟨((Extends.push _ _).push_of _).bindAt_of hne _, hm, hcl, hs⟩
    · intro nJ Γ' jv vs w' hle hvs ⟨hE, hm', hc', hs'⟩
      have hn : Γ.size ≤ nJ := Nat.le_trans hE.size hle
      have hrg : st'.reg = st.reg :=
        funext fun k => runL_frame hw nbuf cta l _ _ (.storeLane0 b ix n) i st st' hr k (by simp [Body.written])
      exact wpE_pure (wpE_pure (hQ _ _ _ (hE.bindAt hn _) (hrg ▸ hreg.ext (hE.bindAt hn _)) hm' hc' hs'))
  | _ => intro i c r Γ w st st' Q hr; simp [runL] at hr


end Steps

end AlgorithmLib.ML.BodyProof
