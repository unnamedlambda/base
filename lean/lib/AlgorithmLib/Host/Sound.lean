module
public import AlgorithmLib.Host.Blocks
meta import AlgorithmLib.Host.Blocks
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `HProgSound` — compiling a term preserves what it does

`compile_sound`: a body that passes `scopeOk`, run by the term interpreter to a
trace, runs to the same trace and the same world as the function `compileBody`
makes of it, under some block budget.

`HProgBlocks` proves this for straight-line bodies, against an invariant that
does not survive control flow: that the block's value array agrees with the
term's environment on *every* slot below the count. Two places break it. The
else arm is numbered past the then arm's slots, so the term pads its
environment over slots the blocks never wrote; and the code after a loop is
numbered past the body's slots, which the blocks hold from whichever trip last
reached them. Neither interpreter reads those slots in a body that passes
`scopeOk`, and the coupling here says exactly that: the two stores agree on the
slots in scope (`Rel`), and the blocks never write below where a region started
(`Frame`).

The simulation is forward and on successful runs. A term that gets stuck has no
behaviour to preserve, and the two interpreters word their failures
differently.
-/

namespace AlgorithmLib.HProg

open AlgorithmLib.IR
open AlgorithmLib.HProg.Sem

-- ---------------------------------------------------------------------------
-- Scopes
-- ---------------------------------------------------------------------------

theorem Scope.mem_nil (i : Nat) : Scope.mem [] i = false := rfl

theorem Scope.mem_cons (c d : Nat) (rest : Scope) (i : Nat) :
    Scope.mem ((c, d) :: rest) i = true ↔ (c ≤ i ∧ i < d) ∨ Scope.mem rest i = true := by
  simp [Scope.mem, List.any_cons]

theorem Scope.mem_add (S : Scope) (a b i : Nat) :
    (S.add a b).mem i = true ↔ S.mem i = true ∨ (a ≤ i ∧ i < b) := by
  cases S with
  | nil => simp [Scope.add, Scope.mem]
  | cons p rest =>
      obtain ⟨c, d⟩ := p
      simp only [Scope.add]
      split
      · rename_i h
        simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
        obtain ⟨⟨h1, h2⟩, h3⟩ := h
        rw [Scope.mem_cons, Scope.mem_cons]
        by_cases hr : Scope.mem rest i = true <;> simp [hr] <;> omega
      · rw [Scope.mem_cons, Scope.mem_cons]
        by_cases hr : Scope.mem rest i = true <;> simp [hr] <;> omega

theorem Scope.sub_mem {S T : Scope} (h : S.sub T = true) {i : Nat} (hi : S.mem i = true) :
    T.mem i = true := by
  simp only [Scope.sub, List.all_eq_true] at h
  simp only [Scope.mem, List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hi
  obtain ⟨⟨a, b⟩, hmem, h1, h2⟩ := hi
  have := h (a, b) hmem
  exact this i (List.mem_range'_1.mpr ⟨h1, by omega⟩)

theorem inS_iff (S : Scope) (n r : Nat) : inS S n r = true ↔ r < n ∧ S.mem r = true := by
  simp [inS]

theorem allIn_iff (S : Scope) (n : Nat) (rs : List R) :
    allIn S n rs = true ↔ ∀ r ∈ rs, r < n ∧ S.mem r = true := by
  simp [allIn, inS]

-- ---------------------------------------------------------------------------
-- The coupling
-- ---------------------------------------------------------------------------

/-- The two stores agree on every slot in scope: both hold a value there, and
    it is the same one. -/
def Rel (S : Scope) (vals : Blocks.Vals) (Γ : Sem.Env) : Prop :=
  ∀ i, S.mem i = true → ∃ x, Γ[i]? = some x ∧ Blocks.getV vals ⟨i⟩ = some x

/-- The blocks wrote nothing below `n`: whatever was readable there still reads
    the same. What carries a loop's entry scope round to its next trip. -/
def Frame (n : Nat) (vals vals' : Blocks.Vals) : Prop :=
  ∀ i, i < n → ∀ x, Blocks.getV vals ⟨i⟩ = some x → Blocks.getV vals' ⟨i⟩ = some x

theorem Frame.refl (n : Nat) (vals : Blocks.Vals) : Frame n vals vals :=
  fun _ _ _ h => h

theorem Frame.trans {n : Nat} {a b c : Blocks.Vals} (h1 : Frame n a b) (h2 : Frame n b c) :
    Frame n a c :=
  fun i hi x hx => h2 i hi x (h1 i hi x hx)

theorem Frame.mono {m n : Nat} {a b : Blocks.Vals} (hle : m ≤ n) (h : Frame n a b) :
    Frame m a b :=
  fun i hi x hx => h i (by omega) x hx

theorem frame_setV (n d : Nat) (hd : n ≤ d) (vals : Blocks.Vals) (x : V) :
    Frame n vals (Blocks.setV vals ⟨d⟩ x) := by
  intro i hi y hy
  rw [getV_setV_ne vals d i x (by omega) (getV_lt vals i y hy)]
  exact hy

theorem Rel.sub {S T : Scope} {vals : Blocks.Vals} {Γ : Sem.Env} (h : Rel T vals Γ)
    (hs : S.sub T = true) : Rel S vals Γ :=
  fun i hi => h i (Scope.sub_mem hs hi)

theorem Rel.lt {S : Scope} {vals : Blocks.Vals} {Γ : Sem.Env} (h : Rel S vals Γ)
    {i : Nat} (hi : S.mem i = true) : i < Γ.size := by
  obtain ⟨x, hx, _⟩ := h i hi
  exact (Array.getElem?_eq_some_iff.mp hx).1

/-- Growing the scope by an empty range changes nothing. -/
theorem Rel.add_empty {S : Scope} {vals : Blocks.Vals} {Γ : Sem.Env} (h : Rel S vals Γ)
    (a : Nat) : Rel (S.add a a) vals Γ := by
  intro i hi
  rcases (Scope.mem_add S a a i).mp hi with h1 | h1
  · exact h i h1
  · omega

/-- **One binding keeps the coupling**, with the new slot in scope. -/
theorem Rel.push {S : Scope} {vals : Blocks.Vals} {Γ : Sem.Env} (h : Rel S vals Γ)
    (n : Nat) (hn : Γ.size = n) (v : V) :
    Rel (S.add n (n + 1)) (Blocks.setV vals ⟨n⟩ v) (Γ.push v) := by
  intro i hi
  rcases (Scope.mem_add S n (n + 1) i).mp hi with h1 | h1
  · obtain ⟨x, hx, hxv⟩ := h i h1
    have hlt : i < n := hn ▸ h.lt h1
    refine ⟨x, ?_, ?_⟩
    · rw [Array.getElem?_push, if_neg (by omega)]; exact hx
    · rw [getV_setV_ne vals n i v (by omega) (getV_lt vals i x hxv)]; exact hxv
  · have hi' : i = n := by omega
    subst hi'
    refine ⟨v, ?_, getV_setV_self vals ⟨i⟩ v⟩
    rw [Array.getElem?_push, if_pos hn.symm]

/-- In-scope slots read the same on both sides, as a list: what makes a call's
    argument vector, a jump's arguments and a join's exports the same. -/
theorem Rel.mapM {S : Scope} {vals : Blocks.Vals} {Γ : Sem.Env} (h : Rel S vals Γ) :
    ∀ (rs : List R), (∀ r ∈ rs, S.mem r = true) →
      (rs.map (fun r => (⟨r⟩ : Val))).mapM (Blocks.getV vals) = rs.mapM (fun r => Γ[r]?) := by
  intro rs
  induction rs with
  | nil => intro _; simp
  | cons r rs ih =>
      intro hall
      obtain ⟨x, hx, hxv⟩ := h r (hall r (by simp))
      have hrs := ih (fun q hq => hall q (by simp [hq]))
      simp only [List.map_cons, List.mapM_cons, hx, hxv, hrs]

-- ---------------------------------------------------------------------------
-- One statement, masked
-- ---------------------------------------------------------------------------

/-- `dyn2` with agreement at the two operands only. -/
theorem mdyn2 (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (f : R → R → Op)
    (hf : Sem.Reloc2 f) (a b : R) (x y : V)
    (hxΓ : Γ[a]? = some x) (hyΓ : Γ[b]? = some y)
    (hxv : Blocks.getV vals ⟨a⟩ = some x) (hyv : Blocks.getV vals ⟨b⟩ = some y)
    (inst : Inst) (d : Val)
    (hi : Blocks.evalInst m vals inst
            = (do let x ← Blocks.getV vals ⟨a⟩
                  let y ← Blocks.getV vals ⟨b⟩
                  pure (d, ← Sem.evalOp m #[x, y] (f 0 1)))) :
    Blocks.evalInst m vals inst = (Sem.evalOp m Γ (f a b)).map (fun w => (d, w)) := by
  rw [hi, hf m Γ a b x y hxΓ hyΓ, hxv, hyv]
  cases hE : Sem.evalOp m #[x, y] (f 0 1) <;> simp [hE]

theorem mdyn1 (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (f : R → Op)
    (hf : Sem.Reloc1 f) (a : R) (x : V)
    (hxΓ : Γ[a]? = some x) (hxv : Blocks.getV vals ⟨a⟩ = some x)
    (inst : Inst) (d : Val)
    (hi : Blocks.evalInst m vals inst
            = (do let x ← Blocks.getV vals ⟨a⟩
                  pure (d, ← Sem.evalOp m #[x] (f 0)))) :
    Blocks.evalInst m vals inst = (Sem.evalOp m Γ (f a)).map (fun w => (d, w)) := by
  rw [hi, hf m Γ a x hxΓ, hxv]
  cases hE : Sem.evalOp m #[x] (f 0) <;> simp [hE]

theorem mdyn3 (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (f : R → R → R → Op)
    (hf : Sem.Reloc3 f) (a b c : R) (x y z : V)
    (hxΓ : Γ[a]? = some x) (hyΓ : Γ[b]? = some y) (hzΓ : Γ[c]? = some z)
    (hxv : Blocks.getV vals ⟨a⟩ = some x) (hyv : Blocks.getV vals ⟨b⟩ = some y)
    (hzv : Blocks.getV vals ⟨c⟩ = some z)
    (inst : Inst) (d : Val)
    (hi : Blocks.evalInst m vals inst
            = (do let x ← Blocks.getV vals ⟨a⟩
                  let y ← Blocks.getV vals ⟨b⟩
                  let z ← Blocks.getV vals ⟨c⟩
                  pure (d, ← Sem.evalOp m #[x, y, z] (f 0 1 2)))) :
    Blocks.evalInst m vals inst = (Sem.evalOp m Γ (f a b c)).map (fun w => (d, w)) := by
  rw [hi, hf m Γ a b c x y z hxΓ hyΓ hzΓ, hxv, hyv, hzv]
  cases hE : Sem.evalOp m #[x, y, z] (f 0 1 2) <;> simp [hE]

theorem instOf_of {s : CS} {st : Stmt} {inst : Inst}
    (h : (emitStmt s st).cur = inst :: s.cur) : instOf s st = inst := by
  rw [instOf, h]; rfl

/-- **Every operation computes what the term computes**, given agreement at its
    operands alone; and its instruction reaches `runInsts`' evaluating arm. -/
theorem op_mstep (env : FnEnv) (s : CS) (n : Nat) (ha : Aligned s n) (o : Op)
    (hr : ∀ r ∈ o.regs, r < n) (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env)
    (hag : ∀ r ∈ o.regs, ∃ x, Γ[r]? = some x ∧ Blocks.getV vals ⟨r⟩ = some x) :
    Blocks.evalInst m vals (instOf s (.op o)) = (Sem.evalOp m Γ o).map (fun x => (⟨n⟩, x))
    ∧ ∀ (w : Sem.World) (rest : List Inst),
        Blocks.runInsts env ⟨vals, w⟩ (instOf s (.op o) :: rest)
          = match Blocks.evalInst w.mem vals (instOf s (.op o)) with
            | none => .stuck "instruction is undefined here"
            | some (d, r) => Blocks.runInsts env ⟨Blocks.setV vals d r, w⟩ rest := by
  cases o with
  | iconst ty k =>
      rw [instOf_of (emit_iconst s n ha ty k)]
      exact ⟨by simp [Blocks.evalInst, Blocks.viaOp, Sem.evalOp], fun _ _ => rfl⟩
  | iadd a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_iadd s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.iadd Sem.reloc2_iadd a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | isub a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_isub s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.isub Sem.reloc2_isub a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | imul a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_imul s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.imul Sem.reloc2_imul a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | udiv a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_udiv s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.udiv Sem.reloc2_udiv a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | ineg a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_ineg s n ha a h1)]
      exact ⟨mdyn1 m vals Γ Op.ineg Sem.reloc1_ineg a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | ishl a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_ishl s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.ishl Sem.reloc2_ishl a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | ushr a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_ushr s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.ushr Sem.reloc2_ushr a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | band a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_band s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.band Sem.reloc2_band a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | bandNot a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_bandNot s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.bandNot Sem.reloc2_bandNot a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | bor a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_bor s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.bor Sem.reloc2_bor a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | bxor a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_bxor s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.bxor Sem.reloc2_bxor a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | ireduce32 a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_ireduce32 s n ha a h1)]
      exact ⟨mdyn1 m vals Γ Op.ireduce32 Sem.reloc1_ireduce32 a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | uextend64 a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_uextend64 s n ha a h1)]
      exact ⟨mdyn1 m vals Γ Op.uextend64 Sem.reloc1_uextend64 a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | sextend64 a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_sextend64 s n ha a h1)]
      exact ⟨mdyn1 m vals Γ Op.sextend64 Sem.reloc1_sextend64 a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | icmp c a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_icmp s n ha c a b h1 h2)]
      exact ⟨mdyn2 m vals Γ (Op.icmp c) (Sem.reloc2_icmp c) a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | select c a b =>
      have h1 : c < n := hr c (by simp [Op.regs])
      have h2 : a < n := hr a (by simp [Op.regs])
      have h3 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag c (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag a (by simp [Op.regs])
      obtain ⟨x2, g2, v2⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_select s n ha c a b h1 h2 h3)]
      exact ⟨mdyn3 m vals Γ Op.select Sem.reloc3_select c a b x0 x1 x2 g0 g1 g2 v0 v1 v2 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | bitselect c a b =>
      have h1 : c < n := hr c (by simp [Op.regs])
      have h2 : a < n := hr a (by simp [Op.regs])
      have h3 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag c (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag a (by simp [Op.regs])
      obtain ⟨x2, g2, v2⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_bitselect s n ha c a b h1 h2 h3)]
      exact ⟨mdyn3 m vals Γ Op.bitselect Sem.reloc3_bitselect c a b x0 x1 x2 g0 g1 g2 v0 v1 v2 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | ctz a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_ctz s n ha a h1)]
      exact ⟨mdyn1 m vals Γ Op.ctz Sem.reloc1_ctz a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | popcnt a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_popcnt s n ha a h1)]
      exact ⟨mdyn1 m vals Γ Op.popcnt Sem.reloc1_popcnt a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | fconst ty b =>
      rw [instOf_of (emit_fconst s n ha ty b)]
      exact ⟨by simp [Blocks.evalInst, Blocks.viaOp, Sem.evalOp], fun _ _ => rfl⟩
  | fadd a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_fadd s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.fadd Sem.reloc2_fadd a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | fsub a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_fsub s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.fsub Sem.reloc2_fsub a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | fmul a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_fmul s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.fmul Sem.reloc2_fmul a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | fmax a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_fmax s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.fmax Sem.reloc2_fmax a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | fmin a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_fmin s n ha a b h1 h2)]
      exact ⟨mdyn2 m vals Γ Op.fmin Sem.reloc2_fmin a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | fneg a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_fneg s n ha a h1)]
      exact ⟨mdyn1 m vals Γ Op.fneg Sem.reloc1_fneg a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | fpromote a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_fpromote s n ha a h1)]
      exact ⟨mdyn1 m vals Γ Op.fpromote Sem.reloc1_fpromote a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | fcmp c a b =>
      have h1 : a < n := hr a (by simp [Op.regs])
      have h2 : b < n := hr b (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      obtain ⟨x1, g1, v1⟩ := hag b (by simp [Op.regs])
      rw [instOf_of (emit_fcmp s n ha c a b h1 h2)]
      exact ⟨mdyn2 m vals Γ (Op.fcmp c) (Sem.reloc2_fcmp c) a b x0 x1 g0 g1 v0 v1 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | fcvtFromSint ty a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_fcvtFromSint s n ha ty a h1)]
      exact ⟨mdyn1 m vals Γ (Op.fcvtFromSint ty) (Sem.reloc1_fcvtFromSint ty) a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | fcvtToUint ty a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_fcvtToUint s n ha ty a h1)]
      exact ⟨mdyn1 m vals Γ (Op.fcvtToUint ty) (Sem.reloc1_fcvtToUint ty) a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | splat ty a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_splat s n ha ty a h1)]
      exact ⟨mdyn1 m vals Γ (Op.splat ty) (Sem.reloc1_splat ty) a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | extractlane a l =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_extractlane s n ha a l h1)]
      exact ⟨mdyn1 m vals Γ (Op.extractlane · l) (Sem.reloc1_extractlane l) a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | vhighBits a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_vhighBits s n ha a h1)]
      exact ⟨mdyn1 m vals Γ Op.vhighBits Sem.reloc1_vhighBits a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | bitcast ty a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_bitcast s n ha ty a h1)]
      exact ⟨mdyn1 m vals Γ (Op.bitcast ty) (Sem.reloc1_bitcast ty) a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩
  | load op a =>
      have h1 : a < n := hr a (by simp [Op.regs])
      obtain ⟨x0, g0, v0⟩ := hag a (by simp [Op.regs])
      rw [instOf_of (emit_load s n ha op a h1)]
      exact ⟨mdyn1 m vals Γ (Op.load op) (Sem.reloc1_load op) a x0 g0 v0 _ ⟨n⟩ rfl, fun _ _ => rfl⟩

/-- The masked frame: one statement, from a state agreeing on `S`, lands on the
    term's world in a state agreeing on `S` and on whatever it bound, having
    written only at `n`. -/
def MStep (env : FnEnv) (cfg : Sem.Cfg) (S : Scope) (n : Nat) (st : Stmt) (inst : Inst) :
    Prop :=
  ∀ (w w₁ : Sem.World) (Γ Γ₁ : Sem.Env) (vals : Blocks.Vals) (rest : List Inst),
    Γ.size = n → Rel S vals Γ →
    Sem.runStmt cfg Γ w st = .ok Γ₁ w₁ →
    ∃ vals₁, Blocks.runInsts env ⟨vals, w⟩ (inst :: rest)
               = Blocks.runInsts env ⟨vals₁, w₁⟩ rest
             ∧ Rel (S.add n (n + st.binds)) vals₁ Γ₁ ∧ Γ₁.size = n + st.binds
             ∧ Frame n vals vals₁

theorem op_MStep (env : FnEnv) (cfg : Sem.Cfg) (s : CS) (S : Scope) (n : Nat)
    (ha : Aligned s n) (o : Op) (hr : ∀ r ∈ o.regs, r < n ∧ S.mem r = true) :
    MStep env cfg S n (.op o) (instOf s (.op o)) := by
  intro w w₁ Γ Γ₁ vals rest hs hrel hrun
  obtain ⟨hE, hR⟩ := op_mstep env s n ha o (fun r h => (hr r h).1) w.mem vals Γ
    (fun r h => hrel r (hr r h).2)
  cases hop : Sem.evalOp w.mem Γ o with
  | none => rw [Sem.runStmt, hop] at hrun; simp at hrun
  | some v =>
      rw [Sem.runStmt, hop] at hrun
      simp only [Sem.Outcome.ok.injEq] at hrun
      obtain ⟨h1, h2⟩ := hrun
      subst h1; subst h2
      refine ⟨Blocks.setV vals ⟨n⟩ v, ?_, by simpa [Stmt.binds] using hrel.push n hs v,
              by simp [hs, Stmt.binds], frame_setV n n (Nat.le_refl n) vals v⟩
      rw [hR w rest, hE, hop]
      rfl

theorem istore8_MStep (env : FnEnv) (cfg : Sem.Cfg) (S : Scope) (n : Nat) (v a : R)
    (hvS : S.mem v = true) (haS : S.mem a = true) :
    MStep env cfg S n (.istore8 v a) (.istore8 ⟨v⟩ ⟨a⟩) := by
  intro w w₁ Γ Γ₁ vals rest hs hrel hr
  obtain ⟨xv, hxΓ, hxv⟩ := hrel v hvS
  obtain ⟨xa, haΓ, hav⟩ := hrel a haS
  rw [Sem.runStmt_istore8, hxΓ, haΓ] at hr
  cases xv with
  | vec _ _ => simp at hr
  | sc t b =>
    cases xa with
    | vec _ _ => simp at hr
    | sc t2 addr =>
      dsimp only at hr
      cases hm : w.mem.store addr 1 (b &&& 0xff) with
      | none => rw [hm] at hr; simp at hr
      | some m =>
          rw [hm] at hr
          simp only [Sem.Outcome.ok.injEq] at hr
          obtain ⟨h1, h2⟩ := hr
          subst h1; subst h2
          refine ⟨vals, ?_, by simpa [Stmt.binds] using hrel.add_empty n,
                  by simp [Stmt.binds, hs], Frame.refl n vals⟩
          rw [runInsts_istore8, hxv, hav]
          dsimp only
          rw [hm]

theorem storeUnaligned_MStep (env : FnEnv) (cfg : Sem.Cfg) (S : Scope) (n : Nat) (v a : R)
    (hvS : S.mem v = true) (haS : S.mem a = true) :
    MStep env cfg S n (.storeUnaligned v a) (.store ⟨v⟩ ⟨a⟩) := by
  intro w w₁ Γ Γ₁ vals rest hs hrel hr
  obtain ⟨xv, hxΓ, hxv⟩ := hrel v hvS
  obtain ⟨xa, haΓ, hav⟩ := hrel a haS
  rw [Sem.runStmt_storeUnaligned, hxΓ, haΓ] at hr
  cases xv with
  | vec _ _ => simp at hr
  | sc t b =>
    cases xa with
    | vec _ _ => simp at hr
    | sc t2 addr =>
      dsimp only at hr
      cases hm : w.mem.store addr (Sem.tyBytes t) b with
      | none => rw [hm] at hr; simp at hr
      | some m =>
          rw [hm] at hr
          simp only [Sem.Outcome.ok.injEq] at hr
          obtain ⟨h1, h2⟩ := hr
          subst h1; subst h2
          refine ⟨vals, ?_, by simpa [Stmt.binds] using hrel.add_empty n,
                  by simp [Stmt.binds, hs], Frame.refl n vals⟩
          rw [runInsts_store]
          have hd : Blocks.doStore ⟨vals, w⟩ ⟨v⟩ ⟨a⟩ none
              = .ok ⟨vals, { Sem.obsStore w addr (Sem.tyBytes t) b with mem := m }⟩ := by
            simp [Blocks.doStore, hxv, hav, hm]
          rw [hd]

theorem store_MStep (env : FnEnv) (cfg : Sem.Cfg) (S : Scope) (n : Nat) (ty : ClifTy)
    (v a : R) (hvS : S.mem v = true) (haS : S.mem a = true) :
    MStep env cfg S n (.store ty v a) (.storeTyped ty ⟨v⟩ ⟨a⟩) := by
  intro w w₁ Γ Γ₁ vals rest hs hrel hr
  obtain ⟨xv, hxΓ, hxv⟩ := hrel v hvS
  obtain ⟨xa, haΓ, hav⟩ := hrel a haS
  rw [Sem.runStmt_store, hxΓ, haΓ] at hr
  cases xa with
  | vec _ _ => simp at hr
  | sc t2 addr =>
    dsimp only at hr
    cases xv with
    | sc t b =>
      dsimp only at hr
      split at hr
      · next m hm =>
        simp only [Sem.Outcome.ok.injEq] at hr
        obtain ⟨h1, h2⟩ := hr
        subst h1; subst h2
        refine ⟨vals, ?_, by simpa [Stmt.binds] using hrel.add_empty n,
                by simp [Stmt.binds, hs], Frame.refl n vals⟩
        rw [runInsts_storeTyped]
        have hd : Blocks.doStore ⟨vals, w⟩ ⟨v⟩ ⟨a⟩ (some ty)
            = .ok ⟨vals, { Sem.obsStore w addr (Sem.tyBytes ty) b with mem := m }⟩ := by
          simp [Blocks.doStore, hxv, hav, hm]
        rw [hd]
      · simp at hr
    | vec t ls =>
      dsimp only at hr
      split at hr
      · next m hm =>
        simp only [Sem.Outcome.ok.injEq] at hr
        obtain ⟨h1, h2⟩ := hr
        subst h1; subst h2
        refine ⟨vals, ?_, by simpa [Stmt.binds] using hrel.add_empty n,
                by simp [Stmt.binds, hs], Frame.refl n vals⟩
        rw [runInsts_storeTyped]
        have hd : Blocks.doStore ⟨vals, w⟩ ⟨v⟩ ⟨a⟩ (some ty)
            = .ok ⟨vals, { Sem.obsStore w addr (Sem.tyBytes ty) 0 with mem := m }⟩ := by
          simp only [Blocks.doStore, hxv, hav]
          rw [hm]
          simp
        rw [hd]
      · simp at hr

theorem call_MStep (env : FnEnv) (cfg : Sem.Cfg) (S : Scope) (n : Nat)
    (c : IR.Callee) (args : List R) (hall : ∀ r ∈ args, S.mem r = true) :
    MStep env cfg S n (.call c args) (.call (some ⟨n⟩) c (args.map (fun r => ⟨r⟩))) := by
  intro w w₁ Γ Γ₁ vals rest hs hrel hr
  have hargs := hrel.mapM args hall
  rw [Sem.runStmt_call] at hr
  rw [runInsts_call]
  simp only [hargs]
  cases hm : args.mapM (fun r => Γ[r]?) with
  | none => rw [hm] at hr; simp at hr
  | some vs =>
    rw [hm] at hr
    cases hc : c with
    | «local» i => rw [hc] at hr; simp at hr
    | native => rw [hc] at hr; simp at hr
    | ffi f =>
      rw [hc] at hr
      dsimp only at hr
      cases hcf : Sem.callImport f.cname vs (Sem.obsCall w (.ffi f) vs) with
      | none => rw [hcf] at hr; simp at hr
      | some p =>
        obtain ⟨res, w'⟩ := p
        rw [hcf] at hr
        dsimp only at hr
        cases hres : res with
        | none => rw [hres] at hr; simp at hr
        | some v =>
          rw [hres] at hr
          dsimp only at hr
          simp only [Sem.Outcome.ok.injEq] at hr
          obtain ⟨h1, h2⟩ := hr
          subst h1; subst h2
          refine ⟨Blocks.setV vals ⟨n⟩ v, ?_, by simpa [Stmt.binds] using hrel.push n hs v,
                  by simp [hs, Stmt.binds], frame_setV n n (Nat.le_refl n) vals v⟩
          simp only [hcf, hres]

theorem callVoid_MStep (env : FnEnv) (cfg : Sem.Cfg) (S : Scope) (n : Nat)
    (c : IR.Callee) (args : List R) (hall : ∀ r ∈ args, S.mem r = true) :
    MStep env cfg S n (.callVoid c args) (.call none c (args.map (fun r => ⟨r⟩))) := by
  intro w w₁ Γ Γ₁ vals rest hs hrel hr
  have hargs := hrel.mapM args hall
  rw [Sem.runStmt_callVoid] at hr
  rw [runInsts_call]
  simp only [hargs]
  cases hm : args.mapM (fun r => Γ[r]?) with
  | none => rw [hm] at hr; simp at hr
  | some vs =>
    rw [hm] at hr
    cases hc : c with
    | «local» i => rw [hc] at hr; simp at hr
    | native => rw [hc] at hr; simp at hr
    | ffi f =>
      rw [hc] at hr
      dsimp only at hr
      cases hcf : Sem.callImport f.cname vs (Sem.obsCall w (.ffi f) vs) with
      | none => rw [hcf] at hr; simp at hr
      | some p =>
        obtain ⟨res, w'⟩ := p
        rw [hcf] at hr
        dsimp only at hr
        simp only [Sem.Outcome.ok.injEq] at hr
        obtain ⟨h1, h2⟩ := hr
        subst h1; subst h2
        refine ⟨vals, ?_, by simpa [Stmt.binds] using hrel.add_empty n,
                by simp [hs, Stmt.binds], Frame.refl n vals⟩
        simp only [hcf]

/-- **Every statement satisfies the masked frame against what it compiles to**,
    given that it reads only slots in scope. -/
theorem mstep_emit (env : FnEnv) (cfg : Sem.Cfg) (s : CS) (S : Scope) (n : Nat)
    (ha : Aligned s n) (st : Stmt) (hr : ∀ r ∈ st.regs, r < n ∧ S.mem r = true) :
    MStep env cfg S n st (instOf s st) := by
  obtain ⟨_, _, he⟩ := ha
  cases st with
  | op o => exact op_MStep env cfg s S n ⟨‹_›, ‹_›, he⟩ o hr
  | store ty v a =>
      have h1 := hr v (by simp [Stmt.regs])
      have h2 := hr a (by simp [Stmt.regs])
      have hi : instOf s (.store ty v a) = .storeTyped ty ⟨v⟩ ⟨a⟩ := by
        simp [instOf, emitStmt, CS.get, he v h1.1, he a h2.1]
      rw [hi]; exact store_MStep env cfg S n ty v a h1.2 h2.2
  | storeUnaligned v a =>
      have h1 := hr v (by simp [Stmt.regs])
      have h2 := hr a (by simp [Stmt.regs])
      have hi : instOf s (.storeUnaligned v a) = .store ⟨v⟩ ⟨a⟩ := by
        simp [instOf, emitStmt, CS.get, he v h1.1, he a h2.1]
      rw [hi]; exact storeUnaligned_MStep env cfg S n v a h1.2 h2.2
  | istore8 v a =>
      have h1 := hr v (by simp [Stmt.regs])
      have h2 := hr a (by simp [Stmt.regs])
      have hi : instOf s (.istore8 v a) = .istore8 ⟨v⟩ ⟨a⟩ := by
        simp [instOf, emitStmt, CS.get, he v h1.1, he a h2.1]
      rw [hi]; exact istore8_MStep env cfg S n v a h1.2 h2.2
  | call c args =>
      have hall : ∀ r ∈ args, r < n := fun r h => (hr r (by simpa [Stmt.regs] using h)).1
      have hi : instOf s (.call c args) = .call (some ⟨n⟩) c (args.map (fun r => ⟨r⟩)) := by
        rw [instOf, emit_call s n ⟨‹_›, ‹_›, he⟩ c args hall]; rfl
      rw [hi]
      exact call_MStep env cfg S n c args (fun r h => (hr r (by simpa [Stmt.regs] using h)).2)
  | callVoid c args =>
      have hall : ∀ r ∈ args, r < n := fun r h => (hr r (by simpa [Stmt.regs] using h)).1
      have hi : instOf s (.callVoid c args) = .call none c (args.map (fun r => ⟨r⟩)) := by
        rw [instOf, emit_callVoid s n ⟨‹_›, ‹_›, he⟩ c args hall]; rfl
      rw [hi]
      exact callVoid_MStep env cfg S n c args (fun r h => (hr r (by simpa [Stmt.regs] using h)).2)

theorem scStmts_count : ∀ (ss : List Stmt) (S : Scope) (n : Nat) (S' : Scope) (n' : Nat),
    scStmts S n ss = some (S', n') → n' = n + (ss.map Stmt.binds).sum := by
  intro ss
  induction ss with
  | nil => intro S n S' n' h; simp [scStmts] at h; simp [h.2]
  | cons st ss ih =>
      intro S n S' n' h
      simp only [scStmts] at h
      split at h
      · have := ih _ _ S' n' h; simp only [List.map_cons, List.sum_cons]; omega
      · simp at h

/-- **A straight-line run, masked.** The instructions `emitStmts` writes reach
    the same continuation, in a state agreeing on the scope `scStmts` computes,
    having written nothing below `n`. -/
theorem stmts_msim (env : FnEnv) (cfg : Sem.Cfg) :
    ∀ (ss : List Stmt) (s : CS) (S : Scope) (n : Nat) (S' : Scope) (n' : Nat)
      (w w' : Sem.World) (Γ Γ' : Sem.Env) (vals : Blocks.Vals) (rest : List Inst),
      scStmts S n ss = some (S', n') → Aligned s n → Γ.size = n → Rel S vals Γ →
      Sem.runStmts cfg Γ w ss = .ok Γ' w' →
      ∃ vals', Blocks.runInsts env ⟨vals, w⟩ (emittedList s ss ++ rest)
                 = Blocks.runInsts env ⟨vals', w'⟩ rest
               ∧ Rel S' vals' Γ' ∧ Γ'.size = n' ∧ Frame n vals vals' := by
  intro ss
  induction ss with
  | nil =>
      intro s S n S' n' w w' Γ Γ' vals rest hsc _ hs hrel hr
      simp [scStmts] at hsc
      simp [Sem.runStmts] at hr
      obtain ⟨h1, h2⟩ := hsc; obtain ⟨h3, h4⟩ := hr
      subst h1; subst h2; subst h3; subst h4
      exact ⟨vals, by simp [emittedList], hrel, hs, Frame.refl n vals⟩
  | cons st ss ih =>
      intro s S n S' n' w w' Γ Γ' vals rest hsc ha hs hrel hr
      simp only [scStmts] at hsc
      split at hsc
      · rename_i hin
        rw [allIn_iff] at hin
        rw [Sem.runStmts] at hr
        cases hone : Sem.runStmt cfg Γ w st with
        | stuck m => rw [hone] at hr; simp at hr
        | ok Γ₁ w₁ =>
            rw [hone] at hr
            obtain ⟨vals₁, hrun₁, hrel₁, hs₁, hf₁⟩ :=
              mstep_emit env cfg s S n ha st hin w w₁ Γ Γ₁ vals
                (emittedList (emitStmt s st) ss ++ rest) hs hrel hone
            obtain ⟨vals', hrun, hrel', hs', hf'⟩ :=
              ih (emitStmt s st) _ _ S' n' w₁ w' Γ₁ Γ' vals₁ rest hsc
                (emitStmt_aligned s n ha st) hs₁ hrel₁ hr
            refine ⟨vals', ?_, hrel', hs', hf₁.trans (hf'.mono (by omega))⟩
            simp only [emittedList, List.cons_append]
            rw [hrun₁]; exact hrun
      · simp at hsc

-- ---------------------------------------------------------------------------
-- What a successful check says, construct by construct
-- ---------------------------------------------------------------------------

theorem scGo_straight_inv {f : Nat} {lb : List SLbl} {S : Scope} {n : Nat} {ss : List Stmt}
    {ps : List Piece} {S' : Scope} {n' : Nat}
    (h : scGo (f + 1) lb S n (.straight ss :: ps) = some (S', n')) :
    ∃ S1 n1, scStmts S n ss = some (S1, n1) ∧ scGo f lb S1 n1 ps = some (S', n') := by
  simp only [scGo] at h
  cases h1 : scStmts S n ss with
  | none => rw [h1] at h; cases h
  | some p => obtain ⟨S1, n1⟩ := p; rw [h1] at h; exact ⟨S1, n1, rfl, h⟩

/-- The check of a top-tested loop, taken apart. -/
structure LoopOk (f : Nat) (lb : List SLbl) (S : Scope) (n : Nat) (l : Loop)
    (pre body : List Piece) (Sp : Scope) (np : Nat) (Sb : Scope) (nb : Nat) : Prop where
  hpre   : scGo f (⟨l.exitTys.length, l.pTys.length, S.add n (n + l.pTys.length)⟩ :: lb)
            (S.add n (n + l.pTys.length)) (n + l.pTys.length) pre = some (Sp, np)
  hbody  : scGo f (⟨l.exitTys.length, l.pTys.length, S.add n (n + l.pTys.length)⟩ :: lb)
            (Sp.add np (np + l.pTys.length)) (np + l.pTys.length) body = some (Sb, nb)
  hinit  : allIn S n l.init = true
  hinitN : l.init.length = l.pTys.length
  hpreT  : termsGo f pre = false
  hflag  : inS Sp np l.flag = true
  hexitR : allIn Sp np l.exitR = true
  hexitN : l.exitR.length = l.exitTys.length
  hneed  : (S.add n (n + l.pTys.length)).sub Sp = true
  hcont  : termsGo f body = true ∨ (allIn Sb nb l.cont = true ∧ l.cont.length = l.pTys.length)

theorem scGo_loop_inv {f : Nat} {lb : List SLbl} {S : Scope} {n : Nat} {l : Loop}
    {pre body ps : List Piece} {S' : Scope} {n' : Nat}
    (h : scGo (f + 1) lb S n (.loop l pre body :: ps) = some (S', n')) :
    ∃ Sp np Sb nb, LoopOk f lb S n l pre body Sp np Sb nb ∧
      scGo f lb ((S.add n (n + l.pTys.length)).add nb (nb + l.exitTys.length))
        (nb + l.exitTys.length) ps = some (S', n') := by
  simp only [scGo] at h
  cases h1 : scGo f (⟨l.exitTys.length, l.pTys.length, S.add n (n + l.pTys.length)⟩ :: lb)
      (S.add n (n + l.pTys.length)) (n + l.pTys.length) pre with
  | none => rw [h1] at h; cases h
  | some p =>
    obtain ⟨Sp, np⟩ := p
    rw [h1] at h
    simp only [Option.bind_eq_bind, Option.bind_some] at h
    cases h2 : scGo f (⟨l.exitTys.length, l.pTys.length, S.add n (n + l.pTys.length)⟩ :: lb)
        (Sp.add np (np + l.pTys.length)) (np + l.pTys.length) body with
    | none => rw [h2] at h; cases h
    | some q =>
      obtain ⟨Sb, nb⟩ := q
      rw [h2] at h
      simp only [Option.bind_some] at h
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true', Bool.or_eq_true] at hc
        obtain ⟨⟨⟨⟨⟨⟨⟨hi, hin⟩, hpt⟩, hf⟩, he⟩, hen⟩, hne⟩, hco⟩ := hc
        exact ⟨Sp, np, Sb, nb, ⟨h1, h2, hi, hin, hpt, hf, he, hen, hne, hco⟩, h⟩
      · cases h

/-- The check of a branch, taken apart. -/
structure IteOk (f : Nat) (lb : List SLbl) (S : Scope) (n : Nat) (m : IteMeta)
    (thn els : List Piece) (thnR elsR : List R)
    (St : Scope) (nt : Nat) (Se : Scope) (ne : Nat) : Prop where
  hthn   : scGo f lb S n thn = some (St, nt)
  hels   : scGo f lb S nt els = some (Se, ne)
  hflag  : inS S n m.flag = true
  hthnX  : termsGo f thn = true ∨
            (allIn St nt thnR = true ∧ thnR.length = m.jTys.length ∧ S.sub St = true)
  helsX  : termsGo f els = true ∨
            (allIn Se ne elsR = true ∧ elsR.length = m.jTys.length ∧ S.sub Se = true)

theorem scGo_ite_inv {f : Nat} {lb : List SLbl} {S : Scope} {n : Nat} {m : IteMeta}
    {thn els : List Piece} {thnR elsR : List R} {ps : List Piece} {S' : Scope} {n' : Nat}
    (h : scGo (f + 1) lb S n (.ite m thn els thnR elsR :: ps) = some (S', n')) :
    ∃ St nt Se ne, IteOk f lb S n m thn els thnR elsR St nt Se ne ∧
      ((termsGo f thn = true ∧ termsGo f els = true ∧ ps = [] ∧ S' = S ∧ n' = ne) ∨
       ((termsGo f thn && termsGo f els) = false ∧
        scGo f lb (S.add ne (ne + m.jTys.length)) (ne + m.jTys.length) ps = some (S', n'))) := by
  simp only [scGo] at h
  cases h1 : scGo f lb S n thn with
  | none => rw [h1] at h; cases h
  | some p =>
    obtain ⟨St, nt⟩ := p
    rw [h1] at h
    simp only [Option.bind_eq_bind, Option.bind_some] at h
    cases h2 : scGo f lb S nt els with
    | none => rw [h2] at h; cases h
    | some q =>
      obtain ⟨Se, ne⟩ := q
      rw [h2] at h
      simp only [Option.bind_some] at h
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true] at hc
        obtain ⟨⟨hfl, htx⟩, hex⟩ := hc
        refine ⟨St, nt, Se, ne, ⟨h1, h2, hfl, ?_, ?_⟩, ?_⟩
        · rcases htx with h | ⟨⟨a, b⟩, c⟩
          · exact .inl h
          · exact .inr ⟨a, b, c⟩
        · rcases hex with h | ⟨⟨a, b⟩, c⟩
          · exact .inl h
          · exact .inr ⟨a, b, c⟩
        · split at h
          · rename_i hte
            simp only [Bool.and_eq_true] at hte
            split at h
            · rename_i hps
              simp only [Option.some.injEq, Prod.mk.injEq] at h
              exact .inl ⟨hte.1, hte.2, List.isEmpty_iff.mp hps, h.1.symm, h.2.symm⟩
            · cases h
          · rename_i hte
            exact .inr ⟨by simpa using hte, h⟩
      · cases h

/-- The check of a bottom-tested loop, taken apart. -/
structure DLoopOk (f : Nat) (lb : List SLbl) (S : Scope) (n : Nat) (l : DLoop)
    (body : List Piece) (Sb : Scope) (nb : Nat) : Prop where
  hbody  : scGo f (⟨l.exitTys.length, l.pTys.length, S⟩ :: lb)
            (S.add n (n + l.pTys.length)) (n + l.pTys.length) body = some (Sb, nb)
  hinit  : allIn S n l.init = true
  hinitN : l.init.length = l.pTys.length
  hguard : ∀ g, l.guard = some g → inS S n g = true
  hexitN : l.exitIdx.length = l.exitTys.length
  hexitI : ∀ i ∈ l.exitIdx, i < l.pTys.length
  hback  : termsGo f body = true ∨
            (inS Sb nb l.flag = true ∧ allIn Sb nb l.cont = true ∧
             l.cont.length = l.pTys.length ∧ S.sub Sb = true)

theorem scGo_dloop_inv {f : Nat} {lb : List SLbl} {S : Scope} {n : Nat} {l : DLoop}
    {body ps : List Piece} {S' : Scope} {n' : Nat}
    (h : scGo (f + 1) lb S n (.dloop l body :: ps) = some (S', n')) :
    ∃ Sb nb, DLoopOk f lb S n l body Sb nb ∧
      scGo f lb (S.add nb (nb + l.exitTys.length)) (nb + l.exitTys.length) ps
        = some (S', n') := by
  simp only [scGo] at h
  cases h1 : scGo f (⟨l.exitTys.length, l.pTys.length, S⟩ :: lb)
      (S.add n (n + l.pTys.length)) (n + l.pTys.length) body with
  | none => rw [h1] at h; cases h
  | some p =>
    obtain ⟨Sb, nb⟩ := p
    rw [h1] at h
    have tail : ∀ (hg : ∀ g, l.guard = some g → inS S n g = true),
        allIn S n l.init = true → l.init.length = l.pTys.length →
        l.exitIdx.length = l.exitTys.length → (∀ i ∈ l.exitIdx, i < l.pTys.length) →
        (termsGo f body = true ∨ ((((inS Sb nb l.flag = true ∧ allIn Sb nb l.cont = true) ∧
            l.cont.length = l.pTys.length) ∧ S.sub Sb = true))) →
        scGo f lb (S.add nb (nb + l.exitTys.length)) (nb + l.exitTys.length) ps
          = some (S', n') →
        ∃ Sb' nb', DLoopOk f lb S n l body Sb' nb' ∧
          scGo f lb (S.add nb' (nb' + l.exitTys.length)) (nb' + l.exitTys.length) ps
            = some (S', n') := by
      intro hg hi hin hen hei hb hk
      refine ⟨Sb, nb, ⟨h1, hi, hin, hg, hen, hei, ?_⟩, hk⟩
      rcases hb with h | ⟨⟨⟨a, b⟩, c⟩, d⟩
      · exact .inl h
      · exact .inr ⟨a, b, c, d⟩
    cases hgl : l.guard with
    | none =>
      simp only [hgl, bind, Option.bind] at h
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, List.all_eq_true,
          decide_eq_true_eq, and_true] at hc
        obtain ⟨⟨⟨⟨hi, hin⟩, hen⟩, hei⟩, hb⟩ := hc
        exact tail (fun g hgs => by rw [hgl] at hgs; cases hgs) hi hin hen hei hb h
      · cases h
    | some g0 =>
      simp only [hgl, bind, Option.bind] at h
      split at h
      · rename_i hc
        simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, List.all_eq_true,
          decide_eq_true_eq] at hc
        obtain ⟨⟨⟨⟨⟨hi, hin⟩, hg⟩, hen⟩, hei⟩, hb⟩ := hc
        refine tail (fun g hgs => ?_) hi hin hen hei hb h
        rw [hgl] at hgs; cases hgs; exact hg
      · cases h

theorem scGo_br_inv {f : Nat} {lb : List SLbl} {S : Scope} {n : Nat} {d : Nat}
    {args : List R} {ps : List Piece} {S' : Scope} {n' : Nat}
    (h : scGo (f + 1) lb S n (.br d args :: ps) = some (S', n')) :
    ∃ L, lb[d]? = some L ∧ ps = [] ∧ allIn S n args = true ∧ args.length = L.exitN ∧
      L.need.sub S = true ∧ S' = S ∧ n' = n := by
  simp only [scGo] at h
  cases hL : lb[d]? with
  | none => rw [hL] at h; cases h
  | some L =>
    rw [hL] at h
    simp only at h
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, List.isEmpty_iff] at hc
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      exact ⟨L, rfl, hc.1.1.1, hc.1.1.2, hc.1.2, hc.2, h.1.symm, h.2.symm⟩
    · cases h

theorem scGo_cont_inv {f : Nat} {lb : List SLbl} {S : Scope} {n : Nat} {d : Nat}
    {args : List R} {ps : List Piece} {S' : Scope} {n' : Nat}
    (h : scGo (f + 1) lb S n (.cont d args :: ps) = some (S', n')) :
    ∃ L, lb[d]? = some L ∧ ps = [] ∧ allIn S n args = true ∧ args.length = L.carryN ∧
      S' = S ∧ n' = n := by
  simp only [scGo] at h
  cases hL : lb[d]? with
  | none => rw [hL] at h; cases h
  | some L =>
    rw [hL] at h
    simp only at h
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, List.isEmpty_iff] at hc
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      exact ⟨L, rfl, hc.1.1, hc.1.2, hc.2, h.1.symm, h.2.symm⟩
    · cases h

-- ---------------------------------------------------------------------------
-- Slot counts and fuel
-- ---------------------------------------------------------------------------

/-- Whether one piece leaves its region on every path. -/
def pieceTerm (f : Nat) : Piece → Bool
  | .br _ _ | .cont _ _ => true
  | .ite _ thn els _ _ => termsGo f thn && termsGo f els
  | _ => false

theorem termsGo_cons (f : Nat) (p : Piece) (ps : List Piece) :
    termsGo (f + 1) (p :: ps) = if ps.isEmpty then pieceTerm f p else termsGo f ps := by
  cases ps with
  | nil => cases p <;> rfl
  | cons q qs => cases p <;> rfl

theorem termsGo_nil (g : Nat) : termsGo g [] = false := by cases g <;> rfl

theorem slotsGo_nil (g n : Nat) : Sem.slotsGo g n [] = n := by cases g <;> rfl

theorem stmtsSlots_eq : ∀ (ss : List Stmt) (n : Nat),
    Sem.stmtsSlots n ss = n + (ss.map Stmt.binds).sum := by
  intro ss
  induction ss with
  | nil => intro n; rfl
  | cons st ss ih =>
      intro n
      have hstep : Sem.stmtsSlots n (st :: ss) = Sem.stmtsSlots (n + st.binds) ss := by
        cases st <;> rfl
      rw [hstep, ih]; simp only [List.map_cons, List.sum_cons]; omega

/-- **The count the check computes is the count the term's interpreter
    computes**, at any fuel from the check's up — so the fuel `emitCode` spends
    at depth and the fresh fuel `slotsOf` starts with agree, and so do the
    `termsGo` answers the slot count depends on. -/
theorem scGo_slots : ∀ (f : Nat) (lb : List SLbl) (S : Scope) (n : Nat) (c : List Piece)
    (S' : Scope) (n' : Nat), scGo f lb S n c = some (S', n') →
    n ≤ n' ∧ ∀ g, f ≤ g → Sem.slotsGo g n c = n' ∧ termsGo g c = termsGo f c := by
  intro f
  induction f with
  | zero => intro lb S n c S' n' h; simp [scGo] at h
  | succ f ih =>
    intro lb S n c S' n' h
    cases c with
    | nil =>
        simp only [scGo, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨_, rfl⟩ := h
        refine ⟨Nat.le_refl n, fun g hg => ?_⟩
        obtain ⟨g', rfl⟩ : ∃ g', g = g' + 1 := ⟨g - 1, by omega⟩
        exact ⟨rfl, by rw [termsGo_nil, termsGo_nil]⟩
    | cons p ps =>
      have hterm : ∀ g', f ≤ g' → (termsGo g' ps = termsGo f ps) → pieceTerm g' p = pieceTerm f p →
          termsGo (g' + 1) (p :: ps) = termsGo (f + 1) (p :: ps) := by
        intro g' _ h1 h2
        rw [termsGo_cons, termsGo_cons, h1, h2]
      cases p with
      | straight ss =>
          obtain ⟨S1, n1, h1, h2⟩ := scGo_straight_inv h
          have hn1 := scStmts_count ss S n S1 n1 h1
          obtain ⟨hle, hg⟩ := ih lb S1 n1 ps S' n' h2
          refine ⟨by omega, fun g hgf => ?_⟩
          obtain ⟨g', rfl⟩ : ∃ g', g = g' + 1 := ⟨g - 1, by omega⟩
          obtain ⟨hs, ht⟩ := hg g' (by omega)
          refine ⟨?_, hterm g' (by omega) ht rfl⟩
          show Sem.slotsGo g' (Sem.stmtsSlots n ss) ps = n'
          rw [stmtsSlots_eq, ← hn1]; exact hs
      | loop l pre body =>
          obtain ⟨Sp, np, Sb, nb, hok, hk⟩ := scGo_loop_inv h
          obtain ⟨hle1, hg1⟩ := ih _ _ _ pre Sp np hok.hpre
          obtain ⟨hle2, hg2⟩ := ih _ _ _ body Sb nb hok.hbody
          obtain ⟨hle3, hg3⟩ := ih _ _ _ ps S' n' hk
          refine ⟨by omega, fun g hgf => ?_⟩
          obtain ⟨g', rfl⟩ : ∃ g', g = g' + 1 := ⟨g - 1, by omega⟩
          refine ⟨?_, hterm g' (by omega) (hg3 g' (by omega)).2 rfl⟩
          show Sem.slotsGo g' (Sem.slotsGo g' (Sem.slotsGo g' (n + l.pTys.length) pre
            + l.pTys.length) body + l.exitTys.length) ps = n'
          rw [(hg1 g' (by omega)).1, (hg2 g' (by omega)).1, (hg3 g' (by omega)).1]
      | ite m thn els thnR elsR =>
          obtain ⟨St, nt, Se, ne, hok, hk⟩ := scGo_ite_inv h
          obtain ⟨hle1, hg1⟩ := ih _ _ _ thn St nt hok.hthn
          obtain ⟨hle2, hg2⟩ := ih _ _ _ els Se ne hok.hels
          rcases hk with ⟨ht1, ht2, hps, hS, hn⟩ | ⟨hnt, hk⟩
          · subst hps; subst hn
            refine ⟨by omega, fun g hgf => ?_⟩
            obtain ⟨g', rfl⟩ : ∃ g', g = g' + 1 := ⟨g - 1, by omega⟩
            have e1 := hg1 g' (by omega); have e2 := hg2 g' (by omega)
            refine ⟨?_, ?_⟩
            · show Sem.slotsGo g' (Sem.slotsGo g' (Sem.slotsGo g' n thn) els
                + (if termsGo g' thn && termsGo g' els then 0 else m.jTys.length)) [] = _
              rw [e1.1, e2.1, e1.2, e2.2, ht1, ht2, slotsGo_nil]; simp
            · rw [termsGo_cons, termsGo_cons]
              simp only [List.isEmpty_nil, if_true, pieceTerm, e1.2, e2.2]
          · obtain ⟨hle3, hg3⟩ := ih _ _ _ ps S' n' hk
            refine ⟨by omega, fun g hgf => ?_⟩
            obtain ⟨g', rfl⟩ : ∃ g', g = g' + 1 := ⟨g - 1, by omega⟩
            have e1 := hg1 g' (by omega); have e2 := hg2 g' (by omega)
            refine ⟨?_, hterm g' (by omega) (hg3 g' (by omega)).2
              (by simp only [pieceTerm, e1.2, e2.2])⟩
            show Sem.slotsGo g' (Sem.slotsGo g' (Sem.slotsGo g' n thn) els
                + (if termsGo g' thn && termsGo g' els then 0 else m.jTys.length)) ps = _
            rw [e1.1, e2.1, e1.2, e2.2, hnt]
            exact (hg3 g' (by omega)).1
      | dloop l body =>
          obtain ⟨Sb, nb, hok, hk⟩ := scGo_dloop_inv h
          obtain ⟨hle2, hg2⟩ := ih _ _ _ body Sb nb hok.hbody
          obtain ⟨hle3, hg3⟩ := ih _ _ _ ps S' n' hk
          refine ⟨by omega, fun g hgf => ?_⟩
          obtain ⟨g', rfl⟩ : ∃ g', g = g' + 1 := ⟨g - 1, by omega⟩
          refine ⟨?_, hterm g' (by omega) (hg3 g' (by omega)).2 rfl⟩
          show Sem.slotsGo g' (Sem.slotsGo g' (n + l.pTys.length) body
            + l.exitTys.length) ps = n'
          rw [(hg2 g' (by omega)).1, (hg3 g' (by omega)).1]
      | br d args =>
          obtain ⟨L, _, hps, _, _, _, _, hn⟩ := scGo_br_inv h
          subst hps; subst hn
          refine ⟨Nat.le_refl _, fun g hgf => ?_⟩
          obtain ⟨g', rfl⟩ : ∃ g', g = g' + 1 := ⟨g - 1, by omega⟩
          exact ⟨slotsGo_nil g' _, rfl⟩
      | cont d args =>
          obtain ⟨L, _, hps, _, _, _, hn⟩ := scGo_cont_inv h
          subst hps; subst hn
          refine ⟨Nat.le_refl _, fun g hgf => ?_⟩
          obtain ⟨g', rfl⟩ : ∃ g', g = g' + 1 := ⟨g - 1, by omega⟩
          exact ⟨slotsGo_nil g' _, rfl⟩

-- ---------------------------------------------------------------------------
-- A region that leaves on every path does not finish
-- ---------------------------------------------------------------------------

theorem runCode_cons (k : Nat) (cfg : Sem.Cfg) (Γ : Sem.Env) (w : Sem.World) (p : Piece)
    (ps : List Piece) :
    Sem.runCode (k + 1) cfg Γ w (p :: ps)
      = match Sem.runPiece k cfg Γ w p with
        | .stuck s => .stuck s
        | .brk d Γb vs w' => .brk d Γb vs w'
        | .cont d vs w' => .cont d vs w'
        | .ok Γ' w' => Sem.runCode k cfg Γ' w' ps := rfl

theorem runCode_nil (k : Nat) (cfg : Sem.Cfg) (Γ : Sem.Env) (w : Sem.World) :
    Sem.runCode (k + 1) cfg Γ w [] = .ok Γ w := rfl

/-- **What `termsGo` says, the interpreter agrees with**: a region it flags
    never finishes normally. So where the emitter left out a back edge or a join
    jump because `termsGo` said nothing reaches it, nothing does. -/
theorem termsGo_run : ∀ (f : Nat) (c : List Piece), termsGo f c = true →
    ∀ (k : Nat) (cfg : Sem.Cfg) (Γ : Sem.Env) (w : Sem.World) (Γ' : Sem.Env) (w' : Sem.World),
      Sem.runCode k cfg Γ w c ≠ .ok Γ' w' := by
  intro f
  induction f with
  | zero => intro c h; simp [termsGo] at h
  | succ f ih =>
    intro c h k cfg Γ w Γ' w' hrun
    cases c with
    | nil => simp [termsGo] at h
    | cons p ps =>
      cases k with
      | zero => simp [Sem.runCode] at hrun
      | succ k =>
        rw [runCode_cons] at hrun
        rw [termsGo_cons] at h
        split at h
        · rename_i hps
          rw [List.isEmpty_iff] at hps
          subst hps
          -- The piece itself never finishes.
          have hp : ∀ Γ1 w1, Sem.runPiece k cfg Γ w p ≠ .ok Γ1 w1 := by
            intro Γ1 w1 hpk
            cases k with
            | zero => simp [Sem.runPiece] at hpk
            | succ k =>
              cases p with
              | br d args =>
                  simp only [Sem.runPiece] at hpk
                  split at hpk <;> simp at hpk
              | cont d args =>
                  simp only [Sem.runPiece] at hpk
                  split at hpk <;> simp at hpk
              | ite m thn els thnR elsR =>
                  simp only [pieceTerm, Bool.and_eq_true] at h
                  simp only [Sem.runPiece] at hpk
                  split at hpk
                  · split at hpk
                    · simp at hpk
                    · simp at hpk
                    · simp at hpk
                    · rename_i Γa wa hra
                      split at hra
                      · exact ih thn h.1 k cfg _ w Γa wa (by simpa using hra)
                      · exact ih els h.2 k cfg _ w Γa wa (by simpa using hra)
                  · simp at hpk
              | straight _ => simp [pieceTerm] at h
              | loop _ _ _ => simp [pieceTerm] at h
              | dloop _ _ => simp [pieceTerm] at h
          split at hrun
          · simp at hrun
          · simp at hrun
          · simp at hrun
          · rename_i Γ1 w1 hpk; exact hp Γ1 w1 hpk
        · split at hrun
          · simp at hrun
          · simp at hrun
          · simp at hrun
          · exact ih ps h k cfg _ _ Γ' w' hrun

-- ---------------------------------------------------------------------------
-- The emitter, one state at a time
-- ---------------------------------------------------------------------------

theorem emitCode_zero (s : CS) (c : List Piece) : emitCode 0 s c = s := by
  rw [emitCode]

theorem emitCode_nil (f : Nat) (s : CS) : emitCode f s [] = s := by
  cases f <;> rw [emitCode]

theorem emitCode_cons (f : Nat) (s : CS) (p : Piece) (ps : List Piece) :
    emitCode (f + 1) s (p :: ps) = emitCode f (emitPiece f s p) ps := by
  rw [emitCode]

/-- `emitLoop`'s entry: reserve three ids, close the current block with a jump
    to the head, open the head with the carries as its parameters. -/
def loopHead (s : CS) (l : Loop) : CS :=
  { (({ s with nextBlk := s.nextBlk + 3 } : CS).close
        (.jump ⟨s.nextBlk⟩ (l.init.map s.get))).open' s.nextBlk l.pTys s.slots with
    slots := s.slots + l.pTys.length,
    labels := (s.nextBlk + 2, some s.nextBlk) :: s.labels }

/-- The head's test: to the exit with the exit values, or to the body with the
    carries, whichever way the flag says. -/
def loopBrif (h : Nat) (l : Loop) (sH : CS) (firstCarry : Nat) : Inst :=
  let carryVals := (List.range l.pTys.length).map (fun i => sH.get (firstCarry + i))
  let exitArgs := l.exitR.map sH.get
  let te := if l.exitOnTrue then (h + 2, exitArgs, h + 1, carryVals)
            else (h + 1, carryVals, h + 2, exitArgs)
  .brif (sH.get l.flag) ⟨te.1⟩ te.2.1 ⟨te.2.2.1⟩ te.2.2.2

def loopBodyStart (h : Nat) (l : Loop) (sH : CS) (firstCarry : Nat) : CS :=
  { (sH.close (loopBrif h l sH firstCarry)).open' (h + 1) l.pTys sH.slots with
    slots := sH.slots + l.pTys.length }

def loopBodyEnd (f h : Nat) (l : Loop) (body : List Piece) (sB : CS) : CS :=
  if termsGo f body then sB else sB.close (.jump ⟨h⟩ (l.cont.map sB.get))

def loopExit (h : Nat) (l : Loop) (outer : List (Nat × Option Nat)) (s8 : CS) : CS :=
  { s8.open' (h + 2) l.exitTys s8.slots with
    slots := s8.slots + l.exitTys.length, labels := outer }

theorem emitLoop_eq (f : Nat) (s : CS) (l : Loop) (pre body : List Piece) :
    emitLoop f s l pre body =
      loopExit s.nextBlk l s.labels
        (loopBodyEnd f s.nextBlk l body
          (emitCode f (loopBodyStart s.nextBlk l (emitCode f (loopHead s l) pre) s.slots)
            body)) := by
  rw [emitLoop]; rfl

/-- `emitDLoop`'s entry: the guard's branch, or a plain jump, into the body. -/
def dloopEntry (s : CS) (l : DLoop) : Inst :=
  let initVals := l.init.map s.get
  match l.guard with
  | some g =>
    let exit0 := l.exitIdx.map (fun i => (initVals[i]?).getD (⟨1000000⟩ : Val))
    let te0 :=
      if l.contOnTrue then (s.nextBlk, initVals, s.nextBlk + 1, exit0)
      else (s.nextBlk + 1, exit0, s.nextBlk, initVals)
    .brif (s.get g) ⟨te0.1⟩ te0.2.1 ⟨te0.2.2.1⟩ te0.2.2.2
  | none => .jump ⟨s.nextBlk⟩ initVals

def dloopHead (s : CS) (l : DLoop) : CS :=
  { (({ s with nextBlk := s.nextBlk + 2 } : CS).close (dloopEntry s l)).open'
        s.nextBlk l.pTys s.slots with
    slots := s.slots + l.pTys.length,
    labels := (s.nextBlk + 1, some s.nextBlk) :: s.labels }

def dloopBack (h : Nat) (l : DLoop) (sB : CS) : Inst :=
  let contVals := l.cont.map sB.get
  let exitN := l.exitIdx.map (fun i => (contVals[i]?).getD ⟨1000000⟩)
  let teN :=
    if l.contOnTrue then (h, contVals, h + 1, exitN)
    else (h + 1, exitN, h, contVals)
  .brif (sB.get l.flag) ⟨teN.1⟩ teN.2.1 ⟨teN.2.2.1⟩ teN.2.2.2

def dloopBodyEnd (f h : Nat) (l : DLoop) (body : List Piece) (sB : CS) : CS :=
  if termsGo f body then sB else sB.close (dloopBack h l sB)

def dloopExit (h : Nat) (l : DLoop) (outer : List (Nat × Option Nat)) (s8 : CS) : CS :=
  { s8.open' (h + 1) l.exitTys s8.slots with
    slots := s8.slots + l.exitTys.length, labels := outer }

theorem emitDLoop_eq (f : Nat) (s : CS) (l : DLoop) (body : List Piece) :
    emitDLoop f s l body =
      dloopExit s.nextBlk l s.labels
        (dloopBodyEnd f s.nextBlk l body (emitCode f (dloopHead s l) body)) := by
  rw [emitDLoop, dloopExit, dloopBodyEnd, dloopHead, dloopEntry]
  cases l.guard <;> rfl

def iteThen (s : CS) (m : IteMeta) : CS :=
  (({ s with nextBlk := s.nextBlk + 3 } : CS).close
      (.brif (s.get m.flag) ⟨s.nextBlk⟩ [] ⟨s.nextBlk + 1⟩ [])).open' s.nextBlk [] s.slots

def iteArmEnd (f h : Nat) (arm : List Piece) (rs : List R) (sA : CS) : CS :=
  if termsGo f arm then sA else sA.close (.jump ⟨h + 2⟩ (rs.map sA.get))

def iteElse (f h : Nat) (thn : List Piece) (thnR : List R) (sT : CS) : CS :=
  let s := iteArmEnd f h thn thnR sT
  s.open' (h + 1) [] s.slots

def iteJoin (f h : Nat) (m : IteMeta) (thn els : List Piece) (s : CS) : CS :=
  if termsGo f thn && termsGo f els then s
  else { s.open' (h + 2) m.jTys s.slots with slots := s.slots + m.jTys.length }

theorem emitIte_eq (f : Nat) (s : CS) (m : IteMeta) (thn els : List Piece) (thnR elsR : List R) :
    emitIte f s m thn els thnR elsR =
      iteJoin f s.nextBlk m thn els
        (iteArmEnd f s.nextBlk els elsR
          (emitCode f (iteElse f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)) els)) := by
  rw [emitIte]; rfl

-- ---------------------------------------------------------------------------
-- Block identity through emission
-- ---------------------------------------------------------------------------

/-- The ids of the finished blocks. -/
def ids (s : CS) : List Nat := s.done.map (·.ref.id)

theorem ids_close (s : CS) (t : Inst) : ids (s.close t) = ids s ++ [s.curRef] := by
  simp [ids, CS.close]

theorem open'_fields (s : CS) (ref : Nat) (tys : List ClifTy) (fs : Nat) :
    (s.open' ref tys fs).done = s.done ∧ (s.open' ref tys fs).nextBlk = s.nextBlk ∧
    (s.open' ref tys fs).curRef = ref ∧ (s.open' ref tys fs).cur = s.cur ∧
    (s.open' ref tys fs).curPars = parsOf s.nextVal tys ∧
    (s.open' ref tys fs).labels = s.labels ∧ (s.open' ref tys fs).slots = s.slots := by
  obtain ⟨h1, h2, h3⟩ := open'_blk s ref tys fs
  refine ⟨h1, h2, h3, ?_, open'_pars s ref tys fs, ?_, ?_⟩
  · simpa [CS.open'] using open'_go_cur tys { s with curRef := ref, curPars := [] } fs 0
  · have : ∀ (tys : List ClifTy) (st : CS) (i : Nat),
        (CS.open'.go fs st i tys).labels = st.labels ∧ (CS.open'.go fs st i tys).slots = st.slots := by
      intro tys; induction tys with
      | nil => intro st i; simp [CS.open'.go]
      | cons t ts ih => intro st i; simpa [CS.open'.go, CS.fresh] using ih _ (i + 1)
    simpa [CS.open'] using (this tys { s with curRef := ref, curPars := [] } 0).1
  · have : ∀ (tys : List ClifTy) (st : CS) (i : Nat),
        (CS.open'.go fs st i tys).slots = st.slots := by
      intro tys; induction tys with
      | nil => intro st i; simp [CS.open'.go]
      | cons t ts ih => intro st i; simpa [CS.open'.go, CS.fresh] using ih _ (i + 1)
    simpa [CS.open'] using this tys { s with curRef := ref, curPars := [] } 0

/-- Nothing finished repeats an id or runs ahead of the reservations, and the
    open block is not one of the finished ones. -/
structure Inv (s : CS) : Prop where
  nodup : (ids s).Nodup
  lt    : ∀ x ∈ ids s, x < s.nextBlk
  cur   : s.curRef < s.nextBlk
  open_ : s.curRef ∉ ids s

/-- An id emission from `s` may have finished: one already finished, the block
    open at `s`, or one reserved after `s`. -/
def Fresh (s : CS) (x : Nat) : Prop := x ∈ ids s ∨ x = s.curRef ∨ s.nextBlk ≤ x

/-- What emitting a region from `s` to `t` does to block identity. `tm` says the
    region left on every path, in which case its last block is closed and the
    caller opens a fresh one next. -/
structure Out (s t : CS) (tm : Bool) : Prop where
  grows : Grows s t
  nodup : (ids t).Nodup
  lt    : ∀ x ∈ ids t, x < t.nextBlk
  cur   : t.curRef < t.nextBlk
  fresh : ∀ x ∈ ids t, Fresh s x
  curF  : t.curRef = s.curRef ∨ s.nextBlk ≤ t.curRef
  open_ : tm = false → t.curRef ∉ ids t

theorem Out.inv {s t : CS} (h : Out s t false) : Inv t :=
  ⟨h.nodup, h.lt, h.cur, h.open_ rfl⟩

theorem Out.refl {s : CS} (h : Inv s) : Out s s false :=
  ⟨Grows.refl s, h.nodup, h.lt, h.cur, fun _ hx => .inl hx, .inl rfl, fun _ => h.open_⟩

theorem Out.trans {s t u : CS} {tm : Bool} (h1 : Out s t false) (h2 : Out t u tm) :
    Out s u tm := by
  refine ⟨h1.grows.trans h2.grows, h2.nodup, h2.lt, h2.cur, ?_, ?_, h2.open_⟩
  · intro x hx
    rcases h2.fresh x hx with h | h | h
    · exact h1.fresh x h
    · subst h
      rcases h1.curF with h' | h'
      · exact .inr (.inl h')
      · exact .inr (.inr h')
    · exact .inr (.inr (Nat.le_trans h1.grows.1 h))
  · rcases h2.curF with h | h
    · rw [h]; exact h1.curF
    · exact .inr (Nat.le_trans h1.grows.1 h)

/-- Closing the open block. -/
theorem Out.close {s t : CS} (h : Out s t false) (x : Inst) : Out s (t.close x) true := by
  refine ⟨h.grows.trans (Grows.close t x), ?_, ?_, ?_, ?_, ?_, fun h => by cases h⟩
  · rw [ids_close, List.nodup_append]
    refine ⟨h.nodup, by simp, fun y hy z hz => ?_⟩
    simp only [List.mem_singleton] at hz
    subst hz
    exact fun heq => h.open_ rfl (heq ▸ hy)
  · rw [ids_close]; intro y hy
    simp only [List.mem_append, List.mem_singleton] at hy
    rcases hy with hy | hy
    · exact h.lt y hy
    · subst hy; exact h.cur
  · exact h.cur
  · rw [ids_close]; intro y hy
    simp only [List.mem_append, List.mem_singleton] at hy
    rcases hy with hy | hy
    · exact h.fresh y hy
    · subst hy
      rcases h.curF with h' | h'
      · exact .inr (.inl h')
      · exact .inr (.inr h')
  · exact h.curF

/-- Reserving more ids. -/
theorem Out.bump {s t : CS} {tm : Bool} (h : Out s t tm) (k : Nat) :
    Out s { t with nextBlk := t.nextBlk + k } tm :=
  ⟨⟨Nat.le_trans h.grows.1 (by simp), h.grows.2⟩, h.nodup,
   fun x hx => by have := h.lt x hx; simp only; omega, by have := h.cur; simp only; omega,
   h.fresh, h.curF, h.open_⟩

/-- A state that differs only in what block identity does not see. -/
theorem Out.same {s t u : CS} {tm : Bool} (h : Out s t tm) (hd : u.done = t.done)
    (hn : u.nextBlk = t.nextBlk) (hc : u.curRef = t.curRef) : Out s u tm := by
  have hi : ids u = ids t := by simp [ids, hd]
  refine ⟨⟨hn ▸ h.grows.1, hd ▸ h.grows.2⟩, hi ▸ h.nodup, ?_, hn ▸ hc ▸ h.cur, ?_,
    hc ▸ h.curF, fun e => hi ▸ hc ▸ h.open_ e⟩
  · rw [hi, hn]; exact h.lt
  · rw [hi]; exact h.fresh

/-- Opening a block at a reserved id that nothing has finished. -/
theorem Out.reopen {s t u : CS} {tm : Bool} (h : Out s t tm) (hd : u.done = t.done)
    (hn : u.nextBlk = t.nextBlk) (hr : u.curRef ∉ ids t) (hrlt : u.curRef < t.nextBlk)
    (hrs : s.nextBlk ≤ u.curRef) : Out s u false := by
  have hi : ids u = ids t := by simp [ids, hd]
  refine ⟨⟨hn ▸ h.grows.1, hd ▸ h.grows.2⟩, hi ▸ h.nodup, ?_, hn ▸ hrlt, ?_, .inr hrs,
    fun _ => hi ▸ hr⟩
  · rw [hi, hn]; exact h.lt
  · rw [hi]; exact h.fresh

-- ---------------------------------------------------------------------------
-- Which block is open
-- ---------------------------------------------------------------------------

/-- `t` is still in `s`'s block, having only added instructions to it. -/
def ExtO (s t : CS) : Prop :=
  t.curRef = s.curRef ∧ t.curPars = s.curPars ∧ ∃ X, t.cur = X ++ s.cur

/-- `s`'s block is finished by `t`, starting with what `s` had put in it. -/
def ExtC (s t : CS) : Prop :=
  ∃ b ∈ t.done, b.ref.id = s.curRef ∧ b.params = s.curPars ∧ ∃ R, b.insts = s.cur.reverse ++ R

theorem ExtO.refl (s : CS) : ExtO s s := ⟨rfl, rfl, [], by simp⟩

theorem ExtO.trans {s t u : CS} (h1 : ExtO s t) (h2 : ExtO t u) : ExtO s u := by
  obtain ⟨a1, b1, X1, c1⟩ := h1
  obtain ⟨a2, b2, X2, c2⟩ := h2
  exact ⟨a2.trans a1, b2.trans b1, X2 ++ X1, by rw [c2, c1, List.append_assoc]⟩

theorem ExtO.extC {s t u : CS} (h1 : ExtO s t) (h2 : ExtC t u) : ExtC s u := by
  obtain ⟨a1, b1, X1, c1⟩ := h1
  obtain ⟨b, hb, hid, hpar, R, hins⟩ := h2
  exact ⟨b, hb, hid.trans a1, hpar.trans b1, X1.reverse ++ R, by
    rw [hins, c1]; simp [List.reverse_append, List.append_assoc]⟩

theorem ExtC.grow {s t u : CS} (h : ExtC s t) (hg : t.done <+: u.done) : ExtC s u := by
  obtain ⟨b, hb, rest⟩ := h
  exact ⟨b, hg.subset hb, rest⟩

theorem ExtC.close (s : CS) (x : Inst) : ExtC s (s.close x) :=
  ⟨{ ref := ⟨s.curRef⟩, params := s.curPars, insts := (x :: s.cur).reverse },
   by simp [CS.close], rfl, rfl, [x], by simp⟩

theorem ExtO.emitStmts (s : CS) (ss : List Stmt) : ExtO s (emitStmts s ss) := by
  obtain ⟨_, _, h3⟩ := emitStmts_blk ss s
  exact ⟨h3, emitStmts_curPars ss s, _, emitStmts_cur ss s⟩

theorem aligned_congr {t u : CS} {n : Nat} (h : Aligned t n) (hv : u.nextVal = t.nextVal)
    (hs : u.slots = t.slots) (he : u.env = t.env) : Aligned u n := by
  obtain ⟨a, b, c⟩ := h
  exact ⟨hv ▸ a, hs ▸ b, fun i hi => he ▸ c i hi⟩

/-- Opening a block at the slot count, and counting its parameters in. -/
theorem aligned_open {s : CS} {n : Nat} (h : Aligned s n) (ref : Nat) (tys : List ClifTy)
    (L : List (Nat × Option Nat)) :
    Aligned { s.open' ref tys s.slots with slots := s.slots + tys.length, labels := L }
      (n + tys.length) := by
  have hs : s.slots = n := h.2.1
  have := open'_aligned s n h ref tys
  rw [hs]; exact this

theorem aligned_open0 {s : CS} {n : Nat} (h : Aligned s n) (ref : Nat) :
    Aligned (s.open' ref [] s.slots) n := by
  have := aligned_open h ref [] s.labels
  simp only [List.length_nil, Nat.add_zero] at this
  exact aligned_congr this rfl (by simp [(open'_fields s ref [] s.slots).2.2.2.2.2.2]) rfl

/-- An id that nothing before a region finished, that is not the region's first
    block, and that was reserved before it began, is not finished by it
    either --- nor is it the block the region ends in. -/
theorem out_avoid {C D : CS} {tm : Bool} (hout : Out C D tm) {r : Nat}
    (h1 : ∀ x ∈ ids C, x ≠ r) (h2 : C.curRef ≠ r) (h3 : r < C.nextBlk) :
    (∀ x ∈ ids D, x ≠ r) ∧ D.curRef ≠ r := by
  refine ⟨fun x hx => ?_, ?_⟩
  · rcases hout.fresh x hx with h | h | h
    · exact h1 x h
    · rw [h]; exact h2
    · omega
  · rcases hout.curF with h | h
    · rw [h]; exact h2
    · omega

theorem avoid_close {D : CS} {r : Nat} (h1 : ∀ x ∈ ids D, x ≠ r) (h2 : D.curRef ≠ r)
    (y : Inst) : ∀ x ∈ ids (D.close y), x ≠ r := by
  rw [ids_close]; intro x hx
  simp only [List.mem_append, List.mem_singleton] at hx
  rcases hx with hx | hx
  · exact h1 x hx
  · rw [hx]; exact h2

theorem loopHead_facts (s : CS) (l : Loop) :
    (loopHead s l).done = s.done ++ [⟨⟨s.curRef⟩, s.curPars,
        (Inst.jump ⟨s.nextBlk⟩ (l.init.map s.get) :: s.cur).reverse⟩] ∧
    (loopHead s l).nextBlk = s.nextBlk + 3 ∧ (loopHead s l).curRef = s.nextBlk ∧
    (loopHead s l).cur = [] ∧ (loopHead s l).curPars = parsOf s.nextVal l.pTys ∧
    (loopHead s l).labels = (s.nextBlk + 2, some s.nextBlk) :: s.labels ∧
    (loopHead s l).slots = s.slots + l.pTys.length := by
  obtain ⟨d, nb, cr, cu, cp, _, _⟩ :=
    open'_fields (({ s with nextBlk := s.nextBlk + 3 } : CS).close
      (.jump ⟨s.nextBlk⟩ (l.init.map s.get))) s.nextBlk l.pTys s.slots
  refine ⟨?_, ?_, ?_, ?_, ?_, rfl, rfl⟩
  · show (CS.open' _ _ _ _).done = _; rw [d]; rfl
  · show (CS.open' _ _ _ _).nextBlk = _; rw [nb]; rfl
  · show (CS.open' _ _ _ _).curRef = _; rw [cr]
  · show (CS.open' _ _ _ _).cur = _; rw [cu]; rfl
  · show (CS.open' _ _ _ _).curPars = _; rw [cp]; rfl

theorem loopBodyStart_facts (h : Nat) (l : Loop) (sH : CS) (fc : Nat) :
    (loopBodyStart h l sH fc).done = sH.done ++ [⟨⟨sH.curRef⟩, sH.curPars,
        (loopBrif h l sH fc :: sH.cur).reverse⟩] ∧
    (loopBodyStart h l sH fc).nextBlk = sH.nextBlk ∧ (loopBodyStart h l sH fc).curRef = h + 1 ∧
    (loopBodyStart h l sH fc).cur = [] ∧
    (loopBodyStart h l sH fc).curPars = parsOf sH.nextVal l.pTys ∧
    (loopBodyStart h l sH fc).labels = sH.labels ∧
    (loopBodyStart h l sH fc).slots = sH.slots + l.pTys.length := by
  obtain ⟨d, nb, cr, cu, cp, lab, _⟩ :=
    open'_fields (sH.close (loopBrif h l sH fc)) (h + 1) l.pTys sH.slots
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, rfl⟩
  · show (CS.open' _ _ _ _).done = _; rw [d]; rfl
  · show (CS.open' _ _ _ _).nextBlk = _; rw [nb]; rfl
  · show (CS.open' _ _ _ _).curRef = _; rw [cr]
  · show (CS.open' _ _ _ _).cur = _; rw [cu]; rfl
  · show (CS.open' _ _ _ _).curPars = _; rw [cp]; rfl
  · show (CS.open' _ _ _ _).labels = _; rw [lab]; rfl

theorem loopExit_facts (h : Nat) (l : Loop) (outer : List (Nat × Option Nat)) (s8 : CS) :
    (loopExit h l outer s8).done = s8.done ∧ (loopExit h l outer s8).nextBlk = s8.nextBlk ∧
    (loopExit h l outer s8).curRef = h + 2 ∧ (loopExit h l outer s8).cur = s8.cur ∧
    (loopExit h l outer s8).curPars = parsOf s8.nextVal l.exitTys ∧
    (loopExit h l outer s8).labels = outer ∧
    (loopExit h l outer s8).slots = s8.slots + l.exitTys.length := by
  obtain ⟨d, nb, cr, cu, cp, _, _⟩ := open'_fields s8 (h + 2) l.exitTys s8.slots
  exact ⟨d, nb, cr, cu, cp, rfl, rfl⟩

theorem dloopHead_facts (s : CS) (l : DLoop) :
    (dloopHead s l).done = s.done ++ [⟨⟨s.curRef⟩, s.curPars,
        (dloopEntry s l :: s.cur).reverse⟩] ∧
    (dloopHead s l).nextBlk = s.nextBlk + 2 ∧ (dloopHead s l).curRef = s.nextBlk ∧
    (dloopHead s l).cur = [] ∧ (dloopHead s l).curPars = parsOf s.nextVal l.pTys ∧
    (dloopHead s l).labels = (s.nextBlk + 1, some s.nextBlk) :: s.labels ∧
    (dloopHead s l).slots = s.slots + l.pTys.length := by
  obtain ⟨d, nb, cr, cu, cp, _, _⟩ :=
    open'_fields (({ s with nextBlk := s.nextBlk + 2 } : CS).close (dloopEntry s l))
      s.nextBlk l.pTys s.slots
  refine ⟨?_, ?_, ?_, ?_, ?_, rfl, rfl⟩
  · show (CS.open' _ _ _ _).done = _; rw [d]; rfl
  · show (CS.open' _ _ _ _).nextBlk = _; rw [nb]; rfl
  · show (CS.open' _ _ _ _).curRef = _; rw [cr]
  · show (CS.open' _ _ _ _).cur = _; rw [cu]; rfl
  · show (CS.open' _ _ _ _).curPars = _; rw [cp]; rfl

theorem dloopExit_facts (h : Nat) (l : DLoop) (outer : List (Nat × Option Nat)) (s8 : CS) :
    (dloopExit h l outer s8).done = s8.done ∧ (dloopExit h l outer s8).nextBlk = s8.nextBlk ∧
    (dloopExit h l outer s8).curRef = h + 1 ∧ (dloopExit h l outer s8).cur = s8.cur ∧
    (dloopExit h l outer s8).curPars = parsOf s8.nextVal l.exitTys ∧
    (dloopExit h l outer s8).labels = outer ∧
    (dloopExit h l outer s8).slots = s8.slots + l.exitTys.length := by
  obtain ⟨d, nb, cr, cu, cp, _, _⟩ := open'_fields s8 (h + 1) l.exitTys s8.slots
  exact ⟨d, nb, cr, cu, cp, rfl, rfl⟩

theorem iteThen_facts (s : CS) (m : IteMeta) :
    (iteThen s m).done = s.done ++ [⟨⟨s.curRef⟩, s.curPars,
        (Inst.brif (s.get m.flag) ⟨s.nextBlk⟩ [] ⟨s.nextBlk + 1⟩ [] :: s.cur).reverse⟩] ∧
    (iteThen s m).nextBlk = s.nextBlk + 3 ∧ (iteThen s m).curRef = s.nextBlk ∧
    (iteThen s m).cur = [] ∧ (iteThen s m).curPars = [] ∧
    (iteThen s m).labels = s.labels ∧ (iteThen s m).slots = s.slots := by
  obtain ⟨d, nb, cr, cu, cp, lab, sl⟩ :=
    open'_fields (({ s with nextBlk := s.nextBlk + 3 } : CS).close
      (.brif (s.get m.flag) ⟨s.nextBlk⟩ [] ⟨s.nextBlk + 1⟩ [])) s.nextBlk [] s.slots
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · show (CS.open' _ _ _ _).done = _; rw [d]; rfl
  · show (CS.open' _ _ _ _).nextBlk = _; rw [nb]; rfl
  · show (CS.open' _ _ _ _).curRef = _; rw [cr]
  · show (CS.open' _ _ _ _).cur = _; rw [cu]; rfl
  · show (CS.open' _ _ _ _).curPars = _; rw [cp]; rfl
  · show (CS.open' _ _ _ _).labels = _; rw [lab]; rfl
  · show (CS.open' _ _ _ _).slots = _; rw [sl]; rfl

theorem iteElse_facts (f h : Nat) (thn : List Piece) (thnR : List R) (sT : CS) :
    (iteElse f h thn thnR sT).done = (iteArmEnd f h thn thnR sT).done ∧
    (iteElse f h thn thnR sT).nextBlk = sT.nextBlk ∧ (iteElse f h thn thnR sT).curRef = h + 1 ∧
    (iteElse f h thn thnR sT).cur = (iteArmEnd f h thn thnR sT).cur ∧
    (iteElse f h thn thnR sT).curPars = [] ∧
    (iteElse f h thn thnR sT).labels = sT.labels ∧ (iteElse f h thn thnR sT).slots = sT.slots := by
  obtain ⟨d, nb, cr, cu, cp, lab, sl⟩ :=
    open'_fields (iteArmEnd f h thn thnR sT) (h + 1) [] (iteArmEnd f h thn thnR sT).slots
  have e1 : (iteArmEnd f h thn thnR sT).nextBlk = sT.nextBlk := by
    unfold iteArmEnd; split <;> rfl
  have e2 : (iteArmEnd f h thn thnR sT).labels = sT.labels := by
    unfold iteArmEnd; split <;> rfl
  have e3 : (iteArmEnd f h thn thnR sT).slots = sT.slots := by
    unfold iteArmEnd; split <;> rfl
  refine ⟨d, nb.trans e1, cr, cu, by rw [iteElse]; simp only [cp]; rfl, lab.trans e2,
    sl.trans e3⟩

theorem aligned_open_nl {s : CS} {n : Nat} (h : Aligned s n) (ref : Nat) (tys : List ClifTy) :
    Aligned { s.open' ref tys s.slots with slots := s.slots + tys.length } (n + tys.length) := by
  have hs : s.slots = n := h.2.1
  have := open'_aligned s n h ref tys
  rw [hs]; exact this

theorem emitStmt_labels (s : CS) (st : Stmt) : (emitStmt s st).labels = s.labels := by
  cases st <;> rfl

theorem emitStmts_labels : ∀ (ss : List Stmt) (s : CS), (emitStmts s ss).labels = s.labels := by
  intro ss; induction ss with
  | nil => intro s; rfl
  | cons st ss ih =>
      intro s; show (emitStmts (emitStmt s st) ss).labels = _
      rw [ih, emitStmt_labels]

theorem ids_of_done {s t : CS} (h : t.done = s.done ++ [b]) : ids t = ids s ++ [b.ref.id] := by
  simp [ids, h]

/-- What emission does to the emitter's state, for a region that passed the
    check: the slot numbering stays the identity, the labels come back, block
    identity is kept (`Out`), and the block it started in is either still open
    or finished starting with what it held. -/
def StructP (f : Nat) : Prop :=
  ∀ (lb : List SLbl) (S : Scope) (n : Nat) (c : List Piece) (S' : Scope) (n' : Nat) (s : CS),
    scGo f lb S n c = some (S', n') → Aligned s n → Inv s →
    Aligned (emitCode f s c) n' ∧ (emitCode f s c).labels = s.labels ∧
    Out s (emitCode f s c) (termsGo f c) ∧
    (ExtO s (emitCode f s c) ∨ ExtC s (emitCode f s c)) ∧
    (termsGo f c = true → ExtC s (emitCode f s c) ∧ (emitCode f s c).cur = [])

theorem struct_loop (f : Nat) (ih : StructP f) {lb : List SLbl} {S : Scope} {n : Nat}
    {l : Loop} {pre body : List Piece} {Sp : Scope} {np : Nat} {Sb : Scope} {nb : Nat}
    (hok : LoopOk f lb S n l pre body Sp np Sb nb) (s : CS) (ha : Aligned s n) (hi : Inv s) :
    Aligned (emitPiece f s (.loop l pre body)) (nb + l.exitTys.length) ∧
    (emitPiece f s (.loop l pre body)).labels = s.labels ∧
    Out s (emitPiece f s (.loop l pre body)) false ∧
    ExtC s (emitPiece f s (.loop l pre body)) := by
  rw [show emitPiece f s (.loop l pre body) = emitLoop f s l pre body by rw [emitPiece],
      emitLoop_eq]
  obtain ⟨hCd, hCn, hCr, _, _, _, _⟩ := loopHead_facts s l
  -- the head
  have hCa : Aligned (loopHead s l) (n + l.pTys.length) :=
    aligned_open (aligned_congr (u := (({ s with nextBlk := s.nextBlk + 3 } : CS).close
      (.jump ⟨s.nextBlk⟩ (l.init.map s.get)))) ha rfl rfl rfl) s.nextBlk l.pTys _
  have hCids : ∀ x ∈ ids (loopHead s l), x < s.nextBlk := by
    rw [ids_of_done hCd]; intro x hx
    simp only [List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | hx
    · exact hi.lt x hx
    · rw [hx]; exact hi.cur
  have hCo : Out s (loopHead s l) false := by
    have h0 := ((Out.refl hi).bump 3).close (.jump ⟨s.nextBlk⟩ (l.init.map s.get))
    refine h0.reopen ?_ ?_ ?_ ?_ ?_
    · rw [hCd]; rfl
    · rw [hCn]; rfl
    · rw [hCr]; intro hm
      rw [ids_close] at hm
      simp only [List.mem_append, List.mem_singleton] at hm
      rcases hm with hm | hm
      · have := hi.lt _ hm; omega
      · have := hi.cur; omega
    · rw [hCr]; show s.nextBlk < s.nextBlk + 3; omega
    · rw [hCr]; exact Nat.le_refl _
  have hCav : ∀ r, s.nextBlk + 1 ≤ r → r < s.nextBlk + 3 →
      (∀ x ∈ ids (loopHead s l), x ≠ r) ∧ (loopHead s l).curRef ≠ r ∧
      r < (loopHead s l).nextBlk := by
    intro r h1 h2
    refine ⟨fun x hx => ?_, by rw [hCr]; omega, by rw [hCn]; omega⟩
    have := hCids x hx; omega
  -- the prefix
  obtain ⟨hDa, hDl, hDo, _, _⟩ := ih _ _ _ pre Sp np (loopHead s l) hok.hpre hCa hCo.inv
  rw [hok.hpreT] at hDo
  generalize hD : emitCode f (loopHead s l) pre = D at hDa hDl hDo
  have hDav1 := out_avoid hDo (hCav _ (Nat.le_refl _) (by omega)).1
    (hCav _ (Nat.le_refl _) (by omega)).2.1 (hCav _ (Nat.le_refl _) (by omega)).2.2
  have hDav2 := out_avoid hDo (hCav (s.nextBlk + 2) (by omega) (by omega)).1
    (hCav (s.nextBlk + 2) (by omega) (by omega)).2.1 (hCav (s.nextBlk + 2) (by omega) (by omega)).2.2
  have hDnb : s.nextBlk + 3 ≤ D.nextBlk := hCn ▸ hDo.grows.1
  -- the body block
  obtain ⟨hFd, hFn, hFr, _, _, _, _⟩ := loopBodyStart_facts s.nextBlk l D s.slots
  have hFa : Aligned (loopBodyStart s.nextBlk l D s.slots) (np + l.pTys.length) :=
    aligned_open_nl (aligned_congr (u := D.close (loopBrif s.nextBlk l D s.slots)) hDa rfl rfl rfl)
      (s.nextBlk + 1) l.pTys
  have hCFo : Out s (loopBodyStart s.nextBlk l D s.slots) false := by
    refine ((hCo.trans hDo).close (loopBrif s.nextBlk l D s.slots)).reopen ?_ ?_ ?_ ?_ ?_
    · rw [hFd]; rfl
    · rw [hFn]; rfl
    · rw [hFr]; intro hm
      exact avoid_close hDav1.1 hDav1.2 _ _ hm rfl
    · rw [hFr]; show s.nextBlk + 1 < D.nextBlk; omega
    · rw [hFr]; omega
  have hFav2 : (∀ x ∈ ids (loopBodyStart s.nextBlk l D s.slots), x ≠ s.nextBlk + 2) ∧
      (loopBodyStart s.nextBlk l D s.slots).curRef ≠ s.nextBlk + 2 ∧
      s.nextBlk + 2 < (loopBodyStart s.nextBlk l D s.slots).nextBlk := by
    refine ⟨?_, ?_, ?_⟩
    · intro x hx
      rw [ids_of_done hFd] at hx
      exact avoid_close hDav2.1 hDav2.2 (loopBrif s.nextBlk l D s.slots) x
        (by rw [ids_close]; exact hx)
    · rw [hFr]; omega
    · rw [hFn]; omega
  -- the body
  obtain ⟨hGa, hGl, hGo, _, hGt⟩ :=
    ih _ _ _ body Sb nb (loopBodyStart s.nextBlk l D s.slots) hok.hbody hFa hCFo.inv
  generalize hG : emitCode f (loopBodyStart s.nextBlk l D s.slots) body = G at hGa hGl hGo hGt
  have hGav := out_avoid hGo hFav2.1 hFav2.2.1 hFav2.2.2
  have hGnb : s.nextBlk + 3 ≤ G.nextBlk := by
    have := hGo.grows.1; rw [hFn] at this; omega
  -- the back edge, and the exit
  have hCHo : Out s (loopBodyEnd f s.nextBlk l body G) true ∧
      G.done <+: (loopBodyEnd f s.nextBlk l body G).done ∧
      (∀ x ∈ ids (loopBodyEnd f s.nextBlk l body G), x ≠ s.nextBlk + 2) ∧
      (loopBodyEnd f s.nextBlk l body G).nextBlk = G.nextBlk ∧
      (loopBodyEnd f s.nextBlk l body G).slots = G.slots ∧
      (loopBodyEnd f s.nextBlk l body G).nextVal = G.nextVal ∧
      (loopBodyEnd f s.nextBlk l body G).env = G.env := by
    unfold loopBodyEnd
    split
    · rename_i htb
      rw [htb] at hGo
      exact ⟨hCFo.trans hGo, List.prefix_refl _, hGav.1, rfl, rfl, rfl, rfl⟩
    · rename_i htb
      simp only [Bool.not_eq_true] at htb
      rw [htb] at hGo
      exact ⟨(hCFo.trans hGo).close _, close_done_prefix G _, avoid_close hGav.1 hGav.2 _,
        rfl, rfl, rfl, rfl⟩
  obtain ⟨hHo, hGH, hHav, hHn, hHs, hHv, hHe⟩ := hCHo
  generalize hH : loopBodyEnd f s.nextBlk l body G = H at hHo hGH hHav hHn hHs hHv hHe
  obtain ⟨hId, hIn, hIr, _, _, hIl, _⟩ := loopExit_facts s.nextBlk l s.labels H
  have hIo : Out s (loopExit s.nextBlk l s.labels H) false := by
    refine hHo.reopen hId hIn ?_ ?_ ?_
    · rw [hIr]; intro hm; exact hHav _ hm rfl
    · rw [hIr, hHn]; omega
    · rw [hIr]; omega
  refine ⟨?_, hIl, hIo, ?_⟩
  · have hHa : Aligned H nb := aligned_congr hGa hHv hHs hHe
    exact aligned_open hHa (s.nextBlk + 2) l.exitTys s.labels
  · refine (ExtC.close ({ s with nextBlk := s.nextBlk + 3 } : CS)
      (.jump ⟨s.nextBlk⟩ (l.init.map s.get))).grow ?_
    have hCI : (loopHead s l).done <+: (loopExit s.nextBlk l s.labels H).done := by
      rw [hId]
      refine hDo.grows.2.trans (List.IsPrefix.trans ?_ (hGo.grows.2.trans hGH))
      rw [hFd]; exact List.prefix_append _ _
    rw [hCd] at hCI
    exact hCI

theorem iteArmEnd_facts (f h : Nat) (arm : List Piece) (rs : List R) (sA : CS) :
    (iteArmEnd f h arm rs sA).nextBlk = sA.nextBlk ∧ (iteArmEnd f h arm rs sA).labels = sA.labels ∧
    (iteArmEnd f h arm rs sA).slots = sA.slots ∧ (iteArmEnd f h arm rs sA).nextVal = sA.nextVal ∧
    (iteArmEnd f h arm rs sA).env = sA.env ∧ sA.done <+: (iteArmEnd f h arm rs sA).done := by
  unfold iteArmEnd; split
  · exact ⟨rfl, rfl, rfl, rfl, rfl, List.prefix_refl _⟩
  · exact ⟨rfl, rfl, rfl, rfl, rfl, close_done_prefix _ _⟩

/-- An arm's end, as block identity sees it: closed either way, and the ids it
    avoided it still avoids. -/
theorem iteArmEnd_out {s sA : CS} {f h : Nat} {arm : List Piece} {rs : List R}
    (ho : Out s sA (termsGo f arm)) (hcur : termsGo f arm = true → sA.cur = [])
    {r : Nat} (hav : (∀ x ∈ ids sA, x ≠ r) ∧ sA.curRef ≠ r) :
    Out s (iteArmEnd f h arm rs sA) true ∧ (iteArmEnd f h arm rs sA).cur = [] ∧
    ∀ x ∈ ids (iteArmEnd f h arm rs sA), x ≠ r := by
  unfold iteArmEnd; split
  · rename_i ht; rw [ht] at ho; exact ⟨ho, hcur ht, hav.1⟩
  · rename_i ht; simp only [Bool.not_eq_true] at ht; rw [ht] at ho
    exact ⟨ho.close _, rfl, avoid_close hav.1 hav.2 _⟩

theorem struct_ite (f : Nat) (ih : StructP f) {lb : List SLbl} {S : Scope} {n : Nat}
    {m : IteMeta} {thn els : List Piece} {thnR elsR : List R} {St : Scope} {nt : Nat}
    {Se : Scope} {ne : Nat}
    (hok : IteOk f lb S n m thn els thnR elsR St nt Se ne) (s : CS) (ha : Aligned s n)
    (hi : Inv s) :
    Aligned (emitPiece f s (.ite m thn els thnR elsR))
      (if termsGo f thn && termsGo f els then ne else ne + m.jTys.length) ∧
    (emitPiece f s (.ite m thn els thnR elsR)).labels = s.labels ∧
    Out s (emitPiece f s (.ite m thn els thnR elsR)) (termsGo f thn && termsGo f els) ∧
    ExtC s (emitPiece f s (.ite m thn els thnR elsR)) ∧
    ((termsGo f thn && termsGo f els) = true →
      (emitPiece f s (.ite m thn els thnR elsR)).cur = []) := by
  rw [show emitPiece f s (.ite m thn els thnR elsR) = emitIte f s m thn els thnR elsR by
        rw [emitPiece], emitIte_eq]
  obtain ⟨hCd, hCn, hCr, _, _, hCl, _⟩ := iteThen_facts s m
  have hCa : Aligned (iteThen s m) n :=
    aligned_open0 (aligned_congr (u := (({ s with nextBlk := s.nextBlk + 3 } : CS).close
      (.brif (s.get m.flag) ⟨s.nextBlk⟩ [] ⟨s.nextBlk + 1⟩ []))) ha rfl rfl rfl) s.nextBlk
  have hCids : ∀ x ∈ ids (iteThen s m), x < s.nextBlk := by
    rw [ids_of_done hCd]; intro x hx
    simp only [List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | hx
    · exact hi.lt x hx
    · rw [hx]; exact hi.cur
  have hCo : Out s (iteThen s m) false := by
    have h0 := ((Out.refl hi).bump 3).close
      (.brif (s.get m.flag) ⟨s.nextBlk⟩ [] ⟨s.nextBlk + 1⟩ [])
    refine h0.reopen ?_ ?_ ?_ ?_ ?_
    · rw [hCd]; rfl
    · rw [hCn]; rfl
    · rw [hCr]; intro hm
      rw [ids_close] at hm
      simp only [List.mem_append, List.mem_singleton] at hm
      rcases hm with hm | hm
      · have := hi.lt _ hm; omega
      · have := hi.cur; omega
    · rw [hCr]; show s.nextBlk < s.nextBlk + 3; omega
    · rw [hCr]; exact Nat.le_refl _
  have hCav : ∀ r, s.nextBlk + 1 ≤ r → r < s.nextBlk + 3 →
      (∀ x ∈ ids (iteThen s m), x ≠ r) ∧ (iteThen s m).curRef ≠ r ∧
      r < (iteThen s m).nextBlk := by
    intro r h1 h2
    refine ⟨fun x hx => ?_, by rw [hCr]; omega, by rw [hCn]; omega⟩
    have := hCids x hx; omega
  -- the then arm
  obtain ⟨hDa, hDl, hDo, _, hDt⟩ := ih _ _ _ thn St nt (iteThen s m) hok.hthn hCa hCo.inv
  generalize hD : emitCode f (iteThen s m) thn = D at hDa hDl hDo hDt
  have hDav1 := out_avoid hDo (hCav (s.nextBlk + 1) (by omega) (by omega)).1
    (hCav (s.nextBlk + 1) (by omega) (by omega)).2.1 (hCav (s.nextBlk + 1) (by omega) (by omega)).2.2
  have hDav2 := out_avoid hDo (hCav (s.nextBlk + 2) (by omega) (by omega)).1
    (hCav (s.nextBlk + 2) (by omega) (by omega)).2.1 (hCav (s.nextBlk + 2) (by omega) (by omega)).2.2
  have hDnb : s.nextBlk + 3 ≤ D.nextBlk := hCn ▸ hDo.grows.1
  obtain ⟨hEo1, hEc1, hEav1⟩ := iteArmEnd_out (h := s.nextBlk) (rs := thnR) (hCo.trans hDo) (fun h => (hDt h).2) hDav1
  obtain ⟨_, _, hEav2⟩ := iteArmEnd_out (h := s.nextBlk) (rs := thnR) (hCo.trans hDo) (fun h => (hDt h).2) hDav2
  obtain ⟨hEn, hEl, hEs, hEv, hEe, hDE⟩ := iteArmEnd_facts f s.nextBlk thn thnR D
  -- the else arm
  obtain ⟨hFd, hFn, hFr, _, _, hFl, hFs⟩ := iteElse_facts f s.nextBlk thn thnR D
  have hFa : Aligned (iteElse f s.nextBlk thn thnR D) nt := by
    have hEa : Aligned (iteArmEnd f s.nextBlk thn thnR D) nt := aligned_congr hDa hEv hEs hEe
    exact aligned_open0 hEa (s.nextBlk + 1)
  have hFo : Out s (iteElse f s.nextBlk thn thnR D) false := by
    refine hEo1.reopen hFd (by rw [hFn, hEn]) ?_ ?_ ?_
    · rw [hFr]; intro hm; exact hEav1 _ hm rfl
    · rw [hFr, hEn]; omega
    · rw [hFr]; omega
  have hFav2 : (∀ x ∈ ids (iteElse f s.nextBlk thn thnR D), x ≠ s.nextBlk + 2) ∧
      (iteElse f s.nextBlk thn thnR D).curRef ≠ s.nextBlk + 2 ∧
      s.nextBlk + 2 < (iteElse f s.nextBlk thn thnR D).nextBlk := by
    refine ⟨?_, ?_, ?_⟩
    · intro x hx; simp only [ids, hFd] at hx; exact hEav2 x hx
    · rw [hFr]; omega
    · rw [hFn]; omega
  obtain ⟨hGa, hGl, hGo, _, hGt⟩ :=
    ih _ _ _ els Se ne (iteElse f s.nextBlk thn thnR D) hok.hels hFa hFo.inv
  generalize hG : emitCode f (iteElse f s.nextBlk thn thnR D) els = G at hGa hGl hGo hGt
  have hGav := out_avoid hGo hFav2.1 hFav2.2.1 hFav2.2.2
  obtain ⟨hHo, hHc, hHav⟩ := iteArmEnd_out (h := s.nextBlk) (rs := elsR) (hFo.trans hGo) (fun h => (hGt h).2) hGav
  obtain ⟨hHn, hHl, hHs, hHv, hHe, hGH⟩ := iteArmEnd_facts f s.nextBlk els elsR G
  generalize hH : iteArmEnd f s.nextBlk els elsR G = H at hHo hHc hHav hHn hHl hHs hHv hHe hGH
  have hHa : Aligned H ne := aligned_congr hGa hHv hHs hHe
  have hCH : (iteThen s m).done <+: H.done := by
    refine hDo.grows.2.trans (hDE.trans ?_)
    refine List.IsPrefix.trans ?_ (hGo.grows.2.trans hGH)
    rw [hFd]; exact List.prefix_refl _
  have hext : ∀ t : CS, H.done <+: t.done → ExtC s t := by
    intro t ht
    refine (ExtC.close ({ s with nextBlk := s.nextBlk + 3 } : CS)
      (.brif (s.get m.flag) ⟨s.nextBlk⟩ [] ⟨s.nextBlk + 1⟩ [])).grow ?_
    have := hCH.trans ht
    rw [hCd] at this; exact this
  have hHl' : H.labels = s.labels := by rw [hHl, hGl, hFl, hDl, hCl]
  unfold iteJoin
  split
  · rename_i hb
    exact ⟨by simpa [hb] using hHa, hHl', hb ▸ hHo, hext H (List.prefix_refl _), fun _ => hHc⟩
  · rename_i hb
    simp only [Bool.not_eq_true] at hb
    obtain ⟨hId, hIn, hIr, _, _, hIl, _⟩ :=
      open'_fields H (s.nextBlk + 2) m.jTys H.slots
    refine ⟨?_, ?_, ?_, hext _ (by show H.done <+: (CS.open' _ _ _ _).done; rw [hId]; exact List.prefix_refl _),
      fun h => by rw [hb] at h; cases h⟩
    · simpa [hb] using aligned_open_nl hHa (s.nextBlk + 2) m.jTys
    · show (CS.open' _ _ _ _).labels = _; rw [hIl, hHl']
    · rw [hb]
      refine hHo.reopen (u := { H.open' (s.nextBlk + 2) m.jTys H.slots with
          slots := H.slots + m.jTys.length }) hId hIn ?_ ?_ ?_
      · show (CS.open' _ _ _ _).curRef ∉ _; rw [hIr]; intro hm; exact hHav _ hm rfl
      · show (CS.open' _ _ _ _).curRef < _; rw [hIr, hHn]
        have := hGo.grows.1; rw [hFn] at this; omega
      · show _ ≤ (CS.open' _ _ _ _).curRef; rw [hIr]; omega

theorem struct_dloop (f : Nat) (ih : StructP f) {lb : List SLbl} {S : Scope} {n : Nat}
    {l : DLoop} {body : List Piece} {Sb : Scope} {nb : Nat}
    (hok : DLoopOk f lb S n l body Sb nb) (s : CS) (ha : Aligned s n) (hi : Inv s) :
    Aligned (emitPiece f s (.dloop l body)) (nb + l.exitTys.length) ∧
    (emitPiece f s (.dloop l body)).labels = s.labels ∧
    Out s (emitPiece f s (.dloop l body)) false ∧
    ExtC s (emitPiece f s (.dloop l body)) := by
  rw [show emitPiece f s (.dloop l body) = emitDLoop f s l body by rw [emitPiece],
      emitDLoop_eq]
  obtain ⟨hCd, hCn, hCr, _, _, _, _⟩ := dloopHead_facts s l
  have hCa : Aligned (dloopHead s l) (n + l.pTys.length) :=
    aligned_open (aligned_congr (u := (({ s with nextBlk := s.nextBlk + 2 } : CS).close
      (dloopEntry s l))) ha rfl rfl rfl) s.nextBlk l.pTys _
  have hCids : ∀ x ∈ ids (dloopHead s l), x < s.nextBlk := by
    rw [ids_of_done hCd]; intro x hx
    simp only [List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | hx
    · exact hi.lt x hx
    · rw [hx]; exact hi.cur
  have hCo : Out s (dloopHead s l) false := by
    have h0 := ((Out.refl hi).bump 2).close (dloopEntry s l)
    refine h0.reopen ?_ ?_ ?_ ?_ ?_
    · rw [hCd]; rfl
    · rw [hCn]; rfl
    · rw [hCr]; intro hm
      rw [ids_close] at hm
      simp only [List.mem_append, List.mem_singleton] at hm
      rcases hm with hm | hm
      · have := hi.lt _ hm; omega
      · have := hi.cur; omega
    · rw [hCr]; show s.nextBlk < s.nextBlk + 2; omega
    · rw [hCr]; exact Nat.le_refl _
  obtain ⟨hDa, _, hDo, _, hDt⟩ := ih _ _ _ body Sb nb (dloopHead s l) hok.hbody hCa hCo.inv
  generalize hD : emitCode f (dloopHead s l) body = D at hDa hDo hDt
  have hDav := out_avoid hDo (r := s.nextBlk + 1)
    (fun x hx => by have := hCids x hx; omega) (by rw [hCr]; omega) (by rw [hCn]; omega)
  have hDnb : s.nextBlk + 2 ≤ D.nextBlk := hCn ▸ hDo.grows.1
  have hEo : Out s (dloopBodyEnd f s.nextBlk l body D) true ∧
      D.done <+: (dloopBodyEnd f s.nextBlk l body D).done ∧
      (∀ x ∈ ids (dloopBodyEnd f s.nextBlk l body D), x ≠ s.nextBlk + 1) ∧
      (dloopBodyEnd f s.nextBlk l body D).nextBlk = D.nextBlk ∧
      (dloopBodyEnd f s.nextBlk l body D).slots = D.slots ∧
      (dloopBodyEnd f s.nextBlk l body D).nextVal = D.nextVal ∧
      (dloopBodyEnd f s.nextBlk l body D).env = D.env := by
    unfold dloopBodyEnd
    split
    · rename_i htb
      rw [htb] at hDo
      exact ⟨hCo.trans hDo, List.prefix_refl _, hDav.1, rfl, rfl, rfl, rfl⟩
    · rename_i htb
      simp only [Bool.not_eq_true] at htb
      rw [htb] at hDo
      exact ⟨(hCo.trans hDo).close _, close_done_prefix D _, avoid_close hDav.1 hDav.2 _,
        rfl, rfl, rfl, rfl⟩
  obtain ⟨hEo, hDE, hEav, hEn, hEs, hEv, hEe⟩ := hEo
  generalize hE : dloopBodyEnd f s.nextBlk l body D = E at hEo hDE hEav hEn hEs hEv hEe
  obtain ⟨hId, hIn, hIr, _, _, hIl, _⟩ := dloopExit_facts s.nextBlk l s.labels E
  have hIo : Out s (dloopExit s.nextBlk l s.labels E) false := by
    refine hEo.reopen hId hIn ?_ ?_ ?_
    · rw [hIr]; intro hm; exact hEav _ hm rfl
    · rw [hIr, hEn]; omega
    · rw [hIr]; omega
  refine ⟨?_, hIl, hIo, ?_⟩
  · exact aligned_open (aligned_congr hDa hEv hEs hEe) (s.nextBlk + 1) l.exitTys s.labels
  · refine (ExtC.close ({ s with nextBlk := s.nextBlk + 2 } : CS) (dloopEntry s l)).grow ?_
    have hCI : (dloopHead s l).done <+: (dloopExit s.nextBlk l s.labels E).done := by
      rw [hId]; exact hDo.grows.2.trans hDE
    rw [hCd] at hCI
    exact hCI

theorem inv_close_same {s : CS} (hi : Inv s) (x : Inst) :
    Out s (s.close x) true := (Out.refl hi).close x

/-- **The emitter's structure, for every region the check accepts.** -/
theorem emit_struct : ∀ f, StructP f := by
  intro f
  induction f with
  | zero => intro lb S n c S' n' s h; simp [scGo] at h
  | succ f ih =>
    intro lb S n c S' n' s h ha hi
    cases c with
    | nil =>
        simp only [scGo, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨_, rfl⟩ := h
        rw [emitCode_nil, termsGo_nil]
        exact ⟨ha, rfl, Out.refl hi, .inl (ExtO.refl s), fun h => by cases h⟩
    | cons p ps =>
      rw [emitCode_cons, termsGo_cons]
      -- The piece, then the rest from where it left off.
      have compose : ∀ (t : CS) (nt : Nat) (St : Scope),
          Aligned t nt → t.labels = s.labels → Out s t false → (ExtO s t ∨ ExtC s t) →
          pieceTerm f p = false → scGo f lb St nt ps = some (S', n') →
          Aligned (emitCode f t ps) n' ∧ (emitCode f t ps).labels = s.labels ∧
          Out s (emitCode f t ps) (if ps.isEmpty then pieceTerm f p else termsGo f ps) ∧
          (ExtO s (emitCode f t ps) ∨ ExtC s (emitCode f t ps)) ∧
          ((if ps.isEmpty then pieceTerm f p else termsGo f ps) = true →
            ExtC s (emitCode f t ps) ∧ (emitCode f t ps).cur = []) := by
        intro t nt St hta htl hto hte hpt hk
        obtain ⟨a1, l1, o1, e1, t1⟩ := ih lb St nt ps S' n' t hk hta hto.inv
        have hsame : (if ps.isEmpty then pieceTerm f p else termsGo f ps) = termsGo f ps := by
          split
          · rename_i he; rw [List.isEmpty_iff] at he; subst he; rw [hpt, termsGo_nil]
          · rfl
        rw [hsame]
        refine ⟨a1, l1.trans htl, hto.trans o1, ?_, fun ht => ?_⟩
        · rcases hte with h1 | h1
          · rcases e1 with h2 | h2
            · exact .inl (h1.trans h2)
            · exact .inr (h1.extC h2)
          · exact .inr (h1.grow o1.grows.2)
        · obtain ⟨c1, c2⟩ := t1 ht
          refine ⟨?_, c2⟩
          rcases hte with h1 | h1
          · exact h1.extC c1
          · exact h1.grow o1.grows.2
      -- A piece that leaves on every path is the last one.
      have last : ∀ (t : CS), ps = [] → Aligned t n' → t.labels = s.labels →
          Out s t true → ExtC s t → t.cur = [] → pieceTerm f p = true →
          Aligned (emitCode f t ps) n' ∧ (emitCode f t ps).labels = s.labels ∧
          Out s (emitCode f t ps) (if ps.isEmpty then pieceTerm f p else termsGo f ps) ∧
          (ExtO s (emitCode f t ps) ∨ ExtC s (emitCode f t ps)) ∧
          ((if ps.isEmpty then pieceTerm f p else termsGo f ps) = true →
            ExtC s (emitCode f t ps) ∧ (emitCode f t ps).cur = []) := by
        intro t hps hta htl hto hte htc hpt
        subst hps
        rw [emitCode_nil]
        simp only [List.isEmpty_nil, if_true, hpt]
        exact ⟨hta, htl, hto, .inr hte, fun _ => ⟨hte, htc⟩⟩
      cases p with
      | straight ss =>
          obtain ⟨S1, n1, h1, h2⟩ := scGo_straight_inv h
          have hn1 := scStmts_count ss S n S1 n1 h1
          have hep : emitPiece f s (.straight ss) = emitStmts s ss := by rw [emitPiece]
          rw [hep]
          obtain ⟨hd, hnb, hcr⟩ := emitStmts_blk ss s
          exact compose _ n1 S1 (hn1 ▸ emitStmts_aligned ss s n ha) (emitStmts_labels ss s)
            ((Out.refl hi).same hd hnb hcr) (.inl (ExtO.emitStmts s ss)) rfl h2
      | loop l pre body =>
          obtain ⟨Sp, np, Sb, nb, hok, hk⟩ := scGo_loop_inv h
          obtain ⟨a1, l1, o1, e1⟩ := struct_loop f ih hok s ha hi
          exact compose _ _ _ a1 l1 o1 (.inr e1) rfl hk
      | dloop l body =>
          obtain ⟨Sb, nb, hok, hk⟩ := scGo_dloop_inv h
          obtain ⟨a1, l1, o1, e1⟩ := struct_dloop f ih hok s ha hi
          exact compose _ _ _ a1 l1 o1 (.inr e1) rfl hk
      | ite m thn els thnR elsR =>
          obtain ⟨St, nt, Se, ne, hok, hk⟩ := scGo_ite_inv h
          obtain ⟨a1, l1, o1, e1, c1⟩ := struct_ite f ih hok s ha hi
          rcases hk with ⟨ht1, ht2, hps, _, hn⟩ | ⟨hnt, hk⟩
          · have hb : (termsGo f thn && termsGo f els) = true := by simp [ht1, ht2]
            rw [hb] at o1 a1
            simp only [if_true] at a1
            exact last _ hps (hn ▸ a1) l1 o1 e1 (c1 hb) (by simpa [pieceTerm] using hb)
          · rw [hnt] at o1 a1
            simp only [Bool.false_eq_true, if_false] at a1
            exact compose _ _ _ a1 l1 o1 (.inr e1) (by simpa [pieceTerm] using hnt) hk
      | br d args =>
          obtain ⟨L, _, hps, _, _, _, _, hn⟩ := scGo_br_inv h
          have hep : emitPiece f s (.br d args) = s.close (.jump ⟨((s.labels[d]?).map (·.1)).getD 1000000⟩
              (args.map s.get)) := by rw [emitPiece]
          rw [hep]
          exact last _ hps (hn ▸ aligned_congr ha rfl rfl rfl) rfl (inv_close_same hi _)
            (ExtC.close s _) rfl rfl
      | cont d args =>
          obtain ⟨L, _, hps, _, _, _, hn⟩ := scGo_cont_inv h
          have hep : emitPiece f s (.cont d args) = s.close (.jump ⟨((s.labels[d]?).bind (·.2)).getD 1000000⟩
              (args.map s.get)) := by rw [emitPiece]
          rw [hep]
          exact last _ hps (hn ▸ aligned_congr ha rfl rfl rfl) rfl (inv_close_same hi _)
            (ExtC.close s _) rfl rfl

-- ---------------------------------------------------------------------------
-- Where a state sits in the finished function
-- ---------------------------------------------------------------------------

/-- The block `s` is building is in `F`, holding what `s` has put in it and then
    `R`. -/
def Placed (F : FuncData) (s : CS) (R : List Inst) : Prop :=
  ∃ b ∈ F.blocks, b.ref.id = s.curRef ∧ b.params = s.curPars ∧ b.insts = s.cur.reverse ++ R

/-- Every block `s` has finished is in `F`. -/
def Done (F : FuncData) (s : CS) : Prop := ∀ b ∈ s.done, b ∈ F.blocks

/-- Block ids in `F` are distinct. -/
def Distinct (F : FuncData) : Prop := (F.blocks.map (·.ref.id)).Nodup

theorem Done.mono {F : FuncData} {s t : CS} (h : Done F t) (hp : s.done <+: t.done) :
    Done F s := fun b hb => h b (hp.subset hb)

theorem Placed.unique {F : FuncData} (hF : Distinct F) {s : CS} {R R' : List Inst}
    (h1 : Placed F s R) (h2 : Placed F s R') : R = R' := by
  obtain ⟨b, hb, hid, _, hins⟩ := h1
  obtain ⟨b', hb', hid', _, hins'⟩ := h2
  have e1 := find_blk F.blocks hF b hb
  have e2 := find_blk F.blocks hF b' hb'
  rw [hid] at e1; rw [hid'] at e2
  rw [e1] at e2
  cases e2
  rw [hins] at hins'
  exact List.append_cancel_left hins'

theorem placed_close {F : FuncData} {s : CS} {x : Inst} (h : Done F (s.close x)) :
    Placed F s [x] :=
  ⟨{ ref := ⟨s.curRef⟩, params := s.curPars, insts := (x :: s.cur).reverse },
   h _ (by simp [CS.close]), rfl, rfl, by simp⟩

theorem ExtO.placed {F : FuncData} {s t : CS} (h : ExtO s t) {R : List Inst}
    (hP : Placed F t R) : ∃ R', Placed F s R' := by
  obtain ⟨a, b, X, c⟩ := h
  obtain ⟨blk, hb, hid, hpar, hins⟩ := hP
  exact ⟨X.reverse ++ R, blk, hb, hid.trans a, hpar.trans b, by
    rw [hins, c]; simp [List.reverse_append, List.append_assoc]⟩

theorem ExtC.placed {F : FuncData} {s t : CS} (h : ExtC s t) (hD : Done F t) :
    ∃ R, Placed F s R := by
  obtain ⟨b, hb, hid, hpar, R, hins⟩ := h
  exact ⟨R, b, hD b hb, hid, hpar, hins⟩

/-- Run a block from part-way through: its remaining instructions, then wherever
    its terminator sends control. -/
def runK (env : FnEnv) (F : FuncData) (steps : Nat) (st : Blocks.BSt) (R : List Inst) :
    Outcome World :=
  match Blocks.runInsts env st R with
  | .stuck m => .stuck m
  | .ok (s', next) w =>
      match next with
      | .done => .ok w w
      | .goto t vs => Blocks.runFrom env F steps { s' with world := w } t vs

/-- What entering a block does to the value array: its parameters bound. -/
def bindVals (vals : Blocks.Vals) (pars : List (Val × ClifTy)) (vs : List V) : Blocks.Vals :=
  (pars.zip vs).foldl (fun vs pa => Blocks.setV vs pa.1.1 pa.2) vals

/-- **Entering a block `s` was building** runs what is left of it, from its
    parameters bound. -/
theorem enter {env : FnEnv} {F : FuncData} (hF : Distinct F) {t : CS} {R : List Inst}
    (hP : Placed F t R) (hcur : t.cur = []) {m : Nat} {tys : List ClifTy}
    (hpars : t.curPars = parsOf m tys) {vs : List V} (hlen : tys.length = vs.length)
    (steps : Nat) (st : Blocks.BSt) :
    Blocks.runFrom env F (steps + 1) st t.curRef vs
      = runK env F steps ⟨bindVals st.vals (parsOf m tys) vs, st.world⟩ R := by
  obtain ⟨b, hb, hid, hpar, hins⟩ := hP
  rw [← hid, runFrom_block env F steps st b hF hb vs
    (by rw [hpar, hpars, parsOf_length]; exact hlen)]
  unfold runK
  rw [hins, hpar, hpars, hcur]
  rfl

theorem runK_insts {env : FnEnv} {F : FuncData} {steps : Nat} {st st' : Blocks.BSt}
    {L R : List Inst} (h : Blocks.runInsts env st (L ++ R) = Blocks.runInsts env st' R) :
    runK env F steps st (L ++ R) = runK env F steps st' R := by
  unfold runK; rw [h]

theorem runK_jump {env : FnEnv} {F : FuncData} {steps : Nat} {vals : Blocks.Vals} {w : World}
    {t : BlockRef} {args : List Val} {rest : List Inst} {vs : List V}
    (h : args.mapM (Blocks.getV vals) = some vs) :
    runK env F steps ⟨vals, w⟩ (.jump t args :: rest)
      = Blocks.runFrom env F steps ⟨vals, w⟩ t.id vs := by
  unfold runK; simp only [Blocks.runInsts, h]

theorem runK_brif {env : FnEnv} {F : FuncData} {steps : Nat} {vals : Blocks.Vals} {w : World}
    {c : Val} {tb eb : BlockRef} {ta ea : List Val} {rest : List Inst} {cv : V}
    (hc : Blocks.getV vals c = some cv) {vs : List V}
    (h : (if Sem.isTrue cv then ta else ea).mapM (Blocks.getV vals) = some vs) :
    runK env F steps ⟨vals, w⟩ (.brif c tb ta eb ea :: rest)
      = Blocks.runFrom env F steps ⟨vals, w⟩ (if Sem.isTrue cv then tb else eb).id vs := by
  unfold runK
  simp only [Blocks.runInsts, hc]
  cases hcv : Sem.isTrue cv <;> simp only [hcv, if_true, if_false, Bool.false_eq_true] at h ⊢ <;>
    simp only [h]

-- ---------------------------------------------------------------------------
-- Binding parameters, on both sides
-- ---------------------------------------------------------------------------

theorem bindAt_size (Γ : Sem.Env) (m : Nat) (vs : List V) :
    (Sem.bindAt Γ m vs).size = m + vs.length := by
  simp only [Sem.bindAt, Array.size_append, Array.size_extract, Array.size_replicate,
    List.size_toArray]
  omega

theorem bindAt_lt (Γ : Sem.Env) (m : Nat) (vs : List V) (i : Nat) (hi : i < m)
    (hΓ : i < Γ.size) : (Sem.bindAt Γ m vs)[i]? = Γ[i]? := by
  simp only [Sem.bindAt]
  rw [Array.getElem?_append_left (by simp; omega), Array.getElem?_append_left (by simp; omega)]
  rw [Array.getElem?_extract]; simp [hΓ]; omega

theorem bindAt_hi (Γ : Sem.Env) (m : Nat) (vs : List V) (j : Nat) :
    (Sem.bindAt Γ m vs)[m + j]? = vs[j]? := by
  simp only [Sem.bindAt]
  rw [Array.getElem?_append_right (by simp; omega)]
  simp only [Array.size_append, Array.size_extract, Array.size_replicate]
  have : m + j - (min m Γ.size + (m - Γ.size)) = j := by omega
  simp [this]

theorem bindVals_frame : ∀ (tys : List ClifTy) (vs : List V) (m : Nat) (vals : Blocks.Vals),
    Frame m vals (bindVals vals (parsOf m tys) vs) := by
  intro tys
  induction tys with
  | nil => intro vs m vals; simp [bindVals, parsOf]; exact Frame.refl m vals
  | cons t ts ih =>
      intro vs m vals
      cases vs with
      | nil => simp [bindVals, parsOf]; exact Frame.refl m vals
      | cons v vs =>
          have h1 := frame_setV m m (Nat.le_refl m) vals v
          have h2 := (ih vs (m + 1) (Blocks.setV vals ⟨m⟩ v)).mono (Nat.le_succ m)
          simpa [bindVals, parsOf] using h1.trans h2

theorem bindVals_hi : ∀ (tys : List ClifTy) (vs : List V) (m : Nat) (vals : Blocks.Vals),
    tys.length = vs.length → ∀ j, j < vs.length →
      Blocks.getV (bindVals vals (parsOf m tys) vs) ⟨m + j⟩ = vs[j]? := by
  intro tys
  induction tys with
  | nil => intro vs m vals hl j hj; cases vs <;> simp at hl hj
  | cons t ts ih =>
      intro vs m vals hl j hj
      cases vs with
      | nil => simp at hj
      | cons v vs =>
        simp only [List.length_cons, Nat.add_right_cancel_iff] at hl
        cases j with
        | zero =>
            have h1 := getV_setV_self vals ⟨m⟩ v
            have h2 := bindVals_frame ts vs (m + 1) (Blocks.setV vals ⟨m⟩ v) m (by omega) v h1
            simpa [bindVals, parsOf] using h2
        | succ j =>
            have := ih vs (m + 1) (Blocks.setV vals ⟨m⟩ v) hl j (by simp at hj; omega)
            simpa [bindVals, parsOf, Nat.add_assoc, Nat.add_comm 1 j] using this

/-- **Binding a block's parameters is binding the term's slots**, masked: what
    agreed below `m` still agrees, and the parameters agree with the values
    bound. -/
theorem bind_rel {A : Scope} {vals : Blocks.Vals} {Γ : Sem.Env} (h : Rel A vals Γ) {m : Nat}
    (hA : ∀ i, A.mem i = true → i < m) {tys : List ClifTy} {vs : List V}
    (hl : tys.length = vs.length) :
    Rel (A.add m (m + vs.length)) (bindVals vals (parsOf m tys) vs) (Sem.bindAt Γ m vs) := by
  intro i hi
  rcases (Scope.mem_add A m (m + vs.length) i).mp hi with h1 | ⟨h1, h2⟩
  · obtain ⟨x, hx, hxv⟩ := h i h1
    have him := hA i h1
    refine ⟨x, ?_, bindVals_frame tys vs m vals i him x hxv⟩
    rw [bindAt_lt Γ m vs i him (h.lt h1)]; exact hx
  · obtain ⟨j, rfl⟩ : ∃ j, i = m + j := ⟨i - m, by omega⟩
    have hj : j < vs.length := by omega
    refine ⟨vs[j], ?_, ?_⟩
    · rw [bindAt_hi]; simp [hj]
    · rw [bindVals_hi tys vs m vals hl j hj]; simp [hj]

theorem mapM_length {α β : Type} {f : α → Option β} :
    ∀ {xs : List α} {ys : List β}, xs.mapM f = some ys → ys.length = xs.length := by
  intro xs
  induction xs with
  | nil => intro ys h; simp at h; subst h; rfl
  | cons x xs ih =>
      intro ys h
      simp only [List.mapM_cons, Option.bind_eq_bind] at h
      cases hx : f x with
      | none => simp [hx] at h
      | some y =>
        cases hr : xs.mapM f with
        | none => simp [hx, hr] at h
        | some zs => simp [hx, hr] at h; subst h; simp [ih hr]

/-- The emitter resolves in-range slots to themselves. -/
theorem map_get_aligned {s : CS} {n : Nat} (ha : Aligned s n) (rs : List R)
    (h : ∀ r ∈ rs, r < n) : rs.map s.get = rs.map (fun r => (⟨r⟩ : Val)) :=
  List.map_congr_left (fun r hr => get_of_aligned s n ha r (h r hr))

-- ---------------------------------------------------------------------------
-- The simulation
-- ---------------------------------------------------------------------------

/-- What running a region's compiled form does, given what running the term did.

    Finishing normally: the blocks, from `R0`, reach the instructions after the
    region in a state agreeing on the region's out-scope. Leaving a loop: they
    jump to that loop's exit block with the same values, agreeing on what the
    loop needs in scope. Going round: they jump to its back-edge target. In
    every case they have written nothing below `n`, and they get there in a
    number of block entries that does not depend on the budget left. -/
def SimRes (env : FnEnv) (F : FuncData) (lb : List SLbl) (labels : List (Nat × Option Nat))
    (n : Nat) (S' : Scope) (n' : Nat) (t : CS) (vals : Blocks.Vals) (w : World)
    (R0 : List Inst) : Sem.CodeRes → Prop
  | .ok Γ' w' => Γ'.size = n' ∧ ∃ vals' cost, Rel S' vals' Γ' ∧ Frame n vals vals' ∧
      ∀ R1, Placed F t R1 → ∀ steps,
        runK env F (steps + cost) ⟨vals, w⟩ R0 = runK env F steps ⟨vals', w'⟩ R1
  | .brk d Γb vs w' => ∃ L, lb[d]? = some L ∧ vs.length = L.exitN ∧ ∃ vals' cost,
      Rel L.need vals' Γb ∧ Frame n vals vals' ∧ ∀ steps,
        runK env F (steps + cost) ⟨vals, w⟩ R0
          = Blocks.runFrom env F steps ⟨vals', w'⟩ (((labels[d]?).map (·.1)).getD 1000000) vs
  | .cont d vs w' => ∃ L, lb[d]? = some L ∧ vs.length = L.carryN ∧ ∃ vals' cost,
      Frame n vals vals' ∧ ∀ steps,
        runK env F (steps + cost) ⟨vals, w⟩ R0
          = Blocks.runFrom env F steps ⟨vals', w'⟩ (((labels[d]?).bind (·.2)).getD 1000000) vs
  | .stuck _ => True

/-- The simulation for every region at one emitter fuel, for every term fuel. -/
def SimP (env : FnEnv) (cfg : Sem.Cfg) (F : FuncData) (f : Nat) : Prop :=
  ∀ (k : Nat) (lb : List SLbl) (S : Scope) (n : Nat) (c : List Piece) (S' : Scope) (n' : Nat)
    (s : CS) (Γ : Sem.Env) (w : World) (vals : Blocks.Vals) (R0 : List Inst),
    f ≤ HProg.fuel → scGo f lb S n c = some (S', n') → Aligned s n → Inv s →
    Placed F s R0 → Done F (emitCode f s c) →
    (termsGo f c = false → ∃ R1, Placed F (emitCode f s c) R1) →
    Γ.size = n → Rel S vals Γ →
    SimRes env F lb s.labels n S' n' (emitCode f s c) vals w R0 (Sem.runCode k cfg Γ w c)

/-- A run that first gets from `R0` to `R_mid` simulates whatever the rest does. -/
theorem SimRes.prepend {env : FnEnv} {F : FuncData} {lb : List SLbl}
    {labels : List (Nat × Option Nat)} {n n1 : Nat} {S' : Scope} {n' : Nat} {t : CS}
    {vals vals1 : Blocks.Vals} {w w1 : World} {R0 Rm : List Inst} {res : Sem.CodeRes} {c : Nat}
    (h : SimRes env F lb labels n1 S' n' t vals1 w1 Rm res)
    (hc : ∀ st, runK env F (st + c) ⟨vals, w⟩ R0 = runK env F st ⟨vals1, w1⟩ Rm)
    (hf : Frame n vals vals1) (hle : n ≤ n1) :
    SimRes env F lb labels n S' n' t vals w R0 res := by
  cases res with
  | ok Γ' w' =>
      obtain ⟨hs, vals', c2, hr, hf2, heq⟩ := h
      refine ⟨hs, vals', c2 + c, hr, hf.trans (hf2.mono hle), fun R1 hP st => ?_⟩
      rw [← Nat.add_assoc, hc, heq R1 hP]
  | brk d Γb vs w' =>
      obtain ⟨L, hL, hlen, vals', c2, hr, hf2, heq⟩ := h
      refine ⟨L, hL, hlen, vals', c2 + c, hr, hf.trans (hf2.mono hle), fun st => ?_⟩
      rw [← Nat.add_assoc, hc, heq]
  | cont d vs w' =>
      obtain ⟨L, hL, hlen, vals', c2, hf2, heq⟩ := h
      refine ⟨L, hL, hlen, vals', c2 + c, hf.trans (hf2.mono hle), fun st => ?_⟩
      rw [← Nat.add_assoc, hc, heq]
  | stuck _ => trivial

/-- What `runCode` does with a piece's result: carry on with the rest when it
    finished, pass it out otherwise. -/
def andThen (k : Nat) (cfg : Sem.Cfg) (ps : List Piece) : Sem.CodeRes → Sem.CodeRes
  | .stuck s => .stuck s
  | .brk d Γb vs w' => .brk d Γb vs w'
  | .cont d vs w' => .cont d vs w'
  | .ok Γ' w' => Sem.runCode k cfg Γ' w' ps

theorem runCode_cons' (k : Nat) (cfg : Sem.Cfg) (Γ : Sem.Env) (w : World) (p : Piece)
    (ps : List Piece) :
    Sem.runCode (k + 1) cfg Γ w (p :: ps) = andThen k cfg ps (Sem.runPiece k cfg Γ w p) := by
  rw [runCode_cons]; cases Sem.runPiece k cfg Γ w p <;> rfl

/-- **A piece followed by the rest.** -/
theorem SimRes.seq {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} {lb : List SLbl}
    {labels : List (Nat × Option Nat)} {n n1 : Nat} {S1 S' : Scope} {n' : Nat} {t1 t : CS}
    {vals : Blocks.Vals} {w : World} {R0 : List Inst} {r : Sem.CodeRes} {k : Nat}
    {ps : List Piece}
    (hp : SimRes env F lb labels n S1 n1 t1 vals w R0 r) (hle : n ≤ n1)
    (hmid : ∀ Γ1 w1, r = .ok Γ1 w1 → ∃ Rm, Placed F t1 Rm)
    (hrest : ∀ Γ1 w1 vals1 Rm, r = .ok Γ1 w1 → Placed F t1 Rm → Γ1.size = n1 →
      Rel S1 vals1 Γ1 →
      SimRes env F lb labels n1 S' n' t vals1 w1 Rm (Sem.runCode k cfg Γ1 w1 ps)) :
    SimRes env F lb labels n S' n' t vals w R0
      (andThen k cfg ps r) := by
  cases r with
  | ok Γ1 w1 =>
      obtain ⟨Rm, hPm⟩ := hmid Γ1 w1 rfl
      obtain ⟨hs, vals1, c1, hr, hf, heq⟩ := hp
      exact (hrest Γ1 w1 vals1 Rm rfl hPm hs hr).prepend (heq Rm hPm) hf hle
  | brk d Γb vs w' => exact hp
  | cont d vs w' => exact hp
  | stuck _ => trivial

theorem placed_emitStmts {F : FuncData} {s : CS} {ss : List Stmt} {R : List Inst}
    (h : Placed F (emitStmts s ss) R) : Placed F s (emittedList s ss ++ R) := by
  obtain ⟨b, hb, hid, hpar, hins⟩ := h
  obtain ⟨_, _, hcr⟩ := emitStmts_blk ss s
  refine ⟨b, hb, hid.trans hcr, hpar.trans (emitStmts_curPars ss s), ?_⟩
  rw [hins, emitStmts_cur]; simp [List.append_assoc]

theorem sim_straight {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} (hF : Distinct F)
    {lb : List SLbl} {S : Scope} {n : Nat} {ss : List Stmt} {S1 : Scope} {n1 : Nat}
    (hsc : scStmts S n ss = some (S1, n1)) (f k : Nat) (s : CS) (ha : Aligned s n)
    (Γ : Sem.Env) (w : World) (vals : Blocks.Vals) (R0 : List Inst) (hP : Placed F s R0)
    (hs : Γ.size = n) (hr : Rel S vals Γ) :
    SimRes env F lb s.labels n S1 n1 (emitPiece f s (.straight ss)) vals w R0
      (Sem.runPiece k cfg Γ w (.straight ss)) := by
  cases k with
  | zero => simp [Sem.runPiece, SimRes]
  | succ k =>
    simp only [Sem.runPiece]
    cases hrun : Sem.runStmts cfg Γ w ss with
    | stuck _ => trivial
    | ok Γ' w' =>
      have hep : emitPiece f s (.straight ss) = emitStmts s ss := by rw [emitPiece]
      rw [hep]
      by_cases hex : ∃ R1, Placed F (emitStmts s ss) R1
      · obtain ⟨R1, hP1⟩ := hex
        obtain ⟨vals', hins, hr', hs', hf⟩ :=
          stmts_msim env cfg ss s S n S1 n1 w w' Γ Γ' vals R1 hsc ha hs hr hrun
        refine ⟨hs', vals', 0, hr', hf, fun R2 hP2 st => ?_⟩
        rw [← hP1.unique hF hP2]
        have := hP.unique hF (placed_emitStmts hP1)
        subst this
        exact runK_insts hins
      · obtain ⟨vals', _, hr', hs', hf⟩ :=
          stmts_msim env cfg ss s S n S1 n1 w w' Γ Γ' vals [] hsc ha hs hr hrun
        exact ⟨hs', vals', 0, hr', hf, fun R2 hP2 => absurd ⟨R2, hP2⟩ hex⟩

theorem runPiece_br (k : Nat) (cfg : Sem.Cfg) (Γ : Sem.Env) (w : World) (d : Nat) (args : List R) :
    Sem.runPiece (k + 1) cfg Γ w (.br d args)
      = match args.mapM (fun r => Γ[r]?) with
        | none => .stuck "branch-out value is not in scope"
        | some vs => .brk d Γ vs w := rfl

theorem runPiece_cont (k : Nat) (cfg : Sem.Cfg) (Γ : Sem.Env) (w : World) (d : Nat)
    (args : List R) :
    Sem.runPiece (k + 1) cfg Γ w (.cont d args)
      = match args.mapM (fun r => Γ[r]?) with
        | none => .stuck "loop carry is not in scope"
        | some vs => .cont d vs w := rfl

/-- The arguments a jump carries, read on the block side, are the term's. -/
theorem args_read {s : CS} {n : Nat} (ha : Aligned s n) {S : Scope} {vals : Blocks.Vals}
    {Γ : Sem.Env} (hr : Rel S vals Γ) {rs : List R} (hin : allIn S n rs = true) :
    (rs.map s.get).mapM (Blocks.getV vals) = rs.mapM (fun r => Γ[r]?) := by
  rw [allIn_iff] at hin
  rw [map_get_aligned ha rs (fun r h => (hin r h).1)]
  exact hr.mapM rs (fun r h => (hin r h).2)

theorem sim_br {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} {lb : List SLbl} {S : Scope}
    {n : Nat} {d : Nat} {args : List R} {L : SLbl} (hL : lb[d]? = some L)
    (hin : allIn S n args = true) (hlen : args.length = L.exitN) (hneed : L.need.sub S = true)
    (f k : Nat) (s : CS) (ha : Aligned s n) (Γ : Sem.Env) (w : World) (vals : Blocks.Vals)
    (R0 : List Inst) (hF : Distinct F) (hP : Placed F s R0)
    (hD : Done F (emitPiece f s (.br d args))) (hr : Rel S vals Γ) (S1 : Scope) (n1 : Nat) :
    SimRes env F lb s.labels n S1 n1 (emitPiece f s (.br d args)) vals w R0
      (Sem.runPiece k cfg Γ w (.br d args)) := by
  have hep : emitPiece f s (.br d args) = s.close (.jump ⟨((s.labels[d]?).map (·.1)).getD 1000000⟩
      (args.map s.get)) := by rw [emitPiece]
  rw [hep] at hD ⊢
  have := hP.unique hF (placed_close hD); subst this
  cases k with
  | zero => trivial
  | succ k =>
    rw [runPiece_br]
    cases hm : args.mapM (fun r => Γ[r]?) with
    | none => trivial
    | some vs =>
      refine ⟨L, hL, (mapM_length hm).trans hlen, vals, 0, hr.sub hneed, Frame.refl n vals,
        fun st => ?_⟩
      exact runK_jump ((args_read ha hr hin).trans hm)

theorem sim_cont {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} {lb : List SLbl} {S : Scope}
    {n : Nat} {d : Nat} {args : List R} {L : SLbl} (hL : lb[d]? = some L)
    (hin : allIn S n args = true) (hlen : args.length = L.carryN)
    (f k : Nat) (s : CS) (ha : Aligned s n) (Γ : Sem.Env) (w : World) (vals : Blocks.Vals)
    (R0 : List Inst) (hF : Distinct F) (hP : Placed F s R0)
    (hD : Done F (emitPiece f s (.cont d args))) (hr : Rel S vals Γ) (S1 : Scope) (n1 : Nat) :
    SimRes env F lb s.labels n S1 n1 (emitPiece f s (.cont d args)) vals w R0
      (Sem.runPiece k cfg Γ w (.cont d args)) := by
  have hep : emitPiece f s (.cont d args) = s.close (.jump ⟨((s.labels[d]?).bind (·.2)).getD 1000000⟩
      (args.map s.get)) := by rw [emitPiece]
  rw [hep] at hD ⊢
  have := hP.unique hF (placed_close hD); subst this
  cases k with
  | zero => trivial
  | succ k =>
    rw [runPiece_cont]
    cases hm : args.mapM (fun r => Γ[r]?) with
    | none => trivial
    | some vs =>
      refine ⟨L, hL, (mapM_length hm).trans hlen, vals, 0, Frame.refl n vals, fun st => ?_⟩
      exact runK_jump ((args_read ha hr hin).trans hm)

/-- The states `emitIte` passes through, by name. -/
def iteD (f : Nat) (s : CS) (m : IteMeta) (thn : List Piece) : CS := emitCode f (iteThen s m) thn
def iteE (f : Nat) (s : CS) (m : IteMeta) (thn : List Piece) (thnR : List R) : CS :=
  iteArmEnd f s.nextBlk thn thnR (iteD f s m thn)
def iteF (f : Nat) (s : CS) (m : IteMeta) (thn : List Piece) (thnR : List R) : CS :=
  iteElse f s.nextBlk thn thnR (iteD f s m thn)
def iteG (f : Nat) (s : CS) (m : IteMeta) (thn els : List Piece) (thnR : List R) : CS :=
  emitCode f (iteF f s m thn thnR) els
def iteH (f : Nat) (s : CS) (m : IteMeta) (thn els : List Piece) (thnR elsR : List R) : CS :=
  iteArmEnd f s.nextBlk els elsR (iteG f s m thn els thnR)

theorem emitPiece_ite (f : Nat) (s : CS) (m : IteMeta) (thn els : List Piece) (thnR elsR : List R) :
    emitPiece f s (.ite m thn els thnR elsR)
      = iteJoin f s.nextBlk m thn els (iteH f s m thn els thnR elsR) := by
  rw [emitPiece, emitIte_eq]; rfl

theorem ite_stages (f : Nat) (ih : StructP f) {lb : List SLbl} {S : Scope} {n : Nat}
    {m : IteMeta} {thn els : List Piece} {thnR elsR : List R} {St : Scope} {nt : Nat}
    {Se : Scope} {ne : Nat}
    (hok : IteOk f lb S n m thn els thnR elsR St nt Se ne) (s : CS) (ha : Aligned s n)
    (hi : Inv s) :
    Inv (iteThen s m) ∧ Aligned (iteThen s m) n ∧
    Inv (iteF f s m thn thnR) ∧ Aligned (iteF f s m thn thnR) nt ∧
    (iteThen s m).done <+: (iteE f s m thn thnR).done ∧
    (iteE f s m thn thnR).done <+: (iteH f s m thn els thnR elsR).done ∧
    (iteF f s m thn thnR).done = (iteE f s m thn thnR).done ∧
    (iteF f s m thn thnR).cur = [] ∧ (iteF f s m thn thnR).curPars = [] ∧
    (iteF f s m thn thnR).curRef = s.nextBlk + 1 ∧ (iteF f s m thn thnR).labels = s.labels ∧
    (iteH f s m thn els thnR elsR).cur = [] ∧ Aligned (iteH f s m thn els thnR elsR) ne := by
  simp only [iteH, iteG, iteE, iteF, iteD]
  obtain ⟨hCd, hCn, hCr, _, _, hCl, _⟩ := iteThen_facts s m
  have hCa : Aligned (iteThen s m) n :=
    aligned_open0 (aligned_congr (u := (({ s with nextBlk := s.nextBlk + 3 } : CS).close
      (.brif (s.get m.flag) ⟨s.nextBlk⟩ [] ⟨s.nextBlk + 1⟩ []))) ha rfl rfl rfl) s.nextBlk
  have hCids : ∀ x ∈ ids (iteThen s m), x < s.nextBlk := by
    rw [ids_of_done hCd]; intro x hx
    simp only [List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | hx
    · exact hi.lt x hx
    · rw [hx]; exact hi.cur
  have hCo : Out s (iteThen s m) false := by
    have h0 := ((Out.refl hi).bump 3).close
      (.brif (s.get m.flag) ⟨s.nextBlk⟩ [] ⟨s.nextBlk + 1⟩ [])
    refine h0.reopen ?_ ?_ ?_ ?_ ?_
    · rw [hCd]; rfl
    · rw [hCn]; rfl
    · rw [hCr]; intro hm
      rw [ids_close] at hm
      simp only [List.mem_append, List.mem_singleton] at hm
      rcases hm with hm | hm
      · have := hi.lt _ hm; omega
      · have := hi.cur; omega
    · rw [hCr]; show s.nextBlk < s.nextBlk + 3; omega
    · rw [hCr]; exact Nat.le_refl _
  have hCav : ∀ r, s.nextBlk + 1 ≤ r → r < s.nextBlk + 3 →
      (∀ x ∈ ids (iteThen s m), x ≠ r) ∧ (iteThen s m).curRef ≠ r ∧
      r < (iteThen s m).nextBlk := by
    intro r h1 h2
    refine ⟨fun x hx => ?_, by rw [hCr]; omega, by rw [hCn]; omega⟩
    have := hCids x hx; omega
  obtain ⟨hDa, hDl, hDo, _, hDt⟩ := ih _ _ _ thn St nt (iteThen s m) hok.hthn hCa hCo.inv
  have hDav1 := out_avoid hDo (hCav (s.nextBlk + 1) (by omega) (by omega)).1
    (hCav (s.nextBlk + 1) (by omega) (by omega)).2.1 (hCav (s.nextBlk + 1) (by omega) (by omega)).2.2
  have hDav2 := out_avoid hDo (hCav (s.nextBlk + 2) (by omega) (by omega)).1
    (hCav (s.nextBlk + 2) (by omega) (by omega)).2.1 (hCav (s.nextBlk + 2) (by omega) (by omega)).2.2
  have hDnb : s.nextBlk + 3 ≤ (emitCode f (iteThen s m) thn).nextBlk := hCn ▸ hDo.grows.1
  obtain ⟨hEo1, hEc1, hEav1⟩ := iteArmEnd_out (h := s.nextBlk) (rs := thnR) (hCo.trans hDo)
    (fun h => (hDt h).2) hDav1
  obtain ⟨_, _, hEav2⟩ := iteArmEnd_out (h := s.nextBlk) (rs := thnR) (hCo.trans hDo)
    (fun h => (hDt h).2) hDav2
  obtain ⟨hEn, _, hEs, hEv, hEe, hDE⟩ := iteArmEnd_facts f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)
  obtain ⟨hFd, hFn, hFr, hFc, hFp, hFl, _⟩ := iteElse_facts f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)
  have hFa : Aligned (iteElse f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)) nt := by
    have hEa : Aligned (iteArmEnd f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)) nt := aligned_congr hDa hEv hEs hEe
    exact aligned_open0 hEa (s.nextBlk + 1)
  have hFo : Out s (iteElse f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)) false := by
    refine hEo1.reopen hFd (by rw [hFn, hEn]) ?_ ?_ ?_
    · rw [hFr]; intro hm; exact hEav1 _ hm rfl
    · rw [hFr, hEn]; omega
    · rw [hFr]; omega
  have hFav2 : (∀ x ∈ ids (iteElse f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)), x ≠ s.nextBlk + 2) ∧
      (iteElse f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)).curRef ≠ s.nextBlk + 2 ∧
      s.nextBlk + 2 < (iteElse f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)).nextBlk := by
    refine ⟨?_, ?_, ?_⟩
    · intro x hx; simp only [ids, hFd] at hx; exact hEav2 x hx
    · show (iteElse _ _ _ _ _).curRef ≠ _; rw [hFr]; omega
    · show _ < (iteElse _ _ _ _ _).nextBlk; rw [hFn]; omega
  obtain ⟨hGa, hGl, hGo, _, hGt⟩ :=
    ih _ _ _ els Se ne (iteElse f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)) hok.hels hFa hFo.inv
  have hGav := out_avoid hGo hFav2.1 hFav2.2.1 hFav2.2.2
  obtain ⟨_, hHc, _⟩ := iteArmEnd_out (h := s.nextBlk) (rs := elsR) (hFo.trans hGo)
    (fun h => (hGt h).2) hGav
  obtain ⟨_, _, hHs, hHv, hHe, hGH⟩ := iteArmEnd_facts f s.nextBlk els elsR (emitCode f (iteElse f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)) els)
  refine ⟨hCo.inv, hCa, hFo.inv, hFa, hDo.grows.2.trans hDE, ?_, hFd, ?_, hFp, hFr, ?_, hHc,
    aligned_congr hGa hHv hHs hHe⟩
  · have := hGo.grows.2.trans hGH
    show (iteArmEnd f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)).done <+: (iteArmEnd f s.nextBlk els elsR (emitCode f (iteElse f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)) els)).done
    rw [← show (iteElse f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)).done = (iteArmEnd f s.nextBlk thn thnR (emitCode f (iteThen s m) thn)).done from hFd]
    exact this
  · show (iteElse _ _ _ _ _).cur = []; rw [hFc]; exact hEc1
  · show (iteElse _ _ _ _ _).labels = _; rw [hFl, hDl, hCl]

theorem Rel.of_add {S : Scope} {a b : Nat} {vals : Blocks.Vals} {Γ : Sem.Env}
    (h : Rel (S.add a b) vals Γ) : Rel S vals Γ :=
  fun i hi => h i ((Scope.mem_add S a b i).mpr (.inl hi))

theorem iteArmEnd_open {f h : Nat} {arm : List Piece} {rs : List R} {sA : CS}
    (ht : termsGo f arm = false) :
    iteArmEnd f h arm rs sA = sA.close (.jump ⟨h + 2⟩ (rs.map sA.get)) := by
  unfold iteArmEnd; simp [ht]

/-- **One arm of a branch, then the jump to the join.** Given the arm's block
    was entered agreeing on `S`, what the term does with the arm and its exports
    the blocks do with the arm's code and the join's parameters. -/
theorem sim_arm {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} (hF : Distinct F) (f : Nat)
    (hfuel : f ≤ HProg.fuel) (ih : SimP env cfg F f) {lb : List SLbl} {S : Scope}
    {n na : Nat} {a : List Piece} {Sx : Scope} {nx : Nat} {rs : List R} {jTys : List ClifTy}
    {ne : Nat} (hsc : scGo f lb S na a = some (Sx, nx))
    (hx : termsGo f a = true ∨
      (allIn Sx nx rs = true ∧ rs.length = jTys.length ∧ S.sub Sx = true))
    (hnle : n ≤ na) (hnx : nx ≤ ne) (k : Nat) (A : CS) (hAa : Aligned A na) (hAi : Inv A)
    (h : Nat) (t1 : CS) (hDE : Done F (iteArmEnd f h a rs (emitCode f A a)))
    (ht1 : termsGo f a = false → t1.curRef = h + 2 ∧ t1.cur = [] ∧ t1.curPars = parsOf ne jTys)
    (Γa : Sem.Env) (w : World) (vals : Blocks.Vals) (hs : Γa.size = na) (hr : Rel S vals Γa) :
    ∃ RA, Placed F A RA ∧
      SimRes env F lb A.labels n (S.add ne (ne + jTys.length)) (ne + jTys.length) t1 vals w RA
        (match Sem.runCode k cfg Γa w a with
          | .stuck s => .stuck s
          | .brk d Γb vs w' => .brk d Γb vs w'
          | .cont d vs w' => .cont d vs w'
          | .ok Γ' w' =>
              match rs.mapM (Sem.get Γ') with
              | none => .stuck "branch export is not in scope"
              | some vs => .ok (Sem.bindAt Γ' ne vs) w') := by
  obtain ⟨hDa, hDl, _, hDx, hDt⟩ := emit_struct f lb S na a Sx nx A hsc hAa hAi
  obtain ⟨_, _, _, _, _, hDE'⟩ := iteArmEnd_facts f h a rs (emitCode f A a)
  have hDD : Done F (emitCode f A a) := hDE.mono hDE'
  have hPD : termsGo f a = false →
      Placed F (emitCode f A a) [.jump ⟨h + 2⟩ (rs.map (emitCode f A a).get)] := by
    intro ht; rw [iteArmEnd_open ht] at hDE; exact placed_close hDE
  have hreach : ∃ RA, Placed F A RA := by
    rcases hDx with ho | hc
    · cases ht : termsGo f a
      · exact ho.placed (hPD ht)
      · exact (hDt ht).1.placed hDD
    · exact hc.placed hDD
  obtain ⟨RA, hPA⟩ := hreach
  refine ⟨RA, hPA, ?_⟩
  have hsim := ih k lb S na a Sx nx A Γa w vals RA hfuel hsc hAa hAi hPA hDD
    (fun ht => ⟨_, hPD ht⟩) hs hr
  have hnax : na ≤ nx := (scGo_slots f lb S na a Sx nx hsc).1
  generalize hres : Sem.runCode k cfg Γa w a = r at hsim
  cases r with
  | stuck _ => trivial
  | brk d Γb vs w' =>
      obtain ⟨L, hL, hlen, vals', c, hrel, hf, heq⟩ := hsim
      exact ⟨L, hL, hlen, vals', c, hrel, hf.mono hnle, heq⟩
  | cont d vs w' =>
      obtain ⟨L, hL, hlen, vals', c, hf, heq⟩ := hsim
      exact ⟨L, hL, hlen, vals', c, hf.mono hnle, heq⟩
  | ok Γ' w' =>
      obtain ⟨hsz, vals', c, hrel, hf, heq⟩ := hsim
      have hterm : termsGo f a = false := by
        cases ht : termsGo f a
        · rfl
        · exact absurd hres (termsGo_run f a ht k cfg Γa w Γ' w')
      obtain ⟨hin, hlen, hsub⟩ : allIn Sx nx rs = true ∧ rs.length = jTys.length ∧
          S.sub Sx = true := by
        rcases hx with h' | h'
        · rw [hterm] at h'; cases h'
        · exact h'
      show SimRes _ _ _ _ _ _ _ _ _ _ _
        (match rs.mapM (fun r => Γ'[r]?) with
          | none => .stuck "branch export is not in scope"
          | some vs => .ok (Sem.bindAt Γ' ne vs) w')
      cases hm : rs.mapM (fun r => Γ'[r]?) with
      | none => trivial
      | some vs =>
        have hvl : vs.length = jTys.length := (mapM_length hm).trans hlen
        obtain ⟨hr1, hc1, hp1⟩ := ht1 hterm
        have hSlt : ∀ i, S.mem i = true → i < ne := by
          intro i hi; have := hr.lt hi; omega
        have hrel' := bind_rel (hrel.sub hsub) hSlt (tys := jTys) hvl.symm
        rw [hvl] at hrel'
        refine ⟨by rw [bindAt_size, hvl], _, c + 1, hrel',
          (hf.mono hnle).trans ((bindVals_frame jTys vs ne vals').mono (by omega)),
          fun R1 hP1 st => ?_⟩
        rw [show st + (c + 1) = (st + 1) + c by omega, heq _ (hPD hterm) (st + 1),
          runK_jump ((args_read hDa hrel hin).trans hm), ← hr1]
        exact enter hF hP1 hc1 hp1 hvl.symm st ⟨vals', w'⟩

theorem sim_ite {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} (hF : Distinct F) (f : Nat)
    (hfuel : f ≤ HProg.fuel) (ih : SimP env cfg F f) {lb : List SLbl} {S : Scope} {n : Nat}
    {m : IteMeta} {thn els : List Piece} {thnR elsR : List R} {St : Scope} {nt : Nat}
    {Se : Scope} {ne : Nat} (hok : IteOk f lb S n m thn els thnR elsR St nt Se ne) (k : Nat)
    (s : CS) (ha : Aligned s n) (hi : Inv s) (Γ : Sem.Env) (w : World) (vals : Blocks.Vals)
    (R0 : List Inst) (hP : Placed F s R0) (hD : Done F (emitPiece f s (.ite m thn els thnR elsR)))
    (hs : Γ.size = n) (hr : Rel S vals Γ) :
    SimRes env F lb s.labels n (S.add ne (ne + m.jTys.length)) (ne + m.jTys.length)
      (emitPiece f s (.ite m thn els thnR elsR)) vals w R0
      (Sem.runPiece k cfg Γ w (.ite m thn els thnR elsR)) := by
  obtain ⟨hCi, hCa, hFi, hFa, hCE, hEH, _, hFc, hFp, hFr, hFl, hHc, hHa⟩ :=
    ite_stages f (emit_struct f) hok s ha hi
  obtain ⟨hCd, _, hCr, hCc, hCp, hCl, _⟩ := iteThen_facts s m
  rw [emitPiece_ite] at hD ⊢
  have hJd : (iteJoin f s.nextBlk m thn els (iteH f s m thn els thnR elsR)).done
      = (iteH f s m thn els thnR elsR).done := by
    unfold iteJoin; split
    · rfl
    · exact (open'_fields _ _ _ _).1
  have hDH : Done F (iteH f s m thn els thnR elsR) :=
    hD.mono (by rw [hJd]; exact List.prefix_refl _)
  have hDE : Done F (iteE f s m thn thnR) := hDH.mono hEH
  have hDC : Done F (iteThen s m) := hDE.mono hCE
  have hP0 : Placed F s [.brif (s.get m.flag) ⟨s.nextBlk⟩ [] ⟨s.nextBlk + 1⟩ []] :=
    ⟨{ ref := ⟨s.curRef⟩, params := s.curPars,
       insts := (Inst.brif (s.get m.flag) ⟨s.nextBlk⟩ [] ⟨s.nextBlk + 1⟩ [] :: s.cur).reverse },
     hDC _ (by rw [hCd]; simp), rfl, rfl, by simp⟩
  have := hP.unique hF hP0; subst this
  have hjoin : (termsGo f thn && termsGo f els) = false →
      (iteJoin f s.nextBlk m thn els (iteH f s m thn els thnR elsR)).curRef = s.nextBlk + 2 ∧
      (iteJoin f s.nextBlk m thn els (iteH f s m thn els thnR elsR)).cur = [] ∧
      (iteJoin f s.nextBlk m thn els (iteH f s m thn els thnR elsR)).curPars
        = parsOf ne m.jTys := by
    intro hb
    unfold iteJoin
    rw [if_neg (by rw [hb]; simp)]
    obtain ⟨_, _, cr, cu, cp, _, _⟩ := open'_fields (iteH f s m thn els thnR elsR)
      (s.nextBlk + 2) m.jTys (iteH f s m thn els thnR elsR).slots
    show (CS.open' _ _ _ _).curRef = _ ∧ (CS.open' _ _ _ _).cur = _ ∧
      (CS.open' _ _ _ _).curPars = _
    rw [cr, cu, cp, hHc, hHa.1]
    exact ⟨rfl, rfl, rfl⟩
  have hsl1 := scGo_slots f lb S n thn St nt hok.hthn
  have hsl2 := scGo_slots f lb S nt els Se ne hok.hels
  have hnt : Sem.slotsOf n thn = nt := (hsl1.2 HProg.fuel hfuel).1
  have hne : Sem.slotsOf nt els = ne := (hsl2.2 HProg.fuel hfuel).1
  obtain ⟨hfl1, hfl2⟩ := (inS_iff S n m.flag).mp hok.hflag
  obtain ⟨x, hxΓ, hxv⟩ := hr _ hfl2
  rw [get_of_aligned s n ha _ hfl1]
  cases k with
  | zero => trivial
  | succ k =>
  cases x with
  | vec t ls => simp only [Sem.runPiece, Sem.get, hxΓ]; trivial
  | sc t fv =>
    by_cases hfv : (fv != 0) = true
    · simp only [Sem.runPiece, Sem.get, hxΓ, hfv, if_true]
      rw [hs, hnt, hne]
      obtain ⟨RA, hPA, hsim⟩ := sim_arm hF f hfuel ih hok.hthn hok.hthnX (Nat.le_refl n)
        hsl2.1 k (iteThen s m) hCa hCi s.nextBlk _ hDE (fun ht => hjoin (by simp [ht]))
        Γ w vals hs hr
      rw [hCl] at hsim
      refine hsim.prepend (c := 1) (fun st => ?_) (Frame.refl n vals) (Nat.le_refl n)
      rw [runK_brif (ta := []) (ea := []) (vs := []) hxv (by simp), show Sem.isTrue (.sc t fv) = true from hfv, if_pos rfl]
      rw [← hCr]
      exact enter hF hPA hCc (m := 0) (tys := []) (vs := []) hCp rfl st ⟨vals, w⟩
    · simp only [Bool.not_eq_true] at hfv
      simp only [Sem.runPiece, Sem.get, hxΓ, hfv, Bool.false_eq_true, if_false]
      rw [hs, hnt, hne]
      have hSlt : ∀ i, S.mem i = true → i < nt := by
        intro i hi; have := hr.lt hi; omega
      have hr0 : Rel S vals (Sem.bindAt Γ nt []) :=
        (bind_rel hr hSlt (tys := []) (vs := []) rfl).of_add
      obtain ⟨RA, hPA, hsim⟩ := sim_arm hF f hfuel ih hok.hels hok.helsX hsl1.1
        (Nat.le_refl ne) k (iteF f s m thn thnR) hFa hFi s.nextBlk _ hDH
        (fun ht => hjoin (by simp [ht])) (Sem.bindAt Γ nt []) w vals
        (by rw [bindAt_size]; rfl) hr0
      rw [hFl] at hsim
      refine hsim.prepend (c := 1) (fun st => ?_) (Frame.refl n vals) (Nat.le_refl n)
      rw [runK_brif (ta := []) (ea := []) (vs := []) hxv (by simp), show Sem.isTrue (.sc t fv) = false from hfv]
      simp only [Bool.false_eq_true, if_false]
      rw [show s.nextBlk + 1 = (iteF f s m thn thnR).curRef from hFr.symm]
      exact enter hF hPA hFc (m := 0) (tys := []) (vs := []) hFp rfl st ⟨vals, w⟩

/-- The states `emitLoop` passes through, by name. -/
def loopD (f : Nat) (s : CS) (l : Loop) (pre : List Piece) : CS := emitCode f (loopHead s l) pre
def loopB (f : Nat) (s : CS) (l : Loop) (pre : List Piece) : CS :=
  loopBodyStart s.nextBlk l (loopD f s l pre) s.slots
def loopG (f : Nat) (s : CS) (l : Loop) (pre body : List Piece) : CS :=
  emitCode f (loopB f s l pre) body
def loopH (f : Nat) (s : CS) (l : Loop) (pre body : List Piece) : CS :=
  loopBodyEnd f s.nextBlk l body (loopG f s l pre body)

theorem emitPiece_loop (f : Nat) (s : CS) (l : Loop) (pre body : List Piece) :
    emitPiece f s (.loop l pre body) = loopExit s.nextBlk l s.labels (loopH f s l pre body) := by
  rw [emitPiece, emitLoop_eq]; rfl

theorem loop_stages (f : Nat) (ih : StructP f) {lb : List SLbl} {S : Scope} {n : Nat}
    {l : Loop} {pre body : List Piece} {Sp : Scope} {np : Nat} {Sb : Scope} {nb : Nat}
    (hok : LoopOk f lb S n l pre body Sp np Sb nb) (s : CS) (ha : Aligned s n) (hi : Inv s) :
    Inv (loopHead s l) ∧ Aligned (loopHead s l) (n + l.pTys.length) ∧
    Aligned (loopD f s l pre) np ∧ (loopHead s l).done <+: (loopD f s l pre).done ∧
    (loopD f s l pre).labels = (loopHead s l).labels ∧
    Inv (loopB f s l pre) ∧ Aligned (loopB f s l pre) (np + l.pTys.length) ∧
    Aligned (loopG f s l pre body) nb ∧
    (loopB f s l pre).done <+: (loopG f s l pre body).done ∧
    (loopG f s l pre body).done <+: (loopH f s l pre body).done ∧
    (loopH f s l pre body).cur = [] ∧ Aligned (loopH f s l pre body) nb := by
  simp only [loopH, loopG, loopB, loopD]
  obtain ⟨hCd, hCn, hCr, _, _, _, _⟩ := loopHead_facts s l
  -- the head
  have hCa : Aligned (loopHead s l) (n + l.pTys.length) :=
    aligned_open (aligned_congr (u := (({ s with nextBlk := s.nextBlk + 3 } : CS).close
      (.jump ⟨s.nextBlk⟩ (l.init.map s.get)))) ha rfl rfl rfl) s.nextBlk l.pTys _
  have hCids : ∀ x ∈ ids (loopHead s l), x < s.nextBlk := by
    rw [ids_of_done hCd]; intro x hx
    simp only [List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | hx
    · exact hi.lt x hx
    · rw [hx]; exact hi.cur
  have hCo : Out s (loopHead s l) false := by
    have h0 := ((Out.refl hi).bump 3).close (.jump ⟨s.nextBlk⟩ (l.init.map s.get))
    refine h0.reopen ?_ ?_ ?_ ?_ ?_
    · rw [hCd]; rfl
    · rw [hCn]; rfl
    · rw [hCr]; intro hm
      rw [ids_close] at hm
      simp only [List.mem_append, List.mem_singleton] at hm
      rcases hm with hm | hm
      · have := hi.lt _ hm; omega
      · have := hi.cur; omega
    · rw [hCr]; show s.nextBlk < s.nextBlk + 3; omega
    · rw [hCr]; exact Nat.le_refl _
  have hCav : ∀ r, s.nextBlk + 1 ≤ r → r < s.nextBlk + 3 →
      (∀ x ∈ ids (loopHead s l), x ≠ r) ∧ (loopHead s l).curRef ≠ r ∧
      r < (loopHead s l).nextBlk := by
    intro r h1 h2
    refine ⟨fun x hx => ?_, by rw [hCr]; omega, by rw [hCn]; omega⟩
    have := hCids x hx; omega
  -- the prefix
  obtain ⟨hDa, hDl, hDo, _, _⟩ := ih _ _ _ pre Sp np (loopHead s l) hok.hpre hCa hCo.inv
  rw [hok.hpreT] at hDo
  generalize hD : emitCode f (loopHead s l) pre = D at hDa hDl hDo
  have hDav1 := out_avoid hDo (hCav _ (Nat.le_refl _) (by omega)).1
    (hCav _ (Nat.le_refl _) (by omega)).2.1 (hCav _ (Nat.le_refl _) (by omega)).2.2
  have hDav2 := out_avoid hDo (hCav (s.nextBlk + 2) (by omega) (by omega)).1
    (hCav (s.nextBlk + 2) (by omega) (by omega)).2.1 (hCav (s.nextBlk + 2) (by omega) (by omega)).2.2
  have hDnb : s.nextBlk + 3 ≤ D.nextBlk := hCn ▸ hDo.grows.1
  -- the body block
  obtain ⟨hFd, hFn, hFr, _, _, _, _⟩ := loopBodyStart_facts s.nextBlk l D s.slots
  have hFa : Aligned (loopBodyStart s.nextBlk l D s.slots) (np + l.pTys.length) :=
    aligned_open_nl (aligned_congr (u := D.close (loopBrif s.nextBlk l D s.slots)) hDa rfl rfl rfl)
      (s.nextBlk + 1) l.pTys
  have hCFo : Out s (loopBodyStart s.nextBlk l D s.slots) false := by
    refine ((hCo.trans hDo).close (loopBrif s.nextBlk l D s.slots)).reopen ?_ ?_ ?_ ?_ ?_
    · rw [hFd]; rfl
    · rw [hFn]; rfl
    · rw [hFr]; intro hm
      exact avoid_close hDav1.1 hDav1.2 _ _ hm rfl
    · rw [hFr]; show s.nextBlk + 1 < D.nextBlk; omega
    · rw [hFr]; omega
  have hFav2 : (∀ x ∈ ids (loopBodyStart s.nextBlk l D s.slots), x ≠ s.nextBlk + 2) ∧
      (loopBodyStart s.nextBlk l D s.slots).curRef ≠ s.nextBlk + 2 ∧
      s.nextBlk + 2 < (loopBodyStart s.nextBlk l D s.slots).nextBlk := by
    refine ⟨?_, ?_, ?_⟩
    · intro x hx
      rw [ids_of_done hFd] at hx
      exact avoid_close hDav2.1 hDav2.2 (loopBrif s.nextBlk l D s.slots) x
        (by rw [ids_close]; exact hx)
    · rw [hFr]; omega
    · rw [hFn]; omega
  -- the body
  obtain ⟨hGa, hGl, hGo, _, hGt⟩ :=
    ih _ _ _ body Sb nb (loopBodyStart s.nextBlk l D s.slots) hok.hbody hFa hCFo.inv
  generalize hG : emitCode f (loopBodyStart s.nextBlk l D s.slots) body = G at hGa hGl hGo hGt
  have hGav := out_avoid hGo hFav2.1 hFav2.2.1 hFav2.2.2
  have hGnb : s.nextBlk + 3 ≤ G.nextBlk := by
    have := hGo.grows.1; rw [hFn] at this; omega
  -- the back edge, and the exit
  have hCHo : Out s (loopBodyEnd f s.nextBlk l body G) true ∧
      G.done <+: (loopBodyEnd f s.nextBlk l body G).done ∧
      (∀ x ∈ ids (loopBodyEnd f s.nextBlk l body G), x ≠ s.nextBlk + 2) ∧
      (loopBodyEnd f s.nextBlk l body G).nextBlk = G.nextBlk ∧
      (loopBodyEnd f s.nextBlk l body G).slots = G.slots ∧
      (loopBodyEnd f s.nextBlk l body G).nextVal = G.nextVal ∧
      (loopBodyEnd f s.nextBlk l body G).env = G.env ∧
      (loopBodyEnd f s.nextBlk l body G).cur = [] := by
    unfold loopBodyEnd
    split
    · rename_i htb
      rw [htb] at hGo
      exact ⟨hCFo.trans hGo, List.prefix_refl _, hGav.1, rfl, rfl, rfl, rfl, (hGt htb).2⟩
    · rename_i htb
      simp only [Bool.not_eq_true] at htb
      rw [htb] at hGo
      exact ⟨(hCFo.trans hGo).close _, close_done_prefix G _, avoid_close hGav.1 hGav.2 _,
        rfl, rfl, rfl, rfl, rfl⟩
  obtain ⟨hHo, hGH, hHav, hHn, hHs, hHv, hHe, hHc⟩ := hCHo
  generalize hH : loopBodyEnd f s.nextBlk l body G = H at hHo hGH hHav hHn hHs hHv hHe hHc
  obtain ⟨hId, hIn, hIr, _, _, hIl, _⟩ := loopExit_facts s.nextBlk l s.labels H
  have hIo : Out s (loopExit s.nextBlk l s.labels H) false := by
    refine hHo.reopen hId hIn ?_ ?_ ?_
    · rw [hIr]; intro hm; exact hHav _ hm rfl
    · rw [hIr, hHn]; omega
    · rw [hIr]; omega
  subst hH hG hD
  exact ⟨hCo.inv, hCa, hDa, hDo.grows.2, hDl, hCFo.inv, hFa, hGa, hGo.grows.2, hGH, hHc,
    aligned_congr hGa hHv hHs hHe⟩

/-- `SimRes` with the blocks' run given as a function of the budget, so a loop
    trip can start at a block entry rather than part-way through a block. -/
def SimResS (env : FnEnv) (F : FuncData) (lb : List SLbl) (labels : List (Nat × Option Nat))
    (n : Nat) (S' : Scope) (n' : Nat) (t : CS) (vals0 : Blocks.Vals)
    (start : Nat → Outcome World) : Sem.CodeRes → Prop
  | .ok Γ' w' => Γ'.size = n' ∧ ∃ vals' cost, Rel S' vals' Γ' ∧ Frame n vals0 vals' ∧
      ∀ R1, Placed F t R1 → ∀ st, start (st + cost) = runK env F st ⟨vals', w'⟩ R1
  | .brk d Γb vs w' => ∃ L, lb[d]? = some L ∧ vs.length = L.exitN ∧ ∃ vals' cost,
      Rel L.need vals' Γb ∧ Frame n vals0 vals' ∧ ∀ st,
        start (st + cost)
          = Blocks.runFrom env F st ⟨vals', w'⟩ (((labels[d]?).map (·.1)).getD 1000000) vs
  | .cont d vs w' => ∃ L, lb[d]? = some L ∧ vs.length = L.carryN ∧ ∃ vals' cost,
      Frame n vals0 vals' ∧ ∀ st,
        start (st + cost)
          = Blocks.runFrom env F st ⟨vals', w'⟩ (((labels[d]?).bind (·.2)).getD 1000000) vs
  | .stuck _ => True

theorem SimResS.restart {env : FnEnv} {F : FuncData} {lb : List SLbl}
    {labels : List (Nat × Option Nat)} {n : Nat} {S' : Scope} {n' : Nat} {t : CS}
    {vals0 : Blocks.Vals} {start start1 : Nat → Outcome World} {res : Sem.CodeRes} {c : Nat}
    (h : SimResS env F lb labels n S' n' t vals0 start1 res)
    (hc : ∀ st, start (st + c) = start1 st) : SimResS env F lb labels n S' n' t vals0 start res := by
  cases res with
  | ok Γ' w' =>
      obtain ⟨hs, vals', c2, hr, hf, heq⟩ := h
      exact ⟨hs, vals', c2 + c, hr, hf, fun R1 hP st => by rw [← Nat.add_assoc, hc, heq R1 hP]⟩
  | brk d Γb vs w' =>
      obtain ⟨L, hL, hlen, vals', c2, hr, hf, heq⟩ := h
      exact ⟨L, hL, hlen, vals', c2 + c, hr, hf, fun st => by rw [← Nat.add_assoc, hc, heq]⟩
  | cont d vs w' =>
      obtain ⟨L, hL, hlen, vals', c2, hf, heq⟩ := h
      exact ⟨L, hL, hlen, vals', c2 + c, hf, fun st => by rw [← Nat.add_assoc, hc, heq]⟩
  | stuck _ => trivial

theorem runK_brif' {env : FnEnv} {F : FuncData} {steps : Nat} {vals : Blocks.Vals} {w : World}
    {c : Val} {tb eb : BlockRef} {ta ea : List Val} {rest : List Inst} {cv : V}
    {tgt : BlockRef} {args : List Val} {vs : List V}
    (hc : Blocks.getV vals c = some cv)
    (htgt : (if Sem.isTrue cv then (tb, ta) else (eb, ea)) = (tgt, args))
    (hargs : args.mapM (Blocks.getV vals) = some vs) :
    runK env F steps ⟨vals, w⟩ (.brif c tb ta eb ea :: rest)
      = Blocks.runFrom env F steps ⟨vals, w⟩ tgt.id vs := by
  unfold runK; simp only [Blocks.runInsts, hc, htgt, hargs]

theorem mapM_of_get {α β : Type} {g : α → Option β} : ∀ (xs : List α) (ys : List β),
    xs.length = ys.length → (∀ i (h1 : i < xs.length) (h2 : i < ys.length), g xs[i] = some ys[i]) →
    xs.mapM g = some ys := by
  intro xs
  induction xs with
  | nil => intro ys hl _; cases ys <;> simp_all
  | cons x xs ih =>
      intro ys hl h
      cases ys with
      | nil => simp at hl
      | cons y ys =>
        simp only [List.length_cons, Nat.add_right_cancel_iff] at hl
        have h0 := h 0 (by simp) (by simp)
        simp only [List.getElem_cons_zero] at h0
        have hr := ih ys hl (fun i h1 h2 => by simpa using h (i + 1) (by simp; omega) (by simp; omega))
        simp [List.mapM_cons, h0, hr]

/-- The carries a loop's head passes on to its body are its own parameters,
    read back. -/
theorem carry_read {vals : Blocks.Vals} {n len : Nat} {cs : List V} (hl : cs.length = len)
    (h : ∀ i, i < len → Blocks.getV vals ⟨n + i⟩ = cs[i]?) :
    ((List.range len).map (fun i => (⟨n + i⟩ : Val))).mapM (Blocks.getV vals) = some cs := by
  refine mapM_of_get _ _ (by simp [hl]) (fun i h1 h2 => ?_)
  simp only [List.getElem_map, List.getElem_range]
  rw [h i (by simpa using h1)]; simp [h2]

/-- The head's test, taken toward the exit. -/
theorem loopBrif_exit {env : FnEnv} {F : FuncData} {st : Nat} {vals : Blocks.Vals} {w : World}
    {h : Nat} {l : Loop} {sH : CS} {fc : Nat} {t : ClifTy} {fv : UInt64} {vs : List V}
    (hflag : Blocks.getV vals (sH.get l.flag) = some (.sc t fv))
    (hcond : ((fv != 0) == l.exitOnTrue) = true)
    (hargs : (l.exitR.map sH.get).mapM (Blocks.getV vals) = some vs) :
    runK env F st ⟨vals, w⟩ [loopBrif h l sH fc] = Blocks.runFrom env F st ⟨vals, w⟩ (h + 2) vs := by
  unfold loopBrif
  refine runK_brif' (tgt := ⟨h + 2⟩) (args := l.exitR.map sH.get) hflag ?_ hargs
  cases hE : l.exitOnTrue <;> cases hf : (fv != 0) <;> simp_all [Sem.isTrue]

/-- The head's test, taken toward the body. -/
theorem loopBrif_body {env : FnEnv} {F : FuncData} {st : Nat} {vals : Blocks.Vals} {w : World}
    {h : Nat} {l : Loop} {sH : CS} {fc : Nat} {t : ClifTy} {fv : UInt64} {cs : List V}
    (hflag : Blocks.getV vals (sH.get l.flag) = some (.sc t fv))
    (hcond : ((fv != 0) == l.exitOnTrue) = false)
    (hargs : ((List.range l.pTys.length).map (fun i => sH.get (fc + i))).mapM (Blocks.getV vals)
      = some cs) :
    runK env F st ⟨vals, w⟩ [loopBrif h l sH fc] = Blocks.runFrom env F st ⟨vals, w⟩ (h + 1) cs := by
  unfold loopBrif
  refine runK_brif' (tgt := ⟨h + 1⟩)
    (args := (List.range l.pTys.length).map (fun i => sH.get (fc + i))) hflag ?_ hargs
  cases hE : l.exitOnTrue <;> cases hf : (fv != 0) <;> simp_all [Sem.isTrue]

/-- **Every trip of a top-tested loop simulates**: entering the head with the
    carries, the blocks do whatever `iter` does, however many trips it takes. By
    induction on the term's fuel, which bounds the trips; the regions inside
    come from the simulation one emitter fuel down. -/
theorem loop_trip {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} (hF : Distinct F) (f : Nat)
    (hfuel : f ≤ HProg.fuel) (ih : SimP env cfg F f) {lb : List SLbl} {S : Scope} {n : Nat}
    {l : Loop} {pre body : List Piece} {Sp : Scope} {np : Nat} {Sb : Scope} {nb : Nat}
    (hok : LoopOk f lb S n l pre body Sp np Sb nb) (s : CS) (ha : Aligned s n) (hi : Inv s)
    (hD : Done F (emitPiece f s (.loop l pre body))) (Γ : Sem.Env) (hs : Γ.size = n)
    (vals0 : Blocks.Vals) (hr0 : Rel S vals0 Γ) :
    ∀ (k : Nat) (vals : Blocks.Vals) (w : World) (carries : List V),
      Frame n vals0 vals → carries.length = l.pTys.length →
      SimResS env F lb s.labels n
        ((S.add n (n + l.pTys.length)).add nb (nb + l.exitTys.length))
        (nb + l.exitTys.length) (emitPiece f s (.loop l pre body)) vals0
        (fun st => Blocks.runFrom env F st ⟨vals, w⟩ s.nextBlk carries)
        (Sem.iter k cfg Γ w l pre body n nb carries) := by
  obtain ⟨hCi, hCa, hDa, hCD, hDl, hBi, hBa, hGa, hBG, hGH, hHc, hHa⟩ :=
    loop_stages f (emit_struct f) hok s ha hi
  obtain ⟨_, _, hCr, hCc, hCp, hCl, _⟩ := loopHead_facts s l
  obtain ⟨hBd, _, hBr, hBc, hBp, hBl, _⟩ := loopBodyStart_facts s.nextBlk l (loopD f s l pre) s.slots
  rw [emitPiece_loop] at hD ⊢
  obtain ⟨hId, _, hIr, hIc, hIp, _, _⟩ := loopExit_facts s.nextBlk l s.labels (loopH f s l pre body)
  -- Everything emitted is in `F`.
  have hDH : Done F (loopH f s l pre body) := hD.mono (by rw [hId]; exact List.prefix_refl _)
  have hDG : Done F (loopG f s l pre body) := hDH.mono hGH
  have hDB : Done F (loopB f s l pre) := hDG.mono hBG
  have hDD : Done F (loopD f s l pre) :=
    hDB.mono (by show _ <+: (loopBodyStart _ _ _ _).done; rw [hBd]; exact List.prefix_append _ _)
  -- The blocks, and what follows each state.
  have hPD : Placed F (loopD f s l pre) [loopBrif s.nextBlk l (loopD f s l pre) s.slots] :=
    ⟨{ ref := ⟨(loopD f s l pre).curRef⟩, params := (loopD f s l pre).curPars,
       insts := (loopBrif s.nextBlk l (loopD f s l pre) s.slots :: (loopD f s l pre).cur).reverse },
     hDB _ (by show _ ∈ (loopBodyStart _ _ _ _).done; rw [hBd]; simp), rfl, rfl, by simp⟩
  obtain ⟨_, _, _, hDx, _⟩ := emit_struct f _ _ _ pre Sp np (loopHead s l) hok.hpre hCa hCi
  obtain ⟨RC, hPC⟩ : ∃ RC, Placed F (loopHead s l) RC := by
    rcases hDx with ho | hc
    · exact ho.placed hPD
    · exact hc.placed hDD
  have hPG : termsGo f body = false →
      Placed F (loopG f s l pre body) [.jump ⟨s.nextBlk⟩ (l.cont.map (loopG f s l pre body).get)] := by
    intro ht
    have e : loopH f s l pre body = (loopG f s l pre body).close
        (.jump ⟨s.nextBlk⟩ (l.cont.map (loopG f s l pre body).get)) := by
      simp only [loopH, loopBodyEnd, ht]; rfl
    rw [e] at hDH; exact placed_close hDH
  obtain ⟨_, hGl, _, hGx, hGt⟩ := emit_struct f _ _ _ body Sb nb (loopB f s l pre) hok.hbody hBa hBi
  obtain ⟨RB, hPB⟩ : ∃ RB, Placed F (loopB f s l pre) RB := by
    rcases hGx with ho | hc
    · cases ht : termsGo f body
      · exact ho.placed (hPG ht)
      · exact (hGt ht).1.placed hDG
    · exact hc.placed hDG
  -- Counts.
  have hsl1 := scGo_slots f _ _ _ pre Sp np hok.hpre
  have hsl2 := scGo_slots f _ _ _ body Sb nb hok.hbody
  have hnp : Sem.slotsOf (n + l.pTys.length) pre = np := (hsl1.2 HProg.fuel hfuel).1
  have hSn : ∀ i, S.mem i = true → i < n := fun i hi => by have := hr0.lt hi; omega
  have hneedlt : ∀ i, (S.add n (n + l.pTys.length)).mem i = true → i < nb := by
    intro i hi
    rcases (Scope.mem_add _ _ _ i).mp hi with h1 | h1
    · have := hSn i h1; omega
    · omega
  have hsn : s.slots = n := ha.2.1
  -- Entering the three blocks.
  have enterC : ∀ st v w cs, cs.length = l.pTys.length →
      Blocks.runFrom env F (st + 1) ⟨v, w⟩ s.nextBlk cs
        = runK env F st ⟨bindVals v (parsOf n l.pTys) cs, w⟩ RC := by
    intro st v w cs hl
    rw [← hCr]
    exact enter hF hPC hCc (by rw [hCp, ha.1]) hl.symm st ⟨v, w⟩
  have enterB : ∀ st v w cs, cs.length = l.pTys.length →
      Blocks.runFrom env F (st + 1) ⟨v, w⟩ (s.nextBlk + 1) cs
        = runK env F st ⟨bindVals v (parsOf np l.pTys) cs, w⟩ RB := by
    intro st v w cs hl
    have e1 : (loopB f s l pre).curRef = s.nextBlk + 1 := hBr
    rw [← e1]
    exact enter hF hPB hBc (by show (loopBodyStart _ _ _ _).curPars = _; rw [hBp, hDa.1])
      hl.symm st ⟨v, w⟩
  -- Leaving by the exit block.
  have hexit : ∀ (vals' : Blocks.Vals) (w' : World) (Γb : Sem.Env) (vs : List V)
      (start : Nat → Outcome World) (c : Nat),
      Rel (S.add n (n + l.pTys.length)) vals' Γb → Frame n vals0 vals' →
      vs.length = l.exitTys.length →
      (∀ st, start (st + c) = Blocks.runFrom env F st ⟨vals', w'⟩ (s.nextBlk + 2) vs) →
      SimResS env F lb s.labels n
        ((S.add n (n + l.pTys.length)).add nb (nb + l.exitTys.length))
        (nb + l.exitTys.length) (loopExit s.nextBlk l s.labels (loopH f s l pre body)) vals0
        start (.ok (Sem.bindAt Γb nb vs) w') := by
    intro vals' w' Γb vs start c hrel hf hvl hst
    have hrel' := bind_rel hrel hneedlt (tys := l.exitTys) hvl.symm
    rw [hvl] at hrel'
    refine ⟨by rw [bindAt_size, hvl], _, c + 1, hrel',
      hf.trans ((bindVals_frame _ _ _ _).mono (by omega)), fun R1 hP1 st => ?_⟩
    rw [show st + (c + 1) = (st + 1) + c by omega, hst, ← hIr]
    exact enter hF hP1 (by rw [hIc, hHc]) (by rw [hIp, hHa.1]) hvl.symm st ⟨vals', w'⟩
  intro k
  induction k with
  | zero => intro vals w carries _ _; simp only [Sem.iter]; trivial
  | succ k ihk =>
    intro vals w carries hfr hlen
    have hrS : Rel S vals Γ := fun i hi => by
      obtain ⟨x, hx, hxv⟩ := hr0 i hi
      exact ⟨x, hx, hfr i (hSn i hi) x hxv⟩
    have hrh : Rel (S.add n (n + l.pTys.length)) (bindVals vals (parsOf n l.pTys) carries)
        (Sem.bindAt Γ n carries) := by
      have := bind_rel hrS hSn (tys := l.pTys) hlen.symm
      rwa [hlen] at this
    have hfh : Frame n vals0 (bindVals vals (parsOf n l.pTys) carries) :=
      hfr.trans (bindVals_frame _ _ _ _)
    have hcarry : ∀ i, i < l.pTys.length →
        Blocks.getV (bindVals vals (parsOf n l.pTys) carries) ⟨n + i⟩ = carries[i]? :=
      fun i hi => bindVals_hi _ _ _ _ hlen.symm i (by omega)
    have hpre := ih k _ _ _ pre Sp np (loopHead s l) (Sem.bindAt Γ n carries) w
      (bindVals vals (parsOf n l.pTys) carries) RC hfuel hok.hpre hCa hCi hPC hDD
      (fun _ => ⟨_, hPD⟩) (by rw [bindAt_size, hlen]) hrh
    rw [hCl] at hpre
    simp only [Sem.iter]
    generalize Sem.runCode k cfg (Sem.bindAt Γ n carries) w pre = r at hpre
    cases r with
    | stuck _ => trivial
    | brk d Γb vs w' =>
      obtain ⟨L, hL, hvl, vals', c, hrel, hf, heq⟩ := hpre
      cases d with
      | zero =>
        simp only [List.getElem?_cons_zero, Option.some.injEq] at hL
        subst hL
        refine hexit vals' w' Γb vs _ (c + 1) hrel (hfh.trans (hf.mono (by omega))) hvl
          (fun st => ?_)
        rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq]; rfl
      | succ d =>
        simp only [List.getElem?_cons_succ] at hL heq
        exact ⟨L, hL, hvl, vals', c + 1, hrel, hfh.trans (hf.mono (by omega)), fun st => by
          dsimp only
          rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq]⟩
    | cont d vs w' =>
      obtain ⟨L, hL, hvl, vals', c, hf, heq⟩ := hpre
      cases d with
      | zero =>
        simp only [List.getElem?_cons_zero, Option.some.injEq] at hL
        subst hL
        refine (ihk vals' w' vs (hfh.trans (hf.mono (by omega))) hvl).restart (c := c + 1)
          (fun st => ?_)
        rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq]; rfl
      | succ d =>
        simp only [List.getElem?_cons_succ] at hL heq
        exact ⟨L, hL, hvl, vals', c + 1, hfh.trans (hf.mono (by omega)), fun st => by
          dsimp only
          rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq]⟩
    | ok Γ1 w1 =>
      obtain ⟨hs1, vals1, c, hrel1, hf1, heq1⟩ := hpre
      have heq1' := heq1 _ hPD
      obtain ⟨hfl1, hfl2⟩ := (inS_iff Sp np l.flag).mp hok.hflag
      obtain ⟨x, hxΓ, hxv⟩ := hrel1 _ hfl2
      rw [← get_of_aligned _ np hDa _ hfl1] at hxv
      simp only [Sem.get, hxΓ]
      cases x with
      | vec _ _ => trivial
      | sc t fv =>
        simp only
        split
        · -- the test says leave
          rename_i hcond
          cases hm : l.exitR.mapM (fun r => Γ1[r]?) with
          | none =>
            show SimResS _ _ _ _ _ _ _ _ _ _
              (match l.exitR.mapM (Sem.get Γ1) with
                | none => .stuck "loop exit value is not in scope"
                | some vs => .ok (Sem.bindAt Γ1 nb vs) w1)
            rw [show l.exitR.mapM (Sem.get Γ1) = l.exitR.mapM (fun r => Γ1[r]?) from rfl, hm]
            trivial
          | some vs =>
            show SimResS _ _ _ _ _ _ _ _ _ _
              (match l.exitR.mapM (Sem.get Γ1) with
                | none => .stuck "loop exit value is not in scope"
                | some vs => .ok (Sem.bindAt Γ1 nb vs) w1)
            rw [show l.exitR.mapM (Sem.get Γ1) = l.exitR.mapM (fun r => Γ1[r]?) from rfl, hm]
            have hargs := (args_read hDa hrel1 hok.hexitR).trans hm
            refine hexit vals1 w1 Γ1 vs _ (c + 1) (hrel1.sub hok.hneed)
              (hfh.trans (hf1.mono (by omega))) ((mapM_length hm).trans hok.hexitN) (fun st => ?_)
            rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq1',
              loopBrif_exit hxv hcond hargs]
        · -- the test says go round: the body
          rename_i hcond
          simp only [Bool.not_eq_true] at hcond
          rw [hnp]
          have hcs : ((List.range l.pTys.length).map
              (fun i => (loopD f s l pre).get (s.slots + i))).mapM (Blocks.getV vals1)
                = some carries := by
            rw [List.map_congr_left (g := fun i => (⟨n + i⟩ : Val)) (fun i hi => by
              rw [hsn]; exact get_of_aligned _ np hDa _ (by simp at hi; omega))]
            exact carry_read hlen (fun i hi => by
              have h1 := hcarry i hi
              cases hc : carries[i]? with
              | none => simp at hc; omega
              | some v => rw [hc] at h1; exact hf1 _ (by omega) v h1)
          have hsb : (Sem.bindAt Γ1 np carries).size = np + l.pTys.length := by
            rw [bindAt_size, hlen]
          have hrb : Rel (Sp.add np (np + l.pTys.length)) (bindVals vals1 (parsOf np l.pTys) carries)
              (Sem.bindAt Γ1 np carries) := by
            have := bind_rel hrel1 (m := np) (fun i hi => by have := hrel1.lt hi; omega)
              (tys := l.pTys) hlen.symm
            rwa [hlen] at this
          have hfb : Frame n vals0 (bindVals vals1 (parsOf np l.pTys) carries) :=
            hfh.trans ((hf1.mono (by omega)).trans ((bindVals_frame _ _ _ _).mono (by omega)))
          have hpfx : ∀ st, Blocks.runFrom env F (st + (c + 1 + 1)) ⟨vals, w⟩ s.nextBlk carries
              = runK env F st ⟨bindVals vals1 (parsOf np l.pTys) carries, w1⟩ RB := by
            intro st
            rw [show st + (c + 1 + 1) = ((st + 1) + c) + 1 by omega, enterC _ _ _ _ hlen,
              heq1', loopBrif_body hxv hcond hcs, enterB _ _ _ _ hlen]
          have hbody := ih k _ _ _ body Sb nb (loopB f s l pre) (Sem.bindAt Γ1 np carries) w1
            (bindVals vals1 (parsOf np l.pTys) carries) RB hfuel hok.hbody hBa hBi hPB hDG
            (fun ht => ⟨_, hPG ht⟩) hsb hrb
          have hBl' : (loopB f s l pre).labels = (s.nextBlk + 2, some s.nextBlk) :: s.labels := by
            show (loopBodyStart _ _ _ _).labels = _; rw [hBl, hDl, hCl]
          rw [hBl'] at hbody
          generalize hres : Sem.runCode k cfg (Sem.bindAt Γ1 np carries) w1 body = r at hbody
          cases r with
          | stuck _ => trivial
          | brk d Γb vs w2 =>
            obtain ⟨L, hL, hvl, vals2, c2, hrel, hf, heq⟩ := hbody
            cases d with
            | zero =>
              simp only [List.getElem?_cons_zero, Option.some.injEq] at hL
              subst hL
              refine hexit vals2 w2 Γb vs _ (c2 + (c + 1 + 1)) hrel
                (hfb.trans (hf.mono (by omega))) hvl (fun st => ?_)
              rw [← Nat.add_assoc, hpfx, heq]; rfl
            | succ d =>
              simp only [List.getElem?_cons_succ] at hL heq
              exact ⟨L, hL, hvl, vals2, c2 + (c + 1 + 1), hrel, hfb.trans (hf.mono (by omega)),
                fun st => by dsimp only; rw [← Nat.add_assoc, hpfx, heq]⟩
          | cont d vs w2 =>
            obtain ⟨L, hL, hvl, vals2, c2, hf, heq⟩ := hbody
            cases d with
            | zero =>
              simp only [List.getElem?_cons_zero, Option.some.injEq] at hL
              subst hL
              refine (ihk vals2 w2 vs (hfb.trans (hf.mono (by omega))) hvl).restart
                (c := c2 + (c + 1 + 1)) (fun st => ?_)
              rw [← Nat.add_assoc, hpfx, heq]; rfl
            | succ d =>
              simp only [List.getElem?_cons_succ] at hL heq
              exact ⟨L, hL, hvl, vals2, c2 + (c + 1 + 1), hfb.trans (hf.mono (by omega)),
                fun st => by dsimp only; rw [← Nat.add_assoc, hpfx, heq]⟩
          | ok Γ2 w2 =>
            obtain ⟨_, vals2, c2, hrel2, hf2, heq2⟩ := hbody
            have hterm : termsGo f body = false := by
              cases ht : termsGo f body
              · rfl
              · exact absurd hres (termsGo_run f body ht k cfg _ w1 Γ2 w2)
            obtain ⟨hcin, hclen⟩ : allIn Sb nb l.cont = true ∧ l.cont.length = l.pTys.length := by
              rcases hok.hcont with h' | h'
              · rw [hterm] at h'; cases h'
              · exact h'
            show SimResS _ _ _ _ _ _ _ _ _ _
              (match l.cont.mapM (Sem.get Γ2) with
                | none => .stuck "loop carry is not in scope"
                | some next => Sem.iter k cfg Γ w2 l pre body n nb next)
            rw [show l.cont.mapM (Sem.get Γ2) = l.cont.mapM (fun r => Γ2[r]?) from rfl]
            cases hm : l.cont.mapM (fun r => Γ2[r]?) with
            | none => trivial
            | some next =>
              have hGa' : Aligned (loopG f s l pre body) nb := hGa
              refine (ihk vals2 w2 next (hfb.trans (hf2.mono (by omega)))
                ((mapM_length hm).trans hclen)).restart (c := c2 + (c + 1 + 1))
                (fun st => ?_)
              rw [← Nat.add_assoc, hpfx, heq2 _ (hPG hterm),
                runK_jump ((args_read hGa' hrel2 hcin).trans hm)]

theorem sim_loop {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} (hF : Distinct F) (f : Nat)
    (hfuel : f ≤ HProg.fuel) (ih : SimP env cfg F f) {lb : List SLbl} {S : Scope} {n : Nat}
    {l : Loop} {pre body : List Piece} {Sp : Scope} {np : Nat} {Sb : Scope} {nb : Nat}
    (hok : LoopOk f lb S n l pre body Sp np Sb nb) (k : Nat) (s : CS) (ha : Aligned s n)
    (hi : Inv s) (Γ : Sem.Env) (w : World) (vals : Blocks.Vals) (R0 : List Inst)
    (hP : Placed F s R0) (hD : Done F (emitPiece f s (.loop l pre body))) (hs : Γ.size = n)
    (hr : Rel S vals Γ) :
    SimRes env F lb s.labels n ((S.add n (n + l.pTys.length)).add nb (nb + l.exitTys.length))
      (nb + l.exitTys.length) (emitPiece f s (.loop l pre body)) vals w R0
      (Sem.runPiece k cfg Γ w (.loop l pre body)) := by
  have htrip := loop_trip hF f hfuel ih hok s ha hi hD Γ hs vals hr
  obtain ⟨_, _, _, hCD, _, _, _, _, hBG, hGH, _, _⟩ := loop_stages f (emit_struct f) hok s ha hi
  obtain ⟨hCd, _, _, _, _, _, _⟩ := loopHead_facts s l
  obtain ⟨hBd, _, _, _, _, _, _⟩ := loopBodyStart_facts s.nextBlk l (loopD f s l pre) s.slots
  obtain ⟨hId, _, _, _, _, _, _⟩ := loopExit_facts s.nextBlk l s.labels (loopH f s l pre body)
  have hD' := hD
  rw [emitPiece_loop] at hD'
  have hDC : Done F (loopHead s l) := by
    refine hD'.mono ?_
    rw [hId]
    refine hCD.trans (List.IsPrefix.trans ?_ (hBG.trans hGH))
    show _ <+: (loopBodyStart _ _ _ _).done
    rw [hBd]; exact List.prefix_append _ _
  have hP0 : Placed F s [.jump ⟨s.nextBlk⟩ (l.init.map s.get)] :=
    ⟨{ ref := ⟨s.curRef⟩, params := s.curPars,
       insts := (Inst.jump ⟨s.nextBlk⟩ (l.init.map s.get) :: s.cur).reverse },
     hDC _ (by rw [hCd]; simp), rfl, rfl, by simp⟩
  have := hP.unique hF hP0; subst this
  have hsl1 := scGo_slots f _ _ _ pre Sp np hok.hpre
  have hsl2 := scGo_slots f _ _ _ body Sb nb hok.hbody
  have hnp : Sem.slotsOf (n + l.pTys.length) pre = np := (hsl1.2 HProg.fuel hfuel).1
  have hnb : Sem.slotsOf (np + l.pTys.length) body = nb := (hsl2.2 HProg.fuel hfuel).1
  cases k with
  | zero => trivial
  | succ k =>
    simp only [Sem.runPiece]
    rw [hs, hnp, hnb, show l.init.mapM (Sem.get Γ) = l.init.mapM (fun r => Γ[r]?) from rfl]
    cases hm : l.init.mapM (fun r => Γ[r]?) with
    | none => trivial
    | some inits =>
      exact (htrip k vals w inits (Frame.refl n vals) ((mapM_length hm).trans hok.hinitN)).restart
        (c := 0) (start := fun st => runK env F st ⟨vals, w⟩ [.jump ⟨s.nextBlk⟩ (l.init.map s.get)])
        (fun st => runK_jump ((args_read ha hr hok.hinit).trans hm))

def dloopD (f : Nat) (s : CS) (l : DLoop) (body : List Piece) : CS := emitCode f (dloopHead s l) body
def dloopE (f : Nat) (s : CS) (l : DLoop) (body : List Piece) : CS :=
  dloopBodyEnd f s.nextBlk l body (dloopD f s l body)

theorem emitPiece_dloop (f : Nat) (s : CS) (l : DLoop) (body : List Piece) :
    emitPiece f s (.dloop l body) = dloopExit s.nextBlk l s.labels (dloopE f s l body) := by
  rw [emitPiece, emitDLoop_eq]; rfl

theorem dloop_stages (f : Nat) (ih : StructP f) {lb : List SLbl} {S : Scope} {n : Nat}
    {l : DLoop} {body : List Piece} {Sb : Scope} {nb : Nat}
    (hok : DLoopOk f lb S n l body Sb nb) (s : CS) (ha : Aligned s n) (hi : Inv s) :
    Inv (dloopHead s l) ∧ Aligned (dloopHead s l) (n + l.pTys.length) ∧
    Aligned (dloopD f s l body) nb ∧ (dloopHead s l).done <+: (dloopD f s l body).done ∧
    (dloopD f s l body).labels = (dloopHead s l).labels ∧
    (dloopD f s l body).done <+: (dloopE f s l body).done ∧
    (dloopE f s l body).cur = [] ∧ Aligned (dloopE f s l body) nb := by
  simp only [dloopE, dloopD]
  obtain ⟨hCd, hCn, hCr, _, _, _, _⟩ := dloopHead_facts s l
  have hCa : Aligned (dloopHead s l) (n + l.pTys.length) :=
    aligned_open (aligned_congr (u := (({ s with nextBlk := s.nextBlk + 2 } : CS).close
      (dloopEntry s l))) ha rfl rfl rfl) s.nextBlk l.pTys _
  have hCids : ∀ x ∈ ids (dloopHead s l), x < s.nextBlk := by
    rw [ids_of_done hCd]; intro x hx
    simp only [List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | hx
    · exact hi.lt x hx
    · rw [hx]; exact hi.cur
  have hCo : Out s (dloopHead s l) false := by
    have h0 := ((Out.refl hi).bump 2).close (dloopEntry s l)
    refine h0.reopen ?_ ?_ ?_ ?_ ?_
    · rw [hCd]; rfl
    · rw [hCn]; rfl
    · rw [hCr]; intro hm
      rw [ids_close] at hm
      simp only [List.mem_append, List.mem_singleton] at hm
      rcases hm with hm | hm
      · have := hi.lt _ hm; omega
      · have := hi.cur; omega
    · rw [hCr]; show s.nextBlk < s.nextBlk + 2; omega
    · rw [hCr]; exact Nat.le_refl _
  obtain ⟨hDa, hDl, hDo, _, hDt⟩ := ih _ _ _ body Sb nb (dloopHead s l) hok.hbody hCa hCo.inv
  generalize hD : emitCode f (dloopHead s l) body = D at hDa hDl hDo hDt
  have hDav := out_avoid hDo (r := s.nextBlk + 1)
    (fun x hx => by have := hCids x hx; omega) (by rw [hCr]; omega) (by rw [hCn]; omega)
  have hDnb : s.nextBlk + 2 ≤ D.nextBlk := hCn ▸ hDo.grows.1
  have hEo : Out s (dloopBodyEnd f s.nextBlk l body D) true ∧
      D.done <+: (dloopBodyEnd f s.nextBlk l body D).done ∧
      (∀ x ∈ ids (dloopBodyEnd f s.nextBlk l body D), x ≠ s.nextBlk + 1) ∧
      (dloopBodyEnd f s.nextBlk l body D).nextBlk = D.nextBlk ∧
      (dloopBodyEnd f s.nextBlk l body D).slots = D.slots ∧
      (dloopBodyEnd f s.nextBlk l body D).nextVal = D.nextVal ∧
      (dloopBodyEnd f s.nextBlk l body D).env = D.env ∧
      (dloopBodyEnd f s.nextBlk l body D).cur = [] := by
    unfold dloopBodyEnd
    split
    · rename_i htb
      rw [htb] at hDo
      exact ⟨hCo.trans hDo, List.prefix_refl _, hDav.1, rfl, rfl, rfl, rfl, (hDt htb).2⟩
    · rename_i htb
      simp only [Bool.not_eq_true] at htb
      rw [htb] at hDo
      exact ⟨(hCo.trans hDo).close _, close_done_prefix D _, avoid_close hDav.1 hDav.2 _,
        rfl, rfl, rfl, rfl, rfl⟩
  obtain ⟨hEo, hDE, hEav, hEn, hEs, hEv, hEe, hEc⟩ := hEo
  generalize hE : dloopBodyEnd f s.nextBlk l body D = E at hEo hDE hEav hEn hEs hEv hEe hEc
  obtain ⟨hId, hIn, hIr, _, _, hIl, _⟩ := dloopExit_facts s.nextBlk l s.labels E
  have hIo : Out s (dloopExit s.nextBlk l s.labels E) false := by
    refine hEo.reopen hId hIn ?_ ?_ ?_
    · rw [hIr]; intro hm; exact hEav _ hm rfl
    · rw [hIr, hEn]; omega
    · rw [hIr]; omega
  subst hE hD
  exact ⟨hCo.inv, hCa, hDa, hDo.grows.2, hDl, hDE, hEc, aligned_congr hDa hEv hEs hEe⟩

theorem mapM_get {α β : Type} {g : α → Option β} : ∀ {xs : List α} {ys : List β},
    xs.mapM g = some ys → ∀ i (h : i < xs.length), g xs[i] = ys[i]? := by
  intro xs
  induction xs with
  | nil => intro ys _ i h; simp at h
  | cons x xs ih =>
      intro ys hm i h
      simp only [List.mapM_cons, Option.bind_eq_bind] at hm
      cases hx : g x with
      | none => simp [hx] at hm
      | some y =>
        cases hr : xs.mapM g with
        | none => simp [hx, hr] at hm
        | some zs =>
          simp [hx, hr] at hm; subst hm
          cases i with
          | zero => simpa using hx
          | succ i => simpa using ih hr i (by simp at h; omega)

/-- The exit values a bottom-tested loop passes are carries picked by position;
    the blocks pick the same ones out of the values they jump with. -/
theorem exit_read {vals : Blocks.Vals} {cvs : List Val} {next : List V}
    (hc : cvs.mapM (Blocks.getV vals) = some next) :
    ∀ (idx : List Nat) (outs : List V), (∀ i ∈ idx, i < cvs.length) →
      idx.mapM (fun i => next[i]?) = some outs →
      (idx.map (fun i => (cvs[i]?).getD (⟨1000000⟩ : Val))).mapM (Blocks.getV vals) = some outs := by
  intro idx
  induction idx with
  | nil => intro outs _ h; simpa using h
  | cons i is ih =>
      intro outs hidx h
      simp only [List.mapM_cons, Option.bind_eq_bind] at h
      cases hi : next[i]? with
      | none => simp [hi] at h
      | some o =>
        cases hr : is.mapM (fun i => next[i]?) with
        | none => simp [hi, hr] at h
        | some os =>
          simp [hi, hr] at h; subst h
          have hlt := hidx i (by simp)
          have e := mapM_get hc i hlt
          rw [hi] at e
          simp only [List.map_cons, List.mapM_cons, Option.bind_eq_bind]
          rw [show (cvs[i]?).getD (⟨1000000⟩ : Val) = cvs[i] by simp [hlt], e,
            ih os (fun j hj => hidx j (by simp [hj])) hr]
          rfl

theorem dloopBack_cont {env : FnEnv} {F : FuncData} {st : Nat} {vals : Blocks.Vals} {w : World}
    {h : Nat} {l : DLoop} {sB : CS} {t : ClifTy} {fv : UInt64} {next : List V}
    (hflag : Blocks.getV vals (sB.get l.flag) = some (.sc t fv))
    (hcond : ((fv != 0) == l.contOnTrue) = true)
    (hargs : (l.cont.map sB.get).mapM (Blocks.getV vals) = some next) :
    runK env F st ⟨vals, w⟩ [dloopBack h l sB] = Blocks.runFrom env F st ⟨vals, w⟩ h next := by
  unfold dloopBack
  refine runK_brif' (tgt := ⟨h⟩) (args := l.cont.map sB.get) hflag ?_ hargs
  cases hE : l.contOnTrue <;> cases hf : (fv != 0) <;> simp_all [Sem.isTrue]

theorem dloopBack_exit {env : FnEnv} {F : FuncData} {st : Nat} {vals : Blocks.Vals} {w : World}
    {h : Nat} {l : DLoop} {sB : CS} {t : ClifTy} {fv : UInt64} {outs : List V}
    (hflag : Blocks.getV vals (sB.get l.flag) = some (.sc t fv))
    (hcond : ((fv != 0) == l.contOnTrue) = false)
    (hargs : (l.exitIdx.map (fun i => ((l.cont.map sB.get)[i]?).getD (⟨1000000⟩ : Val))).mapM
      (Blocks.getV vals) = some outs) :
    runK env F st ⟨vals, w⟩ [dloopBack h l sB] = Blocks.runFrom env F st ⟨vals, w⟩ (h + 1) outs := by
  unfold dloopBack
  refine runK_brif' (tgt := ⟨h + 1⟩)
    (args := l.exitIdx.map (fun i => ((l.cont.map sB.get)[i]?).getD (⟨1000000⟩ : Val)))
    hflag ?_ hargs
  cases hE : l.contOnTrue <;> cases hf : (fv != 0) <;> simp_all [Sem.isTrue]

theorem dloopEntry_body {env : FnEnv} {F : FuncData} {st : Nat} {vals : Blocks.Vals} {w : World}
    {s : CS} {l : DLoop} {g : R} {t : ClifTy} {fv : UInt64} {cs : List V}
    (hg : l.guard = some g) (hflag : Blocks.getV vals (s.get g) = some (.sc t fv))
    (hcond : ((fv != 0) == l.contOnTrue) = true)
    (hargs : (l.init.map s.get).mapM (Blocks.getV vals) = some cs) :
    runK env F st ⟨vals, w⟩ [dloopEntry s l] = Blocks.runFrom env F st ⟨vals, w⟩ s.nextBlk cs := by
  unfold dloopEntry; rw [hg]
  refine runK_brif' (tgt := ⟨s.nextBlk⟩) (args := l.init.map s.get) hflag ?_ hargs
  cases hE : l.contOnTrue <;> cases hf : (fv != 0) <;> simp_all [Sem.isTrue]

theorem dloopEntry_exit {env : FnEnv} {F : FuncData} {st : Nat} {vals : Blocks.Vals} {w : World}
    {s : CS} {l : DLoop} {g : R} {t : ClifTy} {fv : UInt64} {outs : List V}
    (hg : l.guard = some g) (hflag : Blocks.getV vals (s.get g) = some (.sc t fv))
    (hcond : ((fv != 0) == l.contOnTrue) = false)
    (hargs : (l.exitIdx.map (fun i => ((l.init.map s.get)[i]?).getD (⟨1000000⟩ : Val))).mapM
      (Blocks.getV vals) = some outs) :
    runK env F st ⟨vals, w⟩ [dloopEntry s l]
      = Blocks.runFrom env F st ⟨vals, w⟩ (s.nextBlk + 1) outs := by
  unfold dloopEntry; rw [hg]
  refine runK_brif' (tgt := ⟨s.nextBlk + 1⟩)
    (args := l.exitIdx.map (fun i => ((l.init.map s.get)[i]?).getD (⟨1000000⟩ : Val)))
    hflag ?_ hargs
  cases hE : l.contOnTrue <;> cases hf : (fv != 0) <;> simp_all [Sem.isTrue]

theorem dloopEntry_none {env : FnEnv} {F : FuncData} {st : Nat} {vals : Blocks.Vals} {w : World}
    {s : CS} {l : DLoop} {cs : List V} (hg : l.guard = none)
    (hargs : (l.init.map s.get).mapM (Blocks.getV vals) = some cs) :
    runK env F st ⟨vals, w⟩ [dloopEntry s l] = Blocks.runFrom env F st ⟨vals, w⟩ s.nextBlk cs := by
  unfold dloopEntry; rw [hg]
  exact runK_jump hargs

/-- Leaving a bottom-tested loop by its exit block, from anywhere. -/
theorem dloop_exit {env : FnEnv} {F : FuncData} (hF : Distinct F) (f : Nat) {lb : List SLbl}
    {S : Scope} {n : Nat} {l : DLoop} {body : List Piece} {Sb : Scope} {nb : Nat}
    (hok : DLoopOk f lb S n l body Sb nb) (s : CS) (ha : Aligned s n) (hi : Inv s)
    (hSn : ∀ i, S.mem i = true → i < n) (vals0 : Blocks.Vals) (vals' : Blocks.Vals) (w' : World)
    (Γb : Sem.Env) (vs : List V) (start : Nat → Outcome World) (c : Nat)
    (hrel : Rel S vals' Γb) (hf : Frame n vals0 vals') (hvl : vs.length = l.exitTys.length)
    (hst : ∀ st, start (st + c) = Blocks.runFrom env F st ⟨vals', w'⟩ (s.nextBlk + 1) vs) :
    SimResS env F lb s.labels n (S.add nb (nb + l.exitTys.length)) (nb + l.exitTys.length)
      (emitPiece f s (.dloop l body)) vals0 start (.ok (Sem.bindAt Γb nb vs) w') := by
  obtain ⟨_, _, _, _, _, _, hEc, hEa⟩ := dloop_stages f (emit_struct f) hok s ha hi
  obtain ⟨_, _, hIr, hIc, hIp, _, _⟩ := dloopExit_facts s.nextBlk l s.labels (dloopE f s l body)
  have hnb := (scGo_slots f _ _ _ body Sb nb hok.hbody).1
  rw [emitPiece_dloop]
  have hrel' := bind_rel hrel (m := nb) (fun i hi => by have := hSn i hi; omega)
    (tys := l.exitTys) hvl.symm
  rw [hvl] at hrel'
  refine ⟨by rw [bindAt_size, hvl], _, c + 1, hrel',
    hf.trans ((bindVals_frame _ _ _ _).mono (by omega)), fun R1 hP1 st => ?_⟩
  rw [show st + (c + 1) = (st + 1) + c by omega, hst, ← hIr]
  exact enter hF hP1 (by rw [hIc, hEc]) (by rw [hIp, hEa.1]) hvl.symm st ⟨vals', w'⟩

theorem dloop_trip {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} (hF : Distinct F) (f : Nat)
    (hfuel : f ≤ HProg.fuel) (ih : SimP env cfg F f) {lb : List SLbl} {S : Scope} {n : Nat}
    {l : DLoop} {body : List Piece} {Sb : Scope} {nb : Nat}
    (hok : DLoopOk f lb S n l body Sb nb) (s : CS) (ha : Aligned s n) (hi : Inv s)
    (hD : Done F (emitPiece f s (.dloop l body))) (Γ : Sem.Env) (hs : Γ.size = n)
    (vals0 : Blocks.Vals) (hr0 : Rel S vals0 Γ) :
    ∀ (k : Nat) (vals : Blocks.Vals) (w : World) (carries : List V),
      Frame n vals0 vals → carries.length = l.pTys.length →
      SimResS env F lb s.labels n (S.add nb (nb + l.exitTys.length)) (nb + l.exitTys.length)
        (emitPiece f s (.dloop l body)) vals0
        (fun st => Blocks.runFrom env F st ⟨vals, w⟩ s.nextBlk carries)
        (Sem.dtrip k cfg Γ w l body n nb carries false) := by
  obtain ⟨hCi, hCa, hDa, _, hDl, hDE, _, _⟩ := dloop_stages f (emit_struct f) hok s ha hi
  obtain ⟨_, _, hCr, hCc, hCp, hCl, _⟩ := dloopHead_facts s l
  have hD' := hD
  rw [emitPiece_dloop] at hD'
  obtain ⟨hId, _, _, _, _, _, _⟩ := dloopExit_facts s.nextBlk l s.labels (dloopE f s l body)
  have hDE' : Done F (dloopE f s l body) := hD'.mono (by rw [hId]; exact List.prefix_refl _)
  have hDD : Done F (dloopD f s l body) := hDE'.mono hDE
  have hPD : termsGo f body = false →
      Placed F (dloopD f s l body) [dloopBack s.nextBlk l (dloopD f s l body)] := by
    intro ht
    have e : dloopE f s l body = (dloopD f s l body).close (dloopBack s.nextBlk l (dloopD f s l body)) := by
      simp only [dloopE, dloopBodyEnd, ht]; rfl
    rw [e] at hDE'; exact placed_close hDE'
  obtain ⟨_, _, _, hDx, hDt⟩ := emit_struct f _ _ _ body Sb nb (dloopHead s l) hok.hbody hCa hCi
  obtain ⟨RC, hPC⟩ : ∃ RC, Placed F (dloopHead s l) RC := by
    rcases hDx with ho | hc
    · cases ht : termsGo f body
      · exact ho.placed (hPD ht)
      · exact (hDt ht).1.placed hDD
    · exact hc.placed hDD
  have hSn : ∀ i, S.mem i = true → i < n := fun i hi => by have := hr0.lt hi; omega
  have enterC : ∀ st v w cs, cs.length = l.pTys.length →
      Blocks.runFrom env F (st + 1) ⟨v, w⟩ s.nextBlk cs
        = runK env F st ⟨bindVals v (parsOf n l.pTys) cs, w⟩ RC := by
    intro st v w cs hl
    rw [← hCr]
    exact enter hF hPC hCc (by rw [hCp, ha.1]) hl.symm st ⟨v, w⟩
  intro k
  induction k with
  | zero => intro vals w carries _ _; simp only [Sem.dtrip]; trivial
  | succ k ihk =>
    intro vals w carries hfr hlen
    have hrS : Rel S vals Γ := fun i hi => by
      obtain ⟨x, hx, hxv⟩ := hr0 i hi
      exact ⟨x, hx, hfr i (hSn i hi) x hxv⟩
    have hrh : Rel (S.add n (n + l.pTys.length)) (bindVals vals (parsOf n l.pTys) carries)
        (Sem.bindAt Γ n carries) := by
      have := bind_rel hrS hSn (tys := l.pTys) hlen.symm
      rwa [hlen] at this
    have hfh : Frame n vals0 (bindVals vals (parsOf n l.pTys) carries) :=
      hfr.trans (bindVals_frame _ _ _ _)
    have hbody := ih k _ _ _ body Sb nb (dloopHead s l) (Sem.bindAt Γ n carries) w
      (bindVals vals (parsOf n l.pTys) carries) RC hfuel hok.hbody hCa hCi hPC hDD
      (fun ht => ⟨_, hPD ht⟩) (by rw [bindAt_size, hlen]) hrh
    rw [hCl] at hbody
    simp only [Sem.dtrip, Bool.false_eq_true, if_false]
    generalize hres : Sem.runCode k cfg (Sem.bindAt Γ n carries) w body = r at hbody
    cases r with
    | stuck _ => trivial
    | brk d Γb vs w' =>
      obtain ⟨L, hL, hvl, vals', c, hrel, hf, heq⟩ := hbody
      cases d with
      | zero =>
        simp only [List.getElem?_cons_zero, Option.some.injEq] at hL
        subst hL
        refine dloop_exit hF f hok s ha hi hSn vals0 vals' w' Γb vs _ (c + 1) hrel
          (hfh.trans (hf.mono (by omega))) hvl (fun st => ?_)
        rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq]; rfl
      | succ d =>
        simp only [List.getElem?_cons_succ] at hL heq
        exact ⟨L, hL, hvl, vals', c + 1, hrel, hfh.trans (hf.mono (by omega)), fun st => by
          dsimp only
          rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq]⟩
    | cont d vs w' =>
      obtain ⟨L, hL, hvl, vals', c, hf, heq⟩ := hbody
      cases d with
      | zero =>
        simp only [List.getElem?_cons_zero, Option.some.injEq] at hL
        subst hL
        refine (ihk vals' w' vs (hfh.trans (hf.mono (by omega))) hvl).restart (c := c + 1)
          (fun st => ?_)
        rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq]; rfl
      | succ d =>
        simp only [List.getElem?_cons_succ] at hL heq
        exact ⟨L, hL, hvl, vals', c + 1, hfh.trans (hf.mono (by omega)), fun st => by
          dsimp only
          rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq]⟩
    | ok Γ2 w2 =>
      obtain ⟨_, vals2, c, hrel2, hf2, heq2⟩ := hbody
      have hterm : termsGo f body = false := by
        cases ht : termsGo f body
        · rfl
        · exact absurd hres (termsGo_run f body ht k cfg _ w Γ2 w2)
      obtain ⟨hfin, hcin, hclen, hsub⟩ : inS Sb nb l.flag = true ∧ allIn Sb nb l.cont = true ∧
          l.cont.length = l.pTys.length ∧ S.sub Sb = true := by
        rcases hok.hback with h' | h'
        · rw [hterm] at h'; cases h'
        · exact h'
      have hfb : Frame n vals0 vals2 := hfh.trans (hf2.mono (by omega))
      have heq2' := heq2 _ (hPD hterm)
      dsimp only
      rw [show l.cont.mapM (Sem.get Γ2) = l.cont.mapM (fun r => Γ2[r]?) from rfl]
      cases hm : l.cont.mapM (fun r => Γ2[r]?) with
      | none => trivial
      | some next =>
        have hargs := (args_read hDa hrel2 hcin).trans hm
        obtain ⟨hfl1, hfl2⟩ := (inS_iff Sb nb l.flag).mp hfin
        obtain ⟨x, hxΓ, hxv⟩ := hrel2 _ hfl2
        rw [← get_of_aligned _ nb hDa _ hfl1] at hxv
        simp only [Sem.get, hxΓ]
        cases x with
        | vec _ _ => trivial
        | sc t fv =>
          simp only
          split
          · rename_i hcond
            refine (ihk vals2 w2 next hfb ((mapM_length hm).trans hclen)).restart
              (c := c + 1) (fun st => ?_)
            rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq2',
              dloopBack_cont hxv hcond hargs]
          · rename_i hcond
            simp only [Bool.not_eq_true] at hcond
            cases hout : l.exitIdx.mapM (fun i => next[i]?) with
            | none => trivial
            | some outs =>
              refine dloop_exit hF f hok s ha hi hSn vals0 vals2 w2 Γ2 outs _ (c + 1)
                (hrel2.sub hsub) hfb ((mapM_length hout).trans hok.hexitN) (fun st => ?_)
              rw [show st + (c + 1) = (st + c) + 1 by omega, enterC _ _ _ _ hlen, heq2',
                dloopBack_exit hxv hcond (exit_read hargs l.exitIdx outs
                  (fun i hi => by simp; have := hok.hexitI i hi; omega) hout)]

theorem sim_dloop {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} (hF : Distinct F) (f : Nat)
    (hfuel : f ≤ HProg.fuel) (ih : SimP env cfg F f) {lb : List SLbl} {S : Scope} {n : Nat}
    {l : DLoop} {body : List Piece} {Sb : Scope} {nb : Nat}
    (hok : DLoopOk f lb S n l body Sb nb) (k : Nat) (s : CS) (ha : Aligned s n) (hi : Inv s)
    (Γ : Sem.Env) (w : World) (vals : Blocks.Vals) (R0 : List Inst) (hP : Placed F s R0)
    (hD : Done F (emitPiece f s (.dloop l body))) (hs : Γ.size = n) (hr : Rel S vals Γ) :
    SimRes env F lb s.labels n (S.add nb (nb + l.exitTys.length)) (nb + l.exitTys.length)
      (emitPiece f s (.dloop l body)) vals w R0 (Sem.runPiece k cfg Γ w (.dloop l body)) := by
  have htrip := dloop_trip hF f hfuel ih hok s ha hi hD Γ hs vals hr
  obtain ⟨_, _, _, hCD, _, hDE, _, _⟩ := dloop_stages f (emit_struct f) hok s ha hi
  obtain ⟨hCd, _, _, _, _, _, _⟩ := dloopHead_facts s l
  obtain ⟨hId, _, _, _, _, _, _⟩ := dloopExit_facts s.nextBlk l s.labels (dloopE f s l body)
  have hD' := hD
  rw [emitPiece_dloop] at hD'
  have hDC : Done F (dloopHead s l) :=
    hD'.mono (by rw [hId]; exact hCD.trans hDE)
  have hP0 : Placed F s [dloopEntry s l] :=
    ⟨{ ref := ⟨s.curRef⟩, params := s.curPars, insts := (dloopEntry s l :: s.cur).reverse },
     hDC _ (by rw [hCd]; simp), rfl, rfl, by simp⟩
  have := hP.unique hF hP0; subst this
  have hSn : ∀ i, S.mem i = true → i < n := fun i hi => by have := hr.lt hi; omega
  have hnb : Sem.slotsOf (n + l.pTys.length) body = nb :=
    ((scGo_slots f _ _ _ body Sb nb hok.hbody).2 HProg.fuel hfuel).1
  cases k with
  | zero => trivial
  | succ k =>
    simp only [Sem.runPiece]
    rw [hs, hnb, show l.init.mapM (Sem.get Γ) = l.init.mapM (fun r => Γ[r]?) from rfl]
    cases hm : l.init.mapM (fun r => Γ[r]?) with
    | none => trivial
    | some inits =>
      have hargs := (args_read ha hr hok.hinit).trans hm
      have hil : inits.length = l.pTys.length := (mapM_length hm).trans hok.hinitN
      cases hg : l.guard with
      | none =>
        simp only [Option.isSome_none]
        exact (htrip k vals w inits (Frame.refl n vals) hil).restart (c := 0)
          (start := fun st => runK env F st ⟨vals, w⟩ [dloopEntry s l])
          (fun st => dloopEntry_none hg hargs)
      | some g =>
        simp only [Option.isSome_some]
        cases k with
        | zero => simp only [Sem.dtrip]; trivial
        | succ k =>
          simp only [Sem.dtrip, if_true, hg]
          obtain ⟨hg1, hg2⟩ := (inS_iff S n g).mp (hok.hguard g hg)
          obtain ⟨x, hxΓ, hxv⟩ := hr _ hg2
          rw [← get_of_aligned s n ha _ hg1] at hxv
          simp only [Sem.get, hxΓ]
          cases x with
          | vec _ _ => trivial
          | sc t fv =>
            simp only
            split
            · rename_i hcond
              exact (htrip k vals w inits (Frame.refl n vals) hil).restart (c := 0)
                (start := fun st => runK env F st ⟨vals, w⟩ [dloopEntry s l])
                (fun st => dloopEntry_body hg hxv hcond hargs)
            · rename_i hcond
              simp only [Bool.not_eq_true] at hcond
              cases hout : l.exitIdx.mapM (fun i => inits[i]?) with
              | none => trivial
              | some outs =>
                exact dloop_exit hF f hok s ha hi hSn vals vals w Γ outs
                  (fun st => runK env F st ⟨vals, w⟩ [dloopEntry s l]) 0 hr (Frame.refl n vals)
                  ((mapM_length hout).trans hok.hexitN)
                  (fun st => dloopEntry_exit hg hxv hcond (exit_read hargs l.exitIdx outs
                    (fun i hi => by simp; have := hok.hexitI i hi; have := hok.hinitN; omega)
                    hout))

-- ---------------------------------------------------------------------------
-- Composition
-- ---------------------------------------------------------------------------

theorem iteArmEnd_prefix (f h : Nat) (arm : List Piece) (rs : List R) (sA : CS) :
    sA.done <+: (iteArmEnd f h arm rs sA).done := (iteArmEnd_facts f h arm rs sA).2.2.2.2.2

/-- **Emission only appends finished blocks**, whatever the region. -/
theorem emit_prefix : ∀ (f : Nat) (s : CS) (c : List Piece), s.done <+: (emitCode f s c).done := by
  intro f
  induction f with
  | zero => intro s c; rw [emitCode_zero]; exact List.prefix_refl _
  | succ f ih =>
    intro s c
    cases c with
    | nil => rw [emitCode_nil]; exact List.prefix_refl _
    | cons p ps =>
      rw [emitCode_cons]
      refine List.IsPrefix.trans ?_ (ih _ ps)
      cases p with
      | straight ss =>
          rw [show emitPiece f s (.straight ss) = emitStmts s ss by rw [emitPiece]]
          exact emitStmts_done_prefix ss s
      | loop l pre body =>
          rw [emitPiece_loop]
          obtain ⟨hId, _, _, _, _, _, _⟩ := loopExit_facts s.nextBlk l s.labels (loopH f s l pre body)
          obtain ⟨hCd, _, _, _, _, _, _⟩ := loopHead_facts s l
          obtain ⟨hBd, _, _, _, _, _, _⟩ :=
            loopBodyStart_facts s.nextBlk l (loopD f s l pre) s.slots
          rw [hId]
          have h1 : s.done <+: (loopHead s l).done := by rw [hCd]; exact List.prefix_append _ _
          have h2 : (loopD f s l pre).done <+: (loopB f s l pre).done := by
            show _ <+: (loopBodyStart _ _ _ _).done; rw [hBd]; exact List.prefix_append _ _
          have h3 : (loopG f s l pre body).done <+: (loopH f s l pre body).done := by
            simp only [loopH, loopBodyEnd]; split
            · exact List.prefix_refl _
            · exact close_done_prefix _ _
          exact h1.trans ((ih _ pre).trans (h2.trans ((ih _ body).trans h3)))
      | dloop l body =>
          rw [emitPiece_dloop]
          obtain ⟨hId, _, _, _, _, _, _⟩ := dloopExit_facts s.nextBlk l s.labels (dloopE f s l body)
          obtain ⟨hCd, _, _, _, _, _, _⟩ := dloopHead_facts s l
          rw [hId]
          have h1 : s.done <+: (dloopHead s l).done := by rw [hCd]; exact List.prefix_append _ _
          have h3 : (dloopD f s l body).done <+: (dloopE f s l body).done := by
            simp only [dloopE, dloopBodyEnd]; split
            · exact List.prefix_refl _
            · exact close_done_prefix _ _
          exact h1.trans ((ih _ body).trans h3)
      | ite m thn els thnR elsR =>
          rw [emitPiece_ite]
          obtain ⟨hCd, _, _, _, _, _, _⟩ := iteThen_facts s m
          obtain ⟨hFd, _, _, _, _, _, _⟩ := iteElse_facts f s.nextBlk thn thnR (iteD f s m thn)
          have hJ : (iteH f s m thn els thnR elsR).done <+:
              (iteJoin f s.nextBlk m thn els (iteH f s m thn els thnR elsR)).done := by
            unfold iteJoin; split
            · exact List.prefix_refl _
            · rw [(open'_fields _ _ _ _).1]; exact List.prefix_refl _
          have h1 : s.done <+: (iteThen s m).done := by rw [hCd]; exact List.prefix_append _ _
          have h2 : (iteD f s m thn).done <+: (iteF f s m thn thnR).done := by
            show _ <+: (iteElse _ _ _ _ _).done; rw [hFd]; exact iteArmEnd_prefix _ _ _ _ _
          exact h1.trans ((ih _ thn).trans (h2.trans ((ih _ els).trans
            ((iteArmEnd_prefix _ _ _ _ _).trans hJ))))
      | br d args =>
          rw [show emitPiece f s (.br d args) = s.close _ by rw [emitPiece]]
          exact close_done_prefix _ _
      | cont d args =>
          rw [show emitPiece f s (.cont d args) = s.close _ by rw [emitPiece]]
          exact close_done_prefix _ _

/-- A piece that leaves on every path never finishes normally. -/
theorem piece_ok_not_term {f k : Nat} {cfg : Sem.Cfg} {Γ : Sem.Env} {w : World} {p : Piece}
    {Γ1 : Sem.Env} {w1 : World} (ht : pieceTerm f p = true)
    (hok : Sem.runPiece k cfg Γ w p = .ok Γ1 w1) : False := by
  cases k with
  | zero => cases p <;> simp [Sem.runPiece] at hok
  | succ k =>
    apply termsGo_run (f + 1) [p] (by rw [termsGo_cons]; simpa using ht) (k + 2) cfg Γ w Γ1 w1
    rw [runCode_cons, hok]
    rfl

/-- A piece after which nothing runs. -/
theorem SimRes.last {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} {lb : List SLbl}
    {labels : List (Nat × Option Nat)} {n : Nat} {S1 S' : Scope} {n1 n' : Nat} {t1 t : CS}
    {vals : Blocks.Vals} {w : World} {R0 : List Inst} {r : Sem.CodeRes} {k : Nat}
    {ps : List Piece}
    (hp : SimRes env F lb labels n S1 n1 t1 vals w R0 r) (hnot : ∀ Γ1 w1, r ≠ .ok Γ1 w1) :
    SimRes env F lb labels n S' n' t vals w R0
      (andThen k cfg ps r) := by
  cases r with
  | ok Γ1 w1 => exact absurd rfl (hnot Γ1 w1)
  | brk d Γb vs w' => exact hp
  | cont d vs w' => exact hp
  | stuck _ => trivial

/-- A piece that may fall through, followed by the rest of its region. -/
theorem sim_compose {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} (f : Nat)
    (hfuel : f ≤ HProg.fuel) (ih : SimP env cfg F f) {lb : List SLbl} {n : Nat} {p : Piece}
    {ps : List Piece} {S' : Scope} {n' : Nat} {S1 : Scope} {n1 : Nat} (s : CS) (k : Nat)
    (Γ : Sem.Env) (w : World) (vals : Blocks.Vals) (R0 : List Inst)
    (hpt : pieceTerm f p = false) (hsc : scGo f lb S1 n1 ps = some (S', n'))
    (ha1 : Aligned (emitPiece f s p) n1) (hl1 : (emitPiece f s p).labels = s.labels)
    (ho1 : Out s (emitPiece f s p) false) (hle : n ≤ n1)
    (hD : Done F (emitCode f (emitPiece f s p) ps))
    (hreach : termsGo (f + 1) (p :: ps) = false →
      ∃ R1, Placed F (emitCode f (emitPiece f s p) ps) R1)
    (hp : Done F (emitPiece f s p) →
      SimRes env F lb s.labels n S1 n1 (emitPiece f s p) vals w R0 (Sem.runPiece k cfg Γ w p)) :
    SimRes env F lb s.labels n S' n' (emitCode f (emitPiece f s p) ps) vals w R0
      (Sem.runCode (k + 1) cfg Γ w (p :: ps)) := by
  rw [runCode_cons']
  have hD1 : Done F (emitPiece f s p) := hD.mono (emit_prefix f _ ps)
  obtain ⟨_, _, _, hx, ht⟩ := emit_struct f lb S1 n1 ps S' n' (emitPiece f s p) hsc ha1 ho1.inv
  have hreach' : termsGo f ps = false → ∃ R1, Placed F (emitCode f (emitPiece f s p) ps) R1 := by
    intro hf; apply hreach
    rw [termsGo_cons]; split
    · exact hpt
    · exact hf
  refine SimRes.seq (hp hD1) hle (fun Γ1 w1 _ => ?_) (fun Γ1 w1 vals1 Rm _ hPm hs1 hr1 => ?_)
  · rcases hx with ho | hc
    · cases htf : termsGo f ps
      · obtain ⟨R1, hP1⟩ := hreach' htf; exact ho.placed hP1
      · exact (ht htf).1.placed hD
    · exact hc.placed hD
  · have := ih k lb S1 n1 ps S' n' (emitPiece f s p) Γ1 w1 vals1 Rm hfuel hsc ha1 ho1.inv hPm hD
      hreach' hs1 hr1
    rwa [hl1] at this

/-- **Every region the check accepts simulates**, at every emitter fuel. -/
theorem sim_code {env : FnEnv} {cfg : Sem.Cfg} {F : FuncData} (hF : Distinct F) :
    ∀ f, SimP env cfg F f := by
  intro f
  induction f with
  | zero => intro k lb S n c S' n' s Γ w vals R0 _ h; simp [scGo] at h
  | succ f ih =>
    intro k lb S n c S' n' s Γ w vals R0 hfuel h ha hi hP hD hreach hs hr
    have hf : f ≤ HProg.fuel := by omega
    cases c with
    | nil =>
        simp only [scGo, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        rw [emitCode_nil]
        cases k with
        | zero => trivial
        | succ k =>
          rw [runCode_nil]
          exact ⟨hs, vals, 0, hr, Frame.refl _ _, fun R1 hP1 st => by rw [hP.unique hF hP1]; rfl⟩
    | cons p ps =>
      cases k with
      | zero => trivial
      | succ k =>
      rw [emitCode_cons] at hD hreach ⊢
      cases p with
      | straight ss =>
          obtain ⟨S1, n1, h1, h2⟩ := scGo_straight_inv h
          have hn1 := scStmts_count ss S n S1 n1 h1
          have hep : emitPiece f s (.straight ss) = emitStmts s ss := by rw [emitPiece]
          obtain ⟨hd, hnb, hcr⟩ := emitStmts_blk ss s
          exact sim_compose f hf ih s k Γ w vals R0 rfl h2
            (by rw [hep, hn1]; exact emitStmts_aligned ss s n ha)
            (by rw [hep]; exact emitStmts_labels ss s)
            (by rw [hep]; exact (Out.refl hi).same hd hnb hcr) (by omega) hD hreach
            (fun _ => sim_straight hF h1 f k s ha Γ w vals R0 hP hs hr)
      | loop l pre body =>
          obtain ⟨Sp, np, Sb, nb, hok, hk⟩ := scGo_loop_inv h
          obtain ⟨a1, l1, o1, _⟩ := struct_loop f (emit_struct f) hok s ha hi
          have h1 := (scGo_slots f _ _ _ pre Sp np hok.hpre).1
          have h2 := (scGo_slots f _ _ _ body Sb nb hok.hbody).1
          exact sim_compose f hf ih s k Γ w vals R0 rfl hk a1 l1 o1 (by omega) hD hreach
            (fun hD1 => sim_loop hF f hf ih hok k s ha hi Γ w vals R0 hP hD1 hs hr)
      | dloop l body =>
          obtain ⟨Sb, nb, hok, hk⟩ := scGo_dloop_inv h
          obtain ⟨a1, l1, o1, _⟩ := struct_dloop f (emit_struct f) hok s ha hi
          have h2 := (scGo_slots f _ _ _ body Sb nb hok.hbody).1
          exact sim_compose f hf ih s k Γ w vals R0 rfl hk a1 l1 o1 (by omega) hD hreach
            (fun hD1 => sim_dloop hF f hf ih hok k s ha hi Γ w vals R0 hP hD1 hs hr)
      | ite m thn els thnR elsR =>
          obtain ⟨St, nt, Se, ne, hok, hk⟩ := scGo_ite_inv h
          obtain ⟨a1, l1, o1, _, _⟩ := struct_ite f (emit_struct f) hok s ha hi
          rcases hk with ⟨ht1, ht2, hps, _, _⟩ | ⟨hnt, hk⟩
          · subst hps
            have hDp : Done F (emitPiece f s (.ite m thn els thnR elsR)) := by
              rwa [emitCode_nil] at hD
            rw [runCode_cons']
            exact (sim_ite hF f hf ih hok k s ha hi Γ w vals R0 hP hDp hs hr).last
              (fun Γ1 w1 hok1 => piece_ok_not_term (f := f) (by simp [pieceTerm, ht1, ht2]) hok1)
          · rw [hnt] at o1 a1
            simp only [Bool.false_eq_true, if_false] at a1
            have h1 := (scGo_slots f _ _ _ thn St nt hok.hthn).1
            have h2 := (scGo_slots f _ _ _ els Se ne hok.hels).1
            exact sim_compose f hf ih s k Γ w vals R0 (by simpa [pieceTerm] using hnt) hk a1 l1 o1
              (by omega) hD hreach
              (fun hD1 => sim_ite hF f hf ih hok k s ha hi Γ w vals R0 hP hD1 hs hr)
      | br d args =>
          obtain ⟨L, hL, hps, hin, hlen, hneed, _, _⟩ := scGo_br_inv h
          subst hps
          have hDp : Done F (emitPiece f s (.br d args)) := by rwa [emitCode_nil] at hD
          rw [runCode_cons']
          exact (sim_br hL hin hlen hneed f k s ha Γ w vals R0 hF hP hDp hr S n).last
            (fun Γ1 w1 hok1 => piece_ok_not_term (f := f) rfl hok1)
      | cont d args =>
          obtain ⟨L, hL, hps, hin, hlen, _, _⟩ := scGo_cont_inv h
          subst hps
          have hDp : Done F (emitPiece f s (.cont d args)) := by rwa [emitCode_nil] at hD
          rw [runCode_cons']
          exact (sim_cont hL hin hlen f k s ha Γ w vals R0 hF hP hDp hr S n).last
            (fun Γ1 w1 hok1 => piece_ok_not_term (f := f) rfl hok1)

-- ---------------------------------------------------------------------------
-- The theorem
-- ---------------------------------------------------------------------------

theorem entryCS_nextBlk (params : List ClifTy) : (entryCS params).nextBlk = 1 := by
  have := (open'_go_blk params
    { ({ nextVal := 0, nextBlk := 1, slots := 0, env := .nil, curRef := 0,
         curPars := [], cur := [], done := [] } : CS) with curRef := 0, curPars := [] } 0 0).2.1
  simpa [entryCS, CS.open'] using this

theorem entryCS_inv (params : List ClifTy) : Inv (entryCS params) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp [ids, entryCS_done, entryCS_curRef, entryCS_nextBlk]

theorem compileBody_eq (idx : Nat) (c : Code) (env : FnEnv) (params : List ClifTy)
    (status : Option R) :
    compileBody idx c env params status
      = { index := idx,
          blocks := ((emitCode HProg.fuel (entryCS params) c).close
              (.ret (status.map (emitCode HProg.fuel (entryCS params) c).get))).done.mergeSort
            (fun a b => a.ref.id ≤ b.ref.id) } := by
  simp only [compileBody, Id.run, entryCS]; rfl

theorem runK_ret (env : FnEnv) (F : FuncData) (steps : Nat) (st : Blocks.BSt) (v : Option Val)
    (rest : List Inst) : runK env F steps st (.ret v :: rest) = .ok st.world st.world := rfl

/-- **`compile_sound`.** A body that passes `scopeOk`, run by the term
    interpreter to a trace and a world, runs to the same trace and the same world
    as the function `compileBody` makes of it, under some block budget.

    Forward, on successful runs: a term that gets stuck has no behaviour to
    preserve. The block budget is existential because the two interpreters
    count different things --- pieces and trips on one side, block entries on
    the other --- and `run_mono` makes any larger budget give the same answer. -/
theorem compile_sound (idx : Nat) (env : FnEnv) (params : List ClifTy) (c : Code)
    (status : Option R) (cfg : Sem.Cfg) (args : List V) (w : World) (obs : List Sem.Obs)
    (w' : World) (hsc : scopeOk params c = true) (hlen : params.length = args.length)
    (hrun : Sem.run cfg args w c = .ok obs w') :
    ∃ steps, Blocks.run env (compileBody idx c env params status) args w steps = .ok obs w' := by
  -- The term's run.
  simp only [Sem.run] at hrun
  generalize hres : Sem.runCode cfg.steps cfg args.toArray w c = r at hrun
  cases r with
  | stuck _ => simp at hrun
  | brk _ _ _ _ => simp at hrun
  | cont _ _ _ => simp at hrun
  | ok Γf wf =>
  simp only [Sem.Outcome.ok.injEq] at hrun
  obtain ⟨rfl, rfl⟩ := hrun
  -- The check.
  simp only [scopeOk, Option.isSome_iff_exists] at hsc
  obtain ⟨⟨S', n'⟩, hsc⟩ := hsc
  have hterm : termsGo HProg.fuel c = false := by
    cases ht : termsGo HProg.fuel c
    · rfl
    · exact absurd hres (termsGo_run _ c ht _ cfg _ w Γf wf)
  obtain ⟨_, _, hout, hext, _⟩ := emit_struct HProg.fuel [] [(0, params.length)] params.length c
    S' n' (entryCS params) hsc (entryCS_aligned params) (entryCS_inv params)
  rw [hterm] at hout
  generalize ht : emitCode HProg.fuel (entryCS params) c = t at hout hext
  -- The function, and where everything is in it.
  generalize hF : compileBody idx c env params status = F
  have hblocks : F.blocks = (t.close (.ret (status.map t.get))).done.mergeSort
      (fun a b => a.ref.id ≤ b.ref.id) := by
    rw [← hF, compileBody_eq, ht]
  have hFd : Distinct F := by
    unfold Distinct; rw [hblocks]
    apply sorted_nodup
    have hi := hout.inv
    show (ids (t.close _)).Nodup
    rw [ids_close, List.nodup_append]
    refine ⟨hi.nodup, by simp, fun y hy z hz => ?_⟩
    simp only [List.mem_singleton] at hz
    subst hz; exact fun heq => hi.open_ (heq ▸ hy)
  have hDone : Done F (t.close (.ret (status.map t.get))) := by
    intro b hb; rw [hblocks]; exact sorted_mem _ _ b hb
  have hDt : Done F t := hDone.mono (close_done_prefix t _)
  have hPt : Placed F t [.ret (status.map t.get)] := placed_close hDone
  obtain ⟨R0, hP0⟩ : ∃ R0, Placed F (entryCS params) R0 := by
    rcases hext with ho | hc
    · exact ho.placed hPt
    · exact hc.placed hDt
  -- The entry block's parameters are the arguments.
  have hrel0 : Rel [(0, params.length)] (bindVals #[] (parsOf 0 params) args) args.toArray := by
    have h := bind_rel (A := []) (vals := #[]) (Γ := #[]) (fun i hi => by simp [Scope.mem] at hi)
      (m := 0) (fun i hi => by simp [Scope.mem] at hi) (tys := params) hlen
    have e : Sem.bindAt #[] 0 args = args.toArray := by simp [Sem.bindAt]
    rw [e, ← hlen] at h
    simpa [Scope.add] using h
  have hsim := sim_code (env := env) (cfg := cfg) hFd HProg.fuel cfg.steps [] [(0, params.length)]
    params.length c S' n' (entryCS params) args.toArray w (bindVals #[] (parsOf 0 params) args) R0
    (Nat.le_refl _) hsc (entryCS_aligned params) (entryCS_inv params) hP0 (ht ▸ hDt)
    (fun _ => ⟨_, ht ▸ hPt⟩) (by simp [hlen]) hrel0
  rw [hres, ht] at hsim
  obtain ⟨_, vals', cost, _, _, heq⟩ := hsim
  refine ⟨cost + 1, ?_⟩
  have hentry := enter (env := env) hFd hP0 (entryCS_cur params) (entryCS_curPars params) hlen cost
    ⟨#[], w⟩
  rw [entryCS_curRef] at hentry
  simp only [Blocks.run]
  rw [hentry]
  have := heq _ hPt 0
  simp only [Nat.zero_add] at this
  rw [this, runK_ret]

/-- `compile_sound` in `CompileSoundE`'s form: for a body that passes the scope
    check, every run the term completes is the compiled function's run. -/
theorem compileSoundE_of_ok (idx : Nat) (env : FnEnv) (params : List ClifTy) (c : Code)
    (args : List V) (w : World) (fuel : Nat) (obs : List Sem.Obs) (w' : World)
    (hsc : scopeOk params c = true) (hlen : params.length = args.length)
    (hrun : Sem.run { env, steps := fuel } args w c = .ok obs w') :
    CompileSoundE idx env params c args w fuel := by
  obtain ⟨steps, h⟩ := compile_sound idx env params c none _ args w obs w' hsc hlen hrun
  exact ⟨steps, by rw [hrun, h]⟩

end AlgorithmLib.HProg
