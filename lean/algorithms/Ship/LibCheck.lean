import AlgorithmLib.Proof.LibSpec
import AlgorithmLib.Check.Quiet

/-!
# A library's functions, as the checker reads their calls

What `Check.Lib` asks of a library function, from what its statements say:
a call whose statement is a `CallsD` or a `CallsA`, in a program all of whose
functions are quiet, is a `LibCall` whose world the statement's describes
(`libCall_of_callsD`, `libCall_of_callsA`), and what such a call keeps outside
the heap is a `LibStep` (`libStep_of`; `libStep_refl` where it changed
nothing). A call passes its arguments at their types' widths (`Sem.normV`),
which leaves a 64-bit value, and a 32-bit one below `2³²`, as it is.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace LibCheck
open AlgorithmLib.LibSpec AlgorithmLib.Check

@[simp] theorem normV_i64 (x : UInt64) : normV (.sc .i64 x) = .sc .i64 x := by
  simp only [normV, norm, widthMask]; congr 1; exact UInt64.and_neg_one

theorem normV_i32 {x : UInt64} (h : x.toNat < 2 ^ 32) : normV (.sc .i32 x) = .sc .i32 x := by
  simp only [normV, norm, widthMask, V.sc.injEq, true_and]
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and]
  exact Nat.and_two_pow_sub_one_of_lt_two_pow (n := 32) h

/-- A call passes what it is handed at its types' widths, so two lists that
    narrow alike are the same call. -/
theorem libCall_norm {L : Locals} {f : LibFn} {vs vs' : List V} {w : World} {Q : World → Prop}
    (h : vs.map normV = vs'.map normV) : LibCall L f vs w Q ↔ LibCall L f vs' w Q := by
  unfold LibCall; rw [h]

theorem out_cell {c : UInt64} (hc : cellOk c = true) : Out c := by
  obtain ⟨r, off, hd, hp, -⟩ := cellOk_spec hc
  intro r' off' hd'
  rw [hd] at hd'; cases hd'; exact hp

/-- **What a call leaves of the rest of the world**, from what it keeps
    outside the heap and its being quiet. -/
theorem libStep_of {w w' : World} {P : UInt64 → Nat → Prop} {lo hi : Nat} (ho : Outside w w' P)
    (hq : Quiet w w') (hc : Calm w)
    (hP : ∀ c, cellOk c = true → hi ≤ c.toNat ∨ c.toNat + 8 ≤ lo → P c 8) : LibStep w w' lo hi := by
  have hc' := ho.calm hc
  refine ⟨fun r hr => ho.size r hr, hq.lent, by rw [show w'.mem.busy = [] from hc']; rfl, fun c v hcell hm hl => ?_,
    hq.calls⟩
  exact ho.keeps.load c 8 v (by decide) ⟨load_inW hl, out_cell hcell, hP c hcell hm⟩ hl

/-- A call that changed nothing left the rest of the world. -/
theorem libStep_refl {w : World} (hc : Calm w) (lo hi : Nat) : LibStep w w lo hi :=
  ⟨fun _ _ => rfl, rfl, by rw [show w.mem.busy = [] from hc]; rfl, fun _ _ _ _ h => h, ⟨[], rfl⟩⟩

theorem calm_of {w : World} (hi : w.mem.busy.isEmpty = true) : Calm w :=
  List.isEmpty_iff.mp hi

section linked
variable {fns : Nat → Option Fn} {env : FnEnv} {steps : Nat}

/-- A call whose statement is a `CallsD`, as the checker reads it. -/
theorem libCall_of_callsD {f : LibFn} {vs : List V} {w : World} {Q Q' : World → Prop} {k : Nat}
    (hq : ∀ i f, fns i = some f → quietCode f.code = true) (hn : vs.map normV = vs)
    (h : CallsD (termLocals fns env steps k) f.libSlot vs w Q) (hQ : ∀ w', Q w' → Quiet w w' → Q' w') :
    LibCall (termLocals fns env steps k) f vs w Q' := by
  unfold CallsD at h
  unfold LibCall
  rw [hn]
  cases hl : termLocals fns env steps k f.libSlot vs w with
  | error e => rw [hl] at h; exact h
  | ok p =>
      obtain ⟨r, w'⟩ := p
      rw [hl] at h
      exact hQ w' h (termLocals_quiet hq k _ _ _ _ _ hl)

/-- A call whose statement is a `CallsA`, as the checker reads it. -/
theorem libCall_of_callsA {f : LibFn} {vs : List V} {w : World} {Q : UInt64 → World → Prop} {Q' : World → Prop}
    {k : Nat} (hq : ∀ i f, fns i = some f → quietCode f.code = true)
    (hn : vs.map normV = vs)
    (h : CallsA (termLocals fns env steps k) f.libSlot vs w Q) (hQ : ∀ x w', Q x w' → Quiet w w' → Q' w') :
    LibCall (termLocals fns env steps k) f vs w Q' := by
  unfold CallsA at h
  unfold LibCall
  rw [hn]
  cases hl : termLocals fns env steps k f.libSlot vs w with
  | error e => rw [hl] at h; exact h
  | ok p =>
      obtain ⟨r, w'⟩ := p
      rw [hl] at h
      obtain ⟨v, -, hv⟩ := h
      exact hQ v w' hv (termLocals_quiet hq k _ _ _ _ _ hl)

end linked

end LibCheck
