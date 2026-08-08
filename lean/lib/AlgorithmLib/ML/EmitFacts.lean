import AlgorithmLib.ML.PtxPrint

/-!
# Emitter facts that hold for every kernel

A guard that has to be re-checked per net is a guard that will one day be
forgotten.  These are the properties of the emitter itself: true for any
statement, proven by induction rather than decided at a concrete tape, so a
model instantiates them and states nothing.
-/

namespace AlgorithmLib.ML


/-- An index expression emits arithmetic and loads — never a structured loop,
    never an extern. -/
theorem emitIdx_printable (lr : Nat) : ∀ (e : IdxE) (n : Nat),
    ∀ x ∈ (emitIdx lr n e).2.1, SI.printableB x = true := by
  intro e
  induction e with
  | add a b iha ihb =>
      intro n x hx
      simp only [emitIdx, List.mem_append, List.mem_cons,
                 List.not_mem_nil, or_false] at hx
      rcases hx with (h | h) | h
      · exact iha n x h
      · exact ihb _ x h
      · subst h; rfl
  | mul a b iha ihb =>
      intro n x hx
      simp only [emitIdx, List.mem_append, List.mem_cons,
                 List.not_mem_nil, or_false] at hx
      rcases hx with (h | h) | h
      · exact iha n x h
      · exact ihb _ x h
      · subst h; rfl
  | ldIdx b off ih =>
      intro n x hx
      simp only [emitIdx, List.mem_append, List.mem_cons,
                 List.not_mem_nil, or_false] at hx
      rcases hx with h | h
      · exact ih n x h
      · subst h; rfl
  | _ =>
      intro n x hx
      simp only [emitIdx, List.mem_cons, List.not_mem_nil, or_false] at hx
      all_goals (subst hx; rfl)

/-- **Every flattened statement is printable, whatever it is.**

    `flatEW` compiles both loop forms into branches itself, so the fall-through
    to `emitEW` only ever sees a leaf, and no leaf emits a structured loop.
    `EWStmt` has no extern constructor at all.  Nothing about a particular
    kernel enters this. -/
theorem flatEW_printable : ∀ (s : EWStmt) (lr n base : Nat),
    FlatPrintableB (flatEW lr n base s) = true := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro lr n base
      simp only [flatEW, FlatPrintableB, List.all_append, Bool.and_eq_true]
      exact ⟨iha lr n base, ihb lr n (base + flenEW lr n a)⟩
  | forN cnt body ih =>
      intro lr n base
      simp only [flatEW, FlatPrintableB, List.all_cons, List.all_append,
                 Bool.and_eq_true]
      exact ⟨rfl, rfl, rfl, ih n (n + 1) (base + 3),
             by simp [FI.printableB, SI.printableB]⟩
  | forM bu ad body ih =>
      intro lr n base
      simp only [flatEW, FlatPrintableB, List.all_cons, List.all_append,
                 Bool.and_eq_true]
      exact ⟨rfl, rfl, rfl, rfl, rfl, ih n (n + 3) (base + 5),
             by simp [FI.printableB, SI.printableB]⟩
  | _ =>
      intro lr n base
      simp only [flatEW, FlatPrintableB, emitEW, List.all_map, List.all_eq_true,
                 Function.comp_apply]
      intro x hx
      simp only [List.mem_append, List.mem_cons,
                 List.not_mem_nil, or_false] at hx
      repeat' (rcases hx with hx | hx)
      all_goals first
        | (subst hx; rfl)
        | exact emitIdx_printable _ _ _ _ hx
        | (simp only [List.mem_map] at hx
           obtain ⟨p, _, rfl⟩ := hx
           rfl)
        | rfl

/-- **Every emitted kernel is printable.**

    The prologue is a fixed list and the body is `flatEW`, so this holds for any
    statement at all — no kernel, and no net, states it again. -/
theorem flatKernel_printable (s : EWStmt) :
    FlatPrintableB (flatKernel s) = true := by
  simp only [flatKernel, FlatPrintableB, List.all_append, Bool.and_eq_true]
  exact ⟨by decide, flatEW_printable s 2 3 3⟩

theorem targetOkB_mono {h1 h2 : Nat} (hle : h1 ≤ h2) :
    ∀ i : FI, FI.targetOkB h1 i = true → FI.targetOkB h2 i = true := by
  intro i h
  cases i with
  | si _ => rfl
  | jmp t => simp only [FI.targetOkB, decide_eq_true_eq] at h ⊢; omega
  | jmpIf _ t => simp only [FI.targetOkB, decide_eq_true_eq] at h ⊢; omega

/-- **Every branch a flattened statement emits lands inside it.**

    Targets are absolute instruction indices, so this is the fact that keeps
    `flatKernel`'s branches meaningful — and it is a property of the flattener,
    not of any kernel. -/
theorem flatEW_targets : ∀ (s : EWStmt) (lr n base : Nat),
    (flatEW lr n base s).all (FI.targetOkB (base + flenEW lr n s)) = true := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro lr n base
      simp only [flatEW, flenEW, List.all_append, Bool.and_eq_true]
      constructor
      · exact List.all_eq_true.mpr (fun i hi =>
          targetOkB_mono (by omega) i (List.all_eq_true.mp (iha lr n base) i hi))
      · exact List.all_eq_true.mpr (fun i hi =>
          targetOkB_mono (by omega) i
            (List.all_eq_true.mp (ihb lr n (base + flenEW lr n a)) i hi))
  | forN cnt body ih =>
      intro lr n base
      simp only [flatEW, flenEW, List.all_cons, List.all_append, Bool.and_eq_true]
      refine ⟨rfl, rfl, by simp only [FI.targetOkB, decide_eq_true_eq]; omega, ?_, ?_, ?_⟩
      · exact List.all_eq_true.mpr (fun i hi =>
          targetOkB_mono (by omega) i
            (List.all_eq_true.mp (ih n (n + 1) (base + 3)) i hi))
      · rfl
      · simp only [List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true,
                   FI.targetOkB, decide_eq_true_eq, and_true, true_and]
        omega
  | forM bu ad body ih =>
      intro lr n base
      simp only [flatEW, flenEW, List.all_cons, List.all_append, Bool.and_eq_true]
      refine ⟨rfl, rfl, rfl, rfl,
              by simp only [FI.targetOkB, decide_eq_true_eq]; omega, ?_, ?_, ?_⟩
      · exact List.all_eq_true.mpr (fun i hi =>
          targetOkB_mono (by omega) i
            (List.all_eq_true.mp (ih n (n + 3) (base + 5)) i hi))
      · rfl
      · simp only [List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true,
                   FI.targetOkB, decide_eq_true_eq, and_true, true_and]
        omega
  | _ =>
      intro lr n base
      simp only [flatEW, List.all_map, List.all_eq_true, Function.comp_apply]
      intro i _
      rfl

/-- **Every branch in an emitted kernel resolves.**

    The prologue branches nowhere and the body's targets are inside it, so this
    holds for any statement — the guard no net needs to restate. -/
theorem flatKernel_targets (s : EWStmt) :
    FlatTargetsOkB (flatKernel s) = true := by
  have hl : (flatKernel s).length = 3 + flenEW 2 3 s := flatKernel_length s
  simp only [FlatTargetsOkB, flatKernel, List.all_append, Bool.and_eq_true] at hl ⊢
  constructor
  · exact List.all_eq_true.mpr (fun i hi => by
      obtain ⟨j, _, rfl⟩ := List.mem_map.mp hi; rfl)
  · exact List.all_eq_true.mpr (fun i hi =>
      targetOkB_mono (by omega : 3 + flenEW 2 3 s ≤ _) i
        (List.all_eq_true.mp (flatEW_targets s 2 3 3) i hi))

end AlgorithmLib.ML
