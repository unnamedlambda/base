module
public import AlgorithmLib.Host.Summary
public import AlgorithmLib.Proof.Typestate
@[expose] public section

/-!
# `Check.Seq` — the program's own functions called one after another

A body that calls some of the program's own functions in turn, each on the
arguments it was itself called with, neither misuses a call nor faults where
each callee's `Summary` leads from the typestate the call before it leaves to
one the next starts in. Proven once, from the call rule, for any list of
callees.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.Prog

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (TState)

/-- Summaries leading from `S` through each callee in turn, every call on `args`. -/
def SumChain (L : Sem.Locals) (args : List V) : TState → List Nat → Prop
  | _, [] => True
  | S, c :: cs => ∃ S' : TState, Summary L c args S.holds S'.holds ∧ SumChain L args S' cs

/-- The calls an entry point that calls `cs` in turn on its own arguments makes,
    after taking its memory's address. -/
def seqCalls (cs : List Nat) : List Stmt := cs.map fun c => .callVoid (.local c) [0, 1, 2, 3]

/-- What an entry point's environment holds once it has taken its memory's
    address: its arguments, then the address. -/
def entryEnv (dl ol : UInt64) : Env := (runArgs dl ol).toArray.push (.sc .i64 (regionBase .arena))

theorem runArgs_mapM (dl ol : UInt64) :
    [0, 1, 2, 3].mapM (fun r => (entryEnv dl ol)[r]?) = some (runArgs dl ol) := rfl

/-- A straight run's statements from the one state `At` names. -/
theorem stmts_conseq_at {cfg : Cfg} {P Q : Env → World → Prop} {ss : List Stmt} {fo : Bool}
    {Γ0 : Env} {w0 : World} (h : Stmts cfg P ss Q fo) (w : World) (hw : P Γ0 w0) :
    Stmts cfg (At Γ0 w0) ss Q fo := fun Γ w' ⟨e1, e2⟩ => by subst e1; subst e2; exact h _ _ hw

theorem seqCalls_sound {cfg : Cfg} {dl ol : UInt64} :
    ∀ {cs : List Nat} {S : TState}, SumChain cfg.locals (runArgs dl ol) S cs →
      Stmts cfg (fun Γ w => Γ = entryEnv dl ol ∧ S.holds w) (seqCalls cs) (fun _ _ => True) false
  | [], _, _ => fun _ _ _ => trivial
  | c :: cs, S, ⟨S', hs, hrest⟩ => by
      refine stmts_cons (callVoidL_rule fun Γ w vs ⟨e, hw⟩ hvs => ?_) (seqCalls_sound hrest)
      subst e
      rw [runArgs_mapM] at hvs
      cases hvs
      have hw1 := AlgorithmLib.Prog.TState.holds_obsCall (c := .local c) (vs := runArgs dl ol) hw
      have := hs _ hw1
      simp only [callOf, failOf]
      revert this
      cases cfg.locals c (runArgs dl ol) (obsCall w (.local c) (runArgs dl ol)) with
      | error f =>
          intro hf
          exact ⟨fun _ => hf, fun _ _ h => (by cases h)⟩
      | ok p =>
          intro hp
          refine ⟨fun h => (by cases h), fun r w' h => ?_⟩
          cases h
          exact ⟨trivial, hp⟩

/-- One callee called `n` times over, its summary leading back to where it starts. -/
theorem sumChain_repeat {L : Sem.Locals} {args : List V} {S : TState} {c : Nat}
    (h : Summary L c args S.holds S.holds) {rest : List Nat} (hr : SumChain L args S rest) :
    ∀ n, SumChain L args S (List.replicate n c ++ rest)
  | 0 => hr
  | n + 1 => ⟨S, h, sumChain_repeat h hr n⟩

/-- A summary from a smaller typestate is one from a larger. -/
theorem summary_weaken {L : Sem.Locals} {i : Nat} {args : List V} {S T : TState} {W' : World → Prop}
    (h : Summary L i args S.holds W') (hs : ∀ f ∈ S, f ∈ T) : Summary L i args T.holds W' :=
  fun w hw => h w fun f hf => hw f (hs f hf)

/-- **An entry point calling the program's own functions in turn**, on its own
    arguments, neither misuses a call nor faults where their summaries chain. -/
theorem seqCalls_run {cfg : Cfg} {dl ol : UInt64} {cs : List Nat} {S : TState}
    (hc : SumChain cfg.locals (runArgs dl ol) S cs) {w : World} (hw : S.holds w) (m : String) :
    run cfg (runArgs dl ol) w [.straight (.op .memory :: seqCalls cs)] ≠ .misuse m ∧
      run cfg (runArgs dl ol) w [.straight (.op .memory :: seqCalls cs)] ≠ .fault m := by
  have hm : Stmt1 cfg (fun Γ w => Γ = (runArgs dl ol).toArray ∧ S.holds w) (.op .memory)
      (fun Γ w => Γ = entryEnv dl ol ∧ S.holds w) false :=
    op_rule (fun Γ w v ⟨e, hw⟩ hv => by
        simp only [evalOp, Option.some.injEq] at hv
        subst e hv; exact ⟨rfl, hw⟩)
      (fun _ _ _ hv => by simp [evalOp] at hv)
  have ht : Triple cfg (At (runArgs dl ol).toArray w) [.straight (.op .memory :: seqCalls cs)]
      { ok := fun _ _ => True, faultOk := false } :=
    cons_rule (R := fun _ _ => True)
      (straight_rule (stmts_conseq_at (stmts_cons hm (seqCalls_sound hc)) w ⟨rfl, hw⟩))
      (nil_rule fun _ _ _ => trivial)
  exact run_sound ht rfl m

end AlgorithmLib.Check
