module
public import AlgorithmLib.Host.Sound
meta import AlgorithmLib.Host.Sound
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Program` — a program's own calls

`compile_sound` is per function: a call to another of the program's functions
means whatever `cfg.locals` says, on both sides. This ties that knot. A
program is a table of term functions; the term side gives a local call its
meaning by running the callee's term, the machine side by running the
callee's compiled blocks, each to a call depth. **`program_sound`**: at every
depth, a call the terms answer is answered the same way by the compiled
functions.

The machine side picks, for each call, the answer of a block budget that
suffices. Budgets are upward-closed (`runFrom_mono`), so every budget that
suffices gives the same answer and the choice is not a guess; it is
noncomputable, and this is a statement, not an interpreter.
-/

namespace AlgorithmLib.HProg

open AlgorithmLib.IR
open AlgorithmLib.HProg.Sem

/-- One of a program's own functions, as a term: its parameters, its body, and
    the slot it answers with, if it answers. -/
structure Fn where
  params : List ClifTy
  code : Code
  status : Option R
  /-- The signatures of the functions it calls, as it was compiled with them. -/
  env : FnEnv := []

/-- Calling a term function: its body run on the arguments, answering the
    returned slot's value; a body that misuses a call or faults ends the call
    so, and any other end is stuck. Stuck unless the arguments match its
    parameters. -/
def Sem.callFn (cfg : Sem.Cfg) (f : Fn) (args : List V) (w : World) : Except Sem.Fail (Option V × World) :=
  if f.params.length ≠ args.length then .error (.stuck "a local call's arguments do not match its parameters")
  else match Sem.runCode cfg.steps cfg args.toArray w f.code with
    | .ok Γ w' => .ok (f.status.bind (Γ[·]?), w')
    | .misuse m => .error (.misuse m)
    | .fault m => .error (.fault m)
    | .stuck m => .error (.stuck m)
    | _ => .error (.stuck "a local call's body left a loop it is not in")

/-- The term program's local calls, to call depth `k`: each runs the callee's
    body, whose own local calls go one level shallower. -/
def termLocals (fns : Nat → Option Fn) (env : FnEnv) (steps : Nat) : Nat → Sem.Locals
  | 0 => Sem.noLocals
  | k + 1 => fun i args w =>
      match fns i with
      | none => .error (.stuck s!"calls u0:{i}, which the program does not have")
      | some f => Sem.callFn { env, steps, locals := termLocals fns env steps k } f args w

/-- What `compileBody` makes of each function of the table. -/
def compiledFns (fns : Nat → Option Fn) (i : Nat) : Option FuncData :=
  (fns i).map fun f => compileBody i f.code f.env f.params f.status

/-- The compiled program's local calls, to call depth `k`: each runs the
    callee's blocks under a budget that suffices, if one does. -/
noncomputable def blockLocals (fns : Nat → Option FuncData) : Nat → Sem.Locals
  | 0 => Sem.noLocals
  | k + 1 => fun i args w =>
      match fns i with
      | none => .error (.stuck s!"calls u0:{i}, which the program does not have")
      | some F =>
          open Classical in
          if h : ∃ p : Nat × (Option V × World), Blocks.callFn (blockLocals fns k) F p.1 args w = some p.2
          then .ok (Classical.choose h).2 else .error (.stuck "no budget runs the callee to its end")

-- ---------------------------------------------------------------------------
-- A machine run is monotone in what its local calls mean
-- ---------------------------------------------------------------------------

/-- `L₂` answers every call `L₁` answers, the same way. -/
def Sem.Locals.le (L₁ L₂ : Sem.Locals) : Prop :=
  ∀ i args w r, L₁ i args w = .ok r → L₂ i args w = .ok r

theorem spawnWorker_mono {L₁ L₂ : Sem.Locals} (h : L₁.le L₂) (vs : List V) (w : World)
    (r : Option V × World) (hc : Sem.spawnWorker L₁ vs w = some r) :
    Sem.spawnWorker L₂ vs w = some r := by
  unfold Sem.spawnWorker at hc ⊢
  split at hc
  · next _ ctx _ fn _ arg =>
    by_cases h0 : (ctx == 0) = true
    · rw [if_pos h0] at hc ⊢; exact hc
    · rw [if_neg h0] at hc ⊢
      by_cases h1 : (!(ctx == threadCtx && w.thread.live) || w.thread.outstanding.isSome) = true
      · rw [if_pos h1] at hc; cases hc
      · rw [if_neg h1] at hc ⊢
        cases hl : L₁ fn.toNat [.sc .i64 arg] w with
        | error _ => rw [hl] at hc; cases hc
        | ok x => rw [hl] at hc; rw [h _ _ _ _ hl]; exact hc
  · cases hc

theorem callOf_mono {L₁ L₂ : Sem.Locals} (h : L₁.le L₂) (c : Callee) (vs : List V) (w : World)
    (r : Option V × World) (hc : Sem.callOf L₁ c vs w = some r) : Sem.callOf L₂ c vs w = some r := by
  cases c with
  | ffi f =>
      simp only [Sem.callOf] at hc ⊢
      split at hc
      · cases hc
      · rename_i hfz
        rw [if_neg hfz]
        split at hc
        · rename_i hsp
          rw [if_pos hsp]; exact spawnWorker_mono h vs w r hc
        · rename_i hsp
          rw [if_neg hsp]; exact hc
  | «local» i =>
      simp only [Sem.callOf] at hc ⊢
      cases hl : L₁ i vs w with
      | error _ => rw [hl] at hc; simp [Except.toOption] at hc
      | ok x =>
          rw [hl] at hc; simp only [Except.toOption, Option.some.injEq] at hc; subst hc
          rw [h _ _ _ _ hl]; rfl
  | native => cases hc
  | atomic a => exact hc
  | ext e => exact hc

theorem runInsts_locals_mono {L₁ L₂ : Sem.Locals} (h : L₁.le L₂) :
    ∀ (is : List Inst) (s : Blocks.BSt) (x : Blocks.BSt × Blocks.Next) (w : World),
      Blocks.runInsts L₁ s is = .ok x w → Blocks.runInsts L₂ s is = .ok x w
  | [], s, x, w, hr => by simp [Blocks.runInsts] at hr
  | i :: rest, s, x, w, hr => by
    have ih := runInsts_locals_mono h rest
    cases i with
    | call d c args =>
        rw [runInsts_call] at hr ⊢
        split at hr
        · cases hr
        · rename_i vs _
          cases hc : Sem.callOf L₁ c vs (Sem.obsCall s.world c vs) with
          | none => rw [hc] at hr; exact absurd hr Sem.failOf_ne_ok
          | some p =>
              rw [hc] at hr
              rw [callOf_mono h c vs _ p hc]
              obtain ⟨res, w'⟩ := p
              dsimp only at hr ⊢
              split at hr
              · exact ih _ _ _ hr
              · exact ih _ _ _ hr
              · cases hr
    | store v a =>
        simp only [Blocks.runInsts] at hr ⊢
        split at hr
        · cases hr
        · first | exact ih _ _ _ hr | (rename_i heq; rw [heq]; exact ih _ _ _ hr)
    | storeTyped t v a =>
        simp only [Blocks.runInsts] at hr ⊢
        split at hr
        · cases hr
        · first | exact ih _ _ _ hr | (rename_i heq; rw [heq]; exact ih _ _ _ hr)
    | istore8 v a =>
        simp only [Blocks.runInsts] at hr ⊢
        split at hr
        · split at hr
          · first | exact ih _ _ _ hr | (rename_i heq; rw [heq]; exact ih _ _ _ hr)
          · cases hr
        · cases hr
    | ret v => cases v <;> simp only [Blocks.runInsts] at hr ⊢ <;> exact hr
    | jump t args => simp only [Blocks.runInsts] at hr ⊢; exact hr
    | brif c tb ta eb ea => simp only [Blocks.runInsts] at hr ⊢; exact hr
    | _ =>
        simp only [Blocks.runInsts] at hr ⊢
        split at hr
        · cases hr
        · first | exact ih _ _ _ hr | (rename_i heq; rw [heq]; exact ih _ _ _ hr)

theorem runFrom_locals_mono {L₁ L₂ : Sem.Locals} (h : L₁.le L₂) (F : FuncData) :
    ∀ (steps : Nat) (s : Blocks.BSt) (blk : Nat) (args : List V) (r : Option V) (w : World),
      Blocks.runFrom L₁ F steps s blk args = .ok r w → Blocks.runFrom L₂ F steps s blk args = .ok r w := by
  intro steps
  induction steps with
  | zero => intro s blk args r w hr; simp [Blocks.runFrom] at hr
  | succ steps ih =>
    intro s blk args r w hr
    rw [Blocks.runFrom] at hr ⊢
    cases hfind : F.blocks.find? (·.ref.id == blk) with
    | none => simp [hfind] at hr
    | some b =>
      simp only [hfind] at hr ⊢
      split at hr
      · cases hr
      · rename_i hlen
        simp only [hlen, if_false, Bool.false_eq_true]
        cases hins : Blocks.runInsts L₁
            { s with vals := (b.params.zip args).foldl (fun vs x => Blocks.setV vs x.1.1 x.2) s.vals }
            b.insts with
        | stuck m => simp [hins] at hr
        | misuse m => simp [hins] at hr
        | fault m => simp [hins] at hr
        | ok p w' =>
          rw [runInsts_locals_mono h _ _ _ _ hins]
          simp only [hins] at hr
          obtain ⟨s', next⟩ := p
          cases next with
          | done r' => exact hr
          | goto t vs => exact ih _ t vs r w hr

theorem callFn_locals_mono {L₁ L₂ : Sem.Locals} (h : L₁.le L₂) (F : FuncData) (steps : Nat)
    (args : List V) (w : World) (r : Option V × World)
    (hc : Blocks.callFn L₁ F steps args w = some r) : Blocks.callFn L₂ F steps args w = some r := by
  unfold Blocks.callFn at hc ⊢
  cases hr : Blocks.runFrom L₁ F steps ⟨#[], w⟩ 0 args with
  | stuck m => rw [hr] at hc; cases hc
  | misuse m => rw [hr] at hc; cases hc
  | fault m => rw [hr] at hc; cases hc
  | ok a w' => rw [hr] at hc; rw [runFrom_locals_mono h F steps _ 0 args a w' hr]; exact hc

/-- Two budgets that both suffice give the same answer. -/
theorem callFn_det (lc : Sem.Locals) (F : FuncData) (s₁ s₂ : Nat) (args : List V) (w : World)
    (r₁ r₂ : Option V × World) (h₁ : Blocks.callFn lc F s₁ args w = some r₁)
    (h₂ : Blocks.callFn lc F s₂ args w = some r₂) : r₁ = r₂ := by
  unfold Blocks.callFn at h₁ h₂
  cases e₁ : Blocks.runFrom lc F s₁ ⟨#[], w⟩ 0 args with
  | stuck m => rw [e₁] at h₁; cases h₁
  | misuse m => rw [e₁] at h₁; cases h₁
  | fault m => rw [e₁] at h₁; cases h₁
  | ok a₁ w₁ =>
  cases e₂ : Blocks.runFrom lc F s₂ ⟨#[], w⟩ 0 args with
  | stuck m => rw [e₂] at h₂; cases h₂
  | misuse m => rw [e₂] at h₂; cases h₂
  | fault m => rw [e₂] at h₂; cases h₂
  | ok a₂ w₂ =>
  rw [e₁] at h₁; rw [e₂] at h₂
  have m₁ := runFrom_mono lc F s₁ ⟨#[], w⟩ 0 args a₁ w₁ e₁ (max s₁ s₂) (Nat.le_max_left _ _)
  have m₂ := runFrom_mono lc F s₂ ⟨#[], w⟩ 0 args a₂ w₂ e₂ (max s₁ s₂) (Nat.le_max_right _ _)
  rw [m₁] at m₂
  simp only [Sem.Outcome.ok.injEq] at m₂
  obtain ⟨rfl, rfl⟩ := m₂
  simp only [Option.some.injEq] at h₁ h₂
  rw [← h₁, ← h₂]

/-- **The compiled program answers its own calls as the term program does.**

    For a table of functions each of which passes `retOk`, at every call depth:
    whenever running a callee's term answers a call, running its compiled
    blocks answers it the same way --- the same value and the same world, so the
    same trace. By induction on the depth, with `compile_core` for the callee's
    body and `runFrom_locals_mono` to replace the term meaning of its own calls
    by the compiled one. -/
theorem program_sound (fns : Nat → Option Fn) (env : FnEnv) (steps : Nat)
    (hok : ∀ i f, fns i = some f → retOk f.params f.code f.status = true) :
    ∀ k, (termLocals fns env steps k).le (blockLocals (compiledFns fns) k) := by
  intro k
  induction k with
  | zero => intro i args w r h; simp [termLocals, Sem.noLocals] at h
  | succ k ih =>
    intro i args w r h
    simp only [termLocals] at h
    cases hf : fns i with
    | none => rw [hf] at h; cases h
    | some f =>
      rw [hf] at h
      dsimp only at h
      unfold Sem.callFn at h
      dsimp only at h
      split at h
      · cases h
      · rename_i hlen
        have hlen' : f.params.length = args.length := Classical.not_not.mp hlen
        cases hres : Sem.runCode steps { env, steps, locals := termLocals fns env steps k }
            args.toArray w f.code with
        | ok Γ w' =>
          rw [hres] at h
          simp only [Except.ok.injEq] at h
          subst h
          obtain ⟨s, hs⟩ := compile_core i f.env f.params f.code f.status
            { env, steps, locals := termLocals fns env steps k } args w Γ w' (hok i f hf) hlen' hres
          have hs2 := runFrom_locals_mono ih _ _ _ _ _ _ _ hs
          have hcall : Blocks.callFn (blockLocals (compiledFns fns) k)
              (compileBody i f.code f.env f.params f.status) s args w
              = some (f.status.bind (Γ[·]?), w') := by
            simp only [Blocks.callFn, hs2]
          have hex : ∃ p : Nat × (Option V × World), Blocks.callFn (blockLocals (compiledFns fns) k)
              (compileBody i f.code f.env f.params f.status) p.1 args w = some p.2 := ⟨(s, _), hcall⟩
          simp only [blockLocals, compiledFns, hf, Option.map_some, dif_pos hex]
          exact congrArg Except.ok (callFn_det _ _ _ _ _ _ _ _ (Classical.choose_spec hex) hcall)
        | stuck _ => rw [hres] at h; cases h
        | misuse _ => rw [hres] at h; cases h
        | fault _ => rw [hres] at h; cases h
        | brk _ _ _ _ => rw [hres] at h; cases h
        | cont _ _ _ => rw [hres] at h; cases h

/-- **A whole program, compiled, does what its terms do.** The entry function's
    term, with the program's local calls meaning their terms to depth `k`, and
    its compiled blocks, with them meaning their compiled blocks to the same
    depth, make the same trace and leave the same world. -/
theorem program_run_sound (fns : Nat → Option Fn) (env : FnEnv) (steps k : Nat)
    (hok : ∀ i f, fns i = some f → retOk f.params f.code f.status = true)
    (idx : Nat) (main : Fn) (hmain : retOk main.params main.code main.status = true)
    (args : List V) (w : World) (obs : List Sem.Obs) (w' : World)
    (hlen : main.params.length = args.length)
    (hrun : Sem.run { env, steps, locals := termLocals fns env steps k } args w main.code = .ok obs w') :
    ∃ s, Blocks.run (blockLocals (compiledFns fns) k)
      (compileBody idx main.code main.env main.params main.status) args w s = .ok obs w' := by
  obtain ⟨s, hs⟩ := compile_sound idx main.env main.params main.code main.status _ args w obs w'
    hmain hlen hrun
  refine ⟨s, ?_⟩
  unfold Blocks.run at hs ⊢
  cases hr : Blocks.runFrom (termLocals fns env steps k)
      (compileBody idx main.code main.env main.params main.status) s ⟨#[], w⟩ 0 args with
  | stuck m => simp only [hr] at hs; cases hs
  | misuse m => simp only [hr] at hs; cases hs
  | fault m => simp only [hr] at hs; cases hs
  | ok a wf =>
    rw [runFrom_locals_mono (program_sound fns env steps hok k) _ _ _ _ _ _ _ hr]
    simp only [hr] at hs
    exact hs

end AlgorithmLib.HProg
