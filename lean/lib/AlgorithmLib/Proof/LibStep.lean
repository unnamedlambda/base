module
public import AlgorithmLib.Proof.LibSpecFloat
meta import AlgorithmLib.Proof.LibSpecFloat
public import Lean
meta import Lean
@[expose] public section

/-!
# `Proof.LibStep` — the steps of a library function's proof, taken for it

Tactics over `Proof.LibSpec`'s and `Proof.LibSpecFloat`'s rules, for a
function's weakest precondition walked one instruction at a time.

`step` takes one instruction by the rule for it: the program at the head of
the goal is reduced to its first instruction, and that instruction's rules are
tried, each needing the slots it reads (`look`); a construct the rules do not
name goes to `step_ext`, which a proof extends with the specifications of the
sub-programs it has proven. `look` finds a slot's value: pushed since, kept by
a sub-program (an `Extends` hypothesis, `ext_get_step`), or as a branch found it;
`extc` shows an environment keeps an earlier one. `arm` ends an arm that joins
one value, `joined` and `pass` continue after a join (`LibSpec.Joins`), and
`keep_paths` clears all but a branch's path conditions `hc…`.

A proof using these makes `wp` irreducible in its file, so that a rule that
does not match fails at once instead of unfolding the semantics.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace AlgorithmLib.LibSpec

open Lean Elab Tactic Meta in
/-- Apply the lemma `lem h` for the first hypothesis `h : Extends _ _` it fits. -/
meta def viaExt (lem : Name) : TacticM Unit := withReducible do
  let g ← getMainGoal
  g.withContext do
  for d in (← getLCtx) do
    if d.isImplementationDetail then continue
    let ty ← instantiateMVars d.type
    if ty.isAppOfArity ``Sem.Extends 2 then
      let s ← saveState
      try
        let gs ← g.apply (← mkAppM lem #[d.toExpr])
        replaceMainGoal gs
        return
      catch _ => s.restore
  throwError "no Extends hypothesis fits"

elab "ext_get_step" : tactic => viaExt ``Extends.get
elab "ext_after_step" : tactic => viaExt ``ext_after

open Lean Elab Tactic Meta in
/-- A loop's carry: where the slot looked up is the first, second or third of
    those a `bindAt` binds, by the slot's own form, the value bound there. -/
elab "carry_look" : tactic => do
  let g ← getMainGoal
  let t ← instantiateMVars (← g.getType)
  let some (_, lhs, _) := t.eq? | throwError "carry_look: not an equation"
  let args := lhs.getAppArgs
  unless lhs.isAppOf ``GetElem?.getElem? && args.size ≥ 2 do throwError "carry_look: not a lookup"
  let env := args[args.size - 2]!
  let idx := args[args.size - 1]!
  unless env.isAppOfArity ``bindAt 3 do throwError "carry_look: not a loop's environment"
  let n := env.getArg! 1
  let k : Option Nat :=
    if idx == n then some 0
    else if idx.isAppOfArity ``HAdd.hAdd 6 && idx.getArg! 4 == n then
      match (idx.getArg! 5).nat? with
      | some j => some j
      | none => none
    else none
  match k with
  | some 0 => evalTactic (← `(tactic| exact bindAt_head _ _ _ _))
  | some 1 => evalTactic (← `(tactic| exact bindAt_second _ _ _ _ _))
  | some 2 => evalTactic (← `(tactic| exact bindAt_third _ _ _ _ _ _))
  | _ => throwError "carry_look: not a carry"

/-- That an environment keeps an earlier one: pushes, and sub-programs' own
    such facts. -/
syntax "extc" : tactic
macro_rules
  | `(tactic| extc) => `(tactic| with_reducible first
      | exact Extends.refl _
      | (apply Extends.push_of; extc)
      | assumption
      | (ext_after_step; extc)
      | (refine Extends.bindAt_of ?_ (by assumption) _; extc))

open Lean Elab Tactic Meta in
/-- A lookup `Γ[i]? = rhs`, as its environment, slot and right side. -/
meta def lookParts (t : Expr) : Option (Expr × Expr × Expr) := do
  let some (_, lhs, rhs) := t.eq? | none
  let args := lhs.getAppArgs
  unless lhs.isAppOf ``GetElem?.getElem? && args.size ≥ 2 do none
  some (args[args.size - 2]!, args[args.size - 1]!, rhs)

open Lean Elab Tactic Meta in
/-- A slot as written, with a projection out of a pair or a carry list taken
    and arithmetic on numerals done: `(k, r).snd` is `r`, `0 + 1` is `1`. -/
meta partial def slotNorm (e : Expr) : MetaM Expr := do
  let e ← whnfCore e
  if let some n ← evalNat e then return mkNatLit n
  match e.getAppFn.constName? with
  | some c =>
    if (← getEnv).isProjectionFn c then
      let e' ← withReducible (whnf e)
      if e' == e then return e else slotNorm e'
    else return e
  | none => return e

open Lean Elab Tactic Meta in
/-- The lookup `t` with its environment replaced by `G`. -/
meta def lookIn (t G : Expr) : MetaM Expr := do
  let some (_, lhs, rhs) := t.eq? | throwError "look: not a lookup"
  let args := lhs.getAppArgs
  mkEq (mkAppN lhs.getAppFn (args.set! (args.size - 2) G)) rhs

open Lean Elab Tactic Meta in
/-- **A proof of a lookup**, found by the form of the environment and the
    slot: a hypothesis about that slot of that environment; a push, whose own
    slot is the value pushed and whose others are the environment's below; a
    loop's binding, whose first three slots are its carries and whose others
    are the environment's it binds after; or an environment an `Extends`
    hypothesis says keeps an earlier one. Values are unified only where the
    slot is found. -/
meta partial def lookProof (t : Expr) (fuel : Nat) : MetaM Expr := do
  if fuel = 0 then throwError "look: not found"
  let t ← instantiateMVars t
  let some (env, idx, rhs) := lookParts t | throwError "look: not a lookup"
  let idx ← slotNorm idx
  -- a hypothesis about this slot of this environment
  for d in (← getLCtx) do
    if d.isImplementationDetail then continue
    let ty ← instantiateMVars d.type
    if let some (e', i', _) := lookParts ty then
      if e' == env && (← slotNorm i') == idx then
        if ← withReducible (isDefEq t ty) then return d.toExpr
  if env.isAppOfArity ``Array.push 3 then
    let G := env.getArg! 1
    let v := env.getArg! 2
    if idx.isAppOfArity ``Array.size 2 && idx.getArg! 1 == G then
      let pf ← mkAppOptM ``get_push_size #[G, v]
      if ← withReducible (isDefEq t (← inferType pf)) then return pf
      throwError m!"look: the slot's value is not the one asked for: {t} vs {← inferType pf}"
    let x ← (if rhs.isApp then pure rhs.appArg! else throwError "look: right side not `some`")
    let h ← lookProof (← lookIn t G) (fuel - 1)
    return ← mkAppOptM ``get_push #[G, idx, v, x, h]
  if env.isAppOfArity ``bindAt 3 then
    let G := env.getArg! 0
    let n := env.getArg! 1
    let vs := env.getArg! 2
    let k : Option Nat :=
      if idx == n then some 0
      else if idx.isAppOfArity ``HAdd.hAdd 6 && idx.getArg! 4 == n then (idx.getArg! 5).nat?
      else none
    if let some j := k then
      let lem := match j with
        | 0 => ``bindAt_head | 1 => ``bindAt_second | _ => ``bindAt_third
      let vals ← (do
        let mut out := #[]
        let mut l := vs
        for _ in [0:j+1] do
          unless l.isAppOfArity ``List.cons 3 do throwError "look: carries not a list"
          out := out.push (l.getArg! 1)
          l := l.getArg! 2
        return out.push l)
      let pf ← mkAppM lem (#[G, n] ++ vals)
      if ← withReducible (isDefEq t (← inferType pf)) then return pf
      throwError "look: the carry's value is not the one asked for"
    let x ← (if rhs.isApp then pure rhs.appArg! else throwError "look: right side not `some`")
    -- the binding's slots below `n`: the environment's, where it is no longer
    let hn? ← (do
      if n.isAppOfArity ``Array.size 2 && n.getArg! 1 == G then return some (← mkAppM ``Nat.le_refl #[n])
      else if G.isAppOfArity ``Array.push 3 && n.isAppOfArity ``HAdd.hAdd 6 &&
          (n.getArg! 4).isAppOfArity ``Array.size 2 && (n.getArg! 4).getArg! 1 == G.getArg! 1 then
        return some (← mkAppM ``Nat.le_of_eq #[← mkAppOptM ``Array.size_push #[none, G.getArg! 1, G.getArg! 2]])
      else
        let want ← mkAppM ``LE.le #[← mkAppM ``Array.size #[G], n]
        for d in (← getLCtx) do
          if d.isImplementationDetail then continue
          if ← withReducible (isDefEq (← instantiateMVars d.type) want) then return some d.toExpr
        return none)
    if let some hn := hn? then
      let h ← lookProof (← lookIn t G) (fuel - 1)
      return ← mkAppOptM ``get_bindAt #[G, n, idx, vs, x, hn, h]
  -- an environment an `Extends` hypothesis says keeps an earlier one
  for d in (← getLCtx) do
    if d.isImplementationDetail then continue
    let ty ← instantiateMVars d.type
    if ty.isAppOfArity ``Sem.Extends 2 && ty.getArg! 1 == env then
      let A := ty.getArg! 0
      try
        let h ← lookProof (← lookIn t A) (fuel - 1)
        return ← mkAppM ``Extends.get #[d.toExpr, h]
      catch _ => pure ()
  throwError "look: not found"

open Lean Elab Tactic Meta in
/-- **Every slot lookup in the goal, rewritten to its value**: each `Γ[i]?`
    whose value `look` finds becomes `some v`. -/
elab "look_rw" : tactic => do
  for _ in [0:64] do
    let g ← getMainGoal
    let done ← g.withContext do
      let t ← instantiateMVars (← g.getType)
      -- the first lookup of an environment, outermost first
      let found := t.find? fun e =>
        e.isAppOf ``GetElem?.getElem? && e.getAppNumArgs ≥ 2 && (e.getAppArgs[e.getAppNumArgs - 2]!).isApp
      match found with
      | none => return true
      | some lk =>
        let ty ← inferType lk
        let x ← mkFreshExprMVar ((← whnf ty).appArg!)
        let eq ← mkEq lk (← mkAppOptM ``Option.some #[none, x])
        let pf ← lookProof eq 400
        let r ← g.rewrite t pf
        let g' ← g.replaceTargetEq r.eNew r.eqProof
        replaceMainGoal (g' :: r.mvarIds)
        return false
    if done then return

open Lean Elab Tactic Meta in
/-- A slot's value: pushed since, a loop's carry, kept by a sub-program, or as
    a branch found it (`lookProof`). -/
elab "look" : tactic => do
  let g ← getMainGoal
  g.withContext do
    let pf ← lookProof (← g.getType) 400
    g.assign pf
    replaceMainGoal []

/-- A sub-program, by its specification; extended as each is proven. -/
syntax "step_ext" : tactic

open Lean Elab Tactic Meta in
/-- One instruction, by the rule for it: the program at the head of the goal
    is reduced to its first instruction, and that instruction's rules tried. -/
elab "step" : tactic => do
  let g ← getMainGoal
  let t := (← instantiateMVars (← g.getType)).cleanupAnnotations.headBeta
  unless t.isAppOf ``AlgorithmLib.Prog.wp do throwError m!"step: not a wp goal: {t.getAppFn}"
  let args := t.getAppArgs
  let p ← whnfCore args[4]!
  let t := mkAppN t.getAppFn (args.set! 4 p)
  let g ← g.replaceTargetDefEq t
  replaceMainGoal [g]
  let run (stx : TacticM (TSyntax `tactic)) : TacticM Unit := do evalTactic (← stx)
  match p.getAppFn.constName? with
  | some ``Bind.bind => run `(tactic| rw [wp_bind])
  | some ``Pure.pure => run `(tactic| rw [wp_pure])
  | some ``AlgorithmLib.Prog.iconst32 => run `(tactic| refine wp_iconst32u (by decide) ?_)
  | some ``AlgorithmLib.Prog.iconst64 => run `(tactic| refine wp_iconst64 ?_)
  | some ``AlgorithmLib.Prog.iconst => run `(tactic| first | refine wp_iconst8f ?_ | refine wp_iconst8t ?_)
  | some ``AlgorithmLib.Prog.fconst => run `(tactic| refine wp_fconst64 ?_)
  | some ``AlgorithmLib.Prog.bitcast => run `(tactic| with_reducible first
      | refine wp_bitcastI32 (by look) ?_ | refine wp_bitcastF32 (by look) ?_
      | refine wp_bitcastI64 (by look) ?_ | refine wp_bitcastF64 (by look) ?_)
  | some ``AlgorithmLib.Prog.uextend64 => run `(tactic| with_reducible first
      | refine wp_uext32 (by look) ?_ | refine wp_uextend8 (by look) ?_)
  | some ``AlgorithmLib.Prog.fpromote => run `(tactic| with_reducible refine wp_fpromote (by look) ?_)
  | some ``AlgorithmLib.Prog.fdemote => run `(tactic| with_reducible refine wp_fdemote (by look) ?_)
  | some ``AlgorithmLib.Prog.fcvtFromSint => run `(tactic| with_reducible refine wp_fcvtFromSint (by look) ?_)
  | some ``AlgorithmLib.Prog.fcvtToSint => run `(tactic| with_reducible refine wp_fcvtToSint (by look) ?_)
  | some ``AlgorithmLib.Prog.fadd => run `(tactic| with_reducible refine wp_fadd64 (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.fsub => run `(tactic| with_reducible refine wp_fsub64 (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.fmul => run `(tactic| with_reducible refine wp_fmul64 (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.fdiv => run `(tactic| with_reducible refine wp_fdiv64 (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.fneg => run `(tactic| with_reducible refine wp_fneg64 (by look) ?_)
  | some ``AlgorithmLib.Prog.floor => run `(tactic| with_reducible refine wp_floor64 (by look) ?_)
  | some ``AlgorithmLib.Prog.fcmp => run `(tactic| with_reducible refine wp_fcmp64 (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.select => run `(tactic| with_reducible first
      | refine wp_selectF (by look) (by look) (by look) ?_
      | refine wp_select32 (by look) (by look) (by look) ?_
      | refine wp_select (by look) (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.band => run `(tactic| with_reducible first
      | refine wp_band32 (by look) (by look) ?_ | refine wp_band (by look) (by look) ?_
      | refine wp_band8 (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.bor => run `(tactic| with_reducible first
      | refine wp_bor32 (by look) (by look) ?_ | refine wp_bor (by look) (by look) ?_
      | refine wp_bor8 (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.bxor => run `(tactic| with_reducible first
      | refine wp_bxor (by look) (by look) ?_ | refine wp_bxor8 (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.icmp => run `(tactic| with_reducible first
      | refine wp_icmp32 (by look) (by look) ?_ | refine wp_icmp (by look) (by look) ?_
      | refine wp_icmp8 (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.ushrImm => run `(tactic| with_reducible first
      | refine wp_ushrImm32 (by decide) (by look) ?_ | refine wp_ushrImm (by look) ?_)
  | some ``AlgorithmLib.Prog.ishlImm => run `(tactic| with_reducible refine wp_ishlImm (by look) ?_)
  | some ``AlgorithmLib.Prog.iaddImm => run `(tactic| with_reducible refine wp_iaddImm (by look) ?_)
  | some ``AlgorithmLib.Prog.umulhi => run `(tactic| with_reducible refine wp_umulhi (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.umax => run `(tactic| with_reducible first
      | refine wp_umax32 (by look) (by look) ?_ | refine wp_umax (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.umin => run `(tactic| with_reducible refine wp_umin (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.bswap => run `(tactic| with_reducible first
      | refine wp_bswap (by look) (bswap_leN (by omega)) ?_ | refine wp_bswap (by look) (by assumption) ?_)
  | some ``AlgorithmLib.Prog.load64 => run `(tactic| with_reducible refine wp_load64 (by look) (by assumption) ?_)
  | some ``AlgorithmLib.Prog.storeI64 => run `(tactic| with_reducible
      refine wp_storeI64 (by look) (by look) (by assumption) ?_)
  | some ``AlgorithmLib.Prog.uload8_64 => run `(tactic| with_reducible refine wp_uload8 (by look) (by assumption) ?_)
  | some ``AlgorithmLib.Prog.ushr => run `(tactic| with_reducible refine wp_ushr (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.ishl => run `(tactic| with_reducible refine wp_ishl (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.iadd => run `(tactic| with_reducible refine wp_iadd (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.isub => run `(tactic| with_reducible refine wp_isub (by look) (by look) ?_)
  | some ``AlgorithmLib.Prog.imul => run `(tactic| with_reducible refine wp_imul (by look) (by look) ?_)
  | _ => run `(tactic| with_reducible step_ext)

/-- An arm's end: the value it joins, in an environment keeping the branch's. -/
macro "arm" : tactic => `(tactic| ((repeat step); refine ⟨by extc, get_eq (by simp only [Vals.head]; look) ?_, rfl⟩))
/-- A join's continuation, from the joined value. -/
macro "joined" : tactic => `(tactic| (intro _ _ _ _; simp only [Vals.head]))
/-- An inner join handed straight on. -/
macro "pass" : tactic => `(tactic| (intro _ _ _ _; rw [wp_pure]; exact ⟨by extc, by simpa only [Vals.head], rfl⟩))

open Lean Elab Tactic Meta in
/-- Clear every hypothesis but the path conditions `hc…`, so `simp_all`
    works on those alone. -/
elab "keep_paths" : tactic => do
  let g ← getMainGoal
  let mut g := g
  for d in (← g.getDecl).lctx.decls.toList.reverse do
    match d with
    | some d =>
      if d.isImplementationDetail then continue
      if d.userName.toString.startsWith "hc" then continue
      try g ← g.clear d.fvarId catch _ => pure ()
    | none => pure ()
  replaceMainGoal [g]

end AlgorithmLib.LibSpec
