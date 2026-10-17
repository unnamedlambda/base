module
public import AlgorithmLib.Check.Sound
meta import AlgorithmLib.Check.Sound
public meta import AlgorithmLib.Check.Reflect
@[expose] public section

/-!
# `Check.Entry` — a shipped body's no-misuse theorem, by the checker

A body is called with `runArgs`: the arena's, the data's and the output's
addresses, and the data's and output's lengths. Its typestate names the facts
the world starts with; a `.roomArg` fact among them says a length is no larger
than its region. `entryOf` is the checker's state at entry: the addresses
known, each length a number bounded by its region where the typestate says
so, and the rest of the typestate as its facts.

**`check_run`**: where the checker accepts a body from `entryOf`, the body
run from `runArgs` in a world the typestate describes neither misuses a call
nor faults. **`check_body`** proves a shipped theorem of that shape: it reads
the typestate off the hypothesis, splits the length facts out, and decides
acceptance in the kernel.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState)

def isRoomArg : Fact → Bool
  | .roomArg _ _ => true
  | _ => false

/-- The checker's state where a body is called with `runArgs`. -/
def entryOf (S : TState) (dLen oLen : Bool) : AS :=
  AS.ofList [.known (.sc .i64 (regionBase .arena)), .known (.sc .i64 (regionBase .data)),
    if dLen then AV.len .data else .sc .i64, .known (.sc .i64 (regionBase .out)),
    if oLen then AV.len .out else .sc .i64] S

theorem AS.ofList_len (l : List AV) (S : TState) : (AS.ofList l S).len = l.length := by
  unfold AS.ofList; rw [AS.pushAll_len]; simp

theorem AS.ofList_facts (l : List AV) (S : TState) : (AS.ofList l S).facts = S := by
  unfold AS.ofList; rw [AS.pushAll_facts]

theorem AS.ofList_syms (l : List AV) (S : TState) (i : Nat) (hi : i < l.length) :
    (AS.ofList l S).syms.get i = {} := by
  unfold AS.ofList; rw [AS.pushAll_syms _ _ _ (by simpa using hi)]; simp

theorem AS.ofList_fits (l : List AV) (S : TState) (Γ : Env) (w : World) :
    ∀ i, i < (AS.ofList l S).len → ((AS.ofList l S).syms.get i).Fits Γ w i (AS.ofList l S).len :=
  fun i hi => by
    rw [AS.ofList_len] at hi
    rw [AS.ofList_syms l S i hi]
    exact Sym.fits_blank _ _ _ _

theorem AS.ofList_get (l : List AV) (S : TState) (i : Nat) : (AS.ofList l S).get i = l.getD i .any := by
  unfold AS.ofList; rw [AS.pushAll_get]; simp

/-- Whether the checker accepts a body from a state, its loops checked against
    the invariants `cert` gives. -/
def accepts (cert : Cert) (A : AS) (c : Code) : Bool :=
  match execC cert [] 0 A c with
  | .ok _ => true
  | .error _ => false

/-- What a proof hands the kernel, found by running the checker natively: the
    invariant each loop settles at, then the state each piece starts from in a
    run that checks each loop once against it; or why the checker refuses the
    body. -/
def certOrRefusal (fo : Bool) (A : AS) (c : Code) : Except String Cert :=
  match execC { fo } [] 0 A c with
  | .ok o =>
      let loops : Cert := { loops := o.certs.loops, fo }
      match execC loops [] 0 A c with
      | .ok o' => .ok { loops := o.certs.loops, states := o'.certs.states, fo }
      | .error _ => .ok loops
  | .error e => .error e

theorem len_desc {Γ : Env} {w : World} {r : Region} (hr : r ≠ .pinned) {x : UInt64}
    (h : (Fact.roomArg r x).holds w) : (AV.len r).Desc Γ w (.sc .i64 x) :=
  ⟨x, rfl, Nat.zero_le _, by have := h.2; unfold span; omega, Nat.one_dvd _, hr,
    by have := h.1; omega⟩

theorem entryOf_holds {S L : TState} {dLen oLen : Bool} {dataLen outLen : UInt64} {w : World}
    (h : L.holds w) (hS : S = L.filter fun x => !isRoomArg x)
    (hd : dLen = true → Fact.roomArg .data dataLen ∈ L) (ho : oLen = true → Fact.roomArg .out outLen ∈ L) :
    (entryOf S dLen oLen).Holds (runArgs dataLen outLen).toArray w := by
  refine ⟨⟨by unfold entryOf; rw [AS.ofList_len]; rfl, AS.ofList_fits _ _ _ _⟩, fun i hi => ?_, fun x hx => h x (by
    unfold entryOf at hx; rw [AS.ofList_facts, hS] at hx; exact (List.mem_filter.mp hx).1)⟩
  unfold entryOf at hi ⊢
  rw [AS.ofList_len] at hi
  rw [AS.ofList_get]
  simp only [List.length_cons, List.length_nil] at hi
  match i, hi with
  | 0, _ => exact ⟨_, rfl, rfl⟩
  | 1, _ => exact ⟨_, rfl, rfl⟩
  | 2, _ =>
      refine ⟨_, rfl, ?_⟩
      show AV.Desc _ _ (if dLen then AV.len .data else .sc .i64) _
      split
      · rename_i e; exact len_desc (by decide) (h _ (hd e))
      · exact TyV.sc (t := .i64) rfl
  | 3, _ => exact ⟨_, rfl, rfl⟩
  | 4, _ =>
      refine ⟨_, rfl, ?_⟩
      show AV.Desc _ _ (if oLen then AV.len .out else .sc .i64) _
      split
      · rename_i e; exact len_desc (by decide) (h _ (ho e))
      · exact TyV.sc (t := .i64) rfl

/-- **A body the checker accepts from its entry never misuses a call and never
    faults**, run from `runArgs` in any world its typestate describes. -/
theorem check_run {cfg : Cfg} {cert : Cert} {S : TState} {dLen oLen : Bool} {c : Code}
    (hc : accepts cert (entryOf S dLen oLen) c = true) (hfo : cert.fo = false) {L : TState} {dataLen outLen : UInt64} {w : World}
    (h : L.holds w) (hS : S = L.filter fun x => !isRoomArg x)
    (hd : dLen = true → Fact.roomArg .data dataLen ∈ L) (ho : oLen = true → Fact.roomArg .out outLen ∈ L)
    (m : String) :
    run cfg (runArgs dataLen outLen) w c ≠ .misuse m ∧ run cfg (runArgs dataLen outLen) w c ≠ .fault m := by
  unfold accepts at hc
  split at hc
  · rename_i o ho'
    exact execC_run ho' hfo (entryOf_holds h hS hd ho) m
  · cases hc

/-- **A body the checker accepts, as one that may fault, never misuses a
    call**, run from `runArgs` in any world its typestate describes. -/
theorem check_safe {cfg : Cfg} {cert : Cert} {S : TState} {dLen oLen : Bool} {c : Code}
    (hc : accepts cert (entryOf S dLen oLen) c = true) {L : TState} {dataLen outLen : UInt64} {w : World}
    (h : L.holds w) (hS : S = L.filter fun x => !isRoomArg x)
    (hd : dLen = true → Fact.roomArg .data dataLen ∈ L) (ho : oLen = true → Fact.roomArg .out outLen ∈ L)
    (m : String) : run cfg (runArgs dataLen outLen) w c ≠ .misuse m := by
  unfold accepts at hc
  split at hc
  · rename_i o ho'
    exact execC_safe ho' (entryOf_holds h hS hd ho) m
  · cases hc

/-- The checker's state where a body is called with arguments it knows
    exactly, the typestate its facts. -/
def entryKnown (S : TState) (args : List V) : AS := AS.ofList (args.map .known) S

theorem entryKnown_holds {S : TState} {args : List V} {w : World} (h : S.holds w) :
    (entryKnown S args).Holds args.toArray w := by
  unfold entryKnown
  refine ⟨⟨by rw [AS.ofList_len]; simp, AS.ofList_fits _ _ _ _⟩, fun i hi => ?_, by rw [AS.ofList_facts]; exact h⟩
  rw [AS.ofList_len, List.length_map] at hi
  rw [AS.ofList_get]
  refine ⟨args[i], by simp [hi], ?_⟩
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hi, Option.map_some,
    Option.getD_some]
  rfl

/-- **A body the checker accepts from arguments it knows never misuses a call
    and never faults**, run from those arguments in any world its typestate
    describes. -/
theorem check_run_known {cfg : Cfg} {cert : Cert} {S : TState} {args : List V} {c : Code}
    (hc : accepts cert (entryKnown S args) c = true) (hfo : cert.fo = false) {w : World} (h : S.holds w)
    (m : String) : run cfg args w c ≠ .misuse m ∧ run cfg args w c ≠ .fault m := by
  unfold accepts at hc
  split at hc
  · rename_i o ho'
    exact execC_run ho' hfo (entryKnown_holds h) m
  · cases hc

/-- A body the checker accepts from arguments it knows, as one that may fault,
    never misuses a call. -/
theorem check_safe_known {cfg : Cfg} {cert : Cert} {S : TState} {args : List V} {c : Code}
    (hc : accepts cert (entryKnown S args) c = true) {w : World} (h : S.holds w) (m : String) :
    run cfg args w c ≠ .misuse m := by
  unfold accepts at hc
  split at hc
  · rename_i o ho'
    exact execC_safe ho' (entryKnown_holds h) m
  · cases hc

/-- Whether the checker accepts a body from a state, the body leaving only by
    falling off its end, in a state whose facts include `S'`. -/
def acceptsTo (cert : Cert) (A : AS) (c : Code) (S' : TState) : Bool :=
  match execC cert [] 0 A c with
  | .ok o => o.brk.isEmpty && o.cont.isEmpty &&
      (match o.ok with
       | some Af => S'.all (Af.facts.contains ·)
       | none => true)
  | .error _ => false

/-- What a body the checker accepted leaves: the facts the checker says, and
    the lengths the caller passed, still within their regions. -/
theorem roomArg_kept {r : Region} {x : UInt64} {w w' : World} (h : (Fact.roomArg r x).holds w)
    (hr : r ≠ .pinned) (hk : Contracts.KeepsRooms w w') : (Fact.roomArg r x).holds w' :=
  ⟨by rw [hk r hr]; exact h.1, h.2⟩

/-- **A body the checker accepts reaches a typestate**: run from `runArgs` in a
    world its entry typestate describes, it neither misuses a call nor faults,
    and where it ends, the world satisfies `L'`. -/
theorem check_triple {cfg : Cfg} {cert : Cert} {S S' : TState} {dLen oLen : Bool} {c : Code}
    (hc : acceptsTo cert (entryOf S dLen oLen) c S' = true) (hfo : cert.fo = false) {L L' : TState} {dataLen outLen : UInt64}
    {w : World} (h : L.holds w) (hS : S = L.filter fun x => !isRoomArg x)
    (hd : dLen = true → Fact.roomArg .data dataLen ∈ L) (ho : oLen = true → Fact.roomArg .out outLen ∈ L)
    (hS' : S' = L'.filter fun x => !isRoomArg x)
    (hR : ∀ x ∈ L', isRoomArg x = true → x ∈ L ∧ ∀ r v, x = .roomArg r v → r ≠ .pinned) :
    Triple cfg (At (runArgs dataLen outLen).toArray w) c { ok := fun _ w' => L'.holds w', faultOk := false } := by
  unfold acceptsTo at hc
  split at hc
  · rename_i o ho'
    simp only [Bool.and_eq_true, List.isEmpty_iff] at hc
    obtain ⟨⟨hb, hcn⟩, hf⟩ := hc
    have ht := execC_sound (cfg := cfg) ho' (entryOf_holds h hS hd ho)
    rw [hfo] at ht
    refine Hoare.conseq ht (fun _ _ h => h) ?_ ?_ ?_
    · rintro Γ w' ⟨⟨Af, hAf, hAh⟩, -, hk⟩ x hx
      by_cases hra : isRoomArg x = true
      · obtain ⟨hxL, hpin⟩ := hR x hx hra
        cases x with
        | roomArg r v => exact roomArg_kept (h _ hxL) (hpin r v rfl) hk
        | _ => simp [isRoomArg] at hra
      · rw [hAf] at hf
        have hm : x ∈ S' := by rw [hS']; exact List.mem_filter.mpr ⟨hx, by simpa using hra⟩
        exact hAh.2.2 x (by simpa using List.all_eq_true.mp hf x hm)
    · rintro d Γ vs w' ⟨⟨A, args, hm, -⟩, -⟩
      rw [hb] at hm; cases hm
    · rintro d vs w' ⟨⟨A, args, Γ, hm, -⟩, -⟩
      rw [hcn] at hm; cases hm
  · cases hc

/-- The elements of a list literal, unfolding definitions on the way. -/
meta partial def listElems (e : Lean.Expr) : Lean.MetaM (List Lean.Expr) := do
  let e ← Lean.Meta.whnf e
  if e.isAppOfArity ``List.cons 3 then return e.getArg! 1 :: (← listElems (e.getArg! 2))
  return []

open Lean Meta Elab Tactic in
/-- Prove `run cfg args w c ≠ .misuse m` (and `≠ .fault m`) from the
    hypothesis that the world satisfies a typestate, by the checker: `args`
    `runArgs`, or a list it knows exactly. A goal that asks only for no misuse
    is checked as a body that may fault. -/
elab "check_body" : tactic => do
  let g ← getMainGoal
  g.withContext do
    let mut found : Option (Expr × Expr) := none
    for d in ← getLCtx do
      if d.isImplementationDetail then continue
      let t ← whnfR (← instantiateMVars d.type)
      if t.isAppOfArity ``Contracts.TState.holds 2 then found := some (d.toExpr, t.getArg! 0)
    let some (h, L) := found | throwError "check_body: no hypothesis that a typestate holds"
    let tgt ← instantiateMVars (← g.getType)
    let some runE := tgt.find? (fun e => e.isAppOfArity ``Sem.run 4)
      | throwError "check_body: the goal runs no body"
    let code := runE.getArg! 3
    if code.hasFVar then throwError "check_body: the body names a variable"
    let args := runE.getArg! 1
    -- a goal that a run does not misuse a call, and no more: the body may fault
    let fo := !tgt.consumeMData.isAppOf ``And
    -- arguments the goal names exactly: each slot known, the typestate whole
    unless args.isAppOf ``runArgs do
      if args.hasFVar then throwError "check_body: the arguments name a variable"
      let L ← instantiateMVars L
      if L.hasFVar then throwError "check_body: the typestate names a variable"
      let A := mkApp2 (mkConst ``entryKnown) L args
      let cert ← match ← unsafe evalExpr (Except String Cert)
          (mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) (mkConst ``Cert))
          (mkApp3 (mkConst ``certOrRefusal) (toExpr fo) A code) with
        | .ok c => pure c
        | .error e => throwError "check_body: the checker refuses this body:\n{e}"
      let sCert ← Term.exprToSyntax (toExpr cert)
      let sL ← Term.exprToSyntax L
      let sa ← Term.exprToSyntax args
      let sh ← Term.exprToSyntax h
      let sc ← Term.exprToSyntax code
      evalTactic (← `(tactic|
        have hacc : accepts $sCert (entryKnown $sL $sa) $sc = true := by decide +kernel))
      if fo then evalTactic (← `(tactic| exact check_safe_known hacc $sh _))
      else evalTactic (← `(tactic| exact check_run_known hacc rfl $sh _))
      return
    let mut keep : Array Expr := #[]
    let mut dLen := false
    let mut oLen := false
    for f in ← listElems L do
      let f ← instantiateMVars f
      if f.isAppOfArity ``Fact.roomArg 2 then
        let r ← whnf (f.getArg! 0)
        if r.isConstOf ``Region.data then dLen := true
        else if r.isConstOf ``Region.out then oLen := true
      else
        if f.hasFVar then throwError "check_body: the typestate fact {f} names a variable"
        keep := keep.push f
    let S ← mkListLit (mkConst ``Fact) keep.toList
    let A := mkApp3 (mkConst ``entryOf) S (toExpr dLen) (toExpr oLen)
    let cert ← match ← unsafe evalExpr (Except String Cert)
        (mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) (mkConst ``Cert))
        (mkApp3 (mkConst ``certOrRefusal) (toExpr fo) A code) with
      | .ok c => pure c
      | .error e => throwError "check_body: the checker refuses this body:\n{e}"
    let sCert ← Term.exprToSyntax (toExpr cert)
    let sS ← Term.exprToSyntax S
    let sh ← Term.exprToSyntax h
    let sc ← Term.exprToSyntax code
    let sd ← if dLen then `(true) else `(false)
    let so ← if oLen then `(true) else `(false)
    evalTactic (← `(tactic|
      have hacc : accepts $sCert (entryOf $sS $sd $so) $sc = true := by decide +kernel))
    if fo then evalTactic (← `(tactic| exact check_safe hacc $sh rfl (by simp) (by simp) _))
    else evalTactic (← `(tactic| exact check_run hacc rfl $sh rfl (by simp) (by simp) _))

open Lean Meta Elab Tactic in
/-- The typestate a hypothesis says holds, split into its closed facts, and
    whether it gives the data's and the output's lengths. -/
meta def readTypestate (L : Expr) : MetaM (Expr × Bool × Bool) := do
  let mut keep : Array Expr := #[]
  let mut dLen := false
  let mut oLen := false
  for f in ← listElems L do
    let f ← instantiateMVars f
    if f.isAppOfArity ``Fact.roomArg 2 then
      let r ← whnf (f.getArg! 0)
      if r.isConstOf ``Region.data then dLen := true
      else if r.isConstOf ``Region.out then oLen := true
    else
      if f.hasFVar then throwError "the typestate fact {f} names a variable"
      keep := keep.push f
  return (← mkListLit (mkConst ``Fact) keep.toList, dLen, oLen)

open Lean Meta Elab Tactic in
/-- Prove `Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) c
    { ok := fun _ w' => L'.holds w', faultOk := false }` from the hypothesis
    that the world satisfies a typestate, by the checker. -/
elab "check_reach" : tactic => do
  let g ← getMainGoal
  g.withContext do
    let mut found : Option (Expr × Expr) := none
    for d in ← getLCtx do
      if d.isImplementationDetail then continue
      let t ← whnfR (← instantiateMVars d.type)
      if t.isAppOfArity ``Contracts.TState.holds 2 then found := some (d.toExpr, t.getArg! 0)
    let some (h, L) := found | throwError "check_reach: no hypothesis that a typestate holds"
    let (S, dLen, oLen) ← readTypestate L
    let tgt ← instantiateMVars (← g.getType)
    unless tgt.isAppOfArity ``Hoare.Triple 4 do throwError "check_reach: the goal is not a triple"
    let code := tgt.getArg! 2
    let post ← whnfR (tgt.getArg! 3)
    let okF := post.getArg! 0
    let L' ← lambdaTelescope okF fun xs body => do
      let b ← whnfR body
      unless b.isAppOfArity ``Contracts.TState.holds 2 do throwError "check_reach: the post is not a typestate"
      let L' := b.getArg! 0
      if xs.any (fun x => L'.containsFVar x.fvarId!) then throwError "check_reach: the typestate reads the state"
      return L'
    let (S', _, _) ← readTypestate L'
    if code.hasFVar then throwError "check_reach: the body names a variable"
    let A := mkApp3 (mkConst ``entryOf) S (toExpr dLen) (toExpr oLen)
    let cert ← match ← unsafe evalExpr (Except String Cert)
        (mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) (mkConst ``Cert))
        (mkApp3 (mkConst ``certOrRefusal) (toExpr false) A code) with
      | .ok c => pure c
      | .error e => throwError "check_reach: the checker refuses this body:\n{e}"
    let sCert ← Term.exprToSyntax (toExpr cert)
    let sS ← Term.exprToSyntax S
    let sS' ← Term.exprToSyntax S'
    let sh ← Term.exprToSyntax h
    let sc ← Term.exprToSyntax code
    let sd ← if dLen then `(true) else `(false)
    let so ← if oLen then `(true) else `(false)
    evalTactic (← `(tactic|
      have hacc : acceptsTo $sCert (entryOf $sS $sd $so) $sc $sS' = true := by decide +kernel))
    evalTactic (← `(tactic| exact check_triple hacc rfl $sh rfl (by simp) (by simp) rfl (by simp [isRoomArg])))

end AlgorithmLib.Check
