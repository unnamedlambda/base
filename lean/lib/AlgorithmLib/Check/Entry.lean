module
public import AlgorithmLib.Check.Sound
meta import AlgorithmLib.Check.Sound
public meta import AlgorithmLib.Check.Reflect
@[expose] public section

/-!
# `Check.Entry` — a shipped body's no-misuse theorem, by the checker

A body is called with `runArgs`: the data's and the output's addresses and
lengths; it takes the arena's address itself, as its first statement. Its typestate names the facts
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

variable {vals : List UInt64}

open AlgorithmLib.HProg.Contracts (Fact TState)

def isRoomArg : Fact → Bool
  | .roomArg _ _ => true
  | _ => false

/-- The checker's state where a body is called with `runArgs`, `P` what it
    knows of the program's parameters. -/
def entryOf (S : TState) (dLen oLen : Bool) (P : Par) : AS :=
  (AS.ofList [.known (.sc .i64 (regionBase .data)),
    if dLen then AV.len .data else .sc .i64, .known (.sc .i64 (regionBase .out)),
    if oLen then AV.len .out else .sc .i64] S).withPar P

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

theorem AS.ofList_par_at (l : List AV) (S : TState) (Γ : Env) (w : World) :
    (AS.ofList l S).par.At vals Γ w := by
  unfold AS.ofList; rw [AS.pushAll_par]
  exact ⟨rfl, fun _ e => by simp at e, fun _ e => by simp [Par.facts, bufFacts, allocFacts] at e, fun _ e => by simp at e,
    fun _ e => by simp at e, fun _ e => by simp at e⟩

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
    body. `lib` says calls into the libraries the program is linked with are
    read by the libraries' statements. -/
def certOrRefusal (fo : Bool) (A : AS) (c : Code) (lib : Bool := false) : Except String Cert :=
  match execC { fo, lib } [] 0 A c with
  | .ok o =>
      let loops : Cert := { loops := o.certs.loops, fo, lib }
      match execC loops [] 0 A c with
      | .ok o' => .ok { loops := o.certs.loops, states := o'.certs.states, fo, lib }
      | .error _ => .ok loops
  | .error e => .error e

/-- The calls whose preconditions the parameters decide, gathered by a run of
    the checker that gathers them; or why it refuses the body. -/
def oblsOf (fo : Bool) (A : AS) (c : Code) (lib : Bool := false) : Except String (List Obl) :=
  match execC { fo, lib } [] 0 (A.withPar { A.par with collect := true }) c with
  | .ok o =>
      let states := o.ok.toList ++ o.brk.map (·.2.1) ++ o.cont.map (·.2.1)
      .ok (states.flatMap (·.par.obls)).eraseDups
  | .error e => .error e

theorem len_desc {Γ : Env} {w : World} {r : Region} (hr : r ≠ .pinned) {x : UInt64}
    (h : (Fact.roomArg r x).holds w) : (AV.len r).Desc Γ w (.sc .i64 x) :=
  ⟨x, rfl, Nat.zero_le _, by have := h.2; unfold span; omega, Nat.one_dvd _, hr,
    by have := h.1; omega⟩

/-- The facts a list of positions picks out of a typestate hold where it does. -/
theorem holds_pick {L : TState} {w : World} (h : L.holds w) (idx : List Nat) :
    TState.holds (idx.filterMap (L[·]?)) w := fun x hx => by
  obtain ⟨i, -, hi⟩ := List.mem_filterMap.mp hx
  exact h x (List.mem_of_getElem? hi)

/-- **The checker's state at entry describes the call**: `S` the facts picked
    out of the typestate, the lengths bounded where it says so, and the
    parameters' record holding at `vals`, its cells and buffers picked out of
    the typestate, the lengths among its slots where the typestate names
    them by variables, and its obligations shown. -/
theorem entryOf_holds {L S : TState} {dLen oLen : Bool} {P : Par} {dataLen outLen : UInt64} {w : World}
    (h : L.holds w) (idx : List Nat) (hS : idx.filterMap (L[·]?) = S)
    (hd : dLen = true → Fact.roomArg .data dataLen ∈ L) (ho : oLen = true → Fact.roomArg .out outLen ∈ L)
    (pidx : List Nat) (hP : pidx.filterMap (L[·]?) = P.facts vals) (hc : P.collect = false)
    (hs : ∀ s ∈ P.slots, (runArgs dataLen outLen).toArray[s.1]? = some (.sc s.2.1 (s.2.2.eval vals)))
    (hx : P.ctxs = []) (hq : P.conds = []) (hob : ∀ o ∈ P.obls, o.Holds vals) :
    (entryOf S dLen oLen P).Holds vals (runArgs dataLen outLen).toArray w := by
  unfold entryOf
  refine AS.Holds.withPar ⟨⟨by rw [AS.ofList_len]; rfl, AS.ofList_fits _ _ _ _, AS.ofList_par_at _ _ _ _⟩,
    fun i hi => ?_, by rw [AS.ofList_facts, ← hS]; exact holds_pick h idx⟩
    ⟨hc, hs, by rw [← hP]; exact holds_pick h pidx, hob, by simp [hx], by simp [hq]⟩
  rw [AS.ofList_len] at hi
  rw [AS.ofList_get]
  simp only [List.length_cons, List.length_nil] at hi
  match i, hi with
  | 0, _ => exact ⟨_, rfl, rfl⟩
  | 1, _ =>
      refine ⟨_, rfl, ?_⟩
      show AV.Desc _ _ (if dLen then AV.len .data else .sc .i64) _
      split
      · rename_i e; exact len_desc (by decide) (h _ (hd e))
      · exact TyV.sc (t := .i64) rfl
  | 2, _ => exact ⟨_, rfl, rfl⟩
  | 3, _ =>
      refine ⟨_, rfl, ?_⟩
      show AV.Desc _ _ (if oLen then AV.len .out else .sc .i64) _
      split
      · rename_i e; exact len_desc (by decide) (h _ (ho e))
      · exact TyV.sc (t := .i64) rfl

/-- Each of a list of obligations shown, one by one. -/
def OblsHold (vals : List UInt64) : List Obl → Prop
  | [] => True
  | o :: os => o.Holds vals ∧ OblsHold vals os

theorem obls_of : ∀ {os : List Obl}, OblsHold vals os → ∀ o ∈ os, o.Holds vals
  | [], _, _, e => nomatch e
  | _ :: _, ⟨h, _⟩, _, .head _ => h
  | _ :: _, ⟨_, hs⟩, o, .tail _ e => obls_of hs o e

theorem argBits_known : ∀ (bits : List UInt64) (Γ : Env) (w : World),
    ArgBits Γ w (bits.map fun x => .known (.sc .i64 x)) bits
  | [], _, _ => trivial
  | _ :: bits, Γ, w => ⟨⟨.i64, rfl⟩, argBits_known bits Γ w⟩

/-- **An obligation the checker's own decision shows**: where `preB` decides
    the call, handed the arguments' bits, in the obligation's typestate at the
    parameters' values. -/
theorem Obl.holds_of_preB {o : Obl} {f : Ffi} (hc : o.c = .ffi f)
    (h : preB f ((o.args.map (PArg.bits vals)).map fun x => .known (.sc .i64 x))
      (o.facts ++ bufFacts o.bufs vals) = true) : o.Holds vals := by
  unfold Obl.Holds
  rw [hc]
  exact fun _ w hw => preB_sound hw (argBits_known _ ∅ w) h fun _ _ e => by cases e

/-- **A body the checker accepts never misuses a call and never faults**, run
    from any arguments and world its state describes. -/
theorem check_run {cfg : Cfg} {cert : Cert} [Linked cert.lib cfg] {A : AS} {c : Code} (hc : accepts cert A c = true)
    (hfo : cert.fo = false) {args : List V} {w : World} (hA : A.Holds vals args.toArray w) (m : String) :
    run cfg args w c ≠ .misuse m ∧ run cfg args w c ≠ .fault m := by
  unfold accepts at hc
  split at hc
  · rename_i o ho'
    exact execC_run ho' hfo hA m
  · cases hc

/-- **A body the checker accepts, as one that may fault, never misuses a
    call**, run from any arguments and world its state describes. -/
theorem check_safe {cfg : Cfg} {cert : Cert} [Linked cert.lib cfg] {A : AS} {c : Code} (hc : accepts cert A c = true)
    {args : List V} {w : World} (hA : A.Holds vals args.toArray w) (m : String) : run cfg args w c ≠ .misuse m := by
  unfold accepts at hc
  split at hc
  · rename_i o ho'
    exact execC_safe ho' hA m
  · cases hc

/-- The pattern the checker gives the calls of a body it accepts, written
    shorter. -/
def traceOf (cert : Cert) (A : AS) (c : Code) : Option Tr :=
  match execC cert [] 0 A c with
  | .ok o => some o.tr.norm
  | .error _ => none

/-- **A body the checker accepts makes calls its pattern admits**: run from
    any arguments and world its state describes, where it ends the calls it
    observed are those made before it and then ones the pattern admits. -/
theorem check_calls_of {cfg : Cfg} {cert : Cert} [Linked cert.lib cfg] {A : AS} {c : Code} {t : Tr} (ht : traceOf cert A c = some t)
    {args : List V} {w : World} (hA : A.Holds vals args.toArray w) {obs : List Obs} {w' : World}
    (hr : run cfg args w c = .ok obs w') : t.Covers w obs := by
  unfold traceOf at ht
  split at ht
  · rename_i o ho
    cases ht
    obtain ⟨cs, e, hh⟩ := execC_trace ho hA hr
    exact ⟨cs, e, hh.norm⟩
  · cases ht

/-- The checker's state where a body is called with arguments it knows
    exactly, the typestate its facts. -/
def entryKnown (S : TState) (args : List V) : AS := AS.ofList (args.map .known) S

theorem entryKnown_holds {S : TState} {args : List V} {w : World} (h : S.holds w) :
    (entryKnown S args).Holds vals args.toArray w := by
  unfold entryKnown
  refine ⟨⟨by rw [AS.ofList_len]; simp, AS.ofList_fits _ _ _ _, AS.ofList_par_at _ _ _ _⟩, fun i hi => ?_,
    by rw [AS.ofList_facts]; exact h⟩
  rw [AS.ofList_len, List.length_map] at hi
  rw [AS.ofList_get]
  refine ⟨args[i], by simp [hi], ?_⟩
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hi, Option.map_some,
    Option.getD_some]
  rfl

/-- The checker's state where a body is called with arguments each of which
    one of `avs` describes, the typestate its facts: how a library function is
    entered. -/
def entryWith (S : TState) (avs : List AV) : AS := AS.ofList avs S

theorem entryWith_holds {S : TState} {avs : List AV} {args : List V} {w : World} (h : S.holds w)
    (hd : DescAll args.toArray w avs args) : (entryWith S avs).Holds vals args.toArray w := by
  unfold entryWith
  refine ⟨⟨by rw [AS.ofList_len, hd.length]; simp, AS.ofList_fits _ _ _ _, AS.ofList_par_at _ _ _ _⟩,
    fun i hi => ?_, by rw [AS.ofList_facts]; exact h⟩
  rw [AS.ofList_len] at hi
  rw [AS.ofList_get]
  obtain ⟨v, hv, hdv⟩ := hd.get hi
  exact ⟨v, by simpa using hv, hdv⟩

/-- A pointer to `n` writable bytes of region `r`: how a caller hands over a
    place to write. -/
def AV.room (r : Region) (n : Nat) : AV := .ptr r ⟨.i64, 0, span - n, 1, some (r, 1, n)⟩

theorem AV.room_desc {Γ : Env} {w : World} {r : Region} {n off : Nat} (hr : r ≠ .pinned)
    (hn : n ≤ span) (h : off + n ≤ w.mem.sizes r) (hs : off + n ≤ span) :
    (AV.room r n).Desc Γ w (.sc .i64 (regionBase r + UInt64.ofNat off)) :=
  ⟨off, rfl, ⟨Nat.zero_le _, by show off ≤ span - n; omega, Nat.one_dvd _, hr, by omega⟩,
    by show span - n ≤ span; omega⟩

/-- Whether the checker accepts a body from a state, the body leaving only by
    falling off its end, in a state whose facts include `S'` and whose record
    of the parameters names the cells and buffers `P'` does. -/
def acceptsTo (cert : Cert) (A : AS) (c : Code) (S' : TState) (P' : Par) : Bool :=
  match execC cert [] 0 A c with
  | .ok o => o.brk.isEmpty && o.cont.isEmpty &&
      (match o.ok with
       | some Af => S'.all (Af.facts.contains ·) && P'.cells.all (Af.par.cells.contains ·) &&
           P'.bufs.all (Af.par.bufs.contains ·) && P'.allocs.all (Af.par.allocs.contains ·)
       | none => true)
  | .error _ => false

/-- What a body the checker accepted leaves: the facts the checker says, and
    the lengths the caller passed, still within their regions. -/
theorem roomArg_kept {r : Region} {x : UInt64} {w w' : World} (h : (Fact.roomArg r x).holds w)
    (hr : r ≠ .pinned) (hk : Contracts.KeepsRooms w w') : (Fact.roomArg r x).holds w' :=
  ⟨by rw [hk r hr]; exact h.1, h.2⟩

/-- Whether no length a typestate states is the pinned region's. -/
def roomsUnpinned (L : TState) : Bool :=
  L.all fun | .roomArg r _ => r != .pinned | _ => true

/-- A post that is the pre states its lengths where the pre does. -/
theorem rooms_same {L : TState} (h : roomsUnpinned L = true) :
    ∀ x ∈ L, isRoomArg x = true → x ∈ L ∧ ∀ r v, x = .roomArg r v → r ≠ .pinned := by
  intro x hx _
  refine ⟨hx, ?_⟩
  rintro r v rfl
  have := List.all_eq_true.mp h _ hx
  intro e
  subst e
  have h2 : (Region.pinned != Region.pinned) = true := this
  exact absurd h2 (by decide)

/-- A typestate after a body: its lengths those the caller passed, its other
    facts picked out of `S'` and `T`. -/
theorem post_of {L L' S' T : TState} {w w' : World} (h : L.holds w) (hk : Contracts.KeepsRooms w w')
    (hS' : S'.holds w') (hT : T.holds w') (plan : List Nat)
    (hn : plan.filterMap ((S' ++ T)[·]?) = L'.filter fun x => !isRoomArg x)
    (hR : ∀ x ∈ L', isRoomArg x = true → x ∈ L ∧ ∀ r v, x = .roomArg r v → r ≠ .pinned) : L'.holds w' := by
  intro x hx
  by_cases hra : isRoomArg x = true
  · obtain ⟨hxL, hpin⟩ := hR x hx hra
    cases x with
    | roomArg r v => exact roomArg_kept (h _ hxL) (hpin r v rfl) hk
    | _ => simp [isRoomArg] at hra
  · have hm : x ∈ plan.filterMap ((S' ++ T)[·]?) := by
      rw [hn]; exact List.mem_filter.mpr ⟨hx, by simpa using hra⟩
    obtain ⟨i, -, hi⟩ := List.mem_filterMap.mp hm
    rcases List.mem_append.mp (List.mem_of_getElem? hi) with hm | hm
    · exact hS' x hm
    · exact hT x hm

/-- **A body the checker accepts reaches a typestate**: run from arguments and
    a world its state describes, it neither misuses a call nor faults, and
    where it ends, the world satisfies `L'`. -/
theorem check_triple {cfg : Cfg} {cert : Cert} [Linked cert.lib cfg] {A : AS} {c : Code} {S' : TState} {P' : Par}
    (hc : acceptsTo cert A c S' P' = true) (hfo : cert.fo = false) {args : List V} {w : World}
    (hA : A.Holds vals args.toArray w) {L' : TState}
    (hpost : ∀ w', Contracts.KeepsRooms w w' → S'.holds w' → (P'.facts vals).holds w' → L'.holds w') :
    Triple cfg (At args.toArray w) c { ok := fun _ w' => L'.holds w', faultOk := false } := by
  unfold acceptsTo at hc
  split at hc
  · rename_i o ho'
    simp only [Bool.and_eq_true, List.isEmpty_iff] at hc
    obtain ⟨⟨hb, hcn⟩, hf⟩ := hc
    have ht := execC_sound (cfg := cfg) ho' hA
    rw [hfo] at ht
    refine Hoare.conseq ht (fun _ _ h => h) ?_ ?_ ?_
    · rintro Γ w' ⟨⟨Af, hAf, hAh⟩, -, hk⟩
      rw [hAf] at hf
      simp only [Bool.and_eq_true, List.all_eq_true] at hf
      obtain ⟨⟨⟨hfs, hfc⟩, hfb⟩, hfa⟩ := hf
      refine hpost w' hk.rooms (fun x hx => hAh.2.2 x (by simpa using hfs x hx)) fun x hx => ?_
      have hP := hAh.1.2.2.2.2.1
      simp only [Par.facts, bufFacts, allocFacts, List.mem_append, List.mem_map] at hx
      rcases hx with (⟨p, hp, rfl⟩ | ⟨p, hp, rfl⟩) | ⟨p, hp, rfl⟩
      · exact hP _ (List.mem_append_left _ (List.mem_append_left _
          (List.mem_map_of_mem (List.contains_iff_mem.mp (hfc p hp)))))
      · exact hP _ (List.mem_append_left _ (List.mem_append_right _
          (List.mem_map_of_mem (List.contains_iff_mem.mp (hfb p hp)))))
      · exact hP _ (List.mem_append_right _ (List.mem_map_of_mem (List.contains_iff_mem.mp (hfa p hp))))
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

open Lean Meta in
/-- A number the typestate computes from its variables, as the parameters'
    expression, the variables met so far `vs`. -/
meta partial def toPExpr (vs : Array Expr) (e : Expr) : MetaM (PExpr × Array Expr) := do
  if e.isFVar then
    match vs.findIdx? (· == e) with
    | some i => return (.param i, vs)
    | none => return (.param vs.size, vs.push e)
  if !e.hasFVar then
    let c ← unsafe evalExpr UInt64 (mkConst ``UInt64) e
    return (.lit c, vs)
  let e' ← whnfR e
  if e'.isAppOfArity ``HAdd.hAdd 6 then
    let (a, vs) ← toPExpr vs (e'.getArg! 4)
    let (b, vs) ← toPExpr vs (e'.getArg! 5)
    return (.add a b, vs)
  if e'.isAppOfArity ``HSub.hSub 6 then
    let (a, vs) ← toPExpr vs (e'.getArg! 4)
    let (b, vs) ← toPExpr vs (e'.getArg! 5)
    return (.sub a b, vs)
  if e'.isAppOfArity ``HMul.hMul 6 then
    let (a, vs) ← toPExpr vs (e'.getArg! 4)
    let (b, vs) ← toPExpr vs (e'.getArg! 5)
    return (.mul a b, vs)
  if e'.isAppOfArity ``HShiftLeft.hShiftLeft 6 && !(e'.getArg! 5).hasFVar then
    let (a, vs) ← toPExpr vs (e'.getArg! 4)
    let k ← unsafe evalExpr UInt64 (mkConst ``UInt64) (e'.getArg! 5)
    return (.shl a k.toNat, vs)
  throwError "the typestate computes {e} in a way the parameters' record does not name"

open Lean Meta in
/-- A typestate read for the checker: the closed facts and where each sits,
    whether it gives the data's and the output's lengths, the cells and
    buffers the variables `vs` fix and where each sits, and the variables met.
    Read at `entry`, a length it names by a variable is a parameter its slot
    holds. -/
meta def readTypestate (L : Expr) (vs0 : Array Expr) (entry : Bool := true) :
    MetaM (List Expr × List Nat × Bool × Bool × Par × List Nat × List Expr × Array Expr) := do
  let mut keep : Array Expr := #[]
  let mut idx : Array Nat := #[]
  let mut dLen := false
  let mut oLen := false
  let mut cells : Array ((UInt64 × Nat) × Nat) := #[]
  let mut bufs : Array ((UInt64 × PExpr) × Nat) := #[]
  let mut allocs : Array ((Nat × PExpr) × Nat) := #[]
  let mut lens : Array (R × ClifTy × PExpr) := #[]
  let mut vs := vs0
  let mut i := 0
  for f in ← listElems L do
    let f ← instantiateMVars f
    if f.isAppOfArity ``Fact.roomArg 2 then
      let r ← whnf (f.getArg! 0)
      let slot? : Option Nat :=
        if r.isConstOf ``Region.data then some 1 else if r.isConstOf ``Region.out then some 3 else none
      if r.isConstOf ``Region.data then dLen := true
      else if r.isConstOf ``Region.out then oLen := true
      -- a length the typestate names by a variable is a parameter, read from its slot
      if let some slot := slot? then
        if entry && (f.getArg! 1).isFVar then
          let (p, vs') ← toPExpr vs (f.getArg! 1)
          vs := vs'
          lens := lens.push (slot, .i64, p)
    else if !f.hasFVar then
      keep := keep.push f
      idx := idx.push i
    else if f.isAppOfArity ``Fact.cell 2 && !(f.getArg! 0).hasFVar && (f.getArg! 1).isFVar then
      let a ← unsafe evalExpr UInt64 (mkConst ``UInt64) (f.getArg! 0)
      let (p, vs') ← toPExpr vs (f.getArg! 1)
      vs := vs'
      let .param k := p | throwError "unreachable"
      cells := cells.push ((a, k), i)
    else if f.isAppOfArity ``Fact.devBuf 2 && !(f.getArg! 0).hasFVar then
      let b ← unsafe evalExpr UInt64 (mkConst ``UInt64) (f.getArg! 0)
      let (e, vs') ← toPExpr vs (f.getArg! 1)
      vs := vs'
      bufs := bufs.push ((b, e), i)
    else if f.isAppOfArity ``Fact.alloc 2 && !(f.getArg! 0).hasFVar &&
        (f.getArg! 1).isAppOfArity ``UInt64.toNat 1 then
      let a ← unsafe evalExpr Nat (mkConst ``Nat) (f.getArg! 0)
      let (e, vs') ← toPExpr vs ((f.getArg! 1).getArg! 0)
      vs := vs'
      allocs := allocs.push ((a, e), i)
    else throwError "the typestate fact {f} names a variable the checker cannot place"
    i := i + 1
  let P : Par := { cells := (cells.map (·.1)).toList, bufs := (bufs.map (·.1)).toList,
                   allocs := (allocs.map (·.1)).toList, slots := lens.toList }
  let pidx := (cells.map (·.2) ++ bufs.map (·.2) ++ allocs.map (·.2)).toList
  return (keep.toList, idx.toList, dLen, oLen, P, pidx, pidx.map (mkNatLit ·), vs)

open Lean Meta Elab Tactic in
/-- Run the checker natively: the obligations the parameters leave, gathered,
    then the certificate; or why the checker refuses the body. -/
meta def runChecker (who : String) (fo : Bool) (S : Expr) (dLen oLen : Bool) (P : Par) (code : Expr)
    (lib : Bool := false) : MetaM (Par × Cert) := do
  let mkA (P : Par) := mkApp4 (mkConst ``entryOf) S (toExpr dLen) (toExpr oLen) (toExpr P)
  let P ← if P.cells.isEmpty && P.bufs.isEmpty && P.allocs.isEmpty && P.slots.isEmpty then pure P else
    match ← unsafe evalExpr (Except String (List Obl))
        (mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) (mkApp (mkConst ``List [0]) (mkConst ``Obl)))
        (mkApp4 (mkConst ``oblsOf) (toExpr fo) (mkA P) code (toExpr lib)) with
    | .ok os => pure { P with obls := os }
    | .error e => throwError "{who}: the checker refuses this body:\n{e}"
  match ← unsafe evalExpr (Except String Cert)
      (mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) (mkConst ``Cert))
      (mkApp4 (mkConst ``certOrRefusal) (toExpr fo) (mkA P) code (toExpr lib)) with
  | .ok c => return (P, c)
  | .error e => throwError "{who}: the checker refuses this body:\n{e}"

open Lean Meta Elab Tactic in
/-- The hypothesis that a typestate holds. -/
meta def findTypestate (who : String) : TacticM (Expr × Expr) := do
  let mut found : Option (Expr × Expr) := none
  for d in ← getLCtx do
    if d.isImplementationDetail then continue
    let t ← whnfR (← instantiateMVars d.type)
    if t.isAppOfArity ``Contracts.TState.holds 2 then found := some (d.toExpr, t.getArg! 0)
  let some r := found | throwError "{who}: no hypothesis that a typestate holds"
  return r

open Lean Meta Elab Tactic in
/-- Whether the goal's program is linked with the libraries: a hypothesis or
    an instance `Linked true cfg` is in scope. -/
meta def findLinked : TacticM Bool := do
  for d in ← getLCtx do
    if d.isImplementationDetail then continue
    let t ← whnfR (← instantiateMVars d.type)
    if t.isAppOfArity ``Linked 2 && (t.getArg! 0).isConstOf ``Bool.true then return true
  return false

open Lean Meta Elab Tactic in
/-- The proof that the entry state describes the call, its obligations left
    as goals. -/
meta def entryProof (h S : Expr) (idx : List Nat) (dLen oLen : Bool) (P : Par) (pidx : List Nat)
    (vs : Array Expr) : TacticM (TSyntax `term) := do
  let sh ← Term.exprToSyntax h
  let sS ← Term.exprToSyntax S
  let sP ← Term.exprToSyntax (toExpr P)
  let sv ← Term.exprToSyntax (← mkListLit (mkConst ``UInt64) vs.toList)
  let si ← Term.exprToSyntax (toExpr idx)
  let sp ← Term.exprToSyntax (toExpr pidx)
  let sd ← if dLen then `(true) else `(false)
  let so ← if oLen then `(true) else `(false)
  let hob ← if P.obls.isEmpty then `(fun _ e => nomatch e) else
    let holes ← (List.range P.obls.length).mapM fun _ => `(?_)
    `(obls_of ⟨$(holes.toArray),*, trivial⟩)
  `((entryOf_holds (vals := $sv) (S := $sS) (dLen := $sd) (oLen := $so) (P := $sP) $sh $si (by rfl)
      (by simp) (by simp) $sp (by rfl) rfl (by simp [runArgs, PExpr.eval]) rfl rfl $hob))

open Lean Meta Elab Tactic in
/-- Prove `run cfg args w c ≠ .misuse m` (and `≠ .fault m`) from the
    hypothesis that the world satisfies a typestate, by the checker: `args`
    `runArgs`, or a list it knows exactly. A goal that asks only for no misuse
    is checked as a body that may fault. Where the typestate names variables,
    the calls whose preconditions they decide are left as goals. Where
    `Linked true cfg` is in scope, calls into the libraries are read by the
    libraries' statements. -/
meta def checkBodyCore (compiled : Bool) : TacticM Unit := do
  let dec ← if compiled then `(Lean.Parser.Tactic.tacticSeq| native_decide)
    else `(Lean.Parser.Tactic.tacticSeq| decide +kernel)
  let g ← getMainGoal
  g.withContext do
    let (h, L) ← findTypestate "check_body"
    let tgt ← instantiateMVars (← g.getType)
    let some runE := tgt.find? (fun e => e.isAppOfArity ``Sem.run 4)
      | throwError "check_body: the goal runs no body"
    let code := runE.getArg! 3
    if code.hasFVar then throwError "check_body: the body names a variable"
    let args := runE.getArg! 1
    -- a goal that a run does not misuse a call, and no more: the body may fault
    let fo := !tgt.consumeMData.isAppOf ``And
    -- a program linked with the libraries: calls into them are read
    let lib ← findLinked
    -- arguments the goal names exactly: each slot known, the typestate whole
    unless args.isAppOf ``runArgs do
      if args.hasFVar then throwError "check_body: the arguments name a variable"
      let L ← instantiateMVars L
      if L.hasFVar then throwError "check_body: the typestate names a variable"
      let A := mkApp2 (mkConst ``entryKnown) L args
      let cert ← match ← unsafe evalExpr (Except String Cert)
          (mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) (mkConst ``Cert))
          (mkApp4 (mkConst ``certOrRefusal) (toExpr fo) A code (toExpr lib)) with
        | .ok c => pure c
        | .error e => throwError "check_body: the checker refuses this body:\n{e}"
      let sCert ← Term.exprToSyntax (toExpr cert)
      let sA ← Term.exprToSyntax A
      let sh ← Term.exprToSyntax h
      let sc ← Term.exprToSyntax code
      evalTactic (← `(tactic| have hacc : accepts $sCert $sA $sc = true := by $dec))
      if fo then evalTactic (← `(tactic| exact check_safe (vals := []) hacc (entryKnown_holds $sh) _))
      else evalTactic (← `(tactic| exact check_run (vals := []) hacc rfl (entryKnown_holds $sh) _))
      return
    let (keep, idx, dLen, oLen, P, pidx, _, vs) ← readTypestate L #[]
    let S ← mkListLit (mkConst ``Fact) keep
    let (P, cert) ← runChecker "check_body" fo S dLen oLen P code lib
    let sCert ← Term.exprToSyntax (toExpr cert)
    let sA ← Term.exprToSyntax (mkApp4 (mkConst ``entryOf) S (toExpr dLen) (toExpr oLen) (toExpr P))
    let sc ← Term.exprToSyntax code
    evalTactic (← `(tactic| have hacc : accepts $sCert $sA $sc = true := by $dec))
    let hA ← entryProof h S idx dLen oLen P pidx vs
    if fo then evalTactic (← `(tactic| refine check_safe hacc $hA _))
    else evalTactic (← `(tactic| refine check_run hacc rfl $hA _))

open Lean Meta Elab Tactic in
elab "check_body" : tactic => checkBodyCore false

open Lean Meta Elab Tactic in
/-- `check_body`, the checker's acceptance decided by compiled evaluation
    rather than by the kernel: for a body too long for the kernel to
    evaluate. The proof then also rests on the compiler (`Lean.ofReduceBool`). -/
elab "check_body" &" native" : tactic => checkBodyCore true

open Lean Meta Elab Tactic in
/-- Prove `Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) c
    { ok := fun _ w' => L'.holds w', faultOk := false }` from the hypothesis
    that the world satisfies a typestate, by the checker. Where the
    typestates name variables, the calls whose preconditions they decide are
    left as goals. -/
elab "check_reach" : tactic => do
  let g ← getMainGoal
  g.withContext do
    let (h, L) ← findTypestate "check_reach"
    let (keep, idx, dLen, oLen, P, pidx, _, vs) ← readTypestate L #[]
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
    let (keep', _, _, _, P', pidx', _, vs') ← readTypestate L' vs (entry := false)
    if vs'.size != vs.size then throwError "check_reach: the post names a variable the entry does not"
    if code.hasFVar then throwError "check_reach: the body names a variable"
    let S ← mkListLit (mkConst ``Fact) keep
    let S' ← mkListLit (mkConst ``Fact) keep'
    let (P, cert) ← runChecker "check_reach" false S dLen oLen P code
    -- the post's facts but its lengths, in order: closed ones from `S'`, the
    -- others from `P'`'s cells then buffers
    let mut plan : Array Nat := #[]
    let mut k := 0
    let mut j := 0
    for f in ← listElems L' do
      let f ← instantiateMVars f
      if f.isAppOfArity ``Fact.roomArg 2 then
        j := j + 1
        continue
      if !f.hasFVar then
        plan := plan.push k
        k := k + 1
      else
        match pidx'.findIdx? (· == j) with
        | some q => plan := plan.push (keep'.length + q)
        | none => throwError "check_reach: unplaced post fact {f}"
      j := j + 1
    let sCert ← Term.exprToSyntax (toExpr cert)
    let sA ← Term.exprToSyntax (mkApp4 (mkConst ``entryOf) S (toExpr dLen) (toExpr oLen) (toExpr P))
    let sS' ← Term.exprToSyntax S'
    let sP' ← Term.exprToSyntax (toExpr P')
    let sc ← Term.exprToSyntax code
    let sh ← Term.exprToSyntax h
    let splan ← Term.exprToSyntax (toExpr plan.toList)
    let A := mkApp4 (mkConst ``entryOf) S (toExpr dLen) (toExpr oLen) (toExpr P)
    unless ← unsafe evalExpr Bool (mkConst ``Bool) (mkApp5 (mkConst ``acceptsTo) (toExpr cert) A code S' (toExpr P')) do
      throwError "check_reach: the checker accepts the body but does not reach the post typestate"
    evalTactic (← `(tactic|
      have hacc : acceptsTo $sCert $sA $sc $sS' $sP' = true := by decide +kernel))
    let hA ← entryProof h S idx dLen oLen P pidx vs
    let sv ← Term.exprToSyntax (← mkListLit (mkConst ``UInt64) vs.toList)
    evalTactic (← `(tactic| refine check_triple (vals := $sv) hacc rfl $hA ?hpost))
    evalTactic (← `(tactic| case hpost =>
      exact fun w' hk hS' hT => post_of $sh hk hS' hT $splan (by rfl)
        (by first | (simp [isRoomArg]; done) | exact rooms_same (by rfl))))

open Lean Meta Elab Tactic in
/-- Prove `Tr.Covers t w obs` from the hypotheses that the world `w` satisfies
    a typestate and that a body run from `runArgs` and `w` ended observing
    `obs`, by the checker: `t` is the pattern it gives the body's calls. A
    refusal names the pattern the checker finds. -/
meta def checkCallsCore (compiled : Bool) : TacticM Unit := do
  let dec ← if compiled then `(Lean.Parser.Tactic.tacticSeq| native_decide)
    else `(Lean.Parser.Tactic.tacticSeq| decide +kernel)
  let g ← getMainGoal
  g.withContext do
    let (h, L) ← findTypestate "check_calls"
    let tgt ← whnfR (← instantiateMVars (← g.getType))
    unless tgt.isAppOfArity ``Tr.Covers 3 do throwError "check_calls: the goal is not `Tr.Covers t w obs`"
    let goalT := tgt.getArg! 0
    let mut found : Option (Expr × Expr) := none
    for d in ← getLCtx do
      if d.isImplementationDetail then continue
      let t ← instantiateMVars d.type
      if let some (_, lhs, _) := t.eq? then
        if lhs.isAppOfArity ``Sem.run 4 then found := some (d.toExpr, lhs)
    let some (hr, runE) := found | throwError "check_calls: no hypothesis that a run ended"
    let code := runE.getArg! 3
    if code.hasFVar then throwError "check_calls: the body names a variable"
    unless (runE.getArg! 1).isAppOf ``runArgs do throwError "check_calls: the arguments are not `runArgs`"
    let (keep, idx, dLen, oLen, P, pidx, _, vs) ← readTypestate L #[]
    let S ← mkListLit (mkConst ``Fact) keep
    let (P, cert) ← runChecker "check_calls" false S dLen oLen P code
    let cert : Cert := { cert with tr := true }
    let A := mkApp4 (mkConst ``entryOf) S (toExpr dLen) (toExpr oLen) (toExpr P)
    let some t ← unsafe evalExpr (Option Tr) (mkApp (mkConst ``Option [0]) (mkConst ``Tr))
        (mkApp3 (mkConst ``traceOf) (toExpr cert) A code)
      | throwError "check_calls: the checker refuses this body"
    if goalT.hasFVar || goalT.hasMVar then throwError "check_calls: the calls the checker finds are\n{repr t}"
    let want ← unsafe evalExpr Tr (mkConst ``Tr) goalT
    unless want == t do throwError "check_calls: the calls the checker finds are\n{repr t}"
    let sCert ← Term.exprToSyntax (toExpr cert)
    let sA ← Term.exprToSyntax A
    let sc ← Term.exprToSyntax code
    let sT ← Term.exprToSyntax goalT
    let shr ← Term.exprToSyntax hr
    evalTactic (← `(tactic| have hacc : traceOf $sCert $sA $sc = some $sT := by $dec))
    let hA ← entryProof h S idx dLen oLen P pidx vs
    evalTactic (← `(tactic| refine check_calls_of hacc $hA $shr))

open Lean Meta Elab Tactic in
elab "check_calls" : tactic => checkCallsCore false

open Lean Meta Elab Tactic in
/-- `check_calls`, the trace decided by compiled evaluation rather than by
    the kernel: for a body too long for the kernel to evaluate. The proof then
    also rests on the compiler (`Lean.ofReduceBool`). -/
elab "check_calls" &" native" : tactic => checkCallsCore true

open Lean Meta Elab Tactic in
/-- Prove `run cfg args w c ≠ .misuse m ∧ run cfg args w c ≠ .fault m` for a
    body entered with arguments `avs` describes, from the hypothesis that the
    world satisfies a closed typestate, by the checker. What is left is that
    `avs` describe the arguments (`DescAll`). -/
elab "check_lib " avs:term : tactic => do
  let g ← getMainGoal
  g.withContext do
    let (h, L) ← findTypestate "check_lib"
    let L ← instantiateMVars L
    if L.hasFVar then throwError "check_lib: the typestate names a variable"
    let tgt ← instantiateMVars (← g.getType)
    let some runE := tgt.find? (fun e => e.isAppOfArity ``Sem.run 4)
      | throwError "check_lib: the goal runs no body"
    let code := runE.getArg! 3
    if code.hasFVar then throwError "check_lib: the body names a variable"
    let avsE ← Term.elabTermEnsuringType avs (some (mkApp (mkConst ``List [0]) (mkConst ``AV)))
    let avsE ← instantiateMVars avsE
    let A := mkApp2 (mkConst ``entryWith) L avsE
    let cert ← match ← unsafe evalExpr (Except String Cert)
        (mkApp2 (mkConst ``Except [0, 0]) (mkConst ``String) (mkConst ``Cert))
        (mkApp4 (mkConst ``certOrRefusal) (toExpr false) A code (toExpr false)) with
      | .ok c => pure c
      | .error e => throwError "check_lib: the checker refuses this body:\n{e}"
    let sCert ← Term.exprToSyntax (toExpr cert)
    let sA ← Term.exprToSyntax A
    let sh ← Term.exprToSyntax h
    let sc ← Term.exprToSyntax code
    evalTactic (← `(tactic| have hacc : accepts $sCert $sA $sc = true := by decide +kernel))
    evalTactic (← `(tactic| refine check_run (vals := []) hacc rfl (entryWith_holds $sh ?_) _))

/-- `ship_entry X : code under P` states and proves the two theorems of a
    shipped body: `X_fine`, that its emission is well formed, and `X_no_misuse`,
    that from a world satisfying the typestate `P` (which may read `dataLen`
    and `outLen`) the body neither misuses a call nor faults, by `check_body`. -/
syntax (docComment)? "ship_entry " ident " : " term " under " term : command

open Lean in
set_option hygiene false in
macro_rules
  | `($[$doc:docComment]? ship_entry $n:ident : $code:term under $P:term) => do
    let fine := mkIdent (n.getId.appendAfter "_fine")
    let safe := mkIdent (n.getId.appendAfter "_no_misuse")
    `(theorem $fine : Fine (emitGo (fnBody ptrParams ($code : Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
        ⟨by decide +kernel, by decide +kernel⟩
      set_option maxRecDepth 200000 in
      set_option maxHeartbeats 4000000 in
      $[$doc:docComment]?
      theorem $safe (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : $P w) (m : String) :
          Sem.run cfg (runArgs dataLen outLen) w (emit ($code : Prog Slot Lvl Unit)) ≠ .misuse m ∧
          Sem.run cfg (runArgs dataLen outLen) w (emit ($code : Prog Slot Lvl Unit)) ≠ .fault m := by
        check_body)

end AlgorithmLib.Check
