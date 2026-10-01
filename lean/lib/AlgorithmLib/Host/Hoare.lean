module
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Hoare` — what a region may do, and that it never misuses

The core every way of checking a host program is stated against. A triple
`Triple cfg P c Q` says: from any environment and world satisfying `P`, at any
fuel, a run of `c` never ends in `misuse`, and when it ends normally, by a
`br` or by a `cont`, `Q` says what it ends with. Getting stuck is allowed: the
triple is about misuse, not about termination, which is what lets a loop be
checked once with an invariant however many times it runs.

The rules here are the whole of the structural reasoning --- statements,
straight runs, sequencing, branches, both kinds of loop, consequence --- and
each is proven against `Sem`. What a foreign call needs is not here: it is the
call's contract, and `call_rule` takes it as a hypothesis, so a contract's
specification (`Host.DevSpec` for the device) plugs in without this module
knowing it.
-/

namespace AlgorithmLib.HProg.Hoare

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

/-- What a region may end with. Stuck is always allowed; misuse never; a fault
    of the program's own where `faultOk` says so. -/
structure Post where
  ok : Env → World → Prop
  brk : Nat → Env → List V → World → Prop := fun _ _ _ _ => False
  cont : Nat → List V → World → Prop := fun _ _ _ => False
  faultOk : Bool := true

def CodeRes.Sat (Q : Post) : CodeRes → Prop
  | .ok Γ w => Q.ok Γ w
  | .brk d Γ vs w => Q.brk d Γ vs w
  | .cont d vs w => Q.cont d vs w
  | .stuck _ => True
  | .misuse _ => False
  | .fault _ => Q.faultOk = true

/-- An outcome `Q` allows; a fault where `fo` says so. -/
def OutSat (Q : Env → World → Prop) (fo : Bool := true) : Outcome Env → Prop
  | .ok Γ w => Q Γ w
  | .stuck _ => True
  | .misuse _ => False
  | .fault _ => fo = true

/-- One statement. -/
def Stmt1 (cfg : Cfg) (P : Env → World → Prop) (s : Stmt) (Q : Env → World → Prop)
    (fo : Bool := true) : Prop :=
  ∀ Γ w, P Γ w → OutSat Q fo (runStmt cfg Γ w s)

/-- A straight run of statements. -/
def Stmts (cfg : Cfg) (P : Env → World → Prop) (ss : List Stmt) (Q : Env → World → Prop)
    (fo : Bool := true) : Prop :=
  ∀ Γ w, P Γ w → OutSat Q fo (runStmts cfg Γ w ss)

def PieceT (cfg : Cfg) (P : Env → World → Prop) (p : Piece) (Q : Post) : Prop :=
  ∀ fuel Γ w, P Γ w → CodeRes.Sat Q (runPiece fuel cfg Γ w p)

/-- **The triple.** -/
def Triple (cfg : Cfg) (P : Env → World → Prop) (c : Code) (Q : Post) : Prop :=
  ∀ fuel Γ w, P Γ w → CodeRes.Sat Q (runCode fuel cfg Γ w c)

/-- A precondition that names one state. -/
def At (Γ : Env) (w : World) : Env → World → Prop := fun Γ' w' => Γ' = Γ ∧ w' = w

-- ---------------------------------------------------------------------------
-- Statements
-- ---------------------------------------------------------------------------

/-- A statement that calls nothing never misuses: it finishes, is stuck, or
    faults where `fo` allows it. -/
theorem nonCall_rule {cfg : Cfg} {P Q : Env → World → Prop} {s : Stmt} {fo : Bool}
    (hs : ∀ c as, s ≠ .call c as ∧ s ≠ .callVoid c as)
    (h : ∀ Γ w Γ' w', P Γ w → runStmt cfg Γ w s = .ok Γ' w' → Q Γ' w')
    (hnf : ∀ Γ w m, P Γ w → runStmt cfg Γ w s = .fault m → fo = true := by intros; rfl) :
    Stmt1 cfg P s Q fo := by
  intro Γ w hP
  have hq := h Γ w
  cases hr : runStmt cfg Γ w s with
  | ok Γ' w' => exact hq Γ' w' hP hr
  | stuck _ => trivial
  | fault m => exact hnf Γ w m hP hr
  | misuse m =>
      exfalso
      cases s with
      | call c as => exact (hs c as).1 rfl
      | callVoid c as => exact (hs c as).2 rfl
      | op o =>
          simp only [runStmt] at hr
          split at hr <;> cases hr
      | store ty v a =>
          rw [runStmt_store] at hr
          revert hr
          dsimp only
          (repeat' split) <;> intro h <;> cases h
      | storeUnaligned v a =>
          rw [runStmt_storeUnaligned] at hr
          revert hr
          dsimp only
          (repeat' split) <;> intro h <;> cases h
      | istore8 v a =>
          rw [runStmt_istore8] at hr
          revert hr
          dsimp only
          (repeat' split) <;> intro h <;> cases h

theorem op_rule {cfg : Cfg} {P Q : Env → World → Prop} {o : Op} {fo : Bool}
    (h : ∀ Γ w v, P Γ w → evalOp w.mem Γ o = some v → Q (Γ.push v) w)
    (hnf : ∀ Γ w, P Γ w → evalOp w.mem Γ o = none → fo = true := by intros; rfl) :
    Stmt1 cfg P (.op o) Q fo :=
  nonCall_rule (fun _ _ => ⟨nofun, nofun⟩) (fun Γ w Γ' w' hP hr => by
    simp only [runStmt] at hr
    split at hr
    · rename_i v hv
      cases hr
      exact h Γ w v hP hv
    · cases hr) (fun Γ w m hP hr => by
    simp only [runStmt] at hr
    split at hr
    · cases hr
    · rename_i hv; exact hnf Γ w hP hv)

/-- A call that binds its answer: its contract, from every state `P` allows,
    answers, and `Q` holds of what it leaves. -/
theorem call_rule {cfg : Cfg} {P Q : Env → World → Prop} {c : Callee} {args : List R} {fo : Bool}
    (h : ∀ Γ w vs, P Γ w → args.mapM (fun r => Γ[r]?) = some vs →
      ∃ r w', callOf cfg.locals c vs (obsCall w c vs) = some (r, w') ∧
        ∀ v, r = some v → Q (Γ.push v) w') :
    Stmt1 cfg P (.call c args) Q fo := by
  intro Γ w hP
  rw [runStmt_call]
  split
  · trivial
  · rename_i vs hvs
    obtain ⟨r, w', hc, hq⟩ := h Γ w vs hP hvs
    rw [hc]
    cases r with
    | none => trivial
    | some v => exact hq v rfl

theorem callVoid_rule {cfg : Cfg} {P Q : Env → World → Prop} {c : Callee} {args : List R} {fo : Bool}
    (h : ∀ Γ w vs, P Γ w → args.mapM (fun r => Γ[r]?) = some vs →
      ∃ r w', callOf cfg.locals c vs (obsCall w c vs) = some (r, w') ∧ Q Γ w') :
    Stmt1 cfg P (.callVoid c args) Q fo := by
  intro Γ w hP
  rw [runStmt_callVoid]
  split
  · trivial
  · rename_i vs hvs
    obtain ⟨r, w', hc, hq⟩ := h Γ w vs hP hvs
    rw [hc]
    exact hq

/-- What a call that does not answer may end in: not a misuse, and a fault
    only where `fo` allows one. A stuck call is allowed: a callee that runs
    out of steps proves nothing about its caller. -/
def FailOk (o : Outcome Env) (fo : Bool) : Prop :=
  (∀ m, o ≠ .misuse m) ∧ (∀ m, o = .fault m → fo = true)

theorem outSat_failOf {lc : Locals} {c : Callee} {vs : List V} {w : World} {Q : Env → World → Prop}
    {fo : Bool} (h : FailOk (failOf lc c vs w) fo) : OutSat Q fo (failOf lc c vs w) := by
  cases ho : (failOf lc c vs w : Outcome Env) with
  | ok _ _ => exact absurd ho failOf_ne_ok
  | stuck _ => trivial
  | misuse m => exact absurd ho (h.1 m)
  | fault m => exact h.2 m ho

/-- A call that binds its answer and may not answer: where it does not, it
    ends as `FailOk` allows, and where it does, `Q` holds of what it leaves. -/
theorem callL_rule {cfg : Cfg} {P Q : Env → World → Prop} {c : Callee} {args : List R} {fo : Bool}
    (h : ∀ Γ w vs, P Γ w → args.mapM (fun r => Γ[r]?) = some vs →
      (callOf cfg.locals c vs (obsCall w c vs) = none →
        FailOk (failOf cfg.locals c vs (obsCall w c vs)) fo) ∧
      ∀ r w', callOf cfg.locals c vs (obsCall w c vs) = some (r, w') → ∀ v, r = some v → Q (Γ.push v) w') :
    Stmt1 cfg P (.call c args) Q fo := by
  intro Γ w hP
  rw [runStmt_call]
  split
  · trivial
  · rename_i vs hvs
    obtain ⟨hn, hs⟩ := h Γ w vs hP hvs
    cases hc : callOf cfg.locals c vs (obsCall w c vs) with
    | none => exact outSat_failOf (hn hc)
    | some p =>
        obtain ⟨r, w'⟩ := p
        cases r with
        | none => trivial
        | some v => exact hs _ _ hc v rfl

theorem callVoidL_rule {cfg : Cfg} {P Q : Env → World → Prop} {c : Callee} {args : List R} {fo : Bool}
    (h : ∀ Γ w vs, P Γ w → args.mapM (fun r => Γ[r]?) = some vs →
      (callOf cfg.locals c vs (obsCall w c vs) = none →
        FailOk (failOf cfg.locals c vs (obsCall w c vs)) fo) ∧
      ∀ r w', callOf cfg.locals c vs (obsCall w c vs) = some (r, w') → Q Γ w') :
    Stmt1 cfg P (.callVoid c args) Q fo := by
  intro Γ w hP
  rw [runStmt_callVoid]
  split
  · trivial
  · rename_i vs hvs
    obtain ⟨hn, hs⟩ := h Γ w vs hP hvs
    cases hc : callOf cfg.locals c vs (obsCall w c vs) with
    | none => exact outSat_failOf (hn hc)
    | some p => obtain ⟨r, w'⟩ := p; exact hs _ _ hc

/-- A call shown to answer meets the condition `callL_rule` asks. -/
theorem callL_of_some {cfg : Cfg} {c : Callee} {vs : List V} {w : World} {fo : Bool}
    {K : Option V → World → Prop} {x0 : Option V} {w0 : World}
    (hc : callOf cfg.locals c vs w = some (x0, w0)) (hk : K x0 w0) :
    (callOf cfg.locals c vs w = none → FailOk (failOf cfg.locals c vs w) fo) ∧
      ∀ x w', callOf cfg.locals c vs w = some (x, w') → K x w' :=
  ⟨fun h => (by rw [hc] at h; cases h), fun x w' h => (by rw [hc] at h; cases h; exact hk)⟩

theorem stmts_nil {cfg : Cfg} {P Q : Env → World → Prop} (h : ∀ Γ w, P Γ w → Q Γ w) :
    Stmts cfg P [] Q :=
  fun Γ w hP => h Γ w hP

theorem stmts_cons {cfg : Cfg} {P R Q : Env → World → Prop} {s : Stmt} {ss : List Stmt} {fo : Bool}
    (h1 : Stmt1 cfg P s R fo) (h2 : Stmts cfg R ss Q fo) : Stmts cfg P (s :: ss) Q fo := by
  intro Γ w hP
  have := h1 Γ w hP
  simp only [runStmts]
  cases hr : runStmt cfg Γ w s with
  | ok Γ' w' => rw [hr] at this; exact h2 Γ' w' this
  | stuck _ => trivial
  | misuse _ => rw [hr] at this; exact this
  | fault _ => rw [hr] at this; exact this

-- ---------------------------------------------------------------------------
-- Pieces and code
-- ---------------------------------------------------------------------------

theorem straight_rule {cfg : Cfg} {P : Env → World → Prop} {ss : List Stmt} {Q : Post}
    (h : Stmts cfg P ss Q.ok Q.faultOk) : PieceT cfg P (.straight ss) Q := by
  intro fuel Γ w hP
  cases fuel with
  | zero => trivial
  | succ fuel =>
      have := h Γ w hP
      simp only [runPiece]
      cases hr : runStmts cfg Γ w ss with
      | ok Γ' w' => rw [hr] at this; exact this
      | stuck _ => trivial
      | misuse _ => rw [hr] at this; exact this
      | fault _ => rw [hr] at this; exact this

theorem br_rule {cfg : Cfg} {P : Env → World → Prop} {d : Nat} {args : List R} {Q : Post}
    (h : ∀ Γ w vs, P Γ w → args.mapM (fun r => Γ[r]?) = some vs → Q.brk d Γ vs w) :
    PieceT cfg P (.br d args) Q := by
  intro fuel Γ w hP
  cases fuel with
  | zero => trivial
  | succ fuel =>
      simp only [runPiece]
      split
      · trivial
      · rename_i vs hvs
        exact h Γ w vs hP hvs

theorem cont_rule {cfg : Cfg} {P : Env → World → Prop} {d : Nat} {args : List R} {Q : Post}
    (h : ∀ Γ w vs, P Γ w → args.mapM (fun r => Γ[r]?) = some vs → Q.cont d vs w) :
    PieceT cfg P (.cont d args) Q := by
  intro fuel Γ w hP
  cases fuel with
  | zero => trivial
  | succ fuel =>
      simp only [runPiece]
      split
      · trivial
      · rename_i vs hvs
        exact h Γ w vs hP hvs

theorem nil_rule {cfg : Cfg} {P : Env → World → Prop} {Q : Post}
    (h : ∀ Γ w, P Γ w → Q.ok Γ w) : Triple cfg P [] Q := by
  intro fuel Γ w hP
  cases fuel with
  | zero => trivial
  | succ fuel => exact h Γ w hP

/-- Sequencing: the first piece ends normally in `R`, or leaves as the whole
    region would. -/
theorem cons_rule {cfg : Cfg} {P R : Env → World → Prop} {p : Piece} {ps : Code} {Q : Post}
    (h1 : PieceT cfg P p { Q with ok := R }) (h2 : Triple cfg R ps Q) :
    Triple cfg P (p :: ps) Q := by
  intro fuel Γ w hP
  cases fuel with
  | zero => trivial
  | succ fuel =>
      have := h1 fuel Γ w hP
      simp only [runCode]
      cases hr : runPiece fuel cfg Γ w p with
      | ok Γ' w' => rw [hr] at this; exact h2 fuel Γ' w' this
      | brk d Γb vs w' => rw [hr] at this; exact this
      | cont d vs w' => rw [hr] at this; exact this
      | stuck _ => trivial
      | misuse _ => rw [hr] at this; exact this
      | fault _ => rw [hr] at this; exact this

-- Sequencing whole codes, which is what an emitter that appends needs.

/-- `runCode`, handing what is left at the end to `K` with the fuel it has then. -/
def runCodeK (cfg : Cfg) : Nat → Env → World → Code → (Nat → Env → World → CodeRes) → CodeRes
  | 0, _, _, _, _ => .stuck "step budget exhausted"
  | f + 1, Γ, w, [], K => K (f + 1) Γ w
  | f + 1, Γ, w, p :: ps, K =>
      match runPiece f cfg Γ w p with
      | .ok Γ' w' => runCodeK cfg f Γ' w' ps K
      | .stuck s => .stuck s
      | .misuse s => .misuse s
      | .fault s => .fault s
      | .brk d Γb vs w' => .brk d Γb vs w'
      | .cont d vs w' => .cont d vs w'

theorem runCode_append (cfg : Cfg) (F G : Code) : ∀ f Γ w,
    runCode f cfg Γ w (F ++ G) = runCodeK cfg f Γ w F (fun k Γ w => runCode k cfg Γ w G) := by
  induction F with
  | nil => intro f Γ w; cases f <;> simp [runCode, runCodeK]
  | cons p ps ih =>
      intro f Γ w
      cases f with
      | zero => simp [runCode, runCodeK]
      | succ f =>
          simp only [List.cons_append, runCode, runCodeK]
          split <;> simp_all

/-- A prefix either ends the run itself, the same way whatever follows, or hands
    over, with at least one step left, to what follows. -/
theorem runCodeK_cases (cfg : Cfg) (F : Code) : ∀ f Γ w,
    (∃ r, (∀ Γ' w', r ≠ .ok Γ' w') ∧ ∀ K, runCodeK cfg f Γ w F K = r) ∨
    (∃ k Γ' w', 1 ≤ k ∧ ∀ K, runCodeK cfg f Γ w F K = K k Γ' w') := by
  induction F with
  | nil =>
      intro f Γ w
      cases f with
      | zero => refine .inl ⟨_, ?_, fun _ => rfl⟩; intro _ _ h; cases h
      | succ f => exact .inr ⟨f + 1, Γ, w, by omega, fun _ => rfl⟩
  | cons p ps ih =>
      intro f Γ w
      cases f with
      | zero => refine .inl ⟨_, ?_, fun _ => rfl⟩; intro _ _ h; cases h
      | succ f =>
          simp only [runCodeK]
          cases runPiece f cfg Γ w p with
          | ok Γ' w' => exact ih f Γ' w'
          | stuck s => refine .inl ⟨_, ?_, fun _ => rfl⟩; intro _ _ h; cases h
          | misuse s => refine .inl ⟨_, ?_, fun _ => rfl⟩; intro _ _ h; cases h
          | fault s => refine .inl ⟨_, ?_, fun _ => rfl⟩; intro _ _ h; cases h
          | brk d Γb vs w' => refine .inl ⟨_, ?_, fun _ => rfl⟩; intro _ _ h; cases h
          | cont d vs w' => refine .inl ⟨_, ?_, fun _ => rfl⟩; intro _ _ h; cases h

theorem sat_ok_irrel {Q : Post} {R : Env → World → Prop} {r : CodeRes}
    (hr : ∀ Γ w, r ≠ .ok Γ w) (h : CodeRes.Sat { Q with ok := R } r) : CodeRes.Sat Q r := by
  cases r with
  | ok Γ w => exact absurd rfl (hr Γ w)
  | _ => exact h

/-- **Sequencing two codes.** -/
theorem append_rule {cfg : Cfg} {P R : Env → World → Prop} {F G : Code} {Q : Post}
    (h1 : Triple cfg P F { Q with ok := R }) (h2 : Triple cfg R G Q) :
    Triple cfg P (F ++ G) Q := by
  intro f Γ w hP
  have h := h1 f Γ w hP
  rw [← List.append_nil F, runCode_append] at h
  rw [runCode_append]
  rcases runCodeK_cases cfg F f Γ w with ⟨r, hr, hK⟩ | ⟨k, Γ', w', hk, hK⟩
  · rw [hK] at h ⊢; exact sat_ok_irrel hr h
  · rw [hK] at h ⊢
    obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
    exact h2 _ _ _ h

theorem runStmts_append (cfg : Cfg) (x y : List Stmt) : ∀ Γ w,
    runStmts cfg Γ w (x ++ y) =
      match runStmts cfg Γ w x with
      | .ok Γ' w' => runStmts cfg Γ' w' y
      | .stuck m => .stuck m
      | .misuse m => .misuse m
      | .fault m => .fault m := by
  induction x with
  | nil => intro Γ w; rfl
  | cons s ss ih =>
      intro Γ w
      simp only [List.cons_append, runStmts]
      split <;> simp_all

/-- **A straight run that goes on**: statements added to the last straight run
    of a code run after it. -/
theorem merge_rule {cfg : Cfg} {P R : Env → World → Prop} {F : Code} {x y : List Stmt} {Q : Post}
    (h1 : Triple cfg P (F ++ [.straight x]) { Q with ok := R }) (h2 : Stmts cfg R y Q.ok Q.faultOk) :
    Triple cfg P (F ++ [.straight (x ++ y)]) Q := by
  intro f Γ w hP
  have h := h1 f Γ w hP
  rw [runCode_append] at h ⊢
  rcases runCodeK_cases cfg F f Γ w with ⟨r, hr, hK⟩ | ⟨k, Γ', w', hk, hK⟩
  · rw [hK] at h ⊢; exact sat_ok_irrel hr h
  · rw [hK] at h ⊢
    obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
    cases j with
    | zero => simp [runCode, runPiece, CodeRes.Sat]
    | succ i =>
        simp only [runCode, runPiece] at h ⊢
        rw [runStmts_append]
        cases hx : runStmts cfg Γ' w' x with
        | ok Γ1 w1 =>
            rw [hx] at h
            simp only [CodeRes.Sat] at h ⊢
            have := h2 Γ1 w1 (by cases i <;> exact h)
            cases hy : runStmts cfg Γ1 w1 y <;> rw [hy] at this <;>
              cases i <;> simp_all [OutSat]
        | stuck m => cases i <;> simp [CodeRes.Sat]
        | misuse m => rw [hx] at h; cases i <;> simp_all [CodeRes.Sat]
        | fault m => rw [hx] at h; cases i <;> simp_all [CodeRes.Sat]

theorem conseq {cfg : Cfg} {P P' : Env → World → Prop} {c : Code} {Q Q' : Post}
    (h : Triple cfg P c Q) (hP : ∀ Γ w, P' Γ w → P Γ w)
    (hok : ∀ Γ w, Q.ok Γ w → Q'.ok Γ w)
    (hbrk : ∀ d Γ vs w, Q.brk d Γ vs w → Q'.brk d Γ vs w)
    (hcont : ∀ d vs w, Q.cont d vs w → Q'.cont d vs w)
    (hfault : Q.faultOk = true → Q'.faultOk = true := by intro h; first | exact h | rfl | simp_all) :
    Triple cfg P' c Q' := by
  intro fuel Γ w hP'
  have := h fuel Γ w (hP Γ w hP')
  cases hr : runCode fuel cfg Γ w c <;> rw [hr] at this
  · exact hok _ _ this
  · exact hbrk _ _ _ _ this
  · exact hcont _ _ _ this
  · trivial
  · exact this
  · exact hfault this

/-- What an arm of a branch must end with: its exports bound at the join, as the
    branch ends; a `br` or `cont` as the branch would. -/
def joinPost (Q : Post) (exports : List R) (joinAt : Nat) : Post where
  ok Γ' w' := ∀ vs, exports.mapM (fun r => Γ'[r]?) = some vs → Q.ok (bindAt Γ' joinAt vs) w'
  brk := Q.brk
  cont := Q.cont
  faultOk := Q.faultOk

/-- **A branch**: whichever arm the flag picks, from the state the branch starts
    in, ends as `joinPost` says. -/
theorem ite_rule {cfg : Cfg} {P : Env → World → Prop} {m : IteMeta} {thn els : Code}
    {thnR elsR : List R} {Q : Post}
    (h : ∀ Γ w t f, P Γ w → Γ[m.flag]? = some (.sc t f) →
      (f != 0) = true →
        Triple cfg (At Γ w) thn (joinPost Q thnR (slotsOf (slotsOf Γ.size thn) els)))
    (h' : ∀ Γ w t f, P Γ w → Γ[m.flag]? = some (.sc t f) →
      (f != 0) = false →
        Triple cfg (At (bindAt Γ (slotsOf Γ.size thn) []) w) els
          (joinPost Q elsR (slotsOf (slotsOf Γ.size thn) els))) :
    PieceT cfg P (.ite m thn els thnR elsR) Q := by
  intro fuel Γ w hP
  cases fuel with
  | zero => trivial
  | succ fuel =>
      simp only [runPiece]
      split
      · rename_i t f hf
        by_cases hz : (f != 0) = true
        · have := h Γ w t f hP hf hz fuel Γ w ⟨rfl, rfl⟩
          simp only [hz, if_true]
          cases hr : runCode fuel cfg Γ w thn <;> rw [hr] at this
          · dsimp only
            split
            · trivial
            · rename_i vs hvs
              exact this vs hvs
          all_goals first | exact this | trivial
        · have hz' : (f != 0) = false := by simpa using hz
          have := h' Γ w t f hP hf hz' fuel _ w ⟨rfl, rfl⟩
          simp only [hz', Bool.false_eq_true, if_false]
          cases hr : runCode fuel cfg (bindAt Γ (slotsOf Γ.size thn) []) w els <;> rw [hr] at this
          · dsimp only
            split
            · trivial
            · rename_i vs hvs
              exact this vs hvs
          all_goals first | exact this | trivial
      · trivial

-- ---------------------------------------------------------------------------
-- Loops, by invariant
-- ---------------------------------------------------------------------------

/-- What the body of a top-tested loop entered from `Γ` must end with: the next
    carries satisfy the invariant, or it leaves as the loop would. -/
def bodyPost (I : List V → World → Prop) (Q : Post) (l : Loop) (ab : Nat) : Post where
  ok Γ2 w2 := ∀ next, l.cont.mapM (fun r => Γ2[r]?) = some next → I next w2
  brk d Γb vs w' := match d with
    | 0 => Q.ok (bindAt Γb ab vs) w'
    | d + 1 => Q.brk d Γb vs w'
  cont d vs w' := match d with
    | 0 => I vs w'
    | d + 1 => Q.cont d vs w'
  faultOk := Q.faultOk

/-- What the condition prefix of a top-tested loop must end with, on carries
    `cs`: where the test says leave, the exits bound as the loop ends; where it
    says go on, the body from there satisfies `bodyPost`. -/
def headPost (cfg : Cfg) (I : List V → World → Prop) (Q : Post) (l : Loop) (pre body : Code)
    (n0 ab : Nat) (cs : List V) : Post where
  ok Γ1 w1 := ∀ t f, Γ1[l.flag]? = some (.sc t f) →
    (((f != 0) == l.exitOnTrue) = true →
      ∀ vs, l.exitR.mapM (fun r => Γ1[r]?) = some vs → Q.ok (bindAt Γ1 ab vs) w1) ∧
    (((f != 0) == l.exitOnTrue) = false →
      Triple cfg (At (bindAt Γ1 (slotsOf (n0 + l.pTys.length) pre) cs) w1) body (bodyPost I Q l ab))
  brk d Γb vs w' := match d with
    | 0 => Q.ok (bindAt Γb ab vs) w'
    | d + 1 => Q.brk d Γb vs w'
  cont d vs w' := match d with
    | 0 => I vs w'
    | d + 1 => Q.cont d vs w'
  faultOk := Q.faultOk

theorem iter_safe {cfg : Cfg} {Γ : Env} {l : Loop} {pre body : Code} {n0 ab : Nat}
    {I : List V → World → Prop} {Q : Post}
    (htrip : ∀ cs w, I cs w → Triple cfg (At (bindAt Γ n0 cs) w) pre (headPost cfg I Q l pre body n0 ab cs)) :
    ∀ fuel cs w, I cs w → CodeRes.Sat Q (iter fuel cfg Γ w l pre body n0 ab cs) := by
  intro fuel
  induction fuel with
  | zero => intro _ _ _; trivial
  | succ fuel ih =>
      intro cs w hI
      have hh := htrip cs w hI fuel _ w ⟨rfl, rfl⟩
      simp only [iter]
      cases h1 : runCode fuel cfg (bindAt Γ n0 cs) w pre <;> rw [h1] at hh
      · rename_i Γ1 w1
        dsimp only
        split
        · rename_i t f hf
          obtain ⟨hx, hc⟩ := hh t f hf
          split
          · rename_i hxv
            split
            · trivial
            · rename_i vs hvs
              exact hx hxv vs hvs
          · rename_i hxv
            have hb := hc (by simpa using hxv) fuel _ w1 ⟨rfl, rfl⟩
            cases h2 : runCode fuel cfg (bindAt Γ1 (slotsOf (n0 + l.pTys.length) pre) cs) w1 body <;>
              rw [h2] at hb
            · dsimp only
              split
              · trivial
              · rename_i next hn
                exact ih next _ (hb next hn)
            · rename_i d Γb vs w2
              cases d with
              | zero => exact hb
              | succ d => exact hb
            · rename_i d vs w2
              cases d with
              | zero => exact ih vs w2 hb
              | succ d => exact hb
            · trivial
            · exact hb
            · exact hb
        · trivial
      · rename_i d Γb vs w'
        cases d with
        | zero => exact hh
        | succ d => exact hh
      · rename_i d vs w'
        cases d with
        | zero => exact ih vs w' hh
        | succ d => exact hh
      · trivial
      · exact hh
      · exact hh

/-- **A top-tested loop, by invariant.** The initial carries satisfy `I` of the
    state the loop starts in, and one trip from any carries satisfying it ends
    as `headPost` says. No measure: the triple allows getting stuck, so a loop
    that runs out of fuel is not a counterexample, and one trip stands for all. -/
theorem loop_rule {cfg : Cfg} {P : Env → World → Prop} {l : Loop} {pre body : Code} {Q : Post}
    (I : Env → List V → World → Prop)
    (hinit : ∀ Γ w cs, P Γ w → l.init.mapM (fun r => Γ[r]?) = some cs → I Γ cs w)
    (htrip : ∀ Γ w0 cs w, P Γ w0 → I Γ cs w →
      Triple cfg (At (bindAt Γ Γ.size cs) w) pre
        (headPost cfg (I Γ) Q l pre body Γ.size
          (slotsOf (slotsOf (Γ.size + l.pTys.length) pre + l.pTys.length) body) cs)) :
    PieceT cfg P (.loop l pre body) Q := by
  intro fuel Γ w hP
  cases fuel with
  | zero => trivial
  | succ fuel =>
      simp only [runPiece]
      split
      · trivial
      · rename_i cs hcs
        exact iter_safe (fun cs' w' hI => htrip Γ w cs' w' hP hI) fuel cs w (hinit Γ w cs hP hcs)

/-- What one trip of a bottom-tested loop must end with. -/
def dPost (I : List V → World → Prop) (Q : Post) (l : DLoop) (ab : Nat) : Post where
  ok Γ2 w2 := ∀ next, l.cont.mapM (fun r => Γ2[r]?) = some next →
    ∀ t f, Γ2[l.flag]? = some (.sc t f) →
      ((f != 0) == l.contOnTrue) = true → I next w2
  brk d Γb vs w' := match d with
    | 0 => Q.ok (bindAt Γb ab vs) w'
    | d + 1 => Q.brk d Γb vs w'
  cont d vs w' := match d with
    | 0 => I vs w'
    | d + 1 => Q.cont d vs w'
  faultOk := Q.faultOk

/-- The exits of a bottom-tested loop, bound as the loop ends. -/
def dLeave (Q : Post) (l : DLoop) (ab : Nat) (Γ' : Env) (vs : List V) (w' : World) : Prop :=
  ∀ outs, l.exitIdx.mapM (fun i => vs[i]?) = some outs → Q.ok (bindAt Γ' ab outs) w'

theorem dtrip_safe {cfg : Cfg} {Γ : Env} {l : DLoop} {body : Code} {n0 ab : Nat}
    {I : List V → World → Prop} {Q : Post}
    (htrip : ∀ cs w, I cs w → Triple cfg (At (bindAt Γ n0 cs) w) body
      { dPost I Q l ab with
        ok := fun Γ2 w2 => (dPost I Q l ab).ok Γ2 w2 ∧
          ∀ next, l.cont.mapM (fun r => Γ2[r]?) = some next →
            ∀ t f, Γ2[l.flag]? = some (.sc t f) →
              ((f != 0) == l.contOnTrue) = false → dLeave Q l ab Γ2 next w2 }) :
    ∀ fuel cs w, I cs w → CodeRes.Sat Q (dtrip fuel cfg Γ w l body n0 ab cs false) := by
  intro fuel
  induction fuel with
  | zero => intro _ _ _; trivial
  | succ fuel ih =>
      intro cs w hI
      have hh := htrip cs w hI fuel _ w ⟨rfl, rfl⟩
      simp only [dtrip, Bool.false_eq_true, if_false]
      cases h1 : runCode fuel cfg (bindAt Γ n0 cs) w body <;> rw [h1] at hh
      · rename_i Γ2 w2
        obtain ⟨hgo, hstop⟩ := hh
        dsimp only
        split
        · trivial
        · rename_i next hn
          split
          · trivial
          · rename_i c hc
            split at hc
            · rename_i t f hf
              cases hc
              split
              · rename_i hx
                exact ih next w2 (hgo next hn t f hf hx)
              · rename_i hx
                have := hstop next hn t f hf (by simpa using hx)
                split
                · trivial
                · rename_i outs ho
                  exact this outs ho
            · cases hc
      · rename_i d Γb vs w'
        cases d with
        | zero => exact hh
        | succ d => exact hh
      · rename_i d vs w'
        cases d with
        | zero => exact ih vs w' hh
        | succ d => exact hh
      · trivial
      · exact hh
      · exact hh

/-- **A bottom-tested loop, by invariant.** Where there is a guard and it says
    skip, the initial carries leave; otherwise they satisfy `I`, and one trip
    from any carries satisfying `I` either satisfies it again or leaves. -/
theorem dloop_rule {cfg : Cfg} {P : Env → World → Prop} {l : DLoop} {body : Code} {Q : Post}
    (I : Env → List V → World → Prop)
    (hinit : ∀ Γ w cs, P Γ w → l.init.mapM (fun r => Γ[r]?) = some cs →
      match l.guard with
      | none => I Γ cs w
      | some g => ∀ t f, Γ[g]? = some (.sc t f) →
          (((f != 0) == l.contOnTrue) = true → I Γ cs w) ∧
          (((f != 0) == l.contOnTrue) = false →
            dLeave Q l (slotsOf (Γ.size + l.pTys.length) body) Γ cs w))
    (htrip : ∀ Γ w0 cs w, P Γ w0 → I Γ cs w →
      Triple cfg (At (bindAt Γ Γ.size cs) w) body
        { dPost (I Γ) Q l (slotsOf (Γ.size + l.pTys.length) body) with
          ok := fun Γ2 w2 => (dPost (I Γ) Q l (slotsOf (Γ.size + l.pTys.length) body)).ok Γ2 w2 ∧
            ∀ next, l.cont.mapM (fun r => Γ2[r]?) = some next →
              ∀ t f, Γ2[l.flag]? = some (.sc t f) →
                ((f != 0) == l.contOnTrue) = false →
                  dLeave Q l (slotsOf (Γ.size + l.pTys.length) body) Γ2 next w2 }) :
    PieceT cfg P (.dloop l body) Q := by
  intro fuel Γ w hP
  cases fuel with
  | zero => trivial
  | succ fuel =>
      simp only [runPiece]
      split
      · trivial
      · rename_i cs hcs
        have hi := hinit Γ w cs hP hcs
        have hsafe := dtrip_safe (fun cs' w' hI => htrip Γ w cs' w' hP hI)
        cases hg : l.guard with
        | none =>
            simp only [hg, Option.isSome_none] at hi ⊢
            exact hsafe fuel cs w hi
        | some g =>
            simp only [hg, Option.isSome_some] at hi ⊢
            cases fuel with
            | zero => trivial
            | succ fuel =>
              rw [dtrip]
              simp only [hg, if_true]
              split
              · trivial
              · rename_i c hc
                split at hc
                · rename_i t f hf
                  cases hc
                  split
                  · rename_i hx
                    exact hsafe fuel cs w ((hi t f hf).1 hx)
                  · rename_i hx
                    have := (hi t f hf).2 (by simpa using hx)
                    split
                    · trivial
                    · rename_i outs ho
                      exact this outs ho
                · cases hc

-- ---------------------------------------------------------------------------
-- A whole run
-- ---------------------------------------------------------------------------

/-- **A body whose triple holds from its entry never misuses.** -/
theorem run_safe {cfg : Cfg} {args : List V} {w : World} {c : Code} {Q : Post}
    (h : Triple cfg (At args.toArray w) c Q) (m : String) : run cfg args w c ≠ .misuse m := by
  have := h cfg.steps args.toArray w ⟨rfl, rfl⟩
  simp only [run]
  cases hr : runCode cfg.steps cfg args.toArray w c <;> rw [hr] at this <;> simp_all [CodeRes.Sat]

/-- A triple whose post allows no fault: the run neither misuses a call nor
    reads or writes memory that is not there. -/
theorem run_sound {cfg : Cfg} {args : List V} {w : World} {c : Code} {Q : Post}
    (h : Triple cfg (At args.toArray w) c Q) (hQ : Q.faultOk = false) (m : String) :
    run cfg args w c ≠ .misuse m ∧ run cfg args w c ≠ .fault m := by
  have := h cfg.steps args.toArray w ⟨rfl, rfl⟩
  simp only [run]
  cases hr : runCode cfg.steps cfg args.toArray w c <;> rw [hr] at this <;> simp_all [CodeRes.Sat]

-- ---------------------------------------------------------------------------
-- Slots
-- ---------------------------------------------------------------------------

theorem bindAt_get? (Γ : Env) (n : Nat) (vs : List V) (i : Nat) :
    (bindAt Γ n vs)[i]? = if i < n then (if i < Γ.size then Γ[i]? else some default) else vs[i - n]? := by
  simp only [bindAt, Array.take_eq_extract, Array.getElem?_append, Array.size_append, Array.size_extract,
    Array.size_replicate, Array.getElem?_extract, Array.getElem?_replicate, List.getElem?_toArray]
  by_cases h1 : i < n
  · rw [if_pos h1]
    by_cases h2 : i < Γ.size
    · rw [if_pos h2, if_pos (by omega), if_pos (by omega), if_pos (by omega)]; simp
    · rw [if_neg h2, if_pos (by omega), if_neg (by omega), if_pos (by omega)]
  · rw [if_neg h1, if_neg (by omega)]
    congr 1; omega

theorem bindAt_lt {Γ : Env} {n i : Nat} {vs : List V} (hi : i < n) (hs : i < Γ.size) :
    (bindAt Γ n vs)[i]? = Γ[i]? := by
  rw [bindAt_get?]; simp [hi, hs]

theorem bindAt_size {Γ : Env} {n : Nat} {vs : List V} (hn : Γ.size ≤ n) :
    (bindAt Γ n vs).size = n + vs.length := by
  simp [bindAt]; omega

theorem bindAt_at {Γ : Env} {n j : Nat} {vs : List V} : (bindAt Γ n vs)[n + j]? = vs[j]? := by
  rw [bindAt_get?, if_neg (by omega)]; congr 1; omega

-- ---------------------------------------------------------------------------
-- Walking a straight run
--
-- A proof about a concrete run tracks the slots it needs (`Has`) and a
-- predicate on the world; an operation adds its slot, a call moves the world.
-- ---------------------------------------------------------------------------

/-- There are `n` slots, and each of `fs` holds its value. -/
def Has (Γ : Env) (n : Nat) (fs : List (Nat × V)) : Prop := Γ.size = n ∧ ∀ p ∈ fs, Γ[p.1]? = some p.2

theorem Has.get {Γ : Env} {n : Nat} {fs : List (Nat × V)} (h : Has Γ n fs) {i : Nat} {v : V}
    (hp : (i, v) ∈ fs) : Γ[i]? = some v := h.2 _ hp

theorem Has.push {Γ : Env} {n : Nat} {fs : List (Nat × V)} (h : Has Γ n fs) (v : V) :
    Has (Γ.push v) (n + 1) ((n, v) :: fs) := by
  refine ⟨by simp [h.1], fun p hp => ?_⟩
  rcases List.mem_cons.mp hp with rfl | hp
  · rw [← h.1]; exact Array.getElem?_push_size
  · have := h.2 p hp
    have hlt : p.1 < Γ.size := by
      rcases e : Γ[p.1]? with _ | x
      · rw [e] at this; cases this
      · exact (Array.getElem?_eq_some_iff.mp e).1
    rw [Array.getElem?_push, if_neg (by omega)]; exact this

theorem Has.push' {Γ : Env} {n : Nat} {fs : List (Nat × V)} (h : Has Γ n fs) (v : V) :
    Has (Γ.push v) (n + 1) fs := by
  refine ⟨by simp [h.1], fun p hp => ?_⟩
  exact ((h.push v).2 p (List.mem_cons_of_mem _ hp))

theorem Has.drop {Γ : Env} {n : Nat} {fs fs' : List (Nat × V)} (h : Has Γ n fs)
    (hs : ∀ p ∈ fs', p ∈ fs) : Has Γ n fs' := ⟨h.1, fun p hp => h.2 p (hs p hp)⟩

theorem op_step {cfg : Cfg} {J : World → Prop} {n : Nat} {fs : List (Nat × V)} {o : Op} (v : V)
    (hv : ∀ Γ m v', Has Γ n fs → evalOp m Γ o = some v' → v' = v) :
    Stmt1 cfg (fun Γ w => Has Γ n fs ∧ J w) (.op o) (fun Γ w => Has Γ (n + 1) ((n, v) :: fs) ∧ J w) :=
  op_rule fun Γ w v' ⟨hh, hj⟩ he => by
    rw [hv Γ w.mem v' hh he]; exact ⟨hh.push v, hj⟩

theorem call_step {cfg : Cfg} {J J' : World → Prop} {n : Nat} {fs : List (Nat × V)} {c : Callee}
    {args : List R}
    (h : ∀ Γ w vs, Has Γ n fs → J w → args.mapM (fun r => Γ[r]?) = some vs →
      ∃ r w', callOf cfg.locals c vs (obsCall w c vs) = some (r, w') ∧ J' w') :
    Stmt1 cfg (fun Γ w => Has Γ n fs ∧ J w) (.call c args) (fun Γ w => Has Γ (n + 1) fs ∧ J' w) :=
  call_rule fun Γ w vs ⟨hh, hj⟩ hvs => by
    obtain ⟨r, w', hc, hj'⟩ := h Γ w vs hh hj hvs
    exact ⟨r, w', hc, fun v _ => ⟨hh.push' v, hj'⟩⟩

theorem callVoid_step {cfg : Cfg} {J J' : World → Prop} {n : Nat} {fs : List (Nat × V)} {c : Callee}
    {args : List R}
    (h : ∀ Γ w vs, Has Γ n fs → J w → args.mapM (fun r => Γ[r]?) = some vs →
      ∃ r w', callOf cfg.locals c vs (obsCall w c vs) = some (r, w') ∧ J' w') :
    Stmt1 cfg (fun Γ w => Has Γ n fs ∧ J w) (.callVoid c args) (fun Γ w => Has Γ n fs ∧ J' w) :=
  callVoid_rule fun Γ w vs ⟨hh, hj⟩ hvs => by
    obtain ⟨r, w', hc, hj'⟩ := h Γ w vs hh hj hvs
    exact ⟨r, w', hc, hh, hj'⟩

end AlgorithmLib.HProg.Hoare
