module
public import AlgorithmLib.Check.Sound
@[expose] public section

/-!
# `Check.Quiet` — what a library function leaves of the world

A library function is code whose calls are to the C library and to the
program's own functions (`quietCode`). **`run_quiet`**: a run of such code
changes memory and the heap and nothing else the typestate reads, and the calls
observed before it are still observed, in order, after it (`Quiet`), given the
same of the program's functions; `callFn_quiet` and `termLocals_quiet` say so
of a call to a function of a program all of whose functions are such code.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

/-- **`w'` is `w` but for memory, the heap and what has been observed**, and
    the calls `w` had observed are `w'`'s first. -/
structure Quiet (w w' : World) : Prop where
  calls : ∃ cs : List (Callee × List V), callsOf w'.obs = cs.reverse ++ callsOf w.obs
  dev : w'.dev = w.dev
  lmdb : w'.lmdb = w.lmdb
  thread : w'.thread = w.thread
  display : w'.display = w.display
  cudaDevice : w'.cudaDevice = w.cudaDevice
  gpuAdapter : w'.gpuAdapter = w.gpuAdapter
  kernel : w'.kernel = w.kernel
  vendor : w'.vendor = w.vendor

theorem Quiet.refl (w : World) : Quiet w w := ⟨⟨[], rfl⟩, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem Quiet.trans {w w' w'' : World} (h1 : Quiet w w') (h2 : Quiet w' w'') : Quiet w w'' := by
  obtain ⟨a, ha⟩ := h1.calls
  obtain ⟨b, hb⟩ := h2.calls
  exact ⟨⟨a ++ b, by rw [hb, ha, List.reverse_append, List.append_assoc]⟩, h2.dev.trans h1.dev,
    h2.lmdb.trans h1.lmdb, h2.thread.trans h1.thread,
    h2.display.trans h1.display, h2.cudaDevice.trans h1.cudaDevice, h2.gpuAdapter.trans h1.gpuAdapter,
    h2.kernel.trans h1.kernel, h2.vendor.trans h1.vendor⟩

/-- Noting a call is quiet. -/
theorem Quiet.obsCall (w : World) (c : Callee) (vs : List V) : Quiet w (obsCall w c vs) :=
  ⟨⟨[(c, vs ++ callWords w c vs)], rfl⟩, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- A store is quiet. -/
theorem Quiet.stored (w : World) (a : UInt64) (n : Nat) (v : UInt64) (m : Mem) :
    Quiet w { obsStore w a n v with mem := m } :=
  ⟨⟨[], rfl⟩, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- The callees a library function names: the program's own functions and the
    C library's. -/
def quietCallee : Callee → Bool
  | .local _ => true
  | .ext (.c _) => true
  | _ => false

def quietStmt : Stmt → Bool
  | .call c _ | .callVoid c _ => quietCallee c
  | _ => true

/-- Whether every call of a region is to a callee `quietCallee` allows. -/
def quietGo : Nat → List Piece → Bool
  | 0, _ => false
  | _ + 1, [] => true
  | f + 1, .straight ss :: ps => ss.all quietStmt && quietGo f ps
  | f + 1, .loop _ pre body :: ps => quietGo f pre && quietGo f body && quietGo f ps
  | f + 1, .ite _ thn els _ _ :: ps => quietGo f thn && quietGo f els && quietGo f ps
  | f + 1, .dloop _ body :: ps => quietGo f body && quietGo f ps
  | f + 1, .br _ _ :: ps | f + 1, .cont _ _ :: ps => quietGo f ps

/-- **Code whose calls are the C library's and the program's own.** -/
def quietCode (c : Code) : Bool := quietGo fuel c

/-- The program's functions, called, are quiet. -/
def QuietLocals (L : Locals) : Prop := ∀ i vs w r w', L i vs w = .ok (r, w') → Quiet w w'

theorem callOf_quiet {L : Locals} (hL : QuietLocals L) {c : Callee} (hq : quietCallee c = true) {vs : List V}
    {w : World} {r : Option V} {w' : World} (h : callOf L c vs w = some (r, w')) : Quiet w w' := by
  cases c with
  | «local» i =>
      simp only [callOf] at h
      cases hl : L i vs w with
      | error e => rw [hl] at h; cases h
      | ok p => rw [hl] at h; cases h; exact hL _ _ _ _ _ hl
  | ext e =>
      cases e with
      | c f =>
          have hs := Contracts.c_step (bits := []) h
          exact ⟨⟨[], by rw [hs.obs]; rfl⟩, hs.dev, hs.lmdb, hs.thread, hs.display, hs.cudaDevice,
            hs.gpuAdapter, hs.kernel, hs.vendor⟩
      | _ => cases hq
  | _ => cases hq

theorem stmt_quiet {cfg : Cfg} (hL : QuietLocals cfg.locals) {s : Stmt} (hq : quietStmt s = true) {Γ Γ' : Env}
    {w w' : World} (hr : runStmt cfg Γ w s = .ok Γ' w') : Quiet w w' := by
  cases s with
  | op o =>
      simp only [runStmt] at hr
      split at hr
      · cases hr; exact Quiet.refl _
      · cases hr
  | store ty v a =>
      rw [runStmt_store] at hr
      split at hr
      · dsimp only at hr
        split at hr
        · split at hr
          · cases hr; exact Quiet.stored _ _ _ _ _
          · cases hr
        · split at hr
          · cases hr; exact Quiet.stored _ _ _ _ _
          · cases hr
      · cases hr
  | storeUnaligned v a =>
      rw [runStmt_storeUnaligned] at hr
      split at hr
      · split at hr
        · cases hr; exact Quiet.stored _ _ _ _ _
        · cases hr
      · dsimp only at hr
        split at hr
        · cases hr; exact Quiet.stored _ _ _ _ _
        · cases hr
      · cases hr
  | istore8 v a =>
      rw [runStmt_istore8] at hr
      split at hr
      · split at hr
        · cases hr; exact Quiet.stored _ _ _ _ _
        · cases hr
      · cases hr
  | call c args =>
      rw [runStmt_call] at hr
      split at hr
      · cases hr
      · split at hr
        · exact absurd hr failOf_ne_ok
        · rename_i res w2 hc
          split at hr
          · cases hr; exact (Quiet.obsCall _ _ _).trans (callOf_quiet hL hq hc)
          · cases hr
  | callVoid c args =>
      rw [runStmt_callVoid] at hr
      split at hr
      · cases hr
      · split at hr
        · exact absurd hr failOf_ne_ok
        · rename_i res w2 hc
          cases hr
          exact (Quiet.obsCall _ _ _).trans (callOf_quiet hL hq hc)

theorem stmts_quiet {cfg : Cfg} (hL : QuietLocals cfg.locals) :
    ∀ {ss : List Stmt}, ss.all quietStmt = true → ∀ {Γ Γ' : Env} {w w' : World},
      runStmts cfg Γ w ss = .ok Γ' w' → Quiet w w'
  | [], _, Γ, Γ', w, w', hr => by
      simp only [runStmts] at hr
      cases hr; exact Quiet.refl _
  | s :: ss, hq, Γ, Γ', w, w', hr => by
      simp only [List.all_cons, Bool.and_eq_true] at hq
      simp only [runStmts] at hr
      split at hr
      · rename_i Γ1 w1 h1
        exact (stmt_quiet hL hq.1 h1).trans (stmts_quiet hL hq.2 hr)
      all_goals cases hr

/-- What a region that ended, or left by a `br` or a `cont`, left quiet. -/
def ResQ (w : World) : CodeRes → Prop
  | .ok _ w' => Quiet w w'
  | .brk _ _ _ w' => Quiet w w'
  | .cont _ _ w' => Quiet w w'
  | _ => True

theorem ResQ.after {w w1 : World} (h : Quiet w w1) {r : CodeRes} (hr : ResQ w1 r) : ResQ w r := by
  cases r <;> first | exact h.trans hr | trivial

/-- **A run of quiet code is quiet**, however it ends: of a piece, of a
    loop's trips, and of a region, at every step budget. -/
theorem run_quiet {cfg : Cfg} (hL : QuietLocals cfg.locals) : ∀ fuel : Nat,
    (∀ n p ps Γ w, quietGo (n + 1) (p :: ps) = true → ResQ w (runPiece fuel cfg Γ w p)) ∧
    (∀ n l pre body Γ w n0 ab cs, quietGo n pre = true → quietGo n body = true →
      ResQ w (iter fuel cfg Γ w l pre body n0 ab cs)) ∧
    (∀ n l body Γ w n0 ab cs first, quietGo n body = true →
      ResQ w (dtrip fuel cfg Γ w l body n0 ab cs first)) ∧
    (∀ n c Γ w, quietGo n c = true → ResQ w (runCode fuel cfg Γ w c)) := by
  intro fuel
  induction fuel with
  | zero =>
      refine ⟨fun n p ps Γ w _ => ?_, fun n l pre body Γ w n0 ab cs _ _ => ?_,
        fun n l body Γ w n0 ab cs first _ => ?_, fun n c Γ w _ => ?_⟩
      · simp only [runPiece]; trivial
      · simp only [iter]; trivial
      · simp only [dtrip]; trivial
      · simp only [runCode]; trivial
  | succ fuel ih =>
      obtain ⟨ihP, ihI, ihD, ihC⟩ := ih
      refine ⟨fun n p ps Γ w hq => ?_, fun n l pre body Γ w n0 ab cs hp hb => ?_,
        fun n l body Γ w n0 ab cs first hb => ?_, fun n c Γ w hq => ?_⟩
      · -- a piece
        cases p with
        | straight ss =>
            simp only [quietGo, Bool.and_eq_true] at hq
            simp only [runPiece]
            split
            · rename_i Γ' w' h; exact stmts_quiet hL hq.1 h
            all_goals trivial
        | br d args =>
            simp only [runPiece]
            split
            · trivial
            · exact Quiet.refl _
        | cont d args =>
            simp only [runPiece]
            split
            · trivial
            · exact Quiet.refl _
        | dloop l body =>
            simp only [quietGo, Bool.and_eq_true] at hq
            simp only [runPiece]
            split
            · trivial
            · exact ihD n l body Γ w _ _ _ _ hq.1
        | loop l pre body =>
            simp only [quietGo, Bool.and_eq_true] at hq
            simp only [runPiece]
            split
            · trivial
            · exact ihI n l pre body Γ w _ _ _ hq.1.1 hq.1.2
        | ite m thn els thnR elsR =>
            simp only [quietGo, Bool.and_eq_true] at hq
            simp only [runPiece]
            split
            · rename_i t f hf
              by_cases hc : (f != 0) = true
              · simp only [hc, if_true]
                have := ihC n thn Γ w hq.1.1
                split <;> first | trivial | skip
                all_goals (rename_i h; rw [h] at this)
                · exact this
                · exact this
                · split
                  · trivial
                  · exact this
              · simp only [hc, Bool.false_eq_true, if_false]
                have := ihC n els (bindAt Γ (slotsOf Γ.size thn) []) w hq.1.2
                split <;> first | trivial | skip
                all_goals (rename_i h; rw [h] at this)
                · exact this
                · exact this
                · split
                  · trivial
                  · exact this
            · trivial
      · -- a top-tested loop's trip
        simp only [iter]
        have h1 := ihC n pre (bindAt Γ n0 cs) w hp
        split <;> first | trivial | skip
        all_goals (rename_i h; rw [h] at h1)
        · exact h1
        · exact h1
        · exact ResQ.after h1 (ihI n l pre body Γ _ n0 ab _ hp hb)
        · exact h1
        · rename_i Γ1 w1
          split
          · split
            · split
              · trivial
              · exact h1
            · have h2 := ihC n body (bindAt Γ1 (slotsOf (n0 + l.pTys.length) pre) cs) w1 hb
              split <;> first | trivial | skip
              all_goals (rename_i h'; rw [h'] at h2)
              · exact h1.trans h2
              · exact h1.trans h2
              · exact ResQ.after (h1.trans h2) (ihI n l pre body Γ _ n0 ab _ hp hb)
              · exact h1.trans h2
              · split
                · trivial
                · exact ResQ.after (h1.trans h2) (ihI n l pre body Γ _ n0 ab _ hp hb)
          · trivial
      · -- a bottom-tested loop's trip
        simp only [dtrip]
        split
        · split
          · exact ihD n l body Γ w n0 ab cs false hb
          · split
            · trivial
            · split
              · exact ihD n l body Γ w n0 ab cs false hb
              · split
                · trivial
                · exact Quiet.refl _
        · have h2 := ihC n body (bindAt Γ n0 cs) w hb
          split <;> first | trivial | skip
          all_goals (rename_i h'; rw [h'] at h2)
          · exact h2
          · exact h2
          · exact ResQ.after h2 (ihD n l body Γ _ n0 ab _ false hb)
          · exact h2
          · split
            · trivial
            · split
              · trivial
              · split
                · exact ResQ.after h2 (ihD n l body Γ _ n0 ab _ false hb)
                · split
                  · trivial
                  · exact h2
      · -- a region
        cases c with
        | nil => simp only [runCode]; exact Quiet.refl _
        | cons p ps =>
            cases n with
            | zero => simp [quietGo] at hq
            | succ n =>
                simp only [runCode]
                have h1 := ihP n p ps Γ w hq
                have hps : quietGo n ps = true := by
                  cases p <;> simp only [quietGo, Bool.and_eq_true] at hq <;> first | exact hq.2 | exact hq
                split <;> first | trivial | skip
                all_goals (rename_i h; rw [h] at h1)
                · exact h1
                · exact h1
                · exact ResQ.after h1 (ihC n ps _ _ hps)

/-- **A call to a function of quiet code is quiet.** -/
theorem callFn_quiet {cfg : Cfg} (hL : QuietLocals cfg.locals) {f : Fn} (hq : quietCode f.code = true)
    {args : List V} {w : World} {r : Option V} {w' : World} (h : Sem.callFn cfg f args w = .ok (r, w')) :
    Quiet w w' := by
  unfold Sem.callFn at h
  split at h
  · cases h
  · have hr := (run_quiet hL cfg.steps).2.2.2 fuel f.code args.toArray w hq
    split at h
    · rename_i Γ w1 he
      rw [he] at hr
      cases h
      exact hr
    all_goals cases h

/-- **The functions of a program all of quiet code are quiet**, at every call
    depth. -/
theorem termLocals_quiet {fns : Nat → Option Fn} {env : FnEnv} {steps : Nat}
    (hq : ∀ i f, fns i = some f → quietCode f.code = true) : ∀ k, QuietLocals (termLocals fns env steps k)
  | 0 => fun i vs w r w' h => by cases h
  | k + 1 => fun i vs w r w' h => by
      simp only [termLocals] at h
      split at h
      · cases h
      · rename_i f hf
        exact callFn_quiet (cfg := { env, steps, locals := termLocals fns env steps k })
          (termLocals_quiet hq k) (hq i f hf) h

end AlgorithmLib.Check
