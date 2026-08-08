import AlgorithmLib.ML.WarpEmit
import AlgorithmLib.ML.Grad
import AlgorithmLib.ML.Tape

/-!
  # Compiling a model spec onto the machine that actually runs

  `compileW` takes any `Expr` to an `EWStmt` — the one representation that both
  elaborates into the proven warp machine *and* lowers to PTX.  So a spec
  compiled here is a spec that runs, on the same machine the theorems are about.
  (`compileE` targets `Stmt`, which has no emitter; `WStmt` has an emitter but
  no compiler from `Expr`.  `EWStmt` is the intersection.)

  ## SIMT semantics of the spec

  All 32 lanes execute the same instructions on different data.  So a compiled
  expression denotes, **per lane**, the value of the spec under that lane's own
  environment — `compileW_sound` says exactly that, for every lane at once.
  Lane-dependence enters only through `ve`, which maps spec variables to
  per-lane machine registers.

  ## Reductions

  `Expr.sum` commits to a *left fold*, so it compiles to an unrolled sequential
  accumulation — **not** a butterfly, which walks a different tree and would be
  a different function (measurably so in `Float32`).  A user who wants the warp
  tree writes it with `warpReduce`; that path already has its own proof and
  emitter in `Warp.lean`.
-/

namespace AlgorithmLib.ML

/-- How many accumulator registers `e` needs.  One per `sum` on the deepest
    reduction path; siblings share, because a sibling's accumulator is dead by
    the time the next sibling runs. -/
def slots : {Γ : Nat} → Expr Γ → Nat
  | _, .var _    => 0
  | _, .lit _    => 0
  | _, .add a b  => slots a + slots b
  | _, .mul a b  => slots a + slots b
  | _, .neg a    => slots a
  | _, .inv a    => slots a
  | _, .exp a    => slots a
  | _, .rsqrt a  => slots a
  | _, .sum n f  => 1 + (List.finRange n).foldl (fun m j => max m (slots (f j))) 0
  -- a's accumulators, then one register holding the binding, then b's
  | _, .letE a b => slots a + 1 + slots b



variable {Γ : Nat}

-- ---------------------------------------------------------------------------
-- Register footprints for the warp machine
-- ---------------------------------------------------------------------------

def WFExp.regsIn (lo hi : Nat) : WFExp → Prop
  | .reg r     => lo ≤ r ∧ r < hi
  | .lit _     => True
  | .add a b   => a.regsIn lo hi ∧ b.regsIn lo hi
  | .mul a b   => a.regsIn lo hi ∧ b.regsIn lo hi
  | .neg a     => a.regsIn lo hi
  | .inv a     => a.regsIn lo hi
  | .exp a     => a.regsIn lo hi
  | .ex2 a     => a.regsIn lo hi
  | .rsqrt a   => a.regsIn lo hi
  | .maxW a b  => a.regsIn lo hi ∧ b.regsIn lo hi
  | .geF a b   => a.regsIn lo hi ∧ b.regsIn lo hi

theorem WFExp.regsIn_weaken (lo hi lo' hi' : Nat) (hl : lo' ≤ lo) (hh : hi ≤ hi') :
    ∀ e : WFExp, e.regsIn lo hi → e.regsIn lo' hi' := by
  intro e
  induction e with
  | reg r => intro h; exact ⟨Nat.le_trans hl h.1, Nat.lt_of_lt_of_le h.2 hh⟩
  | lit _ => intro _; trivial
  | add a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | mul a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | neg a ih   => intro h; exact ih h
  | inv a ih   => intro h; exact ih h
  | exp a ih   => intro h; exact ih h
  | ex2 a ih   => intro h; exact ih h
  | rsqrt a ih => intro h; exact ih h
  | maxW a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | geF a b iha ihb  => intro h; exact ⟨iha h.1, ihb h.2⟩

/-- Two states agreeing on the registers an expression reads give it the same
    value, in every lane. -/
theorem WFExp.eval_frame (lo hi : Nat) :
    ∀ (e : WFExp) (st st' : WSt) (l : Lane), e.regsIn lo hi →
      (∀ r, lo ≤ r → r < hi → st'.regs r = st.regs r) → e.eval st' l = e.eval st l := by
  intro e
  induction e with
  | reg r => intro st st' l h hr; show st'.regs r l = st.regs r l; rw [hr r h.1 h.2]
  | lit v => intro _ _ _ _ _; rfl
  | add a b iha ihb =>
      intro st st' l h hr; simp only [WFExp.eval, iha st st' l h.1 hr, ihb st st' l h.2 hr]
  | mul a b iha ihb =>
      intro st st' l h hr; simp only [WFExp.eval, iha st st' l h.1 hr, ihb st st' l h.2 hr]
  | neg a ih   => intro st st' l h hr; simp only [WFExp.eval, ih st st' l h hr]
  | inv a ih   => intro st st' l h hr; simp only [WFExp.eval, ih st st' l h hr]
  | exp a ih   => intro st st' l h hr; simp only [WFExp.eval, ih st st' l h hr]
  | ex2 a ih   => intro st st' l h hr; simp only [WFExp.eval, ih st st' l h hr]
  | rsqrt a ih => intro st st' l h hr; simp only [WFExp.eval, ih st st' l h hr]
  | maxW a b iha ihb =>
      intro st st' l h hr; simp only [WFExp.eval, iha st st' l h.1 hr, ihb st st' l h.2 hr]
  | geF a b iha ihb =>
      intro st st' l h hr; simp only [WFExp.eval, iha st st' l h.1 hr, ihb st st' l h.2 hr]

-- ---------------------------------------------------------------------------
-- The compiler
-- ---------------------------------------------------------------------------

/-- Unrolled accumulation for `sum`: `acc := 0; acc += e₀; acc += e₁; …`

    Unrolled rather than looped because `EWStmt.forN` has a *syntactic* body and
    the summand varies with the index.  That is the honest trade: emittable, and
    exactly the left fold `Expr.sum` denotes. -/
def sumSeq (acc : Nat) (bodies : List (EWStmt × WFExp)) : EWStmt :=
  bodies.foldl
    (fun s (p : EWStmt × WFExp) =>
      .seq s (.seq p.1 (.setR acc (.add (.reg acc) p.2))))
    (.setR acc (.lit NumOps.zero))

/-- Compile a spec expression onto the warp machine.  Accumulators come from
    `[c, c + slots e)`, exactly as in `compileE`. -/
def compileW : {Γ : Nat} → (Fin Γ → WFExp) → Nat → Expr Γ → EWStmt × WFExp
  | _, ve, _, .var i    => (.skip, ve i)
  | _, _,  _, .lit n    => (.skip, .lit (NumOps.ofNat n))
  | _, ve, c, .add a b  => (.seq (compileW ve c a).1 (compileW ve (c + slots a) b).1,
                  .add (compileW ve c a).2 (compileW ve (c + slots a) b).2)
  | _, ve, c, .mul a b  => (.seq (compileW ve c a).1 (compileW ve (c + slots a) b).1,
                  .mul (compileW ve c a).2 (compileW ve (c + slots a) b).2)
  | _, ve, c, .neg a    => ((compileW ve c a).1, .neg (compileW ve c a).2)
  | _, ve, c, .inv a    => ((compileW ve c a).1, .inv (compileW ve c a).2)
  | _, ve, c, .exp a    => ((compileW ve c a).1, .exp (compileW ve c a).2)
  | _, ve, c, .rsqrt a  => ((compileW ve c a).1, .rsqrt (compileW ve c a).2)
  | _, ve, c, .sum n f  =>
      (sumSeq c ((List.finRange n).map (fun j => compileW ve (c + 1) (f j))), .reg c)
  | _, ve, c, .letE a b =>
      (.seq (.seq (compileW ve c a).1 (.setR (c + slots a) (compileW ve c a).2))
            (compileW (extend ve (.reg (c + slots a))) (c + slots a + 1) b).1,
       (compileW (extend ve (.reg (c + slots a))) (c + slots a + 1) b).2)


-- ---------------------------------------------------------------------------
-- Running a compiled expression
-- ---------------------------------------------------------------------------

/-- Run a compiled fragment.  The fragment contains no loops or memory ops, so
    elaboration is structural and independent of the block index. -/
def runW (s : EWStmt) (st : WSt) : WSt := (EWStmt.expand s).run st

/-! ### Fragments that do not depend on where they run

    `runW` fixes block 0, iteration 0.  A compiled body placed inside a loop
    needs its correctness at iteration `j`, so the independence has to be stated
    rather than assumed.

    `AddrFree` says a fragment carries no address at all, and `elabAt_addrFree`
    turns that into "elaborates the same everywhere".  With it, everything proven
    about `runW` transfers to any `(block, iteration)`. -/

/-- No address anywhere: only register arithmetic, shuffles and barriers. -/
def EWStmt.AddrFree : EWStmt → Prop
  | .skip          => True
  | .setR _ _      => True
  | .shflXor _ _ _ => True
  | .barrier       => True
  | .seq a b       => a.AddrFree ∧ b.AddrFree
  | _              => False

/-- **An address-free fragment elaborates identically at every block and
    iteration.**  This is what lets a compiled expression live in a loop. -/
theorem elabAt_addrFree : ∀ (s : EWStmt), s.AddrFree →
    ∀ (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat),
      s.elabAt cta i ir im = s.expand := by
  intro s
  induction s with
  | skip => intro _ _ _ _ _; rfl
  | setR _ _ => intro _ _ _ _ _; rfl
  | shflXor _ _ _ => intro _ _ _ _ _; rfl
  | barrier => intro _ _ _ _ _; rfl
  | seq a b iha ihb =>
      intro h cta i ir im
      show WStmt.seq (a.elabAt cta i ir im) (b.elabAt cta i ir im)
            = WStmt.seq (a.elabAt 0 0 _ _) (b.elabAt 0 0 _ _)
      rw [iha h.1 cta i ir im, ihb h.2 cta i ir im]
      rfl
  | loadIdx _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | loadV4 _ _ _ _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | storeLane0 _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | storeLane _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | stSm _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | ldSm _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | forN _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | forM _ _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | cvtIF _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])

theorem sumSeq_addrFree (acc : Nat) :
    ∀ (bodies : List (EWStmt × WFExp)), (∀ p ∈ bodies, p.1.AddrFree) →
      (sumSeq acc bodies).AddrFree := by
  have key : ∀ (bodies : List (EWStmt × WFExp)), (∀ p ∈ bodies, p.1.AddrFree) →
      ∀ (init : EWStmt), init.AddrFree →
      (bodies.foldl (fun s (p : EWStmt × WFExp) =>
        .seq s (.seq p.1 (.setR acc (.add (.reg acc) p.2)))) init).AddrFree := by
    intro bodies
    induction bodies with
    | nil => intro _ init hi; exact hi
    | cons p ps ih =>
        intro h init hi
        exact ih (fun q hq => h q (List.mem_cons_of_mem p hq)) _
          ⟨hi, h p (List.mem_cons_self), trivial⟩
  intro bodies h
  exact key bodies h _ trivial

/-- **The compiler emits no addresses.**  Immediate from the shape of
    `compileW`: every constructor it produces is `skip`, `seq` or `setR`. -/
theorem compileW_addrFree : ∀ {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat),
    ((compileW ve c e).1).AddrFree := by
  intro Γ e
  induction e with
  | var _ => intro _ _; trivial
  | lit _ => intro _ _; trivial
  | add a b iha ihb => intro ve c; exact ⟨iha ve c, ihb ve (c + slots a)⟩
  | mul a b iha ihb => intro ve c; exact ⟨iha ve c, ihb ve (c + slots a)⟩
  | neg a ih => intro ve c; exact ih ve c
  | inv a ih => intro ve c; exact ih ve c
  | exp a ih => intro ve c; exact ih ve c
  | rsqrt a ih => intro ve c; exact ih ve c
  | sum n f ih =>
      intro ve c
      refine sumSeq_addrFree c _ ?_
      intro p hp
      obtain ⟨j, _, hj⟩ := List.mem_map.mp hp
      rw [← hj]
      exact ih j ve (c + 1)
  | letE a b iha ihb =>
      intro ve c
      exact ⟨⟨iha ve c, trivial⟩, ihb (extend ve (.reg (c + slots a))) (c + slots a + 1)⟩

/-- An address-free fragment leaves memory alone — it has no store to make. -/
theorem addrFree_mem : ∀ (s : EWStmt), s.AddrFree → ∀ (st : WSt), (runW s st).mem = st.mem := by
  intro s
  induction s with
  | skip => intro _ st; rfl
  | setR r e => intro _ st; show (WSt.setReg _ r _).mem = _; rw [WSt.mem_setReg]
  | shflXor d s' m => intro _ st; rfl
  | barrier => intro _ st; rfl
  | seq a b iha ihb =>
      intro h st
      show (runW b (runW a st)).mem = _
      rw [ihb h.2 (runW a st), iha h.1 st]
  | loadIdx _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | loadV4 _ _ _ _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | storeLane0 _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | storeLane _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | stSm _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | ldSm _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | forN _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | forM _ _ _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])
  | cvtIF _ _ => intro h; exact absurd h (by simp [EWStmt.AddrFree])

/-- The compiled body leaves memory alone. -/
theorem compileW_mem {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat) (st : WSt) :
    (runW (compileW ve c e).1 st).mem = st.mem :=
  addrFree_mem _ (compileW_addrFree e ve c) st

/-- The compiled body runs the same at any block and iteration. -/
theorem compileW_elabAt {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat)
    (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    (((compileW ve c e).1.elabAt cta i ir im).run st) = runW (compileW ve c e).1 st := by
  rw [elabAt_addrFree _ (compileW_addrFree e ve c) cta i ir im]
  rfl

@[simp] theorem runW_skip (st : WSt) : runW .skip st = st := rfl
/-- The generated body only reads registers below the end of its range. -/
theorem compileW_regsIn :
    ∀ {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat),
      (∀ i, (ve i).regsIn 0 c) →
      ((compileW ve c e).2).regsIn 0 (c + slots e) := by
  intro Γ e
  induction e with
  | var i =>
      intro ve c hve
      exact WFExp.regsIn_weaken 0 c 0 (c + slots (Expr.var i))
        (Nat.le_refl _) (by simp only [slots]; omega) _ (hve i)
  | lit n => intro ve c _; trivial
  | add a b iha ihb =>
      intro ve c hve
      refine ⟨WFExp.regsIn_weaken 0 (c + slots a) 0 (c + slots (Expr.add a b))
                (Nat.le_refl _) (by simp only [slots]; omega) _ (iha ve c hve), ?_⟩
      refine WFExp.regsIn_weaken 0 (c + slots a + slots b) 0 (c + slots (Expr.add a b))
                (Nat.le_refl _) (by simp only [slots]; omega) _ ?_
      exact ihb ve (c + slots a) (fun i =>
        WFExp.regsIn_weaken 0 c 0 (c + slots a) (Nat.le_refl _) (by omega) _ (hve i))
  | mul a b iha ihb =>
      intro ve c hve
      refine ⟨WFExp.regsIn_weaken 0 (c + slots a) 0 (c + slots (Expr.mul a b))
                (Nat.le_refl _) (by simp only [slots]; omega) _ (iha ve c hve), ?_⟩
      refine WFExp.regsIn_weaken 0 (c + slots a + slots b) 0 (c + slots (Expr.mul a b))
                (Nat.le_refl _) (by simp only [slots]; omega) _ ?_
      exact ihb ve (c + slots a) (fun i =>
        WFExp.regsIn_weaken 0 c 0 (c + slots a) (Nat.le_refl _) (by omega) _ (hve i))
  | neg a ih   => intro ve c hve; exact ih ve c hve
  | inv a ih   => intro ve c hve; exact ih ve c hve
  | exp a ih   => intro ve c hve; exact ih ve c hve
  | rsqrt a ih => intro ve c hve; exact ih ve c hve
  | sum n f ih =>
      intro ve c _
      exact ⟨Nat.zero_le _, by simp only [slots]; omega⟩
  | letE a b iha ihb =>
      intro ve c hve
      have hve' : ∀ i, ((extend ve (WFExp.reg (c + slots a))) i).regsIn 0 (c + slots a + 1) := by
        intro i
        show WFExp.regsIn 0 (c + slots a + 1) (if _ : _ then _ else _)
        split
        · exact WFExp.regsIn_weaken 0 c 0 (c + slots a + 1) (Nat.le_refl _) (by omega) _ (hve _)
        · exact ⟨Nat.zero_le _, by omega⟩
      refine WFExp.regsIn_weaken 0 (c + slots a + 1 + slots b) 0 (c + slots (Expr.letE a b))
        (Nat.le_refl _) (by simp only [slots]; omega) _ ?_
      exact ihb (extend ve (WFExp.reg (c + slots a))) (c + slots a + 1) hve'


/-- Running a left-folded sequence is folding the runs. -/
theorem runW_foldl {α : Type} (g : α → EWStmt) :
    ∀ (L : List α) (init : EWStmt) (st : WSt),
      runW (L.foldl (fun s a => .seq s (g a)) init) st
        = L.foldl (fun s' a => runW (g a) s') (runW init st) := by
  intro L
  induction L with
  | nil => intro init st; rfl
  | cons a L ih => intro init st; rw [List.foldl_cons, ih, List.foldl_cons]; rfl

/-- Prologues never write a register below their base. -/
theorem compileW_frame :
    ∀ {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat) (st : WSt) (r : Nat), r < c →
      (runW (compileW ve c e).1 st).regs r = st.regs r := by
  intro Γ e
  induction e with
  | var i => intro ve c st r _; rfl
  | lit n => intro ve c st r _; rfl
  | add a b iha ihb =>
      intro ve c st r hr
      show (runW (compileW ve (c + slots a) b).1 (runW (compileW ve c a).1 st)).regs r = _
      rw [ihb ve (c + slots a) _ r (by omega), iha ve c st r hr]
  | mul a b iha ihb =>
      intro ve c st r hr
      show (runW (compileW ve (c + slots a) b).1 (runW (compileW ve c a).1 st)).regs r = _
      rw [ihb ve (c + slots a) _ r (by omega), iha ve c st r hr]
  | neg a ih   => intro ve c st r hr; exact ih ve c st r hr
  | inv a ih   => intro ve c st r hr; exact ih ve c st r hr
  | exp a ih   => intro ve c st r hr; exact ih ve c st r hr
  | rsqrt a ih => intro ve c st r hr; exact ih ve c st r hr
  | sum n f ih =>
      intro ve c st r hr
      show (runW (sumSeq c _) st).regs r = _
      rw [show sumSeq c ((List.finRange n).map (fun j => compileW ve (c + 1) (f j)))
            = (List.finRange n).foldl
                (fun s j => .seq s (.seq (compileW ve (c+1) (f j)).1
                   (.setR c (.add (.reg c) (compileW ve (c+1) (f j)).2))))
                (.setR c (.lit NumOps.zero)) from by
            simp only [sumSeq, List.foldl_map]]
      rw [runW_foldl]
      have key : ∀ (L : List (Fin n)) (s : WSt), s.regs r = st.regs r →
          (L.foldl (fun s' j => runW (.seq (compileW ve (c+1) (f j)).1
             (.setR c (.add (.reg c) (compileW ve (c+1) (f j)).2))) s') s).regs r
            = st.regs r := by
        intro L
        induction L with
        | nil => intro s hs; exact hs
        | cons j L ihL =>
            intro s hs
            refine ihL _ ?_
            show (WSt.setReg (runW (compileW ve (c+1) (f j)).1 s) c _).regs r = _
            rw [WSt.regs_setReg_other _ c r _ (by omega),
                ih j ve (c+1) s r (by omega), hs]
      exact key (List.finRange n) _ (by
        show (WSt.setReg st c _).regs r = _
        exact WSt.regs_setReg_other st c r _ (by omega))
  | letE a b iha ihb =>
      intro ve c st r hr
      show (runW (compileW (extend ve (WFExp.reg (c + slots a))) (c + slots a + 1) b).1
              (WSt.setReg (runW (compileW ve c a).1 st) (c + slots a) _)).regs r = _
      rw [ihb _ (c + slots a + 1) _ r (by omega),
          WSt.regs_setReg_other _ (c + slots a) r _ (by omega),
          iha ve c st r hr]


/-- **A model spec compiles onto the machine that runs — correctly, per lane.**

    All 32 lanes execute the same instructions; each lane's result is the
    denotation of the spec under *that lane's* environment.  Exact `Float32`
    equality.  With `WarpEmit` this is the missing link: a user's `Expr` now
    reaches PTX, not just the paper machine. -/
theorem compileW_sound :
    ∀ {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat),
      (∀ i, (ve i).regsIn 0 c) → ∀ (st : WSt) (l : Lane),
      ((compileW ve c e).2).eval (runW (compileW ve c e).1 st) l
        = denote (fun i => (ve i).eval st l) e := by
  intro Γ e
  induction e with
  | var i => intro ve c _ st l; rfl
  | lit n => intro ve c _ st l; rfl
  | add a b iha ihb =>
      intro ve c hve st l
      have hveB : ∀ i, (ve i).regsIn 0 (c + slots a) := fun i =>
        WFExp.regsIn_weaken 0 c 0 (c + slots a) (Nat.le_refl _) (by omega) _ (hve i)
      have henv : ∀ i, (ve i).eval (runW (compileW ve c a).1 st) l = (ve i).eval st l := fun i =>
        WFExp.eval_frame 0 c (ve i) st _ l (hve i)
          (fun r _ hhi => compileW_frame a ve c st r hhi)
      have hkeep : ((compileW ve c a).2).eval
            (runW (compileW ve (c + slots a) b).1 (runW (compileW ve c a).1 st)) l
          = ((compileW ve c a).2).eval (runW (compileW ve c a).1 st) l :=
        WFExp.eval_frame 0 (c + slots a) _ _ _ l (compileW_regsIn a ve c hve)
          (fun r _ hhi => compileW_frame b ve (c + slots a) _ r hhi)
      show NumOps.add (((compileW ve c a).2).eval
              (runW (compileW ve (c + slots a) b).1 (runW (compileW ve c a).1 st)) l)
            (((compileW ve (c + slots a) b).2).eval
              (runW (compileW ve (c + slots a) b).1 (runW (compileW ve c a).1 st)) l)
          = NumOps.add _ _
      rw [hkeep, iha ve c hve st l, ihb ve (c + slots a) hveB _ l]
      simp only [henv]
  | mul a b iha ihb =>
      intro ve c hve st l
      have hveB : ∀ i, (ve i).regsIn 0 (c + slots a) := fun i =>
        WFExp.regsIn_weaken 0 c 0 (c + slots a) (Nat.le_refl _) (by omega) _ (hve i)
      have henv : ∀ i, (ve i).eval (runW (compileW ve c a).1 st) l = (ve i).eval st l := fun i =>
        WFExp.eval_frame 0 c (ve i) st _ l (hve i)
          (fun r _ hhi => compileW_frame a ve c st r hhi)
      have hkeep : ((compileW ve c a).2).eval
            (runW (compileW ve (c + slots a) b).1 (runW (compileW ve c a).1 st)) l
          = ((compileW ve c a).2).eval (runW (compileW ve c a).1 st) l :=
        WFExp.eval_frame 0 (c + slots a) _ _ _ l (compileW_regsIn a ve c hve)
          (fun r _ hhi => compileW_frame b ve (c + slots a) _ r hhi)
      show NumOps.mul (((compileW ve c a).2).eval
              (runW (compileW ve (c + slots a) b).1 (runW (compileW ve c a).1 st)) l)
            (((compileW ve (c + slots a) b).2).eval
              (runW (compileW ve (c + slots a) b).1 (runW (compileW ve c a).1 st)) l)
          = NumOps.mul _ _
      rw [hkeep, iha ve c hve st l, ihb ve (c + slots a) hveB _ l]
      simp only [henv]
  | neg a ih =>
      intro ve c hve st l
      show NumOps.neg (((compileW ve c a).2).eval (runW (compileW ve c a).1 st) l) = NumOps.neg _
      rw [ih ve c hve st l]
  | inv a ih =>
      intro ve c hve st l
      show NumOps.inv (((compileW ve c a).2).eval (runW (compileW ve c a).1 st) l) = NumOps.inv _
      rw [ih ve c hve st l]
  | exp a ih =>
      intro ve c hve st l
      show NumOps.exp (((compileW ve c a).2).eval (runW (compileW ve c a).1 st) l) = NumOps.exp _
      rw [ih ve c hve st l]
  | rsqrt a ih =>
      intro ve c hve st l
      show NumOps.rsqrt (((compileW ve c a).2).eval (runW (compileW ve c a).1 st) l)
          = NumOps.rsqrt _
      rw [ih ve c hve st l]
  | sum n f ih =>
      intro ve c hve st l
      have hveB : ∀ i, (ve i).regsIn 0 (c + 1) := fun i =>
        WFExp.regsIn_weaken 0 c 0 (c + 1) (Nat.le_refl _) (by omega) _ (hve i)
      show ((runW (sumSeq c _) st).regs c l) = _
      rw [show sumSeq c ((List.finRange n).map (fun j => compileW ve (c + 1) (f j)))
            = (List.finRange n).foldl
                (fun s j => .seq s (.seq (compileW ve (c+1) (f j)).1
                   (.setR c (.add (.reg c) (compileW ve (c+1) (f j)).2))))
                (.setR c (.lit NumOps.zero)) from by
            simp only [sumSeq, List.foldl_map],
          runW_foldl]
      have key : ∀ (L : List (Fin n)) (s : WSt),
          (∀ r, r < c → s.regs r = st.regs r) →
          (L.foldl (fun s' j => runW (.seq (compileW ve (c+1) (f j)).1
             (.setR c (.add (.reg c) (compileW ve (c+1) (f j)).2))) s') s).regs c l
            = L.foldl (fun v j => NumOps.add v
                (denote (fun i => (ve i).eval st l) (f j))) (s.regs c l) := by
        intro L
        induction L with
        | nil => intro s _; rfl
        | cons j L ihL =>
            intro s hs
            have hstep : (runW (.seq (compileW ve (c+1) (f j)).1
                  (.setR c (.add (.reg c) (compileW ve (c+1) (f j)).2))) s).regs c l
                = NumOps.add (s.regs c l) (denote (fun i => (ve i).eval st l) (f j)) := by
              show (WSt.setReg (runW (compileW ve (c+1) (f j)).1 s) c _).regs c l = _
              rw [WSt.regs_setReg_same]
              show NumOps.add ((runW (compileW ve (c+1) (f j)).1 s).regs c l)
                    (WFExp.eval (runW (compileW ve (c+1) (f j)).1 s) l
                      (compileW ve (c+1) (f j)).2) = _
              rw [compileW_frame (f j) ve (c+1) s c (by omega), ih j ve (c+1) hveB s l]
              have : ∀ i, (ve i).eval s l = (ve i).eval st l := fun i =>
                WFExp.eval_frame 0 c (ve i) st s l (hve i) (fun r _ hhi => hs r hhi)
              simp only [this]
            have hpres : ∀ r, r < c → (runW (.seq (compileW ve (c+1) (f j)).1
                  (.setR c (.add (.reg c) (compileW ve (c+1) (f j)).2))) s).regs r = st.regs r := by
              intro r hr
              show (WSt.setReg (runW (compileW ve (c+1) (f j)).1 s) c _).regs r = _
              rw [WSt.regs_setReg_other _ c r _ (by omega),
                  compileW_frame (f j) ve (c+1) s r (by omega), hs r hr]
            rw [List.foldl_cons, List.foldl_cons, ihL _ hpres, hstep]
      rw [key (List.finRange n) _ (by
            intro r hr
            show (WSt.setReg st c _).regs r = _
            exact WSt.regs_setReg_other st c r _ (by omega))]
      show (List.finRange n).foldl _ ((WSt.setReg st c _).regs c l) = _
      rw [WSt.regs_setReg_same]
      rfl
  | letE a b iha ihb =>
      intro ve c hve st l
      have hve' : ∀ i,
          ((extend ve (WFExp.reg (c + slots a))) i).regsIn 0 (c + slots a + 1) := by
        intro i
        show WFExp.regsIn 0 (c + slots a + 1) (if _ : _ then _ else _)
        split
        · exact WFExp.regsIn_weaken 0 c 0 (c + slots a + 1) (Nat.le_refl _) (by omega) _ (hve _)
        · exact ⟨Nat.zero_le _, by omega⟩
      show ((compileW (extend ve (WFExp.reg (c + slots a))) (c + slots a + 1) b).2).eval
             (runW (compileW (extend ve (WFExp.reg (c + slots a))) (c + slots a + 1) b).1
               (WSt.setReg (runW (compileW ve c a).1 st) (c + slots a) _)) l = _
      rw [ihb _ (c + slots a + 1) hve' _ l]
      show denote _ b
          = denote (extend (fun i => (ve i).eval st l)
                     (denote (fun i => (ve i).eval st l) a)) b
      congr 1
      funext i
      show WFExp.eval _ l (if _ : _ then _ else _) = (if _ : _ then _ else _)
      split
      · refine WFExp.eval_frame 0 c (ve _) st _ l (hve _) (fun r _ hhi => ?_)
        rw [WSt.regs_setReg_other _ (c + slots a) r _ (by omega),
            compileW_frame a ve c st r hhi]
      · show (WSt.setReg _ (c + slots a) _).regs (c + slots a) l = _
        rw [WSt.regs_setReg_same]
        exact iha ve c hve st l


-- ---------------------------------------------------------------------------
-- A complete kernel from a spec
-- ---------------------------------------------------------------------------

/-- Load spec variables `0 .. k-1` into registers `0 .. k-1`.  `WFExp` has no
    `load` constructor by design — memory traffic is a *statement*, so it
    happens in a prologue and the expression language stays pure.

    Recursive on `Nat` rather than folded over a list so the correctness proof
    is a plain forward induction. -/
def loadSeqN (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE) : Nat → EWStmt
  | 0     => .skip
  | k + 1 => .seq (loadSeqN inB inIx k)
                  (if h : k < Γ then .loadIdx k (inB ⟨k, h⟩) (inIx ⟨k, h⟩) else .skip)

def loadSeq (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE) : EWStmt := loadSeqN inB inIx Γ

theorem loadSeqN_mem (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE) :
    ∀ (k : Nat) (st : WSt), (runW (loadSeqN inB inIx k) st).mem = st.mem := by
  intro k
  induction k with
  | zero => intro st; rfl
  | succ k ih =>
      intro st
      show (runW (if _ : _ then _ else _) (runW (loadSeqN inB inIx k) st)).mem = _
      split
      · show (WSt.setReg _ k _).mem = _; rw [WSt.mem_setReg, ih st]
      · show (runW EWStmt.skip _).mem = _; rw [runW_skip, ih st]

/-- After the prologue, variable `i` sits in register `i` holding the memory it
    was loaded from. -/
theorem loadSeqN_regs (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE) (st : WSt) (l : Lane) :
    ∀ (k : Nat) (i : Fin Γ), i.val < k →
      (runW (loadSeqN inB inIx k) st).regs i.val l
        = st.mem (inB i) ((inIx i).eval 0 0 l) := by
  intro k
  induction k with
  | zero => intro i h; exact absurd h (Nat.not_lt_zero _)
  | succ k ih =>
      intro i hi
      show (runW (if _ : _ then _ else _) (runW (loadSeqN inB inIx k) st)).regs i.val l = _
      by_cases hik : i.val = k
      · have hk : k < Γ := hik ▸ i.isLt
        rw [dif_pos hk]
        show (WSt.setReg _ k
          (fun l => WSt.mem _ (inB ⟨k, hk⟩) ((inIx ⟨k, hk⟩).eval 0 0 l))).regs i.val l = _
        have hfi : (⟨k, hk⟩ : Fin Γ) = i := Fin.ext hik.symm
        rw [hik, WSt.regs_setReg_same, loadSeqN_mem inB inIx k st, hfi]
      · have hlt : i.val < k := by omega
        by_cases hk : k < Γ
        · rw [dif_pos hk]
          show (WSt.setReg _ k _).regs i.val l = _
          rw [WSt.regs_setReg_other _ k i.val _ hik, ih i hlt]
        · rw [dif_neg hk]
          show (runW EWStmt.skip _).regs i.val l = _
          rw [runW_skip, ih i hlt]

theorem loadSeqN_memAt (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE)
    (cta j : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (k : Nat) (st : WSt),
      (((loadSeqN inB inIx k).elabAt cta j ir im).run st).mem = st.mem := by
  intro k
  induction k with
  | zero => intro st; rfl
  | succ k ih =>
      intro st
      show (((if _ : _ then _ else _ : EWStmt).elabAt cta j ir im).run
              (((loadSeqN inB inIx k).elabAt cta j ir im).run st)).mem = _
      split
      · show (WSt.setReg _ k _).mem = _; rw [WSt.mem_setReg, ih st]
      · show (WStmt.run .skip _).mem = _
        show (((loadSeqN inB inIx k).elabAt cta j ir im).run st).mem = _
        rw [ih st]

/-- `loadSeqN_regs` at an arbitrary block and iteration. -/
theorem loadSeqN_regsAt (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE) (st : WSt)
    (l : Lane) (cta j : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ (k : Nat) (i : Fin Γ), i.val < k →
      (((loadSeqN inB inIx k).elabAt cta j ir im).run st).regs i.val l
        = st.mem (inB i) ((inIx i).eval cta j l ir im) := by
  intro k
  induction k with
  | zero => intro i h; exact absurd h (Nat.not_lt_zero _)
  | succ k ih =>
      intro i hi
      show (((if _ : _ then _ else _ : EWStmt).elabAt cta j ir im).run
            (((loadSeqN inB inIx k).elabAt cta j ir im).run st)).regs i.val l = _
      by_cases hik : i.val = k
      · have hk : k < Γ := hik ▸ i.isLt
        rw [dif_pos hk]
        show (WSt.setReg _ k
          (fun l => WSt.mem _ (inB ⟨k, hk⟩) ((inIx ⟨k, hk⟩).eval cta j l ir im))).regs i.val l = _
        have hfi : (⟨k, hk⟩ : Fin Γ) = i := Fin.ext hik.symm
        rw [hik, WSt.regs_setReg_same, loadSeqN_memAt inB inIx cta j ir im k st, hfi]
      · have hlt : i.val < k := by omega
        by_cases hk : k < Γ
        · rw [dif_pos hk]
          show (WSt.setReg _ k _).regs i.val l = _
          rw [WSt.regs_setReg_other _ k i.val _ hik, ih i hlt]
        · rw [dif_neg hk]
          show (runW EWStmt.skip _).regs i.val l = _
          rw [runW_skip, ih i hlt]

theorem loadSeq_regs (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE) (st : WSt) (i : Fin Γ)
    (l : Lane) :
    (runW (loadSeq inB inIx) st).regs i.val l = st.mem (inB i) ((inIx i).eval 0 0 l) :=
  loadSeqN_regs inB inIx st l Γ i i.isLt

/-- Compile a spec into a runnable kernel: load the inputs, evaluate, park the
    result in `resReg`, and have every lane store to its own output slot. -/
def compileWKernel (inB : Fin Γ → Buf) (out : Buf) (inIx : Fin Γ → IdxE) (e : Expr Γ)
    (oix : IdxE) (resReg : Nat) : EWStmt :=
  .seq (.seq (loadSeq inB inIx)
             (.seq (compileW (fun i => .reg i.val) Γ e).1
                   (.setR resReg (compileW (fun i => .reg i.val) Γ e).2)))
       (.storeLane out oix resReg)

/-- **A user's spec, compiled, computes the spec from memory.**

    Lane `l` ends with `resReg` holding the denotation of `e` under the
    environment "variable `i` is whatever `inIx i` addresses in `inB`, as seen
    by lane `l`".  Loads, register allocation, evaluation order and per-lane
    divergence are all discharged.  Exact `Float32` equality. -/
theorem compileWKernel_correct (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE) (e : Expr Γ)
    (st : WSt) (l : Lane) :
    ((compileW (fun i => .reg i.val) Γ e).2).eval
        (runW (compileW (fun i => .reg i.val) Γ e).1 (runW (loadSeq inB inIx) st)) l
      = denote (fun i => st.mem (inB i) ((inIx i).eval 0 0 l)) e := by
  rw [compileW_sound e (fun i => .reg i.val) Γ (fun i => ⟨Nat.zero_le _, i.isLt⟩) _ l]
  congr 1
  funext i
  exact loadSeq_regs inB inIx st i l

/-- **The compiled body, at any block and iteration.**

    `compileWKernel_correct` is this at `(0, 0)`.  Generalising it is what lets
    a compiled expression sit inside a loop — i.e. what lets one lane handle
    more than one element. -/
theorem compileWKernel_correctAt (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE)
    (e : Expr Γ) (st : WSt) (l : Lane) (cta j : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) :
    ((compileW (fun i => .reg i.val) Γ e).2).eval
        ((((compileW (fun i => .reg i.val) Γ e).1).elabAt cta j ir im).run
          (((loadSeq inB inIx).elabAt cta j ir im).run st)) l
      = denote (fun i => st.mem (inB i) ((inIx i).eval cta j l ir im)) e := by
  rw [compileW_elabAt e (fun i => .reg i.val) Γ cta j ir im,
      compileW_sound e (fun i => .reg i.val) Γ (fun i => ⟨Nat.zero_le _, i.isLt⟩) _ l]
  congr 1
  funext i
  exact loadSeqN_regsAt inB inIx st l cta j ir im Γ i i.isLt

/-- **…and the result lands at the address it was supposed to.**

    `compileWKernel_correct` proves the *value* a lane computes.  It says
    nothing about *where* that value goes, and a kernel that computes perfectly
    and stores to the wrong slot satisfies it.  This closes that: the output
    buffer, at lane `l₀`'s output address, holds `denote … e`.

    `hagree` is the address side condition, and it is stated as agreement rather
    than injectivity so that kernels which deliberately have several lanes write
    the same value to the same slot are covered.  An injective address map gives
    it immediately. -/
theorem compileWKernel_stores (inB : Fin Γ → Buf) (out : Buf) (inIx : Fin Γ → IdxE)
    (e : Expr Γ) (oix : IdxE) (resReg : Nat) (st : WSt) (l0 : Lane)
    (hagree : ∀ l, oix.eval 0 0 l = oix.eval 0 0 l0 →
      denote (fun i => st.mem (inB i) ((inIx i).eval 0 0 l)) e
        = denote (fun i => st.mem (inB i) ((inIx i).eval 0 0 l0)) e) :
    (((compileWKernel inB out inIx e oix resReg).elabIn 0).run st).mem out
        (oix.eval 0 0 l0)
      = denote (fun i => st.mem (inB i) ((inIx i).eval 0 0 l0)) e := by
  show ((WStmt.storeLane out (fun l => oix.eval 0 0 l) resReg).run
          ((WStmt.setR resReg (compileW (fun i => .reg i.val) Γ e).2).run
            (runW (compileW (fun i => .reg i.val) Γ e).1
              (runW (loadSeq inB inIx) st)))).mem out (oix.eval 0 0 l0) = _
  rw [storeLane_at out (fun l => oix.eval 0 0 l) resReg _ l0
        (fun l h => by
          have hb : oix.eval 0 0 l = oix.eval 0 0 l0 := h
          simp only [wrun_setR, WSt.regs_setReg_same]
          rw [compileWKernel_correct inB inIx e st l,
              compileWKernel_correct inB inIx e st l0]
          exact hagree l hb),
      wrun_setR, WSt.regs_setReg_same]
  exact compileWKernel_correct inB inIx e st l0

/-- **The store theorem at any block and iteration.**

    `compileWKernel_stores` is this at `(0, 0)`.  A *stage* runs a whole grid,
    so its contract has to hold for every block — which is what this supplies. -/
theorem compileWKernel_storesAt (inB : Fin Γ → Buf) (out : Buf) (inIx : Fin Γ → IdxE)
    (e : Expr Γ) (oix : IdxE) (resReg : Nat) (cta j : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt) (l0 : Lane)
    (hagree : ∀ l, oix.eval cta j l ir im = oix.eval cta j l0 ir im →
      denote (fun i => st.mem (inB i) ((inIx i).eval cta j l ir im)) e
        = denote (fun i => st.mem (inB i) ((inIx i).eval cta j l0 ir im)) e) :
    (((compileWKernel inB out inIx e oix resReg).elabAt cta j ir im).run st).mem out
        (oix.eval cta j l0 ir im)
      = denote (fun i => st.mem (inB i) ((inIx i).eval cta j l0 ir im)) e := by
  show ((WStmt.storeLane out (fun l => oix.eval cta j l ir im) resReg).run
          ((WStmt.setR resReg (compileW (fun i => .reg i.val) Γ e).2).run
            (((compileW (fun i => .reg i.val) Γ e).1.elabAt cta j ir im).run
              (((loadSeq inB inIx).elabAt cta j ir im).run st)))).mem out
        (oix.eval cta j l0 ir im) = _
  rw [storeLane_at out (fun l => oix.eval cta j l ir im) resReg _ l0
        (fun l h => by
          have hb : oix.eval cta j l ir im = oix.eval cta j l0 ir im := h
          simp only [wrun_setR, WSt.regs_setReg_same]
          rw [compileWKernel_correctAt inB inIx e st l cta j ir im,
              compileWKernel_correctAt inB inIx e st l0 cta j ir im]
          exact hagree l hb),
      wrun_setR, WSt.regs_setReg_same]
  exact compileWKernel_correctAt inB inIx e st l0 cta j ir im

/-- **A compiled kernel writes only its output buffer.** -/
theorem compileWKernel_otherBuf (inB : Fin Γ → Buf) (out : Buf) (inIx : Fin Γ → IdxE)
    (e : Expr Γ) (oix : IdxE) (resReg : Nat) (cta j : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt) (c : Buf) (hc : c ≠ out) :
    (((compileWKernel inB out inIx e oix resReg).elabAt cta j ir im).run st).mem c
      = st.mem c := by
  show ((WStmt.storeLane out (fun l => oix.eval cta j l ir im) resReg).run
          ((WStmt.setR resReg (compileW (fun i => .reg i.val) Γ e).2).run
            (((compileW (fun i => .reg i.val) Γ e).1.elabAt cta j ir im).run
              (((loadSeq inB inIx).elabAt cta j ir im).run st)))).mem c = _
  rw [storeLane_otherBuf out (fun l => oix.eval cta j l ir im) resReg _ c hc,
      wrun_setR, WSt.mem_setReg, compileW_elabAt, compileW_mem]
  exact congrFun (loadSeqN_memAt inB inIx cta j ir im Γ st) c

/-- **…and only at the addresses some lane owns.** -/
theorem compileWKernel_otherAddr (inB : Fin Γ → Buf) (out : Buf) (inIx : Fin Γ → IdxE)
    (e : Expr Γ) (oix : IdxE) (resReg : Nat) (cta j : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt) (a : Nat)
    (ha : ∀ l : Lane, oix.eval cta j l ir im ≠ a) :
    (((compileWKernel inB out inIx e oix resReg).elabAt cta j ir im).run st).mem out a
      = st.mem out a := by
  show ((WStmt.storeLane out (fun l => oix.eval cta j l ir im) resReg).run
          ((WStmt.setR resReg (compileW (fun i => .reg i.val) Γ e).2).run
            (((compileW (fun i => .reg i.val) Γ e).1.elabAt cta j ir im).run
              (((loadSeq inB inIx).elabAt cta j ir im).run st)))).mem out a = _
  rw [storeLane_otherAddr out (fun l => oix.eval cta j l ir im) resReg _ a ha,
      wrun_setR, WSt.mem_setReg, compileW_elabAt, compileW_mem]
  exact congrFun (congrFun (loadSeqN_memAt inB inIx cta j ir im Γ st) out) a

-- ---------------------------------------------------------------------------
-- The compiler emits no data-dependent addresses
-- ---------------------------------------------------------------------------

/-! `EWStmt.IdxFree` is a precondition of the lowering theorems, so every
    kernel-building function has to say what it preserves.  For the compiler
    the answer is strong and cheap: `compileW` emits only `skip`/`seq`/`setR`,
    none of which carries an address at all, so its output is unconditionally
    address-free.  Only the prologue and the store mention addresses, and they
    inherit the condition from the index expressions the caller supplied. -/

theorem sumSeq_idxFree (acc : Nat) :
    ∀ (bodies : List (EWStmt × WFExp)) (init : EWStmt), init.IdxFree →
      (∀ p ∈ bodies, (p.1).IdxFree) →
      (bodies.foldl
        (fun s (p : EWStmt × WFExp) =>
          EWStmt.seq s (.seq p.1 (.setR acc (.add (.reg acc) p.2)))) init).IdxFree := by
  intro bodies
  induction bodies with
  | nil => intro init hi _; exact hi
  | cons p L ih =>
      intro init hi hp
      exact ih _ ⟨hi, hp p (List.Mem.head _), trivial⟩
        (fun q hq => hp q (List.Mem.tail _ hq))

theorem compileW_idxFree : ∀ {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat),
    (compileW ve c e).1.IdxFree := by
  intro Γ e
  induction e with
  | var i => intro _ _; trivial
  | lit n => intro _ _; trivial
  | add a b iha ihb => intro ve c; exact ⟨iha ve c, ihb ve (c + slots a)⟩
  | mul a b iha ihb => intro ve c; exact ⟨iha ve c, ihb ve (c + slots a)⟩
  | neg a ih   => intro ve c; exact ih ve c
  | inv a ih   => intro ve c; exact ih ve c
  | exp a ih   => intro ve c; exact ih ve c
  | rsqrt a ih => intro ve c; exact ih ve c
  | sum n f ih =>
      intro ve c
      refine sumSeq_idxFree c _ _ trivial ?_
      intro p hp
      rcases List.mem_map.mp hp with ⟨j, _, hj⟩
      rw [← hj]
      exact ih j ve (c + 1)
  | letE a b iha ihb =>
      intro ve c
      exact ⟨⟨iha ve c, trivial⟩, ihb _ (c + slots a + 1)⟩

theorem compileW_flat : ∀ {Γ : Nat} (e : Expr Γ) (ve : Fin Γ → WFExp) (c : Nat),
    (compileW ve c e).1.Flat := by
  intro Γ e
  induction e with
  | var i => intro _ _; trivial
  | lit n => intro _ _; trivial
  | add a b iha ihb => intro ve c; exact ⟨iha ve c, ihb ve (c + slots a)⟩
  | mul a b iha ihb => intro ve c; exact ⟨iha ve c, ihb ve (c + slots a)⟩
  | neg a ih   => intro ve c; exact ih ve c
  | inv a ih   => intro ve c; exact ih ve c
  | exp a ih   => intro ve c; exact ih ve c
  | rsqrt a ih => intro ve c; exact ih ve c
  | sum n f ih =>
      intro ve c
      have key : ∀ (L : List (EWStmt × WFExp)) (init : EWStmt), init.Flat →
          (∀ p ∈ L, (p.1).Flat) →
          (L.foldl (fun s (p : EWStmt × WFExp) =>
            EWStmt.seq s (.seq p.1 (.setR c (.add (.reg c) p.2)))) init).Flat := by
        intro L
        induction L with
        | nil => intro init hi _; exact hi
        | cons p L ihl =>
            intro init hi hp
            exact ihl _ ⟨hi, hp p (List.Mem.head _), trivial⟩
              (fun q hq => hp q (List.Mem.tail _ hq))
      refine key _ _ trivial ?_
      intro p hp
      rcases List.mem_map.mp hp with ⟨j, _, hj⟩
      rw [← hj]
      exact ih j ve (c + 1)
  | letE a b iha ihb =>
      intro ve c
      exact ⟨⟨iha ve c, trivial⟩, ihb _ (c + slots a + 1)⟩

theorem loadSeqN_flat (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE) :
    ∀ k : Nat, (loadSeqN inB inIx k).Flat := by
  intro k
  induction k with
  | zero => trivial
  | succ k ih =>
      refine ⟨ih, ?_⟩
      show EWStmt.Flat (if _ : _ then _ else _)
      split <;> trivial

theorem compileWKernel_flat (inB : Fin Γ → Buf) (out : Buf) (inIx : Fin Γ → IdxE)
    (e : Expr Γ) (oix : IdxE) (resReg : Nat) :
    (compileWKernel inB out inIx e oix resReg).Flat :=
  ⟨⟨loadSeqN_flat inB inIx Γ, compileW_flat e (fun i => .reg i.val) Γ, trivial⟩, trivial⟩

theorem loadSeqN_idxFree (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE)
    (h : ∀ i, (inIx i).IregFree) : ∀ k : Nat, (loadSeqN inB inIx k).IdxFree := by
  intro k
  induction k with
  | zero => trivial
  | succ k ih =>
      refine ⟨ih, ?_⟩
      show EWStmt.IdxFree (if _ : _ then _ else _)
      split
      · exact h _
      · trivial

/-- **A compiled kernel has no data-dependent address unless the caller put one
    there.**  So the lowering theorems' precondition reduces to a condition on
    the index expressions the user supplied, which is where it belongs. -/
theorem compileWKernel_idxFree (inB : Fin Γ → Buf) (out : Buf) (inIx : Fin Γ → IdxE)
    (e : Expr Γ)
    (oix : IdxE) (resReg : Nat) (hin : ∀ i, (inIx i).IregFree) (hout : oix.IregFree) :
    (compileWKernel inB out inIx e oix resReg).IdxFree :=
  ⟨⟨loadSeqN_idxFree inB inIx hin Γ,
    compileW_idxFree e (fun i => .reg i.val) Γ, trivial⟩, hout⟩

-- ---------------------------------------------------------------------------
-- Multi-output programs: emitting the shared bindings once
-- ---------------------------------------------------------------------------

/-! A `VProg` is one telescope and many outputs — the shape `gradProg` produces,
    where every intermediate adjoint is bound once and referred to by variable.
    Compiling `p.get i` for each `i` with `compileW` would re-emit the whole
    telescope `m` times and throw the sharing away, which is the entire point of
    the representation.

    So: emit the telescope once (`compileTele`), then compile each output in the
    environment it leaves behind.

    The proof does not need a single new framing lemma, because a telescope *is*
    a `letE` chain and `compileW` already compiles those.  `compileTele_split`
    says the shared emission is *literally* what `compileW` would have done —
    same result register, same machine state — so `compileW_sound` transfers
    verbatim.  Getting the register allocation to line up exactly is what buys
    that; it is why `teleSlots` mirrors `compileW`'s `letE` case rather than
    picking its own layout. -/

/-- Registers a telescope occupies: each binding's accumulators, plus one to
    hold the binding itself.  Mirrors `slots (.letE a b) = slots a + 1 + slots b`
    exactly. -/
def teleSlots : {n : Nat} → Tele Γ n → Nat
  | 0,     .nil      => 0
  | _ + 1, .cons t e => teleSlots t + slots e + 1

/-- Where each slot of the telescope lives: spec variables stay where `ve` puts
    them, binding `i` lands in its own register. -/
def teleEnv : {n : Nat} → (Fin Γ → WFExp) → Nat → Tele Γ n → (Fin (Γ + n) → WFExp)
  | 0,     ve, _, .nil      => ve
  | _ + 1, ve, c, .cons t e =>
      extend (teleEnv ve c t) (.reg (c + teleSlots t + slots e))

/-- Emit the bindings, once each, in order. -/
def compileTele : {n : Nat} → (Fin Γ → WFExp) → Nat → Tele Γ n → EWStmt
  | 0,     _,  _, .nil      => .skip
  | _ + 1, ve, c, .cons t e =>
      .seq (compileTele ve c t)
           (.seq (compileW (teleEnv ve c t) (c + teleSlots t) e).1
                 (.setR (c + teleSlots t + slots e)
                        (compileW (teleEnv ve c t) (c + teleSlots t) e).2))

/-- **The shared emission is the same machine.**

    Compiling `t.bind b` in one go and compiling the telescope separately then
    `b` against it produce the same result register and the same state.  So `m`
    outputs sharing one emitted telescope compute exactly what `m` independent
    compilations would — sharing is sound, not merely plausible. -/
theorem compileTele_split : ∀ {n : Nat} (t : Tele Γ n) (b : Expr (Γ + n))
    (ve : Fin Γ → WFExp) (c : Nat),
    (compileW ve c (t.bind b)).2 = (compileW (teleEnv ve c t) (c + teleSlots t) b).2
    ∧ ∀ st : WSt, runW (compileW ve c (t.bind b)).1 st
        = runW (compileW (teleEnv ve c t) (c + teleSlots t) b).1
            (runW (compileTele ve c t) st) := by
  intro n t
  induction t with
  | nil => intro b ve c; exact ⟨rfl, fun _ => rfl⟩
  | @cons n t e ih =>
      intro b ve c
      have harith : c + teleSlots (Tele.cons t e) = c + teleSlots t + slots e + 1 := by
        show c + (teleSlots t + slots e + 1) = _
        omega
      have hIH := ih (.letE e b) ve c
      refine ⟨?_, ?_⟩
      · show (compileW ve c (t.bind (.letE e b))).2 = _
        rw [hIH.1, harith]
        rfl
      · intro st
        show runW (compileW ve c (t.bind (.letE e b))).1 st = _
        rw [hIH.2 st, harith]
        rfl

/-- **Every output of a multi-output program is computed correctly, from one
    emitted telescope.**

    This is the machine-level counterpart of `VProg.denote_get`: the shared
    bindings run once, and output `i` still denotes `p.get i`.  Applied to
    `gradProg`, it is the statement that the quadratic-not-exponential gradient
    representation survives compilation. -/
theorem compileVW_sound {m : Nat} (p : VProg Γ m) (i : Fin m) (ve : Fin Γ → WFExp)
    (c : Nat) (hve : ∀ k, (ve k).regsIn 0 c) (st : WSt) (l : Lane) :
    ((compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).2).eval
        (runW (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1
          (runW (compileTele ve c p.binds) st)) l
      = denote (fun k => (ve k).eval st l) (p.get i) := by
  have hsplit := compileTele_split p.binds (p.outs i) ve c
  rw [← hsplit.1, ← hsplit.2 st]
  exact compileW_sound (p.binds.bind (p.outs i)) ve c hve st l

-- ---------------------------------------------------------------------------
-- A multi-output kernel
-- ---------------------------------------------------------------------------

/-- Compute output `i` and store it.  Each output reuses the same scratch
    registers above the telescope, which is safe because the value is stored
    before the next output is computed. -/
def storeOut {m : Nat} (out : Buf) (p : VProg Γ m) (ve : Fin Γ → WFExp) (c : Nat)
    (oix : Fin m → IdxE) (resReg : Nat) (i : Fin m) : EWStmt :=
  .seq (.seq (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1
             (.setR resReg
                (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).2))
       (.storeLane out (oix i) resReg)

/-- A kernel for a whole multi-output program: load the inputs, emit the shared
    bindings once, then compute and store each output in turn. -/
def compileVWKernel {m : Nat} (inB : Fin Γ → Buf) (out : Buf) (inIx : Fin Γ → IdxE)
    (p : VProg Γ m) (oix : Fin m → IdxE) (resReg : Nat) : EWStmt :=
  .seq (.seq (loadSeq inB inIx) (compileTele (fun k => .reg k.val) Γ p.binds))
       ((List.finRange m).foldl
          (fun s i => .seq s (storeOut out p (fun k => .reg k.val) Γ oix resReg i))
          .skip)

/-- The shared-telescope emission carries no addresses at all — it is
    `skip`/`seq`/`setR` throughout, like `compileW`'s own output. -/
theorem compileTele_idxFree : ∀ {n : Nat} (t : Tele Γ n) (ve : Fin Γ → WFExp) (c : Nat),
    (compileTele ve c t).IdxFree := by
  intro n t
  induction t with
  | nil => intro ve c; trivial
  | cons t e ih =>
      intro ve c
      exact ⟨ih ve c, compileW_idxFree e _ _, trivial⟩

/-- **The whole multi-output kernel is address-free** given address-free index
    expressions — the discharge `flatKernel_sound_idxFree` needs, making the
    gradient's PTX-floor theorem stateable. -/
theorem compileTele_flat : ∀ {n : Nat} (t : Tele Γ n) (ve : Fin Γ → WFExp) (c : Nat),
    (compileTele ve c t).Flat := by
  intro n t
  induction t with
  | nil => intro ve c; trivial
  | cons t e ih => intro ve c; exact ⟨ih ve c, compileW_flat e _ _, trivial⟩

theorem compileVWKernel_flat {m : Nat} (inB : Fin Γ → Buf) (out : Buf)
    (inIx : Fin Γ → IdxE) (p : VProg Γ m) (oix : Fin m → IdxE) (resReg : Nat) :
    (compileVWKernel inB out inIx p oix resReg).Flat := by
  refine ⟨⟨loadSeqN_flat inB inIx Γ, compileTele_flat p.binds (fun k => .reg k.val) Γ⟩, ?_⟩
  have key : ∀ (L : List (Fin m)) (init : EWStmt), init.Flat →
      (L.foldl (fun s i => EWStmt.seq s
        (storeOut out p (fun k => .reg k.val) Γ oix resReg i)) init).Flat := by
    intro L
    induction L with
    | nil => intro init h; exact h
    | cons i L ih => intro init h; exact ih _ ⟨h, ⟨compileW_flat _ _ _, trivial⟩, trivial⟩
  exact key (List.finRange m) .skip trivial

theorem compileVWKernel_idxFree {m : Nat} (inB : Fin Γ → Buf) (out : Buf)
    (inIx : Fin Γ → IdxE)
    (p : VProg Γ m) (oix : Fin m → IdxE) (resReg : Nat)
    (hin : ∀ i, (inIx i).IregFree) (hout : ∀ i, (oix i).IregFree) :
    (compileVWKernel inB out inIx p oix resReg).IdxFree := by
  refine ⟨⟨loadSeqN_idxFree inB inIx hin Γ,
           compileTele_idxFree p.binds (fun k => .reg k.val) Γ⟩, ?_⟩
  have key : ∀ (L : List (Fin m)) (init : EWStmt), init.IdxFree →
      (L.foldl (fun s i => EWStmt.seq s
        (storeOut out p (fun k => .reg k.val) Γ oix resReg i)) init).IdxFree := by
    intro L
    induction L with
    | nil => intro init h; exact h
    | cons i L ih =>
        intro init h
        exact ih _ ⟨h, ⟨compileW_idxFree _ _ _, trivial⟩, hout i⟩
  exact key (List.finRange m) .skip trivial

/-- `storeOut` writes only `out`, and only where some lane of *its* output
    address lands. -/
theorem storeOut_frame {m : Nat} (out : Buf) (p : VProg Γ m) (ve : Fin Γ → WFExp)
    (c : Nat) (oix : Fin m → IdxE) (resReg : Nat) (i : Fin m) (st : WSt)
    (b : Buf) (A : Nat) (h : b ≠ out ∨ ∀ l : Lane, (oix i).eval 0 0 l ≠ A) :
    (((storeOut out p ve c oix resReg i).elabIn 0).run st).mem b A = st.mem b A := by
  show ((WStmt.storeLane out (fun l => (oix i).eval 0 0 l) resReg).run
          ((WStmt.setR resReg _).run
            (((compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1).elabIn 0
              |>.run st))).mem b A = _
  rcases h with hb | hl
  · rw [storeLane_otherBuf out _ resReg _ b hb, wrun_setR, WSt.mem_setReg]
    show (runW (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1 st).mem b A = _
    rw [compileW_mem]
  · by_cases hbo : b = out
    · subst hbo
      rw [storeLane_otherAddr _ _ resReg _ A hl, wrun_setR, WSt.mem_setReg]
      show (runW (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1 st).mem b A = _
      rw [compileW_mem]
    · rw [storeLane_otherBuf out _ resReg _ b hbo, wrun_setR, WSt.mem_setReg]
      show (runW (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1 st).mem b A = _
      rw [compileW_mem]

/-- What output `i` leaves at its own address, from the state the fold hands it. -/
theorem storeOut_at {m : Nat} (out : Buf) (p : VProg Γ m) (ve : Fin Γ → WFExp)
    (c : Nat) (oix : Fin m → IdxE) (resReg : Nat) (i : Fin m) (st : WSt) (l0 : Lane)
    (hagree : ∀ l, (oix i).eval 0 0 l = (oix i).eval 0 0 l0 →
      ((compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).2).eval
        (runW (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1 st) l
      = ((compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).2).eval
        (runW (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1 st) l0) :
    (((storeOut out p ve c oix resReg i).elabIn 0).run st).mem out
        ((oix i).eval 0 0 l0)
      = ((compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).2).eval
          (runW (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1 st) l0 := by
  show ((WStmt.storeLane out (fun l => (oix i).eval 0 0 l) resReg).run
          ((WStmt.setR resReg _).run
            (((compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1).elabIn 0
              |>.run st))).mem out _ = _
  rw [storeLane_at out (fun l => (oix i).eval 0 0 l) resReg _ l0
        (fun l h => by
          simp only [wrun_setR, WSt.regs_setReg_same]
          exact hagree l h),
      wrun_setR, WSt.regs_setReg_same]
  rfl

/-- **The fold over outputs preserves an address no later output touches.** -/
theorem storeOutFold_keeps {m : Nat} (out : Buf) (p : VProg Γ m) (ve : Fin Γ → WFExp)
    (c : Nat) (oix : Fin m → IdxE) (resReg : Nat) (A : Nat) :
    ∀ (L : List (Fin m)) (s : WSt),
      (∀ j ∈ L, ∀ l : Lane, (oix j).eval 0 0 l ≠ A) →
      ((L.foldl (fun s' j => ((storeOut out p ve c oix resReg j).elabIn 0).run s') s).mem
        out A) = s.mem out A := by
  intro L
  induction L with
  | nil => intro _ _; rfl
  | cons j L ih =>
      intro s hno
      rw [List.foldl_cons, ih _ (fun j' hj' => hno j' (List.mem_cons_of_mem j hj'))]
      exact storeOut_frame out p ve c oix resReg j s out A
        (Or.inr (hno j (List.mem_cons_self)))

/-- **Output `i` lands in memory at its own address.**

    The hypothesis the single-output path never needs: **outputs stored after
    `i` must not write `i`'s address**.  Without it a later output overwrites an
    earlier one, which no register-level theorem can detect.

    The list is given as `pre ++ i :: post` because that is the shape the
    argument takes — what `pre` wrote at this address is irrelevant, since `i`
    overwrites it, and only `post` can do damage. -/
theorem storeOutFold_at {m : Nat} (out : Buf) (p : VProg Γ m) (ve : Fin Γ → WFExp)
    (c : Nat) (oix : Fin m → IdxE) (resReg : Nat) (i : Fin m) (l0 : Lane)
    (pre post : List (Fin m)) (s : WSt)
    (hpost : ∀ j ∈ post, ∀ l : Lane, (oix j).eval 0 0 l ≠ (oix i).eval 0 0 l0)
    (hag : ∀ (s' : WSt) (l : Lane), (oix i).eval 0 0 l = (oix i).eval 0 0 l0 →
      ((compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).2).eval
        (runW (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1 s') l
      = ((compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).2).eval
        (runW (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1 s') l0) :
    (((pre ++ i :: post).foldl
        (fun s' j => ((storeOut out p ve c oix resReg j).elabIn 0).run s') s).mem
      out ((oix i).eval 0 0 l0))
      = ((compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).2).eval
          (runW (compileW (teleEnv ve c p.binds) (c + teleSlots p.binds) (p.outs i)).1
            (pre.foldl (fun s' j => ((storeOut out p ve c oix resReg j).elabIn 0).run s') s))
          l0 := by
  rw [List.foldl_append, List.foldl_cons,
      storeOutFold_keeps out p ve c oix resReg _ post _ hpost]
  exact storeOut_at out p ve c oix resReg i _ l0 (fun l h => hag _ l h)

/-- **A multi-output spec computes the right value — in a register.**

    Output `i`, evaluated against the once-emitted telescope, denotes `p.get i`
    under lane `l`'s environment.  Exact `Float32` equality.

    This is about the *value a lane computes*, not about memory: no store appears
    in it.  The memory-level statement is `compileVWKernel_stores` below, which
    needs a hypothesis this one does not — distinct outputs must write distinct
    addresses.  The single-output path (`compileWKernel_stores`) never needs
    that, because there is only one store. -/
theorem compileVWKernel_value {m : Nat} (inB : Fin Γ → Buf)
    (inIx : Fin Γ → IdxE)
    (p : VProg Γ m) (i : Fin m) (st : WSt) (l : Lane) :
    ((compileW (teleEnv (fun k => .reg k.val) Γ p.binds) (Γ + teleSlots p.binds)
        (p.outs i)).2).eval
        (runW (compileW (teleEnv (fun k => .reg k.val) Γ p.binds) (Γ + teleSlots p.binds)
                (p.outs i)).1
          (runW (compileTele (fun k => .reg k.val) Γ p.binds)
            (runW (loadSeq inB inIx) st))) l
      = denote (fun k => st.mem (inB k) ((inIx k).eval 0 0 l)) (p.get i) := by
  rw [compileVW_sound p i (fun k => .reg k.val) Γ (fun k => ⟨Nat.zero_le _, k.isLt⟩) _ l]
  congr 1
  funext k
  exact loadSeq_regs inB inIx st k l

/-- Running a left-nested `seq` chain is the same as folding the runs.

    `compileVWKernel` builds one statement by folding `seq` over the outputs;
    reasoning about it needs the fold of *executions*.  The two agree, but not
    definitionally. -/
theorem seqFold_run {α : Type} : ∀ (L : List α) (f : α → EWStmt) (init : EWStmt)
    (st : WSt),
    (((L.foldl (fun s a => EWStmt.seq s (f a)) init).elabIn 0).run st)
      = L.foldl (fun s' a => ((f a).elabIn 0).run s') ((init.elabIn 0).run st) := by
  intro L
  induction L with
  | nil => intro _ _ _; rfl
  | cons a L ih =>
      intro f init st
      rw [List.foldl_cons, List.foldl_cons, ih f (EWStmt.seq init (f a)) st]
      rfl

/-- **The multi-output kernel stores output `i` where it should.**

    `compileVWKernel_correct` says what a lane *computes*; this says where it
    *lands*.

    `hsplit` supplies the position of `i` in the emission order, and `hpost` the
    obligation that matters: nothing emitted after `i` writes `i`'s address.  For
    the usual addressing — output `k` at `base + k` — that is arithmetic the
    caller discharges once. -/
theorem compileVWKernel_stores {m : Nat} (inB : Fin Γ → Buf) (out : Buf)
    (inIx : Fin Γ → IdxE) (p : VProg Γ m) (oix : Fin m → IdxE) (resReg : Nat)
    (i : Fin m) (st : WSt) (l0 : Lane) (pre post : List (Fin m))
    (hsplit : List.finRange m = pre ++ i :: post)
    (hpost : ∀ j ∈ post, ∀ l : Lane, (oix j).eval 0 0 l ≠ (oix i).eval 0 0 l0)
    (hag : ∀ (s' : WSt) (l : Lane), (oix i).eval 0 0 l = (oix i).eval 0 0 l0 →
      ((compileW (teleEnv (fun k => .reg k.val) Γ p.binds) (Γ + teleSlots p.binds)
          (p.outs i)).2).eval
        (runW (compileW (teleEnv (fun k => .reg k.val) Γ p.binds) (Γ + teleSlots p.binds)
          (p.outs i)).1 s') l
      = ((compileW (teleEnv (fun k => .reg k.val) Γ p.binds) (Γ + teleSlots p.binds)
          (p.outs i)).2).eval
        (runW (compileW (teleEnv (fun k => .reg k.val) Γ p.binds) (Γ + teleSlots p.binds)
          (p.outs i)).1 s') l0) :
    ∃ s0 : WSt,
      (((compileVWKernel inB out inIx p oix resReg).elabIn 0).run st).mem out
          ((oix i).eval 0 0 l0)
        = ((compileW (teleEnv (fun k => .reg k.val) Γ p.binds) (Γ + teleSlots p.binds)
            (p.outs i)).2).eval
            (runW (compileW (teleEnv (fun k => .reg k.val) Γ p.binds)
              (Γ + teleSlots p.binds) (p.outs i)).1 s0) l0 := by
  refine ⟨pre.foldl (fun s' j => ((storeOut out p (fun k => .reg k.val) Γ oix resReg j).elabIn 0).run s')
    ((compileTele (fun k => .reg k.val) Γ p.binds).elabIn 0 |>.run
      ((loadSeq inB inIx).elabIn 0 |>.run st)), ?_⟩
  show ((((List.finRange m).foldl
      (fun s a => EWStmt.seq s (storeOut out p (fun k => .reg k.val) Γ oix resReg a))
      .skip).elabIn 0).run
        (((compileTele (fun k => .reg k.val) Γ p.binds).elabIn 0).run
          (((loadSeq inB inIx).elabIn 0).run st))).mem out _ = _
  rw [seqFold_run (List.finRange m) _ .skip, hsplit]
  exact storeOutFold_at out p (fun k => .reg k.val) Γ oix resReg i l0 pre post _ hpost hag

end AlgorithmLib.ML
