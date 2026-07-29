import AlgorithmLib.ML.PtxM

namespace AlgorithmLib.ML

-- ---------------------------------------------------------------------------
-- The flat machine
-- ---------------------------------------------------------------------------

/-- A flat instruction: a machine instruction, `bra`, or `@%p bra`. -/
inductive FI where
  | si    : SI → FI
  | jmp   : Nat → FI
  | jmpIf : Nat → Nat → FI

/-- A configuration: program counter and machine state. -/
abbrev Cfg := Nat × MState

/-- Warp-uniform truth of a predicate. -/
def allLanes (b : Lane → Bool) : Bool := (List.finRange W).all b

/-- One program-counter transition.  Falling off the end, or a *divergent*
    predicated branch, is stuck — neither is silently approximated. -/
def fstep (cta : Nat) (P : List FI) (c : Cfg) : Option Cfg :=
  match P[c.1]? with
  | none => none
  | some (.si i) => some (c.1 + 1, SI.step cta i c.2)
  | some (.jmp t) => some (t, c.2)
  | some (.jmpIf p t) =>
      if allLanes (fun l => c.2.pr p l) then some (t, c.2)
      else if allLanes (fun l => !c.2.pr p l) then some (c.1 + 1, c.2)
      else none

def steps (cta : Nat) (P : List FI) : Nat → Cfg → Option Cfg
  | 0, c => some c
  | k + 1, c => match fstep cta P c with
                | none => none
                | some c' => steps cta P k c'

theorem steps_add (cta : Nat) (P : List FI) : ∀ (a b : Nat) (c : Cfg),
    steps cta P (a + b) c
      = match steps cta P a c with
        | none => none
        | some c' => steps cta P b c' := by
  intro a
  induction a with
  | zero => intro b c; rw [Nat.zero_add]; simp [steps]
  | succ a ih =>
      intro b c
      rw [Nat.succ_add]
      cases h : fstep cta P c with
      | none => simp [steps, h]
      | some c' => simp [steps, h, ih]

/-- Chaining: `a` steps then `b` steps. -/
theorem steps_trans (cta : Nat) (P : List FI) (a b : Nat) (c c' c'' : Cfg)
    (h1 : steps cta P a c = some c') (h2 : steps cta P b c' = some c'') :
    steps cta P (a + b) c = some c'' := by
  rw [steps_add, h1]
  exact h2

-- ---------------------------------------------------------------------------
-- Flattening
-- ---------------------------------------------------------------------------

/-- Instruction count of the flattened statement — independent of where it is
    placed, which is why `flatEW` can compute absolute branch targets. -/
def flenEW (lr n : Nat) : EWStmt → Nat
  | .seq a b => flenEW lr n a + flenEW lr n b
  | .forN _ body => flenEW n (n + 1) body + 5
  -- Two extra instructions materialise the bound: `mov` the address, then
  -- `ld.global.u32`.  All lanes load the same address, which is what makes the
  -- guard register lane-uniform without assuming anything.
  | .forM _ _ body => flenEW n (n + 3) body + 7
  | s => (emitEW lr n s).length

/-- Compile to flat, branch-based code placed at absolute position `base`.

    A counted loop becomes exactly the five-part shape a hand-written kernel
    has: initialise, test, exit-branch, body, increment, back-edge. -/
def flatEW (lr n base : Nat) : EWStmt → List FI
  | .seq a b => flatEW lr n base a ++ flatEW lr n (base + flenEW lr n a) b
  | .forN cnt body =>
      FI.si (.movIC n 0) :: FI.si (.setpGeC n n cnt)
        :: FI.jmpIf n (base + (flenEW n (n + 1) body + 5))
        :: (flatEW n (n + 1) (base + 3) body
              ++ [FI.si (.addRC n n 1), FI.jmp (base + 1)])
  | .forM bu ad body =>
      FI.si (.movIC (n + 1) ad) :: FI.si (.ldGI (n + 2) bu (n + 1))
        :: FI.si (.movIC n 0) :: FI.si (.setpGeR n n (n + 2))
        :: FI.jmpIf n (base + (flenEW n (n + 3) body + 7))
        :: (flatEW n (n + 3) (base + 5) body
              ++ [FI.si (.addRC n n 1), FI.jmp (base + 3)])
  -- Identical shape; only the guard differs — an immediate becomes a register.
  | s => (emitEW lr n s).map FI.si

theorem flatEW_length : ∀ (s : EWStmt) (lr n base : Nat),
    (flatEW lr n base s).length = flenEW lr n s := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro lr n base
      show ((flatEW lr n base a) ++ (flatEW lr n (base + flenEW lr n a) b)).length = _
      rw [List.length_append, iha, ihb]
      rfl
  | forN cnt body ih =>
      intro lr n base
      show (_ :: _ :: _ :: (flatEW n (n + 1) (base + 3) body ++ [_, _])).length = _
      simp only [List.length_cons, List.length_append, ih]
      show flenEW n (n + 1) body + 2 + 1 + 1 + 1 = flenEW n (n + 1) body + 5
      omega
  | forM bu ad body ih =>
      intro lr n base
      show (_ :: _ :: _ :: _ :: _ :: (flatEW n (n + 3) (base + 5) body ++ [_, _])).length = _
      simp only [List.length_cons, List.length_append, ih]
      show flenEW n (n + 3) body + 2 + 1 + 1 + 1 + 1 + 1 = flenEW n (n + 3) body + 7
      omega
  | skip => intro lr n base; simp [flatEW, flenEW]
  | setR _ _ => intro lr n base; simp [flatEW, flenEW]
  | shflXor _ _ _ => intro lr n base; simp [flatEW, flenEW]
  | loadIdx _ _ _ => intro lr n base; simp [flatEW, flenEW]
  | loadV4 _ _ _ _ _ _ => intro lr n base; simp [flatEW, flenEW]
  | storeLane0 _ _ _ => intro lr n base; simp [flatEW, flenEW]
  | storeLane _ _ _ => intro lr n base; simp [flatEW, flenEW]
  | stSm _ _ => intro lr n base; simp [flatEW, flenEW]
  | ldSm _ _ => intro lr n base; simp [flatEW, flenEW]
  | barrier => intro lr n base; simp [flatEW, flenEW]
  | cvtIF _ _ => intro lr n base; simp [flatEW, flenEW]

-- ---------------------------------------------------------------------------
-- Straight-line code runs
-- ---------------------------------------------------------------------------

/-- A run of ordinary instructions advances the program counter past them and
    performs exactly the structured semantics. -/
theorem steps_si (cta : Nat) (P : List FI) : ∀ (code : List SI) (base : Nat) (m : MState),
    (∀ j, j < code.length → P[base + j]? = (code.map FI.si)[j]?) →
    steps cta P code.length (base, m) = some (base + code.length, SI.stepL cta code m) := by
  intro code
  induction code with
  | nil => intro base m _; rfl
  | cons i code ih =>
      intro base m h
      have h0 : P[base]? = some (FI.si i) := by
        have := h 0 (by simp)
        simpa using this
      show (match fstep cta P (base, m) with
            | none => none
            | some c' => steps cta P code.length c') = _
      have hf : fstep cta P (base, m) = some (base + 1, SI.step cta i m) := by
        show (match P[base]? with
              | none => none
              | some (.si i) => some (base + 1, SI.step cta i m)
              | some (.jmp t) => some (t, m)
              | some (.jmpIf p t) => _) = _
        rw [h0]
      rw [hf]
      have h' : ∀ j, j < code.length → P[(base + 1) + j]? = (code.map FI.si)[j]? := by
        intro j hj
        have := h (j + 1) (by simp; omega)
        rw [show base + (j + 1) = base + 1 + j by omega] at this
        simpa using this
      show steps cta P code.length (base + 1, SI.step cta i m) = _
      rw [ih (base + 1) (SI.step cta i m) h',
          show base + 1 + code.length = base + (i :: code).length from by
            simp [List.length_cons]; omega]
      rfl

-- ---------------------------------------------------------------------------
-- Soundness of the flat lowering
-- ---------------------------------------------------------------------------

/-- The straight-line constructors, discharged once. -/
theorem flatEW_leaf (cta : Nat) (s : EWStmt) (hf : s.ExpFree) (lr n base i : Nat)
    (P : List FI) (m : MState)
    (hcode : flatEW lr n base s = (emitEW lr n s).map FI.si)
    (hlen : flenEW lr n s = (emitEW lr n s).length)
    (h2 : 2 ≤ n) (hlr : lr < n) (hb : s.IdxBelow n) (hfl : s.Flat) (hinv : MInv cta i lr m)
    (hseg : ∀ j, j < flenEW lr n s → P[base + j]? = (flatEW lr n base s)[j]?) :
    ∃ k m', steps cta P k (base, m) = some (base + flenEW lr n s, m')
      ∧ m'.toWSt = (s.elabAt cta i m.ir m.imem).run m.toWSt
      ∧ (∀ x, x < n → m'.ir x = m.ir x) ∧ m'.imem = m.imem := by
  refine ⟨(emitEW lr n s).length, SI.stepL cta (emitEW lr n s) m, ?_, ?_, ?_, ?_⟩
  · rw [hlen]
    refine steps_si cta P (emitEW lr n s) base m ?_
    intro j hj
    rw [← hcode]
    exact hseg j (by rw [hlen]; exact hj)
  · exact emitEW_sound cta s hf lr n i m h2 hlr hb hinv
  · intro x hx; exact emitEW_frame cta s lr n m x hx
  · exact emitEW_imem cta s lr n m

/-- **The full lowering is correct: a real branch-based program, executed by a
    program counter, runs the statement.**

    Every construct is covered — address arithmetic, scalar and vectorised
    loads, per-lane and predicated stores, shared memory, externs, and *counted
    loops compiled to labels and branches*.  Nothing about the emitted control
    flow is assumed: the guard, the exit branch, the increment and the back edge
    are all executed by `fstep`, and the loop's termination is a consequence of
    the proof rather than a hypothesis.

    `hexp` is the single declared approximation.  Everything else is exact. -/
theorem flatEW_sound (cta : Nat) :
    ∀ (s : EWStmt), s.ExpFree → ∀ (lr n base i : Nat) (P : List FI) (m : MState),
      2 ≤ n → lr < n → s.IdxBelow n → s.Flat → MInv cta i lr m →
      (∀ j, j < flenEW lr n s → P[base + j]? = (flatEW lr n base s)[j]?) →
      ∃ k m', steps cta P k (base, m) = some (base + flenEW lr n s, m')
        ∧ m'.toWSt = (s.elabAt cta i m.ir m.imem).run m.toWSt
        ∧ (∀ x, x < n → m'.ir x = m.ir x) ∧ m'.imem = m.imem := by
  intro s
  induction s with
  | skip => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
            exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | setR _ _ => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
                exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | shflXor _ _ _ => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
                     exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | loadIdx _ _ _ => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
                     exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | loadV4 _ _ _ _ _ _ => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
                          exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | storeLane0 _ _ _ => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
                        exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | storeLane _ _ _ => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
                       exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | stSm _ _ => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
                exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | ldSm _ _ => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
                exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | barrier => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
               exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | cvtIF _ _ => intro hf lr n base i P m h2 hlr hb hfl hinv hseg
                 exact flatEW_leaf cta _ hf lr n base i P m rfl rfl h2 hlr hb hfl hinv hseg
  | forM bu ad body ih =>
      intro hf lr n base i P m h2 hlr hb hfl hinv hseg
      -- The bound is read from **integer memory**, which carries no lane index,
      -- so the guard register is lane-uniform by construction and the exit
      -- branch is never divergent.  Nothing about uniformity is assumed here.
      have hL : (flatEW n (n + 3) (base + 5) body).length = flenEW n (n + 3) body :=
        flatEW_length body n (n + 3) (base + 5)
      have hflen : flenEW lr n (EWStmt.forM bu ad body) = flenEW n (n + 3) body + 7 := rfl
      -- instruction lookups
      have hP0 : P[base]? = some (FI.si (SI.movIC (n + 1) ad)) := by
        have := hseg 0 (by rw [hflen]; omega); simpa using this
      have hP1 : P[base + 1]? = some (FI.si (SI.ldGI (n + 2) bu (n + 1))) := by
        have := hseg 1 (by rw [hflen]; omega); simpa using this
      have hP2 : P[base + 2]? = some (FI.si (SI.movIC n 0)) := by
        have := hseg 2 (by rw [hflen]; omega); simpa using this
      have hP3 : P[base + 3]? = some (FI.si (SI.setpGeR n n (n + 2))) := by
        have := hseg 3 (by rw [hflen]; omega); simpa using this
      have hP4 : P[base + 4]?
          = some (FI.jmpIf n (base + (flenEW n (n + 3) body + 7))) := by
        have := hseg 4 (by rw [hflen]; omega); simpa using this
      have hPb : ∀ j, j < flenEW n (n + 3) body →
          P[(base + 5) + j]? = (flatEW n (n + 3) (base + 5) body)[j]? := by
        intro j hj
        have := hseg (j + 5) (by rw [hflen]; omega)
        rw [show base + (j + 5) = base + 5 + j by omega] at this
        rw [this]
        show (FI.si _ :: FI.si _ :: FI.si _ :: FI.si _ :: FI.jmpIf _ _
                :: (flatEW n (n + 3) (base + 5) body ++ [_, _]))[j + 5]? = _
        show (flatEW n (n + 3) (base + 5) body ++ [FI.si (SI.addRC n n 1),
                FI.jmp (base + 3)])[j]? = _
        exact List.getElem?_append_left (by omega)
      have hPinc : P[base + (flenEW n (n + 3) body + 5)]?
          = some (FI.si (SI.addRC n n 1)) := by
        have := hseg (flenEW n (n + 3) body + 5) (by rw [hflen]; omega)
        rw [this]
        show (FI.si _ :: FI.si _ :: FI.si _ :: FI.si _ :: FI.jmpIf _ _
                :: (flatEW n (n + 3) (base + 5) body
                      ++ [_, _]))[flenEW n (n + 3) body + 5]? = _
        show (flatEW n (n + 3) (base + 5) body ++ [FI.si (SI.addRC n n 1),
                FI.jmp (base + 3)])[flenEW n (n + 3) body]? = _
        rw [List.getElem?_append_right (by omega), hL]
        simp
      have hPback : P[base + (flenEW n (n + 3) body + 6)]?
          = some (FI.jmp (base + 3)) := by
        have := hseg (flenEW n (n + 3) body + 6) (by rw [hflen]; omega)
        rw [this]
        show (FI.si _ :: FI.si _ :: FI.si _ :: FI.si _ :: FI.jmpIf _ _
                :: (flatEW n (n + 3) (base + 5) body
                      ++ [_, _]))[flenEW n (n + 3) body + 6]? = _
        show (flatEW n (n + 3) (base + 5) body ++ [FI.si (SI.addRC n n 1),
                FI.jmp (base + 3)])[flenEW n (n + 3) body + 1]? = _
        rw [List.getElem?_append_right (by omega), hL]
        simp
      -- the loop invariant, by induction on the remaining trip count.  The
      -- bound register `n+2` is *above* the body's watermark only in the sense
      -- that the body allocates from `n+3`: `emitEW_frame` therefore preserves
      -- it, which is why the guard still reads the right value on every trip.
      have key : ∀ (d j : Nat), j + d = m.imem bu ad → ∀ (m' : MState),
          m'.ir n = (fun _ => j) → m'.ir (n + 2) = (fun _ => m.imem bu ad) →
          m'.ir laneIR = (fun l => (l.val : Nat)) →
          m'.ir ctaIR = (fun _ => cta) → (∀ x, x < n → m'.ir x = m.ir x) →
          m'.imem = m.imem →
          ∃ k m'', steps cta P k (base + 3, m')
                = some (base + (flenEW n (n + 3) body + 7), m'')
            ∧ m''.toWSt
                = (List.range' j d).foldl
                    (fun s' t => (body.elabAt cta t m.ir m.imem).run s') m'.toWSt
            ∧ (∀ x, x < n → m''.ir x = m'.ir x) ∧ m''.imem = m'.imem := by
        intro d
        induction d with
        | zero =>
            intro j hj m' hn hcn h0 h1 hpres hmpres
            have hjc : m.imem bu ad ≤ j := by omega
            refine ⟨2, m'.setPr n (fun l => decide (m'.ir (n + 2) l ≤ m'.ir n l)),
                    ?_, rfl, ?_, rfl⟩
            · show (match fstep cta P (base + 3, m') with
                    | none => none
                    | some c' => steps cta P 1 c') = _
              have hf1 : fstep cta P (base + 3, m')
                  = some (base + 4,
                      m'.setPr n (fun l => decide (m'.ir (n + 2) l ≤ m'.ir n l))) := by
                show (match P[base + 3]? with
                      | none => none
                      | some (.si i) => some (base + 3 + 1, SI.step cta i m')
                      | some (.jmp t) => some (t, m')
                      | some (.jmpIf p t) => _) = _
                rw [hP3]
                show some (base + 3 + 1, _) = _
                rw [show base + 3 + 1 = base + 4 by omega]
                rfl
              rw [hf1]
              show (match fstep cta P (base + 4, _) with
                    | none => none
                    | some c' => steps cta P 0 c') = _
              have hall : allLanes (fun l =>
                  (m'.setPr n (fun l =>
                    decide (m'.ir (n + 2) l ≤ m'.ir n l))).pr n l) = true := by
                rw [MState.pr_setPr_same]
                refine List.all_eq_true.mpr ?_
                intro l _
                simp only [hn, hcn]
                exact decide_eq_true hjc
              have hf2 : fstep cta P (base + 4,
                  m'.setPr n (fun l => decide (m'.ir (n + 2) l ≤ m'.ir n l)))
                  = some (base + (flenEW n (n + 3) body + 7),
                      m'.setPr n (fun l => decide (m'.ir (n + 2) l ≤ m'.ir n l))) := by
                show (match P[base + 4]? with
                      | none => none
                      | some (.si i) => _
                      | some (.jmp t) => _
                      | some (.jmpIf p t) =>
                          if allLanes (fun l => _) then some (t, _)
                          else if allLanes (fun l => !_) then _ else none) = _
                rw [hP4]
                show (if allLanes (fun l =>
                        (m'.setPr n (fun l =>
                          decide (m'.ir (n + 2) l ≤ m'.ir n l))).pr n l)
                      then _ else _) = _
                rw [hall]
                rfl
              rw [hf2]
              rfl
            · intro x hx; rfl
        | succ d ihd =>
            intro j hj m' hn hcn h0 h1 hpres hmpres
            have hjlt : j < m.imem bu ad := by omega
            -- guard test
            have hf1 : fstep cta P (base + 3, m')
                = some (base + 4,
                    m'.setPr n (fun l => decide (m'.ir (n + 2) l ≤ m'.ir n l))) := by
              show (match P[base + 3]? with
                    | none => none
                    | some (.si i) => some (base + 3 + 1, SI.step cta i m')
                    | some (.jmp t) => some (t, m')
                    | some (.jmpIf p t) => _) = _
              rw [hP3]
              show some (base + 3 + 1, _) = _
              rw [show base + 3 + 1 = base + 4 by omega]
              rfl
            let m1 := m'.setPr n (fun l => decide (m'.ir (n + 2) l ≤ m'.ir n l))
            have hnotall : allLanes (fun l => m1.pr n l) = false := by
              show allLanes (fun l => (m'.setPr n _).pr n l) = false
              rw [MState.pr_setPr_same]
              have hne : ¬ (allLanes
                  (fun l => decide (m'.ir (n + 2) l ≤ m'.ir n l)) = true) := by
                intro hc
                have := List.all_eq_true.mp hc ⟨0, by decide⟩ (by simp)
                simp only [hn, hcn] at this
                have := of_decide_eq_true this
                omega
              cases hbb : allLanes (fun l => decide (m'.ir (n + 2) l ≤ m'.ir n l)) with
              | false => rfl
              | true => exact absurd hbb hne
            have hallf : allLanes (fun l => !m1.pr n l) = true := by
              show allLanes (fun l => !(m'.setPr n _).pr n l) = true
              rw [MState.pr_setPr_same]
              refine List.all_eq_true.mpr ?_
              intro l _
              simp only [hn, hcn]
              simp only [Bool.not_eq_true', decide_eq_false_iff_not]
              omega
            have hf2 : fstep cta P (base + 4, m1) = some (base + 5, m1) := by
              show (match P[base + 4]? with
                    | none => none
                    | some (.si i) => _
                    | some (.jmp t) => _
                    | some (.jmpIf p t) =>
                        if allLanes (fun l => _) then some (t, _)
                        else if allLanes (fun l => !_) then some (base + 4 + 1, _)
                        else none) = _
              rw [hP4]
              show (if allLanes (fun l => m1.pr n l) then _
                    else if allLanes (fun l => !m1.pr n l) then some (base + 4 + 1, m1)
                    else none) = _
              rw [hnotall, hallf]
              show some (base + 4 + 1, m1) = _
              rw [show base + 4 + 1 = base + 5 by omega]
            -- body
            have hinvb : MInv cta j n m1 := by
              refine ⟨?_, ?_, ?_⟩
              · show (m'.setPr n _).ir laneIR = _; rw [MState.ir_setPr]; exact h0
              · show (m'.setPr n _).ir ctaIR = _; rw [MState.ir_setPr]; exact h1
              · show (m'.setPr n _).ir n = _; rw [MState.ir_setPr]; exact hn
            obtain ⟨kb, m2, hsb, hwb, hfb, hm2i⟩ :=
              ih hf n (n + 3) (base + 5) j P m1 (by omega) (by omega)
                (EWStmt.idxBelow_mono (by omega) body hb) hfl hinvb hPb
            -- the counter and the bound both survive the body
            have hm2n : m2.ir n = (fun _ => j) := by
              rw [hfb n (by omega)]
              show (m'.setPr n _).ir n = _
              rw [MState.ir_setPr]; exact hn
            have hm2c : m2.ir (n + 2) = (fun _ => m.imem bu ad) := by
              rw [hfb (n + 2) (by omega)]
              show (m'.setPr n _).ir (n + 2) = _
              rw [MState.ir_setPr]; exact hcn
            -- increment and back edge
            have hf3 : fstep cta P (base + 5 + flenEW n (n + 3) body, m2)
                = some (base + (flenEW n (n + 3) body + 6),
                    m2.setI n (fun l => m2.ir n l + 1)) := by
              show (match P[base + 5 + flenEW n (n + 3) body]? with
                    | none => none
                    | some (.si i) => some (base + 5 + flenEW n (n + 3) body + 1,
                        SI.step cta i m2)
                    | some (.jmp t) => _
                    | some (.jmpIf p t) => _) = _
              rw [show base + 5 + flenEW n (n + 3) body
                    = base + (flenEW n (n + 3) body + 5) by omega, hPinc]
              show some (base + (flenEW n (n + 3) body + 5) + 1, _) = _
              rw [show base + (flenEW n (n + 3) body + 5) + 1
                    = base + (flenEW n (n + 3) body + 6) by omega]
              rfl
            have hf4 : fstep cta P (base + (flenEW n (n + 3) body + 6),
                  m2.setI n (fun l => m2.ir n l + 1))
                = some (base + 3, m2.setI n (fun l => m2.ir n l + 1)) := by
              show (match P[base + (flenEW n (n + 3) body + 6)]? with
                    | none => none
                    | some (.si i) => _
                    | some (.jmp t) => some (t, _)
                    | some (.jmpIf p t) => _) = _
              rw [hPback]
            -- next iteration
            have hm3n : (m2.setI n (fun l => m2.ir n l + 1)).ir n = (fun _ => j + 1) := by
              rw [MState.ir_setI_same, hm2n]
            obtain ⟨kr, m4, hsr, hwr, hfr, hm4i⟩ := ihd (j + 1) (by omega)
              (m2.setI n (fun l => m2.ir n l + 1)) hm3n
              (by rw [MState.ir_setI_other _ _ _ _ (show n + 2 ≠ n by omega)]; exact hm2c)
              (by rw [MState.ir_setI_other _ _ _ _ (show 0 ≠ n by omega),
                      hfb laneIR (show 0 < n + 3 by omega)]
                  show (m'.setPr n _).ir laneIR = _
                  rw [MState.ir_setPr]; exact h0)
              (by rw [MState.ir_setI_other _ _ _ _ (show 1 ≠ n by omega),
                      hfb ctaIR (show 1 < n + 3 by omega)]
                  show (m'.setPr n _).ir ctaIR = _
                  rw [MState.ir_setPr]; exact h1)
              (by intro x hx
                  rw [MState.ir_setI_other _ _ _ _ (show x ≠ n by omega),
                      hfb x (show x < n + 3 by omega)]
                  show (m'.setPr n _).ir x = _
                  rw [MState.ir_setPr]; exact hpres x hx)
              (by rw [show (m2.setI n (fun l => m2.ir n l + 1)).imem = m2.imem from rfl,
                      hm2i, show m1.imem = m'.imem from rfl, hmpres])
            refine ⟨2 + (kb + (2 + kr)), m4, ?_, ?_, ?_, ?_⟩
            · refine steps_trans cta P 2 (kb + (2 + kr)) _ (base + 5, m1) _ ?_ ?_
              · show (match fstep cta P (base + 3, m') with
                      | none => none
                      | some c' => steps cta P 1 c') = _
                rw [hf1]
                show (match fstep cta P (base + 4, m1) with
                      | none => none
                      | some c' => steps cta P 0 c') = _
                rw [hf2]
                rfl
              · refine steps_trans cta P kb (2 + kr) _
                  (base + 5 + flenEW n (n + 3) body, m2) _ hsb ?_
                refine steps_trans cta P 2 kr _
                  (base + 3, m2.setI n (fun l => m2.ir n l + 1)) _ ?_ hsr
                show (match fstep cta P (base + 5 + flenEW n (n + 3) body, m2) with
                      | none => none
                      | some c' => steps cta P 1 c') = _
                rw [hf3]
                show (match fstep cta P (base + (flenEW n (n + 3) body + 6), _) with
                      | none => none
                      | some c' => steps cta P 0 c') = _
                rw [hf4]
                rfl
            · rw [hwr]
              show _ = (List.range' j (d + 1)).foldl _ m'.toWSt
              rw [List.range'_succ, List.foldl_cons]
              have hts : (m2.setI n (fun l => m2.ir n l + 1)).toWSt = m2.toWSt := rfl
              have hbr : body.elabAt cta j m1.ir m1.imem
                  = body.elabAt cta j m.ir m.imem := by
                rw [show m1.imem = m'.imem from rfl, hmpres]
                exact EWStmt.elabAt_frame cta n m.ir _ m.imem
                  (fun x hx => by
                    show (m'.setPr n _).ir x = _
                    rw [MState.ir_setPr]; exact hpres x hx) body j hb
              rw [hts, hwb, hbr]
              rfl
            · intro x hx
              rw [hfr x hx, MState.ir_setI_other _ _ _ _ (by omega), hfb x (by omega)]
              show (m'.setPr n _).ir x = _
              rw [MState.ir_setPr]
            · rw [hm4i, show (m2.setI n (fun l => m2.ir n l + 1)).imem = m2.imem from rfl,
                  hm2i]
              rfl
      -- materialise the bound, initialise the counter, then run the loop
      have e0 : ∀ mm : MState, fstep cta P (base, mm)
          = some (base + 1, mm.setI (n + 1) (fun _ => ad)) := by
        intro mm
        show (match P[base]? with
              | none => none
              | some (.si i) => some (base + 1, SI.step cta i mm)
              | some (.jmp t) => _
              | some (.jmpIf p t) => _) = _
        rw [hP0]
        rfl
      have e1 : ∀ mm : MState, fstep cta P (base + 1, mm)
          = some (base + 2, mm.setI (n + 2)
              (fun l => mm.imem bu (mm.ir (n + 1) l))) := by
        intro mm
        show (match P[base + 1]? with
              | none => none
              | some (.si i) => some (base + 1 + 1, SI.step cta i mm)
              | some (.jmp t) => _
              | some (.jmpIf p t) => _) = _
        rw [hP1]
        show some (base + 1 + 1, _) = _
        rw [show base + 1 + 1 = base + 2 by omega]
        rfl
      have e2 : ∀ mm : MState, fstep cta P (base + 2, mm)
          = some (base + 3, mm.setI n (fun _ => 0)) := by
        intro mm
        show (match P[base + 2]? with
              | none => none
              | some (.si i) => some (base + 2 + 1, SI.step cta i mm)
              | some (.jmp t) => _
              | some (.jmpIf p t) => _) = _
        rw [hP2]
        show some (base + 2 + 1, _) = _
        rw [show base + 2 + 1 = base + 3 by omega]
        rfl
      obtain ⟨mc, hmcn, hmcc, hmc0, hmc1, hmcpres, hmcim, hmcws, hstep3⟩ :
          ∃ mc : MState, mc.ir n = (fun _ => 0)
            ∧ mc.ir (n + 2) = (fun _ => m.imem bu ad)
            ∧ mc.ir laneIR = (fun l => (l.val : Nat))
            ∧ mc.ir ctaIR = (fun _ => cta)
            ∧ (∀ x, x < n → mc.ir x = m.ir x)
            ∧ mc.imem = m.imem ∧ mc.toWSt = m.toWSt
            ∧ steps cta P 3 (base, m) = some (base + 3, mc) := by
        refine ⟨((m.setI (n + 1) (fun _ => ad)).setI (n + 2)
                  (fun l => (m.setI (n + 1) (fun _ => ad)).imem bu
                    ((m.setI (n + 1) (fun _ => ad)).ir (n + 1) l))).setI n (fun _ => 0),
                MState.ir_setI_same _ _ _, ?_, ?_, ?_, ?_, rfl, rfl, ?_⟩
        · rw [MState.ir_setI_other _ _ _ _ (show n + 2 ≠ n by omega),
              MState.ir_setI_same]
          funext l
          rw [MState.ir_setI_same]
          rfl
        · rw [MState.ir_setI_other _ _ _ _ (show 0 ≠ n by omega),
              MState.ir_setI_other _ _ _ _ (show 0 ≠ n + 2 by omega),
              MState.ir_setI_other _ _ _ _ (show 0 ≠ n + 1 by omega)]
          exact hinv.1
        · rw [MState.ir_setI_other _ _ _ _ (show 1 ≠ n by omega),
              MState.ir_setI_other _ _ _ _ (show 1 ≠ n + 2 by omega),
              MState.ir_setI_other _ _ _ _ (show 1 ≠ n + 1 by omega)]
          exact hinv.2.1
        · intro x hx
          rw [MState.ir_setI_other _ _ _ _ (show x ≠ n by omega),
              MState.ir_setI_other _ _ _ _ (show x ≠ n + 2 by omega),
              MState.ir_setI_other _ _ _ _ (show x ≠ n + 1 by omega)]
        · show (match fstep cta P (base, m) with
                | none => none
                | some c' => steps cta P 2 c') = _
          rw [e0 m]
          show (match fstep cta P (base + 1, _) with
                | none => none
                | some c' => steps cta P 1 c') = _
          rw [e1 _]
          show (match fstep cta P (base + 2, _) with
                | none => none
                | some c' => steps cta P 0 c') = _
          rw [e2 _]
          rfl
      obtain ⟨k, m'', hs, hw, hfk, hmi⟩ :=
        key (m.imem bu ad) 0 (by omega) mc hmcn hmcc hmc0 hmc1 hmcpres hmcim
      refine ⟨3 + k, m'', steps_trans cta P 3 k _ (base + 3, mc) _ hstep3 hs, ?_, ?_, ?_⟩
      · rw [hw, hmcws]
        show (List.range' 0 (m.imem bu ad)).foldl _ m.toWSt = _
        rw [← List.range_eq_range']
        rfl
      · intro x hx; rw [hfk x hx]; exact hmcpres x hx
      · rw [hmi, hmcim]
  | seq a b iha ihb =>
      intro hf lr n base i P m h2 hlr hb hfl hinv hseg
      have hla : (flatEW lr n base a).length = flenEW lr n a := flatEW_length a lr n base
      have hsa : ∀ j, j < flenEW lr n a → P[base + j]? = (flatEW lr n base a)[j]? := by
        intro j hj
        have := hseg j (by show j < flenEW lr n a + flenEW lr n b; omega)
        rw [this]
        show (flatEW lr n base a ++ flatEW lr n (base + flenEW lr n a) b)[j]? = _
        exact List.getElem?_append_left (by omega)
      obtain ⟨k1, m1, hs1, hw1, hf1, hm1i⟩ := iha hf.1 lr n base i P m h2 hlr hb.1 hfl.1 hinv hsa
      have hinv1 : MInv cta i lr m1 :=
        ⟨by rw [hf1 laneIR (show 0 < n by omega)]; exact hinv.1,
         by rw [hf1 ctaIR (show 1 < n by omega)]; exact hinv.2.1,
         by rw [hf1 lr hlr]; exact hinv.2.2⟩
      have hsb : ∀ j, j < flenEW lr n b →
          P[(base + flenEW lr n a) + j]? = (flatEW lr n (base + flenEW lr n a) b)[j]? := by
        intro j hj
        have hj' := hseg (flenEW lr n a + j)
          (by show flenEW lr n a + j < flenEW lr n a + flenEW lr n b; omega)
        rw [show base + flenEW lr n a + j = base + (flenEW lr n a + j) by omega, hj']
        show (flatEW lr n base a ++ flatEW lr n (base + flenEW lr n a) b)[flenEW lr n a + j]? = _
        rw [List.getElem?_append_right (by omega)]
        congr 1
        omega
      obtain ⟨k2, m2, hs2, hw2, hf2, hm2i⟩ :=
        ihb hf.2 lr n (base + flenEW lr n a) i P m1 h2 hlr hb.2 hfl.2 hinv1 hsb
      refine ⟨k1 + k2, m2, ?_, ?_, ?_, ?_⟩
      · rw [steps_trans cta P k1 k2 _ _ _ hs1 hs2]
        congr 2
        show base + flenEW lr n a + flenEW lr n b = base + flenEW lr n (a.seq b)
        show base + flenEW lr n a + flenEW lr n b = base + (flenEW lr n a + flenEW lr n b)
        omega
      · have hbr : b.elabAt cta i m1.ir m1.imem = b.elabAt cta i m.ir m.imem := by
          rw [hm1i]
          exact EWStmt.elabAt_frame cta n m.ir _ m.imem (fun x hx => hf1 x hx) b i hb.2
        rw [hw2, hbr, hw1]; rfl
      · intro x hx; rw [hf2 x hx, hf1 x hx]
      · rw [hm2i, hm1i]
  | forN cnt body ih =>
      intro hf lr n base i P m h2 hlr hb hfl hinv hseg
      -- names for the pieces
      have hL : (flatEW n (n + 1) (base + 3) body).length = flenEW n (n + 1) body :=
        flatEW_length body n (n + 1) (base + 3)
      have hflen : flenEW lr n (EWStmt.forN cnt body) = flenEW n (n + 1) body + 5 := rfl
      -- instruction lookups
      have hP0 : P[base]? = some (FI.si (SI.movIC n 0)) := by
        have := hseg 0 (by rw [hflen]; omega); simpa using this
      have hP1 : P[base + 1]? = some (FI.si (SI.setpGeC n n cnt)) := by
        have := hseg 1 (by rw [hflen]; omega); simpa using this
      have hP2 : P[base + 2]?
          = some (FI.jmpIf n (base + (flenEW n (n + 1) body + 5))) := by
        have := hseg 2 (by rw [hflen]; omega); simpa using this
      have hPb : ∀ j, j < flenEW n (n + 1) body →
          P[(base + 3) + j]? = (flatEW n (n + 1) (base + 3) body)[j]? := by
        intro j hj
        have := hseg (j + 3) (by rw [hflen]; omega)
        rw [show base + (j + 3) = base + 3 + j by omega] at this
        rw [this]
        show (FI.si _ :: FI.si _ :: FI.jmpIf _ _
                :: (flatEW n (n + 1) (base + 3) body ++ [_, _]))[j + 3]? = _
        show (flatEW n (n + 1) (base + 3) body ++ [FI.si (SI.addRC n n 1),
                FI.jmp (base + 1)])[j]? = _
        exact List.getElem?_append_left (by omega)
      have hPinc : P[base + (flenEW n (n + 1) body + 3)]?
          = some (FI.si (SI.addRC n n 1)) := by
        have := hseg (flenEW n (n + 1) body + 3) (by rw [hflen]; omega)
        rw [this]
        show (FI.si _ :: FI.si _ :: FI.jmpIf _ _
                :: (flatEW n (n + 1) (base + 3) body ++ [_, _]))[flenEW n (n+1) body + 3]? = _
        show (flatEW n (n + 1) (base + 3) body ++ [FI.si (SI.addRC n n 1),
                FI.jmp (base + 1)])[flenEW n (n+1) body]? = _
        rw [List.getElem?_append_right (by omega), hL]
        simp
      have hPback : P[base + (flenEW n (n + 1) body + 4)]?
          = some (FI.jmp (base + 1)) := by
        have := hseg (flenEW n (n + 1) body + 4) (by rw [hflen]; omega)
        rw [this]
        show (FI.si _ :: FI.si _ :: FI.jmpIf _ _
                :: (flatEW n (n + 1) (base + 3) body ++ [_, _]))[flenEW n (n+1) body + 4]? = _
        show (flatEW n (n + 1) (base + 3) body ++ [FI.si (SI.addRC n n 1),
                FI.jmp (base + 1)])[flenEW n (n+1) body + 1]? = _
        rw [List.getElem?_append_right (by omega), hL]
        simp
      -- the loop invariant, by induction on the remaining trip count
      have key : ∀ (d j : Nat), j + d = cnt → ∀ (m' : MState),
          m'.ir n = (fun _ => j) → m'.ir laneIR = (fun l => (l.val : Nat)) →
          m'.ir ctaIR = (fun _ => cta) → (∀ x, x < n → m'.ir x = m.ir x) →
          m'.imem = m.imem →
          ∃ k m'', steps cta P k (base + 1, m')
                = some (base + (flenEW n (n + 1) body + 5), m'')
            ∧ m''.toWSt
                = (List.range' j d).foldl
                    (fun s' t => (body.elabAt cta t m.ir m.imem).run s') m'.toWSt
            ∧ (∀ x, x < n → m''.ir x = m'.ir x) ∧ m''.imem = m'.imem := by
        intro d
        induction d with
        | zero =>
            intro j hj m' hn h0 h1 hpres hmpres
            have hjc : cnt ≤ j := by omega
            refine ⟨2, m'.setPr n (fun l => decide (cnt ≤ m'.ir n l)), ?_, rfl, ?_, rfl⟩
            · show (match fstep cta P (base + 1, m') with
                    | none => none
                    | some c' => steps cta P 1 c') = _
              have hf1 : fstep cta P (base + 1, m')
                  = some (base + 2, m'.setPr n (fun l => decide (cnt ≤ m'.ir n l))) := by
                show (match P[base + 1]? with
                      | none => none
                      | some (.si i) => some (base + 1 + 1, SI.step cta i m')
                      | some (.jmp t) => some (t, m')
                      | some (.jmpIf p t) => _) = _
                rw [hP1]
                show some (base + 1 + 1, _) = _
                rw [show base + 1 + 1 = base + 2 by omega]
                rfl
              rw [hf1]
              show (match fstep cta P (base + 2, _) with
                    | none => none
                    | some c' => steps cta P 0 c') = _
              have hall : allLanes (fun l =>
                  (m'.setPr n (fun l => decide (cnt ≤ m'.ir n l))).pr n l) = true := by
                rw [MState.pr_setPr_same]
                refine List.all_eq_true.mpr ?_
                intro l _
                simp only [hn]
                exact decide_eq_true hjc
              have hf2 : fstep cta P (base + 2,
                  m'.setPr n (fun l => decide (cnt ≤ m'.ir n l)))
                  = some (base + (flenEW n (n + 1) body + 5),
                      m'.setPr n (fun l => decide (cnt ≤ m'.ir n l))) := by
                show (match P[base + 2]? with
                      | none => none
                      | some (.si i) => _
                      | some (.jmp t) => _
                      | some (.jmpIf p t) =>
                          if allLanes (fun l => _) then some (t, _)
                          else if allLanes (fun l => !_) then _ else none) = _
                rw [hP2]
                show (if allLanes (fun l =>
                        (m'.setPr n (fun l => decide (cnt ≤ m'.ir n l))).pr n l)
                      then _ else _) = _
                rw [hall]
                rfl
              rw [hf2]
              rfl
            · intro x hx; rfl
        | succ d ihd =>
            intro j hj m' hn h0 h1 hpres hmpres
            have hjlt : j < cnt := by omega
            -- guard test
            have hf1 : fstep cta P (base + 1, m')
                = some (base + 2, m'.setPr n (fun l => decide (cnt ≤ m'.ir n l))) := by
              show (match P[base + 1]? with
                    | none => none
                    | some (.si i) => some (base + 1 + 1, SI.step cta i m')
                    | some (.jmp t) => some (t, m')
                    | some (.jmpIf p t) => _) = _
              rw [hP1]
              show some (base + 1 + 1, _) = _
              rw [show base + 1 + 1 = base + 2 by omega]
              rfl
            let m1 := m'.setPr n (fun l => decide (cnt ≤ m'.ir n l))
            have hnotall : allLanes (fun l => m1.pr n l) = false := by
              show allLanes (fun l => (m'.setPr n _).pr n l) = false
              rw [MState.pr_setPr_same]
              have : ¬ (allLanes (fun l => decide (cnt ≤ m'.ir n l)) = true) := by
                intro hc
                have := List.all_eq_true.mp hc ⟨0, by decide⟩ (by simp)
                simp only [hn] at this
                have := of_decide_eq_true this
                omega
              cases hb : allLanes (fun l => decide (cnt ≤ m'.ir n l)) with
              | false => rfl
              | true => exact absurd hb this
            have hallf : allLanes (fun l => !m1.pr n l) = true := by
              show allLanes (fun l => !(m'.setPr n _).pr n l) = true
              rw [MState.pr_setPr_same]
              refine List.all_eq_true.mpr ?_
              intro l _
              simp only [hn]
              simp only [Bool.not_eq_true', decide_eq_false_iff_not]
              omega
            have hf2 : fstep cta P (base + 2, m1) = some (base + 3, m1) := by
              show (match P[base + 2]? with
                    | none => none
                    | some (.si i) => _
                    | some (.jmp t) => _
                    | some (.jmpIf p t) =>
                        if allLanes (fun l => _) then some (t, _)
                        else if allLanes (fun l => !_) then some (base + 2 + 1, _) else none) = _
              rw [hP2]
              show (if allLanes (fun l => m1.pr n l) then _
                    else if allLanes (fun l => !m1.pr n l) then some (base + 2 + 1, m1)
                    else none) = _
              rw [hnotall, hallf]
              show some (base + 2 + 1, m1) = _
              rw [show base + 2 + 1 = base + 3 by omega]
            -- body
            have hinvb : MInv cta j n m1 := by
              refine ⟨?_, ?_, ?_⟩
              · show (m'.setPr n _).ir laneIR = _; rw [MState.ir_setPr]; exact h0
              · show (m'.setPr n _).ir ctaIR = _; rw [MState.ir_setPr]; exact h1
              · show (m'.setPr n _).ir n = _; rw [MState.ir_setPr]; exact hn
            obtain ⟨kb, m2, hsb, hwb, hfb, hm2i⟩ :=
              ih hf n (n + 1) (base + 3) j P m1 (by omega) (by omega)
                (EWStmt.idxBelow_mono (by omega) body hb) hfl hinvb hPb
            -- increment
            have hm2n : m2.ir n = (fun _ => j) := by
              rw [hfb n (by omega)]
              show (m'.setPr n _).ir n = _
              rw [MState.ir_setPr]; exact hn
            have hf3 : fstep cta P (base + 3 + flenEW n (n + 1) body, m2)
                = some (base + (flenEW n (n + 1) body + 4),
                    m2.setI n (fun l => m2.ir n l + 1)) := by
              show (match P[base + 3 + flenEW n (n + 1) body]? with
                    | none => none
                    | some (.si i) => some (base + 3 + flenEW n (n + 1) body + 1,
                        SI.step cta i m2)
                    | some (.jmp t) => _
                    | some (.jmpIf p t) => _) = _
              rw [show base + 3 + flenEW n (n + 1) body
                    = base + (flenEW n (n + 1) body + 3) by omega, hPinc]
              show some (base + (flenEW n (n + 1) body + 3) + 1, _) = _
              rw [show base + (flenEW n (n + 1) body + 3) + 1
                    = base + (flenEW n (n + 1) body + 4) by omega]
              rfl
            have hf4 : fstep cta P (base + (flenEW n (n + 1) body + 4),
                  m2.setI n (fun l => m2.ir n l + 1))
                = some (base + 1, m2.setI n (fun l => m2.ir n l + 1)) := by
              show (match P[base + (flenEW n (n + 1) body + 4)]? with
                    | none => none
                    | some (.si i) => _
                    | some (.jmp t) => some (t, _)
                    | some (.jmpIf p t) => _) = _
              rw [hPback]
            -- next iteration
            have hm3n : (m2.setI n (fun l => m2.ir n l + 1)).ir n = (fun _ => j + 1) := by
              rw [MState.ir_setI_same, hm2n]
            obtain ⟨kr, m4, hsr, hwr, hfr, hm4i⟩ := ihd (j + 1) (by omega)
              (m2.setI n (fun l => m2.ir n l + 1)) hm3n
              (by rw [MState.ir_setI_other _ _ _ _ (show 0 ≠ n by omega), hfb laneIR (show 0 < n + 1 by omega)]
                  show (m'.setPr n _).ir laneIR = _
                  rw [MState.ir_setPr]; exact h0)
              (by rw [MState.ir_setI_other _ _ _ _ (show 1 ≠ n by omega), hfb ctaIR (show 1 < n + 1 by omega)]
                  show (m'.setPr n _).ir ctaIR = _
                  rw [MState.ir_setPr]; exact h1)
              (by intro x hx
                  rw [MState.ir_setI_other _ _ _ _ (show x ≠ n by omega),
                      hfb x (show x < n + 1 by omega)]
                  show (m'.setPr n _).ir x = _
                  rw [MState.ir_setPr]; exact hpres x hx)
              (by rw [show (m2.setI n (fun l => m2.ir n l + 1)).imem = m2.imem from rfl,
                      hm2i, show m1.imem = m'.imem from rfl, hmpres])
            refine ⟨2 + (kb + (2 + kr)), m4, ?_, ?_, ?_, ?_⟩
            · refine steps_trans cta P 2 (kb + (2 + kr)) _ (base + 3, m1) _ ?_ ?_
              · show (match fstep cta P (base + 1, m') with
                      | none => none
                      | some c' => steps cta P 1 c') = _
                rw [hf1]
                show (match fstep cta P (base + 2, m1) with
                      | none => none
                      | some c' => steps cta P 0 c') = _
                rw [hf2]
                rfl
              · refine steps_trans cta P kb (2 + kr) _
                  (base + 3 + flenEW n (n + 1) body, m2) _ hsb ?_
                refine steps_trans cta P 2 kr _
                  (base + 1, m2.setI n (fun l => m2.ir n l + 1)) _ ?_ hsr
                show (match fstep cta P (base + 3 + flenEW n (n + 1) body, m2) with
                      | none => none
                      | some c' => steps cta P 1 c') = _
                rw [hf3]
                show (match fstep cta P (base + (flenEW n (n + 1) body + 4), _) with
                      | none => none
                      | some c' => steps cta P 0 c') = _
                rw [hf4]
                rfl
            · rw [hwr]
              show _ = (List.range' j (d + 1)).foldl _ m'.toWSt
              rw [List.range'_succ, List.foldl_cons]
              have : (m2.setI n (fun l => m2.ir n l + 1)).toWSt = m2.toWSt := rfl
              have hbr : body.elabAt cta j m1.ir m1.imem = body.elabAt cta j m.ir m.imem := by
                rw [show m1.imem = m'.imem from rfl, hmpres]
                exact EWStmt.elabAt_frame cta n m.ir _ m.imem
                  (fun x hx => by
                    show (m'.setPr n _).ir x = _
                    rw [MState.ir_setPr]; exact hpres x hx) body j hb
              rw [this, hwb, hbr]
              rfl
            · intro x hx
              rw [hfr x hx, MState.ir_setI_other _ _ _ _ (by omega), hfb x (by omega)]
              show (m'.setPr n _).ir x = _
              rw [MState.ir_setPr]
            · rw [hm4i, show (m2.setI n (fun l => m2.ir n l + 1)).imem = m2.imem from rfl,
                  hm2i]
              rfl
      -- initialise the counter, then run the loop
      have hf0 : fstep cta P (base, m) = some (base + 1, m.setI n (fun _ => 0)) := by
        show (match P[base]? with
              | none => none
              | some (.si i) => some (base + 1, SI.step cta i m)
              | some (.jmp t) => _
              | some (.jmpIf p t) => _) = _
        rw [hP0]
        rfl
      obtain ⟨k, m'', hs, hw, hf, hmi⟩ := key cnt 0 (by omega) (m.setI n (fun _ => 0))
        (MState.ir_setI_same _ _ _)
        (by rw [MState.ir_setI_other _ _ _ _ (show 0 ≠ n by omega)]; exact hinv.1)
        (by rw [MState.ir_setI_other _ _ _ _ (show 1 ≠ n by omega)]; exact hinv.2.1)
        (fun x hx => MState.ir_setI_other _ _ _ _ (show x ≠ n by omega)) rfl
      refine ⟨1 + k, m'', ?_, ?_, ?_, ?_⟩
      · refine steps_trans cta P 1 k _ (base + 1, m.setI n (fun _ => 0)) _ ?_ hs
        show (match fstep cta P (base, m) with
              | none => none
              | some c' => steps cta P 0 c') = _
        rw [hf0]
        rfl
      · rw [hw]
        show (List.range' 0 cnt).foldl _ (m.setI n (fun _ => 0)).toWSt = _
        rw [← List.range_eq_range']
        rfl
      · intro x hx
        rw [hf x hx, MState.ir_setI_other _ _ _ _ (by omega)]
      · rw [hmi, show (m.setI n (fun _ => 0)).imem = m.imem from rfl]

-- ---------------------------------------------------------------------------
-- A whole kernel, from raw launch
-- ---------------------------------------------------------------------------

/-- The complete flat program for a kernel: prologue, then the flattened body. -/
def flatKernel (s : EWStmt) : List FI := emitPrologue.map FI.si ++ flatEW 2 3 3 s

theorem flatKernel_length (s : EWStmt) :
    (flatKernel s).length = 3 + flenEW 2 3 s := by
  show (emitPrologue.map FI.si ++ flatEW 2 3 3 s).length = _
  rw [List.length_append, flatEW_length]
  rfl

/-- **The final theorem: a launched kernel executes its statement.**

    The machine is a program counter over a list of real instructions with real
    branches.  Starting at instruction 0 in an arbitrary state, the program runs
    to completion and leaves exactly `(s.elabIn cta).run` of the incoming warp
    state — for every block index, with no hypothesis on the incoming registers
    (the prologue establishes what it needs) and no epsilon.

    Composed with `warpSumSqV4Store_implements` this reads: *the PTX that ships
    computes the user's spec.* -/
theorem flatKernel_sound (cta : Nat) (s : EWStmt) (hf : s.ExpFree) (hb : s.IdxBelow 3)
    (hfl : s.Flat) (m : MState) :
    ∃ k m', steps cta (flatKernel s) k (0, m) = some ((flatKernel s).length, m')
      ∧ m'.toWSt
        = (s.elabAt cta 0 (SI.stepL cta emitPrologue m).ir m.imem).run m.toWSt := by
  have hpro : ∀ j, j < emitPrologue.length →
      (flatKernel s)[0 + j]? = (emitPrologue.map FI.si)[j]? := by
    intro j hj
    show (emitPrologue.map FI.si ++ flatEW 2 3 3 s)[0 + j]? = _
    rw [show 0 + j = j by omega]
    exact List.getElem?_append_left (by simpa using hj)
  have hs1 := steps_si cta (flatKernel s) emitPrologue 0 m hpro
  have hbody : ∀ j, j < flenEW 2 3 s →
      (flatKernel s)[3 + j]? = (flatEW 2 3 3 s)[j]? := by
    intro j hj
    show (emitPrologue.map FI.si ++ flatEW 2 3 3 s)[3 + j]? = _
    rw [List.getElem?_append_right (by simp [emitPrologue])]
    congr 1
    simp [emitPrologue]
  have hinv : MInv cta 0 2 (SI.stepL cta emitPrologue m) := by
    have hpro' : SI.stepL cta emitPrologue m
        = ((m.setI laneIR (fun l => l.val)).setI ctaIR (fun _ => cta)).setI 2 (fun _ => 0) := rfl
    refine ⟨?_, ?_, ?_⟩
    · rw [hpro', MState.ir_setI_other _ _ _ _ (by decide),
          MState.ir_setI_other _ _ _ _ (by decide), MState.ir_setI_same]
    · rw [hpro', MState.ir_setI_other _ _ _ _ (by decide), MState.ir_setI_same]
    · rw [hpro', MState.ir_setI_same]
  obtain ⟨k, m', hs2, hw, _, _⟩ := flatEW_sound cta s hf 2 3 3 0 (flatKernel s)
    (SI.stepL cta emitPrologue m) (by omega) (by omega) hb hfl hinv hbody
  refine ⟨emitPrologue.length + k, m', ?_, ?_⟩
  · refine steps_trans cta (flatKernel s) emitPrologue.length k _
      (0 + emitPrologue.length, SI.stepL cta emitPrologue m) _ hs1 ?_
    rw [show 0 + emitPrologue.length = 3 from rfl]
    rw [hs2, flatKernel_length]
  · rw [hw]
    have hws : (SI.stepL cta emitPrologue m).toWSt = m.toWSt := rfl
    have him : (SI.stepL cta emitPrologue m).imem = m.imem := rfl
    rw [hws, him]

/-- **The address-free specialisation.**  For a kernel with no data-dependent
    index — every kernel written before `IdxE.ireg` — the register file drops
    out and the conclusion is the original `elabIn` statement. -/
theorem flatKernel_sound_idxFree (cta : Nat) (s : EWStmt) (hf : s.ExpFree)
    (hif : s.IdxFree) (hfl : s.Flat) (m : MState) :
    ∃ k m', steps cta (flatKernel s) k (0, m) = some ((flatKernel s).length, m')
      ∧ m'.toWSt = (s.elabIn cta).run m.toWSt := by
  obtain ⟨k, m', hs, hw⟩ :=
    flatKernel_sound cta s hf (EWStmt.idxBelow_of_idxFree 3 s hif) hfl m
  refine ⟨k, m', hs, ?_⟩
  rw [hw, EWStmt.elabAt_idxFree cta _ _ s 0 hif]
  rfl
