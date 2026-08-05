import AlgorithmLib.LZ4Concurrent

/-!
  # Transporting facts along a warp's trace

  `Lz4Sites.RegConfined` is a statement about `siter prog k st` for every `k`:
  at each step, if the pc sits at a load or store site, the address register
  holds something in range.  Nothing in the development could state, let alone
  prove, a property of *every* step — the body simulation relates the launch
  state to the final state, and the slices in between are internal to its proof.

  This file supplies the two things any such proof needs first, and neither is
  specific to LZ4:

  * `siter_inv` — an invariant preserved by one step holds at every step.  The
    induction is trivial; what matters is that it exists, so a step invariant
    can be *stated* about the shipped kernel.
  * `siter_regs_const` — a register no instruction in the program writes is
    constant along the whole trace, with `noDest` deciding the side condition
    by scanning the program.  This is what turns "`outBase` is set once in the
    prologue" from a remark about the generator into a fact about the emitted
    array.

  Neither closes `RegConfined`.  The bound that does — that the output cursor
  `op` never passes `lenOff` at a store — is the LZ4 worst-case-expansion
  argument restated per step, and it needs an invariant relating `op` to the
  input consumed.  These are the transport lemmas that invariant will be moved
  along by.
-/

namespace AlgorithmLib.LZ4Simt

/-- The register an instruction writes, if any.  Stores, branches, labels,
    `bar.warp.sync` and `ret` write no register. -/
def destOf : SInstr → Option String
  | .mov d _        => some d
  | .bin _ d _ _    => some d
  | .binr _ d _ _   => some d
  | .cvt32 d _      => some d
  | .brev d _       => some d
  | .clz d _        => some d
  | .bnot d _       => some d
  | .setp _ p _ _   => some p
  | .andp d _ _     => some d
  | .selp d _ _ _   => some d
  | .ldgo d _ _     => some d
  | .ldgop _ d _ _  => some d
  | .ldsh d _       => some d
  | .vote d _       => some d
  | .shfl d _ _     => some d
  | _               => none

/-- **An invariant preserved by one step holds at every step.** -/
theorem siter_inv (p : Array SInstr) (I : SState → Prop)
    (hstep : ∀ st, I st → I (sstep p st)) :
    ∀ (k : Nat) (st : SState), I st → I (siter p k st) := by
  intro k
  induction k with
  | zero => intro st h; exact h
  | succ m ih => intro st h; rw [siter]; exact ih _ (hstep st h)

/-- The step-local frame: an instruction that does not name `r` as its
    destination leaves `r` alone, in every lane. -/
theorem sstep_regs_frame (p : Array SInstr) (st : SState) (r : String)
    (h : ∀ i, p[st.pc]? = some i → destOf i ≠ some r) :
    (sstep p st).regs r = st.regs r := by
  cases hp : p[st.pc]? with
  | none => rw [sstep, hp]
  | some i =>
      have hne : destOf i ≠ some r := h i hp
      rw [sstep, hp]
      cases i <;>
        (try simp only [destOf, ne_eq, Option.some.injEq] at hne) <;>
        (try simp only [sstepInstr, SState.setReg, SState.setPc]) <;>
        first
          | rfl
          | exact if_neg (fun e => hne e.symm)

/-- Decides "no instruction in `p` writes `r`" by scanning the program. -/
def noDest (p : Array SInstr) (r : String) : Bool :=
  p.toList.all (fun i => destOf i != some r)

theorem noDest_spec {p : Array SInstr} {r : String} (h : noDest p r = true)
    (n : Nat) (i : SInstr) (hn : p[n]? = some i) : destOf i ≠ some r := by
  have hmem : i ∈ p.toList := List.mem_of_getElem? (by simpa using hn)
  have := List.all_eq_true.mp h i hmem
  simpa using this

/-- **A register the program never writes is constant along the trace.**  The
    side condition is decided by `noDest`, so it is a fact about the emitted
    array rather than about the generator that produced it. -/
theorem siter_regs_const (p : Array SInstr) (r : String) (h : noDest p r = true)
    (st : SState) (k : Nat) : (siter p k st).regs r = st.regs r :=
  siter_inv p (fun s => s.regs r = st.regs r)
    (fun s hs => (sstep_regs_frame p s r (fun i hi => noDest_spec h s.pc i hi)).trans hs)
    k st rfl

-- ── Which program point can precede which ───────────────────────────────────

/-- A branch can only land on a `lbl`.  `sfindLabel` returns the index of a
    matching `.lbl`, or `p.size` when there is none — so if the slot it names
    holds an instruction at all, that instruction is that label. -/
theorem sfindLabel_isLbl (p : Array SInstr) (l : String) (i : SInstr)
    (h : p[sfindLabel p l]? = some i) : i = .lbl l := by
  have h' : p.toList[p.toList.findIdx
      (fun i => match i with | .lbl n => n == l | _ => false)]? = some i := by
    simpa [sfindLabel] using h
  have := List.findIdx_of_getElem?_eq_some h'
  cases i <;> simp_all only [reduceCtorEq, beq_iff_eq]

/-- **A program point that holds neither a label nor `ret` is reachable only by
    falling through from the point before it.**

    This is what makes a local argument about a store site sound: the store at
    pc `q` can only have been preceded by whatever sits at `q - 1`, so a fact
    established by the instruction at `q - 1` holds whenever the machine is at
    `q`.  Without it, "the address register was set just above" is a remark
    about how the generator emits code, not about the trace. -/
theorem pc_pred (p : Array SInstr) (st : SState) (q : Nat) (i : SInstr)
    (hq : p[q]? = some i) (hlbl : ∀ n, i ≠ .lbl n) (hret : i ≠ .ret)
    (h : (sstep p st).pc = q) : st.pc + 1 = q := by
  rw [sstep] at h
  cases hp : p[st.pc]? with
  | none => rw [hp] at h; exact absurd (hq.symm.trans (h ▸ hp)) (by simp)
  | some j =>
      rw [hp] at h
      have hbr : ∀ (l : String), sfindLabel p l ≠ q := by
        intro l e
        exact hlbl l (sfindLabel_isLbl p l i (e ▸ hq))
      cases j <;> simp only [sstepInstr, SState.setReg, SState.setPc] at h
      case ret => rw [← h, hp] at hq; exact absurd (Option.some.inj hq).symm hret
      case braif pr lb =>
          by_cases hc : st.regs pr 0 == 1
          · exact absurd (by rw [← h, if_pos hc]) (hbr lb)
          · rw [if_neg hc] at h; exact h
      case braifnot pr lb =>
          by_cases hc : st.regs pr 0 == 1
          · rw [if_pos hc] at h; exact h
          · exact absurd (by rw [← h, if_neg hc]) (hbr lb)
      case bra lb => exact absurd h.symm (fun e => hbr lb e.symm)
      all_goals exact h

/-- **A register assigned immediately above a program point holds that value at
    it.**  If `q - 1` is `d := a + b` (with `d` distinct from `a` and `b`) and
    `q` is neither a label nor `ret`, then whenever the machine stands at `q`,
    `d` is `a + b` *of the state it is standing in*.

    This is the whole content of "the address register is set just before the
    store", made a fact about the trace rather than about the generator. -/
theorem add_above_holds_at (p : Array SInstr) (q : Nat) (d a b : String) (i : SInstr)
    (hq : p[q]? = some i) (hlbl : ∀ n, i ≠ .lbl n) (hret : i ≠ .ret)
    (hpre : p[q - 1]? = some (.bin .add d a (.reg b)))
    (hda : d ≠ a) (hdb : d ≠ b) (hq0 : 0 < q)
    (st : SState) (hpc : (sstep p st).pc = q) (l : Lane) :
    (sstep p st).regs d l = (sstep p st).regs a l + (sstep p st).regs b l := by
  have hst : st.pc = q - 1 := by have := pc_pred p st q i hq hlbl hret hpc; omega
  have hi : p[st.pc]? = some (.bin .add d a (.reg b)) := by rw [hst]; exact hpre
  rw [sstep, hi]
  simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run,
    if_true, if_neg (Ne.symm hda), if_neg (Ne.symm hdb)]

/-- `true` on the instructions a branch can never land on and that are not the
    halting `ret` — i.e. exactly the points reachable only by fallthrough. -/
def fallthroughOnlyB : SInstr → Bool
  | .lbl _ => false
  | .ret   => false
  | _      => true

/-- `add_above_holds_at` with every side condition on the program decidable, so
    instantiating it at a concrete program point is one `decide` each. -/
theorem add_above_holds_at' (p : Array SInstr) (q : Nat) (d a b : String)
    (hq : (p[q]?.map fallthroughOnlyB) = some true)
    (hpre : p[q - 1]? = some (.bin .add d a (.reg b)))
    (hda : d ≠ a) (hdb : d ≠ b) (hq0 : 0 < q)
    (st : SState) (hpc : (sstep p st).pc = q) (l : Lane) :
    (sstep p st).regs d l = (sstep p st).regs a l + (sstep p st).regs b l := by
  cases hi : p[q]? with
  | none => rw [hi] at hq; exact absurd hq (by simp)
  | some i =>
      rw [hi] at hq
      have hf : fallthroughOnlyB i = true := by simpa using hq
      refine add_above_holds_at p q d a b i hi ?_ ?_ hpre hda hdb hq0 st hpc l
      · intro n e; rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB])
      · intro e; rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB])

/-- `add_above_holds_at'` for an immediate second operand. -/
theorem add_imm_above_holds_at' (p : Array SInstr) (q : Nat) (d a : String) (c : Nat)
    (hq : (p[q]?.map fallthroughOnlyB) = some true)
    (hpre : p[q - 1]? = some (.bin .add d a (.imm c)))
    (hda : d ≠ a) (hq0 : 0 < q)
    (st : SState) (hpc : (sstep p st).pc = q) (l : Lane) :
    (sstep p st).regs d l = (sstep p st).regs a l + UInt64.ofNat c := by
  have hlr : ∀ i, p[q]? = some i → fallthroughOnlyB i = true := by
    intro i hi; rw [hi] at hq; simpa using hq
  cases hi : p[q]? with
  | none => rw [hi] at hq; exact absurd hq (by simp)
  | some i =>
      have hf := hlr i hi
      have hst : st.pc = q - 1 := by
        have := pc_pred p st q i hi
          (fun n e => by rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
          (by intro e; rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB])) hpc
        omega
      have hj : p[st.pc]? = some (.bin .add d a (.imm c)) := by rw [hst]; exact hpre
      rw [sstep, hj]
      simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run,
        if_true, if_neg (Ne.symm hda)]

/-- **The register-register form, stated on the state the instruction ran in.**

    `add_above_holds_at'` phrases its conclusion in the *post*-state, which is
    why it needs `d ≠ a` and `d ≠ b`: a source has to survive the assignment to
    be nameable afterwards.  An accumulating step — `d := d + b` — cannot be
    phrased that way at all.  So this one names the predecessor state instead,
    and pays for it by being composable rather than self-contained: two of them
    in a row place a register built in two instructions. -/
theorem binr_pre_holds_at (p : Array SInstr) (q : Nat) (o : SOp) (d a b : String)
    (hq : (p[q]?.map fallthroughOnlyB) = some true)
    (hpre : p[q - 1]? = some (.binr o d a b))
    (st : SState) (hpc : (sstep p st).pc = q) (l : Lane) :
    (sstep p st).regs d l = o.run (st.regs a l) (st.regs b l) := by
  cases hi : p[q]? with
  | none => rw [hi] at hq; exact absurd hq (by simp)
  | some i =>
      rw [hi] at hq
      have hf : fallthroughOnlyB i = true := by simpa using hq
      have hst : st.pc + 1 = q :=
        pc_pred p st q i hi
          (fun n e => by rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
          (by intro e; rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB])) hpc
      have hj : p[st.pc]? = some (.binr o d a b) := by
        rw [show st.pc = q - 1 from by omega]; exact hpre
      rw [sstep, hj]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_true]

-- ── Walking a fallthrough run backwards ──────────────────────────────────────

/-- **A state standing at `q` stood at `q - n` exactly `n` steps earlier**, when
    every point in `(q - n, q]` is reachable only by fallthrough.

    The generalisation of `pc_pred` from one step to a straight-line run.  It is
    what lets a fact established anywhere inside a basic block be used at the
    memory instruction that ends it, without reasoning about the whole trace. -/
theorem pc_back (p : Array SInstr) (init : SState) (hinit : init.pc = 0) (q : Nat) :
    ∀ n : Nat, n ≤ q →
      (∀ t, t < n → (p[q - t]?.map fallthroughOnlyB) = some true) →
      ∀ k, (siter p k init).pc = q → n ≤ k ∧ (siter p (k - n) init).pc = q - n := by
  intro n
  induction n with
  | zero => intro _ _ k hpc; exact ⟨Nat.zero_le _, by simpa using hpc⟩
  | succ m ih =>
      intro hmq hft k hpc
      obtain ⟨hmk, hpcm⟩ :=
        ih (by omega) (fun t h1 => hft t (by omega)) k hpc
      have hq : (p[q - m]?.map fallthroughOnlyB) = some true := hft m (by omega)
      cases hkm : k - m with
      | zero =>
          rw [hkm] at hpcm
          rw [show siter p 0 init = init from rfl, hinit] at hpcm
          omega
      | succ t =>
          rw [hkm, siter_succ] at hpcm
          cases hi : p[q - m]? with
          | none => rw [hi] at hq; exact absurd hq (by simp)
          | some i =>
              rw [hi] at hq
              have hf : fallthroughOnlyB i = true := by simpa using hq
              have hstep := pc_pred p (siter p t init) (q - m) i hi
                (fun n e => by rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
                (by intro e; rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
                hpcm
              refine ⟨by omega, ?_⟩
              rw [show k - (m + 1) = t from by omega]
              omega

/-- **…and the registers that run does not write are the ones it arrived with.**
    Together with `pc_back` this moves a value forward from wherever it was
    computed to the instruction that uses it. -/
theorem regs_back (p : Array SInstr) (init : SState) (hinit : init.pc = 0)
    (d : String) (q : Nat) :
    ∀ n : Nat, n ≤ q →
      (∀ t, t < n → (p[q - t]?.map fallthroughOnlyB) = some true) →
      (∀ t, t < n → (p[q - t - 1]?.map (fun i => destOf i != some d)) = some true) →
      ∀ k, (siter p k init).pc = q →
        (siter p k init).regs d = (siter p (k - n) init).regs d := by
  intro n
  induction n with
  | zero => intro _ _ _ k _; rfl
  | succ m ih =>
      intro hmq hft hnw k hpc
      have hstep := ih (by omega) (fun t h => hft t (by omega)) (fun t h => hnw t (by omega)) k hpc
      obtain ⟨hmk, hpcm⟩ :=
        pc_back p init hinit q m (by omega) (fun t h => hft t (by omega)) k hpc
      cases hkm : k - m with
      | zero =>
          rw [hkm] at hpcm
          rw [show siter p 0 init = init from rfl, hinit] at hpcm
          omega
      | succ t =>
          rw [hstep, hkm, siter_succ]
          rw [show k - (m + 1) = t from by omega]
          refine sstep_regs_frame p (siter p t init) d (fun i hi => ?_)
          have hpct : (siter p t init).pc = q - (m + 1) := by
            have hq : (p[q - m]?.map fallthroughOnlyB) = some true := hft m (by omega)
            cases hj : p[q - m]? with
            | none => rw [hj] at hq; exact absurd hq (by simp)
            | some j =>
                rw [hj] at hq
                have hf : fallthroughOnlyB j = true := by simpa using hq
                have := pc_pred p (siter p t init) (q - m) j hj
                  (fun n e => by rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
                  (by intro e; rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
                  (by rw [← siter_succ, ← hkm]; exact hpcm)
                omega
          rw [hpct] at hi
          have := hnw m (by omega)
          rw [show q - m - 1 = q - (m + 1) from by omega, hi] at this
          simpa using this

/-- **A value assigned anywhere in a basic block still holds at the end of it.**

    `d := a + c` sits at `q - n - 1`; the run `q - n … q` is fallthrough-only and
    writes neither `d` nor `a`.  Then at `q`, `d` is `a + c` of the state
    standing there.  Every hypothesis is a decidable check on the emitted array,
    so instantiating this at a program point is a handful of `decide`s.

    This is the reusable form of "the address register was computed just above
    the store", and it is what the memory instructions in a straight-line tail
    need. -/
theorem add_imm_carried (p : Array SInstr) (init : SState) (hinit : init.pc = 0)
    (d a : String) (c q n : Nat)
    (hft : ∀ t, t < n + 1 → (p[q - t]?.map fallthroughOnlyB) = some true)
    (hnwd : ∀ t, t < n → (p[q - t - 1]?.map (fun i => destOf i != some d)) = some true)
    (hnwa : ∀ t, t < n → (p[q - t - 1]?.map (fun i => destOf i != some a)) = some true)
    (hpre : p[q - n - 1]? = some (.bin .add d a (.imm c)))
    (hda : d ≠ a) (hn : n + 1 ≤ q)
    (k : Nat) (hpc : (siter p k init).pc = q) (l : Lane) :
    (siter p k init).regs d l = (siter p k init).regs a l + UInt64.ofNat c := by
  have hd := regs_back p init hinit d q n (by omega) (fun t h => hft t (by omega)) hnwd k hpc
  have ha := regs_back p init hinit a q n (by omega) (fun t h => hft t (by omega)) hnwa k hpc
  obtain ⟨hnk, hpcn⟩ :=
    pc_back p init hinit q n (by omega) (fun t h => hft t (by omega)) k hpc
  rw [hd, ha]
  cases hk : k - n with
  | zero =>
      rw [hk, show siter p 0 init = init from rfl, hinit] at hpcn
      omega
  | succ t =>
      rw [siter_succ]
      rw [hk, siter_succ] at hpcn
      exact add_imm_above_holds_at' p (q - n) d a c
        (by have := hft n (by omega); simpa using this)
        (by rw [show q - n - 1 = q - n - 1 from rfl]; exact hpre)
        hda (by omega) _ hpcn l

/-- The post-state form of `binr_pre_holds_at`, for the ordinary case where the
    destination is distinct from both sources.  `add_above_holds_at'` for
    `.binr`. -/
theorem binr_above_holds_at' (p : Array SInstr) (q : Nat) (o : SOp) (d a b : String)
    (hq : (p[q]?.map fallthroughOnlyB) = some true)
    (hpre : p[q - 1]? = some (.binr o d a b))
    (hda : d ≠ a) (hdb : d ≠ b)
    (st : SState) (hpc : (sstep p st).pc = q) (l : Lane) :
    (sstep p st).regs d l = o.run ((sstep p st).regs a l) ((sstep p st).regs b l) := by
  cases hi : p[q]? with
  | none => rw [hi] at hq; exact absurd hq (by simp)
  | some i =>
      rw [hi] at hq
      have hf : fallthroughOnlyB i = true := by simpa using hq
      have hst : st.pc + 1 = q :=
        pc_pred p st q i hi
          (fun n e => by rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
          (by intro e; rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB])) hpc
      have hj : p[st.pc]? = some (.binr o d a b) := by
        rw [show st.pc = q - 1 from by omega]; exact hpre
      rw [sstep, hj]
      simp only [sstepInstr, SState.setReg, SState.setPc, if_true,
        if_neg (Ne.symm hda), if_neg (Ne.symm hdb)]

/-- `add_imm_carried` for a register-register add: `d := a ⊕ b` at `q - n - 1`,
    then a fallthrough run of `n` instructions writing none of `d, a, b`. -/
theorem binr_carried (p : Array SInstr) (init : SState) (hinit : init.pc = 0)
    (o : SOp) (d a b : String) (q n : Nat)
    (hft : ∀ t, t < n + 1 → (p[q - t]?.map fallthroughOnlyB) = some true)
    (hnwd : ∀ t, t < n → (p[q - t - 1]?.map (fun i => destOf i != some d)) = some true)
    (hnwa : ∀ t, t < n → (p[q - t - 1]?.map (fun i => destOf i != some a)) = some true)
    (hnwb : ∀ t, t < n → (p[q - t - 1]?.map (fun i => destOf i != some b)) = some true)
    (hpre : p[q - n - 1]? = some (.binr o d a b))
    (hda : d ≠ a) (hdb : d ≠ b) (hn : n + 1 ≤ q)
    (k : Nat) (hpc : (siter p k init).pc = q) (l : Lane) :
    (siter p k init).regs d l
      = o.run ((siter p k init).regs a l) ((siter p k init).regs b l) := by
  have hd := regs_back p init hinit d q n (by omega) (fun t h => hft t (by omega)) hnwd k hpc
  have ha := regs_back p init hinit a q n (by omega) (fun t h => hft t (by omega)) hnwa k hpc
  have hb := regs_back p init hinit b q n (by omega) (fun t h => hft t (by omega)) hnwb k hpc
  obtain ⟨hnk, hpcn⟩ :=
    pc_back p init hinit q n (by omega) (fun t h => hft t (by omega)) k hpc
  rw [hd, ha, hb]
  cases hk : k - n with
  | zero =>
      rw [hk, show siter p 0 init = init from rfl, hinit] at hpcn
      omega
  | succ t =>
      rw [siter_succ]
      rw [hk, siter_succ] at hpcn
      exact binr_above_holds_at' p (q - n) o d a b
        (by have := hft n (by omega); simpa using this) hpre hda hdb _ hpcn l

/-- **An accumulated register, carried to the instruction that uses it.**

    `d := a ⊕ b` at `q - n - 2`, then `d := d ⊕ c` at `q - n - 1`, then a
    fallthrough run of `n` instructions writing none of `d, a, b, c`.  At `q`,
    `d` is `(a ⊕ b) ⊕ c` of the state standing there.

    `add_imm_carried` cannot cover this: its conclusion names `d` and `a` in the
    same state, which the accumulating step makes impossible.  Two
    `binr_pre_holds_at`s compose instead, and `d ≠ c` falls out of the frame
    condition rather than being assumed — at `t = n` that condition *is* the
    statement that the second instruction's destination is not `c`. -/
theorem binr_pair_carried (p : Array SInstr) (init : SState) (hinit : init.pc = 0)
    (o : SOp) (d a b c : String) (q n : Nat)
    (hft : ∀ t, t < n + 2 → (p[q - t]?.map fallthroughOnlyB) = some true)
    (hnwd : ∀ t, t < n → (p[q - t - 1]?.map (fun i => destOf i != some d)) = some true)
    (hnwa : ∀ t, t < n + 2 → (p[q - t - 1]?.map (fun i => destOf i != some a)) = some true)
    (hnwb : ∀ t, t < n + 2 → (p[q - t - 1]?.map (fun i => destOf i != some b)) = some true)
    (hnwc : ∀ t, t < n + 2 → (p[q - t - 1]?.map (fun i => destOf i != some c)) = some true)
    (h1 : p[q - n - 2]? = some (.binr o d a b))
    (h2 : p[q - n - 1]? = some (.binr o d d c))
    (hn : n + 2 ≤ q)
    (k : Nat) (hpc : (siter p k init).pc = q) (l : Lane) :
    (siter p k init).regs d l
      = o.run (o.run ((siter p k init).regs a l) ((siter p k init).regs b l))
              ((siter p k init).regs c l) := by
  have hdc : d ≠ c := by
    have hx := hnwc n (by omega)
    rw [h2] at hx; simpa [destOf] using hx
  have hd := regs_back p init hinit d q n (by omega) (fun t h => hft t (by omega)) hnwd k hpc
  have ha := regs_back p init hinit a q (n + 2) (by omega) hft hnwa k hpc
  have hb := regs_back p init hinit b q (n + 2) (by omega) hft hnwb k hpc
  have hc := regs_back p init hinit c q (n + 2) (by omega) hft hnwc k hpc
  obtain ⟨hk2, hpc2⟩ := pc_back p init hinit q (n + 2) (by omega) hft k hpc
  obtain ⟨hk1, hpc1⟩ :=
    pc_back p init hinit q (n + 1) (by omega) (fun t h => hft t (by omega)) k hpc
  obtain ⟨hk0, hpc0⟩ :=
    pc_back p init hinit q n (by omega) (fun t h => hft t (by omega)) k hpc
  have e1 : siter p (k - (n + 1)) init = sstep p (siter p (k - (n + 2)) init) := by
    rw [show k - (n + 1) = (k - (n + 2)) + 1 from by omega, siter_succ]
  have e0 : siter p (k - n) init = sstep p (siter p (k - (n + 1)) init) := by
    rw [show k - n = (k - (n + 1)) + 1 from by omega, siter_succ]
  have s1 : (siter p (k - (n + 1)) init).regs d l
      = o.run ((siter p (k - (n + 2)) init).regs a l)
              ((siter p (k - (n + 2)) init).regs b l) := by
    rw [e1]
    refine binr_pre_holds_at p (q - n - 1) o d a b ?_ ?_ _ ?_ l
    · have := hft (n + 1) (by omega)
      rwa [show q - (n + 1) = q - n - 1 from by omega] at this
    · rwa [show q - n - 1 - 1 = q - n - 2 from by omega]
    · rw [← e1]; rwa [show q - n - 1 = q - (n + 1) from by omega]
  have s0 : (siter p (k - n) init).regs d l
      = o.run ((siter p (k - (n + 1)) init).regs d l)
              ((siter p (k - (n + 1)) init).regs c l) := by
    rw [e0]
    exact binr_pre_holds_at p (q - n) o d d c (hft n (by omega)) h2 _ (by rw [← e0]; exact hpc0) l
  have hc1 : (siter p (k - (n + 1)) init).regs c = (siter p (k - (n + 2)) init).regs c := by
    rw [e1]
    refine sstep_regs_frame p _ c (fun i hi => ?_)
    rw [show (siter p (k - (n + 2)) init).pc = q - n - 2 from by rw [hpc2]; omega, h1] at hi
    rw [← Option.some.inj hi]; simpa [destOf] using hdc
  rw [hd, s0, s1, hc1, ha, hb, hc]

/-- **A constant loaded into a register still sits there at the end of the block.**

    The `mov d, imm` companion to `add_imm_carried`.  The clamps the search half
    of the kernel uses (`cap4 = inStride - 4`, `ec1 = inStride - 6`) are set by a
    `mov` and then read by a `min` some instructions later, so this is what turns
    "the kernel clamps its own read pointer" into a usable bound. -/
theorem mov_imm_carried (p : Array SInstr) (init : SState) (hinit : init.pc = 0)
    (d : String) (c q n : Nat)
    (hft : ∀ t, t < n + 1 → (p[q - t]?.map fallthroughOnlyB) = some true)
    (hnwd : ∀ t, t < n → (p[q - t - 1]?.map (fun i => destOf i != some d)) = some true)
    (hpre : p[q - n - 1]? = some (.mov d (.imm c)))
    (hn : n + 1 ≤ q)
    (k : Nat) (hpc : (siter p k init).pc = q) (l : Lane) :
    (siter p k init).regs d l = UInt64.ofNat c := by
  have hd := regs_back p init hinit d q n (by omega) (fun t h => hft t (by omega)) hnwd k hpc
  obtain ⟨hnk, hpcn⟩ :=
    pc_back p init hinit q n (by omega) (fun t h => hft t (by omega)) k hpc
  rw [hd]
  cases hk : k - n with
  | zero =>
      rw [hk, show siter p 0 init = init from rfl, hinit] at hpcn
      omega
  | succ t =>
      rw [siter_succ]
      rw [hk, siter_succ] at hpcn
      -- the state one step back stands at `q - n - 1`, where the `mov` sits
      have hq : (p[q - n]?.map fallthroughOnlyB) = some true := hft n (by omega)
      cases hi : p[q - n]? with
      | none => rw [hi] at hq; exact absurd hq (by simp)
      | some i =>
          rw [hi] at hq
          have hf : fallthroughOnlyB i = true := by simpa using hq
          have hst : (siter p t init).pc + 1 = q - n :=
            pc_pred p (siter p t init) (q - n) i hi
              (fun m e => by rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
              (by intro e; rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB])) hpcn
          have hj : p[(siter p t init).pc]? = some (.mov d (.imm c)) := by
            rw [show (siter p t init).pc = q - n - 1 from by omega]; exact hpre
          rw [sstep, hj]
          simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_true]

/-- Turn a `decide`able check over a concrete window into the interval form the
    region lemmas want.  Ranges are the cheap shape: `decide` over
    `(List.range n).map (· + lo)` touches `n` program points, where a check
    phrased through `succsOf` would rescan all 274 for every label. -/
theorem forall_window {P : Nat → Prop} (lo n : Nat)
    (h : ∀ q ∈ (List.range n).map (· + lo), P q) :
    ∀ q, lo ≤ q → q < lo + n → P q := by
  intro q h1 h2
  refine h q ?_
  simp only [List.mem_map, List.mem_range]
  exact ⟨q - lo, by omega, by omega⟩

/-- **What one instruction establishes about a register holds everywhere below
    it, as long as nothing overwrites it and nothing branches in.**

    `[lo, hi]` is entered only at `lo`, by falling off the instruction at
    `lo - 1`; no instruction of `[lo, hi)` writes `r`.  Then whatever the entry
    instruction makes true of `r` is true of `r` at *every* state of *every*
    trace standing in the region.

    This is the region form of the `…_carried` family, and it is the cheap one.
    A carried lemma pays a `decide` per (site, distance) pair, and every one of
    those `decide`s re-evaluates the whole emitted array — reading a clamp twenty
    instructions below its `mov`, at four load sites, is four rescans of
    twenty-odd program points *each*.  This pays for the window once.  On the
    shipped kernel that was the difference between a 3½-minute file and a
    12-second one. -/
theorem reg_prop_in_region (p : Array SInstr) (r : String) (lo hi : Nat) (P : UInt64 → Prop)
    (hft : ∀ q, lo ≤ q → q ≤ hi → (p[q]?.map fallthroughOnlyB) = some true)
    (hnd : ∀ q, lo ≤ q → q < hi → (p[q]?.map (fun i => destOf i != some r)) = some true)
    (init : SState) (hinit : init.pc < lo)
    (hset : ∀ k : Nat, (siter p k init).pc = lo - 1 →
      ∀ l : Lane, P ((sstep p (siter p k init)).regs r l)) :
    ∀ (k : Nat), lo ≤ (siter p k init).pc → (siter p k init).pc ≤ hi →
      ∀ l : Lane, P ((siter p k init).regs r l) := by
  intro k
  induction k with
  | zero =>
      intro h _ _
      rw [show siter p 0 init = init from rfl] at h
      exact absurd h (by omega)
  | succ m ih =>
      intro hlo hhi l
      have hq := hft _ hlo hhi
      -- the region is fallthrough-only, so step `m` stood one below
      have hst : (siter p m init).pc + 1 = (siter p (m + 1) init).pc := by
        cases hi' : p[(siter p (m + 1) init).pc]? with
        | none => rw [hi'] at hq; exact absurd hq (by simp)
        | some i =>
            rw [hi'] at hq
            have hf : fallthroughOnlyB i = true := by simpa using hq
            exact pc_pred p (siter p m init) _ i hi'
              (fun n e => by rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
              (by intro e; rw [e] at hf; exact absurd hf (by simp [fallthroughOnlyB]))
              (by rw [← siter_succ])
      rw [siter_succ]
      rcases Nat.lt_or_ge (siter p m init).pc lo with hbelow | habove
      · exact hset m (by omega) l
      · have hnw : ∀ i, p[(siter p m init).pc]? = some i → destOf i ≠ some r := by
          intro j hj
          have hx := hnd _ habove (by omega)
          rw [hj] at hx
          simpa using hx
        rw [sstep_regs_frame p _ r hnw]
        exact ih habove (by omega) l

/-- **A constant set just above a region holds everywhere inside it** — the
    `mov r, imm c` instance of `reg_prop_in_region`. -/
theorem const_in_region (p : Array SInstr) (r : String) (lo hi c : Nat)
    (hset : p[lo - 1]? = some (.mov r (.imm c)))
    (hft : ∀ q, lo ≤ q → q ≤ hi → (p[q]?.map fallthroughOnlyB) = some true)
    (hnd : ∀ q, lo ≤ q → q < hi → (p[q]?.map (fun i => destOf i != some r)) = some true)
    (init : SState) (hinit : init.pc < lo) :
    ∀ (k : Nat), lo ≤ (siter p k init).pc → (siter p k init).pc ≤ hi →
      ∀ l : Lane, (siter p k init).regs r l = UInt64.ofNat c :=
  reg_prop_in_region p r lo hi (fun v => v = UInt64.ofNat c) hft hnd init hinit
    (fun k hk l => by
      rw [sstep, show p[(siter p k init).pc]? = some (.mov r (.imm c)) from by rw [hk]; exact hset]
      simp only [sstepInstr, SState.setReg, SState.setPc, SState.get, if_true])

/-- `min` never exceeds its right operand — the whole content of "the kernel
    clamps its own read pointer". -/
theorem min_run_le_right (a b : UInt64) : (SOp.run .min a b).toNat ≤ b.toNat := by
  simp only [SOp.run]
  split
  · exact UInt64.le_iff_toNat_le.mp ‹_›
  · exact Nat.le_refl _

-- ── Lane-uniformity ─────────────────────────────────────────────────────────

/-- Every register in `U` holds the same value in every lane. -/
def Unif (U : List String) (st : SState) : Prop :=
  ∀ r ∈ U, ∀ l l' : Lane, st.regs r l = st.regs r l'

def argUnif (U : List String) : SArg → Bool
  | .reg n => U.contains n
  | .imm _ => true

/-- The instruction computes its destination out of lane-uniform data only.
    `vote` qualifies unconditionally: a ballot is a warp-wide value written
    identically to every lane. -/
def srcUnif (U : List String) : SInstr → Bool
  | .mov _ a        => argUnif U a
  | .bin _ _ a b    => U.contains a && argUnif U b
  | .binr _ _ a b   => U.contains a && U.contains b
  | .cvt32 _ a      => U.contains a
  | .brev _ a       => U.contains a
  | .clz _ a        => U.contains a
  | .bnot _ a       => U.contains a
  | .setp _ _ a b   => U.contains a && argUnif U b
  | .andp _ a b     => U.contains a && U.contains b
  | .selp _ a b q   => U.contains a && U.contains b && U.contains q
  | .vote _ _       => true
  | _               => false

/-- An instruction is safe for `U` if it does not write into `U` at all, or
    computes what it writes from `U` alone. -/
def unifOK (U : List String) (i : SInstr) : Bool :=
  match destOf i with
  | none   => true
  | some d => !U.contains d || srcUnif U i

theorem sstep_unif (p : Array SInstr) (U : List String) (st : SState)
    (hok : ∀ i, p[st.pc]? = some i → unifOK U i = true) (h : Unif U st) :
    Unif U (sstep p st) := by
  intro r hr l l'
  have hrc : U.contains r = true := by simpa using hr
  have hreg : ∀ n, U.contains n = true → st.regs n l = st.regs n l' :=
    fun n hn => h n (by simpa using hn) l l'
  have harg : ∀ a : SArg, argUnif U a = true → st.get l a = st.get l' a := by
    intro a ha
    cases a with
    | reg n => exact hreg n (by simpa [argUnif] using ha)
    | imm v => rfl
  cases hi : p[st.pc]? with
  | none => rw [sstep, hi]; exact h r hr l l'
  | some i =>
      by_cases hdi : destOf i = some r
      · have hu := hok i hi
        rw [unifOK, hdi] at hu
        simp only [hrc, Bool.not_true, Bool.false_or] at hu
        have hst : sstep p st = sstepInstr p i st := by
          rw [sstep, hi]
          cases i <;> first | rfl | (rw [destOf] at hdi; exact absurd hdi (by simp))
        rw [hst]
        cases i
        case pos.mov d a =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [srcUnif] at hu
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
            exact harg a hu
        case pos.bin o d a b =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [srcUnif, Bool.and_eq_true] at hu
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
            rw [hreg a hu.1, harg b hu.2]
        case pos.binr o d a b =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [srcUnif, Bool.and_eq_true] at hu
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
            rw [hreg a hu.1, hreg b hu.2]
        case pos.cvt32 d a =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [srcUnif] at hu
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
            rw [hreg a hu]
        case pos.brev d a =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [srcUnif] at hu
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
            rw [hreg a hu]
        case pos.clz d a =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [srcUnif] at hu
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
            rw [hreg a hu]
        case pos.bnot d a =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [srcUnif] at hu
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
            rw [hreg a hu]
        case pos.setp c d a b =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [srcUnif, Bool.and_eq_true] at hu
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
            rw [hreg a hu.1, harg b hu.2]
        case pos.andp d a b =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [srcUnif, Bool.and_eq_true] at hu
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
            rw [hreg a hu.1, hreg b hu.2]
        case pos.selp d a b q =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [srcUnif, Bool.and_eq_true] at hu
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
            rw [hreg a hu.1.1, hreg b hu.1.2, hreg q hu.2]
        case pos.vote d q =>
            simp only [destOf, Option.some.injEq] at hdi; subst hdi
            simp only [sstepInstr, SState.setReg, SState.setPc, if_pos rfl, if_true]
        all_goals exact absurd hu (by simp [srcUnif])
      · have hfr : (sstep p st).regs r = st.regs r :=
          sstep_regs_frame p st r (fun j hj => by rw [hi] at hj; cases hj; exact hdi)
        rw [congrFun hfr l, congrFun hfr l']
        exact h r hr l l'

/-- **Lane-uniformity is a whole-program invariant** once the closure check
    passes on every instruction.

    The kernel's loop cursor `searchPos` is uniform, but nothing local says so:
    branches read lane 0, so the loop guard `searchPos < searchLim` constrains
    lane 0 and no other.  Recovering the guard for every lane is what this is
    for, and the closure set has to be taken all the way round the cycle —
    `searchPos ← litAnchor ← p0 + ml`, `p0 ← searchPos + fl`, `ml ← ml + adv`,
    with `fl`/`adv` uniform because they are `clz∘brev` of a ballot. -/
theorem siter_unif (p : Array SInstr) (U : List String)
    (hok : ∀ q : Nat, ∀ i, p[q]? = some i → unifOK U i = true)
    (st : SState) (h : Unif U st) : ∀ k, Unif U (siter p k st) :=
  fun k => siter_inv p (Unif U) (fun s hs => sstep_unif p U s (fun i hi => hok _ i hi) hs) k st h

/-- **The instruction a state arrived by, and the state it arrived from.**

    `pc_back`/`regs_back` say *where* the trace was `n` steps ago and which
    registers survived; this says *what it did* on the step after that.  It is
    the one shape that covers `vote`, `shfl`, `clz`, `brev`, `andp` and `setp`
    alike — each of those needs a `…_carried` lemma of its own otherwise, and
    they are all the same proof.  Unfolding `sstepInstr` at the use site is one
    `simp only` and costs no `decide`. -/
theorem pre_state (p : Array SInstr) (init : SState) (hinit : init.pc = 0)
    (q n : Nat) (i : SInstr)
    (hft : ∀ t, t < n + 1 → (p[q - t]?.map fallthroughOnlyB) = some true)
    (hpre : p[q - n - 1]? = some i) (hn : n + 1 ≤ q)
    (k : Nat) (hpc : (siter p k init).pc = q) :
    siter p (k - n) init = sstepInstr p i (siter p (k - n - 1) init)
    ∧ (siter p (k - n - 1) init).pc = q - n - 1
    ∧ n + 1 ≤ k := by
  obtain ⟨hk1, hpc1⟩ :=
    pc_back p init hinit q (n + 1) (by omega) hft k hpc
  obtain ⟨hk0, hpc0⟩ :=
    pc_back p init hinit q n (by omega) (fun t h => hft t (by omega)) k hpc
  rw [show k - (n + 1) = k - n - 1 from by omega] at hpc1
  refine ⟨?_, by rw [hpc1]; omega, by omega⟩
  obtain ⟨j, hj⟩ : ∃ j, k - n - 1 = j := ⟨_, rfl⟩
  rw [hj] at hpc1 ⊢
  rw [show k - n = j + 1 from by omega, siter_succ, sstep,
      show p[(siter p j init).pc]? = some i from by rw [hpc1]; exact hpre]
  cases i <;> rfl

-- ── The cursor cannot run backwards ─────────────────────────────────────────

/-- **A `UInt64` cursor advanced by `L` really advances, when `L` fits.**

    The emit loop's conclusion has the shape `op_after = op_before + ofNat L`, so
    `op_before ≤ op_after` is immediate *provided* the addition does not wrap —
    and that is exactly what the loop's own budget invariant
    (`outBase + op + 9*(inStride - anchor) < 2^64`) plus
    `LZ4WarpDSL.planBlockFrom_encode_le9` supply.

    This is the arithmetic brick that dissolves the apparent circularity in
    "`op` is monotone, so intermediate values are bounded by the final one":
    monotonicity in `UInt64` needs no-overflow, and no-overflow comes from the
    budget rather than from the bound being proved. -/
theorem toNat_add_ofNat_of_lt (a : UInt64) (L : Nat) (h : a.toNat + L < 2 ^ 64) :
    (a + UInt64.ofNat L).toNat = a.toNat + L ∧ a.toNat ≤ (a + UInt64.ofNat L).toNat := by
  have hL : (UInt64.ofNat L).toNat = L :=
    AlgorithmLib.LZ4Ptx.toNat_ofNat_lt L (by omega)
  have hadd := UInt64.toNat_add a (UInt64.ofNat L)
  rw [hL] at hadd
  rw [hadd, Nat.mod_eq_of_lt h]
  exact ⟨rfl, Nat.le_add_right _ _⟩

/-- Two `UInt64`s add without wrapping when their `Nat` sum fits. -/
theorem toNat_add_of_lt (a b : UInt64) (h : a.toNat + b.toNat < 2 ^ 64) :
    (a + b).toNat = a.toNat + b.toNat := by
  rw [UInt64.toNat_add, Nat.mod_eq_of_lt h]

-- ── Frames over a closed region of the program ──────────────────────────────

/-- The instruction at `q` does not write `r` (vacuously true past the end). -/
def noDestAt (p : Array SInstr) (r : String) (q : Nat) : Bool :=
  match p[q]? with
  | some i => destOf i != some r
  | none   => true

/-- The instruction at `q` does not branch below `lo`. -/
def noExitAt (p : Array SInstr) (lo q : Nat) : Bool :=
  match p[q]? with
  | some (.bra l)        => lo ≤ sfindLabel p l
  | some (.braif _ l)    => lo ≤ sfindLabel p l
  | some (.braifnot _ l) => lo ≤ sfindLabel p l
  | _                    => true

def noDestFrom (p : Array SInstr) (r : String) (lo : Nat) : Bool :=
  (List.range (p.size - lo)).all (fun t => noDestAt p r (lo + t))

def noExitBelow (p : Array SInstr) (lo : Nat) : Bool :=
  (List.range (p.size - lo)).all (fun t => noExitAt p lo (lo + t))

private theorem scan_at {p : Array SInstr} {lo : Nat} {g : Nat → Bool}
    (h : (List.range (p.size - lo)).all (fun t => g (lo + t)) = true)
    (q : Nat) (hq : lo ≤ q) (i : SInstr) (hi : p[q]? = some i) : g q = true := by
  have hlt : q < p.size := by
    rcases Nat.lt_or_ge q p.size with h' | h'
    · exact h'
    · rw [Array.getElem?_eq_none_iff.mpr h'] at hi; exact absurd hi (by simp)
  have := List.all_eq_true.mp h (q - lo) (List.mem_range.mpr (by omega))
  rwa [show lo + (q - lo) = q from by omega] at this

/-- **A register is constant from the moment the trace enters a region that
    neither writes it nor is left downwards.**

    `siter_regs_const` needs the register untouched by the *whole* program, which
    fails for anything the prologue sets up — `outBase` is written at exactly one
    program point and then never again.  This is the version that can say so:
    scan only the suffix, and check that nothing branches back below it
    (fallthrough cannot leave a suffix downwards, so branch targets are the only
    way out).

    Both side conditions are `Bool` scans of the emitted array, so instantiating
    this is two `decide`s. -/
theorem regs_const_from (p : Array SInstr) (r : String) (lo : Nat)
    (hno : noDestFrom p r lo = true) (hex : noExitBelow p lo = true)
    (st0 : SState) (h0 : lo ≤ st0.pc) :
    ∀ k, (siter p k st0).regs r = st0.regs r ∧ lo ≤ (siter p k st0).pc := by
  have hstep : ∀ s : SState, (s.regs r = st0.regs r ∧ lo ≤ s.pc) →
      ((sstep p s).regs r = st0.regs r ∧ lo ≤ (sstep p s).pc) := by
    rintro s ⟨hr, hpc⟩
    cases hi : p[s.pc]? with
    | none => rw [sstep, hi]; exact ⟨hr, hpc⟩
    | some i =>
        have hnw : destOf i ≠ some r := by
          have := scan_at hno s.pc hpc i hi
          simp only [noDestAt, hi] at this
          simpa using this
        have hbr : noExitAt p lo s.pc = true := scan_at hex s.pc hpc i hi
        simp only [noExitAt, hi] at hbr
        refine ⟨?_, ?_⟩
        · rw [sstep_regs_frame p s r (fun j hj => by rw [hi] at hj; cases hj; exact hnw), hr]
        · rw [sstep, hi]
          cases i <;>
            simp only [sstepInstr, SState.setReg, SState.setPc] <;>
            first
              | omega
              | (simpa using hbr)
              | (split <;> first | omega | simpa using hbr)
  exact fun k => siter_inv p (fun s => s.regs r = st0.regs r ∧ lo ≤ s.pc) hstep k st0 ⟨rfl, h0⟩

/-- Splitting a trace: `N + j` steps from the launch state is `j` steps from
    whatever `N` steps reach.  Needed to state a confinement property from the
    post-prologue state and transport it back to the launch state. -/
theorem siter_add (p : Array SInstr) (N j : Nat) (st : SState) :
    siter p (N + j) st = siter p j (siter p N st) := by
  induction j generalizing st with
  | zero => simp only [Nat.add_zero, siter]
  | succ t ih =>
      rw [show N + (t + 1) = (N + t) + 1 from by omega, siter_succ, ih, ← siter_succ]

/-- **The intermediate states of a simulated segment, named.**

    The simulation assembly states every segment as `∃ m ss', SReaches prog m ss
    ss' ∧ …`, which reads like an existential over paths — and I read it that
    way, concluding the endpoint form was lossy about what happens in between.
    It is not, and `LZ4Concurrent.sreaches_iff_siter` already said so: the
    machine is deterministic, so the endpoint statement pins down the *entire*
    trace.

    This is that fact in the form the rest of this file consumes.  `j` steps into
    a segment that reaches `ss'` in `m`, the machine stands at `siter p j ss`, so
    a segment-level simulation result composes with `regs_back`, `pc_back` and
    the carried-value lemmas directly — no per-step reformulation of the
    assembly is needed to talk about a store site inside a simulated block. -/
theorem sreaches_mid {p : Array SInstr} {m : Nat} {ss ss' : SState}
    (h : SReaches p m ss ss') (j : Nat) (hj : j ≤ m) :
    siter p (m - j) (siter p j ss) = ss' := by
  rw [← siter_add, show j + (m - j) = m from by omega]
  exact (sreaches_iff_siter p m ss ss').mp h

-- ── Per-step obligations over a simulated segment ───────────────────────────

/-- **`Q` holds at every state of an `n`-step run.**

    `RegConfined` is exactly this shape — a property of *every* step — while the
    simulation assembly speaks of segments.  `sreaches_mid` says the two are
    compatible; this is the predicate that carries an obligation across a segment
    boundary, and the three lemmas below are how a segment tree is descended
    without restating any simulation theorem.

    Deliberately indexed by the step count rather than by a state set: the count
    is what `SReaches` hands you, and `siter_add` is what splits it. -/
def AllSteps (p : Array SInstr) (Q : SState → Prop) (n : Nat) (ss : SState) : Prop :=
  ∀ j, j ≤ n → Q (siter p j ss)

theorem allSteps_at {p Q n ss} (h : AllSteps p Q n ss) (j : Nat) (hj : j ≤ n) :
    Q (siter p j ss) := h j hj

theorem allSteps_mono {p Q Q' n ss} (himp : ∀ st, Q st → Q' st)
    (h : AllSteps p Q n ss) : AllSteps p Q' n ss :=
  fun j hj => himp _ (h j hj)

/-- **The composition step.**  A segment that discharges `Q` for its own `n₁`
    steps, followed by one that discharges it from where the first ended,
    discharges it for the whole run.  This is `simSL'_seq` at the level of
    obligations, and it needs no coupling: `SReaches` is deterministic, so the
    second segment's start state is the first's `siter`. -/
theorem allSteps_seq {p Q n₁ n₂ ss ss'} (h₁ : AllSteps p Q n₁ ss)
    (hr : SReaches p n₁ ss ss') (h₂ : AllSteps p Q n₂ ss') :
    AllSteps p Q (n₁ + n₂) ss := by
  intro j hj
  rcases Nat.lt_or_ge j n₁ with hlt | hge
  · exact h₁ j (by omega)
  · have := h₂ (j - n₁) (by omega)
    rwa [← (sreaches_iff_siter p n₁ ss ss').mp hr, ← siter_add,
      show n₁ + (j - n₁) = j from by omega] at this

/-- **A segment that never stands at a guarded pc discharges a pc-conditioned
    obligation for free.**

    Most of the kernel's steps are not at a memory site, so most of the descent
    is this lemma rather than any argument about registers.  It is what makes the
    tree traversal finite work instead of work proportional to the trace. -/
theorem allSteps_off_site {p : Array SInstr} {S : Nat → Prop} {P : SState → Prop}
    {n : Nat} {ss : SState} (h : ∀ j, j ≤ n → ¬ S (siter p j ss).pc) :
    AllSteps p (fun st => S st.pc → P st) n ss :=
  fun j hj hmem => absurd hmem (h j hj)

/-- A one-instruction segment: check the obligation where it starts and where it
    lands.  The leaf case of the descent, mirroring `simSL'_single`. -/
theorem allSteps_one {p Q ss} (h0 : Q ss) (h1 : Q (sstep p ss)) : AllSteps p Q 1 ss := by
  intro j hj
  match j with
  | 0 => exact h0
  | 1 => exact h1

-- ── A straight-line region with a single escape ─────────────────────────────

/-- Every branch at `q` targets the one escape point `e`. -/
def stepShapeB (p : Array SInstr) (e q : Nat) : Bool :=
  match p[q]? with
  | some (.bra l)        => sfindLabel p l == e
  | some (.braif _ l)    => sfindLabel p l == e
  | some (.braifnot _ l) => sfindLabel p l == e
  | _                    => true

/-- **From `q` the machine can only reach `q + 1`, the escape `e`, or stay put.**

    With `stepShapeB` decided over a prefix of the program, this gives the shape
    of the prologue's pc trace — `pc k = k` until the out-of-range guard fires,
    and `pc = e` (the `OOB` label) afterwards — without simulating it. -/
theorem pc_next (p : Array SInstr) (e q : Nat) (st : SState) (hpc : st.pc = q)
    (h : stepShapeB p e q = true) :
    (sstep p st).pc = q + 1 ∨ (sstep p st).pc = e ∨ (sstep p st).pc = q := by
  subst hpc
  rw [sstep]
  cases hi : p[st.pc]? with
  | none => exact Or.inr (Or.inr rfl)
  | some i =>
      simp only [stepShapeB, hi] at h
      cases i <;>
        first
          | exact Or.inl rfl
          | exact Or.inr (Or.inr rfl)
          | exact Or.inr (Or.inl (by simpa using h))
          | (rename_i pr lb
             by_cases hc : (st.regs pr 0 == 1) = true
             · first
                 | exact Or.inr (Or.inl (by
                     show (if (st.regs pr 0 == 1) = true then sfindLabel p lb
                             else st.pc + 1) = e
                     rw [if_pos hc]; simpa using h))
                 | exact Or.inl (by
                     show (if (st.regs pr 0 == 1) = true then st.pc + 1
                             else sfindLabel p lb) = st.pc + 1
                     rw [if_pos hc])
             · first
                 | exact Or.inl (by
                     show (if (st.regs pr 0 == 1) = true then sfindLabel p lb
                             else st.pc + 1) = st.pc + 1
                     rw [if_neg hc])
                 | exact Or.inr (Or.inl (by
                     show (if (st.regs pr 0 == 1) = true then st.pc + 1
                             else sfindLabel p lb) = e
                     rw [if_neg hc]; simpa using h)))

-- ── Control flow as data: where a pc can go next ─────────────────────────────

/-- **The program points an instruction at `q` can transfer control to.**  Read
    off the emitted array, so it is decidable at a concrete program: a branch
    contributes its resolved label and (for the conditional forms) the
    fallthrough, `ret` and an out-of-range pc are sinks, everything else falls
    through.

    This is what makes pc-shape facts cheap.  Every earlier pc argument in this
    development (`prologue_pc_shape`, `noExitBelow`) had to be hand-rolled for
    the region it talked about; `succsOf` plus `pc_in_closed` below decides them
    from the program instead. -/
def succsOf (p : Array SInstr) (q : Nat) : List Nat :=
  match p[q]? with
  | none => [q]
  | some (.bra L) => [sfindLabel p L]
  | some (.braif _ L) => [q + 1, sfindLabel p L]
  | some (.braifnot _ L) => [q + 1, sfindLabel p L]
  | some .ret => [q]
  | some _ => [q + 1]

/-- One step lands on a successor.  The whole content is that `sstepInstr`'s pc
    is `pc + 1` for the twenty non-branching constructors. -/
theorem sstep_pc_mem_succs (p : Array SInstr) (st : SState) :
    (sstep p st).pc ∈ succsOf p st.pc := by
  unfold succsOf
  cases hi : p[st.pc]? with
  | none => simp [sstep, hi]
  | some i =>
      simp only [sstep, hi]
      cases i <;>
        simp only [sstepInstr, SState.setPc, SState.setReg, List.mem_cons, List.mem_singleton,
          List.not_mem_nil, or_false] <;>
        first
          | rfl
          | exact Or.inl rfl
          | (split <;> simp)

/-- `S` is closed under control flow, except at the `exits` where the trace
    leaves.  A region can have several exits: the `loopC` body is left both by
    falling into the token emit and by the guard branching to the tail. -/
abbrev PcClosed (p : Array SInstr) (S exits : List Nat) : Prop :=
  ∀ q ∈ S, q ∈ exits ∨ ∀ q' ∈ succsOf p q, q' ∈ S

/-- **A trace cannot leave a closed region except through `exit`.**  Combined
    with a `decide`d `PcClosed` for a concrete program this replaces a bespoke
    pc-shape induction with an enumeration. -/
theorem pc_in_closed (p : Array SInstr) (S exits : List Nat)
    (hcl : PcClosed p S exits) (st : SState) (h0 : st.pc ∈ S) :
    ∀ k, (∀ j, j < k → (siter p j st).pc ∉ exits) → (siter p k st).pc ∈ S := by
  intro k
  induction k with
  | zero => intro _; exact h0
  | succ n ih =>
      intro hne
      have hn : (siter p n st).pc ∈ S := ih (fun j hj => hne j (by omega))
      rcases hcl _ hn with he | hs
      · exact absurd he (hne n (by omega))
      · rw [siter_succ]; exact hs _ (sstep_pc_mem_succs p _)

/-- **A register no instruction in a closed region writes is constant on it.**
    The region form of `siter_regs_const`: `noDestOn` is decided by scanning only
    the pcs in `S`, so a register written elsewhere in the program is still
    constant here. -/
theorem regs_const_on (p : Array SInstr) (r : String) (S exits : List Nat)
    (hcl : PcClosed p S exits)
    (hnd : ∀ q ∈ S, (p[q]?.map (fun i => destOf i != some r)) = some true)
    (st : SState) (h0 : st.pc ∈ S) :
    ∀ k, (∀ j, j < k → (siter p j st).pc ∉ exits) →
      (siter p k st).regs r = st.regs r := by
  intro k
  induction k with
  | zero => intro _; rfl
  | succ n ih =>
      intro hne
      have hn : (siter p n st).pc ∈ S := pc_in_closed p S exits hcl st h0 n
        (fun j hj => hne j (by omega))
      have hnw : ∀ i, p[(siter p n st).pc]? = some i → destOf i ≠ some r := by
        intro i hj
        have h := hnd _ hn
        rw [hj] at h
        simpa using h
      rw [siter_succ, sstep_regs_frame p (siter p n st) r hnw,
        ih (fun j hj => hne j (by omega))]

/-- **A potential argument on the machine, restricted to a region.**

    This is the shape the output-cursor bound takes.  `op ≤ lenOff` is *not*
    preserved by a single step — the very next instruction may be `add op, op, 1`
    — and `UInt64` monotonicity is unavailable without a no-wrap bound, which is
    what is being proven.  What *is* preserved is `op + (what is still to be
    emitted) ≤ lenOff`, and inside one token or one final literal run "what is
    still to be emitted" is a function of the pc and the machine's own registers.
    So the bound becomes a potential that never increases, and `siter_inv` moves
    it along the trace with no reference to the simulation assembly at all. -/
theorem potential_on (p : Array SInstr) (Φ : SState → Nat) (S exits : List Nat)
    (hcl : PcClosed p S exits)
    (hstep : ∀ st : SState, st.pc ∈ S → st.pc ∉ exits → Φ (sstep p st) ≤ Φ st)
    (st : SState) (h0 : st.pc ∈ S) :
    ∀ k, (∀ j, j < k → (siter p j st).pc ∉ exits) → Φ (siter p k st) ≤ Φ st := by
  intro k
  induction k with
  | zero => intro _; exact Nat.le_refl _
  | succ n ih =>
      intro hne
      have hn : (siter p n st).pc ∈ S := pc_in_closed p S exits hcl st h0 n
        (fun j hj => hne j (by omega))
      rw [siter_succ]
      exact Nat.le_trans (hstep _ hn (hne n (by omega))) (ih (fun j hj => hne j (by omega)))

/-- **A region invariant, not just a potential.**  `potential_on`'s `Φ` cannot
    carry the loop guards a wrap-free arithmetic step needs (`litExtra ≥ 255`
    before `litExtra -= 255`), so the general form takes an arbitrary `P`
    preserved on the region.  `siter_inv` will not do here: `P` is conditioned on
    being *inside* the region, and a state just outside it steps in without
    establishing the entry facts — the closure is what rules that out. -/
theorem inv_on (p : Array SInstr) (P : SState → Prop) (S exits : List Nat)
    (hcl : PcClosed p S exits)
    (hstep : ∀ st : SState, st.pc ∈ S → st.pc ∉ exits → P st → P (sstep p st))
    (st : SState) (h0 : st.pc ∈ S) (hP : P st) :
    ∀ k, (∀ j, j < k → (siter p j st).pc ∉ exits) → P (siter p k st) := by
  intro k
  induction k with
  | zero => intro _; exact hP
  | succ n ih =>
      intro hne
      have hn : (siter p n st).pc ∈ S := pc_in_closed p S exits hcl st h0 n
        (fun j hj => hne j (by omega))
      rw [siter_succ]
      exact hstep _ hn (hne n (by omega)) (ih (fun j hj => hne j (by omega)))

/-- Weakening a per-step obligation.  Needed to compose pc-confinement across a
    `seq`: each sub-segment proves confinement to *its own* range, and the
    composite wants confinement to the union, which is the weaker claim. -/
theorem allSteps_weaken {p : Array SInstr} {P P' : SState → Prop} {n : Nat} {ss : SState}
    (himp : ∀ st, P st → P' st) (h : AllSteps p P n ss) : AllSteps p P' n ss :=
  fun j hj => himp _ (h j hj)

/-! ### Linear-traversal shape checks

`p[q]?` costs O(n) per lookup during kernel reduction, so a `decide` over
`(List.range n).all (fun q => p[q]? …)` is O(n²) — measured at 7.4s for a
274-instruction program, against 0.6s for one traversal of the element list.
`noDestIn` states the same fact by traversal; `noDestIn_at` converts it back to
the pc-indexed form the trace proofs consume. -/

/-- No instruction in the window `[b, b+k)` writes `r`, checked by one traversal. -/
def noDestIn (p : Array SInstr) (b k : Nat) (r : String) : Bool :=
  ((p.toList.drop b).take k).all (fun i => destOf i != some r)

theorem noDestIn_at (p : Array SInstr) (b k : Nat) (r : String)
    (h : noDestIn p b k r = true) (hsz : b + k ≤ p.size) :
    ∀ q, b ≤ q → q < b + k → (p[q]?.map (fun i => destOf i != some r)) = some true := by
  intro q hb hk
  have hq : q < p.size := by omega
  have hget : p[q]? = some p[q] := getElem?_pos p q hq
  rw [hget, Option.map_some]
  have hmem : p[q] ∈ ((p.toList.drop b).take k) := by
    have h1 : ((p.toList.drop b).take k)[q - b]? = some p[q] := by
      simp only [List.getElem?_take, List.getElem?_drop, if_pos (show q - b < k by omega)]
      rw [show b + (q - b) = q by omega]
      simpa using hget
    exact List.mem_of_getElem? h1
  simpa using List.all_eq_true.mp h _ hmem

/-- `succsOf` computed from the instruction in hand, without re-indexing `p`. -/
def succsOfI (p : Array SInstr) (q : Nat) : SInstr → List Nat
  | .bra L => [sfindLabel p L]
  | .braif _ L => [q + 1, sfindLabel p L]
  | .braifnot _ L => [q + 1, sfindLabel p L]
  | .ret => [q]
  | _ => [q + 1]

theorem succsOf_eq_succsOfI (p : Array SInstr) (q : Nat) (i : SInstr)
    (h : p[q]? = some i) : succsOf p q = succsOfI p q i := by
  unfold succsOf; rw [h]; cases i <;> rfl

/-- A whole-program successor scan, by one traversal instead of 274 indexings.
    `(List.range n).all (fun q => (succsOf p q).all …)` re-indexes `p` at every
    step and is O(n²); this is O(n). -/
def cfgAll (p : Array SInstr) (f : Nat → Nat → Bool) : Bool :=
  p.toList.zipIdx.all (fun x => (succsOfI p x.2 x.1).all (f x.2))

theorem cfgAll_at (p : Array SInstr) (f : Nat → Nat → Bool) (h : cfgAll p f = true)
    (q : Nat) (hq : q < p.size) : (succsOf p q).all (f q) = true := by
  have hget : p[q]? = some p[q] := getElem?_pos p q hq
  rw [succsOf_eq_succsOfI p q p[q] hget]
  have hmem : (p[q], q) ∈ p.toList.zipIdx := by
    have h1 : (p.toList.zipIdx)[q]? = some (p[q], q) := by
      simp only [List.getElem?_zipIdx, Nat.zero_add]
      simpa using hget
    exact List.mem_of_getElem? h1
  simpa using List.all_eq_true.mp h _ hmem

private theorem bool_ext {a b : Bool} (h : a = true ↔ b = true) : a = b := by
  cases a <;> cases b <;> simp at h ⊢

/-- **The two forms of a program-wide successor scan agree.**  Rewriting with this
    lets a shape predicate keep its pc-indexed statement — which the trace proofs
    consume — while `decide` evaluates the linear form.  Pure build-time; the fact
    stated is unchanged. -/
theorem range_succs_eq_cfgAll (p : Array SInstr) (n : Nat) (hn : n = p.size)
    (f : Nat → Nat → Bool) :
    (List.range n).all (fun q => (succsOf p q).all (f q)) = cfgAll p f := by
  apply bool_ext
  constructor
  · intro h
    apply List.all_eq_true.mpr
    intro x hx
    obtain ⟨q, hq, hgx⟩ := List.mem_iff_getElem.mp hx
    have hlen : q < p.toList.length := by simpa using hq
    have hxq : x = (p[q]'(by simpa using hlen), q) := by
      rw [← hgx, List.getElem_zipIdx hq]; simp
    subst hxq
    have hqs : q < p.size := by simpa using hlen
    have := List.all_eq_true.mp h q (List.mem_range.mpr (by omega))
    rwa [succsOf_eq_succsOfI p q _ (getElem?_pos p q hqs)] at this
  · intro h
    apply List.all_eq_true.mpr
    intro q hq
    exact cfgAll_at p f h q (by have := List.mem_range.mp hq; omega)

/-- A window scan by one traversal: the instructions at pcs `[b, b+k)`. -/
def winAll (p : Array SInstr) (b k : Nat) (f : SInstr → Bool) : Bool :=
  ((p.toList.drop b).take k).all f

/-- **The two forms of a window scan agree.**  Same role as `range_succs_eq_cfgAll`,
    for the `p[q]?.map f == some true` idiom. -/
theorem range_win_eq (p : Array SInstr) (b k : Nat) (hsz : b + k ≤ p.size)
    (f : SInstr → Bool) :
    ((List.range k).map (· + b)).all (fun q => p[q]?.map f == some true)
      = winAll p b k f := by
  apply bool_ext
  constructor
  · intro h
    apply List.all_eq_true.mpr
    intro x hx
    obtain ⟨j, hj, hgx⟩ := List.mem_iff_getElem.mp hx
    have hjk : j < k := by
      have := hj; simp only [List.length_take, List.length_drop] at this; omega
    have hxq : x = p[b + j]'(by omega) := by
      rw [← hgx, List.getElem_take, List.getElem_drop]; simp
    subst hxq
    have := List.all_eq_true.mp h (j + b)
      (List.mem_map.mpr ⟨j, List.mem_range.mpr hjk, rfl⟩)
    rw [show j + b = b + j by omega, getElem?_pos p (b + j) (by omega)] at this
    simpa using this
  · intro h
    apply List.all_eq_true.mpr
    intro q hq
    obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hq
    have hjk : j < k := List.mem_range.mp hj
    have hqs : j + b < p.size := by omega
    rw [getElem?_pos p (j + b) hqs]
    have hmem : p[j + b] ∈ ((p.toList.drop b).take k) := by
      have h1 : ((p.toList.drop b).take k)[j]? = some p[j + b] := by
        simp only [List.getElem?_take, List.getElem?_drop, if_pos hjk]
        rw [show b + j = j + b by omega]
        simpa using getElem?_pos p (j + b) hqs
      exact List.mem_of_getElem? h1
    simpa using List.all_eq_true.mp h _ hmem

/-- `noDestFrom` by one traversal.  Same statement, so consumers are untouched;
    the indexed form re-reads the array at every step and is O(n²). -/
theorem noDestFrom_eq (p : Array SInstr) (r : String) (lo : Nat) (hlo : lo ≤ p.size) :
    noDestFrom p r lo = winAll p lo (p.size - lo) (fun i => destOf i != some r) := by
  apply bool_ext
  constructor
  · intro h
    apply List.all_eq_true.mpr
    intro x hx
    obtain ⟨j, hj, hgx⟩ := List.mem_iff_getElem.mp hx
    have hjk : j < p.size - lo := by
      have := hj
      simp only [List.length_take, List.length_drop, Array.length_toList] at this; omega
    have hxq : x = p[lo + j]'(by omega) := by
      rw [← hgx, List.getElem_take, List.getElem_drop]; simp
    subst hxq
    have := List.all_eq_true.mp h j (List.mem_range.mpr hjk)
    simpa [noDestAt, getElem?_pos p (lo + j) (show lo + j < p.size by omega)] using this
  · intro h
    apply List.all_eq_true.mpr
    intro t ht
    have htk : t < p.size - lo := List.mem_range.mp ht
    have hin : lo + t < p.size := by omega
    have hmem : p[lo + t]'hin ∈ ((p.toList.drop lo).take (p.size - lo)) := by
      have h1 : ((p.toList.drop lo).take (p.size - lo))[t]? = some p[lo + t] := by
        simp only [List.getElem?_take, List.getElem?_drop, if_pos htk]
        simpa using getElem?_pos p (lo + t) hin
      exact List.mem_of_getElem? h1
    simpa [noDestAt, getElem?_pos p (lo + t) hin] using List.all_eq_true.mp h _ hmem

/-- The branch-target test, applied to an instruction in hand. -/
def noExitInstr (p : Array SInstr) (lo : Nat) : SInstr → Bool
  | .bra l        => lo ≤ sfindLabel p l
  | .braif _ l    => lo ≤ sfindLabel p l
  | .braifnot _ l => lo ≤ sfindLabel p l
  | _             => true

/-- `noExitBelow` by one traversal.  Same statement; consumers untouched. -/
theorem noExitBelow_eq (p : Array SInstr) (lo : Nat) (hlo : lo ≤ p.size) :
    noExitBelow p lo = winAll p lo (p.size - lo) (noExitInstr p lo) := by
  apply bool_ext
  constructor
  · intro h
    apply List.all_eq_true.mpr
    intro x hx
    obtain ⟨j, hj, hgx⟩ := List.mem_iff_getElem.mp hx
    have hjk : j < p.size - lo := by
      have := hj
      simp only [List.length_take, List.length_drop, Array.length_toList] at this; omega
    have hin : lo + j < p.size := by omega
    have hxq : x = p[lo + j]'hin := by
      rw [← hgx, List.getElem_take, List.getElem_drop]; simp
    subst hxq
    have := List.all_eq_true.mp h j (List.mem_range.mpr hjk)
    rw [noExitAt, getElem?_pos p (lo + j) hin] at this
    revert this
    cases p[lo + j] <;> exact id
  · intro h
    apply List.all_eq_true.mpr
    intro s hs
    have hsk : s < p.size - lo := List.mem_range.mp hs
    have hin : lo + s < p.size := by omega
    have hmem : p[lo + s]'hin ∈ ((p.toList.drop lo).take (p.size - lo)) := by
      have h1 : ((p.toList.drop lo).take (p.size - lo))[s]? = some p[lo + s] := by
        simp only [List.getElem?_take, List.getElem?_drop, if_pos hsk]
        simpa using getElem?_pos p (lo + s) hin
      exact List.mem_of_getElem? h1
    have := List.all_eq_true.mp h _ hmem
    rw [noExitAt, getElem?_pos p (lo + s) hin]
    revert this
    cases p[lo + s] <;> exact id

/-- A window scan whose predicate is guarded by the pc, by one traversal. -/
def winAllG (p : Array SInstr) (lo n : Nat) (P : Nat → Bool) (f : SInstr → Bool) : Bool :=
  (((p.toList.drop lo).take n).zipIdx).all (fun x => !P (lo + x.2) || f x.1)

/-- **The guarded window scan, indexed and traversal forms, agree.** -/
theorem range_winG_eq (p : Array SInstr) (lo n : Nat) (hsz : lo + n ≤ p.size)
    (P : Nat → Bool) (f : SInstr → Bool) :
    ((List.range n).map (· + lo)).all (fun q => !P q || (p[q]?.map f == some true))
      = winAllG p lo n P f := by
  apply bool_ext
  constructor
  · intro h
    apply List.all_eq_true.mpr
    intro x hx
    obtain ⟨j, hj, hgx⟩ := List.mem_iff_getElem.mp hx
    have hjk : j < n := by
      have := hj
      simp only [List.length_zipIdx, List.length_take, List.length_drop,
        Array.length_toList] at this
      omega
    have hin : lo + j < p.size := by omega
    have hxq : x = (p[lo + j]'hin, j) := by
      rw [← hgx, List.getElem_zipIdx, List.getElem_take, List.getElem_drop]; simp
    subst hxq
    have := List.all_eq_true.mp h (j + lo)
      (List.mem_map.mpr ⟨j, List.mem_range.mpr hjk, rfl⟩)
    rw [show j + lo = lo + j by omega, getElem?_pos p (lo + j) hin] at this
    simpa using this
  · intro h
    apply List.all_eq_true.mpr
    intro q hq
    obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hq
    have hjk : j < n := List.mem_range.mp hj
    have hin : j + lo < p.size := by omega
    have hmem : (p[j + lo]'hin, j) ∈ (((p.toList.drop lo).take n).zipIdx) := by
      have h1 : (((p.toList.drop lo).take n).zipIdx)[j]? = some (p[j + lo], j) := by
        simp only [List.getElem?_zipIdx, List.getElem?_take, List.getElem?_drop,
          if_pos hjk, Nat.zero_add]
        rw [show lo + j = j + lo by omega]
        simpa using getElem?_pos p (j + lo) hin
      exact List.mem_of_getElem? h1
    have := List.all_eq_true.mp h _ hmem
    rw [getElem?_pos p (j + lo) hin]
    rw [show lo + j = j + lo by omega] at this
    simpa using this

end AlgorithmLib.LZ4Simt
