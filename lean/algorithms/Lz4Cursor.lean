import Lz4Sites

set_option maxRecDepth 8192

/-!
  # `CursorAtSites.opLe`, and the region machinery it needs

  Split from `Lz4Sites` for a mechanical reason: past roughly 8 MB of olean the
  module's serializer overflows its stack (`exit 134`, no line number, no relation
  to any proof).  Two modules under the threshold write fine.
-/

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt

-- ── The last entry into a region ─────────────────────────────────────────────

/-- **Where a region was entered.**  `AllSteps` gives a property at every step; an
    invariant like `TokInv` needs the state where the region was ENTERED, which is
    a backwards-looking fact.  This supplies it: for a region with a single entry
    point, any step standing inside it descends from a step standing at that entry,
    with the whole stretch between confined to the region.

    The proof is the obvious induction, and the single-entry condition is decided
    against the emitted array — it is a statement about the control-flow graph, not
    about reachability. -/
theorem region_entry (p : Array SInstr) (S : List Nat) (e : Nat)
    (hen : ∀ q, q ∉ S → ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf p q → q' ∈ S → q' = e)
    (ss : SState) (h0 : ss.pc ∉ S) :
    ∀ k, (siter p k ss).pc ∈ S →
      ∃ j, j ≤ k ∧ (siter p j ss).pc = e ∧ ∀ i, j ≤ i → i ≤ k → (siter p i ss).pc ∈ S := by
  intro k
  induction k with
  | zero => intro h; exact absurd h h0
  | succ m ih =>
      intro h
      by_cases hm : (siter p m ss).pc ∈ S
      · obtain ⟨j, hj, hje, hall⟩ := ih hm
        refine ⟨j, by omega, hje, fun i h1 h2 => ?_⟩
        rcases Nat.lt_or_ge i (m + 1) with hlt | hge
        · exact hall i h1 (by omega)
        · rw [show i = m + 1 from by omega]; exact h
      · have hstep : (siter p (m + 1) ss).pc ∈ AlgorithmLib.LZ4Simt.succsOf p (siter p m ss).pc := by
          rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs p _
        refine ⟨m + 1, Nat.le_refl _, hen _ hm _ hstep h, fun i h1 h2 => ?_⟩
        rw [show i = m + 1 from by omega]; exact h

/-- The token emit together with its `wEmitMatchSeq` prologue: pcs 124–198.  The
    only edge into it from outside is `123 → 124`. -/
def tokRegion : List Nat := (List.range 75).map (· + 124)

theorem tokRegion_entry_lt : ∀ q, q < 274 → q ∉ tokRegion →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K q, q' ∈ tokRegion → q' = 124 :=
  ivEntry_at K 124 75 124 shipped32_size (by decide)

theorem tokRegion_entry : ∀ q, q ∉ tokRegion →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K q → q' ∈ tokRegion → q' = 124 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact tokRegion_entry_lt q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

-- ── The body region, and that it is entered exactly once ─────────────────────

/-- Everything between the loop head and the out-of-bounds exit: pcs 40–271.
    All ten store sites lie inside it. -/
def bodyRegion : List Nat := (List.range 232).map (· + 40)

theorem bodyRegion_exit : ∀ q ∈ bodyRegion,
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K q, q' ∈ bodyRegion ∨ 272 ≤ q' :=
  ivExit_at K 40 232 272 shipped32_size (by omega) (by decide)

theorem bodyRegion_entry_lt : ∀ q, q < 274 → q ∉ bodyRegion →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K q, q' ∈ bodyRegion → q' = 40 :=
  ivEntry_at K 40 232 40 shipped32_size (by decide)

theorem bodyRegion_entry : ∀ q, q ∉ bodyRegion →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K q → q' ∈ bodyRegion → q' = 40 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact bodyRegion_entry_lt q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

/-- The four up-closure facts this file needs, in ONE `decide`: the emitted
    array is built once instead of four times, and each scan is a traversal
    rather than 274 indexings. -/
def cursorUpB (p : Array SInstr) : Bool :=
  upClosedB p 208 && upClosedB p 217 && upClosedB p 235 && upClosedB p 272

theorem cursorUp_true : cursorUpB K = true := by decide

theorem beyond_step_lt : ∀ q, q < 274 → 272 ≤ q →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K q, 272 ≤ q' :=
  upClosed_at K 272 shipped32_size (by
    have h := cursorUp_true; simp only [cursorUpB, Bool.and_eq_true] at h; exact h.2)

/-- Past the length-store tail there is no way back: `272` is a label, `273` a
    `ret`, and beyond the array every pc is its own successor. -/
theorem stays_beyond (ss : SState) (a : Nat) (h : 272 ≤ (siter K a ss).pc) :
    ∀ b, a ≤ b → 272 ≤ (siter K b ss).pc := by
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact h
      · have hm := ih hge
        have hs : (siter K (m + 1) ss).pc
            ∈ AlgorithmLib.LZ4Simt.succsOf K (siter K m ss).pc := by
          rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K _
        rcases Nat.lt_or_ge (siter K m ss).pc 274 with hlt2 | hge2
        · exact beyond_step_lt _ hlt2 hm _ hs
        · rw [show AlgorithmLib.LZ4Simt.succsOf K (siter K m ss).pc = [(siter K m ss).pc] from by
            simp only [AlgorithmLib.LZ4Simt.succsOf,
              Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)],
            List.mem_singleton] at hs
          omega

/-- Once inside the body region the machine is inside it or past it, forever. -/
theorem region_or_beyond (ss : SState) (a : Nat) (h : (siter K a ss).pc ∈ bodyRegion) :
    ∀ b, a ≤ b → (siter K b ss).pc ∈ bodyRegion ∨ 272 ≤ (siter K b ss).pc := by
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact Or.inl h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact Or.inl h
      · have hs : (siter K (m + 1) ss).pc
            ∈ AlgorithmLib.LZ4Simt.succsOf K (siter K m ss).pc := by
          rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K _
        rcases ih hge with hin | hbe
        · exact bodyRegion_exit _ hin _ hs
        · exact Or.inr (stays_beyond ss m hbe (m + 1) (by omega))

/-- **The body region is entered exactly once, at the step after the prologue.**

    No trajectory argument for the prologue is needed: `39` is not a successor of
    anything in the region and the region is never re-entered once left, so a
    machine that stands at `39` after `P` steps and inside the region at step `k`
    must have `k > P`, and every step in between is inside.  That is exactly the
    hypothesis shape the region invariants want. -/
theorem body_entry_at (ss : SState) (P : Nat) (hP : (siter K P ss).pc = 39)
    (k : Nat) (hk : (siter K k ss).pc ∈ bodyRegion) :
    P + 1 ≤ k ∧ (siter K (P + 1) ss).pc = 40
      ∧ ∀ i, P + 1 ≤ i → i ≤ k → (siter K i ss).pc ∈ bodyRegion := by
  have h39 : AlgorithmLib.LZ4Simt.succsOf K 39 = [40] := by decide
  have hpc1 : (siter K (P + 1) ss).pc = 40 := by
    have hs : (siter K (P + 1) ss).pc ∈ AlgorithmLib.LZ4Simt.succsOf K (siter K P ss).pc := by
      rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K _
    rw [hP, h39, List.mem_singleton] at hs; exact hs
  have hmem1 : (siter K (P + 1) ss).pc ∈ bodyRegion := by rw [hpc1]; decide
  have hkP : P + 1 ≤ k := by
    rcases Nat.lt_or_ge k (P + 1) with hlt | hge
    · exfalso
      rcases region_or_beyond ss k hk P (by omega) with e | e
      · rw [hP] at e; exact absurd e (by decide)
      · rw [hP] at e; omega
    · exact hge
  refine ⟨hkP, hpc1, fun i h1 h2 => ?_⟩
  rcases region_or_beyond ss (P + 1) hmem1 i h1 with e | e
  · exact e
  · exfalso
    have hbk := stays_beyond ss i e k h2
    have hle : ∀ q ∈ bodyRegion, q ≤ 271 := by decide
    have := hle _ hk
    omega

/-- From the `loopC` exit onward the pc never drops below 208: the tail's only
    back edge is the LSIC loop's `230 → 222`.  This is what says a step past the
    loop's step count cannot be at the match-sequence entry. -/
theorem tail_step_lt : ∀ q, q < 274 → 208 ≤ q →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K q, 208 ≤ q' :=
  upClosed_at K 208 shipped32_size (by
    have h := cursorUp_true; simp only [cursorUpB, Bool.and_eq_true] at h; exact h.1.1.1)

theorem stays_from_208 (ss : SState) (a : Nat) (h : 208 ≤ (siter K a ss).pc) :
    ∀ b, a ≤ b → 208 ≤ (siter K b ss).pc := by
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact h
      · have hm := ih hge
        have hs : (siter K (m + 1) ss).pc
            ∈ AlgorithmLib.LZ4Simt.succsOf K (siter K m ss).pc := by
          rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K _
        rcases Nat.lt_or_ge (siter K m ss).pc 274 with hlt2 | hge2
        · exact tail_step_lt _ hlt2 hm _ hs
        · rw [show AlgorithmLib.LZ4Simt.succsOf K (siter K m ss).pc = [(siter K m ss).pc] from by
            simp only [AlgorithmLib.LZ4Simt.succsOf,
              Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)],
            List.mem_singleton] at hs
          omega

/-- `SReaches` is `siter`: the relation is defined by the same recursion. -/
theorem sreaches_siter (p : Array SInstr) : ∀ (n : Nat) (a b : SState),
    AlgorithmLib.LZ4Simt.SReaches p n a b → siter p n a = b := by
  intro n
  induction n with
  | zero => intro a b h; exact h
  | succ m ih => intro a b h; rw [siter]; exact ih _ _ h

-- ── The loop checkpoint, at the shipped kernel from the launch state ─────────

/-- Prologue length: `head25`, the clear loop, the eight-instruction epilogue and
    the `loop` label. -/
def preSteps : Nat := 25 + 8 * AlgorithmLib.LZ4Simt.clearIters wHashLog + 8 + 1

/-- Recorded before `preSteps` is sealed. -/
theorem preSteps_eq : preSteps = 25 + 8 * AlgorithmLib.LZ4Simt.clearIters wHashLog + 8 + 1 := rfl

-- **Seal `preSteps`.**  Left reducible, the elaborator will happily `whnf` a
-- `siter K preSteps …` — 1058 machine steps over a 274-instruction array — which
-- is what makes the proof terms here enormous and the olean writer overflow.
-- Nothing downstream needs its value.
attribute [irreducible] preSteps

/-- **The prologue, in `siter` form.**  `prologue_couple` is stated with `snsteps`
    and a spelled-out step count; converting it is one rewrite of the whole
    conjunction (nine separate `rw ... at` calls make the term large enough to
    overflow the olean writer). -/
theorem tail_pc_bin (st : SState) (q : Nat) (o : SOp) (d a : String) (b : SArg) (hq : st.pc = q)
    (hi : K[q]? = some (.bin o d a b)) : (sstep K st).pc = q + 1 := by
  rw [sstep, show K[st.pc]? = some (.bin o d a b) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]

theorem tail_pc_mov (st : SState) (q : Nat) (d : String) (a : SArg) (hq : st.pc = q)
    (hi : K[q]? = some (.mov d a)) : (sstep K st).pc = q + 1 := by
  rw [sstep, show K[st.pc]? = some (.mov d a) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]

theorem tail_pc_setp (st : SState) (q : Nat) (c : SCmp) (d a : String) (b : SArg) (hq : st.pc = q)
    (hi : K[q]? = some (.setp c d a b)) : (sstep K st).pc = q + 1 := by
  rw [sstep, show K[st.pc]? = some (.setp c d a b) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]

theorem tail_pc_stg (st : SState) (q : Nat) (d a : String) (hq : st.pc = q)
    (hi : K[q]? = some (.stg d a)) : (sstep K st).pc = q + 1 := by
  rw [sstep, show K[st.pc]? = some (.stg d a) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]

theorem tail_pc_lbl (st : SState) (q : Nat) (L : String) (hq : st.pc = q)
    (hi : K[q]? = some (.lbl L)) : (sstep K st).pc = q + 1 := by
  rw [sstep, show K[st.pc]? = some (.lbl L) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]

theorem tail_frame (st : SState) (q : Nat) (i : SInstr) (r : String) (hq : st.pc = q)
    (hi : K[q]? = some i) (hne : AlgorithmLib.LZ4WarpDSL.wtgt i ≠ some r) :
    (sstep K st).regs r = st.regs r :=
  AlgorithmLib.LZ4WarpDSL.sstep_regs_ne K st r (fun i' hi' => by
    rw [show K[st.pc]? = some i from by rw [hq]; exact hi] at hi'
    cases hi'; exact hne)

/-- **The tail's token store.**  Seven straight steps from the loop exit reach the
    final token store at pc 216, and none of them writes the cursor — so the store
    happens at exactly the cursor the loop left. -/
theorem tail_to_216 (E : SState) (h209 : E.pc = 209) :
    (siter K 7 E).pc = 216 ∧ (siter K 7 E).regs "op" = E.regs "op" := by
  have e0 : siter K 0 E = E := rfl
  have p1 : (siter K 1 E).pc = 210 := by
    rw [siter_succ, e0]; exact tail_pc_mov E 209 "fLen" (SArg.imm 32768) h209 (by decide)
  have p2 : (siter K 2 E).pc = 211 := by
    rw [siter_succ]
    exact tail_pc_bin _ 210 (.sub) "fLen" "fLen" (SArg.reg "litAnchor") p1 (by decide)
  have p3 : (siter K 3 E).pc = 212 := by
    rw [siter_succ]; exact tail_pc_mov _ 211 "zero" (SArg.imm 0) p2 (by decide)
  have p4 : (siter K 4 E).pc = 213 := by
    rw [siter_succ]; exact tail_pc_bin _ 212 (.min) "tokHi" "fLen" (SArg.imm 15) p3 (by decide)
  have p5 : (siter K 5 E).pc = 214 := by
    rw [siter_succ]; exact tail_pc_bin _ 213 (.shl) "tok" "tokHi" (SArg.imm 4) p4 (by decide)
  have p6 : (siter K 6 E).pc = 215 := by
    rw [siter_succ]; exact tail_pc_bin _ 214 (.bor) "tok" "tok" (SArg.reg "zero") p5 (by decide)
  have p7 : (siter K 7 E).pc = 216 := by
    rw [siter_succ]
    exact tail_pc_bin _ 215 (.add) "sbAddr" "outBase" (SArg.reg "op") p6 (by decide)
  refine ⟨p7, ?_⟩
  have f1 : (siter K 1 E).regs "op" = E.regs "op" := by
    rw [siter_succ, e0]
    exact tail_frame E 209 (.mov "fLen" (SArg.imm 32768)) "op" h209 (by decide) (by decide)
  have f2 : (siter K 2 E).regs "op" = (siter K 1 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 210 (.bin (.sub) "fLen" "fLen" (SArg.reg "litAnchor")) "op" p1
      (by decide) (by decide)
  have f3 : (siter K 3 E).regs "op" = (siter K 2 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 211 (.mov "zero" (SArg.imm 0)) "op" p2 (by decide) (by decide)
  have f4 : (siter K 4 E).regs "op" = (siter K 3 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 212 (.bin (.min) "tokHi" "fLen" (SArg.imm 15)) "op" p3
      (by decide) (by decide)
  have f5 : (siter K 5 E).regs "op" = (siter K 4 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 213 (.bin (.shl) "tok" "tokHi" (SArg.imm 4)) "op" p4 (by decide) (by decide)
  have f6 : (siter K 6 E).regs "op" = (siter K 5 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 214 (.bin (.bor) "tok" "tok" (SArg.reg "zero")) "op" p5
      (by decide) (by decide)
  have f7 : (siter K 7 E).regs "op" = (siter K 6 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 215 (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) "op" p6
      (by decide) (by decide)
  rw [f7, f6, f5, f4, f3, f2, f1]

/-- Value read-off for the tail chain: what a step writes into its target. -/
theorem tail_val_bin (st : SState) (q : Nat) (o : SOp) (d a : String) (b : SArg) (hq : st.pc = q)
    (hi : K[q]? = some (.bin o d a b)) (l : Lane) :
    (sstep K st).regs d l = o.run (st.regs a l) (st.get l b) := by
  rw [sstep, show K[st.pc]? = some (.bin o d a b) from by rw [hq]; exact hi]
  simp only [sstepInstr, SState.setPc, SState.setReg, eq_self_iff_true, if_true]

theorem tail_val_mov (st : SState) (q : Nat) (d : String) (a : SArg) (hq : st.pc = q)
    (hi : K[q]? = some (.mov d a)) (l : Lane) :
    (sstep K st).regs d l = st.get l a := by
  rw [sstep, show K[st.pc]? = some (.mov d a) from by rw [hq]; exact hi]
  simp only [sstepInstr, SState.setPc, SState.setReg, eq_self_iff_true, if_true]

theorem tail_val_setp (st : SState) (q : Nat) (c : SCmp) (d a : String) (b : SArg) (hq : st.pc = q)
    (hi : K[q]? = some (.setp c d a b)) (l : Lane) :
    (sstep K st).regs d l = (if c.run (st.regs a l) (st.get l b) then 1 else 0) := by
  rw [sstep, show K[st.pc]? = some (.setp c d a b) from by rw [hq]; exact hi]
  simp only [sstepInstr, SState.setPc, SState.setReg, eq_self_iff_true, if_true]

-- ── Forward confinement in the tail, by interval ─────────────────────────────

/-- One forward-confinement induction, parameterized by the floor.  Instantiated
    at 210 (inside the tail nothing goes back to the loop), 217 (the token store
    happens once) and 235 (the LSIC run is never re-entered). -/
theorem stays_ge (b : Nat) (hb : ∀ q, q < 274 → b ≤ q → ∀ q' ∈ succsOf K q, b ≤ q')
    (ss : SState) (a : Nat) (h : b ≤ (siter K a ss).pc) :
    ∀ c, a ≤ c → b ≤ (siter K c ss).pc := by
  intro c
  induction c with
  | zero => intro hc; rw [show a = 0 from by omega] at h; exact h
  | succ m ih =>
      intro hc
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact h
      · have hm := ih hge
        have hs : (siter K (m + 1) ss).pc ∈ succsOf K (siter K m ss).pc := by
          rw [siter_succ]; exact sstep_pc_mem_succs K _
        rcases Nat.lt_or_ge (siter K m ss).pc 274 with hlt2 | hge2
        · exact hb _ hlt2 hm _ hs
        · rw [show succsOf K (siter K m ss).pc = [(siter K m ss).pc] from by
            simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)],
            List.mem_singleton] at hs
          omega

theorem stays_from_210 (ss : SState) (a : Nat) (h : 210 ≤ (siter K a ss).pc) :
    ∀ b, a ≤ b → 210 ≤ (siter K b ss).pc :=
  stays_ge 210 (by decide) ss a h

/-- From pc 217 on, the pc never drops below 217: the only back edge in the tail
    is the LSIC loop's `230 → 222`. -/
theorem stays_from_217 (ss : SState) (a : Nat) (h : 217 ≤ (siter K a ss).pc) :
    ∀ b, a ≤ b → 217 ≤ (siter K b ss).pc := by
  have hstep : ∀ q, q < 274 → 217 ≤ q → ∀ q' ∈ succsOf K q, 217 ≤ q' :=
    upClosed_at K 217 shipped32_size (by
      have h := cursorUp_true; simp only [cursorUpB, Bool.and_eq_true] at h; exact h.1.1.2)
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact h
      · have hm := ih hge
        have hs : (siter K (m + 1) ss).pc ∈ succsOf K (siter K m ss).pc := by
          rw [siter_succ]; exact sstep_pc_mem_succs K _
        rcases Nat.lt_or_ge (siter K m ss).pc 274 with hlt2 | hge2
        · exact hstep _ hlt2 hm _ hs
        · rw [show succsOf K (siter K m ss).pc = [(siter K m ss).pc] from by
            simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)],
            List.mem_singleton] at hs
          omega

/-- Likewise from 235 on — past the tail's LSIC run there is no way back into it. -/
theorem stays_from_235 (ss : SState) (a : Nat) (h : 235 ≤ (siter K a ss).pc) :
    ∀ b, a ≤ b → 235 ≤ (siter K b ss).pc := by
  have hstep : ∀ q, q < 274 → 235 ≤ q → ∀ q' ∈ succsOf K q, 235 ≤ q' :=
    upClosed_at K 235 shipped32_size (by
      have h := cursorUp_true; simp only [cursorUpB, Bool.and_eq_true] at h; exact h.1.2)
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact h
      · have hm := ih hge
        have hs : (siter K (m + 1) ss).pc ∈ succsOf K (siter K m ss).pc := by
          rw [siter_succ]; exact sstep_pc_mem_succs K _
        rcases Nat.lt_or_ge (siter K m ss).pc 274 with hlt2 | hge2
        · exact hstep _ hlt2 hm _ hs
        · rw [show succsOf K (siter K m ss).pc = [(siter K m ss).pc] from by
            simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)],
            List.mem_singleton] at hs
          omega

/-- The LSIC region `[222, 234]` is left only upward, to 235. -/
theorem lsic_region_exit : ∀ q, q < 235 → 222 ≤ q → ∀ q' ∈ succsOf K q,
    (222 ≤ q' ∧ q' ≤ 234) ∨ 235 ≤ q' := by decide

/-- Once in the LSIC region the machine is in it or past it. -/
theorem lsic_or_beyond (ss : SState) (a : Nat)
    (h : 222 ≤ (siter K a ss).pc ∧ (siter K a ss).pc ≤ 234) :
    ∀ b, a ≤ b → (222 ≤ (siter K b ss).pc ∧ (siter K b ss).pc ≤ 234)
      ∨ 235 ≤ (siter K b ss).pc := by
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact Or.inl h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact Or.inl h
      · have hs : (siter K (m + 1) ss).pc ∈ succsOf K (siter K m ss).pc := by
          rw [siter_succ]; exact sstep_pc_mem_succs K _
        rcases ih hge with hin | hbe
        · exact lsic_region_exit _ (by omega) hin.1 _ hs
        · exact Or.inr (stays_from_235 ss m hbe (m + 1) (by omega))

/-- **From the loop exit through the tail's token store to its LSIC head.**

    Both outcomes of the length-extension branch at pc 219, because the caller
    needs each: when the final literal run is at least 15 bytes the machine walks
    on to the LSIC loop head at 222 with `litExtraF`/`lsicC` set up, and when it
    is not, the branch leaves for 236 and the LSIC stores are never reached. -/
theorem tail_run (E : SState) (h209 : E.pc = 209)
    (hlaN : (E.regs "litAnchor" 0).toNat ≤ 32768)
    (hlau : ∀ j : Lane, E.regs "litAnchor" j = E.regs "litAnchor" 0) :
    (∀ i, i ≤ 10 → (siter K i E).pc = 209 + i)
    ∧ (15 ≤ 32768 - (E.regs "litAnchor" 0).toNat →
        (∀ i, i ≤ 13 → (siter K i E).pc = 209 + i)
        ∧ (∀ j : Lane, (siter K 13 E).regs "op" j = E.regs "op" j + 1)
        ∧ (∀ j : Lane, ((siter K 13 E).regs "litExtraF" j).toNat
            = 32768 - (E.regs "litAnchor" 0).toNat - 15)
        ∧ ((((siter K 13 E).regs "lsicC" 0) == 1) = true
            ↔ 255 ≤ ((siter K 13 E).regs "litExtraF" 0).toNat))
    ∧ (32768 - (E.regs "litAnchor" 0).toNat < 15 → (siter K 11 E).pc = 236)
    ∧ (∀ j : Lane, (siter K 11 E).regs "op" j = E.regs "op" j + 1)
    ∧ (∀ j : Lane, ((siter K 2 E).regs "fLen" j).toNat
        = 32768 - (E.regs "litAnchor" 0).toNat) := by
  have e0 : siter K 0 E = E := rfl
  have p0 : (siter K 0 E).pc = 209 := by rw [e0]; exact h209
  have p1 : (siter K 1 E).pc = 210 := by
    rw [siter_succ]; exact tail_pc_mov _ 209 "fLen" (SArg.imm 32768) p0 (by decide)
  have p2 : (siter K 2 E).pc = 211 := by
    rw [siter_succ]; exact tail_pc_bin _ 210 (.sub) "fLen" "fLen" (SArg.reg "litAnchor") p1 (by decide)
  have p3 : (siter K 3 E).pc = 212 := by
    rw [siter_succ]; exact tail_pc_mov _ 211 "zero" (SArg.imm 0) p2 (by decide)
  have p4 : (siter K 4 E).pc = 213 := by
    rw [siter_succ]; exact tail_pc_bin _ 212 (.min) "tokHi" "fLen" (SArg.imm 15) p3 (by decide)
  have p5 : (siter K 5 E).pc = 214 := by
    rw [siter_succ]; exact tail_pc_bin _ 213 (.shl) "tok" "tokHi" (SArg.imm 4) p4 (by decide)
  have p6 : (siter K 6 E).pc = 215 := by
    rw [siter_succ]; exact tail_pc_bin _ 214 (.bor) "tok" "tok" (SArg.reg "zero") p5 (by decide)
  have p7 : (siter K 7 E).pc = 216 := by
    rw [siter_succ]; exact tail_pc_bin _ 215 (.add) "sbAddr" "outBase" (SArg.reg "op") p6 (by decide)
  have p8 : (siter K 8 E).pc = 217 := by
    rw [siter_succ]; exact tail_pc_stg _ 216 "sbAddr" "tok" p7 (by decide)
  have p9 : (siter K 9 E).pc = 218 := by
    rw [siter_succ]; exact tail_pc_bin _ 217 (.add) "op" "op" (SArg.imm 1) p8 (by decide)
  have p10 : (siter K 10 E).pc = 219 := by
    rw [siter_succ]; exact tail_pc_setp _ 218 (.ge) "pLitBigF" "fLen" (SArg.imm 15) p9 (by decide)
  have flitAnchor1 : (siter K 1 E).regs "litAnchor" = (siter K 0 E).regs "litAnchor" := by
    rw [siter_succ]
    exact tail_frame _ 209 (.mov "fLen" (SArg.imm 32768)) "litAnchor" p0 (by decide) (by decide)
  have hla1 : (siter K 1 E).regs "litAnchor" = E.regs "litAnchor" := by
    rw [flitAnchor1, e0]
  have hf1 : ∀ j : Lane, (siter K 1 E).regs "fLen" j = UInt64.ofNat 32768 := by
    intro j; rw [siter_succ]
    exact tail_val_mov _ 209 "fLen" (SArg.imm 32768) p0 (by decide) j
  have hf2 : ∀ j : Lane, (siter K 2 E).regs "fLen" j
      = UInt64.ofNat 32768 - E.regs "litAnchor" j := by
    intro j
    rw [siter_succ,
      tail_val_bin _ 210 (.sub) "fLen" "fLen" (SArg.reg "litAnchor") p1 (by decide) j]
    show (siter K 1 E).regs "fLen" j - (siter K 1 E).regs "litAnchor" j = _
    rw [hf1 j, hla1]
  have ffLen3 : (siter K 3 E).regs "fLen" = (siter K 2 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame _ 211 (.mov "zero" (SArg.imm 0)) "fLen" p2 (by decide) (by decide)
  have ffLen4 : (siter K 4 E).regs "fLen" = (siter K 3 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame _ 212 (.bin (.min) "tokHi" "fLen" (SArg.imm 15)) "fLen" p3 (by decide) (by decide)
  have ffLen5 : (siter K 5 E).regs "fLen" = (siter K 4 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame _ 213 (.bin (.shl) "tok" "tokHi" (SArg.imm 4)) "fLen" p4 (by decide) (by decide)
  have ffLen6 : (siter K 6 E).regs "fLen" = (siter K 5 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame _ 214 (.bin (.bor) "tok" "tok" (SArg.reg "zero")) "fLen" p5 (by decide) (by decide)
  have ffLen7 : (siter K 7 E).regs "fLen" = (siter K 6 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame _ 215 (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) "fLen" p6 (by decide) (by decide)
  have ffLen8 : (siter K 8 E).regs "fLen" = (siter K 7 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame _ 216 (.stg "sbAddr" "tok") "fLen" p7 (by decide) (by decide)
  have ffLen9 : (siter K 9 E).regs "fLen" = (siter K 8 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame _ 217 (.bin (.add) "op" "op" (SArg.imm 1)) "fLen" p8 (by decide) (by decide)
  have ffLen10 : (siter K 10 E).regs "fLen" = (siter K 9 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame _ 218 (.setp (.ge) "pLitBigF" "fLen" (SArg.imm 15)) "fLen" p9 (by decide) (by decide)
  have hfN : ∀ j : Lane, ((siter K 2 E).regs "fLen" j).toNat
      = 32768 - (E.regs "litAnchor" 0).toNat := by
    intro j
    rw [hf2 j, hlau j, UInt64.toNat_sub,
      show (UInt64.ofNat 32768).toNat = 32768 from by decide,
      show 2 ^ 64 - (E.regs "litAnchor" 0).toNat + 32768
        = 2 ^ 64 + (32768 - (E.regs "litAnchor" 0).toNat) from by
        have := (E.regs "litAnchor" 0).toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by omega)]
  have hfl9 : ∀ j : Lane, (siter K 9 E).regs "fLen" j = (siter K 2 E).regs "fLen" j := by
    intro j; rw [ffLen9, ffLen8, ffLen7, ffLen6, ffLen5, ffLen4, ffLen3]
  have hpbv : (siter K 10 E).regs "pLitBigF" 0
      = (if SCmp.run (.ge) ((siter K 9 E).regs "fLen" 0) (UInt64.ofNat 15) then 1 else 0) := by
    rw [siter_succ]
    exact tail_val_setp _ 218 (.ge) "pLitBigF" "fLen" (SArg.imm 15) p9 (by decide) 0
  have fop1 : (siter K 1 E).regs "op" = (siter K 0 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 209 (.mov "fLen" (SArg.imm 32768)) "op" p0 (by decide) (by decide)
  have fop2 : (siter K 2 E).regs "op" = (siter K 1 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 210 (.bin (.sub) "fLen" "fLen" (SArg.reg "litAnchor")) "op" p1 (by decide) (by decide)
  have fop3 : (siter K 3 E).regs "op" = (siter K 2 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 211 (.mov "zero" (SArg.imm 0)) "op" p2 (by decide) (by decide)
  have fop4 : (siter K 4 E).regs "op" = (siter K 3 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 212 (.bin (.min) "tokHi" "fLen" (SArg.imm 15)) "op" p3 (by decide) (by decide)
  have fop5 : (siter K 5 E).regs "op" = (siter K 4 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 213 (.bin (.shl) "tok" "tokHi" (SArg.imm 4)) "op" p4 (by decide) (by decide)
  have fop6 : (siter K 6 E).regs "op" = (siter K 5 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 214 (.bin (.bor) "tok" "tok" (SArg.reg "zero")) "op" p5 (by decide) (by decide)
  have fop7 : (siter K 7 E).regs "op" = (siter K 6 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 215 (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) "op" p6 (by decide) (by decide)
  have fop8 : (siter K 8 E).regs "op" = (siter K 7 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 216 (.stg "sbAddr" "tok") "op" p7 (by decide) (by decide)
  have hop9 : ∀ j : Lane, (siter K 9 E).regs "op" j = (siter K 8 E).regs "op" j + 1 := by
    intro j
    rw [siter_succ, tail_val_bin _ 217 (.add) "op" "op" (SArg.imm 1) p8 (by decide) j]
    rfl
  have fop10 : (siter K 10 E).regs "op" = (siter K 9 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 218 (.setp (.ge) "pLitBigF" "fLen" (SArg.imm 15)) "op" p9 (by decide) (by decide)
  have fop11 : (siter K 11 E).regs "op" = (siter K 10 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame _ 219 (.braifnot "pLitBigF" "Le16") "op" p10 (by decide) (by decide)
  refine ⟨?_, fun hfl => ?_, fun hfl => ?_,
    fun j => by
      rw [fop11, fop10, hop9 j, fop8, fop7, fop6, fop5, fop4, fop3, fop2, fop1, e0], hfN⟩
  · intro i hi
    match i, hi with
    | 0, _ => rw [e0]; exact h209
    | 1, _ => exact p1
    | 2, _ => exact p2
    | 3, _ => exact p3
    | 4, _ => exact p4
    | 5, _ => exact p5
    | 6, _ => exact p6
    | 7, _ => exact p7
    | 8, _ => exact p8
    | 9, _ => exact p9
    | 10, _ => exact p10
  · -- the branch is taken: `pLitBigF = 1`
    have hpb : (((siter K 10 E).regs "pLitBigF" 0) == 1) = true := by
      rw [hpbv]
      exact (setp_ge_iff _ 15 (by decide)).mpr (by rw [hfl9 0, hfN 0]; omega)
    have p11 : (siter K 11 E).pc = 220 := by
      rw [siter_succ, AlgorithmLib.LZ4WarpDSL.braifnot_step K (siter K 10 E) "pLitBigF" "Le16"
        (by rw [p10]; decide), if_pos hpb]
      simp only [SState.setPc, p10]
    have p12 : (siter K 12 E).pc = 221 := by
      rw [siter_succ]; exact tail_pc_bin _ 220 (.sub) "litExtraF" "fLen" (SArg.imm 15) p11 (by decide)
    have p13 : (siter K 13 E).pc = 222 := by
      rw [siter_succ]; exact tail_pc_setp _ 221 (.ge) "lsicC" "litExtraF" (SArg.imm 255) p12 (by decide)
    have ffLen11 : (siter K 11 E).regs "fLen" = (siter K 10 E).regs "fLen" := by
      rw [siter_succ]
      exact tail_frame _ 219 (.braifnot "pLitBigF" "Le16") "fLen" p10 (by decide) (by decide)
    have hle12 : ∀ j : Lane, (siter K 12 E).regs "litExtraF" j
        = (siter K 11 E).regs "fLen" j - UInt64.ofNat 15 := by
      intro j
      rw [siter_succ,
        tail_val_bin _ 220 (.sub) "litExtraF" "fLen" (SArg.imm 15) p11 (by decide) j]
      rfl
    have flitExtraF13 : (siter K 13 E).regs "litExtraF" = (siter K 12 E).regs "litExtraF" := by
      rw [siter_succ]
      exact tail_frame _ 221 (.setp (.ge) "lsicC" "litExtraF" (SArg.imm 255)) "litExtraF" p12 (by decide) (by decide)
    have hleN : ∀ j : Lane, ((siter K 13 E).regs "litExtraF" j).toNat
        = 32768 - (E.regs "litAnchor" 0).toNat - 15 := by
      intro j
      rw [show (siter K 13 E).regs "litExtraF" j = (siter K 12 E).regs "litExtraF" j from by
        rw [flitExtraF13], hle12 j, ffLen11, ffLen10, hfl9 j,
        uint64_sub_toNat _ 15 (by decide) (by rw [hfN j]; omega), hfN j]
    have fop12 : (siter K 12 E).regs "op" = (siter K 11 E).regs "op" := by
      rw [siter_succ]
      exact tail_frame _ 220 (.bin (.sub) "litExtraF" "fLen" (SArg.imm 15)) "op" p11 (by decide) (by decide)
    have fop13 : (siter K 13 E).regs "op" = (siter K 12 E).regs "op" := by
      rw [siter_succ]
      exact tail_frame _ 221 (.setp (.ge) "lsicC" "litExtraF" (SArg.imm 255)) "op" p12 (by decide) (by decide)
    refine ⟨fun i hi => ?_, fun j => ?_, hleN, ?_⟩
    · match i, hi with
      | 0, _ => rw [e0]; exact h209
      | 1, _ => exact p1
      | 2, _ => exact p2
      | 3, _ => exact p3
      | 4, _ => exact p4
      | 5, _ => exact p5
      | 6, _ => exact p6
      | 7, _ => exact p7
      | 8, _ => exact p8
      | 9, _ => exact p9
      | 10, _ => exact p10
      | 11, _ => exact p11
      | 12, _ => exact p12
      | 13, _ => exact p13
    · rw [fop13, fop12, fop11, fop10, hop9 j, fop8, fop7, fop6, fop5, fop4, fop3, fop2, fop1, e0]
    · have hlv : (siter K 13 E).regs "lsicC" 0
          = (if SCmp.run (.ge) ((siter K 12 E).regs "litExtraF" 0) (UInt64.ofNat 255) then 1 else 0) := by
        rw [siter_succ]
        exact tail_val_setp _ 221 (.ge) "lsicC" "litExtraF" (SArg.imm 255) p12 (by decide) 0
      rw [hlv, show (siter K 13 E).regs "litExtraF" 0 = (siter K 12 E).regs "litExtraF" 0 from by
        rw [flitExtraF13]]
      exact setp_ge_iff _ 255 (by decide)
  · -- the branch is not taken: `pLitBigF = 0`, and 219 jumps to `Le16` = 236
    have hnb : (((siter K 10 E).regs "pLitBigF" 0) == 1) = false := by
      rw [hpbv]
      rcases Bool.eq_false_or_eq_true
        ((if SCmp.run (.ge) ((siter K 9 E).regs "fLen" 0) (UInt64.ofNat 15) then (1:UInt64) else 0) == 1)
        with h | h
      · exfalso
        have h15 := (setp_ge_iff _ 15 (by decide)).mp h
        rw [hfl9 0, hfN 0] at h15
        omega
      · exact h
    rw [siter_succ, AlgorithmLib.LZ4WarpDSL.braifnot_step K (siter K 10 E) "pLitBigF" "Le16"
      (by rw [p10]; decide), if_neg (by rw [hnb]; exact Bool.false_ne_true)]
    simp only [SState.setPc]
    decide


/-- `LsicInv` built through a lemma whose state is a *variable*.  Writing the
    anonymous constructor at a concrete `siter K 13 E` makes the elaborator whnf
    that term — thirteen nested `sstep`s over the 274-instruction array — and it
    never comes back. -/
theorem lsicInv_mk (l : Lane) (B : Nat) (st : SState)
    (h1 : (st.regs "op" l).toNat + lsicRem st.pc ((st.regs "litExtraF" l).toNat) ≤ B)
    (h2 : st.regs "litExtraF" l = st.regs "litExtraF" 0)
    (h3 : st.pc = 222 ∨ st.pc = 223 ∨ st.pc = 230 →
      (((st.regs "lsicC" 0) == 1) = true ↔ 255 ≤ (st.regs "litExtraF" 0).toNat))
    (h4 : 224 ≤ st.pc → st.pc ≤ 228 → 255 ≤ (st.regs "litExtraF" l).toNat) :
    LsicInv l B st := ⟨h1, h2, h3, h4⟩

set_option maxHeartbeats 1000000 in
/-- **The tail's three `sbAddr` stores are confined.**  Given the tight budget at
    the loop exit — `op + fl + fl/255 + 2 ≤ LO` for `fl` the final literal run —
    every visit to pc 216, 226 or 233 has `op ≤ LO`.

    The token store is at step 7 and does not move the cursor; the two LSIC stores
    are reached only when `fl ≥ 15`, and then `LsicInv` holds at the loop head
    (step 13) and `lsic_op_lt` carries it. -/
theorem tail_sites_bounded (LO : Nat) (hLO : LO < 2 ^ 64) (E : SState) (h209 : E.pc = 209)
    (hlaN : (E.regs "litAnchor" 0).toNat ≤ 32768)
    (hlau : ∀ j : Lane, E.regs "litAnchor" j = E.regs "litAnchor" 0)
    (hopu : ∀ j : Lane, E.regs "op" j = E.regs "op" 0)
    (hbud : (E.regs "op" 0).toNat + (32768 - (E.regs "litAnchor" 0).toNat)
        + (32768 - (E.regs "litAnchor" 0).toNat) / 255 + 2 ≤ LO)
    (l : Lane) (k : Nat)
    (hq : (siter K k E).pc = 216 ∨ (siter K k E).pc = 226 ∨ (siter K k E).pc = 233) :
    ((siter K k E).regs "op" l).toNat ≤ LO := by
  obtain ⟨hpcs10, hbr1, hbr0, -, -⟩ := tail_run E h209 hlaN hlau
  have hop0 : (E.regs "op" 0).toNat ≤ LO := by omega
  rcases hq with h216 | hlsic
  · have hk7 : k = 7 := by
      rcases Nat.lt_or_ge k 11 with hlt | hge
      · have hpk := hpcs10 k (by omega); rw [h216] at hpk; omega
      · exfalso
        have h217 : 217 ≤ (siter K 10 E).pc := by rw [hpcs10 10 (by omega)]; omega
        have hst := stays_from_217 E 10 h217 k (by omega)
        rw [h216] at hst; omega
    obtain ⟨-, hopc⟩ := tail_to_216 E h209
    rw [hk7, hopc, hopu l]; exact hop0
  · -- the two LSIC stores
    have hfl : 15 ≤ 32768 - (E.regs "litAnchor" 0).toNat := by
      rcases Nat.lt_or_ge (32768 - (E.regs "litAnchor" 0).toNat) 15 with hlt | hge
      · exfalso
        have h236 := hbr0 hlt
        rcases Nat.lt_or_ge k 11 with h1 | h1
        · have hpk := hpcs10 k (by omega)
          rcases hlsic with e | e <;> rw [e] at hpk <;> omega
        · have hst := stays_from_235 E 11 (by rw [h236]; omega) k h1
          rcases hlsic with e | e <;> rw [e] at hst <;> omega
      · exact hge
    obtain ⟨hpcs13, hopv, hlev, hlsc⟩ := hbr1 hfl
    have hpc222 : (siter K 13 E).pc = 222 := hpcs13 13 (by omega)
    have hk13 : 13 ≤ k := by
      rcases Nat.lt_or_ge k 13 with hlt | hge
      · exfalso
        have hpk := hpcs13 k (by omega)
        rcases hlsic with e | e <;> rw [e] at hpk <;> omega
      · exact hge
    -- the invariant at the LSIC head
    have hopN : ∀ j : Lane, ((siter K 13 E).regs "op" j).toNat = (E.regs "op" 0).toNat + 1 := by
      intro j
      rw [hopv j, hopu j]
      have hL := (AlgorithmLib.LZ4Simt.toNat_add_ofNat_of_lt (E.regs "op" 0) 1 (by omega)).1
      rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
      exact hL
    have hdiv : (32768 - (E.regs "litAnchor" 0).toNat - 15) / 255
        ≤ (32768 - (E.regs "litAnchor" 0).toNat) / 255 := Nat.div_le_div_right (by omega)
    have c1 : ((siter K 13 E).regs "op" l).toNat
        + lsicRem (siter K 13 E).pc (((siter K 13 E).regs "litExtraF" l).toNat) ≤ LO := by
      have hr : lsicRem (siter K 13 E).pc ((siter K 13 E).regs "litExtraF" l).toNat
          = ((siter K 13 E).regs "litExtraF" l).toNat / 255 + 1 := by
        rw [hpc222]; rfl
      rw [hr, hopN l, hlev l]
      omega
    have c2 : (siter K 13 E).regs "litExtraF" l = (siter K 13 E).regs "litExtraF" 0 := by
      rw [← UInt64.toNat_inj, hlev l, hlev 0]
    have c3 : (siter K 13 E).pc = 222 ∨ (siter K 13 E).pc = 223 ∨ (siter K 13 E).pc = 230 →
        ((((siter K 13 E).regs "lsicC" 0) == 1) = true
          ↔ 255 ≤ ((siter K 13 E).regs "litExtraF" 0).toNat) := fun _ => hlsc
    have c4 : 224 ≤ (siter K 13 E).pc → (siter K 13 E).pc ≤ 228 →
        255 ≤ ((siter K 13 E).regs "litExtraF" l).toNat := by
      rw [hpc222]; intro h _; exact absurd h (by omega)
    have hInv : LsicInv l LO (siter K 13 E) := lsicInv_mk l LO (siter K 13 E) c1 c2 c3 c4
    have hne : ∀ i, i < k - 13 → (siter K i (siter K 13 E)).pc ∉ [234] := by
      intro i hi hmem
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
      rw [← siter_add K 13 i E] at hmem
      have hs : (siter K (13 + i + 1) E).pc ∈ succsOf K (siter K (13 + i) E).pc := by
        rw [siter_succ]; exact sstep_pc_mem_succs K _
      rw [hmem, show succsOf K 234 = [235] from by decide, List.mem_singleton] at hs
      have hst := stays_from_235 E (13 + i + 1) (by rw [hs]; omega) k (by omega)
      rcases hlsic with e | e <;> rw [e] at hst <;> omega
    have hsplit : siter K (k - 13) (siter K 13 E) = siter K k E := by
      rw [← siter_add K 13 (k - 13) E, show 13 + (k - 13) = k from by omega]
    have hfin := lsic_op_lt l LO hLO (siter K 13 E) hpc222 hInv (k - 13) hne
      (by rw [hsplit]; exact hlsic)
    rw [hsplit] at hfin
    exact Nat.le_of_lt hfin

-- ── The tail copy's budget: `op + fLen ≤ lenOff` where the copy is set up ────

/-- Past pc 211 the machine never returns to the two instructions that write
    `fLen`, and past 238 it never returns to the copy's setup. -/
theorem stays_from_211 (ss : SState) (a : Nat) (h : 211 ≤ (siter K a ss).pc) :
    ∀ b, a ≤ b → 211 ≤ (siter K b ss).pc :=
  stays_ge 211 (by decide) ss a h

theorem stays_from_239 (ss : SState) (a : Nat) (h : 239 ≤ (siter K a ss).pc) :
    ∀ b, a ≤ b → 239 ≤ (siter K b ss).pc :=
  stays_ge 239 (by decide) ss a h

/-- A register is constant along any stretch whose instructions do not write it.
    The trace form of `regs_const_on`, with the pc side condition supplied per
    step instead of by a region closure — which is what the tail needs, since the
    stretch it cares about spans three regions. -/
theorem const_along (r : String) (E : SState) : ∀ (n : Nat),
    (∀ i, i < n → ∀ ins, K[(siter K i E).pc]? = some ins → destOf ins ≠ some r) →
    (siter K n E).regs r = E.regs r := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ m ih =>
      intro h
      rw [siter_succ, sstep_regs_frame K (siter K m E) r (h m (by omega)),
        ih (fun i hi => h i (by omega))]

/-- `fLen` is written only by the two instructions at 209 and 210. -/
theorem fLen_no_write : ∀ q, 211 ≤ q → ∀ ins, K[q]? = some ins → destOf ins ≠ some "fLen" := by
  have hd : ∀ q, q < 274 → 211 ≤ q → (K[q]?.map (fun i => destOf i != some "fLen")) = some true := by
    decide
  intro q hq ins hins
  rcases Nat.lt_or_ge q 274 with h | h
  · have h' := hd q h hq
    rw [hins] at h'
    simpa using h'
  · rw [Array.getElem?_eq_none_iff.mpr (by rw [shipped32_size]; omega)] at hins
    exact absurd hins (by simp)

/-- …so from step `a` on, with the pc already past them, it never changes. -/
theorem fLen_const (ss : SState) (a : Nat) (ha : 211 ≤ (siter K a ss).pc) (b : Nat) (hab : a ≤ b) :
    (siter K b ss).regs "fLen" = (siter K a ss).regs "fLen" := by
  have hkey : ∀ i, 211 ≤ (siter K i (siter K a ss)).pc := by
    intro i
    rw [← siter_add K a i ss]
    exact stays_from_211 ss a ha (a + i) (by omega)
  have h := const_along "fLen" (siter K a ss) (b - a)
    (fun i _ ins hins => fLen_no_write _ (hkey i) ins hins)
  rw [← siter_add K a (b - a) ss, show a + (b - a) = b from by omega] at h
  exact h

/-- The tail's LSIC region together with its exit run to the copy's setup:
    `222–234` (the loop), `235` (the `bra` out), `236`/`237` (the two labels the
    short and long paths join at) and `238` itself. -/
def tailFS : List Nat := (List.range 17).map (· + 222)

theorem tailFS_closed : PcClosed K tailFS [238] := by decide

/-- **What the tail carries from its LSIC head to the copy's setup.**  Inside the
    LSIC run it is that loop's own invariant; past it the potential is spent and
    what remains is the plain cursor bound, which the exit's `op += 1` is exactly
    paid for by `lsicRem 234 = 1`.  `fLen` rides along because the bound the copy
    needs is stated against it. -/
def TailFQ (l : Lane) (B : Nat) (v : UInt64) (st : SState) : Prop :=
  (st.pc ≤ 234 → LsicInv l B st)
  ∧ (235 ≤ st.pc → (st.regs "op" l).toNat ≤ B)
  ∧ st.regs "fLen" l = v

theorem tailFQ_step (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (v : UInt64) (st : SState)
    (hs : st.pc ∈ tailFS) (hex : st.pc ∉ [238]) (h : TailFQ l B v st) :
    TailFQ l B v (sstep K st) := by
  have dLo : ∀ q ∈ tailFS, q ≤ 233 → ∀ q' ∈ succsOf K q, q' ≤ 234 := by decide
  have dHi : ∀ q ∈ tailFS, 235 ≤ q → q ≠ 238 → ∀ q' ∈ succsOf K q, 235 ≤ q' := by decide
  have dIn : ∀ q ∈ tailFS, q ≤ 233 → q ∈ lsicFS := by decide
  have dOp : ∀ q ∈ tailFS, 235 ≤ q →
      (K[q]?.map (fun i => destOf i != some "op")) = some true := by decide
  have d211 : ∀ q ∈ tailFS, 211 ≤ q := by decide
  obtain ⟨h1, h2, h3⟩ := h
  have hq238 : st.pc ≠ 238 := by simpa using hex
  have hsucc : (sstep K st).pc ∈ succsOf K st.pc := sstep_pc_mem_succs K st
  have hfl : (sstep K st).regs "fLen" = st.regs "fLen" :=
    sstep_regs_frame K st "fLen" (fun ins hins => fLen_no_write _ (d211 _ hs) ins hins)
  refine ⟨?_, ?_, by rw [hfl]; exact h3⟩
  · intro hle
    rcases Nat.lt_or_ge st.pc 234 with hlt | hge
    · have hnot : st.pc ∉ [234] := by
        simp only [List.mem_cons, List.not_mem_nil, or_false]
        omega
      exact lsicFS_hstep l B hB st (dIn _ hs (by omega)) hnot (h1 (by omega))
    · exfalso
      rcases Nat.lt_or_ge st.pc 235 with h234 | h235
      · rw [show st.pc = 234 from by omega, show succsOf K 234 = [235] from by decide,
          List.mem_singleton] at hsucc
        omega
      · have := dHi st.pc hs h235 hq238 _ hsucc
        omega
  · intro hge235
    rcases Nat.lt_or_ge st.pc 234 with hlt | hge
    · exact absurd (dLo st.pc hs (by omega) _ hsucc) (by omega)
    · rcases Nat.lt_or_ge st.pc 235 with h234 | h235
      · have he : st.pc = 234 := by omega
        have hb1 := (h1 (by omega)).1
        rw [show lsicRem st.pc ((st.regs "litExtraF" l).toNat) = 1 from by rw [he]; rfl] at hb1
        have hv : (sstep K st).regs "op" l = st.regs "op" l + 1 := by
          rw [tail_val_bin st 234 (.add) "op" "op" (SArg.imm 1) he (by decide) l]; rfl
        have hL := (AlgorithmLib.LZ4Simt.toNat_add_ofNat_of_lt (st.regs "op" l) 1 (by omega)).1
        rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
        rw [hv, hL]
        omega
      · have hop : (sstep K st).regs "op" = st.regs "op" :=
          sstep_regs_frame K st "op" (fun ins hins => by
            have h' := dOp st.pc hs h235
            rw [hins] at h'
            simpa using h')
        rw [hop]
        exact h2 h235

/-- From any entry into the region, the cursor bound survives to the copy's
    setup — 238 is visited at most once, since everything past it is above it. -/
theorem tail_from_entry (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (v : UInt64) (E : SState)
    (m : Nat) (hmem : (siter K m E).pc ∈ tailFS) (hQ : TailFQ l B v (siter K m E))
    (k : Nat) (hmk : m ≤ k) (hq : (siter K k E).pc = 238) :
    ((siter K k E).regs "op" l).toNat ≤ B ∧ (siter K k E).regs "fLen" l = v := by
  have hne : ∀ i, i < k - m → (siter K i (siter K m E)).pc ∉ [238] := by
    intro i hi hin
    rw [← siter_add K m i E] at hin
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hin
    have hs : (siter K (m + i + 1) E).pc ∈ succsOf K (siter K (m + i) E).pc := by
      rw [siter_succ]; exact sstep_pc_mem_succs K _
    rw [hin, show succsOf K 238 = [239] from by decide, List.mem_singleton] at hs
    have hst := stays_from_239 E (m + i + 1) (by rw [hs]; omega) k (by omega)
    rw [hq] at hst; omega
  have hfin := inv_on K (TailFQ l B v) tailFS [238] tailFS_closed
    (fun s hsm hexs hh => tailFQ_step l B hB v s hsm hexs hh) (siter K m E) hmem hQ (k - m) hne
  rw [← siter_add K m (k - m) E, show m + (k - m) = k from by omega] at hfin
  exact ⟨hfin.2.1 (by rw [hq]; omega), hfin.2.2⟩

set_option maxHeartbeats 1000000 in
/-- **The final literal copy fits in what is left of the block.**  Same budget as
    `tail_sites_bounded`, carried past the LSIC run to the copy's setup at 238:
    the token byte, the `fLen/255` continuation bytes and the final length byte
    are the `+2` and the `/255` term, so what remains covers `fLen` itself. -/
theorem tail_copy_budget (LO : Nat) (hLO : LO < 2 ^ 64) (E : SState) (h209 : E.pc = 209)
    (hlaN : (E.regs "litAnchor" 0).toNat ≤ 32768)
    (hlau : ∀ j : Lane, E.regs "litAnchor" j = E.regs "litAnchor" 0)
    (hopu : ∀ j : Lane, E.regs "op" j = E.regs "op" 0)
    (hbud : (E.regs "op" 0).toNat + (32768 - (E.regs "litAnchor" 0).toNat)
        + (32768 - (E.regs "litAnchor" 0).toNat) / 255 + 2 ≤ LO)
    (l : Lane) (k : Nat) (hq : (siter K k E).pc = 238) :
    ((siter K k E).regs "op" l).toNat + ((siter K k E).regs "fLen" l).toNat ≤ LO := by
  obtain ⟨hpcs10, hbr1, hbr0, hop11, hfN⟩ := tail_run E h209 hlaN hlau
  -- the visit is past the straight run out of 209
  have hk11 : 11 ≤ k := by
    rcases Nat.lt_or_ge k 11 with hlt | hge
    · have := hpcs10 k (by omega); rw [hq] at this; omega
    · exact hge
  have hpc2 : 211 ≤ (siter K 2 E).pc := by rw [hpcs10 2 (by omega)]; omega
  have hfl11 : (siter K 11 E).regs "fLen" = (siter K 2 E).regs "fLen" :=
    fLen_const E 2 hpc2 11 (by omega)
  have hop0 : ((E.regs "op" 0).toNat + 1) < 2 ^ 64 := by omega
  -- the two entries into the region
  have hmain : ∀ m : Nat, m ≤ k → (siter K m E).pc ∈ tailFS →
      TailFQ l (LO - (32768 - (E.regs "litAnchor" 0).toNat))
        ((siter K 2 E).regs "fLen" l) (siter K m E) →
      ((siter K k E).regs "op" l).toNat + ((siter K k E).regs "fLen" l).toNat ≤ LO := by
    intro m hmk hmem hQ
    obtain ⟨hoop, hffl⟩ := tail_from_entry l _ (by omega) _ E m hmem hQ k hmk hq
    rw [hffl, hfN l]
    omega
  rcases Nat.lt_or_ge (32768 - (E.regs "litAnchor" 0).toNat) 15 with hshort | hlong
  · -- the short path: 219 jumps straight to 236
    have h236 := hbr0 hshort
    refine hmain 11 (by omega) (by rw [h236]; decide) ⟨fun hle => absurd hle (by rw [h236]; omega),
      fun _ => ?_, by rw [hfl11]⟩
    have hL := (AlgorithmLib.LZ4Simt.toNat_add_ofNat_of_lt (E.regs "op" l) 1 (by
      rw [hopu l]; omega)).1
    rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
    rw [hop11 l, hL, hopu l]
    omega
  · -- the long path: the LSIC head at 222, with the loop's own invariant
    obtain ⟨hpcs13, hopv, hlev, hlsc⟩ := hbr1 hlong
    have hpc222 : (siter K 13 E).pc = 222 := hpcs13 13 (by omega)
    have hk13 : 13 ≤ k := by
      rcases Nat.lt_or_ge k 13 with hlt | hge
      · have := hpcs13 k (by omega); rw [hq] at this; omega
      · exact hge
    have hopN : ∀ j : Lane, ((siter K 13 E).regs "op" j).toNat = (E.regs "op" 0).toNat + 1 := by
      intro j
      rw [hopv j, hopu j]
      have hL := (AlgorithmLib.LZ4Simt.toNat_add_ofNat_of_lt (E.regs "op" 0) 1 (by omega)).1
      rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
      exact hL
    have hdiv : (32768 - (E.regs "litAnchor" 0).toNat - 15) / 255
        ≤ (32768 - (E.regs "litAnchor" 0).toNat) / 255 := Nat.div_le_div_right (by omega)
    have c1 : ((siter K 13 E).regs "op" l).toNat
        + lsicRem (siter K 13 E).pc (((siter K 13 E).regs "litExtraF" l).toNat)
        ≤ LO - (32768 - (E.regs "litAnchor" 0).toNat) := by
      rw [show lsicRem (siter K 13 E).pc ((siter K 13 E).regs "litExtraF" l).toNat
          = ((siter K 13 E).regs "litExtraF" l).toNat / 255 + 1 from by rw [hpc222]; rfl,
        hopN l, hlev l]
      omega
    have c2 : (siter K 13 E).regs "litExtraF" l = (siter K 13 E).regs "litExtraF" 0 := by
      rw [← UInt64.toNat_inj, hlev l, hlev 0]
    have c3 : (siter K 13 E).pc = 222 ∨ (siter K 13 E).pc = 223 ∨ (siter K 13 E).pc = 230 →
        ((((siter K 13 E).regs "lsicC" 0) == 1) = true
          ↔ 255 ≤ ((siter K 13 E).regs "litExtraF" 0).toNat) := fun _ => hlsc
    have c4 : 224 ≤ (siter K 13 E).pc → (siter K 13 E).pc ≤ 228 →
        255 ≤ ((siter K 13 E).regs "litExtraF" l).toNat := by
      rw [hpc222]; intro h _; exact absurd h (by omega)
    refine hmain 13 hk13 (by rw [hpc222]; decide)
      ⟨fun _ => lsicInv_mk l _ (siter K 13 E) c1 c2 c3 c4,
        fun hge => absurd hge (by rw [hpc222]; omega), ?_⟩
    rw [fLen_const E 2 hpc2 13 (by omega)]

-- ── From the match-sequence entry to the token region ────────────────────────

/-- `lsicLen` and `encNib`'s length are the same function. -/
theorem lsicLen_eq_encNib (n : Nat) : lsicLen n = (AlgorithmLib.LZ4.encNib n).length := by
  rw [AlgorithmLib.LZ4Imp.encNib_length, lsicLen]
  by_cases h : 15 ≤ n
  · rw [if_pos h, if_neg (by omega)]
  · rw [if_neg h, if_pos (by omega)]

/-- The five instructions of `wEmitMatchSeq`'s prologue, pcs 124–128:
    `124 mlm := ml - 4; 125 tokLo := min mlm 15; 126 tokHi := min litLen 15;
     127 tok := tokHi << 4; 128 tok := tok ||| tokLo` — read off the emitted array.

    Stated with `siter K n`, not nested `sstep`: a five-deep `sstep` term in every
    hypothesis type makes the olean writer overflow its stack, which is a build
    failure with no line number and no relation to the proof. -/
theorem me_pc0 (st : SState) (he : st.pc = 124) : (siter K 0 st).pc = 124 := he
theorem me_pc1 (st : SState) (he : st.pc = 124) : (siter K 1 st).pc = 125 := by
  rw [siter_succ]; exact tail_pc_bin _ 124 (.sub) "mlm" "ml" (SArg.imm 4) (me_pc0 st he) (by decide)
theorem me_pc2 (st : SState) (he : st.pc = 124) : (siter K 2 st).pc = 126 := by
  rw [siter_succ]; exact tail_pc_bin _ 125 (.min) "tokLo" "mlm" (SArg.imm 15) (me_pc1 st he) (by decide)
theorem me_pc3 (st : SState) (he : st.pc = 124) : (siter K 3 st).pc = 127 := by
  rw [siter_succ]; exact tail_pc_bin _ 126 (.min) "tokHi" "litLen" (SArg.imm 15) (me_pc2 st he) (by decide)
theorem me_pc4 (st : SState) (he : st.pc = 124) : (siter K 4 st).pc = 128 := by
  rw [siter_succ]; exact tail_pc_bin _ 127 (.shl) "tok" "tokHi" (SArg.imm 4) (me_pc3 st he) (by decide)
theorem me_pc5 (st : SState) (he : st.pc = 124) : (siter K 5 st).pc = 129 := by
  rw [siter_succ]; exact tail_pc_bin _ 128 (.bor) "tok" "tok" (SArg.reg "tokLo") (me_pc4 st he) (by decide)

/-- Everything the token region reads at its entry is untouched by those five. -/
theorem me_frame (st : SState) (he : st.pc = 124) (r : String)
    (h1 : r ≠ "mlm") (h2 : r ≠ "tokLo") (h3 : r ≠ "tokHi") (h4 : r ≠ "tok") :
    (siter K 5 st).regs r = st.regs r := by
  have f1 : (siter K 1 st).regs r = (siter K 0 st).regs r := by
    rw [siter_succ]
    exact tail_frame _ 124 (.bin (.sub) "mlm" "ml" (SArg.imm 4)) r (me_pc0 st he) (by decide)
      (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h1)
  have f2 : (siter K 2 st).regs r = (siter K 1 st).regs r := by
    rw [siter_succ]
    exact tail_frame _ 125 (.bin (.min) "tokLo" "mlm" (SArg.imm 15)) r (me_pc1 st he) (by decide)
      (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h2)
  have f3 : (siter K 3 st).regs r = (siter K 2 st).regs r := by
    rw [siter_succ]
    exact tail_frame _ 126 (.bin (.min) "tokHi" "litLen" (SArg.imm 15)) r (me_pc2 st he) (by decide)
      (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h3)
  have f4 : (siter K 4 st).regs r = (siter K 3 st).regs r := by
    rw [siter_succ]
    exact tail_frame _ 127 (.bin (.shl) "tok" "tokHi" (SArg.imm 4)) r (me_pc3 st he) (by decide)
      (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h4)
  have f5 : (siter K 5 st).regs r = (siter K 4 st).regs r := by
    rw [siter_succ]
    exact tail_frame _ 128 (.bin (.bor) "tok" "tok" (SArg.reg "tokLo")) r (me_pc4 st he) (by decide)
      (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h4)
  rw [f5, f4, f3, f2, f1]
  rfl

/-- …and `mlm` is `ml - 4`, written once at 124 and not again. -/
theorem me_mlm (st : SState) (he : st.pc = 124) (l : Lane) :
    (siter K 5 st).regs "mlm" l = st.regs "ml" l - 4 := by
  have f1 : (siter K 1 st).regs "mlm" l = st.regs "ml" l - 4 := by
    rw [siter_succ, tail_val_bin _ 124 (.sub) "mlm" "ml" (SArg.imm 4) (me_pc0 st he) (by decide) l]
    rfl
  have f2 : (siter K 2 st).regs "mlm" = (siter K 1 st).regs "mlm" := by
    rw [siter_succ]
    exact tail_frame _ 125 (.bin (.min) "tokLo" "mlm" (SArg.imm 15)) "mlm" (me_pc1 st he) (by decide) (by decide)
  have f3 : (siter K 3 st).regs "mlm" = (siter K 2 st).regs "mlm" := by
    rw [siter_succ]
    exact tail_frame _ 126 (.bin (.min) "tokHi" "litLen" (SArg.imm 15)) "mlm" (me_pc2 st he) (by decide) (by decide)
  have f4 : (siter K 4 st).regs "mlm" = (siter K 3 st).regs "mlm" := by
    rw [siter_succ]
    exact tail_frame _ 127 (.bin (.shl) "tok" "tokHi" (SArg.imm 4)) "mlm" (me_pc3 st he) (by decide) (by decide)
  have f5 : (siter K 5 st).regs "mlm" = (siter K 4 st).regs "mlm" := by
    rw [siter_succ]
    exact tail_frame _ 128 (.bin (.bor) "tok" "tok" (SArg.reg "tokLo")) "mlm" (me_pc4 st he) (by decide) (by decide)
  rw [show (siter K 5 st).regs "mlm" l = (siter K 5 st).regs "mlm" l from rfl, f5, f4, f3, f2, f1]

/-- **The token invariant holds at the emit entry.**  From the checkpoint at the
    match-sequence entry (pc 124) — one coupled eval state carrying the tight
    bound `op + |encodeSeq| ≤ LO` — five straight-line steps reach pc 129 with
    `TokInv`.

    Only two of `TokInv`'s ten clauses have content here: the potential, which is
    the checkpoint's bound with `lsicLen` for `encNib`'s length, and lane
    uniformity, which is free because `Couple` equates every lane with the single
    sequential register and `loopR` contains all six.  The other eight are guards
    for pcs 133–192 and are vacuous at 129. -/
theorem tokInv_at_entry (l : Lane) (LO : Nat) (st : SState) (he : st.pc = 124)
    (h : AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 15).inStride LO st) :
    TokInv l LO (siter K 5 st) := by
  obtain ⟨w, hc, -, -, hml4, hbnd⟩ := h
  have hmem : ∀ r ∈ ["op", "litLen", "litExtra", "matExtra", "lsicC", "ml"],
      r ∈ "p0" :: "cand0" :: AlgorithmLib.LZ4WarpDSL.loopR := by decide
  have hR : ∀ (r : String), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "matExtra"
      ∨ r = "lsicC" → ∀ j : Lane,
      (siter K 5 st).regs r j = w.regs r := by
    intro r hr j
    have hne : r ≠ "mlm" ∧ r ≠ "tokLo" ∧ r ≠ "tokHi" ∧ r ≠ "tok" := by
      rcases hr with rfl | rfl | rfl | rfl | rfl <;> exact ⟨by decide, by decide, by decide, by decide⟩
    rw [show (siter K 5 st).regs r j = st.regs r j from by
      rw [me_frame st he r hne.1 hne.2.1 hne.2.2.1 hne.2.2.2]]
    exact hc.reg (by rcases hr with rfl | rfl | rfl | rfl | rfl <;> decide) j
  have hMLM : ∀ j : Lane,
      ((siter K 5 st).regs "mlm" j).toNat
        = (w.regs "ml").toNat - 4 := by
    intro j
    rw [me_mlm st he j, hc.reg (show "ml" ∈ "p0" :: "cand0" :: AlgorithmLib.LZ4WarpDSL.loopR
      by decide) j, show (4 : UInt64) = UInt64.ofNat 4 from rfl,
      uint64_sub_toNat (w.regs "ml") 4 (by decide) hml4]
  have hMLMu : ∀ j : Lane,
      (siter K 5 st).regs "mlm" j
        = (siter K 5 st).regs "mlm" 0 := by
    intro j
    rw [me_mlm st he j, me_mlm st he 0,
      hc.reg (show "ml" ∈ "p0" :: "cand0" :: AlgorithmLib.LZ4WarpDSL.loopR by decide) j,
      hc.reg (show "ml" ∈ "p0" :: "cand0" :: AlgorithmLib.LZ4WarpDSL.loopR by decide) 0]
  have hpc := me_pc5 st he
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc, hR "op" (Or.inl rfl) l, hR "litLen" (Or.inr (Or.inl rfl)) l,
      hR "litExtra" (Or.inr (Or.inr (Or.inl rfl))) l,
      hR "matExtra" (Or.inr (Or.inr (Or.inr (Or.inl rfl)))) l, hMLM l]
    show (w.regs "op").toNat + (1 + lsicLen (w.regs "litLen").toNat + (w.regs "litLen").toNat
      + 2 + lsicLen ((w.regs "ml").toNat - 4)) ≤ LO
    rw [lsicLen_eq_encNib, lsicLen_eq_encNib]
    omega
  · intro r hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hR "op" (Or.inl rfl) l, hR "op" (Or.inl rfl) 0]
    · rw [hR "litLen" (Or.inr (Or.inl rfl)) l, hR "litLen" (Or.inr (Or.inl rfl)) 0]
    · rw [hR "litExtra" (Or.inr (Or.inr (Or.inl rfl))) l,
        hR "litExtra" (Or.inr (Or.inr (Or.inl rfl))) 0]
    · exact hMLMu l
    · rw [hR "matExtra" (Or.inr (Or.inr (Or.inr (Or.inl rfl)))) l,
        hR "matExtra" (Or.inr (Or.inr (Or.inr (Or.inl rfl)))) 0]
    · rw [hR "lsicC" (Or.inr (Or.inr (Or.inr (Or.inr rfl)))) l,
        hR "lsicC" (Or.inr (Or.inr (Or.inr (Or.inr rfl)))) 0]
  all_goals (rw [hpc]; intro hq; first | omega | (exfalso; omega) | (rcases hq with e | e | e <;> omega))


theorem tok_exits : AlgorithmLib.LZ4Simt.succsOf K 197 = [199]
    ∧ AlgorithmLib.LZ4Simt.succsOf K 198 = [199] := by decide

/-- **The seven token stores are bounded, for the whole run.**  Given only that the
    machine reports a coupled eval state with the tight bound whenever it stands at
    the match-sequence entry, every visit to any of the seven `sbAddr` stores of the
    token emit has `op ≤ LO`.

    The three ingredients: `region_entry` finds the entry the visit descends from,
    `tokInv_at_entry` turns the checkpoint there into `TokInv`, and `tok_op_lt`
    propagates it forward.  The "no exit in between" side condition is the region
    confinement `region_entry` already returns — 197 and 198 both step to 199,
    which is outside. -/
theorem tok_sites_bounded (LO : Nat) (hLO : LO < 2 ^ 64) (ss : SState)
    (h0 : ss.pc ∉ tokRegion)
    (l : Lane) (k : Nat)
    (hck : ∀ j, j ≤ k → (siter K j ss).pc = 124 →
      AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 15).inStride LO (siter K j ss))
    (hq : (siter K k ss).pc = 130 ∨ (siter K k ss).pc = 140 ∨ (siter K k ss).pc = 147
      ∨ (siter K k ss).pc = 173 ∨ (siter K k ss).pc = 178 ∨ (siter K k ss).pc = 188
      ∨ (siter K k ss).pc = 195) :
    ((siter K k ss).regs "op" l).toNat ≤ LO := by
  have hmem : (siter K k ss).pc ∈ tokRegion := by
    rcases hq with e | e | e | e | e | e | e <;> rw [e] <;> decide
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K tokRegion 124 tokRegion_entry ss h0 k hmem
  -- the five prologue instructions of the match sequence
  have hp1 := me_pc1 (siter K j ss) hje
  have hp2 := me_pc2 (siter K j ss) hje
  have hp3 := me_pc3 (siter K j ss) hje
  have hp4 := me_pc4 (siter K j ss) hje
  have hp5 := me_pc5 (siter K j ss) hje
  have hshift : ∀ a : Nat, siter K (j + a) ss = siter K a (siter K j ss) := by
    intro a; exact siter_add K j a ss
  have hstep : ∀ a : Nat, a ≤ 5 → (siter K (j + a) ss).pc = 124 + a := by
    intro a ha
    rw [hshift a]
    match a, ha with
    | 0, _ => exact hje
    | 1, _ => exact hp1
    | 2, _ => exact hp2
    | 3, _ => exact hp3
    | 4, _ => exact hp4
    | 5, _ => exact hp5
  -- a site is at least 130, so the visit is past the prologue
  have hk5 : j + 5 ≤ k := by
    rcases Nat.lt_or_ge k (j + 5) with hlt | hge
    · have hle : k - j ≤ 5 := by omega
      have hh := hstep (k - j) hle
      rw [show j + (k - j) = k from by omega] at hh
      rcases hq with e | e | e | e | e | e | e <;> rw [e] at hh <;> omega
    · exact hge
  have hTok : TokInv l LO (siter K (j + 5) ss) := by
    rw [hshift 5]
    exact tokInv_at_entry l LO (siter K j ss) hje (hck j hjk hje)
  have hpc129 : (siter K (j + 5) ss).pc = 129 := by rw [hshift 5]; exact hp5 ▸ rfl
  have hne : ∀ i, i < k - (j + 5) → (siter K i (siter K (j + 5) ss)).pc ∉ [197, 198] := by
    intro i hi hin
    rw [← siter_add K (j + 5) i ss, show j + 5 + i = j + (5 + i) from by omega] at hin
    have hnext : (siter K (j + (5 + i) + 1) ss).pc ∈ tokRegion :=
      hall (j + (5 + i) + 1) (by omega) (by omega)
    have hs : (siter K (j + (5 + i) + 1) ss).pc
        ∈ AlgorithmLib.LZ4Simt.succsOf K (siter K (j + (5 + i)) ss).pc := by
      rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K _
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hin
    rcases hin with e | e
    · rw [e, tok_exits.1, List.mem_singleton] at hs; rw [hs] at hnext; exact absurd hnext (by decide)
    · rw [e, tok_exits.2, List.mem_singleton] at hs; rw [hs] at hnext; exact absurd hnext (by decide)
  have hsite : (siter K (k - (j + 5)) (siter K (j + 5) ss)).pc = 130
      ∨ (siter K (k - (j + 5)) (siter K (j + 5) ss)).pc = 140
      ∨ (siter K (k - (j + 5)) (siter K (j + 5) ss)).pc = 147
      ∨ (siter K (k - (j + 5)) (siter K (j + 5) ss)).pc = 173
      ∨ (siter K (k - (j + 5)) (siter K (j + 5) ss)).pc = 178
      ∨ (siter K (k - (j + 5)) (siter K (j + 5) ss)).pc = 188
      ∨ (siter K (k - (j + 5)) (siter K (j + 5) ss)).pc = 195 := by
    rw [← siter_add K (j + 5) (k - (j + 5)) ss, show j + 5 + (k - (j + 5)) = k from by omega]
    exact hq
  have hfin := tok_op_lt l LO hLO (siter K (j + 5) ss) hpc129 hTok (k - (j + 5)) hne hsite
  rw [← siter_add K (j + 5) (k - (j + 5)) ss, show j + 5 + (k - (j + 5)) = k from by omega] at hfin
  exact Nat.le_of_lt hfin


end Lz4Sites

-- p1
