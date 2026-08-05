import Lz4Sites64
import Lz4Cursor

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

theorem tokRegion_entry_lt64 : ∀ q, q < 274 → q ∉ tokRegion →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q, q' ∈ tokRegion → q' = 124 :=
  ivEntry_at K16 124 75 124 kSize16 (by decide)


theorem tokRegion_entry64 : ∀ q, q ∉ tokRegion →
    ∀ q', q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q → q' ∈ tokRegion → q' = 124 := by
  intro q hq q' hq' hin
  rcases Nat.lt_or_ge q 274 with h | h
  · exact tokRegion_entry_lt64 q h hq q' hq' hin
  · rw [show AlgorithmLib.LZ4Simt.succsOf K16 q = [q] from by
      simp only [AlgorithmLib.LZ4Simt.succsOf,
        Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)]] at hq'
    rw [List.mem_singleton] at hq'
    exact absurd (hq' ▸ hin) hq

-- ── The body region, and that it is entered exactly once ─────────────────────

theorem bodyRegion_exit64 : ∀ q ∈ bodyRegion,
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q, q' ∈ bodyRegion ∨ 272 ≤ q' :=
  ivExit_at K16 40 232 272 kSize16 (by omega) (by decide)





theorem cursorUp_true64 : cursorUpB K16 = true := by decide


theorem beyond_step_lt64 : ∀ q, q < 274 → 272 ≤ q →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q, 272 ≤ q' :=
  upClosed_at K16 272 kSize16 (by
    have h := cursorUp_true64; simp only [cursorUpB, Bool.and_eq_true] at h; exact h.2)

theorem stays_beyond64 (ss : SState) (a : Nat) (h : 272 ≤ (siter K16 a ss).pc) :
    ∀ b, a ≤ b → 272 ≤ (siter K16 b ss).pc := by
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact h
      · have hm := ih hge
        have hs : (siter K16 (m + 1) ss).pc
            ∈ AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 m ss).pc := by
          rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K16 _
        rcases Nat.lt_or_ge (siter K16 m ss).pc 274 with hlt2 | hge2
        · exact beyond_step_lt64 _ hlt2 hm _ hs
        · rw [show AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 m ss).pc = [(siter K16 m ss).pc] from by
            simp only [AlgorithmLib.LZ4Simt.succsOf,
              Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)],
            List.mem_singleton] at hs
          omega

theorem region_or_beyond64 (ss : SState) (a : Nat) (h : (siter K16 a ss).pc ∈ bodyRegion) :
    ∀ b, a ≤ b → (siter K16 b ss).pc ∈ bodyRegion ∨ 272 ≤ (siter K16 b ss).pc := by
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact Or.inl h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact Or.inl h
      · have hs : (siter K16 (m + 1) ss).pc
            ∈ AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 m ss).pc := by
          rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K16 _
        rcases ih hge with hin | hbe
        · exact bodyRegion_exit64 _ hin _ hs
        · exact Or.inr (stays_beyond64 ss m hbe (m + 1) (by omega))

theorem body_entry_at64 (ss : SState) (P : Nat) (hP : (siter K16 P ss).pc = 39)
    (k : Nat) (hk : (siter K16 k ss).pc ∈ bodyRegion) :
    P + 1 ≤ k ∧ (siter K16 (P + 1) ss).pc = 40
      ∧ ∀ i, P + 1 ≤ i → i ≤ k → (siter K16 i ss).pc ∈ bodyRegion := by
  have h39 : AlgorithmLib.LZ4Simt.succsOf K16 39 = [40] := by decide
  have hpc1 : (siter K16 (P + 1) ss).pc = 40 := by
    have hs : (siter K16 (P + 1) ss).pc ∈ AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 P ss).pc := by
      rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K16 _
    rw [hP, h39, List.mem_singleton] at hs; exact hs
  have hmem1 : (siter K16 (P + 1) ss).pc ∈ bodyRegion := by rw [hpc1]; decide
  have hkP : P + 1 ≤ k := by
    rcases Nat.lt_or_ge k (P + 1) with hlt | hge
    · exfalso
      rcases region_or_beyond64 ss k hk P (by omega) with e | e
      · rw [hP] at e; exact absurd e (by decide)
      · rw [hP] at e; omega
    · exact hge
  refine ⟨hkP, hpc1, fun i h1 h2 => ?_⟩
  rcases region_or_beyond64 ss (P + 1) hmem1 i h1 with e | e
  · exact e
  · exfalso
    have hbk := stays_beyond64 ss i e k h2
    have hle : ∀ q ∈ bodyRegion, q ≤ 271 := by decide
    have := hle _ hk
    omega

theorem tail_step_lt64 : ∀ q, q < 274 → 208 ≤ q →
    ∀ q' ∈ AlgorithmLib.LZ4Simt.succsOf K16 q, 208 ≤ q' :=
  upClosed_at K16 208 kSize16 (by
    have h := cursorUp_true64; simp only [cursorUpB, Bool.and_eq_true] at h; exact h.1.1.1)


theorem stays_from_20864 (ss : SState) (a : Nat) (h : 208 ≤ (siter K16 a ss).pc) :
    ∀ b, a ≤ b → 208 ≤ (siter K16 b ss).pc := by
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact h
      · have hm := ih hge
        have hs : (siter K16 (m + 1) ss).pc
            ∈ AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 m ss).pc := by
          rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K16 _
        rcases Nat.lt_or_ge (siter K16 m ss).pc 274 with hlt2 | hge2
        · exact tail_step_lt64 _ hlt2 hm _ hs
        · rw [show AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 m ss).pc = [(siter K16 m ss).pc] from by
            simp only [AlgorithmLib.LZ4Simt.succsOf,
              Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)],
            List.mem_singleton] at hs
          omega


-- **Seal `preSteps`.**  Left reducible, the elaborator will happily `whnf` a
-- `siter K16 preSteps …` — 1058 machine steps over a 274-instruction array — which
-- is what makes the proof terms here enormous and the olean writer overflow.
-- Nothing downstream needs its value.

theorem tail_pc_bin64 (st : SState) (q : Nat) (o : SOp) (d a : String) (b : SArg) (hq : st.pc = q)
    (hi : K16[q]? = some (.bin o d a b)) : (sstep K16 st).pc = q + 1 := by
  rw [sstep, show K16[st.pc]? = some (.bin o d a b) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]


theorem tail_pc_mov64 (st : SState) (q : Nat) (d : String) (a : SArg) (hq : st.pc = q)
    (hi : K16[q]? = some (.mov d a)) : (sstep K16 st).pc = q + 1 := by
  rw [sstep, show K16[st.pc]? = some (.mov d a) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]


theorem tail_pc_setp64 (st : SState) (q : Nat) (c : SCmp) (d a : String) (b : SArg) (hq : st.pc = q)
    (hi : K16[q]? = some (.setp c d a b)) : (sstep K16 st).pc = q + 1 := by
  rw [sstep, show K16[st.pc]? = some (.setp c d a b) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]


theorem tail_pc_stg64 (st : SState) (q : Nat) (d a : String) (hq : st.pc = q)
    (hi : K16[q]? = some (.stg d a)) : (sstep K16 st).pc = q + 1 := by
  rw [sstep, show K16[st.pc]? = some (.stg d a) from by rw [hq]; exact hi]
  show st.pc + 1 = q + 1; rw [hq]




theorem tail_frame64 (st : SState) (q : Nat) (i : SInstr) (r : String) (hq : st.pc = q)
    (hi : K16[q]? = some i) (hne : AlgorithmLib.LZ4WarpDSL.wtgt i ≠ some r) :
    (sstep K16 st).regs r = st.regs r :=
  AlgorithmLib.LZ4WarpDSL.sstep_regs_ne K16 st r (fun i' hi' => by
    rw [show K16[st.pc]? = some i from by rw [hq]; exact hi] at hi'
    cases hi'; exact hne)

theorem tail_to_21664 (E : SState) (h209 : E.pc = 209) :
    (siter K16 7 E).pc = 216 ∧ (siter K16 7 E).regs "op" = E.regs "op" := by
  have e0 : siter K16 0 E = E := rfl
  have p1 : (siter K16 1 E).pc = 210 := by
    rw [siter_succ, e0]; exact tail_pc_mov64 E 209 "fLen" (SArg.imm 65536) h209 (by decide)
  have p2 : (siter K16 2 E).pc = 211 := by
    rw [siter_succ]
    exact tail_pc_bin64 _ 210 (.sub) "fLen" "fLen" (SArg.reg "litAnchor") p1 (by decide)
  have p3 : (siter K16 3 E).pc = 212 := by
    rw [siter_succ]; exact tail_pc_mov64 _ 211 "zero" (SArg.imm 0) p2 (by decide)
  have p4 : (siter K16 4 E).pc = 213 := by
    rw [siter_succ]; exact tail_pc_bin64 _ 212 (.min) "tokHi" "fLen" (SArg.imm 15) p3 (by decide)
  have p5 : (siter K16 5 E).pc = 214 := by
    rw [siter_succ]; exact tail_pc_bin64 _ 213 (.shl) "tok" "tokHi" (SArg.imm 4) p4 (by decide)
  have p6 : (siter K16 6 E).pc = 215 := by
    rw [siter_succ]; exact tail_pc_bin64 _ 214 (.bor) "tok" "tok" (SArg.reg "zero") p5 (by decide)
  have p7 : (siter K16 7 E).pc = 216 := by
    rw [siter_succ]
    exact tail_pc_bin64 _ 215 (.add) "sbAddr" "outBase" (SArg.reg "op") p6 (by decide)
  refine ⟨p7, ?_⟩
  have f1 : (siter K16 1 E).regs "op" = E.regs "op" := by
    rw [siter_succ, e0]
    exact tail_frame64 E 209 (.mov "fLen" (SArg.imm 65536)) "op" h209 (by decide) (by decide)
  have f2 : (siter K16 2 E).regs "op" = (siter K16 1 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 210 (.bin (.sub) "fLen" "fLen" (SArg.reg "litAnchor")) "op" p1
      (by decide) (by decide)
  have f3 : (siter K16 3 E).regs "op" = (siter K16 2 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 211 (.mov "zero" (SArg.imm 0)) "op" p2 (by decide) (by decide)
  have f4 : (siter K16 4 E).regs "op" = (siter K16 3 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 212 (.bin (.min) "tokHi" "fLen" (SArg.imm 15)) "op" p3
      (by decide) (by decide)
  have f5 : (siter K16 5 E).regs "op" = (siter K16 4 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 213 (.bin (.shl) "tok" "tokHi" (SArg.imm 4)) "op" p4 (by decide) (by decide)
  have f6 : (siter K16 6 E).regs "op" = (siter K16 5 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 214 (.bin (.bor) "tok" "tok" (SArg.reg "zero")) "op" p5
      (by decide) (by decide)
  have f7 : (siter K16 7 E).regs "op" = (siter K16 6 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 215 (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) "op" p6
      (by decide) (by decide)
  rw [f7, f6, f5, f4, f3, f2, f1]

theorem tail_val_bin64 (st : SState) (q : Nat) (o : SOp) (d a : String) (b : SArg) (hq : st.pc = q)
    (hi : K16[q]? = some (.bin o d a b)) (l : Lane) :
    (sstep K16 st).regs d l = o.run (st.regs a l) (st.get l b) := by
  rw [sstep, show K16[st.pc]? = some (.bin o d a b) from by rw [hq]; exact hi]
  simp only [sstepInstr, SState.setPc, SState.setReg, eq_self_iff_true, if_true]


theorem tail_val_mov64 (st : SState) (q : Nat) (d : String) (a : SArg) (hq : st.pc = q)
    (hi : K16[q]? = some (.mov d a)) (l : Lane) :
    (sstep K16 st).regs d l = st.get l a := by
  rw [sstep, show K16[st.pc]? = some (.mov d a) from by rw [hq]; exact hi]
  simp only [sstepInstr, SState.setPc, SState.setReg, eq_self_iff_true, if_true]


theorem tail_val_setp64 (st : SState) (q : Nat) (c : SCmp) (d a : String) (b : SArg) (hq : st.pc = q)
    (hi : K16[q]? = some (.setp c d a b)) (l : Lane) :
    (sstep K16 st).regs d l = (if c.run (st.regs a l) (st.get l b) then 1 else 0) := by
  rw [sstep, show K16[st.pc]? = some (.setp c d a b) from by rw [hq]; exact hi]
  simp only [sstepInstr, SState.setPc, SState.setReg, eq_self_iff_true, if_true]

-- ── Forward confinement in the tail, by interval ─────────────────────────────

theorem stays_ge64 (b : Nat) (hb : ∀ q, q < 274 → b ≤ q → ∀ q' ∈ succsOf K16 q, b ≤ q')
    (ss : SState) (a : Nat) (h : b ≤ (siter K16 a ss).pc) :
    ∀ c, a ≤ c → b ≤ (siter K16 c ss).pc := by
  intro c
  induction c with
  | zero => intro hc; rw [show a = 0 from by omega] at h; exact h
  | succ m ih =>
      intro hc
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact h
      · have hm := ih hge
        have hs : (siter K16 (m + 1) ss).pc ∈ succsOf K16 (siter K16 m ss).pc := by
          rw [siter_succ]; exact sstep_pc_mem_succs K16 _
        rcases Nat.lt_or_ge (siter K16 m ss).pc 274 with hlt2 | hge2
        · exact hb _ hlt2 hm _ hs
        · rw [show succsOf K16 (siter K16 m ss).pc = [(siter K16 m ss).pc] from by
            simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)],
            List.mem_singleton] at hs
          omega


theorem stays_from_21064 (ss : SState) (a : Nat) (h : 210 ≤ (siter K16 a ss).pc) :
    ∀ b, a ≤ b → 210 ≤ (siter K16 b ss).pc :=
  stays_ge64 210 (by decide) ss a h

theorem stays_from_21764 (ss : SState) (a : Nat) (h : 217 ≤ (siter K16 a ss).pc) :
    ∀ b, a ≤ b → 217 ≤ (siter K16 b ss).pc := by
  have hstep : ∀ q, q < 274 → 217 ≤ q → ∀ q' ∈ succsOf K16 q, 217 ≤ q' :=
    upClosed_at K16 217 kSize16 (by
      have h := cursorUp_true64; simp only [cursorUpB, Bool.and_eq_true] at h; exact h.1.1.2)
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact h
      · have hm := ih hge
        have hs : (siter K16 (m + 1) ss).pc ∈ succsOf K16 (siter K16 m ss).pc := by
          rw [siter_succ]; exact sstep_pc_mem_succs K16 _
        rcases Nat.lt_or_ge (siter K16 m ss).pc 274 with hlt2 | hge2
        · exact hstep _ hlt2 hm _ hs
        · rw [show succsOf K16 (siter K16 m ss).pc = [(siter K16 m ss).pc] from by
            simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)],
            List.mem_singleton] at hs
          omega

theorem stays_from_23564 (ss : SState) (a : Nat) (h : 235 ≤ (siter K16 a ss).pc) :
    ∀ b, a ≤ b → 235 ≤ (siter K16 b ss).pc := by
  have hstep : ∀ q, q < 274 → 235 ≤ q → ∀ q' ∈ succsOf K16 q, 235 ≤ q' :=
    upClosed_at K16 235 kSize16 (by
      have h := cursorUp_true64; simp only [cursorUpB, Bool.and_eq_true] at h; exact h.1.2)
  intro b
  induction b with
  | zero => intro hb; rw [show a = 0 from by omega] at h; exact h
  | succ m ih =>
      intro hb
      rcases Nat.lt_or_ge m a with hlt | hge
      · rw [show m + 1 = a from by omega]; exact h
      · have hm := ih hge
        have hs : (siter K16 (m + 1) ss).pc ∈ succsOf K16 (siter K16 m ss).pc := by
          rw [siter_succ]; exact sstep_pc_mem_succs K16 _
        rcases Nat.lt_or_ge (siter K16 m ss).pc 274 with hlt2 | hge2
        · exact hstep _ hlt2 hm _ hs
        · rw [show succsOf K16 (siter K16 m ss).pc = [(siter K16 m ss).pc] from by
            simp only [succsOf, Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)],
            List.mem_singleton] at hs
          omega



theorem tail_run64 (E : SState) (h209 : E.pc = 209)
    (hlaN : (E.regs "litAnchor" 0).toNat ≤ 65536)
    (hlau : ∀ j : Lane, E.regs "litAnchor" j = E.regs "litAnchor" 0) :
    (∀ i, i ≤ 10 → (siter K16 i E).pc = 209 + i)
    ∧ (15 ≤ 65536 - (E.regs "litAnchor" 0).toNat →
        (∀ i, i ≤ 13 → (siter K16 i E).pc = 209 + i)
        ∧ (∀ j : Lane, (siter K16 13 E).regs "op" j = E.regs "op" j + 1)
        ∧ (∀ j : Lane, ((siter K16 13 E).regs "litExtraF" j).toNat
            = 65536 - (E.regs "litAnchor" 0).toNat - 15)
        ∧ ((((siter K16 13 E).regs "lsicC" 0) == 1) = true
            ↔ 255 ≤ ((siter K16 13 E).regs "litExtraF" 0).toNat))
    ∧ (65536 - (E.regs "litAnchor" 0).toNat < 15 → (siter K16 11 E).pc = 236)
    ∧ (∀ j : Lane, (siter K16 11 E).regs "op" j = E.regs "op" j + 1)
    ∧ (∀ j : Lane, ((siter K16 2 E).regs "fLen" j).toNat
        = 65536 - (E.regs "litAnchor" 0).toNat) := by
  have e0 : siter K16 0 E = E := rfl
  have p0 : (siter K16 0 E).pc = 209 := by rw [e0]; exact h209
  have p1 : (siter K16 1 E).pc = 210 := by
    rw [siter_succ]; exact tail_pc_mov64 _ 209 "fLen" (SArg.imm 65536) p0 (by decide)
  have p2 : (siter K16 2 E).pc = 211 := by
    rw [siter_succ]; exact tail_pc_bin64 _ 210 (.sub) "fLen" "fLen" (SArg.reg "litAnchor") p1 (by decide)
  have p3 : (siter K16 3 E).pc = 212 := by
    rw [siter_succ]; exact tail_pc_mov64 _ 211 "zero" (SArg.imm 0) p2 (by decide)
  have p4 : (siter K16 4 E).pc = 213 := by
    rw [siter_succ]; exact tail_pc_bin64 _ 212 (.min) "tokHi" "fLen" (SArg.imm 15) p3 (by decide)
  have p5 : (siter K16 5 E).pc = 214 := by
    rw [siter_succ]; exact tail_pc_bin64 _ 213 (.shl) "tok" "tokHi" (SArg.imm 4) p4 (by decide)
  have p6 : (siter K16 6 E).pc = 215 := by
    rw [siter_succ]; exact tail_pc_bin64 _ 214 (.bor) "tok" "tok" (SArg.reg "zero") p5 (by decide)
  have p7 : (siter K16 7 E).pc = 216 := by
    rw [siter_succ]; exact tail_pc_bin64 _ 215 (.add) "sbAddr" "outBase" (SArg.reg "op") p6 (by decide)
  have p8 : (siter K16 8 E).pc = 217 := by
    rw [siter_succ]; exact tail_pc_stg64 _ 216 "sbAddr" "tok" p7 (by decide)
  have p9 : (siter K16 9 E).pc = 218 := by
    rw [siter_succ]; exact tail_pc_bin64 _ 217 (.add) "op" "op" (SArg.imm 1) p8 (by decide)
  have p10 : (siter K16 10 E).pc = 219 := by
    rw [siter_succ]; exact tail_pc_setp64 _ 218 (.ge) "pLitBigF" "fLen" (SArg.imm 15) p9 (by decide)
  have flitAnchor1 : (siter K16 1 E).regs "litAnchor" = (siter K16 0 E).regs "litAnchor" := by
    rw [siter_succ]
    exact tail_frame64 _ 209 (.mov "fLen" (SArg.imm 65536)) "litAnchor" p0 (by decide) (by decide)
  have hla1 : (siter K16 1 E).regs "litAnchor" = E.regs "litAnchor" := by
    rw [flitAnchor1, e0]
  have hf1 : ∀ j : Lane, (siter K16 1 E).regs "fLen" j = UInt64.ofNat 65536 := by
    intro j; rw [siter_succ]
    exact tail_val_mov64 _ 209 "fLen" (SArg.imm 65536) p0 (by decide) j
  have hf2 : ∀ j : Lane, (siter K16 2 E).regs "fLen" j
      = UInt64.ofNat 65536 - E.regs "litAnchor" j := by
    intro j
    rw [siter_succ,
      tail_val_bin64 _ 210 (.sub) "fLen" "fLen" (SArg.reg "litAnchor") p1 (by decide) j]
    show (siter K16 1 E).regs "fLen" j - (siter K16 1 E).regs "litAnchor" j = _
    rw [hf1 j, hla1]
  have ffLen3 : (siter K16 3 E).regs "fLen" = (siter K16 2 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame64 _ 211 (.mov "zero" (SArg.imm 0)) "fLen" p2 (by decide) (by decide)
  have ffLen4 : (siter K16 4 E).regs "fLen" = (siter K16 3 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame64 _ 212 (.bin (.min) "tokHi" "fLen" (SArg.imm 15)) "fLen" p3 (by decide) (by decide)
  have ffLen5 : (siter K16 5 E).regs "fLen" = (siter K16 4 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame64 _ 213 (.bin (.shl) "tok" "tokHi" (SArg.imm 4)) "fLen" p4 (by decide) (by decide)
  have ffLen6 : (siter K16 6 E).regs "fLen" = (siter K16 5 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame64 _ 214 (.bin (.bor) "tok" "tok" (SArg.reg "zero")) "fLen" p5 (by decide) (by decide)
  have ffLen7 : (siter K16 7 E).regs "fLen" = (siter K16 6 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame64 _ 215 (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) "fLen" p6 (by decide) (by decide)
  have ffLen8 : (siter K16 8 E).regs "fLen" = (siter K16 7 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame64 _ 216 (.stg "sbAddr" "tok") "fLen" p7 (by decide) (by decide)
  have ffLen9 : (siter K16 9 E).regs "fLen" = (siter K16 8 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame64 _ 217 (.bin (.add) "op" "op" (SArg.imm 1)) "fLen" p8 (by decide) (by decide)
  have ffLen10 : (siter K16 10 E).regs "fLen" = (siter K16 9 E).regs "fLen" := by
    rw [siter_succ]
    exact tail_frame64 _ 218 (.setp (.ge) "pLitBigF" "fLen" (SArg.imm 15)) "fLen" p9 (by decide) (by decide)
  have hfN : ∀ j : Lane, ((siter K16 2 E).regs "fLen" j).toNat
      = 65536 - (E.regs "litAnchor" 0).toNat := by
    intro j
    rw [hf2 j, hlau j, UInt64.toNat_sub,
      show (UInt64.ofNat 65536).toNat = 65536 from by decide,
      show 2 ^ 64 - (E.regs "litAnchor" 0).toNat + 65536
        = 2 ^ 64 + (65536 - (E.regs "litAnchor" 0).toNat) from by
        have := (E.regs "litAnchor" 0).toNat_lt; omega,
      Nat.add_mod_left, Nat.mod_eq_of_lt (by omega)]
  have hfl9 : ∀ j : Lane, (siter K16 9 E).regs "fLen" j = (siter K16 2 E).regs "fLen" j := by
    intro j; rw [ffLen9, ffLen8, ffLen7, ffLen6, ffLen5, ffLen4, ffLen3]
  have hpbv : (siter K16 10 E).regs "pLitBigF" 0
      = (if SCmp.run (.ge) ((siter K16 9 E).regs "fLen" 0) (UInt64.ofNat 15) then 1 else 0) := by
    rw [siter_succ]
    exact tail_val_setp64 _ 218 (.ge) "pLitBigF" "fLen" (SArg.imm 15) p9 (by decide) 0
  have fop1 : (siter K16 1 E).regs "op" = (siter K16 0 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 209 (.mov "fLen" (SArg.imm 65536)) "op" p0 (by decide) (by decide)
  have fop2 : (siter K16 2 E).regs "op" = (siter K16 1 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 210 (.bin (.sub) "fLen" "fLen" (SArg.reg "litAnchor")) "op" p1 (by decide) (by decide)
  have fop3 : (siter K16 3 E).regs "op" = (siter K16 2 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 211 (.mov "zero" (SArg.imm 0)) "op" p2 (by decide) (by decide)
  have fop4 : (siter K16 4 E).regs "op" = (siter K16 3 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 212 (.bin (.min) "tokHi" "fLen" (SArg.imm 15)) "op" p3 (by decide) (by decide)
  have fop5 : (siter K16 5 E).regs "op" = (siter K16 4 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 213 (.bin (.shl) "tok" "tokHi" (SArg.imm 4)) "op" p4 (by decide) (by decide)
  have fop6 : (siter K16 6 E).regs "op" = (siter K16 5 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 214 (.bin (.bor) "tok" "tok" (SArg.reg "zero")) "op" p5 (by decide) (by decide)
  have fop7 : (siter K16 7 E).regs "op" = (siter K16 6 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 215 (.bin (.add) "sbAddr" "outBase" (SArg.reg "op")) "op" p6 (by decide) (by decide)
  have fop8 : (siter K16 8 E).regs "op" = (siter K16 7 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 216 (.stg "sbAddr" "tok") "op" p7 (by decide) (by decide)
  have hop9 : ∀ j : Lane, (siter K16 9 E).regs "op" j = (siter K16 8 E).regs "op" j + 1 := by
    intro j
    rw [siter_succ, tail_val_bin64 _ 217 (.add) "op" "op" (SArg.imm 1) p8 (by decide) j]
    rfl
  have fop10 : (siter K16 10 E).regs "op" = (siter K16 9 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 218 (.setp (.ge) "pLitBigF" "fLen" (SArg.imm 15)) "op" p9 (by decide) (by decide)
  have fop11 : (siter K16 11 E).regs "op" = (siter K16 10 E).regs "op" := by
    rw [siter_succ]
    exact tail_frame64 _ 219 (.braifnot "pLitBigF" "Le16") "op" p10 (by decide) (by decide)
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
    have hpb : (((siter K16 10 E).regs "pLitBigF" 0) == 1) = true := by
      rw [hpbv]
      exact (setp_ge_iff _ 15 (by decide)).mpr (by rw [hfl9 0, hfN 0]; omega)
    have p11 : (siter K16 11 E).pc = 220 := by
      rw [siter_succ, AlgorithmLib.LZ4WarpDSL.braifnot_step K16 (siter K16 10 E) "pLitBigF" "Le16"
        (by rw [p10]; decide), if_pos hpb]
      simp only [SState.setPc, p10]
    have p12 : (siter K16 12 E).pc = 221 := by
      rw [siter_succ]; exact tail_pc_bin64 _ 220 (.sub) "litExtraF" "fLen" (SArg.imm 15) p11 (by decide)
    have p13 : (siter K16 13 E).pc = 222 := by
      rw [siter_succ]; exact tail_pc_setp64 _ 221 (.ge) "lsicC" "litExtraF" (SArg.imm 255) p12 (by decide)
    have ffLen11 : (siter K16 11 E).regs "fLen" = (siter K16 10 E).regs "fLen" := by
      rw [siter_succ]
      exact tail_frame64 _ 219 (.braifnot "pLitBigF" "Le16") "fLen" p10 (by decide) (by decide)
    have hle12 : ∀ j : Lane, (siter K16 12 E).regs "litExtraF" j
        = (siter K16 11 E).regs "fLen" j - UInt64.ofNat 15 := by
      intro j
      rw [siter_succ,
        tail_val_bin64 _ 220 (.sub) "litExtraF" "fLen" (SArg.imm 15) p11 (by decide) j]
      rfl
    have flitExtraF13 : (siter K16 13 E).regs "litExtraF" = (siter K16 12 E).regs "litExtraF" := by
      rw [siter_succ]
      exact tail_frame64 _ 221 (.setp (.ge) "lsicC" "litExtraF" (SArg.imm 255)) "litExtraF" p12 (by decide) (by decide)
    have hleN : ∀ j : Lane, ((siter K16 13 E).regs "litExtraF" j).toNat
        = 65536 - (E.regs "litAnchor" 0).toNat - 15 := by
      intro j
      rw [show (siter K16 13 E).regs "litExtraF" j = (siter K16 12 E).regs "litExtraF" j from by
        rw [flitExtraF13], hle12 j, ffLen11, ffLen10, hfl9 j,
        uint64_sub_toNat _ 15 (by decide) (by rw [hfN j]; omega), hfN j]
    have fop12 : (siter K16 12 E).regs "op" = (siter K16 11 E).regs "op" := by
      rw [siter_succ]
      exact tail_frame64 _ 220 (.bin (.sub) "litExtraF" "fLen" (SArg.imm 15)) "op" p11 (by decide) (by decide)
    have fop13 : (siter K16 13 E).regs "op" = (siter K16 12 E).regs "op" := by
      rw [siter_succ]
      exact tail_frame64 _ 221 (.setp (.ge) "lsicC" "litExtraF" (SArg.imm 255)) "op" p12 (by decide) (by decide)
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
    · have hlv : (siter K16 13 E).regs "lsicC" 0
          = (if SCmp.run (.ge) ((siter K16 12 E).regs "litExtraF" 0) (UInt64.ofNat 255) then 1 else 0) := by
        rw [siter_succ]
        exact tail_val_setp64 _ 221 (.ge) "lsicC" "litExtraF" (SArg.imm 255) p12 (by decide) 0
      rw [hlv, show (siter K16 13 E).regs "litExtraF" 0 = (siter K16 12 E).regs "litExtraF" 0 from by
        rw [flitExtraF13]]
      exact setp_ge_iff _ 255 (by decide)
  · -- the branch is not taken: `pLitBigF = 0`, and 219 jumps to `Le16` = 236
    have hnb : (((siter K16 10 E).regs "pLitBigF" 0) == 1) = false := by
      rw [hpbv]
      rcases Bool.eq_false_or_eq_true
        ((if SCmp.run (.ge) ((siter K16 9 E).regs "fLen" 0) (UInt64.ofNat 15) then (1:UInt64) else 0) == 1)
        with h | h
      · exfalso
        have h15 := (setp_ge_iff _ 15 (by decide)).mp h
        rw [hfl9 0, hfN 0] at h15
        omega
      · exact h
    rw [siter_succ, AlgorithmLib.LZ4WarpDSL.braifnot_step K16 (siter K16 10 E) "pLitBigF" "Le16"
      (by rw [p10]; decide), if_neg (by rw [hnb]; exact Bool.false_ne_true)]
    simp only [SState.setPc]
    decide


theorem tail_sites_bounded64 (LO : Nat) (hLO : LO < 2 ^ 64) (E : SState) (h209 : E.pc = 209)
    (hlaN : (E.regs "litAnchor" 0).toNat ≤ 65536)
    (hlau : ∀ j : Lane, E.regs "litAnchor" j = E.regs "litAnchor" 0)
    (hopu : ∀ j : Lane, E.regs "op" j = E.regs "op" 0)
    (hbud : (E.regs "op" 0).toNat + (65536 - (E.regs "litAnchor" 0).toNat)
        + (65536 - (E.regs "litAnchor" 0).toNat) / 255 + 2 ≤ LO)
    (l : Lane) (k : Nat)
    (hq : (siter K16 k E).pc = 216 ∨ (siter K16 k E).pc = 226 ∨ (siter K16 k E).pc = 233) :
    ((siter K16 k E).regs "op" l).toNat ≤ LO := by
  obtain ⟨hpcs10, hbr1, hbr0, -, -⟩ := tail_run64 E h209 hlaN hlau
  have hop0 : (E.regs "op" 0).toNat ≤ LO := by omega
  rcases hq with h216 | hlsic
  · have hk7 : k = 7 := by
      rcases Nat.lt_or_ge k 11 with hlt | hge
      · have hpk := hpcs10 k (by omega); rw [h216] at hpk; omega
      · exfalso
        have h217 : 217 ≤ (siter K16 10 E).pc := by rw [hpcs10 10 (by omega)]; omega
        have hst := stays_from_21764 E 10 h217 k (by omega)
        rw [h216] at hst; omega
    obtain ⟨-, hopc⟩ := tail_to_21664 E h209
    rw [hk7, hopc, hopu l]; exact hop0
  · -- the two LSIC stores
    have hfl : 15 ≤ 65536 - (E.regs "litAnchor" 0).toNat := by
      rcases Nat.lt_or_ge (65536 - (E.regs "litAnchor" 0).toNat) 15 with hlt | hge
      · exfalso
        have h236 := hbr0 hlt
        rcases Nat.lt_or_ge k 11 with h1 | h1
        · have hpk := hpcs10 k (by omega)
          rcases hlsic with e | e <;> rw [e] at hpk <;> omega
        · have hst := stays_from_23564 E 11 (by rw [h236]; omega) k h1
          rcases hlsic with e | e <;> rw [e] at hst <;> omega
      · exact hge
    obtain ⟨hpcs13, hopv, hlev, hlsc⟩ := hbr1 hfl
    have hpc222 : (siter K16 13 E).pc = 222 := hpcs13 13 (by omega)
    have hk13 : 13 ≤ k := by
      rcases Nat.lt_or_ge k 13 with hlt | hge
      · exfalso
        have hpk := hpcs13 k (by omega)
        rcases hlsic with e | e <;> rw [e] at hpk <;> omega
      · exact hge
    -- the invariant at the LSIC head
    have hopN : ∀ j : Lane, ((siter K16 13 E).regs "op" j).toNat = (E.regs "op" 0).toNat + 1 := by
      intro j
      rw [hopv j, hopu j]
      have hL := (AlgorithmLib.LZ4Simt.toNat_add_ofNat_of_lt (E.regs "op" 0) 1 (by omega)).1
      rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
      exact hL
    have hdiv : (65536 - (E.regs "litAnchor" 0).toNat - 15) / 255
        ≤ (65536 - (E.regs "litAnchor" 0).toNat) / 255 := Nat.div_le_div_right (by omega)
    have c1 : ((siter K16 13 E).regs "op" l).toNat
        + lsicRem (siter K16 13 E).pc (((siter K16 13 E).regs "litExtraF" l).toNat) ≤ LO := by
      have hr : lsicRem (siter K16 13 E).pc ((siter K16 13 E).regs "litExtraF" l).toNat
          = ((siter K16 13 E).regs "litExtraF" l).toNat / 255 + 1 := by
        rw [hpc222]; rfl
      rw [hr, hopN l, hlev l]
      omega
    have c2 : (siter K16 13 E).regs "litExtraF" l = (siter K16 13 E).regs "litExtraF" 0 := by
      rw [← UInt64.toNat_inj, hlev l, hlev 0]
    have c3 : (siter K16 13 E).pc = 222 ∨ (siter K16 13 E).pc = 223 ∨ (siter K16 13 E).pc = 230 →
        ((((siter K16 13 E).regs "lsicC" 0) == 1) = true
          ↔ 255 ≤ ((siter K16 13 E).regs "litExtraF" 0).toNat) := fun _ => hlsc
    have c4 : 224 ≤ (siter K16 13 E).pc → (siter K16 13 E).pc ≤ 228 →
        255 ≤ ((siter K16 13 E).regs "litExtraF" l).toNat := by
      rw [hpc222]; intro h _; exact absurd h (by omega)
    have hInv : LsicInv l LO (siter K16 13 E) := lsicInv_mk l LO (siter K16 13 E) c1 c2 c3 c4
    have hne : ∀ i, i < k - 13 → (siter K16 i (siter K16 13 E)).pc ∉ [234] := by
      intro i hi hmem
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
      rw [← siter_add K16 13 i E] at hmem
      have hs : (siter K16 (13 + i + 1) E).pc ∈ succsOf K16 (siter K16 (13 + i) E).pc := by
        rw [siter_succ]; exact sstep_pc_mem_succs K16 _
      rw [hmem, show succsOf K16 234 = [235] from by decide, List.mem_singleton] at hs
      have hst := stays_from_23564 E (13 + i + 1) (by rw [hs]; omega) k (by omega)
      rcases hlsic with e | e <;> rw [e] at hst <;> omega
    have hsplit : siter K16 (k - 13) (siter K16 13 E) = siter K16 k E := by
      rw [← siter_add K16 13 (k - 13) E, show 13 + (k - 13) = k from by omega]
    have hfin := lsic_op_lt64 l LO hLO (siter K16 13 E) hpc222 hInv (k - 13) hne
      (by rw [hsplit]; exact hlsic)
    rw [hsplit] at hfin
    exact Nat.le_of_lt hfin

-- ── The tail copy's budget: `op + fLen ≤ lenOff` where the copy is set up ────

theorem stays_from_21164 (ss : SState) (a : Nat) (h : 211 ≤ (siter K16 a ss).pc) :
    ∀ b, a ≤ b → 211 ≤ (siter K16 b ss).pc :=
  stays_ge64 211 (by decide) ss a h


theorem stays_from_23964 (ss : SState) (a : Nat) (h : 239 ≤ (siter K16 a ss).pc) :
    ∀ b, a ≤ b → 239 ≤ (siter K16 b ss).pc :=
  stays_ge64 239 (by decide) ss a h

theorem const_along64 (r : String) (E : SState) : ∀ (n : Nat),
    (∀ i, i < n → ∀ ins, K16[(siter K16 i E).pc]? = some ins → destOf ins ≠ some r) →
    (siter K16 n E).regs r = E.regs r := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ m ih =>
      intro h
      rw [siter_succ, sstep_regs_frame K16 (siter K16 m E) r (h m (by omega)),
        ih (fun i hi => h i (by omega))]

theorem fLen_no_write64 : ∀ q, 211 ≤ q → ∀ ins, K16[q]? = some ins → destOf ins ≠ some "fLen" := by
  have hd : ∀ q, q < 274 → 211 ≤ q → (K16[q]?.map (fun i => destOf i != some "fLen")) = some true := by
    decide
  intro q hq ins hins
  rcases Nat.lt_or_ge q 274 with h | h
  · have h' := hd q h hq
    rw [hins] at h'
    simpa using h'
  · rw [Array.getElem?_eq_none_iff.mpr (by rw [kSize16]; omega)] at hins
    exact absurd hins (by simp)

theorem fLen_const64 (ss : SState) (a : Nat) (ha : 211 ≤ (siter K16 a ss).pc) (b : Nat) (hab : a ≤ b) :
    (siter K16 b ss).regs "fLen" = (siter K16 a ss).regs "fLen" := by
  have hkey : ∀ i, 211 ≤ (siter K16 i (siter K16 a ss)).pc := by
    intro i
    rw [← siter_add K16 a i ss]
    exact stays_from_21164 ss a ha (a + i) (by omega)
  have h := const_along64 "fLen" (siter K16 a ss) (b - a)
    (fun i _ ins hins => fLen_no_write64 _ (hkey i) ins hins)
  rw [← siter_add K16 a (b - a) ss, show a + (b - a) = b from by omega] at h
  exact h

theorem tailFS_closed64 : PcClosed K16 tailFS [238] := by decide

theorem tailFQ_step64 (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (v : UInt64) (st : SState)
    (hs : st.pc ∈ tailFS) (hex : st.pc ∉ [238]) (h : TailFQ l B v st) :
    TailFQ l B v (sstep K16 st) := by
  have dLo : ∀ q ∈ tailFS, q ≤ 233 → ∀ q' ∈ succsOf K16 q, q' ≤ 234 := by decide
  have dHi : ∀ q ∈ tailFS, 235 ≤ q → q ≠ 238 → ∀ q' ∈ succsOf K16 q, 235 ≤ q' := by decide
  have dIn : ∀ q ∈ tailFS, q ≤ 233 → q ∈ lsicFS := by decide
  have dOp : ∀ q ∈ tailFS, 235 ≤ q →
      (K16[q]?.map (fun i => destOf i != some "op")) = some true := by decide
  have d211 : ∀ q ∈ tailFS, 211 ≤ q := by decide
  obtain ⟨h1, h2, h3⟩ := h
  have hq238 : st.pc ≠ 238 := by simpa using hex
  have hsucc : (sstep K16 st).pc ∈ succsOf K16 st.pc := sstep_pc_mem_succs K16 st
  have hfl : (sstep K16 st).regs "fLen" = st.regs "fLen" :=
    sstep_regs_frame K16 st "fLen" (fun ins hins => fLen_no_write64 _ (d211 _ hs) ins hins)
  refine ⟨?_, ?_, by rw [hfl]; exact h3⟩
  · intro hle
    rcases Nat.lt_or_ge st.pc 234 with hlt | hge
    · have hnot : st.pc ∉ [234] := by
        simp only [List.mem_cons, List.not_mem_nil, or_false]
        omega
      exact lsicFS_hstep64 l B hB st (dIn _ hs (by omega)) hnot (h1 (by omega))
    · exfalso
      rcases Nat.lt_or_ge st.pc 235 with h234 | h235
      · rw [show st.pc = 234 from by omega, show succsOf K16 234 = [235] from by decide,
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
        have hv : (sstep K16 st).regs "op" l = st.regs "op" l + 1 := by
          rw [tail_val_bin64 st 234 (.add) "op" "op" (SArg.imm 1) he (by decide) l]; rfl
        have hL := (AlgorithmLib.LZ4Simt.toNat_add_ofNat_of_lt (st.regs "op" l) 1 (by omega)).1
        rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
        rw [hv, hL]
        omega
      · have hop : (sstep K16 st).regs "op" = st.regs "op" :=
          sstep_regs_frame K16 st "op" (fun ins hins => by
            have h' := dOp st.pc hs h235
            rw [hins] at h'
            simpa using h')
        rw [hop]
        exact h2 h235

theorem tail_from_entry64 (l : Lane) (B : Nat) (hB : B < 2 ^ 64) (v : UInt64) (E : SState)
    (m : Nat) (hmem : (siter K16 m E).pc ∈ tailFS) (hQ : TailFQ l B v (siter K16 m E))
    (k : Nat) (hmk : m ≤ k) (hq : (siter K16 k E).pc = 238) :
    ((siter K16 k E).regs "op" l).toNat ≤ B ∧ (siter K16 k E).regs "fLen" l = v := by
  have hne : ∀ i, i < k - m → (siter K16 i (siter K16 m E)).pc ∉ [238] := by
    intro i hi hin
    rw [← siter_add K16 m i E] at hin
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hin
    have hs : (siter K16 (m + i + 1) E).pc ∈ succsOf K16 (siter K16 (m + i) E).pc := by
      rw [siter_succ]; exact sstep_pc_mem_succs K16 _
    rw [hin, show succsOf K16 238 = [239] from by decide, List.mem_singleton] at hs
    have hst := stays_from_23964 E (m + i + 1) (by rw [hs]; omega) k (by omega)
    rw [hq] at hst; omega
  have hfin := inv_on K16 (TailFQ l B v) tailFS [238] tailFS_closed64
    (fun s hsm hexs hh => tailFQ_step64 l B hB v s hsm hexs hh) (siter K16 m E) hmem hQ (k - m) hne
  rw [← siter_add K16 m (k - m) E, show m + (k - m) = k from by omega] at hfin
  exact ⟨hfin.2.1 (by rw [hq]; omega), hfin.2.2⟩

set_option maxHeartbeats 1000000 in
theorem tail_copy_budget64 (LO : Nat) (hLO : LO < 2 ^ 64) (E : SState) (h209 : E.pc = 209)
    (hlaN : (E.regs "litAnchor" 0).toNat ≤ 65536)
    (hlau : ∀ j : Lane, E.regs "litAnchor" j = E.regs "litAnchor" 0)
    (hopu : ∀ j : Lane, E.regs "op" j = E.regs "op" 0)
    (hbud : (E.regs "op" 0).toNat + (65536 - (E.regs "litAnchor" 0).toNat)
        + (65536 - (E.regs "litAnchor" 0).toNat) / 255 + 2 ≤ LO)
    (l : Lane) (k : Nat) (hq : (siter K16 k E).pc = 238) :
    ((siter K16 k E).regs "op" l).toNat + ((siter K16 k E).regs "fLen" l).toNat ≤ LO := by
  obtain ⟨hpcs10, hbr1, hbr0, hop11, hfN⟩ := tail_run64 E h209 hlaN hlau
  -- the visit is past the straight run out of 209
  have hk11 : 11 ≤ k := by
    rcases Nat.lt_or_ge k 11 with hlt | hge
    · have := hpcs10 k (by omega); rw [hq] at this; omega
    · exact hge
  have hpc2 : 211 ≤ (siter K16 2 E).pc := by rw [hpcs10 2 (by omega)]; omega
  have hfl11 : (siter K16 11 E).regs "fLen" = (siter K16 2 E).regs "fLen" :=
    fLen_const64 E 2 hpc2 11 (by omega)
  have hop0 : ((E.regs "op" 0).toNat + 1) < 2 ^ 64 := by omega
  -- the two entries into the region
  have hmain : ∀ m : Nat, m ≤ k → (siter K16 m E).pc ∈ tailFS →
      TailFQ l (LO - (65536 - (E.regs "litAnchor" 0).toNat))
        ((siter K16 2 E).regs "fLen" l) (siter K16 m E) →
      ((siter K16 k E).regs "op" l).toNat + ((siter K16 k E).regs "fLen" l).toNat ≤ LO := by
    intro m hmk hmem hQ
    obtain ⟨hoop, hffl⟩ := tail_from_entry64 l _ (by omega) _ E m hmem hQ k hmk hq
    rw [hffl, hfN l]
    omega
  rcases Nat.lt_or_ge (65536 - (E.regs "litAnchor" 0).toNat) 15 with hshort | hlong
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
    have hpc222 : (siter K16 13 E).pc = 222 := hpcs13 13 (by omega)
    have hk13 : 13 ≤ k := by
      rcases Nat.lt_or_ge k 13 with hlt | hge
      · have := hpcs13 k (by omega); rw [hq] at this; omega
      · exact hge
    have hopN : ∀ j : Lane, ((siter K16 13 E).regs "op" j).toNat = (E.regs "op" 0).toNat + 1 := by
      intro j
      rw [hopv j, hopu j]
      have hL := (AlgorithmLib.LZ4Simt.toNat_add_ofNat_of_lt (E.regs "op" 0) 1 (by omega)).1
      rw [show (UInt64.ofNat 1) = 1 from rfl] at hL
      exact hL
    have hdiv : (65536 - (E.regs "litAnchor" 0).toNat - 15) / 255
        ≤ (65536 - (E.regs "litAnchor" 0).toNat) / 255 := Nat.div_le_div_right (by omega)
    have c1 : ((siter K16 13 E).regs "op" l).toNat
        + lsicRem (siter K16 13 E).pc (((siter K16 13 E).regs "litExtraF" l).toNat)
        ≤ LO - (65536 - (E.regs "litAnchor" 0).toNat) := by
      rw [show lsicRem (siter K16 13 E).pc ((siter K16 13 E).regs "litExtraF" l).toNat
          = ((siter K16 13 E).regs "litExtraF" l).toNat / 255 + 1 from by rw [hpc222]; rfl,
        hopN l, hlev l]
      omega
    have c2 : (siter K16 13 E).regs "litExtraF" l = (siter K16 13 E).regs "litExtraF" 0 := by
      rw [← UInt64.toNat_inj, hlev l, hlev 0]
    have c3 : (siter K16 13 E).pc = 222 ∨ (siter K16 13 E).pc = 223 ∨ (siter K16 13 E).pc = 230 →
        ((((siter K16 13 E).regs "lsicC" 0) == 1) = true
          ↔ 255 ≤ ((siter K16 13 E).regs "litExtraF" 0).toNat) := fun _ => hlsc
    have c4 : 224 ≤ (siter K16 13 E).pc → (siter K16 13 E).pc ≤ 228 →
        255 ≤ ((siter K16 13 E).regs "litExtraF" l).toNat := by
      rw [hpc222]; intro h _; exact absurd h (by omega)
    refine hmain 13 hk13 (by rw [hpc222]; decide)
      ⟨fun _ => lsicInv_mk l _ (siter K16 13 E) c1 c2 c3 c4,
        fun hge => absurd hge (by rw [hpc222]; omega), ?_⟩
    rw [fLen_const64 E 2 hpc2 13 (by omega)]

-- ── From the match-sequence entry to the token region ────────────────────────

theorem lsicLen_eq_encNib64 (n : Nat) : lsicLen n = (AlgorithmLib.LZ4.encNib n).length := by
  rw [AlgorithmLib.LZ4Imp.encNib_length, lsicLen]
  by_cases h : 15 ≤ n
  · rw [if_pos h, if_neg (by omega)]
  · rw [if_neg h, if_pos (by omega)]

theorem me_pc064 (st : SState) (he : st.pc = 124) : (siter K16 0 st).pc = 124 := he

theorem me_pc164 (st : SState) (he : st.pc = 124) : (siter K16 1 st).pc = 125 := by
  rw [siter_succ]; exact tail_pc_bin64 _ 124 (.sub) "mlm" "ml" (SArg.imm 4) (me_pc064 st he) (by decide)

theorem me_pc264 (st : SState) (he : st.pc = 124) : (siter K16 2 st).pc = 126 := by
  rw [siter_succ]; exact tail_pc_bin64 _ 125 (.min) "tokLo" "mlm" (SArg.imm 15) (me_pc164 st he) (by decide)

theorem me_pc364 (st : SState) (he : st.pc = 124) : (siter K16 3 st).pc = 127 := by
  rw [siter_succ]; exact tail_pc_bin64 _ 126 (.min) "tokHi" "litLen" (SArg.imm 15) (me_pc264 st he) (by decide)

theorem me_pc464 (st : SState) (he : st.pc = 124) : (siter K16 4 st).pc = 128 := by
  rw [siter_succ]; exact tail_pc_bin64 _ 127 (.shl) "tok" "tokHi" (SArg.imm 4) (me_pc364 st he) (by decide)

theorem me_pc564 (st : SState) (he : st.pc = 124) : (siter K16 5 st).pc = 129 := by
  rw [siter_succ]; exact tail_pc_bin64 _ 128 (.bor) "tok" "tok" (SArg.reg "tokLo") (me_pc464 st he) (by decide)

theorem me_frame64 (st : SState) (he : st.pc = 124) (r : String)
    (h1 : r ≠ "mlm") (h2 : r ≠ "tokLo") (h3 : r ≠ "tokHi") (h4 : r ≠ "tok") :
    (siter K16 5 st).regs r = st.regs r := by
  have f1 : (siter K16 1 st).regs r = (siter K16 0 st).regs r := by
    rw [siter_succ]
    exact tail_frame64 _ 124 (.bin (.sub) "mlm" "ml" (SArg.imm 4)) r (me_pc064 st he) (by decide)
      (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h1)
  have f2 : (siter K16 2 st).regs r = (siter K16 1 st).regs r := by
    rw [siter_succ]
    exact tail_frame64 _ 125 (.bin (.min) "tokLo" "mlm" (SArg.imm 15)) r (me_pc164 st he) (by decide)
      (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h2)
  have f3 : (siter K16 3 st).regs r = (siter K16 2 st).regs r := by
    rw [siter_succ]
    exact tail_frame64 _ 126 (.bin (.min) "tokHi" "litLen" (SArg.imm 15)) r (me_pc264 st he) (by decide)
      (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h3)
  have f4 : (siter K16 4 st).regs r = (siter K16 3 st).regs r := by
    rw [siter_succ]
    exact tail_frame64 _ 127 (.bin (.shl) "tok" "tokHi" (SArg.imm 4)) r (me_pc364 st he) (by decide)
      (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h4)
  have f5 : (siter K16 5 st).regs r = (siter K16 4 st).regs r := by
    rw [siter_succ]
    exact tail_frame64 _ 128 (.bin (.bor) "tok" "tok" (SArg.reg "tokLo")) r (me_pc464 st he) (by decide)
      (by simpa only [AlgorithmLib.LZ4WarpDSL.wtgt, ne_eq, Option.some.injEq] using Ne.symm h4)
  rw [f5, f4, f3, f2, f1]
  rfl

theorem me_mlm64 (st : SState) (he : st.pc = 124) (l : Lane) :
    (siter K16 5 st).regs "mlm" l = st.regs "ml" l - 4 := by
  have f1 : (siter K16 1 st).regs "mlm" l = st.regs "ml" l - 4 := by
    rw [siter_succ, tail_val_bin64 _ 124 (.sub) "mlm" "ml" (SArg.imm 4) (me_pc064 st he) (by decide) l]
    rfl
  have f2 : (siter K16 2 st).regs "mlm" = (siter K16 1 st).regs "mlm" := by
    rw [siter_succ]
    exact tail_frame64 _ 125 (.bin (.min) "tokLo" "mlm" (SArg.imm 15)) "mlm" (me_pc164 st he) (by decide) (by decide)
  have f3 : (siter K16 3 st).regs "mlm" = (siter K16 2 st).regs "mlm" := by
    rw [siter_succ]
    exact tail_frame64 _ 126 (.bin (.min) "tokHi" "litLen" (SArg.imm 15)) "mlm" (me_pc264 st he) (by decide) (by decide)
  have f4 : (siter K16 4 st).regs "mlm" = (siter K16 3 st).regs "mlm" := by
    rw [siter_succ]
    exact tail_frame64 _ 127 (.bin (.shl) "tok" "tokHi" (SArg.imm 4)) "mlm" (me_pc364 st he) (by decide) (by decide)
  have f5 : (siter K16 5 st).regs "mlm" = (siter K16 4 st).regs "mlm" := by
    rw [siter_succ]
    exact tail_frame64 _ 128 (.bin (.bor) "tok" "tok" (SArg.reg "tokLo")) "mlm" (me_pc464 st he) (by decide) (by decide)
  rw [show (siter K16 5 st).regs "mlm" l = (siter K16 5 st).regs "mlm" l from rfl, f5, f4, f3, f2, f1]

theorem tokInv_at_entry64 (l : Lane) (LO : Nat) (st : SState) (he : st.pc = 124)
    (h : AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 16).inStride LO st) :
    TokInv l LO (siter K16 5 st) := by
  obtain ⟨w, hc, -, -, hml4, hbnd⟩ := h
  have hmem : ∀ r ∈ ["op", "litLen", "litExtra", "matExtra", "lsicC", "ml"],
      r ∈ "p0" :: "cand0" :: AlgorithmLib.LZ4WarpDSL.loopR := by decide
  have hR : ∀ (r : String), r = "op" ∨ r = "litLen" ∨ r = "litExtra" ∨ r = "matExtra"
      ∨ r = "lsicC" → ∀ j : Lane,
      (siter K16 5 st).regs r j = w.regs r := by
    intro r hr j
    have hne : r ≠ "mlm" ∧ r ≠ "tokLo" ∧ r ≠ "tokHi" ∧ r ≠ "tok" := by
      rcases hr with rfl | rfl | rfl | rfl | rfl <;> exact ⟨by decide, by decide, by decide, by decide⟩
    rw [show (siter K16 5 st).regs r j = st.regs r j from by
      rw [me_frame64 st he r hne.1 hne.2.1 hne.2.2.1 hne.2.2.2]]
    exact hc.reg (by rcases hr with rfl | rfl | rfl | rfl | rfl <;> decide) j
  have hMLM : ∀ j : Lane,
      ((siter K16 5 st).regs "mlm" j).toNat
        = (w.regs "ml").toNat - 4 := by
    intro j
    rw [me_mlm64 st he j, hc.reg (show "ml" ∈ "p0" :: "cand0" :: AlgorithmLib.LZ4WarpDSL.loopR
      by decide) j, show (4 : UInt64) = UInt64.ofNat 4 from rfl,
      uint64_sub_toNat (w.regs "ml") 4 (by decide) hml4]
  have hMLMu : ∀ j : Lane,
      (siter K16 5 st).regs "mlm" j
        = (siter K16 5 st).regs "mlm" 0 := by
    intro j
    rw [me_mlm64 st he j, me_mlm64 st he 0,
      hc.reg (show "ml" ∈ "p0" :: "cand0" :: AlgorithmLib.LZ4WarpDSL.loopR by decide) j,
      hc.reg (show "ml" ∈ "p0" :: "cand0" :: AlgorithmLib.LZ4WarpDSL.loopR by decide) 0]
  have hpc := me_pc564 st he
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpc, hR "op" (Or.inl rfl) l, hR "litLen" (Or.inr (Or.inl rfl)) l,
      hR "litExtra" (Or.inr (Or.inr (Or.inl rfl))) l,
      hR "matExtra" (Or.inr (Or.inr (Or.inr (Or.inl rfl)))) l, hMLM l]
    show (w.regs "op").toNat + (1 + lsicLen (w.regs "litLen").toNat + (w.regs "litLen").toNat
      + 2 + lsicLen ((w.regs "ml").toNat - 4)) ≤ LO
    rw [lsicLen_eq_encNib64, lsicLen_eq_encNib64]
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



theorem tok_exits64 : AlgorithmLib.LZ4Simt.succsOf K16 197 = [199]
    ∧ AlgorithmLib.LZ4Simt.succsOf K16 198 = [199] := by decide

theorem tok_sites_bounded64 (LO : Nat) (hLO : LO < 2 ^ 64) (ss : SState)
    (h0 : ss.pc ∉ tokRegion)
    (l : Lane) (k : Nat)
    (hck : ∀ j, j ≤ k → (siter K16 j ss).pc = 124 →
      AlgorithmLib.LZ4WarpDSL.MatchEntryQ (WP.mk 16).inStride LO (siter K16 j ss))
    (hq : (siter K16 k ss).pc = 130 ∨ (siter K16 k ss).pc = 140 ∨ (siter K16 k ss).pc = 147
      ∨ (siter K16 k ss).pc = 173 ∨ (siter K16 k ss).pc = 178 ∨ (siter K16 k ss).pc = 188
      ∨ (siter K16 k ss).pc = 195) :
    ((siter K16 k ss).regs "op" l).toNat ≤ LO := by
  have hmem : (siter K16 k ss).pc ∈ tokRegion := by
    rcases hq with e | e | e | e | e | e | e <;> rw [e] <;> decide
  obtain ⟨j, hjk, hje, hall⟩ := region_entry K16 tokRegion 124 tokRegion_entry64 ss h0 k hmem
  -- the five prologue instructions of the match sequence
  have hp1 := me_pc164 (siter K16 j ss) hje
  have hp2 := me_pc264 (siter K16 j ss) hje
  have hp3 := me_pc364 (siter K16 j ss) hje
  have hp4 := me_pc464 (siter K16 j ss) hje
  have hp5 := me_pc564 (siter K16 j ss) hje
  have hshift : ∀ a : Nat, siter K16 (j + a) ss = siter K16 a (siter K16 j ss) := by
    intro a; exact siter_add K16 j a ss
  have hstep : ∀ a : Nat, a ≤ 5 → (siter K16 (j + a) ss).pc = 124 + a := by
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
  have hTok : TokInv l LO (siter K16 (j + 5) ss) := by
    rw [hshift 5]
    exact tokInv_at_entry64 l LO (siter K16 j ss) hje (hck j hjk hje)
  have hpc129 : (siter K16 (j + 5) ss).pc = 129 := by rw [hshift 5]; exact hp5 ▸ rfl
  have hne : ∀ i, i < k - (j + 5) → (siter K16 i (siter K16 (j + 5) ss)).pc ∉ [197, 198] := by
    intro i hi hin
    rw [← siter_add K16 (j + 5) i ss, show j + 5 + i = j + (5 + i) from by omega] at hin
    have hnext : (siter K16 (j + (5 + i) + 1) ss).pc ∈ tokRegion :=
      hall (j + (5 + i) + 1) (by omega) (by omega)
    have hs : (siter K16 (j + (5 + i) + 1) ss).pc
        ∈ AlgorithmLib.LZ4Simt.succsOf K16 (siter K16 (j + (5 + i)) ss).pc := by
      rw [siter_succ]; exact AlgorithmLib.LZ4Simt.sstep_pc_mem_succs K16 _
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hin
    rcases hin with e | e
    · rw [e, tok_exits64.1, List.mem_singleton] at hs; rw [hs] at hnext; exact absurd hnext (by decide)
    · rw [e, tok_exits64.2, List.mem_singleton] at hs; rw [hs] at hnext; exact absurd hnext (by decide)
  have hsite : (siter K16 (k - (j + 5)) (siter K16 (j + 5) ss)).pc = 130
      ∨ (siter K16 (k - (j + 5)) (siter K16 (j + 5) ss)).pc = 140
      ∨ (siter K16 (k - (j + 5)) (siter K16 (j + 5) ss)).pc = 147
      ∨ (siter K16 (k - (j + 5)) (siter K16 (j + 5) ss)).pc = 173
      ∨ (siter K16 (k - (j + 5)) (siter K16 (j + 5) ss)).pc = 178
      ∨ (siter K16 (k - (j + 5)) (siter K16 (j + 5) ss)).pc = 188
      ∨ (siter K16 (k - (j + 5)) (siter K16 (j + 5) ss)).pc = 195 := by
    rw [← siter_add K16 (j + 5) (k - (j + 5)) ss, show j + 5 + (k - (j + 5)) = k from by omega]
    exact hq
  have hfin := tok_op_lt64 l LO hLO (siter K16 (j + 5) ss) hpc129 hTok (k - (j + 5)) hne hsite
  rw [← siter_add K16 (j + 5) (k - (j + 5)) ss, show j + 5 + (k - (j + 5)) = k from by omega] at hfin
  exact Nat.le_of_lt hfin



-- p1


end Lz4Sites
