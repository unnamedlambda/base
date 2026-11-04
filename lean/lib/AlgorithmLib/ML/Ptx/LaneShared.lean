module
public import AlgorithmLib.ML.Ptx.LaneLaws
meta import AlgorithmLib.ML.Ptx.LaneLaws
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.LaneShared` — shared memory, as the warp has it and as a block's stores leave it

The warp's run makes shared-memory events (`shEvLeaf`): a shared store, at
the state the run reaches it, and a barrier. Its shared memory is its events
applied in order (`run_smem`), every event's state holding the events before
it applied (`ev_smem`); a shared store is its lanes' two-byte writes, lane by
lane (`storeH_flat`, `wrF`).

A block's threads record their shared stores tagged with the barriers before
them (`sharedStores_tag`, `tagFrom`); a lane's are the warp's events' writes
for it (`tagE`) wherever every leaf its run reaches tags alike (`lane_tags`).
The laws' snapshot after a barrier applies the block's stores phase by phase,
thread by thread (`snapMem_fun`), the warp statement by statement
(`warp_bar`); a byte's value is its last write (`applyF_point`), and within
one epoch the two orders write each byte alike where no byte is written by two
lanes at two different events (`ShOk`, `epoch_byte`, from `flatMap_comm`). So
the snapshot is the warp's shared memory there (`snap_law`), and the laws
answer a one-warp block's lanes the warp's answers, barriers included
(`block_laws`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyLane

namespace AlgorithmLib.ML.LaneShared

open LaneWarp LaneSettle

/-- A shared-memory event of the warp's run: a shared store, at the state the
    run reaches it, or a barrier. -/
inductive ShEv where
  | st (ws : WSt F32) (a v g : IExp)
  | bar (ws : WSt F32)

/-- The state the warp's run reaches an event at. -/
def ShEv.ws : ShEv → WSt F32
  | .st ws _ _ _ => ws
  | .bar ws => ws

section
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat)

/-- The shared-memory events at a leaf. -/
def shEvLeaf : Nat → EWStmt → WSt F32 → List ShEv
  | _, .stH a v g, ws => [.st ws a v g]
  | _, .barrier, ws => [.bar ws]
  | _, _, _ => []

/-- Shared memory after an event. -/
def shApply (m : Nat → UInt8) : ShEv → Nat → UInt8
  | .st ws a v g => ({ ws.int with smem := m }.storeH (fun l => a.eval ws.int cta l) (fun l => v.eval ws.int cta l)
      (fun l => g.eval ws.int cta l)).smem
  | .bar _ => m

/-- Every event's state holds the events before it applied to `m0`. -/
def Good (m0 : Nat → UInt8) (E : List ShEv) : Prop :=
  ∀ p ev, E[p]? = some ev → ev.ws.int.smem = (E.take p).foldl (shApply cta) m0

theorem good_append {m0 : Nat → UInt8} {A B : List ShEv} (hA : Good cta m0 A)
    (hB : Good cta (A.foldl (shApply cta) m0) B) : Good cta m0 (A ++ B) := by
  intro p ev h
  by_cases hp : p < A.length
  · rw [List.getElem?_append_left hp] at h
    rw [List.take_append_of_le_length (by omega)]
    exact hA p ev h
  · have hp' : A.length ≤ p := by omega
    rw [List.getElem?_append_right hp'] at h
    rw [hB _ ev h, List.take_append, List.take_of_length_le hp', List.foldl_append]

theorem foldl_pair_smem (m0 : Nat → UInt8) (body : Nat → WSt F32 → WSt F32) (tr : Nat → WSt F32 → List ShEv)
    (hb : ∀ j ws, (body j ws).int.smem = (tr j ws).foldl (shApply cta) ws.int.smem) :
    ∀ (L : List Nat) (acc : List ShEv) (cur : WSt F32), cur.int.smem = acc.foldl (shApply cta) m0 →
      (L.foldl (fun (p : List ShEv × WSt F32) j => (p.1 ++ tr j p.2, body j p.2)) (acc, cur)).2.int.smem =
        (L.foldl (fun (p : List ShEv × WSt F32) j => (p.1 ++ tr j p.2, body j p.2)) (acc, cur)).1.foldl
          (shApply cta) m0
  | [], _, _, h => h
  | j :: L, acc, cur, h => by
      simp only [List.foldl_cons]
      apply foldl_pair_smem m0 body tr hb L
      rw [hb, h, List.foldl_append]

theorem while_smem (m0 : Nat → UInt8) (c : IExp) (cs : List Nat) (b : EWStmt) (ws0 : WSt F32) (i : Nat)
    (hb : ∀ ws, (run hw cta im b ws i).int.smem = (trace hw cta im shEvLeaf i b ws).foldl (shApply cta) ws.int.smem) :
    ∀ (L : List Nat) (acc : List ShEv) (cur : WSt F32), cur.int.smem = acc.foldl (shApply cta) m0 →
      (L.foldl (fun (p : List ShEv × WSt F32) _ =>
          if c.eval p.2.int cta lane0 ≠ 0 then
            (p.1 ++ trace hw cta im shEvLeaf i b p.2,
              (run hw cta im b p.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b p.2 i).int))
          else p) (acc, cur)).2.int.smem =
      (L.foldl (fun (p : List ShEv × WSt F32) _ =>
          if c.eval p.2.int cta lane0 ≠ 0 then
            (p.1 ++ trace hw cta im shEvLeaf i b p.2,
              (run hw cta im b p.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b p.2 i).int))
          else p) (acc, cur)).1.foldl (shApply cta) m0
  | [], _, _, h => h
  | j :: L, acc, cur, h => by
      simp only [List.foldl_cons]
      split
      · apply while_smem m0 c cs b ws0 i hb L
        show (run hw cta im b cur i).int.smem = _
        rw [hb, h, List.foldl_append]
      · exact while_smem m0 c cs b ws0 i hb L acc cur h

/-- **The warp's shared memory is its events applied in order.** -/
theorem run_smem : ∀ (s : EWStmt) (i : Nat) (ws : WSt F32),
    (run hw cta im s ws i).int.smem = (trace hw cta im shEvLeaf i s ws).foldl (shApply cta) ws.int.smem := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro i ws
      simp only [trace, List.foldl_append]
      rw [← iha i ws]; exact ihb i _
  | iif c a b iha ihb =>
      intro i ws
      by_cases hc : c.eval ws.int cta lane0 ≠ 0
      · have hc' : c.eval ws.int cta ⟨0, by decide⟩ ≠ 0 := hc
        simp only [trace, LaneWarp.run, EWStmt.run, if_pos hc, if_pos hc']; exact iha i ws
      · have hc' : ¬ c.eval ws.int cta ⟨0, by decide⟩ ≠ 0 := hc
        simp only [trace, LaneWarp.run, EWStmt.run, if_neg hc, if_neg hc']; exact ihb i ws
  | forN n b ih =>
      intro i ws
      simp only [trace]
      rw [show run hw cta im (.forN n b) ws i = ((List.range n).foldl (fun (p : List ShEv × WSt F32) j =>
        (p.1 ++ trace hw cta im shEvLeaf j b p.2, run hw cta im b p.2 j)) ([], ws)).2 from
        (forN_state hw cta im shEvLeaf b _ ([], ws)).symm]
      exact foldl_pair_smem cta ws.int.smem _ _ (fun j ws => ih j ws) _ [] ws rfl
  | forM bu a b ih =>
      intro i ws
      simp only [trace]
      rw [show run hw cta im (.forM bu a b) ws i = ((List.range (im bu a)).foldl (fun (p : List ShEv × WSt F32) j =>
        (p.1 ++ trace hw cta im shEvLeaf j b p.2, run hw cta im b p.2 j)) ([], ws)).2 from
        (forN_state hw cta im shEvLeaf b _ ([], ws)).symm]
      exact foldl_pair_smem cta ws.int.smem _ _ (fun j ws => ih j ws) _ [] ws rfl
  | iwhile n c cs b ih =>
      intro i ws
      simp only [trace]
      rw [run_iwhile, ← iwhile_state hw cta im shEvLeaf c cs b ws i _ ([], ws)]
      exact while_smem hw cta im ws.int.smem c cs b ws i (fun ws => ih i ws) _ [] ws rfl
  | locals rs b ih => intro i ws; exact ih i ws
  | stH a v g => intro i ws; rfl
  | stB a v g => intro i ws; exact (storeB_regs _ _ _ _).2.1
  | storeLane b ix r => intro i ws; exact congrArg (fun s => s.smem) (storeLanes_int b _ r ws).1
  | _ => intro i ws; rfl

theorem good_foldl_pair (m0 : Nat → UInt8) (b : EWStmt) (tr : Nat → WSt F32 → List ShEv) (body : Nat → WSt F32 → WSt F32)
    (hb : ∀ j ws, (body j ws).int.smem = (tr j ws).foldl (shApply cta) ws.int.smem)
    (hg : ∀ j ws, Good cta ws.int.smem (tr j ws)) :
    ∀ (L : List Nat) (acc : List ShEv) (cur : WSt F32), Good cta m0 acc → cur.int.smem = acc.foldl (shApply cta) m0 →
      Good cta m0 (L.foldl (fun (p : List ShEv × WSt F32) j => (p.1 ++ tr j p.2, body j p.2)) (acc, cur)).1
  | [], _, _, hA, _ => hA
  | j :: L, acc, cur, hA, h => by
      simp only [List.foldl_cons]
      refine good_foldl_pair m0 b tr body hb hg L _ _ (good_append cta hA (by rw [← h]; exact hg j cur)) ?_
      rw [hb, h, List.foldl_append]

theorem good_while (m0 : Nat → UInt8) (c : IExp) (cs : List Nat) (b : EWStmt) (ws0 : WSt F32) (i : Nat)
    (hb : ∀ ws, (run hw cta im b ws i).int.smem = (trace hw cta im shEvLeaf i b ws).foldl (shApply cta) ws.int.smem)
    (hg : ∀ ws, Good cta ws.int.smem (trace hw cta im shEvLeaf i b ws)) :
    ∀ (L : List Nat) (acc : List ShEv) (cur : WSt F32), Good cta m0 acc → cur.int.smem = acc.foldl (shApply cta) m0 →
      Good cta m0 (L.foldl (fun (p : List ShEv × WSt F32) _ =>
          if c.eval p.2.int cta lane0 ≠ 0 then
            (p.1 ++ trace hw cta im shEvLeaf i b p.2,
              (run hw cta im b p.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b p.2 i).int))
          else p) (acc, cur)).1
  | [], _, _, hA, _ => hA
  | j :: L, acc, cur, hA, h => by
      simp only [List.foldl_cons]
      split
      · refine good_while m0 c cs b ws0 i hb hg L _ _ (good_append cta hA (by rw [← h]; exact hg cur)) ?_
        show (run hw cta im b cur i).int.smem = _
        rw [hb, h, List.foldl_append]
      · exact good_while m0 c cs b ws0 i hb hg L acc cur hA h

/-- **Each event's state holds the events before it applied.** -/
theorem ev_smem : ∀ (s : EWStmt) (i : Nat) (ws : WSt F32), Good cta ws.int.smem (trace hw cta im shEvLeaf i s ws) := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro i ws
      simp only [trace]
      exact good_append cta (iha i ws) (by rw [← run_smem]; exact ihb i _)
  | iif c a b iha ihb =>
      intro i ws
      simp only [trace]
      split
      · exact iha i ws
      · exact ihb i ws
  | forN n b ih =>
      intro i ws
      simp only [trace]
      exact good_foldl_pair cta ws.int.smem b _ _ (fun j ws => run_smem hw cta im b j ws) (fun j ws => ih j ws) _ []
        ws (fun p ev h => by simp at h) rfl
  | forM bu a b ih =>
      intro i ws
      simp only [trace]
      exact good_foldl_pair cta ws.int.smem b _ _ (fun j ws => run_smem hw cta im b j ws) (fun j ws => ih j ws) _ []
        ws (fun p ev h => by simp at h) rfl
  | iwhile n c cs b ih =>
      intro i ws
      simp only [trace]
      exact good_while hw cta im ws.int.smem c cs b ws i (fun ws => run_smem hw cta im b i ws) (fun ws => ih i ws) _ []
        ws (fun p ev h => by simp at h) rfl
  | locals rs b ih => intro i ws; exact ih i ws
  | stH a v g =>
      intro i ws p ev h
      simp only [trace, shEvLeaf] at h
      cases p with
      | zero => simp at h; subst h; rfl
      | succ p => simp at h
  | barrier =>
      intro i ws p ev h
      simp only [trace, shEvLeaf] at h
      cases p with
      | zero => simp at h; subst h; rfl
      | succ p => simp at h
  | _ => intro i ws p ev h; simp [trace, shEvLeaf] at h

end

/-! ## Two orders of one epoch's stores -/

/-- A flat map whose function is empty past one element of a list holding it
    once is that element's. -/
theorem flatMap_single {α β : Type} [DecidableEq α] (L : List α) (h : α → List β) (a0 : α) (hn : L.Nodup)
    (hz : ∀ a ∈ L, a ≠ a0 → h a = []) (h0 : a0 ∈ L) : L.flatMap h = h a0 := by
  induction L with
  | nil => cases h0
  | cons a L ih =>
    simp only [List.flatMap_cons]
    have hnd := List.nodup_cons.mp hn
    by_cases e : a = a0
    · subst e
      rw [List.flatMap_eq_nil_iff.mpr (fun x hx => hz x (List.mem_cons_of_mem _ hx) (fun h => hnd.1 (h ▸ hx))),
        List.append_nil]
    · rw [hz a List.mem_cons_self e, List.nil_append]
      exact ih hnd.2 (fun x hx hne => hz x (List.mem_cons_of_mem _ hx) hne)
        (by rcases List.mem_cons.mp h0 with h | h
            · exact absurd h.symm e
            · exact h)

theorem flatMap_congr' {α β : Type} {L : List α} {f g : α → List β} (h : ∀ a ∈ L, f a = g a) :
    L.flatMap f = L.flatMap g := by
  induction L with
  | nil => rfl
  | cons a L ih =>
    simp only [List.flatMap_cons]
    rw [h a List.mem_cons_self, ih (fun x hx => h x (List.mem_cons_of_mem _ hx))]

/-- **Two orders of a matrix's entries agree** where no two nonempty entries
    lie in different rows and different columns: row by row, or column by
    column. -/
theorem flatMap_comm {β : Type} (I L : List Nat) (hI : I.Nodup) (hL : L.Nodup) (g : Nat → Nat → List β)
    (hc : ∀ i ∈ I, ∀ j ∈ I, ∀ l ∈ L, ∀ l' ∈ L, g i l ≠ [] → g j l' ≠ [] → i = j ∨ l = l') :
    I.flatMap (fun i => L.flatMap (g i)) = L.flatMap (fun l => I.flatMap (fun i => g i l)) := by
  by_cases hex : ∃ i ∈ I, ∃ l ∈ L, g i l ≠ []
  · obtain ⟨i0, hi0, l0, hl0, h0⟩ := hex
    by_cases hrow : ∃ i ∈ I, ∃ l ∈ L, g i l ≠ [] ∧ l ≠ l0
    · -- every nonempty entry is in row `i0`
      obtain ⟨i1, hi1, l1, hl1, h1, hne⟩ := hrow
      have hi10 : i1 = i0 := by
        rcases hc i1 hi1 i0 hi0 l1 hl1 l0 hl0 h1 h0 with h | h
        · exact h
        · exact absurd h hne
      have hz : ∀ i ∈ I, i ≠ i0 → ∀ l ∈ L, g i l = [] := by
        intro i hi hne' l hl
        refine Classical.byContradiction fun hg => ?_
        rcases hc i hi i0 hi0 l hl l0 hl0 hg h0 with h | h
        · exact hne' h
        · subst h
          rcases hc i hi i1 hi1 l hl l1 hl1 hg h1 with h' | h'
          · exact hne' (h'.trans hi10)
          · exact hne h'.symm
      rw [flatMap_single I _ i0 hI (fun i hi hne' => by
        rw [List.flatMap_eq_nil_iff]; exact fun l hl => hz i hi hne' l hl) hi0]
      exact flatMap_congr' fun l hl => (flatMap_single I (fun i => g i l) i0 hI
        (fun i hi hne' => hz i hi hne' l hl) hi0).symm
    · -- every nonempty entry is in column `l0`
      have hz : ∀ i ∈ I, ∀ l ∈ L, l ≠ l0 → g i l = [] := by
        intro i hi l hl hne
        exact Classical.byContradiction fun hg => hrow ⟨i, hi, l, hl, hg, hne⟩
      rw [flatMap_single L _ l0 hL (fun l hl hne => by
        rw [List.flatMap_eq_nil_iff]; exact fun i hi => hz i hi l hl hne) hl0]
      exact flatMap_congr' fun i hi => flatMap_single L (g i) l0 hL (fun l hl hne => hz i hi l hl hne) hl0
  · have hz : ∀ i ∈ I, ∀ l ∈ L, g i l = [] := by
      intro i hi l hl
      exact Classical.byContradiction fun hg => hex ⟨i, hi, l, hl, hg⟩
    rw [List.flatMap_eq_nil_iff.mpr (fun i hi => List.flatMap_eq_nil_iff.mpr (fun l hl => hz i hi l hl)),
      List.flatMap_eq_nil_iff.mpr (fun l hl => List.flatMap_eq_nil_iff.mpr (fun i hi => hz i hi l hl))]

/-! ## Shared memory as two-byte writes -/

/-- A two-byte write `(o, v)`: byte `o` the low byte of `v`, byte `o + 1` the next. -/
def applyF (f : Nat → UInt8) (w : Nat × UInt64) : Nat → UInt8 := fun x =>
  if x = w.1 then w.2.toUInt8 else if x = w.1 + 1 then (w.2 >>> 8).toUInt8 else f x

/-- What a write puts at byte `x`, if anything. -/
def byteAt (x : Nat) (w : Nat × UInt64) : Option UInt8 :=
  if x = w.1 then some w.2.toUInt8 else if x = w.1 + 1 then some (w.2 >>> 8).toUInt8 else none

/-- **A byte after writes is the last write to it.** -/
theorem applyF_point (x : Nat) : ∀ (W : List (Nat × UInt64)) (f : Nat → UInt8),
    (W.foldl applyF f) x = ((W.filterMap (byteAt x)).getLast?).getD (f x) := by
  intro W
  induction W with
  | nil => intro f; rfl
  | cons w W ih =>
    intro f
    rw [List.foldl_cons, ih, List.filterMap_cons]
    cases hb : byteAt x w with
    | none =>
      simp only [byteAt] at hb
      split at hb
      · cases hb
      · split at hb
        · cases hb
        · rename_i h1 h2
          simp only [applyF, if_neg h1, if_neg h2]
    | some y =>
      simp only [List.getLast?_cons]
      have : applyF f w x = y := by
        simp only [byteAt] at hb
        simp only [applyF]
        split at hb
        · cases hb; rename_i h; rw [if_pos h]
        · split at hb
          · cases hb; rename_i h1 h2; rw [if_neg h1, if_pos h2]
          · cases hb
      rw [this]
      cases W.filterMap (byteAt x) <;> simp

/-- A store of two bytes leaves them as the write does. -/
theorem storeH_flat (st : ISt) (a v g : Lane → UInt64) :
    (st.storeH a v g).smem =
      ((List.finRange 32).flatMap fun l => if g l ≠ 0 then [((a l).toNat, v l)] else []).foldl applyF st.smem := by
  unfold ISt.storeH
  have key : ∀ (L : List Lane) (s : ISt), (L.foldl (fun s l => if g l ≠ 0 then { s with smem := fun x =>
        if x = (a l).toNat then (v l).toUInt8 else if x = (a l).toNat + 1 then ((v l) >>> 8).toUInt8 else s.smem x }
        else s) s).smem =
      (L.flatMap fun l => if g l ≠ 0 then [((a l).toNat, v l)] else []).foldl applyF s.smem := by
    intro L
    induction L with
    | nil => intro s; rfl
    | cons l L ih =>
      intro s
      simp only [List.foldl_cons, List.flatMap_cons, List.foldl_append]
      rw [ih]
      congr 1
      split <;> rfl
  exact key _ st

/-- A warp's write at a shared-memory event, by lane `l`. -/
def wrF (cta : Nat) (l : Lane) : ShEv → List (Nat × UInt64)
  | .st ws a v g => if g.eval ws.int cta l ≠ 0 then [((a.eval ws.int cta l).toNat, v.eval ws.int cta l)] else []
  | .bar _ => []

theorem shApply_flat (cta : Nat) (m : Nat → UInt8) (e : ShEv) :
    shApply cta m e = ((List.finRange 32).flatMap fun l => wrF cta l e).foldl applyF m := by
  cases e with
  | st ws a v g => exact storeH_flat _ _ _ _
  | bar ws =>
    show m = _
    rw [show ((List.finRange 32).flatMap fun l => wrF cta l (ShEv.bar ws)) = [] from
      List.flatMap_eq_nil_iff.mpr (fun _ _ => rfl)]; rfl

theorem barriers_foldl (cta : Nat) (m : Nat → UInt8) (E : List ShEv) :
    E.foldl (shApply cta) m = (E.flatMap fun e => (List.finRange 32).flatMap fun l => wrF cta l e).foldl applyF m := by
  induction E generalizing m with
  | nil => rfl
  | cons e E ih => rw [List.foldl_cons, ih, shApply_flat, List.flatMap_cons, List.foldl_append]

/-! ## Epochs and a thread's tagged stores -/

/-- Whether an event is a barrier. -/
def ShEv.isBar : ShEv → Bool
  | .bar _ => true
  | .st .. => false

/-- The stores of epoch `b` (after `b` barriers), counting from `b0`. -/
def epochE : Nat → Nat → List ShEv → List ShEv
  | _, _, [] => []
  | b0, b, .bar _ :: r => epochE (b0 + 1) b r
  | b0, b, e :: r => (if b0 = b then [e] else []) ++ epochE b0 b r

/-- Lane `l`'s shared stores, as a thread records them: tagged with the
    barriers before them, from `b0`. -/
def tagE (cta : Nat) (sm : UInt64) (l : Lane) : Nat → List ShEv → List (Nat × UInt64 × Nat × UInt64)
  | _, [] => []
  | b, .bar _ :: r => tagE cta sm l (b + 1) r
  | b, .st ws a v g :: r =>
      (if g.eval ws.int cta l ≠ 0 then [(b, sm + a.eval ws.int cta l, 2, v.eval ws.int cta l &&& 0xffff)] else []) ++
        tagE cta sm l b r

/-- Lane `l`'s write at an event, as a thread's store. -/
def wrT (cta : Nat) (sm : UInt64) (l : Lane) (b : Nat) : ShEv → List (Nat × UInt64 × Nat × UInt64)
  | .st ws a v g => if g.eval ws.int cta l ≠ 0 then [(b, sm + a.eval ws.int cta l, 2, v.eval ws.int cta l &&& 0xffff)]
      else []
  | .bar _ => []

theorem epochE_lt : ∀ (E : List ShEv) (b0 b : Nat), b < b0 → epochE b0 b E = []
  | [], _, _, _ => rfl
  | .bar _ :: r, b0, b, h => epochE_lt r (b0 + 1) b (by omega)
  | .st .. :: r, b0, b, h => by simp only [epochE, if_neg (show b0 ≠ b by omega), List.nil_append]; exact epochE_lt r b0 b h

theorem tagE_filter (cta : Nat) (sm : UInt64) (l : Lane) (b : Nat) : ∀ (E : List ShEv) (b0 : Nat),
    (tagE cta sm l b0 E).filter (·.1 = b) = (epochE b0 b E).flatMap (wrT cta sm l b)
  | [], _ => rfl
  | .bar _ :: r, b0 => tagE_filter cta sm l b r (b0 + 1)
  | .st ws a v g :: r, b0 => by
      simp only [tagE, epochE, List.filter_append, List.flatMap_append, tagE_filter cta sm l b r b0]
      congr 1
      by_cases hb : b0 = b
      · subst hb; by_cases hg : g.eval ws.int cta l ≠ 0 <;> simp [wrT, hg]
      · rw [if_neg hb]; by_cases hg : g.eval ws.int cta l ≠ 0 <;> simp [hg, hb]

theorem epochE_mem {e : ShEv} : ∀ {E : List ShEv} {b0 b : Nat}, e ∈ epochE b0 b E → e ∈ E
  | [], _, _, h => by cases h
  | .bar _ :: r, b0, b, h => List.mem_cons_of_mem _ (epochE_mem h)
  | .st .. :: r, b0, b, h => by
      simp only [epochE, List.mem_append] at h
      rcases h with h | h
      · split at h
        · simp at h; subst h; exact List.mem_cons_self
        · cases h
      · exact List.mem_cons_of_mem _ (epochE_mem h)

/-- The number of barriers among events. -/
def bars (E : List ShEv) : Nat := E.countP ShEv.isBar

theorem flatMap_epochs (h : ShEv → List (Nat × UInt64)) (hb : ∀ ws, h (.bar ws) = []) :
    ∀ (E : List ShEv) (b0 : Nat), E.flatMap h = (List.range' b0 (bars E + 1)).flatMap fun b => (epochE b0 b E).flatMap h
  | [], b0 => by simp [bars, epochE]
  | .bar ws :: r, b0 => by
      rw [List.flatMap_cons, hb, List.nil_append, flatMap_epochs h hb r (b0 + 1)]
      simp only [bars, List.countP_cons, ShEv.isBar, if_true]
      rw [List.range'_succ, List.flatMap_cons]
      show _ = (epochE (b0 + 1) b0 r).flatMap h ++ _
      rw [epochE_lt r (b0 + 1) b0 (by omega)]
      rfl
  | .st ws a v g :: r, b0 => by
      rw [List.flatMap_cons, flatMap_epochs h hb r b0]
      simp only [bars, List.countP_cons, ShEv.isBar, Bool.false_eq_true, if_false, Nat.add_zero]
      rw [List.range'_succ, List.flatMap_cons, List.flatMap_cons]
      simp only [epochE, if_true, List.flatMap_append, List.singleton_append, List.flatMap_cons, List.flatMap_nil,
        List.append_nil, List.append_assoc]
      congr 2
      exact flatMap_congr' fun b hb' => by
        have : b0 ≠ b := by simp [List.mem_range'] at hb'; omega
        simp [this]

/-! ## A block's snapshots of shared memory -/

/-- Shared memory after the first `b` phases of a block's stores. -/
def snapMem (all : List (Nat × UInt64 × Nat × UInt64)) (smem0 : ByteArray) : Nat → Mem
  | 0 => { arena := smem0, data := .empty, out := .empty }
  | b + 1 => phase all b (snapMem all smem0 b)

theorem snapshots_fold (all : List (Nat × UInt64 × Nat × UInt64)) (smem0 : ByteArray) : ∀ m,
    ((List.range m).foldl (fun (acc : Mem × Array ByteArray) b =>
        (phase all b acc.1, acc.2.push (phase all b acc.1).arena))
      (({ arena := smem0, data := .empty, out := .empty } : Mem), #[])) =
      (snapMem all smem0 m, ((List.range m).map fun b => (snapMem all smem0 (b + 1)).arena).toArray)
  | 0 => rfl
  | m + 1 => by
      rw [List.range_succ, List.foldl_append, snapshots_fold all smem0 m]
      simp [snapMem, List.range_succ]

/-- **The `b`-th snapshot is shared memory after phases `0 … b`**, where some
    thread passed that many barriers. -/
theorem snapshots_get {n : Nat} {smem0 : ByteArray} {calls : Nat → List (Intr × List V)}
    {stores : Nat → List (Nat × UInt64 × Nat × UInt64)} {b t : Nat} (ht : t < n)
    (hb : b < ((calls t).filter (·.1 == .barrier)).length) :
    (snapshots n smem0 calls stores)[b]? = some (snapMem ((List.range n).flatMap stores) smem0 (b + 1)).arena := by
  unfold snapshots
  simp only
  have hnb : ((calls t).filter (·.1 == .barrier)).length ≤
      (List.range n).foldl (fun m t => max m ((calls t).filter (·.1 == .barrier)).length) 0 := by
    have key : ∀ (L : List Nat) (m0 : Nat), t ∈ L ∨ ((calls t).filter (·.1 == .barrier)).length ≤ m0 →
        ((calls t).filter (·.1 == .barrier)).length ≤
          L.foldl (fun m t => max m ((calls t).filter (·.1 == .barrier)).length) m0 := by
      intro L
      induction L with
      | nil => intro m0 h; simpa using h
      | cons x L ih =>
          intro m0 h
          rw [List.foldl_cons]
          apply ih
          rcases h with h | h
          · rcases List.mem_cons.mp h with rfl | h
            · right; exact Nat.le_max_right _ _
            · left; exact h
          · right; exact Nat.le_trans h (Nat.le_max_left _ _)
    exact key _ 0 (.inl (List.mem_range.mpr ht))
  rw [snapshots_fold]
  simp only [List.getElem?_toArray, List.getElem?_map]
  rw [List.getElem?_range (by omega)]
  rfl

/-- A phase applies the stores tagged with it, in order. -/
theorem phase_filter (all : List (Nat × UInt64 × Nat × UInt64)) (b : Nat) (cur : Mem) :
    phase all b cur = (all.filter (·.1 = b)).foldl
      (fun c x => (c.store x.2.1 x.2.2.1 x.2.2.2).getD c) cur := by
  unfold phase
  induction all generalizing cur with
  | nil => rfl
  | cons x xs ih =>
      rw [List.foldl_cons, List.filter_cons]
      by_cases hx : x.1 = b
      · rw [if_pos hx, ih]; simp [hx]
      · rw [if_neg hx, ih]; simp [hx]

theorem store_arena2 (m : Mem) {s : Nat} (hs : s < 2 ^ 36) (hsz : s + 2 ≤ m.arena.size) (v : UInt64) :
    ∃ m', m.store (addrOf .arena 0 + UInt64.ofNat s) 2 v = some m' ∧
      m'.arena.data = (m.arena.data.setIfInBounds s ((v &&& 0xff).toUInt8)).setIfInBounds (s + 1)
        (((v >>> 8) &&& 0xff).toUInt8) ∧ m'.arena.size = m.arena.size := by
  have hd : decodeAddr (addrOf .arena 0 + UInt64.ofNat s) = some (.arena, s) := by
    rw [show addrOf .arena 0 + UInt64.ofNat s = regionBase .arena + UInt64.ofNat s by simp [addrOf]]
    exact decodeAddr_base _ _ (by rw [regionSpan_toNat]; exact hs)
  refine ⟨m.setRegion .arena ((List.range 2).foldl (fun (b : ByteArray) i =>
    b.set! (s + i) (((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)) m.arena), ?_, ?_, ?_⟩
  · unfold Mem.store
    rw [hd]
    simp only [bind, Option.bind, Mem.region, Mem.reachable]
    rw [if_neg (by simp; omega)]
  · simp only [Mem.setRegion]
    rw [show List.range 2 = [0, 1] from rfl]
    simp only [List.foldl_cons, List.foldl_nil]
    rw [show (8 : UInt64) * UInt64.ofNat 0 = 0 by decide, show (8 : UInt64) * UInt64.ofNat 1 = 8 by decide,
      UInt64.shiftRight_zero, Nat.add_zero]
    generalize m.arena = A
    cases A
    simp only [ByteArray.set!, Array.set!_eq_setIfInBounds]
  · simp only [Mem.setRegion]
    rw [show List.range 2 = [0, 1] from rfl]
    simp only [List.foldl_cons, List.foldl_nil]
    generalize m.arena = A
    cases A
    simp [ByteArray.set!, ByteArray.size, Array.set!_eq_setIfInBounds]


theorem bytesOf_ext {S : Nat} {f g : Nat → UInt8} (h : ∀ x, x < S → f x = g x) : bytesOf S f = bytesOf S g := by
  unfold bytesOf
  congr 1
  apply Array.ext (by simp)
  intro k h1 h2
  simp only [Array.getElem_map, Array.getElem_range]
  exact h k (by simpa using h1)

theorem byteArray_ext {a b : ByteArray} (h : a.data = b.data) : a = b := by
  cases a; cases b; simp_all

theorem u8_mask_lo (v : UInt64) : (v &&& 0xffff).toUInt8 = v.toUInt8 := by
  apply UInt8.toNat_inj.mp
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_and, show (0xffff : UInt64).toNat = 2 ^ 16 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod]
  exact Nat.mod_mod_of_dvd _ (by decide)

theorem u8_mask_hi (v : UInt64) : ((v &&& 0xffff) >>> 8).toUInt8 = (v >>> 8).toUInt8 := by
  apply UInt8.toNat_inj.mp
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_and, UInt64.toNat_shiftRight,
    show (0xffff : UInt64).toNat = 2 ^ 16 - 1 from rfl, show (8 : UInt64).toNat % 64 = 8 from rfl,
    Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow]
  omega

theorem applyF_mask (f : Nat → UInt8) (o : Nat) (v : UInt64) : applyF f (o, v &&& 0xffff) = applyF f (o, v) := by
  funext x; simp only [applyF, u8_mask_lo, u8_mask_hi]

/-- **A two-byte shared store writes the write's bytes.** -/
theorem store_bytesOf (m : Mem) {S o : Nat} (f : Nat → UInt8) (hS : S ≤ 2 ^ 36) (ho : o + 2 ≤ S)
    (ha : m.arena = bytesOf S f) (v : UInt64) :
    ∃ m', m.store (addrOf .arena 0 + UInt64.ofNat o) 2 v = some m' ∧ m'.arena = bytesOf S (applyF f (o, v)) := by
  obtain ⟨m', hm, hd, _⟩ := store_arena2 m (s := o) (by omega) (by rw [ha, bytesOf_size]; exact ho) v
  refine ⟨m', hm, byteArray_ext ?_⟩
  rw [hd, ha]
  simp only [bytesOf]
  apply Array.ext_getElem?
  intro k
  simp only [Array.getElem?_setIfInBounds, Array.getElem?_map, Array.getElem?_range, Array.size_setIfInBounds,
    Array.size_map, Array.size_range, applyF, and255_toUInt8]
  by_cases hk : k < S
  · simp only [hk, if_true, Option.map_some]
    by_cases e0 : k = o
    · subst e0; simp [applyF, hk]
    · by_cases e1 : k = o + 1
      · subst e1; simp [applyF, hk]
      · have e0' : o ≠ k := fun h => e0 h.symm
        have e1' : o + 1 ≠ k := fun h => e1 h.symm
        simp [applyF, e0, e1, e0', e1', hk]
  · simp only [hk, if_false, Option.map_none]
    split <;> split <;> simp_all <;> omega

/-- **Two-byte shared stores write their writes.** -/
theorem stores_bytesOf {S : Nat} (hS : S ≤ 2 ^ 36) (b : Nat) : ∀ (W : List (Nat × UInt64)) (c : Mem) (f : Nat → UInt8),
    c.arena = bytesOf S f → (∀ w ∈ W, w.1 + 2 ≤ S) →
    ((W.map fun w => (b, addrOf .arena 0 + UInt64.ofNat w.1, 2, w.2 &&& 0xffff)).foldl
      (fun c x => (c.store x.2.1 x.2.2.1 x.2.2.2).getD c) c).arena = bytesOf S (W.foldl applyF f)
  | [], c, f, h, _ => h
  | w :: W, c, f, h, hW => by
      obtain ⟨m', hm, ha⟩ := store_bytesOf c f hS (hW w List.mem_cons_self) h (w.2 &&& 0xffff)
      simp only [List.map_cons, List.foldl_cons, hm, Option.getD_some]
      rw [applyF_mask] at ha
      exact stores_bytesOf hS b W m' _ ha (fun x hx => hW x (List.mem_cons_of_mem _ hx))

theorem wrT_map (cta : Nat) (l : Lane) (b : Nat) (e : ShEv) :
    wrT cta (addrOf .arena 0) l b e = (wrF cta l e).map fun w => (b, addrOf .arena 0 + UInt64.ofNat w.1, 2, w.2 &&& 0xffff) := by
  cases e with
  | st ws a v g => simp only [wrT, wrF]; split <;> simp [UInt64.ofNat_toNat]
  | bar ws => rfl

/-- Lane `t mod 32`. -/
def laneOf (t : Nat) : Lane := ⟨t % 32, Nat.mod_lt _ (by decide)⟩

theorem range_lanes {β : Type} (G : Lane → List β) :
    (List.range 32).flatMap (fun t => G (laneOf t)) = (List.finRange 32).flatMap G := by
  have : (List.range 32).map laneOf = List.finRange 32 := by decide
  rw [← this, List.flatMap_map]

/-- **A block's shared memory after its first phases**: its threads' stores,
    epoch by epoch, thread by thread. -/
theorem snapMem_fun (cta S : Nat) (hS : S ≤ 2 ^ 36) (EV : List ShEv) (m0 : Nat → UInt8)
    (hin : ∀ e ∈ EV, ∀ l, ∀ w ∈ wrF cta l e, w.1 + 2 ≤ S) :
    ∀ n, (snapMem ((List.range 32).flatMap fun t => tagE cta (addrOf .arena 0) (laneOf t) 0 EV) (bytesOf S m0) n).arena =
      bytesOf S (((List.range n).flatMap fun b => (List.finRange 32).flatMap fun l => (epochE 0 b EV).flatMap (wrF cta l)).foldl
        applyF m0)
  | 0 => rfl
  | n + 1 => by
      let tup : Nat × UInt64 → Nat × UInt64 × Nat × UInt64 :=
        fun w => (n, addrOf .arena 0 + UInt64.ofNat w.1, 2, w.2 &&& 0xffff)
      have hall : ((List.range 32).flatMap fun t => tagE cta (addrOf .arena 0) (laneOf t) 0 EV).filter (·.1 = n) =
          ((List.finRange 32).flatMap fun l => (epochE 0 n EV).flatMap (wrF cta l)).map tup := by
        rw [List.filter_flatMap, List.map_flatMap, ← range_lanes]
        refine flatMap_congr' fun t _ => ?_
        rw [tagE_filter, List.map_flatMap]
        exact flatMap_congr' fun e _ => wrT_map cta (laneOf t) n e
      simp only [snapMem]
      rw [phase_filter, hall, stores_bytesOf hS n _ _ _ (snapMem_fun cta S hS EV m0 hin n) ?_, List.range_succ,
        List.flatMap_append, List.foldl_append]
      · simp
      · intro w hw
        simp only [List.mem_flatMap] at hw
        obtain ⟨l, _, e, he, hw⟩ := hw
        exact hin e (epochE_mem he) l w hw

/-! ## The barrier law -/

theorem epochE_append (b : Nat) : ∀ (A B : List ShEv) (b0 : Nat),
    epochE b0 b (A ++ B) = epochE b0 b A ++ epochE (b0 + bars A) b B
  | [], B, b0 => by simp [epochE, bars]
  | .bar ws :: A, B, b0 => by
      simp only [List.cons_append, epochE, epochE_append b A B (b0 + 1), bars, List.countP_cons, ShEv.isBar, if_true]
      congr 2; omega
  | .st ws a v g :: A, B, b0 => by
      simp only [List.cons_append, epochE, epochE_append b A B b0, bars, List.countP_cons, ShEv.isBar,
        Bool.false_eq_true, if_false, Nat.add_zero, List.append_assoc]

/-- **The warp's shared memory at a barrier**: its epochs' writes before it,
    statement by statement, lane by lane. -/
theorem warp_bar (cta : Nat) (m0 : Nat → UInt8) (EV : List ShEv) {p n : Nat} {ws' : WSt F32}
    (hp : EV[p]? = some (.bar ws')) (hn : bars (EV.take p) = n) :
    (EV.take p).foldl (shApply cta) m0 = (((List.range (n + 1)).flatMap fun b =>
      (epochE 0 b EV).flatMap fun e => (List.finRange 32).flatMap fun l => wrF cta l e)).foldl applyF m0 := by
  rw [barriers_foldl, flatMap_epochs _ (fun ws => by simp [wrF]) (EV.take p) 0, hn, ← List.range_eq_range']
  congr 1
  refine flatMap_congr' fun b hb => ?_
  have hb' : b < n + 1 := List.mem_range.mp hb
  have hd : EV.drop p = .bar ws' :: EV.drop (p + 1) := by
    rw [List.drop_eq_getElem_cons (List.getElem?_eq_some_iff.mp hp).1]
    congr 1
    exact (List.getElem?_eq_some_iff.mp hp).2
  conv => rhs; rw [← List.take_append_drop p EV, epochE_append, hd]
  simp only [epochE, Nat.zero_add, hn, epochE_lt _ (n + 1) b (by omega), List.append_nil]

theorem flatMap_index {α β : Type} (F : α → List β) : ∀ (E : List α),
    E.flatMap F = (List.range E.length).flatMap fun i => (E[i]?.map F).getD []
  | [] => rfl
  | e :: E => by
      rw [List.flatMap_cons, List.length_cons, List.range_succ_eq_map, List.flatMap_cons, flatMap_index F E,
        List.flatMap_map]
      rfl

theorem lanes_flatMap {β : Type} (G : Lane → List β) :
    (List.finRange 32).flatMap G = (List.range 32).flatMap fun t => G (laneOf t) := (range_lanes G).symm

/-- **One epoch's writes to a byte, in the two orders, agree** where no byte
    is written by two lanes at two different events. -/
theorem epoch_byte (cta x : Nat) (E : List ShEv)
    (hc : ∀ (i j : Nat) (l l' : Lane) (e e' : ShEv), E[i]? = some e → E[j]? = some e' → (wrF cta l e).filterMap (byteAt x) ≠ [] →
      (wrF cta l' e').filterMap (byteAt x) ≠ [] → i = j ∨ l = l') :
    (E.flatMap fun e => (List.finRange 32).flatMap fun l => wrF cta l e).filterMap (byteAt x) =
      ((List.finRange 32).flatMap fun l => E.flatMap (wrF cta l)).filterMap (byteAt x) := by
  let g (i t : Nat) : List UInt8 := ((E[i]?.map (wrF cta (laneOf t))).getD []).filterMap (byteAt x)
  have h1 : (E.flatMap fun e => (List.finRange 32).flatMap fun l => wrF cta l e).filterMap (byteAt x) =
      (List.range E.length).flatMap fun i => (List.range 32).flatMap (g i) := by
    rw [flatMap_index, List.filterMap_flatMap]
    refine flatMap_congr' fun i _ => ?_
    simp only [g]
    cases h : E[i]? with
    | none => simp
    | some e => simp only [Option.map_some, Option.getD_some, lanes_flatMap, List.filterMap_flatMap]
  have h2 : ((List.finRange 32).flatMap fun l => E.flatMap (wrF cta l)).filterMap (byteAt x) =
      (List.range 32).flatMap fun t => (List.range E.length).flatMap fun i => g i t := by
    rw [lanes_flatMap, List.filterMap_flatMap]
    refine flatMap_congr' fun t _ => ?_
    rw [flatMap_index, List.filterMap_flatMap]
  rw [h1, h2]
  refine flatMap_comm _ _ List.nodup_range List.nodup_range g fun i hi j hj t ht t' ht' hg hg' => ?_
  have hi' := List.mem_range.mp hi
  have hj' := List.mem_range.mp hj
  obtain ⟨e, he⟩ : ∃ e, E[i]? = some e := ⟨_, List.getElem?_eq_getElem hi'⟩
  obtain ⟨e', he'⟩ : ∃ e', E[j]? = some e' := ⟨_, List.getElem?_eq_getElem hj'⟩
  simp only [g, he, he', Option.map_some, Option.getD_some] at hg hg'
  rcases hc i j _ _ e e' he he' hg hg' with h | h
  · exact Or.inl h
  · right
    have := congrArg Fin.val h
    simp only [laneOf] at this
    have := List.mem_range.mp ht; have := List.mem_range.mp ht'
    omega

/-- **No byte of an epoch is written by two lanes at two different events.** -/
def ShOk (cta : Nat) (EV : List ShEv) : Prop :=
  ∀ (b x i j : Nat) (l l' : Lane) (e e' : ShEv), (epochE 0 b EV)[i]? = some e → (epochE 0 b EV)[j]? = some e' →
    (wrF cta l e).filterMap (byteAt x) ≠ [] → (wrF cta l' e').filterMap (byteAt x) ≠ [] → i = j ∨ l = l'

/-- **The barrier law**: a block's shared memory after the phases before a
    barrier is the warp's there, where its writes lie inside shared memory and
    no byte of an epoch is written by two lanes at two different events. -/
theorem snap_law (cta S : Nat) (hS : S ≤ 2 ^ 36) (EV : List ShEv) (m0 : Nat → UInt8)
    (hin : ∀ e ∈ EV, ∀ l, ∀ w ∈ wrF cta l e, w.1 + 2 ≤ S) (hok : ShOk cta EV) {p n : Nat} {ws' : WSt F32}
    (hp : EV[p]? = some (.bar ws')) (hn : bars (EV.take p) = n) :
    (snapMem ((List.range 32).flatMap fun t => tagE cta (addrOf .arena 0) (laneOf t) 0 EV) (bytesOf S m0) (n + 1)).arena =
      bytesOf S ((EV.take p).foldl (shApply cta) m0) := by
  rw [snapMem_fun cta S hS EV m0 hin, warp_bar cta m0 EV hp hn]
  refine bytesOf_ext fun x _ => ?_
  rw [applyF_point, applyF_point, List.filterMap_flatMap, List.filterMap_flatMap]
  congr 2
  exact flatMap_congr' fun b _ => (epoch_byte cta x _ (hok b x)).symm

/-! ## A thread's shared stores -/

section
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat)

/-- The leaves a run reaches, at their trips and states. -/
def allLeaf : Nat → EWStmt → WSt F32 → List (Nat × EWStmt × WSt F32) := fun i s ws => [(i, s, ws)]

theorem foldl_keep {γ : Type} (f : Nat → WSt F32 → List γ) (nxt : Nat → WSt F32 → WSt F32) :
    ∀ (L : List Nat) (acc : List γ) (cur : WSt F32) (x : γ), x ∈ acc →
      x ∈ (L.foldl (fun (p : List γ × WSt F32) j => (p.1 ++ f j p.2, nxt j p.2)) (acc, cur)).1
  | [], _, _, _, h => h
  | j :: L, acc, cur, x, h => by
      simp only [List.foldl_cons]
      exact foldl_keep f nxt L _ _ x (List.mem_append_left _ h)

theorem while_keep {γ : Type} (go : WSt F32 → Prop) [DecidablePred go] (f : WSt F32 → List γ)
    (nxt : WSt F32 → WSt F32) :
    ∀ (L : List Nat) (acc : List γ) (cur : WSt F32) (x : γ), x ∈ acc →
      x ∈ (L.foldl (fun (p : List γ × WSt F32) _ => if go p.2 then (p.1 ++ f p.2, nxt p.2) else p) (acc, cur)).1
  | [], _, _, _, h => h
  | j :: L, acc, cur, x, h => by
      simp only [List.foldl_cons]
      split
      · exact while_keep go f nxt L _ _ x (List.mem_append_left _ h)
      · exact while_keep go f nxt L _ _ x h

theorem trace_rel_foldl {α β : Type} (leaf₁ : Nat → EWStmt → WSt F32 → List α) (leaf₂ : Nat → EWStmt → WSt F32 → List β)
    (R : List α → List β → Prop) (happ : ∀ a b a' b', R a b → R a' b' → R (a ++ a') (b ++ b')) (b : EWStmt)
    (hb : ∀ j ws, (∀ x ∈ trace hw cta im allLeaf j b ws, R (leaf₁ x.1 x.2.1 x.2.2) (leaf₂ x.1 x.2.1 x.2.2)) →
      R (trace hw cta im leaf₁ j b ws) (trace hw cta im leaf₂ j b ws)) :
    ∀ (L : List Nat) (a1 : List α) (a2 : List β) (a3 : List (Nat × EWStmt × WSt F32)) (cur : WSt F32),
      R a1 a2 →
      (∀ x ∈ (L.foldl (fun (p : List (Nat × EWStmt × WSt F32) × WSt F32) j =>
          (p.1 ++ trace hw cta im allLeaf j b p.2, run hw cta im b p.2 j)) (a3, cur)).1,
        R (leaf₁ x.1 x.2.1 x.2.2) (leaf₂ x.1 x.2.1 x.2.2)) →
      R (L.foldl (fun (p : List α × WSt F32) j => (p.1 ++ trace hw cta im leaf₁ j b p.2, run hw cta im b p.2 j))
          (a1, cur)).1
        (L.foldl (fun (p : List β × WSt F32) j => (p.1 ++ trace hw cta im leaf₂ j b p.2, run hw cta im b p.2 j))
          (a2, cur)).1
  | [], _, _, _, _, h, _ => h
  | j :: L, a1, a2, a3, cur, h, hx => by
      simp only [List.foldl_cons] at hx ⊢
      exact trace_rel_foldl leaf₁ leaf₂ R happ b hb L _ _ (a3 ++ trace hw cta im allLeaf j b cur) _
        (happ _ _ _ _ h (hb j cur fun x hx' => hx x (foldl_keep (fun j ws => trace hw cta im allLeaf j b ws)
          (fun j ws => run hw cta im b ws j) L _ _ x (List.mem_append_right _ hx')))) hx

theorem trace_rel_while {α β : Type} (leaf₁ : Nat → EWStmt → WSt F32 → List α) (leaf₂ : Nat → EWStmt → WSt F32 → List β)
    (R : List α → List β → Prop) (happ : ∀ a b a' b', R a b → R a' b' → R (a ++ a') (b ++ b')) (i : Nat) (c : IExp)
    (cs : List Nat) (b : EWStmt) (ws0 : WSt F32)
    (hb : ∀ ws, (∀ x ∈ trace hw cta im allLeaf i b ws, R (leaf₁ x.1 x.2.1 x.2.2) (leaf₂ x.1 x.2.1 x.2.2)) →
      R (trace hw cta im leaf₁ i b ws) (trace hw cta im leaf₂ i b ws)) :
    ∀ (L : List Nat) (a1 : List α) (a2 : List β) (a3 : List (Nat × EWStmt × WSt F32)) (cur : WSt F32),
      R a1 a2 →
      (∀ x ∈ (L.foldl (fun (p : List (Nat × EWStmt × WSt F32) × WSt F32) _ =>
          if c.eval p.2.int cta lane0 ≠ 0 then
            (p.1 ++ trace hw cta im allLeaf i b p.2,
              (run hw cta im b p.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b p.2 i).int))
          else p) (a3, cur)).1, R (leaf₁ x.1 x.2.1 x.2.2) (leaf₂ x.1 x.2.1 x.2.2)) →
      R (L.foldl (fun (p : List α × WSt F32) _ =>
          if c.eval p.2.int cta lane0 ≠ 0 then
            (p.1 ++ trace hw cta im leaf₁ i b p.2,
              (run hw cta im b p.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b p.2 i).int))
          else p) (a1, cur)).1
        (L.foldl (fun (p : List β × WSt F32) _ =>
          if c.eval p.2.int cta lane0 ≠ 0 then
            (p.1 ++ trace hw cta im leaf₂ i b p.2,
              (run hw cta im b p.2 i).withInt (ISt.keep cs ws0.int (run hw cta im b p.2 i).int))
          else p) (a2, cur)).1
  | [], _, _, _, _, h, _ => h
  | j :: L, a1, a2, a3, cur, h, hx => by
      simp only [List.foldl_cons] at hx ⊢
      by_cases hc : c.eval cur.int cta lane0 ≠ 0
      · simp only [if_pos hc] at hx ⊢
        exact trace_rel_while leaf₁ leaf₂ R happ i c cs b ws0 hb L _ _ _ _
          (happ _ _ _ _ h (hb cur fun x hx' => hx x (while_keep (fun s => c.eval s.int cta lane0 ≠ 0)
            (fun ws => trace hw cta im allLeaf i b ws)
            (fun ws => (run hw cta im b ws i).withInt (ISt.keep cs ws0.int (run hw cta im b ws i).int)) L _ _ x
            (List.mem_append_right _ hx')))) hx
      · simp only [if_neg hc] at hx ⊢
        exact trace_rel_while leaf₁ leaf₂ R happ i c cs b ws0 hb L _ _ _ _ h hx

/-- **Two traces whose leaves are related are related**, by a relation
    closed under appending. -/
theorem trace_rel {α β : Type} (leaf₁ : Nat → EWStmt → WSt F32 → List α) (leaf₂ : Nat → EWStmt → WSt F32 → List β)
    (R : List α → List β → Prop) (hnil : R [] []) (happ : ∀ a b a' b', R a b → R a' b' → R (a ++ a') (b ++ b')) :
    ∀ (s : EWStmt) (i : Nat) (ws : WSt F32),
      (∀ x ∈ trace hw cta im allLeaf i s ws, R (leaf₁ x.1 x.2.1 x.2.2) (leaf₂ x.1 x.2.1 x.2.2)) →
      R (trace hw cta im leaf₁ i s ws) (trace hw cta im leaf₂ i s ws) := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro i ws h
      simp only [trace] at h ⊢
      exact happ _ _ _ _ (iha i ws fun x hx => h x (List.mem_append_left _ hx))
        (ihb i _ fun x hx => h x (List.mem_append_right _ hx))
  | iif c a b iha ihb =>
      intro i ws h
      simp only [trace] at h ⊢
      split at h
      · rename_i hc; rw [if_pos hc, if_pos hc]; exact iha i ws h
      · rename_i hc; rw [if_neg hc, if_neg hc]; exact ihb i ws h
  | forN n b ih =>
      intro i ws h
      simp only [trace] at h ⊢
      exact trace_rel_foldl hw cta im leaf₁ leaf₂ R happ b (fun j ws => ih j ws) _ [] [] [] ws hnil h
  | forM bu a b ih =>
      intro i ws h
      simp only [trace] at h ⊢
      exact trace_rel_foldl hw cta im leaf₁ leaf₂ R happ b (fun j ws => ih j ws) _ [] [] [] ws hnil h
  | iwhile n c cs b ih =>
      intro i ws h
      simp only [trace] at h ⊢
      exact trace_rel_while hw cta im leaf₁ leaf₂ R happ i c cs b ws (fun ws => ih i ws) _ [] [] [] ws hnil h
  | locals rs b ih => intro i ws h; exact ih i ws h
  | _ => intro i ws h; exact h _ List.mem_cons_self

/-- A thread's stores and barriers, tagged as its shared stores record them,
    from `b0` barriers: those into shared memory. -/
def tagFrom : Nat → List (Option (UInt64 × Nat × UInt64)) → List (Nat × UInt64 × Nat × UInt64)
  | _, [] => []
  | b, none :: r => tagFrom (b + 1) r
  | b, some (a, n, v) :: r => (match decodeAddr a with
      | some (.arena, _) => [(b, a, n, v)]
      | _ => []) ++ tagFrom b r

theorem tagFrom_append : ∀ (A B : List (Option (UInt64 × Nat × UInt64))) (b : Nat),
    tagFrom b (A ++ B) = tagFrom b A ++ tagFrom (b + A.countP Option.isNone) B
  | [], B, b => by simp [tagFrom]
  | none :: A, B, b => by
      simp only [List.cons_append, tagFrom, tagFrom_append A B (b + 1), List.countP_cons, Option.isNone_none, if_true]
      congr 2; omega
  | some x :: A, B, b => by
      obtain ⟨a, n, v⟩ := x
      simp only [List.cons_append, tagFrom, tagFrom_append A B b, List.countP_cons, Option.isNone_some,
        Bool.false_eq_true, if_false, Nat.add_zero, List.append_assoc]

theorem tagE_append (cta : Nat) (sm : UInt64) (l : Lane) : ∀ (A B : List ShEv) (b : Nat),
    tagE cta sm l b (A ++ B) = tagE cta sm l b A ++ tagE cta sm l (b + bars A) B
  | [], B, b => by simp [tagE, bars]
  | .bar ws :: A, B, b => by
      simp only [List.cons_append, tagE, tagE_append cta sm l A B (b + 1), bars, List.countP_cons, ShEv.isBar, if_true]
      congr 2; omega
  | .st ws a v g :: A, B, b => by
      simp only [List.cons_append, tagE, tagE_append cta sm l A B b, bars, List.countP_cons, ShEv.isBar,
        Bool.false_eq_true, if_false, Nat.add_zero, List.append_assoc]

/-- A lane's stores and barriers tag as its shared-memory events do. -/
def TagRel (cta : Nat) (sm : UInt64) (l : Lane) (A : List (Option (UInt64 × Nat × UInt64))) (E : List ShEv) : Prop :=
  A.countP Option.isNone = bars E ∧ ∀ b, tagFrom b A = tagE cta sm l b E

theorem tagRel_append {cta : Nat} {sm : UInt64} {l : Lane} {A A' : List (Option (UInt64 × Nat × UInt64))}
    {E E' : List ShEv} (h : TagRel cta sm l A E) (h' : TagRel cta sm l A' E') : TagRel cta sm l (A ++ A') (E ++ E') :=
  ⟨by rw [List.countP_append, h.1, h'.1, bars, bars, bars, List.countP_append],
   fun b => by rw [tagFrom_append, tagE_append, h.2, h.1, h'.2]⟩

/-- **A lane's shared stores are its warp's events'** where every leaf the run
    reaches tags alike: a shared store inside shared memory, every other store
    outside it. -/
theorem lane_tags (cta : Nat) (im : Buf → Nat → Nat) (l : Lane) (ptr : Buf → UInt64) (s : EWStmt) (i : Nat)
    (ws : WSt F32)
    (h : ∀ x ∈ trace hw cta im allLeaf i s ws, TagRel cta (addrOf .arena 0) l
      (storeLeaf cta im l ptr (addrOf .arena 0) x.1 x.2.1 x.2.2) (shEvLeaf x.1 x.2.1 x.2.2)) :
    TagRel cta (addrOf .arena 0) l (trace hw cta im (storeLeaf cta im l ptr (addrOf .arena 0)) i s ws)
      (trace hw cta im shEvLeaf i s ws) :=
  trace_rel hw cta im _ _ (TagRel cta (addrOf .arena 0) l) ⟨rfl, fun _ => rfl⟩ (fun _ _ _ _ => tagRel_append) s i ws h

/-- The shared stores a world records are its stores into shared memory,
    tagged with the barriers before them. -/
theorem sharedStores_tag (w : World) (start : Nat) : sharedStores w start = tagFrom 0 (storesOf w start) := by
  unfold sharedStores storesOf
  have key : ∀ (L : List Obs) (b : Nat) (acc : List (Nat × UInt64 × Nat × UInt64)),
      (L.foldl shStep (b, acc)).2.reverse = acc.reverse ++ tagFrom b (L.filterMap fun
        | .store a n v => some (some (a, n, v))
        | .call (.intr .barrier) _ => some none
        | _ => none) := by
    intro L
    induction L with
    | nil => intro b acc; simp [tagFrom]
    | cons o L ih =>
      intro b acc
      rw [List.foldl_cons]
      cases o with
      | store a n v =>
        cases hd : decodeAddr a with
        | none => simp [shStep, tagFrom, hd, ih]
        | some p =>
          obtain ⟨r, o⟩ := p
          cases r <;> simp [shStep, tagFrom, hd, ih]
      | call c args =>
        cases c with
        | intr i =>
          cases i <;> simp only [shStep, List.filterMap_cons, tagFrom, ih] <;> rfl
        | _ => simp only [shStep, List.filterMap_cons, ih]
  rw [key]; rfl

/-! ## A block's answers -/

theorem filterMap_get {α β : Type} (f : α → Option β) : ∀ (L : List α) (k : Nat) (x : α) (y : β),
    L[k]? = some x → f x = some y → (L.filterMap f)[((L.take k).filterMap f).length]? = some y
  | [], _, _, _, h, _ => by cases h
  | a :: L, 0, x, y, h, hf => by
      simp at h; subst h; simp [hf]
  | a :: L, k + 1, x, y, h, hf => by
      simp only [List.getElem?_cons_succ] at h
      have ih := filterMap_get f L k x y h hf
      simp only [List.take_succ_cons, List.filterMap_cons]
      cases f a with
      | none => exact ih
      | some z => simpa using ih

theorem filterMap_pos {α β : Type} (f : α → Option β) : ∀ (L : List α) (b : Nat) (y : β),
    (L.filterMap f)[b]? = some y → ∃ p x, L[p]? = some x ∧ f x = some y ∧ ((L.take p).filterMap f).length = b
  | [], _, _, h => by simp at h
  | a :: L, b, y, h => by
      simp only [List.filterMap_cons] at h
      cases hfa : f a with
      | none =>
        rw [hfa] at h
        obtain ⟨p, x, h1, h2, h3⟩ := filterMap_pos f L b y h
        exact ⟨p + 1, x, by simpa using h1, h2, by simp [hfa, h3]⟩
      | some z =>
        rw [hfa] at h
        cases b with
        | zero => simp at h; subst h; exact ⟨0, a, rfl, hfa, rfl⟩
        | succ b =>
          obtain ⟨p, x, h1, h2, h3⟩ := filterMap_pos f L b y (by simpa using h)
          exact ⟨p + 1, x, by simpa using h1, h2, by simp [hfa, h3]⟩

/-- The state at a barrier leaf. -/
def barOf : Nat × EWStmt × WSt F32 → Option (WSt F32)
  | (_, .barrier, ws) => some ws
  | _ => none

/-- The state at a barrier event. -/
def barOfE : ShEv → Option (WSt F32)
  | .bar ws => some ws
  | .st .. => none

theorem bars_filterMap (E : List ShEv) : bars E = (E.filterMap barOfE).length := by
  induction E with
  | nil => rfl
  | cons e E ih => cases e <;> simp [bars, barOfE, ShEv.isBar, List.countP_cons, ← ih] <;> rfl

theorem bars_trace (s : EWStmt) (i : Nat) (ws : WSt F32) :
    (trace hw cta im jLeaf i s ws).filterMap barOf = (trace hw cta im shEvLeaf i s ws).filterMap barOfE :=
  trace_rel hw cta im _ _ (fun A B => A.filterMap barOf = B.filterMap barOfE) rfl
    (fun a b a' b' h h' => by simp only [List.filterMap_append, h, h']) s i ws
    (fun x _ => by obtain ⟨i, s, ws⟩ := x; cases s <;> rfl)

/-- **The laws give a one-warp block the warp's answers**: where each lane's
    collectives are the warp's (`callOne`) and its shared stores the warp's
    shared stores for it, tagged by the barriers before them, inside shared
    memory and no byte of an epoch written by two lanes at two events. -/
theorem block_laws (S : Nat) (hS : S ≤ 2 ^ 36) (s : EWStmt) (ws0 : WSt F32)
    (calls : Nat → List (Intr × List V)) (stores : Nat → List (Nat × UInt64 × Nat × UInt64))
    (hJ : ∀ e ∈ trace hw cta im jLeaf 0 s ws0, ∃ i ws, (∃ d p, e = (i, .ballot d p, ws)) ∨
      (∃ d v sr, e = (i, .shflIdx d v sr, ws)) ∨ (∃ d sr m, e = (i, .shflXor d sr m, ws) ∧ m < 2 ^ 31) ∨
      e = (i, .barrier, ws))
    (hc : ∀ t, t < 32 → calls t = (trace hw cta im jLeaf 0 s ws0).map (callOne cta (laneOf t)))
    (hst : ∀ t, t < 32 → stores t = tagE cta (addrOf .arena 0) (laneOf t) 0 (trace hw cta im shEvLeaf 0 s ws0))
    (hin : ∀ e ∈ trace hw cta im shEvLeaf 0 s ws0, ∀ l, ∀ w ∈ wrF cta l e, w.1 + 2 ≤ S)
    (hok : ShOk cta (trace hw cta im shEvLeaf 0 s ws0)) :
    ∀ t, t < 32 → answersOf calls (snapshots 32 (bytesOf S ws0.int.smem) calls stores) (bytesOf S ws0.int.smem) t =
      .ok ((trace hw cta im jLeaf 0 s ws0).map (ansOne cta S (laneOf t))) := by
  let J := trace hw cta im jLeaf 0 s ws0
  let EV := trace hw cta im shEvLeaf 0 s ws0
  have hc' : ∀ l : Lane, calls (32 * 0 + l.val) = J.map (callOne cta l) := fun l => by
    have h := hc l.val l.isLt
    rw [show laneOf l.val = l from Fin.ext (Nat.mod_eq_of_lt l.isLt)] at h
    simpa using h
  have hall : (List.range 32).flatMap stores = (List.range 32).flatMap fun t => tagE cta (addrOf .arena 0) (laneOf t) 0 EV :=
    flatMap_congr' fun t ht => hst t (List.mem_range.mp ht)
  intro t ht
  have := LaneLaws.warp_laws (snaps := snapshots 32 (bytesOf S ws0.int.smem) calls stores) (smem0 := bytesOf S ws0.int.smem)
    (S := S) (wi := 0) J hJ hc' ?_ (laneOf t)
  · rwa [show 32 * 0 + (laneOf t).val = t by simp [laneOf]; omega] at this
  intro l k i ws hk
  -- the barriers among a lane's collectives are the warp's
  have hcount : ∀ L : List (Nat × EWStmt × WSt F32), (∀ e ∈ L, e ∈ J) →
      (L.map (callOne cta l)).countP (·.1 == .barrier) = (L.filterMap barOf).length := by
    intro L hL
    induction L with
    | nil => rfl
    | cons e E ih =>
      rw [List.map_cons, List.countP_cons, ih (fun e' he' => hL e' (List.mem_cons_of_mem _ he')), List.filterMap_cons]
      obtain ⟨i', ws', h | h | h | h⟩ := hJ e (hL e List.mem_cons_self)
      · obtain ⟨d, p, rfl⟩ := h; rfl
      · obtain ⟨d, v, sr, rfl⟩ := h; rfl
      · obtain ⟨d, sr, m, rfl, _⟩ := h; rfl
      · subst h; simp [callOne, callLeaf, barOf]; rfl
  have hcnt : ((calls (32 * 0 + l.val)).take k).countP (·.1 == .barrier) = ((J.take k).filterMap barOf).length := by
    rw [hc' l, ← List.map_take]; exact hcount _ (fun e he => List.mem_of_mem_take he)
  rw [hcnt]
  generalize hb : ((J.take k).filterMap barOf).length = b
  have hJb : (J.filterMap barOf)[b]? = some ws := by rw [← hb]; exact filterMap_get barOf J k _ ws hk rfl
  have hlt : b < ((calls (32 * 0 + l.val)).filter (·.1 == .barrier)).length := by
    rw [← List.countP_eq_length_filter, hc' l, hcount J (fun e he => he)]
    exact (List.getElem?_eq_some_iff.mp hJb).1
  rw [bars_trace] at hJb
  obtain ⟨p, e, hp, he, hpb⟩ := filterMap_pos barOfE EV b ws hJb
  cases e with
  | st _ _ _ _ => cases he
  | bar ws' =>
    cases he
    have hn : bars (EV.take p) = b := by rw [bars_filterMap]; exact hpb
    rw [snapshots_get (t := 32 * 0 + l.val) (by have : l.val < 32 := l.isLt; omega) hlt, Option.getD_some, hall,
      snap_law cta S hS EV ws0.int.smem hin hok hp hn]
    exact congrArg (bytesOf S) (ev_smem hw cta im s 0 ws0 p _ hp).symm

end

/-! ## What every leaf a run reaches meets -/

section Leaves
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (S : Nat) (ptr : Buf → UInt64)
  (MF : Lane → Buf → Nat → Prop) (MB : Lane → UInt64 → Prop) (MI : Buf → Nat → Prop) (WS : Lane → UInt64 → Nat → Prop)

theorem forN_leaves (b : EWStmt) (ws : WSt F32) (i : Nat) (x : Nat × EWStmt × WSt F32) :
    ∀ n, x ∈ trace hw cta im allLeaf i (.forN n b) ws →
      ∃ j, j < n ∧ x ∈ trace hw cta im allLeaf j b (run hw cta im (.forN j b) ws 0)
  | 0, h => by simp [trace] at h
  | n + 1, h => by
      rw [trace_forN_succ] at h
      rcases List.mem_append.mp h with h | h
      · obtain ⟨j, hj, h⟩ := forN_leaves b ws i x n h
        exact ⟨j, by omega, h⟩
      · exact ⟨n, by omega, by rw [run_forN_i] at h; exact h⟩

theorem iwhile_leaves (c : IExp) (cs : List Nat) (b : EWStmt) (ws : WSt F32) (i : Nat) (x : Nat × EWStmt × WSt F32) :
    ∀ n, x ∈ trace hw cta im allLeaf i (.iwhile n c cs b) ws →
      ∃ j, j < n ∧ c.eval (run hw cta im (.iwhile j c cs b) ws i).int cta lane0 ≠ 0 ∧
        x ∈ trace hw cta im allLeaf i b (run hw cta im (.iwhile j c cs b) ws i)
  | 0, h => by simp [trace] at h
  | n + 1, h => by
      rw [trace_iwhile_succ] at h
      rcases List.mem_append.mp h with h | h
      · obtain ⟨j, hj, hc, h⟩ := iwhile_leaves c cs b ws i x n h
        exact ⟨j, by omega, hc, h⟩
      · split at h
        · rename_i hc; exact ⟨n, by omega, hc, h⟩
        · cases h

/-- **Every leaf a run reaches meets what `OkW` asks of it there.** -/
theorem okW_leaves : ∀ (s : EWStmt) (i : Nat) (ws : WSt F32) (cl : Bool),
    OkW hw cta im S ptr MF MB MI WS i s ws cl → ∀ x ∈ trace hw cta im allLeaf i s ws,
      ∃ cl', OkW hw cta im S ptr MF MB MI WS x.1 x.2.1 x.2.2 cl' := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro i ws cl h x hx
      simp only [trace] at hx
      rcases List.mem_append.mp hx with hx | hx
      · exact iha i ws cl h.1 x hx
      · exact ihb i _ _ h.2 x hx
  | iif c a b iha ihb =>
      intro i ws cl h x hx
      simp only [trace] at hx
      obtain ⟨_, h⟩ := h
      split at hx
      · rename_i hc; rw [if_pos hc] at h; exact iha i ws cl h x hx
      · rename_i hc; rw [if_neg hc] at h; exact ihb i ws cl h x hx
  | forN n b ih =>
      intro i ws cl h x hx
      obtain ⟨j, hj, hx⟩ := forN_leaves hw cta im b ws i x n hx
      exact ih j _ _ (h.2 j hj) x hx
  | forM bu a b ih =>
      intro i ws cl h x hx
      obtain ⟨j, hj, hx⟩ := forN_leaves hw cta im b ws i x (im bu a) hx
      exact ih j _ _ (h.2.2.2 j hj) x hx
  | iwhile n c cs b ih =>
      intro i ws cl h x hx
      obtain ⟨j, hj, hc, hx⟩ := iwhile_leaves hw cta im c cs b ws i x n hx
      have hu := (h.2 j (by omega)).1 lane0
      exact ih i _ _ ((h.2 j (by omega)).2 (hu.mp hc)) x hx
  | locals rs b ih => intro i ws cl h x hx; exact ih i ws cl h x hx
  | _ => intro i ws cl h x hx; simp only [trace, allLeaf, List.mem_singleton] at hx; subst hx; exact ⟨cl, h⟩

end Leaves

/-! ## A loop, one trip at a time -/

section Peel
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat)

theorem keep_wtrip (c : IExp) (cs : List Nat) (b : EWStmt) (ws : WSt F32) (i : Nat) (X : ISt) :
    ISt.keep cs (wtrip hw cta im c cs b ws i ws).int X = ISt.keep cs ws.int X := by
  simp only [wtrip]
  split
  · simp only [ISt.keep, WSt.withInt]
    congr 1; funext r; split <;> simp_all
  · rfl

theorem foldl_wtrip_congr (c : IExp) (cs : List Nat) (b : EWStmt) (ws ws' : WSt F32) (i : Nat)
    (h : ∀ X, ISt.keep cs ws.int X = ISt.keep cs ws'.int X) :
    ∀ (L : List Nat) (s : WSt F32), L.foldl (fun s' _ => wtrip hw cta im c cs b ws i s') s =
      L.foldl (fun s' _ => wtrip hw cta im c cs b ws' i s') s
  | [], _ => rfl
  | j :: L, s => by
      simp only [List.foldl_cons]
      have : wtrip hw cta im c cs b ws i s = wtrip hw cta im c cs b ws' i s := by
        simp only [wtrip]; split
        · rw [h]
        · rfl
      rw [this]; exact foldl_wtrip_congr c cs b ws ws' i h L _

/-- **A loop whose test holds runs a trip, then the loop.** -/
theorem run_iwhile_peel (n : Nat) (c : IExp) (cs : List Nat) (b : EWStmt) (ws : WSt F32) (i : Nat) :
    run hw cta im (.iwhile (n + 1) c cs b) ws i = run hw cta im (.iwhile n c cs b) (wtrip hw cta im c cs b ws i ws) i := by
  rw [run_iwhile, run_iwhile, List.range_succ_eq_map, List.foldl_cons, List.foldl_map]
  exact foldl_wtrip_congr hw cta im c cs b ws _ i (fun X => (keep_wtrip hw cta im c cs b ws i X).symm) _ _

/-- A loop whose test fails leaves the state. -/
theorem run_iwhile_stop (n : Nat) (c : IExp) (cs : List Nat) (b : EWStmt) (ws : WSt F32) (i : Nat)
    (hc : ¬ c.eval ws.int cta lane0 ≠ 0) : run hw cta im (.iwhile n c cs b) ws i = ws := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [run_iwhile_succ, ih]
    simp only [wtrip, if_neg hc]

theorem trace_iwhile_stop {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (n : Nat) (c : IExp) (cs : List Nat)
    (b : EWStmt) (ws : WSt F32) (i : Nat) (hc : ¬ c.eval ws.int cta lane0 ≠ 0) :
    trace hw cta im leaf i (.iwhile n c cs b) ws = [] := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [trace_iwhile_succ, ih, run_iwhile_stop hw cta im n c cs b ws i hc, if_neg hc]; rfl

theorem trace_iwhile_peel {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (n : Nat) (c : IExp) (cs : List Nat)
    (b : EWStmt) (ws : WSt F32) (i : Nat) (hc : c.eval ws.int cta lane0 ≠ 0) :
    trace hw cta im leaf i (.iwhile (n + 1) c cs b) ws =
      trace hw cta im leaf i b ws ++ trace hw cta im leaf i (.iwhile n c cs b) (wtrip hw cta im c cs b ws i ws) := by
  induction n with
  | zero =>
    rw [trace_iwhile_succ]
    simp only [trace, List.range_zero, List.foldl_nil, List.nil_append]
    rw [show run hw cta im (.iwhile 0 c cs b) ws i = ws from rfl, if_pos hc, List.append_nil]
  | succ n ih =>
    rw [trace_iwhile_succ, ih, trace_iwhile_succ, List.append_assoc, ← run_iwhile_peel]

end Peel

section PeelOk
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (S : Nat) (ptr : Buf → UInt64)
  (MF : Lane → Buf → Nat → Prop) (MB : Lane → UInt64 → Prop) (MI : Buf → Nat → Prop) (WS : Lane → UInt64 → Nat → Prop)

/-- A loop whose test fails in every lane asks nothing more. -/
theorem okW_iwhile_stop (n : Nat) (c : IExp) (cs : List Nat) (b : EWStmt) (ws : WSt F32) (i : Nat) (cl : Bool)
    (hu : ∀ l, (c.eval ws.int cta l ≠ 0 ↔ c.eval ws.int cta lane0 ≠ 0)) (hc : ¬ c.eval ws.int cta lane0 ≠ 0) :
    OkW hw cta im S ptr MF MB MI WS i (.iwhile n c cs b) ws cl := by
  refine ⟨⟨0, Nat.zero_le _, by simpa using hc⟩, fun j _ => ?_⟩
  rw [run_iwhile_stop hw cta im j c cs b ws i hc]
  exact ⟨hu, fun h => absurd h hc⟩

/-- **What a loop asks, one trip at a time.** -/
theorem okW_iwhile_peel (n : Nat) (c : IExp) (cs : List Nat) (b : EWStmt) (ws : WSt F32) (i : Nat) (cl : Bool)
    (hu : ∀ l, (c.eval ws.int cta l ≠ 0 ↔ c.eval ws.int cta lane0 ≠ 0)) (hc : c.eval ws.int cta lane0 ≠ 0)
    (hb : OkW hw cta im S ptr MF MB MI WS i b ws cl)
    (hrest : OkW hw cta im S ptr MF MB MI WS i (.iwhile n c cs b) (wtrip hw cta im c cs b ws i ws)
      (lastOr cl (trace hw cta im shLeaf i b ws))) :
    OkW hw cta im S ptr MF MB MI WS i (.iwhile (n + 1) c cs b) ws cl := by
  obtain ⟨⟨k, hk, hks⟩, hall⟩ := hrest
  refine ⟨⟨k + 1, by omega, by rw [run_iwhile_peel]; exact hks⟩, fun j hj => ?_⟩
  cases j with
  | zero => exact ⟨hu, fun _ => hb⟩
  | succ j =>
    rw [run_iwhile_peel, trace_iwhile_peel hw cta im shLeaf j c cs b ws i hc, lastOr_append]
    exact hall j (by omega)

end PeelOk

end AlgorithmLib.ML.LaneShared
