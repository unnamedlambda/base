module
public import Lz4.CompressorCoop
public import AlgorithmLib.ML.Ptx.BodyEnds
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CompressorBlock` — a block of the compressor runs with its warp

A lane's run keeps every byte outside shared memory as its byte stores leave
it (`laneI_tracks`). The warp's run of a block's thread (`compressorS`)
clears the table, passes the barrier and runs the body as the model does
(`prog_facts`), so each of the block's 32 threads runs with the warp
(`lane_warp`), the laws answer it as the warp does (`block_laws`), and the
block's memory outside shared memory is its lanes' byte stores (`block_runs`).
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML
  AlgorithmLib.ML.BodyLane AlgorithmLib.ML.LaneWarp AlgorithmLib.ML.LaneSettle AlgorithmLib.ML.LaneShared
  AlgorithmLib.ML.BodyEnds

/-! ## A lane's memory follows its byte stores -/

/-- Every store among observations is one byte or into shared memory. -/
def StOk (L : List Obs) : Prop := ∀ a n v, Obs.store a n v ∈ L → n = 1 ∨ ∃ off, decodeAddr a = some (.arena, off)

/-- `w'` goes on from `w`, by its byte stores where its stores are one byte
    or into shared memory. -/
def TrackR (w w' : World) : Prop := ∃ L, w'.obs = L ++ w.obs ∧ (StOk L → Tracks w w')

theorem TrackR.refl (w : World) : TrackR w w := ⟨[], rfl, fun _ => Tracks.refl w⟩

theorem TrackR.trans {a b c : World} (h1 : TrackR a b) (h2 : TrackR b c) : TrackR a c := by
  obtain ⟨L1, h1o, h1t⟩ := h1
  obtain ⟨L2, h2o, h2t⟩ := h2
  refine ⟨L2 ++ L1, by rw [h2o, h1o, List.append_assoc], fun h => ?_⟩
  exact (h1t fun a n v hm => h a n v (List.mem_append_right _ hm)).trans
    (h2t fun a n v hm => h a n v (List.mem_append_left _ hm))

/-- **A lane's run goes on by its byte stores** where its stores are one byte
    or into shared memory. -/
theorem laneI_tracks : ∀ (s : EWStmt) (x : LCtx) (li li' : LI), laneI x s li = some li' → TrackR li.w li'.w :=
  laneI_rel TrackR TrackR.refl (fun _ _ _ => TrackR.trans) (fun w c args => ⟨[_], rfl, fun _ => Tracks.call w c args⟩)
    (fun w b => ⟨[_], rfl, fun _ => (Tracks.call w (.intr .barrier) []).trans (Tracks.arena _ b)⟩)
    (fun w w' a n v h => by
      obtain ⟨m, hm, rfl⟩ := stw_some h
      refine ⟨[.store a n v], rfl, fun hs => ?_⟩
      rcases hs a n v (by simp) with rfl | ⟨off, hd⟩
      · exact Tracks.stw1 h
      · by_cases hn : n = 1
        · subst hn; exact Tracks.stw1 h
        · exact Tracks.stwArena hd hn h)

/-- **The byte stores a world records since `w`'s end** are those of the
    observations it goes on by. -/
theorem byteSt_storesOf {w w' : World} {L : List Obs} (h : w'.obs = L ++ w.obs) :
    byteSt (storesOf w' w.obs.length) = stores1 L.reverse := by
  unfold storesOf
  rw [h, List.reverse_append, List.drop_left' (by simp)]
  generalize L.reverse = M
  induction M with
  | nil => rfl
  | cons o M ih =>
    cases o with
    | store a n v =>
      by_cases hn : n = 1
      · subst hn; simp only [List.filterMap_cons, byteSt, stores1] at ih ⊢; rw [ih]
      · simpa [byteSt, stores1, hn] using ih
    | call c args =>
      cases c with
      | intr i => cases i <;> simpa [byteSt, stores1] using ih
      | _ => simpa [byteSt, stores1] using ih

/-- **A world's stores since `w`'s end are one byte or into shared memory**
    where every store it records since then is. -/
theorem stOk_of_storesOf {w w' : World} {L : List Obs} (h : w'.obs = L ++ w.obs)
    (hs : ∀ a n v, some (a, n, v) ∈ storesOf w' w.obs.length → n = 1 ∨ ∃ off, decodeAddr a = some (.arena, off)) :
    StOk L := by
  intro a n v hm
  apply hs a n v
  unfold storesOf
  rw [h, List.reverse_append, List.drop_left' (by simp)]
  exact List.mem_filterMap.mpr ⟨_, List.mem_reverse.mpr hm, rfl⟩

/-- **A lane's run leaves every byte outside shared memory that could be
    stored storable.** -/
theorem laneI_writable : ∀ (s : EWStmt) (x : LCtx) (li li' : LI), laneI x s li = some li' →
    ∀ a, NotArena a → Writable li.w.mem a 1 → Writable li'.w.mem a 1 :=
  laneI_rel (fun w w' => ∀ a, NotArena a → Writable w.mem a 1 → Writable w'.mem a 1) (fun _ _ _ h => h)
    (fun _ _ _ h1 h2 a ha hw => h2 a ha (h1 a ha hw)) (fun _ _ _ _ _ h => h)
    (fun _ b _ ha hw => writable_arena hw ha b) (fun _ _ _ _ _ h _ _ hw => writable_of_stw h hw)

section Traces
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat)

/-- **What holds of every leaf's entries holds of the trace's.** -/
theorem trace_leafwise {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (P : α → Prop) (s : EWStmt) (i : Nat)
    (ws : WSt F32) (h : ∀ x ∈ LaneWarp.trace hw cta im allLeaf i s ws, ∀ e ∈ leaf x.1 x.2.1 x.2.2, P e) :
    ∀ e ∈ LaneWarp.trace hw cta im leaf i s ws, P e :=
  trace_rel hw cta im leaf allLeaf (fun A B => (∀ x ∈ B, ∀ e ∈ leaf x.1 x.2.1 x.2.2, P e) → ∀ e ∈ A, P e)
    (fun _ e he => by cases he)
    (fun a b a' b' h h' hab e he => by
      rcases List.mem_append.mp he with he | he
      · exact h (fun x hx => hab x (List.mem_append_left _ hx)) e he
      · exact h' (fun x hx => hab x (List.mem_append_right _ hx)) e he)
    s i ws (fun x _ hx => hx x List.mem_cons_self) h

/-- **A lane's answers from the warp are words.** -/
theorem words_ans (S : Nat) (l : Lane) (s : EWStmt) (i : Nat) (ws : WSt F32) :
    Words (LaneWarp.trace hw cta im (ansLeaf cta S l) i s ws) := by
  intro x hx v hv
  refine trace_leafwise hw cta im (ansLeaf cta S l) (fun x => ∀ v, x = .val v → ∃ y : UInt32, v = .sc .i32 (s32 y))
    s i ws (fun y _ e he => ?_) x hx v hv
  obtain ⟨i', s', ws'⟩ := y
  cases s' <;> simp only [ansLeaf, List.mem_cons, List.not_mem_nil, or_false] at he <;>
    first | (subst he; intro v hv; cases hv; exact ⟨_, rfl⟩) | (subst he; intro v hv; cases hv) | exact he.elim

end Traces

/-- **A thread of the compressor's kernel ends as its lane's run does.** -/
theorem compressor_thread (hw : Hw) (hw32 : ∀ k v, (hw.approx k v).toNat < 2 ^ 32) {ans : List Answer}
    (hW : Words ans) (name : String) {numBlocks inStride outStride lenOff hashLog cta t : Nat} {inP outA : UInt64}
    {w0 : World} {li' : LI}
    (hfine : Fine (emitGo (fnBody (idParams ++ [.i64, .i64])
      (ML.Body.body 2 (compressorS numBlocks inStride outStride lenOff hashLog)))
      ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2)
    (hl : laneI (threadCtx cta t [inP, outA] ans w0.obs.length hw) (compressorS numBlocks inStride outStride lenOff hashLog)
      (LI.init w0) = some li') :
    ∃ steps u, thread (compressorKernel name numBlocks inStride outStride lenOff hashLog).code ans
        (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t) w0 hw steps = .ok u ∧ u = li'.w := by
  have hE := emit_ends (cfg := { env := [], locals := threadLocals ans w0.obs.length hw })
    (P0 := HProg.Hoare.At (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t).toArray w0)
    (wpE_bodyI 2 _ cta t 32 [inP, outA] rfl w0 ans hw (laneOk_thread cta t [inP, outA] ans _ hw hW hw32) li' hl)
    hfine
    (fun Γ' w' h => ⟨by rw [h.1]; simp [threadArgs, idParams], h⟩)
  obtain ⟨k, Γ', u, hk, hu⟩ := hE _ _ ⟨rfl, rfl⟩
  refine ⟨k, u, ?_, hu⟩
  unfold thread
  rw [show (compressorKernel name numBlocks inStride outStride lenOff hashLog).code
    = emit (ML.Body.body 2 (compressorS numBlocks inStride outStride lenOff hashLog))
      (idParams ++ List.replicate 2 .i64) from rfl, hk]

/-! ## The compressor's leaves -/

/-- The statements LZ4's translation is made of, below its branches, loops
    and locals. -/
def lz4Leaf : EWStmt → Bool
  | .setI .. | .ldB .. | .stB .. | .ldH .. | .stH .. | .ballot .. | .shflIdx .. | .barrier | .skip => true
  | _ => false

section Leaves
variable (im : Buf → Nat → Nat)

theorem allAt_seqs (i : Nat) : ∀ (L : List EWStmt), (∀ s ∈ L, AllAt im (fun _ t => lz4Leaf t = true) i s) →
    AllAt im (fun _ t => lz4Leaf t = true) i (seqs L)
  | [], _ => rfl
  | [s], h => h s (by simp)
  | s :: t :: L, h => ⟨h s (by simp), allAt_seqs i (t :: L) (fun x hx => h x (by simp [hx]))⟩

/-- **Every leaf of LZ4's translation is one of its statements.** -/
theorem allAt_ofW (R : String → Nat) (T F : Nat) (fl : List String) (i : Nat) :
    ∀ s : WStmt, AllAt im (fun _ t => lz4Leaf t = true) i (ofW R T F fl s) := by
  intro s
  induction s with
  | seq a b iha ihb => exact ⟨iha, ihb⟩
  | uif c a b iha ihb => exact ⟨iha, ihb⟩
  | uwhile c b ih => exact ih
  | coopCopy d s n =>
    refine allAt_seqs im i _ (fun x hx => ?_)
    simp only [List.mem_cons, List.mem_nil_iff, or_false] at hx
    rcases hx with rfl | rfl
    · rfl
    · exact allAt_seqs im i _ (by simp [AllAt, lz4Leaf])
  | coopWindow =>
    show AllAt im _ i (.locals _ _)
    exact allAt_seqs im i _ (by simp [winPre, winSuf, load4E, AllAt, lz4Leaf])
  | coopExtendStep => exact allAt_seqs im i _ (by simp [AllAt, lz4Leaf])
  | _ => rfl

/-- **Every leaf of a block's thread is one of LZ4's statements.** -/
theorem allAt_compressorS (numBlocks inStride outStride lenOff hashLog : Nat) :
    AllAt im (fun _ t => lz4Leaf t = true) 0 (compressorS numBlocks inStride outStride lenOff hashLog) := by
  refine ⟨allAt_seqs im 0 _ (fun x hx => ?_), rfl⟩
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hx
  rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact ⟨rfl, rfl⟩
  · rfl
  · exact allAt_ofW im _ _ _ _ 0 _

end Leaves

theorem keep_regs (cs : List Nat) (st0 st : ISt) (r : Nat) :
    (ISt.keep cs st0 st).regs r = if r ∈ cs then st.regs r else st0.regs r := rfl
theorem keep_smem (cs : List Nat) (st0 st : ISt) : (ISt.keep cs st0 st).smem = st.smem := rfl
theorem keep_gmem (cs : List Nat) (st0 st : ISt) : (ISt.keep cs st0 st).gmem = st.gmem := rfl

/-! ## The table clear -/

theorem storeH_zero (st : ISt) (a v g : Lane → UInt64) (hs : ∀ k, st.smem k = 0) (hv : ∀ l, v l = 0) :
    ∀ k, (st.storeH a v g).smem k = 0 := by
  unfold ISt.storeH
  refine foldl_inv _ (fun s : ISt => ∀ k, s.smem k = 0) (fun s l h => ?_) _ st hs
  split
  · intro k; dsimp only; split
    · rw [hv]; rfl
    · split
      · rw [hv]; rfl
      · exact h k
  · exact h

/-- **A lane's table slot in the clear** is twice its entry. -/
theorem clearA_eval (ci tb hashLog : Nat) (X : ISt) (cta : Nat) (l : Lane) (m : Nat) (hm : m < 2 ^ hashLog)
    (h36 : 2 ^ hashLog < 2 ^ 36) (hci : X.regs ci l = UInt64.ofNat m) (htb : X.regs tb l = 0) :
    ((IExp.bin .band (.bin .add (.reg tb) (.bin .shl (.reg ci) (.lit 1))) (.lit (2 * 2 ^ hashLog - 1))).eval X cta l).toNat =
      2 * m := by
  simp only [IExp.eval, IOp.run, htb, hci, UInt64.zero_add]
  rw [UInt64.toNat_and, show (2 * 2 ^ hashLog - 1 : Nat) = 2 ^ (hashLog + 1) - 1 by rw [Nat.pow_succ]; omega,
    UInt64.toNat_ofNat_of_lt' (show 2 ^ (hashLog + 1) - 1 < 2 ^ 64 by rw [Nat.pow_succ]; omega),
    Nat.and_two_pow_sub_one_eq_mod, UInt64.toNat_shiftLeft, UInt64.toNat_ofNat_of_lt' (show m < 2 ^ 64 by omega),
    show (UInt64.ofNat 1).toNat % 64 = 1 from rfl, Nat.shiftLeft_eq, Nat.pow_one,
    Nat.mod_eq_of_lt (show m * 2 < 2 ^ 64 by omega),
    Nat.mod_eq_of_lt (show m * 2 < 2 ^ (hashLog + 1) by rw [Nat.pow_succ]; omega)]
  omega

theorem run_barrier (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (X : WSt F32) (i : Nat) :
    LaneWarp.run hw cta im .barrier X i = X := rfl

theorem trace_iif {α : Type} (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (leaf : Nat → EWStmt → WSt F32 → List α)
    (c : IExp) (a b : EWStmt) (ws : WSt F32) (i : Nat) (hc : c.eval ws.int cta lane0 ≠ 0) :
    LaneWarp.trace hw cta im leaf i (.iif c a b) ws = LaneWarp.trace hw cta im leaf i a ws := by
  simp only [LaneWarp.trace, if_pos hc]

theorem byteSt_eq_nil : ∀ (L : List (Option (UInt64 × Nat × UInt64))),
    (∀ a n v, some (a, n, v) ∈ L → n ≠ 1) → byteSt L = []
  | [], _ => rfl
  | none :: L, h => by
      rw [show byteSt (none :: L) = byteSt L from rfl]
      exact byteSt_eq_nil L (fun a n v hm => h a n v (List.mem_cons_of_mem _ hm))
  | some (a, n, v) :: L, h => by
      have hn := h a n v List.mem_cons_self
      rw [show byteSt (some (a, n, v) :: L) = byteSt L by simp [byteSt, hn]]
      exact byteSt_eq_nil L (fun a n v hm => h a n v (List.mem_cons_of_mem _ hm))

section Clear
variable {hw : Hw} {cta S : Nat} {ptr : Buf → UInt64} {I O : UInt64 → Prop}

/-- **The table clear's loop**: from trip `k`, lane `l`'s counter `32k + l`,
    it asks what lanes need, leaves every other register and global memory,
    keeps shared memory zero, and its shared events are one store per trip,
    lane `l`'s counter `32j + l` at trip `j`. -/
theorem clear_loop (ci tb hashLog : Nat) (hct : ci ≠ tb) (h5 : 5 ≤ hashLog) (hS : 2 * 2 ^ hashLog ≤ S)
    (hS36 : S ≤ 2 ^ 36) (X0 : WSt F32) (htb : ∀ l, X0.int.regs tb l = 0) (i : Nat) :
    let c : IExp := .cmp .ult (.reg ci) (.lit (2 ^ hashLog))
    let A : IExp := .bin .band (.bin .add (.reg tb) (.bin .shl (.reg ci) (.lit 1))) (.lit (2 * 2 ^ hashLog - 1))
    let body : EWStmt := .seq (.stH A (.lit 0) (.lit 1)) (.setI ci (.bin .add (.reg ci) (.lit 32)))
    ∀ F k (X : WSt F32), 2 ^ hashLog ≤ 32 * (k + F) → 32 * k ≤ 2 ^ hashLog →
      (∀ l : Lane, X.int.regs ci l = UInt64.ofNat (32 * k + l.val)) → (∀ r, r ≠ ci → X.int.regs r = X0.int.regs r) →
      (∀ k', X.int.smem k' = 0) → X.int.gmem = X0.int.gmem → ∀ cl,
      OkL hw cta S ptr I O i (.iwhile F c [ci] body) X cl ∧
      (∃ Y : WSt F32, LaneWarp.run hw cta (fun _ _ => 0) (.iwhile F c [ci] body) X i = Y ∧
        (∀ r, r ≠ ci → Y.int.regs r = X0.int.regs r) ∧ (∀ k', Y.int.smem k' = 0) ∧ Y.int.gmem = X0.int.gmem) ∧
      (LaneWarp.trace hw cta (fun _ _ => 0) shEvLeaf i (.iwhile F c [ci] body) X).length = 2 ^ hashLog / 32 - k ∧
      ∀ j e, (LaneWarp.trace hw cta (fun _ _ => 0) shEvLeaf i (.iwhile F c [ci] body) X)[j]? = some e →
        ∃ Xj : WSt F32, e = .st Xj A (.lit 0) (.lit 1) ∧
          (∀ l : Lane, Xj.int.regs ci l = UInt64.ofNat (32 * (k + j) + l.val)) ∧ ∀ l, Xj.int.regs tb l = 0 := by
  intro c A body
  have hpow : 32 ∣ 2 ^ hashLog := by
    rw [show 2 ^ hashLog = 2 ^ 5 * 2 ^ (hashLog - 5) by rw [← Nat.pow_add]; congr 1; omega]
    exact Nat.dvd_mul_right _ _
  have hN36 : 2 ^ hashLog < 2 ^ 36 := by omega
  have test : ∀ k (X : WSt F32), 32 * k ≤ 2 ^ hashLog → (∀ l : Lane, X.int.regs ci l = UInt64.ofNat (32 * k + l.val)) →
      ∀ l : Lane, (c.eval X.int cta l ≠ 0 ↔ 32 * k < 2 ^ hashLog) := by
    intro k X hk hX l
    have hl : l.val < 32 := l.isLt
    have hlt : UInt64.ofNat (32 * k + l.val) < UInt64.ofNat (2 ^ hashLog) ↔ 32 * k < 2 ^ hashLog := by
      rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' (show 32 * k + l.val < 2 ^ 64 by omega),
        UInt64.toNat_ofNat_of_lt' (show 2 ^ hashLog < 2 ^ 64 by omega)]
      obtain ⟨q, hq⟩ := hpow
      constructor
      · intro h; omega
      · intro h
        have : k < q := by omega
        omega
    simp only [c, IExp.eval, ICmp.run, hX]
    by_cases h : 32 * k < 2 ^ hashLog
    · rw [if_pos (decide_eq_true (hlt.mpr h))]; exact ⟨fun _ => h, fun _ => by decide⟩
    · rw [if_neg (fun e => h (hlt.mp (of_decide_eq_true e)))]; exact ⟨fun e => absurd rfl e, fun e => absurd e h⟩
  have stop : ∀ F k (X : WSt F32), ¬ 32 * k < 2 ^ hashLog → 32 * k ≤ 2 ^ hashLog →
      (∀ l : Lane, X.int.regs ci l = UInt64.ofNat (32 * k + l.val)) → (∀ r, r ≠ ci → X.int.regs r = X0.int.regs r) →
      (∀ k', X.int.smem k' = 0) → X.int.gmem = X0.int.gmem → ∀ cl,
      OkL hw cta S ptr I O i (.iwhile F c [ci] body) X cl ∧
      (∃ Y : WSt F32, LaneWarp.run hw cta (fun _ _ => 0) (.iwhile F c [ci] body) X i = Y ∧
        (∀ r, r ≠ ci → Y.int.regs r = X0.int.regs r) ∧ (∀ k', Y.int.smem k' = 0) ∧ Y.int.gmem = X0.int.gmem) ∧
      (LaneWarp.trace hw cta (fun _ _ => 0) shEvLeaf i (.iwhile F c [ci] body) X).length = 2 ^ hashLog / 32 - k ∧
      ∀ j e, (LaneWarp.trace hw cta (fun _ _ => 0) shEvLeaf i (.iwhile F c [ci] body) X)[j]? = some e →
        ∃ Xj : WSt F32, e = .st Xj A (.lit 0) (.lit 1) ∧
          (∀ l : Lane, Xj.int.regs ci l = UInt64.ofNat (32 * (k + j) + l.val)) ∧ ∀ l, Xj.int.regs tb l = 0 := by
    intro F k X h hk hX hXr hXs hXg cl
    have ht : ¬ c.eval X.int cta lane0 ≠ 0 := fun h' => h ((test k X hk hX lane0).mp h')
    have hu : ∀ l, (c.eval X.int cta l ≠ 0 ↔ c.eval X.int cta lane0 ≠ 0) := fun l => by
      rw [test k X hk hX l, test k X hk hX lane0]
    have hr := run_iwhile_stop hw cta (fun _ _ => 0) F c [ci] body X i ht
    have htr := trace_iwhile_stop hw cta (fun _ _ => 0) shEvLeaf F c [ci] body X i ht
    refine ⟨okW_iwhile_stop hw cta (fun _ _ => 0) S ptr _ _ _ _ F c [ci] body X i cl hu ht, ⟨X, hr, hXr, hXs, hXg⟩,
      ?_, ?_⟩
    · rw [htr]; simp only [List.length_nil]
      have : 2 ^ hashLog / 32 ≤ k := Nat.div_le_of_le_mul (by omega)
      omega
    · rw [htr]; intro j e h; simp at h
  intro F
  induction F with
  | zero =>
    intro k X hF hk hX hXr hXs hXg cl
    exact stop 0 k X (by omega) hk hX hXr hXs hXg cl
  | succ F ih =>
    intro k X hF hk hX hXr hXs hXg cl
    by_cases h : 32 * k < 2 ^ hashLog
    case neg => exact stop _ k X h hk hX hXr hXs hXg cl
    have ht : c.eval X.int cta lane0 ≠ 0 := (test k X hk hX lane0).mpr h
    have hu : ∀ l, (c.eval X.int cta l ≠ 0 ↔ c.eval X.int cta lane0 ≠ 0) := fun l => by
      rw [test k X hk hX l, test k X hk hX lane0]
    obtain ⟨q, hq⟩ := hpow
    have hk1 : 32 * k + 32 ≤ 2 ^ hashLog := by
      have : k < q := by omega
      omega
    have htb' : ∀ l, X.int.regs tb l = 0 := fun l => by rw [hXr tb (Ne.symm hct)]; exact htb l
    have hA : ∀ l : Lane, (A.eval X.int cta l).toNat = 2 * (32 * k + l.val) := fun l => by
      have hl : l.val < 32 := l.isLt
      exact clearA_eval ci tb hashLog X.int cta l (32 * k + l.val) (by omega) (by omega) (hX l) (htb' l)
    obtain ⟨X1, e1⟩ : ∃ X1, X1 = X.int.storeH (fun l => A.eval X.int cta l) (fun l => (IExp.lit 0).eval X.int cta l)
        (fun l => (IExp.lit 1).eval X.int cta l) := ⟨_, rfl⟩
    obtain ⟨X2, e2⟩ : ∃ X2, X2 = X1.set ci (fun l => (IExp.bin .add (.reg ci) (.lit 32)).eval X1 cta l) := ⟨_, rfl⟩
    have hrb : LaneWarp.run hw cta (fun _ _ => 0) body X i = X.withInt X2 := by
      rw [e2, e1]; simp only [body, LaneWarp.run, EWStmt.run, withInt_int, withInt_withInt]
    have hwt : wtrip hw cta (fun _ _ => 0) c [ci] body X i X = X.withInt (ISt.keep [ci] X.int X2) := by
      simp only [wtrip, if_pos ht, hrb, withInt_int, withInt_withInt]
    have hX1 := storeH_regs (a := fun l => A.eval X.int cta l) (v := fun l => (IExp.lit 0).eval X.int cta l)
      (g := fun l => (IExp.lit 1).eval X.int cta l) X.int
    rw [← e1] at hX1
    have k1 : ∀ l : Lane, (ISt.keep [ci] X.int X2).regs ci l = UInt64.ofNat (32 * (k + 1) + l.val) := fun l => by
      rw [keep_regs, if_pos (List.mem_singleton_self _), e2, set_regs_eq]
      simp only [IExp.eval, IOp.run]
      rw [hX1.1, hX, ofNat_add_ofNat]
      congr 1; omega
    have kr : ∀ r, r ≠ ci → (ISt.keep [ci] X.int X2).regs r = X0.int.regs r := fun r hr => by
      rw [keep_regs, if_neg (by simpa using hr)]; exact hXr r hr
    have eKs : (ISt.keep [ci] X.int X2).smem = X1.smem := by rw [keep_smem, e2, set_smem]
    have eKg : (ISt.keep [ci] X.int X2).gmem = X1.gmem := by rw [keep_gmem, e2, set_gmem]
    have ks : ∀ k', (ISt.keep [ci] X.int X2).smem k' = 0 := fun k' => by
      rw [eKs, e1]; exact storeH_zero X.int _ _ _ hXs (fun _ => rfl) k'
    have kg : (ISt.keep [ci] X.int X2).gmem = X0.int.gmem := by
      rw [eKg, hX1.2.1]; exact hXg
    have hF1 : 2 ^ hashLog ≤ 32 * (k + 1 + F) := by omega
    have hk1' : 32 * (k + 1) ≤ 2 ^ hashLog := by omega
    obtain ⟨o', ⟨Y, hY, r', s', g'⟩, len', ev'⟩ := ih (k + 1) (X.withInt (ISt.keep [ci] X.int X2)) hF1 hk1' k1 kr ks kg
      false
    have hbody : OkL hw cta S ptr I O i body X cl := ⟨⟨hS36, fun l _ => by rw [hA]; have : l.val < 32 := l.isLt; omega⟩, trivial⟩
    have hsh : LaneWarp.trace hw cta (fun _ _ => 0) shLeaf i body X = [false] := rfl
    have hev : LaneWarp.trace hw cta (fun _ _ => 0) shEvLeaf i body X = [.st X A (.lit 0) (.lit 1)] := rfl
    refine ⟨okW_iwhile_peel hw cta _ S ptr _ _ _ _ F c [ci] body X i cl hu ht hbody (by rw [hsh, hwt]; exact o'),
      ⟨Y, by rw [run_iwhile_peel, hwt]; exact hY, r', s', g'⟩, ?_, ?_⟩
    · rw [trace_iwhile_peel hw cta _ _ F c [ci] body X i ht, hev, hwt, List.length_append, len']
      simp only [List.length_singleton]
      have : k < 2 ^ hashLog / 32 := by omega
      omega
    · intro j e hj
      rw [trace_iwhile_peel hw cta _ _ F c [ci] body X i ht, hev, hwt] at hj
      cases j with
      | zero =>
        simp only [List.cons_append, List.nil_append, List.getElem?_cons_zero, Option.some.injEq] at hj
        exact ⟨X, hj.symm, fun l => by rw [hX, Nat.add_zero], htb'⟩
      | succ j =>
        simp only [List.cons_append, List.nil_append, List.getElem?_cons_succ] at hj
        obtain ⟨Xj, he, hc, ht'⟩ := ev' j e hj
        exact ⟨Xj, he, fun l => by rw [hc]; congr 2; omega, ht'⟩

end Clear

/-! ## Shared memory's epochs -/

theorem Paired.tail {e : ShEv} {E : List ShEv} (h : Paired (e :: E)) : Paired E :=
  fun k ws a v g hk => by
    obtain ⟨ws', h'⟩ := h (k + 1) ws a v g (by simpa using hk)
    exact ⟨ws', by simpa using h'⟩

/-- **Where every shared store is followed by a barrier, an epoch holds at
    most one store.** -/
theorem paired_epoch : ∀ (E : List ShEv), Paired E → ∀ b0 b, (epochE b0 b E).length ≤ 1
  | [], _, _, _ => by simp [epochE]
  | .bar _ :: r, h, b0, b => paired_epoch r h.tail (b0 + 1) b
  | .st ws a v g :: r, h, b0, b => by
      obtain ⟨ws', h'⟩ := h 0 ws a v g rfl
      cases r with
      | nil => simp at h'
      | cons e r =>
        simp only [List.getElem?_cons_succ, List.getElem?_cons_zero, Option.some.injEq] at h'
        subst h'
        simp only [epochE]
        by_cases hb : b0 = b
        · subst hb; rw [epochE_lt r (b0 + 1) b0 (by omega)]; simp
        · rw [if_neg hb, List.nil_append]; exact paired_epoch r h.tail.tail (b0 + 1) b

theorem epochE_sts : ∀ (C : List ShEv), (∀ e ∈ C, e.isBar = false) → ∀ b0 b,
    epochE b0 b C = if b0 = b then C else []
  | [], _, _, _ => by simp [epochE]
  | .bar _ :: _, h, _, _ => by simp [ShEv.isBar] at h
  | .st ws a v g :: r, h, b0, b => by
      simp only [epochE]
      rw [epochE_sts r (fun e he => h e (List.mem_cons_of_mem _ he)) b0 b]
      split <;> simp

theorem bars_sts : ∀ (C : List ShEv), (∀ e ∈ C, e.isBar = false) → LaneShared.bars C = 0
  | [], _ => rfl
  | e :: r, h => by
      simp only [LaneShared.bars, List.countP_cons, h e List.mem_cons_self, Bool.false_eq_true, if_false,
        Nat.add_zero]
      exact bars_sts r (fun e he => h e (List.mem_cons_of_mem _ he))

theorem byteAt_ne {x m : Nat} {v : UInt64} (h : [(m, v)].filterMap (byteAt x) ≠ []) : x = m ∨ x = m + 1 := by
  by_cases e1 : x = m
  · exact Or.inl e1
  by_cases e2 : x = m + 1
  · exact Or.inr e2
  exfalso; apply h
  simp [byteAt, e1, e2]

/-- **Shared memory's events are apart within each epoch** where the first
    epoch's are stores whose bytes, lane by lane, store by store, are apart
    and every later store is followed by a barrier. -/
theorem shOk_of (cta : Nat) (C E : List ShEv) (w : WSt F32) (hC : ∀ e ∈ C, e.isBar = false)
    (hCd : ∀ (x i j : Nat) (l l' : Lane) (e e' : ShEv), C[i]? = some e → C[j]? = some e' →
      (wrF cta l e).filterMap (byteAt x) ≠ [] → (wrF cta l' e').filterMap (byteAt x) ≠ [] → i = j ∨ l = l')
    (hE : Paired E) : ShOk cta (C ++ .bar w :: E) := by
  intro b x i j l l' e e' hi hj hx hx'
  rw [epochE_append, epochE_sts C hC, bars_sts C hC, Nat.zero_add] at hi hj
  simp only [epochE] at hi hj
  by_cases hb : 0 = b
  · subst hb
    rw [if_pos rfl, epochE_lt E 1 0 (by omega), List.append_nil] at hi hj
    exact hCd x i j l l' e e' hi hj hx hx'
  · rw [if_neg hb, List.nil_append] at hi hj
    have hlen := paired_epoch E hE (0 + 1) b
    have := (List.getElem?_eq_some_iff.mp hi).1
    have := (List.getElem?_eq_some_iff.mp hj).1
    left; omega

/-! ## A block's warp -/

/-- The table a block's body starts from: the clear's zeros. -/
def clearMem (hashLog : Nat) : ByteArray := ⟨Array.replicate (2 ^ hashLog * 2) 0⟩

/-- The model's state a block's body starts from: registers holding the
    block's bases, global memory `G`, the cleared table. -/
def startWS (numBlocks inStride outStride hashLog cta : Nat) (inP : UInt64) (G : Array UInt8) : WState :=
  { regs := regs0 (inP + UInt64.ofNat cta * c64 inStride)
      (inP + c64 (numBlocks * inStride + copySlack) + UInt64.ofNat cta * c64 outStride),
    gmem := G, smem := (clearMem hashLog).data }

/-- The warp's state a block's thread starts from: every register zero, its
    parameters the input and output pointers, global memory `G`, shared
    memory zero. -/
def blockWS (inP outA : UInt64) (G : Array UInt8) : WSt F32 :=
  { regs := (fun _ _ => ⟨0⟩), mem := (fun _ _ => ⟨0⟩), smem := (fun _ => ⟨0⟩),
    int := ⟨fun _ _ => 0, fun k => G.getD k 0, fun _ => 0, fun b => [inP, outA].getD b 0⟩ }

/-! ## The model's stores lie outside the input -/

/-- **The model's byte stores lie outside the input and shared memory** where
    its run makes only the accesses `okOf` allows. -/
theorem gstOf_out {I O : UInt64 → Prop} {okC : WStmt → WState → Prop} :
    ∀ (s : WStmt) (f : Nat) (W : WState), okOf I O okC f s W → ∀ e ∈ gstOf f s W, ¬ I e.2.1 ∧ NotArena e.2.1 := by
  intro s
  induction s with
  | seq a b iha ihb =>
    intro f W h e he
    simp only [okOf] at h
    simp only [gstOf, List.mem_append] at he
    rcases he with he | he
    · exact iha f W h.1 e he
    · exact ihb f _ h.2 e he
  | stgB addr val =>
    intro f W h e he
    simp only [okOf] at h
    simp only [gstOf, List.mem_singleton] at he
    subst he; exact ⟨h.1, h.2.1⟩
  | uif c a b iha ihb =>
    intro f W h e he
    simp only [okOf] at h
    simp only [gstOf] at he
    split at he
    · rename_i hc; rw [if_pos hc] at h; exact iha f W h e he
    · rename_i hc; rw [if_neg hc] at h; exact ihb f W h e he
  | uwhile c b ih =>
    intro f
    induction f with
    | zero => intro W _ e he; simp [gstOf] at he
    | succ f ihf =>
      intro W h e he
      simp only [okOf] at h
      simp only [gstOf] at he
      split at he
      · rename_i hc
        obtain ⟨h1, h2⟩ := h hc
        rcases List.mem_append.mp he with he | he
        · exact ih f W h1 e he
        · exact ihf _ h2 e he
      · cases he
  | coopCopy d s n =>
    intro f W h e he
    simp only [okOf] at h
    simp only [gstOf, copyTr, List.mem_map, List.mem_range] at he
    obtain ⟨i, hi, rfl⟩ := he
    exact ⟨(h.2.2.2.2 i hi).1, (h.2.2.2.2 i hi).2.1⟩
  | _ => intro f W _ e he; simp [gstOf] at he

theorem lanePart_mem {l : Nat} {L : List (Option Nat × UInt64 × UInt64)} {x : UInt64 × UInt64} (h : x ∈ lanePart l L) :
    ∃ who, (who, x.1, x.2) ∈ L := by
  simp only [lanePart, List.mem_filterMap] at h
  obtain ⟨⟨who, a, v⟩, hm, hx⟩ := h
  split at hx
  · obtain rfl := (Option.some.inj hx).symm; exact ⟨who, hm⟩
  · cases hx

theorem replay_notin (a : UInt64) : ∀ (L : List (UInt64 × UInt64)) (d : Option UInt64), (∀ x ∈ L, x.1 ≠ a) →
    replay L a d = d
  | [], _, _ => rfl
  | x :: L, d, h => by
      simp only [replay, List.foldl_cons, if_neg (h x List.mem_cons_self)]
      exact replay_notin a L d (fun y hy => h y (List.mem_cons_of_mem _ hy))

theorem storesOf_self (w : World) : storesOf w w.obs.length = [] := by
  unfold storesOf
  rw [show w.obs.reverse.drop w.obs.length = [] from List.drop_eq_nil_of_le (by simp)]
  rfl

theorem load_decode {m : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.load a n = some v) :
    ∃ r o, decodeAddr a = some (r, o) := by
  unfold Mem.load at h
  cases hd : decodeAddr a with
  | none => rw [hd] at h; cases h
  | some p => exact ⟨p.1, p.2, rfl⟩

section Leaf
variable {hw : Hw} {cta S : Nat} {ptr : Buf → UInt64} {I O : UInt64 → Prop}

/-- **What each of LZ4's leaves gives a block**, from what it asks of the
    warp: its stores one byte or into shared memory, its shared stores its
    events' for each lane, inside shared memory, its collectives a ballot, a
    shuffle or a barrier. -/
theorem leaf_facts (hS : S ≤ 2 ^ 36) (l : Lane) (x : Nat × EWStmt × WSt F32)
    (hok : ∃ cl, OkL hw cta S ptr I O x.1 x.2.1 x.2.2 cl) (hlz : lz4Leaf x.2.1 = true) :
    (∀ a n v, some (a, n, v) ∈ storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0) x.1 x.2.1 x.2.2 →
      n = 1 ∨ ∃ off, decodeAddr a = some (.arena, off)) ∧
    TagRel cta (addrOf .arena 0) l (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0) x.1 x.2.1 x.2.2)
      (shEvLeaf x.1 x.2.1 x.2.2) ∧
    (∀ e ∈ shEvLeaf x.1 x.2.1 x.2.2, ∀ l', ∀ w ∈ wrF cta l' e, w.1 + 2 ≤ S) ∧
    (∀ e ∈ jLeaf x.1 x.2.1 x.2.2, ∃ i ws, (∃ d p, e = (i, .ballot d p, ws)) ∨ (∃ d v sr, e = (i, .shflIdx d v sr, ws)) ∨
      (∃ d sr m, e = (i, .shflXor d sr m, ws) ∧ m < 2 ^ 31) ∨ e = (i, .barrier, ws)) := by
  obtain ⟨i, s, ws⟩ := x
  obtain ⟨cl, hok⟩ := hok
  unfold TagRel
  dsimp only at hok hlz ⊢
  cases s with
  | stB a v g =>
    refine ⟨?_, ⟨?_, ?_⟩, ?_, ?_⟩
    · intro a' n v' hm
      simp only [storeLeaf] at hm
      split at hm
      · simp at hm; exact Or.inl hm.2.1
      · cases hm
    · simp only [storeLeaf]; split <;> rfl
    · intro b
      simp only [storeLeaf]
      split
      · rename_i hg
        obtain ⟨⟨-, r, o, hd, hr, -⟩, -⟩ := hok l hg
        simp only [tagFrom, hd]
        cases r <;> first | exact absurd rfl hr | rfl
      · rfl
    · intro e he; simp [shEvLeaf] at he
    · intro e he; simp [jLeaf] at he
  | stH a v g =>
    obtain ⟨-, hin⟩ := hok
    refine ⟨?_, ⟨?_, ?_⟩, ?_, ?_⟩
    · intro a' n v' hm
      simp only [storeLeaf] at hm
      split at hm
      · rename_i hg
        simp at hm
        obtain ⟨rfl, -, -⟩ := hm
        exact Or.inr ⟨_, arena_decodeL (by have := hin l hg; omega)⟩
      · cases hm
    · simp only [storeLeaf]; split <;> rfl
    · intro b
      simp only [storeLeaf, shEvLeaf, tagE]
      by_cases hg : IExp.eval ws.int cta l g ≠ 0
      · rw [if_pos hg, if_pos hg]
        simp only [tagFrom, arena_decodeL (show (a.eval ws.int cta l).toNat < 2 ^ 36 by have := hin l hg; omega)]
      · rw [if_neg hg, if_neg hg]; rfl
    · intro e he l' w hw
      simp only [shEvLeaf, List.mem_singleton] at he
      subst he
      simp only [wrF] at hw
      split at hw
      · rename_i hg; simp at hw; subst hw; exact hin l' hg
      · cases hw
    · intro e he; simp [jLeaf] at he
  | barrier =>
    refine ⟨?_, ⟨rfl, fun b => rfl⟩, ?_, ?_⟩
    · intro a' n v' hm; simp [storeLeaf] at hm
    · intro e he l' w hw; simp only [shEvLeaf, List.mem_singleton] at he; subst he; cases hw
    · intro e he; simp only [jLeaf, List.mem_singleton] at he; exact ⟨i, ws, Or.inr (Or.inr (Or.inr he))⟩
  | ballot d p =>
    refine ⟨?_, ⟨rfl, fun b => rfl⟩, ?_, ?_⟩
    · intro a' n v' hm; simp [storeLeaf] at hm
    · intro e he; simp [shEvLeaf] at he
    · intro e he; simp only [jLeaf, List.mem_singleton] at he; exact ⟨i, ws, Or.inl ⟨d, p, he⟩⟩
  | shflIdx d v sr =>
    refine ⟨?_, ⟨rfl, fun b => rfl⟩, ?_, ?_⟩
    · intro a' n v' hm; simp [storeLeaf] at hm
    · intro e he; simp [shEvLeaf] at he
    · intro e he; simp only [jLeaf, List.mem_singleton] at he; exact ⟨i, ws, Or.inr (Or.inl ⟨d, v, sr, he⟩)⟩
  | setI | ldB | ldH | skip =>
    refine ⟨?_, ⟨rfl, fun b => rfl⟩, ?_, ?_⟩
    · intro a' n v' hm; simp [storeLeaf] at hm
    · intro e he; simp [shEvLeaf] at he
    · intro e he; simp [jLeaf] at he
  | _ => simp [lz4Leaf] at hlz

end Leaf

section Prog
variable {hw : Hw} {I O : UInt64 → Prop}

set_option maxHeartbeats 4000000 in
/-- **The warp's run of a block's thread**: it clears the table, passes the
    barrier and runs the body as the model does from the block's start
    (`startWS`), so it makes only what lanes need, each lane's byte stores
    are its share of the model's, and its shared stores are apart within
    each epoch. -/
theorem prog_facts {numBlocks inStride outStride lenOff hashLog cta f : Nat} {inP outA : UInt64} {G : Array UInt8}
    (hcta : cta < numBlocks) (hnb : numBlocks < 2 ^ 63) (h5 : 5 ≤ hashLog) (hh : 2 ^ hashLog * 2 ≤ 2 ^ 36)
    (hf : f ≤ 2 ^ 64)
    (hflags : ∀ r, r ∈ ["inBase", "outBase", "tbl", "op", "searchPos", "litAnchor"] →
      (flagRegs (compressorBodyWStmt inStride lenOff hashLog)).contains r = false)
    (hIdec : ∀ a, I a → ∃ r o, decodeAddr a = some (r, o)) (hOdec : ∀ a, O a → ∃ r o, decodeAddr a = some (r, o))
    (hok : okOf I O (okLZ4 I) f (compressorBodyWStmt inStride lenOff hashLog)
      (startWS numBlocks inStride outStride hashLog cta inP G)) :
    OkL hw cta (2 ^ hashLog * 2) (fun b => [inP, outA].getD b 0) I O 0
        (compressorS numBlocks inStride outStride lenOff hashLog) (blockWS inP outA G) true ∧
      (∀ l : Lane, byteSt (LaneWarp.trace hw cta (fun _ _ => 0)
        (storeLeaf cta (fun _ _ => 0) l (fun b => [inP, outA].getD b 0) (addrOf .arena 0)) 0
        (compressorS numBlocks inStride outStride lenOff hashLog) (blockWS inP outA G)) =
        lanePart l.val (gstOf f (compressorBodyWStmt inStride lenOff hashLog)
          (startWS numBlocks inStride outStride hashLog cta inP G))) ∧
      ShOk cta (LaneWarp.trace hw cta (fun _ _ => 0) shEvLeaf 0
        (compressorS numBlocks inStride outStride lenOff hashLog) (blockWS inP outA G)) := by
  let body := compressorBodyWStmt inStride lenOff hashLog
  let ns := bodyNames inStride lenOff hashLog
  let T := ns.length
  let S := 2 ^ hashLog * 2
  let ptr : Buf → UInt64 := fun b => [inP, outA].getD b 0
  let ws0 := blockWS inP outA G
  have mem : ∀ n, n ∈ ["inBase", "outBase", "tbl", "op", "searchPos", "litAnchor"] → n ∈ ns :=
    fun n h => mem_dedup.mpr (List.mem_append_left _ h)
  have mIB := mem "inBase" (by simp)
  have mOB := mem "outBase" (by simp)
  have mTb := mem "tbl" (by simp)
  have lIB := idxOf_lt mIB
  have lOB := idxOf_lt mOB
  have lTb := idxOf_lt mTb
  have dOT : ns.idxOf "outBase" ≠ ns.idxOf "tbl" := fun e => by have := idxOf_inj mOB mTb e; simp at this
  have dIT : ns.idxOf "inBase" ≠ ns.idxOf "tbl" := fun e => by have := idxOf_inj mIB mTb e; simp at this
  have dIO : ns.idxOf "inBase" ≠ ns.idxOf "outBase" := fun e => by have := idxOf_inj mIB mOB e; simp at this
  let P1 : List EWStmt := [.setI (laneR T) .lane,
    .setI (ns.idxOf "inBase") (.bin .add (.ptr 0) (.bin .mul .cta (.lit inStride))),
    .setI (ns.idxOf "outBase") (.bin .add (.bin .add (.ptr 0) (.lit (numBlocks * inStride + copySlack)))
      (.bin .mul .cta (.lit outStride))),
    .setI (ns.idxOf "tbl") (.lit 0),
    .setI (T + 40) (.reg (laneR T))]
  let A : IExp := .bin .band (.bin .add (.reg (ns.idxOf "tbl")) (.bin .shl (.reg (T + 40)) (.lit 1)))
    (.lit (2 * 2 ^ hashLog - 1))
  let loop : EWStmt := .iwhile (2 ^ 64) (.cmp .ult (.reg (T + 40)) (.lit (2 ^ hashLog))) [T + 40]
    (.seq (.stH A (.lit 0) (.lit 1)) (.setI (T + 40) (.bin .add (.reg (T + 40)) (.lit 32))))
  let main := ofWn ns (flagRegs body) body
  have hs : compressorS numBlocks inStride outStride lenOff hashLog =
      .iif (.cmp .ult .cta (.lit numBlocks)) (.locals (List.range (T + 42)) (seqs (P1 ++ [loop, .barrier, main]))) .skip :=
    rfl
  have hc : (IExp.cmp .ult .cta (.lit numBlocks)).eval ws0.int cta lane0 ≠ 0 := by
    have : UInt64.ofNat cta < UInt64.ofNat numBlocks := by
      rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' (show cta < 2 ^ 64 by omega),
        UInt64.toNat_ofNat_of_lt' (show numBlocks < 2 ^ 64 by omega)]
      exact hcta
    simp [IExp.eval, ICmp.run, this]
  -- the prologue
  let IBv := inP + UInt64.ofNat cta * UInt64.ofNat inStride
  let OBv := inP + UInt64.ofNat (numBlocks * inStride + copySlack) + UInt64.ofNat cta * UInt64.ofNat outStride
  let Z5 := LaneWarp.run hw cta (fun _ _ => 0) (seqs P1) ws0 0
  have hLT : laneR T = T + 41 := rfl
  have z_ci : ∀ l : Lane, Z5.int.regs (T + 40) l = UInt64.ofNat (32 * 0 + l.val) := fun l => by
    simp (disch := omega) only [Z5, P1, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne,
      IExp.eval, laneR, Nat.mul_zero, Nat.zero_add]
  have z_tb : ∀ l : Lane, Z5.int.regs (ns.idxOf "tbl") l = 0 := fun l => by
    simp (disch := omega) only [Z5, P1, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne,
      IExp.eval]; rfl
  have z_ob : ∀ l : Lane, Z5.int.regs (ns.idxOf "outBase") l = OBv := fun l => by
    simp (disch := omega) only [Z5, P1, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq,
      set_regs_ne _ _ dOT, set_regs_ne _ _ (show ns.idxOf "outBase" ≠ T + 40 by omega)]; rfl
  have z_ib : ∀ l : Lane, Z5.int.regs (ns.idxOf "inBase") l = IBv := fun l => by
    simp (disch := omega) only [Z5, P1, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq,
      set_regs_ne _ _ dIT, set_regs_ne _ _ dIO, set_regs_ne _ _ (show ns.idxOf "inBase" ≠ T + 40 by omega)]; rfl
  have z_ln : ∀ l : Lane, Z5.int.regs (laneR T) l = UInt64.ofNat l.val := fun l => by
    simp (disch := omega) only [Z5, P1, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne, laneR]
    rfl
  have z_o : ∀ r, r ≠ T + 40 → r ≠ ns.idxOf "tbl" → r ≠ ns.idxOf "outBase" → r ≠ ns.idxOf "inBase" → r ≠ laneR T →
      ∀ l, Z5.int.regs r l = 0 := fun r h1 h2 h3 h4 h5' l => by
    simp only [Z5, P1, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_ne _ _ h1, set_regs_ne _ _ h2,
      set_regs_ne _ _ h3, set_regs_ne _ _ h4, set_regs_ne _ _ h5']
    rfl
  have z_m := regOnly_mem hw cta (fun _ _ => 0) (seqs P1) rfl ws0 0
  have hS36 : S ≤ 2 ^ 36 := hh
  have hcl := clear_loop (hw := hw) (cta := cta) (S := S) (ptr := ptr) (I := I) (O := O) (T + 40) (ns.idxOf "tbl")
    hashLog (by omega) h5 (by omega) hS36 Z5 z_tb 0 (2 ^ 64) 0 Z5 (by omega) (by omega) z_ci (fun _ _ => rfl)
    (fun k => by rw [z_m.2]; rfl) rfl true
  obtain ⟨cOk, ⟨Zc, hZc, cRegs, cSmem, cGmem⟩, cLen, cEv⟩ := hcl
  -- the body starts where the model does
  let W0 := startWS numBlocks inStride outStride hashLog cta inP G
  have kIn : c64 inStride = UInt64.ofNat inStride := c64_ofNat _
  have kOut : c64 outStride = UInt64.ofNat outStride := c64_ofNat _
  have kOff : c64 ((numBlocks : Int) * inStride + copySlack) = UInt64.ofNat (numBlocks * inStride + copySlack) := by
    rw [← c64_ofNat]; push_cast; rfl
  have hn : namesOf body ⊆ ns := fun n h => mem_dedup.mpr (List.mem_append_right _ h)
  have hSim : SimW ns I Zc W0 := by
    refine ⟨fun n hn' l => ?_, fun l => ?_, fun a _ => ?_, fun k hk => ?_⟩
    · have hr := idxOf_lt hn'
      refine (congrFun (cRegs _ (by omega)) l).trans ?_
      simp only [W0, startWS, regs0]
      by_cases e1 : n = "inBase"
      · subst e1; rw [z_ib, if_pos rfl, kIn]
      by_cases e2 : n = "outBase"
      · subst e2; rw [z_ob, if_neg (by decide), if_pos rfl, kOut, kOff]
      rw [if_neg e1, if_neg e2]
      by_cases e3 : n = "tbl"
      · subst e3; exact z_tb l
      exact z_o _ (by omega) (fun h => e3 (idxOf_inj hn' mTb h)) (fun h => e2 (idxOf_inj hn' mOB h))
        (fun h => e1 (idxOf_inj hn' mIB h)) (by simp only [laneR]; omega) l
    · exact (congrFun (cRegs _ (by simp only [laneR]; omega)) l).trans (z_ln l)
    · exact (congrFun cGmem _).trans ((congrFun z_m.1 _).trans rfl)
    · refine (cSmem k).trans ?_
      simp only [W0, startWS, clearMem, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
      split <;> rfl
  have hB : BitsW (flagRegs body) W0 := by
    intro r hr
    have e1 : r ≠ "inBase" := fun e => by subst e; rw [hflags _ (by simp)] at hr; cases hr
    have e2 : r ≠ "outBase" := fun e => by subst e; rw [hflags _ (by simp)] at hr; cases hr
    simp [W0, startWS, regs0, e1, e2]
  have hsz : W0.smem.size ≤ S := by simp [W0, startWS, clearMem, S]
  obtain ⟨bOk, bSim, bLast, bBytes, bPaired⟩ := sim (hw := hw) (cta := cta) (S := S) (ptr := ptr) (I := I) (O := O)
    (ns := ns) (fl := flagRegs body) hIdec hOdec hS36 body hn (wf_flagRegs body) f W0 Zc 0 hf hB hsz hok hSim
  -- the thread's run: the prologue, the clear, the barrier, the body
  have htr : ∀ {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α),
      LaneWarp.trace hw cta (fun _ _ => 0) leaf 0 (compressorS numBlocks inStride outStride lenOff hashLog) ws0 =
        LaneWarp.trace hw cta (fun _ _ => 0) leaf 0 (seqs P1) ws0 ++
          (LaneWarp.trace hw cta (fun _ _ => 0) leaf 0 loop Z5 ++
            (LaneWarp.trace hw cta (fun _ _ => 0) leaf 0 .barrier Zc ++
              LaneWarp.trace hw cta (fun _ _ => 0) leaf 0 main Zc)) := by
    intro α leaf
    rw [hs, trace_iif hw cta _ leaf _ _ _ ws0 0 hc, trace_locals,
      trace_seqs_append hw cta _ leaf [loop, .barrier, main] (by simp) P1 (by simp [P1])]
    rw [show seqs [loop, .barrier, main] = .seq loop (.seq .barrier main) from rfl, trace_seq, trace_seq,
      show LaneWarp.run hw cta (fun _ _ => 0) (seqs P1) ws0 0 = Z5 from rfl, hZc, run_barrier]
  have hq1 := fun l => quiet_trace hw cta (fun _ _ => 0) l ptr (addrOf .arena 0) (seqs P1) rfl 0 ws0
  refine ⟨?_, fun l => ?_, ?_⟩
  · -- what lanes need
    rw [hs]
    refine ⟨fun l => Iff.rfl, ?_⟩
    rw [if_pos hc]
    refine okW_seqs_append hw cta _ S ptr _ _ _ _ [loop, .barrier, main] (by simp) P1 (by simp [P1]) ws0 0 true
      ⟨trivial, trivial, trivial, trivial, trivial⟩ ?_
    rw [(hq1 lane0).2.1]
    refine ⟨cOk, trivial, ?_⟩
    rw [hZc, run_barrier, show LaneWarp.trace hw cta (fun _ _ => 0) shLeaf 0 .barrier Zc = [true] from rfl,
      show ∀ x, lastOr x [true] = true from fun _ => rfl]
    exact bOk
  · -- each lane's byte stores
    rw [htr, byteSt_append, byteSt_append, byteSt_append, (hq1 l).1]
    have hloop : byteSt (LaneWarp.trace hw cta (fun _ _ => 0) (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0)) 0
        loop Z5) = [] := by
      refine byteSt_eq_nil _ (fun a n v hm => ?_)
      obtain ⟨i', s', ws', hQ, hm'⟩ := mem_trace hw cta (fun _ _ => 0) _
        (fun _ t => (∃ a v g, t = EWStmt.stH a v g) ∨ ∃ d e, t = EWStmt.setI d e) _ loop 0 Z5
        ⟨Or.inl ⟨_, _, _, rfl⟩, Or.inr ⟨_, _, rfl⟩⟩ hm
      rcases hQ with ⟨a', v', g', rfl⟩ | ⟨d, e, rfl⟩
      · simp only [storeLeaf] at hm'
        split at hm'
        · simp at hm'; omega
        · cases hm'
      · cases hm'
    rw [hloop, show byteSt (LaneWarp.trace hw cta (fun _ _ => 0) (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0))
      0 .barrier Zc) = [] from rfl, show byteSt ([] : List (Option (UInt64 × Nat × UInt64))) = [] from rfl,
      List.nil_append, List.nil_append, List.nil_append]
    exact bBytes l
  · -- shared memory's epochs
    rw [htr, (hq1 lane0).2.2, List.nil_append,
      show LaneWarp.trace hw cta (fun _ _ => 0) shEvLeaf 0 .barrier Zc = [.bar Zc] from rfl, List.singleton_append]
    refine shOk_of cta _ _ Zc (fun e he => ?_) (fun x i j l l' e e' hi hj hx hx' => ?_) bPaired
    · obtain ⟨j, hj⟩ := List.mem_iff_getElem?.mp he
      obtain ⟨Xj, rfl, -, -⟩ := cEv j e hj
      rfl
    · obtain ⟨Xi, rfl, hXi, hti⟩ := cEv i e hi
      obtain ⟨Xj, rfl, hXj, htj⟩ := cEv j e' hj
      have hil := (List.getElem?_eq_some_iff.mp hi).1
      have hjl := (List.getElem?_eq_some_iff.mp hj).1
      rw [cLen] at hil hjl
      have hl : l.val < 32 := l.isLt
      have hl' : l'.val < 32 := l'.isLt
      have hN : 2 ^ hashLog < 2 ^ 36 := by omega
      have eA := clearA_eval (T + 40) (ns.idxOf "tbl") hashLog Xi.int cta l (32 * (0 + i) + l.val) (by
        have : i < 2 ^ hashLog / 32 := by omega
        omega) hN (hXi l) (hti l)
      have eA' := clearA_eval (T + 40) (ns.idxOf "tbl") hashLog Xj.int cta l' (32 * (0 + j) + l'.val) (by
        have : j < 2 ^ hashLog / 32 := by omega
        omega) hN (hXj l') (htj l')
      have hw1 : ∀ (X : WSt F32) (l : Lane), wrF cta l (.st X A (.lit 0) (.lit 1)) = [((A.eval X.int cta l).toNat, 0)] :=
        fun X l => by simp only [wrF]; rw [if_pos (by simp only [IExp.eval]; decide)]; rfl
      rw [hw1, eA] at hx
      rw [hw1, eA'] at hx'
      have h1 := byteAt_ne hx
      have h2 := byteAt_ne hx'
      left; omega

end Prog

/-! ## A block runs -/

set_option maxHeartbeats 4000000 in
/-- **A block of the compressor runs**: its 32 threads, each its lane of the
    warp's run (`lane_warp`) answered as the warp's run answers, each ends and
    the laws give those answers back (`block_laws`); every byte outside shared
    memory is its lanes' share of the model's stores, every storable byte stays
    storable and the NaN rule is the hardware's — given that the model's run of
    the body from the block's start (`startWS`) makes only the accesses `okOf`
    allows, `I` holds the input bytes and `O`'s bytes may be stored. -/
theorem block_run (hw : Hw) (hw32 : ∀ k v, (hw.approx k v).toNat < 2 ^ 32) (name : String)
    {numBlocks inStride outStride lenOff hashLog cta f : Nat} {inP outA : UInt64} {I O : UInt64 → Prop}
    {G : Array UInt8} (w : World)
    (hcta : cta < numBlocks) (hc32 : cta * 32 + 32 < 2 ^ 32) (hnb : numBlocks < 2 ^ 63) (h5 : 5 ≤ hashLog)
    (hh : 2 * 2 ^ hashLog + 64 < 2 ^ 36) (hf : f ≤ 2 ^ 64)
    (hfine : Fine (emitGo (fnBody (idParams ++ [.i64, .i64])
      (ML.Body.body 2 (compressorS numBlocks inStride outStride lenOff hashLog)))
      ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2)
    (hflags : ∀ r, r ∈ ["inBase", "outBase", "tbl", "op", "searchPos", "litAnchor"] →
      (flagRegs (compressorBodyWStmt inStride lenOff hashLog)).contains r = false)
    (hIna : ∀ a, I a → NotArena a) (hOna : ∀ a, O a → NotArena a)
    (hok : okOf I O (okLZ4 I) f (compressorBodyWStmt inStride lenOff hashLog)
      (startWS numBlocks inStride outStride hashLog cta inP G))
    (hI : ∀ a, I a → w.mem.load a 1 = some (G.getD a.toNat 0).toUInt64)
    (hO : ∀ a, O a → Writable w.mem a 1) (hnan : w.mem.nan = hw.nan) :
    ∃ u, BlockRuns (compressorKernel name numBlocks inStride outStride lenOff hashLog) cta 32
        [.sc .i64 inP, .sc .i64 outA] hw w u ∧
      (∀ a, NotArena a → u.mem.load a 1 = replay ((List.range 32).flatMap fun t =>
        lanePart t (gstOf f (compressorBodyWStmt inStride lenOff hashLog)
          (startWS numBlocks inStride outStride hashLog cta inP G))) a (w.mem.load a 1)) ∧
      (∀ a, NotArena a → Writable w.mem a 1 → Writable u.mem a 1) ∧ u.mem.nan = hw.nan := by
  let s := compressorS numBlocks inStride outStride lenOff hashLog
  let ws0 := blockWS inP outA G
  let S := 2 ^ hashLog * 2
  let ptr : Buf → UInt64 := fun b => [inP, outA].getD b 0
  let W0 := startWS numBlocks inStride outStride hashLog cta inP G
  let body := compressorBodyWStmt inStride lenOff hashLog
  have hS : S ≤ 2 ^ 36 := by omega
  have hIdec : ∀ a, I a → ∃ r o, decodeAddr a = some (r, o) := fun a ha => load_decode (hI a ha)
  have hOdec : ∀ a, O a → ∃ r o, decodeAddr a = some (r, o) := fun a ha => by
    obtain ⟨r, o, hd, -⟩ := hO a ha; exact ⟨r, o, hd⟩
  obtain ⟨hOk, hBytes, hShOk⟩ := prog_facts (hw := hw) (I := I) (O := O) (outA := outA) hcta hnb h5 (by omega) hf
    hflags hIdec hOdec hok
  -- what each leaf gives
  have hleaf : ∀ x ∈ LaneWarp.trace hw cta (fun _ _ => 0) allLeaf 0 s ws0,
      (∃ cl, OkL hw cta S ptr I O x.1 x.2.1 x.2.2 cl) ∧ lz4Leaf x.2.1 = true := fun x hx => by
    refine ⟨okW_leaves hw cta _ S ptr _ _ _ _ s 0 ws0 true hOk x hx, ?_⟩
    obtain ⟨i', s', ws', hQ, hm⟩ := mem_trace hw cta (fun _ _ => 0) allLeaf (fun _ t => lz4Leaf t = true) x s 0 ws0
      (allAt_compressorS _ numBlocks inStride outStride lenOff hashLog) hx
    simp only [allLeaf, List.mem_singleton] at hm
    subst hm; exact hQ
  have hLF := fun l x hx => leaf_facts (hw := hw) (cta := cta) (ptr := ptr) (I := I) (O := O) hS l x (hleaf x hx).1
    (hleaf x hx).2
  have hStOk : ∀ l, ∀ e ∈ LaneWarp.trace hw cta (fun _ _ => 0) (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0)) 0
      s ws0, ∀ a n v, e = some (a, n, v) → n = 1 ∨ ∃ off, decodeAddr a = some (.arena, off) := fun l =>
    trace_leafwise hw cta _ _ _ s 0 ws0 (fun x hx e he a n v hev => (hLF l x hx).1 a n v (hev ▸ he))
  have hTag := fun l => lane_tags hw cta (fun _ _ => 0) l ptr s 0 ws0 (fun x hx => (hLF l x hx).2.1)
  have hin : ∀ e ∈ LaneWarp.trace hw cta (fun _ _ => 0) shEvLeaf 0 s ws0, ∀ l, ∀ w ∈ wrF cta l e, w.1 + 2 ≤ S :=
    trace_leafwise hw cta _ _ _ s 0 ws0 (fun x hx e he => (hLF lane0 x hx).2.2.1 e he)
  have hJ := trace_leafwise hw cta (fun _ _ => 0) jLeaf _ s 0 ws0 (fun x hx e he => (hLF lane0 x hx).2.2.2 e he)
  -- the block's threads, in order, each its lane of the warp
  let A0 : ByteArray := ByteArray.mk (Array.replicate S 0)
  have hA0 : A0 = bytesOf S ws0.int.smem := by
    simp only [A0, bytesOf, ws0, blockWS]
    congr 1
    apply Array.ext (by simp)
    intro k h1 h2
    simp
  let T : Nat → List Answer := fun t => LaneWarp.trace hw cta (fun _ _ => 0) (ansLeaf cta S (laneOf t)) 0 s ws0
  let Vw : World → World := fun W => { W with mem := { W.mem with arena := A0 } }
  let xc : Nat → World → LCtx := fun t W => threadCtx cta t [inP, outA] (T t) W.obs.length hw
  let Inv : Nat → World → Prop := fun t W =>
    (∀ a, I a → W.mem.load a 1 = some (G.getD a.toNat 0).toUInt64) ∧
    (∀ a, NotArena a → Writable w.mem a 1 → Writable W.mem a 1) ∧
    (∀ a, NotArena a → W.mem.load a 1 =
      replay ((List.range t).flatMap fun t' => lanePart t' (gstOf f body W0)) a (w.mem.load a 1)) ∧
    W.mem.nan = hw.nan
  have hthr : ∀ t, t < 32 → ∀ W, Inv t W → ∃ li', laneI (xc t W) s (LI.init (Vw W)) = some li' ∧ Inv (t + 1) li'.w ∧
      collCalls li'.w W.obs.length = (LaneWarp.trace hw cta (fun _ _ => 0) jLeaf 0 s ws0).map (callOne cta (laneOf t)) ∧
      sharedStores li'.w W.obs.length =
        tagE cta (addrOf .arena 0) (laneOf t) 0 (LaneWarp.trace hw cta (fun _ _ => 0) shEvLeaf 0 s ws0) := by
    intro t ht W hW
    obtain ⟨hWI, hWw, hWm, hWn⟩ := hW
    have hlt : (laneOf t).val = t := Nat.mod_eq_of_lt ht
    have hpre : LanePre hw cta (fun _ _ => 0) S (fun _ _ _ => False) (fun _ a => I a) (fun _ _ => False)
        (fun _ a n => n = 1 ∧ O a) (laneOf t) (xc t W) 0 s ws0 true (LI.init (Vw W)) [] := by
      refine ⟨rfl, ?_, ?_, by omega, rfl, rfl, hOk, ?_, ?_, fun _ _ _ h => h.elim, fun _ a ha => hIna a ha,
        fun _ _ h => h.elim⟩
      · show (UInt32.ofNat t).toUInt64 = _
        rw [toUInt64_ofNat (by omega), hlt]
      · show (UInt32.ofNat cta).toUInt64 = _
        rw [toUInt64_ofNat (by omega)]
      · refine ⟨fun _ => rfl, fun _ => rfl, fun _ _ h => h.elim, fun a ha => ?_, fun _ _ h => h.elim,
          fun a n hn => ?_, by show (Array.replicate S (0 : UInt8)).size = S; simp, fun _ => hA0, hWn, Nat.le_refl _, fun _ => rfl,
          ⟨fun _ _ => show (0 : UInt64).toNat < 2 ^ 32 by decide, fun _ _ => show (0 : UInt64).toNat < 2 ^ 32 by decide,
            fun _ => show (0 : UInt64).toNat < 2 ^ 32 by decide⟩⟩
        · show ({ W.mem with arena := A0 } : Mem).load a 1 = _
          rw [load_set_arena (hIna a ha), hWI a ha]; rfl
        · obtain ⟨rfl, ha⟩ := hn
          exact writable_arena (hWw a (hOna a ha) (hO a ha)) (hOna a ha) A0
      · show (T t).drop (collCalls (Vw W) (Vw W).obs.length).length = T t ++ []
        rw [collCalls_self, List.append_nil]; rfl
    obtain ⟨li', hl, hrel, hcc, hst, -⟩ := lane_warp hw cta (fun _ _ => 0) S _ _ _ _ (laneOf t) hw32 s (xc t W) 0 ws0
      true (LI.init (Vw W)) [] hpre
    have hstart : (xc t W).start = (Vw W).obs.length := rfl
    rw [hstart, show (LI.init (Vw W)).w = Vw W from rfl, collCalls_self, List.nil_append] at hcc
    rw [hstart, show (LI.init (Vw W)).w = Vw W from rfl, storesOf_self, List.nil_append] at hst
    obtain ⟨L, hL, htrk⟩ := laneI_tracks s (xc t W) _ _ hl
    have hL2 : li'.w.obs = L ++ (Vw W).obs := hL
    have hsok : StOk L := stOk_of_storesOf hL2 (fun a n v hm => by
      rw [hst] at hm; exact hStOk (laneOf t) _ hm a n v rfl)
    obtain ⟨L', hL', hmem⟩ := htrk hsok
    have hL'2 : li'.w.obs = L' ++ (Vw W).obs := hL'
    have hbytes : stores1 L'.reverse = lanePart t (gstOf f body W0) := by
      rw [← byteSt_storesOf hL'2, hst]
      exact (hBytes (laneOf t)).trans (by rw [hlt])
    refine ⟨li', hl, ⟨fun a ha => ?_, fun a ha hwr => ?_, fun a ha => ?_, hrel.nan⟩, ?_, ?_⟩
    · rw [hmem a (hIna a ha), hbytes]
      show replay _ a (({ W.mem with arena := A0 } : Mem).load a 1) = _
      rw [load_set_arena (hIna a ha), hWI a ha]
      refine replay_notin a _ _ (fun x hx e => ?_)
      obtain ⟨who, hm⟩ := lanePart_mem hx
      exact (gstOf_out body f W0 hok _ hm).1 (by rw [← e] at ha; exact ha)
    · exact laneI_writable s (xc t W) _ _ hl a ha (writable_arena (hWw a ha hwr) ha A0)
    · rw [hmem a ha, hbytes]
      show replay _ a (({ W.mem with arena := A0 } : Mem).load a 1) = _
      rw [load_set_arena ha, hWm a ha, List.range_succ, List.flatMap_append, replay_append]
      simp
    · rw [hcc, trace_call]
    · rw [sharedStores_tag, hst]; exact (hTag (laneOf t)).2 0
  let step : Nat → World → World := fun t W => ((laneI (xc t W) s (LI.init (Vw W))).map LI.w).getD W
  let Ws : Nat → World := fun t => Nat.rec (motive := fun _ => World) (Vw w) (fun t W => step t W) t
  have hWsS : ∀ t, Ws (t + 1) = step t (Ws t) := fun _ => rfl
  have hInv : ∀ t, t ≤ 32 → Inv t (Ws t) := by
    intro t
    induction t with
    | zero =>
      intro _
      refine ⟨fun a ha => ?_, fun a ha hwr => writable_arena hwr ha A0, fun a ha => ?_, hnan⟩
      · show ({ w.mem with arena := A0 } : Mem).load a 1 = _
        rw [load_set_arena (hIna a ha)]; exact hI a ha
      · show ({ w.mem with arena := A0 } : Mem).load a 1 = _
        rw [load_set_arena ha]; rfl
    | succ t ih =>
      intro ht
      obtain ⟨li', hl, hI', -, -⟩ := hthr t (by omega) (Ws t) (ih (by omega))
      rw [hWsS]
      simp only [step, hl, Option.map_some, Option.getD_some]
      exact hI'
  have hthreads : ∀ t, t < 32 → ∃ steps, thread (compressorKernel name numBlocks inStride outStride lenOff hashLog).code
      (T t) (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t) { Ws t with mem := { (Ws t).mem with arena := A0 } } hw
      steps = .ok (Ws (t + 1)) := by
    intro t ht
    obtain ⟨li', hl, -, -, -⟩ := hthr t ht (Ws t) (hInv t (by omega))
    obtain ⟨steps, u, hu, he⟩ := compressor_thread hw hw32 (words_ans hw cta _ S (laneOf t) s 0 ws0) name hfine
      (w0 := Vw (Ws t)) hl
    refine ⟨steps, ?_⟩
    rw [show ({ Ws t with mem := { (Ws t).mem with arena := A0 } } : World) = Vw (Ws t) from rfl, hu, he, hWsS]
    simp only [step, hl, Option.map_some, Option.getD_some]
  obtain ⟨S0, hS0⟩ := pass_worlds (k := compressorKernel name numBlocks inStride outStride lenOff hashLog) (hw := hw)
    (smem0 := A0) (args := threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA]) (ans := T) Ws 32 hthreads
  obtain ⟨c, st, hfold, hc⟩ := hS0 S0 (Nat.le_refl _)
  -- the laws give the warp's answers back
  have hc' : ∀ t, t < 32 → c t = (LaneWarp.trace hw cta (fun _ _ => 0) jLeaf 0 s ws0).map (callOne cta (laneOf t)) := by
    intro t ht
    obtain ⟨li', hl, -, hcc, -⟩ := hthr t ht (Ws t) (hInv t (by omega))
    rw [(hc t ht).1, hWsS]
    simp only [step, hl, Option.map_some, Option.getD_some]
    exact hcc
  have hst' : ∀ t, t < 32 → st t =
      tagE cta (addrOf .arena 0) (laneOf t) 0 (LaneWarp.trace hw cta (fun _ _ => 0) shEvLeaf 0 s ws0) := by
    intro t ht
    obtain ⟨li', hl, -, -, hss⟩ := hthr t ht (Ws t) (hInv t (by omega))
    rw [(hc t ht).2, hWsS]
    simp only [step, hl, Option.map_some, Option.getD_some]
    exact hss
  have hlaws := block_laws hw cta (fun _ _ => 0) S hS s ws0 c st hJ hc' hst' hin hShOk
  have hans_t : ∀ t, t < 32 → answersOf c (snapshots 32 A0 c st) A0 t = .ok (T t) := by
    intro t ht
    rw [hA0, hlaws t ht]
    show _ = Except.ok (LaneWarp.trace hw cta (fun _ _ => 0) (ansLeaf cta S (laneOf t)) 0 s ws0)
    rw [trace_ans]
  have hans : Kernel.answers 32 A0 c st = .ok fun t => ((List.range 32).map T).getD t [] := by
    unfold Kernel.answers
    rw [mapM_range_ok _ T 32 hans_t]
    rfl
  have hIN := hInv 32 (Nat.le_refl _)
  refine ⟨{ Ws 32 with mem := { (Ws 32).mem with arena := w.mem.arena } },
    ⟨S0, T, Ws 32, c, st, _, hfold, hans, fun t ht => ?_, rfl⟩, fun a ha => ?_,
    fun a ha hwr => writable_arena (hIN.2.1 a ha hwr) ha _, hIN.2.2.2⟩
  · simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
    rw [List.getElem?_range ht]
    rfl
  · show ({ (Ws 32).mem with arena := w.mem.arena } : Mem).load a 1 = _
    rw [load_set_arena ha]
    exact hIN.2.2.1 a ha

end Lz4Code
