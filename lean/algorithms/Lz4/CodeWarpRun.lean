module
public import Lz4.CodeExtend
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeWarpRun` — a warp's lanes run a warp statement with the warp

One lane, answered from the warp's tape (`trace lz4Ans`), runs a warp
statement as the warp does: it finishes agreeing with the warp's run, its byte
stores its share of the warp's, its collectives and shared stores the warp's
traces for it (`lane_run`, from `lane_lock` with the cooperative statements'
`lane_window` and `lane_extend`). The laws, from what all 32 lanes finally
passed, give the tape back (`warp_laws`, with `window_laws` and
`extend_laws`). What the warp's run must allow is `okOf` with the cooperative
statements' own conditions (`okLZ4`).
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel

/-- What a cooperative statement needs of the warp's state. -/
def okLZ4 (I : UInt64 → Prop) : WStmt → WState → Prop
  | .coopWindow found p0 cand0 sp inStride searchLim hashLog tbl, ws =>
      searchLim + 4 ≤ inStride ∧ hashLog ≤ 32 ∧ (∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j)) ∧
      (ws.regs "inBase").toNat + inStride < 2 ^ 64 ∧ (ws.regs sp).toNat + 32 < 2 ^ 64 ∧
      ws.regs "tbl" = UInt64.ofNat tbl ∧ tbl + 2 * 2 ^ hashLog + 2 < 2 ^ 36 ∧
      found ≠ p0 ∧ found ≠ cand0 ∧ p0 ≠ cand0 ∧ tbl + 2 * 2 ^ hashLog ≤ ws.smem.size
  | .coopExtendStep _ p0 cand0 ml inStride endCap, ws =>
      ws.regs "ecR" = UInt64.ofNat endCap ∧ ws.regs "ec1" = UInt64.ofNat (endCap - 1) ∧ 1 ≤ endCap ∧
      endCap ≤ inStride ∧ (ws.regs cand0).toNat < (ws.regs p0).toNat ∧
      (ws.regs p0).toNat + (ws.regs ml).toNat + 32 < 2 ^ 64 ∧
      (∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j)) ∧ (ws.regs "inBase").toNat + inStride < 2 ^ 64 ∧
      (ws.regs p0).toNat < endCap
  | _, _ => True

/-- What the warp's cooperative statements answer their collectives. -/
def lz4Ans : WStmt → WState → List Answer
  | .coopWindow _ _ _ sp inStride searchLim hashLog tbl, ws => winAns ws sp inStride searchLim hashLog tbl
  | .coopExtendStep _ p0 cand0 ml inStride endCap, ws => extAns ws p0 cand0 ml inStride endCap
  | _, _ => []

/-- Shared memory after the first `m` phases reads only stores tagged below `m`. -/
theorem snapMem_congr {all all' : List (Nat × UInt64 × Nat × UInt64)} {smem0 : ByteArray} :
    ∀ m, (∀ b, b < m → all.filter (·.1 = b) = all'.filter (·.1 = b)) → snapMem all smem0 m = snapMem all' smem0 m
  | 0, _ => rfl
  | m + 1, h => by
      simp only [snapMem]
      rw [phase_filter, phase_filter, h m (by omega), snapMem_congr m (fun b hb => h b (by omega))]

/-- Stores tagged from `NB` on leave the phases below it. -/
theorem stores_filter {S S' : Nat → List (Nat × UInt64 × Nat × UInt64)} {NB : Nat}
    (h : ∀ t, t < 32 → ∃ E, S' t = S t ++ E ∧ ∀ x ∈ E, NB ≤ x.1) {b : Nat} (hb : b < NB) :
    ((List.range 32).flatMap S).filter (·.1 = b) = ((List.range 32).flatMap S').filter (·.1 = b) := by
  rw [List.filter_flatMap, List.filter_flatMap]
  exact flatMap_congr_mem fun t ht => by
    obtain ⟨E, hE, hx⟩ := h t (List.mem_range.mp ht)
    have hE0 : E.filter (fun x => decide (x.1 = b)) = [] := List.filter_eq_nil_iff.mpr fun x hx' => by
      have := hx x hx'; simp only [decide_eq_true_eq]; omega
    rw [hE, List.filter_append, hE0, List.append_nil]

/-- **What the laws gave back they give back still** as the lanes go on: their
    collectives extended, their new shared stores tagged from `NB` on, every
    lane past the first `K` collectives and `NB` barriers. -/
theorem settled_mono {ans : Nat → List Answer} {smem0 : ByteArray} {F F' : Nat → List (Intr × List V)}
    {S S' : Nat → List (Nat × UInt64 × Nat × UInt64)} {K NB : Nat}
    (hF : ∀ t, t < 32 → ∃ E, F' t = F t ++ E) (hF32 : ∀ t, 32 ≤ t → F' t = F t)
    (hK : ∀ t, t < 32 → (F t).length = K) (hNB : ∀ t, t < 32 → (F t).countP (·.1 == .barrier) = NB)
    (hS : ∀ t, t < 32 → ∃ E, S' t = S t ++ E ∧ ∀ x ∈ E, NB ≤ x.1)
    (hset : ∀ l, l < 32 → ∀ k, k < K → ∃ a, (ans l)[k]? = some a ∧ LawAt F (snapshots 32 smem0 F S) smem0 l k a) :
    ∀ l, l < 32 → ∀ k, k < K → ∃ a, (ans l)[k]? = some a ∧ LawAt F' (snapshots 32 smem0 F' S') smem0 l k a := by
  intro l hl k hk
  obtain ⟨a, ha, hlaw⟩ := hset l hl k hk
  obtain ⟨El, hEl⟩ := hF l hl
  refine ⟨a, ha, hlaw.mono (fun t => ?_) ?_ (fun x hx hb => ?_)⟩
  · rcases Nat.lt_or_ge t 32 with ht | ht
    · obtain ⟨E, hE⟩ := hF t ht
      rw [hE, List.getElem?_append_left (by rw [hK t ht]; exact hk)]
    · rw [hF32 t ht]
  · rw [hEl, List.take_append_of_le_length (by rw [hK l hl]; omega)]
  · -- the barrier's phase is one every lane has passed
    have hnb : ((F l).take k).countP (·.1 == .barrier) < NB := by
      have e : (F l).take (k + 1) = (F l).take k ++ [x] := by
        rw [List.take_succ, hx]; rfl
      have hle : ((F l).take (k + 1)).countP (·.1 == .barrier) ≤ (F l).countP (·.1 == .barrier) :=
        (List.take_sublist _ _).countP_le
      rw [e, List.countP_append, hNB l hl] at hle
      simp [hb, show (Intr.barrier == Intr.barrier) = true by decide] at hle
      omega
    obtain ⟨E0, hE0⟩ := hF 0 (by decide)
    have h0 : NB = ((F 0).filter (·.1 == .barrier)).length := by rw [← List.countP_eq_length_filter, hNB 0 (by decide)]
    rw [snapshots_get (t := 0) (by decide) (by omega), snapshots_get (t := 0) (by decide)
      (by rw [hE0, List.filter_append, List.length_append]; omega),
      snapMem_congr _ (fun b hb => stores_filter hS (by omega))]

/-- Settled answers stay settled past `n` more, each from the tape and given
    back by the laws. -/
theorem settled_extend {ans : Nat → List Answer} {P : Nat → Nat → Answer → Prop} {K : Nat} {T R : List Answer}
    {l : Nat} (hset : ∀ k, k < K → ∃ a, (ans l)[k]? = some a ∧ P l k a)
    (ht : (ans l).drop K = T ++ R) (hlaw : ∀ i a, T[i]? = some a → P l (K + i) a) :
    ∀ k, k < K + T.length → ∃ a, (ans l)[k]? = some a ∧ P l k a := by
  intro k hk
  rcases Nat.lt_or_ge k K with h | h
  · exact hset k h
  · obtain ⟨i, rfl⟩ := Nat.exists_eq_add_of_le h
    have hi : i < T.length := by omega
    refine ⟨T[i], getElem?_of_drop ht (List.getElem?_eq_getElem hi), hlaw i _ (List.getElem?_eq_getElem hi)⟩

theorem sharedStores_byte {w : World} {start : Nat} (hs : start ≤ w.obs.length) {a v : UInt64} (ha : NotArena a)
    (m : Mem) : sharedStores { obsStore w a 1 v with mem := m } start = sharedStores w start := by
  rw [sharedStores_ext hs (L := [_]) rfl]
  have : ∀ s : Nat × List (Nat × UInt64 × Nat × UInt64), shStep s (.store a 1 v) = s := by
    intro s; obtain ⟨n, acc⟩ := s
    simp only [shStep]
    split
    · rename_i off h; exact absurd h (ha off)
    · rfl
  simp [this]

/-- A copy's lane run adds only byte stores, each outside shared memory. -/
theorem copyL_stores {d s n lv : UInt64} (hna : ∀ i, i < n.toNat → NotArena (d + UInt64.ofNat i))
    (hn : n.toNat + 64 < 2 ^ 64) (hl : lv.toNat < 32) :
    ∀ k m (w w' : World), copyL d s n lv k (UInt64.ofNat (32 * m)) w = some w' → 32 * m ≤ n.toNat + 32 →
      ∃ L, w'.obs = L ++ w.obs ∧ ∀ o ∈ L, ∃ a v, o = .store a 1 v ∧ NotArena a := by
  intro k
  induction k with
  | zero =>
      intro m w w' h _
      simp only [copyL] at h
      split at h
      · cases h
      · obtain rfl := (Option.some.inj h).symm; exact ⟨[], rfl, by simp⟩
  | succ k ih =>
      intro m w w' h hm
      simp only [copyL] at h
      split at h
      · rename_i hlt
        have hmn : 32 * m < n.toNat := by
          have := UInt64.lt_iff_toNat_lt.mp hlt
          rwa [UInt64.toNat_ofNat_of_lt' (show 32 * m < 2 ^ 64 by omega)] at this
        obtain ⟨w1, hw1, hrest⟩ := Option.bind_eq_some_iff.mp h
        have hnext : UInt64.ofNat (32 * m) + c64 32 = UInt64.ofNat (32 * (m + 1)) := by
          rw [show c64 32 = UInt64.ofNat 32 by decide, ← UInt64.ofNat_add]; congr 1
        rw [hnext] at hrest
        obtain ⟨L2, hL2, hall2⟩ := ih (m + 1) w1 w' hrest (by omega)
        have step : ∃ L1, w1.obs = L1 ++ w.obs ∧ ∀ o ∈ L1, ∃ a v, o = .store a 1 v ∧ NotArena a := by
          simp only [copyStep] at hw1
          split at hw1
          · rename_i hj
            obtain ⟨v, -, hw1⟩ := Option.bind_eq_some_iff.mp hw1
            obtain ⟨mm, -, rfl⟩ := stw_some hw1
            refine ⟨[_], rfl, ?_⟩
            intro o ho
            simp only [List.mem_singleton] at ho
            subst ho
            refine ⟨_, _, rfl, ?_⟩
            have hj' : 32 * m + lv.toNat < n.toNat := by
              have := UInt64.lt_iff_toNat_lt.mp hj
              rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show 32 * m < 2 ^ 64 by omega),
                Nat.mod_eq_of_lt (by omega)] at this
              exact this
            have := hna _ hj'
            rwa [UInt64.ofNat_add, UInt64.ofNat_toNat] at this
          · obtain rfl := bind_const_some hw1; exact ⟨[], rfl, by simp⟩
        obtain ⟨L1, hL1, hall1⟩ := step
        exact ⟨L2 ++ L1, by rw [hL2, hL1, List.append_assoc], fun o ho => by
          rcases List.mem_append.mp ho with h | h
          · exact hall2 o h
          · exact hall1 o h⟩
      · obtain rfl := (Option.some.inj h).symm; exact ⟨[], rfl, by simp⟩

/-- Observations of byte stores outside shared memory leave a lane's
    collectives and shared stores. -/
theorem plain_obs {w w' : World} {start : Nat} (hs : start ≤ w.obs.length) {L : List Obs} (h : w'.obs = L ++ w.obs)
    (hL : ∀ o ∈ L, ∃ a v, o = .store a 1 v ∧ NotArena a) :
    collCalls w' start = collCalls w start ∧ sharedStores w' start = sharedStores w start := by
  constructor
  · unfold collCalls
    rw [h, List.reverse_append, List.drop_append_of_le_length (by simpa using hs), List.filterMap_append]
    have : L.reverse.filterMap collOf = [] := by
      rw [List.filterMap_eq_nil_iff]
      intro o ho
      obtain ⟨a, v, rfl, -⟩ := hL o (List.mem_reverse.mp ho)
      rfl
    rw [this, List.append_nil]
  · rw [sharedStores_ext hs h]
    have key : ∀ (M : List Obs), (∀ o ∈ M, ∃ a v, o = .store a 1 v ∧ NotArena a) →
        ∀ s0 : Nat × List (Nat × UInt64 × Nat × UInt64), M.foldl shStep s0 = s0 := by
      intro M
      induction M with
      | nil => intro _ s0; rfl
      | cons o M ih =>
          intro hM s0
          rw [List.foldl_cons]
          obtain ⟨a, v, rfl, ha⟩ := hM o (by simp)
          have : shStep s0 (.store a 1 v) = s0 := by
            obtain ⟨n, acc⟩ := s0
            simp only [shStep]
            split
            · rename_i off hd; exact absurd hd (ha off)
            · rfl
          rw [this]; exact ih (fun o' ho' => hM o' (by simp [ho'])) s0
    rw [key _ (fun o ho => hL o (List.mem_reverse.mp ho))]
    simp

/-- A plain statement leaves each lane's collectives and shared stores. -/
theorem plain_lane {I O : UInt64 → Prop} {okC : WStmt → WState → Prop} {ans : List Answer} {start : Nat}
    {ws : WState} {st st' : LSt} {s : WStmt} {f : Nat} {l : Nat} (hl : l < 32)
    (hs : s matches .skip | .bin .. | .setp .. | .mov .. | .stgB .. | .coopCopy ..)
    (hok : okOf I O okC f s ws) (hR : Rel I ws st) (hst : start ≤ st.w.obs.length)
    (hr : laneRun (coopL ans start (addrOf .arena 0) (UInt64.ofNat l)) (addrOf .arena 0) f s st = some st') :
    collCalls st'.w start = collCalls st.w start ∧ sharedStores st'.w start = sharedStores st.w start := by
  cases s with
  | skip => simp [laneRun] at hr; subst hr; exact ⟨rfl, rfl⟩
  | bin o d a b => simp [laneRun] at hr; subst hr; exact ⟨rfl, rfl⟩
  | setp c d a b => simp [laneRun] at hr; subst hr; exact ⟨rfl, rfl⟩
  | mov d a => simp [laneRun] at hr; subst hr; exact ⟨rfl, rfl⟩
  | stgB addr val =>
      simp only [okOf] at hok
      simp only [laneRun] at hr
      obtain ⟨w, hw, hr⟩ := Option.bind_eq_some_iff.mp hr
      obtain rfl := (Option.some.inj hr).symm
      obtain ⟨m, -, rfl⟩ := stw_some hw
      rw [hR.regs] at *
      exact plain_obs hst (L := [_]) rfl (fun o ho => by
        simp only [List.mem_singleton] at ho; subst ho; exact ⟨_, _, rfl, hok.2.1⟩)
  | coopCopy dst src len =>
      simp only [okOf] at hok
      obtain ⟨hn, hsr, -, -, hI⟩ := hok
      simp only [laneRun, coopL] at hr
      obtain ⟨w, hw, hr⟩ := Option.bind_eq_some_iff.mp hr
      obtain rfl := (Option.some.inj hr).symm
      rw [hR.regs, show c64 0 = UInt64.ofNat (32 * 0) by decide] at hw
      obtain ⟨L, hL, hall⟩ := copyL_stores (fun i hi => (hI i hi).2.1) hn
        (by rw [UInt64.toNat_ofNat_of_lt' (show l < 2 ^ 64 by omega)]; exact hl) _ 0 _ _ hw (by omega)
      exact plain_obs hst hL hall
  | _ => cases hs

/-- **A plain statement's lane finishes**: where its store, or its copy's
    bytes, may be stored and its copy's bytes load from `I`. -/
theorem plain_some {I O : UInt64 → Prop} {okC : WStmt → WState → Prop} {ans : List Answer} {start : Nat}
    {ws : WState} {st : LSt} {s : WStmt} {f : Nat} {l : Nat} (hl : l < 32)
    (hs : s matches .skip | .bin .. | .setp .. | .mov .. | .stgB .. | .coopCopy ..)
    (hok : okOf I O okC f s ws) (hR : Rel I ws st) (hwr : ∀ a, O a → Writable st.w.mem a) :
    ∃ st', laneRun (coopL ans start (addrOf .arena 0) (UInt64.ofNat l)) (addrOf .arena 0) f s st = some st' := by
  cases s with
  | skip => simp only [laneRun]; exact ⟨_, rfl⟩
  | bin o d a b => simp only [laneRun]; exact ⟨_, rfl⟩
  | setp c d a b => simp only [laneRun]; exact ⟨_, rfl⟩
  | mov d a => simp only [laneRun]; exact ⟨_, rfl⟩
  | stgB addr val =>
      simp only [okOf] at hok
      obtain ⟨m', hm'⟩ := (hwr _ hok.2.2).store_some ((ws.regs val) &&& 0xff)
      simp only [laneRun, hR.regs, stw, hm', Option.bind_eq_bind, Option.bind_some]
      exact ⟨_, rfl⟩
  | coopCopy dst src len =>
      simp only [okOf] at hok
      obtain ⟨hn, hsr, -, hIs, hI⟩ := hok
      obtain ⟨w', hw'⟩ := copyL_some (I := I) (d := ws.regs dst) (s := ws.regs src) (n := ws.regs len) hl hn
        hIs ((ws.regs len).toNat + 1) 0 st.w (by omega) (by omega)
        (fun a ha => ⟨_, hR.gin a ha⟩) (fun i hi => hwr _ (hI i hi).2.2)
      simp only [laneRun, coopL, hR.regs, show c64 0 = UInt64.ofNat (32 * 0) by decide, hw',
        Option.bind_eq_bind, Option.bind_some]
      exact ⟨_, rfl⟩
  | _ => cases hs

/-- **A lane runs a statement with the warp**: answered from the warp's tape
    (`trace ca`), the lane's run finishes agreeing with the warp's run, its
    byte stores its share of the warp's, its collectives and shared stores the
    warp's traces for it (`trace cc`, `strace cc sc`), the rest of the tape
    ahead; given that the warp's run makes only the accesses `okOf` allows,
    every byte of `O` may be stored, and a cooperative statement's share does
    so (`hcoop`). -/
theorem lane_lock {I O : UInt64 → Prop} {okC : WStmt → WState → Prop} {ans : List Answer} {start l : Nat}
    (hl : l < 32) (hOna : ∀ a, O a → NotArena a)
    (ca : WStmt → WState → List Answer) (cc : WStmt → WState → List (Intr × List V))
    (sc : Nat → WStmt → WState → List (Nat × UInt64 × Nat × UInt64))
    (hcoop : ∀ (s : WStmt), (s matches .coopWindow .. | .coopExtendStep ..) → ∀ (f : Nat) (ws : WState) (st : LSt)
      (R : List Answer), okC s ws → Rel I ws st → start ≤ st.w.obs.length →
      ans.drop (collCalls st.w start).length = ca s ws ++ R →
      ∃ st', coopL ans start (addrOf .arena 0) (UInt64.ofNat l) s st = some st' ∧ Rel I (s.eval f ws) st' ∧
        collCalls st'.w start = collCalls st.w start ++ cc s ws ∧
        sharedStores st'.w start = sharedStores st.w start ++ sc (bars (collCalls st.w start)) s ws ∧
        ans.drop (collCalls st'.w start).length = R ∧ Tracks st.w st'.w) :
    ∀ (s : WStmt) (f : Nat) (ws : WState) (st : LSt) (R : List Answer), okOf I O okC f s ws → Rel I ws st →
      start ≤ st.w.obs.length → (∀ a, O a → Writable st.w.mem a) →
      ans.drop (collCalls st.w start).length = trace ca f s ws ++ R →
      ∃ st', laneRun (coopL ans start (addrOf .arena 0) (UInt64.ofNat l)) (addrOf .arena 0) f s st = some st' ∧
        Rel I (s.eval f ws) st' ∧
        gStores st'.w start = gStores st.w start ++ lanePart l (gstOf f s ws) ∧
        collCalls st'.w start = collCalls st.w start ++ trace cc f s ws ∧
        sharedStores st'.w start = sharedStores st.w start ++ strace cc sc f (bars (collCalls st.w start)) s ws ∧
        ans.drop (collCalls st'.w start).length = R ∧ Tracks st.w st'.w := by
  -- a plain statement passes no collective and no shared store
  have plain : ∀ (s : WStmt), (s matches .skip | .bin .. | .setp .. | .mov .. | .stgB .. | .coopCopy ..) →
      ∀ (f : Nat) (ws : WState) (st : LSt) (R : List Answer), okOf I O okC f s ws → Rel I ws st →
      start ≤ st.w.obs.length → (∀ a, O a → Writable st.w.mem a) →
      ans.drop (collCalls st.w start).length = trace ca f s ws ++ R →
      ∃ st', laneRun (coopL ans start (addrOf .arena 0) (UInt64.ofNat l)) (addrOf .arena 0) f s st = some st' ∧
        collCalls st'.w start = collCalls st.w start ++ trace cc f s ws ∧
        sharedStores st'.w start = sharedStores st.w start ++ strace cc sc f (bars (collCalls st.w start)) s ws ∧
        ans.drop (collCalls st'.w start).length = R ∧ Tracks st.w st'.w := by
    intro s hs f ws st R hok hR hst hwr hT
    obtain ⟨st', hr⟩ := plain_some (ans := ans) (start := start) hl hs hok hR hwr
    obtain ⟨hc, hsh⟩ := plain_lane hl hs hok hR hst hr
    have e1 : trace ca f s ws = [] := by cases s <;> first | (simp [trace]; done) | cases hs
    have e2 : trace cc f s ws = [] := by cases s <;> first | (simp [trace]; done) | cases hs
    have e3 : strace cc sc f (bars (collCalls st.w start)) s ws = [] := by
      cases s <;> first | (simp [strace]; done) | cases hs
    refine ⟨st', hr, by rw [hc, e2, List.append_nil], by rw [hsh, e3, List.append_nil], by rw [hc, hT, e1, List.nil_append],
      ?_⟩
    cases s with
    | skip | bin | setp | mov => simp only [laneRun, Option.some.injEq] at hr; subst hr; exact Tracks.refl _
    | stgB addr val =>
        simp only [laneRun] at hr
        obtain ⟨w, hw, hr⟩ := Option.bind_eq_some_iff.mp hr
        rw [← Option.some.inj hr]; exact Tracks.stw1 hw
    | coopCopy dst src len =>
        simp only [laneRun, coopL] at hr
        obtain ⟨w, hw, hr⟩ := Option.bind_eq_some_iff.mp hr
        rw [← Option.some.inj hr]; exact copyL_tracks _ _ _ _ hw
    | _ => cases hs
  intro s
  induction s with
  | skip =>
      intro f ws st R hok hR hst hwr hT
      obtain ⟨st', hr, hc, hsh, hT', htr⟩ := plain _ rfl f ws st R hok hR hst hwr hT
      refine ⟨st', hr, ?_, ?_, hc, hsh, hT', htr⟩ <;> (simp only [laneRun, Option.some.injEq] at hr; subst hr)
      · simpa [WStmt.eval] using hR
      · simp [gstOf, lanePart]
  | seq a b iha ihb =>
      intro f ws st R hok hR hst hwr hT
      simp only [trace, List.append_assoc] at hT
      simp only [okOf] at hok
      obtain ⟨mid, hra, hA, hgA, hcA, hsA, hTa, htA⟩ := iha f ws st _ hok.1 hR hst hwr hT
      have hstm := Nat.le_trans hst (laneRun_ext a f _ _ hra).len
      obtain ⟨st', hrb, hB, hgB, hcB, hsB, hTb, htB⟩ := ihb f (a.eval f ws) mid R hok.2 hA hstm
        (fun x hx => laneRun_writable (hOna x hx) _ _ _ _ hra (hwr x hx)) hTa
      refine ⟨st', ?_, by simpa [WStmt.eval] using hB, ?_, ?_, ?_, hTb, htA.trans htB⟩
      · simp only [laneRun, hra, Option.bind_eq_bind, Option.bind_some, hrb]
      · rw [hgB, hgA]; simp [gstOf, lanePart, List.filterMap_append]
      · rw [hcB, hcA, List.append_assoc]; simp only [trace]
      · rw [hsB, hsA, hcA, List.append_assoc]
        simp only [strace, bars, List.countP_append]
  | bin o d a b | setp c d a b | mov d a =>
      intro f ws st R hok hR hst hwr hT
      obtain ⟨st', hr, hc, hsh, hT', htr⟩ := plain _ rfl f ws st R hok hR hst hwr hT
      refine ⟨st', hr, ?_, ?_, hc, hsh, hT', htr⟩ <;> (simp only [laneRun, Option.some.injEq] at hr; subst hr)
      · obtain ⟨hreg, hsm, hgin⟩ := hR
        simp only [WStmt.eval]
        refine ⟨?_, hsm, hgin⟩
        funext x
        simp [LSt.set, WState.setReg, hreg, wargL_eval hreg]
      · simp [gstOf, lanePart]; rfl
  | stgB addr val =>
      intro f ws st R hok hR hst hwr hT
      obtain ⟨st', hr, hc, hsh, hT', htr⟩ := plain _ rfl f ws st R hok hR hst hwr hT
      refine ⟨st', hr, ?_, ?_, hc, hsh, hT', htr⟩ <;> simp only [okOf] at hok <;>
        (simp only [laneRun] at hr
         obtain ⟨w, hw, hr⟩ := Option.bind_eq_some_iff.mp hr
         rw [← Option.some.inj hr]
         obtain ⟨hreg, hsm, hgin⟩ := hR
         obtain ⟨m, hm, rfl⟩ := stw_some hw
         rw [hreg] at hm)
      · simp only [WStmt.eval, WState.stgByte]
        refine ⟨hreg, ?_, fun a ha => ?_⟩
        · show m.arena.data = ws.smem
          rw [store_arena hm hok.2.1]; exact hsm
        · show m.load a 1 = _
          rw [load_store_ne hm (fun e => hok.1 (e ▸ ha)) (hgin a ha),
            getD_set!_ne (fun e => hok.1 (by rw [UInt64.toNat_inj.mp e]; exact ha))]
      · rw [gStores_ext hst (L := [.store _ 1 _]) rfl]
        simp [gstOf, lanePart, hreg]
  | uif c t e iht ihe =>
      intro f ws st R hok hR hst hwr hT
      simp only [okOf] at hok
      by_cases hc : (ws.regs c == 1) = true
      · simp only [hc, if_true] at hok
        simp only [trace, hc, if_true] at hT
        obtain ⟨st', hr, hrest⟩ := iht f ws st R hok hR hst hwr hT
        refine ⟨st', ?_, by simpa [WStmt.eval, gstOf, trace, strace, hc] using hrest⟩
        simp only [laneRun, hR.regs, hc, if_true]; exact hr
      · simp only [hc, Bool.false_eq_true, if_false] at hok
        simp only [trace, hc, Bool.false_eq_true, if_false] at hT
        obtain ⟨st', hr, hrest⟩ := ihe f ws st R hok hR hst hwr hT
        refine ⟨st', ?_, by simpa [WStmt.eval, gstOf, trace, strace, hc] using hrest⟩
        simp only [laneRun, hR.regs, hc, Bool.false_eq_true, if_false]; exact hr
  | uwhile c body ih =>
      intro f
      induction f with
      | zero =>
          intro ws st R hok hR hst hwr hT
          simp only [okOf] at hok
          simp only [trace, List.nil_append] at hT
          refine ⟨st, ?_, by simpa [WStmt.eval] using hR, by simp [gstOf, lanePart],
            by simp [trace], by simp [strace], hT, Tracks.refl _⟩
          simp only [laneRun, hR.regs, hok, Bool.false_eq_true, if_false]
      | succ f ihf =>
          intro ws st R hok hR hst hwr hT
          simp only [okOf] at hok
          by_cases hc : (ws.regs c == 1) = true
          · simp only [trace, hc, if_true, List.append_assoc] at hT
            obtain ⟨mid, hra, hA, hgA, hcA, hsA, hTa, htA⟩ := ih f ws st _ (hok hc).1 hR hst hwr hT
            have hstm := Nat.le_trans hst (laneRun_ext body f _ _ hra).len
            obtain ⟨st', hrb, hB, hgB, hcB, hsB, hTb, htB⟩ := ihf (body.eval f ws) mid R (hok hc).2 hA hstm
              (fun x hx => laneRun_writable (hOna x hx) _ _ _ _ hra (hwr x hx)) hTa
            refine ⟨st', ?_, by simpa [WStmt.eval, hc] using hB, ?_, ?_, ?_, hTb, htA.trans htB⟩
            · simp only [laneRun, hR.regs, hc, if_true, hra, Option.bind_eq_bind, Option.bind_some, hrb]
            · rw [hgB, hgA]; simp [gstOf, hc, lanePart, List.filterMap_append]
            · rw [hcB, hcA, List.append_assoc]; simp only [trace, hc, if_true]
            · rw [hsB, hsA, hcA, List.append_assoc]
              simp only [strace, trace, hc, if_true, bars, List.countP_append]
          · simp only [trace, hc, Bool.false_eq_true, if_false, List.nil_append] at hT
            refine ⟨st, ?_, by simpa [WStmt.eval, hc] using hR, by simp [gstOf, hc, lanePart],
              by simp [trace, hc], by simp [strace, hc], hT, Tracks.refl _⟩
            simp only [laneRun, hR.regs, hc, Bool.false_eq_true, if_false]
  | coopCopy dst src len =>
      intro f ws st R hok hR hst hwr hT
      obtain ⟨st', hr, hc, hsh, hT', htr⟩ := plain _ rfl f ws st R hok hR hst hwr hT
      refine ⟨st', hr, ?_, ?_, hc, hsh, hT', htr⟩ <;> simp only [okOf] at hok <;>
        (obtain ⟨hn, hs, -, hIs, hI⟩ := hok
         simp only [laneRun, coopL] at hr
         obtain ⟨w, hw, hr⟩ := Option.bind_eq_some_iff.mp hr
         rw [← Option.some.inj hr]
         obtain ⟨hreg, hsm, hgin⟩ := hR
         rw [hreg, show c64 0 = UInt64.ofNat (32 * 0) by decide] at hw
         obtain ⟨⟨L, hL, hsL⟩, har, hIk⟩ := copyL_lane (g := ws.gmem) hl hn hs
           (fun i hi => ⟨hIs i (by omega), (hI i hi).1, (hI i hi).2.1⟩) _ 0 _ _ hw (by omega) hgin)
      · simp only [WStmt.eval]
        refine ⟨hreg, by rw [har]; exact hsm, fun a ha => ?_⟩
        rw [hIk a ha, hgin a ha, copyGmem_getD_out]
        intro i hi e
        have hdi : a = ws.regs dst + UInt64.ofNat i := by
          apply UInt64.toNat_inj.mp
          rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show i < 2 ^ 64 by omega), ← e]
          exact (Nat.mod_eq_of_lt a.toNat_lt).symm
        exact (hI i hi).1 (hdi ▸ ha)
      · rw [gStores_ext hst hL]
        simp only [gstOf, lanePart_copyTr, ← hsL, stores1]
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      intro f ws st R hok hR hst hwr hT
      simp only [trace] at hT
      obtain ⟨st', hc, hRl, hcl, hsl, hT', htr⟩ := hcoop _ rfl f ws st R (by simpa [okOf] using hok) hR hst hT
      refine ⟨st', by simpa [laneRun] using hc, hRl, ?_, by simpa [trace] using hcl, by simpa [strace] using hsl, hT',
        htr⟩
      rw [gStores_nobyte hst ((coopL_ext _ _ _ hc).2 rfl)]
      simp [gstOf, lanePart]
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      intro f ws st R hok hR hst hwr hT
      simp only [trace] at hT
      obtain ⟨st', hc, hRl, hcl, hsl, hT', htr⟩ := hcoop _ rfl f ws st R (by simpa [okOf] using hok) hR hst hT
      refine ⟨st', by simpa [laneRun] using hc, hRl, ?_, by simpa [trace] using hcl, by simpa [strace] using hsl, hT',
        htr⟩
      rw [gStores_nobyte hst ((coopL_ext _ _ _ hc).2 rfl)]
      simp [gstOf, lanePart]

/-- What lane `l` passes in the warp's cooperative statements. -/
def lz4Calls (l : Nat) : WStmt → WState → List (Intr × List V)
  | .coopWindow _ _ _ sp inStride searchLim hashLog tbl, ws => winCalls ws sp inStride searchLim hashLog tbl l
  | .coopExtendStep _ p0 cand0 ml inStride endCap, ws => extCalls ws p0 cand0 ml inStride endCap l
  | _, _ => []

/-- The shared stores lane `l` makes in the warp's cooperative statements,
    tagged `nb`. -/
def lz4Stores (l : Nat) : Nat → WStmt → WState → List (Nat × UInt64 × Nat × UInt64)
  | nb, .coopWindow _ _ _ sp inStride searchLim hashLog tbl, ws => winStores ws sp inStride searchLim hashLog tbl l nb
  | _, _, _ => []

/-- **A lane of the warp runs a warp statement with the warp**: answered from
    the warp's tape, it finishes agreeing with the warp's run, its byte stores
    its share of the warp's, its collectives and shared stores the warp's for
    it, the rest of the tape ahead. -/
theorem lane_run {I O : UInt64 → Prop} {ans : List Answer} {start l : Nat} (hl : l < 32)
    (hIna : ∀ a, I a → NotArena a) (hOna : ∀ a, O a → NotArena a) :
    ∀ (s : WStmt) (f : Nat) (ws : WState) (st : LSt) (R : List Answer), okOf I O (okLZ4 I) f s ws →
      Rel I ws st → start ≤ st.w.obs.length → (∀ a, O a → Writable st.w.mem a) →
      ans.drop (collCalls st.w start).length = trace lz4Ans f s ws ++ R →
      ∃ st', laneRun (coopL ans start (addrOf .arena 0) (UInt64.ofNat l)) (addrOf .arena 0) f s st = some st' ∧
        Rel I (s.eval f ws) st' ∧
        gStores st'.w start = gStores st.w start ++ lanePart l (gstOf f s ws) ∧
        collCalls st'.w start = collCalls st.w start ++ trace (lz4Calls l) f s ws ∧
        sharedStores st'.w start = sharedStores st.w start ++
          strace (lz4Calls l) (lz4Stores l) f (bars (collCalls st.w start)) s ws ∧
        ans.drop (collCalls st'.w start).length = R ∧ Tracks st.w st'.w := by
  refine lane_lock hl hOna lz4Ans (lz4Calls l) (lz4Stores l) ?_
  intro s hs f ws st R hok hR hst hT
  cases s with
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      obtain ⟨hsl, hh, hin, hib, hsp, htbl, htb, hfp, hfc, hpc, hsz⟩ := hok
      exact lane_window hR hst hl hsl hh hin hib hsp htbl htb hIna hfp hfc hpc hsz hT
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      obtain ⟨hecR, hec1, hend, hendI, hcp, hov, hin, hib, hpe⟩ := hok
      obtain ⟨st', hc, hRl, hcl, hsh, hT', htr⟩ := lane_extend hR hst hl hecR hec1 hend hendI hcp hov hin hib hpe hT
      exact ⟨st', hc, hRl, hcl, by rw [hsh]; simp [lz4Stores], hT', htr⟩
  | _ => cases hs

/-- Every lane passes as many collectives as the warp's tape has answers. -/
theorem trace_len (l : Nat) : ∀ (s : WStmt) (f : Nat) (ws : WState),
    (trace (lz4Calls l) f s ws).length = (trace lz4Ans f s ws).length := by
  intro s
  induction s with
  | seq a b iha ihb => intro f ws; simp only [trace, List.length_append, iha, ihb]
  | uif c t e iht ihe =>
      intro f ws
      by_cases hc : (ws.regs c == 1) = true
      · simp only [trace, hc, if_true]; exact iht f ws
      · simp only [trace, hc, Bool.false_eq_true, if_false]; exact ihe f ws
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws; simp [trace]
      | succ f ihf =>
          intro ws
          by_cases hc : (ws.regs c == 1) = true
          · simp only [trace, hc, if_true, List.length_append, ih, ihf]
          · simp [trace, hc]
  | coopWindow => intro f ws; simp [trace, lz4Calls, lz4Ans, winCalls, winAns]
  | coopExtendStep => intro f ws; simp [trace, lz4Calls, lz4Ans, extCalls, extAns]
  | _ => intro f ws; simp [trace]

/-- Every lane passes the same barriers. -/
theorem bars_lane (l : Nat) : ∀ (s : WStmt) (f : Nat) (ws : WState),
    bars (trace (lz4Calls l) f s ws) = bars (trace (lz4Calls 0) f s ws) := by
  intro s
  induction s with
  | seq a b iha ihb => intro f ws; simp only [trace, bars, List.countP_append] at iha ihb ⊢; rw [iha, ihb]
  | uif c t e iht ihe =>
      intro f ws
      by_cases hc : (ws.regs c == 1) = true
      · simp only [trace, hc, if_true]; exact iht f ws
      · simp only [trace, hc, Bool.false_eq_true, if_false]; exact ihe f ws
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws; simp [trace]
      | succ f ihf =>
          intro ws
          by_cases hc : (ws.regs c == 1) = true
          · simp only [trace, hc, if_true, bars, List.countP_append] at ih ihf ⊢; rw [ih, ihf]
          · simp only [trace, hc, Bool.false_eq_true, if_false]
  | coopWindow => intro f ws; simp [trace, lz4Calls, winCalls, bars, show (Intr.ballot == Intr.barrier) = false by decide, show (Intr.barrier == Intr.barrier) = true by decide, show (Intr.shflIdx == Intr.barrier) = false by decide]
  | coopExtendStep => intro f ws; simp [trace, lz4Calls, extCalls]
  | _ => intro f ws; simp [trace]

/-- A lane's shared stores in a statement are tagged from `nb` to below its
    barriers past `nb`. -/
theorem strace_tags (l : Nat) : ∀ (s : WStmt) (f nb : Nat) (ws : WState),
    ∀ x ∈ strace (lz4Calls l) (lz4Stores l) f nb s ws, nb ≤ x.1 ∧ x.1 < nb + bars (trace (lz4Calls l) f s ws) := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro f nb ws x hx
      simp only [strace, List.mem_append] at hx
      simp only [trace, bars, List.countP_append]
      rcases hx with h | h
      · have := iha f nb ws x h; simp only [bars] at this; omega
      · have := ihb f _ (a.eval f ws) x h; simp only [bars] at this; omega
  | uif c t e iht ihe =>
      intro f nb ws x hx
      by_cases hc : (ws.regs c == 1) = true
      · simp only [strace, hc, if_true] at hx; simp only [trace, hc, if_true]; exact iht f nb ws x hx
      · simp only [strace, hc, Bool.false_eq_true, if_false] at hx
        simp only [trace, hc, Bool.false_eq_true, if_false]; exact ihe f nb ws x hx
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro nb ws x hx; simp [strace] at hx
      | succ f ihf =>
          intro nb ws x hx
          by_cases hc : (ws.regs c == 1) = true
          · simp only [strace, hc, if_true, List.mem_append] at hx
            simp only [trace, hc, if_true, bars, List.countP_append]
            rcases hx with h | h
            · have := ih f nb ws x h; simp only [bars] at this; omega
            · have := ihf _ (b.eval f ws) x h; simp only [bars] at this; omega
          · simp [strace, hc] at hx
  | coopWindow =>
      intro f nb ws x hx
      simp only [strace, lz4Stores, winStores] at hx
      split at hx
      · simp at hx; subst hx; simp [trace, lz4Calls, winCalls, bars, show (Intr.ballot == Intr.barrier) = false by decide, show (Intr.barrier == Intr.barrier) = true by decide, show (Intr.shflIdx == Intr.barrier) = false by decide]
      · cases hx
  | coopExtendStep => intro f nb ws x hx; simp [strace, lz4Stores] at hx
  | _ => intro f nb ws x hx; simp [strace] at hx

/-- Where the lanes stand before a statement: each lane's final collectives
    run through the warp's trace for it after `K` collectives and `NB`
    barriers, its final shared stores through its trace, those before tagged
    below `NB` and those after from past the statement's barriers. -/
def Before (F : Nat → List (Intr × List V)) (S : Nat → List (Nat × UInt64 × Nat × UInt64)) (f : Nat) (s : WStmt)
    (ws : WState) (K NB : Nat) : Prop :=
  (∀ l, l < 32 → ∃ P Q, F l = P ++ trace (lz4Calls l) f s ws ++ Q ∧ P.length = K ∧ bars P = NB) ∧
  ∀ l, l < 32 → ∃ P Q, S l = P ++ strace (lz4Calls l) (lz4Stores l) f NB s ws ++ Q ∧ (∀ x ∈ P, x.1 < NB) ∧
    ∀ x ∈ Q, NB + bars (trace (lz4Calls l) f s ws) ≤ x.1

/-- Lanes before `a ; b` stand before `a`. -/
theorem Before.left {F S f a b ws K NB} (h : Before F S f (.seq a b) ws K NB) : Before F S f a ws K NB := by
  refine ⟨fun l hl => ?_, fun l hl => ?_⟩
  · obtain ⟨P, Q, hP, hK, hN⟩ := h.1 l hl
    exact ⟨P, trace (lz4Calls l) f b (a.eval f ws) ++ Q, by rw [hP]; simp [trace, List.append_assoc], hK, hN⟩
  · obtain ⟨P, Q, hP, hPt, hQt⟩ := h.2 l hl
    refine ⟨P, strace (lz4Calls l) (lz4Stores l) f (NB + bars (trace (lz4Calls l) f a ws)) b (a.eval f ws) ++ Q,
      by rw [hP]; simp [strace, List.append_assoc], hPt, fun x hx => ?_⟩
    rcases List.mem_append.mp hx with h1 | h1
    · exact (strace_tags l b f _ _ x h1).1
    · have := hQt x h1; simp only [trace, bars, List.countP_append] at this ⊢; omega

/-- Lanes before `a ; b` stand before `b` once past `a`. -/
theorem Before.right {F S f a b ws K NB} (h : Before F S f (.seq a b) ws K NB) :
    Before F S f b (a.eval f ws) (K + (trace lz4Ans f a ws).length) (NB + bars (trace (lz4Calls 0) f a ws)) := by
  refine ⟨fun l hl => ?_, fun l hl => ?_⟩
  · obtain ⟨P, Q, hP, hK, hN⟩ := h.1 l hl
    refine ⟨P ++ trace (lz4Calls l) f a ws, Q, by rw [hP]; simp [trace, List.append_assoc], ?_, ?_⟩
    · rw [List.length_append, hK, trace_len]
    · simp only [bars, List.countP_append] at hN ⊢; rw [hN]; have := bars_lane l a f ws; simp only [bars] at this
      rw [this]
  · obtain ⟨P, Q, hP, hPt, hQt⟩ := h.2 l hl
    have hb := bars_lane l a f ws
    refine ⟨P ++ strace (lz4Calls l) (lz4Stores l) f NB a ws, Q, ?_, fun x hx => ?_, fun x hx => ?_⟩
    · rw [hP, ← hb]; simp [strace, List.append_assoc]
    · rcases List.mem_append.mp hx with h1 | h1
      · have := hPt x h1; omega
      · have := (strace_tags l a f NB ws x h1).2; rw [hb] at this; exact this
    · have := hQt x hx; simp only [trace, bars, List.countP_append] at this hb ⊢; omega

set_option maxHeartbeats 2000000 in
/-- **The laws give the warp's tape back**: where the lanes stand before a
    statement (`Before`) and the warp's table is shared memory after the first
    `NB` phases, the laws, from what the lanes finally passed, answer each
    lane's collectives in the statement with the warp's answers, and the
    warp's table after it is shared memory after its barriers' phases. -/
theorem warp_laws {I O : UInt64 → Prop} {smem0 : ByteArray} {F : Nat → List (Intr × List V)}
    {S : Nat → List (Nat × UInt64 × Nat × UInt64)} :
    ∀ (s : WStmt) (f : Nat) (ws : WState) (K NB : Nat), okOf I O (okLZ4 I) f s ws → Before F S f s ws K NB →
      ws.smem = (snapMem ((List.range 32).flatMap S) smem0 NB).arena.data →
      (∀ l, l < 32 → ∀ i a, (trace lz4Ans f s ws)[i]? = some a →
        LawAt F (snapshots 32 smem0 F S) smem0 l (K + i) a) ∧
      (s.eval f ws).smem =
        (snapMem ((List.range 32).flatMap S) smem0 (NB + bars (trace (lz4Calls 0) f s ws))).arena.data := by
  -- a plain statement answers nothing and leaves the table
  have plain : ∀ (s : WStmt), (s matches .skip | .bin .. | .setp .. | .mov .. | .stgB .. | .coopCopy ..) →
      ∀ (f : Nat) (ws : WState) (K NB : Nat),
      ws.smem = (snapMem ((List.range 32).flatMap S) smem0 NB).arena.data →
      (∀ l, l < 32 → ∀ i a, (trace lz4Ans f s ws)[i]? = some a →
        LawAt F (snapshots 32 smem0 F S) smem0 l (K + i) a) ∧
      (s.eval f ws).smem =
        (snapMem ((List.range 32).flatMap S) smem0 (NB + bars (trace (lz4Calls 0) f s ws))).arena.data := by
    intro s hs f ws K NB hsnap
    have e1 : trace lz4Ans f s ws = [] := by cases s <;> first | (simp [trace]; done) | cases hs
    have e2 : trace (lz4Calls 0) f s ws = [] := by cases s <;> first | (simp [trace]; done) | cases hs
    have hsm : (s.eval f ws).smem = ws.smem := by
      cases s <;> simp [WStmt.eval, WState.setReg, WState.stgByte] at hs ⊢
    refine ⟨fun l hl i a hi => by rw [e1] at hi; simp at hi, ?_⟩
    rw [hsm, e2]; simpa using hsnap
  intro s
  induction s with
  | skip => intro f ws K NB _ _ hsnap; exact plain _ rfl f ws K NB hsnap
  | bin o d a b => intro f ws K NB _ _ hsnap; exact plain _ rfl f ws K NB hsnap
  | setp c d a b => intro f ws K NB _ _ hsnap; exact plain _ rfl f ws K NB hsnap
  | mov d a => intro f ws K NB _ _ hsnap; exact plain _ rfl f ws K NB hsnap
  | stgB addr val => intro f ws K NB _ _ hsnap; exact plain _ rfl f ws K NB hsnap
  | coopCopy dst src len => intro f ws K NB _ _ hsnap; exact plain _ rfl f ws K NB hsnap
  | seq a b iha ihb =>
      intro f ws K NB hok hA hsnap
      simp only [okOf] at hok
      obtain ⟨hLa, hsa⟩ := iha f ws K NB hok.1 hA.left hsnap
      obtain ⟨hLb, hsb⟩ := ihb f (a.eval f ws) _ _ hok.2 hA.right hsa
      refine ⟨fun l hl i x hi => ?_, ?_⟩
      · simp only [trace] at hi
        rcases Nat.lt_or_ge i (trace lz4Ans f a ws).length with h | h
        · rw [List.getElem?_append_left h] at hi; exact hLa l hl i x hi
        · rw [List.getElem?_append_right h] at hi
          have := hLb l hl _ x hi
          rwa [show K + (trace lz4Ans f a ws).length + (i - (trace lz4Ans f a ws).length) = K + i by omega] at this
      · simp only [WStmt.eval] at hsb ⊢
        rw [hsb]; simp only [trace, bars, List.countP_append, Nat.add_assoc]
  | uif c t e iht ihe =>
      intro f ws K NB hok hA hsnap
      simp only [okOf] at hok
      by_cases hc : (ws.regs c == 1) = true
      · simp only [hc, if_true] at hok
        have hA' : Before F S f t ws K NB := by simpa [Before, trace, strace, hc] using hA
        simpa [WStmt.eval, trace, hc] using iht f ws K NB hok hA' hsnap
      · simp only [hc, Bool.false_eq_true, if_false] at hok
        have hA' : Before F S f e ws K NB := by simpa [Before, trace, strace, hc] using hA
        simpa [WStmt.eval, trace, hc] using ihe f ws K NB hok hA' hsnap
  | uwhile c body ih =>
      intro f
      induction f with
      | zero =>
          intro ws K NB _ _ hsnap
          refine ⟨fun l hl i a hi => by simp [trace] at hi, ?_⟩
          simpa [WStmt.eval, trace] using hsnap
      | succ f ihf =>
          intro ws K NB hok hA hsnap
          simp only [okOf] at hok
          by_cases hc : (ws.regs c == 1) = true
          · have hA' : Before F S f (.seq body (.uwhile c body)) ws K NB := by simpa [Before, trace, strace, hc] using hA
            obtain ⟨hLa, hsa⟩ := ih f ws K NB (hok hc).1 hA'.left hsnap
            obtain ⟨hLb, hsb⟩ := ihf (body.eval f ws) _ _ (hok hc).2 hA'.right hsa
            refine ⟨fun l hl i x hi => ?_, ?_⟩
            · simp only [trace, hc, if_true] at hi
              rcases Nat.lt_or_ge i (trace lz4Ans f body ws).length with h | h
              · rw [List.getElem?_append_left h] at hi; exact hLa l hl i x hi
              · rw [List.getElem?_append_right h] at hi
                have := hLb l hl _ x hi
                rwa [show K + (trace lz4Ans f body ws).length + (i - (trace lz4Ans f body ws).length) = K + i by omega]
                  at this
            · simp only [WStmt.eval, hc, if_true] at hsb ⊢
              rw [hsb]; simp only [trace, hc, if_true, bars, List.countP_append, Nat.add_assoc]
          · refine ⟨fun l hl i a hi => by simp [trace, hc] at hi, ?_⟩
            simpa [WStmt.eval, trace, hc] using hsnap
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      intro f ws K NB hok hA hsnap
      simp only [okOf, okLZ4] at hok
      obtain ⟨hsl, hh, hin, hib, hsp, htbl, htb, hfp, hfc, hpc, hsz⟩ := hok
      have hF : ∀ l, l < 32 → ∃ P Q, F l = P ++ winCalls ws sp inStride searchLim hashLog tbl l ++ Q ∧
          P.length = K ∧ bars P = NB := fun l hl => by
        obtain ⟨P, Q, hP, hK, hN⟩ := hA.1 l hl
        exact ⟨P, Q, by simpa [trace, lz4Calls] using hP, hK, hN⟩
      have hS : ∀ l, l < 32 → (S l).filter (·.1 = NB) = winStores ws sp inStride searchLim hashLog tbl l NB := by
        intro l hl
        obtain ⟨P, Q, hP, hPt, hQt⟩ := hA.2 l hl
        simp only [strace, lz4Stores] at hP
        have hb1 : bars (winCalls ws sp inStride searchLim hashLog tbl l) = 1 := by simp [winCalls, bars, show (Intr.ballot == Intr.barrier) = false by decide, show (Intr.barrier == Intr.barrier) = true by decide, show (Intr.shflIdx == Intr.barrier) = false by decide]
        simp only [trace, lz4Calls] at hQt
        rw [hb1] at hQt
        have h1 : P.filter (fun x => decide (x.1 = NB)) = [] :=
          List.filter_eq_nil_iff.mpr fun x hx => by have := hPt x hx; simp; omega
        have h2 : Q.filter (fun x => decide (x.1 = NB)) = [] :=
          List.filter_eq_nil_iff.mpr fun x hx => by have := hQt x hx; simp; omega
        have h3 : (winStores ws sp inStride searchLim hashLog tbl l NB).filter (fun x => decide (x.1 = NB)) =
            winStores ws sp inStride searchLim hashLog tbl l NB := by
          unfold winStores; split <;> simp
        rw [hP, List.filter_append, List.filter_append, h1, h2, h3]; simp
      obtain ⟨hL, hsm⟩ := window_laws (found := found) (p0 := p0) (cand0 := cand0) (f := f) hF hS hsnap hsl hsp htb hsz
      refine ⟨fun l hl i a hi => hL l hl i a (by simpa [trace, lz4Ans] using hi), ?_⟩
      rw [hsm]; simp [trace, lz4Calls, winCalls, bars, show (Intr.ballot == Intr.barrier) = false by decide, show (Intr.barrier == Intr.barrier) = true by decide, show (Intr.shflIdx == Intr.barrier) = false by decide]
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      intro f ws K NB hok hA hsnap
      have hF : ∀ l, l < 32 → ∃ P Q, F l = P ++ extCalls ws p0 cand0 ml inStride endCap l ++ Q ∧ P.length = K :=
        fun l hl => by
          obtain ⟨P, Q, hP, hK, -⟩ := hA.1 l hl
          exact ⟨P, Q, by simpa [trace, lz4Calls] using hP, hK⟩
      refine ⟨fun l hl i a hi => extend_laws hF l hl i a (by simpa [trace, lz4Ans] using hi), ?_⟩
      have hev : ((WStmt.coopExtendStep adv p0 cand0 ml inStride endCap).eval f ws).smem = ws.smem := by
        rw [WStmt.eval.eq_12]; rfl
      rw [hev]; simpa [trace, lz4Calls, extCalls, bars, show (Intr.ballot == Intr.barrier) = false by decide, show (Intr.barrier == Intr.barrier) = true by decide, show (Intr.shflIdx == Intr.barrier) = false by decide] using hsnap

end Lz4Code
