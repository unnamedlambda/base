module
public import Lz4.CodeLaunch
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeContent` — what a launch of the compressor leaves

The warp's run of a block's body leaves global memory as its byte stores
make it (`eval_replay`, with `copyGmem_getD`, `replay_copy`). Where the warp
writes each byte once, the block's lanes, each storing its share in thread
order, leave every byte as the warp's order does (`replay_lanes`); other
blocks store only into their own outputs (`replay_blocks`). So a launch, on
the Code and on the PTX it compiles to, leaves each block's output as the
warp's run of that block does (`launch_output`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel

open AlgorithmLib.ML.LaneWarp (NotArena Writable)

namespace Lz4Code

/-- A copy's bytes, where the bytes it reads are not among those it writes:
    each written byte the one `src` held. -/
theorem copyGmem_getD : ∀ (n : Nat) (g : Array UInt8) (dst src : Nat),
    (∀ i j, i < n → j < n → src + i ≠ dst + j) → ∀ k,
      (copyGmem g dst src n).getD k 0 = if dst ≤ k ∧ k < dst + n ∧ k < g.size then g.getD (src + (k - dst)) 0
        else g.getD k 0
  | 0, g, dst, src, _, k => by simp [copyGmem]; omega
  | n + 1, g, dst, src, h, k => by
      simp only [copyGmem]
      rw [copyGmem_getD n _ (dst + 1) (src + 1) (fun i j hi hj e => h (i + 1) (j + 1) (by omega) (by omega)
        (by omega)) k]
      have hsz : (g.setIfInBounds dst (g.getD src 0)).size = g.size := by simp
      rw [hsz]
      have hsrc : ∀ i, i < n → (g.setIfInBounds dst (g.getD src 0)).getD (src + 1 + i) 0 = g.getD (src + 1 + i) 0 := by
        intro i hi
        simp only [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds_ne
          (show dst ≠ src + 1 + i from fun e => h (i + 1) 0 (by omega) (by omega) (by omega))]
      by_cases hk : dst + 1 ≤ k ∧ k < dst + 1 + n ∧ k < g.size
      · rw [if_pos hk, if_pos ⟨by omega, by omega, hk.2.2⟩, show src + 1 + (k - (dst + 1)) = src + (k - dst) by omega]
        rw [← show src + 1 + (k - (dst + 1)) = src + (k - dst) by omega, hsrc _ (by omega)]
      · rw [if_neg hk]
        by_cases hk0 : k = dst ∧ k < g.size
        · obtain ⟨rfl, hkl⟩ := hk0
          rw [if_pos ⟨Nat.le_refl _, by omega, hkl⟩, Nat.sub_self, Nat.add_zero]
          simp [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds_self_of_lt hkl]
        · have : ¬ (dst ≤ k ∧ k < dst + (n + 1) ∧ k < g.size) := by omega
          rw [if_neg this]
          by_cases e : dst = k
          · subst e
            simp only [Array.getD_eq_getD_getElem?]
            rw [Array.getElem?_setIfInBounds_self]
            have : ¬ dst < g.size := by omega
            simp [this]
          · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds_ne e]

/-- The warp's byte stores, without who makes them. -/
abbrev pairsOf (tr : List (Option Nat × UInt64 × UInt64)) : List (UInt64 × UInt64) := tr.map fun x => (x.2.1, x.2.2)

theorem byte_and (b : UInt8) : b.toUInt64 &&& 0xff = b.toUInt64 := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, show (0xff : UInt64).toNat = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
  simp [Nat.mod_eq_of_lt b.toNat_lt]

/-- A copy's byte stores leave each byte of its destination the source's. -/
theorem replay_copy (g : Array UInt8) (d s a : UInt64) (v : Option UInt64) :
    ∀ n, d.toNat + n < 2 ^ 64 → replay (pairsOf (copyTr g d s n)) a v =
      if d.toNat ≤ a.toNat ∧ a.toNat < d.toNat + n then some (g.getD (s.toNat + (a.toNat - d.toNat)) 0).toUInt64
      else v
  | 0, _ => by simp [copyTr, replay]; omega
  | n + 1, hn => by
      have hsplit : pairsOf (copyTr g d s (n + 1)) = pairsOf (copyTr g d s n) ++
          [(d + UInt64.ofNat n, (g.getD (s.toNat + n) 0).toUInt64)] := by
        simp [copyTr, List.range_succ]
      rw [hsplit, replay_append, replay_copy g d s a v n (by omega)]
      have hdn : (d + UInt64.ofNat n).toNat = d.toNat + n := by
        rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show n < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
      simp only [replay, List.foldl_cons, List.foldl_nil]
      by_cases e : d + UInt64.ofNat n = a
      · subst e
        rw [if_pos rfl, hdn, if_pos ⟨by omega, by omega⟩, show d.toNat + n - d.toNat = n by omega, byte_and]
      · rw [if_neg e]
        have hne : a.toNat ≠ d.toNat + n := fun h => e (UInt64.toNat_inj.mp (by rw [hdn, h]))
        by_cases h1 : d.toNat ≤ a.toNat ∧ a.toNat < d.toNat + n
        · rw [if_pos h1, if_pos ⟨h1.1, by omega⟩]
        · rw [if_neg h1, if_neg (by omega)]

/-- The warp's run leaves the size of global memory. -/
theorem eval_size : ∀ (s : WStmt) (f : Nat) (ws : WState), (s.eval f ws).gmem.size = ws.gmem.size := by
  intro s
  induction s with
  | seq a b iha ihb => intro f ws; simp only [WStmt.eval]; rw [ihb, iha]
  | uif c t e iht ihe =>
      intro f ws
      by_cases hc : (ws.regs c == 1) = true
      · simp only [WStmt.eval, hc, if_true]; exact iht f ws
      · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]; exact ihe f ws
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws; simp [WStmt.eval]
      | succ f ihf =>
          intro ws
          by_cases hc : (ws.regs c == 1) = true
          · simp only [WStmt.eval, hc, if_true]; rw [ihf, ih]
          · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]
  | stgB => intro f ws; simp [WStmt.eval, WState.stgByte]
  | coopCopy => intro f ws; simp only [WStmt.eval]; exact copyGmem_size _ _ _ _
  | coopWindow => intro f ws; rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_gmem]
  | coopExtendStep => intro f ws; rw [WStmt.eval.eq_12]; rfl
  | _ => intro f ws; simp [WStmt.eval, WState.setReg]

/-- **The warp's global memory is its byte stores made**: where it makes only
    the accesses `okOf` allows and `O` lies inside global memory, every byte
    after its run is the one before with its byte stores made in order. -/
theorem eval_replay {I O : UInt64 → Prop} {okC : WStmt → WState → Prop} (a : UInt64) :
    ∀ (s : WStmt) (f : Nat) (ws : WState), okOf I O okC f s ws → (∀ b, O b → b.toNat < ws.gmem.size) →
      some ((s.eval f ws).gmem.getD a.toNat 0).toUInt64 =
        replay (pairsOf (gstOf f s ws)) a (some (ws.gmem.getD a.toNat 0).toUInt64) := by
  intro s
  induction s with
  | seq x y ihx ihy =>
      intro f ws hok hO; simp only [okOf] at hok
      simp only [WStmt.eval, gstOf, pairsOf, List.map_append, replay_append]
      rw [ihy f _ hok.2 (fun b hb => by rw [eval_size]; exact hO b hb), ihx f ws hok.1 hO]
  | uif c t e iht ihe =>
      intro f ws hok hO; simp only [okOf] at hok
      by_cases hc : (ws.regs c == 1) = true
      · simp only [hc, if_true] at hok; simp only [WStmt.eval, gstOf, hc, if_true]; exact iht f ws hok hO
      · simp only [hc, Bool.false_eq_true, if_false] at hok
        simp only [WStmt.eval, gstOf, hc, Bool.false_eq_true, if_false]; exact ihe f ws hok hO
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws _ _; simp [WStmt.eval, gstOf, replay]
      | succ f ihf =>
          intro ws hok hO; simp only [okOf] at hok
          by_cases hc : (ws.regs c == 1) = true
          · simp only [WStmt.eval, gstOf, hc, if_true, pairsOf, List.map_append, replay_append]
            rw [ihf _ (hok hc).2 (fun x hx => by rw [eval_size]; exact hO x hx), ih f ws (hok hc).1 hO]
          · simp [WStmt.eval, gstOf, hc, replay]
  | stgB addr val =>
      intro f ws hok hO; simp only [okOf] at hok
      have hlt := hO _ hok.2.2
      simp only [WStmt.eval, WState.stgByte, gstOf, pairsOf, List.map_cons, List.map_nil, replay, List.foldl_cons,
        List.foldl_nil]
      by_cases e : ws.regs addr = a
      · subst e
        rw [if_pos rfl]
        simp only [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds,
          Array.getElem?_setIfInBounds_self_of_lt hlt, Option.getD_some]
        congr 1
        apply UInt64.toNat_inj.mp
        rw [UInt64.toNat_and, UInt64.toNat_and, show (0xff : UInt64).toNat = 2 ^ 8 - 1 from rfl,
          Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod]
        simp
      · rw [if_neg e]
        have hne : (ws.regs addr).toNat ≠ a.toNat := fun h => e (UInt64.toNat_inj.mp h)
        simp only [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_ne hne]
  | coopCopy dst src len =>
      intro f ws hok hO; simp only [okOf] at hok
      obtain ⟨hn, hs, hdn, hIs, hI⟩ := hok
      simp only [WStmt.eval, gstOf]
      rw [replay_copy _ _ _ _ _ _ hdn, copyGmem_getD _ _ _ _ (fun i j hi hj e => by
        have h1 := hIs i (by omega)
        have h2 := (hI j hj).1
        apply h2
        have : ws.regs src + UInt64.ofNat i = ws.regs dst + UInt64.ofNat j := by
          apply UInt64.toNat_inj.mp
          rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show i < 2 ^ 64 by omega),
            UInt64.toNat_ofNat_of_lt' (show j < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega),
            Nat.mod_eq_of_lt (by omega)]
          exact e
        rw [← this]; exact h1)]
      by_cases hk : (ws.regs dst).toNat ≤ a.toNat ∧ a.toNat < (ws.regs dst).toNat + (ws.regs len).toNat
      · have hlt : a.toNat < ws.gmem.size := by
          have := hO _ (hI (a.toNat - (ws.regs dst).toNat) (by omega)).2.2
          rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show a.toNat - (ws.regs dst).toNat < 2 ^ 64 by omega),
            Nat.mod_eq_of_lt (by omega)] at this
          omega
        rw [if_pos ⟨hk.1, hk.2, hlt⟩, if_pos hk]
      · rw [if_neg (fun h => hk ⟨h.1, h.2.1⟩), if_neg hk]
  | coopWindow =>
      intro f ws _ _
      rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_gmem]; simp [gstOf, replay]
  | coopExtendStep => intro f ws _ _; rw [WStmt.eval.eq_12]; simp [gstOf, replay]; rfl
  | _ => intro f ws _ _; simp [WStmt.eval, WState.setReg, gstOf, replay]

/-- Byte stores elsewhere leave a byte. -/
theorem replay_filter (a : UInt64) : ∀ (L : List (UInt64 × UInt64)) (d : Option UInt64),
    replay L a d = replay (L.filter (·.1 = a)) a d
  | [], _ => rfl
  | x :: L, d => by
      by_cases e : x.1 = a
      · simp only [replay, List.foldl_cons, List.filter_cons, e, decide_true, if_true] at *
        exact replay_filter a L _
      · simp only [replay, List.foldl_cons, List.filter_cons, e, decide_false, if_false, Bool.false_eq_true] at *
        exact replay_filter a L _

/-- Byte stores all of one byte leave it. -/
theorem replay_const {a v : UInt64} : ∀ (L : List (UInt64 × UInt64)) (d : Option UInt64),
    (∀ x ∈ L, x = (a, v)) → L ≠ [] → replay L a d = some (v &&& 0xff)
  | [], _, _, h => absurd rfl h
  | x :: L, d, hL, _ => by
      obtain rfl := hL x List.mem_cons_self
      simp only [replay, List.foldl_cons, if_true]
      cases L with
      | nil => rfl
      | cons y L =>
          exact replay_const (y :: L) _ (fun z hz => hL z (List.mem_cons_of_mem _ hz)) (List.cons_ne_nil _ _)

/-- Every byte store of the warp is made by every lane or by one of the 32. -/
theorem gstOf_who : ∀ (s : WStmt) (f : Nat) (ws : WState), ∀ x ∈ gstOf f s ws, x.1 = none ∨ ∃ l, l < 32 ∧ x.1 = some l := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro f ws x hx; simp only [gstOf, List.mem_append] at hx
      rcases hx with h | h
      · exact iha f ws x h
      · exact ihb f _ x h
  | uif c t e iht ihe =>
      intro f ws x hx
      by_cases hc : (ws.regs c == 1) = true
      · simp only [gstOf, hc, if_true] at hx; exact iht f ws x hx
      · simp only [gstOf, hc, Bool.false_eq_true, if_false] at hx; exact ihe f ws x hx
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws x hx; simp [gstOf] at hx
      | succ f ihf =>
          intro ws x hx
          by_cases hc : (ws.regs c == 1) = true
          · simp only [gstOf, hc, if_true, List.mem_append] at hx
            rcases hx with h | h
            · exact ih f ws x h
            · exact ihf _ x h
          · simp [gstOf, hc] at hx
  | stgB => intro f ws x hx; simp [gstOf] at hx; subst hx; exact .inl rfl
  | coopCopy =>
      intro f ws x hx
      simp only [gstOf, copyTr, List.mem_map, List.mem_range] at hx
      obtain ⟨i, -, rfl⟩ := hx
      exact .inr ⟨i % 32, Nat.mod_lt _ (by decide), rfl⟩
  | _ => intro f ws x hx; simp [gstOf] at hx

/-- A trace writing each byte once has at most one store to each byte. -/
theorem filter_once {b : UInt64} : ∀ {tr : List (Option Nat × UInt64 × UInt64)},
    tr.Pairwise (fun x y => x.2.1 ≠ y.2.1) →
      tr.filter (fun x => decide (x.2.1 = b)) = [] ∨ ∃ x, tr.filter (fun x => decide (x.2.1 = b)) = [x]
  | [], _ => .inl rfl
  | x :: l, h => by
      obtain ⟨hx, hl⟩ := List.pairwise_cons.mp h
      by_cases e : x.2.1 = b
      · refine .inr ⟨x, ?_⟩
        rw [List.filter_cons, if_pos (by simpa using e), List.filter_eq_nil_iff.mpr fun y hy hb => hx y hy
          (by rw [e]; exact (of_decide_eq_true hb).symm)]
      · rw [List.filter_cons, if_neg (by simpa using e)]; exact filter_once hl

theorem lanePart_filter (t : Nat) (b : UInt64) (tr : List (Option Nat × UInt64 × UInt64)) :
    (lanePart t tr).filter (fun x => decide (x.1 = b)) = lanePart t (tr.filter fun x => decide (x.2.1 = b)) := by
  induction tr with
  | nil => rfl
  | cons x tr ih =>
      obtain ⟨who, c, v⟩ := x
      simp only [lanePart, List.filterMap_cons, List.filter_cons] at ih ⊢
      by_cases hw : who = none ∨ who = some t <;> by_cases hc : c = b <;> simp [hw, hc, ih]

/-- **Where the warp writes each byte once, the lanes' shares in thread order
    leave each byte as the warp's order does.** -/
theorem replay_lanes {tr : List (Option Nat × UInt64 × UInt64)} (hwo : tr.Pairwise (fun x y => x.2.1 ≠ y.2.1))
    (hwho : ∀ x ∈ tr, x.1 = none ∨ ∃ l, l < 32 ∧ x.1 = some l) (a : UInt64) (d : Option UInt64) :
    replay ((List.range 32).flatMap fun t => lanePart t tr) a d = replay (pairsOf tr) a d := by
  rw [replay_filter a ((List.range 32).flatMap fun t => lanePart t tr), replay_filter a (pairsOf tr),
    List.filter_flatMap]
  simp only [lanePart_filter]
  have hpf : (pairsOf tr).filter (fun x => decide (x.1 = a)) = pairsOf (tr.filter fun x => decide (x.2.1 = a)) := by
    simp only [pairsOf, List.filter_map]; rfl
  rw [hpf]
  rcases filter_once (b := a) hwo with h | ⟨⟨who, c, v⟩, h⟩
  · rw [h]
    simp only [lanePart, List.filterMap_nil, pairsOf, List.map_nil]
    rw [show (List.range 32).flatMap (fun _ => ([] : List (UInt64 × UInt64))) = [] by simp]
  · rw [h]
    have hc : c = a := by
      have : (who, c, v) ∈ tr.filter fun x => decide (x.2.1 = a) := by rw [h]; exact List.mem_cons_self
      simpa using (List.mem_filter.mp this).2
    subst hc
    have hmem : (who, c, v) ∈ tr := by
      have : (who, c, v) ∈ tr.filter fun x => decide (x.2.1 = c) := by rw [h]; exact List.mem_cons_self
      exact (List.mem_filter.mp this).1
    rw [show pairsOf [(who, c, v)] = [(c, v)] from rfl]
    rw [replay_const (a := c) (v := v) _ _ (fun x hx => by
        simp only [List.mem_flatMap, lanePart, List.filterMap_cons, List.filterMap_nil] at hx
        obtain ⟨t, -, hx⟩ := hx
        by_cases hw : who = none ∨ who = some t
        · simp only [hw, if_true] at hx; simpa using hx
        · simp only [hw, if_false] at hx; cases hx) ?_]
    · simp [replay]
    · rcases hwho _ hmem with h0 | ⟨l, hl, h0⟩
      · simp only at h0; subst h0
        exact List.ne_nil_of_mem (a := (c, v)) (List.mem_flatMap.mpr ⟨0, List.mem_range.mpr (by decide), by simp [lanePart]⟩)
      · simp only at h0; subst h0
        exact List.ne_nil_of_mem (a := (c, v)) (List.mem_flatMap.mpr ⟨l, List.mem_range.mpr hl, by simp [lanePart]⟩)

/-- Byte stores of other blocks, each into its own output, leave a byte of
    block `c`'s output to block `c`'s stores. -/
theorem replay_blocks {n c : Nat} (hc : c < n) {O : Nat → UInt64 → Prop} (S : Nat → List (UInt64 × UInt64))
    (hS : ∀ c', c' < n → ∀ x ∈ S c', O c' x.1) (hdis : ∀ c', c' < n → c' ≠ c → ∀ a, O c' a → ¬ O c a)
    {a : UInt64} (ha : O c a) (d : Option UInt64) :
    replay ((List.range n).flatMap S) a d = replay (S c) a d := by
  rw [replay_filter a ((List.range n).flatMap S), replay_filter a (S c), List.filter_flatMap]
  congr 1
  have : ∀ c', c' ∈ List.range n → (S c').filter (fun x => decide (x.1 = a)) =
      if c' = c then (S c).filter (fun x => decide (x.1 = a)) else [] := by
    intro c' hc'
    by_cases e : c' = c
    · subst e; simp
    · rw [if_neg e, List.filter_eq_nil_iff]
      intro x hx hxa
      exact hdis c' (List.mem_range.mp hc') e a (by rw [← of_decide_eq_true hxa]; exact hS c' (List.mem_range.mp hc') x hx) ha
  rw [flatMap_congr_mem this]
  induction n with
  | zero => omega
  | succ n ih =>
      rw [List.range_succ, List.flatMap_append]
      by_cases e : n = c
      · subst e
        have : (List.range n).flatMap (fun c' => if c' = n then (S n).filter (fun x => decide (x.1 = a)) else []) = [] := by
          rw [List.flatMap_eq_nil_iff]; intro c' hc'; rw [if_neg (by have := List.mem_range.mp hc'; omega)]
        rw [this]; simp
      · simp only [List.flatMap_cons, List.flatMap_nil, if_neg e, List.append_nil]
        exact ih (by omega) (fun c' h => hS c' (by omega)) (fun c' h => hdis c' (by omega))
          (fun c' hc' => this c' (List.mem_range.mpr (by have := List.mem_range.mp hc'; omega)))

set_option maxHeartbeats 2000000 in
/-- **A launch of the compressor leaves each block's output as the warp's
    run of that block does**, on the Code and on the PTX it compiles to: given,
    for each block, that the warp's run of its body makes only the accesses
    `okOf` allows and writes each byte once, its input and output hold `G c`'s
    bytes at the start and its output lies inside `G c`, its output may be
    stored, and the blocks' outputs meet neither each other nor a later
    block's input. Every byte outside the outputs is as it was. -/
theorem launch_output (hw : Hw) (hhw : PtxSound.HwOk hw) (name : String) {e : PtxIr.Entry}
    {numBlocks inStride outStride lenOff hashLog f : Nat} {inP outA : UInt64}
    (he : (compressorKernel name numBlocks inStride outStride lenOff hashLog).entry = .ok e)
    (I O : Nat → UInt64 → Prop) (G : Nat → Array UInt8) (bufs : Nat → ByteArray)
    (hnb : numBlocks * 32 + 32 < 2 ^ 32) (h5 : 5 ≤ hashLog) (hh : 2 * 2 ^ hashLog + 64 < 2 ^ 36) (hf : f ≤ 2 ^ 64)
    (hfine : Fine (emitGo (fnBody (idParams ++ [.i64, .i64]) (ML.Body.body 2 (compressorS numBlocks inStride outStride lenOff hashLog)))
      ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2)
    (hflags : ∀ r, r ∈ ["inBase","outBase","tbl","op","searchPos","litAnchor"] →
      (flagRegs (compressorBodyWStmt inStride lenOff hashLog)).contains r = false)
    (hIna : ∀ c a, I c a → NotArena a) (hOna : ∀ c a, O c a → NotArena a)
    (hok : ∀ c, c < numBlocks → okOf (I c) (O c) (okLZ4 (I c)) f (compressorBodyWStmt inStride lenOff hashLog)
      (startWS numBlocks inStride outStride hashLog c inP (G c)))
    (hwo : ∀ c, c < numBlocks → (gstOf f (compressorBodyWStmt inStride lenOff hashLog)
      (startWS numBlocks inStride outStride hashLog c inP (G c))).Pairwise (fun x y => x.2.1 ≠ y.2.1))
    (hI : ∀ c, c < numBlocks → ∀ a, I c a ∨ O c a →
      (launchWorld bufs hw.nan).mem.load a 1 = some ((G c).getD a.toNat 0).toUInt64)
    (hOsz : ∀ c, c < numBlocks → ∀ a, O c a → a.toNat < (G c).size)
    (hO : ∀ c, c < numBlocks → ∀ a, O c a → Writable (launchWorld bufs hw.nan).mem a 1)
    (hdis : ∀ c c', c' < c → c < numBlocks → ∀ a, O c' a → ¬ I c a)
    (hOdis : ∀ c c', c < numBlocks → c' < numBlocks → c' ≠ c → ∀ a, O c' a → ¬ O c a) :
    ∃ u : World, LaunchRuns (compressorKernel name numBlocks inStride outStride lenOff hashLog) numBlocks 32
        [.sc .i64 inP, .sc .i64 outA] bufs u.mem.bufs hw ∧
      RunsLaunch (PtxRun.runner e hw) (compressorKernel name numBlocks inStride outStride lenOff hashLog).smem numBlocks 32
        [.sc .i64 inP, .sc .i64 outA] bufs u.mem.bufs hw.nan ∧
      (∀ c, c < numBlocks → ∀ a, O c a → u.mem.load a 1 = some (((compressorBodyWStmt inStride lenOff hashLog).eval f
        (startWS numBlocks inStride outStride hashLog c inP (G c))).gmem.getD a.toNat 0).toUInt64) ∧
      ∀ a, NotArena a → (∀ c, c < numBlocks → ¬ O c a) → u.mem.load a 1 = (launchWorld bufs hw.nan).mem.load a 1 := by
  obtain ⟨u, hL, hP, hum⟩ := launch_runs (outA := outA) hw hhw name he I O G bufs hnb h5 hh hf hfine hflags hIna hOna hok
    (fun c hc a ha => hI c hc a (.inl ha)) hO hdis
  refine ⟨u, hL, hP, fun c hc a ha => ?_, fun a hna hnO => by
    rw [hum a hna, replay_notin]
    intro x hx
    simp only [List.mem_flatMap, List.mem_range] at hx
    obtain ⟨c, hc, hx⟩ := hx
    intro e
    exact hnO c hc (e ▸ blockStores_O (hok c hc) x hx)⟩
  rw [hum a (hOna c a ha), replay_blocks hc _ (fun c' hc' x hx => blockStores_O (hok c' hc') x hx)
    (fun c' hc' hne a' h' => hOdis c c' hc hc' hne a' h') ha, blockStores,
    replay_lanes (hwo c hc) (gstOf_who _ _ _), hI c hc a (.inr ha),
    eval_replay a _ f _ (hok c hc) (fun b hb => by
      show b.toNat < (G c).size
      exact hOsz c hc b hb)]
  rfl

end Lz4Code
