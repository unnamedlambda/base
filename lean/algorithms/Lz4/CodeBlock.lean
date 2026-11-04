module
public import Lz4.CodeThread
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeBlock` — a block of the compressor

A block's 32 threads, run in order, each answered as the warp's run answers
(`blockAns`: the barrier after the table clear, then the body's tape): each
thread clears its share of the table (`clearL_run`), passes the barrier and
runs the body as its lane does (`lane_run`, `thread_ends`), from a world whose
input bytes are the warp's (the warp's run leaves them, `eval_keeps`) and
whose output bytes may be stored; each thread's collectives and shared stores
are the warp's traces for it, so the laws give the answers back (`warp_laws`)
and the block runs (`block_runs`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel

namespace Lz4Code

/-- Lane `t`'s share of clearing an `N`-entry table, entries `32·m + t` on,
    as shared stores tagged `nb`. -/
def clearSt (N t m nb : Nat) : List (Nat × UInt64 × Nat × UInt64) :=
  (List.range' (32 * m) (N - 32 * m)).filterMap fun j =>
    if j % 32 = t then some (nb, addrOf .arena 0 + (c64 0 + (UInt64.ofNat j <<< c64 1)), 2, c64 0 &&& 0xffff) else none

/-- A lane's share of the table clear makes no byte store. -/
theorem clearL_nobyte {sm tbl n msk : UInt64} : ∀ k ci (w w' : World), clearL sm tbl n msk k ci w = some w' → NoByte w w'
  | 0, ci, w, w', h => by
      simp only [clearL] at h
      split at h
      · cases h
      · cases h; exact NoByte.refl _
  | k + 1, ci, w, w', h => by
      simp only [clearL] at h
      split at h
      · obtain ⟨w1, hw1, h⟩ := Option.bind_eq_some_iff.mp h
        refine NoByte.trans ?_ (clearL_nobyte k _ w1 w' h)
        obtain ⟨m, -, rfl⟩ := stw_some hw1
        exact ⟨[_], rfl, rfl⟩
      · cases h; exact NoByte.refl _

/-- **A lane's share of the table clear finishes**: within shared memory, it
    makes no collective, its shared stores its share (`clearSt`), and it leaves
    every byte outside shared memory and every byte that may be stored. -/
theorem clearL_run {t : Nat} (ht : t < 32) {N : Nat} (hN : 2 * N + 64 < 2 ^ 36) {msk : UInt64}
    (hmsk : ∀ j, j < N → UInt64.ofNat (2 * j) &&& msk = UInt64.ofNat (2 * j)) {start : Nat} :
    ∀ k m (w : World), N ≤ 32 * m + 32 * k → 32 * m ≤ N + 32 → start ≤ w.obs.length → 2 * N ≤ w.mem.arena.size →
      ∃ w', clearL (addrOf .arena 0) (c64 0) (c64 N) msk k (UInt64.ofNat (32 * m + t)) w = some w' ∧
        collCalls w' start = collCalls w start ∧
        sharedStores w' start = sharedStores w start ++ clearSt N t m ((collCalls w start).countP (·.1 == .barrier)) ∧
        w'.mem.arena.size = w.mem.arena.size ∧ (∀ a, NotArena a → w'.mem.load a 1 = w.mem.load a 1) ∧
        (∀ a, Writable w.mem a → Writable w'.mem a) ∧ start ≤ w'.obs.length := by
  have hc : c64 N = UInt64.ofNat N := c64_ofNat N
  intro k
  induction k with
  | zero =>
      intro m w hk hm hs hsz
      have : ¬ UInt64.ofNat (32 * m + t) < c64 N := by
        rw [hc, UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' (show 32 * m + t < 2 ^ 64 by omega),
          UInt64.toNat_ofNat_of_lt' (show N < 2 ^ 64 by omega)]; omega
      refine ⟨w, by simp only [clearL, if_neg this], rfl, ?_, rfl, fun _ _ => rfl, fun _ h => h, hs⟩
      simp [clearSt, show N - 32 * m = 0 by omega]
  | succ k ih =>
      intro m w hk hm hs hsz
      have hnext : UInt64.ofNat (32 * m + t) + c64 32 = UInt64.ofNat (32 * (m + 1) + t) := by
        rw [show c64 32 = UInt64.ofNat 32 by decide, ← UInt64.ofNat_add]; congr 1; omega
      have hsplit := filterMap_blocks N t ht (fun j => ((collCalls w start).countP (·.1 == .barrier),
        addrOf .arena 0 + (c64 0 + (UInt64.ofNat j <<< c64 1)), 2, c64 0 &&& 0xffff)) m
      by_cases hlt : 32 * m + t < N
      · have hlt' : UInt64.ofNat (32 * m + t) < c64 N := by
          rw [hc, UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' (show 32 * m + t < 2 ^ 64 by omega),
            UInt64.toNat_ofNat_of_lt' (show N < 2 ^ 64 by omega)]; exact hlt
        have haddr : c64 0 + (UInt64.ofNat (32 * m + t) <<< c64 1) = UInt64.ofNat (2 * (32 * m + t)) := by
          rw [show c64 0 = 0 by decide, UInt64.zero_add]
          apply UInt64.toNat_inj.mp
          rw [show c64 1 = UInt64.ofNat 1 by decide, shl_toNat (by decide) (by
              rw [UInt64.toNat_ofNat_of_lt' (show 32 * m + t < 2 ^ 64 by omega)]; omega),
            UInt64.toNat_ofNat_of_lt' (show 32 * m + t < 2 ^ 64 by omega),
            UInt64.toNat_ofNat_of_lt' (show 2 * (32 * m + t) < 2 ^ 64 by omega), Nat.shiftLeft_eq]
          omega
        have hmk : (c64 0 + (UInt64.ofNat (32 * m + t) <<< c64 1)) &&& msk =
            c64 0 + (UInt64.ofNat (32 * m + t) <<< c64 1) := by rw [haddr]; exact hmsk _ hlt
        have hd := arena_decode (a := c64 0 + (UInt64.ofNat (32 * m + t) <<< c64 1))
          (by rw [haddr, UInt64.toNat_ofNat_of_lt' (show 2 * (32 * m + t) < 2 ^ 64 by omega)]; omega)
        have hreg : (w.mem.region .arena).size = w.mem.arena.size := rfl
        obtain ⟨m', hm'⟩ := store_some_of (v := c64 0 &&& 0xffff) (n := 2) hd (by
          rw [hreg, haddr, UInt64.toNat_ofNat_of_lt' (show 2 * (32 * m + t) < 2 ^ 64 by omega)]; omega) (by decide)
        have hsame := (store_frame hm').choose_spec.choose_spec.2.same
        have hsz' : m'.arena.size = w.mem.arena.size := by
          have := hsame.size .arena; simp only [Mem.region] at this; exact this.symm
        have hs1 : start ≤ (obsStore w (addrOf .arena 0 + (c64 0 + (UInt64.ofNat (32 * m + t) <<< c64 1))) 2
            (c64 0 &&& 0xffff)).obs.length := by simp [obsStore]; omega
        obtain ⟨w', hw', hc', hst', hsz'', hld, hwr, hs'⟩ := ih (m + 1)
          { obsStore w (addrOf .arena 0 + (c64 0 + (UInt64.ofNat (32 * m + t) <<< c64 1))) 2 (c64 0 &&& 0xffff) with
            mem := m' } (by omega) (by omega) (by simpa using hs1) (by show 2 * N ≤ m'.arena.size; omega)
        have hcalls1 : collCalls ({ obsStore w (addrOf .arena 0 + (c64 0 + (UInt64.ofNat (32 * m + t) <<< c64 1))) 2
            (c64 0 &&& 0xffff) with mem := m' } : World) start = collCalls w start :=
          collCalls_obsStore hs _ _ _
        refine ⟨w', ?_, by rw [hc', hcalls1], ?_, by rw [hsz'']; exact hsz', fun a ha => ?_,
          fun a h => hwr a (h.after_store hm'), hs'⟩
        · simp only [clearL, if_pos hlt', hmk, stw, hm', Option.bind_eq_bind, Option.bind_some]
          rw [hnext]; exact hw'
        · rw [hst', sharedStores_store hs hd, hcalls1, List.append_assoc]
          congr 1
          simp only [clearSt]
          rw [hsplit, if_pos hlt]
        · rw [hld a ha]
          show m'.load a 1 = _
          rw [store_setArena hd hm', load_set_arena ha]
      · have hlt' : ¬ UInt64.ofNat (32 * m + t) < c64 N := by
          rw [hc, UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' (show 32 * m + t < 2 ^ 64 by omega),
            UInt64.toNat_ofNat_of_lt' (show N < 2 ^ 64 by omega)]; exact hlt
        refine ⟨w, by simp only [clearL, if_neg hlt'], rfl, ?_, rfl, fun _ _ => rfl, fun _ h => h, hs⟩
        rw [clearSt, hsplit, if_neg hlt, show N - 32 * (m + 1) = 0 by omega]; simp

/-- **The warp's run leaves `I`**: where it makes only the accesses `okOf`
    allows, every byte of `I` is as it was. -/
theorem eval_keeps {I O : UInt64 → Prop} {okC : WStmt → WState → Prop} {a : UInt64} (ha : I a) :
    ∀ (s : WStmt) (f : Nat) (ws : WState), okOf I O okC f s ws →
      (s.eval f ws).gmem.getD a.toNat 0 = ws.gmem.getD a.toNat 0 := by
  intro s
  induction s with
  | seq x y ihx ihy =>
      intro f ws hok; simp only [okOf] at hok
      simp only [WStmt.eval]; rw [ihy f _ hok.2, ihx f _ hok.1]
  | uif c t e iht ihe =>
      intro f ws hok; simp only [okOf] at hok
      by_cases hc : (ws.regs c == 1) = true
      · simp only [hc, if_true] at hok; simp only [WStmt.eval, hc, if_true]; exact iht f ws hok
      · simp only [hc, Bool.false_eq_true, if_false] at hok
        simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]; exact ihe f ws hok
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws _; simp [WStmt.eval]
      | succ f ihf =>
          intro ws hok; simp only [okOf] at hok
          by_cases hc : (ws.regs c == 1) = true
          · simp only [WStmt.eval, hc, if_true]; rw [ihf _ (hok hc).2, ih f ws (hok hc).1]
          · simp only [WStmt.eval, hc, Bool.false_eq_true, if_false]
  | stgB addr val =>
      intro f ws hok; simp only [okOf] at hok
      simp only [WStmt.eval, WState.stgByte]
      exact getD_set!_ne (fun e => hok.1 (by rw [UInt64.toNat_inj.mp e]; exact ha))
  | coopCopy dst src len =>
      intro f ws hok; simp only [okOf] at hok
      obtain ⟨hn, -, -, -, hI⟩ := hok
      simp only [WStmt.eval]
      rw [copyGmem_getD_out]
      intro i hi e
      have hdi : a = ws.regs dst + UInt64.ofNat i := by
        apply UInt64.toNat_inj.mp
        rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show i < 2 ^ 64 by omega), ← e]
        exact (Nat.mod_eq_of_lt a.toNat_lt).symm
      exact (hI i hi).1 (hdi ▸ ha)
  | coopWindow => intro f ws _; rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_gmem]
  | coopExtendStep => intro f ws _; rw [WStmt.eval.eq_12]; rfl
  | _ => intro f ws _; simp [WStmt.eval, WState.setReg]

theorem sharedStores_self (w : World) : sharedStores w w.obs.length = [] := by
  unfold sharedStores
  rw [show w.obs.reverse.drop w.obs.length = [] from List.drop_eq_nil_of_le (by simp)]
  rfl

theorem gStores_self (w : World) : gStores w w.obs.length = [] := by
  unfold gStores
  rw [show w.obs.reverse.drop w.obs.length = [] from List.drop_eq_nil_of_le (by simp)]
  rfl

theorem clearSt_tags {N t m nb : Nat} : ∀ x ∈ clearSt N t m nb, x.1 = nb := by
  intro x hx
  simp only [clearSt, List.mem_filterMap] at hx
  obtain ⟨j, -, hj⟩ := hx
  split at hj
  · cases hj; rfl
  · cases hj

/-- The warp's answers to its collectives are 32-bit words. -/
theorem words_trace : ∀ (s : WStmt) (f : Nat) (ws : WState), Words (trace lz4Ans f s ws) := by
  have happ : ∀ {a b : List Answer}, Words a → Words b → Words (a ++ b) := fun ha hb x hx v hv => by
    rcases List.mem_append.mp hx with h | h
    · exact ha x h v hv
    · exact hb x h v hv
  intro s
  induction s with
  | seq a b iha ihb => intro f ws; simp only [trace]; exact happ (iha f ws) (ihb f _)
  | uif c t e iht ihe =>
      intro f ws
      by_cases hc : (ws.regs c == 1) = true
      · simp only [trace, hc, if_true]; exact iht f ws
      · simp only [trace, hc, Bool.false_eq_true, if_false]; exact ihe f ws
  | uwhile c b ih =>
      intro f
      induction f with
      | zero => intro ws; simp [trace, Words]
      | succ f ihf =>
          intro ws
          by_cases hc : (ws.regs c == 1) = true
          · simp only [trace, hc, if_true]; exact happ (ih f ws) (ihf _)
          · simp [trace, hc, Words]
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      intro f ws x hx v hv
      simp only [trace, lz4Ans, winAns, List.mem_cons, List.mem_nil_iff, or_false] at hx
      rcases hx with rfl | rfl | rfl
      · cases hv; exact ⟨_, rfl⟩
      · cases hv
      · cases hv
        refine ⟨(winCand ws sp inStride hashLog tbl
          (firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0 % 32)).toUInt32, ?_⟩
        congr 1
        apply UInt64.toNat_inj.mp
        simp only [s32, UInt32.toNat_toUInt64, UInt64.toNat_toUInt32]
        exact (Nat.mod_eq_of_lt winCand_lt).symm
  | coopExtendStep =>
      intro f ws x hx v hv
      simp only [trace, lz4Ans, extAns, List.mem_cons, List.mem_nil_iff, or_false] at hx
      subst hx; cases hv; exact ⟨_, rfl⟩
  | _ => intro f ws; simp [trace, Words]

/-- Shared memory after the table clear: every lane's share stored into the
    block's zeroed shared memory. -/
def clearMem (hashLog : Nat) : ByteArray :=
  (snapMem ((List.range 32).flatMap fun t => clearSt (2 ^ hashLog) t 0 0)
    (ByteArray.mk (Array.replicate (2 ^ hashLog * 2) 0)) 1).arena

/-- The warp's state a block's body starts from: registers holding the
    block's bases, global memory `G`, the table after the clear. -/
def startWS (numBlocks inStride outStride hashLog cta : Nat) (inP : UInt64) (G : Array UInt8) : WState :=
  { regs := regs0 (inP + UInt64.ofNat cta * c64 inStride)
      (inP + c64 (numBlocks * inStride + copySlack) + UInt64.ofNat cta * c64 outStride),
    gmem := G, smem := (clearMem hashLog).data }

/-- The warp's answers to each of a block's threads: the barrier after the
    clear its shared memory, then the body's tape. -/
def blockAns (hashLog f : Nat) (body : WStmt) (ws : WState) : List Answer :=
  .smem (clearMem hashLog) :: trace lz4Ans f body ws

set_option maxHeartbeats 4000000 in
/-- **A block of the compressor runs**: its 32 threads, answered as the
    warp's run answers (`blockAns`), each end, and the laws give those answers
    back; given that the warp's run of the body, from registers holding the
    block's bases and the table after the clear, makes only the accesses
    `okOf` allows, `I` holds the warp's input bytes and `O`'s bytes may be
    stored. -/
theorem block_runs (hw : Hw) (name : String) {numBlocks inStride outStride lenOff hashLog cta f : Nat}
    {inP outA : UInt64} {I O : UInt64 → Prop} {G : Array UInt8} (w : World)
    (hcta : cta < numBlocks) (hc32 : cta * 32 + 32 < 2 ^ 32) (hnb : numBlocks < 2 ^ 63)
    (hh : 2 * 2 ^ hashLog + 64 < 2 ^ 36)
    (hfine : Fine (emitGo (fnBody (idParams ++ [.i64, .i64]) (compressor numBlocks inStride outStride lenOff hashLog))
      ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2)
    (hflags : ∀ r, r ∈ ["inBase","outBase","tbl","op","searchPos","litAnchor"] →
      (flagRegs (compressorBodyWStmt inStride lenOff hashLog)).contains r = false)
    (hIna : ∀ a, I a → NotArena a) (hOna : ∀ a, O a → NotArena a)
    (hok : okOf I O (okLZ4 I) f (compressorBodyWStmt inStride lenOff hashLog)
      (startWS numBlocks inStride outStride hashLog cta inP G))
    (hI : ∀ a, I a → w.mem.load a 1 = some (G.getD a.toNat 0).toUInt64)
    (hO : ∀ a, O a → Writable w.mem a) :
    ∃ u, BlockRuns (codeKernel name numBlocks inStride outStride lenOff hashLog) cta 32
        [.sc .i64 inP, .sc .i64 outA] hw w u ∧
      (∀ a, NotArena a → u.mem.load a 1 = replay ((List.range 32).flatMap fun t =>
        lanePart t (gstOf f (compressorBodyWStmt inStride lenOff hashLog)
          (startWS numBlocks inStride outStride hashLog cta inP G))) a (w.mem.load a 1)) ∧
      ∀ a, NotArena a → Writable w.mem a → Writable u.mem a := by
  generalize hbody : compressorBodyWStmt inStride lenOff hashLog = body at hok hflags
  generalize hws : startWS numBlocks inStride outStride hashLog cta inP G = ws0 at hok
  let A := blockAns hashLog f body ws0
  let smem0 := ByteArray.mk (Array.replicate (2 ^ hashLog * 2) 0)
  let N := 2 ^ hashLog
  let Inv : Nat → PassSt → Prop := fun t s =>
    (∀ a, I a → s.1.mem.load a 1 = some (G.getD a.toNat 0).toUInt64) ∧
    (∀ a, NotArena a → Writable w.mem a → Writable s.1.mem a) ∧
    (∀ a, NotArena a → s.1.mem.load a 1 =
      replay ((List.range t).flatMap fun t' => lanePart t' (gstOf f body ws0)) a (w.mem.load a 1)) ∧
    ∀ t', t' < t → s.2.1 t' = (.barrier, []) :: trace (lz4Calls t') f body ws0 ∧
      s.2.2 t' = clearSt N t' 0 0 ++ strace (lz4Calls t') (lz4Stores t') f 1 body ws0
  have hWA : Words A := by
    intro x hx v hv
    rcases List.mem_cons.mp hx with rfl | h
    · cases hv
    · exact words_trace body f ws0 x h v hv
  have hstep : ∀ t, t < 32 → ∀ s : PassSt, Inv t s → ∃ steps u,
      thread (codeKernel name numBlocks inStride outStride lenOff hashLog).code A
        (threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA] t)
        { s.1 with mem := { s.1.mem with arena := smem0 } } hw steps = .ok u ∧
      Inv (t + 1) (u, fun t' => if t' = t then collCalls u s.1.obs.length else s.2.1 t',
        fun t' => if t' = t then sharedStores u s.1.obs.length else s.2.2 t') := by
    intro t ht s hs
    obtain ⟨hIs, hOs, hMs, hCs⟩ := hs
    generalize hW' : ({ s.1 with mem := { s.1.mem with arena := smem0 } } : World) = W'
    have hW'obs : W'.obs = s.1.obs := by rw [← hW']
    have hW'I : ∀ a, I a → W'.mem.load a 1 = some (G.getD a.toNat 0).toUInt64 := fun a ha => by
      rw [← hW']; show ({ s.1.mem with arena := smem0 } : Mem).load a 1 = _
      rw [load_set_arena (hIna a ha)]; exact hIs a ha
    have hW'N : ∀ a, NotArena a → Writable w.mem a → Writable W'.mem a := fun a ha hw => by
      rw [← hW']; exact (hOs a ha hw).arena ha _
    have hW'O : ∀ a, O a → Writable W'.mem a := fun a ha => hW'N a (hOna a ha) (hO a ha)
    have hW'sz : 2 * N ≤ W'.mem.arena.size := by
      rw [← hW']; show 2 * N ≤ (Array.replicate (2 ^ hashLog * 2) (0 : UInt8)).size; simp [N]; omega
    obtain ⟨w1, hcl, hcc, hcs, -, hcld, hcwr, hcs1⟩ := clearL_run (t := t) ht (N := N) hh (msk := c64 ((2 * 2 ^ hashLog - 1 : Nat) : Int)) (fun j hj => by
      rw [c64_ofNat, show 2 * 2 ^ hashLog - 1 = 2 ^ (hashLog + 1) - 1 by rw [Nat.pow_succ]; omega]
      apply UInt64.toNat_inj.mp
      have hjN : 2 * j < 2 ^ (hashLog + 1) := by rw [Nat.pow_succ]; omega
      rw [UInt64.toNat_and, UInt64.toNat_ofNat_of_lt' (show 2 * j < 2 ^ 64 by omega),
        UInt64.toNat_ofNat_of_lt' (show 2 ^ (hashLog + 1) - 1 < 2 ^ 64 by rw [Nat.pow_succ]; omega),
        Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hjN]) (start := W'.obs.length)
      (N / 32 + 1) 0 W' (by omega) (by omega) (Nat.le_refl _) hW'sz
    rw [show 32 * 0 + t = t by omega] at hcl
    rw [collCalls_self] at hcc hcs
    have hbar : A[(collCalls w1 W'.obs.length).length]? = some (.smem (clearMem hashLog)) := by rw [hcc]; rfl
    let st0 : LSt := ⟨regs0 (inP + UInt64.ofNat cta * c64 inStride)
        (inP + c64 (numBlocks * inStride + copySlack) + UInt64.ofNat cta * c64 outStride),
      { obsCall w1 (.intr .barrier) [] with mem := { (obsCall w1 (.intr .barrier) []).mem with arena := clearMem hashLog } }⟩
    have hR0 : Rel I ws0 st0 := by
      rw [← hws]
      refine ⟨rfl, rfl, fun a ha => ?_⟩
      show ({ (obsCall w1 (.intr .barrier) []).mem with arena := clearMem hashLog } : Mem).load a 1 = _
      rw [load_set_arena (hIna a ha)]
      show w1.mem.load a 1 = _
      rw [hcld a (hIna a ha)]; exact hW'I a ha
    have hs0 : W'.obs.length ≤ st0.w.obs.length := by
      show W'.obs.length ≤ (obsCall w1 (.intr .barrier) []).obs.length
      simp [obsCall_length]; omega
    have hc0 : collCalls st0.w W'.obs.length = [(.barrier, [])] := by
      show collCalls (obsCall w1 (.intr .barrier) []) W'.obs.length = _
      rw [collCalls_call hcs1, hcc]; rfl
    have hsh0 : sharedStores st0.w W'.obs.length = clearSt N t 0 0 := by
      show sharedStores (callW w1 .barrier []) W'.obs.length = _
      rw [sharedStores_bar hcs1, hcs]; simp [sharedStores_self]
    have hwr0 : ∀ a, O a → Writable st0.w.mem a := fun a ha =>
      (hcwr a (hW'O a ha)).arena (hOna a ha) _
    have htape : A.drop (collCalls st0.w W'.obs.length).length = trace lz4Ans f body ws0 ++ [] := by
      rw [hc0, List.append_nil]; rfl
    obtain ⟨st', hrun, hRel, hgs, hcalls, hstores, -, htr⟩ := lane_run (ans := A) (start := W'.obs.length) ht hIna hOna
      body f ws0 st0 [] hok hR0 hs0 hwr0 htape
    rw [hc0] at hcalls
    rw [hsh0, hc0] at hstores
    obtain ⟨steps, u, hth, rfl⟩ := thread_ends hw hWA name ht hcta hc32 hnb hfine (by rw [hbody]; exact hflags)
      hcl hbar (by rw [hbody]; exact hrun)
    have hgs0 : gStores st0.w W'.obs.length = [] := by
      obtain ⟨L, hL, h0⟩ := clearL_nobyte _ _ _ _ hcl
      show gStores (obsCall w1 (.intr .barrier) []) W'.obs.length = []
      rw [gStores_ext hcs1 (L := [_]) rfl, gStores_ext (Nat.le_refl _) hL]
      simp only [stores1] at h0
      simp [gStores_self, List.filterMap_reverse, h0]
    refine ⟨steps, st'.w, hth, fun a ha => ?_, fun a ha => ?_, fun a ha => ?_, fun t' ht' => ?_⟩
    · rw [hRel.gin a ha, eval_keeps ha body f ws0 hok, ← hws]; rfl
    · intro hw
      exact laneRun_writable ha _ _ _ _ hrun ((hcwr a (hW'N a ha hw)).arena ha _)
    · obtain ⟨L, hL, hm⟩ := htr
      have hst : stores1 L.reverse = lanePart t (gstOf f body ws0) := by
        have := gStores_ext hs0 hL
        rw [hgs, hgs0] at this
        simp only [stores1, List.filterMap_reverse]; simpa using this.symm
      rw [hm a ha, hst]
      show replay _ a (({ (obsCall w1 (.intr .barrier) []).mem with arena := clearMem hashLog } : Mem).load a 1) = _
      rw [load_set_arena ha]
      show replay _ a (w1.mem.load a 1) = _
      rw [hcld a ha, ← hW']
      show replay _ a (({ s.1.mem with arena := smem0 } : Mem).load a 1) = _
      rw [load_set_arena ha, hMs a ha, List.range_succ, List.flatMap_append, replay_append]
      simp
    · by_cases e : t' = t
      · subst e
        simp only [if_true]
        rw [← hW'obs]
        refine ⟨hcalls, ?_⟩
        rw [hstores]; simp [bars, show (Intr.barrier == Intr.barrier) = true by decide]
      · simp only [e, if_false]; exact hCs t' (by omega)
  have hInv0 : Inv 0 (({ w with mem := { w.mem with arena := smem0 } } : World), fun _ => [], fun _ => []) :=
    ⟨fun a ha => by
      show ({ w.mem with arena := smem0 } : Mem).load a 1 = _
      rw [load_set_arena (hIna a ha)]; exact hI a ha,
     fun a ha hw => hw.arena ha _,
     fun a ha => by
       show ({ w.mem with arena := smem0 } : Mem).load a 1 = _
       rw [load_set_arena ha]; rfl,
     fun _ h => absurd h (Nat.not_lt_zero _)⟩
  obtain ⟨S0, r, hr⟩ := pass_ends (k := codeKernel name numBlocks inStride outStride lenOff hashLog) (smem0 := smem0)
    (args := threadArgs cta 32 [.sc .i64 inP, .sc .i64 outA]) (ans := fun _ => A) (hw := hw) Inv 32 _ hInv0 hstep
  obtain ⟨hfold, hIr⟩ := hr S0 (Nat.le_refl _)
  obtain ⟨w', F, Sx⟩ := r
  obtain ⟨-, hWr, hMr, hFS⟩ := hIr
  replace hFS : ∀ t', t' < 32 → F t' = (.barrier, []) :: trace (lz4Calls t') f body ws0 ∧
      Sx t' = clearSt N t' 0 0 ++ strace (lz4Calls t') (lz4Stores t') f 1 body ws0 := hFS
  -- the laws give the answers back
  have hBefore : Before F Sx f body ws0 1 1 := by
    refine ⟨fun l hl => ⟨[(.barrier, [])], [], by rw [(hFS l hl).1]; simp, rfl, by simp [bars, show (Intr.ballot == Intr.barrier) = false by decide, show (Intr.barrier == Intr.barrier) = true by decide, show (Intr.shflIdx == Intr.barrier) = false by decide]⟩,
      fun l hl => ⟨clearSt N l 0 0, [], by rw [(hFS l hl).2]; simp, fun x hx => by rw [clearSt_tags x hx]; omega,
        fun x hx => by simp at hx⟩⟩
  have hS1 : ∀ t, t < 32 → ∃ E, Sx t = clearSt N t 0 0 ++ E ∧ ∀ x ∈ E, 1 ≤ x.1 :=
    fun t ht => ⟨_, (hFS t ht).2, fun x hx => (strace_tags t body f 1 ws0 x hx).1⟩
  have hsnap1 : snapMem ((List.range 32).flatMap Sx) smem0 1 =
      snapMem ((List.range 32).flatMap fun t => clearSt N t 0 0) smem0 1 :=
    (snapMem_congr 1 (fun b hb => stores_filter hS1 hb)).symm
  have hsnap : ws0.smem = (snapMem ((List.range 32).flatMap Sx) smem0 1).arena.data := by
    rw [hsnap1, ← hws]; rfl
  obtain ⟨hL, -⟩ := warp_laws (I := I) (O := O) body f ws0 1 1 hok hBefore hsnap
  have hans1 : ∀ t, t < 32 → answersOf F (snapshots 32 smem0 F Sx) smem0 t = .ok A := by
    intro t ht
    refine answersOf_of_get ?_ fun k a hk => ?_
    · rw [(hFS t ht).1]; simp [A, blockAns, trace_len]
    · cases k with
      | zero =>
        simp [A, blockAns] at hk; subst hk
        have hlen : 0 < ((F 0).filter (·.1 == .barrier)).length := by rw [(hFS 0 (by decide)).1]; simp [show (Intr.ballot == Intr.barrier) = false by decide, show (Intr.barrier == Intr.barrier) = true by decide, show (Intr.shflIdx == Intr.barrier) = false by decide]
        have hsn := snapshots_get (n := 32) (smem0 := smem0) (calls := F) (stores := Sx) (by decide : 0 < 32) hlen
        refine ⟨(.barrier, []), 1, by rw [(hFS t ht).1]; rfl, ?_⟩
        simp only [List.take_zero, List.countP_nil, answerStep, pure, Except.pure, hsn, Option.getD_some]
        rw [hsnap1]; rfl
      | succ i =>
        have := hL t ht i a (by simpa [A, blockAns] using hk)
        rwa [Nat.add_comm] at this
  have hans : Kernel.answers 32 smem0 F Sx = .ok (fun t => ((List.range 32).map fun _ => A).getD t []) := by
    unfold Kernel.answers
    rw [mapM_range_ok _ (fun _ => A) 32 hans1]
    rfl
  refine ⟨_, ⟨S0, fun _ => A, w', F, Sx, _, hfold, hans, fun t ht => by simp [List.getD_eq_getElem?_getD, ht], rfl⟩,
    fun a ha => ?_, fun a ha hw => (hWr a ha hw).arena ha _⟩
  show ({ w'.mem with arena := w.mem.arena } : Mem).load a 1 = _
  rw [load_set_arena ha, hMr a ha, ← hws]

end Lz4Code

