module
public import Host.KernelCorpus
meta import Host.KernelCorpus
public import AlgorithmLib.Kernel.Elementwise
meta import AlgorithmLib.Kernel.Elementwise
public import AlgorithmLib.Kernel.Warp
meta import AlgorithmLib.Kernel.Warp
public import AlgorithmLib.Proof.LibStep
meta import AlgorithmLib.Proof.LibStep
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# warpSum, a kernel whose threads meet

The corpus's `warpSum` kernel (`KernelCorpus.warpSum`, which
`tools/kernel_agree.py` runs on the GPU) proven: each lane adds what five
butterfly shuffles answer it to its word of `x`, and lane 0 of each warp
stores the result at `out[w]`.

One thread's body is proven for any answers (`thread_wp`, `thread_spec`): its
shuffles pass its running sums (`sums`), each adding what it was answered.
In a block the answers are the laws' (`BlockRuns`), each the running sum
the partner lane passed, so every lane's running sums are the butterfly's
values over its warp (`warp_sums`, by induction on the rounds). A launch
(`launch_spec`) leaves at `out[w]` the butterfly's sum of warp `w`'s 32 words
(`bfly`), in f32 arithmetic on words.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts AlgorithmLib.LibSpec AlgorithmLib.Kernel

namespace KernelWarpSum

def params : List ClifTy := [.i64, .i64]

def K : Kernel := { name := "warpSum", params, code := emit KernelCorpus.warpSum (idParams ++ params) }

def masks : List Nat := [16, 8, 4, 2, 1]

/-- A lane's running sums from `x`, answered `o j` at its `j`-th shuffle, on a
    platform whose NaN rule is `nan`. -/
def sums (nan : ClifTy → UInt64 → UInt64) (x : UInt32) (o : Nat → UInt32) : Nat → UInt32
  | 0 => x
  | j + 1 => add32N nan (sums nan x o j) (o j)

/-- The shuffles a lane makes: its `j`-th passes its `j`-th running sum. -/
def callsOf (nan : ClifTy → UInt64 → UInt64) (x : UInt32) (o : Nat → UInt32) : List (Intr × List V) :=
  (List.range 5).map fun j => (.shflBfly, [.sc .i32 (s32 (sums nan x o j)), .sc .i32 (s32 (UInt32.ofNat (masks.getD j 0)))])

attribute [local irreducible] wp

/-- How many collectives a thread has made, through the calls since. -/
macro "ncoll" : tactic => `(tactic| (simp (disch := first | omega | (simp only [obsCall_length]; omega)) only
  [collCalls_call, collCalls_self, Intr.isColl, if_true, List.nil_append, List.length_append, List.length_cons,
   List.length_nil]))

/-- The word lane 0 of warp `i / 32` stores its sum at. -/
abbrev outAddr (po : UInt64) (i : UInt32) : UInt64 := po + (s32 i >>> (c64 5 % 64)) <<< (c64 2 % 64)

/-- **What one warpSum thread does**, answered `o j` at its `j`-th shuffle:
    it passes its running sums to the shuffles, and lane 0 of each warp stores
    the last at `out[i / 32]`. -/
theorem thread_wp (cfg : Cfg) (w : World) (b t bd : UInt32) (px po sm : UInt64) (ans : List Answer)
    (hA : AnswersFrom ans w.obs.length cfg) (o : Nat → UInt32)
    (ho : ∀ j, fitTo .shflBfly (valAt ans j) = .sc .i32 (s32 (o j))) (X : UInt32) (m : Mem)
    (hx : w.mem.load (px + (s32 (b * bd + t) <<< (c64 2 % 64))) 4 = some (s32 X))
    (hs : (b * bd + t) &&& 31 = 0 → w.mem.store (outAddr po (b * bd + t)) 4 (s32 (sums w.mem.nan X o 5)) = some m) :
    wp cfg 0 { ok := fun _ _ => True, faultOk := false } (fnBody (idParams ++ params) KernelCorpus.warpSum)
      (fun _ _ u => collCalls u w.obs.length = callsOf w.mem.nan X o ∧
        if (b * bd + t) &&& 31 = 0 then u.mem = m else u.mem = w.mem)
      #[.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm, .sc .i64 px, .sc .i64 po] w := by
  rw [fnBody_of (by decide +kernel)]
  simp only [KernelCorpus.warpSum, KernelCorpus.gid, entryParams, wp_bind, wp_params, wp_ret]
  simp only [idParams, params, List.cons_append, List.nil_append, carriesFrom, Vals.head, Vals.snd, Vals.thd,
    Vals.tail, Vals.drop, Nat.zero_add, Nat.reduceAdd]
  have h4 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po])[4]? = some (.sc .i64 px) := rfl
  have h5 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po])[5]? = some (.sc .i64 po) := rfl
  refine wp_imul32 (x := b) (y := bd) rfl rfl ?_
  refine wp_iadd32 (x := b * bd) (y := t) (by look) rfl ?_
  repeat step
  refine wp_loadF32 (by look) hx ?_
  simp only [List.forIn_cons, List.forIn_nil]
  repeat step
  refine wp_coll2 hA (c := .shflBfly) rfl KernelCorpus.shflBfly rfl (by look) (by look)
    (by first | omega | (simp only [obsCall_length]; omega)) (by ncoll) (ho 0) ?_
  repeat step
  refine wp_fadd32w (nan := w.mem.nan) rfl (by look) (by look) ?_
  repeat step
  refine wp_coll2 hA (c := .shflBfly) rfl KernelCorpus.shflBfly rfl (by look) (by look)
    (by first | omega | (simp only [obsCall_length]; omega)) (by ncoll) (ho 1) ?_
  repeat step
  refine wp_fadd32w (nan := w.mem.nan) rfl (by look) (by look) ?_
  repeat step
  refine wp_coll2 hA (c := .shflBfly) rfl KernelCorpus.shflBfly rfl (by look) (by look)
    (by first | omega | (simp only [obsCall_length]; omega)) (by ncoll) (ho 2) ?_
  repeat step
  refine wp_fadd32w (nan := w.mem.nan) rfl (by look) (by look) ?_
  repeat step
  refine wp_coll2 hA (c := .shflBfly) rfl KernelCorpus.shflBfly rfl (by look) (by look)
    (by first | omega | (simp only [obsCall_length]; omega)) (by ncoll) (ho 3) ?_
  repeat step
  refine wp_fadd32w (nan := w.mem.nan) rfl (by look) (by look) ?_
  repeat step
  refine wp_coll2 hA (c := .shflBfly) rfl KernelCorpus.shflBfly rfl (by look) (by look)
    (by first | omega | (simp only [obsCall_length]; omega)) (by ncoll) (ho 4) ?_
  repeat step
  refine wp_fadd32w (nan := w.mem.nan) rfl (by look) (by look) ?_
  repeat step
  simp only [when, wp_bind]
  refine wp_ifteT (Qj := fun _ _ u => collCalls u w.obs.length = callsOf w.mem.nan X o ∧
        if (b * bd + t) &&& 31 = 0 then u.mem = m else u.mem = w.mem) rfl (by look) (by look) ?_ ?_ ?_
  · intro hc
    rw [cmp32_eq] at hc
    have h0 : (b * bd + t) &&& 31 = 0 := by simpa using hc
    repeat step
    refine wp_storeF32 (m := m) (by look) (by look) (by simp only [obsCall_mem]; exact hs h0) ?_
    rw [wp_pure]
    refine ⟨?_, by simp [h0]⟩
    change collCalls (obsStore _ _ _ _) _ = _
    rw [collCalls_obsStore (by simp only [obsCall_length]; omega)]
    ncoll
    rfl
  · intro hc ne hne
    rw [cmp32_eq] at hc
    have h0 : ¬ (b * bd + t) &&& 31 = 0 := by simpa using hc
    rw [wp_pure]
    constructor
    · ncoll
      rfl
    · rw [if_neg h0]; simp only [obsCall_mem]
  · intro nJ Γ' jv vs w' hle hvs hq
    rw [wp_pure]
    exact hq

/-- Lane `l`'s value after `j` rounds of the butterfly, on the words `X`
    from `base`. -/
def bfly (nan : ClifTy → UInt64 → UInt64) (X : Nat → UInt32) (base : Nat) : Nat → Nat → UInt32
  | 0, l => X (base + l)
  | j + 1, l => add32N nan (bfly nan X base j l) (bfly nan X base j ((l ^^^ masks.getD j 0) % 32))


theorem masks_le : ∀ j, j < 5 → masks.getD j 0 ≤ 16 := by decide

theorem callsOf_at (nan : ClifTy → UInt64 → UInt64) (x : UInt32) (o : Nat → UInt32) {j : Nat} (hj : j < 5) :
    (callsOf nan x o)[j]? = some (.shflBfly, [.sc .i32 (s32 (sums nan x o j)), .sc .i32 (s32 (UInt32.ofNat (masks.getD j 0)))]) := by
  simp [callsOf, hj]

/-- **The warp law, settled**: when every thread of a block passes its running
    sums and is answered what the laws give back, every lane's running sums are
    the butterfly's values over its warp. -/
theorem warp_sums {bd : Nat} (hbd : bd % 32 = 0) {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray}
    {smem0 : ByteArray} {ans : Nat → List Answer} (nan : ClifTy → UInt64 → UInt64) (Xl : Nat → UInt32)
    (hc : ∀ t, t < bd → calls t = callsOf nan (Xl t) (wordAt (ans t)))
    (ha : ∀ t, t < bd → answersOf calls snaps smem0 t = .ok (ans t)) :
    ∀ j, j ≤ 5 → ∀ t, t < bd → sums nan (Xl t) (wordAt (ans t)) j = bfly nan Xl (32 * (t / 32)) j (t % 32)
  | 0, _, t, ht => by simp only [sums, bfly]; congr 1; omega
  | j + 1, hj, t, ht => by
    have ih := warp_sums hbd nan Xl hc ha j (by omega)
    have hm := masks_le j (by omega)
    obtain ⟨args, hl, hans⟩ := answersOf_bfly (ha t ht) (by rw [hc t ht]; exact callsOf_at (j := j) _ _ _ (by omega))
    have hmt : (s32 (UInt32.ofNat (masks.getD j 0))).toNat = masks.getD j 0 := by
      simp only [s32, UInt32.toNat_toUInt64]; exact toNat_ofNat32 (by omega)
    rw [hmt] at hl
    have hp : 32 * (t / 32) + (t % 32 ^^^ masks.getD j 0) % 32 < bd := by
      have : (t % 32 ^^^ masks.getD j 0) % 32 < 32 := Nat.mod_lt _ (by decide)
      omega
    unfold laneArg at hl
    rw [hc _ hp, callsOf_at _ _ _ (by omega)] at hl
    dsimp only at hl
    have hb : (Intr.shflBfly == Intr.shflBfly) = true := rfl
    rw [if_pos hb] at hl
    cases hl
    have hw : wordAt (ans t) j = sums nan (Xl (32 * (t / 32) + (t % 32 ^^^ masks.getD j 0) % 32))
        (wordAt (ans (32 * (t / 32) + (t % 32 ^^^ masks.getD j 0) % 32))) j := by
      rw [lane32_s32] at hans
      exact wordAt_val hans
    show add32N nan (sums nan (Xl t) (wordAt (ans t)) j) (wordAt (ans t) j) = _
    rw [hw, ih t ht, ih _ hp]
    simp only [bfly]
    congr 2 <;> omega

theorem fine : Fine (emitGo (fnBody (idParams ++ params) KernelCorpus.warpSum) ⟨(idParams ++ params).length, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- **What a warpSum thread leaves**, when it finishes: its shuffles pass its
    running sums from `x[i]`, each adding what it was answered, and lane 0 of
    each warp stores the last at `out[i / 32]`. -/
theorem thread_spec (b t bd : UInt32) (px po sm : UInt64) (w u : World) (ans : List Answer) (hw : Words ans)
    (X : UInt32) (m : Mem)
    (hx : w.mem.load (px + (s32 (b * bd + t) <<< (c64 2 % 64))) 4 = some (s32 X))
    (hs : (b * bd + t) &&& 31 = 0 → w.mem.store (outAddr po (b * bd + t)) 4 (s32 (sums w.mem.nan X (wordAt ans) 5)) = some m)
    {steps : Nat} (hr : thread K.code ans [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po] w Hw.rounded steps = .ok u) :
    collCalls u w.obs.length = callsOf w.mem.nan X (wordAt ans) ∧
      if (b * bd + t) &&& 31 = 0 then u.mem = m else u.mem = w.mem := by
  refine thread_of_triple_ans (Q := fun u => collCalls u w.obs.length = callsOf w.mem.nan X (wordAt ans) ∧
      if (b * bd + t) &&& 31 = 0 then u.mem = m else u.mem = w.mem) (fun cfg hA _ => ?_) hr
  have h1 := emit_triple (cfg := cfg) (J := { ok := fun _ _ => True, faultOk := false })
    (P0 := At (List.toArray [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po]) w)
    (PT.conseq (P' := fun Γ' w' => Γ' = (List.toArray [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po]) ∧ w' = w)
      (wp_sound cfg 0 _ (fnBody (idParams ++ params) KernelCorpus.warpSum) _)
      (fun Γ' w' h => by
        rw [h.1, h.2]
        exact thread_wp cfg w b t bd px po sm ans hA _ (fitAt_words hw rfl) X m hx hs)
      (fun _ _ _ h => h)) fine (fun Γ' w' h => ⟨by rw [h.1]; rfl, h⟩)
  exact h1

/-! ## A launch -/

/-- `x` and `out`: device buffers 0 and 1. -/
abbrev pxw : UInt64 := addrOf (.buf ⟨0, by decide⟩) 0
abbrev pow : UInt64 := addrOf (.buf ⟨1, by decide⟩) 0
def args : List V := [.sc .i64 pxw, .sc .i64 pow]

theorem out_eq (i : UInt32) : (s32 i >>> (c64 5 % 64)) <<< (c64 2 % 64) = UInt64.ofNat (4 * (i.toNat / 32)) := by
  rw [show c64 5 % 64 = (5 : UInt64) by decide, show c64 2 % 64 = (2 : UInt64) by decide]
  apply UInt64.toNat_inj.mp
  have := i.toNat_lt
  simp [s32, UInt64.toNat_shiftLeft, UInt64.toNat_shiftRight, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
  omega

theorem decode_out (i : UInt32) :
    decodeAddr (outAddr pow i) = some (.buf ⟨1, by decide⟩, 4 * (i.toNat / 32)) := by
  have := i.toNat_lt
  unfold outAddr; rw [out_eq]; exact decode_buf _ _ (by omega)

theorem lane0_iff {n : Nat} (hn : n < 2 ^ 32) : (UInt32.ofNat n &&& 31 = 0) ↔ n % 32 = 0 := by
  rw [← UInt32.toNat_inj, UInt32.toNat_and, toNat_ofNat32 hn]
  rw [show (UInt32.toNat 31) = 2 ^ 5 - 1 by rfl, Nat.and_two_pow_sub_one_eq_mod]
  simp

/-- **warpSum as an elementwise kernel over warps**: lane 0 of each warp,
    reading its warp's 32 words of `x`, stores the butterfly's sum of them at
    `out[j / 32]`. -/
def elem (X : Nat → UInt32) : Elem where
  out j := outAddr pow (UInt32.ofNat j)
  val j := s32 (bfly gpuNaN X (32 * (j / 32)) 5 0)
  active j := decide (j % 32 = 0)
  reads j := (List.range 32).map fun l => (pxw + off (UInt32.ofNat (j + l)), s32 (X (j + l)))

theorem elem_apart (X : Nat → UInt32) (N : Nat) (hN : N ≤ 2 ^ 32) : (elem X).Apart N where
  buf j _ _ := ⟨_, _, decode_out _⟩
  outs i j hi hj hne hai haj := by
    simp only [elem, decide_eq_true_eq] at hai haj
    refine ⟨_, _, _, _, decode_out _, decode_out _, Or.inr ?_⟩
    rw [toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)]; omega
  reads i j hi hj hne _ _ p hp := by
    simp only [elem, List.mem_map, List.mem_range] at hp
    obtain ⟨l, -, rfl⟩ := hp
    exact ⟨_, _, _, _, decode_out _, decode_elem _ _, Or.inl (by simp)⟩


theorem bfly_shift (nan : ClifTy → UInt64 → UInt64) (X : Nat → UInt32) (c base : Nat) :
    ∀ j l, bfly nan (fun i => X (c + i)) base j l = bfly nan X (c + base) j l
  | 0, l => by simp [bfly, Nat.add_assoc]
  | j + 1, l => by simp only [bfly, bfly_shift nan X c base j]

/-- **A warpSum block**: lane 0 of each of its warps leaves the butterfly's
    sum of the warp's words, the laws answering every shuffle. -/
theorem elem_blocks (X : Nat → UInt32) (grid bd : Nat) (hbd : bd % 32 = 0) (hsz : grid * bd ≤ 2 ^ 32) :
    Elem.Blocks K args grid bd (elem X) := by
  intro b hb w w' harena hnan hlent hpre hr
  have hw0 : ({ w with mem := { w.mem with arena := ByteArray.mk (Array.replicate K.smem 0) } } : World) = w :=
    world_arena w (by rw [harena]; rfl)
  obtain ⟨steps, ans, w'', calls, stores, ans', hfold, hans, hset, rfl⟩ := hr
  rw [hw0] at hfold
  have hlt : ∀ t, t < bd → b * bd + t < 2 ^ 32 := fun t ht => by
    have := Elem.lt_grid (grid := grid) hb ht; omega
  have hq : b * bd = 32 * (b * (bd / 32)) := by
    rw [Nat.mul_left_comm, Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hbd)]
  let Xl (t : Nat) : UInt32 := X (b * bd + t)
  -- what the block's memory keeps: `x`'s words, room at each lane 0's word, shared memory zero
  let Q (m : Mem) : Prop := m.arena = ByteArray.mk (Array.replicate K.smem 0) ∧
    (∀ t, t < bd → m.load (pxw + off (UInt32.ofNat (b * bd + t))) 4 = some (s32 (Xl t))) ∧
    (∀ t, t < bd → t % 32 = 0 → ∀ v, (m.store (outAddr pow (UInt32.ofNat (b * bd + t))) 4 v).isSome) ∧
    m.nan = gpuNaN ∧ m.lent = []
  let f (t : Nat) (m : Mem) : Mem :=
    if t < bd ∧ t % 32 = 0 then
      (m.store (outAddr pow (UInt32.ofNat (b * bd + t))) 4 (s32 (sums gpuNaN (Xl t) (wordAt (ans t)) 5))).getD m
    else m
  have hW : ∀ t, t < bd → Words (ans t) := fun t ht => by
    rw [← hset t ht]; exact answersOf_words (answers_at hans ht)
  have hQ0 : Q w.mem := by
    refine ⟨harena, fun t ht => ?_, fun t ht hl v => ?_, hnan, hlent⟩
    · have hp := hpre (32 * (t / 32)) (by omega) (by simp only [elem, decide_eq_true_eq]; omega)
      have := hp.1 _ (List.mem_map.mpr ⟨t % 32, List.mem_range.mpr (Nat.mod_lt _ (by decide)), rfl⟩)
      rw [show b * bd + 32 * (t / 32) + t % 32 = b * bd + t by omega] at this
      exact this
    · have hp := hpre t ht (by simp only [elem, decide_eq_true_eq]; omega)
      exact hp.2 v
  have hkeep : ∀ t u m, t ≠ u → (u < bd ∧ Q m) → (u < bd ∧ Q (f t m)) := by
    intro t u m _ ⟨hu, ha, hl, hs, hn, hln⟩
    refine ⟨hu, ?_⟩
    simp only [f]
    split
    · rename_i hc
      cases hst : m.store (outAddr pow (UInt32.ofNat (b * bd + t))) 4 (s32 (sums gpuNaN (Xl t) (wordAt (ans t)) 5)) with
      | none => exact ⟨ha, hl, hs, hn, hln⟩
      | some m' =>
        obtain ⟨_, _, _, hF⟩ := store_frame hst
        simp only [Option.getD_some]
        refine ⟨by rw [store_arena (decode_out _) hst]; exact ha, fun t' ht' => ?_, fun t' ht' hl' v => ?_,
          by rw [Mem.store_nan hst]; exact hn, by rw [Mem.store_lent hst]; exact hln⟩
        · rw [load_store_clear hst (decode_out _) (decode_elem _ _) (Or.inl (by simp)), hl t' ht']
        · rw [← MemSame.store_isSome hF.same _ 4 v v]; exact hs t' ht' hl' v
    · exact ⟨ha, hl, hs, hn, hln⟩
  obtain ⟨hmem, hcalls⟩ := pass_spec (fun t m => t < bd ∧ Q m) f
    (fun t _ cs _ => cs = callsOf gpuNaN (Xl t) (wordAt (ans t)))
    (fun t s u ⟨ht, hQ⟩ hu => by
      rw [world_arena s hQ.1] at hu
      by_cases hl : t % 32 = 0
      · obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp
          (hQ.2.2.1 t ht hl (s32 (sums gpuNaN (Xl t) (wordAt (ans t)) 5)))
        have h := thread_spec (UInt32.ofNat b) (UInt32.ofNat t) (UInt32.ofNat bd) pxw pow (addrOf .arena 0)
          s u (ans t) (hW t ht) (Xl t) m (by rw [gi_eq]; exact hQ.2.1 t ht)
          (fun _ => by rw [gi_eq, hQ.2.2.2.1]; exact hm) hu
        rw [hQ.2.2.2.1] at h
        rw [gi_eq, if_pos ((lane0_iff (hlt t ht)).mpr (by omega))] at h
        refine ⟨?_, h.1⟩
        rw [h.2]; simp only [f, if_pos (And.intro ht hl), hm, Option.getD_some]
      · have h := thread_spec (UInt32.ofNat b) (UInt32.ofNat t) (UInt32.ofNat bd) pxw pow (addrOf .arena 0)
          s u (ans t) (hW t ht) (Xl t) s.mem (by rw [gi_eq]; exact hQ.2.1 t ht)
          (fun h => absurd ((lane0_iff (hlt t ht)).mp (by rw [gi_eq] at h; exact h)) (by omega)) hu
        rw [hQ.2.2.2.1] at h
        rw [gi_eq, if_neg (fun h => by have := (lane0_iff (hlt t ht)).mp h; omega)] at h
        refine ⟨?_, h.1⟩
        rw [h.2]; simp only [f, hl, and_false, if_false])
    hkeep (List.range bd) _ _ List.nodup_range (fun t ht => ⟨List.mem_range.mp ht, hQ0⟩) hfold
  have hsum := warp_sums hbd gpuNaN Xl (fun t ht => hcalls t (List.mem_range.mpr ht))
    (fun t ht => by rw [← hset t ht]; exact answers_at hans ht) 5 (Nat.le_refl 5)
  have hf : ∀ t ∈ List.range bd, ∀ m, f t m = (elem X).eff (b * bd + t) m := by
    intro t ht m
    have ht := List.mem_range.mp ht
    simp only [f, Elem.eff, elem, put, decide_eq_true_eq]
    by_cases hl : t % 32 = 0
    · rw [if_pos (And.intro ht hl), if_pos (by omega), hsum t ht, hl]
      congr 3
      show bfly gpuNaN (fun i => X (b * bd + i)) _ 5 0 = _
      rw [bfly_shift]
      congr 1
      omega
    · rw [if_neg (fun h => hl h.2), if_neg (by omega)]
  show { w''.mem with arena := w.mem.arena } = _
  simp only at hmem
  rw [hmem, foldl_congr_mem _ _ _ _ hf]
  apply mem_arena_self
  rw [← List.foldl_map (f := (b * bd + ·)) (g := fun m j => (elem X).eff j m),
    Elem.foldl_eff_arena (elem_apart X (grid * bd) hsz) _ _ (fun j hj => by
      obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hj; exact Elem.lt_grid hb (List.mem_range.mp ht))]

/-- **What a launch of warpSum leaves**: after `grid` blocks of `bd` threads
    (whole warps), `out[w]` holds the butterfly's sum of warp `w`'s 32 words
    of `x`, each shuffle answered as the warp law says. -/
theorem launch_spec (X : Nat → UInt32) (grid bd : Nat) (hbd : bd % 32 = 0) (hsz : grid * bd ≤ 2 ^ 32)
    (bufs bufs' : Nat → ByteArray) (hpre : ∀ j, j < grid * bd → (elem X).Pre (kmem bufs) j)
    (hr : LaunchRuns K grid bd args bufs bufs') :
    ∀ wi, 32 * wi < grid * bd →
      ({ kmem bufs with bufs := bufs' } : Mem).load (outAddr pow (UInt32.ofNat (32 * wi))) 4 =
        some (s32 (bfly gpuNaN X (32 * wi) 5 0)) := by
  intro wi hw
  have h := Elem.launch_blocks (elem_apart X _ hsz) (elem_blocks X grid bd hbd hsz) bufs bufs' hpre
    (fun j _ _ => by simp only [elem, s32, UInt32.toNat_toUInt64]; exact UInt32.toNat_lt _) hr (32 * wi) hw
    (by simp [elem])
  simpa [elem] using h

end KernelWarpSum

