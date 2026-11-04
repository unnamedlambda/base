module
public import Host.KernelCorpus
meta import Host.KernelCorpus
public import AlgorithmLib.Kernel.Warp
meta import AlgorithmLib.Kernel.Warp
public import AlgorithmLib.Proof.LibStep
meta import AlgorithmLib.Proof.LibStep
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# blockSum, a kernel whose threads meet at barriers

The corpus's `blockSum` kernel (`KernelCorpus.blockSum`, which
`tools/kernel_agree.py` runs on the GPU) proven: each of a block's 256
threads stores its word of `x` in shared memory; then eight rounds, each a
barrier and every thread below the round's stride adding the word that far
above its own; and thread 0 stores word 0 at `out[b]`.

One thread is proven for any answers (`thread_wp`, `thread_spec`): its stores
to shared memory and barriers (`evs`), the words it adds read from what each
barrier answered. In a block the barriers are answered as the barrier law
says (`BlockRuns`): the `k`-th, shared memory with every thread's stores
before it applied (`snapshots_eq`, `phase_word`), so its words are the tree's
partial sums (`cum_word`, by induction on the rounds). A launch (`launch_spec`)
leaves at `out[b]` the tree sum of block `b`'s 256 words (`tree`), in f32
arithmetic on words.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts AlgorithmLib.LibSpec AlgorithmLib.Kernel

namespace KernelBlockSum

def params : List ClifTy := [.i64, .i64]

def K : Kernel := { name := "blockSum", params, code := emit KernelCorpus.blockSum (idParams ++ params), smem := 1024 }

/-- f32 addition on words as the GPU leaves it, a NaN its one NaN. -/
abbrev add32G (x y : UInt32) : UInt32 := add32N gpuNaN x y

/-- The stride of round `k`, 1 to 8: 128, 64, …, 1. -/
def str (k : Nat) : Nat := 256 >>> k

/-- Thread `t`'s shared stores and barriers, oldest first, through round `k`. -/
def evs (t : UInt32) (X : UInt32) (W : Nat → Nat → UInt32) : Nat → List Obs
  | 0 => [.store (smemAddr + off t) 4 (s32 X), .call (.intr .barrier) []]
  | k + 1 => evs t X W k ++
      (if t.toNat < str (k + 1) then
        [.store (smemAddr + off t) 4 (s32 (add32G (W k t.toNat) (W k (t.toNat + str (k + 1)))))] else []) ++
      [.call (.intr .barrier) []]

attribute [local irreducible] wp

theorem evs_colls (t X : UInt32) (W : Nat → Nat → UInt32) : ∀ k,
    (evs t X W k).filterMap collOf = List.replicate (k + 1) (Intr.barrier, []) := by
  intro k
  induction k with
  | zero => simp [evs, collOf, Intr.isColl]
  | succ k ih =>
    simp only [evs, List.filterMap_append, ih]
    split <;> simp [collOf, Intr.isColl, List.replicate_succ']

/-- **What one blockSum thread does**, its barriers answered with shared
    memories `S k` holding words `W k`. -/
theorem thread_wp (cfg : Cfg) (w : World) (b t bd : UInt32) (px po : UInt64) (ans : List Answer)
    (hA : AnswersFrom ans w.obs.length cfg)
    (hsh : ∀ (k : Nat) (B : ByteArray), ans[k]? = some (.smem B) → B.size = 1024)
    (X : UInt32) (mo : Mem) (ht : t.toNat < 256) (hnan : w.mem.nan = gpuNaN)
    (hx : w.mem.load (px + off (b * bd + t)) 4 = some (s32 X))
    (hs0 : 4 * t.toNat + 4 ≤ w.mem.arena.size)
    (hout : t = 0 → ({ w.mem with arena := smemAt ans 8 } : Mem).store (po + off b) 4
      (s32 (aword (smemAt ans 8) 0)) = some mo) :
    wp cfg 0 { ok := fun _ _ => True, faultOk := false } (fnBody (idParams ++ params) KernelCorpus.blockSum)
      (fun _ _ u => u.obs = (evs t X (fun k i => aword (smemAt ans k) i) 8 ++ if t = 0 then ([.store (po + off b) 4 (s32 (aword (smemAt ans 8) 0))] : List Obs) else []).reverse
          ++ w.obs ∧ u.mem = if t = 0 then mo else { w.mem with arena := smemAt ans 8 })
      #[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 smemAddr,
        .sc .i64 px, .sc .i64 po] w := by
  let W : Nat → Nat → UInt32 := fun k i => aword (smemAt ans k) i
  show wp _ _ _ _ (fun _ _ u => u.obs = (evs t X W 8 ++ if t = 0 then
      ([.store (po + off b) 4 (s32 (W 8 0))] : List Obs) else []).reverse ++ w.obs ∧
    u.mem = if t = 0 then mo else { w.mem with arena := smemAt ans 8 }) _ _
  rw [fnBody_of (by decide +kernel)]
  simp only [KernelCorpus.blockSum, KernelCorpus.gid, entryParams, wp_bind, wp_params, wp_ret]
  simp only [idParams, params, List.cons_append, List.nil_append, carriesFrom, Vals.head, Vals.snd, Vals.thd,
    Vals.tail, Vals.drop, Nat.zero_add, Nat.reduceAdd]
  have h0 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 smemAddr,
        .sc .i64 px, .sc .i64 po])[0]? = some (.sc .i32 (s32 b)) := rfl
  have h1 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 smemAddr,
        .sc .i64 px, .sc .i64 po])[1]? = some (.sc .i32 (s32 t)) := rfl
  have h2 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 smemAddr,
        .sc .i64 px, .sc .i64 po])[2]? = some (.sc .i32 (s32 bd)) := rfl
  have h3 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 smemAddr,
        .sc .i64 px, .sc .i64 po])[3]? = some (.sc .i64 smemAddr) := rfl
  have h4 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 smemAddr,
        .sc .i64 px, .sc .i64 po])[4]? = some (.sc .i64 px) := rfl
  have h5 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 smemAddr,
        .sc .i64 px, .sc .i64 po])[5]? = some (.sc .i64 po) := rfl
  repeat step
  refine wp_imul32 (x := b) (y := bd) (by look) (by look) ?_
  refine wp_iadd32 (x := b * bd) (y := t) (by look) (by look) ?_
  repeat step
  refine wp_loadF32 (by look) hx ?_
  refine wp_store_arena (by look) (by look) hs0 fun w0' hobs0' hmem0' => ?_
  refine wp_bar hA KernelCorpus.barrier rfl (by rw [hobs0']; simp) (k := 0)
    (by rw [collCalls_evs w _ [.store (smemAddr + off t) 4 (s32 X)] (by rw [hobs0']; rfl)]; rfl)
    fun w0 B0 hb0 hobs00 hmem00 => ?_
  have hobs0 : w0.obs = (evs t X W 0).reverse ++ w.obs := by rw [hobs00, hobs0']; simp [evs]
  have hmem0 : w0.mem = { w.mem with arena := smemAt ans 0 } := by
    rw [hmem00, hmem0']; simp [smemAt, hb0]
  simp only [List.forIn_cons, List.forIn_nil]
  -- round 1
  repeat step
  refine wp_when (R := fun w' => w'.obs = (evs t X W 0 ++ (if t.toNat < str 1 then
        ([.store (smemAddr + off t) 4 (s32 (add32G (W 0 t.toNat) (W 0 (t.toNat + str 1))))] : List Obs) else [])).reverse
          ++ w.obs ∧ w'.mem = { w.mem with arena := w'.mem.arena }) (by look) (by look) ?_ ?_ ?_
  · intro hc
    rw [cmp64_ult t 128 128 rfl (by decide)] at hc
    have hc : t.toNat < 128 := of_decide_eq_true hc
    rw [wp_bind]
    refine wp_loadF32 (p := smemAddr + off t) (v := s32 (W 0 t.toNat)) (by look) ?_ ?_
    · rw [load_arena, hmem0]; exact aload_word _ _ (by omega) (by rw [show smemAt ans 0 = B0 by simp [smemAt, hb0], hsh 0 B0 hb0]; omega)
    repeat step
    refine wp_loadF32 (p := smemAddr + off t + c64 (4 * 128)) (v := s32 (W 0 (t.toNat + 128))) (by look) ?_ ?_
    · rw [addr_step t 128 (4 * 128) rfl (by omega), load_arena, toNat_ofNat32 (by omega), hmem0]
      exact aload_word _ _ (by omega) (by rw [show smemAt ans 0 = B0 by simp [smemAt, hb0], hsh 0 B0 hb0]; omega)
    rw [wp_bind]
    refine wp_fadd32w (nan := gpuNaN) (by rw [hmem0]; exact hnan) (by look) (by look) ?_
    refine wp_store_arena (by look) (by look) (by rw [hmem0]; dsimp only; rw [show smemAt ans 0 = B0 by simp [smemAt, hb0], hsh 0 B0 hb0]; omega)
      fun ws hobsS hmemS => ?_
    refine ⟨by extc, by rw [hobsS, hobs0]; simp [evs, str, hc], by rw [hmemS, hmem0]⟩
  · intro hc
    rw [cmp64_ult t 128 128 rfl (by decide)] at hc
    have hc : ¬ t.toNat < 128 := of_decide_eq_false hc
    exact ⟨by rw [hobs0]; simp [evs, str, hc], by rw [hmem0]⟩
  · intro Γ1 w' hext ⟨hobs', hmem'⟩
    repeat step
    refine wp_bar hA KernelCorpus.barrier rfl (by rw [hobs']; simp; omega) (k := 1)
      (by rw [collCalls_evs w w' _ hobs', List.filterMap_append, evs_colls]; split <;> simp [collOf])
      fun w1 B1 hb1 hobs10 hmem10 => ?_
    have hobs1 : w1.obs = (evs t X W 1).reverse ++ w.obs := by rw [hobs10, hobs']; simp [evs]
    have hmem1 : w1.mem = { w.mem with arena := smemAt ans 1 } := by
      rw [hmem10, hmem']; simp [smemAt, hb1]
    -- round 2
    repeat step
    refine wp_when (R := fun w' => w'.obs = (evs t X W 1 ++ (if t.toNat < str 2 then
          ([.store (smemAddr + off t) 4 (s32 (add32G (W 1 t.toNat) (W 1 (t.toNat + str 2))))] : List Obs) else [])).reverse
            ++ w.obs ∧ w'.mem = { w.mem with arena := w'.mem.arena }) (by look) (by look) ?_ ?_ ?_
    · intro hc
      rw [cmp64_ult t 64 64 rfl (by decide)] at hc
      have hc : t.toNat < 64 := of_decide_eq_true hc
      rw [wp_bind]
      refine wp_loadF32 (p := smemAddr + off t) (v := s32 (W 1 t.toNat)) (by look) ?_ ?_
      · rw [load_arena, hmem1]; exact aload_word _ _ (by omega) (by rw [show smemAt ans 1 = B1 by simp [smemAt, hb1], hsh 1 B1 hb1]; omega)
      repeat step
      refine wp_loadF32 (p := smemAddr + off t + c64 (4 * 64)) (v := s32 (W 1 (t.toNat + 64))) (by look) ?_ ?_
      · rw [addr_step t 64 (4 * 64) rfl (by omega), load_arena, toNat_ofNat32 (by omega), hmem1]
        exact aload_word _ _ (by omega) (by rw [show smemAt ans 1 = B1 by simp [smemAt, hb1], hsh 1 B1 hb1]; omega)
      rw [wp_bind]
      refine wp_fadd32w (nan := gpuNaN) (by rw [hmem1]; exact hnan) (by look) (by look) ?_
      refine wp_store_arena (by look) (by look) (by rw [hmem1]; dsimp only; rw [show smemAt ans 1 = B1 by simp [smemAt, hb1], hsh 1 B1 hb1]; omega)
        fun ws hobsS hmemS => ?_
      refine ⟨by extc, by rw [hobsS, hobs1]; simp [evs, str, hc], by rw [hmemS, hmem1]⟩
    · intro hc
      rw [cmp64_ult t 64 64 rfl (by decide)] at hc
      have hc : ¬ t.toNat < 64 := of_decide_eq_false hc
      exact ⟨by rw [hobs1]; simp [evs, str, hc], by rw [hmem1]⟩
    · intro Γ2 w' hext ⟨hobs', hmem'⟩
      repeat step
      refine wp_bar hA KernelCorpus.barrier rfl (by rw [hobs']; simp; omega) (k := 2)
        (by rw [collCalls_evs w w' _ hobs', List.filterMap_append, evs_colls]; split <;> simp [collOf])
        fun w2 B2 hb2 hobs20 hmem20 => ?_
      have hobs2 : w2.obs = (evs t X W 2).reverse ++ w.obs := by rw [hobs20, hobs']; simp [evs]
      have hmem2 : w2.mem = { w.mem with arena := smemAt ans 2 } := by
        rw [hmem20, hmem']; simp [smemAt, hb2]
      -- round 3
      repeat step
      refine wp_when (R := fun w' => w'.obs = (evs t X W 2 ++ (if t.toNat < str 3 then
            ([.store (smemAddr + off t) 4 (s32 (add32G (W 2 t.toNat) (W 2 (t.toNat + str 3))))] : List Obs) else [])).reverse
              ++ w.obs ∧ w'.mem = { w.mem with arena := w'.mem.arena }) (by look) (by look) ?_ ?_ ?_
      · intro hc
        rw [cmp64_ult t 32 32 rfl (by decide)] at hc
        have hc : t.toNat < 32 := of_decide_eq_true hc
        rw [wp_bind]
        refine wp_loadF32 (p := smemAddr + off t) (v := s32 (W 2 t.toNat)) (by look) ?_ ?_
        · rw [load_arena, hmem2]; exact aload_word _ _ (by omega) (by rw [show smemAt ans 2 = B2 by simp [smemAt, hb2], hsh 2 B2 hb2]; omega)
        repeat step
        refine wp_loadF32 (p := smemAddr + off t + c64 (4 * 32)) (v := s32 (W 2 (t.toNat + 32))) (by look) ?_ ?_
        · rw [addr_step t 32 (4 * 32) rfl (by omega), load_arena, toNat_ofNat32 (by omega), hmem2]
          exact aload_word _ _ (by omega) (by rw [show smemAt ans 2 = B2 by simp [smemAt, hb2], hsh 2 B2 hb2]; omega)
        rw [wp_bind]
        refine wp_fadd32w (nan := gpuNaN) (by rw [hmem2]; exact hnan) (by look) (by look) ?_
        refine wp_store_arena (by look) (by look) (by rw [hmem2]; dsimp only; rw [show smemAt ans 2 = B2 by simp [smemAt, hb2], hsh 2 B2 hb2]; omega)
          fun ws hobsS hmemS => ?_
        refine ⟨by extc, by rw [hobsS, hobs2]; simp [evs, str, hc], by rw [hmemS, hmem2]⟩
      · intro hc
        rw [cmp64_ult t 32 32 rfl (by decide)] at hc
        have hc : ¬ t.toNat < 32 := of_decide_eq_false hc
        exact ⟨by rw [hobs2]; simp [evs, str, hc], by rw [hmem2]⟩
      · intro Γ3 w' hext ⟨hobs', hmem'⟩
        repeat step
        refine wp_bar hA KernelCorpus.barrier rfl (by rw [hobs']; simp; omega) (k := 3)
          (by rw [collCalls_evs w w' _ hobs', List.filterMap_append, evs_colls]; split <;> simp [collOf])
          fun w3 B3 hb3 hobs30 hmem30 => ?_
        have hobs3 : w3.obs = (evs t X W 3).reverse ++ w.obs := by rw [hobs30, hobs']; simp [evs]
        have hmem3 : w3.mem = { w.mem with arena := smemAt ans 3 } := by
          rw [hmem30, hmem']; simp [smemAt, hb3]
        -- round 4
        repeat step
        refine wp_when (R := fun w' => w'.obs = (evs t X W 3 ++ (if t.toNat < str 4 then
              ([.store (smemAddr + off t) 4 (s32 (add32G (W 3 t.toNat) (W 3 (t.toNat + str 4))))] : List Obs) else [])).reverse
                ++ w.obs ∧ w'.mem = { w.mem with arena := w'.mem.arena }) (by look) (by look) ?_ ?_ ?_
        · intro hc
          rw [cmp64_ult t 16 16 rfl (by decide)] at hc
          have hc : t.toNat < 16 := of_decide_eq_true hc
          rw [wp_bind]
          refine wp_loadF32 (p := smemAddr + off t) (v := s32 (W 3 t.toNat)) (by look) ?_ ?_
          · rw [load_arena, hmem3]; exact aload_word _ _ (by omega) (by rw [show smemAt ans 3 = B3 by simp [smemAt, hb3], hsh 3 B3 hb3]; omega)
          repeat step
          refine wp_loadF32 (p := smemAddr + off t + c64 (4 * 16)) (v := s32 (W 3 (t.toNat + 16))) (by look) ?_ ?_
          · rw [addr_step t 16 (4 * 16) rfl (by omega), load_arena, toNat_ofNat32 (by omega), hmem3]
            exact aload_word _ _ (by omega) (by rw [show smemAt ans 3 = B3 by simp [smemAt, hb3], hsh 3 B3 hb3]; omega)
          rw [wp_bind]
          refine wp_fadd32w (nan := gpuNaN) (by rw [hmem3]; exact hnan) (by look) (by look) ?_
          refine wp_store_arena (by look) (by look) (by rw [hmem3]; dsimp only; rw [show smemAt ans 3 = B3 by simp [smemAt, hb3], hsh 3 B3 hb3]; omega)
            fun ws hobsS hmemS => ?_
          refine ⟨by extc, by rw [hobsS, hobs3]; simp [evs, str, hc], by rw [hmemS, hmem3]⟩
        · intro hc
          rw [cmp64_ult t 16 16 rfl (by decide)] at hc
          have hc : ¬ t.toNat < 16 := of_decide_eq_false hc
          exact ⟨by rw [hobs3]; simp [evs, str, hc], by rw [hmem3]⟩
        · intro Γ4 w' hext ⟨hobs', hmem'⟩
          repeat step
          refine wp_bar hA KernelCorpus.barrier rfl (by rw [hobs']; simp; omega) (k := 4)
            (by rw [collCalls_evs w w' _ hobs', List.filterMap_append, evs_colls]; split <;> simp [collOf])
            fun w4 B4 hb4 hobs40 hmem40 => ?_
          have hobs4 : w4.obs = (evs t X W 4).reverse ++ w.obs := by rw [hobs40, hobs']; simp [evs]
          have hmem4 : w4.mem = { w.mem with arena := smemAt ans 4 } := by
            rw [hmem40, hmem']; simp [smemAt, hb4]
          -- round 5
          repeat step
          refine wp_when (R := fun w' => w'.obs = (evs t X W 4 ++ (if t.toNat < str 5 then
                ([.store (smemAddr + off t) 4 (s32 (add32G (W 4 t.toNat) (W 4 (t.toNat + str 5))))] : List Obs) else [])).reverse
                  ++ w.obs ∧ w'.mem = { w.mem with arena := w'.mem.arena }) (by look) (by look) ?_ ?_ ?_
          · intro hc
            rw [cmp64_ult t 8 8 rfl (by decide)] at hc
            have hc : t.toNat < 8 := of_decide_eq_true hc
            rw [wp_bind]
            refine wp_loadF32 (p := smemAddr + off t) (v := s32 (W 4 t.toNat)) (by look) ?_ ?_
            · rw [load_arena, hmem4]; exact aload_word _ _ (by omega) (by rw [show smemAt ans 4 = B4 by simp [smemAt, hb4], hsh 4 B4 hb4]; omega)
            repeat step
            refine wp_loadF32 (p := smemAddr + off t + c64 (4 * 8)) (v := s32 (W 4 (t.toNat + 8))) (by look) ?_ ?_
            · rw [addr_step t 8 (4 * 8) rfl (by omega), load_arena, toNat_ofNat32 (by omega), hmem4]
              exact aload_word _ _ (by omega) (by rw [show smemAt ans 4 = B4 by simp [smemAt, hb4], hsh 4 B4 hb4]; omega)
            rw [wp_bind]
            refine wp_fadd32w (nan := gpuNaN) (by rw [hmem4]; exact hnan) (by look) (by look) ?_
            refine wp_store_arena (by look) (by look) (by rw [hmem4]; dsimp only; rw [show smemAt ans 4 = B4 by simp [smemAt, hb4], hsh 4 B4 hb4]; omega)
              fun ws hobsS hmemS => ?_
            refine ⟨by extc, by rw [hobsS, hobs4]; simp [evs, str, hc], by rw [hmemS, hmem4]⟩
          · intro hc
            rw [cmp64_ult t 8 8 rfl (by decide)] at hc
            have hc : ¬ t.toNat < 8 := of_decide_eq_false hc
            exact ⟨by rw [hobs4]; simp [evs, str, hc], by rw [hmem4]⟩
          · intro Γ5 w' hext ⟨hobs', hmem'⟩
            repeat step
            refine wp_bar hA KernelCorpus.barrier rfl (by rw [hobs']; simp; omega) (k := 5)
              (by rw [collCalls_evs w w' _ hobs', List.filterMap_append, evs_colls]; split <;> simp [collOf])
              fun w5 B5 hb5 hobs50 hmem50 => ?_
            have hobs5 : w5.obs = (evs t X W 5).reverse ++ w.obs := by rw [hobs50, hobs']; simp [evs]
            have hmem5 : w5.mem = { w.mem with arena := smemAt ans 5 } := by
              rw [hmem50, hmem']; simp [smemAt, hb5]
            -- round 6
            repeat step
            refine wp_when (R := fun w' => w'.obs = (evs t X W 5 ++ (if t.toNat < str 6 then
                  ([.store (smemAddr + off t) 4 (s32 (add32G (W 5 t.toNat) (W 5 (t.toNat + str 6))))] : List Obs) else [])).reverse
                    ++ w.obs ∧ w'.mem = { w.mem with arena := w'.mem.arena }) (by look) (by look) ?_ ?_ ?_
            · intro hc
              rw [cmp64_ult t 4 4 rfl (by decide)] at hc
              have hc : t.toNat < 4 := of_decide_eq_true hc
              rw [wp_bind]
              refine wp_loadF32 (p := smemAddr + off t) (v := s32 (W 5 t.toNat)) (by look) ?_ ?_
              · rw [load_arena, hmem5]; exact aload_word _ _ (by omega) (by rw [show smemAt ans 5 = B5 by simp [smemAt, hb5], hsh 5 B5 hb5]; omega)
              repeat step
              refine wp_loadF32 (p := smemAddr + off t + c64 (4 * 4)) (v := s32 (W 5 (t.toNat + 4))) (by look) ?_ ?_
              · rw [addr_step t 4 (4 * 4) rfl (by omega), load_arena, toNat_ofNat32 (by omega), hmem5]
                exact aload_word _ _ (by omega) (by rw [show smemAt ans 5 = B5 by simp [smemAt, hb5], hsh 5 B5 hb5]; omega)
              rw [wp_bind]
              refine wp_fadd32w (nan := gpuNaN) (by rw [hmem5]; exact hnan) (by look) (by look) ?_
              refine wp_store_arena (by look) (by look) (by rw [hmem5]; dsimp only; rw [show smemAt ans 5 = B5 by simp [smemAt, hb5], hsh 5 B5 hb5]; omega)
                fun ws hobsS hmemS => ?_
              refine ⟨by extc, by rw [hobsS, hobs5]; simp [evs, str, hc], by rw [hmemS, hmem5]⟩
            · intro hc
              rw [cmp64_ult t 4 4 rfl (by decide)] at hc
              have hc : ¬ t.toNat < 4 := of_decide_eq_false hc
              exact ⟨by rw [hobs5]; simp [evs, str, hc], by rw [hmem5]⟩
            · intro Γ6 w' hext ⟨hobs', hmem'⟩
              repeat step
              refine wp_bar hA KernelCorpus.barrier rfl (by rw [hobs']; simp; omega) (k := 6)
                (by rw [collCalls_evs w w' _ hobs', List.filterMap_append, evs_colls]; split <;> simp [collOf])
                fun w6 B6 hb6 hobs60 hmem60 => ?_
              have hobs6 : w6.obs = (evs t X W 6).reverse ++ w.obs := by rw [hobs60, hobs']; simp [evs]
              have hmem6 : w6.mem = { w.mem with arena := smemAt ans 6 } := by
                rw [hmem60, hmem']; simp [smemAt, hb6]
              -- round 7
              repeat step
              refine wp_when (R := fun w' => w'.obs = (evs t X W 6 ++ (if t.toNat < str 7 then
                    ([.store (smemAddr + off t) 4 (s32 (add32G (W 6 t.toNat) (W 6 (t.toNat + str 7))))] : List Obs) else [])).reverse
                      ++ w.obs ∧ w'.mem = { w.mem with arena := w'.mem.arena }) (by look) (by look) ?_ ?_ ?_
              · intro hc
                rw [cmp64_ult t 2 2 rfl (by decide)] at hc
                have hc : t.toNat < 2 := of_decide_eq_true hc
                rw [wp_bind]
                refine wp_loadF32 (p := smemAddr + off t) (v := s32 (W 6 t.toNat)) (by look) ?_ ?_
                · rw [load_arena, hmem6]; exact aload_word _ _ (by omega) (by rw [show smemAt ans 6 = B6 by simp [smemAt, hb6], hsh 6 B6 hb6]; omega)
                repeat step
                refine wp_loadF32 (p := smemAddr + off t + c64 (4 * 2)) (v := s32 (W 6 (t.toNat + 2))) (by look) ?_ ?_
                · rw [addr_step t 2 (4 * 2) rfl (by omega), load_arena, toNat_ofNat32 (by omega), hmem6]
                  exact aload_word _ _ (by omega) (by rw [show smemAt ans 6 = B6 by simp [smemAt, hb6], hsh 6 B6 hb6]; omega)
                rw [wp_bind]
                refine wp_fadd32w (nan := gpuNaN) (by rw [hmem6]; exact hnan) (by look) (by look) ?_
                refine wp_store_arena (by look) (by look) (by rw [hmem6]; dsimp only; rw [show smemAt ans 6 = B6 by simp [smemAt, hb6], hsh 6 B6 hb6]; omega)
                  fun ws hobsS hmemS => ?_
                refine ⟨by extc, by rw [hobsS, hobs6]; simp [evs, str, hc], by rw [hmemS, hmem6]⟩
              · intro hc
                rw [cmp64_ult t 2 2 rfl (by decide)] at hc
                have hc : ¬ t.toNat < 2 := of_decide_eq_false hc
                exact ⟨by rw [hobs6]; simp [evs, str, hc], by rw [hmem6]⟩
              · intro Γ7 w' hext ⟨hobs', hmem'⟩
                repeat step
                refine wp_bar hA KernelCorpus.barrier rfl (by rw [hobs']; simp; omega) (k := 7)
                  (by rw [collCalls_evs w w' _ hobs', List.filterMap_append, evs_colls]; split <;> simp [collOf])
                  fun w7 B7 hb7 hobs70 hmem70 => ?_
                have hobs7 : w7.obs = (evs t X W 7).reverse ++ w.obs := by rw [hobs70, hobs']; simp [evs]
                have hmem7 : w7.mem = { w.mem with arena := smemAt ans 7 } := by
                  rw [hmem70, hmem']; simp [smemAt, hb7]
                -- round 8
                repeat step
                refine wp_when (R := fun w' => w'.obs = (evs t X W 7 ++ (if t.toNat < str 8 then
                      ([.store (smemAddr + off t) 4 (s32 (add32G (W 7 t.toNat) (W 7 (t.toNat + str 8))))] : List Obs) else [])).reverse
                        ++ w.obs ∧ w'.mem = { w.mem with arena := w'.mem.arena }) (by look) (by look) ?_ ?_ ?_
                · intro hc
                  rw [cmp64_ult t 1 1 rfl (by decide)] at hc
                  have hc : t.toNat < 1 := of_decide_eq_true hc
                  rw [wp_bind]
                  refine wp_loadF32 (p := smemAddr + off t) (v := s32 (W 7 t.toNat)) (by look) ?_ ?_
                  · rw [load_arena, hmem7]; exact aload_word _ _ (by omega) (by rw [show smemAt ans 7 = B7 by simp [smemAt, hb7], hsh 7 B7 hb7]; omega)
                  repeat step
                  refine wp_loadF32 (p := smemAddr + off t + c64 (4 * 1)) (v := s32 (W 7 (t.toNat + 1))) (by look) ?_ ?_
                  · rw [addr_step t 1 (4 * 1) rfl (by omega), load_arena, toNat_ofNat32 (by omega), hmem7]
                    exact aload_word _ _ (by omega) (by rw [show smemAt ans 7 = B7 by simp [smemAt, hb7], hsh 7 B7 hb7]; omega)
                  rw [wp_bind]
                  refine wp_fadd32w (nan := gpuNaN) (by rw [hmem7]; exact hnan) (by look) (by look) ?_
                  refine wp_store_arena (by look) (by look) (by rw [hmem7]; dsimp only; rw [show smemAt ans 7 = B7 by simp [smemAt, hb7], hsh 7 B7 hb7]; omega)
                    fun ws hobsS hmemS => ?_
                  refine ⟨by extc, by rw [hobsS, hobs7]; simp [evs, str, hc], by rw [hmemS, hmem7]⟩
                · intro hc
                  rw [cmp64_ult t 1 1 rfl (by decide)] at hc
                  have hc : ¬ t.toNat < 1 := of_decide_eq_false hc
                  exact ⟨by rw [hobs7]; simp [evs, str, hc], by rw [hmem7]⟩
                · intro Γ8 w' hext ⟨hobs', hmem'⟩
                  repeat step
                  refine wp_bar hA KernelCorpus.barrier rfl (by rw [hobs']; simp; omega) (k := 8)
                    (by rw [collCalls_evs w w' _ hobs', List.filterMap_append, evs_colls]; split <;> simp [collOf])
                    fun w8 B8 hb8 hobs80 hmem80 => ?_
                  have hobs8 : w8.obs = (evs t X W 8).reverse ++ w.obs := by rw [hobs80, hobs']; simp [evs]
                  have hmem8 : w8.mem = { w.mem with arena := smemAt ans 8 } := by
                    rw [hmem80, hmem']; simp [smemAt, hb8]
                  repeat step
                  refine wp_when (R := fun w' => w'.obs = (evs t X W 8 ++ if t = 0 then
                        ([.store (po + off b) 4 (s32 (W 8 0))] : List Obs) else []).reverse ++ w.obs ∧
                      w'.mem = if t = 0 then mo else { w.mem with arena := smemAt ans 8 }) (by look) (by look) ?_ ?_ ?_
                  · intro hc
                    rw [cmp64_eq0] at hc
                    have hc : t = 0 := of_decide_eq_true hc
                    rw [wp_bind]
                    refine wp_loadF32 (p := smemAddr) (v := s32 (W 8 0)) (by look) ?_ ?_
                    · rw [load_arena0, hmem8]
                      exact aload_word _ _ (by decide) (by rw [show smemAt ans 8 = B8 by simp [smemAt, hb8], hsh 8 B8 hb8]; decide)
                    repeat step
                    refine wp_storeF32 (m := mo) (by look) (by look) (by rw [hmem8]; exact hout hc) ?_
                    exact ⟨by extc, by simp [obsStore, hobs8, hc], by simp [hc]⟩
                  · intro hc
                    rw [cmp64_eq0] at hc
                    have hc : ¬ t = 0 := of_decide_eq_false hc
                    exact ⟨by rw [hobs8]; simp [hc], by rw [hmem8]; simp [hc]⟩
                  · intro Γ' w' _ hR
                    exact hR

theorem fine : Fine (emitGo (fnBody (idParams ++ params) KernelCorpus.blockSum) ⟨(idParams ++ params).length, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- **What a blockSum thread leaves**, when it finishes. -/
theorem thread_spec (b t bd : UInt32) (px po : UInt64) (w u : World) (ans : List Answer)
    (hsh : ∀ (k : Nat) (B : ByteArray), ans[k]? = some (.smem B) → B.size = 1024)
    (X : UInt32) (mo : Mem) (ht : t.toNat < 256) (hnan : w.mem.nan = gpuNaN)
    (hx : w.mem.load (px + off (b * bd + t)) 4 = some (s32 X))
    (hs0 : 4 * t.toNat + 4 ≤ w.mem.arena.size)
    (hout : t = 0 → ({ w.mem with arena := smemAt ans 8 } : Mem).store (po + off b) 4
      (s32 (aword (smemAt ans 8) 0)) = some mo)
    {steps : Nat} (hr : thread K.code ans [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 smemAddr,
        .sc .i64 px, .sc .i64 po] w Hw.rounded steps = .ok u) :
    u.obs = (evs t X (fun k i => aword (smemAt ans k) i) 8 ++ if t = 0 then
        ([.store (po + off b) 4 (s32 (aword (smemAt ans 8) 0))] : List Obs) else []).reverse ++ w.obs ∧
      u.mem = if t = 0 then mo else { w.mem with arena := smemAt ans 8 } := by
  refine thread_of_triple_ans (Q := fun u => u.obs = (evs t X (fun k i => aword (smemAt ans k) i) 8 ++
      if t = 0 then ([.store (po + off b) 4 (s32 (aword (smemAt ans 8) 0))] : List Obs) else []).reverse
        ++ w.obs ∧ u.mem = if t = 0 then mo else { w.mem with arena := smemAt ans 8 }) (fun cfg hA _ => ?_) hr
  have h1 := emit_triple (cfg := cfg) (J := { ok := fun _ _ => True, faultOk := false })
    (P0 := At (List.toArray [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 smemAddr,
        .sc .i64 px, .sc .i64 po]) w)
    (PT.conseq (P' := fun Γ' w' => Γ' = (List.toArray [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd),
        .sc .i64 smemAddr, .sc .i64 px, .sc .i64 po]) ∧ w' = w)
      (wp_sound cfg 0 _ (fnBody (idParams ++ params) KernelCorpus.blockSum) _)
      (fun Γ' w' h => by
        rw [h.1, h.2]
        exact thread_wp cfg w b t bd px po ans hA hsh X mo ht hnan hx hs0 hout)
      (fun _ _ _ h => h)) fine (fun Γ' w' h => ⟨by rw [h.1]; rfl, h⟩)
  exact h1

/-- Thread `t`'s shared stores through round `k`, each with the barriers
    before it. -/
def storesUpTo (t X : UInt32) (W : Nat → Nat → UInt32) : Nat → List (Nat × UInt64 × Nat × UInt64)
  | 0 => [(0, smemAddr + off t, 4, s32 X)]
  | k + 1 => storesUpTo t X W k ++
      (if t.toNat < str (k + 1) then
        [(k + 1, smemAddr + off t, 4, s32 (add32G (W k t.toNat) (W k (t.toNat + str (k + 1)))))] else [])

theorem evs_fold (t X : UInt32) (W : Nat → Nat → UInt32) :
    ∀ k, (evs t X W k).foldl shStep (0, []) = (k + 1, (storesUpTo t X W k).reverse) := by
  intro k
  induction k with
  | zero =>
    simp only [evs, storesUpTo, List.foldl_cons, List.foldl_nil, shStep_arena]
    simp [shStep]
  | succ k ih =>
    simp only [evs, storesUpTo, List.foldl_append, ih]
    split
    · simp only [List.foldl_cons, List.foldl_nil, shStep_arena]
      simp [shStep]
    · simp [shStep]

/-- The value thread `t` stores in phase `k`. -/
def pval (X : Nat → UInt32) (W : Nat → Nat → UInt32) (t : Nat) : Nat → UInt32
  | 0 => X t
  | k + 1 => add32G (W k t) (W k (t + str (k + 1)))

/-- Whether thread `t` stores in phase `k`. -/
def pact (k t : Nat) : Bool := k = 0 || decide (t < str k)

theorem storesUpTo_later (t X : UInt32) (W : Nat → Nat → UInt32) (k : Nat) (c : Mem) :
    ∀ n, n < k →
      (storesUpTo t X W n).foldl
        (fun c (x : Nat × UInt64 × Nat × UInt64) => if x.1 = k then (c.store x.2.1 x.2.2.1 x.2.2.2).getD c else c) c = c := by
  intro n
  induction n with
  | zero => intro h; simp [storesUpTo, show (0 : Nat) ≠ k by omega]
  | succ n ih =>
    intro h
    simp only [storesUpTo, List.foldl_append, ih (by omega)]
    split <;> simp [show n + 1 ≠ k by omega]

theorem storesUpTo_phase (t : Nat) (ht : t < 256) (X : Nat → UInt32) (W : Nat → Nat → UInt32) (k : Nat) (c : Mem) :
    ∀ n, k ≤ n →
      (storesUpTo (UInt32.ofNat t) (X t) W n).foldl
        (fun c (x : Nat × UInt64 × Nat × UInt64) => if x.1 = k then (c.store x.2.1 x.2.2.1 x.2.2.2).getD c else c) c =
      if pact k t then (c.store (smemAddr + off (UInt32.ofNat t)) 4 (s32 (pval X W t k))).getD c else c := by
  have htn : (UInt32.ofNat t).toNat = t := toNat_ofNat32 (by omega)
  intro n
  induction n with
  | zero => intro h; obtain rfl : k = 0 := by omega
            simp [storesUpTo, pact, pval]
  | succ n ih =>
    intro h
    rcases Nat.lt_or_ge n k with hk | hk
    · obtain rfl : k = n + 1 := by omega
      simp only [storesUpTo, List.foldl_append, storesUpTo_later _ _ W (n + 1) c n (by omega), htn]
      split
      · rename_i hlt
        simp [pact, pval, hlt]
      · rename_i hlt
        simp [pact, hlt]
    · simp only [storesUpTo, List.foldl_append, ih hk, htn]
      by_cases hlt : t < str (n + 1) <;> simp [hlt, show n + 1 ≠ k by omega]

theorem arena_apart {t u : Nat} (ht : t < 256) (hu : u < 256) (h : t ≠ u) : Apart4 aaddr t u :=
  ⟨_, _, _, _, decode_arena _, decode_arena _,
    Or.inr (by rw [toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)]; omega)⟩

/-- smemAddr block's shared stores, its 256 threads' in order. -/
def allStores (X : Nat → UInt32) (W : Nat → Nat → UInt32) : List (Nat × UInt64 × Nat × UInt64) :=
  (List.range 256).flatMap fun t => storesUpTo (UInt32.ofNat t) (X t) W 8

theorem phase_all (X : Nat → UInt32) (W : Nat → Nat → UInt32) (k : Nat) (hk : k ≤ 8) (cur : Mem) :
    phase (allStores X W) k cur =
      ((List.range 256).filter (pact k)).foldl (fun m t => put aaddr (fun t => s32 (pval X W t k)) t m) cur := by
  unfold phase allStores
  rw [List.foldl_flatMap]
  rw [foldl_congr_mem _ (fun m t => if pact k t then put aaddr (fun t => s32 (pval X W t k)) t m else m) _ _
    (fun t ht m => by rw [storesUpTo_phase t (List.mem_range.mp ht) X W k m 8 hk]; rfl)]
  exact foldl_put_filter _ _ _ _ _

theorem phase_size (X : Nat → UInt32) (W : Nat → Nat → UInt32) (k : Nat) (hk : k ≤ 8) (cur : Mem) :
    (phase (allStores X W) k cur).arena.size = cur.arena.size := by
  rw [phase_all X W k hk]; exact puts_size _ _ _

/-- **One phase of the barrier law**: word `i` of shared memory after phase
    `k` is what thread `i` stores in it, if it stores, and otherwise as before. -/
theorem phase_word (X : Nat → UInt32) (W : Nat → Nat → UInt32) (k : Nat) (hk : k ≤ 8) (cur : Mem)
    (hsz : 1024 ≤ cur.arena.size) (i : Nat) (hi : i < 256) :
    aword (phase (allStores X W) k cur).arena i = if pact k i then pval X W i k else aword cur.arena i := by
  rw [phase_all X W k hk, aword_of _ _ (by omega), aword_of _ _ (by omega)]
  have hap : ∀ t ∈ (List.range 256).filter (pact k), ∀ u ∈ (List.range 256).filter (pact k), t ≠ u →
      Apart4 aaddr t u := fun t ht u hu h => arena_apart
        (List.mem_range.mp (List.mem_filter.mp ht).1) (List.mem_range.mp (List.mem_filter.mp hu).1) h
  by_cases hp : pact k i
  · rw [if_pos hp, load_puts aaddr _ _ _ (List.nodup_range.filter _) hap
      (fun t ht => ⟨store_arena_some cur _ _ (by
          have := List.mem_range.mp (List.mem_filter.mp ht).1
          rw [toNat_ofNat32 (by omega)]; omega), by simp only [s32]; exact UInt32.toNat_lt _⟩)
      i (List.mem_filter.mpr ⟨List.mem_range.mpr hi, hp⟩)]
    simp
  · rw [if_neg hp, load_puts_other aaddr _ _ _ (fun t ht => arena_apart
      (List.mem_range.mp (List.mem_filter.mp ht).1) hi (fun e => hp (e ▸ (List.mem_filter.mp ht).2)))]

/-- The memory the barrier law starts from: shared memory zeroed. -/
def mem0 (smem0 : ByteArray) : Mem := { arena := smem0, data := .empty, out := .empty }

/-- Shared memory after phases 0 to `k`. -/
def cum (X : Nat → UInt32) (W : Nat → Nat → UInt32) (smem0 : ByteArray) : Nat → Mem
  | 0 => phase (allStores X W) 0 (mem0 smem0)
  | k + 1 => phase (allStores X W) (k + 1) (cum X W smem0 k)

theorem foldl_max9 : ∀ (l : List Nat) (m : Nat), l.foldl (fun m _ => max m 9) (max m 9) = max m 9
  | [], _ => rfl
  | _ :: l, m => by rw [List.foldl_cons, show max (max m 9) 9 = max m 9 by omega, foldl_max9 l m]

/-- **The barrier law's snapshots**, for 256 threads each making nine
    barriers and storing `storesUpTo`: the cumulative phases. -/
theorem snapshots_eq {calls : Nat → List (Intr × List V)} {stores : Nat → List (Nat × UInt64 × Nat × UInt64)}
    (X : Nat → UInt32) (W : Nat → Nat → UInt32) (smem0 : ByteArray)
    (hc : ∀ t, t < 256 → calls t = List.replicate 9 (.barrier, []))
    (hst : ∀ t, t < 256 → stores t = storesUpTo (UInt32.ofNat t) (X t) W 8) :
    snapshots 256 smem0 calls stores = ((List.range 9).map fun k => (cum X W smem0 k).arena).toArray := by
  have hnb : (List.range 256).foldl (fun m t => max m ((calls t).filter (·.1 == .barrier)).length) 0 = 9 := by
    rw [foldl_congr_mem _ (fun m _ => max m 9) _ _ (fun t ht m => by
      rw [hc t (List.mem_range.mp ht)]; simp; rfl)]
    rw [List.range_succ_eq_map, List.foldl_cons]
    exact foldl_max9 _ 0
  have hall : (List.range 256).flatMap stores = allStores X W := by
    unfold allStores
    exact flatMap_congr' _ (fun t ht => hst t (List.mem_range.mp ht))
  have key : ∀ n, n ≤ 9 → (List.range n).foldl (fun (acc : Mem × Array ByteArray) b =>
        (phase (allStores X W) b acc.1, acc.2.push (phase (allStores X W) b acc.1).arena)) (mem0 smem0, #[]) =
      ((match n with | 0 => mem0 smem0 | n + 1 => cum X W smem0 n),
        ((List.range n).map fun k => (cum X W smem0 k).arena).toArray) := by
    intro n hn
    induction n with
    | zero => rfl
    | succ n ih =>
      rw [List.range_succ, List.foldl_append, ih (by omega)]
      simp only [List.foldl_cons, List.foldl_nil, List.map_append, List.map_cons, List.map_nil]
      cases n with
      | zero => simp [cum]
      | succ n => simp [cum]
  unfold snapshots
  simp only [hnb, hall]
  have := key 9 (Nat.le_refl 9)
  simp only [mem0] at this
  rw [this]

/-- The tree's partial sums: word `i` after `k` rounds of adding the word
    `str k` above it. -/
def tree (X : Nat → UInt32) : Nat → Nat → UInt32
  | 0, i => X i
  | k + 1, i => if i < str (k + 1) then add32G (tree X k i) (tree X k (i + str (k + 1))) else tree X k i

theorem cum_size (X : Nat → UInt32) (W : Nat → Nat → UInt32) (smem0 : ByteArray) :
    ∀ k, k ≤ 8 → (cum X W smem0 k).arena.size = smem0.size := by
  intro k hk
  induction k with
  | zero => exact phase_size X W 0 (by omega) _
  | succ k ih => rw [cum, phase_size X W (k + 1) hk, ih (by omega)]

/-- **The snapshots hold the tree**: when every thread is answered the
    snapshots, word `i` of the `k`-th is the tree's partial sum. -/
theorem cum_word (X : Nat → UInt32) (W : Nat → Nat → UInt32) (smem0 : ByteArray) (h0 : smem0.size = 1024)
    (hW : ∀ j, j < 8 → ∀ i, W j i = aword (cum X W smem0 j).arena i) :
    ∀ k, k ≤ 8 → ∀ i, i < 256 → aword (cum X W smem0 k).arena i = tree X k i := by
  intro k hk
  induction k with
  | zero =>
    intro i hi
    rw [cum, phase_word X W 0 (by omega) _ (by simp [mem0, h0]) i hi]
    simp [pact, pval, tree]
  | succ k ih =>
    intro i hi
    rw [cum, phase_word X W (k + 1) hk _ (by rw [cum_size X W smem0 k (by omega), h0]; exact Nat.le_refl _) i hi]
    simp only [pact, pval, tree]
    by_cases hlt : i < str (k + 1)
    · have hs : str (k + 1) ≤ 128 := by
        simp only [str]
        rw [Nat.shiftRight_eq_div_pow]
        exact Nat.div_le_of_le_mul (by
          have : 2 ≤ 2 ^ (k + 1) := Nat.le_self_pow (by omega) 2
          omega)
      simp only [hlt, decide_true, Bool.or_true, if_true]
      rw [hW k (by omega), hW k (by omega), ih (by omega) i hi, ih (by omega) _ (by omega)]
    · simp [hlt, ih (by omega) i hi]

/-! ## smemAddr launch -/

/-- `x` and `out`: device buffers 0 and 1. -/
abbrev pxw : UInt64 := addrOf (.buf ⟨0, by decide⟩) 0

abbrev pow : UInt64 := addrOf (.buf ⟨1, by decide⟩) 0

def args : List V := [.sc .i64 pxw, .sc .i64 pow]

/-- Block `b`'s words of `x`. -/
def blockX (X : Nat → UInt32) (b : Nat) (i : Nat) : UInt32 := X (256 * b + i)

/-- **blockSum as an elementwise kernel over blocks**: thread 0 of each block,
    reading its block's 256 words of `x`, stores their tree sum at
    `out[j / 256]`. -/
def elem (X : Nat → UInt32) : Elem where
  out j := pow + off (UInt32.ofNat (j / 256))
  val j := s32 (tree (blockX X (j / 256)) 8 0)
  active j := decide (j % 256 = 0)
  reads j := (List.range 256).map fun l => (pxw + off (UInt32.ofNat (j + l)), s32 (X (j + l)))

theorem elem_apart (X : Nat → UInt32) (N : Nat) (hN : N ≤ 2 ^ 32) : (elem X).Apart N where
  buf j hj _ := ⟨_, _, decode_elem _ _⟩
  outs i j hi hj hne hai haj := by
    simp only [elem, decide_eq_true_eq] at hai haj
    refine ⟨_, _, _, _, decode_elem _ _, decode_elem _ _, Or.inr ?_⟩
    rw [toNat_ofNat32 (by omega), toNat_ofNat32 (by omega)]; omega
  reads i j hi hj hne _ _ p hp := by
    simp only [elem, List.mem_map, List.mem_range] at hp
    obtain ⟨l, -, rfl⟩ := hp
    exact ⟨_, _, _, _, decode_elem _ _, decode_elem _ _, Or.inl (by simp)⟩

theorem decode_out (b : Nat) (hb : b < 2 ^ 32) :
    decodeAddr (pow + off (UInt32.ofNat b)) = some (.buf ⟨1, by decide⟩, 4 * b) := by
  rw [decode_elem, toNat_ofNat32 hb]

/-- **smemAddr blockSum block**: thread 0 leaves at `out[b]` the tree sum of the
    block's 256 words, the barriers answered as the barrier law says. -/
theorem elem_blocks (X : Nat → UInt32) (grid : Nat) (hsz : grid * 256 ≤ 2 ^ 32) :
    Elem.Blocks K args grid 256 (elem X) := by
  intro b hb w w' harena hnan hpre hr
  obtain ⟨steps, ans, w'', calls, stores, ans', hfold, hans, hset, rfl⟩ := hr
  have hs0 : (ByteArray.mk (Array.replicate K.smem 0)).size = 1024 := by simp [K, ByteArray.size]
  have hlt : ∀ t, t < 256 → b * 256 + t < 2 ^ 32 := fun t ht => by
    have := Elem.lt_grid (grid := grid) hb ht; omega
  have hb32 : b < 2 ^ 32 := by have := hlt 0 (by decide); omega
  have hsh : ∀ t, t < 256 → ∀ (k : Nat) (B : ByteArray), (ans t)[k]? = some (.smem B) → B.size = 1024 := by
    intro t ht k B h
    rw [← hset t ht] at h
    rw [answersOf_smem_size (snapshots_size _ _ _ _) (answers_at hans ht) k B h, hs0]
  let Xl := blockX X b
  let Q (m : Mem) : Prop :=
    (∀ t, t < 256 → m.load (pxw + off (UInt32.ofNat (b * 256 + t))) 4 = some (s32 (Xl t))) ∧
    (∀ v, (m.store (pow + off (UInt32.ofNat b)) 4 v).isSome) ∧ m.nan = gpuNaN
  let A8 (t : Nat) := smemAt (ans t) 8
  let g (t : Nat) (m : Mem) : Mem :=
    if t = 0 then (m.store (pow + off (UInt32.ofNat b)) 4 (s32 (aword (A8 0) 0))).getD m else m
  let f (t : Nat) (m : Mem) : Mem := { g t m with arena := A8 t }
  have hdo := decode_out b hb32
  have hQ0 : Q { w.mem with arena := ByteArray.mk (Array.replicate K.smem 0) } := by
    refine ⟨fun t ht => ?_, fun v => ?_, hnan⟩
    · have hp := (hpre 0 (by decide) (by simp [elem])).1 _
        (List.mem_map.mpr ⟨t, List.mem_range.mpr ht, rfl⟩)
      simp only [Nat.add_zero] at hp
      rw [load_buf_arena _ _ (decode_elem _ _)]
      simpa [Xl, blockX, Nat.mul_comm] using hp
    · have hp := (hpre 0 (by decide) (by simp [elem])).2 v
      rw [store_buf_arena _ _ hdo, Option.isSome_map]
      simpa [elem, Nat.add_zero, Nat.mul_div_cancel] using hp
  have hkeep : ∀ t u m, t ≠ u → (u < 256 ∧ Q m) → (u < 256 ∧ Q (f t m)) := by
    intro t u m _ ⟨hu, hl, hs, hn⟩
    refine ⟨hu, fun t' ht' => ?_, fun v => ?_, ?_⟩
    · simp only [f, g]
      rw [load_buf_arena _ _ (decode_elem _ _)]
      split
      · cases hst : m.store (pow + off (UInt32.ofNat b)) 4 (s32 (aword (A8 0) 0)) with
        | none => exact hl t' ht'
        | some m' =>
          simp only [Option.getD_some]
          rw [load_store_clear hst hdo (decode_elem _ _) (Or.inl (by simp)), hl t' ht']
      · exact hl t' ht'
    · simp only [f, g]
      rw [store_buf_arena _ _ hdo, Option.isSome_map]
      split
      · cases hst : m.store (pow + off (UInt32.ofNat b)) 4 (s32 (aword (A8 0) 0)) with
        | none => exact hs v
        | some m' =>
          obtain ⟨_, _, _, hF⟩ := store_frame hst
          simp only [Option.getD_some]
          rw [← MemSame.store_isSome hF.same _ 4 v v]; exact hs v
      · exact hs v
    · show (g t m).nan = _
      simp only [g]
      split
      · cases hst : m.store (pow + off (UInt32.ofNat b)) 4 (s32 (aword (A8 0) 0)) with
        | none => exact hn
        | some m' => simp only [Option.getD_some]; rw [Mem.store_nan hst]; exact hn
      · exact hn
  obtain ⟨hmem, hC⟩ := pass_spec (fun t m => t < 256 ∧ Q m) f
    (fun t _ cs st => cs = List.replicate 9 (.barrier, []) ∧
      st = storesUpTo (UInt32.ofNat t) (Xl t) (fun k i => aword (smemAt (ans t) k) i) 8)
    (fun t s u ⟨ht, hQ⟩ hu => by
      have htn : (UInt32.ofNat t).toNat = t := toNat_ofNat32 (by omega)
      obtain ⟨m1, hm1⟩ := Option.isSome_iff_exists.mp (hQ.2.1 (s32 (aword (smemAt (ans t) 8) 0)))
      obtain ⟨hobs, hmemu⟩ := thread_spec (UInt32.ofNat b) (UInt32.ofNat t) (UInt32.ofNat 256) pxw pow _ u (ans t)
        (hsh t ht) (Xl t) { m1 with arena := smemAt (ans t) 8 } (by rw [htn]; omega) (by exact hQ.2.2)
        (by rw [gi_eq]; simp only; rw [load_buf_arena _ _ (decode_elem _ _)]; exact hQ.1 t ht)
        (by simp only; rw [hs0, htn]; omega)
        (fun _ => by simp only; rw [store_buf_arena _ _ hdo, store_buf_arena _ _ hdo, hm1]; rfl)
        hu
      refine ⟨?_, ?_, ?_⟩
      · rw [hmemu]
        by_cases h0 : t = 0
        · subst h0
          simp only [f, g, A8, if_true, show UInt32.ofNat 0 = 0 from rfl]
          rw [hm1]; rfl
        · have h0' : UInt32.ofNat t ≠ 0 := fun e => h0 (by rw [← htn, e]; rfl)
          simp only [f, g, h0, h0', if_false]
          rfl
      · rw [collCalls_evs _ u _ hobs, List.filterMap_append, evs_colls]
        split <;> simp [collOf]
      · rw [sharedStores_evs _ u _ hobs, List.foldl_append, evs_fold]
        split
        · rw [List.foldl_cons, List.foldl_nil, shStep_buf _ _ hdo]; simp
        · simp)
    hkeep (List.range 256) _ _ List.nodup_range (fun t ht => ⟨List.mem_range.mp ht, hQ0⟩) hfold
  have hcalls : ∀ t, t < 256 → calls t = List.replicate 9 (.barrier, []) :=
    fun t ht => (hC t (List.mem_range.mpr ht)).1
  -- every thread is answered the snapshots
  have hansT : ∀ t, t < 256 → ans t = (List.range 9).map fun k =>
      Answer.smem ((snapshots 256 (ByteArray.mk (Array.replicate K.smem 0)) calls stores)[k]?.getD
        (ByteArray.mk (Array.replicate K.smem 0))) := fun t ht => by
    rw [← hset t ht]
    have := answers_at hans ht
    rw [answersOf_bars (hcalls t ht)] at this
    exact (Except.ok.inj this).symm
  have hsame : ∀ t, t < 256 → ans t = ans 0 := fun t ht => by rw [hansT t ht, hansT 0 (by decide)]
  let W : Nat → Nat → UInt32 := fun k i => aword (smemAt (ans 0) k) i
  have hst : ∀ t, t < 256 → stores t = storesUpTo (UInt32.ofNat t) (Xl t) W 8 := fun t ht => by
    have h := (hC t (List.mem_range.mpr ht)).2
    simp only at h
    rw [h, hsame t ht]
  have hsnap := snapshots_eq Xl W (ByteArray.mk (Array.replicate K.smem 0)) hcalls hst
  have hS : ∀ k, k < 9 → smemAt (ans 0) k = (cum Xl W (ByteArray.mk (Array.replicate K.smem 0)) k).arena := by
    intro k hk
    rw [hansT 0 (by decide), hsnap]
    simp [smemAt, List.getElem?_range hk]
  have hW : ∀ j, j < 8 → ∀ i, W j i = aword (cum Xl W (ByteArray.mk (Array.replicate K.smem 0)) j).arena i :=
    fun j hj i => by simp only [W]; rw [hS j (by omega)]
  have h8 : aword (smemAt (ans 0) 8) 0 = tree Xl 8 0 := by
    rw [hS 8 (by decide)]
    exact cum_word Xl W _ hs0 hW 8 (Nat.le_refl 8) 0 (by decide)
  have hg : ∀ t m x, g t { m with arena := x } = { g t m with arena := x } := by
    intro t m x
    simp only [g]
    split
    · rw [store_buf_arena _ _ hdo]
      cases m.store (pow + off (UInt32.ofNat b)) 4 (s32 (aword (A8 0) 0)) <;> rfl
    · rfl
  have hge : ∀ t ∈ List.range 256, ∀ m, g t m = (elem X).eff (b * 256 + t) m := by
    intro t ht m
    have ht := List.mem_range.mp ht
    simp only [g, Elem.eff, elem, put, decide_eq_true_eq]
    by_cases h0 : t = 0
    · subst h0
      rw [if_pos rfl, if_pos (by omega), Nat.add_zero, Nat.mul_div_cancel _ (by decide)]
      simp only [A8, h8]
      rfl
    · rw [if_neg h0, if_neg (by omega)]
  show { w''.mem with arena := w.mem.arena } = _
  simp only at hmem
  rw [hmem, foldl_arena_eq A8 (fun t m => rfl) hg, foldl_arena_init hg, foldl_congr_mem _ _ _ _ hge]
  apply mem_arena_self
  rw [← List.foldl_map (f := (b * 256 + ·)) (g := fun m j => (elem X).eff j m),
    Elem.foldl_eff_arena (elem_apart X (grid * 256) hsz) _ _ (fun j hj => by
      obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hj; exact Elem.lt_grid hb (List.mem_range.mp ht))]

/-- **What a launch of blockSum leaves**: after `grid` blocks of 256 threads,
    `out[b]` holds the tree sum of block `b`'s 256 words of `x`, each barrier
    answered as the barrier law says. -/
theorem launch_spec (X : Nat → UInt32) (grid : Nat) (hsz : grid * 256 ≤ 2 ^ 32)
    (bufs bufs' : Nat → ByteArray) (hpre : ∀ j, j < grid * 256 → (elem X).Pre (kmem bufs) j)
    (hr : LaunchRuns K grid 256 args bufs bufs') :
    ∀ b, b < grid →
      ({ kmem bufs with bufs := bufs' } : Mem).load (pow + off (UInt32.ofNat b)) 4 =
        some (s32 (tree (blockX X b) 8 0)) := by
  intro b hb
  have h := Elem.launch_blocks (elem_apart X _ hsz) (elem_blocks X grid hsz) bufs bufs' hpre
    (fun j _ _ => by simp only [elem, s32, UInt32.toNat_toUInt64]; exact UInt32.toNat_lt _) hr (256 * b)
    (by have : (b + 1) * 256 ≤ grid * 256 := Nat.mul_le_mul_right _ hb; rw [Nat.succ_mul] at this; omega)
    (by simp [elem])
  simpa [elem, Nat.mul_div_cancel_left _ (show 0 < 256 by decide)] using h

end KernelBlockSum
