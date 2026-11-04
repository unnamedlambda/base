module
public import GptOss.Attention
meta import GptOss.Attention
public import AlgorithmLib.ML.Kernel.BodyMap
meta import AlgorithmLib.ML.Kernel.BodyMap
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# gpt-oss's decode kernels, proven as the bodies it launches

What the decode's kernels compute, stated of the bodies it ships
(`ML.Body.kernel`) launched as it launches them. Apart from the generators,
which build the artifacts and so may not wait on proofs.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof
open AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyPhase GptOssAttention

namespace GptOssProofs

/-- **The add's launch adds**: `g` one-warp blocks leave each word `j < 32·g`
    of buffer 0 as the binary32 sum of words `j` of buffers 0 and 1 as the
    launch found them. -/
theorem add_body (hw : Kernel.Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (g : Nat) (hg : g * 32 ≤ 2 ^ 32)
    (bufs bufs' : Nat → ByteArray) (hsz : ∀ b, b < 2 → 4 * (g * 32) ≤ (bufs b).size)
    (hr : LaunchRuns addKernel g 32 (BodyWarp.bufArgs 2) bufs bufs' hw) :
    ∀ j, j < g * 32 → ({ Kernel.kmem bufs with bufs := bufs' } : HProg.Sem.Mem).load (BodyProof.bufW 2 0 j) 4 =
      some (HProg.Sem.nanFix hw.nan .f32 (HProg.Sem.ofF32 (HProg.Sem.f32 (((Kernel.kmem bufs).load (BodyProof.bufW 2 0 j) 4).getD 0) +
        HProg.Sem.f32 (((Kernel.kmem bufs).load (BodyProof.bufW 2 1 j) 4).getD 0)))) := by
  have hin : ∀ i, twoIn i < 2 := fun i => by unfold twoIn; split <;> decide
  have e : g * 1 * 32 = g * 32 := by rw [Nat.mul_one]
  intro j hj
  exact BodyMap.mapKernel_body addSpec twoIn 0 "addEW" 2 1 g (by decide) (by decide) hw hw32 hin (by decide)
    (by decide) (by rw [e]; exact hg) ⟨by decide +kernel, by decide +kernel⟩ bufs bufs'
    (fun i => by rw [e]; exact hsz _ (hin i)) (by rw [e]; exact hsz 0 (by decide)) hr j (by rw [e]; exact hj)

/-- RMSNorm's lane `l` touches the elements `≡ l (mod 32)` of its three
    buffers: the input, the weights and the output. -/
def rmsAcc (cta : Nat) : BodyWarp.Acc := fun l b j => cta = 0 ∧ b < 3 ∧ j < H ∧ j % 32 = l.val

theorem rms_acc : BodyWarp.AccOk 0 (rmsAcc 0) (fun _ _ => none) 0 rmsKernelEW := by
  simp only [rmsKernelEW, rmsReduce, storeBody, rmsStoreCompute, warpRoundE, BodyWarp.AccOk, rmsAcc, strideIx32,
    IdxE.eval]
  have hH : H = 2880 := rfl
  have key : ∀ j, j < H / 32 → ∀ l : Lane, j * 32 + l.val < H ∧ (j * 32 + l.val) % 32 = l.val := by
    intro j hj l; have : l.val < 32 := l.isLt; rw [hH] at *; omega
  simp only [true_and, and_true, show (0:Nat) < 3 from by decide, show (1:Nat) < 3 from by decide,
    show (2:Nat) < 3 from by decide]
  refine ⟨fun j hj l => key j hj l, fun j hj => ⟨⟨fun l => key j hj l, fun l => key j hj l⟩, fun l => ⟨key j hj l, ?_⟩⟩⟩
  intro l' h; exact Fin.ext (h.2.symm.trans (key j hj l).2)

/-- **The RMSNorm's launch runs its statement**: one block of one warp leaves
    every word `j < H` of its three buffers as the warp machine's run of the
    statement at `F32` does from the launch's memory, with `hw`'s
    approximations. -/
theorem rms_body (hw : Kernel.Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < 3 → 4 * H ≤ (bufs b).size)
    (hr : LaunchRuns rmsKernel 1 32 (BodyWarp.bufArgs 3) bufs bufs' hw) :
    ∀ b j, b < 3 → j < H → ({ Kernel.kmem bufs with bufs := bufs' } : HProg.Sem.Mem).load (BodyProof.bufW 3 b j) 4 =
      some ((@EWStmt.run F32 (F32.ops hw) _ rmsKernelEW (BodyWarp.ws0 3 (Kernel.kmem bufs)) 0 0).mem b j).bits := by
  intro b j hb hj
  have hH : H = 2880 := rfl
  have h := BodyPhase.launch_lockstep_one hw hw32 "rmsNorm" 3 0 rmsKernelEW 1 (by decide) rmsAcc (fun _ _ => none)
    (fun c hc => by obtain rfl : c = 0 := by omega
                    exact rms_acc)
    (fun c l b j h => by
      obtain ⟨_, h2, h3, _⟩ := h
      exact ⟨h2, Nat.lt_of_lt_of_le h2 (by decide), by rw [hH] at h3; omega⟩)
    (fun c c' l l' b j hc hc' hne _ _ => by omega)
    ⟨by decide +kernel, by decide +kernel⟩ bufs bufs' (fun _ _ _ h => by cases h) (fun _ _ _ _ _ _ => rfl)
    (fun c hc => by obtain rfl : c = 0 := by omega
                    decide +kernel)
    (fun c l b j hc h => by
      obtain ⟨_, h2, h3, _⟩ := h
      have := hsz b h2; omega)
    hr 0 ⟨j % 32, Nat.mod_lt _ (by decide)⟩ b j (by decide) ⟨rfl, hb, hj, rfl⟩
  rw [BodyWarp.imOf_none] at h
  exact h

/-- The meta words the cache store reads: the slot and the head stride. -/
def kvRd (m : Mem) : Buf → Nat → Option Nat :=
  fun b j => if b = 2 ∧ (j = M_SLOT ∨ j = M_KVSTRIDE) then rdM 3 m b j else none

/-- Block `cta`'s lane `l` touches, at each trip `i`, its element of the head
    in the packed row and its element of the head's slot in the cache. -/
def kvAcc (base hd g slot stride : Nat) (cta : Nat) : Acc := fun l b j =>
  cta < g ∧ ∃ i, i < hd / 32 ∧
    ((b = 0 ∧ j = base + (cta * hd + (i * 32 + l.val))) ∨ (b = 1 ∧ j = cta * stride + slot * hd + (i * 32 + l.val)))

theorem kv_rdOk (m : Mem) : RdOk 3 m (kvRd m) := by
  intro b j n h
  simp only [kvRd] at h
  split at h
  · rename_i hb
    obtain ⟨rfl, hj⟩ := hb
    refine ⟨⟨by decide, by decide, ?_⟩, h⟩
    rcases hj with rfl | rfl <;> decide
  · cases h

theorem kv_acc (base hd g slot stride cta : Nat) (m : Mem) (hc : cta < g)
    (hs : imOf (kvRd m) 2 M_SLOT = slot) (ht : imOf (kvRd m) 2 M_KVSTRIDE = stride) :
    AccOk cta (kvAcc base hd g slot stride cta) (kvRd m) 0 (kvStoreEW base hd) := by
  intro i hi
  simp only [AccOk, kvAcc, kvSrcIx, kvDstIx, kvElem, IdxE.eval, hs, ht]
  refine ⟨fun l => ⟨hc, i, hi, Or.inl (by simp)⟩, fun l => ⟨⟨hc, i, hi, Or.inr (by simp)⟩, ?_⟩⟩
  intro l' ⟨_, i', hi', h⟩
  rcases h with ⟨h1, _⟩ | ⟨_, h2⟩
  · cases h1
  · have : l'.val < 32 := l'.isLt
    have : l.val < 32 := l.isLt
    exact Fin.ext (by omega)

theorem kv_fits (base hd g slot stride cta : Nat) (m : Mem) (hc : cta < g)
    (hs : kvRd m 2 M_SLOT = some slot) (ht : kvRd m 2 M_KVSTRIDE = some stride)
    (hbase : base < 2 ^ 31) (hhd : hd < 2 ^ 31) (hsrc : base + g * hd ≤ 2 ^ 32)
    (hdst : slot * hd + hd ≤ stride) (hg : g * stride ≤ 2 ^ 32) (hgrid : g ≤ 2 ^ 32) :
    fits (kvRd m) cta 0 (kvStoreEW base hd) = true := by
  simp only [kvStoreEW, fits, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range]
  refine ⟨by omega, fun i hi => ⟨fun l hl => ?_, fun l hl => ?_⟩⟩
  · have h1 : cta < 2 ^ 32 := by omega
    have h2 : cta * hd < 2 ^ 32 := by
      have : cta * hd + hd ≤ g * hd := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hc
      omega
    have h3 : i * 32 < 2 ^ 32 := by have : i * 32 < hd := by omega
                                    omega
    have h4 : i * 32 + l < 2 ^ 32 := by omega
    have h5 : cta * hd + (i * 32 + l) < 2 ^ 32 := by
      have : cta * hd + hd ≤ g * hd := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hc
      omega
    have h6 : base + (cta * hd + (i * 32 + l)) < 2 ^ 32 := by
      have : cta * hd + hd ≤ g * hd := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hc
      omega
    have hl32 : l < 2 ^ 32 := by omega
    have hi32 : i < 2 ^ 32 := by omega
    simp [kvSrcIx, kvElem, idxL, h1, h2, h3, h4, h5, h6, hbase, hhd, hl32, hi32]
  · have h1 : cta < 2 ^ 32 := by omega
    have hct : cta * stride + stride ≤ g * stride := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hc
    have h2 : cta * stride < 2 ^ 32 := by omega
    have h3 : i * 32 < 2 ^ 32 := by have : i * 32 < hd := by omega
                                    omega
    have h4 : i * 32 + l < 2 ^ 32 := by omega
    have hih : i * 32 + l < hd := by omega
    have h5 : slot * hd < 2 ^ 32 := by omega
    have h6 : cta * stride + slot * hd < 2 ^ 32 := by omega
    have h7 : cta * stride + slot * hd + (i * 32 + l) < 2 ^ 32 := by omega
    have hl32 : l < 2 ^ 32 := by omega
    have hi32 : i < 2 ^ 32 := by omega
    have hs' : kvRd m 2 6 = some slot := hs
    have ht' : kvRd m 2 7 = some stride := ht
    simp [kvDstIx, kvElem, idxL, M_SLOT, M_KVSTRIDE, hs', ht', h1, h2, h3, h4, h5, h6, h7, hhd, hl32, hi32]

/-- Two numbers below `s` past multiples of `s` that agree are past the same one. -/
theorem mul_add_inj {c c' s x y : Nat} (h : c * s + x = c' * s + y) (hx : x < s) (hy : y < s) : c = c' := by
  rcases Nat.lt_trichotomy c c' with h1 | h1 | h1
  · have : (c + 1) * s ≤ c' * s := Nat.mul_le_mul_right _ h1
    rw [Nat.succ_mul] at this; omega
  · exact h1
  · have : (c' + 1) * s ≤ c * s := Nat.mul_le_mul_right _ h1
    rw [Nat.succ_mul] at this; omega

/-- The cache store's launch, for any base and head width it is emitted whole at. -/
theorem kv_launch (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (base hd g : Nat)
    (bufs bufs' : Nat → ByteArray) (slot stride : Nat)
    (hs : rdM 3 (kmem bufs) 2 M_SLOT = some slot) (ht : rdM 3 (kmem bufs) 2 M_KVSTRIDE = some stride)
    (hbase : base < 2 ^ 31) (hhd : hd < 2 ^ 31) (hsrc : base + g * hd ≤ 2 ^ 32)
    (hdst : slot * hd + hd ≤ stride) (hg : g * stride ≤ 2 ^ 32) (hgrid : g ≤ 2 ^ 32)
    (hsz0 : 4 * (base + g * hd) ≤ (bufs 0).size) (hsz1 : 4 * (g * stride) ≤ (bufs 1).size)
    (hfine : Fine (emitGo (fnBody (params 3) (Body.body 3 (kvStoreEW base hd))) ⟨(params 3).length, 0, [], [], [], none⟩).2)
    (hr : LaunchRuns (kvStoreKernel base hd) g 32 (bufArgs 3) bufs bufs' hw) :
    ∀ cta i (l : Lane), cta < g → i < hd / 32 →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW 3 1 (cta * stride + slot * hd + (i * 32 + l.val))) 4 =
        (kmem bufs).load (bufW 3 0 (base + (cta * hd + (i * 32 + l.val)))) 4 := by
  intro cta i l hc hi
  have hs' : kvRd (kmem bufs) 2 M_SLOT = some slot := by simp [kvRd, hs]
  have ht' : kvRd (kmem bufs) 2 M_KVSTRIDE = some stride := by simp [kvRd, ht]
  have hims : imOf (kvRd (kmem bufs)) 2 M_SLOT = slot := by simp [imOf, hs']
  have himt : imOf (kvRd (kmem bufs)) 2 M_KVSTRIDE = stride := by simp [imOf, ht']
  have hl32 : ∀ l : Lane, l.val < 32 := fun l => l.isLt
  have hgh : ∀ c, c < g → c * hd + hd ≤ g * hd := fun c hc => by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hc
  have hgs : ∀ c, c < g → c * stride + stride ≤ g * stride := fun c hc => by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hc
  have hrange : ∀ c (l : Lane) b j, kvAcc base hd g slot stride c l b j →
      (b = 0 ∧ j < base + g * hd) ∨ (b = 1 ∧ j < g * stride) := by
    intro c l b j ⟨hc, i, hi, h⟩
    have := hl32 l
    have : i * 32 + l.val < hd := by omega
    rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · have := hgh c hc; left; exact ⟨rfl, by omega⟩
    · have := hgs c hc; right; exact ⟨rfl, by omega⟩
  have h := launch_lockstep_one hw hw32 "kvStore" 3 0 (kvStoreEW base hd) g hgrid (kvAcc base hd g slot stride)
    (kvRd (kmem bufs)) (fun c hc => kv_acc base hd g slot stride c _ hc hims himt)
    (fun c l b j h => by
      rcases hrange c l b j h with ⟨rfl, hj⟩ | ⟨rfl, hj⟩
      · exact ⟨by decide, by decide, by omega⟩
      · exact ⟨by decide, by decide, by omega⟩)
    (fun c c' l l' b j hc hc' hne h hst => hne (by
      have h' := stores_acc c' _ _ l' _ 0 b j (kv_acc base hd g slot stride c' _ hc' hims himt) hst
      obtain ⟨_, i, hi, h⟩ := h
      obtain ⟨_, i', hi', h'⟩ := h'
      have := hl32 l; have := hl32 l'
      have e1 : i * 32 + l.val < hd := by omega
      have e2 : i' * 32 + l'.val < hd := by omega
      rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> rcases h' with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · exact mul_add_inj (s := hd) (by omega) e1 e2
      · cases h1
      · cases h1
      · exact mul_add_inj (s := stride) (x := slot * hd + (i * 32 + l.val)) (y := slot * hd + (i' * 32 + l'.val))
          (by omega) (by omega) (by omega)))
    hfine bufs bufs' (kv_rdOk _)
    (fun c l b j hc h => by
      rcases hrange c l b j h with ⟨rfl, _⟩ | ⟨rfl, _⟩ <;> simp [kvRd])
    (fun c hc => kv_fits base hd g slot stride c _ hc hs' ht' hbase hhd hsrc hdst hg hgrid)
    (fun c l b j hc h => by
      rcases hrange c l b j h with ⟨rfl, hj⟩ | ⟨rfl, hj⟩ <;> omega)
    hr cta l 1 _ hc ⟨hc, i, hi, Or.inr ⟨rfl, rfl⟩⟩
  have hwr := @kvStore_writes F32 (F32.ops hw) _ base hd cta (fun _ _ => 0) (imOf (kvRd (kmem bufs)))
    (ws0 3 (kmem bufs)) i l hi
  rw [kvDst_eval, hims, himt] at hwr
  rw [h, hwr]
  have hj : base + (cta * hd + (i * 32 + l.val)) < base + g * hd := by have := hgh cta hc; have := hl32 l; omega
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp
    (bufW_ok (nbuf := 3) (b := 0) (j := base + (cta * hd + (i * 32 + l.val))) (m := kmem bufs)
      ⟨by decide, by decide, by omega⟩ (by simp [kmem]; omega) 0).1
  show some (((kmem bufs).load (bufW 3 0 (base + (cta * hd + (i * 32 + l.val)))) 4).getD 0) = _
  rw [hv]; rfl

/-- Every cache store that ships is emitted whole. -/
theorem kv_fine : ∀ p ∈ kvShipped,
    Fine (emitGo (fnBody (params 3) (Body.body 3 (kvStoreEW p.1 p.2))) ⟨(params 3).length, 0, [], [], [], none⟩).2 := by
  intro p hp
  simp only [kvShipped, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl <;> exact ⟨by decide +kernel, by decide +kernel⟩

/-- **Every shipped cache store's launch lands each element of the heads in
    the slot**: `g` one-warp blocks leave the cache's word
    `cta·stride + slot·hd + e`, for every head `cta < g` and element
    `e = 32·i + l` the trips cover, as the packed row's word `base + cta·hd + e`
    was, where `slot` and `stride` are the meta's words `M_SLOT` and
    `M_KVSTRIDE`, a slot's row fits within a head's stride and the heads within
    the cache. -/
theorem kv_body (p : Nat × Nat) (hp : p ∈ kvShipped) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (g : Nat) (bufs bufs' : Nat → ByteArray) (slot stride : Nat)
    (hs : rdM 3 (kmem bufs) 2 M_SLOT = some slot) (ht : rdM 3 (kmem bufs) 2 M_KVSTRIDE = some stride)
    (hsrc : p.1 + g * p.2 ≤ 2 ^ 32) (hdst : slot * p.2 + p.2 ≤ stride) (hg : g * stride ≤ 2 ^ 32)
    (hgrid : g ≤ 2 ^ 32) (hsz0 : 4 * (p.1 + g * p.2) ≤ (bufs 0).size) (hsz1 : 4 * (g * stride) ≤ (bufs 1).size)
    (hr : LaunchRuns (kvStoreKernel p.1 p.2) g 32 (bufArgs 3) bufs bufs' hw) :
    ∀ cta i (l : Lane), cta < g → i < p.2 / 32 →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW 3 1 (cta * stride + slot * p.2 + (i * 32 + l.val))) 4 =
        (kmem bufs).load (bufW 3 0 (p.1 + (cta * p.2 + (i * 32 + l.val)))) 4 := by
  have hb : p.1 < 2 ^ 31 ∧ p.2 < 2 ^ 31 := by
    simp only [kvShipped, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl <;> decide
  exact kv_launch hw hw32 p.1 p.2 g bufs bufs' slot stride hs ht hb.1 hb.2 hsrc hdst hg hgrid hsz0 hsz1
    (kv_fine p hp) hr

/-- The meta word the rotation reads: the position. -/
def ropeRd (m : Mem) : Buf → Nat → Option Nat :=
  fun b j => if b = 1 ∧ j = M_POS then rdM 3 m b j else none

/-- Head `cta`'s lane `l` touches its pair in the packed row and the pair's
    sine and cosine at the position. -/
def ropeAcc (base g pos : Nat) (cta : Nat) : Acc := fun l b j =>
  cta < g ∧ ((b = 0 ∧ (j = base + (cta * HD + l.val) ∨ j = base + (cta * HD + l.val) + HALF)) ∨
    (b = 2 ∧ (j = pos * HALF + l.val ∨ j = ROPE_N * HALF + (pos * HALF + l.val))))

theorem rope_rdOk (m : Mem) : RdOk 3 m (ropeRd m) := by
  intro b j n h
  simp only [ropeRd] at h
  split at h
  · rename_i hb
    obtain ⟨rfl, rfl⟩ := hb
    exact ⟨⟨by decide, by decide, by decide⟩, h⟩
  · cases h

theorem rope_acc (base g pos cta : Nat) (m : Mem) (hc : cta < g) (hp : imOf (ropeRd m) 1 M_POS = pos) :
    AccOk cta (ropeAcc base g pos cta) (ropeRd m) 0 (ropeEW base) := by
  simp only [ropeEW, ropeBodyEW, AccOk, ropeAcc, ropeTblIx, ropeCosIx, ropeLoIx, ropeHiIx, IdxE.eval, hp]
  have hH : HALF = 32 := rfl
  refine ⟨⟨⟨⟨?_, ?_⟩, ?_, ?_⟩, trivial, trivial⟩, fun l => ⟨⟨hc, by simp⟩, ?_⟩, fun l => ⟨⟨hc, by simp⟩, ?_⟩⟩
  iterate 4 (intro l; exact ⟨hc, by simp⟩)
  all_goals
    intro l' ⟨_, h⟩
    have : l.val < 32 := l.isLt; have : l'.val < 32 := l'.isLt
    rcases h with ⟨_, h | h⟩ | ⟨h, _⟩
    · exact Fin.ext (by rw [hH] at *; omega)
    · exact Fin.ext (by rw [hH] at *; omega)
    · cases h

theorem rope_fits (base g pos cta : Nat) (m : Mem) (hc : cta < g) (hp : ropeRd m 1 M_POS = some pos)
    (hpos : pos < ROPE_N) (hbase : base < 2 ^ 31) (hg : base + g * HD ≤ 2 ^ 32) :
    fits (ropeRd m) cta 0 (ropeEW base) = true := by
  have hH : HALF = 32 := rfl
  have hD : HD = 64 := rfl
  have hR : ROPE_N = 131072 := rfl
  have hp' : ropeRd m 1 1 = some pos := hp
  have hgh : cta * HD + HD ≤ g * HD := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hc
  simp only [ropeEW, ropeBodyEW, fits, Bool.and_eq_true, List.all_eq_true, List.mem_range]
  refine ⟨⟨⟨⟨fun l hl => ?_, fun l hl => ?_⟩, fun l hl => ?_, fun l hl => ?_⟩, trivial, trivial⟩,
    fun l hl => ?_, fun l hl => ?_⟩
  all_goals
    have h1 : cta < 2 ^ 32 := by rw [hD] at hg; omega
    have h2 : cta * HD < 2 ^ 32 := by omega
    have h3 : cta * HD + l < 2 ^ 32 := by omega
    have h4 : base + (cta * HD + l) < 2 ^ 32 := by omega
    have h5 : base + (cta * HD + l) + HALF < 2 ^ 32 := by rw [hD] at hgh hg ⊢; rw [hH]; omega
    have h6 : pos * HALF < 2 ^ 32 := by rw [hH]; rw [hR] at hpos; omega
    have h7 : pos * HALF + l < 2 ^ 32 := by rw [hH]; rw [hR] at hpos; omega
    have h8 : ROPE_N * HALF + (pos * HALF + l) < 2 ^ 32 := by rw [hH, hR]; rw [hR] at hpos; omega
    have hl32 : l < 2 ^ 32 := by omega
    have hHl : HALF < 2147483648 := by rw [hH]; decide
    have hDl : HD < 2147483648 := by rw [hD]; decide
    have hRH : ROPE_N * HALF < 2147483648 := by rw [hH, hR]; decide
    simp [ropeTblIx, ropeCosIx, ropeLoIx, ropeHiIx, idxL, M_POS, hp', h1, h2, h3, h4, h5, h6, h7, h8, hl32, hbase,
      hHl, hDl, hRH]

/-- The rotation's launch, for any base it is emitted whole at. -/
theorem rope_launch (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (base g : Nat)
    (bufs bufs' : Nat → ByteArray) (pos : Nat) (hp : rdM 3 (kmem bufs) 1 M_POS = some pos) (hpos : pos < ROPE_N)
    (hbase : base < 2 ^ 31) (hg : base + g * HD ≤ 2 ^ 32)
    (hsz0 : 4 * (base + g * HD) ≤ (bufs 0).size) (hsz2 : 4 * (2 * ROPE_N * HALF) ≤ (bufs 2).size)
    (hfine : Fine (emitGo (fnBody (params 3) (Body.body 3 (ropeEW base))) ⟨(params 3).length, 0, [], [], [], none⟩).2)
    (hr : LaunchRuns (ropeKernel base) g 32 (bufArgs 3) bufs bufs' hw) :
    ∀ cta l b j, cta < g → ropeAcc base g pos cta l b j →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW 3 b j) 4 =
        some ((@EWStmt.run F32 (F32.ops hw) _ (ropeEW base) (ws0 3 (kmem bufs)) cta 0 (fun _ _ => 0) (imOf (ropeRd (kmem bufs)))).mem b j).bits := by
  have hH : HALF = 32 := rfl
  have hD : HD = 64 := rfl
  have hR : ROPE_N = 131072 := rfl
  have hp' : ropeRd (kmem bufs) 1 M_POS = some pos := by simp [ropeRd, hp]
  have hpi : imOf (ropeRd (kmem bufs)) 1 M_POS = pos := by simp [imOf, hp']
  have hgh : ∀ c, c < g → c * HD + HD ≤ g * HD := fun c hc => by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hc
  have hrange : ∀ c (l : Lane) b j, ropeAcc base g pos c l b j →
      (b = 0 ∧ j < base + g * HD) ∨ (b = 2 ∧ j < 2 * ROPE_N * HALF) := by
    intro c l b j ⟨hc, h⟩
    have : l.val < 32 := l.isLt
    have := hgh c hc
    rcases h with ⟨rfl, rfl | rfl⟩ | ⟨rfl, rfl | rfl⟩
    · left; exact ⟨rfl, by rw [hD] at *; omega⟩
    · left; exact ⟨rfl, by rw [hD, hH] at *; omega⟩
    · right; refine ⟨rfl, ?_⟩; have h1 := hpos; rw [hR] at h1; rw [hH, hR]; omega
    · right; refine ⟨rfl, ?_⟩; have h1 := hpos; rw [hR] at h1; rw [hH, hR]; omega
  exact launch_lockstep_one hw hw32 "rope" 3 0 (ropeEW base) g (by rw [hD] at hg; omega) (ropeAcc base g pos)
    (ropeRd (kmem bufs)) (fun c hc => rope_acc base g pos c _ hc hpi)
    (fun c l b j h => by
      rcases hrange c l b j h with ⟨rfl, hj⟩ | ⟨rfl, hj⟩
      · exact ⟨by decide, by decide, by omega⟩
      · exact ⟨by decide, by decide, by rw [hH, hR] at hj; omega⟩)
    (fun c c' l l' b j hc hc' hne h hst => by
      simp only [ropeEW, ropeBodyEW, StoresL, ropeLoIx, ropeHiIx, IdxE.eval, false_or] at hst
      obtain ⟨_, h⟩ := h
      have : l.val < 32 := l.isLt
      have : l'.val < 32 := l'.isLt
      rcases h with ⟨rfl, h⟩ | ⟨rfl, _⟩
      · apply hne
        rcases hst with ⟨_, rfl⟩ | ⟨_, rfl⟩ <;> rcases h with h | h
        · exact mul_add_inj (s := HD) (x := l.val) (y := l'.val) (by omega) (by rw [hD]; omega) (by rw [hD]; omega)
        · exact mul_add_inj (s := HD) (x := l.val + HALF) (y := l'.val) (by omega) (by rw [hD, hH]; omega)
            (by rw [hD]; omega)
        · exact mul_add_inj (s := HD) (x := l.val) (y := l'.val + HALF) (by omega) (by rw [hD]; omega)
            (by rw [hD, hH]; omega)
        · exact mul_add_inj (s := HD) (x := l.val + HALF) (y := l'.val + HALF) (by omega) (by rw [hD, hH]; omega)
            (by rw [hD, hH]; omega)
      · rcases hst with ⟨h1, _⟩ | ⟨h1, _⟩ <;> cases h1)
    hfine bufs bufs' (rope_rdOk _)
    (fun c l b j hc h => by
      rcases hrange c l b j h with ⟨rfl, _⟩ | ⟨rfl, _⟩ <;> simp [ropeRd])
    (fun c hc => rope_fits base g pos c _ hc hp' hpos hbase hg)
    (fun c l b j hc h => by
      rcases hrange c l b j h with ⟨rfl, hj⟩ | ⟨rfl, hj⟩ <;> omega)
    hr

/-- Word `j` of buffer `b` after a rotation's launch on `bufs`, as the warp
    machine's run for head `cta` leaves it. -/
def ropeOut (hw : Hw) (base cta : Nat) (bufs bufs' : Nat → ByteArray) (b : Buf) (j : Nat) : Prop :=
  ({ kmem bufs with bufs := bufs' } : Mem).load (bufW 3 b j) 4 =
    some ((@EWStmt.run F32 (F32.ops hw) _ (ropeEW base) (ws0 3 (kmem bufs)) cta 0 (fun _ _ => 0) (imOf (ropeRd (kmem bufs)))).mem b j).bits

/-- **Every shipped rotation's launch runs its statement**: `g` one-warp blocks
    leave each head's pair in the packed row as the warp machine's run of the
    statement at `F32` does from the launch's memory, reading the table's sine
    and cosine at the meta's position `pos`, where `pos` is within the table
    and the heads within the row. -/
theorem rope_body (p : Nat × Nat) (hp : p ∈ ropeShipped) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray) (pos : Nat) (hpm : rdM 3 (kmem bufs) 1 M_POS = some pos) (hpos : pos < ROPE_N)
    (hsz0 : 4 * (p.1 + p.2 * HD) ≤ (bufs 0).size) (hsz2 : 4 * (2 * ROPE_N * HALF) ≤ (bufs 2).size)
    (hr : LaunchRuns (ropeKernel p.1) p.2 32 (bufArgs 3) bufs bufs' hw) :
    ∀ cta (l : Lane), cta < p.2 →
      ropeOut hw p.1 cta bufs bufs' 0 (p.1 + (cta * HD + l.val)) ∧
      ropeOut hw p.1 cta bufs bufs' 0 (p.1 + (cta * HD + l.val) + HALF) := by
  intro cta l hc
  have hb : p.1 < 2 ^ 31 ∧ p.1 + p.2 * HD ≤ 2 ^ 32 ∧
      Fine (emitGo (fnBody (params 3) (Body.body 3 (ropeEW p.1))) ⟨(params 3).length, 0, [], [], [], none⟩).2 := by
    simp only [ropeShipped, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> exact ⟨by decide, by decide, by decide +kernel, by decide +kernel⟩
  have h := rope_launch hw hw32 p.1 p.2 bufs bufs' pos hpm hpos hb.1 hb.2.1 hsz0 hsz2 hb.2.2 hr cta l
  exact ⟨h 0 _ hc ⟨hc, Or.inl ⟨rfl, Or.inl rfl⟩⟩, h 0 _ hc ⟨hc, Or.inl ⟨rfl, Or.inr rfl⟩⟩⟩

/-- The meta words the softmax reads: the row length, its whole chunks, where
    the remainder starts and how long it is. -/
def smRd (m : Mem) : Buf → Nat → Option Nat :=
  fun b j => if b = 1 ∧ (j = M_SEQ ∨ j = M_CHUNKS ∨ j = M_TAIL ∨ j = M_REM) then rdM 4 m b j else none

/-- Head `cta`'s lane `l` touches its elements of the row's whole chunks in the
    scores and the probabilities, every element of the remainder in the
    scores, the remainder's probabilities if it is lane 0, and the head's
    sink. -/
def smAcc (g seq chunks tail rem : Nat) (cta : Nat) : Acc := fun l b j =>
  cta < g ∧ (((b = 0 ∨ b = 2) ∧ ∃ i, i < chunks ∧ j = cta * seq + (i * 32 + l.val)) ∨
    (b = 0 ∧ ∃ i, i < rem ∧ j = cta * seq + (tail + i)) ∨
    (b = 2 ∧ l = lane0 ∧ ∃ i, i < rem ∧ j = cta * seq + (tail + i)) ∨
    (b = 3 ∧ j = cta))

theorem sm_rdOk (m : Mem) : RdOk 4 m (smRd m) := by
  intro b j n h
  simp only [smRd] at h
  split at h
  · rename_i hb
    obtain ⟨rfl, hj⟩ := hb
    refine ⟨⟨by decide, by decide, ?_⟩, h⟩
    rcases hj with rfl | rfl | rfl | rfl <;> decide
  · cases h

theorem sm_acc (g seq chunks tail rem cta : Nat) (m : Mem) (hc : cta < g) (hct : chunks * 32 ≤ tail)
    (hs : imOf (smRd m) 1 M_SEQ = seq) (hk : imOf (smRd m) 1 M_CHUNKS = chunks)
    (ht : imOf (smRd m) 1 M_TAIL = tail) (hr : imOf (smRd m) 1 M_REM = rem) :
    AccOk cta (smAcc g seq chunks tail rem cta) (smRd m) 0 sinkSoftmaxEW := by
  simp only [sinkSoftmaxEW, smMax, smSum, smSink, smNorm, smMaxReduce, chunkRemReduce, sweepBody, normStep,
    normStep0, normCompute, storeBody, warpReduceMaxE, warpMaxRoundE, smSumBfly, warpRoundE, AccOk, smAcc, smIx,
    smTailIx, rowBaseIx, seqLenIx, sinkIx, IdxE.eval, hs, hk, ht, hr]
  simp only [true_and, and_true, or_true, hc, Nat.add_left_cancel_iff, Nat.add_right_cancel_iff,
    show (0:Nat) ≠ 2 from by decide, show (3:Nat) ≠ 0 from by decide, show (3:Nat) ≠ 2 from by decide,
    show (0:Nat) ≠ 3 from by decide, false_and, or_false]
  have A : ∀ j, j < chunks → ∀ l : Lane, (∃ i, i < chunks ∧ j * 32 = i * 32) ∨
      ∃ i, i < rem ∧ j * 32 + l.val = tail + i := fun j hj l => Or.inl ⟨j, hj, rfl⟩
  have B : ∀ j, j < rem → ∀ l : Lane, (∃ i, i < chunks ∧ tail + j = i * 32 + l.val) ∨ ∃ i, i < rem ∧ j = i :=
    fun j hj l => Or.inr ⟨j, hj, rfl⟩
  have C : ∀ j, j < chunks → ∀ (l l' : Lane), ((∃ i, i < chunks ∧ j * 32 + l.val = i * 32 + l'.val) ∨
      (2 = 0 ∧ ∃ i, i < rem ∧ j * 32 + l.val = tail + i) ∨
      (l' = lane0 ∧ ∃ i, i < rem ∧ j * 32 + l.val = tail + i) ∨ 2 = 3 ∧ cta * seq + (j * 32 + l.val) = cta) →
      l' = l := by
    intro j hj l l' h
    have : l.val < 32 := l.isLt
    have : l'.val < 32 := l'.isLt
    rcases h with ⟨i, hi, he⟩ | ⟨h2, _⟩ | ⟨_, i, hi, he⟩ | ⟨h3, _⟩
    · exact Fin.ext (by omega)
    · exact absurd h2 (by decide)
    · omega
    · exact absurd h3 (by decide)
  have D : ∀ j, j < rem → ∀ (l' : Lane), ((∃ i, i < chunks ∧ tail + j = i * 32 + l'.val) ∨
      (2 = 0 ∧ ∃ i, i < rem ∧ j = i) ∨ (l' = lane0 ∧ ∃ i, i < rem ∧ j = i) ∨ 2 = 3 ∧ cta * seq + (tail + j) = cta) →
      l' = lane0 := by
    intro j hj l' h
    have : l'.val < 32 := l'.isLt
    rcases h with ⟨i, hi, he⟩ | ⟨h2, _⟩ | ⟨hl, _⟩ | ⟨h3, _⟩
    · omega
    · exact absurd h2 (by decide)
    · exact hl
    · exact absurd h3 (by decide)
  repeat' (first | exact A | exact B | exact fun _ => trivial | apply And.intro)
  all_goals intro j hj
  all_goals first
    | exact ⟨A j hj, fun l => ⟨Or.inl ⟨j, hj, rfl⟩, fun l' h => C j hj l l' h⟩⟩
    | exact ⟨B j hj, Or.inr (Or.inr (Or.inl ⟨j, hj, rfl⟩)), fun l' h => D j hj l' h⟩

theorem sm_fits (g seq chunks tail rem cta : Nat) (m : Mem) (hc : cta < g)
    (hs : smRd m 1 M_SEQ = some seq) (hk : smRd m 1 M_CHUNKS = some chunks)
    (ht : smRd m 1 M_TAIL = some tail) (hr : smRd m 1 M_REM = some rem)
    (hct : chunks * 32 ≤ tail) (htr : tail + rem ≤ seq) (hk31 : chunks < 2 ^ 31) (hr31 : rem < 2 ^ 31)
    (hg : g * seq ≤ 2 ^ 32) (hgrid : g ≤ 2 ^ 32) :
    fits (smRd m) cta 0 sinkSoftmaxEW = true := by
  have hs' : smRd m 1 2 = some seq := hs
  have hk' : smRd m 1 3 = some chunks := hk
  have ht' : smRd m 1 4 = some tail := ht
  have hr' : smRd m 1 5 = some rem := hr
  have hcs : cta * seq + seq ≤ g * seq := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hc
  have h1 : cta < 2 ^ 32 := by omega
  have h2 : cta * seq < 2 ^ 32 := by
    rcases Nat.eq_zero_or_pos seq with h | h
    · simp [h]
    · omega
  have hchunk : ∀ j, j < chunks → ∀ l, l < 32 → (idxL (smRd m) cta j l smIx).isSome := by
    intro j hj l hl
    have a1 : j < 2 ^ 32 := by omega
    have a2 : j * 32 < 2 ^ 32 := by omega
    have a3 : j * 32 + l < 2 ^ 32 := by omega
    have a4 : cta * seq + (j * 32 + l) < 2 ^ 32 := by omega
    have a5 : l < 2 ^ 32 := by omega
    simp [smIx, rowBaseIx, seqLenIx, idxL, M_SEQ, hs', h1, h2, a1, a2, a3, a4, a5]
  have htail : ∀ j, j < rem → ∀ l, l < 32 → (idxL (smRd m) cta j l smTailIx).isSome := by
    intro j hj l hl
    have a1 : j < 2 ^ 32 := by omega
    have a3 : tail + j < 2 ^ 32 := by omega
    have a4 : cta * seq + (tail + j) < 2 ^ 32 := by omega
    simp [smTailIx, rowBaseIx, seqLenIx, idxL, M_SEQ, M_TAIL, hs', ht', h1, h2, a1, a3, a4]
  have hsink : ∀ j l, (idxL (smRd m) cta j l sinkIx).isSome := by
    intro j l; simp [sinkIx, idxL, h1]
  simp only [sinkSoftmaxEW, smMax, smSum, smSink, smNorm, smMaxReduce, chunkRemReduce, sweepBody, normStep,
    normStep0, normCompute, storeBody, warpReduceMaxE, warpMaxRoundE, smSumBfly, warpRoundE, fits, M_CHUNKS, M_REM,
    hk', hr', Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range]
  simp (config := { contextual := true }) [hchunk, htail, hsink, hk31, hr31, Body.setsI]

/-- The softmax's launch, every element a head's lane touches. -/
theorem sm_launch (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (g : Nat)
    (bufs bufs' : Nat → ByteArray) (seq chunks tail rem : Nat)
    (hs : rdM 4 (kmem bufs) 1 M_SEQ = some seq) (hk : rdM 4 (kmem bufs) 1 M_CHUNKS = some chunks)
    (ht : rdM 4 (kmem bufs) 1 M_TAIL = some tail) (hr : rdM 4 (kmem bufs) 1 M_REM = some rem)
    (hct : chunks * 32 ≤ tail) (htr : tail + rem ≤ seq) (hk31 : chunks < 2 ^ 31) (hr31 : rem < 2 ^ 31)
    (hg : g * seq ≤ 2 ^ 32) (hgrid : g ≤ 2 ^ 32)
    (hsz0 : 4 * (g * seq) ≤ (bufs 0).size) (hsz2 : 4 * (g * seq) ≤ (bufs 2).size) (hsz3 : 4 * g ≤ (bufs 3).size)
    (hrun : LaunchRuns sinkKernel g 32 (bufArgs 4) bufs bufs' hw) :
    ∀ cta l b j, cta < g → smAcc g seq chunks tail rem cta l b j →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW 4 b j) 4 =
        some ((@EWStmt.run F32 (F32.ops hw) _ sinkSoftmaxEW (ws0 4 (kmem bufs)) cta 0 (fun _ _ => 0) (imOf (smRd (kmem bufs)))).mem b j).bits := by
  have hs' : smRd (kmem bufs) 1 M_SEQ = some seq := by simp [smRd, hs]
  have hk' : smRd (kmem bufs) 1 M_CHUNKS = some chunks := by simp [smRd, hk]
  have ht' : smRd (kmem bufs) 1 M_TAIL = some tail := by simp [smRd, ht]
  have hr' : smRd (kmem bufs) 1 M_REM = some rem := by simp [smRd, hr]
  have hcs : ∀ c, c < g → c * seq + seq ≤ g * seq := fun c hc => by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hc
  have hrow : ∀ c (l : Lane) b j, smAcc g seq chunks tail rem c l b j →
      ((b = 0 ∨ b = 2) ∧ ∃ x, x < seq ∧ j = c * seq + x) ∨ (b = 3 ∧ j = c) := by
    intro c l b j ⟨_, h⟩
    have : l.val < 32 := l.isLt
    rcases h with ⟨hb, i, hi, rfl⟩ | ⟨rfl, i, hi, rfl⟩ | ⟨rfl, _, i, hi, rfl⟩ | ⟨rfl, rfl⟩
    · exact Or.inl ⟨hb, _, by omega, rfl⟩
    · exact Or.inl ⟨Or.inl rfl, _, by omega, rfl⟩
    · exact Or.inl ⟨Or.inr rfl, _, by omega, rfl⟩
    · exact Or.inr ⟨rfl, rfl⟩
  have hrow' : ∀ c (l : Lane) b j, c < g → smAcc g seq chunks tail rem c l b j →
      ((b = 0 ∨ b = 2) ∧ j < g * seq) ∨ (b = 3 ∧ j < g) := by
    intro c l b j hc h
    rcases hrow c l b j h with ⟨hb, x, hx, rfl⟩ | ⟨hb, rfl⟩
    · have := hcs c hc; exact Or.inl ⟨hb, by omega⟩
    · exact Or.inr ⟨hb, hc⟩
  exact launch_lockstep_one hw hw32 "sinkSoftmax" 4 0 sinkSoftmaxEW g hgrid (smAcc g seq chunks tail rem)
    (smRd (kmem bufs))
    (fun c hc => sm_acc g seq chunks tail rem c _ hc hct (by simp [imOf, hs']) (by simp [imOf, hk'])
      (by simp [imOf, ht']) (by simp [imOf, hr']))
    (fun c l b j h => by
      have hc := h.1
      rcases hrow' c l b j hc h with ⟨hb, hj⟩ | ⟨rfl, hj⟩
      · rcases hb with rfl | rfl <;> exact ⟨by decide, by decide, by omega⟩
      · exact ⟨by decide, by decide, by omega⟩)
    (fun c c' l l' b j hc hc' hne h hst => by
      simp only [sinkSoftmaxEW, smMax, smSum, smSink, smNorm, smMaxReduce, chunkRemReduce, sweepBody, normStep,
        normStep0, normCompute, storeBody, warpReduceMaxE, warpMaxRoundE, smSumBfly, warpRoundE, StoresL, smIx,
        smTailIx, rowBaseIx, seqLenIx, IdxE.eval, false_or, or_false, imOf, hs', hk', ht', hr', Option.getD_some]
        at hst
      have : l'.val < 32 := l'.isLt
      apply hne
      rcases hst with ((⟨_, _, hf⟩ | ⟨_, _, hf⟩) | (⟨_, _, hf⟩ | ⟨_, _, hf⟩)) | ⟨t, ht1, hb2, he⟩ | ⟨t, ht1, _, hb2, he⟩
      · exact hf.elim
      · exact hf.elim
      · exact hf.elim
      · exact hf.elim
      · subst hb2
        rcases hrow c l 2 j h with ⟨_, x, hx, hj⟩ | ⟨h3, _⟩
        · exact mul_add_inj (s := seq) (x := x) (y := t * 32 + l'.val) (hj.symm.trans he) hx (by omega)
        · exact absurd h3 (by decide)
      · subst hb2
        rcases hrow c l 2 j h with ⟨_, x, hx, hj⟩ | ⟨h3, _⟩
        · exact mul_add_inj (s := seq) (x := x) (y := tail + t) (hj.symm.trans he) hx (by omega)
        · exact absurd h3 (by decide))
    ⟨by decide +kernel, by decide +kernel⟩ bufs bufs' (sm_rdOk _)
    (fun c l b j hc h => by
      rcases hrow' c l b j hc h with ⟨hb, _⟩ | ⟨rfl, _⟩
      · rcases hb with rfl | rfl <;> simp [smRd]
      · simp [smRd])
    (fun c hc => sm_fits g seq chunks tail rem c _ hc hs' hk' ht' hr' hct htr hk31 hr31 hg hgrid)
    (fun c l b j hc h => by
      rcases hrow' c l b j hc h with ⟨hb, hj⟩ | ⟨rfl, hj⟩
      · rcases hb with rfl | rfl <;> omega
      · omega)
    hrun

/-- Word `j` of buffer `b` after the softmax's launch on `bufs`, as the warp
    machine's run for head `cta` leaves it. -/
def smOut (hw : Hw) (cta : Nat) (bufs bufs' : Nat → ByteArray) (b : Buf) (j : Nat) : Prop :=
  ({ kmem bufs with bufs := bufs' } : Mem).load (bufW 4 b j) 4 =
    some ((@EWStmt.run F32 (F32.ops hw) _ sinkSoftmaxEW (ws0 4 (kmem bufs)) cta 0 (fun _ _ => 0) (imOf (smRd (kmem bufs)))).mem b j).bits

/-- **The softmax's launch runs its statement**: `g` one-warp blocks leave each
    head's row of probabilities, its whole chunks and its remainder, as the
    warp machine's run of the statement at `F32` does from the launch's memory,
    with `hw`'s approximations, where `seq`, `chunks`, `tail` and `rem` are the
    meta's row length, whole chunks, remainder start and remainder length, the
    chunks end where the remainder starts, the remainder ends within the row
    and the rows within the buffers. -/
theorem sm_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (g : Nat)
    (bufs bufs' : Nat → ByteArray) (seq chunks tail rem : Nat)
    (hs : rdM 4 (kmem bufs) 1 M_SEQ = some seq) (hk : rdM 4 (kmem bufs) 1 M_CHUNKS = some chunks)
    (ht : rdM 4 (kmem bufs) 1 M_TAIL = some tail) (hr : rdM 4 (kmem bufs) 1 M_REM = some rem)
    (hct : chunks * 32 ≤ tail) (htr : tail + rem ≤ seq) (hk31 : chunks < 2 ^ 31) (hr31 : rem < 2 ^ 31)
    (hg : g * seq ≤ 2 ^ 32) (hgrid : g ≤ 2 ^ 32)
    (hsz0 : 4 * (g * seq) ≤ (bufs 0).size) (hsz2 : 4 * (g * seq) ≤ (bufs 2).size) (hsz3 : 4 * g ≤ (bufs 3).size)
    (hrun : LaunchRuns sinkKernel g 32 (bufArgs 4) bufs bufs' hw) :
    ∀ cta, cta < g →
      (∀ i, i < chunks → ∀ l : Lane, smOut hw cta bufs bufs' 2 (cta * seq + (i * 32 + l.val))) ∧
      (∀ i, i < rem → smOut hw cta bufs bufs' 2 (cta * seq + (tail + i))) := by
  intro cta hc
  have h := sm_launch hw hw32 g bufs bufs' seq chunks tail rem hs hk ht hr hct htr hk31 hr31 hg hgrid hsz0 hsz2 hsz3
    hrun cta
  exact ⟨fun i hi l => h l 2 _ hc ⟨hc, Or.inl ⟨Or.inr rfl, i, hi, rfl⟩⟩,
    fun i hi => h lane0 2 _ hc ⟨hc, Or.inr (Or.inr (Or.inl ⟨rfl, rfl, i, hi, rfl⟩))⟩⟩

end GptOssProofs
