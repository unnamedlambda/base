module
public import Host.KernelCorpus
meta import Host.KernelCorpus
public import AlgorithmLib.Kernel.Elementwise
meta import AlgorithmLib.Kernel.Elementwise
public import AlgorithmLib.Proof.LibStep
meta import AlgorithmLib.Proof.LibStep
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# rowSum, a kernel with a loop, from its thread to what a launch leaves

The corpus's `rowSum` kernel (`KernelCorpus.rowSum`, which
`tools/kernel_agree.py` runs on the GPU): thread `i < n` sums row `i` of `x`
(`m` floats, added in order as the body's float adds round them, `acc`) into
`out[i]`. One thread by the program logic, its loop by `wp_forLoopAcc`
(`thread_wp`, `thread_spec`); as an elementwise kernel (`elem`), what a launch
leaves (`launch_spec`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts AlgorithmLib.LibSpec AlgorithmLib.Kernel

namespace KernelRowSum

def params : List ClifTy := [.i64, .i64, .i32, .i64]
def K : Kernel := { name := "rowSum", params, code := emit KernelCorpus.rowSum (idParams ++ params) }

/-- The bits `fconst32 0.0` makes: the sum of no terms. -/
def z0 : UInt64 := ((0.0 : Float).toFloat32.toBits.toUInt64) &&& widthMask .f32

/-- The first `k` terms summed in order, as the body's float adds round them
    on a platform whose NaN rule is `nan`. -/
def acc (nan : ClifTy → UInt64 → UInt64) (X : Nat → UInt64) : Nat → UInt64
  | 0 => z0
  | k + 1 => nanFix nan .f32 (ofF32 (f32 (acc nan X k) + f32 (X k)))

attribute [local irreducible] wp

/-- **What one thread does**: thread `i < n` sums row `i` of `x` (`m` floats)
    into `out[i]`; the rest leave memory as it was. -/
theorem thread_wp (cfg : Cfg) (w : World) (b t bd n : UInt32) (px po mm sm : UInt64)
    (X : Nat → UInt64) (m : Mem)
    (hx : (b * bd + t) < n → ∀ k, k < mm.toNat →
      w.mem.load (px + (s32 (b * bd + t) * mm) <<< (c64 2 % 64) + UInt64.ofNat k <<< (c64 2 % 64)) 4 = some (X k))
    (hs : (b * bd + t) < n →
      w.mem.store (po + s32 (b * bd + t) <<< (c64 2 % 64)) 4 (acc w.mem.nan X mm.toNat) = some m) :
    wp cfg 0 { ok := fun _ _ => True, faultOk := false } (fnBody (idParams ++ params) KernelCorpus.rowSum)
      (fun _ _ w' => if (b * bd + t) < n then w'.mem = m ∧ ∃ p x, w'.obs = .store p 4 x :: w.obs
                     else w' = w)
      #[.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po, .sc .i32 (s32 n), .sc .i64 mm] w := by
  rw [fnBody_of (by decide +kernel)]
  simp only [KernelCorpus.rowSum, KernelCorpus.gid, entryParams, wp_bind, wp_params, wp_ret]
  simp only [idParams, params, List.cons_append, List.nil_append, carriesFrom, Vals.head, Vals.snd, Vals.thd,
    Vals.tail, Vals.drop, Nat.zero_add, Nat.reduceAdd]
  have h4 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po, .sc .i32 (s32 n), .sc .i64 mm])[4]? = some (.sc .i64 px) := rfl
  have h5 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po, .sc .i32 (s32 n), .sc .i64 mm])[5]? = some (.sc .i64 po) := rfl
  have h6 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po, .sc .i32 (s32 n), .sc .i64 mm])[6]? = some (.sc .i32 (s32 n)) := rfl
  have h7 : (#[V.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po, .sc .i32 (s32 n), .sc .i64 mm])[7]? = some (.sc .i64 mm) := rfl
  refine wp_imul32 (x := b) (y := bd) rfl rfl ?_
  refine wp_iadd32 (x := b * bd) (y := t) (by look) rfl ?_
  simp only [when, wp_bind]
  refine wp_ifteT (Qj := fun _ _ w' => if (b * bd + t) < n then w'.mem = m ∧ ∃ p x, w'.obs = .store p 4 x :: w.obs
                     else w' = w) rfl (by look) (by look) ?_ ?_ ?_
  · intro hc
    have hlt : (b * bd + t) < n := by rw [cmp32_ult] at hc; exact of_decide_eq_true hc
    repeat step
    refine wp_fconst32 ?_
    rw [wp_bind]
    refine wp_forLoopAcc (Inv := fun k a w' => w' = w ∧ a = acc w.mem.nan X k) (n := mm) (a0 := z0)
      (by look) get_push_size ⟨rfl, rfl⟩ ?_ ?_
    · intro J' Γ' wi k a si sa hE hk ⟨hw, ha⟩ hsi hsa
      subst hw; subst ha
      repeat step
      refine wp_loadF32 (by look) (hx hlt k hk) ?_
      refine wp_fadd32 (x := acc wi.mem.nan X k) (y := X k) (by look) (by look) ?_
      exact ⟨by extc, _, get_push_size, rfl, rfl⟩
    · intro Γ' w' r a hE hr ⟨hw, ha⟩
      subst hw; subst ha
      repeat step
      refine wp_storeF32 (by look) (by look) (hs hlt) ?_
      rw [wp_pure]
      simp [hlt, obsStore]
  · intro hc ne hne
    have hlt : ¬ (b * bd + t) < n := by rw [cmp32_ult] at hc; exact of_decide_eq_false hc
    rw [wp_pure]
    simp only [hlt, if_false]
  · intro nJ Γ' jv vs w' hle hvs hq
    rw [wp_pure]
    exact hq

theorem fine : Fine (emitGo (fnBody (idParams ++ params) KernelCorpus.rowSum) ⟨(idParams ++ params).length, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- **What a rowSum thread leaves**, when it finishes. -/
theorem thread_spec (b t bd n : UInt32) (px po mm sm : UInt64) (X : Nat → UInt64) (m : Mem) (w u : World)
    (hx : (b * bd + t) < n → ∀ k, k < mm.toNat →
      w.mem.load (px + (s32 (b * bd + t) * mm) <<< (c64 2 % 64) + UInt64.ofNat k <<< (c64 2 % 64)) 4 = some (X k))
    (hs : (b * bd + t) < n →
      w.mem.store (po + s32 (b * bd + t) <<< (c64 2 % 64)) 4 (acc w.mem.nan X mm.toNat) = some m)
    {ans : List Answer} {steps : Nat}
    (hr : thread K.code ans [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po, .sc .i32 (s32 n), .sc .i64 mm] w Hw.rounded steps = .ok u) :
    if (b * bd + t) < n then u.mem = m ∧ ∃ p x, u.obs = .store p 4 x :: w.obs else u = w := by
  refine thread_of_triple (Q := fun u => if (b * bd + t) < n then u.mem = m ∧ ∃ p x, u.obs = .store p 4 x :: w.obs else u = w) (fun cfg => ?_) hr
  exact emit_triple (cfg := cfg) (J := { ok := fun _ _ => True, faultOk := false })
    (P0 := At (List.toArray [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po, .sc .i32 (s32 n), .sc .i64 mm]) w)
    (PT.conseq (P' := fun Γ' w' => Γ' = (List.toArray [.sc .i32 (s32 b), .sc .i32 (s32 t), .sc .i32 (s32 bd), .sc .i64 sm,
        .sc .i64 px, .sc .i64 po, .sc .i32 (s32 n), .sc .i64 mm]) ∧ w' = w)
      (wp_sound cfg 0 _ (fnBody (idParams ++ params) KernelCorpus.rowSum) _)
      (fun Γ' w' h => by rw [h.1, h.2]; exact thread_wp cfg w b t bd n px po mm sm X m hx hs)
      (fun _ _ _ h => h)) fine (fun Γ' w' h => ⟨by rw [h.1]; rfl, h⟩)

/-- `x` and `out`: device buffers 0 and 1. -/
abbrev px : UInt64 := addrOf (.buf ⟨0, by decide⟩) 0
abbrev po : UInt64 := addrOf (.buf ⟨1, by decide⟩) 0
def args (n : UInt32) (mm : UInt64) : List V := [.sc .i64 px, .sc .i64 po, .sc .i32 (s32 n), .sc .i64 mm]

abbrev rowAddr (mm : UInt64) (j k : Nat) : UInt64 :=
  px + (s32 (UInt32.ofNat j) * mm) <<< (c64 2 % 64) + UInt64.ofNat k <<< (c64 2 % 64)
abbrev outAddr (j : Nat) : UInt64 := po + s32 (UInt32.ofNat j) <<< (c64 2 % 64)

theorem shl2 (x : UInt64) (hx : x.toNat < 2 ^ 62) : x <<< (c64 2 % 64) = UInt64.ofNat (4 * x.toNat) := by
  rw [show c64 2 % 64 = (2 : UInt64) from by decide]
  apply UInt64.toNat_inj.mp
  simp [UInt64.toNat_shiftLeft, Nat.shiftLeft_eq]
  omega

theorem rowAddr_eq (mm : UInt64) (j k : Nat) (hj : j < 2 ^ 32) (hk : k < 2 ^ 32)
    (hb : 4 * (j * mm.toNat + k) < 2 ^ 36) :
    rowAddr mm j k = addrOf (.buf ⟨0, by decide⟩) 0 + UInt64.ofNat (4 * (j * mm.toNat + k)) := by
  have hjm : j * mm.toNat < 2 ^ 34 := by omega
  have hs : (s32 (UInt32.ofNat j) * mm).toNat = j * mm.toNat := by
    simp [s32, UInt64.toNat_mul, UInt32.toNat_ofNat, Nat.mod_eq_of_lt hj]; omega
  rw [rowAddr, shl2 _ (by omega), shl2 _ (by simp; omega), hs, UInt64.add_assoc, ← UInt64.ofNat_add]
  simp only [UInt64.toNat_ofNat']
  congr 2
  rw [Nat.mod_eq_of_lt (by omega)]; omega

theorem outAddr_eq (j : Nat) (hj : j < 2 ^ 32) : outAddr j = addrOf (.buf ⟨1, by decide⟩) 0 + UInt64.ofNat (4 * j) := by
  rw [outAddr, shl2 _ (by simp [s32]; omega)]
  simp [s32, UInt32.toNat_ofNat, Nat.mod_eq_of_lt hj]
  rfl

/-- **rowSum as an elementwise kernel**: thread `j < n` reads row `j` of `x` and
    stores its sum at `out[j]`. -/
def elem (n : UInt32) (mm : UInt64) (X : Nat → UInt64) : Elem where
  out := outAddr
  val j := acc gpuNaN (fun k => X (j * mm.toNat + k)) mm.toNat
  active j := decide (UInt32.ofNat j < n)
  reads j := (List.range mm.toNat).map fun k => (rowAddr mm j k, X (j * mm.toNat + k))

theorem elem_apart (n : UInt32) (mm : UInt64) (X : Nat → UInt64) (N : Nat) (hN : N ≤ 2 ^ 32)
    (hb : 4 * (N * mm.toNat) ≤ 2 ^ 36) (hm : mm.toNat < 2 ^ 32) : (elem n mm X).Apart N where
  buf j hj _ := by
    refine ⟨⟨1, by decide⟩, 4 * j, ?_⟩
    rw [show (elem n mm X).out j = outAddr j from rfl, outAddr_eq j (by omega)]
    exact decode_buf _ (4 * j) (by omega)
  outs i j hi hj hne _ _ := by
    refine ⟨.buf ⟨1, by decide⟩, 4 * i, .buf ⟨1, by decide⟩, 4 * j, ?_, ?_, Or.inr ?_⟩
    · rw [show (elem n mm X).out i = outAddr i from rfl, outAddr_eq i (by omega)]; exact decode_buf _ (4 * i) (by omega)
    · rw [show (elem n mm X).out j = outAddr j from rfl, outAddr_eq j (by omega)]; exact decode_buf _ (4 * j) (by omega)
    · omega
  reads i j hi hj hne _ _ p hp := by
    obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hp
    have hk := List.mem_range.mp hk
    have hjm : j * mm.toNat + k < N * mm.toNat := by
      have : (j + 1) * mm.toNat ≤ N * mm.toNat := Nat.mul_le_mul_right _ hj
      rw [Nat.succ_mul] at this; omega
    refine ⟨.buf ⟨1, by decide⟩, 4 * i, .buf ⟨0, by decide⟩, 4 * (j * mm.toNat + k), ?_, ?_, Or.inl (by simp)⟩
    · rw [show (elem n mm X).out i = outAddr i from rfl, outAddr_eq i (by omega)]; exact decode_buf _ (4 * i) (by omega)
    · rw [rowAddr_eq mm j k (by omega) (by omega) (by omega)]; exact decode_buf _ _ (by omega)

theorem elem_threads (n : UInt32) (mm : UInt64) (X : Nat → UInt64) (grid bd : Nat) :
    Elem.Threads K (args n mm) grid bd (elem n mm X) := by
  intro b t _ _ s s' ans steps hnan hP hr
  by_cases ha : UInt32.ofNat (b * bd + t) < n
  · have hact : (elem n mm X).active (b * bd + t) = true := by simp only [elem, decide_eq_true_eq]; exact ha
    obtain ⟨hrd, hs⟩ := hP hact
    obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hs ((elem n mm X).val (b * bd + t)))
    simp only [elem, List.mem_map, List.mem_range, forall_exists_index, and_imp] at hrd
    have hx : ∀ k, k < mm.toNat → s.mem.load (rowAddr mm (b * bd + t) k) 4 = some (X ((b * bd + t) * mm.toNat + k)) :=
      fun k hk => hrd (rowAddr mm (b * bd + t) k, X ((b * bd + t) * mm.toNat + k)) k hk rfl
    have h := thread_spec (UInt32.ofNat b) (UInt32.ofNat t) (UInt32.ofNat bd) n px po mm (addrOf .arena 0)
      (fun k => X ((b * bd + t) * mm.toNat + k)) m s s' (fun _ k hk => by rw [gi_eq]; exact hx k hk)
      (fun _ => by rw [gi_eq, hnan]; exact hm) hr
    rw [gi_eq, if_pos ha] at h
    obtain ⟨hmem, p, x, hobs⟩ := h
    refine ⟨?_, collCalls_store hobs⟩
    unfold Elem.eff put
    rw [if_pos hact, hm, Option.getD_some, hmem]
  · have h := thread_spec (UInt32.ofNat b) (UInt32.ofNat t) (UInt32.ofNat bd) n px po mm (addrOf .arena 0)
      (fun _ => 0) s.mem s s' (fun h => absurd (gi_eq b bd t ▸ h) ha) (fun h => absurd (gi_eq b bd t ▸ h) ha) hr
    rw [gi_eq, if_neg ha] at h
    subst h
    refine ⟨?_, collCalls_self s'⟩
    unfold Elem.eff
    rw [if_neg (by simp only [elem, decide_eq_true_eq]; exact ha)]

/-- Every partial sum is a 32-bit word: the first is `0.0`'s, the others
    `Float32` adds'. -/
theorem acc_lt (nan : ClifTy → UInt64 → UInt64) (X : Nat → UInt64) : ∀ k, (acc nan X k).toNat < 2 ^ 32
  | 0 => by
    show (((0.0 : Float).toFloat32.toBits.toUInt64) &&& widthMask .f32).toNat < 2 ^ 32
    rw [UInt64.toNat_and]
    exact Nat.lt_of_le_of_lt Nat.and_le_left (by simp only [UInt32.toNat_toUInt64]; exact UInt32.toNat_lt _)
  | _ + 1 => nanFix_lt _ (ofF32_lt _)

/-- **What a launch of rowSum leaves**: after `grid` blocks of `bd` threads,
    `out[j]` holds row `j`'s sum for every `j < n` the grid covers. -/
theorem launch_spec (n : UInt32) (mm : UInt64) (X : Nat → UInt64) (grid bd : Nat)
    (hsz : grid * bd ≤ 2 ^ 32) (hb : 4 * (grid * bd * mm.toNat) ≤ 2 ^ 36) (hm : mm.toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray)
    (hpre : ∀ j, j < grid * bd → (elem n mm X).Pre (kmem bufs) j)
    (hr : LaunchRuns K grid bd (args n mm) bufs bufs') :
    ∀ j, j < grid * bd → UInt32.ofNat j < n →
      ({ kmem bufs with bufs := bufs' } : Mem).load (outAddr j) 4 =
        some (acc gpuNaN (fun k => X (j * mm.toNat + k)) mm.toNat) :=
  fun j hj ha => Elem.launch_spec rfl (elem_apart n mm X _ hsz hb hm) (elem_threads n mm X grid bd)
    bufs bufs' hpre (fun _ _ _ => acc_lt _ _ _) hr j hj (by simp [elem, ha])

end KernelRowSum
