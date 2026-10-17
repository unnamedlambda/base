module
public import AlgorithmLib.Check.Calls
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.PreSound` — what `preB` decides is the precondition

**`preB_sound`**: where `preB` accepts a call on arguments the checker
describes, `Contracts.Pre` holds of the bits they carry. One lemma per entry
point, each from the atoms' own lemmas.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState holds_of_mem Part readyIn streamsOf streamsOf_holds partyOf partyOf_eq evId evId_spec)

theorem ArgBits.cons_inv {Γ : Env} {w : World} {a : AV} {as : List AV} {bits : List UInt64}
    (h : ArgBits Γ w (a :: as) bits) :
    ∃ x xs, bits = x :: xs ∧ (∃ t, a.Desc Γ w (.sc t x)) ∧ ArgBits Γ w as xs := by
  match bits, h with
  | x :: xs, ⟨hx, hxs⟩ => exact ⟨x, xs, rfl, hx, hxs⟩

theorem ArgBits.nil_inv {Γ : Env} {w : World} {bits : List UInt64} (h : ArgBits Γ w [] bits) : bits = [] := by
  match bits, h with
  | [], _ => rfl

/-- Take apart the bits of a literal argument list, naming each argument's
    description. -/
syntax "argbits " ident " [" ident,* "]" : tactic
set_option hygiene false in
macro_rules
  | `(tactic| argbits $hb []) => `(tactic| (replace $hb := ArgBits.nil_inv $hb; subst $hb))
  | `(tactic| argbits $hb [$h]) =>
      `(tactic| (replace $hb := ArgBits.cons_inv $hb; obtain ⟨_, _, e, $h, $hb⟩ := $hb; subst e; argbits $hb []))
  | `(tactic| argbits $hb [$h, $hs,*]) =>
      `(tactic| (replace $hb := ArgBits.cons_inv $hb; obtain ⟨_, _, e, $h, $hb⟩ := $hb; subst e; argbits $hb [$hs,*]))

-- ---------------------------------------------------------------------------
-- Atoms, in the precondition's vocabulary
-- ---------------------------------------------------------------------------

variable {Γ : Env} {w : World} {S : TState} {rb : Nat → Nat → Bool}

theorem fits_readable {x : UInt64} {k j : Nat} (hf : Fits w.mem x k) (hj : j ≤ k) : Readable w.mem x j := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  exact Contracts.copyOut_isSome hd hp (by omega) hz (by omega)

theorem fits_writable {x : UInt64} {k j : Nat} (hf : Fits w.mem x k) (hj : j ≤ k) : Writable w.mem x j := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  intro src hsrc
  exact Contracts.copyIn_isSome src hd hp (by omega) hz (by omega)

theorem fits_slot8 {x : UInt64} (hf : Fits w.mem x 8) : Contracts.Slot8 w.mem x := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := hf
  exact fun v => Contracts.store_isSome hd hp hs hz v

theorem fits_load8 {x : UInt64} (hf : Fits w.mem x 8) : Contracts.Load8 w.mem x := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := hf
  exact Contracts.load_isSome hd hp hs hz

theorem readIds_of_fits {p : UInt64} {n : Nat} (hf : Fits w.mem p (4 * n)) : (readIds w.mem p n).isSome = true := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  have hl : ((List.range n).mapM fun i => w.mem.load (p + UInt64.ofNat (4 * i)) 4).isSome = true := by
    refine Contracts.mapM_isSome_of fun i hi => ?_
    have hi := List.mem_range.mp hi
    have hd' := Sem.decodeAddr_add (i := 4 * i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
    exact Contracts.load_isSome hd' hp (by omega) hz
  obtain ⟨raw, hraw⟩ := Option.isSome_iff_exists.mp hl
  unfold readIds
  rw [hraw]; rfl

theorem readBinds_of_fits {p : UInt64} {n : Nat} (hf : Fits w.mem p (8 * n)) :
    (readBinds w.mem p n).isSome = true := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  unfold readBinds
  refine Contracts.mapM_isSome_of fun i hi => ?_
  have hi := List.mem_range.mp hi
  have hd1 := Sem.decodeAddr_add (i := 8 * i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  have hd2 := Sem.decodeAddr_add (i := 8 * i + 4) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp (Contracts.load_isSome (n := 4) hd1 hp (by omega) hz)
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (Contracts.load_isSome (n := 4) hd2 hp (by omega) hz)
  rw [ha, hb]; rfl

theorem bindsReady_of_fits {nBufs bindPtr : UInt64} {p : Nat} (hS : S.holds w) (hq : readyIn S p = true)
    (hf : Fits w.mem bindPtr (4 * (asI32 nBufs).toNat)) : BindsReady w p nBufs bindPtr := by
  obtain ⟨ids, hids⟩ := Option.isSome_iff_exists.mp (readIds_of_fits hf)
  exact ⟨ids, hids, fun id _ _ => Contracts.ready_of_state hS hq _ _⟩

theorem memN_fits' {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w) (hd : ∃ t, p.Desc Γ w (.sc t x))
    (h : memN S p k = true) : Fits w.mem x k := memN_spec hS hd h

theorem memB_fits {p n : AV} {x k : UInt64} (hS : S.holds w) (hp : ∃ t, p.Desc Γ w (.sc t x))
    (hn : ∃ t, n.Desc Γ w (.sc t k)) (h : memB S p n = true) : Fits w.mem x k.toNat :=
  memB_spec hS hp hn h

-- ---------------------------------------------------------------------------
-- Entry points
-- ---------------------------------------------------------------------------

section entries

variable {bits : List UInt64}

theorem pre_cudaInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits) (h : preB .cudaInit [s] S rb = true) :
    Contracts.Pre .cudaInit bits w := by
  dsimp only [preB, preCudaInit] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h) _

theorem pre_cudaCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .cudaCleanup [s] S rb = true) : Contracts.Pre .cudaCleanup bits w := by
  dsimp only [preB, preCudaCleanup] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h) _

theorem pre_cudaCreateBuffer {c n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, n] bits)
    (h : preB .cudaCreateBuffer [c, n] S rb = true) : Contracts.Pre .cudaCreateBuffer bits w := by
  dsimp only [preB, preCudaCreateBuffer] at h
  argbits hb [hc, hn]
  exact ctxB_sound hS hc h

theorem pre_cudaSync {c : AV} (hS : S.holds w) (hb : ArgBits Γ w [c] bits)
    (h : preB .cudaSync [c] S rb = true) : Contracts.Pre .cudaSync bits w := by
  dsimp only [preB, preCudaSync] at h
  argbits hb [hc]
  exact ctxB_sound hS hc h

theorem pre_cudaUpload {c b d n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, d, n] bits)
    (h : preB .cudaUpload [c, b, d, n] S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .cudaUpload bits w := by
  dsimp only [preB, preCudaUpload] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hm⟩, hq⟩ := h
  argbits hb [hc, hb', hd, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ _ =>
    ⟨fits_readable (memBR_fits hS hd hn hm (hrb 2 3)) (Nat.le_refl _), Contracts.ready_of_state hS hq _ _⟩⟩

theorem pre_cudaDownload {c b d n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, d, n] bits)
    (h : preB .cudaDownload [c, b, d, n] S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .cudaDownload bits w := by
  dsimp only [preB, preCudaDownload] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hm⟩, hq⟩ := h
  argbits hb [hc, hb', hd, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ _ =>
    ⟨fits_writable (memBR_fits hS hd hn hm (hrb 2 3)) (Nat.le_refl _), Contracts.ready_of_state hS hq _ _⟩⟩

theorem pre_cudaUploadOffset {c b o d n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, o, d, n] bits)
    (h : preB .cudaUploadOffset [c, b, o, d, n] S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .cudaUploadOffset bits w := by
  dsimp only [preB, preCudaUploadOffset] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hm⟩, hq⟩ := h
  argbits hb [hc, hb', ho, hd, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ =>
    ⟨fits_readable (memBR_fits hS hd hn hm (hrb 3 4)) (Nat.le_refl _), Contracts.ready_of_state hS hq _ _⟩⟩

theorem pre_cudaDownloadOffset {c b o d n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, o, d, n] bits)
    (h : preB .cudaDownloadOffset [c, b, o, d, n] S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .cudaDownloadOffset bits w := by
  dsimp only [preB, preCudaDownloadOffset] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hm⟩, hq⟩ := h
  argbits hb [hc, hb', ho, hd, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ =>
    ⟨fits_writable (memBR_fits hS hd hn hm (hrb 3 4)) (Nat.le_refl _), Contracts.ready_of_state hS hq _ _⟩⟩

theorem pre_cudaLaunch {c kp nb bp gx gy gz bx by_ bz : AV} (hS : S.holds w)
    (hb : ArgBits Γ w [c, kp, nb, bp, gx, gy, gz, bx, by_, bz] bits)
    (h : preB .cudaLaunch [c, kp, nb, bp, gx, gy, gz, bx, by_, bz] S rb = true) :
    Contracts.Pre .cudaLaunch bits w := by
  dsimp only [preB, preCudaLaunch] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨hc', ho⟩, hcap⟩, hk⟩, hq⟩, hm⟩ := h
  argbits hb [hc, hkp, hnb, hbp, h5, h6, h7, h8, h9, h10]
  split at hm
  · rename_i k hk'
    have hx := known_bits hnb hk'
    subst hx
    exact ⟨ctxB_sound hS hc hc', Contracts.keepsSize_of_state hS ho, Contracts.capture_of_state hS hcap,
      cstrB_sound hS hkp hk, bindsReady_of_fits hS hq (memN_fits' hS hbp hm)⟩
  · cases hm

theorem pre_cudaLaunchNamed {c kp np nb bp gx gy gz bx by_ bz : AV} (hS : S.holds w)
    (hb : ArgBits Γ w [c, kp, np, nb, bp, gx, gy, gz, bx, by_, bz] bits)
    (h : preB .cudaLaunchNamed [c, kp, np, nb, bp, gx, gy, gz, bx, by_, bz] S rb = true) :
    Contracts.Pre .cudaLaunchNamed bits w := by
  dsimp only [preB, preCudaLaunchNamed] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨hc', ho⟩, hcap⟩, hk⟩, hn⟩, hq⟩, hm⟩ := h
  argbits hb [hc, hkp, hnp, hnb, hbp, h5, h6, h7, h8, h9, h10]
  split at hm
  · rename_i k hk'
    have hx := known_bits hnb hk'
    subst hx
    exact ⟨ctxB_sound hS hc hc', Contracts.keepsSize_of_state hS ho, Contracts.capture_of_state hS hcap,
      cstrB_sound hS hkp hk, cstrB_sound hS hnp hn, bindsReady_of_fits hS hq (memN_fits' hS hbp hm)⟩
  · cases hm

theorem streamReady_sound {sid : AV} {x : UInt64} (hS : S.holds w) (hd : ∃ t, sid.Desc Γ w (.sc t x))
    (h : streamReady S sid = true) : ∀ p, w.dev.party? (asI32 x) = some p → readyIn S p = true := by
  unfold streamReady at h
  split at h
  · rename_i s l hs hl
    have hx := known_bits hd hs
    subst hx
    intro p hp
    rw [partyOf_eq (streamsOf_holds hS hl)] at hp
    split at h
    · rename_i p' hp'
      rw [hp'] at hp; cases hp; exact h
    · rename_i hp'
      rw [hp'] at hp; cases hp
  · cases h

theorem pre_cudaLaunchOnStream {c kp nb bp gx gy gz bx by_ bz sid : AV} (hS : S.holds w)
    (hb : ArgBits Γ w [c, kp, nb, bp, gx, gy, gz, bx, by_, bz, sid] bits)
    (h : preB .cudaLaunchOnStream [c, kp, nb, bp, gx, gy, gz, bx, by_, bz, sid] S rb = true) :
    Contracts.Pre .cudaLaunchOnStream bits w := by
  dsimp only [preB, preCudaLaunchOnStream] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨hc', ho⟩, hcap⟩, hk⟩, hq⟩, hm⟩ := h
  argbits hb [hc, hkp, hnb, hbp, h5, h6, h7, h8, h9, h10, hsid]
  split at hm
  · rename_i k hk'
    have hx := known_bits hnb hk'
    subst hx
    exact ⟨ctxB_sound hS hc hc', Contracts.keepsSize_of_state hS ho, Contracts.capture_of_state hS hcap,
      cstrB_sound hS hkp hk,
      fun p hp => bindsReady_of_fits hS (streamReady_sound hS hsid hq p hp) (memN_fits' hS hbp hm)⟩
  · cases hm

theorem pre_cudaStreamWaitEvent {c s e : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, s, e] bits)
    (h : preB .cudaStreamWaitEvent [c, s, e] S rb = true) : Contracts.Pre .cudaStreamWaitEvent bits w := by
  dsimp only [preB, preCudaStreamWaitEvent] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hcap⟩, he⟩ := h
  argbits hb [hc, hs, hev]
  split at he
  · rename_i eid heid
    have hx := known_bits hev heid
    subst hx
    split at he
    · rename_i n hn
      obtain ⟨clk, hclk, -⟩ := (holds_of_mem hS he : (Fact.evAll n).holds w)
      refine ⟨ctxB_sound hS hc hc', Contracts.capture_of_state hS hcap, fun c' hc'' => ?_⟩
      rw [← evId_spec hn, hclk] at hc''
      simp at hc''
    · cases he
  · cases he

theorem pre_gpuInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits) (h : preB .gpuInit [s] S rb = true) :
    Contracts.Pre .gpuInit bits w := by
  dsimp only [preB, preGpuInit] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h)

theorem pre_gpuCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .gpuCleanup [s] S rb = true) : Contracts.Pre .gpuCleanup bits w := by
  dsimp only [preB, preGpuCleanup] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h)

theorem pre_gpuCreateBuffer {c n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, n] bits)
    (h : preB .gpuCreateBuffer [c, n] S rb = true) : Contracts.Pre .gpuCreateBuffer bits w := by
  dsimp only [preB, preGpuCreateBuffer] at h
  argbits hb [hc, hn]
  exact ctxB_sound hS hc h

theorem pre_gpuDispatch {c p x y z : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, p, x, y, z] bits)
    (h : preB .gpuDispatch [c, p, x, y, z] S rb = true) : Contracts.Pre .gpuDispatch bits w := by
  dsimp only [preB, preGpuDispatch] at h
  argbits hb [hc, hp, hx, hy, hz]
  exact ctxB_sound hS hc h

theorem pre_gpuCreatePipeline {c sp bp n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, sp, bp, n] bits)
    (h : preB .gpuCreatePipeline [c, sp, bp, n] S rb = true) : Contracts.Pre .gpuCreatePipeline bits w := by
  dsimp only [preB, preGpuCreatePipeline] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hs⟩, hm⟩ := h
  argbits hb [hc, hsp, hbp, hn]
  split at hm
  · rename_i k hk'
    have hx := known_bits hn hk'
    subst hx
    exact ⟨ctxB_sound hS hc hc', fun _ _ => ⟨cstrB_sound hS hsp hs, readBinds_of_fits (memN_fits' hS hbp hm)⟩⟩
  · cases hm

theorem pre_gpuUpload {c b s n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, s, n] bits)
    (h : preB .gpuUpload [c, b, s, n] S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .gpuUpload bits w := by
  dsimp only [preB, preGpuUpload] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hm⟩ := h
  argbits hb [hc, hb', hs, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ _ _ _ => fits_readable (memBR_fits hS hs hn hm (hrb 2 3)) (Nat.le_refl _)⟩

theorem pre_gpuDownload {c b d n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, d, n] bits)
    (h : preB .gpuDownload [c, b, d, n] S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .gpuDownload bits w := by
  dsimp only [preB, preGpuDownload] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hm⟩ := h
  argbits hb [hc, hb', hd, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ _ _ _ _ _ => fits_writable (memBR_fits hS hd hn hm (hrb 2 3)) (Nat.le_refl _)⟩

theorem pre_windowInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .windowInit [s] S rb = true) : Contracts.Pre .windowInit bits w := by
  dsimp only [preB, preWindowInit] at h; simp only [Bool.and_eq_true] at h
  argbits hb [hs]
  exact ⟨partB_sound hS h.1, fits_slot8 (memN_fits' hS hs h.2)⟩

theorem pre_windowOpen {c ww hh tp tl bp bl : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, ww, hh, tp, tl, bp, bl] bits)
    (h : preB .windowOpen [c, ww, hh, tp, tl, bp, bl] S rb = true) : Contracts.Pre .windowOpen bits w := by
  dsimp only [preB, preWindowOpen] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hm⟩ := h
  argbits hb [hc, hw, hh', htp, htl, hbp, hbl]
  split at hm
  · rename_i a b ha hb'
    have e1 := known_bits htl ha
    have e2 := known_bits hbl hb'
    subst e1 e2
    simp only [Bool.and_eq_true] at hm
    exact ⟨ctxB_sound hS hc hc', fun _ =>
      ⟨fits_readable (memN_fits' hS htp hm.1) (Nat.le_refl _), fits_readable (memN_fits' hS hbp hm.2) (Nat.le_refl _)⟩⟩
  · cases hm

theorem pre_windowPoll {c ep me : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, ep, me] bits)
    (h : preB .windowPoll [c, ep, me] S rb = true) : Contracts.Pre .windowPoll bits w := by
  dsimp only [preB, preWindowPoll] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hm⟩ := h
  argbits hb [hc, hep, hme]
  split at hm
  · rename_i k hk'
    have e := known_bits hme hk'
    subst e
    refine ⟨ctxB_sound hS hc hc', fun _ => ?_⟩
    rw [(Contracts.pump_life w).1, winEventBytes_size]
    exact fits_writable (memN_fits' hS hep hm) (by simp only [List.length_take]; omega)
  · cases hm

theorem pre_windowPresentGpuBuffer {c g b : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, g, b] bits)
    (h : preB .windowPresentGpuBuffer [c, g, b] S rb = true) : Contracts.Pre .windowPresentGpuBuffer bits w := by
  dsimp only [preB, preWindowPresentGpuBuffer] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hg'⟩ := h
  argbits hb [hc, hg, hb']
  refine ⟨ctxB_sound hS hc hc', fun _ => ?_⟩
  have hgpu := (Contracts.pump_life w).2.2.2.2.1
  have := ctxB_sound hS hg hg'
  unfold Contracts.GpuOk; rw [hgpu]; exact this

theorem pre_windowCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .windowCleanup [s] S rb = true) : Contracts.Pre .windowCleanup bits w := by
  dsimp only [preB, preWindowCleanup] at h; simp only [Bool.and_eq_true] at h
  argbits hb [hs]
  have hf := memN_fits' hS hs h.1
  exact ⟨fits_load8 hf, slotOk_sound hS hs h.2, fits_slot8 hf⟩

theorem pre_lmdbCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .lmdbCleanup [s] S rb = true) : Contracts.Pre .lmdbCleanup bits w := by
  dsimp only [preB, preLmdbCleanup] at h; simp only [Bool.and_eq_true] at h
  argbits hb [hs]
  have hf := memN_fits' hS hs h.1
  exact ⟨fits_load8 hf, slotOk_sound hS hs h.2, fits_slot8 hf⟩

theorem pre_fileRead {b po dO fo n : AV} (hS : S.holds w) (hb : ArgBits Γ w [b, po, dO, fo, n] bits)
    (h : preB .fileRead [b, po, dO, fo, n] S rb = true) : Contracts.Pre .fileRead bits w := by
  dsimp only [preB, preFileRead] at h
  argbits hb [hb', hpo, hdo, hfo, hn]
  split at h
  · rename_i base p d k e1 e2 e3 e4
    have := known_bits hb' e1; subst this
    have := known_bits hpo e2; subst this
    have := known_bits hdo e3; subst this
    have := known_bits hn e4; subst this
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at h
    obtain ⟨⟨hp, hk⟩, hm⟩ := h
    refine ⟨pathB_sound (Γ := Γ) hS (by exact ⟨.i64, rfl⟩) hp, fun _ c _ _ => fits_writable (memN_fits' (Γ := Γ) hS (by exact ⟨.i64, rfl⟩) hm) ?_⟩
    unfold Contracts.readCount
    simp only [beq_iff_eq, hk, if_false]
    exact Nat.min_le_left _ _
  · cases h

theorem pre_fileWrite {b po so fo n : AV} (hS : S.holds w) (hb : ArgBits Γ w [b, po, so, fo, n] bits)
    (h : preB .fileWrite [b, po, so, fo, n] S rb = true) : Contracts.Pre .fileWrite bits w := by
  dsimp only [preB, preFileWrite] at h
  argbits hb [hb', hpo, hso, hfo, hn]
  split at h
  · rename_i base p e1 e2
    have := known_bits hb' e1; subst this
    have := known_bits hpo e2; subst this
    simp only [Bool.and_eq_true] at h
    obtain ⟨hp, hm⟩ := h
    refine ⟨pathB_sound (Γ := Γ) hS (by exact ⟨.i64, rfl⟩) hp, ?_⟩
    split at hm
    · rename_i s k e3 e4
      have := known_bits hso e3; subst this
      have := known_bits hn e4; subst this
      split
      · rename_i hk; rw [if_pos hk] at hm; exact cstrB_sound (Γ := Γ) hS (by exact ⟨.i64, rfl⟩) hm
      · rename_i hk; rw [if_neg hk] at hm
        exact fits_readable (memN_fits' (Γ := Γ) hS (by exact ⟨.i64, rfl⟩) hm) (Nat.le_refl _)
    · split at hm
      · rename_i sn nn e3 e4
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hm
        obtain ⟨hlo, hm⟩ := hm
        obtain ⟨ts, hds⟩ := hso
        obtain ⟨y, ey, hy⟩ := AV.num_desc hds e3
        obtain ⟨tn, hdn⟩ := hn
        obtain ⟨z, ez, hz⟩ := AV.num_desc hdn e4
        simp only [V.sc.injEq] at ey ez
        obtain ⟨-, rfl⟩ := ey
        obtain ⟨-, rfl⟩ := ez
        split
        · rename_i hk0; subst hk0; have := hz.1; simp at this; omega
        · exact fits_readable (memN_fits' (Γ := Γ) hS ⟨.i64, addrAt_desc hy⟩ hm) hz.2.1
      · cases hm
  · cases h

theorem pre_fileReadToPtr {pp d fo n : AV} (hS : S.holds w) (hb : ArgBits Γ w [pp, d, fo, n] bits)
    (h : preB .fileReadToPtr [pp, d, fo, n] S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .fileReadToPtr bits w := by
  dsimp only [preB, preFileReadToPtr] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hp, hm⟩ := h
  argbits hb [hpp, hd, hfo, hn]
  exact ⟨fun _ => pathB_sound hS hpp hp, fun _ _ _ _ =>
    fits_writable (memBR_fits hS hd hn hm (hrb 1 3)) (Nat.min_le_left _ _)⟩

theorem pre_nativeArch (hb : ArgBits Γ w [] bits) (_h : preB .nativeArch [] S = true) :
    Contracts.Pre .nativeArch bits w := by
  argbits hb []
  trivial

theorem pre_nativeLoad {s n : AV} (hS : S.holds w) (hb : ArgBits Γ w [s, n] bits)
    (h : preB .nativeLoad [s, n] S rb = true) : Contracts.Pre .nativeLoad bits w := by
  dsimp only [preB, preNativeLoad] at h
  argbits hb [hs, hn]
  split at h
  · rename_i k hk
    have := known_bits hn hk; subst this
    exact fun _ => fits_readable (memN_fits' hS hs h) (Nat.le_refl _)
  · cases h

theorem pre_cpuHas {n : AV} (hS : S.holds w) (hb : ArgBits Γ w [n] bits)
    (h : preB .cpuHas [n] S rb = true) : Contracts.Pre .cpuHas bits w := by
  dsimp only [preB, preCpuHas] at h
  argbits hb [hn]
  exact fun _ => cstrB_sound hS hn h

theorem pre_htInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits) (h : preB .htInit [s] S rb = true) :
    Contracts.Pre .htInit bits w := by
  dsimp only [preB, preHtInit] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h)

theorem pre_htCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .htCleanup [s] S rb = true) : Contracts.Pre .htCleanup bits w := by
  dsimp only [preB, preHtCleanup] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h)

theorem pre_htCreate {c : AV} (hb : ArgBits Γ w [c] bits) (_h : preB .htCreate [c] S = true) :
    Contracts.Pre .htCreate bits w := by
  argbits hb [hc]
  trivial

theorem pre_lmdbInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .lmdbInit [s] S rb = true) : Contracts.Pre .lmdbInit bits w := by
  dsimp only [preB, preLmdbInit] at h; simp only [Bool.and_eq_true] at h
  argbits hb [hs]
  exact ⟨partB_sound hS h.1, fits_slot8 (memN_fits' hS hs h.2)⟩

theorem pre_lmdbOpen {c pp x : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, pp, x] bits)
    (h : preB .lmdbOpen [c, pp, x] S rb = true) : Contracts.Pre .lmdbOpen bits w := by
  dsimp only [preB, preLmdbOpen] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hp⟩, he⟩ := h
  argbits hb [hc, hpp, hx]
  have hemp : w.lmdb.envs.isEmpty = true := partB_sound hS he
  refine ⟨ctxB_sound hS hc hc', fun _ => cstrB_sound hS hpp hp, fun _ _ => ?_⟩
  have he' : w.lmdb.envs = #[] := by simpa using hemp
  rw [he']; simp

theorem pre_lmdbBeginWriteTxn {c x : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, x] bits)
    (h : preB .lmdbBeginWriteTxn [c, x] S rb = true) : Contracts.Pre .lmdbBeginWriteTxn bits w := by
  dsimp only [preB, preLmdbBeginWriteTxn] at h
  argbits hb [hc, hx]
  exact ctxB_sound hS hc h

theorem pre_lmdbCommitWriteTxn {c x : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, x] bits)
    (h : preB .lmdbCommitWriteTxn [c, x] S rb = true) : Contracts.Pre .lmdbCommitWriteTxn bits w := by
  dsimp only [preB, preLmdbCommitWriteTxn] at h
  argbits hb [hc, hx]
  exact ctxB_sound hS hc h

theorem pre_lmdbPut {c x kp kl vp vl : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, x, kp, kl, vp, vl] bits)
    (h : preB .lmdbPut [c, x, kp, kl, vp, vl] S rb = true) : Contracts.Pre .lmdbPut bits w := by
  dsimp only [preB, preLmdbPut] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hm⟩ := h
  argbits hb [hc, hx, hkp, hkl, hvp, hvl]
  split at hm
  · rename_i a b ha hb'
    have e1 := known_bits hkl ha
    have e2 := known_bits hvl hb'
    subst e1 e2
    simp only [Bool.and_eq_true] at hm
    exact ⟨ctxB_sound hS hc hc', fun _ =>
      ⟨fits_readable (memN_fits' hS hkp hm.1) (Nat.le_refl _), fits_readable (memN_fits' hS hvp hm.2) (Nat.le_refl _)⟩⟩
  · cases hm

theorem pre_htInsert {c kp kl vp vl : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, kp, kl, vp, vl] bits)
    (h : preB .htInsert [c, kp, kl, vp, vl] S rb = true) : Contracts.Pre .htInsert bits w := by
  dsimp only [preB, preHtInsert] at h
  argbits hb [hc, hkp, hkl, hvp, hvl]
  split at h
  · rename_i a b ha hb'
    have e1 := known_bits hkl ha
    have e2 := known_bits hvl hb'
    subst e1 e2
    simp only [Bool.and_eq_true] at h
    exact ⟨fits_readable (memN_fits' hS hkp h.1) (Nat.le_refl _), fits_readable (memN_fits' hS hvp h.2) (Nat.le_refl _)⟩
  · cases h

theorem pre_htLookup {c kp kl rp : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, kp, kl, rp] bits)
    (h : preB .htLookup [c, kp, kl, rp] S rb = true) : Contracts.Pre .htLookup bits w := by
  dsimp only [preB, preHtLookup] at h
  argbits hb [hc, hkp, hkl, hrp]
  split at h
  · rename_i a n ha hn
    have e1 := known_bits hkl ha
    subst e1
    simp only [Bool.and_eq_true] at h
    have hv := hS _ (htValsOf_mem hn)
    exact ⟨fits_readable (memN_fits' hS hkp h.1) (Nat.le_refl _),
      fun _ _ _ hg => fits_writable (memN_fits' hS hrp h.2) (hv _ _ hg)⟩
  · cases h

theorem pre_lmdbCursorScan {c hd kp kl me rp cap : AV} (hS : S.holds w)
    (hb : ArgBits Γ w [c, hd, kp, kl, me, rp, cap] bits)
    (h : preB .lmdbCursorScan [c, hd, kp, kl, me, rp, cap] S rb = true) : Contracts.Pre .lmdbCursorScan bits w := by
  dsimp only [preB, preLmdbCursorScan] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hm⟩ := h
  argbits hb [hc, hhd, hkp, hkl, hme, hrp, hcap]
  split at hm
  · rename_i k n hk hn
    have e1 := known_bits hkl hk
    have e2 := known_bits hcap hn
    subst e1 e2
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hm
    exact ⟨ctxB_sound hS hc hc', fun _ j hj => fits_writable (memN_fits' hS hrp hm.2) hj,
      fun _ hpos => absurd hpos (by omega)⟩
  · cases hm

theorem pre_cublasSgemv {c tr m n al a x be y : AV} (hS : S.holds w)
    (hb : ArgBits Γ w [c, tr, m, n, al, a, x, be, y] bits)
    (h : preB .cublasSgemv [c, tr, m, n, al, a, x, be, y] S rb = true) : Contracts.Pre .cublasSgemv bits w := by
  dsimp only [preB, preCublasSgemv] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨hc', hcap⟩, ho⟩, hq⟩, hm⟩ := h
  argbits hb [hc, htr, hm', hn, hal, ha, hx, hbe, hy]
  split at hm
  · rename_i tr' m' n' a' x' y' e1 e2 e3 e4 e5 e6
    have := known_bits htr e1; subst this
    have := known_bits hm' e2; subst this
    have := known_bits hn e3; subst this
    have := known_bits ha e4; subst this
    have := known_bits hx e5; subst this
    have := known_bits hy e6; subst this
    split at hm
    · rename_i na nx ny f1 f2 f3
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hm
      obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := hm
      exact ⟨ctxB_sound hS hc hc', Contracts.capture_of_state hS hcap, Contracts.vendorKeeps_of_state hS ho,
        Contracts.sgemvOk_of_state hS hq (devBufOf_mem f1) (devBufOf_mem f2) (devBufOf_mem f3) h1 h2 h3 h4 h5⟩
    · cases hm
  · cases hm

theorem pre_cudaInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaInit as S rb = true) : Contracts.Pre .cudaInit bits w := by
  dsimp only [preB] at h
  unfold preCudaInit at h
  split at h
  · exact pre_cudaInit (rb := rb) hS hb h
  · cases h

theorem pre_cudaCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaCleanup as S rb = true) : Contracts.Pre .cudaCleanup bits w := by
  dsimp only [preB] at h
  unfold preCudaCleanup at h
  split at h
  · exact pre_cudaCleanup (rb := rb) hS hb h
  · cases h

theorem pre_cudaCreateBuffer' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaCreateBuffer as S rb = true) : Contracts.Pre .cudaCreateBuffer bits w := by
  dsimp only [preB] at h
  unfold preCudaCreateBuffer at h
  split at h
  · exact pre_cudaCreateBuffer (rb := rb) hS hb h
  · cases h

theorem pre_cudaSync' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaSync as S rb = true) : Contracts.Pre .cudaSync bits w := by
  dsimp only [preB] at h
  unfold preCudaSync at h
  split at h
  · exact pre_cudaSync (rb := rb) hS hb h
  · cases h

theorem pre_cudaUpload' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaUpload as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .cudaUpload bits w := by
  dsimp only [preB] at h
  unfold preCudaUpload at h
  split at h
  · exact pre_cudaUpload (rb := rb) hS hb h hrb
  · cases h

theorem pre_cudaDownload' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaDownload as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .cudaDownload bits w := by
  dsimp only [preB] at h
  unfold preCudaDownload at h
  split at h
  · exact pre_cudaDownload (rb := rb) hS hb h hrb
  · cases h

theorem pre_cudaUploadOffset' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaUploadOffset as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .cudaUploadOffset bits w := by
  dsimp only [preB] at h
  unfold preCudaUploadOffset at h
  split at h
  · exact pre_cudaUploadOffset (rb := rb) hS hb h hrb
  · cases h

theorem pre_cudaDownloadOffset' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaDownloadOffset as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .cudaDownloadOffset bits w := by
  dsimp only [preB] at h
  unfold preCudaDownloadOffset at h
  split at h
  · exact pre_cudaDownloadOffset (rb := rb) hS hb h hrb
  · cases h

theorem pre_cudaLaunch' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaLaunch as S rb = true) : Contracts.Pre .cudaLaunch bits w := by
  dsimp only [preB] at h
  unfold preCudaLaunch at h
  split at h
  · exact pre_cudaLaunch (rb := rb) hS hb h
  · cases h

theorem pre_cudaLaunchNamed' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaLaunchNamed as S rb = true) : Contracts.Pre .cudaLaunchNamed bits w := by
  dsimp only [preB] at h
  unfold preCudaLaunchNamed at h
  split at h
  · exact pre_cudaLaunchNamed (rb := rb) hS hb h
  · cases h

theorem pre_cudaLaunchOnStream' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaLaunchOnStream as S rb = true) : Contracts.Pre .cudaLaunchOnStream bits w := by
  dsimp only [preB] at h
  unfold preCudaLaunchOnStream at h
  split at h
  · exact pre_cudaLaunchOnStream (rb := rb) hS hb h
  · cases h

theorem pre_cudaStreamWaitEvent' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaStreamWaitEvent as S rb = true) : Contracts.Pre .cudaStreamWaitEvent bits w := by
  dsimp only [preB] at h
  unfold preCudaStreamWaitEvent at h
  split at h
  · exact pre_cudaStreamWaitEvent (rb := rb) hS hb h
  · cases h

theorem pre_gpuInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuInit as S rb = true) : Contracts.Pre .gpuInit bits w := by
  dsimp only [preB] at h
  unfold preGpuInit at h
  split at h
  · exact pre_gpuInit (rb := rb) hS hb h
  · cases h

theorem pre_gpuCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuCleanup as S rb = true) : Contracts.Pre .gpuCleanup bits w := by
  dsimp only [preB] at h
  unfold preGpuCleanup at h
  split at h
  · exact pre_gpuCleanup (rb := rb) hS hb h
  · cases h

theorem pre_gpuCreateBuffer' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuCreateBuffer as S rb = true) : Contracts.Pre .gpuCreateBuffer bits w := by
  dsimp only [preB] at h
  unfold preGpuCreateBuffer at h
  split at h
  · exact pre_gpuCreateBuffer (rb := rb) hS hb h
  · cases h

theorem pre_gpuDispatch' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuDispatch as S rb = true) : Contracts.Pre .gpuDispatch bits w := by
  dsimp only [preB] at h
  unfold preGpuDispatch at h
  split at h
  · exact pre_gpuDispatch (rb := rb) hS hb h
  · cases h

theorem pre_gpuCreatePipeline' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuCreatePipeline as S rb = true) : Contracts.Pre .gpuCreatePipeline bits w := by
  dsimp only [preB] at h
  unfold preGpuCreatePipeline at h
  split at h
  · exact pre_gpuCreatePipeline (rb := rb) hS hb h
  · cases h

theorem pre_gpuUpload' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuUpload as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .gpuUpload bits w := by
  dsimp only [preB] at h
  unfold preGpuUpload at h
  split at h
  · exact pre_gpuUpload (rb := rb) hS hb h hrb
  · cases h

theorem pre_gpuDownload' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuDownload as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .gpuDownload bits w := by
  dsimp only [preB] at h
  unfold preGpuDownload at h
  split at h
  · exact pre_gpuDownload (rb := rb) hS hb h hrb
  · cases h

theorem pre_windowInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .windowInit as S rb = true) : Contracts.Pre .windowInit bits w := by
  dsimp only [preB] at h
  unfold preWindowInit at h
  split at h
  · exact pre_windowInit (rb := rb) hS hb h
  · cases h

theorem pre_windowOpen' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .windowOpen as S rb = true) : Contracts.Pre .windowOpen bits w := by
  dsimp only [preB] at h
  unfold preWindowOpen at h
  split at h
  · exact pre_windowOpen (rb := rb) hS hb h
  · cases h

theorem pre_windowPoll' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .windowPoll as S rb = true) : Contracts.Pre .windowPoll bits w := by
  dsimp only [preB] at h
  unfold preWindowPoll at h
  split at h
  · exact pre_windowPoll (rb := rb) hS hb h
  · cases h

theorem pre_windowPresentGpuBuffer' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .windowPresentGpuBuffer as S rb = true) : Contracts.Pre .windowPresentGpuBuffer bits w := by
  dsimp only [preB] at h
  unfold preWindowPresentGpuBuffer at h
  split at h
  · exact pre_windowPresentGpuBuffer (rb := rb) hS hb h
  · cases h

theorem pre_windowCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .windowCleanup as S rb = true) : Contracts.Pre .windowCleanup bits w := by
  dsimp only [preB] at h
  unfold preWindowCleanup at h
  split at h
  · exact pre_windowCleanup (rb := rb) hS hb h
  · cases h

theorem pre_lmdbCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbCleanup as S rb = true) : Contracts.Pre .lmdbCleanup bits w := by
  dsimp only [preB] at h
  unfold preLmdbCleanup at h
  split at h
  · exact pre_lmdbCleanup (rb := rb) hS hb h
  · cases h

theorem pre_fileRead' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .fileRead as S rb = true) : Contracts.Pre .fileRead bits w := by
  dsimp only [preB] at h
  unfold preFileRead at h
  split at h
  · exact pre_fileRead (rb := rb) hS hb h
  · cases h

theorem pre_fileWrite' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .fileWrite as S rb = true) : Contracts.Pre .fileWrite bits w := by
  dsimp only [preB] at h
  unfold preFileWrite at h
  split at h
  · exact pre_fileWrite (rb := rb) hS hb h
  · cases h

theorem pre_fileReadToPtr' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .fileReadToPtr as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .fileReadToPtr bits w := by
  dsimp only [preB] at h
  unfold preFileReadToPtr at h
  split at h
  · exact pre_fileReadToPtr (rb := rb) hS hb h hrb
  · cases h

theorem pre_nativeLoad' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .nativeLoad as S rb = true) : Contracts.Pre .nativeLoad bits w := by
  dsimp only [preB] at h
  unfold preNativeLoad at h
  split at h
  · exact pre_nativeLoad (rb := rb) hS hb h
  · cases h

theorem pre_cpuHas' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cpuHas as S rb = true) : Contracts.Pre .cpuHas bits w := by
  dsimp only [preB] at h
  unfold preCpuHas at h
  split at h
  · exact pre_cpuHas (rb := rb) hS hb h
  · cases h

theorem pre_htInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .htInit as S rb = true) : Contracts.Pre .htInit bits w := by
  dsimp only [preB] at h
  unfold preHtInit at h
  split at h
  · exact pre_htInit (rb := rb) hS hb h
  · cases h

theorem pre_htCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .htCleanup as S rb = true) : Contracts.Pre .htCleanup bits w := by
  dsimp only [preB] at h
  unfold preHtCleanup at h
  split at h
  · exact pre_htCleanup (rb := rb) hS hb h
  · cases h

theorem pre_htCreate' {as : List AV} (hb : ArgBits Γ w as bits)
    (h : preB .htCreate as S rb = true) : Contracts.Pre .htCreate bits w := by
  dsimp only [preB] at h
  unfold preHtCreate at h
  split at h
  · exact pre_htCreate (S := S) hb h
  · cases h

theorem pre_lmdbInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbInit as S rb = true) : Contracts.Pre .lmdbInit bits w := by
  dsimp only [preB] at h
  unfold preLmdbInit at h
  split at h
  · exact pre_lmdbInit (rb := rb) hS hb h
  · cases h

theorem pre_lmdbOpen' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbOpen as S rb = true) : Contracts.Pre .lmdbOpen bits w := by
  dsimp only [preB] at h
  unfold preLmdbOpen at h
  split at h
  · exact pre_lmdbOpen (rb := rb) hS hb h
  · cases h

theorem pre_lmdbBeginWriteTxn' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbBeginWriteTxn as S rb = true) : Contracts.Pre .lmdbBeginWriteTxn bits w := by
  dsimp only [preB] at h
  unfold preLmdbBeginWriteTxn at h
  split at h
  · exact pre_lmdbBeginWriteTxn (rb := rb) hS hb h
  · cases h

theorem pre_lmdbCommitWriteTxn' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbCommitWriteTxn as S rb = true) : Contracts.Pre .lmdbCommitWriteTxn bits w := by
  dsimp only [preB] at h
  unfold preLmdbCommitWriteTxn at h
  split at h
  · exact pre_lmdbCommitWriteTxn (rb := rb) hS hb h
  · cases h

theorem pre_lmdbPut' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbPut as S rb = true) : Contracts.Pre .lmdbPut bits w := by
  dsimp only [preB] at h
  unfold preLmdbPut at h
  split at h
  · exact pre_lmdbPut (rb := rb) hS hb h
  · cases h

theorem pre_htInsert' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .htInsert as S rb = true) : Contracts.Pre .htInsert bits w := by
  dsimp only [preB] at h
  unfold preHtInsert at h
  split at h
  · exact pre_htInsert (rb := rb) hS hb h
  · cases h

theorem pre_htLookup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .htLookup as S rb = true) : Contracts.Pre .htLookup bits w := by
  dsimp only [preB] at h
  unfold preHtLookup at h
  split at h
  · exact pre_htLookup (rb := rb) hS hb h
  · cases h

theorem pre_lmdbCursorScan' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbCursorScan as S rb = true) : Contracts.Pre .lmdbCursorScan bits w := by
  dsimp only [preB] at h
  unfold preLmdbCursorScan at h
  split at h
  · exact pre_lmdbCursorScan (rb := rb) hS hb h
  · cases h

theorem pre_cublasSgemv' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cublasSgemv as S rb = true) : Contracts.Pre .cublasSgemv bits w := by
  dsimp only [preB] at h
  unfold preCublasSgemv at h
  split at h
  · exact pre_cublasSgemv (rb := rb) hS hb h
  · cases h

theorem pre_cublasSgemm' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cublasSgemm as S rb = true) : Contracts.Pre .cublasSgemm bits w := by
  dsimp only [preB] at h
  unfold preCublasSgemm at h
  split at h
  · rename_i x0 ta tb m n k x6 a sa b sb x11 cc sc batch oa ob oc la lb lc heq
    have hbs := allBits_eq hb heq
    subst hbs
    obtain ⟨hc, -⟩ := hb
    simp only [Bool.and_eq_true] at h
    obtain ⟨⟨⟨⟨hc', hcap⟩, ho⟩, hq⟩, hm⟩ := h
    split at hm
    · rename_i na nb nc f1 f2 f3
      exact ⟨ctxB_sound hS hc hc', Contracts.capture_of_state hS hcap, Contracts.vendorKeeps_of_state hS ho,
        Contracts.gemmOk_of_state hS hq (devBufOf_mem f1) (devBufOf_mem f2) (devBufOf_mem f3)
          (fun A B C hA hB hC => by
            unfold sgemmFits; rw [gemmStridedFits_eq, hA, hB, hC]; exact hm)⟩
    · cases hm
  · cases h

theorem pre_nativeArch' {as : List AV} (hb : ArgBits Γ w as bits)
    (h : preB .nativeArch as S rb = true) : Contracts.Pre .nativeArch bits w := by
  dsimp only [preB] at h
  unfold preNativeArch at h
  split at h
  · exact pre_nativeArch (S := S) hb h
  · cases h

end entries

/-- **What `preB` accepts, `Pre` holds of**: where the checker describes the
    arguments and the typestate, a call `preB` accepts has its precondition. -/
theorem handB_sound {a : AV} {x : UInt64} {k : Contracts.Held} (hd : ∃ t, a.Desc Γ w (.sc t x))
    (h : handB k a = true) : k.live w x = true := by
  unfold handB at h
  split at h
  · rename_i k'
    simp only [beq_iff_eq] at h
    subst h
    obtain ⟨t, x', e, hl⟩ := hd
    cases e
    rcases hl with ⟨h1, -⟩ | hl
    · cases h1
    · exact hl
  · cases h

/-- Room for `n` bytes the arguments name, and a handle the arguments name,
    as the library contracts ask them. -/
private theorem ext_room {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w) (hd : ∃ t, p.Desc Γ w (.sc t x))
    (h : memN S p k = true) : ∀ q ∈ [(x, k)], Fits w.mem q.1 q.2 := by
  intro q hq; simp only [List.mem_singleton] at hq; subst hq; exact memN_fits hS hd h

private theorem ext_hand {a : AV} {x : UInt64} {k : Contracts.Held} (hd : ∃ t, a.Desc Γ w (.sc t x))
    (h : handB k a = true) : ∀ p ∈ [(k, x)], p.1.live w p.2 = true := by
  intro q hq; simp only [List.mem_singleton] at hq; subst hq; exact handB_sound hd h

/-- **What `preExt` decides is what the library contracts ask**: the handles
    and the room `extNeeds` names for the bits handed over. -/
theorem preExt_sound (hS : S.holds w) {e : IR.Ext} {as : List AV} {bits : List UInt64} (hb : ArgBits Γ w as bits)
    (h : preExt S e as = true) :
    ∃ hs rs, Contracts.extNeeds e bits = some (hs, rs) ∧ (∀ p ∈ hs, p.1.live w p.2 = true) ∧
      (∀ q ∈ rs, Fits w.mem q.1 q.2) := by
  have none_ : ∀ p ∈ ([] : List (Contracts.Held × UInt64)), p.1.live w p.2 = true := by simp
  have noroom : ∀ q ∈ ([] : List (UInt64 × Nat)), Fits w.mem q.1 q.2 := by simp
  unfold preExt at h
  split at h <;> (try (cases h; done))
  -- C: memcpy
  · rename_i d s n
    split at h
    · rename_i d' s' n' e1 e2 e3
      simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨⟨hc, hs⟩, hd⟩ := h
      argbits hb [h1, h2, h3]
      have := known_bits h1 e1; subst this
      have := known_bits h2 e2; subst this
      have := known_bits h3 e3; subst this
      refine ⟨[], _, by simp only [Contracts.extNeeds]; rw [if_pos hc], none_, fun q hq => ?_⟩
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl
      · exact memN_fits hS h2 hs
      · exact memN_fits hS h1 hd
    · cases h
  -- C: memmove
  · rename_i d s n
    split at h
    · rename_i n' e3
      simp only [Bool.and_eq_true] at h
      argbits hb [h1, h2, h3]
      have := known_bits h3 e3; subst this
      refine ⟨[], _, rfl, none_, fun q hq => ?_⟩
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl
      · exact memN_fits hS h2 h.1
      · exact memN_fits hS h1 h.2
    · cases h
  -- C: memset
  · rename_i d c n
    split at h
    · rename_i n' e3
      argbits hb [h1, h2, h3]
      have := known_bits h3 e3; subst this
      exact ⟨[], _, rfl, none_, ext_room hS h1 h⟩
    · cases h
  -- C: calloc
  · rename_i n sz
    split at h
    · rename_i a b e1 e2
      simp only [decide_eq_true_eq] at h
      argbits hb [h1, h2]
      have := known_bits h1 e1; subst this
      have := known_bits h2 e2; subst this
      exact ⟨[], [], by simp [Contracts.extNeeds, h], none_, noroom⟩
    · cases h
  -- C: free
  · rename_i p
    argbits hb [h1]
    exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  -- window: init, pump
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  -- window: close, pixels
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  -- window: poll
  · simp only [Bool.and_eq_true] at h
    argbits hb [h1, h2]
    exact ⟨_, _, rfl, ext_hand h1 h.1, ext_room hS h2 h.2⟩
  -- cpu
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb [h1]; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb [h1]; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb [h1]; exact ⟨[], [], rfl, none_, noroom⟩
  -- serial: count
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  -- serial: name
  · split at h
    · rename_i c e3
      argbits hb [h1, h2, h3]
      have := known_bits h3 e3; subst this
      exact ⟨[], _, rfl, none_, ext_room hS h2 h⟩
    · cases h
  -- serial: close, pending
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  -- serial: read, write
  · simp only [Bool.and_eq_true] at h
    obtain ⟨hp, h⟩ := h
    split at h
    · rename_i l e3
      argbits hb [h1, h2, h3]
      have := known_bits h3 e3; subst this
      exact ⟨_, _, rfl, ext_hand h1 hp, ext_room hS h2 h⟩
    · cases h
  · simp only [Bool.and_eq_true] at h
    obtain ⟨hp, h⟩ := h
    split at h
    · rename_i l e3
      argbits hb [h1, h2, h3]
      have := known_bits h3 e3; subst this
      exact ⟨_, _, rfl, ext_hand h1 hp, ext_room hS h2 h⟩
    · cases h
  -- usb: count, info, open
  · argbits hb []; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb [h1, h2]; exact ⟨[], [], rfl, none_, noroom⟩
  · argbits hb [h1]; exact ⟨[], [], rfl, none_, noroom⟩
  -- usb: close, claim, release
  · argbits hb [h1]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  · argbits hb [h1, h2]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  · argbits hb [h1, h2]; exact ⟨_, [], rfl, ext_hand h1 h, noroom⟩
  -- usb: control
  · simp only [Bool.and_eq_true] at h
    obtain ⟨hp, h⟩ := h
    split at h
    · rename_i l e7
      argbits hb [h1, h2, h3, h4, h5, h6, h7, h8]
      have := known_bits h7 e7; subst this
      exact ⟨_, _, rfl, ext_hand h1 hp, ext_room hS h6 h⟩
    · cases h
  -- usb: bulk, interrupt
  all_goals
    simp only [Bool.and_eq_true] at h
    obtain ⟨hp, h⟩ := h
    split at h
    · rename_i l e5
      argbits hb [h1, h2, h3, h4, h5, h6]
      have := known_bits h5 e5; subst this
      exact ⟨_, _, rfl, ext_hand h1 hp, ext_room hS h4 h⟩
    · cases h

theorem pre_fileCreateDirAll' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .fileCreateDirAll as S rb = true) : Contracts.Pre .fileCreateDirAll bits w := by
  dsimp only [preB] at h
  unfold preFileCreateDirAll at h
  split at h
  · argbits hb [hp]
    exact pathB_sound hS hp h
  · cases h

theorem pre_fileWriteFromPtr' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .fileWriteFromPtr as S rb = true) (hrb : RbSound w bits rb) :
    Contracts.Pre .fileWriteFromPtr bits w := by
  dsimp only [preB] at h
  unfold preFileWriteFromPtr at h
  split at h
  · simp only [Bool.and_eq_true] at h
    argbits hb [hpp, hs, hfo, hn]
    exact fun _ => ⟨pathB_sound hS hpp h.1, fits_readable (memBR_fits hS hs hn h.2 (hrb 1 3)) (Nat.le_refl _)⟩
  · cases h

theorem pre_stdoutWrite' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .stdoutWrite as S rb = true) : Contracts.Pre .stdoutWrite bits w := by
  dsimp only [preB] at h
  unfold preStdoutWrite at h
  split at h
  · argbits hb [hb', hso, hn]
    split at h
    · rename_i base s k e1 e2 e3
      have := known_bits hb' e1; subst this
      have := known_bits hso e2; subst this
      have := known_bits hn e3; subst this
      exact fun _ => fits_readable (memN_fits' (Γ := Γ) hS (by exact ⟨.i64, rfl⟩) h) (Nat.le_refl _)
    · cases h
  · cases h

theorem pre_cudaFreeBuffer' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaFreeBuffer as S rb = true) : Contracts.Pre .cudaFreeBuffer bits w := by
  dsimp only [preB] at h
  unfold preCudaFreeBuffer at h
  split at h
  · simp only [Bool.and_eq_true] at h
    argbits hb [hc, hbuf]
    exact ⟨ctxB_sound hS hc h.1, fun _ => Contracts.ready_of_state hS h.2 _ _⟩
  · cases h

theorem pre_cublasPtrArray' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cublasPtrArray as S rb = true) : Contracts.Pre .cublasPtrArray bits w := by
  dsimp only [preB] at h
  unfold preCublasPtrArray at h
  split at h
  · simp only [Bool.and_eq_true] at h
    argbits hb [hc, h1, h2, h3, h4]
    exact ⟨ctxB_sound hS hc h.1, fun _ _ => Contracts.ready_of_state hS h.2 _ _⟩
  · cases h

theorem pre_cudaPinnedAlloc' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaPinnedAlloc as S rb = true) : Contracts.Pre .cudaPinnedAlloc bits w := by
  dsimp only [preB] at h
  unfold preCudaPinnedAlloc at h
  split at h
  · simp only [Bool.and_eq_true] at h
    obtain ⟨hc', hm⟩ := h
    argbits hb [hc, hn]
    split at hm
    · rename_i k hk
      have := known_bits hn hk; subst this
      obtain ⟨x, hx, hd⟩ := List.any_eq_true.mp hm
      refine ⟨ctxB_sound hS hc hc', ?_⟩
      cases x <;> simp only [Bool.false_eq_true, decide_eq_true_eq] at hd
      rename_i u
      have hu : w.mem.sizes .pinned ≤ u := hS _ hx
      unfold Contracts.PinnedRoom
      simp only [Mem.sizes, Mem.region] at hu
      omega
    · cases hm
  · cases h

theorem pre_gpuUploadPtr' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuUploadPtr as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .gpuUploadPtr bits w := by
  dsimp only [preB] at h
  unfold preGpuUploadPtr at h
  split at h
  · simp only [Bool.and_eq_true] at h
    argbits hb [hc, hb', hs, hn]
    exact ⟨ctxB_sound hS hc h.1, fun _ _ _ _ _ => fits_readable (memBR_fits hS hs hn h.2 (hrb 2 3)) (Nat.le_refl _)⟩
  · cases h

theorem pre_gpuDownloadPtr' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuDownloadPtr as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .gpuDownloadPtr bits w := by
  dsimp only [preB] at h
  unfold preGpuDownloadPtr at h
  split at h
  · simp only [Bool.and_eq_true] at h
    argbits hb [hc, hb', ho, hd, hn]
    exact ⟨ctxB_sound hS hc h.1, fun _ _ _ _ _ _ _ _ =>
      fits_writable (memBR_fits hS hd hn h.2 (hrb 3 4)) (Nat.le_refl _)⟩
  · cases h

theorem pre_threadInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .threadInit as S rb = true) : Contracts.Pre .threadInit bits w := by
  dsimp only [preB] at h
  unfold preThreadInit at h
  split at h
  · simp only [Bool.and_eq_true] at h
    argbits hb [hs]
    exact ⟨partB_sound hS h.1, fits_slot8 (memN_fits' hS hs h.2)⟩
  · cases h

theorem pre_threadJoin' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .threadJoin as S rb = true) : Contracts.Pre .threadJoin bits w := by
  dsimp only [preB] at h
  unfold preThreadJoin at h
  split at h
  · argbits hb [hc, hd]
    exact ctxB_sound hS hc h
  · cases h

theorem fits_forParty {m : Mem} {x : UInt64} {k : Nat} (p : Nat) (hf : Fits m x k) : Fits (m.forParty p) x k := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  exact ⟨r, off, hd, hp, by rw [Contracts.forParty_sizes]; exact hs, hz, hn⟩

theorem fits_readable_mem {m : Mem} {x : UInt64} {k j : Nat} (hf : Fits m x k) (hj : j ≤ k) : Readable m x j := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  exact Contracts.copyOut_isSome hd hp (by omega) hz (by omega)

theorem fits_writable_mem {m : Mem} {x : UInt64} {k j : Nat} (hf : Fits m x k) (hj : j ≤ k) : Writable m x j := by
  obtain ⟨r, off, hd, hp, hs, hz, hn⟩ := hf
  intro src hsrc
  exact Contracts.copyIn_isSome src hd hp (by omega) hz (by omega)

/-- A party that knows every access may run any operation. -/
theorem opReady_of_ready {p : Nat} (hS : S.holds w) (hq : readyIn S p = true) (rs ws : List Nat) :
    Contracts.OpReady w.dev.race p rs ws :=
  ⟨fun b _ => (Contracts.ready_of_state hS hq b true).1, fun b _ => (Contracts.ready_of_state hS hq b true).2 rfl⟩

theorem sgemvOk_of_ready {p : Nat} {trans m n a x y na nx ny : UInt64} (hS : S.holds w)
    (hq : readyIn S p = true)
    (ha : Fact.devBuf a na ∈ S) (hx : Fact.devBuf x nx ∈ S) (hy : Fact.devBuf y ny ∈ S)
    (hm : 0 < asI32 m) (hn : 0 < asI32 n)
    (hA : 4 * ((asI32 m).toNat * (asI32 n).toNat) ≤ na.toNat)
    (hX : 4 * (if asI32 trans != 0 then (asI32 m).toNat else (asI32 n).toNat) ≤ nx.toNat)
    (hY : 4 * (if asI32 trans != 0 then (asI32 n).toNat else (asI32 m).toNat) ≤ ny.toNat) :
    Contracts.SgemvOk w p trans m n a x y := by
  intro A X Y hA' hX' hY'
  refine ⟨hm, hn, ?_, opReady_of_ready hS hq _ _⟩
  simp only [sgemvFits, Bool.and_eq_true, decide_eq_true_eq]
  rw [Contracts.devBuf_size hS ha hA', Contracts.devBuf_size hS hx hX', Contracts.devBuf_size hS hy hY']
  exact ⟨⟨hA, hX⟩, hY⟩

theorem gemmOk_of_ready {p : Nat} {fits : ByteArray → ByteArray → ByteArray → Bool} {a b c na nb nc : UInt64}
    (hS : S.holds w) (hq : readyIn S p = true)
    (ha : Fact.devBuf a na ∈ S) (hb : Fact.devBuf b nb ∈ S) (hc : Fact.devBuf c nc ∈ S)
    (hf : ∀ A B C : ByteArray, A.size = na.toNat → B.size = nb.toNat → C.size = nc.toNat → fits A B C = true) :
    Contracts.GemmOk fits w p a b c := fun A B C hA hB hC =>
  ⟨hf A B C (Contracts.devBuf_size hS ha hA) (Contracts.devBuf_size hS hb hB) (Contracts.devBuf_size hS hc hC),
    opReady_of_ready hS hq _ _⟩

theorem pre_cudaLaunchNamedOnStream' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaLaunchNamedOnStream as S rb = true) : Contracts.Pre .cudaLaunchNamedOnStream bits w := by
  dsimp only [preB] at h
  unfold preCudaLaunchNamedOnStream at h
  split at h
  · simp only [Bool.and_eq_true] at h
    obtain ⟨⟨⟨⟨⟨⟨hc', ho⟩, hcap⟩, hk⟩, hn⟩, hq⟩, hm⟩ := h
    argbits hb [hc, hkp, hnp, hnb, hbp, h5, h6, h7, h8, h9, h10, hsid]
    split at hm
    · rename_i k hk'
      have hx := known_bits hnb hk'
      subst hx
      exact ⟨ctxB_sound hS hc hc', Contracts.keepsSize_of_state hS ho, Contracts.capture_of_state hS hcap,
        cstrB_sound hS hkp hk, cstrB_sound hS hnp hn,
        fun p hp => bindsReady_of_fits hS (streamReady_sound hS hsid hq p hp) (memN_fits' hS hbp hm)⟩
    · cases hm
  · cases h

theorem pre_cudaUploadAsync' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaUploadAsync as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre .cudaUploadAsync bits w := by
  dsimp only [preB] at h
  unfold preCudaUploadAsync at h
  split at h
  · simp only [Bool.and_eq_true] at h
    obtain ⟨⟨⟨hc', hcap⟩, hm⟩, hq⟩ := h
    argbits hb [hc, hbuf, hs, hn, hsid]
    exact ⟨ctxB_sound hS hc hc', Contracts.capture_of_state hS hcap, fun p _ hp _ _ =>
      ⟨fits_readable_mem (fits_forParty p (memBR_fits hS hs hn hm (hrb 2 3))) (Nat.le_refl _),
        Contracts.ready_of_state hS (streamReady_sound hS hsid hq p hp) _ _⟩⟩
  · cases h

theorem pre_cudaUploadOffsetAsync' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaUploadOffsetAsync as S rb = true) (hrb : RbSound w bits rb) :
    Contracts.Pre .cudaUploadOffsetAsync bits w := by
  dsimp only [preB] at h
  unfold preCudaUploadOffsetAsync at h
  split at h
  · simp only [Bool.and_eq_true] at h
    obtain ⟨⟨⟨hc', hcap⟩, hm⟩, hq⟩ := h
    argbits hb [hc, hbuf, ho, hs, hn, hsid]
    exact ⟨ctxB_sound hS hc hc', Contracts.capture_of_state hS hcap, fun p _ hp _ _ =>
      ⟨fits_readable_mem (fits_forParty p (memBR_fits hS hs hn hm (hrb 3 4))) (Nat.le_refl _),
        Contracts.ready_of_state hS (streamReady_sound hS hsid hq p hp) _ _⟩⟩
  · cases h

theorem pre_cudaDownloadAsync' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaDownloadAsync as S rb = true) (hrb : RbSound w bits rb) :
    Contracts.Pre .cudaDownloadAsync bits w := by
  dsimp only [preB] at h
  unfold preCudaDownloadAsync at h
  split at h
  · simp only [Bool.and_eq_true] at h
    obtain ⟨⟨⟨hc', hcap⟩, hm⟩, hq⟩ := h
    argbits hb [hc, hbuf, hd, hn, hsid]
    exact ⟨ctxB_sound hS hc hc', Contracts.capture_of_state hS hcap, fun p _ hp _ _ =>
      ⟨fits_writable_mem (fits_forParty p (memBR_fits hS hd hn hm (hrb 2 3))) (Nat.le_refl _),
        Contracts.ready_of_state hS (streamReady_sound hS hsid hq p hp) _ _⟩⟩
  · cases h

theorem pre_cublasSgemvOnStream' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cublasSgemvOnStream as S rb = true) : Contracts.Pre .cublasSgemvOnStream bits w := by
  dsimp only [preB] at h
  unfold preCublasSgemvOnStream at h
  split at h
  · simp only [Bool.and_eq_true] at h
    obtain ⟨⟨⟨⟨hc', hcap⟩, ho⟩, hq⟩, hm⟩ := h
    argbits hb [hc, htr, hm', hn, hal, ha, hx, hbe, hy, hsid]
    split at hm
    · rename_i tr' m' n' a' x' y' e1 e2 e3 e4 e5 e6
      have := known_bits htr e1; subst this
      have := known_bits hm' e2; subst this
      have := known_bits hn e3; subst this
      have := known_bits ha e4; subst this
      have := known_bits hx e5; subst this
      have := known_bits hy e6; subst this
      split at hm
      · rename_i na nx ny f1 f2 f3
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hm
        obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := hm
        exact ⟨ctxB_sound hS hc hc', Contracts.capture_of_state hS hcap, Contracts.vendorKeeps_of_state hS ho,
          fun p hp => sgemvOk_of_ready hS (streamReady_sound hS hsid hq p hp) (devBufOf_mem f1) (devBufOf_mem f2)
            (devBufOf_mem f3) h1 h2 h3 h4 h5⟩
      · cases hm
    · cases hm
  · cases h

theorem pre_cublasSgemmOnStream' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cublasSgemmOnStream as S rb = true) : Contracts.Pre .cublasSgemmOnStream bits w := by
  dsimp only [preB] at h
  unfold preCublasSgemmOnStream at h
  split at h
  · rename_i c x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 sid x16 x17 x18 x19 x20 x21
      x0 ta tb m n k y6 a sa b sb y11 cc sc batch y15 oa ob oc la lb lc heq
    have hbs := allBits_eq hb heq
    subst hbs
    obtain ⟨hc, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hsid, -⟩ := hb
    simp only [Bool.and_eq_true] at h
    obtain ⟨⟨⟨⟨hc', hcap⟩, ho⟩, hq⟩, hm⟩ := h
    split at hm
    · rename_i na nb nc f1 f2 f3
      exact ⟨ctxB_sound hS hc hc', Contracts.capture_of_state hS hcap, Contracts.vendorKeeps_of_state hS ho,
        fun p hp => gemmOk_of_ready hS (streamReady_sound hS hsid hq p hp) (devBufOf_mem f1) (devBufOf_mem f2)
          (devBufOf_mem f3) (fun A B C hA hB hC => by
            unfold sgemmFits; rw [gemmStridedFits_eq, hA, hB, hC]; exact hm)⟩
    · cases hm
  · cases h

theorem pre_cublasGemmStridedBatchedExBf16' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cublasGemmStridedBatchedExBf16 as S rb = true) :
    Contracts.Pre .cublasGemmStridedBatchedExBf16 bits w := by
  dsimp only [preB] at h
  unfold preCublasGemmStridedBatchedExBf16 at h
  split at h
  · rename_i heq
    have hbs := allBits_eq hb heq
    subst hbs
    obtain ⟨hc, -⟩ := hb
    simp only [Bool.and_eq_true] at h
    obtain ⟨⟨⟨⟨hc', hcap⟩, ho⟩, hq⟩, hm⟩ := h
    split at hm
    · rename_i na nb nc f1 f2 f3
      exact ⟨ctxB_sound hS hc hc', Contracts.capture_of_state hS hcap, Contracts.vendorKeeps_of_state hS ho,
        Contracts.gemmOk_of_state hS hq (devBufOf_mem f1) (devBufOf_mem f2) (devBufOf_mem f3)
          (fun A B C hA hB hC => by rw [gemmStridedFits_eq, hA, hB, hC]; exact hm)⟩
    · cases hm
  · cases h

theorem pre_cublasGemmExBf16' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cublasGemmExBf16 as S rb = true) : Contracts.Pre .cublasGemmExBf16 bits w := by
  dsimp only [preB] at h
  unfold preCublasGemmExBf16 at h
  split at h
  · rename_i heq
    have hbs := allBits_eq hb heq
    subst hbs
    obtain ⟨hc, -⟩ := hb
    simp only [Bool.and_eq_true] at h
    obtain ⟨⟨⟨⟨hc', hcap⟩, ho⟩, hq⟩, hm⟩ := h
    split at hm
    · rename_i na nb nc f1 f2 f3
      exact ⟨ctxB_sound hS hc hc', Contracts.capture_of_state hS hcap, Contracts.vendorKeeps_of_state hS ho,
        Contracts.gemmOk_of_state hS hq (devBufOf_mem f1) (devBufOf_mem f2) (devBufOf_mem f3)
          (fun A B C hA hB hC => by rw [gemmBf16Fits_eq, hA, hB, hC]; exact hm)⟩
    · cases hm
  · cases h

theorem pre_stdinReadline' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .stdinReadline as S rb = true) : Contracts.Pre .stdinReadline bits w := by
  dsimp only [preB] at h
  unfold preStdinReadline at h
  split at h
  · argbits hb [hb', hdo, hn]
    split at h
    · rename_i base d k e1 e2 e3
      have := known_bits hb' e1; subst this
      have := known_bits hdo e2; subst this
      have := (known_bits hn e3).symm; subst this
      simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at h
      obtain ⟨hk, hm⟩ := h
      intro line rest _ _ src hsrc
      have hk0 : 0 < k.toNat := Nat.pos_of_ne_zero fun e => hk (UInt64.toNat_inj.mp (by simpa using e))
      have hle : src.size + 1 ≤ k.toNat := by rw [hsrc]; omega
      obtain ⟨r, off, hd, hp, hs, hz, hn'⟩ := memN_fits' (Γ := Γ) hS (by exact ⟨.i64, rfl⟩) hm
      have hspan : regionSpan.toNat = 2 ^ 36 := by decide
      obtain ⟨m, hmm⟩ := Option.isSome_iff_exists.mp (Contracts.copyIn_isSome src hd hp (by omega) hz (by omega))
      refine ⟨m, hmm, Contracts.store_isSome (decodeAddr_add hd (by omega)) hp ?_ ?_ 0⟩
      · rw [Contracts.copyIn_sizes_eq hmm]; omega
      · rw [Contracts.copyIn_frozen hmm]; exact hz
    · cases h
  · cases h

theorem pre_threadCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .threadCleanup as S rb = true) : Contracts.Pre .threadCleanup bits w := by
  dsimp only [preB] at h
  unfold preThreadCleanup at h
  split at h
  · simp only [Bool.and_eq_true] at h
    argbits hb [hs]
    have hf := memN_fits' hS hs h.1
    have hz : w.mem.frozen = false := by obtain ⟨_, _, _, _, _, hz, _⟩ := hf; exact hz
    have e : ({ w.mem with frozen := false } : Mem) = w.mem := by rw [← hz]
    show Contracts.ThreadSlotOk w _
    unfold Contracts.ThreadSlotOk
    rw [e]
    exact ⟨slotOk_sound hS hs h.2, fits_load8 hf, fits_slot8 hf⟩
  · cases h

theorem ArgBits.length_eq {Γ : Env} {w : World} :
    ∀ {as : List AV} {bits : List UInt64}, ArgBits Γ w as bits → bits.length = as.length
  | [], [], _ => rfl
  | _ :: _, _ :: _, h => by simp [ArgBits.length_eq h.2]
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

theorem preArity_len {n : Nat} {as : List AV} (h : preArity n as = true) : as.length = n := by
  simpa [preArity] using h

theorem preCtxN_sound (hS : S.holds w) {as : List AV} (hb : ArgBits Γ w as bits) {n : Nat} {cap : Bool}
    (h : preCtxN S n cap as = true) :
    ∃ x rest, bits = x :: rest ∧ rest.length + 1 = n ∧ (x = 0 ∨ (x = cudaCtx ∧ Part.cuda.get w = true)) ∧
      (cap = true → w.dev.capture = none) := by
  unfold preCtxN at h
  split at h
  · rename_i c rest
    cases bits with
    | nil => exact hb.elim
    | cons x xs =>
        obtain ⟨hc, hr⟩ := hb
        simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, Bool.not_eq_true'] at h
        obtain ⟨⟨hl, hcb⟩, hcap⟩ := h
        refine ⟨x, xs, rfl, by rw [ArgBits.length_eq hr]; exact hl, ctxB_sound hS hc hcb, fun hc' => ?_⟩
        rcases hcap with e | e
        · rw [hc'] at e; cases e
        · exact Contracts.capture_of_state hS e
  · cases h

theorem preB_sound {f : Ffi} {as : List AV} {bits : List UInt64} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB f as S rb = true) (hrb : RbSound w bits rb) : Contracts.Pre f bits w := by
  cases f
  case cudaInit => exact pre_cudaInit' hS hb h
  case cudaCleanup => exact pre_cudaCleanup' hS hb h
  case cudaCreateBuffer => exact pre_cudaCreateBuffer' hS hb h
  case cudaSync => exact pre_cudaSync' hS hb h
  case cudaUpload => exact pre_cudaUpload' hS hb h hrb
  case cudaDownload => exact pre_cudaDownload' hS hb h hrb
  case cudaUploadOffset => exact pre_cudaUploadOffset' hS hb h hrb
  case cudaDownloadOffset => exact pre_cudaDownloadOffset' hS hb h hrb
  case cudaLaunch => exact pre_cudaLaunch' hS hb h
  case cudaLaunchNamed => exact pre_cudaLaunchNamed' hS hb h
  case cudaLaunchOnStream => exact pre_cudaLaunchOnStream' hS hb h
  case cudaStreamWaitEvent => exact pre_cudaStreamWaitEvent' hS hb h
  case gpuInit => exact pre_gpuInit' hS hb h
  case gpuCleanup => exact pre_gpuCleanup' hS hb h
  case gpuCreateBuffer => exact pre_gpuCreateBuffer' hS hb h
  case gpuDispatch => exact pre_gpuDispatch' hS hb h
  case gpuCreatePipeline => exact pre_gpuCreatePipeline' hS hb h
  case gpuUpload => exact pre_gpuUpload' hS hb h hrb
  case gpuDownload => exact pre_gpuDownload' hS hb h hrb
  case windowInit => exact pre_windowInit' hS hb h
  case windowOpen => exact pre_windowOpen' hS hb h
  case windowPoll => exact pre_windowPoll' hS hb h
  case windowPresentGpuBuffer => exact pre_windowPresentGpuBuffer' hS hb h
  case windowCleanup => exact pre_windowCleanup' hS hb h
  case lmdbCleanup => exact pre_lmdbCleanup' hS hb h
  case fileRead => exact pre_fileRead' hS hb h
  case fileWrite => exact pre_fileWrite' hS hb h
  case fileReadToPtr => exact pre_fileReadToPtr' hS hb h hrb
  case nativeLoad => exact pre_nativeLoad' hS hb h
  case cpuHas => exact pre_cpuHas' hS hb h
  case htInit => exact pre_htInit' hS hb h
  case htCleanup => exact pre_htCleanup' hS hb h
  case htCreate => exact pre_htCreate' hb h
  case nativeArch => exact pre_nativeArch' hb h
  case lmdbInit => exact pre_lmdbInit' hS hb h
  case lmdbOpen => exact pre_lmdbOpen' hS hb h
  case lmdbBeginWriteTxn => exact pre_lmdbBeginWriteTxn' hS hb h
  case lmdbCommitWriteTxn => exact pre_lmdbCommitWriteTxn' hS hb h
  case lmdbPut => exact pre_lmdbPut' hS hb h
  case cublasSgemv => exact pre_cublasSgemv' hS hb h
  case cublasSgemm => exact pre_cublasSgemm' hS hb h
  case htInsert => exact pre_htInsert' hS hb h
  case htLookup => exact pre_htLookup' hS hb h
  case lmdbCursorScan => exact pre_lmdbCursorScan' hS hb h
  case fileCreateDirAll => exact pre_fileCreateDirAll' hS hb h
  case fileWriteFromPtr => exact pre_fileWriteFromPtr' hS hb h hrb
  case stdoutWrite => exact pre_stdoutWrite' hS hb h
  case cudaFreeBuffer => exact pre_cudaFreeBuffer' hS hb h
  case cublasPtrArray => exact pre_cublasPtrArray' hS hb h
  case cudaPinnedAlloc => exact pre_cudaPinnedAlloc' hS hb h
  case gpuUploadPtr => exact pre_gpuUploadPtr' hS hb h hrb
  case gpuDownloadPtr => exact pre_gpuDownloadPtr' hS hb h hrb
  case threadInit => exact pre_threadInit' hS hb h
  case threadJoin => exact pre_threadJoin' hS hb h
  case stdinReadline => exact pre_stdinReadline' hS hb h
  case threadCleanup => exact pre_threadCleanup' hS hb h
  case cudaLaunchNamedOnStream => exact pre_cudaLaunchNamedOnStream' hS hb h
  case cudaUploadAsync => exact pre_cudaUploadAsync' hS hb h hrb
  case cudaUploadOffsetAsync => exact pre_cudaUploadOffsetAsync' hS hb h hrb
  case cudaDownloadAsync => exact pre_cudaDownloadAsync' hS hb h hrb
  case cublasSgemvOnStream => exact pre_cublasSgemvOnStream' hS hb h
  case cublasSgemmOnStream => exact pre_cublasSgemmOnStream' hS hb h
  case cublasGemmStridedBatchedExBf16 => exact pre_cublasGemmStridedBatchedExBf16' hS hb h
  case cublasGemmExBf16 => exact pre_cublasGemmExBf16' hS hb h
  case cosf =>
    have hl := (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
    match bits, hl with | [_], _ => trivial
  case sinf =>
    have hl := (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
    match bits, hl with | [_], _ => trivial
  case nativeFree =>
    have hl := (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
    match bits, hl with | [_], _ => trivial
  case htCount =>
    have hl := (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
    match bits, hl with | [_], _ => trivial
  case threadPriority =>
    exact (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
  case powf =>
    have hl := (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
    match bits, hl with | [_, _], _ => trivial
  case memAdviseHuge =>
    exact (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
  case memLock =>
    exact (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
  case memUnlock =>
    exact (ArgBits.length_eq hb).trans (preArity_len (by dsimp only [preB] at h; exact h))
  case cudaEventCreate =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [], _ => exact hc
  case cudaMemInfoFree =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [], _ => exact hc
  case cudaMemInfoTotal =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [], _ => exact hc
  case cudaStreamCreate =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [], _ => exact hc
  case cudaEventDestroy =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [_], _ => exact hc
  case cudaGraphDestroy =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [_], _ => exact hc
  case cudaPinnedFree =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [_], _ => exact hc
  case cudaPinnedPtr =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [_], _ => exact hc
  case cudaGraphUpload =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [_, _], _ => exact hc
  case cudaPinnedPtrAt =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [_, _, _], _ => exact hc
  case cudaGraphBeginCapture =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [_], _ => exact ⟨hc, hcap rfl⟩
  case cudaGraphEndCapture =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with
    | [_], _ => exact ⟨hc, fun p c _ hcc => by rw [hcap rfl] at hcc; cases hcc⟩
  case cudaStreamDestroy =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [_], _ => exact ⟨hc, hcap rfl⟩
  case cudaStreamSync =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [_], _ => exact ⟨hc, hcap rfl⟩
  case cudaEventRecord =>
    obtain ⟨x, rest, rfl, hl, hc, hcap⟩ := preCtxN_sound hS hb (by dsimp only [preB] at h; exact h)
    match rest, hl with | [_, _], _ => exact ⟨hc, hcap rfl⟩
  all_goals (dsimp only [preB] at h; cases h)

end AlgorithmLib.Check
