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

open AlgorithmLib.HProg.Contracts (Fact TState holds_of_mem Part)

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

variable {Γ : Env} {w : World} {S : TState}

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
    have hd' := Static.decodeAddr_add (i := 4 * i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
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
  have hd1 := Static.decodeAddr_add (i := 8 * i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  have hd2 := Static.decodeAddr_add (i := 8 * i + 4) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp (Contracts.load_isSome (n := 4) hd1 hp (by omega) hz)
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp (Contracts.load_isSome (n := 4) hd2 hp (by omega) hz)
  rw [ha, hb]; rfl

theorem bindsReady_of_fits {nBufs bindPtr : UInt64} (hS : S.holds w) (hq : S.contains .devSeq = true)
    (hf : Fits w.mem bindPtr (4 * (asI32 nBufs).toNat)) : BindsReady w defaultParty nBufs bindPtr := by
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

theorem pre_cudaInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits) (h : preB .cudaInit [s] S = true) :
    Contracts.Pre .cudaInit bits w := by
  dsimp only [preB, preCudaInit] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h) _

theorem pre_cudaCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .cudaCleanup [s] S = true) : Contracts.Pre .cudaCleanup bits w := by
  dsimp only [preB, preCudaCleanup] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h) _

theorem pre_cudaCreateBuffer {c n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, n] bits)
    (h : preB .cudaCreateBuffer [c, n] S = true) : Contracts.Pre .cudaCreateBuffer bits w := by
  dsimp only [preB, preCudaCreateBuffer] at h
  argbits hb [hc, hn]
  exact ctxB_sound hS hc h

theorem pre_cudaSync {c : AV} (hS : S.holds w) (hb : ArgBits Γ w [c] bits)
    (h : preB .cudaSync [c] S = true) : Contracts.Pre .cudaSync bits w := by
  dsimp only [preB, preCudaSync] at h
  argbits hb [hc]
  exact ctxB_sound hS hc h

theorem pre_cudaUpload {c b d n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, d, n] bits)
    (h : preB .cudaUpload [c, b, d, n] S = true) : Contracts.Pre .cudaUpload bits w := by
  dsimp only [preB, preCudaUpload] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hm⟩, hq⟩ := h
  argbits hb [hc, hb', hd, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ _ =>
    ⟨fits_readable (memB_fits hS hd hn hm) (Nat.le_refl _), Contracts.ready_of_state hS hq _ _⟩⟩

theorem pre_cudaDownload {c b d n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, d, n] bits)
    (h : preB .cudaDownload [c, b, d, n] S = true) : Contracts.Pre .cudaDownload bits w := by
  dsimp only [preB, preCudaDownload] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hm⟩, hq⟩ := h
  argbits hb [hc, hb', hd, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ _ =>
    ⟨fits_writable (memB_fits hS hd hn hm) (Nat.le_refl _), Contracts.ready_of_state hS hq _ _⟩⟩

theorem pre_cudaUploadOffset {c b o d n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, o, d, n] bits)
    (h : preB .cudaUploadOffset [c, b, o, d, n] S = true) : Contracts.Pre .cudaUploadOffset bits w := by
  dsimp only [preB, preCudaUploadOffset] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hm⟩, hq⟩ := h
  argbits hb [hc, hb', ho, hd, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ =>
    ⟨fits_readable (memB_fits hS hd hn hm) (Nat.le_refl _), Contracts.ready_of_state hS hq _ _⟩⟩

theorem pre_cudaDownloadOffset {c b o d n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, o, d, n] bits)
    (h : preB .cudaDownloadOffset [c, b, o, d, n] S = true) : Contracts.Pre .cudaDownloadOffset bits w := by
  dsimp only [preB, preCudaDownloadOffset] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hm⟩, hq⟩ := h
  argbits hb [hc, hb', ho, hd, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ =>
    ⟨fits_writable (memB_fits hS hd hn hm) (Nat.le_refl _), Contracts.ready_of_state hS hq _ _⟩⟩

theorem pre_cudaLaunch {c kp nb bp gx gy gz bx by_ bz : AV} (hS : S.holds w)
    (hb : ArgBits Γ w [c, kp, nb, bp, gx, gy, gz, bx, by_, bz] bits)
    (h : preB .cudaLaunch [c, kp, nb, bp, gx, gy, gz, bx, by_, bz] S = true) :
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
    (h : preB .cudaLaunchNamed [c, kp, np, nb, bp, gx, gy, gz, bx, by_, bz] S = true) :
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

theorem pre_gpuInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits) (h : preB .gpuInit [s] S = true) :
    Contracts.Pre .gpuInit bits w := by
  dsimp only [preB, preGpuInit] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h)

theorem pre_gpuCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .gpuCleanup [s] S = true) : Contracts.Pre .gpuCleanup bits w := by
  dsimp only [preB, preGpuCleanup] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h)

theorem pre_gpuCreateBuffer {c n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, n] bits)
    (h : preB .gpuCreateBuffer [c, n] S = true) : Contracts.Pre .gpuCreateBuffer bits w := by
  dsimp only [preB, preGpuCreateBuffer] at h
  argbits hb [hc, hn]
  exact ctxB_sound hS hc h

theorem pre_gpuDispatch {c p x y z : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, p, x, y, z] bits)
    (h : preB .gpuDispatch [c, p, x, y, z] S = true) : Contracts.Pre .gpuDispatch bits w := by
  dsimp only [preB, preGpuDispatch] at h
  argbits hb [hc, hp, hx, hy, hz]
  exact ctxB_sound hS hc h

theorem pre_gpuCreatePipeline {c sp bp n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, sp, bp, n] bits)
    (h : preB .gpuCreatePipeline [c, sp, bp, n] S = true) : Contracts.Pre .gpuCreatePipeline bits w := by
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
    (h : preB .gpuUpload [c, b, s, n] S = true) : Contracts.Pre .gpuUpload bits w := by
  dsimp only [preB, preGpuUpload] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hm⟩ := h
  argbits hb [hc, hb', hs, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ _ _ _ => fits_readable (memB_fits hS hs hn hm) (Nat.le_refl _)⟩

theorem pre_gpuDownload {c b d n : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, b, d, n] bits)
    (h : preB .gpuDownload [c, b, d, n] S = true) : Contracts.Pre .gpuDownload bits w := by
  dsimp only [preB, preGpuDownload] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hm⟩ := h
  argbits hb [hc, hb', hd, hn]
  exact ⟨ctxB_sound hS hc hc', fun _ _ _ _ _ _ _ => fits_writable (memB_fits hS hd hn hm) (Nat.le_refl _)⟩

theorem pre_windowInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .windowInit [s] S = true) : Contracts.Pre .windowInit bits w := by
  dsimp only [preB, preWindowInit] at h; simp only [Bool.and_eq_true] at h
  argbits hb [hs]
  exact ⟨partB_sound hS h.1, fits_slot8 (memN_fits' hS hs h.2)⟩

theorem pre_windowOpen {c ww hh tp tl bp bl : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, ww, hh, tp, tl, bp, bl] bits)
    (h : preB .windowOpen [c, ww, hh, tp, tl, bp, bl] S = true) : Contracts.Pre .windowOpen bits w := by
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
    (h : preB .windowPoll [c, ep, me] S = true) : Contracts.Pre .windowPoll bits w := by
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
    (h : preB .windowPresentGpuBuffer [c, g, b] S = true) : Contracts.Pre .windowPresentGpuBuffer bits w := by
  dsimp only [preB, preWindowPresentGpuBuffer] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hc', hg'⟩ := h
  argbits hb [hc, hg, hb']
  refine ⟨ctxB_sound hS hc hc', fun _ => ?_⟩
  have hgpu := (Contracts.pump_life w).2.2.2.2.1
  have := ctxB_sound hS hg hg'
  unfold Contracts.GpuOk; rw [hgpu]; exact this

theorem pre_windowCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .windowCleanup [s] S = true) : Contracts.Pre .windowCleanup bits w := by
  dsimp only [preB, preWindowCleanup] at h; simp only [Bool.and_eq_true] at h
  argbits hb [hs]
  have hf := memN_fits' hS hs h.1
  exact ⟨fits_load8 hf, slotOk_sound hS hs h.2, fits_slot8 hf⟩

theorem pre_lmdbCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .lmdbCleanup [s] S = true) : Contracts.Pre .lmdbCleanup bits w := by
  dsimp only [preB, preLmdbCleanup] at h; simp only [Bool.and_eq_true] at h
  argbits hb [hs]
  have hf := memN_fits' hS hs h.1
  exact ⟨fits_load8 hf, slotOk_sound hS hs h.2, fits_slot8 hf⟩

theorem pre_fileRead {b po dO fo n : AV} (hS : S.holds w) (hb : ArgBits Γ w [b, po, dO, fo, n] bits)
    (h : preB .fileRead [b, po, dO, fo, n] S = true) : Contracts.Pre .fileRead bits w := by
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
    (h : preB .fileWrite [b, po, so, fo, n] S = true) : Contracts.Pre .fileWrite bits w := by
  dsimp only [preB, preFileWrite] at h
  argbits hb [hb', hpo, hso, hfo, hn]
  split at h
  · rename_i base p s k e1 e2 e3 e4
    have := known_bits hb' e1; subst this
    have := known_bits hpo e2; subst this
    have := known_bits hso e3; subst this
    have := known_bits hn e4; subst this
    simp only [Bool.and_eq_true] at h
    obtain ⟨hp, hm⟩ := h
    refine ⟨pathB_sound (Γ := Γ) hS (by exact ⟨.i64, rfl⟩) hp, ?_⟩
    split
    · rename_i hk; rw [if_pos hk] at hm; exact cstrB_sound (Γ := Γ) hS (by exact ⟨.i64, rfl⟩) hm
    · rename_i hk; rw [if_neg hk] at hm; exact fits_readable (memN_fits' (Γ := Γ) hS (by exact ⟨.i64, rfl⟩) hm) (Nat.le_refl _)
  · cases h

theorem pre_fileReadToPtr {pp d fo n : AV} (hS : S.holds w) (hb : ArgBits Γ w [pp, d, fo, n] bits)
    (h : preB .fileReadToPtr [pp, d, fo, n] S = true) : Contracts.Pre .fileReadToPtr bits w := by
  dsimp only [preB, preFileReadToPtr] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨hp, hm⟩ := h
  argbits hb [hpp, hd, hfo, hn]
  exact ⟨fun _ => pathB_sound hS hpp hp, fun _ _ _ _ =>
    fits_writable (memB_fits hS hd hn hm) (Nat.min_le_left _ _)⟩

theorem pre_nativeArch (hb : ArgBits Γ w [] bits) (_h : preB .nativeArch [] S = true) :
    Contracts.Pre .nativeArch bits w := by
  argbits hb []
  trivial

theorem pre_nativeLoad {s n : AV} (hS : S.holds w) (hb : ArgBits Γ w [s, n] bits)
    (h : preB .nativeLoad [s, n] S = true) : Contracts.Pre .nativeLoad bits w := by
  dsimp only [preB, preNativeLoad] at h
  argbits hb [hs, hn]
  split at h
  · rename_i k hk
    have := known_bits hn hk; subst this
    exact fun _ => fits_readable (memN_fits' hS hs h) (Nat.le_refl _)
  · cases h

theorem pre_cpuHas {n : AV} (hS : S.holds w) (hb : ArgBits Γ w [n] bits)
    (h : preB .cpuHas [n] S = true) : Contracts.Pre .cpuHas bits w := by
  dsimp only [preB, preCpuHas] at h
  argbits hb [hn]
  exact fun _ => cstrB_sound hS hn h

theorem pre_htInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits) (h : preB .htInit [s] S = true) :
    Contracts.Pre .htInit bits w := by
  dsimp only [preB, preHtInit] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h)

theorem pre_htCleanup {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .htCleanup [s] S = true) : Contracts.Pre .htCleanup bits w := by
  dsimp only [preB, preHtCleanup] at h
  argbits hb [hs]
  exact fits_slot8 (memN_fits' hS hs h)

theorem pre_htCreate {c : AV} (hb : ArgBits Γ w [c] bits) (_h : preB .htCreate [c] S = true) :
    Contracts.Pre .htCreate bits w := by
  argbits hb [hc]
  trivial

theorem pre_lmdbInit {s : AV} (hS : S.holds w) (hb : ArgBits Γ w [s] bits)
    (h : preB .lmdbInit [s] S = true) : Contracts.Pre .lmdbInit bits w := by
  dsimp only [preB, preLmdbInit] at h; simp only [Bool.and_eq_true] at h
  argbits hb [hs]
  exact ⟨partB_sound hS h.1, fits_slot8 (memN_fits' hS hs h.2)⟩

theorem pre_lmdbOpen {c pp x : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, pp, x] bits)
    (h : preB .lmdbOpen [c, pp, x] S = true) : Contracts.Pre .lmdbOpen bits w := by
  dsimp only [preB, preLmdbOpen] at h; simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hc', hp⟩, he⟩ := h
  argbits hb [hc, hpp, hx]
  have hemp : w.lmdb.envs.isEmpty = true := partB_sound hS he
  refine ⟨ctxB_sound hS hc hc', fun _ => cstrB_sound hS hpp hp, fun _ _ => ?_⟩
  have he' : w.lmdb.envs = #[] := by simpa using hemp
  rw [he']; simp

theorem pre_lmdbBeginWriteTxn {c x : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, x] bits)
    (h : preB .lmdbBeginWriteTxn [c, x] S = true) : Contracts.Pre .lmdbBeginWriteTxn bits w := by
  dsimp only [preB, preLmdbBeginWriteTxn] at h
  argbits hb [hc, hx]
  exact ctxB_sound hS hc h

theorem pre_lmdbCommitWriteTxn {c x : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, x] bits)
    (h : preB .lmdbCommitWriteTxn [c, x] S = true) : Contracts.Pre .lmdbCommitWriteTxn bits w := by
  dsimp only [preB, preLmdbCommitWriteTxn] at h
  argbits hb [hc, hx]
  exact ctxB_sound hS hc h

theorem pre_lmdbPut {c x kp kl vp vl : AV} (hS : S.holds w) (hb : ArgBits Γ w [c, x, kp, kl, vp, vl] bits)
    (h : preB .lmdbPut [c, x, kp, kl, vp, vl] S = true) : Contracts.Pre .lmdbPut bits w := by
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
    (h : preB .htInsert [c, kp, kl, vp, vl] S = true) : Contracts.Pre .htInsert bits w := by
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
    (h : preB .htLookup [c, kp, kl, rp] S = true) : Contracts.Pre .htLookup bits w := by
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
    (h : preB .lmdbCursorScan [c, hd, kp, kl, me, rp, cap] S = true) : Contracts.Pre .lmdbCursorScan bits w := by
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
    (h : preB .cublasSgemv [c, tr, m, n, al, a, x, be, y] S = true) : Contracts.Pre .cublasSgemv bits w := by
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
    (h : preB .cudaInit as S = true) : Contracts.Pre .cudaInit bits w := by
  dsimp only [preB] at h
  unfold preCudaInit at h
  split at h
  · exact pre_cudaInit hS hb h
  · cases h

theorem pre_cudaCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaCleanup as S = true) : Contracts.Pre .cudaCleanup bits w := by
  dsimp only [preB] at h
  unfold preCudaCleanup at h
  split at h
  · exact pre_cudaCleanup hS hb h
  · cases h

theorem pre_cudaCreateBuffer' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaCreateBuffer as S = true) : Contracts.Pre .cudaCreateBuffer bits w := by
  dsimp only [preB] at h
  unfold preCudaCreateBuffer at h
  split at h
  · exact pre_cudaCreateBuffer hS hb h
  · cases h

theorem pre_cudaSync' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaSync as S = true) : Contracts.Pre .cudaSync bits w := by
  dsimp only [preB] at h
  unfold preCudaSync at h
  split at h
  · exact pre_cudaSync hS hb h
  · cases h

theorem pre_cudaUpload' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaUpload as S = true) : Contracts.Pre .cudaUpload bits w := by
  dsimp only [preB] at h
  unfold preCudaUpload at h
  split at h
  · exact pre_cudaUpload hS hb h
  · cases h

theorem pre_cudaDownload' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaDownload as S = true) : Contracts.Pre .cudaDownload bits w := by
  dsimp only [preB] at h
  unfold preCudaDownload at h
  split at h
  · exact pre_cudaDownload hS hb h
  · cases h

theorem pre_cudaUploadOffset' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaUploadOffset as S = true) : Contracts.Pre .cudaUploadOffset bits w := by
  dsimp only [preB] at h
  unfold preCudaUploadOffset at h
  split at h
  · exact pre_cudaUploadOffset hS hb h
  · cases h

theorem pre_cudaDownloadOffset' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaDownloadOffset as S = true) : Contracts.Pre .cudaDownloadOffset bits w := by
  dsimp only [preB] at h
  unfold preCudaDownloadOffset at h
  split at h
  · exact pre_cudaDownloadOffset hS hb h
  · cases h

theorem pre_cudaLaunch' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaLaunch as S = true) : Contracts.Pre .cudaLaunch bits w := by
  dsimp only [preB] at h
  unfold preCudaLaunch at h
  split at h
  · exact pre_cudaLaunch hS hb h
  · cases h

theorem pre_cudaLaunchNamed' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cudaLaunchNamed as S = true) : Contracts.Pre .cudaLaunchNamed bits w := by
  dsimp only [preB] at h
  unfold preCudaLaunchNamed at h
  split at h
  · exact pre_cudaLaunchNamed hS hb h
  · cases h

theorem pre_gpuInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuInit as S = true) : Contracts.Pre .gpuInit bits w := by
  dsimp only [preB] at h
  unfold preGpuInit at h
  split at h
  · exact pre_gpuInit hS hb h
  · cases h

theorem pre_gpuCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuCleanup as S = true) : Contracts.Pre .gpuCleanup bits w := by
  dsimp only [preB] at h
  unfold preGpuCleanup at h
  split at h
  · exact pre_gpuCleanup hS hb h
  · cases h

theorem pre_gpuCreateBuffer' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuCreateBuffer as S = true) : Contracts.Pre .gpuCreateBuffer bits w := by
  dsimp only [preB] at h
  unfold preGpuCreateBuffer at h
  split at h
  · exact pre_gpuCreateBuffer hS hb h
  · cases h

theorem pre_gpuDispatch' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuDispatch as S = true) : Contracts.Pre .gpuDispatch bits w := by
  dsimp only [preB] at h
  unfold preGpuDispatch at h
  split at h
  · exact pre_gpuDispatch hS hb h
  · cases h

theorem pre_gpuCreatePipeline' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuCreatePipeline as S = true) : Contracts.Pre .gpuCreatePipeline bits w := by
  dsimp only [preB] at h
  unfold preGpuCreatePipeline at h
  split at h
  · exact pre_gpuCreatePipeline hS hb h
  · cases h

theorem pre_gpuUpload' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuUpload as S = true) : Contracts.Pre .gpuUpload bits w := by
  dsimp only [preB] at h
  unfold preGpuUpload at h
  split at h
  · exact pre_gpuUpload hS hb h
  · cases h

theorem pre_gpuDownload' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .gpuDownload as S = true) : Contracts.Pre .gpuDownload bits w := by
  dsimp only [preB] at h
  unfold preGpuDownload at h
  split at h
  · exact pre_gpuDownload hS hb h
  · cases h

theorem pre_windowInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .windowInit as S = true) : Contracts.Pre .windowInit bits w := by
  dsimp only [preB] at h
  unfold preWindowInit at h
  split at h
  · exact pre_windowInit hS hb h
  · cases h

theorem pre_windowOpen' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .windowOpen as S = true) : Contracts.Pre .windowOpen bits w := by
  dsimp only [preB] at h
  unfold preWindowOpen at h
  split at h
  · exact pre_windowOpen hS hb h
  · cases h

theorem pre_windowPoll' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .windowPoll as S = true) : Contracts.Pre .windowPoll bits w := by
  dsimp only [preB] at h
  unfold preWindowPoll at h
  split at h
  · exact pre_windowPoll hS hb h
  · cases h

theorem pre_windowPresentGpuBuffer' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .windowPresentGpuBuffer as S = true) : Contracts.Pre .windowPresentGpuBuffer bits w := by
  dsimp only [preB] at h
  unfold preWindowPresentGpuBuffer at h
  split at h
  · exact pre_windowPresentGpuBuffer hS hb h
  · cases h

theorem pre_windowCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .windowCleanup as S = true) : Contracts.Pre .windowCleanup bits w := by
  dsimp only [preB] at h
  unfold preWindowCleanup at h
  split at h
  · exact pre_windowCleanup hS hb h
  · cases h

theorem pre_lmdbCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbCleanup as S = true) : Contracts.Pre .lmdbCleanup bits w := by
  dsimp only [preB] at h
  unfold preLmdbCleanup at h
  split at h
  · exact pre_lmdbCleanup hS hb h
  · cases h

theorem pre_fileRead' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .fileRead as S = true) : Contracts.Pre .fileRead bits w := by
  dsimp only [preB] at h
  unfold preFileRead at h
  split at h
  · exact pre_fileRead hS hb h
  · cases h

theorem pre_fileWrite' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .fileWrite as S = true) : Contracts.Pre .fileWrite bits w := by
  dsimp only [preB] at h
  unfold preFileWrite at h
  split at h
  · exact pre_fileWrite hS hb h
  · cases h

theorem pre_fileReadToPtr' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .fileReadToPtr as S = true) : Contracts.Pre .fileReadToPtr bits w := by
  dsimp only [preB] at h
  unfold preFileReadToPtr at h
  split at h
  · exact pre_fileReadToPtr hS hb h
  · cases h

theorem pre_nativeLoad' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .nativeLoad as S = true) : Contracts.Pre .nativeLoad bits w := by
  dsimp only [preB] at h
  unfold preNativeLoad at h
  split at h
  · exact pre_nativeLoad hS hb h
  · cases h

theorem pre_cpuHas' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cpuHas as S = true) : Contracts.Pre .cpuHas bits w := by
  dsimp only [preB] at h
  unfold preCpuHas at h
  split at h
  · exact pre_cpuHas hS hb h
  · cases h

theorem pre_htInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .htInit as S = true) : Contracts.Pre .htInit bits w := by
  dsimp only [preB] at h
  unfold preHtInit at h
  split at h
  · exact pre_htInit hS hb h
  · cases h

theorem pre_htCleanup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .htCleanup as S = true) : Contracts.Pre .htCleanup bits w := by
  dsimp only [preB] at h
  unfold preHtCleanup at h
  split at h
  · exact pre_htCleanup hS hb h
  · cases h

theorem pre_htCreate' {as : List AV} (hb : ArgBits Γ w as bits)
    (h : preB .htCreate as S = true) : Contracts.Pre .htCreate bits w := by
  dsimp only [preB] at h
  unfold preHtCreate at h
  split at h
  · exact pre_htCreate (S := S) hb h
  · cases h

theorem pre_lmdbInit' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbInit as S = true) : Contracts.Pre .lmdbInit bits w := by
  dsimp only [preB] at h
  unfold preLmdbInit at h
  split at h
  · exact pre_lmdbInit hS hb h
  · cases h

theorem pre_lmdbOpen' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbOpen as S = true) : Contracts.Pre .lmdbOpen bits w := by
  dsimp only [preB] at h
  unfold preLmdbOpen at h
  split at h
  · exact pre_lmdbOpen hS hb h
  · cases h

theorem pre_lmdbBeginWriteTxn' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbBeginWriteTxn as S = true) : Contracts.Pre .lmdbBeginWriteTxn bits w := by
  dsimp only [preB] at h
  unfold preLmdbBeginWriteTxn at h
  split at h
  · exact pre_lmdbBeginWriteTxn hS hb h
  · cases h

theorem pre_lmdbCommitWriteTxn' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbCommitWriteTxn as S = true) : Contracts.Pre .lmdbCommitWriteTxn bits w := by
  dsimp only [preB] at h
  unfold preLmdbCommitWriteTxn at h
  split at h
  · exact pre_lmdbCommitWriteTxn hS hb h
  · cases h

theorem pre_lmdbPut' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbPut as S = true) : Contracts.Pre .lmdbPut bits w := by
  dsimp only [preB] at h
  unfold preLmdbPut at h
  split at h
  · exact pre_lmdbPut hS hb h
  · cases h

theorem pre_htInsert' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .htInsert as S = true) : Contracts.Pre .htInsert bits w := by
  dsimp only [preB] at h
  unfold preHtInsert at h
  split at h
  · exact pre_htInsert hS hb h
  · cases h

theorem pre_htLookup' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .htLookup as S = true) : Contracts.Pre .htLookup bits w := by
  dsimp only [preB] at h
  unfold preHtLookup at h
  split at h
  · exact pre_htLookup hS hb h
  · cases h

theorem pre_lmdbCursorScan' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .lmdbCursorScan as S = true) : Contracts.Pre .lmdbCursorScan bits w := by
  dsimp only [preB] at h
  unfold preLmdbCursorScan at h
  split at h
  · exact pre_lmdbCursorScan hS hb h
  · cases h

theorem pre_cublasSgemv' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cublasSgemv as S = true) : Contracts.Pre .cublasSgemv bits w := by
  dsimp only [preB] at h
  unfold preCublasSgemv at h
  split at h
  · exact pre_cublasSgemv hS hb h
  · cases h

theorem pre_cublasSgemm' {as : List AV} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB .cublasSgemm as S = true) : Contracts.Pre .cublasSgemm bits w := by
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
    (h : preB .nativeArch as S = true) : Contracts.Pre .nativeArch bits w := by
  dsimp only [preB] at h
  unfold preNativeArch at h
  split at h
  · exact pre_nativeArch (S := S) hb h
  · cases h

end entries

/-- **What `preB` accepts, `Pre` holds of**: where the checker describes the
    arguments and the typestate, a call `preB` accepts has its precondition. -/
theorem preB_sound {f : Ffi} {as : List AV} {bits : List UInt64} (hS : S.holds w) (hb : ArgBits Γ w as bits)
    (h : preB f as S = true) : Contracts.Pre f bits w := by
  cases f
  case cudaInit => exact pre_cudaInit' hS hb h
  case cudaCleanup => exact pre_cudaCleanup' hS hb h
  case cudaCreateBuffer => exact pre_cudaCreateBuffer' hS hb h
  case cudaSync => exact pre_cudaSync' hS hb h
  case cudaUpload => exact pre_cudaUpload' hS hb h
  case cudaDownload => exact pre_cudaDownload' hS hb h
  case cudaUploadOffset => exact pre_cudaUploadOffset' hS hb h
  case cudaDownloadOffset => exact pre_cudaDownloadOffset' hS hb h
  case cudaLaunch => exact pre_cudaLaunch' hS hb h
  case cudaLaunchNamed => exact pre_cudaLaunchNamed' hS hb h
  case gpuInit => exact pre_gpuInit' hS hb h
  case gpuCleanup => exact pre_gpuCleanup' hS hb h
  case gpuCreateBuffer => exact pre_gpuCreateBuffer' hS hb h
  case gpuDispatch => exact pre_gpuDispatch' hS hb h
  case gpuCreatePipeline => exact pre_gpuCreatePipeline' hS hb h
  case gpuUpload => exact pre_gpuUpload' hS hb h
  case gpuDownload => exact pre_gpuDownload' hS hb h
  case windowInit => exact pre_windowInit' hS hb h
  case windowOpen => exact pre_windowOpen' hS hb h
  case windowPoll => exact pre_windowPoll' hS hb h
  case windowPresentGpuBuffer => exact pre_windowPresentGpuBuffer' hS hb h
  case windowCleanup => exact pre_windowCleanup' hS hb h
  case lmdbCleanup => exact pre_lmdbCleanup' hS hb h
  case fileRead => exact pre_fileRead' hS hb h
  case fileWrite => exact pre_fileWrite' hS hb h
  case fileReadToPtr => exact pre_fileReadToPtr' hS hb h
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
  all_goals exact absurd h (by simp [preB])

end AlgorithmLib.Check
