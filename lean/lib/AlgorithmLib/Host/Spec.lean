module
public import AlgorithmLib.Host.MemLaws
meta import AlgorithmLib.Host.MemLaws
public import AlgorithmLib.Host.Frames
meta import AlgorithmLib.Host.Frames
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# Frames

What a frame lets a call write (`Frame.allows`), checked against a call's
argument bits (`Frame.mayWrite_sound`), and the facts about the base
functions' models that the proofs of their frames read.
-/

namespace AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

-- ---------------------------------------------------------------------------
-- What a frame permits
-- ---------------------------------------------------------------------------

/-- The addresses `fr` lets a call with argument bits `args` and result bits
    `ret` write. A data-dependent frame names where writing *starts*; in a
    wrapping 64-bit address space that bounds nothing, so it permits every
    address, as `anywhere` does. -/
def Frame.allows (fr : Frame) (args : List UInt64) (ret : Option UInt64) (a : UInt64) : Prop :=
  let arg (k : Nat) := args.getD k 0
  let within (b : UInt64) (len : Nat) := ∃ i < len, a = b + UInt64.ofNat i
  match fr with
  | .none => False
  | .at d l => within (arg d) (arg l).toNat
  | .atOff b o l => within (arg b + arg o) (arg l).toNat
  | .fixed d n => within (arg d) n
  | .fixed2 d e n => within (arg d) n ∨ within (arg e) n
  | .atRecords d c n => within (arg d) (n * (asI32 (arg c)).toNat)
  | .dataDependent _ | .dataDependent2 _ _ | .anywhere => True

/-- Whether `x` lies in `len` bytes from `b`. -/
def within (b : UInt64) (len : Nat) (x : UInt64) : Bool := decide ((x - b).toNat < len)

theorem within_of {b x : UInt64} {len : Nat} (hl : len ≤ 2 ^ 64) (h : ∃ i < len, x = b + UInt64.ofNat i) :
    within b len x = true := by
  obtain ⟨i, hi, rfl⟩ := h
  simp only [within, decide_eq_true_eq]
  rw [UInt64.add_comm, UInt64.add_sub_cancel, UInt64.toNat_ofNat']
  have : i % 2 ^ 64 ≤ i := Nat.mod_le _ _
  omega

theorem toNat_le_size (x : UInt64) : x.toNat ≤ 2 ^ 64 := by
  have := UInt64.toNat_lt_size x; simp [UInt64.size] at this; omega

/-- Whether a call with argument bits `args` under frame `fr` may write the
    byte at `x`, over every answer it may give: computed, where `allows` is
    stated. -/
def Frame.mayWrite (fr : Frame) (args : List UInt64) (x : UInt64) : Bool :=
  let arg (k : Nat) := args.getD k 0
  match fr with
  | .none => false
  | .at d l => within (arg d) (arg l).toNat x
  | .atOff b o l => within (arg b + arg o) (arg l).toNat x
  | .fixed d n => n > 2 ^ 64 || within (arg d) n x
  | .fixed2 d e n => n > 2 ^ 64 || within (arg d) n x || within (arg e) n x
  | .atRecords d c n => n * (asI32 (arg c)).toNat > 2 ^ 64 || within (arg d) (n * (asI32 (arg c)).toNat) x
  | _ => true

theorem Frame.mayWrite_sound {fr : Frame} {args : List UInt64} {x : UInt64} (h : fr.mayWrite args x = false)
    (ret : Option UInt64) : ¬ fr.allows args ret x := by
  intro ha
  unfold Frame.mayWrite at h
  unfold Frame.allows at ha
  cases fr with
  | none => exact ha
  | «at» d l =>
      dsimp only at h ha
      rw [within_of (toNat_le_size _) ha] at h; cases h
  | atOff b o l =>
      dsimp only at h ha
      rw [within_of (toNat_le_size _) ha] at h; cases h
  | fixed d n =>
      dsimp only at h ha
      simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not, Nat.not_lt] at h
      rw [within_of h.1 ha] at h; cases h.2
  | fixed2 d e n =>
      dsimp only at h ha
      simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not, Nat.not_lt] at h
      rcases ha with ha | ha
      · rw [within_of h.1.1 ha] at h; cases h.1.2
      · rw [within_of h.1.1 ha] at h; cases h.2
  | atRecords d c n =>
      dsimp only at h ha
      simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not, Nat.not_lt] at h
      rw [within_of h.1 ha] at h; cases h.2
  | dataDependent _ => cases h
  | dataDependent2 _ _ => cases h
  | anywhere => cases h

-- ---------------------------------------------------------------------------
-- The models' memory
-- ---------------------------------------------------------------------------

/-- A count the runtime returns as `i64` reads back as itself. -/
theorem asBits_ofInt_i64_nat (n : Nat) : asBits (ofInt .i64 (n : Int)) = some (UInt64.ofNat n) := by
  simp only [ofInt, asBits, norm, widthMask]
  apply congrArg some
  apply UInt64.toNat_inj.mp
  have e : ((n : Int).emod 18446744073709551616).toNat = n % 18446744073709551616 := by
    show ((n : Int) % 18446744073709551616).toNat = _; omega
  simp only [UInt64.toNat_and, UInt64.toNat_ofNat']
  rw [show (18446744073709551615 : UInt64).toNat = 2 ^ 64 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
  simp only [Nat.mod_mod]
  rw [show (((1 <<< 64 : Nat)) : Int) = 18446744073709551616 from rfl, e]
  simp

/-- The bytes a family copies are as many as it counted. -/
theorem size_toByteArray_map_range (g : Nat → UInt8) (n : Nat) :
    ((List.range n).map g).toByteArray.size = n := by
  simp [List.size_toByteArray]

/-- A length the runtime reads as a positive `i64` is at least one. -/
theorem pos_of_asI64 {x : UInt64} (h : ¬ asI64 x ≤ 0) : 1 ≤ x.toNat := by
  rcases Nat.eq_zero_or_pos x.toNat with hz | hz
  · have : x = 0 := UInt64.toNat_inj.mp (by simpa using hz)
    subst this; exact absurd (by decide) h
  · exact hz

/-- A device-only contract changes no host byte: at most it hands back bytes
    whose copies the host has now waited for. -/
theorem devOnly_mem {w : World} {k : Option (Option V × Dev)} {r : Option V} {w' : World}
    (h : devOnly w k = some (r, w')) : ∀ reg, w'.mem.region reg = w.mem.region reg := by
  unfold devOnly at h
  obtain ⟨⟨_, _⟩, -, h⟩ := Option.map_eq_some_iff.mp h
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨-, rfl⟩ := h
  intro reg; cases reg <;> rfl

theorem le64_size (x : UInt64) : (le64 x).size = 8 := by
  simp [le64]

theorem ffiNativeLoad_mem {bits : List UInt64} {w : World} {r w'}
    (h : ffiNativeLoad bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiNativeLoad at h
  split at h
  · split at h
    · (first | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl) | cases h)
    · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      (first | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl) | cases h)
  · cases h

theorem ffiNativeFree_mem {bits : List UInt64} {w : World} {r w'}
    (h : ffiNativeFree bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiNativeFree at h
  split at h
  · split at h <;> (first | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl) | cases h)
  · cases h

theorem ffiNativeArch_mem {bits : List UInt64} {w : World} {r w'}
    (h : ffiNativeArch bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiNativeArch at h
  split at h
  · (first | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl) | cases h)
  · cases h

theorem ffiCpuHas_mem {bits : List UInt64} {w : World} {r w'}
    (h : ffiCpuHas bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiCpuHas at h
  split at h
  · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    split at h <;> (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl)
  · cases h

theorem ffiFileCreateDirAll_mem {bits : List UInt64} {w : World} {r w'}
    (h : ffiFileCreateDirAll bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiFileCreateDirAll at h
  split at h
  · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    repeat' split at h
    all_goals (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl)
  · cases h

theorem ffiGrant_mem_memLock {bits : List UInt64} {w : World} {r w'}
    (h : ffiGrant "lock" 2 bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiGrant at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
  · cases h

theorem ffiGrant_mem_memUnlock {bits : List UInt64} {w : World} {r w'}
    (h : ffiGrant "unlock" 2 bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiGrant at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
  · cases h

theorem ffiGrant_mem_memAdviseHuge {bits : List UInt64} {w : World} {r w'}
    (h : ffiGrant "hugepages" 2 bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiGrant at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
  · cases h

theorem ffiGrant_mem_threadPriority {bits : List UInt64} {w : World} {r w'}
    (h : ffiGrant "priority" 1 bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiGrant at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
  · cases h

theorem ffiMemWait_world {bits : List UInt64} {w : World} {r w'}
    (h : ffiMemWait bits w = some (r, w')) : w' = w := by
  unfold ffiMemWait at h
  repeat' split at h
  all_goals (cases h <;> rfl)

theorem ffiMemWake_world {bits : List UInt64} {w : World} {r w'}
    (h : ffiMemWake bits w = some (r, w')) : w' = w := by
  unfold ffiMemWake at h
  repeat' split at h
  all_goals (cases h <;> rfl)

theorem ffiFileSize_world {bits : List UInt64} {w : World} {r w'}
    (h : ffiFileSize bits w = some (r, w')) : w' = w := by
  unfold ffiFileSize at h
  split at h
  · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    repeat' split at h
    all_goals (simp only [Option.some.injEq, Prod.mk.injEq] at h; exact h.2.symm)
  · cases h

theorem ffiFileSetLen_mem {bits : List UInt64} {w : World} {r w'}
    (h : ffiFileSetLen bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiFileSetLen at h
  split at h
  · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
    repeat' split at h
    all_goals (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl)
  · cases h

theorem ffiClockNow_world {bits : List UInt64} {w : World} {r w'}
    (h : ffiClockNow bits w = some (r, w')) : w' = w := by
  unfold ffiClockNow at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h; exact h.2.symm
  · cases h

/-- An asynchronous copy changes host bytes only as its host side does. -/
theorem asyncCopy_region {w : World} {p : Nat} {ha : UInt64} {n : Nat} {wr : Bool}
    {rs ws : List Nat} {k : Race → Option (Dev × Mem)} {ret : Option V} {w' : World}
    (h : asyncCopy w p ha n wr rs ws k = some (ret, w')) :
    ∃ r d m, k r = some (d, m) ∧ (∀ reg, w'.mem.region reg = m.region reg) ∧
      w'.mem.pinnedLive = m.pinnedLive ∧
      ∀ b ∈ w.mem.busy, b ∈ w'.mem.busy := by
  unfold asyncCopy at h
  split at h
  · cases h
  · obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨d, m⟩, hk, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    refine ⟨r, d, m, hk, fun reg => by cases reg <;> rfl, rfl, ?_⟩
    intro b hb
    dsimp only
    split
    · exact List.mem_cons_of_mem _ hb
    · exact hb

end AlgorithmLib.HProg
