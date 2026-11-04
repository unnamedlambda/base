module
public import AlgorithmLib.Host.MemLaws
meta import AlgorithmLib.Host.MemLaws
public import AlgorithmLib.Host.Frames
meta import AlgorithmLib.Host.Frames
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# One contract per entry point

What a foreign call does is stated in three places, each for a reason:

* `Ffi.sig` — the ABI: parameter and result types, and so the id. Every entry
  point has one.
* `frame` — which memory the call may write. Every entry point has one.
* `Sem.callFfi` — what the call computes: `callBits`, one transcription of
  `base/src/ffi/` per entry point, checked by the conformance corpora. Every
  entry point but `threadSpawn`, whose callee is one of the program's own
  functions, has one.

`Ffi.spec` gathers the three into one value, and `callFfi_respects_frame`
checks two of them against each other: every contract writes only inside its
declared frame, so no frame is an assumption. `Host.Lifecycle` adds when a call
answers and what it leaves of a typestate, and `Host.Contract` is the table
(`Callee.contract`).
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
-- The table
-- ---------------------------------------------------------------------------

/-- The entry points with an executable contract that may write host memory:
    the libm shims, the contexts' slots, and the LMDB context and scan. -/
def hostTouching : List IR.Ffi :=
  [.sinf, .cosf, .powf, .lmdbInit, .lmdbCleanup, .lmdbCursorScan,
   .threadInit, .threadCleanup, .threadJoin]

/-- The entry points with an executable contract that change only state the
    program cannot address: the LMDB context and what its directories hold. -/
def offHost : List IR.Ffi :=
  [.lmdbOpen, .lmdbBeginWriteTxn, .lmdbPut, .lmdbCommitWriteTxn]

/-- The entry points with an executable contract: the families the corpora run
    through the real symbols. -/
def contracted : List IR.Ffi := hostTouching ++ offHost

/-- Everything known about one entry point. -/
structure Spec where
  /-- Parameter types and result type: the ABI, and so the id. -/
  sig : List IR.ClifTy × Option IR.ClifTy
  /-- The memory a call may write. -/
  frame : Frame
  /-- What a call computes, where the model says. -/
  run? : Option (List V → World → Option (Option V × World))

/-- **The contract table.** -/
def _root_.AlgorithmLib.IR.Ffi.spec (f : IR.Ffi) : Spec where
  sig := f.sig
  frame := frame f
  run? := if f ∈ contracted then some (callFfi f) else none

/-- An entry point outside the table has no behavior in the model: calling it
    is stuck, never a guess. -/
theorem callBits_uncontracted (f : IR.Ffi) (hf : f ∉ contracted) (bits : List UInt64)
    (w : World) : callBits f bits w = none := by
  unfold callBits
  cases f <;> simp_all [contracted, hostTouching, offHost]

theorem callFfi_uncontracted (f : IR.Ffi) (hf : f ∉ contracted) (args : List V) (w : World) :
    callFfi f args w = none := by
  unfold callFfi
  cases args.mapM asBits with
  | none => rfl
  | some bits => exact callBits_uncontracted f hf bits w

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

theorem leBytes_size (n x : Nat) : (leBytes n x).size = n := size_toByteArray_map_range _ n

theorem lmdbScanBytes_fold_size (rows : LmdbTable) (acc : ByteArray) :
    (rows.foldl (fun acc r => acc ++ leBytes 2 r.1.size ++ leBytes 2 r.2.size ++ r.1 ++ r.2) acc).size
      = acc.size + (rows.map fun r => 4 + r.1.size + r.2.size).sum := by
  induction rows generalizing acc with
  | nil => simp
  | cons r rs ih =>
      simp only [List.foldl_cons, ih, ByteArray.size_append, leBytes_size, List.map_cons, List.sum_cons]
      omega

theorem lmdbFit_sum (room : Nat) (rows : LmdbTable) :
    ((lmdbFit room rows).map fun r => 4 + r.1.size + r.2.size).sum ≤ room := by
  induction rows generalizing room with
  | nil => simp [lmdbFit]
  | cons r rs ih =>
      simp only [lmdbFit]
      split
      · have := ih (room - (4 + r.1.size + r.2.size))
        simp only [List.map_cons, List.sum_cons]
        omega
      · simp

/-- **A scan writes no more than its room**: the count and the rows that fit. -/
theorem lmdbScanBytes_fit (room : Nat) (rows : LmdbTable) :
    (lmdbScanBytes (lmdbFit room rows)).size ≤ 4 + room := by
  unfold lmdbScanBytes
  rw [lmdbScanBytes_fold_size, leBytes_size]
  have := lmdbFit_sum room rows
  omega

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

/-- The contracts in `offHost` leave host memory as it was. -/
theorem ffiLmdbOpen_mem {bits : List UInt64} {w : World} {r w'}
    (h : ffiLmdbOpen bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiLmdbOpen at h
  split at h
  · split at h
    · cases h
    · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
    · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
      · split at h
        · cases h
        · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
  · cases h

theorem ffiLmdbBeginWriteTxn_mem {bits : List UInt64} {w : World} {r w'}
    (h : ffiLmdbBeginWriteTxn bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiLmdbBeginWriteTxn at h
  split at h
  · split at h
    · cases h
    · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
    · split at h
      · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
      · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
  · cases h

theorem ffiLmdbPut_mem {bits : List UInt64} {w : World} {r w'}
    (h : ffiLmdbPut bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiLmdbPut at h
  split at h
  · split at h
    · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
    · split at h
      · cases h
      · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
      · split at h
        · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
        · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
          obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
          split at h
          · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
          · split at h <;> (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl)
  · cases h

theorem ffiLmdbCommitWriteTxn_mem {bits : List UInt64} {w : World} {r w'}
    (h : ffiLmdbCommitWriteTxn bits w = some (r, w')) : w'.mem = w.mem := by
  unfold ffiLmdbCommitWriteTxn at h
  split at h
  · split at h
    · cases h
    · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
    · split at h
      · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
      · split at h
        · simp only [lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
        · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; rfl
  · cases h

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
      w'.mem.pinnedLive = m.pinnedLive ∧ w'.mem.frozen = m.frozen ∧
      ∀ b ∈ w.mem.busy, b ∈ w'.mem.busy := by
  unfold asyncCopy at h
  split at h
  · cases h
  · obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨d, m⟩, hk, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    refine ⟨r, d, m, hk, fun reg => by cases reg <;> rfl, rfl, rfl, ?_⟩
    intro b hb
    dsimp only
    split
    · exact List.mem_cons_of_mem _ hb
    · exact hb

set_option maxHeartbeats 2000000 in
set_option maxHeartbeats 8000000 in
/-- **Every executable contract writes only inside its frame.**

    A successful call changes the value of no byte its frame does not name.
    Over the entry points with a contract this is a theorem about the
    model; over the rest it holds because they have no behavior to violate it
    with (`callFfi_uncontracted`).
-/
theorem callFfi_respects_frame (f : IR.Ffi) (args : List V) (w : World) (ret : Option V) (w' : World)
    (h : callFfi f args w = some (ret, w'))
    (bits : List UInt64) (hb : args.mapM asBits = some bits)
    (a : UInt64) (hout : ¬ (frame f).allows bits (ret.bind asBits) a) :
    Unchanged w.mem w'.mem a := by
  by_cases hc : f ∈ contracted
  case neg =>
    have e := callFfi_uncontracted f hc args w
    rw [h] at e; cases e
  unfold callFfi at h; rw [hb, Option.bind_some] at h
  -- and every LMDB contract that only changes the store
  by_cases hst : f ∈ offHost
  case pos =>
    simp only [offHost, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
    · exact Unchanged.of_eq (by rw [ffiLmdbOpen_mem (bits := bits) (w := w) h])
    · exact Unchanged.of_eq (by rw [ffiLmdbBeginWriteTxn_mem (bits := bits) (w := w) h])
    · exact Unchanged.of_eq (by rw [ffiLmdbPut_mem (bits := bits) (w := w) h])
    · exact Unchanged.of_eq (by rw [ffiLmdbCommitWriteTxn_mem (bits := bits) (w := w) h])
  have hc' : f ∈ hostTouching := by
    simp only [contracted, List.mem_append] at hc
    exact hc.resolve_right hst
  simp only [hostTouching, List.mem_cons, List.not_mem_nil, or_false] at hc'
  rcases hc' with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals (delta callBits at h; dsimp only at h; simp only [ffiSinf, ffiCosf, ffiPowf, ffiLmdbInit, ffiLmdbCleanup, ffiLmdbCursorScan, ffiThreadInit, ffiThreadCleanup, ffiThreadJoin] at h)
  all_goals (split at h <;> (try contradiction))
  all_goals (simp only [frame, Frame.allows, ctxSlot, List.getD, List.getElem?_cons_zero,
    List.getElem?_cons_succ, Option.getD_some] at hout)
  all_goals first
    | exact absurd trivial hout
    | skip
  all_goals first
    -- a pure answer: sinf, cosf, powf, and the paths that refuse
    | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact Unchanged.refl _ _)
    | (simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact Unchanged.refl _ _)
    -- a context slot written: the inits and cleanups
    | (obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
       simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
       exact Unchanged.of_eq (Mem.store_other hm a (fun i hi e => hout ⟨i, hi, e⟩)))
    -- threadCleanup
    | (obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
       have hthaw : ∀ reg, ({ w.mem with frozen := false } : Mem).region reg = w.mem.region reg := by
         intro reg; cases reg <;> rfl
       split at h
       · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
         exact Unchanged.of_region hthaw a
       · split at h
         · obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
           simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
           intro x y hx hy
           have e := Mem.store_other hm a (fun i hi e => hout ⟨i, hi, e⟩)
           rw [e] at hy
           exact Mem.load_agree hthaw hx hy
         · cases h)
    -- threadJoin
    | (split at h
       · cases h
       · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact Unchanged.refl _ _
       · split at h
         · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
           exact Unchanged.of_region (fun reg => by cases reg <;> rfl) a
         · simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact Unchanged.refl _ _)
    -- every other: walk the model's branches and binds, then close by the
    -- memory it leaves: the same, one store into a context slot, or a copy the
    -- frame allows
    | (repeat' (first
         | contradiction
         | (split at h)
         | (rw [Option.bind_eq_some_iff] at h; obtain ⟨_, hbind, h⟩ := h)
         | (simp only [Option.bind_eq_bind] at h; rw [Option.bind_eq_some_iff] at h; obtain ⟨_, hbind, h⟩ := h))
       all_goals first
         | (simp only [cudaFail, lmdbI32, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
            exact Unchanged.refl _ _)
         | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
            first
              | exact Unchanged.refl _ _
              | exact Unchanged.of_eq (Mem.store_other hbind a (fun i hi e => hout ⟨i, hi, e⟩))
              | (apply Unchanged.of_eq; apply copyIn_other hbind; intro i hi e
                 refine hout ⟨i, ?_, e⟩
                 simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not,
                   ByteArray.size_extract, decide_eq_true_eq, gt_iff_lt, Nat.not_lt, leBytes_size] at *
                 omega)
              | (apply Unchanged.of_eq; apply copyIn_other hbind; intro i hi e
                 refine hout ⟨i, Nat.lt_of_lt_of_le hi (Nat.le_trans (lmdbScanBytes_fit _ _) ?_), e⟩
                 simp only [Nat.not_lt] at *
                 omega)))

/-- The table's two statements about one entry point agree: what `run?` does,
    `frame` permits. -/
theorem spec_frame_sound (f : IR.Ffi) {run : List V → World → Option (Option V × World)}
    (hr : f.spec.run? = some run) (args : List V) (w : World) (ret : Option V) (w' : World)
    (h : run args w = some (ret, w')) (bits : List UInt64) (hb : args.mapM asBits = some bits)
    (a : UInt64) (hout : ¬ f.spec.frame.allows bits (ret.bind asBits) a) :
    Unchanged w.mem w'.mem a := by
  simp only [IR.Ffi.spec] at hr hout
  split at hr
  · cases hr; exact callFfi_respects_frame f args w ret w' h bits hb a hout
  · cases hr

end AlgorithmLib.HProg
