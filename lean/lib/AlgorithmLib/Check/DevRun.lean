module
public import AlgorithmLib.Check.Sound
meta import AlgorithmLib.Check.Sound
public import AlgorithmLib.Host.DrvLaunch
meta import AlgorithmLib.Host.DrvLaunch
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# What a device run keeps

A run that only launches kernels, calls cuBLAS's products and waits for the
context, outside a capture, leaves the device as it found it but for the race
tracker and the contents of its allocations, each at its size (`DevSame`),
and leaves the kernel and vendor oracles. So what the device was set up with
before such a run --- its functions, their modules' text, the allocations ---
it still has at every call of it.
-/

namespace AlgorithmLib.Check

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Contracts

/-- The device as it was, but for the race tracker and the contents of its
    allocations, each at its size. -/
structure DevSame (d d' : Dev) : Prop where
  live : d'.live = d.live
  streams : d'.streams = d.streams
  capture : d'.capture = d.capture
  events : d'.events = d.events
  modules : d'.modules = d.modules
  funcs : d'.funcs = d.funcs
  drvInit : d'.drvInit = d.drvInit
  retained : d'.retained = d.retained
  size : d'.bufs.size = d.bufs.size
  blas : d'.blas = d.blas
  graphs : d'.graphs = d.graphs
  execs : d'.execs = d.execs
  sizes : ∀ i, (d'.get? i).map ByteArray.size = (d.get? i).map ByteArray.size
  kernelWrites : d'.kernelWrites = d.kernelWrites

theorem DevSame.refl (d : Dev) : DevSame d d :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, fun _ => rfl, rfl⟩

theorem DevSame.trans {a b c : Dev} (h1 : DevSame a b) (h2 : DevSame b c) : DevSame a c :=
  ⟨h2.live.trans h1.live, h2.streams.trans h1.streams, h2.capture.trans h1.capture,
   h2.events.trans h1.events, h2.modules.trans h1.modules, h2.funcs.trans h1.funcs,
   h2.drvInit.trans h1.drvInit, h2.retained.trans h1.retained, h2.size.trans h1.size, h2.blas.trans h1.blas,
   h2.graphs.trans h1.graphs, h2.execs.trans h1.execs, fun i => (h2.sizes i).trans (h1.sizes i),
   h2.kernelWrites.trans h1.kernelWrites⟩

theorem DevSame.of_bufsOnly {d d' : Dev} (h : BufsOnly d d') : DevSame d d' :=
  ⟨h.live, h.streams, h.capture, h.events, h.modules, h.funcs, h.drvInit, h.retained, h.size, h.blas,
   h.graphs, h.execs, h.sizes, h.kernelWrites⟩

theorem DevSame.race (d : Dev) (r : Race) : DevSame d { d with race := r } :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, fun _ => rfl, rfl⟩

/-- **A device operation outside a capture keeps the device** but for the
    race tracker and the contents of its allocations, where kernels keep the
    sizes of what they write. -/
theorem devOp_same {d d' : Dev} {w : World} {p : Nat} {op : DevOp} {rs ws : List Nat}
    (hc : d.capture = none) (hk : KeepsSize w.kernel) (h : d.devOp w p op rs ws = some d') : DevSame d d' := by
  unfold Dev.devOp at h
  rw [hc] at h
  unfold Dev.devOp.run at h
  obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
  split at h
  · exact (DevSame.race d r).trans (DevSame.of_bufsOnly (runLaunch_bufsOnly hk h))
  · exact (DevSame.race d r).trans (DevSame.of_bufsOnly (runVendor_bufsOnly h))

/-- What a call into the device that answered keeps: the device as
    `DevSame` says, and the kernel and vendor oracles. -/
def CallKeeps (w w' : World) : Prop := DevSame w.dev w'.dev ∧ Fixed w w'

theorem CallKeeps.refl (w : World) : CallKeeps w w := ⟨DevSame.refl _, Fixed.refl _⟩

theorem CallKeeps.trans {a b c : World} (h1 : CallKeeps a b) (h2 : CallKeeps b c) : CallKeeps a c :=
  ⟨h1.1.trans h2.1, h1.2.trans h2.2⟩

theorem devOnly_keeps {w w' : World} {k : Option (Option V × Dev)} {r : Option V}
    (h : devOnly w k = some (r, w')) (hk : ∀ r d, k = some (r, d) → DevSame w.dev d) : CallKeeps w w' := by
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq h
  exact ⟨hk _ _ hkd, Fixed.refl _⟩

/-- A driver call, where the driver is loaded, is the model's. -/
theorem extCall_cuda {f : CudaFn} {vs : List V} {w w' : World} {r : Option V}
    (hh : w.has (Ext.cuda f).lib = true) (h : extCall (.cuda f) vs w = some (r, w')) :
    ∃ bits, vs.mapM asBits = some bits ∧ cudaDrv f bits w = some (r, w') := by
  simp only [extCall, hh, Bool.not_true, Bool.false_eq_true, if_false, Option.bind_eq_bind] at h
  obtain ⟨bits, hb, h⟩ := Option.bind_eq_some_iff.mp h
  exact ⟨bits, hb, h⟩

/-- A cuBLAS call, where cuBLAS is loaded, is the model's. -/
theorem extCall_cublas {f : CublasFn} {vs : List V} {w w' : World} {r : Option V}
    (hh : w.has (Ext.cublas f).lib = true) (h : extCall (.cublas f) vs w = some (r, w')) :
    ∃ bits, vs.mapM asBits = some bits ∧ cublasCall f bits w = some (r, w') := by
  simp only [extCall, hh, Bool.not_true, Bool.false_eq_true, if_false, Option.bind_eq_bind] at h
  obtain ⟨bits, hb, h⟩ := Option.bind_eq_some_iff.mp h
  exact ⟨bits, hb, h⟩

/-- **A kernel launch that answered outside a capture keeps the device.** -/
theorem launch_keeps {bits : List UInt64} {w w' : World} {r : Option V} (hcap : w.dev.capture = none)
    (hk : KeepsSize w.kernel) (h : cudaDrv .launchKernel bits w = some (r, w')) : CallKeeps w w' := by
  rw [cudaDrv_step hcap] at h
  simp only [cudaStep] at h
  split at h
  · split at h
    · cases h
    · simp only [Option.bind_eq_bind] at h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨-, rfl⟩ := h
        exact CallKeeps.refl _
      · obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
        obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
        refine devOnly_keeps h fun r d hkd => ?_
        obtain ⟨d0, hd0, hkd⟩ := Option.bind_eq_some_iff.mp hkd
        simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
        obtain ⟨-, rfl⟩ := hkd
        exact devOp_same hcap hk hd0
  · cases h

/-- **A context wait that answered outside a capture keeps the device.** -/
theorem ctxSync_keeps {bits : List UInt64} {w w' : World} {r : Option V} (hcap : w.dev.capture = none)
    (h : cudaDrv .ctxSynchronize bits w = some (r, w')) : CallKeeps w w' := by
  rw [cudaDrv_step hcap] at h
  simp only [cudaStep] at h
  split at h
  · split at h
    · cases h
    · refine devOnly_keeps h fun r d hkd => ?_
      simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
      obtain ⟨-, rfl⟩ := hkd
      exact DevSame.race _ _
  · cases h

/-- **A strided-batched product that answered outside a capture keeps the
    device.** -/
theorem strided_keeps {h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch : UInt64} {w w' : World}
    {r : Option V} (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (hc : cublasCall .sgemmStridedBatched [h, ta, tb, m, n, k, pa, A, lda, sa, B, ldb, sb, pb, C, ldc, sc, batch] w
      = some (r, w')) : CallKeeps w w' := by
  simp only [cublasCall] at hc
  split at hc
  · cases hc
  · unfold gemmCall at hc
    simp only [Option.bind_eq_bind] at hc
    obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
    split at hc
    · simp only [Option.some.injEq, Prod.mk.injEq] at hc
      obtain ⟨-, rfl⟩ := hc
      exact CallKeeps.refl _
    · split at hc
      · cases hc
      · obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
        obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
        obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
        split at hc
        · cases hc
        · obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
          obtain ⟨_, -, hc⟩ := Option.bind_eq_some_iff.mp hc
          refine devOnly_keeps hc fun r d hkd => ?_
          obtain ⟨d0, hd0, hkd⟩ := Option.bind_eq_some_iff.mp hkd
          simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
          obtain ⟨-, rfl⟩ := hkd
          exact devOp_same hcap hk hd0

/-- Which allocation a pointer names, and where in it, the allocations'
    sizes decide. -/
theorem range_fst_same {d d' : Dev} (h : DevSame d d') (p : UInt64) (n : Nat) :
    (d'.range? p n).map (fun x => (x.1, x.2.1)) = (d.range? p n).map (fun x => (x.1, x.2.1)) := by
  unfold Dev.range?
  cases hp : devPtrOf p with
  | none => rfl
  | some q =>
      obtain ⟨id, off⟩ := q
      have hs : ((d'.bufs[id]?).join).map ByteArray.size = ((d.bufs[id]?).join).map ByteArray.size := by
        have := h.sizes (id : Int); simpa [Dev.get?] using this
      simp only [Option.bind_eq_bind, Option.bind_some]
      cases h1 : (d.bufs[id]?).join <;> cases h2 : (d'.bufs[id]?).join <;> rw [h1, h2] at hs <;>
        simp at hs ⊢
      split <;> split <;> simp_all

/-- The allocations a launch's parameters bind, the allocations' sizes
    decide. -/
theorem bindsOf_same {d d' : Dev} (h : DevSame d d') (args : List UInt64) :
    HProg.DrvLaunch.bindsOf d' args = HProg.DrvLaunch.bindsOf d args := by
  unfold HProg.DrvLaunch.bindsOf
  congr 2
  funext a
  have := range_fst_same h a 0
  have e : ∀ (x : Option (Nat × Nat × ByteArray)), x.map (·.1) = (x.map (fun x => (x.1, x.2.1))).map (·.1) :=
    fun x => by cases x <;> rfl
  rw [e, this, ← e]

/-- What a launch writes, the allocations' sizes and the device's write
    flags decide. -/
theorem writesOf_same {d d' : Dev} (h : DevSame d d') (fl : Option (List Bool)) (args : List UInt64) :
    d'.writesOf fl args = d.writesOf fl args := by
  unfold Dev.writesOf
  congr 2
  funext ⟨a, i⟩
  dsimp only
  split
  · have := range_fst_same h a 0
    have e : ∀ (x : Option (Nat × Nat × ByteArray)), x.map (·.1) = (x.map (fun x => (x.1, x.2.1))).map (·.1) :=
      fun x => by cases x <;> rfl
    rw [e, this, ← e]
  · rfl

/-- **A strided-batched product that answered, read from its answer**: on
    arguments cuBLAS takes, outside a capture, it read scalars `alpha` and
    `beta`, and device memory afterwards is the vendor's step for the call
    those and its other arguments make, its operands in allocations `ia`, `ib`,
    `ic` from elements `oa`, `ob`, `oc`, run on device memory before. -/
theorem strided_done {h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch : UInt64} {w w' : World}
    {r : Option V} (hcap : w.dev.capture = none) (hv : gemmInvalid ta tb m n k lda ldb ldc batch = false)
    (hc : cublasCall .sgemmStridedBatched [h, ta, tb, m, n, k, pa, A, lda, sa, B, ldb, sb, pb, C, ldc, sc, batch] w
      = some (r, w')) :
    ∃ alpha beta ia oa ib ob ic oc, w.mem.load pa 4 = some alpha ∧ w.mem.load pb 4 = some beta ∧
      w.dev.operand? A 4 (gemmSpans ta tb m n k lda ldb ldc sa sb sc batch).1 = some (ia, oa) ∧
      w.dev.operand? B 4 (gemmSpans ta tb m n k lda ldb ldc sa sb sc batch).2.1 = some (ib, ob) ∧
      w.dev.operand? C 4 (gemmSpans ta tb m n k lda ldb ldc sa sb sc batch).2.2 = some (ic, oc) ∧
      Device.devMem w'.dev = (Device.vendorStep w.vendor
        ⟨"sgemmStridedBatched", [ta, tb, m, n, k, alpha, beta, sa, sb, sc, batch,
          UInt64.ofNat oa, UInt64.ofNat ob, UInt64.ofNat oc, lda, ldb, ldc]⟩ [ia, ib, ic] ic).run
        (Device.devMem w.dev) := by
  simp only [cublasCall] at hc
  split at hc
  · cases hc
  · unfold gemmCall at hc
    simp only [Option.bind_eq_bind] at hc
    obtain ⟨p, -, hc⟩ := Option.bind_eq_some_iff.mp hc
    rw [if_neg (by rw [hv]; exact Bool.false_ne_true)] at hc
    split at hc
    · cases hc
    · obtain ⟨⟨ia, oa⟩, hA, hc⟩ := Option.bind_eq_some_iff.mp hc
      obtain ⟨⟨ib, ob⟩, hB, hc⟩ := Option.bind_eq_some_iff.mp hc
      obtain ⟨⟨ic, oc⟩, hC, hc⟩ := Option.bind_eq_some_iff.mp hc
      split at hc
      · cases hc
      · obtain ⟨alpha, hal, hc⟩ := Option.bind_eq_some_iff.mp hc
        obtain ⟨beta, hbe, hc⟩ := Option.bind_eq_some_iff.mp hc
        obtain ⟨d, hkd, rfl⟩ := devOnly_eq hc
        obtain ⟨d0, hd0, hkd⟩ := Option.bind_eq_some_iff.mp hkd
        simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
        obtain ⟨-, rfl⟩ := hkd
        exact ⟨alpha, beta, ia, oa, ib, ob, ic, oc, hal, hbe, hA, hB, hC, devOp_vendor_mem hcap hd0⟩

end AlgorithmLib.Check
