import Ship.VitCalls
import Vit.Host
import AlgorithmLib.ML.Model.TapeRun

/-!
# The words the ViT's forward leaves are the tape's

Every run of the forward (`VitCalls.run_calls`) takes its steps in the
schedule's order: each group's kernel launch, each contraction's product, then
a wait. On a device honest about each launch (`TapeLayout.machine`) and
about its products (`VendorGemmHonest`), each step keeps device memory in step
with the tape (`TapeLayout.unit_step_kernel`, `TapeLayout.gemm_step`), so the
run leaves every word inside each buffer, at the size the layout makes it, as
the tape's first `VFWD_N` operations leave it, each contraction read as the
vendor computes it (`fwd_agree`).

Each launch's honesty is that, where the group's statement kernel runs, the
device leaves what one run leaves (`TapeLayout.machine`), and each kernel runs
over buffers of the layout's sizes (`Vit.vit_runs`). That each launch's
parameters name its group's buffers, and that those start at least the
layout's sizes, follows from what `main` leaves and the layout (`ready`,
`RangeFacts`); the rest is `ML.calls_tape`.

A plain file, not a module, as the Ship files it reads are.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
  AlgorithmLib.HProg.Contracts AlgorithmLib.Device AlgorithmLib.ML AlgorithmLib.Kernel

namespace VitWords

/-- The steps `vIssue lo hi` takes, each with the group it runs. -/
def stepUnits (lo hi : Nat) : List (ML.DevCall × List Nat) :=
  ((List.range Vit.VNUNIT).zip Vit.vUnits).filterMap fun (k, u) =>
    if lo ≤ Vit.vUnitLo u && Vit.vUnitHi u < hi then
      if Vit.vIsGemm k then
        ((Vit.vUnitOps (Vit.vUnitArr.getD k [])).head? >>= ML.TOp.vendorGemm).map fun g => (.gemm g, u)
      else some (.kernel (Vit.vKernIx.getD k 0), u)
    else none

theorem stepUnits_map (lo hi : Nat) (f : Sum Nat ML.Gemm → ML.DevCall) (hl : ∀ j, f (.inl j) = .kernel j)
    (hr : ∀ g, f (.inr g) = .gemm g) : (stepUnits lo hi).map Prod.fst = (VitCalls.vSteps lo hi).map f := by
  simp only [stepUnits, VitCalls.vSteps, List.map_filterMap]
  congr 1
  funext p
  obtain ⟨k, u⟩ := p
  dsimp only
  split
  · split
    · cases ((Vit.vUnitOps (Vit.vUnitArr.getD k [])).head? >>= ML.TOp.vendorGemm) <;> simp [hr]
    · simp [hl]
  · rfl

/-- The steps `vIssue lo hi` takes, each with its group's operations. -/
def stepOps (lo hi : Nat) : List (ML.DevCall × List TOp) := (stepUnits lo hi).map fun p => (p.1, Vit.vUnitOps p.2)

theorem stepOps_calls (hi : Nat) : (stepOps 0 hi).map Prod.fst = (VitCalls.vSteps 0 hi).map ML.DevCall.ofStep := by
  rw [stepOps, List.map_map]
  exact stepUnits_map _ _ _ (fun _ => rfl) (fun _ => rfl)

/-- Every launch is of a group the statement kernels prove. -/
def provenOk (p : ML.DevCall × List Nat) : Bool :=
  match p.1 with
  | .kernel _ => Vit.vProvenUnits.contains p.2
  | _ => true

/-- Every allocation of the layout is on record in what `main` leaves, at its
    size. -/
theorem allocs_in : (List.range Vit.vApp.bufs.length).all
    (fun i => (VitState.after 0 0).contains (.alloc i (Vit.vApp.bufs.getD i 0))) = true := by
  native_decide

theorem alloc_of {w : World} (h : VitCalls.Run w) {i : Nat} (hi : i < Vit.vApp.bufs.length) :
    ∃ b, (w.dev.bufs[i]?).join = some b ∧ b.size = Vit.vApp.bufs.getD i 0 := by
  have hm := List.all_eq_true.mp allocs_in i (List.mem_range.mpr hi)
  rw [List.contains_iff_mem] at hm
  exact h _ hm

/-- **What `main` leaves is ready for a range's run**: the driver and cuBLAS
    loaded, no capture in progress, kernels that keep their sizes, every
    allocation live at its size. -/
theorem ready {w : World} (h : VitCalls.Run w) : ML.Ready Vit.vApp w := by
  have ha := VitCalls.after_has
  simp only [Bool.and_eq_true, List.contains_iff_mem] at ha
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := ha
  refine ⟨h _ h1, ?_, (h _ h4 : Contracts.Fact.holds .oracles w).1, fun _ => h _ h2, fun i hi => alloc_of h hi⟩
  have : w.dev.capture.isSome = false := h _ h3
  cases hc : w.dev.capture with
  | none => rfl
  | some _ => simp [hc] at this

/-- **What a range `0..hi` needs for its words to be the tape's**: every run
    makes its units' calls (`RangeCalls`); its steps are its units in the
    schedule's order, each a launch of a proven group or a product, each one
    the device and the step lemmas read, each launch handed exactly its
    group's buffers over its grid and every buffer at least the tape's size
    (`ML.stepsOk`); and its units bring each of the tape's first `hi`
    operations to the front past only ones it cannot observe. -/
structure RangeFacts (hi : Nat) : Prop where
  calls : VitCalls.RangeCalls hi
  units : (stepUnits 0 hi).map Prod.snd = Vit.vRangeUnits 0 hi
  proven : (stepUnits 0 hi).all provenOk = true
  ok : ML.stepsOk Vit.vLayout Vit.vApp (stepOps 0 hi) = true
  pull : pull Vit.SQ ((Vit.vFused.take hi).zipIdx.map fun p => (p.2, p.1)) (Vit.vRangeUnits 0 hi).flatten = true

/-- **The words a range leaves are the tape's**: every run of a range `0..hi`
    with its facts (`RangeFacts`) that ends, from what `main` leaves, on a
    device honest about each group's launch (`TapeLayout.KernelHonest`) and
    about its products, leaves every word inside each buffer, at the size the
    layout makes it, as the tape's first `hi` operations leave it from the
    memory the run started with, each contraction read as the vendor computes
    it (`ML.calls_tape`). -/
theorem range_agree {hi : Nat} (hF : RangeFacts hi) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : VitCalls.Run w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (Vit.vRangeFn 0 hi : Prog Slot Lvl Unit)) = .ok obs w')
    (hv : VendorGemmHonest w.vendor) (hvk : VendorKeeps w.vendor)
    (hHon : ∀ p ∈ stepUnits 0 hi, ∀ j, p.1 = .kernel j → Vit.vLayout.KernelHonest Vit.vApp w j (Vit.vUnitOps p.2) hw) :
    Vit.vLayout.Agree (devMem w'.dev)
      ((Vit.vFused.take hi).foldl (fun M op => TOp.denV hw op M) (wordView (devMem w.dev))) := by
  have hmem : ∀ p ∈ stepOps 0 hi, ∃ q ∈ stepUnits 0 hi, p = (q.1, Vit.vUnitOps q.2) := fun p hp => by
    obtain ⟨q, hq, e⟩ := List.mem_map.mp hp
    exact ⟨q, hq, e.symm⟩
  have hU : ∀ p ∈ stepOps 0 hi, ∀ j, p.1 = .kernel j → Vit.vLayout.UnitOk p.2 := fun p hp j hj => by
    obtain ⟨q, hq, rfl⟩ := hmem p hp
    have := List.all_eq_true.mp hF.proven q hq
    dsimp only at hj
    simp only [provenOk, hj] at this
    exact Vit.vit_unit_ok q.2 (List.contains_iff_mem.mp this)
  have hH : ∀ p ∈ stepOps 0 hi, ∀ j, p.1 = .kernel j → Vit.vLayout.KernelHonest Vit.vApp w j p.2 hw :=
    fun p hp j hj => by
      obtain ⟨q, hq, rfl⟩ := hmem p hp
      exact hHon q hq j hj
  have e := ML.calls_tape Vit.vLayout Vit.vApp rfl Vit.vBufBytes_le hw hw32 (VitCalls.vSteps 0 hi) (stepOps 0 hi)
    (stepOps_calls hi) hF.ok hU (ready h) hr (hF.calls cfg dataLen outLen w h hr) hv hvk hH
  letI := F32.ops hw
  rw [← Vit.vit_range_computes_tape_with (denV_frame hw Vit.SQ) hi hF.pull]
  have hfl : (Vit.vRangeUnits 0 hi).flatMap Vit.vUnitOps = (stepOps 0 hi).flatMap Prod.snd := by
    rw [← hF.units, List.flatMap_map, stepOps, List.flatMap_map]
  rw [hfl]
  exact e

theorem fwd_units : (stepUnits 0 Vit.VFWD_N).map Prod.snd = Vit.vRangeUnits 0 Vit.VFWD_N := by native_decide
theorem fwd_proven : (stepUnits 0 Vit.VFWD_N).all provenOk = true := by native_decide
theorem fwd_ok : ML.stepsOk Vit.vLayout Vit.vApp (stepOps 0 Vit.VFWD_N) = true := by native_decide

/-- The forward's facts. -/
theorem fwdFacts : RangeFacts Vit.VFWD_N :=
  ⟨VitCalls.run_calls, fwd_units, fwd_proven, fwd_ok, Vit.vit_fwd_pull⟩

/-- **The words the ViT's forward leaves are the tape's**: every run of the
    forward that ends, from what `main` leaves, on a device honest about each
    group's launch (`TapeLayout.KernelHonest`) and about its products, leaves
    every word inside each buffer, at the size the layout makes it, as the
    tape's first `VFWD_N` operations leave it from the memory the run started
    with, each contraction read as the vendor computes it. -/
theorem fwd_agree (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : VitCalls.Run w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (Vit.vRangeFn 0 Vit.VFWD_N : Prog Slot Lvl Unit)) = .ok obs w')
    (hv : VendorGemmHonest w.vendor) (hvk : VendorKeeps w.vendor)
    (hHon : ∀ p ∈ stepUnits 0 Vit.VFWD_N, ∀ j, p.1 = .kernel j →
      Vit.vLayout.KernelHonest Vit.vApp w j (Vit.vUnitOps p.2) hw) :
    Vit.vLayout.Agree (devMem w'.dev)
      ((Vit.vFused.take Vit.VFWD_N).foldl (fun M op => TOp.denV hw op M) (wordView (devMem w.dev))) :=
  range_agree fwdFacts hw hw32 cfg dataLen outLen w h hr hv hvk hHon

end VitWords

#print axioms VitWords.fwd_agree
