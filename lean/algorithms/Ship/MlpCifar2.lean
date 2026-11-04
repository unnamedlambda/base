import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Warp.MlpCifarBodies
import AlgorithmLib.MLHost.AppTrace

/-!
# The CIFAR-10 MLP neither misuses a call nor faults, part 2

`main` brings the CUDA driver up, makes the thirteen allocations, loads the
ten kernels, writes their parameters, makes cuBLAS's handle and uploads both
weight matrices from the caller's data, in the driver's standard layout
(`Drv.App`); every other entry starts from what it leaves (`Drv.App.runTS`):
the context current, every allocation live with its pointer in its slot,
every function looked up and its parameters written, the cuBLAS handle on
the default stream. An upload asks for the bytes it reads from the caller's
data, a fetch for those it writes to the caller's output.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace MlpCifarSafe2

abbrev Run (d o : Nat) : World → Prop := Contracts.TState.holds
  (MlpCifar.app.runTS ++ [.room .data d, .room .out o])

ship_entry fetchZ1 : MlpCifar.fetchCode MlpCifar.z1B 8192 under ((Run 0 8192))

ship_entry fetchDh : MlpCifar.fetchCode MlpCifar.dhB 8192 under ((Run 0 8192))

ship_entry fetchAdj : MlpCifar.fetchCode MlpCifar.adjB 8192 under ((Run 0 8192))

ship_entry fetchDw1 : MlpCifar.fetchCode MlpCifar.dw1B 3145728 under ((Run 0 3145728))

ship_entry fetchDw2 : MlpCifar.fetchCode MlpCifar.dw2B 32768 under ((Run 0 32768))

ship_entry fetchW1 : MlpCifar.fetchCode MlpCifar.w1B 3145728 under ((Run 0 3145728))

ship_entry fetchW2 : MlpCifar.fetchCode MlpCifar.w2B 32768 under ((Run 0 32768))

ship_entry runFwd : MlpCifar.runSeq MlpCifar.fwdSteps under ((Run 0 0))

ship_entry runBwd : MlpCifar.runSeq MlpCifar.bwdSteps under ((Run 0 0))

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- Every run of the shipped forward pass that ends launches the forward
    kernels in order on the default stream, then waits, the device changed
    only by those calls. -/
theorem runFwd_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.fwdSteps : Prog Slot Lvl Unit)) = .ok obs w') :
    (MlpCifar.app.runCalls MlpCifar.fwdSteps).Covers w obs ∧ Check.DevPath w w' := by
  check_calls

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- Every run of the shipped backward pass that ends launches the backward
    kernels in order on the default stream, then waits, the device changed
    only by those calls. -/
theorem runBwd_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.bwdSteps : Prog Slot Lvl Unit)) = .ok obs w') :
    (MlpCifar.app.runCalls MlpCifar.bwdSteps).Covers w obs ∧ Check.DevPath w w' := by
  check_calls

theorem bytes_le : MlpCifar.cifarLayout.bytes.all (· ≤ 2 ^ 34) = true := by native_decide
theorem fwd_ok : ML.kernelsOk MlpCifar.cifarLayout MlpCifar.app MlpCifar.cifarUnit MlpCifar.fwdSteps = true := by
  native_decide
theorem bwd_ok : ML.kernelsOk MlpCifar.cifarLayout MlpCifar.app MlpCifar.cifarUnit MlpCifar.bwdSteps = true := by
  native_decide
theorem fwd_units : MlpCifar.fwdSteps.flatMap MlpCifar.cifarUnit = MlpCifar.cifarTape.take 4 := rfl
theorem bwd_units : MlpCifar.bwdSteps.flatMap MlpCifar.cifarUnit = MlpCifar.cifarTape.drop 4 := rfl

/-- **The words every run of the shipped forward pass leaves are the
    tape's**: from what `main` leaves, on a device honest about each launch
    (`TapeLayout.KernelHonest`), every word inside each buffer, at the size
    the program allocates it, is as the tape's first four operations leave
    it from the memory the run started with. -/
theorem runFwd_tape (hw : Kernel.Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.fwdSteps : Prog Slot Lvl Unit)) = .ok obs w')
    (hHon : ∀ j ∈ MlpCifar.fwdSteps, MlpCifar.cifarLayout.KernelHonest MlpCifar.app w j (MlpCifar.cifarUnit j) hw) :
    MlpCifar.cifarLayout.Agree (Device.devMem w'.dev)
      ((MlpCifar.cifarTape.take 4).foldl (fun M op => ML.TOp.denV hw op M) (ML.wordView (Device.devMem w.dev))) := by
  rw [← fwd_units]
  exact ML.kernels_tape _ _ bytes_le hw hw32 _ _ fwd_ok (ML.runTS_ready h) hr (runFwd_calls cfg dataLen outLen w h hr) hHon

/-- **The words every run of the shipped backward pass leaves are the
    tape's**: from what `main` leaves, on a device honest about each launch,
    every word inside each buffer, at the size the program allocates it, is
    as the tape's operations past the forward — the backward and both
    optimiser steps — leave it from the memory the run started with. -/
theorem runBwd_tape (hw : Kernel.Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runSeq MlpCifar.bwdSteps : Prog Slot Lvl Unit)) = .ok obs w')
    (hHon : ∀ j ∈ MlpCifar.bwdSteps, MlpCifar.cifarLayout.KernelHonest MlpCifar.app w j (MlpCifar.cifarUnit j) hw) :
    MlpCifar.cifarLayout.Agree (Device.devMem w'.dev)
      ((MlpCifar.cifarTape.drop 4).foldl (fun M op => ML.TOp.denV hw op M) (ML.wordView (Device.devMem w.dev))) := by
  rw [← bwd_units]
  exact ML.kernels_tape _ _ bytes_le hw hw32 _ _ bwd_ok (ML.runTS_ready h) hr (runBwd_calls cfg dataLen outLen w h hr) hHon

ship_entry runSoftmax : MlpCifar.launchSlot 3 under ((Run 0 0))

ship_entry uploadBias : MlpCifar.uploadCode MlpCifar.biasB (MlpCifar.C * 4) under ((Run 128 0))

ship_entry fetchDlog : MlpCifar.fetchCode MlpCifar.dlogB (MlpCifar.B * MlpCifar.C * 4) under ((Run 0 1024))

ship_entry runFwdBlas : MlpCifar.runMixed MlpCifar.fwdBlasSteps under ((Run 0 0))

ship_entry runBwdBlas : MlpCifar.runMixed MlpCifar.bwdBlasSteps under ((Run 0 0))

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- Every run of the shipped cuBLAS forward pass that ends makes its steps'
    calls in order, launches and products, then waits, the device changed
    only by those calls. -/
theorem runFwdBlas_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runMixed MlpCifar.fwdBlasSteps : Prog Slot Lvl Unit)) = .ok obs w') :
    (MlpCifar.app.callsWith (MlpCifar.app.stridedRec (drvHandle kBlas 0) [0x3f800000, 0]) MlpCifar.fwdBlasSteps).Covers w obs ∧
      Check.DevPath w w' := by
  check_calls

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- …and of the cuBLAS backward pass. -/
theorem runBwdBlas_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runMixed MlpCifar.bwdBlasSteps : Prog Slot Lvl Unit)) = .ok obs w') :
    (MlpCifar.app.callsWith (MlpCifar.app.stridedRec (drvHandle kBlas 0) [0x3f800000, 0]) MlpCifar.bwdBlasSteps).Covers w obs ∧
      Check.DevPath w w' := by
  check_calls

/-- The cuBLAS build's steps `lo..hi`, each with its operation. -/
def blasOps (lo hi : Nat) : List (ML.DevCall × List ML.TOp) :=
  (MlpCifar.blasRange lo hi).map fun (op, i) => (ML.DevCall.ofStep (MlpCifar.blasStep op i), [op])

theorem fwdBlas_ok : ML.stepsOk MlpCifar.cifarLayout MlpCifar.app (blasOps 0 4) = true := by native_decide
theorem bwdBlas_ok : ML.stepsOk MlpCifar.cifarLayout MlpCifar.app (blasOps 4 10) = true := by native_decide
theorem fwdBlas_units : ML.unitsOkB MlpCifar.cifarLayout (blasOps 0 4) = true := by native_decide
theorem bwdBlas_units : ML.unitsOkB MlpCifar.cifarLayout (blasOps 4 10) = true := by native_decide
theorem fwdBlas_tape : (blasOps 0 4).flatMap Prod.snd = MlpCifar.cifarBlasTape.take 4 := rfl
theorem bwdBlas_tape : (blasOps 4 10).flatMap Prod.snd = MlpCifar.cifarBlasTape.drop 4 := rfl

/-- **The words every run of the shipped cuBLAS forward pass leaves are the
    tape's**: from what `main` leaves, on a device honest about each launch
    (`TapeLayout.KernelHonest`) and each product (`VendorGemmHonest`), every
    word inside each buffer, at the size the program allocates it, is as the
    first four operations of the tape under the cuBLAS schedule leave it from
    the memory the run started with, each contraction read as cuBLAS computes
    it. -/
theorem runFwdBlas_tape (hw : Kernel.Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runMixed MlpCifar.fwdBlasSteps : Prog Slot Lvl Unit)) = .ok obs w')
    (hv : ML.VendorGemmHonest w.vendor) (hvk : Contracts.VendorKeeps w.vendor)
    (hHon : ∀ p ∈ blasOps 0 4, ∀ j, p.1 = .kernel j → MlpCifar.cifarLayout.KernelHonest MlpCifar.app w j p.2 hw) :
    MlpCifar.cifarLayout.Agree (Device.devMem w'.dev)
      ((MlpCifar.cifarBlasTape.take 4).foldl (fun M op => ML.TOp.denV hw op M) (ML.wordView (Device.devMem w.dev))) := by
  rw [← fwdBlas_tape]
  exact ML.calls_tape _ _ rfl bytes_le hw hw32 _ _ (by rw [blasOps, List.map_map]; rfl) fwdBlas_ok
    (ML.unitsOk_of fwdBlas_units) (ML.runTS_ready h) hr (runFwdBlas_calls cfg dataLen outLen w h hr) hv hvk hHon

/-- **…and of the shipped cuBLAS backward pass**: the tape's operations past
    the forward, each contraction read as cuBLAS computes it. -/
theorem runBwdBlas_tape (hw : Kernel.Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.runMixed MlpCifar.bwdBlasSteps : Prog Slot Lvl Unit)) = .ok obs w')
    (hv : ML.VendorGemmHonest w.vendor) (hvk : Contracts.VendorKeeps w.vendor)
    (hHon : ∀ p ∈ blasOps 4 10, ∀ j, p.1 = .kernel j → MlpCifar.cifarLayout.KernelHonest MlpCifar.app w j p.2 hw) :
    MlpCifar.cifarLayout.Agree (Device.devMem w'.dev)
      ((MlpCifar.cifarBlasTape.drop 4).foldl (fun M op => ML.TOp.denV hw op M) (ML.wordView (Device.devMem w.dev))) := by
  rw [← bwdBlas_tape]
  exact ML.calls_tape _ _ rfl bytes_le hw hw32 _ _ (by rw [blasOps, List.map_map]; rfl) bwdBlas_ok
    (ML.unitsOk_of bwdBlas_units) (ML.runTS_ready h) hr (runBwdBlas_calls cfg dataLen outLen w h hr) hv hvk hHon

end MlpCifarSafe2

#print axioms MlpCifarSafe2.fetchZ1_no_misuse
#print axioms MlpCifarSafe2.fetchDh_no_misuse
#print axioms MlpCifarSafe2.fetchAdj_no_misuse
#print axioms MlpCifarSafe2.fetchDw1_no_misuse
#print axioms MlpCifarSafe2.fetchDw2_no_misuse
#print axioms MlpCifarSafe2.fetchW1_no_misuse
#print axioms MlpCifarSafe2.fetchW2_no_misuse
#print axioms MlpCifarSafe2.runFwd_no_misuse
#print axioms MlpCifarSafe2.runBwd_no_misuse
#print axioms MlpCifarSafe2.runSoftmax_no_misuse
#print axioms MlpCifarSafe2.uploadBias_no_misuse
#print axioms MlpCifarSafe2.fetchDlog_no_misuse
#print axioms MlpCifarSafe2.runFwdBlas_no_misuse
#print axioms MlpCifarSafe2.runBwdBlas_no_misuse

#print axioms MlpCifarSafe2.runFwd_tape
#print axioms MlpCifarSafe2.runBwd_tape
#print axioms MlpCifarSafe2.runFwdBlas_tape
#print axioms MlpCifarSafe2.runBwdBlas_tape
