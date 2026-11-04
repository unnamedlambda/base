import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Warp.MlpCifarBodies
import AlgorithmLib.MLHost.AppTrace

/-!
# The Qwen block's runs neither misuse a call nor fault, and leave its tape

Every entry here starts from what `main` leaves (`Drv.App.runTS`). `runBlock`
launches the block's ninety kernels in order on the default stream, then
waits; `runFwd` and `runToK` launch the first 28 or `K`, and `runFwdFused`
the fused forward's 27. An upload asks for the bytes it reads from the
caller's data, a fetch for those it writes to the caller's output. Each
neither misuses a call nor faults; every run of `runBlock` that ends makes
exactly its launches (`check_calls`), and so leaves device memory as the
block's tape (`runBlock_tape`).

A run body is up to 1352 statements over a typestate of 1039 facts; the
checker's acceptance and its pattern are decided by compiled evaluation
(`ship_entry`, `check_calls`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace QwenRunSafe

abbrev Run (d o : Nat) : World → Prop := Contracts.TState.holds
  (MlpCifar.qApp.runTS ++ [.room .data d, .room .out o])

/-- The kernels `runBlock` launches, in order. -/
def steps : List Nat := (MlpCifar.qLaunches 0 MlpCifar.qwenTape).map (·.1)

ship_entry runBlock : MlpCifar.qRunFn under (Run 0 0)

ship_entry runFwd : MlpCifar.qRunFwd under (Run 0 0)

ship_entry runFwdFused : MlpCifar.qRunFused under (Run 0 0)

ship_entry runTo30 : MlpCifar.qRunUpto 30 under (Run 0 0)

ship_entry runTo31 : MlpCifar.qRunUpto 31 under (Run 0 0)

ship_entry runTo34 : MlpCifar.qRunUpto 34 under (Run 0 0)

ship_entry runTo35 : MlpCifar.qRunUpto 35 under (Run 0 0)

ship_entry runTo37 : MlpCifar.qRunUpto 37 under (Run 0 0)

ship_entry runTo45 : MlpCifar.qRunUpto 45 under (Run 0 0)

ship_entry runTo60 : MlpCifar.qRunUpto 60 under (Run 0 0)

ship_entry runTo75 : MlpCifar.qRunUpto 75 under (Run 0 0)

ship_entry fetchOut : MlpCifar.qFetchFn 39 (MlpCifar.DM * 4) under (Run 0 (MlpCifar.DM * 4))

ship_entry uploadDOut : MlpCifar.qUploadFn 40 (MlpCifar.DM * 4) under (Run (MlpCifar.DM * 4) 0)

ship_entry fetchDW2 : MlpCifar.qFetchFn 43 (MlpCifar.DM * MlpCifar.DFF * 4)
  under (Run 0 (MlpCifar.DM * MlpCifar.DFF * 4))

ship_entry uploadW2 : MlpCifar.qUploadFn 12 (MlpCifar.DM * MlpCifar.DFF * 4)
  under (Run (MlpCifar.DM * MlpCifar.DFF * 4) 0)

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- Every run of `runBlock` that ends launches the block's kernels in order on
    the default stream, then waits, the device changed only by those
    calls. -/
theorem runBlock_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.qRunFn : Prog Slot Lvl Unit)) = .ok obs w') :
    (MlpCifar.qApp.runCalls steps).Covers w obs ∧ Check.DevPath w w' := by
  check_calls

theorem bytes_le : MlpCifar.qwenLayout.bytes.all (· ≤ 2 ^ 34) = true := by native_decide
theorem block_ok : ML.kernelsOk MlpCifar.qwenLayout MlpCifar.qApp MlpCifar.qwenUnit steps = true := by native_decide
theorem steps_range : steps = List.range MlpCifar.qwenTape.length := by
  simp only [steps, MlpCifar.qLaunches, Nat.add_zero, List.map_id']
  rw [List.map_fst_zip (by simp)]
theorem block_units : steps.flatMap MlpCifar.qwenUnit = MlpCifar.qwenTape := by
  rw [steps_range]
  exact ML.flatMap_range_take _ _

/-- **The words every run of `runBlock` leaves are the tape's**: from what
    `main` leaves, on a device honest about each launch
    (`TapeLayout.KernelHonest`), every word inside each buffer, at the size
    the program allocates it, is as the block's training step — its forward,
    its backward and its update — leaves it from the memory the run started
    with. -/
theorem runBlock_tape (hw : Kernel.Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.qRunFn : Prog Slot Lvl Unit)) = .ok obs w')
    (hHon : ∀ j ∈ steps, MlpCifar.qwenLayout.KernelHonest MlpCifar.qApp w j (MlpCifar.qwenUnit j) hw) :
    MlpCifar.qwenLayout.Agree (Device.devMem w'.dev)
      (MlpCifar.qwenTape.foldl (fun M op => ML.TOp.denV hw op M) (ML.wordView (Device.devMem w.dev))) := by
  rw [← block_units]
  exact ML.kernels_tape _ _ bytes_le hw hw32 _ _ block_ok (ML.runTS_ready h) hr (runBlock_calls cfg dataLen outLen w h hr) hHon

end QwenRunSafe

#print axioms QwenRunSafe.runBlock_no_misuse
#print axioms QwenRunSafe.runBlock_tape
