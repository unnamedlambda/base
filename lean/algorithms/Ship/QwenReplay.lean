import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Warp.MlpCifar
import AlgorithmLib.ML.Launch.AppTrace
import AlgorithmLib.ML.Model.TapeRun
import Ship.QwenLoad
import Ship.QwenRun

/-!
# The Qwen block's replays neither misuse a call nor fault, and leave its tape

`capture` makes a stream, captures the block's launches on it, instantiates
the graph and uploads it, and waits. A replay entry starts from what that
leaves: stream 0 live and executable graph 0, their handles in their slots,
and the host having waited for everything. It replays the graph `k` times on
the stream and waits for it.

Each replay entry neither misuses a call nor faults; every run of it that
ends makes those calls and no others (`check_calls`), and a capture followed
by `k` replays leaves device memory as `k` steps of the block's tape
(`capture_replay_tape`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace QwenReplaySafe

/-- What `capture` leaves that a replay reads. -/
abbrev Replay : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part .capturing false, .part .idle true, .part .cuda true,
   .oracles, .room .arena MlpCifar.qApp.memSize, .streams [true],
   .cell (Drv.App.at_ MlpCifar.QSTREAM_OFF) (drvHandle kStream 0),
   .exec 0, .cell (Drv.App.at_ MlpCifar.QEXEC_OFF) (drvHandle kExec 0), .readyAll hostParty]

ship_entry replay1 : MlpCifar.qReplayFn 1 under Replay

ship_entry replay2 : MlpCifar.qReplayFn 2 under Replay

ship_entry replay4 : MlpCifar.qReplayFn 4 under Replay

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- Every run of a replay entry that ends replays the graph `k` times on the
    stream, then waits for the stream, the device changed only by those
    calls. -/
theorem replay_calls (k : Nat) (hk : k ∈ [1, 2, 4]) (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Replay w) {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.qReplayFn k : Prog Slot Lvl Unit)) = .ok obs w') :
    (Drv.App.replayCalls (drvHandle kExec 0) (drvHandle kStream 0) k).Covers w obs ∧ Check.DevPath w w' := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl | rfl <;> check_calls

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- **Every run of `capture` that ends makes the stream, captures the block's
    launches on it, ends the capture, instantiates and uploads the graph,
    and waits**, the device changed only by those calls. -/
theorem capture_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : QwenLoadSafe.Capture w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.qCaptureFn : Prog Slot Lvl Unit)) = .ok obs w') :
    (Check.Tr.lit (ML.captureRecs MlpCifar.qApp QwenRunSafe.steps (Drv.App.at_ MlpCifar.QSTREAM_OFF)
      (Drv.App.at_ MlpCifar.QGRAPH_OFF) (Drv.App.at_ MlpCifar.QEXEC_OFF))).Covers w obs ∧ Check.DevPath w w' := by
  check_calls

set_option maxRecDepth 200000 in
set_option maxHeartbeats 0 in
/-- **What `capture` leaves is what a replay starts from** (`Replay`). -/
theorem capture_reaches (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : QwenLoadSafe.Capture w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) (emit (MlpCifar.qCaptureFn : Prog Slot Lvl Unit))
      { ok := fun _ w' => Replay w', faultOk := false } := by
  check_reach

/-- **The words a capture and `k` replays leave are the block's training
    step `k` times over**: every run of `capture` that ends, from what `main`
    leaves with no stream or graph yet, then every run of a replay of it `k`
    times that ends, on a device honest about each launch the graph holds
    (`TapeLayout.KernelHonest`), leaves every word inside each buffer, at the
    size the program allocates it, as `k` training steps of the block's tape
    leave it from the memory the capture started with. -/
theorem capture_replay_tape (k : Nat) (hk : k ∈ [1, 2, 4]) (hw : Kernel.Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w0 : World) (h : QwenLoadSafe.Capture w0)
    {obs1 : List Obs} {w1 : World}
    (hr1 : Sem.run cfg (runArgs dataLen outLen) w0 (emit (MlpCifar.qCaptureFn : Prog Slot Lvl Unit)) = .ok obs1 w1)
    (cfg' : Cfg) (dataLen' outLen' : UInt64) {obs2 : List Obs} {w2 : World}
    (hr2 : Sem.run cfg' (runArgs dataLen' outLen') w1 (emit (MlpCifar.qReplayFn k : Prog Slot Lvl Unit)) = .ok obs2 w2)
    (hHon : ∀ j ∈ QwenRunSafe.steps, MlpCifar.qwenLayout.KernelHonest MlpCifar.qApp w1 j (MlpCifar.qwenUnit j) hw) :
    MlpCifar.qwenLayout.Agree (Device.devMem w2.dev)
      (Nat.repeat (fun M => MlpCifar.qwenTape.foldl (fun M op => ML.TOp.denV hw op M) M) k
        (ML.wordView (Device.devMem w0.dev))) := by
  obtain ⟨_, hrep⟩ := ML.triple_run (capture_reaches cfg dataLen outLen w0 h) hr1
  have hs : w0.dev.streams.size = 0 := by
    have : w0.dev.streams.toList = [] := h (.streams []) (by simp)
    simpa using congrArg List.length this
  rw [← QwenRunSafe.block_units]
  exact ML.capture_replay_tape _ _ QwenRunSafe.bytes_le hw hw32 _ _ QwenRunSafe.block_ok k (ML.runTS_ready h) hs
    (h (.ngraphs 0) (by simp)) (h (.nexecs 0) (by simp)) hr1 (capture_calls cfg dataLen outLen w0 h hr1) hr2
    (replay_calls k hk cfg' dataLen' outLen' w1 hrep hr2) hHon

end QwenReplaySafe

#print axioms QwenReplaySafe.capture_replay_tape
