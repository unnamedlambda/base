import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Warp.MlpCifar
import AlgorithmLib.ML.Launch.AppTrace

/-!
# The Qwen block's replays neither misuse a call nor fault, and are its plan

`capture` makes a stream, captures the block's launches on it, instantiates
the graph and uploads it, and waits. A replay entry starts from what that
leaves: stream 0 live and executable graph 0, their handles in their slots,
and the host having waited for everything. It replays the graph `k` times on
the stream and waits for it.

Each replay entry neither misuses a call nor faults; every run of it that
ends makes those calls and no others (`check_calls`), and the device writes
those calls are, read as a plan, are `k` replays of the captured pipeline
(`MlpCifar.qwen_replay_realises`).

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
    stream, then waits for the stream. -/
theorem replay_calls (k : Nat) (hk : k ∈ [1, 2, 4]) (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Replay w) {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.qReplayFn k : Prog Slot Lvl Unit)) = .ok obs w') :
    (Drv.App.replayCalls (drvHandle kExec 0) (drvHandle kStream 0) k).Covers w obs := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl | rfl <;> check_calls

/-- **The device writes every run of a replay entry makes are `k` replays of
    the captured pipeline**, read from its calls. -/
theorem replay_realises (k : Nat) (hk : k ∈ [1, 2, 4]) (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Replay w) {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.qReplayFn k : Prog Slot Lvl Unit)) = .ok obs w') :
    ∃ cs, Check.callsOf obs = (Check.callsOf w.obs).reverse ++ cs ∧
      ML.planOf? MlpCifar.qTable MlpCifar.qDeclared none (ML.appLaunchesOf MlpCifar.qApp cs) =
        some (MlpCifar.qReplayPlan k) := by
  obtain ⟨cs, e, hc⟩ := replay_calls k hk cfg dataLen outLen w h hr
  exact ⟨cs, e, by rw [ML.launchesOf_replayCalls hc]; exact MlpCifar.qwen_replay_realises k⟩

end QwenReplaySafe

#print axioms QwenReplaySafe.replay_realises
