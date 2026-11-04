import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Warp.MlpCifar

/-!
# The Qwen block's `main` and `capture` neither misuse a call nor fault

`main` brings the CUDA driver up, makes every allocation, loads the 117
kernels, writes their parameters and uploads the thirteen inputs from the
caller's data, in the driver's standard layout (`Drv.App`). `capture` starts
from what `main` leaves (`Drv.App.runTS`, with no stream, graph or
executable graph made yet): it makes a stream, captures the block's ninety
launches on it, instantiates the graph, uploads it and waits. What the
capture and the replays after it leave is `QwenReplaySafe.capture_replay_tape`.

The checker's acceptance of both bodies is decided by compiled evaluation
(`ship_entry`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace QwenLoadSafe

abbrev Load : World → Prop := Contracts.TState.holds
  (MlpCifar.qApp.loadTS ++ [.room .data MlpCifar.QHOST_BYTES])

abbrev Capture : World → Prop := Contracts.TState.holds
  (MlpCifar.qApp.runTS ++ [.streams [], .ngraphs 0, .nexecs 0])

ship_entry main : MlpCifar.qLoadFn under Load

ship_entry capture : MlpCifar.qCaptureFn under Capture

end QwenLoadSafe

#print axioms QwenLoadSafe.main_no_misuse
#print axioms QwenLoadSafe.capture_no_misuse
