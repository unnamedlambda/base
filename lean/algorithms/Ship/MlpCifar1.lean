import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Warp.MlpCifar

/-!
# The CIFAR-10 MLP neither misuses a call nor faults, part 1

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

namespace MlpCifarSafe

abbrev Load : World → Prop := Contracts.TState.holds
  (MlpCifar.app.loadTS ++ [.room .data MlpCifar.HOST_BYTES])

abbrev Run (d o : Nat) : World → Prop := Contracts.TState.holds
  (MlpCifar.app.runTS ++ [.room .data d, .room .out o])

ship_entry main : MlpCifar.loadCode under Load

ship_entry uploadX : MlpCifar.uploadCode MlpCifar.xB (MlpCifar.B * MlpCifar.IN * 4) under ((Run 98304 0))

ship_entry uploadOneHot : MlpCifar.uploadCode MlpCifar.ohB (MlpCifar.B * MlpCifar.C * 4) under ((Run 1024 0))

ship_entry fetchLogits : MlpCifar.fetchCode MlpCifar.logB (MlpCifar.B * MlpCifar.C * 4) under ((Run 0 1024))

ship_entry runFwd1 : MlpCifar.launchSlot 0 under ((Run 0 0))

ship_entry runAct : MlpCifar.launchSlot 1 under ((Run 0 0))

ship_entry runFwd2 : MlpCifar.launchSlot 2 under ((Run 0 0))

ship_entry runDw2 : MlpCifar.launchSlot 4 under ((Run 0 0))

ship_entry runDh : MlpCifar.launchSlot 5 under ((Run 0 0))

ship_entry runAdj : MlpCifar.launchSlot 6 under ((Run 0 0))

ship_entry runDw1 : MlpCifar.launchSlot 7 under ((Run 0 0))

ship_entry runSgd1 : MlpCifar.launchSlot 8 under ((Run 0 0))

ship_entry runSgd2 : MlpCifar.launchSlot 9 under ((Run 0 0))

ship_entry fetchH : MlpCifar.fetchCode MlpCifar.hB 8192 under ((Run 0 8192))

end MlpCifarSafe

#print axioms MlpCifarSafe.main_no_misuse
#print axioms MlpCifarSafe.uploadX_no_misuse
#print axioms MlpCifarSafe.uploadOneHot_no_misuse
#print axioms MlpCifarSafe.fetchLogits_no_misuse
#print axioms MlpCifarSafe.runFwd1_no_misuse
#print axioms MlpCifarSafe.runAct_no_misuse
#print axioms MlpCifarSafe.runFwd2_no_misuse
#print axioms MlpCifarSafe.runDw2_no_misuse
#print axioms MlpCifarSafe.runDh_no_misuse
#print axioms MlpCifarSafe.runAdj_no_misuse
#print axioms MlpCifarSafe.runDw1_no_misuse
#print axioms MlpCifarSafe.runSgd1_no_misuse
#print axioms MlpCifarSafe.runSgd2_no_misuse
#print axioms MlpCifarSafe.fetchH_no_misuse
