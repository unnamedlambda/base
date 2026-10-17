import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import Warp.MlpCifar

/-!
# The CIFAR-10 MLP neither misuses a call nor faults, part 2

`main` makes the thirteen buffers, storing each one's id in the binding
table, and uploads both weight matrices from the caller's data; every other
entry starts from what it leaves: a live context, the ids where it stored them,
each buffer's size (`devBuf`), and the ten kernels' text in their slots. An
upload asks for the bytes it reads from the caller's data, a fetch for those
it writes to the caller's output.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace MlpCifarSafe2

abbrev Run (d o : Nat) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena MlpCifar.MEM_SIZE, .room .data d, .room .out o,
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 0)) 4294967296,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 2)) 12884901890,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 4)) 21474836484,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 6)) 30064771078,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 8)) 38654705672,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 10)) 47244640266,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.bindOff 12)) 12,
   .devBuf 0 98304, .devBuf 1 3145728, .devBuf 2 32768, .devBuf 3 128, .devBuf 4 1024, .devBuf 5 8192, .devBuf 6 8192, .devBuf 7 1024, .devBuf 8 1024, .devBuf 9 32768, .devBuf 10 8192, .devBuf 11 8192, .devBuf 12 3145728,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 0)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 1)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 2)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 3)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 4)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 5)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 6)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 7)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 8)) MlpCifar.SLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.slotOff 9)) MlpCifar.SLOT]

ship_entry fetchZ1 : MlpCifar.fetchCode MlpCifar.z1B 8192 under ((Run 0 8192))

ship_entry fetchDh : MlpCifar.fetchCode MlpCifar.dhB 8192 under ((Run 0 8192))

ship_entry fetchAdj : MlpCifar.fetchCode MlpCifar.adjB 8192 under ((Run 0 8192))

ship_entry fetchDw1 : MlpCifar.fetchCode MlpCifar.dw1B 3145728 under ((Run 0 3145728))

ship_entry fetchDw2 : MlpCifar.fetchCode MlpCifar.dw2B 32768 under ((Run 0 32768))

ship_entry fetchW1 : MlpCifar.fetchCode MlpCifar.w1B 3145728 under ((Run 0 3145728))

ship_entry fetchW2 : MlpCifar.fetchCode MlpCifar.w2B 32768 under ((Run 0 32768))

ship_entry runFwd : MlpCifar.runSeq MlpCifar.fwdSteps under ((Run 0 0))

ship_entry runBwd : MlpCifar.runSeq MlpCifar.bwdSteps under ((Run 0 0))

ship_entry runSoftmax : MlpCifar.launchSlot 3 MlpCifar.B under ((Run 0 0))

ship_entry uploadBias : MlpCifar.uploadCode MlpCifar.biasB (MlpCifar.C * 4) under ((Run 128 0))

ship_entry fetchDlog : MlpCifar.fetchCode MlpCifar.dlogB (MlpCifar.B * MlpCifar.C * 4) under ((Run 0 1024))

ship_entry runFwdBlas : MlpCifar.runMixed MlpCifar.fwdBlasSteps under ((Run 0 0))

ship_entry runBwdBlas : MlpCifar.runMixed MlpCifar.bwdBlasSteps under ((Run 0 0))

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
