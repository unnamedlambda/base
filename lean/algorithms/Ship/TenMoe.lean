import AlgorithmLib.Proof.Typestate
import Warp.MlpCifar

/-!
# The mixture-of-experts dispatch neither misuses a call nor faults

`main` makes the 123 buffers, storing each one's id in the binding table, and
uploads the shared inputs and every expert's weights from the caller's data;
every other entry starts from what it leaves: a live context, the ids where it
stored them, each buffer's size (`devBuf`), and the seventeen kernels' text in
their slots. The six ids `bindExperts` rewrites are left out, so each entry is
proven whichever experts were chosen: a launch over an id that names no buffer
answers an error. `bindExperts` calls nothing: it needs only the arena and the
caller's choice, read through the length it is handed (`roomArg`); an upload asks for the bytes it reads from the
caller's data, a fetch for those it writes to the caller's output.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace MoeSafe

abbrev Load : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena MlpCifar.MMEM_SIZE,
   .room .data MlpCifar.MHOST_BYTES]

abbrev Run (d o : Nat) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena MlpCifar.MMEM_SIZE, .room .data d, .room .out o,
   .cell (regionBase .arena + UInt64.ofNat ContextSlots.cuda) cudaCtx,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 0)) 4294967296,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 2)) 12884901890,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 10)) 47244640266,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 12)) 55834574860,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 14)) 64424509454,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 16)) 73014444048,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 18)) 81604378642,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 20)) 90194313236,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 22)) 98784247830,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 24)) 107374182424,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 26)) 115964117018,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 28)) 124554051612,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 30)) 133143986206,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 32)) 141733920800,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 34)) 150323855394,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 36)) 158913789988,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 38)) 167503724582,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 40)) 176093659176,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 42)) 184683593770,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 44)) 193273528364,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 46)) 201863462958,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 48)) 210453397552,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 50)) 219043332146,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 52)) 227633266740,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 54)) 236223201334,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 56)) 244813135928,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 58)) 253403070522,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 60)) 261993005116,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 62)) 270582939710,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 64)) 279172874304,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 66)) 287762808898,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 68)) 296352743492,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 70)) 304942678086,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 72)) 313532612680,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 74)) 322122547274,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 76)) 330712481868,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 78)) 339302416462,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 80)) 347892351056,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 82)) 356482285650,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 84)) 365072220244,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 86)) 373662154838,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 88)) 382252089432,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 90)) 390842024026,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 92)) 399431958620,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 94)) 408021893214,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 96)) 416611827808,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 98)) 425201762402,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 100)) 433791696996,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 102)) 442381631590,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 104)) 450971566184,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 106)) 459561500778,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 108)) 468151435372,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 110)) 476741369966,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 112)) 485331304560,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 114)) 493921239154,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 116)) 502511173748,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 118)) 511101108342,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 120)) 519691042936,
   .cell (regionBase .arena + UInt64.ofNat (MlpCifar.mBindOff 122)) 122,
   .devBuf 0 128, .devBuf 1 16384, .devBuf 2 512, .devBuf 3 128, .devBuf 4 128, .devBuf 5 128, .devBuf 6 128, .devBuf 7 128, .devBuf 8 128, .devBuf 9 128, .devBuf 10 131072, .devBuf 11 131072, .devBuf 12 131072, .devBuf 13 131072, .devBuf 14 131072, .devBuf 15 131072, .devBuf 16 131072, .devBuf 17 131072, .devBuf 18 131072, .devBuf 19 131072, .devBuf 20 131072, .devBuf 21 131072, .devBuf 22 131072, .devBuf 23 131072, .devBuf 24 131072, .devBuf 25 131072, .devBuf 26 131072, .devBuf 27 131072, .devBuf 28 131072, .devBuf 29 131072, .devBuf 30 131072, .devBuf 31 131072, .devBuf 32 131072, .devBuf 33 131072, .devBuf 34 131072, .devBuf 35 131072, .devBuf 36 131072, .devBuf 37 131072, .devBuf 38 131072, .devBuf 39 131072, .devBuf 40 131072, .devBuf 41 131072, .devBuf 42 131072, .devBuf 43 131072, .devBuf 44 131072, .devBuf 45 131072, .devBuf 46 131072, .devBuf 47 131072, .devBuf 48 131072, .devBuf 49 131072, .devBuf 50 131072, .devBuf 51 131072, .devBuf 52 131072, .devBuf 53 131072, .devBuf 54 131072, .devBuf 55 131072, .devBuf 56 131072, .devBuf 57 131072, .devBuf 58 131072, .devBuf 59 131072, .devBuf 60 131072, .devBuf 61 131072, .devBuf 62 131072, .devBuf 63 131072, .devBuf 64 131072, .devBuf 65 131072, .devBuf 66 131072, .devBuf 67 131072, .devBuf 68 131072, .devBuf 69 131072, .devBuf 70 131072, .devBuf 71 131072, .devBuf 72 131072, .devBuf 73 131072, .devBuf 74 131072, .devBuf 75 131072, .devBuf 76 131072, .devBuf 77 131072, .devBuf 78 131072, .devBuf 79 131072, .devBuf 80 131072, .devBuf 81 131072, .devBuf 82 131072, .devBuf 83 131072, .devBuf 84 131072, .devBuf 85 131072, .devBuf 86 131072, .devBuf 87 131072, .devBuf 88 131072, .devBuf 89 131072, .devBuf 90 131072, .devBuf 91 131072, .devBuf 92 131072, .devBuf 93 131072, .devBuf 94 131072, .devBuf 95 131072, .devBuf 96 131072, .devBuf 97 131072, .devBuf 98 131072, .devBuf 99 131072, .devBuf 100 131072, .devBuf 101 131072, .devBuf 102 131072, .devBuf 103 131072, .devBuf 104 131072, .devBuf 105 131072, .devBuf 106 128, .devBuf 107 128, .devBuf 108 128, .devBuf 109 128, .devBuf 110 128, .devBuf 111 128, .devBuf 112 1024, .devBuf 113 1024, .devBuf 114 1024, .devBuf 115 512, .devBuf 116 512, .devBuf 117 1024, .devBuf 118 1024, .devBuf 119 1024, .devBuf 120 512, .devBuf 121 512, .devBuf 122 512,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 0)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 1)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 2)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 3)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 4)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 5)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 6)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 7)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 8)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 9)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 10)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 11)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 12)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 13)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 14)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 15)) MlpCifar.QSLOT,
   .cstrIn (regionBase .arena + UInt64.ofNat (MlpCifar.qSlotOff 16)) MlpCifar.QSLOT]

/-- The arena's size, which the kernel computes from the layout. -/
theorem mem_size : MlpCifar.MMEM_SIZE = 558092 := by decide +kernel

/-- What `bindExperts` reads and writes: the binding table in the arena, and
    the caller's choice through the length it hands over. It calls nothing.
    The arena's size is its number: an address over the caller's choice is
    bounded by arithmetic, which reads a size only as a number. -/
abbrev Bind (dl : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .room .arena 558092, .roomArg .data dl]

theorem main_fine : Fine (emitGo (MlpCifar.mLoadFn : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem main_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Load w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.mLoadFn : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.mLoadFn : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry Load (runArgs dataLen outLen) (by prog_vc Load) main_fine rfl h m

theorem runRouter_fine : Fine (emitGo ((MlpCifar.mRunRange 0 MlpCifar.mRouterTape.length) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runRouter_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mRunRange 0 MlpCifar.mRouterTape.length) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mRunRange 0 MlpCifar.mRouterTape.length) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runRouter_fine rfl h m

theorem fetchGate_fine : Fine (emitGo ((MlpCifar.mFetchFn MlpCifar.MGATE (MlpCifar.NE * 4)) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchGate_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 (MlpCifar.NE * 4)) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mFetchFn MlpCifar.MGATE (MlpCifar.NE * 4)) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mFetchFn MlpCifar.MGATE (MlpCifar.NE * 4)) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 (MlpCifar.NE * 4)) (runArgs dataLen outLen) (by prog_vc (Run 0 (MlpCifar.NE * 4))) fetchGate_fine rfl h m

theorem bindExperts_fine : Fine (emitGo (MlpCifar.mBindExperts : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem bindExperts_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.part .frozen false, .room .arena MlpCifar.MMEM_SIZE, .roomArg .data dataLen] w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.mBindExperts : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.mBindExperts : Prog Slot Lvl Unit)) ≠ .fault m := by
  rw [mem_size] at h
  exact sound_of_wp_entry (Bind dataLen) (runArgs dataLen outLen) (by prog_vc (Bind dataLen)) bindExperts_fine rfl h m

theorem uploadGates_fine : Fine (emitGo ((MlpCifar.mUploadFn 3 128) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem uploadGates_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 128 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mUploadFn 3 128) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mUploadFn 3 128) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 128 0) (runArgs dataLen outLen) (by prog_vc (Run 128 0)) uploadGates_fine rfl h m

theorem runExperts_fine : Fine (emitGo ((MlpCifar.mRunRange MlpCifar.mRouterTape.length MlpCifar.moeTape.length) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem runExperts_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mRunRange MlpCifar.mRouterTape.length MlpCifar.moeTape.length) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mRunRange MlpCifar.mRouterTape.length MlpCifar.moeTape.length) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 0) (runArgs dataLen outLen) (by prog_vc (Run 0 0)) runExperts_fine rfl h m

theorem fetchOut_fine : Fine (emitGo ((MlpCifar.mFetchFn MlpCifar.MOUT (MlpCifar.MD * 4)) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetchOut_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 0 (MlpCifar.MD * 4)) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mFetchFn MlpCifar.MOUT (MlpCifar.MD * 4)) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mFetchFn MlpCifar.MOUT (MlpCifar.MD * 4)) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run 0 (MlpCifar.MD * 4)) (runArgs dataLen outLen) (by prog_vc (Run 0 (MlpCifar.MD * 4))) fetchOut_fine rfl h m

end MoeSafe

#print axioms MoeSafe.main_no_misuse
#print axioms MoeSafe.runRouter_no_misuse
#print axioms MoeSafe.fetchGate_no_misuse
#print axioms MoeSafe.bindExperts_no_misuse
#print axioms MoeSafe.uploadGates_no_misuse
#print axioms MoeSafe.runExperts_no_misuse
#print axioms MoeSafe.fetchOut_no_misuse
