import AlgorithmLib.Proof.Calls
import Host.StaticChecks
import Bench.CudaSoftmaxPersist
import Bench.CudaGemvPersist
import Bench.CudaRmsNormPersist
import Bench.CudaDecodeAttention

/-!
# The CUDA benchmarks neither misuse a call nor fault

Each entry point of each persistent CUDA benchmark, proven by the condition
generator for any lengths of data and output the caller hands over. The sizes
these programs work with are read from their input, so the proofs follow them
as terms: `roomArg` says the data and output regions hold the lengths the
caller passed, and each branch that compares a size to them carries what it
found into the arm it takes. `main` starts from memory alone; the others start
from what `main` leaves: a live context whose cell holds it, every device
access the default stream's, and each kernel's text in the arena.

The cuBLAS entries, gemv's `infer` and decode-attention's `core`, start
from what `main` leaves when it makes its buffers: the dimensions it recorded,
within the bounds it checks before making any, each buffer's id where it
stored it, and each buffer's size (`devBuf`), which every later call keeps.
Each product then fits its buffers.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CudaBenchSafe

-- CudaSoftmaxPersist

abbrev CudaSoftmaxPersistLoad (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena CudaSoftmaxPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol, .cstrIn (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.PTX_SOURCE_OFF) (CudaSoftmaxPersist.BIND_K1_OFF - CudaSoftmaxPersist.PTX_SOURCE_OFF), .cstrIn (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.NAME_BLOCK_REDUCE) (16), .cstrIn (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.NAME_GLOBAL_REDUCE) (16), .cstrIn (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.NAME_NORMALIZE) (16), .cstrIn (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.NAME_SMALL_SOFTMAX) (16)]

abbrev CudaSoftmaxPersistRun (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena CudaSoftmaxPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.CTX_OFF) cudaCtx, .cstrIn (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.PTX_SOURCE_OFF) (CudaSoftmaxPersist.BIND_K1_OFF - CudaSoftmaxPersist.PTX_SOURCE_OFF), .cstrIn (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.NAME_BLOCK_REDUCE) (16), .cstrIn (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.NAME_GLOBAL_REDUCE) (16), .cstrIn (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.NAME_NORMALIZE) (16), .cstrIn (regionBase .arena + UInt64.ofNat CudaSoftmaxPersist.NAME_SMALL_SOFTMAX) (16)]

theorem CudaSoftmaxPersist_load_fine : Fine (emitGo (CudaSoftmaxPersist.loadCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaSoftmaxPersist_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaSoftmaxPersistLoad dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.loadCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.loadCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (CudaSoftmaxPersistLoad dataLen outLen) (runArgs dataLen outLen) (by prog_vc (CudaSoftmaxPersistLoad dataLen outLen))
    CudaSoftmaxPersist_load_fine rfl h m

theorem CudaSoftmaxPersist_prep_fine : Fine (emitGo (CudaSoftmaxPersist.prepCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaSoftmaxPersist_prep_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaSoftmaxPersistRun dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.prepCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.prepCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (CudaSoftmaxPersistRun dataLen outLen) (runArgs dataLen outLen) (by prog_vc (CudaSoftmaxPersistRun dataLen outLen))
    CudaSoftmaxPersist_prep_fine rfl h m

theorem CudaSoftmaxPersist_core_fine : Fine (emitGo (CudaSoftmaxPersist.coreCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaSoftmaxPersist_core_triple (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaSoftmaxPersistRun dataLen outLen w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) (emit (CudaSoftmaxPersist.coreCode : Prog Slot Lvl Unit))
      { ok := fun _ w' => CudaSoftmaxPersistRun dataLen outLen w', faultOk := false } :=
  triple_of_wp_entry (CudaSoftmaxPersistRun dataLen outLen) (CudaSoftmaxPersistRun dataLen outLen) (runArgs dataLen outLen)
    (by prog_vc (CudaSoftmaxPersistRun dataLen outLen)) CudaSoftmaxPersist_core_fine rfl w h

theorem CudaSoftmaxPersist_core_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaSoftmaxPersistRun dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.coreCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.coreCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  Hoare.run_sound (CudaSoftmaxPersist_core_triple cfg dataLen outLen w h) rfl m

theorem CudaSoftmaxPersist_finalize_fine : Fine (emitGo (CudaSoftmaxPersist.finalizeCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaSoftmaxPersist_finalize_triple (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaSoftmaxPersistRun dataLen outLen w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) (emit (CudaSoftmaxPersist.finalizeCode : Prog Slot Lvl Unit))
      { ok := fun _ w' => Contracts.TState.holds [] w', faultOk := false } :=
  triple_of_wp_entry (CudaSoftmaxPersistRun dataLen outLen) (Contracts.TState.holds []) (runArgs dataLen outLen)
    (by prog_vc (CudaSoftmaxPersistRun dataLen outLen)) CudaSoftmaxPersist_finalize_fine rfl w h

theorem CudaSoftmaxPersist_finalize_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaSoftmaxPersistRun dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.finalizeCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.finalizeCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  Hoare.run_sound (CudaSoftmaxPersist_finalize_triple cfg dataLen outLen w h) rfl m

-- CudaGemvPersist

abbrev CudaGemvPersistLoad (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena CudaGemvPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol]

abbrev CudaGemvPersistRun (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena CudaGemvPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat CudaGemvPersist.CTX_OFF) cudaCtx]

theorem CudaGemvPersist_load_fine : Fine (emitGo (CudaGemvPersist.loadCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaGemvPersist_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaGemvPersistLoad dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.loadCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.loadCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (CudaGemvPersistLoad dataLen outLen) (runArgs dataLen outLen) (by prog_vc (CudaGemvPersistLoad dataLen outLen))
    CudaGemvPersist_load_fine rfl h m

theorem CudaGemvPersist_prep_fine : Fine (emitGo (CudaGemvPersist.prepCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaGemvPersist_prep_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaGemvPersistRun dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.prepCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.prepCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (CudaGemvPersistRun dataLen outLen) (runArgs dataLen outLen) (by prog_vc (CudaGemvPersistRun dataLen outLen))
    CudaGemvPersist_prep_fine rfl h m

abbrev CudaGemvPersistInfer (dl ol m n : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena CudaGemvPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat CudaGemvPersist.CTX_OFF) cudaCtx,
   .cell (regionBase .arena + UInt64.ofNat CudaGemvPersist.M_OFF) m,
   .cell (regionBase .arena + UInt64.ofNat CudaGemvPersist.N_OFF) n,
   .devBuf 0 ((m * n) <<< 2), .devBuf 1 (n <<< 2), .devBuf 2 (m <<< 2)]

theorem CudaGemvPersist_infer_fine : Fine (emitGo (CudaGemvPersist.inferCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaGemvPersist_infer_no_misuse (cfg : Cfg) (dataLen outLen m n : UInt64)
    (hm : 0 < m.toNat ∧ m.toNat < 2 ^ 31) (hn : 0 < n.toNat ∧ n.toNat < 2 ^ 31) (w : World)
    (h : CudaGemvPersistInfer dataLen outLen m n w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.inferCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.inferCode : Prog Slot Lvl Unit)) ≠ .fault msg :=
  sound_of_wp_entry (CudaGemvPersistInfer dataLen outLen m n) (runArgs dataLen outLen)
    (by prog_vc (CudaGemvPersistInfer dataLen outLen m n)) CudaGemvPersist_infer_fine rfl h msg

-- CudaRmsNormPersist

abbrev CudaRmsNormPersistLoad (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena CudaRmsNormPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol, .cstrIn (regionBase .arena + UInt64.ofNat CudaRmsNormPersist.PTX_SOURCE_OFF) (CudaRmsNormPersist.BIND_DESC_OFF - CudaRmsNormPersist.PTX_SOURCE_OFF)]

abbrev CudaRmsNormPersistRun (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena CudaRmsNormPersist.MEM_SIZE, .roomArg .data dl, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat CudaRmsNormPersist.CTX_OFF) cudaCtx, .cstrIn (regionBase .arena + UInt64.ofNat CudaRmsNormPersist.PTX_SOURCE_OFF) (CudaRmsNormPersist.BIND_DESC_OFF - CudaRmsNormPersist.PTX_SOURCE_OFF)]

theorem CudaRmsNormPersist_load_fine : Fine (emitGo (CudaRmsNormPersist.loadCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaRmsNormPersist_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaRmsNormPersistLoad dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.loadCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.loadCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (CudaRmsNormPersistLoad dataLen outLen) (runArgs dataLen outLen) (by prog_vc (CudaRmsNormPersistLoad dataLen outLen))
    CudaRmsNormPersist_load_fine rfl h m

theorem CudaRmsNormPersist_prep_fine : Fine (emitGo (CudaRmsNormPersist.prepCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaRmsNormPersist_prep_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaRmsNormPersistRun dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.prepCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.prepCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (CudaRmsNormPersistRun dataLen outLen) (runArgs dataLen outLen) (by prog_vc (CudaRmsNormPersistRun dataLen outLen))
    CudaRmsNormPersist_prep_fine rfl h m

theorem CudaRmsNormPersist_infer_fine : Fine (emitGo (CudaRmsNormPersist.inferCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaRmsNormPersist_infer_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaRmsNormPersistRun dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.inferCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.inferCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (CudaRmsNormPersistRun dataLen outLen) (runArgs dataLen outLen) (by prog_vc (CudaRmsNormPersistRun dataLen outLen))
    CudaRmsNormPersist_infer_fine rfl h m

-- CudaDecodeAttention

abbrev CudaDecodeAttentionLoad (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .oracles, .room .arena CudaDecodeAttention.MEM_SIZE, .roomArg .data dl, .roomArg .out ol, .cstrIn (regionBase .arena + UInt64.ofNat CudaDecodeAttention.PTX_SOURCE_OFF) (CudaDecodeAttention.BIND_DESC_OFF - CudaDecodeAttention.PTX_SOURCE_OFF)]

abbrev CudaDecodeAttentionRun (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena CudaDecodeAttention.MEM_SIZE, .roomArg .data dl, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat CudaDecodeAttention.CTX_OFF) cudaCtx, .cstrIn (regionBase .arena + UInt64.ofNat CudaDecodeAttention.PTX_SOURCE_OFF) (CudaDecodeAttention.BIND_DESC_OFF - CudaDecodeAttention.PTX_SOURCE_OFF)]

theorem CudaDecodeAttention_load_fine : Fine (emitGo (CudaDecodeAttention.loadCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaDecodeAttention_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaDecodeAttentionLoad dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.loadCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.loadCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (CudaDecodeAttentionLoad dataLen outLen) (runArgs dataLen outLen) (by prog_vc (CudaDecodeAttentionLoad dataLen outLen))
    CudaDecodeAttention_load_fine rfl h m

theorem CudaDecodeAttention_prep_fine : Fine (emitGo (CudaDecodeAttention.prepCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaDecodeAttention_prep_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaDecodeAttentionRun dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.prepCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.prepCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (CudaDecodeAttentionRun dataLen outLen) (runArgs dataLen outLen) (by prog_vc (CudaDecodeAttentionRun dataLen outLen))
    CudaDecodeAttention_prep_fine rfl h m

theorem CudaDecodeAttention_finalize_fine : Fine (emitGo (CudaDecodeAttention.finalizeCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaDecodeAttention_finalize_triple (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaDecodeAttentionRun dataLen outLen w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) (emit (CudaDecodeAttention.finalizeCode : Prog Slot Lvl Unit))
      { ok := fun _ w' => Contracts.TState.holds [] w', faultOk := false } :=
  triple_of_wp_entry (CudaDecodeAttentionRun dataLen outLen) (Contracts.TState.holds []) (runArgs dataLen outLen)
    (by prog_vc (CudaDecodeAttentionRun dataLen outLen)) CudaDecodeAttention_finalize_fine rfl w h

theorem CudaDecodeAttention_finalize_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CudaDecodeAttentionRun dataLen outLen w)
    (m : String) : Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.finalizeCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.finalizeCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  Hoare.run_sound (CudaDecodeAttention_finalize_triple cfg dataLen outLen w h) rfl m

open CudaDecodeAttention in
abbrev CudaDecodeAttentionCore (dl ol s : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .part .cuda true, .part .capturing false, .devSeq, .oracles,
   .room .arena MEM_SIZE, .roomArg .data dl, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat CTX_OFF) cudaCtx,
   .cstrIn (regionBase .arena + UInt64.ofNat PTX_SOURCE_OFF) (BIND_DESC_OFF - PTX_SOURCE_OFF),
   -- buffer ids, two to a cell: q 0 and K 1, V 2 and scores 3, probs 4 and out 5, meta 6
   .cell (regionBase .arena + UInt64.ofNat BUF_Q_OFF) 4294967296,
   .cell (regionBase .arena + UInt64.ofNat BUF_V_OFF) 12884901890,
   .cell (regionBase .arena + UInt64.ofNat BUF_PROBS_OFF) 21474836484,
   .cell (regionBase .arena + UInt64.ofNat BUF_META_OFF) 6,
   .cell (regionBase .arena + UInt64.ofNat SEQ_LEN_OFF) s,
   .devBuf 0 3584, .devBuf 1 (s * 3584), .devBuf 2 (s * 3584), .devBuf 3 ((s * 14) <<< 2),
   .devBuf 4 ((s * 14) <<< 2), .devBuf 5 3584, .devBuf 6 8]

theorem CudaDecodeAttention_core_fine : Fine (emitGo (CudaDecodeAttention.coreCode : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaDecodeAttention_core_triple (cfg : Cfg) (dataLen outLen s : UInt64)
    (hs : 0 < s.toNat ∧ s.toNat ≤ CudaDecodeAttention.MAX_SEQ) (w : World)
    (h : CudaDecodeAttentionCore dataLen outLen s w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) (emit (CudaDecodeAttention.coreCode : Prog Slot Lvl Unit))
      { ok := fun _ w' => CudaDecodeAttentionCore dataLen outLen s w', faultOk := false } :=
  triple_of_wp_entry (CudaDecodeAttentionCore dataLen outLen s) (CudaDecodeAttentionCore dataLen outLen s) (runArgs dataLen outLen)
    (by simp only [CudaDecodeAttention.MAX_SEQ] at hs; prog_vc (CudaDecodeAttentionCore dataLen outLen s))
    CudaDecodeAttention_core_fine rfl w h

theorem CudaDecodeAttention_core_no_misuse (cfg : Cfg) (dataLen outLen s : UInt64)
    (hs : 0 < s.toNat ∧ s.toNat ≤ CudaDecodeAttention.MAX_SEQ) (w : World)
    (h : CudaDecodeAttentionCore dataLen outLen s w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.coreCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.coreCode : Prog Slot Lvl Unit)) ≠ .fault msg :=
  Hoare.run_sound (CudaDecodeAttention_core_triple cfg dataLen outLen s hs w h) rfl msg

-- The infer and stack wrappers of softmax and decode-attention

/-- A persistent benchmark's functions by the index each is called at, as its
    `clifIR` numbers them: load, prep, core, finalize, and the two wrappers
    that sequence core and finalize. -/
def persistFns (load prep core finalize : Body) (depth i : Nat) : Option Fn :=
  match i with
  | 1 => some (Fn.ofBody load)
  | 2 => some (Fn.ofBody prep)
  | 3 => some (Fn.ofBody core)
  | 4 => some (Fn.ofBody finalize)
  | 5 => some (Fn.ofBody (Prog.sequenceWrapper [3, 4]))
  | 6 => some (Fn.ofBody (Prog.sequenceWrapper (List.replicate depth 3 ++ [4])))
  | _ => none

abbrev softmaxFns : Nat → Option Fn :=
  persistFns CudaSoftmaxPersist.loadCode CudaSoftmaxPersist.prepCode CudaSoftmaxPersist.coreCode
    CudaSoftmaxPersist.finalizeCode CudaSoftmaxPersist.STACK_DEPTH

abbrev decodeFns : Nat → Option Fn :=
  persistFns CudaDecodeAttention.loadCode CudaDecodeAttention.prepCode CudaDecodeAttention.coreCode
    CudaDecodeAttention.finalizeCode CudaDecodeAttention.STACK_DEPTH

theorem softmax_infer_fine : Fine (emitGo (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

theorem softmax_stack_fine : Fine (emitGo (Prog.sequenceWrapper (List.replicate CudaSoftmaxPersist.STACK_DEPTH 3 ++ [4]) :
    Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- The softmax program's own calls, at any depth, on what an entry is handed:
    core leaves what it starts from, and finalize, the last call, needs to
    leave nothing. -/
theorem softmax_summaries (env : FnEnv) (steps k : Nat) (dataLen outLen : UInt64) :
    Summary (termLocals softmaxFns env steps k) 3 (runArgs dataLen outLen)
      (CudaSoftmaxPersistRun dataLen outLen) (CudaSoftmaxPersistRun dataLen outLen) ∧
    Summary (termLocals softmaxFns env steps k) 4 (runArgs dataLen outLen)
      (CudaSoftmaxPersistRun dataLen outLen) (Contracts.TState.holds []) :=
  ⟨summary_all rfl (fun _ w h => CudaSoftmaxPersist_core_triple _ dataLen outLen w h) k,
   summary_all rfl (fun _ w h => CudaSoftmaxPersist_finalize_triple _ dataLen outLen w h) k⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem CudaSoftmaxPersist_infer_no_misuse (env : FnEnv) (steps k : Nat) (dataLen outLen : UInt64) (w : World)
    (h : CudaSoftmaxPersistRun dataLen outLen w) (m : String) :
    Sem.run { env, steps, locals := termLocals softmaxFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run { env, steps, locals := termLocals softmaxFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) ≠ .fault m :=
  have h3 := (softmax_summaries env steps k dataLen outLen).1
  have h4 := (softmax_summaries env steps k dataLen outLen).2
  sound_of_wp_entry (CudaSoftmaxPersistRun dataLen outLen) (runArgs dataLen outLen)
    (by prog_vc (CudaSoftmaxPersistRun dataLen outLen)) softmax_infer_fine rfl h m

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem CudaSoftmaxPersist_stack_no_misuse (env : FnEnv) (steps k : Nat) (dataLen outLen : UInt64) (w : World)
    (h : CudaSoftmaxPersistRun dataLen outLen w) (m : String) :
    Sem.run { env, steps, locals := termLocals softmaxFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper (List.replicate CudaSoftmaxPersist.STACK_DEPTH 3 ++ [4]) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run { env, steps, locals := termLocals softmaxFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper (List.replicate CudaSoftmaxPersist.STACK_DEPTH 3 ++ [4]) : Prog Slot Lvl Unit)) ≠ .fault m :=
  have h3 := (softmax_summaries env steps k dataLen outLen).1
  have h4 := (softmax_summaries env steps k dataLen outLen).2
  sound_of_wp_entry (CudaSoftmaxPersistRun dataLen outLen) (runArgs dataLen outLen)
    (by prog_vc (CudaSoftmaxPersistRun dataLen outLen)) softmax_stack_fine rfl h m

theorem decode_infer_fine : Fine (emitGo (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  softmax_infer_fine

theorem decode_stack_fine : Fine (emitGo (Prog.sequenceWrapper (List.replicate CudaDecodeAttention.STACK_DEPTH 3 ++ [4]) :
    Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- The decode-attention program's own calls, at any depth, on what an entry
    is handed: core keeps the buffers and sequence length it starts from, and
    finalize, the last call, needs to leave nothing. -/
theorem decode_summaries (env : FnEnv) (steps k : Nat) (dataLen outLen s : UInt64)
    (hs : 0 < s.toNat ∧ s.toNat ≤ CudaDecodeAttention.MAX_SEQ) :
    Summary (termLocals decodeFns env steps k) 3 (runArgs dataLen outLen)
      (CudaDecodeAttentionCore dataLen outLen s) (CudaDecodeAttentionCore dataLen outLen s) ∧
    Summary (termLocals decodeFns env steps k) 4 (runArgs dataLen outLen)
      (CudaDecodeAttentionRun dataLen outLen) (Contracts.TState.holds []) :=
  ⟨summary_all rfl (fun _ w h => CudaDecodeAttention_core_triple _ dataLen outLen s hs w h) k,
   summary_all rfl (fun _ w h => CudaDecodeAttention_finalize_triple _ dataLen outLen w h) k⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem CudaDecodeAttention_infer_no_misuse (env : FnEnv) (steps k : Nat) (dataLen outLen s : UInt64)
    (hs : 0 < s.toNat ∧ s.toNat ≤ CudaDecodeAttention.MAX_SEQ) (w : World)
    (h : CudaDecodeAttentionCore dataLen outLen s w) (m : String) :
    Sem.run { env, steps, locals := termLocals decodeFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run { env, steps, locals := termLocals decodeFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) ≠ .fault m :=
  have h3 := (decode_summaries env steps k dataLen outLen s hs).1
  have h4 := (decode_summaries env steps k dataLen outLen s hs).2
  sound_of_wp_entry (CudaDecodeAttentionCore dataLen outLen s) (runArgs dataLen outLen)
    (by prog_vc (CudaDecodeAttentionCore dataLen outLen s)) decode_infer_fine rfl h m

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem CudaDecodeAttention_stack_no_misuse (env : FnEnv) (steps k : Nat) (dataLen outLen s : UInt64)
    (hs : 0 < s.toNat ∧ s.toNat ≤ CudaDecodeAttention.MAX_SEQ) (w : World)
    (h : CudaDecodeAttentionCore dataLen outLen s w) (m : String) :
    Sem.run { env, steps, locals := termLocals decodeFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper (List.replicate CudaDecodeAttention.STACK_DEPTH 3 ++ [4]) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run { env, steps, locals := termLocals decodeFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper (List.replicate CudaDecodeAttention.STACK_DEPTH 3 ++ [4]) : Prog Slot Lvl Unit)) ≠ .fault m :=
  have h3 := (decode_summaries env steps k dataLen outLen s hs).1
  have h4 := (decode_summaries env steps k dataLen outLen s hs).2
  sound_of_wp_entry (CudaDecodeAttentionCore dataLen outLen s) (runArgs dataLen outLen)
    (by prog_vc (CudaDecodeAttentionCore dataLen outLen s)) decode_stack_fine rfl h m

end CudaBenchSafe

#print axioms CudaBenchSafe.CudaSoftmaxPersist_infer_no_misuse
#print axioms CudaBenchSafe.CudaSoftmaxPersist_stack_no_misuse
#print axioms CudaBenchSafe.CudaDecodeAttention_infer_no_misuse
#print axioms CudaBenchSafe.CudaDecodeAttention_stack_no_misuse
