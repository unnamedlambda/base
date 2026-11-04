import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Check.Seq
import AlgorithmLib.Proof.Calls
import AlgorithmLib.Check.Entry
import Bench.CudaSoftmaxPersist
import Bench.CudaGemvPersist
import Bench.CudaRmsNormPersist
import Bench.CudaDecodeAttention

/-!
# The CUDA benchmarks neither misuse a call nor fault

Each entry point of each persistent CUDA benchmark, proven by the checker for
any lengths of data and output the caller hands over. Each calls the CUDA
driver (and cuBLAS) directly, in `Drv.App`'s layout: `main` starts from them
loaded and untouched (`loadTS`), the others from what `main` leaves for the
sizes it read (`runTSWith`). Where a call's sizes are known only at those
values, what it needs of them is shown here, given the size checks the
program makes before the call. Softmax's and decode attention's `core` and
`finalize` are proven as triples, so the `infer` and `stack` wrappers that
call them are proven through those.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CudaBenchSafe

-- CudaSoftmaxPersist: `main` from the driver loaded and untouched; the others
-- from what it leaves for `n` and its `nb` blocks

abbrev CudaSoftmaxPersistLoad (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  (CudaSoftmaxPersist.app.loadTS ++ [.oracles, .roomArg .data dl, .roomArg .out ol])

abbrev CudaSoftmaxPersistRun (dl ol n nb : UInt64) : World → Prop := Contracts.TState.holds
  (CudaSoftmaxPersist.app.runTSWith [(n * 4).toNat, (n * 4).toNat, 8, (nb * 8).toNat, 8] ++
    [.cell (Drv.App.at_ CudaSoftmaxPersist.N_OFF) n, .cell (Drv.App.at_ CudaSoftmaxPersist.NB_OFF) nb,
     .roomArg .data dl, .roomArg .out ol])

ship_entry CudaSoftmaxPersist_load : CudaSoftmaxPersist.loadCode under CudaSoftmaxPersistLoad dataLen outLen

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaSoftmaxPersist_prep_no_misuse (cfg : Cfg) (dataLen outLen n nb : UInt64) (w : World)
    (h : CudaSoftmaxPersistRun dataLen outLen n nb w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.prepCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.prepCode : Prog Slot Lvl Unit)) ≠ .fault msg := by
  check_body
  -- `x`'s bytes, no more than `4 n`, are its allocation's
  have hp : devPtrOf 412316860416 = some (0, 0) := by decide
  intro hc
  simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp, hc]

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaSoftmaxPersist_core_triple (cfg : Cfg) (dataLen outLen n nb : UInt64) (w : World)
    (h : CudaSoftmaxPersistRun dataLen outLen n nb w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) (emit (CudaSoftmaxPersist.coreCode : Prog Slot Lvl Unit))
      { ok := fun _ w' => CudaSoftmaxPersistRun dataLen outLen n nb w', faultOk := false } := by
  check_reach
  -- the long row's two launches `nb` blocks wide: no dimension empty
  all_goals
    intro hc
    simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
    have hw : nb &&& 4294967295 = nb := by
      apply UInt64.toNat_inj.mp
      simp only [UInt64.toNat_and]
      rw [show (4294967295 : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
      omega
    simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.dimOk, Check.PExpr.eval, Check.PArg.bits, hw]
    intro x hx
    match x, hx with
    | 0, _ =>
        have h0 : nb ≠ 0 := fun e => hc.2.2 (by simp [e])
        simpa [hw] using h0
    | 1, _ | 2, _ | 3, _ | 4, _ | 5, _ => rfl

theorem CudaSoftmaxPersist_core_no_misuse (cfg : Cfg) (dataLen outLen n nb : UInt64) (w : World)
    (h : CudaSoftmaxPersistRun dataLen outLen n nb w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.coreCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.coreCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  Hoare.run_sound (CudaSoftmaxPersist_core_triple cfg dataLen outLen n nb w h) rfl m

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaSoftmaxPersist_finalize_triple (cfg : Cfg) (dataLen outLen n nb : UInt64) (w : World)
    (h : CudaSoftmaxPersistRun dataLen outLen n nb w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) (emit (CudaSoftmaxPersist.finalizeCode : Prog Slot Lvl Unit))
      { ok := fun _ w' => Contracts.TState.holds [] w', faultOk := false } := by
  check_reach
  -- no more than `y`'s bytes
  have hp : devPtrOf 481036337152 = some (1, 0) := by decide
  intro hc
  simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp, hc]

theorem CudaSoftmaxPersist_finalize_no_misuse (cfg : Cfg) (dataLen outLen n nb : UInt64) (w : World)
    (h : CudaSoftmaxPersistRun dataLen outLen n nb w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.finalizeCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaSoftmaxPersist.finalizeCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  Hoare.run_sound (CudaSoftmaxPersist_finalize_triple cfg dataLen outLen n nb w h) rfl m

-- CudaGemvPersist: `main` from the driver and cuBLAS loaded and untouched;
-- `prep` and `infer` from what it leaves for `m` and `n`

abbrev CudaGemvPersistLoad (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  (CudaGemvPersist.app.loadTS ++ [.oracles, .roomArg .data dl, .roomArg .out ol])

abbrev CudaGemvPersistRun (dl ol m n : UInt64) : World → Prop := Contracts.TState.holds
  (CudaGemvPersist.app.runTSWith [(m * n * 4).toNat, (n * 4).toNat, (m * 4).toNat] ++
    [.cell (Drv.App.at_ CudaGemvPersist.M_OFF) m, .cell (Drv.App.at_ CudaGemvPersist.N_OFF) n,
     .roomArg .data dl, .roomArg .out ol])

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaGemvPersist_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : CudaGemvPersistLoad dataLen outLen w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.loadCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.loadCode : Prog Slot Lvl Unit)) ≠ .fault msg := by
  check_body
  -- `A`'s bytes are its allocation's
  have hp : devPtrOf 412316860416 = some (0, 0) := by decide
  intro _
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp]

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaGemvPersist_prep_no_misuse (cfg : Cfg) (dataLen outLen m n : UInt64) (w : World)
    (h : CudaGemvPersistRun dataLen outLen m n w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.prepCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.prepCode : Prog Slot Lvl Unit)) ≠ .fault msg := by
  check_body
  -- `x`'s bytes, no more than `4 n`, are its allocation's
  have hp : devPtrOf 481036337152 = some (1, 0) := by decide
  intro hc
  simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp, hc]

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaGemvPersist_infer_no_misuse (cfg : Cfg) (dataLen outLen m n : UInt64) (w : World)
    (h : CudaGemvPersistRun dataLen outLen m n w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.inferCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaGemvPersist.inferCode : Prog Slot Lvl Unit)) ≠ .fault msg := by
  check_body
  -- the product: `A`, `x` and `y` packed at the starts of their allocations
  · intro hc
    simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
    have em : asI32 (m &&& 4294967295) = m.toNat := Contracts.asI32_low hc.2.2.1
    have en : asI32 (n &&& 4294967295) = n.toNat := Contracts.asI32_low hc.1
    have e1 : asI32 1 = 1 := by decide
    have h0 : devPtrOf 412316860416 = some (0, 0) := by decide
    have h1 : devPtrOf 481036337152 = some (1, 0) := by decide
    have h2 : devPtrOf 549755813888 = some (2, 0) := by decide
    have hmn : m.toNat * n.toNat < 2 ^ 62 :=
      Nat.lt_of_lt_of_le (Nat.mul_lt_mul'' hc.2.2.1 hc.1) (by decide)
    simp [Check.vendorVal, Check.blasVal, Check.getAll, Check.argAt, Check.sgemvVal, Check.operandOf,
      Check.allocFor, Check.allocFacts, Check.PExpr.eval, Check.PArg.bits, h0, h1, h2, em, en, e1,
      sgemvLens, sgemvInvalid, sgemvUnstated, opOk, UInt64.toNat_mul]
    have a0 : 4 * (n.toNat * m.toNat) ≤ m.toNat * n.toNat * 4 % 18446744073709551616 := by
      rw [Nat.mul_comm n.toNat]; generalize m.toNat * n.toNat = p at *; omega
    have a1 : 4 * n.toNat ≤ n.toNat * 4 % 18446744073709551616 := by omega
    have a2 : 4 * m.toNat ≤ m.toNat * 4 % 18446744073709551616 := by omega
    simp only [a0, a1, a2, if_true]
    simp
    omega
  -- the download: no more than `y`'s bytes
  · have hp : devPtrOf 549755813888 = some (2, 0) := by decide
    intro hc
    simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
    simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
      Check.PExpr.eval, Check.PArg.bits, hp, hc]

-- CudaRmsNormPersist

abbrev CudaRmsNormPersistLoad (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  (CudaRmsNormPersist.app.loadTS ++ [.oracles, .roomArg .data dl, .roomArg .out ol])

abbrev CudaRmsNormPersistRun (dl ol n : UInt64) : World → Prop := Contracts.TState.holds
  (CudaRmsNormPersist.app.runTSWith [(n * 8 + 8).toNat, (n * 4).toNat] ++
    [.cell (Drv.App.at_ CudaRmsNormPersist.N_OFF) n, .roomArg .data dl, .roomArg .out ol])

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaRmsNormPersist_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : CudaRmsNormPersistLoad dataLen outLen w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.loadCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.loadCode : Prog Slot Lvl Unit)) ≠ .fault msg := by
  check_body
  -- `N` at the start of allocation 0
  · have hp : devPtrOf 412316860416 = some (0, 0) := by decide
    intro hc
    simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
    simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
      Check.PExpr.eval, Check.PArg.bits, hp, UInt64.toNat_add, UInt64.toNat_sub]
    omega
  -- the weights after `N` and `x`
  · intro hc
    simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
    have hp : devPtrOf (412316860416 + dataLen) = some (0, dataLen.toNat) := by
      unfold devPtrOf
      have : (412316860416 + dataLen).toNat = 412316860416 + dataLen.toNat := by
        simp [UInt64.toNat_add]; omega
      simp [this]
      omega
    simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
      Check.PExpr.eval, Check.PArg.bits, hp, UInt64.toNat_add, UInt64.toNat_sub]
    omega

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaRmsNormPersist_prep_no_misuse (cfg : Cfg) (dataLen outLen n : UInt64) (w : World)
    (h : CudaRmsNormPersistRun dataLen outLen n w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.prepCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.prepCode : Prog Slot Lvl Unit)) ≠ .fault msg := by
  check_body
  -- `x` after `N`
  have hp : devPtrOf 412316860424 = some (0, 8) := by decide
  intro hc
  simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp, UInt64.toNat_add, UInt64.toNat_mul]
  omega

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaRmsNormPersist_infer_no_misuse (cfg : Cfg) (dataLen outLen n : UInt64) (w : World)
    (h : CudaRmsNormPersistRun dataLen outLen n w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.inferCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaRmsNormPersist.inferCode : Prog Slot Lvl Unit)) ≠ .fault msg := by
  check_body
  -- no more than `y`'s bytes
  have hp : devPtrOf 481036337152 = some (1, 0) := by decide
  intro hc
  simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp, hc]

-- CudaDecodeAttention: `main` from the driver and cuBLAS loaded and
-- untouched; the others from what it leaves for `s` tokens

abbrev CudaDecodeAttentionLoad (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  (CudaDecodeAttention.app.loadTS ++ [.oracles, .roomArg .data dl, .roomArg .out ol])

open CudaDecodeAttention in
abbrev CudaDecodeAttentionRun (dl ol s : UInt64) : World → Prop := Contracts.TState.holds
  (app.runTSWith [D_MODEL_BYTES, (s * 7168).toNat, (s * 56).toNat, (s * 56).toNat, D_MODEL_BYTES, 8] ++
    [.cell (Drv.App.at_ SEQ_LEN_OFF) s, .roomArg .data dl, .roomArg .out ol])

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaDecodeAttention_load_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : CudaDecodeAttentionLoad dataLen outLen w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.loadCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.loadCode : Prog Slot Lvl Unit)) ≠ .fault msg := by
  check_body
  -- K and V, the data after `seq_len`, are their allocation's
  have hp : devPtrOf 481036337152 = some (1, 0) := by decide
  intro _
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp]

ship_entry CudaDecodeAttention_prep : CudaDecodeAttention.prepCode under CudaDecodeAttentionRun dataLen outLen s

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaDecodeAttention_finalize_triple (cfg : Cfg) (dataLen outLen s : UInt64) (w : World)
    (h : CudaDecodeAttentionRun dataLen outLen s w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) (emit (CudaDecodeAttention.finalizeCode : Prog Slot Lvl Unit))
      { ok := fun _ w' => Contracts.TState.holds [] w', faultOk := false } := by
  check_reach

theorem CudaDecodeAttention_finalize_no_misuse (cfg : Cfg) (dataLen outLen s : UInt64) (w : World)
    (h : CudaDecodeAttentionRun dataLen outLen s w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.finalizeCode : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.finalizeCode : Prog Slot Lvl Unit)) ≠ .fault m :=
  Hoare.run_sound (CudaDecodeAttention_finalize_triple cfg dataLen outLen s w h) rfl m

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem CudaDecodeAttention_core_triple (cfg : Cfg) (dataLen outLen s : UInt64) (w : World)
    (h : CudaDecodeAttentionRun dataLen outLen s w) :
    Hoare.Triple cfg (Hoare.At (runArgs dataLen outLen).toArray w) (emit (CudaDecodeAttention.coreCode : Prog Slot Lvl Unit))
      { ok := fun _ w' => CudaDecodeAttentionRun dataLen outLen s w', faultOk := false } := by
  check_reach
  -- the two products, 14 heads each, every operand inside its allocation
  all_goals
    intro hc
    simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
    have hs : s.toNat < 2049 := by omega
    have hs0 : s.toNat ≠ 0 := by omega
    have e32 : asI32 (s &&& 4294967295) = s.toNat := Contracts.asI32_low (by omega)
    have m64 : (s * 64).toNat = s.toNat * 64 := by simp [UInt64.toNat_mul]; omega
    have m3584 : (s * 3584).toNat = s.toNat * 3584 := by simp [UInt64.toNat_mul]; omega
    have m7168 : (s * 7168).toNat = s.toNat * 7168 := by simp [UInt64.toNat_mul]; omega
    have m56 : (s * 56).toNat = s.toNat * 56 := by simp [UInt64.toNat_mul]; omega
    have f64 : asI64 (s * 64) = ((s.toNat * 64 : Nat) : Int) := by
      unfold asI64; rw [Check.signed_small rfl (by rw [m64]; show _ < 2 ^ 63; omega), m64]
    have f1 : asI64 s = (s.toNat : Int) := by
      unfold asI64; rw [Check.signed_small rfl (by show _ < 2 ^ 63; omega)]
    have hV : devPtrOf (481036337152 + s * 3584) = some (1, s.toNat * 3584) := by
      unfold devPtrOf
      have : (481036337152 + s * 3584).toNat = 481036337152 + s.toNat * 3584 := by
        simp [UInt64.toNat_add, UInt64.toNat_mul]; omega
      simp [this]
      omega
    have h0 : devPtrOf 412316860416 = some (0, 0) := by decide
    have h1 : devPtrOf 481036337152 = some (1, 0) := by decide
    have h2 : devPtrOf 549755813888 = some (2, 0) := by decide
    have h3 : devPtrOf 618475290624 = some (3, 0) := by decide
    have h4 : devPtrOf 687194767360 = some (4, 0) := by decide
    have c0 : asI32 0 = 0 := by decide
    have c1 : asI32 1 = 1 := by decide
    have c14 : asI32 14 = 14 := by decide
    have c64 : asI32 64 = 64 := by decide
    have d64 : asI64 64 = 64 := by decide
    simp [Check.vendorVal, Check.blasVal, Check.getAll, Check.argAt, Check.gemmVal, Check.operandOf,
      Check.allocFor, Check.allocFacts, Check.PExpr.eval, Check.PArg.bits, gemmSpans, gemmDims, batchSpan,
      colMajorSpan, gemmInvalid, gemmUnstated, opOk, e32, f64, f1, m7168, m56, hV, h0, h1, h2, h3, h4,
      c0, c1, c14, c64, d64]
    have ti : ((s.toNat : Int) * 64).toNat = s.toNat * 64 := by omega
    simp only [ti, hs0, if_false]
    have k1 : 4 * (13 * (s.toNat * 64) + (64 * (s.toNat - 1) + 64)) ≤ s.toNat * 7168 := by omega
    have k2 : 4 * (13 * s.toNat + s.toNat) ≤ s.toNat * 56 := by omega
    have k3 : s.toNat * 3584 + 4 * (13 * (s.toNat * 64) + (64 * (s.toNat - 1) + 64)) ≤ s.toNat * 7168 := by
      omega
    have k4 : s.toNat * 3584 % 4 = 0 := by omega
    simp only [k1, k2, k3, k4, and_self, if_true]
    simp
    omega

theorem CudaDecodeAttention_core_no_misuse (cfg : Cfg) (dataLen outLen s : UInt64) (w : World)
    (h : CudaDecodeAttentionRun dataLen outLen s w) (msg : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.coreCode : Prog Slot Lvl Unit)) ≠ .misuse msg ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CudaDecodeAttention.coreCode : Prog Slot Lvl Unit)) ≠ .fault msg :=
  Hoare.run_sound (CudaDecodeAttention_core_triple cfg dataLen outLen s w h) rfl msg

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

theorem softmax_infer_fine : Fine (emitGo (fnBody ptrParams (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

theorem softmax_stack_fine : Fine (emitGo (fnBody ptrParams (Prog.sequenceWrapper (List.replicate CudaSoftmaxPersist.STACK_DEPTH 3 ++ [4]) :
    Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- The softmax program's own calls, at any depth, on what an entry is handed:
    core leaves what it starts from, and finalize, the last call, needs to
    leave nothing. -/
theorem softmax_summaries (env : FnEnv) (steps k : Nat) (dataLen outLen n nb : UInt64) :
    Summary (termLocals softmaxFns env steps k) 3 (runArgs dataLen outLen)
      (CudaSoftmaxPersistRun dataLen outLen n nb) (CudaSoftmaxPersistRun dataLen outLen n nb) ∧
    Summary (termLocals softmaxFns env steps k) 4 (runArgs dataLen outLen)
      (CudaSoftmaxPersistRun dataLen outLen n nb) (Contracts.TState.holds []) :=
  ⟨summary_all rfl (fun _ w h => CudaSoftmaxPersist_core_triple _ dataLen outLen n nb w h) k,
   summary_all rfl (fun _ w h => CudaSoftmaxPersist_finalize_triple _ dataLen outLen n nb w h) k⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem CudaSoftmaxPersist_infer_no_misuse (env : FnEnv) (steps k : Nat) (dataLen outLen n nb : UInt64) (w : World)
    (h : CudaSoftmaxPersistRun dataLen outLen n nb w) (m : String) :
    Sem.run { env, steps, locals := termLocals softmaxFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run { env, steps, locals := termLocals softmaxFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) ≠ .fault m :=
  have h3 := (softmax_summaries env steps k dataLen outLen n nb).1
  have h4 := (softmax_summaries env steps k dataLen outLen n nb).2
  by
    rw [show (emit (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) =
      [.straight (.op .memory :: Check.seqCalls [3, 4])] from rfl]
    exact Check.seqCalls_run (cs := [3, 4]) ⟨_, h3, _, h4, trivial⟩ h m

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem CudaSoftmaxPersist_stack_no_misuse (env : FnEnv) (steps k : Nat) (dataLen outLen n nb : UInt64) (w : World)
    (h : CudaSoftmaxPersistRun dataLen outLen n nb w) (m : String) :
    Sem.run { env, steps, locals := termLocals softmaxFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper (List.replicate CudaSoftmaxPersist.STACK_DEPTH 3 ++ [4]) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run { env, steps, locals := termLocals softmaxFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper (List.replicate CudaSoftmaxPersist.STACK_DEPTH 3 ++ [4]) : Prog Slot Lvl Unit)) ≠ .fault m :=
  have h3 := (softmax_summaries env steps k dataLen outLen n nb).1
  have h4 := (softmax_summaries env steps k dataLen outLen n nb).2
  by
    rw [show (emit (Prog.sequenceWrapper (List.replicate CudaSoftmaxPersist.STACK_DEPTH 3 ++ [4]) : Prog Slot Lvl Unit)) =
      [.straight (.op .memory :: Check.seqCalls (List.replicate CudaSoftmaxPersist.STACK_DEPTH 3 ++ [4]))] from rfl]
    exact Check.seqCalls_run (Check.sumChain_repeat h3 (rest := [4]) ⟨_, h4, trivial⟩ _) h m

theorem decode_infer_fine : Fine (emitGo (fnBody ptrParams (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  softmax_infer_fine

theorem decode_stack_fine : Fine (emitGo (fnBody ptrParams (Prog.sequenceWrapper (List.replicate CudaDecodeAttention.STACK_DEPTH 3 ++ [4]) :
    Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- The decode-attention program's own calls, at any depth, on what an entry
    is handed: core keeps the buffers and sequence length it starts from, and
    finalize, the last call, needs to leave nothing. -/
theorem decode_summaries (env : FnEnv) (steps k : Nat) (dataLen outLen s : UInt64) :
    Summary (termLocals decodeFns env steps k) 3 (runArgs dataLen outLen)
      (CudaDecodeAttentionRun dataLen outLen s) (CudaDecodeAttentionRun dataLen outLen s) ∧
    Summary (termLocals decodeFns env steps k) 4 (runArgs dataLen outLen)
      (CudaDecodeAttentionRun dataLen outLen s) (Contracts.TState.holds []) :=
  ⟨summary_all rfl (fun _ w h => CudaDecodeAttention_core_triple _ dataLen outLen s w h) k,
   summary_all rfl (fun _ w h => CudaDecodeAttention_finalize_triple _ dataLen outLen s w h) k⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem CudaDecodeAttention_infer_no_misuse (env : FnEnv) (steps k : Nat) (dataLen outLen s : UInt64)
    (w : World) (h : CudaDecodeAttentionRun dataLen outLen s w) (m : String) :
    Sem.run { env, steps, locals := termLocals decodeFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run { env, steps, locals := termLocals decodeFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) ≠ .fault m :=
  have h3 := (decode_summaries env steps k dataLen outLen s).1
  have h4 := (decode_summaries env steps k dataLen outLen s).2
  by
    rw [show (emit (Prog.sequenceWrapper [3, 4] : Prog Slot Lvl Unit)) =
      [.straight (.op .memory :: Check.seqCalls [3, 4])] from rfl]
    exact Check.seqCalls_run (cs := [3, 4]) ⟨_, h3, _, h4, trivial⟩ h m

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem CudaDecodeAttention_stack_no_misuse (env : FnEnv) (steps k : Nat) (dataLen outLen s : UInt64)
    (w : World) (h : CudaDecodeAttentionRun dataLen outLen s w) (m : String) :
    Sem.run { env, steps, locals := termLocals decodeFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper (List.replicate CudaDecodeAttention.STACK_DEPTH 3 ++ [4]) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run { env, steps, locals := termLocals decodeFns env steps k } (runArgs dataLen outLen) w
      (emit (Prog.sequenceWrapper (List.replicate CudaDecodeAttention.STACK_DEPTH 3 ++ [4]) : Prog Slot Lvl Unit)) ≠ .fault m :=
  have h3 := (decode_summaries env steps k dataLen outLen s).1
  have h4 := (decode_summaries env steps k dataLen outLen s).2
  by
    rw [show (emit (Prog.sequenceWrapper (List.replicate CudaDecodeAttention.STACK_DEPTH 3 ++ [4]) : Prog Slot Lvl Unit)) =
      [.straight (.op .memory :: Check.seqCalls (List.replicate CudaDecodeAttention.STACK_DEPTH 3 ++ [4]))] from rfl]
    exact Check.seqCalls_run (Check.sumChain_repeat h3 (rest := [4]) ⟨_, h4, trivial⟩ _) h m

end CudaBenchSafe

#print axioms CudaBenchSafe.CudaSoftmaxPersist_load_no_misuse
#print axioms CudaBenchSafe.CudaSoftmaxPersist_prep_no_misuse
#print axioms CudaBenchSafe.CudaSoftmaxPersist_infer_no_misuse
#print axioms CudaBenchSafe.CudaSoftmaxPersist_stack_no_misuse
#print axioms CudaBenchSafe.CudaDecodeAttention_load_no_misuse
#print axioms CudaBenchSafe.CudaDecodeAttention_prep_no_misuse
#print axioms CudaBenchSafe.CudaDecodeAttention_infer_no_misuse
#print axioms CudaBenchSafe.CudaDecodeAttention_stack_no_misuse
#print axioms CudaBenchSafe.CudaRmsNormPersist_load_no_misuse
#print axioms CudaBenchSafe.CudaRmsNormPersist_prep_no_misuse
#print axioms CudaBenchSafe.CudaRmsNormPersist_infer_no_misuse
#print axioms CudaBenchSafe.CudaGemvPersist_load_no_misuse
#print axioms CudaBenchSafe.CudaGemvPersist_prep_no_misuse
#print axioms CudaBenchSafe.CudaGemvPersist_infer_no_misuse
