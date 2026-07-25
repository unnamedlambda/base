import AlgorithmLib.LZ4WarpKernel

open AlgorithmLib AlgorithmLib.LZ4WarpDSL AlgorithmLib.LZ4Simt AlgorithmLib.LZ4

-- Some `simp`/`<;> simp` calls below are shared across several subgoals coming from
-- a single `refine ⟨_, _, …⟩`; a hypothesis unused in one subgoal can still be load-bearing
-- in another, so the (per-subgoal) "unused simp argument" linter is not reliable here.
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

namespace EmitContent

-- ══════════════════════════════════════════════════════════════════════════
-- §1. Reproduced from `LZ4WarpEvalBytes_proofs.lean` (structural facts)
-- ══════════════════════════════════════════════════════════════════════════

/-- Sequential byte writer at a running `UInt64` address (mirrors the machine:
    each `wStoreByte` writes at `outBase+op` then bumps `op` by one). -/
def putBytesU (g : Array UInt8) (base : UInt64) : List UInt8 → Array UInt8
  | [] => g
  | b :: bs => putBytesU (g.set! base.toNat b) (base + 1) bs

theorem putBytesU_append (g : Array UInt8) (base : UInt64) (xs ys : List UInt8) :
    putBytesU g base (xs ++ ys)
      = putBytesU (putBytesU g base xs) (base + UInt64.ofNat xs.length) ys := by
  induction xs generalizing g base with
  | nil => simp [putBytesU]
  | cons x xs ih =>
    simp only [List.cons_append, putBytesU, List.length_cons, ih]
    have haddr : base + 1 + UInt64.ofNat xs.length = base + UInt64.ofNat (xs.length + 1) := by
      rw [UInt64.ofNat_add]; ac_rfl
    rw [haddr]

/-- The exact loop body of `wEmitLSIC` (matches the inline `wseq` there). -/
def lsicBody (n : String) : WStmt := wseq
  [ .mov "c255" (.imm 255),
    wStoreByte "c255",
    .bin .sub n n (.imm 255),
    .setp .ge "lsicC" n (.imm 255) ]

theorem lsicBodyStep (n : String)
    (hn_c : n ≠ "c255") (hn_l : n ≠ "lsicC") (hn_op : n ≠ "op")
    (hn_ob : n ≠ "outBase") (hn_sb : n ≠ "sbAddr") (ws : WState) (f : Nat) :
    ((lsicBody n).eval f ws).regs n = ws.regs n - UInt64.ofNat 255
    ∧ ((lsicBody n).eval f ws).regs "op" = ws.regs "op" + 1
    ∧ ((lsicBody n).eval f ws).regs "outBase" = ws.regs "outBase"
    ∧ ((lsicBody n).eval f ws).regs "lsicC"
        = (if UInt64.ofNat 255 ≤ ws.regs n - UInt64.ofNat 255 then 1 else 0)
    ∧ ((lsicBody n).eval f ws).gmem
        = ws.gmem.set! (ws.regs "outBase" + ws.regs "op").toNat (255 : UInt8) := by
  simp only [lsicBody, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WArg.eval,
    WOp.run, SCmp.run]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [hn_c, hn_l, hn_op, hn_ob, hn_sb, Ne.symm hn_c, Ne.symm hn_l, Ne.symm hn_op,
      Ne.symm hn_ob, Ne.symm hn_sb]

theorem lsicLoop (n : String)
    (hn_c : n ≠ "c255") (hn_l : n ≠ "lsicC") (hn_op : n ≠ "op")
    (hn_ob : n ≠ "outBase") (hn_sb : n ≠ "sbAddr") :
    ∀ (q : Nat) (ws : WState) (fuel : Nat),
      (ws.regs n).toNat / 255 = q →
      ws.regs "lsicC" = (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0) →
      q < fuel →
      (((WStmt.uwhile "lsicC" (lsicBody n)).eval fuel ws).regs n).toNat = (ws.regs n).toNat % 255
      ∧ ((WStmt.uwhile "lsicC" (lsicBody n)).eval fuel ws).regs "op"
          = ws.regs "op" + UInt64.ofNat q
      ∧ ((WStmt.uwhile "lsicC" (lsicBody n)).eval fuel ws).regs "outBase" = ws.regs "outBase"
      ∧ ((WStmt.uwhile "lsicC" (lsicBody n)).eval fuel ws).gmem
          = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op") (List.replicate q 255) := by
  have h255 : (UInt64.ofNat 255).toNat = 255 := by decide
  intro q
  induction q with
  | zero =>
    intro ws fuel hq hlsic hfuel
    have hvlt : (ws.regs n).toNat < 255 := by omega
    have hnge : ¬ (UInt64.ofNat 255 ≤ ws.regs n) := by
      rw [UInt64.le_iff_toNat_le, h255]; omega
    have hl0 : ws.regs "lsicC" = 0 := by rw [hlsic, if_neg hnge]
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
    simp only [WStmt.eval, hl0]
    refine ⟨?_, ?_, ?_, ?_⟩ <;>
      simp [putBytesU, Nat.mod_eq_of_lt hvlt]
  | succ q ih =>
    intro ws fuel hq hlsic hfuel
    have hvge : 255 ≤ (ws.regs n).toNat := by
      have : 1 ≤ (ws.regs n).toNat / 255 := by omega
      omega
    have hge : UInt64.ofNat 255 ≤ ws.regs n := by rw [UInt64.le_iff_toNat_le, h255]; omega
    have hl1 : ws.regs "lsicC" = 1 := by rw [hlsic, if_pos hge]
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
    obtain ⟨hsn, hsop, hsob, hslc, hsg⟩ := lsicBodyStep n hn_c hn_l hn_op hn_ob hn_sb ws f
    have hred : (WStmt.uwhile "lsicC" (lsicBody n)).eval (f + 1) ws
        = (WStmt.uwhile "lsicC" (lsicBody n)).eval f ((lsicBody n).eval f ws) := by
      simp only [WStmt.eval, hl1]; rfl
    have hsnNat : (((lsicBody n).eval f ws).regs n).toNat = (ws.regs n).toNat - 255 := by
      rw [hsn, UInt64.toNat_sub_of_le _ _ hge, h255]
    have hq1 : (((lsicBody n).eval f ws).regs n).toNat / 255 = q := by rw [hsnNat]; omega
    have hlsic1 : ((lsicBody n).eval f ws).regs "lsicC"
        = (if UInt64.ofNat 255 ≤ ((lsicBody n).eval f ws).regs n then 1 else 0) := by
      rw [hslc, hsn]
    obtain ⟨rn, rop, rob, rg⟩ := ih ((lsicBody n).eval f ws) f hq1 hlsic1 (by omega)
    rw [hred]
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [rn, hsnNat]; omega
    · rw [rop, hsop, show q + 1 = 1 + q from by omega, UInt64.ofNat_add,
        show UInt64.ofNat 1 = 1 from rfl]
      ac_rfl
    · rw [rob, hsob]
    · rw [rg, hsg, hsob, hsop, List.replicate_succ]
      simp only [putBytesU]
      rw [show ws.regs "outBase" + (ws.regs "op" + 1)
            = ws.regs "outBase" + ws.regs "op" + 1 from by ac_rfl]

theorem eval_wStoreByte (val : String) (ws : WState) (fuel : Nat)
    (h1 : val ≠ "sbAddr") (_h2 : val ≠ "op") :
    ((wStoreByte val).eval fuel ws).regs "op" = ws.regs "op" + 1
    ∧ ((wStoreByte val).eval fuel ws).regs "outBase" = ws.regs "outBase"
    ∧ ((wStoreByte val).eval fuel ws).gmem
        = ws.gmem.set! (ws.regs "outBase" + ws.regs "op").toNat (ws.regs val).toUInt8 := by
  simp only [wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WArg.eval, WOp.run]
  refine ⟨?_, ?_, ?_⟩ <;> simp [h1]

theorem lor_mul_add (a b k : Nat) (ha : a < 2 ^ k) : a ||| (b * 2 ^ k) = a + b * 2 ^ k := by
  rw [← Nat.shiftLeft_eq]
  apply Nat.eq_of_testBit_eq
  intro j
  rw [Nat.testBit_or, Nat.testBit_shiftLeft, Nat.shiftLeft_eq, Nat.mul_comm b, Nat.add_comm,
      Nat.testBit_two_pow_mul_add b ha j]
  by_cases hj : j < k
  · simp [hj, ge_iff_le, Nat.not_le.mpr hj]
  · have hkj : k ≤ j := Nat.le_of_not_lt hj
    have hf : a.testBit j = false :=
      Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le ha (Nat.pow_le_pow_right (by decide) hkj))
    simp [hj, ge_iff_le, hkj, hf]

theorem token_byte (a b : UInt64) (hb : b.toNat < 16) :
    ((if a ≤ UInt64.ofNat 15 then a else UInt64.ofNat 15) <<< UInt64.ofNat 4 ||| b).toUInt8
      = UInt8.ofNat (min a.toNat 15 * 16 + b.toNat) := by
  have hmnat : (if a ≤ UInt64.ofNat 15 then a else UInt64.ofNat 15).toNat = min a.toNat 15 := by
    have h15 : (UInt64.ofNat 15).toNat = 15 := by decide
    split
    · next h => rw [UInt64.le_iff_toNat_le, h15] at h; omega
    · next h => rw [UInt64.le_iff_toNat_le, h15] at h; rw [h15]; omega
  generalize hgen : (if a ≤ UInt64.ofNat 15 then a else UInt64.ofNat 15) = m at hmnat ⊢
  have hmle : m.toNat ≤ 15 := by rw [hmnat]; omega
  have hshift : (m <<< UInt64.ofNat 4).toNat = m.toNat * 16 := by
    have h4 : (UInt64.ofNat 4).toNat % 64 = 4 := by decide
    rw [UInt64.toNat_shiftLeft, h4, Nat.shiftLeft_eq, show (2:Nat) ^ 4 = 16 from rfl,
        Nat.mod_eq_of_lt (by omega)]
  have hval : (m <<< UInt64.ofNat 4 ||| b).toNat = min a.toNat 15 * 16 + b.toNat := by
    rw [UInt64.toNat_or, hshift, ← hmnat]
    have hbk : b.toNat < 2 ^ 4 := by omega
    rw [Nat.or_comm, show m.toNat * 16 = m.toNat * 2 ^ 4 from rfl,
        lor_mul_add b.toNat m.toNat 4 hbk, Nat.add_comm]
  show (m <<< UInt64.ofNat 4 ||| b).toNat.toUInt8 = _
  rw [hval]

theorem eval_wEmitToken (litLen tokLo : String) (ws : WState) (fuel : Nat)
    (hbnd : (ws.regs tokLo).toNat < 16)
    (htl1 : tokLo ≠ "tokHi") (htl2 : tokLo ≠ "tok") :
    ((wEmitToken litLen tokLo).eval fuel ws).regs "op" = ws.regs "op" + 1
    ∧ ((wEmitToken litLen tokLo).eval fuel ws).regs "outBase" = ws.regs "outBase"
    ∧ ((wEmitToken litLen tokLo).eval fuel ws).gmem
        = ws.gmem.set! (ws.regs "outBase" + ws.regs "op").toNat
            (UInt8.ofNat ((min (ws.regs litLen).toNat 15) * 16 + (ws.regs tokLo).toNat)) := by
  simp only [wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WArg.eval,
    WOp.run]
  refine ⟨?_, ?_, ?_⟩
  · simp
  · simp
  · simp only [htl1, htl2, ↓reduceIte, String.reduceEq]
    rw [token_byte (ws.regs litLen) (ws.regs tokLo) hbnd]

theorem eval_wEmitLSIC (n : String) (ws : WState) (fuel : Nat)
    (hn_c : n ≠ "c255") (hn_l : n ≠ "lsicC") (hn_op : n ≠ "op")
    (hn_ob : n ≠ "outBase") (hn_sb : n ≠ "sbAddr")
    (hfuel : (ws.regs n).toNat / 255 < fuel) :
    ((wEmitLSIC n).eval fuel ws).regs "op"
        = ws.regs "op" + UInt64.ofNat ((ext (ws.regs n).toNat).length)
    ∧ ((wEmitLSIC n).eval fuel ws).regs "outBase" = ws.regs "outBase"
    ∧ ((wEmitLSIC n).eval fuel ws).gmem
        = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op") (ext (ws.regs n).toNat) := by
  have hsetp : WStmt.eval fuel (WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255)) ws
      = ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0) := by
    simp [WStmt.eval, WArg.eval, SCmp.run]
  have hmaster : (wEmitLSIC n).eval fuel ws
      = (wStoreByte n).eval fuel
          ((WStmt.uwhile "lsicC" (lsicBody n)).eval fuel
            (ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0))) := by
    show WStmt.eval fuel (WStmt.seq (WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255))
        (WStmt.seq (WStmt.uwhile "lsicC" (lsicBody n)) (wStoreByte n))) ws = _
    rw [WStmt.eval, WStmt.eval, hsetp]
  have e_n : (ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0)).regs n
      = ws.regs n := by simp [WState.setReg, hn_l]
  have e_lc : (ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0)).regs "lsicC"
      = (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0) := by simp [WState.setReg]
  have e_op : (ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0)).regs "op"
      = ws.regs "op" := by simp [WState.setReg]
  have e_ob : (ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0)).regs "outBase"
      = ws.regs "outBase" := by simp [WState.setReg]
  have e_gm : (ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0)).gmem
      = ws.gmem := rfl
  obtain ⟨rn, rop, rob, rg⟩ := lsicLoop n hn_c hn_l hn_op hn_ob hn_sb
      ((ws.regs n).toNat / 255)
      (ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0))
      fuel (by rw [e_n]) (by rw [e_lc, e_n]) hfuel
  obtain ⟨sop, sob, sgm⟩ := eval_wStoreByte n
      ((WStmt.uwhile "lsicC" (lsicBody n)).eval fuel
        (ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0))) fuel hn_sb hn_op
  have hlen : (ext (ws.regs n).toNat).length = (ws.regs n).toNat / 255 + 1 := by simp [ext]
  rw [hmaster]
  refine ⟨?_, ?_, ?_⟩
  · rw [sop, rop, e_op, hlen, show (ws.regs n).toNat / 255 + 1 = 1 + (ws.regs n).toNat / 255 from by omega,
        UInt64.ofNat_add, show UInt64.ofNat 1 = 1 from rfl]
    ac_rfl
  · rw [sob, rob, e_ob]
  · rw [sgm, rg, e_gm, rob, rop, e_ob, e_op]
    have hbyte : (((WStmt.uwhile "lsicC" (lsicBody n)).eval fuel
        (ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0))).regs n).toUInt8
        = UInt8.ofNat ((ws.regs n).toNat % 255) := by
      show UInt8.ofNat (((WStmt.uwhile "lsicC" (lsicBody n)).eval fuel
        (ws.setReg "lsicC" (if UInt64.ofNat 255 ≤ ws.regs n then 1 else 0))).regs n).toNat = _
      rw [rn, e_n]
    rw [hbyte, show ext (ws.regs n).toNat
          = List.replicate ((ws.regs n).toNat / 255) 255 ++ [UInt8.ofNat ((ws.regs n).toNat % 255)]
          from rfl, putBytesU_append]
    simp only [List.length_replicate, putBytesU]
    rw [show ws.regs "outBase" + (ws.regs "op" + UInt64.ofNat ((ws.regs n).toNat / 255))
          = ws.regs "outBase" + ws.regs "op" + UInt64.ofNat ((ws.regs n).toNat / 255) from by ac_rfl]

-- ── Register-frame lemmas ──────────────────────────────────────────────────

theorem wStoreByte_reg (val r : String) (ws : WState) (fuel : Nat)
    (hr1 : r ≠ "sbAddr") (hr2 : r ≠ "op") :
    ((wStoreByte val).eval fuel ws).regs r = ws.regs r := by
  simp only [wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WArg.eval, WOp.run]
  simp [hr1, hr2]

theorem wEmitToken_reg (litLen tokLo r : String) (ws : WState) (fuel : Nat)
    (hr1 : r ≠ "tokHi") (hr2 : r ≠ "tok") (hr3 : r ≠ "sbAddr") (hr4 : r ≠ "op") :
    ((wEmitToken litLen tokLo).eval fuel ws).regs r = ws.regs r := by
  simp only [wEmitToken, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WArg.eval,
    WOp.run]
  simp [hr1, hr2, hr3, hr4]

theorem lsicBody_reg (n r : String) (hr_c : r ≠ "c255") (hr_l : r ≠ "lsicC") (hr_op : r ≠ "op")
    (hr_sb : r ≠ "sbAddr") (hr_n : r ≠ n) (ws : WState) (f : Nat) :
    ((lsicBody n).eval f ws).regs r = ws.regs r := by
  simp only [lsicBody, wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WArg.eval,
    WOp.run, SCmp.run]
  simp [hr_c, hr_l, hr_op, hr_sb, hr_n]

theorem uwhile_reg (cond r : String) (body : WStmt)
    (hbody : ∀ (st : WState) (f : Nat), (body.eval f st).regs r = st.regs r) :
    ∀ (fuel : Nat) (ws : WState), ((WStmt.uwhile cond body).eval fuel ws).regs r = ws.regs r := by
  intro fuel
  induction fuel with
  | zero => intro ws; simp only [WStmt.eval]
  | succ f ih =>
    intro ws
    simp only [WStmt.eval]
    split
    · rw [ih (body.eval f ws), hbody]
    · rfl

theorem wEmitLSIC_reg (n r : String) (ws : WState) (fuel : Nat)
    (hr_c : r ≠ "c255") (hr_l : r ≠ "lsicC") (hr_op : r ≠ "op")
    (hr_sb : r ≠ "sbAddr") (hr_n : r ≠ n) :
    ((wEmitLSIC n).eval fuel ws).regs r = ws.regs r := by
  show (WStmt.eval fuel (WStmt.seq (WStmt.setp SCmp.ge "lsicC" n (WArg.imm 255))
      (WStmt.seq (WStmt.uwhile "lsicC" (lsicBody n)) (wStoreByte n))) ws).regs r = _
  simp only [WStmt.eval]
  rw [wStoreByte_reg n r _ fuel hr_sb hr_op,
      uwhile_reg "lsicC" r (lsicBody n)
        (fun st f => lsicBody_reg n r hr_c hr_l hr_op hr_sb hr_n st f) fuel]
  simp [WState.setReg, hr_l]

theorem eval_seq (a b : WStmt) (fuel : Nat) (st : WState) :
    (WStmt.seq a b).eval fuel st = b.eval fuel (a.eval fuel st) := by
  simp [WStmt.eval]

theorem bin_reg (o : WOp) (d a r : String) (b : WArg) (st : WState) (fuel : Nat) (hr : r ≠ d) :
    ((WStmt.bin o d a b).eval fuel st).regs r = st.regs r := by
  simp [WStmt.eval, WState.setReg, hr]

theorem setp_reg (c : SCmp) (d a r : String) (b : WArg) (st : WState) (fuel : Nat) (hr : r ≠ d) :
    ((WStmt.setp c d a b).eval fuel st).regs r = st.regs r := by
  simp [WStmt.eval, WState.setReg, hr]

theorem mov_reg (d r : String) (b : WArg) (st : WState) (fuel : Nat) (hr : r ≠ d) :
    ((WStmt.mov d b).eval fuel st).regs r = st.regs r := by
  simp [WStmt.eval, WState.setReg, hr]

theorem coopCopy_reg (dst src len r : String) (st : WState) (fuel : Nat) :
    ((WStmt.coopCopy dst src len).eval fuel st).regs r = st.regs r := by
  simp [WStmt.eval]

theorem eval_uif_pos (cond : String) (t e : WStmt) (ws : WState) (fuel : Nat)
    (h : ws.regs cond = 1) : (WStmt.uif cond t e).eval fuel ws = t.eval fuel ws := by
  simp only [WStmt.eval, h, beq_self_eq_true, if_true]

theorem eval_uif_neg (cond : String) (t e : WStmt) (ws : WState) (fuel : Nat)
    (h : ws.regs cond = 0) : (WStmt.uif cond t e).eval fuel ws = e.eval fuel ws := by
  simp only [WStmt.eval, h]; rfl

-- ── gmem-frame trivialities (only `stgB`/`coopCopy` touch `gmem`) ──────────

theorem bin_gmem (o : WOp) (d a : String) (b : WArg) (st : WState) (fuel : Nat) :
    ((WStmt.bin o d a b).eval fuel st).gmem = st.gmem := by simp [WStmt.eval, WState.setReg]

theorem setp_gmem (c : SCmp) (d a : String) (b : WArg) (st : WState) (fuel : Nat) :
    ((WStmt.setp c d a b).eval fuel st).gmem = st.gmem := by simp [WStmt.eval, WState.setReg]

theorem mov_gmem (d : String) (b : WArg) (st : WState) (fuel : Nat) :
    ((WStmt.mov d b).eval fuel st).gmem = st.gmem := by simp [WStmt.eval, WState.setReg]

-- ══════════════════════════════════════════════════════════════════════════
-- §2. Reproduced from `ScratchCopyCore.lean` (array extensionality via `getD`)
-- ══════════════════════════════════════════════════════════════════════════

theorem putBytesU_size (g : Array UInt8) (base : UInt64) (xs : List UInt8) :
    (putBytesU g base xs).size = g.size := by
  induction xs generalizing g base with
  | nil => rfl
  | cons x xs ih =>
      simp only [putBytesU]
      rw [ih, Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]

private theorem toNat_succ_of_lt {base : UInt64}
    (hnw : base.toNat + 1 < 2 ^ 64) : (base + 1).toNat = base.toNat + 1 := by
  rw [UInt64.toNat_add, UInt64.toNat_one, Nat.mod_eq_of_lt (by omega)]

theorem putBytesU_getD_ge (j : Nat) : ∀ (xs : List UInt8) (g : Array UInt8) (base : UInt64),
    base.toNat + xs.length < 2 ^ 64 → base.toNat + xs.length ≤ j →
    (putBytesU g base xs).getD j 0 = g.getD j 0
  | [], g, base, _, _ => rfl
  | x :: xs, g, base, hnw, hj => by
      simp only [putBytesU, List.length_cons] at *
      have hb1 : (base + 1).toNat = base.toNat + 1 := toNat_succ_of_lt (by omega)
      rw [putBytesU_getD_ge j xs (g.set! base.toNat x) (base + 1) (by omega) (by omega),
        Array.set!_eq_setIfInBounds]
      simp [Array.getElem?_setIfInBounds_ne (show base.toNat ≠ j by omega)]

theorem putBytesU_getD_lt (j : Nat) : ∀ (xs : List UInt8) (g : Array UInt8) (base : UInt64),
    base.toNat + xs.length < 2 ^ 64 → j < base.toNat →
    (putBytesU g base xs).getD j 0 = g.getD j 0
  | [], g, base, _, _ => rfl
  | x :: xs, g, base, hnw, hj => by
      simp only [putBytesU, List.length_cons] at *
      have hb1 : (base + 1).toNat = base.toNat + 1 := toNat_succ_of_lt (by omega)
      rw [putBytesU_getD_lt j xs (g.set! base.toNat x) (base + 1) (by omega) (by omega),
        Array.set!_eq_setIfInBounds]
      simp [Array.getElem?_setIfInBounds_ne (show base.toNat ≠ j by omega)]

/-- `putBytesU` reproduces the written list's bytes inside its write window. -/
theorem putBytesU_getD_win (base : UInt64) : ∀ (xs : List UInt8) (g : Array UInt8) (i : Nat),
    i < xs.length → base.toNat + xs.length ≤ g.size → base.toNat + xs.length < 2 ^ 64 →
    (putBytesU g base xs).getD (base.toNat + i) 0 = xs.getD i 0
  | [], g, i, hi, _, _ => by simp at hi
  | x :: xs, g, i, hi, hsize, hnw => by
      simp only [putBytesU, List.length_cons] at hi hsize hnw ⊢
      have hb1 : (base + 1).toNat = base.toNat + 1 := toNat_succ_of_lt (by omega)
      match i, hi with
      | 0, _ =>
          simp only [Nat.add_zero]
          rw [putBytesU_getD_lt base.toNat xs (g.set! base.toNat x) (base + 1)
                (by omega) (by omega)]
          rw [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?,
              Array.getElem?_setIfInBounds_self_of_lt (by omega : base.toNat < g.size)]
          rfl
      | j + 1, _ =>
          have heq : base.toNat + (j + 1) = (base + 1).toNat + j := by omega
          rw [heq]
          have hsize' : (base + 1).toNat + xs.length ≤ (g.set! base.toNat x).size := by
            rw [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]; omega
          rw [putBytesU_getD_win (base + 1) xs (g.set! base.toNat x) j (by omega) hsize' (by omega)]
          rfl

/-- `copyGmem` leaves every index at or above its destination window untouched. -/
theorem copyGmem_getD_ge (j : Nat) : ∀ (n : Nat) (g : Array UInt8) (dst src : Nat),
    dst + n ≤ j → (copyGmem g dst src n).getD j 0 = g.getD j 0 := by
  intro n
  induction n with
  | zero => intro g dst src _; rfl
  | succ n ih =>
      intro g dst src hj
      show (copyGmem (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) n).getD j 0
        = g.getD j 0
      rw [ih (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) (by omega)]
      rw [Array.getD_eq_getD_getElem?,
          Array.getElem?_setIfInBounds_ne (by omega : dst ≠ j),
          ← Array.getD_eq_getD_getElem?]

/-- Two byte-arrays of equal size that agree under `getD … 0` at every index are equal. -/
theorem array_eq_of_getD {a b : Array UInt8} (hs : a.size = b.size)
    (h : ∀ j, a.getD j 0 = b.getD j 0) : a = b := by
  apply Array.ext hs
  intro i hi hi2
  have hj := h i
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_getElem hi, Array.getElem?_eq_getElem hi2] at hj
  simpa using hj

theorem getD0_map_range (f : Nat → UInt8) (n k : Nat) :
    ((List.range n).map f).getD k 0 = if k < n then f k else 0 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  by_cases h : k < n
  · rw [List.getElem?_range h]; simp [h]
  · rw [List.getElem?_eq_none (by rw [List.length_range]; omega)]; simp [h]

-- ══════════════════════════════════════════════════════════════════════════
-- §3. New: bridging `copyGmem` (the `coopCopy` model) to `putBytesU`
-- ══════════════════════════════════════════════════════════════════════════

/-- A disjoint, in-bounds `copyGmem` is exactly a `putBytesU` of the source run,
    read off the *pre-copy* array `g` (source and destination don't alias). -/
theorem copyGmem_eq_putBytesU (g : Array UInt8) (dst src len : Nat)
    (hdisj : dst + len ≤ src ∨ src + len ≤ dst) (hsize : dst + len ≤ g.size)
    (hnw : dst + len < 2 ^ 64) :
    copyGmem g dst src len
      = putBytesU g (UInt64.ofNat dst) ((List.range len).map (fun i => g.getD (src + i) 0)) := by
  have hbase : (UInt64.ofNat dst).toNat = dst := by
    rw [UInt64.toNat_ofNat']; omega
  have hmaplen : ((List.range len).map (fun i => g.getD (src + i) 0)).length = len := by simp
  apply array_eq_of_getD
  · rw [copyGmem_size, putBytesU_size]
  · intro j
    by_cases hjd : j < dst
    · rw [copyGmem_getD_lt j len g dst src hjd,
          putBytesU_getD_lt j _ g (UInt64.ofNat dst) (by rw [hbase, hmaplen]; omega)
            (by rw [hbase]; omega)]
    · by_cases hjl : j < dst + len
      · have hk : j - dst < len := by omega
        have hjeq : dst + (j - dst) = j := by omega
        rw [← hjeq, copyGmem_getD len g dst src (j - dst) hk hsize hdisj,
            show dst + (j - dst) = (UInt64.ofNat dst).toNat + (j - dst) from by rw [hbase],
            putBytesU_getD_win (UInt64.ofNat dst) _ g (j - dst) (by rw [hmaplen]; omega)
              (by rw [hbase, hmaplen]; omega) (by rw [hbase, hmaplen]; omega),
            getD0_map_range]
        simp [hk]
      · rw [copyGmem_getD_ge j len g dst src (by omega),
            putBytesU_getD_ge j _ g (UInt64.ofNat dst) (by rw [hbase, hmaplen]; omega)
              (by rw [hbase, hmaplen]; omega)]

-- ══════════════════════════════════════════════════════════════════════════
-- §4. New: content-aware length-extension `uif` (op advance + frame + gmem)
-- ══════════════════════════════════════════════════════════════════════════

theorem eval_uifExt_content (P E src : String) (ws : WState) (fuel : Nat)
    (hP : ws.regs P = (if UInt64.ofNat 15 ≤ ws.regs src then 1 else 0))
    (hEc : E ≠ "c255") (hEl : E ≠ "lsicC") (hEop : E ≠ "op") (hEsb : E ≠ "sbAddr")
    (hEob : E ≠ "outBase") (_hsrcE : src ≠ E)
    (hfuel : (ws.regs src).toNat / 255 < fuel) :
    ((WStmt.uif P (WStmt.seq (WStmt.bin WOp.sub E src (WArg.imm 15)) (wEmitLSIC E))
        WStmt.skip).eval fuel ws).regs "op"
          = ws.regs "op" + UInt64.ofNat (encNib (ws.regs src).toNat).length
    ∧ (∀ r, r ≠ E → r ≠ "c255" → r ≠ "lsicC" → r ≠ "op" → r ≠ "sbAddr" →
        ((WStmt.uif P (WStmt.seq (WStmt.bin WOp.sub E src (WArg.imm 15)) (wEmitLSIC E))
          WStmt.skip).eval fuel ws).regs r = ws.regs r)
    ∧ ((WStmt.uif P (WStmt.seq (WStmt.bin WOp.sub E src (WArg.imm 15)) (wEmitLSIC E))
        WStmt.skip).eval fuel ws).gmem
        = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op") (encNib (ws.regs src).toNat) := by
  have h15 : (UInt64.ofNat 15).toNat = 15 := by decide
  by_cases hbig : UInt64.ofNat 15 ≤ ws.regs src
  · have hP1 : ws.regs P = 1 := by rw [hP, if_pos hbig]
    have hVge : 15 ≤ (ws.regs src).toNat := by
      rw [UInt64.le_iff_toNat_le, h15] at hbig; exact hbig
    rw [eval_uif_pos _ _ _ _ _ hP1, eval_seq]
    have hs1E : ((WStmt.bin WOp.sub E src (WArg.imm 15)).eval fuel ws).regs E
        = ws.regs src - UInt64.ofNat 15 := by simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    have hs1EN : (((WStmt.bin WOp.sub E src (WArg.imm 15)).eval fuel ws).regs E).toNat
        = (ws.regs src).toNat - 15 := by
      rw [hs1E, UInt64.toNat_sub_of_le _ _ hbig, h15]
    have hencNib : encNib (ws.regs src).toNat = ext ((ws.regs src).toNat - 15) := by
      rw [encNib, if_neg (by omega)]
    have hs1ob : ((WStmt.bin WOp.sub E src (WArg.imm 15)).eval fuel ws).regs "outBase"
        = ws.regs "outBase" := bin_reg _ _ _ _ _ ws fuel (Ne.symm hEob)
    have hs1op : ((WStmt.bin WOp.sub E src (WArg.imm 15)).eval fuel ws).regs "op"
        = ws.regs "op" := bin_reg _ _ _ _ _ ws fuel (Ne.symm hEop)
    have hs1gm : ((WStmt.bin WOp.sub E src (WArg.imm 15)).eval fuel ws).gmem = ws.gmem :=
      bin_gmem _ _ _ _ ws fuel
    obtain ⟨lop, lob, lg⟩ := eval_wEmitLSIC E
      ((WStmt.bin WOp.sub E src (WArg.imm 15)).eval fuel ws) fuel hEc hEl hEop hEob hEsb
      (by rw [hs1EN]; omega)
    refine ⟨?_, ?_, ?_⟩
    · rw [lop, bin_reg WOp.sub E src "op" (WArg.imm 15) ws fuel (Ne.symm hEop), hs1EN, ← hencNib]
    · intro r hrE hrc hrl hrop hrsb
      rw [wEmitLSIC_reg E r _ fuel hrc hrl hrop hrsb hrE,
          bin_reg WOp.sub E src r (WArg.imm 15) ws fuel hrE]
    · rw [lg, hs1gm, hs1ob, hs1op, hs1EN, ← hencNib]
  · have hP0 : ws.regs P = 0 := by rw [hP, if_neg hbig]
    have hVlt : (ws.regs src).toNat < 15 := by
      rw [UInt64.le_iff_toNat_le, h15] at hbig; omega
    rw [eval_uif_neg _ _ _ _ _ hP0]
    have hnil : encNib (ws.regs src).toNat = [] := by rw [encNib, if_pos hVlt]
    refine ⟨?_, ?_, ?_⟩
    · simp only [WStmt.eval, hnil, List.length_nil]
      show ws.regs "op" = ws.regs "op" + UInt64.ofNat 0
      simp
    · intro r _ _ _ _ _; simp only [WStmt.eval]
    · simp only [WStmt.eval, hnil]; rfl

-- ══════════════════════════════════════════════════════════════════════════
-- §4b. New: small bit-manipulation lemmas for `wEmitMatchSeq`'s offset bytes
-- ══════════════════════════════════════════════════════════════════════════

theorem min15_toNat (x : UInt64) : (WOp.run WOp.min x (UInt64.ofNat 15)).toNat = min x.toNat 15 := by
  simp only [WOp.run]
  have h15 : (UInt64.ofNat 15).toNat = 15 := by decide
  split
  · next h => rw [UInt64.le_iff_toNat_le, h15] at h; omega
  · next h => rw [UInt64.le_iff_toNat_le, h15] at h; omega

/-- The low byte of `off & 255` is `off % 256` — the LZ4 offset's low byte. -/
theorem offLo_byte (x : UInt64) : (x &&& UInt64.ofNat 255).toUInt8 = UInt8.ofNat (x.toNat % 256) := by
  have h1 : (x &&& UInt64.ofNat 255).toNat = x.toNat % 256 := by
    rw [UInt64.toNat_and, show (UInt64.ofNat 255).toNat = 255 from by decide,
        show (255 : Nat) = 2 ^ 8 - 1 from by decide]
    exact Nat.and_two_pow_sub_one_eq_mod x.toNat 8
  show (x &&& UInt64.ofNat 255).toNat.toUInt8 = _
  rw [h1]

/-- The low byte of `(off >>> 8) & 255` is `off / 256` — the LZ4 offset's high byte
    (`UInt8.ofNat` already truncates mod 256, so the extra `% 256` is redundant). -/
theorem offHi_byte (x : UInt64) :
    ((x >>> UInt64.ofNat 8) &&& UInt64.ofNat 255).toUInt8 = UInt8.ofNat (x.toNat / 256) := by
  have h1 : ((x >>> UInt64.ofNat 8) &&& UInt64.ofNat 255).toNat = (x.toNat / 256) % 256 := by
    rw [UInt64.toNat_and, UInt64.toNat_shiftRight, show (UInt64.ofNat 8).toNat % 64 = 8 from by decide,
        show (UInt64.ofNat 255).toNat = 255 from by decide, show (255 : Nat) = 2 ^ 8 - 1 from by decide,
        Nat.shiftRight_eq_div_pow]
    exact Nat.and_two_pow_sub_one_eq_mod (x.toNat / 2 ^ 8) 8
  have h2 : UInt8.ofNat ((x.toNat / 256) % 256) = UInt8.ofNat (x.toNat / 256) := by
    apply UInt8.toNat_inj.mp
    rw [UInt8.toNat_ofNat', UInt8.toNat_ofNat', Nat.mod_mod]
  show ((x >>> UInt64.ofNat 8) &&& UInt64.ofNat 255).toNat.toUInt8 = _
  rw [h1, show ((x.toNat / 256) % 256).toUInt8 = UInt8.ofNat ((x.toNat / 256) % 256) from rfl, h2]

-- ══════════════════════════════════════════════════════════════════════════
-- §5. `wEmitFinalSeq` — full byte-content spec
-- ══════════════════════════════════════════════════════════════════════════

/-- **`wEmitFinalSeq` byte-content**: the bytes written are exactly
    `LZ4.encodeFinal` of the literal run read from `ws.gmem` at
    `inBase+litStart .. inBase+litStart+litLen`. -/
theorem eval_wEmitFinalSeq_content (litStart litLen : String) (ws : WState) (fuel : Nat)
    (h1 : litLen ≠ "zero") (h2 : litLen ≠ "tokHi") (h3 : litLen ≠ "tok")
    (h4 : litLen ≠ "sbAddr") (h5 : litLen ≠ "op") (h6 : litLen ≠ "pLitBigF")
    (h7 : litLen ≠ "litExtraF") (h8 : litLen ≠ "c255") (h9 : litLen ≠ "lsicC")
    (h10 : litLen ≠ "cpDstF") (h11 : litLen ≠ "cpSrcF")
    (hs1 : litStart ≠ "zero") (hs2 : litStart ≠ "tokHi") (hs3 : litStart ≠ "tok")
    (hs4 : litStart ≠ "sbAddr") (hs5 : litStart ≠ "op") (hs6 : litStart ≠ "pLitBigF")
    (hs7 : litStart ≠ "litExtraF") (hs8 : litStart ≠ "c255") (hs9 : litStart ≠ "lsicC")
    (hs10 : litStart ≠ "cpDstF")
    (hfuel : (ws.regs litLen).toNat / 255 < fuel)
    -- disjointness / no-overflow / in-bounds, stated on the plain `Nat` quantities:
    (hnw1 : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
              + (encNib (ws.regs litLen).toNat).length + (ws.regs litLen).toNat < 2 ^ 64)
    (hnw2 : (ws.regs "inBase").toNat + (ws.regs litStart).toNat + (ws.regs litLen).toNat < 2 ^ 64)
    (hsize : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
              + (encNib (ws.regs litLen).toNat).length + (ws.regs litLen).toNat ≤ ws.gmem.size)
    (hdisj : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
              + (encNib (ws.regs litLen).toNat).length + (ws.regs litLen).toNat
              ≤ (ws.regs "inBase").toNat + (ws.regs litStart).toNat
          ∨ (ws.regs "inBase").toNat + (ws.regs litStart).toNat + (ws.regs litLen).toNat
              ≤ (ws.regs "outBase").toNat + (ws.regs "op").toNat) :
    ((wEmitFinalSeq litStart litLen).eval fuel ws).gmem
      = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
          (LZ4.encodeFinal ((List.range (ws.regs litLen).toNat).map
            (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + (ws.regs litStart).toNat + i) 0))) := by
  simp only [wEmitFinalSeq, wseq, eval_seq]
  -- s1 : after `mov zero 0`
  generalize hs1' : (WStmt.mov "zero" (WArg.imm 0)).eval fuel ws = s1
  have s1_op : s1.regs "op" = ws.regs "op" := by rw [← hs1']; exact mov_reg _ _ _ ws fuel (by decide)
  have s1_ll : s1.regs litLen = ws.regs litLen := by rw [← hs1']; exact mov_reg _ _ _ ws fuel h1
  have s1_ls : s1.regs litStart = ws.regs litStart := by rw [← hs1']; exact mov_reg _ _ _ ws fuel hs1
  have s1_ob : s1.regs "outBase" = ws.regs "outBase" := by
    rw [← hs1']; exact mov_reg _ _ _ ws fuel (by decide)
  have s1_ib : s1.regs "inBase" = ws.regs "inBase" := by
    rw [← hs1']; exact mov_reg _ _ _ ws fuel (by decide)
  have s1_gm : s1.gmem = ws.gmem := by rw [← hs1']; exact mov_gmem _ _ ws fuel
  have s1_z : (s1.regs "zero").toNat < 16 := by
    rw [← hs1']; simp [WStmt.eval, WState.setReg, WArg.eval]
  -- s2 : after `wEmitToken litLen zero`
  generalize hs2' : (wEmitToken litLen "zero").eval fuel s1 = s2
  obtain ⟨t_op, t_ob, t_gm⟩ := eval_wEmitToken litLen "zero" s1 fuel s1_z (by decide) (by decide)
  have s2_op : s2.regs "op" = ws.regs "op" + 1 := by rw [← hs2', t_op, s1_op]
  have s2_ll : s2.regs litLen = ws.regs litLen := by
    rw [← hs2', wEmitToken_reg litLen "zero" litLen s1 fuel h2 h3 h4 h5, s1_ll]
  have s2_ls : s2.regs litStart = ws.regs litStart := by
    rw [← hs2', wEmitToken_reg litLen "zero" litStart s1 fuel hs2 hs3 hs4 hs5, s1_ls]
  have s2_ob : s2.regs "outBase" = ws.regs "outBase" := by rw [← hs2', t_ob, s1_ob]
  have s2_ib : s2.regs "inBase" = ws.regs "inBase" := by
    rw [← hs2', wEmitToken_reg litLen "zero" "inBase" s1 fuel (by decide) (by decide) (by decide)
      (by decide), s1_ib]
  have tokByte : (UInt8.ofNat ((min (s1.regs litLen).toNat 15) * 16 + (s1.regs "zero").toNat))
      = UInt8.ofNat (min (ws.regs litLen).toNat 15 * 16) := by
    rw [s1_ll]
    have : (s1.regs "zero").toNat = 0 := by
      rw [← hs1']; simp [WStmt.eval, WState.setReg, WArg.eval]
    rw [this]
    simp
  have s2_gm : s2.gmem = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
      [UInt8.ofNat (min (ws.regs litLen).toNat 15 * 16)] := by
    rw [← hs2', t_gm, s1_gm, s1_ob, s1_op, tokByte]
    simp [putBytesU]
  -- s3 : after `setp .ge pLitBigF litLen 15`
  generalize hs3' : (WStmt.setp SCmp.ge "pLitBigF" litLen (WArg.imm 15)).eval fuel s2 = s3
  have s3_op : s3.regs "op" = ws.regs "op" + 1 := by
    rw [← hs3', setp_reg _ _ _ _ _ s2 fuel (by decide), s2_op]
  have s3_ll : s3.regs litLen = ws.regs litLen := by
    rw [← hs3', setp_reg _ _ _ _ _ s2 fuel h6, s2_ll]
  have s3_ls : s3.regs litStart = ws.regs litStart := by
    rw [← hs3', setp_reg _ _ _ _ _ s2 fuel hs6, s2_ls]
  have s3_ob : s3.regs "outBase" = ws.regs "outBase" := by
    rw [← hs3', setp_reg _ _ _ _ _ s2 fuel (by decide), s2_ob]
  have s3_ib : s3.regs "inBase" = ws.regs "inBase" := by
    rw [← hs3', setp_reg _ _ _ _ _ s2 fuel (by decide), s2_ib]
  have s3_p : s3.regs "pLitBigF" = (if UInt64.ofNat 15 ≤ s3.regs litLen then 1 else 0) := by
    rw [← hs3']; simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run, h6]
  have s3_gm : s3.gmem = s2.gmem := by rw [← hs3']; exact setp_gmem _ _ _ _ s2 fuel
  -- s4 : after the length-extension `uif`
  obtain ⟨u_op, u_fr, u_gm⟩ := eval_uifExt_content "pLitBigF" "litExtraF" litLen s3 fuel s3_p
    (by decide) (by decide) (by decide) (by decide) (by decide) h7
    (by rw [s3_ll]; exact hfuel)
  generalize hs4' : (WStmt.uif "pLitBigF"
      (WStmt.seq (WStmt.bin WOp.sub "litExtraF" litLen (WArg.imm 15)) (wEmitLSIC "litExtraF"))
      WStmt.skip).eval fuel s3 = s4
  have s4_op : s4.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs4', u_op, s3_op, s3_ll]
  have s4_ll : s4.regs litLen = ws.regs litLen := by
    rw [← hs4', u_fr litLen h7 h8 h9 h5 h4, s3_ll]
  have s4_ls : s4.regs litStart = ws.regs litStart := by
    rw [← hs4', u_fr litStart hs7 hs8 hs9 hs5 hs4, s3_ls]
  have s4_ob : s4.regs "outBase" = ws.regs "outBase" := by
    rw [← hs4', u_fr "outBase" (by decide) (by decide) (by decide) (by decide) (by decide), s3_ob]
  have s4_ib : s4.regs "inBase" = ws.regs "inBase" := by
    rw [← hs4', u_fr "inBase" (by decide) (by decide) (by decide) (by decide) (by decide), s3_ib]
  have s4_gm : s4.gmem = putBytesU s3.gmem (ws.regs "outBase" + (ws.regs "op" + 1))
      (encNib (ws.regs litLen).toNat) := by
    rw [← hs4', u_gm, s3_ob, s3_op, s3_ll]
  -- combine s2/s3/s4 gmem into a single `putBytesU` from `ws.gmem`
  have s4_gm' : s4.gmem = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
      ([UInt8.ofNat (min (ws.regs litLen).toNat 15 * 16)] ++ encNib (ws.regs litLen).toNat) := by
    rw [s4_gm, s3_gm, s2_gm, putBytesU_append,
        show (ws.regs "outBase" + ws.regs "op"
              + UInt64.ofNat ([UInt8.ofNat (min (ws.regs litLen).toNat 15 * 16)] : List UInt8).length)
            = ws.regs "outBase" + (ws.regs "op" + 1) from by
      show ws.regs "outBase" + ws.regs "op" + 1 = ws.regs "outBase" + (ws.regs "op" + 1)
      ac_rfl]
  -- s5 : after `bin add cpDstF outBase op`
  generalize hs5' : (WStmt.bin WOp.add "cpDstF" "outBase" (WArg.reg "op")).eval fuel s4 = s5
  have s5_op : s5.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs5', bin_reg _ _ _ _ _ s4 fuel (by decide), s4_op]
  have s5_ll : s5.regs litLen = ws.regs litLen := by
    rw [← hs5', bin_reg _ _ _ _ _ s4 fuel h10, s4_ll]
  have s5_ls : s5.regs litStart = ws.regs litStart := by
    rw [← hs5', bin_reg _ _ _ _ _ s4 fuel hs10, s4_ls]
  have s5_ob : s5.regs "outBase" = ws.regs "outBase" := by
    rw [← hs5', bin_reg _ _ _ _ _ s4 fuel (by decide), s4_ob]
  have s5_ib : s5.regs "inBase" = ws.regs "inBase" := by
    rw [← hs5', bin_reg _ _ _ _ _ s4 fuel (by decide), s4_ib]
  have s5_cpDstF : s5.regs "cpDstF" = ws.regs "outBase" + (ws.regs "op" + 1
      + UInt64.ofNat (encNib (ws.regs litLen).toNat).length) := by
    have hraw : s5.regs "cpDstF" = s4.regs "outBase" + s4.regs "op" := by
      rw [← hs5']; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    rw [hraw, s4_ob, s4_op]
  have s5_gm : s5.gmem = s4.gmem := by rw [← hs5']; exact bin_gmem _ _ _ _ s4 fuel
  -- s6 : after `bin add cpSrcF inBase litStart`
  generalize hs6' : (WStmt.bin WOp.add "cpSrcF" "inBase" (WArg.reg litStart)).eval fuel s5 = s6
  have s6_op : s6.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs6', bin_reg _ _ _ _ _ s5 fuel (by decide), s5_op]
  have s6_ll : s6.regs litLen = ws.regs litLen := by
    rw [← hs6', bin_reg _ _ _ _ _ s5 fuel h11, s5_ll]
  have s6_ob : s6.regs "outBase" = ws.regs "outBase" := by
    rw [← hs6', bin_reg _ _ _ _ _ s5 fuel (by decide), s5_ob]
  have s6_cpDstF : s6.regs "cpDstF" = ws.regs "outBase" + (ws.regs "op" + 1
      + UInt64.ofNat (encNib (ws.regs litLen).toNat).length) := by
    rw [← hs6', bin_reg _ _ _ _ _ s5 fuel (by decide), s5_cpDstF]
  have s6_cpSrcF : s6.regs "cpSrcF" = ws.regs "inBase" + ws.regs litStart := by
    have hraw : s6.regs "cpSrcF" = s5.regs "inBase" + s5.regs litStart := by
      rw [← hs6']; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    rw [hraw, s5_ib, s5_ls]
  have s6_gm : s6.gmem = s4.gmem := by rw [← hs6', bin_gmem _ _ _ _ s5 fuel, s5_gm]
  -- s7 : after the cooperative literal copy
  generalize hs7' : (WStmt.coopCopy "cpDstF" "cpSrcF" litLen).eval fuel s6 = s7
  have s7_op : s7.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs7', coopCopy_reg _ _ _ _ s6 fuel, s6_op]
  have s7_ll : s7.regs litLen = ws.regs litLen := by
    rw [← hs7', coopCopy_reg _ _ _ _ s6 fuel, s6_ll]
  -- numeric shorthands (`set` is unavailable without Mathlib; `generalize` abstracts
  -- the goal the same way and keeps the naming equation for `omega`)
  generalize hobN : (ws.regs "outBase").toNat = obN
  generalize hopN : (ws.regs "op").toNat = opN
  generalize hllN : (ws.regs litLen).toNat = llN
  generalize hencL : (encNib llN).length = encL
  generalize hibN : (ws.regs "inBase").toNat = ibN
  generalize hlsN : (ws.regs litStart).toNat = lsN
  -- fold the same abbreviations into the numeric side-condition hypotheses
  -- (`generalize` only rewrites the goal, unlike Mathlib's `set`)
  simp only [hllN] at hnw1 hnw2 hsize hdisj
  simp only [hencL] at hnw1 hsize hdisj
  simp only [hobN, hopN] at hnw1 hsize hdisj
  simp only [hibN, hlsN] at hnw2 hdisj
  rw [hllN] at s4_gm'
  have hcpDstN : (s6.regs "cpDstF").toNat = obN + opN + 1 + encL := by
    rw [s6_cpDstF, hllN, hencL, UInt64.toNat_add]
    have e1 : (ws.regs "op" + 1 + UInt64.ofNat encL).toNat = opN + 1 + encL := by
      have h1' : (ws.regs "op" + 1).toNat = opN + 1 := by
        rw [UInt64.toNat_add, show (1 : UInt64).toNat = 1 from rfl, Nat.mod_eq_of_lt (by omega), hopN]
      rw [UInt64.toNat_add, h1', UInt64.toNat_ofNat' (n := encL), Nat.mod_eq_of_lt (show encL < 2 ^ 64 by omega),
          Nat.mod_eq_of_lt (by omega)]
    rw [e1, Nat.mod_eq_of_lt (by omega)]
    omega
  have hcpSrcN : (s6.regs "cpSrcF").toNat = ibN + lsN := by
    rw [s6_cpSrcF, UInt64.toNat_add, Nat.mod_eq_of_lt (by omega)]
    omega
  have s7_gm : s7.gmem = putBytesU s6.gmem (UInt64.ofNat (obN + opN + 1 + encL))
      ((List.range llN).map (fun i => s6.gmem.getD (ibN + lsN + i) 0)) := by
    have hcc : ((WStmt.coopCopy "cpDstF" "cpSrcF" litLen).eval fuel s6).gmem
        = copyGmem s6.gmem (s6.regs "cpDstF").toNat (s6.regs "cpSrcF").toNat (s6.regs litLen).toNat := by
      simp [WStmt.eval]
    rw [← hs7', hcc, hcpDstN, hcpSrcN, s6_ll, hllN]
    apply copyGmem_eq_putBytesU s6.gmem (obN + opN + 1 + encL) (ibN + lsN) llN
    · rcases hdisj with h | h
      · left; omega
      · right; omega
    · rw [s6_gm, s4_gm', putBytesU_size]; exact hsize
    · omega
  have hb0 : (ws.regs "outBase" + ws.regs "op").toNat = obN + opN := by
    rw [UInt64.toNat_add, Nat.mod_eq_of_lt (by omega)]
    omega
  have hxs0len : ([UInt8.ofNat (min llN 15 * 16)] ++ encNib llN).length = 1 + encL := by
    simp [hencL]; omega
  -- the literal bytes read from `s6.gmem` agree with those read from `ws.gmem`
  -- (the token+ext write window is disjoint from the literal-read window)
  have hliteq : ∀ i, i < llN → s6.gmem.getD (ibN + lsN + i) 0 = ws.gmem.getD (ibN + lsN + i) 0 := by
    intro i hi
    rw [s6_gm, s4_gm']
    rcases hdisj with h | h
    · exact putBytesU_getD_ge (ibN + lsN + i) _ ws.gmem (ws.regs "outBase" + ws.regs "op")
        (by rw [hb0, hxs0len]; omega) (by rw [hb0, hxs0len]; omega)
    · exact putBytesU_getD_lt (ibN + lsN + i) _ ws.gmem (ws.regs "outBase" + ws.regs "op")
        (by rw [hb0, hxs0len]; omega) (by rw [hb0]; omega)
  have s7_gm' : s7.gmem = putBytesU s6.gmem (UInt64.ofNat (obN + opN + 1 + encL))
      ((List.range llN).map (fun i => ws.gmem.getD (ibN + lsN + i) 0)) := by
    rw [s7_gm]
    congr 1
    apply List.map_congr_left
    intro i hi
    exact hliteq i (List.mem_range.mp hi)
  -- s8 : final `bin add op op litLen` (gmem untouched)
  have s8_gm : ((WStmt.bin WOp.add "op" "op" (WArg.reg litLen)).eval fuel s7).gmem = s7.gmem :=
    bin_gmem _ _ _ _ s7 fuel
  have hcombBase : ws.regs "outBase" + ws.regs "op"
        + UInt64.ofNat ([UInt8.ofNat (min llN 15 * 16)] ++ encNib llN).length
      = UInt64.ofNat (obN + opN + 1 + encL) := by
    rw [← UInt64.toNat_inj, UInt64.toNat_add, hxs0len, hb0,
        UInt64.toNat_ofNat' (n := 1 + encL), Nat.mod_eq_of_lt (show (1 + encL) < 2 ^ 64 by omega),
        Nat.mod_eq_of_lt (by omega),
        UInt64.toNat_ofNat' (n := obN + opN + 1 + encL),
        Nat.mod_eq_of_lt (show obN + opN + 1 + encL < 2 ^ 64 by omega)]
    omega
  have hliteralLen :
      ((List.range llN).map (fun i => ws.gmem.getD (ibN + lsN + i) 0)).length = llN := by simp
  have hxs_eq : LZ4.encodeFinal ((List.range llN).map (fun i => ws.gmem.getD (ibN + lsN + i) 0))
      = ([UInt8.ofNat (min llN 15 * 16)] ++ encNib llN)
          ++ (List.range llN).map (fun i => ws.gmem.getD (ibN + lsN + i) 0) := by
    simp only [LZ4.encodeFinal, hliteralLen, List.cons_append, List.nil_append]
  rw [s8_gm, s7_gm', s6_gm, s4_gm', hxs_eq, ← hcombBase, ← putBytesU_append]

-- ══════════════════════════════════════════════════════════════════════════
-- §6. Full byte content for `wEmitMatchSeq` (token, litLSIC, literals, offset
--     bytes, matchLSIC = exactly `LZ4.encodeSeq`).
-- ══════════════════════════════════════════════════════════════════════════

/-- Full byte-content spec for a match sequence. The emitted bytes are exactly
    `LZ4.encodeSeq ⟨lits, off, ml⟩`, where `lits` is the literal run read straight
    from `ws.gmem` at `inBase+litStart`, `off = ws.regs off`, `ml = ws.regs ml`. -/
theorem eval_wEmitMatchSeq_content (litStart litLen off ml : String) (ws : WState) (fuel : Nat)
    (hl_mlm : litLen ≠ "mlm") (hl_tokLo : litLen ≠ "tokLo") (hl_tokHi : litLen ≠ "tokHi")
    (hl_tok : litLen ≠ "tok") (hl_sb : litLen ≠ "sbAddr") (hl_op : litLen ≠ "op")
    (hl_pLit : litLen ≠ "pLitBig") (hl_litE : litLen ≠ "litExtra") (hl_c : litLen ≠ "c255")
    (hl_l : litLen ≠ "lsicC") (hl_cpD : litLen ≠ "cpDst") (hl_cpS : litLen ≠ "cpSrc")
    (hl_offLo : litLen ≠ "offLo") (hl_offHi : litLen ≠ "offHi") (hl_pMat : litLen ≠ "pMatBig")
    (hl_matE : litLen ≠ "matExtra") (hl_ob : litLen ≠ "outBase") (hl_ib : litLen ≠ "inBase")
    (hs_mlm : litStart ≠ "mlm") (hs_tokLo : litStart ≠ "tokLo") (hs_tokHi : litStart ≠ "tokHi")
    (hs_tok : litStart ≠ "tok") (hs_sb : litStart ≠ "sbAddr") (hs_op : litStart ≠ "op")
    (hs_pLit : litStart ≠ "pLitBig") (hs_litE : litStart ≠ "litExtra") (hs_c : litStart ≠ "c255")
    (hs_l : litStart ≠ "lsicC") (hs_cpD : litStart ≠ "cpDst") (hs_cpS : litStart ≠ "cpSrc")
    (hs_offLo : litStart ≠ "offLo") (hs_offHi : litStart ≠ "offHi") (hs_pMat : litStart ≠ "pMatBig")
    (hs_matE : litStart ≠ "matExtra") (hs_ob : litStart ≠ "outBase") (hs_ib : litStart ≠ "inBase")
    (ho_mlm : off ≠ "mlm") (ho_tokLo : off ≠ "tokLo") (ho_tokHi : off ≠ "tokHi")
    (ho_tok : off ≠ "tok") (ho_sb : off ≠ "sbAddr") (ho_op : off ≠ "op")
    (ho_pLit : off ≠ "pLitBig") (ho_litE : off ≠ "litExtra") (ho_c : off ≠ "c255")
    (ho_l : off ≠ "lsicC") (ho_cpD : off ≠ "cpDst") (ho_cpS : off ≠ "cpSrc")
    (ho_offLo : off ≠ "offLo") (ho_offHi : off ≠ "offHi") (ho_pMat : off ≠ "pMatBig")
    (ho_matE : off ≠ "matExtra") (ho_ob : off ≠ "outBase") (ho_ib : off ≠ "inBase")
    (hm_mlm : ml ≠ "mlm") (hm_tokLo : ml ≠ "tokLo") (hm_tokHi : ml ≠ "tokHi")
    (hm_tok : ml ≠ "tok") (hm_sb : ml ≠ "sbAddr") (hm_op : ml ≠ "op")
    (hm_pLit : ml ≠ "pLitBig") (hm_litE : ml ≠ "litExtra") (hm_c : ml ≠ "c255")
    (hm_l : ml ≠ "lsicC") (hm_cpD : ml ≠ "cpDst") (hm_cpS : ml ≠ "cpSrc")
    (hm_offLo : ml ≠ "offLo") (hm_offHi : ml ≠ "offHi") (hm_pMat : ml ≠ "pMatBig")
    (hm_matE : ml ≠ "matExtra") (hm_ob : ml ≠ "outBase") (hm_ib : ml ≠ "inBase")
    (hml4 : 4 ≤ (ws.regs ml).toNat)
    (hfuelL : (ws.regs litLen).toNat / 255 < fuel)
    (hfuelM : ((ws.regs ml).toNat - 4) / 255 < fuel)
    -- the full encoded sequence fits below `2^64` (the "output address space" precondition):
    (hnwFull : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
                + (encNib (ws.regs litLen).toNat).length + (ws.regs litLen).toNat + 2
                + (encNib ((ws.regs ml).toNat - 4)).length < 2 ^ 64)
    (hnw2 : (ws.regs "inBase").toNat + (ws.regs litStart).toNat + (ws.regs litLen).toNat < 2 ^ 64)
    (hsize : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
               + (encNib (ws.regs litLen).toNat).length + (ws.regs litLen).toNat ≤ ws.gmem.size)
    (hdisj : (ws.regs "outBase").toNat + (ws.regs "op").toNat + 1
               + (encNib (ws.regs litLen).toNat).length + (ws.regs litLen).toNat
               ≤ (ws.regs "inBase").toNat + (ws.regs litStart).toNat
           ∨ (ws.regs "inBase").toNat + (ws.regs litStart).toNat + (ws.regs litLen).toNat
               ≤ (ws.regs "outBase").toNat + (ws.regs "op").toNat) :
    ((wEmitMatchSeq litStart litLen off ml).eval fuel ws).gmem
      = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
          (LZ4.encodeSeq ⟨(List.range (ws.regs litLen).toNat).map
            (fun i => ws.gmem.getD ((ws.regs "inBase").toNat + (ws.regs litStart).toNat + i) 0),
            (ws.regs off).toNat, (ws.regs ml).toNat⟩) := by
  simp only [wEmitMatchSeq, wseq, eval_seq]
  -- a1 : `bin sub mlm ml 4`
  generalize ha1 : (WStmt.bin WOp.sub "mlm" ml (WArg.imm 4)).eval fuel ws = a1
  have a1_op : a1.regs "op" = ws.regs "op" := by rw [← ha1]; exact bin_reg _ _ _ _ _ ws fuel (by decide)
  have a1_ob : a1.regs "outBase" = ws.regs "outBase" := by rw [← ha1]; exact bin_reg _ _ _ _ _ ws fuel (by decide)
  have a1_ib : a1.regs "inBase" = ws.regs "inBase" := by rw [← ha1]; exact bin_reg _ _ _ _ _ ws fuel (by decide)
  have a1_ll : a1.regs litLen = ws.regs litLen := by rw [← ha1]; exact bin_reg _ _ _ _ _ ws fuel hl_mlm
  have a1_ls : a1.regs litStart = ws.regs litStart := by rw [← ha1]; exact bin_reg _ _ _ _ _ ws fuel hs_mlm
  have a1_off : a1.regs off = ws.regs off := by rw [← ha1]; exact bin_reg _ _ _ _ _ ws fuel ho_mlm
  have a1_gm : a1.gmem = ws.gmem := by rw [← ha1]; exact bin_gmem _ _ _ _ ws fuel
  have a1_mlm : a1.regs "mlm" = ws.regs ml - UInt64.ofNat 4 := by
    rw [← ha1]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
  have a1_mlmN : (a1.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [a1_mlm, UInt64.toNat_sub_of_le, show (UInt64.ofNat 4).toNat = 4 from by decide]
    rw [UInt64.le_iff_toNat_le, show (UInt64.ofNat 4).toNat = 4 from by decide]; exact hml4
  -- a2 : `bin min tokLo mlm 15`
  generalize ha2 : (WStmt.bin WOp.min "tokLo" "mlm" (WArg.imm 15)).eval fuel a1 = a2
  have a2_op : a2.regs "op" = ws.regs "op" := by rw [← ha2, bin_reg _ _ _ _ _ a1 fuel (by decide), a1_op]
  have a2_ob : a2.regs "outBase" = ws.regs "outBase" := by rw [← ha2, bin_reg _ _ _ _ _ a1 fuel (by decide), a1_ob]
  have a2_ib : a2.regs "inBase" = ws.regs "inBase" := by rw [← ha2, bin_reg _ _ _ _ _ a1 fuel (by decide), a1_ib]
  have a2_ll : a2.regs litLen = ws.regs litLen := by rw [← ha2, bin_reg _ _ _ _ _ a1 fuel hl_tokLo, a1_ll]
  have a2_ls : a2.regs litStart = ws.regs litStart := by rw [← ha2, bin_reg _ _ _ _ _ a1 fuel hs_tokLo, a1_ls]
  have a2_off : a2.regs off = ws.regs off := by rw [← ha2, bin_reg _ _ _ _ _ a1 fuel ho_tokLo, a1_off]
  have a2_gm : a2.gmem = ws.gmem := by rw [← ha2, bin_gmem _ _ _ _ a1 fuel, a1_gm]
  have a2_tok : (a2.regs "tokLo").toNat < 16 := by
    have : a2.regs "tokLo" = WOp.run WOp.min (a1.regs "mlm") (UInt64.ofNat 15) := by
      rw [← ha2]; simp [WStmt.eval, WState.setReg, WArg.eval]
    rw [this, min15_toNat]; omega
  have a2_tokVal : (a2.regs "tokLo").toNat = min ((ws.regs ml).toNat - 4) 15 := by
    have : a2.regs "tokLo" = WOp.run WOp.min (a1.regs "mlm") (UInt64.ofNat 15) := by
      rw [← ha2]; simp [WStmt.eval, WState.setReg, WArg.eval]
    rw [this, min15_toNat, a1_mlmN]
  -- a3 : token byte (via `wEmitToken litLen "tokLo"`)
  generalize ha3 : (wEmitToken litLen "tokLo").eval fuel a2 = a3
  obtain ⟨t_op, t_ob, t_gm⟩ := eval_wEmitToken litLen "tokLo" a2 fuel a2_tok (by decide) (by decide)
  have a3_op : a3.regs "op" = ws.regs "op" + 1 := by rw [← ha3, t_op, a2_op]
  have a3_ob : a3.regs "outBase" = ws.regs "outBase" := by rw [← ha3, t_ob, a2_ob]
  have a3_ib : a3.regs "inBase" = ws.regs "inBase" := by
    rw [← ha3, wEmitToken_reg litLen "tokLo" "inBase" a2 fuel (by decide) (by decide) (by decide) (by decide), a2_ib]
  have a3_ll : a3.regs litLen = ws.regs litLen := by
    rw [← ha3, wEmitToken_reg litLen "tokLo" litLen a2 fuel hl_tokHi hl_tok hl_sb hl_op, a2_ll]
  have a3_ls : a3.regs litStart = ws.regs litStart := by
    rw [← ha3, wEmitToken_reg litLen "tokLo" litStart a2 fuel hs_tokHi hs_tok hs_sb hs_op, a2_ls]
  have a3_off : a3.regs off = ws.regs off := by
    rw [← ha3, wEmitToken_reg litLen "tokLo" off a2 fuel ho_tokHi ho_tok ho_sb ho_op, a2_off]
  have tokByte : UInt8.ofNat ((min (a2.regs litLen).toNat 15) * 16 + (a2.regs "tokLo").toNat)
      = UInt8.ofNat (min (ws.regs litLen).toNat 15 * 16 + min ((ws.regs ml).toNat - 4) 15) := by
    rw [a2_ll, a2_tokVal]
  have a3_gm : a3.gmem = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
      [UInt8.ofNat (min (ws.regs litLen).toNat 15 * 16 + min ((ws.regs ml).toNat - 4) 15)] := by
    rw [← ha3, t_gm, a2_gm, a2_ob, a2_op, tokByte]; simp [putBytesU]
  -- a4 : `setp ge pLitBig litLen 15`
  generalize ha4 : (WStmt.setp SCmp.ge "pLitBig" litLen (WArg.imm 15)).eval fuel a3 = a4
  have a4_op : a4.regs "op" = ws.regs "op" + 1 := by rw [← ha4, setp_reg _ _ _ _ _ a3 fuel (by decide), a3_op]
  have a4_ob : a4.regs "outBase" = ws.regs "outBase" := by rw [← ha4, setp_reg _ _ _ _ _ a3 fuel (by decide), a3_ob]
  have a4_ib : a4.regs "inBase" = ws.regs "inBase" := by rw [← ha4, setp_reg _ _ _ _ _ a3 fuel (by decide), a3_ib]
  have a4_ll : a4.regs litLen = ws.regs litLen := by rw [← ha4, setp_reg _ _ _ _ _ a3 fuel hl_pLit, a3_ll]
  have a4_ls : a4.regs litStart = ws.regs litStart := by rw [← ha4, setp_reg _ _ _ _ _ a3 fuel hs_pLit, a3_ls]
  have a4_off : a4.regs off = ws.regs off := by rw [← ha4, setp_reg _ _ _ _ _ a3 fuel ho_pLit, a3_off]
  have a4_p : a4.regs "pLitBig" = (if UInt64.ofNat 15 ≤ a4.regs litLen then 1 else 0) := by
    rw [← ha4]; simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run, hl_pLit]
  have a4_gm : a4.gmem = a3.gmem := by rw [← ha4]; exact setp_gmem _ _ _ _ a3 fuel
  -- a5 : litLen length-extension `uif`
  obtain ⟨u_op, u_fr, u_gm⟩ := eval_uifExt_content "pLitBig" "litExtra" litLen a4 fuel
    (by rw [a4_ll] at a4_p ⊢; exact a4_p)
    (by decide) (by decide) (by decide) (by decide) (by decide) hl_litE
    (by rw [a4_ll]; exact hfuelL)
  generalize ha5 : (WStmt.uif "pLitBig"
      (WStmt.seq (WStmt.bin WOp.sub "litExtra" litLen (WArg.imm 15)) (wEmitLSIC "litExtra"))
      WStmt.skip).eval fuel a4 = a5
  have a5_op : a5.regs "op" = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← ha5, u_op, a4_op, a4_ll]
  have a5_ob : a5.regs "outBase" = ws.regs "outBase" := by
    rw [← ha5, u_fr "outBase" (by decide) (by decide) (by decide) (by decide) (by decide), a4_ob]
  have a5_ib : a5.regs "inBase" = ws.regs "inBase" := by
    rw [← ha5, u_fr "inBase" (by decide) (by decide) (by decide) (by decide) (by decide), a4_ib]
  have a5_ll : a5.regs litLen = ws.regs litLen := by
    rw [← ha5, u_fr litLen hl_litE hl_c hl_l hl_op hl_sb, a4_ll]
  have a5_ls : a5.regs litStart = ws.regs litStart := by
    rw [← ha5, u_fr litStart hs_litE hs_c hs_l hs_op hs_sb, a4_ls]
  have a5_off : a5.regs off = ws.regs off := by
    rw [← ha5, u_fr off ho_litE ho_c ho_l ho_op ho_sb, a4_off]
  have a5_gm : a5.gmem = putBytesU a3.gmem (ws.regs "outBase" + (ws.regs "op" + 1))
      (encNib (ws.regs litLen).toNat) := by
    rw [← ha5, u_gm, a4_gm, a4_ob, a4_op, a4_ll]
  -- combine a3/a5 gmem: token ++ encNib(ll)
  have a5_gm' : a5.gmem = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
      ([UInt8.ofNat (min (ws.regs litLen).toNat 15 * 16 + min ((ws.regs ml).toNat - 4) 15)]
        ++ encNib (ws.regs litLen).toNat) := by
    rw [a5_gm, a3_gm, putBytesU_append,
      show (ws.regs "outBase" + ws.regs "op"
            + UInt64.ofNat ([UInt8.ofNat (min (ws.regs litLen).toNat 15 * 16
              + min ((ws.regs ml).toNat - 4) 15)] : List UInt8).length)
          = ws.regs "outBase" + (ws.regs "op" + 1) from by
        show ws.regs "outBase" + ws.regs "op" + 1 = ws.regs "outBase" + (ws.regs "op" + 1); ac_rfl]
  -- numeric shorthands
  generalize hencM : (encNib ((ws.regs ml).toNat - 4)).length = encM
  generalize hobN : (ws.regs "outBase").toNat = obN
  generalize hopN : (ws.regs "op").toNat = opN
  generalize hllN : (ws.regs litLen).toNat = llN
  generalize hencL : (encNib llN).length = encL
  generalize hibN : (ws.regs "inBase").toNat = ibN
  generalize hlsN : (ws.regs litStart).toNat = lsN
  simp only [hllN] at hnwFull hnw2 hsize hdisj a5_op
  simp only [hencL] at hnwFull hsize hdisj a5_op
  simp only [hencM] at hnwFull
  simp only [hobN, hopN] at hnwFull hsize hdisj
  simp only [hibN, hlsN] at hnw2 hdisj
  rw [hllN] at a5_gm'
  -- derived: the prefix length (through the literal run) also fits below 2^64:
  have hnwPre : obN + opN + 1 + encL + llN < 2 ^ 64 := by omega
  -- a6 : `bin add cpDst outBase op`
  generalize ha6 : (WStmt.bin WOp.add "cpDst" "outBase" (WArg.reg "op")).eval fuel a5 = a6
  have a6_ob : a6.regs "outBase" = ws.regs "outBase" := by rw [← ha6, bin_reg _ _ _ _ _ a5 fuel (by decide), a5_ob]
  have a6_ib : a6.regs "inBase" = ws.regs "inBase" := by rw [← ha6, bin_reg _ _ _ _ _ a5 fuel (by decide), a5_ib]
  have a6_ll : a6.regs litLen = ws.regs litLen := by rw [← ha6, bin_reg _ _ _ _ _ a5 fuel hl_cpD, a5_ll]
  have a6_ls : a6.regs litStart = ws.regs litStart := by rw [← ha6, bin_reg _ _ _ _ _ a5 fuel hs_cpD, a5_ls]
  have a6_off : a6.regs off = ws.regs off := by rw [← ha6, bin_reg _ _ _ _ _ a5 fuel ho_cpD, a5_off]
  have a6_op : a6.regs "op" = ws.regs "op" + 1 + UInt64.ofNat encL := by
    rw [← ha6, bin_reg _ _ _ _ _ a5 fuel (by decide)]; rw [a5_op]
  have a6_cpDst : a6.regs "cpDst" = ws.regs "outBase" + (ws.regs "op" + 1 + UInt64.ofNat encL) := by
    have hraw : a6.regs "cpDst" = a5.regs "outBase" + a5.regs "op" := by
      rw [← ha6]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    rw [hraw, a5_ob, a5_op]
  have a6_gm : a6.gmem = a5.gmem := by rw [← ha6]; exact bin_gmem _ _ _ _ a5 fuel
  -- a7 : `bin add cpSrc inBase litStart`
  generalize ha7 : (WStmt.bin WOp.add "cpSrc" "inBase" (WArg.reg litStart)).eval fuel a6 = a7
  have a7_ob : a7.regs "outBase" = ws.regs "outBase" := by rw [← ha7, bin_reg _ _ _ _ _ a6 fuel (by decide), a6_ob]
  have a7_ll : a7.regs litLen = ws.regs litLen := by rw [← ha7, bin_reg _ _ _ _ _ a6 fuel hl_cpS, a6_ll]
  have a7_off : a7.regs off = ws.regs off := by rw [← ha7, bin_reg _ _ _ _ _ a6 fuel ho_cpS, a6_off]
  have a7_op : a7.regs "op" = ws.regs "op" + 1 + UInt64.ofNat encL := by
    rw [← ha7, bin_reg _ _ _ _ _ a6 fuel (by decide), a6_op]
  have a7_cpDst : a7.regs "cpDst" = ws.regs "outBase" + (ws.regs "op" + 1 + UInt64.ofNat encL) := by
    rw [← ha7, bin_reg _ _ _ _ _ a6 fuel (by decide), a6_cpDst]
  have a7_cpSrc : a7.regs "cpSrc" = ws.regs "inBase" + ws.regs litStart := by
    have hraw : a7.regs "cpSrc" = a6.regs "inBase" + a6.regs litStart := by
      rw [← ha7]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    rw [hraw, a6_ib, a6_ls]
  have a7_gm : a7.gmem = a5.gmem := by rw [← ha7, bin_gmem _ _ _ _ a6 fuel, a6_gm]
  -- a8 : coopCopy of literals
  generalize ha8 : (WStmt.coopCopy "cpDst" "cpSrc" litLen).eval fuel a7 = a8
  have a8_ob : a8.regs "outBase" = ws.regs "outBase" := by rw [← ha8, coopCopy_reg _ _ _ _ a7 fuel, a7_ob]
  have a8_ll : a8.regs litLen = ws.regs litLen := by rw [← ha8, coopCopy_reg _ _ _ _ a7 fuel, a7_ll]
  have a8_off : a8.regs off = ws.regs off := by rw [← ha8, coopCopy_reg _ _ _ _ a7 fuel, a7_off]
  have a8_op : a8.regs "op" = ws.regs "op" + 1 + UInt64.ofNat encL := by
    rw [← ha8, coopCopy_reg _ _ _ _ a7 fuel, a7_op]
  have hcpDstN : (a7.regs "cpDst").toNat = obN + opN + 1 + encL := by
    rw [a7_cpDst, UInt64.toNat_add]
    have e1 : (ws.regs "op" + 1 + UInt64.ofNat encL).toNat = opN + 1 + encL := by
      have h1' : (ws.regs "op" + 1).toNat = opN + 1 := by
        rw [UInt64.toNat_add, show (1 : UInt64).toNat = 1 from rfl, Nat.mod_eq_of_lt (by omega), hopN]
      rw [UInt64.toNat_add, h1', UInt64.toNat_ofNat' (n := encL),
          Nat.mod_eq_of_lt (show encL < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
    rw [e1, hobN, Nat.mod_eq_of_lt (by omega)]; omega
  have hcpSrcN : (a7.regs "cpSrc").toNat = ibN + lsN := by
    rw [a7_cpSrc, UInt64.toNat_add, hibN, hlsN, Nat.mod_eq_of_lt (by omega)]
  have a8_gm : a8.gmem = putBytesU a7.gmem (UInt64.ofNat (obN + opN + 1 + encL))
      ((List.range llN).map (fun i => a7.gmem.getD (ibN + lsN + i) 0)) := by
    have hcc : ((WStmt.coopCopy "cpDst" "cpSrc" litLen).eval fuel a7).gmem
        = copyGmem a7.gmem (a7.regs "cpDst").toNat (a7.regs "cpSrc").toNat (a7.regs litLen).toNat := by
      simp [WStmt.eval]
    rw [← ha8, hcc, hcpDstN, hcpSrcN, a7_ll, hllN]
    apply copyGmem_eq_putBytesU a7.gmem (obN + opN + 1 + encL) (ibN + lsN) llN
    · rcases hdisj with h | h
      · left; omega
      · right; omega
    · rw [a7_gm, a5_gm', putBytesU_size]
      have : ([UInt8.ofNat (min llN 15 * 16 + min ((ws.regs ml).toNat - 4) 15)]
          ++ encNib llN).length = 1 + encL := by simp [hencL]; omega
      omega
    · omega
  have hb0 : (ws.regs "outBase" + ws.regs "op").toNat = obN + opN := by
    rw [UInt64.toNat_add, Nat.mod_eq_of_lt (by omega)]; omega
  have hxs0len : ([UInt8.ofNat (min llN 15 * 16 + min ((ws.regs ml).toNat - 4) 15)]
      ++ encNib llN).length = 1 + encL := by simp [hencL]; omega
  -- literal bytes read from a7.gmem = those from ws.gmem (write window disjoint)
  have hliteq : ∀ i, i < llN → a7.gmem.getD (ibN + lsN + i) 0 = ws.gmem.getD (ibN + lsN + i) 0 := by
    intro i hi
    rw [a7_gm, a5_gm']
    rcases hdisj with h | h
    · exact putBytesU_getD_ge (ibN + lsN + i) _ ws.gmem (ws.regs "outBase" + ws.regs "op")
        (by rw [hb0, hxs0len]; omega) (by rw [hb0, hxs0len]; omega)
    · exact putBytesU_getD_lt (ibN + lsN + i) _ ws.gmem (ws.regs "outBase" + ws.regs "op")
        (by rw [hb0, hxs0len]; omega) (by rw [hb0]; omega)
  have a8_gm' : a8.gmem = putBytesU a7.gmem (UInt64.ofNat (obN + opN + 1 + encL))
      ((List.range llN).map (fun i => ws.gmem.getD (ibN + lsN + i) 0)) := by
    rw [a8_gm]; congr 1; apply List.map_congr_left; intro i hi; exact hliteq i (List.mem_range.mp hi)
  -- the literal run, as a fixed list:
  generalize hlits : (List.range llN).map (fun i => ws.gmem.getD (ibN + lsN + i) 0) = lits
  have hlitslen : lits.length = llN := by rw [← hlits]; simp
  generalize htokB : UInt8.ofNat (min llN 15 * 16 + min ((ws.regs ml).toNat - 4) 15) = tokB
  -- combine a5' + a8' into one putBytesU of  [tokB] ++ encNib(ll) ++ lits
  have hbaseComb : ws.regs "outBase" + ws.regs "op"
        + UInt64.ofNat ([tokB] ++ encNib llN).length = UInt64.ofNat (obN + opN + 1 + encL) := by
    rw [← UInt64.toNat_inj, UInt64.toNat_add, hb0,
        show ([tokB] ++ encNib llN).length = 1 + encL from by simp [hencL]; omega,
        UInt64.toNat_ofNat' (n := 1 + encL), Nat.mod_eq_of_lt (show (1 + encL) < 2 ^ 64 by omega),
        Nat.mod_eq_of_lt (by omega),
        UInt64.toNat_ofNat' (n := obN + opN + 1 + encL),
        Nat.mod_eq_of_lt (show obN + opN + 1 + encL < 2 ^ 64 by omega)]
    omega
  have a8_gmFull : a8.gmem = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
      (([tokB] ++ encNib llN) ++ lits) := by
    rw [putBytesU_append, hbaseComb, a8_gm', a7_gm, a5_gm', ← htokB, hlits]
  -- a9 : `bin add op op litLen` (gmem untouched, op advances by litLen)
  generalize ha9 : (WStmt.bin WOp.add "op" "op" (WArg.reg litLen)).eval fuel a8 = a9
  have a9_ob : a9.regs "outBase" = ws.regs "outBase" := by rw [← ha9, bin_reg _ _ _ _ _ a8 fuel (by decide), a8_ob]
  have a9_off : a9.regs off = ws.regs off := by rw [← ha9, bin_reg _ _ _ _ _ a8 fuel ho_op, a8_off]
  have a9_op : a9.regs "op" = ws.regs "op" + 1 + UInt64.ofNat encL + ws.regs litLen := by
    have hraw : a9.regs "op" = a8.regs "op" + a8.regs litLen := by
      rw [← ha9]; simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    rw [hraw, a8_op, a8_ll]
  have a9_opN : (a9.regs "op").toNat = opN + 1 + encL + llN := by
    rw [a9_op]
    rw [UInt64.toNat_add]
    have hpre : (ws.regs "op" + 1 + UInt64.ofNat encL).toNat = opN + 1 + encL := by
      have h1' : (ws.regs "op" + 1).toNat = opN + 1 := by
        rw [UInt64.toNat_add, show (1 : UInt64).toNat = 1 from rfl, Nat.mod_eq_of_lt (by omega), hopN]
      rw [UInt64.toNat_add, h1', UInt64.toNat_ofNat' (n := encL),
          Nat.mod_eq_of_lt (show encL < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
    rw [hpre, hllN, Nat.mod_eq_of_lt (by omega)]
  have a9_gm : a9.gmem = a8.gmem := by rw [← ha9]; exact bin_gmem _ _ _ _ a8 fuel
  -- the running output base for the offset bytes (`= obN + opN + 1 + encL + llN`):
  have a9_opN' : (a9.regs "op").toNat = opN + 1 + encL + llN := a9_opN
  -- a10 : `bin band offLo off 255`
  generalize ha10 : (WStmt.bin WOp.band "offLo" off (WArg.imm 255)).eval fuel a9 = a10
  have a10_ob : a10.regs "outBase" = ws.regs "outBase" := by rw [← ha10, bin_reg _ _ _ _ _ a9 fuel (by decide), a9_ob]
  have a10_op : a10.regs "op" = a9.regs "op" := by rw [← ha10, bin_reg _ _ _ _ _ a9 fuel (by decide)]
  have a10_off : a10.regs off = ws.regs off := by rw [← ha10, bin_reg _ _ _ _ _ a9 fuel ho_offLo, a9_off]
  have a10_offLo : a10.regs "offLo" = ws.regs off &&& UInt64.ofNat 255 := by
    have hraw : a10.regs "offLo" = WOp.run WOp.band (a9.regs off) (UInt64.ofNat 255) := by
      rw [← ha10]; simp [WStmt.eval, WState.setReg, WArg.eval]
    rw [hraw, WOp.run, a9_off]
  have a10_gm : a10.gmem = a8.gmem := by rw [← ha10, bin_gmem _ _ _ _ a9 fuel, a9_gm]
  -- a11 : `wStoreByte offLo`
  generalize ha11 : (wStoreByte "offLo").eval fuel a10 = a11
  obtain ⟨sb_op, sb_ob, sb_gm⟩ := eval_wStoreByte "offLo" a10 fuel (by decide) (by decide)
  have a11_ob : a11.regs "outBase" = ws.regs "outBase" := by rw [← ha11, sb_ob, a10_ob]
  have a11_op : a11.regs "op" = a9.regs "op" + 1 := by rw [← ha11, sb_op, a10_op]
  have a11_off : a11.regs off = ws.regs off := by
    rw [← ha11, wStoreByte_reg "offLo" off a10 fuel ho_sb ho_op, a10_off]
  have a11_gm : a11.gmem = (a8.gmem).set! (ws.regs "outBase" + a9.regs "op").toNat
      (UInt8.ofNat ((ws.regs off).toNat % 256)) := by
    rw [← ha11, sb_gm, a10_ob, a10_op, a10_gm, a10_offLo, offLo_byte]
  -- a12 : `bin shr offHi off 8`
  generalize ha12 : (WStmt.bin WOp.shr "offHi" off (WArg.imm 8)).eval fuel a11 = a12
  have a12_ob : a12.regs "outBase" = ws.regs "outBase" := by rw [← ha12, bin_reg _ _ _ _ _ a11 fuel (by decide), a11_ob]
  have a12_op : a12.regs "op" = a9.regs "op" + 1 := by rw [← ha12, bin_reg _ _ _ _ _ a11 fuel (by decide), a11_op]
  have a12_off : a12.regs off = ws.regs off := by rw [← ha12, bin_reg _ _ _ _ _ a11 fuel ho_offHi, a11_off]
  have a12_offHi : a12.regs "offHi" = ws.regs off >>> UInt64.ofNat 8 := by
    have hraw : a12.regs "offHi" = WOp.run WOp.shr (a11.regs off) (UInt64.ofNat 8) := by
      rw [← ha12]; simp [WStmt.eval, WState.setReg, WArg.eval]
    rw [hraw, WOp.run, a11_off]
  have a12_gm : a12.gmem = a11.gmem := by rw [← ha12]; exact bin_gmem _ _ _ _ a11 fuel
  -- a13 : `bin band offHi offHi 255`
  generalize ha13 : (WStmt.bin WOp.band "offHi" "offHi" (WArg.imm 255)).eval fuel a12 = a13
  have a13_ob : a13.regs "outBase" = ws.regs "outBase" := by rw [← ha13, bin_reg _ _ _ _ _ a12 fuel (by decide), a12_ob]
  have a13_op : a13.regs "op" = a9.regs "op" + 1 := by rw [← ha13, bin_reg _ _ _ _ _ a12 fuel (by decide), a12_op]
  have a13_offHi : a13.regs "offHi" = (ws.regs off >>> UInt64.ofNat 8) &&& UInt64.ofNat 255 := by
    have hraw : a13.regs "offHi" = WOp.run WOp.band (a12.regs "offHi") (UInt64.ofNat 255) := by
      rw [← ha13]; simp [WStmt.eval, WState.setReg, WArg.eval]
    rw [hraw, WOp.run, a12_offHi]
  have a13_gm : a13.gmem = a11.gmem := by rw [← ha13, bin_gmem _ _ _ _ a12 fuel, a12_gm]
  -- a14 : `wStoreByte offHi`
  generalize ha14 : (wStoreByte "offHi").eval fuel a13 = a14
  obtain ⟨sb2_op, sb2_ob, sb2_gm⟩ := eval_wStoreByte "offHi" a13 fuel (by decide) (by decide)
  have a14_ob : a14.regs "outBase" = ws.regs "outBase" := by rw [← ha14, sb2_ob, a13_ob]
  have a14_op : a14.regs "op" = a9.regs "op" + 1 + 1 := by rw [← ha14, sb2_op, a13_op]
  have a14_gm : a14.gmem = (a11.gmem).set! (ws.regs "outBase" + (a9.regs "op" + 1)).toNat
      (UInt8.ofNat ((ws.regs off).toNat / 256)) := by
    rw [← ha14, sb2_gm, a13_ob, a13_op, a13_gm, a13_offHi, offHi_byte]
  -- express the two offset writes as `putBytesU a8.gmem (outBase + a9.op) [offLo, offHi]`
  have hoffLoAddr : (ws.regs "outBase" + a9.regs "op").toNat = obN + opN + 1 + encL + llN := by
    rw [UInt64.toNat_add, hobN, a9_opN', Nat.mod_eq_of_lt (by omega)]; omega
  have hoffHi_eq : ws.regs "outBase" + (a9.regs "op" + 1) = ws.regs "outBase" + a9.regs "op" + 1 := by
    ac_rfl
  have a14_gmOff : a14.gmem = putBytesU a8.gmem (ws.regs "outBase" + a9.regs "op")
      [UInt8.ofNat ((ws.regs off).toNat % 256), UInt8.ofNat ((ws.regs off).toNat / 256)] := by
    rw [a14_gm, a11_gm, hoffHi_eq]
    show (a8.gmem.set! (ws.regs "outBase" + a9.regs "op").toNat (UInt8.ofNat ((ws.regs off).toNat % 256))).set!
        (ws.regs "outBase" + a9.regs "op" + 1).toNat (UInt8.ofNat ((ws.regs off).toNat / 256))
      = putBytesU a8.gmem (ws.regs "outBase" + a9.regs "op")
          [UInt8.ofNat ((ws.regs off).toNat % 256), UInt8.ofNat ((ws.regs off).toNat / 256)]
    simp only [putBytesU]
  -- combine into a8.gmem's byte list ++ [offLo, offHi]
  have hbigAddr : ws.regs "outBase" + ws.regs "op"
        + UInt64.ofNat (([tokB] ++ encNib llN) ++ lits).length = ws.regs "outBase" + a9.regs "op" := by
    rw [← UInt64.toNat_inj, UInt64.toNat_add, hb0,
        show (([tokB] ++ encNib llN) ++ lits).length = 1 + encL + llN from by
          simp [hencL, hlitslen]; omega,
        UInt64.toNat_ofNat' (n := 1 + encL + llN),
        Nat.mod_eq_of_lt (show 1 + encL + llN < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega),
        UInt64.toNat_add, hobN, a9_opN', Nat.mod_eq_of_lt (by omega)]
    omega
  have a14_gmFull : a14.gmem = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
      ((([tokB] ++ encNib llN) ++ lits) ++
        [UInt8.ofNat ((ws.regs off).toNat % 256), UInt8.ofNat ((ws.regs off).toNat / 256)]) := by
    rw [putBytesU_append, hbigAddr, ← a8_gmFull, a14_gmOff]
  -- `mlm`, `outBase`, `op` values at a14 (mlm untouched since a1; op advanced to +2)
  have a14_opN : (a14.regs "op").toNat = opN + 1 + encL + llN + 2 := by
    rw [a14_op]
    have e1 : (a9.regs "op" + 1).toNat = opN + 1 + encL + llN + 1 := by
      rw [UInt64.toNat_add, a9_opN', show (1:UInt64).toNat = 1 from rfl, Nat.mod_eq_of_lt (by omega)]
    rw [UInt64.toNat_add, e1, show (1:UInt64).toNat = 1 from rfl, Nat.mod_eq_of_lt (by omega)]
  -- mlm register: unchanged from a1 through a14 (no stage writes "mlm")
  have a3_mlm : a3.regs "mlm" = a1.regs "mlm" := by
    rw [← ha3, wEmitToken_reg litLen "tokLo" "mlm" a2 fuel (by decide) (by decide) (by decide) (by decide),
        ← ha2, bin_reg _ _ _ _ _ a1 fuel (by decide)]
  have a5_mlm : a5.regs "mlm" = a1.regs "mlm" := by
    rw [← ha5, u_fr "mlm" (by decide) (by decide) (by decide) (by decide) (by decide),
        ← ha4, setp_reg _ _ _ _ _ a3 fuel (by decide), a3_mlm]
  have a14_mlm : a14.regs "mlm" = a1.regs "mlm" := by
    rw [← ha14, wStoreByte_reg "offHi" "mlm" a13 fuel (by decide) (by decide),
        ← ha13, bin_reg _ _ _ _ _ a12 fuel (by decide),
        ← ha12, bin_reg _ _ _ _ _ a11 fuel (by decide),
        ← ha11, wStoreByte_reg "offLo" "mlm" a10 fuel (by decide) (by decide),
        ← ha10, bin_reg _ _ _ _ _ a9 fuel (by decide),
        ← ha9, bin_reg _ _ _ _ _ a8 fuel (by decide),
        ← ha8, coopCopy_reg _ _ _ _ a7 fuel,
        ← ha7, bin_reg _ _ _ _ _ a6 fuel (by decide),
        ← ha6, bin_reg _ _ _ _ _ a5 fuel (by decide), a5_mlm]
  have a14_mlmN : (a14.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by rw [a14_mlm, a1_mlmN]
  have a14_ob' : a14.regs "outBase" = ws.regs "outBase" := a14_ob
  -- a15 : `setp ge pMatBig mlm 15`
  generalize ha15 : (WStmt.setp SCmp.ge "pMatBig" "mlm" (WArg.imm 15)).eval fuel a14 = a15
  have a15_ob : a15.regs "outBase" = ws.regs "outBase" := by
    rw [← ha15, setp_reg _ _ _ _ _ a14 fuel (by decide), a14_ob']
  have a15_op : a15.regs "op" = a14.regs "op" := by rw [← ha15, setp_reg _ _ _ _ _ a14 fuel (by decide)]
  have a15_mlm : a15.regs "mlm" = a1.regs "mlm" := by
    rw [← ha15, setp_reg _ _ _ _ _ a14 fuel (by decide), a14_mlm]
  have a15_p : a15.regs "pMatBig" = (if UInt64.ofNat 15 ≤ a15.regs "mlm" then 1 else 0) := by
    rw [← ha15]; simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  have a15_gm : a15.gmem = a14.gmem := by rw [← ha15]; exact setp_gmem _ _ _ _ a14 fuel
  -- a16 : match length-extension `uif`  → encNib(mlm) = encNib(ml-4)
  have a15_mlmN : (a15.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by rw [a15_mlm, a1_mlmN]
  obtain ⟨mu_op, mu_fr, mu_gm⟩ := eval_uifExt_content "pMatBig" "matExtra" "mlm" a15 fuel
    a15_p
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by rw [a15_mlmN]; exact hfuelM)
  generalize ha16 : (WStmt.uif "pMatBig"
      (WStmt.seq (WStmt.bin WOp.sub "matExtra" "mlm" (WArg.imm 15)) (wEmitLSIC "matExtra"))
      WStmt.skip).eval fuel a15 = a16
  have a16_gm : a16.gmem = putBytesU a15.gmem (ws.regs "outBase" + a15.regs "op")
      (encNib ((ws.regs ml).toNat - 4)) := by
    rw [← ha16, mu_gm, a15_mlmN, a15_ob]
  -- ── grand assembly: a16.gmem = putBytesU ws.gmem base (encodeSeq ⟨lits, off, ml⟩) ──
  have hmatAddr : ws.regs "outBase" + ws.regs "op"
        + UInt64.ofNat ((([tokB] ++ encNib llN) ++ lits)
            ++ [UInt8.ofNat ((ws.regs off).toNat % 256), UInt8.ofNat ((ws.regs off).toNat / 256)]).length
      = ws.regs "outBase" + a15.regs "op" := by
    rw [a15_op,
        ← UInt64.toNat_inj, UInt64.toNat_add, hb0,
        show ((([tokB] ++ encNib llN) ++ lits)
              ++ [UInt8.ofNat ((ws.regs off).toNat % 256), UInt8.ofNat ((ws.regs off).toNat / 256)]).length
            = 1 + encL + llN + 2 from by simp [hencL, hlitslen]; omega,
        UInt64.toNat_ofNat' (n := 1 + encL + llN + 2),
        Nat.mod_eq_of_lt (show 1 + encL + llN + 2 < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega),
        UInt64.toNat_add, hobN, a14_opN, Nat.mod_eq_of_lt (by omega)]
    omega
  have a16_gmFull : a16.gmem = putBytesU ws.gmem (ws.regs "outBase" + ws.regs "op")
      (((([tokB] ++ encNib llN) ++ lits)
        ++ [UInt8.ofNat ((ws.regs off).toNat % 256), UInt8.ofNat ((ws.regs off).toNat / 256)])
        ++ encNib ((ws.regs ml).toNat - 4)) := by
    rw [putBytesU_append, hmatAddr, ← a14_gmFull, ← a15_gm, a16_gm]
  -- match the byte list against `encodeSeq`:
  rw [a16_gmFull]
  congr 1
  -- goal: [tokB]++encNib(llN)++lits++[offLo,offHi]++encNib(ml-4) = encodeSeq ⟨lits, off, ml⟩
  simp only [LZ4.encodeSeq]
  rw [hlitslen, ← htokB]
  simp only [List.cons_append, List.nil_append, List.append_assoc]

end EmitContent
