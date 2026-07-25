import AlgorithmLib.LZ4WarpKernel
open AlgorithmLib AlgorithmLib.LZ4WarpDSL AlgorithmLib.LZ4Simt AlgorithmLib.LZ4

namespace LZ4WarpEvalBytes

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
    (h1 : val ≠ "sbAddr") (h2 : val ≠ "op") :
    ((wStoreByte val).eval fuel ws).regs "op" = ws.regs "op" + 1
    ∧ ((wStoreByte val).eval fuel ws).regs "outBase" = ws.regs "outBase"
    ∧ ((wStoreByte val).eval fuel ws).gmem
        = ws.gmem.set! (ws.regs "outBase" + ws.regs "op").toNat (ws.regs val).toUInt8 := by
  simp only [wStoreByte, wseq, WStmt.eval, WState.setReg, WState.stgByte, WArg.eval, WOp.run]
  refine ⟨?_, ?_, ?_⟩ <;> simp [h1, h2]

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
  · simp [htl1, htl2]
  · simp [htl1, htl2]
  · simp only [htl1, htl2, if_neg, if_pos, ↓reduceIte, String.reduceEq]
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
  -- field values of the setp state s0
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

-- ── Register-frame lemmas (registers outside a builder's write-set are preserved) ──

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
  simp [hr_c, hr_l, hr_op, hr_sb, hr_n, Ne.symm hr_n]

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

-- ── Transparency of the simple statements to an untouched register ──

theorem bin_reg (o : WOp) (d a r : String) (b : WArg) (st : WState) (fuel : Nat) (hr : r ≠ d) :
    ((WStmt.bin o d a b).eval fuel st).regs r = st.regs r := by
  simp [WStmt.eval, WState.setReg, hr]

theorem setp_reg (c : SCmp) (d a r : String) (b : WArg) (st : WState) (fuel : Nat) (hr : r ≠ d) :
    ((WStmt.setp c d a b).eval fuel st).regs r = st.regs r := by
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

/-- The length-extension `if`: `setp P (src ≥ 15)` has already run (given by `hP`);
    this evaluates `if P then (E := src-15; wEmitLSIC E)` and shows it advances `op`
    by exactly `(encNib src).length` bytes and frames untouched registers. -/
theorem eval_uifExt (P E src : String) (ws : WState) (fuel : Nat)
    (hP : ws.regs P = (if UInt64.ofNat 15 ≤ ws.regs src then 1 else 0))
    (hEc : E ≠ "c255") (hEl : E ≠ "lsicC") (hEop : E ≠ "op") (hEsb : E ≠ "sbAddr")
    (hEob : E ≠ "outBase") (hsrcE : src ≠ E)
    (hfuel : (ws.regs src).toNat / 255 < fuel) :
    ((WStmt.uif P (WStmt.seq (WStmt.bin WOp.sub E src (WArg.imm 15)) (wEmitLSIC E))
        WStmt.skip).eval fuel ws).regs "op"
          = ws.regs "op" + UInt64.ofNat (encNib (ws.regs src).toNat).length
    ∧ (∀ r, r ≠ E → r ≠ "c255" → r ≠ "lsicC" → r ≠ "op" → r ≠ "sbAddr" →
        ((WStmt.uif P (WStmt.seq (WStmt.bin WOp.sub E src (WArg.imm 15)) (wEmitLSIC E))
          WStmt.skip).eval fuel ws).regs r = ws.regs r) := by
  have h15 : (UInt64.ofNat 15).toNat = 15 := by decide
  by_cases hbig : UInt64.ofNat 15 ≤ ws.regs src
  · -- src ≥ 15 : the extension runs
    have hP1 : ws.regs P = 1 := by rw [hP, if_pos hbig]
    have hVge : 15 ≤ (ws.regs src).toNat := by
      rw [UInt64.le_iff_toNat_le, h15] at hbig; exact hbig
    rw [eval_uif_pos _ _ _ _ _ hP1, eval_seq]
    -- s1 = (sub E src 15).eval : sets E := src - 15
    have hs1E : ((WStmt.bin WOp.sub E src (WArg.imm 15)).eval fuel ws).regs E
        = ws.regs src - UInt64.ofNat 15 := by simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run]
    have hs1EN : (((WStmt.bin WOp.sub E src (WArg.imm 15)).eval fuel ws).regs E).toNat
        = (ws.regs src).toNat - 15 := by
      rw [hs1E, UInt64.toNat_sub_of_le _ _ hbig, h15]
    have hencNib : encNib (ws.regs src).toNat = ext ((ws.regs src).toNat - 15) := by
      rw [encNib, if_neg (by omega)]
    obtain ⟨lop, lob, lg⟩ := eval_wEmitLSIC E
      ((WStmt.bin WOp.sub E src (WArg.imm 15)).eval fuel ws) fuel hEc hEl hEop hEob hEsb
      (by rw [hs1EN]; omega)
    refine ⟨?_, ?_⟩
    · rw [lop, bin_reg WOp.sub E src "op" (WArg.imm 15) ws fuel (Ne.symm hEop), hs1EN, ← hencNib]
    · intro r hrE hrc hrl hrop hrsb
      rw [wEmitLSIC_reg E r _ fuel hrc hrl hrop hrsb hrE,
          bin_reg WOp.sub E src r (WArg.imm 15) ws fuel hrE]
  · -- src < 15 : the extension is skipped (empty)
    have hP0 : ws.regs P = 0 := by rw [hP, if_neg hbig]
    have hVlt : (ws.regs src).toNat < 15 := by
      rw [UInt64.le_iff_toNat_le, h15] at hbig; omega
    rw [eval_uif_neg _ _ _ _ _ hP0]
    have hnil : encNib (ws.regs src).toNat = [] := by rw [encNib, if_pos hVlt]
    refine ⟨?_, ?_⟩
    · simp only [WStmt.eval, hnil, List.length_nil]
      show ws.regs "op" = ws.regs "op" + UInt64.ofNat 0
      simp
    · intro r _ _ _ _ _; simp only [WStmt.eval]

/-- **`wEmitMatchSeq`** (structural): advances `op` by exactly the encoded length of
    `encodeSeq` for this sequence — token byte, `encNib` of the literal length, the
    `litLen` literals, the two offset bytes, and `encNib` of the match length — and
    preserves `outBase`. -/
theorem eval_wEmitMatchSeq (litStart litLen off ml : String) (ws : WState) (fuel : Nat)
    (hl_mlm : litLen ≠ "mlm") (hl_tokLo : litLen ≠ "tokLo") (hl_tokHi : litLen ≠ "tokHi")
    (hl_tok : litLen ≠ "tok") (hl_sb : litLen ≠ "sbAddr") (hl_op : litLen ≠ "op")
    (hl_pLit : litLen ≠ "pLitBig") (hl_litE : litLen ≠ "litExtra") (hl_c : litLen ≠ "c255")
    (hl_l : litLen ≠ "lsicC") (hl_cpD : litLen ≠ "cpDst") (hl_cpS : litLen ≠ "cpSrc")
    (hml : 4 ≤ (ws.regs ml).toNat)
    (hfuelL : (ws.regs litLen).toNat / 255 < fuel)
    (hfuelM : ((ws.regs ml).toNat - 4) / 255 < fuel) :
    ((wEmitMatchSeq litStart litLen off ml).eval fuel ws).regs "op"
        = ws.regs "op"
          + UInt64.ofNat (1 + (encNib (ws.regs litLen).toNat).length + (ws.regs litLen).toNat
              + 2 + (encNib ((ws.regs ml).toNat - 4)).length)
    ∧ ((wEmitMatchSeq litStart litLen off ml).eval fuel ws).regs "outBase" = ws.regs "outBase" := by
  have min15 : ∀ x : UInt64, (WOp.run WOp.min x (UInt64.ofNat 15)).toNat < 16 := by
    intro x
    have h15 : (UInt64.ofNat 15).toNat = 15 := by decide
    simp only [WOp.run]; split
    · next h => rw [UInt64.le_iff_toNat_le, h15] at h; omega
    · rw [h15]; omega
  have arith2 : ∀ (op ll : UInt64) (a b : Nat),
      op + 1 + UInt64.ofNat a + ll + 1 + 1 + UInt64.ofNat b
        = op + UInt64.ofNat (1 + a + ll.toNat + 2 + b) := by
    intro op ll a b
    rw [show (1 + a + ll.toNat + 2 + b) = 1 + a + ll.toNat + 1 + 1 + b from by omega]
    simp only [UInt64.ofNat_add, UInt64.ofNat_toNat, show UInt64.ofNat 1 = 1 from rfl]
    ac_rfl
  have hml4 : UInt64.ofNat 4 ≤ ws.regs ml := by
    rw [UInt64.le_iff_toNat_le, show (UInt64.ofNat 4).toNat = 4 from by decide]; omega
  simp only [wEmitMatchSeq, wseq, eval_seq]
  -- S1 : mlm := ml - 4
  generalize hs1 : (WStmt.bin WOp.sub "mlm" ml (WArg.imm 4)).eval fuel ws = s1
  have s1_op : s1.regs "op" = ws.regs "op" := by rw [← hs1]; exact bin_reg _ _ _ _ _ ws fuel (by decide)
  have s1_ob : s1.regs "outBase" = ws.regs "outBase" := by
    rw [← hs1]; exact bin_reg _ _ _ _ _ ws fuel (by decide)
  have s1_ll : s1.regs litLen = ws.regs litLen := by rw [← hs1]; exact bin_reg _ _ _ _ _ ws fuel hl_mlm
  have s1_mlmN : (s1.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs1,
        show ((WStmt.bin WOp.sub "mlm" ml (WArg.imm 4)).eval fuel ws).regs "mlm"
          = ws.regs ml - UInt64.ofNat 4 from by simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run],
        UInt64.toNat_sub_of_le _ _ hml4, show (UInt64.ofNat 4).toNat = 4 from by decide]
  -- S2 : tokLo := min mlm 15
  generalize hs2 : (WStmt.bin WOp.min "tokLo" "mlm" (WArg.imm 15)).eval fuel s1 = s2
  have s2_op : s2.regs "op" = ws.regs "op" := by
    rw [← hs2, bin_reg _ _ _ _ _ s1 fuel (by decide), s1_op]
  have s2_ob : s2.regs "outBase" = ws.regs "outBase" := by
    rw [← hs2, bin_reg _ _ _ _ _ s1 fuel (by decide), s1_ob]
  have s2_ll : s2.regs litLen = ws.regs litLen := by
    rw [← hs2, bin_reg _ _ _ _ _ s1 fuel hl_tokLo, s1_ll]
  have s2_mlmN : (s2.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs2, bin_reg _ _ _ _ _ s1 fuel (by decide), s1_mlmN]
  have s2_tok : (s2.regs "tokLo").toNat < 16 := by
    have hval : s2.regs "tokLo" = WOp.run WOp.min (s1.regs "mlm") (UInt64.ofNat 15) := by
      rw [← hs2]; simp [WStmt.eval, WState.setReg, WArg.eval]
    rw [hval]; exact min15 _
  -- S3 : token
  generalize hs3 : (wEmitToken litLen "tokLo").eval fuel s2 = s3
  obtain ⟨t_op, t_ob, _⟩ := eval_wEmitToken litLen "tokLo" s2 fuel s2_tok (by decide) (by decide)
  have s3_op : s3.regs "op" = ws.regs "op" + 1 := by rw [← hs3, t_op, s2_op]
  have s3_ob : s3.regs "outBase" = ws.regs "outBase" := by rw [← hs3, t_ob, s2_ob]
  have s3_ll : s3.regs litLen = ws.regs litLen := by
    rw [← hs3, wEmitToken_reg litLen "tokLo" litLen s2 fuel hl_tokHi hl_tok hl_sb hl_op, s2_ll]
  have s3_mlmN : (s3.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs3, wEmitToken_reg litLen "tokLo" "mlm" s2 fuel (by decide) (by decide) (by decide)
      (by decide), s2_mlmN]
  -- S4 : setp pLitBig
  generalize hs4 : (WStmt.setp SCmp.ge "pLitBig" litLen (WArg.imm 15)).eval fuel s3 = s4
  have s4_op : s4.regs "op" = ws.regs "op" + 1 := by
    rw [← hs4, setp_reg _ _ _ _ _ s3 fuel (by decide), s3_op]
  have s4_ob : s4.regs "outBase" = ws.regs "outBase" := by
    rw [← hs4, setp_reg _ _ _ _ _ s3 fuel (by decide), s3_ob]
  have s4_ll : s4.regs litLen = ws.regs litLen := by
    rw [← hs4, setp_reg _ _ _ _ _ s3 fuel hl_pLit, s3_ll]
  have s4_mlmN : (s4.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs4, setp_reg _ _ _ _ _ s3 fuel (by decide), s3_mlmN]
  have s4_pLit : s4.regs "pLitBig" = (if UInt64.ofNat 15 ≤ s4.regs litLen then 1 else 0) := by
    rw [← hs4]; simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run, hl_pLit]
  -- S5 : literal length-extension uif
  obtain ⟨lu_op, lu_fr⟩ := eval_uifExt "pLitBig" "litExtra" litLen s4 fuel s4_pLit
    (by decide) (by decide) (by decide) (by decide) (by decide) hl_litE
    (by rw [s4_ll]; exact hfuelL)
  generalize hs5 : (WStmt.uif "pLitBig"
      (WStmt.seq (WStmt.bin WOp.sub "litExtra" litLen (WArg.imm 15)) (wEmitLSIC "litExtra"))
      WStmt.skip).eval fuel s4 = s5
  have s5_op : s5.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs5, lu_op, s4_op, s4_ll]
  have s5_ob : s5.regs "outBase" = ws.regs "outBase" := by
    rw [← hs5, lu_fr "outBase" (by decide) (by decide) (by decide) (by decide) (by decide), s4_ob]
  have s5_ll : s5.regs litLen = ws.regs litLen := by
    rw [← hs5, lu_fr litLen hl_litE hl_c hl_l hl_op hl_sb, s4_ll]
  have s5_mlmN : (s5.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs5, lu_fr "mlm" (by decide) (by decide) (by decide) (by decide) (by decide), s4_mlmN]
  -- S6 : cpDst := outBase + op
  generalize hs6 : (WStmt.bin WOp.add "cpDst" "outBase" (WArg.reg "op")).eval fuel s5 = s6
  have s6_op : s6.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs6, bin_reg _ _ _ _ _ s5 fuel (by decide), s5_op]
  have s6_ob : s6.regs "outBase" = ws.regs "outBase" := by
    rw [← hs6, bin_reg _ _ _ _ _ s5 fuel (by decide), s5_ob]
  have s6_ll : s6.regs litLen = ws.regs litLen := by
    rw [← hs6, bin_reg _ _ _ _ _ s5 fuel hl_cpD, s5_ll]
  have s6_mlmN : (s6.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs6, bin_reg _ _ _ _ _ s5 fuel (by decide), s5_mlmN]
  -- S7 : cpSrc := inBase + litStart
  generalize hs7 : (WStmt.bin WOp.add "cpSrc" "inBase" (WArg.reg litStart)).eval fuel s6 = s7
  have s7_op : s7.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs7, bin_reg _ _ _ _ _ s6 fuel (by decide), s6_op]
  have s7_ob : s7.regs "outBase" = ws.regs "outBase" := by
    rw [← hs7, bin_reg _ _ _ _ _ s6 fuel (by decide), s6_ob]
  have s7_ll : s7.regs litLen = ws.regs litLen := by
    rw [← hs7, bin_reg _ _ _ _ _ s6 fuel hl_cpS, s6_ll]
  have s7_mlmN : (s7.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs7, bin_reg _ _ _ _ _ s6 fuel (by decide), s6_mlmN]
  -- S8 : coopCopy
  generalize hs8 : (WStmt.coopCopy "cpDst" "cpSrc" litLen).eval fuel s7 = s8
  have s8_op : s8.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs8, coopCopy_reg _ _ _ _ s7 fuel, s7_op]
  have s8_ob : s8.regs "outBase" = ws.regs "outBase" := by
    rw [← hs8, coopCopy_reg _ _ _ _ s7 fuel, s7_ob]
  have s8_ll : s8.regs litLen = ws.regs litLen := by
    rw [← hs8, coopCopy_reg _ _ _ _ s7 fuel, s7_ll]
  have s8_mlmN : (s8.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs8, coopCopy_reg _ _ _ _ s7 fuel, s7_mlmN]
  -- S9 : op += litLen
  generalize hs9 : (WStmt.bin WOp.add "op" "op" (WArg.reg litLen)).eval fuel s8 = s9
  have s9_op : s9.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length + ws.regs litLen := by
    rw [← hs9,
        show ((WStmt.bin WOp.add "op" "op" (WArg.reg litLen)).eval fuel s8).regs "op"
          = s8.regs "op" + s8.regs litLen from by
        simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run], s8_op, s8_ll]
  have s9_ob : s9.regs "outBase" = ws.regs "outBase" := by
    rw [← hs9, bin_reg _ _ _ _ _ s8 fuel (by decide), s8_ob]
  have s9_mlmN : (s9.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs9, bin_reg _ _ _ _ _ s8 fuel (by decide), s8_mlmN]
  -- S10 : offLo := off & 255
  generalize hs10 : (WStmt.bin WOp.band "offLo" off (WArg.imm 255)).eval fuel s9 = s10
  have s10_op : s10.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length + ws.regs litLen := by
    rw [← hs10, bin_reg _ _ _ _ _ s9 fuel (by decide), s9_op]
  have s10_ob : s10.regs "outBase" = ws.regs "outBase" := by
    rw [← hs10, bin_reg _ _ _ _ _ s9 fuel (by decide), s9_ob]
  have s10_mlmN : (s10.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs10, bin_reg _ _ _ _ _ s9 fuel (by decide), s9_mlmN]
  -- S11 : store offLo byte
  generalize hs11 : (wStoreByte "offLo").eval fuel s10 = s11
  obtain ⟨st11_op, st11_ob, _⟩ := eval_wStoreByte "offLo" s10 fuel (by decide) (by decide)
  have s11_op : s11.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length + ws.regs litLen
          + 1 := by rw [← hs11, st11_op, s10_op]
  have s11_ob : s11.regs "outBase" = ws.regs "outBase" := by rw [← hs11, st11_ob, s10_ob]
  have s11_mlmN : (s11.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs11, wStoreByte_reg "offLo" "mlm" s10 fuel (by decide) (by decide), s10_mlmN]
  -- S12 : offHi := off >> 8
  generalize hs12 : (WStmt.bin WOp.shr "offHi" off (WArg.imm 8)).eval fuel s11 = s12
  have s12_op : s12.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length + ws.regs litLen
          + 1 := by rw [← hs12, bin_reg _ _ _ _ _ s11 fuel (by decide), s11_op]
  have s12_ob : s12.regs "outBase" = ws.regs "outBase" := by
    rw [← hs12, bin_reg _ _ _ _ _ s11 fuel (by decide), s11_ob]
  have s12_mlmN : (s12.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs12, bin_reg _ _ _ _ _ s11 fuel (by decide), s11_mlmN]
  -- S13 : offHi := offHi & 255
  generalize hs13 : (WStmt.bin WOp.band "offHi" "offHi" (WArg.imm 255)).eval fuel s12 = s13
  have s13_op : s13.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length + ws.regs litLen
          + 1 := by rw [← hs13, bin_reg _ _ _ _ _ s12 fuel (by decide), s12_op]
  have s13_ob : s13.regs "outBase" = ws.regs "outBase" := by
    rw [← hs13, bin_reg _ _ _ _ _ s12 fuel (by decide), s12_ob]
  have s13_mlmN : (s13.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs13, bin_reg _ _ _ _ _ s12 fuel (by decide), s12_mlmN]
  -- S14 : store offHi byte
  generalize hs14 : (wStoreByte "offHi").eval fuel s13 = s14
  obtain ⟨st14_op, st14_ob, _⟩ := eval_wStoreByte "offHi" s13 fuel (by decide) (by decide)
  have s14_op : s14.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length + ws.regs litLen
          + 1 + 1 := by rw [← hs14, st14_op, s13_op]
  have s14_ob : s14.regs "outBase" = ws.regs "outBase" := by rw [← hs14, st14_ob, s13_ob]
  have s14_mlmN : (s14.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs14, wStoreByte_reg "offHi" "mlm" s13 fuel (by decide) (by decide), s13_mlmN]
  -- S15 : setp pMatBig
  generalize hs15 : (WStmt.setp SCmp.ge "pMatBig" "mlm" (WArg.imm 15)).eval fuel s14 = s15
  have s15_op : s15.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length + ws.regs litLen
          + 1 + 1 := by rw [← hs15, setp_reg _ _ _ _ _ s14 fuel (by decide), s14_op]
  have s15_ob : s15.regs "outBase" = ws.regs "outBase" := by
    rw [← hs15, setp_reg _ _ _ _ _ s14 fuel (by decide), s14_ob]
  have s15_mlmN : (s15.regs "mlm").toNat = (ws.regs ml).toNat - 4 := by
    rw [← hs15, setp_reg _ _ _ _ _ s14 fuel (by decide), s14_mlmN]
  have s15_pMat : s15.regs "pMatBig" = (if UInt64.ofNat 15 ≤ s15.regs "mlm" then 1 else 0) := by
    rw [← hs15]; simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run]
  -- S16 : match length-extension uif (the last statement)
  obtain ⟨mu_op, mu_fr⟩ := eval_uifExt "pMatBig" "matExtra" "mlm" s15 fuel s15_pMat
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by rw [s15_mlmN]; exact hfuelM)
  refine ⟨?_, ?_⟩
  · rw [mu_op, s15_op, s15_mlmN, arith2]
  · rw [mu_fr "outBase" (by decide) (by decide) (by decide) (by decide) (by decide), s15_ob]

-- ── Supporting array lemmas for byte-content assembly ────────────────────────

/-- `(base+1).toNat = base.toNat + 1` when the window `[base, base+len)` fits
    below `2^64` (no UInt64 wrap). -/
private theorem toNat_succ_of_lt {base : UInt64}
    (hnw : base.toNat + 1 < 2 ^ 64) : (base + 1).toNat = base.toNat + 1 := by
  rw [UInt64.toNat_add, UInt64.toNat_one, Nat.mod_eq_of_lt (by omega)]

theorem mov_reg (d r : String) (b : WArg) (st : WState) (fuel : Nat) (hr : r ≠ d) :
    ((WStmt.mov d b).eval fuel st).regs r = st.regs r := by
  simp [WStmt.eval, WState.setReg, hr]

/-- **`wEmitFinalSeq`** (structural): advances `op` by exactly the encoded length of
    `encodeFinal` of the `litLen` literals (`1` token byte + `encNib` length bytes +
    `litLen` literal bytes) and preserves `outBase`. -/
theorem eval_wEmitFinalSeq (litStart litLen : String) (ws : WState) (fuel : Nat)
    (h1 : litLen ≠ "zero") (h2 : litLen ≠ "tokHi") (h3 : litLen ≠ "tok")
    (h4 : litLen ≠ "sbAddr") (h5 : litLen ≠ "op") (h6 : litLen ≠ "pLitBigF")
    (h7 : litLen ≠ "litExtraF") (h8 : litLen ≠ "c255") (h9 : litLen ≠ "lsicC")
    (h10 : litLen ≠ "cpDstF") (h11 : litLen ≠ "cpSrcF")
    (hfuel : (ws.regs litLen).toNat / 255 < fuel) :
    ((wEmitFinalSeq litStart litLen).eval fuel ws).regs "op"
        = ws.regs "op"
          + UInt64.ofNat (1 + (encNib (ws.regs litLen).toNat).length + (ws.regs litLen).toNat)
    ∧ ((wEmitFinalSeq litStart litLen).eval fuel ws).regs "outBase" = ws.regs "outBase" := by
  have arith : ∀ (op ll : UInt64) (k : Nat),
      op + 1 + UInt64.ofNat k + ll = op + UInt64.ofNat (1 + k + ll.toNat) := by
    intro op ll k
    rw [UInt64.ofNat_add, UInt64.ofNat_add, UInt64.ofNat_toNat, show UInt64.ofNat 1 = 1 from rfl]
    ac_rfl
  simp only [wEmitFinalSeq, wseq, eval_seq]
  -- s1 : after `mov zero 0`
  generalize hs1 : (WStmt.mov "zero" (WArg.imm 0)).eval fuel ws = s1
  have s1_op : s1.regs "op" = ws.regs "op" := by rw [← hs1]; exact mov_reg _ _ _ ws fuel (by decide)
  have s1_ll : s1.regs litLen = ws.regs litLen := by rw [← hs1]; exact mov_reg _ _ _ ws fuel h1
  have s1_ob : s1.regs "outBase" = ws.regs "outBase" := by
    rw [← hs1]; exact mov_reg _ _ _ ws fuel (by decide)
  have s1_z : (s1.regs "zero").toNat < 16 := by
    rw [← hs1]; simp [WStmt.eval, WState.setReg, WArg.eval]
  -- s2 : after `wEmitToken litLen zero`
  generalize hs2 : (wEmitToken litLen "zero").eval fuel s1 = s2
  obtain ⟨t_op, t_ob, _⟩ := eval_wEmitToken litLen "zero" s1 fuel s1_z (by decide) (by decide)
  have s2_op : s2.regs "op" = ws.regs "op" + 1 := by rw [← hs2, t_op, s1_op]
  have s2_ll : s2.regs litLen = ws.regs litLen := by
    rw [← hs2, wEmitToken_reg litLen "zero" litLen s1 fuel h2 h3 h4 h5, s1_ll]
  have s2_ob : s2.regs "outBase" = ws.regs "outBase" := by rw [← hs2, t_ob, s1_ob]
  -- s3 : after `setp .ge pLitBigF litLen 15`
  generalize hs3 : (WStmt.setp SCmp.ge "pLitBigF" litLen (WArg.imm 15)).eval fuel s2 = s3
  have s3_op : s3.regs "op" = ws.regs "op" + 1 := by
    rw [← hs3, setp_reg _ _ _ _ _ s2 fuel (by decide), s2_op]
  have s3_ll : s3.regs litLen = ws.regs litLen := by
    rw [← hs3, setp_reg _ _ _ _ _ s2 fuel h6, s2_ll]
  have s3_ob : s3.regs "outBase" = ws.regs "outBase" := by
    rw [← hs3, setp_reg _ _ _ _ _ s2 fuel (by decide), s2_ob]
  have s3_p : s3.regs "pLitBigF" = (if UInt64.ofNat 15 ≤ s3.regs litLen then 1 else 0) := by
    rw [← hs3]; simp [WStmt.eval, WState.setReg, WArg.eval, SCmp.run, h6]
  -- s4 : after the length-extension `uif`
  obtain ⟨u_op, u_fr⟩ := eval_uifExt "pLitBigF" "litExtraF" litLen s3 fuel s3_p
    (by decide) (by decide) (by decide) (by decide) (by decide) h7
    (by rw [s3_ll]; exact hfuel)
  generalize hs4 : (WStmt.uif "pLitBigF"
      (WStmt.seq (WStmt.bin WOp.sub "litExtraF" litLen (WArg.imm 15)) (wEmitLSIC "litExtraF"))
      WStmt.skip).eval fuel s3 = s4
  have s4_op : s4.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs4, u_op, s3_op, s3_ll]
  have s4_ll : s4.regs litLen = ws.regs litLen := by
    rw [← hs4, u_fr litLen h7 h8 h9 h5 h4, s3_ll]
  have s4_ob : s4.regs "outBase" = ws.regs "outBase" := by
    rw [← hs4, u_fr "outBase" (by decide) (by decide) (by decide) (by decide) (by decide), s3_ob]
  -- s5 : after `bin add cpDstF outBase op`
  generalize hs5 : (WStmt.bin WOp.add "cpDstF" "outBase" (WArg.reg "op")).eval fuel s4 = s5
  have s5_op : s5.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs5, bin_reg _ _ _ _ _ s4 fuel (by decide), s4_op]
  have s5_ll : s5.regs litLen = ws.regs litLen := by
    rw [← hs5, bin_reg _ _ _ _ _ s4 fuel h10, s4_ll]
  have s5_ob : s5.regs "outBase" = ws.regs "outBase" := by
    rw [← hs5, bin_reg _ _ _ _ _ s4 fuel (by decide), s4_ob]
  -- s6 : after `bin add cpSrcF inBase litStart`
  generalize hs6 : (WStmt.bin WOp.add "cpSrcF" "inBase" (WArg.reg litStart)).eval fuel s5 = s6
  have s6_op : s6.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs6, bin_reg _ _ _ _ _ s5 fuel (by decide), s5_op]
  have s6_ll : s6.regs litLen = ws.regs litLen := by
    rw [← hs6, bin_reg _ _ _ _ _ s5 fuel h11, s5_ll]
  have s6_ob : s6.regs "outBase" = ws.regs "outBase" := by
    rw [← hs6, bin_reg _ _ _ _ _ s5 fuel (by decide), s5_ob]
  -- s7 : after the cooperative literal copy
  generalize hs7 : (WStmt.coopCopy "cpDstF" "cpSrcF" litLen).eval fuel s6 = s7
  have s7_op : s7.regs "op"
      = ws.regs "op" + 1 + UInt64.ofNat (encNib (ws.regs litLen).toNat).length := by
    rw [← hs7, coopCopy_reg _ _ _ _ s6 fuel, s6_op]
  have s7_ll : s7.regs litLen = ws.regs litLen := by
    rw [← hs7, coopCopy_reg _ _ _ _ s6 fuel, s6_ll]
  have s7_ob : s7.regs "outBase" = ws.regs "outBase" := by
    rw [← hs7, coopCopy_reg _ _ _ _ s6 fuel, s6_ob]
  -- final `bin add op op litLen`
  refine ⟨?_, ?_⟩
  · rw [show ((WStmt.bin WOp.add "op" "op" (WArg.reg litLen)).eval fuel s7).regs "op"
          = s7.regs "op" + s7.regs litLen from by
        simp [WStmt.eval, WState.setReg, WArg.eval, WOp.run], s7_op, s7_ll, arith]
  · rw [bin_reg _ _ _ _ _ s7 fuel (by decide), s7_ob]

end LZ4WarpEvalBytes
