import AlgorithmLib.LZ4WarpKernelProof
open AlgorithmLib.LZ4WarpDSL AlgorithmLib.LZ4Simt AlgorithmLib.LZ4WarpFind
open AlgorithmLib.LZ4SimtBits (ballotOf_testBit ballotOf_toNat)
open AlgorithmLib.LZ4Ptx (toNat_ofNat_lt u64_add_ofNat u64_sub_ofNat)

set_option maxHeartbeats 1200000
set_option maxRecDepth 4000

namespace AlgorithmLib.LZ4WarpDSL

-- Reduction+extract: from a state `st` positioned at the vote (step 40 of the
-- window segment), the 11 instrs `vote;brev;clz;setp;andp;bin;stshp;barwarp;
-- binr;shfl;setp` compute `found = (bal ≠ 0)` where `bal = ballotOf pHit`.
theorem cwFound (prog : Array SInstr) (st : SState) (l : Fin 32)
    (h0 : prog[st.pc]? = some (.vote "bal" "pHit"))
    (h1 : prog[st.pc + 1]? = some (.brev "rev" "bal"))
    (h2 : prog[st.pc + 2]? = some (.clz "fl" "rev"))
    (h3 : prog[st.pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h4 : prog[st.pc + 4]? = some (.andp "pIns" "pLe" "pValid"))
    (h5 : prog[st.pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h6 : prog[st.pc + 6]? = some (.stshp "pIns" "addr" "pp1"))
    (h7 : prog[st.pc + 7]? = some (.barwarp))
    (h8 : prog[st.pc + 8]? = some (.binr .add "p0" "searchPos" "fl"))
    (h9 : prog[st.pc + 9]? = some (.shfl "cand0" "cand" "fl"))
    (h10 : prog[st.pc + 10]? = some (.setp .ne "found" "bal" (.imm 0))) :
    (snsteps prog 11 st).regs "found" l
      = (if ballotOf st.regs "pHit" == 0 then 0 else 1) := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10,
    sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run]

/-- The 11-instr `vote…found` tail preserves `searchPos` (none of it writes `searchPos`). -/
theorem cwTail_searchPos (prog : Array SInstr) (st : SState) (l : Fin 32)
    (h0 : prog[st.pc]? = some (.vote "bal" "pHit"))
    (h1 : prog[st.pc + 1]? = some (.brev "rev" "bal"))
    (h2 : prog[st.pc + 2]? = some (.clz "fl" "rev"))
    (h3 : prog[st.pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h4 : prog[st.pc + 4]? = some (.andp "pIns" "pLe" "pValid"))
    (h5 : prog[st.pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h6 : prog[st.pc + 6]? = some (.stshp "pIns" "addr" "pp1"))
    (h7 : prog[st.pc + 7]? = some (.barwarp))
    (h8 : prog[st.pc + 8]? = some (.binr .add "p0" "searchPos" "fl"))
    (h9 : prog[st.pc + 9]? = some (.shfl "cand0" "cand" "fl"))
    (h10 : prog[st.pc + 10]? = some (.setp .ne "found" "bal" (.imm 0))) :
    (snsteps prog 11 st).regs "searchPos" l = st.regs "searchPos" l := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10,
    sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run]

/-- `p0` ends holding `searchPos + fl` (uniform, since both summands are). -/
theorem cwP0 (prog : Array SInstr) (st : SState) (l : Fin 32)
    (h0 : prog[st.pc]? = some (.vote "bal" "pHit"))
    (h1 : prog[st.pc + 1]? = some (.brev "rev" "bal"))
    (h2 : prog[st.pc + 2]? = some (.clz "fl" "rev"))
    (h3 : prog[st.pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h4 : prog[st.pc + 4]? = some (.andp "pIns" "pLe" "pValid"))
    (h5 : prog[st.pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h6 : prog[st.pc + 6]? = some (.stshp "pIns" "addr" "pp1"))
    (h7 : prog[st.pc + 7]? = some (.barwarp))
    (h8 : prog[st.pc + 8]? = some (.binr .add "p0" "searchPos" "fl"))
    (h9 : prog[st.pc + 9]? = some (.shfl "cand0" "cand" "fl"))
    (h10 : prog[st.pc + 10]? = some (.setp .ne "found" "bal" (.imm 0))) :
    (snsteps prog 11 st).regs "p0" l
      = st.regs "searchPos" l + clz32 (brev32 (ballotOf st.regs "pHit")) := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10,
    sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run]

/-- `cand0` ends holding `cand` broadcast from the winning lane (`shfl … fl`). -/
theorem cwCand0 (prog : Array SInstr) (st : SState) (l : Fin 32)
    (h0 : prog[st.pc]? = some (.vote "bal" "pHit"))
    (h1 : prog[st.pc + 1]? = some (.brev "rev" "bal"))
    (h2 : prog[st.pc + 2]? = some (.clz "fl" "rev"))
    (h3 : prog[st.pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h4 : prog[st.pc + 4]? = some (.andp "pIns" "pLe" "pValid"))
    (h5 : prog[st.pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h6 : prog[st.pc + 6]? = some (.stshp "pIns" "addr" "pp1"))
    (h7 : prog[st.pc + 7]? = some (.barwarp))
    (h8 : prog[st.pc + 8]? = some (.binr .add "p0" "searchPos" "fl"))
    (h9 : prog[st.pc + 9]? = some (.shfl "cand0" "cand" "fl"))
    (h10 : prog[st.pc + 10]? = some (.setp .ne "found" "bal" (.imm 0))) :
    (snsteps prog 11 st).regs "cand0" l
      = st.regs "cand" (toLane (clz32 (brev32 (ballotOf st.regs "pHit")))) := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10,
    sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run]

-- ── Model-side isSome characterisations of `firstHit` / `coopWindow` ─────────────

/-- A predicated OR-fold from 0 stays 0 when the predicate never fires. -/
theorem foldl_or_zero (p : Fin 32 → Bool) (f : Fin 32 → Nat) :
    ∀ (L : List (Fin 32)), (∀ x ∈ L, p x = false) →
      L.foldl (fun acc l => if p l then acc ||| f l else acc) 0 = 0 := by
  intro L
  induction L with
  | nil => intro _; rfl
  | cons a t ih =>
      intro h
      rw [List.foldl_cons, if_neg (by rw [h a (by simp)]; simp)]
      exact ih (fun x hx => h x (by simp [hx]))

/-- `ballotOf p = 0` (as `UInt64`) iff no lane holds `1`. -/
theorem ballot_eq_zero_iff (regs : String → Lane → UInt64) (p : String) :
    (ballotOf regs p == 0) = true ↔ ∀ l : Fin 32, (regs p l == 1) = false := by
  rw [beq_iff_eq]
  constructor
  · intro h l
    have hb := ballotOf_testBit regs p l
    rw [h] at hb
    simp only [UInt64.toNat_ofNat, Nat.zero_mod, Nat.zero_testBit] at hb
    exact hb.symm
  · intro h
    have : (ballotOf regs p).toNat = 0 := by
      rw [ballotOf_toNat]
      exact foldl_or_zero (fun l => regs p l == 1) (fun l => 2 ^ l.val)
        (List.finRange 32) (fun x _ => h x)
    exact UInt64.toNat_inj.mp (by rw [this]; rfl)

/-- If `firstHit` returns an index, the predicate holds there. -/
theorem firstHit_some_pred (P : Nat → Bool) :
    ∀ (b start L : Nat), AlgorithmLib.LZ4WarpSched.firstHit P b start = some L → P L = true := by
  intro b
  induction b with
  | zero => intro start L h; simp [AlgorithmLib.LZ4WarpSched.firstHit] at h
  | succ n ih =>
      intro start L h
      rw [AlgorithmLib.LZ4WarpSched.firstHit] at h
      by_cases hs : P start = true
      · rw [if_pos hs] at h; rw [Option.some.injEq] at h; rw [← h]; exact hs
      · rw [if_neg hs] at h; exact ih (start + 1) L h

theorem firstHit_isSome (P : Nat → Bool) :
    ∀ (b start : Nat), (AlgorithmLib.LZ4WarpSched.firstHit P b start).isSome = true
      ↔ ∃ k, start ≤ k ∧ k < start + b ∧ P k = true := by
  intro b
  induction b with
  | zero => intro start; simp only [AlgorithmLib.LZ4WarpSched.firstHit, Option.isSome_none,
      Bool.false_eq_true, false_iff, not_exists]; intro k; omega
  | succ n ih =>
      intro start
      show (if P start then some start
              else AlgorithmLib.LZ4WarpSched.firstHit P n (start + 1)).isSome = true ↔ _
      by_cases hs : P start = true
      · rw [if_pos hs]
        simp only [Option.isSome_some, true_iff]
        exact ⟨start, Nat.le_refl _, by omega, hs⟩
      · rw [if_neg hs, ih (start + 1)]
        constructor
        · rintro ⟨k, hk1, hk2, hk3⟩; exact ⟨k, by omega, by omega, hk3⟩
        · rintro ⟨k, hk1, hk2, hk3⟩
          rcases Nat.eq_or_lt_of_le hk1 with he | hlt
          · exact absurd (he ▸ hk3) (by simp [hs])
          · exact ⟨k, hlt, by omega, hk3⟩

/-- `coopWindow` is `some` exactly when the earliest-hit scan is. -/
theorem coopWindow_isSome (inp : List UInt8) (oracle : Nat → Option Nat) (searchLim s : Nat) :
    (AlgorithmLib.LZ4WarpSched.coopWindow inp oracle searchLim s).isSome
      = (AlgorithmLib.LZ4WarpSched.firstHit
          (fun L => (probe inp oracle searchLim (s + L)).isSome) 32 0).isSome := by
  unfold AlgorithmLib.LZ4WarpSched.coopWindow
  cases hf : AlgorithmLib.LZ4WarpSched.firstHit
      (fun L => (probe inp oracle searchLim (s + L)).isSome) 32 0 with
  | none => simp
  | some L =>
      have hp : (probe inp oracle searchLim (s + L)).isSome = true :=
        firstHit_some_pred _ _ _ _ hf
      simp only [Option.isSome_some]
      cases hpr : probe inp oracle searchLim (s + L) with
      | none => rw [hpr] at hp; simp at hp
      | some c => simp

/-- Abstract bridge (small context — kernel-cheap): the earliest-hit lane offset
    equals the model `window`'s position offset `p - s`.  Kept separate from the
    heavy 40-instruction `coopWindow_upto40_eq` so the nested `match`-on-`match`
    reduction is checked in a tiny context (mirrors `coopWindow_isSome`). -/
theorem firstHit_getD_eq_window (inp : List UInt8) (oracle : Nat → Option Nat)
    (searchLim s : Nat) :
    (AlgorithmLib.LZ4WarpSched.firstHit
        (fun L => (probe inp oracle searchLim (s + L)).isSome) 32 0).getD 32
      = (match AlgorithmLib.LZ4WarpFind.window inp oracle searchLim s with
          | some (p, _) => p - s | none => 32) := by
  rw [← AlgorithmLib.LZ4WarpSched.coopWindow_eq_window]
  unfold AlgorithmLib.LZ4WarpSched.coopWindow
  cases hf : AlgorithmLib.LZ4WarpSched.firstHit
      (fun L => (probe inp oracle searchLim (s + L)).isSome) 32 0 with
  | none => rfl
  | some L =>
      have hp : (probe inp oracle searchLim (s + L)).isSome = true :=
        firstHit_some_pred _ _ _ _ hf
      cases hpr : probe inp oracle searchLim (s + L) with
      | none => rw [hpr] at hp; simp at hp
      | some c =>
          dsimp only
          rw [hpr]
          show L = s + L - s
          omega

-- ── Address-only variant of `cw_candRaw` (5 instrs, before the `ldsh`) ────────

/-- The 5-instr hash-address computation (`mul,shr,band,shl,add`) gives `addr`'s
    value directly as the model `tableOracle`/`tableInsert` index `tbl + 2*wHash`.
    (Same technique as `cw_candRaw`, stopping one instruction earlier.) -/
theorem cw_addr5 (prog : Array SInstr) (ss : SState) (l : Fin 32) (wl hashLog tbl : Nat)
    (hv : ss.regs "v32" l = UInt64.ofNat wl)
    (htbl : ss.regs "tbl" l = UInt64.ofNat tbl)
    (h0 : prog[ss.pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h1 : prog[ss.pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h2 : prog[ss.pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h3 : prog[ss.pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h4 : prog[ss.pc + 4]? = some (.binr .add "addr" "hh" "tbl"))
    (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40) :
    ((snsteps prog 5 ss).regs "addr" l).toNat
      = tbl + 2 * (((UInt64.ofNat wl * UInt64.ofNat wHashK) >>> UInt64.ofNat (32 - hashLog)).toNat % 2 ^ hashLog) := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, sstepInstr, SState.setReg, SState.setPc, SState.get,
    SOp.run, hv, htbl]
  have hpow : (18446744073709551616 : Nat) = 2 ^ 64 := by rfl
  rw [hpow, cw_addr_nat ((wl * wHashK % 2 ^ 64) >>> ((32 - hashLog) % 64)) hashLog tbl hhl htblb]

theorem segB_pc (prog : Array SInstr) (st : SState) (hashLog : Nat)
    (h0 : prog[st.pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h1 : prog[st.pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h2 : prog[st.pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h3 : prog[st.pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h4 : prog[st.pc + 4]? = some (.binr .add "addr" "hh" "tbl"))
    (h5 : prog[st.pc + 5]? = some (.ldsh "candRaw" "addr")) :
    (snsteps prog 6 st).pc = st.pc + 6 := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, sstepInstr, SState.setReg, SState.setPc]

theorem segB_gmem (prog : Array SInstr) (st : SState) (hashLog : Nat)
    (h0 : prog[st.pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h1 : prog[st.pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h2 : prog[st.pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h3 : prog[st.pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h4 : prog[st.pc + 4]? = some (.binr .add "addr" "hh" "tbl"))
    (h5 : prog[st.pc + 5]? = some (.ldsh "candRaw" "addr")) :
    (snsteps prog 6 st).gmem = st.gmem := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, sstepInstr, SState.setReg, SState.setPc]

theorem segB_frame (prog : Array SInstr) (st : SState) (r : String) (l : Fin 32) (hashLog : Nat)
    (h0 : prog[st.pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h1 : prog[st.pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h2 : prog[st.pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h3 : prog[st.pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h4 : prog[st.pc + 4]? = some (.binr .add "addr" "hh" "tbl"))
    (h5 : prog[st.pc + 5]? = some (.ldsh "candRaw" "addr"))
    (hr : r ∉ ["hh", "addr", "candRaw"]) :
    (snsteps prog 6 st).regs r l = st.regs r l := by
  simp only [List.mem_cons, List.mem_singleton, not_or, List.not_mem_nil, not_false_iff,
    and_true] at hr
  obtain ⟨n0, n1, n2⟩ := hr
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, sstepInstr, SState.setReg, SState.setPc,
    SState.get, SOp.run, n0, n1, n2]

theorem segC_pc (prog : Array SInstr) (st : SState)
    (h0 : prog[st.pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h1 : prog[st.pc + 1]? = some (.binr .min "rc" "cand" "cap4"))
    (h2 : prog[st.pc + 2]? = some (.binr .add "rcA" "inBase" "rc"))
    (h3 : prog[st.pc + 3]? = some (.ldgo "c0" "rcA" 0))
    (h4 : prog[st.pc + 4]? = some (.ldgo "c1" "rcA" 1))
    (h5 : prog[st.pc + 5]? = some (.ldgo "c2" "rcA" 2))
    (h6 : prog[st.pc + 6]? = some (.ldgo "c3" "rcA" 3))
    (h7 : prog[st.pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h8 : prog[st.pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h9 : prog[st.pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h10 : prog[st.pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h11 : prog[st.pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h12 : prog[st.pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3"))) :
    (snsteps prog 13 st).pc = st.pc + 13 := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12,
    sstepInstr, SState.setReg, SState.setPc]

theorem segC_frame (prog : Array SInstr) (st : SState) (r : String) (l : Fin 32)
    (h0 : prog[st.pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h1 : prog[st.pc + 1]? = some (.binr .min "rc" "cand" "cap4"))
    (h2 : prog[st.pc + 2]? = some (.binr .add "rcA" "inBase" "rc"))
    (h3 : prog[st.pc + 3]? = some (.ldgo "c0" "rcA" 0))
    (h4 : prog[st.pc + 4]? = some (.ldgo "c1" "rcA" 1))
    (h5 : prog[st.pc + 5]? = some (.ldgo "c2" "rcA" 2))
    (h6 : prog[st.pc + 6]? = some (.ldgo "c3" "rcA" 3))
    (h7 : prog[st.pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h8 : prog[st.pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h9 : prog[st.pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h10 : prog[st.pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h11 : prog[st.pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h12 : prog[st.pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (hr : r ∉ ["cand", "rc", "rcA", "c0", "c1", "c2", "c3", "vc"]) :
    (snsteps prog 13 st).regs r l = st.regs r l := by
  simp only [List.mem_cons, List.mem_singleton, not_or, List.not_mem_nil, not_false_iff,
    and_true] at hr
  obtain ⟨n0, n1, n2, n3, n4, n5, n6, n7⟩ := hr
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12,
    sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run,
    n0, n1, n2, n3, n4, n5, n6, n7]

theorem segC_cand (prog : Array SInstr) (st : SState) (l : Fin 32)
    (h0 : prog[st.pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h1 : prog[st.pc + 1]? = some (.binr .min "rc" "cand" "cap4"))
    (h2 : prog[st.pc + 2]? = some (.binr .add "rcA" "inBase" "rc"))
    (h3 : prog[st.pc + 3]? = some (.ldgo "c0" "rcA" 0))
    (h4 : prog[st.pc + 4]? = some (.ldgo "c1" "rcA" 1))
    (h5 : prog[st.pc + 5]? = some (.ldgo "c2" "rcA" 2))
    (h6 : prog[st.pc + 6]? = some (.ldgo "c3" "rcA" 3))
    (h7 : prog[st.pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h8 : prog[st.pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h9 : prog[st.pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h10 : prog[st.pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h11 : prog[st.pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h12 : prog[st.pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3"))) :
    (snsteps prog 13 st).regs "cand" l = st.regs "candRaw" l - UInt64.ofNat 1 := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12,
    sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]

-- ── Main bridge: per-lane `pHit` (after 40 steps) ↔ `probe … .isSome` ────────
-- (Adapted from `_integ.lean`'s `coopWindow_probe_iff`, generalized to an
-- arbitrary `tbl` and to *all* lanes — including invalid ones (`s+l.val ≥
-- searchLim`), which the un-generalized version didn't need to cover.)

/-- The 40 instruction facts of the `coopWindow` probe body (segments A/B/C/D),
    bundled into ONE argument.  Passing 40 separate `prog[base+i]? = ...` hyps into
    several composed lemmas builds a huge application spine that overflows the
    kernel C-stack; a single structure argument collapses it. -/
structure ProbeInstrs (prog : Array SInstr) (base searchLim capC hashLog : Nat) : Prop where
  h0 : prog[base]? = some (.binr .add "posP" "searchPos" "lane")
  h1 : prog[base+1]? = some (.setp .lt "pValid" "posP" (.imm searchLim))
  h2 : prog[base+2]? = some (.mov "cap4" (.imm capC))
  h3 : prog[base+3]? = some (.binr .min "rp" "posP" "cap4")
  h4 : prog[base+4]? = some (.binr .add "rpA" "inBase" "rp")
  h5 : prog[base+5]? = some (.ldgo "b0" "rpA" 0)
  h6 : prog[base+6]? = some (.ldgo "b1" "rpA" 1)
  h7 : prog[base+7]? = some (.ldgo "b2" "rpA" 2)
  h8 : prog[base+8]? = some (.ldgo "b3" "rpA" 3)
  h9 : prog[base+9]? = some (.bin .shl "b1" "b1" (.imm 8))
  h10 : prog[base+10]? = some (.bin .shl "b2" "b2" (.imm 16))
  h11 : prog[base+11]? = some (.bin .shl "b3" "b3" (.imm 24))
  h12 : prog[base+12]? = some (.bin .bor "v32" "b0" (.reg "b1"))
  h13 : prog[base+13]? = some (.bin .bor "v32" "v32" (.reg "b2"))
  h14 : prog[base+14]? = some (.bin .bor "v32" "v32" (.reg "b3"))
  h15 : prog[base+15]? = some (.bin .mul "hh" "v32" (.imm wHashK))
  h16 : prog[base+16]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog)))
  h17 : prog[base+17]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1)))
  h18 : prog[base+18]? = some (.bin .shl "hh" "hh" (.imm 1))
  h19 : prog[base+19]? = some (.binr .add "addr" "hh" "tbl")
  h20 : prog[base+20]? = some (.ldsh "candRaw" "addr")
  h21 : prog[base+21]? = some (.bin .sub "cand" "candRaw" (.imm 1))
  h22 : prog[base+22]? = some (.binr .min "rc" "cand" "cap4")
  h23 : prog[base+23]? = some (.binr .add "rcA" "inBase" "rc")
  h24 : prog[base+24]? = some (.ldgo "c0" "rcA" 0)
  h25 : prog[base+25]? = some (.ldgo "c1" "rcA" 1)
  h26 : prog[base+26]? = some (.ldgo "c2" "rcA" 2)
  h27 : prog[base+27]? = some (.ldgo "c3" "rcA" 3)
  h28 : prog[base+28]? = some (.bin .shl "c1" "c1" (.imm 8))
  h29 : prog[base+29]? = some (.bin .shl "c2" "c2" (.imm 16))
  h30 : prog[base+30]? = some (.bin .shl "c3" "c3" (.imm 24))
  h31 : prog[base+31]? = some (.bin .bor "vc" "c0" (.reg "c1"))
  h32 : prog[base+32]? = some (.bin .bor "vc" "vc" (.reg "c2"))
  h33 : prog[base+33]? = some (.bin .bor "vc" "vc" (.reg "c3"))
  h34 : prog[base+34]? = some (.setp .ne "pNE" "candRaw" (.imm 0))
  h35 : prog[base+35]? = some (.setp .lt "pCO" "cand" (.reg "posP"))
  h36 : prog[base+36]? = some (.setp .eq "pEq" "vc" (.reg "v32"))
  h37 : prog[base+37]? = some (.andp "pH1" "pValid" "pNE")
  h38 : prog[base+38]? = some (.andp "pH2" "pH1" "pCO")
  h39 : prog[base+39]? = some (.andp "pHit" "pH2" "pEq")

theorem coopWindow_pHit_iff (prog : Array SInstr) (ss : SState) (base ib : Nat)
    (s inStride searchLim capC hashLog tbl : Nat) (l : Fin 32)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[base+1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[base+2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[base+3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[base+4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[base+5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[base+6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[base+7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[base+8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[base+9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[base+10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[base+11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[base+12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[base+13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[base+14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (h15 : prog[base+15]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h16 : prog[base+16]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h17 : prog[base+17]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h18 : prog[base+18]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h19 : prog[base+19]? = some (.binr .add "addr" "hh" "tbl"))
    (h20 : prog[base+20]? = some (.ldsh "candRaw" "addr"))
    (h21 : prog[base+21]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h22 : prog[base+22]? = some (.binr .min "rc" "cand" "cap4"))
    (h23 : prog[base+23]? = some (.binr .add "rcA" "inBase" "rc"))
    (h24 : prog[base+24]? = some (.ldgo "c0" "rcA" 0))
    (h25 : prog[base+25]? = some (.ldgo "c1" "rcA" 1))
    (h26 : prog[base+26]? = some (.ldgo "c2" "rcA" 2))
    (h27 : prog[base+27]? = some (.ldgo "c3" "rcA" 3))
    (h28 : prog[base+28]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h29 : prog[base+29]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h30 : prog[base+30]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h31 : prog[base+31]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h32 : prog[base+32]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h33 : prog[base+33]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (h34 : prog[base+34]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h35 : prog[base+35]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h36 : prog[base+36]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h37 : prog[base+37]? = some (.andp "pH1" "pValid" "pNE"))
    (h38 : prog[base+38]? = some (.andp "pH2" "pH1" "pCO"))
    (h39 : prog[base+39]? = some (.andp "pHit" "pH2" "pEq"))
    (hinb : ss.regs "inBase" l = UInt64.ofNat ib)
    (htbl : ss.regs "tbl" l = UInt64.ofNat tbl)
    (hsl : searchLim ≤ capC)
    (hcapdef : capC = inStride - 4)
    (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32)
    (htblb : tbl < 2 ^ 40)
    (hcapv : s + l.val ≤ capC)
    (hlane : ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + l.val < 2 ^ 64) (hib64 : ib + capC < 2 ^ 64) :
    ((snsteps prog 40 ss).regs "pHit" l = 1) ↔
      (probe (gmemInpAt ss.gmem ib inStride) (tableOracle ss.gmem ss.smem hashLog tbl ib)
        searchLim (s + l.val)).isSome := by
  subst hpc
  have e21 : snsteps prog 21 ss = snsteps prog 6 (snsteps prog 15 ss) := snsteps_add prog 15 6 ss
  have e34 : snsteps prog 34 ss = snsteps prog 13 (snsteps prog 21 ss) := snsteps_add prog 21 13 ss
  have e40 : snsteps prog 40 ss = snsteps prog 6 (snsteps prog 34 ss) := snsteps_add prog 34 6 ss
  ----------------------------------------------------------------
  have hpc15 : (snsteps prog 15 ss).pc = ss.pc + 15 := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc]
  have hB0 : prog[(snsteps prog 15 ss).pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)) := by rw [hpc15]; exact h15
  have hB1 : prog[(snsteps prog 15 ss).pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))) := by rw [hpc15]; exact h16
  have hB2 : prog[(snsteps prog 15 ss).pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))) := by rw [hpc15]; exact h17
  have hB3 : prog[(snsteps prog 15 ss).pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)) := by rw [hpc15]; exact h18
  have hB4 : prog[(snsteps prog 15 ss).pc + 4]? = some (.binr .add "addr" "hh" "tbl") := by rw [hpc15]; exact h19
  have hB5 : prog[(snsteps prog 15 ss).pc + 5]? = some (.ldsh "candRaw" "addr") := by rw [hpc15]; exact h20
  have hpc21 : (snsteps prog 21 ss).pc = ss.pc + 21 := by
    rw [e21, segB_pc prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4 hB5, hpc15]
  have hC0 : prog[(snsteps prog 21 ss).pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)) := by rw [hpc21]; exact h21
  have hC1 : prog[(snsteps prog 21 ss).pc + 1]? = some (.binr .min "rc" "cand" "cap4") := by rw [hpc21]; exact h22
  have hC2 : prog[(snsteps prog 21 ss).pc + 2]? = some (.binr .add "rcA" "inBase" "rc") := by rw [hpc21]; exact h23
  have hC3 : prog[(snsteps prog 21 ss).pc + 3]? = some (.ldgo "c0" "rcA" 0) := by rw [hpc21]; exact h24
  have hC4 : prog[(snsteps prog 21 ss).pc + 4]? = some (.ldgo "c1" "rcA" 1) := by rw [hpc21]; exact h25
  have hC5 : prog[(snsteps prog 21 ss).pc + 5]? = some (.ldgo "c2" "rcA" 2) := by rw [hpc21]; exact h26
  have hC6 : prog[(snsteps prog 21 ss).pc + 6]? = some (.ldgo "c3" "rcA" 3) := by rw [hpc21]; exact h27
  have hC7 : prog[(snsteps prog 21 ss).pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)) := by rw [hpc21]; exact h28
  have hC8 : prog[(snsteps prog 21 ss).pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)) := by rw [hpc21]; exact h29
  have hC9 : prog[(snsteps prog 21 ss).pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)) := by rw [hpc21]; exact h30
  have hC10 : prog[(snsteps prog 21 ss).pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")) := by rw [hpc21]; exact h31
  have hC11 : prog[(snsteps prog 21 ss).pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")) := by rw [hpc21]; exact h32
  have hC12 : prog[(snsteps prog 21 ss).pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3")) := by rw [hpc21]; exact h33
  have hpc34 : (snsteps prog 34 ss).pc = ss.pc + 34 := by
    rw [e34, segC_pc prog (snsteps prog 21 ss) hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12, hpc21]
  have hD0 : prog[(snsteps prog 34 ss).pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)) := by rw [hpc34]; exact h34
  have hD1 : prog[(snsteps prog 34 ss).pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")) := by rw [hpc34]; exact h35
  have hD2 : prog[(snsteps prog 34 ss).pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")) := by rw [hpc34]; exact h36
  have hD3 : prog[(snsteps prog 34 ss).pc + 3]? = some (.andp "pH1" "pValid" "pNE") := by rw [hpc34]; exact h37
  have hD4 : prog[(snsteps prog 34 ss).pc + 4]? = some (.andp "pH2" "pH1" "pCO") := by rw [hpc34]; exact h38
  have hD5 : prog[(snsteps prog 34 ss).pc + 5]? = some (.andp "pHit" "pH2" "pEq") := by rw [hpc34]; exact h39
  ----------------------------------------------------------------
  have hpval : (ss.regs "searchPos" l + ss.regs "lane" l).toNat = s + l.val := by
    rw [hsp, hlane]; exact u64_add_ofNat s l.val hp64
  have hxle : ss.regs "searchPos" l + ss.regs "lane" l ≤ UInt64.ofNat capC := by
    rw [UInt64.le_iff_toNat_le, hpval, toNat_ofNat_lt capC hcapb]; exact hcapv
  have hwl_lt : wLoad4 ss.gmem (ib + (s + l.val)) < 2 ^ 32 := by
    unfold wLoad4
    have a0 := (ss.gmem.getD (ib + (s + l.val)) 0).toNat_lt
    have a1 := (ss.gmem.getD (ib + (s + l.val) + 1) 0).toNat_lt
    have a2 := (ss.gmem.getD (ib + (s + l.val) + 2) 0).toNat_lt
    have a3 := (ss.gmem.getD (ib + (s + l.val) + 3) 0).toNat_lt
    omega
  ----------------------------------------------------------------
  have hpsum : ss.regs "searchPos" l + ss.regs "lane" l = UInt64.ofNat (s + l.val) := by
    rw [hsp, hlane]; exact (UInt64.ofNat_add s l.val).symm
  have hidx : (SOp.run .add (ss.regs "inBase" l)
        (SOp.run .min (SOp.run .add (ss.regs "searchPos" l) (ss.regs "lane" l))
          (UInt64.ofNat capC))).toNat = ib + (s + l.val) := by
    show (ss.regs "inBase" l + (if ss.regs "searchPos" l + ss.regs "lane" l ≤ UInt64.ofNat capC
            then ss.regs "searchPos" l + ss.regs "lane" l else UInt64.ofNat capC)).toNat
      = ib + (s + l.val)
    rw [hinb, if_pos hxle, hpsum]; exact u64_add_ofNat ib (s + l.val) (by omega)
  have v32_15 : (snsteps prog 15 ss).regs "v32" l = UInt64.ofNat (wLoad4 ss.gmem (ib + (s + l.val))) := by
    have hc := cw_v32 prog ss searchLim capC h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 l
    rw [hidx] at hc
    rw [← UInt64.toNat_inj, hc, toNat_ofNat_lt _ (by omega)]
  have posP15 : (snsteps prog 15 ss).regs "posP" l = ss.regs "searchPos" l + ss.regs "lane" l := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
  have pValid15 : (snsteps prog 15 ss).regs "pValid" l
      = (if ss.regs "searchPos" l + ss.regs "lane" l < UInt64.ofNat searchLim then 1 else 0) := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run]
  have cap415 : (snsteps prog 15 ss).regs "cap4" l = UInt64.ofNat capC := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
  have tbl15 : (snsteps prog 15 ss).regs "tbl" l = UInt64.ofNat tbl := by
    have : (snsteps prog 15 ss).regs "tbl" l = ss.regs "tbl" l := by
      simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
        sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
    rw [this, htbl]
  have inBase15 : (snsteps prog 15 ss).regs "inBase" l = UInt64.ofNat ib := by
    have : (snsteps prog 15 ss).regs "inBase" l = ss.regs "inBase" l := by
      simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
        sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
    rw [this, hinb]
  have gmem15 : (snsteps prog 15 ss).gmem = ss.gmem := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc]
  have smem15 : (snsteps prog 15 ss).smem = ss.smem := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc]
  ----------------------------------------------------------------
  have gmem21 : (snsteps prog 21 ss).gmem = ss.gmem := by
    rw [e21, segB_gmem prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4 hB5, gmem15]
  have inBase21 : (snsteps prog 21 ss).regs "inBase" l = UInt64.ofNat ib := by
    rw [e21, segB_frame prog (snsteps prog 15 ss) "inBase" l hashLog hB0 hB1 hB2 hB3 hB4 hB5
      (by decide), inBase15]
  have cap421 : (snsteps prog 21 ss).regs "cap4" l = UInt64.ofNat capC := by
    rw [e21, segB_frame prog (snsteps prog 15 ss) "cap4" l hashLog hB0 hB1 hB2 hB3 hB4 hB5
      (by decide), cap415]
  have candRaw21_nat : ((snsteps prog 21 ss).regs "candRaw" l).toNat
      = (ss.smem.getD (tbl + 2 * wHash ss.gmem hashLog (ib + (s + l.val))) 0).toNat
        + 256 * (ss.smem.getD (tbl + 2 * wHash ss.gmem hashLog (ib + (s + l.val)) + 1) 0).toNat := by
    rw [e21]
    have hc := cw_candRaw prog (snsteps prog 15 ss) l (wLoad4 ss.gmem (ib + (s + l.val))) hashLog tbl
      v32_15 tbl15 hB0 hB1 hB2 hB3 hB4 hB5 hhl htblb
    rw [smem15] at hc
    rw [hc]
    rfl
  have raw_lt : ((snsteps prog 21 ss).regs "candRaw" l).toNat < 2 ^ 64 := by
    rw [candRaw21_nat]
    have a0 := (ss.smem.getD (tbl + 2 * wHash ss.gmem hashLog (ib + (s + l.val))) 0).toNat_lt
    have a1 := (ss.smem.getD (tbl + 2 * wHash ss.gmem hashLog (ib + (s + l.val)) + 1) 0).toNat_lt
    omega
  ----------------------------------------------------------------
  have pValid34 : (snsteps prog 34 ss).regs "pValid" l
      = (if s + l.val < searchLim then 1 else 0) := by
    have h34 : (snsteps prog 34 ss).regs "pValid" l = (snsteps prog 15 ss).regs "pValid" l := by
      rw [e34, segC_frame prog (snsteps prog 21 ss) "pValid" l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7
        hC8 hC9 hC10 hC11 hC12 (by decide)]
      rw [e21, segB_frame prog (snsteps prog 15 ss) "pValid" l hashLog hB0 hB1 hB2 hB3 hB4 hB5
        (by decide)]
    rw [h34, pValid15]
    by_cases hv : s + l.val < searchLim
    · rw [if_pos hv, if_pos (by rw [UInt64.lt_iff_toNat_lt, hpval, toNat_ofNat_lt searchLim (by omega)]; exact hv)]
    · rw [if_neg hv, if_neg (by rw [UInt64.lt_iff_toNat_lt, hpval, toNat_ofNat_lt searchLim (by omega)]; exact hv)]
  have v32_34 : (snsteps prog 34 ss).regs "v32" l = UInt64.ofNat (wLoad4 ss.gmem (ib + (s + l.val))) := by
    have h34 : (snsteps prog 34 ss).regs "v32" l = (snsteps prog 15 ss).regs "v32" l := by
      rw [e34, segC_frame prog (snsteps prog 21 ss) "v32" l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7
        hC8 hC9 hC10 hC11 hC12 (by decide)]
      rw [e21, segB_frame prog (snsteps prog 15 ss) "v32" l hashLog hB0 hB1 hB2 hB3 hB4 hB5
        (by decide)]
    rw [h34, v32_15]
  have posP34 : (snsteps prog 34 ss).regs "posP" l = ss.regs "searchPos" l + ss.regs "lane" l := by
    rw [e34, segC_frame prog (snsteps prog 21 ss) "posP" l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7
      hC8 hC9 hC10 hC11 hC12 (by decide)]
    rw [e21, segB_frame prog (snsteps prog 15 ss) "posP" l hashLog hB0 hB1 hB2 hB3 hB4 hB5
      (by decide)]
    exact posP15
  have candRaw34 : (snsteps prog 34 ss).regs "candRaw" l = (snsteps prog 21 ss).regs "candRaw" l := by
    rw [e34, segC_frame prog (snsteps prog 21 ss) "candRaw" l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7
      hC8 hC9 hC10 hC11 hC12 (by decide)]
  have cand34 : (snsteps prog 34 ss).regs "cand" l
      = (snsteps prog 21 ss).regs "candRaw" l - UInt64.ofNat 1 := by
    rw [e34]
    exact segC_cand prog (snsteps prog 21 ss) l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12
  have vc34_nat : ((snsteps prog 34 ss).regs "vc" l).toNat
      = wLoad4 ss.gmem (SOp.run .add (UInt64.ofNat ib)
          (SOp.run .min (SOp.run .sub ((snsteps prog 21 ss).regs "candRaw" l) (UInt64.ofNat 1))
            (UInt64.ofNat capC))).toNat := by
    rw [e34]
    have hc := cw_vc prog (snsteps prog 21 ss) l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12
    rw [hc, gmem21, inBase21, cap421]
  ----------------------------------------------------------------
  have hwl_any : ∀ q, wLoad4 ss.gmem q < 2 ^ 64 := by
    intro q; unfold wLoad4
    have a0 := (ss.gmem.getD q 0).toNat_lt
    have a1 := (ss.gmem.getD (q + 1) 0).toNat_lt
    have a2 := (ss.gmem.getD (q + 2) 0).toNat_lt
    have a3 := (ss.gmem.getD (q + 3) 0).toNat_lt
    omega
  have hgi : ∀ j, j < inStride →
      (gmemInpAt ss.gmem ib inStride).getD j 0 = ss.gmem.getD (ib + j) 0 := by
    intro j hj
    simp only [gmemInpAt, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range, hj,
      Option.map_some, Option.getD_some]
  have hora : tableOracle ss.gmem ss.smem hashLog tbl ib (s + l.val)
      = if ((snsteps prog 21 ss).regs "candRaw" l).toNat = 0 then none
        else some (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1) := by
    rw [candRaw21_nat]; rfl
  ----------------------------------------------------------------
  have probe_none : ∀ (o : Nat → Option Nat), o (s + l.val) = none →
      probe (gmemInpAt ss.gmem ib inStride) o searchLim (s + l.val) = none := by
    intro o ho; simp only [probe, ho]
  have probe_some : ∀ (o : Nat → Option Nat) (c : Nat), o (s + l.val) = some c →
      probe (gmemInpAt ss.gmem ib inStride) o searchLim (s + l.val)
        = if (s + l.val) < searchLim ∧ c < (s + l.val)
            ∧ verify4 (gmemInpAt ss.gmem ib inStride) (s + l.val) c = true then some c else none := by
    intro o c ho; simp only [probe, ho]
  rw [e40, cw_pHit_comb prog (snsteps prog 34 ss) l hD0 hD1 hD2 hD3 hD4 hD5,
      pValid34, candRaw34, cand34, posP34, v32_34]
  by_cases hraw : ((snsteps prog 21 ss).regs "candRaw" l).toNat = 0
  · have hR0 : ((snsteps prog 21 ss).regs "candRaw" l) = 0 := by rw [← UInt64.toNat_inj]; simpa using hraw
    have hpn : probe (gmemInpAt ss.gmem ib inStride) (tableOracle ss.gmem ss.smem hashLog tbl ib)
        searchLim (s + l.val) = none :=
      probe_none _ (by rw [hora, if_pos hraw])
    constructor
    · rintro ⟨⟨-, hne, -⟩, -⟩; exact absurd hR0 hne
    · intro h; rw [hpn] at h; simp at h
  · have hRne : ((snsteps prog 21 ss).regs "candRaw" l) ≠ 0 := by intro hc; apply hraw; rw [hc]; rfl
    have h1le : 1 ≤ ((snsteps prog 21 ss).regs "candRaw" l).toNat := Nat.one_le_iff_ne_zero.mpr hraw
    have hRofNat : UInt64.ofNat ((snsteps prog 21 ss).regs "candRaw" l).toNat = ((snsteps prog 21 ss).regs "candRaw" l) := UInt64.ofNat_toNat
    have hsub : (((snsteps prog 21 ss).regs "candRaw" l) - UInt64.ofNat 1).toNat = ((snsteps prog 21 ss).regs "candRaw" l).toNat - 1 := by
      have := u64_sub_ofNat ((snsteps prog 21 ss).regs "candRaw" l).toNat 1 h1le raw_lt
      rwa [hRofNat] at this
    have hconj3 : (((snsteps prog 21 ss).regs "candRaw" l) - UInt64.ofNat 1
          < ss.regs "searchPos" l + ss.regs "lane" l)
        ↔ (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1 < s + l.val) := by
      rw [UInt64.lt_iff_toNat_lt, hsub, hpval]
    by_cases hv : s + l.val < searchLim
    · rw [if_pos hv]
      rw [hconj3]
      have hps := probe_some (tableOracle ss.gmem ss.smem hashLog tbl ib)
        (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1) (by rw [hora, if_neg hraw])
      rw [hps]
      have hconj4 : (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1 < s + l.val) →
          (((snsteps prog 34 ss).regs "vc" l = UInt64.ofNat (wLoad4 ss.gmem (ib + (s + l.val))))
            ↔ verify4 (gmemInpAt ss.gmem ib inStride) (s + l.val) (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1) = true) := by
        intro hlt
        have hle : ((snsteps prog 21 ss).regs "candRaw" l) - UInt64.ofNat 1 ≤ UInt64.ofNat capC := by
          rw [UInt64.le_iff_toNat_le, hsub, toNat_ofNat_lt capC hcapb]; omega
        have haddr : (SOp.run .add (UInt64.ofNat ib)
              (SOp.run .min (SOp.run .sub ((snsteps prog 21 ss).regs "candRaw" l) (UInt64.ofNat 1)) (UInt64.ofNat capC))).toNat = ib + (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1) := by
          simp only [SOp.run, if_pos hle]
          rw [UInt64.toNat_add, hsub, toNat_ofNat_lt ib (by omega), Nat.mod_eq_of_lt (by omega)]
        have hvcv : (snsteps prog 34 ss).regs "vc" l = UInt64.ofNat (wLoad4 ss.gmem (ib + (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1))) := by
          rw [← UInt64.toNat_inj, vc34_nat, haddr, toNat_ofNat_lt _ (hwl_any _)]
        have hverify : verify4 (gmemInpAt ss.gmem ib inStride) (s + l.val) (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1) = true
            ↔ wLoad4 ss.gmem (ib + (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1)) = wLoad4 ss.gmem (ib + (s + l.val)) := by
          rw [wLoad4_eq_iff]
          simp only [verify4, List.range_succ, List.range_zero, List.all_cons, List.all_nil,
            List.nil_append, List.cons_append, Bool.and_true, byte, Bool.and_eq_true, beq_iff_eq,
            Nat.add_zero]
          rw [hgi (s + l.val) (by omega), hgi (s + l.val + 1) (by omega),
              hgi (s + l.val + 2) (by omega), hgi (s + l.val + 3) (by omega),
              hgi (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1) (by omega), hgi (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1 + 1) (by omega),
              hgi (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1 + 2) (by omega), hgi (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1 + 3) (by omega),
              ← UInt8.toNat_inj, ← UInt8.toNat_inj, ← UInt8.toNat_inj, ← UInt8.toNat_inj]
          simp only [Nat.add_assoc]
          constructor <;> rintro ⟨e0, e1, e2, e3⟩ <;> refine ⟨?_, ?_, ?_, ?_⟩ <;> omega
        rw [hvcv, ← UInt64.toNat_inj, toNat_ofNat_lt _ (hwl_any _), toNat_ofNat_lt _ (hwl_any _)]
        exact hverify.symm
      by_cases hlt : ((snsteps prog 21 ss).regs "candRaw" l).toNat - 1 < s + l.val
      · rw [hconj4 hlt]
        constructor
        · rintro ⟨-, hv2⟩; rw [if_pos ⟨hv, hlt, hv2⟩]; rfl
        · intro h
          have hv2 : verify4 (gmemInpAt ss.gmem ib inStride) (s + l.val) (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1) = true := by
            by_cases hc : (s + l.val) < searchLim ∧ (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1) < (s + l.val)
                ∧ verify4 (gmemInpAt ss.gmem ib inStride) (s + l.val) (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1) = true
            · exact hc.2.2
            · rw [if_neg hc] at h; simp at h
          exact ⟨⟨rfl, hRne, hlt⟩, hv2⟩
      · constructor
        · rintro ⟨⟨-, -, h3⟩, -⟩; exact absurd h3 hlt
        · intro h; rw [if_neg (fun hc => hlt hc.2.1)] at h; simp at h
    · rw [if_neg hv]
      have hpn : probe (gmemInpAt ss.gmem ib inStride) (tableOracle ss.gmem ss.smem hashLog tbl ib)
          searchLim (s + l.val) = none := by
        have hps := probe_some (tableOracle ss.gmem ss.smem hashLog tbl ib)
          (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1) (by rw [hora, if_neg hraw])
        rw [hps, if_neg (fun hc => hv hc.1)]
      constructor
      · rintro ⟨⟨hv1, -, -⟩, -⟩
        exact absurd hv1 (by decide)
      · intro h; rw [hpn] at h; simp at h
-- ── Companion: the candidate value at state-34, in terms of `tableOracle` ────
-- (Steps 0..33 only — segments A/B/C, no segment D.  Needed to identify the
-- `cand` register (broadcast by `shfl` at the end) with the model's winning
-- candidate `c`, alongside `coopWindow_pHit_iff` above.)

-- ── Generic pc lemma for the 5-step hash prefix, and single-step `addr` frame ──
theorem segB5_pc (prog : Array SInstr) (st : SState) (hashLog : Nat)
    (h0 : prog[st.pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h1 : prog[st.pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h2 : prog[st.pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h3 : prog[st.pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h4 : prog[st.pc + 4]? = some (.binr .add "addr" "hh" "tbl")) :
    (snsteps prog 5 st).pc = st.pc + 5 := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, sstepInstr, SState.setReg, SState.setPc, SState.get,
    SOp.run]

theorem ldsh_addr_frame (prog : Array SInstr) (st : SState) (l : Fin 32)
    (h0 : prog[st.pc]? = some (.ldsh "candRaw" "addr")) :
    (snsteps prog 1 st).regs "addr" l = st.regs "addr" l := by
  simp [snsteps, sstep, h0, sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]

-- ── Generic 6-step frame for the predicate segment (34..39) ──────────────────
theorem segD_frame (prog : Array SInstr) (st : SState) (r : String) (l : Fin 32)
    (h0 : prog[st.pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h1 : prog[st.pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h2 : prog[st.pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h3 : prog[st.pc + 3]? = some (.andp "pH1" "pValid" "pNE"))
    (h4 : prog[st.pc + 4]? = some (.andp "pH2" "pH1" "pCO"))
    (h5 : prog[st.pc + 5]? = some (.andp "pHit" "pH2" "pEq"))
    (hr : r ∉ ["pNE", "pCO", "pEq", "pH1", "pH2", "pHit"]) :
    (snsteps prog 6 st).regs r l = st.regs r l := by
  simp only [List.mem_cons, List.mem_singleton, not_or, List.not_mem_nil, not_false_iff,
    and_true] at hr
  obtain ⟨n0, n1, n2, n3, n4, n5⟩ := hr
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, sstepInstr, SState.setReg, SState.setPc,
    SState.get, SOp.run, SCmp.run, n0, n1, n2, n3, n4, n5]

/-- **Companion to `coopWindow_pHit_iff`**: the candidate register value.  Exposes
    `cand@34 = candRaw@21 - 1` and the `tableOracle` characterisation of `candRaw@21`
    (extracted from `coopWindow_pHit_iff`'s internals) so the found-branch can couple
    `cand0` to the model's winning candidate `c`. -/
theorem coopWindow_cand34_val (prog : Array SInstr) (ss : SState) (base ib : Nat)
    (s inStride searchLim capC hashLog tbl : Nat) (l : Fin 32)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[base+1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[base+2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[base+3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[base+4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[base+5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[base+6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[base+7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[base+8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[base+9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[base+10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[base+11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[base+12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[base+13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[base+14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (h15 : prog[base+15]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h16 : prog[base+16]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h17 : prog[base+17]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h18 : prog[base+18]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h19 : prog[base+19]? = some (.binr .add "addr" "hh" "tbl"))
    (h20 : prog[base+20]? = some (.ldsh "candRaw" "addr"))
    (h21 : prog[base+21]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h22 : prog[base+22]? = some (.binr .min "rc" "cand" "cap4"))
    (h23 : prog[base+23]? = some (.binr .add "rcA" "inBase" "rc"))
    (h24 : prog[base+24]? = some (.ldgo "c0" "rcA" 0))
    (h25 : prog[base+25]? = some (.ldgo "c1" "rcA" 1))
    (h26 : prog[base+26]? = some (.ldgo "c2" "rcA" 2))
    (h27 : prog[base+27]? = some (.ldgo "c3" "rcA" 3))
    (h28 : prog[base+28]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h29 : prog[base+29]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h30 : prog[base+30]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h31 : prog[base+31]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h32 : prog[base+32]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h33 : prog[base+33]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (h34 : prog[base+34]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h35 : prog[base+35]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h36 : prog[base+36]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h37 : prog[base+37]? = some (.andp "pH1" "pValid" "pNE"))
    (h38 : prog[base+38]? = some (.andp "pH2" "pH1" "pCO"))
    (h39 : prog[base+39]? = some (.andp "pHit" "pH2" "pEq"))
    (hinb : ss.regs "inBase" l = UInt64.ofNat ib)
    (htbl : ss.regs "tbl" l = UInt64.ofNat tbl)
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40) (hcapv : s + l.val ≤ capC)
    (hlane : ss.regs "lane" l = UInt64.ofNat l.val) (hsp : ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + l.val < 2 ^ 64) (hib64 : ib + capC < 2 ^ 64) :
    (snsteps prog 40 ss).regs "cand" l = (snsteps prog 21 ss).regs "candRaw" l - UInt64.ofNat 1
    ∧ tableOracle ss.gmem ss.smem hashLog tbl ib (s + l.val)
        = (if ((snsteps prog 21 ss).regs "candRaw" l).toNat = 0 then none
           else some (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1))
    ∧ ((snsteps prog 21 ss).regs "candRaw" l).toNat < 2 ^ 64 := by
  subst hpc
  have e21 : snsteps prog 21 ss = snsteps prog 6 (snsteps prog 15 ss) := snsteps_add prog 15 6 ss
  have e34 : snsteps prog 34 ss = snsteps prog 13 (snsteps prog 21 ss) := snsteps_add prog 21 13 ss
  have hpc15 : (snsteps prog 15 ss).pc = ss.pc + 15 := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc]
  have hB0 : prog[(snsteps prog 15 ss).pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)) := by rw [hpc15]; exact h15
  have hB1 : prog[(snsteps prog 15 ss).pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))) := by rw [hpc15]; exact h16
  have hB2 : prog[(snsteps prog 15 ss).pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))) := by rw [hpc15]; exact h17
  have hB3 : prog[(snsteps prog 15 ss).pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)) := by rw [hpc15]; exact h18
  have hB4 : prog[(snsteps prog 15 ss).pc + 4]? = some (.binr .add "addr" "hh" "tbl") := by rw [hpc15]; exact h19
  have hB5 : prog[(snsteps prog 15 ss).pc + 5]? = some (.ldsh "candRaw" "addr") := by rw [hpc15]; exact h20
  have hpc21 : (snsteps prog 21 ss).pc = ss.pc + 21 := by
    rw [e21, segB_pc prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4 hB5, hpc15]
  have hC0 : prog[(snsteps prog 21 ss).pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)) := by rw [hpc21]; exact h21
  have hC1 : prog[(snsteps prog 21 ss).pc + 1]? = some (.binr .min "rc" "cand" "cap4") := by rw [hpc21]; exact h22
  have hC2 : prog[(snsteps prog 21 ss).pc + 2]? = some (.binr .add "rcA" "inBase" "rc") := by rw [hpc21]; exact h23
  have hC3 : prog[(snsteps prog 21 ss).pc + 3]? = some (.ldgo "c0" "rcA" 0) := by rw [hpc21]; exact h24
  have hC4 : prog[(snsteps prog 21 ss).pc + 4]? = some (.ldgo "c1" "rcA" 1) := by rw [hpc21]; exact h25
  have hC5 : prog[(snsteps prog 21 ss).pc + 5]? = some (.ldgo "c2" "rcA" 2) := by rw [hpc21]; exact h26
  have hC6 : prog[(snsteps prog 21 ss).pc + 6]? = some (.ldgo "c3" "rcA" 3) := by rw [hpc21]; exact h27
  have hC7 : prog[(snsteps prog 21 ss).pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)) := by rw [hpc21]; exact h28
  have hC8 : prog[(snsteps prog 21 ss).pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)) := by rw [hpc21]; exact h29
  have hC9 : prog[(snsteps prog 21 ss).pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)) := by rw [hpc21]; exact h30
  have hC10 : prog[(snsteps prog 21 ss).pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")) := by rw [hpc21]; exact h31
  have hC11 : prog[(snsteps prog 21 ss).pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")) := by rw [hpc21]; exact h32
  have hC12 : prog[(snsteps prog 21 ss).pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3")) := by rw [hpc21]; exact h33
  have hpval : (ss.regs "searchPos" l + ss.regs "lane" l).toNat = s + l.val := by
    rw [hsp, hlane]; exact u64_add_ofNat s l.val hp64
  have hxle : ss.regs "searchPos" l + ss.regs "lane" l ≤ UInt64.ofNat capC := by
    rw [UInt64.le_iff_toNat_le, hpval, toNat_ofNat_lt capC hcapb]; exact hcapv
  have hpsum : ss.regs "searchPos" l + ss.regs "lane" l = UInt64.ofNat (s + l.val) := by
    rw [hsp, hlane]; exact (UInt64.ofNat_add s l.val).symm
  have hidx : (SOp.run .add (ss.regs "inBase" l)
        (SOp.run .min (SOp.run .add (ss.regs "searchPos" l) (ss.regs "lane" l))
          (UInt64.ofNat capC))).toNat = ib + (s + l.val) := by
    show (ss.regs "inBase" l + (if ss.regs "searchPos" l + ss.regs "lane" l ≤ UInt64.ofNat capC
            then ss.regs "searchPos" l + ss.regs "lane" l else UInt64.ofNat capC)).toNat
      = ib + (s + l.val)
    rw [hinb, if_pos hxle, hpsum]; exact u64_add_ofNat ib (s + l.val) (by omega)
  have v32_15 : (snsteps prog 15 ss).regs "v32" l = UInt64.ofNat (wLoad4 ss.gmem (ib + (s + l.val))) := by
    have hc := cw_v32 prog ss searchLim capC h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 l
    rw [hidx] at hc
    rw [← UInt64.toNat_inj, hc, toNat_ofNat_lt _ (by
      unfold wLoad4
      have a0 := (ss.gmem.getD (ib + (s + l.val)) 0).toNat_lt
      have a1 := (ss.gmem.getD (ib + (s + l.val) + 1) 0).toNat_lt
      have a2 := (ss.gmem.getD (ib + (s + l.val) + 2) 0).toNat_lt
      have a3 := (ss.gmem.getD (ib + (s + l.val) + 3) 0).toNat_lt
      omega)]
  have tbl15 : (snsteps prog 15 ss).regs "tbl" l = UInt64.ofNat tbl := by
    have : (snsteps prog 15 ss).regs "tbl" l = ss.regs "tbl" l := by
      simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
        sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
    rw [this, htbl]
  have smem15 : (snsteps prog 15 ss).smem = ss.smem := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc]
  have candRaw21_nat : ((snsteps prog 21 ss).regs "candRaw" l).toNat
      = (ss.smem.getD (tbl + 2 * wHash ss.gmem hashLog (ib + (s + l.val))) 0).toNat
        + 256 * (ss.smem.getD (tbl + 2 * wHash ss.gmem hashLog (ib + (s + l.val)) + 1) 0).toNat := by
    rw [e21]
    have hc := cw_candRaw prog (snsteps prog 15 ss) l (wLoad4 ss.gmem (ib + (s + l.val))) hashLog tbl
      v32_15 tbl15 hB0 hB1 hB2 hB3 hB4 hB5 hhl htblb
    rw [smem15] at hc
    rw [hc]
    rfl
  have raw_lt : ((snsteps prog 21 ss).regs "candRaw" l).toNat < 2 ^ 64 := by
    rw [candRaw21_nat]
    have a0 := (ss.smem.getD (tbl + 2 * wHash ss.gmem hashLog (ib + (s + l.val))) 0).toNat_lt
    have a1 := (ss.smem.getD (tbl + 2 * wHash ss.gmem hashLog (ib + (s + l.val)) + 1) 0).toNat_lt
    omega
  have cand34 : (snsteps prog 34 ss).regs "cand" l
      = (snsteps prog 21 ss).regs "candRaw" l - UInt64.ofNat 1 := by
    rw [e34]
    exact segC_cand prog (snsteps prog 21 ss) l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12
  have e40 : snsteps prog 40 ss = snsteps prog 6 (snsteps prog 34 ss) := snsteps_add prog 34 6 ss
  have hpc34 : (snsteps prog 34 ss).pc = ss.pc + 34 := by
    rw [e34, segC_pc prog (snsteps prog 21 ss) hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12,
      hpc21]
  have hD0 : prog[(snsteps prog 34 ss).pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)) := by
    rw [hpc34]; exact h34
  have hD1 : prog[(snsteps prog 34 ss).pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")) := by
    rw [hpc34]; exact h35
  have hD2 : prog[(snsteps prog 34 ss).pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")) := by
    rw [hpc34]; exact h36
  have hD3 : prog[(snsteps prog 34 ss).pc + 3]? = some (.andp "pH1" "pValid" "pNE") := by
    rw [hpc34]; exact h37
  have hD4 : prog[(snsteps prog 34 ss).pc + 4]? = some (.andp "pH2" "pH1" "pCO") := by
    rw [hpc34]; exact h38
  have hD5 : prog[(snsteps prog 34 ss).pc + 5]? = some (.andp "pHit" "pH2" "pEq") := by
    rw [hpc34]; exact h39
  have cand40 : (snsteps prog 40 ss).regs "cand" l
      = (snsteps prog 21 ss).regs "candRaw" l - UInt64.ofNat 1 := by
    rw [e40, segD_frame prog (snsteps prog 34 ss) "cand" l hD0 hD1 hD2 hD3 hD4 hD5 (by decide), cand34]
  have hora : tableOracle ss.gmem ss.smem hashLog tbl ib (s + l.val)
      = (if ((snsteps prog 21 ss).regs "candRaw" l).toNat = 0 then none
         else some (((snsteps prog 21 ss).regs "candRaw" l).toNat - 1)) := by
    rw [candRaw21_nat]; rfl
  exact ⟨cand40, hora, raw_lt⟩

-- ── Companion: `posP`/`pValid` at step 40 (needed for the `stshp` insert). ────
theorem coopWindow_posPValid40 (prog : Array SInstr) (ss : SState) (base : Nat)
    (s inStride searchLim capC hashLog tbl : Nat) (l : Fin 32)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[base+1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[base+2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[base+3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[base+4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[base+5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[base+6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[base+7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[base+8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[base+9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[base+10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[base+11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[base+12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[base+13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[base+14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (h15 : prog[base+15]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h16 : prog[base+16]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h17 : prog[base+17]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h18 : prog[base+18]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h19 : prog[base+19]? = some (.binr .add "addr" "hh" "tbl"))
    (h20 : prog[base+20]? = some (.ldsh "candRaw" "addr"))
    (h21 : prog[base+21]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h22 : prog[base+22]? = some (.binr .min "rc" "cand" "cap4"))
    (h23 : prog[base+23]? = some (.binr .add "rcA" "inBase" "rc"))
    (h24 : prog[base+24]? = some (.ldgo "c0" "rcA" 0))
    (h25 : prog[base+25]? = some (.ldgo "c1" "rcA" 1))
    (h26 : prog[base+26]? = some (.ldgo "c2" "rcA" 2))
    (h27 : prog[base+27]? = some (.ldgo "c3" "rcA" 3))
    (h28 : prog[base+28]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h29 : prog[base+29]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h30 : prog[base+30]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h31 : prog[base+31]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h32 : prog[base+32]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h33 : prog[base+33]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (h34 : prog[base+34]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h35 : prog[base+35]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h36 : prog[base+36]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h37 : prog[base+37]? = some (.andp "pH1" "pValid" "pNE"))
    (h38 : prog[base+38]? = some (.andp "pH2" "pH1" "pCO"))
    (h39 : prog[base+39]? = some (.andp "pHit" "pH2" "pEq"))
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40)
    (hlane : ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + l.val < 2 ^ 64) :
    (snsteps prog 40 ss).regs "posP" l = ss.regs "searchPos" l + ss.regs "lane" l
    ∧ (snsteps prog 40 ss).regs "pValid" l
        = (if ss.regs "searchPos" l + ss.regs "lane" l < UInt64.ofNat searchLim then 1 else 0) := by
  subst hpc
  have posP15 : (snsteps prog 15 ss).regs "posP" l = ss.regs "searchPos" l + ss.regs "lane" l := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
  have pValid15 : (snsteps prog 15 ss).regs "pValid" l
      = (if ss.regs "searchPos" l + ss.regs "lane" l < UInt64.ofNat searchLim then 1 else 0) := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run]
  have hpc15 : (snsteps prog 15 ss).pc = ss.pc + 15 := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
  have hB0 : prog[(snsteps prog 15 ss).pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)) := by
    rw [hpc15]; exact h15
  have hB1 : prog[(snsteps prog 15 ss).pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))) := by
    rw [hpc15]; exact h16
  have hB2 : prog[(snsteps prog 15 ss).pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))) := by
    rw [hpc15]; exact h17
  have hB3 : prog[(snsteps prog 15 ss).pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)) := by
    rw [hpc15]; exact h18
  have hB4 : prog[(snsteps prog 15 ss).pc + 4]? = some (.binr .add "addr" "hh" "tbl") := by
    rw [hpc15]; exact h19
  have hB5 : prog[(snsteps prog 15 ss).pc + 5]? = some (.ldsh "candRaw" "addr") := by
    rw [hpc15]; exact h20
  have e21 : snsteps prog 21 ss = snsteps prog 6 (snsteps prog 15 ss) := snsteps_add prog 15 6 ss
  have hpc21 : (snsteps prog 21 ss).pc = ss.pc + 21 := by
    rw [e21]
    have := segB_pc prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4 hB5
    omega
  have hC0 : prog[(snsteps prog 21 ss).pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)) := by
    rw [hpc21]; exact h21
  have hC1 : prog[(snsteps prog 21 ss).pc + 1]? = some (.binr .min "rc" "cand" "cap4") := by
    rw [hpc21]; exact h22
  have hC2 : prog[(snsteps prog 21 ss).pc + 2]? = some (.binr .add "rcA" "inBase" "rc") := by
    rw [hpc21]; exact h23
  have hC3 : prog[(snsteps prog 21 ss).pc + 3]? = some (.ldgo "c0" "rcA" 0) := by
    rw [hpc21]; exact h24
  have hC4 : prog[(snsteps prog 21 ss).pc + 4]? = some (.ldgo "c1" "rcA" 1) := by
    rw [hpc21]; exact h25
  have hC5 : prog[(snsteps prog 21 ss).pc + 5]? = some (.ldgo "c2" "rcA" 2) := by
    rw [hpc21]; exact h26
  have hC6 : prog[(snsteps prog 21 ss).pc + 6]? = some (.ldgo "c3" "rcA" 3) := by
    rw [hpc21]; exact h27
  have hC7 : prog[(snsteps prog 21 ss).pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)) := by
    rw [hpc21]; exact h28
  have hC8 : prog[(snsteps prog 21 ss).pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)) := by
    rw [hpc21]; exact h29
  have hC9 : prog[(snsteps prog 21 ss).pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)) := by
    rw [hpc21]; exact h30
  have hC10 : prog[(snsteps prog 21 ss).pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")) := by
    rw [hpc21]; exact h31
  have hC11 : prog[(snsteps prog 21 ss).pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")) := by
    rw [hpc21]; exact h32
  have hC12 : prog[(snsteps prog 21 ss).pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3")) := by
    rw [hpc21]; exact h33
  have e34 : snsteps prog 34 ss = snsteps prog 13 (snsteps prog 21 ss) := snsteps_add prog 21 13 ss
  have hpc34 : (snsteps prog 34 ss).pc = ss.pc + 34 := by
    rw [e34]
    have := segC_pc prog (snsteps prog 21 ss) hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12
    omega
  have hD0 : prog[(snsteps prog 34 ss).pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)) := by
    rw [hpc34]; exact h34
  have hD1 : prog[(snsteps prog 34 ss).pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")) := by
    rw [hpc34]; exact h35
  have hD2 : prog[(snsteps prog 34 ss).pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")) := by
    rw [hpc34]; exact h36
  have hD3 : prog[(snsteps prog 34 ss).pc + 3]? = some (.andp "pH1" "pValid" "pNE") := by
    rw [hpc34]; exact h37
  have hD4 : prog[(snsteps prog 34 ss).pc + 4]? = some (.andp "pH2" "pH1" "pCO") := by
    rw [hpc34]; exact h38
  have hD5 : prog[(snsteps prog 34 ss).pc + 5]? = some (.andp "pHit" "pH2" "pEq") := by
    rw [hpc34]; exact h39
  have e40 : snsteps prog 40 ss = snsteps prog 6 (snsteps prog 34 ss) := snsteps_add prog 34 6 ss
  have posPB : (snsteps prog 21 ss).regs "posP" l = (snsteps prog 15 ss).regs "posP" l := by
    rw [e21]; exact segB_frame prog (snsteps prog 15 ss) "posP" l hashLog hB0 hB1 hB2 hB3 hB4 hB5 (by decide)
  have pValidB : (snsteps prog 21 ss).regs "pValid" l = (snsteps prog 15 ss).regs "pValid" l := by
    rw [e21]; exact segB_frame prog (snsteps prog 15 ss) "pValid" l hashLog hB0 hB1 hB2 hB3 hB4 hB5 (by decide)
  have posPC : (snsteps prog 34 ss).regs "posP" l = (snsteps prog 21 ss).regs "posP" l := by
    rw [e34]
    exact segC_frame prog (snsteps prog 21 ss) "posP" l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12 (by decide)
  have pValidC : (snsteps prog 34 ss).regs "pValid" l = (snsteps prog 21 ss).regs "pValid" l := by
    rw [e34]
    exact segC_frame prog (snsteps prog 21 ss) "pValid" l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12 (by decide)
  have posPD : (snsteps prog 40 ss).regs "posP" l = (snsteps prog 34 ss).regs "posP" l := by
    rw [e40]; exact segD_frame prog (snsteps prog 34 ss) "posP" l hD0 hD1 hD2 hD3 hD4 hD5 (by decide)
  have pValidD : (snsteps prog 40 ss).regs "pValid" l = (snsteps prog 34 ss).regs "pValid" l := by
    rw [e40]; exact segD_frame prog (snsteps prog 34 ss) "pValid" l hD0 hD1 hD2 hD3 hD4 hD5 (by decide)
  exact ⟨by rw [posPD, posPC, posPB, posP15], by rw [pValidD, pValidC, pValidB, pValid15]⟩

theorem coopWindow_hpc34 (prog : Array SInstr) (ss : SState) (base : Nat)
    (searchLim capC hashLog : Nat) (hpc : ss.pc = base)
    (P : ProbeInstrs prog base searchLim capC hashLog) :
    (snsteps prog 34 ss).pc = ss.pc + 34 := by
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19,
    h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35, h36, h37, h38,
    h39⟩ := P
  subst hpc
  have e21 : snsteps prog 21 ss = snsteps prog 6 (snsteps prog 15 ss) := snsteps_add prog 15 6 ss
  have e34 : snsteps prog 34 ss = snsteps prog 13 (snsteps prog 21 ss) := snsteps_add prog 21 13 ss
  have hpc15 : (snsteps prog 15 ss).pc = ss.pc + 15 := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc]
  have hB0 : prog[(snsteps prog 15 ss).pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)) := by
    rw [hpc15]; exact h15
  have hB1 : prog[(snsteps prog 15 ss).pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))) := by
    rw [hpc15]; exact h16
  have hB2 : prog[(snsteps prog 15 ss).pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))) := by
    rw [hpc15]; exact h17
  have hB3 : prog[(snsteps prog 15 ss).pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)) := by
    rw [hpc15]; exact h18
  have hB4 : prog[(snsteps prog 15 ss).pc + 4]? = some (.binr .add "addr" "hh" "tbl") := by
    rw [hpc15]; exact h19
  have hB5 : prog[(snsteps prog 15 ss).pc + 5]? = some (.ldsh "candRaw" "addr") := by
    rw [hpc15]; exact h20
  have hpc21 : (snsteps prog 21 ss).pc = ss.pc + 21 := by
    rw [e21, segB_pc prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4 hB5, hpc15]
  have hC0 : prog[(snsteps prog 21 ss).pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)) := by
    rw [hpc21]; exact h21
  have hC1 : prog[(snsteps prog 21 ss).pc + 1]? = some (.binr .min "rc" "cand" "cap4") := by
    rw [hpc21]; exact h22
  have hC2 : prog[(snsteps prog 21 ss).pc + 2]? = some (.binr .add "rcA" "inBase" "rc") := by
    rw [hpc21]; exact h23
  have hC3 : prog[(snsteps prog 21 ss).pc + 3]? = some (.ldgo "c0" "rcA" 0) := by rw [hpc21]; exact h24
  have hC4 : prog[(snsteps prog 21 ss).pc + 4]? = some (.ldgo "c1" "rcA" 1) := by rw [hpc21]; exact h25
  have hC5 : prog[(snsteps prog 21 ss).pc + 5]? = some (.ldgo "c2" "rcA" 2) := by rw [hpc21]; exact h26
  have hC6 : prog[(snsteps prog 21 ss).pc + 6]? = some (.ldgo "c3" "rcA" 3) := by rw [hpc21]; exact h27
  have hC7 : prog[(snsteps prog 21 ss).pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)) := by
    rw [hpc21]; exact h28
  have hC8 : prog[(snsteps prog 21 ss).pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)) := by
    rw [hpc21]; exact h29
  have hC9 : prog[(snsteps prog 21 ss).pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)) := by
    rw [hpc21]; exact h30
  have hC10 : prog[(snsteps prog 21 ss).pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")) := by
    rw [hpc21]; exact h31
  have hC11 : prog[(snsteps prog 21 ss).pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")) := by
    rw [hpc21]; exact h32
  have hC12 : prog[(snsteps prog 21 ss).pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3")) := by
    rw [hpc21]; exact h33
  rw [e34, segC_pc prog (snsteps prog 21 ss) hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12, hpc21]

-- ── Companion: `smem` is untouched across the 40-instr probe. ────────────────
-- ── `smem`-frame chunks for the probe prefix (no `.stshp`/`.stsh` in steps 0–45). ──
theorem segA_smem (prog : Array SInstr) (st : SState) (searchLim capC : Nat)
    (h0 : prog[st.pc]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[st.pc + 1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[st.pc + 2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[st.pc + 3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[st.pc + 4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[st.pc + 5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[st.pc + 6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[st.pc + 7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[st.pc + 8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[st.pc + 9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[st.pc + 10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[st.pc + 11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[st.pc + 12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[st.pc + 13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[st.pc + 14]? = some (.bin .bor "v32" "v32" (.reg "b3"))) :
    (snsteps prog 15 st).smem = st.smem := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
    sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]

theorem segB_smem (prog : Array SInstr) (st : SState) (hashLog : Nat)
    (h0 : prog[st.pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h1 : prog[st.pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h2 : prog[st.pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h3 : prog[st.pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h4 : prog[st.pc + 4]? = some (.binr .add "addr" "hh" "tbl"))
    (h5 : prog[st.pc + 5]? = some (.ldsh "candRaw" "addr")) :
    (snsteps prog 6 st).smem = st.smem := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, sstepInstr, SState.setReg, SState.setPc,
    SState.get, SOp.run]

theorem segC_smem (prog : Array SInstr) (st : SState)
    (h0 : prog[st.pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h1 : prog[st.pc + 1]? = some (.binr .min "rc" "cand" "cap4"))
    (h2 : prog[st.pc + 2]? = some (.binr .add "rcA" "inBase" "rc"))
    (h3 : prog[st.pc + 3]? = some (.ldgo "c0" "rcA" 0))
    (h4 : prog[st.pc + 4]? = some (.ldgo "c1" "rcA" 1))
    (h5 : prog[st.pc + 5]? = some (.ldgo "c2" "rcA" 2))
    (h6 : prog[st.pc + 6]? = some (.ldgo "c3" "rcA" 3))
    (h7 : prog[st.pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h8 : prog[st.pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h9 : prog[st.pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h10 : prog[st.pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h11 : prog[st.pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h12 : prog[st.pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3"))) :
    (snsteps prog 13 st).smem = st.smem := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12,
    sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]

theorem segD_smem (prog : Array SInstr) (st : SState)
    (h0 : prog[st.pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h1 : prog[st.pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h2 : prog[st.pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h3 : prog[st.pc + 3]? = some (.andp "pH1" "pValid" "pNE"))
    (h4 : prog[st.pc + 4]? = some (.andp "pH2" "pH1" "pCO"))
    (h5 : prog[st.pc + 5]? = some (.andp "pHit" "pH2" "pEq")) :
    (snsteps prog 6 st).smem = st.smem := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, sstepInstr, SState.setReg, SState.setPc,
    SState.get, SOp.run, SCmp.run]

-- Steps 40..45: vote/brev/clz/setp/andp/add — the pre-`stshp` tail, `smem` untouched.
theorem tailPre_smem (prog : Array SInstr) (st : SState)
    (h0 : prog[st.pc]? = some (.vote "bal" "pHit"))
    (h1 : prog[st.pc + 1]? = some (.brev "rev" "bal"))
    (h2 : prog[st.pc + 2]? = some (.clz "fl" "rev"))
    (h3 : prog[st.pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h4 : prog[st.pc + 4]? = some (.andp "pIns" "pLe" "pValid"))
    (h5 : prog[st.pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1))) :
    (snsteps prog 6 st).smem = st.smem := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, sstepInstr, SState.setReg, SState.setPc,
    SState.get, SOp.run, SCmp.run]

theorem coopWindow_smem40 (prog : Array SInstr) (ss : SState) (base : Nat)
    (searchLim capC hashLog : Nat)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[base+1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[base+2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[base+3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[base+4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[base+5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[base+6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[base+7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[base+8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[base+9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[base+10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[base+11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[base+12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[base+13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[base+14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (h15 : prog[base+15]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h16 : prog[base+16]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h17 : prog[base+17]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h18 : prog[base+18]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h19 : prog[base+19]? = some (.binr .add "addr" "hh" "tbl"))
    (h20 : prog[base+20]? = some (.ldsh "candRaw" "addr"))
    (h21 : prog[base+21]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h22 : prog[base+22]? = some (.binr .min "rc" "cand" "cap4"))
    (h23 : prog[base+23]? = some (.binr .add "rcA" "inBase" "rc"))
    (h24 : prog[base+24]? = some (.ldgo "c0" "rcA" 0))
    (h25 : prog[base+25]? = some (.ldgo "c1" "rcA" 1))
    (h26 : prog[base+26]? = some (.ldgo "c2" "rcA" 2))
    (h27 : prog[base+27]? = some (.ldgo "c3" "rcA" 3))
    (h28 : prog[base+28]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h29 : prog[base+29]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h30 : prog[base+30]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h31 : prog[base+31]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h32 : prog[base+32]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h33 : prog[base+33]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (h34 : prog[base+34]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h35 : prog[base+35]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h36 : prog[base+36]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h37 : prog[base+37]? = some (.andp "pH1" "pValid" "pNE"))
    (h38 : prog[base+38]? = some (.andp "pH2" "pH1" "pCO"))
    (h39 : prog[base+39]? = some (.andp "pHit" "pH2" "pEq")) :
    (snsteps prog 40 ss).smem = ss.smem := by
  subst hpc
  have eB : snsteps prog 21 ss = snsteps prog 6 (snsteps prog 15 ss) := snsteps_add prog 15 6 ss
  have eC : snsteps prog 34 ss = snsteps prog 13 (snsteps prog 21 ss) := snsteps_add prog 21 13 ss
  have eD : snsteps prog 40 ss = snsteps prog 6 (snsteps prog 34 ss) := snsteps_add prog 34 6 ss
  have hpc15 : (snsteps prog 15 ss).pc = ss.pc + 15 := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
  have hB0 : prog[(snsteps prog 15 ss).pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)) := by
    rw [hpc15]; exact h15
  have hB1 : prog[(snsteps prog 15 ss).pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))) := by
    rw [hpc15]; exact h16
  have hB2 : prog[(snsteps prog 15 ss).pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))) := by
    rw [hpc15]; exact h17
  have hB3 : prog[(snsteps prog 15 ss).pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)) := by
    rw [hpc15]; exact h18
  have hB4 : prog[(snsteps prog 15 ss).pc + 4]? = some (.binr .add "addr" "hh" "tbl") := by
    rw [hpc15]; exact h19
  have hB5 : prog[(snsteps prog 15 ss).pc + 5]? = some (.ldsh "candRaw" "addr") := by
    rw [hpc15]; exact h20
  have hpc21 : (snsteps prog 21 ss).pc = ss.pc + 21 := by
    rw [eB, segB_pc prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4 hB5, hpc15]
  have hC0 : prog[(snsteps prog 21 ss).pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)) := by
    rw [hpc21]; exact h21
  have hC1 : prog[(snsteps prog 21 ss).pc + 1]? = some (.binr .min "rc" "cand" "cap4") := by
    rw [hpc21]; exact h22
  have hC2 : prog[(snsteps prog 21 ss).pc + 2]? = some (.binr .add "rcA" "inBase" "rc") := by
    rw [hpc21]; exact h23
  have hC3 : prog[(snsteps prog 21 ss).pc + 3]? = some (.ldgo "c0" "rcA" 0) := by
    rw [hpc21]; exact h24
  have hC4 : prog[(snsteps prog 21 ss).pc + 4]? = some (.ldgo "c1" "rcA" 1) := by
    rw [hpc21]; exact h25
  have hC5 : prog[(snsteps prog 21 ss).pc + 5]? = some (.ldgo "c2" "rcA" 2) := by
    rw [hpc21]; exact h26
  have hC6 : prog[(snsteps prog 21 ss).pc + 6]? = some (.ldgo "c3" "rcA" 3) := by
    rw [hpc21]; exact h27
  have hC7 : prog[(snsteps prog 21 ss).pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)) := by
    rw [hpc21]; exact h28
  have hC8 : prog[(snsteps prog 21 ss).pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)) := by
    rw [hpc21]; exact h29
  have hC9 : prog[(snsteps prog 21 ss).pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)) := by
    rw [hpc21]; exact h30
  have hC10 : prog[(snsteps prog 21 ss).pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")) := by
    rw [hpc21]; exact h31
  have hC11 : prog[(snsteps prog 21 ss).pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")) := by
    rw [hpc21]; exact h32
  have hC12 : prog[(snsteps prog 21 ss).pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3")) := by
    rw [hpc21]; exact h33
  have hpc34 : (snsteps prog 34 ss).pc = ss.pc + 34 := by
    rw [eC, segC_pc prog (snsteps prog 21 ss) hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12,
      hpc21]
  have hD0 : prog[(snsteps prog 34 ss).pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)) := by
    rw [hpc34]; exact h34
  have hD1 : prog[(snsteps prog 34 ss).pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")) := by
    rw [hpc34]; exact h35
  have hD2 : prog[(snsteps prog 34 ss).pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")) := by
    rw [hpc34]; exact h36
  have hD3 : prog[(snsteps prog 34 ss).pc + 3]? = some (.andp "pH1" "pValid" "pNE") := by
    rw [hpc34]; exact h37
  have hD4 : prog[(snsteps prog 34 ss).pc + 4]? = some (.andp "pH2" "pH1" "pCO") := by
    rw [hpc34]; exact h38
  have hD5 : prog[(snsteps prog 34 ss).pc + 5]? = some (.andp "pHit" "pH2" "pEq") := by
    rw [hpc34]; exact h39
  have sA : (snsteps prog 15 ss).smem = ss.smem :=
    segA_smem prog ss searchLim capC h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14
  have sB : (snsteps prog 21 ss).smem = (snsteps prog 15 ss).smem := by
    rw [eB]; exact segB_smem prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4 hB5
  have sC : (snsteps prog 34 ss).smem = (snsteps prog 21 ss).smem := by
    rw [eC]; exact segC_smem prog (snsteps prog 21 ss) hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9
      hC10 hC11 hC12
  have sD : (snsteps prog 40 ss).smem = (snsteps prog 34 ss).smem := by
    rw [eD]; exact segD_smem prog (snsteps prog 34 ss) hD0 hD1 hD2 hD3 hD4 hD5
  rw [sD, sC, sB, sA]

-- ── Companion: `lane` is never written across the 40-instr probe. ────────────
theorem coopWindow_lane40 (prog : Array SInstr) (ss : SState) (base : Nat)
    (s inStride searchLim capC hashLog tbl : Nat) (l : Fin 32)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[base+1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[base+2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[base+3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[base+4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[base+5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[base+6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[base+7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[base+8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[base+9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[base+10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[base+11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[base+12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[base+13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[base+14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (h15 : prog[base+15]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h16 : prog[base+16]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h17 : prog[base+17]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h18 : prog[base+18]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h19 : prog[base+19]? = some (.binr .add "addr" "hh" "tbl"))
    (h20 : prog[base+20]? = some (.ldsh "candRaw" "addr"))
    (h21 : prog[base+21]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h22 : prog[base+22]? = some (.binr .min "rc" "cand" "cap4"))
    (h23 : prog[base+23]? = some (.binr .add "rcA" "inBase" "rc"))
    (h24 : prog[base+24]? = some (.ldgo "c0" "rcA" 0))
    (h25 : prog[base+25]? = some (.ldgo "c1" "rcA" 1))
    (h26 : prog[base+26]? = some (.ldgo "c2" "rcA" 2))
    (h27 : prog[base+27]? = some (.ldgo "c3" "rcA" 3))
    (h28 : prog[base+28]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h29 : prog[base+29]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h30 : prog[base+30]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h31 : prog[base+31]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h32 : prog[base+32]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h33 : prog[base+33]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (h34 : prog[base+34]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h35 : prog[base+35]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h36 : prog[base+36]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h37 : prog[base+37]? = some (.andp "pH1" "pValid" "pNE"))
    (h38 : prog[base+38]? = some (.andp "pH2" "pH1" "pCO"))
    (h39 : prog[base+39]? = some (.andp "pHit" "pH2" "pEq")) :
    (snsteps prog 40 ss).regs "lane" l = ss.regs "lane" l := by
  subst hpc
  have e21 : snsteps prog 21 ss = snsteps prog 6 (snsteps prog 15 ss) := snsteps_add prog 15 6 ss
  have e34 : snsteps prog 34 ss = snsteps prog 13 (snsteps prog 21 ss) := snsteps_add prog 21 13 ss
  have e40 : snsteps prog 40 ss = snsteps prog 6 (snsteps prog 34 ss) := snsteps_add prog 34 6 ss
  have hlane15 : (snsteps prog 15 ss).regs "lane" l = ss.regs "lane" l := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
  have hpc15 : (snsteps prog 15 ss).pc = ss.pc + 15 := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
  have hB0 : prog[(snsteps prog 15 ss).pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)) := by
    rw [hpc15]; exact h15
  have hB1 : prog[(snsteps prog 15 ss).pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))) := by
    rw [hpc15]; exact h16
  have hB2 : prog[(snsteps prog 15 ss).pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))) := by
    rw [hpc15]; exact h17
  have hB3 : prog[(snsteps prog 15 ss).pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)) := by
    rw [hpc15]; exact h18
  have hB4 : prog[(snsteps prog 15 ss).pc + 4]? = some (.binr .add "addr" "hh" "tbl") := by
    rw [hpc15]; exact h19
  have hB5 : prog[(snsteps prog 15 ss).pc + 5]? = some (.ldsh "candRaw" "addr") := by
    rw [hpc15]; exact h20
  have hpc21 : (snsteps prog 21 ss).pc = ss.pc + 21 := by
    rw [e21, segB_pc prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4 hB5, hpc15]
  have hC0 : prog[(snsteps prog 21 ss).pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)) := by
    rw [hpc21]; exact h21
  have hC1 : prog[(snsteps prog 21 ss).pc + 1]? = some (.binr .min "rc" "cand" "cap4") := by
    rw [hpc21]; exact h22
  have hC2 : prog[(snsteps prog 21 ss).pc + 2]? = some (.binr .add "rcA" "inBase" "rc") := by
    rw [hpc21]; exact h23
  have hC3 : prog[(snsteps prog 21 ss).pc + 3]? = some (.ldgo "c0" "rcA" 0) := by
    rw [hpc21]; exact h24
  have hC4 : prog[(snsteps prog 21 ss).pc + 4]? = some (.ldgo "c1" "rcA" 1) := by
    rw [hpc21]; exact h25
  have hC5 : prog[(snsteps prog 21 ss).pc + 5]? = some (.ldgo "c2" "rcA" 2) := by
    rw [hpc21]; exact h26
  have hC6 : prog[(snsteps prog 21 ss).pc + 6]? = some (.ldgo "c3" "rcA" 3) := by
    rw [hpc21]; exact h27
  have hC7 : prog[(snsteps prog 21 ss).pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)) := by
    rw [hpc21]; exact h28
  have hC8 : prog[(snsteps prog 21 ss).pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)) := by
    rw [hpc21]; exact h29
  have hC9 : prog[(snsteps prog 21 ss).pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)) := by
    rw [hpc21]; exact h30
  have hC10 : prog[(snsteps prog 21 ss).pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")) := by
    rw [hpc21]; exact h31
  have hC11 : prog[(snsteps prog 21 ss).pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")) := by
    rw [hpc21]; exact h32
  have hC12 : prog[(snsteps prog 21 ss).pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3")) := by
    rw [hpc21]; exact h33
  have hpc34 : (snsteps prog 34 ss).pc = ss.pc + 34 := by
    rw [e34, segC_pc prog (snsteps prog 21 ss) hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12,
      hpc21]
  have hD0 : prog[(snsteps prog 34 ss).pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)) := by
    rw [hpc34]; exact h34
  have hD1 : prog[(snsteps prog 34 ss).pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")) := by
    rw [hpc34]; exact h35
  have hD2 : prog[(snsteps prog 34 ss).pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")) := by
    rw [hpc34]; exact h36
  have hD3 : prog[(snsteps prog 34 ss).pc + 3]? = some (.andp "pH1" "pValid" "pNE") := by
    rw [hpc34]; exact h37
  have hD4 : prog[(snsteps prog 34 ss).pc + 4]? = some (.andp "pH2" "pH1" "pCO") := by
    rw [hpc34]; exact h38
  have hD5 : prog[(snsteps prog 34 ss).pc + 5]? = some (.andp "pHit" "pH2" "pEq") := by
    rw [hpc34]; exact h39
  have hlaneB : (snsteps prog 21 ss).regs "lane" l = (snsteps prog 15 ss).regs "lane" l := by
    rw [e21]; exact segB_frame prog (snsteps prog 15 ss) "lane" l hashLog hB0 hB1 hB2 hB3 hB4 hB5
      (by decide)
  have hlaneC : (snsteps prog 34 ss).regs "lane" l = (snsteps prog 21 ss).regs "lane" l := by
    rw [e34]
    exact segC_frame prog (snsteps prog 21 ss) "lane" l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9
      hC10 hC11 hC12 (by decide)
  have hlaneD : (snsteps prog 40 ss).regs "lane" l = (snsteps prog 34 ss).regs "lane" l := by
    rw [e40]
    exact segD_frame prog (snsteps prog 34 ss) "lane" l hD0 hD1 hD2 hD3 hD4 hD5 (by decide)
  rw [hlaneD, hlaneC, hlaneB, hlane15]

-- ── Companion: `addr` at step 40, in terms of `wHash`. ────────────────────────
theorem coopWindow_addr40 (prog : Array SInstr) (ss : SState) (base ib : Nat)
    (s inStride searchLim capC hashLog tbl : Nat) (l : Fin 32)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[base+1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[base+2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[base+3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[base+4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[base+5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[base+6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[base+7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[base+8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[base+9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[base+10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[base+11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[base+12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[base+13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[base+14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (h15 : prog[base+15]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h16 : prog[base+16]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h17 : prog[base+17]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h18 : prog[base+18]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h19 : prog[base+19]? = some (.binr .add "addr" "hh" "tbl"))
    (h20 : prog[base+20]? = some (.ldsh "candRaw" "addr"))
    (h21 : prog[base+21]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h22 : prog[base+22]? = some (.binr .min "rc" "cand" "cap4"))
    (h23 : prog[base+23]? = some (.binr .add "rcA" "inBase" "rc"))
    (h24 : prog[base+24]? = some (.ldgo "c0" "rcA" 0))
    (h25 : prog[base+25]? = some (.ldgo "c1" "rcA" 1))
    (h26 : prog[base+26]? = some (.ldgo "c2" "rcA" 2))
    (h27 : prog[base+27]? = some (.ldgo "c3" "rcA" 3))
    (h28 : prog[base+28]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h29 : prog[base+29]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h30 : prog[base+30]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h31 : prog[base+31]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h32 : prog[base+32]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h33 : prog[base+33]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (h34 : prog[base+34]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h35 : prog[base+35]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h36 : prog[base+36]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h37 : prog[base+37]? = some (.andp "pH1" "pValid" "pNE"))
    (h38 : prog[base+38]? = some (.andp "pH2" "pH1" "pCO"))
    (h39 : prog[base+39]? = some (.andp "pHit" "pH2" "pEq"))
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40)
    (hinb : ss.regs "inBase" l = UInt64.ofNat ib) (htbl : ss.regs "tbl" l = UInt64.ofNat tbl)
    (hcapv : s + l.val ≤ capC)
    (hlane : ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + l.val < 2 ^ 64) (hib64 : ib + capC < 2 ^ 64) :
    (snsteps prog 40 ss).regs "addr" l = UInt64.ofNat (tbl + 2 * wHash ss.gmem hashLog (ib + (s + l.val))) := by
  subst hpc
  have hpval : (ss.regs "searchPos" l + ss.regs "lane" l).toNat = s + l.val := by
    rw [hsp, hlane]; exact u64_add_ofNat s l.val hp64
  have hxle : ss.regs "searchPos" l + ss.regs "lane" l ≤ UInt64.ofNat capC := by
    rw [UInt64.le_iff_toNat_le, hpval, toNat_ofNat_lt capC hcapb]; exact hcapv
  have hpsum : ss.regs "searchPos" l + ss.regs "lane" l = UInt64.ofNat (s + l.val) := by
    rw [hsp, hlane]; exact (UInt64.ofNat_add s l.val).symm
  have hidx : (SOp.run .add (ss.regs "inBase" l)
        (SOp.run .min (SOp.run .add (ss.regs "searchPos" l) (ss.regs "lane" l))
          (UInt64.ofNat capC))).toNat = ib + (s + l.val) := by
    show (ss.regs "inBase" l + (if ss.regs "searchPos" l + ss.regs "lane" l ≤ UInt64.ofNat capC
            then ss.regs "searchPos" l + ss.regs "lane" l else UInt64.ofNat capC)).toNat
      = ib + (s + l.val)
    rw [hinb, if_pos hxle, hpsum]; exact u64_add_ofNat ib (s + l.val) (by omega)
  have hwl_lt : wLoad4 ss.gmem (ib + (s + l.val)) < 2 ^ 32 := by
    unfold wLoad4
    have a0 := (ss.gmem.getD (ib + (s + l.val)) 0).toNat_lt
    have a1 := (ss.gmem.getD (ib + (s + l.val) + 1) 0).toNat_lt
    have a2 := (ss.gmem.getD (ib + (s + l.val) + 2) 0).toNat_lt
    have a3 := (ss.gmem.getD (ib + (s + l.val) + 3) 0).toNat_lt
    omega
  have v32_15 : (snsteps prog 15 ss).regs "v32" l = UInt64.ofNat (wLoad4 ss.gmem (ib + (s + l.val))) := by
    have hc := cw_v32 prog ss searchLim capC h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 l
    rw [hidx] at hc
    rw [← UInt64.toNat_inj, hc, toNat_ofNat_lt _ (by omega)]
  have htbl15 : (snsteps prog 15 ss).regs "tbl" l = UInt64.ofNat tbl := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, htbl]
  have hpc15 : (snsteps prog 15 ss).pc = ss.pc + 15 := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
  have hB0 : prog[(snsteps prog 15 ss).pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)) := by
    rw [hpc15]; exact h15
  have hB1 : prog[(snsteps prog 15 ss).pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))) := by
    rw [hpc15]; exact h16
  have hB2 : prog[(snsteps prog 15 ss).pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))) := by
    rw [hpc15]; exact h17
  have hB3 : prog[(snsteps prog 15 ss).pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)) := by
    rw [hpc15]; exact h18
  have hB4 : prog[(snsteps prog 15 ss).pc + 4]? = some (.binr .add "addr" "hh" "tbl") := by
    rw [hpc15]; exact h19
  have hB5 : prog[(snsteps prog 15 ss).pc + 5]? = some (.ldsh "candRaw" "addr") := by
    rw [hpc15]; exact h20
  have haddr20 : ((snsteps prog 20 ss).regs "addr" l).toNat
      = tbl + 2 * (((UInt64.ofNat (wLoad4 ss.gmem (ib + (s + l.val))) * UInt64.ofNat wHashK)
          >>> UInt64.ofNat (32 - hashLog)).toNat % 2 ^ hashLog) := by
    have := cw_addr5 prog (snsteps prog 15 ss) l (wLoad4 ss.gmem (ib + (s + l.val))) hashLog tbl
      v32_15 htbl15 hB0 hB1 hB2 hB3 hB4 hhl htblb
    rwa [show (20:Nat) = 15 + 5 from rfl, snsteps_add]
  have haddr20' : (snsteps prog 20 ss).regs "addr" l
      = UInt64.ofNat (tbl + 2 * wHash ss.gmem hashLog (ib + (s + l.val))) := by
    apply UInt64.toNat_inj.mp
    have hhlpow : (2:Nat) ^ hashLog ≤ 4294967296 := by
      calc (2:Nat) ^ hashLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hhl
        _ = 4294967296 := by decide
    have hbound : wHash ss.gmem hashLog (ib + (s + l.val)) < 4294967296 := by
      unfold wHash; omega
    rw [haddr20, toNat_ofNat_lt _ (by omega)]
    unfold wHash
    rfl
  have e21 : snsteps prog 21 ss = snsteps prog 6 (snsteps prog 15 ss) := snsteps_add prog 15 6 ss
  have hpc21 : (snsteps prog 21 ss).pc = ss.pc + 21 := by
    rw [e21]
    have := segB_pc prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4 hB5
    omega
  have hC0 : prog[(snsteps prog 21 ss).pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)) := by
    rw [hpc21]; exact h21
  have hC1 : prog[(snsteps prog 21 ss).pc + 1]? = some (.binr .min "rc" "cand" "cap4") := by
    rw [hpc21]; exact h22
  have hC2 : prog[(snsteps prog 21 ss).pc + 2]? = some (.binr .add "rcA" "inBase" "rc") := by
    rw [hpc21]; exact h23
  have hC3 : prog[(snsteps prog 21 ss).pc + 3]? = some (.ldgo "c0" "rcA" 0) := by
    rw [hpc21]; exact h24
  have hC4 : prog[(snsteps prog 21 ss).pc + 4]? = some (.ldgo "c1" "rcA" 1) := by
    rw [hpc21]; exact h25
  have hC5 : prog[(snsteps prog 21 ss).pc + 5]? = some (.ldgo "c2" "rcA" 2) := by
    rw [hpc21]; exact h26
  have hC6 : prog[(snsteps prog 21 ss).pc + 6]? = some (.ldgo "c3" "rcA" 3) := by
    rw [hpc21]; exact h27
  have hC7 : prog[(snsteps prog 21 ss).pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)) := by
    rw [hpc21]; exact h28
  have hC8 : prog[(snsteps prog 21 ss).pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)) := by
    rw [hpc21]; exact h29
  have hC9 : prog[(snsteps prog 21 ss).pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)) := by
    rw [hpc21]; exact h30
  have hC10 : prog[(snsteps prog 21 ss).pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")) := by
    rw [hpc21]; exact h31
  have hC11 : prog[(snsteps prog 21 ss).pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")) := by
    rw [hpc21]; exact h32
  have hC12 : prog[(snsteps prog 21 ss).pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3")) := by
    rw [hpc21]; exact h33
  have e34 : snsteps prog 34 ss = snsteps prog 13 (snsteps prog 21 ss) := snsteps_add prog 21 13 ss
  have hpc34 : (snsteps prog 34 ss).pc = ss.pc + 34 := by
    rw [e34]
    have := segC_pc prog (snsteps prog 21 ss) hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12
    omega
  have hD0 : prog[(snsteps prog 34 ss).pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)) := by
    rw [hpc34]; exact h34
  have hD1 : prog[(snsteps prog 34 ss).pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")) := by
    rw [hpc34]; exact h35
  have hD2 : prog[(snsteps prog 34 ss).pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")) := by
    rw [hpc34]; exact h36
  have hD3 : prog[(snsteps prog 34 ss).pc + 3]? = some (.andp "pH1" "pValid" "pNE") := by
    rw [hpc34]; exact h37
  have hD4 : prog[(snsteps prog 34 ss).pc + 4]? = some (.andp "pH2" "pH1" "pCO") := by
    rw [hpc34]; exact h38
  have hD5 : prog[(snsteps prog 34 ss).pc + 5]? = some (.andp "pHit" "pH2" "pEq") := by
    rw [hpc34]; exact h39
  have e40 : snsteps prog 40 ss = snsteps prog 6 (snsteps prog 34 ss) := snsteps_add prog 34 6 ss
  have haddrC : (snsteps prog 34 ss).regs "addr" l = (snsteps prog 21 ss).regs "addr" l := by
    rw [e34]
    exact segC_frame prog (snsteps prog 21 ss) "addr" l hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9
      hC10 hC11 hC12 (by decide)
  have haddrD : (snsteps prog 40 ss).regs "addr" l = (snsteps prog 34 ss).regs "addr" l := by
    rw [e40]; exact segD_frame prog (snsteps prog 34 ss) "addr" l hD0 hD1 hD2 hD3 hD4 hD5 (by decide)
  have e20 : snsteps prog 20 ss = snsteps prog 5 (snsteps prog 15 ss) := snsteps_add prog 15 5 ss
  have hpc20 : (snsteps prog 20 ss).pc = ss.pc + 20 := by
    rw [e20, segB5_pc prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4, hpc15]
  have h20' : prog[(snsteps prog 20 ss).pc]? = some (.ldsh "candRaw" "addr") := by
    rw [hpc20]; exact h20
  have e21' : snsteps prog 21 ss = snsteps prog 1 (snsteps prog 20 ss) := snsteps_add prog 20 1 ss
  have haddrB21 : (snsteps prog 21 ss).regs "addr" l = (snsteps prog 20 ss).regs "addr" l := by
    rw [e21']; exact ldsh_addr_frame prog (snsteps prog 20 ss) l h20'
  rw [haddrD, haddrC, haddrB21, haddr20']

theorem coopWindow_pc40 (prog : Array SInstr) (ss : SState) (base : Nat)
    (searchLim capC hashLog : Nat)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[base+1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[base+2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[base+3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[base+4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[base+5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[base+6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[base+7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[base+8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[base+9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[base+10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[base+11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[base+12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[base+13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[base+14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (h15 : prog[base+15]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h16 : prog[base+16]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h17 : prog[base+17]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h18 : prog[base+18]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h19 : prog[base+19]? = some (.binr .add "addr" "hh" "tbl"))
    (h20 : prog[base+20]? = some (.ldsh "candRaw" "addr"))
    (h21 : prog[base+21]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h22 : prog[base+22]? = some (.binr .min "rc" "cand" "cap4"))
    (h23 : prog[base+23]? = some (.binr .add "rcA" "inBase" "rc"))
    (h24 : prog[base+24]? = some (.ldgo "c0" "rcA" 0))
    (h25 : prog[base+25]? = some (.ldgo "c1" "rcA" 1))
    (h26 : prog[base+26]? = some (.ldgo "c2" "rcA" 2))
    (h27 : prog[base+27]? = some (.ldgo "c3" "rcA" 3))
    (h28 : prog[base+28]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h29 : prog[base+29]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h30 : prog[base+30]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h31 : prog[base+31]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h32 : prog[base+32]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h33 : prog[base+33]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (h34 : prog[base+34]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h35 : prog[base+35]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h36 : prog[base+36]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h37 : prog[base+37]? = some (.andp "pH1" "pValid" "pNE"))
    (h38 : prog[base+38]? = some (.andp "pH2" "pH1" "pCO"))
    (h39 : prog[base+39]? = some (.andp "pHit" "pH2" "pEq")) :
    (snsteps prog 40 ss).pc = base + 40 := by
  subst hpc
  have e21 : snsteps prog 21 ss = snsteps prog 6 (snsteps prog 15 ss) := snsteps_add prog 15 6 ss
  have e34 : snsteps prog 34 ss = snsteps prog 13 (snsteps prog 21 ss) := snsteps_add prog 21 13 ss
  have e40 : snsteps prog 40 ss = snsteps prog 6 (snsteps prog 34 ss) := snsteps_add prog 34 6 ss
  have hpc15 : (snsteps prog 15 ss).pc = ss.pc + 15 := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc]
  have hB0 : prog[(snsteps prog 15 ss).pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)) := by rw [hpc15]; exact h15
  have hB1 : prog[(snsteps prog 15 ss).pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))) := by rw [hpc15]; exact h16
  have hB2 : prog[(snsteps prog 15 ss).pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))) := by rw [hpc15]; exact h17
  have hB3 : prog[(snsteps prog 15 ss).pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)) := by rw [hpc15]; exact h18
  have hB4 : prog[(snsteps prog 15 ss).pc + 4]? = some (.binr .add "addr" "hh" "tbl") := by rw [hpc15]; exact h19
  have hB5 : prog[(snsteps prog 15 ss).pc + 5]? = some (.ldsh "candRaw" "addr") := by rw [hpc15]; exact h20
  have hpc21 : (snsteps prog 21 ss).pc = ss.pc + 21 := by
    rw [e21, segB_pc prog (snsteps prog 15 ss) hashLog hB0 hB1 hB2 hB3 hB4 hB5, hpc15]
  have hC0 : prog[(snsteps prog 21 ss).pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)) := by rw [hpc21]; exact h21
  have hC1 : prog[(snsteps prog 21 ss).pc + 1]? = some (.binr .min "rc" "cand" "cap4") := by rw [hpc21]; exact h22
  have hC2 : prog[(snsteps prog 21 ss).pc + 2]? = some (.binr .add "rcA" "inBase" "rc") := by rw [hpc21]; exact h23
  have hC3 : prog[(snsteps prog 21 ss).pc + 3]? = some (.ldgo "c0" "rcA" 0) := by rw [hpc21]; exact h24
  have hC4 : prog[(snsteps prog 21 ss).pc + 4]? = some (.ldgo "c1" "rcA" 1) := by rw [hpc21]; exact h25
  have hC5 : prog[(snsteps prog 21 ss).pc + 5]? = some (.ldgo "c2" "rcA" 2) := by rw [hpc21]; exact h26
  have hC6 : prog[(snsteps prog 21 ss).pc + 6]? = some (.ldgo "c3" "rcA" 3) := by rw [hpc21]; exact h27
  have hC7 : prog[(snsteps prog 21 ss).pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)) := by rw [hpc21]; exact h28
  have hC8 : prog[(snsteps prog 21 ss).pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)) := by rw [hpc21]; exact h29
  have hC9 : prog[(snsteps prog 21 ss).pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)) := by rw [hpc21]; exact h30
  have hC10 : prog[(snsteps prog 21 ss).pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")) := by rw [hpc21]; exact h31
  have hC11 : prog[(snsteps prog 21 ss).pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")) := by rw [hpc21]; exact h32
  have hC12 : prog[(snsteps prog 21 ss).pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3")) := by rw [hpc21]; exact h33
  have hpc34 : (snsteps prog 34 ss).pc = ss.pc + 34 := by
    rw [e34, segC_pc prog (snsteps prog 21 ss) hC0 hC1 hC2 hC3 hC4 hC5 hC6 hC7 hC8 hC9 hC10 hC11 hC12, hpc21]
  have hD0 : prog[(snsteps prog 34 ss).pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)) := by rw [hpc34]; exact h34
  have hD1 : prog[(snsteps prog 34 ss).pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")) := by rw [hpc34]; exact h35
  have hD2 : prog[(snsteps prog 34 ss).pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")) := by rw [hpc34]; exact h36
  have hD3 : prog[(snsteps prog 34 ss).pc + 3]? = some (.andp "pH1" "pValid" "pNE") := by rw [hpc34]; exact h37
  have hD4 : prog[(snsteps prog 34 ss).pc + 4]? = some (.andp "pH2" "pH1" "pCO") := by rw [hpc34]; exact h38
  have hD5 : prog[(snsteps prog 34 ss).pc + 5]? = some (.andp "pHit" "pH2" "pEq") := by rw [hpc34]; exact h39
  rw [e40]
  generalize hX : snsteps prog 34 ss = X at hD0 hD1 hD2 hD3 hD4 hD5 hpc34 ⊢
  have hX6 : (snsteps prog 6 X).pc = X.pc + 6 := by
    simp [snsteps, sstep, hD0, hD1, hD2, hD3, hD4, hD5, sstepInstr, SState.setReg, SState.setPc]
  rw [hX6, hpc34]

/-- The closing 11-instr reduction/broadcast segment advances the pc by 11. -/
theorem cwPc11 (prog : Array SInstr) (st : SState)
    (h0 : prog[st.pc]? = some (.vote "bal" "pHit"))
    (h1 : prog[st.pc + 1]? = some (.brev "rev" "bal"))
    (h2 : prog[st.pc + 2]? = some (.clz "fl" "rev"))
    (h3 : prog[st.pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h4 : prog[st.pc + 4]? = some (.andp "pIns" "pLe" "pValid"))
    (h5 : prog[st.pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h6 : prog[st.pc + 6]? = some (.stshp "pIns" "addr" "pp1"))
    (h7 : prog[st.pc + 7]? = some (.barwarp))
    (h8 : prog[st.pc + 8]? = some (.binr .add "p0" "searchPos" "fl"))
    (h9 : prog[st.pc + 9]? = some (.shfl "cand0" "cand" "fl"))
    (h10 : prog[st.pc + 10]? = some (.setp .ne "found" "bal" (.imm 0))) :
    (snsteps prog 11 st).pc = st.pc + 11 := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10,
    sstepInstr, SState.setReg, SState.setPc]

-- ── Reassociation infrastructure: `stshp` two-pass store = model `tableInsert` ──
namespace Reassoc

theorem getD_set! (m : Array UInt8) (a j : Nat) (v : UInt8) :
    (m.set! a v).getD j 0 = if a = j ∧ a < m.size then v else m.getD j 0 := by
  rw [Array.set!_eq_setIfInBounds]
  by_cases haj : a = j
  · subst haj
    by_cases hb : a < m.size
    · rw [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds_self_of_lt hb]; simp [hb]
    · rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?]
      congr 1; rw [Array.getElem?_setIfInBounds]; simp [hb]
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds_ne haj,
        ← Array.getD_eq_getD_getElem?]; simp [haj]

theorem size_set! (m : Array UInt8) (a : Nat) (v : UInt8) : (m.set! a v).size = m.size := by
  rw [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]

theorem array_eq_of_getD {a b : Array UInt8} (hs : a.size = b.size)
    (h : ∀ j, a.getD j 0 = b.getD j 0) : a = b := by
  apply Array.ext hs
  intro i hi hi2
  have hj := h i
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_getElem hi, Array.getElem?_eq_getElem hi2] at hj
  simpa using hj

/-- `set!` commutes across distinct addresses — via `getD` characterization. -/
theorem set!_comm (m : Array UInt8) (a b : Nat) (u v : UInt8) (hab : a ≠ b) :
    (m.set! a u).set! b v = (m.set! b v).set! a u := by
  apply array_eq_of_getD
  · rw [size_set!, size_set!, size_set!, size_set!]
  · intro j
    rw [getD_set!, getD_set!, getD_set!, getD_set!, size_set!, size_set!]
    by_cases hbj : b = j <;> by_cases haj : a = j <;> simp_all

/-- The `tableInsert` interleaved foldl over `ls`, parameterized. -/
def tiFold (g : Array UInt8) (hashLog tbl searchLim ib sp upto : Nat) :
    List Nat → Array UInt8 → Array UInt8
  | [], acc => acc
  | k :: ks, acc =>
    tiFold g hashLog tbl searchLim ib sp upto ks
      (if k ≤ upto ∧ sp + k < searchLim then
        let a := tbl + 2 * wHash g hashLog (ib + (sp + k))
        let v := sp + k + 1
        (acc.setIfInBounds a (UInt8.ofNat (v % 256))).setIfInBounds (a + 1) (UInt8.ofNat (v / 256 % 256))
      else acc)

theorem tableInsert_eq_tiFold (sm g : Array UInt8) (hashLog tbl searchLim ib sp upto : Nat) :
    tableInsert sm g hashLog tbl searchLim ib sp upto
      = tiFold g hashLog tbl searchLim ib sp upto (List.range 32) sm := by
  unfold tableInsert
  suffices h : ∀ (ls : List Nat) (acc : Array UInt8),
      List.foldl (fun acc k =>
        if k ≤ upto ∧ sp + k < searchLim then
          let a := tbl + 2 * wHash g hashLog (ib + (sp + k))
          let v := sp + k + 1
          (acc.setIfInBounds a (UInt8.ofNat (v % 256))).setIfInBounds (a + 1) (UInt8.ofNat (v / 256 % 256))
        else acc) acc ls
      = tiFold g hashLog tbl searchLim ib sp upto ls acc by exact h _ _
  intro ls
  induction ls with
  | nil => intro acc; rfl
  | cons k ks ih => intro acc; simp only [List.foldl_cons, tiFold]; rw [ih]

/-- Low-write foldl over the k-list. -/
def loFold (g : Array UInt8) (hashLog tbl searchLim ib sp upto : Nat) :
    List Nat → Array UInt8 → Array UInt8
  | [], acc => acc
  | k :: ks, acc =>
    loFold g hashLog tbl searchLim ib sp upto ks
      (if k ≤ upto ∧ sp + k < searchLim then
        acc.set! (tbl + 2 * wHash g hashLog (ib + (sp + k))) (UInt8.ofNat ((sp+k+1) % 256))
      else acc)

/-- High-write foldl over the k-list. -/
def hiFold (g : Array UInt8) (hashLog tbl searchLim ib sp upto : Nat) :
    List Nat → Array UInt8 → Array UInt8
  | [], acc => acc
  | k :: ks, acc =>
    hiFold g hashLog tbl searchLim ib sp upto ks
      (if k ≤ upto ∧ sp + k < searchLim then
        acc.set! (tbl + 2 * wHash g hashLog (ib + (sp + k)) + 1) (UInt8.ofNat ((sp+k+1) / 256 % 256))
      else acc)

/-- A generic set!-at-`a` commutes left past a `loFold` when `a` is odd-parity
    (all `loFold` writes are even-parity `tbl + 2*·`), given `tbl` even.  We use the
    generic `set!_comm_foldl` after recasting `loFold` as a predicated single-write
    foldl over `List.range 32`. -/
theorem loFold_as_foldl (g : Array UInt8) (hashLog tbl searchLim ib sp upto : Nat) :
    ∀ (ls : List Nat) (acc : Array UInt8),
    loFold g hashLog tbl searchLim ib sp upto ls acc
      = List.foldl (fun m k => if k ≤ upto ∧ sp + k < searchLim then
          m.set! (tbl + 2 * wHash g hashLog (ib + (sp + k))) (UInt8.ofNat ((sp+k+1) % 256)) else m) acc ls := by
  intro ls; induction ls with
  | nil => intro acc; rfl
  | cons k ks ih => intro acc; simp only [loFold, List.foldl_cons]; rw [ih]

theorem hiFold_as_foldl (g : Array UInt8) (hashLog tbl searchLim ib sp upto : Nat) :
    ∀ (ls : List Nat) (acc : Array UInt8),
    hiFold g hashLog tbl searchLim ib sp upto ls acc
      = List.foldl (fun m k => if k ≤ upto ∧ sp + k < searchLim then
          m.set! (tbl + 2 * wHash g hashLog (ib + (sp + k)) + 1) (UInt8.ofNat ((sp+k+1) / 256 % 256)) else m) acc ls := by
  intro ls; induction ls with
  | nil => intro acc; rfl
  | cons k ks ih => intro acc; simp only [hiFold, List.foldl_cons]; rw [ih]

/-- Nat-indexed version of `set!_comm_foldl`. -/
theorem set!_comm_foldlN (a : Nat) (u : UInt8) (wa : Nat → Nat) (wv : Nat → UInt8) (wp : Nat → Prop)
    [DecidablePred wp] :
    ∀ (ls : List Nat) (m : Array UInt8),
    (∀ k ∈ ls, wp k → wa k ≠ a) →
    (List.foldl (fun m k => if wp k then m.set! (wa k) (wv k) else m) (m.set! a u) ls)
      = (List.foldl (fun m k => if wp k then m.set! (wa k) (wv k) else m) m ls).set! a u := by
  intro ls; induction ls with
  | nil => intro m _; rfl
  | cons x xs ih =>
    intro m hne
    simp only [List.foldl_cons]
    by_cases hp : wp x
    · rw [if_pos hp, if_pos hp, set!_comm m a (wa x) u (wv x) (Ne.symm (hne x (by simp) hp))]
      exact ih (m.set! (wa x) (wv x)) (fun l hl => hne l (by simp [hl]))
    · rw [if_neg hp, if_neg hp]; exact ih m (fun l hl => hne l (by simp [hl]))

/-- **Channel split**: the interleaved `tiFold` equals the low pass followed by the
    high pass, provided `tbl` is even (so low addresses `tbl+2h` are even, high
    `tbl+2h+1` odd — the two channels' addresses never coincide). -/
theorem tiFold_split (g : Array UInt8) (hashLog tbl searchLim ib sp upto : Nat) :
    ∀ (ls : List Nat) (acc : Array UInt8),
    tiFold g hashLog tbl searchLim ib sp upto ls acc
      = hiFold g hashLog tbl searchLim ib sp upto ls
          (loFold g hashLog tbl searchLim ib sp upto ls acc) := by
  intro ls; induction ls with
  | nil => intro acc; rfl
  | cons k ks ih =>
    intro acc
    simp only [tiFold, loFold, hiFold]
    by_cases hc : k ≤ upto ∧ sp + k < searchLim
    · rw [if_pos hc, if_pos hc, if_pos hc]
      rw [ih]
      -- goal: hiFold ks (loFold ks ((acc.setIfInBounds lo _).setIfInBounds (lo+1) _))
      --     = hiFold ks ((loFold ks (acc.set! lo _)).set! (lo+1) _)
      rw [← Array.set!_eq_setIfInBounds, ← Array.set!_eq_setIfInBounds]
      congr 1
      -- (loFold ks (acc.set! lo _).set! (lo+1) _) : bubble the (lo+1) set! out through loFold
      rw [loFold_as_foldl, loFold_as_foldl]
      rw [set!_comm_foldlN (tbl + 2 * wHash g hashLog (ib + (sp+k)) + 1) _
            (fun k' => tbl + 2 * wHash g hashLog (ib + (sp+k'))) _
            (fun k' => k' ≤ upto ∧ sp + k' < searchLim)]
      intro k' _ _
      omega
    · rw [if_neg hc, if_neg hc, if_neg hc, ih]

/-- `storeBytes` (a `finRange 32` predicated foldl) equals the corresponding
    `Nat`-foldl over `List.range 32`, matching lane `l ↔ l.val`. -/
theorem storeBytes_eq_natFold (mem : Array UInt8) (pred : Fin 32 → Bool) (addr val : Fin 32 → UInt64)
    (wp : Nat → Prop) [DecidablePred wp] (wa : Nat → Nat) (wv : Nat → UInt8)
    (hp : ∀ l : Fin 32, (pred l = true) = wp l.val)
    (ha : ∀ l : Fin 32, pred l = true → (addr l).toNat = wa l.val)
    (hv : ∀ l : Fin 32, pred l = true → (val l).toUInt8 = wv l.val) :
    storeBytes mem pred addr val
      = List.foldl (fun m k => if wp k then m.set! (wa k) (wv k) else m) mem (List.range 32) := by
  unfold storeBytes
  rw [show (List.finRange 32) = (List.range 32).map (fun k => (⟨k % 32, Nat.mod_lt _ (by decide)⟩ : Fin 32))
        from by decide]
  rw [List.foldl_map]
  -- both fold over `List.range 32`; the step functions agree on every element
  suffices h : ∀ (ks : List Nat) (m : Array UInt8), (∀ k ∈ ks, k < 32) →
      List.foldl (fun m k => (fun m l => if pred l = true then m.set! (addr l).toNat (val l).toUInt8 else m) m
          ⟨k % 32, Nat.mod_lt _ (by decide)⟩) m ks
        = List.foldl (fun m k => if wp k then m.set! (wa k) (wv k) else m) m ks by
    exact h (List.range 32) mem (fun k hk => List.mem_range.mp hk)
  intro ks
  induction ks with
  | nil => intro m _; rfl
  | cons k rest ih =>
    intro m hk32all
    have hk32 : k < 32 := hk32all k (by simp)
    have hkeq : (⟨k % 32, Nat.mod_lt _ (by decide)⟩ : Fin 32).val = k := by simp [Nat.mod_eq_of_lt hk32]
    simp only [List.foldl_cons]
    have hwpk : wp k = (pred ⟨k % 32, Nat.mod_lt _ (by decide)⟩ = true) := by
      have := hp ⟨k % 32, Nat.mod_lt _ (by decide)⟩; rw [hkeq] at this; rw [this]
    have hstep : (if pred (⟨k % 32, Nat.mod_lt _ (by decide)⟩ : Fin 32) = true
          then m.set! (addr ⟨k % 32, Nat.mod_lt _ (by decide)⟩).toNat
            (val ⟨k % 32, Nat.mod_lt _ (by decide)⟩).toUInt8 else m)
        = (if wp k then m.set! (wa k) (wv k) else m) := by
      by_cases hpk : pred ⟨k % 32, Nat.mod_lt _ (by decide)⟩ = true
      · rw [if_pos hpk, ha _ hpk, hv _ hpk, hkeq, if_pos (hwpk ▸ hpk)]
      · rw [if_neg hpk, if_neg (fun h => hpk (hwpk ▸ h))]
    rw [hstep]
    exact ih _ (fun k' hk' => hk32all k' (by simp [hk']))

theorem loByte (v : Nat) : (UInt64.ofNat v).toUInt8 = UInt8.ofNat (v % 256) := by
  apply UInt8.toNat_inj.mp
  show (UInt64.ofNat v).toNat % 256 = (UInt8.ofNat (v % 256)).toNat
  rw [UInt64.toNat_ofNat', UInt8.toNat_ofNat']; omega

theorem hiByte (v : Nat) (hv : v < 2^64) : (UInt64.ofNat v >>> 8).toUInt8 = UInt8.ofNat (v / 256 % 256) := by
  apply UInt8.toNat_inj.mp
  show ((UInt64.ofNat v >>> 8).toNat) % 256 = (UInt8.ofNat (v / 256 % 256)).toNat
  rw [UInt64.toNat_shiftRight, UInt64.toNat_ofNat', UInt8.toNat_ofNat',
      show ((8:UInt64).toNat % 64) = 8 from by decide, Nat.shiftRight_eq_div_pow,
      Nat.mod_eq_of_lt hv, show (2^8:Nat) = 256 from by decide]; omega

/-- **The `stshp` cooperative store computes the model `tableInsert`.**  Given the
    hypotheses fixing `pIns`/`addr`/`pp1` per lane, the machine's two-pass 2-byte
    store equals `tableInsert`. The proof: two `storeBytes` passes = `loFold` then
    `hiFold` (each via `storeBytes_eq_natFold`), which by channel-split (`tiFold_split`)
    equals `tiFold` = `tableInsert`.  Requires `s + l + 1 < 2^64` for every lane. -/
theorem stshp_tableInsert (st : SState) (g : Array UInt8) (hashLog tbl searchLim ib s upto : Nat)
    (hbound : s + 32 < 2 ^ 64) (htbl : tbl < 2 ^ 40) (hhl : hashLog ≤ 32)
    (hpIns : ∀ l : Fin 32, st.regs "pIns" l = if l.val ≤ upto ∧ s + l.val < searchLim then 1 else 0)
    (haddr : ∀ l : Fin 32, (l.val ≤ upto ∧ s + l.val < searchLim) →
      st.regs "addr" l = UInt64.ofNat (tbl + 2 * wHash g hashLog (ib + (s + l.val))))
    (hpp1 : ∀ l : Fin 32, st.regs "pp1" l = UInt64.ofNat (s + l.val + 1)) :
    (let s0 := storeBytes st.smem (fun l => st.regs "pIns" l == 1) (st.regs "addr") (st.regs "pp1")
     storeBytes s0 (fun l => st.regs "pIns" l == 1)
       (fun l => st.regs "addr" l + 1) (fun l => st.regs "pp1" l >>> 8))
      = tableInsert st.smem g hashLog tbl searchLim ib s upto := by
  simp only
  -- pass 1 (low) = loFold
  have hlo : storeBytes st.smem (fun l => st.regs "pIns" l == 1) (st.regs "addr") (st.regs "pp1")
      = loFold g hashLog tbl searchLim ib s upto (List.range 32) st.smem := by
    rw [storeBytes_eq_natFold st.smem _ _ _
        (fun k => k ≤ upto ∧ s + k < searchLim)
        (fun k => tbl + 2 * wHash g hashLog (ib + (s + k)))
        (fun k => UInt8.ofNat ((s + k + 1) % 256))]
    · rw [loFold_as_foldl]
    · intro l; rw [hpIns]; by_cases hh : l.val ≤ upto ∧ s + l.val < searchLim <;> simp [hh]
    · intro l hg
      have hguard : l.val ≤ upto ∧ s + l.val < searchLim := by
        rw [hpIns] at hg
        by_cases hh : l.val ≤ upto ∧ s + l.val < searchLim
        · exact hh
        · rw [if_neg hh] at hg; exact absurd hg (by decide)
      rw [haddr l hguard, UInt64.toNat_ofNat', Nat.mod_eq_of_lt]
      have hwb : wHash g hashLog (ib + (s + l.val)) < 2 ^ 32 := by
        unfold wHash; exact Nat.lt_of_lt_of_le (Nat.mod_lt _ (Nat.two_pow_pos _))
          (Nat.pow_le_pow_right (by decide) (by omega))
      omega
    · intro l _; rw [hpp1]; exact loByte _
  -- pass 2 (high) = hiFold, applied to the low result
  have hhi : ∀ acc : Array UInt8,
      storeBytes acc (fun l => st.regs "pIns" l == 1)
        (fun l => st.regs "addr" l + 1) (fun l => st.regs "pp1" l >>> 8)
      = hiFold g hashLog tbl searchLim ib s upto (List.range 32) acc := by
    intro acc
    rw [storeBytes_eq_natFold acc _ _ _
        (fun k => k ≤ upto ∧ s + k < searchLim)
        (fun k => tbl + 2 * wHash g hashLog (ib + (s + k)) + 1)
        (fun k => UInt8.ofNat ((s + k + 1) / 256 % 256))]
    · rw [hiFold_as_foldl]
    · intro l; rw [hpIns]; by_cases hh : l.val ≤ upto ∧ s + l.val < searchLim <;> simp [hh]
    · intro l hg
      have hguard : l.val ≤ upto ∧ s + l.val < searchLim := by
        rw [hpIns] at hg
        by_cases hh : l.val ≤ upto ∧ s + l.val < searchLim
        · exact hh
        · rw [if_neg hh] at hg; exact absurd hg (by decide)
      rw [haddr l hguard]
      have hwb : wHash g hashLog (ib + (s + l.val)) < 2 ^ 32 := by
        unfold wHash; exact Nat.lt_of_lt_of_le (Nat.mod_lt _ (Nat.two_pow_pos _))
          (Nat.pow_le_pow_right (by decide) (by omega))
      rw [show UInt64.ofNat (tbl + 2 * wHash g hashLog (ib + (s + l.val))) + 1
            = UInt64.ofNat (tbl + 2 * wHash g hashLog (ib + (s + l.val)) + 1) from by
          apply UInt64.toNat_inj.mp
          rw [UInt64.toNat_add, UInt64.toNat_ofNat', UInt64.toNat_ofNat',
              show (1:UInt64).toNat = 1 from rfl, Nat.mod_eq_of_lt (by omega),
              Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]]
      rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt]; omega
    · intro l _; rw [hpp1]; exact hiByte _ (by omega)
  rw [hlo, hhi, ← tiFold_split, tableInsert_eq_tiFold]

end Reassoc

-- ── The `stshp` cooperative table-insert = model `tableInsert` ───────────────
-- The single `stshp` instruction performs its 2-byte little-endian store as
-- *two* full cooperative passes (all lanes' low bytes, then all lanes' high
-- bytes — see `sstepInstr`'s `.stshp` case), whereas the model `tableInsert`
-- performs, for each `k` in increasing order, both bytes of one lane's entry
-- together.  The two orders agree because the u16 entries are 2-byte-aligned
-- (low addresses even-offset from `tbl`, high odd-offset), so the low and high
-- channels never interfere; the reassociation is proven in full above (namespace
-- `Reassoc`: `tiFold_split` splits the interleaved fold into a low pass then a
-- high pass, matched to the two `storeBytes` passes via `storeBytes_eq_natFold`).
-- ── Generic 6-step reduction of the tail's first half (vote..pp1), landing ───
-- exactly at the `stshp` instruction — gives `pIns`/`pp1`/`addr`(preserved)/
-- `gmem`(preserved)/`pc` all at once, from a fresh (unconcretized) state `st`.
theorem tailPre_vals (prog : Array SInstr) (st : SState) (l : Fin 32)
    (h0 : prog[st.pc]? = some (.vote "bal" "pHit"))
    (h1 : prog[st.pc + 1]? = some (.brev "rev" "bal"))
    (h2 : prog[st.pc + 2]? = some (.clz "fl" "rev"))
    (h3 : prog[st.pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h4 : prog[st.pc + 4]? = some (.andp "pIns" "pLe" "pValid"))
    (h5 : prog[st.pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1))) :
    (snsteps prog 6 st).regs "pIns" l
        = (if st.regs "lane" l ≤ clz32 (brev32 (ballotOf st.regs "pHit"))
              ∧ st.regs "pValid" l = 1 then 1 else 0)
    ∧ (snsteps prog 6 st).regs "pp1" l = st.regs "posP" l + 1
    ∧ (snsteps prog 6 st).regs "addr" l = st.regs "addr" l
    ∧ (snsteps prog 6 st).gmem = st.gmem
    ∧ (snsteps prog 6 st).pc = st.pc + 6 := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, sstepInstr, SState.setReg, SState.setPc,
      SState.get, SOp.run, SCmp.run]

-- ── Generic 4-step frame for the tail's second half (barwarp..found setp), ───
-- showing `smem` is unchanged (none of these instructions touch shared memory).
theorem tailPost_smem (prog : Array SInstr) (st : SState)
    (h0 : prog[st.pc]? = some .barwarp)
    (h1 : prog[st.pc + 1]? = some (.binr .add "p0" "searchPos" "fl"))
    (h2 : prog[st.pc + 2]? = some (.shfl "cand0" "cand" "fl"))
    (h3 : prog[st.pc + 3]? = some (.setp .ne "found" "bal" (.imm 0))) :
    (snsteps prog 4 st).smem = st.smem := by
  simp [snsteps, sstep, h0, h1, h2, h3, sstepInstr, SState.setReg, SState.setPc, SState.get,
    SOp.run, SCmp.run]

theorem coopWindow_stshp_tableInsert (prog : Array SInstr) (st : SState) (g : Array UInt8)
    (hashLog tbl searchLim ib s upto : Nat)
    (hbound : s + 32 < 2 ^ 64) (htbl : tbl < 2 ^ 40) (hhl : hashLog ≤ 32)
    (hpIns : ∀ l : Fin 32, st.regs "pIns" l = if l.val ≤ upto ∧ s + l.val < searchLim then 1 else 0)
    (haddr : ∀ l : Fin 32, (l.val ≤ upto ∧ s + l.val < searchLim) →
      st.regs "addr" l = UInt64.ofNat (tbl + 2 * wHash g hashLog (ib + (s + l.val))))
    (hpp1 : ∀ l : Fin 32, st.regs "pp1" l = UInt64.ofNat (s + l.val + 1))
    (h0 : prog[st.pc]? = some (.stshp "pIns" "addr" "pp1")) :
    (sstep prog st).smem = tableInsert st.smem g hashLog tbl searchLim ib s upto := by
  rw [show sstep prog st = sstepInstr prog (.stshp "pIns" "addr" "pp1") st from by
    simp only [sstep, h0]]
  show (storeBytes (storeBytes st.smem (fun l => st.regs "pIns" l == 1) (st.regs "addr") (st.regs "pp1"))
      (fun l => st.regs "pIns" l == 1) (fun l => st.regs "addr" l + 1)
      (fun l => st.regs "pp1" l >>> 8)) = _
  exact Reassoc.stshp_tableInsert st g hashLog tbl searchLim ib s upto hbound htbl hhl hpIns haddr hpp1

end AlgorithmLib.LZ4WarpDSL
