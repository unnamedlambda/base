import AlgorithmLib.LZ4WarpEmit
import AlgorithmLib.LZ4SimtBits
import AlgorithmLib.LZ4WarpSched
set_option maxRecDepth 4096

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib AlgorithmLib.LZ4Simt AlgorithmLib.LZ4SimtBits AlgorithmLib.LZ4WarpFind

-- ── Memory ↔ model bridge: gmem reads in the input region are model bytes ──────

theorem vote_step (prog : Array SInstr) (ss : SState) (d p : String)
    (h : prog[ss.pc]? = some (.vote d p)) :
    sstep prog ss = (ss.setReg d (fun _ => ballotOf ss.regs p)).setPc (ss.pc + 1) := by
  simp only [sstep, h, sstepInstr]

theorem mov_step (prog : Array SInstr) (ss : SState) (d : String) (a : SArg)
    (h : prog[ss.pc]? = some (.mov d a)) :
    sstep prog ss = (ss.setReg d (fun l => ss.get l a)).setPc (ss.pc + 1) := by
  simp only [sstep, h, sstepInstr]

theorem brev_step (prog : Array SInstr) (ss : SState) (d a : String)
    (h : prog[ss.pc]? = some (.brev d a)) :
    sstep prog ss = (ss.setReg d (fun l => brev32 (ss.regs a l))).setPc (ss.pc + 1) := by
  simp only [sstep, h, sstepInstr]

theorem clz_step (prog : Array SInstr) (ss : SState) (d a : String)
    (h : prog[ss.pc]? = some (.clz d a)) :
    sstep prog ss = (ss.setReg d (fun l => clz32 (ss.regs a l))).setPc (ss.pc + 1) := by
  simp only [sstep, h, sstepInstr]

theorem bin_step (prog : Array SInstr) (ss : SState) (o : SOp) (d a : String) (b : SArg)
    (h : prog[ss.pc]? = some (.bin o d a b)) :
    sstep prog ss = (ss.setReg d (fun l => o.run (ss.regs a l) (ss.get l b))).setPc (ss.pc + 1) := by
  simp only [sstep, h, sstepInstr]

theorem binr_step (prog : Array SInstr) (ss : SState) (o : SOp) (d a b : String)
    (h : prog[ss.pc]? = some (.binr o d a b)) :
    sstep prog ss = (ss.setReg d (fun l => o.run (ss.regs a l) (ss.regs b l))).setPc (ss.pc + 1) := by
  simp only [sstep, h, sstepInstr]

theorem stsh_step (prog : Array SInstr) (ss : SState) (addr s : String)
    (h : prog[ss.pc]? = some (.stsh addr s)) :
    sstep prog ss =
      { ss with
        smem :=
          let s0 := storeBytes ss.smem (fun _ => true) (ss.regs addr) (ss.regs s)
          storeBytes s0 (fun _ => true) (fun l => ss.regs addr l + 1) (fun l => ss.regs s l >>> 8)
        pc := ss.pc + 1 } := by
  simp only [sstep, h, sstepInstr]

theorem setp_step (prog : Array SInstr) (ss : SState) (c : SCmp) (pp a : String) (b : SArg)
    (h : prog[ss.pc]? = some (.setp c pp a b)) :
    sstep prog ss = (ss.setReg pp
      (fun l => if c.run (ss.regs a l) (ss.get l b) then 1 else 0)).setPc (ss.pc + 1) := by
  simp only [sstep, h, sstepInstr]

theorem bnot_step (prog : Array SInstr) (ss : SState) (d a : String)
    (h : prog[ss.pc]? = some (.bnot d a)) :
    sstep prog ss = (ss.setReg d (fun l => ~~~ (ss.regs a l))).setPc (ss.pc + 1) := by
  simp only [sstep, h, sstepInstr]

theorem barwarp_step (prog : Array SInstr) (ss : SState)
    (h : prog[ss.pc]? = some .barwarp) :
    sstep prog ss = ss.setPc (ss.pc + 1) := by
  simp only [sstep, h, sstepInstr]

/-- A 0/1 predicate register equals `1` exactly on its condition. -/
theorem ite01_eq_one (c : Prop) [Decidable c] :
    ((if c then (1 : UInt64) else 0) == 1) = decide c := by
  by_cases hc : c <;> simp [hc]

/-- Exact kernel extend predicate segment.  Starting at the first instruction
    after label `ext`, the concrete compressor computes:
    `idx = ml + lane; pe = p0 + idx; pIn = pe < ecR; peC = min pe ec1;
    dfe = peC - p0; caC = cand0 + dfe; ...; pOk = pIn && byte-eq`.

    This lemma is deliberately still in machine terms; the model-facing bridge
    discharges the `UInt64` no-overflow/clamp facts separately. -/
theorem extendKernelGoodPred_machine (prog : Array SInstr) (ss : SState)
    (h0 : prog[ss.pc]? = some (.binr .add "idx" "ml" "lane"))
    (h1 : prog[ss.pc + 1]? = some (.binr .add "pe" "p0" "idx"))
    (h2 : prog[ss.pc + 2]? = some (.setp .lt "pIn" "pe" (.reg "ecR")))
    (h3 : prog[ss.pc + 3]? = some (.binr .min "peC" "pe" "ec1"))
    (h4 : prog[ss.pc + 4]? = some (.binr .sub "dfe" "peC" "p0"))
    (h5 : prog[ss.pc + 5]? = some (.binr .add "caC" "cand0" "dfe"))
    (h6 : prog[ss.pc + 6]? = some (.mov "peD" (.reg "peC")))
    (h7 : prog[ss.pc + 7]? = some (.binr .add "aP" "inBase" "peD"))
    (h8 : prog[ss.pc + 8]? = some (.mov "caD" (.reg "caC")))
    (h9 : prog[ss.pc + 9]? = some (.binr .add "aC" "inBase" "caD"))
    (h10 : prog[ss.pc + 10]? = some (.ldgo "bP" "aP" 0))
    (h11 : prog[ss.pc + 11]? = some (.ldgo "bC" "aC" 0))
    (h12 : prog[ss.pc + 12]? = some (.setp .eq "pEqB" "bP" (.reg "bC")))
    (h13 : prog[ss.pc + 13]? = some (.andp "pOk" "pIn" "pEqB"))
    (l : Fin 32) :
    (snsteps prog 14 ss).regs "pOk" l
      =
        (if ((SOp.run .add (ss.regs "p0" l)
                (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l))) < ss.regs "ecR" l ∧
             UInt64.ofNat
                (ss.gmem.getD
                  (SOp.run .add (ss.regs "inBase" l)
                    (SOp.run .min
                      (SOp.run .add (ss.regs "p0" l)
                        (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                      (ss.regs "ec1" l))).toNat 0).toNat
             ==
             UInt64.ofNat
                (ss.gmem.getD
                  (SOp.run .add (ss.regs "inBase" l)
                    (SOp.run .add (ss.regs "cand0" l)
                      (SOp.run .sub
                        (SOp.run .min
                          (SOp.run .add (ss.regs "p0" l)
                            (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                          (ss.regs "ec1" l))
                        (ss.regs "p0" l)))).toNat 0).toNat)
          then 1 else 0) := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13,
    sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run, ite01_eq_one]

/-- Bridge the exact machine expression produced by `extendKernelGoodPred_machine`
    to W2's `good`, assuming the loop invariant has already related the machine
    bounds test and byte comparison to model indices. -/
theorem extendKernelGoodPred_machine_to_good (ss : SState) (inp : List UInt8)
    (p c endCap ml : Nat) (l : Fin 32)
    (hbound :
      ((SOp.run .add (ss.regs "p0" l)
          (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l))) < ss.regs "ecR" l)
        ↔ p + (ml + l.val) < endCap)
    (hbytes : p + (ml + l.val) < endCap →
      (UInt64.ofNat
          (ss.gmem.getD
            (SOp.run .add (ss.regs "inBase" l)
              (SOp.run .min
                (SOp.run .add (ss.regs "p0" l)
                  (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                (ss.regs "ec1" l))).toNat 0).toNat
       ==
       UInt64.ofNat
          (ss.gmem.getD
            (SOp.run .add (ss.regs "inBase" l)
              (SOp.run .add (ss.regs "cand0" l)
                (SOp.run .sub
                  (SOp.run .min
                    (SOp.run .add (ss.regs "p0" l)
                      (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                    (ss.regs "ec1" l))
                  (ss.regs "p0" l)))).toNat 0).toNat)
        = (byte inp (p + (ml + l.val)) == byte inp (c + (ml + l.val)))) :
    (if ((SOp.run .add (ss.regs "p0" l)
            (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l))) < ss.regs "ecR" l ∧
         UInt64.ofNat
            (ss.gmem.getD
              (SOp.run .add (ss.regs "inBase" l)
                (SOp.run .min
                  (SOp.run .add (ss.regs "p0" l)
                    (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                  (ss.regs "ec1" l))).toNat 0).toNat
         ==
         UInt64.ofNat
            (ss.gmem.getD
              (SOp.run .add (ss.regs "inBase" l)
                (SOp.run .add (ss.regs "cand0" l)
                  (SOp.run .sub
                    (SOp.run .min
                      (SOp.run .add (ss.regs "p0" l)
                        (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                      (ss.regs "ec1" l))
                    (ss.regs "p0" l)))).toNat 0).toNat)
      then (1 : UInt64) else 0)
      = (if AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + l.val) then (1 : UInt64) else 0) := by
  unfold AlgorithmLib.LZ4WarpSched.good
  by_cases hb : p + (ml + l.val) < endCap
  · have hmb :
        (SOp.run .add (ss.regs "p0" l)
          (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l))) < ss.regs "ecR" l :=
      hbound.mpr hb
    by_cases hbeq : byte inp (p + (ml + l.val)) == byte inp (c + (ml + l.val))
    · have hmbeq :
          (UInt64.ofNat
              (ss.gmem.getD
                (SOp.run .add (ss.regs "inBase" l)
                  (SOp.run .min
                    (SOp.run .add (ss.regs "p0" l)
                      (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                    (ss.regs "ec1" l))).toNat 0).toNat
           ==
           UInt64.ofNat
              (ss.gmem.getD
                (SOp.run .add (ss.regs "inBase" l)
                  (SOp.run .add (ss.regs "cand0" l)
                    (SOp.run .sub
                      (SOp.run .min
                        (SOp.run .add (ss.regs "p0" l)
                          (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                        (ss.regs "ec1" l))
                      (ss.regs "p0" l)))).toNat 0).toNat) = true := by
        rw [hbytes hb, hbeq]
      rw [if_pos ⟨hmb, hmbeq⟩]
      simp [hb]
      exact beq_iff_eq.mp hbeq
    · have hmbeq :
          (UInt64.ofNat
              (ss.gmem.getD
                (SOp.run .add (ss.regs "inBase" l)
                  (SOp.run .min
                    (SOp.run .add (ss.regs "p0" l)
                      (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                    (ss.regs "ec1" l))).toNat 0).toNat
           ==
           UInt64.ofNat
              (ss.gmem.getD
                (SOp.run .add (ss.regs "inBase" l)
                  (SOp.run .add (ss.regs "cand0" l)
                    (SOp.run .sub
                      (SOp.run .min
                        (SOp.run .add (ss.regs "p0" l)
                          (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l)))
                        (ss.regs "ec1" l))
                      (ss.regs "p0" l)))).toNat 0).toNat) = false := by
        rw [hbytes hb]
        cases hbv : (byte inp (p + (ml + l.val)) == byte inp (c + (ml + l.val))) with
        | false => rfl
        | true => exact False.elim (hbeq hbv)
      rw [if_neg (fun h => by exact Bool.noConfusion (hmbeq ▸ h.2))]
      simp [hb]
      intro heq
      exact hbeq (beq_iff_eq.mpr heq)
  · have hmb :
        ¬ (SOp.run .add (ss.regs "p0" l)
          (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l))) < ss.regs "ecR" l :=
      fun h => hb (hbound.mp h)
    rw [if_neg (fun h => hmb h.1), decide_eq_false hb]
    simp

private theorem u64_beq_byte_toNat (a b : UInt8) :
    (UInt64.ofNat a.toNat == UInt64.ofNat b.toNat) = (a.toNat == b.toNat) := by
  have ha : a.toNat < 2 ^ 64 := by
    have := a.toNat_lt
    omega
  have hb : b.toNat < 2 ^ 64 := by
    have := b.toNat_lt
    omega
  rw [Bool.eq_iff_iff, beq_iff_eq, beq_iff_eq, ← UInt64.toNat_inj,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt a.toNat ha,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt b.toNat hb]

private theorem u8_toNat_beq (a b : UInt8) :
    (a.toNat == b.toNat) = (a == b) := by
  rw [Bool.eq_iff_iff, beq_iff_eq, beq_iff_eq]
  constructor
  · intro h
    exact UInt8.toNat_inj.mp h
  · intro h
    exact congrArg UInt8.toNat h

theorem extendKernelGoodPred_pc (prog : Array SInstr) (ss : SState)
    (h0 : prog[ss.pc]? = some (.binr .add "idx" "ml" "lane"))
    (h1 : prog[ss.pc + 1]? = some (.binr .add "pe" "p0" "idx"))
    (h2 : prog[ss.pc + 2]? = some (.setp .lt "pIn" "pe" (.reg "ecR")))
    (h3 : prog[ss.pc + 3]? = some (.binr .min "peC" "pe" "ec1"))
    (h4 : prog[ss.pc + 4]? = some (.binr .sub "dfe" "peC" "p0"))
    (h5 : prog[ss.pc + 5]? = some (.binr .add "caC" "cand0" "dfe"))
    (h6 : prog[ss.pc + 6]? = some (.mov "peD" (.reg "peC")))
    (h7 : prog[ss.pc + 7]? = some (.binr .add "aP" "inBase" "peD"))
    (h8 : prog[ss.pc + 8]? = some (.mov "caD" (.reg "caC")))
    (h9 : prog[ss.pc + 9]? = some (.binr .add "aC" "inBase" "caD"))
    (h10 : prog[ss.pc + 10]? = some (.ldgo "bP" "aP" 0))
    (h11 : prog[ss.pc + 11]? = some (.ldgo "bC" "aC" 0))
    (h12 : prog[ss.pc + 12]? = some (.setp .eq "pEqB" "bP" (.reg "bC")))
    (h13 : prog[ss.pc + 13]? = some (.andp "pOk" "pIn" "pEqB")) :
    (snsteps prog 14 ss).pc = ss.pc + 14 := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13,
    sstepInstr, SState.setReg, SState.setPc, SState.get]

theorem extendKernelGoodPred_frame_core (prog : Array SInstr) (ss : SState)
    (h0 : prog[ss.pc]? = some (.binr .add "idx" "ml" "lane"))
    (h1 : prog[ss.pc + 1]? = some (.binr .add "pe" "p0" "idx"))
    (h2 : prog[ss.pc + 2]? = some (.setp .lt "pIn" "pe" (.reg "ecR")))
    (h3 : prog[ss.pc + 3]? = some (.binr .min "peC" "pe" "ec1"))
    (h4 : prog[ss.pc + 4]? = some (.binr .sub "dfe" "peC" "p0"))
    (h5 : prog[ss.pc + 5]? = some (.binr .add "caC" "cand0" "dfe"))
    (h6 : prog[ss.pc + 6]? = some (.mov "peD" (.reg "peC")))
    (h7 : prog[ss.pc + 7]? = some (.binr .add "aP" "inBase" "peD"))
    (h8 : prog[ss.pc + 8]? = some (.mov "caD" (.reg "caC")))
    (h9 : prog[ss.pc + 9]? = some (.binr .add "aC" "inBase" "caD"))
    (h10 : prog[ss.pc + 10]? = some (.ldgo "bP" "aP" 0))
    (h11 : prog[ss.pc + 11]? = some (.ldgo "bC" "aC" 0))
    (h12 : prog[ss.pc + 12]? = some (.setp .eq "pEqB" "bP" (.reg "bC")))
    (h13 : prog[ss.pc + 13]? = some (.andp "pOk" "pIn" "pEqB")) :
    (snsteps prog 14 ss).regs "p0" = ss.regs "p0" ∧
    (snsteps prog 14 ss).regs "cand0" = ss.regs "cand0" ∧
    (snsteps prog 14 ss).regs "ml" = ss.regs "ml" ∧
    (snsteps prog 14 ss).regs "lane" = ss.regs "lane" ∧
    (snsteps prog 14 ss).regs "ecR" = ss.regs "ecR" ∧
    (snsteps prog 14 ss).regs "ec1" = ss.regs "ec1" ∧
    (snsteps prog 14 ss).regs "inBase" = ss.regs "inBase" ∧
    (snsteps prog 14 ss).gmem = ss.gmem ∧
    (snsteps prog 14 ss).smem = ss.smem := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13,
    sstepInstr, SState.setReg, SState.setPc, SState.get]

-- ── The cooperative select: `vote d p; brev d d; clz d d` ─────────────────────

/-- `firstHit` only inspects its argument on `[start, start+b)`, so it respects
    pointwise agreement there. -/
theorem firstHit_congr (f g : Nat → Bool) : ∀ (b start : Nat),
    (∀ k, start ≤ k → k < start + b → f k = g k) →
    AlgorithmLib.LZ4WarpSched.firstHit f b start
      = AlgorithmLib.LZ4WarpSched.firstHit g b start := by
  intro b
  induction b with
  | zero => intro start _; rfl
  | succ b ih =>
      intro start h
      show (if f start then some start else AlgorithmLib.LZ4WarpSched.firstHit f b (start + 1))
        = (if g start then some start else AlgorithmLib.LZ4WarpSched.firstHit g b (start + 1))
      rw [h start (Nat.le_refl _) (by omega),
          ih (start + 1) (fun k hk1 hk2 => h k (by omega) (by omega))]

/-- Shifting `firstHit`'s start by one shifts its argument window and result. -/
theorem firstHit_shift (f : Nat → Bool) : ∀ (b start : Nat),
    AlgorithmLib.LZ4WarpSched.firstHit f b (start + 1)
      = (AlgorithmLib.LZ4WarpSched.firstHit (fun k => f (k + 1)) b start).map (· + 1) := by
  intro b
  induction b with
  | zero => intro start; rfl
  | succ b ih =>
      intro start
      show (if f (start + 1) then some (start + 1)
              else AlgorithmLib.LZ4WarpSched.firstHit f b (start + 1 + 1))
        = ((if f (start + 1) then some start
              else AlgorithmLib.LZ4WarpSched.firstHit (fun k => f (k + 1)) b (start + 1)).map (· + 1))
      by_cases hf : f (start + 1)
      · simp [hf]
      · simp only [hf, if_false]; exact ih (start + 1)

/-- **Extend window = model `leadingGood`**: the ballot-scan for the first
    mismatch (`firstHit` of `¬good`, defaulting to the full window `b`) equals W2's
    leading-good count — the value one `extendChunk` window advances by. -/
theorem firstHit_leadingGood (inp : List UInt8) (p c endCap : Nat) : ∀ (b ml : Nat),
    (AlgorithmLib.LZ4WarpSched.firstHit
        (fun k => ! AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + k)) b 0).getD b
      = AlgorithmLib.LZ4WarpSched.leadingGood inp p c endCap ml b := by
  intro b
  induction b with
  | zero => intro ml; rfl
  | succ b ih =>
      intro ml
      show (if (! AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + 0)) then some 0
              else AlgorithmLib.LZ4WarpSched.firstHit
                (fun k => ! AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + k)) b 1).getD (b + 1)
        = (if AlgorithmLib.LZ4WarpSched.good inp p c endCap ml = true
            then 1 + AlgorithmLib.LZ4WarpSched.leadingGood inp p c endCap (ml + 1) b else 0)
      rw [Nat.add_zero]
      by_cases hg : AlgorithmLib.LZ4WarpSched.good inp p c endCap ml = true
      · simp only [hg, Bool.not_true, if_false]
        rw [firstHit_shift (fun k => ! AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + k)) b 0]
        rw [firstHit_congr (fun k => ! AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + (k + 1)))
              (fun k => ! AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + 1 + k)) b 0
              (fun k _ _ => by simp only []; rw [show ml + (k + 1) = ml + 1 + k from by omega])]
        rw [← ih (ml + 1)]
        cases AlgorithmLib.LZ4WarpSched.firstHit
          (fun k => ! AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + 1 + k)) b 0 <;> simp <;> omega
      · simp [hg]

/-- Bit `i` of a `UInt64` complement is the negation of the original bit. -/
theorem not_toNat_testBit (x : UInt64) (i : Nat) (hi : i < 64) :
    (~~~x).toNat.testBit i = ! x.toNat.testBit i := by
  simp only [← UInt64.toNat_toBitVec, BitVec.testBit_toNat, UInt64.toBitVec_not,
             BitVec.getLsbD_not, hi, decide_true, Bool.true_and]

/-- Generic `clz∘brev` = least set bit (`collective_select` without the ballot). -/
theorem clz32_brev32_firstSetNat (x : UInt64) :
    (clz32 (brev32 x)).toNat = firstSetNat (fun k => x.toNat.testBit k) 32 0 := by
  rw [clz32_brev32, firstSetNat_eq_find]
  simp only [leastSetBit, List.range_eq_range', Nat.zero_add]

/-- The kernel's extend reduction `vote balOk pOk; bnot mis balOk; brev r mis;
    clz d r` leaves every lane of `d` holding `clz∘brev∘~~~ballot`. -/
theorem select_complement_regs (prog : Array SInstr) (ss : SState) (d r mis balOk pOk : String)
    (h0 : prog[ss.pc]? = some (.vote balOk pOk))
    (h1 : prog[ss.pc + 1]? = some (.bnot mis balOk))
    (h2 : prog[ss.pc + 2]? = some (.brev r mis))
    (h3 : prog[ss.pc + 3]? = some (.clz d r)) (l : Fin 32) :
    (sstep prog (sstep prog (sstep prog (sstep prog ss)))).regs d l
      = clz32 (brev32 (~~~ (ballotOf ss.regs pOk))) := by
  rw [vote_step prog ss balOk pOk h0]
  rw [bnot_step prog ((ss.setReg balOk (fun _ => ballotOf ss.regs pOk)).setPc (ss.pc + 1))
        mis balOk h1]
  rw [brev_step prog
        ((((ss.setReg balOk (fun _ => ballotOf ss.regs pOk)).setPc (ss.pc + 1)).setReg mis
            (fun l => ~~~ (((ss.setReg balOk (fun _ => ballotOf ss.regs pOk)).setPc
              (ss.pc + 1)).regs balOk l))).setPc
          (((ss.setReg balOk (fun _ => ballotOf ss.regs pOk)).setPc (ss.pc + 1)).pc + 1)) r mis h2]
  rw [clz_step prog _ d r h3]
  simp [SState.setReg, SState.setPc]

/-- **Complement-select = `firstHit` of the negated ballot** (the kernel's extend
    first-mismatch reduction): `d.toNat` is the earliest lane where the good-ballot
    is `0`, i.e. `firstHit (¬ballot-bit)` — feeds `firstHit_leadingGood`. -/
theorem select_complement_firstHit (prog : Array SInstr) (ss : SState) (d r mis balOk pOk : String)
    (h0 : prog[ss.pc]? = some (.vote balOk pOk))
    (h1 : prog[ss.pc + 1]? = some (.bnot mis balOk))
    (h2 : prog[ss.pc + 2]? = some (.brev r mis))
    (h3 : prog[ss.pc + 3]? = some (.clz d r)) (l : Fin 32) :
    ((sstep prog (sstep prog (sstep prog (sstep prog ss)))).regs d l).toNat
      = (AlgorithmLib.LZ4WarpSched.firstHit
          (fun k => ! (ballotOf ss.regs pOk).toNat.testBit k) 32 0).getD 32 := by
  rw [select_complement_regs prog ss d r mis balOk pOk h0 h1 h2 h3 l, clz32_brev32_firstSetNat,
      firstSetNat_eq_firstHit]
  apply congrArg (Option.getD · 32)
  apply firstHit_congr
  intro k _ hk
  have hk64 : k < 64 := by omega
  rw [not_toNat_testBit _ k hk64]

/-- **Kernel extend window = model `leadingGood`**: the kernel's exact 4-instruction
    reduction (`vote balOk pOk; bnot mis balOk; brev r mis; clz d r`), given `pOk`
    holds the `good` predicate per lane, computes W2's leading-good count. -/
theorem extendKernelWindow (prog : Array SInstr) (ss : SState) (d r mis balOk pOk : String)
    (inp : List UInt8) (p c endCap ml : Nat)
    (h0 : prog[ss.pc]? = some (.vote balOk pOk))
    (h1 : prog[ss.pc + 1]? = some (.bnot mis balOk))
    (h2 : prog[ss.pc + 2]? = some (.brev r mis))
    (h3 : prog[ss.pc + 3]? = some (.clz d r))
    (hgood : ∀ k : Fin 32,
      ss.regs pOk k = if AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + k.val) then 1 else 0)
    (l : Fin 32) :
    ((sstep prog (sstep prog (sstep prog (sstep prog ss)))).regs d l).toNat
      = AlgorithmLib.LZ4WarpSched.leadingGood inp p c endCap ml 32 := by
  rw [select_complement_firstHit prog ss d r mis balOk pOk h0 h1 h2 h3 l,
      firstHit_congr _ (fun k => ! AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + k)) 32 0
        (fun k _ hk => ?_)]
  · exact firstHit_leadingGood inp p c endCap 32 ml
  · have hk32 : k < 32 := by omega
    rw [ballotOf_testBit ss.regs pOk ⟨k, hk32⟩, hgood ⟨k, hk32⟩]
    by_cases hg : AlgorithmLib.LZ4WarpSched.good inp p c endCap (ml + k) = true <;> simp [hg]

-- ── Extend collective capstone: one 32-window = model `leadingGood` ───────────

end AlgorithmLib.LZ4WarpDSL
