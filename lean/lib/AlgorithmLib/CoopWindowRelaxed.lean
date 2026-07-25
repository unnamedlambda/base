import AlgorithmLib.CoopWindowLeaf

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib.LZ4WarpDSL AlgorithmLib.LZ4Simt AlgorithmLib.LZ4WarpFind
open AlgorithmLib.LZ4SimtBits (ballotOf_testBit ballotOf_toNat)
open AlgorithmLib.LZ4Ptx (toNat_ofNat_lt u64_add_ofNat)

/-- `probe` returns `none` past `searchLim` — its `p < searchLim` guard fails.
    Stated over abstract `inp`/`oracle` so no heavy definition (`gmemInp`/`tableOracle`)
    unfolds. -/
theorem probe_isNone_of_ge (inp : List UInt8) (oracle : Nat → Option Nat)
    (searchLim p : Nat) (hp : ¬ (p < searchLim)) :
    (probe inp oracle searchLim p).isSome = false := by
  unfold probe
  cases oracle p with
  | none => rfl
  | some c =>
      simp only [if_neg (show ¬ (p < searchLim ∧ c < p ∧ verify4 inp p c = true)
        from fun hc => hp hc.1), Option.isSome_none]

/-- Out-of-range lane: `(snsteps 34).pValid l = 0`. -/
theorem coopWindow_pValid34_zero (prog : Array SInstr) (ss : SState) (base : Nat)
    (s inStride searchLim capC hashLog tbl : Nat) (l : Fin 32) (hpc : ss.pc = base)
    (P : ProbeInstrs prog base searchLim capC hashLog)
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40)
    (hlane : ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + l.val < 2 ^ 64) (hoob : ¬ (s + l.val < searchLim)) :
    (snsteps prog 34 ss).regs "pValid" l = 0 := by
  subst hpc
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19,
    h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35, h36, h37, h38,
    h39⟩ := P
  have hpc34 : (snsteps prog 34 ss).pc = ss.pc + 34 :=
    coopWindow_hpc34 prog ss ss.pc searchLim capC hashLog rfl ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8,
      h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27,
      h28, h29, h30, h31, h32, h33, h34, h35, h36, h37, h38, h39⟩
  obtain ⟨_, hpv40⟩ := coopWindow_posPValid40 prog ss ss.pc s inStride searchLim capC hashLog tbl
    l rfl h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23
    h24 h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39
    hsl hcapdef hcapb hhl htblb hlane hsp hp64
  have e40 : snsteps prog 40 ss = snsteps prog 6 (snsteps prog 34 ss) := snsteps_add prog 34 6 ss
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
  have hpvD : (snsteps prog 40 ss).regs "pValid" l = (snsteps prog 34 ss).regs "pValid" l := by
    rw [e40]; exact segD_frame prog (snsteps prog 34 ss) "pValid" l hD0 hD1 hD2 hD3 hD4 hD5 (by decide)
  rw [← hpvD, hpv40, hsp, hlane, if_neg]
  rw [UInt64.lt_iff_toNat_lt, u64_add_ofNat s l.val hp64, toNat_ofNat_lt searchLim (by omega)]
  omega

/-- Predicate-segment core: from the 6 instruction facts (34..39) as OPAQUE
    hypotheses and `pValid34 = 0`, `cw_pHit_comb` gives `pHit40 = 1 ↔ (pValid34 = 1
    ∧ …) ∧ …`, refuted by `pValid34 = 0`.  Kept separate so `cw_pHit_comb` only ever
    sees `hD0..hD5` as opaque variables (deriving them inline here would embed the
    large `hpc34`/`coopWindow_hpc34` proof and overflow the kernel). -/
theorem coopWindow_pHit40_core (prog : Array SInstr) (ss : SState) (l : Fin 32)
    (hD0 : prog[(snsteps prog 34 ss).pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (hD1 : prog[(snsteps prog 34 ss).pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (hD2 : prog[(snsteps prog 34 ss).pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (hD3 : prog[(snsteps prog 34 ss).pc + 3]? = some (.andp "pH1" "pValid" "pNE"))
    (hD4 : prog[(snsteps prog 34 ss).pc + 4]? = some (.andp "pH2" "pH1" "pCO"))
    (hD5 : prog[(snsteps prog 34 ss).pc + 5]? = some (.andp "pHit" "pH2" "pEq"))
    (hpValid34 : (snsteps prog 34 ss).regs "pValid" l = 0) :
    (snsteps prog 40 ss).regs "pHit" l = 1 ↔ False := by
  have e40 : snsteps prog 40 ss = snsteps prog 6 (snsteps prog 34 ss) := snsteps_add prog 34 6 ss
  constructor
  · intro h
    rw [e40] at h
    obtain ⟨⟨hpv1, -, -⟩, -⟩ := (cw_pHit_comb prog (snsteps prog 34 ss) l hD0 hD1 hD2 hD3 hD4 hD5).mp h
    rw [hpValid34] at hpv1; exact absurd hpv1 (by decide)
  · exact False.elim

/-- Out-of-range lane: `pHit40 = 0`.  Derives the 6 instruction facts from `hpc34`
    and hands them to `coopWindow_pHit40_core` (the `cw_pHit_comb`/derivation split
    keeps both proof terms small). -/
theorem coopWindow_pHit40_zero (prog : Array SInstr) (ss : SState) (base : Nat)
    (searchLim capC hashLog : Nat) (l : Fin 32) (hpc : ss.pc = base)
    (P : ProbeInstrs prog base searchLim capC hashLog)
    (hpc34 : (snsteps prog 34 ss).pc = ss.pc + 34)
    (hpValid34 : (snsteps prog 34 ss).regs "pValid" l = 0) :
    (snsteps prog 40 ss).regs "pHit" l = 1 ↔ False := by
  have hD0 : prog[(snsteps prog 34 ss).pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)) := by
    rw [hpc34, hpc]; exact P.h34
  have hD1 : prog[(snsteps prog 34 ss).pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")) := by
    rw [hpc34, hpc]; exact P.h35
  have hD2 : prog[(snsteps prog 34 ss).pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")) := by
    rw [hpc34, hpc]; exact P.h36
  have hD3 : prog[(snsteps prog 34 ss).pc + 3]? = some (.andp "pH1" "pValid" "pNE") := by
    rw [hpc34, hpc]; exact P.h37
  have hD4 : prog[(snsteps prog 34 ss).pc + 4]? = some (.andp "pH2" "pH1" "pCO") := by
    rw [hpc34, hpc]; exact P.h38
  have hD5 : prog[(snsteps prog 34 ss).pc + 5]? = some (.andp "pHit" "pH2" "pEq") := by
    rw [hpc34, hpc]; exact P.h39
  exact coopWindow_pHit40_core prog ss l hD0 hD1 hD2 hD3 hD4 hD5 hpValid34

/-- In-range branch: `s + l.val < searchLim ≤ capC`, so the original `coopWindow_pHit_iff`
    applies (its bound derived from `hv`).  Own lemma to keep the big `pHit_iff` call
    out of the dispatcher's term. -/
theorem coopWindow_pHit_relaxed_valid (prog : Array SInstr) (ss : SState) (base : Nat)
    (s inStride searchLim capC hashLog tbl ib : Nat) (l : Fin 32) (hpc : ss.pc = base)
    (P : ProbeInstrs prog base searchLim capC hashLog)
    (hinb : ss.regs "inBase" l = UInt64.ofNat ib) (htbl : ss.regs "tbl" l = UInt64.ofNat tbl)
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40)
    (hlane : ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + l.val < 2 ^ 64) (hv : s + l.val < searchLim)
    (hib64 : ib + capC < 2 ^ 64) :
    ((snsteps prog 40 ss).regs "pHit" l = 1) ↔
      (probe (gmemInpAt ss.gmem ib inStride) (tableOracle ss.gmem ss.smem hashLog tbl ib)
        searchLim (s + l.val)).isSome := by
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19,
    h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35, h36, h37, h38,
    h39⟩ := P
  exact coopWindow_pHit_iff prog ss base ib s inStride searchLim capC hashLog tbl l hpc
    h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24
    h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39
    hinb htbl hsl hcapdef hcapb hhl htblb (by omega) hlane hsp hp64 hib64

/-- Out-of-range branch: `pValid = 0 ⟹ pHit = 0` and `probe = none` (both false).
    Own lemma (`hpc34` + `pValid34_zero` + `pHit40_zero`) so the dispatcher stays tiny. -/
theorem coopWindow_pHit_relaxed_oob (prog : Array SInstr) (ss : SState) (base : Nat)
    (s inStride searchLim capC hashLog tbl ib : Nat) (l : Fin 32) (hpc : ss.pc = base)
    (P : ProbeInstrs prog base searchLim capC hashLog)
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40)
    (hlane : ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + l.val < 2 ^ 64) (hv : ¬ (s + l.val < searchLim))
    (hib64 : ib + capC < 2 ^ 64) :
    ((snsteps prog 40 ss).regs "pHit" l = 1) ↔
      (probe (gmemInpAt ss.gmem ib inStride) (tableOracle ss.gmem ss.smem hashLog tbl ib)
        searchLim (s + l.val)).isSome := by
  have hpc34 : (snsteps prog 34 ss).pc = ss.pc + 34 :=
    coopWindow_hpc34 prog ss base searchLim capC hashLog hpc P
  have hpValid34 : (snsteps prog 34 ss).regs "pValid" l = 0 :=
    coopWindow_pValid34_zero prog ss base s inStride searchLim capC hashLog tbl l hpc P
      hsl hcapdef hcapb hhl htblb hlane hsp hp64 hv
  have hpHit : (snsteps prog 40 ss).regs "pHit" l = 1 ↔ False :=
    coopWindow_pHit40_zero prog ss base searchLim capC hashLog l hpc P hpc34 hpValid34
  have hprobe := probe_isNone_of_ge (gmemInpAt ss.gmem ib inStride)
    (tableOracle ss.gmem ss.smem hashLog tbl ib) searchLim (s + l.val) hv
  rw [hpHit]; simp [hprobe]

/-- **Relaxed `pHit` bridge** — `coopWindow_pHit_iff` WITHOUT `s + l.val ≤ capC`.
    In-range lanes defer to the original; out-of-range lanes give both sides false. -/
theorem coopWindow_pHit_iff_relaxed (prog : Array SInstr) (ss : SState) (base : Nat)
    (s inStride searchLim capC hashLog tbl ib : Nat) (l : Fin 32) (hpc : ss.pc = base)
    (P : ProbeInstrs prog base searchLim capC hashLog)
    (hinb : ss.regs "inBase" l = UInt64.ofNat ib) (htbl : ss.regs "tbl" l = UInt64.ofNat tbl)
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40)
    (hlane : ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + l.val < 2 ^ 64)
    (hib64 : ib + capC < 2 ^ 64) :
    ((snsteps prog 40 ss).regs "pHit" l = 1) ↔
      (probe (gmemInpAt ss.gmem ib inStride) (tableOracle ss.gmem ss.smem hashLog tbl ib)
        searchLim (s + l.val)).isSome := by
  by_cases hv : s + l.val < searchLim
  · exact coopWindow_pHit_relaxed_valid prog ss base s inStride searchLim capC hashLog tbl ib l hpc P
      hinb htbl hsl hcapdef hcapb hhl htblb hlane hsp hp64 hv hib64
  · exact coopWindow_pHit_relaxed_oob prog ss base s inStride searchLim capC hashLog tbl ib l hpc P
      hsl hcapdef hcapb hhl htblb hlane hsp hp64 hv hib64

/-- `coopWindow_upto40_eq` with `hsbound` dropped: the `clz∘brev∘ballot` extend count
    equals the model `window` result, needing only the overflow bound `s + 32 < 2^64`
    (out-of-range lanes handled by `coopWindow_pHit_iff_relaxed`). -/
theorem coopWindow_upto40_relaxed (prog : Array SInstr) (ss : SState) (base : Nat)
    (s inStride searchLim capC hashLog tbl ib : Nat) (hpc : ss.pc = base)
    (P : ProbeInstrs prog base searchLim capC hashLog)
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40)
    (hInb0 : ∀ l : Fin 32, ss.regs "inBase" l = UInt64.ofNat ib)
    (htbl0 : ∀ l : Fin 32, ss.regs "tbl" l = UInt64.ofNat tbl)
    (hlane : ∀ l : Fin 32, ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ∀ l : Fin 32, ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + 32 < 2 ^ 64)
    (hib64 : ib + capC < 2 ^ 64) :
    (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat
      = (match AlgorithmLib.LZ4WarpFind.window (gmemInpAt ss.gmem ib inStride)
            (tableOracle ss.gmem ss.smem hashLog tbl ib) searchLim s with
          | some (p, _) => p - s | none => 32) := by
  have hballot' : ∀ k, k < 32 →
      (ballotOf (snsteps prog 40 ss).regs "pHit").toNat.testBit k
        = (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
            (tableOracle ss.gmem ss.smem hashLog tbl ib) searchLim (s + k)).isSome := by
    intro k hk
    rw [AlgorithmLib.LZ4SimtBits.ballotOf_testBit (snsteps prog 40 ss).regs "pHit" ⟨k, hk⟩]
    have hiff := coopWindow_pHit_iff_relaxed prog ss base s inStride searchLim capC hashLog tbl ib
      ⟨k, hk⟩ hpc P (hInb0 ⟨k, hk⟩) (htbl0 ⟨k, hk⟩) hsl hcapdef hcapb hhl htblb
      (hlane ⟨k, hk⟩) (hsp ⟨k, hk⟩) (by simp only []; omega) hib64
    by_cases hp : ((snsteps prog 40 ss).regs "pHit" ⟨k, hk⟩) = 1
    · rw [beq_iff_eq.mpr hp]
      exact (hiff.mp hp).symm ▸ rfl
    · rw [beq_eq_false_iff_ne.mpr hp]
      rcases hprobe : (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
          (tableOracle ss.gmem ss.smem hashLog tbl ib) searchLim (s + k)).isSome with _ | _
      · rfl
      · exact absurd (hiff.mpr (by rw [hprobe])) hp
  have hclz' : (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat
      = (AlgorithmLib.LZ4WarpSched.firstHit
          (fun k => (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
            (tableOracle ss.gmem ss.smem hashLog tbl ib) searchLim (s + k)).isSome) 32 0).getD 32 := by
    rw [AlgorithmLib.LZ4SimtBits.collective_select_firstHit (snsteps prog 40 ss).regs "pHit",
        firstHit_congr _ _ 32 0 (fun k _ hk => hballot' k hk)]
  rw [hclz']
  exact firstHit_getD_eq_window (gmemInpAt ss.gmem ib inStride)
    (tableOracle ss.gmem ss.smem hashLog tbl ib) searchLim s

/-- `coopWindow_stshp46` with `hsbound` dropped: `haddr` is only needed for the
    writing (in-range) lanes, where `addr40` applies without the bound. -/
theorem coopWindow_stshp46_relaxed (prog : Array SInstr) (ss : SState) (base : Nat)
    (s inStride searchLim capC hashLog ib : Nat)
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
    (hhl : hashLog ≤ 32) (hInb0 : ∀ l : Fin 32, ss.regs "inBase" l = UInt64.ofNat ib)
    (htbl0 : ∀ l : Fin 32, ss.regs "tbl" l = 0)
    (hlane : ∀ l : Fin 32, ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ∀ l : Fin 32, ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + 32 < 2 ^ 64)
    (hpc : ss.pc = base)
    (hpc40 : (snsteps prog 40 ss).pc = base + 40)
    (e46 : snsteps prog 46 ss = snsteps prog 6 (snsteps prog 40 ss))
    (htail1 : ∀ l : Fin 32, (snsteps prog 46 ss).regs "pIns" l
        = (if (snsteps prog 40 ss).regs "lane" l ≤ clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))
              ∧ (snsteps prog 40 ss).regs "pValid" l = 1 then 1 else 0))
    (htail2 : ∀ l : Fin 32, (snsteps prog 46 ss).regs "pp1" l = (snsteps prog 40 ss).regs "posP" l + 1)
    (htail3 : ∀ l : Fin 32, (snsteps prog 46 ss).regs "addr" l = (snsteps prog 40 ss).regs "addr" l)
    (hstshp : prog[(snsteps prog 46 ss).pc]? = some (.stshp "pIns" "addr" "pp1"))
    (hib64 : ib + capC < 2 ^ 64) :
    (sstep prog (snsteps prog 46 ss)).smem
      = tableInsert (snsteps prog 46 ss).smem ss.gmem hashLog 0 searchLim ib s
          ((clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat) := by
  have hposPValid := fun l : Fin 32 =>
    coopWindow_posPValid40 prog ss base s inStride searchLim capC hashLog 0 l hpc
      h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24
      h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39
      hsl hcapdef hcapb hhl (by decide) (hlane l) (hsp l) (by
        have hl32 : l.val < 32 := l.isLt; omega)
  have haddr40 : ∀ l : Fin 32, (l.val ≤ (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat ∧ s + l.val < searchLim) →
      (snsteps prog 40 ss).regs "addr" l = UInt64.ofNat (0 + 2 * wHash ss.gmem hashLog (ib + (s + l.val))) :=
    fun l hg =>
    coopWindow_addr40 prog ss base ib s inStride searchLim capC hashLog 0 l hpc
      h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24
      h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39
      hsl hcapdef hcapb hhl (by decide) (hInb0 l) (by rw [htbl0 l]; rfl) (by
        have := hg.2; omega)
      (hlane l) (hsp l) (by have hl32 : l.val < 32 := l.isLt; omega) hib64
  have hlane40 : ∀ l : Fin 32, (snsteps prog 40 ss).regs "lane" l = UInt64.ofNat l.val := by
    intro l
    rw [coopWindow_lane40 prog ss base s inStride searchLim capC hashLog 0 l hpc
      h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24
      h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39, hlane l]
  have hpIns46 : ∀ l : Fin 32, (snsteps prog 46 ss).regs "pIns" l
      = (if l.val ≤ (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat
              ∧ s + l.val < searchLim then 1 else 0) := by
    intro l
    rw [htail1 l, hlane40 l]
    rw [(hposPValid l).2, hsp, hlane]
    have hslt : (UInt64.ofNat s + UInt64.ofNat l.val < UInt64.ofNat searchLim)
        ↔ s + l.val < searchLim := by
      rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_add,
        AlgorithmLib.LZ4Ptx.toNat_ofNat_lt s (by have := l.isLt; omega),
        AlgorithmLib.LZ4Ptx.toNat_ofNat_lt l.val (by have := l.isLt; omega),
        AlgorithmLib.LZ4Ptx.toNat_ofNat_lt searchLim (by omega),
        Nat.mod_eq_of_lt (by have := l.isLt; omega)]
    have hpv : (if UInt64.ofNat s + UInt64.ofNat l.val < UInt64.ofNat searchLim then (1:UInt64) else 0) = 1
        ↔ s + l.val < searchLim := by
      by_cases hh : s + l.val < searchLim
      · simp [hslt.mpr hh, hh]
      · have : ¬ (UInt64.ofNat s + UInt64.ofNat l.val < UInt64.ofNat searchLim) := fun hx => hh (hslt.mp hx)
        simp [this, hh]
    have hle : (UInt64.ofNat l.val ≤ clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit")))
        ↔ l.val ≤ (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat := by
      rw [UInt64.le_iff_toNat_le,
        AlgorithmLib.LZ4Ptx.toNat_ofNat_lt l.val (by have := l.isLt; omega)]
    by_cases hc1 : l.val ≤ (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat
    · by_cases hc2 : s + l.val < searchLim
      · rw [if_pos ⟨hle.mpr hc1, hpv.mpr hc2⟩, if_pos ⟨hc1, hc2⟩]
      · rw [if_neg (fun h => hc2 (hpv.mp h.2)), if_neg (fun h => hc2 h.2)]
    · rw [if_neg (fun h => hc1 (hle.mp h.1)), if_neg (fun h => hc1 h.1)]
  have haddr46 : ∀ l : Fin 32, (l.val ≤ (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat ∧ s + l.val < searchLim) →
      (snsteps prog 46 ss).regs "addr" l = UInt64.ofNat (0 + 2 * wHash ss.gmem hashLog (ib + (s + l.val))) := by
    intro l hg; rw [htail3 l]; exact haddr40 l hg
  have hpp1_46 : ∀ l : Fin 32, (snsteps prog 46 ss).regs "pp1" l = UInt64.ofNat (s + l.val + 1) := by
    intro l
    rw [htail2 l, (hposPValid l).1, hsp, hlane]
    rw [← UInt64.toNat_inj, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt _ (by have := l.isLt; omega)]
    rw [UInt64.toNat_add, UInt64.toNat_add,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt s (by have := l.isLt; omega),
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt l.val (by have := l.isLt; omega)]
    have h1 : (1 : UInt64).toNat = 1 := rfl
    rw [h1, Nat.mod_eq_of_lt (by have := l.isLt; omega), Nat.mod_eq_of_lt (by have := l.isLt; omega)]
  exact coopWindow_stshp_tableInsert prog (snsteps prog 46 ss) ss.gmem
    hashLog 0 searchLim ib s ((clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat)
    (by have hl32 : (0:Fin 32).val < 32 := (0:Fin 32).isLt; omega) (by decide) hhl
    (fun l => hpIns46 l) haddr46 hpp1_46 hstshp

/-- `coopWindow_couple_smem` with `hsbound` dropped (via `stshp46_relaxed`). -/
theorem coopWindow_couple_smem_relaxed (prog : Array SInstr) (ss : SState) (ws : WState)
    (s inStride searchLim capC hashLog ib : Nat)
    (h0 : prog[ss.pc]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[ss.pc+1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[ss.pc+2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[ss.pc+3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[ss.pc+4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[ss.pc+5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[ss.pc+6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[ss.pc+7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[ss.pc+8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[ss.pc+9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[ss.pc+10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[ss.pc+11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[ss.pc+12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[ss.pc+13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[ss.pc+14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (h15 : prog[ss.pc+15]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h16 : prog[ss.pc+16]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h17 : prog[ss.pc+17]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h18 : prog[ss.pc+18]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h19 : prog[ss.pc+19]? = some (.binr .add "addr" "hh" "tbl"))
    (h20 : prog[ss.pc+20]? = some (.ldsh "candRaw" "addr"))
    (h21 : prog[ss.pc+21]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h22 : prog[ss.pc+22]? = some (.binr .min "rc" "cand" "cap4"))
    (h23 : prog[ss.pc+23]? = some (.binr .add "rcA" "inBase" "rc"))
    (h24 : prog[ss.pc+24]? = some (.ldgo "c0" "rcA" 0))
    (h25 : prog[ss.pc+25]? = some (.ldgo "c1" "rcA" 1))
    (h26 : prog[ss.pc+26]? = some (.ldgo "c2" "rcA" 2))
    (h27 : prog[ss.pc+27]? = some (.ldgo "c3" "rcA" 3))
    (h28 : prog[ss.pc+28]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h29 : prog[ss.pc+29]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h30 : prog[ss.pc+30]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h31 : prog[ss.pc+31]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h32 : prog[ss.pc+32]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h33 : prog[ss.pc+33]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (h34 : prog[ss.pc+34]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h35 : prog[ss.pc+35]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h36 : prog[ss.pc+36]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h37 : prog[ss.pc+37]? = some (.andp "pH1" "pValid" "pNE"))
    (h38 : prog[ss.pc+38]? = some (.andp "pH2" "pH1" "pCO"))
    (h39 : prog[ss.pc+39]? = some (.andp "pHit" "pH2" "pEq"))
    (h40 : prog[ss.pc+40]? = some (.vote "bal" "pHit"))
    (h41 : prog[ss.pc+41]? = some (.brev "rev" "bal"))
    (h42 : prog[ss.pc+42]? = some (.clz "fl" "rev"))
    (h43 : prog[ss.pc+43]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h44 : prog[ss.pc+44]? = some (.andp "pIns" "pLe" "pValid"))
    (h45 : prog[ss.pc+45]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h46 : prog[ss.pc+46]? = some (.stshp "pIns" "addr" "pp1"))
    (h47 : prog[ss.pc+47]? = some (.barwarp))
    (h48 : prog[ss.pc+48]? = some (.binr .add "p0" "searchPos" "fl"))
    (h49 : prog[ss.pc+49]? = some (.shfl "cand0" "cand" "fl"))
    (h50 : prog[ss.pc+50]? = some (.setp .ne "found" "bal" (.imm 0)))
    (hsmemC : ss.smem = ws.smem) (hgmemC : ss.gmem = ws.gmem)
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (hInb0 : ∀ l : Fin 32, ss.regs "inBase" l = UInt64.ofNat ib)
    (htbl0 : ∀ l : Fin 32, ss.regs "tbl" l = 0)
    (hlane : ∀ l : Fin 32, ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ∀ l : Fin 32, ss.regs "searchPos" l = UInt64.ofNat s)
    (hsval : (ws.regs "searchPos").toNat = s)
    (hwsib : ws.regs "inBase" = UInt64.ofNat ib)
    (hp64 : s + 32 < 2 ^ 64)
    (hpc40 : (snsteps prog 40 ss).pc = ss.pc + 40)
    (t0 : prog[(snsteps prog 40 ss).pc]? = some (.vote "bal" "pHit"))
    (t1 : prog[(snsteps prog 40 ss).pc + 1]? = some (.brev "rev" "bal"))
    (t2 : prog[(snsteps prog 40 ss).pc + 2]? = some (.clz "fl" "rev"))
    (t3 : prog[(snsteps prog 40 ss).pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (t4 : prog[(snsteps prog 40 ss).pc + 4]? = some (.andp "pIns" "pLe" "pValid"))
    (t5 : prog[(snsteps prog 40 ss).pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (hib64 : ib + capC < 2 ^ 64) :
    (snsteps prog 51 ss).smem
      = (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0 ws).smem := by
  have e46 : snsteps prog 46 ss = snsteps prog 6 (snsteps prog 40 ss) := snsteps_add prog 40 6 ss
  have htail := fun l : Fin 32 => tailPre_vals prog (snsteps prog 40 ss) l t0 t1 t2 t3 t4 t5
  have hpc46 : (snsteps prog 46 ss).pc = ss.pc + 46 := by
    rw [e46, (htail 0).2.2.2.2, hpc40]
  have hstshp : prog[(snsteps prog 46 ss).pc]? = some (.stshp "pIns" "addr" "pp1") := by
    rw [hpc46]; exact h46
  have hstshp_main := coopWindow_stshp46_relaxed prog ss ss.pc s inStride searchLim capC hashLog ib
    h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24
    h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39
    hsl hcapdef hcapb hhl hInb0 htbl0 hlane hsp hp64 rfl hpc40 e46
    (fun l => (htail l).1) (fun l => (htail l).2.1) (fun l => (htail l).2.2.1) hstshp
  have e47 : snsteps prog 47 ss = sstep prog (snsteps prog 46 ss) := by
    rw [show (47:Nat) = 46 + 1 from rfl, snsteps_add]; rfl
  have hpc47 : (snsteps prog 47 ss).pc = ss.pc + 47 := by
    rw [e47, sstep, hstshp]; simp [sstepInstr, hpc46]
  have hsmem51 : (snsteps prog 51 ss).smem = (snsteps prog 47 ss).smem := by
    have e51 : snsteps prog 51 ss = snsteps prog 4 (snsteps prog 47 ss) := by
      rw [show (51:Nat) = 47 + 4 from rfl, snsteps_add]
    rw [e51]
    exact tailPost_smem prog (snsteps prog 47 ss)
      (by rw [hpc47]; exact h47) (by rw [hpc47]; exact h48)
      (by rw [hpc47]; exact h49) (by rw [hpc47]; exact h50)
  have hsmem46 : (snsteps prog 46 ss).smem = ss.smem := by
    have sD40 : (snsteps prog 40 ss).smem = ss.smem :=
      coopWindow_smem40 prog ss ss.pc searchLim capC hashLog rfl
        h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24
        h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39
    have sTail : (snsteps prog 46 ss).smem = (snsteps prog 40 ss).smem := by
      rw [e46]; exact tailPre_smem prog (snsteps prog 40 ss) t0 t1 t2 t3 t4 t5
    rw [sTail, sD40]
  have hwsibN : (ws.regs "inBase").toNat = ib := by
    rw [hwsib]; exact AlgorithmLib.LZ4Ptx.toNat_ofNat_lt ib (by omega)
  rw [evalCoopWindow_eq_go, evalCoopWindowGo_smem, hwsibN, hsval, ← hgmemC, ← hsmemC]
  rw [hsmem51, e47, hstshp_main, hsmem46]
  congr 1
  exact coopWindow_upto40_relaxed prog ss ss.pc s inStride searchLim capC hashLog 0 ib rfl
    ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20,
      h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35, h36, h37, h38, h39⟩
    hsl hcapdef hcapb hhl (by decide) hInb0 (fun l => by rw [htbl0 l]; rfl) hlane hsp hp64 hib64
  exact hib64

/-- `coopWindow_couple` with `hsbound` dropped (relaxed smem + per-lane pHit). -/
theorem coopWindow_couple_relaxed (R : List String) (prog : Array SInstr) (ss : SState) (ws : WState)
    (base s inStride searchLim capC hashLog ib : Nat)
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
    (h40 : prog[base+40]? = some (.vote "bal" "pHit"))
    (h41 : prog[base+41]? = some (.brev "rev" "bal"))
    (h42 : prog[base+42]? = some (.clz "fl" "rev"))
    (h43 : prog[base+43]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h44 : prog[base+44]? = some (.andp "pIns" "pLe" "pValid"))
    (h45 : prog[base+45]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h46 : prog[base+46]? = some (.stshp "pIns" "addr" "pp1"))
    (h47 : prog[base+47]? = some (.barwarp))
    (h48 : prog[base+48]? = some (.binr .add "p0" "searchPos" "fl"))
    (h49 : prog[base+49]? = some (.shfl "cand0" "cand" "fl"))
    (h50 : prog[base+50]? = some (.setp .ne "found" "bal" (.imm 0)))
    (hc : Couple R ss ws)
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (hInb0 : ∀ l : Fin 32, ss.regs "inBase" l = UInt64.ofNat ib)
    (hlen : inStride < 2 ^ 40)
    (htbl0 : ∀ l : Fin 32, ss.regs "tbl" l = 0)
    (hlane : ∀ l : Fin 32, ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ∀ l : Fin 32, ss.regs "searchPos" l = UInt64.ofNat s)
    (hsval : (ws.regs "searchPos").toNat = s)
    (hwsib : ws.regs "inBase" = UInt64.ofNat ib)
    (hp64 : s + 32 < 2 ^ 64)
    -- `p0`/`cand0` are dead when `found = 0` (the match model leaves them unchanged,
    -- the machine overwrites them), so they are excluded from the coupling `R`.
    (hp0R : "p0" ∉ R) (hcand0R : "cand0" ∉ R)
    -- `R` avoids the collective's internal scratch registers (but may contain `found`,
    -- the meaningful output, handled separately below).
    (hRdisj : ∀ r ∈ R, r ∉ ["posP", "pValid", "cap4", "rp", "rpA", "b0", "b1", "b2", "b3", "v32",
        "hh", "addr", "candRaw", "cand", "rc", "rcA", "c0", "c1", "c2", "c3", "vc",
        "pNE", "pCO", "pEq", "pH1", "pH2", "pHit", "bal", "rev", "fl", "pLe",
        "pIns", "pp1"])
    (hib64 : ib + capC < 2 ^ 64) :
    SReaches prog 51 ss (snsteps prog 51 ss) ∧ (snsteps prog 51 ss).pc = base + 51 ∧
      Couple R (snsteps prog 51 ss)
        (evalCoopWindow "found" "p0" "cand0" "searchPos" inStride searchLim hashLog 0 ws) := by
  subst hpc
  -- pc after the first 40 (probe) steps.
  have hpc40 : (snsteps prog 40 ss).pc = ss.pc + 40 :=
    coopWindow_pc40 prog ss ss.pc searchLim capC hashLog rfl h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12
      h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24 h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35
      h36 h37 h38 h39
  -- tail instruction facts, re-anchored at `(snsteps prog 40 ss).pc = ss.pc + 40`.
  have t0 : prog[(snsteps prog 40 ss).pc]? = some (.vote "bal" "pHit") := by rw [hpc40]; exact h40
  have t1 : prog[(snsteps prog 40 ss).pc + 1]? = some (.brev "rev" "bal") := by rw [hpc40]; exact h41
  have t2 : prog[(snsteps prog 40 ss).pc + 2]? = some (.clz "fl" "rev") := by rw [hpc40]; exact h42
  have t3 : prog[(snsteps prog 40 ss).pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")) := by
    rw [hpc40]; exact h43
  have t4 : prog[(snsteps prog 40 ss).pc + 4]? = some (.andp "pIns" "pLe" "pValid") := by
    rw [hpc40]; exact h44
  have t5 : prog[(snsteps prog 40 ss).pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1)) := by
    rw [hpc40]; exact h45
  have t6 : prog[(snsteps prog 40 ss).pc + 6]? = some (.stshp "pIns" "addr" "pp1") := by
    rw [hpc40]; exact h46
  have t7 : prog[(snsteps prog 40 ss).pc + 7]? = some (.barwarp) := by rw [hpc40]; exact h47
  have t8 : prog[(snsteps prog 40 ss).pc + 8]? = some (.binr .add "p0" "searchPos" "fl") := by
    rw [hpc40]; exact h48
  have t9 : prog[(snsteps prog 40 ss).pc + 9]? = some (.shfl "cand0" "cand" "fl") := by
    rw [hpc40]; exact h49
  have t10 : prog[(snsteps prog 40 ss).pc + 10]? = some (.setp .ne "found" "bal" (.imm 0)) := by
    rw [hpc40]; exact h50
  have hpc51 : (snsteps prog 51 ss).pc = ss.pc + 51 := by
    have := cwPc11 prog (snsteps prog 40 ss) t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10
    rw [show (51 : Nat) = 40 + 11 from rfl, snsteps_add, this, hpc40]
  -- The 51-instr frame: every non-scratch register + gmem preserved.
  have hframe : ∀ r : String,
      r ∉ ["posP", "pValid", "cap4", "rp", "rpA", "b0", "b1", "b2", "b3", "v32",
           "hh", "addr", "candRaw", "cand", "rc", "rcA", "c0", "c1", "c2", "c3", "vc",
           "pNE", "pCO", "pEq", "pH1", "pH2", "pHit", "bal", "rev", "fl", "pLe",
           "pIns", "pp1", "p0", "cand0", "found"] →
      (snsteps prog 51 ss).regs r = ss.regs r ∧ (snsteps prog 51 ss).gmem = ss.gmem :=
    fun r hr => coopWindow_frame prog ss ss.pc r searchLim capC (32 - hashLog) (2 ^ hashLog - 1)
      rfl h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24
      h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39 h40 h41 h42 h43 h44 h45 h46 h47
      h48 h49 h50 hr
  have hgmem : (snsteps prog 51 ss).gmem = ss.gmem := (hframe "X" (by decide)).2
  refine ⟨sreaches_snsteps prog 51 ss, hpc51, ?_, ?_, ?_⟩
  · -- gmem preserved: evalCoopWindow leaves gmem unchanged
    rw [hgmem, hc.gmem, evalCoopWindow_eq_go, evalCoopWindowGo_gmem]
  · -- smem = tableInsert (via stshp = tableInsert); extracted to `coopWindow_couple_smem`
    -- to keep this proof term small enough for the kernel typechecker.
    exact coopWindow_couple_smem_relaxed prog ss ws s inStride searchLim capC hashLog ib
      h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24
      h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39 h40 h41 h42 h43 h44 h45 h46
      h47 h48 h49 h50 hc.smem hc.gmem hsl hcapdef hcapb hhl hInb0 htbl0 hlane hsp hsval
      hwsib hp64 hpc40 t0 t1 t2 t3 t4 t5 hib64
  · -- per-register: found / p0 / cand0 match window result; others preserved
    -- The model oracle and window (shared, since gmem/smem agree via couple).
    have hsmem := hc.smem
    have hgm := hc.gmem
    -- ballot bit k = probe (s+k), per lane, from `coopWindow_pHit_iff`.
    have hballot : ∀ k, k < 32 →
        (ballotOf (snsteps prog 40 ss).regs "pHit").toNat.testBit k
          = (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
              (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim (s + k)).isSome := by
      intro k hk
      rw [AlgorithmLib.LZ4SimtBits.ballotOf_testBit (snsteps prog 40 ss).regs "pHit" ⟨k, hk⟩]
      have hiff := coopWindow_pHit_iff_relaxed prog ss ss.pc s inStride searchLim capC hashLog 0 ib
        ⟨k, hk⟩ rfl ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17,
          h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35,
          h36, h37, h38, h39⟩
        (hInb0 ⟨k, hk⟩) (by rw [htbl0 ⟨k, hk⟩]; rfl) hsl hcapdef hcapb hhl (by decide)
        (hlane ⟨k, hk⟩) (hsp ⟨k, hk⟩) (by simp only []; omega) hib64
      by_cases hp : ((snsteps prog 40 ss).regs "pHit" ⟨k, hk⟩) = 1
      · rw [beq_iff_eq.mpr hp]
        exact (hiff.mp hp).symm ▸ rfl
      · rw [beq_eq_false_iff_ne.mpr hp]
        rcases hprobe : (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
            (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim (s + k)).isSome with _ | _
        · rfl
        · exact absurd (hiff.mpr (by rw [hprobe])) hp
    -- `clz∘brev∘ballot(pHit@40) = firstHit(probe) 32 0` (the vote/brev/clz reduction).
    have hclz : (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat
        = (AlgorithmLib.LZ4WarpSched.firstHit
            (fun k => (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
              (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim (s + k)).isSome) 32 0).getD 32 := by
      rw [AlgorithmLib.LZ4SimtBits.collective_select_firstHit (snsteps prog 40 ss).regs "pHit",
          firstHit_congr _ _ 32 0 (fun k _ hk => hballot k hk)]
    -- The model window on the `ws` side = the `ss` side (couple: gmem/smem/searchPos agree).
    have hwsibN : (ws.regs "inBase").toNat = ib := by
      rw [hwsib]; exact AlgorithmLib.LZ4Ptx.toNat_ofNat_lt ib (by omega)
    have hwsWin : AlgorithmLib.LZ4WarpFind.window
          (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (tableOracle ws.gmem ws.smem hashLog 0 (ws.regs "inBase").toNat)
          searchLim (ws.regs "searchPos").toNat
        = AlgorithmLib.LZ4WarpFind.window (gmemInpAt ss.gmem ib inStride)
          (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s := by
      rw [hc.gmem, hc.smem, hsval, hwsibN]
    -- The tail-lemma instruction facts, anchored at `(snsteps prog 40 ss).pc = ss.pc + 40`.
    -- (already have t0..t10). Now match found / p0 / cand0, else preserve via hframe.
    intro r hr l
    rw [evalCoopWindow_eq_go]
    -- register value from `evalCoopWindowGo_regs`, with the ws-window rewritten to the ss-window.
    rw [evalCoopWindowGo_regs, hwsWin]
    -- machine tail-values (all in terms of `clz∘brev∘ballot(pHit@40)` via cwFound/cwP0/cwCand0).
    have hfound : (snsteps prog 51 ss).regs "found" l
        = (if ballotOf (snsteps prog 40 ss).regs "pHit" == 0 then 0 else 1) := by
      have := cwFound prog (snsteps prog 40 ss) l t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10
      rwa [show (51 : Nat) = 40 + 11 from rfl, snsteps_add]
    have hp0 : (snsteps prog 51 ss).regs "p0" l
        = (snsteps prog 40 ss).regs "searchPos" l + clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit")) := by
      have := cwP0 prog (snsteps prog 40 ss) l t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10
      rwa [show (51 : Nat) = 40 + 11 from rfl, snsteps_add]
    have hcand0 : (snsteps prog 51 ss).regs "cand0" l
        = (snsteps prog 40 ss).regs "cand" (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit")))) := by
      have := cwCand0 prog (snsteps prog 40 ss) l t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10
      rwa [show (51 : Nat) = 40 + 11 from rfl, snsteps_add]
    -- Case on the register name.
    by_cases hrf : r = "found"
    · subst hrf; rw [hfound]
      -- `window.isSome ↔ ballot(pHit@40) ≠ 0`.
      have hisSome : (AlgorithmLib.LZ4WarpFind.window (gmemInpAt ss.gmem ib inStride)
          (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s).isSome = true
          ↔ ballotOf (snsteps prog 40 ss).regs "pHit" ≠ 0 := by
        rw [← AlgorithmLib.LZ4WarpSched.coopWindow_eq_window, coopWindow_isSome, firstHit_isSome]
        constructor
        · rintro ⟨k, hk1, hk2, hk3⟩ hbz
          have hbk := hballot k (by omega)
          rw [hk3] at hbk
          rw [hbz] at hbk
          simp at hbk
        · intro hne
          have hex : ∃ l : Fin 32, (snsteps prog 40 ss).regs "pHit" l == 1 := by
            apply Classical.byContradiction
            intro hcon
            simp only [not_exists] at hcon
            exact hne (beq_iff_eq.mp
              ((ballot_eq_zero_iff (snsteps prog 40 ss).regs "pHit").mpr (fun l =>
                Bool.eq_false_iff.mpr (hcon l))))
          obtain ⟨l, hl⟩ := hex
          refine ⟨l.val, Nat.zero_le _, by omega, ?_⟩
          rw [← hballot l.val l.isLt, ballotOf_testBit]
          exact hl
      -- assemble the `evalCoopWindowGo_regs` match on `found`.
      rcases hSome : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ss.gmem ib inStride)
          (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s with _ | pc
      · have hb0 : ballotOf (snsteps prog 40 ss).regs "pHit" = 0 :=
          Classical.byContradiction (fun hne => absurd (hisSome.mpr hne) (by rw [hSome]; simp))
        simp only [WState.setReg, hb0]
        simp
      · obtain ⟨p, c⟩ := pc
        have hbne : ballotOf (snsteps prog 40 ss).regs "pHit" ≠ 0 :=
          hisSome.mp (by rw [hSome]; simp)
        simp only [WState.setReg, hbne]
        simp [hbne]
    · -- r ≠ found; and p0/cand0 ∉ R so r ≠ p0/cand0; use hframe for other regs.
      have hrp0 : r ≠ "p0" := fun h => hp0R (h ▸ hr)
      have hrc0 : r ≠ "cand0" := fun h => hcand0R (h ▸ hr)
      have hrint := hRdisj r hr
      -- r is outside the FULL scratch set, so `hframe` preserves it.
      have hrfull : r ∉ ["posP", "pValid", "cap4", "rp", "rpA", "b0", "b1", "b2", "b3", "v32",
          "hh", "addr", "candRaw", "cand", "rc", "rcA", "c0", "c1", "c2", "c3", "vc",
          "pNE", "pCO", "pEq", "pH1", "pH2", "pHit", "bal", "rev", "fl", "pLe",
          "pIns", "pp1", "p0", "cand0", "found"] := by
        intro hmem
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
        rcases hmem with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h
        all_goals first
          | exact hrp0 h | exact hrc0 h | exact hrf h
          | exact hrint (by simp [h])
      rw [(hframe r hrfull).1, hc.reg hr l]
      -- model side: r ∉ {found,p0,cand0} leaves it unchanged in both window branches.
      cases AlgorithmLib.LZ4WarpFind.window (gmemInpAt ss.gmem ib inStride)
          (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s with
      | none => simp [WState.setReg, hrf]
      | some pc => obtain ⟨p, c⟩ := pc; simp [WState.setReg, hrf, hrp0, hrc0]

/-- **The found match position couples**: after `coopWindow`, when the model `window`
    returns `some (p, c)`, the machine's `p0 = searchPos + fl` equals `ofNat p`
    (`fl = clz∘brev∘ballot(pHit)` is the earliest hitting lane, and `p = s + fl` by the
    `firstHit`/`window` correspondence).  This is the coupling `coopWindow_couple` omits
    (it excludes `p0`), needed to feed the found-branch. -/
theorem coopWindow_p0_found (prog : Array SInstr) (ss : SState)
    (base s inStride searchLim capC hashLog p c ib : Nat)
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
    (h40 : prog[base+40]? = some (.vote "bal" "pHit"))
    (h41 : prog[base+41]? = some (.brev "rev" "bal"))
    (h42 : prog[base+42]? = some (.clz "fl" "rev"))
    (h43 : prog[base+43]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h44 : prog[base+44]? = some (.andp "pIns" "pLe" "pValid"))
    (h45 : prog[base+45]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h46 : prog[base+46]? = some (.stshp "pIns" "addr" "pp1"))
    (h47 : prog[base+47]? = some (.barwarp))
    (h48 : prog[base+48]? = some (.binr .add "p0" "searchPos" "fl"))
    (h49 : prog[base+49]? = some (.shfl "cand0" "cand" "fl"))
    (h50 : prog[base+50]? = some (.setp .ne "found" "bal" (.imm 0)))
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (hInb0 : ∀ l : Fin 32, ss.regs "inBase" l = UInt64.ofNat ib)
    (hlen : inStride < 2 ^ 40)
    (htbl0 : ∀ l : Fin 32, ss.regs "tbl" l = 0)
    (hlane : ∀ l : Fin 32, ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ∀ l : Fin 32, ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + 32 < 2 ^ 64)
    (hwin : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ss.gmem ib inStride)
        (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s = some (p, c))
    (hib64 : ib + capC < 2 ^ 64) :
    ∀ l : Fin 32, (snsteps prog 51 ss).regs "p0" l = UInt64.ofNat p := by
  subst hpc
  have hpc40 : (snsteps prog 40 ss).pc = ss.pc + 40 :=
    coopWindow_pc40 prog ss ss.pc searchLim capC hashLog rfl h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12
      h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24 h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35
      h36 h37 h38 h39
  have t0 : prog[(snsteps prog 40 ss).pc]? = some (.vote "bal" "pHit") := by rw [hpc40]; exact h40
  have t1 : prog[(snsteps prog 40 ss).pc + 1]? = some (.brev "rev" "bal") := by rw [hpc40]; exact h41
  have t2 : prog[(snsteps prog 40 ss).pc + 2]? = some (.clz "fl" "rev") := by rw [hpc40]; exact h42
  have t3 : prog[(snsteps prog 40 ss).pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")) := by
    rw [hpc40]; exact h43
  have t4 : prog[(snsteps prog 40 ss).pc + 4]? = some (.andp "pIns" "pLe" "pValid") := by
    rw [hpc40]; exact h44
  have t5 : prog[(snsteps prog 40 ss).pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1)) := by
    rw [hpc40]; exact h45
  have t6 : prog[(snsteps prog 40 ss).pc + 6]? = some (.stshp "pIns" "addr" "pp1") := by
    rw [hpc40]; exact h46
  have t7 : prog[(snsteps prog 40 ss).pc + 7]? = some (.barwarp) := by rw [hpc40]; exact h47
  have t8 : prog[(snsteps prog 40 ss).pc + 8]? = some (.binr .add "p0" "searchPos" "fl") := by
    rw [hpc40]; exact h48
  have t9 : prog[(snsteps prog 40 ss).pc + 9]? = some (.shfl "cand0" "cand" "fl") := by
    rw [hpc40]; exact h49
  have t10 : prog[(snsteps prog 40 ss).pc + 10]? = some (.setp .ne "found" "bal" (.imm 0)) := by
    rw [hpc40]; exact h50
  have hballot : ∀ k, k < 32 →
      (ballotOf (snsteps prog 40 ss).regs "pHit").toNat.testBit k
        = (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
            (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim (s + k)).isSome := by
    intro k hk
    rw [AlgorithmLib.LZ4SimtBits.ballotOf_testBit (snsteps prog 40 ss).regs "pHit" ⟨k, hk⟩]
    have hiff := coopWindow_pHit_iff_relaxed prog ss ss.pc s inStride searchLim capC hashLog 0 ib
      ⟨k, hk⟩ rfl ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17,
        h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35,
        h36, h37, h38, h39⟩
      (hInb0 ⟨k, hk⟩) (by rw [htbl0 ⟨k, hk⟩]; rfl) hsl hcapdef hcapb hhl (by decide)
      (hlane ⟨k, hk⟩) (hsp ⟨k, hk⟩) (by simp only []; omega) hib64
    by_cases hp : ((snsteps prog 40 ss).regs "pHit" ⟨k, hk⟩) = 1
    · rw [beq_iff_eq.mpr hp]; exact (hiff.mp hp).symm ▸ rfl
    · rw [beq_eq_false_iff_ne.mpr hp]
      rcases hprobe : (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
          (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim (s + k)).isSome with _ | _
      · rfl
      · exact absurd (hiff.mpr (by rw [hprobe])) hp
  have hclz : (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat
      = (AlgorithmLib.LZ4WarpSched.firstHit
          (fun k => (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
            (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim (s + k)).isSome) 32 0).getD 32 := by
    rw [AlgorithmLib.LZ4SimtBits.collective_select_firstHit (snsteps prog 40 ss).regs "pHit",
        firstHit_congr _ _ 32 0 (fun k _ hk => hballot k hk)]
  have hclzval : (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat = p - s := by
    rw [hclz, firstHit_getD_eq_window (gmemInpAt ss.gmem ib inStride)
      (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s, hwin]
  obtain ⟨hsp_le, hp_lt, _, _⟩ := AlgorithmLib.LZ4WarpFind.window_sound
    (gmemInpAt ss.gmem ib inStride) (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s p c hwin
  have hp2p64 : p < 2 ^ 64 := by
    have h4064 : (2 : Nat) ^ 40 ≤ 2 ^ 64 := Nat.pow_le_pow_right (by omega) (by omega)
    omega
  have hframe51 := coopWindow_frame prog ss ss.pc "searchPos" searchLim capC (32 - hashLog)
    (2 ^ hashLog - 1) rfl h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20
    h21 h22 h23 h24 h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39 h40 h41 h42 h43 h44
    h45 h46 h47 h48 h49 h50 (by decide)
  intro l
  have ht := cwTail_searchPos prog (snsteps prog 40 ss) l t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10
  have h51add : snsteps prog 51 ss = snsteps prog 11 (snsteps prog 40 ss) := by
    rw [show (51 : Nat) = 40 + 11 from rfl, snsteps_add]
  have hsp40 : (snsteps prog 40 ss).regs "searchPos" l = UInt64.ofNat s := by
    rw [← ht, ← h51add, congrFun hframe51.1 l, hsp l]
  have hp0 : (snsteps prog 51 ss).regs "p0" l
      = (snsteps prog 40 ss).regs "searchPos" l
        + clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit")) := by
    have := cwP0 prog (snsteps prog 40 ss) l t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10
    rwa [show (51 : Nat) = 40 + 11 from rfl, snsteps_add]
  have hclzeq : clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))
      = UInt64.ofNat (p - s) := by rw [← hclzval, UInt64.ofNat_toNat]
  rw [hp0, hsp40, hclzeq, ← UInt64.toNat_inj, u64_add_ofNat s (p - s) (by omega),
    toNat_ofNat_lt p hp2p64]
  omega

/-- **The found candidate couples**: after `coopWindow`, when `window = some (p, c)`,
    the machine's `cand0 = cand@lane(fl)` equals `ofNat c` (the winning lane's oracle
    candidate).  Companion to `coopWindow_p0_found`. -/
theorem coopWindow_cand0_found (prog : Array SInstr) (ss : SState)
    (base s inStride searchLim capC hashLog p c ib : Nat)
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
    (h40 : prog[base+40]? = some (.vote "bal" "pHit"))
    (h41 : prog[base+41]? = some (.brev "rev" "bal"))
    (h42 : prog[base+42]? = some (.clz "fl" "rev"))
    (h43 : prog[base+43]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h44 : prog[base+44]? = some (.andp "pIns" "pLe" "pValid"))
    (h45 : prog[base+45]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h46 : prog[base+46]? = some (.stshp "pIns" "addr" "pp1"))
    (h47 : prog[base+47]? = some (.barwarp))
    (h48 : prog[base+48]? = some (.binr .add "p0" "searchPos" "fl"))
    (h49 : prog[base+49]? = some (.shfl "cand0" "cand" "fl"))
    (h50 : prog[base+50]? = some (.setp .ne "found" "bal" (.imm 0)))
    (hsl : searchLim ≤ capC) (hcapdef : capC = inStride - 4) (hcapb : capC < 2 ^ 64)
    (hhl : hashLog ≤ 32) (hInb0 : ∀ l : Fin 32, ss.regs "inBase" l = UInt64.ofNat ib)
    (hlen : inStride < 2 ^ 40)
    (htbl0 : ∀ l : Fin 32, ss.regs "tbl" l = 0)
    (hlane : ∀ l : Fin 32, ss.regs "lane" l = UInt64.ofNat l.val)
    (hsp : ∀ l : Fin 32, ss.regs "searchPos" l = UInt64.ofNat s)
    (hp64 : s + 32 < 2 ^ 64)
    (hwin : AlgorithmLib.LZ4WarpFind.window (gmemInpAt ss.gmem ib inStride)
        (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s = some (p, c))
    (hib64 : ib + capC < 2 ^ 64) :
    ∀ l : Fin 32, (snsteps prog 51 ss).regs "cand0" l = UInt64.ofNat c := by
  subst hpc
  obtain ⟨hsp_le, hp_lt, hc_lt, _⟩ := AlgorithmLib.LZ4WarpFind.window_sound
    (gmemInpAt ss.gmem ib inStride) (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s p c hwin
  have hps32 : p - s < 32 := by
    have := AlgorithmLib.LZ4WarpFind.window_lt (gmemInpAt ss.gmem ib inStride)
      (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s p c hwin
    omega
  have hp2p64 : p < 2 ^ 64 := by
    have h4064 : (2 : Nat) ^ 40 ≤ 2 ^ 64 := Nat.pow_le_pow_right (by omega) (by omega)
    omega
  have horc : tableOracle ss.gmem ss.smem hashLog 0 ib p = some c :=
    AlgorithmLib.LZ4WarpFind.window_oracle (gmemInpAt ss.gmem ib inStride)
      (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s p c hwin
  have hpc40 : (snsteps prog 40 ss).pc = ss.pc + 40 :=
    coopWindow_pc40 prog ss ss.pc searchLim capC hashLog rfl h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12
      h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24 h25 h26 h27 h28 h29 h30 h31 h32 h33 h34 h35
      h36 h37 h38 h39
  have t0 : prog[(snsteps prog 40 ss).pc]? = some (.vote "bal" "pHit") := by rw [hpc40]; exact h40
  have t1 : prog[(snsteps prog 40 ss).pc + 1]? = some (.brev "rev" "bal") := by rw [hpc40]; exact h41
  have t2 : prog[(snsteps prog 40 ss).pc + 2]? = some (.clz "fl" "rev") := by rw [hpc40]; exact h42
  have t3 : prog[(snsteps prog 40 ss).pc + 3]? = some (.setp .le "pLe" "lane" (.reg "fl")) := by
    rw [hpc40]; exact h43
  have t4 : prog[(snsteps prog 40 ss).pc + 4]? = some (.andp "pIns" "pLe" "pValid") := by
    rw [hpc40]; exact h44
  have t5 : prog[(snsteps prog 40 ss).pc + 5]? = some (.bin .add "pp1" "posP" (.imm 1)) := by
    rw [hpc40]; exact h45
  have t6 : prog[(snsteps prog 40 ss).pc + 6]? = some (.stshp "pIns" "addr" "pp1") := by
    rw [hpc40]; exact h46
  have t7 : prog[(snsteps prog 40 ss).pc + 7]? = some (.barwarp) := by rw [hpc40]; exact h47
  have t8 : prog[(snsteps prog 40 ss).pc + 8]? = some (.binr .add "p0" "searchPos" "fl") := by
    rw [hpc40]; exact h48
  have t9 : prog[(snsteps prog 40 ss).pc + 9]? = some (.shfl "cand0" "cand" "fl") := by
    rw [hpc40]; exact h49
  have t10 : prog[(snsteps prog 40 ss).pc + 10]? = some (.setp .ne "found" "bal" (.imm 0)) := by
    rw [hpc40]; exact h50
  have hballot : ∀ k, k < 32 →
      (ballotOf (snsteps prog 40 ss).regs "pHit").toNat.testBit k
        = (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
            (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim (s + k)).isSome := by
    intro k hk
    rw [AlgorithmLib.LZ4SimtBits.ballotOf_testBit (snsteps prog 40 ss).regs "pHit" ⟨k, hk⟩]
    have hiff := coopWindow_pHit_iff_relaxed prog ss ss.pc s inStride searchLim capC hashLog 0 ib
      ⟨k, hk⟩ rfl ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17,
        h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35,
        h36, h37, h38, h39⟩
      (hInb0 ⟨k, hk⟩) (by rw [htbl0 ⟨k, hk⟩]; rfl) hsl hcapdef hcapb hhl (by decide)
      (hlane ⟨k, hk⟩) (hsp ⟨k, hk⟩) (by simp only []; omega) hib64
    by_cases hp : ((snsteps prog 40 ss).regs "pHit" ⟨k, hk⟩) = 1
    · rw [beq_iff_eq.mpr hp]; exact (hiff.mp hp).symm ▸ rfl
    · rw [beq_eq_false_iff_ne.mpr hp]
      rcases hprobe : (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ss.gmem ib inStride)
          (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim (s + k)).isSome with _ | _
      · rfl
      · exact absurd (hiff.mpr (by rw [hprobe])) hp
  have hclzval : (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat = p - s := by
    rw [AlgorithmLib.LZ4SimtBits.collective_select_firstHit (snsteps prog 40 ss).regs "pHit",
        firstHit_congr _ _ 32 0 (fun k _ hk => hballot k hk),
        firstHit_getD_eq_window (gmemInpAt ss.gmem ib inStride)
          (tableOracle ss.gmem ss.smem hashLog 0 ib) searchLim s, hwin]
  intro l
  have hLval : (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit")))).val = p - s := by
    show (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))).toNat % W = p - s
    rw [hclzval]
    have hWeq : W = 32 := rfl
    rw [hWeq]; exact Nat.mod_eq_of_lt hps32
  obtain ⟨hc1, hc2, hc3⟩ := coopWindow_cand34_val prog ss ss.pc ib s inStride searchLim capC hashLog 0
    (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit")))) rfl
    h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19 h20 h21 h22 h23 h24 h25 h26
    h27 h28 h29 h30 h31 h32 h33 h34 h35 h36 h37 h38 h39
    (hInb0 _) (by rw [htbl0 _]; rfl) hsl hcapdef hcapb hhl (by decide) (by rw [hLval]; omega)
    (hlane _) (hsp _) (by rw [hLval]; omega) hib64
  rw [hLval, show s + (p - s) = p from by omega] at hc2
  rw [horc] at hc2
  have hraw0 : ((snsteps prog 21 ss).regs "candRaw"
      (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))))).toNat ≠ 0 := by
    intro h0'; rw [if_pos h0'] at hc2; exact absurd hc2 (by simp)
  have hcval : ((snsteps prog 21 ss).regs "candRaw"
      (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))))).toNat - 1 = c := by
    rw [if_neg hraw0] at hc2; injection hc2 with hc2'; omega
  have hcand0 : (snsteps prog 51 ss).regs "cand0" l
      = (snsteps prog 40 ss).regs "cand"
          (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit")))) := by
    have := cwCand0 prog (snsteps prog 40 ss) l t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10
    rwa [show (51 : Nat) = 40 + 11 from rfl, snsteps_add]
  rw [hcand0, hc1]
  have hsub : ((snsteps prog 21 ss).regs "candRaw"
      (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit")))) - UInt64.ofNat 1).toNat
      = c := by
    have h1le : 1 ≤ ((snsteps prog 21 ss).regs "candRaw"
        (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))))).toNat :=
      Nat.one_le_iff_ne_zero.mpr hraw0
    have hRofNat : UInt64.ofNat ((snsteps prog 21 ss).regs "candRaw"
        (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))))).toNat
        = (snsteps prog 21 ss).regs "candRaw"
          (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit")))) := UInt64.ofNat_toNat
    have := AlgorithmLib.LZ4Ptx.u64_sub_ofNat ((snsteps prog 21 ss).regs "candRaw"
      (toLane (clz32 (brev32 (ballotOf (snsteps prog 40 ss).regs "pHit"))))).toNat 1 (by omega) hc3
    rw [hRofNat] at this
    rw [this, hcval]
  rw [← hsub, UInt64.ofNat_toNat]

end AlgorithmLib.LZ4WarpDSL
