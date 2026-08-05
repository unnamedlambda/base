import AlgorithmLib.LZ4Prologue
namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib AlgorithmLib.LZ4 AlgorithmLib.LZ4Simt AlgorithmLib.LZ4Plan AlgorithmLib.LZ4Imp
open AlgorithmLib.LZ4Ptx (toNat_ofNat_lt)

theorem encSeqLen_le9 (s : PlanStep) (h : 4 ≤ s.mlen) : encSeqLen s ≤ 8 * (s.litLen + s.mlen) := by
  have h1 := encNib_len_le s.litLen
  have h2 := encNib_len_le (s.mlen - 4)
  simp only [encSeqLen]; omega

theorem encFinalLen_le9 (fl : Nat) : encFinalLen fl ≤ 8 * fl + 1 := by
  have := encNib_len_le fl; simp only [encFinalLen]; omega

theorem encodeSum_le9 (inp : List UInt8) : ∀ (anchor : Nat) (steps : List PlanStep) (fl : Nat),
    ValidStepsFrom inp anchor steps fl → (steps.map encSeqLen).sum ≤ 8 * stepsLen steps
  | _, [], _, _ => by simp [stepsLen]
  | anchor, st :: rest, fl, hv => by
    obtain ⟨hag, hrest⟩ := hv
    have ih := encodeSum_le9 inp (anchor + st.litLen + st.mlen) rest fl hrest
    have hs := encSeqLen_le9 st hag.2.2.1
    simp only [List.map_cons, List.sum_cons, stepsLen]; omega

/-- **The generous 9x output budget, from an ARBITRARY literal anchor.**

    `planBlock_encode_le9` below is this at `anchor = 0`.  The suffix form is what
    the emit loop needs: its invariant is indexed by the literal anchor, and the
    quantity it must bound is what is still to be emitted from there.

    Every ingredient was already anchor-parametric (`encode_planBlockFrom_length`,
    `encodeSum_le9`, `ValidStepsFrom_sum`); only the instance at `0` had been
    stated.  This is the no-overflow side condition for showing the output cursor
    never runs backwards. -/
theorem planBlockFrom_encode_le9 (inp : List UInt8) (anchor : Nat) (steps : List PlanStep)
    (fl : Nat) (hv : ValidStepsFrom inp anchor steps fl) (hpos : anchor < inp.length) :
    (planBlockFrom inp anchor steps fl).encode.length ≤ 9 * (inp.length - anchor) := by
  have hlen := encode_planBlockFrom_length inp anchor steps fl hv
  have hsum := encodeSum_le9 inp anchor steps fl hv
  have hf := encFinalLen_le9 fl
  have htot := ValidStepsFrom_sum inp anchor steps fl hv
  rw [hlen]; omega

theorem planBlock_encode_le9 (inp : List UInt8) (p : Plan) (hv : ValidPlan inp p)
    (hpos : 1 ≤ inp.length) : (planToBlock inp p).encode.length ≤ 9 * inp.length := by
  have hlen := encode_planBlockFrom_length inp 0 p.steps p.finalLen hv
  have hsum := encodeSum_le9 inp 0 p.steps p.finalLen hv
  have hf := encFinalLen_le9 p.finalLen
  have htot := ValidStepsFrom_sum inp 0 p.steps p.finalLen hv
  simp only [planToBlock]; rw [hlen]; omega

section WarpLaunch

variable (nb iS oS lO hL w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)

/-- Every encoded block is non-empty: the final literal run emits its token byte. -/
theorem planBlock_encode_pos (inp : List UInt8) (p : Plan) (hv : ValidPlan inp p) :
    0 < (planToBlock inp p).encode.length := by
  have hlen := encode_planBlockFrom_length inp 0 p.steps p.finalLen hv
  have hf : 1 ≤ encFinalLen p.finalLen := by simp only [encFinalLen]; omega
  simp only [planToBlock]; rw [hlen]; omega

/-- Roundtrip to the `sstep` floor, for every warp `w < nb`: warp `w` reads its
    block at `w*iS`, writes at `iT + w*oS`, and that window decodes back to it. -/
theorem warpKernelDSL_prologue_roundtrips
    (hnb : 0 < nb) (hnb2 : nb < 2 ^ 64) (hw : w < nb) (hHash : hL ≤ 32)
    (hstride : iS ≤ 65536) (hipos : 12 ≤ iS)
    (hw64 : w * 32 + 32 < 2 ^ 64) (hib40 : inPtr + w * iS < 2 ^ 40)
    (htop : outPtr + w * oS + 9 * iS < 2 ^ 32)
    (hbuf : outPtr + w * oS + 9 * iS ≤ gm.size)
    (hderive : outPtr = inPtr + (nb * iS + copySlack)) (hdisj : inPtr + w * iS + iS ≤ outPtr + w * oS) :
    ∃ (n : Nat) (ss' : SState) (k : Nat),
      SReaches (warpKernelDSL nb iS oS lO hL) n (initSt w inPtr outPtr gm smemB) ss' ∧
      decompress ((List.range k).map (fun i => ss'.gmem.getD (outPtr + w * oS + i) 0)) iS
        = some (gmemInpAt gm (inPtr + w * iS) iS) := by
  obtain ⟨hpc39, hMI, hCouple, hop0, hla0, hsp0, hinB0, houtB0, hgmem⟩ :=
    prologue_couple nb iS oS lO hL w inPtr outPtr gm smemB hnb hnb2 hw hw64 hderive hHash
  obtain ⟨S0, hS0⟩ : ∃ x, snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1)
      (initSt w inPtr outPtr gm smemB) = x := ⟨_, rfl⟩
  rw [hS0] at hpc39 hMI hCouple hop0 hla0 hsp0 hinB0 houtB0 hgmem
  have hoT : (S0.regs "outBase" 0).toNat = outPtr + w * oS := by
    rw [houtB0, toNat_ofNat_lt _ (by omega)]
  have hiB : (S0.regs "inBase" 0).toNat = inPtr + w * iS := by
    rw [hinB0, toNat_ofNat_lt _ (by omega)]
  have hgS : S0.gmem.size = gm.size := by rw [hgmem]
  have hgi : gmemInpAt S0.gmem (inPtr + w * iS) iS = gmemInpAt gm (inPtr + w * iS) iS := by
    rw [hgmem]
  have hEnc : (planToBlock (gmemInpAt S0.gmem (inPtr + w * iS) iS)
      (evalPlan S0.gmem S0.smem (inPtr + w * iS) iS hL)).encode.length ≤ 9 * iS := by
    have hv := evalPlan_valid S0.gmem S0.smem (inPtr + w * iS) iS hL hstride
    have hb := planBlock_encode_le9 _ _ hv
      (by simp only [gmemInpAt, List.length_map, List.length_range]; omega)
    simpa only [gmemInpAt, List.length_map, List.length_range] using hb
  obtain ⟨m, ss', hr2, hdec, hcpl, hmiF, hobPres, hopVal, hgsz, hpcF⟩ :=
    warpKernelDSL_sstep_roundtrips_discharged iS hL
    "Lh0" "Lx1" "Le2" "Ln3" "Lh4" "Lx5" "Le6" "Ln7" "Lh8" "Lx9" "Ch10" "Cx11" "Le12" "Ln13" "Lh14" "Lx15"
    "Le16" "Ln17" "Lh18" "Lx19" "Ch20" "Cx21"
    (myLsic "litExtra") (myLsic "matExtra") (myLsic "litExtraF")
    rfl rfl rfl hstride hipos (by omega) hib40 (by omega) hHash
    { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }
    hop0 hla0 hsp0
    (by show (S0.regs "inBase" 0).toNat < 2 ^ 40; rw [hiB]; exact hib40)
    (by show (S0.regs "outBase" 0).toNat + 9 * iS < 2 ^ 32; rw [hoT]; omega)
    (by show (S0.regs "outBase" 0).toNat + 9 * iS ≤ S0.gmem.size; rw [hoT, hgS]; omega)
    (by show (S0.regs "inBase" 0).toNat + iS ≤ (S0.regs "outBase" 0).toNat; rw [hiB, hoT]; omega)
    (by show (S0.regs "inBase" 0).toNat + iS ≤ (S0.regs "outBase" 0).toNat; rw [hiB, hoT]; omega)
    (by show (S0.regs "inBase" 0).toNat + iS < 2 ^ 64; rw [hiB]; omega)
    (by show (planToBlock (gmemInpAt S0.gmem (S0.regs "inBase" 0).toNat iS)
          (evalPlan S0.gmem S0.smem (S0.regs "inBase" 0).toNat iS hL)).encode.length ≤ 9 * iS
        rw [hiB]; exact hEnc)
    (warpKernelDSL nb iS oS lO hL) 39 S0 hpc39
    (body_segAt nb iS oS lO hL) (body_labelsResolve nb iS oS lO hL) hCouple hMI
  have hreach0 : SReaches (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1)
      (initSt w inPtr outPtr gm smemB) S0 := by
    rw [← hS0]
    exact sreaches_snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1)
      (initSt w inPtr outPtr gm smemB)
  dsimp only at hdec
  rw [hiB, hgi, hoT, show (gmemInpAt gm (inPtr + w * iS) iS).length = iS from by
    simp [gmemInpAt]] at hdec
  exact ⟨_, ss', _, sreaches_trans _ _ _ _ _ _ hreach0 hr2, hdec⟩

/-- As `warpKernelDSL_prologue_roundtrips` but through the length-store tail to
    `pc = 272`: the tail writes only at `outBase+lO … +3`, which the tight bound
    places beyond the window, so the window still decodes. -/
theorem warpKernelDSL_tail_roundtrips
    (hnb : 0 < nb) (hnb2 : nb < 2 ^ 64) (hw : w < nb) (hHash : hL ≤ 32)
    (hstride : iS ≤ 65536) (hipos : 12 ≤ iS)
    (hw64 : w * 32 + 32 < 2 ^ 64) (hib40 : inPtr + w * iS < 2 ^ 40)
    (htop : outPtr + w * oS + 9 * iS < 2 ^ 32)
    (hbuf : outPtr + w * oS + 9 * iS ≤ gm.size)
    (hderive : outPtr = inPtr + (nb * iS + copySlack)) (hdisj : inPtr + w * iS + iS ≤ outPtr + w * oS)
    (hlO : iS + iS / 16 + 256 ≤ lO) (hlOtop : outPtr + w * oS + lO + 3 < 2 ^ 64)
    (hlOfit : outPtr + w * oS + lO + 4 ≤ gm.size) :
    ∃ (n : Nat) (ss' : SState) (k : Nat),
      SReaches (warpKernelDSL nb iS oS lO hL) n (initSt w inPtr outPtr gm smemB) ss' ∧
      ss'.pc = 272 ∧ 0 < k ∧ k ≤ lO ∧
      AlgorithmLib.readU32LE ss'.gmem (outPtr + w * oS + lO) = k ∧
      decompress ((List.range k).map (fun i => ss'.gmem.getD (outPtr + w * oS + i) 0)) iS
        = some (gmemInpAt gm (inPtr + w * iS) iS) ∧
      -- write confinement: warp `w` touches ONLY its own output stride, so the
      -- per-warp results compose across the launch.
      (∀ j, j < outPtr + w * oS ∨ outPtr + w * oS + lO + 4 ≤ j →
        ss'.gmem.getD j 0 = gm.getD j 0) := by
  obtain ⟨hpc39, hMI, hCouple, hop0, hla0, hsp0, hinB0, houtB0, hgmem⟩ :=
    prologue_couple nb iS oS lO hL w inPtr outPtr gm smemB hnb hnb2 hw hw64 hderive hHash
  obtain ⟨S0, hS0⟩ : ∃ x, snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1)
      (initSt w inPtr outPtr gm smemB) = x := ⟨_, rfl⟩
  rw [hS0] at hpc39 hMI hCouple hop0 hla0 hsp0 hinB0 houtB0 hgmem
  have hoT : (S0.regs "outBase" 0).toNat = outPtr + w * oS := by
    rw [houtB0, toNat_ofNat_lt _ (by omega)]
  have hiB : (S0.regs "inBase" 0).toNat = inPtr + w * iS := by
    rw [hinB0, toNat_ofNat_lt _ (by omega)]
  have hgS : S0.gmem.size = gm.size := by rw [hgmem]
  have hgi : gmemInpAt S0.gmem (inPtr + w * iS) iS = gmemInpAt gm (inPtr + w * iS) iS := by
    rw [hgmem]
  have hvalid := evalPlan_valid S0.gmem S0.smem (inPtr + w * iS) iS hL hstride
  have hTight : (planToBlock (gmemInpAt S0.gmem (inPtr + w * iS) iS)
      (evalPlan S0.gmem S0.smem (inPtr + w * iS) iS hL)).encode.length ≤ iS + iS / 16 + 256 :=
    planBlock_encode_le_lenOff _ _ hvalid iS
      (by simp only [gmemInpAt, List.length_map, List.length_range])
  have hEncTight : (planToBlock (gmemInpAt S0.gmem (inPtr + w * iS) iS)
      (evalPlan S0.gmem S0.smem (inPtr + w * iS) iS hL)).encode.length ≤ lO := by omega
  have hEnc : (planToBlock (gmemInpAt S0.gmem (inPtr + w * iS) iS)
      (evalPlan S0.gmem S0.smem (inPtr + w * iS) iS hL)).encode.length ≤ 9 * iS := by
    have hb := planBlock_encode_le9 _ _ hvalid
      (by simp only [gmemInpAt, List.length_map, List.length_range]; omega)
    simpa only [gmemInpAt, List.length_map, List.length_range] using hb
  obtain ⟨m, ss', hr2, hdec, hcpl, hmiF, hobPres, hopVal, hgsz, hpcF⟩ :=
    warpKernelDSL_sstep_roundtrips_discharged iS hL
    "Lh0" "Lx1" "Le2" "Ln3" "Lh4" "Lx5" "Le6" "Ln7" "Lh8" "Lx9" "Ch10" "Cx11" "Le12" "Ln13" "Lh14" "Lx15"
    "Le16" "Ln17" "Lh18" "Lx19" "Ch20" "Cx21"
    (myLsic "litExtra") (myLsic "matExtra") (myLsic "litExtraF")
    rfl rfl rfl hstride hipos (by omega) hib40 (by omega) hHash
    { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }
    hop0 hla0 hsp0
    (by show (S0.regs "inBase" 0).toNat < 2 ^ 40; rw [hiB]; exact hib40)
    (by show (S0.regs "outBase" 0).toNat + 9 * iS < 2 ^ 32; rw [hoT]; omega)
    (by show (S0.regs "outBase" 0).toNat + 9 * iS ≤ S0.gmem.size; rw [hoT, hgS]; omega)
    (by show (S0.regs "inBase" 0).toNat + iS ≤ (S0.regs "outBase" 0).toNat; rw [hiB, hoT]; omega)
    (by show (S0.regs "inBase" 0).toNat + iS ≤ (S0.regs "outBase" 0).toNat; rw [hiB, hoT]; omega)
    (by show (S0.regs "inBase" 0).toNat + iS < 2 ^ 64; rw [hiB]; omega)
    (by show (planToBlock (gmemInpAt S0.gmem (S0.regs "inBase" 0).toNat iS)
          (evalPlan S0.gmem S0.smem (S0.regs "inBase" 0).toNat iS hL)).encode.length ≤ 9 * iS
        rw [hiB]; exact hEnc)
    (warpKernelDSL nb iS oS lO hL) 39 S0 hpc39
    (body_segAt nb iS oS lO hL) (body_labelsResolve nb iS oS lO hL) hCouple hMI
  have hpc257 : ss'.pc = 257 := by
    have h218 : (bodyPrefixSeg iS hL).length = 218 := bodyPrefixSeg_length iS hL
    simp only [bodyPrefixSeg] at h218
    rw [hpcF, h218]
  obtain ⟨n3, ss3, hr3, hpc3, hcpl3, _⟩ :=
    tail_sim (inPtr + w * iS) lO (warpKernelDSL nb iS oS lO hL) 257 ss'
      ((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
        { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }) (iS + 34 * iS)
      hpc257 (tail_segAt nb iS oS lO hL) (tail_labelsResolve nb iS oS lO hL) hcpl hmiF
  have hreach0 : SReaches (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1)
      (initSt w inPtr outPtr gm smemB) S0 := by
    rw [← hS0]
    exact sreaches_snsteps (warpKernelDSL nb iS oS lO hL) (25 + 8 * clearIters hL + 8 + 1)
      (initSt w inPtr outPtr gm smemB)
  dsimp only at hdec
  rw [hiB, hgi, hoT, show (gmemInpAt gm (inPtr + w * iS) iS).length = iS from by
    simp [gmemInpAt]] at hdec
  rw [hgi] at hEncTight hTight hvalid
  rw [hiB, hgi] at hopVal
  refine ⟨_, ss3, (planToBlock (gmemInpAt gm (inPtr + w * iS) iS) (evalPlan S0.gmem S0.smem (inPtr + w * iS) iS hL)).encode.length,
    sreaches_trans _ _ _ _ _ _ hreach0 (sreaches_trans _ _ _ _ _ _ hr2 hr3), ?_,
    planBlock_encode_pos _ _ hvalid, hEncTight, ?_, ?_, ?_⟩
  · rw [hpc3, tailEmit_length]
  · have hobN2 : (((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
        { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }).regs "outBase").toNat
        = outPtr + w * oS := by rw [hobPres]; exact hoT
    have hopN : (((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
        { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }).regs "op").toNat
        = (planToBlock (gmemInpAt gm (inPtr + w * iS) iS) (evalPlan S0.gmem S0.smem (inPtr + w * iS) iS hL)).encode.length := by
      rw [hopVal, toNat_ofNat_lt]
      exact Nat.lt_of_le_of_lt hEncTight (by omega)
    have hszN : (((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
        { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }).gmem).size
        = gm.size := by rw [hgsz]; exact hgS
    have hlf := tailStmt_lenField lO ((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
        { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }) (iS + 34 * iS)
      (by rw [hobN2]; omega) (by rw [hobN2, hszN]; omega)
      (by rw [hopN]; omega)
    rw [hobN2] at hlf
    rw [show ss3.gmem = ((tailStmt lO).eval (iS + 34 * iS)
      ((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
        { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem })).gmem from hcpl3.gmem]
    rw [hlf, hopN]
  · rw [← hdec]
    congr 1
    apply List.map_congr_left
    intro i hi
    rw [List.mem_range] at hi
    have hobN : (((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
        { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }).regs "outBase").toNat
        = outPtr + w * oS := by rw [hobPres]; exact hoT
    have hgm3 : ss3.gmem = ((tailStmt lO).eval (iS + 34 * iS)
        ((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
          { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem })).gmem := hcpl3.gmem
    have hgmPrev : ((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
        { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }).gmem = ss'.gmem :=
      hcpl.gmem.symm
    rw [hgm3, tailStmt_frame lO _ (iS + 34 * iS) (outPtr + w * oS + i)
      (by rw [hobN]; omega) (by rw [hobN]; omega), hgmPrev]
  · -- write confinement: outside `[outBase, outBase+lO+4)` nothing changed
    intro j hj
    have hobN : (((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
        { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }).regs "outBase").toNat
        = outPtr + w * oS := by rw [hobPres]; exact hoT
    have hgm3 : ss3.gmem = ((tailStmt lO).eval (iS + 34 * iS)
        ((bodyEncodePrefix iS hL).eval (iS + 34 * iS)
          { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem })).gmem := hcpl3.gmem
    rw [hgm3, tailStmt_frame lO _ (iS + 34 * iS) j
      (by rw [hobN]; rcases hj with h | h <;> omega) (by rw [hobN]; omega)]
    have hbody := (compressorBody_output_eq iS hL
        { regs := fun r => S0.regs r 0, gmem := S0.gmem, smem := S0.smem }
        hstride (by omega) hipos hop0 hla0 hsp0
        (by show (S0.regs "inBase" 0).toNat < 2 ^ 40; rw [hiB]; exact hib40)
        (by show (S0.regs "outBase" 0).toNat + 9 * iS < 2 ^ 64; rw [hoT]; omega)
        (by show (S0.regs "outBase" 0).toNat + 9 * iS ≤ S0.gmem.size; rw [hoT, hgS]; omega)
        (by show (S0.regs "inBase" 0).toNat + iS ≤ (S0.regs "outBase" 0).toNat
            rw [hiB, hoT]; omega)
        (by show (S0.regs "inBase" 0).toNat + iS < 2 ^ 64; rw [hiB]; omega)
        (by show (planToBlock (gmemInpAt S0.gmem (S0.regs "inBase" 0).toNat iS)
              (evalPlan S0.gmem S0.smem (S0.regs "inBase" 0).toNat iS hL)).encode.length ≤ 9 * iS
            rw [hiB]; exact hEnc)).1
    rw [hbody, hiB, hgi]
    rcases hj with h | h
    · rw [EmitContent.putBytesU_getD_lt j
        (planToBlock (gmemInpAt gm (inPtr + w * iS) iS) (evalPlan S0.gmem S0.smem (inPtr + w * iS) iS hL)).encode
        S0.gmem (S0.regs "outBase" 0) (by rw [hoT]; omega) (by rw [hoT]; omega), hgmem]
    · rw [EmitContent.putBytesU_getD_ge j
        (planToBlock (gmemInpAt gm (inPtr + w * iS) iS) (evalPlan S0.gmem S0.smem (inPtr + w * iS) iS hL)).encode
        S0.gmem (S0.regs "outBase" 0) (by rw [hoT]; omega) (by rw [hoT]; omega), hgmem]

end WarpLaunch

end AlgorithmLib.LZ4WarpDSL
