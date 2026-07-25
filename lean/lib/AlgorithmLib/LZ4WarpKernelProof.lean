import AlgorithmLib.LZ4WarpKernel

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib AlgorithmLib.LZ4Simt AlgorithmLib.LZ4WarpSched AlgorithmLib.LZ4WarpFind

-- ── Collective leaf 1: one extend 32-window computes `leadingGood` ─────────────

/-- **`coopExtendStep` = `leadingGood`**: from the segment `coopExtendEmit "adv"`
    (per-lane compare + `vote;bnot;brev;clz`), after 18 machine steps every lane of
    `adv` holds W2's `leadingGood inp p c endCap ml 32`.  Composes
    `extendKernelGoodPred_machine` (compare → `pOk`), `..._machine_to_good` (→ model
    `good`, via the caller's `hbound`/`hbytes`), and `extendKernelWindow`
    (reduction → `leadingGood`).  `hbound`/`hbytes` are discharged at assembly from
    the loop invariant (uniform regs, `lane = l.val`, in-bounds). -/
theorem coopExtendStep_adv (prog : Array SInstr) (ss : SState) (base : Nat)
    (inp : List UInt8) (p c endCap ml : Nat)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "idx" "ml" "lane"))
    (h1 : prog[base + 1]? = some (.binr .add "pe" "p0" "idx"))
    (h2 : prog[base + 2]? = some (.setp .lt "pIn" "pe" (.reg "ecR")))
    (h3 : prog[base + 3]? = some (.binr .min "peC" "pe" "ec1"))
    (h4 : prog[base + 4]? = some (.binr .sub "dfe" "peC" "p0"))
    (h5 : prog[base + 5]? = some (.binr .add "caC" "cand0" "dfe"))
    (h6 : prog[base + 6]? = some (.mov "peD" (.reg "peC")))
    (h7 : prog[base + 7]? = some (.binr .add "aP" "inBase" "peD"))
    (h8 : prog[base + 8]? = some (.mov "caD" (.reg "caC")))
    (h9 : prog[base + 9]? = some (.binr .add "aC" "inBase" "caD"))
    (h10 : prog[base + 10]? = some (.ldgo "bP" "aP" 0))
    (h11 : prog[base + 11]? = some (.ldgo "bC" "aC" 0))
    (h12 : prog[base + 12]? = some (.setp .eq "pEqB" "bP" (.reg "bC")))
    (h13 : prog[base + 13]? = some (.andp "pOk" "pIn" "pEqB"))
    (h14 : prog[base + 14]? = some (.vote "balOk" "pOk"))
    (h15 : prog[base + 15]? = some (.bnot "mis" "balOk"))
    (h16 : prog[base + 16]? = some (.brev "revM" "mis"))
    (h17 : prog[base + 17]? = some (.clz "adv" "revM"))
    (hbound : ∀ l : Fin 32,
      ((SOp.run .add (ss.regs "p0" l)
          (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l))) < ss.regs "ecR" l)
        ↔ p + (ml + l.val) < endCap)
    (hbytes : ∀ l : Fin 32, p + (ml + l.val) < endCap →
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
        = (byte inp (p + (ml + l.val)) == byte inp (c + (ml + l.val))))
    (l : Fin 32) :
    ((snsteps prog 18 ss).regs "adv" l).toNat = leadingGood inp p c endCap ml 32 := by
  -- pc-anchored instruction hypotheses at `ss.pc`.
  subst hpc
  -- the good-predicate register after 14 steps equals the model `good`.
  have hgood : ∀ k : Fin 32,
      (snsteps prog 14 ss).regs "pOk" k
        = if good inp p c endCap (ml + k.val) then 1 else 0 := by
    intro k
    rw [extendKernelGoodPred_machine prog ss h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 k]
    exact extendKernelGoodPred_machine_to_good ss inp p c endCap ml k (hbound k) (hbytes k)
  -- pc after the 14 good-predicate steps.
  have hpc14 : (snsteps prog 14 ss).pc = ss.pc + 14 :=
    extendKernelGoodPred_pc prog ss h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13
  -- the reduction instructions relocated to the post-goodPred pc.
  have h14' : prog[(snsteps prog 14 ss).pc]? = some (.vote "balOk" "pOk") := by rw [hpc14]; exact h14
  have h15' : prog[(snsteps prog 14 ss).pc + 1]? = some (.bnot "mis" "balOk") := by rw [hpc14]; exact h15
  have h16' : prog[(snsteps prog 14 ss).pc + 2]? = some (.brev "revM" "mis") := by rw [hpc14]; exact h16
  have h17' : prog[(snsteps prog 14 ss).pc + 3]? = some (.clz "adv" "revM") := by rw [hpc14]; exact h17
  -- extend reduction from the post-goodPred state → `leadingGood`.
  have hwin :
      ((sstep prog (sstep prog (sstep prog (sstep prog (snsteps prog 14 ss))))).regs "adv" l).toNat
        = leadingGood inp p c endCap ml 32 :=
    extendKernelWindow prog (snsteps prog 14 ss) "adv" "revM" "mis" "balOk" "pOk"
      inp p c endCap ml h14' h15' h16' h17' hgood l
  -- `snsteps 18 = sstep⁴ ∘ snsteps 14`.
  have h18 : snsteps prog 18 ss = sstep prog (sstep prog (sstep prog (sstep prog (snsteps prog 14 ss)))) := by
    rw [show (18 : Nat) = 14 + 4 by rfl, snsteps_add]
    rfl
  rw [h18]; exact hwin

-- ── Frame for the 4-instr complement reduction (vote;bnot;brev;clz) ────────────

/-- The extend reduction `vote balOk pOk; bnot mis balOk; brev revM mis; clz adv revM`
    writes only `balOk/mis/revM/adv`; every other register is preserved.  (The
    complement analogue of `select3_frame`, needed for `Couple` preservation.) -/
theorem coopExtend_reduction_frame (prog : Array SInstr) (ss : SState) (r : String)
    (h0 : prog[ss.pc]? = some (.vote "balOk" "pOk"))
    (h1 : prog[ss.pc + 1]? = some (.bnot "mis" "balOk"))
    (h2 : prog[ss.pc + 2]? = some (.brev "revM" "mis"))
    (h3 : prog[ss.pc + 3]? = some (.clz "adv" "revM"))
    (hb : r ≠ "balOk") (hm : r ≠ "mis") (hv : r ≠ "revM") (ha : r ≠ "adv") :
    (sstep prog (sstep prog (sstep prog (sstep prog ss)))).regs r = ss.regs r := by
  -- Step 1: vote balOk.
  have pc1 : (sstep prog ss).pc = ss.pc + 1 := by rw [vote_step prog ss "balOk" "pOk" h0]; rfl
  have r1 : (sstep prog ss).regs r = ss.regs r := by
    rw [vote_step prog ss "balOk" "pOk" h0]; simp [SState.setReg, SState.setPc, hb]
  -- Step 2: bnot mis.
  have h1' : prog[(sstep prog ss).pc]? = some (.bnot "mis" "balOk") := by rw [pc1]; exact h1
  have pc2 : (sstep prog (sstep prog ss)).pc = ss.pc + 2 := by
    rw [bnot_step prog _ "mis" "balOk" h1']; simp [SState.setPc, pc1]
  have r2 : (sstep prog (sstep prog ss)).regs r = ss.regs r := by
    rw [bnot_step prog _ "mis" "balOk" h1']; simp [SState.setReg, SState.setPc, hm]; exact r1
  -- Step 3: brev revM.
  have h2' : prog[(sstep prog (sstep prog ss)).pc]? = some (.brev "revM" "mis") := by rw [pc2]; exact h2
  have pc3 : (sstep prog (sstep prog (sstep prog ss))).pc = ss.pc + 3 := by
    rw [brev_step prog _ "revM" "mis" h2']; simp [SState.setPc, pc2]
  have r3 : (sstep prog (sstep prog (sstep prog ss))).regs r = ss.regs r := by
    rw [brev_step prog _ "revM" "mis" h2']; simp [SState.setReg, SState.setPc, hv]; exact r2
  -- Step 4: clz adv.
  have h3' : prog[(sstep prog (sstep prog (sstep prog ss))).pc]? = some (.clz "adv" "revM") := by
    rw [pc3]; exact h3
  rw [clz_step prog _ "adv" "revM" h3']; simp [SState.setReg, SState.setPc, ha]; exact r3

-- ── General frame for the 14-instr good-predicate segment ─────────────────────

/-- The good-predicate segment writes only its 14 scratch registers; any register
    `r` outside that set is preserved.  (General `r` version of
    `extendKernelGoodPred_frame_core`, needed to preserve the full `Couple` reg set.) -/
theorem coopExtend_goodPred_frame (prog : Array SInstr) (ss : SState) (r : String)
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
    (hr : r ∉ ["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
               "bP", "bC", "pEqB", "pOk"]) :
    (snsteps prog 14 ss).regs r = ss.regs r := by
  simp only [List.mem_cons, List.mem_singleton, not_or, List.not_mem_nil,
    not_false_iff, and_true] at hr
  obtain ⟨n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13⟩ := hr
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13,
    sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run,
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]

-- ── Full 18-step frame for the extend segment ─────────────────────────────────

/-- The whole `coopExtendEmit "adv"` segment (18 steps) preserves every register
    outside its 18 scratch/result names.  Composes the goodPred + reduction frames. -/
theorem coopExtendStep_frame (prog : Array SInstr) (ss : SState) (base : Nat) (r : String)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "idx" "ml" "lane"))
    (h1 : prog[base + 1]? = some (.binr .add "pe" "p0" "idx"))
    (h2 : prog[base + 2]? = some (.setp .lt "pIn" "pe" (.reg "ecR")))
    (h3 : prog[base + 3]? = some (.binr .min "peC" "pe" "ec1"))
    (h4 : prog[base + 4]? = some (.binr .sub "dfe" "peC" "p0"))
    (h5 : prog[base + 5]? = some (.binr .add "caC" "cand0" "dfe"))
    (h6 : prog[base + 6]? = some (.mov "peD" (.reg "peC")))
    (h7 : prog[base + 7]? = some (.binr .add "aP" "inBase" "peD"))
    (h8 : prog[base + 8]? = some (.mov "caD" (.reg "caC")))
    (h9 : prog[base + 9]? = some (.binr .add "aC" "inBase" "caD"))
    (h10 : prog[base + 10]? = some (.ldgo "bP" "aP" 0))
    (h11 : prog[base + 11]? = some (.ldgo "bC" "aC" 0))
    (h12 : prog[base + 12]? = some (.setp .eq "pEqB" "bP" (.reg "bC")))
    (h13 : prog[base + 13]? = some (.andp "pOk" "pIn" "pEqB"))
    (h14 : prog[base + 14]? = some (.vote "balOk" "pOk"))
    (h15 : prog[base + 15]? = some (.bnot "mis" "balOk"))
    (h16 : prog[base + 16]? = some (.brev "revM" "mis"))
    (h17 : prog[base + 17]? = some (.clz "adv" "revM"))
    (hr : r ∉ ["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
               "bP", "bC", "pEqB", "pOk", "balOk", "mis", "revM", "adv"]) :
    (snsteps prog 18 ss).regs r = ss.regs r := by
  subst hpc
  simp only [List.mem_cons, List.mem_singleton, not_or, List.not_mem_nil,
    not_false_iff, and_true] at hr
  obtain ⟨n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, nb, nm, nv, na⟩ := hr
  rw [show (18 : Nat) = 14 + 4 by rfl, snsteps_add]
  have hpc14 : (snsteps prog 14 ss).pc = ss.pc + 14 :=
    extendKernelGoodPred_pc prog ss h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13
  have h14' : prog[(snsteps prog 14 ss).pc]? = some (.vote "balOk" "pOk") := by rw [hpc14]; exact h14
  have h15' : prog[(snsteps prog 14 ss).pc + 1]? = some (.bnot "mis" "balOk") := by rw [hpc14]; exact h15
  have h16' : prog[(snsteps prog 14 ss).pc + 2]? = some (.brev "revM" "mis") := by rw [hpc14]; exact h16
  have h17' : prog[(snsteps prog 14 ss).pc + 3]? = some (.clz "adv" "revM") := by rw [hpc14]; exact h17
  have hred := coopExtend_reduction_frame prog (snsteps prog 14 ss) r h14' h15' h16' h17' nb nm nv na
  have h4eq : snsteps prog 4 (snsteps prog 14 ss)
      = sstep prog (sstep prog (sstep prog (sstep prog (snsteps prog 14 ss)))) := rfl
  rw [h4eq, hred]
  exact coopExtend_goodPred_frame prog ss r h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13
    (by simp only [List.mem_cons, List.mem_singleton, not_or, List.not_mem_nil,
          not_false_iff, and_true]
        exact ⟨n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13⟩)

-- ── The reduction preserves gmem/smem (reg-only ops) ──────────────────────────

/-- `vote;bnot;brev;clz` touch no memory. -/
theorem coopExtend_reduction_mem (prog : Array SInstr) (ss : SState)
    (h0 : prog[ss.pc]? = some (.vote "balOk" "pOk"))
    (h1 : prog[ss.pc + 1]? = some (.bnot "mis" "balOk"))
    (h2 : prog[ss.pc + 2]? = some (.brev "revM" "mis"))
    (h3 : prog[ss.pc + 3]? = some (.clz "adv" "revM")) :
    (sstep prog (sstep prog (sstep prog (sstep prog ss)))).gmem = ss.gmem ∧
    (sstep prog (sstep prog (sstep prog (sstep prog ss)))).smem = ss.smem := by
  have pc1 : (sstep prog ss).pc = ss.pc + 1 := by rw [vote_step prog ss "balOk" "pOk" h0]; rfl
  have h1' : prog[(sstep prog ss).pc]? = some (.bnot "mis" "balOk") := by rw [pc1]; exact h1
  have pc2 : (sstep prog (sstep prog ss)).pc = ss.pc + 2 := by
    rw [bnot_step prog _ "mis" "balOk" h1']; simp [SState.setPc, pc1]
  have h2' : prog[(sstep prog (sstep prog ss)).pc]? = some (.brev "revM" "mis") := by rw [pc2]; exact h2
  have pc3 : (sstep prog (sstep prog (sstep prog ss))).pc = ss.pc + 3 := by
    rw [brev_step prog _ "revM" "mis" h2']; simp [SState.setPc, pc2]
  have h3' : prog[(sstep prog (sstep prog (sstep prog ss))).pc]? = some (.clz "adv" "revM") := by
    rw [pc3]; exact h3
  refine ⟨?_, ?_⟩ <;>
    rw [clz_step prog _ "adv" "revM" h3', brev_step prog _ "revM" "mis" h2',
        bnot_step prog _ "mis" "balOk" h1', vote_step prog ss "balOk" "pOk" h0] <;> rfl

/-- The reduction advances the pc by 4. -/
theorem coopExtend_reduction_pc (prog : Array SInstr) (ss : SState)
    (h0 : prog[ss.pc]? = some (.vote "balOk" "pOk"))
    (h1 : prog[ss.pc + 1]? = some (.bnot "mis" "balOk"))
    (h2 : prog[ss.pc + 2]? = some (.brev "revM" "mis"))
    (h3 : prog[ss.pc + 3]? = some (.clz "adv" "revM")) :
    (sstep prog (sstep prog (sstep prog (sstep prog ss)))).pc = ss.pc + 4 := by
  have pc1 : (sstep prog ss).pc = ss.pc + 1 := by rw [vote_step prog ss "balOk" "pOk" h0]; rfl
  have h1' : prog[(sstep prog ss).pc]? = some (.bnot "mis" "balOk") := by rw [pc1]; exact h1
  have pc2 : (sstep prog (sstep prog ss)).pc = ss.pc + 2 := by
    rw [bnot_step prog _ "mis" "balOk" h1']; simp [SState.setPc, pc1]
  have h2' : prog[(sstep prog (sstep prog ss)).pc]? = some (.brev "revM" "mis") := by rw [pc2]; exact h2
  have pc3 : (sstep prog (sstep prog (sstep prog ss))).pc = ss.pc + 3 := by
    rw [brev_step prog _ "revM" "mis" h2']; simp [SState.setPc, pc2]
  have h3' : prog[(sstep prog (sstep prog (sstep prog ss))).pc]? = some (.clz "adv" "revM") := by
    rw [pc3]; exact h3
  rw [clz_step prog _ "adv" "revM" h3']; simp [SState.setPc, pc3]

-- ── The extend collective `Couple` leaf ───────────────────────────────────────

/-- **Extend collective leaf**: from a `Couple`d state at the segment start, the
    18-step `coopExtendEmit "adv"` reaches a state coupled to `evalCoopExtendStep`.
    `hbound`/`hbytes` are the loop invariant's in-bounds facts (discharged at assembly). -/
theorem coopExtendStep_couple (R : List String) (prog : Array SInstr) (ss : SState) (ws : WState)
    (base inStride endCap : Nat)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "idx" "ml" "lane"))
    (h1 : prog[base + 1]? = some (.binr .add "pe" "p0" "idx"))
    (h2 : prog[base + 2]? = some (.setp .lt "pIn" "pe" (.reg "ecR")))
    (h3 : prog[base + 3]? = some (.binr .min "peC" "pe" "ec1"))
    (h4 : prog[base + 4]? = some (.binr .sub "dfe" "peC" "p0"))
    (h5 : prog[base + 5]? = some (.binr .add "caC" "cand0" "dfe"))
    (h6 : prog[base + 6]? = some (.mov "peD" (.reg "peC")))
    (h7 : prog[base + 7]? = some (.binr .add "aP" "inBase" "peD"))
    (h8 : prog[base + 8]? = some (.mov "caD" (.reg "caC")))
    (h9 : prog[base + 9]? = some (.binr .add "aC" "inBase" "caD"))
    (h10 : prog[base + 10]? = some (.ldgo "bP" "aP" 0))
    (h11 : prog[base + 11]? = some (.ldgo "bC" "aC" 0))
    (h12 : prog[base + 12]? = some (.setp .eq "pEqB" "bP" (.reg "bC")))
    (h13 : prog[base + 13]? = some (.andp "pOk" "pIn" "pEqB"))
    (h14 : prog[base + 14]? = some (.vote "balOk" "pOk"))
    (h15 : prog[base + 15]? = some (.bnot "mis" "balOk"))
    (h16 : prog[base + 16]? = some (.brev "revM" "mis"))
    (h17 : prog[base + 17]? = some (.clz "adv" "revM"))
    (hc : Couple R ss ws)
    (hadv : "adv" ∈ R)
    (hRdisj : ∀ r ∈ R, r ∉ ["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
               "bP", "bC", "pEqB", "pOk", "balOk", "mis", "revM"])
    (hbound : ∀ l : Fin 32,
      ((SOp.run .add (ss.regs "p0" l)
          (SOp.run .add (ss.regs "ml" l) (ss.regs "lane" l))) < ss.regs "ecR" l)
        ↔ (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap)
    (hbytes : ∀ l : Fin 32,
      (ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val) < endCap →
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
        = (byte (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) ((ws.regs "p0").toNat + ((ws.regs "ml").toNat + l.val))
           == byte (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) ((ws.regs "cand0").toNat + ((ws.regs "ml").toNat + l.val)))) :
    SReaches prog 18 ss (snsteps prog 18 ss) ∧ (snsteps prog 18 ss).pc = base + 18 ∧
      Couple R (snsteps prog 18 ss)
        (evalCoopExtendStep "adv" "p0" "cand0" "ml" inStride endCap ws) := by
  subst hpc
  -- adv result (per lane).
  have hadvVal : ∀ l : Fin 32,
      ((snsteps prog 18 ss).regs "adv" l).toNat
        = leadingGood (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (ws.regs "p0").toNat (ws.regs "cand0").toNat
            endCap (ws.regs "ml").toNat 32 := fun l =>
    coopExtendStep_adv prog ss ss.pc (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (ws.regs "p0").toNat
      (ws.regs "cand0").toNat endCap (ws.regs "ml").toNat rfl
      h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 hbound hbytes l
  -- gmem/smem preserved over 18 steps.
  have hpc14 : (snsteps prog 14 ss).pc = ss.pc + 14 :=
    extendKernelGoodPred_pc prog ss h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13
  have h14' : prog[(snsteps prog 14 ss).pc]? = some (.vote "balOk" "pOk") := by rw [hpc14]; exact h14
  have h15' : prog[(snsteps prog 14 ss).pc + 1]? = some (.bnot "mis" "balOk") := by rw [hpc14]; exact h15
  have h16' : prog[(snsteps prog 14 ss).pc + 2]? = some (.brev "revM" "mis") := by rw [hpc14]; exact h16
  have h17' : prog[(snsteps prog 14 ss).pc + 3]? = some (.clz "adv" "revM") := by rw [hpc14]; exact h17
  obtain ⟨_, _, _, _, _, _, _, hgmem14, hsmem14⟩ :=
    extendKernelGoodPred_frame_core prog ss h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13
  obtain ⟨hgmemR, hsmemR⟩ := coopExtend_reduction_mem prog (snsteps prog 14 ss) h14' h15' h16' h17'
  have h4eq : snsteps prog 4 (snsteps prog 14 ss)
      = sstep prog (sstep prog (sstep prog (sstep prog (snsteps prog 14 ss)))) := rfl
  have hgmem18 : (snsteps prog 18 ss).gmem = ss.gmem := by
    rw [show (18 : Nat) = 14 + 4 by rfl, snsteps_add, h4eq, hgmemR, hgmem14]
  have hsmem18 : (snsteps prog 18 ss).smem = ss.smem := by
    rw [show (18 : Nat) = 14 + 4 by rfl, snsteps_add, h4eq, hsmemR, hsmem14]
  -- pc after 18 steps.
  have hpc18 : (snsteps prog 18 ss).pc = ss.pc + 18 := by
    rw [show (18 : Nat) = 14 + 4 by rfl, snsteps_add, h4eq,
        coopExtend_reduction_pc prog (snsteps prog 14 ss) h14' h15' h16' h17', hpc14]
  refine ⟨sreaches_snsteps prog 18 ss, hpc18, ?_, ?_, ?_⟩
  · rw [hgmem18]; exact hc.1
  · rw [hsmem18]; exact hc.2.1
  -- regs coupling.
  intro r hrR l
  by_cases hra : r = "adv"
  · subst hra
    rw [evalCoopExtendStep, ← hadvVal l]
    simp [WState.setReg, UInt64.ofNat_toNat]
  · have hframe := coopExtendStep_frame prog ss ss.pc r rfl
      h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17
      (by
        have hh := hRdisj r hrR
        intro hmem
        have hmem2 : r ∈ (["idx", "pe", "pIn", "peC", "dfe", "caC", "peD", "aP", "caD", "aC",
            "bP", "bC", "pEqB", "pOk", "balOk", "mis", "revM"] : List String) ++ ["adv"] := hmem
        rcases List.mem_append.mp hmem2 with h | h
        · exact hh h
        · exact hra (List.mem_singleton.mp h))
    rw [hframe, hc.reg hrR l, evalCoopExtendStep]
    simp [WState.setReg, hra]

-- ── Byte-assembly foundation for the coopWindow probe (load4 = model wLoad4) ──

/-- Disjoint `Nat.lor` is addition: `a ||| (b <<< k) = a + b <<< k` when `a < 2^k`. -/
theorem lor_shift_add (a b k : Nat) (ha : a < 2 ^ k) : a ||| (b <<< k) = a + b <<< k := by
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

theorem lor_mul_add (a b k : Nat) (ha : a < 2 ^ k) : a ||| (b * 2 ^ k) = a + b * 2 ^ k := by
  rw [← Nat.shiftLeft_eq]; exact lor_shift_add a b k ha

/-- A `UInt64` left shift by a small constant, on a byte-sized value, is a multiply. -/
theorem shift_eq (x m : UInt64) (mn : Nat) (hm : m.toNat = mn) (hmn : mn < 64)
    (hlt : x.toNat * 2 ^ mn < 2 ^ 64) : (x <<< m).toNat = x.toNat * 2 ^ mn := by
  rw [UInt64.toNat_shiftLeft, hm, Nat.mod_eq_of_lt hmn, Nat.shiftLeft_eq, Nat.mod_eq_of_lt hlt]

/-- **Little-endian 4-byte assembly = the model integer** (`wLoad4`'s shape): the
    kernel's `b0 | b1<<8 | b2<<16 | b3<<24` on byte-sized lanes equals the LE sum. -/
theorem byte4_toNat (b0 b1 b2 b3 : UInt64)
    (h0 : b0.toNat < 256) (h1 : b1.toNat < 256) (h2 : b2.toNat < 256) (h3 : b3.toNat < 256) :
    (((b0 ||| b1 <<< 8) ||| b2 <<< 16) ||| b3 <<< 24).toNat
      = b0.toNat + 256 * b1.toNat + 65536 * b2.toNat + 16777216 * b3.toNat := by
  have e1 := shift_eq b1 8 8 (by decide) (by decide) (by omega)
  have e2 := shift_eq b2 16 16 (by decide) (by decide) (by omega)
  have e3 := shift_eq b3 24 24 (by decide) (by decide) (by omega)
  rw [UInt64.toNat_or, UInt64.toNat_or, UInt64.toNat_or, e1, e2, e3,
      lor_mul_add b0.toNat b1.toNat 8 (by omega),
      lor_mul_add _ b2.toNat 16 (by omega),
      lor_mul_add _ b3.toNat 24 (by omega)]
  omega

-- ── coopWindow probe: current 4-byte load = model wLoad4 ─────────────────────

/-- The 15-instr load4 prefix computes `v32` = the model `wLoad4` of the current
    position's 4 bytes (via `byte4_toNat`).  `capC = inStride-4`. -/
theorem cw_v32 (prog : Array SInstr) (ss : SState) (searchLim capC : Nat)
    (h0 : prog[ss.pc]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[ss.pc + 1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[ss.pc + 2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[ss.pc + 3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[ss.pc + 4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[ss.pc + 5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[ss.pc + 6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[ss.pc + 7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[ss.pc + 8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[ss.pc + 9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[ss.pc + 10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[ss.pc + 11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[ss.pc + 12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[ss.pc + 13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[ss.pc + 14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (l : Fin 32) :
    ((snsteps prog 15 ss).regs "v32" l).toNat
      = wLoad4 ss.gmem
          (SOp.run .add (ss.regs "inBase" l)
            (SOp.run .min (SOp.run .add (ss.regs "searchPos" l) (ss.regs "lane" l))
              (UInt64.ofNat capC))).toNat := by
  have hv : (snsteps prog 15 ss).regs "v32" l
      = (((UInt64.ofNat (ss.gmem.getD (SOp.run .add (ss.regs "inBase" l)
            (SOp.run .min (SOp.run .add (ss.regs "searchPos" l) (ss.regs "lane" l))
              (UInt64.ofNat capC))).toNat 0).toNat)
        ||| UInt64.ofNat (ss.gmem.getD (SOp.run .add (ss.regs "inBase" l)
            (SOp.run .min (SOp.run .add (ss.regs "searchPos" l) (ss.regs "lane" l))
              (UInt64.ofNat capC))).toNat.succ 0).toNat <<< 8)
        ||| UInt64.ofNat (ss.gmem.getD ((SOp.run .add (ss.regs "inBase" l)
            (SOp.run .min (SOp.run .add (ss.regs "searchPos" l) (ss.regs "lane" l))
              (UInt64.ofNat capC))).toNat + 2) 0).toNat <<< 16)
        ||| UInt64.ofNat (ss.gmem.getD ((SOp.run .add (ss.regs "inBase" l)
            (SOp.run .min (SOp.run .add (ss.regs "searchPos" l) (ss.regs "lane" l))
              (UInt64.ofNat capC))).toNat + 3) 0).toNat <<< 24 := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, SCmp.run,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
  rw [hv]
  have hb : ∀ (idx : Nat), (UInt64.ofNat ((ss.gmem.getD idx 0).toNat)).toNat = (ss.gmem.getD idx 0).toNat := by
    intro idx
    have h : (ss.gmem.getD idx 0).toNat < 2 ^ 64 := by have := (ss.gmem.getD idx 0).toNat_lt; omega
    simp [UInt64.toNat_ofNat, Nat.mod_eq_of_lt h]
  have hlt : ∀ (idx : Nat), (UInt64.ofNat ((ss.gmem.getD idx 0).toNat)).toNat < 256 := by
    intro idx; rw [hb]; have := (ss.gmem.getD idx 0).toNat_lt; omega
  rw [byte4_toNat _ _ _ _ (hlt _) (hlt _) (hlt _) (hlt _)]
  simp only [wLoad4, hb, Nat.succ_eq_add_one]

-- ── coopWindow probe: hash = model wHash ─────────────────────────────────────

/-- The kernel's `vc == v32` (4-byte UInt64 compare) equals the model's per-byte
    `verify4`: two `wLoad4`s are equal iff their 4 bytes agree. -/
theorem wLoad4_eq_iff (g : Array UInt8) (p c : Nat) :
    wLoad4 g p = wLoad4 g c ↔
      ((g.getD p 0).toNat = (g.getD c 0).toNat ∧
       (g.getD (p + 1) 0).toNat = (g.getD (c + 1) 0).toNat ∧
       (g.getD (p + 2) 0).toNat = (g.getD (c + 2) 0).toNat ∧
       (g.getD (p + 3) 0).toNat = (g.getD (c + 3) 0).toNat) := by
  unfold wLoad4
  have a0 := (g.getD p 0).toNat_lt; have a1 := (g.getD (p+1) 0).toNat_lt
  have a2 := (g.getD (p+2) 0).toNat_lt; have a3 := (g.getD (p+3) 0).toNat_lt
  have b0 := (g.getD c 0).toNat_lt; have b1 := (g.getD (c+1) 0).toNat_lt
  have b2 := (g.getD (c+2) 0).toNat_lt; have b3 := (g.getD (c+3) 0).toNat_lt
  omega

-- ── coopWindow probe: candidate 4-byte load = model wLoad4 ───────────────────

/-- The 13-instr candidate load (`cand=candRaw-1; rc=min; rcA=inBase+rc; load4`)
    computes `vc` = the model `wLoad4` of the candidate's 4 bytes.  Reuses `byte4_toNat`. -/
theorem cw_vc (prog : Array SInstr) (ss : SState) (l : Fin 32)
    (h0 : prog[ss.pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h1 : prog[ss.pc + 1]? = some (.binr .min "rc" "cand" "cap4"))
    (h2 : prog[ss.pc + 2]? = some (.binr .add "rcA" "inBase" "rc"))
    (h3 : prog[ss.pc + 3]? = some (.ldgo "c0" "rcA" 0))
    (h4 : prog[ss.pc + 4]? = some (.ldgo "c1" "rcA" 1))
    (h5 : prog[ss.pc + 5]? = some (.ldgo "c2" "rcA" 2))
    (h6 : prog[ss.pc + 6]? = some (.ldgo "c3" "rcA" 3))
    (h7 : prog[ss.pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h8 : prog[ss.pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h9 : prog[ss.pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h10 : prog[ss.pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h11 : prog[ss.pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h12 : prog[ss.pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3"))) :
    ((snsteps prog 13 ss).regs "vc" l).toNat
      = wLoad4 ss.gmem
          (SOp.run .add (ss.regs "inBase" l)
            (SOp.run .min (SOp.run .sub (ss.regs "candRaw" l) (UInt64.ofNat 1)) (ss.regs "cap4" l))).toNat := by
  have hv : (snsteps prog 13 ss).regs "vc" l
      = (((UInt64.ofNat (ss.gmem.getD (SOp.run .add (ss.regs "inBase" l)
            (SOp.run .min (SOp.run .sub (ss.regs "candRaw" l) (UInt64.ofNat 1)) (ss.regs "cap4" l))).toNat 0).toNat)
        ||| UInt64.ofNat (ss.gmem.getD (SOp.run .add (ss.regs "inBase" l)
            (SOp.run .min (SOp.run .sub (ss.regs "candRaw" l) (UInt64.ofNat 1)) (ss.regs "cap4" l))).toNat.succ 0).toNat <<< 8)
        ||| UInt64.ofNat (ss.gmem.getD ((SOp.run .add (ss.regs "inBase" l)
            (SOp.run .min (SOp.run .sub (ss.regs "candRaw" l) (UInt64.ofNat 1)) (ss.regs "cap4" l))).toNat + 2) 0).toNat <<< 16)
        ||| UInt64.ofNat (ss.gmem.getD ((SOp.run .add (ss.regs "inBase" l)
            (SOp.run .min (SOp.run .sub (ss.regs "candRaw" l) (UInt64.ofNat 1)) (ss.regs "cap4" l))).toNat + 3) 0).toNat <<< 24 := by
    simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run]
  rw [hv]
  have hb : ∀ (idx : Nat), (UInt64.ofNat ((ss.gmem.getD idx 0).toNat)).toNat = (ss.gmem.getD idx 0).toNat := by
    intro idx
    have h : (ss.gmem.getD idx 0).toNat < 2 ^ 64 := by have := (ss.gmem.getD idx 0).toNat_lt; omega
    simp [UInt64.toNat_ofNat, Nat.mod_eq_of_lt h]
  have hlt : ∀ (idx : Nat), (UInt64.ofNat ((ss.gmem.getD idx 0).toNat)).toNat < 256 := by
    intro idx; rw [hb]; have := (ss.gmem.getD idx 0).toNat_lt; omega
  rw [byte4_toNat _ _ _ _ (hlt _) (hlt _) (hlt _) (hlt _)]
  simp only [wLoad4, hb, Nat.succ_eq_add_one]

-- ── coopWindow probe: pHit boolean = the four probe conditions ───────────────

/-- The closing `setp ne/lt/eq; andp x3` computes `pHit = 1` iff all four probe
    conditions hold (pValid, candRaw≠0, cand<posP, vc==v32). -/
theorem cw_pHit_comb (prog : Array SInstr) (ss : SState) (l : Fin 32)
    (h0 : prog[ss.pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h1 : prog[ss.pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h2 : prog[ss.pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h3 : prog[ss.pc + 3]? = some (.andp "pH1" "pValid" "pNE"))
    (h4 : prog[ss.pc + 4]? = some (.andp "pH2" "pH1" "pCO"))
    (h5 : prog[ss.pc + 5]? = some (.andp "pHit" "pH2" "pEq")) :
    (snsteps prog 6 ss).regs "pHit" l = 1 ↔
      ((ss.regs "pValid" l = 1 ∧ ss.regs "candRaw" l ≠ 0 ∧
        ss.regs "cand" l < ss.regs "posP" l) ∧ ss.regs "vc" l = ss.regs "v32" l) := by
  simp only [snsteps, sstep, h0, h1, h2, h3, h4, h5, sstepInstr, SState.setReg, SState.setPc,
    SState.get, SCmp.run, beq_iff_eq, bne_iff_ne, ne_eq]
  by_cases hp : (ss.regs "pValid" l = 1 ∧ ss.regs "candRaw" l ≠ 0 ∧
        ss.regs "cand" l < ss.regs "posP" l) ∧ ss.regs "vc" l = ss.regs "v32" l
  · simp [hp]
  · simp [hp]

-- ── coopWindow probe: table byte-address = model tableOracle index ───────────

/-- The full-simp Nat form of the table byte-address reduces to `tbl + 2*(hash %
    2^hashLog)` — the `tableOracle` index.  Mask = mod (`and_two_pow_sub_one`), shl = *2. -/
theorem cw_addr_nat (Y hashLog tbl : Nat) (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40) :
    ((Y &&& (2 ^ hashLog - 1) % 2 ^ 64) <<< 1 + tbl) % 2 ^ 64 = tbl + 2 * (Y % 2 ^ hashLog) := by
  have hpow : (2:Nat) ^ hashLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hhl
  have h1 : 1 ≤ (2:Nat) ^ hashLog := Nat.one_le_two_pow
  have hm : (2 ^ hashLog - 1) % 2 ^ 64 = 2 ^ hashLog - 1 := by omega
  rw [hm, Nat.and_two_pow_sub_one_eq_mod, Nat.shiftLeft_eq]
  have hyb : Y % 2 ^ hashLog < 2 ^ hashLog := Nat.mod_lt _ (by omega)
  omega

-- ── coopWindow probe entry: posP / pValid / rp ───────────────────────────────

theorem cw_candRaw (prog : Array SInstr) (ss : SState) (l : Fin 32) (wl hashLog tbl : Nat)
    (hv : ss.regs "v32" l = UInt64.ofNat wl)
    (htbl : ss.regs "tbl" l = UInt64.ofNat tbl)
    (h0 : prog[ss.pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h1 : prog[ss.pc + 1]? = some (.bin .shr "hh" "hh" (.imm (32 - hashLog))))
    (h2 : prog[ss.pc + 2]? = some (.bin .band "hh" "hh" (.imm (2 ^ hashLog - 1))))
    (h3 : prog[ss.pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h4 : prog[ss.pc + 4]? = some (.binr .add "addr" "hh" "tbl"))
    (h5 : prog[ss.pc + 5]? = some (.ldsh "candRaw" "addr"))
    (hhl : hashLog ≤ 32) (htblb : tbl < 2 ^ 40) :
    ((snsteps prog 6 ss).regs "candRaw" l).toNat
      = (ss.smem.getD (tbl + 2 * (((UInt64.ofNat wl * UInt64.ofNat wHashK) >>> UInt64.ofNat (32 - hashLog)).toNat % 2 ^ hashLog)) 0).toNat + 256 * (ss.smem.getD (tbl + 2 * (((UInt64.ofNat wl * UInt64.ofNat wHashK) >>> UInt64.ofNat (32 - hashLog)).toNat % 2 ^ hashLog) + 1) 0).toNat := by
  simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, sstepInstr, SState.setReg, SState.setPc,
    SState.get, SOp.run, hv, htbl]
  have hpow : (18446744073709551616 : Nat) = 2 ^ 64 := by rfl
  rw [hpow, AlgorithmLib.LZ4WarpDSL.cw_addr_nat ((wl * wHashK % 2 ^ 64) >>> ((32 - hashLog) % 64)) hashLog tbl hhl htblb]
  have key : ∀ (x y : UInt8), (x.toNat + 256 * y.toNat) % 2 ^ 64 = x.toNat + 256 * y.toNat := by
    intro x y
    have hx := x.toNat_lt; have hy := y.toNat_lt
    exact Nat.mod_eq_of_lt (by omega)
  rw [key]

-- ── coopWindow 51-instr segment frame: regs + gmem preserved ─────────────────

theorem cwA (prog : Array SInstr) (ss : SState) (r : String) (searchLim capC : Nat)
    (h0 : prog[ss.pc]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[ss.pc + 1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[ss.pc + 2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[ss.pc + 3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[ss.pc + 4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[ss.pc + 5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[ss.pc + 6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[ss.pc + 7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[ss.pc + 8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[ss.pc + 9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[ss.pc + 10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[ss.pc + 11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[ss.pc + 12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[ss.pc + 13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[ss.pc + 14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (n_posP : r ≠ "posP") (n_pValid : r ≠ "pValid") (n_cap4 : r ≠ "cap4")
    (n_rp : r ≠ "rp") (n_rpA : r ≠ "rpA") (n_b0 : r ≠ "b0") (n_b1 : r ≠ "b1")
    (n_b2 : r ≠ "b2") (n_b3 : r ≠ "b3") (n_v32 : r ≠ "v32") :
    (snsteps prog 15 ss).pc = ss.pc + 15
    ∧ (snsteps prog 15 ss).gmem = ss.gmem
    ∧ (snsteps prog 15 ss).regs r = ss.regs r := by
  refine ⟨?_, ?_, ?_⟩
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get]
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get]
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run,
      n_posP, n_pValid, n_cap4, n_rp, n_rpA, n_b0, n_b1, n_b2, n_b3, n_v32]

-- ── Chunk B: steps 15..20 (6 instrs) ──────────────────────────────────────────
theorem cwB (prog : Array SInstr) (ss : SState) (r : String) (hashShift mask : Nat)
    (h0 : prog[ss.pc]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h1 : prog[ss.pc + 1]? = some (.bin .shr "hh" "hh" (.imm hashShift)))
    (h2 : prog[ss.pc + 2]? = some (.bin .band "hh" "hh" (.imm mask)))
    (h3 : prog[ss.pc + 3]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h4 : prog[ss.pc + 4]? = some (.binr .add "addr" "hh" "tbl"))
    (h5 : prog[ss.pc + 5]? = some (.ldsh "candRaw" "addr"))
    (n_hh : r ≠ "hh") (n_addr : r ≠ "addr") (n_candRaw : r ≠ "candRaw") :
    (snsteps prog 6 ss).pc = ss.pc + 6
    ∧ (snsteps prog 6 ss).gmem = ss.gmem
    ∧ (snsteps prog 6 ss).regs r = ss.regs r := by
  refine ⟨?_, ?_, ?_⟩
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5,
      sstepInstr, SState.setReg, SState.setPc, SState.get]
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5,
      sstepInstr, SState.setReg, SState.setPc, SState.get]
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run,
      n_hh, n_addr, n_candRaw]

-- ── Chunk C: steps 21..33 (13 instrs) ─────────────────────────────────────────
theorem cwC (prog : Array SInstr) (ss : SState) (r : String) (capC : Nat)
    (h0 : prog[ss.pc]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h1 : prog[ss.pc + 1]? = some (.binr .min "rc" "cand" "cap4"))
    (h2 : prog[ss.pc + 2]? = some (.binr .add "rcA" "inBase" "rc"))
    (h3 : prog[ss.pc + 3]? = some (.ldgo "c0" "rcA" 0))
    (h4 : prog[ss.pc + 4]? = some (.ldgo "c1" "rcA" 1))
    (h5 : prog[ss.pc + 5]? = some (.ldgo "c2" "rcA" 2))
    (h6 : prog[ss.pc + 6]? = some (.ldgo "c3" "rcA" 3))
    (h7 : prog[ss.pc + 7]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h8 : prog[ss.pc + 8]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h9 : prog[ss.pc + 9]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h10 : prog[ss.pc + 10]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h11 : prog[ss.pc + 11]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h12 : prog[ss.pc + 12]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (n_cand : r ≠ "cand") (n_rc : r ≠ "rc") (n_rcA : r ≠ "rcA") (n_c0 : r ≠ "c0")
    (n_c1 : r ≠ "c1") (n_c2 : r ≠ "c2") (n_c3 : r ≠ "c3") (n_vc : r ≠ "vc") :
    (snsteps prog 13 ss).pc = ss.pc + 13
    ∧ (snsteps prog 13 ss).gmem = ss.gmem
    ∧ (snsteps prog 13 ss).regs r = ss.regs r := by
  refine ⟨?_, ?_, ?_⟩
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12,
      sstepInstr, SState.setReg, SState.setPc, SState.get]
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12,
      sstepInstr, SState.setReg, SState.setPc, SState.get]
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run,
      n_cand, n_rc, n_rcA, n_c0, n_c1, n_c2, n_c3, n_vc]

-- ── Chunk D: steps 34..50 (17 instrs) ─────────────────────────────────────────
theorem cwD (prog : Array SInstr) (ss : SState) (r : String)
    (h0 : prog[ss.pc]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h1 : prog[ss.pc + 1]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h2 : prog[ss.pc + 2]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h3 : prog[ss.pc + 3]? = some (.andp "pH1" "pValid" "pNE"))
    (h4 : prog[ss.pc + 4]? = some (.andp "pH2" "pH1" "pCO"))
    (h5 : prog[ss.pc + 5]? = some (.andp "pHit" "pH2" "pEq"))
    (h6 : prog[ss.pc + 6]? = some (.vote "bal" "pHit"))
    (h7 : prog[ss.pc + 7]? = some (.brev "rev" "bal"))
    (h8 : prog[ss.pc + 8]? = some (.clz "fl" "rev"))
    (h9 : prog[ss.pc + 9]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h10 : prog[ss.pc + 10]? = some (.andp "pIns" "pLe" "pValid"))
    (h11 : prog[ss.pc + 11]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h12 : prog[ss.pc + 12]? = some (.stshp "pIns" "addr" "pp1"))
    (h13 : prog[ss.pc + 13]? = some (.barwarp))
    (h14 : prog[ss.pc + 14]? = some (.binr .add "p0" "searchPos" "fl"))
    (h15 : prog[ss.pc + 15]? = some (.shfl "cand0" "cand" "fl"))
    (h16 : prog[ss.pc + 16]? = some (.setp .ne "found" "bal" (.imm 0)))
    (n_pNE : r ≠ "pNE") (n_pCO : r ≠ "pCO") (n_pEq : r ≠ "pEq") (n_pH1 : r ≠ "pH1")
    (n_pH2 : r ≠ "pH2") (n_pHit : r ≠ "pHit") (n_bal : r ≠ "bal") (n_rev : r ≠ "rev")
    (n_fl : r ≠ "fl") (n_pLe : r ≠ "pLe") (n_pIns : r ≠ "pIns") (n_pp1 : r ≠ "pp1")
    (n_p0 : r ≠ "p0") (n_cand0 : r ≠ "cand0") (n_found : r ≠ "found") :
    (snsteps prog 17 ss).pc = ss.pc + 17
    ∧ (snsteps prog 17 ss).gmem = ss.gmem
    ∧ (snsteps prog 17 ss).regs r = ss.regs r := by
  refine ⟨?_, ?_, ?_⟩
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16,
      sstepInstr, SState.setReg, SState.setPc, SState.get]
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16,
      sstepInstr, SState.setReg, SState.setPc, SState.get]
  · simp [snsteps, sstep, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16,
      sstepInstr, SState.setReg, SState.setPc, SState.get, SOp.run, SCmp.run,
      n_pNE, n_pCO, n_pEq, n_pH1, n_pH2, n_pHit, n_bal, n_rev, n_fl, n_pLe, n_pIns,
      n_pp1, n_p0, n_cand0, n_found]

-- ── The full 51-step frame ────────────────────────────────────────────────────
theorem coopWindow_frame (prog : Array SInstr) (ss : SState) (base : Nat) (r : String)
    (searchLim capC hashShift mask : Nat)
    (hpc : ss.pc = base)
    (h0 : prog[base]? = some (.binr .add "posP" "searchPos" "lane"))
    (h1 : prog[base + 1]? = some (.setp .lt "pValid" "posP" (.imm searchLim)))
    (h2 : prog[base + 2]? = some (.mov "cap4" (.imm capC)))
    (h3 : prog[base + 3]? = some (.binr .min "rp" "posP" "cap4"))
    (h4 : prog[base + 4]? = some (.binr .add "rpA" "inBase" "rp"))
    (h5 : prog[base + 5]? = some (.ldgo "b0" "rpA" 0))
    (h6 : prog[base + 6]? = some (.ldgo "b1" "rpA" 1))
    (h7 : prog[base + 7]? = some (.ldgo "b2" "rpA" 2))
    (h8 : prog[base + 8]? = some (.ldgo "b3" "rpA" 3))
    (h9 : prog[base + 9]? = some (.bin .shl "b1" "b1" (.imm 8)))
    (h10 : prog[base + 10]? = some (.bin .shl "b2" "b2" (.imm 16)))
    (h11 : prog[base + 11]? = some (.bin .shl "b3" "b3" (.imm 24)))
    (h12 : prog[base + 12]? = some (.bin .bor "v32" "b0" (.reg "b1")))
    (h13 : prog[base + 13]? = some (.bin .bor "v32" "v32" (.reg "b2")))
    (h14 : prog[base + 14]? = some (.bin .bor "v32" "v32" (.reg "b3")))
    (h15 : prog[base + 15]? = some (.bin .mul "hh" "v32" (.imm wHashK)))
    (h16 : prog[base + 16]? = some (.bin .shr "hh" "hh" (.imm hashShift)))
    (h17 : prog[base + 17]? = some (.bin .band "hh" "hh" (.imm mask)))
    (h18 : prog[base + 18]? = some (.bin .shl "hh" "hh" (.imm 1)))
    (h19 : prog[base + 19]? = some (.binr .add "addr" "hh" "tbl"))
    (h20 : prog[base + 20]? = some (.ldsh "candRaw" "addr"))
    (h21 : prog[base + 21]? = some (.bin .sub "cand" "candRaw" (.imm 1)))
    (h22 : prog[base + 22]? = some (.binr .min "rc" "cand" "cap4"))
    (h23 : prog[base + 23]? = some (.binr .add "rcA" "inBase" "rc"))
    (h24 : prog[base + 24]? = some (.ldgo "c0" "rcA" 0))
    (h25 : prog[base + 25]? = some (.ldgo "c1" "rcA" 1))
    (h26 : prog[base + 26]? = some (.ldgo "c2" "rcA" 2))
    (h27 : prog[base + 27]? = some (.ldgo "c3" "rcA" 3))
    (h28 : prog[base + 28]? = some (.bin .shl "c1" "c1" (.imm 8)))
    (h29 : prog[base + 29]? = some (.bin .shl "c2" "c2" (.imm 16)))
    (h30 : prog[base + 30]? = some (.bin .shl "c3" "c3" (.imm 24)))
    (h31 : prog[base + 31]? = some (.bin .bor "vc" "c0" (.reg "c1")))
    (h32 : prog[base + 32]? = some (.bin .bor "vc" "vc" (.reg "c2")))
    (h33 : prog[base + 33]? = some (.bin .bor "vc" "vc" (.reg "c3")))
    (h34 : prog[base + 34]? = some (.setp .ne "pNE" "candRaw" (.imm 0)))
    (h35 : prog[base + 35]? = some (.setp .lt "pCO" "cand" (.reg "posP")))
    (h36 : prog[base + 36]? = some (.setp .eq "pEq" "vc" (.reg "v32")))
    (h37 : prog[base + 37]? = some (.andp "pH1" "pValid" "pNE"))
    (h38 : prog[base + 38]? = some (.andp "pH2" "pH1" "pCO"))
    (h39 : prog[base + 39]? = some (.andp "pHit" "pH2" "pEq"))
    (h40 : prog[base + 40]? = some (.vote "bal" "pHit"))
    (h41 : prog[base + 41]? = some (.brev "rev" "bal"))
    (h42 : prog[base + 42]? = some (.clz "fl" "rev"))
    (h43 : prog[base + 43]? = some (.setp .le "pLe" "lane" (.reg "fl")))
    (h44 : prog[base + 44]? = some (.andp "pIns" "pLe" "pValid"))
    (h45 : prog[base + 45]? = some (.bin .add "pp1" "posP" (.imm 1)))
    (h46 : prog[base + 46]? = some (.stshp "pIns" "addr" "pp1"))
    (h47 : prog[base + 47]? = some (.barwarp))
    (h48 : prog[base + 48]? = some (.binr .add "p0" "searchPos" "fl"))
    (h49 : prog[base + 49]? = some (.shfl "cand0" "cand" "fl"))
    (h50 : prog[base + 50]? = some (.setp .ne "found" "bal" (.imm 0)))
    (hr : r ∉ ["posP", "pValid", "cap4", "rp", "rpA", "b0", "b1", "b2", "b3", "v32",
               "hh", "addr", "candRaw", "cand", "rc", "rcA", "c0", "c1", "c2", "c3", "vc",
               "pNE", "pCO", "pEq", "pH1", "pH2", "pHit", "bal", "rev", "fl", "pLe",
               "pIns", "pp1", "p0", "cand0", "found"]) :
    (snsteps prog 51 ss).regs r = ss.regs r ∧ (snsteps prog 51 ss).gmem = ss.gmem := by
  subst hpc
  simp only [List.mem_cons, List.mem_singleton, not_or, List.not_mem_nil,
    not_false_iff, and_true] at hr
  obtain ⟨n_posP, n_pValid, n_cap4, n_rp, n_rpA, n_b0, n_b1, n_b2, n_b3, n_v32,
    n_hh, n_addr, n_candRaw, n_cand, n_rc, n_rcA, n_c0, n_c1, n_c2, n_c3, n_vc,
    n_pNE, n_pCO, n_pEq, n_pH1, n_pH2, n_pHit, n_bal, n_rev, n_fl, n_pLe,
    n_pIns, n_pp1, n_p0, n_cand0, n_found⟩ := hr
  -- Chunk A (steps 0..14)
  obtain ⟨pcA, gA, rfA⟩ := cwA prog ss r searchLim capC
    h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14
    n_posP n_pValid n_cap4 n_rp n_rpA n_b0 n_b1 n_b2 n_b3 n_v32
  -- Chunk B (steps 15..20), state = snsteps prog 15 ss, pc = ss.pc + 15
  obtain ⟨pcB, gB, rfB⟩ := cwB prog (snsteps prog 15 ss) r hashShift mask
    (by rw [pcA]; exact h15) (by rw [pcA]; exact h16) (by rw [pcA]; exact h17)
    (by rw [pcA]; exact h18) (by rw [pcA]; exact h19) (by rw [pcA]; exact h20)
    n_hh n_addr n_candRaw
  have pcB' : (snsteps prog 6 (snsteps prog 15 ss)).pc = ss.pc + 21 := by rw [pcB, pcA]
  -- Chunk C (steps 21..33), state = snsteps prog 6 (snsteps prog 15 ss), pc = ss.pc + 21
  obtain ⟨pcC, gC, rfC⟩ := cwC prog (snsteps prog 6 (snsteps prog 15 ss)) r capC
    (by rw [pcB']; exact h21) (by rw [pcB']; exact h22) (by rw [pcB']; exact h23)
    (by rw [pcB']; exact h24) (by rw [pcB']; exact h25) (by rw [pcB']; exact h26)
    (by rw [pcB']; exact h27) (by rw [pcB']; exact h28) (by rw [pcB']; exact h29)
    (by rw [pcB']; exact h30) (by rw [pcB']; exact h31) (by rw [pcB']; exact h32)
    (by rw [pcB']; exact h33)
    n_cand n_rc n_rcA n_c0 n_c1 n_c2 n_c3 n_vc
  have pcC' : (snsteps prog 13 (snsteps prog 6 (snsteps prog 15 ss))).pc = ss.pc + 34 := by
    rw [pcC, pcB']
  -- Chunk D (steps 34..50), state = snsteps prog 13 (...), pc = ss.pc + 34
  obtain ⟨pcD, gD, rfD⟩ := cwD prog (snsteps prog 13 (snsteps prog 6 (snsteps prog 15 ss))) r
    (by rw [pcC']; exact h34) (by rw [pcC']; exact h35) (by rw [pcC']; exact h36)
    (by rw [pcC']; exact h37) (by rw [pcC']; exact h38) (by rw [pcC']; exact h39)
    (by rw [pcC']; exact h40) (by rw [pcC']; exact h41) (by rw [pcC']; exact h42)
    (by rw [pcC']; exact h43) (by rw [pcC']; exact h44) (by rw [pcC']; exact h45)
    (by rw [pcC']; exact h46) (by rw [pcC']; exact h47) (by rw [pcC']; exact h48)
    (by rw [pcC']; exact h49) (by rw [pcC']; exact h50)
    n_pNE n_pCO n_pEq n_pH1 n_pH2 n_pHit n_bal n_rev n_fl n_pLe n_pIns
    n_pp1 n_p0 n_cand0 n_found
  -- Compose: 51 = 15 + 6 + 13 + 17
  have hchain : snsteps prog 51 ss
      = snsteps prog 17 (snsteps prog 13 (snsteps prog 6 (snsteps prog 15 ss))) := by
    rw [show (51 : Nat) = 15 + 36 by rfl, snsteps_add,
        show (36 : Nat) = 6 + 30 by rfl, snsteps_add,
        show (30 : Nat) = 13 + 17 by rfl, snsteps_add]
  refine ⟨?_, ?_⟩
  · rw [hchain, rfD, rfC, rfB, rfA]
  · rw [hchain, gD, gC, gB, gA]

-- ── Uniform emit-seq SimSL leaf template (reconciled .bin/.stg emit) ──────────
