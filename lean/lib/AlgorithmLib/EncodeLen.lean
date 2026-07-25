import AlgorithmLib.LZ4WarpKernel
import AlgorithmLib.LZ4Plan

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib.LZ4Plan AlgorithmLib.LZ4

/-- Telescoping (duplicated from ByteLayer.lean — cross-scratch-file `import`
    isn't resolvable via `lake env lean`; kept identical, zero risk). -/
theorem encode_planBlockFrom_cons (inp : List UInt8) (anchor : Nat)
    (step : PlanStep) (rest : List PlanStep) (fl : Nat) :
    (planBlockFrom inp anchor (step :: rest) fl).encode
      = encodeSeq ⟨(inp.drop anchor).take step.litLen, step.offset, step.mlen⟩
        ++ (planBlockFrom inp (anchor + step.litLen + step.mlen) rest fl).encode := by
  simp only [planBlockFrom, Block.encode, List.flatMap_cons, List.append_assoc]

theorem encode_planBlockFrom_nil (inp : List UInt8) (anchor fl : Nat) :
    (planBlockFrom inp anchor [] fl).encode
      = encodeFinal ((inp.drop anchor).take fl) := by
  simp only [planBlockFrom, Block.encode, List.flatMap_nil, List.nil_append]

/-- A `ValidStepsFrom` walk consumes exactly `stepsLen steps + fl` bytes from
    `anchor` to the end of `inp` — the length fact each `take` in
    `planBlockFrom` needs to not be silently truncated. -/
theorem ValidStepsFrom_sum (inp : List UInt8) :
    ∀ (anchor : Nat) (steps : List PlanStep) (fl : Nat),
      ValidStepsFrom inp anchor steps fl → anchor + stepsLen steps + fl = inp.length
  | anchor, [], fl, hv => by
      have h1 : anchor + fl = inp.length := hv.1
      simp only [stepsLen]; omega
  | anchor, step :: rest, fl, hv => by
      obtain ⟨_, hrest⟩ := hv
      have := ValidStepsFrom_sum inp (anchor + step.litLen + step.mlen) rest fl hrest
      simp only [stepsLen]; omega

/-- Byte-length of one match sequence's encoding — matches the `op`-advance
    RHS in `eval_wEmitMatchSeq` exactly (`1 + encNib(ll) + ll + 2 + encNib(ml-4)`). -/
def encSeqLen (s : PlanStep) : Nat :=
  1 + (encNib s.litLen).length + s.litLen + 2 + (encNib (s.mlen - 4)).length

/-- Byte-length of the final literal run's encoding — matches `op`-advance in
    `eval_wEmitFinalSeq` (`1 + encNib(fl) + fl`). -/
def encFinalLen (fl : Nat) : Nat :=
  1 + (encNib fl).length + fl

theorem encodeSeq_length (s : Seq) :
    (encodeSeq s).length = 1 + (encNib s.lits.length).length + s.lits.length
      + 2 + (encNib (s.mlen - 4)).length := by
  simp [encodeSeq, List.length_append, List.length_cons]
  omega

theorem encodeFinal_length (ls : List UInt8) :
    (encodeFinal ls).length = 1 + (encNib ls.length).length + ls.length := by
  simp [encodeFinal, List.length_append, List.length_cons]
  omega

/-- The plan-block's encoded length is the sum of each step's `encSeqLen` plus
    the final run's `encFinalLen`, for a `ValidStepsFrom` walk (so every `take`
    below is exactly its requested length, never silently truncated). -/
theorem encode_planBlockFrom_length (inp : List UInt8) :
    ∀ (anchor : Nat) (steps : List PlanStep) (fl : Nat),
      ValidStepsFrom inp anchor steps fl →
      (planBlockFrom inp anchor steps fl).encode.length
        = (steps.map encSeqLen).sum + encFinalLen fl
  | anchor, [], fl, hv => by
      rw [encode_planBlockFrom_nil, encodeFinal_length]
      have hsum := ValidStepsFrom_sum inp anchor [] fl hv
      simp only [stepsLen] at hsum
      have htake : ((inp.drop anchor).take fl).length = fl := by
        simp only [List.length_take, List.length_drop]; omega
      rw [htake]; simp [encFinalLen]
  | anchor, step :: rest, fl, hv => by
      have hsum := ValidStepsFrom_sum inp anchor (step :: rest) fl hv
      simp only [stepsLen] at hsum
      obtain ⟨hstep, hrest⟩ := hv
      rw [encode_planBlockFrom_cons, List.length_append,
        encode_planBlockFrom_length inp (anchor + step.litLen + step.mlen) rest fl hrest]
      have hlen : ((inp.drop anchor).take step.litLen).length = step.litLen := by
        simp only [List.length_take, List.length_drop]; omega
      rw [show (encodeSeq ⟨(inp.drop anchor).take step.litLen, step.offset, step.mlen⟩).length
            = 1 + (encNib step.litLen).length + step.litLen + 2
              + (encNib (step.mlen - 4)).length from by
        rw [encodeSeq_length]; simp [hlen]]
      simp only [List.map_cons, List.sum_cons, encSeqLen]
      omega

-- ── Tight encode bound: `255·encode ≤ 256·inp + 495` (i.e. `inp + inp/255 + 2`) ──
-- Stated multiplied through by 255 so all reasoning stays linear.

theorem encNib_len_small (n : Nat) (h : n < 15) : (encNib n).length = 0 := by
  simp [encNib, h]

theorem encNib_len_255 (n : Nat) : 255 * (encNib n).length ≤ n + 240 := by
  by_cases h : n < 15
  · rw [encNib_len_small n h]; omega
  · have hlen : (encNib n).length = (n - 15) / 255 + 1 := by
      simp [encNib, h, ext]
    rw [hlen]; omega

/-- One match sequence never expands: a match consumes `≥ 4` input bytes but the
    sequence costs only `3 + litLen` plus LSIC bytes, so `255·out ≤ 256·in`. -/
theorem encSeqLen_tight (s : PlanStep) (h : 4 ≤ s.mlen) :
    255 * encSeqLen s ≤ 256 * (s.litLen + s.mlen) := by
  have hL := encNib_len_255 s.litLen
  have hM := encNib_len_255 (s.mlen - 4)
  simp only [encSeqLen]
  by_cases hsmall : s.mlen - 4 < 15
  · have hM0 : (encNib (s.mlen - 4)).length = 0 := encNib_len_small _ hsmall
    omega
  · omega

theorem encFinalLen_tight (fl : Nat) : 255 * encFinalLen fl ≤ 256 * fl + 495 := by
  have h := encNib_len_255 fl
  simp only [encFinalLen]; omega

theorem encodeSum_tight (inp : List UInt8) :
    ∀ (anchor : Nat) (steps : List PlanStep) (fl : Nat),
      ValidStepsFrom inp anchor steps fl →
      255 * (steps.map encSeqLen).sum ≤ 256 * stepsLen steps
  | _, [], _, _ => by simp [stepsLen]
  | anchor, st :: rest, fl, hv => by
      obtain ⟨hag, hrest⟩ := hv
      have ih := encodeSum_tight inp (anchor + st.litLen + st.mlen) rest fl hrest
      have hs := encSeqLen_tight st hag.2.2.1
      simp only [List.map_cons, List.sum_cons, stepsLen]
      omega

/-- **Tight LZ4 worst case** for any valid plan: `255·|encode| ≤ 256·|inp| + 495`,
    i.e. `|encode| ≤ |inp| + |inp|/255 + 2`. -/
theorem planBlock_encode_tight (inp : List UInt8) (p : Plan) (hv : ValidPlan inp p) :
    255 * (planToBlock inp p).encode.length ≤ 256 * inp.length + 495 := by
  have hlen := encode_planBlockFrom_length inp 0 p.steps p.finalLen hv
  have hsum := encodeSum_tight inp 0 p.steps p.finalLen hv
  have hf := encFinalLen_tight p.finalLen
  have htot := ValidStepsFrom_sum inp 0 p.steps p.finalLen hv
  simp only [planToBlock]; rw [hlen]; omega

/-- The compressed block always fits the shipped output slot
    `lenOff = iS + iS/16 + 256` (with room: the true excess is `iS/255 + 2`). -/
theorem planBlock_encode_le_lenOff (inp : List UInt8) (p : Plan) (hv : ValidPlan inp p)
    (iS : Nat) (hlen : inp.length = iS) :
    (planToBlock inp p).encode.length ≤ iS + iS / 16 + 256 := by
  have h := planBlock_encode_tight inp p hv
  rw [hlen] at h
  omega

end AlgorithmLib.LZ4WarpDSL
