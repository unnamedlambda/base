import AlgorithmLib.LZ4Refine

namespace AlgorithmLib.LZ4Plan

open AlgorithmLib
open AlgorithmLib.LZ4Imp

-- ---------------------------------------------------------------------------
-- 1. Plans: a positional segmentation of the input (no bytes carried)
-- ---------------------------------------------------------------------------

/-- One planned step: a literal run of `litLen` bytes, then a back-reference of
    `mlen` bytes at distance `offset`. Positions are into the INPUT. -/
structure PlanStep where
  litLen : Nat
  offset : Nat
  mlen   : Nat
  deriving Repr, DecidableEq

/-- A plan: a list of steps followed by a trailing literal run of `finalLen`. -/
structure Plan where
  steps    : List PlanStep
  finalLen : Nat
  deriving Repr, DecidableEq

/-- Running position advanced by all steps (Σ litLen + mlen). -/
def stepsLen : List PlanStep → Nat
  | [] => 0
  | st :: rest => st.litLen + st.mlen + stepsLen rest

-- ---------------------------------------------------------------------------
-- 2. The guard: `ValidPlan` (Prop) and the decidable `checkPlan` (Bool)
-- ---------------------------------------------------------------------------

/-- Per-step guard. The last conjunct is the *self-referential match agreement*
    on the input: every matched byte equals the byte `offset` back — overlap
    allowed. This is exactly what a compare-extend loop checks. -/
def stepAgree (inp : List UInt8) (pos : Nat) (st : PlanStep) : Prop :=
  1 ≤ st.offset ∧ st.offset ≤ 65535 ∧ 4 ≤ st.mlen ∧ st.offset ≤ pos + st.litLen ∧
    ∀ i, i < st.mlen →
      inp.getD (pos + st.litLen + i) 0 = inp.getD (pos + st.litLen + i - st.offset) 0

/-- Walk the position `pos` over the steps; require `stepAgree` at each, then
    that the final literal run exactly reaches the end of the input. -/
def ValidStepsFrom (inp : List UInt8) : Nat → List PlanStep → Nat → Prop
  | pos, [], finalLen => pos + finalLen = inp.length ∧ finalLen ≤ (inp.drop pos).length
  | pos, st :: rest, finalLen =>
      stepAgree inp pos st ∧
        ValidStepsFrom inp (pos + st.litLen + st.mlen) rest finalLen

/-- A plan is valid for an input iff the walk from position 0 succeeds. -/
def ValidPlan (inp : List UInt8) (p : Plan) : Prop :=
  ValidStepsFrom inp 0 p.steps p.finalLen

theorem ValidStepsFrom_total (inp : List UInt8) :
    ∀ (pos : Nat) (steps : List PlanStep) (fl : Nat),
    ValidStepsFrom inp pos steps fl → pos + stepsLen steps + fl = inp.length
  | pos, [], fl, h => by
      have := h.1; simp only [stepsLen]; omega
  | pos, st :: rest, fl, h => by
      obtain ⟨_, hrest⟩ := h
      have := ValidStepsFrom_total inp (pos + st.litLen + st.mlen) rest fl hrest
      simp only [stepsLen]; omega

-- ---------------------------------------------------------------------------
-- List helpers
-- ---------------------------------------------------------------------------

private theorem getD_take_lt : ∀ (l : List UInt8) (k j : Nat) (d : UInt8),
    j < k → (l.take k).getD j d = l.getD j d
  | [], _, _, _, _ => by simp
  | _ :: _, 0, _, _, hjk => by omega
  | _ :: _, _ + 1, 0, _, _ => by simp
  | _ :: l, k + 1, j + 1, d, hjk => by
      simp only [List.take_succ_cons, List.getD_cons_succ]
      exact getD_take_lt l k j d (by omega)

private theorem drop_take_one : ∀ (l : List UInt8) (p : Nat),
    p < l.length → (l.drop p).take 1 = [l.getD p 0]
  | _ :: _, 0, _ => by simp
  | _ :: l, p + 1, h => by
      simp only [List.drop_succ_cons, List.getD_cons_succ]
      exact drop_take_one l p (by simpa using h)
  | [], _, h => by simp at h

-- ---------------------------------------------------------------------------
-- 4. THE KEYSTONE lemma: the match-agreement guard makes `copyMatch`
--    reproduce the input's next bytes.
-- ---------------------------------------------------------------------------

/-- If the accumulator is the input prefix `inp.take p0`, the offset points back
    into it, and every matched byte agrees with the byte `off` back (the guard),
    then `copyMatch` extends the prefix to `inp.take (p0 + n)`. The overlap case
    (`off < n`) is handled: the freshly appended bytes are themselves consistent
    with the guard's self-reference, so the induction carries through. -/
theorem copyMatch_reproduces (inp : List UInt8) (off : Nat) (hoff1 : 1 ≤ off) :
    ∀ (n p0 : Nat) (acc : List UInt8),
    acc = inp.take p0 →
    (∀ i, i < n → inp.getD (p0 + i) 0 = inp.getD (p0 + i - off) 0) →
    off ≤ p0 → p0 + n ≤ inp.length →
    LZ4.copyMatch acc off n = inp.take (p0 + n)
  | 0, p0, acc, hacc, _, _, _ => by subst hacc; simp [LZ4.copyMatch]
  | n + 1, p0, acc, hacc, hguard, hoffp, hlen => by
      have hp0len : p0 ≤ inp.length := by omega
      have hp0lt : p0 < inp.length := by omega
      have hacclen : acc.length = p0 := by
        rw [hacc, List.length_take]; omega
      have hlt : p0 - off < p0 := by omega
      have hbyte : acc.getD (acc.length - off) 0 = inp.getD p0 0 := by
        rw [hacclen, hacc, getD_take_lt inp p0 (p0 - off) 0 hlt]
        have hg := hguard 0 (by omega)
        simp only [Nat.add_zero] at hg
        exact hg.symm
      rw [show LZ4.copyMatch acc off (n + 1)
            = LZ4.copyMatch (acc ++ [acc.getD (acc.length - off) 0]) off n from rfl, hbyte]
      have hnext : acc ++ [inp.getD p0 0] = inp.take (p0 + 1) := by
        rw [hacc, List.take_add, drop_take_one inp p0 hp0lt]
      rw [hnext]
      have hrec := copyMatch_reproduces inp off hoff1 n (p0 + 1) (inp.take (p0 + 1)) rfl
        (by intro i hi
            have hg := hguard (i + 1) (by omega)
            rw [show p0 + (i + 1) = (p0 + 1) + i from by omega] at hg
            exact hg)
        (by omega) (by omega)
      rw [hrec, show p0 + 1 + n = p0 + (n + 1) from by omega]

/-- One planned step, applied to the running input prefix, extends it. -/
theorem expandSeq_step (inp : List UInt8) (pos litLen off mlen : Nat)
    (hoff1 : 1 ≤ off) (hoffp : off ≤ pos + litLen)
    (hguard : ∀ i, i < mlen →
      inp.getD (pos + litLen + i) 0 = inp.getD (pos + litLen + i - off) 0)
    (hlen : pos + litLen + mlen ≤ inp.length) :
    LZ4.expandSeq (inp.take pos) ⟨(inp.drop pos).take litLen, off, mlen⟩
      = inp.take (pos + litLen + mlen) := by
  show LZ4.copyMatch (inp.take pos ++ (inp.drop pos).take litLen) off mlen = _
  rw [← List.take_add]
  exact copyMatch_reproduces inp off hoff1 mlen (pos + litLen) (inp.take (pos + litLen))
    rfl hguard hoffp (by omega)

-- ---------------------------------------------------------------------------
-- 3. `planToBlock`: slice the input into an `LZ4.Block`
-- ---------------------------------------------------------------------------

/-- Slice `inp` from running position `pos`: step `st` becomes a `Seq` whose
    literals are `(inp.drop pos).take st.litLen` and whose match is
    `(st.offset, st.mlen)`; the trailing literals are `(inp.drop pos').take fl`
    where `pos'` is the position after all steps. Carries no bytes of its own. -/
def planBlockFrom (inp : List UInt8) : Nat → List PlanStep → Nat → LZ4.Block
  | pos, [], finalLen => ⟨[], (inp.drop pos).take finalLen⟩
  | pos, st :: rest, finalLen =>
      ⟨⟨(inp.drop pos).take st.litLen, st.offset, st.mlen⟩ ::
          (planBlockFrom inp (pos + st.litLen + st.mlen) rest finalLen).seqs,
       (planBlockFrom inp (pos + st.litLen + st.mlen) rest finalLen).final⟩

def planToBlock (inp : List UInt8) (p : Plan) : LZ4.Block :=
  planBlockFrom inp 0 p.steps p.finalLen

theorem planBlockFrom_final (inp : List UInt8) :
    ∀ (steps : List PlanStep) (pos fl : Nat),
    (planBlockFrom inp pos steps fl).final = (inp.drop (pos + stepsLen steps)).take fl
  | [], pos, fl => by simp [planBlockFrom, stepsLen]
  | st :: rest, pos, fl => by
      show (planBlockFrom inp (pos + st.litLen + st.mlen) rest fl).final
            = (inp.drop (pos + stepsLen (st :: rest))).take fl
      have hidx : pos + st.litLen + st.mlen + stepsLen rest = pos + stepsLen (st :: rest) := by
        simp only [stepsLen]; omega
      rw [planBlockFrom_final inp rest (pos + st.litLen + st.mlen) fl, hidx]

/-- The running expansion over `planBlockFrom`'s sequences is the input prefix
    `inp.take (pos + stepsLen steps)` — the loop invariant. -/
theorem planBlockFrom_expand (inp : List UInt8) :
    ∀ (steps : List PlanStep) (pos fl : Nat),
    pos ≤ inp.length → ValidStepsFrom inp pos steps fl →
    List.foldl LZ4.expandSeq (inp.take pos) (planBlockFrom inp pos steps fl).seqs
      = inp.take (pos + stepsLen steps)
  | [], pos, fl, _, _ => by simp [planBlockFrom, stepsLen]
  | st :: rest, pos, fl, _, hv => by
      obtain ⟨hag, hvrest⟩ := hv
      obtain ⟨ho1, _, _, hoffp, hguard⟩ := hag
      have htot := ValidStepsFrom_total inp (pos + st.litLen + st.mlen) rest fl hvrest
      have hlenstep : pos + st.litLen + st.mlen ≤ inp.length := by omega
      have hstep := expandSeq_step inp pos st.litLen st.offset st.mlen ho1 hoffp hguard hlenstep
      have ih := planBlockFrom_expand inp rest (pos + st.litLen + st.mlen) fl hlenstep hvrest
      show List.foldl LZ4.expandSeq (inp.take pos)
            (⟨(inp.drop pos).take st.litLen, st.offset, st.mlen⟩ ::
              (planBlockFrom inp (pos + st.litLen + st.mlen) rest fl).seqs)
            = inp.take (pos + stepsLen (st :: rest))
      rw [List.foldl_cons, hstep, ih]
      have hidx : pos + st.litLen + st.mlen + stepsLen rest = pos + stepsLen (st :: rest) := by
        simp only [stepsLen]; omega
      rw [hidx]

theorem planBlockFrom_wf (inp : List UInt8) :
    ∀ (steps : List PlanStep) (pos fl : Nat),
    ValidStepsFrom inp pos steps fl → (planBlockFrom inp pos steps fl).WF
  | [], pos, fl, _ => by
      intro s hs; simp [planBlockFrom] at hs
  | st :: rest, pos, fl, hv => by
      obtain ⟨hag, hvrest⟩ := hv
      obtain ⟨ho1, ho2, hm4, _, _⟩ := hag
      intro s hs
      rw [show (planBlockFrom inp pos (st :: rest) fl).seqs
            = ⟨(inp.drop pos).take st.litLen, st.offset, st.mlen⟩ ::
                (planBlockFrom inp (pos + st.litLen + st.mlen) rest fl).seqs from rfl] at hs
      rcases List.mem_cons.mp hs with rfl | hs'
      · exact ⟨ho1, ho2, hm4⟩
      · exact planBlockFrom_wf inp rest (pos + st.litLen + st.mlen) fl hvrest s hs'

/-- Cumulative offset validity (`LZ4Imp.ValidFrom`): every match offset points
    into bytes already produced, because the running output length equals the
    running input position. -/
theorem planBlockFrom_validFrom (inp : List UInt8) :
    ∀ (steps : List PlanStep) (pos fl : Nat),
    pos ≤ inp.length → ValidStepsFrom inp pos steps fl →
    ValidFrom (inp.take pos) (planBlockFrom inp pos steps fl).seqs
  | [], pos, fl, _, _ => by simp [planBlockFrom, ValidFrom]
  | st :: rest, pos, fl, hpos, hv => by
      obtain ⟨hag, hvrest⟩ := hv
      obtain ⟨ho1, _, _, hoffp, hguard⟩ := hag
      have htot := ValidStepsFrom_total inp (pos + st.litLen + st.mlen) rest fl hvrest
      have hlenstep : pos + st.litLen + st.mlen ≤ inp.length := by omega
      have hstep := expandSeq_step inp pos st.litLen st.offset st.mlen ho1 hoffp hguard hlenstep
      have ih := planBlockFrom_validFrom inp rest (pos + st.litLen + st.mlen) fl hlenstep hvrest
      have hlitlen : ((inp.drop pos).take st.litLen).length = st.litLen := by
        rw [List.length_take, List.length_drop]; omega
      have htklen : (inp.take pos).length = pos := by
        rw [List.length_take]; omega
      rw [show (planBlockFrom inp pos (st :: rest) fl).seqs
            = ⟨(inp.drop pos).take st.litLen, st.offset, st.mlen⟩ ::
                (planBlockFrom inp (pos + st.litLen + st.mlen) rest fl).seqs from rfl]
      refine ⟨?_, ?_⟩
      · show st.offset ≤ (inp.take pos).length + ((inp.drop pos).take st.litLen).length
        rw [htklen, hlitlen]; exact hoffp
      · show ValidFrom (LZ4.expandSeq (inp.take pos)
              ⟨(inp.drop pos).take st.litLen, st.offset, st.mlen⟩)
              (planBlockFrom inp (pos + st.litLen + st.mlen) rest fl).seqs
        rw [hstep]; exact ih

-- ---------------------------------------------------------------------------
-- THE KEYSTONE theorem
-- ---------------------------------------------------------------------------

/-- **Keystone**: a valid plan slices the input into a block that is
    LZ4-well-formed and expands back to the input. The proof inducts over steps
    with invariant "the running expansion so far = `inp.take pos`", and its
    heart is `copyMatch_reproduces`. Nothing here mentions match-finding. -/
theorem planToBlock_sound (inp : List UInt8) (p : Plan) (h : ValidPlan inp p) :
    (planToBlock inp p).WF ∧ (planToBlock inp p).expand = inp := by
  refine ⟨planBlockFrom_wf inp p.steps 0 p.finalLen h, ?_⟩
  have hexp := planBlockFrom_expand inp p.steps 0 p.finalLen (by simp) h
  simp only [List.take_zero, Nat.zero_add] at hexp
  have hfin := planBlockFrom_final inp p.steps 0 p.finalLen
  simp only [Nat.zero_add] at hfin
  have htot := ValidStepsFrom_total inp 0 p.steps p.finalLen h
  simp only [Nat.zero_add] at htot
  unfold planToBlock LZ4.Block.expand
  rw [hexp, hfin]
  have hdroplen : (inp.drop (stepsLen p.steps)).length = p.finalLen := by
    rw [List.length_drop]; omega
  rw [show (inp.drop (stepsLen p.steps)).take p.finalLen = inp.drop (stepsLen p.steps)
        from by rw [← hdroplen, List.take_length]]
  exact List.take_append_drop (stepsLen p.steps) inp

theorem planToBlock_validFrom (inp : List UInt8) (p : Plan) (h : ValidPlan inp p) :
    ValidFrom [] (planToBlock inp p).seqs := by
  have := planBlockFrom_validFrom inp p.steps 0 p.finalLen (by simp) h
  simpa [planToBlock] using this

-- ---------------------------------------------------------------------------
-- 5. Corollaries: sound compressors and the executable roundtrip
-- ---------------------------------------------------------------------------

/-- Executable register-machine roundtrip for a valid plan: encode the sliced
    block and run the proven Lean decompressor — it returns the input. Uses the
    `ValidFrom` fact plus `LZ4Imp.decompress_encode`. -/
theorem plan_roundtrip (inp : List UInt8) (p : Plan) (h : ValidPlan inp p) :
    decompress (planToBlock inp p).encode inp.length = some inp := by
  have hsound := planToBlock_sound inp p h
  have hvalid := planToBlock_validFrom inp p h
  have hde := decompress_encode (planToBlock inp p) hsound.1 hvalid
  rw [hsound.2] at hde
  exact hde

-- ---------------------------------------------------------------------------
-- 6. Demos (native_decide is fine in `example`s — never in the theorem chain)
-- ---------------------------------------------------------------------------

/-- A self-overlapping plan on `[1,2,3]` repeated: 3 literals then a length-6
    match at offset 3 (overlap replicates the period). -/
example :
    (planToBlock ([1,2,3,1,2,3,1,2,3] : List UInt8) ⟨[⟨3,3,6⟩], 0⟩).expand
      = [1,2,3,1,2,3,1,2,3] := by native_decide

/-- A pure-literal plan (no steps, all trailing literals). -/
example : (planToBlock ([5,6,7] : List UInt8) ⟨[], 3⟩).expand = [5,6,7] := by native_decide

end AlgorithmLib.LZ4Plan
