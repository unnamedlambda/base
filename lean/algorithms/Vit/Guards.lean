module
public import Vit.Units
meta import Vit.Units
public import AlgorithmLib.ML.Model.Reorder
meta import AlgorithmLib.ML.Model.Reorder
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open AlgorithmLib AlgorithmLib.ML

/-!
  # What the shipped ViT tape is checked to be

  Separated from the generator for one reason: these guards evaluate the whole
  tape, and a module that a schedule experiment edits should not have to.
  `VitAlgorithm` builds without them; they build once, here.
-/

namespace Vit
open AlgorithmLib.IR

-- ---------------------------------------------------------------------------
-- Everything decided about the shipped tape, in one pass
-- ---------------------------------------------------------------------------

/-! A `native_decide` costs about ten seconds before it computes anything —
    measured, on a module with one trivial one and this import.  So the price of
    this file is the number of *sites*, not the size of what they decide, and
    what is decided is split across modules by *cost* rather than by subject:
    this one holds everything cheap, and the two expensive evaluations — every
    kernel's text (`Vit.Slot`) and every launch's body obligations
    (`Vit.Body`) — are modules of their own so they run beside it rather than
    after it, as are the two launch scans.  Every named claim below is a
    projection of the one evaluation this module performs. -/

open AlgorithmLib.ML in
/-- **The whole checked bill of the shipped tape**, decided once.

    Read the projections below rather than this; it is a conjunction only so
    that the generator runs once. -/
theorem vit_checked :
    (vKilled ≠ [])
      ∧ (VOUT ∉ vKilled)
      ∧ vPadTailUnwritten = true
      ∧ (vPadded ≠ [])
      ∧ vDepsSound = true
      ∧ AlgorithmLib.ML.VendorKernel.assumes vDeclaredKernel
          = [AlgorithmLib.ML.Law.cublasGemmIsSomeReassoc]
      ∧ AlgorithmLib.ML.VendorKernel.lawless vDeclaredKernel = false
      ∧ vDeclaredLaunches = 435
      ∧ vProvenLaunches = 806
      ∧ 0 < vDeclaredLaunches
      ∧ (VMASK_VAL.toBits).toNat = VMASK_BITS
      ∧ decide (VMASK_VAL ≤ AlgorithmLib.ML.softmaxFloor) = true
      ∧ NumOps.le VMASK_VAL (NumOps.neg (NumOps.ofNat 1000000000)) = true
      ∧ NumOps.le VBOUND (NumOps.ofNat 1000000) = true
      ∧ pull SQ (vFused.zipIdx.map fun p => (p.2, p.1)) vUnits.flatten = true := by native_decide

/-- **It is about a tape that was really fused.**  With no sites taken
    `vit_fusion_sound` is `rfl` and says nothing; this is the guard against
    reading a green build as evidence of a schedule that never fired. -/
theorem vit_fusion_fired : vKilled ≠ [] := vit_checked.1

/-- **…and it covers the buffer the answer is read from.**  `vit_fusion_sound`
    says nothing about a removed temporary, so a fusion that ate the logits
    would satisfy it and destroy the model. -/
theorem vit_out_survives : VOUT ∉ vKilled := vit_checked.2.1

/-- **Nothing on the tape writes a padded buffer's tail.** -/
theorem vit_pad_tail_unwritten : vPadTailUnwritten = true := vit_checked.2.2.1

/-- …and it is not vacuous: there are padded buffers to say it about. -/
theorem vit_pad_exists : vPadded ≠ [] := vit_checked.2.2.2.1

/-- **…and the dependence graph orders every conflicting pair.**  Build-enforced,
    so a scheduler change that drops an edge fails the build instead of
    producing a race that shows up as a wrong number once in a while. -/
theorem vit_deps_ordered : vDepsSound = true := vit_checked.2.2.2.2.1

/-- **The launches, in the order the schedule issues them, compute what the
    tape does**, on every buffer and at every carrier: each unit's operations
    folded unit after unit, `vUnits` in order, leave memory as the tape
    `vFused` folded in its own order does. Each operation the schedule takes
    next could have run first among what the tape had left, past only
    operations it cannot observe and that cannot observe it (`pull_tape`). -/
theorem vit_schedule_computes_tape {R : Type} [NumOps R] [NumLit R] (m : Buf → Nat → R) :
    (vUnits.flatMap vUnitOps).foldl (fun mm o => o.den mm) m = vFused.foldl (fun mm o => o.den mm) m := by
  have e : vUnits.flatMap vUnitOps = vUnits.flatten.filterMap (vFused[·]?) := by
    rw [List.filterMap_flatten]
    unfold List.flatMap
    congr 1
    apply List.map_congr_left
    intro u _
    simp [vUnitOps, vAllArr]
  rw [e]
  exact pull_tape SQ vFused vUnits.flatten vit_checked.2.2.2.2.2.2.2.2.2.2.2.2.2.2 m

/-- **…under any meaning of the operations that keeps the frame**: one where
    each writes only its output and reads only what it reads, such as one that
    reads each contraction as the vendor computes it. -/
theorem vit_schedule_computes_tape_with {R : Type} [NumOps R] [NumLit R] {S : TOp → (Buf → Nat → R) → Buf → Nat → R}
    (hS : TOp.Frame SQ S) (m : Buf → Nat → R) :
    (vUnits.flatMap vUnitOps).foldl (fun mm o => S o mm) m = vFused.foldl (fun mm o => S o mm) m := by
  have e : vUnits.flatMap vUnitOps = vUnits.flatten.filterMap (vFused[·]?) := by
    rw [List.filterMap_flatten]
    unfold List.flatMap
    congr 1
    apply List.map_congr_left
    intro u _
    simp [vUnitOps, vAllArr]
  rw [e]
  exact pull_tape_with hS vFused vUnits.flatten vit_checked.2.2.2.2.2.2.2.2.2.2.2.2.2.2 m

/-- Filtering by a predicate that implies another can be done through it. -/
theorem filter_of_sub {α : Type} (p q : α → Bool) (h : ∀ a, p a = true → q a = true) :
    ∀ l : List α, l.filter p = (l.filter q).filter p := by
  intro l
  induction l with
  | nil => rfl
  | cons a rest ih =>
      by_cases hp : p a = true
      · simp [hp, h a hp, ih]
      · simp only [Bool.not_eq_true] at hp
        by_cases hq : q a = true <;> simp [hp, hq, ih]

/-- A group that prints a kernel is a group whose operation is not a
    contraction — `vTextOf` returns the empty string for the others, which is
    the definition rather than a fact about the tape. -/
theorem vTextOf_proven (u : List Nat) (h : (vTextOf u != "") = true) :
    (((vUnitOps u).head? >>= TOp.vendorGemm).isNone) = true := by
  cases hh : (vUnitOps u).head? with
  | none => simp [vTextOf, hh] at h
  | some op =>
      cases hg : TOp.vendorGemm op with
      | none => simp [hg]
      | some g => simp [vTextOf, hh, hg] at h

/-- **The units the guards are stated over are the units that print.**  So
    nothing emitted is left out of `vit_body` and the rest — and it costs no
    emission to say so, which matters because saying it by evaluation would
    print every kernel in the model a second time. -/
theorem vit_proven_units_are_printed :
    vUnits.filter (fun u => vTextOf u != "")
      = vProvenUnits.filter (fun u => vTextOf u != "") :=
  filter_of_sub _ _ vTextOf_proven vUnits

/-- **What this artifact rests on, as numbers.**

    Not "cuBLAS is fine here".  Of 1241 launches, 435 are one primitive, and
    what that primitive may be assumed to be is one named law:
    `Law.cublasGemmIsSomeReassoc` — each output element sums its own `k`
    products, each once, in some association.

    The law is stated of a contraction at a batch of one, which is a
    *configuration* and not a symbol, so it is checked rather than asserted:
    `vit_fwd_contractions_are_plain_gemms` and its step counterpart decide, on
    the arguments recovered from the emitted program, that every contraction
    passes zero strides, zero offsets and a batch count of one.  Outside that
    configuration the operand slices come from arguments this model does not
    interpret, and `VendorKernel.withholds` says so.

    What this does *not* say: the law constrains the primitive, and
    `cublasGemmStep_isSomeReassoc` is where it reaches a plan step — but this
    artifact's tape is a list of `TOp`s, not a list of `DeclaredStep`s, so the
    law is not yet applied to *this* model's contractions.  That is the same
    missing instantiation `VitScan.notYetStated` names.

    The numbers are stated rather than counted at read time, so a schedule that
    quietly moved work onto the declared side fails the build. -/
theorem vit_law_bill :
    AlgorithmLib.ML.VendorKernel.assumes vDeclaredKernel
        = [AlgorithmLib.ML.Law.cublasGemmIsSomeReassoc]
      ∧ AlgorithmLib.ML.VendorKernel.lawless vDeclaredKernel = false
      ∧ vDeclaredLaunches = 435
      ∧ vProvenLaunches = 806 :=
  ⟨vit_checked.2.2.2.2.2.1, vit_checked.2.2.2.2.2.2.1,
   vit_checked.2.2.2.2.2.2.2.1, vit_checked.2.2.2.2.2.2.2.2.1⟩

/-- **…and it is a bill for something.**  A count of zero declared launches
    beside an empty law list would read the same way and mean the opposite. -/
theorem vit_law_bill_nonempty : 0 < vDeclaredLaunches :=
  vit_checked.2.2.2.2.2.2.2.2.2.1


-- ---------------------------------------------------------------------------
-- The padding mask
-- ---------------------------------------------------------------------------

/-- **The mask is the padding, and it is the artifact's own.**

    `vit_pad_tail_unwritten` says nothing writes a padded key's rows, so they
    are the zeros the buffer was allocated with.  What makes those zeros
    *harmless* rather than merely definite is this: every padded key scores the
    floor, so its exponential underflows and the value it would have been
    multiplied by never reaches the output.

    It is laid out into `initial_memory` from `TOK` and `SK` and uploaded from
    there, so the claim is about the bytes that ship. -/
theorem vit_mask_is_padding :
    vMaskWords.length = SK
      ∧ (∀ j, j < TOK → vMaskWords.getD j 0 = F32_ZERO)
      ∧ (∀ j, TOK ≤ j → j < SK → vMaskWords.getD j 0 = VMASK_BITS)
      ∧ TOK < SK := by
  refine ⟨by simp [vMaskWords], ?_, ?_, by decide⟩
  · intro j hj
    have hs : j < SK := Nat.lt_trans hj (by decide)
    simp [vMaskWords, List.getD, List.getElem?_map, List.getElem?_range, hs, hj]
  · intro j h1 h2
    simp [vMaskWords, List.getD, List.getElem?_map, List.getElem?_range, h2,
          Nat.not_lt.mpr h1]

/-- **…and the bytes are that float.**  The pattern in `initial_memory` is the
    one `VMASK_VAL` has, so the constant and the layout cannot drift. -/
theorem vit_mask_bits : (VMASK_VAL.toBits).toNat = VMASK_BITS := vit_checked.2.2.2.2.2.2.2.2.2.2.1

/-- **…and it is at or below the seed the row maximum starts from.**  A padded
    key therefore never *is* the maximum, which is what stops it shifting the
    whole row. -/
theorem vit_mask_below_floor :
    decide (VMASK_VAL ≤ AlgorithmLib.ML.softmaxFloor) = true := vit_checked.2.2.2.2.2.2.2.2.2.2.2.1

/-- **…and no host blob carries it.**  The mask's share of the host region is
    zero bytes, so a packing script has nothing to be wrong about. -/
theorem vit_mask_not_from_host : vHostBytesOf VMASK_BUF = 0 := by decide

/-- **…and the law reaches the artifact.**  A law nobody instantiates is a line
    in a registry; this is the check that `VMASK_VAL` and `VBOUND` are the
    values the theorem below needs, decided rather than asserted. -/
theorem vit_mask_law_applies :
    NumOps.le VMASK_VAL (NumOps.neg (NumOps.ofNat 1000000000)) = true
      ∧ NumOps.le VBOUND (NumOps.ofNat 1000000) = true :=
  ⟨vit_checked.2.2.2.2.2.2.2.2.2.2.2.2.1, vit_checked.2.2.2.2.2.2.2.2.2.2.2.2.2.1⟩

/-- **A padded key takes exactly zero softmax weight — at this artifact's own
    constants.**

    The composition of the two lane expressions the shipped kernels evaluate:
    `.exp (.add (.reg 1) (.neg (.reg 2)))`, which is the fused shift-and-
    exponentiate, and `.mul (.reg 1) (.inv (.reg 2))`, which is the scale by the
    row sum.  `raw` is the score before the mask pass added `VMASK_VAL` to it.

    The law is `Law.maskUnderflows` and it is in the type — that is the point.
    The side conditions on the *constants* are discharged here rather than
    assumed: `VMASK_VAL` really is below `-1e9`, and `VBOUND` really is at most
    `1e6`.  What remains hypothesis is only the range of the model's own values,
    which is what a bound on activations is. -/
theorem vit_padded_key_is_weightless
    (h : AlgorithmLib.ML.AllHold [AlgorithmLib.ML.Law.maskUnderflows])
    (raw mx s : Float32)
    (hr : NumOps.le raw VBOUND = true)
    (hm1 : NumOps.le (NumOps.neg VBOUND) mx = true)
    (hm2 : NumOps.le mx VBOUND = true)
    (hs1 : NumOps.le NumOps.one s = true)
    (hs2 : NumOps.le s (NumOps.ofNat 1000000) = true) :
    NumOps.mul
        (NumOps.exp (NumOps.add (NumOps.add raw VMASK_VAL) (NumOps.neg mx)))
        (NumOps.inv s)
      = NumOps.ofNat 0 :=
  h AlgorithmLib.ML.Law.maskUnderflows (by simp) VMASK_VAL VBOUND raw mx s
    vit_mask_law_applies.1 hr hm1 hm2 vit_mask_law_applies.2 hs1 hs2
end Vit
