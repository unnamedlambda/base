import AlgorithmLib.LZ4CompTop
import AlgorithmLib.LZ4Confine

/-!
  # The output cursor never runs backwards

  `Lz4Sites.RegConfined`'s ten `sbAddr` stores reduce, via
  `Lz4Sites.sbAddr_is_outBase_add_op`, to a single inequality: `op ≤ lenOff` at
  every step.  The obvious route is "`op` is monotone, and it ends at
  `encode.length ≤ lenOff`, so every intermediate value is bounded" — and that
  looks circular, because monotonicity of a `UInt64` needs no-wraparound, which
  is what the bound would give.

  It is not circular.  No-overflow comes from the emit loop's OWN budget
  invariant (`outBase + op + 9*(inStride - anchor) < 2^64`), which is carried
  independently of the tight bound, together with the fact that what remains to
  be emitted from the literal anchor fits that budget
  (`planBlockFrom_encode_le9`).  And monotonicity itself needs no induction at
  all: `OuterByteLayer.emitLoop_eq` already concludes

      st'.regs "op" = ws.regs "op" + UInt64.ofNat (what is emitted from here).length

  i.e. *final = current + rest*, directly.  So the step below takes that equation
  as a hypothesis — exactly the shape the loop theorem hands you — and turns it
  into the inequality.
-/

namespace AlgorithmLib.LZ4WarpDSL
open AlgorithmLib AlgorithmLib.LZ4 AlgorithmLib.LZ4Simt AlgorithmLib.LZ4Plan

/-- The sequence bytes are a prefix of the block's encoding, which also carries
    the final literal run. -/
theorem seqsFlat_le_encode (b : LZ4.Block) :
    (b.seqs.flatMap LZ4.encodeSeq).length ≤ b.encode.length := by
  simp only [LZ4.Block.encode, List.length_append]
  omega

/-- **The cursor-advance step, with the plan abstracted away.**

    Anything whose effect on `op` is `op + ofNat L` — the whole emit loop, one
    iteration of it, one `wEmitMatchSeq`, one token byte — moves the cursor
    forward and by exactly `L`, provided `L` fits a budget the caller already
    holds.  Every emit lemma in `EvalValid` concludes in this shape, so the
    interior of an iteration is reached by the same lemma as the loop itself
    rather than by a second argument. -/
theorem op_le_of_add (ws st' : WState) (L B : Nat)
    (hop : st'.regs "op" = ws.regs "op" + UInt64.ofNat L)
    (hLB : L ≤ B) (hbud : (ws.regs "op").toNat + B < 2 ^ 64) :
    (ws.regs "op").toNat ≤ (st'.regs "op").toNat ∧
    (st'.regs "op").toNat = (ws.regs "op").toNat + L := by
  obtain ⟨heq, hmono⟩ :=
    AlgorithmLib.LZ4Simt.toNat_add_ofNat_of_lt (ws.regs "op") L (by omega)
  rw [hop]
  exact ⟨hmono, heq⟩

/-- **The emit loop cannot move the output cursor backwards.**

    Stated against the `op` equation `emitLoop_eq` produces rather than against
    the loop statement itself, so it costs no restatement of that theorem's ten
    hypotheses and applies at every anchor the loop passes through.

    With `op_final = encode.length ≤ lenOff` (`planBlock_encode_le_lenOff`, at
    anchor 0) this is what bounds `op` at every intermediate loop head. -/
theorem op_le_of_emitLoop (ws st' : WState) (inp : List UInt8) (anchor : Nat)
    (steps : List PlanStep) (fl : Nat)
    (hv : ValidStepsFrom inp anchor steps fl) (hal : anchor < inp.length)
    (hop : st'.regs "op" = ws.regs "op" + UInt64.ofNat
      ((planBlockFrom inp anchor steps fl).seqs.flatMap LZ4.encodeSeq).length)
    (hbud : (ws.regs "op").toNat + 9 * (inp.length - anchor) < 2 ^ 64) :
    (ws.regs "op").toNat ≤ (st'.regs "op").toNat ∧
    (st'.regs "op").toNat = (ws.regs "op").toNat
      + ((planBlockFrom inp anchor steps fl).seqs.flatMap LZ4.encodeSeq).length := by
  exact op_le_of_add ws st' _ (9 * (inp.length - anchor)) hop
    (Nat.le_trans (seqsFlat_le_encode _) (planBlockFrom_encode_le9 inp anchor steps fl hv hal))
    hbud

/-- The emit lemmas leave `op` as `a + 1 + ofNat n` (the token byte, then the
    LSIC bytes).  `op_le_of_add` wants `a + ofNat L`; this is the normalisation
    between them, and it holds unconditionally — `UInt64.ofNat_add` needs no
    bound because both sides wrap alike. -/
theorem add_one_ofNat (a : UInt64) (n : Nat) :
    a + 1 + UInt64.ofNat n = a + UInt64.ofNat (1 + n) := by
  rw [UInt64.ofNat_add, show (UInt64.ofNat 1) = 1 from rfl, UInt64.add_assoc]

/-- **`op_le_of_add` reaching inside an iteration.**  The literal-LSIC emit is
    the innermost place the cursor moves, and it moves it forward by exactly
    `1 + |encNib litLen|` — the token byte plus the length-extension bytes.

    Stated by composing `finalUifOut_op` with `add_one_ofNat` and
    `op_le_of_add`, which is the pattern every remaining emit statement follows:
    no new argument, just the statement's own byte count as the budget. -/
theorem op_le_finalUifOut (litLen : String) (fuel : Nat) (ws : WState) (B : Nat)
    (h1 : litLen ≠ "zero") (h2 : litLen ≠ "tokHi") (h3 : litLen ≠ "tok")
    (h4 : litLen ≠ "sbAddr") (h5 : litLen ≠ "op") (h6 : litLen ≠ "pLitBigF")
    (hfuel : ((ws.regs litLen).toNat - 15) / 255 < fuel)
    (hLB : 1 + (LZ4.encNib (ws.regs litLen).toNat).length ≤ B)
    (hbud : (ws.regs "op").toNat + B < 2 ^ 64) :
    (ws.regs "op").toNat ≤
      ((WStmt.eval fuel (.uif "pLitBigF"
        (wseq [ .bin .sub "litExtraF" litLen (.imm 15), wEmitLSIC "litExtraF" ]) .skip)
        (finalAfterSetp litLen fuel ws)).regs "op").toNat := by
  refine (op_le_of_add ws _ _ B ?_ hLB hbud).1
  rw [finalUifOut_op litLen fuel ws h1 h2 h3 h4 h5 h6 hfuel, add_one_ofNat]

/-- **The output cursor at a loop head is bounded by where the loop leaves it.**

    This is the eval-level half of `Lz4Sites.CursorAtSites.opLe`, and it needs no
    induction of its own.  `OuterByteLayer.emitLoop_eq` already concludes
    *final = current + rest* at an arbitrary loop head, so `op_le_of_emitLoop`
    turns that into the inequality once the addition is known not to wrap — and
    no-wraparound is the loop's OWN budget hypothesis `hnwFull`, which is carried
    independently of the tight bound.  That is what dissolves the apparent
    circularity recorded above.

    The hypotheses are `emitLoop_eq`'s, verbatim, plus `anchor < inStride`
    (`planBlockFrom_encode_le9` needs a non-empty remaining input; at
    `anchor = inStride` there is nothing left to emit and `op` cannot move).

    What is still missing to reach `opLe` itself is *not* this bound: it is
    (a) the same statement for the store sites that sit strictly inside an
    iteration, and (b) the machine/eval correspondence, which exists
    instruction-by-instruction inside `warpKernelDSL_sstep_roundtrips`' proof but
    is not exposed in its statement. -/
theorem emitLoop_head_op_le (inStride hashLog : Nat) (hstride : inStride ≤ 65536)
    (hp64 : inStride < 2 ^ 64)
    (F anchor s : Nat) (ws : WState)
    (has : anchor ≤ s) (hsval : (ws.regs "searchPos").toNat = s)
    (hlaval : (ws.regs "litAnchor").toNat = anchor)
    (hloopC : (ws.regs "loopC")
      = (if UInt64.ofNat s < UInt64.ofNat (inStride - 12) then 1 else 0))
    (hFsuff : ((inStride - 12) - s) + 33 * inStride ≤ F)
    (hnwFull : (ws.regs "outBase").toNat + (ws.regs "op").toNat
      + 9 * (inStride - anchor) < 2 ^ 64)
    (hsize : (ws.regs "outBase").toNat + (ws.regs "op").toNat
      + 9 * (inStride - anchor) ≤ ws.gmem.size)
    (hdisj : (ws.regs "inBase").toNat + inStride
      ≤ (ws.regs "outBase").toNat + (ws.regs "op").toNat)
    (hinBound : (ws.regs "inBase").toNat + inStride < 2 ^ 64)
    (hal : anchor < inStride) :
    (ws.regs "op").toNat
      ≤ (((WStmt.uwhile "loopC" (wseq
          [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
            .uif "found" (wseq
              [ .mov "ecR" (.imm (inStride - 5)), .mov "ec1" (.imm (inStride - 5 - 1)),
                .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
                .uwhile "extC" (wseq
                  [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride (inStride - 5),
                    .bin .add "ml" "ml" (.reg "adv"),
                    .setp .eq "extC" "adv" (.imm 32) ]),
                .bin .sub "off0" "p0" (.reg "cand0"),
                .bin .sub "litLen" "p0" (.reg "litAnchor"),
                wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
                .bin .add "litAnchor" "p0" (.reg "ml"),
                .mov "searchPos" (.reg "litAnchor") ])
              (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]),
            .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ])).eval F ws).regs "op").toNat := by
  have hlen : (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride).length = inStride := by
    simp only [gmemInpAt, List.length_map, List.length_range]
  obtain ⟨-, -, -, -, hop⟩ :=
    emitLoop_eq inStride hashLog hstride hp64 F anchor s ws has hsval hlaval hloopC hFsuff
      hnwFull hsize hdisj hinBound
  have hv := genLoop_valid
    (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
    (fun sm => tableOracle ws.gmem sm hashLog 0 (ws.regs "inBase").toNat)
    (fun sm sp upto => tableInsert sm ws.gmem hashLog 0 (inStride - 12)
      (ws.regs "inBase").toNat sp upto)
    (by rw [hlen]; exact hstride) F anchor s ws.smem has (by rw [hlen]; omega)
  rw [hlen] at hv
  exact (op_le_of_emitLoop ws _ _ anchor _ _ hv (by rw [hlen]; exact hal) hop
    (by rw [hlen]; omega)).1

/-- **One iteration cannot move the output cursor backwards either.**

    The companion to `emitLoop_head_op_le` one level down, and the reason the
    loop-head statement is not the end of the story: the ten `sbAddr` stores sit
    strictly *inside* an iteration, so a bound that only holds where the loop
    tests its guard never reaches them.

    Proved as part of `loopCBody_Qadvance` rather than separately: that theorem
    already case-splits on whether the window found a match and already computes
    the `op` equation in each branch for its budget clauses, so the bound is one
    `omega` per branch.  Doing it there also means it cannot drift from the
    `LoopCQ` invariant it travels with. -/
theorem loopCBody_op_le (inStride hashLog fuel : Nat) (ws : WState)
    (hstride : inStride ≤ 65536) (hipos : 12 ≤ inStride) (hp64 : inStride < 2 ^ 64)
    (hfuel : inStride ≤ fuel) (hQ : LoopCQ inStride ws)
    (hguard : (ws.regs "loopC" == 1) = true) :
    (ws.regs "op").toNat
      ≤ (((loopCBodyStmt inStride hashLog).eval fuel ws).regs "op").toNat :=
  (loopCBody_Qadvance inStride hashLog fuel ws hstride hipos hp64 hfuel hQ hguard).2.2

/-- **The bound at a loop head, in the form the descent consumes.**

    `op ≤ lenOff` at a loop head follows from the same bound at the loop's *exit*
    — the cursor only moves forward, so the exit value dominates every head value.
    That is the whole of the eval-level argument, and it is why no forward
    induction on `op ≤ lenOff` is needed (it is not preserved forward; the
    inductive statement is `op + remaining = op_final`, which is exactly what
    `emitLoop_eq` concludes).

    At the outermost call `op_final` is `encode.length`, bounded by
    `EncodeLen.planBlock_encode_le_lenOff`. -/
theorem emitLoop_head_op_le_of_final (inStride hashLog : Nat) (hstride : inStride ≤ 65536)
    (hp64 : inStride < 2 ^ 64)
    (F anchor s : Nat) (ws : WState) (lenOff : Nat)
    (has : anchor ≤ s) (hsval : (ws.regs "searchPos").toNat = s)
    (hlaval : (ws.regs "litAnchor").toNat = anchor)
    (hloopC : (ws.regs "loopC")
      = (if UInt64.ofNat s < UInt64.ofNat (inStride - 12) then 1 else 0))
    (hFsuff : ((inStride - 12) - s) + 33 * inStride ≤ F)
    (hnwFull : (ws.regs "outBase").toNat + (ws.regs "op").toNat
      + 9 * (inStride - anchor) < 2 ^ 64)
    (hsize : (ws.regs "outBase").toNat + (ws.regs "op").toNat
      + 9 * (inStride - anchor) ≤ ws.gmem.size)
    (hdisj : (ws.regs "inBase").toNat + inStride
      ≤ (ws.regs "outBase").toNat + (ws.regs "op").toNat)
    (hinBound : (ws.regs "inBase").toNat + inStride < 2 ^ 64)
    (hal : anchor < inStride)
    (hfinal : (((WStmt.uwhile "loopC" (wseq
          [ .coopWindow "found" "p0" "cand0" "searchPos" inStride (inStride - 12) hashLog 0,
            .uif "found" (wseq
              [ .mov "ecR" (.imm (inStride - 5)), .mov "ec1" (.imm (inStride - 5 - 1)),
                .mov "ml" (.imm 4), .setp .ge "extC" "ml" (.imm 0),
                .uwhile "extC" (wseq
                  [ .coopExtendStep "adv" "p0" "cand0" "ml" inStride (inStride - 5),
                    .bin .add "ml" "ml" (.reg "adv"),
                    .setp .eq "extC" "adv" (.imm 32) ]),
                .bin .sub "off0" "p0" (.reg "cand0"),
                .bin .sub "litLen" "p0" (.reg "litAnchor"),
                wEmitMatchSeq "litAnchor" "litLen" "off0" "ml",
                .bin .add "litAnchor" "p0" (.reg "ml"),
                .mov "searchPos" (.reg "litAnchor") ])
              (wseq [ .bin .add "searchPos" "searchPos" (.imm 32) ]),
            .setp .lt "loopC" "searchPos" (.imm (inStride - 12)) ])).eval F ws).regs "op").toNat
        ≤ lenOff) :
    (ws.regs "op").toNat ≤ lenOff :=
  Nat.le_trans
    (emitLoop_head_op_le inStride hashLog hstride hp64 F anchor s ws has hsval hlaval hloopC
      hFsuff hnwFull hsize hdisj hinBound hal)
    hfinal

-- ── Inside one token: the cursor at the emit points, not just at the ends ─────

/-- **The token byte.**  `matchAfterSetp` is the straight-line prefix of
    `wEmitMatchSeq` (sub `mlm`; min `tokLo`; `wEmitToken`; setp `pLitBig`), and
    the only thing in it that moves the cursor is the single token store.  This
    is the first of the ten `sbAddr` emit points, so it is the first place
    `CursorAtSites.opLe` has to hold. -/
theorem op_le_matchAfterSetp (litLen ml : String) (fuel : Nat) (ws : WState) (B : Nat)
    (hLB : 1 ≤ B) (hbud : (ws.regs "op").toNat + B < 2 ^ 64) :
    (ws.regs "op").toNat ≤ ((matchAfterSetp litLen ml fuel ws).regs "op").toNat ∧
    ((matchAfterSetp litLen ml fuel ws).regs "op").toNat = (ws.regs "op").toNat + 1 :=
  op_le_of_add ws _ 1 B
    (by rw [matchAfterSetp_op, show (UInt64.ofNat 1) = 1 from rfl]) hLB hbud

/-- **The token byte plus the literal-length extension.**  Same shape one level
    out: `matchUifOut_op` says the cursor lands on `op + 1 + |encNib litLen|`, so
    the `lsicL` stores (the loop at pcs 140/147) are bounded by the same budget
    the loop head already carries.  `B` is the caller's slack — at a loop head
    `LoopCQ` supplies `9 * (inStride - litAnchor)`. -/
theorem op_le_matchUifOut (litLen ml : String) (fuel : Nat) (ws : WState) (B : Nat)
    (hml : litLen ≠ "mlm") (htl : litLen ≠ "tokLo") (hth : litLen ≠ "tokHi") (htk : litLen ≠ "tok")
    (hsb : litLen ≠ "sbAddr") (hop : litLen ≠ "op") (hpb : litLen ≠ "pLitBig")
    (hfuel : ((ws.regs litLen).toNat - 15) / 255 < fuel)
    (hLB : 1 + (LZ4.encNib (ws.regs litLen).toNat).length ≤ B)
    (hbud : (ws.regs "op").toNat + B < 2 ^ 64) :
    (ws.regs "op").toNat
      ≤ ((WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
          (matchAfterSetp litLen ml fuel ws)).regs "op").toNat ∧
    ((WStmt.eval fuel (.uif "pLitBig"
            (wseq [ .bin .sub "litExtra" litLen (.imm 15), wEmitLSIC "litExtra" ]) .skip)
          (matchAfterSetp litLen ml fuel ws)).regs "op").toNat
      = (ws.regs "op").toNat + (1 + (LZ4.encNib (ws.regs litLen).toNat).length) :=
  op_le_of_add ws _ _ B
    (by rw [matchUifOut_op litLen ml fuel ws hml htl hth htk hsb hop hpb hfuel, add_one_ofNat])
    hLB hbud

end AlgorithmLib.LZ4WarpDSL
