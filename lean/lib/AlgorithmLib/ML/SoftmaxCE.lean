import AlgorithmLib.ML.Kernels

/-!
  # Softmax and the cross-entropy gradient, in one warp

  A classifier's last step is `dlogits = softmax(logits) − onehot(label)`.  In
  the CIFAR model it ran on the host, and the cost was not the arithmetic — a
  kilobyte each way — but the **synchronisation**: a mid-step download forces
  the device to drain before the backward pass can be enqueued, so the step
  cannot be a single run.

  It looked expensive to move because a softmax over an arbitrary width needs
  `chunkRemReduce`, whose chunk and remainder counts are metadata the host must
  compute and the kernel must be trusted to receive.  That is not the situation
  here.  A classifier pads its class axis to the warp width — the same padding
  that makes every other reduction in the model divide — so the row **is** one
  warp, the reduction is a single butterfly, and there is no loop and no
  metadata at all.

  `warpReduceMaxE` leaves the maximum in *every* lane, so no shared memory and
  no barrier is needed either.  What is left is nine instructions:

      lm  = logit + bias        (bias is 0 for a class, very negative for padding)
      m   = maxⱼ lm             (butterfly)
      p   = exp(lm − m)         (padding underflows to zero, by construction)
      S   = Σⱼ p                (butterfly)
      out = p/S − onehot

  The padding bias is what removes the mask: `exp` of a very negative number is
  zero, so a padded lane contributes nothing to `S` and its output is exactly
  `−onehot`, which the host sets to zero.  One buffer of `C` floats, uploaded
  once.
-/

namespace AlgorithmLib.ML

/-- `%fw0` accumulates and `%fw1` is the shuffle temporary, as every butterfly
    in this stack expects; `%fw2` takes loads; `%fw3` holds the stabilised
    logit across the max, `%fw4` the exponential across the sum, `%fw5` the
    result. -/
def SM_LM : Nat := 3
def SM_P : Nat := 4
def SM_OUT : Nat := 5

/-- The five-round sum, on the butterfly's own register pair. -/
def warpReduceSumE : EWStmt :=
  .seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
    (.seq (warpRoundE 2) (warpRoundE 1))))

/-- A straight-line sequence, as a list.  Nine nested `.seq`s is nine chances
    to miscount a parenthesis, and a miscount there is a kernel that still
    compiles. -/
def seqAll : List EWStmt → EWStmt
  | []      => .skip
  | x :: xs => .seq x (seqAll xs)

/-! The body is split at the two butterflies rather than written as one
    ten-statement block.  That is not cosmetic: a bulk `simp only [wrun_seq,
    wrun_setR]` unfolds a reduction network into individual register writes
    before `warpReduceSum_spec` can fire, and afterwards there is no pattern
    left to rewrite.  Keeping each butterfly as its own statement is what lets
    its spec and its frame lemma apply. -/

/-- Loads, the bias add, and the accumulator for the maximum. -/
def smPre (logits bias : Buf) (biasIx : IdxE) : EWStmt :=
  seqAll [ .loadIdx SM_LM logits elemIx
         , .loadIdx 2 bias biasIx
         , .setR SM_LM (.add (.reg SM_LM) (.reg 2))
         , .setR 0 (.reg SM_LM) ]

/-- The exponential, and the accumulator for the sum. -/
def smMid : EWStmt :=
  seqAll [ .setR SM_P (.exp (.add (.reg SM_LM) (.neg (.reg 0))))
         , .setR 0 (.reg SM_P) ]

/-- The normalisation and the label subtraction. -/
def smPost (oneHot : Buf) : EWStmt :=
  seqAll [ .loadIdx 2 oneHot elemIx
         , .setR SM_OUT (.add (.mul (.reg SM_P) (.inv (.reg 0))) (.neg (.reg 2))) ]

/-- Everything before the store. -/
def smBody (logits bias oneHot : Buf) (biasIx : IdxE) : EWStmt :=
  seqAll [ smPre logits bias biasIx, warpReduceMaxE, smMid, warpReduceSumE,
           smPost oneHot ]

/-- **Softmax and the cross-entropy gradient, one warp per row.**

    `bias` holds `0` at a real class and a large negative value at a padding
    lane; `oneHot` holds the label.  Both are read at the lane's own element,
    which is the addressing `elemIx` fixes — so this schema leaves the caller
    no obligation. -/
def softmaxCE (logits bias oneHot out : Buf) (biasIx : IdxE) : EWStmt :=
  .seq (smBody logits bias oneHot biasIx) (.storeLane out elemIx SM_OUT)

/-- The stabilised logit each lane holds: its own logit plus the bias. -/
def smLm (memL memB : Nat → Float32) (il ib : Lane → Nat) : Lane → Float32 :=
  fun l => NumOps.add (memL (il l)) (memB (ib l))

/-- **What `softmaxCE` computes**, in the committed order: the maximum and the
    sum are the *same* five-round butterfly tree every other reduction in this
    stack uses, so this is an exact `Float32` equality with no hypothesis. -/
def smSpec (memL memB memO : Nat → Float32) (il ib : Lane → Nat) : Lane → Float32 :=
  let lm := smLm memL memB il ib
  let m := bflyFoldOp (fun a b => NumOps.max a b) lm
  let p := fun l => NumOps.exp (NumOps.add (lm l) (NumOps.neg (m l)))
  let s := bflyFold p
  fun l => NumOps.add (NumOps.mul (p l) (NumOps.inv (s l)))
                      (NumOps.neg (memO (il l)))

/-- The sum butterfly, elaborated, is the network `warpReduceSum_spec` is about. -/
theorem warpReduceSumE_elab (cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) :
    warpReduceSumE.elabAt cta i ir im = warpReduceSum 0 1 := rfl

/-- The sum network's frame, at the shape `warpReduceSum` is written in.
    `warpReduceOp_frame` proves it; it is restated because `rw` matches on
    syntax and `warpRound` and `bflyRoundOp` are equal only definitionally. -/
theorem warpReduceSum_frame (acc tmp r : Nat) (hr : r ≠ acc) (hr' : r ≠ tmp)
    (st : WSt) : ((warpReduceSum acc tmp).run st).regs r = st.regs r :=
  warpReduceOp_frame acc tmp (.add (.reg acc) (.reg tmp)) r hr hr' st

/-- …and the maximum network's, likewise. -/
theorem warpReduceMaxE_frame (cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (r : Nat) (hr : r ≠ 0) (hr' : r ≠ 1) (st : WSt) :
    ((warpReduceMaxE.elabAt cta i ir im).run st).regs r = st.regs r :=
  warpReduceOp_frame 0 1 (.maxW (.reg 0) (.reg 1)) r hr hr' st

-- ---------------------------------------------------------------------------
-- What each piece leaves behind
-- ---------------------------------------------------------------------------

theorem smPre_mem (logits bias : Buf) (biasIx : IdxE) (cta i : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    (((smPre logits bias biasIx).elabAt cta i ir im).run st).mem = st.mem := rfl

theorem smPre_lm (logits bias : Buf) (biasIx : IdxE) (cta i : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    (((smPre logits bias biasIx).elabAt cta i ir im).run st).regs SM_LM
      = smLm (st.mem logits) (st.mem bias)
          (fun l => elemIx.eval cta i l ir im) (fun l => biasIx.eval cta i l ir im) := by
  funext l
  simp only [smPre, seqAll, EWStmt.elabAt, wrun_seq, wrun_skip, wrun_loadIdx, wrun_setR,
    WSt.regs_setReg_same, WSt.mem_setReg, WFExp.eval, smLm,
    WSt.regs_setReg_other _ 2 SM_LM _ (by decide : SM_LM ≠ 2),
    WSt.regs_setReg_other _ 0 SM_LM _ (by decide : SM_LM ≠ 0)]

theorem smPre_acc (logits bias : Buf) (biasIx : IdxE) (cta i : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    (((smPre logits bias biasIx).elabAt cta i ir im).run st).regs 0
      = smLm (st.mem logits) (st.mem bias)
          (fun l => elemIx.eval cta i l ir im) (fun l => biasIx.eval cta i l ir im) := by
  funext l
  simp only [smPre, seqAll, EWStmt.elabAt, wrun_seq, wrun_skip, wrun_loadIdx, wrun_setR,
    WSt.regs_setReg_same, WSt.mem_setReg, WFExp.eval, smLm,
    WSt.regs_setReg_other _ 2 SM_LM _ (by decide : SM_LM ≠ 2),
    WSt.regs_setReg_other _ 0 SM_LM _ (by decide : SM_LM ≠ 0)]

theorem smMid_mem (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) : ((smMid.elabAt cta i ir im).run st).mem = st.mem := rfl

theorem smMid_p (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) :
    ((smMid.elabAt cta i ir im).run st).regs SM_P
      = fun l => NumOps.exp (NumOps.add (st.regs SM_LM l) (NumOps.neg (st.regs 0 l))) := by
  funext l
  simp only [smMid, seqAll, EWStmt.elabAt, wrun_seq, wrun_skip, wrun_setR, WFExp.eval,
    WSt.regs_setReg_same, WSt.regs_setReg_other _ 0 SM_P _ (by decide : SM_P ≠ 0)]

theorem smMid_acc (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) :
    ((smMid.elabAt cta i ir im).run st).regs 0
      = fun l => NumOps.exp (NumOps.add (st.regs SM_LM l) (NumOps.neg (st.regs 0 l))) := by
  funext l
  simp only [smMid, seqAll, EWStmt.elabAt, wrun_seq, wrun_skip, wrun_setR, WFExp.eval,
    WSt.regs_setReg_same, WSt.regs_setReg_other _ SM_P SM_LM _ (by decide : SM_LM ≠ SM_P),
    WSt.regs_setReg_other _ SM_P 0 _ (by decide : (0:Nat) ≠ SM_P)]

theorem smPost_out (oneHot : Buf) (cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt) :
    (((smPost oneHot).elabAt cta i ir im).run st).regs SM_OUT
      = fun l => NumOps.add (NumOps.mul (st.regs SM_P l) (NumOps.inv (st.regs 0 l)))
                   (NumOps.neg (st.mem oneHot (elemIx.eval cta i l ir im))) := by
  funext l
  simp only [smPost, seqAll, EWStmt.elabAt, wrun_seq, wrun_skip, wrun_loadIdx, wrun_setR,
    WFExp.eval, WSt.regs_setReg_same, WSt.mem_setReg,
    WSt.regs_setReg_other _ 2 SM_P _ (by decide : SM_P ≠ 2),
    WSt.regs_setReg_other _ 2 0 _ (by decide : (0:Nat) ≠ 2)]

/-- **The body leaves the spec in `%fw5`.**

    Five steps, each a single lemma: the prefix, the maximum butterfly (spec on
    `%fw0`, frame on `%fw3`), the exponential, the sum butterfly (spec on
    `%fw0`, frame on `%fw4`), and the normalisation.  The two frames are the
    whole reason this needs no shared memory and no barrier. -/
theorem smBody_regs (logits bias oneHot : Buf) (biasIx : IdxE) (cta : Nat)
    (st : WSt) :
    (((smBody logits bias oneHot biasIx).elabIn cta).run st).regs SM_OUT
      = smSpec (st.mem logits) (st.mem bias) (st.mem oneHot)
          (fun l => elemIx.eval cta 0 l) (fun l => biasIx.eval cta 0 l) := by
  show (((smPost oneHot).elabAt cta 0 _ _).run
        ((warpReduceSumE.elabAt cta 0 _ _).run
          ((smMid.elabAt cta 0 _ _).run
            ((warpReduceMaxE.elabAt cta 0 _ _).run
              (((smPre logits bias biasIx).elabAt cta 0 _ _).run st))))).regs SM_OUT = _
  rw [smPost_out]
  -- the sum butterfly: spec on the accumulator, frame on the exponential
  rw [warpReduceSumE_elab, warpReduceSum_spec 0 1 (by decide),
      warpReduceSum_frame 0 1 SM_P (by decide) (by decide), warpReduceSum_mem]
  rw [smMid_p, smMid_acc, smMid_mem]
  -- the maximum butterfly: same shape
  rw [warpReduceMaxE_spec, warpReduceMaxE_frame cta 0 _ _ SM_LM (by decide) (by decide),
      warpReduceMaxE_mem]
  rw [smPre_lm, smPre_acc, smPre_mem]
  rfl

/-- **The body writes no memory** — it loads, reduces and computes in
    registers, and the one store is the schema's own `storeLane`.  The frame
    half of the stage contract. -/
theorem smBody_mem (logits bias oneHot : Buf) (biasIx : IdxE) (cta i : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    (((smBody logits bias oneHot biasIx).elabAt cta i ir im).run st).mem = st.mem := rfl

/-- **…and the store puts it at the lane's own address.**

    `elemIx` distinguishes lanes, so no two collide and the read-back is exact.
    Nothing reaches the caller. -/
theorem softmaxCE_stores (logits bias oneHot out : Buf) (biasIx : IdxE)
    (cta : Nat) (st : WSt) (l0 : Lane) :
    (((softmaxCE logits bias oneHot out biasIx).elabIn cta).run st).mem out
        (elemIx.eval cta 0 l0)
      = smSpec (st.mem logits) (st.mem bias) (st.mem oneHot)
          (fun l => elemIx.eval cta 0 l) (fun l => biasIx.eval cta 0 l) l0 := by
  have hinj : ∀ l l' : Lane, elemIx.eval cta 0 l = elemIx.eval cta 0 l' → l = l' := by
    intro l l' h
    have hv : cta * 32 + l.val = cta * 32 + l'.val := h
    exact Fin.ext (by omega)
  show ((WStmt.storeLane out (fun l => elemIx.eval cta 0 l) SM_OUT).run
      (((smBody logits bias oneHot biasIx).elabIn cta).run st)).mem out _ = _
  rw [storeLane_at _ _ _ _ l0 (fun l h => by rw [hinj l l0 h]),
      smBody_regs logits bias oneHot biasIx cta st]

end AlgorithmLib.ML
