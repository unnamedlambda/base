import AlgorithmLib.ML.LocalBind

/-!
# Which buffers a schema names

`EWStmt.bufBelow_of_bufsOf` turns a buffer bound into a question about a list.
This file answers that list for every kernel schema the vocabulary has, once —
so an operation's binding guard is *derived* from the buffers it declares
rather than decided at each operation in a tape.
-/

namespace AlgorithmLib.ML

/-- Address expressions the schemas build never load from memory. -/
theorem stride32_bufsOf (base : IdxE) (h : base.bufsOf = []) :
    (stride32 base).bufsOf = [] := by
  simp [stride32, IdxE.bufsOf, h]

/-- A statically unrolled batch names what one step names. -/
theorem seqN_bufsOf (S : List Buf) (f : Nat → EWStmt) (h : ∀ s, (f s).bufsOf ⊆ S) :
    ∀ n, (seqN n f).bufsOf ⊆ S := by
  intro n
  induction n with
  | zero => simp [seqN, EWStmt.bufsOf]
  | succ k ih =>
      simp only [seqN, EWStmt.bufsOf]
      exact List.append_subset.mpr ⟨ih, h k⟩

theorem warpRoundE_bufsOf (m : Nat) : (warpRoundE m).bufsOf = [] := by
  simp [warpRoundE, EWStmt.bufsOf]

theorem warpReduceE_bufsOf (acc tmp : Nat) : (warpReduceE acc tmp).bufsOf = [] := by
  simp [warpReduceE, bflyRoundE, EWStmt.bufsOf]

/-- **The strided reduction names its two operands and its output.** -/
theorem dotStrided_bufsOf (bA bB out : Buf) (ixA ixB oi : IdxE) (K : Nat)
    (hA : ixA.bufsOf = []) (hB : ixB.bufsOf = []) (ho : oi.bufsOf = []) :
    (dotStrided bA bB ixA ixB out oi K).bufsOf ⊆ [bA, bB, out] := by
  simp [dotStrided, dotStridedBody, EWStmt.bufsOf, warpRoundE_bufsOf, hA, hB, ho]

/-- **The batched reduction names the same three.** -/
theorem dotBatched_bufsOf (bA bB out : Buf) (ixA : IdxE) (ixB oi : Nat → IdxE)
    (B K : Nat) (hA : ixA.bufsOf = []) (hB : ∀ s, (ixB s).bufsOf = [])
    (ho : ∀ s, (oi s).bufsOf = []) :
    (dotBatched bA bB ixA ixB out oi B K).bufsOf ⊆ [bA, bB, out] := by
  simp only [dotBatched, batchBodyE, batchTripE, batchEpilogueE, EWStmt.bufsOf]
  refine List.append_subset.mpr ⟨List.append_subset.mpr ⟨?_, ?_⟩, ?_⟩
  · exact seqN_bufsOf _ _ (fun s => by simp [EWStmt.bufsOf]) B
  · refine List.append_subset.mpr ⟨by simp [hA], ?_⟩
    exact seqN_bufsOf _ _ (fun s => by simp [batchStepE, EWStmt.bufsOf, hB s]) B
  · exact seqN_bufsOf _ _
      (fun s => by simp [EWStmt.bufsOf, warpReduceE_bufsOf, ho s]) B

/-- **The batched outer product names the same three.** -/
theorem outerBatched_bufsOf (bA bB out : Buf) (ixA ixB : Nat → IdxE) (base : IdxE)
    (B K : Nat) (hA : ∀ s, (ixA s).bufsOf = []) (hB : ∀ s, (ixB s).bufsOf = [])
    (hbase : base.bufsOf = []) :
    (outerBatched bA bB out ixA ixB base B K).bufsOf ⊆ [bA, bB, out] := by
  simp only [outerBatched, storeBody, outerBodyE, EWStmt.bufsOf]
  refine List.append_subset.mpr ⟨List.append_subset.mpr ⟨by simp [EWStmt.bufsOf], ?_⟩, ?_⟩
  · exact seqN_bufsOf _ _
      (fun s => by simp [outerStepE, EWStmt.bufsOf, hA s, hB s]) B
  · simp [stride32, IdxE.bufsOf, hbase]

/-- **The strided maximum names its operand and its output.** -/
theorem maxStrided_bufsOf (b out : Buf) (ix oi : IdxE) (K : Nat) (init : Float32)
    (hix : ix.bufsOf = []) (ho : oi.bufsOf = []) :
    (maxStrided b ix out oi K init).bufsOf ⊆ [b, out] := by
  simp [maxStrided, maxStridedBody, warpReduceMaxE, warpMaxRoundE, EWStmt.bufsOf, hix, ho]

/-- **A two-buffer row pass names its two operands and its output.** -/
theorem zipPassEW_bufsOf (bA bB out : Buf) (dA dB r : Nat) (f : WFExp)
    (ixA ixB oix : IdxE) (K : Nat)
    (hA : ixA.bufsOf = []) (hB : ixB.bufsOf = []) (ho : oix.bufsOf = []) :
    (zipPassEW bA bB out dA dB r f ixA ixB oix K).bufsOf ⊆ [bA, bB, out] := by
  simp [zipPassEW, storeBody, EWStmt.bufsOf, hA, hB, ho]

/-- **…and a three-buffer pass names three.** -/
theorem zip3PassEW_bufsOf (bA bB bC out : Buf) (dA dB dC r : Nat) (f : WFExp)
    (ixA ixB ixC oix : IdxE) (K : Nat)
    (hA : ixA.bufsOf = []) (hB : ixB.bufsOf = []) (hC : ixC.bufsOf = [])
    (ho : oix.bufsOf = []) :
    (zip3PassEW bA bB bC out dA dB dC r f ixA ixB ixC oix K).bufsOf
      ⊆ [bA, bB, bC, out] := by
  simp [zip3PassEW, storeBody, EWStmt.bufsOf, hA, hB, hC, ho]

/-- **The softmax and cross-entropy pass names its three inputs and its
    output** — the butterflies in between touch no memory. -/
theorem softmaxCE_bufsOf (logits bias oneHot out : Buf) (biasIx : IdxE)
    (hb : biasIx.bufsOf = []) :
    (softmaxCE logits bias oneHot out biasIx).bufsOf ⊆ [logits, bias, oneHot, out] := by
  simp [softmaxCE, smBody, smPre, smMid, smPost, seqAll, warpReduceMaxE,
        warpMaxRoundE, warpReduceSumE, warpRoundE, bflyRoundE, elemIx,
        EWStmt.bufsOf, IdxE.bufsOf, hb]

/-- **Compiled expression code touches no memory.**  Register arithmetic and
    butterflies only: the loads happen before it. -/
theorem compileW_bufsOf : ∀ {Γ : Nat} (ve : Fin Γ → WFExp) (c : Nat) (e : Expr Γ),
    ((compileW ve c e).1).bufsOf = [] := by
  intro Γ ve c e
  induction e generalizing c with
  | var i => rfl
  | lit n => rfl
  | add a b iha ihb => simp [compileW, EWStmt.bufsOf, iha, ihb]
  | mul a b iha ihb => simp [compileW, EWStmt.bufsOf, iha, ihb]
  | neg a ih => simp [compileW, ih]
  | inv a ih => simp [compileW, ih]
  | exp a ih => simp [compileW, ih]
  | rsqrt a ih => simp [compileW, ih]
  | letE a b iha ihb => simp [compileW, EWStmt.bufsOf, iha, ihb]
  | sum n f ih =>
      simp only [compileW, sumSeq]
      have key : ∀ (L : List (EWStmt × WFExp)) (init : EWStmt),
          init.bufsOf = [] → (∀ p ∈ L, p.1.bufsOf = []) →
          (L.foldl (fun s (p : EWStmt × WFExp) =>
            .seq s (.seq p.1 (.setR c (.add (.reg c) p.2)))) init).bufsOf = [] := by
        intro L
        induction L with
        | nil => intro init hi _; exact hi
        | cons x xs ihL =>
            intro init hi hall
            exact ihL _ (by simp [EWStmt.bufsOf, hi, hall x (by simp)])
              (fun p hp => hall p (by simp [hp]))
      exact key _ _ rfl (fun p hp => by
        simp only [List.mem_map] at hp
        obtain ⟨j, _, hj⟩ := hp
        rw [← hj]; exact ih j _ _)

/-- **The load prologue names exactly the buffers it was given.** -/
theorem loadSeqN_bufsOf {Γ : Nat} (inB : Fin Γ → Buf) (inIx : Fin Γ → IdxE)
    (S : List Buf) (hS : ∀ i : Fin Γ, inB i ∈ S) (hix : ∀ i : Fin Γ, (inIx i).bufsOf = []) :
    ∀ k, (loadSeqN inB inIx k).bufsOf ⊆ S := by
  intro k
  induction k with
  | zero => simp [loadSeqN, EWStmt.bufsOf]
  | succ j ih =>
      simp only [loadSeqN, EWStmt.bufsOf]
      refine List.append_subset.mpr ⟨ih, ?_⟩
      by_cases h : j < Γ
      · simp only [dif_pos h, EWStmt.bufsOf, hix ⟨j, h⟩]
        intro b hb
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hb
        rw [hb]; exact hS ⟨j, h⟩
      · simp [dif_neg h, EWStmt.bufsOf]

/-- **A map kernel names its inputs and its output, whatever the expression.**

    The one that had to be an induction: the body is a compiled `Expr`, and the
    buffers are all in the prologue. -/
theorem mapKernelAt_bufsOf {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf)
    (inIx : Fin Γ → IdxE) (out : Buf) (S : List Buf)
    (hS : ∀ i : Fin Γ, inB i ∈ S) (hout : out ∈ S)
    (hix : ∀ i : Fin Γ, (inIx i).bufsOf = []) :
    ((mapKernelAt spec inB inIx out).ew).bufsOf ⊆ S := by
  simp only [mapKernelAt, compileWKernel, EWStmt.bufsOf]
  refine List.append_subset.mpr ⟨List.append_subset.mpr ⟨?_, ?_⟩, ?_⟩
  · exact loadSeqN_bufsOf inB inIx S hS hix Γ
  · simp [EWStmt.bufsOf, compileW_bufsOf]
  · intro b hb
    simp only [List.mem_cons, IdxE.bufsOf, elemIx, List.append_nil,
               List.not_mem_nil, or_false] at hb
    rw [hb]; exact hout

theorem mapKernel_bufsOf {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (S : List Buf) (hS : ∀ i : Fin Γ, inB i ∈ S) (hout : out ∈ S) :
    ((mapKernel spec inB out).ew).bufsOf ⊆ S :=
  mapKernelAt_bufsOf spec inB _ out S hS hout (fun _ => by simp [elemIx, IdxE.bufsOf])

theorem BCast_ix_bufsOf (m : BCast) : m.ix.bufsOf = [] := by
  cases m <;> simp [BCast.ix, stride32, IdxE.bufsOf]

/-- **Every operation names only the buffers it declares.**

    The whole point of the file: a tape's binding guard follows from this by
    `bufs_le_five` and `EWStmt.bufBelow_of_bufsOf`, at no cost per operation. -/
theorem TOp.stmt_reads (batch : Nat) (op : TOp) :
    (op.stmt batch).bufsOf ⊆ TOp.reads op ++ [(TOp.outSize batch op).1] := by
  cases op <;>
    simp only [TOp.stmt, TOp.reads, TOp.outSize, List.cons_append,
               List.nil_append]
  case mv _ w x o b inW outW =>
      refine dotBatched_bufsOf w x o _ _ _ b (inW / 32) ?_ ?_ ?_ <;>
        intros <;> simp [stride32, IdxE.bufsOf]
  case mvT _ w d o b inW outW =>
      refine dotBatched_bufsOf w d o _ _ _ b (outW / 32) ?_ ?_ ?_ <;>
        intros <;> simp [stride32, IdxE.bufsOf]
  case outer _ d x o b inW outW =>
      refine outerBatched_bufsOf d x o _ _ _ b (inW / 32) ?_ ?_ ?_ <;>
        intros <;> simp [stride32, IdxE.bufsOf]
  case ew1 f a o g => exact mapKernel_bufsOf f _ o _ (fun _ => by simp) (by simp)
  case ew2 f a b o g =>
      exact mapKernel_bufsOf f _ o _
        (fun i => by by_cases h : i.val = 0 <;> simp [h]) (by simp)
  case ew3 f a b c o g =>
      exact mapKernel_bufsOf f _ o _
        (fun i => by by_cases h : i.val = 0 <;> by_cases h1 : i.val = 1 <;> simp [h, h1])
        (by simp)
  case ew4 f a b c d o g =>
      exact mapKernel_bufsOf f _ o _
        (fun i => by
          by_cases h : i.val = 0 <;> by_cases h1 : i.val = 1 <;> by_cases h2 : i.val = 2 <;>
            simp [h, h1, h2])
        (by simp)
  case upd2 f a b g =>
      exact mapKernel_bufsOf f _ a _
        (fun i => by by_cases h : i.val = 0 <;> simp [h]) (by simp)
  case smce l bi oh o g =>
      refine softmaxCE_bufsOf l bi oh o _ ?_ ; simp [IdxE.bufsOf]
  case rowsq x o n r =>
      refine List.Subset.trans (dotStrided_bufsOf x x o _ _ _ (n / 32) ?_ ?_ ?_)
        (fun c hc => by simpa using hc) <;>
        simp [stride32, IdxE.bufsOf]
  case rowdot a b o mA mB n r =>
      refine dotStrided_bufsOf a b o _ _ _ (n / 32) ?_ ?_ ?_ <;>
        simp [BCast_ix_bufsOf, IdxE.bufsOf]
  case rowmax x o n r init =>
      refine maxStrided_bufsOf x o _ _ (n / 32) init ?_ ?_ <;>
        simp [stride32, IdxE.bufsOf]
  case ziprow a b o f mA mB n off w r =>
      refine zipPassEW_bufsOf a b o 1 2 0 f _ _ _ (w / 32) ?_ ?_ ?_ <;>
        simp [BCast_ix_bufsOf, stride32, IdxE.bufsOf]
  case ziprow3 a b c o f mA mB mC n off w r =>
      refine zip3PassEW_bufsOf a b c o 1 2 3 0 f _ _ _ _ (w / 32) ?_ ?_ ?_ ?_ <;>
        simp [BCast_ix_bufsOf, stride32, IdxE.bufsOf]

/-- **…and so only the buffers its own table holds.**

    The table is the same buffers with the repeat an in-place operation
    introduces removed, and removing a repeat removes no member. -/
theorem TOp.stmt_bufsOf (batch : Nat) (op : TOp) :
    (op.stmt batch).bufsOf ⊆ op.bufs batch := fun _ hc =>
  (mem_eraseDups _ _).mpr (TOp.stmt_reads batch op hc)

/-- Renaming renames the names. -/
theorem IdxE.bufsOf_renameBuf (f : Buf → Buf) : ∀ ix : IdxE,
    (ix.renameBuf f).bufsOf = ix.bufsOf.map f := by
  intro ix
  induction ix with
  | ldIdx b off ih => simp [IdxE.renameBuf, IdxE.bufsOf, ih]
  | add a b iha ihb => simp [IdxE.renameBuf, IdxE.bufsOf, iha, ihb]
  | mul a b iha ihb => simp [IdxE.renameBuf, IdxE.bufsOf, iha, ihb]
  | _ => rfl

theorem EWStmt.bufsOf_renameBuf (f : Buf → Buf) : ∀ s : EWStmt,
    (s.renameBuf f).bufsOf = s.bufsOf.map f := by
  intro s
  induction s with
  | seq a b iha ihb => simp [EWStmt.renameBuf, EWStmt.bufsOf, iha, ihb]
  | loadIdx d b ix => simp [EWStmt.renameBuf, EWStmt.bufsOf, IdxE.bufsOf_renameBuf]
  | loadV4 a b c d bu ix => simp [EWStmt.renameBuf, EWStmt.bufsOf, IdxE.bufsOf_renameBuf]
  | storeLane0 b ix r => simp [EWStmt.renameBuf, EWStmt.bufsOf, IdxE.bufsOf_renameBuf]
  | storeLane b ix r => simp [EWStmt.renameBuf, EWStmt.bufsOf, IdxE.bufsOf_renameBuf]
  | stSm ix r => simp [EWStmt.renameBuf, EWStmt.bufsOf, IdxE.bufsOf_renameBuf]
  | ldSm d ix => simp [EWStmt.renameBuf, EWStmt.bufsOf, IdxE.bufsOf_renameBuf]
  | forN k body ih => simp [EWStmt.renameBuf, EWStmt.bufsOf, ih]
  | forM bu a body ih => simp [EWStmt.renameBuf, EWStmt.bufsOf, ih]
  | cvtIF d ix => simp [EWStmt.renameBuf, EWStmt.bufsOf, IdxE.bufsOf_renameBuf]
  | _ => rfl

/-- A buffer in the table compacts to a slot in the table. -/
theorem compactMap_lt (bs : List Buf) (b : Buf) (h : b ∈ bs) :
    compactMap bs b < bs.length :=
  List.findIdx_lt_length_of_exists ⟨b, h, by simp⟩

/-- **Every kernel's buffers are inside its own table — for every operation,
    at no cost per operation.**

    What `qwen_bufs_bound` asks, answered once for all operations rather than
    decided per kernel: an operation names only the buffers it declares
    (`TOp.stmt_bufsOf`), and the compaction sends those to slots below the
    table's length. -/
theorem TOp.localBelow (batch : Nat) (op : TOp) : op.localBelowB batch = true := by
  simp only [TOp.localBelowB, decide_eq_true_eq, TOp.localStmt]
  refine EWStmt.bufBelow_of_bufsOf _ _ ?_
  intro b hb
  rw [EWStmt.bufsOf_renameBuf] at hb
  simp only [List.mem_map] at hb
  obtain ⟨c, hc, hcb⟩ := hb
  rw [← hcb]
  exact compactMap_lt _ c (TOp.stmt_bufsOf batch op hc)


end AlgorithmLib.ML
