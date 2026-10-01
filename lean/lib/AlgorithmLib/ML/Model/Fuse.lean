import AlgorithmLib.ML.Model.TenDenote

/-!
# Fusion as a tape rewrite

A schedule that fuses is not a second model.  `fuse_tape_den` says so once, for
any tape: replacing two row passes by the three-operand pass that computes their
composition leaves every buffer the tape produces unchanged, from any starting
memory, with no law invoked.

The hypothesis that carries the weight is `hpost` — nothing after the fusion may
read the buffer that held the intermediate.  A forward pass whose intermediate
is a saved activation the backward reads does not satisfy it, and that is the
case the check exists for.
-/

namespace AlgorithmLib.ML


/-- The buffers an operation reads. -/
def TOp.reads : TOp → List Buf
  | .mv _ w x _ _ _ _ _      => [w, x]
  | .mvT _ w d _ _ _ _       => [w, d]
  | .outer _ d x _ _ _ _     => [d, x]
  | .ew1 _ i _ _             => [i]
  | .ew2 _ i j _ _           => [i, j]
  | .ew3 _ i j k _ _         => [i, j, k]
  | .ew4 _ i j k n _ _       => [i, j, k, n]
  | .smce l bi oh _ _        => [l, bi, oh]
  | .upd2 _ i j _            => [i, j]
  | .rowsq x _ _ _           => [x]
  | .rowmax x _ _ _ _        => [x]
  | .rowdot i j _ _ _ _ _    => [i, j]
  | .ziprow a b _ _ _ _ _ _ _ _      => [a, b]
  | .ziprow3 a b c _ _ _ _ _ _ _ _ _ => [a, b, c]
  | .ziprow4 a b c d _ _ _ _ _ _ _ _ _ _ => [a, b, c, d]
  | .rowdot4 a b c d _ _ _ _ _ _ _ _ => [a, b, c, d]

/-- **An operation reads only `reads` and writes only its output.** -/
theorem den_congr (op : TOp) (m m' : Buf → Nat → Float32)
    (h : ∀ b ∈ op.reads, m b = m' b) (b : Buf) (a : Nat) (hb : m b a = m' b a) :
    (op.den m) b a = (op.den m') b a := by
  cases op <;>
    simp only [TOp.reads, List.mem_cons, List.not_mem_nil, or_false,
               forall_eq_or_imp, forall_eq] at h <;>
    simp only [TOp.den] <;>
    split <;>
    simp_all

/-- **An operation writes only its own output.** -/
theorem den_out (batch : Nat) (op : TOp) (m : Buf → Nat → Float32) (b : Buf)
    (hb : b ≠ (TOp.outSize batch op).1) : (op.den m) b = m b := by
  cases op <;>
    simp only [TOp.outSize] at hb <;>
    funext a <;>
    simp only [TOp.den] <;>
    split <;>
    simp_all

/-- **Two operations neither of which can observe the other commute.**

    The three conditions are the whole of it: they do not write the same buffer,
    and neither reads what the other writes.  Nothing about what either
    computes, so this holds for a contraction as much as for a row pass — which
    is what makes it usable to bring a producer and its consumer together
    without knowing why they are being brought together. -/
theorem den_swap (batch : Nat) (a b : TOp) (m : Buf → Nat → Float32)
    (hne : (TOp.outSize batch a).1 ≠ (TOp.outSize batch b).1)
    (hab : (TOp.outSize batch a).1 ∉ b.reads)
    (hba : (TOp.outSize batch b).1 ∉ a.reads) :
    b.den (a.den m) = a.den (b.den m) := by
  have hA : ∀ r ∈ a.reads, (b.den m) r = m r :=
    fun r hr => den_out batch b m r (fun e => hba (e ▸ hr))
  have hB : ∀ r ∈ b.reads, (a.den m) r = m r :=
    fun r hr => den_out batch a m r (fun e => hab (e ▸ hr))
  funext c x
  by_cases hca : c = (TOp.outSize batch a).1
  · subst hca
    rw [congrFun (den_out batch b (a.den m) _ hne) x]
    exact (den_congr a (b.den m) m hA _ x
      (congrFun (den_out batch b m _ hne) x)).symm
  · by_cases hcb : c = (TOp.outSize batch b).1
    · subst hcb
      rw [congrFun (den_out batch a (b.den m) _ (Ne.symm hne)) x]
      exact den_congr b (a.den m) m hB _ x
        (congrFun (den_out batch a m _ (Ne.symm hne)) x)
    · rw [congrFun (den_out batch b (a.den m) c hcb) x,
          congrFun (den_out batch a m c hca) x,
          congrFun (den_out batch a (b.den m) c hca) x,
          congrFun (den_out batch b m c hcb) x]

/-- **Agreement off a buffer survives a whole tape that never reads it.** -/
theorem foldl_den_frame (t : Buf) : ∀ (ops : List TOp) (m m' : Buf → Nat → Float32),
    (∀ op ∈ ops, t ∉ op.reads) → (∀ b, b ≠ t → m b = m' b) →
    ∀ b, b ≠ t → (ops.foldl (fun mm o => o.den mm) m) b
                = (ops.foldl (fun mm o => o.den mm) m') b := by
  intro ops
  induction ops with
  | nil => intro m m' _ hag b hb; exact hag b hb
  | cons o os ih =>
      intro m m' hr hag b hb
      refine ih (o.den m) (o.den m') (fun op hop => hr op (List.mem_cons_of_mem _ hop)) ?_ b hb
      intro c hc
      funext a
      refine den_congr o m m' (fun r hrr => ?_) c a (congrFun (hag c hc) a)
      have : r ≠ t := fun h => (hr o (List.mem_cons_self)) (h ▸ hrr)
      exact hag r this

/-- **The fused pair agrees with the two passes on every buffer but the
    temporary the fusion removes.** -/
theorem fuse_pair_frame (x ss g t out : Ref) (f1 f2 : WFExp)
    (h1 : f1.pairOnly = true) (h2 : f2.pairOnly = true)
    (mA mB mC : BCast) (nP offP nC offC w rows : Nat)
    (hP : offP + w ≤ nP) (hC : offC + w ≤ nC) (hnP : 0 < nP)
    (htg : g ≠ t) (hot : out ≠ t)
    (m : Buf → Nat → Float32) (b : Buf) (hb : b ≠ t) :
    ((TOp.ziprow t g out f2 (.rowOf nP offP) mC nC offC w rows).den
      ((TOp.ziprow x ss t f1 mA mB nP offP w rows).den m)) b
      = (TOp.ziprow3 x ss g out (f2.fuseA f1) mA mB mC nC offC w rows).den m b := by
  funext a
  by_cases ho : b = out
  · subst ho
    exact fuse_ziprow_den x ss g t b f1 f2 h1 h2 mA mB mC nP offP nC offC w rows
      hP hC hnP htg hot m a
  · simp only [TOp.den]
    rw [if_neg (fun hc => ho hc.1), if_neg (fun hc => hb hc.1),
        if_neg (fun hc => ho hc.1)]

/-- **Fusing a norm-shaped pair anywhere in a tape preserves what the tape
    computes.**

    The two row passes become one three-operand pass, and the buffer that held
    the intermediate is never written.  Every other buffer — in particular the
    one the tape produces — carries the same value, from any starting memory,
    with no law invoked: the pair is bit-exact by `fuse_ziprow_den` and the rest
    of the tape cannot tell the difference because it never reads the
    temporary.

    Generic in the surrounding tape, so this is proven once for every model
    rather than per schedule. -/
theorem fuse_tape_den (pre post : List TOp) (x ss g t out : Ref) (f1 f2 : WFExp)
    (h1 : f1.pairOnly = true) (h2 : f2.pairOnly = true)
    (mA mB mC : BCast) (nP offP nC offC w rows : Nat)
    (hP : offP + w ≤ nP) (hC : offC + w ≤ nC) (hnP : 0 < nP)
    (htg : g ≠ t) (hot : out ≠ t)
    (hpost : ∀ op ∈ post, t ∉ op.reads)
    (m : Buf → Nat → Float32) (b : Buf) (hb : b ≠ t) :
    ((pre ++ [TOp.ziprow x ss t f1 mA mB nP offP w rows,
              TOp.ziprow t g out f2 (.rowOf nP offP) mC nC offC w rows] ++ post).foldl
        (fun mm o => o.den mm) m) b
      = ((pre ++ [TOp.ziprow3 x ss g out (f2.fuseA f1) mA mB mC nC offC w rows]
            ++ post).foldl (fun mm o => o.den mm) m) b := by
  simp only [List.append_assoc, List.foldl_append, List.foldl_cons, List.foldl_nil]
  exact foldl_den_frame t post _ _ hpost
    (fun c hc => fuse_pair_frame x ss g t out f1 f2 h1 h2 mA mB mC nP offP nC offC w rows
      hP hC hnP htg hot _ c hc) b hb


/-- **The fused triple agrees with the two passes on every buffer but the
    temporary the fusion removes.** -/
theorem fuse_triple_frame (x ss y g t out : Ref) (f1 f2 : WFExp)
    (h1 : f1.tripleOnly = true) (h2 : f2.pairOnly = true)
    (mA mB mC mD : BCast) (nP offP nC offC w rows : Nat)
    (hP : offP + w ≤ nP) (hC : offC + w ≤ nC) (hnP : 0 < nP)
    (htg : g ≠ t) (hot : out ≠ t)
    (m : Buf → Nat → Float32) (b : Buf) (hb : b ≠ t) :
    ((TOp.ziprow t g out f2 (.rowOf nP offP) mD nC offC w rows).den
      ((TOp.ziprow3 x ss y t f1 mA mB mC nP offP w rows).den m)) b
      = (TOp.ziprow4 x ss y g out (f2.fuseA4 f1) mA mB mC mD nC offC w rows).den m b := by
  funext a
  by_cases ho : b = out
  · subst ho
    exact fuse_ziprow3_den x ss y g t b f1 f2 h1 h2 mA mB mC mD nP offP nC offC w rows
      hP hC hnP htg hot m a
  · simp only [TOp.den]
    rw [if_neg (fun hc => ho hc.1), if_neg (fun hc => hb hc.1),
        if_neg (fun hc => ho hc.1)]

/-- **Fusing a three-into-one chain anywhere in a tape preserves what the tape
    computes** — `fuse_tape_den` one arity up, and generic in the surrounding
    tape for the same reason. -/
theorem fuse_tape3_den (pre post : List TOp) (x ss y g t out : Ref) (f1 f2 : WFExp)
    (h1 : f1.tripleOnly = true) (h2 : f2.pairOnly = true)
    (mA mB mC mD : BCast) (nP offP nC offC w rows : Nat)
    (hP : offP + w ≤ nP) (hC : offC + w ≤ nC) (hnP : 0 < nP)
    (htg : g ≠ t) (hot : out ≠ t)
    (hpost : ∀ op ∈ post, t ∉ op.reads)
    (m : Buf → Nat → Float32) (b : Buf) (hb : b ≠ t) :
    ((pre ++ [TOp.ziprow3 x ss y t f1 mA mB mC nP offP w rows,
              TOp.ziprow t g out f2 (.rowOf nP offP) mD nC offC w rows] ++ post).foldl
        (fun mm o => o.den mm) m) b
      = ((pre ++ [TOp.ziprow4 x ss y g out (f2.fuseA4 f1) mA mB mC mD nC offC w rows]
            ++ post).foldl (fun mm o => o.den mm) m) b := by
  simp only [List.append_assoc, List.foldl_append, List.foldl_cons, List.foldl_nil]
  exact foldl_den_frame t post _ _ hpost
    (fun c hc => fuse_triple_frame x ss y g t out f1 f2 h1 h2 mA mB mC mD nP offP nC offC
      w rows hP hC hnP htg hot _ c hc) b hb

/-- **The fused reduction agrees with the pair on every buffer but the
    temporary the fusion removes.** -/
theorem fuse_rowdot_frame (x ss y g t out : Ref) (f1 : WFExp)
    (h1 : f1.tripleOnly = true) (mA mB mC mD : BCast) (nP offP n rows : Nat)
    (hP : offP + n ≤ nP) (hnP : 0 < nP) (hK : (n / 32) * 32 = n)
    (htg : g ≠ t) (hot : out ≠ t)
    (m : Buf → Nat → Float32) (b : Buf) (hb : b ≠ t) :
    ((TOp.rowdot t g out (.rowOf nP offP) mD n rows).den
      ((TOp.ziprow3 x ss y t f1 mA mB mC nP offP n rows).den m)) b
      = (TOp.rowdot4 x ss y g out f1 mA mB mC mD n rows).den m b := by
  funext a
  by_cases ho : b = out
  · subst ho
    exact fuse_rowdot_den x ss y g t b f1 h1 mA mB mC mD nP offP n rows
      hP hnP hK htg hot m a
  · simp only [TOp.den]
    rw [if_neg (fun hc => ho hc.1), if_neg (fun hc => hb hc.1),
        if_neg (fun hc => ho hc.1)]

/-- **Fusing a row pass into the reduction that consumes it, anywhere in a
    tape, preserves what the tape computes.** -/
theorem fuse_tapeR_den (pre post : List TOp) (x ss y g t out : Ref) (f1 : WFExp)
    (h1 : f1.tripleOnly = true) (mA mB mC mD : BCast) (nP offP n rows : Nat)
    (hP : offP + n ≤ nP) (hnP : 0 < nP) (hK : (n / 32) * 32 = n)
    (htg : g ≠ t) (hot : out ≠ t)
    (hpost : ∀ op ∈ post, t ∉ op.reads)
    (m : Buf → Nat → Float32) (b : Buf) (hb : b ≠ t) :
    ((pre ++ [TOp.ziprow3 x ss y t f1 mA mB mC nP offP n rows,
              TOp.rowdot t g out (.rowOf nP offP) mD n rows] ++ post).foldl
        (fun mm o => o.den mm) m) b
      = ((pre ++ [TOp.rowdot4 x ss y g out f1 mA mB mC mD n rows]
            ++ post).foldl (fun mm o => o.den mm) m) b := by
  simp only [List.append_assoc, List.foldl_append, List.foldl_cons, List.foldl_nil]
  exact foldl_den_frame t post _ _ hpost
    (fun c hc => fuse_rowdot_frame x ss y g t out f1 h1 mA mB mC mD nP offP n rows
      hP hnP hK htg hot _ c hc) b hb

/-! ### Orienting a consumer toward what the operation before it wrote

    `fuseNormAt` consumes the produced value in the *first* operand slot.  On a
    real tape the value lands in the second about as often — 134 adjacent pairs
    against 110 on the ViT step — and a second fusion lemma for that case would
    double the shapes the guard has to know.

    Commuting is the cheaper route and adds no shape: a row pass reading
    `(a, b)` through `f` is the same pass reading `(b, a)` through `f.swap12`.
    Orienting the tape first turns every second-slot site into a first-slot one,
    and the fusion guard is left alone. -/

/-- The operation with its operands commuted, when the value it consumes sits in
    the second slot.  Anything else is returned as it stands. -/
def TOp.orient (tmp : Buf) : TOp → TOp
  | .ziprow a b o f mA mB n off w rows =>
      if b == tmp && a != tmp && f.pairOnly then
        .ziprow b a o f.swap12 mB mA n off w rows
      else .ziprow a b o f mA mB n off w rows
  | op => op

/-- **Commuting a row pass changes nothing it computes.** -/
theorem TOp.orient_den (tmp : Buf) (op : TOp) (m : Buf → Nat → Float32) :
    (op.orient tmp).den m = op.den m := by
  cases op with
  | ziprow a b o f mA mB n off w rows =>
      simp only [TOp.orient]
      split
      · rename_i hc
        have hf : f.pairOnly = true := ((Bool.and_eq_true ..).mp hc).2
        funext b' x
        simp only [TOp.den]
        split
        · exact WFExp.swap12_evalPair f hf _ _
        · rfl
      · rfl
  | _ => rfl

/-- **Orient every operation toward the one before it.**

    One pass, left to right, so an operation already oriented by an earlier step
    is what the next step reads. -/
def orientTape (outOf : TOp → Buf) : List TOp → List TOp
  | []             => []
  | [x]            => [x]
  | a :: b :: rest => a :: orientTape outOf ((b.orient (outOf a)) :: rest)
  termination_by t => t.length

/-- **Orienting a tape preserves what it computes**, on every buffer — unlike
    fusion, which is allowed to stop writing the temporary it removes. -/
theorem orientTape_den (outOf : TOp → Buf) : ∀ (t : List TOp) (m : Buf → Nat → Float32),
    (orientTape outOf t).foldl (fun mm o => o.den mm) m
      = t.foldl (fun mm o => o.den mm) m := by
  intro t
  induction t using orientTape.induct (outOf := outOf) with
  | case1 => intro m; simp only [orientTape]
  | case2 x => intro m; simp only [orientTape]
  | case3 a b rest ih =>
      intro m
      simp only [orientTape, List.foldl_cons]
      rw [ih (a.den m)]
      simp only [List.foldl_cons, TOp.orient_den]

/-- **Fuse the row-pass pair at a named index.**

    A schedule says *where*; the library checks the site is fusable and refuses
    otherwise.  `none` is a schedule that named a position no fusion applies to
    — not a silent no-op.  The returned buffer is the intermediate the fusion
    removes, which is what the denotation theorem quantifies away. -/
def fuseNormAt (i : Nat) (t : List TOp) : Option (Buf × List TOp) :=
  match t[i]?, t[i+1]? with
  | some (.ziprow x ss tmp f1 mA mB nP offP w rows),
    some (.ziprow tmp2 g out f2 mP mC nC offC w2 rows2) =>
      if tmp2 = tmp ∧ w2 = w ∧ rows2 = rows ∧ mP = BCast.rowOf nP offP
          ∧ f1.pairOnly = true ∧ f2.pairOnly = true
          ∧ ((t.drop (i + 2)).all (fun o => !(TOp.reads o).contains tmp)) = true
          ∧ g ≠ tmp ∧ out ≠ tmp ∧ offP + w ≤ nP ∧ offC + w ≤ nC ∧ 0 < nP then
        some (tmp, t.take i
          ++ [TOp.ziprow3 x ss g out (f2.fuseA f1) mA mB mC nC offC w rows]
          ++ t.drop (i + 2))
      else none
  | some (.ziprow3 x ss y tmp f1 mA mB mC nP offP w rows),
    some (.ziprow tmp2 g out f2 mP mD nC offC w2 rows2) =>
      if tmp2 = tmp ∧ w2 = w ∧ rows2 = rows ∧ mP = BCast.rowOf nP offP
          ∧ f1.tripleOnly = true ∧ f2.pairOnly = true
          ∧ ((t.drop (i + 2)).all (fun o => !(TOp.reads o).contains tmp)) = true
          ∧ g ≠ tmp ∧ out ≠ tmp ∧ offP + w ≤ nP ∧ offC + w ≤ nC ∧ 0 < nP then
        some (tmp, t.take i
          ++ [TOp.ziprow4 x ss y g out (f2.fuseA4 f1) mA mB mC mD nC offC w rows]
          ++ t.drop (i + 2))
      else none
  | some (.ziprow3 x ss y tmp f1 mA mB mC nP offP w rows),
    some (.rowdot tmp2 g out mP mD n rows2) =>
      if tmp2 = tmp ∧ n = w ∧ rows2 = rows ∧ mP = BCast.rowOf nP offP
          ∧ f1.tripleOnly = true
          ∧ ((t.drop (i + 2)).all (fun o => !(TOp.reads o).contains tmp)) = true
          ∧ g ≠ tmp ∧ out ≠ tmp ∧ offP + w ≤ nP ∧ 0 < nP ∧ (w / 32) * 32 = w then
        some (tmp, t.take i
          ++ [TOp.rowdot4 x ss y g out f1 mA mB mC mD w rows]
          ++ t.drop (i + 2))
      else none
  | _, _ => none

/-- A tape splits at any position two elements exist at. -/
theorem split_at_pair : ∀ (t : List TOp) (i : Nat) (a b : TOp),
    t[i]? = some a → t[i+1]? = some b →
    t = t.take i ++ [a, b] ++ t.drop (i + 2) := by
  intro t
  induction t with
  | nil => intro i a b h; simp at h
  | cons x xs ih =>
      intro i a b ha hb
      cases i with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at ha
          subst ha
          simp only [List.take_zero, List.nil_append, List.cons_append]
          congr 1
          simp only [Nat.zero_add, List.getElem?_cons_succ] at hb
          cases xs with
          | nil => simp at hb
          | cons y ys =>
              simp only [List.getElem?_cons_zero, Option.some.injEq] at hb
              subst hb; rfl
      | succ j =>
          simp only [List.getElem?_cons_succ] at ha hb
          simp only [List.take_succ_cons, List.cons_append, List.drop_succ_cons]
          exact congrArg _ (ih j a b ha (by simpa using hb))

/-- **A fusion the library accepted preserves what the tape computes.**

    Every side condition `fuse_tape_den` needs is decided inside `fuseNormAt`,
    so a schedule that names a site gets this with no proof text: if the fusion
    was accepted at all, it was accepted because the conditions hold. -/
theorem fuseNormAt_den (i : Nat) (t t' : List TOp) (tmp : Buf)
    (h : fuseNormAt i t = some (tmp, t'))
    (m : Buf → Nat → Float32) (b : Buf) (hb : b ≠ tmp) :
    (t'.foldl (fun mm o => o.den mm) m) b
      = (t.foldl (fun mm o => o.den mm) m) b := by
  unfold fuseNormAt at h
  split at h
  · rename_i x ss tmp0 f1 mA mB nP offP w rows tmp2 g out f2 mP mC nC offC w2 rows2 ha hb2
    split at h
    · rename_i hc
      obtain ⟨e1, e2, e3, e4, h1, h2, hlive, htg, hot, hP, hC, hnP⟩ := hc
      subst e1; subst e2; subst e3; subst e4
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨htmp, ht'⟩ := h
      subst htmp; subst ht'
      refine Eq.trans (fuse_tape_den (t.take i) (t.drop (i + 2)) x ss g _ out f1 f2 h1 h2
        mA mB mC nP offP nC offC _ _ hP hC hnP htg hot
        (fun o ho => by
          have := List.all_eq_true.mp hlive o ho
          simpa using this) m b hb).symm ?_
      exact (congrArg (fun l => List.foldl (fun mm o => TOp.den o mm) m l b)
        (split_at_pair t i _ _ ha hb2)).symm
    · exact absurd h (by simp)
  · rename_i x ss y tmp0 f1 mA mB mC nP offP w rows tmp2 g out f2 mP mD nC offC w2 rows2
      ha hb2
    split at h
    · rename_i hc
      obtain ⟨e1, e2, e3, e4, h1, h2, hlive, htg, hot, hP, hC, hnP⟩ := hc
      subst e1; subst e2; subst e3; subst e4
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨htmp, ht'⟩ := h
      subst htmp; subst ht'
      refine Eq.trans (fuse_tape3_den (t.take i) (t.drop (i + 2)) x ss y g _ out f1 f2
        h1 h2 mA mB mC mD nP offP nC offC _ _ hP hC hnP htg hot
        (fun o ho => by
          have := List.all_eq_true.mp hlive o ho
          simpa using this) m b hb).symm ?_
      exact (congrArg (fun l => List.foldl (fun mm o => TOp.den o mm) m l b)
        (split_at_pair t i _ _ ha hb2)).symm
    · exact absurd h (by simp)
  · rename_i x ss y tmp0 f1 mA mB mC nP offP w rows tmp2 g out mP mD n rows2 ha hb2
    split at h
    · rename_i hc
      obtain ⟨e1, e2, e3, e4, h1, hlive, htg, hot, hP, hnP, hK⟩ := hc
      subst e1; subst e2; subst e3; subst e4
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨htmp, ht'⟩ := h
      subst htmp; subst ht'
      refine Eq.trans (fuse_tapeR_den (t.take i) (t.drop (i + 2)) x ss y g _ out f1
        h1 mA mB mC mD nP offP _ _ hP hnP hK htg hot
        (fun o ho => by
          have := List.all_eq_true.mp hlive o ho
          simpa using this) m b hb).symm ?_
      exact (congrArg (fun l => List.foldl (fun mm o => TOp.den o mm) m l b)
        (split_at_pair t i _ _ ha hb2)).symm
    · exact absurd h (by simp)
  · exact absurd h (by simp)

/-! ### Bringing a producer and its consumer together

    `fuseNormAt` fuses a pair only where it is *adjacent*, which is a fact about
    the order the tape happens to be written in rather than about the model.  On
    the ViT step 23 MiB of intermediates sit between a producer and a consumer
    the guard would otherwise accept, with unrelated operations in between.

    Exchanging those is not a fusion and needs none of a fusion's reasoning: two
    operations that cannot observe each other commute, by `den_swap`, and the
    result computes the same thing on *every* buffer — not merely on every
    buffer but one. -/

/-- Exchange the operations at `i` and `i+1`, when neither can observe the
    other.  `none` is a pair that cannot be exchanged, never a silent no-op. -/
def swapAt (batch i : Nat) (t : List TOp) : Option (List TOp) :=
  match t[i]?, t[i+1]? with
  | some a, some b =>
      if (TOp.outSize batch a).1 ≠ (TOp.outSize batch b).1
          ∧ (TOp.outSize batch a).1 ∉ b.reads
          ∧ (TOp.outSize batch b).1 ∉ a.reads then
        some (t.take i ++ [b, a] ++ t.drop (i + 2))
      else none
  | _, _ => none

/-- **An exchange the library accepted changes nothing the tape computes** —
    on every buffer, since nothing was removed. -/
theorem swapAt_den (batch i : Nat) (t t' : List TOp) (h : swapAt batch i t = some t')
    (m : Buf → Nat → Float32) :
    t'.foldl (fun mm o => o.den mm) m = t.foldl (fun mm o => o.den mm) m := by
  unfold swapAt at h
  split at h
  · rename_i a b ha hb
    split at h
    · rename_i hc
      simp only [Option.some.injEq] at h
      subst h
      refine Eq.trans ?_ (congrArg (List.foldl (fun mm o => TOp.den o mm) m)
        (split_at_pair t i a b ha hb).symm)
      simp only [List.append_assoc, List.foldl_append, List.foldl_cons, List.foldl_nil]
      exact congrArg
        (fun mm => List.foldl (fun mm o => TOp.den o mm) mm (List.drop (i + 2) t))
        (den_swap batch a b _ hc.1 hc.2.1 hc.2.2).symm
    · exact absurd h (by simp)
  · exact absurd h (by simp)

/-- Bring the operation at `j` up to sit immediately after position `i`, by
    adjacent exchanges.  One refusal abandons the whole move: a partial hoist
    would reorder the tape without buying the fusion it was for.

    `fuel` bounds the recursion so the definition is structural; `hoistTo` passes
    `j`, which is more steps than the move can take. -/
def bubbleTo (batch i : Nat) : Nat → Nat → List TOp → Option (List TOp)
  | 0,        _, t => some t
  | fuel + 1, j, t =>
      if i + 1 < j then
        match swapAt batch (j - 1) t with
        | some t' => bubbleTo batch i fuel (j - 1) t'
        | none    => none
      else some t

/-- **…and a hoist is a composition of exchanges, so it too changes nothing.** -/
theorem bubbleTo_den (batch i : Nat) : ∀ (fuel j : Nat) (t t' : List TOp),
    bubbleTo batch i fuel j t = some t' → ∀ (m : Buf → Nat → Float32),
    t'.foldl (fun mm o => o.den mm) m = t.foldl (fun mm o => o.den mm) m := by
  intro fuel
  induction fuel with
  | zero =>
      intro j t t' ht m
      simp only [bubbleTo, Option.some.injEq] at ht
      subst ht; rfl
  | succ fuel ih =>
      intro j t t' ht m
      simp only [bubbleTo] at ht
      split at ht
      · cases hs : swapAt batch (j - 1) t with
        | none => rw [hs] at ht; exact absurd ht (by simp)
        | some t'' =>
            rw [hs] at ht
            exact (ih (j - 1) t'' t' ht m).trans (swapAt_den batch (j - 1) t t'' hs m)
      · simp only [Option.some.injEq] at ht
        subst ht; rfl

/-- Push the operation at `i` down to sit immediately before position `j`, by
    adjacent exchanges.

    The opposite direction to `bubbleTo`, and not the same move: bringing a
    consumer up asks that it can pass everything between, while sending a
    producer down asks that everything between can pass *it*.  On this tape the
    second succeeds where the first does not, because a consumer usually reads
    something written in between and a producer usually writes only the one
    buffer nothing in between touches. -/
def bubbleDown (batch j : Nat) : Nat → Nat → List TOp → Option (List TOp)
  | 0,        _, t => some t
  | fuel + 1, i, t =>
      if i + 1 < j then
        match swapAt batch i t with
        | some t' => bubbleDown batch j fuel (i + 1) t'
        | none    => none
      else some t

/-- **…and it too changes nothing**, being a composition of exchanges. -/
theorem bubbleDown_den (batch j : Nat) : ∀ (fuel i : Nat) (t t' : List TOp),
    bubbleDown batch j fuel i t = some t' → ∀ (m : Buf → Nat → Float32),
    t'.foldl (fun mm o => o.den mm) m = t.foldl (fun mm o => o.den mm) m := by
  intro fuel
  induction fuel with
  | zero =>
      intro i t t' ht m
      simp only [bubbleDown, Option.some.injEq] at ht
      subst ht; rfl
  | succ fuel ih =>
      intro i t t' ht m
      simp only [bubbleDown] at ht
      split at ht
      · cases hs : swapAt batch i t with
        | none => rw [hs] at ht; exact absurd ht (by simp)
        | some t'' =>
            rw [hs] at ht
            exact (ih (i + 1) t'' t' ht m).trans (swapAt_den batch i t t'' hs m)
      · simp only [Option.some.injEq] at ht
        subst ht; rfl

/-- Move the operation that writes `tmp` down next to the one that reads it. -/
def sinkTo (batch : Nat) (tmp : Buf) (t : List TOp) : Option (List TOp) :=
  match (List.range t.length).find? (fun i =>
      match t[i]? with
      | some o => (TOp.outSize batch o).1 == tmp
      | none   => false) with
  | none   => none
  | some i =>
    match (List.range t.length).find? (fun j =>
        i < j && (match t[j]? with
                  | some o => (TOp.reads o).contains tmp
                  | none   => false)) with
    | none   => none
    | some j => bubbleDown batch j j i t

theorem sinkTo_den (batch : Nat) (tmp : Buf) (t t' : List TOp)
    (h : sinkTo batch tmp t = some t') (m : Buf → Nat → Float32) :
    t'.foldl (fun mm o => o.den mm) m = t.foldl (fun mm o => o.den mm) m := by
  unfold sinkTo at h
  split at h
  · exact absurd h (by simp)
  · split at h
    · exact absurd h (by simp)
    · exact bubbleDown_den batch _ _ _ t t' h m

/-- Sink each named intermediate's producer in turn. -/
def applySinks (batch : Nat) : List Buf → List TOp → List TOp
  | [],      t => t
  | b :: bs, t =>
      match sinkTo batch b t with
      | some t' => applySinks batch bs t'
      | none    => applySinks batch bs t

/-- **Sinking preserves the tape's mathematics on every buffer**, with no side
    condition, for any list of intermediates. -/
theorem applySinks_den (batch : Nat) : ∀ (bs : List Buf) (t : List TOp)
    (m : Buf → Nat → Float32),
    (applySinks batch bs t).foldl (fun mm o => o.den mm) m
      = t.foldl (fun mm o => o.den mm) m := by
  intro bs
  induction bs with
  | nil => intro t m; rfl
  | cons b bs ih =>
      intro t m
      cases hh : sinkTo batch b t with
      | none =>
          rw [show applySinks batch (b :: bs) t = applySinks batch bs t by
                simp only [applySinks, hh]]
          exact ih t m
      | some t'' =>
          rw [show applySinks batch (b :: bs) t = applySinks batch bs t'' by
                simp only [applySinks, hh]]
          exact (ih t'' m).trans (sinkTo_den batch b t t'' hh m)

/-- Move the operation that reads `tmp` up next to the one that writes it. -/
def hoistTo (batch : Nat) (tmp : Buf) (t : List TOp) : Option (List TOp) :=
  match (List.range t.length).find? (fun i =>
      match t[i]? with
      | some o => (TOp.outSize batch o).1 == tmp
      | none   => false) with
  | none   => none
  | some i =>
    match (List.range t.length).find? (fun j =>
        i < j && (match t[j]? with
                  | some o => (TOp.reads o).contains tmp
                  | none   => false)) with
    | none   => none
    | some j => bubbleTo batch i j j t

theorem hoistTo_den (batch : Nat) (tmp : Buf) (t t' : List TOp)
    (h : hoistTo batch tmp t = some t') (m : Buf → Nat → Float32) :
    t'.foldl (fun mm o => o.den mm) m = t.foldl (fun mm o => o.den mm) m := by
  unfold hoistTo at h
  split at h
  · exact absurd h (by simp)
  · split at h
    · exact absurd h (by simp)
    · exact bubbleTo_den batch _ _ _ t t' h m

/-- Hoist for each named intermediate in turn.  One that cannot be hoisted
    leaves the tape alone, exactly as a fusion site that does not apply does. -/
def applyHoists (batch : Nat) : List Buf → List TOp → List TOp
  | [],      t => t
  | b :: bs, t =>
      match hoistTo batch b t with
      | some t' => applyHoists batch bs t'
      | none    => applyHoists batch bs t

/-- **Hoisting preserves the tape's mathematics on every buffer**, for any list
    of intermediates.  Unlike fusion this carries no side condition at all. -/
theorem applyHoists_den (batch : Nat) : ∀ (bs : List Buf) (t : List TOp)
    (m : Buf → Nat → Float32),
    (applyHoists batch bs t).foldl (fun mm o => o.den mm) m
      = t.foldl (fun mm o => o.den mm) m := by
  intro bs
  induction bs with
  | nil => intro t m; rfl
  | cons b bs ih =>
      intro t m
      cases hh : hoistTo batch b t with
      | none =>
          rw [show applyHoists batch (b :: bs) t = applyHoists batch bs t by
                simp only [applyHoists, hh]]
          exact ih t m
      | some t'' =>
          rw [show applyHoists batch (b :: bs) t = applyHoists batch bs t'' by
                simp only [applyHoists, hh]]
          exact (ih t'' m).trans (hoistTo_den batch b t t'' hh m)

/-! ### Finding the sites, and naming them

    A schedule that has to count operations is a schedule written against the
    compiler's output rather than against the model.  Both of these exist so it
    does not have to: `fuseTargets` reports every site the guard accepts, and
    `FuseSite.killing` names one by the intermediate it removes — which is the
    same thing Halide's `compute_inline` is keyed on, the producer being
    inlined, and unlike a position it does not move when an earlier fusion
    shortens the tape. -/

/-- The sites with the buffer each one eliminates — what a schedule author
    reads to choose. -/
def fuseTargets (t : List TOp) : List (Nat × Buf) :=
  (List.range t.length).filterMap (fun i =>
    match fuseNormAt i t with
    | some (tmp, _) => some (i, tmp)
    | none          => none)

/-- **The fusable sites a tape offers, under the names the model bound.**

    What a schedule is written from: `fuse := [.named s]` for any `s` this
    reports is a site the guard accepts.  A target the model never named is
    absent, which is the honest answer — it can still be reached by the buffer
    it removes. -/
def fuseNames (tbl : List (String × Buf)) (t : List TOp) : List String :=
  (fuseTargets t).filterMap (fun p =>
    (tbl.find? (fun q => q.2 == p.2)).map Prod.fst)

/-- **Where a schedule says to fuse.** -/
inductive FuseSite where
  /-- By the intermediate it removes.  Stable: fusing elsewhere first does not
      renumber it. -/
  | killing : Buf → FuseSite
  /-- By position in the tape.  Fragile under earlier fusions, and kept because
      a generated schedule has already resolved its sites. -/
  | at      : Nat → FuseSite
  /-- **By the name the model bound.**  `tlet ss := …` labels its buffer, so a
      schedule says `.named "ss"` and never sees a number.  This is what Halide
      does — directives key on the algorithm's own handles — and it survives
      both a renumbering and an edit that moves the site. -/
  | named   : String → FuseSite
  deriving Repr, DecidableEq

/-- Where the fusion that removes `b` sits, if the guard accepts one. -/
def killPos (t : List TOp) (b : Buf) : Option Nat :=
  (List.range t.length).find? (fun i =>
    match fuseNormAt i t with
    | some (tmp, _) => tmp == b
    | none          => false)

def FuseSite.resolve (tbl : List (String × Buf)) (t : List TOp) : FuseSite → Option Nat
  | .at i      => some i
  | .killing b => killPos t b
  | .named s   => (tbl.lookup s).bind (killPos t)

/-- Resolve a named site and fuse there, in one step. -/
def fuseAt (tbl : List (String × Buf)) (t : List TOp) (s : FuseSite) :
    Option (Buf × List TOp) :=
  (s.resolve tbl t).bind (fun i => fuseNormAt i t)

/-- **Naming a site changes nothing about what fusing there does.**  The
    denotation theorem is `fuseNormAt`'s, reached through whichever position the
    name resolved to. -/
theorem fuseAt_den (tbl : List (String × Buf)) (t t' : List TOp) (s : FuseSite) (tmp : Buf)
    (h : fuseAt tbl t s = some (tmp, t'))
    (m : Buf → Nat → Float32) (b : Buf) (hb : b ≠ tmp) :
    (t'.foldl (fun mm o => o.den mm) m) b
      = (t.foldl (fun mm o => o.den mm) m) b := by
  simp only [fuseAt, Option.bind_eq_some_iff] at h
  obtain ⟨i, _, hi⟩ := h
  exact fuseNormAt_den i t t' tmp hi m b hb

end AlgorithmLib.ML
