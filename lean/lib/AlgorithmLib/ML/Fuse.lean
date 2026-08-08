import AlgorithmLib.ML.TenDenote

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
  | .mv _ w x _ _ _ _        => [w, x]
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
  · exact absurd h (by simp)

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
